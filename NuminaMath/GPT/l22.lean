import Mathlib

namespace part_a_part_b_l22_22023

-- Define the pupils and groups
def pupil : Type := ℕ
def group : Type := ℕ

-- Define the conditions
axiom groups_division : ∀ g : group, ∃ members : finset pupil, members.card = 11
axiom unique_intersection : ∀ g1 g2 : group, g1 ≠ g2 → ∃! p : pupil, p ∈ groups_division g1 ∧ p ∈ groups_division g2
axiom total_groups : finset group := (finset.range 112).map ⟨id, λ n _, n < 112⟩

-- Statement for Part (a)
theorem part_a : ∃ p : pupil, (finset.filter (λ g, p ∈ groups_division g) total_groups).card ≥ 12 :=
by sorry

-- Statement for Part (b)
theorem part_b : ∃ p : pupil, (finset.filter (λ g, p ∈ groups_division g) total_groups).card = 112 :=
by sorry

end part_a_part_b_l22_22023


namespace thabo_number_of_hardcover_nonfiction_books_l22_22035

variables (P_f H_f P_nf H_nf A : ℕ)

theorem thabo_number_of_hardcover_nonfiction_books
  (h1 : P_nf = H_nf + 15)
  (h2 : H_f = P_f + 10)
  (h3 : P_f = 3 * A)
  (h4 : A + H_f = 70)
  (h5 : P_f + H_f + P_nf + H_nf + A = 250) :
  H_nf = 30 :=
by {
  sorry
}

end thabo_number_of_hardcover_nonfiction_books_l22_22035


namespace area_of_region_l22_22637

noncomputable def area_between_line_curve : ℝ :=
∫ x in 0..3, (3 * x - x^2)

theorem area_of_region :
  area_between_line_curve = 9 / 2 := 
sorry

end area_of_region_l22_22637


namespace candy_eaten_l22_22771

theorem candy_eaten 
  {initial_pieces remaining_pieces eaten_pieces : ℕ} 
  (h₁ : initial_pieces = 12) 
  (h₂ : remaining_pieces = 3) 
  (h₃ : eaten_pieces = initial_pieces - remaining_pieces) 
  : eaten_pieces = 9 := 
by 
  sorry

end candy_eaten_l22_22771


namespace floor_sq_minus_sq_floor_l22_22425

noncomputable def floor_13_2 : ℤ := int.floor 13.2

theorem floor_sq_minus_sq_floor :
  int.floor ((13.2 : ℝ) ^ 2) - (floor_13_2 * floor_13_2) = 5 :=
by
  let floor_13_2_sq := floor_13_2 * floor_13_2
  have h1 : int.floor (13.2 : ℝ) = 13 := by norm_num
  have h2 : int.floor ((13.2 : ℝ) ^ 2) = 174 := by norm_num
  rw [h1] at floor_13_2
  rw [h2]
  rw [floor_13_2, floor_13_2, floor_13_2]
  exact sorry

end floor_sq_minus_sq_floor_l22_22425


namespace center_determines_position_l22_22708

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define what it means for the Circle's position being determined by its center.
theorem center_determines_position (c : Circle) : c.center = c.center :=
by
  sorry

end center_determines_position_l22_22708


namespace probability_at_least_one_boy_and_one_girl_l22_22037

noncomputable def mathematics_club_prob : ℚ :=
  let boys := 14
  let girls := 10
  let total_members := 24
  let total_committees := Nat.choose total_members 5
  let boys_committees := Nat.choose boys 5
  let girls_committees := Nat.choose girls 5
  let committees_with_at_least_one_boy_and_one_girl := total_committees - (boys_committees + girls_committees)
  let probability := (committees_with_at_least_one_boy_and_one_girl : ℚ) / (total_committees : ℚ)
  probability

theorem probability_at_least_one_boy_and_one_girl :
  mathematics_club_prob = (4025 : ℚ) / 4251 :=
by
  sorry

end probability_at_least_one_boy_and_one_girl_l22_22037


namespace regular_octagon_interior_angle_l22_22235

theorem regular_octagon_interior_angle : 
  ∀ n : ℕ, n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  intros n hn
  simp [sum_of_interior_angles]
  sorry

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
  180 * (n - 2)

end regular_octagon_interior_angle_l22_22235


namespace product_of_axes_l22_22604

noncomputable def focus_distance : ℝ := 8
noncomputable def incircle_diameter : ℝ := 6

def ellipse_area_product : ℝ :=
let a := 17 in -- Given from solution, a = OA = OB 
let b := 15 in -- Given from solution, b = OC = OD 
(2 * a) * (2 * b)

theorem product_of_axes :
  ellipse_area_product = 1020 :=
by {
  -- The detailed proof steps are omitted as specified
  sorry
}

end product_of_axes_l22_22604


namespace regular_octagon_interior_angle_measure_l22_22254

theorem regular_octagon_interior_angle_measure :
  ∀ (n : ℕ), n = 8 → (∑ i in finset.range n, (180 : ℝ)) / n = 135 :=
  by
  sorry

end regular_octagon_interior_angle_measure_l22_22254


namespace reb_leave_earlier_l22_22286

variable (D_drive : ℝ := 40 * 0.75) -- distance driven, in miles
variable (reduction_percentage : ℝ := 0.20) -- reduction in driving distance percentage
variable (speed_bike_slowest : ℝ := 12) -- slowest biking speed in miles per hour
variable (time_drive : ℝ := 45) -- driving time in minutes

def bike_distance := D_drive * (1 - reduction_percentage) -- bike distance in miles

def time_bike := (bike_distance / speed_bike_slowest) * 60 -- biking time in minutes

theorem reb_leave_earlier :
  time_bike - time_drive = 75 :=
by
  -- To be proven
  sorry

end reb_leave_earlier_l22_22286


namespace minimum_cubes_required_l22_22398

def cube : Type :=
  { snap : bool // snap = true }

structure CubeFormation :=
  (cubes : List cube)
  (hides_all_snaps : ∀ c ∈ cubes, ¬c.snap)
  (perfectly_aligned : ∀ c1 c2 ∈ cubes, c1 ≠ c2 → aligned_perfectly c1 c2)

noncomputable def minimum_number_of_cubes : Nat :=
  4

theorem minimum_cubes_required
  (conditions_holds : ∀ c, c ∈ CubeFormation.cubes → c.snap = true → c ∉ CubeFormation.cubes)
  : CubeFormation.cubes.length = minimum_number_of_cubes :=
sorry

end minimum_cubes_required_l22_22398


namespace angle_DAB_22_5_l22_22964

-- Define a right-angled triangle at B, with AC = CD and BC = 2AB.
variables (A B C D : Type) [MetricSpace (A B C D)] [HasMem ℝ (Set (A B C D))]
variables [Triangle BAD] [IsRightAngled B] [IsOnSegment A D C]
#check MetricSpace -- this is required to fix the broader import and generalize points.

-- Definitions relevant to the problem setup.
def AngleDAB (A B C D : Type) [MetricSpace (A B C D)] [IsRightAngled B] 
  (C_on_AD : IsOnSegment A D C) (AC_eq_CD : AC = CD) (BC_eq_2AB : BC = 2 * AB) : Angle DAB := sorry

-- Proof that under these conditions, the angle $\angle DAB$ is $22.5^\circ$.
theorem angle_DAB_22_5 (A B C D : Type) [MetricSpace (A B C D)] [Triangle B A D] [IsRightAngled B]
  (C_on_AD : IsOnSegment A D C) (AC_eq_CD : AC = CD) (BC_eq_2AB : BC = 2 * AB) : 
  Angle DAB = 22.5 :=
sorry

end angle_DAB_22_5_l22_22964


namespace length_AB_when_alpha_zero_range_PA_PB_squared_l22_22491

-- Definitions and Conditions
def polar_curve (θ : ℝ) : ℝ := 2 * real.sqrt 2 * real.sin (θ - real.pi / 4)

def parametric_line (t α : ℝ) : ℝ × ℝ := 
  (1 + t * real.cos α, 2 + t * real.sin α)

def P : ℝ × ℝ := (1, 2)

-- Lemmas to prove the statements
theorem length_AB_when_alpha_zero : 
  (let α := 0 in 
    let intersection_points := 
      {t | parametric_line t α.1 + 1 = real.sqrt 2 ∨ parametric_line t α.1 - 1 = - real.sqrt 2} in
    real.abs (parametric_line (0 : ℝ) α.fst - parametric_line (-2 : ℝ) α.fst) = 2) := 
sorry

theorem range_PA_PB_squared : 
  (let α_range := {α | 0 ≤ α ∧ α < real.pi} in
    ∀ α ∈ α_range,
    let t1_t2 := {t | t ^ 2 + (4 * real.cos α + 2 * real.sin α) * t + 3 = 0} in
    (let t1 := some (t1_t2), 
         t2 := some (t1_t2) in
     let PA_PB_squared := (1 + t1 * real.cos α - P.1) ^ 2 + (2 + t1 * real.sin α - P.2) ^ 2 +
                          (1 + t2 * real.cos α - P.1) ^ 2 + (2 + t2 * real.sin α - P.2) ^ 2 in
    6 < PA_PB_squared ∧ PA_PB_squared ≤ 14)) := 
sorry

end length_AB_when_alpha_zero_range_PA_PB_squared_l22_22491


namespace max_changes_in_direction_l22_22077

theorem max_changes_in_direction (n m : ℕ) (h_n : n = 2004) (h_m : m = 2004) :
  ∃ R : list (ℕ × ℕ), (∀ (u v : (ℕ × ℕ)), u ≠ v → R u ≠ R v) ∧ max_changes R = n * (m + 1) - 1 := 
sorry

end max_changes_in_direction_l22_22077


namespace tan20_plus_4sin20_l22_22761

noncomputable def problem_statement : Prop :=
  tan (20 * Real.pi / 180) + 4 * sin (20 * Real.pi / 180) = Real.sqrt 3

theorem tan20_plus_4sin20 :
  problem_statement :=
by
  sorry

end tan20_plus_4sin20_l22_22761


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22140

theorem measure_of_one_interior_angle_of_regular_octagon : 
  let n := 8 in 
  let sum_of_interior_angles := 180 * (n - 2) in
  let measure_of_one_interior_angle := sum_of_interior_angles / n in 
  measure_of_one_interior_angle = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22140


namespace max_buses_l22_22925

-- Define the total number of stops and buses as finite sets
def stops : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define a bus as a subset of stops with exactly 3 stops
structure Bus :=
(stops : Finset ℕ)
(h_stops : stops.card = 3)

-- Define the condition that any two buses share at most one stop
def shares_at_most_one (b1 b2 : Bus) : Prop :=
(b1.stops ∩ b2.stops).card ≤ 1

-- Define the predicate for a valid set of buses
def valid_buses (buses : Finset Bus) : Prop :=
∀ b1 b2 ∈ buses, b1 ≠ b2 → shares_at_most_one b1 b2

-- The main theorem statement
theorem max_buses (buses : Finset Bus) (h_valid : valid_buses buses) : buses.card ≤ 12 :=
sorry

end max_buses_l22_22925


namespace umbrella_arrangements_l22_22670

theorem umbrella_arrangements : ∃ (poster_count : ℕ), poster_count = 20 :=
by
  -- Definitions based on conditions
  let actors := 7
  let tallest := 1
  let remaining := actors - tallest -- which is 6
  let groups := 2 -- two groups of 3 actors each
  let arrangements := Nat.choose remaining (remaining / groups) * 2! -- combinations x arrangements factor

  -- Prove the transformation
  have h: arrangements = 20 := sorry
  use 20
  exact h

end umbrella_arrangements_l22_22670


namespace max_buses_in_city_l22_22934

/--
In a city with 9 bus stops where each bus serves exactly 3 stops and any two buses share at most one stop, 
prove that the maximum number of buses that can be in the city is 12.
-/
theorem max_buses_in_city :
  ∃ (B : Set (Finset ℕ)), 
    (∀ b ∈ B, b.card = 3) ∧ 
    (∀ b1 b2 ∈ B, b1 ≠ b2 → (b1 ∩ b2).card ≤ 1) ∧ 
    (∀ b ∈ B, ∀ s ∈ b, s < 9) ∧ 
    B.card = 12 := sorry

end max_buses_in_city_l22_22934


namespace interior_angle_regular_octagon_l22_22081

theorem interior_angle_regular_octagon (n : ℕ) (h : n = 8) : 
  let sum_of_interior_angles := 180 * (n - 2)
  let number_of_sides := n
  let one_interior_angle := sum_of_interior_angles / number_of_sides
  in one_interior_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l22_22081


namespace regular_octagon_interior_angle_l22_22245

theorem regular_octagon_interior_angle : 
  ∀ n : ℕ, n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  intros n hn
  simp [sum_of_interior_angles]
  sorry

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
  180 * (n - 2)

end regular_octagon_interior_angle_l22_22245


namespace tan70_cos10_sqrt3_tan20_minus_1_l22_22660

theorem tan70_cos10_sqrt3_tan20_minus_1 :
  ∀ (tan cos sin : ℝ → ℝ),
  (∀ x, tan x = sin x / cos x) →
  sin 70 = cos 20 →
  tan 20 = sin 20 / cos 20 →
  sin 20 = 2 * sin 10 * cos 10 →
  (∀ a b, sin (a - b) = sin a * cos b - cos a * sin b) →
  sin (-10) = - sin 10 →
  tan 70 * cos 10 * (sqrt 3 * tan 20 - 1) = -1 := 
by { 
  intros tan cos sin h1 h2 h3 h4 h5 h6,
  sorry 
}

end tan70_cos10_sqrt3_tan20_minus_1_l22_22660


namespace triangle_angle_property_l22_22576

-- Define the given data for the problem
variables {A B C : ℝ} {h_a t_a θ : ℝ}
variables (b c : ℝ)

-- Define the conditions
def triangle (a b c : ℝ) := a + b > c ∧ b + c > a ∧ c + a > b
def altitude (A : ℝ) (h_a : ℝ) : Prop := true  -- A stub definition for altitude
def angle_bisector (A : ℝ) (t_a : ℝ) : Prop := true  -- A stub definition for angle bisector
def angle_between (X Y θ : ℝ) : Prop := true  -- A stub definition for angle between two lines

-- Translate the problem statement
theorem triangle_angle_property
  (H₁ : triangle A B C)
  (H₂ : altitude A h_a)
  (H₃ : angle_bisector A t_a)
  (H₄ : angle_between h_a t_a θ) :
  tan θ = (|c - b| / (c + b)) * cot (A / 2) :=
sorry

end triangle_angle_property_l22_22576


namespace speed_in_still_water_l22_22283

theorem speed_in_still_water (u d s : ℝ) (hu : u = 20) (hd : d = 60) (hs : s = (u + d) / 2) : s = 40 := 
by 
  sorry

end speed_in_still_water_l22_22283


namespace simplify_complex_expression_l22_22626

open Complex

theorem simplify_complex_expression : (5 - 3 * Complex.i)^2 = 16 - 30 * Complex.i :=
by
  sorry

end simplify_complex_expression_l22_22626


namespace oil_bill_january_l22_22653

-- Declare the constants for January and February oil bills
variables (J F : ℝ)

-- State the conditions
def condition_1 : Prop := F / J = 3 / 2
def condition_2 : Prop := (F + 20) / J = 5 / 3

-- State the theorem based on the conditions and the target statement
theorem oil_bill_january (h1 : condition_1 F J) (h2 : condition_2 F J) : J = 120 :=
by
  sorry

end oil_bill_january_l22_22653


namespace largest_prime_factor_of_sum_of_divisors_of_180_eq_13_l22_22981

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, d ∣ n).sum

def largest_prime_factor (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ p, Nat.Prime p ∧ p ∣ n).max' sorry

theorem largest_prime_factor_of_sum_of_divisors_of_180_eq_13 :
  largest_prime_factor (sum_of_divisors 180) = 13 :=
by
  sorry

end largest_prime_factor_of_sum_of_divisors_of_180_eq_13_l22_22981


namespace interior_angle_regular_octagon_l22_22085

theorem interior_angle_regular_octagon (n : ℕ) (h : n = 8) : 
  let sum_of_interior_angles := 180 * (n - 2)
  let number_of_sides := n
  let one_interior_angle := sum_of_interior_angles / number_of_sides
  in one_interior_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l22_22085


namespace find_value_of_x8_plus_x4_plus_1_l22_22853

theorem find_value_of_x8_plus_x4_plus_1 (x : ℂ) (hx : x^2 + x + 1 = 0) : x^8 + x^4 + 1 = 0 :=
sorry

end find_value_of_x8_plus_x4_plus_1_l22_22853


namespace problem_statement_l22_22634

noncomputable def P : ℝ → ℝ → ℝ → ℝ := sorry -- Define P which satisfies all the given conditions

theorem problem_statement : 
  ∃ (P : ℝ → ℝ → ℝ → ℝ), 
  (∀ (k : ℝ) (a b c : ℝ), P (k*a) (k*b) (k*c) = k^4 * P a b c) ∧  
  (∀ a b c : ℝ, P a b c = P b c a) ∧ 
  (∀ a b : ℝ, P a a b = 0) ∧ 
  P 1 2 3 = 1 ∧ 
  P 2 4 8 = 56 := by {
  use P,
  -- Homogeneous of degree 4 condition
  split, 
  { intros k a b c, sorry },
  split, 
  { intros a b c, sorry },
  split,
  { intros a b, sorry },
  split,
  { exact sorry },
  { exact sorry }
}

end problem_statement_l22_22634


namespace sec_225_eq_neg_sqrt_2_l22_22453

theorem sec_225_eq_neg_sqrt_2 :
  let sec (x : ℝ) := 1 / Real.cos x
  ∧ Real.cos (225 * Real.pi / 180) = Real.cos (180 * Real.pi / 180 + 45 * Real.pi / 180)
  ∧ Real.cos (180 * Real.pi / 180 + 45 * Real.pi / 180) = -Real.cos (45 * Real.pi / 180)
  ∧ Real.cos (45 * Real.pi / 180) = 1 / Real.sqrt 2
  → sec (225 * Real.pi / 180) = -Real.sqrt 2 :=
by 
  intros sec_h cos_h1 cos_h2 cos_h3
  sorry

end sec_225_eq_neg_sqrt_2_l22_22453


namespace regular_octagon_interior_angle_measure_l22_22258

theorem regular_octagon_interior_angle_measure :
  ∀ (n : ℕ), n = 8 → (∑ i in finset.range n, (180 : ℝ)) / n = 135 :=
  by
  sorry

end regular_octagon_interior_angle_measure_l22_22258


namespace probability_both_selected_l22_22672

theorem probability_both_selected (P_R : ℚ) (P_V : ℚ) (h1 : P_R = 3 / 7) (h2 : P_V = 1 / 5) :
  P_R * P_V = 3 / 35 :=
by {
  sorry
}

end probability_both_selected_l22_22672


namespace amaya_total_marks_l22_22744

theorem amaya_total_marks 
  (m_a s_a a m m_s : ℕ) 
  (h_music : m_a = 70)
  (h_social_studies : s_a = m_a + 10)
  (h_maths_art_diff : m = a - 20)
  (h_maths_fraction : m = a - 1/10 * a)
  (h_maths_eq_fraction : m = 9/10 * a)
  (h_arts : 9/10 * a = a - 20)
  (h_total : m_a + s_a + a + m = 530) :
  m_a + s_a + a + m = 530 :=
by
  -- Proof to be completed
  sorry

end amaya_total_marks_l22_22744


namespace sin4_minus_cos4_eq_neg_3_div_5_l22_22849

noncomputable def alpha := Real.arcsin (Real.sqrt 5 / 5)

theorem sin4_minus_cos4_eq_neg_3_div_5 : 
  sin^4(alpha) - cos^4(alpha) = -3 / 5 := by
  sorry

end sin4_minus_cos4_eq_neg_3_div_5_l22_22849


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22161

theorem measure_of_one_interior_angle_of_regular_octagon :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 := 
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22161


namespace regular_octagon_interior_angle_l22_22224

theorem regular_octagon_interior_angle:
  let n := 8
  let sum_interior_angles (n: ℕ) := 180 * (n - 2)
  let interior_angle (n: ℕ) := sum_interior_angles n / n
  interior_angle n = 135 :=
by
  sorry

end regular_octagon_interior_angle_l22_22224


namespace cost_of_soda_l22_22069

-- Define the system of equations
theorem cost_of_soda (b s f : ℕ): 
  3 * b + s = 390 ∧ 
  2 * b + 3 * s = 440 ∧ 
  b + 2 * f = 230 ∧ 
  s + 3 * f = 270 → 
  s = 234 := 
by 
  sorry

end cost_of_soda_l22_22069


namespace sum_S_2019_l22_22049

def sequence (n : ℕ) : ℤ := n * int.cos ((n : ℤ) * int.pi / 2)

def S (n : ℕ) : ℤ := (finset.range n).sum (λ k, sequence (k + 1))

theorem sum_S_2019 : S 2019 = -1010 :=
by { sorry }

end sum_S_2019_l22_22049


namespace interior_angle_regular_octagon_l22_22271

theorem interior_angle_regular_octagon : 
  let n := 8 in 
  ∀ (sum_of_interior_angles : ℕ → ℕ) (regular_polygon_interior_angle : ℕ → ℕ),
  sum_of_interior_angles n = (180 * (n - 2)) →
  regular_polygon_interior_angle n = (sum_of_interior_angles n / n) →
  regular_polygon_interior_angle n = 135 :=
by
  intros n sum_of_interior_angles regular_polygon_interior_angle h1 h2
  sorry

end interior_angle_regular_octagon_l22_22271


namespace regular_octagon_interior_angle_measure_l22_22259

theorem regular_octagon_interior_angle_measure :
  ∀ (n : ℕ), n = 8 → (∑ i in finset.range n, (180 : ℝ)) / n = 135 :=
  by
  sorry

end regular_octagon_interior_angle_measure_l22_22259


namespace inclination_angles_ordered_l22_22557

open Real

-- Definitions of the lines
def l1 : ℝ × ℝ → Prop := λ p, p.1 - p.2 = 0
def l2 : ℝ × ℝ → Prop := λ p, p.1 + 2 * p.2 = 0
def l3 : ℝ × ℝ → Prop := λ p, p.1 + 3 * p.2 = 0

-- Inclination angles of the lines
def inclination_angle (l : ℝ × ℝ → Prop) : ℝ :=
  if h : l = l1 then arctan (1)
  else if l = l2 then arctan (-1/2)
  else arctan (-1/3)

def alpha1 := inclination_angle l1
def alpha2 := inclination_angle l2
def alpha3 := inclination_angle l3

-- The Lean statement to prove the ordering of the angles
theorem inclination_angles_ordered :
  alpha1 < alpha2 ∧ alpha2 < alpha3 := by
  sorry

end inclination_angles_ordered_l22_22557


namespace regular_octagon_angle_measure_l22_22217

theorem regular_octagon_angle_measure :
  let n := 8 in
  (180 * (n - 2)) / n = 135 := 
by
  sorry

end regular_octagon_angle_measure_l22_22217


namespace fraction_addition_l22_22694

theorem fraction_addition (d : ℤ) :
  (6 + 4 * d) / 9 + 3 / 2 = (39 + 8 * d) / 18 := sorry

end fraction_addition_l22_22694


namespace max_buses_in_city_l22_22942

theorem max_buses_in_city (n : ℕ) (stops : ℕ) (shared : ℕ) (condition1 : n = 9) (condition2 : stops = 3) (condition3 : shared ≤ 1) : n = 12 :=
sorry

end max_buses_in_city_l22_22942


namespace regular_octagon_angle_measure_l22_22204

theorem regular_octagon_angle_measure :
  let n := 8 in
  (180 * (n - 2)) / n = 135 := 
by
  sorry

end regular_octagon_angle_measure_l22_22204


namespace nested_radical_expr_eq_3_l22_22347

theorem nested_radical_expr_eq_3 :
  sqrt (1 + 2 * sqrt (1 + 3 * sqrt (1 + ... + 2017 * sqrt (1 + 2018 * 2020)))) = 3 :=
sorry

end nested_radical_expr_eq_3_l22_22347


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22170

-- Define the conditions for the problem
def num_sides : ℕ := 8
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Statement of the problem
theorem measure_of_one_interior_angle_of_regular_octagon : 
  (sum_of_interior_angles num_sides) / num_sides = 135 :=
by
  -- Conditions are defined above
  -- Proof steps are omitted
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22170


namespace max_value_expression_l22_22802

open Real

theorem max_value_expression (x y z : ℝ) : 
  (sin (3 * x) + sin (2 * y) + sin z) * (cos (3 * x) + cos (2 * y) + cos z) ≤ 9 / 2 :=
begin
  sorry,  -- Placeholder for the actual proof
end

end max_value_expression_l22_22802


namespace remainder_of_f_div_x_minus_2_is_48_l22_22275

-- Define the polynomial f(x)
noncomputable def f (x : ℝ) : ℝ := x^5 - 5 * x^4 + 8 * x^3 + 25 * x^2 - 14 * x - 40

-- State the theorem to prove that the remainder of f(x) when divided by x - 2 is 48
theorem remainder_of_f_div_x_minus_2_is_48 : f 2 = 48 :=
by sorry

end remainder_of_f_div_x_minus_2_is_48_l22_22275


namespace ratio_dark_blue_to_total_l22_22843

-- Definitions based on the conditions
def total_marbles := 63
def red_marbles := 38
def green_marbles := 4
def dark_blue_marbles := total_marbles - red_marbles - green_marbles

-- The statement to be proven
theorem ratio_dark_blue_to_total : (dark_blue_marbles : ℚ) / total_marbles = 1 / 3 := by
  sorry

end ratio_dark_blue_to_total_l22_22843


namespace floor_abs_of_neg_num_l22_22403

theorem floor_abs_of_neg_num : ((Real.floor (| -57.6 |)) = 57) := 
by
  sorry

end floor_abs_of_neg_num_l22_22403


namespace shaded_rectangle_ratio_l22_22326

variable (a : ℝ) (h : 0 < a)  -- side length of the square is 'a' and it is positive

theorem shaded_rectangle_ratio :
  (∃ l w : ℝ, (l = a / 2 ∧ w = a / 3 ∧ (l * w = a^2 / 6) ∧ (a^2 / 6 = a * a / 6))) → (l / w = 1.5) :=
by {
  -- Proof is to be provided
  sorry
}

end shaded_rectangle_ratio_l22_22326


namespace count_negative_terms_in_sequence_l22_22520

theorem count_negative_terms_in_sequence : 
  ∃ (s : List ℕ), (∀ n ∈ s, n^2 - 8*n + 12 < 0) ∧ s.length = 3 ∧ (∀ n ∈ s, 2 < n ∧ n < 6) :=
by
  sorry

end count_negative_terms_in_sequence_l22_22520


namespace nested_radical_value_l22_22358

theorem nested_radical_value :
  (nat.sqrt (1 + 2 * nat.sqrt (1 + 3 * nat.sqrt (1 + ... + 2017 * nat.sqrt (1 + 2018 * 2020))))) = 3 :=
sorry

end nested_radical_value_l22_22358


namespace exist_fixed_distance_constants_l22_22525

variables {V : Type*} [inner_product_space ℝ V]

open_locale real_inner_product_space

def fixed_distance_from_linear_combination
  (a b : V) (p : V) (s v : ℝ) : Prop :=
  ∃ (dist : ℝ), ∀ p, 
  (\<norm>p - (s • a + v • b)\<norm> = dist)

-- Given vector properties and constraints
variables (a b : V)
axiom h : ∀ p : V, ∥p - a∥ = 3 * ∥p - b∥

-- Main theorem statement
theorem exist_fixed_distance_constants : 
  ∃ (s v : ℝ), 
  fixed_distance_from_linear_combination a b p s v :=
begin
  use (9 / 32, -9 / 16),
  sorry
end

end exist_fixed_distance_constants_l22_22525


namespace max_value_sine_cosine_expression_l22_22799

theorem max_value_sine_cosine_expression :
  ∀ x y z : ℝ, 
    (sin (3 * x) + sin (2 * y) + sin z) * (cos (3 * x) + cos (2 * y) + cos z) ≤ 4.5 :=
by
  intros x y z
  sorry

end max_value_sine_cosine_expression_l22_22799


namespace count_ordered_pairs_l22_22395

open Finset

-- Define ℕ set 1 to 10
def U : Finset ℕ := (range 11).filter (λ x, x > 0)

-- Define the conditions
def conditions (A B : Finset ℕ) : Prop :=
  A ∪ B = U ∧
  A ∩ B = ∅ ∧
  ¬ A.card ∈ A ∧
  A.card % 2 = 0 ∧
  ¬ B.card ∈ B

-- Define the function to count valid sets
noncomputable def count_valid_sets : ℕ :=
  ∑ n in U.filter (λ n, n % 2 = 0 ∧ (n ≠ 0 ∧ n ≠ 10)), 
    (choose (10 - 1 - n) n) + (choose (10 - 1 - 10 + n) (10 - n))

-- The final theorem
theorem count_ordered_pairs : count_valid_sets = 43 := by sorry

end count_ordered_pairs_l22_22395


namespace area_condition_l22_22072

-- We define the conditions
variable (F : ℝ) (S : ℝ) (P : ℝ)

-- Given specific values for F and constraint on P
def force := 50 -- in Newtons
def pressure_greater_than := 500 -- in Pascal

-- Condition: P = F / S
def pressure := F / S

-- Lean statement to prove the problem
theorem area_condition (S : ℝ) (h1 : F = force) (h2 : P > pressure_greater_than) (h3 : P = F / S) : S < 0.1 :=
by
  sorry

end area_condition_l22_22072


namespace exponential_inequality_l22_22540

theorem exponential_inequality (m n : ℝ) (hmn : m > n) (hn : n > 0) : 0.3 ^ m < 0.3 ^ n :=
sorry

end exponential_inequality_l22_22540


namespace f_functional_eq_f_neg_one_l22_22998

-- Define the function and conditions
def f : ℝ → ℝ := λ x, -x - 1

-- Establish the problem in Lean
theorem f_functional_eq (x y : ℝ) : f (f (x) + y) = f (x + y) + x * f (f (y)) - 2 * x * y + x - 1 :=
by
  -- Introduce the function and its properties
  let f : ℝ → ℝ := λ x, -x - 1
  sorry

-- Determine the value of f(-1)
theorem f_neg_one : f (-1) = 1 :=
by
  let f : ℝ → ℝ := λ x, -x - 1
  show f (-1) = 1
  sorry

end f_functional_eq_f_neg_one_l22_22998


namespace player_A_wins_3_to_1_l22_22603

theorem player_A_wins_3_to_1 (p : ℚ) (C : ℕ → ℕ → ℚ) :
  (p = 2/3) →
  (C 3 1 = 3) →
  (3 * (p ^ 3) * (1 - p) = 8/27) :=
by
  intros h_p h_C
  rw h_p at *
  linarith [h_C]

end player_A_wins_3_to_1_l22_22603


namespace find_PR_in_triangle_l22_22965

noncomputable def find_length_PR (PQ : ℝ) (tan_R : ℝ) : ℝ :=
  let PR := tan_R * PQ in
  PR

theorem find_PR_in_triangle (PQ PR : ℝ) (tan_R : ℝ) : PR = 4 :=
  assume h1: PQ = 3,
  assume h2: tan_R = 4/3,
  have h3: PR = tan_R * PQ, from rfl,
  calc
    PR = tan_R * PQ : by rw h3
    ... = (4/3) * 3 : by rw [h2, h1]
    ... = 4 : by norm_num

end find_PR_in_triangle_l22_22965


namespace largest_prime_factor_sum_of_divisors_180_l22_22989

def sum_of_divisors (n : ℕ) : ℕ :=
  ∏ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d

theorem largest_prime_factor_sum_of_divisors_180 :
  let M := sum_of_divisors 180 in
  ∀ p, nat.prime p → p ∣ M → p ≤ 13 :=
by
  let M := sum_of_divisors 180
  have p_factors : multiset ℕ := (unique_factorization_monoid.factorization M).to_multiset
  exact sorry

end largest_prime_factor_sum_of_divisors_180_l22_22989


namespace find_inverse_squared_value_l22_22539

theorem find_inverse_squared_value :
  (let f (x : ℝ) := 24 / (7 + 4 * x) in
  (f⁻¹ 3)⁻² = 16) :=
by
  let f (x : ℝ) := 24 / (7 + 4 * x)
  sorry

end find_inverse_squared_value_l22_22539


namespace interior_angle_regular_octagon_l22_22270

theorem interior_angle_regular_octagon : 
  let n := 8 in 
  ∀ (sum_of_interior_angles : ℕ → ℕ) (regular_polygon_interior_angle : ℕ → ℕ),
  sum_of_interior_angles n = (180 * (n - 2)) →
  regular_polygon_interior_angle n = (sum_of_interior_angles n / n) →
  regular_polygon_interior_angle n = 135 :=
by
  intros n sum_of_interior_angles regular_polygon_interior_angle h1 h2
  sorry

end interior_angle_regular_octagon_l22_22270


namespace max_buses_in_city_l22_22938

-- Definition of the problem statement and the maximal number of buses
theorem max_buses_in_city (bus_stops : Finset ℕ) (buses : Finset (Finset ℕ)) 
  (h_stops : bus_stops.card = 9) 
  (h_buses_stops : ∀ b ∈ buses, b.card = 3) 
  (h_shared_stops : ∀ b₁ b₂ ∈ buses, b₁ ≠ b₂ → (b₁ ∩ b₂).card ≤ 1) : 
  buses.card ≤ 12 := 
sorry

end max_buses_in_city_l22_22938


namespace simplify_complex_square_l22_22622

def complex_sq_simplification (a b : ℕ) (i : ℂ) (h : i^2 = -1) : ℂ :=
  (a - b * i)^2

theorem simplify_complex_square : complex_sq_simplification 5 3 complex.I (complex.I_sq) = 16 - 30 * complex.I :=
by sorry

end simplify_complex_square_l22_22622


namespace find_b_over_a_l22_22279

variable (a b x y z : ℝ)
variable (a_pos : a > 0)
variable (hx : z = ln y)
variable (hy : y = a * exp (b * x + 1))
variable (hz_hat : z = 2 * x + a)

theorem find_b_over_a (a_pos : a > 0) (hx : z = ln y) (hy : y = a * exp (b * x + 1)) (hz_hat : z = 2 * x + a) : 
  b / a = 2 := sorry

end find_b_over_a_l22_22279


namespace sum_binomial_coeff_exp_l22_22758

theorem sum_binomial_coeff_exp (n : ℕ) : 
  ∑ k in Finset.range (n + 1), (2:ℕ)^k * Nat.choose n k = 3^n :=
by
  sorry

end sum_binomial_coeff_exp_l22_22758


namespace nested_radical_value_l22_22357

theorem nested_radical_value :
  (nat.sqrt (1 + 2 * nat.sqrt (1 + 3 * nat.sqrt (1 + ... + 2017 * nat.sqrt (1 + 2018 * 2020))))) = 3 :=
sorry

end nested_radical_value_l22_22357


namespace ratio_of_c_to_a_l22_22480

theorem ratio_of_c_to_a (a b c : ℝ) (h : a ≠ 0) (five_points : set (set ℝ)) (h1 : five_points.card = 5) 
  (h2 : five_points.pairwise_disjoint) 
  (h3 : ∃ segments : multiset ℝ, 
          segments = {a, a, a, a, a, b, b, 2 * a, c}) :
  c / a = 2 * real.sqrt 3 := 
sorry

end ratio_of_c_to_a_l22_22480


namespace problem_1_problem_2_l22_22020

theorem problem_1 (P_A P_B P_notA P_notB : ℚ) (hA: P_A = 1/2) (hB: P_B = 2/5) (hNotA: P_notA = 1/2) (hNotB: P_notB = 3/5) : 
  P_A * P_notB + P_B * P_notA = 1/2 := 
by 
  rw [hA, hB, hNotA, hNotB]
  -- exact calculations here
  sorry

theorem problem_2 (P_A P_B : ℚ) (hA: P_A = 1/2) (hB: P_B = 2/5) :
  (1 - (P_A * P_A * (1 - P_B) * (1 - P_B))) = 91/100 := 
by 
  rw [hA, hB]
  -- exact calculations here
  sorry

end problem_1_problem_2_l22_22020


namespace area_of_convex_quadrilateral_l22_22385

theorem area_of_convex_quadrilateral
  (PQ QR RS PS : ℝ)
  (hPQ : PQ = 5)
  (hQR : QR = 12)
  (hRS : RS = 13)
  (hPS : PS = 13)
  (anglePQR : ℝ)
  (hAnglePQR : anglePQR = 90) :
  let PR := Real.sqrt (PQ * PQ + QR * QR)
  in PQ * QR * 1 / 2 + PR * PS * 1 / 2  = 114.5 :=
by {
  have hPR : PR = 13 := by sorry,
  have hAreaPQR : PQ * QR * 1 / 2 = 30 := by sorry,
  have hAreaPRS : PR * PS * 1 / 2 = 84.5 := by sorry,
  rw [hAreaPQR, hAreaPRS],
  exact (30:ℝ) + (84.5:ℝ),
  sorry
}

end area_of_convex_quadrilateral_l22_22385


namespace range_of_k_l22_22892

-- Lean statement for the equivalent math proof problem based on given conditions and answer.
theorem range_of_k (k : ℝ) 
  (h₁ : k > 0) 
  (h₂ : ∃ A B : ℝ × ℝ, 
           A ≠ B ∧ 
           A.1 + A.2 - k = 0 ∧ A.1 ^ 2 + A.2 ^ 2 = 4 ∧ 
           B.1 + B.2 - k = 0 ∧ B.1 ^ 2 + B.2 ^ 2 = 4) 
  (h₃ : |((0, 0) : ℝ × ℝ).1 + A.1 + ((0, 0) : ℝ × ℝ).2 + A.2 - k| ≥ sqrt 3 / 3 * sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)) :
  (sqrt 2 ≤ k ∧ k < 2 * sqrt 2) :=
sorry

end range_of_k_l22_22892


namespace unique_flavors_count_l22_22400

-- Let red_flavors represent the number of red candy
-- Let green_flavors represent the number of green candy
-- Given conditions:
def redCandies : ℕ := 8
def greenCandies : ℕ := 5
def totalCandies : ℕ := redCandies + greenCandies

-- The object flavor is based on the percentage representation of the red candies.
def isUniqueFlavor (x y : ℕ) : Prop := ∃ s t : ℕ, x * t = y * s

-- theorem to prove the number of unique flavors possible given the conditions
theorem unique_flavors_count : 
  ∃ unique_flavors : ℕ, unique_flavors = 39 ∧
  ∀ (x y : ℕ), x ≤ redCandies → y ≤ greenCandies → isUniqueFlavor x y ↔ unique_flavors = 39 :=
begin
  -- Proof steps here
  sorry
end

end unique_flavors_count_l22_22400


namespace least_lucky_multiple_of_six_not_lucky_l22_22313

def digit_sum (n : ℕ) : ℕ :=
  (n.to_digits 10).sum

def is_lucky_integer (n : ℕ) : Prop :=
  n % digit_sum n = 0

def is_multiple_of_six (n : ℕ) : Prop :=
  n % 6 = 0

theorem least_lucky_multiple_of_six_not_lucky : ∃ n, is_multiple_of_six n ∧ ¬ is_lucky_integer n ∧ (∀ m, is_multiple_of_six m ∧ ¬ is_lucky_integer m → n ≤ m) :=
by {
  use 12,
  split,
  { sorry },  -- Proof that 12 is a multiple of 6
  split,
  { sorry },  -- Proof that 12 is not a lucky integer
  { sorry },  -- Proof that there is no smaller multiple of 6 that is not a lucky integer
}

end least_lucky_multiple_of_six_not_lucky_l22_22313


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22144

theorem measure_of_one_interior_angle_of_regular_octagon : 
  let n := 8 in 
  let sum_of_interior_angles := 180 * (n - 2) in
  let measure_of_one_interior_angle := sum_of_interior_angles / n in 
  measure_of_one_interior_angle = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22144


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22195

theorem measure_of_one_interior_angle_of_regular_octagon 
  (n : Nat) (h_n : n = 8) : 
  (one_interior_angle : ℝ) = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22195


namespace interior_angle_regular_octagon_l22_22267

theorem interior_angle_regular_octagon : 
  let n := 8 in 
  ∀ (sum_of_interior_angles : ℕ → ℕ) (regular_polygon_interior_angle : ℕ → ℕ),
  sum_of_interior_angles n = (180 * (n - 2)) →
  regular_polygon_interior_angle n = (sum_of_interior_angles n / n) →
  regular_polygon_interior_angle n = 135 :=
by
  intros n sum_of_interior_angles regular_polygon_interior_angle h1 h2
  sorry

end interior_angle_regular_octagon_l22_22267


namespace sequence_k_eq_5_l22_22574

theorem sequence_k_eq_5  (a : ℕ → ℕ) (b : ℕ → ℕ) (h₁ : a 1 = 1)
  (h₂ : ∀ (p q : ℕ), a (p + q) = a p + a q) 
  (h₃ : ∀ (n : ℕ), b n = 2 ^ (a n)) :
  (Σ (k : ℕ), Σ (hk : (finset.range (k + 1)).sum (λ i, b i) = 62), k = 5) :=
sorry

end sequence_k_eq_5_l22_22574


namespace new_ratio_books_clothes_l22_22652

theorem new_ratio_books_clothes :
  ∀ (B C E : ℝ), (B = 22.5) → (C = 18) → (E = 9) → (C_new = C - 9) → C_new = 9 → B / C_new = 2.5 :=
by
  intros B C E HB HC HE HCnew Hnew
  sorry

end new_ratio_books_clothes_l22_22652


namespace max_distance_sin_cos_intersection_l22_22552

open Real

theorem max_distance_sin_cos_intersection (a : ℝ) :
  let y1 := sin a,
      y2 := cos a,
      MN := abs (y1 - y2)
  in MN ≤ sqrt 2 :=
by 
  let y1 := sin a
  let y2 := cos a
  let MN := abs (y1 - y2)
  have h1 : MN = sqrt 2 * abs (sin (a - π / 4)) := by sorry
  have h2 : abs (sin (a - π / 4)) ≤ 1 := abs_sin_le_one (a - π / 4)
  exact le_trans (by rw h1) (mul_le_mul_of_nonneg_left h2 (sqrt_nonneg 2))

end max_distance_sin_cos_intersection_l22_22552


namespace intersection_A_B_l22_22895

noncomputable def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_A_B : A ∩ B = {0, 1} := by 
  sorry

end intersection_A_B_l22_22895


namespace sum_mod_7_eq_5_l22_22787

theorem sum_mod_7_eq_5 : 
  (51730 + 51731 + 51732 + 51733 + 51734 + 51735) % 7 = 5 := 
by 
  sorry

end sum_mod_7_eq_5_l22_22787


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22192

theorem measure_of_one_interior_angle_of_regular_octagon 
  (n : Nat) (h_n : n = 8) : 
  (one_interior_angle : ℝ) = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22192


namespace regular_octagon_interior_angle_measure_l22_22246

theorem regular_octagon_interior_angle_measure :
  ∀ (n : ℕ), n = 8 → (∑ i in finset.range n, (180 : ℝ)) / n = 135 :=
  by
  sorry

end regular_octagon_interior_angle_measure_l22_22246


namespace sec_225_eq_neg_sqrt2_l22_22467

theorem sec_225_eq_neg_sqrt2 : ∀ θ : ℝ, θ = 225 * (π / 180) → sec θ = -real.sqrt 2 :=
by
  intro θ hθ
  sorry

end sec_225_eq_neg_sqrt2_l22_22467


namespace proof_problem_l22_22518

open Real

variable (p : Prop) (q : Prop)
def condition_p : Prop := ∃ x : ℝ, x - 10 > log10 x
def condition_q : Prop := ∀ x : ℝ, x^2 > 0

theorem proof_problem (h1 : condition_p p) (h2 : ¬ condition_q q) : p ∧ ¬ q :=
sorry

end proof_problem_l22_22518


namespace price_of_food_before_tax_and_tip_l22_22725

noncomputable def actual_price_of_food (total_paid : ℝ) (tip_rate tax_rate : ℝ) : ℝ :=
  total_paid / (1 + tip_rate) / (1 + tax_rate)

theorem price_of_food_before_tax_and_tip :
  actual_price_of_food 211.20 0.20 0.10 = 160 :=
by
  sorry

end price_of_food_before_tax_and_tip_l22_22725


namespace new_line_length_l22_22783

/-- Eli drew a line that was 1.5 meters long and then erased 37.5 centimeters of it.
    We need to prove that the length of the line now is 112.5 centimeters. -/
theorem new_line_length (initial_length_m : ℝ) (erased_length_cm : ℝ) 
    (h1 : initial_length_m = 1.5) (h2 : erased_length_cm = 37.5) :
    initial_length_m * 100 - erased_length_cm = 112.5 :=
by
  sorry

end new_line_length_l22_22783


namespace cylinder_lateral_area_l22_22643

theorem cylinder_lateral_area :
  ∀ (d h : ℝ), 
    d = 4 → h = 4 → 
      let r := d / 2 
      in 2 * Real.pi * r * h = 16 * Real.pi := 
by
  intros d h h_d h_h
  let r := d / 2
  sorry

end cylinder_lateral_area_l22_22643


namespace sec_225_deg_l22_22458

theorem sec_225_deg : real.sec (225 * real.pi / 180) = -real.sqrt 2 :=
by
  -- Conditions
  have h1 : real.cos (225 * real.pi / 180) = -real.cos (45 * real.pi / 180),
  { rw [←real.cos_add_pi_div_two, real.cos_pi_div_two_sub],
    norm_num, },
  have h2 : real.cos (45 * real.pi / 180) = real.sqrt 2 / 2,
  { exact real.cos_pi_div_four, },
  rw [h1, h2, real.sec],
  field_simp,
  norm_num,
  sorry

end sec_225_deg_l22_22458


namespace infinite_double_sum_l22_22379

theorem infinite_double_sum :
  (∑ j, ∑ k, 2 ^ - (4 * k + j + (k + j) ^ 2) : ℝ) = 4 / 3 :=
sorry

end infinite_double_sum_l22_22379


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22152

theorem measure_of_one_interior_angle_of_regular_octagon :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 := 
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22152


namespace partition_nat_set_l22_22021

theorem partition_nat_set :
  ∃ (P : ℕ → ℕ), (∀ (n : ℕ), P n < 100) ∧ (∀ (a b c : ℕ), a + 99 * b = c → (P a = P b ∨ P b = P c ∨ P c = P a)) :=
sorry

end partition_nat_set_l22_22021


namespace Odette_first_no_wins_Evie_first_odd_n_wins_l22_22435

-- Definitions and conditions
def pebble_positions : Type := ℤ × ℤ × ℤ

def game_ends (positions : pebble_positions) : Prop :=
  abs (positions.1 - positions.2) = 1 ∧ abs (positions.2 - positions.3) = 1 ∧ abs (positions.1 - positions.3) = 2

def sum_is_even (positions : pebble_positions) : Prop :=
  (positions.1 + positions.2 + positions.3) % 2 = 0

def sum_is_odd (positions : pebble_positions) : Prop :=
  ¬sum_is_even positions

def Evie_wins (positions : pebble_positions) : Prop := sum_is_even positions
def Odette_wins (positions : pebble_positions) : Prop := sum_is_odd positions

-- Problem statements
theorem Odette_first_no_wins {n : ℤ} (hn : -2020 ≤ n ∧ n ≤ 2020) :
  (∀ positions : pebble_positions, game_ends positions → ¬Evie_wins positions) :=
sorry

theorem Evie_first_odd_n_wins :
  (∀ n, -2020 ≤ n ∧ n ≤ 2020 → odd n → ∃ positions : pebble_positions, game_ends positions ∧ Evie_wins positions) :=
sorry

end Odette_first_no_wins_Evie_first_odd_n_wins_l22_22435


namespace sin_C_half_l22_22559

variables {α β γ a b c : ℝ}

-- Definitions used in the conditions
def is_triangle (A B C : Type) : Prop :=
  ∀ {α β γ : ℝ}, α + β + γ = π

def law_of_sines (a b c : ℝ) (α β γ : ℝ) : Prop :=
  a / sin α = b / sin β ∧ b / sin β = c / sin γ

-- The condition given in the problem
def given_condition (b c sinB : ℝ) : Prop :=
  b = 2 * c * sinB

-- The statement of the theorem
theorem sin_C_half (a b c sinB α β γ : ℝ) 
  (triangle : is_triangle α β γ) 
  (law_sines : law_of_sines a b c α β γ)
  (cond : given_condition b c (sin β)) : 
  sin γ = 1 / 2 :=
sorry

end sin_C_half_l22_22559


namespace enclosed_area_by_four_smaller_circles_l22_22305

theorem enclosed_area_by_four_smaller_circles (R : ℝ) (r : ℝ) 
  (h1 : r = R * (Real.sqrt 2 - 1)) 
  (h2 : r > 0) :
  let shaded_area := R^2 * (4 - Real.pi) * (3 - 2 * Real.sqrt 2)
  in shaded_area = R^2 * (4 - Real.pi) * (3 - 2 * Real.sqrt 2) :=
by
  intro shaded_area
  rw [h1]
  sorry

end enclosed_area_by_four_smaller_circles_l22_22305


namespace sum_xyz_l22_22545

open Real

noncomputable def x (x_val : ℝ) := log 3 (log 2 (log 5 x_val)) = 0
noncomputable def y (y_val : ℝ) := log 2 (log 5 (log 3 y_val)) = 0
noncomputable def z (z_val : ℝ) := log 5 (log 3 (log 2 z_val)) = 0

theorem sum_xyz {x_val y_val z_val : ℝ} (hx : x x_val) (hy : y y_val) (hz : z z_val) : x_val + y_val + z_val = 276 := 
by 
  sorry

end sum_xyz_l22_22545


namespace geometric_sequence_a_n_common_ratio_general_term_b_n_l22_22880

open Nat

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

axiom a1 : ∀ n : ℕ, 2 * S n = a (n + 2) - 2
axiom a2 : ∀ n : ℕ, S (n + 1) = S n + a (n + 1)

theorem geometric_sequence_a_n_common_ratio (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = q * a n) ∧ a 0 ≠ 0 ∧ q > 0 →
  q = 2 :=
by admit

theorem general_term_b_n (a1 a2 : ℝ) (q : ℝ) :
  a 1 = a1 → a 2 = a2 → q = 2 →
  ∀ n : ℕ, b n = if n = 0 then a1 + a2 else 5 * 2^(n-1) :=
by admit

end geometric_sequence_a_n_common_ratio_general_term_b_n_l22_22880


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22137

theorem measure_of_one_interior_angle_of_regular_octagon : 
  let n := 8 in 
  let sum_of_interior_angles := 180 * (n - 2) in
  let measure_of_one_interior_angle := sum_of_interior_angles / n in 
  measure_of_one_interior_angle = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22137


namespace sharpened_off_pencils_length_l22_22578

theorem sharpened_off_pencils_length :
  let original_lengths := [31, 42, 25]
  let sharpened_lengths := [14, 19, 11]
  let sharpened_off_parts := list.map2 (λ o s => o - s) original_lengths sharpened_lengths
  list.sum sharpened_off_parts = 54 :=
by
  sorry

end sharpened_off_pencils_length_l22_22578


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22175

-- Define the conditions for the problem
def num_sides : ℕ := 8
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Statement of the problem
theorem measure_of_one_interior_angle_of_regular_octagon : 
  (sum_of_interior_angles num_sides) / num_sides = 135 :=
by
  -- Conditions are defined above
  -- Proof steps are omitted
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22175


namespace largest_A_divisible_by_8_l22_22277

theorem largest_A_divisible_by_8 (A B C : ℕ) (h1 : A = 8 * B + C) (h2 : B = C) (h3 : C < 8) : A ≤ 9 * 7 :=
by sorry

end largest_A_divisible_by_8_l22_22277


namespace EF_parallel_BC_l22_22493

variable {A B C I E F : Type*}
variable [nonempty (triangle A B C)]
variable [incenter_triangle I A B C]
variable [points_on_rays E F B I C I]
variable (h1 : distance I A = distance A E)
variable (h2 : distance A E = distance A F)

theorem EF_parallel_BC (h : distance I A = distance A E ∧ distance A E = distance A F) : 
  parallel E F B C := sorry

end EF_parallel_BC_l22_22493


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22173

-- Define the conditions for the problem
def num_sides : ℕ := 8
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Statement of the problem
theorem measure_of_one_interior_angle_of_regular_octagon : 
  (sum_of_interior_angles num_sides) / num_sides = 135 :=
by
  -- Conditions are defined above
  -- Proof steps are omitted
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22173


namespace sum_arithmetic_series_base_6_to_base_10_l22_22437

theorem sum_arithmetic_series_base_6_to_base_10 :
  let sum_n_base_6 (n : ℕ) := ∑ i in Finset.range (n + 1), base6_to_nat i 
  in sum_n_base_6 55 = 630 :=
sorry

end sum_arithmetic_series_base_6_to_base_10_l22_22437


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22142

theorem measure_of_one_interior_angle_of_regular_octagon : 
  let n := 8 in 
  let sum_of_interior_angles := 180 * (n - 2) in
  let measure_of_one_interior_angle := sum_of_interior_angles / n in 
  measure_of_one_interior_angle = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22142


namespace lcm_24_90_l22_22473

theorem lcm_24_90 : lcm 24 90 = 360 :=
by 
-- lcm is the least common multiple of 24 and 90.
-- lcm 24 90 is defined as 360.
sorry

end lcm_24_90_l22_22473


namespace total_distance_l22_22754

-- We will define the points as tuples of real numbers
def A : ℝ × ℝ := (-3, 6)
def B : ℝ × ℝ := (2, 2)
def C : ℝ × ℝ := (0, 0)
def D : ℝ × ℝ := (6, -3)

-- Function to calculate Euclidean distance between two points (x1, y1) and (x2, y2)
def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

-- Prove the total distance using the distances from the specified segments
theorem total_distance : distance A B + distance B C + distance C D = Real.sqrt 41 + 2 * Real.sqrt 2 + 3 * Real.sqrt 5 := by
  sorry

end total_distance_l22_22754


namespace janet_time_to_home_l22_22579

-- Janet's initial and final positions
def initial_position : ℕ × ℕ := (0, 0) -- (x, y)
def north_blocks : ℕ := 3
def west_multiplier : ℕ := 7
def south_blocks : ℕ := 8
def east_multiplier : ℕ := 2
def speed_blocks_per_minute : ℕ := 2

def west_blocks : ℕ := west_multiplier * north_blocks
def east_blocks : ℕ := east_multiplier * south_blocks

-- Net movement calculations
def net_south_blocks : ℕ := south_blocks - north_blocks
def net_west_blocks : ℕ := west_blocks - east_blocks

-- Time calculation
def total_blocks_to_home : ℕ := net_south_blocks + net_west_blocks
def time_to_home : ℕ := total_blocks_to_home / speed_blocks_per_minute

theorem janet_time_to_home : time_to_home = 5 := by
  -- Proof goes here
  sorry

end janet_time_to_home_l22_22579


namespace lebesgue_measure_invariance_l22_22030

-- Definitions of the properties of Lebesgue measure to translate the problem conditions

open Set
open MeasureTheory

variables {n : ℕ} (hn1 : n > 1) (hn2 : n ≥ 1)

def lebesgue_measure_invariant_under_translations (λ : MeasureTheory.Measure ℝ^n) : Prop :=
  ∀ (x : ℝ^n) (E : Set ℝ^n), MeasurableSet E → λ (x +ᵥ E) = λ E

def lebesgue_measure_invariant_under_rotations (λ : MeasureTheory.Measure ℝ^n) : Prop :=
  ∀ (φ : ℝ^n →ₗ[ℝ] ℝ^n), φ.IsLinearMap φ → φ ∘ φ⁻¹ = id →
  ∀ E : Set ℝ^n, MeasurableSet E → λ (E) = λ (φ '' E)

theorem lebesgue_measure_invariance {λ : MeasureTheory.Measure ℝ^n} :
  (∀ B ∈ MeasurableSet ℝ^n, λ(B) = inf { λ(G) | B ⊆ G ∧ IsOpen G }) →
  lebesgue_measure_invariant_under_translations λ →
  lebesgue_measure_invariant_under_rotations λ :=
by 
  sorry

# Let λ be the Lebesgue measure on ℝ^n, the theorem states that
# λ is invariant under translations for n ≥ 1 and under rotations for n > 1.

end lebesgue_measure_invariance_l22_22030


namespace inscribed_circle_radius_l22_22587

theorem inscribed_circle_radius (a r : ℝ) (unit_square : a = 1)
  (touches_arc_AC : ∀ (x : ℝ × ℝ), x.1^2 + x.2^2 = (a - r)^2)
  (touches_arc_BD : ∀ (y : ℝ × ℝ), y.1^2 + y.2^2 = (a - r)^2)
  (touches_side_AB : ∀ (z : ℝ × ℝ), z.1 = r ∨ z.2 = r) :
  r = 3 / 8 := by sorry

end inscribed_circle_radius_l22_22587


namespace number_of_non_intersecting_paths_l22_22338

open Set

-- Define the up-right path predicate
def up_right_path (a b c d : ℕ × ℕ) (path : List (ℕ × ℕ)) : Prop :=
  path.head? = some a ∧ path.head? = some b ∧ (path :|> init).tail? = some c ∧ 
  (path :|> init).tail? = some d ∧ 
  ∀ i, i < path.length - 1 → 
    (path.nth i).isSome ∧ (path.nth (i + 1)).isSome ∧
    ((path.nth i).get = (path.nth (i + 1)).get + (1, 0) ∨ 
    (path.nth i).get = (path.nth (i + 1)).get + (0, 1))

-- Define the paths non-intersect condition
def paths_non_intersect (A B : List (ℕ × ℕ)) : Prop :=
  disjoint (A.to_set) (B.to_set)

-- The main theorem statement
theorem number_of_non_intersecting_paths :
  {A | up_right_path (0, 0) (4, 4) A}.card *
  {B | up_right_path (2, 0) (6, 4) B}.card -
  {C | up_right_path (0, 0) (6, 4) C}.card *
  {D | up_right_path (2, 0) (4, 4) D}.card = 1750 :=
sorry

end number_of_non_intersecting_paths_l22_22338


namespace max_correct_answers_l22_22561

theorem max_correct_answers : ∃ (x y z : ℕ), 
  x + y + z = 50 ∧
  3 * x - y = 120 ∧
  (∀ (x' : ℕ), x' + y + z = 50 → 3 * x' - y = 120 → x' ≤ 42) ∧
  x = 42 :=
by
  have h : ∃ (x y z : ℕ), x + y + z = 50 ∧ 3 * x - y = 120 ∧ x = 42
  {
    use 42, 8, 0,
    split; norm_num,
    split; norm_num,
  }
  cases h with x hx,
  use x,
  exact hx,
  sorry

end max_correct_answers_l22_22561


namespace thomas_savings_years_l22_22665

def weekly_allowance : ℕ := 50
def weekly_coffee_shop_earning : ℕ := 9 * 30
def weekly_spending : ℕ := 35
def car_cost : ℕ := 15000
def additional_amount_needed : ℕ := 2000
def weeks_in_a_year : ℕ := 52

def first_year_savings : ℕ := weeks_in_a_year * (weekly_allowance - weekly_spending)
def second_year_savings : ℕ := weeks_in_a_year * (weekly_coffee_shop_earning - weekly_spending)

noncomputable def total_savings_needed : ℕ := car_cost - additional_amount_needed

theorem thomas_savings_years : 
  first_year_savings + second_year_savings = total_savings_needed → 2 = 2 :=
by
  sorry

end thomas_savings_years_l22_22665


namespace mark_charged_more_hours_than_kate_l22_22017

variables (K P M : ℝ)
variables (h1 : K + P + M = 198) (h2 : P = 2 * K) (h3 : M = 3 * P)

theorem mark_charged_more_hours_than_kate : M - K = 110 :=
by
  sorry

end mark_charged_more_hours_than_kate_l22_22017


namespace max_buses_l22_22926

-- Define the total number of stops and buses as finite sets
def stops : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define a bus as a subset of stops with exactly 3 stops
structure Bus :=
(stops : Finset ℕ)
(h_stops : stops.card = 3)

-- Define the condition that any two buses share at most one stop
def shares_at_most_one (b1 b2 : Bus) : Prop :=
(b1.stops ∩ b2.stops).card ≤ 1

-- Define the predicate for a valid set of buses
def valid_buses (buses : Finset Bus) : Prop :=
∀ b1 b2 ∈ buses, b1 ≠ b2 → shares_at_most_one b1 b2

-- The main theorem statement
theorem max_buses (buses : Finset Bus) (h_valid : valid_buses buses) : buses.card ≤ 12 :=
sorry

end max_buses_l22_22926


namespace inequality_solution_set_l22_22060

theorem inequality_solution_set (x : ℝ) : (x - 1) * abs (x + 2) ≥ 0 ↔ (x ≥ 1 ∨ x = -2) :=
by
  sorry

end inequality_solution_set_l22_22060


namespace sec_225_deg_l22_22456

theorem sec_225_deg : real.sec (225 * real.pi / 180) = -real.sqrt 2 :=
by
  -- Conditions
  have h1 : real.cos (225 * real.pi / 180) = -real.cos (45 * real.pi / 180),
  { rw [←real.cos_add_pi_div_two, real.cos_pi_div_two_sub],
    norm_num, },
  have h2 : real.cos (45 * real.pi / 180) = real.sqrt 2 / 2,
  { exact real.cos_pi_div_four, },
  rw [h1, h2, real.sec],
  field_simp,
  norm_num,
  sorry

end sec_225_deg_l22_22456


namespace probability_two_defective_l22_22062

theorem probability_two_defective (total_screws : ℕ) (defective_screws : ℕ) (drawn_screws : ℕ) (exactly_defective : ℕ) 
  (h_total : total_screws = 10) (h_defective : defective_screws = 3) (h_drawn : drawn_screws = 4) (h_exactly : exactly_defective = 2) :
  let C := Nat.choose in
  (C defective_screws exactly_defective * C (total_screws - defective_screws) (drawn_screws - exactly_defective)) / 
  (C total_screws drawn_screws) = 3 / 10 :=
by
  sorry

end probability_two_defective_l22_22062


namespace range_of_a_l22_22882

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), e-1 < x ∧ x < e^2-1 → f x = 1/(a * ln (x+1)) ∧ g x < 0 → g x = -x^3 + x^2) →
  (∃ (x1 x2 : ℝ), e - 1 < x1 ∧ x1 < e^2 - 1 ∧ x2 < 0 ∧
    (x1, f x1) ≠ (0, 0) ∧ (x2, g x2) ≠ (0, 0) ∧
    (x1 = -x2) ∧
    ((x1^2 - 1 / (a * ln (x1+1))) * (x1^3 + x1^2) = 0)) → 
  e < a ∧ a < e^2 / 2 := 
begin
  sorry
end

end range_of_a_l22_22882


namespace ratio_of_side_lengths_of_frustum_l22_22845

theorem ratio_of_side_lengths_of_frustum (L1 L2 H : ℚ) (V_prism V_frustum : ℚ)
  (h1 : V_prism = L1^2 * H)
  (h2 : V_frustum = (1/3) * (L1^2 * (H * (L1 / (L1 - L2))) - L2^2 * (H * (L2 / (L1 - L2)))))
  (h3 : V_frustum = (2/3) * V_prism) :
  L1 / L2 = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end ratio_of_side_lengths_of_frustum_l22_22845


namespace heaviest_tv_l22_22752

theorem heaviest_tv :
  let area (width height : ℝ) := width * height
  let weight (area : ℝ) := area * 4
  let weight_in_pounds (weight : ℝ) := weight / 16
  let bill_area := area 48 100
  let bob_area := area 70 60
  let steve_area := area 84 92
  let bill_weight := weight bill_area
  let bob_weight := weight bob_area
  let steve_weight := weight steve_area
  let bill_weight_pounds := weight_in_pounds (weight bill_area)
  let bob_weight_pounds := weight_in_pounds (weight bob_area)
  let steve_weight_pounds := weight_in_pounds (weight steve_area)
  bob_weight_pounds + bill_weight_pounds < steve_weight_pounds
  ∧ abs ((steve_weight_pounds) - (bill_weight_pounds + bob_weight_pounds)) = 318 :=
by
  sorry

end heaviest_tv_l22_22752


namespace base9_problem_l22_22741

def base9_add (a : ℕ) (b : ℕ) : ℕ := sorry -- Define actual addition for base 9
def base9_mul (a : ℕ) (b : ℕ) : ℕ := sorry -- Define actual multiplication for base 9

theorem base9_problem : base9_mul (base9_add 35 273) 2 = 620 := sorry

end base9_problem_l22_22741


namespace floor_difference_l22_22405

theorem floor_difference (x : ℝ) (h : x = 13.2) : 
  (⌊x^2⌋ : ℤ) - ((⌊x⌋ : ℤ) * (⌊x⌋ : ℤ)) = 5 := 
by
  -- proof skipped
  sorry

end floor_difference_l22_22405


namespace total_colors_needed_l22_22946

def num_planets : ℕ := 8
def num_people : ℕ := 3

theorem total_colors_needed : num_people * num_planets = 24 := by
  sorry

end total_colors_needed_l22_22946


namespace largest_integer_with_square_three_digits_base_7_l22_22582

theorem largest_integer_with_square_three_digits_base_7 : 
  ∃ M : ℕ, (7^2 ≤ M^2 ∧ M^2 < 7^3) ∧ ∀ n : ℕ, (7^2 ≤ n^2 ∧ n^2 < 7^3) → n ≤ M := 
sorry

end largest_integer_with_square_three_digits_base_7_l22_22582


namespace regular_octagon_interior_angle_l22_22218

theorem regular_octagon_interior_angle:
  let n := 8
  let sum_interior_angles (n: ℕ) := 180 * (n - 2)
  let interior_angle (n: ℕ) := sum_interior_angles n / n
  interior_angle n = 135 :=
by
  sorry

end regular_octagon_interior_angle_l22_22218


namespace trajectory_of_M_minimum_area_PRN_l22_22956

-- Define the coordinates of point A
def A := (1/2 : ℝ, 0 : ℝ)

-- Define the line on which point B lies
def line_B (x : ℝ) := x = -1/2

-- Define point C as the intersection of line segment AB and the y-axis
def point_C (A B : ℝ × ℝ) : ℝ × ℝ := (0, (snd B * (1/2) - snd A * (1/2)) / (fst B - fst A / 2))

-- Define vector dot product condition for vectors BM and OC
def BM_dot_OC_zero (B M C : ℝ × ℝ) : Bool := (fst (M - B)) * (fst (C - (0 : ℝ × ℝ))) + 
                                              (snd (M - B)) * (snd (C - (0 : ℝ × ℝ))) = 0

-- Define vector dot product condition for vectors CM and AB
def CM_dot_AB_zero (C M A B : ℝ × ℝ) : Bool := (fst (M - C)) * (fst (B - A)) + 
                                               (snd (M - C)) * (snd (B - A)) = 0

-- Define the trajectory E of moving point M
theorem trajectory_of_M (A : ℝ × ℝ) :
  ∀ A B C M, (A = (1/2, 0)) → (line_B (fst B)) →
             (point_C A B = C) →
             (BM_dot_OC_zero B M C) →
             (CM_dot_AB_zero C M A B) →
             ((snd M)^2 = 2 * fst M) :=
begin
  sorry
end

-- Define the distance from a point to a line
def distance_from_point_to_line (P R : ℝ × ℝ) (slope intercept : ℝ) : ℝ :=
  abs ((snd P - snd R) + slope * (fst P) - (fst P) * (intercept)) /
      sqrt (slope^2 + 1)

-- Define the minimum area of triangle PRN
theorem minimum_area_PRN :
  ∀ P R N, (circle_center = (1, 0)) → 
           ((fst P - 1)^2 + (snd P) ^ 2 = 1) →
           (fst R = 0) → (fst N = 0) →
           ∃ (x0 y0 : ℝ), 
           (distance_from_point_to_line (1, 0) P ((y0 - snd R) / (fst P)) ((1, 0)) = 1) →
           (minimum_area_triangle_PRN (P R N) = 8) :=
begin
  sorry
end

end trajectory_of_M_minimum_area_PRN_l22_22956


namespace floor_difference_l22_22407

theorem floor_difference (x : ℝ) (h : x = 13.2) : 
  (⌊x^2⌋ : ℤ) - ((⌊x⌋ : ℤ) * (⌊x⌋ : ℤ)) = 5 := 
by
  -- proof skipped
  sorry

end floor_difference_l22_22407


namespace largest_prime_factor_of_sum_of_divisors_180_l22_22980

def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in (list.range (n+1)).filter (λ d, n % d = 0), d

def largest_prime_factor (n : ℕ) : ℕ :=
  (list.range (n+1)).filter (λ p, nat.prime p ∧ n % p = 0).last'

theorem largest_prime_factor_of_sum_of_divisors_180 :
  largest_prime_factor (sum_of_divisors 180) = 13 := by sorry

end largest_prime_factor_of_sum_of_divisors_180_l22_22980


namespace ellipse_representation_l22_22871

theorem ellipse_representation (θ : ℝ) (h1 : sin θ * cos θ ≠ 0) (h2 : sin θ > 0) :
  ∃ a b c e : ℝ, (x^2 * (sin θ)^2 * (cos θ)^2 + y^2 * (sin θ)^2 = (cos θ)^2) ∧
              (Foci_on_x_axis (x^2 / (1 / (sin θ)^2) + y^2 / ((cos θ)^2 / (sin θ)^2) = 1)) ∧ 
              (e = sin θ) :=
sorry

end ellipse_representation_l22_22871


namespace calculate_angle_l22_22756

def degrees_to_seconds (d m s : ℕ) : ℕ :=
  d * 3600 + m * 60 + s

def seconds_to_degrees (s : ℕ) : (ℕ × ℕ × ℕ) :=
  (s / 3600, (s % 3600) / 60, s % 60)

theorem calculate_angle : 
  (let d1 := 50
   let m1 := 24
   let angle1_sec := degrees_to_seconds d1 m1 0
   let angle1_sec_tripled := 3 * angle1_sec
   let (d1', m1', s1') := seconds_to_degrees angle1_sec_tripled

   let d2 := 98
   let m2 := 12
   let s2 := 25
   let angle2_sec := degrees_to_seconds d2 m2 s2
   let angle2_sec_divided := angle2_sec / 5
   let (d2', m2', s2') := seconds_to_degrees angle2_sec_divided

   let total_sec := degrees_to_seconds d1' m1' s1' + degrees_to_seconds d2' m2' s2'
   let (final_d, final_m, final_s) := seconds_to_degrees total_sec
   (final_d, final_m, final_s)) = (170, 50, 29) := by sorry

end calculate_angle_l22_22756


namespace largest_prime_factor_of_sum_of_divisors_180_l22_22979

def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in (list.range (n+1)).filter (λ d, n % d = 0), d

def largest_prime_factor (n : ℕ) : ℕ :=
  (list.range (n+1)).filter (λ p, nat.prime p ∧ n % p = 0).last'

theorem largest_prime_factor_of_sum_of_divisors_180 :
  largest_prime_factor (sum_of_divisors 180) = 13 := by sorry

end largest_prime_factor_of_sum_of_divisors_180_l22_22979


namespace find_number_divided_by_7_l22_22684

theorem find_number_divided_by_7 : ∃ x : ℕ, x = 7 * 4 + 6 ∧ x = 34 :=
by {
  use 34,
  split,
  { sorry },  -- Here we would need to show 34 = 7 * 4 + 6
  { sorry },  -- Here we would need to show 34 = 34
}

end find_number_divided_by_7_l22_22684


namespace regular_octagon_interior_angle_l22_22189

theorem regular_octagon_interior_angle : 
  ∀ (n : ℕ), n = 8 → (sum_of_internal_angles n) / n = 135 := 
by
  intros n hn
  rw hn
  sorry

end regular_octagon_interior_angle_l22_22189


namespace range_of_m_intersection_l22_22499

theorem range_of_m_intersection (m : ℝ) :
  (∀ k : ℝ, ∃ x y : ℝ, y - k * x - 1 = 0 ∧ (x^2 / 4) + (y^2 / m) = 1) ↔ (m ∈ Set.Ico 1 4 ∪ Set.Ioi 4) :=
by
  sorry

end range_of_m_intersection_l22_22499


namespace simplify_and_rationalize_denominator_l22_22628

theorem simplify_and_rationalize_denominator :
  (∀ (a b c : ℝ), a = (3 : ℝ) ∧ b = 10010 ∧ c = 1001 → 
  (∃ (x y z : ℝ), x = a ∧ y = b ∧ z = c → 
  ((a * (Real.sqrt b)) / c = (3 * Real.sqrt 10010) / 1001))) :=
by
  intro a b c h
  obtain ⟨ha, hb, hc⟩ := h
  use [a, b, c]
  simp [ha, hb, hc]
  sorry

end simplify_and_rationalize_denominator_l22_22628


namespace find_point_B_coords_l22_22958

theorem find_point_B_coords
  (A : ℝ × ℝ)
  (hA : A = (2, -1))
  (AB_parallel_y : ∀ B : ℝ × ℝ, B.1 = A.1 → LineSegmentLength A B = 3 → B.1 = A.1)
  (hAB_len : ∀ B : ℝ × ℝ, LineSegmentLength A B = 3 → (B.2 = A.2 + 3 ∨ B.2 = A.2 - 3)) :
  ∃ B : ℝ × ℝ, (B = (2, 2) ∨ B = (2, -4)) := by
  sorry

-- Helper definition for computing the length of a line segment
def LineSegmentLength (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

end find_point_B_coords_l22_22958


namespace find_m_value_l22_22868

variable (m a0 a1 a2 a3 a4 a5 : ℚ)

-- Defining the conditions given in the problem
def poly_expansion_condition : Prop := (m * 1 - 1)^5 = a5 * 1^5 + a4 * 1^4 + a3 * 1^3 + a2 * 1^2 + a1 * 1 + a0
def a1_a2_a3_a4_a5_condition : Prop := a1 + a2 + a3 + a4 + a5 = 33

-- We are required to prove that given these conditions, m = 3.
theorem find_m_value (h1 : a0 = -1) (h2 : poly_expansion_condition m a0 a1 a2 a3 a4 a5) 
(h3 : a1_a2_a3_a4_a5_condition a1 a2 a3 a4 a5) : m = 3 := by
  sorry

end find_m_value_l22_22868


namespace angle_is_60_degrees_l22_22899

noncomputable theory

open Real

def vector_a : ℝ × ℝ := (1, real.sqrt 3)
def vector_b : ℝ × ℝ := (-2, 2 * real.sqrt 3)

def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def angle_between_vectors (a b : ℝ × ℝ) : ℝ :=
  real.acos ((dot_product a b) / (magnitude a * magnitude b))

theorem angle_is_60_degrees :
  angle_between_vectors vector_a vector_b = real.pi / 3 := -- 60 degrees in radians
sorry

end angle_is_60_degrees_l22_22899


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22134

theorem measure_of_one_interior_angle_of_regular_octagon : 
  let n := 8 in 
  let sum_of_interior_angles := 180 * (n - 2) in
  let measure_of_one_interior_angle := sum_of_interior_angles / n in 
  measure_of_one_interior_angle = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22134


namespace interior_angle_regular_octagon_l22_22261

theorem interior_angle_regular_octagon : 
  let n := 8 in 
  ∀ (sum_of_interior_angles : ℕ → ℕ) (regular_polygon_interior_angle : ℕ → ℕ),
  sum_of_interior_angles n = (180 * (n - 2)) →
  regular_polygon_interior_angle n = (sum_of_interior_angles n / n) →
  regular_polygon_interior_angle n = 135 :=
by
  intros n sum_of_interior_angles regular_polygon_interior_angle h1 h2
  sorry

end interior_angle_regular_octagon_l22_22261


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22201

theorem measure_of_one_interior_angle_of_regular_octagon 
  (n : Nat) (h_n : n = 8) : 
  (one_interior_angle : ℝ) = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22201


namespace arrangement_count_l22_22654

def student := {A, B, C, D, E}

def track := ℕ
def swimming := ℕ
def ball_games := ℕ 

def valid_arrangement (students : {track := track, swimming := swimming, ball_games := ball_games}) : Prop :=
  students.swimming ≠ student.A

theorem arrangement_count : 
  ∃ arrangements : finset ({track := track, swimming := swimming, ball_games := ball_games}),
  arrangements.card = 48 ∧ ∀ arrangement ∈ arrangements, valid_arrangement arrangement := 
sorry

end arrangement_count_l22_22654


namespace regular_octagon_angle_measure_l22_22207

theorem regular_octagon_angle_measure :
  let n := 8 in
  (180 * (n - 2)) / n = 135 := 
by
  sorry

end regular_octagon_angle_measure_l22_22207


namespace floor_sq_minus_sq_floor_l22_22422

noncomputable def floor_13_2 : ℤ := int.floor 13.2

theorem floor_sq_minus_sq_floor :
  int.floor ((13.2 : ℝ) ^ 2) - (floor_13_2 * floor_13_2) = 5 :=
by
  let floor_13_2_sq := floor_13_2 * floor_13_2
  have h1 : int.floor (13.2 : ℝ) = 13 := by norm_num
  have h2 : int.floor ((13.2 : ℝ) ^ 2) = 174 := by norm_num
  rw [h1] at floor_13_2
  rw [h2]
  rw [floor_13_2, floor_13_2, floor_13_2]
  exact sorry

end floor_sq_minus_sq_floor_l22_22422


namespace regular_octagon_interior_angle_l22_22123

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : 
  let sum_interior_angles := 180 * (n - 2)
  let num_sides := 8
  let one_interior_angle := sum_interior_angles / num_sides
  one_interior_angle = 135 :=
by
  sorry

end regular_octagon_interior_angle_l22_22123


namespace fields_acres_l22_22569

variables (x y : ℝ)

def good_field_value := 300
def bad_field_value := 500
def total_area := 100 -- Given in acreage (1 hectare = 100 acres)
def total_cost := 10000

theorem fields_acres (h1 : x + y = total_area) 
                     (h2 : good_field_value * x + (bad_field_value / 7) * y = total_cost) : 
                     x + y = total_area ∧ good_field_value * x + (bad_field_value / 7) * y = total_cost :=
by {
    exact ⟨h1, h2⟩, sorry
}

end fields_acres_l22_22569


namespace olympic_system_participants_win_more_than_lose_l22_22321

theorem olympic_system_participants_win_more_than_lose :
  (∃ (participants : ℕ), participants = 64) →
  (∃ (rounds : ℕ), rounds = 6) →
  (∃ (pairs_per_round : ℕ → ℕ), pairs_per_round = λ round, 2^round) →
  (∀ (wins : ℕ), wins = 6 → 64 / (2^6) = 1) →
  ∀ (participants_adv_third_round : ℕ), participants_adv_third_round = 16 :=
begin
  intros _ _ _ _,
  exact 16,
end

end olympic_system_participants_win_more_than_lose_l22_22321


namespace regular_octagon_angle_measure_l22_22216

theorem regular_octagon_angle_measure :
  let n := 8 in
  (180 * (n - 2)) / n = 135 := 
by
  sorry

end regular_octagon_angle_measure_l22_22216


namespace triangle_area_ratio_l22_22960

theorem triangle_area_ratio
  (A B C D E: Type*)
  (hA : angle A = 60 * (π / 180))
  (hB : angle B = 45 * (π / 180))
  (hADE : angle ADE = 45 * (π / 180))
  (hEqualAreas : area (triangle A D E) = area (triangle D E B)) :
  AD / AB = (Real.sqrt 2) / 4 :=
by
  /- Proof goes here -/
  sorry

end triangle_area_ratio_l22_22960


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22149

theorem measure_of_one_interior_angle_of_regular_octagon :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 := 
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22149


namespace parabola_c_value_l22_22318

theorem parabola_c_value (b c : ℝ) 
  (h1 : 6 = 2^2 + 2 * b + c) 
  (h2 : 20 = 4^2 + 4 * b + c) : 
  c = 0 :=
by {
  -- We state that we're skipping the proof
  sorry
}

end parabola_c_value_l22_22318


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22168

-- Define the conditions for the problem
def num_sides : ℕ := 8
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Statement of the problem
theorem measure_of_one_interior_angle_of_regular_octagon : 
  (sum_of_interior_angles num_sides) / num_sides = 135 :=
by
  -- Conditions are defined above
  -- Proof steps are omitted
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22168


namespace sum_of_squares_of_solutions_l22_22827

theorem sum_of_squares_of_solutions :
  (∑ x in {x : ℝ | | x^2 - x + 1/2010 | = 1/2010}, x^2) = 2008 / 1005 :=
by
  sorry

end sum_of_squares_of_solutions_l22_22827


namespace train_passing_time_l22_22740

-- Define the conditions
def length_of_train := 360 -- in meters
def speed_of_train_kmph := 45 -- in km/hr
def length_of_platform := 240 -- in meters

-- Conversion constant
def kmph_to_mps (kmph: ℕ) : ℕ := kmph * 1000 / 3600

-- Define the proof problem statement
theorem train_passing_time : 
  let total_distance := length_of_train + length_of_platform in
  let speed_of_train := kmph_to_mps speed_of_train_kmph in
  let passing_time := total_distance / speed_of_train in
  passing_time = 48 := 
by
  sorry

end train_passing_time_l22_22740


namespace sum_of_squares_of_solutions_l22_22826

theorem sum_of_squares_of_solutions :
  let a := (1 : ℝ) / 2010
  in (∀ x : ℝ, abs(x^2 - x + a) = a) → 
     ((0^2 + 1^2) + ((((1*1) - 2 * a) + 1) / a)) = 2008 / 1005 :=
by
  intros a h
  sorry

end sum_of_squares_of_solutions_l22_22826


namespace sum_of_squares_of_solutions_l22_22815

theorem sum_of_squares_of_solutions :
  (∑ α in {x | ∃ ε : ℝ, ε = x² - x + 1/2010 ∧ |ε| = 1/2010}, α^2) = 2008 / 1005 :=
by
  sorry

end sum_of_squares_of_solutions_l22_22815


namespace sec_225_eq_neg_sqrt2_l22_22439

theorem sec_225_eq_neg_sqrt2 :
  sec 225 = -sqrt 2 :=
by 
  -- Definitions for conditions
  have h1 : ∀ θ : ℕ, sec θ = 1 / cos θ := sorry,
  have h2 : cos 225 = cos (180 + 45) := sorry,
  have h3 : cos (180 + 45) = -cos 45 := sorry,
  have h4 : cos 45 = 1 / sqrt 2 := sorry,
  -- Required proof statement
  sorry

end sec_225_eq_neg_sqrt2_l22_22439


namespace floor_sq_minus_sq_floor_l22_22420

noncomputable def floor_13_2 : ℤ := int.floor 13.2

theorem floor_sq_minus_sq_floor :
  int.floor ((13.2 : ℝ) ^ 2) - (floor_13_2 * floor_13_2) = 5 :=
by
  let floor_13_2_sq := floor_13_2 * floor_13_2
  have h1 : int.floor (13.2 : ℝ) = 13 := by norm_num
  have h2 : int.floor ((13.2 : ℝ) ^ 2) = 174 := by norm_num
  rw [h1] at floor_13_2
  rw [h2]
  rw [floor_13_2, floor_13_2, floor_13_2]
  exact sorry

end floor_sq_minus_sq_floor_l22_22420


namespace sec_225_eq_neg_sqrt_2_l22_22452

theorem sec_225_eq_neg_sqrt_2 :
  let sec (x : ℝ) := 1 / Real.cos x
  ∧ Real.cos (225 * Real.pi / 180) = Real.cos (180 * Real.pi / 180 + 45 * Real.pi / 180)
  ∧ Real.cos (180 * Real.pi / 180 + 45 * Real.pi / 180) = -Real.cos (45 * Real.pi / 180)
  ∧ Real.cos (45 * Real.pi / 180) = 1 / Real.sqrt 2
  → sec (225 * Real.pi / 180) = -Real.sqrt 2 :=
by 
  intros sec_h cos_h1 cos_h2 cos_h3
  sorry

end sec_225_eq_neg_sqrt_2_l22_22452


namespace max_buses_in_city_l22_22936

-- Definition of the problem statement and the maximal number of buses
theorem max_buses_in_city (bus_stops : Finset ℕ) (buses : Finset (Finset ℕ)) 
  (h_stops : bus_stops.card = 9) 
  (h_buses_stops : ∀ b ∈ buses, b.card = 3) 
  (h_shared_stops : ∀ b₁ b₂ ∈ buses, b₁ ≠ b₂ → (b₁ ∩ b₂).card ≤ 1) : 
  buses.card ≤ 12 := 
sorry

end max_buses_in_city_l22_22936


namespace maximum_buses_l22_22920

-- Definition of the problem constraints as conditions in Lean
variables (bus : Type) (stop : Type) (buses : set bus) (stops : set stop)
variables (stops_served_by_bus : bus → finset stop)
variables (stop_occurrences : stop → finset bus)

-- Conditions
def has_9_stops (stops : set stop) : Prop := stops.card = 9
def bus_serves_3_stops (b : bus) : Prop := (stops_served_by_bus b).card = 3
def at_most_1_common_stop (b1 b2 : bus) (h1 : b1 ∈ buses) (h2 : b2 ∈ buses) : Prop :=
  (stops_served_by_bus b1 ∩ stops_served_by_bus b2).card ≤ 1

-- Goal
theorem maximum_buses (h1 : has_9_stops stops)
                      (h2 : ∀ b ∈ buses, bus_serves_3_stops b)
                      (h3 : ∀ b1 b2 ∈ buses, at_most_1_common_stop b1 b2 h1 h2) : 
  buses.card ≤ 12 :=
sorry

end maximum_buses_l22_22920


namespace find_value_of_expression_l22_22606

-- Define non-negative variables
variables (x y z : ℝ) 

-- Conditions
def cond1 := x ^ 2 + x * y + y ^ 2 / 3 = 25
def cond2 := y ^ 2 / 3 + z ^ 2 = 9
def cond3 := z ^ 2 + z * x + x ^ 2 = 16

-- Target statement to be proven
theorem find_value_of_expression (h1 : cond1 x y) (h2 : cond2 y z) (h3 : cond3 z x) : 
  x * y + 2 * y * z + 3 * z * x = 24 * Real.sqrt 3 :=
sorry

end find_value_of_expression_l22_22606


namespace determine_a_l22_22886

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x - a) + a

theorem determine_a : (∃ a: ℝ, (∀ x: ℝ, -2 ≤ x ∧ x ≤ 3 → f x a ≤ 6) ∧ ∀ x: ℝ, f x a ≤ 6 → -2 ≤ x ∧ x ≤ 3) ↔ a = 1 :=
by
  sorry

end determine_a_l22_22886


namespace smallest_prime_square_mod_six_l22_22914

theorem smallest_prime_square_mod_six (p : ℕ) (h_prime : Nat.Prime p) (h_mod : p^2 % 6 = 1) : p = 5 :=
sorry

end smallest_prime_square_mod_six_l22_22914


namespace nested_radical_simplification_l22_22369

theorem nested_radical_simplification : 
  (sqrt (1 + 2 * sqrt (1 + 3 * sqrt (1 + ⋯ + 2017 * sqrt (1 + 2018 * 2020))))) = 3 :=
by
  -- sorry to skip the proof
  sorry

end nested_radical_simplification_l22_22369


namespace minimum_possible_product_l22_22680

def my_set : Set ℤ := {-9, -5, -3, 0, 4, 6, 8}

def product_of_three (a b c : ℤ) : ℤ := a * b * c

def distinct_elements (a b c : ℤ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem minimum_possible_product :
  ∃ a b c ∈ my_set, distinct_elements a b c ∧ product_of_three a b c = -432 :=
by
  sorry

end minimum_possible_product_l22_22680


namespace no_cracked_seashells_l22_22671

theorem no_cracked_seashells (tom_seashells : ℕ) (fred_seashells : ℕ) (total_seashells : ℕ)
  (h1 : tom_seashells = 15) (h2 : fred_seashells = 43) (h3 : total_seashells = 58)
  (h4 : tom_seashells + fred_seashells = total_seashells) : 
  (total_seashells - (tom_seashells + fred_seashells) = 0) :=
by
  sorry

end no_cracked_seashells_l22_22671


namespace weighted_average_correct_l22_22482

def total_problems : ℕ := 30
def math_problems : ℕ := 10
def science_problems : ℕ := 12
def history_problems : ℕ := 8

def math_points_each : ℝ := 2.5
def science_points_each : ℝ := 1.5
def history_points_each : ℝ := 3.0

def math_problems_completed : ℕ := 3
def science_problems_completed : ℕ := 2
def history_problems_completed : ℕ := 1

noncomputable def total_points_math : ℝ := math_problems * math_points_each
noncomputable def total_points_science : ℝ := science_problems * science_points_each
noncomputable def total_points_history : ℝ := history_problems * history_points_each

noncomputable def points_math_completed : ℝ := math_problems_completed * math_points_each
noncomputable def points_science_completed : ℝ := science_problems_completed * science_points_each
noncomputable def points_history_completed : ℝ := history_problems_completed * history_points_each

noncomputable def points_math_remaining : ℝ := total_points_math - points_math_completed
noncomputable def points_science_remaining : ℝ := total_points_science - points_science_completed
noncomputable def points_history_remaining : ℝ := total_points_history - points_history_completed

noncomputable def total_points_earned : ℝ := points_math_completed + points_science_completed + points_history_completed
noncomputable def total_points_remaining : ℝ := points_math_remaining + points_science_remaining + points_history_remaining

noncomputable def weighted_average : ℝ := total_points_remaining / total_points_earned

theorem weighted_average_correct :
  weighted_average ≈ 3.96 :=
sorry

end weighted_average_correct_l22_22482


namespace floor_difference_l22_22418

theorem floor_difference (x : ℝ) (h : x = 13.2) : 
  (⌊x^2⌋ - ⌊x⌋ * ⌊x⌋) = 5 := by
  have h1 : ⌊x⌋ = 13 := by sorry
  have h2 : ⌊x^2⌋ = 174 := by sorry
  calc
  ⌊x^2⌋ - ⌊x⌋ * ⌊x⌋ = 174 - 13 * 13 : by rw [h1, h2]
                 ... = 5 : by norm_num

end floor_difference_l22_22418


namespace floor_difference_l22_22411

theorem floor_difference (x : ℝ) (h : x = 13.2) : 
  (⌊x^2⌋ : ℤ) - ((⌊x⌋ : ℤ) * (⌊x⌋ : ℤ)) = 5 := 
by
  -- proof skipped
  sorry

end floor_difference_l22_22411


namespace series_sum_l22_22475

theorem series_sum : ∑ k in (None : Option ℕ), (λ k : ℕ, 2^(2^k) / (4^(2^k) - 1)) = 1 :=
sorry

end series_sum_l22_22475


namespace part1_solution_set_k_3_part2_solution_set_k_lt_0_l22_22841

open Set

-- Definitions
def inequality (k : ℝ) (x : ℝ) : Prop :=
  k * x^2 + (k - 2) * x - 2 < 0

-- Part 1: When k = 3
theorem part1_solution_set_k_3 : ∀ x : ℝ, inequality 3 x ↔ -1 < x ∧ x < (2 / 3) :=
by
  sorry

-- Part 2: When k < 0
theorem part2_solution_set_k_lt_0 :
  ∀ k : ℝ, k < 0 → 
    (k = -2 → ∀ x : ℝ, inequality k x ↔ x ≠ -1) ∧
    (k < -2 → ∀ x : ℝ, inequality k x ↔ x < -1 ∨ x > 2 / k) ∧
    (-2 < k → ∀ x : ℝ, inequality k x ↔ x > -1 ∨ x < 2 / k) :=
by
  sorry

end part1_solution_set_k_3_part2_solution_set_k_lt_0_l22_22841


namespace problem_solution_l22_22066

noncomputable def triangle_def_points : list (ℝ × ℝ) := [(-3, 6), (0, -3), (6, -3)]

noncomputable def point_P_coordinates (triangle_points : list (ℝ × ℝ)) (area_PQF : ℝ) : ℝ :=
  let D := (triangle_points.nth_le 0 (by simp))
      E := (triangle_points.nth_le 1 (by simp))
      F := (triangle_points.nth_le 2 (by simp))
      y := 2 * real.sqrt(10) in
  (6 - 2 * real.sqrt(10)) - (-3 + 2 * real.sqrt(10))

noncomputable def positive_difference_of_P_coordinates (triangle_points : list (ℝ × ℝ)) (area_PQF : ℝ) : ℝ :=
  (point_P_coordinates triangle_points area_PQF).abs 

theorem problem_solution :
  positive_difference_of_P_coordinates triangle_def_points 20 = 4 * real.sqrt(10) - 9 :=
by
  sorry

end problem_solution_l22_22066


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22199

theorem measure_of_one_interior_angle_of_regular_octagon 
  (n : Nat) (h_n : n = 8) : 
  (one_interior_angle : ℝ) = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22199


namespace part1_part2_part3_l22_22512

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1/2) * x^2 - a * Real.log x

-- Problem 1:
theorem part1 (a x : ℝ) (h: x = 2) (h_fx_extreme : deriv (λ x => (1/2) * x^2 - a * Real.log x) x = 0) :
  a = 4 :=
by
  sorry

-- Problem 2:
theorem part2 (a x : ℝ) :
  if a < 0 then Monotone (λ x => (1/2) * x^2 - a * Real.log x) 
  else if a = 0 then MonotoneOn (λ x => (1/2) * x^2) (Set.Ici 0) 
  else StrictAnti (λ x => (1/2) * x^2 - a * Real.log x) 0 (Real.sqrt a) ∧ 
       StrictMono (λ x => (1/2) * x^2 - a * Real.log x) (Real.sqrt a) :=
by
  sorry

-- Problem 3:
theorem part3 (x : ℝ) (h : x > 1) : 
  (1/2) * x^2 + Real.log x < (2/3) * x^3 := 
by
  sorry

end part1_part2_part3_l22_22512


namespace interior_angle_regular_octagon_l22_22269

theorem interior_angle_regular_octagon : 
  let n := 8 in 
  ∀ (sum_of_interior_angles : ℕ → ℕ) (regular_polygon_interior_angle : ℕ → ℕ),
  sum_of_interior_angles n = (180 * (n - 2)) →
  regular_polygon_interior_angle n = (sum_of_interior_angles n / n) →
  regular_polygon_interior_angle n = 135 :=
by
  intros n sum_of_interior_angles regular_polygon_interior_angle h1 h2
  sorry

end interior_angle_regular_octagon_l22_22269


namespace people_eating_vegetarian_l22_22285

theorem people_eating_vegetarian (only_veg : ℕ) (both_veg_nonveg : ℕ) (total_veg : ℕ) :
  only_veg = 13 ∧ both_veg_nonveg = 6 → total_veg = 19 := 
by
  sorry

end people_eating_vegetarian_l22_22285


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22196

theorem measure_of_one_interior_angle_of_regular_octagon 
  (n : Nat) (h_n : n = 8) : 
  (one_interior_angle : ℝ) = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22196


namespace semicircle_circumference_is_correct_l22_22649

noncomputable def pi_approx : Real := 3.14

def rectangle_perimeter (length breadth : ℝ) : ℝ := 2 * (length + breadth)

def square_side (perimeter : ℝ) : ℝ := perimeter / 4

def semicircle_circumference (diameter : ℝ) : ℝ :=
  (pi_approx * diameter) / 2 + diameter

theorem semicircle_circumference_is_correct :
  let length := 22
  let breadth := 16
  let p_rect := rectangle_perimeter length breadth
  let s := square_side p_rect
  let diameter := s
  let circumference := semicircle_circumference diameter
  circumference = 48.83 := by
  sorry

end semicircle_circumference_is_correct_l22_22649


namespace floor_difference_l22_22416

theorem floor_difference (x : ℝ) (h : x = 13.2) : 
  (⌊x^2⌋ - ⌊x⌋ * ⌊x⌋) = 5 := by
  have h1 : ⌊x⌋ = 13 := by sorry
  have h2 : ⌊x^2⌋ = 174 := by sorry
  calc
  ⌊x^2⌋ - ⌊x⌋ * ⌊x⌋ = 174 - 13 * 13 : by rw [h1, h2]
                 ... = 5 : by norm_num

end floor_difference_l22_22416


namespace find_lambda_l22_22848

-- Define the vectors
def a : ℝ × ℝ × ℝ := (2, -1, 3)
def b : ℝ × ℝ × ℝ := (-1, 4, -2)
def c (λ : ℝ) : ℝ × ℝ × ℝ := (7, 5, λ)

-- Define the coplanarity condition
def are_coplanar (a b : ℝ × ℝ × ℝ) (c : ℝ × ℝ × ℝ) : Prop :=
  ∃ (p q : ℝ), c = (p * (a.1), p * (a.2), p * (a.3)) + 
                (q * (b.1), q * (b.2), q * (b.3))

-- State the theorem
theorem find_lambda : 
  are_coplanar a b (c (65/7)) := sorry

end find_lambda_l22_22848


namespace interior_angle_of_regular_octagon_l22_22114

theorem interior_angle_of_regular_octagon :
  let n := 8 in
  let sum_interior_angles := 180 * (n - 2) in
  let angle := sum_interior_angles / n in
  angle = 135 :=
by
  sorry

end interior_angle_of_regular_octagon_l22_22114


namespace floor_diff_l22_22426

theorem floor_diff (x : ℝ) (h : x = 13.2) : 
  (Int.floor (x^2) - (Int.floor x) * (Int.floor x) = 5) := by
  sorry

end floor_diff_l22_22426


namespace sin_diff_value_l22_22485

theorem sin_diff_value (α : ℝ) (h : sin (α + π/6) - cos α = 1/2) : 
  sin (α - π/6) = 1/2 :=
sorry

end sin_diff_value_l22_22485


namespace minimum_value_OC_l22_22869

variables (OA OB OC : ℝ × ℝ × ℝ)
variables (θ : ℝ)

def magnitude (v : ℝ × ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2 + v.3^2)
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3

axiom condition1 : magnitude OA = 3
axiom condition2 : magnitude OB = 4
axiom condition3 : dot_product OA OB = 0
axiom condition4 : OC = (real.sin θ)^2 • OA + (real.cos θ)^2 • OB

theorem minimum_value_OC : 
  OC = ((16 : ℝ)/25) • OA + ((9 : ℝ)/25) • OB :=
sorry

end minimum_value_OC_l22_22869


namespace intersection_point_l22_22972

noncomputable def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 9*x + 15

theorem intersection_point :
  ∃ x : ℝ, f(x) = x ∧ (x = -1) :=
by
  sorry

end intersection_point_l22_22972


namespace regular_octagon_interior_angle_l22_22093

theorem regular_octagon_interior_angle (n : ℕ) (h₁ : n = 8) :
  (180 * (n - 2)) / n = 135 :=
by
  rw [h₁]
  sorry

end regular_octagon_interior_angle_l22_22093


namespace graph_cycles_l22_22029

/-- 
A graph with 2n vertices and n^2 + 1 edges necessarily contains a 3-cycle.
And a bipartite graph with 2n vertices and n^2 edges does not contain a 3-cycle.
-/
theorem graph_cycles (n : ℕ) :
  (∃ (G : SimpleGraph (Fin (2 * n))), G.edgeCount = n^2 + 1 ∧ G.contains_triangle) ∧
  (∃ (G : SimpleGraph (Fin (2 * n))), G.edgeCount = n^2 ∧ ¬G.contains_triangle) :=
sorry

end graph_cycles_l22_22029


namespace regular_octagon_interior_angle_l22_22128

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : 
  let sum_interior_angles := 180 * (n - 2)
  let num_sides := 8
  let one_interior_angle := sum_interior_angles / num_sides
  one_interior_angle = 135 :=
by
  sorry

end regular_octagon_interior_angle_l22_22128


namespace problem1_problem2_l22_22510

noncomputable def f : ℝ → ℝ
| x => if x > 0 then Real.logBase (1 / 3) x else 2^x

theorem problem1 :
  f (Real.logBase 2 (Real.sqrt 2 / 2)) + f (f 9) = (1 + 2 * Real.sqrt 2) / 4 :=
by {
  sorry
}

theorem problem2 (a : ℝ) :
  f (f a) ≤ 1 → (Real.logBase 2 (1 / 3) ≤ a ∧ a ≤ (1 / 3)^(1 / 3) ∨ 1 ≤ a) :=
by {
  sorry
}

end problem1_problem2_l22_22510


namespace interior_angle_regular_octagon_l22_22264

theorem interior_angle_regular_octagon : 
  let n := 8 in 
  ∀ (sum_of_interior_angles : ℕ → ℕ) (regular_polygon_interior_angle : ℕ → ℕ),
  sum_of_interior_angles n = (180 * (n - 2)) →
  regular_polygon_interior_angle n = (sum_of_interior_angles n / n) →
  regular_polygon_interior_angle n = 135 :=
by
  intros n sum_of_interior_angles regular_polygon_interior_angle h1 h2
  sorry

end interior_angle_regular_octagon_l22_22264


namespace parametric_eq_C1_correct_distance_AB_l22_22573

noncomputable def parametric_eq_C1 (α : ℝ) : ℝ × ℝ :=
(2 + Real.cos α, Real.sin α)

def curve_C1 (ρ θ : ℝ) : Prop :=
ρ^2 - 4 * ρ * Real.cos θ + 3 = 0 ∧ 0 ≤ θ ∧ θ ≤ 2 * Real.pi

def curve_C2 (ρ θ : ℝ) : Prop :=
ρ = 3 / (4 * Real.sin (Real.pi / 6 - θ)) ∧ 0 ≤ θ ∧ θ ≤ 2 * Real.pi

theorem parametric_eq_C1_correct :
  ∀ (ρ θ : ℝ), curve_C1 ρ θ →
  ∃ (α : ℝ), parametric_eq_C1 α = (2 + Real.cos (Real.atan2 ρ θ), Real.sin (Real.atan2 ρ θ)) :=
sorry

theorem distance_AB :
  ∀ (ρ1 θ1 ρ2 θ2 : ℝ), curve_C1 ρ1 θ1 ∧ curve_C2 ρ2 θ2 →
  abs (ρ1 * (Real.cos θ1) - ρ2 * (Real.cos θ2)) = 1/4 →
  abs (ρ1 * (Real.sin θ1) - ρ2 * (Real.sin θ2)) = Real.sqrt 15 / 2 :=
sorry

end parametric_eq_C1_correct_distance_AB_l22_22573


namespace part_I3_1_part_I3_2_part_I3_3_part_I3_4_l22_22033

-- Part I3.1: Prove a = -1
theorem part_I3_1 (θ : ℝ) : 
  let a := cos(θ)^4 - sin(θ)^4 - 2 * cos(θ)^2
  in a = -1 :=
by
  sorry

-- Part I3.2: Prove b = 17
theorem part_I3_2 (x y : ℝ) : 
  let a := -1
  let b := x^(3*y) + 10*a
  in x^y = 3 ∧ b = 17 :=
by
  sorry

-- Part I3.3: Prove c = 8
theorem part_I3_3 :
  let b := 17
  let f (n : ℕ) := (n + b) % (n - 7) = 0
  let candidates := { n | n > 7 ∧ f n }
  in candidates.card = 8 :=
by
  sorry

-- Part I3.4: Prove d = 18
theorem part_I3_4 :
  let c := 8
  let d := Σ k in Finset.range (c + 1), log 4 (2 ^ k : ℝ)
  in d = 18 :=
by
  sorry

end part_I3_1_part_I3_2_part_I3_3_part_I3_4_l22_22033


namespace max_months_to_build_l22_22036

theorem max_months_to_build (a b c x : ℝ) (h1 : 1/a + 1/b = 1/6)
                            (h2 : 1/a + 1/c = 1/5)
                            (h3 : 1/c + 1/b = 1/4)
                            (h4 : (1/a + 1/b + 1/c) * x = 1) :
                            x = 4 :=
sorry

end max_months_to_build_l22_22036


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22138

theorem measure_of_one_interior_angle_of_regular_octagon : 
  let n := 8 in 
  let sum_of_interior_angles := 180 * (n - 2) in
  let measure_of_one_interior_angle := sum_of_interior_angles / n in 
  measure_of_one_interior_angle = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22138


namespace max_lambda_inequality_l22_22835

theorem max_lambda_inequality (z : Fin 2016 → ℂ) (hz : z 2016 = z 0) :
  ∑ k in Finset.range 2016, ‖z k‖^2 ≥ 504 * (Finset.min' (Finset.range 2016) (by decide) (λ k, ‖z (k + 1) % 2016 - z k‖^2)) := 
sorry

end max_lambda_inequality_l22_22835


namespace circle_radius_l22_22919

open Real

theorem circle_radius (a : Real) (R : ℝ) :
  (∃ (ch1 ch2 ch3 : chord), 
    (length (ch1) = a) ∧ 
    (length (ch2) = a) ∧
    (length (ch3) = a) ∧
    (ch1 ∩ ch2 ∩ ch3).count = 3 ∧ 
    (∀ p ∈ ch1 ∩ ch2 ∩ ch3, distance (p, center) = R)
  ) →
  R = a * sqrt(3) / 3 :=
by
  sorry

end circle_radius_l22_22919


namespace hyperbola_s_squared_l22_22729

theorem hyperbola_s_squared 
  (s : ℝ) 
  (a b : ℝ) 
  (h1 : a = 3)
  (h2 : b^2 = 144 / 13) 
  (h3 : (2, s) ∈ {p : ℝ × ℝ | (p.2)^2 / a^2 - (p.1)^2 / b^2 = 1}) :
  s^2 = 441 / 36 :=
by sorry

end hyperbola_s_squared_l22_22729


namespace part_a_part_b_l22_22862

noncomputable theory
open Set

-- Define the segment AB and the conditions for set M
def segment_AB (A B : ℝ) : Set ℝ := {x | A ≤ x ∧ x ≤ B}

def set_M (A B : ℝ) : Set ℝ :=
  let M0 := {A, B}
  let rule (M : Set ℝ) : Set ℝ := M ∪ {z | ∃ (X Y ∈ M), z = X + (Y - X) / 4}
  ⋃ n, (rule^[n]) M0  -- repeatedly applying the rule n times

-- Part (a)
theorem part_a (A B : ℝ) (h_AB : abs (B - A) = 1) 
: ∀ X ∈ set_M A B, ∃ (n k : ℕ), AX = (3 * k) / (4^n) ∨ AX = (3 * k - 2) / (4^n) := 
sorry

-- Part (b)
theorem part_b (A B : ℝ) (h_AB : abs (B - A) = 1) 
: ¬ (1/2) ∈ set_M A B := 
sorry

end part_a_part_b_l22_22862


namespace find_m_of_ellipse_l22_22883

theorem find_m_of_ellipse (m : ℝ) : 
  (∃ x y : ℝ, (x^2 / (10 - m) + y^2 / (m - 2) = 1) ∧ (m - 2 > 10 - m) ∧ ((4)^2 = (m - 2) - (10 - m))) → m = 14 :=
by sorry

end find_m_of_ellipse_l22_22883


namespace most_profitable_investment_l22_22657

def inheritance : ℕ := 500000

def option1_profit (P : ℕ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r)^t - P

def option2_profit (P : ℕ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r * t) - P

def option3_profit (P : ℕ) (price_per_gram : ℝ) (increase_factor : ℝ) (future_price_per_gram : ℝ) : ℝ :=
  let grams := P / price_per_gram
  let future_value := grams * future_price_per_gram
  future_value - P

def option4_profit (P : ℕ) (price_per_gram : ℝ) (vat_markup : ℝ) (increase_factor : ℝ) (future_price_per_gram : ℝ) : ℝ :=
  let adjusted_price := price_per_gram * vat_markup
  let grams := P / adjusted_price
  let future_value := grams * future_price_per_gram
  future_value - P

def option5_profit (P : ℕ) (share_price : ℕ) (nominal_value : ℕ) (dividend_rate : ℝ) (increase_factor : ℝ) : ℝ :=
  let shares := P / share_price
  let dividends := shares * nominal_value * dividend_rate * 5
  let future_value := P * increase_factor
  let total_value := future_value + dividends
  total_value - P

theorem most_profitable_investment : 
  let profit1 := option1_profit inheritance 0.07 5
  let profit2 := option2_profit inheritance 0.08 5
  let profit3 := option3_profit inheritance 2350 1.6 3760
  let profit4 := option4_profit inheritance 31 1.29 1.7 52.7
  let profit5 := option5_profit inheritance 500 400 0.05 1.18
  profit1 = max profit1 (max profit2 (max profit3 (max profit4 profit5))) :=
by 
  let profit1 := option1_profit inheritance 0.07 5
  let profit2 := option2_profit inheritance 0.08 5
  let profit3 := option3_profit inheritance 2350 1.6 3760
  let profit4 := option4_profit inheritance 31 1.29 1.7 52.7
  let profit5 := option5_profit inheritance 500 400 0.05 1.18
  sorry

end most_profitable_investment_l22_22657


namespace max_buses_l22_22927

-- Define the total number of stops and buses as finite sets
def stops : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define a bus as a subset of stops with exactly 3 stops
structure Bus :=
(stops : Finset ℕ)
(h_stops : stops.card = 3)

-- Define the condition that any two buses share at most one stop
def shares_at_most_one (b1 b2 : Bus) : Prop :=
(b1.stops ∩ b2.stops).card ≤ 1

-- Define the predicate for a valid set of buses
def valid_buses (buses : Finset Bus) : Prop :=
∀ b1 b2 ∈ buses, b1 ≠ b2 → shares_at_most_one b1 b2

-- The main theorem statement
theorem max_buses (buses : Finset Bus) (h_valid : valid_buses buses) : buses.card ≤ 12 :=
sorry

end max_buses_l22_22927


namespace gravitational_force_at_25000_miles_l22_22051

-- Define the gravitational force as inversely proportional to the square of the distance
def gravitational_force (k : ℝ) (d : ℝ) : ℝ := k / (d * d)

theorem gravitational_force_at_25000_miles (k : ℝ) :
  (gravitational_force k 5000 = 500) → (gravitational_force k 25000 = 1 / 5) :=
by
  sorry

end gravitational_force_at_25000_miles_l22_22051


namespace max_buses_l22_22929

-- Define the total number of stops and buses as finite sets
def stops : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define a bus as a subset of stops with exactly 3 stops
structure Bus :=
(stops : Finset ℕ)
(h_stops : stops.card = 3)

-- Define the condition that any two buses share at most one stop
def shares_at_most_one (b1 b2 : Bus) : Prop :=
(b1.stops ∩ b2.stops).card ≤ 1

-- Define the predicate for a valid set of buses
def valid_buses (buses : Finset Bus) : Prop :=
∀ b1 b2 ∈ buses, b1 ≠ b2 → shares_at_most_one b1 b2

-- The main theorem statement
theorem max_buses (buses : Finset Bus) (h_valid : valid_buses buses) : buses.card ≤ 12 :=
sorry

end max_buses_l22_22929


namespace floor_difference_l22_22408

theorem floor_difference (x : ℝ) (h : x = 13.2) : 
  (⌊x^2⌋ : ℤ) - ((⌊x⌋ : ℤ) * (⌊x⌋ : ℤ)) = 5 := 
by
  -- proof skipped
  sorry

end floor_difference_l22_22408


namespace nested_radical_value_l22_22372

theorem nested_radical_value : 
  let rec nest k :=
    if k = 1 then 2019 else sqrt (1 + k * (k + 2)) in
  nest 2018 = 3 :=
by
  intros
  sorry

end nested_radical_value_l22_22372


namespace range_of_a_l22_22057

theorem range_of_a {a : ℝ} (hp : ∀ {x : ℝ}, x^2 + (a - 1) * x + a^2 ≤ 0 → false)
  (hq : ∀ {x : ℝ}, x ∈ set.Icc (-2) 4 → |x| ≤ a → (x + 2) / 6 ≥ 5 / 6): 
  a ∈ set.Ioo (1/3 : ℝ) 3 :=
by
  sorry

end range_of_a_l22_22057


namespace parabola_ratio_l22_22860

theorem parabola_ratio (p : ℝ) (h : p > 0) :
  let F := (p, 0)
      O := (0, 0)
      AF := 8 * real.sqrt(ht^2 + (p-0)^2)
      BF := 12 * real.sqrt(ht^2 + (p-0)^2) in
  AF / BF = 2 / 3 :=
by
  let F := (p, 0)
  let O := (0, 0)
  let AF := 8 * real.sqrt(ht^2 + (p-0)^2)
  let BF := 12 * real.sqrt(ht^2 + (p-0)^2)
  unfold F O AF BF 
  have a := 8 * real.sqrt(ht^2 + (p-0)^2)
  have b := 12 * real.sqrt(ht^2 + (p-0)^2)
  rw [h]
  sorry

end parabola_ratio_l22_22860


namespace trader_overall_loss_l22_22547

noncomputable def selling_price : ℝ := 325475

def gain_percentage_first_car : ℝ := 0.15
def loss_percentage_second_car : ℝ := 0.15

def cost_price_first_car : ℝ := selling_price / (1 + gain_percentage_first_car)
def cost_price_second_car : ℝ := selling_price / (1 - loss_percentage_second_car)

def total_cost_price : ℝ := cost_price_first_car + cost_price_second_car
def total_selling_price : ℝ := 2 * selling_price

def overall_profit_or_loss : ℝ := total_selling_price - total_cost_price
def overall_loss_percent : ℝ := (overall_profit_or_loss / total_cost_price) * 100

theorem trader_overall_loss :
  overall_loss_percent = -2.33 := 
sorry

end trader_overall_loss_l22_22547


namespace area_of_one_petal_l22_22757

theorem area_of_one_petal (f : ℝ → ℝ) (h : ∀ θ, f θ = sin θ ^ 2) :
  (1 / 2) * ∫ θ in 0..π, (f θ) ^ 2 = 3 * π / 16 :=
by
  sorry

end area_of_one_petal_l22_22757


namespace least_not_lucky_multiple_of_6_l22_22309

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

theorem least_not_lucky_multiple_of_6 : ∃ k : ℕ, k > 0 ∧ k % 6 = 0 ∧ ¬ is_lucky k ∧ ∀ m : ℕ, m > 0 ∧ m % 6 = 0 ∧ ¬ is_lucky m → k ≤ m :=
  sorry

end least_not_lucky_multiple_of_6_l22_22309


namespace person_income_l22_22641

/-- If the income and expenditure of a person are in the ratio 15:8 and the savings are Rs. 7000, then the income of the person is Rs. 15000. -/
theorem person_income (x : ℝ) (income expenditure : ℝ) (savings : ℝ) 
  (h1 : income = 15 * x) 
  (h2 : expenditure = 8 * x) 
  (h3 : savings = income - expenditure) 
  (h4 : savings = 7000) : 
  income = 15000 := 
by 
  sorry

end person_income_l22_22641


namespace intersection_M_N_l22_22523

def M : Set ℝ := {y | y > 1}
def N : Set ℝ := {x | 0 < x ∧ x < 2}

theorem intersection_M_N : M ∩ N = set.Ioo (1:ℝ) 2 :=
sorry

end intersection_M_N_l22_22523


namespace ellipse_proof_chord_proof_l22_22863

noncomputable def ellipse_eq : Prop := 
  ∃ a b c : ℝ, 
    (a > b ∧ b > 0) ∧ 
    (∃ x y : ℝ, (x = c ∧ y = 0) ∧ (real.sqrt ((abs (c + 3 * real.sqrt 2) ^ 2) / 2) = 5)) ∧ 
    (real.sqrt (a^2 + b^2) = real.sqrt 10) ∧ 
    (a^2 = b^2 + c^2) ∧ 
    (a = 3 ∧ b = 1) ∧ 
    (∀ x y : ℝ, ((x^2) / (3^2) + y^2 = 1))

noncomputable def chord_constant : Prop := 
  ∃ Q : ℝ × ℝ, 
    (Q = (6 * real.sqrt 5 / 6, 0)) ∧ 
    (∀ (A B : ℝ), ∀ (m : ℝ), (x = m * y + 6 / real.sqrt 5) ∧ 
      (∃ x1 y1 x2 y2 : ℝ, 
        (∀ m : ℝ, 
          (y1 + y2 = - (12 * m) / (real.sqrt 5 * (m^2 + 9))) ∧ 
          (y1 * y2 = - (9 / 5) * (1 / ((m^2 + 9) ^ 2))) ∧ 
          (1/((x1 - 6 / real.sqrt 5) ^ 2 + y1^2) = 1 / (m^2 + 1) * (1 / y1^2)) ∧ 
          (1/((x2 - 6 / real.sqrt 5) ^ 2 + y2^2) = 1 / (m^2 + 1) * (1 / y2^2)))) ∘ 
    (∀ m x1 x2 y1 y2, 
      ( (1 / (m^2 + 1) * 1 / y1^2) + (1 / (m^2 + 1) * 1 / y2^2) = 10))

theorem ellipse_proof : ellipse_eq :=
sorry

theorem chord_proof : chord_constant :=
sorry

end ellipse_proof_chord_proof_l22_22863


namespace g_is_odd_l22_22781

def g (x : ℝ) : ℝ := (7 ^ x - 1) / (7 ^ x + 1)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  sorry

end g_is_odd_l22_22781


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22145

theorem measure_of_one_interior_angle_of_regular_octagon : 
  let n := 8 in 
  let sum_of_interior_angles := 180 * (n - 2) in
  let measure_of_one_interior_angle := sum_of_interior_angles / n in 
  measure_of_one_interior_angle = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22145


namespace q_q_neg1_1_q_2_neg2_eq_zero_l22_22591

def q (x y : ℝ) : ℝ :=
  if x ≥ 0 ∧ y > 0 then
    x^2 + 2 * y^2
  else if x < 0 ∧ y ≤ 0 then
    x^2 - 3 * y
  else
    2 * x + 2 * y

theorem q_q_neg1_1_q_2_neg2_eq_zero : q (q (-1) 1) (q 2 (-2)) = 0 := by
  sorry

end q_q_neg1_1_q_2_neg2_eq_zero_l22_22591


namespace sum_edges_gt_3d_l22_22611

-- Defining a polyhedron as a set of vertices and edges
structure Polyhedron where
  vertices : Set Point
  edges    : Set (Point × Point)
  edge_lengths : Point × Point → ℝ
  (condition_edges_length : ∀ e ∈ edges, edge_lengths e = euclidean_distance (fst e) (snd e))

-- Defining Euclidean distance function
def euclidean_distance (p1 p2 : Point) : ℝ := sorry

-- Point as a basic structure to simplify the definition
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- The distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ := euclidean_distance p1 p2

-- Main theorem to prove
theorem sum_edges_gt_3d (P : Polyhedron) (A B : Point) (h₁ : A ∈ P.vertices) (h₂ : B ∈ P.vertices) (h₃ : ∀ C D ∈ P.vertices, distance C D ≤ distance A B) :
  (∑ e in P.edges, P.edge_lengths e) > 3 * distance A B := 
by
  sorry

end sum_edges_gt_3d_l22_22611


namespace interior_angle_regular_octagon_l22_22084

theorem interior_angle_regular_octagon (n : ℕ) (h : n = 8) : 
  let sum_of_interior_angles := 180 * (n - 2)
  let number_of_sides := n
  let one_interior_angle := sum_of_interior_angles / number_of_sides
  in one_interior_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l22_22084


namespace sum_of_squares_of_solutions_l22_22828

theorem sum_of_squares_of_solutions :
  (∑ x in {x : ℝ | | x^2 - x + 1/2010 | = 1/2010}, x^2) = 2008 / 1005 :=
by
  sorry

end sum_of_squares_of_solutions_l22_22828


namespace correct_divisor_l22_22945

noncomputable def dividend := 12 * 35

theorem correct_divisor (x : ℕ) : (x * 20 = dividend) → x = 21 :=
sorry

end correct_divisor_l22_22945


namespace min_value_expression_l22_22483

theorem min_value_expression (x : ℝ) (hx : x > 4) : 
  ∃ y : ℝ, (y = 2 * Real.sqrt 14) ∧ (∀ z : ℝ, (z = (x + 10) / Real.sqrt (x - 4)) → y ≤ z) := sorry

end min_value_expression_l22_22483


namespace problem_solution_l22_22695

noncomputable def expression_value : ℝ :=
  ((12.983 * 26) / 200) ^ 3 * Real.log 5 / Real.log 10

theorem problem_solution : expression_value = 3.361 := by
  sorry

end problem_solution_l22_22695


namespace frog_jump_a_frog_jump_b_frog_probability_alive_c_frog_average_lifespan_d_l22_22727
-- The following import covers necessary libraries.

-- Part (a) statement
theorem frog_jump_a (n : ℕ) (h : n % 2 = 0) : 
  number_of_ways_from_A_to_C (n : ℕ) = (1 / 3) * (4^((n/2)) - 1) :=
sorry

-- Part (b) statement
theorem frog_jump_b (n : ℕ) (h : n % 2 = 0) :
  number_of_ways_from_A_to_C_without_jumping_to_D (n: ℕ) = 3^((n/2)-1) :=
sorry

-- Part (c) statement
theorem frog_probability_alive_c (n : ℕ) (h : n % 2 = 1) :
  probability_frog_alive (n : ℕ) = (3 / 4)^((n - 1) / 2) :=
sorry

-- Part (d) statement
theorem frog_average_lifespan_d : 
  average_lifespan = 9 :=
sorry

end frog_jump_a_frog_jump_b_frog_probability_alive_c_frog_average_lifespan_d_l22_22727


namespace largest_prime_factor_sum_divisors_180_is_7_l22_22987

-- Definitions based on conditions in a)
def prime_factors_180 : list (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1)]

def sum_of_divisors (n : ℕ) : ℕ := 
  let factors := prime_factors_180
  in factors.foldl
       (λ acc pf, match pf with
                  | (p, a) => acc * (list.range (a + 1)).map (λ i, p ^ i).sum
                  end)
       1

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  (nat.factors n).last sorry

-- The theorem we want to prove
theorem largest_prime_factor_sum_divisors_180_is_7 :
  largest_prime_factor (sum_of_divisors 180) = 7 := sorry

end largest_prime_factor_sum_divisors_180_is_7_l22_22987


namespace ratio_week3_to_first_two_weeks_l22_22971

-- Definitions based on the conditions
def week1 : ℕ := 12
def week2 : ℕ := 16
def week3 : ℕ := 16
def week4 : ℕ := week3 - 3
def total_pairs : ℕ := week1 + week2 + week3 + week4

-- Statement to be proven
theorem ratio_week3_to_first_two_weeks : 
  week1 + week2 = 28 →
  total_pairs = 57 →
  week3 = 16 →
  week4 = week3 - 3 →
  (week3 : ℚ) / (week1 + week2 : ℚ) = 4 / 7 :=
by
  intros h1 h2 h3 h4
  calc
    (week3 : ℚ) / (week1 + week2 : ℚ) = 16 / 28 : by rw [h1, h3]
    ... = 4 / 7 : sorry

end ratio_week3_to_first_two_weeks_l22_22971


namespace color_of_75th_bead_l22_22968

noncomputable def bead_sequence := ["red", "orange", "yellow", "yellow", "green", "green", "blue"]

def nth_bead (n : Nat) : String :=
  let index := (n - 1) % 7
  bead_sequence.getD index "error"

theorem color_of_75th_bead : nth_bead 75 = "green" :=
by
  -- This will utilize the defined sequence and index calculation
  sorry

end color_of_75th_bead_l22_22968


namespace number_not_round_to_72_36_l22_22281

def round_to_hundredth (x : ℝ) : ℝ :=
  (Real.floor (x * 100 + 0.5)) / 100

theorem number_not_round_to_72_36 :
  ¬ (round_to_hundredth 72.3539999 = 72.36) ∧ 
  (round_to_hundredth 72.361 = 72.36) ∧
  (round_to_hundredth 72.358 = 72.36) ∧
  (round_to_hundredth 72.3601 = 72.36) ∧
  (round_to_hundredth 72.35999 = 72.36) :=
by
  sorry

end number_not_round_to_72_36_l22_22281


namespace increasing_function_range_l22_22884

theorem increasing_function_range (a : ℝ) (h : ∀ x : ℝ, deriv (fun x => exp (2 * x) - a * exp x + 2 * x) x ≥ 0) : 
  a ≤ 4 :=
sorry

end increasing_function_range_l22_22884


namespace nested_radicals_equivalent_l22_22363

theorem nested_radicals_equivalent :
  (∃ (f : ℕ → ℕ),
    (∀ k, 1 ≤ k → f k = (k + 1)^2 - 1) ∧
    (sqrt (1 + 2 * sqrt (1 + 3 * sqrt (1 + 2017 * sqrt (1 + 2018 * (f 2018))))) = 3)) :=
begin
  use (λ k, (k + 1)^2 - 1),
  split,
  { intros k hk,
    exact rfl, },
  { sorry },
end

end nested_radicals_equivalent_l22_22363


namespace largest_sum_of_squares_l22_22045

theorem largest_sum_of_squares (x y : ℕ) (h : x^2 - y^2 = 221) : x^2 + y^2 ≤ 24421 :=
begin
  sorry,
end

example : ∃ x y : ℕ, x^2 - y^2 = 221 ∧ x^2 + y^2 = 24421 :=
begin
  use [111, 110],
  split,
  { linarith, },
  { linarith, }
end

end largest_sum_of_squares_l22_22045


namespace mow_lawn_time_l22_22617

noncomputable def time_to_mow (length width swath_width overlap speed : ℝ) : ℝ :=
  let effective_swath := (swath_width - overlap) / 12 -- Convert inches to feet
  let strips_needed := width / effective_swath
  let total_distance := strips_needed * length
  total_distance / speed

theorem mow_lawn_time : time_to_mow 100 140 30 6 4500 = 1.6 :=
by
  sorry

end mow_lawn_time_l22_22617


namespace parabola_and_hyperbola_equations_l22_22872

-- Definitions: Conditions from a)
def common_focus_on_x_axis : Prop :=
  ∃ p : ℝ, p > 0 ∧ (λ x y : ℝ, y^2 = 2 * p * x) ∧ (λ x y : ℝ, y = 2 * x ∨ y = -2 * x)

def asymptote_intersects_at_point : Prop :=
  ∃ P : ℝ × ℝ, P = (4, 8)

-- Prove that:
theorem parabola_and_hyperbola_equations :
  common_focus_on_x_axis → asymptote_intersects_at_point →
  (∃ p : ℝ, p = 8 ∧ ∀ x y : ℝ, y^2 = 16 * x) ∧ 
  (∃ λ : ℝ, λ = 16 / 5 ∧ ∀ x y : ℝ, (5 * x^2) / 16 - (5 * y^2) / 64 = 1) :=
by
  intros
  sorry

end parabola_and_hyperbola_equations_l22_22872


namespace number_of_correct_statements_is_2_l22_22690

-- Define each statement as a boolean
def statement1 : Prop := ∀ (A B C : Point), ¬ collinear A B C → exists_circle A B C
def statement2 : Prop := ∀ (O A B : Point), diameter_perpendicular_bisects_chord O A B
def statement3 : Prop := ∀ (O A B C D : Point), subtend_equal_central_angles_equals O A B C D
def statement4 : Prop := ∀ (O A B C : Point), distance_from_circumcenter_is_equal O A B C

-- Sum the number of true statements
def correct_statements_count : ℕ :=
if statement1 then 1 else 0 +
if statement2 then 1 else 0 +
if statement3 then 1 else 0 +
if statement4 then 1 else 0

-- State the theorem to prove there are 2 correct statements
theorem number_of_correct_statements_is_2 :
  correct_statements_count = 2 :=
by {
  -- State the condition of each statement correctness based on the given problem.
  have statement1_incorrect : ¬ statement1,
  sorry,
  have statement2_correct : statement2,
  sorry,
  have statement3_incorrect : ¬ statement3,
  sorry,
  have statement4_correct : statement4,
  sorry,
  -- Use these conditions to establish the number of correct statements.
  unfold correct_statements_count,
  split_ifs,
  -- Here we expect the correct answer derived from the conditions above.
  exact dec_trivial,
}

end number_of_correct_statements_is_2_l22_22690


namespace floor_diff_l22_22430

theorem floor_diff (x : ℝ) (h : x = 13.2) : 
  (Int.floor (x^2) - (Int.floor x) * (Int.floor x) = 5) := by
  sorry

end floor_diff_l22_22430


namespace find_a1_l22_22873

theorem find_a1 
  (a_n : ℕ → ℝ)
  (S_n : ℕ → ℝ)
  (h_pos : ∀ n, 0 < a_n n)
  (h_S6 : S_n 6 = 3 / 8)
  (h_log_seq : ∀ n, log 2 (a_n n) = log 2 (a_n 1) - n + 1)
  (h_sum_Sn : ∀ n, S_n n = ∑ i in finset.range (n+1), a_n i) :
  a_n 1 = 4 / 21 :=
by
  sorry

end find_a1_l22_22873


namespace sum_squares_of_solutions_eq_l22_22818

noncomputable def sum_of_squares_of_solutions : ℚ := sorry

theorem sum_squares_of_solutions_eq :
  (∃ x : ℚ, abs (x^2 - x + (1 : ℚ) / 2010) = (1 : ℚ) / 2010) →
  sum_of_squares_of_solutions = (2008 : ℚ) / 1005 :=
sorry

end sum_squares_of_solutions_eq_l22_22818


namespace interior_angle_regular_octagon_l22_22273

theorem interior_angle_regular_octagon : 
  let n := 8 in 
  ∀ (sum_of_interior_angles : ℕ → ℕ) (regular_polygon_interior_angle : ℕ → ℕ),
  sum_of_interior_angles n = (180 * (n - 2)) →
  regular_polygon_interior_angle n = (sum_of_interior_angles n / n) →
  regular_polygon_interior_angle n = 135 :=
by
  intros n sum_of_interior_angles regular_polygon_interior_angle h1 h2
  sorry

end interior_angle_regular_octagon_l22_22273


namespace find_w_over_x_l22_22881

variables (w x y : ℚ)

-- The conditions
def condition1 : Prop := ∃ k : ℚ, w = k * x
def condition2 : Prop := w / y = 2 / 3
def condition3 : Prop := (x + y) / y = 3

-- The theorem we need to prove
theorem find_w_over_x (h1 : condition1) (h2 : condition2) (h3 : condition3) : w / x = 1 / 3 :=
sorry

end find_w_over_x_l22_22881


namespace simplify_complex_fraction_l22_22620

theorem simplify_complex_fraction :
  ( (4 + 7 * complex.i) / (4 - 7 * complex.i) ) - ( (4 - 7 * complex.i) / (4 + 7 * complex.i) ) = (112 * complex.i) / 65 :=
by
  sorry

end simplify_complex_fraction_l22_22620


namespace part1_part2_l22_22888

-- Define given conditions
def f (x a b : ℝ) := 2 * Real.sqrt (x^2 + 2 * a * x + a^2) - 2 * abs (x - b)

-- a, b > 0 and max value of f(x) = 2
variables (a b : ℝ)
hypothesis ha : a > 0
hypothesis hb : b > 0
hypothesis hf_max : ∃ x, f x a b = 2

-- Prove a + b = 1
theorem part1 : a + b = 1 :=
by
  sorry

-- Prove inequality
theorem part2 : 1 / a + 4 / b + 4 / ((3 * a + 1) * b) ≥ 12 :=
by
  sorry

end part1_part2_l22_22888


namespace total_money_from_tshirts_l22_22038

def num_tshirts_sold := 20
def money_per_tshirt := 215

theorem total_money_from_tshirts :
  num_tshirts_sold * money_per_tshirt = 4300 :=
by
  sorry

end total_money_from_tshirts_l22_22038


namespace regular_octagon_angle_measure_l22_22210

theorem regular_octagon_angle_measure :
  let n := 8 in
  (180 * (n - 2)) / n = 135 := 
by
  sorry

end regular_octagon_angle_measure_l22_22210


namespace regular_octagon_interior_angle_measure_l22_22257

theorem regular_octagon_interior_angle_measure :
  ∀ (n : ℕ), n = 8 → (∑ i in finset.range n, (180 : ℝ)) / n = 135 :=
  by
  sorry

end regular_octagon_interior_angle_measure_l22_22257


namespace nested_radical_simplification_l22_22367

theorem nested_radical_simplification : 
  (sqrt (1 + 2 * sqrt (1 + 3 * sqrt (1 + ⋯ + 2017 * sqrt (1 + 2018 * 2020))))) = 3 :=
by
  -- sorry to skip the proof
  sorry

end nested_radical_simplification_l22_22367


namespace newspaper_price_l22_22530

-- Define the conditions as variables
variables 
  (P : ℝ)                    -- Price per edition for Wednesday, Thursday, and Friday
  (total_cost : ℝ := 28)     -- Total cost over 8 weeks
  (sunday_cost : ℝ := 2)     -- Cost of Sunday edition
  (weeks : ℕ := 8)           -- Number of weeks
  (wednesday_thursday_friday_editions : ℕ := 3 * weeks) -- Total number of editions for Wednesday, Thursday, and Friday over 8 weeks

-- Math proof problem statement
theorem newspaper_price : 
  (total_cost - weeks * sunday_cost) / wednesday_thursday_friday_editions = 0.5 :=
  sorry

end newspaper_price_l22_22530


namespace max_value_expression_exist_x_y_z_l22_22804

theorem max_value_expression (x y z : ℝ) : 
  (sin (3 * x) + sin (2 * y) + sin z) * (cos (3 * x) + cos (2 * y) + cos z) ≤ 9 / 2 :=
sorry

theorem exist_x_y_z (x y z : ℝ) :
  ∃ x y z : ℝ, (sin (3 * x) + sin (2 * y) + sin z) * (cos (3 * x) + cos (2 * y) + cos z) = 9 / 2 :=
sorry

end max_value_expression_exist_x_y_z_l22_22804


namespace spinner_prob_is_one_third_l22_22399

noncomputable def spinner_prob_div_by_5 : ℚ :=
let outcomes := {1, 3, 5} in
let total_outcomes := 3 * 3 * 3 in
let valid_outcomes := 3 * 3 in
(valid_outcomes : ℚ) / total_outcomes

theorem spinner_prob_is_one_third :
  spinner_prob_div_by_5 = 1 / 3 :=
sorry

end spinner_prob_is_one_third_l22_22399


namespace regular_octagon_interior_angle_l22_22131

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : 
  let sum_interior_angles := 180 * (n - 2)
  let num_sides := 8
  let one_interior_angle := sum_interior_angles / num_sides
  one_interior_angle = 135 :=
by
  sorry

end regular_octagon_interior_angle_l22_22131


namespace num_values_n_T_n_T_T_n_eq_2187_l22_22836

def T (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem num_values_n_T_n_T_T_n_eq_2187 :
  (finset.filter (λ n, n + T(n) + T(T(n)) = 2187) (finset.Icc 1 2187)).card = 2 := by
  sorry

end num_values_n_T_n_T_T_n_eq_2187_l22_22836


namespace least_lucky_multiple_of_six_not_lucky_l22_22312

def digit_sum (n : ℕ) : ℕ :=
  (n.to_digits 10).sum

def is_lucky_integer (n : ℕ) : Prop :=
  n % digit_sum n = 0

def is_multiple_of_six (n : ℕ) : Prop :=
  n % 6 = 0

theorem least_lucky_multiple_of_six_not_lucky : ∃ n, is_multiple_of_six n ∧ ¬ is_lucky_integer n ∧ (∀ m, is_multiple_of_six m ∧ ¬ is_lucky_integer m → n ≤ m) :=
by {
  use 12,
  split,
  { sorry },  -- Proof that 12 is a multiple of 6
  split,
  { sorry },  -- Proof that 12 is not a lucky integer
  { sorry },  -- Proof that there is no smaller multiple of 6 that is not a lucky integer
}

end least_lucky_multiple_of_six_not_lucky_l22_22312


namespace floor_difference_l22_22417

theorem floor_difference (x : ℝ) (h : x = 13.2) : 
  (⌊x^2⌋ - ⌊x⌋ * ⌊x⌋) = 5 := by
  have h1 : ⌊x⌋ = 13 := by sorry
  have h2 : ⌊x^2⌋ = 174 := by sorry
  calc
  ⌊x^2⌋ - ⌊x⌋ * ⌊x⌋ = 174 - 13 * 13 : by rw [h1, h2]
                 ... = 5 : by norm_num

end floor_difference_l22_22417


namespace sum_squares_of_solutions_eq_l22_22819

noncomputable def sum_of_squares_of_solutions : ℚ := sorry

theorem sum_squares_of_solutions_eq :
  (∃ x : ℚ, abs (x^2 - x + (1 : ℚ) / 2010) = (1 : ℚ) / 2010) →
  sum_of_squares_of_solutions = (2008 : ℚ) / 1005 :=
sorry

end sum_squares_of_solutions_eq_l22_22819


namespace probability_all_three_same_number_of_flips_l22_22947

noncomputable def probability_two_heads_in_n_flips (n : ℕ) : ℚ :=
  if h : n ≥ 2 then
    (nat.choose n 2) * (1/3:ℚ)^2 * (2/3:ℚ)^(n-2)
  else 0

noncomputable def total_probability : ℚ :=
  (∑' n, (probability_two_heads_in_n_flips n)^3)

theorem probability_all_three_same_number_of_flips :
  total_probability = ∑' n, (nat.choose n 2 * (1/3:ℚ)^2 * (2/3:ℚ)^(n-2))^3 := by
  sorry

end probability_all_three_same_number_of_flips_l22_22947


namespace integers_same_remainder_mod_100_l22_22854

theorem integers_same_remainder_mod_100
  (a : Fin 99 → ℤ)
  (h : ∀ (s : Finset (Fin 99)), (∑ i in s, a i) % 100 ≠ 0) :
  ∃ r : ℤ, ∀ i, a i % 100 = r :=
begin
  sorry
end

end integers_same_remainder_mod_100_l22_22854


namespace nested_radical_value_l22_22376

theorem nested_radical_value : 
  let rec nest k :=
    if k = 1 then 2019 else sqrt (1 + k * (k + 2)) in
  nest 2018 = 3 :=
by
  intros
  sorry

end nested_radical_value_l22_22376


namespace fraction_subtraction_l22_22786

def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem fraction_subtraction : 
  (18 / 42 - 2 / 9) = (13 / 63) := 
by 
  sorry

end fraction_subtraction_l22_22786


namespace team_score_is_correct_l22_22382

-- Definitions based on given conditions
def connor_score : ℕ := 2
def amy_score : ℕ := connor_score + 4
def jason_score : ℕ := 2 * amy_score
def combined_score : ℕ := connor_score + amy_score + jason_score
def emily_score : ℕ := 3 * combined_score
def team_score : ℕ := connor_score + amy_score + jason_score + emily_score

-- Theorem stating team_score should be 80
theorem team_score_is_correct : team_score = 80 := by
  sorry

end team_score_is_correct_l22_22382


namespace largest_prime_factor_of_sum_of_divisors_of_180_eq_13_l22_22984

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, d ∣ n).sum

def largest_prime_factor (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ p, Nat.Prime p ∧ p ∣ n).max' sorry

theorem largest_prime_factor_of_sum_of_divisors_of_180_eq_13 :
  largest_prime_factor (sum_of_divisors 180) = 13 :=
by
  sorry

end largest_prime_factor_of_sum_of_divisors_of_180_eq_13_l22_22984


namespace largest_prime_factor_of_sum_of_divisors_of_180_eq_13_l22_22982

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, d ∣ n).sum

def largest_prime_factor (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ p, Nat.Prime p ∧ p ∣ n).max' sorry

theorem largest_prime_factor_of_sum_of_divisors_of_180_eq_13 :
  largest_prime_factor (sum_of_divisors 180) = 13 :=
by
  sorry

end largest_prime_factor_of_sum_of_divisors_of_180_eq_13_l22_22982


namespace find_diagonal_of_rectangular_plate_l22_22058

noncomputable def diagonal_of_rectangular_plate 
  (C : ℝ) (H : ℝ) (D : ℝ) : Prop := 
  C = 6 → 
  H = 8 → 
  D = real.sqrt (6^2 + 8^2)

theorem find_diagonal_of_rectangular_plate 
  : diagonal_of_rectangular_plate 6 8 10 :=
by 
  intros,
  sorry

end find_diagonal_of_rectangular_plate_l22_22058


namespace sec_225_deg_l22_22459

theorem sec_225_deg : real.sec (225 * real.pi / 180) = -real.sqrt 2 :=
by
  -- Conditions
  have h1 : real.cos (225 * real.pi / 180) = -real.cos (45 * real.pi / 180),
  { rw [←real.cos_add_pi_div_two, real.cos_pi_div_two_sub],
    norm_num, },
  have h2 : real.cos (45 * real.pi / 180) = real.sqrt 2 / 2,
  { exact real.cos_pi_div_four, },
  rw [h1, h2, real.sec],
  field_simp,
  norm_num,
  sorry

end sec_225_deg_l22_22459


namespace Nell_cards_difference_l22_22007

-- Definitions
def initial_baseball_cards : ℕ := 438
def initial_ace_cards : ℕ := 18
def given_ace_cards : ℕ := 55
def given_baseball_cards : ℕ := 178

-- Theorem statement
theorem Nell_cards_difference :
  given_baseball_cards - given_ace_cards = 123 := 
by
  sorry

end Nell_cards_difference_l22_22007


namespace part_1_part_2_part_3_l22_22514

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x + m / x

theorem part_1 (h : f 1 m = 5) : m = 4 :=
sorry

theorem part_2 (m : ℝ) (h : m = 4) : ∀ x : ℝ, f (-x) m = -f x m :=
sorry

theorem part_3 (m : ℝ) (h : m = 4) : ∀ x1 x2 : ℝ, 2 < x1 → x1 < x2 → f x1 m < f x2 m :=
sorry

end part_1_part_2_part_3_l22_22514


namespace interior_angle_regular_octagon_l22_22089

theorem interior_angle_regular_octagon (n : ℕ) (h : n = 8) : 
  let sum_of_interior_angles := 180 * (n - 2)
  let number_of_sides := n
  let one_interior_angle := sum_of_interior_angles / number_of_sides
  in one_interior_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l22_22089


namespace buratino_cafe_workdays_l22_22635

-- Define the conditions as given in the problem statement
def days_in_april (d : Nat) : Prop := d >= 1 ∧ d <= 30
def is_monday (d : Nat) : Prop := d = 1 ∨ d = 8 ∨ d = 15 ∨ d = 22 ∨ d = 29

-- Define the period April 1 to April 13
def period_1_13 (d : Nat) : Prop := d >= 1 ∧ d <= 13

-- Define the statements made by Kolya
def kolya_statement_1 : Prop := ∀ d : Nat, days_in_april d → (d >= 1 ∧ d <= 20) → ¬is_monday d → ∃ n : Nat, n = 18
def kolya_statement_2 : Prop := ∀ d : Nat, days_in_april d → (d >= 10 ∧ d <= 30) → ¬is_monday d → ∃ n : Nat, n = 18

-- Define the condition stating Kolya made a mistake once
def kolya_made_mistake_once : Prop := kolya_statement_1 ∨ kolya_statement_2

-- The proof problem: Prove the number of working days from April 1 to April 13 is 11
theorem buratino_cafe_workdays : period_1_13 (d) → (¬is_monday d → (∃ n : Nat, n = 11)) := sorry

end buratino_cafe_workdays_l22_22635


namespace bowling_tournament_orders_l22_22955

theorem bowling_tournament_orders :
  let num_games := 5 in
  2 ^ num_games = 32 :=
by
  let num_games := 5
  show 2 ^ num_games = 32
  sorry

end bowling_tournament_orders_l22_22955


namespace find_smallest_k_l22_22772

noncomputable def sequence (n : ℕ) : ℝ :=
  ite (n = 0) 1
    (ite (n = 1) (3^(1/17))
      (sequence (n-1) * (sequence (n-2))^3))

def product_sequence (k : ℕ) : ℝ :=
  ∏ i in finset.range k, sequence (i + 1)

theorem find_smallest_k : ∃ k : ℕ, product_sequence k ∈ ℤ ∧ k = 3 :=
by
  -- the actual proof is skipped, just stating it here
  sorry

end find_smallest_k_l22_22772


namespace infinite_double_sum_equals_l22_22377

-- We define j and k as non-negative integers
def is_nonneg_int (n : ℕ) := n ≥ 0

-- The theorem states that the double infinite sum equals 4/3
theorem infinite_double_sum_equals :
  (∑ j in Nat.range (Nat.succ (Nat.max 0 0)), ∑ k in Nat.range (Nat.succ (Nat.max 0 0)), 2 ^ (- (4 * k + j + (k + j) ^ 2))) = 4 / 3 :=
by
  sorry

end infinite_double_sum_equals_l22_22377


namespace sum_of_squares_of_solutions_l22_22812

theorem sum_of_squares_of_solutions :
  (∑ α in {x | ∃ ε : ℝ, ε = x² - x + 1/2010 ∧ |ε| = 1/2010}, α^2) = 2008 / 1005 :=
by
  sorry

end sum_of_squares_of_solutions_l22_22812


namespace f_2a_eq_3_l22_22488

noncomputable def f (x : ℝ) : ℝ := 2^x + 1 / 2^x

theorem f_2a_eq_3 (a : ℝ) (h : f a = Real.sqrt 5) : f (2 * a) = 3 := by
  sorry

end f_2a_eq_3_l22_22488


namespace ratio_percent_is_33_point_33_l22_22303

def part1 : ℕ := 25
def part2 : ℕ := 50
def total : ℕ := part1 + part2
def ratio_as_percent : ℚ := (part1 : ℚ) / (total : ℚ) * 100

theorem ratio_percent_is_33_point_33 : ratio_as_percent ≈ 33.33 :=
by 
  -- calculation proof will be filled here
  sorry

end ratio_percent_is_33_point_33_l22_22303


namespace range_of_a_l22_22555

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 - a * x - a ≤ -3) ↔ a ∈ (-∞ : set ℝ) ∪ {a | a ≤ -6} ∪ {a | a ≥ 2} := 
by
  sorry

end range_of_a_l22_22555


namespace mikhailov_family_expenses_l22_22703

/- Conditions -/
def utility_payments : ℕ := 5250
def food_purchases : ℕ := 15000
def household_chemicals : ℕ := 3000
def clothing_footwear : ℕ := 15000
def car_loan : ℕ := 10000
def transport_costs : ℕ := 2000
def summer_trip_savings : ℕ := 5000
def medical_services : ℕ := 1500
def phone_internet : ℕ := 2000
def other_expenses : ℕ := 3000

def monthly_expenses : ℕ := 
  utility_payments + food_purchases + household_chemicals + clothing_footwear +
  car_loan + transport_costs + summer_trip_savings + medical_services + 
  phone_internet + other_expenses

/- Definitions -/
def monthly_income : ℕ := 65000
def monthly_savings : ℕ := 3250
def total_savings_10_months : ℕ := 82500

/- Theorem stating the problem -/
theorem mikhailov_family_expenses :
  monthly_expenses = 61750 ∧ 
  (monthly_expenses / 0.95).nat_abs = monthly_income ∧
  (0.05 * monthly_income).nat_abs = monthly_savings ∧ 
  (monthly_savings * 10 + summer_trip_savings * 10) = total_savings_10_months :=
by
  sorry

end mikhailov_family_expenses_l22_22703


namespace sec_225_eq_neg_sqrt_2_l22_22450

theorem sec_225_eq_neg_sqrt_2 :
  let sec (x : ℝ) := 1 / Real.cos x
  ∧ Real.cos (225 * Real.pi / 180) = Real.cos (180 * Real.pi / 180 + 45 * Real.pi / 180)
  ∧ Real.cos (180 * Real.pi / 180 + 45 * Real.pi / 180) = -Real.cos (45 * Real.pi / 180)
  ∧ Real.cos (45 * Real.pi / 180) = 1 / Real.sqrt 2
  → sec (225 * Real.pi / 180) = -Real.sqrt 2 :=
by 
  intros sec_h cos_h1 cos_h2 cos_h3
  sorry

end sec_225_eq_neg_sqrt_2_l22_22450


namespace find_divisor_l22_22638

def divisor (D Q R : ℕ) (hD : D = 162) (hQ : Q = 9) (hR : R = 9) : ℕ :=
if h : D = (Q * 17) + R then 17 else 0

theorem find_divisor (D Q R : ℕ) (hD : D = 162) (hQ : Q = 9) (hR : R = 9) :
  divisor D Q R hD hQ hR = 17 :=
by
  simp [divisor, hD, hQ, hR]
  sorry

end find_divisor_l22_22638


namespace regular_octagon_interior_angle_l22_22237

theorem regular_octagon_interior_angle : 
  ∀ n : ℕ, n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  intros n hn
  simp [sum_of_interior_angles]
  sorry

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
  180 * (n - 2)

end regular_octagon_interior_angle_l22_22237


namespace interval_of_monotonic_increase_l22_22778

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x - 12

theorem interval_of_monotonic_increase :
  ∀ x ∈ Set.Icc (-5 : ℝ) 5, (∃ y ∈ Set.Icc 2 5, y = x) → 
  (monotone_incr_on := ∀ x₁ x₂ ∈ Set.Icc 2 5, x₁ ≤ x₂ → f x₁ ≤ f x₂) :=
sorry

end interval_of_monotonic_increase_l22_22778


namespace town_population_l22_22529

theorem town_population (females males attending total_population : ℕ)
    (h1 : females = 50)
    (h2 : males = 2 * females)
    (h3 : attending = females + males)
    (h4 : total_population = 2 * attending) : 
    total_population = 300 :=
by
    have h_females := h1, -- females = 50
    have h_males := h2, -- males = 2 * females
    have h_attending := by rw [←h_females, ←h_males, nat.mul_comm] at h3; exact h3, -- attending = females + males
    have h_population := by rw [←h_attending, nat.mul_comm] at h4; exact h4, -- total_population = 2 * attending
    rw [h_population] -- concluding total_population = 300
    exact rfl

end town_population_l22_22529


namespace largest_prime_factor_sum_divisors_180_is_7_l22_22985

-- Definitions based on conditions in a)
def prime_factors_180 : list (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1)]

def sum_of_divisors (n : ℕ) : ℕ := 
  let factors := prime_factors_180
  in factors.foldl
       (λ acc pf, match pf with
                  | (p, a) => acc * (list.range (a + 1)).map (λ i, p ^ i).sum
                  end)
       1

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  (nat.factors n).last sorry

-- The theorem we want to prove
theorem largest_prime_factor_sum_divisors_180_is_7 :
  largest_prime_factor (sum_of_divisors 180) = 7 := sorry

end largest_prime_factor_sum_divisors_180_is_7_l22_22985


namespace regular_octagon_interior_angle_l22_22184

theorem regular_octagon_interior_angle : 
  ∀ (n : ℕ), n = 8 → (sum_of_internal_angles n) / n = 135 := 
by
  intros n hn
  rw hn
  sorry

end regular_octagon_interior_angle_l22_22184


namespace birdseed_needed_weekly_birdseed_needed_l22_22019

def parakeet_daily_consumption := 2
def parrot_daily_consumption := 14
def finch_daily_consumption := parakeet_daily_consumption / 2
def num_parakeets := 3
def num_parrots := 2
def num_finches := 4
def days_in_week := 7

theorem birdseed_needed :
  num_parakeets * parakeet_daily_consumption +
  num_parrots * parrot_daily_consumption +
  num_finches * finch_daily_consumption = 38 :=
by
  sorry

theorem weekly_birdseed_needed :
  38 * days_in_week = 266 :=
by
  sorry

end birdseed_needed_weekly_birdseed_needed_l22_22019


namespace count_perfect_cubes_l22_22535

theorem count_perfect_cubes (a b : ℕ) (h₁ : 200 < a) (h₂ : a < 1500) (h₃ : b = 6^3) :
  (∃! n : ℕ, 200 < n^3 ∧ n^3 < 1500) :=
sorry

end count_perfect_cubes_l22_22535


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22200

theorem measure_of_one_interior_angle_of_regular_octagon 
  (n : Nat) (h_n : n = 8) : 
  (one_interior_angle : ℝ) = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22200


namespace unique_solution_zmod_11_l22_22392

theorem unique_solution_zmod_11 : 
  ∀ (n : ℕ), 
  (2 ≤ n → 
  (∀ x : ZMod n, (x^2 - 3 * x + 5 = 0) → (∃! x : ZMod n, x^2 - (3 : ZMod n) * x + (5 : ZMod n) = 0)) → 
  n = 11) := 
by
  sorry

end unique_solution_zmod_11_l22_22392


namespace passes_through_origin_l22_22687

def parabola_A (x : ℝ) : ℝ := x^2 + 1
def parabola_B (x : ℝ) : ℝ := (x + 1)^2
def parabola_C (x : ℝ) : ℝ := x^2 + 2 * x
def parabola_D (x : ℝ) : ℝ := x^2 - x + 1

theorem passes_through_origin : 
  (parabola_A 0 ≠ 0) ∧
  (parabola_B 0 ≠ 0) ∧
  (parabola_C 0 = 0) ∧
  (parabola_D 0 ≠ 0) := 
by 
  sorry

end passes_through_origin_l22_22687


namespace Seryozha_healthy_eating_l22_22619

-- Define the total consumption of chocolate and sugar-free cookies
def total_chocolate : ℕ := 264
def total_sugar_free : ℕ := 187

-- Define the total number of days
def total_days : ℕ := 11

-- A function to calculate the sum of an arithmetic sequence
def arithmetic_sum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Given conditions rewritten in Lean statements
theorem Seryozha_healthy_eating :
  (∃ n : ℕ, n = total_days ∧
    arithmetic_sum n 0 1 = total_chocolate ∧
    arithmetic_sum 1 (n - 1) n = total_sugar_free) :=
begin
  use total_days,
  split,
  { refl },
  split,
  { have h1 : arithmetic_sum total_days 0 1 = total_chocolate := rfl,
    exact h1, },
  { have h2 : arithmetic_sum 1 (total_days - 1) total_days = total_sugar_free := rfl,
    exact h2, },
end

end Seryozha_healthy_eating_l22_22619


namespace alfred_gain_percent_l22_22697

noncomputable def gain_percent (purchase_price repair_costs selling_price : ℕ) :=
  ((selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs).toℝ) * 100

theorem alfred_gain_percent :
  gain_percent 4400 800 5800 ≈ 11.54 :=
by
  sorry

end alfred_gain_percent_l22_22697


namespace sec_225_eq_neg_sqrt2_l22_22465

theorem sec_225_eq_neg_sqrt2 : ∀ θ : ℝ, θ = 225 * (π / 180) → sec θ = -real.sqrt 2 :=
by
  intro θ hθ
  sorry

end sec_225_eq_neg_sqrt2_l22_22465


namespace interior_angle_regular_octagon_l22_22263

theorem interior_angle_regular_octagon : 
  let n := 8 in 
  ∀ (sum_of_interior_angles : ℕ → ℕ) (regular_polygon_interior_angle : ℕ → ℕ),
  sum_of_interior_angles n = (180 * (n - 2)) →
  regular_polygon_interior_angle n = (sum_of_interior_angles n / n) →
  regular_polygon_interior_angle n = 135 :=
by
  intros n sum_of_interior_angles regular_polygon_interior_angle h1 h2
  sorry

end interior_angle_regular_octagon_l22_22263


namespace final_score_is_correct_l22_22951

-- Definitions based on given conditions
def speechContentScore : ℕ := 90
def speechDeliveryScore : ℕ := 85
def weightContent : ℕ := 6
def weightDelivery : ℕ := 4

-- The final score calculation theorem
theorem final_score_is_correct : 
  (speechContentScore * weightContent + speechDeliveryScore * weightDelivery) / (weightContent + weightDelivery) = 88 :=
  by
    sorry

end final_score_is_correct_l22_22951


namespace solve_for_x_l22_22071

-- Define the custom operation for real numbers
def custom_op (a b c d : ℝ) : ℝ := a * c - b * d

-- The theorem to prove
theorem solve_for_x (x : ℝ) (h : custom_op (-x) 3 (x - 2) (-6) = 10) :
  x = 4 ∨ x = -2 :=
sorry

end solve_for_x_l22_22071


namespace sec_225_deg_l22_22457

theorem sec_225_deg : real.sec (225 * real.pi / 180) = -real.sqrt 2 :=
by
  -- Conditions
  have h1 : real.cos (225 * real.pi / 180) = -real.cos (45 * real.pi / 180),
  { rw [←real.cos_add_pi_div_two, real.cos_pi_div_two_sub],
    norm_num, },
  have h2 : real.cos (45 * real.pi / 180) = real.sqrt 2 / 2,
  { exact real.cos_pi_div_four, },
  rw [h1, h2, real.sec],
  field_simp,
  norm_num,
  sorry

end sec_225_deg_l22_22457


namespace nested_radical_simplification_l22_22370

theorem nested_radical_simplification : 
  (sqrt (1 + 2 * sqrt (1 + 3 * sqrt (1 + ⋯ + 2017 * sqrt (1 + 2018 * 2020))))) = 3 :=
by
  -- sorry to skip the proof
  sorry

end nested_radical_simplification_l22_22370


namespace sec_225_eq_neg_sqrt2_l22_22464

theorem sec_225_eq_neg_sqrt2 : ∀ θ : ℝ, θ = 225 * (π / 180) → sec θ = -real.sqrt 2 :=
by
  intro θ hθ
  sorry

end sec_225_eq_neg_sqrt2_l22_22464


namespace shift_to_obtain_f_l22_22065

def period : ℝ := Real.pi

def f (x : ℝ) : ℝ := Real.sin (2 * x)

def g (x : ℝ) : ℝ := Real.cos (2 * x)

theorem shift_to_obtain_f :
  ∀ x, f (x) = g (x + period / 4) := 
sorry

end shift_to_obtain_f_l22_22065


namespace total_weight_is_correct_l22_22324

-- Define the variables
def envelope_weight : ℝ := 8.5
def additional_weight_per_envelope : ℝ := 2
def num_envelopes : ℝ := 880

-- Define the total weight calculation
def total_weight : ℝ := num_envelopes * (envelope_weight + additional_weight_per_envelope)

-- State the theorem to prove that the total weight is as expected
theorem total_weight_is_correct : total_weight = 9240 :=
by
  sorry

end total_weight_is_correct_l22_22324


namespace segment_length_Q_Q_l22_22067

theorem segment_length_Q_Q' (Q Q' : ℝ × ℝ) (Hx1 : Q = (-3, 1)) (Hx2 : Q' = (3, 1)) :
  dist Q Q' = 6 :=
by
  cases Q with x1 y1,
  cases Q' with x2 y2,
  simp [dist, real.dist],
  rw [hx1, hx2],
  norm_num,
  sorry

end segment_length_Q_Q_l22_22067


namespace sum_arithmetic_sequence_sum_sequence_a_div_b_l22_22501

-- Define the arithmetic sequence {a_n} with common difference d
def a_n (a₁ d n : ℤ) : ℤ := a₁ + (n - 1) * d

-- Define the sequence {b_n} as 2^(a_n)
def b_n (a₁ d n : ℤ) : ℤ := 2^(a_n a₁ d n)

-- Sum of the first n terms of the arithmetic sequence {a_n}, denoted as S_n
def S_n (a₁ d n : ℤ) : ℤ := n * a₁ + (n * (n - 1) * d) / 2

-- Sequence {a_n / b_n}
def a_div_b (a₁ d n : ℤ) : ℚ := (a_n a₁ d n) / (b_n a₁ d n)

-- Sum of the first n terms of the sequence {a_n / b_n}, denoted as T_n
noncomputable def T_n (a₁ d n : ℤ) : ℚ := ∑ i in finset.range n, a_div_b a₁ d (i + 1)

-- The first goal: proving the sum of the first n terms of the sequence {a_n}
theorem sum_arithmetic_sequence (n : ℤ) : S_n (-2) 2 n = n^2 - 3 * n := by
  sorry

-- The second goal: proving the sum of the first n terms of the sequence {a_n / b_n}
theorem sum_sequence_a_div_b (n : ℤ) : T_n 1 1 n = (2^(n + 1) - 2 - n) / 2^n := by
  sorry

end sum_arithmetic_sequence_sum_sequence_a_div_b_l22_22501


namespace max_distinct_numbers_example_l22_22601

def max_distinct_numbers (a b c d e : ℕ) : ℕ := sorry

theorem max_distinct_numbers_example
  (A B : ℕ) :
  max_distinct_numbers 100 200 400 A B = 64 := sorry

end max_distinct_numbers_example_l22_22601


namespace regular_octagon_interior_angle_l22_22132

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : 
  let sum_interior_angles := 180 * (n - 2)
  let num_sides := 8
  let one_interior_angle := sum_interior_angles / num_sides
  one_interior_angle = 135 :=
by
  sorry

end regular_octagon_interior_angle_l22_22132


namespace product_of_extrema_of_f_l22_22890

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (x + Real.pi / 4) / Real.sin (x + Real.pi / 3)

theorem product_of_extrema_of_f :
  \let min_val := f 0
  let max_val := f (Real.pi / 2) in
  min_val * max_val = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end product_of_extrema_of_f_l22_22890


namespace find_total_cards_l22_22043

def numCardsInStack (n : ℕ) : Prop :=
  let cards : List ℕ := List.range' 1 (2 * n + 1)
  let pileA := cards.take n
  let pileB := cards.drop n
  let restack := List.zipWith (fun x y => [y, x]) pileA pileB |> List.join
  (restack.take 13).getLastD 0 = 13 ∧ 2 * n = 26

theorem find_total_cards : ∃ (n : ℕ), numCardsInStack n :=
sorry

end find_total_cards_l22_22043


namespace regular_octagon_interior_angle_l22_22183

theorem regular_octagon_interior_angle : 
  ∀ (n : ℕ), n = 8 → (sum_of_internal_angles n) / n = 135 := 
by
  intros n hn
  rw hn
  sorry

end regular_octagon_interior_angle_l22_22183


namespace find_angle_A_find_tan_C_l22_22870

theorem find_angle_A (A B C : ℝ) (h1 : A + B + C = π) 
(h2 : ∃ k : ℝ, (\cos A + 1) / \sqrt 3 = k * (\sin A) / 1) : A = π / 3 :=
sorry

theorem find_tan_C (A B C : ℝ) (h1 : A + B + C = π) 
(h2 : \frac {1 + \sin (2 * B)}{\cos B^2 - \sin B^2} = -3) : tan C = \frac {8 + 5 \sqrt 3}{11} :=
sorry

end find_angle_A_find_tan_C_l22_22870


namespace polygon_division_false_convex_polygon_division_false_convex_polygon_orientation_preserving_true_l22_22749

theorem polygon_division_false (P : polygon) :
  (∃ (A B : polygon), divides_by_broken_line P A B ∧ A = B) →
  (∃ (A B : polygon), divides_by_segment P A B ∧ A = B) → 
  False :=
sorry

theorem convex_polygon_division_false (P : convex_polygon) :
  (∃ (A B : polygon), divides_by_broken_line P A B ∧ A = B) →
  (∃ (A B : polygon), divides_by_segment P A B ∧ A = B) → 
  False :=
sorry

theorem convex_polygon_orientation_preserving_true (P : convex_polygon) :
  (∃ (A B : polygon) (Φ : motion),
     divides_by_broken_line P A B ∧ 
     (Φ.orientation_preserving ∧ Φ.maps A B)) →
  (∃ (A B : polygon) (Φ : motion), 
     divides_by_segment P A B ∧ 
     (Φ.orientation_preserving ∧ Φ.maps A B)) :=
sorry

end polygon_division_false_convex_polygon_division_false_convex_polygon_orientation_preserving_true_l22_22749


namespace find_an_l22_22655

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 1 then 4 else 3 * 4^(n-1)

theorem find_an (n : ℕ) : sequence n = if n = 1 then 4 else 3 * 4^(n-1) := 
by {
  sorry
}

end find_an_l22_22655


namespace nested_radical_simplification_l22_22366

theorem nested_radical_simplification : 
  (sqrt (1 + 2 * sqrt (1 + 3 * sqrt (1 + ⋯ + 2017 * sqrt (1 + 2018 * 2020))))) = 3 :=
by
  -- sorry to skip the proof
  sorry

end nested_radical_simplification_l22_22366


namespace regular_octagon_interior_angle_l22_22095

theorem regular_octagon_interior_angle (n : ℕ) (h₁ : n = 8) :
  (180 * (n - 2)) / n = 135 :=
by
  rw [h₁]
  sorry

end regular_octagon_interior_angle_l22_22095


namespace ratio_is_five_ninths_l22_22300

-- Define the conditions
def total_profit : ℕ := 48000
def total_income : ℕ := 108000

-- Define the total spending based on conditions
def total_spending : ℕ := total_income - total_profit

-- Define the ratio of spending to income
def ratio_spending_to_income : ℚ := total_spending / total_income

-- The theorem we need to prove
theorem ratio_is_five_ninths : ratio_spending_to_income = 5 / 9 := 
  sorry

end ratio_is_five_ninths_l22_22300


namespace max_buses_in_city_l22_22943

theorem max_buses_in_city (n : ℕ) (stops : ℕ) (shared : ℕ) (condition1 : n = 9) (condition2 : stops = 3) (condition3 : shared ≤ 1) : n = 12 :=
sorry

end max_buses_in_city_l22_22943


namespace remainder_is_1_l22_22711

noncomputable def sum_1_div_2008_array_modulo_2008_remainder : ℕ :=
  let p := 2008 in
  let m := 2 * p^2 in
  let n := (2 * p - 1) * (p^2 - 1) in
  (m + n) % 2008

theorem remainder_is_1 :
  sum_1_div_2008_array_modulo_2008_remainder = 1 :=
by
  sorry -- proof placeholder

end remainder_is_1_l22_22711


namespace sec_225_deg_l22_22460

theorem sec_225_deg : real.sec (225 * real.pi / 180) = -real.sqrt 2 :=
by
  -- Conditions
  have h1 : real.cos (225 * real.pi / 180) = -real.cos (45 * real.pi / 180),
  { rw [←real.cos_add_pi_div_two, real.cos_pi_div_two_sub],
    norm_num, },
  have h2 : real.cos (45 * real.pi / 180) = real.sqrt 2 / 2,
  { exact real.cos_pi_div_four, },
  rw [h1, h2, real.sec],
  field_simp,
  norm_num,
  sorry

end sec_225_deg_l22_22460


namespace option_B_identical_sets_l22_22745

-- Define the sets for each option
def setA_M : Set (ℕ × ℕ) := {(3, 2)}
def setA_N : Set (ℕ × ℕ) := {(2, 3)}

def setB_M : Set ℕ := {3, 2}
def setB_N : Set ℕ := {2, 3}

def setC_M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 1}
def setC_N : Set ℝ := {y | ∃ x, x + y = 1}

def setD_M : Set ℕ := {1, 2}
def setD_N : Set (ℕ × ℕ) := {(1, 2)}

-- Define the theorem to identify the correct option
theorem option_B_identical_sets : (setB_M = setB_N) ∧ 
                                   (setA_M ≠ setA_N) ∧
                                   (setC_M ≠ setC_N) ∧
                                   (setD_M ≠ setD_N) 
                                   := by
sorry

end option_B_identical_sets_l22_22745


namespace max_value_expression_l22_22800

open Real

theorem max_value_expression (x y z : ℝ) : 
  (sin (3 * x) + sin (2 * y) + sin z) * (cos (3 * x) + cos (2 * y) + cos z) ≤ 9 / 2 :=
begin
  sorry,  -- Placeholder for the actual proof
end

end max_value_expression_l22_22800


namespace valid_pairs_l22_22774

def is_positive_integer (k : ℤ) : Prop := k > 0 ∧ ∃ n : ℕ, k = n

theorem valid_pairs :
  ∀ (a b : ℤ),
    (∃ t : ℕ, (a = 2 * t ∧ b = 1) ∨ (a = t ∧ b = 2 * t) ∨ (a = 8 * t^4 - t ∧ b = 2 * t))
    → is_positive_integer (a^2 / (2 * a * b^2 - b^3 + 1)) :=
by
  intros a b h
  cases h with t ht
  cases ht with ht1 ht_other
  case Or.inl =>
    have h_eq : a = 2 * t ∧ b = 1 := ht1
    sorry  -- Proof goes here
  case Or.inr ht_other =>
    cases ht_other with ht2 ht3
    case Or.inl =>
      have h_eq : a = t ∧ b = 2 * t := ht2
      sorry  -- Proof goes here
    case Or.inr ht3 =>
      have h_eq : a = 8 * t^4 - t ∧ b = 2 * t := ht3
      sorry  -- Proof goes here

end valid_pairs_l22_22774


namespace angle_complement_supplement_l22_22794

theorem angle_complement_supplement (x : ℝ) (h : 90 - x = 3 / 4 * (180 - x)) : x = 180 :=
by
  sorry

end angle_complement_supplement_l22_22794


namespace distance_from_Bangalore_l22_22042

noncomputable def calculate_distance (speed : ℕ) (start_hour start_minute end_hour end_minute halt_minutes : ℕ) : ℕ :=
  let total_travel_minutes := (end_hour * 60 + end_minute) - (start_hour * 60 + start_minute) - halt_minutes
  let total_travel_hours := total_travel_minutes / 60
  speed * total_travel_hours

theorem distance_from_Bangalore (speed : ℕ) (start_hour start_minute end_hour end_minute halt_minutes : ℕ) :
  speed = 87 ∧ start_hour = 9 ∧ start_minute = 0 ∧ end_hour = 13 ∧ end_minute = 45 ∧ halt_minutes = 45 →
  calculate_distance speed start_hour start_minute end_hour end_minute halt_minutes = 348 := by
  sorry

end distance_from_Bangalore_l22_22042


namespace find_numbers_l22_22390

theorem find_numbers (n : ℕ) (h1 : n ≥ 2) (a : ℕ) (ha : a ≠ 1) (ha_min : ∀ d, d ∣ n → d ≠ 1 → a ≤ d) (b : ℕ) (hb : b ∣ n) :
  n = a^2 + b^2 ↔ n = 8 ∨ n = 20 :=
by sorry

end find_numbers_l22_22390


namespace sum_squares_of_solutions_eq_l22_22820

noncomputable def sum_of_squares_of_solutions : ℚ := sorry

theorem sum_squares_of_solutions_eq :
  (∃ x : ℚ, abs (x^2 - x + (1 : ℚ) / 2010) = (1 : ℚ) / 2010) →
  sum_of_squares_of_solutions = (2008 : ℚ) / 1005 :=
sorry

end sum_squares_of_solutions_eq_l22_22820


namespace find_y_when_x_is_4_l22_22289

-- Conditions from the problem
variables (x y : ℕ) (k : ℕ)
axiom direct_variation : 5 * y = k * x^2
axiom initial_condition : y = 8 ∧ x = 2 → k = 10

-- Prove that the value of y is 32 when x is 4
theorem find_y_when_x_is_4 (h : x = 4) : y = 32 :=
by
  have k_value : k = 10 := sorry -- This comes from initial_condition
  have calculation : 5 * y = 10 * x^2 := sorry -- This is from direct_variation
  rw h at calculation -- Substitute x = 4
  have value_of_5y : 5 * y = 10 * 4^2 := sorry -- Simplify the equation
  have value_of_5y_simplified : 5 * y = 160 := sorry -- Evaluate the expression
  have value_of_y : y = 160 / 5 := sorry -- Solve for y
  exact value_of_y

end find_y_when_x_is_4_l22_22289


namespace volleyball_team_selection_l22_22016

theorem volleyball_team_selection : 
  ∃ (ways : ℕ), ways = 4004 ∧ 
  (∀ (team : finset ℕ), team.card = 6 → 
  (∀ (x : ℕ), x ∈ {Beth, Bonnie} → x ∈ team) ∨ 
  (∀ (x : ℕ), x ∈ {Beth, Bonnie} → x ∉ team)) :=
begin
  sorry
end

end volleyball_team_selection_l22_22016


namespace max_number_of_binary_sequences_l22_22976

theorem max_number_of_binary_sequences (n k : ℕ) (hn : n ≥ k) (hk : k > 0) : 
  ∃ f : ℕ → ℕ → ℕ, f(n, k) = ∑ i in finset.range k, nat.choose n i :=
sorry

end max_number_of_binary_sequences_l22_22976


namespace interior_angle_of_regular_octagon_l22_22110

theorem interior_angle_of_regular_octagon :
  let n := 8 in
  let sum_interior_angles := 180 * (n - 2) in
  let angle := sum_interior_angles / n in
  angle = 135 :=
by
  sorry

end interior_angle_of_regular_octagon_l22_22110


namespace original_number_is_509_l22_22302

theorem original_number_is_509 (n : ℕ) (h : n - 5 = 504) : n = 509 :=
by {
    sorry
}

end original_number_is_509_l22_22302


namespace climb_stairs_l22_22288

noncomputable def u (n : ℕ) : ℝ :=
  let Φ := (1 + Real.sqrt 5) / 2
  let φ := (1 - Real.sqrt 5) / 2
  let A := (1 + Real.sqrt 5) / (2 * Real.sqrt 5)
  let B := (Real.sqrt 5 - 1) / (2 * Real.sqrt 5)
  A * (Φ ^ n) + B * (φ ^ n)

theorem climb_stairs (n : ℕ) (hn : n ≥ 1) : u n = A * (Φ ^ n) + B * (φ ^ n) := sorry

end climb_stairs_l22_22288


namespace grains_difference_l22_22334

/-- Define the function to calculate the number of grains on a square -/
def grains_on_square (k : ℕ) : ℕ := 2^k

/-- Define the summation function for the first n squares -/
def sum_of_first_n_squares (n : ℕ) : ℕ := ∑ i in Finset.range (n+1), grains_on_square i

/-- Prove the main statement -/
theorem grains_difference :
  grains_on_square 15 - sum_of_first_n_squares 12 = 24578 :=
by
  sorry

end grains_difference_l22_22334


namespace find_m_for_parallel_lines_l22_22878

theorem find_m_for_parallel_lines (m : ℝ) :
  (∀ x y : ℝ, 6 * x + m * y - 1 = 0 ↔ 2 * x - y + 1 = 0) → m = -3 :=
by
  sorry

end find_m_for_parallel_lines_l22_22878


namespace find_unknown_rate_l22_22316

def blankets_cost (num : ℕ) (rate : ℕ) (discount_tax : ℕ) (is_discount : Bool) : ℕ :=
  if is_discount then rate * (100 - discount_tax) / 100 * num
  else (rate * (100 + discount_tax) / 100) * num

def total_cost := blankets_cost 3 100 10 true +
                  blankets_cost 4 150 0 false +
                  blankets_cost 3 200 20 false

def avg_cost (total : ℕ) (num : ℕ) : ℕ :=
  total / num

theorem find_unknown_rate
  (unknown_rate : ℕ)
  (h1 : total_cost + 2 * unknown_rate = 1800)
  (h2 : avg_cost (total_cost + 2 * unknown_rate) 12 = 150) :
  unknown_rate = 105 :=
by
  sorry

end find_unknown_rate_l22_22316


namespace intersection_angle_zero_or_pi_l22_22577

theorem intersection_angle_zero_or_pi
  (A B C A' B' C' O x y z : Point)
  (h1 : concurrent [A, A', O])
  (h2 : concurrent [B, B', O])
  (h3 : concurrent [C, C', O])
  (hx : line_intersection (line_through B C) (line_through B' C') = x)
  (hy : line_intersection (line_through C A) (line_through C' A') = y)
  (hz : line_intersection (line_through A B) (line_through A' B') = z) :
  angle x y z = 0 ∨ angle x y z = π := 
sorry

end intersection_angle_zero_or_pi_l22_22577


namespace any_power_ends_in_12890625_l22_22607

theorem any_power_ends_in_12890625 (a : ℕ) (m k : ℕ) (h : a = 10^m * k + 12890625) : ∀ (n : ℕ), 0 < n → ((a ^ n) % 10^8 = 12890625 % 10^8) :=
by
  intros
  sorry

end any_power_ends_in_12890625_l22_22607


namespace combined_fuel_efficiency_l22_22024

-- Define the fuel efficiencies of the cars
def fuel_efficiency_ray : ℝ := 50
def fuel_efficiency_tom : ℝ := 25
def fuel_efficiency_alice : ℝ := 20

-- Define the number of miles driven by each car
variable (m : ℝ) (h₀ : 0 < m)

-- Calculate the gasoline usage for each car
def gas_used_ray := m / fuel_efficiency_ray
def gas_used_tom := m / fuel_efficiency_tom
def gas_used_alice := m / fuel_efficiency_alice

-- Calculate the total gasoline used
def total_gas_used := gas_used_ray + gas_used_tom + gas_used_alice

-- Calculate the total distance driven
def total_distance_driven := 3 * m

-- Calculate the combined miles per gallon
def combined_mpg := total_distance_driven / total_gas_used

-- The target statement to be proved
theorem combined_fuel_efficiency : combined_mpg = 300 / 11 :=
by 
  sorry

end combined_fuel_efficiency_l22_22024


namespace area_of_trapezoid_EFGH_l22_22613

open Real

-- Define the geometric properties of the trapezoid
def trapezoid_EFGH (EH FG EG EF GH h: ℝ) := 
  EH = 60 ∧ FG = 2 * real.sqrt 56 + 60 + 5 * real.sqrt 21 ∧
  EG = 18 ∧ EF = 60 ∧ GH = 25 ∧ h = 10

-- Define the area calculation for a trapezoid
def area_trapezoid (a b h : ℝ) := (1 / 2) * (a + b) * h

-- Prove that the area of trapezoid EFGH is 600 + 10 * sqrt 56 + 25 * sqrt 21
theorem area_of_trapezoid_EFGH : 
  ∀ EH FG EG EF GH h,
  trapezoid_EFGH EH FG EG EF GH h →
  area_trapezoid EH FG h = 600 + 10 * real.sqrt 56 + 25 * real.sqrt 21 :=
by
  intros EH FG EG EF GH h h_def
  -- The proof is omitted
  sorry

end area_of_trapezoid_EFGH_l22_22613


namespace regular_octagon_interior_angle_l22_22097

theorem regular_octagon_interior_angle (n : ℕ) (h₁ : n = 8) :
  (180 * (n - 2)) / n = 135 :=
by
  rw [h₁]
  sorry

end regular_octagon_interior_angle_l22_22097


namespace plane_equation_correct_l22_22767

def plane_equation (x y z : ℝ) : ℝ := 10 * x - 5 * y + 4 * z - 141

noncomputable def gcd (a b c d : ℤ) : ℤ := Int.gcd (Int.gcd a b) (Int.gcd c d)

theorem plane_equation_correct :
  (∀ x y z, plane_equation x y z = 0 ↔ 10 * x - 5 * y + 4 * z - 141 = 0)
  ∧ gcd 10 (-5) 4 (-141) = 1
  ∧ 10 > 0 := by
  sorry

end plane_equation_correct_l22_22767


namespace floor_diff_l22_22427

theorem floor_diff (x : ℝ) (h : x = 13.2) : 
  (Int.floor (x^2) - (Int.floor x) * (Int.floor x) = 5) := by
  sorry

end floor_diff_l22_22427


namespace find_max_n_l22_22524

variables {α : Type*} [LinearOrderedField α]

-- Define the sum S_n of the first n terms of an arithmetic sequence
noncomputable def S_n (a d : α) (n : ℕ) : α := 
  (n : α) / 2 * (2 * a + (n - 1) * d)

-- Given conditions
variable {a d : α}
axiom S11_pos : S_n a d 11 > 0
axiom S12_neg : S_n a d 12 < 0

theorem find_max_n : ∃ (n : ℕ), ∀ k < n, S_n a d k ≤ S_n a d n ∧ (k ≠ n → S_n a d k < S_n a d n) :=
sorry

end find_max_n_l22_22524


namespace sum_of_squares_of_solutions_l22_22825

theorem sum_of_squares_of_solutions :
  let a := (1 : ℝ) / 2010
  in (∀ x : ℝ, abs(x^2 - x + a) = a) → 
     ((0^2 + 1^2) + ((((1*1) - 2 * a) + 1) / a)) = 2008 / 1005 :=
by
  intros a h
  sorry

end sum_of_squares_of_solutions_l22_22825


namespace area_of_region_R_calculation_l22_22764

def side_length_of_hexagon : ℝ := 3
def internal_angle_of_hexagon : ℝ := 120
def area_of_region_R : ℝ := 9 * Real.sqrt 3 / 4

-- Prove that the area of the region R is as calculated
theorem area_of_region_R_calculation
  (hexagon : Π (sides : ℕ) (angle : ℝ), (sides = 6) ∧ (angle = 120)) -- Regular hexagon with 6 sides and each internal angle 120°
  (side_length : ℝ) (cond : side_length = side_length_of_hexagon): -- Each side length is 3
  region_area = area_of_region_R :=  -- The area to be proved is 9sqrt(3)/4
sorry

end area_of_region_R_calculation_l22_22764


namespace zero_in_interval_l22_22039

noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) : ℝ := -2 * x + 3
noncomputable def h (x : ℝ) : ℝ := f x + 2 * x - 3

theorem zero_in_interval : ∃ x : ℝ, x ∈ Set.Ioo (1/2 : ℝ) 1 ∧ h x = 0 := 
sorry

end zero_in_interval_l22_22039


namespace shared_earnings_eq_27_l22_22616

theorem shared_earnings_eq_27
    (shoes_pairs : ℤ) (shoes_cost : ℤ) (shirts : ℤ) (shirts_cost : ℤ)
    (h1 : shoes_pairs = 6) (h2 : shoes_cost = 3)
    (h3 : shirts = 18) (h4 : shirts_cost = 2) :
    (shoes_pairs * shoes_cost + shirts * shirts_cost) / 2 = 27 := by
  sorry

end shared_earnings_eq_27_l22_22616


namespace ali_sold_books_on_friday_l22_22742

theorem ali_sold_books_on_friday :
  ∀ (initial_stock books_sold_Mon books_sold_Tue books_sold_Wed books_sold_Thu books_not_sold books_sold_Friday : ℕ),
  initial_stock = 800 →
  books_sold_Mon = 60 →
  books_sold_Tue = 10 →
  books_sold_Wed = 20 →
  books_sold_Thu = 44 →
  books_not_sold = 600 →
  books_sold_Friday = (initial_stock - books_not_sold) - (books_sold_Mon + books_sold_Tue + books_sold_Wed + books_sold_Thu) →
  books_sold_Friday = 66 :=
by
  intros initial_stock books_sold_Mon books_sold_Tue books_sold_Wed books_sold_Thu books_not_sold books_sold_Friday h1 h2 h3 h4 h5 h6 h7
  simp at *
  sorry

end ali_sold_books_on_friday_l22_22742


namespace health_risk_probability_l22_22341

theorem health_risk_probability :
  let p := 26
  let q := 57
  p + q = 83 :=
by {
  sorry
}

end health_risk_probability_l22_22341


namespace sec_225_deg_l22_22461

theorem sec_225_deg : real.sec (225 * real.pi / 180) = -real.sqrt 2 :=
by
  -- Conditions
  have h1 : real.cos (225 * real.pi / 180) = -real.cos (45 * real.pi / 180),
  { rw [←real.cos_add_pi_div_two, real.cos_pi_div_two_sub],
    norm_num, },
  have h2 : real.cos (45 * real.pi / 180) = real.sqrt 2 / 2,
  { exact real.cos_pi_div_four, },
  rw [h1, h2, real.sec],
  field_simp,
  norm_num,
  sorry

end sec_225_deg_l22_22461


namespace regular_octagon_interior_angle_l22_22125

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : 
  let sum_interior_angles := 180 * (n - 2)
  let num_sides := 8
  let one_interior_angle := sum_interior_angles / num_sides
  one_interior_angle = 135 :=
by
  sorry

end regular_octagon_interior_angle_l22_22125


namespace find_k_perpendicular_l22_22900

def a : ℝ × ℝ × ℝ := (1, 1, 0)
def b : ℝ × ℝ × ℝ := (-1, 0, 2)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem find_k_perpendicular :
  let k := 7 / 5 in
  dot_product ((k - 1, k, 2)) (3, 2, -2) = 0 :=
by
  sorry

end find_k_perpendicular_l22_22900


namespace nested_radicals_equivalent_l22_22364

theorem nested_radicals_equivalent :
  (∃ (f : ℕ → ℕ),
    (∀ k, 1 ≤ k → f k = (k + 1)^2 - 1) ∧
    (sqrt (1 + 2 * sqrt (1 + 3 * sqrt (1 + 2017 * sqrt (1 + 2018 * (f 2018))))) = 3)) :=
begin
  use (λ k, (k + 1)^2 - 1),
  split,
  { intros k hk,
    exact rfl, },
  { sorry },
end

end nested_radicals_equivalent_l22_22364


namespace power_function_difference_l22_22893

def power_function (m : ℝ) (x : ℝ) : ℝ := (m + 3) * (x ^ m)

theorem power_function_difference (m : ℝ) (h : m = -2) : power_function m 2 - power_function m (-2) = 0 :=
by
  have hf : power_function m 2 = (m + 3) * (2 ^ m), from rfl
  have hg : power_function m (-2) = (m + 3) * ((-2) ^ m), from rfl
  rw h at hf hg
  simp at hf hg
  sorry

end power_function_difference_l22_22893


namespace gcd_condition_rational_exponentiation_l22_22612

noncomputable def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_condition_rational_exponentiation (p q : ℕ) :
  gcd p q = 1 ↔ (∀ x : ℝ, (x ∈ ℚ ↔ (x ^ p ∈ ℚ ∧ x ^ q ∈ ℚ))) :=
by
  sorry

end gcd_condition_rational_exponentiation_l22_22612


namespace non_empty_proper_subsets_count_l22_22896

def M : set ℕ := {x | x^2 - 5 * x - 6 ≤ 0}
def N : set ℕ := {s | ∃ a b ∈ M, a ≠ b ∧ s = a + b}

theorem non_empty_proper_subsets_count : ∃ (n : ℕ), n = finset.card (finset.powerset (N.to_finset) \ {∅}) - 1 ∧ n = 2046 :=
by
  sorry

end non_empty_proper_subsets_count_l22_22896


namespace regular_octagon_interior_angle_l22_22243

theorem regular_octagon_interior_angle : 
  ∀ n : ℕ, n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  intros n hn
  simp [sum_of_interior_angles]
  sorry

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
  180 * (n - 2)

end regular_octagon_interior_angle_l22_22243


namespace angle_bisector_MT_l22_22673

-- Definitions of the given conditions
def circles_touch_internally_at_point (C1 C2 : Circle) (M : Point) : Prop :=
  C1.contains M ∧ C2.contains M ∧ tangent_at_point C1 C2 M

def chord_touches_smaller_circle (C1 C2 : Circle) (chord : Line) (T : Point) : Prop :=
  ∃ A B : Point, chord = Line.through A B ∧ C1.is_chord chord ∧ C2.contains T ∧ tangent_at_point_line C2 chord T

-- The main theorem statement to prove
theorem angle_bisector_MT (C1 C2 : Circle) (M A B T : Point) (MT : Line) :
  circles_touch_internally_at_point C1 C2 M →
  chord_touches_smaller_circle C1 C2 (Line.through A B) T →
  MT = Line.through M T →
  is_angle_bisector MT (angle A M B) :=
by
  sorry

end angle_bisector_MT_l22_22673


namespace length_of_common_chord_AB_l22_22496

def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 25

def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 20 = 0

def intersects_at_points_A_B (x y : ℝ) : Prop :=
  circle_O x y ∧ circle_C x y

noncomputable def distance_from_center_to_line : ℝ :=
  (5 : ℝ) / real.sqrt ((4 : ℝ) ^ 2 + (2 : ℝ) ^ 2)

theorem length_of_common_chord_AB :
  let r := (5 : ℝ)
  let d := distance_from_center_to_line
  2 * real.sqrt (r^2 - d^2) = real.sqrt (95) :=
sorry

end length_of_common_chord_AB_l22_22496


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22174

-- Define the conditions for the problem
def num_sides : ℕ := 8
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Statement of the problem
theorem measure_of_one_interior_angle_of_regular_octagon : 
  (sum_of_interior_angles num_sides) / num_sides = 135 :=
by
  -- Conditions are defined above
  -- Proof steps are omitted
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22174


namespace ad_space_length_l22_22666

theorem ad_space_length 
  (num_companies : ℕ)
  (ads_per_company : ℕ)
  (width : ℝ)
  (cost_per_sq_ft : ℝ)
  (total_cost : ℝ) 
  (H1 : num_companies = 3)
  (H2 : ads_per_company = 10)
  (H3 : width = 5)
  (H4 : cost_per_sq_ft = 60)
  (H5 : total_cost = 108000) :
  ∃ L : ℝ, (num_companies * ads_per_company * width * L * cost_per_sq_ft = total_cost) ∧ (L = 12) :=
by
  sorry

end ad_space_length_l22_22666


namespace students_not_make_cut_l22_22704

theorem students_not_make_cut (girls boys called_back total try_outs not_make_cut : ℕ) 
    (h1 : girls = 17) 
    (h2 : boys = 32) 
    (h3 : called_back = 10) 
    (h4 : total = girls + boys) 
    (h5 : try_outs = total) 
    (h6 : not_make_cut = try_outs - called_back) : 
    not_make_cut = 39 := 
by 
    rw [h1, h2, h3] at h4 
    rw h4 at h5 
    rw h5 at h6 
    rw [h1, h2] 
    norm_num at h6 
    exact h6

end students_not_make_cut_l22_22704


namespace regular_octagon_interior_angle_l22_22185

theorem regular_octagon_interior_angle : 
  ∀ (n : ℕ), n = 8 → (sum_of_internal_angles n) / n = 135 := 
by
  intros n hn
  rw hn
  sorry

end regular_octagon_interior_angle_l22_22185


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22169

-- Define the conditions for the problem
def num_sides : ℕ := 8
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Statement of the problem
theorem measure_of_one_interior_angle_of_regular_octagon : 
  (sum_of_interior_angles num_sides) / num_sides = 135 :=
by
  -- Conditions are defined above
  -- Proof steps are omitted
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22169


namespace regular_octagon_interior_angle_l22_22241

theorem regular_octagon_interior_angle : 
  ∀ n : ℕ, n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  intros n hn
  simp [sum_of_interior_angles]
  sorry

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
  180 * (n - 2)

end regular_octagon_interior_angle_l22_22241


namespace log10_800_l22_22784

-- Defining conditions as constants for logarithmic values
constant log10_2 : Real := 0.3010
constant log10_5 : Real := 0.6990

-- Theorem statement
theorem log10_800 : log (800) = 2.903 := by
  sorry

end log10_800_l22_22784


namespace new_angle_after_rotation_l22_22642

def initial_angle : ℝ := 25
def rotation_clockwise : ℝ := 350
def equivalent_rotation := rotation_clockwise - 360  -- equivalent to -10 degrees

theorem new_angle_after_rotation :
  initial_angle + equivalent_rotation = 15 := by
  sorry

end new_angle_after_rotation_l22_22642


namespace count_ordered_pairs_ints_l22_22837

theorem count_ordered_pairs_ints (satisfy_conditions : (x y : ℕ) → Prop) :

(∑ y in Finset.range 149, ((150 - y) / ((y + 1) * (y + 3)))) = 100 :=
by 
  sorry

end count_ordered_pairs_ints_l22_22837


namespace iron_wire_square_rectangle_l22_22319

theorem iron_wire_square_rectangle 
  (total_length : ℕ) 
  (rect_length : ℕ) 
  (h1 : total_length = 28) 
  (h2 : rect_length = 12) :
  (total_length / 4 = 7) ∧
  ((total_length / 2) - rect_length = 2) :=
by 
  sorry

end iron_wire_square_rectangle_l22_22319


namespace floor_of_abs_of_neg_57point6_l22_22401

theorem floor_of_abs_of_neg_57point6 : floor (|(-57.6 : ℝ)|) = 57 := by
  sorry

end floor_of_abs_of_neg_57point6_l22_22401


namespace nested_radicals_equivalent_l22_22360

theorem nested_radicals_equivalent :
  (∃ (f : ℕ → ℕ),
    (∀ k, 1 ≤ k → f k = (k + 1)^2 - 1) ∧
    (sqrt (1 + 2 * sqrt (1 + 3 * sqrt (1 + 2017 * sqrt (1 + 2018 * (f 2018))))) = 3)) :=
begin
  use (λ k, (k + 1)^2 - 1),
  split,
  { intros k hk,
    exact rfl, },
  { sorry },
end

end nested_radicals_equivalent_l22_22360


namespace regular_octagon_interior_angle_l22_22239

theorem regular_octagon_interior_angle : 
  ∀ n : ℕ, n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  intros n hn
  simp [sum_of_interior_angles]
  sorry

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
  180 * (n - 2)

end regular_octagon_interior_angle_l22_22239


namespace degree_of_polynomial_is_14_l22_22076

-- Define the polynomial and the conditions
def poly (x g h i j k l : ℤ) : ℤ := (x^5 + g * x^8 + h * x + i) * (x^4 + j * x^3 + k) * (x^2 + l)

-- The theorem statement
theorem degree_of_polynomial_is_14 (g h i j k l : ℤ) (hg : g ≠ 0) (hh : h ≠ 0) (hi : i ≠ 0) (hj : j ≠ 0) (hk : k ≠ 0) (hl : l ≠ 0) :
  polynomial.degree (poly x g h i j k l) = 14 :=
sorry

end degree_of_polynomial_is_14_l22_22076


namespace P_greater_than_one_over_sixtyfour_l22_22056

open Polynomial

noncomputable def P (a b d: ℝ) := λ x: ℝ, x^3 + a*x^2 + b*x + d
noncomputable def Q := λ x: ℝ, x^2 + x + 2001

theorem P_greater_than_one_over_sixtyfour {a b d: ℝ} (h_distinct_roots : ∃ r1 r2 r3: ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ P a b d r1 = 0 ∧ P a b d r2 = 0 ∧ P a b d r3 = 0) 
    (h_no_real_roots : ∀ x: ℝ, P a b d (Q x) ≠ 0) : 
    P a b d 2001 > 1 / 64 :=
by 
  sorry

end P_greater_than_one_over_sixtyfour_l22_22056


namespace arithmetic_sequence_40th_term_difference_l22_22748

noncomputable def solve_arithmetic_sequence : ℚ :=
  let a : ℚ := 40 in
  let d : ℚ := 35 / 149 in
  let m : ℚ := a - 111 * d in
  let n : ℚ := a + 111 * d in
  n - m

theorem arithmetic_sequence_40th_term_difference :
  solve_arithmetic_sequence = 7770 / 149 :=
by
  sorry

end arithmetic_sequence_40th_term_difference_l22_22748


namespace probability_abs_diff_gt_half_l22_22615

noncomputable theory

open ProbabilityTheory

-- Define the conditions of the problem
def biased_flip (p : ℚ) : Distribution Bool := Bernoulli p 

def uniform_random_variable : Distribution ℝ := uniform01

def generate_number (p : ℚ) : Distribution ℝ :=
  do
    is_heads ← biased_flip p
    if is_heads then pure 0 else uniform_random_variable

-- Statement of the mathematical problem
theorem probability_abs_diff_gt_half :
  let dist := generate_number (2/3)
  P (| dist.sample - dist.sample | > 1/2) = 1/3 :=
sorry

end probability_abs_diff_gt_half_l22_22615


namespace nested_radical_expr_eq_3_l22_22351

theorem nested_radical_expr_eq_3 :
  sqrt (1 + 2 * sqrt (1 + 3 * sqrt (1 + ... + 2017 * sqrt (1 + 2018 * 2020)))) = 3 :=
sorry

end nested_radical_expr_eq_3_l22_22351


namespace regular_octagon_interior_angle_measure_l22_22256

theorem regular_octagon_interior_angle_measure :
  ∀ (n : ℕ), n = 8 → (∑ i in finset.range n, (180 : ℝ)) / n = 135 :=
  by
  sorry

end regular_octagon_interior_angle_measure_l22_22256


namespace perimeter_of_triangle_l22_22656

theorem perimeter_of_triangle (x : ℝ) (h_ratio : 5 * x + 12 * x + 13 * x = 300) (h_area : 3000 = (1/2) * (5 * x) * (12 * x)) : 
  (5 * x + 12 * x + 13 * x) = 300 :=
  
begin
  sorry,
end

end perimeter_of_triangle_l22_22656


namespace domain_of_k_l22_22032

noncomputable def domain_of_h := Set.Icc (-10 : ℝ) 6

def h (x : ℝ) : Prop := x ∈ domain_of_h
def k (x : ℝ) : Prop := h (-3 * x + 1)

theorem domain_of_k : ∀ x : ℝ, k x ↔ x ∈ Set.Icc (-5/3) (11/3) :=
by
  intro x
  change (-3 * x + 1 ∈ Set.Icc (-10 : ℝ) 6) ↔ (x ∈ Set.Icc (-5/3 : ℝ) (11/3))
  sorry

end domain_of_k_l22_22032


namespace unique_triangle_labeling_l22_22639

def convex_polygon_with_triangulation (n : ℕ) := 
  n = 9

def contains_vertex (vertex : ℕ) (triangle_label : ℕ) : Prop :=
  match vertex, triangle_label with
  | 2, 2 | 5, 5 | 1, 1 | 4, 4 | 3, 3 | 6, 6 | 7, 7 => True
  | _, _ => False

theorem unique_triangle_labeling : convex_polygon_with_triangulation 9 →
  ∃! labeling : (ℕ → ℕ),
  (∀ i, i ∈ {1, 2, 3, 4, 5, 6, 7} → contains_vertex i (labeling i)) :=
begin
  intro h,
  use (λ i, i),
  intros i hi,
  cases i; simp [contains_vertex],
  all_goals { trivial },
end

end unique_triangle_labeling_l22_22639


namespace gcd_sequence_relatively_prime_l22_22973

variable {k : ℕ} (h_k : k > 0)

def a : ℕ → ℕ
| 0     := k + 1
| (n+1) := a n * a n - k * a n + k

theorem gcd_sequence_relatively_prime (m n : ℕ) (h_mn : m ≠ n) :
  Nat.gcd (a h_k m) (a h_k n) = 1 := by
  sorry

end gcd_sequence_relatively_prime_l22_22973


namespace marked_price_approx_l22_22344

noncomputable def calculate_marked_price
  (cost_price : ℝ) (tax_rate : ℝ) (shipping_fees : ℝ) (discounts : List ℝ) (desired_profit_rate : ℝ) 
  (selling_price : ℝ) 
  : ℝ :=
    let tax_amount := cost_price * tax_rate
    let total_cost := cost_price + tax_amount + shipping_fees
    let desired_profit := total_cost * desired_profit_rate
    let final_selling_price := total_cost + desired_profit
    let marked_price := (selling_price / (discounts.foldl (λ acc d, acc * (1 - d)) 1))
    marked_price

theorem marked_price_approx : 
  calculate_marked_price 50 0.12 8 [0.10, 0.15, 0.05] 0.3315 85.216 ≈ 117.24 :=
by
  sorry

end marked_price_approx_l22_22344


namespace sec_225_eq_neg_sqrt2_l22_22463

theorem sec_225_eq_neg_sqrt2 : ∀ θ : ℝ, θ = 225 * (π / 180) → sec θ = -real.sqrt 2 :=
by
  intro θ hθ
  sorry

end sec_225_eq_neg_sqrt2_l22_22463


namespace secant_225_equals_neg_sqrt_two_l22_22447

/-- Definition of secant in terms of cosine -/
def sec (θ : ℝ) := 1 / Real.cos θ

/-- Given angles in radians for trig identities -/
def degree_to_radian (d : ℝ) := d * Real.pi / 180

/-- Main theorem statement -/
theorem secant_225_equals_neg_sqrt_two : sec (degree_to_radian 225) = -Real.sqrt 2 :=
by
  sorry

end secant_225_equals_neg_sqrt_two_l22_22447


namespace minimize_translation_symmetry_l22_22551

theorem minimize_translation_symmetry:
  ∃ (ϕ : ℝ), ϕ > 0 ∧ (∀ x : ℝ, sin(2 * (x - ϕ) + π / 4) = sin(-2 * (x - ϕ) - π / 4)) ∧ ϕ = 3 * π / 8 :=
sorry

end minimize_translation_symmetry_l22_22551


namespace regular_octagon_interior_angle_l22_22180

theorem regular_octagon_interior_angle : 
  ∀ (n : ℕ), n = 8 → (sum_of_internal_angles n) / n = 135 := 
by
  intros n hn
  rw hn
  sorry

end regular_octagon_interior_angle_l22_22180


namespace number_of_changing_quantities_l22_22763

-- Define the geometric setup and properties
variable {Point : Type}
variable [MetricSpace Point]

structure Triangle (P A C : Point) :=
  (R : Point) (S : Point)
  (midpoint_PA : Metric.midpoint P A = R)
  (midpoint_PC : Metric.midpoint P C = S)
  (P_moves_parallel_to_AC : ∀ t : ℝ, Metric.line P (Metric.vector R S) (Metric.vector A C).normal)

-- Define the changeable properties
def length_RS_changes (Δ : Triangle P A C) : Prop :=
  ∃ t₁ t₂, (Metric.dist Δ.R Δ.S t₁) ≠ (Metric.dist Δ.R Δ.S t₂)

def perimeter_changes (Δ : Triangle P A C) : Prop :=
  ∃ t₁ t₂, ((Metric.dist Δ.P Δ.A) t₁ + (Metric.dist Δ.A Δ.C) t₁ + (Metric.dist Δ.P Δ.C) t₁)
         ≠ ((Metric.dist Δ.P Δ.A) t₂ + (Metric.dist Δ.A Δ.C) t₂ + (Metric.dist Δ.P Δ.C) t₂)

def area_changes (Δ : Triangle P A C) : Prop :=
  ∃ t₁ t₂, (area Δ.P Δ.A Δ.C t₁) ≠ (area Δ.P Δ.A Δ.C t₂)

def centroid_position_changes (Δ : Triangle P A C) : Prop :=
  ∃ t₁ t₂, (centroid Δ.P Δ.A Δ.C t₁) ≠ (centroid Δ.P Δ.A Δ.C t₂)

-- Triangle setup
variable (P A C : Point) (Δ : Triangle P A C)

theorem number_of_changing_quantities :
  (if length_RS_changes Δ then 1 else 0) +
  (if perimeter_changes Δ then 1 else 0) +
  (if area_changes Δ then 1 else 0) +
  (if centroid_position_changes Δ then 1 else 0) = 2 :=
sorry

end number_of_changing_quantities_l22_22763


namespace sequence_expression_l22_22586

theorem sequence_expression (s a : ℕ → ℝ) (h : ∀ n : ℕ, 1 ≤ n → s n = (3 / 2 * (a n - 1))) :
  ∀ n : ℕ, 1 ≤ n → a n = 3^n :=
by
  sorry

end sequence_expression_l22_22586


namespace regular_octagon_interior_angle_measure_l22_22249

theorem regular_octagon_interior_angle_measure :
  ∀ (n : ℕ), n = 8 → (∑ i in finset.range n, (180 : ℝ)) / n = 135 :=
  by
  sorry

end regular_octagon_interior_angle_measure_l22_22249


namespace seven_lines_regions_l22_22028

def binom (n k : ℕ) : ℕ := if k > n then 0 else Nat.choose n k

theorem seven_lines_regions :
  ∀ (n : ℕ), n = 7 → (∑ x in Finset.range (n+1), binom n 2) + n + 1 = 29 :=
by
  intros n h
  rw [h]
  rw [Finset.sum_range_succ]
  rw [Nat.add_comm 7 1]
  rw [binom]
  sorry

end seven_lines_regions_l22_22028


namespace floor_diff_l22_22428

theorem floor_diff (x : ℝ) (h : x = 13.2) : 
  (Int.floor (x^2) - (Int.floor x) * (Int.floor x) = 5) := by
  sorry

end floor_diff_l22_22428


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22146

theorem measure_of_one_interior_angle_of_regular_octagon : 
  let n := 8 in 
  let sum_of_interior_angles := 180 * (n - 2) in
  let measure_of_one_interior_angle := sum_of_interior_angles / n in 
  measure_of_one_interior_angle = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22146


namespace regular_octagon_interior_angle_measure_l22_22247

theorem regular_octagon_interior_angle_measure :
  ∀ (n : ℕ), n = 8 → (∑ i in finset.range n, (180 : ℝ)) / n = 135 :=
  by
  sorry

end regular_octagon_interior_angle_measure_l22_22247


namespace sum_of_real_solutions_l22_22476

theorem sum_of_real_solutions : 
  ∀ (x : ℝ), sqrt x + sqrt (16 / x) + sqrt (x + 16 / x) = 10 → real_solution_sum = 17.64 :=
begin
  sorry
end

end sum_of_real_solutions_l22_22476


namespace percentage_parents_agree_l22_22329

def total_parents : ℕ := 800
def disagree_parents : ℕ := 640

theorem percentage_parents_agree : 
  ((total_parents - disagree_parents) / total_parents : ℚ) * 100 = 20 := 
by 
  sorry

end percentage_parents_agree_l22_22329


namespace regular_octagon_interior_angle_l22_22219

theorem regular_octagon_interior_angle:
  let n := 8
  let sum_interior_angles (n: ℕ) := 180 * (n - 2)
  let interior_angle (n: ℕ) := sum_interior_angles n / n
  interior_angle n = 135 :=
by
  sorry

end regular_octagon_interior_angle_l22_22219


namespace regular_octagon_interior_angle_l22_22226

theorem regular_octagon_interior_angle:
  let n := 8
  let sum_interior_angles (n: ℕ) := 180 * (n - 2)
  let interior_angle (n: ℕ) := sum_interior_angles n / n
  interior_angle n = 135 :=
by
  sorry

end regular_octagon_interior_angle_l22_22226


namespace no_polynomial_exists_l22_22589

noncomputable theory

def not_exist_polynomial (a b c : ℤ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) : Prop :=
  ¬ ∃ P : Polynomial ℤ, P.eval a = b ∧ P.eval b = c ∧ P.eval c = a

theorem no_polynomial_exists (a b c : ℤ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) : 
  not_exist_polynomial a b c h_distinct :=
sorry

end no_polynomial_exists_l22_22589


namespace distribution_plans_l22_22659

def factories := {A, B, C, D}
def classes := {class1, class2, class3}

theorem distribution_plans :
  ∃ f : classes → factories, (∃ c ∈ classes, f c = A) ∧ fintype.card {f : classes → factories // ∃ c ∈ classes, f c = A} = 37 :=
sorry

end distribution_plans_l22_22659


namespace max_buses_in_city_l22_22939

-- Definition of the problem statement and the maximal number of buses
theorem max_buses_in_city (bus_stops : Finset ℕ) (buses : Finset (Finset ℕ)) 
  (h_stops : bus_stops.card = 9) 
  (h_buses_stops : ∀ b ∈ buses, b.card = 3) 
  (h_shared_stops : ∀ b₁ b₂ ∈ buses, b₁ ≠ b₂ → (b₁ ∩ b₂).card ≤ 1) : 
  buses.card ≤ 12 := 
sorry

end max_buses_in_city_l22_22939


namespace mid_segment_ratio_l22_22605

open Geometry

theorem mid_segment_ratio (A B C D E F X Y : Point) (hABCD : ConvexQuadrilateral A B C D) 
  (hMidE : Midpoint E B C) (hMidF : Midpoint F A D) (hIntersectX : LineSegment E F intersects_diagonal AC at_point X) 
  (hIntersectY : LineSegment E F intersects_diagonal BD at_point Y) :
  (SegmentRatio C X A = SegmentRatio B Y D) :=
  sorry

end mid_segment_ratio_l22_22605


namespace inverse_of_neg_nine_l22_22874

noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then (1 / 3) ^ x 
  else -(3 ^ x)

theorem inverse_of_neg_nine : f⁻¹ (-9) = 2 :=
by {
  have H1: ∀ x : ℝ, x > 0 → f(-x) = (1 / 3) ^ (-x),
  { intros x hx, simp, exact if_neg (not_lt_of_ge ⟨hx⟩) },
  have H2: ∀ x : ℝ, f(x) = -(3 ^ x) when x > 0,
  { intros x hx, have Hfx := H1 x hx, simp at Hfx, simp, exact Hfx },
  have H3: f 2 = -9,
  { rw H2 2, exact pow_one 3 },
  sorry
}

end inverse_of_neg_nine_l22_22874


namespace shopkeeper_gain_l22_22323

theorem shopkeeper_gain :
  ∀ (x : ℝ), (750 / (66.66666666666667 / 100) = x) → x = 1125 :=
by
  intro x
  intro h
  rw [div_div_eq_mul_div, div_self] at h
  subst h
  -- other manipulations are omitted
  sorry

end shopkeeper_gain_l22_22323


namespace nested_radical_value_l22_22354

theorem nested_radical_value :
  (nat.sqrt (1 + 2 * nat.sqrt (1 + 3 * nat.sqrt (1 + ... + 2017 * nat.sqrt (1 + 2018 * 2020))))) = 3 :=
sorry

end nested_radical_value_l22_22354


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22167

-- Define the conditions for the problem
def num_sides : ℕ := 8
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Statement of the problem
theorem measure_of_one_interior_angle_of_regular_octagon : 
  (sum_of_interior_angles num_sides) / num_sides = 135 :=
by
  -- Conditions are defined above
  -- Proof steps are omitted
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22167


namespace snow_probability_first_week_l22_22015

def prob_no_snow_first_four_days : ℚ := (3/4)^4

def prob_no_snow_next_three_days : ℚ := (2/3)^3

def prob_no_snow_first_week : ℚ := prob_no_snow_first_four_days * prob_no_snow_next_three_days

def prob_snow_at_least_once_first_week : ℚ := 1 - prob_no_snow_first_week

theorem snow_probability_first_week :
  prob_snow_at_least_once_first_week = 29/32 :=
by
  have prob_no_snow_first_four_days_calc : prob_no_snow_first_four_days = (81 : ℚ) / 256 := by sorry
  have prob_no_snow_next_three_days_calc : prob_no_snow_next_three_days = (8 : ℚ) / 27 := by sorry
  have prob_no_snow_first_week_calc : prob_no_snow_first_week = (3 : ℚ) / 32 := by sorry
  have prob_snow_at_least_once_first_week_calc : prob_snow_at_least_once_first_week = 1 - (3 / 32) := by sorry
  have prob_snow_at_least_once_first_week_correct : 1 - (3 / 32) = 29 / 32 := by sorry
  exact prob_snow_at_least_once_first_week_correct

end snow_probability_first_week_l22_22015


namespace csc_product_power_l22_22693

theorem csc_product_power (p q : ℕ) (h1 : p > 1) (h2 : q > 1) (h3 : (∏ k in finset.range 1 31, (1 / (Real.sin (3 * k : ℝ)).pow 2)) = p^q) :
  p + q = 61 :=
sorry

end csc_product_power_l22_22693


namespace interior_angle_of_regular_octagon_l22_22113

theorem interior_angle_of_regular_octagon :
  let n := 8 in
  let sum_interior_angles := 180 * (n - 2) in
  let angle := sum_interior_angles / n in
  angle = 135 :=
by
  sorry

end interior_angle_of_regular_octagon_l22_22113


namespace regular_octagon_interior_angle_l22_22221

theorem regular_octagon_interior_angle:
  let n := 8
  let sum_interior_angles (n: ℕ) := 180 * (n - 2)
  let interior_angle (n: ℕ) := sum_interior_angles n / n
  interior_angle n = 135 :=
by
  sorry

end regular_octagon_interior_angle_l22_22221


namespace unique_pair_bound_l22_22609

theorem unique_pair_bound (x : ℝ) (p q : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ 1) :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → |sqrt(1 - x^2) - p * x - q| ≤ (sqrt 2 - 1) / 2) ↔ (p = -1 ∧ q = (1 + sqrt 2) / 2) :=
  sorry

end unique_pair_bound_l22_22609


namespace regular_octagon_interior_angle_measure_l22_22251

theorem regular_octagon_interior_angle_measure :
  ∀ (n : ℕ), n = 8 → (∑ i in finset.range n, (180 : ℝ)) / n = 135 :=
  by
  sorry

end regular_octagon_interior_angle_measure_l22_22251


namespace probability_of_two_distinct_colors_in_sample_of_five_l22_22301

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.factorial n / (nat.factorial k * nat.factorial (n - k))

noncomputable def num_ways_to_choose_two_colors (total_colors : ℕ) : ℕ :=
  binomial total_colors 2

noncomputable def total_possible_sequences (total_colors sample_size : ℕ) : ℕ :=
  total_colors ^ sample_size

noncomputable def num_favorable_sequences (sample_size : ℕ) : ℕ :=
  (2^sample_size) - 2

noncomputable def total_favorable_sequences (total_colors sample_size : ℕ) : ℕ :=
  num_ways_to_choose_two_colors total_colors * num_favorable_sequences sample_size

noncomputable def probability_two_distinct_colors (total_colors sample_size : ℕ) : ℚ :=
  total_favorable_sequences total_colors sample_size / total_possible_sequences total_colors sample_size

theorem probability_of_two_distinct_colors_in_sample_of_five :
  probability_two_distinct_colors 5 5 = 12 / 125 :=
by
  sorry

end probability_of_two_distinct_colors_in_sample_of_five_l22_22301


namespace calc_f_10_l22_22389

def f : ℕ → ℕ 
| 1 := 2
| 2 := 3
| (n + 3) := (f (n + 2)) + (f (n + 1)) + (2 * (n + 3))

theorem calc_f_10 : f 10 = 69 := by
  sorry

end calc_f_10_l22_22389


namespace interior_angle_of_regular_octagon_l22_22116

theorem interior_angle_of_regular_octagon :
  let n := 8 in
  let sum_interior_angles := 180 * (n - 2) in
  let angle := sum_interior_angles / n in
  angle = 135 :=
by
  sorry

end interior_angle_of_regular_octagon_l22_22116


namespace percentage_blue_flags_l22_22298

theorem percentage_blue_flags 
  (F : ℕ) (hF : F % 2 = 0) -- Condition 2: Total number of flags is even
  (C : ℕ) (hC : C = F / 2) -- Condition 3: Each child picks up two flags, C = F / 2
  (BR : ℕ) (hBR : BR = 0.1 * C) -- Condition 6: 10% of children with flags of both colors
  (RR : ℕ) (hRR : RR = 0.4 * C) -- Derived condition: 40% with two red flags
  (HR : ℕ) (hHR : HR = 0.5 * C) -- Condition 5: 50% of children have red flags
  : 50% of children have blue flags := 
begin
  sorry,
end

end percentage_blue_flags_l22_22298


namespace necessarily_negative_sum_l22_22614

theorem necessarily_negative_sum 
  (u v w : ℝ)
  (hu : -1 < u ∧ u < 0)
  (hv : 0 < v ∧ v < 1)
  (hw : -2 < w ∧ w < -1) :
  v + w < 0 :=
sorry

end necessarily_negative_sum_l22_22614


namespace range_of_a_l22_22507

-- Define the input conditions and requirements, and then state the theorem.
def is_acute_angle_cos_inequality (a b c : ℝ) : Prop :=
  a^2 + b^2 > c^2

theorem range_of_a (a : ℝ) :
  is_acute_angle_cos_inequality a 1 3 ∧ is_acute_angle_cos_inequality 1 3 a ∧
  is_acute_angle_cos_inequality 3 a 1 ↔ 2 * Real.sqrt 2 < a ∧ a < Real.sqrt 10 :=
by
  sorry

end range_of_a_l22_22507


namespace milk_original_price_l22_22596

theorem milk_original_price 
  (budget : ℝ) (cost_celery : ℝ) (cost_cereal_discounted : ℝ) 
  (cost_bread : ℝ) (cost_potato : ℝ) (num_potatoes : ℝ) 
  (remainder : ℝ) (discount_milk : ℝ) (total_spent : ℝ) (cost_milk_after_discount : ℝ)
  (total_cost_discounted_items : ℝ) :
  budget = 60 →
  cost_celery = 5 →
  cost_cereal_discounted = 12 * 0.5 →
  cost_bread = 8 →
  cost_potato = 1 →
  num_potatoes = 6 →
  remainder = 26 →
  discount_milk = 0.1 →
  total_cost_discounted_items = cost_celery + cost_cereal_discounted + cost_bread + num_potatoes * cost_potato →
  total_spent = budget - remainder →
  cost_milk_after_discount = total_spent - total_cost_discounted_items →
  total_spent = 34 →
  cost_milk_after_discount = 9 →
  10 = cost_milk_after_discount / (1 - discount_milk) := 
by 
  intros,
  sorry

end milk_original_price_l22_22596


namespace rectangle_area_l22_22736

theorem rectangle_area (x : ℝ) (w : ℝ) (h1 : (3 * w)^2 + w^2 = x^2) : (3 * w) * w = 3 * x^2 / 10 :=
by
  sorry

end rectangle_area_l22_22736


namespace floor_difference_l22_22410

theorem floor_difference (x : ℝ) (h : x = 13.2) : 
  (⌊x^2⌋ : ℤ) - ((⌊x⌋ : ℤ) * (⌊x⌋ : ℤ)) = 5 := 
by
  -- proof skipped
  sorry

end floor_difference_l22_22410


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22166

-- Define the conditions for the problem
def num_sides : ℕ := 8
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Statement of the problem
theorem measure_of_one_interior_angle_of_regular_octagon : 
  (sum_of_interior_angles num_sides) / num_sides = 135 :=
by
  -- Conditions are defined above
  -- Proof steps are omitted
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22166


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22197

theorem measure_of_one_interior_angle_of_regular_octagon 
  (n : Nat) (h_n : n = 8) : 
  (one_interior_angle : ℝ) = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22197


namespace S₉_eq_81_l22_22494

variable (aₙ : ℕ → ℕ) (S : ℕ → ℕ)
variable (n : ℕ)
variable (a₁ d : ℕ)

-- Conditions
axiom S₃_eq_9 : S 3 = 9
axiom S₆_eq_36 : S 6 = 36
axiom S_n_def : ∀ n, S n = n * a₁ + n * (n - 1) / 2 * d

-- Proof obligation
theorem S₉_eq_81 : S 9 = 81 :=
by
  sorry

end S₉_eq_81_l22_22494


namespace solve_inequality_l22_22632

theorem solve_inequality {a x : ℝ} (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ (x : ℝ), (a > 1 ∧ (a^(2/3) ≤ x ∧ x < a^(3/4) ∨ x > a)) ∨ (0 < a ∧ a < 1 ∧ (a^(3/4) < x ∧ x ≤ a^(2/3) ∨ 0 < x ∧ x < a))) :=
sorry

end solve_inequality_l22_22632


namespace floor_difference_l22_22412

theorem floor_difference (x : ℝ) (h : x = 13.2) : 
  (⌊x^2⌋ - ⌊x⌋ * ⌊x⌋) = 5 := by
  have h1 : ⌊x⌋ = 13 := by sorry
  have h2 : ⌊x^2⌋ = 174 := by sorry
  calc
  ⌊x^2⌋ - ⌊x⌋ * ⌊x⌋ = 174 - 13 * 13 : by rw [h1, h2]
                 ... = 5 : by norm_num

end floor_difference_l22_22412


namespace remainder_8_digit_non_decreasing_integers_mod_1000_l22_22583

noncomputable def M : ℕ :=
  Nat.choose 17 8

theorem remainder_8_digit_non_decreasing_integers_mod_1000 :
  M % 1000 = 310 :=
by
  sorry

end remainder_8_digit_non_decreasing_integers_mod_1000_l22_22583


namespace convert_to_rectangular_eq_l22_22384

noncomputable def convert_to_rectangular : Complex :=
  3 * Real.sqrt 2 * Complex.exp (- (5 * Real.pi * Complex.i) / 4)

theorem convert_to_rectangular_eq :
  convert_to_rectangular = -3 - 3 * Complex.i :=
by
  -- Proof placeholder
  sorry

end convert_to_rectangular_eq_l22_22384


namespace least_possible_number_l22_22284

theorem least_possible_number :
  ∃ x : ℕ, (∃ q r : ℕ, x = 34 * q + r ∧ 0 ≤ r ∧ r < 34) ∧
            (∃ q' : ℕ, x = 5 * q' ∧ q' = r + 8) ∧
            x = 75 :=
by
  sorry

end least_possible_number_l22_22284


namespace nested_radical_expr_eq_3_l22_22352

theorem nested_radical_expr_eq_3 :
  sqrt (1 + 2 * sqrt (1 + 3 * sqrt (1 + ... + 2017 * sqrt (1 + 2018 * 2020)))) = 3 :=
sorry

end nested_radical_expr_eq_3_l22_22352


namespace triangle_side_a_value_l22_22963

noncomputable def a_value (A B c : ℝ) : ℝ :=
  30 * Real.sqrt 2 - 10 * Real.sqrt 6

theorem triangle_side_a_value
  (A B : ℝ) (c : ℝ)
  (hA : A = 60)
  (hB : B = 45)
  (hc : c = 20) :
  a_value A B c = 30 * Real.sqrt 2 - 10 * Real.sqrt 6 :=
by
  sorry

end triangle_side_a_value_l22_22963


namespace nested_radical_expr_eq_3_l22_22350

theorem nested_radical_expr_eq_3 :
  sqrt (1 + 2 * sqrt (1 + 3 * sqrt (1 + ... + 2017 * sqrt (1 + 2018 * 2020)))) = 3 :=
sorry

end nested_radical_expr_eq_3_l22_22350


namespace sum_of_squares_of_solutions_l22_22816

theorem sum_of_squares_of_solutions :
  (∑ α in {x | ∃ ε : ℝ, ε = x² - x + 1/2010 ∧ |ε| = 1/2010}, α^2) = 2008 / 1005 :=
by
  sorry

end sum_of_squares_of_solutions_l22_22816


namespace problem_l22_22902

def point := ℕ × ℕ

def steps (start finish : point) := 
  ((finish.fst - start.fst) + (finish.snd - start.snd))

def paths (start finish : point) :=
  if (finish.fst >= start.fst) ∧ (finish.snd >= start.snd) then
    Nat.choose ((finish.fst - start.fst) + (finish.snd - start.snd)) (finish.snd - start.snd)
  else 0

def A : point := (0, 0)
def B : point := (4, 2)
def C : point := (7, 4)

theorem problem : steps A C = 11 ∧ paths A B * paths B C = 150 :=
by
  sorry

end problem_l22_22902


namespace train_crosses_bridge_in_time_l22_22698

-- Definitions for the problem conditions
def length_of_train : ℝ := 165
def speed_of_train_kmph : ℝ := 54
def speed_of_train_mps : ℝ := speed_of_train_kmph * 1000 / 3600
def length_of_bridge : ℝ := 625
def total_distance : ℝ := length_of_train + length_of_bridge
def time_to_cross : ℝ := total_distance / speed_of_train_mps

-- Theorem statement
theorem train_crosses_bridge_in_time :
  time_to_cross = 790 / 15 := by sorry

end train_crosses_bridge_in_time_l22_22698


namespace area_of_triangle_medians_perpendicular_l22_22004

theorem area_of_triangle_medians_perpendicular (X Y Z D E : Type) [AddGroup X] [Module ℝ X]
  (AD BE : ℝ)
  (hAD : AD = 18) (hBE : BE = 24)
  (h_perpendicular : ∀ G : Type, is_centroid G X Y Z → is_perpendicular AD BE) :
  area_of_triangle XYZ = 288 :=
by
  sorry

end area_of_triangle_medians_perpendicular_l22_22004


namespace scientific_notation_of_74850000_l22_22304

theorem scientific_notation_of_74850000 : 74850000 = 7.485 * 10^7 :=
  by
  sorry

end scientific_notation_of_74850000_l22_22304


namespace necessary_and_sufficient_condition_l22_22022

theorem necessary_and_sufficient_condition {a : ℝ} :
    (∃ x : ℝ, a * x^2 + 2 * x + 1 = 0) ↔ a ≤ 1 :=
by
  sorry

end necessary_and_sufficient_condition_l22_22022


namespace jackson_earned_tuesday_l22_22969

-- Definitions from conditions
def works_5_days_week : Prop := True
def goal_week : ℕ := 1000
def earned_monday : ℕ := 300
def per_4_houses : ℕ := 10
def houses_per_day : ℕ := 88

-- Proving Jackson earned $40 on Tuesday 
theorem jackson_earned_tuesday : 
  (exists (earnings_friday: ℕ) (earnings_thursday: ℕ) (earnings_wednesday: ℕ),
   earnings_friday = 220 ∧ earnings_thursday = 220 ∧ earnings_wednesday = 220) →
  let total_needed := goal_week - earned_monday in
  let earnings_per_house := per_4_houses / 4 in
  let earnings_per_day := houses_per_day * earnings_per_house in
  earnings_per_day = 220 →
  (forall (n m o p: ℕ), n + m + o + p = total_needed → p = 40) :=
begin
  intros,
  apply exists.intro _ 220,
  apply exists.intro _ 220,
  apply exists.intro _ 220,

  -- We assume earnings per day is 220 dollars
  use 220,
  use 220,
  use 220,

  split; try {refl},
  split; try {refl},
  split; try {refl},

  let total_needed := goal_week - earned_monday,
  let earnings_per_house := per_4_houses / 4,
  let earnings_per_day := houses_per_day * earnings_per_house,
  -- We assume the earnings per day is 220 dollars
  
  have H: total_needed = 700, by sorry,
  have H1: earnings_per_day = 220, by sorry,
  have H2: earnings_per_house = 2.5, by sorry,

  intros,
  assumption,
  assumption,
  assumption,
  
end

end jackson_earned_tuesday_l22_22969


namespace graph_f_l22_22050

noncomputable def f : ℝ → ℝ := 
  λ x, if x = -2 then 2 else 
       (if x = 0 then 2 else 
       (if x = 4 then 2 else 
       (if x = 2 then 4 else 
       (if x = -2 then -2 else 0))))

theorem graph_f (x : ℝ) : (f (f x) = 2) ↔ x = 2 :=
  by sorry

end graph_f_l22_22050


namespace not_rented_two_bedroom_units_l22_22564

theorem not_rented_two_bedroom_units (total_units : ℕ)
  (units_rented_ratio : ℚ)
  (total_rented_units : ℕ)
  (one_bed_room_rented_ratio two_bed_room_rented_ratio three_bed_room_rented_ratio : ℚ)
  (one_bed_room_rented_count two_bed_room_rented_count three_bed_room_rented_count : ℕ)
  (x : ℕ) 
  (total_two_bed_room_units rented_two_bed_room_units : ℕ)
  (units_ratio_condition : 2*x + 3*x + 4*x = total_rented_units)
  (total_units_condition : total_units = 1200)
  (ratio_condition : units_rented_ratio = 7/12)
  (rented_units_condition : total_rented_units = (7/12) * total_units)
  (one_bed_condition : one_bed_room_rented_ratio = 2/5)
  (two_bed_condition : two_bed_room_rented_ratio = 1/2)
  (three_bed_condition : three_bed_room_rented_ratio = 3/8)
  (one_bed_count : one_bed_room_rented_count = 2 * x)
  (two_bed_count : two_bed_room_rented_count = 3 * x)
  (three_bed_count : three_bed_room_rented_count = 4 * x)
  (x_value : x = total_rented_units / 9)
  (total_two_bed_units_calc : total_two_bed_room_units = 2 * two_bed_room_rented_count)
  : total_two_bed_room_units - two_bed_room_rented_count = 231 :=
  by
  sorry

end not_rented_two_bedroom_units_l22_22564


namespace evaluates_to_m_times_10_pow_1012_l22_22434

theorem evaluates_to_m_times_10_pow_1012 :
  let a := (3:ℤ) ^ 1010
  let b := (4:ℤ) ^ 1012
  (a + b) ^ 2 - (a - b) ^ 2 = 10 ^ 3642 := by
  sorry

end evaluates_to_m_times_10_pow_1012_l22_22434


namespace min_value_of_reciprocal_sum_l22_22498

theorem min_value_of_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : (a - b)^2 = 4 * (a * b)^3) : 
  (1 / a + 1 / b) >= 2 * sqrt 2 := 
begin
  sorry
end

end min_value_of_reciprocal_sum_l22_22498


namespace secant_225_equals_neg_sqrt_two_l22_22445

/-- Definition of secant in terms of cosine -/
def sec (θ : ℝ) := 1 / Real.cos θ

/-- Given angles in radians for trig identities -/
def degree_to_radian (d : ℝ) := d * Real.pi / 180

/-- Main theorem statement -/
theorem secant_225_equals_neg_sqrt_two : sec (degree_to_radian 225) = -Real.sqrt 2 :=
by
  sorry

end secant_225_equals_neg_sqrt_two_l22_22445


namespace percentage_failed_hindi_l22_22954

theorem percentage_failed_hindi 
  (F_E F_B P_BE : ℕ) 
  (h₁ : F_E = 42) 
  (h₂ : F_B = 28) 
  (h₃ : P_BE = 56) :
  ∃ F_H, F_H = 30 := 
by
  sorry

end percentage_failed_hindi_l22_22954


namespace rotate_circles_alignment_l22_22068

theorem rotate_circles_alignment :
  ∃ θ : Fin 200, ∀ (S : Fin 200 → Bool) (L : Fin 200 → Bool),
  (∃ φ : Fin 200, (∑ i, (if (S i = L (i + φ) % 200) then 1 else 0)) ≥ 100) :=
by
  sorry

end rotate_circles_alignment_l22_22068


namespace interior_angle_regular_octagon_l22_22265

theorem interior_angle_regular_octagon : 
  let n := 8 in 
  ∀ (sum_of_interior_angles : ℕ → ℕ) (regular_polygon_interior_angle : ℕ → ℕ),
  sum_of_interior_angles n = (180 * (n - 2)) →
  regular_polygon_interior_angle n = (sum_of_interior_angles n / n) →
  regular_polygon_interior_angle n = 135 :=
by
  intros n sum_of_interior_angles regular_polygon_interior_angle h1 h2
  sorry

end interior_angle_regular_octagon_l22_22265


namespace proof_of_problem_l22_22487

-- Problem Statement
def problem_statement : Prop :=
  ∀ (points : Finset (ℝ × ℝ)) (h_points_count : points.card = 1004),
  ∃ (midpoints : Finset (ℝ × ℝ)), 
    (∀ p1 p2 ∈ points, midpoints ∈ midpoints ( (p1.1 + p2.1)/2, (p1.2 + p2.2)/2)) ∧ midpoints.card ≥ 2005 ∧
    (∃ (specific_points : Finset (ℝ × ℝ)) (h_specific_points_count : specific_points.card = 1004), 
      ∃ (specific_midpoints : Finset (ℝ × ℝ)), 
      (∀ p1 p2 ∈ specific_points, specific_midpoints ∈ specific_midpoints ( (p1.1 + p2.1)/2, (p1.2 + p2.2)/2)) ∧
       specific_midpoints.card = 2005)

theorem proof_of_problem : problem_statement :=
begin
  sorry
end

end proof_of_problem_l22_22487


namespace sara_peaches_l22_22026

theorem sara_peaches (initial_peaches : ℕ) (picked_peaches : ℕ) (total_peaches : ℕ) 
  (h1 : initial_peaches = 24) (h2 : picked_peaches = 37) : 
  total_peaches = 61 :=
by
  sorry

end sara_peaches_l22_22026


namespace ratio_of_triangle_sector_l22_22575

variable (r : ℝ) (θ : ℝ) (T S : ℝ)
-- Conditions
notation "equilateral_triangle_area" => (√3 / 4) * r^2
notation "sector_area" => (1 / 2) * r^2 * θ

-- Statement
theorem ratio_of_triangle_sector {r θ : ℝ} (h1 : T = equilateral_triangle_area r) (h2 : S = sector_area r θ) (h3 : θ = π / 3) : T / S = (3 * √3) / (2 * π) := sorry

end ratio_of_triangle_sector_l22_22575


namespace indicator_light_probability_l22_22343

variable {p : ℝ} (hp: 0 < p ∧ p < 1)
variable (JA JB JC JD : Prop)
variables (hJAjc : JA → JB → JC → JD → False) 
variable (independence : ∀ {E F}, E ↔ F → P[E] = P[F])
variable (P : Prop → ℝ)

def closed (J : Prop) : ℝ := if J then p else 1 - p

noncomputable def prob_flash : ℝ :=
  4 * p - 6 * p^2 + 4 * p^3 - p^4

theorem indicator_light_probability :
  P[(JA ∨ JB ∨ JC ∨ JD)] = 4 * p - 6 * p^2 + 4 * p^3 - p^4 :=
by 
  sorry

end indicator_light_probability_l22_22343


namespace boat_distance_downstream_l22_22297

theorem boat_distance_downstream
  (speed_boat : ℕ)
  (speed_stream : ℕ)
  (time_downstream : ℕ)
  (h1 : speed_boat = 22)
  (h2 : speed_stream = 5)
  (h3 : time_downstream = 8) :
  speed_boat + speed_stream * time_downstream = 216 :=
by
  sorry

end boat_distance_downstream_l22_22297


namespace max_value_expression_exist_x_y_z_l22_22803

theorem max_value_expression (x y z : ℝ) : 
  (sin (3 * x) + sin (2 * y) + sin z) * (cos (3 * x) + cos (2 * y) + cos z) ≤ 9 / 2 :=
sorry

theorem exist_x_y_z (x y z : ℝ) :
  ∃ x y z : ℝ, (sin (3 * x) + sin (2 * y) + sin z) * (cos (3 * x) + cos (2 * y) + cos z) = 9 / 2 :=
sorry

end max_value_expression_exist_x_y_z_l22_22803


namespace sec_225_eq_neg_sqrt2_l22_22441

theorem sec_225_eq_neg_sqrt2 :
  sec 225 = -sqrt 2 :=
by 
  -- Definitions for conditions
  have h1 : ∀ θ : ℕ, sec θ = 1 / cos θ := sorry,
  have h2 : cos 225 = cos (180 + 45) := sorry,
  have h3 : cos (180 + 45) = -cos 45 := sorry,
  have h4 : cos 45 = 1 / sqrt 2 := sorry,
  -- Required proof statement
  sorry

end sec_225_eq_neg_sqrt2_l22_22441


namespace regular_octagon_angle_measure_l22_22209

theorem regular_octagon_angle_measure :
  let n := 8 in
  (180 * (n - 2)) / n = 135 := 
by
  sorry

end regular_octagon_angle_measure_l22_22209


namespace regular_octagon_interior_angle_measure_l22_22250

theorem regular_octagon_interior_angle_measure :
  ∀ (n : ℕ), n = 8 → (∑ i in finset.range n, (180 : ℝ)) / n = 135 :=
  by
  sorry

end regular_octagon_interior_angle_measure_l22_22250


namespace least_possible_mn_correct_l22_22974

def least_possible_mn (m n : ℕ) : ℕ :=
  m + n

theorem least_possible_mn_correct (m n : ℕ) :
  (Nat.gcd (m + n) 210 = 1) →
  (n^n ∣ m^m) →
  ¬(n ∣ m) →
  least_possible_mn m n = 407 :=
by
  sorry

end least_possible_mn_correct_l22_22974


namespace Caitlin_age_l22_22345

theorem Caitlin_age (Aunt_Anna_age : ℕ) (Brianna_age : ℕ) (Caitlin_age : ℕ)
    (h1 : Aunt_Anna_age = 48)
    (h2 : Brianna_age = Aunt_Anna_age / 3)
    (h3 : Caitlin_age = Brianna_age - 6) : 
    Caitlin_age = 10 := by 
  -- proof here
  sorry

end Caitlin_age_l22_22345


namespace train_stop_time_per_hour_l22_22788

theorem train_stop_time_per_hour
    (speed_excl_stoppages : ℕ)
    (speed_incl_stoppages : ℕ)
    (h1 : speed_excl_stoppages = 48)
    (h2 : speed_incl_stoppages = 36) :
    ∃ (t : ℕ), t = 15 :=
by
  sorry

end train_stop_time_per_hour_l22_22788


namespace regular_octagon_interior_angle_l22_22186

theorem regular_octagon_interior_angle : 
  ∀ (n : ℕ), n = 8 → (sum_of_internal_angles n) / n = 135 := 
by
  intros n hn
  rw hn
  sorry

end regular_octagon_interior_angle_l22_22186


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22157

theorem measure_of_one_interior_angle_of_regular_octagon :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 := 
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22157


namespace regular_octagon_interior_angle_l22_22220

theorem regular_octagon_interior_angle:
  let n := 8
  let sum_interior_angles (n: ℕ) := 180 * (n - 2)
  let interior_angle (n: ℕ) := sum_interior_angles n / n
  interior_angle n = 135 :=
by
  sorry

end regular_octagon_interior_angle_l22_22220


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22172

-- Define the conditions for the problem
def num_sides : ℕ := 8
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Statement of the problem
theorem measure_of_one_interior_angle_of_regular_octagon : 
  (sum_of_interior_angles num_sides) / num_sides = 135 :=
by
  -- Conditions are defined above
  -- Proof steps are omitted
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22172


namespace interior_angle_of_regular_octagon_l22_22115

theorem interior_angle_of_regular_octagon :
  let n := 8 in
  let sum_interior_angles := 180 * (n - 2) in
  let angle := sum_interior_angles / n in
  angle = 135 :=
by
  sorry

end interior_angle_of_regular_octagon_l22_22115


namespace sum_of_squares_of_solutions_l22_22813

theorem sum_of_squares_of_solutions :
  (∑ α in {x | ∃ ε : ℝ, ε = x² - x + 1/2010 ∧ |ε| = 1/2010}, α^2) = 2008 / 1005 :=
by
  sorry

end sum_of_squares_of_solutions_l22_22813


namespace max_value_of_quadratic_l22_22679

theorem max_value_of_quadratic : ∀ p : ℝ, -3 * p^2 + 30 * p - 8 ≤ 67 :=
begin
  -- Sorry to skip the proof
  sorry
end

end max_value_of_quadratic_l22_22679


namespace average_age_of_adults_l22_22041

theorem average_age_of_adults (n_total n_girls n_boys n_adults : ℕ) 
                              (avg_age_total avg_age_girls avg_age_boys avg_age_adults : ℕ)
                              (h1 : n_total = 60)
                              (h2 : avg_age_total = 18)
                              (h3 : n_girls = 30)
                              (h4 : avg_age_girls = 16)
                              (h5 : n_boys = 20)
                              (h6 : avg_age_boys = 17)
                              (h7 : n_adults = 10) :
                              avg_age_adults = 26 :=
sorry

end average_age_of_adults_l22_22041


namespace student_prob_correct_l22_22918

theorem student_prob_correct {A B C : ℕ} (has_idea : ℕ) (no_idea : ℕ) (p_idea : ℝ) (p_guess : ℝ) :
  A = 3 → B = 1 → has_idea = A ∧ no_idea = B → p_idea = 0.8 → p_guess = 0.25 → 
  (let total_prob := 
    ( ( Real.binom has_idea 2 / Real.binom 4 2 ) * p_idea^2 ) + 
    ( ( Real.binom no_idea 1 * Real.binom has_idea 1 / Real.binom 4 2 ) * p_idea * p_guess )
  in total_prob = 0.42 ) :=
begin
  intros A_eq B_eq ideas_eq p_idea_eq p_guess_eq,
  sorry,
end

end student_prob_correct_l22_22918


namespace floor_sqrt_divides_iff_l22_22544

def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def sqrt (n : ℤ) : ℝ := Real.sqrt (Int.toReal n)

def divides (a b : ℤ) : Prop := ∃ k, b = a * k

theorem floor_sqrt_divides_iff (n : ℤ) (hn_pos : n > 0) :
  (divides (floor (sqrt n)) n) ↔ 
  ∃ k : ℤ, k > 0 ∧ (n = k^2 ∨ n = k^2 + k ∨ n = k^2 + 2 * k) :=
  sorry

end floor_sqrt_divides_iff_l22_22544


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22148

theorem measure_of_one_interior_angle_of_regular_octagon :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 := 
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22148


namespace pencils_added_l22_22063

theorem pencils_added (initial_pencils total_pencils Mike_pencils : ℕ) 
    (h1 : initial_pencils = 41) 
    (h2 : total_pencils = 71) 
    (h3 : total_pencils = initial_pencils + Mike_pencils) :
    Mike_pencils = 30 := by
  sorry

end pencils_added_l22_22063


namespace journey_distance_l22_22278

theorem journey_distance :
  ∃ D T : ℝ,
    D = 100 * T ∧
    D = 80 * (T + 1/3) ∧
    D = 400 / 3 :=
by
  sorry

end journey_distance_l22_22278


namespace intersection_point_l22_22677

-- Definitions of the equations of the lines and the given point
def line1 (x : ℝ) : ℝ := 3 * x - 4
def line2 (x : ℝ) : ℝ := - (1 / 3) * x + 5
def point : ℝ × ℝ := (3, 2)

-- Statement of the proof problem
theorem intersection_point :
  ∃ (x y : ℝ), (line1 x = y) ∧ (line2 x = y) ∧ (x = 27 / 10) ∧ (y = 41 / 10) :=
by
  sorry

end intersection_point_l22_22677


namespace solve_inequalities_l22_22469

theorem solve_inequalities (x : ℝ) :
  4 * x + 2 < (x - 1) ^ 2 ∧ (x - 1) ^ 2 < 5 * x + 5 ↔
  x ∈ set.Ioo (3 + real.sqrt 10) ((7 + real.sqrt 65) / 2) :=
by sorry

end solve_inequalities_l22_22469


namespace polygon_sides_l22_22734

theorem polygon_sides (h : ∀ (n : ℕ), (180 * (n - 2)) / n = 150) : n = 12 :=
by
  sorry

end polygon_sides_l22_22734


namespace max_answer_yes_max_answer_no_l22_22664

-- Statement (a) - maximum number of islanders who could have answered "Yes" is 35.
theorem max_answer_yes (knights liars : ℕ) (total_islanders : ℕ) 
  (knights_truthful : ∀ n, n = knights → n ≥ 0) 
  (n_islanders : total_islanders = 35) 
  (obs: knights + liars = total_islanders) 
  (statement : ∃ k, k > 3 ∧ 3 * k ≤ knights :
): 
  35 := 
sorry

-- Statement (b) - maximum number of islanders who could have answered "No" is 23.
theorem max_answer_no (knights liars : ℕ) (total_islanders : ℕ)
  (knights_truthful : ∀ n, n = knights → n ≥ 0)
  (n_islanders : total_islanders = 35)
  (obs : knights + liars = total_islanders)
  (statement : ∃ k, k ≤ 3 ∧ 3 * k ≤ knights):
  23 := 
sorry

end max_answer_yes_max_answer_no_l22_22664


namespace maximum_buses_l22_22921

-- Definition of the problem constraints as conditions in Lean
variables (bus : Type) (stop : Type) (buses : set bus) (stops : set stop)
variables (stops_served_by_bus : bus → finset stop)
variables (stop_occurrences : stop → finset bus)

-- Conditions
def has_9_stops (stops : set stop) : Prop := stops.card = 9
def bus_serves_3_stops (b : bus) : Prop := (stops_served_by_bus b).card = 3
def at_most_1_common_stop (b1 b2 : bus) (h1 : b1 ∈ buses) (h2 : b2 ∈ buses) : Prop :=
  (stops_served_by_bus b1 ∩ stops_served_by_bus b2).card ≤ 1

-- Goal
theorem maximum_buses (h1 : has_9_stops stops)
                      (h2 : ∀ b ∈ buses, bus_serves_3_stops b)
                      (h3 : ∀ b1 b2 ∈ buses, at_most_1_common_stop b1 b2 h1 h2) : 
  buses.card ≤ 12 :=
sorry

end maximum_buses_l22_22921


namespace max_arc_length_and_area_l22_22509

-- Define the conditions
variables {C : ℝ}
def r (l : ℝ) := (C - l) / 2
def S (l : ℝ) := (1 / 2) * l * r l

-- Define the proof problem
theorem max_arc_length_and_area :
  ∃ (l_max S_max : ℝ), (l_max = C / 2) ∧ (S_max = C^2 / 16) ∧ (∀ l : ℝ, S l ≤ S_max) :=
sorry

end max_arc_length_and_area_l22_22509


namespace sum_of_odd_integers_between_500_and_800_l22_22681

noncomputable def sum_of_arithmetic_series (first last num_terms : ℕ) : ℕ :=
  (num_terms / 2) * (first + last)

theorem sum_of_odd_integers_between_500_and_800 :
  let first := 501
  let last := 799
  let diff := 2
  let num_terms := (last - first) / diff + 1
  sum_of_arithmetic_series first last num_terms = 97500 :=
by
  let first := 501
  let last := 799
  let diff := 2
  let num_terms := (last - first) / diff + 1
  have : sum_of_arithmetic_series first last num_terms = 97500, by sorry
  exact this

end sum_of_odd_integers_between_500_and_800_l22_22681


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22153

theorem measure_of_one_interior_angle_of_regular_octagon :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 := 
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22153


namespace sara_letters_ratio_l22_22027

variable (L_J : ℕ) (L_F : ℕ) (L_T : ℕ)

theorem sara_letters_ratio (hLJ : L_J = 6) (hLF : L_F = 9) (hLT : L_T = 33) : 
  (L_T - (L_J + L_F)) / L_J = 3 := by
  sorry

end sara_letters_ratio_l22_22027


namespace who_was_hired_l22_22719

theorem who_was_hired :
  (∀ A B C: Prop, 
    let one_hired := (A ∨ B ∨ C) ∧ (¬A ∨ ¬B ∨ ¬C) in
    let statements := 
      (A → ¬C) ∧ -- A says "C was not hired"
      (B → B) ∧ -- B says "I was hired"
      (C → (A → ¬C)) in -- C says "What A said is true" (C → ¬C)
    let one_lies := 
      (A ∧ ¬(A → ¬C) ∧ ¬B ∧ ¬C) ∨ -- A lies
      (¬A ∧ B ∧ ¬(B → B) ∧ ¬C) ∨ -- B lies
      (¬A ∧ ¬B ∧ C ∧ ¬(C → (A → ¬C))) in -- C lies

    one_hired → statements → one_lies → A
  ) :=
begin
  intros A B C,
  let one_hired := (A ∨ B ∨ C) ∧ (¬A ∨ ¬B ∨ ¬C),
  let statements := 
      (A → ¬C) ∧ -- A says "C was not hired"
      (B → B) ∧ -- B says "I was hired"
      (C → (A → ¬C)),
  let one_lies := 
    (A ∧ ¬(A → ¬C) ∧ ¬B ∧ ¬C) ∨ -- A lies
    (¬A ∧ B ∧ ¬(B → B) ∧ ¬C) ∨ -- B lies
    (¬A ∧ ¬B ∧ C ∧ ¬(C → (A → ¬C))),
  intros hire_condition statements_condition one_lie_condition,
  sorry,
end

end who_was_hired_l22_22719


namespace max_value_expression_exist_x_y_z_l22_22805

theorem max_value_expression (x y z : ℝ) : 
  (sin (3 * x) + sin (2 * y) + sin z) * (cos (3 * x) + cos (2 * y) + cos z) ≤ 9 / 2 :=
sorry

theorem exist_x_y_z (x y z : ℝ) :
  ∃ x y z : ℝ, (sin (3 * x) + sin (2 * y) + sin z) * (cos (3 * x) + cos (2 * y) + cos z) = 9 / 2 :=
sorry

end max_value_expression_exist_x_y_z_l22_22805


namespace units_digit_of_power_1_5_l22_22779

theorem units_digit_of_power_1_5 (n : ℕ) (h : n > 0) : 
  (1.5 ^ n).units_digit = 5 := 
by 
  -- Since the units digit of 1.5 is 5, we can infer...
  sorry

end units_digit_of_power_1_5_l22_22779


namespace least_lucky_multiple_of_six_not_lucky_l22_22314

def digit_sum (n : ℕ) : ℕ :=
  (n.to_digits 10).sum

def is_lucky_integer (n : ℕ) : Prop :=
  n % digit_sum n = 0

def is_multiple_of_six (n : ℕ) : Prop :=
  n % 6 = 0

theorem least_lucky_multiple_of_six_not_lucky : ∃ n, is_multiple_of_six n ∧ ¬ is_lucky_integer n ∧ (∀ m, is_multiple_of_six m ∧ ¬ is_lucky_integer m → n ≤ m) :=
by {
  use 12,
  split,
  { sorry },  -- Proof that 12 is a multiple of 6
  split,
  { sorry },  -- Proof that 12 is not a lucky integer
  { sorry },  -- Proof that there is no smaller multiple of 6 that is not a lucky integer
}

end least_lucky_multiple_of_six_not_lucky_l22_22314


namespace regular_octagon_interior_angle_l22_22232

theorem regular_octagon_interior_angle : 
  ∀ n : ℕ, n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  intros n hn
  simp [sum_of_interior_angles]
  sorry

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
  180 * (n - 2)

end regular_octagon_interior_angle_l22_22232


namespace max_seq_length_l22_22678

theorem max_seq_length (a : ℕ → ℕ) (n : ℕ) (h1 : ∀ i, i < n - 2 → even (a i + a (i+1) + a (i+2))) 
(h2 : ∀ i, i < n - 3 → odd (a i + a (i+1) + a (i+2) + a (i+3))) : n ≤ 5 :=
sorry

end max_seq_length_l22_22678


namespace ratio_of_boys_to_girls_l22_22563

theorem ratio_of_boys_to_girls (G T : ℕ) (hG : G = 190) (hT : T = 494) : 
  let B := T - G in 
  ∃ r : ℕ × ℕ, r = (152, 95) ∧ B = 152 * 2 ∧ G = 95 * 2 :=
by
  sorry

end ratio_of_boys_to_girls_l22_22563


namespace infinite_double_sum_l22_22380

theorem infinite_double_sum :
  (∑ j, ∑ k, 2 ^ - (4 * k + j + (k + j) ^ 2) : ℝ) = 4 / 3 :=
sorry

end infinite_double_sum_l22_22380


namespace incenter_invariance_l22_22046

theorem incenter_invariance 
  (ABC : Triangle)
  (circumABC : Circle)
  (w : Circle)
  (F E : Point)
  (h_angle : ∠ ABE + ∠ ACF = 60°) 
  (D : Point)
  (circumAFE : Circle)
  (X Y : Point)
  : incenter (Triangle D X Y) = center w := 
begin
  -- Given conditions
  assume h1 : is_equilateral_triangle ABC,
  assume h2 : is_inscribed_in_circle ABC w,
  assume h3 : F ∈ segment AB,
  assume h4 : E ∈ segment AC,
  assume h5 : second_intersection D circumAFE w,
  assume h6 : intersection D line BC = X,
  assume h7 : intersection F line BC = Y,
  sorry
end

end incenter_invariance_l22_22046


namespace triangle_PQR_area_l22_22777

structure Point where
  x : ℝ
  y : ℝ

def triangle_area (P Q R : Point) : ℝ :=
  1/2 * abs (P.x * (Q.y - R.y) + Q.x * (R.y - P.y) + R.x * (P.y - Q.y))

theorem triangle_PQR_area : 
  let P := Point.mk (-2) 0
  let Q := Point.mk 6 0
  let R := Point.mk 3 (-5)
  triangle_area P Q R = 20 :=
by
  -- Proof omitted as specified
  sorry

end triangle_PQR_area_l22_22777


namespace max_buses_in_city_l22_22932

/--
In a city with 9 bus stops where each bus serves exactly 3 stops and any two buses share at most one stop, 
prove that the maximum number of buses that can be in the city is 12.
-/
theorem max_buses_in_city :
  ∃ (B : Set (Finset ℕ)), 
    (∀ b ∈ B, b.card = 3) ∧ 
    (∀ b1 b2 ∈ B, b1 ≠ b2 → (b1 ∩ b2).card ≤ 1) ∧ 
    (∀ b ∈ B, ∀ s ∈ b, s < 9) ∧ 
    B.card = 12 := sorry

end max_buses_in_city_l22_22932


namespace average_of_rstu_l22_22912

theorem average_of_rstu (r s t u : ℝ) (h : (5 / 4) * (r + s + t + u) = 15) : (r + s + t + u) / 4 = 3 :=
by
  sorry

end average_of_rstu_l22_22912


namespace intersection_A_B_l22_22522

def A : Set ℝ := { x | 1 < x - 1 ∧ x - 1 ≤ 3 }
def B : Set ℝ := { 2, 3, 4 }

theorem intersection_A_B : A ∩ B = {3, 4} := 
by 
  sorry

end intersection_A_B_l22_22522


namespace regular_octagon_interior_angle_l22_22102

theorem regular_octagon_interior_angle (n : ℕ) (h₁ : n = 8) :
  (180 * (n - 2)) / n = 135 :=
by
  rw [h₁]
  sorry

end regular_octagon_interior_angle_l22_22102


namespace general_term_formula_l22_22034

noncomputable def sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n : ℕ, S n = (finset.range n).sum a

theorem general_term_formula (a S : ℕ → ℝ) (hS : sequence a S) :
  (∀ n : ℕ, 3 * (S n) - 2 * (a n) = 1) → ∀ n, a n = (-2)^(n-1) := by
  sorry

end general_term_formula_l22_22034


namespace max_value_sine_cosine_expression_l22_22797

theorem max_value_sine_cosine_expression :
  ∀ x y z : ℝ, 
    (sin (3 * x) + sin (2 * y) + sin z) * (cos (3 * x) + cos (2 * y) + cos z) ≤ 4.5 :=
by
  intros x y z
  sorry

end max_value_sine_cosine_expression_l22_22797


namespace p_finishes_work_l22_22700

noncomputable def p_total_days : ℝ := 24
noncomputable def q_total_days : ℝ := 9
noncomputable def r_total_days : ℝ := 12
noncomputable def q_r_joint_days : ℝ := 3
noncomputable def remaining_work_days := 10

theorem p_finishes_work :
  let q_work_rate := 1 / q_total_days,
      r_work_rate := 1 / r_total_days,
      q_r_combined_rate := q_work_rate + r_work_rate,
      work_done_by_q_r := q_r_combined_rate * q_r_joint_days,
      remaining_work := 1 - work_done_by_q_r,
      p_work_rate := 1 / p_total_days in
  remaining_work / p_work_rate = remaining_work_days :=
by
  sorry

end p_finishes_work_l22_22700


namespace simplify_complex_square_l22_22623

def complex_sq_simplification (a b : ℕ) (i : ℂ) (h : i^2 = -1) : ℂ :=
  (a - b * i)^2

theorem simplify_complex_square : complex_sq_simplification 5 3 complex.I (complex.I_sq) = 16 - 30 * complex.I :=
by sorry

end simplify_complex_square_l22_22623


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22155

theorem measure_of_one_interior_angle_of_regular_octagon :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 := 
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22155


namespace find_value_of_x_plus_5_l22_22633

-- Define a variable x
variable (x : ℕ)

-- Define the condition given in the problem
def condition := x - 10 = 15

-- The statement we need to prove
theorem find_value_of_x_plus_5 (h : x - 10 = 15) : x + 5 = 30 := 
by sorry

end find_value_of_x_plus_5_l22_22633


namespace find_complex_number_l22_22846

-- Define the real parts a and b
variables (a b : ℝ)
-- Define the imaginary unit i
def i : ℂ := complex.I

-- Complex number a + bi
def complex_number (a b : ℝ) : ℂ := a + b * I

-- Given conditions and proof goal
theorem find_complex_number (h : (⟨a, -2⟩ : ℂ) * i = (⟨b, -1⟩ : ℂ)) :
  complex_number a b = -1 + 2 * I :=
sorry

end find_complex_number_l22_22846


namespace interior_angle_regular_octagon_l22_22079

theorem interior_angle_regular_octagon (n : ℕ) (h : n = 8) : 
  let sum_of_interior_angles := 180 * (n - 2)
  let number_of_sides := n
  let one_interior_angle := sum_of_interior_angles / number_of_sides
  in one_interior_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l22_22079


namespace tamara_total_earnings_l22_22751

-- Definitions derived from the conditions in the problem statement.
def pans : ℕ := 2
def pieces_per_pan : ℕ := 8
def price_per_piece : ℕ := 2

-- Theorem stating the required proof problem.
theorem tamara_total_earnings : 
  (pans * pieces_per_pan * price_per_piece) = 32 :=
by
  sorry

end tamara_total_earnings_l22_22751


namespace angle_sum_of_roots_of_complex_eq_32i_l22_22776

noncomputable def root_angle_sum : ℝ :=
  let θ1 := 22.5
  let θ2 := 112.5
  let θ3 := 202.5
  let θ4 := 292.5
  θ1 + θ2 + θ3 + θ4

theorem angle_sum_of_roots_of_complex_eq_32i :
  root_angle_sum = 630 := by
  sorry

end angle_sum_of_roots_of_complex_eq_32i_l22_22776


namespace totalHindi_speaking_children_l22_22699

theorem totalHindi_speaking_children :
  let totalChildren := 60
  let onlyEnglish := 0.30 * totalChildren
  let bothHindiAndEnglish := 0.20 * totalChildren
  let onlyHindi := totalChildren - onlyEnglish - bothHindiAndEnglish
  onlyHindi + bothHindiAndEnglish = 42 :=
by
  let totalChildren := 60
  let onlyEnglish := 0.30 * totalChildren
  let bothHindiAndEnglish := 0.20 * totalChildren
  let onlyHindi := totalChildren - onlyEnglish - bothHindiAndEnglish
  have h : onlyHindi + bothHindiAndEnglish = 42 := by sorry
  exact h

end totalHindi_speaking_children_l22_22699


namespace brokerage_percentage_l22_22075

theorem brokerage_percentage (face_value discount total_cost price brokerage_fee brokerage_percentage : ℝ) 
  (h1 : face_value = 100) 
  (h2 : discount = 5) 
  (h3 : total_cost = 95.2) 
  (h4 : price = face_value - discount) 
  (h5 : brokerage_fee = total_cost - price) 
  (h6 : brokerage_percentage = (brokerage_fee / price) * 100) : 
  brokerage_percentage ≈ 0.21 := 
by 
  rw [h4, h5, h6, h1, h2, h3]
  sorry

end brokerage_percentage_l22_22075


namespace find_a_l22_22556

noncomputable def calculate_a (a : ℝ) : Prop :=
  ∀ x y : ℝ, (4 * x + 3 * y - 10 = 0) →
             (2 * x - y = 0) →
             (a * x + 2 * y + 8 = 0) →
             a = -12

theorem find_a : calculate_a -12 :=
by
  sorry

end find_a_l22_22556


namespace sec_225_eq_neg_sqrt2_l22_22442

theorem sec_225_eq_neg_sqrt2 :
  sec 225 = -sqrt 2 :=
by 
  -- Definitions for conditions
  have h1 : ∀ θ : ℕ, sec θ = 1 / cos θ := sorry,
  have h2 : cos 225 = cos (180 + 45) := sorry,
  have h3 : cos (180 + 45) = -cos 45 := sorry,
  have h4 : cos 45 = 1 / sqrt 2 := sorry,
  -- Required proof statement
  sorry

end sec_225_eq_neg_sqrt2_l22_22442


namespace vector_angle_l22_22497

open Real

variables {V : Type*} [inner_product_space ℝ V]

noncomputable def vec_angle (a b : V) : ℝ :=
if h : a ≠ 0 ∧ b ≠ 0 then
  arccos ((inner a b) / (∥a∥ * ∥b∥))
else 0

theorem vector_angle {a b : V} (ha : a ≠ 0) (hb : b ≠ 0)
  (h1 : ∥a∥ = sqrt 3 * ∥b∥)
  (h2 : inner (a - b) (a - 3 • b) = 0) :
  vec_angle a b = π / 6 :=
by
  sorry

end vector_angle_l22_22497


namespace Lewis_more_items_than_Samantha_l22_22595

def Tanya_items : ℕ := 4
def Samantha_items : ℕ := 4 * Tanya_items
def Lewis_items : ℕ := 20

theorem Lewis_more_items_than_Samantha : (Lewis_items - Samantha_items) = 4 := by
  sorry

end Lewis_more_items_than_Samantha_l22_22595


namespace regular_octagon_interior_angle_l22_22096

theorem regular_octagon_interior_angle (n : ℕ) (h₁ : n = 8) :
  (180 * (n - 2)) / n = 135 :=
by
  rw [h₁]
  sorry

end regular_octagon_interior_angle_l22_22096


namespace volume_ratio_l22_22651

-- Defining the volumes of the sphere and the hemisphere
def volume_sphere (r : ℝ) : ℝ := (4/3) * Real.pi * r^3
def volume_hemisphere (r : ℝ) : ℝ := (2/3) * Real.pi * r^3

-- Establishing the conditions
def radius_sphere (a : ℝ) : ℝ := 4 * a
def radius_hemisphere (a : ℝ) : ℝ := 3 * a

-- The ratio of the volumes of the sphere and the hemisphere given the specified radii
theorem volume_ratio (a : ℝ) : 
  (volume_sphere (radius_sphere a)) / (volume_hemisphere (radius_hemisphere a)) = (128/27) :=
by
  sorry

end volume_ratio_l22_22651


namespace count_numbers_without_digit_2_l22_22904

def does_not_contain_digit (n : ℕ) (d : ℕ) : Prop :=
  ¬(d ∈ (n.digits 10))

def count_valid_numbers (m : ℕ) (d : ℕ) : ℕ :=
  (list.range (m + 1)).countp (λ n => does_not_contain_digit n d)

theorem count_numbers_without_digit_2 : 
  count_valid_numbers 500 2 = 323 :=
by
  sorry

end count_numbers_without_digit_2_l22_22904


namespace find_A_l22_22961

variables {A B C a b c : ℝ} -- Defining variables for angles and sides

-- Given conditions
axiom condition1 : a ^ 2 - b ^ 2 = (sqrt 3) * b * c
axiom condition2 : sin C = 2 * (sqrt 3) * sin B

-- The theorem we aim to prove
theorem find_A (h1 : a ^ 2 - b ^ 2 = (sqrt 3) * b * c) (h2 : sin C = 2 * (sqrt 3) * sin B) : A = 30 :=
begin
  -- Proof goes here
  sorry
end

end find_A_l22_22961


namespace triangle_shortest_side_l22_22332

/--
A triangle has an inscribed circle, with one side being divided into segments
of 5 and 10 units by the point of tangency of the inscribed circle. 
The radius of the inscribed circle is 5 units. Prove that the length of 
the shortest side of the triangle is 25 units.
-/
theorem triangle_shortest_side (a b r : ℕ) (h1 : a = 5) (h2 : b = 10) (h3 : r = 5) : 
  shortest_side_length a b r = 25 :=
by sorry

end triangle_shortest_side_l22_22332


namespace range_of_g_l22_22728

-- Define non-zero constants
variable {p q r s : ℝ}
variable (h_nonzero : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0)

-- Define the function g
def g (x : ℝ) := (p * x + q) / (r * x + s)

-- Define the conditions
axiom g_self_23 : g 23 = 23
axiom g_self_101 : g 101 = 101
axiom g_g_self (x : ℝ) (h_ne : x ≠ -s / r) : g (g x) = x

-- Prove that 62 is not in the range of g
theorem range_of_g : ¬ ∃ x : ℝ, g x = 62 := sorry

end range_of_g_l22_22728


namespace nested_radical_expr_eq_3_l22_22348

theorem nested_radical_expr_eq_3 :
  sqrt (1 + 2 * sqrt (1 + 3 * sqrt (1 + ... + 2017 * sqrt (1 + 2018 * 2020)))) = 3 :=
sorry

end nested_radical_expr_eq_3_l22_22348


namespace regular_octagon_interior_angle_l22_22182

theorem regular_octagon_interior_angle : 
  ∀ (n : ℕ), n = 8 → (sum_of_internal_angles n) / n = 135 := 
by
  intros n hn
  rw hn
  sorry

end regular_octagon_interior_angle_l22_22182


namespace interior_angle_regular_octagon_l22_22266

theorem interior_angle_regular_octagon : 
  let n := 8 in 
  ∀ (sum_of_interior_angles : ℕ → ℕ) (regular_polygon_interior_angle : ℕ → ℕ),
  sum_of_interior_angles n = (180 * (n - 2)) →
  regular_polygon_interior_angle n = (sum_of_interior_angles n / n) →
  regular_polygon_interior_angle n = 135 :=
by
  intros n sum_of_interior_angles regular_polygon_interior_angle h1 h2
  sorry

end interior_angle_regular_octagon_l22_22266


namespace find_n_times_s_l22_22993

noncomputable def S := {x : ℝ // x > 0}

noncomputable def f (x : S) : ℝ := sorry

axiom functional_condition :
  ∀ (x y : S), f(x) * f(y) = f(x * y) + 2023 * (1 / x + 1 / y + 2022 : ℝ)

theorem find_n_times_s : 
  let n := {f_val : ℝ // ∃ (x : S), f ⟨2, sorry⟩ = f_val}.1.card 
  let s := {f_val : ℝ // ∃ (x : S), f ⟨2, sorry⟩ = f_val}.1.sum
  in n * s = 4047 / 2 :=
by
  sorry

end find_n_times_s_l22_22993


namespace smallest_period_monotonically_decreasing_max_min_values_l22_22889
noncomputable def f (x : ℝ) : ℝ := cos (2 * x - π / 6) * sin (2 * x) - 1 / 4

-- Prove that the smallest positive period of f(x) is π/2
theorem smallest_period : ∃ T > 0, ∀ x : ℝ, f (x + T) = f x := 
by
  use π/2
  sorry

-- Prove the monotonically decreasing interval of f(x) is [π/6 + kπ/2, 5π/12 + kπ/2], k ∈ ℤ
theorem monotonically_decreasing : ∀ k : ℤ, ∃ a b : ℝ, a = π/6 + k * (π/2) ∧ b = 5 * π / 12 + k * (π/2) ∧ ∀ x : ℝ, a ≤ x ∧ x ≤ b → f'(x) < 0 := 
by
  intro k
  use π/6 + k * (π/2)
  use 5 * π / 12 + k * (π/2)
  sorry

-- Prove the maximum and minimum values of f(x) on [-π/4, 0] are 1/4 and -1/2 respectively
theorem max_min_values : ∃ max min : ℝ, (max = 1/4) ∧ (min = -1/2) ∧ ∀ x : ℝ, (-π/4 ≤ x ∧ x ≤ 0 → min ≤ f x ∧ f x ≤ max) :=
by
  use 1/4
  use -1/2
  sorry

end smallest_period_monotonically_decreasing_max_min_values_l22_22889


namespace regular_octagon_interior_angle_l22_22094

theorem regular_octagon_interior_angle (n : ℕ) (h₁ : n = 8) :
  (180 * (n - 2)) / n = 135 :=
by
  rw [h₁]
  sorry

end regular_octagon_interior_angle_l22_22094


namespace sum_of_squares_of_solutions_l22_22822

theorem sum_of_squares_of_solutions :
  let a := (1 : ℝ) / 2010
  in (∀ x : ℝ, abs(x^2 - x + a) = a) → 
     ((0^2 + 1^2) + ((((1*1) - 2 * a) + 1) / a)) = 2008 / 1005 :=
by
  intros a h
  sorry

end sum_of_squares_of_solutions_l22_22822


namespace regular_octagon_interior_angle_l22_22230

theorem regular_octagon_interior_angle:
  let n := 8
  let sum_interior_angles (n: ℕ) := 180 * (n - 2)
  let interior_angle (n: ℕ) := sum_interior_angles n / n
  interior_angle n = 135 :=
by
  sorry

end regular_octagon_interior_angle_l22_22230


namespace triangle_area_intercepts_l22_22073

theorem triangle_area_intercepts : 
  let curve := λ x : ℝ, (x - 4)^2 * (x + 3),
      x_intercepts := [(-3 : ℝ), 4],
      y_intercept := (0, curve 0)
  in 
  let base := 4 - (-3),
      height := curve 0,
      area := (1 / 2) * base * height
  in
  area = 168 := by
  -- sorry to skip the proof
  sorry

end triangle_area_intercepts_l22_22073


namespace regular_octagon_interior_angle_l22_22100

theorem regular_octagon_interior_angle (n : ℕ) (h₁ : n = 8) :
  (180 * (n - 2)) / n = 135 :=
by
  rw [h₁]
  sorry

end regular_octagon_interior_angle_l22_22100


namespace baseball_league_games_l22_22713

theorem baseball_league_games
  (N M : ℕ)
  (hN_gt_2M : N > 2 * M)
  (hM_gt_4 : M > 4)
  (h_total_games : 4 * N + 5 * M = 94) :
  4 * N = 64 :=
by
  sorry

end baseball_league_games_l22_22713


namespace not_determinable_parallel_l22_22865

variables {a b c : EuclideanSpace ℝ (Fin 3)}

theorem not_determinable_parallel (h : ∥a∥ = 2 * ∥b∥) : ¬ (a ∥ b) :=
sorry

end not_determinable_parallel_l22_22865


namespace floor_diff_l22_22432

theorem floor_diff (x : ℝ) (h : x = 13.2) : 
  (Int.floor (x^2) - (Int.floor x) * (Int.floor x) = 5) := by
  sorry

end floor_diff_l22_22432


namespace percentage_correct_answers_l22_22782

theorem percentage_correct_answers :
  ∀ (correct incorrect : ℕ), 
    correct = 35 → incorrect = 13 →
    let total := correct + incorrect in
    let percentage := (correct.toReal / total.toReal) * 100 in
    Float.round (percentage * 100) / 100 = 72.92 :=
by
  intros correct incorrect h_correct h_incorrect
  rw [h_correct, h_incorrect]
  let total := 35 + 13
  let percentage := (35.toReal / total.toReal) * 100
  have h1 : Float.round (percentage * 100) / 100 = 72.92 := sorry
  exact h1

end percentage_correct_answers_l22_22782


namespace max_value_expression_l22_22801

open Real

theorem max_value_expression (x y z : ℝ) : 
  (sin (3 * x) + sin (2 * y) + sin z) * (cos (3 * x) + cos (2 * y) + cos z) ≤ 9 / 2 :=
begin
  sorry,  -- Placeholder for the actual proof
end

end max_value_expression_l22_22801


namespace number_of_ordered_pairs_l22_22839

def count_valid_pairs : ℕ :=
  ∑ y in Finset.range 149, (150 - y) / (y * (y + 2))

theorem number_of_ordered_pairs :
  count_valid_pairs = ∑ y in Finset.range 149, (150 - y) / (y * (y + 2)) := by
  sorry

end number_of_ordered_pairs_l22_22839


namespace interior_angle_regular_octagon_l22_22078

theorem interior_angle_regular_octagon (n : ℕ) (h : n = 8) : 
  let sum_of_interior_angles := 180 * (n - 2)
  let number_of_sides := n
  let one_interior_angle := sum_of_interior_angles / number_of_sides
  in one_interior_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l22_22078


namespace height_is_geometric_mean_of_bases_l22_22610

-- Given conditions
variables (a c m : ℝ)
-- we declare the condition that the given trapezoid is symmetric and tangential
variables (isSymmetricTangentialTrapezoid : Prop)

-- The theorem to be proven
theorem height_is_geometric_mean_of_bases 
(isSymmetricTangentialTrapezoid: isSymmetricTangentialTrapezoid) 
: m = Real.sqrt (a * c) :=
sorry

end height_is_geometric_mean_of_bases_l22_22610


namespace no_intersect_x_axis_intersection_points_m_minus3_l22_22861

-- Define the quadratic function y = x^2 - 6x + 2m - 1
def quadratic_function (x m : ℝ) : ℝ := x^2 - 6 * x + 2 * m - 1

-- Theorem for Question 1: The function does not intersect the x-axis if and only if m > 5
theorem no_intersect_x_axis (m : ℝ) : (∀ x : ℝ, quadratic_function x m ≠ 0) ↔ m > 5 := sorry

-- Specific case when m = -3
def quadratic_function_m_minus3 (x : ℝ) : ℝ := x^2 - 6 * x - 7

-- Theorem for Question 2: Intersection points with coordinate axes for m = -3
theorem intersection_points_m_minus3 :
  ((∃ x : ℝ, quadratic_function_m_minus3 x = 0 ∧ (x = -1 ∨ x = 7)) ∧
   quadratic_function_m_minus3 0 = -7) := sorry

end no_intersect_x_axis_intersection_points_m_minus3_l22_22861


namespace count_ordered_pairs_ints_l22_22838

theorem count_ordered_pairs_ints (satisfy_conditions : (x y : ℕ) → Prop) :

(∑ y in Finset.range 149, ((150 - y) / ((y + 1) * (y + 3)))) = 100 :=
by 
  sorry

end count_ordered_pairs_ints_l22_22838


namespace maximum_buses_l22_22923

-- Definition of the problem constraints as conditions in Lean
variables (bus : Type) (stop : Type) (buses : set bus) (stops : set stop)
variables (stops_served_by_bus : bus → finset stop)
variables (stop_occurrences : stop → finset bus)

-- Conditions
def has_9_stops (stops : set stop) : Prop := stops.card = 9
def bus_serves_3_stops (b : bus) : Prop := (stops_served_by_bus b).card = 3
def at_most_1_common_stop (b1 b2 : bus) (h1 : b1 ∈ buses) (h2 : b2 ∈ buses) : Prop :=
  (stops_served_by_bus b1 ∩ stops_served_by_bus b2).card ≤ 1

-- Goal
theorem maximum_buses (h1 : has_9_stops stops)
                      (h2 : ∀ b ∈ buses, bus_serves_3_stops b)
                      (h3 : ∀ b1 b2 ∈ buses, at_most_1_common_stop b1 b2 h1 h2) : 
  buses.card ≤ 12 :=
sorry

end maximum_buses_l22_22923


namespace sum_of_squares_of_solutions_l22_22823

theorem sum_of_squares_of_solutions :
  let a := (1 : ℝ) / 2010
  in (∀ x : ℝ, abs(x^2 - x + a) = a) → 
     ((0^2 + 1^2) + ((((1*1) - 2 * a) + 1) / a)) = 2008 / 1005 :=
by
  intros a h
  sorry

end sum_of_squares_of_solutions_l22_22823


namespace quadrilateral_angle_proof_l22_22723

variables {A B C D E K : Type} [AffineGeometry A B C D E K]

def right_angle_at (P Q R : Type) [AffineGeometry P Q R] : Prop := 
  ∃ a b : P, a ⊥ b

def reflection_of (P Q : Type) [AffineGeometry P Q] (M : Type) [AffineGeometry M] : Prop := 
  is_reflection P Q M

theorem quadrilateral_angle_proof
  (convex : is_convex_quadrilateral A B C D)
  (right_angles : right_angle_at A B C ∧ right_angle_at C D A)
  (extension_of_AD_beyond_D : lies_on_extension E D A)
  (equal_angles : ∠ A B E = ∠ A D C)
  (K_reflection : reflection_of C A K)
  : ∠ A D B = ∠ A K E :=
by
  sorry

end quadrilateral_angle_proof_l22_22723


namespace infinite_double_sum_equals_l22_22378

-- We define j and k as non-negative integers
def is_nonneg_int (n : ℕ) := n ≥ 0

-- The theorem states that the double infinite sum equals 4/3
theorem infinite_double_sum_equals :
  (∑ j in Nat.range (Nat.succ (Nat.max 0 0)), ∑ k in Nat.range (Nat.succ (Nat.max 0 0)), 2 ^ (- (4 * k + j + (k + j) ^ 2))) = 4 / 3 :=
by
  sorry

end infinite_double_sum_equals_l22_22378


namespace regular_octagon_angle_measure_l22_22208

theorem regular_octagon_angle_measure :
  let n := 8 in
  (180 * (n - 2)) / n = 135 := 
by
  sorry

end regular_octagon_angle_measure_l22_22208


namespace regular_octagon_angle_measure_l22_22213

theorem regular_octagon_angle_measure :
  let n := 8 in
  (180 * (n - 2)) / n = 135 := 
by
  sorry

end regular_octagon_angle_measure_l22_22213


namespace inradius_of_right_triangle_is_2_l22_22765

theorem inradius_of_right_triangle_is_2 (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : c = 13)
  (h₄ : a^2 + b^2 = c^2) : ∃ r, r = 2 :=
by
  -- Define the side lengths and conditions
  let A := (1 / 2) * a * b
  let s := (a + b + c) / 2
  -- Express the inradius r
  have hA : A = s * 2,
  sorry -- Proof steps are omitted as per the instructions

-- Note: Steps and details are omitted because the problem only asks for the Lean statement with conditions and goal.

end inradius_of_right_triangle_is_2_l22_22765


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22151

theorem measure_of_one_interior_angle_of_regular_octagon :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 := 
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22151


namespace binomial_identity_sum_binomial_coeff_binomial_expectation_l22_22705

-- Definition of binomial coefficient
def binomial_coeff (n r : ℕ) : ℕ := Nat.choose n r

-- 1) Prove that rC_n^r = nC_{n-1}^{r-1}
theorem binomial_identity {n r : ℕ} (hr : 1 ≤ r) : r * binomial_coeff n r = n * binomial_coeff (n - 1) (r - 1) := sorry

-- 2) Prove the sum C_n^1 + 2C_n^2 + 3C_n^3 + ... + nC_n^n
theorem sum_binomial_coeff (n : ℕ) : (Finset.range n).sum (λ r, (r+1) * binomial_coeff n (r+1)) = n * 2^(n-1) := sorry

-- Definition of binomial distribution for random variable
noncomputable def Binomial (n : ℕ) (p : ℝ) : ProbabilityMassFunction ℕ :=
  ProbabilityMassFunction.ofFintype (λ k, Nat.choose n k * p^k * (1 - p)^(n - k))

-- 3) Prove that for a random variable X ∼ B(n,p), E(X) = np
theorem binomial_expectation {n : ℕ} {p : ℝ} (hp : 0 ≤ p ∧ p ≤ 1) : 
  (ProbabilityMassFunction.ofFintype (λ k, (↑k : ℝ) * (Binomial n p).pdf k)).support.sum = n * p := sorry

end binomial_identity_sum_binomial_coeff_binomial_expectation_l22_22705


namespace sum_squares_of_solutions_eq_l22_22817

noncomputable def sum_of_squares_of_solutions : ℚ := sorry

theorem sum_squares_of_solutions_eq :
  (∃ x : ℚ, abs (x^2 - x + (1 : ℚ) / 2010) = (1 : ℚ) / 2010) →
  sum_of_squares_of_solutions = (2008 : ℚ) / 1005 :=
sorry

end sum_squares_of_solutions_eq_l22_22817


namespace regular_octagon_interior_angle_l22_22227

theorem regular_octagon_interior_angle:
  let n := 8
  let sum_interior_angles (n: ℕ) := 180 * (n - 2)
  let interior_angle (n: ℕ) := sum_interior_angles n / n
  interior_angle n = 135 :=
by
  sorry

end regular_octagon_interior_angle_l22_22227


namespace regular_octagon_interior_angle_l22_22233

theorem regular_octagon_interior_angle : 
  ∀ n : ℕ, n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  intros n hn
  simp [sum_of_interior_angles]
  sorry

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
  180 * (n - 2)

end regular_octagon_interior_angle_l22_22233


namespace max_buses_in_city_l22_22933

/--
In a city with 9 bus stops where each bus serves exactly 3 stops and any two buses share at most one stop, 
prove that the maximum number of buses that can be in the city is 12.
-/
theorem max_buses_in_city :
  ∃ (B : Set (Finset ℕ)), 
    (∀ b ∈ B, b.card = 3) ∧ 
    (∀ b1 b2 ∈ B, b1 ≠ b2 → (b1 ∩ b2).card ≤ 1) ∧ 
    (∀ b ∈ B, ∀ s ∈ b, s < 9) ∧ 
    B.card = 12 := sorry

end max_buses_in_city_l22_22933


namespace scientific_notation_470M_l22_22013

theorem scientific_notation_470M :
  (470000000 : ℝ) = 4.7 * 10^8 :=
sorry

end scientific_notation_470M_l22_22013


namespace cube_root_polynomial_roots_l22_22593

theorem cube_root_polynomial_roots (a b : ℝ)
  (h_pos : ∀ x ∈ {x | (x^3 - x^2 - a*x - b = 0)}, x > 0 ∧ (∃ y z : ℝ, x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ y > 0 ∧ z > 0))
  (h_vieta : ∀ x1 x2 x3 : ℝ, (x1^3 - x1^2 - a * x1 - b = 0) ∧ (x2^3 - x2^2 - a * x2 - b = 0) ∧ (x3^3 - x3^2 - a * x3 - b = 0) ∧ x1 + x2 + x3 = 1 ∧ x1 * x2 + x2 * x3 + x3 * x1 = -a ∧ x1 * x2 * x3 = b) : 
  ∃ y : ℝ, y > 0 ∧ (∀ z : ℂ, is_root (z^3 - z^2 + complex.of_real b * z + complex.of_real a) ∧ (z ≠ y ∨ z.1 ≠ 0)) :=
begin
  sorry
end

end cube_root_polynomial_roots_l22_22593


namespace regular_octagon_interior_angle_l22_22244

theorem regular_octagon_interior_angle : 
  ∀ n : ℕ, n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  intros n hn
  simp [sum_of_interior_angles]
  sorry

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
  180 * (n - 2)

end regular_octagon_interior_angle_l22_22244


namespace max_trees_cut_l22_22566

-- Definitions and conditions:
def grid_size := 100
def trees_planted := (grid_size * grid_size : Nat)
def max_cuttable_trees := 2500

-- Theorem statement:
theorem max_trees_cut : ∃ max_cuttable_trees, ∀ stump, stump_visible(stump) = false :=
  sorry

end max_trees_cut_l22_22566


namespace smallest_positive_and_largest_negative_l22_22059

theorem smallest_positive_and_largest_negative:
  (∃ (a : ℤ), a > 0 ∧ ∀ (b : ℤ), b > 0 → b ≥ a ∧ a = 1) ∧
  (∃ (c : ℤ), c < 0 ∧ ∀ (d : ℤ), d < 0 → d ≤ c ∧ c = -1) :=
by
  sorry

end smallest_positive_and_largest_negative_l22_22059


namespace find_d_l22_22793

-- Define the required polynomial and division setup
def polynomial_division_remainder (d : ℚ) :=
  let p := 3 * X^3 + d * X^2 + 7 * X - 27
  let q := 3 * X + 5
  -- Performing polynomial division to find the remainder
  let r := p % q
  r = -3

-- The theorem stating d = 14/25 when the remainder is -3
theorem find_d : polynomial_division_remainder (14 / 25) :=
by
  sorry

end find_d_l22_22793


namespace largest_prime_factor_sum_of_divisors_180_l22_22991

def sum_of_divisors (n : ℕ) : ℕ :=
  ∏ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d

theorem largest_prime_factor_sum_of_divisors_180 :
  let M := sum_of_divisors 180 in
  ∀ p, nat.prime p → p ∣ M → p ≤ 13 :=
by
  let M := sum_of_divisors 180
  have p_factors : multiset ℕ := (unique_factorization_monoid.factorization M).to_multiset
  exact sorry

end largest_prime_factor_sum_of_divisors_180_l22_22991


namespace problem_l22_22911

theorem problem (x y z : ℕ) (h1 : xy + z = 56) (h2 : yz + x = 56) (h3 : zx + y = 56) : x + y + z = 21 :=
sorry

end problem_l22_22911


namespace percent_less_l22_22915

theorem percent_less (w u y z : ℝ) (P : ℝ) (hP : P = 0.40)
  (h1 : u = 0.60 * y)
  (h2 : z = 0.54 * y)
  (h3 : z = 1.50 * w) :
  w = (1 - P) * u := 
sorry

end percent_less_l22_22915


namespace min_value_expression_l22_22995

theorem min_value_expression (a b c d : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : 0 < d) :
  ∃ m : ℝ, m = 12 ∧ (∀ (a b c d : ℝ), 0 < a → 0 < b → 0 < c → 0 < d →
  ((a + b + c) / d + (a + b + d) / c + (a + c + d) / b + (b + c + d) / a) ≥ m) :=
begin
  use 12,
  intros a b c d ha hb hc hd,
  sorry
end

end min_value_expression_l22_22995


namespace regular_octagon_interior_angle_l22_22238

theorem regular_octagon_interior_angle : 
  ∀ n : ℕ, n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  intros n hn
  simp [sum_of_interior_angles]
  sorry

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
  180 * (n - 2)

end regular_octagon_interior_angle_l22_22238


namespace tiling_6x6_square_l22_22710

theorem tiling_6x6_square (k : ℕ) (hk : 0 ≤ k ∧ k ≤ 12) :
  (∃ (L : ℕ) (R : ℕ), L = k ∧ R = 12 - k ∧ (0 < L → L % 2 = 0 ∨ L ∈ {5, 7, 9, 11})) :=
by {
  sorry
}

end tiling_6x6_square_l22_22710


namespace time_to_plant_2500_trees_l22_22031

def rate := 10 / 3 -- trees per minute
def time_per_tree := 3 / 10 -- minutes per tree
def total_time_minutes := 2500 * time_per_tree
def total_time_hours := total_time_minutes / 60

theorem time_to_plant_2500_trees : total_time_hours = 12.5 := by
  -- Since this is a statement, we add 'sorry' for the proof part
  sorry

end time_to_plant_2500_trees_l22_22031


namespace smallest_sum_B_d_l22_22906

theorem smallest_sum_B_d :
  ∃ B d : ℕ, (B < 5) ∧ (d > 6) ∧ (125 * B + 25 * B + B = 4 * d + 4) ∧ (B + d = 77) :=
by
  sorry

end smallest_sum_B_d_l22_22906


namespace angle_measure_in_octagon_square_l22_22327

theorem angle_measure_in_octagon_square (A B C D E : Point) 
  (is_octagon : regular_octagon A B)
  (is_square : square_touching_octagon_side D E) 
  (is_consecutive_vertices_AB : consecutive_vertices A B)
  (C_on_line_ext_AB : C = extension_of_AB A B)
  (interior_angle_octagon_exp : interior_angle_octagon = 135) 
  (interior_angle_square_exp : interior_angle_square = 90) :
  ∠ABC = 67.5 := 
sorry

end angle_measure_in_octagon_square_l22_22327


namespace flensburgian_set_l22_22775

namespace Flensburgian

def isFlensburgian (a b c : ℝ) (n : ℕ) : Prop :=
  (∀ {x y z : ℝ}, x ≠ y ∧ y ≠ z ∧ z ≠ x → (a^n + b = a) ∧ (c^(n + 1) + b^2 = a * b) → 
    (∃ i ∈ {a, b, c}, (∀ j ∈ {a, b, c}, j ≠ i → i > j)))

theorem flensburgian_set (n : ℕ) (h : n ≥ 2) : 
  (∀ a b c : ℝ, isFlensburgian a b c n) ↔ even n :=
by
  sorry

end Flensburgian

end flensburgian_set_l22_22775


namespace inscribed_circle_radius_l22_22274

-- Define the triangle sides
def DE := 8
def DF := 10
def EF := 12

-- Define the semiperimeter
def s := (DE + DF + EF) / 2

-- Using Heron's formula to define the area
def area := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))

-- Define the inscribed circle radius
def inscribed_radius := area / s

-- Statement to prove
theorem inscribed_circle_radius : inscribed_radius = Real.sqrt 7 :=
by
  sorry

end inscribed_circle_radius_l22_22274


namespace secant_225_equals_neg_sqrt_two_l22_22449

/-- Definition of secant in terms of cosine -/
def sec (θ : ℝ) := 1 / Real.cos θ

/-- Given angles in radians for trig identities -/
def degree_to_radian (d : ℝ) := d * Real.pi / 180

/-- Main theorem statement -/
theorem secant_225_equals_neg_sqrt_two : sec (degree_to_radian 225) = -Real.sqrt 2 :=
by
  sorry

end secant_225_equals_neg_sqrt_two_l22_22449


namespace regular_octagon_interior_angle_l22_22130

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : 
  let sum_interior_angles := 180 * (n - 2)
  let num_sides := 8
  let one_interior_angle := sum_interior_angles / num_sides
  one_interior_angle = 135 :=
by
  sorry

end regular_octagon_interior_angle_l22_22130


namespace ball_diameter_correct_l22_22296

noncomputable def ball_diameter : ℝ :=
  let r : ℝ := 5 in
  2 * r

theorem ball_diameter_correct : ball_diameter = 10 := by
  let r := 5
  have h : r = 5 := rfl
  calc
    ball_diameter
        = 2 * r : by rfl
    ... = 2 * 5 : by rw [h]
    ... = 10   : by norm_num

#eval ball_diameter

end ball_diameter_correct_l22_22296


namespace interior_angle_of_regular_octagon_l22_22112

theorem interior_angle_of_regular_octagon :
  let n := 8 in
  let sum_interior_angles := 180 * (n - 2) in
  let angle := sum_interior_angles / n in
  angle = 135 :=
by
  sorry

end interior_angle_of_regular_octagon_l22_22112


namespace sum_squares_of_solutions_eq_l22_22821

noncomputable def sum_of_squares_of_solutions : ℚ := sorry

theorem sum_squares_of_solutions_eq :
  (∃ x : ℚ, abs (x^2 - x + (1 : ℚ) / 2010) = (1 : ℚ) / 2010) →
  sum_of_squares_of_solutions = (2008 : ℚ) / 1005 :=
sorry

end sum_squares_of_solutions_eq_l22_22821


namespace lambda_correct_l22_22495

-- Defining the problem with the variables and conditions
variables (n : ℕ) (θ : Fin n → ℝ)
noncomputable def lambda (n : ℕ) : ℝ := 5 * n / 2

-- The condition that must be satisfied
def condition (θ : Fin n → ℝ) (n : ℕ) :=
  (∀ i, 0 < θ i ∧ θ i < π / 2) ∧
  (  (∑ i, Real.tan (θ i)) * (∑ i, Real.cot (θ i))
     ≥ (∑ i, Real.sin (θ i))^2 +
       (∑ i, Real.cos (θ i))^2 +
       lambda n * (θ 0 - θ (n - 1))^2 )

-- The theorem that proves the equivalent math statement
theorem lambda_correct (h : n ≥ 2) :
  ∀ θ : Fin n → ℝ, condition θ n → lambda n = 5 * n / 2 :=
by
  sorry

end lambda_correct_l22_22495


namespace contestant_final_score_l22_22953

theorem contestant_final_score 
    (content_score : ℕ)
    (delivery_score : ℕ)
    (weight_content : ℕ)
    (weight_delivery : ℕ)
    (h1 : content_score = 90)
    (h2 : delivery_score = 85)
    (h3 : weight_content = 6)
    (h4 : weight_delivery = 4) : 
    (content_score * weight_content + delivery_score * weight_delivery) / (weight_content + weight_delivery) = 88 := 
sorry

end contestant_final_score_l22_22953


namespace secant_225_equals_neg_sqrt_two_l22_22448

/-- Definition of secant in terms of cosine -/
def sec (θ : ℝ) := 1 / Real.cos θ

/-- Given angles in radians for trig identities -/
def degree_to_radian (d : ℝ) := d * Real.pi / 180

/-- Main theorem statement -/
theorem secant_225_equals_neg_sqrt_two : sec (degree_to_radian 225) = -Real.sqrt 2 :=
by
  sorry

end secant_225_equals_neg_sqrt_two_l22_22448


namespace lucky_sum_mod_1000_l22_22317

def is_lucky (n : ℕ) : Prop := ∀ d ∈ n.digits 10, d = 7

def first_twenty_lucky_numbers : List ℕ :=
  [7, 77] ++ List.replicate 18 777

theorem lucky_sum_mod_1000 :
  (first_twenty_lucky_numbers.sum % 1000) = 70 := 
sorry

end lucky_sum_mod_1000_l22_22317


namespace elegant_polygon_max_sides_l22_22722

-- Define an elegant polygon in terms of decomposition into equilateral triangles and squares.
def is_elegant_polygon (n : ℕ) (polygon : Type) : Prop :=
  ∃ (edges : polygon → ℕ) (decompose: polygon → list (Type)),
  (∀ p ∈ decompose polygon, p = ℤ) ∧ -- For simplicity, assume equilateral triangles and squares are represented by integers.
  (∀ s, s ∈ decompose polygon → edges s = n)

-- The sum of the internal angles of a convex polygon
def sum_internal_angles (n : ℕ) : ℕ :=
  (n - 2) * 180

-- The maximum angle for any elegant polygon
def max_internal_angle : ℕ := 150

-- Constraint: n * max_internal_angle
def max_sum_internal_angles (n : ℕ) : ℕ :=
  n * max_internal_angle

-- The main theorem to state that an elegant polygon cannot have more than 12 sides.
theorem elegant_polygon_max_sides (n : ℕ) (polygon : Type) :
  is_elegant_polygon n polygon → n ≤ 12 :=
by
  -- Proof details are omitted
  sorry

end elegant_polygon_max_sides_l22_22722


namespace average_annual_growth_rate_in_2014_and_2015_l22_22714

noncomputable def average_annual_growth_rate (p2013 p2015 : ℝ) (x : ℝ) : Prop :=
  p2013 * (1 + x)^2 = p2015

theorem average_annual_growth_rate_in_2014_and_2015 :
  average_annual_growth_rate 6.4 10 0.25 :=
by
  unfold average_annual_growth_rate
  sorry

end average_annual_growth_rate_in_2014_and_2015_l22_22714


namespace find_number_l22_22294

theorem find_number (x : ℕ) (h : 5 * x = 100) : x = 20 :=
by
  sorry

end find_number_l22_22294


namespace maximum_buses_l22_22922

-- Definition of the problem constraints as conditions in Lean
variables (bus : Type) (stop : Type) (buses : set bus) (stops : set stop)
variables (stops_served_by_bus : bus → finset stop)
variables (stop_occurrences : stop → finset bus)

-- Conditions
def has_9_stops (stops : set stop) : Prop := stops.card = 9
def bus_serves_3_stops (b : bus) : Prop := (stops_served_by_bus b).card = 3
def at_most_1_common_stop (b1 b2 : bus) (h1 : b1 ∈ buses) (h2 : b2 ∈ buses) : Prop :=
  (stops_served_by_bus b1 ∩ stops_served_by_bus b2).card ≤ 1

-- Goal
theorem maximum_buses (h1 : has_9_stops stops)
                      (h2 : ∀ b ∈ buses, bus_serves_3_stops b)
                      (h3 : ∀ b1 b2 ∈ buses, at_most_1_common_stop b1 b2 h1 h2) : 
  buses.card ≤ 12 :=
sorry

end maximum_buses_l22_22922


namespace floor_difference_l22_22409

theorem floor_difference (x : ℝ) (h : x = 13.2) : 
  (⌊x^2⌋ : ℤ) - ((⌊x⌋ : ℤ) * (⌊x⌋ : ℤ)) = 5 := 
by
  -- proof skipped
  sorry

end floor_difference_l22_22409


namespace hundreds_digit_of_factorial_difference_l22_22676

theorem hundreds_digit_of_factorial_difference :
  (15! % 1000 = 0) → 
  (20! % 1000 = 0) → 
  (25! % 1000 = 0) → 
  (10! % 1000 = 800) → 
  (let result := (25! - 20! + (15! - 10!)) % 1000 in
   (result / 100) % 10 = 2) :=
by
  intros h1 h2 h3 h4
  let result := (25! - 20! + (15! - 10!)) % 1000
  have : result / 100 % 10 = 2 := sorry
  exact this

end hundreds_digit_of_factorial_difference_l22_22676


namespace proof_problem_l22_22568

-- Assume the polar equation of circle C
def polar_eq_circle_C (ρ θ : ℝ) : Prop :=
  ρ = 2 * sqrt 2 * sin (θ - π / 4)

-- Assume the parametric equation of line l
def param_eq_line_l (t x y : ℝ) : Prop :=
  x = -t ∧ y = 1 + t

-- Define the Cartesian coordinate equation of circle C
def cartesian_eq_circle_C (x y : ℝ) : Prop :=
  (x + 1)^2 + (y - 1)^2 = 2

-- Define distances from point M(0,1) to points A and B
def distance_MA_MB (MA MB : ℝ) : ℝ :=
  |MA * MB|

-- Define the problem statement
theorem proof_problem (t ρ θ x y MA MB : ℝ) :
  polar_eq_circle_C ρ θ ∧ param_eq_line_l t x y ∧ 
  ((MA ≠ MB) → M(0,1)orthogonal distance) →
  (cartesian_eq_circle_C x y ∧ distance_MA_MB MA MB = 1) :=
by
  sorry

end proof_problem_l22_22568


namespace functional_form_l22_22546

noncomputable def f_condition (f : ℝ → ℝ) (k : ℝ) : Prop :=
  continuous f ∧ 
  f 0 = 0 ∧ 
  ∀ x y : ℝ, f (x + y) ≥ f x + f y + k * x * y

theorem functional_form {k : ℝ} (f : ℝ → ℝ) (b : ℝ) 
  (h : f_condition f k) : ∀ x, f x = (k / 2) * x^2 + b * x :=
by
  sorry

end functional_form_l22_22546


namespace floor_difference_l22_22415

theorem floor_difference (x : ℝ) (h : x = 13.2) : 
  (⌊x^2⌋ - ⌊x⌋ * ⌊x⌋) = 5 := by
  have h1 : ⌊x⌋ = 13 := by sorry
  have h2 : ⌊x^2⌋ = 174 := by sorry
  calc
  ⌊x^2⌋ - ⌊x⌋ * ⌊x⌋ = 174 - 13 * 13 : by rw [h1, h2]
                 ... = 5 : by norm_num

end floor_difference_l22_22415


namespace charity_event_equation_l22_22636

variable (x : ℕ)

theorem charity_event_equation : x + 5 * (12 - x) = 48 :=
sorry

end charity_event_equation_l22_22636


namespace regular_octagon_interior_angle_measure_l22_22248

theorem regular_octagon_interior_angle_measure :
  ∀ (n : ℕ), n = 8 → (∑ i in finset.range n, (180 : ℝ)) / n = 135 :=
  by
  sorry

end regular_octagon_interior_angle_measure_l22_22248


namespace max_n_for_positive_sum_l22_22908

variables {a : ℕ → ℝ} (d : ℝ) (a_1 : ℝ) (n : ℕ)

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := ∀ n, a (n + 1) = a n + d

theorem max_n_for_positive_sum 
  (h_seq : arithmetic_sequence a d)
  (h_a1_pos : a 1 > 0)
  (h_sum_2011_2012 : a 2011 + a 2012 > 0)
  (h_prod_2011_2012 : a 2011 * a 2012 < 0) :
  ∃ (n : ℕ), S_n (λ k, a (k + 1)) > 0 := sorry

end max_n_for_positive_sum_l22_22908


namespace regular_octagon_interior_angle_l22_22176

theorem regular_octagon_interior_angle : 
  ∀ (n : ℕ), n = 8 → (sum_of_internal_angles n) / n = 135 := 
by
  intros n hn
  rw hn
  sorry

end regular_octagon_interior_angle_l22_22176


namespace contestant_final_score_l22_22952

theorem contestant_final_score 
    (content_score : ℕ)
    (delivery_score : ℕ)
    (weight_content : ℕ)
    (weight_delivery : ℕ)
    (h1 : content_score = 90)
    (h2 : delivery_score = 85)
    (h3 : weight_content = 6)
    (h4 : weight_delivery = 4) : 
    (content_score * weight_content + delivery_score * weight_delivery) / (weight_content + weight_delivery) = 88 := 
sorry

end contestant_final_score_l22_22952


namespace regular_octagon_interior_angle_l22_22103

theorem regular_octagon_interior_angle (n : ℕ) (h₁ : n = 8) :
  (180 * (n - 2)) / n = 135 :=
by
  rw [h₁]
  sorry

end regular_octagon_interior_angle_l22_22103


namespace right_triangle_third_side_product_l22_22949

theorem right_triangle_third_side_product :
  ∃ c₁ c₂ : ℝ, c₁ = real.sqrt (7^2 + 24^2) ∧ c₂ = real.sqrt (24^2 - 7^2) ∧
  (c₁ * c₂ ≈ 574.0) :=
by {
  let c1 := real.sqrt (7^2 + 24^2),
  let c2 := real.sqrt (24^2 - 7^2),
  have h1 : c1 = real.sqrt (7^2 + 24^2) := rfl,
  have h2 : c2 = real.sqrt (24^2 - 7^2) := rfl,
  have h3 : c1 * c2 ≈ 574.0 := sorry,
  exact ⟨c1, c2, h1, h2, h3⟩
}

end right_triangle_third_side_product_l22_22949


namespace zeroes_in_interval_l22_22511

noncomputable def f (x : ℝ) : ℝ := (2 - 1 / (2 * Real.exp 1)) * x ^ 2 - Real.exp x

theorem zeroes_in_interval :
  (∃ x ∈ Ioo (-1 : ℝ) 0, f x = 0) := sorry

end zeroes_in_interval_l22_22511


namespace sum_of_real_solutions_eq_l22_22994

noncomputable def sum_of_real_solutions (a b : ℝ) (h₀ : a > 1) (h₁ : b > 0) : ℝ := 
  ∑ x in {x : ℝ | (sqrt (a - sqrt (a + b^x)) = x)}, x

theorem sum_of_real_solutions_eq (a b : ℝ) (h₀ : a > 1) (h₁ : b > 0) :
  sum_of_real_solutions a b h₀ h₁ = (sqrt (4*a - 3*b) - 1) / 2 := 
sorry

end sum_of_real_solutions_eq_l22_22994


namespace max_buses_in_city_l22_22944

theorem max_buses_in_city (n : ℕ) (stops : ℕ) (shared : ℕ) (condition1 : n = 9) (condition2 : stops = 3) (condition3 : shared ≤ 1) : n = 12 :=
sorry

end max_buses_in_city_l22_22944


namespace min_cost_theater_tickets_l22_22718

open Real

variable (x y : ℝ)

theorem min_cost_theater_tickets :
  (x + y = 140) →
  (y ≥ 2 * x) →
  ∀ x y, 60 * x + 100 * y ≥ 12160 :=
by
  sorry

end min_cost_theater_tickets_l22_22718


namespace floor_sq_minus_sq_floor_l22_22419

noncomputable def floor_13_2 : ℤ := int.floor 13.2

theorem floor_sq_minus_sq_floor :
  int.floor ((13.2 : ℝ) ^ 2) - (floor_13_2 * floor_13_2) = 5 :=
by
  let floor_13_2_sq := floor_13_2 * floor_13_2
  have h1 : int.floor (13.2 : ℝ) = 13 := by norm_num
  have h2 : int.floor ((13.2 : ℝ) ^ 2) = 174 := by norm_num
  rw [h1] at floor_13_2
  rw [h2]
  rw [floor_13_2, floor_13_2, floor_13_2]
  exact sorry

end floor_sq_minus_sq_floor_l22_22419


namespace safe_password_l22_22331

-- Define the digits
def digit_set := {2, 5, 6}

-- Function to create the largest three-digit number from a set of digits
def largest_number (s : Set ℕ) : ℕ :=
  if s = {6, 2, 5} then 652 else 0 -- Assuming digits are always {6, 2, 5}

-- Function to create the smallest three-digit number from a set of digits
def smallest_number (s : Set ℕ) : ℕ :=
  if s = {6, 2, 5} then 256 else 0 -- Assuming digits are always {6, 2, 5}

-- Statement: Prove that the sum of the largest and smallest number from the set {2, 5, 6} is 908
theorem safe_password : largest_number digit_set + smallest_number digit_set = 908 := by
  sorry

end safe_password_l22_22331


namespace largest_prime_factor_sum_of_divisors_180_l22_22990

def sum_of_divisors (n : ℕ) : ℕ :=
  ∏ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d

theorem largest_prime_factor_sum_of_divisors_180 :
  let M := sum_of_divisors 180 in
  ∀ p, nat.prime p → p ∣ M → p ≤ 13 :=
by
  let M := sum_of_divisors 180
  have p_factors : multiset ℕ := (unique_factorization_monoid.factorization M).to_multiset
  exact sorry

end largest_prime_factor_sum_of_divisors_180_l22_22990


namespace least_not_lucky_multiple_of_6_l22_22310

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

theorem least_not_lucky_multiple_of_6 : ∃ k : ℕ, k > 0 ∧ k % 6 = 0 ∧ ¬ is_lucky k ∧ ∀ m : ℕ, m > 0 ∧ m % 6 = 0 ∧ ¬ is_lucky m → k ≤ m :=
  sorry

end least_not_lucky_multiple_of_6_l22_22310


namespace magnitude_of_a_minus_2b_eq_sqrt_5_lambda_perpendicular_to_b_l22_22527

variables (a b : ℝ × ℝ)
variables (m : ℝ)
variables (λ : ℝ)

def a := (1 : ℝ, 0 : ℝ)
def b := (1 : ℝ, 1 : ℝ)

-- 1. Prove that |a - 2b| = sqrt 5
theorem magnitude_of_a_minus_2b_eq_sqrt_5 (ha : a = (1, 0)) (hb : b = (1, 1)) (angle_ab : real.angle (1, 0) (1, 1) = real.pi / 4) :
  real.sqrt ((1 - 2 * 1)^2 + (0 - 2 * 1)^2) = real.sqrt 5 :=
sorry

-- 2. Prove that λ = -1/2 if (a + λb) is perpendicular to b.
theorem lambda_perpendicular_to_b (ha : a = (1, 0)) (hb : b = (1, 1)) (perpendicular : ((1 + λ * 1 : ℝ), (λ : ℝ)) = ((1 + λ), λ) ∧ ((1 + λ), λ) • (1, 1) = 0) :
  λ = -1 / 2 :=
sorry

end magnitude_of_a_minus_2b_eq_sqrt_5_lambda_perpendicular_to_b_l22_22527


namespace cartesian_eq_of_curve_C_distance_AB_l22_22957

variables {α x y ρ θ : ℝ}

theorem cartesian_eq_of_curve_C (h1 : x = sin α + cos α) (h2 : y = sin α - cos α) : 
  x^2 + y^2 = 2 :=
by sorry

theorem distance_AB (h3 : √2 * ρ * sin (π / 4 - θ) + 1 / 2 = 0)
  (h4 : x^2 + y^2 = 2)
  (h5 : ∀ (x y : ℝ), x - y + 1 / 2 = 0) :
  |AB| = (√30) / 2 :=
by sorry

end cartesian_eq_of_curve_C_distance_AB_l22_22957


namespace final_score_is_correct_l22_22950

-- Definitions based on given conditions
def speechContentScore : ℕ := 90
def speechDeliveryScore : ℕ := 85
def weightContent : ℕ := 6
def weightDelivery : ℕ := 4

-- The final score calculation theorem
theorem final_score_is_correct : 
  (speechContentScore * weightContent + speechDeliveryScore * weightDelivery) / (weightContent + weightDelivery) = 88 :=
  by
    sorry

end final_score_is_correct_l22_22950


namespace sin_x_eq_x_div_100_has_63_roots_l22_22536

noncomputable def count_sin_eq_x_div_100_roots : ℕ := 63

theorem sin_x_eq_x_div_100_has_63_roots :
  let f x := sin x - x / 100 in
  (∀ x : ℝ, f x = 0 ↔ x ∈ Icc (-100) 100) ∧
  ∃ S : Finset ℝ, (S.card = count_sin_eq_x_div_100_roots) ∧ (∀ x ∈ S, f x = 0) :=
by sorry

end sin_x_eq_x_div_100_has_63_roots_l22_22536


namespace exterior_angle_decreases_l22_22383

theorem exterior_angle_decreases (n : ℕ) (hn : n ≥ 3) (n' : ℕ) (hn' : n' ≥ n) :
  (360 : ℝ) / n' < (360 : ℝ) / n := by sorry

end exterior_angle_decreases_l22_22383


namespace max_gcd_seq_l22_22053

open Int Nat

-- Defining the sequence a_n
def a (n : ℕ) : ℤ := 100 + 2 * (n ^ 2)

-- Define the maximum value of d_n as the gcd of a_n and a_(n+1)
theorem max_gcd_seq : ∀ n : ℕ, gcd (a n) (a (n + 1)) = 1 := by
  sorry

end max_gcd_seq_l22_22053


namespace negation_of_sin_prop_l22_22647

theorem negation_of_sin_prop :
  ¬ (∀ x : ℝ, x ≥ 0 → sin x ≤ 1) ↔ ∃ x : ℝ, x ≥ 0 ∧ sin x > 1 :=
by
  sorry

end negation_of_sin_prop_l22_22647


namespace nested_radical_expr_eq_3_l22_22349

theorem nested_radical_expr_eq_3 :
  sqrt (1 + 2 * sqrt (1 + 3 * sqrt (1 + ... + 2017 * sqrt (1 + 2018 * 2020)))) = 3 :=
sorry

end nested_radical_expr_eq_3_l22_22349


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22162

-- Define the conditions for the problem
def num_sides : ℕ := 8
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Statement of the problem
theorem measure_of_one_interior_angle_of_regular_octagon : 
  (sum_of_interior_angles num_sides) / num_sides = 135 :=
by
  -- Conditions are defined above
  -- Proof steps are omitted
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22162


namespace regular_octagon_interior_angle_l22_22122

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : 
  let sum_interior_angles := 180 * (n - 2)
  let num_sides := 8
  let one_interior_angle := sum_interior_angles / num_sides
  one_interior_angle = 135 :=
by
  sorry

end regular_octagon_interior_angle_l22_22122


namespace intersection_proof_l22_22472

noncomputable def intersection_of_lines: (ℚ × ℚ) :=
  let (x, y) := (82 / 31, 66 / 31) in
  if (8 * x - 5 * y = 11 ∧ 3 * x + 2 * y = 12) then (x, y)
  else (0, 0)   -- Placeholder in case of incorrect computation

theorem intersection_proof : 
  ∃ (x y : ℚ), (8 * x - 5 * y = 11 ∧ 3 * x + 2 * y = 12) ∧ (x, y) = (82 / 31, 66 / 31) :=
by
  use (82 / 31, 66 / 31)
  split
  { split
    { norm_num }
    { norm_num }
  }
  { refl }

end intersection_proof_l22_22472


namespace max_cookies_l22_22724

theorem max_cookies (chocolate sugar flour eggs : ℕ) (cookie_chocolate cookie_sugar cookie_flour cookie_eggs cookie_num : ℕ) (ella_chocolate ella_sugar ella_flour ella_eggs : ℕ) :
  cookie_chocolate = 1 →
  cookie_sugar = 1 / 2 →
  cookie_eggs = 1 →
  cookie_flour = 1 →
  cookie_num = 4 →
  ella_chocolate = 4 →
  ella_sugar = 3 →
  ella_eggs = 6 →
  ella_flour = 10 →
  (max_cookies_made ella_chocolate ella_sugar ella_eggs ella_flour cookie_chocolate cookie_sugar cookie_eggs cookie_flour cookie_num = 16) :=
begin
  sorry
end

end max_cookies_l22_22724


namespace distance_new_detroit_quantum_vegas_l22_22565

def new_detroit := (0 : ℂ)
def hyper_york := (0 + 1560 * Complex.I : ℂ)
def quantum_vegas := (1300 + 3120 * Complex.I : ℂ)

theorem distance_new_detroit_quantum_vegas : Complex.abs (quantum_vegas - new_detroit) = 3380 := 
by 
  sorry

end distance_new_detroit_quantum_vegas_l22_22565


namespace hyperbola_eqn_l22_22500

theorem hyperbola_eqn (a b : ℝ) (h1 : a ^ 2 + b ^ 2 = 1) 
                      (h2 : (a ^ 2 + b ^ 2) / a ^ 2 = 5) :
                      (5 * x ^ 2 - (5 / 4) * y ^ 2 = 1) := 
begin
  sorry
end

end hyperbola_eqn_l22_22500


namespace solution_set_of_inequality_l22_22997

variable {a : ℝ} (h₁ : a > 0) (h₂ : a ≠ 1) 
def f (x : ℝ) : ℝ := log a (x^2 - 2*x + 3)

theorem solution_set_of_inequality (h₃ : ∃ x, is_maximum (f x)) :
  { x | log a (x^2 - 5*x + 7) > 0 } = { x | 2 < x ∧ x < 3 } :=
sorry

end solution_set_of_inequality_l22_22997


namespace conjugate_in_fourth_quadrant_l22_22857

def complexNumber := complex
def z : complexNumber := (1 + 2 * complex.i) ^ 2 / complex.i
def conjugate_z : complexNumber := complex.conj z

theorem conjugate_in_fourth_quadrant :
  (conjugate_z.re > 0) ∧ (conjugate_z.im < 0) :=
by
  -- z = (1 + 2 * complex.i) ^ 2 / complex.i
  -- conjugate_z = complex.conj z
  -- need to show: conjugate_z.re > 0 ∧ conjugate_z.im < 0
  sorry

end conjugate_in_fourth_quadrant_l22_22857


namespace interior_angle_of_regular_octagon_l22_22109

theorem interior_angle_of_regular_octagon :
  let n := 8 in
  let sum_interior_angles := 180 * (n - 2) in
  let angle := sum_interior_angles / n in
  angle = 135 :=
by
  sorry

end interior_angle_of_regular_octagon_l22_22109


namespace nine_digit_numbers_divisible_by_2_l22_22534

noncomputable def permutations_count_divisible_by_2 (digits: List ℕ) : ℕ :=
  let n := digits.length
  let freqs := digits.foldr (λ d acc, acc.update d (acc.get d + 1)) (Std.HashMap.empty ℕ ℕ)
  let factorial (n : ℕ) : ℕ :=
    if n = 0 then 1 else n * factorial (n - 1)
  factorial n / (freqs.toList.foldr (λ (_, f) acc, acc * factorial f) 1)

theorem nine_digit_numbers_divisible_by_2 :
  permutations_count_divisible_by_2 [1, 1, 1, 3, 5, 5, 2, 2, 2] = 3360 :=
  by
  -- proof steps can be filled here
  sorry

end nine_digit_numbers_divisible_by_2_l22_22534


namespace divide_equilateral_figure_l22_22386

/-- A figure composed of equal equilateral triangles can be divided into two congruent parts
  by a line passing through one of the vertices and the midpoint along the height of one equilateral triangle. -/
theorem divide_equilateral_figure (figure : Type) 
  (is_composed_of_equilateral_triangles : ∀ (t : figure), equilateral_triangle t)
  (has_rotational_symmetry : ∀ (t : figure), rotational_symmetry t) :
  ∃ (line : Line), divides_figure_into_two_congruent_parts figure line :=
sorry

end divide_equilateral_figure_l22_22386


namespace sec_225_eq_neg_sqrt2_l22_22438

theorem sec_225_eq_neg_sqrt2 :
  sec 225 = -sqrt 2 :=
by 
  -- Definitions for conditions
  have h1 : ∀ θ : ℕ, sec θ = 1 / cos θ := sorry,
  have h2 : cos 225 = cos (180 + 45) := sorry,
  have h3 : cos (180 + 45) = -cos 45 := sorry,
  have h4 : cos 45 = 1 / sqrt 2 := sorry,
  -- Required proof statement
  sorry

end sec_225_eq_neg_sqrt2_l22_22438


namespace find_k_l22_22645

theorem find_k 
  (k: ℝ) 
  (h_line: ∀ x y: ℝ, y = k * x + 3) 
  (h_circle: ∀ x y: ℝ, (x - 3) ^ 2 + (y - 2) ^ 2 = 4)
  (h_chord_length: ∀ A B: ℝ × ℝ, dist A B = 2 * sqrt 3):
  k = -3 / 4 := 
sorry

end find_k_l22_22645


namespace sequence_inequality_l22_22894

noncomputable def a : ℕ → ℝ
| 0     := 1
| (n+1) := (a n)^2 / (2 * a n + 1)

theorem sequence_inequality (n : ℕ) : 
  (∑ i in Finset.range (n + 1), (8 * a i / (1 + a i))) < 7 :=
sorry

end sequence_inequality_l22_22894


namespace largest_integer_m_l22_22975

theorem largest_integer_m (n : ℕ) (hn : n > 0) :
  (∃ (x : Fin (2 * n) → ℝ),
    (-1 < x 0) ∧ (x (2 * n - 1) < 1) ∧
    (∀ i j : Fin (2 * n), i < j → x i < x j) ∧
    (∀ (k : ℕ), (1 ≤ k ∧ k ≤ n) →
      (∑ i in Finset.range n,
        (x (Fin.ofNat (2 * i + 1)) ^ (2 * k - 1) - 
         x (Fin.ofNat (2 * i)) ^ (2 * k - 1))) = 1)) →
  ∃ (m : ℕ), m = n :=
by {
  sorry
}

end largest_integer_m_l22_22975


namespace norb_age_is_47_l22_22008

section NorbAge

def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def exactlyHalfGuessesTooLow (guesses : List ℕ) (age : ℕ) : Prop :=
  (guesses.filter (λ x => x < age)).length = (guesses.length / 2)

def oneGuessOffByTwo (guesses : List ℕ) (age : ℕ) : Prop :=
  guesses.any (λ x => x = age + 2 ∨ x = age - 2)

def validAge (guesses : List ℕ) (age : ℕ) : Prop :=
  exactlyHalfGuessesTooLow guesses age ∧ oneGuessOffByTwo guesses age ∧ isPrime age

theorem norb_age_is_47 : validAge [23, 29, 33, 35, 39, 41, 46, 48, 50, 54] 47 :=
sorry

end NorbAge

end norb_age_is_47_l22_22008


namespace quadratic_sum_solutions_l22_22768

theorem quadratic_sum_solutions {a b : ℝ} (h : a ≥ b) (h1: a = 1 + Real.sqrt 17) (h2: b = 1 - Real.sqrt 17) :
  3 * a + 2 * b = 5 + Real.sqrt 17 := by
  sorry

end quadratic_sum_solutions_l22_22768


namespace least_gumballs_to_ensure_five_gumballs_of_same_color_l22_22307

-- Define the number of gumballs for each color
def red_gumballs := 12
def white_gumballs := 10
def blue_gumballs := 11

-- Define the minimum number of gumballs required to ensure five of the same color
def min_gumballs_to_ensure_five_of_same_color := 13

-- Prove the question == answer given conditions
theorem least_gumballs_to_ensure_five_gumballs_of_same_color :
  (red_gumballs + white_gumballs + blue_gumballs) = 33 → min_gumballs_to_ensure_five_of_same_color = 13 :=
by {
  sorry
}

end least_gumballs_to_ensure_five_gumballs_of_same_color_l22_22307


namespace chord_eq_l22_22913

/-- 
If a chord of the ellipse x^2 / 36 + y^2 / 9 = 1 is bisected by the point (4,2),
then the equation of the line on which this chord lies is x + 2y - 8 = 0.
-/
theorem chord_eq {x y : ℝ} (H : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 / 36 + A.2 ^ 2 / 9 = 1) ∧ 
  (B.1 ^ 2 / 36 + B.2 ^ 2 / 9 = 1) ∧ 
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (4, 2)) :
  x + 2 * y = 8 :=
sorry

end chord_eq_l22_22913


namespace nine_digit_numbers_divisible_by_2_l22_22533

noncomputable def permutations_count_divisible_by_2 (digits: List ℕ) : ℕ :=
  let n := digits.length
  let freqs := digits.foldr (λ d acc, acc.update d (acc.get d + 1)) (Std.HashMap.empty ℕ ℕ)
  let factorial (n : ℕ) : ℕ :=
    if n = 0 then 1 else n * factorial (n - 1)
  factorial n / (freqs.toList.foldr (λ (_, f) acc, acc * factorial f) 1)

theorem nine_digit_numbers_divisible_by_2 :
  permutations_count_divisible_by_2 [1, 1, 1, 3, 5, 5, 2, 2, 2] = 3360 :=
  by
  -- proof steps can be filled here
  sorry

end nine_digit_numbers_divisible_by_2_l22_22533


namespace median_reading_time_l22_22715

def students_surveyed := 51
def reading_times := [0.5, 1, 1.5, 2, 2.5]
def students_count := [12, 22, 10, 4, 3]

theorem median_reading_time (students_surveyed = 51) 
  (reading_times = [0.5, 1, 1.5, 2, 2.5]) 
  (students_count = [12, 22, 10, 4, 3]) : 
  median [List.replicate 12 0.5, List.replicate 22 1.0, List.replicate 10 1.5, List.replicate 4 2.0, List.replicate 3 2.5].join = 1 :=
by sorry

end median_reading_time_l22_22715


namespace max_buses_in_city_l22_22935

-- Definition of the problem statement and the maximal number of buses
theorem max_buses_in_city (bus_stops : Finset ℕ) (buses : Finset (Finset ℕ)) 
  (h_stops : bus_stops.card = 9) 
  (h_buses_stops : ∀ b ∈ buses, b.card = 3) 
  (h_shared_stops : ∀ b₁ b₂ ∈ buses, b₁ ≠ b₂ → (b₁ ∩ b₂).card ≤ 1) : 
  buses.card ≤ 12 := 
sorry

end max_buses_in_city_l22_22935


namespace simplify_complex_square_l22_22624

def complex_sq_simplification (a b : ℕ) (i : ℂ) (h : i^2 = -1) : ℂ :=
  (a - b * i)^2

theorem simplify_complex_square : complex_sq_simplification 5 3 complex.I (complex.I_sq) = 16 - 30 * complex.I :=
by sorry

end simplify_complex_square_l22_22624


namespace interior_angle_regular_octagon_l22_22087

theorem interior_angle_regular_octagon (n : ℕ) (h : n = 8) : 
  let sum_of_interior_angles := 180 * (n - 2)
  let number_of_sides := n
  let one_interior_angle := sum_of_interior_angles / number_of_sides
  in one_interior_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l22_22087


namespace find_least_n_l22_22590

noncomputable def b : ℕ → ℕ
| 10 := 10
| (n+1) := if n ≥ 10 then 150 * (b n) + (n+1)^2 else b (n+1)

theorem find_least_n (n : ℕ) (hn : n > 10 ∧ n < 100) (hbn : b n % 121 = 0) : n = 35 :=
by {
  -- Proving this is left as an exercise
  sorry
}

end find_least_n_l22_22590


namespace interior_angle_regular_octagon_l22_22262

theorem interior_angle_regular_octagon : 
  let n := 8 in 
  ∀ (sum_of_interior_angles : ℕ → ℕ) (regular_polygon_interior_angle : ℕ → ℕ),
  sum_of_interior_angles n = (180 * (n - 2)) →
  regular_polygon_interior_angle n = (sum_of_interior_angles n / n) →
  regular_polygon_interior_angle n = 135 :=
by
  intros n sum_of_interior_angles regular_polygon_interior_angle h1 h2
  sorry

end interior_angle_regular_octagon_l22_22262


namespace evaluate_expression_l22_22785

theorem evaluate_expression (a : ℝ) (h : a = 4 / 3) : 
  (4 * a^2 - 12 * a + 9) * (3 * a - 4) = 0 :=
by
  rw [h]
  sorry

end evaluate_expression_l22_22785


namespace interior_angle_regular_octagon_l22_22091

theorem interior_angle_regular_octagon (n : ℕ) (h : n = 8) : 
  let sum_of_interior_angles := 180 * (n - 2)
  let number_of_sides := n
  let one_interior_angle := sum_of_interior_angles / number_of_sides
  in one_interior_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l22_22091


namespace length_OQ_eq_two_line_l2_passes_fixed_point_l22_22855

-- Variables and conditions
variables (O P Q : Point) (n m t : ℝ)
variables (C : Parabola)
variables (l1 l2 : Line)
variables (A B E : Point)

-- Definitions based on given conditions
def origin : Point := {x := 0, y := 0}
def parabola_C : Parabola := { equation := λ x y, y^2 = n*x, n > 0 }
def point_P : Point := {x := 2, y := t}
def distance_P_focus : ℝ := 5/2
def line_tangent_at_P (P : Point) : Line := 
  { equation := λ x y, y - 2 = 1/2 * (x - 2) }

-- Problem's conditions for Q
def tangent_intersects_x_axis (P : Point) : Line := line_tangent_at_P P
def point_Q : Point := tangent_intersects_x_axis P intersects x_axis -- assume we have a way to get this point
def l1_vertical_through_Q : Line := { equation := λ x y, x = Q.x }

-- First part: Prove the length of segment OQ is 2
theorem length_OQ_eq_two :
  distance O Q = 2 := 
  sorry

-- Second part: line l2 passes through a fixed point
def line_l2 := { equation := λ x y, x = m*y + b }

/- The slopes of PA, PE, and PB forming an arithmetic sequence
    - conditions ensuring these slopes are determined in the same way as in the problem
-/

theorem line_l2_passes_fixed_point :
  ∃ (F : Point), ∀ (m b : ℝ), l2 P Q m b intersects parabola_C A B intersects l1 P Q,
  A B P E satisfy_arithmetic_seq_slope → passes_through_fixed (2, 0) := 
  sorry

end length_OQ_eq_two_line_l2_passes_fixed_point_l22_22855


namespace unique_number_not_in_range_l22_22048

-- Define the function g with given conditions
def g (p q r s : ℝ) (x : ℝ) : ℝ := (p * x + q) / (r * x + s)

-- Define the conditions
variables (p q r s : ℝ)
variable (p_nonzero : p ≠ 0)
variable (q_nonzero : q ≠ 0)
variable (r_nonzero : r ≠ 0)
variable (s_nonzero : s ≠ 0)

-- Define the properties of g
axiom g_13_eq_13 : g p q r s 13 = 13
axiom g_31_eq_31 : g p q r s 31 = 31
axiom g_involution : ∀ x : ℝ, x ≠ -s / r → g p q r s (g p q r s x) = x

-- Main theorem stating the unique number not in the range of g
theorem unique_number_not_in_range : ∃ k : ℝ, (k ≠ 22) ∧ (¬ ∃ x : ℝ, g p q r s x = k) :=
sorry

end unique_number_not_in_range_l22_22048


namespace express_as_terminating_decimal_l22_22789

section terminating_decimal

theorem express_as_terminating_decimal
  (a b : ℚ)
  (h1 : a = 125)
  (h2 : b = 144)
  (h3 : b = 2^4 * 3^2): 
  a / b = 0.78125 := 
by 
  sorry

end terminating_decimal

end express_as_terminating_decimal_l22_22789


namespace range_of_a_l22_22891

theorem range_of_a (a : ℝ) : (∃ x : ℝ, a * x^2 + x + 1 < 0) ↔ (a < 1/4) := 
sorry

end range_of_a_l22_22891


namespace sequences_length_16_l22_22766

noncomputable def a : ℕ → ℕ
| 0 := 1
| 1 := 0
| 2 := 1
| (n+3) := a (n+1) + b (n+1) + c (n+1)

noncomputable def b : ℕ → ℕ
| 0 := 0
| 1 := 1
| 2 := 0
| (n+3) := a (n+2) + b (n+1)

noncomputable def c : ℕ → ℕ
| 0 := 1
| 1 := 0
| 2 := 1
| (n+3) := a (n+1) + b (n+1) + c (n+1)

theorem sequences_length_16 : a 16 + b 16 + c 16 = 726 :=
by {
  simp [a, b, c],
  norm_num,
  sorry
}

end sequences_length_16_l22_22766


namespace problem1_problem2_problem3_problem4_l22_22706

-- Problem 1: 27 - 16 + (-7) - 18 = -14
theorem problem1 : 27 - 16 + (-7) - 18 = -14 := 
by 
  sorry

-- Problem 2: (-6) * (-3/4) / (-3/2) = -3
theorem problem2 : (-6) * (-3/4) / (-3/2) = -3 := 
by
  sorry

-- Problem 3: (1/2 - 3 + 5/6 - 7/12) / (-1/36) = 81
theorem problem3 : (1/2 - 3 + 5/6 - 7/12) / (-1/36) = 81 := 
by
  sorry

-- Problem 4: -2^4 + 3 * (-1)^4 - (-2)^3 = -5
theorem problem4 : -2^4 + 3 * (-1)^4 - (-2)^3 = -5 := 
by
  sorry

end problem1_problem2_problem3_problem4_l22_22706


namespace relation_xy_l22_22910

theorem relation_xy (a c b d : ℝ) (x y p : ℝ) 
  (h1 : a^x = c^(3 * p))
  (h2 : c^(3 * p) = b^2)
  (h3 : c^y = b^p)
  (h4 : b^p = d^3) :
  y = 3 * p^2 / 2 :=
by
  sorry

end relation_xy_l22_22910


namespace correct_calculation_l22_22436

theorem correct_calculation :
  -4^2 / (-2)^3 * (-1 / 8) = -1 / 4 := by
  sorry

end correct_calculation_l22_22436


namespace sum_of_squares_of_solutions_l22_22810

theorem sum_of_squares_of_solutions :
  (∀ x, (abs (x^2 - x + (1 / 2010)) = (1 / 2010)) → (x = 0 ∨ x = 1 ∨ (∃ a b, 
  a + b = 1 ∧ a * b = 1 / 1005 ∧ x = a ∧ x = b))) →
  ((0^2 + 1^2) + (1 - 2 * (1 / 1005)) = (2008 / 1005)) :=
by
  intro h
  sorry

end sum_of_squares_of_solutions_l22_22810


namespace find_f_l22_22502

-- Define the derivative of the function
def derivative (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x, deriv f x = f' x

-- Define the given function f(x)
def f (f'_pi_over_2 : ℝ) (x : ℝ) : ℝ :=
  f'_pi_over_2 * sin x + cos x

-- Define the derivative of the given function 
def f' (f'_pi_over_2 : ℝ) (x : ℝ) : ℝ :=
  f'_pi_over_2 * cos x - sin x

-- The conditions from the problem statement
lemma condition1 {x : ℝ} : derivative (f (-1)) (f' (-1)) :=
  by sorry -- Placeholder for the proof that f(x) = -sin(x) + cos(x) has the correct derivative

lemma condition2 : f' (π / 2) (-1) = -1 :=
  by sorry -- Placeholder for the proof f' when x = π / 2

-- The main theorem
theorem find_f'_pi_over_4 : f' (-1) (π / 4) = -real.sqrt 2 :=
  by sorry

end find_f_l22_22502


namespace regular_octagon_interior_angle_l22_22181

theorem regular_octagon_interior_angle : 
  ∀ (n : ℕ), n = 8 → (sum_of_internal_angles n) / n = 135 := 
by
  intros n hn
  rw hn
  sorry

end regular_octagon_interior_angle_l22_22181


namespace even_sum_combinations_count_greater_than_twenty_sum_combinations_count_l22_22747

-- Statement for question 1
theorem even_sum_combinations_count :
    ∃ (count : ℕ), count = 90 ∧ count = (nat.choose 10 2) * 2 :=
begin
  existsi 90,
  split,
  { refl },
  { rw nat.choose, sorry }
end

-- Statement for question 2
theorem greater_than_twenty_sum_combinations_count :
    ∃ (count : ℕ), count = 100 ∧ 
      (count = (nat.choose 10 2) + 
              (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10)) :=
begin
  existsi 100,
  split,
  { refl },
  { rw nat.choose, sorry }
end

end even_sum_combinations_count_greater_than_twenty_sum_combinations_count_l22_22747


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22158

theorem measure_of_one_interior_angle_of_regular_octagon :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 := 
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22158


namespace compound_O_atoms_l22_22720

theorem compound_O_atoms (Cu_weight C_weight O_weight compound_weight : ℝ)
  (Cu_atoms : ℕ) (C_atoms : ℕ) (O_atoms : ℕ)
  (hCu : Cu_weight = 63.55)
  (hC : C_weight = 12.01)
  (hO : O_weight = 16.00)
  (h_compound_weight : compound_weight = 124)
  (h_atoms : Cu_atoms = 1 ∧ C_atoms = 1)
  : O_atoms = 3 :=
sorry

end compound_O_atoms_l22_22720


namespace interior_angle_regular_octagon_l22_22090

theorem interior_angle_regular_octagon (n : ℕ) (h : n = 8) : 
  let sum_of_interior_angles := 180 * (n - 2)
  let number_of_sides := n
  let one_interior_angle := sum_of_interior_angles / number_of_sides
  in one_interior_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l22_22090


namespace max_buses_l22_22928

-- Define the total number of stops and buses as finite sets
def stops : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define a bus as a subset of stops with exactly 3 stops
structure Bus :=
(stops : Finset ℕ)
(h_stops : stops.card = 3)

-- Define the condition that any two buses share at most one stop
def shares_at_most_one (b1 b2 : Bus) : Prop :=
(b1.stops ∩ b2.stops).card ≤ 1

-- Define the predicate for a valid set of buses
def valid_buses (buses : Finset Bus) : Prop :=
∀ b1 b2 ∈ buses, b1 ≠ b2 → shares_at_most_one b1 b2

-- The main theorem statement
theorem max_buses (buses : Finset Bus) (h_valid : valid_buses buses) : buses.card ≤ 12 :=
sorry

end max_buses_l22_22928


namespace BM_eq_CM_l22_22858

noncomputable theory

variables {A B C D K L M : Point}

-- Assuming the geometrical properties and relationships
variables (AB BC CD AK DL AM MD KL : ℝ)

constants (convex : Quadrilateral A B C D)
           (on_AB : PointOnSegment K A B)
           (on_CD : PointOnSegment L C D)
           (constructed_externally : Triangle A M D)
           (KL_equals_2 : KL = 2)

axiom AB_equals_4 : AB = 4
axiom BC_equals_4 : BC = 4
axiom CD_equals_4 : CD = 4
axiom AK_equals_1 : AK = 1
axiom DL_equals_1 : DL = 1
axiom AM_equals_2 : AM = 2
axiom MD_equals_2 : MD = 2

-- To prove that BM = CM
theorem BM_eq_CM : BM = CM :=
by {
  sorry
}

end BM_eq_CM_l22_22858


namespace esteban_exercise_days_l22_22006

theorem esteban_exercise_days
  (natasha_exercise_per_day : ℕ)
  (natasha_days : ℕ)
  (esteban_exercise_per_day : ℕ)
  (total_exercise_hours : ℕ)
  (hours_to_minutes : ℕ)
  (natasha_exercise_total : ℕ)
  (total_exercise_minutes : ℕ)
  (esteban_exercise_total : ℕ)
  (esteban_days : ℕ) :
  natasha_exercise_per_day = 30 →
  natasha_days = 7 →
  esteban_exercise_per_day = 10 →
  total_exercise_hours = 5 →
  hours_to_minutes = 60 →
  natasha_exercise_total = natasha_exercise_per_day * natasha_days →
  total_exercise_minutes = total_exercise_hours * hours_to_minutes →
  esteban_exercise_total = total_exercise_minutes - natasha_exercise_total →
  esteban_days = esteban_exercise_total / esteban_exercise_per_day →
  esteban_days = 9 :=
by
  sorry

end esteban_exercise_days_l22_22006


namespace interior_angle_regular_octagon_l22_22082

theorem interior_angle_regular_octagon (n : ℕ) (h : n = 8) : 
  let sum_of_interior_angles := 180 * (n - 2)
  let number_of_sides := n
  let one_interior_angle := sum_of_interior_angles / number_of_sides
  in one_interior_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l22_22082


namespace nested_radical_simplification_l22_22365

theorem nested_radical_simplification : 
  (sqrt (1 + 2 * sqrt (1 + 3 * sqrt (1 + ⋯ + 2017 * sqrt (1 + 2018 * 2020))))) = 3 :=
by
  -- sorry to skip the proof
  sorry

end nested_radical_simplification_l22_22365


namespace paperclips_in_larger_box_l22_22299

-- Defining the conditions:
def initial_volume : ℝ := 18
def initial_clips : ℝ := 60
def larger_volume : ℝ := 72
def density_decrease : ℝ := 0.1

-- Defining the main theorem:
theorem paperclips_in_larger_box :
  let x := (initial_clips / initial_volume) * larger_volume in
  let x_adjusted := x * (1 - density_decrease) in
  x_adjusted = 216 :=
by
  -- The conditions are defined
  let x := (initial_clips / initial_volume) * larger_volume
  let x_adjusted := x * (1 - density_decrease)
  have h1 : x = (initial_clips / initial_volume) * larger_volume := rfl
  have h2 : x_adjusted = x * (1 - density_decrease) := rfl
  sorry

end paperclips_in_larger_box_l22_22299


namespace salt_mixture_l22_22903

theorem salt_mixture (x y : ℝ) (p c z : ℝ) (hx : x = 50) (hp : p = 0.60) (hc : c = 0.40) (hy_eq : y = 50) :
  (50 * z) + (50 * 0.60) = 0.40 * (50 + 50) → (50 * z) + (50 * p) = c * (x + y) → y = 50 :=
by sorry

end salt_mixture_l22_22903


namespace cylinder_volume_l22_22780

-- Define the dimensions of the rectangle
def width : ℝ := 10
def height : ℝ := 20

-- Define the radius and height of the resulting cylinder
def radius : ℝ := width / 2
def cylinder_height : ℝ := height

-- Define the volume formula for a cylinder
def volume (r h : ℝ) : ℝ := π * r^2 * h

-- The theorem we want to prove:
theorem cylinder_volume : volume radius cylinder_height = 500 * π :=
by 
  sorry

end cylinder_volume_l22_22780


namespace find_value_of_a_l22_22486

noncomputable def log_base_four (a : ℝ) : ℝ := Real.log a / Real.log 4

theorem find_value_of_a (a : ℝ) (h : log_base_four a = (1 : ℝ) / (2 : ℝ)) : a = 2 := by
  sorry

end find_value_of_a_l22_22486


namespace simplify_expression_l22_22621

variable (q : ℚ)

theorem simplify_expression :
  (2 * q^3 - 7 * q^2 + 3 * q - 4) + (5 * q^2 - 4 * q + 8) = 2 * q^3 - 2 * q^2 - q + 4 :=
by
  sorry

end simplify_expression_l22_22621


namespace floor_difference_l22_22414

theorem floor_difference (x : ℝ) (h : x = 13.2) : 
  (⌊x^2⌋ - ⌊x⌋ * ⌊x⌋) = 5 := by
  have h1 : ⌊x⌋ = 13 := by sorry
  have h2 : ⌊x^2⌋ = 174 := by sorry
  calc
  ⌊x^2⌋ - ⌊x⌋ * ⌊x⌋ = 174 - 13 * 13 : by rw [h1, h2]
                 ... = 5 : by norm_num

end floor_difference_l22_22414


namespace asymptotes_eq_line_eq_given_P_product_of_distances_constant_l22_22730

-- Definitions and problem conditions
def hyperbola_eq (x y : ℝ) : Prop := x^2 - y^2 / 4 = 1

def asymptote1 (x y : ℝ) : Prop := y = 2 * x
def asymptote2 (x y : ℝ) : Prop := y = -2 * x

def midpoint (P A B : ℝ × ℝ) : Prop := (P.1 = (A.1 + B.1) / 2) ∧ (P.2 = (A.2 + B.2) / 2)

variables (x0 : ℝ)
variable (P := (x0, 2))

def line_l (x y : ℝ) : Prop := y = 2 * Real.sqrt 2 * x - 2

-- Proof goals
theorem asymptotes_eq :
  asymptote1 x y ∨ asymptote2 x y ↔ y = 2 * x ∨ y = -2 * x := sorry

theorem line_eq_given_P (x0 : ℝ) (hx0 : x0^2 = 2) :
  ∀ x y, line_l x y ↔ y = 2 * Real.sqrt 2 * x - 2 := sorry

theorem product_of_distances_constant (x0 y0 : ℝ)
  (hP : hyperbola_eq x0 y0) :
  let A := (x0 + y0 / 2, 2 * (x0 + y0 / 2)),
      B := (x0 - y0 / 2, -2 * (x0 - y0 / 2)) in
  (Real.sqrt (1 + 4) * |A.1|) * (Real.sqrt (1 + 4) * |B.1|) = 5 := sorry

end asymptotes_eq_line_eq_given_P_product_of_distances_constant_l22_22730


namespace poly_sum_of_coeffs_l22_22553

noncomputable def g (a b c d x : ℂ) : ℂ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem poly_sum_of_coeffs (a b c d : ℝ)
  (h1 : g a b c d (3 * complex.I) = 0)
  (h2 : g a b c d (1 + complex.I) = 0) :
  a + b + c + d = 9 := 
sorry

end poly_sum_of_coeffs_l22_22553


namespace interior_angle_of_regular_octagon_l22_22111

theorem interior_angle_of_regular_octagon :
  let n := 8 in
  let sum_interior_angles := 180 * (n - 2) in
  let angle := sum_interior_angles / n in
  angle = 135 :=
by
  sorry

end interior_angle_of_regular_octagon_l22_22111


namespace number_of_people_chose_pop_l22_22560

theorem number_of_people_chose_pop (total_people surveyed: ℕ) (central_angle_pop: ℤ) 
  (h_total: surveyed = 600) (h_angle: central_angle_pop = 270) :
  ∃ n : ℕ, n = (600 * 3) / 4 ∧ n = 450 :=
by 
  use 450
  split
  {
    exact calc
      450 = (600 * 3) / 4 : by sorry
  }
  exact rfl

end number_of_people_chose_pop_l22_22560


namespace final_number_even_l22_22342

/-
Define the final number calculation function
-/
def final_number : ℕ := 
  let seq := (list.range 64).map (λ n, n + 1) in
  (list.range (seq.length / 2)).foldl (λ acc i, acc + |(seq[2*i] - seq[2*i+1])|) 0

/-
Prove that the final number is even
-/
theorem final_number_even : final_number % 2 = 0 := 
by
  sorry

end final_number_even_l22_22342


namespace angle_is_approximately_9574_degrees_l22_22791

open Real

noncomputable def u : ℝ × ℝ × ℝ := (3, 0, -2)
noncomputable def v : ℝ × ℝ × ℝ := (1, -4, 2)

def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

def magnitude (a : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (a.1^2 + a.2^2 + a.3^2)

def angle_between_vectors (a b : ℝ × ℝ × ℝ) : ℝ :=
  real.arccos ((dot_product a b) / (magnitude a * magnitude b))

def angle_between_u_and_v : ℝ :=
  angle_between_vectors u v

theorem angle_is_approximately_9574_degrees : abs ((angle_between_u_and_v * (180 / π)) - 95.74) < 1e-2 :=
  sorry

end angle_is_approximately_9574_degrees_l22_22791


namespace regular_octagon_angle_measure_l22_22206

theorem regular_octagon_angle_measure :
  let n := 8 in
  (180 * (n - 2)) / n = 135 := 
by
  sorry

end regular_octagon_angle_measure_l22_22206


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22159

theorem measure_of_one_interior_angle_of_regular_octagon :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 := 
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22159


namespace interior_angle_regular_octagon_l22_22268

theorem interior_angle_regular_octagon : 
  let n := 8 in 
  ∀ (sum_of_interior_angles : ℕ → ℕ) (regular_polygon_interior_angle : ℕ → ℕ),
  sum_of_interior_angles n = (180 * (n - 2)) →
  regular_polygon_interior_angle n = (sum_of_interior_angles n / n) →
  regular_polygon_interior_angle n = 135 :=
by
  intros n sum_of_interior_angles regular_polygon_interior_angle h1 h2
  sorry

end interior_angle_regular_octagon_l22_22268


namespace quadrilateral_diagonals_bisect_parallelogram_l22_22686

theorem quadrilateral_diagonals_bisect_parallelogram
  (Q : Type) [quadrilateral Q] (d₁ d₂ : diagonal Q)
  (h : bisecting d₁ d₂) : parallelogram Q :=
begin
  sorry
end

end quadrilateral_diagonals_bisect_parallelogram_l22_22686


namespace isothermal_compression_work_l22_22306

noncomputable def work_done_isothermal_compression (H h R : ℝ) (p0 : ℝ) : ℝ :=
  let S := Real.pi * R^2
  in  p0 * S * H * Real.log (H / (H - h))

theorem isothermal_compression_work :
  work_done_isothermal_compression 0.4 0.3 0.1 103300 ≈ 1799.6 := 
by
  sorry

end isothermal_compression_work_l22_22306


namespace sec_225_eq_neg_sqrt_2_l22_22451

theorem sec_225_eq_neg_sqrt_2 :
  let sec (x : ℝ) := 1 / Real.cos x
  ∧ Real.cos (225 * Real.pi / 180) = Real.cos (180 * Real.pi / 180 + 45 * Real.pi / 180)
  ∧ Real.cos (180 * Real.pi / 180 + 45 * Real.pi / 180) = -Real.cos (45 * Real.pi / 180)
  ∧ Real.cos (45 * Real.pi / 180) = 1 / Real.sqrt 2
  → sec (225 * Real.pi / 180) = -Real.sqrt 2 :=
by 
  intros sec_h cos_h1 cos_h2 cos_h3
  sorry

end sec_225_eq_neg_sqrt_2_l22_22451


namespace max_value_of_b_minus_a_l22_22856

theorem max_value_of_b_minus_a (a b : ℝ) (h1 : a < 0) (h2 : ∀ x : ℝ, a < x ∧ x < b → (3 * x^2 + a) * (2 * x + b) ≥ 0) : b - a ≤ 1 / 3 :=
by
  sorry

end max_value_of_b_minus_a_l22_22856


namespace theater_revenue_l22_22731

/-- 
The theater's ticket prices and sales:
- Matinee tickets cost $5.
- Evening tickets cost $7.
- Opening night tickets cost $10.
- A bucket of popcorn costs $10.
- On Friday, the theater had 32 matinee customers, 40 evening customers, and 58 opening night customers.
- Half the customers bought popcorn.

Question: How much money did the theater make on Friday night?
-/

def matinee_ticket_price := 5
def evening_ticket_price := 7
def opening_night_ticket_price := 10
def popcorn_price := 10
def matinee_customers := 32
def evening_customers := 40
def opening_night_customers := 58
def customers_bought_popcorn (total_customers : ℕ) : ℕ := total_customers / 2

theorem theater_revenue : 
  let total_customers := matinee_customers + evening_customers + opening_night_customers
    in
  let total_ticket_revenue := matinee_customers * matinee_ticket_price +
                              evening_customers * evening_ticket_price +
                              opening_night_customers * opening_night_ticket_price
    in
  let total_popcorn_revenue := customers_bought_popcorn(total_customers) * popcorn_price
    in
  total_ticket_revenue + total_popcorn_revenue = 1670 :=
by 
  sorry -- proof to be filled in

end theater_revenue_l22_22731


namespace max_min_difference_l22_22885

noncomputable def f (x : ℝ) : ℝ := x^3 - 12 * x + 8

theorem max_min_difference : 
  ∃ (M m : ℝ), 
  (∀ x ∈ set.Icc (-3 : ℝ) 3, m ≤ f x ∧ f x ≤ M) ∧ 
  (∃ a ∈ set.Icc (-3 : ℝ) 3, f a = M) ∧ 
  (∃ b ∈ set.Icc (-3 : ℝ) 3, f b = m) ∧ 
  M - m = 32 :=
sorry

end max_min_difference_l22_22885


namespace least_integer_divisors_l22_22644

theorem least_integer_divisors (n m k : ℕ)
  (h_divisors : 3003 = 3 * 7 * 11 * 13)
  (h_form : n = m * 30 ^ k)
  (h_no_div_30 : ¬(30 ∣ m))
  (h_divisor_count : ∀ (p : ℕ) (h : n = p), (p + 1) * (p + 1) * (p + 1) * (p + 1) = 3003)
  : m + k = 104978 :=
sorry

end least_integer_divisors_l22_22644


namespace Brandy_caffeine_intake_l22_22646

theorem Brandy_caffeine_intake :
  let weight := 60
  let recommended_limit_per_kg := 2.5
  let tolerance := 50
  let coffee_cups := 2
  let coffee_per_cup := 95
  let energy_drinks := 4
  let caffeine_per_energy_drink := 120
  let max_safe_caffeine := weight * recommended_limit_per_kg + tolerance
  let caffeine_from_coffee := coffee_cups * coffee_per_cup
  let caffeine_from_energy_drinks := energy_drinks * caffeine_per_energy_drink
  let total_caffeine_consumed := caffeine_from_coffee + caffeine_from_energy_drinks
  max_safe_caffeine - total_caffeine_consumed = -470 := 
by
  sorry

end Brandy_caffeine_intake_l22_22646


namespace find_triplet_l22_22478

noncomputable def triplet_property (p q n : ℕ) [prime p] [prime q] : Prop :=
(p % 2 = 1) ∧ (q % 2 = 1) ∧ (n > 1) ∧ 
(q ^ (n + 2) ≡ 3 ^ (n + 2) [MOD (p ^ n)]) ∧ 
(p ^ (n + 2) ≡ 3 ^ (n + 2) [MOD (q ^ n)])

theorem find_triplet (p q n : ℕ) [hp : prime p] [hq : prime q] :
triplet_property p q n → (p = 3) ∧ (q = 3) ∧ (n ≥ 2) :=
sorry

end find_triplet_l22_22478


namespace sum_of_products_l22_22521

noncomputable def A : Finset ℤ := {-100, -50, -1, 1, 2, 4, 8, 16, 32, 2003}

-- Define a_i as the product of elements for non-empty subsets.
noncomputable def prod_elem (s : Finset ℤ) : ℤ :=
  s.prod id

-- Summation of a_i for all non-empty subsets
noncomputable def sum_prod (A : Finset ℤ) : ℤ :=
  (A.powerset.filter (λ x => x.nonempty)).sum prod_elem

theorem sum_of_products : sum_prod A = -1 :=
sorry

end sum_of_products_l22_22521


namespace max_value_sine_cosine_expression_l22_22798

theorem max_value_sine_cosine_expression :
  ∀ x y z : ℝ, 
    (sin (3 * x) + sin (2 * y) + sin z) * (cos (3 * x) + cos (2 * y) + cos z) ≤ 4.5 :=
by
  intros x y z
  sorry

end max_value_sine_cosine_expression_l22_22798


namespace problem1_problem2_problem3_l22_22339

theorem problem1 (n : ℕ) : 
  (n > 0) → (1 : ℝ) / (n * (n + 1)) = (1 : ℝ) / n - (1 : ℝ) / (n + 1) :=
by sorry

theorem problem2 : 
  (∑ k in finset.range 99, (1 : ℝ) / (k + 1) / (k + 2)) = 99 / 100 :=
by sorry

theorem problem3 : 
  (∑ k in finset.range 1011, (1 : ℝ) / (- (2 * k + 1) * (2 * k + 3))) = -(1011 / 2023) :=
by sorry

end problem1_problem2_problem3_l22_22339


namespace snow_probability_at_least_once_l22_22014

noncomputable def prob_snow (day : ℕ) (previous_snow : bool) : ℝ :=
if day ≤ 5 then
  if previous_snow then 1/2 else 1/4
else
  1/3

noncomputable def prob_no_snow_all_days : ℝ :=
(3/4)^5 * (2/3)^3

noncomputable def prob_snow_at_least_once : ℝ :=
1 - prob_no_snow_all_days

theorem snow_probability_at_least_once :
  prob_snow_at_least_once = 0.98242 :=
by
  sorry

end snow_probability_at_least_once_l22_22014


namespace regular_octagon_interior_angle_l22_22104

theorem regular_octagon_interior_angle (n : ℕ) (h₁ : n = 8) :
  (180 * (n - 2)) / n = 135 :=
by
  rw [h₁]
  sorry

end regular_octagon_interior_angle_l22_22104


namespace interior_angle_regular_octagon_l22_22260

theorem interior_angle_regular_octagon : 
  let n := 8 in 
  ∀ (sum_of_interior_angles : ℕ → ℕ) (regular_polygon_interior_angle : ℕ → ℕ),
  sum_of_interior_angles n = (180 * (n - 2)) →
  regular_polygon_interior_angle n = (sum_of_interior_angles n / n) →
  regular_polygon_interior_angle n = 135 :=
by
  intros n sum_of_interior_angles regular_polygon_interior_angle h1 h2
  sorry

end interior_angle_regular_octagon_l22_22260


namespace modulus_z1_div_of_10i_and_z1_l22_22762

-- Define the properties of the complex numbers and the imaginary unit
def complex_modulus (a b : ℝ) : ℝ := real.sqrt (a^2 + b^2)

noncomputable def complex_div (a b c d : ℝ) : ℂ :=
  let numerator := complex.mk 0 a * complex.conj (complex.mk c d)
  let denominator := complex.norm_sq (complex.mk c d)
  numerator / denominator

-- Define the specified complex numbers using Lean's complex number type
def z1 : ℂ := complex.mk 3 (-1)
def z2 : ℂ := complex.mk 0 10

-- State the theorems to be proven
theorem modulus_z1 : complex_modulus 3 (-1) = real.sqrt 10 := by
  sorry

theorem div_of_10i_and_z1 : complex_div 0 10 3 (-1) = complex.mk (-1) 3 := by
  sorry

end modulus_z1_div_of_10i_and_z1_l22_22762


namespace sum_odd_impossible_l22_22541

theorem sum_odd_impossible (n m : ℤ) (h : (n^2 + m^2) % 2 = 1) :
  (n + m) % 2 ≠ 0 :=
begin
  sorry
end

end sum_odd_impossible_l22_22541


namespace regular_octagon_angle_measure_l22_22212

theorem regular_octagon_angle_measure :
  let n := 8 in
  (180 * (n - 2)) / n = 135 := 
by
  sorry

end regular_octagon_angle_measure_l22_22212


namespace cost_of_largest_pot_l22_22001

theorem cost_of_largest_pot
  (total_cost : ℝ)
  (n : ℕ)
  (a b : ℝ)
  (h_total_cost : total_cost = 7.80)
  (h_n : n = 6)
  (h_b : b = 0.25)
  (h_small_cost : ∃ x : ℝ, ∃ is_odd : ℤ → Prop, (∃ c: ℤ, x = c / 100 ∧ is_odd c) ∧
                  total_cost = x + (x + b) + (x + 2 * b) + (x + 3 * b) + (x + 4 * b) + (x + 5 * b)) :
  ∃ y, y = (x + 5*b) ∧ y = 1.92 :=
  sorry

end cost_of_largest_pot_l22_22001


namespace algebraic_inequality_l22_22852

variable {X : Type*} [OrderedField X]

theorem algebraic_inequality (x y z : X) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  x / (y * z) + y / (z * x) + z / (x * y) ≥ 1 / x + 1 / y + 1 / z := sorry

end algebraic_inequality_l22_22852


namespace option_A_correct_option_B_incorrect_option_C_correct_option_D_correct_l22_22875

theorem option_A_correct (P A B C : Point) (h_orthocenter : Orthocenter P A B C) 
    (h_dot : dot (AB A B) (AC A C) = 2) : 
    dot (AP A P) (AB A B) = 2 := sorry

theorem option_B_incorrect (P A B C : Point) (h_eq_triangle : EquilateralTriangle A B C 2) : 
    ∃ (v : Float), v = min_value (dot (PA P A) (PB P B + PC P C)) (v = -1) := sorry

theorem option_C_correct (P A B C : Point) (h_acute_triangle : AcuteTriangle A B C) 
    (h_circumcenter : Circumcenter P A B C) (x y : Float) 
    (h_vector : vector_eq (AP A P) (x * AB A B + y * AC A C)) 
    (h_sum_coeff : x + 2 * y = 1) : 
    length (AB A B) = length (BC B C) := sorry

theorem option_D_correct (P A B C : Point) (h_locus : locus_eq (AP A P) 
    ((1 / (length (AB A B) * cos ((∠ B A C))) + 1/2) * AB A B + 
    (1 / (length (AC A C) * cos ((∠ C A B))) + 1/2) * AC A C)) : 
    locus_passing_circumcenter P A B C := sorry

end option_A_correct_option_B_incorrect_option_C_correct_option_D_correct_l22_22875


namespace books_arrangement_l22_22397

theorem books_arrangement:
  let books := (3, 3, 3, 3)  -- (Russian, English, Spanish, French)
  let r_unit := 3             -- Russian books grouped as a unit
  let s_unit := 3             -- Spanish books grouped as a unit
  let f_unit := 3             -- French books grouped as a unit
  let e_books := 3            -- English books not grouped
  (r_unit + e_books + s_unit + f_unit)! * (r_unit! * s_unit! * f_unit!) = 155520 := by
  sorry

end books_arrangement_l22_22397


namespace parallelogram_perimeter_eq_60_l22_22917

-- Given conditions from the problem
variables (P Q R M N O : Type*)
variables (PQ PR QR PM MN NO PO : ℝ)
variables {PQ_eq_PR : PQ = PR}
variables {PQ_val : PQ = 30}
variables {PR_val : PR = 30}
variables {QR_val : QR = 28}
variables {MN_parallel_PR : true}  -- Parallel condition we can treat as true for simplification
variables {NO_parallel_PQ : true}  -- Another parallel condition treated as true

-- Statement of the problem to be proved
theorem parallelogram_perimeter_eq_60 :
  PM + MN + NO + PO = 60 :=
sorry

end parallelogram_perimeter_eq_60_l22_22917


namespace range_of_a_l22_22550

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) 
(h_def : ∀ x, f x = sqrt (x^2 - 1) + sqrt (a - x^2)) 
(h_even : ∀ x, f (-x) = f x)
(h_not_odd : ∀ x, f (-x) ≠ -f x) :
	a > 1 :=
sorry

end range_of_a_l22_22550


namespace regular_octagon_interior_angle_measure_l22_22252

theorem regular_octagon_interior_angle_measure :
  ∀ (n : ℕ), n = 8 → (∑ i in finset.range n, (180 : ℝ)) / n = 135 :=
  by
  sorry

end regular_octagon_interior_angle_measure_l22_22252


namespace find_value_of_f2_plus_g3_l22_22999

def f (x : ℝ) : ℝ := 3 * x - 2
def g (x : ℝ) : ℝ := x^2 + 2

theorem find_value_of_f2_plus_g3 : f (2 + g 3) = 37 :=
by
  simp [f, g]
  norm_num
  done

end find_value_of_f2_plus_g3_l22_22999


namespace positive_integers_in_range_l22_22393

theorem positive_integers_in_range (x : ℕ) : 
  {x : ℕ | 144 ≤ x^2 ∧ x^2 ≤ 289}.toFinset.card = 6 :=
by
  sorry

end positive_integers_in_range_l22_22393


namespace expansion_dissimilar_terms_l22_22570

theorem expansion_dissimilar_terms (a b c d : ℕ) :
  let P := (a + b + c + d) in
  P ^ 7 = 120 := 
by
  let n := 7
  let k := 4
  have h : (n + k - 1).choose (k - 1) = 120 := by sorry
  exact h

end expansion_dissimilar_terms_l22_22570


namespace probability_interval_l22_22506

-- Definitions based on the given conditions
noncomputable def ξ : MeasureTheory.ProbabilitySpace ℝ :=
  MeasureTheory.gaussian 0 (σ^2)

axiom normal_distribution_symmetry {σ : ℝ} (hσ : 0 < σ) :
  P(ξ > 2) = 0.023

-- The proof problem in Lean 4 statement
theorem probability_interval {σ : ℝ} (hσ : 0 < σ) :
  P(-2 ≤ ξ ∧ ξ ≤ 2) = 0.954 := by
  -- the proof is to be filled in
  sorry

end probability_interval_l22_22506


namespace nested_radical_value_l22_22355

theorem nested_radical_value :
  (nat.sqrt (1 + 2 * nat.sqrt (1 + 3 * nat.sqrt (1 + ... + 2017 * nat.sqrt (1 + 2018 * 2020))))) = 3 :=
sorry

end nested_radical_value_l22_22355


namespace number_of_ordered_pairs_l22_22840

def count_valid_pairs : ℕ :=
  ∑ y in Finset.range 149, (150 - y) / (y * (y + 2))

theorem number_of_ordered_pairs :
  count_valid_pairs = ∑ y in Finset.range 149, (150 - y) / (y * (y + 2)) := by
  sorry

end number_of_ordered_pairs_l22_22840


namespace find_ratio_l22_22834

-- Definitions based on the problem conditions
def hyperbola (a b x y : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1
def angle_between_asymptotes_45 (a b : ℝ) := ∀ θ : ℝ, tan θ = (2 * (b/a) / (1 - (b/a)^2)) → θ = π / 4

-- Theorem statement proving that if conditions hold, the result is as expected
theorem find_ratio (a b : ℝ) (h_hyp : hyperbola a b x y) (h_ineq : a > b) (h_angle : angle_between_asymptotes_45 a b) :
  a / b = 1 / (-1 + Real.sqrt 2) :=
by
  sorry

end find_ratio_l22_22834


namespace sum_real_roots_eq_sqrt3_l22_22477

theorem sum_real_roots_eq_sqrt3 : 
  polynomial.sum_roots (polynomial.C 1 * polynomial.X^4 + polynomial.C (-6) * polynomial.X + polynomial.C (-1)) = real.sqrt 3 :=
sorry

end sum_real_roots_eq_sqrt3_l22_22477


namespace determine_a_l22_22516

noncomputable theory

def inequality (a : ℝ) (x : ℝ) : Prop := (a * x - 1) * (x + 1) < 0

def solution_set (x : ℝ) : Prop := (x < -1) ∨ (-1/2 < x)

theorem determine_a (a : ℝ) : (∀ x : ℝ, inequality a x ↔ solution_set x) → a = -2 :=
by
  intros h
  sorry

end determine_a_l22_22516


namespace nested_radical_value_l22_22353

theorem nested_radical_value :
  (nat.sqrt (1 + 2 * nat.sqrt (1 + 3 * nat.sqrt (1 + ... + 2017 * nat.sqrt (1 + 2018 * 2020))))) = 3 :=
sorry

end nested_radical_value_l22_22353


namespace height_of_balloon_is_correct_l22_22833

noncomputable def balloon_height : ℝ :=
  let c := 120 / 2 -- Since c and d are equal due to symmetry
  in let d := 120 / 2
  in let HC := 160
  in let HD := 140
  in Real.sqrt ((HC^2 + HD^2 - 120^2) / 2)

theorem height_of_balloon_is_correct :
  balloon_height = Real.sqrt 15400 :=
by
  sorry

end height_of_balloon_is_correct_l22_22833


namespace floor_abs_of_neg_num_l22_22404

theorem floor_abs_of_neg_num : ((Real.floor (| -57.6 |)) = 57) := 
by
  sorry

end floor_abs_of_neg_num_l22_22404


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22203

theorem measure_of_one_interior_angle_of_regular_octagon 
  (n : Nat) (h_n : n = 8) : 
  (one_interior_angle : ℝ) = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22203


namespace sum_of_squares_of_solutions_l22_22830

theorem sum_of_squares_of_solutions :
  (∑ x in {x : ℝ | | x^2 - x + 1/2010 | = 1/2010}, x^2) = 2008 / 1005 :=
by
  sorry

end sum_of_squares_of_solutions_l22_22830


namespace regular_octagon_interior_angle_l22_22105

theorem regular_octagon_interior_angle (n : ℕ) (h₁ : n = 8) :
  (180 * (n - 2)) / n = 135 :=
by
  rw [h₁]
  sorry

end regular_octagon_interior_angle_l22_22105


namespace interior_angle_of_regular_octagon_l22_22119

theorem interior_angle_of_regular_octagon :
  let n := 8 in
  let sum_interior_angles := 180 * (n - 2) in
  let angle := sum_interior_angles / n in
  angle = 135 :=
by
  sorry

end interior_angle_of_regular_octagon_l22_22119


namespace length_of_BD_l22_22584

theorem length_of_BD 
  (A B C D : Point)
  (hABC : right_triangle A B C)
  (hCircle : circle_diameter_intersects B C A C D)
  (hAreaABC : area A B C = 180)
  (hAC : distance A C = 30) : 
  distance B D = 12 :=
by
  sorry

end length_of_BD_l22_22584


namespace knights_reduce_41_fact_to_zero_l22_22668

-- Define the knight attack functions
def LancelotAttack (x : ℕ) := if x % 2 == 0 then (x / 2) - 1 else x
def GawainAttack (x : ℕ) := if (x % 3 == 0) && (x / 3 > 2) then 2 * (x / 3 - 1) else x
def PercivalAttack (x : ℕ) := if (x % 4 == 0) && (x / 4 > 3) then 3 * (x / 4 - 1) else x

-- The main theorem
theorem knights_reduce_41_fact_to_zero : 
  ∃ (f : (ℕ → ℕ)), (f = LancelotAttack ∨ f = GawainAttack ∨ f = PercivalAttack) → 
  ∀ x, x = 41! → (∃ n, (f^[n] x = 0)) := 
sorry

end knights_reduce_41_fact_to_zero_l22_22668


namespace sin_2012_eq_neg_sin_32_l22_22290

theorem sin_2012_eq_neg_sin_32 : Real.sin (2012 * Real.pi / 180) = - Real.sin (32 * Real.pi / 180) :=
by
  sorry

end sin_2012_eq_neg_sin_32_l22_22290


namespace nested_radical_value_l22_22371

theorem nested_radical_value : 
  let rec nest k :=
    if k = 1 then 2019 else sqrt (1 + k * (k + 2)) in
  nest 2018 = 3 :=
by
  intros
  sorry

end nested_radical_value_l22_22371


namespace regular_octagon_interior_angle_l22_22225

theorem regular_octagon_interior_angle:
  let n := 8
  let sum_interior_angles (n: ℕ) := 180 * (n - 2)
  let interior_angle (n: ℕ) := sum_interior_angles n / n
  interior_angle n = 135 :=
by
  sorry

end regular_octagon_interior_angle_l22_22225


namespace find_length_AD_l22_22346

noncomputable def problem_statement : Prop :=
  ∃ (B D O : Type) [distinct_points B D (line AC)]
    (triangles_congruent : congruent_triangles ABC ADC)
    (intersection_point : O = intersection AD BC)
    (angle_measure : ∠COD = 120°)
    (is_perpendicular : is_altitude OH AC)
    (AH : ℝ) (AH_eq : AH = 10)
    (OB : ℝ) (OB_eq : OB = 8),
    length AD = 28

theorem find_length_AD : problem_statement :=
begin
  sorry
end

end find_length_AD_l22_22346


namespace max_buses_in_city_l22_22941

theorem max_buses_in_city (n : ℕ) (stops : ℕ) (shared : ℕ) (condition1 : n = 9) (condition2 : stops = 3) (condition3 : shared ≤ 1) : n = 12 :=
sorry

end max_buses_in_city_l22_22941


namespace apply_functions_to_obtain_2010_l22_22966

def f (x : ℝ) : ℝ := 1 / x

noncomputable def g (x : ℝ) : ℝ := x / real.sqrt (1 + x^2)

theorem apply_functions_to_obtain_2010 :
  (∃ n : ℕ, n = 2010^2 - 1 ∧ f (nat.iterate g n 1) = 2010) :=
sorry

end apply_functions_to_obtain_2010_l22_22966


namespace quadrilateral_area_l22_22792

theorem quadrilateral_area (d h1 h2 : ℤ) (hd : d = 26) (hh1 : h1 = 9) (hh2 : h2 = 6) :
  1 / 2 * d * h1 + 1 / 2 * d * h2 = 195 :=
by
  have hd' : (d : ℚ) = 26 := by rw [hd]
  have hh1' : (h1 : ℚ) = 9 := by rw [hh1]
  have hh2' : (h2 : ℚ) = 6 := by rw [hh2]
  calc
    1 / 2 * d * h1 + 1 / 2 * d * h2
        = 1 / 2 * 26 * 9 + 1 / 2 * 26 * 6 : by rw [hd', hh1', hh2']
    ... = (13 * 9 + 13 * 6 : ℚ)         : by norm_num
    ... = 195                           : by norm_num


end quadrilateral_area_l22_22792


namespace interior_angle_regular_octagon_l22_22272

theorem interior_angle_regular_octagon : 
  let n := 8 in 
  ∀ (sum_of_interior_angles : ℕ → ℕ) (regular_polygon_interior_angle : ℕ → ℕ),
  sum_of_interior_angles n = (180 * (n - 2)) →
  regular_polygon_interior_angle n = (sum_of_interior_angles n / n) →
  regular_polygon_interior_angle n = 135 :=
by
  intros n sum_of_interior_angles regular_polygon_interior_angle h1 h2
  sorry

end interior_angle_regular_octagon_l22_22272


namespace hyperbola_real_axis_length_l22_22549

theorem hyperbola_real_axis_length (P : ℝ × ℝ) (a b λ : ℝ)
  (h_passes : P = (5, -2))
  (h_asymptote1 : ∀ x y, x - 2 * y = 0 → (x, y) = (0, 0))
  (h_asymptote2 : ∀ x y, x + 2 * y = 0 → (x, y) = (0, 0))
  (h_hyperbola : ∀ x y, x^2 - 4 * y^2 = λ)
  (h_nonzero : λ ≠ 0)
  (h_P_on_hyperbola : (5 : ℝ)^2 - 4 * (-2 : ℝ)^2 = λ) :
  2 * a = 6 :=
by
  sorry

end hyperbola_real_axis_length_l22_22549


namespace scientific_notation_470000000_l22_22010

theorem scientific_notation_470000000 : 470000000 = 4.7 * 10^8 :=
by
  sorry

end scientific_notation_470000000_l22_22010


namespace range_of_a_plus_abs_b_l22_22866

theorem range_of_a_plus_abs_b (a b : ℝ)
  (h1 : -1 ≤ a) (h2 : a ≤ 3)
  (h3 : -5 < b) (h4 : b < 3) :
  -1 ≤ a + |b| ∧ a + |b| < 8 := by
sorry

end range_of_a_plus_abs_b_l22_22866


namespace floor_of_abs_of_neg_57point6_l22_22402

theorem floor_of_abs_of_neg_57point6 : floor (|(-57.6 : ℝ)|) = 57 := by
  sorry

end floor_of_abs_of_neg_57point6_l22_22402


namespace range_of_a_l22_22047

noncomputable def f (x a : ℝ) : ℝ := exp x * (x^2 + 2*a*x + 2)

theorem range_of_a {a : ℝ} : 
  (∀ x : ℝ, 0 ≤ deriv (λ x, f x a) x) ↔ (-1 ≤ a ∧ a ≤ 1) := 
begin
  sorry
end

end range_of_a_l22_22047


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22135

theorem measure_of_one_interior_angle_of_regular_octagon : 
  let n := 8 in 
  let sum_of_interior_angles := 180 * (n - 2) in
  let measure_of_one_interior_angle := sum_of_interior_angles / n in 
  measure_of_one_interior_angle = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22135


namespace vertices_sum_zero_l22_22959

theorem vertices_sum_zero
  (a b c d e f g h : ℝ)
  (h1 : a = (b + e + d) / 3)
  (h2 : b = (c + f + a) / 3)
  (h3 : c = (d + g + b) / 3)
  (h4 : d = (a + h + e) / 3)
  :
  (a + b + c + d) - (e + f + g + h) = 0 :=
by
  sorry

end vertices_sum_zero_l22_22959


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22194

theorem measure_of_one_interior_angle_of_regular_octagon 
  (n : Nat) (h_n : n = 8) : 
  (one_interior_angle : ℝ) = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22194


namespace larger_of_two_numbers_l22_22052

theorem larger_of_two_numbers
  (A B hcf : ℕ)
  (factor1 factor2 : ℕ)
  (h_hcf : hcf = 23)
  (h_factor1 : factor1 = 9)
  (h_factor2 : factor2 = 10)
  (h_lcm : (A * B) / (hcf) = (hcf * factor1 * factor2))
  (h_A : A = hcf * 9)
  (h_B : B = hcf * 10) :
  max A B = 230 := by
  sorry

end larger_of_two_numbers_l22_22052


namespace number_of_divisible_permutations_l22_22531

def digit_list := [1, 3, 1, 1, 5, 2, 1, 5, 2]
def count_permutations (d : List ℕ) (n : ℕ) : ℕ :=
  let fact := Nat.factorial
  let number := fact 8 / (fact 3 * fact 2 * fact 1)
  number

theorem number_of_divisible_permutations : count_permutations digit_list 2 = 3360 := by
  sorry

end number_of_divisible_permutations_l22_22531


namespace remainder_when_square_divided_l22_22585

theorem remainder_when_square_divided (n : ℕ) (a : ℤ) (h1 : a * a ≡ 1 [MOD n]) (h2 : (a + 1) * (a + 1) ≡ 1 [MOD n]) : (a + 1) ^ 2 ≡ 1 [MOD n] := 
by
  sorry

end remainder_when_square_divided_l22_22585


namespace nested_radical_value_l22_22356

theorem nested_radical_value :
  (nat.sqrt (1 + 2 * nat.sqrt (1 + 3 * nat.sqrt (1 + ... + 2017 * nat.sqrt (1 + 2018 * 2020))))) = 3 :=
sorry

end nested_radical_value_l22_22356


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22154

theorem measure_of_one_interior_angle_of_regular_octagon :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 := 
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22154


namespace initial_guinea_fowls_l22_22325

theorem initial_guinea_fowls (initial_chickens initial_turkeys : ℕ) 
  (initial_guinea_fowls : ℕ) (lost_chickens lost_turkeys lost_guinea_fowls : ℕ) 
  (total_birds_end : ℕ) (days : ℕ)
  (hc : initial_chickens = 300) (ht : initial_turkeys = 200) 
  (lc : lost_chickens = 20) (lt : lost_turkeys = 8) (lg : lost_guinea_fowls = 5) 
  (d : days = 7) (tb : total_birds_end = 349) :
  initial_guinea_fowls = 80 := 
by 
  sorry

end initial_guinea_fowls_l22_22325


namespace sum_of_squares_of_solutions_l22_22831

theorem sum_of_squares_of_solutions :
  (∑ x in {x : ℝ | | x^2 - x + 1/2010 | = 1/2010}, x^2) = 2008 / 1005 :=
by
  sorry

end sum_of_squares_of_solutions_l22_22831


namespace sequence_contains_perfect_square_l22_22773

def largest_prime_factor (n : ℕ) : ℕ :=
  sorry  -- This function should calculate the largest prime factor of n

def sequence (a1 : ℕ) (n : ℕ) : ℕ :=
  nat.rec_on n a1 (λ n' an, an + largest_prime_factor an)

theorem sequence_contains_perfect_square (a1 : ℕ) (h : a1 > 1) :
  ∃ n, ∃ k, sequence a1 n = k^2 :=
begin
  sorry
end

end sequence_contains_perfect_square_l22_22773


namespace sec_225_eq_neg_sqrt_2_l22_22454

theorem sec_225_eq_neg_sqrt_2 :
  let sec (x : ℝ) := 1 / Real.cos x
  ∧ Real.cos (225 * Real.pi / 180) = Real.cos (180 * Real.pi / 180 + 45 * Real.pi / 180)
  ∧ Real.cos (180 * Real.pi / 180 + 45 * Real.pi / 180) = -Real.cos (45 * Real.pi / 180)
  ∧ Real.cos (45 * Real.pi / 180) = 1 / Real.sqrt 2
  → sec (225 * Real.pi / 180) = -Real.sqrt 2 :=
by 
  intros sec_h cos_h1 cos_h2 cos_h3
  sorry

end sec_225_eq_neg_sqrt_2_l22_22454


namespace mira_jogs_hours_each_morning_l22_22005

theorem mira_jogs_hours_each_morning 
  (h : ℝ) -- number of hours Mira jogs each morning
  (speed : ℝ) -- Mira's jogging speed in miles per hour
  (days : ℝ) -- number of days Mira jogs
  (total_distance : ℝ) -- total distance Mira jogs

  (H1 : speed = 5) 
  (H2 : days = 5) 
  (H3 : total_distance = 50) 
  (H4 : total_distance = speed * h * days) :

  h = 2 :=
by
  sorry

end mira_jogs_hours_each_morning_l22_22005


namespace shaded_area_of_triangles_l22_22025

/-- A rectangle PQRS with given dimensions and specific points T, U, V on its sides.
    Proving the area of the shaded region within triangles PTU and URV is 12. -/
theorem shaded_area_of_triangles
  (P Q R S T U V : Point ℝ)
  (hPQ : distance P Q = 5)
  (hQR : distance Q R = 6)
  (hPT : distance P T = 2)
  (hTR : distance T R = 2)
  (hPU : distance P U = 1.5)
  (hUQ : distance U Q = 1.5)
  (hQV : distance Q V = 3)
  (hVS : distance V S = 3)
  (hPR : line_through P R)
  (hPQ_TS : line_through P Q ∧ line_through Q S ∧ line_through T S)
  (hT_PR : distance P T + distance T R = distance P R)
  (hU_PQ : distance P U + distance U Q = distance P Q)
  (hV_QS : distance Q V + distance V S = distance Q S)
  : area (Triangle P T V) + area (Triangle U R V) = 12 :=
sorry

end shaded_area_of_triangles_l22_22025


namespace roof_angle_with_north_l22_22054

theorem roof_angle_with_north (south_tilt west_tilt : ℝ) (h_south_tilt : south_tilt = 30) (h_west_tilt : west_tilt = 15) : 
  ∃ α, α = 65.1 ∧ α = arctan ((1 / (tan (west_tilt * pi / 180))) / (cos (south_tilt * pi / 180))) * 180 / pi :=
by
  use 65.1
  simp [Real.atan, Real.tan, Real.cos]
  rw [←Real.mul_div_cancel, mul_comm]
  sorry

end roof_angle_with_north_l22_22054


namespace tangent_length_from_origin_l22_22381

-- Given points
def A : ℝ × ℝ := (5, 6)
def B : ℝ × ℝ := (8, 10)
def C : ℝ × ℝ := (7, 15)
def O : ℝ × ℝ := (0, 0)

-- Definitions for the center D, radius R and distance d
noncomputable def D : ℝ × ℝ := sorry -- Center of the circle through A, B, C
noncomputable def R : ℝ := sorry -- Radius of circle through A, B, C from center D
noncomputable def d : ℝ := sorry -- Distance from O to D

-- The target theorem
theorem tangent_length_from_origin
  (h1 : R = dist D A) -- Radius is the distance from D to A
  (h2 : D = (something something solving intersection)) -- Solution to perpendicular bisector equations
  (h3 : d = dist O D): 
  dist O D = sqrt (R^2 - d^2) :=
sorry

end tangent_length_from_origin_l22_22381


namespace arithmetic_sequence_sum_l22_22548

theorem arithmetic_sequence_sum (a b : ℝ) (h : 0.1 : (List ℝ)  := [0, a, b, 5]) : a + b = 5 :=
begin
  -- Since 0, a, b, 5 form an arithmetic sequence
  -- The property of an arithmetic sequence gives us: 0 + 5 = a + b
  sorry
end

end arithmetic_sequence_sum_l22_22548


namespace find_a_l22_22876

theorem find_a (a : ℝ) (h : (a + 3) = 0) : a = -3 :=
by sorry

end find_a_l22_22876


namespace triangle_partition_l22_22608

theorem triangle_partition (n : ℕ) (h : n ≥ 6) : 
  ∀ (T : Triangle), ∃ (Ts : list IsoscelesTriangle), Ts.length = n :=
by
  sorry

end triangle_partition_l22_22608


namespace nested_radical_value_l22_22375

theorem nested_radical_value : 
  let rec nest k :=
    if k = 1 then 2019 else sqrt (1 + k * (k + 2)) in
  nest 2018 = 3 :=
by
  intros
  sorry

end nested_radical_value_l22_22375


namespace centroid_of_equal_areas_l22_22291

open Set

variable {A B C P D E F : Type}
variables [LinearOrder A] [LinearOrder B] [LinearOrder C] [LinearOrder P] [LinearOrder D] [LinearOrder E] [LinearOrder F]

/-- Given a triangle ABC with a point P inside it, extending lines AP, BP, CP to meet BC, AC, AB at points D, E, F respectively,
    if the areas of triangles APF, BPD, and CPE are equal, then P is the centroid of triangle ABC. -/
theorem centroid_of_equal_areas
  (ABC : Triangle ℝ)
  (P : Point ℝ)
  (D E F : Point ℝ)
  (h1 : D ∈ line_through (BC ABC) (P))
  (h2 : E ∈ line_through (AC ABC) (P))
  (h3 : F ∈ line_through (AB ABC) (P))
  (h4 : area (Triangle.mk (A ABC) (P) (F)) = area (Triangle.mk (B ABC) (P) (D)) ∧
         area (Triangle.mk (B ABC) (P) (D)) = area (Triangle.mk (C ABC) (P) (E)))
  : is_centroid (ABC) P := 
sorry

end centroid_of_equal_areas_l22_22291


namespace find_value_of_expression_l22_22832

theorem find_value_of_expression 
  (x y z w : ℤ)
  (hx : x = 3)
  (hy : y = 2)
  (hz : z = 4)
  (hw : w = -1) :
  x^2 * y - 2 * x * y + 3 * x * z - (x + y) * (y + z) * (z + w) = -48 :=
by
  sorry

end find_value_of_expression_l22_22832


namespace regular_octagon_interior_angle_l22_22242

theorem regular_octagon_interior_angle : 
  ∀ n : ℕ, n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  intros n hn
  simp [sum_of_interior_angles]
  sorry

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
  180 * (n - 2)

end regular_octagon_interior_angle_l22_22242


namespace nested_radical_value_l22_22373

theorem nested_radical_value : 
  let rec nest k :=
    if k = 1 then 2019 else sqrt (1 + k * (k + 2)) in
  nest 2018 = 3 :=
by
  intros
  sorry

end nested_radical_value_l22_22373


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22202

theorem measure_of_one_interior_angle_of_regular_octagon 
  (n : Nat) (h_n : n = 8) : 
  (one_interior_angle : ℝ) = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22202


namespace problem_statement_l22_22850

noncomputable def f (x : ℕ) : ℕ := sorry -- Definition for f is not necessary, using sorry to skip

theorem problem_statement (f : ℕ → ℕ)
  (hf_add : ∀ x y : ℕ, f (x + y) = f x + f y)
  (hf_one : f 1 = 2) :
  let sum_f : ℕ → ℕ := λ n, ∑ i in Finset.range (n+1), f i in
  (∃ n : ℕ, sum_f n ≠ n * (n + 1) * f 1) :=
begin
  sorry -- The proof is omitted
end

end problem_statement_l22_22850


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22143

theorem measure_of_one_interior_angle_of_regular_octagon : 
  let n := 8 in 
  let sum_of_interior_angles := 180 * (n - 2) in
  let measure_of_one_interior_angle := sum_of_interior_angles / n in 
  measure_of_one_interior_angle = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22143


namespace min_value_expression_l22_22996

open Real

theorem min_value_expression (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
    (h_abc : a * b * c = 1 / 2) : 
    a^2 + 8 * a * b + 32 * b^2 + 16 * b * c + 8 * c^2 ≥ 18 :=
sorry

end min_value_expression_l22_22996


namespace range_of_R_is_1_to_3_l22_22602

theorem range_of_R_is_1_to_3 (R : ℝ) (hR : 0 < R)
    (circle_eq : ∀ x y : ℝ, x^2 + (y - 2)^2 = R^2)
    (line_eq : ∀ x y : ℝ, y = sqrt 3 * x - 2) :
  1 < R ∧ R < 3 :=
by
  sorry

end range_of_R_is_1_to_3_l22_22602


namespace regular_octagon_interior_angle_l22_22236

theorem regular_octagon_interior_angle : 
  ∀ n : ℕ, n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  intros n hn
  simp [sum_of_interior_angles]
  sorry

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
  180 * (n - 2)

end regular_octagon_interior_angle_l22_22236


namespace largest_prime_factor_of_sum_of_divisors_180_l22_22977

def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in (list.range (n+1)).filter (λ d, n % d = 0), d

def largest_prime_factor (n : ℕ) : ℕ :=
  (list.range (n+1)).filter (λ p, nat.prime p ∧ n % p = 0).last'

theorem largest_prime_factor_of_sum_of_divisors_180 :
  largest_prime_factor (sum_of_divisors 180) = 13 := by sorry

end largest_prime_factor_of_sum_of_divisors_180_l22_22977


namespace largest_prime_factor_sum_divisors_180_is_7_l22_22986

-- Definitions based on conditions in a)
def prime_factors_180 : list (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1)]

def sum_of_divisors (n : ℕ) : ℕ := 
  let factors := prime_factors_180
  in factors.foldl
       (λ acc pf, match pf with
                  | (p, a) => acc * (list.range (a + 1)).map (λ i, p ^ i).sum
                  end)
       1

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  (nat.factors n).last sorry

-- The theorem we want to prove
theorem largest_prime_factor_sum_divisors_180_is_7 :
  largest_prime_factor (sum_of_divisors 180) = 7 := sorry

end largest_prime_factor_sum_divisors_180_is_7_l22_22986


namespace proportion_correct_l22_22905

theorem proportion_correct {a b : ℝ} (h : 2 * a = 5 * b) : a / 5 = b / 2 :=
by {
  sorry
}

end proportion_correct_l22_22905


namespace problem_1_problem_2_l22_22770

def P (event : Type) : ℝ := sorry

variables (A1 A2 B1 B2 A B : Type)

axiom P_A1 : P A1 = 0.6
axiom P_A2 : P A2 = 0.5
axiom P_B1 : P B1 = 0.7
axiom P_B2 : P B2 = 0.9

-- Proof 1
theorem problem_1 : 
  P (A1 ∪ A2) = 0.8 := by
  sorry

-- Proof 2
theorem problem_2 :
  P ((A ∩ Bᶜ) ∪ (Aᶜ ∩ B)) = 0.492 := by
  sorry

end problem_1_problem_2_l22_22770


namespace denominator_of_speed_l22_22716

theorem denominator_of_speed (h : 0.8 = 8 / d * 3600 / 1000) : d = 36 := 
by
  sorry

end denominator_of_speed_l22_22716


namespace possible_coordinates_of_A_B_C_l22_22055

structure Point :=
  (x : ℝ)
  (y : ℝ)

def A := Point
def B := Point
def C := Point

-- Given definitions for angles
def tan_alpha : ℝ := 3 / 4
def cos_beta : ℝ := 7 / 25

theorem possible_coordinates_of_A_B_C 
  (A B C : Point)
  (tan_alpha : tan α = 3 / 4)
  (cos_beta : cos β = 7 / 25)
  (h1 : A = Point.mk 0 0) 
  (h2 : B.y = 0)
  : 
  (∃ m n : ℝ, C = Point.mk m (3 / 4 * m)) ∧
  (∃ x : ℝ, B = Point.mk x 0) 
  :=
  sorry

end possible_coordinates_of_A_B_C_l22_22055


namespace nested_radicals_equivalent_l22_22362

theorem nested_radicals_equivalent :
  (∃ (f : ℕ → ℕ),
    (∀ k, 1 ≤ k → f k = (k + 1)^2 - 1) ∧
    (sqrt (1 + 2 * sqrt (1 + 3 * sqrt (1 + 2017 * sqrt (1 + 2018 * (f 2018))))) = 3)) :=
begin
  use (λ k, (k + 1)^2 - 1),
  split,
  { intros k hk,
    exact rfl, },
  { sorry },
end

end nested_radicals_equivalent_l22_22362


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22164

-- Define the conditions for the problem
def num_sides : ℕ := 8
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Statement of the problem
theorem measure_of_one_interior_angle_of_regular_octagon : 
  (sum_of_interior_angles num_sides) / num_sides = 135 :=
by
  -- Conditions are defined above
  -- Proof steps are omitted
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22164


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22139

theorem measure_of_one_interior_angle_of_regular_octagon : 
  let n := 8 in 
  let sum_of_interior_angles := 180 * (n - 2) in
  let measure_of_one_interior_angle := sum_of_interior_angles / n in 
  measure_of_one_interior_angle = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22139


namespace regular_octagon_interior_angle_l22_22229

theorem regular_octagon_interior_angle:
  let n := 8
  let sum_interior_angles (n: ℕ) := 180 * (n - 2)
  let interior_angle (n: ℕ) := sum_interior_angles n / n
  interior_angle n = 135 :=
by
  sorry

end regular_octagon_interior_angle_l22_22229


namespace sale_price_after_discounts_l22_22330

/-- The sale price of the television as a percentage of its original price after successive discounts of 25% followed by 10%. -/
theorem sale_price_after_discounts (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  original_price = 350 → discount1 = 0.25 → discount2 = 0.10 →
  (original_price * (1 - discount1) * (1 - discount2) / original_price) * 100 = 67.5 :=
by
  intro h_price h_discount1 h_discount2
  sorry

end sale_price_after_discounts_l22_22330


namespace f_inequality_l22_22859

noncomputable def f : ℝ → ℝ := sorry

theorem f_inequality (x1 x2 : ℝ) (hx1 : x1 > 0) (hx2 : x2 > 0)
  (H : ∀ x > 0, x * (deriv (deriv f) x) > f x) :
  f(x1) + f(x2) < f(x1 + x2) :=
sorry

end f_inequality_l22_22859


namespace floor_difference_l22_22413

theorem floor_difference (x : ℝ) (h : x = 13.2) : 
  (⌊x^2⌋ - ⌊x⌋ * ⌊x⌋) = 5 := by
  have h1 : ⌊x⌋ = 13 := by sorry
  have h2 : ⌊x^2⌋ = 174 := by sorry
  calc
  ⌊x^2⌋ - ⌊x⌋ * ⌊x⌋ = 174 - 13 * 13 : by rw [h1, h2]
                 ... = 5 : by norm_num

end floor_difference_l22_22413


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22147

theorem measure_of_one_interior_angle_of_regular_octagon : 
  let n := 8 in 
  let sum_of_interior_angles := 180 * (n - 2) in
  let measure_of_one_interior_angle := sum_of_interior_angles / n in 
  measure_of_one_interior_angle = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22147


namespace find_knight_liar_l22_22739

-- Define the types for individuals and their statements
inductive Person
| A
| B
| C

def is_knight : Person → Prop := sorry
def is_liar : Person → Prop := sorry

-- Conditions
axiom A_statement : (∃ (n : ℕ), n = 1 ∧ ∀ p : Person, is_knight p → p = Person.A)
axiom B_statement : A_statement ∧ (is_knight Person.B ∨ is_liar Person.B)
axiom C_statement : ¬(is_knight Person.B)

-- Conclusion to be proven
theorem find_knight_liar : is_knight Person.C ∧ is_liar Person.B :=
by
  sorry

end find_knight_liar_l22_22739


namespace regular_octagon_interior_angle_l22_22234

theorem regular_octagon_interior_angle : 
  ∀ n : ℕ, n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  intros n hn
  simp [sum_of_interior_angles]
  sorry

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
  180 * (n - 2)

end regular_octagon_interior_angle_l22_22234


namespace regular_octagon_interior_angle_l22_22240

theorem regular_octagon_interior_angle : 
  ∀ n : ℕ, n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  intros n hn
  simp [sum_of_interior_angles]
  sorry

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
  180 * (n - 2)

end regular_octagon_interior_angle_l22_22240


namespace regular_octagon_interior_angle_l22_22223

theorem regular_octagon_interior_angle:
  let n := 8
  let sum_interior_angles (n: ℕ) := 180 * (n - 2)
  let interior_angle (n: ℕ) := sum_interior_angles n / n
  interior_angle n = 135 :=
by
  sorry

end regular_octagon_interior_angle_l22_22223


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22136

theorem measure_of_one_interior_angle_of_regular_octagon : 
  let n := 8 in 
  let sum_of_interior_angles := 180 * (n - 2) in
  let measure_of_one_interior_angle := sum_of_interior_angles / n in 
  measure_of_one_interior_angle = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22136


namespace sum_of_x_coords_eq_l22_22044

-- Define the line segments as functions for the piecewise function y = f(x)
def segment1 (x : ℝ) : ℝ := 2 * x   -- From (-1, -2) to (0, 0)
def segment2 (x : ℝ) : ℝ := -x + 3  -- From (0, 3) to (2, 1)
def segment3 (x : ℝ) : ℝ := 2 * x - 3 -- From (2, 1) to (4, 5)

-- Function y = f(x) definition
def f (x : ℝ) : ℝ :=
  if h : -1 ≤ x ∧ x ≤ 0 then segment1 x
  else if h : 0 < x ∧ x ≤ 2 then segment2 x
  else if h : 2 < x ∧ x ≤ 4 then segment3 x
  else 0

-- Define the proof statement
theorem sum_of_x_coords_eq : ∑ x in {0.9, 1.2, 2.4}, x = 4.5 :=
  by
    sorry

end sum_of_x_coords_eq_l22_22044


namespace cousins_assignment_l22_22597

open Finset

def choose (n k : ℕ) : ℕ := nat.choose n k

theorem cousins_assignment (five_cousins : ℕ := 5) (rooms : ℕ := 4) (small_room_limit : ℕ := 2) :
  (∑ (r : ℕ) in {((choose 5 3) + (choose 5 3) + (choose 5 2 * choose 3 2 * 2)), 1}, r) = 80 :=
by
  sorry

end cousins_assignment_l22_22597


namespace digits_exceed_10_power_15_l22_22538

noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem digits_exceed_10_power_15 (x : ℝ) 
  (h : log3 (log2 (log2 x)) = 3) : log10 x > 10^15 := 
sorry

end digits_exceed_10_power_15_l22_22538


namespace maximizeCars_l22_22732

-- Define the dimensions of the parking lot
def parkingLotDimensions : ℕ × ℕ := (7, 7)

-- Define a condition for a valid parking configuration where all cars can exit without blocking each other
def canExit (parkingConfig : ℕ → ℕ → bool) (gate : ℕ × ℕ) : Prop :=
  ∀ x y, parkingConfig x y = true → ∃ path : list (ℕ × ℕ), 
    path ≠ [] ∧ path.head = some (x, y) ∧ 
    path.last = some gate ∧ 
    (∀ (i j : ℕ × ℕ), (i, j) ∈ list.zip path (list.tail path) → isAdjacent i j) ∧ 
    (∀ (a : ℕ × ℕ), a ∈ path → parkingConfig a.1 a.2 = false ∨ a = (x, y))

-- Define the condition that the gate is at one corner
def isGateAtCorner (gate : ℕ × ℕ) : Prop :=
  gate = (0, 0) ∨ gate = (0, 6) ∨ gate = (6, 0) ∨ gate = (6, 6)

-- Predicate that specifies the count of parked cars
def carCount (parkingConfig : ℕ → ℕ → bool) : ℕ :=
  (Finset.univ.product Finset.univ).count (λ (coords : ℕ × ℕ), parkingConfig coords.1 coords.2 = true)

-- The main theorem
theorem maximizeCars 
(parkingConfig : ℕ → ℕ → bool) 
(gate : ℕ × ℕ) 
(h1 : parkingLotDimensions = (7, 7)) 
(h2 : isGateAtCorner gate) 
(h3 : canExit parkingConfig gate) : 
  carCount parkingConfig = 28 :=
sorry

end maximizeCars_l22_22732


namespace sec_225_eq_neg_sqrt2_l22_22466

theorem sec_225_eq_neg_sqrt2 : ∀ θ : ℝ, θ = 225 * (π / 180) → sec θ = -real.sqrt 2 :=
by
  intro θ hθ
  sorry

end sec_225_eq_neg_sqrt2_l22_22466


namespace derivative_log_l22_22471

-- Define the function f(x) = log_a(2x^2 - 1)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (2 * x^2 - 1)

-- Define the derivative expression we expect
noncomputable def derivative_expr (a : ℝ) (x : ℝ) : ℝ := (4 * x) / ((2 * x^2 - 1) * log a)

-- State that the derivative of f is equal to the given derivative expression
theorem derivative_log (a x : ℝ) (hx : 2 * x^2 - 1 > 0) :
  deriv (f a) x = derivative_expr a x :=
sorry

end derivative_log_l22_22471


namespace max_additional_bags_correct_l22_22064

-- Definitions from conditions
def num_people : ℕ := 6
def bags_per_person : ℕ := 5
def weight_per_bag : ℕ := 50
def max_plane_capacity : ℕ := 6000

-- Derived definitions from conditions
def total_bags : ℕ := num_people * bags_per_person
def total_weight_of_bags : ℕ := total_bags * weight_per_bag
def remaining_capacity : ℕ := max_plane_capacity - total_weight_of_bags 
def max_additional_bags : ℕ := remaining_capacity / weight_per_bag

-- Theorem statement
theorem max_additional_bags_correct : max_additional_bags = 90 := by
  -- Proof skipped
  sorry

end max_additional_bags_correct_l22_22064


namespace sum_of_squares_of_solutions_l22_22814

theorem sum_of_squares_of_solutions :
  (∑ α in {x | ∃ ε : ℝ, ε = x² - x + 1/2010 ∧ |ε| = 1/2010}, α^2) = 2008 / 1005 :=
by
  sorry

end sum_of_squares_of_solutions_l22_22814


namespace simplify_complex_expression_l22_22625

open Complex

theorem simplify_complex_expression : (5 - 3 * Complex.i)^2 = 16 - 30 * Complex.i :=
by
  sorry

end simplify_complex_expression_l22_22625


namespace problem1_problem2_l22_22675

theorem problem1 (a b c : ℝ) (h1 : a = 5.42) (h2 : b = 3.75) (h3 : c = 0.58) :
  a - (b - c) = 2.25 :=
by sorry

theorem problem2 (d e f g h : ℝ) (h4 : d = 4 / 5) (h5 : e = 7.7) (h6 : f = 0.8) (h7 : g = 3.3) (h8 : h = 1) :
  d * e + f * g - d = 8 :=
by sorry

end problem1_problem2_l22_22675


namespace concyclic_proof_l22_22898

open EuclideanGeometry

-- Given points A, B on circle Γ, points C, D on Γ such that C, D, X are collinear,
-- and such that CA ⊥ BD, CA intersects BD at F, CD intersects AB at G,
-- and H is the intersection of BD and the perpendicular bisector of GX,
-- prove that X, F, G, H are concyclic.

noncomputable def cyclic_points {A B C D F G H X : Point} (Γ : Circle) 
  (hA : A ∈ Γ) (hB : B ∈ Γ) 
  (hC : C ∈ Γ) (hD : D ∈ Γ)
  (hX : tangent γ A ∩ tangent γ B = X)
  (hCollinear : collinear ({C, D, X} : Set Point))
  (hPerp : perpendicular (line_through C A) (line_through B D))
  (hIntF : intersect_lines (line_through C A) (line_through B D) = F)
  (hIntG : intersect_lines (line_through C D) (line_through A B) = G)
  (hH : is_perpendicular_bisector (segment G X) (line_through B D) = H) : Prop :=
  concyclic {X, F, G, H}

-- Provide statement that these points are concyclic
theorem concyclic_proof 
  (A B C D F G H X : Point) 
  (Γ : Circle) 
  (hA : A ∈ Γ) (hB : B ∈ Γ) 
  (hC : C ∈ Γ) (hD : D ∈ Γ)
  (hX : tangent γ A ∩ tangent γ B = X)
  (hCollinear : collinear ({C, D, X} : Set Point))
  (hPerp : perpendicular (line_through C A) (line_through B D))
  (hIntF : intersect_lines (line_through C A) (line_through B D) = F)
  (hIntG : intersect_lines (line_through C D) (line_through A B) = G)
  (hH : is_perpendicular_bisector (segment G X) (line_through B D) = H) : 
  cyclic_points Γ hA hB hC hD hX hCollinear hPerp hIntF hIntG hH :=
  sorry

end concyclic_proof_l22_22898


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22198

theorem measure_of_one_interior_angle_of_regular_octagon 
  (n : Nat) (h_n : n = 8) : 
  (one_interior_angle : ℝ) = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22198


namespace regular_octagon_interior_angle_l22_22179

theorem regular_octagon_interior_angle : 
  ∀ (n : ℕ), n = 8 → (sum_of_internal_angles n) / n = 135 := 
by
  intros n hn
  rw hn
  sorry

end regular_octagon_interior_angle_l22_22179


namespace price_per_candy_bar_l22_22599

-- Definitions from conditions
def chocolate_oranges_sold : ℕ := 20
def price_per_chocolate_orange : ℕ := 10
def total_goal : ℕ := 1000
def candy_bars_sold : ℕ := 160

-- The remaining amount Nick needs to raise
def remaining_amount : ℕ := total_goal - (chocolate_oranges_sold * price_per_chocolate_orange)

-- The price per candy bar
theorem price_per_candy_bar : ℕ :=
  let amount_per_candy_bar := remaining_amount / candy_bars_sold in
  amount_per_candy_bar = 5 :=
begin
  -- The proof is omitted, use sorry to skip it
  sorry
end

end price_per_candy_bar_l22_22599


namespace value_of_k_l22_22484

theorem value_of_k (k : ℕ) (h : 3 * 10 * 4 * k = nat.factorial 9) : k = 15120 :=
sorry

end value_of_k_l22_22484


namespace sum_of_midpoints_double_l22_22658

theorem sum_of_midpoints_double (a b c : ℝ) (h : a + b + c = 15) : 
  (a + b) + (a + c) + (b + c) = 30 :=
by
  -- We skip the proof according to the instruction
  sorry

end sum_of_midpoints_double_l22_22658


namespace expected_earnings_l22_22315

theorem expected_earnings : 
  let pA := 1 / 4
  let pB := 1 / 4
  let pC := 1 / 3
  let pDisappear := 1 / 6
  let payoutA := 2
  let payoutB := -1
  let payoutC := 4
  let payoutDisappear := -3
  in pA * payoutA + pB * payoutB + pC * payoutC + pDisappear * payoutDisappear = 13 / 12 :=
by
  let pA := 1 / 4
  let pB := 1 / 4
  let pC := 1 / 3
  let pDisappear := 1 / 6
  let payoutA := 2
  let payoutB := -1
  let payoutC := 4
  let payoutDisappear := -3
  calc 
    pA * payoutA + pB * payoutB + pC * payoutC + pDisappear * payoutDisappear
    = (1 / 4) * 2 + (1 / 4) * (-1) + (1 / 3) * 4 + (1 / 6) * (-3) : by rfl
    ... = 1 / 2 - 1 / 4 + 4 / 3 - 1 / 2 : by norm_num
    ... = 1 / 2 - 1 / 4 + 4 / 3 - 1 / 2 : by simp
    ... = (1 / 2 - 1 / 2) - 1 / 4 + 4 / 3 : by norm_num
    ... = - 1 / 4 + 4 / 3 : by simp
    ... = - 1 / 4 + 4 / 3 : by linarith
    ... = (- 1 / 4) + (4 / 3) : by norm_num
    ... = (-3 / 12) + (16 / 12) : by norm_num
    ... = 13 / 12 : by norm_num

end expected_earnings_l22_22315


namespace regular_octagon_interior_angle_l22_22099

theorem regular_octagon_interior_angle (n : ℕ) (h₁ : n = 8) :
  (180 * (n - 2)) / n = 135 :=
by
  rw [h₁]
  sorry

end regular_octagon_interior_angle_l22_22099


namespace regular_octagon_interior_angle_l22_22178

theorem regular_octagon_interior_angle : 
  ∀ (n : ℕ), n = 8 → (sum_of_internal_angles n) / n = 135 := 
by
  intros n hn
  rw hn
  sorry

end regular_octagon_interior_angle_l22_22178


namespace ab_non_positive_l22_22517

-- Define the conditions as a structure if necessary.
variables {a b : ℝ}

-- State the theorem.
theorem ab_non_positive (h : 3 * a + 8 * b = 0) : a * b ≤ 0 :=
sorry

end ab_non_positive_l22_22517


namespace number_of_divisible_permutations_l22_22532

def digit_list := [1, 3, 1, 1, 5, 2, 1, 5, 2]
def count_permutations (d : List ℕ) (n : ℕ) : ℕ :=
  let fact := Nat.factorial
  let number := fact 8 / (fact 3 * fact 2 * fact 1)
  number

theorem number_of_divisible_permutations : count_permutations digit_list 2 = 3360 := by
  sorry

end number_of_divisible_permutations_l22_22532


namespace coefficient_of_x3_in_expansion_l22_22074

open Real Nat

theorem coefficient_of_x3_in_expansion :
  let a := 3
  let b := 2
  let n := 8
  let k := 3
  (a + b)^n = ∑ i in range (n+1), (binom n i * (a^i) * b^(n-i)) →
  (∃ c : ℤ, (c * (x^3) ⊂ ∑ i in range (n+1), (binom n i * (a * x)^i * b^(n-i))) ∧ c = 48384) := 
by
  sorry

end coefficient_of_x3_in_expansion_l22_22074


namespace probability_of_exactly_one_solves_l22_22018

variable (p1 p2 : ℝ)

theorem probability_of_exactly_one_solves (h1 : 0 ≤ p1) (h2 : p1 ≤ 1) (h3 : 0 ≤ p2) (h4 : p2 ≤ 1) :
  (p1 * (1 - p2) + p2 * (1 - p1)) = (p1 * (1 - p2) + p2 * (1 - p1)) :=
by
  sorry

end probability_of_exactly_one_solves_l22_22018


namespace find_f_l22_22750

noncomputable def T (n : ℕ) : ℝ := 2 * real.pi * n

-- Define the conditions related to the periodic function and the given functional equation.
axiom periodic_f (f : ℝ → ℝ) (n : ℕ) : (∀ x, sin x = f x - 0.4 * f (x - real.pi)) ∧ (∀ x, sin x = f (x - T n) - 0.4 * f (x - T n - real.pi)) ∧ (∀ x, sin x = sin (x - T n))

-- Define the theorem statement
theorem find_f (f : ℝ → ℝ) (n : ℕ) (h : periodic_f f n) : ∀ x, f x = (5 / 7) * sin x := sorry

end find_f_l22_22750


namespace solve_system_l22_22631

theorem solve_system :
  ∃ (x y: ℝ), 4 * x - 6 * y = -2 ∧ 5 * x + 3 * y = 2.6 ∧ x ≈ 0.4571 ∧ y ≈ 0.1048 :=
begin
  sorry
end

end solve_system_l22_22631


namespace regular_octagon_interior_angle_l22_22228

theorem regular_octagon_interior_angle:
  let n := 8
  let sum_interior_angles (n: ℕ) := 180 * (n - 2)
  let interior_angle (n: ℕ) := sum_interior_angles n / n
  interior_angle n = 135 :=
by
  sorry

end regular_octagon_interior_angle_l22_22228


namespace floor_diff_l22_22429

theorem floor_diff (x : ℝ) (h : x = 13.2) : 
  (Int.floor (x^2) - (Int.floor x) * (Int.floor x) = 5) := by
  sorry

end floor_diff_l22_22429


namespace regular_octagon_interior_angle_l22_22133

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : 
  let sum_interior_angles := 180 * (n - 2)
  let num_sides := 8
  let one_interior_angle := sum_interior_angles / num_sides
  one_interior_angle = 135 :=
by
  sorry

end regular_octagon_interior_angle_l22_22133


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22156

theorem measure_of_one_interior_angle_of_regular_octagon :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 := 
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22156


namespace initial_average_age_l22_22040

theorem initial_average_age (A : ℕ) (h1 : ∀ x : ℕ, 10 * A = 10 * A)
  (h2 : 5 * 17 + 10 * A = 15 * (A + 1)) : A = 14 :=
by 
  sorry

end initial_average_age_l22_22040


namespace exist_m_squared_plus_9_mod_2_pow_n_minus_1_l22_22790

theorem exist_m_squared_plus_9_mod_2_pow_n_minus_1 (n : ℕ) (hn : n > 0) :
  (∃ m : ℤ, (m^2 + 9) % (2^n - 1) = 0) ↔ ∃ k : ℕ, n = 2^k :=
by
  sorry

end exist_m_squared_plus_9_mod_2_pow_n_minus_1_l22_22790


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22141

theorem measure_of_one_interior_angle_of_regular_octagon : 
  let n := 8 in 
  let sum_of_interior_angles := 180 * (n - 2) in
  let measure_of_one_interior_angle := sum_of_interior_angles / n in 
  measure_of_one_interior_angle = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22141


namespace diagonal_plane_angle_l22_22650

theorem diagonal_plane_angle
  (α : Real)
  (a : Real)
  (plane_square_angle_with_plane : Real)
  (diagonal_plane_angle : Real) 
  (h1 : plane_square_angle_with_plane = α) :
  diagonal_plane_angle = Real.arcsin (Real.sin α / Real.sqrt 2) :=
sorry

end diagonal_plane_angle_l22_22650


namespace sum_of_squares_of_solutions_l22_22829

theorem sum_of_squares_of_solutions :
  (∑ x in {x : ℝ | | x^2 - x + 1/2010 | = 1/2010}, x^2) = 2008 / 1005 :=
by
  sorry

end sum_of_squares_of_solutions_l22_22829


namespace initial_mixture_equals_50_l22_22543

theorem initial_mixture_equals_50 (x : ℝ) (h1 : 0.10 * x + 10 = 0.25 * (x + 10)) : x = 50 :=
by
  sorry

end initial_mixture_equals_50_l22_22543


namespace mathlib_problem_l22_22391

noncomputable def is_impossible_partition_of_N (A B : Set ℕ) : Prop :=
  ∃ n : ℕ, 
    let count_sums (S : Set ℕ) := ∑ k in S, ∑ l in S, if k ≠ l ∧ k + l = n then 1 else 0 in
    count_sums A ≠ count_sums B

theorem mathlib_problem : ∀ A B : Set ℕ, ¬ is_impossible_partition_of_N A B := 
by 
  sorry

end mathlib_problem_l22_22391


namespace difference_surface_area_l22_22726

noncomputable def R : ℝ := sorry
noncomputable def α : ℝ := sorry

def r := R * Real.cos α
def h := 2 * R * Real.sin α
def LSA := 2 * Real.pi * r * h
def SA_sphere := 4 * Real.pi * R^2

theorem difference_surface_area :
  ∀ (R : ℝ) (α : ℝ), 4 * Real.pi * R^2 - 2 * Real.pi * R^2 * Real.sin (2 * α) = 2 * Real.pi * R^2 := by
  sorry

end difference_surface_area_l22_22726


namespace regular_octagon_interior_angle_l22_22127

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : 
  let sum_interior_angles := 180 * (n - 2)
  let num_sides := 8
  let one_interior_angle := sum_interior_angles / num_sides
  one_interior_angle = 135 :=
by
  sorry

end regular_octagon_interior_angle_l22_22127


namespace solve_trig_eq_l22_22630

-- Define the conditions using the triple-angle identities
def sin_triple_angle (x : ℝ) : ℝ := 3 * sin x - 4 * (sin x) ^ 3
def cos_triple_angle (x : ℝ) : ℝ := 4 * (cos x) ^ 3 - 3 * cos x

-- State the theorem
theorem solve_trig_eq (x : ℝ) (k : ℤ) :
  (x = 45 * (π / 180) + k * π ∨ x = (116 + 33 / 60 + 54 / 3600) * (π / 180) + k * π) →
  2 * sin (3 * x) = 3 * cos x + cos (3 * x) :=
by
  sorry

end solve_trig_eq_l22_22630


namespace regular_octagon_interior_angle_l22_22101

theorem regular_octagon_interior_angle (n : ℕ) (h₁ : n = 8) :
  (180 * (n - 2)) / n = 135 :=
by
  rw [h₁]
  sorry

end regular_octagon_interior_angle_l22_22101


namespace weight_of_B_l22_22701

-- Definitions for the weights
variables (A B C : ℝ)

-- Conditions from the problem
def avg_ABC : Prop := (A + B + C) / 3 = 45
def avg_AB : Prop := (A + B) / 2 = 40
def avg_BC : Prop := (B + C) / 2 = 43

-- The theorem to prove the weight of B
theorem weight_of_B (h1 : avg_ABC A B C) (h2 : avg_AB A B) (h3 : avg_BC B C) : B = 31 :=
sorry

end weight_of_B_l22_22701


namespace nested_fraction_sum_l22_22394

theorem nested_fraction_sum : 
  2010 + (3⁻¹ * (2007 + (3⁻¹ * (2004 + ... + (3⁻¹ * (6 + 3⁻¹ * 3)) ...)))) = 2006 := 
sorry

end nested_fraction_sum_l22_22394


namespace expression_value_l22_22479

theorem expression_value (a b c : ℝ) (ha : a ≠ b) (hb : b ≠ c) (hc : a ≠ c) 
  (ha_eq : a^3 - 2022 * a^2 + 1011 = 0) 
  (hb_eq : b^3 - 2022 * b^2 + 1011 = 0) 
  (hc_eq : c^3 - 2022 * c^2 + 1011 = 0) 
  : (1 / (a * b)) + (1 / (b * c)) + (1 / (a * c)) = -2 :=
by
  let h1 : a + b + c = 2022 := sorry -- From Vieta's formulas
  let h2 : a * b + b * c + c * a = 0 := sorry -- From Vieta's formulas
  let h3 : a * b * c = -1011 := sorry -- From Vieta's formulas
  calc
    (1 / (a * b)) + (1 / (b * c)) + (1 / (a * c)) 
        = (c + a + b) / (a * b * c) : sorry
    ... = (a + b + c) / (a * b * c) : by rw [add_comm c (a + b)]
    ... = (a + b + c) / (-1011) : by rw [h3]
    ... = 2022 / -1011 : by rw [h1]
    ... = -2 : sorry

end expression_value_l22_22479


namespace number_of_correct_conclusions_l22_22513

noncomputable def f (x : ℝ) : ℝ := cos (2 * x + π / 3) - cos (2 * x)

theorem number_of_correct_conclusions : 
  (¬ (∀ x, f (-x) = -f x) ∧
   (∃ x, x = 2 * π / 3 ∧ f (2 * π / 3) = 1) ∧
   (∃ x, x = 5 * π / 12 ∧ f (5 * π / 12) = 0) ∧
   (∀ k : ℤ, ∀ x, k * π + π / 6 ≤ x ∧ x ≤ k * π + 2 * π / 3 → f' x > 0)) → 
   3 := 
sorry

end number_of_correct_conclusions_l22_22513


namespace largest_prime_factor_sum_of_divisors_180_l22_22992

def sum_of_divisors (n : ℕ) : ℕ :=
  ∏ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d

theorem largest_prime_factor_sum_of_divisors_180 :
  let M := sum_of_divisors 180 in
  ∀ p, nat.prime p → p ∣ M → p ≤ 13 :=
by
  let M := sum_of_divisors 180
  have p_factors : multiset ℕ := (unique_factorization_monoid.factorization M).to_multiset
  exact sorry

end largest_prime_factor_sum_of_divisors_180_l22_22992


namespace dot_product_is_2_l22_22847

variable (a : ℝ × ℝ) (b : ℝ × ℝ)

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

theorem dot_product_is_2 (ha : a = (1, 0)) (hb : b = (2, 1)) :
  dot_product a b = 2 := by
  sorry

end dot_product_is_2_l22_22847


namespace tan_add_condition_l22_22907

theorem tan_add_condition (x y : ℝ) (hx : Real.tan x + Real.tan y = 40) (hy : Real.cot x + Real.cot y = 1) : 
  Real.tan (x + y) = -40 / 39 := 
  by sorry

end tan_add_condition_l22_22907


namespace correct_statements_l22_22689

-- Define the statements as predicates
def statement1 : Prop := ∀ (A B C : Point), ¬ (is_collinear A B C) → ∃ (O : Point) (r : ℝ), circle O r A B C
def statement2 : Prop := ∀ (O : Point) (r : ℝ) (A B : Point), is_diameter O r → is_chord_perpendicular_bisector O A B
def statement3 : Prop := ∀ (O : Point) (r : ℝ) (A B C D : Point), subtends_equal_central_angles O A B C D → equal_chords O A B C D
def statement4 : Prop := ∀ (ABC : Triangle) (O : Point), is_circumcenter O ABC → is_equidistant_from_vertices O ABC

-- Main theorem to prove
theorem correct_statements : (statement2 ∧ statement4) ∧ ¬statement1 ∧ ¬statement3 :=
by
  sorry

end correct_statements_l22_22689


namespace interior_angle_of_regular_octagon_l22_22106

theorem interior_angle_of_regular_octagon :
  let n := 8 in
  let sum_interior_angles := 180 * (n - 2) in
  let angle := sum_interior_angles / n in
  angle = 135 :=
by
  sorry

end interior_angle_of_regular_octagon_l22_22106


namespace part1_part2_l22_22707

-- Statement for part (1)
theorem part1 (a b : ℝ) (ha : abs a < 1) (hb : abs b < 1) : abs (1 - a * b) / abs (a - b) > 1 := 
sorry

-- Statement for part (2)
theorem part2 (a b λ : ℝ) (ha : abs a < 1) (hb : abs b < 1) : abs (1 - a * b * λ) / abs (a * λ - b) > 1 ↔ -1 ≤ λ ∧ λ ≤ 1 :=
sorry

end part1_part2_l22_22707


namespace min_rectangle_perimeter_l22_22572

noncomputable def curve_C (θ : ℝ) : ℝ := real.sqrt (3 / (1 + 2 * real.sin θ ^ 2))

def point_R : ℝ × ℝ := (2 * real.sqrt 2, real.pi / 4)

def point_on_curve_C (θ : ℝ) : ℝ × ℝ :=
  let ρ := curve_C θ in (ρ * real.cos θ, ρ * real.sin θ)

structure rectangle (P Q R S : ℝ × ℝ) : Prop :=
(perpendicular_to_polar_axis : (Q.2 - P.2) = 0)
(diagonal_PR : R = (P.1 + Q.1 + S.1, P.2 + Q.2 + S.2))

theorem min_rectangle_perimeter :
  ∃ (P Q R S : ℝ × ℝ) (θ : ℝ),
    rectangle P Q R S ∧
    point_on_curve_C θ = P ∧
    Q.1 - P.1 = 0 ∧
    2 * (Q.2 - P.2 + R.1 - P.1) = 4 ∧
    θ = real.pi / 6 :=
by sorry


end min_rectangle_perimeter_l22_22572


namespace interior_angle_regular_octagon_l22_22083

theorem interior_angle_regular_octagon (n : ℕ) (h : n = 8) : 
  let sum_of_interior_angles := 180 * (n - 2)
  let number_of_sides := n
  let one_interior_angle := sum_of_interior_angles / number_of_sides
  in one_interior_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l22_22083


namespace part_I_part_II_l22_22887

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x - π / 6) - 1 / 2

theorem part_I (ω : ℝ) (b : ℝ) (h1 : ω > 0)
  (h2 : ∃ x : ℝ, f x = sqrt 3 * sin (ω * x - π / 6) + b)
  (h3 : (∀ x ∈ Icc 0 (π / 4), sqrt 3 * sin (2 * x - π / 6) - 1 / 2 ≤ 1))
  (h4 : (f (π / 4) = 1)) : f = fun x => sqrt 3 * sin (2 * x - π / 6) - 1 / 2 := sorry

noncomputable def g (x : ℝ) : ℝ := sqrt 3 * sin (2 * x - π / 3) - 1 / 2

theorem part_II (m : ℝ)
  (h1 : ∀ x ∈ Icc 0 (π / 3), g x - 3 ≤ m ∧ m ≤ g x + 3) : 
  -2 ≤ m ∧ m ≤ 1 := sorry


end part_I_part_II_l22_22887


namespace sum_of_squares_of_solutions_l22_22808

theorem sum_of_squares_of_solutions :
  (∀ x, (abs (x^2 - x + (1 / 2010)) = (1 / 2010)) → (x = 0 ∨ x = 1 ∨ (∃ a b, 
  a + b = 1 ∧ a * b = 1 / 1005 ∧ x = a ∧ x = b))) →
  ((0^2 + 1^2) + (1 - 2 * (1 / 1005)) = (2008 / 1005)) :=
by
  intro h
  sorry

end sum_of_squares_of_solutions_l22_22808


namespace part_I_a_part_I_b_part_II_l22_22322

-- Definitions based on the problem conditions
def seq_a : ℕ → ℕ 
| 1 := 1
| (n + 1) := 2 * (finset.range (n+1)).sum seq_a + 1

def S (n : ℕ) : ℕ := (finset.range n).sum seq_a

def seq_b (n : ℕ) : ℕ := 3 * n - 6

-- The proof problem statements
theorem part_I_a (n : ℕ) : seq_a n = 3^(n - 1) := sorry

theorem part_I_b (n : ℕ) : seq_b n = 3n - 6 := sorry

theorem part_II (k : ℝ) : (∀ n : ℕ, n ≥ 1 → (S n + 1 / 2) * k ≥ 3 * n - 6) → k ≥ 2 / 9 := sorry

end part_I_a_part_I_b_part_II_l22_22322


namespace squirrel_circumference_l22_22738

theorem squirrel_circumference
  (h1 : ∃ C : ℝ, C > 0) -- Assume there exists a positive circumference.
  (h2 : ∀ rise : ℝ, rise = 5) -- Assume the rise for each circuit is 5 feet.
  (h3 : ∀ height : ℝ, height = 25) -- Assume the post height is 25 feet.
  (h4 : ∀ travel : ℝ, travel = 15) -- Assume the squirrel travels 15 feet.
  (h5 : ∀ circuits : ℝ, circuits = travel / rise) -- Circuits made by the squirrel.
  (h6 : ∀ horizontal_distance : ℝ, horizontal_distance = circuits * C) -- Horizontal distance traveled by the squirrel.
  (h7 : horizontal_distance = travel) -- Horizontal distance equals the travel distance.
  : ∃ C : ℝ, C = 5 := -- Conclusion: the circumference C is 5 feet.
begin
  sorry
end

end squirrel_circumference_l22_22738


namespace sufficient_y_wages_l22_22328

noncomputable def days_sufficient_for_y_wages (Wx Wy : ℝ) (total_money : ℝ) : ℝ :=
  total_money / Wy

theorem sufficient_y_wages
  (Wx Wy : ℝ)
  (H1 : ∀(D : ℝ), total_money = D * Wx → D = 36 )
  (H2 : total_money = 20 * (Wx + Wy)) :
  days_sufficient_for_y_wages Wx Wy total_money = 45 := by
  sorry

end sufficient_y_wages_l22_22328


namespace max_buses_in_city_l22_22930

/--
In a city with 9 bus stops where each bus serves exactly 3 stops and any two buses share at most one stop, 
prove that the maximum number of buses that can be in the city is 12.
-/
theorem max_buses_in_city :
  ∃ (B : Set (Finset ℕ)), 
    (∀ b ∈ B, b.card = 3) ∧ 
    (∀ b1 b2 ∈ B, b1 ≠ b2 → (b1 ∩ b2).card ≤ 1) ∧ 
    (∀ b ∈ B, ∀ s ∈ b, s < 9) ∧ 
    B.card = 12 := sorry

end max_buses_in_city_l22_22930


namespace regular_octagon_interior_angle_l22_22231

theorem regular_octagon_interior_angle:
  let n := 8
  let sum_interior_angles (n: ℕ) := 180 * (n - 2)
  let interior_angle (n: ℕ) := sum_interior_angles n / n
  interior_angle n = 135 :=
by
  sorry

end regular_octagon_interior_angle_l22_22231


namespace max_buses_in_city_l22_22931

/--
In a city with 9 bus stops where each bus serves exactly 3 stops and any two buses share at most one stop, 
prove that the maximum number of buses that can be in the city is 12.
-/
theorem max_buses_in_city :
  ∃ (B : Set (Finset ℕ)), 
    (∀ b ∈ B, b.card = 3) ∧ 
    (∀ b1 b2 ∈ B, b1 ≠ b2 → (b1 ∩ b2).card ≤ 1) ∧ 
    (∀ b ∈ B, ∀ s ∈ b, s < 9) ∧ 
    B.card = 12 := sorry

end max_buses_in_city_l22_22931


namespace scientific_notation_470000000_l22_22011

theorem scientific_notation_470000000 : 470000000 = 4.7 * 10^8 :=
by
  sorry

end scientific_notation_470000000_l22_22011


namespace probability_two_presidents_l22_22009

noncomputable def probability_two_presidents_receive_books : ℚ :=
  1 / 3 * ((6.choose 4)⁻¹ * ((4.choose 2) / 5) +
           (8.choose 4)⁻¹ * ((6.choose 2) / 70) +
           (9.choose 4)⁻¹ * ((7.choose 2) / 42))

theorem probability_two_presidents :
  probability_two_presidents_receive_books = 82 / 315 :=
sorry

end probability_two_presidents_l22_22009


namespace maximum_M_k_l22_22481

def J (k : ℕ) : ℕ :=
10^ (k + 3) + 128 

def M (k : ℕ) : ℕ :=
nat.factorization (J k) 2

theorem maximum_M_k : ∀ k > 0, M k ≤ 8 ∧ ∃ k0, k0 > 0 ∧ M k0 = 8 := 
begin
  sorry
end

end maximum_M_k_l22_22481


namespace part1_part2_l22_22489

noncomputable def f (x a : ℝ) := x^2 - 2 * a * x + 5

theorem part1 (a : ℝ) (h1 : 1 < a) (h2 : ∀ x, 1 ≤ x ∧ x ≤ a → f x a ∈ set.Icc 1 a) : a = 2 :=
sorry

theorem part2 (a : ℝ) (h1 : 1 < a) (h2 : ∀ x1 x2 : ℝ, (1 ≤ x1 ∧ x1 ≤ a + 1) → (1 ≤ x2 ∧ x2 ≤ a + 1) → |f x1 a - f x2 a| ≤ 4) : 1 < a ∧ a ≤ 3 :=
sorry

end part1_part2_l22_22489


namespace parabola_focus_coordinates_l22_22470

theorem parabola_focus_coordinates : 
  (∃ x y : ℝ, x^2 = (1/2 : ℝ) * y) → (0, (1/8 : ℝ)) = (0, 1 * (1/8)) :=
by
  intros
  have h : ∀ p : ℝ, (x^2 : ℝ) = 4 * p * y ↔ x^2 = (1/2 : ℝ) * y := 
    sorry -- Placeholder for the equivalence statement
  sorry -- Placeholder for the proof

end parabola_focus_coordinates_l22_22470


namespace sec_225_eq_neg_sqrt2_l22_22440

theorem sec_225_eq_neg_sqrt2 :
  sec 225 = -sqrt 2 :=
by 
  -- Definitions for conditions
  have h1 : ∀ θ : ℕ, sec θ = 1 / cos θ := sorry,
  have h2 : cos 225 = cos (180 + 45) := sorry,
  have h3 : cos (180 + 45) = -cos 45 := sorry,
  have h4 : cos 45 = 1 / sqrt 2 := sorry,
  -- Required proof statement
  sorry

end sec_225_eq_neg_sqrt2_l22_22440


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22160

theorem measure_of_one_interior_angle_of_regular_octagon :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 := 
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22160


namespace fibonacci_cosine_identity_l22_22588

noncomputable def theta (θ : ℝ) (hθ : 0 < θ ∧ θ < π) : Prop := true
noncomputable def fib (n : ℕ) : ℕ -- Define Fibonacci sequence

theorem fibonacci_cosine_identity (θ : ℝ) (hθ : 0 < θ ∧ θ < π) (x : ℂ) (hn : x + 1/x = 2 * Real.cos (2 * θ)) (F_n : ℕ) :
  x^F_n + 1/x^F_n = 2 * Real.cos (2 * F_n * θ) :=
sorry

end fibonacci_cosine_identity_l22_22588


namespace woman_born_second_half_20th_century_l22_22333

theorem woman_born_second_half_20th_century (x : ℕ) (hx : 45 < x ∧ x < 50) (h_year : x * x = 2025) :
  x * x - x = 1980 :=
by {
  -- Add the crux of the problem here.
  sorry
}

end woman_born_second_half_20th_century_l22_22333


namespace teacher_assignment_l22_22737

def teachers : Type := fin 6
def grades : Type := fin 4
def is_A_or_B (teacher : teachers) : Prop := teacher.val = 0 ∨ teacher.val = 1

theorem teacher_assignment :
  (∃ (f : teachers → grades), 
    (∀ g : grades, ∃ t : teachers, f t = g) ∧
    (∃ g : grades, ∀ t : teachers, is_A_or_B t → f t = g)) →
  (fintype.card (subtype (λ f : teachers → grades, 
    (∀ g : grades, ∃ t : teachers, f t = g) ∧
    (∃ g : grades, ∀ t : teachers, is_A_or_B t → f t = g))) = 240) :=
sorry

end teacher_assignment_l22_22737


namespace complex_product_l22_22594

-- Defining the complex numbers and the imaginary unit.
def z1 : ℂ := -1 + 2 * complex.I  -- ℂ is the type for complex numbers and I represents the imaginary unit i
def z2 : ℂ := 2 + complex.I

-- Statement of the proof problem.
theorem complex_product : z1 * z2 = -4 + 3 * complex.I :=
by sorry

end complex_product_l22_22594


namespace vector_sum_eq_c_l22_22901

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (-1, 1)
def c : ℝ × ℝ := (5, 1)

theorem vector_sum_eq_c : (c.1 + a.1 + b.1, c.2 + a.2 + b.2) = c :=
by
  simp [a, b, c]
  sorry

end vector_sum_eq_c_l22_22901


namespace interior_angle_of_regular_octagon_l22_22107

theorem interior_angle_of_regular_octagon :
  let n := 8 in
  let sum_interior_angles := 180 * (n - 2) in
  let angle := sum_interior_angles / n in
  angle = 135 :=
by
  sorry

end interior_angle_of_regular_octagon_l22_22107


namespace train_platform_length_l22_22696

theorem train_platform_length
  (t_platform : ℕ) (t_man : ℕ) (v_km_hr : ℕ) 
  (v : ℝ := v_km_hr * 1000 / 3600)
  (l_train := v * t_man) :
  t_platform = 35 → t_man = 20 → v_km_hr = 54 → 
  ∃ l_platform : ℝ, v * t_platform = l_train + l_platform ∧ l_platform = 225 :=
by
  intros h1 h2 h3
  existsi (v * t_platform - l_train)
  split
  sorry -- Here is where the steps of proof would come in
  sorry -- Placeholder to complete the proof

end train_platform_length_l22_22696


namespace largest_prime_factor_sum_divisors_180_is_7_l22_22988

-- Definitions based on conditions in a)
def prime_factors_180 : list (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1)]

def sum_of_divisors (n : ℕ) : ℕ := 
  let factors := prime_factors_180
  in factors.foldl
       (λ acc pf, match pf with
                  | (p, a) => acc * (list.range (a + 1)).map (λ i, p ^ i).sum
                  end)
       1

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  (nat.factors n).last sorry

-- The theorem we want to prove
theorem largest_prime_factor_sum_divisors_180_is_7 :
  largest_prime_factor (sum_of_divisors 180) = 7 := sorry

end largest_prime_factor_sum_divisors_180_is_7_l22_22988


namespace max_buses_in_city_l22_22940

theorem max_buses_in_city (n : ℕ) (stops : ℕ) (shared : ℕ) (condition1 : n = 9) (condition2 : stops = 3) (condition3 : shared ≤ 1) : n = 12 :=
sorry

end max_buses_in_city_l22_22940


namespace problem_statement_l22_22308

noncomputable def parabolaFocus (a : ℝ) (ha : a > 0) : ℝ × ℝ := (a / 4, 0)

noncomputable def lineEquation (k a : ℝ) (y : ℝ) : ℝ := k * y + a / 4

theorem problem_statement (a k: ℝ) (ha: a > 0) :
  let F := parabolaFocus a ha
  let l (y : ℝ) := lineEquation k a y
  let A := -- assume some intersection computation
  let B := -- assume some intersection computation
  ∃ (AF BF : ℝ), AF = A.1 + a / 4 ∧ BF = B.1 + a / 4 ∧
  (|AF| * |BF|) / (|AF| + |BF|) = a / 4 := by
  sorry

end problem_statement_l22_22308


namespace sum_of_squares_of_solutions_l22_22809

theorem sum_of_squares_of_solutions :
  (∀ x, (abs (x^2 - x + (1 / 2010)) = (1 / 2010)) → (x = 0 ∨ x = 1 ∨ (∃ a b, 
  a + b = 1 ∧ a * b = 1 / 1005 ∧ x = a ∧ x = b))) →
  ((0^2 + 1^2) + (1 - 2 * (1 / 1005)) = (2008 / 1005)) :=
by
  intro h
  sorry

end sum_of_squares_of_solutions_l22_22809


namespace regular_octagon_interior_angle_measure_l22_22253

theorem regular_octagon_interior_angle_measure :
  ∀ (n : ℕ), n = 8 → (∑ i in finset.range n, (180 : ℝ)) / n = 135 :=
  by
  sorry

end regular_octagon_interior_angle_measure_l22_22253


namespace percentage_not_filled_is_approx_42_1300_l22_22721

-- Define the heights and radius
variable (h r : ℝ)

-- Define the total volume of the cone
def total_volume (h r : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

-- Define the height of the water-filled part of the cone
def height_water (h : ℝ) : ℝ := (5/6) * h

-- Define the radius of the water-filled part of the cone due to similarity
def radius_water (r : ℝ) : ℝ := (5/6) * r

-- Define the volume of the water-filled cone
def volume_water (h r : ℝ) : ℝ := (1/3) * Real.pi * (radius_water r)^2 * (height_water h)

-- Define the ratio of the water-filled cone volume to the total cone volume
def ratio_volume_water (h r : ℝ) : ℝ := (volume_water h r) / (total_volume h r)

-- Define the percentage of the cone's volume not filled with water
def percentage_not_filled (h r : ℝ) : ℝ := 100 * (1 - ratio_volume_water h r)

-- The theorem to prove the percentage not filled is approximately 42.1300%
theorem percentage_not_filled_is_approx_42_1300 (h r : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) : 
  42.1295 < percentage_not_filled h r ∧ percentage_not_filled h r < 42.1305 :=
sorry

end percentage_not_filled_is_approx_42_1300_l22_22721


namespace regular_octagon_angle_measure_l22_22205

theorem regular_octagon_angle_measure :
  let n := 8 in
  (180 * (n - 2)) / n = 135 := 
by
  sorry

end regular_octagon_angle_measure_l22_22205


namespace max_odd_integers_l22_22743

theorem max_odd_integers (a b c d e f : ℕ) 
  (hprod : a * b * c * d * e * f % 2 = 0) 
  (hpos_a : 0 < a) (hpos_b : 0 < b) 
  (hpos_c : 0 < c) (hpos_d : 0 < d) 
  (hpos_e : 0 < e) (hpos_f : 0 < f) : 
  ∃ x : ℕ, x ≤ 5 ∧ x = 5 :=
by sorry

end max_odd_integers_l22_22743


namespace area_of_given_rectangle_l22_22948

def side_of_square (area_sq : ℝ) : ℝ :=
  real.sqrt area_sq

def radius_of_circle (area_sq : ℝ) : ℝ :=
  side_of_square area_sq

def length_of_rectangle (radius : ℝ) : ℝ :=
  (2 / 5) * radius

def area_of_rectangle (length : ℝ) (width : ℝ) : ℝ :=
  length * width

theorem area_of_given_rectangle (area_sq : ℝ) (width : ℝ) (h_area_sq : area_sq = 900) (h_width : width = 10) :
  area_of_rectangle (length_of_rectangle (radius_of_circle area_sq)) width = 120 :=
by
  sorry

end area_of_given_rectangle_l22_22948


namespace red_apples_l22_22662

theorem red_apples (R G T : ℕ) (h_G : G = R + 12) (h_T : T = 44) (h_apples : R + G = T) : R = 16 :=
by
  -- Rewrite equations in Lean syntax and assumptions
  have h1 : G = R + 12, from h_G,
  have h2 : T = 44, from h_T,
  have h3 : R + G = T, from h_apples,
  calc
    R = 16 : by sorry

end red_apples_l22_22662


namespace determinant_identity_l22_22433

open Real

theorem determinant_identity (α β : ℝ) : 
  det ![
    ![1, sin α, -cos α],
    ![-sin α, 0, sin (α + β)],
    ![cos α, -sin (α + β), 1]
  ] = sin^2 (α + β) + sin^2 α :=
by
  sorry

end determinant_identity_l22_22433


namespace true_propositions_count_even_l22_22337

theorem true_propositions_count_even (P : Prop) :
  ∃ (n : ℕ), (n = 0 ∨ n = 2 ∨ n = 4) ∧ 
  (list.count true [P, P.converse, P.inverse, P.contrapositive] = n) :=
sorry

end true_propositions_count_even_l22_22337


namespace simplify_complex_expression_l22_22627

open Complex

theorem simplify_complex_expression : (5 - 3 * Complex.i)^2 = 16 - 30 * Complex.i :=
by
  sorry

end simplify_complex_expression_l22_22627


namespace angle_values_l22_22842

theorem angle_values (AB AC BC : ℝ) : 
  AB = 3 → AC = 4 → BC = 5 → 
  ∃ α : ℝ, α = 90 * real.pi / 180 + 30 * real.pi / 180 + real.arcsin (1/3) ∨
           α = 90 * real.pi / 180 + 30 * real.pi / 180 - real.arcsin (1/3) ∨
           α = 90 * real.pi / 180 - 30 * real.pi / 180 + real.arcsin (1/3) ∨
           α = 90 * real.pi / 180 - 30 * real.pi / 180 - real.arcsin (1/3) :=
by sorry

end angle_values_l22_22842


namespace cos_pi_over_5_gt_cos_4pi_over_5_l22_22280

theorem cos_pi_over_5_gt_cos_4pi_over_5 :
  (∀ x ∈ set.Icc (0 : ℝ) real.pi, monotone_decreasing (λ x, real.cos x)) →
  real.sin (real.pi / 5) = real.sin (4 * real.pi / 5) →
  real.cos (real.pi / 5) > real.cos (4 * real.pi / 5) :=
by
  intro h_mono h_sin_eq
  sorry -- Proof is omitted, only the statement is needed

end cos_pi_over_5_gt_cos_4pi_over_5_l22_22280


namespace max_min_values_comparison1_conjecture_correct_l22_22490

section
variables {f : ℝ → ℝ} 

-- Condition declarations
variable (condition1 : f 1 = 3)
variable (condition2 : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → f x ≥ 2)
variable (condition3 : ∀ x1 x2 : ℝ, 0 ≤ x1 → 0 ≤ x2 → x1 + x2 ≤ 1 → f (x1 + x2) ≥ f x1 + f x2 - 2)

-- Theorem statements
theorem max_min_values : (∀ x : ℝ, 0 ≤ x → x ≤ 1 → f x ≥ 2) ∧ f 1 = 3 :=
by
  exact (condition2, condition1)

theorem comparison1 (n : ℕ) (hn : 1 ≤ n) : f (1 / 2^n) ≤ 1 / 2^n + 2 :=
by
  sorry

theorem conjecture_correct (x : ℝ) (hx : 0 < x) (hx1 : x ≤ 1) : f x < 2 * x + 2 :=
by
  sorry
end

end max_min_values_comparison1_conjecture_correct_l22_22490


namespace cone_volume_correct_l22_22879

noncomputable def cone_volume (r l h : ℝ) : ℝ :=
  (1 / 3) * Mathlib.pi * r^2 * h

theorem cone_volume_correct :
  let r := 3
  let angle := (2 * Mathlib.pi) / 3
  let circumference := 2 * Mathlib.pi * r in
  circumference = 6 * Mathlib.pi →
  let l := 9 in
  let h := Real.sqrt (l^2 - r^2) in
  h = 6 * Real.sqrt 2 →
  cone_volume r l h = 18 * Real.sqrt 2 * Mathlib.pi :=
by
  intros
  sorry

end cone_volume_correct_l22_22879


namespace triangle_area_l22_22581

   theorem triangle_area (BC AD: ℝ) (HN HD: ℝ)
     (h_length_BC: BC = 4 * real.sqrt 6)
     (h_length_HN_HD: HN = 6 ∧ HD = 6) :
     ∃ a b c : ℕ, (a + b + c = 52 ∧ 
                   ∃ area: ℝ, area = (a * real.sqrt b) / c ∧ 
                               ((a: ℝ) / (c: ℝ)) ∈ ℚ)
   :=
   sorry
   
end triangle_area_l22_22581


namespace investment_amount_l22_22340

noncomputable def compound_interest_investment (A : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  A / (1 + r / n) ^ (n * t)

theorem investment_amount:
  let A := 50000
  let r := 0.08
  let n := 12
  let t := 3
  let P := compound_interest_investment A r n t
  P ≈ 39410 := 
by
  sorry

end investment_amount_l22_22340


namespace transition_algebraic_expression_l22_22070

theorem transition_algebraic_expression (k : ℕ) (hk : k > 0) :
  (k + 1 + k) * (k + 1 + k + 1) / (k + 1) = 4 * k + 2 :=
sorry

end transition_algebraic_expression_l22_22070


namespace trig_identity_l22_22867

variables {α : ℝ}

theorem trig_identity (h : sin α + 3 * cos α = 0) : 2 * sin (2 * α) - cos α ^ 2 = -13 / 10 := 
sorry

end trig_identity_l22_22867


namespace regular_octagon_interior_angle_l22_22124

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : 
  let sum_interior_angles := 180 * (n - 2)
  let num_sides := 8
  let one_interior_angle := sum_interior_angles / num_sides
  one_interior_angle = 135 :=
by
  sorry

end regular_octagon_interior_angle_l22_22124


namespace total_weight_of_towels_is_40_lbs_l22_22002

def number_of_towels_Mary := 24
def factor_Mary_Frances := 4
def weight_Frances_towels_oz := 128
def pounds_per_ounce := 1 / 16

def number_of_towels_Frances := number_of_towels_Mary / factor_Mary_Frances

def total_number_of_towels := number_of_towels_Mary + number_of_towels_Frances
def weight_per_towel_oz := weight_Frances_towels_oz / number_of_towels_Frances

def total_weight_oz := total_number_of_towels * weight_per_towel_oz
def total_weight_lbs := total_weight_oz * pounds_per_ounce

theorem total_weight_of_towels_is_40_lbs :
  total_weight_lbs = 40 :=
sorry

end total_weight_of_towels_is_40_lbs_l22_22002


namespace number_of_correct_statements_is_2_l22_22691

-- Define each statement as a boolean
def statement1 : Prop := ∀ (A B C : Point), ¬ collinear A B C → exists_circle A B C
def statement2 : Prop := ∀ (O A B : Point), diameter_perpendicular_bisects_chord O A B
def statement3 : Prop := ∀ (O A B C D : Point), subtend_equal_central_angles_equals O A B C D
def statement4 : Prop := ∀ (O A B C : Point), distance_from_circumcenter_is_equal O A B C

-- Sum the number of true statements
def correct_statements_count : ℕ :=
if statement1 then 1 else 0 +
if statement2 then 1 else 0 +
if statement3 then 1 else 0 +
if statement4 then 1 else 0

-- State the theorem to prove there are 2 correct statements
theorem number_of_correct_statements_is_2 :
  correct_statements_count = 2 :=
by {
  -- State the condition of each statement correctness based on the given problem.
  have statement1_incorrect : ¬ statement1,
  sorry,
  have statement2_correct : statement2,
  sorry,
  have statement3_incorrect : ¬ statement3,
  sorry,
  have statement4_correct : statement4,
  sorry,
  -- Use these conditions to establish the number of correct statements.
  unfold correct_statements_count,
  split_ifs,
  -- Here we expect the correct answer derived from the conditions above.
  exact dec_trivial,
}

end number_of_correct_statements_is_2_l22_22691


namespace interior_angle_of_regular_octagon_l22_22108

theorem interior_angle_of_regular_octagon :
  let n := 8 in
  let sum_interior_angles := 180 * (n - 2) in
  let angle := sum_interior_angles / n in
  angle = 135 :=
by
  sorry

end interior_angle_of_regular_octagon_l22_22108


namespace beavers_working_l22_22292

theorem beavers_working (a b : ℝ) (h₁ : a = 2.0) (h₂ : b = 1.0) : a + b = 3.0 := 
by 
  rw [h₁, h₂]
  norm_num

end beavers_working_l22_22292


namespace calc_problem_l22_22760

-- Define the components of the problem
def a := 32^6
def b := (7 : ℝ)/(5 : ℝ) * (49 / 25 : ℝ)^(-1/2 : ℝ)
def c := log 10 (1 / 10 : ℝ)

-- State the theorem that encapsulates the given problem and its solution.
theorem calc_problem : a - b - c = 4 := by
  sorry

end calc_problem_l22_22760


namespace graph_func_1_graph_func_2_graph_func_3_graph_func_4_l22_22709

-- Function 1
theorem graph_func_1 (x : ℝ) (h : abs x ≤ 2) : (4 - x^2 = y) :=
begin
  sorry
end

-- Function 2
theorem graph_func_2 (x : ℝ) (h : true) : 
  y = if x ≥ 0 then x^2 + 2 else 2 :=
begin
  sorry
end

-- Function 3
theorem graph_func_3 (x : ℝ) (h : x ≠ 0) : 
  y = if x > 0 then (x^2 - 1) else (-x^2 + 1) :=
begin
  sorry
end

-- Function 4
theorem graph_func_4 (x : ℝ) (h : true) : 
  y = if x ≥ 0 then (x * x - 2 * x) else (-x * x + 2 * x) :=
begin
  sorry
end

end graph_func_1_graph_func_2_graph_func_3_graph_func_4_l22_22709


namespace interior_angle_of_regular_octagon_l22_22117

theorem interior_angle_of_regular_octagon :
  let n := 8 in
  let sum_interior_angles := 180 * (n - 2) in
  let angle := sum_interior_angles / n in
  angle = 135 :=
by
  sorry

end interior_angle_of_regular_octagon_l22_22117


namespace bacterium_splits_l22_22712

theorem bacterium_splits (x : ℕ) 
  (h1 : ∀ n : ℕ, n = 1 → ∃ m : ℕ, m = 4096 * 10^6)
  (h2 : ∀ g : ℕ, g = 7 → (1 * ((x / 2) ^ 6)) = 4096 * 10^6) : 
  x = 80 := by
sorrcode


end bacterium_splits_l22_22712


namespace find_function_g_l22_22796

noncomputable def g (x : ℝ) : ℝ := (5^x - 3^x) / 8

theorem find_function_g (x y : ℝ) (h1 : g 2 = 2) (h2 : ∀ x y : ℝ, g (x + y) = 5^y * g x + 3^x * g y) :
  g x = (5^x - 3^x) / 8 :=
by
  sorry

end find_function_g_l22_22796


namespace regular_octagon_interior_angle_l22_22120

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : 
  let sum_interior_angles := 180 * (n - 2)
  let num_sides := 8
  let one_interior_angle := sum_interior_angles / num_sides
  one_interior_angle = 135 :=
by
  sorry

end regular_octagon_interior_angle_l22_22120


namespace investment_difference_l22_22003

def mauriceInvestment (P r t : ℝ) : ℝ := P * (1 + r)^t
def rickyInvestment (P r : ℝ) (n : ℕ) : ℝ := P * (1 + r / 12)^(12 * n)

theorem investment_difference :
  let P := 65000
  let r := 0.05
  let t := 3
  let monthlyRate := (r / 12)
  let months := 12 * t
  let Maurice := mauriceInvestment P r t
  let Ricky := rickyInvestment P monthlyRate t
  let difference := Ricky - Maurice
  Int.floor difference = 264 := 
by
  sorry

end investment_difference_l22_22003


namespace scientific_notation_470M_l22_22012

theorem scientific_notation_470M :
  (470000000 : ℝ) = 4.7 * 10^8 :=
sorry

end scientific_notation_470M_l22_22012


namespace floor_diff_l22_22431

theorem floor_diff (x : ℝ) (h : x = 13.2) : 
  (Int.floor (x^2) - (Int.floor x) * (Int.floor x) = 5) := by
  sorry

end floor_diff_l22_22431


namespace regular_octagon_interior_angle_l22_22188

theorem regular_octagon_interior_angle : 
  ∀ (n : ℕ), n = 8 → (sum_of_internal_angles n) / n = 135 := 
by
  intros n hn
  rw hn
  sorry

end regular_octagon_interior_angle_l22_22188


namespace sum_of_six_distinct_roots_l22_22640

variable {ℝ : Type*} [Nonempty ℝ] [TopologicalSpace ℝ]

-- Translation of the mathematical conditions into Lean 4
def g (x : ℝ) : ℝ := sorry

lemma g_symmetric (x : ℝ) : g (3 + x) = g (3 - x) := sorry

-- The actual statement of the problem to be proved
theorem sum_of_six_distinct_roots (roots : Finset ℝ) (h : roots.card = 6) (h_roots : ∀ x ∈ roots, g x = 0) :
  roots.sum (λ x, x) = 18 :=
begin
  sorry
end

end sum_of_six_distinct_roots_l22_22640


namespace students_present_in_class_l22_22287

theorem students_present_in_class :
  ∀ (total_students absent_percentage : ℕ), 
    total_students = 50 → absent_percentage = 12 → 
    (88 * total_students / 100) = 44 :=
by
  intros total_students absent_percentage h1 h2
  sorry

end students_present_in_class_l22_22287


namespace sum_of_squares_fraction_l22_22396

theorem sum_of_squares_fraction (n : ℕ) (h : n > 0) : 
  (∑ i in Finset.range n, (i + 1)^2 / ((2 * (i + 1) - 1) * (2 * (i + 1) + 1) : ℝ)) = (n^2 + n) / (4 * n + 2) := 
sorry

end sum_of_squares_fraction_l22_22396


namespace binary_decimal_conversion_correct_l22_22769

-- Define the binary number
def binaryNum : list ℕ := [1, 0, 1, 0, 1, 0, 1]

-- Noncomputable definition to represent the decimal value conversion from binary
noncomputable def binaryToDecimal (bn : list ℕ) : ℕ :=
  bn.foldr (λ (bit posSum : ℕ) (acc : ℕ), acc + bit * 2 ^ posSum) 0

-- Statement to prove
theorem binary_decimal_conversion_correct :
  binaryToDecimal binaryNum = 85 :=
by
  sorry

end binary_decimal_conversion_correct_l22_22769


namespace school_students_and_capacity_l22_22580

theorem school_students_and_capacity :
  let schools_230 := 10 in
  let students_per_school_230 := 230 in
  let schools_275 := 5 in
  let students_per_school_275 := 275 in
  let schools_180 := 3 in
  let students_per_school_180 := 180 in
  let schools_260 := 7 in
  let students_per_school_260 := 260 in
  let transfer_students := 15 in
  let total_students_before_transfers := 
    (schools_230 * students_per_school_230) + 
    (schools_275 * students_per_school_275) + 
    (schools_180 * students_per_school_180) + 
    (schools_260 * students_per_school_260) in
  let total_students_after_transfers := total_students_before_transfers + transfer_students in
  let max_capacity_250 := 4 in
  let max_capacity_300 := 2 in
  let max_capacity_200 := 1 in
  let num_max_capacity_schools := 0 in
  total_students_before_transfers = 6035 ∧ 
  total_students_after_transfers = 6050 ∧
  num_max_capacity_schools = 0 :=
by sorry

end school_students_and_capacity_l22_22580


namespace floor_difference_l22_22406

theorem floor_difference (x : ℝ) (h : x = 13.2) : 
  (⌊x^2⌋ : ℤ) - ((⌊x⌋ : ℤ) * (⌊x⌋ : ℤ)) = 5 := 
by
  -- proof skipped
  sorry

end floor_difference_l22_22406


namespace secant_225_equals_neg_sqrt_two_l22_22444

/-- Definition of secant in terms of cosine -/
def sec (θ : ℝ) := 1 / Real.cos θ

/-- Given angles in radians for trig identities -/
def degree_to_radian (d : ℝ) := d * Real.pi / 180

/-- Main theorem statement -/
theorem secant_225_equals_neg_sqrt_two : sec (degree_to_radian 225) = -Real.sqrt 2 :=
by
  sorry

end secant_225_equals_neg_sqrt_two_l22_22444


namespace floor_sq_minus_sq_floor_l22_22421

noncomputable def floor_13_2 : ℤ := int.floor 13.2

theorem floor_sq_minus_sq_floor :
  int.floor ((13.2 : ℝ) ^ 2) - (floor_13_2 * floor_13_2) = 5 :=
by
  let floor_13_2_sq := floor_13_2 * floor_13_2
  have h1 : int.floor (13.2 : ℝ) = 13 := by norm_num
  have h2 : int.floor ((13.2 : ℝ) ^ 2) = 174 := by norm_num
  rw [h1] at floor_13_2
  rw [h2]
  rw [floor_13_2, floor_13_2, floor_13_2]
  exact sorry

end floor_sq_minus_sq_floor_l22_22421


namespace NY_Mets_count_l22_22562

-- Define the conditions as variables
variables {Y M B : ℕ}

-- Define the ratios
def ratio_Y_M : Prop := 3 * M = 2 * Y
def ratio_M_B : Prop := 4 * B = 5 * M

-- Define the total number of fans
def total_fans : Prop := Y + M + B = 390

-- Mathematically equivalent proof problem (Question == Answer)
theorem NY_Mets_count : ratio_Y_M → ratio_M_B → total_fans → M = 104 :=
by
  intros hYM hMB h_total
  -- Proof steps to be filled in
  sorry

end NY_Mets_count_l22_22562


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22193

theorem measure_of_one_interior_angle_of_regular_octagon 
  (n : Nat) (h_n : n = 8) : 
  (one_interior_angle : ℝ) = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22193


namespace derivative_y_partial_derivative_p_x_partial_derivative_p_y_derivative_z_l22_22795

-- Problem 1
theorem derivative_y ( x : ℝ ) : 
  let u := sin x in
  let v := cos x in
  let y := u^2 * exp v in
  deriv y = 2 * sin x * cos x * exp (cos x) - (sin x)^3 * exp (cos x) :=
sorry

-- Problem 2 - Part 1
theorem partial_derivative_p_x (x y : ℝ) : 
  let u := log (x - y) in
  let v := exp (x / y) in
  let p := u^v in
  partial_deriv p x = v * u^(v-1) * (1 / (x - y)) + u^v * log u * (exp (x / y) / y) :=
sorry

-- Problem 2 - Part 2
theorem partial_derivative_p_y (x y : ℝ) : 
  let u := log (x - y) in
  let v := exp (x / y) in
  let p := u^v in
  partial_deriv p y = v * u^(v-1) * (-1 / (x - y)) + u^v * log u * ((-x / y^2) * exp (x / y)) :=
sorry

-- Problem 3
theorem derivative_z (x : ℝ) : 
  let v := log (x^2 + 1) in
  let w := -sqrt (1 - x^2) in
  let z := x * sin v * cos w in
  deriv z x = sin v * cos w + x * cos v * cos w * (2x / (x^2 + 1)) - x * sin v * sin w * (x / sqrt (1 - x^2)) :=
sorry

end derivative_y_partial_derivative_p_x_partial_derivative_p_y_derivative_z_l22_22795


namespace valid_third_side_l22_22504

-- Define a structure for the triangle with given sides
structure Triangle where
  a : ℝ
  b : ℝ
  x : ℝ

-- Define the conditions using the triangle inequality theorem
def valid_triangle (T : Triangle) : Prop :=
  T.a + T.x > T.b ∧ T.b + T.x > T.a ∧ T.a + T.b > T.x

-- Given values of a and b, and the condition on x
def specific_triangle : Triangle :=
  { a := 4, b := 9, x := 6 }

-- Statement to prove valid_triangle holds for specific_triangle
theorem valid_third_side : valid_triangle specific_triangle :=
by
  -- Import or assumptions about inequalities can be skipped or replaced by sorry
  sorry

end valid_third_side_l22_22504


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22191

theorem measure_of_one_interior_angle_of_regular_octagon 
  (n : Nat) (h_n : n = 8) : 
  (one_interior_angle : ℝ) = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22191


namespace regular_octagon_interior_angle_l22_22222

theorem regular_octagon_interior_angle:
  let n := 8
  let sum_interior_angles (n: ℕ) := 180 * (n - 2)
  let interior_angle (n: ℕ) := sum_interior_angles n / n
  interior_angle n = 135 :=
by
  sorry

end regular_octagon_interior_angle_l22_22222


namespace fg_of_neg2_l22_22515

def f (x : ℤ) : ℤ := x^2 + 4
def g (x : ℤ) : ℤ := 3 * x + 2

theorem fg_of_neg2 : f (g (-2)) = 20 := by
  sorry

end fg_of_neg2_l22_22515


namespace nested_radicals_equivalent_l22_22361

theorem nested_radicals_equivalent :
  (∃ (f : ℕ → ℕ),
    (∀ k, 1 ≤ k → f k = (k + 1)^2 - 1) ∧
    (sqrt (1 + 2 * sqrt (1 + 3 * sqrt (1 + 2017 * sqrt (1 + 2018 * (f 2018))))) = 3)) :=
begin
  use (λ k, (k + 1)^2 - 1),
  split,
  { intros k hk,
    exact rfl, },
  { sorry },
end

end nested_radicals_equivalent_l22_22361


namespace regular_octagon_interior_angle_l22_22092

theorem regular_octagon_interior_angle (n : ℕ) (h₁ : n = 8) :
  (180 * (n - 2)) / n = 135 :=
by
  rw [h₁]
  sorry

end regular_octagon_interior_angle_l22_22092


namespace maximum_buses_l22_22924

-- Definition of the problem constraints as conditions in Lean
variables (bus : Type) (stop : Type) (buses : set bus) (stops : set stop)
variables (stops_served_by_bus : bus → finset stop)
variables (stop_occurrences : stop → finset bus)

-- Conditions
def has_9_stops (stops : set stop) : Prop := stops.card = 9
def bus_serves_3_stops (b : bus) : Prop := (stops_served_by_bus b).card = 3
def at_most_1_common_stop (b1 b2 : bus) (h1 : b1 ∈ buses) (h2 : b2 ∈ buses) : Prop :=
  (stops_served_by_bus b1 ∩ stops_served_by_bus b2).card ≤ 1

-- Goal
theorem maximum_buses (h1 : has_9_stops stops)
                      (h2 : ∀ b ∈ buses, bus_serves_3_stops b)
                      (h3 : ∀ b1 b2 ∈ buses, at_most_1_common_stop b1 b2 h1 h2) : 
  buses.card ≤ 12 :=
sorry

end maximum_buses_l22_22924


namespace floor_sq_minus_sq_floor_l22_22424

noncomputable def floor_13_2 : ℤ := int.floor 13.2

theorem floor_sq_minus_sq_floor :
  int.floor ((13.2 : ℝ) ^ 2) - (floor_13_2 * floor_13_2) = 5 :=
by
  let floor_13_2_sq := floor_13_2 * floor_13_2
  have h1 : int.floor (13.2 : ℝ) = 13 := by norm_num
  have h2 : int.floor ((13.2 : ℝ) ^ 2) = 174 := by norm_num
  rw [h1] at floor_13_2
  rw [h2]
  rw [floor_13_2, floor_13_2, floor_13_2]
  exact sorry

end floor_sq_minus_sq_floor_l22_22424


namespace least_not_lucky_multiple_of_6_l22_22311

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

theorem least_not_lucky_multiple_of_6 : ∃ k : ℕ, k > 0 ∧ k % 6 = 0 ∧ ¬ is_lucky k ∧ ∀ m : ℕ, m > 0 ∧ m % 6 = 0 ∧ ¬ is_lucky m → k ≤ m :=
  sorry

end least_not_lucky_multiple_of_6_l22_22311


namespace bruce_total_payment_l22_22755

-- Definitions for the quantities of fruits and their costs per kilogram
def kgGrapes := 8
def priceGrapesPerKg := 70
def kgMangoes := 9
def priceMangoesPerKg := 55
def kgOranges := 5
def priceOrangesPerKg := 40
def kgStrawberries := 4
def priceStrawberriesPerKg := 90
def kgPineapples := 6
def pricePineapplesPerKg := 45

-- Definitions for discount and tax rates
def discountRate := 0.10
def taxRate := 0.05

-- Calculation of the total cost before discount and tax
def totalCostBeforeDiscount := (kgGrapes * priceGrapesPerKg) +
                               (kgMangoes * priceMangoesPerKg) +
                               (kgOranges * priceOrangesPerKg) +
                               (kgStrawberries * priceStrawberriesPerKg) +
                               (kgPineapples * pricePineapplesPerKg)

-- Applying the discount
def discountAmount := totalCostBeforeDiscount * discountRate
def totalAfterDiscount := totalCostBeforeDiscount - discountAmount

-- Applying the tax
def taxAmount := totalAfterDiscount * taxRate
def finalAmount := totalAfterDiscount + taxAmount

-- The proof problem statement
theorem bruce_total_payment : finalAmount = 1781.33 :=
sorry

end bruce_total_payment_l22_22755


namespace regular_octagon_angle_measure_l22_22214

theorem regular_octagon_angle_measure :
  let n := 8 in
  (180 * (n - 2)) / n = 135 := 
by
  sorry

end regular_octagon_angle_measure_l22_22214


namespace preimage_f_l22_22503

/-- Given a function f that maps (x, y) to (x - y, x + y), 
  the preimage of (2, -4) under f is (-1, -3) -/
theorem preimage_f {a b : ℝ} (f : ℝ × ℝ → ℝ × ℝ) (h : f = λ p, (p.1 - p.2, p.1 + p.2)) :
  f (a, b) = (2, -4) → (a, b) = (-1, -3) := by
  sorry

end preimage_f_l22_22503


namespace sum_of_squares_of_solutions_l22_22811

theorem sum_of_squares_of_solutions :
  (∀ x, (abs (x^2 - x + (1 / 2010)) = (1 / 2010)) → (x = 0 ∨ x = 1 ∨ (∃ a b, 
  a + b = 1 ∧ a * b = 1 / 1005 ∧ x = a ∧ x = b))) →
  ((0^2 + 1^2) + (1 - 2 * (1 / 1005)) = (2008 / 1005)) :=
by
  intro h
  sorry

end sum_of_squares_of_solutions_l22_22811


namespace blanch_slices_eaten_for_dinner_l22_22753

theorem blanch_slices_eaten_for_dinner :
  ∀ (total_slices eaten_breakfast eaten_lunch eaten_snack slices_left eaten_dinner : ℕ),
  total_slices = 15 →
  eaten_breakfast = 4 →
  eaten_lunch = 2 →
  eaten_snack = 2 →
  slices_left = 2 →
  eaten_dinner = total_slices - (eaten_breakfast + eaten_lunch + eaten_snack) - slices_left →
  eaten_dinner = 5 := by
  intros total_slices eaten_breakfast eaten_lunch eaten_snack slices_left eaten_dinner
  intros h_total_slices h_eaten_breakfast h_eaten_lunch h_eaten_snack h_slices_left h_eaten_dinner
  rw [h_total_slices, h_eaten_breakfast, h_eaten_lunch, h_eaten_snack, h_slices_left] at h_eaten_dinner
  exact h_eaten_dinner

end blanch_slices_eaten_for_dinner_l22_22753


namespace value_of_expression_l22_22648

-- Conditions
def isOdd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def isIncreasingOn (f : ℝ → ℝ) (a b : ℝ) := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y
def hasMaxOn (f : ℝ → ℝ) (a b : ℝ) (M : ℝ) := ∃ x, a ≤ x ∧ x ≤ b ∧ f x = M
def hasMinOn (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) := ∃ x, a ≤ x ∧ x ≤ b ∧ f x = m

-- Proof statement
theorem value_of_expression (f : ℝ → ℝ) 
  (hf1 : isOdd f)
  (hf2 : isIncreasingOn f 3 7)
  (hf3 : hasMaxOn f 3 6 8)
  (hf4 : hasMinOn f 3 6 (-1)) :
  2 * f (-6) + f (-3) = -15 :=
sorry

end value_of_expression_l22_22648


namespace covering_schemes_eq_fibonacci_l22_22674

-- Definitions of the recurrence relation and initial conditions for F_n
def F : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := F(n+1) + F(n)

-- Main theorem stating that F_n satisfies the Fibonacci-like recurrence
theorem covering_schemes_eq_fibonacci (n : ℕ) : 
  F n = nat.fib (n + 1) := sorry

end covering_schemes_eq_fibonacci_l22_22674


namespace sec_225_eq_neg_sqrt2_l22_22462

theorem sec_225_eq_neg_sqrt2 : ∀ θ : ℝ, θ = 225 * (π / 180) → sec θ = -real.sqrt 2 :=
by
  intro θ hθ
  sorry

end sec_225_eq_neg_sqrt2_l22_22462


namespace interior_angle_regular_octagon_l22_22080

theorem interior_angle_regular_octagon (n : ℕ) (h : n = 8) : 
  let sum_of_interior_angles := 180 * (n - 2)
  let number_of_sides := n
  let one_interior_angle := sum_of_interior_angles / number_of_sides
  in one_interior_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l22_22080


namespace simplify_and_rationalize_denominator_l22_22629

theorem simplify_and_rationalize_denominator :
  (∀ (a b c : ℝ), a = (3 : ℝ) ∧ b = 10010 ∧ c = 1001 → 
  (∃ (x y z : ℝ), x = a ∧ y = b ∧ z = c → 
  ((a * (Real.sqrt b)) / c = (3 * Real.sqrt 10010) / 1001))) :=
by
  intro a b c h
  obtain ⟨ha, hb, hc⟩ := h
  use [a, b, c]
  simp [ha, hb, hc]
  sorry

end simplify_and_rationalize_denominator_l22_22629


namespace sin_B_eq_sin_C_given_conditions_height_from_A_given_conditions_l22_22962

noncomputable def triangle_height_obtuse (sin_A sin_C : ℝ) (b : ℝ) (cos_2B : ℝ) : ℝ :=
  let h := (√3 * sin_C) * sin_C
  h

theorem sin_B_eq_sin_C_given_conditions (sin_A sin_C : ℝ) (b : ℝ) (B : ℝ) :
  sin_A = √3 * sin_C → b = √7 → B = π / 6 → sin B = sin C :=
sorry

theorem height_from_A_given_conditions (sin_A sin_C : ℝ) (b : ℝ) (B : ℝ) :
  sin_A = √3 * sin_C → b = √7 → B > π / 2 → cos 2B = 1 / 2 → 
  let h := triangle_height_obtuse sin_A sin_C b (1 / 2) in 
  h = √21 / 14  :=
sorry

end sin_B_eq_sin_C_given_conditions_height_from_A_given_conditions_l22_22962


namespace correct_propositions_l22_22746

theorem correct_propositions :
  (∀ (P : Type) (skew1 skew2 : P), ∃ (plane : P), (∀ (point : P), plane = point) → False) ∧
  (∀ (plane : Type) (a b : plane), (a ∩ plane = a) ∧ (b ⟂ a) → (b ⟂ plane) → False) ∧
  (∀ (prism : Type), (∃ (lateral1 lateral2 : prism), (lateral1 ⟂ lateral2) ∧ (lateral2 ⟂ lateral2))
  → False) ∧
  (∀ (tetra : Type) (e1 e2 e3 e4 e5 e6 : tetra),
    (e1 ⟂ e2) ∧ (e3 ⟂ e4) → (e5 ⟂ e6)) ∧
  (∀ (tetra : Type) (face1 face2 face3 face4 : tetra),
    (face1 ∈ (right_triangle tetra)) ∧ (face2 ∈ (right_triangle tetra)) ∧ 
    (face3 ∈ (right_triangle tetra)) ∧ (face4 ∈ (right_triangle tetra))) := 
by {
  sorry
}

end correct_propositions_l22_22746


namespace five_letter_word_combinations_l22_22970

open Nat

theorem five_letter_word_combinations :
  let first_letter_choices := 26
  let other_letter_choices := 26
  let total_combinations := first_letter_choices ^ 1 * other_letter_choices ^ 3
  total_combinations = 456976 := 
by
  let first_letter_choices := 26
  let other_letter_choices := 26
  let total_combinations := first_letter_choices ^ 1 * other_letter_choices ^ 3
  show total_combinations = 456976
  sorry

end five_letter_word_combinations_l22_22970


namespace max_n_value_l22_22661

theorem max_n_value (participants : ℕ) (plays_game : ℕ → ℕ → Prop) (n : ℕ) : 
  participants = 300 →
  (∀ a b, plays_game a b → plays_game b a) → -- game is symmetric
  (∀ a b, plays_game a b → a ≠ b) → -- no self-loops
  (∀ a b c, a ≠ b ∧ b ≠ c ∧ c ≠ a → ¬(plays_game a b ∧ plays_game b c ∧ plays_game c a)) → -- no triangles
  (∀ p, (∃ k, 1 ≤ k ∧ k ≤ n ∧ (∃ N, (N = plays_game p) ∧ N = k))) →
  (∀ p, plays_game p ≤ n) →
  n = 200 :=
sorry

end max_n_value_l22_22661


namespace walking_distances_l22_22598

theorem walking_distances:
    let total_time := 120 -- in minutes
    let nadia_speed := 6 -- in km/h
    let hannah_speed := 4 -- in km/h
    let ethan_speed := 5 -- in km/h
    let nadia_breaks := 30 -- total break time in minutes
    let hannah_break := 15 -- break time in minutes
    let ethan_breaks := 30 -- total break time in minutes
    let effective_walking_time (total_time: ℕ) (break_time: ℕ) := (total_time - break_time) / 60.0 -- in hours
    let nadia_walk_time := effective_walking_time total_time nadia_breaks
    let hannah_walk_time := effective_walking_time total_time hannah_break
    let ethan_walk_time := effective_walking_time total_time ethan_breaks
    let distance_walked (speed: ℕ) (time: ℚ) := speed * time -- in km
    let nadia_distance := distance_walked nadia_speed nadia_walk_time
    let hannah_distance := distance_walked hannah_speed hannah_walk_time
    let ethan_distance := distance_walked ethan_speed ethan_walk_time
    let total_distance := nadia_distance + hannah_distance + ethan_distance
    nadia_walk_time = 1.5 ∧ hannah_walk_time = 1.75 ∧ ethan_walk_time = 1.5 ∧ 
    nadia_distance = 9 ∧ hannah_distance = 7 ∧ ethan_distance = 7.5 ∧ 
    total_distance = 23.5 :=
begin
    sorry
end

end walking_distances_l22_22598


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22163

-- Define the conditions for the problem
def num_sides : ℕ := 8
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Statement of the problem
theorem measure_of_one_interior_angle_of_regular_octagon : 
  (sum_of_interior_angles num_sides) / num_sides = 135 :=
by
  -- Conditions are defined above
  -- Proof steps are omitted
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22163


namespace nested_radicals_equivalent_l22_22359

theorem nested_radicals_equivalent :
  (∃ (f : ℕ → ℕ),
    (∀ k, 1 ≤ k → f k = (k + 1)^2 - 1) ∧
    (sqrt (1 + 2 * sqrt (1 + 3 * sqrt (1 + 2017 * sqrt (1 + 2018 * (f 2018))))) = 3)) :=
begin
  use (λ k, (k + 1)^2 - 1),
  split,
  { intros k hk,
    exact rfl, },
  { sorry },
end

end nested_radicals_equivalent_l22_22359


namespace shape_to_square_l22_22387

-- Define the shape and partition properties
def shape : Type := -- define the type representing the initial shape, e.g., a subset of a grid.
sorry

def part (S : shape) : Type := -- define the type representing parts of the initial shape
sorry

-- Define the main theorem statement
theorem shape_to_square (S : shape) :
  ∃ (P1 P2 P3 : part S), 
    (S = P1 ∪ P2 ∪ P3) ∧
    (P1 ∩ P2 = ∅) ∧ (P2 ∩ P3 = ∅) ∧ (P1 ∩ P3 = ∅) ∧
    -- Additional properties ensuring these parts can form a square
    (can_rotate_to_form_square P1 P2 P3) :=
sorry

end shape_to_square_l22_22387


namespace fraction_work_AC_l22_22335

theorem fraction_work_AC (total_payment Rs B_payment : ℝ)
  (payment_AC : ℝ)
  (h1 : total_payment = 529)
  (h2 : B_payment = 12)
  (h3 : payment_AC = total_payment - B_payment) : 
  payment_AC / total_payment = 517 / 529 :=
by
  rw [h1, h2] at h3
  rw [h3]
  norm_num
  sorry

end fraction_work_AC_l22_22335


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22150

theorem measure_of_one_interior_angle_of_regular_octagon :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 := 
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22150


namespace mutually_exclusive_events_A_B_l22_22669

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

def A : set ℕ := {n | is_odd n ∧ n ∈ {1, 2, 3, 4, 5, 6}}
def B : set ℕ := {n | is_even n ∧ n ∈ {1, 2, 3, 4, 5, 6}}

theorem mutually_exclusive_events_A_B : 
  disjoint A B :=
by sorry

end mutually_exclusive_events_A_B_l22_22669


namespace sin2A_div_sinC_eq_8_div_5_l22_22916

theorem sin2A_div_sinC_eq_8_div_5
  (a b : ℝ) (cosC : ℝ)
  (h_a : a = 4) (h_b : b = 5) (h_cosC : cosC = 4 / 5) :
  ∃ (A C : ℝ), sin(2 * A) / sin(C) = 8 / 5 :=
by
  sorry

end sin2A_div_sinC_eq_8_div_5_l22_22916


namespace nested_radical_simplification_l22_22368

theorem nested_radical_simplification : 
  (sqrt (1 + 2 * sqrt (1 + 3 * sqrt (1 + ⋯ + 2017 * sqrt (1 + 2018 * 2020))))) = 3 :=
by
  -- sorry to skip the proof
  sorry

end nested_radical_simplification_l22_22368


namespace correct_statement_l22_22692

theorem correct_statement :
  (∀ a b c : ℕ, (a = 2 ∧ b = 2 ∧ c = 3) → a < c ∧ b < c → (2 * x ^ a - 3 * x * y + 5 * x * y ^ c = 2 * x ^ a - 3 * x * y + 5 * x * y ^ c)) ∧
  (2 * x ^ 2 - 3 * x * y + 5 * x * y ^ 2 = 2 * x ^ 2 - 3 * x * y + 5 * x * y ^ 2) ∧
  (∀ n m k : ℕ, (n = 2 ∧ m = 2 ∧ k = 3) → max n m < k → polynomial.degree (2 * x ^ n - 3 * x * y + 5 * x * y ^ k) = k) :=
by
  sorry

end correct_statement_l22_22692


namespace sum_of_proper_divisors_72_l22_22683

def divisors (n : ℕ) : List ℕ :=
  List.filter (λ d, n % d = 0) (List.range (n + 1))

def properDivisors (n : ℕ) : List ℕ :=
  List.filter (λ d, d ≠ n) (divisors n)

def sumProperDivisors (n : ℕ) : ℕ :=
  List.sum (properDivisors n)

theorem sum_of_proper_divisors_72 : sumProperDivisors 72 = 123 :=
by
  sorry

end sum_of_proper_divisors_72_l22_22683


namespace find_divisor_l22_22554

-- Definitions from the conditions
def remainder : ℤ := 8
def quotient : ℤ := 43
def dividend : ℤ := 997
def is_prime (n : ℤ) : Prop := n ≠ 1 ∧ (∀ d : ℤ, d ∣ n → d = 1 ∨ d = n)

-- The proof problem statement
theorem find_divisor (d : ℤ) 
  (hd : is_prime d) 
  (hdiv : dividend = (d * quotient) + remainder) : 
  d = 23 := 
sorry

end find_divisor_l22_22554


namespace sum_of_squares_of_solutions_l22_22807

theorem sum_of_squares_of_solutions :
  (∀ x, (abs (x^2 - x + (1 / 2010)) = (1 / 2010)) → (x = 0 ∨ x = 1 ∨ (∃ a b, 
  a + b = 1 ∧ a * b = 1 / 1005 ∧ x = a ∧ x = b))) →
  ((0^2 + 1^2) + (1 - 2 * (1 / 1005)) = (2008 / 1005)) :=
by
  intro h
  sorry

end sum_of_squares_of_solutions_l22_22807


namespace probability_function_increasing_l22_22276

theorem probability_function_increasing : 
  let outcomes := [(m, n) | m ∈ Finset.range(1, 7), n ∈ Finset.range(1, 7)],
      condition := (fun (mn: ℕ × ℕ) => let (m, n) := mn in (2 * m - n ≤ 6)),
      favorable := filter condition outcomes in
  (favorable.card : ℚ) / (outcomes.card : ℚ) = 3 / 4 :=
by sorry

end probability_function_increasing_l22_22276


namespace Sue_button_count_l22_22000

variable (K S : ℕ)

theorem Sue_button_count (H1 : 64 = 5 * K + 4) (H2 : S = K / 2) : S = 6 := 
by
sorry

end Sue_button_count_l22_22000


namespace parabola_vertex_l22_22468

theorem parabola_vertex (a b c : ℝ) :
  (∀ x, y = ax^2 + bx + c ↔ 
   y = a*((x+3)^2) + 4) ∧
   (∀ x y, (x, y) = ((1:ℝ), (2:ℝ))) →
   a + b + c = 3 := by
  sorry

end parabola_vertex_l22_22468


namespace correct_statement_l22_22336

-- Defining the conditions
def freq_eq_prob : Prop :=
  ∀ (f p : ℝ), f = p

def freq_objective : Prop :=
  ∀ (f : ℝ) (n : ℕ), f = f

def freq_stabilizes : Prop :=
  ∀ (p : ℝ), ∃ (f : ℝ) (n : ℕ), f = p

def prob_random : Prop :=
  ∀ (p : ℝ), p = p

-- The statement we need to prove
theorem correct_statement :
  ¬freq_eq_prob ∧ ¬freq_objective ∧ freq_stabilizes ∧ ¬prob_random :=
by
  sorry

end correct_statement_l22_22336


namespace sum_of_squares_of_solutions_l22_22824

theorem sum_of_squares_of_solutions :
  let a := (1 : ℝ) / 2010
  in (∀ x : ℝ, abs(x^2 - x + a) = a) → 
     ((0^2 + 1^2) + ((((1*1) - 2 * a) + 1) / a)) = 2008 / 1005 :=
by
  intros a h
  sorry

end sum_of_squares_of_solutions_l22_22824


namespace incorrect_statement_l22_22592

variables {α β γ : Type} [plane α] [plane β] [plane γ]
variable  (l : line)

-- Assuming given conditions as hypotheses
variables (h1 : α ⊥ β)
variables (h2 : α ⊥ γ)
variables (h3 : β ⊥ γ)
variables (h4 : line_of_intersection α β l)

-- In Lean 4, we need to show that the statement "If α ⊥ β, then all lines within α are perpendicular to β" is incorrect
theorem incorrect_statement {h1 : α ⊥ β} 
: ¬ (∀ (l' : line within α), l' ⊥ β) :=
sorry

end incorrect_statement_l22_22592


namespace initial_charge_first_mile_l22_22717

noncomputable def initial_charge (total_distance : ℝ) (total_cost : ℝ) (additional_charge : ℝ) : ℝ :=
  let num_increments := (total_distance * 5) - 1 in
  total_cost - (num_increments * additional_charge)

theorem initial_charge_first_mile :
  initial_charge 8 19.1 0.4 = 3.5 :=
by
  rw [initial_charge]
  sorry

end initial_charge_first_mile_l22_22717


namespace probability_point_closer_to_center_l22_22733

theorem probability_point_closer_to_center (r : ℝ) (h : r = 4) :
  let A_outer := π * r ^ 2,
      A_inner := π * (sqrt r) ^ 2 in
  (A_inner / A_outer) = 1 / 4 :=
by
  sorry

end probability_point_closer_to_center_l22_22733


namespace regular_octagon_interior_angle_l22_22126

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : 
  let sum_interior_angles := 180 * (n - 2)
  let num_sides := 8
  let one_interior_angle := sum_interior_angles / num_sides
  one_interior_angle = 135 :=
by
  sorry

end regular_octagon_interior_angle_l22_22126


namespace isosceles_triangle_base_length_l22_22864

noncomputable def length_of_base : ℕ → ℕ → ℕ := λ (x y : ℕ), y

theorem isosceles_triangle_base_length (x y : ℕ) (h1 : x + x + y = 18) (h2 : x + (x / 2) = 12)
  (h3 : (x / 2) + y = 6) (h4 : x + x > y) : length_of_base x y = 2 := by
  sorry

end isosceles_triangle_base_length_l22_22864


namespace hour_hand_rotation_6_to_9_l22_22844

theorem hour_hand_rotation_6_to_9 : ∀ (h₁ h₂ : ℝ), h₁ = 6 ∧ h₂ = 9 → (rotation h₁ h₂) = - (Real.pi / 2) :=
by
  sorry

end hour_hand_rotation_6_to_9_l22_22844


namespace largest_prime_factor_of_sum_of_divisors_180_l22_22978

def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in (list.range (n+1)).filter (λ d, n % d = 0), d

def largest_prime_factor (n : ℕ) : ℕ :=
  (list.range (n+1)).filter (λ p, nat.prime p ∧ n % p = 0).last'

theorem largest_prime_factor_of_sum_of_divisors_180 :
  largest_prime_factor (sum_of_divisors 180) = 13 := by sorry

end largest_prime_factor_of_sum_of_divisors_180_l22_22978


namespace triangle_area_l22_22320

theorem triangle_area (total_length total_width rect_length rect_width : ℝ)
  (h_total_size : total_length = 20 ∧ total_width = 6)
  (h_rect_size : rect_length = 15 ∧ rect_width = 6) :
  let total_area := total_length * total_width in
  let rect_area := rect_length * rect_width in
  let tri_area := total_area - rect_area in
  tri_area = 30 := by
  sorry

end triangle_area_l22_22320


namespace equivalence_prime_divisibility_l22_22492

theorem equivalence_prime_divisibility {p : ℕ} [hp : Fact (Nat.Prime p)] :
  (∃ x_0 : ℤ, p ∣ (x_0^2 - x_0 + 3)) ↔ (∃ y_0 : ℤ, p ∣ (y_0^2 - y_0 + 25)) :=
sorry

end equivalence_prime_divisibility_l22_22492


namespace eq_conj_iff_eq_inv_conj_iff_eq_neg_conj_iff_l22_22806

variable {a b : ℝ}

-- Part 1: Prove that a + bi = a - bi if and only if b = 0
theorem eq_conj_iff : a + b * complex.I = a - b * complex.I ↔ b = 0 := sorry

-- Part 2: Prove that a + bi = 1 / (a - bi) if and only if a^2 + b^2 = 1
theorem eq_inv_conj_iff : a + b * complex.I = 1 / (a - b * complex.I) ↔ a^2 + b^2 = 1 := sorry

-- Part 3: Prove that a + bi = -(a - bi) if and only if a = 0
theorem eq_neg_conj_iff : a + b * complex.I = -(a - b * complex.I) ↔ a = 0 := sorry

end eq_conj_iff_eq_inv_conj_iff_eq_neg_conj_iff_l22_22806


namespace secant_225_equals_neg_sqrt_two_l22_22446

/-- Definition of secant in terms of cosine -/
def sec (θ : ℝ) := 1 / Real.cos θ

/-- Given angles in radians for trig identities -/
def degree_to_radian (d : ℝ) := d * Real.pi / 180

/-- Main theorem statement -/
theorem secant_225_equals_neg_sqrt_two : sec (degree_to_radian 225) = -Real.sqrt 2 :=
by
  sorry

end secant_225_equals_neg_sqrt_two_l22_22446


namespace regular_octagon_interior_angle_l22_22187

theorem regular_octagon_interior_angle : 
  ∀ (n : ℕ), n = 8 → (sum_of_internal_angles n) / n = 135 := 
by
  intros n hn
  rw hn
  sorry

end regular_octagon_interior_angle_l22_22187


namespace gcd_qr_least_value_l22_22542

theorem gcd_qr_least_value (p q r : ℕ)
  (hpq : Nat.gcd p q = 210)
  (hpr : Nat.gcd p r = 1050) :
  ∃ m, m = 210 ∧ Nat.gcd q r = m :=
begin
  sorry
end

end gcd_qr_least_value_l22_22542


namespace largest_value_is_D_l22_22282

noncomputable def A := 13579 + 1/2468
noncomputable def B := 13579 - 1/2468
noncomputable def C := 13579 * (1/2468)
noncomputable def D := 13579 / (1/2468)
noncomputable def E := 13579.2468

theorem largest_value_is_D : D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  sorry

end largest_value_is_D_l22_22282


namespace regular_octagon_interior_angle_measure_l22_22255

theorem regular_octagon_interior_angle_measure :
  ∀ (n : ℕ), n = 8 → (∑ i in finset.range n, (180 : ℝ)) / n = 135 :=
  by
  sorry

end regular_octagon_interior_angle_measure_l22_22255


namespace ordering_of_four_numbers_l22_22851

variable (m n α β : ℝ)
variable (h1 : m < n)
variable (h2 : α < β)
variable (h3 : 2 * (α - m) * (α - n) - 7 = 0)
variable (h4 : 2 * (β - m) * (β - n) - 7 = 0)

theorem ordering_of_four_numbers : α < m ∧ m < n ∧ n < β :=
by
  sorry

end ordering_of_four_numbers_l22_22851


namespace regular_octagon_interior_angle_l22_22129

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : 
  let sum_interior_angles := 180 * (n - 2)
  let num_sides := 8
  let one_interior_angle := sum_interior_angles / num_sides
  one_interior_angle = 135 :=
by
  sorry

end regular_octagon_interior_angle_l22_22129


namespace midpoint_sum_eq_six_l22_22682

theorem midpoint_sum_eq_six :
  let x1 := 6
  let y1 := 12
  let x2 := 0
  let y2 := -6
  let midpoint_x := (x1 + x2) / 2 
  let midpoint_y := (y1 + y2) / 2 
  (midpoint_x + midpoint_y) = 6 :=
by
  let x1 := 6
  let y1 := 12
  let x2 := 0
  let y2 := -6
  let midpoint_x := (x1 + x2) / 2 
  let midpoint_y := (y1 + y2) / 2 
  sorry

end midpoint_sum_eq_six_l22_22682


namespace nested_radical_value_l22_22374

theorem nested_radical_value : 
  let rec nest k :=
    if k = 1 then 2019 else sqrt (1 + k * (k + 2)) in
  nest 2018 = 3 :=
by
  intros
  sorry

end nested_radical_value_l22_22374


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22171

-- Define the conditions for the problem
def num_sides : ℕ := 8
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Statement of the problem
theorem measure_of_one_interior_angle_of_regular_octagon : 
  (sum_of_interior_angles num_sides) / num_sides = 135 :=
by
  -- Conditions are defined above
  -- Proof steps are omitted
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22171


namespace no_matching_results_l22_22600

theorem no_matching_results : 
  ∀ (a b : ℕ), 1 ≤ a ∧ a < b ∧ b ≤ 15 → (a+1) * (b+1) ≠ 121 :=
by
  intros a b h
  cases h with ha resting
  cases resting with hab hb
  sorry

end no_matching_results_l22_22600


namespace train_cross_time_l22_22295

theorem train_cross_time
  (train_length : ℕ)
  (train_speed_kmh : ℕ)
  (conversion_factor : ℝ)
  (train_speed_ms : ℝ)
  (time_seconds : ℕ) :
  train_length = 160 →
  train_speed_kmh = 64 →
  conversion_factor = 1000 / 3600 →
  train_speed_ms = train_speed_kmh * conversion_factor →
  time_seconds = 160 / train_speed_ms →
  time_seconds = 9 :=
by
  intro h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  simp at h5
  exact h5

end train_cross_time_l22_22295


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22190

theorem measure_of_one_interior_angle_of_regular_octagon 
  (n : Nat) (h_n : n = 8) : 
  (one_interior_angle : ℝ) = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22190


namespace min_time_to_shoe_60_horses_l22_22293

theorem min_time_to_shoe_60_horses (
    blacksmiths : ℕ := 48,
    horses : ℕ := 60,
    shoe_time_per_hoof : ℕ := 5,
    hooves_per_horse : ℕ := 4,
    horses_shoed_at_once : ℕ := blacksmiths / hooves_per_horse
  ) :
  (minimum_time : ℕ) =
    let total_hooves := horses * hooves_per_horse in
    let total_time_single_blacksmith := total_hooves * shoe_time_per_hoof in
    let time_per_blacksmith := total_time_single_blacksmith / blacksmiths in
    let total_sets := horses / horses_shoed_at_once in
    let minimum_time := total_sets * time_per_blacksmith in
    minimum_time = 125 :=
  sorry

end min_time_to_shoe_60_horses_l22_22293


namespace parametric_curve_to_polar_and_tangent_length_l22_22505

open Real

noncomputable def parametric_to_polar (x y : ℝ) : Prop :=
  (∃ α : ℝ, x = 3 + 2 * cos α ∧ y = 2 * sin α) ↔ ρ^2 - 6 * ρ * cos θ + 5 = 0

noncomputable def tangent_line_length (x y : ℝ) : ℝ :=
  let d := abs (3 - y + x) / sqrt 2 in
  sqrt (d^2 - 4)

theorem parametric_curve_to_polar_and_tangent_length :
  (∀ x y ρ θ,
    parametric_to_polar x y ∧
    (sqrt 2) * ρ * sin (θ - π / 4) = 1) →
  (tangent_line_length x y = 2) :=
sorry

end parametric_curve_to_polar_and_tangent_length_l22_22505


namespace prob_log3_integer_l22_22735

theorem prob_log3_integer : 
  (∃ (N: ℕ), (100 ≤ N ∧ N ≤ 999) ∧ ∃ (k: ℕ), N = 3^k) → 
  (∃ (prob : ℚ), prob = 1 / 450) :=
sorry

end prob_log3_integer_l22_22735


namespace valid_pairs_for_area_18_l22_22618

theorem valid_pairs_for_area_18 (w l : ℕ) (hw : 0 < w) (hl : 0 < l) (h_area : w * l = 18) (h_lt : w < l) :
  (w, l) = (1, 18) ∨ (w, l) = (2, 9) ∨ (w, l) = (3, 6) :=
sorry

end valid_pairs_for_area_18_l22_22618


namespace measure_of_one_interior_angle_of_regular_octagon_l22_22165

-- Define the conditions for the problem
def num_sides : ℕ := 8
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Statement of the problem
theorem measure_of_one_interior_angle_of_regular_octagon : 
  (sum_of_interior_angles num_sides) / num_sides = 135 :=
by
  -- Conditions are defined above
  -- Proof steps are omitted
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l22_22165


namespace num_three_digit_strictly_ordered_l22_22474

theorem num_three_digit_strictly_ordered :
  ∃ n : ℕ, (∀ (d1 d2 d3 : ℕ), 
    d1 ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    d2 ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    d3 ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    (d1 < d2 < d3 ∨ d1 > d2 > d3) → 
    d1 ≠ 0 ∧ d2 ≠ 0 ∧ d3 ≠ 0 → 
    ∃ a b c : ℕ, (a, b, c) = (d1, d2, d3) ∧ 
    n = 204) :=
by sorry

end num_three_digit_strictly_ordered_l22_22474


namespace min_frac_seq_l22_22519

noncomputable def a_sequence : ℕ+ → ℤ
| 1       := 98
| 2       := 102
| (n + 1) + 1 := a_sequence (n + 1) + 4 * (n : ℕ)

def frac_seq_min_value : ℕ+ → ℚ := λ n, (a_sequence n : ℚ) / (n : ℚ)

theorem min_frac_seq : ∃ m, ∀ n, n ∈ ℕ+ → frac_seq_min_value n ≥ 26 := sorry

end min_frac_seq_l22_22519


namespace find_value_of_expression_l22_22508

-- Definitions based on the given conditions
def x : ℤ := 4
def y : ℤ := -3
def r : ℝ := real.sqrt ((x:ℝ)^2 + (y:ℝ)^2)

lemma sin_alpha : real.sin (real.arctan (y / x)) = y / r := sorry
lemma cos_alpha : real.cos (real.arctan (y / x)) = x / r := sorry

-- The main proof problem
theorem find_value_of_expression : 2 * real.sin (real.arctan (y / x)) + real.cos (real.arctan (y / x)) = -2/5 := sorry

end find_value_of_expression_l22_22508


namespace interior_angle_regular_octagon_l22_22086

theorem interior_angle_regular_octagon (n : ℕ) (h : n = 8) : 
  let sum_of_interior_angles := 180 * (n - 2)
  let number_of_sides := n
  let one_interior_angle := sum_of_interior_angles / number_of_sides
  in one_interior_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l22_22086


namespace total_balloons_correct_l22_22388

-- Define the number of balloons each person has
def dan_balloons : ℕ := 29
def tim_balloons : ℕ := 7 * dan_balloons
def molly_balloons : ℕ := 5 * dan_balloons

-- Define the total number of balloons
def total_balloons : ℕ := dan_balloons + tim_balloons + molly_balloons

-- The theorem to prove
theorem total_balloons_correct : total_balloons = 377 :=
by
  -- This part is where the proof will go
  sorry

end total_balloons_correct_l22_22388


namespace twenty_percent_greater_than_l22_22558

theorem twenty_percent_greater_than :
  ∃ x : ℝ, x = 40 + 0.20 * 40 ∧ x = 48 :=
by
  use 40 + 0.2 * 40
  split
  . refl
  . norm_num
  sorry

end twenty_percent_greater_than_l22_22558


namespace find_t_collinear_l22_22528

variables (t : ℝ)

-- Definitions for given vectors
def m : ℝ × ℝ := (real.sqrt 3, 1)
def n : ℝ × ℝ := (0, -1)
def k : ℝ × ℝ := (t, real.sqrt 3)

-- Vector subtraction and scalar multiplication
def vec_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
def scalar_mul (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)

-- Given condition that p = m - 2 * n
def p : ℝ × ℝ := vec_sub m (scalar_mul 2 n)

-- Cross product should be zero for collinear vectors
def cross_product (u v : ℝ × ℝ) : ℝ := u.1 * v.2 - u.2 * v.1

-- Theorem to be proven
theorem find_t_collinear (h : cross_product p k = 0) : t = 1 :=
by { sorry }

end find_t_collinear_l22_22528


namespace regular_octagon_angle_measure_l22_22211

theorem regular_octagon_angle_measure :
  let n := 8 in
  (180 * (n - 2)) / n = 135 := 
by
  sorry

end regular_octagon_angle_measure_l22_22211


namespace interior_angle_regular_octagon_l22_22088

theorem interior_angle_regular_octagon (n : ℕ) (h : n = 8) : 
  let sum_of_interior_angles := 180 * (n - 2)
  let number_of_sides := n
  let one_interior_angle := sum_of_interior_angles / number_of_sides
  in one_interior_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l22_22088


namespace max_buses_in_city_l22_22937

-- Definition of the problem statement and the maximal number of buses
theorem max_buses_in_city (bus_stops : Finset ℕ) (buses : Finset (Finset ℕ)) 
  (h_stops : bus_stops.card = 9) 
  (h_buses_stops : ∀ b ∈ buses, b.card = 3) 
  (h_shared_stops : ∀ b₁ b₂ ∈ buses, b₁ ≠ b₂ → (b₁ ∩ b₂).card ≤ 1) : 
  buses.card ≤ 12 := 
sorry

end max_buses_in_city_l22_22937


namespace relay_race_arrangements_l22_22663

theorem relay_race_arrangements :
  let students := ["Sun Ming", "Zhu Liang", "Tang Qiang", "Sha Qigang"]
  let positions := [1, 2, 3]
  let arrangements := positions.perm.length
  arrangements = 6 :=
by
  sorry

end relay_race_arrangements_l22_22663


namespace correct_statements_l22_22688

-- Define the statements as predicates
def statement1 : Prop := ∀ (A B C : Point), ¬ (is_collinear A B C) → ∃ (O : Point) (r : ℝ), circle O r A B C
def statement2 : Prop := ∀ (O : Point) (r : ℝ) (A B : Point), is_diameter O r → is_chord_perpendicular_bisector O A B
def statement3 : Prop := ∀ (O : Point) (r : ℝ) (A B C D : Point), subtends_equal_central_angles O A B C D → equal_chords O A B C D
def statement4 : Prop := ∀ (ABC : Triangle) (O : Point), is_circumcenter O ABC → is_equidistant_from_vertices O ABC

-- Main theorem to prove
theorem correct_statements : (statement2 ∧ statement4) ∧ ¬statement1 ∧ ¬statement3 :=
by
  sorry

end correct_statements_l22_22688


namespace exp_ln_simplification_l22_22061

theorem exp_ln_simplification (x : ℝ) (h : x > 0) : 
  exp (2 * log 2) = 4 := 
  by
  sorry

end exp_ln_simplification_l22_22061


namespace regular_octagon_interior_angle_l22_22177

theorem regular_octagon_interior_angle : 
  ∀ (n : ℕ), n = 8 → (sum_of_internal_angles n) / n = 135 := 
by
  intros n hn
  rw hn
  sorry

end regular_octagon_interior_angle_l22_22177


namespace interior_angle_of_regular_octagon_l22_22118

theorem interior_angle_of_regular_octagon :
  let n := 8 in
  let sum_interior_angles := 180 * (n - 2) in
  let angle := sum_interior_angles / n in
  angle = 135 :=
by
  sorry

end interior_angle_of_regular_octagon_l22_22118


namespace regular_octagon_angle_measure_l22_22215

theorem regular_octagon_angle_measure :
  let n := 8 in
  (180 * (n - 2)) / n = 135 := 
by
  sorry

end regular_octagon_angle_measure_l22_22215


namespace election_majority_l22_22567

theorem election_majority
  (total_votes : ℕ)
  (winning_percent : ℝ)
  (other_percent : ℝ)
  (votes_cast : total_votes = 700)
  (winning_share : winning_percent = 0.84)
  (other_share : other_percent = 0.16) :
  ∃ majority : ℕ, majority = 476 := by
  sorry

end election_majority_l22_22567


namespace tetrahedron_dihedral_angle_l22_22667

noncomputable def dihedral_angle_tetrahedron : ℝ :=
  arccos ((Real.sqrt 5 - 1) / 2)

theorem tetrahedron_dihedral_angle 
  (A B C D : ℝ × ℝ × ℝ)
  (h₁ : angle A B C = π / 2)
  (h₂ : angle A B D = π / 2)
  (h₃ : angle A C D = π / 2)
  (h₄ : angle B C D = angle B D C)
  (h₅ : angle B C D = angle C D B) : 
  angle B C D = arccos ((Real.sqrt 5 - 1) / 2) :=
sorry

end tetrahedron_dihedral_angle_l22_22667


namespace problem1_problem2_problem3_l22_22526

-- Definitions of vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-3, -4)

-- Problems
/-- Problem 1: Prove that 2a + 3b = (-5, -10) --/
theorem problem1 : 2 • a + 3 • b = (-5, -10) := 
  sorry

/-- Problem 2: Prove that the magnitude of (a - 2b) is sqrt(145) --/
theorem problem2 : ∥a - 2 • b∥ = real.sqrt 145 := 
  sorry

-- Definition of vector c
def c (x y : ℝ) : Prop := (∥(x, y)∥ = 1) ∧ ((5, 5).fst * x + (5, 5).snd * y = 0)

/-- Problem 3: Prove that vector c has coordinates (sqrt(2)/2, -sqrt(2)/2) or 
               (-sqrt(2)/2, sqrt(2)/2) if c is a unit vector perpendicular to (a - b) --/
theorem problem3 (x y : ℝ) (h : c x y) : 
  (x = real.sqrt 2 / 2 ∧ y = -real.sqrt 2 / 2) ∨ 
  (x = -real.sqrt 2 / 2 ∧ y = real.sqrt 2 / 2) := 
  sorry

end problem1_problem2_problem3_l22_22526


namespace sec_225_eq_neg_sqrt2_l22_22443

theorem sec_225_eq_neg_sqrt2 :
  sec 225 = -sqrt 2 :=
by 
  -- Definitions for conditions
  have h1 : ∀ θ : ℕ, sec θ = 1 / cos θ := sorry,
  have h2 : cos 225 = cos (180 + 45) := sorry,
  have h3 : cos (180 + 45) = -cos 45 := sorry,
  have h4 : cos 45 = 1 / sqrt 2 := sorry,
  -- Required proof statement
  sorry

end sec_225_eq_neg_sqrt2_l22_22443


namespace area_bounded_by_curves_l22_22702

noncomputable def areaOfBoundedRegion : real := 
  (1 / 2) * ∫ φ in 0..(π / 2), (sin φ)^2 + 
  (1 / 2) * ∫ φ in (π / 2)..(3 * π / 4), (√2 * cos (φ - π / 4))^2

theorem area_bounded_by_curves : areaOfBoundedRegion = π / 4 := by
  sorry

end area_bounded_by_curves_l22_22702


namespace equilateral_triangle_l22_22897

-- Define the vectors and the conditions of collinearity
variables {a b c A B C : ℝ}

def vector_m := (a, Real.cos (A / 2))
def vector_n := (b, Real.cos (B / 2))
def vector_p := (c, Real.cos (C / 2))

def collinear (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

theorem equilateral_triangle
  (h₁ : collinear vector_m vector_n)
  (h₂ : collinear vector_n vector_p)
  (h₃ : collinear vector_m vector_p)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hA : 0 < A / 2 ∧ A / 2 < π / 2)
  (hB : 0 < B / 2 ∧ B / 2 < π / 2)
  (hC : 0 < C / 2 ∧ C / 2 < π / 2)
: A = B ∧ B = C := sorry

end equilateral_triangle_l22_22897


namespace regular_octagon_interior_angle_l22_22121

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : 
  let sum_interior_angles := 180 * (n - 2)
  let num_sides := 8
  let one_interior_angle := sum_interior_angles / num_sides
  one_interior_angle = 135 :=
by
  sorry

end regular_octagon_interior_angle_l22_22121


namespace problem1_problem2_problem3_problem4_problem5_problem6_l22_22759

-- First problem: \(\frac{1}{3} + \left(-\frac{1}{2}\right) = -\frac{1}{6}\)
theorem problem1 : (1 / 3 : ℚ) + (-1 / 2) = -1 / 6 := by sorry

-- Second problem: \(-2 - \left(-9\right) = 7\)
theorem problem2 : (-2 : ℚ) - (-9) = 7 := by sorry

-- Third problem: \(\frac{15}{16} - \left(-7\frac{1}{16}\right) = 8\)
theorem problem3 : (15 / 16 : ℚ) - (-(7 + 1 / 16)) = 8 := by sorry

-- Fourth problem: \(-\left|-4\frac{2}{7}\right| - \left|+1\frac{5}{7}\right| = -6\)
theorem problem4 : -|(-4 - 2 / 7 : ℚ)| - |(1 + 5 / 7)| = -6 := by sorry

-- Fifth problem: \(6 + \left(-12\right) + 8.3 + \left(-7.5\right) = -5.2\)
theorem problem5 : (6 : ℚ) + (-12) + (83 / 10) + (-75 / 10) = -52 / 10 := by sorry

-- Sixth problem: \(\left(-\frac{1}{8}\right) + 3.25 + 2\frac{3}{5} + \left(-5.875\right) + 1.15 = 1\)
theorem problem6 : (-1 / 8 : ℚ) + 3 + 1 / 4 + 2 + 3 / 5 + (-5 - 875 / 1000) + 1 + 15 / 100 = 1 := by sorry

end problem1_problem2_problem3_problem4_problem5_problem6_l22_22759


namespace regular_octagon_interior_angle_l22_22098

theorem regular_octagon_interior_angle (n : ℕ) (h₁ : n = 8) :
  (180 * (n - 2)) / n = 135 :=
by
  rw [h₁]
  sorry

end regular_octagon_interior_angle_l22_22098


namespace inequality_example_l22_22909

theorem inequality_example (a b : ℝ) (h : a - b > 0) : a + 1 > b + 1 :=
sorry

end inequality_example_l22_22909


namespace geom_sequence_a_n_l22_22571

variable {a : ℕ → ℝ}

-- Given conditions
def is_geom_seq (a : ℕ → ℝ) : Prop :=
  |a 1| = 1 ∧ a 5 = -8 * a 2 ∧ a 5 > a 2

-- Statement to prove
theorem geom_sequence_a_n (h : is_geom_seq a) : ∀ n, a n = (-2) ^ (n - 1) := 
by
  sorry

end geom_sequence_a_n_l22_22571


namespace assignment_correct_l22_22685

-- Definitions of the conditions as hypotheses.
variables (A M B x y : ℕ)

-- The statement to prove that B: \(M = -M\) is the correct assignment.
theorem assignment_correct : M = -M :=
sorry

end assignment_correct_l22_22685


namespace largest_prime_factor_of_sum_of_divisors_of_180_eq_13_l22_22983

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, d ∣ n).sum

def largest_prime_factor (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ p, Nat.Prime p ∧ p ∣ n).max' sorry

theorem largest_prime_factor_of_sum_of_divisors_of_180_eq_13 :
  largest_prime_factor (sum_of_divisors 180) = 13 :=
by
  sorry

end largest_prime_factor_of_sum_of_divisors_of_180_eq_13_l22_22983


namespace floor_sq_minus_sq_floor_l22_22423

noncomputable def floor_13_2 : ℤ := int.floor 13.2

theorem floor_sq_minus_sq_floor :
  int.floor ((13.2 : ℝ) ^ 2) - (floor_13_2 * floor_13_2) = 5 :=
by
  let floor_13_2_sq := floor_13_2 * floor_13_2
  have h1 : int.floor (13.2 : ℝ) = 13 := by norm_num
  have h2 : int.floor ((13.2 : ℝ) ^ 2) = 174 := by norm_num
  rw [h1] at floor_13_2
  rw [h2]
  rw [floor_13_2, floor_13_2, floor_13_2]
  exact sorry

end floor_sq_minus_sq_floor_l22_22423


namespace exists_quadratic_polynomial_l22_22967

theorem exists_quadratic_polynomial
  (a b c : ℕ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  : ∃ f : ℤ → ℤ, (∃ (p q r : ℤ), f = λ x, p * x^2 + q * x + r ∧ p > 0) ∧ 
  (f a = ↑(a^3) ∧ f b = ↑(b^3) ∧ f c = ↑(c^3)) :=
sorry

end exists_quadratic_polynomial_l22_22967


namespace sec_225_eq_neg_sqrt_2_l22_22455

theorem sec_225_eq_neg_sqrt_2 :
  let sec (x : ℝ) := 1 / Real.cos x
  ∧ Real.cos (225 * Real.pi / 180) = Real.cos (180 * Real.pi / 180 + 45 * Real.pi / 180)
  ∧ Real.cos (180 * Real.pi / 180 + 45 * Real.pi / 180) = -Real.cos (45 * Real.pi / 180)
  ∧ Real.cos (45 * Real.pi / 180) = 1 / Real.sqrt 2
  → sec (225 * Real.pi / 180) = -Real.sqrt 2 :=
by 
  intros sec_h cos_h1 cos_h2 cos_h3
  sorry

end sec_225_eq_neg_sqrt_2_l22_22455


namespace increasing_function_range_l22_22877

theorem increasing_function_range (a : ℝ) (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f x = x^3 - a * x - 1) :
  (∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) ↔ a ≤ 0 :=
sorry

end increasing_function_range_l22_22877


namespace consecutive_sets_sum_to_20_l22_22537

theorem consecutive_sets_sum_to_20 : 
  (count (λ (n a : ℕ), n ≥ 2 ∧ a ≥ 2 ∧ (n * (2 * a + n - 1)) = 40) = 1) :=
by
  sorry

end consecutive_sets_sum_to_20_l22_22537
