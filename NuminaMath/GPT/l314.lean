import Mathlib

namespace average_is_correct_l314_314268

def numbers : List ℕ := [12, 13, 14, 510, 520, 530, 1115, 1120, 1, 1252140, 2345]

def sum_of_numbers : ℕ := numbers.foldr (· + ·) 0

def number_of_values : ℕ := numbers.length

def average : ℚ := sum_of_numbers / number_of_values

theorem average_is_correct : average = 114391.82 := by
  sorry

end average_is_correct_l314_314268


namespace inequality_for_all_real_l314_314240

theorem inequality_for_all_real (a b c : ℝ) : 
  a^6 + b^6 + c^6 - 3 * a^2 * b^2 * c^2 ≥ 1/2 * (a - b)^2 * (b - c)^2 * (c - a)^2 :=
by 
  sorry

end inequality_for_all_real_l314_314240


namespace workers_contribution_l314_314077

theorem workers_contribution (N C : ℕ) 
(h1 : N * C = 300000) 
(h2 : N * (C + 50) = 360000) : 
N = 1200 :=
sorry

end workers_contribution_l314_314077


namespace distance_from_center_to_line_of_tangent_circle_l314_314183

theorem distance_from_center_to_line_of_tangent_circle 
  (a : ℝ) (ha : 0 < a) 
  (h_circle : (2 - a)^2 + (1 - a)^2 = a^2)
  (h_tangent : ∀ x y : ℝ, x = 0 ∨ y = 0): 
  (|2 * a - a - 3| / ((2:ℝ)^2 + (-1)^2).sqrt) = (2 * (5:ℝ).sqrt) / 5 :=
by
  sorry

end distance_from_center_to_line_of_tangent_circle_l314_314183


namespace james_passenger_count_l314_314385

theorem james_passenger_count :
  ∀ (total_vehicles trucks buses taxis motorbikes cars trucks_population buses_population taxis_population motorbikes_population cars_population : ℕ),
  total_vehicles = 52 →
  trucks = 12 →
  buses = 2 →
  taxis = 2 * buses →
  motorbikes = total_vehicles - (trucks + buses + taxis + cars) →
  cars = 30 →
  trucks_population = 2 →
  buses_population = 15 →
  taxis_population = 2 →
  motorbikes_population = 1 →
  cars_population = 3 →
  (trucks * trucks_population + buses * buses_population + taxis * taxis_population +
   motorbikes * motorbikes_population + cars * cars_population) = 156 := 
by
  -- Placeholder for the proof
  sorry

end james_passenger_count_l314_314385


namespace milburg_population_l314_314038

-- Define the number of grown-ups and children in Milburg
def grown_ups : ℕ := 5256
def children : ℕ := 2987

-- The total population is defined as the sum of grown-ups and children
def total_population : ℕ := grown_ups + children

-- Goal: Prove that the total population in Milburg is 8243
theorem milburg_population : total_population = 8243 := 
by {
  -- the proof should be here, but we use sorry to skip it
  sorry
}

end milburg_population_l314_314038


namespace gcd_values_count_l314_314073

theorem gcd_values_count (a b : ℕ) (h : a * b = 360) : 
  ∃ n, n = (Set.toFinset (Set.image2 Nat.gcd {d | ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * y = 360 ∧ x = d ∨ y = d} {d | ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * y = 360 ∧ x = d ∨ y = d})).card ∧ n = 12 :=
by
  sorry

end gcd_values_count_l314_314073


namespace blue_face_area_factor_l314_314438

def cube_side_length : ℕ := 13

def original_red_face_area : ℕ := 6 * cube_side_length ^ 2
def total_mini_cube_face_area : ℕ := 6 * cube_side_length ^ 3
def blue_face_area : ℕ := total_mini_cube_face_area - original_red_face_area

theorem blue_face_area_factor :
  (blue_face_area / original_red_face_area) = 12 :=
by
  sorry

end blue_face_area_factor_l314_314438


namespace sum_ratio_l314_314388

def arithmetic_sequence (a_1 d : ℚ) (n : ℕ) : ℚ :=
  a_1 + (n - 1) * d

def S_n (a_1 d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a_1 + (n - 1) * d) / 2 -- sum of first n terms of arithmetic sequence

theorem sum_ratio (a_1 d : ℚ) (h : 13 * (a_1 + 6 * d) = 7 * (a_1 + 3 * d)) :
  S_n a_1 d 13 / S_n a_1 d 7 = 1 :=
by
  -- Proof omitted
  sorry

end sum_ratio_l314_314388


namespace circle_distance_condition_l314_314181

theorem circle_distance_condition (a : ℝ) (h1 : (2 - a)^2 + (1 - a)^2 = a^2) (h2 : ∀ (x y : ℝ), ∃ c : ℝ, y = 2*x - c) :
    ∃ (center : ℝ × ℝ), ((center.1 = a) ∧ (center.2 = a) ∧ (center = (1, 1) ∨ center = (5, 5))) 
    → (∀ (line : ℝ × ℝ × ℝ), line = (2, -1, -3) → abs ((line.1 * a + line.2 * a + line.3) / real.sqrt (line.1 ^ 2 + line.2 ^ 2)) = (2 * real.sqrt 5 / 5)) :=
by
  intro a h1 h2 center h3 line h4
  sorry

end circle_distance_condition_l314_314181


namespace inequality_reciprocal_l314_314021

theorem inequality_reciprocal (a b : Real) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) : (1 / a < 1 / b) :=
by
  sorry

end inequality_reciprocal_l314_314021


namespace greg_age_is_16_l314_314136

-- Define the ages of Cindy, Jan, Marcia, and Greg based on the conditions
def cindy_age : ℕ := 5
def jan_age : ℕ := cindy_age + 2
def marcia_age : ℕ := 2 * jan_age
def greg_age : ℕ := marcia_age + 2

-- Theorem statement: Prove that Greg's age is 16
theorem greg_age_is_16 : greg_age = 16 :=
by
  -- Proof would go here
  sorry

end greg_age_is_16_l314_314136


namespace ratio_of_octagon_areas_l314_314102

-- Define the relevant terms and conditions
noncomputable def radius_of_circle (r : ℝ) := r
noncomputable def side_length_inscribed_octagon (r : ℝ) := r * Real.sqrt 2
noncomputable def side_length_circumscribed_octagon (r : ℝ) := r * Real.sqrt 2

-- Area of a regular octagon in terms of its side length
noncomputable def area_regular_octagon (s : ℝ) := 2 * (1 + Real.sqrt 2) * s^2

-- Areas of inscribed and circumscribed octagons
noncomputable def area_inscribed_octagon (r : ℝ) := area_regular_octagon (side_length_inscribed_octagon r)
noncomputable def area_circumscribed_octagon (r : ℝ) := area_regular_octagon (side_length_circumscribed_octagon r)

-- The main statement to prove
theorem ratio_of_octagon_areas (r : ℝ) : 
  area_circumscribed_octagon r / area_inscribed_octagon r = 1 :=
by
  sorry

end ratio_of_octagon_areas_l314_314102


namespace blue_faces_ratio_l314_314414

-- Define the cube dimensions and derived properties
def cube_dimension : ℕ := 13
def original_cube_surface_area : ℕ := 6 * (cube_dimension ^ 2)
def total_number_of_mini_cubes : ℕ := cube_dimension ^ 3
def total_mini_cube_faces : ℕ := 6 * total_number_of_mini_cubes
def red_faces : ℕ := original_cube_surface_area
def blue_faces : ℕ := total_mini_cube_faces - red_faces
def blue_to_red_ratio : ℕ := blue_faces / red_faces

-- The statement to be proved
theorem blue_faces_ratio :
  blue_to_red_ratio = 12 := by
  sorry

end blue_faces_ratio_l314_314414


namespace andrew_total_hours_l314_314324

theorem andrew_total_hours (days_worked : ℕ) (hours_per_day : ℝ)
    (h1 : days_worked = 3) (h2 : hours_per_day = 2.5) : 
    days_worked * hours_per_day = 7.5 := by
  sorry

end andrew_total_hours_l314_314324


namespace total_sand_arrived_l314_314260

theorem total_sand_arrived :
  let truck1_carry := 4.1
  let truck1_loss := 2.4
  let truck2_carry := 5.7
  let truck2_loss := 3.6
  let truck3_carry := 8.2
  let truck3_loss := 1.9
  (truck1_carry - truck1_loss) + 
  (truck2_carry - truck2_loss) + 
  (truck3_carry - truck3_loss) = 10.1 :=
by
  sorry

end total_sand_arrived_l314_314260


namespace find_speed_of_current_l314_314308

variable {m c : ℝ}

theorem find_speed_of_current
  (h1 : m + c = 15)
  (h2 : m - c = 10) :
  c = 2.5 :=
sorry

end find_speed_of_current_l314_314308


namespace circle_tangent_distance_l314_314185

theorem circle_tangent_distance 
    (a : ℝ) (h_center : ∃ a : ℝ, a > 0 ∧ (circle_center = (a, a)) ∧ ((2 - a)^2 + (1 - a)^2 = a^2)) 
    (h_line : ∀ x y, distance_to_line (circle_center x y) (2 * x - y - 3) = 2 / sqrt (2^2 + 1^2)) :
    distance_to_line (circle_center 1 1) (2 * 1 - 1 - 3) = 2 * sqrt 5 / 5 ∧ distance_to_line (circle_center 5 5) (2 * 5 - 5 - 3) = 2 * sqrt 5 / 5 :=
by
    sorry

end circle_tangent_distance_l314_314185


namespace other_root_is_five_l314_314152

theorem other_root_is_five (m : ℝ) 
  (h : -1 is_root_m x^2 - 4 * x + m = 0) : 
  is_root x^2 - 4 * x + m = 0 5 := 
sorry

end other_root_is_five_l314_314152


namespace not_every_constant_is_geometric_l314_314253

def is_constant_sequence (s : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, s n = s m

def is_geometric_sequence (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

theorem not_every_constant_is_geometric :
  (¬ ∀ s : ℕ → ℝ, is_constant_sequence s → is_geometric_sequence s) ↔
  ∃ s : ℕ → ℝ, is_constant_sequence s ∧ ¬ is_geometric_sequence s := 
by
  sorry

end not_every_constant_is_geometric_l314_314253


namespace ratio_area_octagons_correct_l314_314106

noncomputable def ratio_area_octagons (r : ℝ) : ℝ :=
  let s := 2 * r * Real.sin (Real.pi / 8)   -- side length of inscribed octagon
  let S := 2 * r * Real.cos (Real.pi / 8)   -- side length of circumscribed octagon
  let ratio_side_lengths := S / s
  let ratio_areas := (ratio_side_lengths)^2
  ratio_areas

theorem ratio_area_octagons_correct (r : ℝ) :
  ratio_area_octagons r = 6 + 4 * Real.sqrt 2 :=
by
  sorry

end ratio_area_octagons_correct_l314_314106


namespace area_ratio_is_correct_l314_314111

noncomputable def areaRatioInscribedCircumscribedOctagon (r : ℝ) : ℝ :=
  let s1 := r * Real.sin (67.5 * Real.pi / 180)
  let s2 := r * Real.sin (67.5 * Real.pi / 180) / Real.cos (67.5 * Real.pi / 180)
  (s2 / s1) ^ 2

theorem area_ratio_is_correct (r : ℝ) : areaRatioInscribedCircumscribedOctagon r = 2 + Real.sqrt 2 :=
by
  sorry

end area_ratio_is_correct_l314_314111


namespace circle_distance_to_line_l314_314196

-- Define the conditions
def circle_pass_through (P : ℝ × ℝ) (a : ℝ) : Prop :=
  let (x, y) := P
  (x - a)^2 + (y - a)^2 = a^2

def is_tangent_to_axes (a : ℝ) : Prop :=
  a > 0

def distance_to_line (center : ℝ × ℝ) (L : ℝ × ℝ → ℝ) : ℝ :=
  let (x₁, y₁) := center
  abs (L (x₁, y₁)) / sqrt ((L (1,0))^2 + (L (0,1))^2)

-- Define the line equation as a function
def line_eq (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  2*x - y - 3

-- Define the main proof goal
theorem circle_distance_to_line (a : ℝ) (h1 : circle_pass_through (2, 1) a) (h2 : is_tangent_to_axes a) :
  distance_to_line (a, a) line_eq = 2*sqrt 5 / 5 :=
sorry

end circle_distance_to_line_l314_314196


namespace probability_12th_roll_correct_l314_314000

noncomputable def probability_12th_roll_is_last : ℚ :=
  (7 / 8) ^ 10 * (1 / 8)

theorem probability_12th_roll_correct :
  probability_12th_roll_is_last = 282475249 / 8589934592 :=
by
  sorry

end probability_12th_roll_correct_l314_314000


namespace christine_sales_value_l314_314335

variable {X : ℝ}

def commission_rate : ℝ := 0.12
def personal_needs_percent : ℝ := 0.60
def savings_amount : ℝ := 1152
def savings_percent : ℝ := 0.40

theorem christine_sales_value:
  (savings_percent * (commission_rate * X) = savings_amount) → 
  (X = 24000) := 
by
  intro h
  sorry

end christine_sales_value_l314_314335


namespace units_digit_17_pow_28_l314_314467

theorem units_digit_17_pow_28 : (17 ^ 28) % 10 = 1 :=
by
  sorry

end units_digit_17_pow_28_l314_314467


namespace find_B_l314_314363

variable (A B : Set ℤ)
variable (U : Set ℤ := {x | 0 ≤ x ∧ x ≤ 6})

theorem find_B (hU : U = {x | 0 ≤ x ∧ x ≤ 6})
               (hA_complement_B : A ∩ (U \ B) = {1, 3, 5}) :
  B = {0, 2, 4, 6} :=
sorry

end find_B_l314_314363


namespace Patrick_fish_count_l314_314329

variable (Angus Patrick Ollie : ℕ)

-- Conditions
axiom h1 : Ollie + 7 = Angus
axiom h2 : Angus = Patrick + 4
axiom h3 : Ollie = 5

-- Theorem statement
theorem Patrick_fish_count : Patrick = 8 := 
by
  sorry

end Patrick_fish_count_l314_314329


namespace candy_distribution_l314_314343

-- Definitions based on conditions
def distribution_problem (r b g w : Nat) : Prop :=
  (1 ≤ r) ∧ (1 ≤ b) ∧ (1 ≤ g) ∧ (r + b + g + w = 8)

-- Main theorem statement
theorem candy_distribution : 
  ∑ r in (Finset.range 7 \ {0}), ∑ b in (Finset.range (7-r) \ {0}), ∑ g in (Finset.range (7-r-b) \ {0}),
    Nat.choose 8 r * Nat.choose (8 - r) b * Nat.choose (8 - r - b) g * 2^(8 - (r + b + g)) = 1600 :=
by
  sorry

end candy_distribution_l314_314343


namespace find_x_when_areas_equal_l314_314013

-- Definitions based on the problem conditions
def glass_area : ℕ := 4 * (30 * 20)
def window_area (x : ℕ) : ℕ := (60 + 3 * x) * (40 + 3 * x)
def total_area_of_glass : ℕ := glass_area
def total_area_of_wood (x : ℕ) : ℕ := window_area x - glass_area

-- Proof problem, proving x == 20 / 3 when total area of glass equals total area of wood
theorem find_x_when_areas_equal : 
  ∃ x : ℕ, (total_area_of_glass = total_area_of_wood x) ∧ x = 20 / 3 :=
sorry

end find_x_when_areas_equal_l314_314013


namespace examination_duration_in_hours_l314_314011

theorem examination_duration_in_hours 
  (total_questions : ℕ)
  (type_A_questions : ℕ)
  (time_for_A_problems : ℝ) 
  (time_ratio_A_to_B : ℝ)
  (total_time_for_A : ℝ) 
  (total_time : ℝ) :
  total_questions = 200 → 
  type_A_questions = 15 → 
  time_ratio_A_to_B = 2 → 
  total_time_for_A = 25.116279069767444 →
  total_time = (total_time_for_A + 185 * (25.116279069767444 / 15 / 2)) → 
  total_time / 60 = 3 :=
by sorry

end examination_duration_in_hours_l314_314011


namespace rob_travel_time_to_park_l314_314405

theorem rob_travel_time_to_park : 
  ∃ R : ℝ, 
    (∀ Tm : ℝ, Tm = 3 * R) ∧ -- Mark's travel time is three times Rob's travel time
    (∀ Tr : ℝ, Tm - 2 = R) → -- Considering Mark's head start of 2 hours
    R = 1 :=
sorry

end rob_travel_time_to_park_l314_314405


namespace sqrt_two_irrational_l314_314025

theorem sqrt_two_irrational :
  ¬ ∃ (p q : ℕ), p ≠ 0 ∧ q ≠ 0 ∧ gcd p q = 1 ∧ (↑q / ↑p) ^ 2 = (2:ℝ) :=
sorry

end sqrt_two_irrational_l314_314025


namespace ratio_of_areas_of_octagons_l314_314122

theorem ratio_of_areas_of_octagons
  (r : ℝ)
  (sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2)
  (sin_67_5 : Real.sin (Real.pi / 180 * 67.5) = Real.sqrt (2 + Real.sqrt 2) / 2)
  : let smaller_octagon_area := 2 * (1 + Real.sqrt 2) * r^2 in
    let larger_octagon_area := 
      2 * (1 + Real.sqrt 2) * (2 * r * (1 + Real.sqrt 2))^2 in
    larger_octagon_area / smaller_octagon_area = 12 + 8 * Real.sqrt 2 :=
by
  sorry

end ratio_of_areas_of_octagons_l314_314122


namespace hyperbola_with_foci_on_y_axis_l314_314280

variable (α : ℝ) {x y : ℝ}

theorem hyperbola_with_foci_on_y_axis (h : α ∈ (real.pi / 2, 3 * real.pi / 4)) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  (x^2*sin α - y^2*cos α = 1) → (x^2 / a^2 - y^2 / b^2 = 1) ∧ 
  (∀ (a b : ℝ), a > b) → 
  (y - axis contains foci) := 
sorry

end hyperbola_with_foci_on_y_axis_l314_314280


namespace equation_of_line_l_l314_314034

theorem equation_of_line_l
  (a : ℝ)
  (l_intersects_circle : ∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y + a = 0)
  (midpoint_chord : ∃ C : ℝ × ℝ, C = (-2, 3) ∧ ∃ A B : ℝ × ℝ, A ≠ B ∧ (A.1 + B.1) / 2 = C.1 ∧ (A.2 + B.2) / 2 = C.2) :
  a < 3 →
  ∃ l : ℝ × ℝ → Prop, (∀ x y : ℝ, l (x, y) ↔ x - y + 5 = 0) :=
by {
  sorry
}

end equation_of_line_l_l314_314034


namespace number_of_good_numbers_l314_314224

def floor_div (x : ℝ) (n : ℕ) : ℤ := ⌊ x / n! ⌋

def f (x : ℝ) : ℤ :=
  ∑ k in Finset.range 2013, floor_div x (k + 1)

def is_good_number (n : ℤ) : Prop :=
  ∃ x : ℝ, f x = n

def good_numbers : Finset ℤ :=
  (Finset.range 1007).map ⟨λ k, 2 * k + 1, sorry⟩ -- Every odd number in the set.

theorem number_of_good_numbers : good_numbers.filter is_good_number = 587 := sorry

end number_of_good_numbers_l314_314224


namespace age_of_15th_student_l314_314247

theorem age_of_15th_student 
  (average_age_15 : ℕ → ℕ → ℕ)
  (average_age_5 : ℕ → ℕ → ℕ)
  (average_age_9 : ℕ → ℕ → ℕ)
  (h1 : average_age_15 15 15 = 15)
  (h2 : average_age_5 5 14 = 14)
  (h3 : average_age_9 9 16 = 16) :
  let total_age_15 := 15 * 15 in
  let total_age_5 := 5 * 14 in
  let total_age_9 := 9 * 16 in
  let combined_total_age := total_age_5 + total_age_9 in
  let age_15th_student := total_age_15 - combined_total_age in
  age_15th_student = 11 := 
by
  simp [total_age_15, total_age_5, total_age_9, combined_total_age, age_15th_student]
  exact eq.refl 11

end age_of_15th_student_l314_314247


namespace train_length_correct_l314_314473

noncomputable def train_speed_kmph : ℝ := 60
noncomputable def train_time_seconds : ℝ := 15

noncomputable def length_of_train : ℝ :=
  let speed_mps := train_speed_kmph * 1000 / 3600
  speed_mps * train_time_seconds

theorem train_length_correct :
  length_of_train = 250.05 :=
by
  -- Proof goes here
  sorry

end train_length_correct_l314_314473


namespace remaining_movie_duration_l314_314319

/--
Given:
1. The laptop was fully charged at 3:20 pm.
2. Hannah started watching a 3-hour series.
3. The laptop turned off at 5:44 pm (fully discharged).

Prove:
The remaining duration of the movie Hannah needs to watch is 36 minutes.
-/
theorem remaining_movie_duration
    (start_full_charge : ℕ := 200)  -- representing 3:20 pm as 200 (20 minutes past 3:00)
    (end_discharge : ℕ := 344)  -- representing 5:44 pm as 344 (44 minutes past 5:00)
    (total_duration_minutes : ℕ := 180)  -- 3 hours in minutes
    (start_time_minutes : ℕ := 200)  -- convert 3:20 pm to minutes past noon
    (end_time_minutes : ℕ := 344)  -- convert 5:44 pm to minutes past noon
    : (total_duration_minutes - (end_time_minutes - start_time_minutes)) = 36 :=
by
  sorry

end remaining_movie_duration_l314_314319


namespace sin_theta_plus_pi_over_4_l314_314166

noncomputable def sin_sum_angle : Real :=
  let x := -3
  let y := 4
  let r := Real.sqrt (x*x + y*y)
  let sinθ := y / r
  let cosθ := x / r
  Real.sin (θ + π / 4)

theorem sin_theta_plus_pi_over_4 :
  sin_sum_angle (-3) (4) = Real.sqrt 2 / 10 :=
by
  sorry

end sin_theta_plus_pi_over_4_l314_314166


namespace base10_equivalent_of_43210_7_l314_314274

def base7ToDecimal (num : Nat) : Nat :=
  let digits := [4, 3, 2, 1, 0]
  digits[0] * 7^4 + digits[1] * 7^3 + digits[2] * 7^2 + digits[3] * 7^1 + digits[4] * 7^0

theorem base10_equivalent_of_43210_7 :
  base7ToDecimal 43210 = 10738 :=
by
  sorry

end base10_equivalent_of_43210_7_l314_314274


namespace solution_is_unique_l314_314170

def conditions (x y : ℝ) : Prop :=
  (x/y + y/x) * (x + y) = 15 ∧
  (x^2/y^2 + y^2/x^2) * (x^2 + y^2) = 85

theorem solution_is_unique : ∀ x y : ℝ, conditions x y → (x = 2 ∧ y = 4) ∨ (x = 4 ∧ y = 2) :=
by
  intro x y
  assume h : conditions x y
  sorry

end solution_is_unique_l314_314170


namespace incorrect_statement_C_l314_314287

theorem incorrect_statement_C :
  (∀ r : ℚ, ∃ p : ℝ, p = r) ∧  -- Condition A: All rational numbers can be represented by points on the number line.
  (∀ x : ℝ, x = 1 / x → x = 1 ∨ x = -1) ∧  -- Condition B: The reciprocal of a number equal to itself is ±1.
  (∀ f : ℚ, ∃ q : ℝ, q = f) →  -- Condition C (negation of C as presented): Fractions cannot be represented by points on the number line.
  (∀ x : ℝ, abs x ≥ 0) ∧ (∀ x : ℝ, abs x = 0 ↔ x = 0) →  -- Condition D: The number with the smallest absolute value is 0.
  false :=                      -- Prove that statement C is incorrect
by
  sorry

end incorrect_statement_C_l314_314287


namespace given_fraction_l314_314387

variable (initial_cards : ℕ)
variable (cards_given_to_friend : ℕ)
variable (fraction_given_to_brother : ℚ)

noncomputable def fraction_given (initial_cards cards_given_to_friend : ℕ) (fraction_given_to_brother : ℚ) : Prop :=
  let cards_left := initial_cards / 2
  initial_cards - cards_left - cards_given_to_friend = fraction_given_to_brother * initial_cards

theorem given_fraction
  (h_initial : initial_cards = 16)
  (h_given_to_friend : cards_given_to_friend = 2)
  (h_fraction : fraction_given_to_brother = 3 / 8) :
  fraction_given initial_cards cards_given_to_friend fraction_given_to_brother :=
by
  sorry

end given_fraction_l314_314387


namespace area_of_quadrilateral_l314_314130

theorem area_of_quadrilateral 
  (area_ΔBDF : ℝ) (area_ΔBFE : ℝ) (area_ΔEFC : ℝ) (area_ΔCDF : ℝ) (h₁ : area_ΔBDF = 5)
  (h₂ : area_ΔBFE = 10) (h₃ : area_ΔEFC = 10) (h₄ : area_ΔCDF = 15) :
  (80 - (area_ΔBDF + area_ΔBFE + area_ΔEFC + area_ΔCDF)) = 40 := 
  by sorry

end area_of_quadrilateral_l314_314130


namespace absolute_value_equality_l314_314410

variables {a b c d : ℝ}

theorem absolute_value_equality (h1 : |a - b| + |c - d| = 99) (h2 : |a - c| + |b - d| = 1) : |a - d| + |b - c| = 99 :=
sorry

end absolute_value_equality_l314_314410


namespace distance_from_center_of_circle_to_line_l314_314193

open Real

def circle_passes_through (a : ℝ) :=
  (2 - a) ^ 2 + (1 - a) ^ 2 = a ^ 2

def circle_tangent_to_axes (a : ℝ) :=
  a > 0

def distance_from_center_to_line (a : ℝ) : ℝ :=
  abs (2 * a - a - 3) / sqrt (2 ^ 2 + (-1) ^ 2)

theorem distance_from_center_of_circle_to_line
  (a : ℝ) (h1 : circle_passes_through a) (h2 : circle_tangent_to_axes a) :
  distance_from_center_to_line a = 2 * sqrt 5 / 5 :=
sorry

end distance_from_center_of_circle_to_line_l314_314193


namespace probability_of_winning_noughts_l314_314315

def ticTacToeProbability : ℚ :=
  let totalWays := Nat.choose 9 3
  let winningWays := 8
  winningWays / totalWays

theorem probability_of_winning_noughts :
  ticTacToeProbability = 2 / 21 := by
  sorry

end probability_of_winning_noughts_l314_314315


namespace exponent_equality_l314_314177

theorem exponent_equality (y : ℕ) (z : ℕ) (h1 : 16 ^ y = 4 ^ z) (h2 : y = 8) : z = 16 := by
  sorry

end exponent_equality_l314_314177


namespace gcd_values_count_l314_314064

theorem gcd_values_count (a b : ℕ) (h : a * b = 360) : 
  ∃ g : ℕ, g ∈ {1, 2, 3, 4, 5, 6, 8, 9, 12, 18}.card := sorry

end gcd_values_count_l314_314064


namespace find_dividend_l314_314300

theorem find_dividend (q : ℕ) (d : ℕ) (r : ℕ) (D : ℕ) 
  (h_q : q = 15000)
  (h_d : d = 82675)
  (h_r : r = 57801)
  (h_D : D = 1240182801) :
  D = d * q + r := by 
  sorry

end find_dividend_l314_314300


namespace graph_two_intersecting_lines_l314_314076

theorem graph_two_intersecting_lines (x y : ℝ) : (x + y)^2 = x^2 + y^2 + 3 * x * y ↔ x = 0 ∨ y = 0 :=
by
  -- Placeholder for the proof
  sorry

end graph_two_intersecting_lines_l314_314076


namespace blue_red_area_ratio_l314_314418

theorem blue_red_area_ratio (n : ℕ) (h : n = 13) :
    let total_red_faces := 6 * n^2,
        total_faces := 6 * n^3,
        blue_faces := total_faces - total_red_faces
    in (blue_faces / total_red_faces : ℕ) = 12 :=
  by
  sorry

end blue_red_area_ratio_l314_314418


namespace smallest_n_l314_314049

theorem smallest_n (n : ℕ) : 634 * n ≡ 1275 * n [MOD 30] ↔ n = 30 :=
by
  sorry

end smallest_n_l314_314049


namespace gcd_possible_values_count_l314_314061

theorem gcd_possible_values_count : ∃ a b : ℕ, a * b = 360 ∧ (∃ gcds : Finset ℕ, gcds = {d | ∃ a b : ℕ, a * b = 360 ∧ d = Nat.gcd a b} ∧ gcds.card = 6) :=
sorry

end gcd_possible_values_count_l314_314061


namespace ab_value_l314_314001

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := by
  sorry

end ab_value_l314_314001


namespace gcd_lcm_product_360_l314_314054

theorem gcd_lcm_product_360 :
  ∃ (d : ℕ), (∃ x y : ℕ, x * y = 360 ∧ d = Nat.gcd x y) ∧ 
  {d | ∃ x y : ℕ, x * y = 360 ∧ d = Nat.gcd x y}.to_finset.card = 8 := 
sorry

end gcd_lcm_product_360_l314_314054


namespace gcd_values_count_l314_314066

theorem gcd_values_count (a b : ℕ) (h : a * b = 360) : 
  ∃ g : ℕ, g ∈ {1, 2, 3, 4, 5, 6, 8, 9, 12, 18}.card := sorry

end gcd_values_count_l314_314066


namespace apple_difference_l314_314141

def carla_apples : ℕ := 7
def tim_apples : ℕ := 1

theorem apple_difference : carla_apples - tim_apples = 6 := by
  sorry

end apple_difference_l314_314141


namespace gcd_possible_values_count_l314_314067

theorem gcd_possible_values_count (a b : ℕ) (h_ab : a * b = 360) : 
  (∃ d, d = Nat.gcd a b ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 12)) ∧ 
  (∃ n, n = 6) := 
by
  sorry

end gcd_possible_values_count_l314_314067


namespace num_five_dollar_coins_l314_314045

theorem num_five_dollar_coins (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 5 * y = 125) : y = 15 :=
by
  sorry -- Proof to be completed

end num_five_dollar_coins_l314_314045


namespace James_total_passengers_l314_314383

def trucks := 12
def buses := 2
def taxis := 2 * buses
def cars := 30
def motorbikes := 52 - trucks - buses - taxis - cars

def people_in_truck := 2
def people_in_bus := 15
def people_in_taxi := 2
def people_in_motorbike := 1
def people_in_car := 3

def total_passengers := 
  trucks * people_in_truck + 
  buses * people_in_bus + 
  taxis * people_in_taxi + 
  motorbikes * people_in_motorbike + 
  cars * people_in_car

theorem James_total_passengers : total_passengers = 156 := 
by 
  -- Placeholder proof, needs to be completed
  sorry

end James_total_passengers_l314_314383


namespace product_of_x_y_l314_314078

theorem product_of_x_y (x y : ℝ) (h1 : 3 * x + 4 * y = 60) (h2 : 6 * x - 4 * y = 12) : x * y = 72 :=
by
  sorry

end product_of_x_y_l314_314078


namespace gcd_lcm_product_360_l314_314053

theorem gcd_lcm_product_360 :
  ∃ (d : ℕ), (∃ x y : ℕ, x * y = 360 ∧ d = Nat.gcd x y) ∧ 
  {d | ∃ x y : ℕ, x * y = 360 ∧ d = Nat.gcd x y}.to_finset.card = 8 := 
sorry

end gcd_lcm_product_360_l314_314053


namespace find_area_of_oblique_triangle_l314_314162

noncomputable def area_triangle (a b c : ℝ) (A B C : ℝ) : ℝ :=
  1 / 2 * a * b * Real.sin C

theorem find_area_of_oblique_triangle
  (A B C a b c : ℝ)
  (h1 : c = Real.sqrt 21)
  (h2 : c * Real.sin A = Real.sqrt 3 * a * Real.cos C)
  (h3 : Real.sin C + Real.sin (B - A) = 5 * Real.sin (2 * A))
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum_ABC : A + B + C = Real.pi)
  (habc_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (tri_angle_pos : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi) :
  area_triangle a b c A B C = 5 * Real.sqrt 3 / 4 := 
sorry

end find_area_of_oblique_triangle_l314_314162


namespace trig_identity_l314_314161

variable {α : ℝ}

theorem trig_identity (h : Real.sin α = 2 * Real.cos α) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 / 3 := by
  sorry

end trig_identity_l314_314161


namespace people_sharing_pizzas_l314_314048

-- Definitions based on conditions
def number_of_pizzas : ℝ := 21.0
def slices_per_person : ℝ := 2.625
def slices_per_pizza : ℝ := 8.0

-- Theorem to prove the number of people
theorem people_sharing_pizzas : (number_of_pizzas * slices_per_pizza) / slices_per_person = 64 :=
by
  sorry

end people_sharing_pizzas_l314_314048


namespace milburg_population_l314_314037

-- Define the number of grown-ups and children in Milburg
def grown_ups : ℕ := 5256
def children : ℕ := 2987

-- The total population is defined as the sum of grown-ups and children
def total_population : ℕ := grown_ups + children

-- Goal: Prove that the total population in Milburg is 8243
theorem milburg_population : total_population = 8243 := 
by {
  -- the proof should be here, but we use sorry to skip it
  sorry
}

end milburg_population_l314_314037


namespace possible_length_of_third_side_l314_314372

theorem possible_length_of_third_side (a b c : ℤ) (h1 : a - b = 7) (h2 : (a + b + c) % 2 = 1) : c = 8 :=
sorry

end possible_length_of_third_side_l314_314372


namespace fib_mod_150_eq_8_l314_314411

def fib_mod_9 (n : ℕ) : ℕ :=
  (Nat.fib n) % 9

theorem fib_mod_150_eq_8 : fib_mod_9 150 = 8 :=
  sorry

end fib_mod_150_eq_8_l314_314411


namespace slope_of_perpendicular_line_l314_314134

theorem slope_of_perpendicular_line (x y : ℝ) (h : 5 * x - 4 * y = 20) : 
  ∃ m : ℝ, m = -4 / 5 :=
sorry

end slope_of_perpendicular_line_l314_314134


namespace value_of_a_plus_b_l314_314175

theorem value_of_a_plus_b (a b : ℝ) (h1 : sqrt 44 = 2 * sqrt a) (h2 : sqrt 54 = 3 * sqrt b) : a + b = 17 := 
sorry

end value_of_a_plus_b_l314_314175


namespace solution_set_xf_pos_l314_314356

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

theorem solution_set_xf_pos (f : ℝ → ℝ) (h₁ : odd_function f) (h₂ : f 2 = 0)
  (h₃ : ∀ x : ℝ, 0 < x → x * (deriv f x) + f x > 0) :
  { x | x * f x > 0 } = { x | x < -2 } ∪ { x | x > 2 } := by
  sorry

end solution_set_xf_pos_l314_314356


namespace area_ratio_of_octagons_is_4_l314_314099

-- Define the given conditions
variable (r : ℝ) -- radius of the common circle

def side_length_circumscribed_octagon := √2 * r
def side_length_inscribed_octagon := r / √2

-- Areas of polygons scale with the square of the side length
def area_ratio_circumscribed_to_inscribed := (side_length_circumscribed_octagon r / side_length_inscribed_octagon r)^2

theorem area_ratio_of_octagons_is_4 :
  area_ratio_circumscribed_to_inscribed r = 4 := by
  sorry

end area_ratio_of_octagons_is_4_l314_314099


namespace blue_face_area_factor_l314_314441

def cube_side_length : ℕ := 13

def original_red_face_area : ℕ := 6 * cube_side_length ^ 2
def total_mini_cube_face_area : ℕ := 6 * cube_side_length ^ 3
def blue_face_area : ℕ := total_mini_cube_face_area - original_red_face_area

theorem blue_face_area_factor :
  (blue_face_area / original_red_face_area) = 12 :=
by
  sorry

end blue_face_area_factor_l314_314441


namespace base7_to_base10_of_43210_l314_314271

theorem base7_to_base10_of_43210 : 
  base7_to_base10 (list.num_from_digits [4, 3, 2, 1, 0]) 7 = 10738 :=
by
  def base7_to_base10 (digits : list ℕ) (base : ℕ) : ℕ :=
    digits.reverse.join_with base
  
  show base7_to_base10 [4, 3, 2, 1, 0] 7 = 10738
  sorry

end base7_to_base10_of_43210_l314_314271


namespace chocolates_not_in_box_initially_l314_314230

theorem chocolates_not_in_box_initially 
  (total_chocolates : ℕ) 
  (chocolates_friend_brought : ℕ) 
  (initial_boxes : ℕ) 
  (additional_boxes : ℕ)
  (total_after_friend : ℕ)
  (chocolates_each_box : ℕ)
  (total_chocolates_initial : ℕ) :
  total_chocolates = 50 ∧ initial_boxes = 3 ∧ chocolates_friend_brought = 25 ∧ total_after_friend = 75 
  ∧ additional_boxes = 2 ∧ chocolates_each_box = 15 ∧ total_chocolates_initial = 50
  → (total_chocolates_initial - (initial_boxes * chocolates_each_box)) = 5 :=
by
  sorry

end chocolates_not_in_box_initially_l314_314230


namespace sequence_periodic_a2014_l314_314015

theorem sequence_periodic_a2014 (a : ℕ → ℚ) 
  (h1 : a 1 = -1/4) 
  (h2 : ∀ n > 1, a n = 1 - (1 / (a (n - 1)))) : 
  a 2014 = -1/4 :=
sorry

end sequence_periodic_a2014_l314_314015


namespace units_digit_of_17_pow_28_l314_314465

theorem units_digit_of_17_pow_28 : ((17:ℕ) ^ 28) % 10 = 1 := by
  sorry

end units_digit_of_17_pow_28_l314_314465


namespace find_x_value_l314_314142

theorem find_x_value (x : ℤ)
    (h1 : (5 + 9) / 2 = 7)
    (h2 : (5 + x) / 2 = 10)
    (h3 : (x + 9) / 2 = 12) : 
    x = 15 := 
sorry

end find_x_value_l314_314142


namespace quadratic_inequality_solution_set_l314_314035

theorem quadratic_inequality_solution_set :
  {x : ℝ | 2 * x^2 - x - 3 ≥ 0} = {x : ℝ | x ≤ -1 ∨ x ≥ 3 / 2} :=
sorry

end quadratic_inequality_solution_set_l314_314035


namespace distance_between_points_l314_314276

def point := (ℝ × ℝ)

noncomputable def distance (p1 p2 : point) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem distance_between_points :
  distance (2, -2) (8, 8) = real.sqrt 136 :=
by
  sorry

end distance_between_points_l314_314276


namespace survey_response_total_l314_314023

theorem survey_response_total
  (X Y Z : ℕ)
  (h_ratio : X / 4 = Y / 2 ∧ X / 4 = Z)
  (h_X : X = 200) :
  X + Y + Z = 350 :=
sorry

end survey_response_total_l314_314023


namespace octagon_area_ratio_l314_314089

theorem octagon_area_ratio (r : ℝ) :
  let A_inscribed := 2 * (1 + Real.sqrt 2) * (r * Real.tan (Real.pi / 8))^2 in
  let R := r * (1 / Real.cos (Real.pi / 8)) in
  let A_circumscribed := 2 * (1 + Real.sqrt 2) * (2 * R * Real.sin (Real.pi / 8))^2 in
  A_circumscribed / A_inscribed = 4 * (3 + 2 * Real.sqrt 2) :=
by
  sorry

end octagon_area_ratio_l314_314089


namespace mean_tasks_b_l314_314043

variable (a b : ℕ)
variable (m_a m_b : ℕ)
variable (h1 : a + b = 260)
variable (h2 : a = 3 * b / 10 + b)
variable (h3 : m_a = 80)
variable (h4 : m_b = 12 * m_a / 10)

theorem mean_tasks_b :
  m_b = 96 := by
  -- This is where the proof would go
  sorry

end mean_tasks_b_l314_314043


namespace angle_in_third_quadrant_l314_314325

theorem angle_in_third_quadrant (α : ℝ) (h : α = 2023) : 180 < α % 360 ∧ α % 360 < 270 := by
  sorry

end angle_in_third_quadrant_l314_314325


namespace find_f_2023_l314_314159

theorem find_f_2023 :
  ∃ f : ℕ → ℕ, (∀ m n : ℕ, f(n + f(m)) = f(n) + m + 1) ∧ (∀ k l : ℕ, k < l → f(k) < f(l)) ∧ f 2023 = 2024 :=
by
  sorry

end find_f_2023_l314_314159


namespace number_of_hens_l314_314084

theorem number_of_hens (H C G : ℕ) 
  (h1 : H + C + G = 120) 
  (h2 : 2 * H + 4 * C + 4 * G = 348) : 
  H = 66 := 
by 
  sorry

end number_of_hens_l314_314084


namespace toy_store_problem_l314_314128

variables (x y : ℕ)

theorem toy_store_problem (h1 : 8 * x + 26 * y + 33 * (31 - x - y) / 2 = 370)
                          (h2 : x + y + (31 - x - y) / 2 = 31) :
    x = 20 :=
sorry

end toy_store_problem_l314_314128


namespace circle_distance_condition_l314_314180

theorem circle_distance_condition (a : ℝ) (h1 : (2 - a)^2 + (1 - a)^2 = a^2) (h2 : ∀ (x y : ℝ), ∃ c : ℝ, y = 2*x - c) :
    ∃ (center : ℝ × ℝ), ((center.1 = a) ∧ (center.2 = a) ∧ (center = (1, 1) ∨ center = (5, 5))) 
    → (∀ (line : ℝ × ℝ × ℝ), line = (2, -1, -3) → abs ((line.1 * a + line.2 * a + line.3) / real.sqrt (line.1 ^ 2 + line.2 ^ 2)) = (2 * real.sqrt 5 / 5)) :=
by
  intro a h1 h2 center h3 line h4
  sorry

end circle_distance_condition_l314_314180


namespace neg_prop_p_equiv_l314_314451

open Classical

variable (x : ℝ)
def prop_p : Prop := ∀ x : ℝ, x^2 + 1 ≥ 0

theorem neg_prop_p_equiv : ¬ prop_p ↔ ∃ x : ℝ, x^2 + 1 < 0 := by
  sorry

end neg_prop_p_equiv_l314_314451


namespace total_amount_l314_314304

theorem total_amount (N50 N: ℕ) (h1: N = 90) (h2: N50 = 77) : 
  (N50 * 50 + (N - N50) * 500) = 10350 :=
by
  sorry

end total_amount_l314_314304


namespace range_a_sufficient_not_necessary_l314_314349

theorem range_a_sufficient_not_necessary (a : ℝ) (h : 0 < a) :
  (∀ x : ℝ, (x ^ 2 - 4 * a * x + 3 * a ^ 2 < 0) → (|x - 3| > 1)) →
  a ≥ 4 ∨ (0 < a ∧ a ≤ 2 / 3) :=
sorry

end range_a_sufficient_not_necessary_l314_314349


namespace avg_of_last_three_l314_314031

-- Define the conditions given in the problem
def avg_5 : Nat := 54
def avg_2 : Nat := 48
def num_list_length : Nat := 5
def first_two_length : Nat := 2

-- State the theorem
theorem avg_of_last_three
    (h_avg5 : 5 * avg_5 = 270)
    (h_avg2 : 2 * avg_2 = 96) :
  (270 - 96) / 3 = 58 :=
sorry

end avg_of_last_three_l314_314031


namespace determine_m_l314_314391

-- Define f and g according to the given conditions
def f (x : ℝ) (m : ℝ) : ℝ := x^2 - 3 * x + m
def g (x : ℝ) (m : ℝ) : ℝ := x^2 - 3 * x + 5 * m

-- Define the value of x
def x := 5

-- State the main theorem we need to prove
theorem determine_m 
  (h : 3 * f x m = 2 * g x m) : m = 10 / 7 :=
by
  -- Proof is omitted
  sorry

end determine_m_l314_314391


namespace bathtub_fill_time_l314_314471

-- Defining the conditions as given in the problem
def fill_rate (t_fill : ℕ) : ℚ := 1 / t_fill
def drain_rate (t_drain : ℕ) : ℚ := 1 / t_drain

-- Given specific values for the problem
def t_fill : ℕ := 10
def t_drain : ℕ := 12

-- Net fill rate calculation
def net_fill_rate (t_fill t_drain : ℕ) : ℚ :=
  fill_rate t_fill - drain_rate t_drain

-- Time to fill the bathtub given net fill rate
def time_to_fill (net_rate : ℚ) : ℚ :=
  1 / net_rate

-- The proof statement:
theorem bathtub_fill_time : time_to_fill (net_fill_rate t_fill t_drain) = 60 := by
  sorry

end bathtub_fill_time_l314_314471


namespace minimum_d_exists_l314_314212

open Nat

theorem minimum_d_exists :
  ∃ (a b c d e f g h i k : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ k ∧
                                b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ k ∧
                                c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ k ∧
                                d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ k ∧
                                e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ k ∧
                                f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ k ∧
                                g ≠ h ∧ g ≠ i ∧ g ≠ k ∧
                                h ≠ i ∧ h ≠ k ∧
                                i ≠ k ∧
                                d = a + 3 * (e + h) + k ∧
                                d = 20 :=
by
  sorry

end minimum_d_exists_l314_314212


namespace joan_dimes_l314_314017

theorem joan_dimes (initial_dimes spent_dimes remaining_dimes : ℕ) 
    (h1 : initial_dimes = 5) (h2 : spent_dimes = 2) 
    (h3 : remaining_dimes = initial_dimes - spent_dimes) : 
    remaining_dimes = 3 := 
sorry

end joan_dimes_l314_314017


namespace simplify_fraction_l314_314292

theorem simplify_fraction (x y : ℕ) : (x + y)^3 / (x + y) = (x + y)^2 := by
  sorry

end simplify_fraction_l314_314292


namespace cuboid_edge_length_l314_314413

theorem cuboid_edge_length
  (x : ℝ)
  (h_surface_area : 2 * (4 * x + 24 + 6 * x) = 148) :
  x = 5 :=
by
  sorry

end cuboid_edge_length_l314_314413


namespace michael_twice_jacob_l314_314382

variable {J M Y : ℕ}

theorem michael_twice_jacob :
  (J + 4 = 13) → (M = J + 12) → (M + Y = 2 * (J + Y)) → (Y = 3) := by
  sorry

end michael_twice_jacob_l314_314382


namespace find_number_l314_314082

theorem find_number (x : ℝ) (h : 0.6667 * x + 0.75 = 1.6667) : x = 1.375 :=
sorry

end find_number_l314_314082


namespace gcd_values_count_l314_314065

theorem gcd_values_count (a b : ℕ) (h : a * b = 360) : 
  ∃ g : ℕ, g ∈ {1, 2, 3, 4, 5, 6, 8, 9, 12, 18}.card := sorry

end gcd_values_count_l314_314065


namespace find_f_2023_l314_314158

def is_strictly_increasing (f : ℕ → ℕ) : Prop :=
  ∀ {a b : ℕ}, a < b → f a < f b

theorem find_f_2023 (f : ℕ → ℕ)
  (h_inc : is_strictly_increasing f)
  (h_relation : ∀ m n : ℕ, f (n + f m) = f n + m + 1) :
  f 2023 = 2024 :=
sorry

end find_f_2023_l314_314158


namespace correct_calculation_l314_314284

theorem correct_calculation (x y : ℝ) : 
  ¬(2 * x^2 + 3 * x^2 = 6 * x^2) ∧ 
  ¬(x^4 * x^2 = x^8) ∧ 
  ¬(x^6 / x^2 = x^3) ∧ 
  ((x * y^2)^2 = x^2 * y^4) :=
by
  sorry

end correct_calculation_l314_314284


namespace correct_calculation_l314_314286

theorem correct_calculation (x y : ℝ) : (x * y^2) ^ 2 = x^2 * y^4 :=
by
  sorry

end correct_calculation_l314_314286


namespace red_blue_area_ratio_is_12_l314_314431

def red_blue_face_area_ratio : ℕ :=
  let big_cube_dim : ℕ := 13 in
  let total_red_faces : ℕ := 6 * big_cube_dim^2 in
  let total_mini_cube_faces : ℕ := 6 * big_cube_dim^3 in
  let total_blue_faces : ℕ := total_mini_cube_faces - total_red_faces in
  total_blue_faces / total_red_faces

theorem red_blue_area_ratio_is_12 :
  red_blue_face_area_ratio = 12 := by
  sorry

end red_blue_area_ratio_is_12_l314_314431


namespace James_total_passengers_l314_314384

def trucks := 12
def buses := 2
def taxis := 2 * buses
def cars := 30
def motorbikes := 52 - trucks - buses - taxis - cars

def people_in_truck := 2
def people_in_bus := 15
def people_in_taxi := 2
def people_in_motorbike := 1
def people_in_car := 3

def total_passengers := 
  trucks * people_in_truck + 
  buses * people_in_bus + 
  taxis * people_in_taxi + 
  motorbikes * people_in_motorbike + 
  cars * people_in_car

theorem James_total_passengers : total_passengers = 156 := 
by 
  -- Placeholder proof, needs to be completed
  sorry

end James_total_passengers_l314_314384


namespace find_M_value_when_x_3_l314_314226

-- Definitions based on the given conditions
def polynomial (a b c d x : ℝ) : ℝ := a*x^5 + b*x^3 + c*x + d

-- Given conditions
variables (a b c d : ℝ)
axiom h₀ : polynomial a b c d 0 = -5
axiom h₁ : polynomial a b c d (-3) = 7

-- Desired statement: Prove that the value of polynomial at x = 3 is -17
theorem find_M_value_when_x_3 : polynomial a b c d 3 = -17 :=
by sorry

end find_M_value_when_x_3_l314_314226


namespace y_work_duration_l314_314079

theorem y_work_duration (x_rate y_rate : ℝ) (d : ℝ) :
  -- 1. x and y together can do the work in 20 days.
  (x_rate + y_rate = 1/20) →
  -- 2. x started the work alone and after 4 days y joined him till the work completed.
  -- 3. The total work lasted 10 days.
  (4 * x_rate + 6 * (x_rate + y_rate) = 1) →
  -- Prove: y can do the work alone in 12 days.
  y_rate = 1/12 :=
by {
  sorry
}

end y_work_duration_l314_314079


namespace value_of_a_plus_b_l314_314174

theorem value_of_a_plus_b (a b : ℕ) (h1 : Real.sqrt 44 = 2 * Real.sqrt a) (h2 : Real.sqrt 54 = 3 * Real.sqrt b) : a + b = 17 := 
sorry

end value_of_a_plus_b_l314_314174


namespace jan_drives_more_miles_than_ian_l314_314289

-- Definitions of conditions
variables (s t d m: ℝ)

-- Ian's travel equation
def ian_distance := d = s * t

-- Han's travel equation
def han_distance := (d + 115) = (s + 8) * (t + 2)

-- Jan's travel equation
def jan_distance := m = (s + 12) * (t + 3)

-- The proof statement we want to prove
theorem jan_drives_more_miles_than_ian :
    (∀ (s t d m : ℝ),
    d = s * t →
    (d + 115) = (s + 8) * (t + 2) →
    m = (s + 12) * (t + 3) →
    (m - d) = 184.5) :=
    sorry

end jan_drives_more_miles_than_ian_l314_314289


namespace custom_op_example_l314_314206

def custom_op (a b : ℤ) : ℤ := a + 2 * b^2

theorem custom_op_example : custom_op (-4) 6 = 68 :=
by
  sorry

end custom_op_example_l314_314206


namespace excircle_problem_l314_314020

open EuclideanGeometry

variables {A B C : Point ℝ} -- Points of the triangle
variables {O_a O_b O_c I : Point ℝ} -- Centers and incenter
variables {AB AC : Line ℝ}
variable {R : ℝ} -- Radius of the circumcircle
variables (P_1 : Point ℝ) -- Intersection point

-- Let us define the required conditions
def is_excircle_center (O : Point ℝ) (A B C : Point ℝ) : Prop := -- O is the center of the excircle of the triangle
  ∃ (r: ℝ), ∀ P: Point ℝ, distance O P = r

def is_incenter (I A B C: Point ℝ) : Prop := -- I is the actual incenter of the triangle
  ∃ rI: ℝ, ∀ P, on_incircle I A B C P = distance I P = rI

def is_circumradius (A B C O : Point ℝ) (R : ℝ) : Prop := -- R is the radius of the circumcircle
  ∃ O', circumscribed_circle A B C O' R

-- Define the problem statement in Lean
theorem excircle_problem (h1 : is_excircle_center O_a A B C) 
  (h2 : is_excircle_center O_b B C A)
  (h3 : is_excircle_center O_c C A B)
  (h4 : is_incenter I A B C)
  (h5 : is_circumradius A B C O_a R)
  (h6 : is_perpendicular P_1 O_b O_a AB)
  (h7 : is_perpendicular P_1 O_c O_a AC):
  distance P_1 I = 2 * R :=
begin
  sorry -- Proof to be implemented here
end

end excircle_problem_l314_314020


namespace distance_from_center_of_circle_to_line_l314_314191

open Real

def circle_passes_through (a : ℝ) :=
  (2 - a) ^ 2 + (1 - a) ^ 2 = a ^ 2

def circle_tangent_to_axes (a : ℝ) :=
  a > 0

def distance_from_center_to_line (a : ℝ) : ℝ :=
  abs (2 * a - a - 3) / sqrt (2 ^ 2 + (-1) ^ 2)

theorem distance_from_center_of_circle_to_line
  (a : ℝ) (h1 : circle_passes_through a) (h2 : circle_tangent_to_axes a) :
  distance_from_center_to_line a = 2 * sqrt 5 / 5 :=
sorry

end distance_from_center_of_circle_to_line_l314_314191


namespace range_of_a_l314_314361

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 0) ↔ (0 ≤ a ∧ a ≤ Real.exp 1) := by
  sorry

end range_of_a_l314_314361


namespace domain_ln_x_squared_minus_2_l314_314338

theorem domain_ln_x_squared_minus_2 (x : ℝ) : 
  x^2 - 2 > 0 ↔ (x < -Real.sqrt 2 ∨ x > Real.sqrt 2) := 
by 
  sorry

end domain_ln_x_squared_minus_2_l314_314338


namespace problem_solution_l314_314477

open Finset

def S : Finset ℝ := {-10, -6, -5, -4, -2.5, -1, 0, 2.5, 4, 6, 7, 10}

def is_solution (x : ℝ) : Prop := (x - 5 = 0) ∨ (x + 10 = 0) ∨ (2 * x - 5 = 0)

def probability_solution_in_set (S : Finset ℝ) : ℚ :=
  (S.filter is_solution).card / S.card

theorem problem_solution :
  probability_solution_in_set S = 1 / 6 :=
by
  sorry

end problem_solution_l314_314477


namespace find_b_l314_314448

theorem find_b (a b : ℤ) (h1 : 0 ≤ a) (h2 : a < 2^2008) (h3 : 0 ≤ b) (h4 : b < 8) (h5 : 7 * (a + 2^2008 * b) % 2^2011 = 1) :
  b = 3 :=
sorry

end find_b_l314_314448


namespace remainder_of_addition_and_division_l314_314279

theorem remainder_of_addition_and_division :
  (3452179 + 50) % 7 = 4 :=
by
  sorry

end remainder_of_addition_and_division_l314_314279


namespace max_parts_three_planes_divide_space_l314_314171

-- Define the conditions given in the problem.
-- Condition 1: A plane divides the space into two parts.
def plane_divides_space (n : ℕ) : ℕ := 2

-- Condition 2: Two planes can divide the space into either three or four parts.
def two_planes_divide_space (n : ℕ) : ℕ := if n = 2 then 3 else 4

-- Condition 3: Three planes can divide the space into four, six, seven, or eight parts.
def three_planes_divide_space (n : ℕ) : ℕ := if n = 4 then 8 else sorry

-- The statement to be proved.
theorem max_parts_three_planes_divide_space : 
  ∃ n, three_planes_divide_space n = 8 := by
  use 4
  sorry

end max_parts_three_planes_divide_space_l314_314171


namespace quotient_of_division_l314_314250

theorem quotient_of_division
  (larger smaller : ℕ)
  (h1 : larger - smaller = 1370)
  (h2 : larger = 1626)
  (h3 : ∃ q r, larger = smaller * q + r ∧ r = 15) :
  ∃ q, larger = smaller * q + 15 ∧ q = 6 :=
by
  sorry

end quotient_of_division_l314_314250


namespace square_area_l314_314028

theorem square_area (A : ℝ) (s : ℝ) (prob_not_in_B : ℝ)
  (h1 : s * 4 = 32)
  (h2 : prob_not_in_B = 0.20987654320987653)
  (h3 : A - s^2 = prob_not_in_B * A) :
  A = 81 :=
by
  sorry

end square_area_l314_314028


namespace stuffed_animal_cost_l314_314227

variable (S : ℝ)  -- Cost of the stuffed animal
variable (total_cost_after_discount_gave_30_dollars : S * 0.10 = 3.6) 
-- Condition: cost of stuffed animal = $4.44
theorem stuffed_animal_cost :
  S = 4.44 :=
by
  sorry

end stuffed_animal_cost_l314_314227


namespace greg_age_is_16_l314_314137

-- Definitions based on given conditions
def cindy_age : ℕ := 5
def jan_age : ℕ := cindy_age + 2
def marcia_age : ℕ := 2 * jan_age
def greg_age : ℕ := marcia_age + 2

-- Theorem stating that Greg's age is 16 years given the above conditions
theorem greg_age_is_16 : greg_age = 16 := by
  sorry

end greg_age_is_16_l314_314137


namespace distance_to_line_is_constant_l314_314201

noncomputable def center_of_circle (a : ℝ) : (ℝ × ℝ) := (a, a)

noncomputable def circle_equation (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - a)^2 = a^2

noncomputable def distance_from_center_to_line (x1 y1 : ℝ) : ℝ :=
  abs (2 * x1 - y1 - 3) / sqrt (2^2 + (-1)^2)

theorem distance_to_line_is_constant (a : ℝ) (h : circle_equation a 2 1) (hx : a = 5 ∨ a = 1) :
  distance_from_center_to_line a a = 2 * sqrt 5 / 5 :=
by
  sorry

end distance_to_line_is_constant_l314_314201


namespace more_red_than_white_red_not_less_than_white_l314_314044

open Nat

theorem more_red_than_white :
  let red := 4
  let white := 6
  (choose red 4) + (choose red 3) * (choose white 1) = 25 := 
by
  intros
  rw [choose_self, choose_succ_self, mul_comm]
  exact rfl

theorem red_not_less_than_white :
  let red := 4
  let white := 6
  (choose red 4) + (choose red 3) * (choose white 1) + (choose red 2) * (choose white 2) = 115 := 
by
  intros
  rw [choose_self, choose_succ_self, choose_two, mul_comm]
  exact rfl


end more_red_than_white_red_not_less_than_white_l314_314044


namespace parallelogram_angle_B_l314_314213

theorem parallelogram_angle_B (A C B D : ℝ) (h₁ : A + C = 110) (h₂ : A = C) : B = 125 :=
by sorry

end parallelogram_angle_B_l314_314213


namespace reciprocal_of_repeating_decimal_l314_314278

theorem reciprocal_of_repeating_decimal : 
  (1 : ℚ) / (34 / 99 : ℚ) = 99 / 34 :=
by sorry

end reciprocal_of_repeating_decimal_l314_314278


namespace minimum_value_of_g_l314_314393

noncomputable def g (a b x : ℝ) : ℝ :=
  max (|x + a|) (|x + b|)

theorem minimum_value_of_g (a b : ℝ) (h : a < b) :
  ∃ x : ℝ, g a b x = (b - a) / 2 :=
by
  use - (a + b) / 2
  sorry

end minimum_value_of_g_l314_314393


namespace line_circle_relationship_l314_314339

noncomputable def point_on_line (k : ℝ) : Prop :=
  (0 : ℝ, 1 : ℝ) ∈ {p | p.2 = k * p.1 + 1}

noncomputable def point_in_circle : Prop :=
  (0 : ℝ)^2 + (1 : ℝ)^2 < 2

noncomputable def center_of_circle_in_line (k : ℝ) : Prop :=
  (0 : ℝ, 0 : ℝ) ∈ {p | p.2 = k * p.1 + 1}

noncomputable def line_intersects_circle (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 = 2 ∧ y = k * x + 1

theorem line_circle_relationship (k : ℝ) :
  point_on_line k ∧ point_in_circle ∧ ¬center_of_circle_in_line k → line_intersects_circle k :=
by
  sorry

end line_circle_relationship_l314_314339


namespace distance_to_line_is_constant_l314_314202

noncomputable def center_of_circle (a : ℝ) : (ℝ × ℝ) := (a, a)

noncomputable def circle_equation (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - a)^2 = a^2

noncomputable def distance_from_center_to_line (x1 y1 : ℝ) : ℝ :=
  abs (2 * x1 - y1 - 3) / sqrt (2^2 + (-1)^2)

theorem distance_to_line_is_constant (a : ℝ) (h : circle_equation a 2 1) (hx : a = 5 ∨ a = 1) :
  distance_from_center_to_line a a = 2 * sqrt 5 / 5 :=
by
  sorry

end distance_to_line_is_constant_l314_314202


namespace circle_tangent_distance_l314_314186

theorem circle_tangent_distance 
    (a : ℝ) (h_center : ∃ a : ℝ, a > 0 ∧ (circle_center = (a, a)) ∧ ((2 - a)^2 + (1 - a)^2 = a^2)) 
    (h_line : ∀ x y, distance_to_line (circle_center x y) (2 * x - y - 3) = 2 / sqrt (2^2 + 1^2)) :
    distance_to_line (circle_center 1 1) (2 * 1 - 1 - 3) = 2 * sqrt 5 / 5 ∧ distance_to_line (circle_center 5 5) (2 * 5 - 5 - 3) = 2 * sqrt 5 / 5 :=
by
    sorry

end circle_tangent_distance_l314_314186


namespace square_sides_product_l314_314450

theorem square_sides_product (a : ℝ) : 
  (∃ s : ℝ, s = 5 ∧ (a = -3 + s ∨ a = -3 - s)) → (a = 2 ∨ a = -8) → -8 * 2 = -16 :=
by
  intro _ _
  exact rfl

end square_sides_product_l314_314450


namespace number_being_divided_l314_314232

theorem number_being_divided (divisor quotient remainder number : ℕ) 
  (h_divisor : divisor = 3) 
  (h_quotient : quotient = 7) 
  (h_remainder : remainder = 1)
  (h_number : number = divisor * quotient + remainder) : 
  number = 22 :=
by
  rw [h_divisor, h_quotient, h_remainder] at h_number
  exact h_number

end number_being_divided_l314_314232


namespace gcd_possible_values_count_l314_314060

theorem gcd_possible_values_count : ∃ a b : ℕ, a * b = 360 ∧ (∃ gcds : Finset ℕ, gcds = {d | ∃ a b : ℕ, a * b = 360 ∧ d = Nat.gcd a b} ∧ gcds.card = 6) :=
sorry

end gcd_possible_values_count_l314_314060


namespace john_average_speed_l314_314216

noncomputable def time_uphill : ℝ := 45 / 60 -- 45 minutes converted to hours
noncomputable def distance_uphill : ℝ := 2   -- 2 km

noncomputable def time_downhill : ℝ := 15 / 60 -- 15 minutes converted to hours
noncomputable def distance_downhill : ℝ := 2   -- 2 km

noncomputable def total_distance : ℝ := distance_uphill + distance_downhill
noncomputable def total_time : ℝ := time_uphill + time_downhill

theorem john_average_speed : total_distance / total_time = 4 :=
by
  have h1 : total_distance = 4 := by sorry
  have h2 : total_time = 1 := by sorry
  rw [h1, h2]
  norm_num

end john_average_speed_l314_314216


namespace photograph_goal_reach_l314_314231

-- Define the initial number of photographs
def initial_photos : ℕ := 250

-- Define the percentage splits initially
def beth_pct_init : ℝ := 0.40
def my_pct_init : ℝ := 0.35
def julia_pct_init : ℝ := 0.25

-- Define the photographs taken initially by each person
def beth_photos_init : ℕ := 100
def my_photos_init : ℕ := 88
def julia_photos_init : ℕ := 63

-- Confirm initial photographs sum
example (h : beth_photos_init + my_photos_init + julia_photos_init = 251) : true := 
by trivial

-- Define today's decreased productivity percentages
def beth_decrease_pct : ℝ := 0.35
def my_decrease_pct : ℝ := 0.45
def julia_decrease_pct : ℝ := 0.25

-- Define the photographs taken today by each person after decreases
def beth_photos_today : ℕ := 65
def my_photos_today : ℕ := 48
def julia_photos_today : ℕ := 47

-- Sum of photographs taken today
def total_photos_today : ℕ := 160

-- Define the initial plus today's needed photographs to reach goal
def goal_photos : ℕ := 650

-- Define the additional number of photographs needed
def additional_photos_needed : ℕ := 399 - total_photos_today

-- Final proof statement
theorem photograph_goal_reach : 
  (beth_photos_init + my_photos_init + julia_photos_init) + (beth_photos_today + my_photos_today + julia_photos_today) + additional_photos_needed = goal_photos := 
by sorry

end photograph_goal_reach_l314_314231


namespace problem_l314_314355

def f (x : ℝ) : ℝ := x^2 - 3 * x + 7
def g (x : ℝ) : ℝ := 2 * x + 4
theorem problem : f (g 5) - g (f 5) = 123 := 
by 
  sorry

end problem_l314_314355


namespace gcd_possible_values_count_l314_314069

theorem gcd_possible_values_count (a b : ℕ) (h_ab : a * b = 360) : 
  (∃ d, d = Nat.gcd a b ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 12)) ∧ 
  (∃ n, n = 6) := 
by
  sorry

end gcd_possible_values_count_l314_314069


namespace total_miles_traveled_l314_314394

-- Define the conditions
def travel_time_per_mile (n : ℕ) : ℕ :=
  match n with
  | 0 => 10
  | _ => 10 + 6 * n

def daily_miles (n : ℕ) : ℕ :=
  60 / travel_time_per_mile n

-- Statement of the problem
theorem total_miles_traveled : (daily_miles 0 + daily_miles 1 + daily_miles 2 + daily_miles 3 + daily_miles 4) = 20 := by
  sorry

end total_miles_traveled_l314_314394


namespace angle_in_third_quadrant_l314_314326

theorem angle_in_third_quadrant (α : ℝ) (h : α = 2023) : 180 < α % 360 ∧ α % 360 < 270 := by
  sorry

end angle_in_third_quadrant_l314_314326


namespace selling_price_is_correct_l314_314322

noncomputable def cost_price : ℝ := 192
def profit_percentage : ℝ := 0.25
def profit (cp : ℝ) (pp : ℝ) : ℝ := pp * cp
def selling_price (cp : ℝ) (pft : ℝ) : ℝ := cp + pft

theorem selling_price_is_correct : selling_price cost_price (profit cost_price profit_percentage) = 240 :=
sorry

end selling_price_is_correct_l314_314322


namespace blue_faces_ratio_l314_314417

-- Define the cube dimensions and derived properties
def cube_dimension : ℕ := 13
def original_cube_surface_area : ℕ := 6 * (cube_dimension ^ 2)
def total_number_of_mini_cubes : ℕ := cube_dimension ^ 3
def total_mini_cube_faces : ℕ := 6 * total_number_of_mini_cubes
def red_faces : ℕ := original_cube_surface_area
def blue_faces : ℕ := total_mini_cube_faces - red_faces
def blue_to_red_ratio : ℕ := blue_faces / red_faces

-- The statement to be proved
theorem blue_faces_ratio :
  blue_to_red_ratio = 12 := by
  sorry

end blue_faces_ratio_l314_314417


namespace area_ratio_of_octagons_is_4_l314_314098

-- Define the given conditions
variable (r : ℝ) -- radius of the common circle

def side_length_circumscribed_octagon := √2 * r
def side_length_inscribed_octagon := r / √2

-- Areas of polygons scale with the square of the side length
def area_ratio_circumscribed_to_inscribed := (side_length_circumscribed_octagon r / side_length_inscribed_octagon r)^2

theorem area_ratio_of_octagons_is_4 :
  area_ratio_circumscribed_to_inscribed r = 4 := by
  sorry

end area_ratio_of_octagons_is_4_l314_314098


namespace closest_angle_l314_314032

/-- Prove the angle formed by the side edge and the base of a regular quadrilateral pyramid 
with base edge length 2017 and side edge length 2000 is closest to 40 degrees from the given options. -/
theorem closest_angle (a : ℝ) (l : ℝ) (θ : ℝ) (h : ℝ) :
  a = 2017 → l = 2000 →
  h = Real.sqrt (l^2 - (a/2)^2) →
  θ = Real.arccos ((a/2) / l) →
  θ < Real.pi / 4 →
  40 < θ * 180 / Real.pi * 1.0 ∧  θ * 180 / Real.pi < 50 -> θ * 180 / Real.pi = 40.0 :=
sorry

end closest_angle_l314_314032


namespace blue_to_red_face_area_ratio_l314_314445

theorem blue_to_red_face_area_ratio :
  let original_cube_dim := 13
  let red_face_area := 6 * original_cube_dim^2
  let total_faces := 6 * original_cube_dim^3
  let blue_face_area := total_faces - red_face_area
  (blue_face_area / red_face_area) = 12 :=
by
  sorry

end blue_to_red_face_area_ratio_l314_314445


namespace ratio_of_octagon_areas_l314_314103

-- Define the relevant terms and conditions
noncomputable def radius_of_circle (r : ℝ) := r
noncomputable def side_length_inscribed_octagon (r : ℝ) := r * Real.sqrt 2
noncomputable def side_length_circumscribed_octagon (r : ℝ) := r * Real.sqrt 2

-- Area of a regular octagon in terms of its side length
noncomputable def area_regular_octagon (s : ℝ) := 2 * (1 + Real.sqrt 2) * s^2

-- Areas of inscribed and circumscribed octagons
noncomputable def area_inscribed_octagon (r : ℝ) := area_regular_octagon (side_length_inscribed_octagon r)
noncomputable def area_circumscribed_octagon (r : ℝ) := area_regular_octagon (side_length_circumscribed_octagon r)

-- The main statement to prove
theorem ratio_of_octagon_areas (r : ℝ) : 
  area_circumscribed_octagon r / area_inscribed_octagon r = 1 :=
by
  sorry

end ratio_of_octagon_areas_l314_314103


namespace blue_red_face_area_ratio_l314_314435

theorem blue_red_face_area_ratio : 
  let n := 13
  let total_red_faces := 6 * n^2
  let total_faces := 6 * n^3
  let total_blue_faces := total_faces - total_red_faces
  let ratio := total_blue_faces / total_red_faces
  ratio = 12 :=
by
  let n := 13
  let total_red_faces := 6 * n^2
  let total_faces := 6 * n^3
  let total_blue_faces := total_faces - total_red_faces
  let ratio := total_blue_faces / total_red_faces
  have h : ratio = (6 * n^3 - 6 * n^2) / (6 * n^2) := rfl
  have h2 : ratio = (6 * n^2 * (n - 1)) / (6 * n^2) := by rw h
  have h3 : ratio = n - 1 := by rw [mul_div_cancel' (6 * n^2) (6 * n^2), h2]
  exact h3
  simp only [n]
  sorry

end blue_red_face_area_ratio_l314_314435


namespace flight_duration_l314_314018

theorem flight_duration (departure_time arrival_time : ℕ) (time_difference : ℕ) (h m : ℕ) (m_bound : 0 < m ∧ m < 60) 
  (h_val : h = 1) (m_val : m = 35)  : h + m = 36 := by
  sorry

end flight_duration_l314_314018


namespace paper_cups_pallets_l314_314126

theorem paper_cups_pallets (total_pallets : ℕ) (paper_towels_fraction tissues_fraction paper_plates_fraction : ℚ) :
  total_pallets = 20 → paper_towels_fraction = 1 / 2 → tissues_fraction = 1 / 4 → paper_plates_fraction = 1 / 5 →
  total_pallets - (total_pallets * paper_towels_fraction + total_pallets * tissues_fraction + total_pallets * paper_plates_fraction) = 1 :=
by sorry

end paper_cups_pallets_l314_314126


namespace probability_one_black_ball_l314_314012

/-- There are 10 balls in a box, among which 3 are black and 7 are white. Each person draws a ball,
records its color, and puts it back before the next draw. We need to prove that the probability
that exactly one out of three people will draw a black ball is 0.441. -/
theorem probability_one_black_ball (h : 0 < 10 ∧ 3 < 10):
Prob = (finset.choose 3 1) * (0.7 * 0.7 * 0.3) := 
sorry

end probability_one_black_ball_l314_314012


namespace quadratic_root_exists_l314_314215

theorem quadratic_root_exists (a b c : ℝ) : 
  ∃ x : ℝ, (a * x^2 + 2 * b * x + c = 0) ∨ (b * x^2 + 2 * c * x + a = 0) ∨ (c * x^2 + 2 * a * x + b = 0) :=
by sorry

end quadratic_root_exists_l314_314215


namespace rectangle_area_increase_l314_314449

theorem rectangle_area_increase (b : ℕ) (h1 : 2 * b = 40) (h2 : b = 20) : 
  let l := 2 * b
  let A_original := l * b
  let l_new := l - 5
  let b_new := b + 5
  let A_new := l_new * b_new
  A_new - A_original = 75 := 
by
  sorry

end rectangle_area_increase_l314_314449


namespace ratio_of_areas_of_octagons_l314_314094

theorem ratio_of_areas_of_octagons (r : ℝ) :
  let side_length_inscribed := r
  let side_length_circumscribed := r * real.sec (real.pi / 8)
  let area_inscribed := 2 * (1 + real.sqrt 2) * side_length_inscribed ^ 2
  let area_circumscribed := 2 * (1 + real.sqrt 2) * side_length_circumscribed ^ 2
  area_circumscribed / area_inscribed = 4 - 2 * real.sqrt 2 :=
sorry

end ratio_of_areas_of_octagons_l314_314094


namespace distance_from_center_to_line_l314_314190

noncomputable def circle_center_is_a (a : ℝ) : Prop :=
  (2 - a)^2 + (1 - a)^2 = a^2

theorem distance_from_center_to_line (a : ℝ) (h : 0 < a) (hab : circle_center_is_a a) :
  (∀x0 y0 : ℝ, (x0, y0) = (a, a) → (|2 * x0 - y0 - 3| / Real.sqrt (2^2 + 1^2)) = (2 * Real.sqrt 5 / 5)) :=
by
  sorry

end distance_from_center_to_line_l314_314190


namespace find_missing_number_l314_314214

theorem find_missing_number (x : ℚ) (h : 11 * x + 4 = 7) : x = 9 / 11 :=
sorry

end find_missing_number_l314_314214


namespace strawberries_count_l314_314019

def strawberries_total (J M Z : ℕ) : ℕ :=
  J + M + Z

theorem strawberries_count (J M Z : ℕ) (h1 : J + M = 350) (h2 : M + Z = 250) (h3 : Z = 200) : 
  strawberries_total J M Z = 550 :=
by
  sorry

end strawberries_count_l314_314019


namespace octagon_area_ratio_l314_314119

theorem octagon_area_ratio (r : ℝ) : 
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2 := 
by
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  have h : circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2
  exact h
  sorry

end octagon_area_ratio_l314_314119


namespace yellow_beads_needed_l314_314334

variable (Total green yellow : ℕ)

theorem yellow_beads_needed (h_green : green = 4) (h_yellow : yellow = 0) (h_fraction : (4 / 5 : ℚ) = 4 / (green + yellow + 16)) :
    4 + 16 + green = Total := by
  sorry

end yellow_beads_needed_l314_314334


namespace area_of_triangle_MEF_correct_l314_314211

noncomputable def area_of_triangle_MEF : ℝ :=
  let r := 10
  let chord_length := 12
  let parallel_segment_length := 15
  let angle_MOA := 30.0
  (1 / 2) * chord_length * (2 * Real.sqrt 21)

theorem area_of_triangle_MEF_correct :
  area_of_triangle_MEF = 12 * Real.sqrt 21 :=
by
  -- proof will go here
  sorry

end area_of_triangle_MEF_correct_l314_314211


namespace cone_from_sector_l314_314075

def cone_can_be_formed (θ : ℝ) (r_sector : ℝ) (r_cone_base : ℝ) (l_slant_height : ℝ) : Prop :=
  θ = 270 ∧ r_sector = 12 ∧ ∃ L, L = θ / 360 * (2 * Real.pi * r_sector) ∧ 2 * Real.pi * r_cone_base = L ∧ l_slant_height = r_sector

theorem cone_from_sector (base_radius slant_height : ℝ) :
  cone_can_be_formed 270 12 base_radius slant_height ↔ base_radius = 9 ∧ slant_height = 12 :=
by
  sorry

end cone_from_sector_l314_314075


namespace distribution_not_possible_l314_314458

open Finset

-- Definitions of regions
inductive Region
| A | B | C | D | E | F | G

-- Establishing distinct values for these regions
open Region
def regions := {A, B, C, D, E, F, G}

-- Assume there exist three lines which divide the regions
noncomputable def lines : Finset (Finset Region) :=
  {{A, B, C}, {D, E, F, G}, -- Line 1
   {A, D, E}, {B, C, F, G}, -- Line 2
   {A, B, F}, {C, D, E, G}} -- Line 3

-- Sum conditions for each line's partition regions 
def sum {α : Type*} [AddCommMonoid α] (s : Finset α) (f : α → ℕ) : ℕ := s.sum f

-- Distributing numbers 1 to 7 in such a way that sums on either side of lines are equal
theorem distribution_not_possible :
  (∀ (a b c d e f g : ℕ), a ∈ {1, 2, 3, 4, 5, 6, 7} ∧
                         b ∈ {1, 2, 3, 4, 5, 6, 7} ∧
                         c ∈ {1, 2, 3, 4, 5, 6, 7} ∧
                         d ∈ {1, 2, 3, 4, 5, 6, 7} ∧
                         e ∈ {1, 2, 3, 4, 5, 6, 7} ∧
                         f ∈ {1, 2, 3, 4, 5, 6, 7} ∧
                         g ∈ {1, 2, 3, 4, 5, 6, 7} ∧
                         sum {A, B, C} (λ x, if x = A then a else if x = B then b else if x = C then c else 0) = 
                         sum {D, E, F, G} (λ x, if x = D then d else if x = E then e else if x = F then f else if x = G then g else 0) ∧
                         sum {A, D, E} (λ x, if x = A then a else if x = D then d else if x = E then e else 0) = 
                         sum {B, C, F, G} (λ x, if x = B then b else if x = C then c else if x = F then f else if x = G then g else 0) ∧
                         sum {A, B, F} (λ x, if x = A then a else if x = B then b else if x = F then f else 0) = 
                         sum {C, D, E, G} (λ x, if x = C then c else if x = D then d else if x = E then e else if x = G then g else 0)) → false :=
sorry

end distribution_not_possible_l314_314458


namespace total_population_l314_314042

def grown_ups : ℕ := 5256
def children : ℕ := 2987

theorem total_population : grown_ups + children = 8243 :=
by
  sorry

end total_population_l314_314042


namespace blue_red_face_area_ratio_l314_314436

theorem blue_red_face_area_ratio : 
  let n := 13
  let total_red_faces := 6 * n^2
  let total_faces := 6 * n^3
  let total_blue_faces := total_faces - total_red_faces
  let ratio := total_blue_faces / total_red_faces
  ratio = 12 :=
by
  let n := 13
  let total_red_faces := 6 * n^2
  let total_faces := 6 * n^3
  let total_blue_faces := total_faces - total_red_faces
  let ratio := total_blue_faces / total_red_faces
  have h : ratio = (6 * n^3 - 6 * n^2) / (6 * n^2) := rfl
  have h2 : ratio = (6 * n^2 * (n - 1)) / (6 * n^2) := by rw h
  have h3 : ratio = n - 1 := by rw [mul_div_cancel' (6 * n^2) (6 * n^2), h2]
  exact h3
  simp only [n]
  sorry

end blue_red_face_area_ratio_l314_314436


namespace gcd_values_count_l314_314074

theorem gcd_values_count (a b : ℕ) (h : a * b = 360) : 
  ∃ n, n = (Set.toFinset (Set.image2 Nat.gcd {d | ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * y = 360 ∧ x = d ∨ y = d} {d | ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * y = 360 ∧ x = d ∨ y = d})).card ∧ n = 12 :=
by
  sorry

end gcd_values_count_l314_314074


namespace spending_required_for_free_shipping_l314_314217

def shampoo_cost : ℕ := 10
def conditioner_cost : ℕ := 10
def lotion_cost : ℕ := 6
def shampoo_count : ℕ := 1
def conditioner_count : ℕ := 1
def lotion_count : ℕ := 3
def additional_spending_needed : ℕ := 12
def current_spending : ℕ := (shampoo_cost * shampoo_count) + (conditioner_cost * conditioner_count) + (lotion_cost * lotion_count)

theorem spending_required_for_free_shipping : current_spending + additional_spending_needed = 50 := by
  sorry

end spending_required_for_free_shipping_l314_314217


namespace x_y_value_l314_314027

theorem x_y_value (x y : ℝ) (h : x^2 + y^2 = 8 * x - 4 * y - 30) : x + y = 2 :=
sorry

end x_y_value_l314_314027


namespace blue_faces_ratio_l314_314416

-- Define the cube dimensions and derived properties
def cube_dimension : ℕ := 13
def original_cube_surface_area : ℕ := 6 * (cube_dimension ^ 2)
def total_number_of_mini_cubes : ℕ := cube_dimension ^ 3
def total_mini_cube_faces : ℕ := 6 * total_number_of_mini_cubes
def red_faces : ℕ := original_cube_surface_area
def blue_faces : ℕ := total_mini_cube_faces - red_faces
def blue_to_red_ratio : ℕ := blue_faces / red_faces

-- The statement to be proved
theorem blue_faces_ratio :
  blue_to_red_ratio = 12 := by
  sorry

end blue_faces_ratio_l314_314416


namespace difference_is_2395_l314_314249

def S : ℕ := 476
def L : ℕ := 6 * S + 15
def difference : ℕ := L - S

theorem difference_is_2395 : difference = 2395 :=
by
  sorry

end difference_is_2395_l314_314249


namespace problem_statement_l314_314222

theorem problem_statement (M N : ℕ) 
  (hM : M = 2020 / 5) 
  (hN : N = 2020 / 20) : 10 * M / N = 40 := 
by
  sorry

end problem_statement_l314_314222


namespace num_terms_arith_seq_l314_314365

theorem num_terms_arith_seq {a d t : ℕ} (h_a : a = 5) (h_d : d = 3) (h_t : t = 140) :
  ∃ n : ℕ, t = a + (n-1) * d ∧ n = 46 :=
by
  sorry

end num_terms_arith_seq_l314_314365


namespace swimmers_meetings_in_15_minutes_l314_314459

noncomputable def swimmers_pass_each_other_count 
    (pool_length : ℕ) (rate_swimmer1 : ℕ) (rate_swimmer2 : ℕ) (time_minutes : ℕ) : ℕ :=
sorry -- Definition of the function to count passing times

theorem swimmers_meetings_in_15_minutes :
  swimmers_pass_each_other_count 120 4 3 15 = 23 :=
sorry -- The proof is not required as per instruction.

end swimmers_meetings_in_15_minutes_l314_314459


namespace no_root_l314_314244

theorem no_root :
  ∀ x : ℝ, x - (9 / (x - 4)) ≠ 4 - (9 / (x - 4)) :=
by
  intro x
  sorry

end no_root_l314_314244


namespace angle_in_third_quadrant_l314_314327

theorem angle_in_third_quadrant (θ : ℝ) (hθ : θ = 2023) :
    ∃ k : ℤ, (2023 - k * 360) = 223 ∧ 180 ≤ 223 ∧ 223 < 270 := by
sorry

end angle_in_third_quadrant_l314_314327


namespace min_marked_cells_l314_314461

theorem min_marked_cells (marking : Fin 15 → Fin 15 → Prop) :
  (∀ i : Fin 15, ∃ j : Fin 15, ∀ k : Fin 10, marking i (j + k % 15)) ∧
  (∀ j : Fin 15, ∃ i : Fin 15, ∀ k : Fin 10, marking (i + k % 15) j) →
  ∃s : Finset (Fin 15 × Fin 15), s.card = 20 ∧ ∀ i : Fin 15, (∃ j, (i, j) ∈ s ∨ (j, i) ∈ s) :=
sorry

end min_marked_cells_l314_314461


namespace number_of_jump_sequences_l314_314081

def jump_sequences (a : ℕ → ℕ) : Prop :=
  (a 1 = 1) ∧
  (a 2 = 2) ∧
  (a 3 = 3) ∧
  (∀ n, n ≥ 3 → a (n + 1) = a n + a (n - 2))

theorem number_of_jump_sequences :
  ∃ a : ℕ → ℕ, jump_sequences a ∧ a 11 = 60 :=
by
  sorry

end number_of_jump_sequences_l314_314081


namespace initial_num_balls_eq_18_l314_314301

variable (N B : ℕ)
variable (B_eq : B = 6)
variable (prob_cond : ((B - 3) : ℚ) / ((N - 3) : ℚ) = 1 / 5)

-- Prove that N = 18 given the conditions
theorem initial_num_balls_eq_18 (N B : ℕ) (B_eq : B = 6) (prob_cond : ((B - 3) : ℚ) / ((N - 3) : ℚ) = 1 / 5) : N = 18 := 
  sorry

end initial_num_balls_eq_18_l314_314301


namespace blue_to_red_face_area_ratio_l314_314444

theorem blue_to_red_face_area_ratio :
  let original_cube_dim := 13
  let red_face_area := 6 * original_cube_dim^2
  let total_faces := 6 * original_cube_dim^3
  let blue_face_area := total_faces - red_face_area
  (blue_face_area / red_face_area) = 12 :=
by
  sorry

end blue_to_red_face_area_ratio_l314_314444


namespace blue_face_area_factor_l314_314439

def cube_side_length : ℕ := 13

def original_red_face_area : ℕ := 6 * cube_side_length ^ 2
def total_mini_cube_face_area : ℕ := 6 * cube_side_length ^ 3
def blue_face_area : ℕ := total_mini_cube_face_area - original_red_face_area

theorem blue_face_area_factor :
  (blue_face_area / original_red_face_area) = 12 :=
by
  sorry

end blue_face_area_factor_l314_314439


namespace ratio_of_areas_of_octagons_l314_314093

theorem ratio_of_areas_of_octagons (r : ℝ) :
  let side_length_inscribed := r
  let side_length_circumscribed := r * real.sec (real.pi / 8)
  let area_inscribed := 2 * (1 + real.sqrt 2) * side_length_inscribed ^ 2
  let area_circumscribed := 2 * (1 + real.sqrt 2) * side_length_circumscribed ^ 2
  area_circumscribed / area_inscribed = 4 - 2 * real.sqrt 2 :=
sorry

end ratio_of_areas_of_octagons_l314_314093


namespace shorter_leg_length_l314_314125

theorem shorter_leg_length (a b c : ℝ) (h1 : b = 10) (h2 : a^2 + b^2 = c^2) (h3 : c = 2 * a) : 
  a = 10 * Real.sqrt 3 / 3 :=
by
  sorry

end shorter_leg_length_l314_314125


namespace units_digit_of_17_pow_28_l314_314464

theorem units_digit_of_17_pow_28 : ((17:ℕ) ^ 28) % 10 = 1 := by
  sorry

end units_digit_of_17_pow_28_l314_314464


namespace gcd_possible_values_count_l314_314062

theorem gcd_possible_values_count : ∃ a b : ℕ, a * b = 360 ∧ (∃ gcds : Finset ℕ, gcds = {d | ∃ a b : ℕ, a * b = 360 ∧ d = Nat.gcd a b} ∧ gcds.card = 6) :=
sorry

end gcd_possible_values_count_l314_314062


namespace blue_faces_ratio_l314_314415

-- Define the cube dimensions and derived properties
def cube_dimension : ℕ := 13
def original_cube_surface_area : ℕ := 6 * (cube_dimension ^ 2)
def total_number_of_mini_cubes : ℕ := cube_dimension ^ 3
def total_mini_cube_faces : ℕ := 6 * total_number_of_mini_cubes
def red_faces : ℕ := original_cube_surface_area
def blue_faces : ℕ := total_mini_cube_faces - red_faces
def blue_to_red_ratio : ℕ := blue_faces / red_faces

-- The statement to be proved
theorem blue_faces_ratio :
  blue_to_red_ratio = 12 := by
  sorry

end blue_faces_ratio_l314_314415


namespace West_oil_production_NonWest_oil_production_Russia_oil_production_oil_production_all_regions_l314_314208

-- Definitions for the conditions
def oil_per_person_West : ℝ := 55.084
def oil_per_person_NonWest : ℝ := 214.59
def oil_per_person_Russia : ℝ := 1038.33

-- Theorem statements asserting the oil production per person
theorem West_oil_production : 
  (55.084 : ℝ) = oil_per_person_West := sorry

theorem NonWest_oil_production : 
  (214.59 : ℝ) = oil_per_person_NonWest := sorry

theorem Russia_oil_production : 
  (1038.33 : ℝ) = oil_per_person_Russia := sorry

-- Conjunction of all the statements proving the original problem's solutions
theorem oil_production_all_regions : 
  (55.084 : ℝ) = oil_per_person_West ∧
  (214.59 : ℝ) = oil_per_person_NonWest ∧
  (1038.33 : ℝ) = oil_per_person_Russia := 
by 
    apply And.intro
    . exact West_oil_production
    . apply And.intro
        . exact NonWest_oil_production
        . exact Russia_oil_production

end West_oil_production_NonWest_oil_production_Russia_oil_production_oil_production_all_regions_l314_314208


namespace ratio_of_octagon_areas_l314_314104

-- Define the relevant terms and conditions
noncomputable def radius_of_circle (r : ℝ) := r
noncomputable def side_length_inscribed_octagon (r : ℝ) := r * Real.sqrt 2
noncomputable def side_length_circumscribed_octagon (r : ℝ) := r * Real.sqrt 2

-- Area of a regular octagon in terms of its side length
noncomputable def area_regular_octagon (s : ℝ) := 2 * (1 + Real.sqrt 2) * s^2

-- Areas of inscribed and circumscribed octagons
noncomputable def area_inscribed_octagon (r : ℝ) := area_regular_octagon (side_length_inscribed_octagon r)
noncomputable def area_circumscribed_octagon (r : ℝ) := area_regular_octagon (side_length_circumscribed_octagon r)

-- The main statement to prove
theorem ratio_of_octagon_areas (r : ℝ) : 
  area_circumscribed_octagon r / area_inscribed_octagon r = 1 :=
by
  sorry

end ratio_of_octagon_areas_l314_314104


namespace problem_statement_l314_314298

variables {totalBuyers : ℕ}
variables {C M K CM CK MK CMK : ℕ}

-- Given conditions
def conditions (totalBuyers : ℕ) (C : ℕ) (M : ℕ) (K : ℕ)
  (CM : ℕ) (CK : ℕ) (MK : ℕ) (CMK : ℕ) : Prop :=
  totalBuyers = 150 ∧
  C = 70 ∧
  M = 60 ∧
  K = 50 ∧
  CM = 25 ∧
  CK = 15 ∧
  MK = 10 ∧
  CMK = 5

-- Number of buyers who purchase at least one mixture
def buyersAtLeastOne (C : ℕ) (M : ℕ) (K : ℕ)
  (CM : ℕ) (CK : ℕ) (MK : ℕ) (CMK : ℕ) : ℕ :=
  C + M + K - CM - CK - MK + CMK

-- Number of buyers who purchase none
def buyersNone (totalBuyers : ℕ) (buyersAtLeastOne : ℕ) : ℕ :=
  totalBuyers - buyersAtLeastOne

-- Probability computation
def probabilityNone (totalBuyers : ℕ) (buyersNone : ℕ) : ℚ :=
  buyersNone / totalBuyers

-- Theorem statement
theorem problem_statement : conditions totalBuyers C M K CM CK MK CMK →
  probabilityNone totalBuyers (buyersNone totalBuyers (buyersAtLeastOne C M K CM CK MK CMK)) = 0.1 :=
by
  intros h
  -- Assumptions from the problem
  have h_total : totalBuyers = 150 := h.left
  have hC : C = 70 := h.right.left
  have hM : M = 60 := h.right.right.left
  have hK : K = 50 := h.right.right.right.left
  have hCM : CM = 25 := h.right.right.right.right.left
  have hCK : CK = 15 := h.right.right.right.right.right.left
  have hMK : MK = 10 := h.right.right.right.right.right.right.left
  have hCMK : CMK = 5 := h.right.right.right.right.right.right.right
  sorry

end problem_statement_l314_314298


namespace blue_red_face_area_ratio_l314_314434

theorem blue_red_face_area_ratio : 
  let n := 13
  let total_red_faces := 6 * n^2
  let total_faces := 6 * n^3
  let total_blue_faces := total_faces - total_red_faces
  let ratio := total_blue_faces / total_red_faces
  ratio = 12 :=
by
  let n := 13
  let total_red_faces := 6 * n^2
  let total_faces := 6 * n^3
  let total_blue_faces := total_faces - total_red_faces
  let ratio := total_blue_faces / total_red_faces
  have h : ratio = (6 * n^3 - 6 * n^2) / (6 * n^2) := rfl
  have h2 : ratio = (6 * n^2 * (n - 1)) / (6 * n^2) := by rw h
  have h3 : ratio = n - 1 := by rw [mul_div_cancel' (6 * n^2) (6 * n^2), h2]
  exact h3
  simp only [n]
  sorry

end blue_red_face_area_ratio_l314_314434


namespace distance_from_center_to_line_l314_314189

noncomputable def circle_center_is_a (a : ℝ) : Prop :=
  (2 - a)^2 + (1 - a)^2 = a^2

theorem distance_from_center_to_line (a : ℝ) (h : 0 < a) (hab : circle_center_is_a a) :
  (∀x0 y0 : ℝ, (x0, y0) = (a, a) → (|2 * x0 - y0 - 3| / Real.sqrt (2^2 + 1^2)) = (2 * Real.sqrt 5 / 5)) :=
by
  sorry

end distance_from_center_to_line_l314_314189


namespace cube_sum_from_square_l314_314173

noncomputable def a_plus_inv_a_squared_eq_5 (a : ℝ) : Prop :=
  (a + 1/a) ^ 2 = 5

theorem cube_sum_from_square (a : ℝ) (h : a_plus_inv_a_squared_eq_5 a) :
  a^3 + (1/a)^3 = 2 * Real.sqrt 5 ∨ a^3 + (1/a)^3 = -2 * Real.sqrt 5 :=
by
  sorry

end cube_sum_from_square_l314_314173


namespace keiko_speed_calc_l314_314219

noncomputable def keiko_speed (r : ℝ) (time_diff : ℝ) : ℝ :=
  let circumference_diff := 2 * Real.pi * 8
  circumference_diff / time_diff

theorem keiko_speed_calc (r : ℝ) (time_diff : ℝ) :
  keiko_speed r 48 = Real.pi / 3 := by
  sorry

end keiko_speed_calc_l314_314219


namespace probability_at_least_one_bean_distribution_of_X_expectation_of_X_l314_314341

noncomputable def total_ways := Nat.choose 6 3
noncomputable def ways_select_2_egg_1_bean := (Nat.choose 4 2) * (Nat.choose 2 1)
noncomputable def ways_select_1_egg_2_bean := (Nat.choose 4 1) * (Nat.choose 2 2)
noncomputable def at_least_one_bean_probability := (ways_select_2_egg_1_bean + ways_select_1_egg_2_bean) / total_ways

theorem probability_at_least_one_bean : at_least_one_bean_probability = 4 / 5 :=
by sorry

noncomputable def p_X_eq_0 := (Nat.choose 4 3) / total_ways
noncomputable def p_X_eq_1 := ways_select_2_egg_1_bean / total_ways
noncomputable def p_X_eq_2 := ways_select_1_egg_2_bean / total_ways

theorem distribution_of_X : p_X_eq_0 = 1 / 5 ∧ p_X_eq_1 = 3 / 5 ∧ p_X_eq_2 = 1 / 5 :=
by sorry

noncomputable def E_X := (0 * p_X_eq_0) + (1 * p_X_eq_1) + (2 * p_X_eq_2)

theorem expectation_of_X : E_X = 1 :=
by sorry

end probability_at_least_one_bean_distribution_of_X_expectation_of_X_l314_314341


namespace least_perimeter_of_triangle_l314_314033

-- Define the sides of the triangle
def side1 : ℕ := 40
def side2 : ℕ := 48

-- Given condition for the third side
def valid_third_side (x : ℕ) : Prop :=
  8 < x ∧ x < 88

-- The least possible perimeter given the conditions
def least_possible_perimeter : ℕ :=
  side1 + side2 + 9

theorem least_perimeter_of_triangle (x : ℕ) (h : valid_third_side x) (hx : x = 9) : least_possible_perimeter = 97 :=
by
  rw [least_possible_perimeter]
  exact rfl

end least_perimeter_of_triangle_l314_314033


namespace trapezoid_area_l314_314147

variables (R₁ R₂ : ℝ)

theorem trapezoid_area (h_eq : h = 4 * R₁ * R₂ / (R₁ + R₂)) (mn_eq : mn = 2 * Real.sqrt (R₁ * R₂)) :
  S_ABCD = 8 * R₁ * R₂ * Real.sqrt (R₁ * R₂) / (R₁ + R₂) :=
sorry

end trapezoid_area_l314_314147


namespace james_passenger_count_l314_314386

theorem james_passenger_count :
  ∀ (total_vehicles trucks buses taxis motorbikes cars trucks_population buses_population taxis_population motorbikes_population cars_population : ℕ),
  total_vehicles = 52 →
  trucks = 12 →
  buses = 2 →
  taxis = 2 * buses →
  motorbikes = total_vehicles - (trucks + buses + taxis + cars) →
  cars = 30 →
  trucks_population = 2 →
  buses_population = 15 →
  taxis_population = 2 →
  motorbikes_population = 1 →
  cars_population = 3 →
  (trucks * trucks_population + buses * buses_population + taxis * taxis_population +
   motorbikes * motorbikes_population + cars * cars_population) = 156 := 
by
  -- Placeholder for the proof
  sorry

end james_passenger_count_l314_314386


namespace xiao_ming_arrival_time_l314_314469

def left_home (departure_time : String) : Prop :=
  departure_time = "6:55"

def time_spent (duration : Nat) : Prop :=
  duration = 30

def arrival_time (arrival : String) : Prop :=
  arrival = "7:25"

theorem xiao_ming_arrival_time :
  left_home "6:55" → time_spent 30 → arrival_time "7:25" :=
by sorry

end xiao_ming_arrival_time_l314_314469


namespace solution_l314_314282

noncomputable def x : ℕ := 13

theorem solution : (3 * x) - (36 - x) = 16 := by
  sorry

end solution_l314_314282


namespace linear_function_does_not_pass_through_quadrant_3_l314_314252

theorem linear_function_does_not_pass_through_quadrant_3
  (f : ℝ → ℝ) (h : ∀ x, f x = -3 * x + 5) :
  ¬ (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ f x = y) :=
by
  sorry

end linear_function_does_not_pass_through_quadrant_3_l314_314252


namespace tic_tac_toe_winning_probability_l314_314316

theorem tic_tac_toe_winning_probability :
  let total_ways := Nat.binomial 9 3
  let winning_ways := 8
  let probability := winning_ways / total_ways
  probability = (2 : ℚ) / 21 :=
by
  let total_ways := Nat.binomial 9 3
  let winning_ways := 8
  let probability := winning_ways / total_ways
  have probability_correct : probability = (2 : ℚ) / 21 := sorry
  exact probability_correct

end tic_tac_toe_winning_probability_l314_314316


namespace red_blue_area_ratio_is_12_l314_314433

def red_blue_face_area_ratio : ℕ :=
  let big_cube_dim : ℕ := 13 in
  let total_red_faces : ℕ := 6 * big_cube_dim^2 in
  let total_mini_cube_faces : ℕ := 6 * big_cube_dim^3 in
  let total_blue_faces : ℕ := total_mini_cube_faces - total_red_faces in
  total_blue_faces / total_red_faces

theorem red_blue_area_ratio_is_12 :
  red_blue_face_area_ratio = 12 := by
  sorry

end red_blue_area_ratio_is_12_l314_314433


namespace gcd_possible_values_count_l314_314056

theorem gcd_possible_values_count (a b : ℕ) (h : a * b = 360) : 
  {d : ℕ | d ∣ a ∧ d ∣ b}.card = 6 :=
sorry

end gcd_possible_values_count_l314_314056


namespace sticker_count_l314_314258

def stickers_per_page : ℕ := 25
def num_pages : ℕ := 35
def total_stickers : ℕ := 875

theorem sticker_count : num_pages * stickers_per_page = total_stickers :=
by {
  sorry
}

end sticker_count_l314_314258


namespace octagon_area_ratio_l314_314120

theorem octagon_area_ratio (r : ℝ) : 
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2 := 
by
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  have h : circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2
  exact h
  sorry

end octagon_area_ratio_l314_314120


namespace range_of_a_l314_314164

theorem range_of_a {A : Set ℝ} (h1: ∀ x ∈ A, 2 * x + a > 0) (h2: 1 ∉ A) (h3: 2 ∈ A) : -4 < a ∧ a ≤ -2 := 
sorry

end range_of_a_l314_314164


namespace find_AC_l314_314210

theorem find_AC (AB DC AD : ℕ) (hAB : AB = 13) (hDC : DC = 20) (hAD : AD = 5) : 
  AC = 24.2 := 
sorry

end find_AC_l314_314210


namespace probability_sum_at_least_fifteen_l314_314239

theorem probability_sum_at_least_fifteen (s : Finset ℕ) (h_s : s = (Finset.range 14).map Nat.succ) :
  (∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + c ≥ 15 ∧ a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a < b ∧ b < c) →
  ((369 : ℚ) ^ -1 * 76 = 19 / 91) :=
by
  sorry

end probability_sum_at_least_fifteen_l314_314239


namespace direct_proportion_function_l314_314371

theorem direct_proportion_function (m : ℝ) (h : ∀ x : ℝ, -2*x + m = k*x → m = 0) : m = 0 :=
sorry

end direct_proportion_function_l314_314371


namespace gcd_possible_values_count_l314_314070

theorem gcd_possible_values_count (a b : ℕ) (h_ab : a * b = 360) : 
  (∃ d, d = Nat.gcd a b ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 12)) ∧ 
  (∃ n, n = 6) := 
by
  sorry

end gcd_possible_values_count_l314_314070


namespace ratio_of_Lev_to_Akeno_l314_314320

theorem ratio_of_Lev_to_Akeno (L : ℤ) (A : ℤ) (Ambrocio : ℤ) :
  A = 2985 ∧ Ambrocio = L - 177 ∧ A = L + Ambrocio + 1172 → L / A = 1 / 3 :=
by
  intro h
  sorry

end ratio_of_Lev_to_Akeno_l314_314320


namespace base_six_digits_unique_l314_314378

theorem base_six_digits_unique (b : ℕ) (h : (b-1)^2*(b-2) = 100) : b = 6 :=
by
  sorry

end base_six_digits_unique_l314_314378


namespace tic_tac_toe_probability_l314_314314

theorem tic_tac_toe_probability :
  let total_positions := Nat.choose 9 3,
      winning_positions := 8 in
  (winning_positions / total_positions : ℚ) = 2 / 21 :=
by
  sorry

end tic_tac_toe_probability_l314_314314


namespace find_s_l314_314390

theorem find_s (n r s c d : ℚ) 
  (h1 : Polynomial.X ^ 2 - Polynomial.C n * Polynomial.X + Polynomial.C 3 = 0) 
  (h2 : c * d = 3)
  (h3 : Polynomial.X ^ 2 - Polynomial.C r * Polynomial.X + Polynomial.C s = 
        Polynomial.C (c + d⁻¹) * Polynomial.C (d + c⁻¹)) : 
  s = 16 / 3 := 
by
  sorry

end find_s_l314_314390


namespace gcd_possible_values_count_l314_314057

theorem gcd_possible_values_count (a b : ℕ) (h : a * b = 360) : 
  {d : ℕ | d ∣ a ∧ d ∣ b}.card = 6 :=
sorry

end gcd_possible_values_count_l314_314057


namespace gcd_values_count_l314_314072

theorem gcd_values_count (a b : ℕ) (h : a * b = 360) : 
  ∃ n, n = (Set.toFinset (Set.image2 Nat.gcd {d | ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * y = 360 ∧ x = d ∨ y = d} {d | ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * y = 360 ∧ x = d ∨ y = d})).card ∧ n = 12 :=
by
  sorry

end gcd_values_count_l314_314072


namespace area_ratio_is_correct_l314_314112

noncomputable def areaRatioInscribedCircumscribedOctagon (r : ℝ) : ℝ :=
  let s1 := r * Real.sin (67.5 * Real.pi / 180)
  let s2 := r * Real.sin (67.5 * Real.pi / 180) / Real.cos (67.5 * Real.pi / 180)
  (s2 / s1) ^ 2

theorem area_ratio_is_correct (r : ℝ) : areaRatioInscribedCircumscribedOctagon r = 2 + Real.sqrt 2 :=
by
  sorry

end area_ratio_is_correct_l314_314112


namespace solve_inequality_l314_314408

theorem solve_inequality (x : ℝ) : 2 * (5 * x + 3) ≤ x - 3 * (1 - 2 * x) → x ≤ -3 :=
by
  sorry

end solve_inequality_l314_314408


namespace divisor_is_four_l314_314476

theorem divisor_is_four (n d k l : ℤ) (hn : n % d = 3) (h2n : (2 * n) % d = 2) (hd : d > 3) : d = 4 :=
by
  sorry

end divisor_is_four_l314_314476


namespace calculate_number_of_girls_l314_314375

-- Definitions based on the conditions provided
def ratio_girls_to_boys : ℕ := 3
def ratio_boys_to_girls : ℕ := 4
def total_students : ℕ := 35

-- The proof statement
theorem calculate_number_of_girls (k : ℕ) (hk : ratio_girls_to_boys * k + ratio_boys_to_girls * k = total_students) :
  ratio_girls_to_boys * k = 15 :=
by sorry

end calculate_number_of_girls_l314_314375


namespace sum_of_a_b_l314_314172

-- Define the conditions in Lean
def a : ℝ := 1
def b : ℝ := 1

-- Define the proof statement
theorem sum_of_a_b : a + b = 2 := by
  sorry

end sum_of_a_b_l314_314172


namespace angle_between_line_and_plane_l314_314146

open Real

def plane1 (x y z : ℝ) : Prop := 2*x - y - 3*z + 5 = 0
def plane2 (x y z : ℝ) : Prop := x + y - 2 = 0

def point_M : ℝ × ℝ × ℝ := (-2, 0, 3)
def point_N : ℝ × ℝ × ℝ := (0, 2, 2)
def point_K : ℝ × ℝ × ℝ := (3, -3, 1)

theorem angle_between_line_and_plane :
  ∃ α : ℝ, α = arcsin (22 / (3 * sqrt 102)) :=
by sorry

end angle_between_line_and_plane_l314_314146


namespace jade_transactions_l314_314024

theorem jade_transactions (mabel anthony cal jade : ℕ) 
    (h1 : mabel = 90) 
    (h2 : anthony = mabel + (10 * mabel / 100)) 
    (h3 : cal = 2 * anthony / 3) 
    (h4 : jade = cal + 18) : 
    jade = 84 := by 
  -- Start with given conditions
  rw [h1] at h2 
  have h2a : anthony = 99 := by norm_num; exact h2 
  rw [h2a] at h3 
  have h3a : cal = 66 := by norm_num; exact h3 
  rw [h3a] at h4 
  norm_num at h4 
  exact h4

end jade_transactions_l314_314024


namespace blue_face_area_greater_than_red_face_area_l314_314424

theorem blue_face_area_greater_than_red_face_area :
  let original_cube_side := 13
  let total_red_area := 6 * original_cube_side^2
  let num_mini_cubes := original_cube_side^3
  let total_faces_mini_cubes := 6 * num_mini_cubes
  let total_blue_area := total_faces_mini_cubes - total_red_area
  (total_blue_area / total_red_area) = 12 :=
by
  sorry

end blue_face_area_greater_than_red_face_area_l314_314424


namespace length_of_QR_of_triangle_l314_314454

def length_of_QR (PQ PR PM : ℝ) : ℝ := sorry

theorem length_of_QR_of_triangle (PQ PR : ℝ) (PM : ℝ) (hPQ : PQ = 4) (hPR : PR = 7) (hPM : PM = 7 / 2) : length_of_QR PQ PR PM = 9 := by
  sorry

end length_of_QR_of_triangle_l314_314454


namespace range_of_a_l314_314167

noncomputable def f (a x : ℝ) : ℝ := (a - 1) * x^2 + (a - 1) * x + 1

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f a x > 0) ↔ (1 ≤ a ∧ a < 5) := by
  sorry

end range_of_a_l314_314167


namespace solve_for_q_l314_314407

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 3 * q = 7) (h2 : 3 * p + 5 * q = 8) : q = 19 / 16 :=
by
  sorry

end solve_for_q_l314_314407


namespace same_side_of_line_l314_314205

open Real

theorem same_side_of_line (a : ℝ) :
  let O := (0, 0)
  let A := (1, 1)
  (O.1 + O.2 < a ↔ A.1 + A.2 < a) →
  a < 0 ∨ a > 2 := by
  sorry

end same_side_of_line_l314_314205


namespace non_working_games_l314_314399

def total_games : ℕ := 30
def working_games : ℕ := 17

theorem non_working_games :
  total_games - working_games = 13 := 
by 
  sorry

end non_working_games_l314_314399


namespace blue_face_area_greater_than_red_face_area_l314_314422

theorem blue_face_area_greater_than_red_face_area :
  let original_cube_side := 13
  let total_red_area := 6 * original_cube_side^2
  let num_mini_cubes := original_cube_side^3
  let total_faces_mini_cubes := 6 * num_mini_cubes
  let total_blue_area := total_faces_mini_cubes - total_red_area
  (total_blue_area / total_red_area) = 12 :=
by
  sorry

end blue_face_area_greater_than_red_face_area_l314_314422


namespace problem_l314_314366

variables {a b : ℝ}

theorem problem (h₁ : -1 < a) (h₂ : a < b) (h₃ : b < 0) : 
  (1/a > 1/b) ∧ (a^2 + b^2 > 2 * a * b) ∧ (a + (1/a) > b + (1/b)) :=
by
  sorry

end problem_l314_314366


namespace value_of_expression_l314_314309

def a : ℕ := 7
def b : ℕ := 5

theorem value_of_expression : (a^2 - b^2)^4 = 331776 := by
  sorry

end value_of_expression_l314_314309


namespace no_root_l314_314243

theorem no_root :
  ∀ x : ℝ, x - (9 / (x - 4)) ≠ 4 - (9 / (x - 4)) :=
by
  intro x
  sorry

end no_root_l314_314243


namespace find_omega_l314_314165

noncomputable def f (x : ℝ) (ω φ : ℝ) := Real.sin (ω * x + φ)

theorem find_omega (ω φ : ℝ) (hω : ω > 0) (hφ : 0 ≤ φ ∧ φ ≤ π)
  (h_even : ∀ x : ℝ, f x ω φ = f (-x) ω φ)
  (h_symm : ∀ x : ℝ, f (3 * π / 4 + x) ω φ = f (3 * π / 4 - x) ω φ)
  (h_mono : ∀ x1 x2 : ℝ, 0 ≤ x1 → x1 ≤ x2 → x2 ≤ π / 2 → f x1 ω φ ≤ f x2 ω φ) :
  ω = 2 / 3 ∨ ω = 2 :=
sorry

end find_omega_l314_314165


namespace red_blue_area_ratio_is_12_l314_314432

def red_blue_face_area_ratio : ℕ :=
  let big_cube_dim : ℕ := 13 in
  let total_red_faces : ℕ := 6 * big_cube_dim^2 in
  let total_mini_cube_faces : ℕ := 6 * big_cube_dim^3 in
  let total_blue_faces : ℕ := total_mini_cube_faces - total_red_faces in
  total_blue_faces / total_red_faces

theorem red_blue_area_ratio_is_12 :
  red_blue_face_area_ratio = 12 := by
  sorry

end red_blue_area_ratio_is_12_l314_314432


namespace ratio_of_area_of_inscribed_and_circumscribed_octagons_l314_314116

noncomputable def area_ratio_of_octagons (r : ℝ) : ℝ :=
  let cos22_5 := (√(2 + √2) / 2)
  let r_prime := r / cos22_5
  let area_ratio := (r_prime / r)^2
  area_ratio

theorem ratio_of_area_of_inscribed_and_circumscribed_octagons (r : ℝ) :
  area_ratio_of_octagons r = (4 - 2 * √2) / 2 := by
  sorry

end ratio_of_area_of_inscribed_and_circumscribed_octagons_l314_314116


namespace probability_at_least_one_red_bean_paste_probability_distribution_X_expectation_X_l314_314342

-- Definition of the conditions
def total_zongzi : ℕ := 6
def egg_yolk_zongzi : ℕ := 4
def red_bean_paste_zongzi : ℕ := 2
def total_selected_zongzi : ℕ := 3

-- Definitions for probability calculations
noncomputable def combination (n r : ℕ) : ℚ := (nat.factorial n) / ((nat.factorial r) * (nat.factorial (n-r)))

-- Statement 1: Prove the probability of at least one red bean paste zongzi
theorem probability_at_least_one_red_bean_paste :
  (combination egg_yolk_zongzi 2 * combination red_bean_paste_zongzi 1 + combination egg_yolk_zongzi 1 * combination red_bean_paste_zongzi 2) / combination total_zongzi total_selected_zongzi = 4 / 5 :=
by sorry

-- Definitions and theorems for probability distribution and expectation of X
def P_X_0 : ℚ := combination egg_yolk_zongzi 3 / combination total_zongzi total_selected_zongzi
def P_X_1 : ℚ := combination egg_yolk_zongzi 2 * combination red_bean_paste_zongzi 1 / combination total_zongzi total_selected_zongzi
def P_X_2 : ℚ := combination egg_yolk_zongzi 1 * combination red_bean_paste_zongzi 2 / combination total_zongzi total_selected_zongzi

theorem probability_distribution_X :
  (P_X_0 = 1 / 5) ∧ (P_X_1 = 3 / 5) ∧ (P_X_2 = 1 / 5) :=
by sorry

theorem expectation_X :
  (0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2) = 1 :=
by sorry

end probability_at_least_one_red_bean_paste_probability_distribution_X_expectation_X_l314_314342


namespace units_digit_17_pow_28_l314_314466

theorem units_digit_17_pow_28 : (17 ^ 28) % 10 = 1 :=
by
  sorry

end units_digit_17_pow_28_l314_314466


namespace lily_pads_cover_entire_lake_l314_314007

/-- 
If a patch of lily pads doubles in size every day and takes 57 days to cover half the lake,
then it will take 58 days to cover the entire lake.
-/
theorem lily_pads_cover_entire_lake (days_to_half : ℕ) (h : days_to_half = 57) : (days_to_half + 1 = 58) := by
  sorry

end lily_pads_cover_entire_lake_l314_314007


namespace greg_age_is_16_l314_314135

-- Define the ages of Cindy, Jan, Marcia, and Greg based on the conditions
def cindy_age : ℕ := 5
def jan_age : ℕ := cindy_age + 2
def marcia_age : ℕ := 2 * jan_age
def greg_age : ℕ := marcia_age + 2

-- Theorem statement: Prove that Greg's age is 16
theorem greg_age_is_16 : greg_age = 16 :=
by
  -- Proof would go here
  sorry

end greg_age_is_16_l314_314135


namespace concurrency_of_perpendiculars_l314_314374

universe u

-- Define the necessary concepts
variables {A B C D I I_a A' B' C' : Type u} [triangle ABC A B C]

-- Assume the required conditions
structure triangle_configuration (ABC I I_a A' B' C' : Type u) :=
(D : Type u) -- D is the intersection of the external bisector of angle A and line BC
(I : Type u) -- I is the incenter of triangle ABC
(I_a : Type u) -- I_a is the excenter opposite angle A
(A' : Type u) -- A' is the intersection of the perpendicular from I to DI_a and the circumcircle
(B' : Type u) -- Similarly defined point B'
(C' : Type u) -- Similarly defined point C'

theorem concurrency_of_perpendiculars 
  (config : triangle_configuration ABC I I_a A' B' C') :
  concurrent {A A'} ( {B B'}) ( {C C'} := sorry

end concurrency_of_perpendiculars_l314_314374


namespace quadrilateral_is_parallelogram_l314_314446

theorem quadrilateral_is_parallelogram (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 2 * a * b + 2 * c * d) : a = b ∧ c = d :=
by
  sorry

end quadrilateral_is_parallelogram_l314_314446


namespace gcd_possible_values_count_l314_314068

theorem gcd_possible_values_count (a b : ℕ) (h_ab : a * b = 360) : 
  (∃ d, d = Nat.gcd a b ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 12)) ∧ 
  (∃ n, n = 6) := 
by
  sorry

end gcd_possible_values_count_l314_314068


namespace octagon_area_ratio_l314_314118

theorem octagon_area_ratio (r : ℝ) : 
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2 := 
by
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  have h : circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2
  exact h
  sorry

end octagon_area_ratio_l314_314118


namespace eval_f_at_5_l314_314333

def f (x : ℝ) : ℝ := 2 * x^7 - 9 * x^6 + 5 * x^5 - 49 * x^4 - 5 * x^3 + 2 * x^2 + x + 1

theorem eval_f_at_5 : f 5 = 56 := 
 by 
   sorry

end eval_f_at_5_l314_314333


namespace total_pennies_after_addition_l314_314236

def initial_pennies_per_compartment : ℕ := 10
def compartments : ℕ := 20
def added_pennies_per_compartment : ℕ := 15

theorem total_pennies_after_addition :
  (initial_pennies_per_compartment + added_pennies_per_compartment) * compartments = 500 :=
by 
  sorry

end total_pennies_after_addition_l314_314236


namespace ratio_of_areas_of_octagons_l314_314121

theorem ratio_of_areas_of_octagons
  (r : ℝ)
  (sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2)
  (sin_67_5 : Real.sin (Real.pi / 180 * 67.5) = Real.sqrt (2 + Real.sqrt 2) / 2)
  : let smaller_octagon_area := 2 * (1 + Real.sqrt 2) * r^2 in
    let larger_octagon_area := 
      2 * (1 + Real.sqrt 2) * (2 * r * (1 + Real.sqrt 2))^2 in
    larger_octagon_area / smaller_octagon_area = 12 + 8 * Real.sqrt 2 :=
by
  sorry

end ratio_of_areas_of_octagons_l314_314121


namespace base10_equivalent_of_43210_7_l314_314273

def base7ToDecimal (num : Nat) : Nat :=
  let digits := [4, 3, 2, 1, 0]
  digits[0] * 7^4 + digits[1] * 7^3 + digits[2] * 7^2 + digits[3] * 7^1 + digits[4] * 7^0

theorem base10_equivalent_of_43210_7 :
  base7ToDecimal 43210 = 10738 :=
by
  sorry

end base10_equivalent_of_43210_7_l314_314273


namespace milburg_population_l314_314039

/-- Number of grown-ups in Milburg --/
def grownUps : ℕ := 5256

/-- Number of children in Milburg --/
def children : ℕ := 2987

/-- Total number of people in Milburg --/
def totalPeople : ℕ := grownUps + children

theorem milburg_population : totalPeople = 8243 := by
  have h1 : grownUps = 5256 := rfl
  have h2 : children = 2987 := rfl
  have h3 : totalPeople = grownUps + children := rfl
  have h4 : grownUps + children = 8243 := by
    calc
      5256 + 2987 = 8243 := by sorry -- Proof step to be filled in
  exact h4

end milburg_population_l314_314039


namespace sufficient_but_not_necessary_condition_l314_314353

variable (a₁ d : ℝ)

def S₄ := 4 * a₁ + 6 * d
def S₅ := 5 * a₁ + 10 * d
def S₆ := 6 * a₁ + 15 * d

theorem sufficient_but_not_necessary_condition (h : d > 1) :
  S₄ a₁ d + S₆ a₁ d > 2 * S₅ a₁ d :=
by
  -- proof omitted
  sorry

end sufficient_but_not_necessary_condition_l314_314353


namespace effective_speed_against_current_l314_314085

theorem effective_speed_against_current
  (speed_with_current : ℝ)
  (speed_of_current : ℝ)
  (headwind_speed : ℝ)
  (obstacle_reduction_pct : ℝ)
  (h_speed_with_current : speed_with_current = 25)
  (h_speed_of_current : speed_of_current = 4)
  (h_headwind_speed : headwind_speed = 2)
  (h_obstacle_reduction_pct : obstacle_reduction_pct = 0.15) :
  let speed_in_still_water := speed_with_current - speed_of_current
  let speed_against_current_headwind := speed_in_still_water - speed_of_current - headwind_speed
  let reduction_due_to_obstacles := obstacle_reduction_pct * speed_against_current_headwind
  let effective_speed := speed_against_current_headwind - reduction_due_to_obstacles
  effective_speed = 12.75 := by
{
  sorry
}

end effective_speed_against_current_l314_314085


namespace min_distance_to_line_l314_314373

theorem min_distance_to_line (m n : ℝ) (h : 4 * m + 3 * n = 10)
  : m^2 + n^2 ≥ 4 :=
sorry

end min_distance_to_line_l314_314373


namespace percent_of_day_is_hours_l314_314468

theorem percent_of_day_is_hours (h : ℝ) (day_hours : ℝ) (percent : ℝ) 
  (day_hours_def : day_hours = 24)
  (percent_def : percent = 29.166666666666668) :
  h = 7 :=
by
  sorry

end percent_of_day_is_hours_l314_314468


namespace time_to_drain_tank_l314_314087

theorem time_to_drain_tank (P L: ℝ) (hP : P = 1/3) (h_combined : P - L = 2/7) : 1 / L = 21 :=
by
  -- Proof omitted. Use the conditions given to show that 1 / L = 21.
  sorry

end time_to_drain_tank_l314_314087


namespace blue_area_factor_12_l314_314429

theorem blue_area_factor_12 : 
  let n := 13 in
  let total_red_area := 6 * n^2 in         -- The total red area on the surface of the original cube.
  let total_faces := 6 * n^3 in            -- The total number of faces considering all mini-cubes.
  let total_blue_area := total_faces - total_red_area in -- The total blue area.
  total_blue_area / total_red_area = 12 := -- The desired relationship.
by
  sorry

end blue_area_factor_12_l314_314429


namespace janelle_marbles_l314_314016

variable (initial_green : ℕ) (bags : ℕ) (marbles_per_bag : ℕ) (gift_green : ℕ) (gift_blue : ℕ)

def marbles_left (initial_green bags marbles_per_bag gift_green gift_blue : ℕ) : ℕ :=
  initial_green + (bags * marbles_per_bag) - (gift_green + gift_blue)

theorem janelle_marbles : marbles_left 26 6 10 6 8 = 72 := 
by 
  simp [marbles_left]
  sorry

end janelle_marbles_l314_314016


namespace train_length_l314_314129

theorem train_length (time : ℕ) (speed_kmh : ℕ) (conversion_factor : ℚ) (speed_ms : ℚ) (length : ℚ) :
  time = 50 ∧ speed_kmh = 36 ∧ conversion_factor = 5 / 18 ∧ speed_ms = speed_kmh * conversion_factor ∧ length = speed_ms * time →
  length = 500 :=
by
  sorry

end train_length_l314_314129


namespace area_ratio_of_octagons_is_4_l314_314097

-- Define the given conditions
variable (r : ℝ) -- radius of the common circle

def side_length_circumscribed_octagon := √2 * r
def side_length_inscribed_octagon := r / √2

-- Areas of polygons scale with the square of the side length
def area_ratio_circumscribed_to_inscribed := (side_length_circumscribed_octagon r / side_length_inscribed_octagon r)^2

theorem area_ratio_of_octagons_is_4 :
  area_ratio_circumscribed_to_inscribed r = 4 := by
  sorry

end area_ratio_of_octagons_is_4_l314_314097


namespace total_trees_after_planting_l314_314377

-- Define the initial counts of the trees
def initial_maple_trees : ℕ := 2
def initial_poplar_trees : ℕ := 5
def initial_oak_trees : ℕ := 4

-- Define the planting rules
def maple_trees_planted (initial_maple : ℕ) : ℕ := 3 * initial_maple
def poplar_trees_planted (initial_poplar : ℕ) : ℕ := 3 * initial_poplar

-- Calculate the total number of each type of tree after planting
def total_maple_trees (initial_maple : ℕ) : ℕ :=
  initial_maple + maple_trees_planted initial_maple

def total_poplar_trees (initial_poplar : ℕ) : ℕ :=
  initial_poplar + poplar_trees_planted initial_poplar

def total_oak_trees (initial_oak : ℕ) : ℕ := initial_oak

-- Calculate the total number of trees in the park
def total_trees (initial_maple initial_poplar initial_oak : ℕ) : ℕ :=
  total_maple_trees initial_maple + total_poplar_trees initial_poplar + total_oak_trees initial_oak

-- The proof statement
theorem total_trees_after_planting :
  total_trees initial_maple_trees initial_poplar_trees initial_oak_trees = 32 := 
by
  -- Proof placeholder
  sorry

end total_trees_after_planting_l314_314377


namespace real_roots_of_polynomial_l314_314346

theorem real_roots_of_polynomial :
  {x : ℝ | (x^4 - 4*x^3 + 5*x^2 - 2*x + 2) = 0} = {1, -1} :=
sorry

end real_roots_of_polynomial_l314_314346


namespace octagon_area_ratio_l314_314090

theorem octagon_area_ratio (r : ℝ) :
  let A_inscribed := 2 * (1 + Real.sqrt 2) * (r * Real.tan (Real.pi / 8))^2 in
  let R := r * (1 / Real.cos (Real.pi / 8)) in
  let A_circumscribed := 2 * (1 + Real.sqrt 2) * (2 * R * Real.sin (Real.pi / 8))^2 in
  A_circumscribed / A_inscribed = 4 * (3 + 2 * Real.sqrt 2) :=
by
  sorry

end octagon_area_ratio_l314_314090


namespace tan_A_in_triangle_ABC_l314_314004

theorem tan_A_in_triangle_ABC (a b c : ℝ) (A B C : ℝ) (ha : 0 < A) (ha_90 : A < π / 2) 
(hb : b = 3 * a * Real.sin B) : Real.tan A = Real.sqrt 2 / 4 :=
sorry

end tan_A_in_triangle_ABC_l314_314004


namespace total_students_l314_314264

-- Definitions based on conditions
variable (T M Z : ℕ)  -- T for Tina's students, M for Maura's students, Z for Zack's students

-- Conditions as hypotheses
axiom h1 : T = M  -- Tina's classroom has the same amount of students as Maura's
axiom h2 : Z = (T + M) / 2  -- Zack's classroom has half the amount of total students between Tina and Maura's classrooms
axiom h3 : Z = 23  -- There are 23 students in Zack's class when present

-- Proof statement
theorem total_students : T + M + Z = 69 :=
  sorry

end total_students_l314_314264


namespace ratio_of_areas_of_octagons_l314_314124

theorem ratio_of_areas_of_octagons
  (r : ℝ)
  (sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2)
  (sin_67_5 : Real.sin (Real.pi / 180 * 67.5) = Real.sqrt (2 + Real.sqrt 2) / 2)
  : let smaller_octagon_area := 2 * (1 + Real.sqrt 2) * r^2 in
    let larger_octagon_area := 
      2 * (1 + Real.sqrt 2) * (2 * r * (1 + Real.sqrt 2))^2 in
    larger_octagon_area / smaller_octagon_area = 12 + 8 * Real.sqrt 2 :=
by
  sorry

end ratio_of_areas_of_octagons_l314_314124


namespace balance_equation_l314_314400

variable (G Y W B : ℝ)
variable (balance1 : 4 * G = 8 * B)
variable (balance2 : 3 * Y = 7.5 * B)
variable (balance3 : 8 * B = 6 * W)

theorem balance_equation : 5 * G + 3 * Y + 4 * W = 23.5 * B := by
  sorry

end balance_equation_l314_314400


namespace gcd_values_count_l314_314063

theorem gcd_values_count (a b : ℕ) (h : a * b = 360) : 
  ∃ g : ℕ, g ∈ {1, 2, 3, 4, 5, 6, 8, 9, 12, 18}.card := sorry

end gcd_values_count_l314_314063


namespace distance_from_center_to_line_l314_314188

noncomputable def circle_center_is_a (a : ℝ) : Prop :=
  (2 - a)^2 + (1 - a)^2 = a^2

theorem distance_from_center_to_line (a : ℝ) (h : 0 < a) (hab : circle_center_is_a a) :
  (∀x0 y0 : ℝ, (x0, y0) = (a, a) → (|2 * x0 - y0 - 3| / Real.sqrt (2^2 + 1^2)) = (2 * Real.sqrt 5 / 5)) :=
by
  sorry

end distance_from_center_to_line_l314_314188


namespace people_present_l314_314046

-- Number of parents, pupils, teachers, staff members, and volunteers
def num_parents : ℕ := 105
def num_pupils : ℕ := 698
def num_teachers : ℕ := 35
def num_staff_members : ℕ := 20
def num_volunteers : ℕ := 50

-- The total number of people present in the program
def total_people : ℕ := num_parents + num_pupils + num_teachers + num_staff_members + num_volunteers

-- Proof statement
theorem people_present : total_people = 908 := by
  -- Proof goes here, but adding sorry for now
  sorry

end people_present_l314_314046


namespace number_of_males_choosing_malt_l314_314006

-- Definitions of conditions as provided in the problem
def total_males : Nat := 10
def total_females : Nat := 16

def total_cheerleaders : Nat := total_males + total_females

def females_choosing_malt : Nat := 8
def females_choosing_coke : Nat := total_females - females_choosing_malt

noncomputable def cheerleaders_choosing_malt (M_males : Nat) : Nat :=
  females_choosing_malt + M_males

noncomputable def cheerleaders_choosing_coke (M_males : Nat) : Nat :=
  females_choosing_coke + (total_males - M_males)

theorem number_of_males_choosing_malt : ∃ (M_males : Nat), 
  cheerleaders_choosing_malt M_males = 2 * cheerleaders_choosing_coke M_males ∧
  cheerleaders_choosing_malt M_males + cheerleaders_choosing_coke M_males = total_cheerleaders ∧
  M_males = 9 := 
by
  sorry

end number_of_males_choosing_malt_l314_314006


namespace oilProductionPerPerson_west_oilProductionPerPerson_nonwest_oilProductionPerPerson_russia_l314_314207

-- Definitions for oil consumption per person
def oilConsumptionWest : ℝ := 55.084
def oilConsumptionNonWest : ℝ := 214.59
def oilConsumptionRussia : ℝ := 1038.33

-- Lean statements
theorem oilProductionPerPerson_west : oilConsumptionWest = 55.084 := by
  sorry

theorem oilProductionPerPerson_nonwest : oilConsumptionNonWest = 214.59 := by
  sorry

theorem oilProductionPerPerson_russia : oilConsumptionRussia = 1038.33 := by
  sorry

end oilProductionPerPerson_west_oilProductionPerPerson_nonwest_oilProductionPerPerson_russia_l314_314207


namespace total_number_of_values_l314_314259

theorem total_number_of_values (S n : ℕ) (h1 : (S - 165 + 135) / n = 150) (h2 : S / n = 151) : n = 30 :=
by {
  sorry
}

end total_number_of_values_l314_314259


namespace compartments_count_l314_314237

-- Definition of initial pennies per compartment
def initial_pennies_per_compartment : ℕ := 2

-- Definition of additional pennies added to each compartment
def additional_pennies_per_compartment : ℕ := 6

-- Definition of total pennies is 96
def total_pennies : ℕ := 96

-- Prove the number of compartments is 12
theorem compartments_count (c : ℕ) 
  (h1 : initial_pennies_per_compartment + additional_pennies_per_compartment = 8)
  (h2 : 8 * c = total_pennies) : 
  c = 12 :=
by
  sorry

end compartments_count_l314_314237


namespace car_average_speed_is_correct_l314_314297

noncomputable def average_speed_of_car : ℝ :=
  let d1 := 30
  let s1 := 30
  let d2 := 35
  let s2 := 55
  let t3 := 0.5
  let s3 := 70
  let t4 := 40 / 60 -- 40 minutes converted to hours
  let s4 := 36
  let t1 := d1 / s1
  let t2 := d2 / s2
  let d3 := s3 * t3
  let d4 := s4 * t4
  let total_distance := d1 + d2 + d3 + d4
  let total_time := t1 + t2 + t3 + t4
  total_distance / total_time

theorem car_average_speed_is_correct :
  average_speed_of_car = 44.238 := 
sorry

end car_average_speed_is_correct_l314_314297


namespace determine_asymptotes_l314_314351

noncomputable def hyperbola_eccentricity_asymptote_relation (a b : ℝ) (e : ℝ) (k : ℝ) :=
  a > 0 ∧ b > 0 ∧ (e = Real.sqrt 2 * |k|) ∧ (k = b / a)

theorem determine_asymptotes (a b : ℝ) (h : hyperbola_eccentricity_asymptote_relation a b (Real.sqrt (a^2 + b^2) / a) (b / a)) :
  true := sorry

end determine_asymptotes_l314_314351


namespace ratio_of_area_of_inscribed_and_circumscribed_octagons_l314_314113

noncomputable def area_ratio_of_octagons (r : ℝ) : ℝ :=
  let cos22_5 := (√(2 + √2) / 2)
  let r_prime := r / cos22_5
  let area_ratio := (r_prime / r)^2
  area_ratio

theorem ratio_of_area_of_inscribed_and_circumscribed_octagons (r : ℝ) :
  area_ratio_of_octagons r = (4 - 2 * √2) / 2 := by
  sorry

end ratio_of_area_of_inscribed_and_circumscribed_octagons_l314_314113


namespace ratio_of_areas_of_octagons_l314_314123

theorem ratio_of_areas_of_octagons
  (r : ℝ)
  (sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2)
  (sin_67_5 : Real.sin (Real.pi / 180 * 67.5) = Real.sqrt (2 + Real.sqrt 2) / 2)
  : let smaller_octagon_area := 2 * (1 + Real.sqrt 2) * r^2 in
    let larger_octagon_area := 
      2 * (1 + Real.sqrt 2) * (2 * r * (1 + Real.sqrt 2))^2 in
    larger_octagon_area / smaller_octagon_area = 12 + 8 * Real.sqrt 2 :=
by
  sorry

end ratio_of_areas_of_octagons_l314_314123


namespace incorrect_statement_A_l314_314379

-- Definitions based on conditions
variable {α : Plane}
variable {m n : Line}

-- Option A conditions
def m_parallel_alpha (m : Line) (α : Plane) : Prop := m ∥ α
def m_not_parallel_n (m n : Line) : Prop := ¬ (m ∥ n)

-- Option A conclusion to disprove
def statement_A (m n : Line) (α : Plane) [m_parallel_alpha m α] [m_not_parallel_n m n] : Prop :=
  ¬(n ∥ α)

theorem incorrect_statement_A (α : Plane) (m n : Line) (h1 : m_parallel_alpha m α) (h2 : m_not_parallel_n m n) : ¬(statement_A m n α) := by
  sorry

end incorrect_statement_A_l314_314379


namespace ratio_of_areas_of_octagons_l314_314095

theorem ratio_of_areas_of_octagons (r : ℝ) :
  let side_length_inscribed := r
  let side_length_circumscribed := r * real.sec (real.pi / 8)
  let area_inscribed := 2 * (1 + real.sqrt 2) * side_length_inscribed ^ 2
  let area_circumscribed := 2 * (1 + real.sqrt 2) * side_length_circumscribed ^ 2
  area_circumscribed / area_inscribed = 4 - 2 * real.sqrt 2 :=
sorry

end ratio_of_areas_of_octagons_l314_314095


namespace ranking_of_anna_bella_carol_l314_314330

-- Define three people and their scores
variables (Anna Bella Carol : ℕ)

-- Define conditions based on problem statements
axiom Anna_not_highest : ∃ x : ℕ, x > Anna
axiom Bella_not_lowest : ∃ x : ℕ, x < Bella
axiom Bella_higher_than_Carol : Bella > Carol

-- The theorem to be proven
theorem ranking_of_anna_bella_carol (h : Anna < Bella ∧ Carol < Anna) :
  (Bella > Anna ∧ Anna > Carol) :=
by sorry

end ranking_of_anna_bella_carol_l314_314330


namespace average_weight_of_abc_l314_314412

theorem average_weight_of_abc 
  (A B C : ℝ) 
  (h1 : (A + B) / 2 = 40)
  (h2 : (B + C) / 2 = 46)
  (h3 : B = 37) :
  (A + B + C) / 3 = 45 := 
by
  sorry

end average_weight_of_abc_l314_314412


namespace brendan_fish_caught_afternoon_l314_314132

theorem brendan_fish_caught_afternoon (morning_fish : ℕ) (thrown_fish : ℕ) (dads_fish : ℕ) (total_fish : ℕ) :
  morning_fish = 8 → thrown_fish = 3 → dads_fish = 13 → total_fish = 23 → 
  (morning_fish - thrown_fish) + dads_fish + brendan_afternoon_catch = total_fish → 
  brendan_afternoon_catch = 5 :=
by
  intros morning_fish_eq thrown_fish_eq dads_fish_eq total_fish_eq fish_sum_eq
  sorry

end brendan_fish_caught_afternoon_l314_314132


namespace ratio_of_area_to_breadth_l314_314030

variable (l b : ℕ)

theorem ratio_of_area_to_breadth 
  (h1 : b = 14) 
  (h2 : l - b = 10) : 
  (l * b) / b = 24 := by
  sorry

end ratio_of_area_to_breadth_l314_314030


namespace circle_distance_to_line_l314_314195

-- Define the conditions
def circle_pass_through (P : ℝ × ℝ) (a : ℝ) : Prop :=
  let (x, y) := P
  (x - a)^2 + (y - a)^2 = a^2

def is_tangent_to_axes (a : ℝ) : Prop :=
  a > 0

def distance_to_line (center : ℝ × ℝ) (L : ℝ × ℝ → ℝ) : ℝ :=
  let (x₁, y₁) := center
  abs (L (x₁, y₁)) / sqrt ((L (1,0))^2 + (L (0,1))^2)

-- Define the line equation as a function
def line_eq (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  2*x - y - 3

-- Define the main proof goal
theorem circle_distance_to_line (a : ℝ) (h1 : circle_pass_through (2, 1) a) (h2 : is_tangent_to_axes a) :
  distance_to_line (a, a) line_eq = 2*sqrt 5 / 5 :=
sorry

end circle_distance_to_line_l314_314195


namespace gcd_lcm_product_360_l314_314051

theorem gcd_lcm_product_360 :
  ∃ (d : ℕ), (∃ x y : ℕ, x * y = 360 ∧ d = Nat.gcd x y) ∧ 
  {d | ∃ x y : ℕ, x * y = 360 ∧ d = Nat.gcd x y}.to_finset.card = 8 := 
sorry

end gcd_lcm_product_360_l314_314051


namespace distance_between_points_l314_314275

theorem distance_between_points :
  let x1 := 2
  let y1 := -2
  let x2 := 8
  let y2 := 8
  let dist := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  dist = Real.sqrt 136 :=
by
  -- Proof to be filled in here.
  sorry

end distance_between_points_l314_314275


namespace blue_area_factor_12_l314_314428

theorem blue_area_factor_12 : 
  let n := 13 in
  let total_red_area := 6 * n^2 in         -- The total red area on the surface of the original cube.
  let total_faces := 6 * n^3 in            -- The total number of faces considering all mini-cubes.
  let total_blue_area := total_faces - total_red_area in -- The total blue area.
  total_blue_area / total_red_area = 12 := -- The desired relationship.
by
  sorry

end blue_area_factor_12_l314_314428


namespace total_students_l314_314263

-- Definitions based on conditions
variable (T M Z : ℕ)  -- T for Tina's students, M for Maura's students, Z for Zack's students

-- Conditions as hypotheses
axiom h1 : T = M  -- Tina's classroom has the same amount of students as Maura's
axiom h2 : Z = (T + M) / 2  -- Zack's classroom has half the amount of total students between Tina and Maura's classrooms
axiom h3 : Z = 23  -- There are 23 students in Zack's class when present

-- Proof statement
theorem total_students : T + M + Z = 69 :=
  sorry

end total_students_l314_314263


namespace gcd_possible_values_count_l314_314058

theorem gcd_possible_values_count (a b : ℕ) (h : a * b = 360) : 
  {d : ℕ | d ∣ a ∧ d ∣ b}.card = 6 :=
sorry

end gcd_possible_values_count_l314_314058


namespace sum_of_roots_l314_314050

-- Define the polynomial
def poly : Polynomial ℝ := 3 * Polynomial.X^3 + 7 * Polynomial.X^2 - 6 * Polynomial.X - 10

-- The goal is to prove that the sum of the roots is equal to -7/3
theorem sum_of_roots : Σ p : Polynomial ℝ, p = poly ∧ p.root_sum p.splits = -7/3 :=
begin
  use poly,
  split,
  {
    -- First, we confirm poly is equal to the given polynomial
    refl,
  },
  {
    -- Then, we prove that the sum of its roots equals -7/3
    sorry,  -- We skip the proof details
  }
end

end sum_of_roots_l314_314050


namespace speed_of_current_l314_314305

variable (m c : ℝ)

theorem speed_of_current (h1 : m + c = 15) (h2 : m - c = 10) : c = 2.5 :=
sorry

end speed_of_current_l314_314305


namespace students_behind_Yoongi_l314_314293

theorem students_behind_Yoongi 
  (total_students : ℕ) 
  (position_Jungkook : ℕ) 
  (students_between : ℕ) 
  (position_Yoongi : ℕ) : 
  total_students = 20 → 
  position_Jungkook = 1 → 
  students_between = 5 → 
  position_Yoongi = position_Jungkook + students_between + 1 → 
  (total_students - position_Yoongi) = 13 :=
by
  sorry

end students_behind_Yoongi_l314_314293


namespace mia_weight_l314_314256

theorem mia_weight (a m : ℝ) (h1 : a + m = 220) (h2 : m - a = 2 * a) : m = 165 :=
sorry

end mia_weight_l314_314256


namespace part_one_part_two_l314_314362

noncomputable def f (x : ℝ) (a : ℝ) := x^2 + a * x + 6

theorem part_one (x : ℝ) : ∀ a, a = 5 → f x a < 0 ↔ -3 < x ∧ x < -2 :=
by
  sorry

theorem part_two : ∀ a, (∀ x, f x a > 0) ↔ - 2 * Real.sqrt 6 < a ∧ a < 2 * Real.sqrt 6 :=
by
  sorry

end part_one_part_two_l314_314362


namespace blue_face_area_factor_l314_314440

def cube_side_length : ℕ := 13

def original_red_face_area : ℕ := 6 * cube_side_length ^ 2
def total_mini_cube_face_area : ℕ := 6 * cube_side_length ^ 3
def blue_face_area : ℕ := total_mini_cube_face_area - original_red_face_area

theorem blue_face_area_factor :
  (blue_face_area / original_red_face_area) = 12 :=
by
  sorry

end blue_face_area_factor_l314_314440


namespace vet_appointments_cost_l314_314218

variable (x : ℝ)

def JohnVetAppointments (x : ℝ) : Prop := 
  (x + 0.20 * x + 0.20 * x + 100 = 660)

theorem vet_appointments_cost :
  (∃ x : ℝ, JohnVetAppointments x) → x = 400 :=
by
  intro h
  obtain ⟨x, hx⟩ := h
  simp [JohnVetAppointments] at hx
  sorry

end vet_appointments_cost_l314_314218


namespace gcd_possible_values_count_l314_314059

theorem gcd_possible_values_count : ∃ a b : ℕ, a * b = 360 ∧ (∃ gcds : Finset ℕ, gcds = {d | ∃ a b : ℕ, a * b = 360 ∧ d = Nat.gcd a b} ∧ gcds.card = 6) :=
sorry

end gcd_possible_values_count_l314_314059


namespace find_certain_number_l314_314370

theorem find_certain_number (a : ℤ) (certain_number : ℤ) (h₁ : a = 105) (h₂ : a^3 = 21 * 25 * 45 * certain_number) : certain_number = 49 := 
sorry

end find_certain_number_l314_314370


namespace circle_center_line_distance_l314_314197

noncomputable def distance_point_to_line (x₁ y₁ A B C : ℝ) : ℝ :=
|A * x₁ + B * y₁ + C| / Real.sqrt (A^2 + B^2)

theorem circle_center_line_distance (a : ℝ) (h : a^2 - 6 * a + 5 = 0) :
  distance_point_to_line a a 2 (-1) (-3) = (2 * Real.sqrt 5 / 5) :=
by
  sorry

end circle_center_line_distance_l314_314197


namespace red_blue_area_ratio_is_12_l314_314430

def red_blue_face_area_ratio : ℕ :=
  let big_cube_dim : ℕ := 13 in
  let total_red_faces : ℕ := 6 * big_cube_dim^2 in
  let total_mini_cube_faces : ℕ := 6 * big_cube_dim^3 in
  let total_blue_faces : ℕ := total_mini_cube_faces - total_red_faces in
  total_blue_faces / total_red_faces

theorem red_blue_area_ratio_is_12 :
  red_blue_face_area_ratio = 12 := by
  sorry

end red_blue_area_ratio_is_12_l314_314430


namespace initial_pencils_correct_l314_314457

variable (pencils_taken remaining_pencils initial_pencils : ℕ)

def initial_number_of_pencils (pencils_taken remaining_pencils : ℕ) : ℕ :=
  pencils_taken + remaining_pencils

theorem initial_pencils_correct (h₁ : pencils_taken = 22) (h₂ : remaining_pencils = 12) :
  initial_number_of_pencils pencils_taken remaining_pencils = 34 := by
  rw [h₁, h₂]
  rfl

end initial_pencils_correct_l314_314457


namespace constant_ratio_arithmetic_progressions_l314_314220

theorem constant_ratio_arithmetic_progressions
  (a : ℕ → ℝ) (b : ℕ → ℝ) (d p a1 b1 : ℝ)
  (h_a : ∀ k : ℕ, a (k + 1) = a1 + k * d)
  (h_b : ∀ k : ℕ, b (k + 1) = b1 + k * p)
  (h_pos : ∀ k : ℕ, a (k + 1) > 0 ∧ b (k + 1) > 0)
  (h_int : ∀ k : ℕ, ∃ n : ℤ, (a (k + 1) / b (k + 1)) = n) :
  ∃ r : ℝ, ∀ k : ℕ, (a (k + 1) / b (k + 1)) = r :=
by
  sorry

end constant_ratio_arithmetic_progressions_l314_314220


namespace ferry_speed_difference_l314_314155

theorem ferry_speed_difference :
  let V_p := 6
  let Time_P := 3
  let Distance_P := V_p * Time_P
  let Distance_Q := 2 * Distance_P
  let Time_Q := Time_P + 1
  let V_q := Distance_Q / Time_Q
  V_q - V_p = 3 := by
  sorry

end ferry_speed_difference_l314_314155


namespace parallel_planes_l314_314364

variables {Point Line Plane : Type}
variables (a : Line) (α β : Plane)

-- Conditions
def line_perpendicular_plane (l: Line) (p: Plane) : Prop := sorry
def planes_parallel (p₁ p₂: Plane) : Prop := sorry

-- Problem statement
theorem parallel_planes (h1: line_perpendicular_plane a α) 
                        (h2: line_perpendicular_plane a β) : 
                        planes_parallel α β :=
sorry

end parallel_planes_l314_314364


namespace simplify_fraction_rationalize_denominator_l314_314241

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x
noncomputable def fraction := 5 / (sqrt 125 + 3 * sqrt 45 + 4 * sqrt 20 + sqrt 75)

theorem simplify_fraction_rationalize_denominator :
  fraction = sqrt 5 / 27 :=
by
  sorry

end simplify_fraction_rationalize_denominator_l314_314241


namespace blue_red_area_ratio_l314_314419

theorem blue_red_area_ratio (n : ℕ) (h : n = 13) :
    let total_red_faces := 6 * n^2,
        total_faces := 6 * n^3,
        blue_faces := total_faces - total_red_faces
    in (blue_faces / total_red_faces : ℕ) = 12 :=
  by
  sorry

end blue_red_area_ratio_l314_314419


namespace probability_of_winning_position_l314_314313

-- Given Conditions
def tic_tac_toe_board : Type := Fin 3 × Fin 3
def is_nought (b : tic_tac_toe_board → bool) (p : tic_tac_toe_board) : Prop := b p
def is_cross (b : tic_tac_toe_board → bool) (p : tic_tac_toe_board) : Prop := ¬ b p

-- Winning positions in Tic-Tac-Toe are vertical, horizontal, or diagonal lines
def is_winning_position (b : tic_tac_toe_board → bool) : Prop :=
  (∃ i : Fin 3, ∀ j : Fin 3, is_nought b (i, j)) ∨ -- Row
  (∃ j : Fin 3, ∀ i : Fin 3, is_nought b (i, j)) ∨ -- Column
  (∀ i : Fin 3, is_nought b (i, i)) ∨ -- Main diagonal
  (∀ i : Fin 3, is_nought b (i, Fin.mk (2 - i.1) (by simp [i.1]))) -- Anti-diagonal

-- Problem Statement
theorem probability_of_winning_position :
  let total_positions := Nat.choose 9 3
  let winning_positions := 8
  winning_positions / total_positions = (2:ℚ) / 21 :=
by sorry

end probability_of_winning_position_l314_314313


namespace ratio_of_octagon_areas_l314_314101

-- Define the relevant terms and conditions
noncomputable def radius_of_circle (r : ℝ) := r
noncomputable def side_length_inscribed_octagon (r : ℝ) := r * Real.sqrt 2
noncomputable def side_length_circumscribed_octagon (r : ℝ) := r * Real.sqrt 2

-- Area of a regular octagon in terms of its side length
noncomputable def area_regular_octagon (s : ℝ) := 2 * (1 + Real.sqrt 2) * s^2

-- Areas of inscribed and circumscribed octagons
noncomputable def area_inscribed_octagon (r : ℝ) := area_regular_octagon (side_length_inscribed_octagon r)
noncomputable def area_circumscribed_octagon (r : ℝ) := area_regular_octagon (side_length_circumscribed_octagon r)

-- The main statement to prove
theorem ratio_of_octagon_areas (r : ℝ) : 
  area_circumscribed_octagon r / area_inscribed_octagon r = 1 :=
by
  sorry

end ratio_of_octagon_areas_l314_314101


namespace greg_age_is_16_l314_314138

-- Definitions based on given conditions
def cindy_age : ℕ := 5
def jan_age : ℕ := cindy_age + 2
def marcia_age : ℕ := 2 * jan_age
def greg_age : ℕ := marcia_age + 2

-- Theorem stating that Greg's age is 16 years given the above conditions
theorem greg_age_is_16 : greg_age = 16 := by
  sorry

end greg_age_is_16_l314_314138


namespace queenie_overtime_hours_l314_314404

theorem queenie_overtime_hours:
  ∀ (daily_pay overtime_pay total_pay days hours:int),
    daily_pay = 150 -> 
    overtime_pay = 5 -> 
    total_pay = 770 -> 
    days = 5 -> 
    (total_pay - (daily_pay * days)) / overtime_pay = 4 :=
by 
  intros daily_pay overtime_pay total_pay days hours
  assume h1: daily_pay = 150
  assume h2: overtime_pay = 5
  assume h3: total_pay = 770
  assume h4: days = 5
  calc 
    (total_pay - (daily_pay * days)) / overtime_pay 
        = (770 - (150 * 5)) / 5 : by rw [h1, h2, h3, h4]
    ... = 4                        : by norm_num

end queenie_overtime_hours_l314_314404


namespace alcohol_to_water_ratio_l314_314474

theorem alcohol_to_water_ratio (alcohol water : ℚ) (h_alcohol : alcohol = 2/7) (h_water : water = 3/7) : alcohol / water = 2 / 3 := by
  sorry

end alcohol_to_water_ratio_l314_314474


namespace no_factors_multiple_of_210_l314_314178

theorem no_factors_multiple_of_210 (n : ℕ) (h : n = 2^12 * 3^18 * 5^10) : ∀ d : ℕ, d ∣ n → ¬ (210 ∣ d) :=
by
  sorry

end no_factors_multiple_of_210_l314_314178


namespace arithmetic_geometric_mean_l314_314235

theorem arithmetic_geometric_mean (a b : ℝ) (h1 : a + b = 48) (h2 : a * b = 440) : a^2 + b^2 = 1424 := 
by 
  -- Proof goes here
  sorry

end arithmetic_geometric_mean_l314_314235


namespace max_sin_A_plus_sin_C_l314_314163

variables {a b c S : ℝ}
variables {A B C : ℝ}

-- Assume the sides of the triangle
variables (ha : a > 0) (hb : b > 0) (hc : c > 0)

-- Assume the angles of the triangle
variables (hA : A > 0) (hB : B > (Real.pi / 2)) (hC : C > 0)
variables (hSumAngles : A + B + C = Real.pi)

-- Assume the relationship between the area and the sides
variables (hArea : S = (1/2) * a * c * Real.sin B)

-- Assume the given equation holds
variables (hEquation : 4 * b * S = a * (b^2 + c^2 - a^2))

-- The statement to prove
theorem max_sin_A_plus_sin_C : (Real.sin A + Real.sin C) ≤ 9 / 8 :=
sorry

end max_sin_A_plus_sin_C_l314_314163


namespace ounces_per_bowl_l314_314344

theorem ounces_per_bowl (oz_per_gallon : ℕ) (gallons : ℕ) (bowls_per_minute : ℕ) (minutes : ℕ) (total_ounces : ℕ) (total_bowls : ℕ) (oz_per_bowl : ℕ) : 
  oz_per_gallon = 128 → 
  gallons = 6 →
  bowls_per_minute = 5 →
  minutes = 15 →
  total_ounces = oz_per_gallon * gallons →
  total_bowls = bowls_per_minute * minutes →
  oz_per_bowl = total_ounces / total_bowls →
  round (oz_per_bowl : ℚ) = 10 :=
by
  sorry

end ounces_per_bowl_l314_314344


namespace circle_center_line_distance_l314_314199

noncomputable def distance_point_to_line (x₁ y₁ A B C : ℝ) : ℝ :=
|A * x₁ + B * y₁ + C| / Real.sqrt (A^2 + B^2)

theorem circle_center_line_distance (a : ℝ) (h : a^2 - 6 * a + 5 = 0) :
  distance_point_to_line a a 2 (-1) (-3) = (2 * Real.sqrt 5 / 5) :=
by
  sorry

end circle_center_line_distance_l314_314199


namespace cider_apples_production_l314_314299

def apples_total : Real := 8.0
def baking_fraction : Real := 0.30
def cider_fraction : Real := 0.60

def apples_remaining : Real := apples_total * (1 - baking_fraction)
def apples_for_cider : Real := apples_remaining * cider_fraction

theorem cider_apples_production : 
    apples_for_cider = 3.4 := 
by
  sorry

end cider_apples_production_l314_314299


namespace relay_race_orders_l314_314139

open Finset

def athletes : finset (ℕ × ℕ × ℕ × ℕ) :=
  univ.filter (λ ⟨a, b, c, d⟩, a ≠ 1 ∧ b ≠ 2 ∧ c ≠ 3)

theorem relay_race_orders : athletes.card = 11 := by
  sorry

end relay_race_orders_l314_314139


namespace net_increase_in_wealth_l314_314296

-- Definitions for yearly changes and fees
def firstYearChange (initialAmt : ℝ) : ℝ := initialAmt * 1.75 - 0.02 * initialAmt * 1.75
def secondYearChange (amt : ℝ) : ℝ := amt * 0.7 - 0.02 * amt * 0.7
def thirdYearChange (amt : ℝ) : ℝ := amt * 1.45 - 0.02 * amt * 1.45
def fourthYearChange (amt : ℝ) : ℝ := amt * 0.85 - 0.02 * amt * 0.85

-- Total Value after 4th year accounting all changes and fees
def totalAfterFourYears (initialAmt : ℝ) : ℝ :=
  let afterFirstYear := firstYearChange initialAmt
  let afterSecondYear := secondYearChange afterFirstYear
  let afterThirdYear := thirdYearChange afterSecondYear
  fourthYearChange afterThirdYear

-- Capital gains tax calculation
def capitalGainsTax (initialAmt finalAmt : ℝ) : ℝ :=
  0.20 * (finalAmt - initialAmt)

-- Net value after taxes
def netValueAfterTaxes (initialAmt : ℝ) : ℝ :=
  let total := totalAfterFourYears initialAmt
  total - capitalGainsTax initialAmt total

-- Main theorem statement
theorem net_increase_in_wealth :
  ∀ (initialAmt : ℝ), netValueAfterTaxes initialAmt = initialAmt * 1.31408238206 := sorry

end net_increase_in_wealth_l314_314296


namespace fraction_difference_l314_314403

theorem fraction_difference (a b : ℝ) : 
  (a / (a + 1)) - (b / (b + 1)) = (a - b) / ((a + 1) * (b + 1)) :=
sorry

end fraction_difference_l314_314403


namespace circle_area_of_white_cube_l314_314229

/-- 
Marla has a large white cube with an edge length of 12 feet and enough green paint to cover 432 square feet.
Marla paints a white circle centered on each face of the cube, surrounded by a green border.
Prove the area of one of the white circles is 72 square feet.
 -/
theorem circle_area_of_white_cube
  (edge_length : ℝ) (paint_area : ℝ) (faces : ℕ)
  (h_edge_length : edge_length = 12)
  (h_paint_area : paint_area = 432)
  (h_faces : faces = 6) :
  ∃ (circle_area : ℝ), circle_area = 72 :=
by
  sorry

end circle_area_of_white_cube_l314_314229


namespace dave_diner_total_cost_l314_314131

theorem dave_diner_total_cost (burger_count : ℕ) (fries_count : ℕ)
  (burger_cost : ℕ) (fries_cost : ℕ)
  (discount_threshold : ℕ) (discount_amount : ℕ)
  (h1 : burger_count >= discount_threshold) :
  burger_count = 6 → fries_count = 5 → burger_cost = 4 → fries_cost = 3 →
  discount_threshold = 4 → discount_amount = 2 →
  (burger_count * (burger_cost - discount_amount) + fries_count * fries_cost) = 27 :=
by
  intros hbc hfc hbcost hfcs dth da
  sorry

end dave_diner_total_cost_l314_314131


namespace sum_of_possible_values_of_g_l314_314223

def f (x : ℝ) : ℝ := x^2 - 9 * x + 20
def g (x : ℝ) : ℝ := 3 * x - 4

theorem sum_of_possible_values_of_g :
  let x1 := (9 + 3 * Real.sqrt 5) / 2
  let x2 := (9 - 3 * Real.sqrt 5) / 2
  g x1 + g x2 = 19 :=
by
  sorry

end sum_of_possible_values_of_g_l314_314223


namespace area_ratio_of_octagons_is_4_l314_314100

-- Define the given conditions
variable (r : ℝ) -- radius of the common circle

def side_length_circumscribed_octagon := √2 * r
def side_length_inscribed_octagon := r / √2

-- Areas of polygons scale with the square of the side length
def area_ratio_circumscribed_to_inscribed := (side_length_circumscribed_octagon r / side_length_inscribed_octagon r)^2

theorem area_ratio_of_octagons_is_4 :
  area_ratio_circumscribed_to_inscribed r = 4 := by
  sorry

end area_ratio_of_octagons_is_4_l314_314100


namespace find_a_minus_b_l314_314360

theorem find_a_minus_b (a b : ℝ) (h1: ∀ x : ℝ, (ax^2 + bx - 2 = 0 → x = -2 ∨ x = -1/4)) : (a - b = 5) :=
sorry

end find_a_minus_b_l314_314360


namespace Q_not_invertible_l314_314221

-- Define the vector v
def v : ℝ × ℝ := (4, -5)

-- Define the unit vector u in the direction of v
def v_length : ℝ := Real.sqrt (4 ^ 2 + (-5) ^ 2)
def u : ℝ × ℝ := (4 / v_length, -5 / v_length)

-- Define the projection matrix Q
def Q : Matrix (Fin 2) (Fin 2) ℝ :=
  let uuT := Matrix.mulVecLin u (Matrix.vecLin u) in
  (1 / (41 : ℝ)) • uuT

-- Determine if Q is invertible
theorem Q_not_invertible : det Q = 0 :=
  sorry

end Q_not_invertible_l314_314221


namespace total_students_in_classrooms_l314_314262

theorem total_students_in_classrooms (tina_students maura_students zack_students : ℕ) 
    (h1 : tina_students = maura_students)
    (h2 : zack_students = (tina_students + maura_students) / 2)
    (h3 : 22 + 1 = zack_students) : 
    tina_students + maura_students + zack_students = 69 := 
by 
  -- Proof steps would go here, but we include 'sorry' as per the instructions.
  sorry

end total_students_in_classrooms_l314_314262


namespace average_production_last_5_days_l314_314005

theorem average_production_last_5_days (tv_per_day_25 : ℕ) (total_tv_30 : ℕ) :
  tv_per_day_25 = 63 →
  total_tv_30 = 58 * 30 →
  (total_tv_30 - tv_per_day_25 * 25) / 5 = 33 :=
by
  intros h1 h2
  sorry

end average_production_last_5_days_l314_314005


namespace ott_fraction_is_3_over_13_l314_314395

-- Defining the types and quantities involved
noncomputable def moes_original_money (amount_given: ℚ) := amount_given * 5
noncomputable def lokis_original_money (amount_given: ℚ) := amount_given * 5
noncomputable def nicks_original_money (amount_given: ℚ) := amount_given * 3

-- Total original money of the group (excluding Ott)
noncomputable def total_original_money (amount_given: ℚ) :=
  moes_original_money amount_given + lokis_original_money amount_given + nicks_original_money amount_given

-- Total money received by Ott
noncomputable def otts_received_money (amount_given: ℚ) := 3 * amount_given

-- Fraction of the group's total money Ott now has
noncomputable def otts_fraction_of_total_money (amount_given: ℚ) : ℚ :=
  otts_received_money amount_given / total_original_money amount_given

-- The theorem to be proved
theorem ott_fraction_is_3_over_13 :
  otts_fraction_of_total_money 1 = 3 / 13 :=
by
  -- The body of the proof is skipped with sorry
  sorry

end ott_fraction_is_3_over_13_l314_314395


namespace green_balloons_correct_l314_314321

-- Defining the quantities
def total_balloons : ℕ := 67
def red_balloons : ℕ := 29
def blue_balloons : ℕ := 21

-- Calculating the green balloons
def green_balloons : ℕ := total_balloons - red_balloons - blue_balloons

-- The theorem we want to prove
theorem green_balloons_correct : green_balloons = 17 :=
by
  -- proof goes here
  sorry

end green_balloons_correct_l314_314321


namespace ratio_area_octagons_correct_l314_314107

noncomputable def ratio_area_octagons (r : ℝ) : ℝ :=
  let s := 2 * r * Real.sin (Real.pi / 8)   -- side length of inscribed octagon
  let S := 2 * r * Real.cos (Real.pi / 8)   -- side length of circumscribed octagon
  let ratio_side_lengths := S / s
  let ratio_areas := (ratio_side_lengths)^2
  ratio_areas

theorem ratio_area_octagons_correct (r : ℝ) :
  ratio_area_octagons r = 6 + 4 * Real.sqrt 2 :=
by
  sorry

end ratio_area_octagons_correct_l314_314107


namespace ln_binom_le_sum_floor_log_exists_c_for_pi_l314_314291

open BigOperators

-- Theorem 1: Show that
-- \ln \left(\binom{2 n}{n}\right) \leq 
-- \sum_{\substack{p \text{ prime} \\ p \leq 2 n}}\left\lfloor\frac{\ln (2 n)}{\ln p}\right\rfloor \ln p
theorem ln_binom_le_sum_floor_log (n : ℕ) :
  Real.log (Nat.choose (2 * n) n) ≤ ∑ p in Finset.filter Nat.prime (Finset.range (2 * n + 1)), 
    (Real.log (2 * n) / Real.log p).floor * Real.log p :=
by
  sorry

-- Theorem 2: Using the first theorem,
-- show that there exists a constant c > 0 such that for any real x,
-- \pi(x) \geq c \frac{x}{\ln x}
theorem exists_c_for_pi (c : ℝ) (hc : 0 < c) : ∃ c > 0, ∀ (x : ℝ), 
  0 < x → Nat.PrimeCounting.pi x ≥ c * x / Real.log x :=
by
  use c
  split
  · sorry
  · intro x hx
    rw [Nat.PrimeCounting.pi_le_iff]
    apply le_of_eq
    sorry

end ln_binom_le_sum_floor_log_exists_c_for_pi_l314_314291


namespace ratio_of_areas_of_octagons_l314_314096

theorem ratio_of_areas_of_octagons (r : ℝ) :
  let side_length_inscribed := r
  let side_length_circumscribed := r * real.sec (real.pi / 8)
  let area_inscribed := 2 * (1 + real.sqrt 2) * side_length_inscribed ^ 2
  let area_circumscribed := 2 * (1 + real.sqrt 2) * side_length_circumscribed ^ 2
  area_circumscribed / area_inscribed = 4 - 2 * real.sqrt 2 :=
sorry

end ratio_of_areas_of_octagons_l314_314096


namespace age_of_15th_student_l314_314246

theorem age_of_15th_student (avg_age_15_students avg_age_5_students avg_age_9_students : ℕ)
  (total_students total_age_15_students total_age_5_students total_age_9_students : ℕ)
  (h1 : total_students = 15)
  (h2 : avg_age_15_students = 15)
  (h3 : avg_age_5_students = 14)
  (h4 : avg_age_9_students = 16)
  (h5 : total_age_15_students = total_students * avg_age_15_students)
  (h6 : total_age_5_students = 5 * avg_age_5_students)
  (h7 : total_age_9_students = 9 * avg_age_9_students):
  total_age_15_students = total_age_5_students + total_age_9_students + 11 :=
by
  sorry

end age_of_15th_student_l314_314246


namespace mark_profit_l314_314228

def initialPrice : ℝ := 100
def finalPrice : ℝ := 3 * initialPrice
def salesTax : ℝ := 0.05 * initialPrice
def totalInitialCost : ℝ := initialPrice + salesTax
def transactionFee : ℝ := 0.03 * finalPrice
def profitBeforeTax : ℝ := finalPrice - totalInitialCost
def capitalGainsTax : ℝ := 0.15 * profitBeforeTax
def totalProfit : ℝ := profitBeforeTax - transactionFee - capitalGainsTax

theorem mark_profit : totalProfit = 147.75 := sorry

end mark_profit_l314_314228


namespace treadmill_time_saved_l314_314397

theorem treadmill_time_saved:
  let monday_speed := 6
  let tuesday_speed := 4
  let wednesday_speed := 5
  let thursday_speed := 6
  let friday_speed := 3
  let distance := 3 
  let daily_times : List ℚ := 
    [distance/monday_speed, distance/tuesday_speed, distance/wednesday_speed, distance/thursday_speed, distance/friday_speed]
  let total_time := (daily_times.map (λ t => t)).sum
  let total_distance := 5 * distance 
  let uniform_speed := 5 
  let uniform_time := total_distance / uniform_speed 
  let time_difference := total_time - uniform_time 
  let time_in_minutes := time_difference * 60 
  time_in_minutes = 21 := 
by 
  sorry

end treadmill_time_saved_l314_314397


namespace solve_for_diamond_l314_314409

theorem solve_for_diamond (d : ℤ) (h : d * 9 + 5 = d * 10 + 2) : d = 3 :=
by
  sorry

end solve_for_diamond_l314_314409


namespace inequality_cannot_hold_l314_314367

theorem inequality_cannot_hold (a b : ℝ) (h : a < b ∧ b < 0) : ¬ (1 / (a - b) > 1 / a) :=
by {
  sorry
}

end inequality_cannot_hold_l314_314367


namespace octagon_area_ratio_l314_314092

theorem octagon_area_ratio (r : ℝ) :
  let A_inscribed := 2 * (1 + Real.sqrt 2) * (r * Real.tan (Real.pi / 8))^2 in
  let R := r * (1 / Real.cos (Real.pi / 8)) in
  let A_circumscribed := 2 * (1 + Real.sqrt 2) * (2 * R * Real.sin (Real.pi / 8))^2 in
  A_circumscribed / A_inscribed = 4 * (3 + 2 * Real.sqrt 2) :=
by
  sorry

end octagon_area_ratio_l314_314092


namespace small_triangle_count_l314_314350

theorem small_triangle_count (n : ℕ) (h : n = 2009) : (2 * n + 1) = 4019 := 
by {
    sorry
}

end small_triangle_count_l314_314350


namespace river_depth_difference_l314_314014

theorem river_depth_difference
  (mid_may_depth : ℕ)
  (mid_july_depth : ℕ)
  (mid_june_depth : ℕ)
  (H1 : mid_july_depth = 45)
  (H2 : mid_may_depth = 5)
  (H3 : 3 * mid_june_depth = mid_july_depth) :
  mid_june_depth - mid_may_depth = 10 := 
sorry

end river_depth_difference_l314_314014


namespace hypotenuse_length_l314_314133

theorem hypotenuse_length (a b : ℕ) (h1 : a = 36) (h2 : b = 48) : 
  ∃ c : ℕ, c * c = a * a + b * b ∧ c = 60 := 
by 
  use 60
  sorry

end hypotenuse_length_l314_314133


namespace sin1993_cos1993_leq_zero_l314_314358

theorem sin1993_cos1993_leq_zero (x : ℝ) (h : Real.sin x + Real.cos x ≤ 0) : 
  Real.sin x ^ 1993 + Real.cos x ^ 1993 ≤ 0 := 
by 
  sorry

end sin1993_cos1993_leq_zero_l314_314358


namespace circle_distance_to_line_l314_314194

-- Define the conditions
def circle_pass_through (P : ℝ × ℝ) (a : ℝ) : Prop :=
  let (x, y) := P
  (x - a)^2 + (y - a)^2 = a^2

def is_tangent_to_axes (a : ℝ) : Prop :=
  a > 0

def distance_to_line (center : ℝ × ℝ) (L : ℝ × ℝ → ℝ) : ℝ :=
  let (x₁, y₁) := center
  abs (L (x₁, y₁)) / sqrt ((L (1,0))^2 + (L (0,1))^2)

-- Define the line equation as a function
def line_eq (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  2*x - y - 3

-- Define the main proof goal
theorem circle_distance_to_line (a : ℝ) (h1 : circle_pass_through (2, 1) a) (h2 : is_tangent_to_axes a) :
  distance_to_line (a, a) line_eq = 2*sqrt 5 / 5 :=
sorry

end circle_distance_to_line_l314_314194


namespace factorization_1_factorization_2_l314_314145

variables {x y m n : ℝ}

theorem factorization_1 : x^3 + 2 * x^2 * y + x * y^2 = x * (x + y)^2 :=
sorry

theorem factorization_2 : 4 * m^2 - n^2 - 4 * m + 1 = (2 * m - 1 + n) * (2 * m - 1 - n) :=
sorry

end factorization_1_factorization_2_l314_314145


namespace total_students_in_classrooms_l314_314261

theorem total_students_in_classrooms (tina_students maura_students zack_students : ℕ) 
    (h1 : tina_students = maura_students)
    (h2 : zack_students = (tina_students + maura_students) / 2)
    (h3 : 22 + 1 = zack_students) : 
    tina_students + maura_students + zack_students = 69 := 
by 
  -- Proof steps would go here, but we include 'sorry' as per the instructions.
  sorry

end total_students_in_classrooms_l314_314261


namespace ratio_of_area_of_inscribed_and_circumscribed_octagons_l314_314114

noncomputable def area_ratio_of_octagons (r : ℝ) : ℝ :=
  let cos22_5 := (√(2 + √2) / 2)
  let r_prime := r / cos22_5
  let area_ratio := (r_prime / r)^2
  area_ratio

theorem ratio_of_area_of_inscribed_and_circumscribed_octagons (r : ℝ) :
  area_ratio_of_octagons r = (4 - 2 * √2) / 2 := by
  sorry

end ratio_of_area_of_inscribed_and_circumscribed_octagons_l314_314114


namespace consecutive_ints_prod_square_l314_314026

theorem consecutive_ints_prod_square (n : ℤ) : 
  ∃ k : ℤ, n * (n + 1) * (n + 2) * (n + 3) + 1 = k^2 :=
sorry

end consecutive_ints_prod_square_l314_314026


namespace ratio_area_octagons_correct_l314_314105

noncomputable def ratio_area_octagons (r : ℝ) : ℝ :=
  let s := 2 * r * Real.sin (Real.pi / 8)   -- side length of inscribed octagon
  let S := 2 * r * Real.cos (Real.pi / 8)   -- side length of circumscribed octagon
  let ratio_side_lengths := S / s
  let ratio_areas := (ratio_side_lengths)^2
  ratio_areas

theorem ratio_area_octagons_correct (r : ℝ) :
  ratio_area_octagons r = 6 + 4 * Real.sqrt 2 :=
by
  sorry

end ratio_area_octagons_correct_l314_314105


namespace distance_from_center_of_circle_to_line_l314_314192

open Real

def circle_passes_through (a : ℝ) :=
  (2 - a) ^ 2 + (1 - a) ^ 2 = a ^ 2

def circle_tangent_to_axes (a : ℝ) :=
  a > 0

def distance_from_center_to_line (a : ℝ) : ℝ :=
  abs (2 * a - a - 3) / sqrt (2 ^ 2 + (-1) ^ 2)

theorem distance_from_center_of_circle_to_line
  (a : ℝ) (h1 : circle_passes_through a) (h2 : circle_tangent_to_axes a) :
  distance_from_center_to_line a = 2 * sqrt 5 / 5 :=
sorry

end distance_from_center_of_circle_to_line_l314_314192


namespace distance_to_line_is_constant_l314_314200

noncomputable def center_of_circle (a : ℝ) : (ℝ × ℝ) := (a, a)

noncomputable def circle_equation (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - a)^2 = a^2

noncomputable def distance_from_center_to_line (x1 y1 : ℝ) : ℝ :=
  abs (2 * x1 - y1 - 3) / sqrt (2^2 + (-1)^2)

theorem distance_to_line_is_constant (a : ℝ) (h : circle_equation a 2 1) (hx : a = 5 ∨ a = 1) :
  distance_from_center_to_line a a = 2 * sqrt 5 / 5 :=
by
  sorry

end distance_to_line_is_constant_l314_314200


namespace reflect_over_x_axis_reflect_over_y_axis_l314_314352

-- Mathematical Definitions
def Point := (ℝ × ℝ)

-- Reflect a point over the x-axis
def reflectOverX (M : Point) : Point :=
  (M.1, -M.2)

-- Reflect a point over the y-axis
def reflectOverY (M : Point) : Point :=
  (-M.1, M.2)

-- Theorem statements
theorem reflect_over_x_axis (M : Point) : reflectOverX M = (M.1, -M.2) :=
by
  sorry

theorem reflect_over_y_axis (M : Point) : reflectOverY M = (-M.1, M.2) :=
by
  sorry

end reflect_over_x_axis_reflect_over_y_axis_l314_314352


namespace speed_A_correct_l314_314290

noncomputable def speed_A : ℝ :=
  200 / (19.99840012798976 * 60)

theorem speed_A_correct :
  speed_A = 0.16668 :=
sorry

end speed_A_correct_l314_314290


namespace nell_initial_ace_cards_l314_314398

def initial_ace_cards (initial_baseball_cards final_ace_cards final_baseball_cards given_difference : ℕ) : ℕ :=
  final_ace_cards + (initial_baseball_cards - final_baseball_cards)

theorem nell_initial_ace_cards : 
  initial_ace_cards 239 376 111 265 = 504 :=
by
  /- This is to show that the initial count of Ace cards Nell had is 504 given the conditions -/
  sorry

end nell_initial_ace_cards_l314_314398


namespace gcd_possible_values_count_l314_314055

theorem gcd_possible_values_count (a b : ℕ) (h : a * b = 360) : 
  {d : ℕ | d ∣ a ∧ d ∣ b}.card = 6 :=
sorry

end gcd_possible_values_count_l314_314055


namespace miles_per_gallon_l314_314022

theorem miles_per_gallon (miles gallons : ℝ) (h : miles = 100 ∧ gallons = 5) : miles / gallons = 20 := by
  cases h with
  | intro miles_eq gallons_eq =>
    rw [miles_eq, gallons_eq]
    norm_num

end miles_per_gallon_l314_314022


namespace fill_tanker_time_l314_314472

/-- Given that pipe A can fill the tanker in 60 minutes and pipe B can fill the tanker in 40 minutes,
    prove that the time T to fill the tanker if pipe B is used for half the time and both pipes 
    A and B are used together for the other half is equal to 30 minutes. -/
theorem fill_tanker_time (T : ℝ) (hA : ∀ (a : ℝ), a = 1/60) (hB : ∀ (b : ℝ), b = 1/40) :
  (T / 2) * (1 / 40) + (T / 2) * (1 / 24) = 1 → T = 30 :=
by
  sorry

end fill_tanker_time_l314_314472


namespace sum_of_opposites_is_zero_l314_314204

theorem sum_of_opposites_is_zero (a b : ℚ) (h : a = -b) : a + b = 0 := 
by sorry

end sum_of_opposites_is_zero_l314_314204


namespace range_of_f_l314_314157

def f (x : ℝ) : ℝ := (x + (1 / x)) / (⌊x⌋ * ⌊1 / x⌋ + ⌊x⌋ + ⌊1 / x⌋ + 1)

theorem range_of_f (x : ℝ) (hx : x > 0) :
  (∃ y, y = f x ∧ (y = 1/2 ∨ (5/6 ≤ y ∧ y < 5/4))) :=
by
  sorry

end range_of_f_l314_314157


namespace swimming_pool_width_l314_314456

theorem swimming_pool_width 
  (V_G : ℝ) (G_CF : ℝ) (height_inch : ℝ) (L : ℝ) (V_CF : ℝ) (height_ft : ℝ) (A : ℝ) (W : ℝ) :
  V_G = 3750 → G_CF = 7.48052 → height_inch = 6 → L = 40 →
  V_CF = V_G / G_CF → height_ft = height_inch / 12 →
  A = L * W → V_CF = A * height_ft →
  W = 25.067 :=
by
  intros hV hG hH hL hVC hHF hA hVF
  sorry

end swimming_pool_width_l314_314456


namespace problem_1_problem_2_l314_314225

section Problem1

variable (x a : ℝ)

-- Proposition p
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0

-- Proposition q
def q (x : ℝ) : Prop := (x - 3) / (2 - x) ≥ 0

-- Problem 1
theorem problem_1 : p 1 x ∧ q x → 2 < x ∧ x < 3 :=
by { sorry }

end Problem1

section Problem2

variable (a : ℝ)

-- Proposition p with a as a variable
def p_a (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0

-- Proposition q with x as a variable
def q_x (x : ℝ) : Prop := (x - 3) / (2 - x) ≥ 0

-- Problem 2
theorem problem_2 : (∀ (x : ℝ), ¬p_a a x → ¬q_x x) → (1 < a ∧ a ≤ 2) :=
by { sorry }

end Problem2

end problem_1_problem_2_l314_314225


namespace prove_tan_570_eq_sqrt_3_over_3_l314_314150

noncomputable def tan_570_eq_sqrt_3_over_3 : Prop :=
  Real.tan (570 * Real.pi / 180) = Real.sqrt 3 / 3

theorem prove_tan_570_eq_sqrt_3_over_3 : tan_570_eq_sqrt_3_over_3 :=
by
  sorry

end prove_tan_570_eq_sqrt_3_over_3_l314_314150


namespace evaluate_expression_l314_314143

theorem evaluate_expression (x z : ℤ) (h1 : x = 2) (h2 : z = 1) : z * (z - 4 * x) = -7 :=
by
  rw [h1, h2]
  sorry

end evaluate_expression_l314_314143


namespace theater_ticket_sales_l314_314267

theorem theater_ticket_sales 
  (total_tickets : ℕ) (price_adult_ticket : ℕ) (price_senior_ticket : ℕ) (senior_tickets_sold : ℕ) 
  (Total_tickets_condition : total_tickets = 510)
  (Price_adult_ticket_condition : price_adult_ticket = 21)
  (Price_senior_ticket_condition : price_senior_ticket = 15)
  (Senior_tickets_sold_condition : senior_tickets_sold = 327) : 
  (183 * 21 + 327 * 15 = 8748) :=
by
  sorry

end theater_ticket_sales_l314_314267


namespace four_digit_swap_square_l314_314003

theorem four_digit_swap_square (a b : ℤ) (N M : ℤ) : 
  N = 1111 * a + 123 ∧ 
  M = 1111 * a + 1023 ∧ 
  M = b ^ 2 → 
  N = 3456 := 
by sorry

end four_digit_swap_square_l314_314003


namespace part1_part2_find_min_value_l314_314156

open Real

-- Proof of Part 1
theorem part1 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : a^2 / b + b^2 / a ≥ a + b :=
by sorry

-- Proof of Part 2
theorem part2 (x : ℝ) (hx : 0 < x) (hx1 : x < 1) : (1 - x)^2 / x + x^2 / (1 - x) ≥ 1 :=
by sorry

-- Corollary to find the minimum value
theorem find_min_value (x : ℝ) (hx : 0 < x) (hx1 : x < 1) : (1 - x)^2 / x + x^2 / (1 - x) = 1 ↔ x = 1 / 2 :=
by sorry

end part1_part2_find_min_value_l314_314156


namespace ak_divisibility_l314_314402

theorem ak_divisibility {a k m n : ℕ} (h : a ^ k % (m ^ n) = 0) : a ^ (k * m) % (m ^ (n + 1)) = 0 :=
sorry

end ak_divisibility_l314_314402


namespace cost_of_each_orange_l314_314233

theorem cost_of_each_orange (calories_per_orange : ℝ) (total_money : ℝ) (calories_needed : ℝ) (money_left : ℝ) :
  calories_per_orange = 80 → 
  total_money = 10 → 
  calories_needed = 400 → 
  money_left = 4 → 
  (total_money - money_left) / (calories_needed / calories_per_orange) = 1.2 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end cost_of_each_orange_l314_314233


namespace total_money_l314_314318

variable (A B C : ℕ)

theorem total_money
  (h1 : A + C = 250)
  (h2 : B + C = 450)
  (h3 : C = 100) :
  A + B + C = 600 := by
  sorry

end total_money_l314_314318


namespace blue_face_area_greater_than_red_face_area_l314_314425

theorem blue_face_area_greater_than_red_face_area :
  let original_cube_side := 13
  let total_red_area := 6 * original_cube_side^2
  let num_mini_cubes := original_cube_side^3
  let total_faces_mini_cubes := 6 * num_mini_cubes
  let total_blue_area := total_faces_mini_cubes - total_red_area
  (total_blue_area / total_red_area) = 12 :=
by
  sorry

end blue_face_area_greater_than_red_face_area_l314_314425


namespace ratio_area_octagons_correct_l314_314108

noncomputable def ratio_area_octagons (r : ℝ) : ℝ :=
  let s := 2 * r * Real.sin (Real.pi / 8)   -- side length of inscribed octagon
  let S := 2 * r * Real.cos (Real.pi / 8)   -- side length of circumscribed octagon
  let ratio_side_lengths := S / s
  let ratio_areas := (ratio_side_lengths)^2
  ratio_areas

theorem ratio_area_octagons_correct (r : ℝ) :
  ratio_area_octagons r = 6 + 4 * Real.sqrt 2 :=
by
  sorry

end ratio_area_octagons_correct_l314_314108


namespace base_seven_to_ten_l314_314269

theorem base_seven_to_ten :
  4 * 7^4 + 3 * 7^3 + 2 * 7^2 + 1 * 7^1 + 0 * 7^0 = 10738 :=
by
  sorry

end base_seven_to_ten_l314_314269


namespace solve_integral_problem_l314_314257

noncomputable def integral_problem : Prop :=
  ∫ x in 0 .. 1, (x^2 + Real.exp x - 1/3) = Real.exp 1 - 1

theorem solve_integral_problem : integral_problem := by
  sorry

end solve_integral_problem_l314_314257


namespace gcd_three_digit_palindromes_l314_314460

theorem gcd_three_digit_palindromes : 
  ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) → (1 ≤ b ∧ b ≤ 9) → 
  ∃ d : ℕ, d = 1 ∧ ∀ n m : ℕ, (n = 101 * a + 10 * b) → (m = 101 * a + 10 * b) → gcd n m = d := 
by sorry

end gcd_three_digit_palindromes_l314_314460


namespace blue_red_area_ratio_l314_314420

theorem blue_red_area_ratio (n : ℕ) (h : n = 13) :
    let total_red_faces := 6 * n^2,
        total_faces := 6 * n^3,
        blue_faces := total_faces - total_red_faces
    in (blue_faces / total_red_faces : ℕ) = 12 :=
  by
  sorry

end blue_red_area_ratio_l314_314420


namespace cos_B_in_third_quadrant_l314_314203

theorem cos_B_in_third_quadrant (B : ℝ) (hB: π < B ∧ B < 3 * π / 2) (hSinB: Real.sin B = 5 / 13) : Real.cos B = - 12 / 13 := by
  sorry

end cos_B_in_third_quadrant_l314_314203


namespace basketball_success_rate_l314_314080

theorem basketball_success_rate (p : ℝ) (h : 1 - p^2 = 16 / 25) : p = 3 / 5 :=
sorry

end basketball_success_rate_l314_314080


namespace probability_two_heads_one_tail_in_three_tosses_l314_314281

theorem probability_two_heads_one_tail_in_three_tosses
(P : ℕ → Prop) (pr : ℤ) : 
  (∀ n, P n → pr = 1 / 2) -> 
  P 3 → pr = 3 / 8 :=
by
  sorry

end probability_two_heads_one_tail_in_three_tosses_l314_314281


namespace base7_to_base10_of_43210_l314_314272

theorem base7_to_base10_of_43210 : 
  base7_to_base10 (list.num_from_digits [4, 3, 2, 1, 0]) 7 = 10738 :=
by
  def base7_to_base10 (digits : list ℕ) (base : ℕ) : ℕ :=
    digits.reverse.join_with base
  
  show base7_to_base10 [4, 3, 2, 1, 0] 7 = 10738
  sorry

end base7_to_base10_of_43210_l314_314272


namespace find_hansol_weight_l314_314455

variable (H : ℕ)

theorem find_hansol_weight (h : H + (H + 4) = 88) : H = 42 :=
by
  sorry

end find_hansol_weight_l314_314455


namespace speed_of_current_l314_314306

variable (m c : ℝ)

theorem speed_of_current (h1 : m + c = 15) (h2 : m - c = 10) : c = 2.5 :=
sorry

end speed_of_current_l314_314306


namespace solve_system_of_equations_l314_314169

theorem solve_system_of_equations : 
  ∃ (x y : ℝ), 
  (x / y + y / x) * (x + y) = 15 ∧ 
  (x^2 / y^2 + y^2 / x^2) * (x^2 + y^2) = 85 ∧
  ((x = 2 ∧ y = 4) ∨ (x = 4 ∧ y = 2)) :=
by
  sorry

end solve_system_of_equations_l314_314169


namespace teresa_age_when_michiko_born_l314_314245

def conditions (T M Michiko K Yuki : ℕ) : Prop := 
  T = 59 ∧ 
  M = 71 ∧ 
  M - Michiko = 38 ∧ 
  K = Michiko - 4 ∧ 
  Yuki = K - 3 ∧ 
  (Yuki + 3) - (26 - 25) = 25

theorem teresa_age_when_michiko_born :
  ∃ T M Michiko K Yuki, conditions T M Michiko K Yuki → T - Michiko = 26 :=
  by
  sorry

end teresa_age_when_michiko_born_l314_314245


namespace solve_expression_l314_314144

theorem solve_expression (a x : ℝ) (h1 : a ≠ 0) (h2 : x ≠ a) : 
  (a / (2 * a + x) - x / (a - x)) / (x / (2 * a + x) + a / (a - x)) = -1 → 
  x = a / 2 :=
by
  sorry

end solve_expression_l314_314144


namespace angle_sum_l314_314160

theorem angle_sum (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (tan_α : Real.tan α = 3 / 4)
  (sin_β : Real.sin β = 3 / 5) :
  α + 3 * β = 5 * Real.pi / 4 := 
sorry

end angle_sum_l314_314160


namespace average_apples_per_hour_l314_314392

theorem average_apples_per_hour (A H : ℝ) (hA : A = 12) (hH : H = 5) : A / H = 2.4 := by
  -- sorry skips the proof
  sorry

end average_apples_per_hour_l314_314392


namespace box_volume_l314_314088

theorem box_volume (x : ℕ) (h_ratio : (x > 0)) (V : ℕ) (h_volume : V = 20 * x^3) : V = 160 :=
by
  sorry

end box_volume_l314_314088


namespace original_number_l314_314277

theorem original_number (n : ℕ) (h : (n + 1) % 30 = 0) : n = 29 :=
by
  sorry

end original_number_l314_314277


namespace system_of_equations_solution_l314_314337

theorem system_of_equations_solution :
  ∃ x y : ℚ, (3 * x + 4 * y = 10) ∧ (12 * x - 8 * y = 8) ∧ (x = 14 / 9) ∧ (y = 4 / 3) :=
by
  sorry

end system_of_equations_solution_l314_314337


namespace ratio_of_increase_to_current_l314_314331

-- Define the constants for the problem
def current_deductible : ℝ := 3000
def increase_deductible : ℝ := 2000

-- State the theorem that needs to be proven
theorem ratio_of_increase_to_current : 
  (increase_deductible / current_deductible) = (2 / 3) :=
by sorry

end ratio_of_increase_to_current_l314_314331


namespace quadratic_inequality_solution_l314_314359

theorem quadratic_inequality_solution (a b c : ℝ) (h1 : 0 > a) 
(h2 : ∀ x : ℝ, (1 < x ∧ x < 2) ↔ (0 < ax^2 + bx + c)) : 
(∀ x : ℝ, (x < 1/2 ∨ 1 < x) ↔ (0 < 2*a*x^2 - 3*a*x + a)) :=
sorry

end quadratic_inequality_solution_l314_314359


namespace xiaoming_probability_l314_314009

-- Define the binomial probability function
def prob_exactly_k_successes (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

-- State the theorem
theorem xiaoming_probability :
  prob_exactly_k_successes 6 2 (1 / 3) = 240 / 729 :=
by sorry

end xiaoming_probability_l314_314009


namespace instantaneous_speed_at_4_l314_314086

def motion_equation (t : ℝ) : ℝ := t^2 - 2 * t + 5

theorem instantaneous_speed_at_4 :
  (deriv motion_equation 4) = 6 :=
by
  sorry

end instantaneous_speed_at_4_l314_314086


namespace cuboid_surface_area_l314_314036

noncomputable def total_surface_area (x y z : ℝ) : ℝ :=
  2 * (x * y + y * z + z * x)

theorem cuboid_surface_area (x y z : ℝ) (h1 : x + y + z = 40) (h2 : x^2 + y^2 + z^2 = 625) :
  total_surface_area x y z = 975 :=
sorry

end cuboid_surface_area_l314_314036


namespace largest_divisor_39_l314_314475

theorem largest_divisor_39 (m : ℕ) (hm : 0 < m) (h : 39 ∣ m ^ 2) : 39 ∣ m :=
by sorry

end largest_divisor_39_l314_314475


namespace smallest_integer_mod_inverse_l314_314462

theorem smallest_integer_mod_inverse (n : ℕ) (h1 : n > 1) (h2 : gcd n 1001 = 1) : n = 2 :=
sorry

end smallest_integer_mod_inverse_l314_314462


namespace find_n_l314_314345

theorem find_n : ∃ (n : ℤ), -150 < n ∧ n < 150 ∧ Real.tan (n * Real.pi / 180) = Real.tan (1600 * Real.pi / 180) :=
sorry

end find_n_l314_314345


namespace smallest_positive_period_of_sin_2x_l314_314347

noncomputable def period_of_sine (B : ℝ) : ℝ := (2 * Real.pi) / B

theorem smallest_positive_period_of_sin_2x :
  period_of_sine 2 = Real.pi := sorry

end smallest_positive_period_of_sin_2x_l314_314347


namespace percentage_of_students_owning_cats_l314_314008

theorem percentage_of_students_owning_cats (dogs cats total : ℕ) (h_dogs : dogs = 45) (h_cats : cats = 75) (h_total : total = 500) : 
  (cats / total) * 100 = 15 :=
by
  sorry

end percentage_of_students_owning_cats_l314_314008


namespace maximum_volume_pyramid_is_one_sixteenth_l314_314248

open Real  -- Opening Real namespace for real number operations

noncomputable def maximum_volume_pyramid : ℝ :=
  let a := 1 -- side length of the equilateral triangle base
  let base_area := (sqrt 3 / 4) * (a * a) -- area of the equilateral triangle with side length 1
  let median := sqrt 3 / 2 * a -- median length of the triangle
  let height := 1 / 2 * median -- height of the pyramid
  let volume := 1 / 3 * base_area * height -- volume formula for a pyramid
  volume

theorem maximum_volume_pyramid_is_one_sixteenth :
  maximum_volume_pyramid = 1 / 16 :=
by
  simp [maximum_volume_pyramid] -- Simplify the volume definition
  sorry -- Proof omitted

end maximum_volume_pyramid_is_one_sixteenth_l314_314248


namespace cone_lateral_surface_area_ratio_l314_314002

/-- Let a be the side length of the equilateral triangle front view of a cone.
    The base area of the cone is (π * (a / 2)^2).
    The lateral surface area of the cone is (π * (a / 2) * a).
    We want to show that the ratio of the lateral surface area to the base area is 2.
 -/
theorem cone_lateral_surface_area_ratio 
  (a : ℝ) 
  (base_area : ℝ := π * (a / 2)^2) 
  (lateral_surface_area : ℝ := π * (a / 2) * a) 
  : lateral_surface_area / base_area = 2 :=
by
  sorry

end cone_lateral_surface_area_ratio_l314_314002


namespace max_surface_area_l314_314311

theorem max_surface_area (l w h : ℕ) (h_conditions : l + w + h = 88) : 
  2 * (l * w + l * h + w * h) ≤ 224 :=
sorry

end max_surface_area_l314_314311


namespace value_of_m_minus_n_l314_314368

variables {a b : ℕ}
variables {m n : ℤ}

def are_like_terms (m n : ℤ) : Prop :=
  (m - 2 = 4) ∧ (n + 7 = 4)

theorem value_of_m_minus_n (h : are_like_terms m n) : m - n = 9 :=
by
  sorry

end value_of_m_minus_n_l314_314368


namespace total_population_l314_314041

def grown_ups : ℕ := 5256
def children : ℕ := 2987

theorem total_population : grown_ups + children = 8243 :=
by
  sorry

end total_population_l314_314041


namespace blue_red_area_ratio_l314_314421

theorem blue_red_area_ratio (n : ℕ) (h : n = 13) :
    let total_red_faces := 6 * n^2,
        total_faces := 6 * n^3,
        blue_faces := total_faces - total_red_faces
    in (blue_faces / total_red_faces : ℕ) = 12 :=
  by
  sorry

end blue_red_area_ratio_l314_314421


namespace binomial_multiplication_subtract_240_l314_314336

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem binomial_multiplication_subtract_240 :
  binom 10 3 * binom 8 3 - 240 = 6480 :=
by
  sorry

end binomial_multiplication_subtract_240_l314_314336


namespace jacob_peter_age_ratio_l314_314340

theorem jacob_peter_age_ratio
  (Drew Maya Peter John Jacob : ℕ)
  (h1: Drew = Maya + 5)
  (h2: Peter = Drew + 4)
  (h3: John = 2 * Maya)
  (h4: John = 30)
  (h5: Jacob = 11) :
  Jacob + 2 = 1 / 2 * (Peter + 2) := by
  sorry

end jacob_peter_age_ratio_l314_314340


namespace blue_area_factor_12_l314_314427

theorem blue_area_factor_12 : 
  let n := 13 in
  let total_red_area := 6 * n^2 in         -- The total red area on the surface of the original cube.
  let total_faces := 6 * n^3 in            -- The total number of faces considering all mini-cubes.
  let total_blue_area := total_faces - total_red_area in -- The total blue area.
  total_blue_area / total_red_area = 12 := -- The desired relationship.
by
  sorry

end blue_area_factor_12_l314_314427


namespace milburg_population_l314_314040

/-- Number of grown-ups in Milburg --/
def grownUps : ℕ := 5256

/-- Number of children in Milburg --/
def children : ℕ := 2987

/-- Total number of people in Milburg --/
def totalPeople : ℕ := grownUps + children

theorem milburg_population : totalPeople = 8243 := by
  have h1 : grownUps = 5256 := rfl
  have h2 : children = 2987 := rfl
  have h3 : totalPeople = grownUps + children := rfl
  have h4 : grownUps + children = 8243 := by
    calc
      5256 + 2987 = 8243 := by sorry -- Proof step to be filled in
  exact h4

end milburg_population_l314_314040


namespace circle_distance_condition_l314_314179

theorem circle_distance_condition (a : ℝ) (h1 : (2 - a)^2 + (1 - a)^2 = a^2) (h2 : ∀ (x y : ℝ), ∃ c : ℝ, y = 2*x - c) :
    ∃ (center : ℝ × ℝ), ((center.1 = a) ∧ (center.2 = a) ∧ (center = (1, 1) ∨ center = (5, 5))) 
    → (∀ (line : ℝ × ℝ × ℝ), line = (2, -1, -3) → abs ((line.1 * a + line.2 * a + line.3) / real.sqrt (line.1 ^ 2 + line.2 ^ 2)) = (2 * real.sqrt 5 / 5)) :=
by
  intro a h1 h2 center h3 line h4
  sorry

end circle_distance_condition_l314_314179


namespace blue_to_red_face_area_ratio_l314_314442

theorem blue_to_red_face_area_ratio :
  let original_cube_dim := 13
  let red_face_area := 6 * original_cube_dim^2
  let total_faces := 6 * original_cube_dim^3
  let blue_face_area := total_faces - red_face_area
  (blue_face_area / red_face_area) = 12 :=
by
  sorry

end blue_to_red_face_area_ratio_l314_314442


namespace problem_l314_314168

noncomputable def f (ω x : ℝ) : ℝ := (Real.sin (ω * x / 2))^2 + (1 / 2) * Real.sin (ω * x) - 1 / 2

theorem problem (ω : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, x ∈ Set.Ioo (Real.pi : ℝ) (2 * Real.pi) → f ω x ≠ 0) →
  ω ∈ Set.Icc 0 (1 / 8) ∪ Set.Icc (1 / 4) (5 / 8) :=
by
  sorry

end problem_l314_314168


namespace cost_of_four_stamps_l314_314401

theorem cost_of_four_stamps (cost_one_stamp : ℝ) (h : cost_one_stamp = 0.34) : 4 * cost_one_stamp = 1.36 := 
by
  rw [h]
  norm_num

end cost_of_four_stamps_l314_314401


namespace gcd_values_count_l314_314071

theorem gcd_values_count (a b : ℕ) (h : a * b = 360) : 
  ∃ n, n = (Set.toFinset (Set.image2 Nat.gcd {d | ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * y = 360 ∧ x = d ∨ y = d} {d | ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * y = 360 ∧ x = d ∨ y = d})).card ∧ n = 12 :=
by
  sorry

end gcd_values_count_l314_314071


namespace octagon_area_ratio_l314_314091

theorem octagon_area_ratio (r : ℝ) :
  let A_inscribed := 2 * (1 + Real.sqrt 2) * (r * Real.tan (Real.pi / 8))^2 in
  let R := r * (1 / Real.cos (Real.pi / 8)) in
  let A_circumscribed := 2 * (1 + Real.sqrt 2) * (2 * R * Real.sin (Real.pi / 8))^2 in
  A_circumscribed / A_inscribed = 4 * (3 + 2 * Real.sqrt 2) :=
by
  sorry

end octagon_area_ratio_l314_314091


namespace ratio_one_six_to_five_eighths_l314_314453

theorem ratio_one_six_to_five_eighths : (1 / 6) / (5 / 8) = 4 / 15 := by
  sorry

end ratio_one_six_to_five_eighths_l314_314453


namespace blue_face_area_greater_than_red_face_area_l314_314423

theorem blue_face_area_greater_than_red_face_area :
  let original_cube_side := 13
  let total_red_area := 6 * original_cube_side^2
  let num_mini_cubes := original_cube_side^3
  let total_faces_mini_cubes := 6 * num_mini_cubes
  let total_blue_area := total_faces_mini_cubes - total_red_area
  (total_blue_area / total_red_area) = 12 :=
by
  sorry

end blue_face_area_greater_than_red_face_area_l314_314423


namespace base_seven_to_ten_l314_314270

theorem base_seven_to_ten :
  4 * 7^4 + 3 * 7^3 + 2 * 7^2 + 1 * 7^1 + 0 * 7^0 = 10738 :=
by
  sorry

end base_seven_to_ten_l314_314270


namespace area_ratio_is_correct_l314_314110

noncomputable def areaRatioInscribedCircumscribedOctagon (r : ℝ) : ℝ :=
  let s1 := r * Real.sin (67.5 * Real.pi / 180)
  let s2 := r * Real.sin (67.5 * Real.pi / 180) / Real.cos (67.5 * Real.pi / 180)
  (s2 / s1) ^ 2

theorem area_ratio_is_correct (r : ℝ) : areaRatioInscribedCircumscribedOctagon r = 2 + Real.sqrt 2 :=
by
  sorry

end area_ratio_is_correct_l314_314110


namespace initial_stock_before_shipment_l314_314317

-- Define the conditions for the problem
def initial_stock (total_shelves new_shipment_bears bears_per_shelf: ℕ) : ℕ :=
  let total_bears_on_shelves := total_shelves * bears_per_shelf
  total_bears_on_shelves - new_shipment_bears

-- State the theorem with the conditions
theorem initial_stock_before_shipment : initial_stock 2 10 7 = 4 := by
  -- Mathematically, the calculation details will be handled here
  sorry

end initial_stock_before_shipment_l314_314317


namespace sum_of_digits_l314_314470

theorem sum_of_digits (a b : ℕ) (h1 : 4 * 100 + a * 10 + 3 + 984 = 1 * 1000 + 3 * 100 + b * 10 + 7)
  (h2 : (1 + b) - (3 + 7) % 11 = 0) : a + b = 10 := 
by
  sorry

end sum_of_digits_l314_314470


namespace total_savings_at_end_of_year_l314_314234

-- Defining constants for daily savings and the number of days in a year
def daily_savings : ℕ := 24
def days_in_year : ℕ := 365

-- Stating the theorem
theorem total_savings_at_end_of_year : daily_savings * days_in_year = 8760 :=
by
  sorry

end total_savings_at_end_of_year_l314_314234


namespace trapezoid_area_is_correct_l314_314323

noncomputable def isosceles_trapezoid_area : ℝ :=
  let a : ℝ := 12
  let b : ℝ := 24 - 12 * Real.sqrt 2
  let h : ℝ := 6 * Real.sqrt 2
  (24 + b) / 2 * h

theorem trapezoid_area_is_correct :
  let a := 12
  let b := 24 - 12 * Real.sqrt 2
  let h := 6 * Real.sqrt 2
  (24 + b) / 2 * h = 144 * Real.sqrt 2 - 72 :=
by
  sorry

end trapezoid_area_is_correct_l314_314323


namespace geometric_sequence_common_ratio_l314_314376

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : a 3 - 3 * a 2 = 3 * a 2 - 5 * a 1)
  (h2 : ∀ n, a n > 0)
  (h3 : ∀ n, a n < a (n + 1))
  (h4 : ∀ n, a (n + 1) = a n * q) : 
  q = 5 :=
  sorry

end geometric_sequence_common_ratio_l314_314376


namespace joined_toucans_is_1_l314_314083

-- Define the number of toucans initially
def initial_toucans : ℕ := 2

-- Define the total number of toucans after some join
def total_toucans : ℕ := 3

-- Define the number of toucans that joined
def toucans_joined : ℕ := total_toucans - initial_toucans

-- State the theorem to prove that 1 toucan joined
theorem joined_toucans_is_1 : toucans_joined = 1 :=
by
  sorry

end joined_toucans_is_1_l314_314083


namespace area_ratio_is_correct_l314_314109

noncomputable def areaRatioInscribedCircumscribedOctagon (r : ℝ) : ℝ :=
  let s1 := r * Real.sin (67.5 * Real.pi / 180)
  let s2 := r * Real.sin (67.5 * Real.pi / 180) / Real.cos (67.5 * Real.pi / 180)
  (s2 / s1) ^ 2

theorem area_ratio_is_correct (r : ℝ) : areaRatioInscribedCircumscribedOctagon r = 2 + Real.sqrt 2 :=
by
  sorry

end area_ratio_is_correct_l314_314109


namespace common_area_of_triangles_is_25_l314_314265

-- Define basic properties and conditions of an isosceles right triangle with hypotenuse = 10 units
def hypotenuse (a b : ℝ) : Prop := a^2 + b^2 = 10^2
def is_isosceles_right_triangle (a b : ℝ) : Prop := a = b ∧ hypotenuse a b

-- Definitions representing the triangls
noncomputable def triangle1 := ∃ a b : ℝ, is_isosceles_right_triangle a b
noncomputable def triangle2 := ∃ a b : ℝ, is_isosceles_right_triangle a b

-- The area common to both triangles is the focus
theorem common_area_of_triangles_is_25 : 
  triangle1 ∧ triangle2 → 
  ∃ area : ℝ, area = 25 
  := 
sorry

end common_area_of_triangles_is_25_l314_314265


namespace angle_in_third_quadrant_l314_314328

theorem angle_in_third_quadrant (θ : ℝ) (hθ : θ = 2023) :
    ∃ k : ℤ, (2023 - k * 360) = 223 ∧ 180 ≤ 223 ∧ 223 < 270 := by
sorry

end angle_in_third_quadrant_l314_314328


namespace parabola_directrix_l314_314251

theorem parabola_directrix (x y : ℝ) :
    x^2 = - (1 / 4) * y → y = - (1 / 16) :=
by
  sorry

end parabola_directrix_l314_314251


namespace correct_calculation_l314_314283

theorem correct_calculation (x y : ℝ) : 
  ¬(2 * x^2 + 3 * x^2 = 6 * x^2) ∧ 
  ¬(x^4 * x^2 = x^8) ∧ 
  ¬(x^6 / x^2 = x^3) ∧ 
  ((x * y^2)^2 = x^2 * y^4) :=
by
  sorry

end correct_calculation_l314_314283


namespace distance_from_center_to_line_of_tangent_circle_l314_314182

theorem distance_from_center_to_line_of_tangent_circle 
  (a : ℝ) (ha : 0 < a) 
  (h_circle : (2 - a)^2 + (1 - a)^2 = a^2)
  (h_tangent : ∀ x y : ℝ, x = 0 ∨ y = 0): 
  (|2 * a - a - 3| / ((2:ℝ)^2 + (-1)^2).sqrt) = (2 * (5:ℝ).sqrt) / 5 :=
by
  sorry

end distance_from_center_to_line_of_tangent_circle_l314_314182


namespace blue_area_factor_12_l314_314426

theorem blue_area_factor_12 : 
  let n := 13 in
  let total_red_area := 6 * n^2 in         -- The total red area on the surface of the original cube.
  let total_faces := 6 * n^3 in            -- The total number of faces considering all mini-cubes.
  let total_blue_area := total_faces - total_red_area in -- The total blue area.
  total_blue_area / total_red_area = 12 := -- The desired relationship.
by
  sorry

end blue_area_factor_12_l314_314426


namespace ruiz_original_salary_l314_314406

theorem ruiz_original_salary (S : ℝ) (h : 1.06 * S = 530) : S = 500 :=
by {
  -- Proof goes here
  sorry
}

end ruiz_original_salary_l314_314406


namespace correct_calculation_l314_314285

theorem correct_calculation (x y : ℝ) : (x * y^2) ^ 2 = x^2 * y^4 :=
by
  sorry

end correct_calculation_l314_314285


namespace range_of_f_l314_314148

noncomputable def f (x : ℝ) : ℝ :=
  (Real.arccos (x / 2))^2 + 
  Real.pi * Real.arcsin (x / 2) - 
  (Real.arcsin (x / 2))^2 + 
  (Real.pi^2 / 6) * (x^2 + 2 * x + 1)

theorem range_of_f (x : ℝ) (hx : -2 ≤ x ∧ x ≤ 2) :
  ∃ y : ℝ, (f y) = x ∧  (Real.pi^2 / 4) ≤ y ∧ y ≤ (39 * Real.pi^2 / 96) := 
sorry

end range_of_f_l314_314148


namespace incorrect_conclusion_D_l314_314380

-- Define lines and planes
variables (l m n : Type) -- lines
variables (α β γ : Type) -- planes

-- Define the conditions
def intersection_planes (p1 p2 : Type) : Type := sorry
def perpendicular (a b : Type) : Prop := sorry

-- Given conditions for option D
axiom h1 : intersection_planes α β = m
axiom h2 : intersection_planes β γ = l
axiom h3 : intersection_planes γ α = n
axiom h4 : perpendicular l m
axiom h5 : perpendicular l n

-- Theorem stating that the conclusion of option D is incorrect
theorem incorrect_conclusion_D : ¬ perpendicular m n :=
by sorry

end incorrect_conclusion_D_l314_314380


namespace octagon_area_ratio_l314_314117

theorem octagon_area_ratio (r : ℝ) : 
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2 := 
by
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  have h : circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2
  exact h
  sorry

end octagon_area_ratio_l314_314117


namespace smallest_n_for_terminating_decimal_l314_314463

theorem smallest_n_for_terminating_decimal : 
  ∃ n : ℕ, (0 < n) ∧ (∃ k m : ℕ, (n + 70 = 2 ^ k * 5 ^ m) ∧ k = 0 ∨ k = 1) ∧ n = 55 :=
by sorry

end smallest_n_for_terminating_decimal_l314_314463


namespace sum_first_50_arithmetic_sequence_l314_314029

theorem sum_first_50_arithmetic_sequence : 
  let a : ℕ := 2
  let d : ℕ := 4
  let n : ℕ := 50
  let a_n (n : ℕ) : ℕ := a + (n - 1) * d
  let S_n (n : ℕ) : ℕ := n / 2 * (2 * a + (n - 1) * d)
  S_n n = 5000 :=
by
  sorry

end sum_first_50_arithmetic_sequence_l314_314029


namespace even_function_exists_l314_314447

noncomputable def example_even_function : ℝ → ℝ :=
  λ x, (8 / 21) * x^4 - (80 / 21) * x^2 + (24 / 7)

theorem even_function_exists :
  (example_even_function (-1) = 0) ∧
  (example_even_function (0.5) = 2.5) ∧
  (example_even_function 3 = 0) ∧
  (∀ x : ℝ, example_even_function x = example_even_function (-x)) :=
begin
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  {
    intro x,
    simp,
    sorry
  }
end

end even_function_exists_l314_314447


namespace range_of_function_l314_314149

theorem range_of_function :
  (∀ y : ℝ, (∃ x : ℝ, y = (x + 1) / (x ^ 2 + 1)) ↔ 0 ≤ y ∧ y ≤ 4/3) :=
by
  sorry

end range_of_function_l314_314149


namespace total_profit_l314_314127

-- Define the relevant variables and conditions
variables (x y : ℝ) -- Cost prices of the two music players

-- Given conditions
axiom cost_price_first : x * 1.2 = 132
axiom cost_price_second : y * 1.1 = 132

theorem total_profit : 132 + 132 - y - x = 34 :=
by
  -- The proof body is not required
  sorry

end total_profit_l314_314127


namespace solve_inequality_l314_314154

theorem solve_inequality (x: ℝ) : (25 - 5 * Real.sqrt 3) ≤ x ∧ x ≤ (25 + 5 * Real.sqrt 3) ↔ x ^ 2 - 50 * x + 575 ≤ 25 :=
by
  sorry

end solve_inequality_l314_314154


namespace B_squared_B_sixth_l314_314389

noncomputable def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![0, 3], ![2, -1]]

noncomputable def I : Matrix (Fin 2) (Fin 2) ℤ :=
  1

theorem B_squared :
  B * B = 3 * B - I := by
  sorry

theorem B_sixth :
  B^6 = 84 * B - 44 * I := by
  sorry

end B_squared_B_sixth_l314_314389


namespace average_mark_is_correct_l314_314010

-- Define the maximum score in the exam
def max_score := 1100

-- Define the percentages scored by Amar, Bhavan, Chetan, and Deepak
def score_percentage_amar := 64 / 100
def score_percentage_bhavan := 36 / 100
def score_percentage_chetan := 44 / 100
def score_percentage_deepak := 52 / 100

-- Calculate the actual scores based on percentages
def score_amar := score_percentage_amar * max_score
def score_bhavan := score_percentage_bhavan * max_score
def score_chetan := score_percentage_chetan * max_score
def score_deepak := score_percentage_deepak * max_score

-- Define the total score
def total_score := score_amar + score_bhavan + score_chetan + score_deepak

-- Define the number of students
def number_of_students := 4

-- Define the average score
def average_score := total_score / number_of_students

-- The theorem to prove that the average score is 539
theorem average_mark_is_correct : average_score = 539 := by
  -- Proof skipped
  sorry

end average_mark_is_correct_l314_314010


namespace fifth_pile_magazines_l314_314288

theorem fifth_pile_magazines :
  let first_pile := 3
  let second_pile := first_pile + 1
  let third_pile := second_pile + 2
  let fourth_pile := third_pile + 3
  let fifth_pile := fourth_pile + (3 + 1)
  fifth_pile = 13 :=
by
  let first_pile := 3
  let second_pile := first_pile + 1
  let third_pile := second_pile + 2
  let fourth_pile := third_pile + 3
  let fifth_pile := fourth_pile + (3 + 1)
  show fifth_pile = 13
  sorry

end fifth_pile_magazines_l314_314288


namespace tangent_length_external_tangent_length_internal_l314_314255

noncomputable def tangent_length_ext (R r a : ℝ) (h : R > r) (hAB : AB = a) : ℝ :=
  a * Real.sqrt ((R + r) / R)

noncomputable def tangent_length_int (R r a : ℝ) (h : R > r) (hAB : AB = a) : ℝ :=
  a * Real.sqrt ((R - r) / R)

theorem tangent_length_external (R r a T : ℝ) (h : R > r) (hAB : AB = a) :
  T = tangent_length_ext R r a h hAB :=
sorry

theorem tangent_length_internal (R r a T : ℝ) (h : R > r) (hAB : AB = a) :
  T = tangent_length_int R r a h hAB :=
sorry

end tangent_length_external_tangent_length_internal_l314_314255


namespace train_speed_l314_314047

theorem train_speed (v : ℝ) :
  let speed_train1 := 80  -- speed of the first train in km/h
  let length_train1 := 150 / 1000 -- length of the first train in km
  let length_train2 := 100 / 1000 -- length of the second train in km
  let total_time := 5.999520038396928 / 3600 -- time in hours
  let total_length := length_train1 + length_train2 -- total length in km
  let relative_speed := total_length / total_time -- relative speed in km/h
  relative_speed = speed_train1 + v → v = 70 :=
by
  sorry

end train_speed_l314_314047


namespace probability_of_winning_position_l314_314312

theorem probability_of_winning_position : 
    (let total_positions := Nat.choose 9 3,
         winning_positions := 8 in
       (winning_positions : ℚ) / total_positions = 2 / 21) := 
by
  sorry

end probability_of_winning_position_l314_314312


namespace volume_proof_l314_314151

variables (m n p d x V : Real)

namespace VolumeProof

-- Define the conditions
def diag_eq : Prop :=
  d^2 = (m * x)^2 + (n * x)^2 + (p * x)^2

def x_val : Prop :=
  x = d / sqrt(m^2 + n^2 + p^2)

-- Define the volume formula to be proven
def volume_formula : Prop :=
  V = (m * n * p * (d / sqrt(m^2 + n^2 + p^2))^3)

-- The main statement to be proven
theorem volume_proof 
  (h1 : diag_eq) 
  (h2 : x_val) 
  : volume_formula :=
  sorry

end VolumeProof

end volume_proof_l314_314151


namespace line_intersects_y_axis_at_point_l314_314302

def line_intersects_y_axis (x1 y1 x2 y2 : ℚ) : Prop :=
  ∃ c : ℚ, ∀ x : ℚ, y1 + (y2 - y1) / (x2 - x1) * (x - x1) = (y2 - y1) / (x2 - x1) * x + c

theorem line_intersects_y_axis_at_point :
  line_intersects_y_axis 3 21 (-9) (-6) :=
  sorry

end line_intersects_y_axis_at_point_l314_314302


namespace circle_tangent_distance_l314_314187

theorem circle_tangent_distance 
    (a : ℝ) (h_center : ∃ a : ℝ, a > 0 ∧ (circle_center = (a, a)) ∧ ((2 - a)^2 + (1 - a)^2 = a^2)) 
    (h_line : ∀ x y, distance_to_line (circle_center x y) (2 * x - y - 3) = 2 / sqrt (2^2 + 1^2)) :
    distance_to_line (circle_center 1 1) (2 * 1 - 1 - 3) = 2 * sqrt 5 / 5 ∧ distance_to_line (circle_center 5 5) (2 * 5 - 5 - 3) = 2 * sqrt 5 / 5 :=
by
    sorry

end circle_tangent_distance_l314_314187


namespace other_root_of_quadratic_l314_314153

theorem other_root_of_quadratic (m : ℝ) :
  (∀ x : ℝ, x^2 - 4 * x + m = 0 → x = -1) → (∀ y : ℝ, y^2 - 4 * y + m = 0 → y = 5) :=
sorry

end other_root_of_quadratic_l314_314153


namespace solved_work_problem_l314_314294

noncomputable def work_problem : Prop :=
  ∃ (m w x : ℝ), 
  (3 * m + 8 * w = 6 * m + x * w) ∧ 
  (4 * m + 5 * w = 0.9285714285714286 * (3 * m + 8 * w)) ∧
  (x = 14)

theorem solved_work_problem : work_problem := sorry

end solved_work_problem_l314_314294


namespace meaningful_expression_range_l314_314348

theorem meaningful_expression_range (x : ℝ) (h1 : 3 * x + 2 ≥ 0) (h2 : x ≠ 0) : 
  x ∈ Set.Ico (-2 / 3) 0 ∪ Set.Ioi 0 := 
  sorry

end meaningful_expression_range_l314_314348


namespace sin_beta_l314_314357

theorem sin_beta (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) 
  (h3 : Real.cos α = 3 / 5) (h4 : Real.cos (α + β) = -5 / 13)
  : Real.sin β = 56 / 65 :=
by
  sorry

end sin_beta_l314_314357


namespace find_f_29_l314_314209

theorem find_f_29 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (2 * x + 3) = (x - 3) * (x + 4)) : f 29 = 170 := 
by
  sorry

end find_f_29_l314_314209


namespace bird_needs_more_twigs_l314_314295

variable (base_twigs : ℕ := 12)
variable (additional_twigs_per_base : ℕ := 6)
variable (fraction_dropped : ℚ := 1/3)

theorem bird_needs_more_twigs (tree_dropped : ℕ) : 
  tree_dropped = (additional_twigs_per_base * base_twigs) * 1/3 →
  (base_twigs * additional_twigs_per_base - tree_dropped) = 48 :=
by
  sorry

end bird_needs_more_twigs_l314_314295


namespace edward_initial_money_l314_314140

theorem edward_initial_money (initial_cost_books : ℝ) (discount_percent : ℝ) (num_pens : ℕ) 
  (cost_per_pen : ℝ) (money_left : ℝ) : 
  initial_cost_books = 40 → discount_percent = 0.25 → num_pens = 3 → cost_per_pen = 2 → money_left = 6 → 
  (initial_cost_books * (1 - discount_percent) + num_pens * cost_per_pen + money_left) = 42 :=
by
  sorry

end edward_initial_money_l314_314140


namespace find_speed_of_current_l314_314307

variable {m c : ℝ}

theorem find_speed_of_current
  (h1 : m + c = 15)
  (h2 : m - c = 10) :
  c = 2.5 :=
sorry

end find_speed_of_current_l314_314307


namespace current_population_is_15336_l314_314254

noncomputable def current_population : ℝ :=
  let growth_rate := 1.28
  let future_population : ℝ := 25460.736
  let years := 2
  future_population / (growth_rate ^ years)

theorem current_population_is_15336 :
  current_population = 15536 := sorry

end current_population_is_15336_l314_314254


namespace distance_from_center_to_line_of_tangent_circle_l314_314184

theorem distance_from_center_to_line_of_tangent_circle 
  (a : ℝ) (ha : 0 < a) 
  (h_circle : (2 - a)^2 + (1 - a)^2 = a^2)
  (h_tangent : ∀ x y : ℝ, x = 0 ∨ y = 0): 
  (|2 * a - a - 3| / ((2:ℝ)^2 + (-1)^2).sqrt) = (2 * (5:ℝ).sqrt) / 5 :=
by
  sorry

end distance_from_center_to_line_of_tangent_circle_l314_314184


namespace gcd_lcm_product_360_l314_314052

theorem gcd_lcm_product_360 :
  ∃ (d : ℕ), (∃ x y : ℕ, x * y = 360 ∧ d = Nat.gcd x y) ∧ 
  {d | ∃ x y : ℕ, x * y = 360 ∧ d = Nat.gcd x y}.to_finset.card = 8 := 
sorry

end gcd_lcm_product_360_l314_314052


namespace blue_to_red_face_area_ratio_l314_314443

theorem blue_to_red_face_area_ratio :
  let original_cube_dim := 13
  let red_face_area := 6 * original_cube_dim^2
  let total_faces := 6 * original_cube_dim^3
  let blue_face_area := total_faces - red_face_area
  (blue_face_area / red_face_area) = 12 :=
by
  sorry

end blue_to_red_face_area_ratio_l314_314443


namespace total_wire_length_l314_314303

theorem total_wire_length
  (A B C D E : ℕ)
  (hA : A = 16)
  (h_ratio : 4 * A = 5 * B ∧ 4 * A = 7 * C ∧ 4 * A = 3 * D ∧ 4 * A = 2 * E)
  (hC : C = B + 8) :
  (A + B + C + D + E) = 84 := 
sorry

end total_wire_length_l314_314303


namespace blue_red_face_area_ratio_l314_314437

theorem blue_red_face_area_ratio : 
  let n := 13
  let total_red_faces := 6 * n^2
  let total_faces := 6 * n^3
  let total_blue_faces := total_faces - total_red_faces
  let ratio := total_blue_faces / total_red_faces
  ratio = 12 :=
by
  let n := 13
  let total_red_faces := 6 * n^2
  let total_faces := 6 * n^3
  let total_blue_faces := total_faces - total_red_faces
  let ratio := total_blue_faces / total_red_faces
  have h : ratio = (6 * n^3 - 6 * n^2) / (6 * n^2) := rfl
  have h2 : ratio = (6 * n^2 * (n - 1)) / (6 * n^2) := by rw h
  have h3 : ratio = n - 1 := by rw [mul_div_cancel' (6 * n^2) (6 * n^2), h2]
  exact h3
  simp only [n]
  sorry

end blue_red_face_area_ratio_l314_314437


namespace mark_money_left_l314_314396

theorem mark_money_left (initial_money : ℕ) (cost_book1 cost_book2 cost_book3 : ℕ) (n_book1 n_book2 n_book3 : ℕ) 
  (total_cost : ℕ) (money_left : ℕ) 
  (h1 : initial_money = 85)
  (h2 : cost_book1 = 7)
  (h3 : n_book1 = 3)
  (h4 : cost_book2 = 5)
  (h5 : n_book2 = 4)
  (h6 : cost_book3 = 9)
  (h7 : n_book3 = 2)
  (h8 : total_cost = 21 + 20 + 18)
  (h9 : money_left = initial_money - total_cost):
  money_left = 26 := by
  sorry

end mark_money_left_l314_314396


namespace Ryan_funding_goal_l314_314238

theorem Ryan_funding_goal 
  (avg_fund_per_person : ℕ := 10) 
  (people_recruited : ℕ := 80)
  (pre_existing_fund : ℕ := 200) :
  (avg_fund_per_person * people_recruited + pre_existing_fund = 1000) :=
by
  sorry

end Ryan_funding_goal_l314_314238


namespace sum_of_coefficients_of_y_terms_l314_314332

theorem sum_of_coefficients_of_y_terms: 
  let p := (5 * x + 3 * y + 2) * (2 * x + 5 * y + 3)
  ∃ (a b c: ℝ), p = (10 * x^2 + a * x * y + 19 * x + b * y^2 + c * y + 6) ∧ a + b + c = 65 :=
by
  sorry

end sum_of_coefficients_of_y_terms_l314_314332


namespace initial_percentage_reduction_l314_314452

theorem initial_percentage_reduction
  (x: ℕ)
  (h1: ∀ P: ℝ, P * (1 - x / 100) * 0.85 * 1.5686274509803921 = P) :
  x = 25 :=
by
  sorry

end initial_percentage_reduction_l314_314452


namespace apples_total_l314_314369

theorem apples_total (apples_per_person : ℝ) (number_of_people : ℝ) (h_apples : apples_per_person = 15.0) (h_people : number_of_people = 3.0) : 
  apples_per_person * number_of_people = 45.0 := by
  sorry

end apples_total_l314_314369


namespace cubes_difference_l314_314354

theorem cubes_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 65) (h3 : a + b = 6) : a^3 - b^3 = 432.25 :=
by
  sorry

end cubes_difference_l314_314354


namespace monkey_ladder_min_rungs_l314_314310

/-- 
  Proof that the minimum number of rungs n that allows the monkey to climb 
  to the top of the ladder and return to the ground, given that the monkey 
  ascends 16 rungs or descends 9 rungs at a time, is 24. 
-/
theorem monkey_ladder_min_rungs (n : ℕ) (ascend descend : ℕ) 
  (h1 : ascend = 16) (h2 : descend = 9) 
  (h3 : (∃ x y : ℤ, 16 * x - 9 * y = n) ∧ 
        (∃ x' y' : ℤ, 16 * x' - 9 * y' = 0)) : 
  n = 24 :=
sorry

end monkey_ladder_min_rungs_l314_314310


namespace negation_statement_l314_314176

theorem negation_statement (x y : ℝ) (h : x ^ 2 + y ^ 2 ≠ 0) : ¬ (x = 0 ∧ y = 0) :=
sorry

end negation_statement_l314_314176


namespace circle_sum_value_l314_314381

-- Define the problem
theorem circle_sum_value (a b x : ℕ) (h1 : a = 35) (h2 : b = 47) : x = a + b :=
by
  -- Given conditions
  have ha : a = 35 := h1
  have hb : b = 47 := h2
  -- Prove that the value of x is the sum of a and b
  have h_sum : x = a + b := sorry
  -- Assert the value of x is 82 based on given a and b
  exact h_sum

end circle_sum_value_l314_314381


namespace ratio_of_area_of_inscribed_and_circumscribed_octagons_l314_314115

noncomputable def area_ratio_of_octagons (r : ℝ) : ℝ :=
  let cos22_5 := (√(2 + √2) / 2)
  let r_prime := r / cos22_5
  let area_ratio := (r_prime / r)^2
  area_ratio

theorem ratio_of_area_of_inscribed_and_circumscribed_octagons (r : ℝ) :
  area_ratio_of_octagons r = (4 - 2 * √2) / 2 := by
  sorry

end ratio_of_area_of_inscribed_and_circumscribed_octagons_l314_314115


namespace circle_center_line_distance_l314_314198

noncomputable def distance_point_to_line (x₁ y₁ A B C : ℝ) : ℝ :=
|A * x₁ + B * y₁ + C| / Real.sqrt (A^2 + B^2)

theorem circle_center_line_distance (a : ℝ) (h : a^2 - 6 * a + 5 = 0) :
  distance_point_to_line a a 2 (-1) (-3) = (2 * Real.sqrt 5 / 5) :=
by
  sorry

end circle_center_line_distance_l314_314198


namespace trains_meeting_distance_l314_314266

theorem trains_meeting_distance :
  ∃ D T : ℕ, (D = 20 * T) ∧ (D + 60 = 25 * T) ∧ (2 * D + 60 = 540) :=
by
  sorry

end trains_meeting_distance_l314_314266


namespace solve_quadratic_l314_314242

theorem solve_quadratic : ∃ x : ℚ, x > 0 ∧ 3 * x^2 + 7 * x - 20 = 0 ∧ x = 5/3 := 
by
  sorry

end solve_quadratic_l314_314242
