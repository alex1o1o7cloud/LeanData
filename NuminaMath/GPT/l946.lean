import Mathlib

namespace football_cost_correct_l946_94624

def cost_marble : ℝ := 9.05
def cost_baseball : ℝ := 6.52
def total_cost : ℝ := 20.52
def cost_football : ℝ := total_cost - cost_marble - cost_baseball

theorem football_cost_correct : cost_football = 4.95 := 
by
  -- The proof is omitted, as per instructions.
  sorry

end football_cost_correct_l946_94624


namespace eval_expression_l946_94619

theorem eval_expression : (3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 + (2 * Real.sqrt 3 / 3)) :=
by sorry

end eval_expression_l946_94619


namespace complement_of_A_eq_l946_94630

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x > 1}

theorem complement_of_A_eq {U : Set ℝ} (U_eq : U = Set.univ) {A : Set ℝ} (A_eq : A = {x | x > 1}) :
    U \ A = {x | x ≤ 1} :=
by
  sorry

end complement_of_A_eq_l946_94630


namespace least_multiplier_l946_94685

theorem least_multiplier (x: ℕ) (h1: 72 * x % 112 = 0) (h2: ∀ y, 72 * y % 112 = 0 → x ≤ y) : x = 14 :=
sorry

end least_multiplier_l946_94685


namespace lion_cubs_per_month_l946_94640

theorem lion_cubs_per_month
  (initial_lions : ℕ)
  (final_lions : ℕ)
  (months : ℕ)
  (lions_dying_per_month : ℕ)
  (net_increase : ℕ)
  (x : ℕ) : 
  initial_lions = 100 → 
  final_lions = 148 → 
  months = 12 → 
  lions_dying_per_month = 1 → 
  net_increase = 48 → 
  12 * (x - 1) = net_increase → 
  x = 5 := by
  intros initial_lions_eq final_lions_eq months_eq lions_dying_eq net_increase_eq equation
  sorry

end lion_cubs_per_month_l946_94640


namespace minimize_average_comprehensive_cost_l946_94662

theorem minimize_average_comprehensive_cost :
  ∀ (f : ℕ → ℝ), (∀ (x : ℕ), x ≥ 10 → f x = 560 + 48 * x + 10800 / x) →
  ∃ x : ℕ, x = 15 ∧ ( ∀ y : ℕ, y ≥ 10 → f y ≥ f 15 ) :=
by
  sorry

end minimize_average_comprehensive_cost_l946_94662


namespace fraction_is_three_halves_l946_94654

theorem fraction_is_three_halves (a b : ℝ) (hb : b ≠ 0) (h : 2 * a = 3 * b) : a / b = 3 / 2 :=
sorry

end fraction_is_three_halves_l946_94654


namespace parabola_line_intersection_l946_94682

/-- 
Given the parabola y^2 = -x and the line l: y = k(x + 1) intersect at points A and B,
(Ⅰ) Find the range of values for k;
(Ⅱ) Let O be the vertex of the parabola, prove that OA ⟂ OB.
-/
theorem parabola_line_intersection (k : ℝ) (A B : ℝ × ℝ)
  (hA : A.2 ^ 2 = -A.1) (hB : B.2 ^ 2 = -B.1)
  (hlineA : A.2 = k * (A.1 + 1)) (hlineB : B.2 = k * (B.1 + 1)) :
  (k ≠ 0) ∧ ((A.2 * B.2 = -1) → A.1 * B.1 * (A.2 * B.2) = -1) :=
by
  sorry

end parabola_line_intersection_l946_94682


namespace ellipse_foci_distance_l946_94693

theorem ellipse_foci_distance 
  (a b : ℝ) 
  (h_a : a = 8) 
  (h_b : b = 3) : 
  2 * (Real.sqrt (a^2 - b^2)) = 2 * Real.sqrt 55 := 
by
  rw [h_a, h_b]
  sorry

end ellipse_foci_distance_l946_94693


namespace tomatoes_picked_yesterday_l946_94665

/-
Given:
1. The farmer initially had 171 tomatoes.
2. The farmer picked some tomatoes yesterday (Y).
3. The farmer picked 30 tomatoes today.
4. The farmer will have 7 tomatoes left after today.

Prove:
The number of tomatoes the farmer picked yesterday (Y) is 134.
-/

theorem tomatoes_picked_yesterday (Y : ℕ) (h : 171 - Y - 30 = 7) : Y = 134 :=
sorry

end tomatoes_picked_yesterday_l946_94665


namespace find_m_minus_n_l946_94649

noncomputable def m_abs := 4
noncomputable def n_abs := 6

theorem find_m_minus_n (m n : ℝ) (h1 : |m| = m_abs) (h2 : |n| = n_abs) (h3 : |m + n| = m + n) : m - n = -2 ∨ m - n = -10 :=
sorry

end find_m_minus_n_l946_94649


namespace max_value_expression_l946_94646

noncomputable def max_expression (a b c : ℝ) : ℝ :=
  (a * b * c * (a + b + c)) / ((a + b)^2 * (b + c)^3)

theorem max_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  max_expression a b c ≤ 1 / 12 := 
sorry

end max_value_expression_l946_94646


namespace power_function_k_values_l946_94676

theorem power_function_k_values (k : ℝ) :
  (∃ (a : ℝ), (k^2 - k - 5) = a ∧ (∀ x : ℝ, (k^2 - k - 5) * x^3 = a * x^3)) →
  (k = 3 ∨ k = -2) :=
by
  intro h
  sorry

end power_function_k_values_l946_94676


namespace stickers_in_either_not_both_l946_94614

def stickers_shared := 12
def emily_total_stickers := 22
def mia_unique_stickers := 10

theorem stickers_in_either_not_both : 
  (emily_total_stickers - stickers_shared) + mia_unique_stickers = 20 :=
by
  sorry

end stickers_in_either_not_both_l946_94614


namespace unique_solution_l946_94608

noncomputable def f (a b x : ℝ) := 2 * (a + b) * Real.exp (2 * x) + 2 * a * b
noncomputable def g (a b x : ℝ) := 4 * Real.exp (2 * x) + a + b

theorem unique_solution (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  ∃! x, f a b x = ( (a^(1/3) + b^(1/3))/2 )^3 * g a b x :=
sorry

end unique_solution_l946_94608


namespace age_difference_is_16_l946_94642

-- Variables
variables (y : ℕ) -- y represents the present age of the younger person

-- Conditions from the problem
def elder_present_age := 30
def elder_age_6_years_ago := elder_present_age - 6
def younger_age_6_years_ago := y - 6

-- Given condition 6 years ago:
def condition_6_years_ago := elder_age_6_years_ago = 3 * younger_age_6_years_ago

-- The theorem to prove the difference in ages is 16 years
theorem age_difference_is_16
  (h1 : elder_present_age = 30)
  (h2 : condition_6_years_ago) :
  elder_present_age - y = 16 :=
by sorry

end age_difference_is_16_l946_94642


namespace curve_not_parabola_l946_94690

theorem curve_not_parabola (k : ℝ) : ¬ (∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c = 1 ∧ a * x^2 + b * y = c) :=
sorry

end curve_not_parabola_l946_94690


namespace train_passing_time_l946_94660

theorem train_passing_time (L : ℕ) (v_kmph : ℕ) (v_mps : ℕ) (time : ℕ)
  (h1 : L = 90)
  (h2 : v_kmph = 36)
  (h3 : v_mps = v_kmph * (1000 / 3600))
  (h4 : v_mps = 10)
  (h5 : time = L / v_mps) :
  time = 9 := by
  sorry

end train_passing_time_l946_94660


namespace part1_l946_94696

theorem part1 : 2 * Real.tan (60 * Real.pi / 180) * Real.cos (30 * Real.pi / 180) - (Real.sin (45 * Real.pi / 180)) ^ 2 = 5 / 2 := 
sorry

end part1_l946_94696


namespace intersection_eq_l946_94606

def S : Set ℝ := { x | x > -2 }
def T : Set ℝ := { x | -4 ≤ x ∧ x ≤ 1 }

theorem intersection_eq : S ∩ T = { x | -2 < x ∧ x ≤ 1 } :=
by
  simp [S, T]
  sorry

end intersection_eq_l946_94606


namespace number_of_kids_stayed_home_is_668278_l946_94615

  def number_of_kids_who_stayed_home : Prop :=
    ∃ X : ℕ, X + 150780 = 819058 ∧ X = 668278

  theorem number_of_kids_stayed_home_is_668278 : number_of_kids_who_stayed_home :=
    sorry
  
end number_of_kids_stayed_home_is_668278_l946_94615


namespace length_of_bridge_l946_94692

theorem length_of_bridge
    (speed_kmh : Real)
    (time_minutes : Real)
    (speed_cond : speed_kmh = 5)
    (time_cond : time_minutes = 15) :
    let speed_mmin := speed_kmh * 1000 / 60
    let distance_m := speed_mmin * time_minutes
    distance_m = 1250 :=
by
    sorry

end length_of_bridge_l946_94692


namespace distance_after_one_hour_l946_94655

-- Definitions representing the problem's conditions
def initial_distance : ℕ := 20
def speed_athos : ℕ := 4
def speed_aramis : ℕ := 5

-- The goal is to prove that the possible distances after one hour are among the specified values
theorem distance_after_one_hour :
  ∃ d : ℕ, d = 11 ∨ d = 29 ∨ d = 21 ∨ d = 19 :=
sorry -- proof not required as per the instructions

end distance_after_one_hour_l946_94655


namespace scientific_notation_of_105000_l946_94697

theorem scientific_notation_of_105000 : (105000 : ℝ) = 1.05 * 10^5 := 
by {
  sorry
}

end scientific_notation_of_105000_l946_94697


namespace feet_in_mile_l946_94699

theorem feet_in_mile (d t : ℝ) (speed_mph : ℝ) (speed_fps : ℝ) (miles_to_feet : ℝ) (hours_to_seconds : ℝ) :
  d = 200 → t = 4 → speed_mph = 34.09 → miles_to_feet = 5280 → hours_to_seconds = 3600 → 
  speed_fps = d / t → speed_fps = speed_mph * miles_to_feet / hours_to_seconds → 
  miles_to_feet = 5280 :=
by
  intros hd ht hspeed_mph hmiles_to_feet hhours_to_seconds hspeed_fps_eq hconversion
  -- You can add the proof steps here.
  sorry

end feet_in_mile_l946_94699


namespace radius_of_circle_centered_at_l946_94610

def center : ℝ × ℝ := (3, 4)

def intersects_axes_at_three_points (A : ℝ × ℝ) (r : ℝ) : Prop :=
  (A.1 - r = 0 ∨ A.1 + r = 0) ∧ (A.2 - r = 0 ∨ A.2 + r = 0)

theorem radius_of_circle_centered_at (A : ℝ × ℝ) : 
  (intersects_axes_at_three_points A 4) ∨ (intersects_axes_at_three_points A 5) :=
by
  sorry

end radius_of_circle_centered_at_l946_94610


namespace min_value_f_l946_94638

open Real

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (exp x - 1) + x

theorem min_value_f {x0 : ℝ} (hx0 : 0 < x0) (hx0_min : ∀ x > 0, f x ≥ f x0) :
  f x0 = x0 + 1 ∧ f x0 < 3 :=
by sorry

end min_value_f_l946_94638


namespace factor_sum_l946_94647

variable (x y : ℝ)

theorem factor_sum :
  let a := 1
  let b := -2
  let c := 1
  let d := 2
  let e := 4
  let f := 1
  let g := 2
  let h := 1
  let j := -2
  let k := 4
  (27 * x^9 - 512 * y^9) = ((a * x + b * y) * (c * x^3 + d * x * y^2 + e * y^3) * 
  (f * x + g * y) * (h * x^3 + j * x * y^2 + k * y^3)) → 
  (a + b + c + d + e + f + g + h + j + k = 12) :=
by
  sorry

end factor_sum_l946_94647


namespace polynomial_sum_l946_94644

theorem polynomial_sum :
  let f := (x^3 + 9*x^2 + 26*x + 24) 
  let g := (x + 3)
  let A := 1
  let B := 6
  let C := 8
  let D := -3
  (y = f/g) → (A + B + C + D = 12) :=
by 
  sorry

end polynomial_sum_l946_94644


namespace circumradius_of_sector_l946_94602

noncomputable def R_circumradius (θ : ℝ) (r : ℝ) := r / (2 * Real.sin (θ / 2))

theorem circumradius_of_sector (r : ℝ) (θ : ℝ) (hθ : θ = 120) (hr : r = 8) :
  R_circumradius θ r = (8 * Real.sqrt 3) / 3 :=
by
  rw [hθ, hr, R_circumradius]
  sorry

end circumradius_of_sector_l946_94602


namespace seventh_numbers_sum_l946_94695

def first_row_seq (n : ℕ) : ℕ := n^2 + n - 1

def second_row_seq (n : ℕ) : ℕ := n * (n + 1) / 2

theorem seventh_numbers_sum :
  first_row_seq 7 + second_row_seq 7 = 83 :=
by
  -- Skipping the proof
  sorry

end seventh_numbers_sum_l946_94695


namespace inequality_1_inequality_2_l946_94603

theorem inequality_1 (x : ℝ) : 4 * x + 5 ≤ 2 * (x + 1) → x ≤ -3/2 :=
by
  sorry

theorem inequality_2 (x : ℝ) : (2 * x - 1) / 3 - (9 * x + 2) / 6 ≤ 1 → x ≥ -2 :=
by
  sorry

end inequality_1_inequality_2_l946_94603


namespace hemisphere_surface_area_l946_94686

theorem hemisphere_surface_area (r : ℝ) (h : r = 10) : 
  (4 * Real.pi * r^2) / 2 + (Real.pi * r^2) = 300 * Real.pi := by
  sorry

end hemisphere_surface_area_l946_94686


namespace fuel_tank_capacity_l946_94604

theorem fuel_tank_capacity (C : ℝ) (h1 : 0.12 * 106 + 0.16 * (C - 106) = 30) : C = 214 :=
by
  sorry

end fuel_tank_capacity_l946_94604


namespace find_m_l946_94645

theorem find_m (m : ℕ) (h1 : 0 ≤ m ∧ m ≤ 9) (h2 : (8 + 4 + 5 + 9) - (6 + m + 3 + 7) % 11 = 0) : m = 9 :=
by
  sorry

end find_m_l946_94645


namespace profit_growth_equation_l946_94620

noncomputable def profitApril : ℝ := 250000
noncomputable def profitJune : ℝ := 360000
noncomputable def averageMonthlyGrowth (x : ℝ) : ℝ := 25 * (1 + x) * (1 + x)

theorem profit_growth_equation (x : ℝ) :
  averageMonthlyGrowth x = 36 * 10000 ↔ 25 * (1 + x)^2 = 36 :=
by
  sorry

end profit_growth_equation_l946_94620


namespace abes_age_after_x_years_l946_94667

-- Given conditions
def A : ℕ := 28
def sum_condition (x : ℕ) : Prop := (A + (A - x) = 35)

-- Proof statement
theorem abes_age_after_x_years
  (x : ℕ)
  (h : sum_condition x) :
  (A + x = 49) :=
  sorry

end abes_age_after_x_years_l946_94667


namespace actual_area_of_park_l946_94698

-- Definitions of given conditions
def map_scale : ℕ := 250 -- scale: 1 inch = 250 miles
def map_length : ℕ := 6 -- length on map in inches
def map_width : ℕ := 4 -- width on map in inches

-- Definition of actual lengths
def actual_length : ℕ := map_length * map_scale -- actual length in miles
def actual_width : ℕ := map_width * map_scale -- actual width in miles

-- Theorem to prove the actual area
theorem actual_area_of_park : actual_length * actual_width = 1500000 := by
  -- By the conditions provided, the actual length and width in miles can be calculated directly:
  -- actual_length = 6 * 250 = 1500
  -- actual_width = 4 * 250 = 1000
  -- actual_area = 1500 * 1000 = 1500000
  sorry

end actual_area_of_park_l946_94698


namespace line_circle_intersection_range_l946_94631

theorem line_circle_intersection_range (b : ℝ) :
    (2 - Real.sqrt 2) < b ∧ b < (2 + Real.sqrt 2) ↔
    ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧ ((p1.1 - 2)^2 + p1.2^2 = 1) ∧ ((p2.1 - 2)^2 + p2.2^2 = 1) ∧ (p1.2 = p1.1 - b ∧ p2.2 = p2.1 - b) :=
by
  sorry

end line_circle_intersection_range_l946_94631


namespace bicycle_has_four_wheels_l946_94669

variables (Car : Type) (Bicycle : Car) (FourWheeled : Car → Prop)
axiom car_four_wheels : ∀ (c : Car), FourWheeled c

theorem bicycle_has_four_wheels : FourWheeled Bicycle :=
by {
  apply car_four_wheels
}

end bicycle_has_four_wheels_l946_94669


namespace find_A_l946_94684

variable {a b : ℝ}

theorem find_A (h : (5 * a + 3 * b)^2 = (5 * a - 3 * b)^2 + A) : A = 60 * a * b :=
sorry

end find_A_l946_94684


namespace park_maple_trees_total_l946_94612

theorem park_maple_trees_total (current_maples planted_maples : ℕ) 
    (h1 : current_maples = 2) (h2 : planted_maples = 9) 
    : current_maples + planted_maples = 11 := 
by
  sorry

end park_maple_trees_total_l946_94612


namespace prime_square_mod_30_l946_94657

theorem prime_square_mod_30 (p : ℕ) (hp : Nat.Prime p) (hp_gt_5 : p > 5) : 
  p^2 % 30 = 1 ∨ p^2 % 30 = 19 := 
sorry

end prime_square_mod_30_l946_94657


namespace problem1_problem2_l946_94618

-- Problem 1
theorem problem1 : 40 + ((1 / 6) - (2 / 3) + (3 / 4)) * 12 = 43 :=
by
  sorry

-- Problem 2
theorem problem2 : (-1) ^ 2 * (-5) + ((-3) ^ 2 + 2 * (-5)) = 4 :=
by
  sorry

end problem1_problem2_l946_94618


namespace number_of_insects_l946_94694

theorem number_of_insects (total_legs : ℕ) (legs_per_insect : ℕ) (h1 : total_legs = 48) (h2 : legs_per_insect = 6) : (total_legs / legs_per_insect) = 8 := by
  sorry

end number_of_insects_l946_94694


namespace box_volume_l946_94683

theorem box_volume (initial_length initial_width cut_length : ℕ)
  (length_condition : initial_length = 13) (width_condition : initial_width = 9)
  (cut_condition : cut_length = 2) : 
  (initial_length - 2 * cut_length) * (initial_width - 2 * cut_length) * cut_length = 90 := 
by
  sorry

end box_volume_l946_94683


namespace total_distance_traveled_l946_94639

def speed := 60  -- Jace drives 60 miles per hour
def first_leg_time := 4  -- Jace drives for 4 hours straight
def break_time := 0.5  -- Jace takes a 30-minute break (0.5 hours)
def second_leg_time := 9  -- Jace drives for another 9 hours straight

def distance (speed : ℕ) (time : ℕ) : ℕ := speed * time  -- Distance formula

theorem total_distance_traveled : 
  distance speed first_leg_time + distance speed second_leg_time = 780 := by
-- Sorry allows us to skip the proof, since only the statement is required.
sorry

end total_distance_traveled_l946_94639


namespace Norine_retire_age_l946_94672

theorem Norine_retire_age:
  ∀ (A W : ℕ),
    (A = 50) →
    (W = 19) →
    (A + W = 85) →
    (A = 50 + 8) :=
by
  intros A W hA hW hAW
  sorry

end Norine_retire_age_l946_94672


namespace min_races_required_to_determine_top_3_horses_l946_94635

def maxHorsesPerRace := 6
def totalHorses := 30
def possibleConditions := "track conditions and layouts change for each race"

noncomputable def minRacesToDetermineTop3 : Nat :=
  7

-- Problem Statement: Prove that given the conditions on track and race layout changes,
-- the minimum number of races needed to confidently determine the top 3 fastest horses is 7.
theorem min_races_required_to_determine_top_3_horses 
  (maxHorsesPerRace : Nat := 6) 
  (totalHorses : Nat := 30)
  (possibleConditions : String := "track conditions and layouts change for each race") :
  minRacesToDetermineTop3 = 7 :=
  sorry

end min_races_required_to_determine_top_3_horses_l946_94635


namespace arrangement_proof_l946_94633

/-- The Happy Valley Zoo houses 5 chickens, 3 dogs, and 6 cats in a large exhibit area
    with separate but adjacent enclosures. We need to find the number of ways to place
    the 14 animals in a row of 14 enclosures, ensuring all animals of each type are together,
    and that chickens are always placed before cats, but with no restrictions regarding the
    placement of dogs. -/
def number_of_arrangements : ℕ :=
  let chickens := 5
  let dogs := 3
  let cats := 6
  let chicken_permutations := Nat.factorial chickens
  let dog_permutations := Nat.factorial dogs
  let cat_permutations := Nat.factorial cats
  let group_arrangements := 3 -- Chickens-Dogs-Cats, Dogs-Chickens-Cats, Chickens-Cats-Dogs
  group_arrangements * chicken_permutations * dog_permutations * cat_permutations

theorem arrangement_proof : number_of_arrangements = 1555200 :=
by 
  sorry

end arrangement_proof_l946_94633


namespace range_of_a_l946_94617

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x^2 + 2*a*x + a) > 0) → (0 < a ∧ a < 1) :=
sorry

end range_of_a_l946_94617


namespace rachel_colored_pictures_l946_94629

theorem rachel_colored_pictures :
  ∃ b1 b2 : ℕ, b1 = 23 ∧ b2 = 32 ∧ ∃ remaining: ℕ, remaining = 11 ∧ (b1 + b2) - remaining = 44 :=
by
  sorry

end rachel_colored_pictures_l946_94629


namespace point_on_angle_bisector_l946_94658

theorem point_on_angle_bisector (a b : ℝ) (h : (a, b) = (b, a)) : a = b ∨ a = -b := 
by
  sorry

end point_on_angle_bisector_l946_94658


namespace race_length_l946_94656

theorem race_length (A_time : ℕ) (diff_distance diff_time : ℕ) (A_time_eq : A_time = 380)
  (diff_distance_eq : diff_distance = 50) (diff_time_eq : diff_time = 20) :
  let B_speed := diff_distance / diff_time
  let B_time := A_time + diff_time
  let race_length := B_speed * B_time
  race_length = 1000 := 
by
  sorry

end race_length_l946_94656


namespace trains_meet_distance_l946_94607

noncomputable def time_difference : ℝ :=
  5 -- Time difference between two departures in hours

noncomputable def speed_train_a : ℝ :=
  30 -- Speed of Train A in km/h

noncomputable def speed_train_b : ℝ :=
  40 -- Speed of Train B in km/h

noncomputable def distance_train_a : ℝ :=
  speed_train_a * time_difference -- Distance covered by Train A before Train B starts

noncomputable def relative_speed : ℝ :=
  speed_train_b - speed_train_a -- Relative speed of Train B with respect to Train A

noncomputable def catch_up_time : ℝ :=
  distance_train_a / relative_speed -- Time taken for Train B to catch up with Train A

noncomputable def distance_from_delhi : ℝ :=
  speed_train_b * catch_up_time -- Distance from Delhi where the two trains will meet

theorem trains_meet_distance :
  distance_from_delhi = 600 := by
  sorry

end trains_meet_distance_l946_94607


namespace gcd_2728_1575_l946_94651

theorem gcd_2728_1575 : Int.gcd 2728 1575 = 1 :=
by sorry

end gcd_2728_1575_l946_94651


namespace smallest_repeating_block_fraction_3_over_11_l946_94637

theorem smallest_repeating_block_fraction_3_over_11 :
  ∃ n : ℕ, (∃ d : ℕ, (3/11 : ℚ) = d / 10^n) ∧ n = 2 :=
sorry

end smallest_repeating_block_fraction_3_over_11_l946_94637


namespace polynomial_expansion_l946_94643

theorem polynomial_expansion (a_0 a_1 a_2 a_3 a_4 : ℤ)
  (h1 : a_0 + a_1 + a_2 + a_3 + a_4 = 5^4)
  (h2 : a_0 - a_1 + a_2 - a_3 + a_4 = 1) :
  (a_0 + a_2 + a_4)^2 - (a_1 + a_3)^2 = 625 :=
by
  sorry

end polynomial_expansion_l946_94643


namespace cos_neg_pi_over_3_eq_one_half_sin_eq_sqrt3_over_2_solutions_l946_94641

noncomputable def cos_negative_pi_over_3 : Real :=
  Real.cos (-Real.pi / 3)

theorem cos_neg_pi_over_3_eq_one_half :
  cos_negative_pi_over_3 = 1 / 2 :=
  by
    sorry

noncomputable def solutions_sin_eq_sqrt3_over_2 (x : Real) : Prop :=
  Real.sin x = Real.sqrt 3 / 2 ∧ 0 ≤ x ∧ x < 2 * Real.pi

theorem sin_eq_sqrt3_over_2_solutions :
  {x : Real | solutions_sin_eq_sqrt3_over_2 x} = {Real.pi / 3, 2 * Real.pi / 3} :=
  by
    sorry

end cos_neg_pi_over_3_eq_one_half_sin_eq_sqrt3_over_2_solutions_l946_94641


namespace vegetables_sold_ratio_l946_94605

def totalMassInstalled (carrots zucchini broccoli : ℕ) : ℕ := carrots + zucchini + broccoli

def massSold (soldMass : ℕ) : ℕ := soldMass

def vegetablesSoldRatio (carrots zucchini broccoli soldMass : ℕ) : ℚ :=
  soldMass / (carrots + zucchini + broccoli)

theorem vegetables_sold_ratio
  (carrots zucchini broccoli soldMass : ℕ)
  (h_carrots : carrots = 15)
  (h_zucchini : zucchini = 13)
  (h_broccoli : broccoli = 8)
  (h_soldMass : soldMass = 18) :
  vegetablesSoldRatio carrots zucchini broccoli soldMass = 1 / 2 := by
  sorry

end vegetables_sold_ratio_l946_94605


namespace claire_balance_after_week_l946_94661

theorem claire_balance_after_week :
  ∀ (gift_card : ℝ) (latte_cost croissant_cost : ℝ) (days : ℕ) (cookie_cost : ℝ) (cookies : ℕ),
  gift_card = 100 ∧
  latte_cost = 3.75 ∧
  croissant_cost = 3.50 ∧
  days = 7 ∧
  cookie_cost = 1.25 ∧
  cookies = 5 →
  (gift_card - (days * (latte_cost + croissant_cost) + cookie_cost * cookies) = 43) :=
by
  -- Skipping proof details with sorry
  sorry

end claire_balance_after_week_l946_94661


namespace cos_of_sum_eq_one_l946_94674

theorem cos_of_sum_eq_one
  (x y : ℝ)
  (a : ℝ)
  (h1 : x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4))
  (h2 : y ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4))
  (h3 : x^3 + Real.sin x - 2 * a = 0)
  (h4 : 4 * y^3 + Real.sin y * Real.cos y + a = 0) :
  Real.cos (x + 2 * y) = 1 := 
by
  sorry

end cos_of_sum_eq_one_l946_94674


namespace symmetric_circle_eq_l946_94636

theorem symmetric_circle_eq :
  ∀ (x y : ℝ),
  ((x + 2)^2 + y^2 = 5) →
  (x - y + 1 = 0) →
  (∃ (a b : ℝ), ((a + 1)^2 + (b + 1)^2 = 5)) := 
by
  intros x y h_circle h_line
  -- skip the proof
  sorry

end symmetric_circle_eq_l946_94636


namespace number_of_rectangles_required_l946_94653

theorem number_of_rectangles_required
  (width : ℝ) (area : ℝ) (total_length : ℝ) (length : ℝ)
  (H1 : width = 42) (H2 : area = 1638) (H3 : total_length = 390) (H4 : length = area / width)
  : (total_length / length) = 10 := 
sorry

end number_of_rectangles_required_l946_94653


namespace sum_a1_a11_l946_94678

theorem sum_a1_a11 
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 : ℤ) 
  (h1 : a_0 = -512) 
  (h2 : -2 = a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 + a_11) 
  : a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 + a_11 = 510 :=
sorry

end sum_a1_a11_l946_94678


namespace largest_four_digit_number_mod_l946_94679

theorem largest_four_digit_number_mod (n : ℕ) : 
  (n < 10000) → 
  (n % 11 = 2) → 
  (n % 7 = 4) → 
  n ≤ 9973 :=
by
  sorry

end largest_four_digit_number_mod_l946_94679


namespace purchase_costs_10_l946_94648

def total_cost (a b c d e : ℝ) := a + b + c + d + e
def cost_dates (a : ℝ) := 3 * a
def cost_cantaloupe (a b : ℝ) := a - b
def cost_eggs (b c : ℝ) := b + c

theorem purchase_costs_10 (a b c d e : ℝ) 
  (h_total_cost : total_cost a b c d e = 30)
  (h_cost_dates : d = cost_dates a)
  (h_cost_cantaloupe : c = cost_cantaloupe a b)
  (h_cost_eggs : e = cost_eggs b c) :
  b + c + e = 10 :=
by
  have := h_total_cost
  have := h_cost_dates
  have := h_cost_cantaloupe
  have := h_cost_eggs
  sorry

end purchase_costs_10_l946_94648


namespace jenny_stamps_l946_94675

theorem jenny_stamps :
  let num_books := 8
  let pages_per_book := 42
  let stamps_per_page := 6
  let new_stamps_per_page := 10
  let complete_books_in_new_system := 4
  let pages_in_fifth_book := 33
  (num_books * pages_per_book * stamps_per_page) % new_stamps_per_page = 6 :=
by
  sorry

end jenny_stamps_l946_94675


namespace minimum_g7_l946_94622

def is_tenuous (g : ℕ → ℤ) : Prop :=
∀ x y : ℕ, 0 < x → 0 < y → g x + g y > x^2

noncomputable def min_possible_value_g7 (g : ℕ → ℤ) (h : is_tenuous g) 
  (h_sum : (g 1 + g 2 + g 3 + g 4 + g 5 + g 6 + g 7 + g 8 + g 9 + g 10) = 
             -29) : ℤ :=
g 7

theorem minimum_g7 (g : ℕ → ℤ) (h : is_tenuous g)
  (h_sum : (g 1 + g 2 + g 3 + g 4 + g 5 + g 6 + g 7 + g 8 + g 9 + g 10) = 
             -29) :
  min_possible_value_g7 g h h_sum = 49 :=
sorry

end minimum_g7_l946_94622


namespace solve_equation_real_l946_94677

theorem solve_equation_real (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) (h3 : x ≠ 4) :
  (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 2) * (x - 1) / ((x - 4) * (x - 2) * (x - 1)) = 1 ↔
  x = (9 + Real.sqrt 5) / 2 ∨ x = (9 - Real.sqrt 5) / 2 :=
by  
  sorry

end solve_equation_real_l946_94677


namespace symmetrical_shapes_congruent_l946_94627

theorem symmetrical_shapes_congruent
  (shapes : Type)
  (is_symmetrical : shapes → shapes → Prop)
  (congruent : shapes → shapes → Prop)
  (symmetrical_implies_equal_segments : ∀ (s1 s2 : shapes), is_symmetrical s1 s2 → ∀ (segment : ℝ), segment_s1 = segment_s2)
  (symmetrical_implies_equal_angles : ∀ (s1 s2 : shapes), is_symmetrical s1 s2 → ∀ (angle : ℝ), angle_s1 = angle_s2) :
  ∀ (s1 s2 : shapes), is_symmetrical s1 s2 → congruent s1 s2 :=
by
  sorry

end symmetrical_shapes_congruent_l946_94627


namespace parabola_hyperbola_focus_l946_94673

theorem parabola_hyperbola_focus (p : ℝ) (h : p > 0) :
  (∀ x y : ℝ, (y ^ 2 = 2 * p * x) ∧ (x ^ 2 / 4 - y ^ 2 / 5 = 1) → p = 6) :=
by
  sorry

end parabola_hyperbola_focus_l946_94673


namespace smallest_positive_period_of_f_max_min_values_of_f_in_interval_l946_94681

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (Real.cos x, -1 / 2)
noncomputable def vec_b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos (2 * x))
noncomputable def f (x : ℝ) : ℝ := (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2

theorem smallest_positive_period_of_f :
  (∀ x : ℝ, f (x + Real.pi) = f x) ∧ (∀ T > 0, (∀ x : ℝ, f (x + T) = f x) → T ≥ Real.pi) :=
by sorry

theorem max_min_values_of_f_in_interval :
  ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 1 ∧ f x ≥ -1 / 2 :=
by sorry

end smallest_positive_period_of_f_max_min_values_of_f_in_interval_l946_94681


namespace caitlin_bracelets_l946_94659

-- Define the conditions
def twice_as_many_small_beads (x y : Nat) : Prop :=
  y = 2 * x

def total_large_small_beads (total large small : Nat) : Prop :=
  total = large + small ∧ large = small

def bracelet_beads (large_beads_per_bracelet small_beads_per_bracelet large_per_bracelet : Nat) : Prop :=
  small_beads_per_bracelet = 2 * large_per_bracelet

def total_bracelets (total_large_beads large_per_bracelet bracelets : Nat) : Prop :=
  bracelets = total_large_beads / large_per_bracelet

-- The theorem to be proved
theorem caitlin_bracelets (total_beads large_per_bracelet small_per_bracelet : Nat) (bracelets : Nat) :
    total_beads = 528 ∧
    large_per_bracelet = 12 ∧
    twice_as_many_small_beads large_per_bracelet small_per_bracelet ∧
    total_large_small_beads total_beads 264 264 ∧
    bracelet_beads large_per_bracelet small_per_bracelet 12 ∧
    total_bracelets 264 12 bracelets
  → bracelets = 22 := by
  sorry

end caitlin_bracelets_l946_94659


namespace coterminal_angle_l946_94609

theorem coterminal_angle :
  ∀ θ : ℤ, (θ - 60) % 360 = 0 → θ = -300 ∨ θ = -60 ∨ θ = 600 ∨ θ = 1380 :=
by
  sorry

end coterminal_angle_l946_94609


namespace contradiction_assumption_l946_94671

-- Proposition P: "Among a, b, c, d, at least one is negative"
def P (a b c d : ℝ) : Prop :=
  a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0

-- Correct assumption when using contradiction: all are non-negative
def notP (a b c d : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0

-- Proof problem statement: assuming notP leads to contradiction to prove P
theorem contradiction_assumption (a b c d : ℝ) (h : ¬ P a b c d) : notP a b c d :=
by
  sorry

end contradiction_assumption_l946_94671


namespace larry_result_is_correct_l946_94680

theorem larry_result_is_correct (a b c d e : ℤ) 
  (h1: a = 2) (h2: b = 4) (h3: c = 3) (h4: d = 5) (h5: e = -15) :
  a - (b - (c * (d + e))) = (-17 + e) :=
by 
  rw [h1, h2, h3, h4, h5]
  sorry

end larry_result_is_correct_l946_94680


namespace rate_per_meter_eq_2_5_l946_94687

-- Definitions of the conditions
def diameter : ℝ := 14
def total_cost : ℝ := 109.96

-- The theorem to be proven
theorem rate_per_meter_eq_2_5 (π : ℝ) (hπ : π = 3.14159) : 
  diameter = 14 ∧ total_cost = 109.96 → (109.96 / (π * 14)) = 2.5 :=
by
  sorry

end rate_per_meter_eq_2_5_l946_94687


namespace angle_ABC_measure_l946_94650

theorem angle_ABC_measure
  (angle_CBD : ℝ)
  (angle_sum_around_B : ℝ)
  (angle_ABD : ℝ)
  (h1 : angle_CBD = 90)
  (h2 : angle_sum_around_B = 200)
  (h3 : angle_ABD = 60) :
  ∃ angle_ABC : ℝ, angle_ABC = 50 :=
by
  sorry

end angle_ABC_measure_l946_94650


namespace amc_inequality_l946_94613

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)

theorem amc_inequality : (a / (b + c) + b / (a + c) + c / (a + b)) ≥ 3 / 2 :=
sorry

end amc_inequality_l946_94613


namespace find_x_l946_94625

theorem find_x (x : ℝ) : 
  3.5 * ( (3.6 * 0.48 * 2.50) / (0.12 * 0.09 * x) ) = 2800.0000000000005 → x = 1.25 :=
by
  sorry

end find_x_l946_94625


namespace remainder_b_91_mod_49_l946_94621

def b (n : ℕ) := 12^n + 14^n

theorem remainder_b_91_mod_49 : (b 91) % 49 = 38 := by
  sorry

end remainder_b_91_mod_49_l946_94621


namespace ratio_of_number_halving_l946_94670

theorem ratio_of_number_halving (x y : ℕ) (h1 : y = x / 2) (h2 : y = 9) : x / y = 2 :=
by
  sorry

end ratio_of_number_halving_l946_94670


namespace inequality_solution_l946_94666

theorem inequality_solution (x : ℝ) (h : 1 - x > x - 1) : x < 1 :=
sorry

end inequality_solution_l946_94666


namespace percentage_reduction_price_increase_l946_94626

-- Part 1: Prove the percentage reduction 
theorem percentage_reduction (P0 P1 : ℝ) (r : ℝ) (hp0 : P0 = 50) (hp1 : P1 = 32) :
  P1 = P0 * (1 - r) ^ 2 → r = 1 - 2 * Real.sqrt 2 / 5 :=
by
  intro h
  rw [hp0, hp1] at h
  sorry

-- Part 2: Prove the required price increase
theorem price_increase (G p0 V0 y : ℝ) (hp0 : p0 = 10) (hV0 : V0 = 500) (hG : G = 6000) (hy_range : 0 < y ∧ y ≤ 8):
  G = (p0 + y) * (V0 - 20 * y) → y = 5 :=
by
  intro h
  rw [hp0, hV0, hG] at h
  sorry

end percentage_reduction_price_increase_l946_94626


namespace twenty_seven_cubes_volume_l946_94691

def volume_surface_relation (x V S : ℝ) : Prop :=
  V = x^3 ∧ S = 6 * x^2 ∧ V + S = (4 / 3) * (12 * x)

theorem twenty_seven_cubes_volume (x : ℝ) (hx : volume_surface_relation x (x^3) (6 * x^2)) : 
  27 * (x^3) = 216 :=
by
  sorry

end twenty_seven_cubes_volume_l946_94691


namespace lock_combination_l946_94628

def valid_combination (T I D E b : ℕ) : Prop :=
  (T > 0) ∧ (I > 0) ∧ (D > 0) ∧ (E > 0) ∧
  (T ≠ I) ∧ (T ≠ D) ∧ (T ≠ E) ∧ (I ≠ D) ∧ (I ≠ E) ∧ (D ≠ E) ∧
  (T * b^3 + I * b^2 + D * b + E) + 
  (E * b^3 + D * b^2 + I * b + T) + 
  (T * b^3 + I * b^2 + D * b + E) = 
  (D * b^3 + I * b^2 + E * b + T)

theorem lock_combination : ∃ (T I D E b : ℕ), valid_combination T I D E b ∧ (T * 100 + I * 10 + D = 984) :=
sorry

end lock_combination_l946_94628


namespace contractor_engaged_days_l946_94668

/-
  Prove that the contractor was engaged for 30 days given that:
  1. The contractor receives Rs. 25 for each day he works.
  2. The contractor is fined Rs. 7.50 for each day he is absent.
  3. The contractor gets Rs. 425 in total.
  4. The contractor was absent for 10 days.
-/

theorem contractor_engaged_days
  (wage_per_day : ℕ) (fine_per_absent_day : ℕ) (total_amount_received : ℕ) (absent_days : ℕ) 
  (total_days_engaged : ℕ) :
  wage_per_day = 25 →
  fine_per_absent_day = 7500 / 1000 → -- Rs. 7.50 = 7500 / 1000 (to avoid float, using integers)
  total_amount_received = 425 →
  absent_days = 10 →
  total_days_engaged = (total_amount_received + absent_days * fine_per_absent_day) / wage_per_day + absent_days →
  total_days_engaged = 30 :=
by
  sorry

end contractor_engaged_days_l946_94668


namespace probability_rain_weekend_l946_94652

theorem probability_rain_weekend :
  let p_rain_saturday := 0.30
  let p_rain_sunday := 0.60
  let p_rain_sunday_given_rain_saturday := 0.40
  let p_no_rain_saturday := 1 - p_rain_saturday
  let p_no_rain_sunday_given_no_rain_saturday := 1 - p_rain_sunday
  let p_no_rain_both_days := p_no_rain_saturday * p_no_rain_sunday_given_no_rain_saturday
  let p_rain_sunday_given_rain_saturday := 1 - p_rain_sunday_given_rain_saturday
  let p_no_rain_sunday_given_rain_saturday := p_rain_saturday * p_rain_sunday_given_rain_saturday
  let p_no_rain_all_scenarios := p_no_rain_both_days + p_no_rain_sunday_given_rain_saturday
  let p_rain_weekend := 1 - p_no_rain_all_scenarios
  p_rain_weekend = 0.54 :=
sorry

end probability_rain_weekend_l946_94652


namespace trig_inequality_l946_94664
open Real

theorem trig_inequality (α β γ x y z : ℝ) 
  (h1 : α + β + γ = π)
  (h2 : x + y + z = 0) :
  y * z * (sin α)^2 + z * x * (sin β)^2 + x * y * (sin γ)^2 ≤ 0 := 
sorry

end trig_inequality_l946_94664


namespace russom_greatest_number_of_envelopes_l946_94689

theorem russom_greatest_number_of_envelopes :
  ∃ n, n > 0 ∧ 18 % n = 0 ∧ 12 % n = 0 ∧ ∀ m, m > 0 ∧ 18 % m = 0 ∧ 12 % m = 0 → m ≤ n :=
sorry

end russom_greatest_number_of_envelopes_l946_94689


namespace Sophia_fraction_finished_l946_94623

/--
Sophia finished a fraction of a book.
She calculated that she finished 90 more pages than she has yet to read.
Her book is 270.00000000000006 pages long.
Prove that the fraction of the book she finished is 2/3.
-/
theorem Sophia_fraction_finished :
  let total_pages : ℝ := 270.00000000000006
  let yet_to_read : ℝ := (total_pages - 90) / 2
  let finished_pages : ℝ := yet_to_read + 90
  finished_pages / total_pages = 2 / 3 :=
by
  sorry

end Sophia_fraction_finished_l946_94623


namespace correct_statement_l946_94601

theorem correct_statement :
  (∃ (A : Prop), A = (2 * x^3 - 4 * x - 3 ≠ 3)) ∧
  (∃ (B : Prop), B = ((2 + 3) ≠ 6)) ∧
  (∃ (C : Prop), C = (-4 * x^2 * y = -4)) ∧
  (∃ (D : Prop), D = (1 = 1 ∧ 1 = 1 / 8)) →
  (C) :=
by sorry

end correct_statement_l946_94601


namespace deductive_vs_inductive_l946_94688

def is_inductive_reasoning (stmt : String) : Prop :=
  match stmt with
  | "C" => True
  | _ => False

theorem deductive_vs_inductive (A B C D : String) 
  (hA : A = "All trigonometric functions are periodic functions, sin(x) is a trigonometric function, therefore sin(x) is a periodic function.")
  (hB : B = "All odd numbers cannot be divided by 2, 525 is an odd number, therefore 525 cannot be divided by 2.")
  (hC : C = "From 1=1^2, 1+3=2^2, 1+3+5=3^2, it follows that 1+3+…+(2n-1)=n^2 (n ∈ ℕ*)")
  (hD : D = "If two lines are parallel, the corresponding angles are equal. If ∠A and ∠B are corresponding angles of two parallel lines, then ∠A = ∠B.") :
  is_inductive_reasoning C :=
by
  sorry

end deductive_vs_inductive_l946_94688


namespace sumata_family_total_miles_l946_94616

theorem sumata_family_total_miles
  (days : ℝ) (miles_per_day : ℝ)
  (h1 : days = 5.0)
  (h2 : miles_per_day = 250) : 
  miles_per_day * days = 1250 := 
by
  sorry

end sumata_family_total_miles_l946_94616


namespace sum_of_A_and_B_l946_94611

theorem sum_of_A_and_B (A B : ℕ) (h1 : A ≠ B) (h2 : A < 10) (h3 : B < 10) :
  (10 * A + B) * 6 = 111 * B → A + B = 11 :=
by
  intros h
  sorry

end sum_of_A_and_B_l946_94611


namespace remove_brackets_l946_94663

-- Define the variables a, b, and c
variables (a b c : ℝ)

-- State the theorem
theorem remove_brackets (a b c : ℝ) : a - (b - c) = a - b + c := 
sorry

end remove_brackets_l946_94663


namespace garden_perimeter_l946_94632

theorem garden_perimeter (L B : ℕ) (hL : L = 100) (hB : B = 200) : 
  2 * (L + B) = 600 := by
sorry

end garden_perimeter_l946_94632


namespace water_spilled_l946_94634

theorem water_spilled (x s : ℕ) (h1 : s = x + 7) : s = 8 := by
  -- The proof would go here
  sorry

end water_spilled_l946_94634


namespace line_circle_no_intersection_l946_94600

theorem line_circle_no_intersection : 
  ∀ (x y : ℝ), 3 * x + 4 * y ≠ 12 ∧ x^2 + y^2 = 4 :=
by
  sorry

end line_circle_no_intersection_l946_94600
