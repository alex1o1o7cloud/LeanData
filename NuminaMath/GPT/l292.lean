import Mathlib

namespace george_speed_downhill_l292_292636

-- Definitions 
def total_distance_miles : ℝ := 2
def distance_uphill_miles : ℝ := 1
def speed_uphill_mph : ℝ := 2.5
def total_time_hours : ℝ := 1

-- Theorem statement
theorem george_speed_downhill (total_distance distance_uphill speed_uphill total_time : ℝ)
    (h_total_distance : total_distance = total_distance_miles)
    (h_distance_uphill : distance_uphill = distance_uphill_miles)
    (h_speed_uphill : speed_uphill = speed_uphill_mph)
    (h_total_time : total_time = total_time_hours) :
    let time_uphill := distance_uphill / speed_uphill in
    let time_remaining := total_time - time_uphill in
    distance_uphill / time_remaining = 1.67 :=
by
  sorry

end george_speed_downhill_l292_292636


namespace tan_A_tan_B_l292_292404

theorem tan_A_tan_B (ABC : Type) [triangle ABC] (H : orthocenter ABC) (A B C : ABC)
  (HF HC : ℝ) (h_eq : HF = 9) (h_eq2 : HC = 24) (angle_A : angle A = 45) :
  (tan (angle A) * tan (angle B)) = 27 / 196 :=
sorry

end tan_A_tan_B_l292_292404


namespace factorial_expression_l292_292598

theorem factorial_expression : 10.factorial - 8.factorial + 6.factorial = 3589200 := by
  sorry

end factorial_expression_l292_292598


namespace function_domain_l292_292252

theorem function_domain (x : ℝ) (y : ℝ) (h : y = sqrt ((x - 1) * (x - 2)) + sqrt (x - 1)) :
    (x = 1 ∨ x ≥ 2) ↔ ((x - 1) * (x - 2) ≥ 0 ∧ x - 1 ≥ 0) :=
by
  sorry

end function_domain_l292_292252


namespace conj_in_fourth_quadrant_l292_292740

theorem conj_in_fourth_quadrant :
  let z := i / (1 + 2 * i : ℂ) in
  let z_conj := complex.conj z in
  z_conj.re > 0 ∧ z_conj.im < 0 :=
by
  sorry

end conj_in_fourth_quadrant_l292_292740


namespace pyramid_ratio_l292_292027

noncomputable def midpoints (A B C D M F K P Q: Point) (BC AD CD: Line): Prop :=
  is_midpoint M BC ∧ is_midpoint F AD ∧ is_midpoint K CD ∧
  (∃ AM CF : Line, on_line P AM ∧ on_line Q CF ∧ parallel PQ BK)

noncomputable def ratio (PQ BK: Segment) : ℝ := 2 / 5

theorem pyramid_ratio (A B C D M F K P Q: Point) 
  (BC AD CD: Line) (PQ BK: Segment)
  (h: midpoints A B C D M F K P Q BC AD CD) : 
  ratio PQ BK = 2 / 5 :=
sorry

end pyramid_ratio_l292_292027


namespace unique_songs_after_operations_l292_292825

def initial_songs : ℕ := 300
def deleted_songs : ℕ := 80
def new_songs : ℕ := 120
def duplicate_percentage : ℕ := 20

theorem unique_songs_after_operations :
  let remaining_songs := initial_songs - deleted_songs in
  let duplicates := (duplicate_percentage * new_songs) / 100 in
  let unique_new_songs := new_songs - duplicates in
  let total_unique_songs := remaining_songs + unique_new_songs in
  total_unique_songs = 316 :=
by
  sorry

end unique_songs_after_operations_l292_292825


namespace quadratic_sequence_consistency_l292_292211

theorem quadratic_sequence_consistency :
  ∃ (a b c : ℝ), 
    ∀ n : ℕ, (n ≥ 1 ∧ n ≤ 8 → 
      ∃ k : ℕ, k ≥ 0 ∧ k ≤ 7 ∧ 
        let values := [1600, 1764, 1936, 2116, 2304, 2500, 2704, 2916] in
        values.nth(k) = some (a * (n : ℝ)^2 + b * (n : ℝ) + c)) :=
sorry

end quadratic_sequence_consistency_l292_292211


namespace average_speed_sf_l292_292529

variables
  (v d t : ℝ)  -- Representing the average speed to SF, the distance, and time to SF
  (h1 : 42 = (2 * d) / (3 * t))  -- Condition: Average speed of the round trip is 42 mph
  (h2 : t = d / v)  -- Definition of time t in terms of distance and speed

theorem average_speed_sf : v = 63 :=
by
  sorry

end average_speed_sf_l292_292529


namespace sum_of_valid_two_digit_numbers_l292_292290

theorem sum_of_valid_two_digit_numbers :
  let two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99} in
  let valid_ab := {ab : ℕ | ab ∈ two_digit_numbers ∧ ∃ x y : ℕ, 3^(x + y) = 3^x + 3^y + ab} in
  (∑ ab in valid_ab, ab) = 78 :=
by
  sorry

end sum_of_valid_two_digit_numbers_l292_292290


namespace total_time_in_minutes_l292_292133

def rowing_speed_still_water : ℝ := 15
def distance_first_stretch : ℝ := 2
def current_speed_first_stretch : ℝ := 5
def distance_second_stretch : ℝ := 3
def current_speed_second_stretch : ℝ := 7
def distance_last_stretch : ℝ := 1
def current_speed_last_stretch : ℝ := 4

theorem total_time_in_minutes :
  let effective_speed_first_stretch := rowing_speed_still_water + current_speed_first_stretch,
      effective_speed_second_stretch := rowing_speed_still_water + current_speed_second_stretch,
      effective_speed_last_stretch := rowing_speed_still_water + current_speed_last_stretch,
      time_first_stretch := distance_first_stretch / effective_speed_first_stretch,
      time_second_stretch := distance_second_stretch / effective_speed_second_stretch,
      time_last_stretch := distance_last_stretch / effective_speed_last_stretch,
      total_time_hours := time_first_stretch + time_second_stretch + time_last_stretch,
      total_time_minutes := total_time_hours * 60
  in total_time_minutes ≈ 17.34 := by sorry

end total_time_in_minutes_l292_292133


namespace julie_earnings_school_year_l292_292050

-- Conditions
def hours_per_week_summer : ℕ := 40
def weeks_summer : ℕ := 10
def total_earnings_summer : ℕ := 4000
def hourly_increase_rate : ℝ := 1.25
def weeks_school_year : ℕ := 30
def hours_per_week_school_year : ℕ := 10

-- Define the hourly rate during the summer
def hourly_rate_summer : ℝ :=
  total_earnings_summer / (hours_per_week_summer * weeks_summer)

-- Define the new hourly rate for the school year
def hourly_rate_school_year : ℝ :=
  hourly_rate_summer * hourly_increase_rate

-- Define the total earnings during the school year
def total_earnings_school_year : ℝ :=
  hours_per_week_school_year * weeks_school_year * hourly_rate_school_year

-- Claim to be proved
theorem julie_earnings_school_year : total_earnings_school_year = 3750 := by
  sorry

end julie_earnings_school_year_l292_292050


namespace exists_infinite_sequence_same_color_l292_292595

-- Definitions for coloring the integers
def is_red (n : ℕ) : Prop := sorry  -- Definition for red color predicate
def is_blue (n : ℕ) : Prop := sorry  -- Definition for blue color predicate

-- The main theorem statement
theorem exists_infinite_sequence_same_color :
  ∃ (a : ℕ → ℕ), (∀ n : ℕ, a n > 0) ∧ 
  (∀ n : ℕ, a n < a (n + 1)) ∧
  (∀ n : ℕ, 
    let m := n + 1 in 
    (is_red (a n) ∧ is_red ((a n + a m) / 2) ∧ is_red (a m)) ∨
    (is_blue (a n) ∧ is_blue ((a n + a m) / 2) ∧ is_blue (a m))) ∧
  (∀ n : ℕ, (a n + a (n + 1)) % 2 = 0) :=
sorry

end exists_infinite_sequence_same_color_l292_292595


namespace incircle_tangent_sum_l292_292744

open Real

/-- In a right triangle ABC with ∠ACB = 90°, if the incircle of ABC (touching BC, CA, and AB 
at points D, E, and F respectively), intersects AD at P, and ∠BPC = 90°, then AE + AP = PD -/
theorem incircle_tangent_sum 
  (A B C D E F P : Point) -- Points A, B, C, D, E, F, P are given
  (h : ∠ACB = 90°)       -- Angle ACB is 90 degrees
  (tangent_BC : IncircleTangent O A B C D E F) -- O is the incircle tangent to BC, CA, and AB
  (AD_intersect : LineSegment AD P)   -- AD intersects the incircle at P
  (angle_BPC : ∠BPC = 90°)            -- Angle BPC is 90 degrees
  (lengths_AE_AF : AE = AF)            -- AE = AF
  (BD_BF : BD = BF)                    -- BD = BF
  (CD_CE : CD = CE)                    -- CD = CE
  (AP : AP)                            -- AP exists
  (PD : PD)                            -- PD exists
  : AE + AP = PD := 
sorry

end incircle_tangent_sum_l292_292744


namespace original_number_unique_l292_292510

noncomputable def original_number_satisfies_equation (x : ℝ) : Prop :=
  1000 * x = 3 / x

theorem original_number_unique (x : ℝ) (h : original_number_satisfies_equation x) : 
  x = real.sqrt 3 / 1000 := 
by
  sorry

end original_number_unique_l292_292510


namespace at_least_one_divisible_by_3_l292_292798

-- Define a function that describes the properties of the numbers as per conditions.
def circle_99_numbers (numbers: Fin 99 → ℕ) : Prop :=
  ∀ n : Fin 99, let neighbor := (n + 1) % 99 
                in abs (numbers n - numbers neighbor) = 1 ∨ 
                   abs (numbers n - numbers neighbor) = 2 ∨ 
                   (numbers n = 2 * numbers neighbor) ∨ 
                   (numbers neighbor = 2 * numbers n)

theorem at_least_one_divisible_by_3 :
  ∀ (numbers: Fin 99 → ℕ), circle_99_numbers numbers → ∃ n : Fin 99, numbers n % 3 = 0 :=
by
  intro numbers
  intro h
  sorry

end at_least_one_divisible_by_3_l292_292798


namespace number_of_intersections_l292_292075

theorem number_of_intersections (n : ℕ) (h1 : n = 10) (h2 : ∀ (i j : ℕ), i ≠ j → i, j ∈ ({1, 2, 3, ..., 10} : set ℕ) → (∃! x, intersection_point i j)) : 
  number_of_intersections n = 45 := 
sorry

end number_of_intersections_l292_292075


namespace trig_identity_l292_292520

theorem trig_identity : 4 * real.cos (real.pi / 3 - 10 * real.pi / 180) - real.tan (10 * real.pi / 180) = real.sqrt 3 := 
by sorry

end trig_identity_l292_292520


namespace red_cars_count_l292_292517

variable (R B : ℕ)
variable (h1 : R * 8 = 3 * B)
variable (h2 : B = 90)

theorem red_cars_count : R = 33 :=
by
  -- here we would provide the proof
  sorry

end red_cars_count_l292_292517


namespace determine_complement_l292_292353

namespace SetsProof

def U := {a : 'a, b, c, d} : set char := {'a', 'b', 'c', 'd'}
def A := {a : 'a, b} : set char := {'a', 'b'}
def B := {c : 'c} : set char := {'c'}

/- We define the complement of a set within a universal set -/
def comp (U : set char) (S : set char) : set char := {x | x ∈ U ∧ x ∉ S}

theorem determine_complement :
  comp U (A ∪ B) = {'d'} :=
by
  sorry

end SetsProof

end determine_complement_l292_292353


namespace area_inequality_quadrilateral_l292_292421

noncomputable theory

variables (A B C D K L M N : Type) [linear_ordered_field A] [linear_ordered_field B]
[hAB : segment AB ≠ ∅] [hAD : segment AD ≠ ∅] [hAC : segment AC ≠ ∅] [hBC : segment BC ≠ ∅]
[intersection : ∃ K : A, is_intersection (AB, CD) K]

theorem area_inequality_quadrilateral (cond1 : K ∈ [AD]) (cond2 : L ∈ [AC]) (cond3 : M ∈ [BC])
  (kl_parallel_ab : parallel KL AB) (lm_parallel_dc : parallel LM DC) (mn_parallel_ab : parallel MN AB) :
  (area_of_convex_quadrilateral K L M N) / (area_of_convex_quadrilateral A B C D) < 8 / 27 :=
by
  -- proof goes here
  sorry

end area_inequality_quadrilateral_l292_292421


namespace region_relation_l292_292191

theorem region_relation (A B C : ℝ)
  (a b c : ℝ) (h1 : a = 15) (h2 : b = 36) (h3 : c = 39)
  (h_triangle : a^2 + b^2 = c^2)
  (h_right_triangle : true) -- Since the triangle is already confirmed as right-angle
  (h_A : A = (π * (c / 2)^2 / 2 - 270) / 2)
  (h_B : B = (π * (c / 2)^2 / 2 - 270) / 2)
  (h_C : C = π * (c / 2)^2 / 2) :
  A + B + 270 = C :=
by
  sorry

end region_relation_l292_292191


namespace valid_coloring_schemes_l292_292323

def four_points_coloring (A B C D : Type) (colors : set Type) : Prop :=
  colors.card = 4 ∧
  ∀ (P Q : Type), P ∈ colors ∧ Q ∈ colors ∧ P ≠ Q

theorem valid_coloring_schemes : 
  ∃ (A B C D : Type) (colors: set Type), four_points_coloring A B C D colors ∧ (valid_coloring_schemes A B C D colors = 72) := sorry

end valid_coloring_schemes_l292_292323


namespace triangle_ADC_properties_l292_292025

-- Defining the conditions given in the problem
variables {a : ℝ} (A B C D : Type*) [Triangle ABC] [Equilateral ABC] [°] (AD : A → B → C)

-- Definitions used directly only from the problem conditions
def is_equilateral (ABC : Triangle) : Prop :=
  ∀ (a b c : Point), length (side a b) = length (side b c)

def perpendicular (A B : Point) (l m : Line) : Prop :=
  angle l m = 90

def length_AB_equal_a (AB : Segment) (a : ℝ) : Prop :=
  length (side A B) = a

-- Mathematical properties to prove
def angles_of_ADC (∠ADC : Triangle) : Prop :=
  angle ∠ADC D = 150 ∧ angle ∠ADC A = 15 ∧ angle ∠ADC C = 15

def sides_of_ADC (∠ADC : Triangle) : Prop :=
  length (side A D) = a ∧
  length (side A C) = a ∧
  length (side D C) = a * sqrt (2 + sqrt 3)

def area_of_ADC (∠ADC : Triangle) : Prop :=
  area ∠ADC = (1 / 4) * a^2

-- Combining the conditions and what needs to be proved into a single theorem statement
theorem triangle_ADC_properties {ABC : Triangle} (h1 : is_equilateral ABC)
  (h2 : perpendicular AD (side A B)) (h3 : length_AB_equal_a (side A B) a):
  ∃ (ADC : Triangle), angles_of_ADC ADC ∧ sides_of_ADC ADC ∧ area_of_ADC ADC :=
sorry

end triangle_ADC_properties_l292_292025


namespace angle_at_point_P_l292_292455

-- Definitions
variables {P A B C D E F : Type}
variables [hexagon : regular_hexagon A B C D E F]
variables {P_meets_extended_AB_EF : P ∈ extended_meet_of AB EF}

-- Theorem statement
theorem angle_at_point_P : measure_angle P = 60 :=
by
  -- proof goes here
  sorry

end angle_at_point_P_l292_292455


namespace problem1_problem2_problem3_problem4_l292_292653

noncomputable def f1 : ℝ → ℝ := λ x, x^3 - x^2 + 1
noncomputable def f2 : ℝ → ℝ := λ x, x^2 + 1
noncomputable def f3 : ℝ → ℝ := λ x, exp x

def curvature (f : ℝ → ℝ) (x1 x2 : ℝ) : ℝ :=
  let y1 := f x1
  let y2 := f x2
  let kA := (deriv f) x1
  let kB := (deriv f) x2
  let distAB := real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)
  abs (kA - kB) / distAB

theorem problem1 :
  curvature f1 1 2 ≤ real.sqrt 3 :=
sorry

theorem problem2 :
  ∃ f : ℝ → ℝ, ∀ x1 x2 : ℝ, curvature f x1 x2 = 0 :=
begin
  use (λ x, 1),
  intros,
  simp [curvature, deriv, abs],
end

theorem problem3 :
  ∀ x1 x2 : ℝ, x1 ≠ x2 → curvature f2 x1 x2 ≤ 2 :=
sorry

theorem problem4 (x1 x2 : ℝ) (h : x1 - x2 = 1) (t : ℝ) :
  t * curvature f3 x1 x2 < 1 → t < 1 :=
sorry

end problem1_problem2_problem3_problem4_l292_292653


namespace number_of_trips_l292_292410

variables (π : ℝ)
-- Define the volumes
def volume_hemisphere (r : ℝ) : ℝ := (2 / 3) * π * r^3
def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h

-- Given values
def r_bucket : ℝ := 5
def r_container : ℝ := 8
def h_container : ℝ := 24

-- Calculate the required number of trips
theorem number_of_trips : 
  let V_bucket := volume_hemisphere π r_bucket,
      V_container := volume_cylinder π r_container h_container in
  Nat.ceil (V_container / V_bucket) = 19 :=
by
  sorry

end number_of_trips_l292_292410


namespace coloring_four_cells_with_diff_colors_l292_292315

theorem coloring_four_cells_with_diff_colors {n k : ℕ} (h : n ≥ 2) 
    (hk : k = 2 * n) 
    (color : fin n × fin n → fin k) 
    (hcolor : ∀ c, ∃ r c : fin k, ∃ a b : fin k, color (r, c) = a ∧ color (r, c) = b) :
    ∃ r1 r2 c1 c2, color (r1, c1) ≠ color (r1, c2) ∧ color (r1, c1) ≠ color (r2, c1) ∧
                    color (r2, c1) ≠ color (r2, c2) ∧ color (r1, c2) ≠ color (r2, c2) :=
by
  sorry

end coloring_four_cells_with_diff_colors_l292_292315


namespace sufficient_but_not_necessary_condition_l292_292184

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 1 → |x| > 1) ∧ ¬ (|x| > 1 → x > 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l292_292184


namespace parametric_equation_correct_l292_292912

theorem parametric_equation_correct (t : ℝ) :
  let x := 1 - t,
      y := 3 - 2t in
  2 * x - y + 1 = 0 :=
by
  let x := 1 - t
  let y := 3 - 2t
  sorry

end parametric_equation_correct_l292_292912


namespace C_investment_l292_292514

theorem C_investment (A B C_profit total_profit : ℝ) (hA : A = 24000) (hB : B = 32000) (hC_profit : C_profit = 36000) (h_total_profit : total_profit = 92000) (x : ℝ) (h : x / (A + B + x) = C_profit / total_profit) : x = 36000 := 
by
  sorry

end C_investment_l292_292514


namespace total_distance_first_route_l292_292111

example (average_speed_route1 : ℕ) (time_fastest_route : ℕ) : ℕ :=
by
  have D := average_speed_route1 * time_fastest_route
  exact D

-- Define the conditions given in the problem
def average_speed_route1 : ℕ := 75 -- average speed for the first route in MPH
def time_fastest_route : ℕ := 20  -- time for the fastest route in hours

-- Formulate the statement to be proved
theorem total_distance_first_route : ∃ D : ℕ, D = 1500 :=
by
  let D := average_speed_route1 * time_fastest_route
  use D
  show D = 1500, by rfl
  sorry

end total_distance_first_route_l292_292111


namespace flag_pole_height_eq_150_l292_292555

-- Define the conditions
def tree_height : ℝ := 12
def tree_shadow_length : ℝ := 8
def flag_pole_shadow_length : ℝ := 100

-- Problem statement: prove the height of the flag pole equals 150 meters
theorem flag_pole_height_eq_150 :
  ∃ (F : ℝ), (tree_height / tree_shadow_length) = (F / flag_pole_shadow_length) ∧ F = 150 :=
by
  -- Setup the proof scaffold
  have h : (tree_height / tree_shadow_length) = (150 / flag_pole_shadow_length) := by sorry
  exact ⟨150, h, rfl⟩

end flag_pole_height_eq_150_l292_292555


namespace distance_from_SF_to_Atlantis_l292_292218

theorem distance_from_SF_to_Atlantis :
  let sf := (0 : ℂ)
  let miami := (3120 * Complex.I : ℂ)
  let atlantis := (1300 + 3120 * Complex.I : ℂ)
  Complex.abs (atlantis - sf) = 3380 :=
begin
  sorry
end

end distance_from_SF_to_Atlantis_l292_292218


namespace annual_feeding_cost_is_correct_l292_292691

-- Definitions based on conditions
def number_of_geckos : Nat := 3
def number_of_iguanas : Nat := 2
def number_of_snakes : Nat := 4
def cost_per_gecko_per_month : Nat := 15
def cost_per_iguana_per_month : Nat := 5
def cost_per_snake_per_month : Nat := 10

-- Statement of the theorem
theorem annual_feeding_cost_is_correct : 
    (number_of_geckos * cost_per_gecko_per_month
    + number_of_iguanas * cost_per_iguana_per_month 
    + number_of_snakes * cost_per_snake_per_month) * 12 = 1140 := by
  sorry

end annual_feeding_cost_is_correct_l292_292691


namespace time_after_interval_l292_292910

noncomputable def starting_time : String := "2020-02-01T18:00:00"
noncomputable def interval_minutes : Int := 3457
noncomputable def target_time : String := "2020-02-04T03:37:00"

theorem time_after_interval :
  let parsed_starting_time := starting_time -- Parse starting time string to datetime format
  let final_time := add_minutes_to_time(parsed_starting_time, interval_minutes) -- Functionally equivalent action
  final_time = target_time := sorry

end time_after_interval_l292_292910


namespace solve_congruence_l292_292106

-- Define the condition and residue modulo 47
def residue_modulo (a b n : ℕ) : Prop := (a ≡ b [MOD n])

-- The main theorem to be proved
theorem solve_congruence (m : ℕ) (h : residue_modulo (13 * m) 9 47) : residue_modulo m 26 47 :=
sorry

end solve_congruence_l292_292106


namespace point_in_fourth_quadrant_l292_292709

-- Define complex number z and its coordinates
def z : ℂ := 3 - Complex.i
def x : ℝ := 3
def y : ℝ := -1

-- Define the conditions under which we check the quadrant
def point_coordinates : ℂ → ℝ × ℝ 
| z := (x, y)

-- The theorem stating that the point corresponding to z is in the fourth quadrant
theorem point_in_fourth_quadrant : point_coordinates z = (3, -1) → (x > 0 ∧ y < 0) := 
sorry

end point_in_fourth_quadrant_l292_292709


namespace mike_taller_than_mark_l292_292782

def height_mark_feet : ℕ := 5
def height_mark_inches : ℕ := 3
def height_mike_feet : ℕ := 6
def height_mike_inches : ℕ := 1
def feet_to_inches : ℕ := 12

-- Calculate heights in inches.
def height_mark_total_inches : ℕ := height_mark_feet * feet_to_inches + height_mark_inches
def height_mike_total_inches : ℕ := height_mike_feet * feet_to_inches + height_mike_inches

-- Prove the height difference.
theorem mike_taller_than_mark : height_mike_total_inches - height_mark_total_inches = 10 :=
by
  sorry

end mike_taller_than_mark_l292_292782


namespace arithmetic_sequence_formula_and_inequality_l292_292659

noncomputable def a_n (n : ℕ) : ℤ :=
  -1 + (n - 1) * 3

def S_12 : ℤ :=
  12 * a_n 1 + (12 * 11 / 2) * 3

def b_n (n : ℕ) : ℝ :=
  (1/2) ^ a_n n

def T_n (n : ℕ) : ℝ :=
  (2 * (1 - (1/8) ^ n)) / (1 - 1/8)

theorem arithmetic_sequence_formula_and_inequality : 
  (∀ n : ℕ, a_n n = 3 * n - 4) ∧ (∀ (m : ℝ) (n : ℕ), T_n n < m → m ≥ 16 / 7) := by 
  sorry

end arithmetic_sequence_formula_and_inequality_l292_292659


namespace ellipse_and_line_equation_l292_292321

-- Given an ellipse G
def ellipse (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Given conditions
variables (a b c : ℝ)
variable (eccentricity : ℝ := sqrt 6 / 3)
variable (focus_x : ℝ := 2 * sqrt 2)
axiom focus_non_negative: a > b ∧ b > 0

-- Equations
def ellipse_eq : Prop := ellipse x y (2 * sqrt 3) 2
def line_slope_eq : Prop := (x - y + 2) = 0

-- Proof problem
theorem ellipse_and_line_equation (condition1 : focus_x = c)
                                   (condition2 : eccentricity = c/a)
                                   (condition3 : b^2 = a^2 - c^2) 
                                   (condition4 : ∃ m : ℝ, 4 * x^2 + 6 * m * x + 3 * m^2 - 12 = 0 ∧ 2 = m) : 
                                   ellipse_eq ∧ line_slope_eq :=
by sorry

end ellipse_and_line_equation_l292_292321


namespace problem_solution_l292_292474

noncomputable def equation_satisfied (x : ℝ) : Prop :=
  2^(2 * x) - 8 * 2^x + 12 = 0

theorem problem_solution : equation_satisfied (1 + real.log 3 / real.log 2) :=
by sorry

end problem_solution_l292_292474


namespace proof_problem_l292_292778

variable (x y : ℝ)

theorem proof_problem 
  (h1 : 0.30 * x = 0.40 * 150 + 90)
  (h2 : 0.20 * x = 0.50 * 180 - 60)
  (h3 : y = 0.75 * x)
  (h4 : y^2 > x + 100) :
  x = 150 ∧ y = 112.5 :=
by
  sorry

end proof_problem_l292_292778


namespace find_max_value_l292_292423

noncomputable def max_value_expression (α β : ℂ) (h1 : |β| = 1) (h2 : α.conj * β ≠ 1) : ℝ :=
  |(β - α) / (1 - α.conj * β)|

theorem find_max_value (α β : ℂ) (h1 : |β| = 1) (h2 : α.conj * β ≠ 1) :
  max_value_expression α β h1 h2 ≤ 1 :=
sorry

end find_max_value_l292_292423


namespace cubic_polynomial_roots_product_l292_292425

theorem cubic_polynomial_roots_product :
  (∃ a b c : ℝ, (3*a^3 - 9*a^2 + 5*a - 15 = 0) ∧
               (3*b^3 - 9*b^2 + 5*b - 15 = 0) ∧
               (3*c^3 - 9*c^2 + 5*c - 15 = 0) ∧
               a ≠ b ∧ b ≠ c ∧ c ≠ a) →
  ∃ a b c : ℝ, (3*a*b*c = 5) := 
sorry

end cubic_polynomial_roots_product_l292_292425


namespace collinear_points_arithmetic_sum_l292_292690

theorem collinear_points_arithmetic_sum
  (A B C O : Type)
  (collinear : ℝ → ℝ → ℝ → Prop)
  (a : ℕ → ℝ)
  (h1 : collinear A B C) 
  (h2 : ∃ k l : ℝ, ∀ p q r : Type, collinear p q r → a 15 * k + a 24 * l = 1)
  :
  (∑ i in finset.range 38, a (i + 1)) = 19 :=
by
  sorry

end collinear_points_arithmetic_sum_l292_292690


namespace b_share_of_payment_l292_292921

def work_fraction (d : ℕ) : ℚ := 1 / d

def total_one_day_work (a_days b_days c_days : ℕ) : ℚ :=
  work_fraction a_days + work_fraction b_days + work_fraction c_days

def share_of_work (b_days : ℕ) (total_work : ℚ) : ℚ :=
  work_fraction b_days / total_work

def share_of_payment (total_payment : ℚ) (work_share : ℚ) : ℚ :=
  total_payment * work_share

theorem b_share_of_payment 
  (a_days b_days c_days : ℕ) (total_payment : ℚ):
  a_days = 6 → b_days = 8 → c_days = 12 → total_payment = 1800 →
  share_of_payment total_payment (share_of_work b_days (total_one_day_work a_days b_days c_days)) = 600 :=
by
  intros ha hb hc hp
  unfold total_one_day_work work_fraction share_of_work share_of_payment
  rw [ha, hb, hc, hp]
  -- Simplify the fractions and the multiplication
  sorry

end b_share_of_payment_l292_292921


namespace slope_angle_tangent_line_45_degrees_l292_292850

theorem slope_angle_tangent_line_45_degrees :
  let f (x : ℝ) := x^3 - 2 * x + 4
  let df := deriv f
  let k := df 1
  k = 1 → atan k = Real.pi / 4 :=
by
  intro h
  sorry

end slope_angle_tangent_line_45_degrees_l292_292850


namespace problem_T_valid_probability_final_problem_l292_292765

def point (x y z : ℤ) : Prop := 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 2 ∧ 0 ≤ z ∧ z ≤ 5

noncomputable def points : Finset (ℤ × ℤ × ℤ) := 
  { p | ∃ x y z, point x y z ∧ p = (x, y, z) }.to_finset

def midpoint (p₁ p₂ : ℤ × ℤ × ℤ) : ℤ × ℤ × ℤ := 
  ( (p₁.1 + p₂.1) / 2, (p₁.2.1 + p₂.2.1) / 2, (p₁.2.2 + p₂.2.2) / 2 )

def valid_midpoint_pair (p₁ p₂ : ℤ × ℤ × ℤ) : Prop := 
  p₁ ≠ p₂ ∧ (midpoint p₁ p₂) ∈ points

theorem problem_T_valid_probability : 
  let all_pairs := (points ×ˢ points).filter (λ p, p.fst ≠ p.snd) in
  let valid_pairs := all_pairs.filter (λ p, valid_midpoint_pair p.fst p.snd) in
  (valid_pairs.card / all_pairs.card : ℚ) = 3 / 14 :=
sorry

theorem final_problem : 
  (3 : ℕ) + 14 = 17 :=
by norm_num

end problem_T_valid_probability_final_problem_l292_292765


namespace determine_a_range_l292_292477

variables (x a : ℝ)

def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x)
def g (x : ℝ) : ℝ := 2 * Real.cos (2 * x - Real.pi / 3)

theorem determine_a_range (h1 : ∀ x, g x = 2 * Real.cos (2 * x - Real.pi / 3))
  (h2 : ∀ x, 0 ≤ x → x ≤ a / 3 → g x ≤ g (x + 0.1))
  (h3 : ∀ x, 2 * a ≤ x → x ≤ 7 * Real.pi / 6 → g x ≤ g (x + 0.1)) :
  a ∈ Set.Icc (Real.pi / 3) (Real.pi / 2) :=
sorry

end determine_a_range_l292_292477


namespace exists_kings_tour_l292_292986

-- Step (a) conditions represented as definitions or assumptions
def is_8x8_chessboard (b : matrix (fin 8) (fin 8) bool) : Prop := 
  ∀ i j, true

def is_tiled_by_2x1_dominoes (b : matrix (fin 8) (fin 8) bool) : Prop :=
  ∀ i j, true -- simplified; we'd define actual tiling properties

def king_move (pos1 pos2 : (fin 8) × (fin 8)) : Prop :=
  let diff := (fin.ring pos1.1 - fin.ring pos2.1, fin.ring pos1.2 - fin.ring pos2.2) in
  abs diff.1 ≤ 1 ∧ abs diff.2 ≤ 1

def is_valid_tour (tour : list ((fin 8) × (fin 8))) : Prop :=
  (tour.length = 64) ∧ 
  (∀ i, king_move (tour.nth i) (tour.nth (i+1))) ∧
  (∀ d ∈ tour, count tour d = 1)

-- Step (c) & (d): final theorem statement
theorem exists_kings_tour (chessboard : matrix (fin 8) (fin 8) bool) 
  (h1 : is_8x8_chessboard chessboard) 
  (h2 : is_tiled_by_2x1_dominoes chessboard) :
  ∃ tour : list ((fin 8) × (fin 8)), is_valid_tour tour :=
begin
  sorry -- proof will be added here
end

end exists_kings_tour_l292_292986


namespace geometric_series_sum_l292_292587

theorem geometric_series_sum (a r : ℝ) (h : |r| < 1) (h_a : a = 2 / 3) (h_r : r = 2 / 3) :
  ∑' i : ℕ, (a * r^i) = 2 :=
by
  sorry

end geometric_series_sum_l292_292587


namespace find_B_radian_measure_l292_292718

-- Define the geometric progression of angles in the triangle 
variables {A B C a b c : ℝ}

-- Conditions
def conditions (A B C a b c : ℝ) : Prop :=
  (A + B + C = π) ∧           -- Sum of angles in a triangle
  (A = θ) ∧
  (B = q * θ) ∧
  (C = q^2 * θ) ∧
  (b^2 - a^2 = a * c)

-- Proposition to prove
theorem find_B_radian_measure (A B C a b c θ q : ℝ) (h : conditions A B C a b c) : 
  B = 2 * π / 7 :=
sorry

end find_B_radian_measure_l292_292718


namespace ellipse_c_equation_and_slopes_range_l292_292320

theorem ellipse_c_equation_and_slopes_range (a b: ℝ) (M : ℝ × ℝ) (F : ℝ × ℝ) (k : ℝ) (N : ℝ × ℝ)
  (C : ℝ → ℝ → Prop)
  (h1 : a > b) (h2 : b > 0) (h3 : C 1 (3 / 2)) (h4 : F = (-1, 0))
  (h5 : C = (λ x y, (x^2 / a^2 + y^2 / b^2 = 1)))
  (h6 : N = (0, -2))
  (h7 : ∀ x y, (y = k * x + 2) → (C x y . x . y . x . y . true)) :
  C = (λ x y, (x^2 / 4 + y^2 / 3 = 1)) ∧
  (∃ k1 k2 : ℝ, k1 * k2 > 49 / 4) :=
by sorry

end ellipse_c_equation_and_slopes_range_l292_292320


namespace train_speed_l292_292554

theorem train_speed
  (train_length : ℕ) (crossing_time : ℕ)
  (h_train_length : train_length = 300)
  (h_crossing_time : crossing_time = 20) :
  train_length / crossing_time = 15 := by
  rw [h_train_length, h_crossing_time]
  norm_num
  sorry

end train_speed_l292_292554


namespace solve_ranking_problem_l292_292612

-- Define the three girls
inductive Girl
| Hannah
| Cassie
| Bridget

-- Define the ranking relation
def ranking : Girl → Girl → Prop
| Girl.Cassie, Girl.Bridget := true
| Girl.Cassie, Girl.Hannah := true
| Girl.Bridget, Girl.Hannah := true
| _, _ := false

-- Hannah's statement: She didn't get the highest score.
def hannah_not_highest : Prop := ranking Girl.Cassie Girl.Hannah ∨ ranking Girl.Bridget Girl.Hannah

-- Bridget's statement: She didn't get the lowest score.
def bridget_not_lowest : Prop := ranking Girl.Bridget Girl.Hannah ∨ ranking Girl.Cassie Girl.Hannah

-- The correct ranking we want to prove
def correct_ranking : Prop := ranking Girl.Cassie Girl.Bridget ∧ ranking Girl.Bridget Girl.Hannah

theorem solve_ranking_problem :
  hannah_not_highest →
  bridget_not_lowest →
  correct_ranking :=
by
  sorry

end solve_ranking_problem_l292_292612


namespace find_triples_l292_292619

theorem find_triples 
  (x y z : ℝ)
  (h1 : x + y * z = 2)
  (h2 : y + z * x = 2)
  (h3 : z + x * y = 2)
 : (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -2 ∧ y = -2 ∧ z = -2) :=
sorry

end find_triples_l292_292619


namespace triangle_parallel_diagonal_l292_292756

theorem triangle_parallel_diagonal 
  (A B C D E : Point) 
  (h_iso : dist A B = dist A C)
  (h_angle_bisector : isAngleBisector B D A C)
  (h_segment_equal : dist B E = dist C D)
  : isParallel E D A B :=
sorry

end triangle_parallel_diagonal_l292_292756


namespace find_original_number_l292_292213
noncomputable def original_number (a b c : ℕ) (N : ℕ) : Prop :=
  N = 400 + 10 * b + c ∧ 100 * b + 10 * c + 4 = 3 / 4 * N

theorem find_original_number : ∃ N, original_number 4 3 2 N :=
by
  unfold original_number
  existsi 432
  split
  { norm_num }
  { sorry }

end find_original_number_l292_292213


namespace floor_sum_example_l292_292273

theorem floor_sum_example : ⌊18.7⌋ + ⌊-18.7⌋ = -1 := by
  sorry

end floor_sum_example_l292_292273


namespace digits_divisibility_problem_l292_292749

-- Define basic properties
def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

noncomputable def is_possible_combination : Prop :=
  ∃ (ab cde : ℕ),
    is_two_digit_number ab ∧
    is_three_digit_number cde ∧
    (∃ (a b c d e : ℕ),
      {a, b, c, d, e} = {1, 2, 3, 4, 5} ∧
      ab = 10 * a + b ∧
      cde = 100 * c + 10 * d + e ∧
      ab ∣ cde)

theorem digits_divisibility_problem : is_possible_combination := sorry

end digits_divisibility_problem_l292_292749


namespace units_digit_of_pow_sum_is_correct_l292_292908

theorem units_digit_of_pow_sum_is_correct : 
  (24^3 + 42^3) % 10 = 2 := by
  -- Start the proof block
  sorry

end units_digit_of_pow_sum_is_correct_l292_292908


namespace units_digit_sum_cubes_l292_292902

theorem units_digit_sum_cubes (n1 n2 : ℕ) 
  (h1 : n1 = 24) (h2 : n2 = 42) : 
  (n1 ^ 3 + n2 ^ 3) % 10 = 6 :=
by 
  -- substitution based on h1 and h2 can be done here.
  sorry

end units_digit_sum_cubes_l292_292902


namespace a_and_b_divisible_by_p_l292_292762

theorem a_and_b_divisible_by_p
  (p : ℕ) (hp_prime : Nat.Prime p) (hp_form : ∃ k, p = 3 * k + 2)
  (a b : ℤ) (h_div : p ∣ (a^2 + a * b + b^2)) : p ∣ a ∧ p ∣ b :=
by
sorry

end a_and_b_divisible_by_p_l292_292762


namespace no_regular_polygon_in_unequal_ellipse_l292_292548

noncomputable theory

open Real

-- Define necessary conditions and the statement
def regular_polygon (n : ℕ) (vertices : Fin n → Real × Real) : Prop := sorry

def ellipse (a b : Real) : Prop := sorry -- a is the major axis, b is the minor axis

theorem no_regular_polygon_in_unequal_ellipse
  (E : Type)
  (P : Type)
  (n : ℕ)
  -- Assume a regular polygon with more than four sides
  (polygon_conditions : regular_polygon n P)
  -- Assume the ellipse's axes are not equal to each other
  (ellipse_conditions : ∃ (a b : Real), ellipse a b ∧ a ≠ b)
  (n_gt_4 : n > 4) :
  ¬ ∃ (E ellipse : ellipse_conditions), (inscribed_polygon P E) :=
sorry

end no_regular_polygon_in_unequal_ellipse_l292_292548


namespace cos_arcsin_eq_tan_arcsin_eq_l292_292596

open Real

theorem cos_arcsin_eq (h : arcsin (3 / 5) = θ) : cos (arcsin (3 / 5)) = 4 / 5 := by
  sorry

theorem tan_arcsin_eq (h : arcsin (3 / 5) = θ) : tan (arcsin (3 / 5)) = 3 / 4 := by
  sorry

end cos_arcsin_eq_tan_arcsin_eq_l292_292596


namespace largest_divisor_of_n_cubed_minus_n_l292_292100

theorem largest_divisor_of_n_cubed_minus_n (n : ℤ) : ∃ d : ℕ, d = 6 ∧ ∀ k : ℕ, k ∣ (n^3 - n) → k ≤ d :=
sorry

end largest_divisor_of_n_cubed_minus_n_l292_292100


namespace angle_quadrant_l292_292333

theorem angle_quadrant (α : ℝ) (h1 : 0 < α ∧ α < π) (h2 : sin α * tan α < 0) : π/2 < α ∧ α < π :=
by 
  -- We state the theorem with the conditions given.
  sorry

end angle_quadrant_l292_292333


namespace xy_value_l292_292687

theorem xy_value (x y : ℝ) (h1 : x + 2 * y = 8) (h2 : 2 * x + y = -5) : x + y = 1 := 
sorry

end xy_value_l292_292687


namespace contrapositive_l292_292085

variable (P Q : Prop)

theorem contrapositive (h : P → Q) : ¬Q → ¬P :=
sorry

end contrapositive_l292_292085


namespace inequality_proof_l292_292699

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 + 4 * a / (b + c)) * (1 + 4 * b / (a + c)) * (1 + 4 * c / (a + b)) > 25 :=
sorry

end inequality_proof_l292_292699


namespace f_500_l292_292063

-- Define a function f on positive integers
def f (n : ℕ) : ℕ := sorry

-- Assume the given conditions
axiom f_mul (x y : ℕ) (hx : x > 0) (hy : y > 0) : f (x * y) = f x + f y
axiom f_10 : f 10 = 14
axiom f_40 : f 40 = 20

-- Prove the required result
theorem f_500 : f 500 = 39 := by
  sorry

end f_500_l292_292063


namespace total_books_received_l292_292809

theorem total_books_received (initial_books additional_books total_books: ℕ)
  (h1 : initial_books = 54)
  (h2 : additional_books = 23) :
  (initial_books + additional_books = 77) := by
  sorry

end total_books_received_l292_292809


namespace arithmetic_sequence_common_difference_l292_292294

theorem arithmetic_sequence_common_difference (a_1 d : ℝ) (S : ℕ → ℝ)
  (hS2 : S 2 = 4) (hS4 : S 4 = 20)
  (hS_formula : ∀ n, S n = n / 2 * (2 * a_1 + (n - 1) * d)) : 
  d = 3 :=
by sorry

end arithmetic_sequence_common_difference_l292_292294


namespace exists_linear_eq_solution_x_2_l292_292092

theorem exists_linear_eq_solution_x_2 : ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x : ℝ, a * x + b = 0 ↔ x = 2 :=
by
  sorry

end exists_linear_eq_solution_x_2_l292_292092


namespace acute_triangle_bc_ratio_l292_292737

-- Define the conditions and statement
theorem acute_triangle_bc_ratio (A B C : ℝ) (a b c : ℝ)
    (hABC: 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
    (h_ab: a = c * cos B + b * cos C)
    (h_bc: b = c * sin A + a * sin C)
    (h_condition: sin B - sqrt 3 * cos C = (sqrt 3 * (c^2 - a^2)) / (2 * a * b)) :
    1 / 2 < b / c ∧ b / c < 2 := 
sorry

end acute_triangle_bc_ratio_l292_292737


namespace seating_arrangements_correct_l292_292863

noncomputable def seating_arrangements : ℕ :=
  -- Choosing 3 positions out of 5 slots for the 3 people, ensuring empty seats between them.
  let choose_3_out_of_5 := Nat.choose 5 3 in
  -- There are 3! ways to arrange 3 people in the 3 chosen positions.
  let permutations_3 := 3! in
  -- Initial total combinations.
  let initial_total := choose_3_out_of_5 * permutations_3 in
  -- Correct for the overcount when considering end seats.
  let end_corrected := 2 * (Nat.choose 4 2) * 2! in
  -- Total corrected arrangements.
  initial_total - end_corrected

theorem seating_arrangements_correct : seating_arrangements = 36 := by
  sorry

end seating_arrangements_correct_l292_292863


namespace price_of_notebook_and_pen_l292_292928

def notebook_price := ℝ
def pen_price := ℝ

variables (x : notebook_price) (y : pen_price)

-- Conditions
def cond1 : Prop := 3 * x + 5 * y = 30
def cond2 : Prop := 3 * x + 7 * y = 30.4
def cond3 : Prop := x + y = 6.4

-- Problem statement
theorem price_of_notebook_and_pen (hx : cond1 x y) (hy : cond2 x y) (hz : cond3 x y) :
  x = 6.2 ∧ y = 0.2 :=
sorry

end price_of_notebook_and_pen_l292_292928


namespace probability_B_given_A_l292_292949

def die_rolls : Finset (ℕ × ℕ) := 
  Finset.product (Finset.range 6) (Finset.range 6)

def event_A (x : ℕ × ℕ) : Prop := 
  x.1 + x.2 + 2 > 6 -- Adding 2 because die faces are 1-indexed

def event_B (x : ℕ × ℕ) : Prop := 
  x.1 % 2 = 1 ∧ x.2 % 2 = 1 -- Checking if both are even in 0-indexed

def P_event_A : ℚ := 
  (die_rolls.filter event_A).card / die_rolls.card

def P_event_B : ℚ := 
  (die_rolls.filter event_B).card / die_rolls.card

def P_event_AB : ℚ := 
  (die_rolls.filter (λ x, event_A x ∧ event_B x)).card / die_rolls.card

theorem probability_B_given_A : 
  P_event_AB / P_event_A = 2 / 7 :=
sorry

end probability_B_given_A_l292_292949


namespace count_irrationals_l292_292221

def is_irrational (x : ℝ) : Prop := ¬∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

-- Given numbers
def num1 := 22 / 7
def num2 := Real.sqrt 5
def num3 := 0
def num4 := 3.1415
def num5 := Real.sqrt 16
def num6 := 3 * Real.pi
def num7 : ℝ := 0 -- I have removed the complex definition of increasing zeros number since it has to be handled in a more detailed way in Lean.

-- Problem statement
theorem count_irrationals : (is_irrational num2) ∧ (is_irrational num6) ∧ (is_irrational num7) ∧ ¬(is_irrational num1) ∧ ¬(is_irrational num3) ∧ ¬(is_irrational num4) ∧ ¬(is_irrational num5) → 3 = 3 :=
by
  intro h,
  sorry

end count_irrationals_l292_292221


namespace determine_c_l292_292363

theorem determine_c (a c : ℝ) (h : (2 * a - 1) / -3 < - (c + 1) / -4) : c ≠ -1 ∧ (c > 0 ∨ c < 0) :=
by sorry

end determine_c_l292_292363


namespace perimeter_combined_shape_l292_292971

/-- Define the sides of the rectangle and triangle -/
def rect_width : ℝ := 6
def rect_height (x : ℝ) : ℝ := x
def tri_leg1 (x : ℝ) : ℝ := x
def tri_leg2 : ℝ := 6

/-- Hypotenuse of the triangle using Pythagorean theorem -/
def hypotenuse (x : ℝ) : ℝ := Real.sqrt (x^2 + 36)

/-- Perimeter of the combined shape (rectangle + triangle) -/
def combined_perimeter (x : ℝ) : ℝ :=
  2 * (rect_width + rect_height x) + tri_leg2 + hypotenuse x

theorem perimeter_combined_shape (x : ℝ) :
  combined_perimeter x = 18 + 2 * x + Real.sqrt (x^2 + 36) := by
    sorry

end perimeter_combined_shape_l292_292971


namespace investment_amount_l292_292989

def monthly_interest_payment : ℝ := 234
def annual_interest_rate : ℝ := 0.09
def annual_interest_payment := monthly_interest_payment * 12

theorem investment_amount : ∃ P : ℝ, P = 31200 ∧ annual_interest_payment = P * annual_interest_rate * 1 :=
begin
  sorry
end

end investment_amount_l292_292989


namespace estimate_fish_population_l292_292738

theorem estimate_fish_population (n m k : ℕ) (h1 : n > 0) (h2 : m > 0) (h3 : k > 0) (h4 : k ≤ m) : 
  ∃ N : ℕ, N = m * n / k :=
by
  sorry

end estimate_fish_population_l292_292738


namespace truck_passes_car_in_5_5_hours_l292_292939

variables (speed_car speed_truck : ℝ) (head_start : ℝ)

def distance_car := speed_car * head_start
def relative_speed := speed_truck - speed_car

theorem truck_passes_car_in_5_5_hours :
  speed_car = 55 ∧ speed_truck = 65 ∧ head_start = 1 →
  (distance_car speed_car head_start) / (relative_speed speed_truck speed_car) = 5.5 :=
by
  intros h,
  sorry

end truck_passes_car_in_5_5_hours_l292_292939


namespace fraction_of_three_fourths_is_one_fifth_l292_292871

theorem fraction_of_three_fourths_is_one_fifth :
  (∃ x : ℚ, x * (3 / 4) = (1 / 5)) ↔ (x = 4 / 15) :=
begin
  sorry
end

end fraction_of_three_fourths_is_one_fifth_l292_292871


namespace product_of_sequence_l292_292878

theorem product_of_sequence : 
  (∏ (n : ℕ) in finset.range (2009), (5 + n) / (4 + n)) = 502.5 :=
by
  sorry

end product_of_sequence_l292_292878


namespace calculate_nabla_l292_292367

def nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

theorem calculate_nabla :
  nabla (nabla 1 2) 3 = 1 :=
by
  unfold nabla
  have h1 : nabla 1 2 = 1 := by
    simp [nabla]
    rw [add_mul, one_mul, add_comm (1 + 2)]
  rw [h1, nabla]
  simp [nabla]
  rw [add_comm (1 + 3)]
  sorry

end calculate_nabla_l292_292367


namespace zoo_visitors_per_hour_l292_292857

theorem zoo_visitors_per_hour 
    (h1 : ∃ V, 0.80 * V = 320)
    (h2 : ∃ H : Nat, H = 8)
    : ∃ N : Nat, N = 50 :=
by
  sorry

end zoo_visitors_per_hour_l292_292857


namespace expression_value_l292_292007

theorem expression_value (x y : ℤ) (h1 : x = -6) (h2 : y = -3) : (x - y) ^ 2 - x * y = -9 :=
by {
    have hxy : (x - y) ^ 2 = 9, from sorry,
    have hproduct : x * y = 18, from sorry,
    calc
    (x - y) ^ 2 - x * y
        = 9 - 18 : by rw [hxy, hproduct]
    ... = -9   : by norm_num,
}

end expression_value_l292_292007


namespace derivative_of_function_l292_292840

-- Given function
def y (x : ℝ) : ℝ := x^2 * Real.cos x

-- Statement of the problem
theorem derivative_of_function : 
  ∀ x : ℝ, deriv y x = 2 * x * Real.cos x - x^2 * Real.sin x :=
by
  sorry

end derivative_of_function_l292_292840


namespace probability_phenolphthalein_red_l292_292813

-- Define the given conditions
def is_acidic (solution : ℕ) : Prop := solution = 2 ∨ solution = 4
def is_alkaline (solution : ℕ) : Prop := solution = 3 ∨ solution = 5
def is_neutral (solution : ℕ) : Prop := solution = 1

-- Define the problem
def phenolphthalein_turns_red (solution : ℕ) : Prop := is_alkaline solution

-- The given problem with the correct answer translated into a Lean statement
theorem probability_phenolphthalein_red : 
  (∑ solution in Finset.range 6, if phenolphthalein_turns_red solution then 1 else 0).val / 5 = 2 / 5 := 
by
  sorry

end probability_phenolphthalein_red_l292_292813


namespace sum_first_five_terms_is_15_l292_292319

variable {α : Type*} [AddCommGroup α] [Module ℚ α]
variables (a : ℕ → α) (d : α)

-- Assuming the sequence is arithmetic, i.e., a_n = a_1 + (n - 1) * d
def is_arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
∀ n, a (n + 1) = a n + d

-- Given conditions
axiom a3_is_3 : a 3 = 3
axiom arithmetic_seq : is_arithmetic_sequence a d

-- The value of S_5 is 15
theorem sum_first_five_terms_is_15 : a 1 + a 2 + a 3 + a 4 + a 5 = 15 := by
  sorry

end sum_first_five_terms_is_15_l292_292319


namespace systematic_sampling_method_l292_292951

noncomputable def factory_setup (T : Type) : Prop :=
  ∃ (conveyor_belt : T → Prop) (products : T → Prop), 
  ∀ (product : T), products product → conveyor_belt product

noncomputable def inspection_interval (T : Type) : Prop :=
  ∃ (inspectors : T → Prop) (interval : ℕ), (interval = 10 ∧ ∀ (time : ℕ) (product : T), time % interval = 0 → inspectors product)

theorem systematic_sampling_method (T : Type) :
  factory_setup T → inspection_interval T → ∃ method : Prop, method = "systematic sampling" :=
by
  sorry

end systematic_sampling_method_l292_292951


namespace car_avg_speed_l292_292189

theorem car_avg_speed (d t v : ℝ) (h1 : t = d / v) (h2 : average_speed : (2 * d) / (3 * t) = 30) : 
  v = 45 := 
sorry

end car_avg_speed_l292_292189


namespace at_least_one_divisible_by_three_l292_292801

theorem at_least_one_divisible_by_three (n : ℕ) (h1 : n > 0) (h2 : n ≡ 99)
  (h3 : ∀ i, (i < n) → (∃ m : ℤ, abs (m(i+1) - m(i)) = 1 ∨ abs (m(i+1) - m(i)) = 2 ∨ m(i+1) = 2 * m(i))) :
  ∃ k, k ≤ 99 ∧ (k % 3 = 0) := sorry

end at_least_one_divisible_by_three_l292_292801


namespace math_problem_proof_l292_292632

-- Define the condition for the tens digit of n^2 being 1
def tens_digit_is_1 (n : ℕ) : Prop :=
  (n ^ 2 / 10) % 10 = 1

-- Define the set of interest
def valid_numbers : Finset ℕ := Finset.range 151

-- Define the number of such integers
def count_valid_numbers : ℕ := (valid_numbers.filter tens_digit_is_1).card

-- Prove the main statement
theorem math_problem_proof : count_valid_numbers = 15 := by
  sorry

end math_problem_proof_l292_292632


namespace sum_even_positive_integers_less_than_102_l292_292891

theorem sum_even_positive_integers_less_than_102 : 
  let sum_even : ℕ := ∑ n in Finset.filter (λ x, even x) (Finset.range 102), n
  in sum_even = 2550 :=
by
  sorry

end sum_even_positive_integers_less_than_102_l292_292891


namespace angle_CAB_l292_292021

theorem angle_CAB (A B C D : Type) [angle : angle A B C D] 
  (angle_B : angle B = 90) 
  (angle_DAB : angle DAB = 96)
  (angle_D : angle D = 96)
  (angle_BCD : angle BCD = 78)
  (DA_twice_AB : DA = 2 * AB) : 
  angle CAB = 66 :=
by
  sorry

end angle_CAB_l292_292021


namespace total_volume_structure_l292_292233

theorem total_volume_structure (d : ℝ) (h_cone : ℝ) (h_cylinder : ℝ) 
  (r := d / 2) 
  (V_cone := (1 / 3) * π * r^2 * h_cone) 
  (V_cylinder := π * r^2 * h_cylinder) 
  (V_total := V_cone + V_cylinder) :
  d = 8 → h_cone = 9 → h_cylinder = 4 → V_total = 112 * π :=
by
  intros
  sorry

end total_volume_structure_l292_292233


namespace circle_diameter_l292_292534

theorem circle_diameter (r : ℝ) (h : π * r^2 = 9 * π) : 2 * r = 6 :=
by sorry

end circle_diameter_l292_292534


namespace is_perpendicular_dp_an_l292_292420

open EuclideanGeometry

variables {A B C N D P : Point}
variables [IsIsoscelesTriangle A B C]
variables [AngleCondition : ∃ θ, 2 * ∠ N B A + θ = 180 + ∠ A C B]
variables [N_inside : PointInsideTriangle N A B C]
variables [D_intersection : ∃ D, LineThrough D B ∧ LineParallelThrough N C A]
variables [P_bisectors : IsAngleBisector P A N ∧ IsAngleBisector P B N]

theorem is_perpendicular_dp_an :
  LineThrough D P ⊥ LineThrough A N := sorry

end is_perpendicular_dp_an_l292_292420


namespace find_conjugate_coordinates_l292_292338

def complex_conjugate_coordinates (z : ℂ) : ℂ :=
  conj z

theorem find_conjugate_coordinates :
  let z : ℂ := -1 / (complex.I) - 1
    in complex_conjugate_coordinates z = -1 - complex.I :=
by
  simp [complex_conjugate_coordinates, complex.I, complex.conj, complex.ofReal, div_eq_mul_inv, complex.inv_I, mul_neg_eq_neg_mul_symm, sub_eq_add_neg]
  sorry

end find_conjugate_coordinates_l292_292338


namespace max_S_n_of_decreasing_arithmetic_seq_l292_292652

theorem max_S_n_of_decreasing_arithmetic_seq {a_n : ℕ → ℝ} (h1 : ∀ n, a_n.succ < a_n) 
(h2 : ∑ i in (finset.range 5), a_n i = ∑ i in (finset.range 10), a_n i) : 
∀ n, (n = 7 ∨ n = 8) ↔ ∀ m, S m ≤ S n :=
by
  -- the proof is omitted
  sorry

def S (n : ℕ) : ℝ := (finset.range n).sum a_n

end max_S_n_of_decreasing_arithmetic_seq_l292_292652


namespace ten_cubed_largest_odd_number_l292_292633

theorem ten_cubed_largest_odd_number : 
  (∀ n : ℕ, n ≥ 2 → ∃ lst : list ℕ, 
    (list.range (2*n)+1 = lst) ∧ 
    (10^3 = (list.sum lst))) → 
    (list.range (2*54)+1).last! = 109 :=
by {
  sorry
}

end ten_cubed_largest_odd_number_l292_292633


namespace sum_binom_10_eq_1024_l292_292246

def binomial_sum_identity (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 1), Nat.choose n k

theorem sum_binom_10_eq_1024 : binomial_sum_identity 10 = 1024 := by
  sorry

end sum_binom_10_eq_1024_l292_292246


namespace equation_of_asymptotes_of_hyperbola_l292_292335

-- Define the conditions for the hyperbola and the parabola
def hyperbola_asymptotes (a : ℝ) (b : ℝ) (c : ℝ) : Prop := 
  (c = 2) ∧ (c = real.sqrt(a^2 + b^2)) ∧ (a = real.sqrt(3)) ∧ (y = b/a * x)

-- Statement of the theorem
theorem equation_of_asymptotes_of_hyperbola :
  ∀ (a b : ℝ), (a > 0) → (∃ c, (c = 2) ∧ (c = real.sqrt(a^2 + b^2)) ∧ (b = 1)) →
  (y = b / a * x ↔ y = (1/real.sqrt(3)) * x) :=
by
  intros a b ha h
  sorry

end equation_of_asymptotes_of_hyperbola_l292_292335


namespace concurrency_of_lines_l292_292029

open EuclideanGeometry

noncomputable def triangle_not_isosceles (ABC : Triangle) : Prop :=
¬(ABC.isIsosceles)

noncomputable def points_on_side_AC
  (A C P Q : Point) (h : Seg A C)
  (hP : isPointOnSegment P h) (hQ : isPointOnSegment Q h) : Prop :=
∃ (hA : Angle A B P) (hC : Angle Q B C), 
(hA.measure = hC.measure) ∧ (hA.measure < (1 / 2) * hABC.measure)

noncomputable def angle_bisector_intersections
  (A B C P Q K L M N : Point) : Prop :=
  ∃ (line_BP : Line), (isIntersectionPoint K (angleBisector A C) line_BP) ∧
  ∃ (line_BQ : Line), (isIntersectionPoint L (angleBisector A C) line_BP) ∧
  ∃ (line_BQ2 : Line), (isIntersectionPoint M (angleBisector A C) line_BQ2) ∧
  ∃ (line_BP2 : Line), (isIntersectionPoint N (angleBisector A C) line_BQ)

theorem concurrency_of_lines
  (A B C P Q K L M N : Point) 
  (h_triangle_not_isosceles : triangle_not_isosceles (triangle.mk A B C))
  (h_points_on_side_AC : points_on_side_AC A C P Q (Seg.mk A C) (isPointOnSegment P (Seg.mk A C)) (isPointOnSegment Q (Seg.mk A C)))
  (h_angle_bisector_intersections : angle_bisector_intersections A B C P Q K L M N) :
  areConcurrent (line.mk A C) (line.mk K N) (line.mk L M) :=
sorry

end concurrency_of_lines_l292_292029


namespace greatest_distance_between_vertices_l292_292552

theorem greatest_distance_between_vertices 
  (inner_perimeter outer_perimeter : ℝ)
  (h_inner : inner_perimeter = 24)
  (h_outer : outer_perimeter = 32) : 
  let a := inner_perimeter / 4 
  let b := outer_perimeter / 4 
  in (a = 6 ∧ b = 8) → 
  ∃ d, d = 2 * real.sqrt 17 :=
by 
  intros _ _ h_inner h_outer
  have ha : a = 6 := by 
    rw [<-h_inner]
    norm_num
  have hb : b = 8 := by 
    rw [<-h_outer]
    norm_num

  -- Defining the maximum distance
  use 2 * real.sqrt 17
  sorry

end greatest_distance_between_vertices_l292_292552


namespace solve_for_x_l292_292371

-- Definitions based on provided conditions
variables (x : ℝ) -- defining x as a real number
def condition : Prop := 0.25 * x = 0.15 * 1600 - 15

-- The theorem stating that x equals 900 given the condition
theorem solve_for_x (h : condition x) : x = 900 :=
by
  sorry

end solve_for_x_l292_292371


namespace range_of_c_l292_292303

variable {c : ℝ}

def p := ∀ x₁ < x₂, x₁ < x₂ → c ^ x₁ > c ^ x₂
def q := ∃ x : ℝ, 1/2 < x ∧ ∀ y > x, (f(y) := y^2 - 2 * c * y + 1) > f(x)

theorem range_of_c (h1 : c > 0) (h2 : c ≠ 1) (h3 : ¬ (p ∧ q)) (h4 : p ∨ q) :
  1/2 < c ∧ c < 1 := sorry

end range_of_c_l292_292303


namespace descending_order_l292_292328

theorem descending_order (a b c : ℝ) (ha : a = (1/2)^10) (hb : b = (1/5)^(-1/2)) (hc : c = log (1/5) 10) : 
  b > a ∧ a > c := 
by 
  sorry

end descending_order_l292_292328


namespace count_integer_points_on_curve_l292_292844

theorem count_integer_points_on_curve :
  let P := λ x y : Int, (x^2 + y^2 - 1)^3 ≤ x^2 * y^3 in
  (∃ k : Nat, k = 7 ∧ setOf (uncurry P) {p : Int × Int | true}).card = k :=
by
  let P := λ x y : Int, (x^2 + y^2 - 1)^3 ≤ x^2 * y^3
  exists 7
  split
  · reflexivity
  · exact (setOf (uncurry P) {p : Int × Int | true}).card

sorry

end count_integer_points_on_curve_l292_292844


namespace hyperbola_eccentricity_greater_than_sqrt2_l292_292667

noncomputable def hyperbola_eccentricity_range (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c ≥ 0) : Set ℝ :=
  {e | ∃ P : ℝ × ℝ, (∃ x y : ℝ, P = (x, y) ∧ (x^2 / a^2) - (y^2 / b^2) = 1) ∧
                     (euclidean_dist P (-c, 0) = euclidean_dist P (0, c)) ∧
                     e = c / a }

theorem hyperbola_eccentricity_greater_than_sqrt2 (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c ≥ 0)
  (h₃ : ∃ P : ℝ × ℝ, (∃ x y : ℝ, P = (x, y) ∧ (x^2 / a^2) - (y^2 / b^2) = 1) ∧
                      (euclidean_dist P (-c, 0) = euclidean_dist P (0, c))) :
  hyperbola_eccentricity_range a b c h₀ h₁ h₂ ⊆ {e | e > sqrt 2} :=
sorry

end hyperbola_eccentricity_greater_than_sqrt2_l292_292667


namespace BG_eq_2EG_l292_292923

theorem BG_eq_2EG
  {A B C O₁ O₂ D E F P Q G K L N M : Point}
  (h₁ : Tangent O₁ AC at D)
  (h₂ : Tangent O₁ BC at E)
  (h₃ : Tangent O₂ BC at E)
  (h₄ : Tangent O₂ AB at F)
  (h₅ : Tangent O₂ at P where Line_through_points D E ∩ O₂ and P ≠ E)
  (h₆ : Line_through_point P meets AB at Q)
  (h₇ : Line_through_point O₁ parallel to Line_through_points B O₂ meets BC at G)
  (h₈ : Line_through_points E Q ∩ AC = K)
  (h₉ : Line_through_points K G ∩ EF = L)
  (h₁₀ : Line_through_points E O₂ meets O₂ at N and N ≠ E)
  (h₁₁ : Line_through_points L O₂ ∩ FN = M)
  (h₁₂ : Midpoint N F M)
  : BG = 2 * EG :=
sorry

end BG_eq_2EG_l292_292923


namespace interest_difference_l292_292807

theorem interest_difference :
  let P := 10000
  let R := 6
  let T := 2
  let SI := P * R * T / 100
  let CI := P * (1 + R / 100)^T - P
  CI - SI = 36 :=
by
  let P := 10000
  let R := 6
  let T := 2
  let SI := P * R * T / 100
  let CI := P * (1 + R / 100)^T - P
  show CI - SI = 36
  sorry

end interest_difference_l292_292807


namespace gcd_m_n_15_lcm_m_n_45_l292_292426

-- Let m and n be integers greater than 0, and 3m + 2n = 225.
variables (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : 3 * m + 2 * n = 225)

-- First part: If the greatest common divisor of m and n is 15, then m + n = 105.
theorem gcd_m_n_15 (h4 : Int.gcd m n = 15) : m + n = 105 :=
sorry

-- Second part: If the least common multiple of m and n is 45, then m + n = 90.
theorem lcm_m_n_45 (h5 : Int.lcm m n = 45) : m + n = 90 :=
sorry

end gcd_m_n_15_lcm_m_n_45_l292_292426


namespace hours_per_day_for_first_group_l292_292525

theorem hours_per_day_for_first_group (h : ℕ) :
  (39 * h * 12 = 30 * 6 * 26) → h = 10 :=
by
  sorry

end hours_per_day_for_first_group_l292_292525


namespace bridge_length_correct_l292_292553

noncomputable def length_of_bridge 
  (train_length : ℝ) 
  (time_to_cross : ℝ) 
  (train_speed_kmph : ℝ) : ℝ :=
  (train_speed_kmph * (5 / 18) * time_to_cross) - train_length

theorem bridge_length_correct :
  length_of_bridge 120 31.99744020478362 36 = 199.9744020478362 :=
by
  -- Skipping the proof details
  sorry

end bridge_length_correct_l292_292553


namespace sequence_bound_l292_292436

-- Define the conditions
variables (a : ℕ → ℝ)
variables (h_nonneg : ∀ n, 0 ≤ a n)
variables (h_additive : ∀ m n, a (n + m) ≤ a n + a m)

-- The theorem statement
theorem sequence_bound (n m : ℕ) (h : n ≥ m) : 
  a n ≤ m * a 1 + ((n.to_real / m.to_real) - 1) * a m :=
by
  sorry

end sequence_bound_l292_292436


namespace cistern_fill_time_l292_292535

theorem cistern_fill_time (t_a t_b : ℕ) (ha : t_a = 7) (hb : t_b = 9) :
  let fill_rate := 1 / (t_a : ℚ), 
      empty_rate := 1 / (t_b : ℚ),
      net_rate := fill_rate - empty_rate,
      time := 1 / net_rate in
  time = 31.5 :=
by
  sorry

end cistern_fill_time_l292_292535


namespace general_term_is_linear_a20_correct_398_in_sequence_l292_292028

variable (a : ℕ → ℤ)
variable (n : ℕ)

-- Condition: a_1 = 2
def a1 : Prop := a 1 = 2

-- Condition: a_17 = 66
def a17 : Prop := a 17 = 66

-- Condition: General term is a linear function of n
noncomputable def linear_term : Prop := ∃ b c : ℤ, ∀ n : ℕ, a n = b * (n : ℤ) + c

-- General term formula
noncomputable def general_term_formula : Prop := ∀ n : ℕ, a n = 4 * (n : ℤ) - 2

-- Question 1: Prove general term formula is a_n = 4n - 2
theorem general_term_is_linear (h1 : a1) (h17 : a17) (h_linear : linear_term a) : general_term_formula a := sorry

-- Question 2: Calculate value of a_20
def a20_value : Prop := a 20 = 78

theorem a20_correct (fact : general_term_formula a) : a20_value a := sorry

-- Question 3: Discuss whether 398 is an element of the sequence
def is_element_398 : Prop := ∃ n : ℕ, a n = 398

theorem 398_in_sequence (fact : general_term_formula a) : is_element_398 a := sorry

end general_term_is_linear_a20_correct_398_in_sequence_l292_292028


namespace problem_solution_l292_292433

theorem problem_solution (x1 x2 x3 : ℝ) (h1: x1 < x2) (h2: x2 < x3)
(h3 : 10 * x1^3 - 201 * x1^2 + 3 = 0)
(h4 : 10 * x2^3 - 201 * x2^2 + 3 = 0)
(h5 : 10 * x3^3 - 201 * x3^2 + 3 = 0) :
x2 * (x1 + x3) = 398 :=
sorry

end problem_solution_l292_292433


namespace cos_C_value_l292_292724

variable (k : ℝ) (ABC : Type) [triangle ABC]
variables (A B C : ABC) (h1 : angle A = 90) (h2 : tan angle C = 4)

noncomputable def cos_C_proof : ℝ :=
\(cos_C : ℝ), (cos angle C = sqrt(17) / 17)
by
  sorry


theorem cos_C_value : cos_C_proof ABC A B C h1 h2 = sqrt(17) / 17 :=
begin
  sorry
end

end cos_C_value_l292_292724


namespace number_of_true_propositions_l292_292247

theorem number_of_true_propositions :
  let prop1 := ∀ (a x : ℝ), 0 < a ∧ a < 1 ∧ x < 0 → a^x > 1,
      prop2 := (∀ (a m n : ℝ), 2 = m ∧ 1 = n → ∃ (a : ℝ), y = log a (x - 1) + 1 → log m n = 0),
      prop3 := ∀ (x : ℝ), ¬ ((y = x^(-1)) ∈ Icc -∞ (0:ℝ)) ∧ ¬ ((y = x^(-1)) ∈ Icc (0:ℝ) ∞),
      prop4 := ∃ (x : ℝ), tan x = 2011
  in (prop1 ∧ prop2 ∧ prop4 ∧ ¬ prop3) → true :=
by {
  sorry
}

end number_of_true_propositions_l292_292247


namespace average_disk_space_per_hour_l292_292950

theorem average_disk_space_per_hour :
  let days : ℕ := 15
  let total_mb : ℕ := 20000
  let hours_per_day : ℕ := 24
  let total_hours := days * hours_per_day
  total_mb / total_hours = 56 :=
by
  let days := 15
  let total_mb := 20000
  let hours_per_day := 24
  let total_hours := days * hours_per_day
  have h : total_mb / total_hours = 56 := sorry
  exact h

end average_disk_space_per_hour_l292_292950


namespace furthest_distance_l292_292484

-- Definitions of point distances as given conditions
def PQ : ℝ := 13
def QR : ℝ := 11
def RS : ℝ := 14
def SP : ℝ := 12

-- Statement of the problem in Lean
theorem furthest_distance :
  ∃ (P Q R S : ℝ),
    |P - Q| = PQ ∧
    |Q - R| = QR ∧
    |R - S| = RS ∧
    |S - P| = SP ∧
    ∀ (a b : ℝ), a ≠ b →
      |a - b| ≤ 25 :=
sorry

end furthest_distance_l292_292484


namespace tangent_line_eqn_of_sine_at_point_l292_292621

theorem tangent_line_eqn_of_sine_at_point :
  ∀ (f : ℝ → ℝ), (∀ x, f x = Real.sin (x + Real.pi / 3)) →
  ∀ (p : ℝ × ℝ), p = (0, Real.sqrt 3 / 2) →
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (∀ x, f x = Real.sin (x + Real.pi / 3)) ∧
  (∀ x y, y = f x → a * x + b * y + c = 0 → x - 2 * y + Real.sqrt 3 = 0) :=
by
  sorry

end tangent_line_eqn_of_sine_at_point_l292_292621


namespace range_of_f_is_R_l292_292674

-- Define the piecewise function f
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (1 - 2 * a) * x + 3 * a else Real.ln x

-- State the theorem corresponding to the proof problem
theorem range_of_f_is_R (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) ↔ (-1 ≤ a ∧ a < (1/2)) :=
by {
  -- The proof steps would go here
  sorry
}

end range_of_f_is_R_l292_292674


namespace general_term_a_minimum_n_for_T_l292_292655

noncomputable def sequence_a (n : ℕ) : ℕ := if n = 2 then 3 else 2 * n - 1

noncomputable def sequence_S (n : ℕ) : ℕ := n * (n + 1) / 2

theorem general_term_a (n : ℕ) (h : n > 1) : 
  ∃ a : ℕ → ℕ, (∀ n, 2 * sequence_S n - n * a n = n) ∧ (a n = 2 * n - 1) :=
sorry

noncomputable def sequence_b (n : ℕ) : ℝ := 1 / ((2 * n - 1) * real.sqrt (2 * n + 1) + (2 * n + 1) * real.sqrt (2 * n - 1))

noncomputable def sequence_T (n : ℕ) : ℝ := 1 / 2 * (1 - 1 / real.sqrt (2 * n + 1))

theorem minimum_n_for_T (n : ℕ) : 
  ∃ n : ℕ, sequence_T n > 9 / 20 ∧ n = 50 :=
sorry

end general_term_a_minimum_n_for_T_l292_292655


namespace board_cut_equal_areas_l292_292190

-- Definitions of lengths and widths
def L : ℝ := 120
def W_A : ℝ := 6
def W_B : ℝ := 12

-- The theorem we wish to prove
theorem board_cut_equal_areas (x : ℝ) (hx : 0 < x ∧ x < L) :
  let Wx := W_B - (W_B - W_A) * (x / L) in
  0.5 * (Wx + W_B) * x = 540 → x = 50.263 :=
sorry

end board_cut_equal_areas_l292_292190


namespace range_of_k_l292_292125

theorem range_of_k (k : ℝ) :
  (- (Real.sqrt 15) / 15 ≤ k ∧ k ≤ (Real.sqrt 15) / 15) ↔
  (∀ x y : ℝ, y = k * x + 3 → (x - 4)^2 + (y - 3)^2 = 4 → |(x, y) - (4,3)| ≥ 2 * Real.sqrt 3) :=
begin
  sorry
end

end range_of_k_l292_292125


namespace find_x_collinear_l292_292357

theorem find_x_collinear (x : ℝ) (a : ℝ × ℝ := (2, -1)) (b : ℝ × ℝ := (x, 1)) 
  (h_collinear : ∃ k : ℝ, (2 * 2 + x) = k * x ∧ (2 * -1 + 1) = k * 1) : x = -2 :=
by
  sorry

end find_x_collinear_l292_292357


namespace sum_of_numerical_coefficients_l292_292628

-- Define the multinomial expansion for (x + y + z)^3
def multinomial_expansion (x y z : ℚ) : ℚ :=
  (x + y + z)^3

-- The main theorem to prove: that the sum of all numerical coefficients in the expansion is 27
theorem sum_of_numerical_coefficients : 
  (let (x y z : ℚ) := (1, 1, 1) in
   let polynomial := multinomial_expansion x y z in
   polynomial.sum_numerical_coefficients = 27) :=
begin
  sorry -- proof will be filled in here
end

end sum_of_numerical_coefficients_l292_292628


namespace coffee_last_days_l292_292225

theorem coffee_last_days (weight : ℕ) (cups_per_lb : ℕ) (cups_per_day : ℕ) 
  (h_weight : weight = 3) 
  (h_cups_per_lb : cups_per_lb = 40) 
  (h_cups_per_day : cups_per_day = 3) : 
  (weight * cups_per_lb) / cups_per_day = 40 := 
by 
  sorry

end coffee_last_days_l292_292225


namespace probability_of_diff_three_l292_292779

-- We define the problem conditions and question
def roll_dice := ℕ
def diff_three := λ (x y : roll_dice), |x - y| = 3

-- Formalizing the expected probability
def probability (p : ℝ) : Prop :=
  p = 1 / 6

-- Predicate for checking if (x, y) is one of the pairs where |x-y| = 3
def valid_pairs (x y : ℕ) : Prop :=
  diff_three x y

-- Summarizing the proof problem that needs to be formalized and proved
theorem probability_of_diff_three :
  probability (6 / 36) :=
by
  sorry

end probability_of_diff_three_l292_292779


namespace max_f_value_l292_292480

def f (x : ℝ) : ℝ := (Real.cos (2 * x)) + 6 * (Real.cos (Real.pi / 2 - x))

theorem max_f_value : ∃ x : ℝ, f(x) = 5 ∧ (∀ y : ℝ, f(y) ≤ f(x)) :=
by 
  sorry

end max_f_value_l292_292480


namespace problem_probability_l292_292227

theorem problem_probability :
  let p_arthur := (1 : ℚ) / 4
  let p_bella := (3 : ℚ) / 10
  let p_xavier := (1 : ℚ) / 6
  let p_yvonne := (1 : ℚ) / 2
  let p_zelda := (5 : ℚ) / 8
  let p_zelda_failure := 1 - p_zelda
  let result := p_arthur * p_bella * p_xavier * p_yvonne * p_zelda_failure
  result = 9 / 3840 := by
  sorry

end problem_probability_l292_292227


namespace cos_neg_sixteen_pi_over_three_l292_292616

theorem cos_neg_sixteen_pi_over_three : 
  ∀ (x : ℝ), (∀ θ : ℝ, real.cos (-θ) = real.cos θ) → 
             (∀ k : ℤ, real.cos (x + 2 * k * real.pi) = real.cos x) → 
             real.cos (-16 * real.pi / 3) = -1 / 2 :=
by
  intros x h_even h_period
  sorry

end cos_neg_sixteen_pi_over_three_l292_292616


namespace rational_points_on_circle_l292_292593

theorem rational_points_on_circle : ∃ (points : Fin 1975 → ℂ), (∀ i j, i ≠ j → points i ∈ {z : ℂ | abs (z) = 1}) ∧ (∀ i j, i ≠ j → (abs (points i - points j)).denom = 1) :=
by {
  sorry
}

end rational_points_on_circle_l292_292593


namespace geometric_sequence_product_l292_292291

open_locale big_operators

theorem geometric_sequence_product :
  ∃ q : ℚ, (∃ a : ℕ → ℚ, a 0 = 8 / 3 ∧ a 4 = 27 / 2 ∧ (∀ n, a (n + 1) = a n * q) ∧ (a 1 * a 2 * a 3 = 216)) :=
sorry

end geometric_sequence_product_l292_292291


namespace count_positive_factors_of_144_multiple_of_18_l292_292360

-- Definition of n as 144.
def n : ℕ := 144

-- Function to check if a number is a factor of n.
def is_factor (m : ℕ) : Prop := m ∣ n

-- Function to check if a number is a multiple of 18.
def is_multiple_of_18 (m : ℕ) : Prop := 18 ∣ m

-- Function to count how many elements in a list satisfy given conditions.
def count_satisfying (l : List ℕ) (p : ℕ → Prop) : ℕ :=
  l.filter p |>.length

-- List of factors of n.
def factors_of_n : List ℕ :=
  (List.range (n + 1)).filter (λ m => is_factor m)

-- List of factors of n which are multiples of 18.
def relevant_factors : List ℕ :=
  factors_of_n.filter is_multiple_of_18

-- The main theorem
theorem count_positive_factors_of_144_multiple_of_18 :
  relevant_factors.length = 4 :=
by
  -- sorry is used to skip the proof.
  sorry

end count_positive_factors_of_144_multiple_of_18_l292_292360


namespace part_one_part_two_l292_292681

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 3| - 2 * |x + a|

-- Question (I)
theorem part_one (a : ℝ) (h : a = 3) : 
  {x : ℝ | f x a > 2} = set.Ioo (-7 : ℝ) (-5/3 : ℝ) :=
by {
  sorry
}

-- Question (II)
theorem part_two (A : set ℝ)
  (h : ∀ x ∈ set.Icc (-2 : ℝ) (-1 : ℝ), f x a + x + 1 ≤ 0) :
  a ∈ set.Iic (-1 : ℝ) ∪ set.Ici (4 : ℝ) :=
by {
  sorry
}

end part_one_part_two_l292_292681


namespace laura_interest_rate_l292_292419

/-- 
This statement corresponds to the setup where Laura took out a charge account with simple annual interest, 
charged $35 on her account and owed $36.75 a year later, with no additional payments or charges.
-/
theorem laura_interest_rate
  (principal : ℝ) (total_owed : ℝ) (time : ℝ) (principal_eq : principal = 35)
  (total_owed_eq : total_owed = 36.75) (time_eq : time = 1) :
  let interest := total_owed - principal in
  let rate := (interest / (principal * time) : ℝ) in
  rate * 100 = 5 :=
by
  -- Definitions from the conditions in the problem.
  let interest := total_owed - principal
  let rate := interest / (principal * time)
  -- Property to be proven
  have h1 : rate * 100 = 5 := sorry
  exact h1

end laura_interest_rate_l292_292419


namespace sum_of_real_solutions_sqrt_eq_8_l292_292289

theorem sum_of_real_solutions_sqrt_eq_8 :
  let f : ℝ → ℝ := λ x, sqrt (2 * x) + sqrt (8 / x) + sqrt (2 * x + 8 / x)
  in ∑ x in { x : ℝ | f x = 8 }.toFinset, x = 6.125 :=
by
  sorry

end sum_of_real_solutions_sqrt_eq_8_l292_292289


namespace equal_real_roots_l292_292119

theorem equal_real_roots (m : ℝ) : (∃ x : ℝ, x * x - 4 * x - m = 0) → (16 + 4 * m = 0) → m = -4 :=
by
  sorry

end equal_real_roots_l292_292119


namespace fourth_rectangle_area_l292_292195

theorem fourth_rectangle_area
  (dimension_large_rect_width : ℕ)
  (dimension_large_rect_height : ℕ)
  (area_rect1 : ℕ)
  (area_rect2 : ℕ)
  (area_rect3 : ℕ) :
  dimension_large_rect_width = 20 →
  dimension_large_rect_height = 15 →
  area_rect1 = 100 →
  area_rect2 = 50 →
  area_rect3 = 200 →
  let total_area := dimension_large_rect_width * dimension_large_rect_height in
  let area_rect4 := total_area - (area_rect1 + area_rect2 + area_rect3) in
  area_rect4 = 50 :=
begin
  intros h1 h2 h3 h4 h5,
  dsimp only,
  rw [h1, h2] at *,
  rw [h3, h4, h5],
  norm_num,
end

end fourth_rectangle_area_l292_292195


namespace total_distance_traveled_l292_292169

theorem total_distance_traveled (d : ℝ) (h1 : d/3 + d/4 + d/5 = 47/60) : 3 * d = 3 :=
by
  sorry

end total_distance_traveled_l292_292169


namespace total_expense_is_correct_l292_292754

noncomputable def soap_bars : ℕ := 20
noncomputable def soap_weight_per_bar : ℝ := 1.5
noncomputable def soap_cost_per_pound : ℝ := 0.5

noncomputable def shampoo_bottles : ℕ := 15
noncomputable def shampoo_weight_per_bottle : ℝ := 2.2
noncomputable def shampoo_cost_per_pound : ℝ := 0.8

noncomputable def soap_discount_rate : ℝ := 0.10
noncomputable def sales_tax_rate : ℝ := 0.05

noncomputable def total_soap_cost_before_discount : ℝ :=
  soap_bars * soap_weight_per_bar * soap_cost_per_pound

noncomputable def total_soap_discount : ℝ :=
  soap_discount_rate * total_soap_cost_before_discount

noncomputable def total_soap_cost : ℝ :=
  total_soap_cost_before_discount - total_soap_discount

noncomputable def total_shampoo_cost : ℝ :=
  shampoo_bottles * shampoo_weight_per_bottle * shampoo_cost_per_pound

noncomputable def total_cost_before_tax : ℝ := total_soap_cost + total_shampoo_cost

noncomputable def total_sales_tax : ℝ := sales_tax_rate * total_cost_before_tax

noncomputable def round_to_cents (x : ℝ) : ℝ := (x * 100).round / 100

noncomputable def total_cost : ℝ := total_cost_before_tax + round_to_cents total_sales_tax

theorem total_expense_is_correct :
  total_cost = 41.9 := by
  sorry

end total_expense_is_correct_l292_292754


namespace mike_taller_than_mark_l292_292790

-- Define the heights of Mark and Mike in terms of feet and inches
def mark_height_feet : ℕ := 5
def mark_height_inches : ℕ := 3
def mike_height_feet : ℕ := 6
def mike_height_inches : ℕ := 1

-- Define the conversion factor from feet to inches
def feet_to_inches : ℕ := 12

-- Conversion of heights to inches
def mark_total_height_in_inches : ℕ := mark_height_feet * feet_to_inches + mark_height_inches
def mike_total_height_in_inches : ℕ := mike_height_feet * feet_to_inches + mike_height_inches

-- Define the problem statement: proving Mike is 10 inches taller than Mark
theorem mike_taller_than_mark : mike_total_height_in_inches - mark_total_height_in_inches = 10 :=
by sorry

end mike_taller_than_mark_l292_292790


namespace area_of_desired_sector_l292_292507

noncomputable def area_of_sector_above_x_axis_and_right_of_line (x y : ℝ) : ℝ :=
  if x^2 - 10*x + y^2 = 34 ∧ y ≥ 0 ∧ y ≤ 5 - x then (3/8) * 59 * real.pi else 0

theorem area_of_desired_sector : area_of_sector_above_x_axis_and_right_of_line = (177 / 8) * real.pi :=
by
  -- provided conditions of the circle and the line
  sorry

end area_of_desired_sector_l292_292507


namespace sequence_bound_l292_292437

-- Define the conditions
variables (a : ℕ → ℝ)
variables (h_nonneg : ∀ n, 0 ≤ a n)
variables (h_additive : ∀ m n, a (n + m) ≤ a n + a m)

-- The theorem statement
theorem sequence_bound (n m : ℕ) (h : n ≥ m) : 
  a n ≤ m * a 1 + ((n.to_real / m.to_real) - 1) * a m :=
by
  sorry

end sequence_bound_l292_292437


namespace mike_taller_than_mark_l292_292789

-- Define the heights of Mark and Mike in terms of feet and inches
def mark_height_feet : ℕ := 5
def mark_height_inches : ℕ := 3
def mike_height_feet : ℕ := 6
def mike_height_inches : ℕ := 1

-- Define the conversion factor from feet to inches
def feet_to_inches : ℕ := 12

-- Conversion of heights to inches
def mark_total_height_in_inches : ℕ := mark_height_feet * feet_to_inches + mark_height_inches
def mike_total_height_in_inches : ℕ := mike_height_feet * feet_to_inches + mike_height_inches

-- Define the problem statement: proving Mike is 10 inches taller than Mark
theorem mike_taller_than_mark : mike_total_height_in_inches - mark_total_height_in_inches = 10 :=
by sorry

end mike_taller_than_mark_l292_292789


namespace scientific_notation_GDP_l292_292012

theorem scientific_notation_GDP (h : 1 = 10^9) : 32.07 * 10^9 = 3.207 * 10^10 := by
  sorry

end scientific_notation_GDP_l292_292012


namespace krishan_money_l292_292486

noncomputable def money_ratios (x y : ℝ) : Prop :=
  let R := 1503 in
  let total := 15000 in
  let G := (17 / 7) * R in
  let K := (17 / 7) * (y / x) * R in
  R + G + K = total

theorem krishan_money (x y : ℝ) (h : money_ratios x y) :
  let K := (17 / 7) * (y / x) * 1503 in
  K ≈ 9845 :=
sorry


end krishan_money_l292_292486


namespace ariel_fish_l292_292578

theorem ariel_fish (total_fish : ℕ) (male_fraction female_fraction : ℚ) (h1 : total_fish = 45) (h2 : male_fraction = 2 / 3) (h3 : female_fraction = 1 - male_fraction) : total_fish * female_fraction = 15 :=
by
  sorry

end ariel_fish_l292_292578


namespace angle_CDE_gt_45_l292_292445

open Triangle

theorem angle_CDE_gt_45 
  (A B C D E : Point)
  (hABC_acute : isAcuteTriangle A B C)
  (hBE_bisector : isInternalAngleBisector A B C E)
  (hAD_altitude : isAltitude A D B C) :
  ∠CDE > 45 :=
sorry

end angle_CDE_gt_45_l292_292445


namespace arithmetic_mean_minima_l292_292431

open_locale big_operators

-- Define the binomial coefficient function (although usually provided by Mathlib)
noncomputable def binom (n k : ℕ) : ℕ := nat.choose n k

-- Define the function f(n, r) as the arithmetic mean of the minima of all r-element subsets of {1, 2, ..., n}
noncomputable def f (n r : ℕ) : ℚ :=
  (∑ k in finset.range (n - r + 1) + 1, (k + 1) * binom (n - (k + 1)) (r - 1)) / binom n r

-- State the theorem
theorem arithmetic_mean_minima (n r : ℕ) (hnr : r ≤ n) : f n r = (n + 1) / (r + 1) := 
by sorry

end arithmetic_mean_minima_l292_292431


namespace range_of_norm_c_l292_292642

-- Given conditions
variables (a b c : ℝ^3) -- representing the vectors a, b, and c in R^3
hypothesis (h₁: ∥a∥ = 3) -- |a| = 3
hypothesis (h₂: ∥b∥ = 4) -- |b| = 4
hypothesis (h₃: a ⬝ b = 0) -- a · b = 0
hypothesis (h₄: (a - c) ⬝ (b - c) = 0) -- (a - c) · (b - c) = 0

-- Prove that the range of |c| is [0, 5]
theorem range_of_norm_c : 0 ≤ ∥c∥ ∧ ∥c∥ ≤ 5 :=
sorry

end range_of_norm_c_l292_292642


namespace baseball_cards_per_pack_l292_292444

theorem baseball_cards_per_pack (cards_each : ℕ) (packs_total : ℕ) (total_cards : ℕ) (cards_per_pack : ℕ) :
    (cards_each = 540) →
    (packs_total = 108) →
    (total_cards = cards_each * 4) →
    (cards_per_pack = total_cards / packs_total) →
    cards_per_pack = 20 :=
by
  intros h1 h2 h3 h4
  sorry

end baseball_cards_per_pack_l292_292444


namespace min_value_eq_l292_292330

open Real
open Classical

noncomputable def min_value (x y : ℝ) : ℝ := x + 4 * y

theorem min_value_eq :
  ∀ (x y : ℝ), (x > 0) → (y > 0) → (1 / x + 1 / (2 * y) = 1) → (min_value x y) = 3 + 2 * sqrt 2 :=
by
  sorry

end min_value_eq_l292_292330


namespace remainder_of_sum_mod_9_l292_292627

theorem remainder_of_sum_mod_9 :
  (9023 + 9024 + 9025 + 9026 + 9027) % 9 = 2 :=
by
  sorry

end remainder_of_sum_mod_9_l292_292627


namespace sum_even_integers_less_than_102_l292_292888

theorem sum_even_integers_less_than_102 :
  ∑ k in Finset.filter (λ x => (even x) ∧ (0 < x) ∧ (x < 102)) (Finset.range 102), k = 2550 := sorry

end sum_even_integers_less_than_102_l292_292888


namespace solve_trig_eq_l292_292830

theorem solve_trig_eq (x : ℝ) :
  (cos x + 2 * cos (6 * x))^2 = 9 + (sin (3 * x))^2 →
  (-3 : ℝ) ≤ cos x + 2 * cos (6 * x) ∧ cos x + 2 * cos (6 * x) ≤ 3 →
  9 + (sin (3 * x))^2 ≥ 9 →
  ∃ k : ℤ, x = 2 * k * Real.pi :=
  sorry

end solve_trig_eq_l292_292830


namespace regular_polygon_diagonals_l292_292388

theorem regular_polygon_diagonals (n : ℕ): 
  (n > 2) → 
  (120° = ((n - 2) * 180°) / n) → 
  (n - 3 = 3) :=
by 
  have h120 := (120 : ℝ)
  sorry

end regular_polygon_diagonals_l292_292388


namespace total_polynomials_in_H_l292_292422

noncomputable def number_of_polynomials_in_H (n : ℕ) (c : Fin n → ℤ) : ℕ :=
  -- The definition for calculating the number of polynomials in H
  sorry

theorem total_polynomials_in_H
  (n : ℕ)
  (c : Fin n → ℤ)
  (Q : Polynomial ℂ)
  (h₁ : ∀ i : Fin n, c i ∈ ℤ)
  (h₂ : ∀ z : ℂ, Q.eval z = 0 → Q.eval (conj z) = 0)
  (h₃ : Q.coeff 0 = -36) :
  number_of_polynomials_in_H n c = total :=
begin
  sorry,
end

end total_polynomials_in_H_l292_292422


namespace seven_vectors_sum_to_zero_l292_292317

noncomputable def is_regular_12_polygon (A : Fin 12 → ℝ × ℝ) : Prop :=
  ∀ i j, dist (A i) (A j) = 2 * sin (π / 12 * (abs (i - j)))

def vectors_of_12_polygon (A : Fin 12 → ℝ × ℝ) : Fin 12 → ℝ × ℝ :=
  λ i, (A ((i + 1) % 12)).1 - (A i).1, (A ((i + 1) % 12)).2 - (A i).2

theorem seven_vectors_sum_to_zero (A : Fin 12 → ℝ × ℝ) (h : is_regular_12_polygon A) :
  ∃ (S : Finset (Fin 12)), S.card = 7 ∧
  Finset.sum S (vectors_of_12_polygon A) = (0, 0) :=
begin
  sorry
end

end seven_vectors_sum_to_zero_l292_292317


namespace union_of_sets_l292_292688

theorem union_of_sets (M N : Set ℂ) (m : ℂ) 
  (hM : M = {1, m, 3 + (m^2 - 5*m - 6)*complex.i})
  (hN : N = {x | x^2 - 2*x - 3 = 0})
  (hMN : M ∩ N = {3}) :
  M ∪ N = {(-1 : ℂ), 1, 3, m} ∨ M ∪ N = {(-1 : ℂ), 1, 3, 3 - 12 * complex.i} :=
by {
  sorry
}

end union_of_sets_l292_292688


namespace part1_part2_part3_l292_292341

-- Define the function f(x) and its derivative
def f (a x : ℝ) := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x

-- The value of a such that the slope of the tangent line to y = f(x) at x = 0 is 3
theorem part1 (a : ℝ) (h : deriv (λ x, f a x) 0 = 3) : a = 1/2 :=
by
  sorry

-- The range of a such that f(x) + f(-x) ≥ 12 * log x for any x ∈ (0, +∞)
theorem part2 (a : ℝ) (h : ∀ x : ℝ, 0 < x → f a x + f a (-x) ≥ 12 * log x) : a ≤ -1 - 1/Real.e :=
by
  sorry

-- If a > 1, the minimum value of h(a) = M(a) - m(a) on [1, 2] is 8/27
theorem part3 (a : ℝ) (ha : 1 < a) :
  let M := λ a, max (f a 1) (f a 2)
      m := λ a, min (f a 1) (f a 2)
      h := λ a, M a - m a
  in h a = 8/27 :=
by
  sorry

end part1_part2_part3_l292_292341


namespace percentage_water_in_puree_l292_292695

/-- Given that tomato juice is 90% water and Heinz obtains 2.5 litres of tomato puree from 20 litres of tomato juice,
proves that the percentage of water in the tomato puree is 20%. -/
theorem percentage_water_in_puree (tj_volume : ℝ) (tj_water_content : ℝ) (tp_volume : ℝ) (tj_to_tp_ratio : ℝ) 
  (h1 : tj_water_content = 0.90) 
  (h2 : tj_volume = 20) 
  (h3 : tp_volume = 2.5) 
  (h4 : tj_to_tp_ratio = tj_volume / tp_volume) : 
  ((tp_volume - (1 - tj_water_content) * (tj_volume * (tp_volume / tj_volume))) / tp_volume) * 100 = 20 := 
sorry

end percentage_water_in_puree_l292_292695


namespace original_price_of_coat_l292_292128

theorem original_price_of_coat (P : ℝ) (h : 0.40 * P = 200) : P = 500 :=
by {
  sorry
}

end original_price_of_coat_l292_292128


namespace ariel_fish_l292_292572

theorem ariel_fish (total_fish : ℕ) (male_ratio : ℚ) (female_ratio : ℚ) (female_fish : ℕ) : 
  total_fish = 45 ∧ male_ratio = 2/3 ∧ female_ratio = 1/3 → female_fish = 15 :=
by
  sorry

end ariel_fish_l292_292572


namespace original_inequality_solution_l292_292152

theorem original_inequality_solution (b c : ℚ) 
  (hA : ∀ x : ℚ, -6 < x ∧ x < 2 → x^2 + (6 + 2) * x + (-6 * 2) < 0)
  (hB : ∀ x : ℚ, -3 < x ∧ x < 2 → x^2 + (3 + 2) * x + (-(3 + 2)x * x) < 0) :
  ∀ x : ℚ, -4 < x ∧ x < 3 → x^2 + x - 12 < 0 :=
by
  sorry

end original_inequality_solution_l292_292152


namespace geometric_series_sum_l292_292163

theorem geometric_series_sum :
  let a := 1
  let r := 3
  let last_term := 19683
  let n := 10
  ∑ (i : Nat) in Finset.range n, a * r^i = 29524 := by
  sorry

end geometric_series_sum_l292_292163


namespace peanut_cluster_percentage_l292_292238

def chocolates_total : ℕ := 50
def caramels : ℕ := 3
def nougats : ℕ := 2 * caramels
def truffles : ℕ := caramels + 6
def peanut_clusters : ℕ := chocolates_total - caramels - nougats - truffles

theorem peanut_cluster_percentage : 
  (peanut_clusters / chocolates_total.to_real * 100) = 64 := 
by 
  sorry

end peanut_cluster_percentage_l292_292238


namespace complex_number_properties_l292_292327

open real

theorem complex_number_properties
  (a : ℝ)
  (i : ℂ)
  (h1 : i = complex.I)
  (p : (a + 1 < 0))
  (q : (complex.abs (a - i) = 2)) :
  a = -real.sqrt 3 := 
sorry

end complex_number_properties_l292_292327


namespace sum_c_case2_l292_292322

variable {a b c : ℕ → ℕ}

def a (n : ℕ) : ℕ := 2 * n - 1

def b (n : ℕ) : ℕ := 3^(n-1)

def c (n : ℕ) : ℕ := a n * b n

noncomputable def S (n : ℕ) : ℕ := 
  ∑ i in Finset.range n, c (i + 1)

theorem sum_c_case2 (n : ℕ) : S n = (n - 1) * 3^n + 1 := 
by
  sorry

end sum_c_case2_l292_292322


namespace sum_of_roots_l292_292893

-- Define the polynomial
def poly := 3 * (Polynomial.X ^ 3) - 6 * (Polynomial.X ^ 2) - 9 * Polynomial.X

-- State the theorem about the sum of the roots of the polynomial
theorem sum_of_roots : polynomial.sum_roots poly = 2 :=
by sorry

end sum_of_roots_l292_292893


namespace perpendicular_distance_sum_l292_292817

theorem perpendicular_distance_sum
  {A B C S : Type} [MetricSpace S]
  (centroid : centroid A B C = S)
  (line_through_S : ∀ l, Line l ∧ S ∈ l)
  (perpendicular_distances : ∀ (A B C : S), ∃ (A1 B1 C1 : S), (PerpendicularDistance A l = A1) ∧ (PerpendicularDistance B l = B1) ∧ (PerpendicularDistance C l = C1)) :
  ∀ (l : Line S), (PerpendicularDistance C l) = (PerpendicularDistance A l) + (PerpendicularDistance B l) :=
by
  intro l
  have A1 := perpendicular_distances A
  have B1 := perpendicular_distances B
  have C1 := perpendicular_distances C
  sorry

end perpendicular_distance_sum_l292_292817


namespace bridge_length_l292_292987

theorem bridge_length (length_train : ℝ) (speed_train : ℝ) (time : ℝ) (h1 : length_train = 15) (h2 : speed_train = 275) (h3 : time = 48) : 
    (speed_train / 100) * time - length_train = 117 := 
by
    -- these are the provided conditions, enabling us to skip actual proof steps with 'sorry'
    sorry

end bridge_length_l292_292987


namespace average_weight_of_new_people_l292_292735

-- Define the weights and conditions from the problem
def initial_group_size : ℕ := 10
def initial_average_weight : ℝ := 75
def mistaken_weight : ℝ := 65
def new_people_count : ℕ := 3
def increased_average_weight : ℝ := 77

-- Prove the average weight of the 3 newly joined individuals
theorem average_weight_of_new_people :
  ∀ (initial_group_weight new_total_weight corrected_initial_group_weight total_new_people_weight : ℝ),
  initial_group_weight = initial_group_size * initial_average_weight →
  corrected_initial_group_weight = initial_group_weight - mistaken_weight →
  new_total_weight = (initial_group_size + new_people_count) * increased_average_weight →
  total_new_people_weight = new_total_weight - corrected_initial_group_weight →
  total_new_people_weight / real.of_nat new_people_count = 79.67 :=
by
  intros initial_group_weight new_total_weight corrected_initial_group_weight total_new_people_weight
  intros h1 h2 h3 h4
  sorry

end average_weight_of_new_people_l292_292735


namespace rational_functional_equation_l292_292768

theorem rational_functional_equation (f : ℚ → ℚ) (h : ∀ x y : ℚ, f (x + f y) = f x + y) :
  (f = λ x => x) ∨ (f = λ x => -x) :=
by
  sorry

end rational_functional_equation_l292_292768


namespace quadratic_no_third_quadrant_l292_292122

theorem quadratic_no_third_quadrant (x y : ℝ) : 
  (y = x^2 - 2 * x) → ¬(x < 0 ∧ y < 0) :=
by
  intro hy
  sorry

end quadratic_no_third_quadrant_l292_292122


namespace geometric_series_sum_l292_292589

theorem geometric_series_sum :
  ∑' i : ℕ, (2 / 3) ^ (i + 1) = 2 :=
by
  sorry

end geometric_series_sum_l292_292589


namespace ratio_john_to_jenna_l292_292044

theorem ratio_john_to_jenna (J : ℕ) 
  (h1 : 100 - J - 40 = 35) : 
  J = 25 ∧ (J / 100 = 1 / 4) := 
by
  sorry

end ratio_john_to_jenna_l292_292044


namespace projectile_height_35_l292_292202

noncomputable def projectile_height (t : ℝ) : ℝ := -4.9 * t^2 + 30 * t

theorem projectile_height_35 (t : ℝ) :
  projectile_height t = 35 ↔ t = 10/7 :=
by {
  sorry
}

end projectile_height_35_l292_292202


namespace solve_for_z_l292_292105

noncomputable def i : ℂ := complex.I

theorem solve_for_z : (z : ℂ) (h : i^2 = -1) (eq1 : 3 - 2 * i * z = 1 + 4 * i * z) : 
  z = -i / 3 := 
by
  sorry

end solve_for_z_l292_292105


namespace eval_expression_l292_292268

theorem eval_expression (a x : ℕ) (h : x = a + 9) : x - a + 5 = 14 :=
by 
  sorry

end eval_expression_l292_292268


namespace mike_taller_than_mark_l292_292788

-- Define the heights of Mark and Mike in terms of feet and inches
def mark_height_feet : ℕ := 5
def mark_height_inches : ℕ := 3
def mike_height_feet : ℕ := 6
def mike_height_inches : ℕ := 1

-- Define the conversion factor from feet to inches
def feet_to_inches : ℕ := 12

-- Conversion of heights to inches
def mark_total_height_in_inches : ℕ := mark_height_feet * feet_to_inches + mark_height_inches
def mike_total_height_in_inches : ℕ := mike_height_feet * feet_to_inches + mike_height_inches

-- Define the problem statement: proving Mike is 10 inches taller than Mark
theorem mike_taller_than_mark : mike_total_height_in_inches - mark_total_height_in_inches = 10 :=
by sorry

end mike_taller_than_mark_l292_292788


namespace average_pairs_of_consecutive_integers_l292_292875

open Finset

theorem average_pairs_of_consecutive_integers (s : Finset ℕ) (h1 : s.card = 6) (h2 : ∀ x ∈ s, 1 ≤ x ∧ x ≤ 40) :
  ∃ result : ℝ, average_consecutive_pairs s = result :=
by {
  sorry, -- This is where the proof would go
}

end average_pairs_of_consecutive_integers_l292_292875


namespace max_odd_digits_l292_292492

noncomputable def a : ℕ := sorry
noncomputable def b : ℕ := sorry
noncomputable def c : ℕ := a + b

def is_10_digit_number (x : ℕ) : Prop := 10^9 ≤ x ∧ x < 10^10

def is_odd_digit (d : ℕ) : Prop := d % 2 = 1

def count_odd_digits (n : ℕ) : ℕ :=
  (nat.digits 10 n).countp is_odd_digit

theorem max_odd_digits (a b c : ℕ) (ha : is_10_digit_number a) (hb : is_10_digit_number b) (hc : is_10_digit_number c) (habc : a + b = c) :
  count_odd_digits a + count_odd_digits b + count_odd_digits c ≤ 29 :=
by
  sorry

end max_odd_digits_l292_292492


namespace correct_statements_count_l292_292564

-- Definitions for the statements
def statement1 := (∅ = ({0} : Set Nat))
def statement2 := (∅ ⊆ ({0} : Set Nat))
def statement3 := (∅ ∈ ({0} : Set (Set Nat)))
def statement4 := ((0 : Nat) = ({0} : Set Nat))
def statement5 := ((0 : Nat) ∈ ({0} : Set Nat))
def statement6 := ({1} ∈ ({1, 2, 3} : Set Nat))
def statement7 := ({1, 2} ⊆ ({1, 2, 3} : Set Nat))
def statement8 := ({'a', 'b'} : Set Char) = ({'b', 'a'} : Set Char)

-- Proof problem in Lean 4
theorem correct_statements_count : 
  (count (λ s, s) [statement1, statement2, statement3, statement4, statement5, statement6, statement7, statement8] (λ b, b = true)) = 4 :=
by
  sorry

end correct_statements_count_l292_292564


namespace bailey_points_final_game_l292_292026

def chandra_points (a: ℕ) := 2 * a
def akiko_points (m: ℕ) := m + 4
def michiko_points (b: ℕ) := b / 2
def team_total_points (b c a m: ℕ) := b + c + a + m

theorem bailey_points_final_game (B: ℕ) 
  (M : ℕ := michiko_points B)
  (A : ℕ := akiko_points M)
  (C : ℕ := chandra_points A)
  (H : team_total_points B C A M = 54): B = 14 :=
by 
  sorry

end bailey_points_final_game_l292_292026


namespace a_2016_value_l292_292299

def S (n : ℕ) : ℕ := n^2 - 1

def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem a_2016_value : a 2016 = 4031 := by
  sorry

end a_2016_value_l292_292299


namespace length_of_projection_on_yOz_l292_292665

-- Define points A and B in 3D space
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point3D := { x := 3, y := 5, z := -7 }
def B : Point3D := { x := -2, y := 4, z := -6 }

-- Projection of points A and B on the yOz plane
def projection_on_yOz (p : Point3D) : Point3D :=
  { x := 0, y := p.y, z := p.z }

def A' : Point3D := projection_on_yOz A
def B' : Point3D := projection_on_yOz B

-- Define the distance formula in 3D
def distance (p1 p2 : Point3D) : ℝ :=
  real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2 + (p1.z - p2.z) ^ 2)

-- Prove the length of the projection of AB on yOz is √2
theorem length_of_projection_on_yOz :
  distance A' B' = real.sqrt 2 :=
by
  sorry

end length_of_projection_on_yOz_l292_292665


namespace correct_propositions_count_l292_292568

-- Definitions based on the conditions
variables (l : Type) (P A : l) (a : l) (α : l) (β : l) (γ : l)

-- Condition statements
def proposition1 : Prop := ∀ (l1 l2 : l), (l1 ⊥ l2 ∧ l1 ⊥ l2) → l1 ⊥ α
def proposition2 : Prop := ∃! (π : l), (π ⊥ l)
def proposition3 : Prop := ∀ (l1 l2 l3 : l), (l1 ⊥ l2 ∧ l2 ⊥ l3 ∧ l1 ⊥ l3) → l1 ⊥ β ∧ β ∈ plane l2 l3
def proposition4 : Prop := ∀ (l1 l2 : l), (l1 ⊥ l2) → l1 ⊥ γ
def proposition5 : Prop := ∀ (l : l), (l ⊥ a) → l ∈ plane a P

-- Theorem to prove the number of correct propositions
theorem correct_propositions_count : 
  (proposition2 P l) ∧
  (proposition3 α β γ) ∧
  (proposition5 a A) → 
  ∃ c, c = 3 := 
by {
  sorry
}

end correct_propositions_count_l292_292568


namespace problem_even_and_monotonically_increasing_l292_292566

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_monotonically_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f x ≤ f y

theorem problem_even_and_monotonically_increasing :
  is_even_function (fun x => Real.exp (|x|)) ∧ is_monotonically_increasing_on (fun x => Real.exp (|x|)) (Set.Ioo 0 1) :=
by
  sorry

end problem_even_and_monotonically_increasing_l292_292566


namespace xiao_xuan_wins_l292_292805

def cards_game (n : ℕ) (min_take : ℕ) (max_take : ℕ) (initial_turn : String) : String :=
  if initial_turn = "Xiao Liang" then "Xiao Xuan" else "Xiao Liang"

theorem xiao_xuan_wins :
  cards_game 17 1 2 "Xiao Liang" = "Xiao Xuan" :=
sorry

end xiao_xuan_wins_l292_292805


namespace ellipse_proof_min_MN_length_l292_292355

noncomputable def ellipse_eqn (a b : ℝ) (x y : ℝ) : Prop :=
  (a > b ∧ b > 0) ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def circle1_eqn (x y : ℝ) : Prop :=
  (x + sqrt 3)^2 + y^2 = 9

def circle2_eqn (x y : ℝ) : Prop :=
  (x - sqrt 3)^2 + y^2 = 1

def ellipse_foci (x1 y1 x2 y2 : ℝ) : Prop :=
  circle1_eqn x1 y1 ∧ circle2_eqn x2 y2

theorem ellipse_proof (a b x y x1 y1 x2 y2 : ℝ) (h1 : ellipse_foci x1 y1 x2 y2) (c1 : circle1_eqn (2 * sqrt 3 / 3) (sqrt 6 / 3)) (c2 : circle1_eqn (2 * sqrt 3 / 3) (-sqrt 6 / 3)) :
  ellipse_eqn a b x y → a = sqrt 2 ∧ b = 1 ∧ x^2 / 2 + y^2 = 1 := sorry

variables (k : ℝ)

def point_on_line (x y : ℝ) : Prop :=
  x = 2 * sqrt 3

def F1M_intersection (M_x M_y k : ℝ) : Prop :=
  M_x = 2 * sqrt 3 ∧ M_y = 3 * sqrt 3 * k

def F2N_intersection (N_x N_y k : ℝ) : Prop :=
  N_x = 2 * sqrt 3 ∧ N_y = -sqrt 3 / k

def is_perpendicular (F1_x F1_y M_x M_y F2_x F2_y N_x N_y : ℝ) : Prop :=
  (F1_x * M_x + F1_y * M_y) * (F2_x * N_x + F2_y * N_y) = 0

theorem min_MN_length (F1_x F1_y F2_x F2_y M_x M_y N_x N_y k : ℝ) (h1 : point_on_line M_x M_y) (h2 : point_on_line N_x N_y)
  (p1 : circle1_eqn F1_x F1_y) (p2 : circle2_eqn F2_x F2_y)
  (p3 : is_perpendicular F1_x F1_y M_x M_y F2_x F2_y N_x N_y) :
  |sqrt 3 * (3 * k + 1 / k)| = 6 → F1_x = 0 ∧ F1_y = -sqrt 3 ∧ 
  F2_x = 2 * sqrt 3 ∧ F2_y = - sqrt 3 / (sqrt 3 / 3) ∧ 
  M_x = 2 * sqrt 3 ∧ M_y = 3 ∧ 
  N_x = 2 * sqrt 3 ∧ N_y = -sqrt 3 / (sqrt 3 / 3) := sorry

end ellipse_proof_min_MN_length_l292_292355


namespace jungkook_seokjin_books_l292_292415

/-- Given the number of books Jungkook and Seokjin originally had and the number of books they 
   bought, prove that Jungkook has 7 more books than Seokjin. -/
theorem jungkook_seokjin_books
  (jungkook_initial : ℕ)
  (seokjin_initial : ℕ)
  (jungkook_bought : ℕ)
  (seokjin_bought : ℕ)
  (h1 : jungkook_initial = 28)
  (h2 : seokjin_initial = 28)
  (h3 : jungkook_bought = 18)
  (h4 : seokjin_bought = 11) :
  (jungkook_initial + jungkook_bought) - (seokjin_initial + seokjin_bought) = 7 :=
by
  sorry

end jungkook_seokjin_books_l292_292415


namespace total_number_of_days_2005_to_2008_l292_292232

def is_common_year (year : ℕ) : Prop :=
  ¬(year % 4 = 0 ∧ (year % 100 ≠ 0 ∨ year % 400 = 0))

def is_leap_year (year : ℕ) : Prop :=
  year % 4 = 0 ∧ (year % 100 ≠ 0 ∨ year % 400 = 0)

def days_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then 366 else 365

theorem total_number_of_days_2005_to_2008 :
  days_in_year 2005 + days_in_year 2006 + days_in_year 2007 + days_in_year 2008 = 1461 :=
by
  -- the proof would go here
  sorry

end total_number_of_days_2005_to_2008_l292_292232


namespace find_average_age_of_women_l292_292463

/-- This definition captures the conditions and asserts the correct answer -/
def average_age_of_women  (A W1 W2 : ℕ) : Prop :=
  let men_total_age := 8 * A in
  let removed_men_age := 20 + 22 in
  let remaining_men_total_age := men_total_age - removed_men_age in
  let replaced_group_total_age := remaining_men_total_age + W1 + W2 in
  let increased_average := A + 2 in
  let new_group_total_age := 8 * increased_average in
  (remaining_men_total_age + W1 + W2 = new_group_total_age) → (W1 + W2) / 2 = 29

theorem find_average_age_of_women (A W1 W2 : ℕ) (h: average_age_of_women A W1 W2) : (W1 + W2) / 2 = 29 :=
sorry

end find_average_age_of_women_l292_292463


namespace gcd_2023_2052_eq_1_l292_292876

theorem gcd_2023_2052_eq_1 : Int.gcd 2023 2052 = 1 :=
by
  sorry

end gcd_2023_2052_eq_1_l292_292876


namespace area_of_circumcircle_l292_292147

variables {α : Type} [linear_ordered_field α] [char_zero α]
variables {a b c A B C : α}

-- Definitions from conditions
def cos_C := (2 * real.sqrt 2) / 3
def b_cos_A_plus_a_cos_B := (b * cos(A) + a * cos(B) : α) = 2
def triangle_sides := (b, c, a)

-- The statement to prove
theorem area_of_circumcircle
  (hc : cos C = cos_C)
  (hb_ca : b * cos A + a * cos B = 2)
  : π * (c/real.sin(C)) ^ 2 = 9 * π :=
by sorry

end area_of_circumcircle_l292_292147


namespace sequences_and_sum_l292_292336

structure SequenceProperties :=
  (a : ℕ → ℤ)
  (b : ℕ → ℕ)
  (d : ℤ)
  (d_pos : 0 < d)
  (a1 : a 1 = 1)
  (b1 : b 1 = 2)
  (b2_minus_a2 : b 2 - a 2 = 1)
  (a3_plus_b3 : a 3 + b 3 = 13)

def arithmetic_seq (d : ℤ) (a1 : ℤ) (n : ℕ) : ℤ := a1 + d * (n - 1)

def geometric_seq (q : ℕ) (b1 : ℕ) (n : ℕ) : ℕ := b1 * q^(n-1)

noncomputable def c (a : ℕ → ℤ) (b : ℕ → ℕ) (n : ℕ) : ℤ := a n * b n

noncomputable def S (a : ℕ → ℤ) (b : ℕ → ℕ) (n : ℕ) : ℤ := 
  ∑ i in finRange(n+1), c a b i

theorem sequences_and_sum (seq_props : SequenceProperties) :
  (∀ n, seq_props.a n = 2n - 1) ∧
  (∀ n, seq_props.b n = 2^n) ∧ 
  (∀ n, S seq_props.a seq_props.b n = 6 + (2n - 3) * 2^(n + 1)) :=
sorry

end sequences_and_sum_l292_292336


namespace problem_inequality_l292_292773

variable (a b c : ℝ)

theorem problem_inequality (h_pos : a > 0) (h_bc : b > 0) (h_c : c > 0)
  (h_abc_sum : a * b * c ≤ a + b + c) : 
  a ^ 2 + b ^ 2 + c ^ 2 ≥ real.sqrt 3 * (a * b * c) := 
  sorry

end problem_inequality_l292_292773


namespace Nikki_movie_length_l292_292049

variable (M : ℝ) -- Length of Michael's movie

-- Conditions
def Joyce_movie := M + 2
def Nikki_movie := 3 * M
def Ryn_movie := (4 / 5) * (3 * M)
def total_length := M + Joyce_movie + Nikki_movie + Ryn_movie

-- Theorem statement
theorem Nikki_movie_length (h : total_length M = 76) : Nikki_movie M = 30 := sorry

end Nikki_movie_length_l292_292049


namespace find_b_values_l292_292279

theorem find_b_values :
  ∃ (b2 b3 b4 b5 : ℤ), 0 ≤ b2 ∧ b2 < 2 ∧ 0 ≤ b3 ∧ b3 < 3 ∧ 
    0 ≤ b4 ∧ b4 < 4 ∧ 0 ≤ b5 ∧ b5 < 5 ∧ 
    (3 : ℚ) / 5 = b2 / 2.factorial + b3 / 3.factorial + b4 / 4.factorial + b5 / 5.factorial ∧
    b2 + b3 + b4 + b5 = 4 :=
sorry

end find_b_values_l292_292279


namespace range_of_combined_set_is_86_l292_292925

def is_prime (n : ℕ) : Prop :=
∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def set_x : set ℕ := {n | is_prime n ∧ n ≥ 10 ∧ n < 100}

def is_odd_multiple_of_7 (n : ℕ) : Prop :=
n % 2 = 1 ∧ n % 7 = 0 ∧ n < 100

def set_y : set ℕ := {n | is_odd_multiple_of_7 n}

def combined_set : set ℕ := set_x ∪ set_y

def range_of_set (s : set ℕ) : ℕ :=
finset.max' (set.to_finset s) sorry - finset.min' (set.to_finset s) sorry

theorem range_of_combined_set_is_86 : range_of_set combined_set = 86 :=
sorry

end range_of_combined_set_is_86_l292_292925


namespace remainder_of_P_div_x_minus_2_l292_292284

def P (x : ℝ) : ℝ := 5*x^5 - 8*x^4 + 3*x^3 - x^2 + 4*x - 15

theorem remainder_of_P_div_x_minus_2 : P 2 = 45 := by
  sorry

end remainder_of_P_div_x_minus_2_l292_292284


namespace exam_fail_percentage_l292_292173

theorem exam_fail_percentage
  (total_candidates : ℕ := 2000)
  (girls : ℕ := 900)
  (pass_percent : ℝ := 0.32) :
  ((total_candidates - ((pass_percent * (total_candidates - girls)) + (pass_percent * girls))) / total_candidates) * 100 = 68 :=
by
  sorry

end exam_fail_percentage_l292_292173


namespace find_q_l292_292478

theorem find_q 
  (p q r s : ℝ)
  (h_poly : ∀ x : ℝ, g x = p * x^3 + q * x^2 + r * x + s)
  (h_roots : g (-2) = 0 ∧ g 0 = 0 ∧ g 2 = 0)
  (h_point : g 1 = -3) : 
  q = 0 :=
sorry

end find_q_l292_292478


namespace estevan_initial_blankets_l292_292263

theorem estevan_initial_blankets (B : ℕ) 
  (polka_dot_initial : ℕ) 
  (polka_dot_total : ℕ) 
  (h1 : (1 / 3 : ℚ) * B = polka_dot_initial) 
  (h2 : polka_dot_initial + 2 = polka_dot_total) 
  (h3 : polka_dot_total = 10) : 
  B = 24 := 
by 
  sorry

end estevan_initial_blankets_l292_292263


namespace min_fib_angle_l292_292460

def fibonacci : ℕ → ℕ
| 0     := 1
| 1     := 2
| (n+2) := fibonacci (n+1) + fibonacci n

theorem min_fib_angle 
  (a b : ℕ) 
  (h1 : a + b = 90) 
  (h2 : ∃ n m, fibonacci n = a ∧ fibonacci m = b)
  (h3 : a > b) : b = 1 := 
sorry

end min_fib_angle_l292_292460


namespace quadrilateral_area_l292_292400

open Real

theorem quadrilateral_area {APQ PQR PRS : Triangle} (APQ_right : ∀ P Q : Point, right_angle AP P Q)
  (PQR_right : ∀ P Q : Point, right_angle P Q R)
  (PRS_right : ∀ P Q : Point, right_angle P R S)
  (angles_60 : ∀ P Q R S : Point, angle PA Q = 60 ∧ angle QPR = 60 ∧ angle RPS = 60)
  (AP_length : length AP = 36)
  (triangle_type : ∀ T : Triangle, right_angle T ∧ angle_60 T → is_30_60_90 T) :
  area_quad AP RS = 405 + 20.25 * sqrt 3 := 
sorry

end quadrilateral_area_l292_292400


namespace first_part_results_count_l292_292139

theorem first_part_results_count : 
    ∃ n, n * 10 + 90 + (25 - n) * 20 = 25 * 18 ∧ n = 14 :=
by
  sorry

end first_part_results_count_l292_292139


namespace extreme_value_sufficient_but_not_necessary_l292_292192

variable {α : Type} [Real]

def has_extreme_value_at (f : α → α) (c : α) : Prop :=
  (∀ x, f x ≥ f c ∨ f x ≤ f c)

def differentiable_at (f : α → α) (c : α) : Prop :=
  ∃ f', ∀ x ≠ c, lim (h → 0) ((f (x + h) - f x) / h) = f' x

theorem extreme_value_sufficient_but_not_necessary (f : α → α) (c : α) 
  (h_diff_at_c : differentiable_at f c) : has_extreme_value_at f c →
  (∀ x, differentiable_at f x → deriv f x = 0) ↔ false :=
begin
  sorry
end

end extreme_value_sufficient_but_not_necessary_l292_292192


namespace smallest_sum_of_two_3_digit_numbers_l292_292882

/-- The smallest sum of two 3-digit numbers
that can be obtained by placing each of the six digits
3, 4, 5, 6, 7, 8 in one of the six boxes in an addition problem is 825.
-/
theorem smallest_sum_of_two_3_digit_numbers : ∃ a b c d e f : ℕ,
  (a = 3 ∧ b = 5 ∧ c = 7 ∧ d = 4 ∧ e = 6 ∧ f = 8) ∧
  (100 * a + 10 * b + c) + (100 * d + 10 * e + f) = 825 :=
by {
  use [3, 5, 7, 4, 6, 8],
  split,
  { exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩ },
  { dsimp, norm_num }
}

end smallest_sum_of_two_3_digit_numbers_l292_292882


namespace time_to_see_slow_train_l292_292502

noncomputable def time_to_pass (length_fast_train length_slow_train relative_time_fast seconds_observed_by_slow : ℕ) : ℕ := 
  length_slow_train * seconds_observed_by_slow / length_fast_train

theorem time_to_see_slow_train :
  let length_fast_train := 150
  let length_slow_train := 200
  let seconds_observed_by_slow := 6
  let expected_time := 8
  time_to_pass length_fast_train length_slow_train length_fast_train seconds_observed_by_slow = expected_time :=
by sorry

end time_to_see_slow_train_l292_292502


namespace sum_sqrt_inverse_eq_9_l292_292803

theorem sum_sqrt_inverse_eq_9 :
  (∑ n in Finset.range 99 \ Finset.range 1, 1 / (Real.sqrt (n + 2) + Real.sqrt (n + 1))) = 9 :=
by
  sorry

end sum_sqrt_inverse_eq_9_l292_292803


namespace power_function_decreasing_l292_292713

theorem power_function_decreasing (m : ℤ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x ^ ((m + 1) * (m - 2)))
    (h2 : f 3 > f 5) : f = (λ x, x ^ (-2)) := by
  sorry

end power_function_decreasing_l292_292713


namespace no_solution_for_t_and_s_l292_292607

theorem no_solution_for_t_and_s (m : ℝ) :
  (¬∃ t s : ℝ, (1 + 7 * t = -3 + 2 * s) ∧ (3 - 5 * t = 4 + m * s)) ↔ m = -10 / 7 :=
by
  sorry

end no_solution_for_t_and_s_l292_292607


namespace volume_of_pyramid_l292_292402

-- Definitions of conditions
variable (S A B C D : Point)
variable (area_SAB area_SBC area_SCD area_SDA : ℝ)
variable (angle_AB angle_BC angle_CD angle_DA : ℝ)
variable (area_quadrilateral : ℝ)

-- Given conditions
def conditions : Prop := 
  area_SAB = 9 ∧ area_SBC = 9 ∧ area_SCD = 27 ∧ area_SDA = 27 ∧ 
  angle_AB = angle_BC ∧ angle_BC = angle_CD ∧ angle_CD = angle_DA ∧ 
  area_quadrilateral = 36

-- Volume of the pyramid to be proven
theorem volume_of_pyramid (h : conditions S A B C D area_SAB area_SBC area_SCD area_SDA angle_AB angle_BC angle_CD angle_DA area_quadrilateral) : 
  volume_pyramid S A B C D = 54 :=
sorry

end volume_of_pyramid_l292_292402


namespace math_proof_problem_l292_292253

-- Define the conditions of the problem
def ellipse_symmetric (a b : ℝ) :=
  ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 ↔ (x = 3 ∧ y = -16/5) ∨ (x = -4 ∧ y = -12/5)

-- The canonical equation of the ellipse
def canonical_eq (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

-- Prove the main results given the conditions
theorem math_proof_problem :
  ∃ (a b : ℝ), ellipse_symmetric a b ∧
  (a = 5 ∧ b = 4) ∧
  (canonical_eq = λ x y, (x^2 / 25) + (y^2 / 16) = 1) ∧
  (√(25 - 16) = 3) ∧
  ((√(25 - 16)) / 5 = 0.6) ∧
  (∃ r1 r2 : ℝ, r1 = 6.8 ∧ r2 = 3.2) ∧
  (∃ d1 d2 : ℝ, d1 = 25/3 ∧ d2 = -25/3) :=
by
  sorry

end math_proof_problem_l292_292253


namespace theta_range_proof_l292_292676

noncomputable def theta_range := set.Icc (5 * Real.pi / 12) (13 * Real.pi / 12)

theorem theta_range_proof :
  ∀ (θ : ℝ), 
    (0 ≤ θ ∧ θ ≤ 2 * Real.pi) → 
    (∀ (x y : ℝ), (x - 2 * Real.cos θ) ^ 2 + (y - 2 * Real.sin θ) ^ 2 = 1 → x ≤ y) ↔ 
    θ ∈ theta_range := 
by
  sorry

end theta_range_proof_l292_292676


namespace infinite_set_of_midpoints_l292_292064

open Set

noncomputable theory

variable {E : Set (ℝ × ℝ)}

-- Conditions
def is_midpoint_of_two_others (E : Set (ℝ × ℝ)) : Prop :=
  ∀ (p ∈ E), ∃ (a b ∈ E), (a + b) / 2 = p

-- Theorem statement
theorem infinite_set_of_midpoints (h₁ : E.nonempty) (h₂ : is_midpoint_of_two_others E) : ¬ E.finite :=
sorry

end infinite_set_of_midpoints_l292_292064


namespace pq_squared_equation_l292_292991

-- Define the context and necessary points and lines
variables {O P A B C D Q : Type}
variables {circle : O} {lineOPA : P → A → Prop} {lineOPB : P → B → Prop}
variables {linePCD : P → C → D → Prop} {linePQAB : Q → A → B → Prop}

-- Define the tangency condition
def tangents (PA PB : Type) :=
  PA == PB

-- Define the intersection points conditions
def intersection (P Q : Type) :=
  ∃ (linePQA : P → Q → A → Prop) (linePQB : P → Q → B → Prop), linePQA ∧ linePQB

-- Define the proof goal as a theorem
theorem pq_squared_equation (PQ PC PD QC QD : ℝ) 
  (h1 : tangents PA PB) 
  (h2 : intersection P Q) :
  PQ^2 = PC * PD - QC * QD :=
sorry

end pq_squared_equation_l292_292991


namespace sum_powers_mod_l292_292879

theorem sum_powers_mod (n : ℕ) : (finset.range n).sum (λ k, (5^k : ℕ) % 7) % 7 = 1 :=
by
  sorry

end sum_powers_mod_l292_292879


namespace mike_taller_than_mark_l292_292787

def feet_to_inches (feet : ℕ) : ℕ := 12 * feet

def mark_height_feet := 5
def mark_height_inches := 3
def mike_height_feet := 6
def mike_height_inches := 1

def mark_total_height := feet_to_inches mark_height_feet + mark_height_inches
def mike_total_height := feet_to_inches mike_height_feet + mike_height_inches

theorem mike_taller_than_mark : mike_total_height - mark_total_height = 10 :=
by
  sorry

end mike_taller_than_mark_l292_292787


namespace digit_usage_problem_l292_292493

theorem digit_usage_problem (a : Finset ℕ) (h : a = {1, 2, 3, 4, 5}) (arrangements : ℕ) (h_arrangements : arrangements = 120) :
  ∃ x : ℕ, (∑ b in a, x!) / (x!)^(Finset.card a) = arrangements ∧ x = 1 :=
by
  sorry

end digit_usage_problem_l292_292493


namespace probability_reroll_two_dice_l292_292039

-- Defining the problem parameters
def dice_faces := {1, 2, 3, 4, 5, 6} -- Standard six-sided die

-- The event that Jason wins: sum of dice is 9
def wins (a b c : ℕ) : Prop := a + b + c = 9

-- Rerolling exactly two dice: Define as reroll(event winning)
def reroll_two_dice (a b c : ℕ) : Prop := 
  ∃ x y z : ℕ, (x ∈ dice_faces ∧ y ∈ dice_faces ∧ z ∈ dice_faces) ∧ 
  (x + z + y = 9 ∧ (x ≠ a ∨ y ≠ b ∨ z ≠ c))

-- Main problem: Calculate probability of rerolling exactly two dice and winning
theorem probability_reroll_two_dice : 
  (∃ (a b c : ℕ), a ∈ dice_faces ∧ b ∈ dice_faces ∧ c ∈ dice_faces ∧ reroll_two_dice a b c) → 
  (1 / 9 : ℝ) := 
sorry

end probability_reroll_two_dice_l292_292039


namespace find_b_collinearity_l292_292854

def collinear (A B C : ℝ × ℝ) : Prop := 
  (B.2 - A.2) * (C.1 - A.1) = (C.2 - A.2) * (B.1 - A.1)

theorem find_b_collinearity (b : ℝ) : 
  collinear (4, -6) (-b + 3, 4) (3b + 4, 3) ↔ b = -3 / 13 := 
by sorry

end find_b_collinearity_l292_292854


namespace magnitude_of_T_l292_292065

theorem magnitude_of_T : 
  let i := Complex.I
  let T := 3 * ((1 + i) ^ 15 - (1 - i) ^ 15)
  Complex.abs T = 768 := by
  sorry

end magnitude_of_T_l292_292065


namespace y_is_integer_fraction_is_simplified_y_greater_than_5_y_no_more_than_3_y_between_1_l292_292715

theorem y_is_integer (x : ℕ) (hx : x > 1) :
  (x = 2 ∨ x = 3 ∨ x = 6 ∨ x = 11) → (∃ y : ℕ, y = (x + 9) / (x - 1)) :=
by sorry

theorem fraction_is_simplified (x : ℚ) (hx : ∃ n : ℕ, x = n + 1/2) :
  let k := nat.floor x in
  k ≥ 3 ∧ k ∈ {3, 4, 7, 12, 22} → (∃ y : ℕ, y = 10 / (x - 1)) :=
by sorry

theorem y_greater_than_5 (x : ℚ) (hx : x > 1) :
  (x = 3 ∨ x = 2.5 ∨ x = 2) → (∃ y : ℚ, y = (x + 9) / (x - 1) ∧ y > 5) :=
by sorry

theorem y_no_more_than_3 (x : ℕ) (hx : x > 1) :
  (x ≥ 6) → (∃ y : ℕ, y = (x + 9) / (x - 1) ∧ y ≤ 3) :=
by sorry

theorem y_between_1.5_and_2.5 (x : ℕ) (hx : x > 1) :
  (15 ≤ 10 / (x - 1) ∧ 10 / (x-1) ≤ 2) → (∃ y : ℚ, y = (x + 9) / (x - 1)) :=
by sorry

end y_is_integer_fraction_is_simplified_y_greater_than_5_y_no_more_than_3_y_between_1_l292_292715


namespace workshop_total_workers_l292_292465

theorem workshop_total_workers 
  (avg_worker_salary : ℝ) 
  (techs_cnt : ℕ)
  (avg_tech_salary : ℝ) 
  (num_workers : ℕ)
  (avg_nontech_salary : ℝ) 
  (known_avg_salary: avg_worker_salary = 8000)
  (known_tech_salary: avg_tech_salary = 10000) 
  (num_techs: techs_cnt = 7)
  (known_nontech_salary: avg_nontech_salary = 6000):
  num_workers = 14 := by
  let S_t := techs_cnt * avg_tech_salary
  let S_n := (num_workers - techs_cnt) * avg_nontech_salary
  have total_salary : S_t + S_n = avg_worker_salary * num_workers
  sorry

end workshop_total_workers_l292_292465


namespace sum_of_solutions_eq_10_l292_292258

theorem sum_of_solutions_eq_10 :
  let eq1 := x^2 - 4*x + 3 = 0
  let eq2 := x^2 - 5*x + 5 = 1
  let eq3 := x^2 - 5*x + 5 = -1 in
  let solutions := {x : ℝ | eq1 ∨ eq2 ∨ eq3} in
  ∑ x in solutions.to_finset, x = 10 := by
  sorry

end sum_of_solutions_eq_10_l292_292258


namespace polygon_sides_l292_292483

theorem polygon_sides (n : ℕ) (z : ℕ) (h1 : z = n * (n - 3) / 2) (h2 : z = 3 * n) : n = 9 := by
  sorry

end polygon_sides_l292_292483


namespace smallest_in_list_l292_292565

theorem smallest_in_list (a b c d : ℤ) (h1 : a = 1) (h2 : b = -2) (h3 : c = 0) (h4 : d = -3) :
  ∀ x ∈ {a, b, c, d}, d ≤ x :=
by {
  intros x hx,
  simp at hx,
  -- by brute-force unfolding hypotheses and evaluating simplifications
  rcases hx with rfl | rfl | rfl | rfl; simp [h1, h2, h3, h4],
  sorry
}

end smallest_in_list_l292_292565


namespace square_perimeter_l292_292551

theorem square_perimeter (s : ℝ) 
  (h1 : ∃ s, ∀ r : ℝ, rectangle_perimeter s r = 40) :
  4 * s = 640 / 9 :=
by
  sorry

def rectangle_perimeter (side1 side2 : ℝ) : ℝ := 2 * (side1 + side2)

end square_perimeter_l292_292551


namespace problem_statement_l292_292639

noncomputable def ab_min_value (a b : ℝ) : Prop :=
  (3 / a + 2 / b = 2) ∧ (0 < a) ∧ (0 < b) → ab ≥ 6

theorem problem_statement (a b : ℝ) (h1 : 3 / a + 2 / b = 2) (h2 : 0 < a) (h3 : 0 < b) : (a * b ≥ 6) :=
  sorry

end problem_statement_l292_292639


namespace sum_of_even_factors_l292_292161

theorem sum_of_even_factors (n : ℕ) (h : n = 1176) :
  ∑ i in (finset.filter (λ d, even d) (finset.divisors n)), i = 3192 := 
by
  have fact_summary: 2^3 * 3 * 7^2 = 1176 := by norm_num
  rw ←h at fact_summary  -- Aligning given n to its prime factorization.
  sorry

end sum_of_even_factors_l292_292161


namespace compound_interest_doubling_time_l292_292623

theorem compound_interest_doubling_time :
  ∃ t : ℕ, (2 : ℝ) < (1 + (0.13 : ℝ))^t ∧ t = 6 :=
by
  sorry

end compound_interest_doubling_time_l292_292623


namespace stable_table_legs_count_l292_292185

theorem stable_table_legs_count (n : ℕ) (h : 0 < n) :
  ∑ m in finset.range (2*n + 1), (m + 1)^2 = (2 * n^3 + 6 * n^2 + 7 * n + 3) / 3 :=
by
  sorry

end stable_table_legs_count_l292_292185


namespace find_x_l292_292521

variable (α : Real)
variable (x : Real)
variable (r : Real := Real.sqrt (x^2 + 5))
variable (P : Real × Real := (x, Real.sqrt 5))

axiom cos_alpha_def : cos α = x / r
axiom cos_alpha_cond : cos α = (Real.sqrt 2 / 4) * x
axiom second_quadrant : π / 2 < α ∧ α < π

theorem find_x (h1 : cos α = x / r)
    (h2 : cos α = (Real.sqrt 2 / 4) * x)
    (h3 : π / 2 < α ∧ α < π) : x = -Real.sqrt 3 :=
sorry

end find_x_l292_292521


namespace biggest_shadow_cube_max_l292_292157

def max_shadow_area_of_cube (side_length : ℝ) : ℝ :=
  if side_length = 1 then sqrt 3 else 0

theorem biggest_shadow_cube_max (side_length : ℝ) (h : side_length = 1) :
  max_shadow_area_of_cube side_length = sqrt 3 :=
by
  sorry

end biggest_shadow_cube_max_l292_292157


namespace coloring_four_cells_with_diff_colors_l292_292314

theorem coloring_four_cells_with_diff_colors {n k : ℕ} (h : n ≥ 2) 
    (hk : k = 2 * n) 
    (color : fin n × fin n → fin k) 
    (hcolor : ∀ c, ∃ r c : fin k, ∃ a b : fin k, color (r, c) = a ∧ color (r, c) = b) :
    ∃ r1 r2 c1 c2, color (r1, c1) ≠ color (r1, c2) ∧ color (r1, c1) ≠ color (r2, c1) ∧
                    color (r2, c1) ≠ color (r2, c2) ∧ color (r1, c2) ≠ color (r2, c2) :=
by
  sorry

end coloring_four_cells_with_diff_colors_l292_292314


namespace smallest_number_l292_292518

-- Define the conditions
def is_divisible_by (n d : ℕ) : Prop := d ∣ n

def conditions (n : ℕ) : Prop := 
  (n > 12) ∧ 
  is_divisible_by (n - 12) 12 ∧ 
  is_divisible_by (n - 12) 24 ∧
  is_divisible_by (n - 12) 36 ∧
  is_divisible_by (n - 12) 48 ∧
  is_divisible_by (n - 12) 56

-- State the theorem
theorem smallest_number : ∃ n : ℕ, conditions n ∧ n = 1020 :=
by
  sorry

end smallest_number_l292_292518


namespace no_more_than_n_lines_divide_area_l292_292650

theorem no_more_than_n_lines_divide_area {n : ℕ} (polygon : ConvexPolygon n) (O : Point) :
  (∀ l : Line, divides_area l polygon O) → (counts_lines l ≤ n) := 
sorry

end no_more_than_n_lines_divide_area_l292_292650


namespace triangle_angle_A_l292_292010

theorem triangle_angle_A (A B C : ℝ) (a b c : ℝ)
  (h1 : 0 < B ∧ B < π)
  (h2 : a = sin B + cos B)
  (h3 : a = √2)
  (h4 : b = 2)
  (h5 : ∃ (A : ℝ), ∀ (hA : 0 < A ∧ A < π), (a / sin A = b / sin B)) :
  A = π / 6 :=
by sorry

end triangle_angle_A_l292_292010


namespace problem_statement_l292_292067

noncomputable def a := 9
noncomputable def b := 729

theorem problem_statement (h1 : ∃ (terms : ℕ), terms = 430)
                          (h2 : ∃ (value : ℕ), value = 3) : a + b = 738 :=
by
  sorry

end problem_statement_l292_292067


namespace zero_if_special_bound_l292_292053

open Complex

theorem zero_if_special_bound (z : ℂ) (h : ∀ k : ℕ, k ∈ {1, 2, 3} → |z^k + 1| ≤ 1) : z = 0 := 
  sorry

end zero_if_special_bound_l292_292053


namespace parallelogram_area_eq_l292_292354

theorem parallelogram_area_eq {a : ℝ} 
  (h1 : ∀ x y z : ℝ, x ≠ y → y ≠ z → x ≠ z)
  (h2 : ∀ x y z : ℝ, ∃ k : ℝ, k = a) : 
  (∀ K L M N : ℝ, K ≠ L ∧ M ≠ N ∧ K ≠ M ∧ L ≠ N) → 
  KLMN.area = a^2 * sqrt 2 :=
by
  sorry

end parallelogram_area_eq_l292_292354


namespace prove_XY_passes_through_O_l292_292775

variables {Point : Type} [EuclideanGeometry Point]
variables (e f b d a c : Line Point)
variables (A B C D O X Y : Point)

noncomputable def problem_statement :=
  (dist_point : e ∩ f = {O}) ∧ 
  (A ∈ e ∧ B ∈ e) ∧
  (C ∈ f ∧ D ∈ f) ∧ 
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ O ∧ B ≠ C ∧ B ≠ D ∧ B ≠ O ∧ C ≠ D ∧ C ≠ O ∧ D ≠ O) ∧
  (line_contains_point b B ∧ perpendicular_to_line b AC) ∧
  (line_contains_point d D ∧ perpendicular_to_line d AC) ∧ 
  (line_contains_point a A ∧ perpendicular_to_line a BD) ∧
  (line_contains_point c C ∧ perpendicular_to_line c BD) ∧
  (line_intersects a b = {X}) ∧
  (line_intersects c d = {Y}) 

theorem prove_XY_passes_through_O : 
  problem_statement e f b d a c A B C D O X Y → 
  line_contains_point XY O :=
sorry

end prove_XY_passes_through_O_l292_292775


namespace length_of_QR_l292_292033

theorem length_of_QR 
  (P Q R N : Type)
  (segment_PR : ℝ)
  (segment_PQ : ℝ)
  (segment_PN : ℝ)
  (midpoint_N : QR / 2 = N)
  (length_PR : segment_PR = 5)
  (length_PQ : segment_PQ = 7)
  (length_PN : segment_PN = 4) :
  ∃ x : ℝ, QR = 2 * real.sqrt 21 :=
sorry

end length_of_QR_l292_292033


namespace find_x_l292_292832

variables {x y z : ℝ}

theorem find_x (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 2) (h2 : y^2 / z = 3) (h3 : z^2 / x = 4) :
  x = 144^(1 / 5) :=
by
  sorry

end find_x_l292_292832


namespace inequality_proof_l292_292701

variable (a b c : ℝ)

theorem inequality_proof
  (h1 : a > b) :
  a * c^2 ≥ b * c^2 := 
sorry

end inequality_proof_l292_292701


namespace length_bf_l292_292741

/-- Given a quadrilateral ABCD with right angles at A and C,
    E and F are points on line segment AC,
    DE and BF are perpendicular to AC,
    AB and CD are extended to meet at point P,
    AE = 4, DE = 6, and CE = 8,
    Prove BF = 8.47. -/
theorem length_bf (A B C D E F P : Point)
    (h_quad : is_quadrilateral ABCD)
    (h_right_angles : ∠A = 90 ∧ ∠C = 90)
    (h_on_ac : E ∈ line A C ∧ F ∈ line A C)
    (h_perpendicular : is_perpendicular DE AC ∧ is_perpendicular BF AC)
    (h_meeting_point : ∃ P, is_meeting_point AB CD P)
    (h_AE : AE = 4)
    (h_DE : DE = 6)
    (h_CE : CE = 8) : 
  ∃ BF, BF = 8.47 :=
  sorry

end length_bf_l292_292741


namespace correct_propositions_l292_292678

-- Define the propositions
def prop_1 : Prop := ∃ α : ℝ, sin α + cos α = 3/2
def prop_2 : Prop := ∀ x : ℝ, sin (5 * π / 2 - 2 * x) = cos (2 * x)
def prop_3 : Prop := ∀ k : ℤ, x = π / 8 + k * π / 2 → sin (2 * x + 5 * π / 4) = sin (5 * π / 2)
def prop_4 : Prop := ∀ x : ℝ, (0 < x ∧ x < π / 2) → e^(sin (2 * x)) > e^(sin (2 * x)) -- This will need to be corrected; needs the precise interpretation from math.
def prop_5 : Prop := ∀ α β : ℝ, (0 < α ∧ α < π / 2 ∧ 0 < β ∧ β < π / 2 ∧ α > β) → tan α > tan β
def prop_6 : Prop := ∀ x : ℝ, 3 * sin (2 * x + π / 3) = 3 * sin (2 * (x + π / 6))

-- Define the main statement
theorem correct_propositions : prop_2 ∧ prop_3 :=
by
  sorry

end correct_propositions_l292_292678


namespace Bob_head_start_is_1_mile_l292_292995

-- define the necessary variables and conditions
def speed_Bob : ℝ := 6  -- Bob's speed in miles per hour
def speed_Jim : ℝ := 9  -- Jim's speed in miles per hour
def time_hours : ℝ := 1 / 3  -- Time in hours

-- distances run by Bob and Jim in the given time
def distance_Bob : ℝ := speed_Bob * time_hours
def distance_Jim : ℝ := speed_Jim * time_hours

-- definition of head-start
def head_start (x : ℝ) : Prop := distance_Jim = distance_Bob + x

-- theorem to be proven
theorem Bob_head_start_is_1_mile :
  ∃ x : ℝ, head_start x ∧ x = 1 :=
by
  use 1
  sorry

end Bob_head_start_is_1_mile_l292_292995


namespace units_digit_sum_l292_292897

theorem units_digit_sum (h₁ : (24 : ℕ) % 10 = 4) 
                        (h₂ : (42 : ℕ) % 10 = 2) : 
  ((24^3 + 42^3) % 10 = 2) :=
by
  sorry

end units_digit_sum_l292_292897


namespace platform_length_is_correct_l292_292512

noncomputable def length_of_platform 
  (train_length : ℝ) (crosses_platform_time : ℝ) (crosses_signal_time : ℝ) : ℝ :=
let v := train_length / crosses_signal_time in
let L := v * crosses_platform_time - train_length in
L

theorem platform_length_is_correct : 
  length_of_platform 300 51 18 = 550.17 := 
by
  unfold length_of_platform
  have v := 300 / 18
  have L := v * 51 - 300
  calc 
    v = 16.67 : sorry
    L = 850.17 - 300 : sorry
    550.17 = 550.17 : by norm_num
  done

end platform_length_is_correct_l292_292512


namespace triangle_perimeter_l292_292391

theorem triangle_perimeter (x : ℝ) (h1 : x = 9) (h2 : ∃ (y : ℝ), y = 13 ∨ y = 9 ∧ x = y) :
  x = 9 → 3 + 8 + x = 20 :=
by {
  intro h3,
  rw h3,
  exact rfl,
}

end triangle_perimeter_l292_292391


namespace asymptotes_of_hyperbola_l292_292683

theorem asymptotes_of_hyperbola (a b : ℝ) (h : a > b ∧ b > 0) (e : 2 = a / b) :
  ∃ h : b = sqrt ((2 * a)^2 - a^2), asymptote_eqn = (y = a * sqrt (3) * x) := 
sorry

end asymptotes_of_hyperbola_l292_292683


namespace find_f_at_1_l292_292035

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f'' : ℝ → ℝ := λ x, (derivative[2] f) x

theorem find_f_at_1 
  (h_diff : Differentiable ℝ f)
  (h_f'' : ∀ x, x > 0 → x ≠ 1 → (2 * f x + x * f'' x) / (x - 1) > 0)
  (h_tangent : (derivative f 1) = -3 / 4): 
  f 1 = 3 / 8 := 
sorry

end find_f_at_1_l292_292035


namespace non_intersecting_segments_l292_292088

theorem non_intersecting_segments (n : ℕ) (red_points blue_points : Fin n → (ℝ × ℝ))
  (h_no_collinear : ∀ (i j k : Fin n), i ≠ j → j ≠ k → k ≠ i → 
                       ¬ collinear {red_points i, red_points j, red_points k} ∧ 
                       ¬ collinear {blue_points i, blue_points j, blue_points k} ∧ 
                       ∀ (r : Fin n) (b1 b2 : Fin n), r ≠ b1 → b1 ≠ b2 → b2 ≠ r → 
                        ¬ collinear {red_points r, blue_points b1, blue_points b2}  ∧ 
                       ∀ (b : Fin n) (r1 r2 : Fin n), b ≠ r1 → r1 ≠ r2 → r2 ≠ b → 
                        ¬ collinear {blue_points b, red_points r1, red_points r2}) :
  ∃ (segments : Fin n → Fin n), injective segments ∧
    ∀ (i j : Fin n), i ≠ j → ¬ intersects (red_points i, blue_points (segments i)) 
                                  (red_points j, blue_points (segments j)) := 
by
  sorry

end non_intersecting_segments_l292_292088


namespace percentage_peanut_clusters_is_64_l292_292241

def total_chocolates := 50
def caramels := 3
def nougats := 2 * caramels
def truffles := caramels + 6
def other_chocolates := caramels + nougats + truffles
def peanut_clusters := total_chocolates - other_chocolates
def percentage_peanut_clusters := (peanut_clusters * 100) / total_chocolates

theorem percentage_peanut_clusters_is_64 :
  percentage_peanut_clusters = 64 := by
  sorry

end percentage_peanut_clusters_is_64_l292_292241


namespace speed_of_stream_l292_292509

theorem speed_of_stream (v c : ℝ) (h1 : c - v = 6) (h2 : c + v = 10) : v = 2 :=
by
  sorry

end speed_of_stream_l292_292509


namespace four_balls_three_boxes_l292_292634

theorem four_balls_three_boxes :
  ∃ (ways : ℕ), ways = (C 4 2) * (A 3 3) ∧
     (∀ (balls : fin 4 → ℕ), ∃ (boxes : fin 3 → set (fin 4)), 
         (∀ i, boxes i ≠ ∅) ∧ 
         (∀ b, b ∈ ⋃ i, boxes i) ∧ 
         ways = (C 4 2) * (A 3 3)) :=
sorry

end four_balls_three_boxes_l292_292634


namespace sqrt_minus_one_eq_five_l292_292933

theorem sqrt_minus_one_eq_five : sqrt ((-6)^2) - 1 = 5 := by
  sorry

end sqrt_minus_one_eq_five_l292_292933


namespace value_of_a_plus_d_l292_292705

theorem value_of_a_plus_d (a b c d : ℕ) (h1 : a + b = 16) (h2 : b + c = 9) (h3 : c + d = 3) : a + d = 10 := 
by 
  sorry

end value_of_a_plus_d_l292_292705


namespace solve_equation_l292_292104

theorem solve_equation (x : ℂ) : 
  (3 * x - 6) / (x + 2) + (3 * x ^ 2 - 12) / (3 - x) = 3 → 
  x = -2 + 2 * complex.I ∨ x = -2 - 2 * complex.I :=
by sorry

end solve_equation_l292_292104


namespace no_real_solution_l292_292457

theorem no_real_solution (x : ℝ) (h : x + 16 ≥ 0) : 
  ¬(sqrt (x + 16) - (8 / sqrt (x + 16)) + 1 = 7) :=
sorry

end no_real_solution_l292_292457


namespace player_A_wins_by_strategy_l292_292860

-- Definitions based on the conditions in the problem
noncomputable def neg_signs : list char := list.repeat '-' 1997

-- Define the game state with 1997 symbols
structure GameState :=
  (signs : list char)
  (length_1997 : signs.length = 1997)
  (counts_1997 : (signs.filter (λx, x = '-')).length = 1997)

-- Define the turns and moves
inductive Player
| A
| B

open Player

structure Move :=
  (signs : list char)
  (change_one : signs.length = 1 ∨ signs.length = 2)
  (only_change_neg_to_pos : ∀ s ∈ signs, s = '+')

def play_move (state : GameState) (move : Move) : GameState :=
  let new_signs := state.signs.map_with_index (λ i s, if (i < state.signs.length - move.signs.length + 1) ∧ (list.take move.signs.length (list.drop i state.signs) = list.repeat '-' move.signs.length) 
    then list.append move.signs (list.drop (i + move.signs.length) (list.take state.signs.length state.signs)) else s) in
  { GameState . signs := new_signs, length_1997 := state.length_1997, counts_1997 := sorry }

-- Prove player A wins by the strategy
theorem player_A_wins_by_strategy : ∀ state : GameState, ∀ move : Move, ∀ player : Player,
      (player = A) → (state.signs.nth 998 = some '-') →
      (play_move state move).signs.nth 998 = some '+' ∧ 
      (∀ opponent_move, play_move (play_move state move) opponent_move = sorry)
      :=
sorry

end player_A_wins_by_strategy_l292_292860


namespace joseph_drives_more_l292_292048

def joseph_speed : ℝ := 50
def joseph_time : ℝ := 2.5
def kyle_speed : ℝ := 62
def kyle_time : ℝ := 2

def joseph_distance : ℝ := joseph_speed * joseph_time
def kyle_distance : ℝ := kyle_speed * kyle_time

theorem joseph_drives_more : (joseph_distance - kyle_distance) = 1 := by
  sorry

end joseph_drives_more_l292_292048


namespace angle_B_is_90_l292_292748

def is_median (B H W : Point) : Prop :=
  B.distance H = H.distance W

def is_altitude (B O M : Point) : Prop :=
  ∃ l : Line, B ∈ l ∧ M ∈ l ∧ O ∈ l ∧ O ∈ perp l

def is_symmetrical (M O K : Point) : Prop :=
  dist O M = dist O K

def is_perpendicular (MP BH : Line) : Prop :=
  MP ⊥ BH

def conditions (BM BW MW : ℝ) (B M W O K P H : Point) : Prop :=
  BM < BW ∧ BW < MW ∧ is_altitude B O M ∧ is_median B H W ∧ is_symmetrical M O K ∧ ∃ l, K ∈ l ∧ l ⊥ MW ∧ ∃ p, P ∈ p ∧ P ∈ BW ∧ is_perpendicular MP BH

theorem angle_B_is_90 (BM BW MW : ℝ) (B M W O K P H : Point) :
  conditions BM BW MW B M W O K P H → ∠B = 90 :=
begin
  sorry -- Proof goes here
end

end angle_B_is_90_l292_292748


namespace polynomials_equal_constants_times_q_l292_292771

theorem polynomials_equal_constants_times_q
  (m : ℕ) (hm : m > 1)
  (a1 a2 a3 b1 b2 b3 : ℝ)
  (h1 : a1 ≠ 0) (h2 : a2 ≠ 0) (h3 : a3 ≠ 0)
  (h4 : ∀ x : ℝ, (a1 * x + b1)^m + (a2 * x + b2)^m = (a3 * x + b3)^m) :
  ∃ (c1 c2 c3 : ℝ) (q : ℝ → ℝ), (∀ x : ℝ, q x ∈ Polynomial ℝ) ∧ 
  (∀ x : ℝ, a1 * x + b1 = c1 * q x) ∧
  (∀ x : ℝ, a2 * x + b2 = c2 * q x) ∧
  (∀ x : ℝ, a3 * x + b3 = c3 * q x) := 
sorry

end polynomials_equal_constants_times_q_l292_292771


namespace ellipse_standard_eq_parabola_standard_eq_l292_292288

-- Proof for Question 1: Ellipse
theorem ellipse_standard_eq (a b c : ℝ) (h_a : a = 6) (h_b : b = 2 * real.sqrt 5) (h_c : c = 4) :
  a^2 = b^2 + c^2 →
  (λ x y : ℝ, y^2 / 36 + x^2 / 20 = 1) sorry

-- Proof for Question 2: Parabola
theorem parabola_standard_eq (x y : ℝ) (p : ℝ) (h : p = 3) :
  (λ x y : ℝ, y^2 = 12 * x) sorry

end ellipse_standard_eq_parabola_standard_eq_l292_292288


namespace f_log₂_20_l292_292293

noncomputable def f (x : ℝ) : ℝ := sorry -- This is a placeholder for the function f.

lemma f_neg (x : ℝ) : f (-x) = -f (x) := sorry
lemma f_shift (x : ℝ) : f (x + 1) = f (1 - x) := sorry
lemma f_special (x : ℝ) (hx : -1 < x ∧ x < 0) : f (x) = 2^x + 6 / 5 := sorry

theorem f_log₂_20 : f (Real.log 20 / Real.log 2) = -2 := by
  -- Proof details would go here.
  sorry

end f_log₂_20_l292_292293


namespace object_speed_approximation_l292_292373

   /-- Given that an object travels 200 feet in 4 seconds, 
       we need to prove the object's approximate speed in miles per hour. 
       Note: 1 mile = 5280 feet, 1 hour = 3600 seconds --/
   theorem object_speed_approximation 
     (distance_feet : ℝ) (time_seconds : ℝ) (feet_to_miles : ℝ) (seconds_to_hours : ℝ) : 
     distance_feet = 200 → 
     time_seconds = 4 → 
     feet_to_miles = 1 / 5280 → 
     seconds_to_hours = 1 / 3600 → 
     (distance_feet * feet_to_miles) / (time_seconds * seconds_to_hours) ≈ 34.09 :=
   by
     intros h_dist h_time h_feet h_seconds
     sorry
   
end object_speed_approximation_l292_292373


namespace grid_selection_l292_292729

theorem grid_selection (G : matrix (fin 2000) (fin 2000) ℤ)
  (H : ∀ i j, G i j = 1 ∨ G i j = -1)
  (sum_nonneg : 0 ≤ ∑ i j, G i j) :
  ∃ rows cols : finset (fin 2000),
    rows.card = 1000 ∧ cols.card = 1000 ∧
    1000 ≤ ∑ i in rows, ∑ j in cols, G i j :=
by sorry

end grid_selection_l292_292729


namespace triangle_right_angled_l292_292489

-- Define the variables and the condition of the problem
variables {a b c : ℝ}

-- Given condition of the problem
def triangle_condition (a b c : ℝ) : Prop :=
  2 * (a ^ 8 + b ^ 8 + c ^ 8) = (a ^ 4 + b ^ 4 + c ^ 4) ^ 2

-- The theorem to prove the triangle is right-angled
theorem triangle_right_angled (h : triangle_condition a b c) : a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2 :=
sorry

end triangle_right_angled_l292_292489


namespace problem_nabla_l292_292369

variable (a b : ℝ)
variable (h_a : a > 0) (h_b : b > 0)

def nabla (x y : ℝ) : ℝ := (x + y) / (1 + x * y)

theorem problem_nabla : (nabla (nabla 1 2) 3) = 1 :=
by {
  -- sorry: The proof details are omitted as per instructions
  sorry
}

end problem_nabla_l292_292369


namespace cartesian_equation_max_area_triangle_l292_292397

-- Definition of the Cartesian equation for the curve C2
def trajectory_equation (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Proof problem for Part (I)
theorem cartesian_equation (ρ θ : ℝ) (M P : ℝ × ℝ)
  (hC1 : ρ * Math.cos θ = 4)
  (hM_on_C1 : M = (ρ * Math.cos θ, ρ * Math.sin θ))
  (hP_on_segment_OM : ∃ k ∈ Icc 0 1, P = k • (0, 0) + (1 - k) • M)
  (hOM_OP : dist (0, 0) M * dist (0, 0) P = 16) :
  trajectory_equation P.1 P.2 := sorry

-- Proof problem for Part (II)
theorem max_area_triangle (A B : ℝ × ℝ)
  (hA_polar : A = (2 * Math.cos (π / 3), 2 * Math.sin (π / 3)))
  (hB_on_C2 : trajectory_equation B.1 B.2) :
  ∃ T : ℝ, T = 2 + sqrt 3 ∧
  let area := (1 / 2) * dist (0, 0) A * dist (2, 0) B in
  area = T := sorry

end cartesian_equation_max_area_triangle_l292_292397


namespace cos_half_pi_plus_theta_sin_double_alpha_plus_quarter_pi_l292_292036

/-
We assume the following conditions:
- The measure of angle θ is such that its terminal side lies on the ray y = (1/2) x for x ≤ 0.
- α is an angle such that cos(α + π/4) = sin(θ).
-/
noncomputable def θ_terminal_ray : Prop :=
∃ θ, ∃ r, r > 0 ∧ ∀ (x : ℝ) (hx : x = -2), ∀ (y : ℝ) (hy : y = -1),
  y = (1/2) * x

theorem cos_half_pi_plus_theta :
  θ_terminal_ray →
  ∀ θ, θ_terminal_ray →
  cos (π / 2 + θ) = sqrt 5 / 5 :=
begin
  intros h θ hθ,
  sorry
end

theorem sin_double_alpha_plus_quarter_pi :
  θ_terminal_ray →
  ∀ α θ, 
    (cos (α + π / 4) = sin θ) →
    (cos (π / 2 + θ) = sqrt 5 / 5) →
    (sin (2 * α + π / 4) = 7 * sqrt 2 / 10 ∨ sin (2 * α + π / 4) = -sqrt 2 / 10) :=
begin
  intros h α θ hcos htheta,
  sorry
end

end cos_half_pi_plus_theta_sin_double_alpha_plus_quarter_pi_l292_292036


namespace circles_intersecting_l292_292256
-- Import the full Mathlib library

-- Define the equations of the circles
def circle1_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 2 = 0
def circle2_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 1 = 0

-- Define the centers and radii
def center_radius_c1 : (ℝ × ℝ) × ℝ := ((-1, -1), 2)
def center_radius_c2 : (ℝ × ℝ) × ℝ := ((2, 1), 2)

-- Define the distance formula for the centers
def distance_centers : ℝ := Real.sqrt ((2 - (-1))^2 + (1 - (-1))^2)

-- Define the sum and absolute difference of radii
def sum_radii : ℝ := 2 + 2
def diff_radii : ℝ := abs (2 - 2)

-- Define the final problem statement
theorem circles_intersecting :
  (0 < distance_centers ∧ distance_centers < sum_radii ∧ distance_centers > diff_radii) :=
by
  sorry

end circles_intersecting_l292_292256


namespace derivative_of_f_l292_292006

variable (a : ℝ)

def f (x : ℝ) : ℝ := a^2 - cos x

theorem derivative_of_f (x : ℝ) : deriv (f a) x = sin x := by
  sorry

end derivative_of_f_l292_292006


namespace add_one_gt_add_one_l292_292644

theorem add_one_gt_add_one (a b c : ℝ) (h : a > b) : (a + c) > (b + c) :=
sorry

end add_one_gt_add_one_l292_292644


namespace inscribed_square_area_l292_292392

theorem inscribed_square_area (A B C : ℝ) 
  (h₁ : A = B)
  (h₂ : A^2 / 2 = 625) :
  let h := A * Real.sqrt 2,
      t := h / 2
  in t^2 = 625 :=
by 
  -- specify A and B are the legs of the isosceles right triangle,
  -- the given area of the initial inscribed square is used in the conditions.
  sorry

end inscribed_square_area_l292_292392


namespace movie_ticket_distribution_l292_292259

theorem movie_ticket_distribution :
  ∃ (dist : List (ℕ × ℕ)), 
  (length dist = 5) ∧ 
  (∃ i : ℕ, i < 4 ∧ ((dist[i].snd = dist[i+1].snd+1 ∨ dist[i].snd+1 = dist[i+1].snd))) ∧ 
  (dist.map Prod.fst).nodup ∧ 
  (dist.map Prod.snd).nodup ∧
  (dist.map Prod.fst ∈ [1, 2, 3, 4, 5]) ∧
  (dist.map Prod.snd == [A, B, 1, 2, 3, 4, 5] \ {A, B}) ∧
  bits ([A,B] as ℕ × (A,B) consecutive_pair) := 
  length (consecutive_pair.dist) = 2 →  4 × 2 × 6 = 48 :=
sorry

end movie_ticket_distribution_l292_292259


namespace eval_expression_l292_292266

theorem eval_expression (a x : ℕ) (h : x = a + 9) : x - a + 5 = 14 :=
by 
  sorry

end eval_expression_l292_292266


namespace leo_current_weight_l292_292001

variables (L K J : ℝ)

def condition1 := L + 12 = 1.7 * K
def condition2 := L + K + J = 270
def condition3 := J = K + 30

theorem leo_current_weight (h1 : condition1 L K)
                           (h2 : condition2 L K J)
                           (h3 : condition3 K J) : L = 103.6 :=
sorry

end leo_current_weight_l292_292001


namespace symmetry_of_graph_l292_292121

def f (x : ℝ) : ℝ := x - 1 / x

theorem symmetry_of_graph :
  ∀ x : ℝ, f (-x) = - (f x) := 
by
  sorry

end symmetry_of_graph_l292_292121


namespace prove_function_value_l292_292606

noncomputable def f : ℝ → ℝ
| x := if 0 ≤ x ∧ x ≤ 1 then x * (3 - 2 * x) else
        if 1 ≤ x ∧ x ≤ 4 then f(x - 4) else
        if x < 0 then -f(-x) else 0 -- Placeholder for general definition outside [0, 1]

-- The condition definitions
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x : ℝ, f (x + p) = f x

-- Proof problem statement
theorem prove_function_value :
  is_odd f ∧ (is_even (λ x, f (x + 1))) ∧ (∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x * (3 - 2 * x)) →
  f (31 / 2) = -1 :=
by
  -- Proof steps would go here
  sorry

end prove_function_value_l292_292606


namespace problem_a_plus_e_eq_zero_l292_292000

-- Let's define our function f
def f (a b c d e : ℝ) (x : ℝ) : ℝ :=
  (a * x^2 + b * x + c) / (d * x + e)

-- Now we state that f(f(x)) = x
def f_involution (a b c d e : ℝ) : Prop :=
  ∀ x, f a b c d e (f a b c d e x) = x

-- We now need to state the problem as a theorem
theorem problem_a_plus_e_eq_zero (a b c d e : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) (h5 : e ≠ 0) 
  (h6 : f_involution a b c d e) : a + e = 0 :=
sorry

end problem_a_plus_e_eq_zero_l292_292000


namespace match_polygon_possible_l292_292859

-- Define the problem constraints in Lean
def match_length : ℝ := 2
def number_of_matches : ℕ := 12
def total_length : ℝ := match_length * number_of_matches
def target_area : ℝ := 16

-- Define the proof statement
theorem match_polygon_possible : 
  (∃ (vertices : Type) [Fintype vertices] (coords : vertices → EuclideanSpace ℝ 2), 
    (∀ (e : vertices × vertices), edge_length e = match_length)
    ∧ (polygon_area coords = target_area)
    ∧ (polygon_perimeter coords = total_length)) :=
  sorry -- Proof to be provided

end match_polygon_possible_l292_292859


namespace problem1_problem2_l292_292235

/-- Problem 1: Prove that the given expression equals -6 --/
theorem problem1 : (- (1 / 2) ^ (-3) + | real.sqrt 3 - 2 | + (real.sqrt 3 - real.sqrt 2) * (real.sqrt 3 + real.sqrt 2) - (-2023) ^ 0 = -6) :=
by
  sorry

/-- Problem 2: Prove that (3-y)^2 + y^2 = 12 has roots (3 ± sqrt 15)/2 --/
theorem problem2 (y : ℝ) : ((3 - y) ^ 2 + y ^ 2 = 12) ↔ (y = (3 + real.sqrt 15) / 2 ∨ y = (3 - real.sqrt 15) / 2) :=
by
  sorry

end problem1_problem2_l292_292235


namespace largest_percentage_increase_l292_292230

def students_2003 := 80
def students_2004 := 88
def students_2005 := 94
def students_2006 := 106
def students_2007 := 130

theorem largest_percentage_increase :
  let incr_03_04 := (students_2004 - students_2003) / students_2003 * 100
  let incr_04_05 := (students_2005 - students_2004) / students_2004 * 100
  let incr_05_06 := (students_2006 - students_2005) / students_2005 * 100
  let incr_06_07 := (students_2007 - students_2006) / students_2006 * 100
  incr_06_07 > incr_03_04 ∧
  incr_06_07 > incr_04_05 ∧
  incr_06_07 > incr_05_06 :=
by
  -- Proof goes here
  sorry

end largest_percentage_increase_l292_292230


namespace scientific_notation_l292_292165

theorem scientific_notation (n : ℕ) (h : n = 2023000) : ∃ k, 2.023 * 10^k = n :=
by {
  use 6,
  rw h,
  -- Additional proof steps would go here
  sorry
}

end scientific_notation_l292_292165


namespace problem1_problem2_l292_292522

-- Problem 1
theorem problem1 (m : ℝ) (h : ∀ x : ℝ, (0 < x ∧ x < 2) → - (1 / 2) * x^2 + 2 * x > m * x) : m = 1 := 
sorry

-- Problem 2
theorem problem2 (A B C : ℝ) (h₀ : ∃ A B C, A + B + C = π)
                (h₁ : sin A = 5 / 13)
                (h₂ : cos B = 3 / 5) : cos C = -56 / 65 :=
sorry

end problem1_problem2_l292_292522


namespace no_nat_satisfying_n_squared_plus_5n_plus_1_divisible_by_49_l292_292516

theorem no_nat_satisfying_n_squared_plus_5n_plus_1_divisible_by_49 :
  ∀ n : ℕ, ¬ (∃ k : ℤ, (n^2 + 5 * n + 1) = 49 * k) :=
by
  sorry

end no_nat_satisfying_n_squared_plus_5n_plus_1_divisible_by_49_l292_292516


namespace number_of_registration_methods_l292_292496

theorem number_of_registration_methods :
  ∃ (f : Fin 5 → Fin 3), (∀ c : Fin 3, ∃ s : Fin 5, f s = c) ∧ (f.injective) ∧ 
  (∃ n : ℕ, n = 150) :=
sorry

end number_of_registration_methods_l292_292496


namespace gumballs_last_days_l292_292416

theorem gumballs_last_days :
  let g := 9 in
  let p1 := 3 in
  let p2 := 2 * p1 in
  let p3 := p2 - 1 in
  let G := (p1 * g) + (p2 * g) + (p3 * g) in
  let c := 3 in
  G / c = 42 :=
by
  sorry

end gumballs_last_days_l292_292416


namespace problem_nabla_l292_292368

variable (a b : ℝ)
variable (h_a : a > 0) (h_b : b > 0)

def nabla (x y : ℝ) : ℝ := (x + y) / (1 + x * y)

theorem problem_nabla : (nabla (nabla 1 2) 3) = 1 :=
by {
  -- sorry: The proof details are omitted as per instructions
  sorry
}

end problem_nabla_l292_292368


namespace total_tiles_l292_292208

theorem total_tiles (s : ℕ) (h_black_tiles : 2 * s - 1 = 75) : s^2 = 1444 :=
by {
  sorry
}

end total_tiles_l292_292208


namespace production_volume_increase_l292_292946

theorem production_volume_increase (x : ℝ) :
    200 + 200 * (1 + x) + 200 * (1 + x)^2 = 1400 :=
sorry

end production_volume_increase_l292_292946


namespace proof_of_integers_l292_292491

-- Define the given conditions
def m : ℕ := 21
def n : ℕ := 39

-- Define the conditions as constraints
def condition1 : m + n = 60 := by
  simp [m, n]
  exact add_eq_self (21 + 39)

def condition2 : Nat.lcm m n = 273 := by
  simp [Nat.lcm]
  -- Proof showing that lcm(21, 39) = 273 can involve checking the prime factors etc.
  sorry

-- The statement combining both conditions
theorem proof_of_integers (m n : ℕ) (h1 : m + n = 60) (h2 : Nat.lcm m n = 273) : m = 21 ∧ n = 39 :=
by
  simp [condition1, condition2]
  -- Equivalently prove m = 21 and n = 39
  sorry

end proof_of_integers_l292_292491


namespace prime_sums_count_is_2_l292_292129

-- Define the first 10 sums of consecutive primes starting at 3
def S_1 := 3
def S_2 := 3 + 5
def S_3 := 3 + 5 + 7
def S_4 := 3 + 5 + 7 + 11
def S_5 := 3 + 5 + 7 + 11 + 13
def S_6 := 3 + 5 + 7 + 11 + 13 + 17
def S_7 := 3 + 5 + 7 + 11 + 13 + 17 + 19
def S_8 := 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23
def S_9 := 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 + 29
def S_10 := 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 + 29 + 31

-- Define a predicate for primality
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Count how many of the sums are prime
def count_prime_sums : ℕ :=
  [S_1, S_2, S_3, S_4, S_5, S_6, S_7, S_8, S_9, S_10].count is_prime

-- Statement of the proof problem
theorem prime_sums_count_is_2 : count_prime_sums = 2 :=
by
  sorry

end prime_sums_count_is_2_l292_292129


namespace arithmetic_progression_possibility_l292_292134

theorem arithmetic_progression_possibility (S_n : ℕ) (d : ℕ) (h1 : S_n = 104) (h2 : d = 3) : 
  ∃ n, (n > 1) ∧ ∀ a, (104 = (n * (2 * a + (n - 1) * 3)) / 2) → ∃! n, n = 2 :=
by {
  use n,
  sorry
}

end arithmetic_progression_possibility_l292_292134


namespace spherical_coordinates_negate_y_l292_292963

theorem spherical_coordinates_negate_y :
  ∀ (x y z : ℝ), 
  x = 3 * sin (Real.pi / 3) * cos (5 * Real.pi / 6) →
  y = 3 * sin (Real.pi / 3) * sin (5 * Real.pi / 6) →
  z = 3 * cos (Real.pi / 3) →
  ∃ (ρ θ φ : ℝ), 
  ρ = 3 ∧ θ = 7 * Real.pi / 6 ∧ φ = Real.pi / 3 ∧ 
  (x, -y, z) = (ρ * sin φ * cos θ, ρ * sin φ * sin θ, ρ * cos φ) :=
begin
  intros x y z hx hy hz,
  use [3, 7 * Real.pi / 6, Real.pi / 3],
  split, exact rfl,
  split, exact rfl,
  split, exact rfl,
  sorry
end

end spherical_coordinates_negate_y_l292_292963


namespace floor_not_monotonically_increasing_l292_292115

def floor_def (x : ℝ) : ℤ := floor x

-- Prove that y = [x] is not monotonically increasing on ℝ
theorem floor_not_monotonically_increasing : ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ floor_def x₁ > floor_def x₂ :=
by
  sorry

end floor_not_monotonically_increasing_l292_292115


namespace intersection_of_A_and_B_l292_292350

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := { x | log 2 (x + 1 : ℝ) > 0 }

theorem intersection_of_A_and_B :
  A ∩ B = {1, 2} := 
sorry

end intersection_of_A_and_B_l292_292350


namespace bottle_volume_is_correct_l292_292476

noncomputable def calculate_bottle_volume
  (diameter : ℝ)
  (total_height : ℝ)
  (liquid_height_upright : ℝ)
  (liquid_height_inverted : ℝ)
  (pi_val : ℝ) : ℝ :=
  let radius := diameter / 2
  let effective_height := 22 -- derived from proportional behavior explained
  pi_val * radius ^ 2 * effective_height 

theorem bottle_volume_is_correct :
  calculate_bottle_volume 10 26 12 16 3.14 = 1727 :=
by
  have diameter := 10
  have total_height := 26
  have liquid_height_upright := 12
  have liquid_height_inverted := 16
  have pi_val := 3.14
  let result := calculate_bottle_volume diameter total_height liquid_height_upright liquid_height_inverted pi_val
  exact rfl

end bottle_volume_is_correct_l292_292476


namespace find_number_l292_292515

theorem find_number (x : ℝ) 
  (h1 : 0.15 * 40 = 6) 
  (h2 : 6 = 0.25 * x + 2) : 
  x = 16 := 
sorry

end find_number_l292_292515


namespace sequences_properties_l292_292427

def arithmetic_mean (a b : ℝ) := (a + b) / 2
def geometric_mean (a b : ℝ) := Real.sqrt (a * b)
def harmonic_mean (a b : ℝ) := 2 / ((1 / a) + (1 / b))

noncomputable def sequence_A : ℕ → ℝ
| 1     := arithmetic_mean x y
| (n+1) := arithmetic_mean (sequence_G n) (sequence_H n)

noncomputable def sequence_G : ℕ → ℝ
| 1     := geometric_mean x y
| (n+1) := geometric_mean (sequence_G n) (sequence_H n)

noncomputable def sequence_H : ℕ → ℝ
| 1     := harmonic_mean x y
| (n+1) := harmonic_mean (sequence_G n) (sequence_H n)

theorem sequences_properties (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) :
  (∀ n : ℕ, sequence_A x y n > sequence_A x y (n+1)) ∧
  (∀ n : ℕ, sequence_G x y n = sequence_G x y (n+1)) ∧
  (∀ n : ℕ, sequence_H x y n < sequence_H x y (n+1)) :=
sorry

end sequences_properties_l292_292427


namespace bug_probability_at_A_after_9_meters_l292_292763

/--
Let P(n) be the probability that a bug is at vertex A after crawling n meters in a regular tetrahedron with vertices A, B, C, and D, each edge measuring 1 meter, starting at vertex A. Each movement to an adjacent vertex has equal probability. If P(n+1) = 1/3 (1 - P(n)) with P(0) = 1, then prove that after 9 meters, n = 4920 when p = n / 19683 represents the probability the bug is back at vertex A.
-/
theorem bug_probability_at_A_after_9_meters :
  let P : ℕ → ℚ := λ n, if n = 0 then 1 else (1/3) * (1 - P (n - 1)) in
  P 9 = 1640 / 6561 → (∃ n : ℕ, 1640 / 6561 = n / 19683 ∧ n = 4920) :=
by {
  sorry,
}

end bug_probability_at_A_after_9_meters_l292_292763


namespace program_result_l292_292880

def program_loop (i : ℕ) (s : ℕ) : ℕ :=
if i < 9 then s else program_loop (i - 1) (s * i)

theorem program_result : 
  program_loop 11 1 = 990 :=
by 
  sorry

end program_result_l292_292880


namespace at_least_one_divisible_by_three_l292_292800

theorem at_least_one_divisible_by_three (n : ℕ) (h1 : n > 0) (h2 : n ≡ 99)
  (h3 : ∀ i, (i < n) → (∃ m : ℤ, abs (m(i+1) - m(i)) = 1 ∨ abs (m(i+1) - m(i)) = 2 ∨ m(i+1) = 2 * m(i))) :
  ∃ k, k ≤ 99 ∧ (k % 3 = 0) := sorry

end at_least_one_divisible_by_three_l292_292800


namespace hair_accessories_total_l292_292418

-- Define costs
def barrettes_cost : ℕ → ℝ := λ n, n * 4
def combs_cost : ℕ → ℝ := λ n, n * 2
def hairbands_cost : ℕ → ℝ := λ n, n * 3
def hairties_cost : ℕ → ℝ := λ n, n * 2.5

-- Define quantities
def kristine_quantities : (ℕ × ℕ × ℕ × ℕ) := (2, 3, 4, 5)
def crystal_quantities : (ℕ × ℕ × ℕ × ℕ) := (3, 2, 1, 7)

-- Calculate totals before discounts
def total_cost (quantities : (ℕ × ℕ × ℕ × ℕ)) : ℝ :=
  let (b, c, hb, ht) := quantities in
  barrettes_cost b + combs_cost c + hairbands_cost hb + hairties_cost ht

def kristine_total_before_discount : ℝ := total_cost kristine_quantities
def crystal_total_before_discount : ℝ := total_cost crystal_quantities

-- Define discounts
def discount (items : ℕ) (total : ℝ) : ℝ :=
  if items >= 11 then total * 0.15 else if items >= 6 then total * 0.10 else 0

def items_purchased (quantities : (ℕ × ℕ × ℕ × ℕ)) : ℕ :=
  let (b, c, hb, ht) := quantities in b + c + hb + ht

-- Calculate totals after discount
def total_after_discount (total : ℝ) (items : ℕ) : ℝ :=
  total - discount items total

def kristine_total_after_discount : ℝ :=
  total_after_discount kristine_total_before_discount (items_purchased kristine_quantities)

def crystal_total_after_discount : ℝ :=
  total_after_discount crystal_total_before_discount (items_purchased crystal_quantities)

-- Define sales tax
def sales_tax (total : ℝ) : ℝ := total * 0.085

-- Calculate final totals including tax
def final_total (total : ℝ) : ℝ := total + sales_tax total

def kristine_final_total : ℝ := final_total kristine_total_after_discount
def crystal_final_total : ℝ := final_total crystal_total_after_discount

-- Combined total
def combined_total : ℝ := kristine_final_total + crystal_final_total

-- Theorem statement
theorem hair_accessories_total :
  combined_total = 69.17 :=
by
  -- Computations would take place here to prove the theorem
  sorry

end hair_accessories_total_l292_292418


namespace triangle_is_right_l292_292254

noncomputable def triangle_sides := (3: ℝ, 4: ℝ, 5: ℝ)

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem triangle_is_right : 
  let (a, b, c) := triangle_sides in is_right_triangle a b c :=
by {
  have h1 : (3: ℝ)^2 + (4: ℝ)^2 = ((5: ℝ)^2), by {
    calc 
      (3: ℝ)^2 + (4: ℝ)^2 = 9 + 16 : by norm_num
      ... = 25 : by norm_num
      ... = (5: ℝ)^2 : by norm_num,
  },
  let trisides := triangle_sides,
  cases trisides with a trisides,
  cases trisides with b c,
  show is_right_triangle a b c, by {
    simp [is_right_triangle, h1],
  },
}

end triangle_is_right_l292_292254


namespace moles_of_NaNO3_formed_l292_292626

/- 
  Define the reaction and given conditions.
  The following assumptions and definitions will directly come from the problem's conditions.
-/

/-- 
  Represents a chemical reaction: 1 molecule of AgNO3,
  1 molecule of NaOH producing 1 molecule of NaNO3 and 1 molecule of AgOH.
-/
def balanced_reaction (agNO3 naOH naNO3 agOH : ℕ) := agNO3 = 1 ∧ naOH = 1 ∧ naNO3 = 1 ∧ agOH = 1

/-- 
  Proves that the number of moles of NaNO3 formed is 1,
  given 1 mole of AgNO3 and 1 mole of NaOH.
-/
theorem moles_of_NaNO3_formed (agNO3 naOH naNO3 agOH : ℕ)
  (h : balanced_reaction agNO3 naOH naNO3 agOH) :
  naNO3 = 1 := 
by
  sorry  -- Proof will be added here later

end moles_of_NaNO3_formed_l292_292626


namespace stripe_area_l292_292538

-- Definitions based on conditions:
def diameter : ℝ := 40
def height : ℝ := 90
def stripe_width : ℝ := 4
def revolutions : ℕ := 3

-- Lean theorem statement to prove the area of the stripe:
theorem stripe_area : 
  (π * diameter * revolutions * stripe_width) = 480 * π := by
  sorry

end stripe_area_l292_292538


namespace arithmetic_sequence_a1_a10_l292_292739

variable {α : Type*} [linear_ordered_field α]
variables (a : ℕ → α) (d : α) (a_2_eq_3 : a 2 = 3) (a_5_plus_a_7_eq_10 : a 5 + a 7 = 10)

#check a_2_eq_3

theorem arithmetic_sequence_a1_a10 (hn : ∀ n:ℕ, a (n+1) = a n + d) :
  a 1 + a 10 = 9.5 :=
by
  -- proof steps here
  sorry

end arithmetic_sequence_a1_a10_l292_292739


namespace triangle_side_eq_nine_l292_292061

theorem triangle_side_eq_nine (a b c : ℕ) 
  (h_tri_ineq : a + b > c ∧ a + c > b ∧ b + c > a)
  (h_sqrt_eq : (Nat.sqrt (a - 9)) + (b - 2)^2 = 0)
  (h_c_odd : c % 2 = 1) :
  c = 9 :=
sorry

end triangle_side_eq_nine_l292_292061


namespace min_value_of_expression_l292_292428

theorem min_value_of_expression (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) (hxyz : x * y * z = 27) :
  x + 3 * y + 6 * z >= 27 :=
by
  sorry

end min_value_of_expression_l292_292428


namespace ariel_fish_l292_292571

theorem ariel_fish (total_fish : ℕ) (male_ratio : ℚ) (female_ratio : ℚ) (female_fish : ℕ) : 
  total_fish = 45 ∧ male_ratio = 2/3 ∧ female_ratio = 1/3 → female_fish = 15 :=
by
  sorry

end ariel_fish_l292_292571


namespace fettuccine_to_penne_ratio_l292_292018

theorem fettuccine_to_penne_ratio
  (num_surveyed : ℕ)
  (num_spaghetti : ℕ)
  (num_ravioli : ℕ)
  (num_fettuccine : ℕ)
  (num_penne : ℕ)
  (h_surveyed : num_surveyed = 800)
  (h_spaghetti : num_spaghetti = 300)
  (h_ravioli : num_ravioli = 200)
  (h_fettuccine : num_fettuccine = 150)
  (h_penne : num_penne = 150) :
  num_fettuccine / num_penne = 1 :=
by
  sorry

end fettuccine_to_penne_ratio_l292_292018


namespace option_A_option_B_option_C_l292_292166

-- Defining the first function
def f₁ (x : ℝ) : ℝ := x^(-1 / 4)

-- Defining the second function
def f₂ (x : ℝ) : ℝ := x / (x + 1)

-- Defining the third function
def f₃ (x : ℝ) : ℝ := x + 9 / x

-- Defining the fourth function
def f₄ (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b

-- Proof statement for option A
theorem option_A : ∀ x ∈ Ioi 0, monotone_decreasing f₁ :=
sorry

-- Proof statement for option B
theorem option_B : ∀ x ∈ Ioi (-1), monotone_increasing f₂ :=
sorry

-- Proof statement for option C
theorem option_C {a b : ℝ} (h : a < b) :
  (∀ x ∈ Icc a b, monotone_decreasing f₃) →
  (∀ x ∈ Icc (-b) (-a), monotone_decreasing f₃) :=
sorry

end option_A_option_B_option_C_l292_292166


namespace determine_a_l292_292375

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (1 - 4 * m) * real.sqrt x

theorem determine_a (a : ℝ) (m : ℝ) 
  (h₀ : 0 < a ∧ a ≠ 1)
  (h₁ : ∀ x ∈ set.Icc (-2 : ℝ) 1, f a x ≤ 4)
  (h₂ : ∃ x ∈ set.Icc (-2 : ℝ) 1, f a x = 4)
  (h₃ : ∃ x ∈ set.Icc (-2 : ℝ) 1, f a x = m)
  (h₄ : ∀ x ∈ set.Ici 0, has_deriv_at (g m) ((1 - 4 * m) * (1 / (2 * real.sqrt x))) x) :
  a = 1 / 2 :=
sorry

end determine_a_l292_292375


namespace sin_double_angle_l292_292670

theorem sin_double_angle (θ : ℝ) 
    (h : Real.sin (Real.pi / 4 + θ) = 1 / 3) : 
    Real.sin (2 * θ) = -7 * Real.sqrt 2 / 9 :=
by
  sorry

end sin_double_angle_l292_292670


namespace area_of_triangle_ABF_l292_292110

-- Definitions of the problem conditions

variables (A B C D E F G : Point)
variables
  (A_eq := square A B C D)
  (area_ABCD := 256)
  (E_on_BC := E ∈ line_segment B C)
  (F_mid_AE : midpoint F A E)
  (G_mid_DE : midpoint G D E)
  (area_BEGF := 50)

-- Statement of the theorem
theorem area_of_triangle_ABF : 
  ∃ (a : ℝ), a = area (triangle A B F) ∧ a = 18 :=
by sorry

end area_of_triangle_ABF_l292_292110


namespace exists_divisible_by_3_l292_292795

open Nat

-- Definitions used in Lean 4 statement to represent conditions from part a)
def neighbors (n m : ℕ) : Prop := (m = n + 1) ∨ (m = n + 2) ∨ (2 * m = n) ∨ (m = 2 * n)

def circle_arrangement (ns : Fin 99 → ℕ) : Prop :=
  ∀ i : Fin 99, (neighbors (ns i) (ns ((i + 1) % 99)))

-- Proof problem:
theorem exists_divisible_by_3 (ns : Fin 99 → ℕ) (h : circle_arrangement ns) :
  ∃ i : Fin 99, 3 ∣ ns i :=
sorry

end exists_divisible_by_3_l292_292795


namespace units_digit_sum_cubes_l292_292903

theorem units_digit_sum_cubes (n1 n2 : ℕ) 
  (h1 : n1 = 24) (h2 : n2 = 42) : 
  (n1 ^ 3 + n2 ^ 3) % 10 = 6 :=
by 
  -- substitution based on h1 and h2 can be done here.
  sorry

end units_digit_sum_cubes_l292_292903


namespace complete_time_is_correct_l292_292186

def start_time : ℕ := 7 -- in hours, 7:00 AM
def quarter_completion_time : ℕ := 3 -- 3 hours from start
def maintenance_duration : ℕ := 30 -- in minutes

noncomputable def end_time : ℕ → ℕ → ℕ → string
| start_time, quarter_completion_time, maintenance_duration := 
  let total_time := 4 * quarter_completion_time in
  let effective_work_time := total_time + ((maintenance_duration : ℕ) / 60) in
  let completion_time := start_time + effective_work_time in
  if completion_time >= 12 then 
    let hour := completion_time - 12 in
    if (maintenance_duration : ℕ) % 60 = 0 then 
      toString hour ++ ":00 PM"
    else 
      toString hour ++ ":30 PM"
  else 
    toString completion_time ++ ":00 PM"

theorem complete_time_is_correct : end_time start_time quarter_completion_time maintenance_duration = "7:30 PM" :=
by
  sorry

end complete_time_is_correct_l292_292186


namespace function_is_odd_and_monotonically_decreasing_on_positive_real_interval_l292_292984

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def is_monotonically_decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a < x → x < y → y < b → f y ≤ f x

theorem function_is_odd_and_monotonically_decreasing_on_positive_real_interval :
  is_odd_function (λ x : ℝ, -x^3) ∧
  is_monotonically_decreasing_on_interval (λ x : ℝ, -x^3) 0 +∞ :=
sorry

end function_is_odd_and_monotonically_decreasing_on_positive_real_interval_l292_292984


namespace complex_sum_in_exponential_form_l292_292853

noncomputable def complex_sum (a b : ℂ) : (ℝ × ℝ) :=
  let r := abs (a + b)
  let theta := arg (a + b)
  (r, theta)

theorem complex_sum_in_exponential_form :
  let a : ℂ := 10 * Complex.exp (2 * Real.pi * Complex.I / 11)
  let b : ℂ := 10 * Complex.exp (15 * Real.pi * Complex.I / 22)
  complex_sum a b = (10 * Real.sqrt 2, 19 * Real.pi / 44) := 
by 
  sorry

end complex_sum_in_exponential_form_l292_292853


namespace amount_after_two_years_l292_292172

-- Define the conditions
def initial_amount : ℝ := 59000
def rate_of_increase : ℝ := 1 / 8
def duration : ℕ := 2

-- Define the formula to calculate the amount after given years
def amount_after_years (present_value : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  present_value * (1 + rate)^years

-- The theorem to be proved
theorem amount_after_two_years :
  amount_after_years initial_amount rate_of_increase duration = 74671.875 :=
  by sorry

end amount_after_two_years_l292_292172


namespace math_problem_l292_292999

theorem math_problem :
  (Int.ceil ((16 / 5 : ℚ) * (-34 / 4 : ℚ)) - Int.floor ((16 / 5 : ℚ) * Int.floor (-34 / 4 : ℚ))) = 2 :=
by
  sorry

end math_problem_l292_292999


namespace f_solution_l292_292712

noncomputable def f (x : ℝ) (h₁ : x ≠ 0) (h₂ : x ≠ 1) (h₃ : x ≠ -1) : ℝ := 
  f (x - 1 / x) = (x / (x^2 - 1)) - x^2 - 1 / x^2

theorem f_solution {x : ℝ} (h₁ : x ≠ 0) (h₂ : x ≠ 1) (h₃ : x ≠ -1) :
  f x h₁ h₂ h₃ = -x^2 + (1 / x) - 2 := sorry

end f_solution_l292_292712


namespace hall_reunion_attendees_l292_292499

noncomputable def Oates : ℕ := 40
noncomputable def both : ℕ := 10
noncomputable def total : ℕ := 100
noncomputable def onlyOates := Oates - both
noncomputable def onlyHall := total - onlyOates - both
noncomputable def Hall := onlyHall + both

theorem hall_reunion_attendees : Hall = 70 := by {
  sorry
}

end hall_reunion_attendees_l292_292499


namespace probability_divisible_by_7_p_plus_q_equals_1385_l292_292057

def is_valid_number (n : ℕ) : Prop :=
  n ≤ 2^30 ∧ (Integer.digits 2 n).count 1 = 3

def set_S := { n : ℕ | is_valid_number n }

def count_valid_numbers : ℕ := 99
def total_numbers_in_S : ℕ := 4060
def p : ℕ := 33
def q : ℕ := 1352

theorem probability_divisible_by_7 :
  (count_valid_numbers : ℝ) / total_numbers_in_S = (p : ℝ) / (q : ℝ) :=
sorry

theorem p_plus_q_equals_1385 : p + q = 1385 :=
sorry

end probability_divisible_by_7_p_plus_q_equals_1385_l292_292057


namespace total_wheels_and_cost_l292_292733

def ratio_unicycles : ℕ := 2
def ratio_bicycles : ℕ := 3
def ratio_tricycles : ℕ := 4
def ratio_quadricycles : ℕ := 1
def total_people : ℕ := 40
def cost_unicycle : ℕ := 50
def cost_bicycle : ℕ := 100
def cost_tricycle : ℕ := 150
def cost_quadricycle : ℕ := 200

theorem total_wheels_and_cost :
  let part := total_people / (ratio_unicycles + ratio_bicycles + ratio_tricycles + ratio_quadricycles) in
  let num_unicycles := ratio_unicycles * part in
  let num_bicycles := ratio_bicycles * part in
  let num_tricycles := ratio_tricycles * part in
  let num_quadricycles := ratio_quadricycles * part in
  let total_wheels := num_unicycles * 1 + num_bicycles * 2 + num_tricycles * 3 + num_quadricycles * 4 in
  let total_cost := num_unicycles * cost_unicycle + num_bicycles * cost_bicycle + num_tricycles * cost_tricycle + num_quadricycles * cost_quadricycle in
  total_wheels = 96 ∧ total_cost = 4800 :=
by 
  let part := total_people / (ratio_unicycles + ratio_bicycles + ratio_tricycles + ratio_quadricycles)
  let num_unicycles := ratio_unicycles * part
  let num_bicycles := ratio_bicycles * part
  let num_tricycles := ratio_tricycles * part
  let num_quadricycles := ratio_quadricycles * part
  let total_wheels := num_unicycles * 1 + num_bicycles * 2 + num_tricycles * 3 + num_quadricycles * 4
  let total_cost := num_unicycles * cost_unicycle + num_bicycles * cost_bicycle + num_tricycles * cost_tricycle + num_quadricycles * cost_quadricycle
  show total_wheels = 96 ∧ total_cost = 4800 from sorry

end total_wheels_and_cost_l292_292733


namespace john_finishes_fourth_task_at_120_pm_l292_292041

def time := ℕ -- This represents time in minutes from a fixed reference point (e.g., midnight).

-- Conditions
def start_time : time := 9 * 60 -- 9:00 AM in minutes
def end_of_third_task : time := 12 * 60 + 15 -- 12:15 PM in minutes
def n_tasks : ℕ := 4 -- number of tasks
def duration_of_each_task : time := (end_of_third_task - start_time) / (n_tasks - 1) -- duration of each task

-- Function to calculate the end time of the nth task
def end_time_of_nth_task (n : ℕ) : time :=
  start_time + n * duration_of_each_task

-- Theorem to prove
theorem john_finishes_fourth_task_at_120_pm :
  end_time_of_nth_task 4 = 13 * 60 + 20 := by
  sorry

end john_finishes_fourth_task_at_120_pm_l292_292041


namespace crease_length_l292_292204

-- Define the conditions
def width (rectangle: Type) : ℝ := 8
def inward_fold (rectangle: Type) : ℝ := 2
def angle (theta : Type) : ℝ → ℝ

-- Define the proof problem
theorem crease_length (rectangle: Type) (theta : ℝ) : 
  (∃ l : ℝ, l = 8 * sqrt 2 * tan θ) := sorry

end crease_length_l292_292204


namespace no_more_than_five_planes_can_arrive_l292_292013

theorem no_more_than_five_planes_can_arrive 
  (airfields : Fin 100 → α)
  (distance : α → α → ℝ)
  (h_distinct : ∀ i j, i ≠ j → distance (airfields i) (airfields j) ≠ distance (airfields i) (airfields k))
  (nearest_airfield : Fin 100 → Fin 100)
  (h_nearest : ∀ i, nearest_airfield i ≠ i ∧ 
    ∀ j ≠ i, distance (airfields i) (airfields (nearest_airfield i)) < distance (airfields i) (airfields j)) 
  :
  ∀ i, finset.card (finset.univ.filter (λ j, nearest_airfield j = i)) ≤ 5 := 
by
  sorry

end no_more_than_five_planes_can_arrive_l292_292013


namespace find_parabola_standard_eq_l292_292287

noncomputable def point := (ℝ, ℝ)
noncomputable def line := { p : point | ∃a b c, a * (fst p) + b * (snd p) + c = 0 }

def parabola_eqns (p1 p2 : point) : Prop :=
  (fst p1) ^ 2 = 16 * (snd p1)
  ∨ (snd p2) ^ 2 = -8 * (fst p2)

theorem find_parabola_standard_eq :
  ∃ (p : point) (l : line),
  p = (-3, 2) ∧ l = { p : point | ∃ a b c, a * (fst p) + b * (snd p) + c = 0, a = 1, b = -2, c = -4 } ∧
  ∃ f1 f2 : point, f1 = (4, 0) ∨ f2 = (0, -2) ∧ parabola_eqns f1 f2 :=
sorry

end find_parabola_standard_eq_l292_292287


namespace sin_cos_sum_l292_292640

variable {θ : ℝ}
variable (h1 : sin θ * cos θ = 2 / 5)
variable (h2 : real.sqrt (cos θ ^ 2) = -cos θ)

theorem sin_cos_sum :
  sin θ + cos θ = - (3 * real.sqrt 5) / 5 :=
by
  sorry

end sin_cos_sum_l292_292640


namespace angle_between_rays_at_3_and_7_l292_292947

theorem angle_between_rays_at_3_and_7 :
  ∀ (clock_face : ℕ),
    clock_face = 12 →
    (∃ θ : ℝ, θ = 30) →
    (∀ i : ℕ, 1 ≤ i ∧ i ≤ 12 →  ∃ θ : ℝ, θ = 30 * i) →
    (∀ a b : ℕ, 3 = a ∧ 7 = b → ∃ θ : ℝ, θ = 30 * (b - a)) →
    ∃ angle : ℝ, angle = 120 :=
by
  assume clock_face hcf hθ hi hab,
  sorry

end angle_between_rays_at_3_and_7_l292_292947


namespace triangle_APQ_equilateral_l292_292615

-- Definitions of the geometric constructs from the conditions
structure Point where
  x : ℝ
  y : ℝ

def Triangle (A B C : Point) : Prop := 
  A ≠ B ∧ B ≠ C ∧ C ≠ A

def Equilateral (A B C : Point) : Prop := 
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B

def Midpoint (P A B : Point) : Prop := 
  P.x = (A.x + B.x)/2 ∧ P.y = (A.y + B.y)/2

-- Given conditions as per the problem
variable (A B C C1 B1 A1 P Q : Point)
variable (h1 : Equilateral A B C1)
variable (h2 : Equilateral A B1 C)
variable (h3 : Equilateral A1 B C)
variable (h4 : Midpoint P A1 B1)
variable (h5 : Midpoint Q A1 C1)

-- The goal to prove
theorem triangle_APQ_equilateral : Equilateral A P Q := by
  sorry

end triangle_APQ_equilateral_l292_292615


namespace selling_price_l292_292168

theorem selling_price (cost_price : ℝ) (loss_percentage : ℝ) : 
    cost_price = 1600 → loss_percentage = 0.15 → 
    (cost_price - (loss_percentage * cost_price)) = 1360 :=
by
  intros h_cp h_lp
  rw [h_cp, h_lp]
  norm_num

end selling_price_l292_292168


namespace find_a_plus_b_l292_292370

variables (a b c d x : ℝ)

def conditions (a b c d x : ℝ) : Prop :=
  (a + b = x) ∧
  (b + c = 9) ∧
  (c + d = 3) ∧
  (a + d = 5)

theorem find_a_plus_b (a b c d x : ℝ) (h : conditions a b c d x) : a + b = 11 :=
by
  have h1 : a + b = x := h.1
  have h2 : b + c = 9 := h.2.1
  have h3 : c + d = 3 := h.2.2.1
  have h4 : a + d = 5 := h.2.2.2
  sorry

end find_a_plus_b_l292_292370


namespace circumcircle_equation_l292_292301

theorem circumcircle_equation (A B : ℝ × ℝ) (hA : A = (3, 2)) (hB : B = (-1, 5))
    (C : ℝ × ℝ) (hC : C ∈ {p : ℝ × ℝ | 3 * p.1 - p.2 + 3 = 0})
    (h_area : ∃ d : ℝ, S_triangle A B C = 10 ∧ S_triangle A B C = 1/2 * dist A B * d) :
  (∃ D E F, ∀ p : ℝ × ℝ, p ∈ {A, B, C} → (p.1^2 + p.2^2 + D * p.1 + E * p.2 + F = 0)) ∧
  ((D = -1/2 ∧ E = -5 ∧ F = -3/2) ∨ (D = -25/6 ∧ E = -89/9 ∧ F = 347/18)) :=
sorry -- Proof to be constructed

end circumcircle_equation_l292_292301


namespace area_of_S_l292_292964

-- Define the given conditions
def regular_octagon_center_mathematical_properties 
  (d : ℝ) : Prop :=
  (d = 2)

-- Define the region outside the octagon and transformation set S
def region_outside_octagon (R : set ℂ) : Prop :=
  ∃ z ∈ R, abs (complex.re z) ≥ 2 / (1 + real.sqrt 2) ∨ 
            abs (complex.im z) ≥ 2 / (1 + real.sqrt 2)

def set_S (S R : set ℂ) : Prop :=
  ∀ z : ℂ, z ∈ S ↔ ∃ w : ℂ, w ∈ R ∧ z = 1 / w

-- The theorem to prove the area of S
theorem area_of_S 
  (d : ℝ) (R S : set ℂ) 
  (h_octagon : regular_octagon_center_mathematical_properties d) 
  (h_region : region_outside_octagon R)
  (h_transformation : set_S S R) :
  ∃ area : ℝ, area = π / 2 := 
by
  sorry

end area_of_S_l292_292964


namespace tile_covering_l292_292205

theorem tile_covering :
  let tile_width : ℕ := 5
  let tile_height : ℕ := 7
  let region_width : ℕ := 36
  let region_length : ℕ := 84
  let tile_area : ℕ := tile_width * tile_height
  let region_area : ℕ := region_width * region_length
  (region_area / tile_area).ceil = 87 := by
  sorry

end tile_covering_l292_292205


namespace cos_C_in_right_triangle_l292_292721

theorem cos_C_in_right_triangle
  (A B C : Type)
  [has_angle A]
  [has_angle B]
  [has_angle C]
  (hA : has_angle.angle A = 90)
  (hT : has_tangent.tangent C = 4) :
  has_cosine.cosine C = sqrt 17 / 17 := 
by sorry

end cos_C_in_right_triangle_l292_292721


namespace similar_triangles_l292_292966

-- Define the similarity condition between the triangles
theorem similar_triangles (x : ℝ) (h₁ : 12 / x = 9 / 6) : x = 8 := 
by sorry

end similar_triangles_l292_292966


namespace new_triangle_cannot_be_formed_l292_292032

theorem new_triangle_cannot_be_formed (PQ PR QR : ℝ) (h1 : PQ = 8) (h2 : PR = 6) (h3 : QR = 9) :
  let PQ' := 8 * 1.5
      PR' := 6 * (1 - 0.333)
      QR' := QR in
  ¬(PQ' + PR' > QR' ∧ PQ' + QR' > PR' ∧ PR' + QR' > PQ') :=
by
  let PQ' := 12
  let PR' := 4
  let QR' := 9
  show ¬(PQ' + PR' > QR' ∧ PQ' + QR' > PR' ∧ PR' + QR' > PQ')
  sorry

end new_triangle_cannot_be_formed_l292_292032


namespace ellipse_distance_l292_292281

noncomputable def distance_between_left_vertex_and_right_focus {a b c : ℝ} 
  (h1 : a = 4) 
  (h2 : b = 2 * Real.sqrt 3) 
  (h3 : c = 2) 
  (ellipse_eq : ∀ x y : ℝ, x^2 / 16 + y^2 / 12 = 1) 
  : Real :=
  a + c

theorem ellipse_distance : ∀ x y : ℝ, 
  (x^2 / 16 + y^2 / 12 = 1) →
  ∃ (d : ℝ), d = distance_between_left_vertex_and_right_focus 4 (2 * Real.sqrt 3) 2 (λ _ _, 1 = 1) ∧ d = 6 :=
by {
  intros x y ellipse_eq,
  use distance_between_left_vertex_and_right_focus 4 (2 * Real.sqrt 3) 2 ellipse_eq,
  split,
  { refl },
  { exact rfl }
} 

end ellipse_distance_l292_292281


namespace find_a_b_l292_292842

def f (x a b : ℝ) := x^3 - a*x^2 - b*x + a^2

theorem find_a_b (a b : ℝ) :
  (∀ x : ℝ, deriv (f x a b) = 3*x^2 - 2*a*x - b)
  ∧ (deriv (f 1 a b) = 0) 
  ∧ (f 1 a b = 10)
  → (a = -4 ∧ b = 11) :=
by
  sorry

end find_a_b_l292_292842


namespace sum_of_number_and_reverse_divisible_by_11_l292_292450

theorem sum_of_number_and_reverse_divisible_by_11 (A B : ℕ) (hA : 0 ≤ A) (hA9 : A ≤ 9) (hB : 0 ≤ B) (hB9 : B ≤ 9) :
  11 ∣ ((10 * A + B) + (10 * B + A)) :=
by
  sorry

end sum_of_number_and_reverse_divisible_by_11_l292_292450


namespace direct_sum_span_subspaces_l292_292757

section representation_p_group

variables {G : Type*} [group G] [fintype G] {V : Type*} [add_comm_group V] [module ℝ V]
variable {ρ : G →* linear_map ℝ V V}
variable {W : submodule ℝ V}

theorem direct_sum_span_subspaces 
  (h_char : char_p ℝ p) 
  (h_injective : function.injective ((fintype.univ.sum (λ g : G, ρ g)).comp_submodule W)) :
  ∀ g1 g2 : G, g1 ≠ g2 → disjoint (ρ g1 '' W) (ρ g2 '' W) :=
sorry

end representation_p_group

end direct_sum_span_subspaces_l292_292757


namespace only_two_greater_than_one_l292_292567

theorem only_two_greater_than_one (a b c d : ℤ) (h₀ : a = 0) (h₁ : b = 2) (h₂ : c = -1) (h₃ : d = -3) :
  ∀ x ∈ {a, b, c, d}, x > 1 ↔ x = b := 
by
  sorry

end only_two_greater_than_one_l292_292567


namespace circle_area_circle_circumference_l292_292645

section CircleProperties

variable (r : ℝ) -- Define the radius of the circle as a real number

-- State the theorem for the area of the circle
theorem circle_area (A : ℝ) : A = π * r^2 :=
sorry

-- State the theorem for the circumference of the circle
theorem circle_circumference (C : ℝ) : C = 2 * π * r :=
sorry

end CircleProperties

end circle_area_circle_circumference_l292_292645


namespace c_symmetry_l292_292250

-- The function c satisfying the given conditions
noncomputable def c : ℕ → ℕ → ℕ
| n, 0   => 1
| n, k   => if k = n then 1 else (2 * k * c n k + c n (k - 1))

-- The theorem to show c(n, k) = c(n - k, k)
theorem c_symmetry (n k : ℕ) : c n k = c (n - k) k := 
sorry

end c_symmetry_l292_292250


namespace david_chemistry_marks_l292_292605

theorem david_chemistry_marks :
  let english := 96
  let mathematics := 95
  let physics := 82
  let biology := 95
  let average_all := 93
  let total_subjects := 5
  let total_other_subjects := english + mathematics + physics + biology
  let total_all_subjects := average_all * total_subjects
  let chemistry := total_all_subjects - total_other_subjects
  chemistry = 97 :=
by
  -- Definition of variables
  let english := 96
  let mathematics := 95
  let physics := 82
  let biology := 95
  let average_all := 93
  let total_subjects := 5
  let total_other_subjects := english + mathematics + physics + biology
  let total_all_subjects := average_all * total_subjects
  let chemistry := total_all_subjects - total_other_subjects

  -- Assert the final value
  show chemistry = 97
  sorry

end david_chemistry_marks_l292_292605


namespace min_value_of_f_l292_292777

theorem min_value_of_f (a b : ℝ) (m : ℝ) 
  (h1 : ∀ a b, z = a - 3b → z ≥ m):
  ∃ x, f(x) = 2 := 
by 
  sorry

end min_value_of_f_l292_292777


namespace min_colors_rect_condition_l292_292309

theorem min_colors_rect_condition (n : ℕ) (hn : n ≥ 2) :
  ∃ k : ℕ, (∀ (coloring : Fin n → Fin n → Fin k), 
           (∀ i j, coloring i j < k) → 
           (∀ c, ∃ i j, coloring i j = c) →
           (∃ i1 i2 j1 j2, i1 ≠ i2 ∧ j1 ≠ j2 ∧ 
                            coloring i1 j1 ≠ coloring i1 j2 ∧ 
                            coloring i1 j1 ≠ coloring i2 j1 ∧ 
                            coloring i1 j2 ≠ coloring i2 j2 ∧ 
                            coloring i2 j1 ≠ coloring i2 j2)) → 
           k = 2 * n :=
sorry

end min_colors_rect_condition_l292_292309


namespace prob_xi_less_2mu_plus_1_l292_292776

-- Define the conditions
variables {μ σ : ℝ}
variable {ξ : ℝ → ℝ}
variable hξ : ∀ x, ξ x = pdf (Normal μ σ) x
variable P1 : cdf (Normal μ σ) (-1) = 0.3
variable P2 : cdf (Normal μ σ) 2 = 0.7  -- derived from P(ξ > 2) = 0.3

-- Translate the question and conditions into a Lean statement
theorem prob_xi_less_2mu_plus_1 :
  P (λ x, ξ x < 2 * μ + 1) = 0.7 :=
sorry

end prob_xi_less_2mu_plus_1_l292_292776


namespace monotonic_decrease_interval_l292_292846

noncomputable def interval_of_monotonic_decrease := 
  ∀ (x : ℝ), x ∈ Icc (-π) 0 → (2 * sin (2 * x + π / 6) is decreasing on Icc (-5 * π / 6) (-π / 3))

theorem monotonic_decrease_interval :
  ∀ (x : ℝ), x ∈ Icc (-π) 0 ↔ x ∈ Icc (-5 * π / 6) (-π / 3) :=
by
  sorry

end monotonic_decrease_interval_l292_292846


namespace fraction_multiplication_simplifies_l292_292156

theorem fraction_multiplication_simplifies :
  (3 : ℚ) / 4 * (4 / 5) * (2 / 3) = 2 / 5 := 
by 
  -- Prove the equality step-by-step
  sorry

end fraction_multiplication_simplifies_l292_292156


namespace find_K4_minus_one_edge_l292_292052

variable (V : Type) [Fintype V] [DecidableEq V]

structure SimpleGraph (V : Type _) :=
(adj : V → V → Prop)
(sym : symmetric adj . obviously)
(loopless : irreflexive adj . obviously)

namespace SimpleGraph

def card {V : Type _} [Fintype V] [DecidableEq V] (G : SimpleGraph V) : Nat :=
Fintype.card V

def edge_count {V : Type _} [Fintype V] [DecidableEq V] (G : SimpleGraph V) : Nat :=
Fintype.card (G.edge V)

def contains_K4_minus_one_edge (G : SimpleGraph V) : Prop :=
∃ (a b c d : V), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  G.adj a b ∧ G.adj a c ∧ G.adj b c ∧ G.adj a d ∧ G.adj b d ∧ ¬ G.adj c d

end SimpleGraph

theorem find_K4_minus_one_edge (G : SimpleGraph V) (n : Nat)
  (h_vertices : G.card = 2 * n)
  (h_edges : G.edge_count = n^2 + 1) :
  G.contains_K4_minus_one_edge :=
sorry

end find_K4_minus_one_edge_l292_292052


namespace part1_part2_l292_292679

-- Define the function f(x)
def f (x : ℝ) := sqrt 3 * sin (2 * x) - 2 * cos x ^ 2

-- Part 1: Prove that f(π/6) = 0
theorem part1 : f (π / 6) = 0 := by
  sorry

-- Part 2: Prove that f(x) is monotonically increasing in intervals
-- [ -π/6 + kπ, π/3 + kπ ] for k in ℤ
theorem part2 (k : ℤ) : ∀ x, -π / 6 + k * π ≤ x ∧ x ≤ π / 3 + k * π → (∀ x y, x < y → f x ≤ f y) := by
  sorry

end part1_part2_l292_292679


namespace min_degree_of_polynomial_with_rational_coeffs_l292_292112

noncomputable def polynomial_with_rational_coeffs := 
  {p : Polynomial ℚ | (p.eval (3 - Real.sqrt 8) = 0) ∧ 
                      (p.eval (5 + Real.sqrt 13) = 0) ∧ 
                      (p.eval (16 - 2 * Real.sqrt 10) = 0) ∧ 
                      (p.eval (- Real.sqrt 3) = 0)}

theorem min_degree_of_polynomial_with_rational_coeffs : 
  ∀ (p : polynomial_with_rational_coeffs), 
  Polynomial.degree p.to_fun ≥ 8 :=
sorry

end min_degree_of_polynomial_with_rational_coeffs_l292_292112


namespace square_area_l292_292973

theorem square_area (d : ℝ) (h : d = 12 * Real.sqrt 2) : ∃ (A : ℝ), A = 144 :=
by
  use 144
  sorry

end square_area_l292_292973


namespace card_trick_successful_l292_292513

-- Define the problem conditions
inductive Card : Type
| mk : ℕ → Card

def deck := {c : Card // ∃ n, c = Card.mk n ∧ 1 ≤ n ∧ n ≤ 52}

noncomputable def select_five_cards (d : finset deck) : finset deck := sorry

-- Prove that the card trick always works
theorem card_trick_successful : 
  ∀ (d : finset deck) (h : d.cardinality = 52),
    ∃ (selected : finset deck) (facedown : Card), 
    select_five_cards d = selected ∧ facedown ∈ selected ∧ 
    (∀ faceup ∈ selected.erase facedown, faceup ∈ selected)  → true := 
  sorry

end card_trick_successful_l292_292513


namespace sum_digits_l292_292629

def repeat_pattern (d: ℕ) (n: ℕ) : ℕ :=
  let pattern := if d = 404 then 404 else if d = 707 then 707 else 0
  pattern * 10^(n / 3)

def N1 := repeat_pattern 404 101
def N2 := repeat_pattern 707 101
def P := N1 * N2

def thousands_digit (n: ℕ) : ℕ :=
  (n / 1000) % 10

def units_digit (n: ℕ) : ℕ :=
  n % 10

theorem sum_digits : thousands_digit P + units_digit P = 10 := by
  sorry

end sum_digits_l292_292629


namespace possible_vertex_angles_of_isosceles_triangle_l292_292461

def isosceles_triangle (α β γ : ℝ) : Prop := (α = β) ∨ (β = γ) ∨ (γ = α)

def altitude_half_side (α β γ a b c : ℝ) : Prop :=
  (a = α / 2) ∨ (b = β / 2) ∨ (c = γ / 2)

theorem possible_vertex_angles_of_isosceles_triangle (α β γ a b c : ℝ) :
  isosceles_triangle α β γ →
  altitude_half_side α β γ a b c →
  α = 30 ∨ α = 120 ∨ α = 150 :=
by
  sorry

end possible_vertex_angles_of_isosceles_triangle_l292_292461


namespace square_garden_perimeter_l292_292209

theorem square_garden_perimeter (A : ℝ) (s : ℝ) (N : ℝ) 
  (h1 : A = 9)
  (h2 : s^2 = A)
  (h3 : N = 4 * s) 
  : N = 12 := 
by
  sorry

end square_garden_perimeter_l292_292209


namespace probability_of_two_science_questions_is_correct_l292_292141

/-- Define the scenario with 5 questions in total - 3 science and 2 humanities - drawing 2 questions sequentially without replacement -/
def total_questions := 5
def science_questions := 3
def humanities_questions := 2
def drawn_questions := 2

/-- Define the probability calculation -/
def probability_two_science_questions : ℚ :=
  (science_questions * (science_questions - 1)).to_rational / 
  ((total_questions * (total_questions - 1)).to_rational)

/-- Theorem stating that the probability of drawing two science questions is 3/10 -/
theorem probability_of_two_science_questions_is_correct :
  probability_two_science_questions = 3 / 10 :=
by
  sorry

end probability_of_two_science_questions_is_correct_l292_292141


namespace area_ratio_of_triangles_l292_292176

theorem area_ratio_of_triangles :
  let A := (2, 0)
  let B := (8, 12)
  let C := (14, 0)
  let X := (6, 0)
  let Y := (8, 4)
  let Z := (10, 0)
  let area_triangle := λ p q r : ℝ × ℝ, (1 / 2) * abs (p.1 * (q.2 - r.2) + q.1 * (r.2 - p.2) + r.1 * (p.2 - q.2))
  c = area_triangle X Y Z / area_triangle A B C
  : c = 1 / 9
:= sorry

end area_ratio_of_triangles_l292_292176


namespace ed_marbles_l292_292614

theorem ed_marbles (doug_initial_marbles : ℕ) (marbles_lost : ℕ) (ed_doug_difference : ℕ) 
  (h1 : doug_initial_marbles = 22) (h2 : marbles_lost = 3) (h3 : ed_doug_difference = 5) : 
  (doug_initial_marbles + ed_doug_difference) = 27 :=
by
  sorry

end ed_marbles_l292_292614


namespace roger_daily_goal_l292_292826

-- Conditions
def steps_in_30_minutes : ℕ := 2000
def time_to_reach_goal_min : ℕ := 150
def time_interval_min : ℕ := 30

-- Theorem to prove
theorem roger_daily_goal : steps_in_30_minutes * (time_to_reach_goal_min / time_interval_min) = 10000 := by
  sorry

end roger_daily_goal_l292_292826


namespace smallest_semicircle_area_l292_292089

theorem smallest_semicircle_area (x : ℝ) (h1 : x^2 < 180) (h2 : 3x < 180) : x^2 + 3x = 180 → x^2 = 144 :=
by
  sorry

end smallest_semicircle_area_l292_292089


namespace bike_travel_distance_l292_292938

-- Declaring the conditions as definitions
def speed : ℝ := 50 -- Speed in meters per second
def time : ℝ := 7 -- Time in seconds

-- Declaring the question and expected answer
def expected_distance : ℝ := 350 -- Expected distance in meters

-- The proof statement that needs to be proved
theorem bike_travel_distance : (speed * time = expected_distance) :=
by
  sorry

end bike_travel_distance_l292_292938


namespace find_x_l292_292296

theorem find_x (y : ℝ) (x : ℝ) : 
  (5 + 2*x) / (7 + 3*x + y) = (3 + 4*x) / (4 + 2*x + y) ↔ 
  x = (-19 + Real.sqrt 329) / 16 ∨ x = (-19 - Real.sqrt 329) / 16 :=
by
  sorry

end find_x_l292_292296


namespace sum_of_k_for_distinct_integer_roots_l292_292883

theorem sum_of_k_for_distinct_integer_roots :
  (∑ k in {k : ℤ | ∃ p q : ℤ, p ≠ q ∧ p * q = 15 ∧ p + q = k}, k) = 0 :=
by {
  sorry
}

end sum_of_k_for_distinct_integer_roots_l292_292883


namespace fraction_of_3_over_4_is_1_over_5_is_4_over_15_l292_292872

theorem fraction_of_3_over_4_is_1_over_5_is_4_over_15 : 
  (∃ x : ℚ, (x * (3/4) = (1/5)) ∧ x = (4/15)) := 
begin
  use 4/15,
  split,
  { -- show x * (3/4) = (1/5)
    sorry },
  { -- show x = 4/15
    refl },
end

end fraction_of_3_over_4_is_1_over_5_is_4_over_15_l292_292872


namespace sin_angle_GAC_eq_sqrt3_div_3_l292_292182

-- Define the points of the cube
variables (A B C D E F G H : ℝ × ℝ × ℝ)
-- Additional necessary conditions (e.g., A = (0,0,0) would be implicitly provided in context)

-- Define the edge length for the sides of the cube
variable (s : ℝ)
variable (h : ∀ {A B C D E F G H : ℝ × ℝ × ℝ}, ∃(s : ℝ), s > 0 ∧  

-- Represent the cube structure
structure cube (A B C D E F G H : ℝ × ℝ × ℝ) :=
  (edge_length : ℝ)
  (edges      : list (ℝ × ℝ × ℝ))
  (equal_edges : ∀ e ∈ edges, e = edge_length)

variable {c : cube A B C D E F G H}
-- The proposition to be proven: the value of sin(GAC) is sqrt(3)/3
theorem sin_angle_GAC_eq_sqrt3_div_3 : 
  let GAC := ∠((s, 0, 0), (0, s, 0), (s, s, s)) in
  sin GAC = (√3) / 3 :=
sorry

end sin_angle_GAC_eq_sqrt3_div_3_l292_292182


namespace similar_triangles_l292_292967

-- Define the similarity condition between the triangles
theorem similar_triangles (x : ℝ) (h₁ : 12 / x = 9 / 6) : x = 8 := 
by sorry

end similar_triangles_l292_292967


namespace uki_total_earnings_l292_292153

def cupcake_price : ℝ := 1.50
def cookie_price : ℝ := 2.00
def biscuit_price : ℝ := 1.00
def daily_cupcakes : ℕ := 20
def daily_cookies : ℕ := 10
def daily_biscuits : ℕ := 20
def days : ℕ := 5

theorem uki_total_earnings :
  5 * ((daily_cupcakes * cupcake_price) + (daily_cookies * cookie_price) + (daily_biscuits * biscuit_price)) = 350 :=
by
  -- This is a placeholder for the proof
  sorry

end uki_total_earnings_l292_292153


namespace possible_values_of_n_l292_292915

noncomputable def circle_center : ℝ × ℝ := (5 / 2, 0)
noncomputable def circle_radius : ℝ := 5 / 2
noncomputable def point_P : ℝ × ℝ := (5 / 2, 3 / 2)
noncomputable def shortest_chord_length : ℝ := 4
noncomputable def longest_chord_length : ℝ := 5

def in_range (d : ℝ) : Prop := 1 / 6 ≤ d ∧ d ≤ 1 / 3
noncomputable def n_values : set ℕ := {n | ∃ d, in_range (1 / (n - 1 : ℝ)) ∧ 4 + (n - 1) * d = 5}

theorem possible_values_of_n : n_values = {4, 5, 6, 7} :=
sorry

end possible_values_of_n_l292_292915


namespace geometric_series_sum_l292_292586

theorem geometric_series_sum (a r : ℝ) (h : |r| < 1) (h_a : a = 2 / 3) (h_r : r = 2 / 3) :
  ∑' i : ℕ, (a * r^i) = 2 :=
by
  sorry

end geometric_series_sum_l292_292586


namespace calculate_nabla_l292_292366

def nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

theorem calculate_nabla :
  nabla (nabla 1 2) 3 = 1 :=
by
  unfold nabla
  have h1 : nabla 1 2 = 1 := by
    simp [nabla]
    rw [add_mul, one_mul, add_comm (1 + 2)]
  rw [h1, nabla]
  simp [nabla]
  rw [add_comm (1 + 3)]
  sorry

end calculate_nabla_l292_292366


namespace function_divisibility_l292_292758

theorem function_divisibility
    (f : ℤ → ℕ)
    (h_pos : ∀ x, 0 < f x)
    (h_div : ∀ m n : ℤ, (f m - f n) % f (m - n) = 0) :
    ∀ m n : ℤ, f m ≤ f n → f m ∣ f n :=
by sorry

end function_divisibility_l292_292758


namespace Emily_spent_28_dollars_l292_292569

theorem Emily_spent_28_dollars :
  let roses_cost := 4
  let daisies_cost := 3
  let tulips_cost := 5
  let lilies_cost := 6
  let roses_qty := 2
  let daisies_qty := 3
  let tulips_qty := 1
  let lilies_qty := 1
  (roses_qty * roses_cost) + (daisies_qty * daisies_cost) + (tulips_qty * tulips_cost) + (lilies_qty * lilies_cost) = 28 :=
by
  sorry

end Emily_spent_28_dollars_l292_292569


namespace ellipse_tangent_to_rectangle_satisfies_equation_l292_292837

theorem ellipse_tangent_to_rectangle_satisfies_equation
  (a b : ℝ) -- lengths of the semi-major and semi-minor axes of the ellipse
  (h_rect : 4 * a * b = 48) -- the area condition (since the rectangle sides are 2a and 2b)
  (h_ellipse_form : ∃ (a b : ℝ), ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) : 
  a = 4 ∧ b = 3 ∨ a = 3 ∧ b = 4 := 
sorry

end ellipse_tangent_to_rectangle_satisfies_equation_l292_292837


namespace min_colors_rect_condition_l292_292308

theorem min_colors_rect_condition (n : ℕ) (hn : n ≥ 2) :
  ∃ k : ℕ, (∀ (coloring : Fin n → Fin n → Fin k), 
           (∀ i j, coloring i j < k) → 
           (∀ c, ∃ i j, coloring i j = c) →
           (∃ i1 i2 j1 j2, i1 ≠ i2 ∧ j1 ≠ j2 ∧ 
                            coloring i1 j1 ≠ coloring i1 j2 ∧ 
                            coloring i1 j1 ≠ coloring i2 j1 ∧ 
                            coloring i1 j2 ≠ coloring i2 j2 ∧ 
                            coloring i2 j1 ≠ coloring i2 j2)) → 
           k = 2 * n :=
sorry

end min_colors_rect_condition_l292_292308


namespace problem1_problem2_problem3_l292_292344

-- Define f(x) as given in the problem
def f (x : ℝ) (a : ℝ) : ℝ := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x

-- Problem 1
theorem problem1 (a : ℝ) (h_slope : 6 * a = 3) : a = 1/2 :=
by
  sorry

-- Problem 2
theorem problem2 (a : ℝ) : 
  (∀ x : ℝ, x > 0 → f(x) a + f(-x) a ≥ 12 * Real.log x) → a ≤ -1 - (1/Real.exp 1) :=
by
  sorry

-- Problem 3
noncomputable def M (a : ℝ) : ℝ := 
  if 1 < a ∧ a ≤ 5/3 then 4 else if 5/3 < a ∧ a < 2 then 3*a - 1 else 3*a - 1

noncomputable def m (a : ℝ) : ℝ :=
  if 1 < a ∧ a ≤ 5/3 then -a^3 + 3 * a^2 else if 5/3 < a ∧ a < 2 then -a^3 + 3 * a^2 else 4

noncomputable def h (a : ℝ) : ℝ := M(a) - m(a)

theorem problem3 (ha : 1 < a) : ∃ a, h(a) = 8/27 :=
by
  sorry

end problem1_problem2_problem3_l292_292344


namespace mike_taller_than_mark_l292_292786

def feet_to_inches (feet : ℕ) : ℕ := 12 * feet

def mark_height_feet := 5
def mark_height_inches := 3
def mike_height_feet := 6
def mike_height_inches := 1

def mark_total_height := feet_to_inches mark_height_feet + mark_height_inches
def mike_total_height := feet_to_inches mike_height_feet + mike_height_inches

theorem mike_taller_than_mark : mike_total_height - mark_total_height = 10 :=
by
  sorry

end mike_taller_than_mark_l292_292786


namespace find_x_value_l292_292843

noncomputable def infinite_geometric_sequence_sum (x : ℝ) : ℝ := 
  if h : |sin x| < 1 then sin x / (1 - sin x) else 0

theorem find_x_value (x : ℝ) (h1 : 0 < x) (h2 : x < π) (h3 : infinite_geometric_sequence_sum x = 1) :
  x = π / 6 ∨ x = 5 * π / 6 :=
by
  sorry

end find_x_value_l292_292843


namespace volume_ratio_pyramid_cylinder_l292_292743

-- Define the problem conditions
variables (a : ℝ)  -- Edge length of the pyramid
variables (h : ℝ)  -- Height of the pyramid
variables (r : ℝ)  -- Radius of the cylinder base
variables (v_pyramid v_cylinder : ℝ)  -- Volumes of the pyramid and the cylinder respectively

-- Given conditions
def pyramid_edge_twice_height : Prop := a = 2 * h
def orthogonal_projection_rectangles : Prop := True  -- This needs rigorous geometric definitions
def base_circle_pass_through_center : Prop := True  -- This needs rigorous geometric definitions

-- Volumes computed under the conditions
def volume_pyramid (a h : ℝ) : ℝ := (1/3) * (a^2) * h
def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * (a/√3)

-- The proof statement
theorem volume_ratio_pyramid_cylinder (h a r : ℝ) (h1 : pyramid_edge_twice_height a h) (h2 r3 : orthogonal_projection_rectangles) (h3 : base_circle_pass_through_center) :
  v_cylinder / v_pyramid = (π / √3) :=
sorry

end volume_ratio_pyramid_cylinder_l292_292743


namespace series_sum_l292_292802

theorem series_sum:
  let S := ∑ k in [1, 2, 3, 4, 5], (2 + k) / (k * (k + 1)) * (1 / 2^k) 
  in S = 1 - 1 / (6 * 2^5) := 
sorry

end series_sum_l292_292802


namespace shortest_distance_from_point_to_parabola_l292_292285

-- Define the parabola
def parabola (y : ℝ) : ℝ := y^2 / 4

-- Define the distance function
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Define the function to calculate the distance from the point (4,10) to the parabola point (b^2/4, b)
noncomputable def point_to_parabola_distance (b : ℝ) : ℝ := distance 4 10 (b^2 / 4) b

-- The proof problem
theorem shortest_distance_from_point_to_parabola : ∃ (b : ℝ), point_to_parabola_distance b = real.sqrt 95.0625 :=
by
  sorry

end shortest_distance_from_point_to_parabola_l292_292285


namespace distinct_A_B_C_placements_l292_292972

theorem distinct_A_B_C_placements : 
  let board := fin 3 × fin 3 in
  let positions := { (r, c) | r ∈ fin 3 ∧ c ∈ fin 3 } in
  let letter := { 'A', 'B', 'C' } in
  (∃ f : letter → board, injective f ∧ (∀ l1 l2 : letter, l1 ≠ l2 → (f l1).1 ≠ (f l2).1 )) :=
  fintype.card (finset.filter (λ f, function.injective f ∧ ∀ (l1 l2 : letter), l1 ≠ l2 → (f l1).fst ≠ (f l2).fst)
    (finset.univ : finset (letter → board))) = 648 := 
sorry

end distinct_A_B_C_placements_l292_292972


namespace no_bijection_exists_bijection_l292_292055

open Nat

-- Define the binary relations
def binRel1 (a b : ℕ) : Prop := test_bit b a = tt
def binRel2 (a b : ℕ) : Prop := ∃ n : ℕ, b = ((prime (a + 1))) * n

-- Define the negation statement for part (i)
theorem no_bijection (f : ℕ → ℕ) :
  ¬(bijective f ∧ ∀ a b : ℕ, binRel1 a b ↔ binRel2 (f a) (f b)) :=
sorry

-- Define the existence statement for part (ii)
theorem exists_bijection :
  ∃ g : ℕ → ℕ, bijective g ∧ ∀ a b : ℕ, (binRel1 a b ∨ binRel1 b a) ↔ (binRel2 (g a) (g b) ∨ binRel2 (g b) (g a)) :=
sorry

end no_bijection_exists_bijection_l292_292055


namespace f_properties_l292_292671

noncomputable def f (x : ℝ) : ℝ :=
if -2 < x ∧ x < 0 then 2^x else sorry

theorem f_properties (f_odd : ∀ x : ℝ, f (-x) = -f x)
                     (f_periodic : ∀ x : ℝ, f (x + 3 / 2) = -f x) :
  f 2014 + f 2015 + f 2016 = 0 :=
by 
  -- The proof will go here
  sorry

end f_properties_l292_292671


namespace length_AP_of_inscribed_circle_l292_292022

theorem length_AP_of_inscribed_circle
  (A B C D : Point)
  (omega : Circle)
  (M P : Point)
  (AB_eq_2 : AB.length = 2)
  (BC_eq_1 : BC.length = 1)
  (circle_inscribed : omega.isInscribedIn ABCD)
  (M_on_CD : M ∈ (CD : Line))
  (AM_intersects_omega : ∃ P, P ≠ M ∧ P ∈ (AM : Line) ∧ P ∈ omega) :
  dist A P = (sqrt 2) / 2 := by
  sorry

end length_AP_of_inscribed_circle_l292_292022


namespace expected_value_of_coins_l292_292543

def coin_values : List ℕ := [1, 5, 10, 25, 50]
def coin_prob : ℕ := 2
def expected_value (values : List ℕ) (prob : ℕ) : ℝ :=
  values.sum * (1.0 / prob)

theorem expected_value_of_coins : expected_value coin_values coin_prob = 45.5 := by
  sorry

end expected_value_of_coins_l292_292543


namespace find_least_positive_x_l292_292282

theorem find_least_positive_x :
  ∃ x : ℕ, x + 5419 ≡ 3789 [MOD 15] ∧ x = 5 :=
by
  use 5
  constructor
  · sorry
  · rfl

end find_least_positive_x_l292_292282


namespace units_digit_sum_l292_292895

theorem units_digit_sum (h₁ : (24 : ℕ) % 10 = 4) 
                        (h₂ : (42 : ℕ) % 10 = 2) : 
  ((24^3 + 42^3) % 10 = 2) :=
by
  sorry

end units_digit_sum_l292_292895


namespace at_least_one_divisible_by_three_l292_292799

theorem at_least_one_divisible_by_three (n : ℕ) (h1 : n > 0) (h2 : n ≡ 99)
  (h3 : ∀ i, (i < n) → (∃ m : ℤ, abs (m(i+1) - m(i)) = 1 ∨ abs (m(i+1) - m(i)) = 2 ∨ m(i+1) = 2 * m(i))) :
  ∃ k, k ≤ 99 ∧ (k % 3 = 0) := sorry

end at_least_one_divisible_by_three_l292_292799


namespace min_n_for_Sn_gt_2_l292_292399

theorem min_n_for_Sn_gt_2 :
  (∃ (a b : ℕ → ℕ) (S : ℕ → ℝ),
    (a 1 + a 2 + a 3 = 9) ∧
    (a 2 * a 4 = 21) ∧
    (∀ n: ℕ, n > 0 → (∑ k in Finset.range n, (b k) / (a k) = 1 - 1 / (2 : ℝ)^n)) ∧
    (∀ n: ℕ, n > 0 → (S n = ∑ k in Finset.range n, b k)) ∧
    (∃ n > 0, S n > 2) ∧
    (∀ m > 0, S m > 2 → m ≥ 4)) :=
sorry

end min_n_for_Sn_gt_2_l292_292399


namespace polyhedron_faces_with_same_sides_l292_292097

theorem polyhedron_faces_with_same_sides (n : ℕ) (h : n > 0) :
  ∃ k, ∃ (faces_with_k_sides : ℕ), faces_with_k_sides ≥ n ∧ ∀ (P : polyhedron), convex P ∧ P.faces = 10 * n → count_faces_with_sides P k = faces_with_k_sides :=
by
  sorry

end polyhedron_faces_with_same_sides_l292_292097


namespace conjugate_of_z_l292_292646

-- Define the complex number z
def z : ℂ := -complex.I * (2 + complex.I)

-- Define the conjugate of z
def conj_z : ℂ := complex.conj z

-- State the theorem
theorem conjugate_of_z : conj_z = 1 + 2 * complex.I := by
  sorry

end conjugate_of_z_l292_292646


namespace sum_even_positive_integers_less_than_102_l292_292886

theorem sum_even_positive_integers_less_than_102 : 
  let a := 2
  let d := 2
  let l := 100
  let n := (l - a) / d + 1
  let sum := n / 2 * (a + l)
  (sum = 2550) :=
by
  let a := 2
  let d := 2
  let l := 100
  let n := (l - a) / d + 1
  let sum := n * (a + l) / 2
  show sum = 2550
  sorry

end sum_even_positive_integers_less_than_102_l292_292886


namespace first_machine_rate_l292_292948

theorem first_machine_rate (x : ℝ) (h : (x + 55) * 30 = 2400) : x = 25 :=
by
  sorry

end first_machine_rate_l292_292948


namespace cuboid_volume_ratio_l292_292199

theorem cuboid_volume_ratio 
  (width length height : ℕ) 
  (h_width : width = 28) 
  (h_length : length = 24) 
  (h_height : height = 19) :
  let V1 := width * length * height;
  let new_width := 2 * width;
  let new_length := 2 * length;
  let new_height := 2 * height;
  let V2 := new_width * new_length * new_height 
in V2 = 8 * V1 :=
by
  sorry

end cuboid_volume_ratio_l292_292199


namespace general_term_formula_l292_292746

noncomputable def sequence (n : ℕ+) : ℚ :=
  match n with
  | 1 => 1 / 2
  | n+1 =>
    let an := sequence n
    in ((n : ℚ) * an) / ((n+1 : ℚ) * ((n : ℚ) * an + 2))

theorem general_term_formula (n : ℕ+) : sequence n = 1 / (n * (3 * 2^(n - 1) - 1)) :=
sorry

end general_term_formula_l292_292746


namespace units_digit_of_sum_of_cubes_l292_292898

theorem units_digit_of_sum_of_cubes : 
  (24^3 + 42^3) % 10 = 2 := by
sorry

end units_digit_of_sum_of_cubes_l292_292898


namespace problem1_problem2_l292_292680

section
noncomputable def f (a x : ℝ) := a * Real.exp x - x^2
noncomputable def g (a x : ℝ) := f a x + x^2 - x

theorem problem1 (x : ℝ) (h : x > Real.exp 1) : f 1 x > 0 := 
sorry

theorem problem2 (a : ℝ) : (∀ x1 x2 : ℝ, x1 ≠ x2 → g a x1 = 0 → g a x2 = 0) → 0 < a ∧ a < 1 / Real.exp 1 := 
sorry
end

end problem1_problem2_l292_292680


namespace probability_of_three_positive_answers_l292_292411

noncomputable def probability_exactly_three_positive_answers : ℚ :=
  (7.choose 3) * (3/7)^3 * (4/7)^4

theorem probability_of_three_positive_answers :
  probability_exactly_three_positive_answers = 242520 / 823543 :=
by
  unfold probability_exactly_three_positive_answers
  sorry

end probability_of_three_positive_answers_l292_292411


namespace final_numbers_l292_292020

-- Define the initial sequence of squares of the first 2022 natural numbers.
def initial_sequence : ℕ → ℕ := λ n, (n + 1) ^ 2

-- Define the arithmetic mean operation on the sequence for a given index.
def arithmetic_mean (seq : ℕ → ℕ) (i : ℕ) : ℕ :=
  (seq (i - 1) + seq (i + 1)) / 2

-- Define the sequence transformation for each step until only two numbers remain.
noncomputable def transform_sequence (seq : ℕ → ℕ) (steps : ℕ) : List ℕ :=
  sorry -- transformation logic here

-- Given sequence after 1010 operations.
noncomputable def final_sequence : List ℕ :=
  transform_sequence initial_sequence 1010

-- The statement to be proven
theorem final_numbers :
  final_sequence = [1023131, 1025154] :=
sorry

end final_numbers_l292_292020


namespace max_lines_dividing_ngon_l292_292648

theorem max_lines_dividing_ngon (n : ℕ) (ngon : convex_polygon) 
  (non_parallel_sides : ∀ (i j : ℕ) (hi : i < n) (hj : j < n), i ≠ j → ¬is_parallel (ngon.side i) (ngon.side j))
  (O : Point) (inside_ngon : ngon.contains O) :
  ∀ (l1 l2 : Line), (passes_through l1 O) ∧ (passes_through l2 O) → 
  divides_area_in_half l1 ngon → divides_area_in_half l2 ngon → 
  count_lines_dividing_half O ngon ≤ n := 
sorry

end max_lines_dividing_ngon_l292_292648


namespace no_points_C_for_conditions_l292_292015

theorem no_points_C_for_conditions : 
  ∀ (A B C : ℝ × ℝ), 
  dist A B = 12 → (
  let P := ∑ (dist A C) (dist B C) in P = 60 → 
  let A := 240 in ∃ 0, ¬ (Σ area_triangle ABC)) := (
  sorry
  )  

end no_points_C_for_conditions_l292_292015


namespace part1_f_f5_eq_5_part2_range_a_part3_range_t_l292_292307

-- Part (1)
def f_a (a : ℝ) (x : ℝ) : ℝ :=
if x > 0 then -x^2 + 2*a*x - a^2 + 2*a else x^2 + 2*a*x + a^2 - 2*a

theorem part1_f_f5_eq_5 (a : ℝ) (h : a = 2) : f_a a (f_a a 5) = 5 := by
  sorry

-- Part (2)
theorem part2_range_a (a : ℝ) (h : (∃ x1 x2 x3 : ℝ, x1 < x2 ∧ x2 < x3 ∧ (f_a a x1 = 3) ∧ (f_a a x2 = 3) ∧ (f_a a x3 = 3))) : (3 / 2 < a ∧ a ≤ 3) := by
  sorry

-- Part (3)
theorem part3_range_t (a : ℝ) 
  (x1 x2 x3 : ℝ)
  (h1 : x1 < x2 ∧ x2 < x3)
  (h2 : f_a a x1 = 3 ∧ f_a a x2 = 3 ∧ f_a a x3 = 3)
  (h3 : ∀ t : ℝ, (x1 / (x2 + x3)) < t) : t > -1 := by
  sorry

end part1_f_f5_eq_5_part2_range_a_part3_range_t_l292_292307


namespace wheel_travel_distance_l292_292558

theorem wheel_travel_distance (r : ℝ) (h_r : r = 2) : 
  ∃ D : ℝ, D = 3 * (2 * π * r) ∧ D = 12 * π :=
by 
  use 12 * π
  split
  · calc
      3 * (2 * π * r) = 3 * (4 * π) : by rw [h_r, mul_assoc]
      _ = 12 * π : by norm_num
    
  · rfl

end wheel_travel_distance_l292_292558


namespace distance_travelled_downstream_l292_292852

noncomputable def boat_speed_still_water : ℝ := 20 -- km/hr
noncomputable def current_rate : ℝ := 3 -- km/hr
noncomputable def travel_time_minutes : ℝ := 24 -- minutes
noncomputable def travel_time_hours : ℝ := travel_time_minutes / 60 -- convert to hours

theorem distance_travelled_downstream : 
  let effective_speed := boat_speed_still_water + current_rate in
  let travel_time := travel_time_hours in
  let distance := effective_speed * travel_time in
  distance = 9.2 :=
by 
  -- The proof will go here
  unfold effective_speed travel_time distance
  sorry

end distance_travelled_downstream_l292_292852


namespace transformed_solution_set_l292_292101

theorem transformed_solution_set (k a b c : ℝ) :
  (∀ x : ℝ, ( (k / (x + a) + (x + b) / (x + c)) < 0) → x ∈ (Ioo (-2) (-1) ∪ Ioo 2 3)) →
  (∀ x : ℝ, ( (k * x / (a * x - 1) + (b * x - 1) / (c * x - 1)) < 0) → x ∈ (Ioo (-1 / 2) (-1 / 3) ∪ Ioo (1 / 2) 1)) :=
by
  sorry

end transformed_solution_set_l292_292101


namespace sufficient_not_necessary_range_l292_292661

variable (m x : ℝ)

def condition_p : Prop := abs(x - 4) ≤ 6
def condition_q : Prop := x ≤ 1 + m

theorem sufficient_not_necessary_range : (∀ x, condition_p x → condition_q x) ∧ ( ∃ x, ¬ condition_q x → condition_p x ) → m ≥ 9 := sorry

end sufficient_not_necessary_range_l292_292661


namespace chessboard_diagonal_pairs_l292_292761

theorem chessboard_diagonal_pairs (n : ℕ) (h1 : odd n) (h2 : n > 1) :
  (∃ pairs : set (ℕ × ℕ) → set (ℕ × ℕ) → Prop,
    ∀ s ∈ pairs, ∃ i j (ij_diagonal : ∀ s1 s2 ∈ pairs, s1 ≠ s2 → (s1.1, s1.2) ≠ (s2.1, s2.2) → abs (s1.1 - s2.1) = 1 ∧ abs (s1.2 - s2.2) = 1)), 
  n = 3 :=
by sorry

end chessboard_diagonal_pairs_l292_292761


namespace solve_congruence_l292_292108

theorem solve_congruence (m : ℤ) : 13 * m ≡ 9 [MOD 47] → m ≡ 29 [MOD 47] :=
by
  sorry

end solve_congruence_l292_292108


namespace racing_cars_lcm_l292_292806

theorem racing_cars_lcm :
  let a := 28
  let b := 24
  let c := 32
  Nat.lcm a (Nat.lcm b c) = 672 :=
by
  sorry

end racing_cars_lcm_l292_292806


namespace vector_a_equals_given_conditions_l292_292004

open Real

variables (a : ℝ × ℝ) (b : ℝ × ℝ)

theorem vector_a_equals_given_conditions :
  (b = (1, -2)) →
  (∃θ : ℝ, θ = π ∧ |a| = 3 * sqrt 5) →
  a = (-3, 6) :=
sorry

end vector_a_equals_given_conditions_l292_292004


namespace zach_needs_more_money_l292_292917

theorem zach_needs_more_money : 
  let bike_cost := 150
  let discount := 0.10
  let weekly_allowance := 5
  let lawn_min := 8
  let lawn_max := 12
  let garage_cleaning := 15
  let babysitting_rate := 7
  let babysitting_hours := 3
  let loan := 10
  let savings := 65
  let discounted_bike_cost := bike_cost * (1 - discount)
  let remaining_after_savings := discounted_bike_cost - savings
  let total_needed := remaining_after_savings + loan
  let max_earnings := weekly_allowance + lawn_max + garage_cleaning + (babysitting_rate * babysitting_hours)
  let more_needed := total_needed - max_earnings
  in more_needed = 27 :=
by 
  sorry

end zach_needs_more_money_l292_292917


namespace books_bought_l292_292440

theorem books_bought (initial_books bought_books total_books : ℕ) 
    (h_initial : initial_books = 35)
    (h_total : total_books = 56) :
    bought_books = total_books - initial_books → bought_books = 21 := 
by
  sorry

end books_bought_l292_292440


namespace total_shared_amount_l292_292533

noncomputable def A : ℝ := sorry
noncomputable def B : ℝ := sorry
noncomputable def C : ℝ := sorry

axiom h1 : A = 1 / 3 * (B + C)
axiom h2 : B = 2 / 7 * (A + C)
axiom h3 : A = B + 20

theorem total_shared_amount : A + B + C = 720 := by
  sorry

end total_shared_amount_l292_292533


namespace total_notebooks_eq_216_l292_292911

theorem total_notebooks_eq_216 (n : ℕ) 
  (h1 : total_notebooks = n^2 + 20)
  (h2 : total_notebooks = (n + 1)^2 - 9) : 
  total_notebooks = 216 := 
by 
  sorry

end total_notebooks_eq_216_l292_292911


namespace julio_twice_james_age_in_14_years_l292_292175

theorem julio_twice_james_age_in_14_years :
  ∃ x : ℕ, 36 + x = 2 * (11 + x) ∧ x = 14 :=
by
  use 14
  simp
  sorry

end julio_twice_james_age_in_14_years_l292_292175


namespace complement_intersection_l292_292072

universe u

noncomputable def U : Set ℝ := Set.univ

noncomputable def M : Set ℝ := {x : ℝ | x^2 > 1}

noncomputable def N : Set ℤ := {x : ℤ | abs x ≤ (2 : ℤ)}

theorem complement_intersection : (U \ M : Set ℝ) ∩ (N : Set ℝ) = ({-1, 0, 1} : Set ℝ) :=
by
  sorry

end complement_intersection_l292_292072


namespace weighted_mean_proof_l292_292996

def scores : List ℕ := [85, 90, 88, 92, 86, 94]
def weights : List ℕ := [1, 2, 1, 3, 1, 2]

def weighted_mean_is_correct : Prop :=
  let weighted_sum := List.sum $ List.zipWith (· * ·) scores weights
  let total_weights := List.sum weights
  (weighted_sum : ℚ) / total_weights = 90.3

theorem weighted_mean_proof : weighted_mean_is_correct :=
by
  sorry

end weighted_mean_proof_l292_292996


namespace remainder_when_N_div_1000_l292_292760

noncomputable def sum_digits_base (q n : ℕ) : ℕ :=
sorry

def fn (n x : ℕ) (p : ℕ) [prime p] : ℕ :=
(sum_digits_base p (n - x)) + (sum_digits_base p x) - (sum_digits_base p n)

def num_values_n (n : ℕ) : ℕ :=
(finset.range n).filter (λ x, 4 ∣ (∑ p in finset.primes, fn n x p)).card

theorem remainder_when_N_div_1000 :
  let n := 2^2015 - 1 in
  let N := num_values_n n in
  N % 1000 = 382 :=
sorry

end remainder_when_N_div_1000_l292_292760


namespace joseph_drives_more_l292_292046

-- Definitions for the problem
def v_j : ℝ := 50 -- Joseph's speed in mph
def t_j : ℝ := 2.5 -- Joseph's time in hours
def v_k : ℝ := 62 -- Kyle's speed in mph
def t_k : ℝ := 2 -- Kyle's time in hours

-- Prove that Joseph drives 1 more mile than Kyle
theorem joseph_drives_more : (v_j * t_j) - (v_k * t_k) = 1 := 
by 
  sorry

end joseph_drives_more_l292_292046


namespace find_value_of_x_l292_292380
-- Import the broader Mathlib to bring in the entirety of the necessary library

-- Definitions for the conditions
variables {x y z : ℝ}

-- Assume the given conditions
axiom h1 : x = y
axiom h2 : y = 2 * z
axiom h3 : x * y * z = 256

-- Statement to prove
theorem find_value_of_x : x = 8 :=
by {
  -- Proof goes here
  sorry
}

end find_value_of_x_l292_292380


namespace combined_shape_perimeter_l292_292958

noncomputable def π := Real.pi

structure GeometricShape where
  square_side : ℝ
  semicircle_diameter : ℝ

def perimeter (shape : GeometricShape) : ℝ :=
  let square_perimeter := 4 * shape.square_side
  let semicircle_perimeter := (π * shape.semicircle_diameter) / 2 + shape.semicircle_diameter
  square_perimeter + semicircle_perimeter

theorem combined_shape_perimeter
  (shape : GeometricShape)
  (h1 : shape.square_side = 8)
  (h2 : shape.semicircle_diameter = 8)
  : abs (perimeter shape - 52.57) < 0.01 :=
by
  simp [perimeter, h1, h2, π]
  norm_num
  sorry

end combined_shape_perimeter_l292_292958


namespace sum_1998_eq_3985_no_n_sum_eq_2001_l292_292206

-- Define the sequence based on the given conditions.
def seq : ℕ → ℕ
| 0     := 1
| (n+1) := if ∃ k, seq n = 1 ∧ seq.count 2 (0..n) = 2^(k-1) then 2 else 1

-- Define the sum of the first n terms of the sequence.
def S (n : ℕ) := (list.range (n+1)).sum (λ i, seq i)

-- The first question: Prove that the sum of the first 1998 terms of the sequence is 3985.
theorem sum_1998_eq_3985 : S 1998 = 3985 :=
by sorry

-- The second question: Prove that there does not exist a positive integer n such that the sum 
-- of the first n terms of the sequence is 2001.
theorem no_n_sum_eq_2001 : ¬ ∃ n : ℕ+, S n = 2001 :=
by sorry

end sum_1998_eq_3985_no_n_sum_eq_2001_l292_292206


namespace total_cost_correct_l292_292079

noncomputable def camera_old_cost : ℝ := 4000
noncomputable def camera_new_cost := camera_old_cost * 1.30
noncomputable def lens_cost := 400
noncomputable def lens_discount := 200
noncomputable def lens_discounted_price := lens_cost - lens_discount
noncomputable def total_cost := camera_new_cost + lens_discounted_price

theorem total_cost_correct :
  total_cost = 5400 := by
  sorry

end total_cost_correct_l292_292079


namespace floor_sum_example_l292_292274

theorem floor_sum_example : ⌊18.7⌋ + ⌊-18.7⌋ = -1 := by
  sorry

end floor_sum_example_l292_292274


namespace ineq_a3b3c3_l292_292816

theorem ineq_a3b3c3 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^3 + b^3 + c^3 ≥ a^2 * b + b^2 * c + c^2 * a ∧ (a^3 + b^3 + c^3 = a^2 * b + b^2 * c + c^2 * a ↔ a = b ∧ b = c) :=
by
  sorry

end ineq_a3b3c3_l292_292816


namespace find_k_and_MN_length_l292_292654

open Real

noncomputable def line_through (A : Point ℝ) (k : ℝ) (x : ℝ) : ℝ := k * (x - A.x) + A.y

def circle (C : Point ℝ) (r : ℝ) (x y : ℝ) : Prop := (x - C.x)^2 + (y - C.y)^2 = r^2

def intersects_circle (l : ℝ → ℝ) (C : Point ℝ) (r : ℝ) : Prop :=
  ∃ x1 x2 y1 y2, y1 = l x1 ∧ y2 = l x2 ∧ circle C r x1 y1 ∧ circle C r x2 y2

theorem find_k_and_MN_length 
  (A : Point ℝ) (k : ℝ) (C : Point ℝ) (r : ℝ)
  (H1 : A = (1, 0))
  (H2 : C = (2, 3))
  (H3 : r = 1)
  (H4 : intersects_circle (line_through A k) C r) :
  k > 4/3 ∧ ∃ M N : Point ℝ, 
    (some (intersects_circle (line_through A k) C r)) = (M, N) ∧
    ((M.x, M.y).1 * (N.x, N.y).1 + (M.x, M.y).2 * (N.x, N.y).2 = 12) →
    dist M N = 2 :=
by sorry

end find_k_and_MN_length_l292_292654


namespace infinite_series_equals_l292_292597

noncomputable def infinite_series : Real :=
  ∑' n, if h : (n : ℕ) ≥ 2 then (n^4 + 2 * n^3 + 8 * n^2 + 8 * n + 8) / (2^n * (n^4 + 4)) else 0

theorem infinite_series_equals : infinite_series = 11 / 10 :=
  sorry

end infinite_series_equals_l292_292597


namespace students_activity_details_l292_292395

open Finset

-- Definitions for sets and their cardinalities
variable (S G P : Finset ℕ)
variable (total_students spanish_students german_students sports_students : ℕ)
variable (intersection_SP intersection_SG intersection_GP intersection_SGP : ℕ)

def only_one_activity_students : ℕ :=
  (spanish_students - intersection_SG - intersection_SP + intersection_SGP) +
  (german_students - intersection_SG - intersection_GP + intersection_SGP) +
  (sports_students - intersection_SP - intersection_GP + intersection_SGP)

def no_activity_students : ℕ :=
  total_students - (spanish_students + german_students + sports_students -
                     intersection_SP - intersection_SG - intersection_GP + intersection_SGP)

theorem students_activity_details
  (h1 : total_students = 94)
  (h2 : spanish_students = 40)
  (h3 : german_students = 27)
  (h4 : sports_students = 60)
  (h5 : intersection_SP = 24)
  (h6 : intersection_SG = 10)
  (h7 : intersection_GP = 12)
  (h8 : intersection_SGP = 4) :
  only_one_activity_students total_students spanish_students german_students sports_students intersection_SP intersection_SG intersection_GP intersection_SGP = 47 ∧
  no_activity_students total_students spanish_students german_students sports_students intersection_SP intersection_SG intersection_GP intersection_SGP = 9 := by
  sorry

end students_activity_details_l292_292395


namespace good_permutation_errors_l292_292378

theorem good_permutation_errors :
  let n := 4 -- Total number of letters
  let n2 := 2 -- Number of repetitions of letter 'o'
  let factorial (n : ℕ) := if n = 0 then 1 else List.prod (List.range (n+1).tail)
  (factorial n / (factorial n2)) - 1 = 11 := by
  sorry

end good_permutation_errors_l292_292378


namespace total_cost_l292_292080

-- Definitions based on conditions
def old_camera_cost : ℝ := 4000
def new_model_cost_increase_rate : ℝ := 0.3
def lens_initial_cost : ℝ := 400
def lens_discount : ℝ := 200

-- Main statement to prove
theorem total_cost (old_camera_cost new_model_cost_increase_rate lens_initial_cost lens_discount : ℝ) : 
  let new_camera_cost := old_camera_cost * (1 + new_model_cost_increase_rate)
  let lens_cost_after_discount := lens_initial_cost - lens_discount
  (new_camera_cost + lens_cost_after_discount) = 5400 :=
by
  sorry

end total_cost_l292_292080


namespace cos_value_l292_292727

-- Definitions
def angle_A : ℝ := 90
def tan_C : ℝ := 4
def cos_C := real.cos

-- Problem statement
theorem cos_value (A C : ℝ) (hA : A = 90) (hC : real.tan C = 4) :
  cos_C C = sqrt 17 / 17 :=
sorry

end cos_value_l292_292727


namespace fraction_to_percentage_l292_292008

theorem fraction_to_percentage (x : ℝ) (hx : 0 < x) : 
  (x / 50 + x / 25) = 0.06 * x := 
sorry

end fraction_to_percentage_l292_292008


namespace find_g_find_a_range_l292_292069

noncomputable def f (a x : ℝ) := log a (x - 3 * a)
noncomputable def g (a x : ℝ) := log a (1 / (x - a))

theorem find_g (a : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) (x : ℝ) :
    g a x = -log a (x - a) :=
by
  sorry

theorem find_a_range (a : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) :
    ∀ x ∈ Icc (a + 2) (a + 3), |f a x - g a x| ≤ 1 → 0 < a ∧ a ≤ (9 - real.sqrt 57) / 12 :=
by
  sorry

end find_g_find_a_range_l292_292069


namespace arrangement_with_A_head_arrangement_with_adjacent_A_B_arrangement_with_A_not_head_B_not_end_arrangement_with_A_B_taller_shorter_not_adjacent_l292_292862

-- Definition for the arrangement problem
noncomputable def numArrangementsWithAHead : Nat := 24
noncomputable def numArrangementsWithAdjacentAB : Nat := 48
noncomputable def numArrangementsWithANotHeadBNotEnd : Nat := 72
noncomputable def numArrangementsWithAFirstBSecondNotAdjacent : Nat := 18

theorem arrangement_with_A_head (A B C D E : Char) :
  ∃ l : List Char, (l.head = A) ∧ l ~ L := numArrangementsWithAHead := by sorry

theorem arrangement_with_adjacent_A_B (A B C D E : Char) :
  ∃ l : List Char, (adjacent A B l) ∧ l ~ L := numArrangementsWithAdjacentAB := by sorry

theorem arrangement_with_A_not_head_B_not_end (A B C D E : Char) :
  ∃ l : List Char, (¬(l.head = A) ∧ ¬(l.last = B)) ∧ l ~ L := numArrangementsWithANotHeadBNotEnd := by sorry

theorem arrangement_with_A_B_taller_shorter_not_adjacent (A B C D E : Char) :
  ∃ l : List Char, (taller_shorter_not_adjacent A B l) ∧ l ~ L := numArrangementsWithAFirstBSecondNotAdjacent := by sorry

end arrangement_with_A_head_arrangement_with_adjacent_A_B_arrangement_with_A_not_head_B_not_end_arrangement_with_A_B_taller_shorter_not_adjacent_l292_292862


namespace construct_P4_l292_292979

-- Let's define the problem in terms of Lean 4
theorem construct_P4 (g : Line) (g' g'' gIV : Line) (P1 P4 : Plane) (beta1 beta2 : ℝ) :
  -- Conditions given in the problem:
  beta1 = beta2 →
  g ∥ P1 →
  g' ∥ g'' →
  ∠ (g, P4) = 45 ° →
  -- Goal to prove:
  (∠ (g, P1) = ∠ (g, P4)) :=
begin
  sorry
end

end construct_P4_l292_292979


namespace phenolphthalein_probability_l292_292812

theorem phenolphthalein_probability :
  let total_bottles := 5
  let alkaline_bottles := 2
  total_bottles > 0 ->
  alkaline_bottles >= 0 ->
  alkaline_bottles <= total_bottles ->
  (alkaline_bottles / total_bottles : ℚ) = (2 / 5 : ℚ) :=
by {
  let total_bottles := 5
  let alkaline_bottles := 2
  intros _ _ _
  have : (alkaline_bottles / total_bottles : ℚ) = (2 / 5 : ℚ) := by norm_num
  assumption
  sorry
}

end phenolphthalein_probability_l292_292812


namespace find_cos_alpha_l292_292668

open Real
open Complex

-- Definitions of the conditions
variables {α : ℝ} -- α is an angle in the second quadrant
def is_second_quadrant_angle (α : ℝ) : Prop := π / 2 < α ∧ α < π
def equation (α : ℝ) : Prop := 2 * sin (2 * α) = cos (2 * α) - 1

-- The statement we want to prove
theorem find_cos_alpha (α : ℝ) (h1 : is_second_quadrant_angle α) (h2 : equation α) : 
  cos α = - sqrt 5 / 5 :=
by
  sorry

end find_cos_alpha_l292_292668


namespace intersection_lies_on_midline_l292_292820

noncomputable def midline_intersection (A B C : Point) (r r_a r_b r_c : ℝ)
  (A1 : Point) (B1 : Point) (C1 : Point) (F : Point) 
  (Fa : Point) (Fb : Point) (Fc : Point) : Prop :=
  let C1Fc := dist C1 Fc
  let FA1 := dist F A1
  let FaFb := dist Fa Fb
  let FcF := dist Fc F
  let A1Fa := dist A1 Fa
  let FbC1 := dist Fb C1
  in (C1Fc * FA1 * FaFb = FcF * A1Fa * FbC1)

-- Now we state the theorem we want to prove.
theorem intersection_lies_on_midline (A B C : Point) (r r_a r_b r_c : ℝ)
  (A1 : Point) (B1 : Point) (C1 : Point) (F : Point) 
  (Fa : Point) (Fb : Point) (Fc : Point) 
  (hA1 : midpoint A B C A1) (hB1 : midpoint B C A B1) 
  (hC1 : midpoint C A B C1) (hF : nine_point_circle_tangent F)
  (hFa : excircle_tangent Fa r_a) (hFb : excircle_tangent Fb r_b) 
  (hFc : excircle_tangent Fc r_c) :
  midline_intersection A B C r r_a r_b r_c A1 B1 C1 F Fa Fb Fc :=
sorry

end intersection_lies_on_midline_l292_292820


namespace probability_phenolphthalein_red_l292_292814

-- Define the given conditions
def is_acidic (solution : ℕ) : Prop := solution = 2 ∨ solution = 4
def is_alkaline (solution : ℕ) : Prop := solution = 3 ∨ solution = 5
def is_neutral (solution : ℕ) : Prop := solution = 1

-- Define the problem
def phenolphthalein_turns_red (solution : ℕ) : Prop := is_alkaline solution

-- The given problem with the correct answer translated into a Lean statement
theorem probability_phenolphthalein_red : 
  (∑ solution in Finset.range 6, if phenolphthalein_turns_red solution then 1 else 0).val / 5 = 2 / 5 := 
by
  sorry

end probability_phenolphthalein_red_l292_292814


namespace maria_travel_fraction_l292_292611

theorem maria_travel_fraction (x : ℝ) (total_distance : ℝ)
  (h1 : ∀ d1 d2, d1 + d2 = total_distance)
  (h2 : total_distance = 360)
  (h3 : ∃ d1 d2 d3, d1 = 360 * x ∧ d2 = (1 / 4) * (360 - 360 * x) ∧ d3 = 135)
  (h4 : d1 + d2 + d3 = total_distance)
  : x = 1 / 2 :=
by
  sorry

end maria_travel_fraction_l292_292611


namespace square_side_increase_factor_l292_292126

theorem square_side_increase_factor (s k : ℕ) (x new_x : ℕ) (h1 : x = 4 * s) (h2 : new_x = 4 * x) (h3 : new_x = 4 * (k * s)) : k = 4 :=
by
  sorry

end square_side_increase_factor_l292_292126


namespace max_price_per_shirt_l292_292981

theorem max_price_per_shirt :
  let total_budget := 200
  let entrance_fee := 5
  let num_shirts := 20
  let sales_tax := 0.06
  let effective_budget := (total_budget - entrance_fee) / (1 + sales_tax)
  floor (effective_budget / num_shirts) = 9 := by
  sorry

end max_price_per_shirt_l292_292981


namespace length_of_unfenced_side_l292_292635

theorem length_of_unfenced_side
  (L W : ℝ)
  (h1 : L * W = 200)
  (h2 : 2 * W + L = 50) :
  L = 10 :=
sorry

end length_of_unfenced_side_l292_292635


namespace jenny_distance_from_school_l292_292752

-- Definitions based on the given conditions.
def kernels_per_feet : ℕ := 1
def feet_per_kernel : ℕ := 25
def squirrel_fraction_eaten : ℚ := 1/4
def remaining_kernels : ℕ := 150

-- Problem statement in Lean 4.
theorem jenny_distance_from_school : 
  ∀ (P : ℕ), (3/4:ℚ) * P = 150 → P * feet_per_kernel = 5000 :=
by
  intros P h
  sorry

end jenny_distance_from_school_l292_292752


namespace probability_two_forks_two_spoons_l292_292386

theorem probability_two_forks_two_spoons :
  let total_silverware := 24
  let total_ways_to_choose_4 := (24.choose 4)
  let ways_to_choose_2_forks := (8.choose 2)
  let ways_to_choose_2_spoons := (10.choose 2)
  let favorable_ways := ways_to_choose_2_forks * ways_to_choose_2_spoons
  favorable_ways / total_ways_to_choose_4 = (18 / 91) := by
  sorry

end probability_two_forks_two_spoons_l292_292386


namespace dragon_rope_touching_tower_l292_292193

noncomputable def rope_length_touching_tower : ℝ :=
  let OA := 25
  let OB := 10
  let tangent_length := (λ (x y : ℝ), real.sqrt (x^2 - y^2)) 25 10
  let theta := real.acos (2 / 5)
  OB * theta

theorem dragon_rope_touching_tower :
  (∃ (a b c : ℕ), c = 2 ∧ rope_length_touching_tower = (a - real.sqrt b) / c ∧ a + b + c = 352) :=
begin
  sorry
end

end dragon_rope_touching_tower_l292_292193


namespace sequence_a3_l292_292403

theorem sequence_a3 (a : ℕ → ℚ) 
  (h₁ : a 1 = 1)
  (recursion : ∀ n, a (n + 1) = a n / (1 + a n)) : 
  a 3 = 1 / 3 :=
by 
  sorry

end sequence_a3_l292_292403


namespace negation_propositional_logic_l292_292127

theorem negation_propositional_logic :
  ¬ (∀ x : ℝ, x^2 + x + 1 < 0) ↔ ∃ x : ℝ, x^2 + x + 1 ≥ 0 :=
by sorry

end negation_propositional_logic_l292_292127


namespace probability_john_david_chosen_l292_292174

theorem probability_john_david_chosen (total_workers : ℕ) (choose : ℕ) : 
  total_workers = 6 ∧ choose = 2 → 
  probability (John_and_David_chosen total_workers choose) = 1 / 15 :=
by
  sorry

end probability_john_david_chosen_l292_292174


namespace d_is_not_axiom_l292_292985

-- Define the given propositions as conditions
def PropositionA := ∀ (l1 l2 l3 : Type), (l1 ∥ l2) → (l2 ∥ l3) → (l1 ∥ l3)
def PropositionB := ∀ (p1 p2 : Type) (l : Type), (p1 ∈ l) → (p2 ∈ l) → (p1 ≠ p2) → (∃ (pl : Type), l ⊂ pl)
def PropositionC := ∀ (pl1 pl2 : Type) (p : Type), (pl1 ≠ pl2) → (p ∈ pl1) → (p ∈ pl2) → (∃! (l : Type), p ∈ l ∧ l ⊂ pl1 ∧ l ⊂ pl2)
def PropositionD := ∀ (A B A' B' : Type), (A ∥ A') → (B ∥ B') → (∠AOB = ∠A'OB' + 180 ∘ ∨ ∠AOB = ∠A'OB')

-- We're asked to prove that Proposition D is not an axiom.
theorem d_is_not_axiom : ¬ ProposionD := by
  sorry

end d_is_not_axiom_l292_292985


namespace value_of_f2_l292_292343

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x ^ 3 + b * x + 8

theorem value_of_f2 (a b : ℝ) (h : f a b (-2) = 10) : f a b 2 = 6 := by
  have g : ℝ → ℝ := λ x => a * x ^ 3 + b * x
  have odd_g : ∀ x, g (-x) = -g (x) := by
    intros x
    calc
      g (-x) = a * (-x) ^ 3 + b * (-x) : by sorry
          ... = -a * x ^ 3 + -b * x    : by sorry
          ... = - (a * x ^ 3 + b * x)  : by sorry
          ... = - g x                  : by sorry
  have g_neg2 : g (-2) + 8 = f a b (-2) := by rfl
  rw [h, g_neg2] at h
  have g_neg2_value : g (-2) = 2 := by linarith
  have g_2_value : g 2 = - g (-2) := by exact (congr_fun odd_g 2).symm
  rw [g_neg2_value] at g_2_value
  have g_2_value_final : g 2 = -2 := by linarith
  calc
    f a b 2 = g 2 + 8 : by rfl
        ... = -2 + 8  : by rw [g_2_value_final]
        ... = 6       : by linarith

end value_of_f2_l292_292343


namespace remaining_height_after_three_fourths_total_burn_time_l292_292556

def burn_time (k : ℕ) : ℕ := 1510 - 10 * k

def total_burn_time : ℕ := ∑ k in finset.range 151, burn_time k

theorem remaining_height_after_three_fourths_total_burn_time :
  let T := total_burn_time in
  let three_fourths_T := 3 * T / 4 in
  let m := finset.range 150.filter (λ k, ∑ i in finset.range k, burn_time i ≤ three_fourths_T).sup' (finset.nonempty_of_mem (finset.mem_range.mpr (nat.lt_succ_self 149))) in
  150 - m = 38 :=
by sorry

end remaining_height_after_three_fourths_total_burn_time_l292_292556


namespace cos_C_in_right_triangle_l292_292719

theorem cos_C_in_right_triangle
  (A B C : Type)
  [has_angle A]
  [has_angle B]
  [has_angle C]
  (hA : has_angle.angle A = 90)
  (hT : has_tangent.tangent C = 4) :
  has_cosine.cosine C = sqrt 17 / 17 := 
by sorry

end cos_C_in_right_triangle_l292_292719


namespace number_drawn_from_3rd_group_l292_292968

theorem number_drawn_from_3rd_group {n k : ℕ} (pop_size : ℕ) (sample_size : ℕ) 
  (drawn_from_group : ℕ → ℕ) (group_id : ℕ) (num_in_13th_group : ℕ) : 
  pop_size = 160 → 
  sample_size = 20 → 
  (∀ i, 1 ≤ i ∧ i ≤ sample_size → ∃ j, group_id = i ∧ 
    (j = (i - 1) * (pop_size / sample_size) + drawn_from_group 1)) → 
  num_in_13th_group = 101 → 
  drawn_from_group 3 = 21 := 
by
  intros hp hs hg h13
  sorry

end number_drawn_from_3rd_group_l292_292968


namespace nonoverlapping_unit_squares_in_figure_50_l292_292390

def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

theorem nonoverlapping_unit_squares_in_figure_50 :
  f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37 → f 50 = 7651 :=
by
  sorry

end nonoverlapping_unit_squares_in_figure_50_l292_292390


namespace unit_digit_smaller_by_four_l292_292214

theorem unit_digit_smaller_by_four (x : ℤ) : x^2 + (x + 4)^2 = 10 * (x + 4) + x - 4 :=
by
  sorry

end unit_digit_smaller_by_four_l292_292214


namespace max_lines_dividing_ngon_l292_292649

theorem max_lines_dividing_ngon (n : ℕ) (ngon : convex_polygon) 
  (non_parallel_sides : ∀ (i j : ℕ) (hi : i < n) (hj : j < n), i ≠ j → ¬is_parallel (ngon.side i) (ngon.side j))
  (O : Point) (inside_ngon : ngon.contains O) :
  ∀ (l1 l2 : Line), (passes_through l1 O) ∧ (passes_through l2 O) → 
  divides_area_in_half l1 ngon → divides_area_in_half l2 ngon → 
  count_lines_dividing_half O ngon ≤ n := 
sorry

end max_lines_dividing_ngon_l292_292649


namespace production_volume_increase_l292_292945

theorem production_volume_increase (x : ℝ) :
    200 + 200 * (1 + x) + 200 * (1 + x)^2 = 1400 :=
sorry

end production_volume_increase_l292_292945


namespace base_subtraction_354_6_231_5_l292_292998

theorem base_subtraction_354_6_231_5 :
  (354₆:ℕ) - (231₅:ℕ) = 76 := by
  sorry

end base_subtraction_354_6_231_5_l292_292998


namespace regular_hexagon_collinear_l292_292228

theorem regular_hexagon_collinear (r : ℝ) : 
  (M divides AC in the ratio r) ∧ 
  (N divides CE in the ratio r) ∧ 
  (collinear B M N) -> 
  r = real.sqrt 3 / 3 :=
by 
  intro h
  sorry

end regular_hexagon_collinear_l292_292228


namespace max_parts_pretzel_l292_292459

-- Define the structure and properties of the Viennese pretzel
structure Pretzel :=
  (loops: ℕ)  -- Number of loops
  (intersections: ℕ)  -- Number of intersection points

-- Mathematically equivalent proof problem statement
theorem max_parts_pretzel (P: Pretzel) (h: P.intersections = 9) : 
  (P.intersections + 1) = 10 :=
by
  rw h
  exact Nat.add_one 9

end max_parts_pretzel_l292_292459


namespace max_longitudinal_moves_correct_min_longitudinal_moves_correct_l292_292927

noncomputable def max_longitudinal_moves (N : ℕ) := 2 * N * N

noncomputable def min_longitudinal_moves (N : ℕ) : ℕ :=
  if N = 1 then 1 else 2

theorem max_longitudinal_moves_correct (N : ℕ) (hN : N ≥ 1) :
  ∃ M, M = max_longitudinal_moves N :=
begin
  use 2 * N * N,
  sorry
end

theorem min_longitudinal_moves_correct (N : ℕ) (hN : N ≥ 1) :
  ∃ m, m = min_longitudinal_moves N :=
begin
  cases N,
  { exfalso, exact Nat.not_lt_zero 1 hN, }, -- Handles case N = 0 (impossible due to hN)
  { use if N = 0 then 1 else 2,
    sorry }
end

end max_longitudinal_moves_correct_min_longitudinal_moves_correct_l292_292927


namespace min_value_of_ratio_l292_292260

def min_ratio_of_products : ℕ :=
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  ∃ (g1 g2 : Finset ℕ), g1 ∪ g2 = S ∧ g1 ∩ g2 = ∅ ∧ 
  let p1 := g1.prod id in
  let p2 := g2.prod id in
  p1 % p2 = 0 ∧ p1 / p2 = 7

theorem min_value_of_ratio : min_ratio_of_products :=
by sorry

end min_value_of_ratio_l292_292260


namespace identify_fraction_l292_292983

def is_fraction_with_variable_denom (e : ℚ → ℚ) : Prop :=
∃ x, ∃ denom, (denom /= 0) ∧ (denom × e x \ne 0 ∧ denom ≠ 1)

theorem identify_fraction :
  let A := (fun x => 3 * x + 1 / 2)
  let B := (fun m n => -(m + n) / 3)
  let C := (fun x => 3 / (x + 3))
  let D := (fun x => x-1) in
is_fraction_with_variable_denom C := sorry

end identify_fraction_l292_292983


namespace calculate_gg_neg3_l292_292604

def g (x : ℚ) : ℚ :=
  x⁻² + (x⁻² / (1 - x⁻²))

theorem calculate_gg_neg3 :
  g (g (-3)) = 23887824 / 1416805 :=
by
  sorry

end calculate_gg_neg3_l292_292604


namespace triangle_area_l292_292148

variable (A B C O : Type) [RealNumber A] [RealNumber B] [RealNumber C] [RealNumber O]
variable (AB AC : Real) 
variable (r : Real) (π : Real) [Inhabited Real]

-- Triangle ABC is isosceles with AB = AC
def is_isosceles (AB AC : Real) := AB = AC

-- O is the center of the inscribed circle, area of the circle is 9π
def inscribed_circle_area (r : Real) (π : Real) : Prop := π * r^2 = 9 * π

-- To prove that the area of triangle ABC is 36 square centimeters
theorem triangle_area :
  is_isosceles AB AC →
  inscribed_circle_area r π → 
  1/2 * (6 * 2 * r * (r + r)) = 36 :=
by 
  intros h_AB_AC h_inscribed_circle_area
  -- The fundamental steps and transformation in the problem would be filled in here
  sorry

end triangle_area_l292_292148


namespace b_payment_l292_292922

theorem b_payment (b_days : ℕ) (a_days : ℕ) (total_wages : ℕ) (b_payment : ℕ) :
  b_days = 10 →
  a_days = 15 →
  total_wages = 5000 →
  b_payment = 3000 :=
by
  intros h1 h2 h3
  -- conditions
  have hb := h1
  have ha := h2
  have ht := h3
  -- skipping proof
  sorry

end b_payment_l292_292922


namespace sqrt_difference_inequality_l292_292245

noncomputable def sqrt10 := Real.sqrt 10
noncomputable def sqrt6 := Real.sqrt 6
noncomputable def sqrt7 := Real.sqrt 7
noncomputable def sqrt3 := Real.sqrt 3

theorem sqrt_difference_inequality : sqrt10 - sqrt6 < sqrt7 - sqrt3 :=
by 
  sorry

end sqrt_difference_inequality_l292_292245


namespace repaved_before_today_l292_292537

/--
A construction company is repaving a damaged road. 
So far, they have repaved a total of 4938 inches of the road. 
Today, they repaved 805 inches of the road. 
Prove that before today, they had repaved 4133 inches of the road.
-/
theorem repaved_before_today (total_inch : ℕ) (today_inch : ℕ) (repaved_before_today : ℕ) 
    (h1 : total_inch = 4938) 
    (h2 : today_inch = 805) 
    (h3 : repaved_before_today = total_inch - today_inch) :
    repaved_before_today = 4133 :=
begin
  sorry
end

end repaved_before_today_l292_292537


namespace complex_multiplication_l292_292590

theorem complex_multiplication (a b c d : ℤ) (i : ℂ) (hi : i^2 = -1) : 
  ((3 : ℂ) - 4 * i) * ((-7 : ℂ) + 6 * i) = (3 : ℂ) + 46 * i := 
  by
    sorry

end complex_multiplication_l292_292590


namespace smallest_k_for_unique_regular_polygon_l292_292550

theorem smallest_k_for_unique_regular_polygon (n : ℕ) (n_gt2 : 2 < n) :
  ∃ k, k = 5 ∧ ∀ (P Q : Set Point) (P_reg : is_regular_polygon P n) (Q_reg : ∃ m, is_regular_polygon Q m)
    (pts : Finset Point) (pts_card : pts.card = k) 
    (pts_on_P : ∀ p ∈ pts, p ∈ P)
    (pts_on_Q : ∀ p ∈ pts, p ∈ Q),
  P = Q :=
sorry

end smallest_k_for_unique_regular_polygon_l292_292550


namespace functional_relationship_profit_price_l292_292030

-- Definitions based on problem conditions
def cost_price : ℝ := 10
def profit (x y : ℕ) : ℝ := (x - cost_price) * y

-- Given data points
def point1 : (ℝ × ℕ) := (20, 200)
def point2 : (ℝ × ℕ) := (25, 150)
def point3 : (ℝ × ℕ) := (30, 100)

-- Functional relationship derived from part 1
def y (x : ℝ) : ℝ :=
  -10 * x + 400

theorem functional_relationship :
  ∀ (x : ℝ), y 20 = 200 ∧ y 25 = 150 ∧ y 30 = 100 :=
by
  intro x
  have h1 : y 20 = 200 := by simp [y]; norm_num
  have h2 : y 25 = 150 := by simp [y]; norm_num
  have h3 : y 30 = 100 := by simp [y]; norm_num
  exact ⟨h1, h2, h3⟩

theorem profit_price :
  ∃ x : ℝ, profit x (y x).nat_abs = 2160 ∧ ∀ y : ℝ, y > x := 
by
  exists 22
  have h_profit : profit 22 (y 22).nat_abs = 2160 := by
    simp [profit, y]
    norm_num
  have h_minimize : ∀ y : ℝ, y > 22 := by
    assume y
    -- you may use other simple logical steps to verify y > 22
    sorry
  exact ⟨h_profit, h_minimize⟩


end functional_relationship_profit_price_l292_292030


namespace CombinedHeightOfTowersIsCorrect_l292_292244

-- Define the heights as non-negative reals for clarity.
noncomputable def ClydeTowerHeight : ℝ := 5.0625
noncomputable def GraceTowerHeight : ℝ := 40.5
noncomputable def SarahTowerHeight : ℝ := 2 * ClydeTowerHeight
noncomputable def LindaTowerHeight : ℝ := (ClydeTowerHeight + GraceTowerHeight + SarahTowerHeight) / 3
noncomputable def CombinedHeight : ℝ := ClydeTowerHeight + GraceTowerHeight + SarahTowerHeight + LindaTowerHeight

-- State the theorem to be proven
theorem CombinedHeightOfTowersIsCorrect : CombinedHeight = 74.25 := 
by
  sorry

end CombinedHeightOfTowersIsCorrect_l292_292244


namespace courtyard_width_theorem_l292_292145

noncomputable def paving_stone_area (length : ℝ) (width : ℝ) : ℝ :=
  length * width

noncomputable def total_paving_stones_area (num_stones : ℕ) (stone_area : ℝ) : ℝ :=
  num_stones * stone_area

noncomputable def courtyard_width (total_area : ℝ) (length : ℝ) : ℝ :=
  total_area / length

theorem courtyard_width_theorem :
  let length_courtyard := 50
  let num_stones := 165
  let length_stone := 5 / 2  -- 2 1/2 meters
  let width_stone := 2
  let stone_area := paving_stone_area length_stone width_stone
  let total_area := total_paving_stones_area num_stones stone_area
  courtyard_width total_area length_courtyard = 16.5 :=
by
  let length_courtyard := 50
  let num_stones := 165
  let length_stone := 5 / 2
  let width_stone := 2
  let stone_area := paving_stone_area length_stone width_stone
  let total_area := total_paving_stones_area num_stones stone_area
  calc
    courtyard_width total_area length_courtyard = total_area / length_courtyard := rfl
    ... = 825 / 50 := by calc
      total_area = num_stones * stone_area := rfl
      ... = 165 * (5 / 2 * 2) := rfl
      ... = 165 * 5 := by norm_num
      ... = 825 := rfl
    ... = 16.5 := by norm_num

end courtyard_width_theorem_l292_292145


namespace slope_of_monotonically_decreasing_function_l292_292710

theorem slope_of_monotonically_decreasing_function
  (k b : ℝ)
  (H : ∀ x₁ x₂, x₁ ≤ x₂ → k * x₁ + b ≥ k * x₂ + b) : k < 0 := sorry

end slope_of_monotonically_decreasing_function_l292_292710


namespace maximum_flights_same_airline_l292_292734

theorem maximum_flights_same_airline (n : ℕ) (h_n : n = 2015) : 
  ∃ k, k = ⌈2 * n / 5⌉ ∧ ∀ flights : (Σ (i j : Fin n), i ≠ j) → ℕ, 
  ∃ city : Fin n, ∃ a : Fin n → ℕ, (∀ (i j k : Fin n), i ≠ j → j ≠ k → k ≠ i → flights ⟨i, j, h⟩ = flights ⟨i, k, h⟩ ∨ flights ⟨j, k, h⟩) → 
  a city = k := sorry

end maximum_flights_same_airline_l292_292734


namespace crescent_hexagon_area_ratio_l292_292504

-- Define the structure, properties, and shapes in the problem.
structure RegularHexagon (r : ℝ) :=
  (vertices : Fin 6 → ℝ × ℝ)
  (radius : ℝ := r)
  (center : ℝ × ℝ := (0, 0)) -- Assume the hexagon is centered at the origin
  (side_length : ℝ)

-- Four semicircles being drawn on specific chords of the hexagon
def semicircle_area (s : ℝ) : ℝ :=
  (1 / 2) * real.pi * (s / 2) ^ 2

def total_semicircles_area (s : ℝ) : ℝ :=
  4 * semicircle_area(s)

def circle_area (r : ℝ) : ℝ :=
  real.pi * r ^ 2

def hexagon_area (s : ℝ) : ℝ :=
  (3 * real.sqrt 3 / 2) * s ^ 2

def crescents_area (s : ℝ) (r : ℝ) : ℝ :=
  total_semicircles_area(s) - circle_area(r)

def ratio_of_crescents_area_to_hexagon_area (s : ℝ) : ℝ :=
  crescents_area(s, s) / hexagon_area(s)

-- The proof statement
theorem crescent_hexagon_area_ratio : ∀ (s : ℝ) (h : s > 0),
  ratio_of_crescents_area_to_hexagon_area(s) = 2 / 3 := 
by
  sorry

end crescent_hexagon_area_ratio_l292_292504


namespace vector_ED_eq_l292_292009

structure Triangle (Point : Type _) :=
(A B C : Point)

variables {Point : Type _} [AddCommGroup Point] [VectorSpace ℚ Point]

def isMidpoint (E A C : Point) : Prop :=
E = (A + C) / 2

def satisfiesBDCond (B D C : Point) : Prop :=
B - D = 3 • (D - C)

theorem vector_ED_eq 
(T : Triangle Point) (D E : Point)
(h1 : satisfiesBDCond T.B D T.C)
(h2 : isMidpoint E T.A T.C) : 
E - D = (1 / 4 : ℚ) • (T.B - T.A) + (1 / 4 : ℚ) • (T.C - T.A) :=
sorry

end vector_ED_eq_l292_292009


namespace t_minus_s_eq_l292_292970

def students : ℕ := 120
def teachers : ℕ := 6
def enrollments : List ℕ := [40, 30, 20, 10, 10, 10]

noncomputable def t : ℚ :=
  (enrollments.sum : ℚ) / (teachers : ℚ)

noncomputable def s : ℚ :=
  (∑ e in enrollments, (e * e : ℚ) / (students : ℚ)) / (students : ℚ)

theorem t_minus_s_eq : t - s = -6.67 := by
  sorry

end t_minus_s_eq_l292_292970


namespace checkerboard_squares_count_l292_292526

-- Define the problem in a Lean 4 theorem statement.
theorem checkerboard_squares_count :
  ∀ (n : ℕ), n = 10 →
    (nat.sum (list.map (λ x, nat.sum (list.map (λ y, if x ≥ 5 ∧ y ≥ 5 then 1 else 0) (list.range (n - x + 1)))) (list.range n)) = 91) :=
by
  intros n h
  sorry

end checkerboard_squares_count_l292_292526


namespace units_digit_of_pow_sum_is_correct_l292_292909

theorem units_digit_of_pow_sum_is_correct : 
  (24^3 + 42^3) % 10 = 2 := by
  -- Start the proof block
  sorry

end units_digit_of_pow_sum_is_correct_l292_292909


namespace min_cubic_mice_is_27_l292_292804

noncomputable def min_cubic_mice : ℕ :=
  let a := [1, 2, 4, 5, 7, 8]  -- Represent number of mice on each face
  let min_mice := a.sum       -- Sum up the values to get the total number of mice
  in min_mice

theorem min_cubic_mice_is_27 :
  let a := [1, 2, 4, 5, 7, 8] in
  (∀ i j, i ≠ j → a[i] ≠ a[j]) ∧                    -- Different number of mice on each face
  (∀ i j, adjacent_faces i j → (abs (a[i] - a[j]) ≥ 2)) ∧  -- Number of mice on neighboring faces differ by at least 2
  (∀ i, 0 < a[i]) →         
  min_cubic_mice = 27 :=  -- At least 1 mouse on each face
by
  sorry

end min_cubic_mice_is_27_l292_292804


namespace find_initial_pomelos_l292_292974

theorem find_initial_pomelos (g w w' g' : ℕ) 
  (h1 : w = 3 * g)
  (h2 : w' = w - 90)
  (h3 : g' = g - 60)
  (h4 : w' = 4 * g' - 26) 
  : g = 176 :=
by
  sorry

end find_initial_pomelos_l292_292974


namespace smallest_n_l292_292286

def power_tower (a : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | 1     => a
  | (n+1) => a ^ (power_tower a n)

def pow3_cubed : ℕ := 3 ^ (3 ^ (3 ^ 3))

theorem smallest_n : ∃ n, (∃ k : ℕ, (power_tower 2 n) = k ∧ k > pow3_cubed) ∧ ∀ m, (∃ k : ℕ, (power_tower 2 m) = k ∧ k > pow3_cubed) → m ≥ n :=
  by
  sorry

end smallest_n_l292_292286


namespace total_spent_correct_l292_292780

def t_shirts_Lisa : ℝ := 40
def jeans_Lisa : ℝ := t_shirts_Lisa / 2
def coats_Lisa : ℝ := 2 * t_shirts_Lisa

def t_shirts_Carly : ℝ := t_shirts_Lisa / 4
def jeans_Carly : ℝ := 3 * jeans_Lisa
def coats_Carly : ℝ := coats_Lisa / 4

def total_Lisa : ℝ := t_shirts_Lisa + jeans_Lisa + coats_Lisa
def total_Carly : ℝ := t_shirts_Carly + jeans_Carly + coats_Carly
def total_spent : ℝ := total_Lisa + total_Carly

theorem total_spent_correct : total_spent = 230 := 
by 
  sorry

end total_spent_correct_l292_292780


namespace log_equation_sol_to_inv_pow_l292_292248

theorem log_equation_sol_to_inv_pow (x : ℝ) (h : log (10 * x ^ 3) / log 10 + log (100 * x ^ 4) / log 10 = -1) : 
  1 / (x ^ 6) = 1000 :=
sorry

end log_equation_sol_to_inv_pow_l292_292248


namespace sum_of_arc_lengths_equals_circumference_l292_292839

noncomputable def circumference_of_circle (R : ℝ) : ℝ := 2 * Real.pi * R

theorem sum_of_arc_lengths_equals_circumference (R : ℝ) (n : ℕ) (hn : 0 < n) :
  let C := circumference_of_circle R in
  2 * n * (Real.pi * C / (2 * n)) = C :=
by
  -- Let C be the circumference of the circle
  let C := circumference_of_circle R
  
  -- Intermediate steps involving definitions can be skipped for this task
  -- Proof steps to be filled
  
  -- First separate the portions for easier understanding if necessary
  -- Final equality to show, hence this is a placeholder for the actual proof steps
  sorry

end sum_of_arc_lengths_equals_circumference_l292_292839


namespace height_of_small_cone_removed_l292_292957

-- Definitions based on conditions
def frustum_height : ℝ := 30
def lower_base_area : ℝ := 400 * Real.pi
def upper_base_area : ℝ := 100 * Real.pi

-- The statement that needs to be proved
theorem height_of_small_cone_removed (H h : ℝ) 
  (height_frustum : H / 2 = frustum_height) 
  (h_calculated : h = H - frustum_height) : 
  h = 30 :=
by
  sorry

end height_of_small_cone_removed_l292_292957


namespace outlet_pipe_empties_2_over_3_in_16_min_l292_292223

def outlet_pipe_part_empty_in_t (t : ℕ) (part_per_8_min : ℚ) : ℚ :=
  (part_per_8_min / 8) * t

theorem outlet_pipe_empties_2_over_3_in_16_min (
  part_per_8_min : ℚ := 1/3
) : outlet_pipe_part_empty_in_t 16 part_per_8_min = 2/3 :=
by
  sorry

end outlet_pipe_empties_2_over_3_in_16_min_l292_292223


namespace lg_equation_solution_exponential_inequality_solution_l292_292524

-- Definition and corresponding problem for the equation
theorem lg_equation_solution (x : ℝ) (h1 : x + 1 > 0) (h2 : x - 2 > 0) : 
  (Real.log (x + 1) + Real.log (x - 2) = Real.log 4) → x = 3 := by
  sorry

-- Definition and corresponding problem for the inequality
theorem exponential_inequality_solution (x : ℝ) : 
  (2^(1 - 2 * x) > 1 / 4) → 
  x < 3 / 2 := by
  sorry

end lg_equation_solution_exponential_inequality_solution_l292_292524


namespace EP_EQ_l292_292580

-- Define circles, points, and tangents
variables (Γ1 Γ2 : Type) [circle Γ1] [circle Γ2]
variables (M N A B C D E P Q : point)

-- Define that Γ1 and Γ2 intersect at M and N
axiom intersect_circles : ∀ (M N : point), ∃ (p : point), p ∈ Γ1 ∧ p ∈ Γ2 ∧ (p = M ∨ p = N)

-- Define a common tangent line closer to M
axiom tangent_line_closer_to_M : ∀ l : line, tangent_tangent l Γ1 A ∧ tangent_tangent l Γ2 B ∧ closer_to M l

-- Define line through M parallel to l intersects Γ1 at C and Γ2 at D
axiom parallel_intersection : ∀ l : line, ∃ M l M' C D,
  parallel l M l ∧ (M' ∈ Γ1 → M' = C) ∧ (M' ∈ Γ2 → M' = D)

-- Define intersections and properties
axiom intersect_CA_DB : ∀ (CA DB : line), intersect CA DB E
axiom intersect_AN_BN_CD : ∀ (AN BN CD : line), intersect AN P Q, intersect BN Q P

-- The goal to prove
theorem EP_EQ : EP = EQ :=
  by {
    -- Given the provided axioms and geometric configurations
    sorry
  }

end EP_EQ_l292_292580


namespace units_digit_sum_l292_292894

theorem units_digit_sum (h₁ : (24 : ℕ) % 10 = 4) 
                        (h₂ : (42 : ℕ) % 10 = 2) : 
  ((24^3 + 42^3) % 10 = 2) :=
by
  sorry

end units_digit_sum_l292_292894


namespace units_digit_of_sum_of_cubes_l292_292901

theorem units_digit_of_sum_of_cubes : 
  (24^3 + 42^3) % 10 = 2 := by
sorry

end units_digit_of_sum_of_cubes_l292_292901


namespace total_cost_correct_l292_292078

noncomputable def camera_old_cost : ℝ := 4000
noncomputable def camera_new_cost := camera_old_cost * 1.30
noncomputable def lens_cost := 400
noncomputable def lens_discount := 200
noncomputable def lens_discounted_price := lens_cost - lens_discount
noncomputable def total_cost := camera_new_cost + lens_discounted_price

theorem total_cost_correct :
  total_cost = 5400 := by
  sorry

end total_cost_correct_l292_292078


namespace yearly_feeding_cost_l292_292694

-- Defining the conditions
def num_geckos := 3
def num_iguanas := 2
def num_snakes := 4

def cost_per_snake_per_month := 10
def cost_per_iguana_per_month := 5
def cost_per_gecko_per_month := 15

-- Statement of the proof problem
theorem yearly_feeding_cost : 
  (num_snakes * cost_per_snake_per_month + num_iguanas * cost_per_iguana_per_month + num_geckos * cost_per_gecko_per_month) * 12 = 1140 := 
  by 
    sorry

end yearly_feeding_cost_l292_292694


namespace find_vertex_angle_of_third_cone_l292_292144

noncomputable def vertex_angle_cone_1 : ℝ := π / 3
noncomputable def vertex_angle_cone_2 : ℝ := π / 3
noncomputable def vertex_angle_cone_3 : ℝ := 2 * Real.arccot (2 * (Real.sqrt 3 - Real.sqrt 2))

lemma conditions {α β γ : ℝ} (h1 : α = π / 3) (h2 : β = π / 3) :
  ∃ γ, γ = 2 * Real.arccot (2 * (Real.sqrt 3 - Real.sqrt 2)) ∨ γ = 2 * Real.arccot (2 * (Real.sqrt 3 + Real.sqrt 2)) :=
sorry

theorem find_vertex_angle_of_third_cone :
  ∃ γ, γ = 2 * Real.arccot (2 * (Real.sqrt 3 - Real.sqrt 2)) ∨ γ = 2 * Real.arccot (2 * (Real.sqrt 3 + Real.sqrt 2)) :=
begin
  apply conditions,
  exact π / 3, -- this is condition alpha
  exact π / 3  -- this is condition beta
end

end find_vertex_angle_of_third_cone_l292_292144


namespace find_B_inter_complement_U_A_l292_292352

-- Define Universal set U
def U : Set ℤ := {-1, 0, 1, 2, 3, 4}

-- Define Set A
def A : Set ℤ := {2, 3}

-- Define complement of A relative to U
def complement_U_A : Set ℤ := U \ A

-- Define set B
def B : Set ℤ := {1, 4}

-- The goal to prove
theorem find_B_inter_complement_U_A : B ∩ complement_U_A = {1, 4} :=
by 
  have h1 : A = {2, 3} := rfl
  have h2 : U = {-1, 0, 1, 2, 3, 4} := rfl
  have h3 : B = {1, 4} := rfl
  sorry

end find_B_inter_complement_U_A_l292_292352


namespace smallest_circle_N_l292_292932

theorem smallest_circle_N 
    (N : ℕ)
    (digits : Fin N → ℕ)
    (h1 : ∀ i, digits i = 1 ∨ digits i = 2)
    (h2 : ∀ seq : List ℕ, (seq = [1,1,1,1] ∨ seq = [2,1,1,2] ∨ seq = [2,1,2,2] ∨ seq = [2,2,2,2] ∨ seq = [1,2,2,1] ∨ seq = [1,2,1,1]) → 
          ∃ (i : Fin N), seq = (digits i :: digits (i + 1) :: digits (i + 2) :: digits (i + 3) :: [])) :
    N = 14 :=
by
  sorry

end smallest_circle_N_l292_292932


namespace correct_statements_l292_292711

variable (f : ℝ → ℝ)

-- Conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x) = f(-x)

def y_is_f_of_x_plus_1_is_even : Prop :=
  is_even_function (λ x, f(x + 1))

-- Lean statement for the problem
theorem correct_statements (h : y_is_f_of_x_plus_1_is_even f) :
  (∀ x : ℝ, f (1 + x) = f (1 - x)) ∧
  (∀ x : ℝ, f (x + 1) = f(x - 1)) :=
sorry

end correct_statements_l292_292711


namespace eval_expression_l292_292270

theorem eval_expression (a x : ℤ) (h : x = a + 9) : (x - a + 5) = 14 :=
by
  sorry

end eval_expression_l292_292270


namespace percentage_of_students_70_79_l292_292381

-- Defining basic conditions
def students_in_range_90_100 := 5
def students_in_range_80_89 := 9
def students_in_range_70_79 := 7
def students_in_range_60_69 := 4
def students_below_60 := 3

-- Total number of students
def total_students := students_in_range_90_100 + students_in_range_80_89 + students_in_range_70_79 + students_in_range_60_69 + students_below_60

-- Percentage of students in the 70%-79% range
def percent_students_70_79 := (students_in_range_70_79 / total_students) * 100

theorem percentage_of_students_70_79 : percent_students_70_79 = 25 := by
  sorry

end percentage_of_students_70_79_l292_292381


namespace solve1_solution_solve2_solution_solve3_solution_solve4_solution_l292_292458

noncomputable def solve1 (x : ℝ) : Prop := x^2 - 7 = 0
noncomputable def solve2 (x : ℝ) : Prop := x^2 + 8x = 0
noncomputable def solve3 (x : ℝ) : Prop := x^2 - 4x - 3 = 0
noncomputable def solve4 (x : ℝ) : Prop := x * (x - 2) = 2 - x

theorem solve1_solution :
  ∃ x1 x2 : ℝ, solve1 x1 ∧ solve1 x2 ∧ x1 = Real.sqrt 7 ∧ x2 = -Real.sqrt 7 := by
  sorry

theorem solve2_solution :
  ∃ x1 x2 : ℝ, solve2 x1 ∧ solve2 x2 ∧ x1 = 0 ∧ x2 = -8 := by
  sorry

theorem solve3_solution :
  ∃ x1 x2 : ℝ, solve3 x1 ∧ solve3 x2 ∧ x1 = 2 + Real.sqrt 7 ∧ x2 = 2 - Real.sqrt 7 := by
  sorry

theorem solve4_solution :
  ∃ x1 x2 : ℝ, solve4 x1 ∧ solve4 x2 ∧ x1 = 2 ∧ x2 = -1 := by
  sorry

end solve1_solution_solve2_solution_solve3_solution_solve4_solution_l292_292458


namespace probability_product_divisible_by_4_l292_292154

theorem probability_product_divisible_by_4 :
  (∀ (rolls : list (fin 6)) (h : rolls.length = 8), (∃ (p : ℚ), p = (247/256) ∧ product_divisible_by_4 rolls)) :=
begin
  sorry
end

def product_divisible_by_4 (rolls : list (fin 6)) : Prop :=
  4 ∣ rolls.prod

end probability_product_divisible_by_4_l292_292154


namespace expression_invariant_l292_292759

theorem expression_invariant (m n : ℕ) (hm : Nat.Coprime m n) (hmeven : Even m) (hnodd : Odd n) :
  (1 / (2 * n : ℚ) + ∑ k in Finset.range n \ {0}, (-1) ^ ⌊(m * k / n : ℚ)⌋ * (m * k / n - ⌊(m * k / n : ℚ)⌋)) = 1 / 2 :=
sorry

end expression_invariant_l292_292759


namespace ratio_of_A_to_B_l292_292940

-- Definitions of the conditions.
def amount_A : ℕ := 200
def total_amount : ℕ := 600
def amount_B : ℕ := total_amount - amount_A

-- The proof statement.
theorem ratio_of_A_to_B :
  amount_A / amount_B = 1 / 2 := 
sorry

end ratio_of_A_to_B_l292_292940


namespace geometric_sequence_sixth_term_l292_292539

theorem geometric_sequence_sixth_term :
  (∃ r : ℕ, r^4 = 81 ∧ 3 * r^5 = 729) :=
begin
  use 3, -- r
  split,
  { -- r^4 = 81
    sorry
  },
  { -- 3 * r^5 = 729
    sorry
  }
end

end geometric_sequence_sixth_term_l292_292539


namespace right_angled_triangle_set_B_not_right_angled_triangle_set_A_not_right_angled_triangle_set_C_not_right_angled_triangle_set_D_l292_292220

/-!
# Right-Angled Triangle Verification
We need to verify which of the given sets of numbers can form a right-angled triangle.
-/

theorem right_angled_triangle_set_B :
  let a := 1
  let b := Real.sqrt 2
  let c := Real.sqrt 3
  (a^2 + b^2 = c^2) := by
    unfold a
    unfold b
    unfold c
    sorry

theorem not_right_angled_triangle_set_A :
  let a := 2
  let b := 3
  let c := 4
  ¬ (a^2 + b^2 = c^2) := by
    unfold a
    unfold b
    unfold c
    sorry

theorem not_right_angled_triangle_set_C :
  let a := 4
  let b := 6
  let c := 8
  ¬ (a^2 + b^2 = c^2) := by
    unfold a
    unfold b
    unfold c
    sorry

theorem not_right_angled_triangle_set_D :
  let a := 5
  let b := 12
  let c := 15
  ¬ (a^2 + b^2 = c^2) := by
    unfold a
    unfold b
    unfold c
    sorry

end right_angled_triangle_set_B_not_right_angled_triangle_set_A_not_right_angled_triangle_set_C_not_right_angled_triangle_set_D_l292_292220


namespace sequence_existence_l292_292666

variable (a b : ℕ) (b_seq : ℕ → ℕ)

noncomputable def satisfies_conditions (a_seq : ℕ → ℕ) : Prop :=
  ∀ n : ℕ,
    (a_seq (n+1) - a_seq n ∈ {a, b}) ∧
    ∀ m l : ℕ, a_seq l + a_seq m ∉ {b_seq k | k : ℕ}

theorem sequence_existence (ha : a > 1)
                          (hb : b > a)
                          (h_adivb : ¬ a ∣ b)
                          (h_bseq : ∀ n : ℕ, b_seq (n+1) ≥ 2 * b_seq n) :
  ∃ a_seq : ℕ → ℕ, satisfies_conditions a b b_seq a_seq :=
by
  sorry

end sequence_existence_l292_292666


namespace atomic_weight_aluminum_l292_292625

theorem atomic_weight_aluminum :
  ∀ (al_weight br_weight : ℝ) (num_br_atoms : ℕ) (total_weight : ℝ),
  num_br_atoms = 3 →
  br_weight = 79.9 →
  total_weight = 267 →
  al_weight = total_weight - (num_br_atoms * br_weight) →
  al_weight = 27.3 :=
by
  intros al_weight br_weight num_br_atoms total_weight h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  linarith

end atomic_weight_aluminum_l292_292625


namespace math_proof_l292_292432

noncomputable def proof_problem (n : ℕ) (x : ℝ) (a : ℕ → ℝ) : Prop :=
  (n ≥ 2) →
  (∀ i j : ℕ, 1 ≤ i → i ≤ j → j ≤ n → a i ≤ a j) →
  (0 < a 1) →
  ((∏ i in (finset.range n).card, a (i + 1)) ≤ x) →
  ((∏ i in (finset.range (n - 1)).card, a (i + 1)) ≤ x^(1 - 1/n))

theorem math_proof (n : ℕ) (x : ℝ) (a : ℕ → ℝ) : proof_problem n x a :=
begin
  intros h1 h2 h3 h4,
  sorry
end

end math_proof_l292_292432


namespace faye_total_crayons_l292_292618

  def num_rows : ℕ := 16
  def crayons_per_row : ℕ := 6
  def total_crayons : ℕ := num_rows * crayons_per_row

  theorem faye_total_crayons : total_crayons = 96 :=
  by
  sorry
  
end faye_total_crayons_l292_292618


namespace parabola_vertex_n_l292_292137

theorem parabola_vertex_n (x y : ℝ) (h : y = -3 * x^2 - 24 * x - 72) : ∃ m n : ℝ, (m, n) = (-4, -24) :=
by
  sorry

end parabola_vertex_n_l292_292137


namespace megan_works_per_day_hours_l292_292439

theorem megan_works_per_day_hours
  (h : ℝ)
  (earnings_per_hour : ℝ)
  (days_per_month : ℝ)
  (total_earnings_two_months : ℝ) :
  earnings_per_hour = 7.50 →
  days_per_month = 20 →
  total_earnings_two_months = 2400 →
  2 * days_per_month * earnings_per_hour * h = total_earnings_two_months →
  h = 8 :=
by {
  sorry
}

end megan_works_per_day_hours_l292_292439


namespace geometric_series_sum_l292_292588

theorem geometric_series_sum :
  ∑' i : ℕ, (2 / 3) ^ (i + 1) = 2 :=
by
  sorry

end geometric_series_sum_l292_292588


namespace mike_taller_than_mark_l292_292783

def height_mark_feet : ℕ := 5
def height_mark_inches : ℕ := 3
def height_mike_feet : ℕ := 6
def height_mike_inches : ℕ := 1
def feet_to_inches : ℕ := 12

-- Calculate heights in inches.
def height_mark_total_inches : ℕ := height_mark_feet * feet_to_inches + height_mark_inches
def height_mike_total_inches : ℕ := height_mike_feet * feet_to_inches + height_mike_inches

-- Prove the height difference.
theorem mike_taller_than_mark : height_mike_total_inches - height_mark_total_inches = 10 :=
by
  sorry

end mike_taller_than_mark_l292_292783


namespace four_cells_different_colors_l292_292312

theorem four_cells_different_colors
  (n : ℕ)
  (h_n : n ≥ 2)
  (coloring : Fin n → Fin n → Fin (2 * n)) :
  ∃ (r1 r2 c1 c2 : Fin n),
    r1 ≠ r2 ∧ c1 ≠ c2 ∧
    (coloring r1 c1 ≠ coloring r1 c2) ∧
    (coloring r1 c1 ≠ coloring r2 c1) ∧
    (coloring r1 c2 ≠ coloring r2 c2) ∧
    (coloring r2 c1 ≠ coloring r2 c2) := 
sorry

end four_cells_different_colors_l292_292312


namespace solution_set_of_inequality_l292_292851

theorem solution_set_of_inequality (x : ℝ) : x > 1 ∨ (-1 < x ∧ x < 0) ↔ x > 1 ∨ (-1 < x ∧ x < 0) :=
by sorry

end solution_set_of_inequality_l292_292851


namespace number_of_students_in_class_l292_292464

theorem number_of_students_in_class :
  ∃ n : ℕ, n > 0 ∧ (∀ avg_age teacher_age total_avg_age, avg_age = 26 ∧ teacher_age = 52 ∧ total_avg_age = 27 →
    (∃ total_student_age total_age_with_teacher, 
      total_student_age = n * avg_age ∧ 
      total_age_with_teacher = total_student_age + teacher_age ∧ 
      (total_age_with_teacher / (n + 1) = total_avg_age) → n = 25)) :=
sorry

end number_of_students_in_class_l292_292464


namespace ab_product_eq_four_l292_292702

theorem ab_product_eq_four (a b : ℝ) (h1: 0 < a) (h2: 0 < b) 
  (h3: (1/2) * (4 / a) * (6 / b) = 3) : 
  a * b = 4 :=
by 
  sorry

end ab_product_eq_four_l292_292702


namespace axis_of_symmetry_l292_292466

variable (A B : ℝ × ℝ)
variable (hA : A = (-1, 2))
variable (hB : B = (-1, 6))

theorem axis_of_symmetry : ∃ l : ℝ, l = 4 ∧ (∀ x : ℝ, A = (x, 2) ∧ B = (x, 6)) -> (l = 1/2 * ((A.2 + B.2))) :=
by intros A B hA hB
sorry

end axis_of_symmetry_l292_292466


namespace derivative_of_f_l292_292620

-- Define the function f(x) = x * exp(x)
def f (x : ℝ) : ℝ := x * exp x

-- State the main theorem problem
theorem derivative_of_f : 
  ∀ x : ℝ, deriv f x = (1 + x) * exp x :=
by sorry

end derivative_of_f_l292_292620


namespace bookstore_purchase_ways_l292_292212

theorem bookstore_purchase_ways (h : ∃ (b1 b2 b3 : bool), b1 ∨ b2 ∨ b3) : 
  ∃ n, n = 7 :=
by
  sorry

end bookstore_purchase_ways_l292_292212


namespace unique_real_solution_l292_292255

theorem unique_real_solution :
  ∀ x : ℝ, (2^(8*x + 4)) * (16^(x + 2)) = 8^(6*x + 10) → x = -3 := 
  by 
    sorry

end unique_real_solution_l292_292255


namespace length_of_bridge_is_1082_l292_292976

-- Define the given conditions
def length_of_train : ℝ := 250
def crossing_time : ℝ := 20
def speed_of_train : ℝ := 66.6
def total_distance : ℝ := speed_of_train * crossing_time

-- Define the total distance covered by the train as the sum of the train's length and the bridge's length
def bridge_length := total_distance - length_of_train

-- State the theorem to prove the length of the bridge
theorem length_of_bridge_is_1082 : bridge_length = 1082 := by
  have h1: total_distance = speed_of_train * crossing_time := rfl
  have h2: bridge_length = total_distance - length_of_train := rfl
  have h3: total_distance = 1332 := by
    calc total_distance = speed_of_train * crossing_time : h1
                    ... = 66.6 * 20                  : by rfl
                    ... = 1332                      : by sorry
  have h4: bridge_length = 1332 - length_of_train := by rw [h2, h3]
  have h5: bridge_length = 1332 - 250 := by rw [h4]
  have h6: bridge_length = 1082 := by sorry
  exact h6

end length_of_bridge_is_1082_l292_292976


namespace rhombus_area_l292_292471

-- Define the lengths of the diagonals
def d1 : ℝ := 25
def d2 : ℝ := 30

-- Statement to prove that the area of the rhombus is 375 square centimeters
theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 25) (h2 : d2 = 30) : 
  (d1 * d2) / 2 = 375 := by
  -- Proof to be provided
  sorry

end rhombus_area_l292_292471


namespace distance_traveled_eq_12pi_l292_292559

-- Define the radius of the wheel as 2 meters
def radius : ℝ := 2

-- Define the number of revolutions
def revolutions : ℕ := 3

-- Define the formula for the circumference of a circle
def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

-- Define the total distance traveled given the number of revolutions and the circumference
def total_distance_traveled (r : ℝ) (revs : ℕ) : ℝ := revs * (circumference r)

-- Prove that the total distance traveled is 12π meters
theorem distance_traveled_eq_12pi : total_distance_traveled radius revolutions = 12 * Real.pi := by
  sorry

end distance_traveled_eq_12pi_l292_292559


namespace general_formula_an_sum_first_n_terms_bn_l292_292656

open Nat

def seq_an (n : ℕ) : ℕ := 2n + 1

def seq_bn (n : ℕ) : ℕ := 3^(seq_an n)

def sum_bn (n : ℕ) : ℚ := (27 / 8) * (9^n - 1)

theorem general_formula_an (n : ℕ) :
  seq_an n = 2 * n + 1 := sorry

theorem sum_first_n_terms_bn (n : ℕ) :
  (∑ i in range n, seq_bn i) = sum_bn n := sorry

end general_formula_an_sum_first_n_terms_bn_l292_292656


namespace rectangle_no_shaded_square_l292_292599

noncomputable def total_rectangles (cols : ℕ) : ℕ :=
  (cols + 1) * (cols + 1 - 1) / 2

noncomputable def shaded_rectangles (cols : ℕ) : ℕ :=
  cols + 1 - 1

noncomputable def probability_no_shaded (cols : ℕ) : ℚ :=
  let n := total_rectangles cols
  let m := shaded_rectangles cols
  1 - (m / n)

theorem rectangle_no_shaded_square :
  probability_no_shaded 2003 = 2002 / 2003 :=
by
  sorry

end rectangle_no_shaded_square_l292_292599


namespace petya_points_l292_292384

noncomputable def points_after_disqualification : ℕ :=
4

theorem petya_points (players: ℕ) (initial_points: ℕ) (disqualified: ℕ) (new_points: ℕ) : 
  players = 10 → 
  initial_points < (players * (players - 1) / 2) / players → 
  disqualified = 2 → 
  (players - disqualified) * (players - disqualified - 1) / 2 = new_points →
  new_points / (players - disqualified) < points_after_disqualification →
  points_after_disqualification > new_points / (players - disqualified) →
  points_after_disqualification = 4 :=
by 
  intros 
  exact sorry

end petya_points_l292_292384


namespace find_a_l292_292994

variable {a b : ℝ} (h_pos_a : 0 < a) (h_pos_b : 0 < b)
variable (h_min_y : ∀ x, a * Real.sec (b * x) > 0 → a * Real.sec (b * x) ≥ 3)

theorem find_a : a = 3 :=
sorry

end find_a_l292_292994


namespace horizontal_distance_l292_292541

def curve (x : ℝ) := x^3 - x^2 - x - 6

def P_condition (x : ℝ) := curve x = 10
def Q_condition1 (x : ℝ) := curve x = 2
def Q_condition2 (x : ℝ) := curve x = -2

theorem horizontal_distance (x_P x_Q: ℝ) (hP: P_condition x_P) (hQ1: Q_condition1 x_Q ∨ Q_condition2 x_Q) :
  |x_P - x_Q| = 3 := sorry

end horizontal_distance_l292_292541


namespace monotonic_increasing_interval_f_l292_292848

-- Define the function and conditions
def t (x : ℝ) : ℝ := Real.sqrt (x - x^2)

def f (x : ℝ) : ℝ := (1 / 2) ^ t x

-- Define the condition x - x^2 ≥ 0, and its domain 0 ≤ x ≤ 1
axiom cond : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 ↔ x - x^2 ≥ 0

-- The statement to prove: The monotonic increasing interval of f(x) is [1/2, 1]
theorem monotonic_increasing_interval_f : ∀ x : ℝ, cond x → (1 / 2) ^ t x ∈ Set.Icc (1 / 2) 1 := 
sorry

end monotonic_increasing_interval_f_l292_292848


namespace modulus_of_alpha_l292_292058

variable (α β : ℂ)

-- Assumptions
axiom conj_of_alpha_and_beta : β = conj α
axiom distance_between_conjugates : abs (α - β) = 2 * sqrt 3
axiom alpha_over_beta_squared_is_real : (α / β^2).im = 0

-- The theorem to prove
theorem modulus_of_alpha : abs α = 2 :=
by
  sorry

end modulus_of_alpha_l292_292058


namespace select_cells_possible_l292_292838

def exists_selection (board : Fin 8 → Fin 8 → Fin 32) : Prop :=
  ∃ (selected : Fin 8 → Fin 8 → Prop), 
    (∀ r c, ∃ r', selected r' c) ∧
    (∀ r c, ∃ c', selected r c') ∧
    (∀ r c₁ c₂, selected r c₁ → selected r c₂ → c₁ = c₂) ∧
    (∀ c r₁ r₂, selected r₁ c → selected r₂ c → r₁ = r₂) ∧
    (∀ r c₁ c₂, r ≠ c₁ → r ≠ c₂ → board r c₁ = board r c₂ → 
      selected r c₁ ∧ selected r c₂ → c₁ = c₂) ∧
    (∀ c r₁ r₂, c ≠ r₁ → c ≠ r₂ → board r₁ c = board r₂ c → 
      selected r₁ c ∧ selected r₂ c → r₁ = r₂)

theorem select_cells_possible (board : Fin 8 → Fin 8 → Fin 32) 
  (h : ∀ n : Fin 32, ∃ p₁ p₂ : (Fin 8 × Fin 8), 
    p₁ ≠ p₂ ∧ board p₁.1 p₁.2 = n ∧ board p₂.1 p₂.2 = n) : 
    exists_selection board :=
sorry

end select_cells_possible_l292_292838


namespace Christopher_speed_is_4_l292_292243

variable (distance : ℝ) (time : ℝ)

def Christopher_speed := distance / time

theorem Christopher_speed_is_4
  (h1 : distance = 5)
  (h2 : time = 1.25) : Christopher_speed distance time = 4 := by
  sorry

end Christopher_speed_is_4_l292_292243


namespace fraction_staff_ate_pizza_l292_292229

theorem fraction_staff_ate_pizza (teachers staff : ℕ) (fraction_teachers_ate_pizza : ℚ) (non_pizza_eaters : ℕ) 
  (h_teachers : teachers = 30) 
  (h_staff : staff = 45) 
  (h_fraction_teachers_ate_pizza : fraction_teachers_ate_pizza = 2 / 3) 
  (h_non_pizza_eaters : non_pizza_eaters = 19) : 
  let teachers_ate_pizza := fraction_teachers_ate_pizza * teachers in ∃ (fraction_staff_ate_pizza : ℚ), fraction_staff_ate_pizza = 4 / 5 :=
by
  let teachers_ate_pizza := fraction_teachers_ate_pizza * teachers
  let teachers_did_not_eat_pizza := teachers - teachers_ate_pizza
  let staff_did_not_eat_pizza := non_pizza_eaters - teachers_did_not_eat_pizza
  let staff_ate_pizza := staff - staff_did_not_eat_pizza
  let fraction_staff_ate_pizza := staff_ate_pizza / staff
  use fraction_staff_ate_pizza
  sorry

end fraction_staff_ate_pizza_l292_292229


namespace angle_E_in_quadrilateral_l292_292394

theorem angle_E_in_quadrilateral (E F G H : ℝ) 
  (h1 : E = 5 * H)
  (h2 : E = 4 * G)
  (h3 : E = (5/3) * F)
  (h_sum : E + F + G + H = 360) : 
  E = 131 := by 
  sorry

end angle_E_in_quadrilateral_l292_292394


namespace min_y_coord_l292_292624

noncomputable def y_coord (theta : ℝ) : ℝ :=
  (Real.cos (2 * theta)) * (Real.sin theta)

theorem min_y_coord : ∃ theta : ℝ, y_coord theta = - (Real.sqrt 6) / 3 := by
  sorry

end min_y_coord_l292_292624


namespace total_tickets_correct_l292_292849

/-- Represent the ticket cost conditions -/
def cost_of_first_coaster (rides : ℕ) (tickets_per_ride : ℕ) : ℕ := rides * tickets_per_ride

def discount (total_tickets : ℕ) (rate : ℕ → ℕ) : ℕ := rate total_tickets

def cost_of_new_coaster (rides : ℕ) (tickets_per_ride : ℕ) : ℕ := rides * tickets_per_ride

def total_tickets_needed 
  (num_friends : ℕ) 
  (rides_first_coaster : ℕ) 
  (tickets_first_coaster : ℕ) 
  (rides_new_coaster : ℕ) 
  (tickets_new_coaster : ℕ) 
  (discount_rate : ℕ → ℕ) 
  : ℕ :=
let rides_first_total := num_friends * rides_first_coaster in
let rides_new_total := num_friends * rides_new_coaster in
let actual_tickets_first := cost_of_first_coaster rides_first_total tickets_first_coaster in
let rides_with_discount := 10 in
let discount_tickets_first := discount (rides_with_discount * tickets_first_coaster) discount_rate in
let tickets_first_with_discount := rides_with_discount * tickets_first_coaster - discount_tickets_first in
let remaining_rides := rides_first_total - rides_with_discount in
let tickets_for_remaining_ride := remaining_rides * tickets_first_coaster in
let total_first_coaster := tickets_first_with_discount + tickets_for_remaining_ride in
let total_new_coaster := cost_of_new_coaster rides_new_total tickets_new_coaster in
total_first_coaster + total_new_coaster

theorem total_tickets_correct : 
  total_tickets_needed 8 2 6 1 8 (λ x => x * 15 / 100) = 160 := 
by sorry

end total_tickets_correct_l292_292849


namespace angle_KMD_arccos_l292_292736

open Real

theorem angle_KMD_arccos :
  ∃ (K L M D : Point) (KLM : Triangle K L M), 
  right_angle KLM M ∧ (KLM.hypotenuse KL)
  ∧ (segment_md : Segment M D) ∧ (D ∈ L K) 
  ∧ (DL = 1) 
  ∧ (DM = sqrt 2)
  ∧ (DK = 2)
  ∧ is_Geometric_Progression DL DM DK (sqrt 2) 
  ∧ (∠ KMD = arccos (1 / sqrt 3)) :=
sorry

end angle_KMD_arccos_l292_292736


namespace range_m_l292_292638

variable (x m : ℝ)

def P : Prop := |(4 - x) / 3| ≤ 2
def q : Prop := (x + m - 1) * (x - m - 1) ≤ 0

theorem range_m (h : ∀ m > 0, ¬P → ¬q) : 9 ≤ m :=
by
  sorry

end range_m_l292_292638


namespace rectangular_field_length_l292_292479

noncomputable def length_of_rectangular_field (b d : ℝ) : ℝ :=
  let s := d / Real.sqrt 2 in
  let area_square := s ^ 2 in
  area_square / b

theorem rectangular_field_length (b d : ℝ) (hb : b = 80) (hd : d = 120) :
  length_of_rectangular_field b d ≈ 90.009 :=
by
  rw [hb, hd]
  -- Placeholder for proof steps
  sorry

end rectangular_field_length_l292_292479


namespace right_triangle_BD_l292_292430

noncomputable def triangle_ABC_area := 180
noncomputable def AC_length := 30

-- Given that the triangle ABC is a right triangle with the right angle at B,
-- and a circle with diameter BC intersects AC at D,
-- prove that BD is 12
theorem right_triangle_BD (triangle_ABC : Type) [Geometry triangle_ABC]
  (A B C D : triangle_ABC)
  (h_right_angle_B : ∠ B = π / 2)
  (circle_diameter_BC : Circle (segment B C))
  (h_circle_intersects_AC_at_D : D ∈ circle_diameter_BC ∩ line A C)
  (h_area_ABC : area (triangle A B C) = triangle_ABC_area)
  (h_AC : dist A C = AC_length) :
  dist B D = 12 := 
by {
  sorry -- Proof goes here
}

end right_triangle_BD_l292_292430


namespace find_b1_l292_292750

noncomputable def polynomial_f : Polynomial ℝ := 8 + 32 * x - 12 * x^2 - 4 * x^3 + x^4

def roots_of_f : Set ℝ := {x_1, x_2, x_3, x_4}

noncomputable def polynomial_g : Polynomial ℝ := b_0 + b_1 * x + b_2 * x^2 + b_3 * x^3 + x^4

def roots_of_g : Set ℝ := {x_1^2, x_2^2, x_3^2, x_4^2}

theorem find_b1 (h_roots_f : roots polynomial_f = roots_of_f)
                (h_roots_g : roots polynomial_g = roots_of_g) : b_1 = -1216 :=
by
  sorry

end find_b1_l292_292750


namespace mina_crafts_total_l292_292083

theorem mina_crafts_total :
  let a₁ := 3
  let d := 4
  let n := 10
  let crafts_sold_on_day (d: ℕ) := a₁ + (d - 1) * d
  let S (n: ℕ) := (n * (2 * a₁ + (n - 1) * d)) / 2
  S n = 210 :=
by
  sorry

end mina_crafts_total_l292_292083


namespace bella_roses_from_parents_l292_292562

theorem bella_roses_from_parents (roses_from_parents : ℕ) :
  (∃ n, n = roses_from_parents) →
  let roses_from_friends := 2 * 10 in
  let total_roses := 44 in
  total_roses - roses_from_friends = 24 :=
by
  sorry

end bella_roses_from_parents_l292_292562


namespace at_least_one_divisible_by_3_l292_292796

-- Define a function that describes the properties of the numbers as per conditions.
def circle_99_numbers (numbers: Fin 99 → ℕ) : Prop :=
  ∀ n : Fin 99, let neighbor := (n + 1) % 99 
                in abs (numbers n - numbers neighbor) = 1 ∨ 
                   abs (numbers n - numbers neighbor) = 2 ∨ 
                   (numbers n = 2 * numbers neighbor) ∨ 
                   (numbers neighbor = 2 * numbers n)

theorem at_least_one_divisible_by_3 :
  ∀ (numbers: Fin 99 → ℕ), circle_99_numbers numbers → ∃ n : Fin 99, numbers n % 3 = 0 :=
by
  intro numbers
  intro h
  sorry

end at_least_one_divisible_by_3_l292_292796


namespace g_symmetry_g_periodic_value_l292_292304

noncomputable def f (x : ℝ) : ℝ :=
  sin x * cos x + sqrt 3 * cos x ^ 2 - sqrt 3 / 2

noncomputable def g (a x : ℝ) : ℝ := do
  let shiftedF := sin (2 * (a + x / 2)) + 1
  return shiftedF

theorem g_symmetry (a : ℝ) (x : ℝ) : g a (a - x) = g a (a + x) :=
  sorry

theorem g_periodic_value (a : ℝ) : g a (a + π/4) = 0 :=
  sorry

end g_symmetry_g_periodic_value_l292_292304


namespace sum_of_lucky_tickets_is_divisible_l292_292584

def is_lucky (n : ℕ) : Prop :=
  let digits := n.digits 10
  (digits.take 3).sum = (digits.drop 3).sum

def is_in_range (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 999999

theorem sum_of_lucky_tickets_is_divisible :
  ∃ (s : ℕ), (∀ n, is_lucky n → is_in_range n → s = n + ... + n) ∧ 
  9 ∣ s ∧ 13 ∣ s ∧ 37 ∣ s ∧ 1001 ∣ s :=
by
  sorry

end sum_of_lucky_tickets_is_divisible_l292_292584


namespace part1_part2_l292_292684

-- (1) Prove that if 2 ∈ M and M is the solution set of ax^2 + 5x - 2 > 0, then a > -2.
theorem part1 (a : ℝ) (h : 2 * (a * 4 + 10) - 2 > 0) : a > -2 :=
sorry

-- (2) Given M = {x | 1/2 < x < 2} and M is the solution set of ax^2 + 5x - 2 > 0,
-- prove that the solution set of ax^2 - 5x + a^2 - 1 > 0 is -3 < x < 1/2
theorem part2 (a : ℝ) (h1 : ∀ x : ℝ, (1/2 < x ∧ x < 2) ↔ ax^2 + 5*x - 2 > 0) (h2 : a = -2) :
  ∀ x : ℝ, (-3 < x ∧ x < 1/2) ↔ (-2 * x^2 - 5 * x + 3 > 0) :=
sorry

end part1_part2_l292_292684


namespace sum_w_equals_n_times_4_pow_n_minus_1_l292_292051

-- Define the types for sequences of zeros and ones.
def B_n (n : ℕ) := Vector (Finite 2) n

-- Define the sequences ε and δ.
noncomputable def epsilon_seq (a b : B_n n) : Fin (n+1) → ℕ
| ⟨0, _⟩ => 0
| ⟨i+1, p⟩ => (delta_seq a b ⟨i, Nat.le_of_succ_le_succ p⟩ - a.get ⟨i, Nat.le_of_succ_le_succ p⟩) * 
               (delta_seq a b ⟨i, Nat.le_of_succ_le_succ p⟩ - b.get ⟨i, Nat.le_of_succ_le_succ p⟩)

noncomputable def delta_seq (a b : B_n n) : Fin (n+1) → ℕ
| ⟨0, _⟩ => 0
| ⟨i+1, p⟩ => let δ_i := delta_seq a b ⟨i, Nat.le_of_succ_le_succ p⟩
               let ε_i1 := epsilon_seq a b ⟨i+1, p⟩
               δ_i + (-1)^(δ_i) * ε_i1

-- Define w(a, b) as the sum of the epsilon sequence.
def w (a b : B_n n) := (Finset.range (n + 1)).sum (λ i, epsilon_seq a b ⟨i, Nat.le_add_right i n⟩)

-- Define f(n) as the sum of w over all pairs (a, b) in B_n.
def f (n : ℕ) := (Finset.univ : Finset (B_n n)).sum (λ a, (Finset.univ : Finset (B_n n)).sum (λ b, w a b))

theorem sum_w_equals_n_times_4_pow_n_minus_1 (n : ℕ) : f n = n * 4^(n-1) := by
  sorry

end sum_w_equals_n_times_4_pow_n_minus_1_l292_292051


namespace bubble_pass_probability_l292_292833

theorem bubble_pass_probability (s : Fin 50 → ℝ) 
  (hs : Function.Injective s) : 
  let p := 1 in let q := 1190
  in (p + q = 1191) ∧ 
  ∃ t : Fin 50 → ℝ, 
  t 34 = s 24 ∧ 
  (∀ i j : Fin 50, i < j → t i ≤ t j) ∧ 
  (∃ u : Fin 50, u = t ∧ 
  (∀ i j : Fin 50, i < j ∧ i ≠ 34 ∧ j ≠ 34 → s i ≤ s j)) → 
  Rational (1 / 1190) := sorry

end bubble_pass_probability_l292_292833


namespace number_of_sequences_with_constraints_l292_292091

-- Definitions of conditions
def is_valid_sequence (a : ℕ → ℕ) : Prop :=
  (∀ i < 99, (a i = 3 ∨ a (i + 1) = 3) ∧ (a i = a (i + 1) ∨ a i + 1 = a (i + 1) ∨ a i = a (i + 1) + 1))

-- Proof problem statement
theorem number_of_sequences_with_constraints : 
  ∀ (a : ℕ → ℕ) (h : ∀ i < 100, a i ∈ ℕ), 
    (is_valid_sequence a) → (∑ i in finset.range 100, if a i = 3 then 1 else 0) ≥ 1 
      → Σ (λ n, Σ (λ k, (k < n ∧ 3^n - 2^n = k))) :=
sorry

end number_of_sequences_with_constraints_l292_292091


namespace toby_steps_l292_292497

theorem toby_steps (sunday tuesday wednesday thursday friday_saturday monday : ℕ) :
    sunday = 9400 →
    tuesday = 8300 →
    wednesday = 9200 →
    thursday = 8900 →
    friday_saturday = 9050 →
    7 * 9000 = 63000 →
    monday = 63000 - (sunday + tuesday + wednesday + thursday + 2 * friday_saturday) → monday = 9100 :=
by
  intros hs ht hw hth hfs htc hnm
  sorry

end toby_steps_l292_292497


namespace find_sqrt_abc_abc_plus_l292_292424

theorem find_sqrt_abc_abc_plus (a b c : ℝ) (h₁ : b + c = 7) (h₂ : c + a = 8) (h₃ : a + b = 9) :
    sqrt (a * b * c * (a + b + c)) = 12 * sqrt 5 := 
by
  sorry

end find_sqrt_abc_abc_plus_l292_292424


namespace expenditure_constant_if_reduction_l292_292194

def initial_price_A : ℝ := 1
def initial_price_B : ℝ := 1
def initial_price_C : ℝ := 1

def proportion_A : ℝ := 0.40
def proportion_B : ℝ := 0.35
def proportion_C : ℝ := 0.25

def price_increase_A : ℝ := 1.30 * 1.15
def price_increase_B : ℝ := 1.25 * 1.10
def price_increase_C : ℝ := 1.20 * 1.05

def final_price_A := initial_price_A * price_increase_A
def final_price_B := initial_price_B * price_increase_B
def final_price_C := initial_price_C * price_increase_C

def weighted_average_price_increase : ℝ :=
  (proportion_A * final_price_A) + (proportion_B * final_price_B) + (proportion_C * final_price_C)

def required_reduction : ℝ := 1 - 1 / weighted_average_price_increase

theorem expenditure_constant_if_reduction (p_A p_B p_C : ℝ) :
  final_price_A * (1 - required_reduction) * proportion_A + 
  final_price_B * (1 - required_reduction) * proportion_B + 
  final_price_C * (1 - required_reduction) * proportion_C = 
  1 :=
by
  sorry

end expenditure_constant_if_reduction_l292_292194


namespace smallest_number_of_sums_l292_292056

theorem smallest_number_of_sums (n : ℕ) (h : n ≥ 3) 
  (d : ∀ (i j : ℕ), i ≠ j → a i ≠ a j) : 
  ∃ m : ℕ, m = 3 ∧ (forall (S : fin n → ℝ), 
  (fin n).to_finset.card = n → 
  let sums := λ i, S i + S ((i+1) % n) in
  (sums '' (fin n).to_finset).card = m) := 
  sorry

end smallest_number_of_sums_l292_292056


namespace ariel_fish_l292_292570

theorem ariel_fish (total_fish : ℕ) (male_ratio : ℚ) (female_ratio : ℚ) (female_fish : ℕ) : 
  total_fish = 45 ∧ male_ratio = 2/3 ∧ female_ratio = 1/3 → female_fish = 15 :=
by
  sorry

end ariel_fish_l292_292570


namespace min_value_expression_l292_292159

/--
  Prove that the minimum value of the expression (xy - 2)^2 + (x + y - 1)^2 
  for real numbers x and y is 2.
--/
theorem min_value_expression : 
  ∃ x y : ℝ, (∀ a b : ℝ, (a * b - 2)^2 + (a + b - 1)^2 ≥ (x * y - 2)^2 + (x + y - 1)^2 ) ∧ 
  (x * y - 2)^2 + (x + y - 1)^2 = 2 :=
by
  sorry

end min_value_expression_l292_292159


namespace farmer_profit_l292_292954

noncomputable def profit_earned : ℕ :=
  let pigs := 6
  let sale_price := 300
  let food_cost_per_month := 10
  let months_group1 := 12
  let months_group2 := 16
  let pigs_group1 := 3
  let pigs_group2 := 3
  let total_food_cost := (pigs_group1 * months_group1 * food_cost_per_month) + 
                         (pigs_group2 * months_group2 * food_cost_per_month)
  let total_revenue := pigs * sale_price
  total_revenue - total_food_cost

theorem farmer_profit : profit_earned = 960 := by
  unfold profit_earned
  sorry

end farmer_profit_l292_292954


namespace find_tan_2x_l292_292347

-- Define the function f
def f (x : ℝ) : ℝ := sin x - cos x

-- Define the derivative of the function f
def f' (x : ℝ) : ℝ := cos x + sin x

-- State the given condition f'(x) = 2 * f(x)
axiom condition (x : ℝ) : f' x = 2 * f x

-- State the theorem to prove
theorem find_tan_2x (x : ℝ) (h : condition x) : tan (2 * x) = -3 / 4 :=
by
  sorry

end find_tan_2x_l292_292347


namespace circumcircle_eq_l292_292306

-- Define the given circle and external point
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4
def P : ℝ × ℝ := (4, 2)

-- The target is to find the equation of circumcircle of triangle ABP
theorem circumcircle_eq : ∃ h k r : ℝ, (h, k) = P ∧ r = 4 ∧ ∀ x y : ℝ, ((x - h)^2 + (y - k)^2 = r^2) :=
by
  let h := 4
  let k := 2
  let r := 4
  use [h, k, r]
  split
  { exact rfl }
  split
  { exact rfl }
  { intros x y; simp [h, k, r] }
  sorry

end circumcircle_eq_l292_292306


namespace solve_for_a_b_l292_292324

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x + b
def g (x : ℝ) : ℝ := x^2 - x

theorem solve_for_a_b :
  (∃ a b : ℝ, f(a, b, 1) = 0 ∧ (Real.exp 1 - a = 1) ∧ (Real.exp 1 - a + b = 0)) ↔
  (a = Real.exp 1 - 1 ∧ b = -1) :=
by
  sorry

end solve_for_a_b_l292_292324


namespace nails_needed_correct_l292_292043

def nails_needed_for_house_walls : ℕ := 27 * 36 + 15

theorem nails_needed_correct : nails_needed_for_house_walls = 987 := by
  -- Calculation:
  -- nails_needed_for_house_walls = 27 * 36 + 15
  --                         = 972 + 15
  --                         = 987
  rw [nails_needed_for_house_walls]
  norm_num
  done

end nails_needed_correct_l292_292043


namespace M_add_N_l292_292764

-- Definitions based on the conditions in the problem
def is_permutation (x : list ℕ) : Prop := (list.perm x [1, 2, 3, 4, 5, 6])

def my_sum (x : list ℕ) : ℕ :=
  match x with
  | [x₁, x₂, x₃, x₄, x₅, x₆] => x₁ * x₂ + x₂ * x₃ + x₃ * x₄ + x₄ * x₅ + x₅ * x₆ + x₆ * x₁
  | _ => 0 -- This case won't happen for permutations of length 6

def max_value_reached (l : list ℕ) : ℕ :=
  l.permutations.map my_sum).maximum

-- The final goal
theorem M_add_N : ∃ M N : ℕ, M = 76 ∧ N = 12 ∧ M + N = 88 :=
  exists.intro 76 (exists.intro 12 (and.intro rfl (and.intro rfl rfl)))

end M_add_N_l292_292764


namespace c_minus_three_eq_neg_two_l292_292231

variable (g : ℝ → ℝ) (c : ℝ)

-- Define the conditions
axiom gc_three : g(c) = 3
axiom g_three_c : g(3) = c
axiom invertible_g : Function.Injective g

-- The main statement to prove
theorem c_minus_three_eq_neg_two : c - 3 = -2 :=
by
  sorry

end c_minus_three_eq_neg_two_l292_292231


namespace fixed_point_through_l_find_ellipse_equation_l292_292396

noncomputable def ellipse (x y : ℝ) : Prop := 
  (x^2) / 12 + (y^2) / 4 = 1

def point_on_line (x : ℝ): Prop := 
  x = -2 * Real.sqrt 2

theorem fixed_point_through_l' : 
  ∀ (y_0 : ℝ), 
  (-3 < y_0) ∧ (y_0 < 3) → 
  (∃ (P : ℝ × ℝ), 
  P = (-4 * Real.sqrt 2 / 3, 0) ∧ 
  ∀ (x y : ℝ), 
  ellipse x y → 
  ellipse (sqrt 3, sqrt 3) ∧ 
  point_on_line (-2 * Real.sqrt 2)) :=
sorry

theorem find_ellipse_equation : 
  ellipse 3 1 ∧ 
  ellipse 3 (-1) ∧ 
  ellipse (sqrt 3) (sqrt 3) →
  ∃ (a b : ℝ), 
  a = 12 ∧ 
  b = 4 :=
sorry

end fixed_point_through_l_find_ellipse_equation_l292_292396


namespace career_preference_degrees_l292_292178

theorem career_preference_degrees (ratio_male_female : ℕ × ℕ)
                                   (male_prefers_career_ratio : ℚ)
                                   (female_prefers_career_ratio : ℚ)
                                   (circle_total_degrees : ℚ) :
  ratio_male_female = (2, 7) →
  male_prefers_career_ratio = 1/3 →
  female_prefers_career_ratio = 2/3 →
  circle_total_degrees = 360 →
  ((male_prefers_career_ratio * ratio_male_female.1 + female_prefers_career_ratio * ratio_male_female.2) / (ratio_male_female.1 + ratio_male_female.2) * circle_total_degrees ≈ 213.33) :=
by
  sorry

end career_preference_degrees_l292_292178


namespace area_of_rhombus_l292_292118

theorem area_of_rhombus (d1 d2 : ℝ) (h1 : d1 = 62) (h2 : d2 = 80) : (d1 * d2) / 2 = 2480 := by
  have h3 : d1 * d2 = 62 * 80 := by
    rw [h1, h2]
    exact mul_eq_mul_right _ _
  have h4 : (d1 * d2) / 2 = 4960 / 2 := by
    rw [h3]
  rw [div_eq_div_right (show (2:ℝ) ≠ 0 by norm_num)] at h4
  exact h4.symm

end area_of_rhombus_l292_292118


namespace number_of_female_fish_l292_292574

-- Defining the constants given in the problem
def total_fish : ℕ := 45
def fraction_male : ℚ := 2 / 3

-- The statement we aim to prove in Lean
theorem number_of_female_fish : 
  (total_fish : ℚ) * (1 - fraction_male) = 15 :=
by
  sorry

end number_of_female_fish_l292_292574


namespace sara_lucas_difference_l292_292453

-- Sara's numbers from 1 to 50
def sara_sum : ℕ := (list.range' 1 50).sum

-- Lucas's transformation: replace digit '3' with '2'
def transform_digit (n : ℕ) : ℕ :=
  read (n.repr.map (λ c, if c = '3' then '2' else c)).to_nat

-- Lucas's numbers from 1 to 50 after transformation
def lucas_sum : ℕ := ((list.range' 1 50).map transform_digit).sum

-- The difference between Sara's sum and Lucas's sum
def sum_difference : ℕ := sara_sum - lucas_sum

-- Theorem: proving the difference is 105
theorem sara_lucas_difference : sum_difference = 105 := by
  sorry

end sara_lucas_difference_l292_292453


namespace sequence_properties_l292_292767

-- Definitions based on the initial conditions
def A (x y : ℕ) (n : ℕ) : ℚ :=
  if n = 1 then (x^2 + y^2) / 2 else (A x y (n - 1) + H x y (n - 1)) / 2

def G (x y : ℕ) (n : ℕ) : ℚ :=
  if n = 1 then real.sqrt (x * y) else real.sqrt ((A x y (n - 1)) * (H x y (n - 1)))

def H (x y : ℕ) (n : ℕ) : ℚ :=
  if n = 1 then (2 * x * y) / (x + y) else 2 / ((1 / A x y (n - 1)) + (1 / H x y (n - 1)))

theorem sequence_properties (x y : ℕ) (hxy : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y) :
  (∀ n : ℕ, n > 0 → A x y n > A x y (n + 1)) ∧
  (∀ n : ℕ, n > 0 → G x y n = G x y (n + 1)) ∧
  (∀ n : ℕ, n > 0 → H x y n < H x y (n + 1)) :=
by
  sorry

end sequence_properties_l292_292767


namespace number_of_correct_propositions_is_one_l292_292143

-- Define the propositions as predicates.
def Prop1 (a b : Line) : Prop :=
  sufficient_but_not_necessary_condition (skew_lines a b) (not (intersect a b))

def Prop2 (a : Line) (α : Plane) (b : Line) : Prop :=
  necessary_but_not_sufficient_condition (parallel a α) (parallel a b ∧ b ∈ α)

def Prop3 (a : Line) (α : Plane) : Prop :=
  necessary_and_sufficient_condition (perpendicular a α) (∀ (b : Line), b ∈ α → perpendicular a b)

-- Define the main theorem.
theorem number_of_correct_propositions_is_one (a b : Line) (α : Plane) :
  (¬ Prop1 a b) ∧ (¬ Prop2 a α b) ∧ Prop3 a α → number_of_correct_propositions = 1 :=
by
  sorry

end number_of_correct_propositions_is_one_l292_292143


namespace true_discount_is_36_l292_292467

noncomputable def calc_true_discount (BD SD : ℝ) : ℝ := BD / (1 + BD / SD)

theorem true_discount_is_36 :
  let BD := 42
  let SD := 252
  calc_true_discount BD SD = 36 := 
by
  -- proof here
  sorry

end true_discount_is_36_l292_292467


namespace sum_even_integers_less_than_102_l292_292887

theorem sum_even_integers_less_than_102 :
  ∑ k in Finset.filter (λ x => (even x) ∧ (0 < x) ∧ (x < 102)) (Finset.range 102), k = 2550 := sorry

end sum_even_integers_less_than_102_l292_292887


namespace four_cells_different_colors_l292_292313

theorem four_cells_different_colors
  (n : ℕ)
  (h_n : n ≥ 2)
  (coloring : Fin n → Fin n → Fin (2 * n)) :
  ∃ (r1 r2 c1 c2 : Fin n),
    r1 ≠ r2 ∧ c1 ≠ c2 ∧
    (coloring r1 c1 ≠ coloring r1 c2) ∧
    (coloring r1 c1 ≠ coloring r2 c1) ∧
    (coloring r1 c2 ≠ coloring r2 c2) ∧
    (coloring r2 c1 ≠ coloring r2 c2) := 
sorry

end four_cells_different_colors_l292_292313


namespace total_capacity_l292_292038

variable (vans : Nat → Nat)
-- Conditions
def van_count : Nat := 6
def van_capacity_1 := 8000
def van_capacity_2 := 0.7 * van_capacity_1

-- Remaining 3 vans' capacity setup
def remaining_vans_capacity := 57600 - (2 * van_capacity_1 + van_capacity_2)

def van_capacity_3 : Nat := remaining_vans_capacity / 3

-- Question to be proved: Total capacity equals 57600 gallons
theorem total_capacity (h1 : vans 0 = van_capacity_1)
                      (h2 : vans 1 = van_capacity_1)
                      (h3 : vans 2 = van_capacity_2)
                      (h4 : vans 3 = van_capacity_3)
                      (h5 : vans 4 = van_capacity_3)
                      (h6 : vans 5 = van_capacity_3) :
    (vans 0 + vans 1 + vans 2 + vans 3 + vans 4 + vans 5) = 57600 := 
by
  sorry

end total_capacity_l292_292038


namespace solve_for_x_l292_292942

-- Define the list of heights in ascending order
def heights := [158, 165, 165, 167, 168, 169, x, 172, 173, 175]

-- Define the 60th percentile of the sample data
def percentile_60 (l : List ℕ) : ℕ := (l[5] + l[6]) / 2

-- Define x
variable (x : ℕ)

-- Given conditions:
-- The 60th percentile is 170
axiom percentile_condition : percentile_60 heights = 170

-- The proof statement: Solve for x given the conditions
theorem solve_for_x : x = 171 :=
by
  sorry

end solve_for_x_l292_292942


namespace total_students_in_school_l292_292389

def total_students (S : ℕ) : Prop :=
  (0.45 * S) + (0.23 * S) + (0.15 * S) = (S - 102)

theorem total_students_in_school : ∃ S : ℕ, total_students S ∧ S = 600 :=
by
  -- Maths proof would go here
  sorry

end total_students_in_school_l292_292389


namespace carla_blue_paint_cans_l292_292810

def ratio_blue_green := (4 : ℕ, 3 : ℕ)
def total_cans := 35
def blue_fraction (ratio : ℕ × ℕ) : ℚ := ratio.1 / (ratio.1 + ratio.2)
def blue_cans (fraction : ℚ) (total : ℕ) : ℚ := fraction * total

theorem carla_blue_paint_cans : blue_cans (blue_fraction ratio_blue_green) total_cans = 20 := by
  sorry

end carla_blue_paint_cans_l292_292810


namespace logan_score_l292_292087

-- Define the conditions as constants
def total_students : ℕ := 20 
def avg_score_first_19 : ℕ := 85
def avg_score_all_20 : ℕ := 86

-- The theorem to prove
theorem logan_score : 
  let total_score_19 := 19 * avg_score_first_19 in
  let total_score_20 := total_students * avg_score_all_20 in
  let logan_score := total_score_20 - total_score_19 in
  logan_score = 105 :=
by
  sorry

end logan_score_l292_292087


namespace AT_eq_TB_l292_292066

noncomputable def circle (radius : ℝ) (center : (ℝ × ℝ)) : set (ℝ × ℝ) := 
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

variable {k₁ k₂ : set (ℝ × ℝ)}
variable {A B C D T : (ℝ × ℝ)}
variable {l : set (ℝ × ℝ)}

-- Circle k₁ and line l intersect at A and B
variable (h₁ : A ∈ k₁) (h₂ : B ∈ k₁)
variable (h₃ : A ∈ l) (h₄ : B ∈ l)

-- Circle k₂ touches k₁ at C and line l at D
variable (h₅ : C ∈ k₁) (h₆ : C ∈ k₂)
variable (h₇ : D ∈ k₂) (h₈ : D ∈ l)

-- T is the second intersection of k₁ and line CD
variable (h₉ : T ∈ k₁) (h₁₀ : T ∈ {p | ∃ q ∈ (line_through_points C D), q = p})

theorem AT_eq_TB : dist A T = dist T B :=
by
  sorry

end AT_eq_TB_l292_292066


namespace length_of_x_l292_292508

constant (BD : ℝ) (ABD_angle_B : ℝ) (ABD_angle_D : ℝ) (ACD_angle_C : ℝ) (ACD_angle_A : ℝ)
constant x : ℝ

axiom BD_def : BD = 12
axiom ABD_angle_B_def : ABD_angle_B = 30 * (Real.pi / 180)  -- 30 degrees in radians
axiom ABD_angle_D_def : ABD_angle_D = 60 * (Real.pi / 180)  -- 60 degrees in radians
axiom ACD_angle_C_def : ACD_angle_C = 45 * (Real.pi / 180)  -- 45 degrees in radians
axiom ACD_angle_A_def : ACD_angle_A = 45 * (Real.pi / 180)  -- 45 degrees in radians

theorem length_of_x : x = 6 * Real.sqrt 6 :=
by
  sorry

end length_of_x_l292_292508


namespace find_area_triangle_boc_l292_292407

noncomputable def area_ΔBOC := 21

theorem find_area_triangle_boc (A B C K O : Type) 
  [NormedAddCommGroup A] [NormedAddCommGroup B] [NormedAddCommGroup C] [NormedAddCommGroup K] [NormedAddCommGroup O]
  (AC : ℝ) (AB : ℝ) (h1 : AC = 14) (h2 : AB = 6)
  (circle_centered_on_AC : Prop)
  (K_on_BC : Prop)
  (angle_BAK_eq_angle_ACB : Prop)
  (midpoint_O_AC : Prop)
  (angle_AKC_eq_90 : Prop)
  (area_ABC : Prop) : 
  area_ΔBOC = 21 := 
sorry

end find_area_triangle_boc_l292_292407


namespace fabric_needed_l292_292753

theorem fabric_needed (shirts_per_day k_per_day shirt_fabric p_fabric : ℕ) (days : ℕ) :
  shirts_per_day = 3 → k_per_day = 5 → shirt_fabric = 2 → p_fabric = 5 → days = 3 →
  (shirts_per_day * shirt_fabric + k_per_day * p_fabric) * days = 93 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end fabric_needed_l292_292753


namespace mean_value_point_unique_l292_292600

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x

theorem mean_value_point_unique : 
  ∃! x_0 ∈ set.Icc (-1 : ℝ) (1 : ℝ), f x_0 = (∫ x in (-1:ℝ)..(1:ℝ), f x) / 2 := 
by
  sorry

end mean_value_point_unique_l292_292600


namespace is_isosceles_triangle_l292_292351

variables (A B C : Type) [LinearOrder B] [Zero B] [Add B] 
variables (A_1 A_2 B_1 B_2 : A → B)
variables (angle : A → A → A → B)

def divides_three_equal_parts (a b c : A → B) :=
  a + c = 3 * b

-- Defining the actual proposition to be proved.
theorem is_isosceles_triangle (ABC : A)
  (A1_A2_eq_parts : divides_three_equal_parts ABC A_1 A_2)
  (B1_B2_eq_parts : divides_three_equal_parts ABC B_1 B_2)
  (angles_equal : angle A_1 B A_2 = angle B_1 A B_2) :
  (is_isosceles ABC) :=
sorry -- proof goes here

end is_isosceles_triangle_l292_292351


namespace solve_congruence_l292_292109

theorem solve_congruence (m : ℤ) : 13 * m ≡ 9 [MOD 47] → m ≡ 29 [MOD 47] :=
by
  sorry

end solve_congruence_l292_292109


namespace number_of_pupils_in_class_l292_292170

theorem number_of_pupils_in_class
(U V : ℕ) (increase : ℕ) (avg_increase : ℕ) (n : ℕ) 
(h1 : U = 85) (h2 : V = 45) (h3 : increase = U - V) (h4 : avg_increase = 1 / 2) (h5 : increase / avg_increase = n) :
n = 80 := by
sorry

end number_of_pupils_in_class_l292_292170


namespace friends_bill_is_12_7_l292_292167

-- Defining the variables and the conditions
def cost_of_taco : ℝ := 0.9
def your_bill_before_tax : ℝ := 7.80
def number_of_your_tacos : ℕ := 2
def number_of_your_enchiladas : ℕ := 3
def number_of_friends_tacos : ℕ := 3
def number_of_friends_enchiladas : ℕ := 5

-- Calculations based on conditions
def cost_of_your_tacos : ℝ := number_of_your_tacos * cost_of_taco
def cost_of_your_enchiladas : ℝ := your_bill_before_tax - cost_of_your_tacos
def cost_of_one_enchilada : ℝ := cost_of_your_enchiladas / number_of_your_enchiladas
def cost_of_friends_tacos : ℝ := number_of_friends_tacos * cost_of_taco
def cost_of_friends_enchiladas : ℝ := number_of_friends_enchiladas * cost_of_one_enchilada
def friends_bill_before_tax : ℝ := cost_of_friends_tacos + cost_of_friends_enchiladas

-- Theorem stating the friend's bill before tax
theorem friends_bill_is_12_7 : friends_bill_before_tax = 12.7 := by
  sorry

end friends_bill_is_12_7_l292_292167


namespace no_3_digit_odd_digit_sum_30_l292_292696

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + (n / 10 % 10) + (n % 10)

theorem no_3_digit_odd_digit_sum_30 :
  ¬ ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ digit_sum n = 30 ∧ n % 2 = 1 :=
begin
  sorry
end

end no_3_digit_odd_digit_sum_30_l292_292696


namespace arithmetic_progression_sum_l292_292630

theorem arithmetic_progression_sum : 
  (∑ p in Finset.range 10, 
      let a_30 := p + 1 + 29 * (3 * (p + 1) + 1)
      let S_30 := 15 * ((p + 1) + a_30)
      in S_30) = 77675 :=
by
  sorry

end arithmetic_progression_sum_l292_292630


namespace problem_proof_l292_292495

-- Definitions of the events based on given conditions
def event_A (x y : ℕ) : Prop := x + y = 7
def event_B (x y : ℕ) : Prop := x % 2 = 1 ∧ y % 2 = 1
def event_C (x y : ℕ) : Prop := x > 3

-- Records of dice outcomes
def outcomes : List (ℕ × ℕ) := [
  (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
  (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
  (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
  (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
  (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6),
  (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)
]

-- Helper function to calculate probability
def probability (event : ℕ → ℕ → Prop) : ℚ :=
  (outcomes.filter (λ x, event x.fst x.snd)).length / outcomes.length

-- Proof statement
theorem problem_proof :
  (∀ x y, event_A x y → ¬ event_B x y) ∧
  probability event_A * probability (λ x y, x > 3) = probability (λ x y, event_A x y ∧ event_C x y) :=
by
  sorry

end problem_proof_l292_292495


namespace problem_solution_l292_292376

def satisfies_graph_condition (m : ℤ) : Prop :=
  m < 7

def satisfies_fractional_equation (m : ℤ) : Prop :=
  m > 2 ∧ m ≠ 4

def valid_values : set ℤ := {m | satisfies_graph_condition m ∧ satisfies_fractional_equation m}

def sum_valid_values (s : set ℤ) : ℤ :=
  s.to_finset.sum id

theorem problem_solution :
  sum_valid_values valid_values = 14 :=
by
  sorry

end problem_solution_l292_292376


namespace wendy_facial_products_l292_292155

def total_time (P : ℕ) : ℕ :=
  5 * (P - 1) + 30

theorem wendy_facial_products :
  (total_time 6 = 55) :=
by
  sorry

end wendy_facial_products_l292_292155


namespace find_sum_l292_292935

theorem find_sum {x y : ℝ} (h1 : x = 13.0) (h2 : x + y = 24) : 7 * x + 5 * y = 146 := 
by
  sorry

end find_sum_l292_292935


namespace kirsty_initial_models_l292_292417

theorem kirsty_initial_models 
  (x : ℕ)
  (initial_price : ℝ)
  (increased_price : ℝ)
  (models_bought : ℕ)
  (h_initial_price : initial_price = 0.45)
  (h_increased_price : increased_price = 0.5)
  (h_models_bought : models_bought = 27) 
  (h_total_saved : x * initial_price = models_bought * increased_price) :
  x = 30 :=
by 
  sorry

end kirsty_initial_models_l292_292417


namespace polygon_deformable_to_triangle_l292_292547

-- Definitions based on the conditions extracted
def planar_polygon (n : ℕ) : Type := 
{ rods : fin n → ℝ // ∀ i, rods i > 0 }

-- Definition of deformable to a triangle
def deformable_to_triangle (P : planar_polygon n) : Prop := 
∃ (a b c : ℝ), a + b > c ∧ a + c > b ∧ b + c > a

-- The main theorem statement
theorem polygon_deformable_to_triangle (n : ℕ) (P : planar_polygon n) : n > 4 → deformable_to_triangle P :=
by
  intro h
  sorry

end polygon_deformable_to_triangle_l292_292547


namespace sum_of_digits_of_palindrome_l292_292201

-- Definition of a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

-- The problem statement in Lean
theorem sum_of_digits_of_palindrome {x : ℕ} : 
  (100 ≤ x ∧ x ≤ 999) ∧ is_palindrome x ∧ is_palindrome (x + 50) ∧ (1000 ≤ x + 50 ∧ x + 50 < 10000) → 
  (x.digits 10).sum = 8 :=
begin
  sorry
end

end sum_of_digits_of_palindrome_l292_292201


namespace quadrilateral_area_ratio_l292_292822

theorem quadrilateral_area_ratio (r : ℝ) (EFG_angle : ℝ) (FEG_angle : ℝ) (circle_area : ℝ) : 
  EFG_angle = 30 → FEG_angle = 60 → circle_area = π * (r ^ 2) →
  let EFG_area := (1 / 2) * r * (r * (sqrt 3)) in
  let EHG_area := (1 / 2) * (r * sqrt 2) ^ 2 in
  let EFGH_area := EFG_area + EHG_area in
  (EFGH_area / circle_area) = ((sqrt 3 + 2) / (2 * π)) → 
  (let a := 2 in let b := 3 in let c := 2 in a + b + c = 7) := 
sorry

end quadrilateral_area_ratio_l292_292822


namespace price_per_pound_second_coffee_l292_292959

theorem price_per_pound_second_coffee
  (price_first : ℝ) (total_mix_weight : ℝ) (sell_price_per_pound : ℝ) (each_kind_weight : ℝ) 
  (total_sell_price : ℝ) (total_first_cost : ℝ) (total_second_cost : ℝ) (price_second : ℝ) :
  price_first = 2.15 →
  total_mix_weight = 18 →
  sell_price_per_pound = 2.30 →
  each_kind_weight = 9 →
  total_sell_price = total_mix_weight * sell_price_per_pound →
  total_first_cost = each_kind_weight * price_first →
  total_second_cost = total_sell_price - total_first_cost →
  price_second = total_second_cost / each_kind_weight →
  price_second = 2.45 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end price_per_pound_second_coffee_l292_292959


namespace a_plus_b_l292_292637

open Complex

theorem a_plus_b (a b : ℝ) (h : (a - I) * I = -b + 2 * I) : a + b = 1 := by
  sorry

end a_plus_b_l292_292637


namespace min_distance_correct_l292_292672

noncomputable def trajectory_condition : Prop :=
  ∀ x y : ℝ, x ≤ 1 → y^2 = -4 * (x - 1)

noncomputable def minimum_distance (m : ℝ) : ℝ :=
  if m ≥ -1 then |m - 1|
  else 2 * real.sqrt (-m)

theorem min_distance_correct (m : ℝ) :
  ∀ x y : ℝ, trajectory_condition x y → 
  (x ≤ 1 ∧ y^2 = -4 * (x - 1)) → 
  let pq_distance := real.sqrt ((x - m)^2 + y^2) in
  pq_distance ≥ minimum_distance m := 
sorry

end min_distance_correct_l292_292672


namespace equation_of_hyperbola_value_of_k_l292_292469

-- Given conditions as definitions
def center_at_origin : Prop := ∃ (x y : ℝ), x = 0 ∧ y = 0

def right_focus (c : ℝ) : Prop := F = (2 * sqrt 3 / 3, 0)

def asymptote_equations (b a : ℝ) : Prop := b / a = sqrt 3

def line_intersection_with_hyperbola (k a b : ℝ) : Prop :=
  ∃ (x₁ x₂ y₁ y₂ : ℝ), y₁ = k * x₁ + 1 ∧ y₂ = k * x₂ + 1 ∧ 3 * (x₁ ^ 2) - (y₁ ^ 2) = 1 ∧ 3 * (x₂ ^ 2) - (y₂ ^ 2) = 1

def diameter_circle_passing_origin (x₁ x₂ y₁ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

-- Statements to prove
theorem equation_of_hyperbola :
  (center_at_origin ∧ right_focus c ∧ asymptote_equations b a) →
  (3 * x ^ 2 - y ^ 2 = 1) := 
sorry

theorem value_of_k :
  (asymptote_equations b a ∧ line_intersection_with_hyperbola k a b) →
  (diameter_circle_passing_origin x₁ x₂ y₁ y₂) →
  k = 1 ∨ k = -1 := 
sorry

end equation_of_hyperbola_value_of_k_l292_292469


namespace cannot_distribute_44_coins_diff_10_pockets_l292_292592

/-- Petya cannot distribute 44 coins into 10 pockets so that the number of coins in each pocket is different. -/
theorem cannot_distribute_44_coins_diff_10_pockets :
  ¬(∃ a : Fin 10 → ℕ, (∀ i j : Fin 10, i ≠ j → a i ≠ a j) ∧ (∑ i, a i = 44)) :=
by
  sorry

end cannot_distribute_44_coins_diff_10_pockets_l292_292592


namespace man_work_days_l292_292196

theorem man_work_days (M : ℕ) (h1 : (1 : ℝ)/M + (1 : ℝ)/10 = 1/5) : M = 10 :=
sorry

end man_work_days_l292_292196


namespace trig_identity_l292_292325

theorem trig_identity (α : ℝ) (h : sin α + 2 * cos α = 0) : 2 * sin α * cos α - cos α ^ 2 = -1 :=
sorry

end trig_identity_l292_292325


namespace complex_combination_l292_292874

open Complex

def a : ℂ := 2 - I
def b : ℂ := -1 + I

theorem complex_combination : 2 * a + 3 * b = 1 + I :=
by
  -- Proof goes here
  sorry

end complex_combination_l292_292874


namespace cos_C_in_right_triangle_l292_292720

theorem cos_C_in_right_triangle
  (A B C : Type)
  [has_angle A]
  [has_angle B]
  [has_angle C]
  (hA : has_angle.angle A = 90)
  (hT : has_tangent.tangent C = 4) :
  has_cosine.cosine C = sqrt 17 / 17 := 
by sorry

end cos_C_in_right_triangle_l292_292720


namespace Rafael_worked_10_hours_on_Monday_l292_292824

noncomputable def M : ℕ := 10

theorem Rafael_worked_10_hours_on_Monday
    (hours_on_Tuesday : ℕ)
    (hours_left_in_week : ℕ)
    (total_earnings : ℕ)
    (hourly_rate : ℕ)
    (total_hours_worked : ℕ)
    (h1 : hours_on_Tuesday = 8)
    (h2 : hours_left_in_week = 20)
    (h3 : total_earnings = 760)
    (h4 : hourly_rate = 20)
    (h5 : total_hours_worked = total_earnings / hourly_rate)
    (h6 : total_hours_worked = M + hours_on_Tuesday + hours_left_in_week) :
  M = 10 :=
begin
  sorry
end

end Rafael_worked_10_hours_on_Monday_l292_292824


namespace number_of_elements_in_intersection_l292_292068

def f (x : ℝ) : ℝ := log (3 - x)
def A : Set ℝ := { x | 3 - x > 0 }
def N_star : Set ℕ := { n | n > 0 }

theorem number_of_elements_in_intersection : 
  (A ∩ {x : ℝ | x ∈ (N_star : Set ℝ)}).to_finset.card = 2 :=
by
  sorry

end number_of_elements_in_intersection_l292_292068


namespace num_impossible_events_l292_292982

def water_boils_at_90C := false
def iron_melts_at_room_temp := false
def coin_flip_results_heads := true
def abs_value_not_less_than_zero := true

theorem num_impossible_events :
  water_boils_at_90C = false ∧ iron_melts_at_room_temp = false ∧ 
  coin_flip_results_heads = true ∧ abs_value_not_less_than_zero = true →
  (if ¬water_boils_at_90C then 1 else 0) + (if ¬iron_melts_at_room_temp then 1 else 0) +
  (if ¬coin_flip_results_heads then 1 else 0) + (if ¬abs_value_not_less_than_zero then 1 else 0) = 2
:= by
  intro h
  have : 
    water_boils_at_90C = false ∧ iron_melts_at_room_temp = false ∧ 
    coin_flip_results_heads = true ∧ abs_value_not_less_than_zero = true := h
  sorry

end num_impossible_events_l292_292982


namespace no_sum_of_19_l292_292503

theorem no_sum_of_19 (a b c d : ℕ) (ha : 1 ≤ a ∧ a ≤ 6) (hb : 1 ≤ b ∧ b ≤ 6) (hc : 1 ≤ c ∧ c ≤ 6) (hd : 1 ≤ d ∧ d ≤ 6)
  (hprod : a * b * c * d = 180) : a + b + c + d ≠ 19 :=
sorry

end no_sum_of_19_l292_292503


namespace count_valid_propositions_is_zero_l292_292677

theorem count_valid_propositions_is_zero :
  (∀ (a b : ℝ), (a > b → a^2 > b^2) = false) ∧
  (∀ (a b : ℝ), (a^2 > b^2 → a > b) = false) ∧
  (∀ (a b : ℝ), (a > b → b / a < 1) = false) ∧
  (∀ (a b : ℝ), (a > b → 1 / a < 1 / b) = false) :=
by
  sorry

end count_valid_propositions_is_zero_l292_292677


namespace rank_A_second_l292_292215

-- We define the conditions provided in the problem
variables (a b c : ℕ) -- defining the scores of A, B, and C as natural numbers

-- Conditions given
def A_said (a b c : ℕ) := b < a ∧ c < a
def B_said (b c : ℕ) := b > c
def C_said (a b c : ℕ) := a > c ∧ b > c

-- Conditions as hypotheses
variable (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c) -- the scores are different
variable (h2 : A_said a b c ∨ B_said b c ∨ C_said a b c) -- exactly one of the statements is incorrect

-- The theorem to prove
theorem rank_A_second : ∃ (rankA : ℕ), rankA = 2 := by
  sorry

end rank_A_second_l292_292215


namespace matrix_sum_lower_bound_l292_292393

theorem matrix_sum_lower_bound (M : ℕ) (A : Matrix (Fin M) (Fin M) ℕ)
  (h : ∀ i j, A i j = 0 → (∑ k, A i k ≥ M) ∧ (∑ k, A k j ≥ M)) : 
  ∑ i j, A i j ≥ M^2 / 2 := 
sorry

end matrix_sum_lower_bound_l292_292393


namespace satisfies_conditions_l292_292815

def in_second_quadrant (x : ℝ) : Prop :=
  x < 0

def is_increasing_in_second_quadrant (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, in_second_quadrant x → in_second_quadrant y → x < y → f x < f y

def passes_through_point (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  f p.1 = p.2

theorem satisfies_conditions (f : ℝ → ℝ) :
  passes_through_point f (-2, 1) →
  is_increasing_in_second_quadrant f →
  (f = λ x, x + 3 ∨ f = λ x, -2 / x ∨ f = λ x, -x^2 + 5) := 
sorry

end satisfies_conditions_l292_292815


namespace star_value_l292_292631

noncomputable def star (a b : ℝ) (h : a ≠ b) : ℝ := (a + b) / (a - b)

theorem star_value :
  star (star 3 5 (by norm_num)) (star 1 4 (by norm_num)) (by linarith) = 17 / 7 :=
by
  sorry

end star_value_l292_292631


namespace midpoints_sum_is_15_l292_292490

variable (a b : ℝ)

-- Sum of x-coordinates equals 15
def vertices_sum_condition (a b : ℝ) : Prop := a + b + (a + 5) = 15

-- Definition of the x-coordinates of the midpoints
def midpoints_sum (a b : ℝ) : ℝ :=
  ((a + b) / 2) + ((a + (a + 5)) / 2) + ((b + (a + 5)) / 2)

-- Theorem stating the required sum of the midpoints' x-coordinates is 15
theorem midpoints_sum_is_15 (a b : ℝ) (h : vertices_sum_condition a b) :
  midpoints_sum a b = 15 :=
  sorry

end midpoints_sum_is_15_l292_292490


namespace average_hours_l292_292594

def hours_studied (week1 week2 week3 week4 week5 week6 week7 : ℕ) : ℕ :=
  week1 + week2 + week3 + week4 + week5 + week6 + week7

theorem average_hours (x : ℕ)
  (h1 : hours_studied 8 10 9 11 10 7 x / 7 = 9) :
  x = 8 :=
by
  sorry

end average_hours_l292_292594


namespace area_of_square_plot_l292_292164

-- Defining the given conditions and question in Lean 4
theorem area_of_square_plot 
  (cost_per_foot : ℕ := 58)
  (total_cost : ℕ := 2784) :
  ∃ (s : ℕ), (4 * s * cost_per_foot = total_cost) ∧ (s * s = 144) :=
by
  sorry

end area_of_square_plot_l292_292164


namespace ariel_fish_l292_292577

theorem ariel_fish (total_fish : ℕ) (male_fraction female_fraction : ℚ) (h1 : total_fish = 45) (h2 : male_fraction = 2 / 3) (h3 : female_fraction = 1 - male_fraction) : total_fish * female_fraction = 15 :=
by
  sorry

end ariel_fish_l292_292577


namespace find_b_l292_292011

noncomputable def A : ℝ := 30 * Real.pi / 180  -- converting 30 degrees to radians
def a : ℝ := 1
def c : ℝ := Real.sqrt 3

theorem find_b : ∃ b : ℝ, (a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) ∧ (b = 1 ∨ b = 2) := 
  by
  sorry

end find_b_l292_292011


namespace prob_one_first_class_prob_any_first_class_prob_least_3_first_class_l292_292781

def prob_A_first_class : ℝ := 0.7
def prob_B_first_class : ℝ := 0.6
def prob_C_first_class : ℝ := 0.8

def n_B_parts : ℕ := 100 -- Arbitrary constant since probabilities are relative
def n_C_parts : ℕ := n_B_parts
def n_A_parts : ℕ := 2 * n_B_parts

noncomputable def prob_at_least_one_first_class :=
1 - (1 - prob_A_first_class) * (1 - prob_B_first_class) * (1 - prob_C_first_class)

noncomputable def prob_random_first_class :=
(n_A_parts * prob_A_first_class + n_B_parts * prob_B_first_class + n_C_parts * prob_C_first_class) / 
(n_A_parts + n_B_parts + n_C_parts)

noncomputable def prob_at_least_3_first_class (total_parts : ℕ) (n : ℕ) :=
/- The calculation of this probability involves combinatorial calculations.
This function approximates the probability based on the given conditions -/
sorry

theorem prob_one_first_class : prob_at_least_one_first_class = 0.976 := 
by sorry

theorem prob_any_first_class : prob_random_first_class = 0.7 := 
by sorry

theorem prob_least_3_first_class : prob_at_least_3_first_class (n_A_parts + n_B_parts + n_C_parts) 4 = 0.6517 :=
by sorry

end prob_one_first_class_prob_any_first_class_prob_least_3_first_class_l292_292781


namespace distribution_plans_l292_292990

-- Definitions and conditions
def arrange_teachers (teachers schools : ℕ) : ℕ := teachers * schools
def ways_one_per_school : ℕ := nat.factorial 6 / (nat.factorial (6-3))
def ways_one_with_two_others : ℕ := nat.choose 3 2 * (nat.factorial 6 / (nat.factorial (6-2)))

-- Problem statement
theorem distribution_plans : arrange_teachers 3 6 = 210 :=
by
  have one_per_school := ways_one_per_school
  have one_with_two_others := ways_one_with_two_others
  let total_ways := one_per_school + one_with_two_others
  exact congr_arg Nat.succ (Nat.pred_inj (by linarith))

end distribution_plans_l292_292990


namespace expected_value_of_flipped_coins_is_45_point_5_cents_l292_292545

theorem expected_value_of_flipped_coins_is_45_point_5_cents:
  let coin_values := [1, 5, 10, 25, 50] in
  let coin_prob := 1 / 2 in
  let expected_value := (coin_prob * (List.sum coin_values : ℝ)) in
  expected_value = 45.5 := 
by 
  sorry

end expected_value_of_flipped_coins_is_45_point_5_cents_l292_292545


namespace sum_of_squares_formula_l292_292868

theorem sum_of_squares_formula (n : ℕ) (h : 0 < n) : 
  (∑ i in Finset.range n.succ, i^2) = n * (n + 1) * (2 * n + 1) / 6 := by
sorry

end sum_of_squares_formula_l292_292868


namespace no_angle_sat_sin_cos_eq_sin_40_l292_292261

open Real

theorem no_angle_sat_sin_cos_eq_sin_40 :
  ¬∃ α : ℝ, sin α * cos α = sin (40 * π / 180) := 
by 
  sorry

end no_angle_sat_sin_cos_eq_sin_40_l292_292261


namespace adjusted_roster_ways_l292_292536

theorem adjusted_roster_ways : 
  ∃ (n : ℕ), 
    (n = 24) ∧
    ∃ (arrangement : list ℕ → ℕ → bool),
      (∀ (l : list ℕ) (d : ℕ),
        l.length = 5 ∧
        (arrangement l d ↔ 
          (d ≠ l.head ∧ d ≠ l.last ∧ 
           ∀ (i : ℕ) (hi : i < 5 - 2), d ≠ nth_le l (i+1) hi)
        )
      ) 
    :=
begin
  sorry
end

end adjusted_roster_ways_l292_292536


namespace remainder_of_power_division_l292_292179

-- Define the main entities
def power : ℕ := 3
def exponent : ℕ := 19
def divisor : ℕ := 10

-- Define the proof problem
theorem remainder_of_power_division :
  (power ^ exponent) % divisor = 7 := 
  by 
    sorry

end remainder_of_power_division_l292_292179


namespace multinomial_theorem_l292_292278

open Nat

noncomputable def multinomial_coefficient (n : ℕ) (ks : List ℕ) : ℕ := 
  n.factorial / List.prod (List.map factorial ks)

theorem multinomial_theorem (m n : ℕ) (xs : Fin m → ℝ) :
  (Finset.sum (Finset.univ : Finset (Fin m)) fun i => (xs i)) ^ n =
  Finset.sum (Finset.univ : Finset { ks : Fin m → ℕ // (Finset.sum Finset.univ fun k => (ks k) = n) }) 
    fun k => (multinomial_coefficient n (List.ofFn (fun i => (k.val i)))) * 
              (Finset.prod (Finset.univ : Finset (Fin m)) fun i => (xs i) ^ (k.val i)) := 
sorry

end multinomial_theorem_l292_292278


namespace palindrome_percentage_contains_7_l292_292961

/-- A palindrome is an integer that reads the same forward and backward. -/
def is_palindrome (n : ℕ) : Prop :=
  let s := toString n
  s = s.reverse

/-- The percentage of palindromes between 1000 and 2000 that contain at least one digit 7 is 10%. -/
theorem palindrome_percentage_contains_7 : 
  (1000 ≤ n ∧ n < 2000 ∧ is_palindrome n) → 
  (10 * (count (λ n, 1000 ≤ n ∧ n < 2000 ∧ is_palindrome n ∧ '7' ∈ toString n) (1000, 2000)) / (count is_palindrome (1000, 2000))) = 10 := 
sorry

end palindrome_percentage_contains_7_l292_292961


namespace samantha_birth_year_l292_292120

theorem samantha_birth_year :
  ∀ (first_amc : ℕ) (amc9_year : ℕ) (samantha_age_in_amc9 : ℕ),
  (first_amc = 1983) →
  (amc9_year = first_amc + 8) →
  (samantha_age_in_amc9 = 13) →
  (amc9_year - samantha_age_in_amc9 = 1978) :=
by
  intros first_amc amc9_year samantha_age_in_amc9 h1 h2 h3
  sorry

end samantha_birth_year_l292_292120


namespace no_nine_diagonals_intersect_single_point_l292_292387

theorem no_nine_diagonals_intersect_single_point (P : Type) [Polygon P] (n : ℕ) (h : n = 25) :
  ¬∃ p : Point P, ∃ (diagonals : Finset (Diagonal P)),
    (diagonals.card = 9) ∧ (∀ d ∈ diagonals, p ∈ d) := sorry

end no_nine_diagonals_intersect_single_point_l292_292387


namespace distinct_xy_values_l292_292834

theorem distinct_xy_values : ∃ (xy_values : Finset ℕ), 
  (∀ (x y : ℕ), (0 < x ∧ 0 < y) → (1 / Real.sqrt x + 1 / Real.sqrt y = 1 / Real.sqrt 20) → (xy_values = {8100, 6400})) ∧
  (xy_values.card = 2) :=
by
  sorry

end distinct_xy_values_l292_292834


namespace lattice_points_count_l292_292697

-- Defining the equation as a predicate
def on_graph (x y : ℤ) : Prop := x^2 - y^2 = 60

-- List of valid integer pairs
def lattice_points : List (ℤ × ℤ) := [(16, -14), (8, -2), (-16, 14), (-8, 2)]

theorem lattice_points_count : 
  list.countp (λ (xy : ℤ × ℤ), on_graph xy.1 xy.2) lattice_points = 4 := sorry

end lattice_points_count_l292_292697


namespace car_owners_without_motorcycles_l292_292014

theorem car_owners_without_motorcycles 
  (total_adults : ℕ) (car_owners : ℕ) (motorcycle_owners : ℕ) (bicycle_owners : ℕ) (total_own_vehicle : ℕ)
  (h1 : total_adults = 400) (h2 : car_owners = 350) (h3 : motorcycle_owners = 60) (h4 : bicycle_owners = 30)
  (h5 : total_own_vehicle = total_adults)
  : (car_owners - 10 = 340) :=
by
  sorry

end car_owners_without_motorcycles_l292_292014


namespace production_volume_l292_292943

/-- 
A certain school's factory produces 200 units of a certain product this year.
It is planned to increase the production volume by the same percentage \( x \)
over the next two years such that the total production volume over three years is 1400 units.
The goal is to prove that the correct equation for this scenario is:
200 + 200 * (1 + x) + 200 * (1 + x)^2 = 1400.
-/
theorem production_volume (x : ℝ) :
    200 + 200 * (1 + x) + 200 * (1 + x)^2 = 1400 := 
sorry

end production_volume_l292_292943


namespace area_of_yellow_stripe_l292_292187

-- Definitions and given conditions
def diameter : ℝ := 40
def height : ℝ := 100
def stripe_width : ℝ := 4
def revolutions : ℝ := 3

-- Derived quantities
def circumference : ℝ := Real.pi * diameter
def length_of_stripe : ℝ := revolutions * circumference
def area_of_stripe : ℝ := stripe_width * length_of_stripe

-- Statement to prove
theorem area_of_yellow_stripe :
  area_of_stripe = 480 * Real.pi :=
by
  sorry

end area_of_yellow_stripe_l292_292187


namespace SamaraSpentOnDetailing_l292_292216

def costSamara (D : ℝ) : ℝ := 25 + 467 + D
def costAlberto : ℝ := 2457
def difference : ℝ := 1886

theorem SamaraSpentOnDetailing : 
  ∃ (D : ℝ), costAlberto = costSamara D + difference ∧ D = 79 := 
sorry

end SamaraSpentOnDetailing_l292_292216


namespace number_of_cookies_l292_292074

def total_cake := 22
def total_chocolate := 16
def total_groceries := 42

theorem number_of_cookies :
  ∃ C : ℕ, total_groceries = total_cake + total_chocolate + C ∧ C = 4 := 
by
  sorry

end number_of_cookies_l292_292074


namespace sum_of_slope_and_y_intercept_l292_292447

def slope (C D : ℝ × ℝ) : ℝ :=
  (D.2 - C.2) / (D.1 - C.1)

def y_intercept (C D : ℝ × ℝ) : ℝ :=
  let m := slope C D in
  C.2 - m * C.1

def sum_slope_y_intercept (C D : ℝ × ℝ) : ℝ :=
  let m := slope C D in
  let b := y_intercept C D in
  m + b

theorem sum_of_slope_and_y_intercept {C D : ℝ × ℝ}
  (hC : C = (-3, 5)) (hD : D = (4, -5)) :
  sum_slope_y_intercept C D = -5 / 7 :=
by {
  sorry
}

end sum_of_slope_and_y_intercept_l292_292447


namespace perimeter_of_billboard_l292_292203
noncomputable def perimeter_billboard : ℝ :=
  let width := 8
  let area := 104
  let length := area / width
  let perimeter := 2 * (length + width)
  perimeter

theorem perimeter_of_billboard (width area : ℝ) (P : width = 8 ∧ area = 104) :
    perimeter_billboard = 42 :=
by
  sorry

end perimeter_of_billboard_l292_292203


namespace cube_in_regular_quadrilateral_pyramid_l292_292965

noncomputable def edge_length_of_pyramid : ℝ := sorry
noncomputable def surface_area_of_cube (a : ℝ) : ℝ := 6 * (a * real.sqrt 2 / 4) ^ 2
noncomputable def volume_of_cube (a : ℝ) : ℝ := (a * real.sqrt 2 / 4) ^ 3

theorem cube_in_regular_quadrilateral_pyramid (a : ℝ) 
    (h : edge_length_of_pyramid = a) :
    surface_area_of_cube a = 3 * a ^ 2 / 4 ∧ volume_of_cube a = a ^ 3 * real.sqrt 2 / 32 := 
sorry

end cube_in_regular_quadrilateral_pyramid_l292_292965


namespace all_lamps_can_be_turned_on_l292_292023

-- Define the initial condition where only the lamp at (0, 0) is lit
def initial_lamp_state : ℤ × ℤ → Bool
  | (0, 0) => true
  | _ => false

-- Define the condition for turning on a new lamp
def can_be_turned_on (x y : ℤ) (previously_lit : ℤ × ℤ → Bool) : Bool :=
  ∃ (a b : ℤ), abs ((x - a)*(x - a) + (y - b)*(y - b)) = 2005^2 ∧ previously_lit (a, b)

theorem all_lamps_can_be_turned_on :
  ∃ (turn_on_time : ℤ × ℤ → ℕ), 
  (turn_on_time (0, 0) = 0) ∧
  (∀ (x y : ℤ), ∃ t : ℕ, turn_on_time (x, y) = t ∧
  (∀ t' < t, can_be_turned_on x y (λ p, turn_on_time p ≤ t'))) :=
sorry

end all_lamps_can_be_turned_on_l292_292023


namespace soda_cost_original_l292_292414

theorem soda_cost_original 
  (x : ℚ) -- note: x in rational numbers to capture fractional cost accurately
  (h1 : 3 * (0.90 * x) = 6) :
  x = 20 / 9 :=
by
  sorry

end soda_cost_original_l292_292414


namespace sequence_term_a_1000_eq_2340_l292_292383

theorem sequence_term_a_1000_eq_2340
  (a : ℕ → ℤ)
  (h1 : a 1 = 2007)
  (h2 : a 2 = 2008)
  (h_rec : ∀ n : ℕ, 1 ≤ n → a n + a (n + 1) + a (n + 2) = n) :
  a 1000 = 2340 :=
sorry

end sequence_term_a_1000_eq_2340_l292_292383


namespace area_of_centroid_curve_l292_292073

theorem area_of_centroid_curve {A B C : Type*} [metric_space A] [metric_space B] [metric_space C] 
  (circle : metric.sphere (0 : real × real) 12)
  (C : real × real)
  (hC : C ∈ circle)
  (A : real × real := (-12, 0))
  (B : real × real := (12, 0)) :
  let G := ((C.1 / 3), (C.2 / 3)) in
  let G_locus := metric.sphere (0 : real × real) 4 in
  measure_theory.measure.area G_locus = 50 :=
begin
  sorry
end

end area_of_centroid_curve_l292_292073


namespace initial_money_l292_292506

/-
We had $3500 left after spending 30% of our money on clothing, 
25% on electronics, and saving 15% in a bank account. 
How much money (X) did we start with before shopping and saving?
-/

theorem initial_money (M : ℝ) 
  (h_clothing : 0.30 * M ≠ 0) 
  (h_electronics : 0.25 * M ≠ 0) 
  (h_savings : 0.15 * M ≠ 0) 
  (remaining_money : 0.30 * M = 3500) : 
  M = 11666.67 := 
sorry

end initial_money_l292_292506


namespace mike_taller_than_mark_l292_292784

def height_mark_feet : ℕ := 5
def height_mark_inches : ℕ := 3
def height_mike_feet : ℕ := 6
def height_mike_inches : ℕ := 1
def feet_to_inches : ℕ := 12

-- Calculate heights in inches.
def height_mark_total_inches : ℕ := height_mark_feet * feet_to_inches + height_mark_inches
def height_mike_total_inches : ℕ := height_mike_feet * feet_to_inches + height_mike_inches

-- Prove the height difference.
theorem mike_taller_than_mark : height_mike_total_inches - height_mark_total_inches = 10 :=
by
  sorry

end mike_taller_than_mark_l292_292784


namespace num_valid_sequences_l292_292362

-- Define what it means for a digit to have a specific parity.
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Define the problem conditions
def valid_sequence (xs : Fin 8 → ℕ) : Prop :=
  ∀ i : Fin 7, (is_even (xs i.val) ↔ is_odd (xs (i.val + 1))) ∨ (is_odd (xs i.val) ↔ is_even (xs (i.val + 1)))

-- The number of valid sequences
theorem num_valid_sequences : ∃ n : ℕ, n = 781250 ∧ ∀ xs : Fin 8 → ℕ, valid_sequence xs → ∃ xs' : Fin 8 → ℕ, valid_sequence xs' :=
begin
  sorry
end

end num_valid_sequences_l292_292362


namespace units_digit_of_pow_sum_is_correct_l292_292907

theorem units_digit_of_pow_sum_is_correct : 
  (24^3 + 42^3) % 10 = 2 := by
  -- Start the proof block
  sorry

end units_digit_of_pow_sum_is_correct_l292_292907


namespace Sophie_Spends_72_80_l292_292831

noncomputable def SophieTotalCost : ℝ :=
  let cupcakesCost := 5 * 2
  let doughnutsCost := 6 * 1
  let applePieCost := 4 * 2
  let cookiesCost := 15 * 0.60
  let chocolateBarsCost := 8 * 1.50
  let sodaCost := 12 * 1.20
  let gumCost := 3 * 0.80
  let chipsCost := 10 * 1.10
  cupcakesCost + doughnutsCost + applePieCost + cookiesCost + chocolateBarsCost + sodaCost + gumCost + chipsCost

theorem Sophie_Spends_72_80 : SophieTotalCost = 72.80 :=
by
  sorry

end Sophie_Spends_72_80_l292_292831


namespace semi_integer_tiling_divisibility_l292_292102

theorem semi_integer_tiling_divisibility
  (d : ℕ)
  (B : fin d → ℤ)
  (b : fin d → ℤ)
  (h_tiling : ∀ i : fin d, ∃ j : fin d, b i ∣ B j)
  (h_semi_integer : ∀ (i j k : fin d), ∃ (n : ℤ), B k = n * b i) : 
  (∀ i : fin d, ∃ j : fin d, b i ∣ B j) :=
sorry

end semi_integer_tiling_divisibility_l292_292102


namespace last_guard_hours_l292_292960

theorem last_guard_hours (total_shift : ℕ) (first_guard_hours : ℕ) 
                         (second_guard_hours : ℕ) (third_guard_hours : ℕ) : 
                         first_guard_hours = 3 → 
                         second_guard_hours = 2 → 
                         third_guard_hours = 2 →
                         total_shift = 9 → 
                         ∃ last_guard_hours : ℕ, last_guard_hours = 2 := 
by 
  assume h1 : first_guard_hours = 3
  assume h2 : second_guard_hours = 2
  assume h3 : third_guard_hours = 2
  assume h4 : total_shift = 9
  let used_hours := first_guard_hours + second_guard_hours + third_guard_hours
  have : used_hours = 7 := by 
    rw [h1, h2, h3]
    exact rfl
  let last_guard_hours := total_shift - used_hours
  have : last_guard_hours = 2 := by 
    rw [h4, this]
    exact rfl
  exact ⟨last_guard_hours, this⟩

end last_guard_hours_l292_292960


namespace consecutive_same_factors_l292_292505

theorem consecutive_same_factors (n : ℕ) : 
  (∀ i, i = n ∨ i = n+1 ∨ i = n+2 → (nat.factors_count n) = (nat.factors_count i)) ↔ n = 33 :=
by sorry

end consecutive_same_factors_l292_292505


namespace peanut_cluster_percentage_l292_292239

def chocolates_total : ℕ := 50
def caramels : ℕ := 3
def nougats : ℕ := 2 * caramels
def truffles : ℕ := caramels + 6
def peanut_clusters : ℕ := chocolates_total - caramels - nougats - truffles

theorem peanut_cluster_percentage : 
  (peanut_clusters / chocolates_total.to_real * 100) = 64 := 
by 
  sorry

end peanut_cluster_percentage_l292_292239


namespace smallest_five_digit_multiple_of_9_starting_with_7_l292_292511

theorem smallest_five_digit_multiple_of_9_starting_with_7 :
  ∃ (n : ℕ), (70000 ≤ n ∧ n < 80000) ∧ (n % 9 = 0) ∧ n = 70002 :=
sorry

end smallest_five_digit_multiple_of_9_starting_with_7_l292_292511


namespace average_speed_l292_292926

theorem average_speed (d1 d2 t1 t2 : ℝ) 
  (h1 : d1 = 100) 
  (h2 : d2 = 80) 
  (h3 : t1 = 1) 
  (h4 : t2 = 1) : 
  (d1 + d2) / (t1 + t2) = 90 := 
by 
  sorry

end average_speed_l292_292926


namespace log_pairs_count_l292_292298

theorem log_pairs_count :
  let L := {2, 3, 4, 5}
  in (∃ P : Finset (ℕ × ℕ), ∃ S : P ⊆ L.product L, 
        P.card = 6 ∧ ∀ (a b : ℕ), (a, b) ∈ P → a < b) :=
sorry

end log_pairs_count_l292_292298


namespace hotel_charge_difference_l292_292470

variables (G : ℝ)

-- Given conditions
def R := 3.0000000000000006 * G
def P := 0.3 * R

-- The goal is to prove the percentage difference
theorem hotel_charge_difference : ((G - P) / G) * 100 = 10 :=
by 
  let R := 3.0000000000000006 * G
  let P := 0.3 * R
  sorry

end hotel_charge_difference_l292_292470


namespace tom_remaining_cars_l292_292498

noncomputable def remaining_cars_with_Tom 
  (cars_per_pkg_4 : ℕ) (cars_per_pkg_7 : ℕ) (total_pkg_4 : ℕ) (total_pkg_7 : ℕ) 
  (f_nephew_4 : ℚ) (f_nephew_7 : ℚ) 
  (s_nephew_4 : ℚ) (s_nephew_7 : ℚ) 
  (t_nephew_4 : ℚ) (t_nephew_7 : ℚ) 
  (fo_nephew_4 : ℚ) (fo_nephew_7 : ℚ) : ℕ :=
  let total_cars_4 := total_pkg_4 * cars_per_pkg_4
  let total_cars_7 := total_pkg_7 * cars_per_pkg_7
  let f_nephew_cars_4 := (total_pkg_4 * cars_per_pkg_4 * f_nephew_4).toNat
  let f_nephew_cars_7 := (total_pkg_7 * cars_per_pkg_7 * f_nephew_7).toNat
  let s_nephew_cars_4 := (total_pkg_4 * cars_per_pkg_4 * s_nephew_4).toNat
  let s_nephew_cars_7 := (total_pkg_7 * cars_per_pkg_7 * s_nephew_7).toNat
  let t_nephew_cars_4 := (total_pkg_4 * cars_per_pkg_4 * t_nephew_4).toNat
  let t_nephew_cars_7 := (total_pkg_7 * cars_per_pkg_7 * t_nephew_7).toNat
  let remaining_cars_4 := total_cars_4 - (f_nephew_cars_4 + s_nephew_cars_4 + t_nephew_cars_4)
  let remaining_cars_7 := total_cars_7 - (f_nephew_cars_7 + s_nephew_cars_7 + t_nephew_cars_7)
  let fo_nephew_cars_4 := (remaining_cars_4 * fo_nephew_4).toNat
  let fo_nephew_cars_7 := (remaining_cars_7 * fo_nephew_7).toNat
  (remaining_cars_4 - fo_nephew_cars_4) + (remaining_cars_7 - fo_nephew_cars_7)

theorem tom_remaining_cars : 
  remaining_cars_with_Tom 4 7 6 6 (1/5) (3/14) (1/4) (1/7) (1/6) (4/14) (1/3) (5/14) = 17 := 
  by
  sorry

end tom_remaining_cars_l292_292498


namespace number_of_female_fish_l292_292575

-- Defining the constants given in the problem
def total_fish : ℕ := 45
def fraction_male : ℚ := 2 / 3

-- The statement we aim to prove in Lean
theorem number_of_female_fish : 
  (total_fish : ℚ) * (1 - fraction_male) = 15 :=
by
  sorry

end number_of_female_fish_l292_292575


namespace max_cube_edge_colors_l292_292283

-- Definitions based on conditions
def is_adjacent (e1 e2 : ℕ) : Prop := 
  -- e1 and e2 are adjacent if they share a common vertex
  sorry

def edge_colors (n : ℕ) : Type :=
  { colors // ∀ c1 c2 ∈ colors, ∃ e1 e2, e1 ≠ e2 ∧ is_adjacent e1 e2 ∧ (e1 = c1 ∨ e2 = c1) ∧ (e1 = c2 ∨ e2 = c2) }

-- Problem statement to prove the maximum number of colors is 6
theorem max_cube_edge_colors : ∃ (colors : edge_colors 12), colors.card = 6 :=
sorry

end max_cube_edge_colors_l292_292283


namespace ellipse_eccentricity_l292_292473

theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : ∃ r : ℝ, r = a ∧ ∀ x y : ℝ, bx - ay + 2ab = 0 → ∃ d : ℝ, d = a ∧ d = r) :
  √6 / 3 :=
begin
  -- We need to prove the eccentricity is √6 / 3 under given conditions
  sorry
end

end ellipse_eccentricity_l292_292473


namespace phenolphthalein_probability_l292_292811

theorem phenolphthalein_probability :
  let total_bottles := 5
  let alkaline_bottles := 2
  total_bottles > 0 ->
  alkaline_bottles >= 0 ->
  alkaline_bottles <= total_bottles ->
  (alkaline_bottles / total_bottles : ℚ) = (2 / 5 : ℚ) :=
by {
  let total_bottles := 5
  let alkaline_bottles := 2
  intros _ _ _
  have : (alkaline_bottles / total_bottles : ℚ) = (2 / 5 : ℚ) := by norm_num
  assumption
  sorry
}

end phenolphthalein_probability_l292_292811


namespace expected_value_of_coins_l292_292542

def coin_values : List ℕ := [1, 5, 10, 25, 50]
def coin_prob : ℕ := 2
def expected_value (values : List ℕ) (prob : ℕ) : ℝ :=
  values.sum * (1.0 / prob)

theorem expected_value_of_coins : expected_value coin_values coin_prob = 45.5 := by
  sorry

end expected_value_of_coins_l292_292542


namespace time_in_2700_minutes_is_3_am_l292_292855

def minutes_in_hour : ℕ := 60
def hours_in_day : ℕ := 24
def current_hour : ℕ := 6
def minutes_later : ℕ := 2700

-- Calculate the final hour after adding the given minutes
def final_hour (current_hour minutes_later minutes_in_hour hours_in_day: ℕ) : ℕ :=
  (current_hour + (minutes_later / minutes_in_hour) % hours_in_day) % hours_in_day

theorem time_in_2700_minutes_is_3_am :
  final_hour current_hour minutes_later minutes_in_hour hours_in_day = 3 :=
by
  sorry

end time_in_2700_minutes_is_3_am_l292_292855


namespace joe_toy_cars_l292_292413

theorem joe_toy_cars (initial_cars : ℕ) (additional_cars : ℕ) (total_cars : ℕ) : 
  initial_cars = 50 → additional_cars = 12 → total_cars = 62 → initial_cars + additional_cars = total_cars := 
by
  intros h1 h2 h3
  rw [h1, h2]
  exact h3
  sorry

end joe_toy_cars_l292_292413


namespace segment_sum_equal_segments_proportional_l292_292180

open Classical

variable (A B C M A1 B1 C1 : Point)
variable (AA1 BB1 MC1 : Length)
variable (m n p CM : Line)

-- Given conditions
def triangle_geom_condition (A B C M : Point) (AA1 BB1 MC1 : Length) (m n p CM : Line) : Prop := 
  -- Definitions of Point and Length equality
  AA1 = BB1 ∧ BB1 = MC1 ∧
  -- Definitions of lines meeting at points along AB
  (parallel_to CM m) ∧
  (parallel_to CM n) ∧
  (parallel_to CM p) ∧
  (intersects_at m A1 AB) ∧
  (intersects_at n B1 AB) ∧
  (intersects_at p C1 AB)

-- Goals
theorem segment_sum_equal (A B C M A1 B1 C1 : Point) (AA1 BB1 MC1 : Length) (m n p CM : Line)
  (condition : triangle_geom_condition A B C M AA1 BB1 MC1 m n p CM) :
  segment_length A1 C1 = AA1 + BB1 := sorry

theorem segments_proportional (A B C M A1 B1 C1 : Point) (AA1 BB1 MC1 : Length) (m n p CM : Line)
  (condition : triangle_geom_condition A B C M AA1 BB1 MC1 m n p CM) :
  proportionality BC CA AB p := sorry

end segment_sum_equal_segments_proportional_l292_292180


namespace min_value_of_quadratic_expression_l292_292881

theorem min_value_of_quadratic_expression : ∃ x : ℝ, (∀ y : ℝ, x^2 + 6 * x + 3 ≤ y) ∧ x^2 + 6 * x + 3 = -6 :=
sorry

end min_value_of_quadratic_expression_l292_292881


namespace A_profit_share_l292_292977

variables (profit : ℚ) (A_share B_share C_share D_share : ℚ)

-- Given conditions
def conditions : Prop :=
  A_share = 1/3 ∧
  B_share = 1/4 ∧
  C_share = 1/5 ∧
  D_share = 1 - (A_share + B_share + C_share) ∧
  profit = 2490

-- The main theorem statement
theorem A_profit_share (h : conditions profit A_share B_share C_share D_share) :
  A_share * profit = 830 :=
sorry

end A_profit_share_l292_292977


namespace cos_C_value_l292_292722

variable (k : ℝ) (ABC : Type) [triangle ABC]
variables (A B C : ABC) (h1 : angle A = 90) (h2 : tan angle C = 4)

noncomputable def cos_C_proof : ℝ :=
\(cos_C : ℝ), (cos angle C = sqrt(17) / 17)
by
  sorry


theorem cos_C_value : cos_C_proof ABC A B C h1 h2 = sqrt(17) / 17 :=
begin
  sorry
end

end cos_C_value_l292_292722


namespace sum_of_ammeter_readings_l292_292292

def I1 := 4 
def I2 := 4
def I3 := 2 * I2
def I5 := I3 + I2
def I4 := (5 / 3) * I5

theorem sum_of_ammeter_readings : I1 + I2 + I3 + I4 + I5 = 48 := by
  sorry

end sum_of_ammeter_readings_l292_292292


namespace midpoint_segment_A_l292_292658

variables {A B C D A' C' : Point}
variables (angle_ABC : Angle A B C)
variables (inside_angle : Inside D angle_ABC)
variables (reflect_B : A' = reflection B D)
variables (reflect_C : C' = reflection C D)

theorem midpoint_segment_A'C' {A B C D A' C' : Point} 
  (h1 : angle_ABC A B C) 
  (h2 : inside_angle D angle_ABC) 
  (h3 : reflect_B A' = reflection B D) 
  (h4 : reflect_C C' = reflection C D) 
  : midpoint D A' C' :=
sorry

end midpoint_segment_A_l292_292658


namespace total_tickets_sold_l292_292561

/-
We define the conditions given in the problem.
-/
variables (A S T : ℕ) -- Define A, S, and T as natural numbers.

def ticketPriceAdvance : ℕ := 20 -- Price of an advance ticket.
def ticketPriceSameDay : ℕ := 30 -- Price of a same-day ticket.
def totalReceipts : ℕ := 1600 -- Total receipts from all tickets sold.
def advanceTicketsSold : ℕ := 20 -- Number of advance tickets sold.

/-
The theorem to prove the total number of tickets sold.
-/
theorem total_tickets_sold : T = 60 :=
by
  have h1 : 20 * advanceTicketsSold + 30 * S = totalReceipts := by
   ---- This is the equation derived from the conditions.
    sorry

  have h2 : S = 40 := by
    ---- Solving for S from h1.
    sorry

  have h3 : T = advanceTicketsSold + S := by
    ---- Total tickets T is the sum of advance tickets and same-day tickets.
    sorry

  ---- Finally combining all the parts.
  show T = 60 by
    rw [h3, h2]
    exact rfl

end total_tickets_sold_l292_292561


namespace ping_pong_balls_count_l292_292441

theorem ping_pong_balls_count :
  (∀ (b t p : ℕ), b = 35 * 4 → t = 6 * 3 → b + t + p = 240 → p = 82) :=
by
  intro b t p hb ht htotal
  have hb_number : b = 140 := by rw [hb]
  have ht_number : t = 18 := by rw [ht]
  have ht_sum : b + t = 140 + 18 := by rw [hb_number, ht_number]
  have htotal_number :  b + t + p = 240 := htotal
  simp [ht_sum, htotal_number]
  sorry

end ping_pong_balls_count_l292_292441


namespace soccer_ball_distribution_l292_292527

theorem soccer_ball_distribution :
  ∃ n : ℕ, n = 10 ∧ (∀ a b c : ℕ, a >= 1 ∧ b >= 2 ∧ c >= 3 ∧ a + b + c = 9 → 
                    (finset.card ((finset.range (a+b+c)).powerset ∩ 
                                  {x : finset ℕ | x.card = 3 ∧ 
                                                  x.sum = a + b + c } 
                    )) = n) :=
sorry

end soccer_ball_distribution_l292_292527


namespace minimum_distance_parallel_lines_l292_292700

theorem minimum_distance_parallel_lines :
  let P : ℝ × ℝ := (x1, y1)
  let Q : ℝ × ℝ := (x2, y2)
  (H1 : 3 * P.fst + 4 * P.snd - 6 = 0)
  (H2 : 6 * Q.fst + 8 * Q.snd + 3 = 0)
  (H3 : ∀ x' y', 3 * x' + 4 * y' - 6 = 0 → ∀ x'' y'', 6 * x'' + 8 * y'' + 3 = 0 → parallel (3 * x' + 4 * y' - 6) (6 * x'' + 8 * y'' + 3)) :
  ∃ (min_dist : ℝ), min_dist = 3 / 2 :=
begin
  sorry
end

end minimum_distance_parallel_lines_l292_292700


namespace line_hyperbola_intersect_once_l292_292124

theorem line_hyperbola_intersect_once (k : ℝ) :
  (∃! p : ℝ × ℝ, p.2 = k * p.1 + 2 ∧ p.1^2 - p.2^2 = 2) ↔ (k = 1 ∨ k = -1 ∨ k = sqrt 3 ∨ k = -sqrt 3) :=
by
  sorry

end line_hyperbola_intersect_once_l292_292124


namespace james_income_ratio_l292_292751

theorem james_income_ratio
  (January_earnings : ℕ := 4000)
  (Total_earnings : ℕ := 18000)
  (Earnings_difference : ℕ := 2000) :
  ∃ (February_earnings : ℕ), 
    (January_earnings + February_earnings + (February_earnings - Earnings_difference) = Total_earnings) ∧
    (February_earnings / January_earnings = 2) := by
  sorry

end james_income_ratio_l292_292751


namespace hyperbola_eccentricity_l292_292005

theorem hyperbola_eccentricity (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (h_asymptote : b / a = 3 / 2) : sqrt (1 + (3/2)^2) = (sqrt 13) / 2 :=
by sorry

end hyperbola_eccentricity_l292_292005


namespace rhombus_area_l292_292251

theorem rhombus_area (a d1 d2 : ℝ) (a_pos : 0 < a) (h_a : a = Real.sqrt 113) (h_diff : d1 - d2 = 8) :
  let x := -2 + Real.sqrt 210 in
  let area := 6 * Real.sqrt 210 - 12 in
  (1 / 2) * d1 * d2 = area :=
by
  sorry

end rhombus_area_l292_292251


namespace tan_pi_over_4_plus_2a_cos_5pi_over_6_minus_2a_l292_292326

-- Given conditions
variable (a : ℝ) (h_a : a ∈ set.Ioo (Real.pi / 2) Real.pi)
variable (h_sin_a : Real.sin a = Real.sqrt 5 / 5)

-- Proof problems as Lean statements
theorem tan_pi_over_4_plus_2a : Real.tan (Real.pi / 4 + 2 * a) = -1 / 7 :=
by
  -- Placeholder for the proof
  sorry

theorem cos_5pi_over_6_minus_2a : Real.cos (5 * Real.pi / 6 - 2 * a) = -(3 * Real.sqrt 3 + 4) / 10 :=
by
  -- Placeholder for the proof
  sorry

end tan_pi_over_4_plus_2a_cos_5pi_over_6_minus_2a_l292_292326


namespace augmented_matrix_solution_l292_292337

theorem augmented_matrix_solution :
  ∀ (m n : ℝ),
  (∃ (x y : ℝ), (m * x = 6 ∧ 3 * y = n) ∧ (x = -3 ∧ y = 4)) →
  m + n = 10 :=
by
  intros m n h
  sorry

end augmented_matrix_solution_l292_292337


namespace line_parallel_plane_l292_292356

open Plane Line

-- Assuming definitions for lines and planes
variables {Plane : Type} {Line : Type}
variables (a b : Line) (α : Plane)

-- Assumptions
variables (h1 : a ∥ b)
variables (h2 : a ∥ α)
variables (h3 : ¬ (b ⊆ α))

-- Claim
theorem line_parallel_plane : b ∥ α := 
by 
  sorry

end line_parallel_plane_l292_292356


namespace solve_inequality_l292_292181

noncomputable def inequality_solution (x : ℝ) : Prop :=
  ((1 / 5) * 5^(2 * x) * 7^(3 * x + 2) ≤ (25 / 7) * 7^(2 * x) * 5^(3 * x)) ↔ (x ≤ -3)

theorem solve_inequality (x : ℝ) : inequality_solution x :=
  sorry

end solve_inequality_l292_292181


namespace inclination_angle_of_line_l292_292339

theorem inclination_angle_of_line (θ : ℝ) :
  (∃ x y : ℝ, sqrt 3 * x - y + 1 = 0) → θ = 60 :=
by
  sorry

end inclination_angle_of_line_l292_292339


namespace expectation_of_η_l292_292673

-- Let ξ be a random variable with E(ξ) = 0.05
axiom ξ : Type → ℝ
axiom hξ : E (ξ) = 0.05

-- Define η as η = 5ξ + 1
noncomputable def η (X : Type → ℝ) : Type → ℝ := λ ω, 5 * X ω + 1

-- Proof statement: E(η) should be 1.25
theorem expectation_of_η (X : Type → ℝ) (hX : E (X) = 0.05) : E (η X) = 1.25 :=
by
  sorry

end expectation_of_η_l292_292673


namespace number_of_single_windows_upstairs_l292_292540

theorem number_of_single_windows_upstairs :
  ∀ (num_double_windows_downstairs : ℕ)
    (glass_panels_per_double_window : ℕ)
    (num_single_windows_upstairs : ℕ)
    (glass_panels_per_single_window : ℕ)
    (total_glass_panels : ℕ),
  num_double_windows_downstairs = 6 →
  glass_panels_per_double_window = 4 →
  glass_panels_per_single_window = 4 →
  total_glass_panels = 80 →
  num_single_windows_upstairs = (total_glass_panels - (num_double_windows_downstairs * glass_panels_per_double_window)) / glass_panels_per_single_window →
  num_single_windows_upstairs = 14 :=
by
  intros
  sorry

end number_of_single_windows_upstairs_l292_292540


namespace aardvark_total_distance_l292_292866

theorem aardvark_total_distance :
  ∀ (r₁ r₂ : ℝ), r₁ = 7 ∧ r₂ = 15 →
  let d := (1/2 * 2 * real.pi * r₂) + (r₂ - r₁) + (1/4 * 2 * real.pi * r₁) + (r₂ - r₁) + r₁ in
  d = (37 * real.pi / 2) + 23 :=
by
  intros r₁ r₂ h
  let d := (1/2 * 2 * real.pi * r₂) + (r₂ - r₁) + (1/4 * 2 * real.pi * r₁) + (r₂ - r₁) + r₁
  sorry

end aardvark_total_distance_l292_292866


namespace initial_cabinets_l292_292040

theorem initial_cabinets (C : ℤ) (h1 : 26 = C + 6 * C + 5) : C = 3 := 
by 
  sorry

end initial_cabinets_l292_292040


namespace bd_squared_l292_292054

theorem bd_squared:
  (exists (O : Point) (Ω : Circle) (A B C D P : Point),
    Ω.center = O ∧
    Ω.inscribed A B C D ∧
    dist A B = 12 ∧
    dist A D = 18 ∧
    ∠ A C B = 90 ∧
    Ω.circumcircle_intersect A O C (ray_through D B) = P ∧
    ∠ P A D = 90
  ) →
  ∃ (BD : ℝ), BD^2 = (1980 / 7) :=
begin
  sorry
end

end bd_squared_l292_292054


namespace apple_weight_l292_292084

theorem apple_weight :
  ∃ (a1 a2 a3 a4 a5 a6 a7 : ℤ), 
    221 ≤ a1 ∧ a1 ≤ 230 ∧
    221 ≤ a2 ∧ a2 ≤ 230 ∧
    221 ≤ a3 ∧ a3 ≤ 230 ∧
    221 ≤ a4 ∧ a4 ≤ 230 ∧
    221 ≤ a5 ∧ a5 ≤ 230 ∧
    221 ≤ a6 ∧ a6 ≤ 230 ∧
    221 ≤ a7 ∧ a7 ≤ 230 ∧
    a7 = 225 ∧
    (a1 + a2 + a3 + a4 + a5 + a6 + a7) % 7 = 0 ∧ 
    ∀ i j, i ≠ j → (a i ≠ a j)
    → a6 = 230 :=
sorry

end apple_weight_l292_292084


namespace car_travel_time_l292_292941

-- Definitions for the conditions
def car_efficiency_kmpl : ℝ := 64    -- car can travel 64 km per liter
def fuel_consumption_gallons : ℝ := 3.9  -- fuel decrease by 3.9 gallons
def car_speed_mph : ℝ := 104  -- car speed is 104 miles per hour
def gallon_to_liter : ℝ := 3.8  -- 1 gallon = 3.8 liters
def mile_to_kilometer : ℝ := 1.6  -- 1 mile = 1.6 kilometers

-- Proof statement
theorem car_travel_time : 
  let fuel_consumption_liters := fuel_consumption_gallons * gallon_to_liter,
  let distance_km := fuel_consumption_liters * car_efficiency_kmpl,
  let distance_miles := distance_km / mile_to_kilometer,
  let travel_time_hours := distance_miles / car_speed_mph 
  in travel_time_hours = 5.7 :=
sorry

end car_travel_time_l292_292941


namespace leah_ride_time_l292_292037

theorem leah_ride_time (x y : ℝ) (h1 : 90 * x = y) (h2 : 30 * (x + 2 * x) = y)
: ∃ t : ℝ, t = 67.5 :=
by
  -- Define 50% increase in length
  let y' := 1.5 * y
  -- Define escalator speed without Leah walking
  let k := 2 * x
  -- Calculate the time taken
  let t := y' / k
  -- Prove that this time is 67.5 seconds
  have ht : t = 67.5 := sorry
  exact ⟨t, ht⟩

end leah_ride_time_l292_292037


namespace exponent_value_l292_292706

theorem exponent_value (y k : ℕ) (h1 : 9^y = 3^k) (h2 : y = 7) : k = 14 := by
  sorry

end exponent_value_l292_292706


namespace find_target_number_l292_292140

theorem find_target_number : ∃ n ≥ 0, (∀ k < 5, ∃ m, 0 ≤ m ∧ m ≤ n ∧ m % 11 = 3 ∧ m = 3 + k * 11) ∧ n = 47 :=
by
  sorry

end find_target_number_l292_292140


namespace reduced_price_per_kg_l292_292171

-- Define the conditions
def reduction_factor : ℝ := 0.80
def extra_kg : ℝ := 4
def total_cost : ℝ := 684

-- Assume the original price P and reduced price R
variables (P R : ℝ)

-- Define the equations derived from the conditions
def original_cost_eq := (P * 16 = total_cost)
def reduced_cost_eq := (0.80 * P * (16 + extra_kg) = total_cost)

-- The final theorem stating the reduced price per kg of oil is 34.20 Rs
theorem reduced_price_per_kg : R = 34.20 :=
by
  have h1: P * 16 = total_cost := sorry -- This will establish the original cost
  have h2: 0.80 * P * (16 + extra_kg) = total_cost := sorry -- This will establish the reduced cost
  have Q: 16 = 16 := sorry -- Calculation of Q (original quantity)
  have h3: P = 42.75 := sorry -- Calculation of original price
  have h4: R = 0.80 * P := sorry -- Calculation of reduced price
  have h5: R = 34.20 := sorry -- Final calculation matching the required answer
  exact h5

end reduced_price_per_kg_l292_292171


namespace part1_B_part1_complement_part2_a_range_l292_292070

noncomputable def U := set.univ

def A : set ℝ := {x | -1 <= x ∧ x < 3}

def B : set ℝ := {x | 2 * x - 4 >= x - 2}

theorem part1_B :
  B = {x : ℝ | x >= 2} :=
sorry

theorem part1_complement :
  (U \ (A ∩ B)) = {x : ℝ | x < 2 ∨ x >= 3} :=
sorry

def C (a : ℝ) : set ℝ := {x | 2 * x + a > 0}

theorem part2_a_range (a : ℝ) :
  B ∪ C a = C a → a > -4 :=
sorry

end part1_B_part1_complement_part2_a_range_l292_292070


namespace find_coordinates_l292_292663

def point_in_fourth_quadrant (P : ℝ × ℝ) : Prop := P.1 > 0 ∧ P.2 < 0
def distance_to_x_axis (P : ℝ × ℝ) (d : ℝ) : Prop := |P.2| = d
def distance_to_y_axis (P : ℝ × ℝ) (d : ℝ) : Prop := |P.1| = d

theorem find_coordinates :
  ∃ P : ℝ × ℝ, point_in_fourth_quadrant P ∧ distance_to_x_axis P 2 ∧ distance_to_y_axis P 5 ∧ P = (5, -2) :=
by
  sorry

end find_coordinates_l292_292663


namespace joseph_drives_more_l292_292047

def joseph_speed : ℝ := 50
def joseph_time : ℝ := 2.5
def kyle_speed : ℝ := 62
def kyle_time : ℝ := 2

def joseph_distance : ℝ := joseph_speed * joseph_time
def kyle_distance : ℝ := kyle_speed * kyle_time

theorem joseph_drives_more : (joseph_distance - kyle_distance) = 1 := by
  sorry

end joseph_drives_more_l292_292047


namespace expected_value_of_flipped_coins_is_45_point_5_cents_l292_292544

theorem expected_value_of_flipped_coins_is_45_point_5_cents:
  let coin_values := [1, 5, 10, 25, 50] in
  let coin_prob := 1 / 2 in
  let expected_value := (coin_prob * (List.sum coin_values : ℝ)) in
  expected_value = 45.5 := 
by 
  sorry

end expected_value_of_flipped_coins_is_45_point_5_cents_l292_292544


namespace cos_C_value_l292_292723

variable (k : ℝ) (ABC : Type) [triangle ABC]
variables (A B C : ABC) (h1 : angle A = 90) (h2 : tan angle C = 4)

noncomputable def cos_C_proof : ℝ :=
\(cos_C : ℝ), (cos angle C = sqrt(17) / 17)
by
  sorry


theorem cos_C_value : cos_C_proof ABC A B C h1 h2 = sqrt(17) / 17 :=
begin
  sorry
end

end cos_C_value_l292_292723


namespace number_of_arrangements_l292_292532

-- Define the classes as a type
inductive Class : Type
| Chinese
| Mathematics
| English
| Physics
| Chemistry
| Physical_Education

open Class

-- Define the main constraints
def validSchedule (schedule : List Class) : Prop :=
  schedule.length = 6 ∧
  schedule.head ≠ some Physical_Education ∧
  schedule.nth 3 ≠ some Mathematics

-- The proof problem statement
theorem number_of_arrangements : 
  ∃ s : List Class, validSchedule s ∧ 
  (finset.univ : finset (List Class)).filter validSchedule.size = 504 :=
sorry

end number_of_arrangements_l292_292532


namespace rectangle_dimensions_l292_292482

theorem rectangle_dimensions (x y : ℝ) (h1 : y = 2 * x) (h2 : 2 * (x + y) = 2 * (x * y)) :
  (x = 3 / 2) ∧ (y = 3) := by
  sorry

end rectangle_dimensions_l292_292482


namespace ab_bc_ca_plus_one_pos_l292_292643

variable (a b c : ℝ)
variable (h₁ : |a| < 1)
variable (h₂ : |b| < 1)
variable (h₃ : |c| < 1)

theorem ab_bc_ca_plus_one_pos :
  ab + bc + ca + 1 > 0 := sorry

end ab_bc_ca_plus_one_pos_l292_292643


namespace projection_matrix_inverse_is_zero_l292_292059

noncomputable def vector_v : Matrix (Fin 2) (Fin 1) ℚ := ![![4], ![-3]]

noncomputable def projection_matrix (v : Matrix (Fin 2) (Fin 1) ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  (v ⬝ vᵀ) ⬝ (1 / (vᵀ ⬝ v))

noncomputable def P : Matrix (Fin 2) (Fin 2) ℚ := projection_matrix vector_v
noncomputable def zero_matrix : Matrix (Fin 2) (Fin 2) ℚ := 0

theorem projection_matrix_inverse_is_zero :
  !((Matrix.det P ≠ 0)) ∧ (P⁻¹ = zero_matrix) := 
by
  sorry

end projection_matrix_inverse_is_zero_l292_292059


namespace complement_intersection_l292_292071

open Set

def U : Set ℕ := {2, 3, 4, 5, 6}
def A : Set ℕ := {2, 5, 6}
def B : Set ℕ := {3, 5}

theorem complement_intersection :
  (U \ A) ∩ B = {3} :=
sorry

end complement_intersection_l292_292071


namespace band_and_chorus_but_not_orchestra_l292_292077

theorem band_and_chorus_but_not_orchestra (B C O : Finset ℕ)
  (hB : B.card = 100) 
  (hC : C.card = 120) 
  (hO : O.card = 60)
  (hUnion : (B ∪ C ∪ O).card = 200)
  (hIntersection : (B ∩ C ∩ O).card = 10) : 
  ((B ∩ C).card - (B ∩ C ∩ O).card = 30) :=
by sorry

end band_and_chorus_but_not_orchestra_l292_292077


namespace total_balls_l292_292262

theorem total_balls {balls_per_box boxes : ℕ} (h1 : balls_per_box = 3) (h2 : boxes = 2) : balls_per_box * boxes = 6 :=
by
  sorry

end total_balls_l292_292262


namespace distance_between_lines_l292_292340

theorem distance_between_lines (a b c : ℝ) (h : a + b = -1) (h2 : a * b = c) (h3 : 0 ≤ c ∧ c ≤ 1 / 8) :
  let max_dist := real.sqrt 2 / 2
  let min_dist := 1 / 2
  distance (line (a, b)) (line (b, a)) = \max_dist ∧ distance (line (a, b)) (line (b, a)) = min_dist :=
sorry

end distance_between_lines_l292_292340


namespace carly_cooks_in_72_minutes_l292_292242

def total_time_to_cook_burgers (total_guests : ℕ) (cook_time_per_side : ℕ) (burgers_per_grill : ℕ) : ℕ :=
  let guests_who_want_two_burgers := total_guests / 2
  let guests_who_want_one_burger := total_guests - guests_who_want_two_burgers
  let total_burgers := (guests_who_want_two_burgers * 2) + guests_who_want_one_burger
  let total_batches := (total_burgers + burgers_per_grill - 1) / burgers_per_grill  -- ceil division for total batches
  total_batches * (2 * cook_time_per_side)  -- total time

theorem carly_cooks_in_72_minutes : 
  total_time_to_cook_burgers 30 4 5 = 72 :=
by 
  sorry

end carly_cooks_in_72_minutes_l292_292242


namespace max_m_value_range_a_for_g_zero_l292_292348

-- Definitions
def f (x a : ℝ) : ℝ := |x - a| + (1 / (2 * a))
def g (x a : ℝ) : ℝ := f x a + |2 * x - 1|

-- Question (1): Prove max value of m is 1 if inequality always holds
theorem max_m_value (a : ℝ) (h : a ≠ 0) :
  (∀ x m : ℝ, f x a - f (x + m) a ≤ 1) ↔ ∃ (m₀ : ℝ), m₀ = 1 :=
sorry

-- Question (2): Prove the range for a for g(x) to have a zero
theorem range_a_for_g_zero (a : ℝ) (h : a < 1/2) :
  (∃ x : ℝ, g x a = 0) ↔ -1/2 ≤ a ∧ a < 0 :=
sorry

end max_m_value_range_a_for_g_zero_l292_292348


namespace number_of_days_l292_292409

theorem number_of_days (m1 d1 m2 d2 : ℕ) (h1 : m1 * d1 = m2 * d2) (k : ℕ) 
(h2 : m1 = 10) (h3 : d1 = 6) (h4 : m2 = 15) (h5 : k = 60) : 
d2 = 4 :=
by
  have : 10 * 6 = 60 := by sorry
  have : 15 * d2 = 60 := by sorry
  exact sorry

end number_of_days_l292_292409


namespace correct_conclusion_l292_292300

def U : Set ℝ := Set.univ

def f (x : ℝ) : ℝ := Real.log (1 - x^2)

def M : Set ℝ := {x | -1 < x ∧ x < 1}

def N : Set ℝ := {x | x^2 - x < 0}

theorem correct_conclusion : M ∩ N = N := by
  sorry

end correct_conclusion_l292_292300


namespace isosceles_triangle_l292_292717

noncomputable def sin (x : ℝ) : ℝ := Real.sin x
noncomputable def cos (x : ℝ) : ℝ := Real.cos x

variables {A B C : ℝ}
variable (h : sin C = 2 * sin (B + C) * cos B)

theorem isosceles_triangle (h : sin C = 2 * sin (B + C) * cos B) : A = B :=
by
  sorry

end isosceles_triangle_l292_292717


namespace swap_square_digit_l292_292608

theorem swap_square_digit (n : ℕ) (h1 : n ≥ 10 ∧ n < 100) : 
  ∃ (x y : ℕ), n = 10 * x + y ∧ (x < 10 ∧ y < 10) ∧ (y * 100 + x * 10 + y^2 + 20 * x * y - 1) = n * n + 2 * n + 1 :=
by 
    sorry

end swap_square_digit_l292_292608


namespace farmer_profit_l292_292952

def piglet_cost_per_month : Int := 10
def pig_revenue : Int := 300
def num_piglets_sold_early : Int := 3
def num_piglets_sold_late : Int := 3
def early_sale_months : Int := 12
def late_sale_months : Int := 16

def total_profit (num_piglets_sold_early num_piglets_sold_late early_sale_months late_sale_months : Int) 
  (piglet_cost_per_month pig_revenue : Int) : Int := 
  let early_cost := num_piglets_sold_early * piglet_cost_per_month * early_sale_months
  let late_cost := num_piglets_sold_late * piglet_cost_per_month * late_sale_months
  let total_cost := early_cost + late_cost
  let total_revenue := (num_piglets_sold_early + num_piglets_sold_late) * pig_revenue
  total_revenue - total_cost

theorem farmer_profit : total_profit num_piglets_sold_early num_piglets_sold_late early_sale_months late_sale_months piglet_cost_per_month pig_revenue = 960 := by
  sorry

end farmer_profit_l292_292952


namespace middle_fraction_sorted_l292_292707

theorem middle_fraction_sorted :
    let f1 := (2 : ℕ) / 3
    let f2 := 23 / 30
    let f3 := 9 / 10
    let f4 := 11 / 15
    let f5 := 4 / 5 in
    (List.sort (λ x y => x ≤ y) [f1, f2, f3, f4, f5]).nth! 2 = f2 :=
by
    sorry

end middle_fraction_sorted_l292_292707


namespace total_cost_l292_292081

-- Definitions based on conditions
def old_camera_cost : ℝ := 4000
def new_model_cost_increase_rate : ℝ := 0.3
def lens_initial_cost : ℝ := 400
def lens_discount : ℝ := 200

-- Main statement to prove
theorem total_cost (old_camera_cost new_model_cost_increase_rate lens_initial_cost lens_discount : ℝ) : 
  let new_camera_cost := old_camera_cost * (1 + new_model_cost_increase_rate)
  let lens_cost_after_discount := lens_initial_cost - lens_discount
  (new_camera_cost + lens_cost_after_discount) = 5400 :=
by
  sorry

end total_cost_l292_292081


namespace complement_of_M_in_U_l292_292689

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 4, 6}
def complement_U_M : Set ℕ := {1, 3, 5}

theorem complement_of_M_in_U :
  (U \ M) = complement_U_M :=
by
  sorry

end complement_of_M_in_U_l292_292689


namespace student_chose_number_l292_292210

theorem student_chose_number : 
  ∀ (x : ℤ), x = 129 → 2 * x - 148 = 110 := 
by
  intros x h
  rw [h]
  calc
    2 * 129 - 148 = 258 - 148 := by norm_num
    ... = 110 := by norm_num

end student_chose_number_l292_292210


namespace school_robes_l292_292969

theorem school_robes (total_singers robes_needed : ℕ) (robe_cost total_spent existing_robes : ℕ) 
  (h1 : total_singers = 30)
  (h2 : robe_cost = 2)
  (h3 : total_spent = 36)
  (h4 : total_singers - total_spent / robe_cost = existing_robes) :
  existing_robes = 12 :=
by sorry

end school_robes_l292_292969


namespace total_pieces_of_gum_l292_292451

def packages : ℕ := 12
def pieces_per_package : ℕ := 20

theorem total_pieces_of_gum : packages * pieces_per_package = 240 :=
by
  -- proof is skipped
  sorry

end total_pieces_of_gum_l292_292451


namespace triangle_cosine_problem_l292_292847

theorem triangle_cosine_problem
  (n : ℕ) 
  (h1 : Odd (n + 1))
  (h2 : Odd (n + 3))
  (h3 : n + 3 = 3 * (n + 1)) :
  cos (angle_opposite_side (n+1) (n+2) (n+3)) = 6/11 := by
  sorry

end triangle_cosine_problem_l292_292847


namespace exists_n_faces_with_same_sides_l292_292094

-- Define the conditions of the problem in Lean
variable (n : ℕ) (P : Type)
variable [ConvexPolyhedron P] [HasFaces P (10 * n)]

-- State the theorem
theorem exists_n_faces_with_same_sides (P : ConvexPolyhedron) (n : ℕ) 
  (hP : P.faces.count = 10 * n) : 
  ∃ (F : Set Face), F.card = n ∧ ∀ f ∈ F, ∃ k : ℕ, Face.sides f = k := 
s類product sorry

end exists_n_faces_with_same_sides_l292_292094


namespace least_positive_difference_between_A_and_B_is_6_l292_292454

noncomputable def a (n : ℕ) : ℕ := 2 * 3^(n-1)
noncomputable def b (n : ℕ) : ℕ := 30 * n

def validA (a : ℕ) : Prop := ∃ n, a = 2 * 3^(n-1) ∧ a ≤ 500
def validB (b : ℕ) : Prop := ∃ n, b = 30 * n ∧ b ≤ 500

theorem least_positive_difference_between_A_and_B_is_6 :
  ∃ d, d = 6 ∧ ∀ a ∈ {x | validA x}, ∀ b ∈ {y | validB y}, abs (a - b) ≥ d :=
sorry

end least_positive_difference_between_A_and_B_is_6_l292_292454


namespace distance_traveled_eq_12pi_l292_292560

-- Define the radius of the wheel as 2 meters
def radius : ℝ := 2

-- Define the number of revolutions
def revolutions : ℕ := 3

-- Define the formula for the circumference of a circle
def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

-- Define the total distance traveled given the number of revolutions and the circumference
def total_distance_traveled (r : ℝ) (revs : ℕ) : ℝ := revs * (circumference r)

-- Prove that the total distance traveled is 12π meters
theorem distance_traveled_eq_12pi : total_distance_traveled radius revolutions = 12 * Real.pi := by
  sorry

end distance_traveled_eq_12pi_l292_292560


namespace irises_to_add_and_total_flowers_l292_292130

theorem irises_to_add_and_total_flowers
  (initial_roses : ℕ)
  (added_roses : ℕ)
  (initial_ratio_irises_roses : ℕ × ℕ)
  (maintain_ratio_irises_roses : ℕ × ℕ) :
  initial_roses = 49 →
  added_roses = 35 →
  initial_ratio_irises_roses = (3, 7) →
  maintain_ratio_irises_roses = (3, 7) →
  let initial_irises := (initial_ratio_irises_roses.1 * initial_roses) / initial_ratio_irises_roses.2 in
  let final_roses := initial_roses + added_roses in
  let final_irises_needed := (maintain_ratio_irises_roses.1 * final_roses) / maintain_ratio_irises_roses.2 in
  final_irises_needed - initial_irises = 15 ∧
  final_roses + final_irises_needed = 120 :=
by
  intros h1 h2 h3 h4
  let initial_irises : ℕ := (initial_ratio_irises_roses.1 * initial_roses) / initial_ratio_irises_roses.2
  let final_roses : ℕ := initial_roses + added_roses
  let final_irises_needed : ℕ := (maintain_ratio_irises_roses.1 * final_roses) / maintain_ratio_irises_roses.2
  have h_initial_irises : initial_irises = 21 := by sorry
  have h_final_roses : final_roses = 84 := by sorry
  have h_final_irises_needed : final_irises_needed = 36 := by sorry
  have h_irises_to_add : final_irises_needed - initial_irises = 15 := by sorry
  have h_total_flowers : final_roses + final_irises_needed = 120 := by sorry
  exact ⟨h_irises_to_add, h_total_flowers⟩

end irises_to_add_and_total_flowers_l292_292130


namespace reaction_sequence_l292_292603

-- Define the chemical equation reactions and conditions
def reaction_step1 (C5H12O HCl C5H11Cl H2O : ℕ → Prop) (x : ℕ) : Prop :=
  ∀ (k : ℕ), C5H12O (2 * k) → HCl (x * k) → C5H11Cl (2 * k) ∧ H2O (2 * k)

def reaction_step2 (C5H11Cl PCl3 C5H11PCl3 HCl : ℕ → Prop) (y z : ℕ) : Prop :=
  ∀ (k : ℕ), C5H11Cl (2 * k) → PCl3 (y * k) → C5H11PCl3 (2 * k) ∧ HCl (z * k)

-- Prove that x = 2, y = 2, and z = 2
theorem reaction_sequence :
  ∃ (x y z : ℕ), 
    (∀ (C5H12O HCl C5H11Cl H2O : ℕ → Prop), reaction_step1 C5H12O HCl C5H11Cl H2O x) ∧ 
    (∀ (C5H11Cl PCl3 C5H11PCl3 HCl : ℕ → Prop), reaction_step2 C5H11Cl PCl3 C5H11PCl3 HCl y z) ∧ 
    x = 2 ∧ y = 2 ∧ z = 2 :=
by
  use 2, 2, 2
  split
  { intros C5H12O HCl C5H11Cl H2O h k,
    refine ⟨h.left k, h.right.left (@eq.refl (2 * k)), h.right.right (@eq.refl (2 * k))⟩ }
  { intros C5H11Cl PCl3 C5H11PCl3 HCl h k,
    refine ⟨h.left k, h.right.left (@eq.refl (2 * k)), h.right.right (@eq.refl (2 * k))⟩ }
  repeat { refl }

end reaction_sequence_l292_292603


namespace units_digit_sum_l292_292896

theorem units_digit_sum (h₁ : (24 : ℕ) % 10 = 4) 
                        (h₂ : (42 : ℕ) % 10 = 2) : 
  ((24^3 + 42^3) % 10 = 2) :=
by
  sorry

end units_digit_sum_l292_292896


namespace max_students_seated_l292_292222

theorem max_students_seated : 
  let seats_in_row (i : ℕ) := 13 + 2 * i
  let max_students_in_row (n : ℕ) := (n + 1) / 2
  ∑ i in Finset.range 10, max_students_in_row (seats_in_row i) = 99 :=
by
  sorry

end max_students_seated_l292_292222


namespace cos_value_l292_292725

-- Definitions
def angle_A : ℝ := 90
def tan_C : ℝ := 4
def cos_C := real.cos

-- Problem statement
theorem cos_value (A C : ℝ) (hA : A = 90) (hC : real.tan C = 4) :
  cos_C C = sqrt 17 / 17 :=
sorry

end cos_value_l292_292725


namespace central_angle_sine_product_l292_292732

theorem central_angle_sine_product {radius : ℝ} (h_radius : radius = 10) 
  {PQ : ℝ} {RS : ℝ} {T : ℝ} {PT : ℝ} {PR : ℝ} 
  (h_T_bisects_PQ : T = PQ / 2) 
  (h_PT : PT = 8) 
  (h_unique_bisected : PQ = 2 * PT) :
  let PR := 8 in
  let cos_PRT := 7 / 16 in
  let sine_value := Real.sqrt (207) / 16 in
  ∃ m n : ℝ, (m = Real.sqrt (207)) ∧ (n = 16) ∧ (m * n = 52.8) := 
begin
  use [Real.sqrt (207), 16],
  split,
  { refl, },
  split,
  { refl, },
  { norm_num, }
end
sorry

end central_angle_sine_product_l292_292732


namespace probability_of_three_positive_answers_l292_292412

noncomputable def probability_exactly_three_positive_answers : ℚ :=
  (7.choose 3) * (3/7)^3 * (4/7)^4

theorem probability_of_three_positive_answers :
  probability_exactly_three_positive_answers = 242520 / 823543 :=
by
  unfold probability_exactly_three_positive_answers
  sorry

end probability_of_three_positive_answers_l292_292412


namespace birdseed_weekly_consumption_l292_292446

def parakeets := 3
def parakeet_consumption := 2
def parrots := 2
def parrot_consumption := 14
def finches := 4
def finch_consumption := parakeet_consumption / 2
def canaries := 5
def canary_consumption := 3
def african_grey_parrots := 2
def african_grey_parrot_consumption := 18
def toucans := 3
def toucan_consumption := 25

noncomputable def daily_consumption := 
  parakeets * parakeet_consumption +
  parrots * parrot_consumption +
  finches * finch_consumption +
  canaries * canary_consumption +
  african_grey_parrots * african_grey_parrot_consumption +
  toucans * toucan_consumption

noncomputable def weekly_consumption := 7 * daily_consumption

theorem birdseed_weekly_consumption : weekly_consumption = 1148 := by
  sorry

end birdseed_weekly_consumption_l292_292446


namespace sum_even_positive_integers_less_than_102_l292_292890

theorem sum_even_positive_integers_less_than_102 : 
  let sum_even : ℕ := ∑ n in Finset.filter (λ x, even x) (Finset.range 102), n
  in sum_even = 2550 :=
by
  sorry

end sum_even_positive_integers_less_than_102_l292_292890


namespace eval_expression_l292_292265

theorem eval_expression (a x : ℕ) (h : x = a + 9) : x - a + 5 = 14 :=
by 
  sorry

end eval_expression_l292_292265


namespace probability_two_number_cards_sum_to_17_l292_292149

/-- Two cards are drawn from a standard 52-card deck. Number cards range from 2 to 10. 
    The probability that both cards are number cards totalling to 17 is 8/663. -/
theorem probability_two_number_cards_sum_to_17 : 
  let number_card_count := 36 in -- There are 36 number cards (4 each from 2 to 10)
  let pair_count := 2 in -- Two ways to choose a pair (8,9) and (9,8)
  let single_card_prob := (4:ℚ) / 52 in -- Probability of drawing a specific number card (e.g., 8 or 9)
  let second_card_prob := (4:ℚ) / 51 in -- Probability of drawing the matching card after the first
  (pair_count * single_card_prob * second_card_prob) = (8 / 663) :=
by
  let number_card_count := 36
  let pair_count := 2
  let single_card_prob := (4:ℚ) / 52
  let second_card_prob := (4:ℚ) / 51
  sorry -- Proof is omitted.

end probability_two_number_cards_sum_to_17_l292_292149


namespace farmer_profit_l292_292953

def piglet_cost_per_month : Int := 10
def pig_revenue : Int := 300
def num_piglets_sold_early : Int := 3
def num_piglets_sold_late : Int := 3
def early_sale_months : Int := 12
def late_sale_months : Int := 16

def total_profit (num_piglets_sold_early num_piglets_sold_late early_sale_months late_sale_months : Int) 
  (piglet_cost_per_month pig_revenue : Int) : Int := 
  let early_cost := num_piglets_sold_early * piglet_cost_per_month * early_sale_months
  let late_cost := num_piglets_sold_late * piglet_cost_per_month * late_sale_months
  let total_cost := early_cost + late_cost
  let total_revenue := (num_piglets_sold_early + num_piglets_sold_late) * pig_revenue
  total_revenue - total_cost

theorem farmer_profit : total_profit num_piglets_sold_early num_piglets_sold_late early_sale_months late_sale_months piglet_cost_per_month pig_revenue = 960 := by
  sorry

end farmer_profit_l292_292953


namespace ratio_of_segments_l292_292664

theorem ratio_of_segments (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 :=
sorry

end ratio_of_segments_l292_292664


namespace pqyx_cyclic_l292_292931

theorem pqyx_cyclic (A B C P Q X Y : Type*)
  [triangle_ABC_isosceles : is_isosceles_triangle A B C]
  (hP_inside : point_inside_triangle P A B C)
  (hQ_inside : point_inside_triangle Q A P C)
  (angle_PAQ : ∠PAQ = ∠BAC / 2)
  (hBP_PQ : BP = PQ)
  (hPQ_CQ : PQ = CQ)
  (h_intersections : intersect (line_through A P) (line_through B Q) X ∧ intersect (line_through A Q) (line_through C P) Y) :
  cyclic_quadrilateral P Q Y X :=
sorry

end pqyx_cyclic_l292_292931


namespace ratio_of_circle_areas_correct_l292_292150

noncomputable def ratio_of_circle_areas 
  (O Y P : Point) 
  (r R : ℝ)
  (h1 : dist O P = 3 * dist O Y) 
  (h2 : Y ∈ line_segment O P)
  : ℝ :=
  (π * (dist O Y)^2) / (π * (dist O P)^2)

theorem ratio_of_circle_areas_correct
  (O Y P : Point)
  (h1 : dist O P = 3 * dist O Y)
  (h2 : Y ∈ line_segment O P)
  : ratio_of_circle_areas O Y P (dist O Y) (dist O P) = 4 / 9 := 
  sorry

end ratio_of_circle_areas_correct_l292_292150


namespace batsman_running_percentage_l292_292918

theorem batsman_running_percentage :
  ∀ (total_runs runs_from_boundaries runs_from_sixes : ℕ),
  (boundaries sixes : ℕ)
  (boundary_runs_per_boundary six_runs_per_six : ℕ)
  (H_total_runs : total_runs = 120)
  (H_boundaries : boundaries = 6)
  (H_sixes : sixes = 4)
  (H_boundary_runs_per_boundary : boundary_runs_per_boundary = 4)
  (H_six_runs_per_six : six_runs_per_six = 6)
  (H_boundary_runs : runs_from_boundaries = boundaries * boundary_runs_per_boundary)
  (H_six_runs : runs_from_sixes = sixes * six_runs_per_six)
  (H_non_running_runs : runs_from_boundaries + runs_from_sixes = 48)
  (H_running_runs : total_runs - (runs_from_boundaries + runs_from_sixes) = 72),
  (72 * 100 / 120) = 60 :=
by
  sorry

end batsman_running_percentage_l292_292918


namespace sequence_term_10_l292_292682

theorem sequence_term_10 : ∃ n : ℕ, (1 / (n * (n + 2)) = 1 / 120) ∧ n = 10 := by
  sorry

end sequence_term_10_l292_292682


namespace trigonometric_function_relation_l292_292346

theorem trigonometric_function_relation 
    (A ω ϕ : ℝ)
    (hA : 0 < A)
    (hω : ω = 2)
    (hϕ : ϕ = π / 6)
    (period : ∀ x : ℝ, f (x + π / 2) = f x)
    (min_val : f (2 * π / 3) = -A) :
    f (π / 2) < f 0 ∧ f 0 < f (π / 6) :=
by
  let f : ℝ → ℝ := λ x, A * sin (ω * x + ϕ)
  sorry

end trigonometric_function_relation_l292_292346


namespace maximum_value_l292_292934

theorem maximum_value (f : ℝ → ℝ) (hf : ∀ x, f x = x * (1 - x)) :
  ∃ x ∈ Ioo 0 1, ∀ y ∈ Ioo 0 1, f y ≤ 1 / 4 :=
by
  sorry

end maximum_value_l292_292934


namespace expected_variance_l292_292937

section

variable (p : ℝ) (n : ℕ) (X : ℕ → ℝ) (Y : ℕ → ℝ)

# Conditions
def prob_success : ℝ := 0.6
def num_shots : ℕ := 5
def points_per_shot : ℕ := 10
def successful_shots (i : ℕ) : ℝ := binomial n p i

# Given that Y = 10X
def total_points (i : ℕ) : ℝ := points_per_shot * (successful_shots i)

# Lean Statement
theorem expected_variance :
  E (fun i => successful_shots i) = 3 ∧ D (fun i => total_points i) = 120 :=
sorry

end

end expected_variance_l292_292937


namespace constant_term_expansion_l292_292117

theorem constant_term_expansion :
  let f := λ x : ℝ, x + (1 / x) - 2 in
  let expansion := (f(1))^5 in
  (∃ r : ℤ, expansion = r) ∧ r = -252 :=
sorry

end constant_term_expansion_l292_292117


namespace eval_expression_l292_292267

theorem eval_expression (a x : ℕ) (h : x = a + 9) : x - a + 5 = 14 :=
by 
  sorry

end eval_expression_l292_292267


namespace first_grade_frequency_is_correct_second_grade_frequency_is_correct_l292_292988

def total_items : ℕ := 400
def second_grade_items : ℕ := 20
def first_grade_items : ℕ := total_items - second_grade_items

def frequency_first_grade : ℚ := first_grade_items / total_items
def frequency_second_grade : ℚ := second_grade_items / total_items

theorem first_grade_frequency_is_correct : frequency_first_grade = 0.95 := 
 by
 sorry

theorem second_grade_frequency_is_correct : frequency_second_grade = 0.05 := 
 by 
 sorry

end first_grade_frequency_is_correct_second_grade_frequency_is_correct_l292_292988


namespace simplify_fraction_l292_292827

theorem simplify_fraction (x : ℝ) (hx : x ≠ 0) :
  (5 / (4 * x^(-4))) * (4 * x^3 / 3) = 5 * x^7 / 3 :=
by
  sorry

end simplify_fraction_l292_292827


namespace sum_max_min_neg_7_neg_3_l292_292379

noncomputable def f : ℝ → ℝ := sorry

variable (f_odd : ∀ x, f (-x) = -f (x))
variable (f_min_3_7 : ∀ x, 3 ≤ x ∧ x ≤ 7 → 5 ≤ f (x))
variable (f_max_3_7 : ∀ x, 3 ≤ x ∧ x ≤ 7 → f (x) ≤ 6)

theorem sum_max_min_neg_7_neg_3 :
  (∀ x, -7 ≤ x ∧ x ≤ -3 → -6 ≤ f (x)) ∧
  (∀ x, -7 ≤ x ∧ x ≤ -3 → f (x) ≤ -5) →
  (Sup (Set.Image f (Set.Icc (-7:ℝ) (-3)))).toReal +
  (Inf (Set.Image f (Set.Icc (-7:ℝ) (-3)))).toReal = -11 := by
    intros
    sorry

end sum_max_min_neg_7_neg_3_l292_292379


namespace solve_for_P_l292_292103

theorem solve_for_P (P : ℝ) :
  sqrt (P^3) = 81 * real.cbrt 81 → P = real.exp ((32 / 9) * real.ln 3) :=
by
  intro h
  sorry

end solve_for_P_l292_292103


namespace eval_expression_l292_292269

theorem eval_expression (a x : ℤ) (h : x = a + 9) : (x - a + 5) = 14 :=
by
  sorry

end eval_expression_l292_292269


namespace books_and_games_left_to_experience_l292_292138

def booksLeft (B_total B_read : Nat) : Nat := B_total - B_read
def gamesLeft (G_total G_played : Nat) : Nat := G_total - G_played
def totalLeft (B_total B_read G_total G_played : Nat) : Nat := booksLeft B_total B_read + gamesLeft G_total G_played

theorem books_and_games_left_to_experience :
  totalLeft 150 74 50 17 = 109 := by
  sorry

end books_and_games_left_to_experience_l292_292138


namespace solve_for_angle_B_solutions_l292_292747

noncomputable def number_of_solutions_for_angle_B (BC AC : ℝ) (angle_A : ℝ) : ℕ :=
  if (BC = 6 ∧ AC = 8 ∧ angle_A = 40) then 2 else 0

theorem solve_for_angle_B_solutions : number_of_solutions_for_angle_B 6 8 40 = 2 :=
  by sorry

end solve_for_angle_B_solutions_l292_292747


namespace normal_dist_probability_l292_292472

noncomputable def normalDist (μ σ : ℝ) : ℝ → ℝ := sorry -- placeholder for actual normal distribution function

theorem normal_dist_probability (μ σ : ℝ) (a : ℝ)
  (hμ : μ = 4.5) (hσ : σ = 0.05) (ha : a = 0.1) :
  normalDist μ σ (|4.5 - ha|) = 0.9544 :=
sorry

end normal_dist_probability_l292_292472


namespace no_real_roots_l292_292332

theorem no_real_roots (n : ℕ) (h_even : n % 2 = 0) (c : Finₓ (n - 1) → ℝ)
  (h_sum : ∑ i, |(c i) - 1| < 1) :
  ¬ ∃ x : ℝ, 2 * x^n + ∑ i, (-1)^(i + 1) * (c i) * x^(i + 1) + 2 = 0 := sorry

end no_real_roots_l292_292332


namespace total_female_officers_proof_l292_292808

variable (F : ℕ)
variable (on_duty : ℕ := 210)
variable (female_on_duty : ℕ := (2 / 3) * on_duty)
def percentage_on_duty : ℝ := 0.24
def total_female_officers (F : ℕ) : Prop := F = Int.toNat (Int.ofNat female_on_duty / percentage_on_duty).floor

theorem total_female_officers_proof : total_female_officers 583 := by
  sorry

end total_female_officers_proof_l292_292808


namespace joska_has_higher_probability_l292_292755

open Nat

def num_4_digit_with_all_diff_digits := 10 * 9 * 8 * 7
def total_4_digit_combinations := 10^4
def num_4_digit_with_repeated_digits := total_4_digit_combinations - num_4_digit_with_all_diff_digits

-- Calculate probabilities
noncomputable def prob_joska := (num_4_digit_with_all_diff_digits : ℝ) / (total_4_digit_combinations : ℝ)
noncomputable def prob_gabor := (num_4_digit_with_repeated_digits : ℝ) / (total_4_digit_combinations : ℝ)

theorem joska_has_higher_probability :
  prob_joska > prob_gabor :=
  by
    sorry

end joska_has_higher_probability_l292_292755


namespace num_men_in_second_group_l292_292372

-- Define the conditions
def numMen1 := 4
def hoursPerDay1 := 10
def daysPerWeek := 7
def earningsPerWeek1 := 1200

def hoursPerDay2 := 6
def earningsPerWeek2 := 1620

-- Define the earning per man-hour
def earningPerManHour := earningsPerWeek1 / (numMen1 * hoursPerDay1 * daysPerWeek)

-- Define the total man-hours required for the second amount of earnings
def totalManHours2 := earningsPerWeek2 / earningPerManHour

-- Define the number of men in the second group
def numMen2 := totalManHours2 / (hoursPerDay2 * daysPerWeek)

-- Theorem stating the number of men in the second group 
theorem num_men_in_second_group : numMen2 = 9 := by
  sorry

end num_men_in_second_group_l292_292372


namespace eval_expression_l292_292272

theorem eval_expression (a x : ℤ) (h : x = a + 9) : (x - a + 5) = 14 :=
by
  sorry

end eval_expression_l292_292272


namespace smallest_n_with_monochromatic_rectangle_l292_292016

theorem smallest_n_with_monochromatic_rectangle (n : ℕ) (h : n ≥ 5) : 
  ∀ (grid : (fin n → fin n → bool)), 
    ∃ (r1 r2 c1 c2 : fin n), 
    r1 ≠ r2 ∧ c1 ≠ c2 ∧ grid r1 c1 = grid r1 c2 ∧ grid r1 c1 = grid r2 c1 ∧ grid r1 c1 = grid r2 c2 :=
sorry

end smallest_n_with_monochromatic_rectangle_l292_292016


namespace value_of_a8_l292_292024

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

noncomputable def arithmetic_sequence (a : ℕ → α) : Prop :=
∀ n : ℕ, ∃ d : α, a (n + 1) = a n + d

variable {a : ℕ → ℝ}

axiom seq_is_arithmetic : arithmetic_sequence a

axiom initial_condition :
  a 1 + 3 * a 8 + a 15 = 120

axiom arithmetic_property :
  a 1 + a 15 = 2 * a 8

theorem value_of_a8 : a 8 = 24 :=
by {
  sorry
}

end value_of_a8_l292_292024


namespace min_circles_cover_l292_292099

/--
Prove that the minimum number of identical circles $K_{1}, K_{2}, K_{3}, \ldots$ needed to cover a circle $K$ with twice the radius is 7.
Conditions:
- Let $K$ be a circle with radius $R$.
- Each smaller circle $K_{i}$ (where $i \geq 1$) has a radius $\frac{1}{2} R$.
-/
theorem min_circles_cover (R : ℝ) (K : TopologicalSpace Basic := by apply_instance) (K1 K2 K3 K4 K5 K6 K7 : TopologicalSpace Basic := by apply_instance) :
  ∃ (K₁ K₂ K₃ K₄ K₅ K₆ K₇ : Circle),
    (radius K = 2 * radius K₁) ∧ 
    (radius K₁ = radius K₂) ∧ (radius K₁ = radius K₃) ∧ (radius K₁ = radius K₄) ∧ (radius K₁ = radius K₅) ∧ (radius K₁ = radius K₆) ∧ (radius K₁ = radius K₇) ∧
    covers K (K₁ ∪ K₂ ∪ K₃ ∪ K₄ ∪ K₅ ∪ K₆ ∪ K₇) :=
begin
  sorry
end

end min_circles_cover_l292_292099


namespace sum_of_all_potential_real_values_of_x_l292_292257

/-- Determine the sum of all potential real values of x such that when the mean, median, 
and mode of the list [12, 3, 6, 3, 8, 3, x, 15] are arranged in increasing order, they 
form a non-constant arithmetic progression. -/
def sum_potential_x_values : ℚ :=
    let values := [12, 3, 6, 3, 8, 3, 15]
    let mean (x : ℚ) : ℚ := (50 + x) / 8
    let mode : ℚ := 3
    let median (x : ℚ) : ℚ := 
      if x ≤ 3 then 3.5 else if x < 6 then (x + 6) / 2 else 6
    let is_arithmetic_seq (a b c : ℚ) : Prop := 2 * b = a + c
    let valid_x_values : List ℚ := 
      (if is_arithmetic_seq mode 3.5 (mean (3.5)) then [] else []) ++
      (if is_arithmetic_seq mode 6 (mean 6) then [22] else []) ++
      (if is_arithmetic_seq mode (median (50 / 7)) (mean (50 / 7)) then [50 / 7] else [])
    (valid_x_values.sum)
theorem sum_of_all_potential_real_values_of_x :
  sum_potential_x_values = 204 / 7 :=
  sorry

end sum_of_all_potential_real_values_of_x_l292_292257


namespace real_number_m_value_purely_imaginary_number_m_value_l292_292647

variable {m : ℝ}

def z : ℂ := ⟨5 * m^2 - 45, m + 3⟩

theorem real_number_m_value (h : z.im = 0) : m = -3 := by
  sorry

theorem purely_imaginary_number_m_value (h : z.re = 0) (h_im : z.im ≠ 0) : m = 3 := by
  sorry

end real_number_m_value_purely_imaginary_number_m_value_l292_292647


namespace sequence_a1_a3_sum_l292_292657

-- Definitions based on given condition
def S (n : ℕ) : ℕ := 2^n + n - 1
def sequence_sum (n : ℕ) (a : ℕ → ℕ) : ℕ := (Finset.range n).sum a

-- Main statement rewriting the problem
theorem sequence_a1_a3_sum :
  (∃ (a : ℕ → ℕ), a 1 + a 3 = 7 ∧ 
  (∀ n, (Finset.range (n + 1)).sum a = S (n + 1))) :=
begin
  use λ n, S n - S (n-1),
  sorry 
end

end sequence_a1_a3_sum_l292_292657


namespace diagonals_perpendicular_if_midsegments_equal_midsegments_equal_if_diagonals_perpendicular_l292_292449

structure Quadrilateral :=
(A B C D : Point)

-- Function to compute the midpoint of a segment
def midpoint (p1 p2 : Point) : Point := 
  sorry  -- Implement the function to find the midpoint

-- Function to compute the length of a segment between two points
def segment_length (p1 p2 : Point) : ℝ :=
  sorry  -- Implement this function to find the length of the segment

-- Function to check if two segments are perpendicular
def perpendicular (p1 p2 p3 p4 : Point) : Prop :=
  sorry  -- Implement this function to check if two segments are perpendicular

-- a) Given: segments connecting midpoints of opposite sides of the quadrilateral are equal
-- Prove: the diagonals of the quadrilateral are perpendicular
theorem diagonals_perpendicular_if_midsegments_equal (q : Quadrilateral) :
  (segment_length (midpoint q.A q.B) (midpoint q.C q.D) = segment_length (midpoint q.B q.C) (midpoint q.D q.A)) →
  perpendicular q.A q.C q.B q.D :=
by
  sorry

-- b) Given: diagonals of the quadrilateral are perpendicular
-- Prove: segments connecting midpoints of opposite sides are equal
theorem midsegments_equal_if_diagonals_perpendicular (q : Quadrilateral) :
  perpendicular q.A q.C q.B q.D →
  (segment_length (midpoint q.A q.B) (midpoint q.C q.D) = segment_length (midpoint q.B q.C) (midpoint q.D q.A)) :=
by
  sorry

end diagonals_perpendicular_if_midsegments_equal_midsegments_equal_if_diagonals_perpendicular_l292_292449


namespace problem_statement_l292_292060

noncomputable def vector_a := ℝ × ℝ
noncomputable def vector_b := ℝ × ℝ
noncomputable def vector_m := ℝ × ℝ
noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2
noncomputable def norm_squared (v : ℝ × ℝ) : ℝ := v.1 * v.1 + v.2 * v.2

theorem problem_statement
  (a b : ℝ × ℝ)
  (m : vector_m)
  (h_m : m = (4, 9))
  (h_midpoint : m = ((a.1 + b.1) / 2, (a.2 + b.2) / 2))
  (h_dot : dot_product a b = 10) :
  norm_squared a + norm_squared b = 368 :=
begin
  sorry
end

end problem_statement_l292_292060


namespace books_at_beginning_l292_292962

-- Definitions based on the conditions
def pct_returned : Real := 0.65
def books_at_end : Nat := 122
def books_loaned : Nat := 80

-- Prove the number of books at the beginning of the month is 150
theorem books_at_beginning (B : Nat) (h1 : pct_returned * books_loaned = frac 65 100 * books_loaned)
                            (h2 : books_at_end = 122)
                            (h3 : books_loaned = 80)
                            (h4 : B = books_at_end + (1 - pct_returned) * books_loaned) : 
                            B = 150 :=
by
  -- Additional details would be filled by the actual proof here
  sorry

end books_at_beginning_l292_292962


namespace values_for_a_l292_292349

def has_two (A : Set ℤ) : Prop :=
  2 ∈ A

def candidate_values (a : ℤ) : Set ℤ :=
  {-2, 2 * a, a * a - a}

theorem values_for_a (a : ℤ) :
  has_two (candidate_values a) ↔ a = 1 ∨ a = 2 :=
by
  sorry

end values_for_a_l292_292349


namespace range_of_a_l292_292714

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 :=
sorry

end range_of_a_l292_292714


namespace star_sum_interior_angles_l292_292113

theorem star_sum_interior_angles (n : ℕ) (h : n ≥ 6) :
  let S := 180 * n - 360
  S = 180 * (n - 2) :=
by
  let S := 180 * n - 360
  show S = 180 * (n - 2)
  sorry

end star_sum_interior_angles_l292_292113


namespace area_of_triangle_DEF_l292_292406

def point (α : Type) := α × α
def triangle (α : Type) := α × α × α

variable {α : Type} [linear_ordered_field α]

noncomputable def distance (A B : point α) : α :=
  real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

-- Conditions
variables (D E F L : point α)
variable (hL : ∃ x : α, L.1 = x ∧ L.2 = 0 ∧ E.1 < x ∧ x < F.1)
variable (h_EL : distance E L = 9)
variable (h_DE : distance D E = 15)
variable (h_EF : distance E F = 17)
variable (h_altitude : L.1 = ((E.1 + F.1) / 2) ∧ D.2 = real.sqrt (h_DE * h_DE - h_EL * h_EL))

-- The proof problem
theorem area_of_triangle_DEF : 
  ∃ (A : α), A = (1/2) * 17 * 12 ∧ A = 102 :=
by
  sorry

end area_of_triangle_DEF_l292_292406


namespace exists_rectangle_area_perimeter_int_exists_rectangle_area_perimeter_diagonal_int_l292_292610

-- Define the conditions
def conditions (m n : ℕ) : Prop :=
  n < m^2 ∧ ¬is_square n

-- Question (a)
theorem exists_rectangle_area_perimeter_int :
  ∃ (m n : ℕ), conditions m n ∧
  ∃ (a b : ℝ), a = m + Real.sqrt n ∧ b = m - Real.sqrt n ∧
  (a * b).denom = 1 ∧ (2 * a + 2 * b).denom = 1 :=
sorry

-- Question (b)
theorem exists_rectangle_area_perimeter_diagonal_int :
  ∃ (m n : ℕ), conditions m n ∧ 2 * (m^2 + n) = (k:ℤ)^2 ∧
  ∃ (a b d : ℝ), a = m + Real.sqrt n ∧ b = m - Real.sqrt n ∧
  (a * b).denom = 1 ∧ (2 * a + 2 * b).denom = 1 ∧
  d = Real.sqrt (2 * (m^2 + n)) ∧ d.denom = 1 :=
sorry

end exists_rectangle_area_perimeter_int_exists_rectangle_area_perimeter_diagonal_int_l292_292610


namespace constant_term_is_1440_l292_292135

noncomputable def constant_term_expansion (a : ℝ) : ℝ :=
  let f := (λ x : ℝ, (x + a / x) * (3 * x - 2 / x) ^ 5) in
  sorry -- The actual expansion calculation goes here

theorem constant_term_is_1440 (h : (constant_term_expansion 2) = 1440) : true :=
by
  have h1 : ∑ c in (range (nat_degree_expansion (3 * x - 2 / x))) = 3 := sorry
  have a := 2
  have sum_of_coeffs : (constant_term_expansion a) = 3 := sorry
  have binomial_expansion : (x + 2 / x) * (3 * x - 2 / x) ^ 5 =  1440 := sorry

  trivial -- This is just a placeholder to complete the theorem format

end constant_term_is_1440_l292_292135


namespace gain_percent_l292_292919

def cycle_gain_percent (cp sp : ℕ) : ℚ :=
  (sp - cp) / cp * 100

theorem gain_percent {cp sp : ℕ} (h1 : cp = 1500) (h2 : sp = 1620) : cycle_gain_percent cp sp = 8 := by
  sorry

end gain_percent_l292_292919


namespace units_digit_of_sum_of_cubes_l292_292900

theorem units_digit_of_sum_of_cubes : 
  (24^3 + 42^3) % 10 = 2 := by
sorry

end units_digit_of_sum_of_cubes_l292_292900


namespace max_value_ab_bc_cd_da_l292_292062

theorem max_value_ab_bc_cd_da (a b c d : ℝ) (a_nonneg : 0 ≤ a) (b_nonneg : 0 ≤ b) (c_nonneg : 0 ≤ c)
  (d_nonneg : 0 ≤ d) (sum_eq_200 : a + b + c + d = 200) : 
  ab + bc + cd + 0.5 * d * a ≤ 11250 := 
sorry


end max_value_ab_bc_cd_da_l292_292062


namespace calculate_expression_l292_292585

theorem calculate_expression : 
  ((13^13 / 13^12)^3 * 3^3) / 3^6 = 27 :=
by
  sorry

end calculate_expression_l292_292585


namespace max_value_expression_l292_292002

theorem max_value_expression (a b c : ℝ) 
  (ha : 300 ≤ a ∧ a ≤ 500) 
  (hb : 500 ≤ b ∧ b ≤ 1500) 
  (hc : c = 100) : 
  (∃ M, M = 8 ∧ ∀ x, x = (b + c) / (a - c) → x ≤ M) := 
sorry

end max_value_expression_l292_292002


namespace prop_1_prop_2_prop_3_prop_4_correct_props_l292_292581

noncomputable def f (x b c : ℝ) : ℝ := x * |x| + b * x + c

/-- Proposition 1: If c = 0, then f(x) is an odd function. -/
theorem prop_1 (b : ℝ) : ∀ x : ℝ, f x b 0 = - f (-x) b 0 := sorry

/-- Proposition 2: If b = 0, then f(x) is increasing over ℝ. -/
theorem prop_2 (c : ℝ) : ∀ x y : ℝ, x < y → f x 0 c < f y 0 c := sorry

/-- Proposition 3: The graph of y = f(x) is centrally symmetric about the point (0, c). -/
theorem prop_3 (b c : ℝ) : ∀ x : ℝ, f x b c = 2 * c - f (-x) b c := sorry

/-- Proposition 4: The equation f(x) = 0 has at most two real roots. -/
theorem prop_4 (b c : ℝ) : ∀ x1 x2 x3 : ℝ, 
  f x1 b c = 0 → f x2 b c = 0 → f x3 b c = 0 → (x1 = x2 ∨ x2 = x3 ∨ x1 = x3) := sorry

/-- Correct propositions are 1, 2, and 3. -/
theorem correct_props (b c : ℝ) : prop_1 b ∧ prop_2 c ∧ prop_3 b c ∧ ¬ prop_4 b c := sorry

end prop_1_prop_2_prop_3_prop_4_correct_props_l292_292581


namespace mouse_shortest_path_on_cube_l292_292198

noncomputable def shortest_path_length (edge_length : ℝ) : ℝ :=
  2 * edge_length * Real.sqrt 2

theorem mouse_shortest_path_on_cube :
  shortest_path_length 2 = 4 * Real.sqrt 2 :=
by
  sorry

end mouse_shortest_path_on_cube_l292_292198


namespace fraction_meaningful_iff_l292_292374

theorem fraction_meaningful_iff (x : ℝ) : (∃ y, y = x / (x - 2)) ↔ x ≠ 2 :=
by
  sorry

end fraction_meaningful_iff_l292_292374


namespace total_donated_area_l292_292277

structure Cloth (total_area : ℝ) (pieces : list ℝ)

def donated_area (pieces : list ℝ) (total_area : ℝ) (indices : list ℕ) : ℝ :=
  ∑ i in indices, pieces.nth i * total_area

def cloth1 : Cloth :=
  { total_area := 100,
    pieces := [0.375, 0.255, 0.22, 0.15] }

def cloth2 : Cloth :=
  { total_area := 65,
    pieces := [0.487, 0.323, 0.19] }

def cloth3 : Cloth :=
  { total_area := 48,
    pieces := [0.295, 0.272, 0.238, 0.195] }

theorem total_donated_area :
  let d1 := donated_area cloth1.pieces cloth1.total_area [1, 2, 3],
      d2 := donated_area cloth2.pieces cloth2.total_area [1, 2],
      d3 := donated_area cloth3.pieces cloth3.total_area [1, 2, 3]
  in d1 + d2 + d3 = 129.685 :=
by {
  sorry
}

end total_donated_area_l292_292277


namespace fibonacci_formula_correct_l292_292602

def fibonacci (n : ℕ) : ℕ
| 0 := 0
| 1 := 1
| (n+2) := fibonacci n + fibonacci (n+1)

noncomputable def phi := (1 + Real.sqrt 5) / 2
noncomputable def phi' := (1 - Real.sqrt 5) / 2

noncomputable def fib_formula (n : ℕ) : ℝ := (phi^n - phi'^n) / Real.sqrt 5

theorem fibonacci_formula_correct (n : ℕ) : 
  (fibonacci (n+1) : ℝ) = fib_formula n := 
by 
  sorry

end fibonacci_formula_correct_l292_292602


namespace solve_congruence_l292_292107

-- Define the condition and residue modulo 47
def residue_modulo (a b n : ℕ) : Prop := (a ≡ b [MOD n])

-- The main theorem to be proved
theorem solve_congruence (m : ℕ) (h : residue_modulo (13 * m) 9 47) : residue_modulo m 26 47 :=
sorry

end solve_congruence_l292_292107


namespace triangle_midpoint_proof_l292_292090

theorem triangle_midpoint_proof
  (A B C C1 B1 M : Point)
  (hA : Triangle A B C)
  (hC1_perp : ∠C1 = 90)
  (hB1_perp : ∠B1 = 90)
  (hABC1_angle : ∠ABC1 = φ)
  (hACB1_angle : ∠ACB1 = φ)
  (hM_midpoint : M = midpoint B C) :
  (dist M B1 = dist M C1) ∧ (angle B1 M C1 = 2 * φ) :=
by
  sorry

end triangle_midpoint_proof_l292_292090


namespace find_smallest_y_l292_292200

noncomputable def x : ℕ := 5 * 15 * 35

def is_perfect_fourth_power (n : ℕ) : Prop :=
  ∃ m : ℕ, m ^ 4 = n

theorem find_smallest_y : ∃ y : ℕ, y > 0 ∧ is_perfect_fourth_power (x * y) ∧ y = 46485 := by
  sorry

end find_smallest_y_l292_292200


namespace yearly_feeding_cost_l292_292693

-- Defining the conditions
def num_geckos := 3
def num_iguanas := 2
def num_snakes := 4

def cost_per_snake_per_month := 10
def cost_per_iguana_per_month := 5
def cost_per_gecko_per_month := 15

-- Statement of the proof problem
theorem yearly_feeding_cost : 
  (num_snakes * cost_per_snake_per_month + num_iguanas * cost_per_iguana_per_month + num_geckos * cost_per_gecko_per_month) * 12 = 1140 := 
  by 
    sorry

end yearly_feeding_cost_l292_292693


namespace four_cells_different_colors_l292_292311

theorem four_cells_different_colors
  (n : ℕ)
  (h_n : n ≥ 2)
  (coloring : Fin n → Fin n → Fin (2 * n)) :
  ∃ (r1 r2 c1 c2 : Fin n),
    r1 ≠ r2 ∧ c1 ≠ c2 ∧
    (coloring r1 c1 ≠ coloring r1 c2) ∧
    (coloring r1 c1 ≠ coloring r2 c1) ∧
    (coloring r1 c2 ≠ coloring r2 c2) ∧
    (coloring r2 c1 ≠ coloring r2 c2) := 
sorry

end four_cells_different_colors_l292_292311


namespace find_first_number_l292_292531

noncomputable def first_number : ℝ :=
  let percentage : ℝ := 16.666 / 100
  let other_number : ℝ := 9032
  let target_sum : ℝ := 10500
  let part_of_other : ℝ := percentage * other_number
  target_sum - part_of_other

theorem find_first_number :
  (10500 : ℝ) = first_number + ((16.666 / 100) * 9032) :=
by 
  have h1 : (10500 : ℝ) = 10500 := by rfl
  have h2 : ((16.666 / 100) * 9032).floor ≈ 1505 := by linarith
  have h3 : (10500 - 1505) ≈ 8995 := by linarith
  apply h3

end find_first_number_l292_292531


namespace probability_six_faces_each_appear_at_least_once_l292_292160

noncomputable def probability_all_faces_appear_at_least_once_when_rolling_ten_dice : ℝ :=
  let term1 := (6 * ((5: ℝ) / 6) ^ 10)
  let term2 := (15 * ((4: ℝ) / 6) ^ 10)
  let term3 := (20 * ((3: ℝ) / 6) ^ 10)
  let term4 := (15 * ((2: ℝ) / 6) ^ 10)
  let term5 := (6 * ((1: ℝ) / 6) ^ 10)
  1 - (term1 - term2 + term3 - term4 + term5)

theorem probability_six_faces_each_appear_at_least_once :
  probability_all_faces_appear_at_least_once_when_rolling_ten_dice ≈ 0.272 :=
  sorry

end probability_six_faces_each_appear_at_least_once_l292_292160


namespace ellipse_properties_l292_292675

-- Definitions of conditions
def ellipse_eq (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2 + y^2 / b^2 = 1)
def passes_through_A (b : ℝ) : Prop := (b = 4)
def eccentricity (a b : ℝ) : Prop := (sqrt (1 - b^2 / a^2) = 3 / 5)
def line_eq (x y : ℝ) : Prop := (y = 4/5 * (x - 3))

-- Proving the queries given the conditions
theorem ellipse_properties (a b : ℝ) : 
  (a > b ∧ b > 0 
  ∧ passes_through_A b 
  ∧ eccentricity a b) 
  ∧ (∀ x y, line_eq x y → ellipse_eq 5 4 x y) 
  → (ellipse_eq 5 4 0 4 
  ∧ ∀ x1 x2 y1 y2, x1 = (3 - sqrt 41) / 2 → x2 = (3 + sqrt 41) / 2 
  → y1 = 4 / 5 * (x1 - 3) → y2 = 4 / 5 * (x2 - 3) 
  → (1 / 2 * (x1 + x2) = 3 / 2 ∧ 1 / 2 * (y1 + y2) = -6 / 5)) :=
by
  sorry

end ellipse_properties_l292_292675


namespace domain_g_solution_set_g_leq_zero_l292_292334

-- Given conditions
def domain_f (x : ℝ) : Prop := -2 < x ∧ x < 2

def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x - 1) + f (3 - 2 * x)

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

def is_monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y ≤ f x

-- Proof problems
theorem domain_g (f : ℝ → ℝ) (h_domain : ∀ x, domain_f (f x))
  (h_def_g : ∀ x, domain_f x) :
  set_of (λ x, ∃ y, f (x - 1) + f (3 - 2 * x)) = set.Ioo 0.5 2.5 := sorry

theorem solution_set_g_leq_zero (f : ℝ → ℝ) (h_odd : is_odd_function f)
  (h_mono : is_monotonically_decreasing f) :
  set_of (λ x, (f (x - 1) + f (3 - 2 * x)) ≤ 0) = set.Ioc 0.5 2 := sorry

end domain_g_solution_set_g_leq_zero_l292_292334


namespace sin_C_value_l292_292716

theorem sin_C_value (a c : ℝ) (A : ℝ) (h_a : a = 7) (h_c : c = 3) (h_A : A = π / 3) :
  sin ((π / 3)) * c / a = sin  (((π / 3) * c) / a) :=
by
  rw [h_a, h_c, h_A]
  simpa using h_eq

end sin_C_value_l292_292716


namespace angle_ABC_l292_292132

theorem angle_ABC (D : Point) (A B C : Point) 
  (h_incenter : incenter D A B C)
  (h_angle_BAC : ∠BAC = 45)
  (h_angle_ACB : ∠ACB = 40) : 
  ∠ABC = 95 := by
  sorry

end angle_ABC_l292_292132


namespace find_angle_x_l292_292841

-- Definitions for geometric constructs involved in the problem
def is_regular_pentagon (P : Type) [geometry P] (p : set P) : Prop :=
  is_polygon p ∧ regular_polygon p ∧ polygon_sides p = 5

def is_equilateral_triangle (T : Type) [geometry T] (t : set T) : Prop :=
  is_polygon t ∧ equilateral_polygon t ∧ polygon_sides t = 3

def is_square (S : Type) [geometry S] (s : set S) : Prop :=
  is_quadrilateral s ∧ square s

-- The main conjecture to be proven
theorem find_angle_x
  (P Q S: Type) [geometry P] [geometry Q] [geometry S]
  (p : set P) (t : set Q) (sq : set S)
  (AB : line) 
  (h1 : is_regular_pentagon P p) 
  (h2 : is_equilateral_triangle Q t)
  (h3 : is_square S sq)
  (h4 : parallel AB (line_base sq))
  : ∃ x : ℝ, x = 24 :=
by
  sorry

end find_angle_x_l292_292841


namespace train_passing_time_l292_292920

-- Define the problem parameters
def length_of_train : ℝ := 288  -- in meters
def length_of_bridge : ℝ := 101  -- in meters
def speed_of_train_kmh : ℝ := 29  -- in km/hour

-- Convert speed from km/h to m/s
def speed_of_train_ms : ℝ := speed_of_train_kmh * 1000 / 3600

-- Total distance to be covered by the train
def total_distance : ℝ := length_of_train + length_of_bridge

-- Time required to pass the bridge
def time_to_pass_bridge : ℝ := total_distance / speed_of_train_ms

-- Proof problem statement
theorem train_passing_time : abs (time_to_pass_bridge - 48.3) < 0.1 :=
by
  sorry

end train_passing_time_l292_292920


namespace equal_areas_of_pentagon_and_heptagon_l292_292224

def A (s : ℝ) : ℝ :=
  let θ := real.pi / 5
  let apothem := real.cos θ * s / (2 * real.sin θ)
  let circum_radius := s / (2 * real.sin θ)
  real.pi * (circum_radius ^ 2 - apothem ^ 2)

def B (s : ℝ) : ℝ :=
  let θ := real.pi / 7
  let apothem := real.cos θ * s / (2 * real.sin θ)
  let circum_radius := s / (2 * real.sin θ)
  real.pi * (circum_radius ^ 2 - apothem ^ 2)

theorem equal_areas_of_pentagon_and_heptagon (s : ℝ) (hs : s = 2) : A s = B s := by
  have h1 : A 2 = real.pi := sorry
  have h2 : B 2 = real.pi := sorry
  rw [←hs] at *
  exact (eq.trans h1 h2).symm

end equal_areas_of_pentagon_and_heptagon_l292_292224


namespace hexagon_sum_of_squares_l292_292930

variables {V : Type*} [inner_product_space ℝ V]
variables {a b c d e f : V}

-- Conditions: Convex hexagon with parallel sides
variables (h1 : is_convex_hexagon a b c d e f)
variables (h2 : parallel (c - a) (f - d))
variables (h3 : parallel (d - b) (e - a))
variables (h4 : parallel (e - c) (f - b))

theorem hexagon_sum_of_squares :
  ∥b - a∥^2 + ∥d - c∥^2 + ∥f - e∥^2 = ∥c - b∥^2 + ∥e - d∥^2 + ∥f - a∥^2 :=
sorry

end hexagon_sum_of_squares_l292_292930


namespace problem_B_false_l292_292249

def diamondsuit (x y : ℝ) : ℝ := abs (x + y - 1)

theorem problem_B_false : ∀ x y : ℝ, 2 * (diamondsuit x y) ≠ diamondsuit (2 * x) (2 * y) :=
by
  intro x y
  dsimp [diamondsuit]
  sorry

end problem_B_false_l292_292249


namespace max_value_dot_product_l292_292669

open Real EuclideanGeometry

theorem max_value_dot_product (a b c : ℝ × ℝ) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hc : ‖c‖ = 1)
    (hab : dot_product a b = 0) : max ((a - c) • (b - c)) = 1 + sqrt 2 :=
by
  sorry

end max_value_dot_product_l292_292669


namespace floor_function_inequality_l292_292481

theorem floor_function_inequality (n : ℕ) (h : n > 0) :
  let I := (n + 1)^2 + n - (⌊Real.sqrt ((n + 1)^2 + n + 1)⌋)^2
  in I > 0 :=
by
  let I := (n + 1)^2 + n - (⌊Real.sqrt ((n + 1)^2 + n + 1)⌋)^2
  sorry

end floor_function_inequality_l292_292481


namespace range_of_a_g_function_l292_292345

noncomputable def f (x: ℝ) (a: ℝ) := 2 * |x - 2| + ax

def piecewise_f (x: ℝ) (a: ℝ) :=
  if x >= 2 then 
    (a + 2) * x - 4 
  else 
    (a - 2) * x + 4

def g (x: ℝ) (a: ℝ) :=
  if x < 0 then 
    (a - 2) * x + 4 
  else if x = 0 then 
    0 
  else 
    (a - 2) * x - 4

theorem range_of_a (a: ℝ) : 
  (∃ x: ℝ, (f x a < f (x+1) a ∧ f x a < f (x-1) a)) ↔ (-2 ≤ a ∧ a ≤ 2) := 
sorry

theorem g_function (a: ℝ) (x: ℝ) : 
  g x a = 
  if x < 0 then 
    (a - 2) * x + 4 
  else if x = 0 then 
    0 
  else 
    (a - 2) * x - 4 := 
sorry

end range_of_a_g_function_l292_292345


namespace find_line_equation_l292_292494

theorem find_line_equation (k m b : ℝ) :
  (∃ k, |(k^2 + 7*k + 10) - (m*k + b)| = 8) ∧ (8 = 2*m + b) ∧ (b ≠ 0) → (m = 5 ∧ b = 3) := 
by
  intro h
  sorry

end find_line_equation_l292_292494


namespace color_intersection_exists_l292_292468

theorem color_intersection_exists :
  ∃ (i j k l : ℕ), 1 ≤ i ∧ i ≤ 100 ∧ 1 ≤ j ∧ j ≤ 100 ∧ 1 ≤ k ∧ k ≤ 100 ∧ 1 ≤ l ∧ l ≤ 100 ∧ i ≠ k ∧ j ≠ l ∧
  let color : ℕ × ℕ → Fin 4 := λ _, sorry in
  color (i, j) ≠ color (i, l) ∧ color (i, j) ≠ color (k, j) ∧ color (i, l) ≠ color (k, l) ∧ 
  color (k, j) ≠ color (k, l) :=
begin
  sorry
end

end color_intersection_exists_l292_292468


namespace find_cost_per_pound_of_mixture_l292_292836

-- Problem Definitions and Conditions
variable (x : ℝ) -- the variable x represents the pounds of Spanish peanuts used
variable (y : ℝ) -- the cost per pound of the mixture we're trying to find
def cost_virginia_pound : ℝ := 3.50
def cost_spanish_pound : ℝ := 3.00
def weight_virginia : ℝ := 10.0

-- Formula for the cost per pound of the mixture
noncomputable def cost_per_pound_of_mixture : ℝ := (weight_virginia * cost_virginia_pound + x * cost_spanish_pound) / (weight_virginia + x)

-- Proof Problem Statement
theorem find_cost_per_pound_of_mixture (h : cost_per_pound_of_mixture x = y) : 
  y = (weight_virginia * cost_virginia_pound + x * cost_spanish_pound) / (weight_virginia + x) := sorry

end find_cost_per_pound_of_mixture_l292_292836


namespace pancake_cut_l292_292488

theorem pancake_cut (n : ℕ) (h : 3 ≤ n) :
  ∃ (cut_piece : ℝ), cut_piece > 0 :=
sorry

end pancake_cut_l292_292488


namespace number_of_male_athletes_selected_l292_292975

theorem number_of_male_athletes_selected (m f n : ℕ) (h1 : m = 28) (h2 : f = 21) (h3 : n = 14) :
  (m * n) / (m + f) = 8 :=
by
  rw [h1, h2, h3]
  norm_num
  congr
  sorry -- Further steps to refine the proof

end number_of_male_athletes_selected_l292_292975


namespace speaker_discounted_price_correct_l292_292358

-- Define the initial price and the discount
def initial_price : ℝ := 475.00
def discount : ℝ := 276.00

-- Define the discounted price
def discounted_price : ℝ := initial_price - discount

-- The theorem to prove that the discounted price is 199.00
theorem speaker_discounted_price_correct : discounted_price = 199.00 :=
by
  -- Proof is omitted here, adding sorry to indicate it.
  sorry

end speaker_discounted_price_correct_l292_292358


namespace tangent_line_y_intercept_is_1_l292_292305

-- Define the function f and its derivative
def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x
def f' (a : ℝ) (x : ℝ) : ℝ := a - 1 / x

-- Tangent line properties at x = 1
def tangent_point (a : ℝ) : ℝ × ℝ := (1, f a 1)
def slope_at_1 (a : ℝ) : ℝ := f' a 1
def tangent_line (a : ℝ) (x : ℝ) : ℝ := f a 1 + (f' a 1) * (x - 1)

-- Prove that the y-intercept of the tangent line is 1
theorem tangent_line_y_intercept_is_1 (a : ℝ) : tangent_line a 0 = 1 := by
  sorry

end tangent_line_y_intercept_is_1_l292_292305


namespace trains_crossing_time_l292_292500

-- Definitions for the conditions
def length_TrainA := 300 -- meters
def speed_TrainA_kmph := 160 -- kmph
def length_TrainB := 400 -- meters
def speed_TrainB_kmph := 180 -- kmph
def time_cross_pole_TrainA := 18 -- seconds

def kmph_to_mps (speed: ℕ) := speed * 1000 / 3600

-- Convert speed from kmph to m/s
def speed_TrainA_mps := kmph_to_mps speed_TrainA_kmph
def speed_TrainB_mps := kmph_to_mps speed_TrainB_kmph

-- Relative speed in the same direction
def relative_speed := speed_TrainB_mps - speed_TrainA_mps

-- Total length to be covered
def total_length_to_be_covered := length_TrainA + length_TrainB

-- Correct answer
def time_to_cross_each_other := 125.9 -- seconds

-- Proof statement
theorem trains_crossing_time :
  (total_length_to_be_covered / relative_speed) = time_to_cross_each_other :=
by
  sorry

end trains_crossing_time_l292_292500


namespace king_reach_squares_l292_292698

theorem king_reach_squares (k : ℕ) :
  (0 < k) →
  (∃ (f : ℤ × ℤ → ℕ), 
    (∀ pos : ℤ × ℤ, (f pos) = max (abs pos.1) (abs pos.2)) ∧ 
    (∃! pos, f pos = k)) →
  (∀ (f : ℤ × ℤ → ℕ), 
    (∀ pos : ℤ × ℤ, (f pos) = max (abs pos.1) (abs pos.2)) → 
    {pos : ℤ × ℤ | f pos = k}.card = 8 * k) :=
begin
  sorry
end

end king_reach_squares_l292_292698


namespace train_crossing_time_l292_292361

def length_of_train : ℝ := 250
def length_of_bridge : ℝ := 300
def speed_kmph : ℝ := 72

def speed_mps : ℝ := speed_kmph * (5 / 18)
def total_distance : ℝ := length_of_train + length_of_bridge
def time_to_cross : ℝ := total_distance / speed_mps

theorem train_crossing_time : time_to_cross = 27.5 := by
  sorry

end train_crossing_time_l292_292361


namespace units_digit_of_pow_sum_is_correct_l292_292906

theorem units_digit_of_pow_sum_is_correct : 
  (24^3 + 42^3) % 10 = 2 := by
  -- Start the proof block
  sorry

end units_digit_of_pow_sum_is_correct_l292_292906


namespace calculate_length_of_median_l292_292408

noncomputable def length_of_median_in_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (AB BC : ℝ) (BD : ℝ) : Option ℝ :=
some (sqrt (190) / 2)

theorem calculate_length_of_median {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (AB_len BC_len BD_len : ℝ) :
  AB_len = 8 → BC_len = 6 → BD_len = 6 → length_of_median_in_triangle A B C AB_len BC_len BD_len = some (sqrt (190) / 2) :=
by
  intros hAB hBC hBD
  have h1 : length_of_median_in_triangle A B C 8 6 6 = some (sqrt (190) / 2),
  sorry
  exact h1

end calculate_length_of_median_l292_292408


namespace cupboard_slots_l292_292980

theorem cupboard_slots (shelves_from_top shelves_from_bottom slots_from_left slots_from_right : ℕ)
  (h_top : shelves_from_top = 1)
  (h_bottom : shelves_from_bottom = 3)
  (h_left : slots_from_left = 0)
  (h_right : slots_from_right = 6) :
  (shelves_from_top + 1 + shelves_from_bottom) * (slots_from_left + 1 + slots_from_right) = 35 := by
  sorry

end cupboard_slots_l292_292980


namespace units_digit_sum_cubes_l292_292905

theorem units_digit_sum_cubes (n1 n2 : ℕ) 
  (h1 : n1 = 24) (h2 : n2 = 42) : 
  (n1 ^ 3 + n2 ^ 3) % 10 = 6 :=
by 
  -- substitution based on h1 and h2 can be done here.
  sorry

end units_digit_sum_cubes_l292_292905


namespace cannot_compute_l292_292835

noncomputable def g : ℝ → ℝ := sorry
axiom g_invertible : ∀ x : ℝ, (∃ y : ℝ, g y = x)

def f := g(g(1)) + g(g⁻¹ 2) + g⁻¹ (g⁻¹ 3)

theorem cannot_compute :
  (g(1) = 1) ∧ (g⁻¹ 3 = 2) ∧ (¬ ∃ x : ℝ, g x = 2) →
  f = "NEI" :=
by
  sorry

end cannot_compute_l292_292835


namespace time_taken_by_slower_train_to_pass_l292_292501

-- Definitions
def length_of_train : ℝ := 350 -- in meters
def speed_faster_train : ℝ := 65 -- in km/hr
def speed_slower_train : ℝ := 45 -- in km/hr

-- Auxiliary conversion definition
def convert_km_per_hr_to_m_per_s (speed: ℝ) : ℝ :=
  speed * (5 / 18) -- Conversion factor

-- Computation of relative speed
def relative_speed : ℝ :=
  (convert_km_per_hr_to_m_per_s speed_faster_train) + (convert_km_per_hr_to_m_per_s speed_slower_train)

-- Time Computation
def time_taken : ℝ :=
  length_of_train / relative_speed

-- Theorem
theorem time_taken_by_slower_train_to_pass :
  abs (time_taken - 11.44) < 0.01 :=
sorry

end time_taken_by_slower_train_to_pass_l292_292501


namespace min_colors_rect_condition_l292_292310

theorem min_colors_rect_condition (n : ℕ) (hn : n ≥ 2) :
  ∃ k : ℕ, (∀ (coloring : Fin n → Fin n → Fin k), 
           (∀ i j, coloring i j < k) → 
           (∀ c, ∃ i j, coloring i j = c) →
           (∃ i1 i2 j1 j2, i1 ≠ i2 ∧ j1 ≠ j2 ∧ 
                            coloring i1 j1 ≠ coloring i1 j2 ∧ 
                            coloring i1 j1 ≠ coloring i2 j1 ∧ 
                            coloring i1 j2 ≠ coloring i2 j2 ∧ 
                            coloring i2 j1 ≠ coloring i2 j2)) → 
           k = 2 * n :=
sorry

end min_colors_rect_condition_l292_292310


namespace max_proj_area_l292_292877

variable {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem max_proj_area : 
  ∃ max_area : ℝ, max_area = Real.sqrt (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) :=
by
  sorry

end max_proj_area_l292_292877


namespace proof_statement_l292_292792

-- Define the initial dimensions and areas
def initial_length : ℕ := 7
def initial_width : ℕ := 5

-- Shortened dimensions by one side and the corresponding area condition
def shortened_new_width : ℕ := 3
def shortened_area : ℕ := 21

-- Define the task
def task_statement : Prop :=
  (initial_length - 2) * initial_width = shortened_area ∧
  (initial_width - 2) * initial_length = shortened_area →
  (initial_length - 2) * (initial_width - 2) = 25

theorem proof_statement : task_statement :=
by {
  sorry -- Proof goes here
}

end proof_statement_l292_292792


namespace calculate_expression_l292_292234

theorem calculate_expression : 
  (27 ^ (1 / 3 : ℝ)) + ((real.sqrt 2 - 1) ^ 2) - (1 / (1 / 2 : ℝ)) + (2 / (real.sqrt 2 - 1)) = 6 := 
by 
  sorry

end calculate_expression_l292_292234


namespace anya_additional_hair_growth_l292_292226

noncomputable def washes_per_week : ℕ := 5
noncomputable def hair_loss_per_wash : ℕ := 32
noncomputable def brushes_per_week : ℕ := 7
noncomputable def growth_rate : ℕ := 70
noncomputable def growth_period : ℕ := 2
noncomputable def week_length : ℕ := 7

noncomputable def total_wash_loss : ℕ :=
  washes_per_week * hair_loss_per_wash

noncomputable def hair_loss_per_brush : ℕ :=
  hair_loss_per_wash / 2

noncomputable def total_brush_loss : ℕ :=
  brushes_per_week * hair_loss_per_brush

noncomputable def total_hair_loss : ℕ :=
  total_wash_loss + total_brush_loss

noncomputable def total_hair_growth : ℕ :=
  (week_length / growth_period) * growth_rate

noncomputable def needed_extra_growth : ℕ :=
  total_hair_loss - total_hair_growth + 1

theorem anya_additional_hair_growth : needed_extra_growth = 63 :=
  by sorry

end anya_additional_hair_growth_l292_292226


namespace base_area_of_cone_l292_292462

-- Given conditions
def height_of_cone : ℝ := 6
def volume_of_cone : ℝ := 60

-- The volume formula for a cone
def volume_formula (r h : ℝ) : ℝ := (1/3) * π * r^2 * h

-- Proving the problem statement
theorem base_area_of_cone : 
  (∃ r : ℝ, volume_formula r height_of_cone = volume_of_cone) → 
  (π * (30 / π) = 30) :=
by 
  sorry

end base_area_of_cone_l292_292462


namespace inscribe_circle_in_convex_polygon_l292_292818

theorem inscribe_circle_in_convex_polygon
  (S P r : ℝ) 
  (hP_pos : P > 0)
  (h_poly_area : S > 0)
  (h_nonneg : r ≥ 0) :
  S / P ≤ r :=
sorry

end inscribe_circle_in_convex_polygon_l292_292818


namespace infinite_quadriples_exists_l292_292821

theorem infinite_quadriples_exists : ∃ (f : ℕ → ℤ × ℤ × ℤ × ℤ), function.injective f ∧
  (∀ n, (let a := (f n).1 in
          let b := (f n).2.1 in
          let c := (f n).2.2.1 in
          let d := (f n).2.2.2 in
          a^2 + b^2 + 3 = 4 * a * b ∧ 
          c^2 + d^2 + 3 = 4 * c * d ∧ 
          4 * c^3 - 3 * c = a )) :=
sorry

end infinite_quadriples_exists_l292_292821


namespace product_greater_than_one_l292_292579

theorem product_greater_than_one (x : Fin 2015 → ℝ) (hx : ∀ i : Fin 2015, 0 < x i) 
  (h : ∀ i : Fin 2015, x i + x (i + 1) > (1 / x (i + 2)) + (1 / x (i + 3))) : 
  (∏ i, x i) > 1 := 
sorry

end product_greater_than_one_l292_292579


namespace teacher_zhang_friends_l292_292385

-- Define the conditions
def num_students : ℕ := 50
def both_friends : ℕ := 30
def neither_friend : ℕ := 1
def diff_in_friends : ℕ := 7

-- Prove that Teacher Zhang has 43 friends on social media
theorem teacher_zhang_friends : ∃ x : ℕ, 
  x + (x - diff_in_friends) - both_friends + neither_friend = num_students ∧ x = 43 := 
by
  sorry

end teacher_zhang_friends_l292_292385


namespace fraction_of_3_over_4_is_1_over_5_is_4_over_15_l292_292873

theorem fraction_of_3_over_4_is_1_over_5_is_4_over_15 : 
  (∃ x : ℚ, (x * (3/4) = (1/5)) ∧ x = (4/15)) := 
begin
  use 4/15,
  split,
  { -- show x * (3/4) = (1/5)
    sorry },
  { -- show x = 4/15
    refl },
end

end fraction_of_3_over_4_is_1_over_5_is_4_over_15_l292_292873


namespace example_problem_l292_292772

theorem example_problem
  (n : ℕ)
  (y : ℕ → ℤ)
  (h1 : ∀ i, (0 ≤ i ∧ i < n) → -1 ≤ y i ∧ y i ≤ 2)
  (h2 : ∑ i in finset.range n, y i = 23)
  (h3 : ∑ i in finset.range n, (y i)^2 = 105)
  : ∃ (max_ratio : ℤ), (max_ratio = 269 / 23) :=
by
  sorry

end example_problem_l292_292772


namespace value_after_5_years_l292_292617

noncomputable def investment_value_after_years 
  (initial_amount : ℝ) 
  (fractional_increase : ℕ → ℝ) 
  (interest_rate : ℝ) 
  (inflation_rate : ℝ)
  (years : ℕ) : ℝ :=
  let adjust_for_inflation (amount : ℝ) : ℝ := amount / (1 + inflation_rate)
  let adjust_for_interest (amount : ℝ) : ℝ := amount * (1 + interest_rate)
  let adjust_for_increase (year : ℕ) (amount : ℝ) : ℝ := amount * (1 + fractional_increase year)
  (List.range years).foldl 
    (λ amount year, 
      let increased_value := adjust_for_increase (year + 1) amount
      let interest_value := adjust_for_interest increased_value
      adjust_for_inflation interest_value) 
    initial_amount

theorem value_after_5_years
  (initial_amount : ℝ := 62000)
  (fractional_increase : ℕ → ℝ := λ year, 
    match year with 
    | 1 => 1/8 
    | 2 => 1/7 
    | 3 => 1/6 
    | 4 => 1/5 
    | 5 => 1/4 
    | _ => 0)
  (interest_rate : ℝ := 0.05)
  (inflation_rate : ℝ := 0.03)
  (years : ℕ := 5) :
  investment_value_after_years initial_amount fractional_increase interest_rate inflation_rate years ≈ 154655.73 :=
by
  sorry

end value_after_5_years_l292_292617


namespace divisor_of_930_l292_292158

theorem divisor_of_930 : ∃ d > 1, d ∣ 930 ∧ ∀ e, e ∣ 930 → e > 1 → d ≤ e :=
by
  sorry

end divisor_of_930_l292_292158


namespace hyunwoo_cookies_l292_292142

theorem hyunwoo_cookies (packs_initial : Nat) (pieces_per_pack : Nat) (packs_given_away : Nat)
  (h1 : packs_initial = 226) (h2 : pieces_per_pack = 3) (h3 : packs_given_away = 3) :
  (packs_initial - packs_given_away) * pieces_per_pack = 669 := 
by
  sorry

end hyunwoo_cookies_l292_292142


namespace factor_expression_l292_292276

theorem factor_expression (b : ℤ) : 52 * b ^ 2 + 208 * b = 52 * b * (b + 4) := 
by {
  sorry
}

end factor_expression_l292_292276


namespace exists_desired_chord_l292_292978

variables {α : Type*} [normed_group α] [normed_space ℝ α]

/-- Definitions of the Sphere, Plane and Point P given in conditions -/
structure Sphere :=
(center : α)
(radius : ℝ)

structure Plane :=
(normal : α)
(point : α)

structure Point :=
(coord : α)

variables (S : Sphere) (pl : Plane) (P : Point)

/-- Definition of Chord passing through P, bisected by the plane, and parallel to the first principal plane -/
def is_desired_chord (chord : α × α) : Prop :=
  let (Q1, Q2) := chord in
  Q1 ∈ S ∧ Q2 ∈ S ∧
  ∃ M : α, M = (Q1 + Q2) / 2 ∧ pl.normal • (M - pl.point) = 0

/-- The Proof Problem -/
theorem exists_desired_chord :
  ∃ Q1 Q2 : α, is_desired_chord S pl P (Q1, Q2) :=
sorry

end exists_desired_chord_l292_292978


namespace max_value_f_exp_inequality_f_l292_292766

-- Define f(x) function
def f (m : Nat) (x : ℝ) : ℝ := (2 * x / (x^2 + 1))^m

-- Define the maximum value condition as a theorem
theorem max_value_f (m : Nat) (hm : m > 1) :
  ∃ x ∈ (Set.Icc (0 : ℝ) ((m-1 : ℝ) / m)), f m x = (2 * m^2 - 2 * m + 1 : ℝ)^m :=
sorry

-- Define the exponential inequality condition as a theorem
theorem exp_inequality_f (m : Nat) (hm : m > 1) (x : ℝ) (hx : x ∈ Set.Icc (0 : ℝ) ((m-1 : ℝ) / m)) :
  exp (1 / (2 * m : ℝ)) * f m x < 1 :=
sorry

end max_value_f_exp_inequality_f_l292_292766


namespace true_propositions_l292_292114

def p : Prop := (∅ = {0})
def q : Prop := (7 ≥ 3)

theorem true_propositions : (p ∨ q) = true ∧ (¬ p) = true := by
  sorry

end true_propositions_l292_292114


namespace log_equation_solution_l292_292365

theorem log_equation_solution (x : ℝ) (h : log 3 (x ^ 3) + log (1 / 3) x = 6) : x = 27 := by
  sorry

end log_equation_solution_l292_292365


namespace puppies_per_cage_l292_292546

theorem puppies_per_cage (initial_puppies : ℕ) (sold_puppies : ℕ) (remaining_puppies : ℕ) (cages : ℕ) (puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 102)
  (h2 : sold_puppies = 21)
  (h3 : remaining_puppies = initial_puppies - sold_puppies)
  (h4 : cages = 9)
  (h5 : puppies_per_cage = remaining_puppies / cages) : 
  puppies_per_cage = 9 := 
by
  -- The proof should go here
  sorry

end puppies_per_cage_l292_292546


namespace option_a_correct_option_b_correct_option_c_incorrect_option_d_correct_l292_292730

-- Definitions
def total_balls := 6
def total_red_balls := 4
def total_white_balls := 2

-- Probability of picking exactly one white ball out of 3 balls
theorem option_a_correct : 
  (comb 2 1 * comb 4 2) / comb 6 3 = 3 / 5 := 
sorry

-- Variance of the number of red balls picked in 6 picks with replacement
theorem option_b_correct :
  let p_red := total_red_balls / total_balls in
  let n := 6 in
  (n * p_red * (1 - p_red) = 4 / 3) :=
sorry

-- Probability of picking a red ball on the second pick after picking a red ball on the first pick without replacement
theorem option_c_incorrect :
  let p_first_red := total_red_balls / total_balls in
  let p_second_red_given_first_red := (total_red_balls - 1) / (total_balls - 1) in
  (p_second_red_given_first_red / p_first_red = 3 / 5) :=
sorry

-- Probability of picking at least one red ball in 3 picks with replacement
theorem option_d_correct :
  let p_not_red_each_time := 1 - total_red_balls / total_balls in
  (1 - p_not_red_each_time ^ 3 = 26 / 27) :=
sorry

end option_a_correct_option_b_correct_option_c_incorrect_option_d_correct_l292_292730


namespace sum_even_integers_less_than_102_l292_292889

theorem sum_even_integers_less_than_102 :
  ∑ k in Finset.filter (λ x => (even x) ∧ (0 < x) ∧ (x < 102)) (Finset.range 102), k = 2550 := sorry

end sum_even_integers_less_than_102_l292_292889


namespace units_digit_sum_cubes_l292_292904

theorem units_digit_sum_cubes (n1 n2 : ℕ) 
  (h1 : n1 = 24) (h2 : n2 = 42) : 
  (n1 ^ 3 + n2 ^ 3) % 10 = 6 :=
by 
  -- substitution based on h1 and h2 can be done here.
  sorry

end units_digit_sum_cubes_l292_292904


namespace error_percentage_calc_l292_292549

theorem error_percentage_calc (y : ℝ) (hy : y > 0) : 
  let correct_result := 8 * y
  let erroneous_result := y / 8
  let error := abs (correct_result - erroneous_result)
  let error_percentage := (error / correct_result) * 100
  error_percentage = 98 := by
  sorry

end error_percentage_calc_l292_292549


namespace true_propositions_l292_292662

-- Define the propositions as conditions
def proposition1 : Prop := ∀ (P₁ P₂ P : Plane), P₁ ≠ P₂ → (P₁ ∥ P ∧ P₂ ∥ P) → P₁ ∥ P₂
def proposition2 : Prop := ∀ (P₁ P₂ L : Line), P₁ ≠ P₂ → (P₁ ∥ L ∧ P₂ ∥ L) → P₁ ∥ P₂
def proposition3 : Prop := ∀ (P₁ P₂ P : Plane), P₁ ≠ P₂ → (P₁ ⊥ P ∧ P₂ ⊥ P) → P₁ ∥ P₂
def proposition4 : Prop := ∀ (P₁ P₂ L : Line), P₁ ≠ P₂ → (P₁ ⊥ L ∧ P₂ ⊥ L) → P₁ ∥ P₂

-- State the theorem to prove that propositions 1 and 4 are true
theorem true_propositions : proposition1 ∧ proposition4 :=
by sorry

end true_propositions_l292_292662


namespace cube_identity_l292_292364

theorem cube_identity (a : ℝ) (h : (a + 1/a) ^ 2 = 3) : a^3 + 1/a^3 = 0 := 
by
  sorry

end cube_identity_l292_292364


namespace derivative_at_zero_l292_292582

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≠ 0 then (real.cbrt (1 - 2 * x ^ 3 * real.sin (5 / x)) - 1 + x)
  else 0

-- State the theorem to be proved
theorem derivative_at_zero (h : has_deriv_at f 1 0) : 
  deriv f 0 = 1 := 
sorry

end derivative_at_zero_l292_292582


namespace cube_root_of_x_l292_292704

theorem cube_root_of_x {x : ℝ} (h : x^2 = 64) : (Real.cbrt x = 2) ∨ (Real.cbrt x = -2) :=
sorry

end cube_root_of_x_l292_292704


namespace ariel_fish_l292_292576

theorem ariel_fish (total_fish : ℕ) (male_fraction female_fraction : ℚ) (h1 : total_fish = 45) (h2 : male_fraction = 2 / 3) (h3 : female_fraction = 1 - male_fraction) : total_fish * female_fraction = 15 :=
by
  sorry

end ariel_fish_l292_292576


namespace same_product_quality_l292_292936

-- Define the probabilities of defective products for A
def pA : ℕ → ℝ
| 0 := 0.4
| 1 := 0.3
| 2 := 0.2
| 3 := 0.1
| _ := 0  -- Beyond 3, the probability is zero.

-- Define the probabilities of defective products for B
def pB : ℕ → ℝ
| 0 := 0.4
| 1 := 0.2
| 2 := 0.4
| 3 := 0
| _ := 0  -- Beyond 3, the probability is zero.

-- Define the expected value calculation
def expected_value (p : ℕ → ℝ) : ℝ :=
  (0 * p 0) + (1 * p 1) + (2 * p 2) + (3 * p 3)

-- The proof statement to be proved
theorem same_product_quality :
  expected_value pA = expected_value pB :=
by sorry

end same_product_quality_l292_292936


namespace ink_cartridge_15th_month_l292_292275

def months_in_year : ℕ := 12
def first_change_month : ℕ := 1   -- January is the first month

def nth_change_month (n : ℕ) : ℕ :=
  (first_change_month + (3 * (n - 1))) % months_in_year

theorem ink_cartridge_15th_month : nth_change_month 15 = 7 := by
  -- This is where the proof would go
  sorry

end ink_cartridge_15th_month_l292_292275


namespace omega_monotonic_decreasing_l292_292342

theorem omega_monotonic_decreasing (ω : ℝ) (hω : ω > 0) :
  (∀ x y ∈ Ioo (-π/2 : ℝ) (π/2), x < y → cos (ω * x) - sin (ω * x) > cos (ω * y) - sin (ω * y)) ↔ (ω ≤ 1/2) :=
by
  sorry

end omega_monotonic_decreasing_l292_292342


namespace white_wash_cost_l292_292177

noncomputable def room_length : ℝ := 25
noncomputable def room_width : ℝ := 15
noncomputable def room_height : ℝ := 12
noncomputable def door_height : ℝ := 6
noncomputable def door_width : ℝ := 3
noncomputable def window_height : ℝ := 4
noncomputable def window_width : ℝ := 3
noncomputable def num_windows : ℕ := 3
noncomputable def cost_per_sqft : ℝ := 3

theorem white_wash_cost :
  let wall_area := 2 * (room_length * room_height + room_width * room_height)
  let door_area := door_height * door_width
  let window_area := window_height * window_width
  let total_non_white_wash_area := door_area + ↑num_windows * window_area
  let white_wash_area := wall_area - total_non_white_wash_area
  let total_cost := white_wash_area * cost_per_sqft
  total_cost = 2718 :=  
by
  sorry

end white_wash_cost_l292_292177


namespace probability_of_two_blue_jellybeans_l292_292188

def total_jellybeans := 12
def red_jellybeans := 3
def blue_jellybeans := 4
def white_jellybeans := 5
def total_picks := 3

theorem probability_of_two_blue_jellybeans : 
  (∃ (n : ℚ), n = ( (nat.choose blue_jellybeans 2) * (nat.choose (total_jellybeans - blue_jellybeans) (total_picks - 2)) / (nat.choose total_jellybeans total_picks) )) ∧ n = 12 / 55 :=
by
  sorry

end probability_of_two_blue_jellybeans_l292_292188


namespace abs_x_minus_one_iff_log2_x_lt_one_l292_292519

theorem abs_x_minus_one_iff_log2_x_lt_one (x : ℝ) : (|x - 1| < 1) ↔ (log 2 x < 1) := sorry

end abs_x_minus_one_iff_log2_x_lt_one_l292_292519


namespace center_of_gravity_correct_l292_292280

noncomputable def center_of_gravity_coordinates : ℝ × ℝ :=
  let region := { p : ℝ × ℝ | p.2 = (1/2) * p.1^2 ∧ p.2 ≤ 2 }
  let integral_dA := 2 * (∫ x in 0..2, (2 - (1/2)*x^2))
  let integral_y_dA := 2 * (∫ x in 0..2, (2 - (x^4 / 8)))
  (0, integral_y_dA / integral_dA)

theorem center_of_gravity_correct :
  center_of_gravity_coordinates = (0, 1.2) :=
begin
  sorry  -- proof not provided as instructed
end

end center_of_gravity_correct_l292_292280


namespace angle_C_side_c_cos_2B_C_l292_292405

-- Define the problem conditions
variable {A B C : Real}
variable {a b c : Real}
variable (cos_eq : cos A * cos B = sin A * sin B - sqrt 2 / 2)
variable (side_b : b = 4)
variable (area_abc : 1 / 2 * a * b * sin C = 6)

-- Define the problems to be proved
theorem angle_C (h : cos_eq) : C = π / 4 :=
sorry

theorem side_c (h1 : cos_eq) (h2 : side_b) (h3 : area_abc) : c = sqrt 10 :=
sorry

theorem cos_2B_C (h1 : cos_eq) (h2 : side_b) (h3 : area_abc) : cos (2 * B - C) = sqrt 2 / 10 :=
sorry

end angle_C_side_c_cos_2B_C_l292_292405


namespace nondegenerate_ellipse_iff_l292_292609

theorem nondegenerate_ellipse_iff (k : ℝ) :
  (∃ x y : ℝ, x^2 + 9*y^2 - 6*x + 27*y = k) ↔ k > -117/4 :=
by
  sorry

end nondegenerate_ellipse_iff_l292_292609


namespace exists_covering_circle_l292_292769

noncomputable def cover_interior_of_polygon (P : set (ℝ × ℝ)) (is_polygon : non_self_intersecting P) (C : ℕ → set (ℝ × ℝ))
    (r : ℕ → ℝ) (covers_P : ∀ x ∈ P, ∃ i, x ∈ C i ∧ C i = ∀ {y}, ∥y - center (C i)∥ < r i) 
    (sum_r := ∑ i, r i) : Prop :=
∃ (O : ℝ × ℝ), ∀ x ∈ P, ∥x - O∥ < sum_r
   
theorem exists_covering_circle (P : set (ℝ × ℝ)) (is_polygon : non_self_intersecting P) 
    (C : ℕ → set (ℝ × ℝ)) (r : ℕ → ℝ)
    (covers_P : ∀ x ∈ P, ∃ i, x ∈ C i ∧ C i = {y | ∥y - center (C i)∥ < r i}) :
    ∃ (O : ℝ × ℝ), ∀ x ∈ P, ∥x - O∥ < ∑ i, r i :=
sorry

end exists_covering_circle_l292_292769


namespace magnitude_projection_l292_292331

variables (a b : ℝ^3)
variables (ha : ∥a∥ = 3) (hb : ∥b∥ = 5) (hab : inner a b = 12)

theorem magnitude_projection : ∥((inner a b) / ∥b∥)∥ = 12 / 5 :=
by 
  sorry

end magnitude_projection_l292_292331


namespace problem_solution_l292_292829

-- Define necessary conditions and recursion relationship
def a : ℕ → ℕ
| 0 := 1
| 1 := 3
| (n + 2) := 3 * (a (n + 1)) - (a n)

-- Define probability as a fraction
def probability_no_problem := a 6

def total_arrangements := 3^6

def is_relatively_prime (m n : ℕ) : Prop :=
  Nat.gcd m n = 1

noncomputable def m : ℕ := a 6
noncomputable def n : ℕ := total_arrangements

noncomputable def m_plus_n := m + n

theorem problem_solution : m_plus_n = 1106 :=
by
  have h1 : probability_no_problem = 377 := rfl
  have h2 : total_arrangements = 729 := rfl
  have h3 : is_relatively_prime 377 729 := by sorry
  rw [h1, h2]
  -- Assuming m and n are relatively prime
  exact sorry

end problem_solution_l292_292829


namespace yasna_finish_books_in_two_weeks_l292_292916

theorem yasna_finish_books_in_two_weeks (pages_book1 : ℕ) (pages_book2 : ℕ) (pages_per_day : ℕ) (days_per_week : ℕ) 
  (h1 : pages_book1 = 180) (h2 : pages_book2 = 100) (h3 : pages_per_day = 20) (h4 : days_per_week = 7) : 
  ((pages_book1 + pages_book2) / pages_per_day) / days_per_week = 2 := 
by
  sorry

end yasna_finish_books_in_two_weeks_l292_292916


namespace two_digit_integers_sum_divisible_by_difference_and_product_digits_l292_292162

theorem two_digit_integers_sum_divisible_by_difference_and_product_digits : 
  ∑ n in Finset.filter 
      (λ n : ℕ, (10 ≤ n ∧ n < 100 ∧ ∃ a b : ℕ, n = 10 * a + b ∧ a - b ∣ n ∧ a * b ∣ n)) 
      (Finset.range 100) = 73 :=
by 
  sorry

end two_digit_integers_sum_divisible_by_difference_and_product_digits_l292_292162


namespace bowl_cost_is_10_yuan_l292_292131

def sets : ℕ := 12
def bowls_per_set : ℕ := 2
def total_price : ℝ := 240.0

def total_bowls : ℕ := sets * bowls_per_set
def price_per_bowl : ℝ := total_price / total_bowls

theorem bowl_cost_is_10_yuan :
  price_per_bowl = 10 := by
  sorry

end bowl_cost_is_10_yuan_l292_292131


namespace find_digit_H_l292_292613

theorem find_digit_H (G H I J K L : ℕ) (hG : G ∈ {1, 2, 3, 4, 5, 6}) (hH : H ∈ {1, 2, 3, 4, 5, 6})
  (hI : I ∈ {1, 2, 3, 4, 5, 6}) (hJ : J ∈ {1, 2, 3, 4, 5, 6}) (hK : K ∈ {1, 2, 3, 4, 5, 6})
  (hL : L ∈ {1, 2, 3, 4, 5, 6}) (h_distinct : List.nodup [G, H, I, J, K, L])
  (sum_of_lines : (G + H + I) + (G + J + K) + (I + J + L) + (H + K) + (H + L + I) = 53)
  (sum_of_digits : G + H + I + J + K + L = 21) : H = 3 :=
sorry

end find_digit_H_l292_292613


namespace workers_complete_task_in_8_days_l292_292924

theorem workers_complete_task_in_8_days : 
  (1 / 24 : ℝ)  -- p's work rate per day
  ∧ (1 / 9 : ℝ)  -- q's work rate per day
  ∧ (1 / 12 : ℝ) -- r's work rate per day
  ∧ (1 / 18 : ℝ) -- s's work rate per day
  ∧ (∀ days: ℝ, days ≤ 3 → (q + r + s) * days = (3 / 4 : ℝ)) -- q, r and s complete 3/4 work together in 3 days
  ∧ (∀ remaining_task days: ℝ, (remaining_task = 1 / 4)
     → ((1 / 24) + (1 / 36) * 2 * (⌈days⌉) = 8)) -- proving that remaining work is done in 8 days
  :=
sorry

end workers_complete_task_in_8_days_l292_292924


namespace coloring_four_cells_with_diff_colors_l292_292316

theorem coloring_four_cells_with_diff_colors {n k : ℕ} (h : n ≥ 2) 
    (hk : k = 2 * n) 
    (color : fin n × fin n → fin k) 
    (hcolor : ∀ c, ∃ r c : fin k, ∃ a b : fin k, color (r, c) = a ∧ color (r, c) = b) :
    ∃ r1 r2 c1 c2, color (r1, c1) ≠ color (r1, c2) ∧ color (r1, c1) ≠ color (r2, c1) ∧
                    color (r2, c1) ≠ color (r2, c2) ∧ color (r1, c2) ≠ color (r2, c2) :=
by
  sorry

end coloring_four_cells_with_diff_colors_l292_292316


namespace prob_one_late_out_of_three_l292_292019

/--
In a seminar, each participant has a 1/50 chance of being late. 
Prove that the probability that, out of any three participants chosen at random, 
exactly one will be late while the others are on time is 5.8%.
-/
theorem prob_one_late_out_of_three (h : ∀ (p : ℕ), p = 1 / 50) :
  (3 * (1 / 50) * ((49 / 50) ^ 2) * 100) = 5.8 := 
begin
  sorry
end

end prob_one_late_out_of_three_l292_292019


namespace find_angle_between_vectors_l292_292641

variables (a b : ℝ) (θ : ℝ)
variable (cosθ : ℝ)

-- Assume conditions
def norm_a : ℝ := 1
def norm_b : ℝ := Real.sqrt 2
def norm_expr : ℝ := Real.sqrt 5

-- Define the equation based on the conditions
def cosine_theta_condition : Prop :=
  norm_expr ^ 2 = norm_a ^ 2 + 4 * norm_b ^ 2 + 4 * norm_a * norm_b * cosθ

-- Define the range of θ
def theta_range : Prop := 0 ≤ θ ∧ θ ≤ Real.pi

-- Establish the result to be proven
theorem find_angle_between_vectors :
  cosine_theta_condition a b θ cosθ ∧ theta_range θ → θ = 3 * Real.pi / 4 :=
by
  sorry

end find_angle_between_vectors_l292_292641


namespace isosceles_triangle_base_leg_ratio_l292_292660

theorem isosceles_triangle_base_leg_ratio (a b : ℝ) (h₁ : is_isosceles_triangle a b 36) :
  a / b = (Real.sqrt 5 - 1) / 2 := 
sorry

end isosceles_triangle_base_leg_ratio_l292_292660


namespace joseph_drives_more_l292_292045

-- Definitions for the problem
def v_j : ℝ := 50 -- Joseph's speed in mph
def t_j : ℝ := 2.5 -- Joseph's time in hours
def v_k : ℝ := 62 -- Kyle's speed in mph
def t_k : ℝ := 2 -- Kyle's time in hours

-- Prove that Joseph drives 1 more mile than Kyle
theorem joseph_drives_more : (v_j * t_j) - (v_k * t_k) = 1 := 
by 
  sorry

end joseph_drives_more_l292_292045


namespace erased_number_is_30_l292_292207

-- Definitions based on conditions
def consecutiveNumbers (start n : ℕ) : List ℕ :=
  List.range' start n

def erase (l : List ℕ) (x : ℕ) : List ℕ :=
  List.filter (λ y => y ≠ x) l

def average (l : List ℕ) : ℚ :=
  l.sum / l.length

-- Statement to prove
theorem erased_number_is_30 :
  ∃ n x, average (erase (consecutiveNumbers 11 n) x) = 23 ∧ x = 30 := by
  sorry

end erased_number_is_30_l292_292207


namespace infinite_solutions_x2_y3_z5_l292_292098

theorem infinite_solutions_x2_y3_z5 :
  ∃ (t : ℕ), ∃ (x y z : ℕ), x = 2^(15*t + 12) ∧ y = 2^(10*t + 8) ∧ z = 2^(6*t + 5) ∧ (x^2 + y^3 = z^5) :=
sorry

end infinite_solutions_x2_y3_z5_l292_292098


namespace Rachel_and_Mike_l292_292823

theorem Rachel_and_Mike :
  ∃ b c : ℤ,
    (∀ x : ℝ, |x - 3| = 4 ↔ (x = 7 ∨ x = -1)) ∧
    (∀ x : ℝ, (x - 7) * (x + 1) = 0 ↔ x * x + b * x + c = 0) ∧
    (b, c) = (-6, -7) := by
sorry

end Rachel_and_Mike_l292_292823


namespace books_read_l292_292086

theorem books_read (miles1 miles2 miles3 : ℝ) (rate1 rate2 rate3 : ℝ) : 
  miles1 = 680.5 ∧ rate1 = 360 ∧ 
  miles2 = 2346.7 ∧ rate2 = 520 ∧ 
  miles3 = 3960.3 ∧ rate3 = 640 → 
  let books1 := (miles1 / rate1).floor;
      books2 := (miles2 / rate2).floor;
      books3 := (miles3 / rate3).floor in
  books1 + books2 + books3 = 11 :=
by
  intros h
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := h
  let books1 := (680.5 / 360).floor
  let books2 := (2346.7 / 520).floor
  let books3 := (3960.3 / 640).floor
  have h_books1: books1 = 1 := by sorry
  have h_books2: books2 = 4 := by sorry
  have h_books3: books3 = 6 := by sorry
  rw [h_books1, h_books2, h_books3]
  norm_num
  exact eq.refl 11

end books_read_l292_292086


namespace first_hive_honey_production_l292_292042

theorem first_hive_honey_production (H : ℝ) (hive1_bees hive2_bees : ℝ) (hive2_honey_multiplier : ℝ) (total_honey : ℝ) :
  hive1_bees = 1000 →
  hive2_bees = 800 →
  hive2_honey_multiplier = 1.40 →
  total_honey = 2460 →
  1000 * H + 800 * 1.40 * H = 2460 →
  1000 * H ≈ 1160.38 := 
by
  sorry

end first_hive_honey_production_l292_292042


namespace probability_of_divisible_by_3_l292_292003

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_valid_number (n : ℕ) : Prop :=
  n >= 10 ∧ n < 100 ∧ is_prime_digit (n / 10) ∧ is_prime_digit (n % 10)

def digit_sum (n : ℕ) : ℕ :=
  n / 10 + n % 10

def divisible_by_3 (n : ℕ) : Prop :=
  digit_sum n % 3 = 0

theorem probability_of_divisible_by_3 :
  (∑ n in (Finset.filter (λ n, is_valid_number n) (Finset.range 100)).filter divisible_by_3, 1) * 16 =
  (∑ n in Finset.filter (λ n, is_valid_number n) (Finset.range 100), 1) * 5 :=
sorry

end probability_of_divisible_by_3_l292_292003


namespace problem_1_problem_2_problem_3_l292_292382

theorem problem_1 (total_students group_size : ℕ) (total_students = 60) (group_size = 4) :
  (P : ℚ) = (group_size : ℚ) / (total_students : ℚ) ∧
  (male_group_size female_group_size : ℕ) = (3,1) :=
by
sorry

theorem problem_2 (number_of_pairs pairs_with_female : ℕ)
  (number_of_pairs = 6) (pairs_with_female = 3) :
  (P : ℚ) = (pairs_with_female : ℚ) / (number_of_pairs : ℚ) :=
by
sorry

theorem problem_3 (data1 data2 : List ℚ)
  (data1 = [68, 70, 71, 72, 74]) (data2 = [69, 70, 70, 72, 74])
  (mean1 mean2 : ℚ) (mean1 = 71) (mean2 = 71)
  (variance1 variance2 : ℚ) (variance1 = 4) (variance2 = 3.2) :
  variance2 < variance1 :=
by
sorry

end problem_1_problem_2_problem_3_l292_292382


namespace probability_two_same_correct_l292_292869

def factorial (n : ℕ) : ℕ :=
nat.factorial n

def probability_two_same (n : ℕ) : ℝ :=
if n ≥ 7 then 1
else if 2 ≤ n ∧ n ≤ 6 then
  (6 ^ n - factorial 6 / factorial (6 - n)) / 6 ^ n
else
  0  -- Typically, the probability problem wouldn't be meaningful for n < 2.

theorem probability_two_same_correct (n : ℕ) :
  probability_two_same n = 
  if n ≥ 7 then 1 else 
    if 2 ≤ n ∧ n ≤ 6 then 
      (6 ^ n - factorial 6 / factorial (6 - n)) / 6 ^ n 
    else 0 := 
sorry

end probability_two_same_correct_l292_292869


namespace longest_side_of_triangle_l292_292123

-- Defining variables and constants
variables (x : ℕ)

-- Defining the side lengths of the triangle
def side1 := 7
def side2 := x + 4
def side3 := 2 * x + 1

-- Defining the perimeter of the triangle
def perimeter := side1 + side2 + side3

-- Statement of the main theorem
theorem longest_side_of_triangle (h : perimeter x = 36) : max side1 (max (side2 x) (side3 x)) = 17 :=
by sorry

end longest_side_of_triangle_l292_292123


namespace simplify_expression_and_find_ratio_l292_292828

theorem simplify_expression_and_find_ratio:
  ∀ (k : ℤ), (∃ (a b : ℤ), (a = 1 ∧ b = 3) ∧ (6 * k + 18 = 6 * (a * k + b))) →
  (1 : ℤ) / (3 : ℤ) = (1 : ℤ) / (3 : ℤ) :=
by
  intro k
  intro h
  sorry

end simplify_expression_and_find_ratio_l292_292828


namespace range_of_a_l292_292685

/-- Let \(A\) and \(B\) be sets defined as:
    \(A = \{x \mid x^2 + 4x = 0\} = \{0, -4\}\) and 
    \(B = \{x \mid x^2 + 2(a+1)x + a^2 - 1 = 0\}\).
    Prove that the range of values of \(a\) for which \(A \cap B = B\) is 
    \(a = 1\) or \(a \leq -1\). -/
theorem range_of_a (a : ℝ) :
  (∀ x, (x^2 + 4 * x = 0 → x ∈ ({0, -4} : Set ℝ)) ∧
        (x^2 + 2 * (a + 1) * x + a^2 - 1 = 0 → x ∈ Set.Union Set.Empty {0, -4})
        → (x = 0 ∨ x = -4)) ↔ (a = 1 ∨ a ≤ -1) :=
sorry

end range_of_a_l292_292685


namespace complement_of_M_in_U_l292_292523

namespace ProofProblem

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 4}

theorem complement_of_M_in_U : U \ M = {3, 5, 6} := by
  sorry

end ProofProblem

end complement_of_M_in_U_l292_292523


namespace math_proof_problem_l292_292318

variable (a : ℕ → ℕ)  -- sequence a_n
variable (b : ℕ → ℕ)  -- sequence b_n
variable (c : ℕ → ℕ)  -- sequence c_n, given by a_n * b_n
variable (S : ℕ → ℕ)  -- sum of the first n terms of a_n
variable (T : ℕ → ℕ)  -- sum of the first n terms of c_n

-- Given conditions
axiom a1 : a 1 = 3
axiom a_rec : ∀ n, a (n + 1) = 2 * S n + 3
axiom b1 : b 1 = 1
axiom b_rec : ∀ n, b n - b (n + 1) + 1 = 0
axiom S_def : ∀ n, S n = ∑ i in finset.range (n + 1), a i
axiom c_def : ∀ n, c n = a n * b n
axiom T_def : ∀ n, T n = ∑ i in finset.range (n + 1), c i

-- Proof problem statements
def general_formulas : Prop := 
  (∀ n, a n = 3 ^ n) ∧ (∀ n, b n = n)

def sum_of_terms : Prop := 
  ∀ n, T n = ((2 * n - 1) * 3 ^ (n + 1)) / 4 + 3 / 4

-- Conjecture to prove
theorem math_proof_problem : general_formulas ∧ sum_of_terms := 
  by 
    split
    -- proof for a_n and b_n general formulas
    sorry
    -- proof for T_n sum of first n terms of c_n
    sorry

end math_proof_problem_l292_292318


namespace farmer_profit_l292_292955

noncomputable def profit_earned : ℕ :=
  let pigs := 6
  let sale_price := 300
  let food_cost_per_month := 10
  let months_group1 := 12
  let months_group2 := 16
  let pigs_group1 := 3
  let pigs_group2 := 3
  let total_food_cost := (pigs_group1 * months_group1 * food_cost_per_month) + 
                         (pigs_group2 * months_group2 * food_cost_per_month)
  let total_revenue := pigs * sale_price
  total_revenue - total_food_cost

theorem farmer_profit : profit_earned = 960 := by
  unfold profit_earned
  sorry

end farmer_profit_l292_292955


namespace best_fitting_model_l292_292742

theorem best_fitting_model :
  ∀ R1 R2 R3 R4 : ℝ, 
  R1 = 0.21 → R2 = 0.80 → R3 = 0.50 → R4 = 0.98 → 
  abs (R4 - 1) < abs (R1 - 1) ∧ abs (R4 - 1) < abs (R2 - 1) 
    ∧ abs (R4 - 1) < abs (R3 - 1) :=
by
  intros R1 R2 R3 R4 hR1 hR2 hR3 hR4
  rw [hR1, hR2, hR3, hR4]
  exact sorry

end best_fitting_model_l292_292742


namespace cube_root_of_x_l292_292703

theorem cube_root_of_x {x : ℝ} (h : x^2 = 64) : (Real.cbrt x = 2) ∨ (Real.cbrt x = -2) :=
sorry

end cube_root_of_x_l292_292703


namespace color_map_with_two_colors_l292_292861

theorem color_map_with_two_colors (n : ℕ) (h : n ≥ 1) : 
  ∃ (f : ℕ → bool), ∀ r₁ r₂ : ℕ, (adjacent r₁ r₂) → (f r₁ ≠ f r₂) := 
sorry

end color_map_with_two_colors_l292_292861


namespace Lynne_bought_3_magazines_l292_292438

open Nat

def books_about_cats : Nat := 7
def books_about_solar_system : Nat := 2
def book_cost : Nat := 7
def magazine_cost : Nat := 4
def total_spent : Nat := 75

theorem Lynne_bought_3_magazines:
  let total_books := books_about_cats + books_about_solar_system
  let total_cost_books := total_books * book_cost
  let total_cost_magazines := total_spent - total_cost_books
  total_cost_magazines / magazine_cost = 3 :=
by sorry

end Lynne_bought_3_magazines_l292_292438


namespace A_intersection_B_is_expected_l292_292708

-- Define set A
def setA : Set ℝ := { x | 1 ≤ 3 ^ x ∧ 3 ^ x ≤ 81 }

-- Define set B
def setB : Set ℝ := { x | Real.log x (x^2 - x) > 1 }

-- Define the intersection of setA and setB
def intersectionAB : Set ℝ := setA ∩ setB

-- Define the expected result
def expectedResult : Set ℝ := { x | 2 < x ∧ x ≤ 4 }

-- State the theorem
theorem A_intersection_B_is_expected :
  intersectionAB = expectedResult :=
sorry

end A_intersection_B_is_expected_l292_292708


namespace triangle_inequality_l292_292448

-- Define that a, b, c are the lengths of the sides of the triangle and S is the area.
variables {a b c S : ℝ}
-- Define area S and sides a, b, c satisfy the properties of a triangle.
variables (triangle_sides : a > 0 ∧ b > 0 ∧ c > 0)
variables (area_def : S = (sqrt ((a + b + c) * (a + b - c) * (a - b + c) * (-a + b + c))) / 4)

theorem triangle_inequality (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ S > 0) :
  (a * b + b * c + c * a) / (4 * S) ≥ Real.cot (Real.pi / 6) :=
sorry

end triangle_inequality_l292_292448


namespace imaginary_part_of_complex_l292_292845

theorem imaginary_part_of_complex :
  (Complex.mul (2 * Complex.I) (Complex.inv (1 + Complex.I))).im = 1 := 
sorry

end imaginary_part_of_complex_l292_292845


namespace difference_legs_heads_l292_292017

-- Define the conditions
def number_of_cows : ℕ := 7
def number_of_chickens : ℕ
def legs (C H : ℕ) : ℕ := 4 * C + 2 * H
def heads (C H : ℕ) : ℕ := C + H

-- Statement: Prove that the difference between the number of legs and twice the number of heads is 14.
theorem difference_legs_heads (H : ℕ) : 
  let C := number_of_cows in
  legs C H - 2 * heads C H = 14 :=
by {
  sorry
}

end difference_legs_heads_l292_292017


namespace bud_age_uncle_age_relation_l292_292583

variable (bud_age uncle_age : Nat)

theorem bud_age_uncle_age_relation (h : bud_age = 8) (h0 : bud_age = uncle_age / 3) : uncle_age = 24 := by
  sorry

end bud_age_uncle_age_relation_l292_292583


namespace correct_operation_l292_292913

theorem correct_operation : 
  ∀ x y : ℝ, 
  (2 * x + 3 * x ≠ 5 * x^2) ∧
  ((-2 * x * y^2)^3 ≠ -6 * x^3 * y^6) ∧
  ((3 * x + 2 * y)^2 ≠ 9 * x^2 + 4 * y^2) ∧
  (((-x)^8 / (-x)^4) = x^4) :=
by
  intros x y
  split
  { intro h
    sorry }
  split
  { intro h
    sorry }
  split
  { intro h
    sorry }
  { sorry }

end correct_operation_l292_292913


namespace parallel_AD_BC_l292_292992

noncomputable def ConvexQuadrilateral (A B C D P Q : Type*) [Geometry A B C D P Q] :=
  let α := ∠ C D Q
  let β := ∠ C Q B
  let γ := ∠ B A P
  let δ := ∠ C P B
  let ε := ∠ A P B
  let ζ := ∠ D Q C
  α = β ∧ γ = δ ∧ ε = ζ

theorem parallel_AD_BC {A B C D P Q : Type*} [Geometry A B C D P Q]
  (h1 : ConvexQuadrilateral A B C D P Q)
  : parallel AD BC := 
by 
  sorry

end parallel_AD_BC_l292_292992


namespace find_total_numbers_l292_292116

theorem find_total_numbers (n : ℕ) (h1 : (20 : ℝ) * n = ∑ i in (finset.range n), i) 
  (h2 : 48 = ∑ i in finset.range 3, i)
  (h3 : 52 = 2 * 26) : 
  n = 5 := 
sorry

end find_total_numbers_l292_292116


namespace eval_expression_l292_292271

theorem eval_expression (a x : ℤ) (h : x = a + 9) : (x - a + 5) = 14 :=
by
  sorry

end eval_expression_l292_292271


namespace water_settles_at_34_cm_l292_292867

-- Conditions definitions
def h : ℝ := 40 -- Initial height of the liquids in cm
def ρ_w : ℝ := 1000 -- Density of water in kg/m^3
def ρ_o : ℝ := 700  -- Density of oil in kg/m^3

-- Given the conditions provided above,
-- prove that the new height level of water in the first vessel is 34 cm
theorem water_settles_at_34_cm :
  (40 / (1 + (ρ_o / ρ_w))) = 34 := 
sorry

end water_settles_at_34_cm_l292_292867


namespace khalil_dogs_l292_292442

theorem khalil_dogs (D : ℕ) (cost_dog cost_cat : ℕ) (num_cats total_cost : ℕ) 
  (h1 : cost_dog = 60)
  (h2 : cost_cat = 40)
  (h3 : num_cats = 60)
  (h4 : total_cost = 3600) :
  (num_cats * cost_cat + D * cost_dog = total_cost) → D = 20 :=
by
  intros h
  sorry

end khalil_dogs_l292_292442


namespace marla_colors_red_squares_l292_292791

-- Conditions
def total_rows : Nat := 10
def squares_per_row : Nat := 15
def total_squares : Nat := total_rows * squares_per_row

def blue_rows_top : Nat := 2
def blue_rows_bottom : Nat := 2
def total_blue_rows : Nat := blue_rows_top + blue_rows_bottom
def total_blue_squares : Nat := total_blue_rows * squares_per_row

def green_squares : Nat := 66
def red_rows : Nat := 4

-- Theorem to prove 
theorem marla_colors_red_squares : 
  total_squares - total_blue_squares - green_squares = red_rows * 6 :=
by
  sorry -- This skips the proof

end marla_colors_red_squares_l292_292791


namespace parallel_lines_perpendicular_planes_l292_292329

  variable {m n : Line}
  variable {α β : Plane}

  -- Conditions
  def different_lines (m n : Line) : Prop := m ≠ n
  def different_planes (α β : Plane) : Prop := α ≠ β
  def line_in_plane (n : Line) (β : Plane) : Prop := n ⊆ β

  -- Proof problem
  theorem parallel_lines_perpendicular_planes (hmn : m ∥ n) (hmα : m ⊥ α) (hnβ : n ⊆ β) (hml : different_lines m n) (hp : different_planes α β) :
    α ⊥ β :=
  by
    sorry
  
end parallel_lines_perpendicular_planes_l292_292329


namespace pq_perpendicular_ab_l292_292485

-- Define the quadrilateral ABCD
variables {A B C D : Type} [Inhabited A]

-- Assume ABCD is inscribed in a circle
variable (h1 : ∃ O r, circle O r ∧ [A, B, C, D] ⊆ set.circle O r)

-- Length conditions
variable (h2 : dist B C = dist D C)
variable (h3 : dist A B = dist A C)

-- Define midpoint P of arc CD not containing A
variables {P : Type} [Inhabited P]
variable (h4 : is_midpoint_of_arc_not_containing P C D A)

-- Define Q as the intersection of diagonals AC and BD
variables {Q : Type} [Inhabited Q]
variable (h5 : is_intersection_of_diagonals Q A C B D)

-- Prove that PQ is perpendicular to AB
theorem pq_perpendicular_ab 
  (h1 : ∃ O r, circle O r ∧ [A, B, C, D] ⊆ set.circle O r)
  (h2 : dist B C = dist D C)
  (h3 : dist A B = dist A C)
  (h4 : is_midpoint_of_arc_not_containing P C D A)
  (h5 : is_intersection_of_diagonals Q A C B D) : 
  is_perpendicular (line_through P Q) (line_through A B) := sorry

end pq_perpendicular_ab_l292_292485


namespace shortest_distance_is_zero_l292_292770

open Real

def line1 (t : ℝ) : ℝ × ℝ × ℝ :=
  (4 + 3 * t, 1 - t, 3 + 2 * t)

def line2 (s : ℝ) : ℝ × ℝ × ℝ :=
  (1 + 2 * s, 2 + 3 * s, 5 - 2 * s)

def distance_squared (P Q : ℝ × ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 + (P.3 - Q.3)^2

theorem shortest_distance_is_zero :
  ∃ (t s : ℝ), distance_squared (line1 t) (line2 s) = 0 :=
by
  sorry

end shortest_distance_is_zero_l292_292770


namespace probability_different_suits_correct_l292_292076

-- Definitions based on conditions
def cards_in_deck : ℕ := 52
def cards_picked : ℕ := 3
def first_card_suit_not_matter : Prop := True
def second_card_different_suit : Prop := True
def third_card_different_suit : Prop := True

-- Definition of the probability function
def probability_different_suits (cards_total : ℕ) (cards_picked : ℕ) : Rat :=
  let first_card_prob := 1
  let second_card_prob := 39 / 51
  let third_card_prob := 26 / 50
  first_card_prob * second_card_prob * third_card_prob

-- The theorem statement to prove the probability each card is of a different suit
theorem probability_different_suits_correct :
  probability_different_suits cards_in_deck cards_picked = 169 / 425 :=
by
  -- Proof should be written here
  sorry

end probability_different_suits_correct_l292_292076


namespace baseball_cards_per_pack_l292_292443

theorem baseball_cards_per_pack (cards_each : ℕ) (packs_total : ℕ) (total_cards : ℕ) (cards_per_pack : ℕ) :
    (cards_each = 540) →
    (packs_total = 108) →
    (total_cards = cards_each * 4) →
    (cards_per_pack = total_cards / packs_total) →
    cards_per_pack = 20 :=
by
  intros h1 h2 h3 h4
  sorry

end baseball_cards_per_pack_l292_292443


namespace cos_value_l292_292726

-- Definitions
def angle_A : ℝ := 90
def tan_C : ℝ := 4
def cos_C := real.cos

-- Problem statement
theorem cos_value (A C : ℝ) (hA : A = 90) (hC : real.tan C = 4) :
  cos_C C = sqrt 17 / 17 :=
sorry

end cos_value_l292_292726


namespace minimum_score_required_for_fifth_term_to_average_85_l292_292865

def average_score (scores : List ℝ) : ℝ :=
  scores.sum / scores.length

theorem minimum_score_required_for_fifth_term_to_average_85 :
  let scores := [84, 80, 82, 83]
  let total_score_needed := 5 * 85
  let current_total := (scores.sum : ℝ)
  let required_score := total_score_needed - current_total
  average_score (scores ++ [required_score]) = 85 :=
by
  sorry

end minimum_score_required_for_fifth_term_to_average_85_l292_292865


namespace exists_n_faces_with_same_sides_l292_292095

-- Define the conditions of the problem in Lean
variable (n : ℕ) (P : Type)
variable [ConvexPolyhedron P] [HasFaces P (10 * n)]

-- State the theorem
theorem exists_n_faces_with_same_sides (P : ConvexPolyhedron) (n : ℕ) 
  (hP : P.faces.count = 10 * n) : 
  ∃ (F : Set Face), F.card = n ∧ ∀ f ∈ F, ∃ k : ℕ, Face.sides f = k := 
s類product sorry

end exists_n_faces_with_same_sides_l292_292095


namespace exists_divisible_by_3_l292_292794

open Nat

-- Definitions used in Lean 4 statement to represent conditions from part a)
def neighbors (n m : ℕ) : Prop := (m = n + 1) ∨ (m = n + 2) ∨ (2 * m = n) ∨ (m = 2 * n)

def circle_arrangement (ns : Fin 99 → ℕ) : Prop :=
  ∀ i : Fin 99, (neighbors (ns i) (ns ((i + 1) % 99)))

-- Proof problem:
theorem exists_divisible_by_3 (ns : Fin 99 → ℕ) (h : circle_arrangement ns) :
  ∃ i : Fin 99, 3 ∣ ns i :=
sorry

end exists_divisible_by_3_l292_292794


namespace minimum_polynomial_degree_for_separation_l292_292456

open Polynomial

theorem minimum_polynomial_degree_for_separation {a : ℕ → ℝ} (h : ∀ i j, i ≠ j → a i ≠ a j) :
  ∃ (P : Polynomial ℝ), degree P ≤ 12 ∧
    (∀ i j, i ∈ {0, 1, 2, 3, 4, 5} → j ∈ {6, 7, 8, 9, 10, 11, 12} → eval a i P > eval a j P) :=
begin
  sorry
end

end minimum_polynomial_degree_for_separation_l292_292456


namespace flagpole_break_height_l292_292956

theorem flagpole_break_height (h h_break distance : ℝ) (h_pos : 0 < h) (h_break_pos : 0 < h_break)
  (h_flagpole : h = 8) (d_distance : distance = 3) (h_relationship : (h_break ^ 2 + distance^2) = (h - h_break)^2) :
  h_break = Real.sqrt 3 :=
  sorry

end flagpole_break_height_l292_292956


namespace angle_proof_l292_292034

-- Given definitions
-- Condition: Angle BAC is 15 degrees (or π/12 in radians)
def angle_BAC_degrees : ℝ := 15
def angle_BAC_radians := angle_BAC_degrees * (π / 180)

-- Condition: AC is the angle bisector of ∠BAD, to formally claim further
def angle_BAC_bisector (BAC BAD DAC : ℝ) : Prop :=
  BAD + DAC = BAC

-- Proof goal
theorem angle_proof (angle_BAC : ℝ) (AC_is_bisector : angle_BAC_bisector angle_BAC 30 15) :
  (angle_BAD = 30 ∧ angle_BCD = 150) :=
by sorry

end angle_proof_l292_292034


namespace number_of_female_fish_l292_292573

-- Defining the constants given in the problem
def total_fish : ℕ := 45
def fraction_male : ℚ := 2 / 3

-- The statement we aim to prove in Lean
theorem number_of_female_fish : 
  (total_fish : ℚ) * (1 - fraction_male) = 15 :=
by
  sorry

end number_of_female_fish_l292_292573


namespace approximate_GDP_l292_292728

noncomputable def GDP (initial : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial * (1 + rate) ^ years

theorem approximate_GDP : 
  GDP 95933 0.073 4 ≈ 127165 :=
by
  sorry

end approximate_GDP_l292_292728


namespace inequality_example_l292_292183

theorem inequality_example (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) : 2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := 
  sorry

end inequality_example_l292_292183


namespace polyhedra_arrangement_l292_292819

noncomputable def exist_arrangement_of_convex_polyhedra : Prop :=
  ∃ (polyhedra : Fin 2001 → Set Point3D),
    (∀ (i j : Fin 2001), (i ≠ j) → (polyhedra i ∩ polyhedra j).Nonempty ∧
      Int (polyhedra i ∩ polyhedra j) = ∅) ∧
    ∀ (i j k : Fin 2001), (i ≠ j ∧ j ≠ k ∧ i ≠ k) → (polyhedra i ∩ polyhedra j ∩ polyhedra k) = ∅

theorem polyhedra_arrangement :
  ∃ (polyhedra : Fin 2001 → Set Point3D),
    (∀ (i j : Fin 2001), (i ≠ j) → (polyhedra i ∩ polyhedra j).Nonempty ∧
      Int (polyhedra i ∩ polyhedra j) = ∅) ∧
    ∀ (i j k : Fin 2001), (i ≠ j ∧ j ≠ k ∧ i ≠ k) → (polyhedra i ∩ polyhedra j ∩ polyhedra k) = ∅ :=
begin
  sorry
end

end polyhedra_arrangement_l292_292819


namespace length_GH_l292_292856

noncomputable def midpoint (a b : ℝ × ℝ) : ℝ × ℝ :=
  ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

noncomputable def length (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem length_GH :
  let A := (0, 0 : ℝ)
  let B := (1, 0 : ℝ)
  let C := (1, 1 : ℝ)
  let D := (0, 1 : ℝ)
  let E := midpoint A B
  let F := midpoint D C
  let G := ((1:ℝ)/3, (1:ℝ)/3)
  let H := ((2:ℝ)/3, (2:ℝ)/3)
  length G H = Real.sqrt 2 / 3 :=
by
  sorry

end length_GH_l292_292856


namespace sum_even_positive_integers_less_than_102_l292_292885

theorem sum_even_positive_integers_less_than_102 : 
  let a := 2
  let d := 2
  let l := 100
  let n := (l - a) / d + 1
  let sum := n / 2 * (a + l)
  (sum = 2550) :=
by
  let a := 2
  let d := 2
  let l := 100
  let n := (l - a) / d + 1
  let sum := n * (a + l) / 2
  show sum = 2550
  sorry

end sum_even_positive_integers_less_than_102_l292_292885


namespace part_a_part_b_l292_292929

noncomputable def part_a_result : ℝ :=
  1 / 2

noncomputable def part_b_result (k : ℕ) (hk : k ≥ 1) : ℝ :=
  k.fact / (2 ^ (k + 1))

theorem part_a :
  tendsto (λ n : ℕ, n * ∫ x in 0..1, (1-x) / (1+x) ^ n) at_top (𝓝 part_a_result) :=
by sorry

theorem part_b (k : ℕ) (hk : k ≥ 1) :
  tendsto (λ n : ℕ, n^(k+1) * ∫ x in 0..1, ((1-x)/(1+x))^n * x^k) at_top (𝓝 (part_b_result k hk)) :=
by sorry

end part_a_part_b_l292_292929


namespace number_of_squares_with_odd_tens_digit_l292_292295

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

def squares_with_odd_tens_digit : ℕ :=
  {m | m ≤ 200 ∧ (tens_digit (m * m)) % 2 = 1}.card

theorem number_of_squares_with_odd_tens_digit :
  squares_with_odd_tens_digit = 40 :=
sorry

end number_of_squares_with_odd_tens_digit_l292_292295


namespace roots_equal_when_m_l292_292401

noncomputable def equal_roots_condition (k n m : ℝ) : Prop :=
  1 + 4 * m^2 * k + 4 * m * n = 0

theorem roots_equal_when_m :
  equal_roots_condition 1 3 (-1.5 + Real.sqrt 2) ∧ 
  equal_roots_condition 1 3 (-1.5 - Real.sqrt 2) :=
by 
  sorry

end roots_equal_when_m_l292_292401


namespace fraction_of_three_fourths_is_one_fifth_l292_292870

theorem fraction_of_three_fourths_is_one_fifth :
  (∃ x : ℚ, x * (3 / 4) = (1 / 5)) ↔ (x = 4 / 15) :=
begin
  sorry
end

end fraction_of_three_fourths_is_one_fifth_l292_292870


namespace total_volume_of_cubes_l292_292236

theorem total_volume_of_cubes : 
  let carl_cubes := 5
  let kate_cubes := 5
  let carl_side := 1
  let kate_side := 2
  let carl_volume := carl_side ^ 3
  let kate_volume := kate_side ^ 3
  let total_carl_volume := carl_cubes * carl_volume
  let total_kate_volume := kate_cubes * kate_volume
  let total_volume := total_carl_volume + total_kate_volume
  in total_volume = 45 :=
by
  sorry

end total_volume_of_cubes_l292_292236


namespace amount_given_by_mom_l292_292082

def amount_spent_by_Mildred : ℕ := 25
def amount_spent_by_Candice : ℕ := 35
def amount_left : ℕ := 40

theorem amount_given_by_mom : 
  (amount_spent_by_Mildred + amount_spent_by_Candice + amount_left) = 100 := by
  sorry

end amount_given_by_mom_l292_292082


namespace select_televisions_l292_292297

theorem select_televisions
  (total_A : ℕ)
  (total_B : ℕ)
  (total_televisions : ℕ)
  (select_count : ℕ)
  (at_least_one_each : Prop) :
  total_A = 3 → total_B = 4 → total_televisions = 7 → select_count = 3 →
  at_least_one_each → 
  ∃ (ways : ℕ), ways = 30 :=
by
  intros hA hB hT hS hE
  have h_comb : int.choose total_televisions select_count = 35 := sorry
  have h_invalid_A : int.choose total_A select_count = 1 := sorry
  have h_invalid_B : int.choose total_B select_count = 4 := sorry
  let valid_ways := 35 - 1 - 4
  use valid_ways
  simp [valid_ways]
  assumption
  sorry

end select_televisions_l292_292297


namespace relation_w_z_relation_s_t_relation_x_r_relation_y_q_relation_z_x_t_relation_z_t_v_l292_292093

-- Prove that w - 2z = 0
theorem relation_w_z (w z : ℝ) : w - 2 * z = 0 :=
sorry

-- Prove that 2s + t - 8 = 0
theorem relation_s_t (s t : ℝ) : 2 * s + t - 8 = 0 :=
sorry

-- Prove that x - r - 2 = 0
theorem relation_x_r (x r : ℝ) : x - r - 2 = 0 :=
sorry

-- Prove that y + q - 6 = 0
theorem relation_y_q (y q : ℝ) : y + q - 6 = 0 :=
sorry

-- Prove that 3z - x - 2t + 6 = 0
theorem relation_z_x_t (z x t : ℝ) : 3 * z - x - 2 * t + 6 = 0 :=
sorry

-- Prove that 8z - 4t - v + 12 = 0
theorem relation_z_t_v (z t v : ℝ) : 8 * z - 4 * t - v + 12 = 0 :=
sorry

end relation_w_z_relation_s_t_relation_x_r_relation_y_q_relation_z_x_t_relation_z_t_v_l292_292093


namespace nut_game_winning_strategy_l292_292858

theorem nut_game_winning_strategy (N : ℕ) (h : N > 2) : ∃ second_player_wins : Prop, second_player_wins :=
sorry

end nut_game_winning_strategy_l292_292858


namespace wheel_travel_distance_l292_292557

theorem wheel_travel_distance (r : ℝ) (h_r : r = 2) : 
  ∃ D : ℝ, D = 3 * (2 * π * r) ∧ D = 12 * π :=
by 
  use 12 * π
  split
  · calc
      3 * (2 * π * r) = 3 * (4 * π) : by rw [h_r, mul_assoc]
      _ = 12 * π : by norm_num
    
  · rfl

end wheel_travel_distance_l292_292557


namespace sequence_inequality_l292_292435

theorem sequence_inequality (a : ℕ → ℝ) (h_nonneg : ∀ n, 0 ≤ a n)
    (h_subadd : ∀ m n : ℕ, a (n + m) ≤ a n + a m) :
  ∀ (n m : ℕ), m ≤ n → a n ≤ m * a 1 + ((n : ℝ) / m - 1) * a m := 
by
  intros n m hnm
  sorry

end sequence_inequality_l292_292435


namespace unique_solution_l292_292997

theorem unique_solution (x y z : ℝ) 
  (h : x^2 + 2*x + y^2 + 4*y + z^2 + 6*z = -14) : 
  x = -1 ∧ y = -2 ∧ z = -3 :=
by
  -- entering main proof section
  sorry

end unique_solution_l292_292997


namespace balance_equivalency_l292_292864

def Delta (a : ℕ) : Prop
def Diamond (b : ℕ) : Prop
def Bullet (c : ℕ) : Prop

theorem balance_equivalency (a b c : ℕ) (h1 : 3 * a + 2 * b = 12 * c) (h2 : a = 2 * b + 3 * c) : 4 * b = (32 / 3) * b :=
by
  sorry

end balance_equivalency_l292_292864


namespace carla_food_bank_l292_292237

theorem carla_food_bank :
  ∀ (initial_cans first_day_people first_day_take first_day_restock second_day_restock 
     second_day_giveaway second_day_take : ℕ),
  initial_cans = 2000 →
  first_day_people = 500 →
  first_day_take = 1 →
  first_day_restock = 1500 →
  second_day_restock = 3000 →
  second_day_giveaway = 2500 →
  second_day_take = 2 →
  let cans_after_first_day := initial_cans - (first_day_people * first_day_take) + first_day_restock in
  let cans_after_second_day := cans_after_first_day + second_day_restock in
  let number_of_people_second_day := second_day_giveaway / second_day_take in
  number_of_people_second_day = 1250 :=
by
  intros initial_cans first_day_people first_day_take first_day_restock second_day_restock second_day_giveaway second_day_take
         h_initial_cans h_first_day_people h_first_day_take h_first_day_restock h_second_day_restock h_second_day_giveaway h_second_day_take
  unfold cans_after_first_day
  unfold cans_after_second_day
  unfold number_of_people_second_day
  sorry

end carla_food_bank_l292_292237


namespace period_and_symmetry_center_and_range_l292_292302

-- Define necessary variables and functions
def a (x : ℝ) : ℝ × ℝ := (Real.sin x, -Real.cos x)
def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.cos x)
def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

noncomputable def smallest_positive_period : ℝ := Real.pi

def symmetry_center_coords (k : ℤ) : ℝ × ℝ := (Real.pi / 2 + k * Real.pi, 0)

theorem period_and_symmetry_center_and_range :
  (smallest_positive_period = Real.pi) ∧
  (∀ k : ℤ, symmetry_center_coords k = (Real.pi / 2 + k * Real.pi, 0)) ∧
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi → -1 ≤ f x ∧ f x ≤ 1) :=
by
  sorry

end period_and_symmetry_center_and_range_l292_292302


namespace exists_divisible_by_3_l292_292793

open Nat

-- Definitions used in Lean 4 statement to represent conditions from part a)
def neighbors (n m : ℕ) : Prop := (m = n + 1) ∨ (m = n + 2) ∨ (2 * m = n) ∨ (m = 2 * n)

def circle_arrangement (ns : Fin 99 → ℕ) : Prop :=
  ∀ i : Fin 99, (neighbors (ns i) (ns ((i + 1) % 99)))

-- Proof problem:
theorem exists_divisible_by_3 (ns : Fin 99 → ℕ) (h : circle_arrangement ns) :
  ∃ i : Fin 99, 3 ∣ ns i :=
sorry

end exists_divisible_by_3_l292_292793


namespace angle_BIC_is_125_degrees_l292_292031

/-- 
In triangle ABC, the angle bisectors are AD, BE, and CF,
which intersect at the incenter I. If ∠ACB = 70°, then 
the measure of ∠BIC is 125°.
-/
theorem angle_BIC_is_125_degrees (A B C D E F I : Type)
  [Incenter I A B C D E F]
  (h1 : angle A C B = 70) :
  angle B I C = 125 :=
sorry

/-- Definitions for the Incenter and angles might be necessary depending on Lean's library.
-- Here we assume these definitions and imports are handled adequately. 
structure Incenter (I A B C D E F : Type) : Prop :=
(AD_bisector : is_bisector AD A B C)
(BE_bisector : is_bisector BE B A C)
(CF_bisector : is_bisector CF C A B)
(intersect_at_I : is_incenter I A B C D E F)

end angle_BIC_is_125_degrees_l292_292031


namespace production_volume_l292_292944

/-- 
A certain school's factory produces 200 units of a certain product this year.
It is planned to increase the production volume by the same percentage \( x \)
over the next two years such that the total production volume over three years is 1400 units.
The goal is to prove that the correct equation for this scenario is:
200 + 200 * (1 + x) + 200 * (1 + x)^2 = 1400.
-/
theorem production_volume (x : ℝ) :
    200 + 200 * (1 + x) + 200 * (1 + x)^2 = 1400 := 
sorry

end production_volume_l292_292944


namespace sum_of_side_length_ratio_l292_292487

theorem sum_of_side_length_ratio (h : (75 : ℚ) / 27 = (a : ℚ) * real.sqrt b / c) : a + b + c = 9 :=
sorry

end sum_of_side_length_ratio_l292_292487


namespace at_least_one_divisible_by_3_l292_292797

-- Define a function that describes the properties of the numbers as per conditions.
def circle_99_numbers (numbers: Fin 99 → ℕ) : Prop :=
  ∀ n : Fin 99, let neighbor := (n + 1) % 99 
                in abs (numbers n - numbers neighbor) = 1 ∨ 
                   abs (numbers n - numbers neighbor) = 2 ∨ 
                   (numbers n = 2 * numbers neighbor) ∨ 
                   (numbers neighbor = 2 * numbers n)

theorem at_least_one_divisible_by_3 :
  ∀ (numbers: Fin 99 → ℕ), circle_99_numbers numbers → ∃ n : Fin 99, numbers n % 3 = 0 :=
by
  intro numbers
  intro h
  sorry

end at_least_one_divisible_by_3_l292_292797


namespace number_of_triangles_formed_by_vertices_l292_292429

theorem number_of_triangles_formed_by_vertices (P : SimpleGraph ℕ) (hP : ∀ (i j : ℕ), P.Adj i j ↔ abs (i - j) ≠ 1)
  [Poly : ∀ (n : ℕ), P = simple_graph.Cyclic (Fin n)] (h : Poly P 40) :
  let eligible_triples (P : SimpleGraph) := {abc : (Fin 40) × (Fin 40) × (Fin 40) |
     abs (abc.1.val - abc.2.val) ≥ 3 ∧ abs (abc.2.val - abc.3.val) ≥ 3 ∧ abs (abc.3.val - abc.1.val) ≥ 3}
  in eligible_triples P = 7040 := sorry

end number_of_triangles_formed_by_vertices_l292_292429


namespace dihedral_angle_sum_eq_l292_292136

-- Define tetrahedron edges and their lengths explicitly
variables {A B C D : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables (AB CD BC AD : ℝ)

-- Define dihedral angles as functions of edges
variables (θ_AB θ_CD θ_BC θ_AD : ℝ)

-- Assumption: AB + CD = BC + AD
axiom length_sum_eq (h : AB + CD = BC + AD)

-- Define the sum of dihedral angles at respective edges
def angle_sum_first_pair : ℝ := θ_AB + θ_CD
def angle_sum_second_pair : ℝ := θ_BC + θ_AD

-- State the theorem to prove
theorem dihedral_angle_sum_eq (h : AB + CD = BC + AD) :
  angle_sum_first_pair θ_AB θ_CD = angle_sum_second_pair θ_BC θ_AD :=
sorry

end dihedral_angle_sum_eq_l292_292136


namespace log2_of_fraction_l292_292591

theorem log2_of_fraction : Real.logb 2 0.03125 = -5 := by
  sorry

end log2_of_fraction_l292_292591


namespace annual_feeding_cost_is_correct_l292_292692

-- Definitions based on conditions
def number_of_geckos : Nat := 3
def number_of_iguanas : Nat := 2
def number_of_snakes : Nat := 4
def cost_per_gecko_per_month : Nat := 15
def cost_per_iguana_per_month : Nat := 5
def cost_per_snake_per_month : Nat := 10

-- Statement of the theorem
theorem annual_feeding_cost_is_correct : 
    (number_of_geckos * cost_per_gecko_per_month
    + number_of_iguanas * cost_per_iguana_per_month 
    + number_of_snakes * cost_per_snake_per_month) * 12 = 1140 := by
  sorry

end annual_feeding_cost_is_correct_l292_292692


namespace units_digit_of_sum_of_cubes_l292_292899

theorem units_digit_of_sum_of_cubes : 
  (24^3 + 42^3) % 10 = 2 := by
sorry

end units_digit_of_sum_of_cubes_l292_292899


namespace count_even_digit_sum_below_million_correct_l292_292359

open Nat

def is_even_digit_sum (n : ℕ) : Prop :=
  (digits 10 n).sum % 2 = 0

def count_even_digit_sums_below_million : ℕ :=
  (Nat.range 1000000).filter (λ n, is_even_digit_sum n ∧ is_even_digit_sum (n + 1)).length

theorem count_even_digit_sum_below_million_correct :
  count_even_digit_sums_below_million = 45454 := by
  sorry

end count_even_digit_sum_below_million_correct_l292_292359


namespace correct_statement_about_algorithms_l292_292914

-- Definitions based on conditions
def is_algorithm (A B C D : Prop) : Prop :=
  ¬A ∧ B ∧ ¬C ∧ ¬D

-- Ensure the correct statement using the conditions specified
theorem correct_statement_about_algorithms (A B C D : Prop) (h : is_algorithm A B C D) : B :=
by
  obtain ⟨hnA, hB, hnC, hnD⟩ := h
  exact hB

end correct_statement_about_algorithms_l292_292914


namespace sin_pi_div_2_minus_2alpha_l292_292377

noncomputable def inclination_angle (m : ℝ) : ℝ :=
real.atan m

theorem sin_pi_div_2_minus_2alpha (α : ℝ) (h : α = inclination_angle (-1/2)) :
  Real.sin (Real.pi / 2 - 2 * α) = 3 / 5 :=
by
  sorry

end sin_pi_div_2_minus_2alpha_l292_292377


namespace conditional_prob_eventA_given_eventB_l292_292452

noncomputable def eventA (d1 d2 d3 : ℕ) : Prop :=
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3

noncomputable def eventB (d1 d2 d3 : ℕ) : Prop :=
  d1 = 2 ∨ d2 = 2 ∨ d3 = 2

theorem conditional_prob_eventA_given_eventB :
  -- Total number of outcomes for three dice
  let total_outcomes := 6^3 in
  -- Number of outcomes where no die shows a 2
  let no_two_outcomes := 5^3 in
  -- Number of outcomes for event B
  let B_outcomes := total_outcomes - no_two_outcomes in
  -- Number of favorable outcomes for both A and B
  let favorable_outcomes := 3 * 5 * 4 in
  -- Conditional probability P(A|B)
  let P_A_given_B := (favorable_outcomes : ℝ) / (B_outcomes : ℝ) in
  P_A_given_B = 60 / 91 :=
begin
  sorry
end

end conditional_prob_eventA_given_eventB_l292_292452


namespace min_dist_PQ_l292_292774

noncomputable def minimum_distance (P Q : ℝ × ℝ) : ℝ :=
  let d := dist P XY;
  2 * d

theorem min_dist_PQ (P Q : ℝ × ℝ) (hP : P.2 = real.exp P.1) (hQ : Q.2 = real.log Q.1) : 
  minimum_distance P Q = real.sqrt 2 :=
begin
  sorry
end

end min_dist_PQ_l292_292774


namespace proposition_B_is_true_l292_292219

theorem proposition_B_is_true (parallel_lines : ∀ (l1 l2 : Line), l1 ∥ l2 → corresponding_angles_equal)
  (vertically_opposite_angles : ∀ (l1 l2 : Line), intersect l1 l2 → vertically_opposite_angles_equal)
  (a b: ℝ) : (∃ a b, a^2 = b^2 ∧ a ≠ b) 
  ∧ ¬ (∀ a b, a > b → |a| > |b|) 
  → vertically_opposite_angles_equal :=
by sorry

end proposition_B_is_true_l292_292219


namespace no_more_than_n_lines_divide_area_l292_292651

theorem no_more_than_n_lines_divide_area {n : ℕ} (polygon : ConvexPolygon n) (O : Point) :
  (∀ l : Line, divides_area l polygon O) → (counts_lines l ≤ n) := 
sorry

end no_more_than_n_lines_divide_area_l292_292651


namespace volume_P3_correct_m_plus_n_l292_292601

noncomputable def P_0_volume : ℚ := 1

noncomputable def tet_volume (v : ℚ) : ℚ := (1/27) * v

noncomputable def volume_P3 : ℚ := 
  let ΔP1 := 4 * tet_volume P_0_volume
  let ΔP2 := (2/9) * ΔP1
  let ΔP3 := (2/9) * ΔP2
  P_0_volume + ΔP1 + ΔP2 + ΔP3

theorem volume_P3_correct : volume_P3 = 22615 / 6561 := 
by {
  sorry
}

theorem m_plus_n : 22615 + 6561 = 29176 := 
by {
  sorry
}

end volume_P3_correct_m_plus_n_l292_292601


namespace sum_even_positive_integers_less_than_102_l292_292884

theorem sum_even_positive_integers_less_than_102 : 
  let a := 2
  let d := 2
  let l := 100
  let n := (l - a) / d + 1
  let sum := n / 2 * (a + l)
  (sum = 2550) :=
by
  let a := 2
  let d := 2
  let l := 100
  let n := (l - a) / d + 1
  let sum := n * (a + l) / 2
  show sum = 2550
  sorry

end sum_even_positive_integers_less_than_102_l292_292884


namespace sequence_a6_value_l292_292745

theorem sequence_a6_value :
  ∃ (a : ℕ → ℚ), (a 1 = 1) ∧ (∀ n, a (n + 1) = a n / (2 * a n + 1)) ∧ (a 6 = 1 / 11) :=
by
  sorry

end sequence_a6_value_l292_292745


namespace long_furred_brown_dogs_l292_292731

theorem long_furred_brown_dogs :
  ∀ (T L B N LB : ℕ), T = 60 → L = 45 → B = 35 → N = 12 →
  (LB = L + B - (T - N)) → LB = 32 :=
by
  intros T L B N LB hT hL hB hN hLB
  sorry

end long_furred_brown_dogs_l292_292731


namespace sum_even_positive_integers_less_than_102_l292_292892

theorem sum_even_positive_integers_less_than_102 : 
  let sum_even : ℕ := ∑ n in Finset.filter (λ x, even x) (Finset.range 102), n
  in sum_even = 2550 :=
by
  sorry

end sum_even_positive_integers_less_than_102_l292_292892


namespace instantaneous_velocity_at_3_l292_292475

-- Define the position function s(t)
def s (t : ℝ) : ℝ := 1 - t + t^2

-- The main statement we need to prove
theorem instantaneous_velocity_at_3 : (deriv s 3) = 5 :=
by 
  -- The theorem requires a proof which we mark as sorry for now.
  sorry

end instantaneous_velocity_at_3_l292_292475


namespace max_ball_height_l292_292528

/-- 
The height (in feet) of a ball traveling on a parabolic path is given by -20t^2 + 80t + 36,
where t is the time after launch. This theorem shows that the maximum height of the ball is 116 feet.
-/
theorem max_ball_height : ∃ t : ℝ, ∀ t', -20 * t^2 + 80 * t + 36 ≤ -20 * t'^2 + 80 * t' + 36 → -20 * t^2 + 80 * t + 36 = 116 :=
sorry

end max_ball_height_l292_292528


namespace min_sum_slopes_tangents_l292_292151

open Real

-- Define the equation of the parabola
def parabola (x y : ℝ) : Prop := x^2 - 4 * (x + y) + y^2 = 2 * x * y + 8

-- Given conditions
def intersection_condition (p q : ℝ) : Prop := p + q = -32

-- Statement that needs to be proved
theorem min_sum_slopes_tangents :
  ∃ (m n : ℕ), gcd m n = 1 ∧
  (∀ l1 l2 : ℝ → ℝ, (∀ x y, parabola x y → tangent_line l1 x y ∧ tangent_line l2 x y) →
    (∃ (p q : ℝ), intersection_condition p q) →
    assumes_sum_slopes_min l1 l2 (62 / 29) ∧ (m + n) = 91) :=
begin
  sorry
end

end min_sum_slopes_tangents_l292_292151


namespace mike_taller_than_mark_l292_292785

def feet_to_inches (feet : ℕ) : ℕ := 12 * feet

def mark_height_feet := 5
def mark_height_inches := 3
def mike_height_feet := 6
def mike_height_inches := 1

def mark_total_height := feet_to_inches mark_height_feet + mark_height_inches
def mike_total_height := feet_to_inches mike_height_feet + mike_height_inches

theorem mike_taller_than_mark : mike_total_height - mark_total_height = 10 :=
by
  sorry

end mike_taller_than_mark_l292_292785


namespace sequence_inequality_l292_292434

theorem sequence_inequality (a : ℕ → ℝ) (h_nonneg : ∀ n, 0 ≤ a n)
    (h_subadd : ∀ m n : ℕ, a (n + m) ≤ a n + a m) :
  ∀ (n m : ℕ), m ≤ n → a n ≤ m * a 1 + ((n : ℝ) / m - 1) * a m := 
by
  intros n m hnm
  sorry

end sequence_inequality_l292_292434


namespace complex_fraction_power_l292_292264

theorem complex_fraction_power :
  ( (2 + 2 * Complex.i) / (2 - 2 * Complex.i) ) ^ 8 = 1 := 
by
  sorry

end complex_fraction_power_l292_292264


namespace pizza_share_l292_292993

theorem pizza_share (remaining_pizza : ℚ) (employees : ℚ) (portion : ℚ) :
  remaining_pizza = 5 / 8 → employees = 4 → portion = (remaining_pizza / employees) → portion = 5 / 32 := by
  intros hr he hp
  rw hr at hp
  rw he at hp
  norm_num at hp
  exact hp

end pizza_share_l292_292993


namespace max_k_constant_l292_292622

theorem max_k_constant : 
  (∃ k, (∀ (x y z : ℝ), 0 < x → 0 < y → 0 < z → 
  (x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y) <= k * Real.sqrt (x + y + z))) 
  ∧ k = Real.sqrt 6 / 2) :=
sorry

end max_k_constant_l292_292622


namespace percentage_peanut_clusters_is_64_l292_292240

def total_chocolates := 50
def caramels := 3
def nougats := 2 * caramels
def truffles := caramels + 6
def other_chocolates := caramels + nougats + truffles
def peanut_clusters := total_chocolates - other_chocolates
def percentage_peanut_clusters := (peanut_clusters * 100) / total_chocolates

theorem percentage_peanut_clusters_is_64 :
  percentage_peanut_clusters = 64 := by
  sorry

end percentage_peanut_clusters_is_64_l292_292240


namespace total_amount_divided_l292_292530

theorem total_amount_divided 
    (A B C : ℝ) 
    (h1 : A = (2 / 3) * (B + C)) 
    (h2 : B = (2 / 3) * (A + C)) 
    (h3 : A = 160) : 
    A + B + C = 400 := 
by 
  sorry

end total_amount_divided_l292_292530


namespace general_formula_of_a_sum_of_c_n_l292_292398

-- Given conditions
variables (a : ℕ → ℕ)
def S (n : ℕ) : ℕ := n * (a 1 + a n) / 2

-- Constants
axiom a1 : a 1 = 3
axiom a3_S3_eq_27 : a 3 + S a 3 = 27

-- Sequence c_n
def c (n : ℕ) : ℕ := 3 / (2 * S n)

-- Sum of sequence c_n
def T (n : ℕ) : ℕ := n - n / (n + 1)

-- Proof problem
theorem general_formula_of_a (n : ℕ) : a n = 3 * n :=
sorry

theorem sum_of_c_n (n : ℕ) : T n = n / (n + 1) :=
sorry

end general_formula_of_a_sum_of_c_n_l292_292398


namespace ball_beyond_hole_l292_292217

theorem ball_beyond_hole
  (first_turn_distance : ℕ)
  (second_turn_distance : ℕ)
  (total_distance_to_hole : ℕ) :
  first_turn_distance = 180 →
  second_turn_distance = first_turn_distance / 2 →
  total_distance_to_hole = 250 →
  second_turn_distance - (total_distance_to_hole - first_turn_distance) = 20 :=
by
  intros
  -- Proof omitted
  sorry

end ball_beyond_hole_l292_292217


namespace tony_needs_36_gallons_of_paint_l292_292146

theorem tony_needs_36_gallons_of_paint :
  (let radius := 5
     height := 20
     num_columns := 20
     paint_coverage := 350
     total_paintable_area := num_columns * 2 * Real.pi * radius * height in
   Real.ceil (total_paintable_area / paint_coverage)
  ) = 36 := 
by sorry

end tony_needs_36_gallons_of_paint_l292_292146


namespace polyhedron_faces_with_same_sides_l292_292096

theorem polyhedron_faces_with_same_sides (n : ℕ) (h : n > 0) :
  ∃ k, ∃ (faces_with_k_sides : ℕ), faces_with_k_sides ≥ n ∧ ∀ (P : polyhedron), convex P ∧ P.faces = 10 * n → count_faces_with_sides P k = faces_with_k_sides :=
by
  sorry

end polyhedron_faces_with_same_sides_l292_292096


namespace inscribed_sphere_property_l292_292686

-- Define what it means for a circle to be inscribed in an equilateral triangle
def inscribed_circle_tangent_to_midpoints (T : Type) [equilateral_triangle T] (C : circle) : Prop :=
  ∀ side, C.tangent_to (midpoint side)

-- Define what it means for a sphere to be inscribed in a regular tetrahedron
def inscribed_sphere_tangent_to_centers (T : Type) [regular_tetrahedron T] (S : sphere) : Prop :=
  ∀ face, S.tangent_to (center face)

-- The given condition: the property of the inscribed circle in an equilateral triangle
axiom inscribed_circle_property (T : Type) [equilateral_triangle T] :
  ∃ C : circle, inscribed_circle_tangent_to_midpoints T C

-- The conjectured property for the tetrahedron based on the analogy
theorem inscribed_sphere_property (T : Type) [regular_tetrahedron T] :
  ∃ S : sphere, inscribed_sphere_tangent_to_centers T S :=
sorry

end inscribed_sphere_property_l292_292686


namespace rate_of_stream_l292_292197

def effectiveSpeedDownstream (v : ℝ) : ℝ := 36 + v
def effectiveSpeedUpstream (v : ℝ) : ℝ := 36 - v

theorem rate_of_stream (v : ℝ) (hf1 : effectiveSpeedUpstream v = 3 * effectiveSpeedDownstream v) : v = 18 := by
  sorry

end rate_of_stream_l292_292197


namespace alice_gadgets_sales_l292_292563

variable (S : ℝ) -- Variable to denote the worth of gadgets Alice sold
variable (E : ℝ) -- Variable to denote Alice's total earnings

theorem alice_gadgets_sales :
  let basic_salary := 240
  let commission_percentage := 0.02
  let save_amount := 29
  let save_percentage := 0.10
  
  -- Total earnings equation
  let earnings_eq := E = basic_salary + commission_percentage * S
  
  -- Savings equation
  let savings_eq := save_percentage * E = save_amount
  
  -- Solve the system of equations to show S = 2500
  S = 2500 :=
by
  sorry

end alice_gadgets_sales_l292_292563
