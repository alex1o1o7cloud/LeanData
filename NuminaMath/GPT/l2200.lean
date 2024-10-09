import Mathlib

namespace correct_judgment_l2200_220018

open Real

def period_sin2x (T : ℝ) : Prop := ∀ x, sin (2 * x) = sin (2 * (x + T))
def smallest_positive_period_sin2x : Prop := ∃ T > 0, period_sin2x T ∧ ∀ T' > 0, period_sin2x T' → T ≤ T'
def smallest_positive_period_sin2x_is_pi : Prop := ∃ T, smallest_positive_period_sin2x ∧ T = π

def symmetry_cosx (L : ℝ) : Prop := ∀ x, cos (L - x) = cos (L + x)
def symmetry_about_line_cosx (L : ℝ) : Prop := L = π / 2

def p : Prop := smallest_positive_period_sin2x_is_pi
def q : Prop := symmetry_about_line_cosx (π / 2)

theorem correct_judgment : ¬ (p ∧ q) :=
by 
  sorry

end correct_judgment_l2200_220018


namespace mary_initial_sugar_eq_4_l2200_220054

/-- Mary is baking a cake. The recipe calls for 7 cups of sugar and she needs to add 3 more cups of sugar. -/
def total_sugar : ℕ := 7
def additional_sugar : ℕ := 3

theorem mary_initial_sugar_eq_4 :
  ∃ initial_sugar : ℕ, initial_sugar + additional_sugar = total_sugar ∧ initial_sugar = 4 :=
sorry

end mary_initial_sugar_eq_4_l2200_220054


namespace abs_eq_inequality_l2200_220045

theorem abs_eq_inequality (m : ℝ) (h : |m - 9| = 9 - m) : m ≤ 9 :=
sorry

end abs_eq_inequality_l2200_220045


namespace James_total_passengers_l2200_220041

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

end James_total_passengers_l2200_220041


namespace part1_part2_l2200_220006

-- Define the universal set U as real numbers ℝ
def U : Set ℝ := Set.univ

-- Define Set A
def A (a : ℝ) : Set ℝ := {x | abs (x - a) ≤ 1 }

-- Define Set B
def B : Set ℝ := {x | (4 - x) * (x - 1) ≤ 0 }

-- Part 1: Prove A ∪ B when a = 4
theorem part1 : A 4 ∪ B = {x : ℝ | x ≥ 3 ∨ x ≤ 1} :=
sorry

-- Part 2: Prove the range of values for a given A ∩ B = A
theorem part2 (a : ℝ) (h : A a ∩ B = A a) : a ≥ 5 ∨ a ≤ 0 :=
sorry

end part1_part2_l2200_220006


namespace balcony_more_than_orchestra_l2200_220095

theorem balcony_more_than_orchestra (x y : ℕ) 
  (h1 : x + y = 370) 
  (h2 : 12 * x + 8 * y = 3320) : y - x = 190 :=
sorry

end balcony_more_than_orchestra_l2200_220095


namespace total_fault_line_movement_l2200_220042

-- Define the movements in specific years.
def movement_past_year : ℝ := 1.25
def movement_year_before : ℝ := 5.25

-- Theorem stating the total movement of the fault line over the two years.
theorem total_fault_line_movement : movement_past_year + movement_year_before = 6.50 :=
by
  -- Proof is omitted.
  sorry

end total_fault_line_movement_l2200_220042


namespace tan_alpha_value_l2200_220017

theorem tan_alpha_value (α : ℝ) 
  (h1 : Real.sin (Real.pi + α) = 3 / 5) 
  (h2 : Real.pi < α ∧ α < 3 * Real.pi / 2) : 
  Real.tan α = 3 / 4 := 
sorry

end tan_alpha_value_l2200_220017


namespace leopards_arrangement_l2200_220083

theorem leopards_arrangement :
  let leopards := 10
  let shortest := 3
  let remaining := leopards - shortest
  (shortest! * remaining! = 30240) :=
by
  let leopards := 10
  let shortest := 3
  let remaining := leopards - shortest
  have factorials_eq: shortest! * remaining! = 30240 := sorry
  exact factorials_eq

end leopards_arrangement_l2200_220083


namespace solve_equation1_solve_equation2_l2200_220067

theorem solve_equation1 (x : ℝ) : x^2 + 2 * x = 0 ↔ x = 0 ∨ x = -2 :=
by sorry

theorem solve_equation2 (x : ℝ) : 2 * x^2 - 6 * x = 3 ↔ x = (3 + Real.sqrt 15) / 2 ∨ x = (3 - Real.sqrt 15) / 2 :=
by sorry

end solve_equation1_solve_equation2_l2200_220067


namespace age_of_youngest_child_l2200_220020

theorem age_of_youngest_child (x : ℕ) 
  (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 60) : 
  x = 6 :=
sorry

end age_of_youngest_child_l2200_220020


namespace dihedral_angle_is_60_degrees_l2200_220094

def point (x y z : ℝ) := (x, y, z)

noncomputable def dihedral_angle (P Q R S T : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem dihedral_angle_is_60_degrees :
  dihedral_angle 
    (point 1 0 0)  -- A
    (point 1 1 0)  -- B
    (point 0 0 0)  -- D
    (point 1 0 1)  -- A₁
    (point 0 0 1)  -- D₁
 = 60 :=
sorry

end dihedral_angle_is_60_degrees_l2200_220094


namespace sum_coordinates_l2200_220021

variables (x y : ℝ)
def A_coord := (9, 3)
def M_coord := (3, 7)

def midpoint_condition_x : Prop := (x + 9) / 2 = 3
def midpoint_condition_y : Prop := (y + 3) / 2 = 7

theorem sum_coordinates (h1 : midpoint_condition_x x) (h2 : midpoint_condition_y y) : 
  x + y = 8 :=
by 
  sorry

end sum_coordinates_l2200_220021


namespace heaviest_person_is_Vanya_l2200_220060

variables (A D T V M : ℕ)

-- conditions
def condition1 : Prop := A + D = 82
def condition2 : Prop := D + T = 74
def condition3 : Prop := T + V = 75
def condition4 : Prop := V + M = 65
def condition5 : Prop := M + A = 62

theorem heaviest_person_is_Vanya (h1 : condition1 A D) (h2 : condition2 D T) (h3 : condition3 T V) (h4 : condition4 V M) (h5 : condition5 M A) :
  V = 43 :=
sorry

end heaviest_person_is_Vanya_l2200_220060


namespace abraham_initial_budget_l2200_220091

-- Definitions based on conditions
def shower_gel_price := 4
def shower_gel_quantity := 4
def toothpaste_price := 3
def laundry_detergent_price := 11
def remaining_budget := 30

-- Calculations based on the conditions
def spent_on_shower_gels := shower_gel_quantity * shower_gel_price
def spent_on_toothpaste := toothpaste_price
def spent_on_laundry_detergent := laundry_detergent_price
def total_spent := spent_on_shower_gels + spent_on_toothpaste + spent_on_laundry_detergent

-- The theorem to prove
theorem abraham_initial_budget :
  (total_spent + remaining_budget) = 60 :=
by
  sorry

end abraham_initial_budget_l2200_220091


namespace right_triangles_count_l2200_220000

theorem right_triangles_count (b a : ℕ) (h₁: b < 150) (h₂: (a^2 + b^2 = (b + 2)^2)) :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 12 ∧ b = n^2 - 1 :=
by
  -- This intended to state the desired number and form of the right triangles.
  sorry

def count_right_triangles : ℕ :=
  12 -- Result as a constant based on proof steps

#eval count_right_triangles -- Should output 12

end right_triangles_count_l2200_220000


namespace farmer_brown_leg_wing_count_l2200_220023

theorem farmer_brown_leg_wing_count :
  let chickens := 7
  let sheep := 5
  let grasshoppers := 10
  let spiders := 3
  let pigeons := 4
  let kangaroos := 2
  
  let chicken_legs := 2
  let chicken_wings := 2
  let sheep_legs := 4
  let grasshopper_legs := 6
  let grasshopper_wings := 2
  let spider_legs := 8
  let pigeon_legs := 2
  let pigeon_wings := 2
  let kangaroo_legs := 2

  (chickens * (chicken_legs + chicken_wings) +
  sheep * sheep_legs +
  grasshoppers * (grasshopper_legs + grasshopper_wings) +
  spiders * spider_legs +
  pigeons * (pigeon_legs + pigeon_wings) +
  kangaroos * kangaroo_legs) = 172 := 
by
  sorry

end farmer_brown_leg_wing_count_l2200_220023


namespace circle_intersects_y_axis_at_one_l2200_220016

theorem circle_intersects_y_axis_at_one :
  let A := (-2011, 0)
  let B := (2010, 0)
  let C := (0, (-2010) * 2011)
  ∃ (D : ℝ × ℝ), D = (0, 1) ∧
    (∃ O : ℝ × ℝ, O = (0, 0) ∧
    (dist O A) * (dist O B) = (dist O C) * (dist O D)) :=
by
  sorry -- Proof of the theorem

end circle_intersects_y_axis_at_one_l2200_220016


namespace han_xin_troop_min_soldiers_l2200_220096

theorem han_xin_troop_min_soldiers (n : ℕ) : 
  (n % 3 = 2) ∧ (n % 5 = 3) ∧ (n % 7 = 4) → n = 53 :=
  sorry

end han_xin_troop_min_soldiers_l2200_220096


namespace diophantine_infinite_solutions_l2200_220009

theorem diophantine_infinite_solutions
  (l m n : ℕ) (h_l_positive : l > 0) (h_m_positive : m > 0) (h_n_positive : n > 0)
  (h_gcd_lm_n : gcd (l * m) n = 1) (h_gcd_ln_m : gcd (l * n) m = 1) (h_gcd_mn_l : gcd (m * n) l = 1)
  : ∃ x y z : ℕ, (x > 0 ∧ y > 0 ∧ z > 0 ∧ (x ^ l + y ^ m = z ^ n)) ∧ (∀ a b c : ℕ, (a > 0 ∧ b > 0 ∧ c > 0 ∧ (a ^ l + b ^ m = c ^ n)) → ∀ d : ℕ, d > 0 → ∃ e f g : ℕ, (e > 0 ∧ f > 0 ∧ g > 0 ∧ (e ^ l + f ^ m = g ^ n))) :=
sorry

end diophantine_infinite_solutions_l2200_220009


namespace exists_subset_sum_divisible_by_2n_l2200_220014

open BigOperators

theorem exists_subset_sum_divisible_by_2n (n : ℕ) (hn : n ≥ 4) (a : Fin n → ℤ)
  (h_distinct : ∀ i j : Fin n, i ≠ j → a i ≠ a j)
  (h_interval : ∀ i : Fin n, 0 < a i ∧ a i < 2 * n) :
  ∃ (s : Finset (Fin n)), (∑ i in s, a i) % (2 * n) = 0 :=
sorry

end exists_subset_sum_divisible_by_2n_l2200_220014


namespace question_1_question_2_l2200_220043

def curve_is_ellipse (m : ℝ) : Prop :=
  (3 - m > 0) ∧ (m - 1 > 0) ∧ (3 - m > m - 1)

def domain_is_R (m : ℝ) : Prop :=
  m^2 < (9 / 4)

theorem question_1 (m : ℝ) :
  curve_is_ellipse m → 1 < m ∧ m < 2 :=
sorry

theorem question_2 (m : ℝ) :
  (curve_is_ellipse m ∧ domain_is_R m) → 1 < m ∧ m < (3 / 2) :=
sorry

end question_1_question_2_l2200_220043


namespace expression_value_l2200_220004

theorem expression_value (a b : ℚ) (h : a + 2 * b = 0) : 
  abs (a / |b| - 1) + abs (|a| / b - 2) + abs (|a / b| - 3) = 4 :=
sorry

end expression_value_l2200_220004


namespace desks_increase_l2200_220049

theorem desks_increase 
  (rows : ℕ) (first_row_desks : ℕ) (total_desks : ℕ) 
  (d : ℕ) 
  (h_rows : rows = 8) 
  (h_first_row : first_row_desks = 10) 
  (h_total_desks : total_desks = 136)
  (h_desks_sum : 10 + (10 + d) + (10 + 2 * d) + (10 + 3 * d) + (10 + 4 * d) + (10 + 5 * d) + (10 + 6 * d) + (10 + 7 * d) = total_desks) : 
  d = 2 := 
by 
  sorry

end desks_increase_l2200_220049


namespace ceil_add_eq_double_of_int_l2200_220074

theorem ceil_add_eq_double_of_int {x : ℤ} (h : ⌈(x : ℝ)⌉ + ⌊(x : ℝ)⌋ = 2 * (x : ℝ)) : ⌈(x : ℝ)⌉ + x = 2 * x :=
by
  sorry

end ceil_add_eq_double_of_int_l2200_220074


namespace find_some_number_l2200_220084

def some_number (x : Int) (some_num : Int) : Prop :=
  (3 < x ∧ x < 10) ∧
  (5 < x ∧ x < 18) ∧
  (9 > x ∧ x > -2) ∧
  (8 > x ∧ x > 0) ∧
  (x + some_num < 9)

theorem find_some_number :
  ∀ (some_num : Int), some_number 7 some_num → some_num < 2 :=
by
  intros some_num H
  sorry

end find_some_number_l2200_220084


namespace betty_red_beads_l2200_220011

theorem betty_red_beads (r b : ℕ) (h_ratio : r / b = 3 / 2) (h_blue_beads : b = 20) : r = 30 :=
by
  sorry

end betty_red_beads_l2200_220011


namespace area_of_L_shape_l2200_220029

theorem area_of_L_shape (a : ℝ) (h_pos : a > 0) (h_eq : 4 * ((a + 3)^2 - a^2) = 5 * a^2) : 
  (a + 3)^2 - a^2 = 45 :=
by
  sorry

end area_of_L_shape_l2200_220029


namespace find_number_l2200_220076

theorem find_number (A : ℕ) (B : ℕ) (H1 : B = 300) (H2 : Nat.lcm A B = 2310) (H3 : Nat.gcd A B = 30) : A = 231 := 
by 
  sorry

end find_number_l2200_220076


namespace smallest_rel_prime_to_180_l2200_220070

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ x ≤ 7 ∧ (∀ y : ℕ, y > 1 ∧ y < x → y.gcd 180 ≠ 1) ∧ x.gcd 180 = 1 :=
by
  sorry

end smallest_rel_prime_to_180_l2200_220070


namespace estimated_height_is_644_l2200_220022

noncomputable def height_of_second_building : ℝ := 100
noncomputable def height_of_first_building : ℝ := 0.8 * height_of_second_building
noncomputable def height_of_third_building : ℝ := (height_of_first_building + height_of_second_building) - 20
noncomputable def height_of_fourth_building : ℝ := 1.15 * height_of_third_building
noncomputable def height_of_fifth_building : ℝ := 2 * |height_of_second_building - height_of_third_building|
noncomputable def total_estimated_height : ℝ := height_of_first_building + height_of_second_building + height_of_third_building + height_of_fourth_building + height_of_fifth_building

theorem estimated_height_is_644 : total_estimated_height = 644 := by
  sorry

end estimated_height_is_644_l2200_220022


namespace three_points_in_circle_of_radius_one_seventh_l2200_220048

-- Define the problem
theorem three_points_in_circle_of_radius_one_seventh (P : Fin 51 → ℝ × ℝ) :
  (∀ i, 0 ≤ (P i).1 ∧ (P i).1 ≤ 1 ∧ 0 ≤ (P i).2 ∧ (P i).2 ≤ 1) →
  ∃ (i j k : Fin 51), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    dist (P i) (P j) ≤ 2/7 ∧ dist (P j) (P k) ≤ 2/7 ∧ dist (P k) (P i) ≤ 2/7 :=
by
  sorry

end three_points_in_circle_of_radius_one_seventh_l2200_220048


namespace bug_visits_exactly_16_pavers_l2200_220002

-- Defining the dimensions of the garden and the pavers
def garden_width : ℕ := 14
def garden_length : ℕ := 19
def paver_size : ℕ := 2

-- Calculating the number of pavers in width and length
def pavers_width : ℕ := garden_width / paver_size
def pavers_length : ℕ := (garden_length + paver_size - 1) / paver_size  -- Taking ceiling of 19/2

-- Calculating the GCD of the pavers count in width and length
def gcd_pavers : ℕ := Nat.gcd pavers_width pavers_length

-- Calculating the number of pavers the bug crosses
def pavers_crossed : ℕ := pavers_width + pavers_length - gcd_pavers

-- Theorem that states the number of pavers visited
theorem bug_visits_exactly_16_pavers :
  pavers_crossed = 16 := by
  -- Sorry is used to skip the proof steps
  sorry

end bug_visits_exactly_16_pavers_l2200_220002


namespace rowing_speed_in_still_water_l2200_220061

variable (v c t : ℝ)
variable (h1 : c = 1.3)
variable (h2 : 2 * ((v - c) * t) = ((v + c) * t))

theorem rowing_speed_in_still_water : v = 3.9 := by
  sorry

end rowing_speed_in_still_water_l2200_220061


namespace loaves_of_bread_l2200_220037

-- Definitions for the given conditions
def total_flour : ℝ := 5
def flour_per_loaf : ℝ := 2.5

-- The statement of the problem
theorem loaves_of_bread (total_flour : ℝ) (flour_per_loaf : ℝ) : 
  total_flour / flour_per_loaf = 2 :=
by
  -- Proof is not required
  sorry

end loaves_of_bread_l2200_220037


namespace product_neg_six_l2200_220052

theorem product_neg_six (m b : ℝ)
  (h1 : m = 2)
  (h2 : b = -3) : m * b < -3 := by
-- Proof skipped
sorry

end product_neg_six_l2200_220052


namespace initial_pile_counts_l2200_220046

def pile_transfers (A B C : ℕ) : Prop :=
  (A + B + C = 48) ∧
  ∃ (A' B' C' : ℕ), 
    (A' = A + B) ∧ (B' = B + C) ∧ (C' = C + A) ∧
    (A' = 2 * 16) ∧ (B' = 2 * 12) ∧ (C' = 2 * 14)

theorem initial_pile_counts :
  ∃ A B C : ℕ, pile_transfers A B C ∧ A = 22 ∧ B = 14 ∧ C = 12 :=
by
  sorry

end initial_pile_counts_l2200_220046


namespace distance_traveled_l2200_220078

-- Given conditions
def speed : ℕ := 100 -- Speed in km/hr
def time : ℕ := 5    -- Time in hours

-- The goal is to prove the distance traveled is 500 km
theorem distance_traveled : speed * time = 500 := by
  -- we state the proof goal
  sorry

end distance_traveled_l2200_220078


namespace wendy_created_albums_l2200_220001

theorem wendy_created_albums (phone_pics : ℕ) (camera_pics : ℕ) (pics_per_album : ℕ) (total_pics : ℕ) (albums : ℕ) :
  phone_pics = 22 → camera_pics = 2 → pics_per_album = 6 → total_pics = phone_pics + camera_pics → albums = total_pics / pics_per_album → albums = 4 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end wendy_created_albums_l2200_220001


namespace find_f_inv_difference_l2200_220089

axiom f : ℤ → ℤ
axiom f_inv : ℤ → ℤ
axiom f_has_inverse : ∀ x : ℤ, f_inv (f x) = x ∧ f (f_inv x) = x
axiom f_inverse_conditions : ∀ x : ℤ, f (x + 2) = f_inv (x - 1)

theorem find_f_inv_difference :
  f_inv 2004 - f_inv 1 = 4006 :=
sorry

end find_f_inv_difference_l2200_220089


namespace binom_n_plus_1_n_minus_1_eq_l2200_220039

theorem binom_n_plus_1_n_minus_1_eq (n : ℕ) (h : 0 < n) : (Nat.choose (n + 1) (n - 1)) = n * (n + 1) / 2 := 
by sorry

end binom_n_plus_1_n_minus_1_eq_l2200_220039


namespace least_number_subtracted_divisible_by_17_and_23_l2200_220019

-- Conditions
def is_divisible_by_17_and_23 (n : ℕ) : Prop := 
  n % 17 = 0 ∧ n % 23 = 0

def target_number : ℕ := 7538

-- The least number to be subtracted
noncomputable def least_number_to_subtract : ℕ := 109

-- Theorem statement
theorem least_number_subtracted_divisible_by_17_and_23 : 
  is_divisible_by_17_and_23 (target_number - least_number_to_subtract) :=
by 
  -- Proof details would normally follow here.
  sorry

end least_number_subtracted_divisible_by_17_and_23_l2200_220019


namespace sum_lent_250_l2200_220030

theorem sum_lent_250 (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) 
  (hR : R = 4) (hT : T = 8) (hSI1 : SI = P - 170) 
  (hSI2 : SI = (P * R * T) / 100) : 
  P = 250 := 
by 
  sorry

end sum_lent_250_l2200_220030


namespace solve_for_x_l2200_220047

theorem solve_for_x (x : ℝ) (y : ℝ) (h : y = 3 * x) :
  (4 * y^2 + y + 6 = 3 * (9 * x^2 + y + 3)) ↔ (x = 1 ∨ x = -1/3) :=
by
  sorry

end solve_for_x_l2200_220047


namespace find_a_and_root_l2200_220072

def equation_has_double_root (a x : ℝ) : Prop :=
  a * x^2 + 4 * x - 1 = 0

theorem find_a_and_root (a x : ℝ)
  (h_eqn : equation_has_double_root a x)
  (h_discriminant : 16 + 4 * a = 0) :
  a = -4 ∧ x = 1 / 2 :=
sorry

end find_a_and_root_l2200_220072


namespace remainder_b94_mod_55_eq_29_l2200_220007

theorem remainder_b94_mod_55_eq_29 :
  (5^94 + 7^94) % 55 = 29 := 
by
  -- conditions: local definitions for bn, modulo, etc.
  sorry

end remainder_b94_mod_55_eq_29_l2200_220007


namespace find_value_of_c_l2200_220013

noncomputable def parabola (b c x : ℝ) : ℝ := x^2 + b * x + c

theorem find_value_of_c (b c : ℝ) 
    (h1 : parabola b c 1 = 2)
    (h2 : parabola b c 5 = 2) :
    c = 7 :=
by
  sorry

end find_value_of_c_l2200_220013


namespace simplify_fraction_subtraction_l2200_220005

theorem simplify_fraction_subtraction : (7 / 3) - (5 / 6) = 3 / 2 := by
  sorry

end simplify_fraction_subtraction_l2200_220005


namespace max_true_statements_l2200_220033

theorem max_true_statements {p q : ℝ} (hp : p > 0) (hq : q < 0) :
  ∀ (s1 s2 s3 s4 s5 : Prop), 
  s1 = (1 / p > 1 / q) →
  s2 = (p^3 > q^3) →
  s3 = (p^2 < q^2) →
  s4 = (p > 0) →
  s5 = (q < 0) →
  s1 ∧ s2 ∧ s4 ∧ s5 ∧ ¬s3 → 
  ∃ m : ℕ, m = 4 := 
by {
  sorry
}

end max_true_statements_l2200_220033


namespace dave_paid_more_l2200_220055

-- Definitions based on conditions in the problem statement
def total_pizza_cost : ℕ := 11  -- Total cost of the pizza in dollars
def num_slices : ℕ := 8  -- Total number of slices in the pizza
def plain_pizza_cost : ℕ := 8  -- Cost of the plain pizza in dollars
def anchovies_cost : ℕ := 2  -- Extra cost of adding anchovies in dollars
def mushrooms_cost : ℕ := 1  -- Extra cost of adding mushrooms in dollars
def dave_slices : ℕ := 7  -- Number of slices Dave ate
def doug_slices : ℕ := 1  -- Number of slices Doug ate
def doug_payment : ℕ := 1  -- Amount Doug paid in dollars
def dave_payment : ℕ := total_pizza_cost - doug_payment  -- Amount Dave paid in dollars

-- Prove that Dave paid 9 dollars more than Doug
theorem dave_paid_more : dave_payment - doug_payment = 9 := by
  -- Proof to be filled in
  sorry

end dave_paid_more_l2200_220055


namespace pyramid_base_length_l2200_220028

theorem pyramid_base_length (A s h : ℝ): A = 120 ∧ h = 40 ∧ (A = 1/2 * s * h) → s = 6 := 
by
  sorry

end pyramid_base_length_l2200_220028


namespace one_set_working_communication_possible_l2200_220040

variable (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1)

def P_A : ℝ := p^3
def P_B : ℝ := p^3
def P_not_A : ℝ := 1 - p^3
def P_not_B : ℝ := 1 - p^3

theorem one_set_working : 2 * P_A p - 2 * (P_A p)^2 = 2 * p^3 - 2 * p^6 :=
by 
  sorry

theorem communication_possible : 2 * P_A p - (P_A p)^2 = 2 * p^3 - p^6 :=
by 
  sorry

end one_set_working_communication_possible_l2200_220040


namespace x_value_l2200_220026

def x_is_75_percent_greater (x : ℝ) (y : ℝ) : Prop := x = y + 0.75 * y

theorem x_value (x : ℝ) : x_is_75_percent_greater x 150 → x = 262.5 :=
by
  intro h
  rw [x_is_75_percent_greater] at h
  sorry

end x_value_l2200_220026


namespace sequence_from_520_to_523_is_0_to_3_l2200_220065

theorem sequence_from_520_to_523_is_0_to_3 
  (repeating_pattern : ℕ → ℕ)
  (h_periodic : ∀ n, repeating_pattern (n + 5) = repeating_pattern n) :
  ((repeating_pattern 520, repeating_pattern 521, repeating_pattern 522, repeating_pattern 523) = (repeating_pattern 0, repeating_pattern 1, repeating_pattern 2, repeating_pattern 3)) :=
by {
  sorry
}

end sequence_from_520_to_523_is_0_to_3_l2200_220065


namespace value_of_a_l2200_220010

theorem value_of_a (x y a : ℝ) (h1 : x = 1) (h2 : y = 2) (h3 : 3 * x - a * y = 1) : a = 1 := by
  sorry

end value_of_a_l2200_220010


namespace brogan_total_red_apples_l2200_220050

def red_apples (total_apples percentage_red : ℕ) : ℕ :=
  (total_apples * percentage_red) / 100

theorem brogan_total_red_apples :
  red_apples 20 40 + red_apples 20 50 = 18 :=
by
  sorry

end brogan_total_red_apples_l2200_220050


namespace clock_hands_coincide_22_times_clock_hands_straight_angle_24_times_clock_hands_right_angle_48_times_l2200_220092

theorem clock_hands_coincide_22_times
  (minute_hand_cycles_24_hours : ℕ := 24)
  (hour_hand_cycles_24_hours : ℕ := 2)
  (minute_hand_overtakes_hour_hand_per_12_hours : ℕ := 11) :
  2 * minute_hand_overtakes_hour_hand_per_12_hours = 22 :=
by
  -- Proof should be filled here
  sorry

theorem clock_hands_straight_angle_24_times
  (hours_in_day : ℕ := 24)
  (straight_angle_per_hour : ℕ := 1) :
  hours_in_day * straight_angle_per_hour = 24 :=
by
  -- Proof should be filled here
  sorry

theorem clock_hands_right_angle_48_times
  (hours_in_day : ℕ := 24)
  (right_angles_per_hour : ℕ := 2) :
  hours_in_day * right_angles_per_hour = 48 :=
by
  -- Proof should be filled here
  sorry

end clock_hands_coincide_22_times_clock_hands_straight_angle_24_times_clock_hands_right_angle_48_times_l2200_220092


namespace more_likely_millionaire_city_resident_l2200_220059

noncomputable def probability_A (M N m n : ℝ) : ℝ :=
  m * M / (m * M + n * N)

noncomputable def probability_B (M N : ℝ) : ℝ :=
  M / (M + N)

theorem more_likely_millionaire_city_resident
  (M N : ℕ) (m n : ℝ) (hM : M > 0) (hN : N > 0)
  (hm : m > 10^6) (hn : n ≤ 10^6) :
  probability_A M N m n > probability_B M N :=
by {
  sorry
}

end more_likely_millionaire_city_resident_l2200_220059


namespace train_pass_bridge_time_l2200_220063

/-- A train is 460 meters long and runs at a speed of 45 km/h. The bridge is 140 meters long. 
Prove that the time it takes for the train to pass the bridge is 48 seconds. -/
theorem train_pass_bridge_time (train_length : ℝ) (bridge_length : ℝ) (speed_kmh : ℝ) 
  (h_train_length : train_length = 460) 
  (h_bridge_length : bridge_length = 140)
  (h_speed_kmh : speed_kmh = 45)
  : (train_length + bridge_length) / (speed_kmh * 1000 / 3600) = 48 := 
by
  sorry

end train_pass_bridge_time_l2200_220063


namespace dart_within_triangle_probability_l2200_220012

theorem dart_within_triangle_probability (s : ℝ) : 
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  let triangle_area := (Real.sqrt 3 / 16) * s^2
  (triangle_area / hexagon_area) = 1 / 24 :=
by sorry

end dart_within_triangle_probability_l2200_220012


namespace solve_for_n_l2200_220087

theorem solve_for_n (n : ℕ) : 
  9^n * 9^n * 9^(2*n) = 81^4 → n = 2 :=
by
  sorry

end solve_for_n_l2200_220087


namespace find_f_at_6_l2200_220008

noncomputable def example_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (4 * x - 2) = x^2 - x + 2

theorem find_f_at_6 (f : ℝ → ℝ) (h : example_function f) : f 6 = 4 := 
by
  sorry

end find_f_at_6_l2200_220008


namespace intersection_complement_l2200_220051

universe u

def U := Real

def M : Set Real := { x | -2 ≤ x ∧ x ≤ 2 }

def N : Set Real := { x | x * (x - 3) ≤ 0 }

def complement_U (S : Set Real) : Set Real := { x | x ∉ S }

theorem intersection_complement :
  M ∩ (complement_U N) = { x | -2 ≤ x ∧ x < 0 } := by
  sorry

end intersection_complement_l2200_220051


namespace cricket_current_average_l2200_220080

theorem cricket_current_average (A : ℕ) (h1: 10 * A + 77 = 11 * (A + 4)) : 
  A = 33 := 
by 
  sorry

end cricket_current_average_l2200_220080


namespace polynomial_R_result_l2200_220082

noncomputable def polynomial_Q_R (z : ℤ) : Prop :=
  ∃ Q R : Polynomial ℂ, 
  z ^ 2020 + 1 = (z ^ 2 - z + 1) * Q + R ∧ R.degree < 2 ∧ R = 2

theorem polynomial_R_result :
  polynomial_Q_R z :=
by 
  sorry

end polynomial_R_result_l2200_220082


namespace find_weight_B_l2200_220099

-- Define the weights of A, B, and C
variables (A B C : ℝ)

-- Conditions
def avg_weight_ABC := A + B + C = 135
def avg_weight_AB := A + B = 80
def avg_weight_BC := B + C = 86

-- The statement to be proved
theorem find_weight_B (h1: avg_weight_ABC A B C) (h2: avg_weight_AB A B) (h3: avg_weight_BC B C) : B = 31 :=
sorry

end find_weight_B_l2200_220099


namespace ratio_of_millipedes_l2200_220053

-- Define the given conditions
def total_segments_needed : ℕ := 800
def first_millipede_segments : ℕ := 60
def millipedes_segments (x : ℕ) : ℕ := x
def ten_millipedes_segments : ℕ := 10 * 50

-- State the main theorem
theorem ratio_of_millipedes (x : ℕ) : 
  total_segments_needed = 60 + 2 * x + 10 * 50 →
  2 * x / 60 = 4 :=
sorry

end ratio_of_millipedes_l2200_220053


namespace boys_on_trip_l2200_220044

theorem boys_on_trip (B G : ℕ) 
    (h1 : G = B + (2 / 5 : ℚ) * B) 
    (h2 : 1 + 1 + 1 + B + G = 123) : 
    B = 50 := 
by 
  -- Proof skipped 
  sorry

end boys_on_trip_l2200_220044


namespace initial_men_count_l2200_220025

theorem initial_men_count (M : ℕ) (F : ℕ) (h1 : F = M * 22) (h2 : (M + 2280) * 5 = M * 20) : M = 760 := by
  sorry

end initial_men_count_l2200_220025


namespace comedies_in_terms_of_a_l2200_220062

variable (T a : ℝ)
variables (Comedies Dramas Action : ℝ)
axiom Condition1 : Comedies = 0.64 * T
axiom Condition2 : Dramas = 5 * a
axiom Condition3 : Action = a
axiom Condition4 : Comedies + Dramas + Action = T

theorem comedies_in_terms_of_a : Comedies = 10.67 * a :=
by sorry

end comedies_in_terms_of_a_l2200_220062


namespace complement_of_N_is_135_l2200_220088

-- Define the universal set M and subset N
def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {2, 4}

-- Prove that the complement of N in M is {1, 3, 5}
theorem complement_of_N_is_135 : M \ N = {1, 3, 5} := 
by
  sorry

end complement_of_N_is_135_l2200_220088


namespace avery_shirts_count_l2200_220090

theorem avery_shirts_count {S : ℕ} (h_total : S + 2 * S + S = 16) : S = 4 :=
by
  sorry

end avery_shirts_count_l2200_220090


namespace fewest_number_of_gymnasts_l2200_220057

theorem fewest_number_of_gymnasts (n : ℕ) (h : n % 2 = 0)
  (handshakes : ∀ (n : ℕ), (n * (n - 1) / 2) + n = 465) : 
  n = 30 :=
by
  sorry

end fewest_number_of_gymnasts_l2200_220057


namespace unique_positive_integers_pqr_l2200_220034

noncomputable def y : ℝ := Real.sqrt ((Real.sqrt 61) / 2 + 5 / 2)

lemma problem_condition (p q r : ℕ) (py : ℝ) :
  py = y^100
  ∧ py = 2 * (y^98)
  ∧ py = 16 * (y^96)
  ∧ py = 13 * (y^94)
  ∧ py = - y^50
  ∧ py = ↑p * y^46
  ∧ py = ↑q * y^44
  ∧ py = ↑r * y^40 :=
sorry

theorem unique_positive_integers_pqr : 
  ∃! (p q r : ℕ), 
    p = 37 ∧ q = 47 ∧ r = 298 ∧ 
    y^100 = 2 * y^98 + 16 * y^96 + 13 * y^94 - y^50 + ↑p * y^46 + ↑q * y^44 + ↑r * y^40 :=
sorry

end unique_positive_integers_pqr_l2200_220034


namespace gcd_euclidean_120_168_gcd_subtraction_459_357_l2200_220024

theorem gcd_euclidean_120_168 : Nat.gcd 120 168 = 24 := by
  sorry

theorem gcd_subtraction_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_euclidean_120_168_gcd_subtraction_459_357_l2200_220024


namespace probability_heads_and_3_l2200_220069

noncomputable def biased_coin_heads_prob : ℝ := 0.4
def die_sides : ℕ := 8

theorem probability_heads_and_3 : biased_coin_heads_prob * (1 / die_sides) = 0.05 := sorry

end probability_heads_and_3_l2200_220069


namespace trader_profit_loss_l2200_220064

noncomputable def profit_loss_percentage (sp1 sp2: ℝ) (gain_loss_rate1 gain_loss_rate2: ℝ) : ℝ :=
  let cp1 := sp1 / (1 + gain_loss_rate1)
  let cp2 := sp2 / (1 - gain_loss_rate2)
  let tcp := cp1 + cp2
  let tsp := sp1 + sp2
  let profit_or_loss := tsp - tcp
  profit_or_loss / tcp * 100

theorem trader_profit_loss : 
  profit_loss_percentage 325475 325475 0.15 0.15 = -2.33 := 
by 
  sorry

end trader_profit_loss_l2200_220064


namespace max_ab_l2200_220027

theorem max_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 6) : ab ≤ 9 / 2 :=
by
  sorry

end max_ab_l2200_220027


namespace problem_statement_l2200_220073

theorem problem_statement : (29.7 + 83.45) - 0.3 = 112.85 := sorry

end problem_statement_l2200_220073


namespace solve_system_l2200_220077

theorem solve_system (x y : ℚ) 
  (h1 : x + 2 * y = -1) 
  (h2 : 2 * x + y = 3) : 
  x + y = 2 / 3 := 
sorry

end solve_system_l2200_220077


namespace circle_tangent_sum_radii_l2200_220058

theorem circle_tangent_sum_radii :
  let r1 := 6 + 2 * Real.sqrt 6
  let r2 := 6 - 2 * Real.sqrt 6
  r1 + r2 = 12 :=
by
  sorry

end circle_tangent_sum_radii_l2200_220058


namespace triangle_is_right_angled_l2200_220075

theorem triangle_is_right_angled
  (a b c : ℝ)
  (h1 : a ≠ c)
  (h2 : a > 0) 
  (h3 : b > 0) 
  (h4 : c > 0)
  (h5 : ∃ x : ℝ, x^2 + 2*a*x + b^2 = 0 ∧ x^2 + 2*c*x - b^2 = 0 ∧ x ≠ 0) :
  c^2 + b^2 = a^2 :=
by sorry

end triangle_is_right_angled_l2200_220075


namespace find_integer_n_l2200_220079

theorem find_integer_n : 
  ∃ n : ℤ, 50 ≤ n ∧ n ≤ 150 ∧ (n % 7 = 0) ∧ (n % 9 = 3) ∧ (n % 4 = 3) ∧ n = 147 :=
by 
  -- sorry is used here as a placeholder for the actual proof
  sorry

end find_integer_n_l2200_220079


namespace t_shirt_cost_l2200_220056

theorem t_shirt_cost (total_amount_spent : ℝ) (number_of_t_shirts : ℕ) (cost_per_t_shirt : ℝ)
  (h0 : total_amount_spent = 201) 
  (h1 : number_of_t_shirts = 22)
  (h2 : cost_per_t_shirt = total_amount_spent / number_of_t_shirts) :
  cost_per_t_shirt = 9.14 := 
sorry

end t_shirt_cost_l2200_220056


namespace probability_heads_3_ace_l2200_220068

def fair_coin_flip : ℕ := 2
def six_sided_die : ℕ := 6
def standard_deck_cards : ℕ := 52

def successful_outcomes : ℕ := 1 * 1 * 4
def total_possible_outcomes : ℕ := fair_coin_flip * six_sided_die * standard_deck_cards

theorem probability_heads_3_ace :
  (successful_outcomes : ℚ) / (total_possible_outcomes : ℚ) = 1 / 156 := 
sorry

end probability_heads_3_ace_l2200_220068


namespace specified_time_eq_l2200_220036

noncomputable def slow_horse_days (x : ℝ) := x + 1
noncomputable def fast_horse_days (x : ℝ) := x - 3

theorem specified_time_eq (x : ℝ) (h1 : slow_horse_days x > 0) (h2 : fast_horse_days x > 0) :
  (900 / slow_horse_days x) * 2 = 900 / fast_horse_days x :=
by
  sorry

end specified_time_eq_l2200_220036


namespace part_I_part_II_l2200_220015

def f (x a : ℝ) := |x - a| + |x - 1|

theorem part_I {x : ℝ} : Set.Icc 0 4 = {y | f y 3 ≤ 4} := 
sorry

theorem part_II {a : ℝ} : (∀ x, ¬ (f x a < 2)) ↔ a ∈ Set.Ici 3 ∪ Set.Iic (-1) :=
sorry

end part_I_part_II_l2200_220015


namespace yellow_ball_kids_l2200_220003

theorem yellow_ball_kids (total_kids white_ball_kids both_ball_kids : ℕ) :
  total_kids = 35 → white_ball_kids = 26 → both_ball_kids = 19 → 
  (total_kids = white_ball_kids + (total_kids - both_ball_kids)) → 
  (total_kids - (white_ball_kids - both_ball_kids)) = 28 :=
by
  sorry

end yellow_ball_kids_l2200_220003


namespace min_coins_cover_99_l2200_220085

def coin_values : List ℕ := [1, 5, 10, 25, 50]

noncomputable def min_coins_cover (n : ℕ) : ℕ := sorry

theorem min_coins_cover_99 : min_coins_cover 99 = 9 :=
  sorry

end min_coins_cover_99_l2200_220085


namespace germs_left_percentage_l2200_220038

-- Defining the conditions
def first_spray_kill_percentage : ℝ := 0.50
def second_spray_kill_percentage : ℝ := 0.25
def overlap_percentage : ℝ := 0.05
def total_kill_percentage : ℝ := first_spray_kill_percentage + second_spray_kill_percentage - overlap_percentage

-- The statement to be proved
theorem germs_left_percentage :
  1 - total_kill_percentage = 0.30 :=
by
  -- The proof would go here.
  sorry

end germs_left_percentage_l2200_220038


namespace isosceles_triangle_vertex_angle_l2200_220086

theorem isosceles_triangle_vertex_angle (a b : ℕ) (h : a = 2 * b) 
  (h1 : a + b + b = 180): a = 90 ∨ a = 36 :=
by
  sorry

end isosceles_triangle_vertex_angle_l2200_220086


namespace max_consecutive_sum_terms_l2200_220066

theorem max_consecutive_sum_terms (S : ℤ) (n : ℕ) (H1 : S = 2015) (H2 : 0 < n) :
  (∃ a : ℤ, S = (a * n + (n * (n - 1)) / 2)) → n = 4030 :=
sorry

end max_consecutive_sum_terms_l2200_220066


namespace find_x_l2200_220031

theorem find_x (x y z : ℝ) (h1 : x^2 / y = 4) (h2 : y^2 / z = 9) (h3 : z^2 / x = 16) : x = 4 :=
sorry

end find_x_l2200_220031


namespace derivative_of_log_base_3_derivative_of_exp_base_2_l2200_220032

noncomputable def log_base_3_deriv (x : ℝ) : ℝ := (Real.log x / Real.log 3)
noncomputable def exp_base_2_deriv (x : ℝ) : ℝ := Real.exp (x * Real.log 2)

theorem derivative_of_log_base_3 (x : ℝ) (h : x > 0) :
  (log_base_3_deriv x) = (1 / (x * Real.log 3)) :=
by
  sorry

theorem derivative_of_exp_base_2 (x : ℝ) :
  (exp_base_2_deriv x) = (Real.exp (x * Real.log 2) * Real.log 2) :=
by
  sorry

end derivative_of_log_base_3_derivative_of_exp_base_2_l2200_220032


namespace Isaabel_math_pages_l2200_220098

theorem Isaabel_math_pages (x : ℕ) (total_problems : ℕ) (reading_pages : ℕ) (problems_per_page : ℕ) :
  (reading_pages * problems_per_page = 20) ∧ (total_problems = 30) →
  x * problems_per_page + 20 = total_problems →
  x = 2 := by
  sorry

end Isaabel_math_pages_l2200_220098


namespace sheila_will_attend_picnic_l2200_220071

noncomputable def prob_sheila_attends_picnic (P_Rain P_Attend_if_Rain P_Attend_if_Sunny P_Special : ℝ) : ℝ :=
  let P_Sunny := 1 - P_Rain
  let P_Rain_and_Attend := P_Rain * P_Attend_if_Rain
  let P_Sunny_and_Attend := P_Sunny * P_Attend_if_Sunny
  let P_Attends := P_Rain_and_Attend + P_Sunny_and_Attend + P_Special - P_Rain_and_Attend * P_Special - P_Sunny_and_Attend * P_Special
  P_Attends

theorem sheila_will_attend_picnic :
  prob_sheila_attends_picnic 0.3 0.25 0.7 0.15 = 0.63025 :=
by
  sorry

end sheila_will_attend_picnic_l2200_220071


namespace car_trip_cost_proof_l2200_220035

def car_trip_cost 
  (d1 d2 d3 d4 : ℕ) 
  (efficiency : ℕ) 
  (cost_per_gallon : ℕ) 
  (total_distance : ℕ) 
  (gallons_used : ℕ) 
  (cost : ℕ) : Prop :=
  d1 = 8 ∧
  d2 = 6 ∧
  d3 = 12 ∧
  d4 = 2 * d3 ∧
  efficiency = 25 ∧
  cost_per_gallon = 250 ∧
  total_distance = d1 + d2 + d3 + d4 ∧
  gallons_used = total_distance / efficiency ∧
  cost = gallons_used * cost_per_gallon ∧
  cost = 500

theorem car_trip_cost_proof : car_trip_cost 8 6 12 (2 * 12) 25 250 (8 + 6 + 12 + (2 * 12)) ((8 + 6 + 12 + (2 * 12)) / 25) (((8 + 6 + 12 + (2 * 12)) / 25) * 250) :=
by 
  sorry

end car_trip_cost_proof_l2200_220035


namespace range_of_inverse_proportion_function_l2200_220093

noncomputable def f (x : ℝ) : ℝ := 6 / x

theorem range_of_inverse_proportion_function (x : ℝ) (hx : x > 2) : 
  0 < f x ∧ f x < 3 :=
sorry

end range_of_inverse_proportion_function_l2200_220093


namespace percentage_of_alcohol_in_original_solution_l2200_220097

noncomputable def alcohol_percentage_in_original_solution (P: ℝ) (V_original: ℝ) (V_water: ℝ) (percentage_new: ℝ): ℝ :=
  (P * V_original) / (V_original + V_water) * 100

theorem percentage_of_alcohol_in_original_solution : 
  ∀ (P: ℝ) (V_original : ℝ) (V_water : ℝ) (percentage_new : ℝ), 
  V_original = 3 → 
  V_water = 1 → 
  percentage_new = 24.75 →
  alcohol_percentage_in_original_solution P V_original V_water percentage_new = 33 := 
by
  sorry

end percentage_of_alcohol_in_original_solution_l2200_220097


namespace angle_condition_l2200_220081

theorem angle_condition
  {θ : ℝ}
  (h₀ : 0 ≤ θ)
  (h₁ : θ < π)
  (h₂ : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → x^2 * Real.cos θ - x * (1 - x) + 2 * (1 - x)^2 * Real.sin θ > 0) :
  0 < θ ∧ θ < π / 2 :=
by
  sorry

end angle_condition_l2200_220081
