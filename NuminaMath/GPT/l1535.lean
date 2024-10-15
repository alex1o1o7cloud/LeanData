import Mathlib

namespace NUMINAMATH_GPT_book_price_is_correct_l1535_153557

-- Define the cost of the album
def album_cost : ℝ := 20

-- Define the discount percentage
def discount_percentage : ℝ := 0.30

-- Calculate the CD cost
def cd_cost : ℝ := album_cost * (1 - discount_percentage)

-- Define the additional cost of the book over the CD
def book_cd_diff : ℝ := 4

-- Calculate the book cost
def book_cost : ℝ := cd_cost + book_cd_diff

-- State the proposition to be proved
theorem book_price_is_correct : book_cost = 18 := by
  -- Provide the details of the calculations (optionally)
  sorry

end NUMINAMATH_GPT_book_price_is_correct_l1535_153557


namespace NUMINAMATH_GPT_complex_subtraction_l1535_153535

open Complex

def z1 : ℂ := 3 + 4 * I
def z2 : ℂ := 1 + I

theorem complex_subtraction : z1 - z2 = 2 + 3 * I := by
  sorry

end NUMINAMATH_GPT_complex_subtraction_l1535_153535


namespace NUMINAMATH_GPT_mrs_hilt_additional_rocks_l1535_153580

-- Definitions from the conditions
def total_rocks : ℕ := 125
def rocks_she_has : ℕ := 64
def additional_rocks_needed : ℕ := total_rocks - rocks_she_has

-- The theorem to prove the question equals the answer given the conditions
theorem mrs_hilt_additional_rocks : additional_rocks_needed = 61 := 
by
  sorry

end NUMINAMATH_GPT_mrs_hilt_additional_rocks_l1535_153580


namespace NUMINAMATH_GPT_together_work_days_l1535_153522

theorem together_work_days (A B C : ℕ) (nine_days : A = 9) (eighteen_days : B = 18) (twelve_days : C = 12) :
  (1 / A + 1 / B + 1 / C) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_together_work_days_l1535_153522


namespace NUMINAMATH_GPT_exp_value_l1535_153506

theorem exp_value (a : ℝ) (m n : ℕ) (h1 : a^m = 2) (h2 : a^n = 3) : a^(m + 2 * n) = 18 := 
by
  sorry

end NUMINAMATH_GPT_exp_value_l1535_153506


namespace NUMINAMATH_GPT_find_A_from_complement_l1535_153508

-- Define the universal set U
def U : Set ℕ := {0, 1, 2}

-- Define the complement of set A in U
variable (A : Set ℕ)
def complement_U_A : Set ℕ := {n | n ∈ U ∧ n ∉ A}

-- Define the condition given in the problem
axiom h : complement_U_A A = {2}

-- State the theorem to be proven
theorem find_A_from_complement : A = {0, 1} :=
sorry

end NUMINAMATH_GPT_find_A_from_complement_l1535_153508


namespace NUMINAMATH_GPT_colten_chickens_l1535_153560

/-
Define variables to represent the number of chickens each person has.
-/

variables (C : ℕ)   -- Number of chickens Colten has.
variables (S : ℕ)   -- Number of chickens Skylar has.
variables (Q : ℕ)   -- Number of chickens Quentin has.

/-
Define the given conditions
-/
def condition1 := Q + S + C = 383
def condition2 := Q = 2 * S + 25
def condition3 := S = 3 * C - 4

theorem colten_chickens : C = 37 :=
by
  -- Proof elaboration to be done with sorry for the auto proof
  sorry

end NUMINAMATH_GPT_colten_chickens_l1535_153560


namespace NUMINAMATH_GPT_second_puppy_weight_l1535_153564

variables (p1 p2 c1 c2 : ℝ)

-- Conditions from the problem statement
axiom h1 : p1 + p2 + c1 + c2 = 36
axiom h2 : p1 + c2 = 3 * c1
axiom h3 : p1 + c1 = c2
axiom h4 : p2 = 1.5 * p1

-- The question to prove: how much does the second puppy weigh
theorem second_puppy_weight : p2 = 108 / 11 :=
by sorry

end NUMINAMATH_GPT_second_puppy_weight_l1535_153564


namespace NUMINAMATH_GPT_find_integer_pairs_l1535_153584

theorem find_integer_pairs (x y : ℤ) :
  x^4 + (y+2)^3 = (x+2)^4 ↔ (x, y) = (0, 0) ∨ (x, y) = (-1, -2) := sorry

end NUMINAMATH_GPT_find_integer_pairs_l1535_153584


namespace NUMINAMATH_GPT_find_m_l1535_153555

theorem find_m (m : ℕ) :
  (∀ x : ℝ, -2 * x ^ 2 + 5 * x - 2 <= 9 / m) →
  m = 8 :=
sorry

end NUMINAMATH_GPT_find_m_l1535_153555


namespace NUMINAMATH_GPT_ladder_slip_l1535_153554

theorem ladder_slip (l : ℝ) (d1 d2 : ℝ) (h1 h2 : ℝ) :
  l = 30 → d1 = 8 → h1^2 + d1^2 = l^2 → h2 = h1 - 4 → 
  (h2^2 + (d1 + d2)^2 = l^2) → d2 = 2 :=
by
  intros h_l h_d1 h_h1_eq h_h2 h2_eq_l   
  sorry

end NUMINAMATH_GPT_ladder_slip_l1535_153554


namespace NUMINAMATH_GPT_cube_edge_length_l1535_153546

theorem cube_edge_length (n_edges : ℕ) (total_length : ℝ) (length_one_edge : ℝ) 
  (h1: n_edges = 12) (h2: total_length = 96) : length_one_edge = 8 :=
by
  sorry

end NUMINAMATH_GPT_cube_edge_length_l1535_153546


namespace NUMINAMATH_GPT_car_speed_problem_l1535_153529

theorem car_speed_problem (x : ℝ) (h1 : ∀ x, x + 30 / 2 = 65) : x = 100 :=
by
  sorry

end NUMINAMATH_GPT_car_speed_problem_l1535_153529


namespace NUMINAMATH_GPT_surjective_injective_eq_l1535_153586

theorem surjective_injective_eq (f g : ℕ → ℕ) 
  (hf : Function.Surjective f) 
  (hg : Function.Injective g) 
  (h : ∀ n : ℕ, f n ≥ g n) : 
  ∀ n : ℕ, f n = g n := 
by
  sorry

end NUMINAMATH_GPT_surjective_injective_eq_l1535_153586


namespace NUMINAMATH_GPT_probability_snow_first_week_l1535_153504

theorem probability_snow_first_week :
  let p1 := 1/4
  let p2 := 1/3
  let no_snow := (3/4)^4 * (2/3)^3
  let snows_at_least_once := 1 - no_snow
  snows_at_least_once = 29 / 32 := by
  sorry

end NUMINAMATH_GPT_probability_snow_first_week_l1535_153504


namespace NUMINAMATH_GPT_black_shirts_in_pack_l1535_153588

-- defining the conditions
variables (B : ℕ) -- the number of black shirts in each pack
variable (total_shirts : ℕ := 21)
variable (yellow_shirts_per_pack : ℕ := 2)
variable (black_packs : ℕ := 3)
variable (yellow_packs : ℕ := 3)

-- ensuring the conditions are met, the total shirts equals 21
def total_black_shirts := black_packs * B
def total_yellow_shirts := yellow_packs * yellow_shirts_per_pack

-- the proof problem
theorem black_shirts_in_pack : total_black_shirts + total_yellow_shirts = total_shirts → B = 5 := by
  sorry

end NUMINAMATH_GPT_black_shirts_in_pack_l1535_153588


namespace NUMINAMATH_GPT_lisa_likes_only_last_digit_zero_l1535_153524

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 10 = 0 ∨ n % 10 = 5

def is_divisible_by_2 (n : ℕ) : Prop :=
  n % 10 = 0 ∨ n % 10 = 2 ∨ n % 10 = 4 ∨ n % 10 = 6 ∨ n % 10 = 8

def is_divisible_by_5_and_2 (n : ℕ) : Prop :=
  is_divisible_by_5 n ∧ is_divisible_by_2 n

theorem lisa_likes_only_last_digit_zero : ∀ n, is_divisible_by_5_and_2 n → n % 10 = 0 :=
by
  sorry

end NUMINAMATH_GPT_lisa_likes_only_last_digit_zero_l1535_153524


namespace NUMINAMATH_GPT_elevator_time_l1535_153527

theorem elevator_time :
  ∀ (floors steps_per_floor steps_per_second extra_time : ℕ) (elevator_time_sec elevator_time_min : ℚ),
    floors = 8 →
    steps_per_floor = 30 →
    steps_per_second = 3 →
    extra_time = 30 →
    elevator_time_sec = ((floors * steps_per_floor) / steps_per_second) - extra_time →
    elevator_time_min = elevator_time_sec / 60 →
    elevator_time_min = 0.833 :=
by
  intros floors steps_per_floor steps_per_second extra_time elevator_time_sec elevator_time_min
  intros h_floors h_steps_per_floor h_steps_per_second h_extra_time h_elevator_time_sec h_elevator_time_min
  rw [h_floors, h_steps_per_floor, h_steps_per_second, h_extra_time] at *
  sorry

end NUMINAMATH_GPT_elevator_time_l1535_153527


namespace NUMINAMATH_GPT_five_equal_angles_72_degrees_l1535_153593

theorem five_equal_angles_72_degrees
  (five_rays : ℝ)
  (equal_angles : ℝ) 
  (sum_angles : five_rays * equal_angles = 360) :
  equal_angles = 72 :=
by
  sorry

end NUMINAMATH_GPT_five_equal_angles_72_degrees_l1535_153593


namespace NUMINAMATH_GPT_post_office_mail_in_six_months_l1535_153534

/-- The number of pieces of mail the post office receives per day -/
def mail_per_day : ℕ := 60 + 20

/-- The number of days in six months, assuming each month has 30 days -/
def days_in_six_months : ℕ := 6 * 30

/-- The total number of pieces of mail handled in six months -/
def total_mail_in_six_months : ℕ := mail_per_day * days_in_six_months

/-- The post office handles 14400 pieces of mail in six months -/
theorem post_office_mail_in_six_months : total_mail_in_six_months = 14400 := by
  sorry

end NUMINAMATH_GPT_post_office_mail_in_six_months_l1535_153534


namespace NUMINAMATH_GPT_john_children_l1535_153532

def total_notebooks (john_notebooks : ℕ) (wife_notebooks : ℕ) (children : ℕ) := 
  2 * children + 5 * children

theorem john_children (c : ℕ) (h : total_notebooks 2 5 c = 21) :
  c = 3 :=
sorry

end NUMINAMATH_GPT_john_children_l1535_153532


namespace NUMINAMATH_GPT_find_acute_angle_l1535_153500

theorem find_acute_angle (α : ℝ) (h1 : 0 < α ∧ α < 90) (h2 : ∃ k : ℤ, 10 * α = α + k * 360) :
  α = 40 ∨ α = 80 :=
by
  sorry

end NUMINAMATH_GPT_find_acute_angle_l1535_153500


namespace NUMINAMATH_GPT_area_of_defined_region_eq_14_point_4_l1535_153559

def defined_region (x y : ℝ) : Prop :=
  |5 * x - 20| + |3 * y + 9| ≤ 6

def region_area : ℝ :=
  14.4

theorem area_of_defined_region_eq_14_point_4 :
  (∃ (x y : ℝ), defined_region x y) → region_area = 14.4 :=
by
  sorry

end NUMINAMATH_GPT_area_of_defined_region_eq_14_point_4_l1535_153559


namespace NUMINAMATH_GPT_percent_round_trip_tickets_l1535_153526

variable (P : ℕ) -- total number of passengers

def passengers_with_round_trip_tickets (P : ℕ) : ℕ :=
  2 * (P / 5 / 2)

theorem percent_round_trip_tickets (P : ℕ) : 
  passengers_with_round_trip_tickets P = 2 * (P / 5 / 2) :=
by
  sorry

end NUMINAMATH_GPT_percent_round_trip_tickets_l1535_153526


namespace NUMINAMATH_GPT_train_crossing_time_approx_l1535_153525

noncomputable def train_length : ℝ := 90 -- in meters
noncomputable def speed_kmh : ℝ := 124 -- in km/hr
noncomputable def conversion_factor : ℝ := 1000 / 3600 -- km/hr to m/s conversion factor
noncomputable def speed_ms : ℝ := speed_kmh * conversion_factor -- speed in m/s
noncomputable def time_to_cross : ℝ := train_length / speed_ms -- time in seconds

theorem train_crossing_time_approx :
  abs (time_to_cross - 2.61) < 0.01 := 
by 
  sorry

end NUMINAMATH_GPT_train_crossing_time_approx_l1535_153525


namespace NUMINAMATH_GPT_solve_quadratic_eq_l1535_153573

theorem solve_quadratic_eq : ∃ (a b : ℕ), a = 145 ∧ b = 7 ∧ a + b = 152 ∧ 
  ∀ x, x = Real.sqrt a - b → x^2 + 14 * x = 96 :=
by 
  use 145, 7
  simp
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l1535_153573


namespace NUMINAMATH_GPT_point_A_in_QuadrantIII_l1535_153537

-- Define the Cartesian Point structure
structure Point where
  x : ℝ
  y : ℝ

-- Define the condition for point being in Quadrant III
def inQuadrantIII (P : Point) : Prop :=
  P.x < 0 ∧ P.y < 0

-- Given point A
def A : Point := { x := -1, y := -2 }

-- The theorem stating that point A lies in Quadrant III
theorem point_A_in_QuadrantIII : inQuadrantIII A :=
  by
    sorry

end NUMINAMATH_GPT_point_A_in_QuadrantIII_l1535_153537


namespace NUMINAMATH_GPT_inequality_proof_l1535_153596

theorem inequality_proof (a b : Real) (h1 : (1 / a) < (1 / b)) (h2 : (1 / b) < 0) : 
  (b / a) + (a / b) > 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1535_153596


namespace NUMINAMATH_GPT_mean_transformation_l1535_153578

theorem mean_transformation (x1 x2 x3 x4 : ℝ)
                            (h_pos : 0 < x1 ∧ 0 < x2 ∧ 0 < x3 ∧ 0 < x4)
                            (s2 : ℝ)
                            (h_var : s2 = (1 / 4) * (x1^2 + x2^2 + x3^2 + x4^2 - 16)) :
                            (x1 + 2 + x2 + 2 + x3 + 2 + x4 + 2) / 4 = 4 :=
by
  sorry

end NUMINAMATH_GPT_mean_transformation_l1535_153578


namespace NUMINAMATH_GPT_two_point_two_five_as_fraction_l1535_153576

theorem two_point_two_five_as_fraction : (2.25 : ℚ) = 9 / 4 := 
by 
  -- Proof steps would be added here
  sorry

end NUMINAMATH_GPT_two_point_two_five_as_fraction_l1535_153576


namespace NUMINAMATH_GPT_team_combination_count_l1535_153565

theorem team_combination_count (n k : ℕ) (hn : n = 7) (hk : k = 4) :
  ∃ m, m = Nat.choose n k ∧ m = 35 :=
by
  sorry

end NUMINAMATH_GPT_team_combination_count_l1535_153565


namespace NUMINAMATH_GPT_remainder_777_777_mod_13_l1535_153572

theorem remainder_777_777_mod_13 : (777^777) % 13 = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_777_777_mod_13_l1535_153572


namespace NUMINAMATH_GPT_diff_of_cubes_is_sum_of_squares_l1535_153544

theorem diff_of_cubes_is_sum_of_squares (n : ℤ) : 
  (n+2)^3 - n^3 = n^2 + (n+2)^2 + (2*n+2)^2 := 
by sorry

end NUMINAMATH_GPT_diff_of_cubes_is_sum_of_squares_l1535_153544


namespace NUMINAMATH_GPT_if_2_3_4_then_1_if_1_3_4_then_2_l1535_153562

variables {Plane Line : Type} 
variables (α β : Plane) (m n : Line)

-- assuming the perpendicular relationships as predicates
variable (perp : Plane → Plane → Prop) -- perpendicularity between planes
variable (perp' : Line → Line → Prop) -- perpendicularity between lines
variable (perp'' : Line → Plane → Prop) -- perpendicularity between line and plane

theorem if_2_3_4_then_1 :
  perp α β → perp'' m β → perp'' n α → perp' m n :=
by
  sorry

theorem if_1_3_4_then_2 :
  perp' m n → perp'' m β → perp'' n α → perp α β :=
by
  sorry

end NUMINAMATH_GPT_if_2_3_4_then_1_if_1_3_4_then_2_l1535_153562


namespace NUMINAMATH_GPT_expression_value_l1535_153511

-- Define the given condition as an assumption
variable (x : ℝ)
variable (h : 2 * x^2 + 3 * x - 1 = 7)

-- Define the target expression and the required result
theorem expression_value :
  4 * x^2 + 6 * x + 9 = 25 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l1535_153511


namespace NUMINAMATH_GPT_parabola_vertex_coordinates_l1535_153597

theorem parabola_vertex_coordinates {a b c : ℝ} (h_eq : ∀ x : ℝ, a * x^2 + b * x + c = 3)
  (h_root : a * 2^2 + b * 2 + c = 3) (h_symm : ∀ x : ℝ, a * (2 - x)^2 + b * (2 - x) + c = a * x^2 + b * x + c) :
  (2, 3) = (2, 3) :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_coordinates_l1535_153597


namespace NUMINAMATH_GPT_man_l1535_153594

theorem man's_speed_upstream :
  ∀ (R : ℝ), (R + 1.5 = 11) → (R - 1.5 = 8) :=
by
  intros R h
  sorry

end NUMINAMATH_GPT_man_l1535_153594


namespace NUMINAMATH_GPT_find_y_given_x_eq_0_l1535_153514

theorem find_y_given_x_eq_0 (t : ℚ) (x y : ℚ) (h1 : x = 3 - 2 * t) (h2 : y = 3 * t + 6) (h3 : x = 0) : 
  y = 21 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_y_given_x_eq_0_l1535_153514


namespace NUMINAMATH_GPT_translate_line_downwards_l1535_153541

theorem translate_line_downwards :
  ∀ (x : ℝ), (∀ (y : ℝ), (y = 2 * x + 1) → (y - 2 = 2 * x - 1)) :=
by
  intros x y h
  rw [h]
  sorry

end NUMINAMATH_GPT_translate_line_downwards_l1535_153541


namespace NUMINAMATH_GPT_equation_solution_l1535_153591

noncomputable def solve_equation (x : ℝ) : Prop :=
  (4 / (x - 1) + 1 / (1 - x) = 1) → x = 4

theorem equation_solution (x : ℝ) (h : 4 / (x - 1) + 1 / (1 - x) = 1) : x = 4 := by
  sorry

end NUMINAMATH_GPT_equation_solution_l1535_153591


namespace NUMINAMATH_GPT_function_intersection_at_most_one_l1535_153545

theorem function_intersection_at_most_one (f : ℝ → ℝ) (a : ℝ) :
  ∃! b, f b = a := sorry

end NUMINAMATH_GPT_function_intersection_at_most_one_l1535_153545


namespace NUMINAMATH_GPT_eric_green_marbles_l1535_153540

theorem eric_green_marbles (total_marbles white_marbles blue_marbles : ℕ) (h_total : total_marbles = 20)
  (h_white : white_marbles = 12) (h_blue : blue_marbles = 6) :
  total_marbles - (white_marbles + blue_marbles) = 2 := 
by
  sorry

end NUMINAMATH_GPT_eric_green_marbles_l1535_153540


namespace NUMINAMATH_GPT_parallel_vectors_m_l1535_153592

theorem parallel_vectors_m (m : ℝ) :
  let a := (1, 2)
  let b := (m, m + 1)
  a.1 * b.2 = a.2 * b.1 → m = 1 :=
by
  intros a b h
  dsimp at *
  sorry

end NUMINAMATH_GPT_parallel_vectors_m_l1535_153592


namespace NUMINAMATH_GPT_maximum_elephants_l1535_153523

theorem maximum_elephants (e_1 e_2 : ℕ) :
  (∃ e_1 e_2 : ℕ, 28 * e_1 + 37 * e_2 = 1036 ∧ (∀ k, 28 * e_1 + 37 * e_2 = k → k ≤ 1036 )) → 
  28 * e_1 + 37 * e_2 = 1036 :=
sorry

end NUMINAMATH_GPT_maximum_elephants_l1535_153523


namespace NUMINAMATH_GPT_banana_to_pear_equiv_l1535_153539

/-
Given conditions:
1. 5 bananas cost as much as 3 apples.
2. 9 apples cost the same as 6 pears.
Prove the equivalence between 30 bananas and 12 pears.

We will define the equivalences as constants and prove the cost equivalence.
-/

variable (cost_banana cost_apple cost_pear : ℤ)

noncomputable def cost_equiv : Prop :=
  (5 * cost_banana = 3 * cost_apple) ∧ 
  (9 * cost_apple = 6 * cost_pear) →
  (30 * cost_banana = 12 * cost_pear)

theorem banana_to_pear_equiv :
  cost_equiv cost_banana cost_apple cost_pear :=
by
  sorry

end NUMINAMATH_GPT_banana_to_pear_equiv_l1535_153539


namespace NUMINAMATH_GPT_jon_coffee_spending_in_april_l1535_153599

def cost_per_coffee : ℕ := 2
def coffees_per_day : ℕ := 2
def days_in_april : ℕ := 30

theorem jon_coffee_spending_in_april :
  (coffees_per_day * cost_per_coffee) * days_in_april = 120 :=
by
  sorry

end NUMINAMATH_GPT_jon_coffee_spending_in_april_l1535_153599


namespace NUMINAMATH_GPT_min_product_of_prime_triplet_l1535_153547

theorem min_product_of_prime_triplet
  (x y z : ℕ) (hx : Prime x) (hy : Prime y) (hz : Prime z)
  (hx_odd : x % 2 = 1) (hy_odd : y % 2 = 1) (hz_odd : z % 2 = 1)
  (h1 : x ∣ (y^5 + 1)) (h2 : y ∣ (z^5 + 1)) (h3 : z ∣ (x^5 + 1)) :
  (x * y * z) = 2013 := by
  sorry

end NUMINAMATH_GPT_min_product_of_prime_triplet_l1535_153547


namespace NUMINAMATH_GPT_inequality_system_solution_l1535_153551

theorem inequality_system_solution (x : ℝ) : 
  (6 * x + 1 ≤ 4 * (x - 1)) ∧ (1 - x / 4 > (x + 5) / 2) → x ≤ -5/2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_system_solution_l1535_153551


namespace NUMINAMATH_GPT_triangle_side_lengths_l1535_153533

theorem triangle_side_lengths (a b c : ℝ) 
  (h1 : a + b + c = 18) 
  (h2 : a + b = 2 * c) 
  (h3 : b = 2 * a):
  a = 4 ∧ b = 8 ∧ c = 6 := 
by
  sorry

end NUMINAMATH_GPT_triangle_side_lengths_l1535_153533


namespace NUMINAMATH_GPT_size_of_can_of_concentrate_l1535_153558

theorem size_of_can_of_concentrate
  (can_to_water_ratio : ℕ := 1 + 3)
  (servings_needed : ℕ := 320)
  (serving_size : ℕ := 6)
  (total_volume : ℕ := servings_needed * serving_size) :
  ∃ C : ℕ, C = total_volume / can_to_water_ratio :=
by
  sorry

end NUMINAMATH_GPT_size_of_can_of_concentrate_l1535_153558


namespace NUMINAMATH_GPT_probability_heads_is_one_eighth_l1535_153542

-- Define the probability problem
def probability_heads_penny_nickel_dime (total_coins: ℕ) (successful_events: ℕ) : ℚ :=
  successful_events / total_coins

-- Define the constants
def total_outcomes : ℕ := 2^5  -- Number of possible outcomes with 5 coins
def successful_outcomes : ℕ := 4  -- Number of successful outcomes where penny, nickel, and dime are heads

-- State the theorem to be proven
theorem probability_heads_is_one_eighth : 
  probability_heads_penny_nickel_dime total_outcomes successful_outcomes = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_probability_heads_is_one_eighth_l1535_153542


namespace NUMINAMATH_GPT_interest_rate_per_annum_l1535_153528

theorem interest_rate_per_annum (P T : ℝ) (r : ℝ) 
  (h1 : P = 15000) 
  (h2 : T = 2)
  (h3 : P * (1 + r)^T - P - (P * r * T) = 150) : 
  r = 0.1 :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_per_annum_l1535_153528


namespace NUMINAMATH_GPT_calc_result_neg2xy2_pow3_l1535_153566

theorem calc_result_neg2xy2_pow3 (x y : ℝ) : 
  (-2 * x * y^2)^3 = -8 * x^3 * y^6 := 
by 
  sorry

end NUMINAMATH_GPT_calc_result_neg2xy2_pow3_l1535_153566


namespace NUMINAMATH_GPT_medicine_dosage_l1535_153581

theorem medicine_dosage (weight_kg dose_per_kg parts : ℕ) (h_weight : weight_kg = 30) (h_dose_per_kg : dose_per_kg = 5) (h_parts : parts = 3) :
  ((weight_kg * dose_per_kg) / parts) = 50 :=
by sorry

end NUMINAMATH_GPT_medicine_dosage_l1535_153581


namespace NUMINAMATH_GPT_wheel_circumferences_satisfy_conditions_l1535_153575

def C_f : ℝ := 24
def C_r : ℝ := 18

theorem wheel_circumferences_satisfy_conditions:
  360 / C_f = 360 / C_r + 4 ∧ 360 / (C_f - 3) = 360 / (C_r - 3) + 6 :=
by 
  have h1: 360 / C_f = 360 / C_r + 4 := sorry
  have h2: 360 / (C_f - 3) = 360 / (C_r - 3) + 6 := sorry
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_wheel_circumferences_satisfy_conditions_l1535_153575


namespace NUMINAMATH_GPT_maximize_S_n_l1535_153587

variable {a : ℕ → ℝ} -- Sequence term definition
variable {S : ℕ → ℝ} -- Sum of first n terms

-- Definitions based on conditions
noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := 
  ∀ n, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  n * a 1 + (n * (n - 1) / 2) * ((a 2) - (a 1))

axiom a1_positive (a1 : ℝ) : 0 < a1 -- given a1 > 0
axiom S3_eq_S16 (a1 d : ℝ) : sum_of_first_n_terms a 3 = sum_of_first_n_terms a 16

-- Problem Statement
theorem maximize_S_n (a : ℕ → ℝ) (d : ℝ) : is_arithmetic_sequence a d →
  a 1 > 0 →
  sum_of_first_n_terms a 3 = sum_of_first_n_terms a 16 →
  (∀ n, sum_of_first_n_terms a n = sum_of_first_n_terms a 9 ∨ sum_of_first_n_terms a n = sum_of_first_n_terms a 10) :=
by
  sorry

end NUMINAMATH_GPT_maximize_S_n_l1535_153587


namespace NUMINAMATH_GPT_line_equation_l1535_153550

-- Define the conditions as given in the problem
def passes_through (P : ℝ × ℝ) (line : ℝ × ℝ) : Prop :=
  line.fst * P.fst + line.snd * P.snd + 1 = 0

def equal_intercepts (line : ℝ × ℝ) : Prop :=
  line.fst = line.snd

theorem line_equation (P : ℝ × ℝ) (hP : P = (-2, -1)) :
  (∃ (k : ℝ), passes_through P (1, -2 * k)) ∨ (∃ (m : ℝ), passes_through P (1, m) ∧ m = - 1) :=
sorry

end NUMINAMATH_GPT_line_equation_l1535_153550


namespace NUMINAMATH_GPT_find_x_if_delta_phi_x_eq_3_l1535_153549

def delta (x : ℚ) : ℚ := 2 * x + 5
def phi (x : ℚ) : ℚ := 9 * x + 6

theorem find_x_if_delta_phi_x_eq_3 :
  ∃ (x : ℚ), delta (phi x) = 3 ∧ x = -7/9 := by
sorry

end NUMINAMATH_GPT_find_x_if_delta_phi_x_eq_3_l1535_153549


namespace NUMINAMATH_GPT_investment_ratio_same_period_l1535_153517

-- Define the profits of A and B
def profit_A : ℕ := 60000
def profit_B : ℕ := 6000

-- Define their investment ratio given the same time period
theorem investment_ratio_same_period : profit_A / profit_B = 10 :=
by
  -- Proof skipped 
  sorry

end NUMINAMATH_GPT_investment_ratio_same_period_l1535_153517


namespace NUMINAMATH_GPT_find_base_l1535_153503

theorem find_base (b : ℕ) (h : (3 * b + 5) ^ 2 = b ^ 3 + 3 * b ^ 2 + 2 * b + 5) : b = 7 :=
sorry

end NUMINAMATH_GPT_find_base_l1535_153503


namespace NUMINAMATH_GPT_farmer_profit_l1535_153552

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

end NUMINAMATH_GPT_farmer_profit_l1535_153552


namespace NUMINAMATH_GPT_highest_wave_height_l1535_153510

-- Definitions of surfboard length and shortest wave conditions
def surfboard_length : ℕ := 7
def shortest_wave_height (H : ℕ) : ℕ := H + 4

-- Lean statement to be proved
theorem highest_wave_height (H : ℕ) (condition1 : H + 4 = surfboard_length + 3) : 
  4 * H + 2 = 26 :=
sorry

end NUMINAMATH_GPT_highest_wave_height_l1535_153510


namespace NUMINAMATH_GPT_bus_speed_including_stoppages_l1535_153505

theorem bus_speed_including_stoppages 
  (speed_without_stoppages : ℕ) 
  (stoppage_time_per_hour : ℕ) 
  (correct_speed_including_stoppages : ℕ) :
  speed_without_stoppages = 54 →
  stoppage_time_per_hour = 10 →
  correct_speed_including_stoppages = 45 :=
by
sorry

end NUMINAMATH_GPT_bus_speed_including_stoppages_l1535_153505


namespace NUMINAMATH_GPT_seashells_given_joan_to_mike_l1535_153563

-- Declaring the context for the problem: Joan's seashells
def initial_seashells := 79
def remaining_seashells := 16

-- Proving how many seashells Joan gave to Mike
theorem seashells_given_joan_to_mike : (initial_seashells - remaining_seashells) = 63 :=
by
  -- This proof needs to be completed
  sorry

end NUMINAMATH_GPT_seashells_given_joan_to_mike_l1535_153563


namespace NUMINAMATH_GPT_brian_commission_rate_l1535_153502

noncomputable def commission_rate (sale1 sale2 sale3 commission : ℝ) : ℝ :=
  (commission / (sale1 + sale2 + sale3)) * 100

theorem brian_commission_rate :
  commission_rate 157000 499000 125000 15620 = 2 :=
by
  unfold commission_rate
  sorry

end NUMINAMATH_GPT_brian_commission_rate_l1535_153502


namespace NUMINAMATH_GPT_tangent_line_parallel_l1535_153530

theorem tangent_line_parallel (x y : ℝ) (h_parab : y = 2 * x^2) (h_parallel : ∃ (m b : ℝ), 4 * x - y + b = 0) : 
    (∃ b, 4 * x - y - b = 0) := 
by
  sorry

end NUMINAMATH_GPT_tangent_line_parallel_l1535_153530


namespace NUMINAMATH_GPT_fractions_order_l1535_153531

theorem fractions_order:
  (20 / 15) < (25 / 18) ∧ (25 / 18) < (23 / 16) ∧ (23 / 16) < (21 / 14) :=
by
  sorry

end NUMINAMATH_GPT_fractions_order_l1535_153531


namespace NUMINAMATH_GPT_truck_travel_distance_l1535_153579

def original_distance : ℝ := 300
def original_gas : ℝ := 10
def increased_efficiency_percent : ℝ := 1.10
def new_gas : ℝ := 15

theorem truck_travel_distance :
  let original_efficiency := original_distance / original_gas;
  let new_efficiency := original_efficiency * increased_efficiency_percent;
  let distance := new_gas * new_efficiency;
  distance = 495 :=
by
  sorry

end NUMINAMATH_GPT_truck_travel_distance_l1535_153579


namespace NUMINAMATH_GPT_area_of_region_B_l1535_153518

noncomputable def region_B_area : ℝ :=
  let square_area := 900
  let excluded_area := 28.125 * Real.pi
  square_area - excluded_area

theorem area_of_region_B : region_B_area = 900 - 28.125 * Real.pi :=
by {
  sorry
}

end NUMINAMATH_GPT_area_of_region_B_l1535_153518


namespace NUMINAMATH_GPT_car_speed_in_first_hour_l1535_153543

theorem car_speed_in_first_hour (x : ℝ) 
  (second_hour_speed : ℝ := 40)
  (average_speed : ℝ := 60)
  (h : (x + second_hour_speed) / 2 = average_speed) :
  x = 80 := 
by
  -- Additional steps needed to solve this theorem
  sorry

end NUMINAMATH_GPT_car_speed_in_first_hour_l1535_153543


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_l1535_153569

theorem arithmetic_geometric_sequence (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ)
  (h1 : d ≠ 0)
  (h2 : ∀ n, a (n + 1) = a n + d)
  (h3 : a 3 * 2 = a 1 + 2 * d)
  (h4 : a 4 = a 1 + 3 * d)
  (h5 : a 8 = a 1 + 7 * d)
  (h_geo : (a 1 + 3 * d) ^ 2 = (a 1 + 2 * d) * (a 1 + 7 * d))
  (h_sum : S 4 = (a 1 * 4) + (d * (4 * 3 / 2))) :
  a 1 * d < 0 ∧ d * S 4 < 0 :=
by sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_l1535_153569


namespace NUMINAMATH_GPT_match_processes_count_l1535_153513

-- Define the sets and the number of interleavings
def team_size : ℕ := 4 -- Each team has 4 players

-- Define the problem statement
theorem match_processes_count :
  (Nat.choose (2 * team_size) team_size) = 70 := by
  -- This is where the proof would go, but we'll use sorry as specified
  sorry

end NUMINAMATH_GPT_match_processes_count_l1535_153513


namespace NUMINAMATH_GPT_coordinates_of_P_respect_to_symmetric_y_axis_l1535_153571

-- Definition of points in the Cartesian coordinate system
structure Point where
  x : ℝ
  y : ℝ

def symmetric_x_axis (p : Point) : Point :=
  { p with y := -p.y }

def symmetric_y_axis (p : Point) : Point :=
  { p with x := -p.x }

-- The given condition
def P_with_respect_to_symmetric_x_axis := Point.mk (-1) 2

-- The problem statement
theorem coordinates_of_P_respect_to_symmetric_y_axis :
    symmetric_y_axis (symmetric_x_axis P_with_respect_to_symmetric_x_axis) = Point.mk 1 (-2) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_P_respect_to_symmetric_y_axis_l1535_153571


namespace NUMINAMATH_GPT_housewife_saving_l1535_153520

theorem housewife_saving :
  let total_money := 450
  let groceries_fraction := 3 / 5
  let household_items_fraction := 1 / 6
  let personal_care_items_fraction := 1 / 10
  let groceries_expense := groceries_fraction * total_money
  let household_items_expense := household_items_fraction * total_money
  let personal_care_items_expense := personal_care_items_fraction * total_money
  let total_expense := groceries_expense + household_items_expense + personal_care_items_expense
  total_money - total_expense = 60 :=
by
  sorry

end NUMINAMATH_GPT_housewife_saving_l1535_153520


namespace NUMINAMATH_GPT_wealth_ratio_l1535_153570

theorem wealth_ratio (W P : ℝ) (hW_pos : 0 < W) (hP_pos : 0 < P) :
  let wX := 0.54 * W / (0.40 * P)
  let wY := 0.30 * W / (0.20 * P)
  wX / wY = 0.9 := 
by
  sorry

end NUMINAMATH_GPT_wealth_ratio_l1535_153570


namespace NUMINAMATH_GPT_max_marks_400_l1535_153568

theorem max_marks_400 {M : ℝ} (h : 0.45 * M = 150 + 30) : M = 400 := 
by
  sorry

end NUMINAMATH_GPT_max_marks_400_l1535_153568


namespace NUMINAMATH_GPT_find_line_equation_l1535_153507

-- Define the point (2, -1) which the line passes through
def point : ℝ × ℝ := (2, -1)

-- Define the line perpendicular to 2x - 3y = 1
def perpendicular_line (x y : ℝ) : Prop := 2 * x - 3 * y - 1 = 0

-- The equation of the line we are supposed to find
def equation_of_line (x y : ℝ) : Prop := 3 * x + 2 * y - 4 = 0

-- Proof problem: prove the equation satisfies given the conditions
theorem find_line_equation :
  (equation_of_line point.1 point.2) ∧ 
  (∃ (a b c : ℝ), ∀ (x y : ℝ), perpendicular_line x y → equation_of_line x y) := sorry

end NUMINAMATH_GPT_find_line_equation_l1535_153507


namespace NUMINAMATH_GPT_find_angle_B_find_area_of_ABC_l1535_153598

noncomputable def angle_B (a b c : ℝ) (C : ℝ) (h1 : 2 * b * Real.cos C = 2 * a + c) : ℝ := 
  if b * Real.cos C = -a then Real.pi - 2 * Real.arctan (a / c)
  else 2 * Real.pi / 3

theorem find_angle_B (a b c : ℝ) (C : ℝ) (h1 : 2 * b * Real.cos C = 2 * a + c) :
  angle_B a b c C h1 = 2 * Real.pi / 3 := 
sorry

noncomputable def area_of_ABC (a b c : ℝ) (C B : ℝ) (d : ℝ) (position : ℕ) (h1 : 2 * b * Real.cos C = 2 * a + c) (h2 : b = 2 * Real.sqrt 3) (h3 : d = 1) : ℝ :=
  if position = 1 then /- calculation for BD bisector case -/ (a * c / 2) * Real.sin (2 * Real.pi / 3)
  else /- calculation for midpoint case -/ (a * c / 2) * Real.sin (2 * Real.pi / 3)

theorem find_area_of_ABC (a b c : ℝ) (C B : ℝ) (d : ℝ) (position : ℕ) (h1 : 2 * b * Real.cos C = 2 * a + c) (h2 : b = 2 * Real.sqrt 3) (h3 : d = 1) (hB : angle_B a b c C h1 = 2 * Real.pi / 3) :
  area_of_ABC a b c C (2 * Real.pi / 3) d position h1 h2 h3 = Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_find_angle_B_find_area_of_ABC_l1535_153598


namespace NUMINAMATH_GPT_tan_alpha_solution_l1535_153548

theorem tan_alpha_solution (α : Real) (h : Real.tan α = 1 / 2) : 
  (2 * Real.sin α + Real.cos α) / (4 * Real.sin α - Real.cos α) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_tan_alpha_solution_l1535_153548


namespace NUMINAMATH_GPT_square_of_neg_3b_l1535_153582

theorem square_of_neg_3b (b : ℝ) : (-3 * b)^2 = 9 * b^2 :=
by sorry

end NUMINAMATH_GPT_square_of_neg_3b_l1535_153582


namespace NUMINAMATH_GPT_ratio_sum_div_c_l1535_153516

theorem ratio_sum_div_c (a b c : ℚ) (h : a / 3 = b / 4 ∧ b / 4 = c / 5) : (a + b + c) / c = 12 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_sum_div_c_l1535_153516


namespace NUMINAMATH_GPT_find_t_l1535_153590

noncomputable def g (x : ℝ) (p q s t : ℝ) : ℝ :=
  x^4 + p*x^3 + q*x^2 + s*x + t

theorem find_t {p q s t : ℝ}
  (h1 : ∀ r : ℝ, g r p q s t = 0 → r < 0 ∧ Int.mod (round r) 2 = 1)
  (h2 : p + q + s + t = 2047) :
  t = 5715 :=
sorry

end NUMINAMATH_GPT_find_t_l1535_153590


namespace NUMINAMATH_GPT_perimeter_difference_l1535_153512

-- Define the height of the screen
def height_of_screen : ℕ := 100

-- Define the side length of the square paper
def side_of_square_paper : ℕ := 20

-- Define the perimeter of the square paper
def perimeter_of_paper : ℕ := 4 * side_of_square_paper

-- Prove the difference between the height of the screen and the perimeter of the paper
theorem perimeter_difference : height_of_screen - perimeter_of_paper = 20 := by
  -- Sorry is used here to skip the actual proof
  sorry

end NUMINAMATH_GPT_perimeter_difference_l1535_153512


namespace NUMINAMATH_GPT_bunnies_out_of_burrow_l1535_153509

theorem bunnies_out_of_burrow:
  (3 * 60 * 10 * 20) = 36000 :=
by 
  sorry

end NUMINAMATH_GPT_bunnies_out_of_burrow_l1535_153509


namespace NUMINAMATH_GPT_infinite_geometric_series_sum_l1535_153538

theorem infinite_geometric_series_sum :
  let a := (4 : ℚ) / 3
  let r := -(9 : ℚ) / 16
  (a / (1 - r)) = (64 : ℚ) / 75 :=
by
  sorry

end NUMINAMATH_GPT_infinite_geometric_series_sum_l1535_153538


namespace NUMINAMATH_GPT_circle_radius_squared_l1535_153595

open Real

/-- Prove that the square of the radius of a circle is 200 given the conditions provided. -/

theorem circle_radius_squared {r : ℝ}
  (AB CD : ℝ)
  (BP : ℝ) 
  (APD : ℝ) 
  (hAB : AB = 12)
  (hCD : CD = 9)
  (hBP : BP = 10)
  (hAPD : APD = 45) :
  r^2 = 200 := 
sorry

end NUMINAMATH_GPT_circle_radius_squared_l1535_153595


namespace NUMINAMATH_GPT_option_A_is_translation_l1535_153536

-- Define what constitutes a translation transformation
def is_translation (description : String) : Prop :=
  description = "Pulling open a drawer"

-- Define each option
def option_A : String := "Pulling open a drawer"
def option_B : String := "Viewing text through a magnifying glass"
def option_C : String := "The movement of the minute hand on a clock"
def option_D : String := "You and the image in a plane mirror"

-- The main theorem stating that option A is the translation transformation
theorem option_A_is_translation : is_translation option_A :=
by
  -- skip the proof, adding sorry
  sorry

end NUMINAMATH_GPT_option_A_is_translation_l1535_153536


namespace NUMINAMATH_GPT_no_perfect_squares_in_sequence_l1535_153519

theorem no_perfect_squares_in_sequence (x : ℕ → ℤ) (h₀ : x 0 = 1) (h₁ : x 1 = 3)
  (h_rec : ∀ n : ℕ, x (n + 1) = 6 * x n - x (n - 1)) 
  : ∀ n : ℕ, ¬ ∃ k : ℤ, x n = k * k := 
sorry

end NUMINAMATH_GPT_no_perfect_squares_in_sequence_l1535_153519


namespace NUMINAMATH_GPT_cat_mouse_position_after_299_moves_l1535_153589

-- Definitions based on conditions
def cat_position (move : Nat) : Nat :=
  let active_moves := move - (move / 100)
  active_moves % 4

def mouse_position (move : Nat) : Nat :=
  move % 8

-- Main theorem
theorem cat_mouse_position_after_299_moves :
  cat_position 299 = 0 ∧ mouse_position 299 = 3 :=
by
  sorry

end NUMINAMATH_GPT_cat_mouse_position_after_299_moves_l1535_153589


namespace NUMINAMATH_GPT_hexagon_colorings_l1535_153577

-- Definitions based on conditions
def isValidColoring (A B C D E F : ℕ) (colors : Fin 7 → ℕ) : Prop :=
  -- Adjacent vertices must have different colors
  A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ F ∧ F ≠ A ∧
  -- Diagonal vertices must have different colors
  A ≠ D ∧ B ≠ E ∧ C ≠ F

-- Function to count all valid colorings
def countValidColorings : ℕ :=
  let colors := List.range 7
  -- Calculate total number of valid colorings
  7 * 6 * 5 * 4 * 3 * 2

theorem hexagon_colorings : countValidColorings = 5040 := by
  sorry

end NUMINAMATH_GPT_hexagon_colorings_l1535_153577


namespace NUMINAMATH_GPT_present_value_of_machine_l1535_153553

theorem present_value_of_machine (r : ℝ) (t : ℕ) (V : ℝ) (P : ℝ) (h1 : r = 0.10) (h2 : t = 2) (h3 : V = 891) :
  V = P * (1 - r)^t → P = 1100 :=
by
  intro h
  rw [h3, h1, h2] at h
  -- The steps to solve for P are omitted as instructed
  sorry

end NUMINAMATH_GPT_present_value_of_machine_l1535_153553


namespace NUMINAMATH_GPT_hannahs_trip_cost_l1535_153556

noncomputable def calculate_gas_cost (initial_odometer final_odometer : ℕ) (fuel_economy_mpg : ℚ) (cost_per_gallon : ℚ) : ℚ :=
  let distance := final_odometer - initial_odometer
  let fuel_used := distance / fuel_economy_mpg
  fuel_used * cost_per_gallon

theorem hannahs_trip_cost :
  calculate_gas_cost 36102 36131 32 (385 / 100) = 276 / 100 :=
by
  sorry

end NUMINAMATH_GPT_hannahs_trip_cost_l1535_153556


namespace NUMINAMATH_GPT_binary101_to_decimal_l1535_153561

theorem binary101_to_decimal :
  let binary_101 := 1 * 2^2 + 0 * 2^1 + 1 * 2^0
  binary_101 = 5 := 
by
  let binary_101 := 1 * 2^2 + 0 * 2^1 + 1 * 2^0
  show binary_101 = 5
  sorry

end NUMINAMATH_GPT_binary101_to_decimal_l1535_153561


namespace NUMINAMATH_GPT_proof_seq_l1535_153585

open Nat

-- Definition of sequence {a_n}
def seq_a : ℕ → ℕ
| 0 => 1
| n + 1 => 3 * seq_a n

-- Definition of sum S_n of sequence {b_n}
def sum_S : ℕ → ℕ
| 0 => 0
| n + 1 => sum_S n + (2^n)

-- Definition of sequence {b_n}
def seq_b : ℕ → ℕ
| 0 => 1
| n + 1 => 2 * seq_b n

-- Definition of sequence {c_n}
def seq_c (n : ℕ) : ℕ := seq_b n * log 3 (seq_a n) -- Note: log base 3

-- Sum of first n terms of {c_n}
def sum_T : ℕ → ℕ
| 0 => 0
| n + 1 => sum_T n + seq_c n

-- Proof statement
theorem proof_seq (n : ℕ) :
  (seq_a n = 3 ^ n) ∧
  (2 * seq_b n - 1 = sum_S 0 * sum_S n) ∧
  (sum_T n = (n - 2) * 2 ^ (n + 2)) :=
sorry

end NUMINAMATH_GPT_proof_seq_l1535_153585


namespace NUMINAMATH_GPT_Dan_reaches_Cate_in_25_seconds_l1535_153515

theorem Dan_reaches_Cate_in_25_seconds
  (d : ℝ) (v_d : ℝ) (v_c : ℝ)
  (h1 : d = 50)
  (h2 : v_d = 8)
  (h3 : v_c = 6) :
  (d / (v_d - v_c) = 25) :=
by
  sorry

end NUMINAMATH_GPT_Dan_reaches_Cate_in_25_seconds_l1535_153515


namespace NUMINAMATH_GPT_inequality_holds_l1535_153574

theorem inequality_holds (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (a * b + b * c + c * a)^2 ≥ 3 * a * b * c * (a + b + c) :=
sorry

end NUMINAMATH_GPT_inequality_holds_l1535_153574


namespace NUMINAMATH_GPT_madeline_part_time_hours_l1535_153521

theorem madeline_part_time_hours :
  let hours_in_class := 18
  let days_in_week := 7
  let hours_homework_per_day := 4
  let hours_sleeping_per_day := 8
  let leftover_hours := 46
  let hours_per_day := 24
  let total_hours_per_week := hours_per_day * days_in_week
  let total_homework_hours := hours_homework_per_day * days_in_week
  let total_sleeping_hours := hours_sleeping_per_day * days_in_week
  let total_other_activities := hours_in_class + total_homework_hours + total_sleeping_hours
  let available_hours := total_hours_per_week - total_other_activities
  available_hours - leftover_hours = 20 := by
  sorry

end NUMINAMATH_GPT_madeline_part_time_hours_l1535_153521


namespace NUMINAMATH_GPT_reciprocal_of_negative_2023_l1535_153583

theorem reciprocal_of_negative_2023 : (1 / (-2023 : ℤ)) = -(1 / (2023 : ℤ)) := by
  sorry

end NUMINAMATH_GPT_reciprocal_of_negative_2023_l1535_153583


namespace NUMINAMATH_GPT_general_formula_for_a_n_l1535_153501

-- Given conditions
variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Conditions
variable (h1 : ∀ n : ℕ, a n > 0)
variable (h2 : ∀ n : ℕ, 4 * S n = (a n - 1) * (a n + 3))

theorem general_formula_for_a_n :
  ∀ n : ℕ, a n = 2 * n + 1 :=
sorry

end NUMINAMATH_GPT_general_formula_for_a_n_l1535_153501


namespace NUMINAMATH_GPT_line_through_intersection_of_circles_l1535_153567

theorem line_through_intersection_of_circles 
  (x y : ℝ)
  (C1 : x^2 + y^2 = 10)
  (C2 : (x-1)^2 + (y-3)^2 = 20) :
  x + 3 * y = 0 :=
sorry

end NUMINAMATH_GPT_line_through_intersection_of_circles_l1535_153567
