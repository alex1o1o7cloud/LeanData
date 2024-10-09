import Mathlib

namespace each_friend_gave_bella_2_roses_l778_77855

-- Define the given conditions
def total_roses_from_parents : ℕ := 2 * 12
def total_roses_bella_received : ℕ := 44
def number_of_dancer_friends : ℕ := 10

-- Define the mathematical goal
def roses_from_each_friend (total_roses_from_parents total_roses_bella_received number_of_dancer_friends : ℕ) : ℕ :=
  (total_roses_bella_received - total_roses_from_parents) / number_of_dancer_friends

-- Prove that each dancer friend gave Bella 2 roses
theorem each_friend_gave_bella_2_roses :
  roses_from_each_friend total_roses_from_parents total_roses_bella_received number_of_dancer_friends = 2 :=
by
  sorry

end each_friend_gave_bella_2_roses_l778_77855


namespace apollo_total_cost_l778_77801

def hephaestus_first_half_months : ℕ := 6
def hephaestus_first_half_rate : ℕ := 3
def hephaestus_second_half_rate : ℕ := hephaestus_first_half_rate * 2

def athena_rate : ℕ := 5
def athena_months : ℕ := 12

def ares_first_period_months : ℕ := 9
def ares_first_period_rate : ℕ := 4
def ares_second_period_months : ℕ := 3
def ares_second_period_rate : ℕ := 6

def total_cost := hephaestus_first_half_months * hephaestus_first_half_rate
               + hephaestus_first_half_months * hephaestus_second_half_rate
               + athena_months * athena_rate
               + ares_first_period_months * ares_first_period_rate
               + ares_second_period_months * ares_second_period_rate

theorem apollo_total_cost : total_cost = 168 := by
  -- placeholder for the proof
  sorry

end apollo_total_cost_l778_77801


namespace find_triplets_l778_77843

theorem find_triplets (a b c : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) 
  (h4 : a ∣ b + c + 1) (h5 : b ∣ c + a + 1) (h6 : c ∣ a + b + 1) :
  (a, b, c) = (1, 1, 1) ∨ (a, b, c) = (1, 2, 2) ∨ (a, b, c) = (3, 4, 4) ∨ 
  (a, b, c) = (1, 1, 3) ∨ (a, b, c) = (2, 2, 5) :=
sorry

end find_triplets_l778_77843


namespace ordered_pair_exists_l778_77867

theorem ordered_pair_exists (x y : ℤ) (h1 : x + y = (7 - x) + (7 - y)) (h2 : x - y = (x + 1) + (y + 1)) :
  (x, y) = (8, -1) :=
by
  sorry

end ordered_pair_exists_l778_77867


namespace min_eq_floor_sqrt_l778_77809

theorem min_eq_floor_sqrt (n : ℕ) (h : n > 0) : 
  (∀ k : ℕ, k > 0 → (k + n / k) ≥ ⌊(Real.sqrt (4 * n + 1))⌋) := 
sorry

end min_eq_floor_sqrt_l778_77809


namespace anika_sequence_correct_l778_77811

noncomputable def anika_sequence : ℚ :=
  let s0 := 1458
  let s1 := s0 * 3
  let s2 := s1 / 2
  let s3 := s2 * 3
  let s4 := s3 / 2
  let s5 := s4 * 3
  s5

theorem anika_sequence_correct :
  anika_sequence = (3^9 : ℚ) / 2 := by
  sorry

end anika_sequence_correct_l778_77811


namespace volume_of_pond_rect_prism_l778_77805

-- Define the problem as a proposition
theorem volume_of_pond_rect_prism :
  let l := 28
  let w := 10
  let h := 5
  V = l * w * h →
  V = 1400 :=
by
  intros l w h h1
  -- Here, the theorem states the equivalence of the volume given the defined length, width, and height being equal to 1400 cubic meters.
  have : V = 28 * 10 * 5 := by sorry
  exact this

end volume_of_pond_rect_prism_l778_77805


namespace smallest_of_seven_even_numbers_sum_448_l778_77820

theorem smallest_of_seven_even_numbers_sum_448 :
  ∃ n : ℤ, n + (n+2) + (n+4) + (n+6) + (n+8) + (n+10) + (n+12) = 448 ∧ n = 58 := 
by
  sorry

end smallest_of_seven_even_numbers_sum_448_l778_77820


namespace additional_plates_correct_l778_77884

-- Define the conditions
def original_set_1 : Finset Char := {'B', 'F', 'J', 'N', 'T'}
def original_set_2 : Finset Char := {'E', 'U'}
def original_set_3 : Finset Char := {'G', 'K', 'R', 'Z'}

-- Define the sizes of the original sets
def size_set_1 := (original_set_1.card : Nat) -- 5
def size_set_2 := (original_set_2.card : Nat) -- 2
def size_set_3 := (original_set_3.card : Nat) -- 4

-- Sizes after adding new letters
def new_size_set_1 := size_set_1 + 1 -- 6
def new_size_set_2 := size_set_2 + 1 -- 3
def new_size_set_3 := size_set_3 + 1 -- 5

-- Calculate the original and new number of plates
def original_plates : Nat := size_set_1 * size_set_2 * size_set_3 -- 5 * 2 * 4 = 40
def new_plates : Nat := new_size_set_1 * new_size_set_2 * new_size_set_3 -- 6 * 3 * 5 = 90

-- Calculate the additional plates
def additional_plates : Nat := new_plates - original_plates -- 90 - 40 = 50

-- The proof statement
theorem additional_plates_correct : additional_plates = 50 :=
by
  -- Proof can be filled in here
  sorry

end additional_plates_correct_l778_77884


namespace garage_sale_total_l778_77869

theorem garage_sale_total (treadmill chest_of_drawers television total_sales : ℝ)
  (h1 : treadmill = 100) 
  (h2 : chest_of_drawers = treadmill / 2) 
  (h3 : television = treadmill * 3) 
  (partial_sales : ℝ) 
  (h4 : partial_sales = treadmill + chest_of_drawers + television) 
  (h5 : partial_sales = total_sales * 0.75) : 
  total_sales = 600 := 
by
  sorry

end garage_sale_total_l778_77869


namespace train_speed_l778_77822

theorem train_speed (L : ℝ) (T : ℝ) (L_pos : 0 < L) (T_pos : 0 < T) (L_eq : L = 150) (T_eq : T = 3) : L / T = 50 := by
  sorry

end train_speed_l778_77822


namespace solution_set_of_inequality_l778_77866

theorem solution_set_of_inequality :
  {x : ℝ | x^2 - 5 * x + 6 ≤ 0} = {x : ℝ | 2 ≤ x ∧ x ≤ 3} :=
sorry

end solution_set_of_inequality_l778_77866


namespace melanie_average_speed_l778_77898

theorem melanie_average_speed
  (bike_distance run_distance total_time : ℝ)
  (h_bike : bike_distance = 15)
  (h_run : run_distance = 5)
  (h_time : total_time = 4) :
  (bike_distance + run_distance) / total_time = 5 :=
by
  sorry

end melanie_average_speed_l778_77898


namespace prime_only_one_solution_l778_77825

theorem prime_only_one_solution (p : ℕ) (hp : Nat.Prime p) : 
  (∃ k : ℕ, 2 * p^4 - p^2 + 16 = k^2) → p = 3 := 
by 
  sorry

end prime_only_one_solution_l778_77825


namespace bamboo_tube_rice_capacity_l778_77848

theorem bamboo_tube_rice_capacity :
  ∃ (a d : ℝ), 3 * a + 3 * d * (1 + 2) = 4.5 ∧ 
               4 * (a + 5 * d) + 4 * d * (6 + 7 + 8) = 3.8 ∧ 
               (a + 3 * d) + (a + 4 * d) = 2.5 :=
by
  sorry

end bamboo_tube_rice_capacity_l778_77848


namespace mary_sailboat_canvas_l778_77816

def rectangular_sail_area (length width : ℕ) : ℕ :=
  length * width

def triangular_sail_area (base height : ℕ) : ℕ :=
  (base * height) / 2

def total_canvas_area (length₁ width₁ base₁ height₁ base₂ height₂ : ℕ) : ℕ :=
  rectangular_sail_area length₁ width₁ +
  triangular_sail_area base₁ height₁ +
  triangular_sail_area base₂ height₂

theorem mary_sailboat_canvas :
  total_canvas_area 5 8 3 4 4 6 = 58 :=
by
  -- Begin proof (proof steps omitted, we just need the structure here)
  sorry -- end proof

end mary_sailboat_canvas_l778_77816


namespace sin_sq_sub_sin_double_l778_77878

open Real

theorem sin_sq_sub_sin_double (alpha : ℝ) (h : tan alpha = 1 / 2) : sin alpha ^ 2 - sin (2 * alpha) = -3 / 5 := 
by 
  sorry

end sin_sq_sub_sin_double_l778_77878


namespace sum_13_gt_0_l778_77813

noncomputable def a_n : ℕ → ℝ := sorry
noncomputable def S_n : ℕ → ℝ := sorry

axiom a7_gt_0 : 0 < a_n 7
axiom a8_lt_0 : a_n 8 < 0

theorem sum_13_gt_0 : S_n 13 > 0 :=
sorry

end sum_13_gt_0_l778_77813


namespace ab_le_one_l778_77881

theorem ab_le_one {a b : ℝ} (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : a + b = 2) : ab ≤ 1 :=
by
  sorry

end ab_le_one_l778_77881


namespace complex_number_solution_l778_77844

open Complex

theorem complex_number_solution (z : ℂ) (h : (1 + I) * z = 2 * I) : z = 1 + I :=
sorry

end complex_number_solution_l778_77844


namespace rectangle_area_l778_77883

variable (a b c : ℝ)

theorem rectangle_area (h : a^2 + b^2 = c^2) : a * b = area :=
by sorry

end rectangle_area_l778_77883


namespace exists_k_for_binary_operation_l778_77890

noncomputable def binary_operation (a b : ℤ) : ℤ := sorry

theorem exists_k_for_binary_operation :
  (∀ (a b c : ℤ), binary_operation a (b + c) = 
      binary_operation b a + binary_operation c a) →
  ∃ (k : ℤ), ∀ (a b : ℤ), binary_operation a b = k * a * b :=
by
  sorry

end exists_k_for_binary_operation_l778_77890


namespace jason_total_spent_l778_77894

theorem jason_total_spent (h_shorts : ℝ) (h_jacket : ℝ) (h1 : h_shorts = 14.28) (h2 : h_jacket = 4.74) : h_shorts + h_jacket = 19.02 :=
by
  rw [h1, h2]
  norm_num

end jason_total_spent_l778_77894


namespace find_y_given_area_l778_77847

-- Define the problem parameters and conditions
namespace RectangleArea

variables {y : ℝ} (y_pos : y > 0)

-- Define the vertices, they can be expressed but are not required in the statement
def vertices := [(-2, y), (8, y), (-2, 3), (8, 3)]

-- Define the area condition
def area_condition := 10 * (y - 3) = 90

-- Lean statement proving y = 12 given the conditions
theorem find_y_given_area (y_pos : y > 0) (h : 10 * (y - 3) = 90) : y = 12 :=
by
  sorry

end RectangleArea

end find_y_given_area_l778_77847


namespace theresa_more_than_thrice_julia_l778_77845

-- Define the problem parameters
variable (tory julia theresa : ℕ)

def tory_videogames : ℕ := 6
def theresa_videogames : ℕ := 11

-- Define the relationships between the numbers of video games
def julia_relationship := julia = tory / 3
def theresa_compared_to_julia := theresa = theresa_videogames
def tory_value := tory = tory_videogames

theorem theresa_more_than_thrice_julia (h1 : julia_relationship tory julia) 
                                       (h2 : tory_value tory)
                                       (h3 : theresa_compared_to_julia theresa) :
  theresa - 3 * julia = 5 :=
by 
  -- Here comes the proof (not required for the task)
  sorry

end theresa_more_than_thrice_julia_l778_77845


namespace min_value_am_gm_l778_77865

theorem min_value_am_gm (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a / b) + (b / c) + (c / d) + (d / a) ≥ 4 := 
sorry

end min_value_am_gm_l778_77865


namespace base_salary_l778_77899

theorem base_salary {B : ℝ} {C : ℝ} :
  (B + 200 * C = 2000) → 
  (B + 200 * 15 = 4000) → 
  B = 1000 :=
by
  sorry

end base_salary_l778_77899


namespace quadrilateral_has_four_sides_and_angles_l778_77838

-- Define the conditions based on the characteristics of a quadrilateral
def quadrilateral (sides angles : Nat) : Prop :=
  sides = 4 ∧ angles = 4

-- Statement: Verify the property of a quadrilateral
theorem quadrilateral_has_four_sides_and_angles (sides angles : Nat) (h : quadrilateral sides angles) : sides = 4 ∧ angles = 4 :=
by
  -- We provide a proof by the characteristics of a quadrilateral
  sorry

end quadrilateral_has_four_sides_and_angles_l778_77838


namespace intersection_complement_M_N_l778_77800

def M := { x : ℝ | x ≤ 1 / 2 }
def N := { x : ℝ | x^2 ≤ 1 }
def complement_M := { x : ℝ | x > 1 / 2 }

theorem intersection_complement_M_N :
  (complement_M ∩ N = { x : ℝ | 1 / 2 < x ∧ x ≤ 1 }) :=
by
  sorry

end intersection_complement_M_N_l778_77800


namespace dried_mushrooms_weight_l778_77852

theorem dried_mushrooms_weight (fresh_weight : ℝ) (water_content_fresh : ℝ) (water_content_dried : ℝ) :
  fresh_weight = 22 →
  water_content_fresh = 0.90 →
  water_content_dried = 0.12 →
  ∃ x : ℝ, x = 2.5 :=
by
  intros h1 h2 h3
  have hw_fresh : ℝ := fresh_weight * water_content_fresh
  have dry_material_fresh : ℝ := fresh_weight - hw_fresh
  have dry_material_dried : ℝ := 1.0 - water_content_dried
  have hw_dried := dry_material_fresh / dry_material_dried
  use hw_dried
  sorry

end dried_mushrooms_weight_l778_77852


namespace smallest_n_l778_77806

/-- The smallest value of n > 20 that satisfies
    n ≡ 4 [MOD 6]
    n ≡ 3 [MOD 7]
    n ≡ 5 [MOD 8] is 220. -/
theorem smallest_n (n : ℕ) : 
  (n > 20) ∧ 
  (n % 6 = 4) ∧ 
  (n % 7 = 3) ∧ 
  (n % 8 = 5) ↔ (n = 220) :=
by 
  sorry

end smallest_n_l778_77806


namespace sum_of_terms_in_sequence_is_215_l778_77818

theorem sum_of_terms_in_sequence_is_215 (a d : ℕ) (h1: Nat.Prime a) (h2: Nat.Prime d)
  (hAP : a + 50 = a + 50)
  (hGP : (a + d) * (a + 50) = (a + 2 * d) ^ 2) :
  (a + (a + d) + (a + 2 * d) + (a + 50)) = 215 := sorry

end sum_of_terms_in_sequence_is_215_l778_77818


namespace problem_statement_l778_77830

noncomputable def x : ℝ := (3 + Real.sqrt 8) ^ 30
noncomputable def n : ℤ := Int.floor x
noncomputable def f : ℝ := x - n

theorem problem_statement : x * (1 - f) = 1 := 
by 
  sorry

end problem_statement_l778_77830


namespace multiply_exponents_l778_77891

variable (a : ℝ)

theorem multiply_exponents :
  a * a^2 * (-a)^3 = -a^6 := 
sorry

end multiply_exponents_l778_77891


namespace colored_pictures_count_l778_77875

def initial_pictures_count : ℕ := 44 + 44
def pictures_left : ℕ := 68

theorem colored_pictures_count : initial_pictures_count - pictures_left = 20 := by
  -- Definitions and proof will go here
  sorry

end colored_pictures_count_l778_77875


namespace problem_I_problem_II_l778_77850

-- Definition of the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Problem (I): Prove solution set
theorem problem_I (x : ℝ) : f (x - 1) + f (x + 3) ≥ 6 ↔ (x ≤ -3 ∨ x ≥ 3) := by
  sorry

-- Problem (II): Prove inequality given conditions
theorem problem_II (a b : ℝ) (ha: |a| < 1) (hb: |b| < 1) (hano: a ≠ 0) : 
  f (a * b) > |a| * f (b / a) := by
  sorry

end problem_I_problem_II_l778_77850


namespace distance_from_pole_to_line_l778_77821

/-- Definition of the line in polar coordinates -/
def line_polar (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 2

/-- Definition of the pole in Cartesian coordinates -/
def pole_cartesian : ℝ × ℝ := (0, 0)

/-- Convert the line from polar to Cartesian -/
def line_cartesian (x y : ℝ) : Prop := x = 2

/-- The distance function between a point and a line in Cartesian coordinates -/
def distance_to_line (p : ℝ × ℝ) : ℝ := abs (p.1 - 2)

/-- Prove that the distance from the pole to the line is 2 -/
theorem distance_from_pole_to_line : distance_to_line pole_cartesian = 2 := by
  sorry

end distance_from_pole_to_line_l778_77821


namespace overlapping_area_fraction_l778_77882

variable (Y X : ℝ)
variable (hY : 0 < Y)
variable (hX : X = (1 / 8) * (2 * Y - X))

theorem overlapping_area_fraction : X = (2 / 9) * Y :=
by
  -- We define the conditions and relationships stated in the problem
  -- Prove the theorem accordingly
  sorry

end overlapping_area_fraction_l778_77882


namespace factor_expression_l778_77846

theorem factor_expression (x : ℝ) : 
  75 * x^11 + 225 * x^22 = 75 * x^11 * (1 + 3 * x^11) :=
by sorry

end factor_expression_l778_77846


namespace part1_part2_l778_77859

-- Definitions and conditions
def prop_p (a : ℝ) : Prop := 
  let Δ := -4 * a^2 + 4 * a + 24 
  Δ ≥ 0

def neg_prop_p (a : ℝ) : Prop := ¬ prop_p a

def prop_q (m a : ℝ) : Prop := 
  (m - 1 ≤ a ∧ a ≤ m + 3)

-- Part 1 theorem statement
theorem part1 (a : ℝ) : neg_prop_p a → (a < -2 ∨ a > 3) :=
by sorry

-- Part 2 theorem statement
theorem part2 (m : ℝ) : 
  (∀ a : ℝ, prop_q m a → prop_p a) ∧ (∃ a : ℝ, prop_p a ∧ ¬ prop_q m a) → (-1 ≤ m ∧ m < 0) :=
by sorry

end part1_part2_l778_77859


namespace range_of_k_l778_77849

theorem range_of_k (k : ℝ) : (2 > 0) ∧ (k > 0) ∧ (k < 2) ↔ (0 < k ∧ k < 2) :=
by
  sorry

end range_of_k_l778_77849


namespace fraction_spent_l778_77823

theorem fraction_spent (borrowed_from_brother borrowed_from_father borrowed_from_mother gift_from_granny savings remaining amount_spent : ℕ)
  (h_borrowed_from_brother : borrowed_from_brother = 20)
  (h_borrowed_from_father : borrowed_from_father = 40)
  (h_borrowed_from_mother : borrowed_from_mother = 30)
  (h_gift_from_granny : gift_from_granny = 70)
  (h_savings : savings = 100)
  (h_remaining : remaining = 65)
  (h_amount_spent : amount_spent = borrowed_from_brother + borrowed_from_father + borrowed_from_mother + gift_from_granny + savings - remaining) :
  (amount_spent : ℚ) / (borrowed_from_brother + borrowed_from_father + borrowed_from_mother + gift_from_granny + savings) = 3 / 4 :=
by
  sorry

end fraction_spent_l778_77823


namespace sum_first_six_terms_geometric_seq_l778_77829

theorem sum_first_six_terms_geometric_seq (a r : ℝ)
  (h1 : a + a * r = 12)
  (h2 : a + a * r + a * r^2 + a * r^3 = 36) :
  a + a * r + a * r^2 + a * r^3 + a * r^4 + a * r^5 = 84 :=
sorry

end sum_first_six_terms_geometric_seq_l778_77829


namespace train_speed_l778_77892

theorem train_speed
  (length_train : ℝ)
  (length_bridge : ℝ)
  (time_seconds : ℝ) :
  length_train = 140 →
  length_bridge = 235.03 →
  time_seconds = 30 →
  (length_train + length_bridge) / time_seconds * 3.6 = 45.0036 :=
by
  intros h1 h2 h3
  sorry

end train_speed_l778_77892


namespace winner_won_by_l778_77861

theorem winner_won_by (V : ℝ) (h₁ : 0.62 * V = 806) : 806 - 0.38 * V = 312 :=
by
  sorry

end winner_won_by_l778_77861


namespace halloween_candy_l778_77853

theorem halloween_candy : 23 - 7 + 21 = 37 :=
by
  sorry

end halloween_candy_l778_77853


namespace initial_games_l778_77835

theorem initial_games (X : ℕ) (h1 : X - 68 + 47 = 74) : X = 95 :=
by
  sorry

end initial_games_l778_77835


namespace Mary_chestnuts_l778_77826

noncomputable def MaryPickedTwicePeter (P M : ℕ) := M = 2 * P
noncomputable def LucyPickedMorePeter (P L : ℕ) := L = P + 2
noncomputable def TotalPicked (P M L : ℕ) := P + M + L = 26

theorem Mary_chestnuts (P M L : ℕ) (h1 : MaryPickedTwicePeter P M) (h2 : LucyPickedMorePeter P L) (h3 : TotalPicked P M L) :
  M = 12 :=
sorry

end Mary_chestnuts_l778_77826


namespace students_remaining_after_fifth_stop_l778_77837

theorem students_remaining_after_fifth_stop (initial_students : ℕ) (stops : ℕ) :
  initial_students = 60 →
  stops = 5 →
  (∀ n, (n < stops → ∃ k, n = 3 * k + 1) → ∀ x, x = initial_students * ((2 : ℚ) / 3)^stops) →
  initial_students * ((2 : ℚ) / 3)^stops = (640 / 81 : ℚ) :=
by
  intros h_initial h_stops h_formula
  sorry

end students_remaining_after_fifth_stop_l778_77837


namespace average_class_score_l778_77863

theorem average_class_score (total_students assigned_day_students make_up_date_students : ℕ)
  (assigned_day_percentage make_up_date_percentage assigned_day_avg_score make_up_date_avg_score : ℝ)
  (h1 : total_students = 100)
  (h2 : assigned_day_percentage = 0.70)
  (h3 : make_up_date_percentage = 0.30)
  (h4 : assigned_day_students = 70)
  (h5 : make_up_date_students = 30)
  (h6 : assigned_day_avg_score = 55)
  (h7 : make_up_date_avg_score = 95) :
  (assigned_day_avg_score * assigned_day_students + make_up_date_avg_score * make_up_date_students) / total_students = 67 :=
by
  sorry

end average_class_score_l778_77863


namespace length_AC_l778_77803

theorem length_AC {AB BC : ℝ} (h1: AB = 6) (h2: BC = 4) : (AC = 2 ∨ AC = 10) :=
sorry

end length_AC_l778_77803


namespace Kyler_wins_l778_77824

variable (K : ℕ) -- Kyler's wins

/- Constants based on the problem statement -/
def Peter_wins := 5
def Peter_losses := 3
def Emma_wins := 2
def Emma_losses := 4
def Total_games := 15
def Kyler_losses := 4

/- Definition that calculates total games played -/
def total_games_played := 2 * Total_games

/- Game equation based on the total count of played games -/
def game_equation := Peter_wins + Peter_losses + Emma_wins + Emma_losses + K + Kyler_losses = total_games_played

/- Question: Calculate Kyler's wins assuming the given conditions -/
theorem Kyler_wins : K = 1 :=
by
  sorry

end Kyler_wins_l778_77824


namespace red_apples_sold_l778_77840

-- Define the variables and constants
variables (R G : ℕ)

-- Conditions (Definitions)
def ratio_condition : Prop := R / G = 8 / 3
def combine_condition : Prop := R + G = 44

-- Theorem statement to show number of red apples sold is 32 under given conditions
theorem red_apples_sold : ratio_condition R G → combine_condition R G → R = 32 :=
by
sorry

end red_apples_sold_l778_77840


namespace max_load_per_truck_l778_77856

-- Definitions based on given conditions
def num_trucks : ℕ := 3
def total_boxes : ℕ := 240
def lighter_box_weight : ℕ := 10
def heavier_box_weight : ℕ := 40

-- Proof problem statement
theorem max_load_per_truck :
  (total_boxes / 2) * lighter_box_weight + (total_boxes / 2) * heavier_box_weight = 6000 →
  6000 / num_trucks = 2000 :=
by sorry

end max_load_per_truck_l778_77856


namespace percent_decrease_call_cost_l778_77807

theorem percent_decrease_call_cost (c1990 c2010 : ℝ) (h1990 : c1990 = 50) (h2010 : c2010 = 10) :
  ((c1990 - c2010) / c1990) * 100 = 80 :=
by
  sorry

end percent_decrease_call_cost_l778_77807


namespace min_value_of_expression_l778_77857

noncomputable def given_expression (x : ℝ) : ℝ := 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) + Real.sqrt ((2 + x)^2 + x^2)

theorem min_value_of_expression : ∃ x : ℝ, given_expression x = 6 * Real.sqrt 2 := 
by 
  use 0
  sorry

end min_value_of_expression_l778_77857


namespace cylinder_sphere_surface_area_ratio_l778_77828

theorem cylinder_sphere_surface_area_ratio 
  (d : ℝ) -- d represents the diameter of the sphere and the height of the cylinder
  (S1 S2 : ℝ) -- Surface areas of the cylinder and the sphere
  (r := d / 2) -- radius of the sphere
  (S1 := 6 * π * r ^ 2) -- surface area of the cylinder
  (S2 := 4 * π * r ^ 2) -- surface area of the sphere
  : S1 / S2 = 3 / 2 :=
  sorry

end cylinder_sphere_surface_area_ratio_l778_77828


namespace diana_erasers_l778_77868

theorem diana_erasers (number_of_friends : ℕ) (erasers_per_friend : ℕ) (total_erasers : ℕ) :
  number_of_friends = 48 →
  erasers_per_friend = 80 →
  total_erasers = number_of_friends * erasers_per_friend →
  total_erasers = 3840 :=
by
  intros h_friends h_erasers h_total
  sorry

end diana_erasers_l778_77868


namespace min_1x1_tiles_l778_77887

/-- To cover a 23x23 grid using 1x1, 2x2, and 3x3 tiles (without gaps or overlaps),
the minimum number of 1x1 tiles required is 1. -/
theorem min_1x1_tiles (a b c : ℕ) (h : a + 2 * b + 3 * c = 23 * 23) : 
  a ≥ 1 :=
sorry

end min_1x1_tiles_l778_77887


namespace total_juice_boxes_needed_l778_77888

-- Definitions for the conditions
def john_juice_per_week : Nat := 2 * 5
def john_school_weeks : Nat := 18 - 2 -- taking into account the holiday break

def samantha_juice_per_week : Nat := 1 * 5
def samantha_school_weeks : Nat := 16 - 2 -- taking into account after-school and holiday break

def heather_mon_wed_juice : Nat := 3 * 2
def heather_tue_thu_juice : Nat := 2 * 2
def heather_fri_juice : Nat := 1
def heather_juice_per_week : Nat := heather_mon_wed_juice + heather_tue_thu_juice + heather_fri_juice
def heather_school_weeks : Nat := 17 - 2 -- taking into account personal break and holiday break

-- Question and Answer in lean
theorem total_juice_boxes_needed : 
  (john_juice_per_week * john_school_weeks) + 
  (samantha_juice_per_week * samantha_school_weeks) + 
  (heather_juice_per_week * heather_school_weeks) = 395 := 
by
  sorry

end total_juice_boxes_needed_l778_77888


namespace smallest_value_of_y_l778_77808

open Real

theorem smallest_value_of_y : 
  ∃ (y : ℝ), 6 * y^2 - 29 * y + 24 = 0 ∧ (∀ z : ℝ, 6 * z^2 - 29 * z + 24 = 0 → y ≤ z) ∧ y = 4 / 3 := 
sorry

end smallest_value_of_y_l778_77808


namespace parrots_per_cage_l778_77812

theorem parrots_per_cage (P : ℕ) (total_birds total_cages parakeets_per_cage : ℕ)
  (h1 : total_cages = 4)
  (h2 : parakeets_per_cage = 2)
  (h3 : total_birds = 40)
  (h4 : total_birds = total_cages * (P + parakeets_per_cage)) :
  P = 8 :=
by
  sorry

end parrots_per_cage_l778_77812


namespace no_such_function_exists_l778_77879

theorem no_such_function_exists (f : ℕ → ℕ) (h : ∀ n, f (f n) = n + 2019) : false :=
sorry

end no_such_function_exists_l778_77879


namespace find_a_l778_77895
open Real

noncomputable def f (a x : ℝ) := x * sin x + a * x

theorem find_a (a : ℝ) : (deriv (f a) (π / 2) = 1) → a = 0 := by
  sorry

end find_a_l778_77895


namespace square_area_25_l778_77854

theorem square_area_25 (side_length : ℝ) (h_side_length : side_length = 5) : side_length * side_length = 25 := 
by
  rw [h_side_length]
  norm_num
  done

end square_area_25_l778_77854


namespace acetone_C_mass_percentage_l778_77897

noncomputable def mass_percentage_C_in_acetone : ℝ :=
  let atomic_mass_C := 12.01
  let atomic_mass_H := 1.01
  let atomic_mass_O := 16.00
  let molar_mass_acetone := (3 * atomic_mass_C) + (6 * atomic_mass_H) + (1 * atomic_mass_O)
  let total_mass_C := 3 * atomic_mass_C
  (total_mass_C / molar_mass_acetone) * 100

theorem acetone_C_mass_percentage :
  abs (mass_percentage_C_in_acetone - 62.01) < 0.01 := by
  sorry

end acetone_C_mass_percentage_l778_77897


namespace remainder_when_sum_divided_by_5_l778_77880

theorem remainder_when_sum_divided_by_5 (f y : ℤ) (k m : ℤ) 
  (hf : f = 5 * k + 3) (hy : y = 5 * m + 4) : 
  (f + y) % 5 = 2 := 
by {
  sorry
}

end remainder_when_sum_divided_by_5_l778_77880


namespace cylinder_surface_area_l778_77804

namespace SurfaceAreaProof

variables (a b : ℝ)

theorem cylinder_surface_area (a b : ℝ) :
  (2 * Real.pi * a * b) = (2 * Real.pi * a * b) :=
by sorry

end SurfaceAreaProof

end cylinder_surface_area_l778_77804


namespace find_omega2019_value_l778_77864

noncomputable def omega_n (n : ℕ) : ℝ := (2 * n - 1) * Real.pi / 2

theorem find_omega2019_value :
  omega_n 2019 = 4037 * Real.pi / 2 :=
by
  sorry

end find_omega2019_value_l778_77864


namespace katherine_time_20_l778_77886

noncomputable def time_katherine_takes (k : ℝ) :=
  let time_naomi_takes_per_website := (5/4) * k
  let total_websites := 30
  let total_time_naomi := 750
  time_naomi_takes_per_website = 25 ∧ k = 20

theorem katherine_time_20 :
  ∃ k : ℝ, time_katherine_takes k :=
by
  use 20
  sorry

end katherine_time_20_l778_77886


namespace PlanY_more_cost_effective_l778_77815

-- Define the gigabytes Tim uses
variable (y : ℕ)

-- Define the cost functions for Plan X and Plan Y in cents
def cost_PlanX (y : ℕ) := 25 * y
def cost_PlanY (y : ℕ) := 1500 + 15 * y

-- Prove that Plan Y is cheaper than Plan X when y >= 150
theorem PlanY_more_cost_effective (y : ℕ) : y ≥ 150 → cost_PlanY y < cost_PlanX y := by
  sorry

end PlanY_more_cost_effective_l778_77815


namespace solve_for_x_l778_77833

noncomputable def simplified_end_expr (x : ℝ) := x = 4 - Real.sqrt 7 
noncomputable def expressed_as_2_statement (x : ℝ) := (x ^ 2 - 4 * x + 5) = (4 * (x - 1))
noncomputable def domain_condition (x : ℝ) := (-5 < x) ∧ (x < 3)

theorem solve_for_x (x : ℝ) :
  domain_condition x →
  (expressed_as_2_statement x ↔ simplified_end_expr x) :=
by
  sorry

end solve_for_x_l778_77833


namespace value_of_a_l778_77810

theorem value_of_a (a b : ℝ) (A B : Set ℝ) 
  (hA : A = {a, a^2}) (hB : B = {1, b}) (hAB : A = B) : a = -1 := 
by 
  sorry

end value_of_a_l778_77810


namespace two_vertical_asymptotes_l778_77802

theorem two_vertical_asymptotes (k : ℝ) : 
  (∀ x : ℝ, (x ≠ 3 ∧ x ≠ -2) → 
           (∃ δ > 0, ∀ ε > 0, ∃ x' : ℝ, x + δ > x' ∧ x' > x - δ ∧ 
                             (x' ≠ 3 ∧ x' ≠ -2) → 
                             |(x'^2 + 2 * x' + k) / (x'^2 - x' - 6)| > 1/ε)) ↔ 
  (k ≠ -15 ∧ k ≠ 0) :=
sorry

end two_vertical_asymptotes_l778_77802


namespace rachel_lunch_problems_l778_77817

theorem rachel_lunch_problems (problems_per_minute minutes_before_bed total_problems : ℕ) 
    (h1 : problems_per_minute = 5)
    (h2 : minutes_before_bed = 12)
    (h3 : total_problems = 76) : 
    (total_problems - problems_per_minute * minutes_before_bed) = 16 :=
by
    sorry

end rachel_lunch_problems_l778_77817


namespace Roger_needs_to_delete_20_apps_l778_77842

def max_apps := 50
def recommended_apps := 35
def current_apps := 2 * recommended_apps
def apps_to_delete := current_apps - max_apps

theorem Roger_needs_to_delete_20_apps : apps_to_delete = 20 := by
  sorry

end Roger_needs_to_delete_20_apps_l778_77842


namespace instrument_costs_purchasing_plans_l778_77896

variable (x y : ℕ)
variable (a b : ℕ)

theorem instrument_costs : 
  (2 * x + 3 * y = 1700 ∧ 3 * x + y = 1500) →
  x = 400 ∧ y = 300 := 
by 
  intros h
  sorry

theorem purchasing_plans :
  (x = 400) → (y = 300) → (3 * a + 10 = b) →
  (400 * a + 300 * b ≤ 30000) →
  ((760 - 400) * a + (540 - 300) * b ≥ 21600) →
  (a = 18 ∧ b = 64 ∨ a = 19 ∧ b = 67 ∨ a = 20 ∧ b = 70) :=
by
  intros hx hy hab hcost hprofit
  sorry

end instrument_costs_purchasing_plans_l778_77896


namespace parabola_chord_constant_l778_77851

noncomputable def calcT (x₁ x₂ c : ℝ) : ℝ :=
  let a := x₁^2 + (2*x₁^2 - c)^2
  let b := x₂^2 + (2*x₂^2 - c)^2
  1 / Real.sqrt a + 1 / Real.sqrt b

theorem parabola_chord_constant (c : ℝ) (m x₁ x₂ : ℝ) 
    (h₁ : 2*x₁^2 - m*x₁ - c = 0) 
    (h₂ : 2*x₂^2 - m*x₂ - c = 0) : 
    calcT x₁ x₂ c = -20 / (7 * c) :=
by
  sorry

end parabola_chord_constant_l778_77851


namespace train_speed_l778_77871

/-- Proof problem: Speed calculation of a train -/
theorem train_speed :
  ∀ (length : ℝ) (time_seconds : ℝ) (speed_kmph : ℝ),
    length = 40 →
    time_seconds = 0.9999200063994881 →
    speed_kmph = (length / 1000) / (time_seconds / 3600) →
    speed_kmph = 144 :=
by
  intros length time_seconds speed_kmph h_length h_time_seconds h_speed_kmph
  rw [h_length, h_time_seconds] at h_speed_kmph
  -- sorry is used to skip the proof steps
  sorry 

end train_speed_l778_77871


namespace cargo_loaded_in_bahamas_l778_77870

def initial : ℕ := 5973
def final : ℕ := 14696
def loaded : ℕ := final - initial

theorem cargo_loaded_in_bahamas : loaded = 8723 := by
  sorry

end cargo_loaded_in_bahamas_l778_77870


namespace alice_lawn_area_l778_77831

theorem alice_lawn_area (posts : ℕ) (distance : ℕ) (ratio : ℕ) : 
    posts = 24 → distance = 5 → ratio = 3 → 
    ∃ (short_side long_side : ℕ), 
        (2 * (short_side + long_side - 2) = posts) ∧
        (long_side = ratio * short_side) ∧
        (distance * (short_side - 1) * distance * (long_side - 1) = 825) :=
by
  intros h_posts h_distance h_ratio
  sorry

end alice_lawn_area_l778_77831


namespace find_x_l778_77839

theorem find_x (x : ℝ) (h : 0 < x) (hx : 0.01 * x * x^2 = 16) : x = 12 :=
sorry

end find_x_l778_77839


namespace select_two_people_l778_77873

theorem select_two_people {n : ℕ} (h1 : n ≠ 0) (h2 : n ≥ 2) (h3 : (n - 1) ^ 2 = 25) : n = 6 :=
by
  sorry

end select_two_people_l778_77873


namespace probability_exactly_three_primes_l778_77819

noncomputable def prime_faces : Finset ℕ := {2, 3, 5, 7, 11}

def num_faces : ℕ := 12
def num_dice : ℕ := 7
def target_primes : ℕ := 3

def probability_three_primes : ℚ :=
  35 * ((5 / 12)^3 * (7 / 12)^4)

theorem probability_exactly_three_primes :
  probability_three_primes = (4375 / 51821766) :=
by
  sorry

end probability_exactly_three_primes_l778_77819


namespace find_number_of_cows_l778_77832

-- Definitions from the conditions
def number_of_ducks : ℕ := sorry
def number_of_cows : ℕ := sorry

-- Define the number of legs and heads
def legs := 2 * number_of_ducks + 4 * number_of_cows
def heads := number_of_ducks + number_of_cows

-- Given condition from the problem
def condition := legs = 2 * heads + 32

-- Assert the number of cows
theorem find_number_of_cows (h : condition) : number_of_cows = 16 :=
sorry

end find_number_of_cows_l778_77832


namespace trig_identity_l778_77874

theorem trig_identity (α : ℝ) (h : Real.tan (π - α) = 2) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 3 :=
by
  sorry

end trig_identity_l778_77874


namespace xiaobin_duration_l778_77889

def t1 : ℕ := 9
def t2 : ℕ := 15

theorem xiaobin_duration : t2 - t1 = 6 := by
  sorry

end xiaobin_duration_l778_77889


namespace arrange_polynomial_ascending_order_l778_77877

variable {R : Type} [Ring R] (x : R)

def p : R := 3 * x ^ 2 - x + x ^ 3 - 1

theorem arrange_polynomial_ascending_order : 
  p x = -1 - x + 3 * x ^ 2 + x ^ 3 :=
by
  sorry

end arrange_polynomial_ascending_order_l778_77877


namespace symmetry_axis_of_function_l778_77858

noncomputable def f (varphi : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (2 * x + varphi)

theorem symmetry_axis_of_function
  (varphi : ℝ) (h1 : |varphi| < Real.pi / 2)
  (h2 : f varphi (Real.pi / 6) = 1) :
  ∃ k : ℤ, (k * Real.pi / 2 + Real.pi / 3 = Real.pi / 3) :=
sorry

end symmetry_axis_of_function_l778_77858


namespace prime_solution_l778_77893

open Nat

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem prime_solution : ∀ (p q : ℕ), 
  is_prime p → is_prime q → 7 * p * q^2 + p = q^3 + 43 * p^3 + 1 → (p = 2 ∧ q = 7) :=
by
  intros p q hp hq h
  sorry

end prime_solution_l778_77893


namespace cos_segments_ratio_proof_l778_77860

open Real

noncomputable def cos_segments_ratio := 
  let p := 5
  let q := 26
  ∀ x : ℝ, (cos x = cos 50) → (p, q) = (5, 26)

theorem cos_segments_ratio_proof : cos_segments_ratio :=
by 
  sorry

end cos_segments_ratio_proof_l778_77860


namespace teacherZhangAge_in_5_years_correct_l778_77885

variable (a : ℕ)

def teacherZhangAgeCurrent := 3 * a - 2

def teacherZhangAgeIn5Years := teacherZhangAgeCurrent a + 5

theorem teacherZhangAge_in_5_years_correct :
  teacherZhangAgeIn5Years a = 3 * a + 3 := by
  sorry

end teacherZhangAge_in_5_years_correct_l778_77885


namespace percent_defective_units_l778_77862

-- Definition of the given problem conditions
variable (D : ℝ) -- D represents the percentage of defective units

-- The main statement we want to prove
theorem percent_defective_units (h1 : 0.04 * D = 0.36) : D = 9 := by
  sorry

end percent_defective_units_l778_77862


namespace not_algorithm_is_C_l778_77836

-- Definitions based on the conditions recognized in a)
def option_A := "To go from Zhongshan to Beijing, first take a bus, then take a train."
def option_B := "The steps to solve a linear equation are to eliminate the denominator, remove the brackets, transpose terms, combine like terms, and make the coefficient 1."
def option_C := "The equation x^2 - 4x + 3 = 0 has two distinct real roots."
def option_D := "When solving the inequality ax + 3 > 0, the first step is to transpose terms, and the second step is to discuss the sign of a."

-- Problem statement
theorem not_algorithm_is_C : 
  (option_C ≠ "algorithm for solving a problem") ∧ 
  (option_A = "algorithm for solving a problem") ∧ 
  (option_B = "algorithm for solving a problem") ∧ 
  (option_D = "algorithm for solving a problem") :=
  by 
  sorry

end not_algorithm_is_C_l778_77836


namespace distance_is_correct_l778_77834

noncomputable def distance_from_center_to_plane
  (O : Point)
  (radius : ℝ)
  (vertices : Point × Point × Point)
  (side_lengths : (ℝ × ℝ × ℝ)) :
  ℝ :=
  8.772

theorem distance_is_correct
  (O : Point)
  (radius : ℝ)
  (A B C : Point)
  (h_radius : radius = 10)
  (h_sides : side_lengths = (17, 17, 16))
  (vertices := (A, B, C)) :
  distance_from_center_to_plane O radius vertices side_lengths = 8.772 := by
  sorry

end distance_is_correct_l778_77834


namespace number_of_bottles_poured_l778_77841

/-- Definition of full cylinder capacity (fixed as 80 bottles) --/
def full_capacity : ℕ := 80

/-- Initial fraction of full capacity --/
def initial_fraction : ℚ := 3 / 4

/-- Final fraction of full capacity --/
def final_fraction : ℚ := 4 / 5

/-- Proof problem: Prove the number of bottles of oil poured into the cylinder --/
theorem number_of_bottles_poured :
  (final_fraction * full_capacity) - (initial_fraction * full_capacity) = 4 := by
  sorry

end number_of_bottles_poured_l778_77841


namespace three_digit_non_multiples_of_3_or_11_l778_77827

theorem three_digit_non_multiples_of_3_or_11 : 
  ∃ (n : ℕ), n = 546 ∧ 
  (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → 
    ¬ (x % 3 = 0 ∨ x % 11 = 0) → 
    n = (900 - (300 + 81 - 27))) := 
by 
  sorry

end three_digit_non_multiples_of_3_or_11_l778_77827


namespace problem_1_problem_2_l778_77814

open Set Real

-- Definition of the sets A, B, and the complement of B in the real numbers
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x > 2}
def C (a : ℝ) : Set ℝ := {x | 1 < x ∧ x < a}

-- Proof problem (1): Prove that A ∩ (complement of B) = [1, 2]
theorem problem_1 : (A ∩ (compl B)) = {x | 1 ≤ x ∧ x ≤ 2} := sorry

-- Proof problem (2): Prove that the set of values for the real number a such that C(a) ∩ A = C(a)
-- is (-∞, 3]
theorem problem_2 : { a : ℝ | C a ⊆ A } = { a : ℝ | a ≤ 3 } := sorry

end problem_1_problem_2_l778_77814


namespace cost_of_fencing_l778_77876

open Real

theorem cost_of_fencing
  (ratio_length_width : ∃ x : ℝ, 3 * x * 2 * x = 3750)
  (cost_per_meter : ℝ := 0.50) :
  ∃ cost : ℝ, cost = 125 := by
  sorry

end cost_of_fencing_l778_77876


namespace greatest_value_of_a_greatest_value_of_a_achieved_l778_77872

theorem greatest_value_of_a (a b : ℕ) (h1 : 5 * Nat.lcm a b + 2 * Nat.gcd a b = 120) : a ≤ 20 :=
sorry

theorem greatest_value_of_a_achieved (a b : ℕ) (h1 : 5 * Nat.lcm a b + 2 * Nat.gcd a b = 120)
  (h2 : Nat.gcd a b = 10) (h3 : 10 ∣ a ∧ 10 ∣ b) (h4 : Nat.lcm a b = 20) : a = 20 :=
sorry

end greatest_value_of_a_greatest_value_of_a_achieved_l778_77872
