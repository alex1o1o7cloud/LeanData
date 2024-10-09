import Mathlib

namespace total_distance_is_correct_l1890_189027

noncomputable def boat_speed : ℝ := 20 -- boat speed in still water (km/hr)
noncomputable def current_speed_first : ℝ := 5 -- current speed for the first 6 minutes (km/hr)
noncomputable def current_speed_second : ℝ := 8 -- current speed for the next 6 minutes (km/hr)
noncomputable def current_speed_third : ℝ := 3 -- current speed for the last 6 minutes (km/hr)
noncomputable def time_in_hours : ℝ := 6 / 60 -- 6 minutes in hours (0.1 hours)

noncomputable def total_distance_downstream := 
  (boat_speed + current_speed_first) * time_in_hours +
  (boat_speed + current_speed_second) * time_in_hours +
  (boat_speed + current_speed_third) * time_in_hours

theorem total_distance_is_correct : total_distance_downstream = 7.6 :=
  by 
  sorry

end total_distance_is_correct_l1890_189027


namespace cond_prob_B_given_A_l1890_189002

-- Definitions based on the conditions
def eventA := {n : ℕ | n > 4 ∧ n ≤ 6}
def eventB := {k : ℕ × ℕ | (k.1 + k.2) = 7}

-- Probability of event A
def probA := (2 : ℚ) / 6

-- Joint probability of events A and B
def probAB := (1 : ℚ) / (6 * 6)

-- Conditional probability P(B|A)
def cond_prob := probAB / probA

-- The final statement to prove
theorem cond_prob_B_given_A : cond_prob = 1 / 6 := by
  sorry

end cond_prob_B_given_A_l1890_189002


namespace total_volume_correct_l1890_189069

-- Defining the initial conditions
def carl_cubes : ℕ := 4
def carl_side_length : ℕ := 3
def kate_cubes : ℕ := 6
def kate_side_length : ℕ := 1

-- Given the above conditions, define the total volume of all cubes.
def total_volume_of_all_cubes : ℕ := (carl_cubes * carl_side_length ^ 3) + (kate_cubes * kate_side_length ^ 3)

-- The statement we need to prove
theorem total_volume_correct :
  total_volume_of_all_cubes = 114 :=
by
  -- Skipping the proof with sorry as per the instruction
  sorry

end total_volume_correct_l1890_189069


namespace sufficient_not_necessary_l1890_189061

theorem sufficient_not_necessary (a b : ℝ) :
  (a = -1 ∧ b = 2 → a * b = -2) ∧ (a * b = -2 → ¬(a = -1 ∧ b = 2)) :=
by
  sorry

end sufficient_not_necessary_l1890_189061


namespace pet_store_satisfaction_l1890_189028

theorem pet_store_satisfaction :
  let puppies := 15
  let kittens := 6
  let hamsters := 8
  let friends := 3
  puppies * kittens * hamsters * friends.factorial = 4320 := by
  sorry

end pet_store_satisfaction_l1890_189028


namespace rectangle_diagonal_length_l1890_189009

theorem rectangle_diagonal_length (p : ℝ) (r_lw : ℝ) (l w d : ℝ) 
    (h_p : p = 84) 
    (h_ratio : r_lw = 5 / 2) 
    (h_l : l = 5 * (p / 2) / 7) 
    (h_w : w = 2 * (p / 2) / 7) 
    (h_d : d = Real.sqrt (l ^ 2 + w ^ 2)) :
  d = 2 * Real.sqrt 261 :=
by
  sorry

end rectangle_diagonal_length_l1890_189009


namespace number_of_ways_to_choose_museums_l1890_189084

-- Define the conditions
def number_of_grades : Nat := 6
def number_of_museums : Nat := 6
def number_of_grades_Museum_A : Nat := 2

-- Prove the number of ways to choose museums such that exactly two grades visit Museum A
theorem number_of_ways_to_choose_museums :
  (Nat.choose number_of_grades number_of_grades_Museum_A) * (5 ^ (number_of_grades - number_of_grades_Museum_A)) = Nat.choose 6 2 * 5 ^ 4 :=
by
  sorry

end number_of_ways_to_choose_museums_l1890_189084


namespace fraction_sum_l1890_189014

theorem fraction_sum : (1 / 3 : ℚ) + (2 / 7) + (3 / 8) = 167 / 168 := by
  sorry

end fraction_sum_l1890_189014


namespace range_of_a_l1890_189013

theorem range_of_a (a : ℝ) :
  ¬ (∃ x0 : ℝ, 2^x0 - 2 ≤ a^2 - 3 * a) → (1 ≤ a ∧ a ≤ 2) :=
by
  sorry

end range_of_a_l1890_189013


namespace find_radius_of_base_of_cone_l1890_189018

noncomputable def radius_of_cone (CSA : ℝ) (l : ℝ) : ℝ :=
  CSA / (Real.pi * l)

theorem find_radius_of_base_of_cone :
  radius_of_cone 527.7875658030853 14 = 12 :=
by
  sorry

end find_radius_of_base_of_cone_l1890_189018


namespace geometric_sequence_common_ratio_l1890_189050

theorem geometric_sequence_common_ratio
  (S : ℕ → ℝ)
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : S 1 = a 1)
  (h2 : S 2 = a 1 + a 1 * q)
  (h3 : a 2 = a 1 * q)
  (h4 : a 3 = a 1 * q^2)
  (h5 : 3 * S 2 = a 3 - 2)
  (h6 : 3 * S 1 = a 2 - 2) :
  q = 4 :=
sorry

end geometric_sequence_common_ratio_l1890_189050


namespace part1_part2_l1890_189065

open Set

def A (a : ℝ) : Set ℝ := { x | x^2 - a * x + a^2 - 19 = 0 }
def B : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }
def C : Set ℝ := { x | x^2 + 2 * x - 8 = 0 }

theorem part1 (a : ℝ) : (A a ∩ B = A a ∪ B) → a = 5 :=
by
  sorry

theorem part2 (a : ℝ) : (A a ∩ B ≠ ∅) ∧ (A a ∩ C = ∅) → a = -2 :=
by
  sorry

end part1_part2_l1890_189065


namespace part_I_part_II_l1890_189079

-- Define the function f
def f (x a : ℝ) := |x - a| + |x - 2|

-- Statement for part (I)
theorem part_I (a : ℝ) (h : ∃ x : ℝ, f x a ≤ a) : a ≥ 1 := sorry

-- Statement for part (II)
theorem part_II (m n p : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : p > 0) (h4 : m + 2 * n + 3 * p = 1) : 
  (3 / m) + (2 / n) + (1 / p) ≥ 6 + 2 * Real.sqrt 6 + 2 * Real.sqrt 2 := sorry

end part_I_part_II_l1890_189079


namespace squares_overlap_ratio_l1890_189029

theorem squares_overlap_ratio (a b : ℝ) (h1 : 0.52 * a^2 = a^2 - (a^2 - 0.52 * a^2))
                             (h2 : 0.73 * b^2 = b^2 - (b^2 - 0.73 * b^2)) :
                             a / b = 3 / 4 := by
sorry

end squares_overlap_ratio_l1890_189029


namespace kabadi_players_l1890_189071

def people_play_kabadi (Kho_only Both Total : ℕ) : Prop :=
  ∃ K : ℕ, Kho_only = 20 ∧ Both = 5 ∧ Total = 30 ∧ K = Total - Kho_only ∧ (K + Both) = 15

theorem kabadi_players :
  people_play_kabadi 20 5 30 :=
by
  sorry

end kabadi_players_l1890_189071


namespace cookies_on_third_plate_l1890_189064

theorem cookies_on_third_plate :
  ∀ (a5 a7 a14 a19 a25 : ℕ),
  (a5 = 5) ∧ (a7 = 7) ∧ (a14 = 14) ∧ (a19 = 19) ∧ (a25 = 25) →
  ∃ (a12 : ℕ), a12 = 12 :=
by
  sorry

end cookies_on_third_plate_l1890_189064


namespace abs_diff_eq_seven_l1890_189037

theorem abs_diff_eq_seven (m n : ℤ) (h1 : |m| = 5) (h2 : |n| = 2) (h3 : m * n < 0) : |m - n| = 7 := 
sorry

end abs_diff_eq_seven_l1890_189037


namespace real_solutions_l1890_189042

-- Given the condition (equation)
def quadratic_equation (x y : ℝ) : Prop :=
  x^2 + 2 * x * Real.sin (x * y) + 1 = 0

-- The main theorem statement proving the solutions for x and y
theorem real_solutions (x y : ℝ) (k : ℤ) :
  quadratic_equation x y ↔
  (x = 1 ∧ (y = (Real.pi / 2 + 2 * k * Real.pi) ∨ y = (-Real.pi / 2 + 2 * k * Real.pi))) ∨
  (x = -1 ∧ (y = (-Real.pi / 2 + 2 * k * Real.pi) ∨ y = (Real.pi / 2 + 2 * k * Real.pi))) :=
by
  sorry

end real_solutions_l1890_189042


namespace shaded_area_l1890_189059

theorem shaded_area (PR PV PQ QR : ℝ) (hPR : PR = 20) (hPV : PV = 12) (hPQ_QR : PQ + QR = PR) :
  PR * PV - 1 / 2 * 12 * PR = 120 :=
by
  -- Definitions used earlier
  have h_area_rectangle : PR * PV = 240 := by
    rw [hPR, hPV]
    norm_num
  have h_half_total_unshaded : (1 / 2) * 12 * PR = 120 := by
    rw [hPR]
    norm_num
  rw [h_area_rectangle, h_half_total_unshaded]
  norm_num

end shaded_area_l1890_189059


namespace smallest_sum_is_S5_l1890_189039

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Definitions of arithmetic sequence sum
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Conditions
axiom h1 : a 3 + a 8 > 0
axiom h2 : S 9 < 0

-- Statements relating terms and sums in arithmetic sequence
axiom h3 : ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

theorem smallest_sum_is_S5 (seq_a : arithmetic_sequence a) : S 5 ≤ S 1 ∧ S 5 ≤ S 2 ∧ S 5 ≤ S 3 ∧ S 5 ≤ S 4 ∧ S 5 ≤ S 6 ∧ S 5 ≤ S 7 ∧ S 5 ≤ S 8 ∧ S 5 ≤ S 9 :=
by {
    sorry
}

end smallest_sum_is_S5_l1890_189039


namespace min_colors_5x5_grid_l1890_189074

def is_valid_coloring (grid : Fin 5 × Fin 5 → ℕ) (k : ℕ) : Prop :=
  ∀ i j : Fin 5, ∀ di dj : Fin 2, ∀ c : ℕ,
    (di ≠ 0 ∨ dj ≠ 0) →
    (grid (i, j) = c ∧ grid (i + di, j + dj) = c ∧ grid (i + 2 * di, j + 2 * dj) = c) → 
    False

theorem min_colors_5x5_grid : 
  ∀ (grid : Fin 5 × Fin 5 → ℕ), (∀ i j, grid (i, j) < 3) → is_valid_coloring grid 3 := 
by
  sorry

end min_colors_5x5_grid_l1890_189074


namespace max_value_l1890_189003

open Real

/-- Given vectors a, b, and c, and real numbers m and n such that m * a + n * b = c,
prove that the maximum value for (m - 3)^2 + n^2 is 16. --/
theorem max_value
  (α : ℝ)
  (a : ℝ × ℝ) (b : ℝ × ℝ) (c : ℝ × ℝ)
  (m n : ℝ)
  (ha : a = (1, 1))
  (hb : b = (1, -1))
  (hc : c = (sqrt 2 * cos α, sqrt 2 * sin α))
  (h : m * a.1 + n * b.1 = c.1 ∧ m * a.2 + n * b.2 = c.2) :
  (m - 3)^2 + n^2 ≤ 16 :=
by
  sorry

end max_value_l1890_189003


namespace maximum_x_y_value_l1890_189081

theorem maximum_x_y_value (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h1 : x + 2 * y ≤ 6) (h2 : 2 * x + y ≤ 6) : x + y ≤ 4 := 
sorry

end maximum_x_y_value_l1890_189081


namespace quadratic_solution_unique_l1890_189049

noncomputable def solve_quad_eq (a : ℝ) (h : a ≠ 0) (h_uniq : (36^2 - 4 * a * 12) = 0) : ℝ :=
-2 / 3

theorem quadratic_solution_unique (a : ℝ) (h : a ≠ 0) (h_uniq : (36^2 - 4 * a * 12) = 0) :
  (∃! x : ℝ, a * x^2 + 36 * x + 12 = 0) ∧ (solve_quad_eq a h h_uniq) = -2 / 3 :=
by
  sorry

end quadratic_solution_unique_l1890_189049


namespace cube_volume_and_diagonal_from_surface_area_l1890_189058

theorem cube_volume_and_diagonal_from_surface_area
    (A : ℝ) (h : A = 150) :
    ∃ (V : ℝ) (d : ℝ), V = 125 ∧ d = 5 * Real.sqrt 3 :=
by
  sorry

end cube_volume_and_diagonal_from_surface_area_l1890_189058


namespace number_of_diagonals_in_hexagon_l1890_189021

-- Define the number of sides of the hexagon
def sides_of_hexagon : ℕ := 6

-- Define the formula for the number of diagonals in an n-sided polygon
def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- The theorem we want to prove
theorem number_of_diagonals_in_hexagon : number_of_diagonals sides_of_hexagon = 9 :=
by
  sorry

end number_of_diagonals_in_hexagon_l1890_189021


namespace order_wxyz_l1890_189090

def w : ℕ := 2^129 * 3^81 * 5^128
def x : ℕ := 2^127 * 3^81 * 5^128
def y : ℕ := 2^126 * 3^82 * 5^128
def z : ℕ := 2^125 * 3^82 * 5^129

theorem order_wxyz : x < y ∧ y < z ∧ z < w := by
  sorry

end order_wxyz_l1890_189090


namespace isosceles_triangle_perimeter_l1890_189031

/-- 
Prove that the perimeter of an isosceles triangle with sides 6 cm and 8 cm, 
and an area of 12 cm², is 20 cm.
--/
theorem isosceles_triangle_perimeter (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (S : ℝ) (h3 : S = 12) :
  a ≠ b →
  a = c ∨ b = c →
  ∃ P : ℝ, P = 20 := sorry

end isosceles_triangle_perimeter_l1890_189031


namespace smoking_lung_disease_confidence_l1890_189062

/-- Prove that given the conditions, the correct statement is C:
   If it is concluded from the statistic that there is a 95% confidence 
   that smoking is related to lung disease, then there is a 5% chance of
   making a wrong judgment. -/
theorem smoking_lung_disease_confidence 
  (P Q : Prop) 
  (confidence_level : ℝ) 
  (h_conf : confidence_level = 0.95) 
  (h_PQ : P → (Q → true)) :
  ¬Q → (confidence_level = 1 - 0.05) :=
by
  sorry

end smoking_lung_disease_confidence_l1890_189062


namespace total_food_eaten_l1890_189099

theorem total_food_eaten (num_puppies num_dogs : ℕ)
    (dog_food_per_meal dog_meals_per_day puppy_food_per_day : ℕ)
    (dog_food_mult puppy_meal_mult : ℕ)
    (h1 : num_puppies = 6)
    (h2 : num_dogs = 5)
    (h3 : dog_food_per_meal = 6)
    (h4 : dog_meals_per_day = 2)
    (h5 : dog_food_mult = 3)
    (h6 : puppy_meal_mult = 4)
    (h7 : puppy_food_per_day = (dog_food_per_meal / dog_food_mult) * puppy_meal_mult * dog_meals_per_day) :
    (num_dogs * dog_food_per_meal * dog_meals_per_day + num_puppies * puppy_food_per_day) = 108 := by
  -- conclude the theorem
  sorry

end total_food_eaten_l1890_189099


namespace broadway_show_total_amount_collected_l1890_189006

theorem broadway_show_total_amount_collected (num_adults num_children : ℕ) 
  (adult_ticket_price child_ticket_ratio : ℕ) 
  (child_ticket_price : ℕ) 
  (h1 : num_adults = 400) 
  (h2 : num_children = 200) 
  (h3 : adult_ticket_price = 32) 
  (h4 : child_ticket_ratio = 2) 
  (h5 : adult_ticket_price = child_ticket_ratio * child_ticket_price) : 
  num_adults * adult_ticket_price + num_children * child_ticket_price = 16000 := 
  by 
    sorry

end broadway_show_total_amount_collected_l1890_189006


namespace solve_quadratic_eq_solve_linear_system_l1890_189094

theorem solve_quadratic_eq (x : ℚ) : 4 * (x - 1) ^ 2 - 25 = 0 ↔ x = 7 / 2 ∨ x = -3 / 2 := 
by sorry

theorem solve_linear_system (x y : ℚ) : (2 * x - y = 4) ∧ (3 * x + 2 * y = 1) ↔ (x = 9 / 7 ∧ y = -10 / 7) :=
by sorry

end solve_quadratic_eq_solve_linear_system_l1890_189094


namespace problem_l1890_189086

noncomputable def k : ℝ := 2.9

theorem problem (k : ℝ) (hₖ : k > 1) 
    (h_sum : ∑' n, (7 * n + 2) / k^n = 20 / 3) : 
    k = 2.9 := 
sorry

end problem_l1890_189086


namespace fully_charge_tablet_time_l1890_189080

def time_to_fully_charge_smartphone := 26 -- 26 minutes to fully charge a smartphone
def total_charge_time := 66 -- 66 minutes to charge tablet fully and phone halfway
def halfway_charge_time := time_to_fully_charge_smartphone / 2 -- 13 minutes to charge phone halfway

theorem fully_charge_tablet_time : 
  ∃ T : ℕ, T + halfway_charge_time = total_charge_time ∧ T = 53 := 
by
  sorry

end fully_charge_tablet_time_l1890_189080


namespace necklace_length_l1890_189041

-- Given conditions as definitions in Lean
def num_pieces : ℕ := 16
def piece_length : ℝ := 10.4
def overlap_length : ℝ := 3.5
def effective_length : ℝ := piece_length - overlap_length
def total_length : ℝ := effective_length * num_pieces

-- The theorem to prove
theorem necklace_length :
  total_length = 110.4 :=
by
  -- Proof omitted
  sorry

end necklace_length_l1890_189041


namespace min_max_F_l1890_189023

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

def F (x y : ℝ) : ℝ := x^2 + y^2

theorem min_max_F (x y : ℝ) (h1 : f (y^2 - 6 * y + 11) + f (x^2 - 8 * x + 10) ≤ 0) (h2 : y ≥ 3) :
  ∃ (min_val max_val : ℝ), min_val = 13 ∧ max_val = 49 ∧
    min_val ≤ F x y ∧ F x y ≤ max_val :=
sorry

end min_max_F_l1890_189023


namespace find_f_24_25_26_l1890_189097

-- Given conditions
def homogeneous (f : ℤ → ℤ → ℤ → ℝ) : Prop :=
  ∀ (n a b c : ℤ), f (n * a) (n * b) (n * c) = n * f a b c

def shift_invariance (f : ℤ → ℤ → ℤ → ℝ) : Prop :=
  ∀ (a b c n : ℤ), f (a + n) (b + n) (c + n) = f a b c + n

def symmetry (f : ℤ → ℤ → ℤ → ℝ) : Prop :=
  ∀ (a b c : ℤ), f a b c = f c b a

-- Proving the required value under the conditions
theorem find_f_24_25_26 (f : ℤ → ℤ → ℤ → ℝ)
  (homo : homogeneous f) 
  (shift : shift_invariance f) 
  (symm : symmetry f) : 
  f 24 25 26 = 25 := 
sorry

end find_f_24_25_26_l1890_189097


namespace trigonometric_identity_l1890_189056

theorem trigonometric_identity (x : ℝ) (h : Real.sin (x + Real.pi / 4) = 1 / 3) :
  Real.sin (4 * x) - 2 * Real.cos (3 * x) * Real.sin x = -7 / 9 :=
by
  sorry

end trigonometric_identity_l1890_189056


namespace x_cubed_plus_square_plus_lin_plus_a_l1890_189036

theorem x_cubed_plus_square_plus_lin_plus_a (a b x : ℝ) (h : b / x^3 + 1 / x^2 + 1 / x + 1 = 0) :
  x^3 + x^2 + x + a = a - b :=
by {
  sorry
}

end x_cubed_plus_square_plus_lin_plus_a_l1890_189036


namespace min_S_l1890_189075

variable {x y : ℝ}
def condition (x y : ℝ) : Prop := (4 * x^2 + 5 * x * y + 4 * y^2 = 5)
def S (x y : ℝ) : ℝ := x^2 + y^2
theorem min_S (hx : condition x y) : S x y = (10 / 13) :=
sorry

end min_S_l1890_189075


namespace eccentricity_of_hyperbola_l1890_189035

theorem eccentricity_of_hyperbola
  (a b c e : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c = a * e)
  (h4 : c^2 = a^2 + b^2)
  (h5 : ∀ B : ℝ × ℝ, B = (0, b))
  (h6 : ∀ F : ℝ × ℝ, F = (c, 0))
  (h7 : ∀ m_FB m_asymptote : ℝ, m_FB * m_asymptote = -1 → (m_FB = -b / c) ∧ (m_asymptote = b / a)) :
  e = (1 + Real.sqrt 5) / 2 :=
sorry

end eccentricity_of_hyperbola_l1890_189035


namespace compute_operation_l1890_189012

def operation_and (x : ℝ) := 10 - x
def operation_and_prefix (x : ℝ) := x - 10

theorem compute_operation (x : ℝ) : operation_and_prefix (operation_and 15) = -15 :=
by
  sorry

end compute_operation_l1890_189012


namespace initial_seashells_l1890_189007

-- Definitions based on the problem conditions
def gave_to_joan : ℕ := 6
def left_with_jessica : ℕ := 2

-- Theorem statement to prove the number of seashells initially found by Jessica
theorem initial_seashells : gave_to_joan + left_with_jessica = 8 := by
  -- Proof goes here
  sorry

end initial_seashells_l1890_189007


namespace worker_allocation_correct_l1890_189095

variable (x y : ℕ)
variable (H1 : x + y = 50)
variable (H2 : x = 30)
variable (H3 : y = 20)
variable (H4 : 120 * (50 - x) = 2 * 40 * x)

theorem worker_allocation_correct 
  (h₁ : x = 30) 
  (h₂ : y = 20) 
  (h₃ : x + y = 50) 
  (h₄ : 120 * (50 - x) = 2 * 40 * x) 
  : true := 
by
  sorry

end worker_allocation_correct_l1890_189095


namespace distance_from_A_to_C_l1890_189096

theorem distance_from_A_to_C (x y : ℕ) (d : ℚ)
  (h1 : d = x / 3) 
  (h2 : 13 + (d * 15) / (y - 13) = 2 * x)
  (h3 : y = 2 * x + 13) 
  : x + y = 26 := 
  sorry

end distance_from_A_to_C_l1890_189096


namespace only_set_C_forms_triangle_l1890_189089

def triangle_inequality (a b c : ℝ) : Prop := 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem only_set_C_forms_triangle : 
  (¬ triangle_inequality 1 2 3) ∧ 
  (¬ triangle_inequality 2 3 6) ∧ 
  triangle_inequality 4 6 8 ∧ 
  (¬ triangle_inequality 5 6 12) := 
by 
  sorry

end only_set_C_forms_triangle_l1890_189089


namespace abs_difference_of_mn_6_and_sum_7_l1890_189092

theorem abs_difference_of_mn_6_and_sum_7 (m n : ℝ) (h₁ : m * n = 6) (h₂ : m + n = 7) : |m - n| = 5 := 
sorry

end abs_difference_of_mn_6_and_sum_7_l1890_189092


namespace remaining_money_after_purchases_l1890_189043

def initial_amount : ℝ := 100
def bread_cost : ℝ := 4
def candy_cost : ℝ := 3
def cereal_cost : ℝ := 6
def fruit_percentage : ℝ := 0.2
def milk_cost_each : ℝ := 4.50
def turkey_fraction : ℝ := 0.25

-- Calculate total spent on initial purchases
def initial_spent : ℝ := bread_cost + (2 * candy_cost) + cereal_cost

-- Remaining amount after initial purchases
def remaining_after_initial : ℝ := initial_amount - initial_spent

-- Spend 20% on fruits
def spent_on_fruits : ℝ := fruit_percentage * remaining_after_initial
def remaining_after_fruits : ℝ := remaining_after_initial - spent_on_fruits

-- Spend on two gallons of milk
def spent_on_milk : ℝ := 2 * milk_cost_each
def remaining_after_milk : ℝ := remaining_after_fruits - spent_on_milk

-- Spend 1/4 on turkey
def spent_on_turkey : ℝ := turkey_fraction * remaining_after_milk
def final_remaining : ℝ := remaining_after_milk - spent_on_turkey

theorem remaining_money_after_purchases : final_remaining = 43.65 := by
  sorry

end remaining_money_after_purchases_l1890_189043


namespace circle_condition_l1890_189025

theorem circle_condition (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 4 * m * x - 2 * y + 5 * m = 0) ↔ (m < 1 / 4 ∨ m > 1) :=
sorry

end circle_condition_l1890_189025


namespace find_t_plus_a3_l1890_189052

noncomputable def geometric_sequence_sum (n : ℕ) (t : ℤ) : ℤ :=
  3 ^ n + t

noncomputable def a_1 (t : ℤ) : ℤ :=
  geometric_sequence_sum 1 t

noncomputable def a_2 (t : ℤ) : ℤ :=
  geometric_sequence_sum 2 t - geometric_sequence_sum 1 t

noncomputable def a_3 (t : ℤ) : ℤ :=
  geometric_sequence_sum 3 t - geometric_sequence_sum 2 t

theorem find_t_plus_a3 (t : ℤ) : t + a_3 t = 17 :=
sorry

end find_t_plus_a3_l1890_189052


namespace intersection_M_N_l1890_189020

-- Define the sets M and N according to the conditions given in the problem
def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 ≤ x}

-- State the theorem to prove the intersection of M and N
theorem intersection_M_N : (M ∩ N) = {0, 1} := 
  sorry

end intersection_M_N_l1890_189020


namespace valentine_giveaway_l1890_189078

theorem valentine_giveaway (initial : ℕ) (left : ℕ) (given : ℕ) (h1 : initial = 30) (h2 : left = 22) : given = initial - left → given = 8 :=
by
  sorry

end valentine_giveaway_l1890_189078


namespace equalize_cheese_pieces_l1890_189077

-- Defining the initial masses of the three pieces of cheese
def cheese1 : ℕ := 5
def cheese2 : ℕ := 8
def cheese3 : ℕ := 11

-- State that the fox can cut 1g simultaneously from any two pieces
def can_equalize_masses (cut_action : ℕ → ℕ → ℕ → Prop) : Prop :=
  ∃ n1 n2 n3 _ : ℕ,
    cut_action cheese1 cheese2 cheese3 ∧
    (n1 = 0 ∧ n2 = 0 ∧ n3 = 0)

-- Introducing the fox's cut action
def cut_action (a b c : ℕ) : Prop :=
  (∃ x : ℕ, x ≥ 0 ∧ a - x ≥ 0 ∧ b - x ≥ 0 ∧ c ≤ cheese3) ∧
  (∃ y : ℕ, y ≥ 0 ∧ a - y ≥ 0 ∧ b ≤ cheese2 ∧ c - y ≥ 0) ∧
  (∃ z : ℕ, z ≥ 0 ∧ a ≤ cheese1 ∧ b - z ≥ 0 ∧ c - z ≥ 0) 

-- The theorem that proves it's possible to equalize the masses
theorem equalize_cheese_pieces : can_equalize_masses cut_action :=
by
  sorry

end equalize_cheese_pieces_l1890_189077


namespace ab_condition_l1890_189054

theorem ab_condition (a b : ℝ) : ¬((a + b > 1 → a^2 + b^2 > 1) ∧ (a^2 + b^2 > 1 → a + b > 1)) :=
by {
  -- This proof problem states that the condition "a + b > 1" is neither sufficient nor necessary for "a^2 + b^2 > 1".
  sorry
}

end ab_condition_l1890_189054


namespace regular_polygon_sides_l1890_189088

theorem regular_polygon_sides (P s : ℕ) (hP : P = 180) (hs : s = 15) : P / s = 12 := by
  -- Given
  -- P = 180  -- the perimeter in cm
  -- s = 15   -- the side length in cm
  sorry

end regular_polygon_sides_l1890_189088


namespace difference_between_mean_and_median_l1890_189093

namespace MathProof

noncomputable def percentage_72 := 0.12
noncomputable def percentage_82 := 0.30
noncomputable def percentage_87 := 0.18
noncomputable def percentage_91 := 0.10
noncomputable def percentage_96 := 1 - (percentage_72 + percentage_82 + percentage_87 + percentage_91)

noncomputable def num_students := 20
noncomputable def scores := [72, 72, 82, 82, 82, 82, 82, 82, 87, 87, 87, 87, 91, 91, 96, 96, 96, 96, 96, 96]

noncomputable def mean_score : ℚ := (72 * 2 + 82 * 6 + 87 * 4 + 91 * 2 + 96 * 6) / num_students
noncomputable def median_score : ℚ := 87

theorem difference_between_mean_and_median :
  mean_score - median_score = 0.1 := by
  sorry

end MathProof

end difference_between_mean_and_median_l1890_189093


namespace number_of_pairings_l1890_189060

-- Definitions for conditions.
def bowls : Finset String := {"red", "blue", "yellow", "green"}
def glasses : Finset String := {"red", "blue", "yellow", "green"}

-- The theorem statement
theorem number_of_pairings : bowls.card * glasses.card = 16 := by
  sorry

end number_of_pairings_l1890_189060


namespace test_completion_days_l1890_189051

theorem test_completion_days :
  let barbara_days := 10
  let edward_days := 9
  let abhinav_days := 11
  let alex_days := 12
  let barbara_rate := 1 / barbara_days
  let edward_rate := 1 / edward_days
  let abhinav_rate := 1 / abhinav_days
  let alex_rate := 1 / alex_days
  let one_cycle_work := barbara_rate + edward_rate + abhinav_rate + alex_rate
  let cycles_needed := (1 : ℚ) / one_cycle_work
  Nat.ceil cycles_needed = 3 :=
by
  sorry

end test_completion_days_l1890_189051


namespace negation_of_universal_statement_l1890_189008

theorem negation_of_universal_statement :
  ¬(∀ a : ℝ, ∃ x : ℝ, x > 0 ∧ a * x^2 - 3 * x - a = 0) ↔ ∃ a : ℝ, ∀ x : ℝ, ¬(x > 0 ∧ a * x^2 - 3 * x - a = 0) :=
by sorry

end negation_of_universal_statement_l1890_189008


namespace solve_for_a_l1890_189030

theorem solve_for_a (a : ℚ) (h : a + a / 3 = 8 / 3) : a = 2 :=
sorry

end solve_for_a_l1890_189030


namespace absolute_value_condition_necessary_non_sufficient_l1890_189072

theorem absolute_value_condition_necessary_non_sufficient (x : ℝ) :
  (abs (x - 1) < 2 → x^2 < x) ∧ ¬ (x^2 < x → abs (x - 1) < 2) := sorry

end absolute_value_condition_necessary_non_sufficient_l1890_189072


namespace poodle_terrier_bark_ratio_l1890_189046

theorem poodle_terrier_bark_ratio :
  ∀ (P T : ℕ),
  (T = 12) →
  (P = 24) →
  (P / T = 2) :=
by intros P T hT hP
   sorry

end poodle_terrier_bark_ratio_l1890_189046


namespace musketeers_strength_order_l1890_189091

variables {A P R D : ℝ}

theorem musketeers_strength_order 
  (h1 : P + D > A + R)
  (h2 : P + A > R + D)
  (h3 : P + R = A + D) : 
  P > D ∧ D > A ∧ A > R :=
by
  sorry

end musketeers_strength_order_l1890_189091


namespace opposite_of_2023_l1890_189016

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l1890_189016


namespace emma_investment_l1890_189068

-- Define the basic problem parameters
def P : ℝ := 2500
def r : ℝ := 0.04
def n : ℕ := 21
def expected_amount : ℝ := 6101.50

-- Define the compound interest formula result
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- State the theorem
theorem emma_investment : 
  compound_interest P r n = expected_amount := 
  sorry

end emma_investment_l1890_189068


namespace harry_to_sue_nuts_ratio_l1890_189005

-- Definitions based on conditions
def sue_nuts : ℕ := 48
def bill_nuts (harry_nuts : ℕ) : ℕ := 6 * harry_nuts
def total_nuts (harry_nuts : ℕ) : ℕ := bill_nuts harry_nuts + harry_nuts

-- Proving the ratio
theorem harry_to_sue_nuts_ratio (H : ℕ) (h1 : sue_nuts = 48) (h2 : bill_nuts H + H = 672) : H / sue_nuts = 2 :=
by
  sorry

end harry_to_sue_nuts_ratio_l1890_189005


namespace inequality_satisfied_equality_condition_l1890_189057

theorem inequality_satisfied (x y : ℝ) : x^2 + y^2 + 1 ≥ 2 * (x * y - x + y) :=
sorry

theorem equality_condition (x y : ℝ) : (x^2 + y^2 + 1 = 2 * (x * y - x + y)) ↔ (x = y - 1) :=
sorry

end inequality_satisfied_equality_condition_l1890_189057


namespace evaluate_fraction_l1890_189026

theorem evaluate_fraction : 
  ( (20 - 19) + (18 - 17) + (16 - 15) + (14 - 13) + (12 - 11) + (10 - 9) + (8 - 7) + (6 - 5) + (4 - 3) + (2 - 1) ) 
  / 
  ( (1 - 2) + (3 - 4) + (5 - 6) + (7 - 8) + (9 - 10) + (11 - 12) + 13 ) 
  = (10 / 7) := 
by
  -- proof skipped
  sorry

end evaluate_fraction_l1890_189026


namespace mrs_hilt_apples_l1890_189040

theorem mrs_hilt_apples (hours : ℕ := 3) (rate : ℕ := 5) : 
  (rate * hours) = 15 := 
by sorry

end mrs_hilt_apples_l1890_189040


namespace relationship_a_b_c_l1890_189004

theorem relationship_a_b_c (x y a b c : ℝ) (h1 : x + y = a)
  (h2 : x^2 + y^2 = b) (h3 : x^3 + y^3 = c) : a^3 - 3*a*b + 2*c = 0 := by
  sorry

end relationship_a_b_c_l1890_189004


namespace square_of_binomial_is_25_l1890_189000

theorem square_of_binomial_is_25 (a : ℝ)
  (h : ∃ b : ℝ, (4 * (x : ℝ) + b)^2 = 16 * x^2 + 40 * x + a) : a = 25 :=
sorry

end square_of_binomial_is_25_l1890_189000


namespace tan_C_over_tan_A_max_tan_B_l1890_189085

theorem tan_C_over_tan_A {A B C : ℝ} {a b c : ℝ} (h : a^2 + 2 * b^2 = c^2) :
  let tan_A := Real.tan A
  let tan_C := Real.tan C
  (Real.tan C / Real.tan A) = -3 :=
sorry

theorem max_tan_B {A B C : ℝ} {a b c : ℝ} (h : a^2 + 2 * b^2 = c^2) :
  let B := Real.arctan (Real.tan B)
  ∃ (x : ℝ), x = Real.tan B ∧ ∀ y, y = Real.tan B → y ≤ (Real.sqrt 3) / 3 :=
sorry

end tan_C_over_tan_A_max_tan_B_l1890_189085


namespace chemistry_textbook_weight_l1890_189010

theorem chemistry_textbook_weight (G C : ℝ) 
  (h1 : G = 0.625) 
  (h2 : C = G + 6.5) : 
  C = 7.125 := 
by 
  sorry

end chemistry_textbook_weight_l1890_189010


namespace problem_remainder_P2017_mod_1000_l1890_189011

def P (x : ℤ) : ℤ := x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

theorem problem_remainder_P2017_mod_1000 :
  (P 2017) % 1000 = 167 :=
by
  -- this proof examines \( P(2017) \) modulo 1000
  sorry

end problem_remainder_P2017_mod_1000_l1890_189011


namespace cost_effectiveness_order_l1890_189017

variables {cS cM cL qS qM qL : ℝ}
variables (h1 : cM = 2 * cS)
variables (h2 : qM = 0.7 * qL)
variables (h3 : qL = 3 * qS)
variables (h4 : cL = 1.2 * cM)

theorem cost_effectiveness_order :
  (cL / qL <= cM / qM) ∧ (cM / qM <= cS / qS) :=
by
  sorry

end cost_effectiveness_order_l1890_189017


namespace cubic_roots_a_b_third_root_l1890_189053

theorem cubic_roots_a_b_third_root (a b : ℝ) :
  (∀ x, x^3 + a * x^2 + b * x + 6 = 0 → (x = 2 ∨ x = 3 ∨ x = -1)) →
  a = -4 ∧ b = 1 :=
by
  intro h
  -- We're skipping the proof steps and focusing on definite the goal
  sorry

end cubic_roots_a_b_third_root_l1890_189053


namespace table_filling_impossible_l1890_189083

theorem table_filling_impossible :
  ∀ (table : Fin 5 → Fin 8 → Fin 10),
  (∀ digit : Fin 10, ∃ row_set : Finset (Fin 5), row_set.card = 4 ∧
    (∀ row : Fin 5, row ∈ row_set → ∃ col_set : Finset (Fin 8), col_set.card = 4 ∧
      (∀ col : Fin 8, col ∈ col_set → table row col = digit))) →
  False :=
by
  sorry

end table_filling_impossible_l1890_189083


namespace exponents_product_as_cube_l1890_189048

theorem exponents_product_as_cube :
  (3^12 * 3^3) = 243^3 :=
sorry

end exponents_product_as_cube_l1890_189048


namespace min_transport_cost_l1890_189033

-- Definitions based on conditions
def total_washing_machines : ℕ := 100
def typeA_max_count : ℕ := 4
def typeB_max_count : ℕ := 8
def typeA_cost : ℕ := 400
def typeA_capacity : ℕ := 20
def typeB_cost : ℕ := 300
def typeB_capacity : ℕ := 10

-- Minimum transportation cost calculation
def min_transportation_cost : ℕ :=
  let typeA_trucks_used := min typeA_max_count (total_washing_machines / typeA_capacity)
  let remaining_washing_machines := total_washing_machines - typeA_trucks_used * typeA_capacity
  let typeB_trucks_used := min typeB_max_count (remaining_washing_machines / typeB_capacity)
  typeA_trucks_used * typeA_cost + typeB_trucks_used * typeB_cost

-- Lean 4 statement to prove the minimum transportation cost
theorem min_transport_cost : min_transportation_cost = 2200 := by
  sorry

end min_transport_cost_l1890_189033


namespace number_of_sodas_bought_l1890_189076

theorem number_of_sodas_bought
  (sandwich_cost : ℝ)
  (num_sandwiches : ℝ)
  (soda_cost : ℝ)
  (total_cost : ℝ)
  (h1 : sandwich_cost = 3.49)
  (h2 : num_sandwiches = 2)
  (h3 : soda_cost = 0.87)
  (h4 : total_cost = 10.46) :
  (total_cost - num_sandwiches * sandwich_cost) / soda_cost = 4 := 
sorry

end number_of_sodas_bought_l1890_189076


namespace shorter_piece_length_l1890_189082

theorem shorter_piece_length (x : ℕ) (h1 : ∃ l : ℕ, x + l = 120 ∧ l = 2 * x + 15) : x = 35 :=
sorry

end shorter_piece_length_l1890_189082


namespace average_age_decrease_l1890_189019

theorem average_age_decrease (N : ℕ) (T : ℝ) 
  (h1 : T = 40 * N) 
  (h2 : ∀ new_average_age : ℝ, (T + 12 * 34) / (N + 12) = new_average_age → new_average_age = 34) :
  ∃ decrease : ℝ, decrease = 6 :=
by
  sorry

end average_age_decrease_l1890_189019


namespace max_value_of_quadratic_l1890_189063

theorem max_value_of_quadratic : 
  ∃ x : ℝ, (∃ M : ℝ, ∀ y : ℝ, (-3 * y^2 + 15 * y + 9 <= M)) ∧ M = 111 / 4 :=
by
  sorry

end max_value_of_quadratic_l1890_189063


namespace train_length_proof_l1890_189024

-- Define the conditions
def train_speed_kmph := 72
def platform_length_m := 290
def crossing_time_s := 26

-- Conversion factor
def kmph_to_mps := 5 / 18

-- Convert speed to m/s
def train_speed_mps := train_speed_kmph * kmph_to_mps

-- Distance covered by train while crossing the platform (in meters)
def distance_covered := train_speed_mps * crossing_time_s

-- Length of the train (in meters)
def train_length := distance_covered - platform_length_m

-- The theorem to be proved
theorem train_length_proof : train_length = 230 :=
by 
  -- proof would be placed here 
  sorry

end train_length_proof_l1890_189024


namespace outfit_count_l1890_189032

section OutfitProblem

-- Define the number of each type of shirts, pants, and hats
def num_red_shirts : ℕ := 7
def num_blue_shirts : ℕ := 5
def num_green_shirts : ℕ := 8

def num_pants : ℕ := 10

def num_green_hats : ℕ := 10
def num_red_hats : ℕ := 6
def num_blue_hats : ℕ := 7

-- The main theorem to prove the number of outfits where shirt and hat are not the same color
theorem outfit_count : 
  (num_red_shirts * num_pants * (num_green_hats + num_blue_hats) +
  num_blue_shirts * num_pants * (num_green_hats + num_red_hats) +
  num_green_shirts * num_pants * (num_red_hats + num_blue_hats)) = 3030 :=
  sorry

end OutfitProblem

end outfit_count_l1890_189032


namespace degree_of_resulting_poly_l1890_189001

-- Define the polynomials involved in the problem
noncomputable def poly_1 : Polynomial ℝ := 3 * Polynomial.X ^ 5 + 2 * Polynomial.X ^ 3 - Polynomial.X - 16
noncomputable def poly_2 : Polynomial ℝ := 4 * Polynomial.X ^ 11 - 8 * Polynomial.X ^ 8 + 6 * Polynomial.X ^ 5 + 35
noncomputable def poly_3 : Polynomial ℝ := (Polynomial.X ^ 2 + 4) ^ 8

-- Define the resulting polynomial
noncomputable def resulting_poly : Polynomial ℝ :=
  poly_1 * poly_2 - poly_3

-- The goal is to prove that the degree of the resulting polynomial is 16
theorem degree_of_resulting_poly : resulting_poly.degree = 16 := 
sorry

end degree_of_resulting_poly_l1890_189001


namespace xyz_abs_eq_one_l1890_189034

theorem xyz_abs_eq_one (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) (h4 : x ≠ y) (h5 : y ≠ z) (h6 : z ≠ x)
  (cond : x + 1/y = y + 1/z ∧ y + 1/z = z + 1/x + 1) : |x * y * z| = 1 :=
sorry

end xyz_abs_eq_one_l1890_189034


namespace units_digit_is_seven_l1890_189038

-- Defining the structure of the three-digit number and its properties
def original_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c
def reversed_number (a b c : ℕ) : ℕ := 100 * c + 10 * b + a

def four_times_original (a b c : ℕ) : ℕ := 4 * original_number a b c
def subtract_reversed (a b c : ℕ) : ℕ := four_times_original a b c - reversed_number a b c

-- Theorem statement: Given the condition, what is the units digit of the result?
theorem units_digit_is_seven (a b c : ℕ) (h : a = c + 3) : (subtract_reversed a b c) % 10 = 7 :=
by
  sorry

end units_digit_is_seven_l1890_189038


namespace total_road_length_l1890_189044

theorem total_road_length (L : ℚ) : (1/3) * L + (2/5) * (2/3) * L = 135 → L = 225 := 
by
  intro h
  sorry

end total_road_length_l1890_189044


namespace root_in_interval_l1890_189087

noncomputable def f (x : ℝ) : ℝ := x + Real.log x - 3

theorem root_in_interval : ∃ m, f m = 0 ∧ 2 < m ∧ m < 3 :=
by
  sorry

end root_in_interval_l1890_189087


namespace factor_correct_l1890_189070

noncomputable def factor_expression (x : ℝ) : ℝ :=
  66 * x^6 - 231 * x^12

theorem factor_correct (x : ℝ) :
  factor_expression x = 33 * x^6 * (2 - 7 * x^6) :=
by 
  sorry

end factor_correct_l1890_189070


namespace no_such_positive_integer_l1890_189098

theorem no_such_positive_integer (n : ℕ) (d : ℕ → ℕ)
  (h₁ : ∃ d1 d2 d3 d4 d5, d 1 = d1 ∧ d 2 = d2 ∧ d 3 = d3 ∧ d 4 = d4 ∧ d 5 = d5) 
  (h₂ : 1 ≤ d 1 ∧ d 1 < d 2 ∧ d 2 < d 3 ∧ d 3 < d 4 ∧ d 4 < d 5)
  (h₃ : ∀ i, 1 ≤ i → i ≤ 5 → d i ∣ n)
  (h₄ : ∀ i, 1 ≤ i → i ≤ 5 → ∀ j, i ≠ j → d i ≠ d j)
  (h₅ : ∃ x, 1 + (d 2)^2 + (d 3)^2 + (d 4)^2 + (d 5)^2 = x^2) :
  false :=
sorry

end no_such_positive_integer_l1890_189098


namespace bicyclist_speed_first_100_km_l1890_189066

theorem bicyclist_speed_first_100_km (v : ℝ) :
  (16 = 400 / ((100 / v) + 20)) →
  v = 20 :=
by
  sorry

end bicyclist_speed_first_100_km_l1890_189066


namespace cricket_run_rate_l1890_189045

theorem cricket_run_rate (run_rate_first_10_overs : ℝ) (target_runs : ℝ) (first_overs : ℝ) (remaining_overs : ℝ):
  run_rate_first_10_overs = 6.2 → 
  target_runs = 282 →
  first_overs = 10 →
  remaining_overs = 40 →
  (target_runs - run_rate_first_10_overs * first_overs) / remaining_overs = 5.5 :=
by
  intros h1 h2 h3 h4
  -- Insert proof here
  sorry

end cricket_run_rate_l1890_189045


namespace camera_filter_kit_savings_l1890_189015

variable (kit_price : ℝ) (single_prices : List ℝ)
variable (correct_saving_amount : ℝ)

theorem camera_filter_kit_savings
    (h1 : kit_price = 145.75)
    (h2 : single_prices = [3 * 9.50, 2 * 15.30, 1 * 20.75, 2 * 25.80])
    (h3 : correct_saving_amount = -14.30) :
    (single_prices.sum - kit_price = correct_saving_amount) :=
by
  sorry

end camera_filter_kit_savings_l1890_189015


namespace num_people_visited_iceland_l1890_189067

noncomputable def total := 100
noncomputable def N := 43  -- Number of people who visited Norway
noncomputable def B := 61  -- Number of people who visited both Iceland and Norway
noncomputable def Neither := 63  -- Number of people who visited neither country
noncomputable def I : ℕ := 55  -- Number of people who visited Iceland (need to prove)

-- Lean statement to prove
theorem num_people_visited_iceland : I = total - Neither + B - N := by
  sorry

end num_people_visited_iceland_l1890_189067


namespace farmer_earns_from_runt_pig_l1890_189073

def average_bacon_per_pig : ℕ := 20
def price_per_pound : ℕ := 6
def runt_pig_bacon : ℕ := average_bacon_per_pig / 2
def total_money_made (bacon_pounds : ℕ) (price_per_pound : ℕ) : ℕ := bacon_pounds * price_per_pound

theorem farmer_earns_from_runt_pig :
  total_money_made runt_pig_bacon price_per_pound = 60 :=
sorry

end farmer_earns_from_runt_pig_l1890_189073


namespace possible_values_of_t_l1890_189022

theorem possible_values_of_t
  (theta : ℝ) 
  (x y t : ℝ) :
  x = Real.cos theta →
  y = Real.sin theta →
  t = (Real.sin theta) ^ 2 + (Real.cos theta) ^ 2 →
  x^2 + y^2 = 1 →
  t = 1 := by
  sorry

end possible_values_of_t_l1890_189022


namespace solution_set_f_pos_l1890_189047

open Set Function

variables (f : ℝ → ℝ)
variables (h_even : ∀ x : ℝ, f (-x) = f x)
variables (h_diff : ∀ x ≠ 0, DifferentiableAt ℝ f x)
variables (h_pos : ∀ x : ℝ, x > 0 → f x + x * (f' x) > 0)
variables (h_at_2 : f 2 = 0)

theorem solution_set_f_pos :
  {x : ℝ | f x > 0} = (Iio (-2)) ∪ (Ioi 2) :=
by 
  sorry

end solution_set_f_pos_l1890_189047


namespace sector_max_area_l1890_189055

-- Define the problem conditions
variables (α : ℝ) (R : ℝ)
variables (h_perimeter : 2 * R + R * α = 40)
variables (h_positive_radius : 0 < R)

-- State the theorem
theorem sector_max_area (h_alpha : α = 2) : 
  1/2 * α * (40 - 2 * R) * R = 100 := 
sorry

end sector_max_area_l1890_189055
