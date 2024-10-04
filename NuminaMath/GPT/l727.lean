import Mathlib

namespace in_first_quadrant_l727_727907

open complex

theorem in_first_quadrant (z : ℂ) (h : z = (2 - I) / (1 - 3 * I)) : 
  z.re > 0 ∧ z.im > 0 :=
by
  have h1 : z = (2 - I) / (1 - 3 * I) := h
  -- Continue from here to prove that z.re > 0 ∧ z.im > 0
  sorry

end in_first_quadrant_l727_727907


namespace regular_pentagon_l727_727562

structure Pentagon (α β γ : ℝ):
  -- Defining a structure which represents a pentagon with angles α, β and γ.
  (BC_eq_CD : BC = CD)
  (CD_eq_DE : CD = DE)
  (parallel_diagonals : ∀ d : Diagonal, parallel d.side)
  
theorem regular_pentagon (ABCDE : Pentagon) : 
  -- The theorem that says the given Pentagon ABCDE with the specified properties is a regular pentagon.
  is_regular_pentagon ABCDE := 
sorry

end regular_pentagon_l727_727562


namespace total_money_is_145_83_l727_727202

noncomputable def jackson_money : ℝ := 125

noncomputable def williams_money : ℝ := jackson_money / 6

noncomputable def total_money : ℝ := jackson_money + williams_money

theorem total_money_is_145_83 :
  total_money = 145.83 := by
sorry

end total_money_is_145_83_l727_727202


namespace prism_lateral_edges_correct_cone_axial_section_equilateral_l727_727674

/-- Defining the lateral edges of a prism and its properties --/
structure Prism (r : ℝ) :=
(lateral_edges_equal : ∀ (e1 e2 : ℝ), e1 = r ∧ e2 = r)

/-- Defining the axial section of a cone with properties of base radius and generatrix length --/
structure Cone (r : ℝ) :=
(base_radius : ℝ := r)
(generatrix_length : ℝ := 2 * r)
(is_equilateral : base_radius * 2 = generatrix_length)

theorem prism_lateral_edges_correct (r : ℝ) (P : Prism r) : 
 ∃ e, e = r ∧ ∀ e', e' = r :=
by {
  sorry
}

theorem cone_axial_section_equilateral (r : ℝ) (C : Cone r) : 
 base_radius * 2 = generatrix_length :=
by {
  sorry
}

end prism_lateral_edges_correct_cone_axial_section_equilateral_l727_727674


namespace points_on_line_eqdist_quadrants_l727_727402

theorem points_on_line_eqdist_quadrants :
  ∀ (x y : ℝ), 4 * x - 3 * y = 12 ∧ |x| = |y| → 
  (x > 0 ∧ y > 0 ∨ x > 0 ∧ y < 0) :=
by
  sorry

end points_on_line_eqdist_quadrants_l727_727402


namespace pow_log_exp_simplification_l727_727671

def exponent_logarithm_values : Real :=
  let base := 625
  let log_value := log 2 250
  let exponent_expr := base ^ log_value
  exponent_expr ^ (1/4)

theorem pow_log_exp_simplification :
  (625 ^ (Real.log 2 250)) ^ (1 / 4) = 250 := by
  sorry

end pow_log_exp_simplification_l727_727671


namespace border_collie_catches_ball_in_32_seconds_l727_727748

noncomputable def time_to_catch_ball (v_ball : ℕ) (t_ball : ℕ) (v_collie : ℕ) : ℕ := 
  (v_ball * t_ball) / v_collie

theorem border_collie_catches_ball_in_32_seconds :
  time_to_catch_ball 20 8 5 = 32 :=
by
  sorry

end border_collie_catches_ball_in_32_seconds_l727_727748


namespace south_pole_is_center_of_circumcircle_of_BIC_l727_727217

-- Definitions for the given conditions
variables {A B C S I : Point}

-- Define the conditions
def is_triangle (A B C : Point) : Prop := A ≠ B ∧ B ≠ C ∧ C ≠ A
def south_pole (A B C S : Point) : Prop := sorry -- Assuming the definition of South Pole
def incenter (A B C I : Point) : Prop := sorry -- Assuming the definition of incenter

-- Theorem statement
theorem south_pole_is_center_of_circumcircle_of_BIC
  (h_triangle : is_triangle A B C)
  (h_south_pole : south_pole A B C S)
  (h_incenter : incenter A B C I) :
  is_circumcenter S B I C :=
sorry

end south_pole_is_center_of_circumcircle_of_BIC_l727_727217


namespace total_candies_count_l727_727520

variable (purple_candies orange_candies yellow_candies : ℕ)

theorem total_candies_count
  (ratio_condition : purple_candies / orange_candies = 2 / 4 ∧ purple_candies / yellow_candies = 2 / 5)
  (yellow_candies_count : yellow_candies = 40) :
  purple_candies + orange_candies + yellow_candies = 88 :=
by
  sorry

end total_candies_count_l727_727520


namespace find_a_l727_727831

def P (a : ℕ) : Set ℕ := {x | 2 < x ∧ x < a}

theorem find_a (a : ℕ) :
  (∃ a, ∃ p: Set ℕ, p = P a ∧ p.card = 3) → a = 6 := sorry

end find_a_l727_727831


namespace part_a_part_b_part_c_l727_727341

namespace StudentSequences

-- Define the set of students
def students := ["Arnaldo", "Bernaldo", "Cernaldo", "Dernaldo", "Ernaldo"]

-- Total number of ways to arrange 5 students
theorem part_a : (Nat.factorial (List.length students)) = 120 :=
by
  have length_students : List.length students = 5 := rfl
  rw [length_students, Nat.factorial]
  norm_num

-- Number of sequences that are not in alphabetical order
theorem part_b (total_sequences : Nat.factorial (List.length students) = 120) : 120 - 1 = 119 :=
by
  rw total_sequences
  norm_num

-- Number of sequences where Arnaldo and Bernaldo are consecutive
theorem part_c : 
  let pairs := List.perm ["ArnaldoBernaldo", "Cernaldo", "Dernaldo", "Ernaldo"] ++ List.perm ["BernaldoArnaldo", "Cernaldo", "Dernaldo", "Ernaldo"]
  in List.length pairs = 48 :=
by
  let pairs_factorial := Nat.factorial 4 -- Number of ways to arrange 4 items
  have pairs_length : List.length (List.perm ["ArnaldoBernaldo", "Cernaldo", "Dernaldo", "Ernaldo"]) 
                    + List.length (List.perm ["BernaldoArnaldo", "Cernaldo", "Dernaldo", "Ernaldo"]) = 2 * pairs_factorial := by
    norm_num [List.perm]
  rw pairs_factorial
  norm_num
  apply pairs_length

end StudentSequences

end part_a_part_b_part_c_l727_727341


namespace cost_of_cheaper_feed_l727_727650

theorem cost_of_cheaper_feed (C : ℝ)
  (total_weight : ℝ) (weight_cheaper : ℝ) (price_expensive : ℝ) (total_value : ℝ) : 
  total_weight = 35 → 
  total_value = 0.36 * total_weight → 
  weight_cheaper = 17 → 
  price_expensive = 0.53 →
  (total_value = weight_cheaper * C + (total_weight - weight_cheaper) * price_expensive) →
  C = 0.18 := 
by
  sorry

end cost_of_cheaper_feed_l727_727650


namespace candidate_lost_by_2460_votes_l727_727700

noncomputable def total_votes : ℝ := 8199.999999999998
noncomputable def candidate_percentage : ℝ := 0.35
noncomputable def rival_percentage : ℝ := 1 - candidate_percentage
noncomputable def candidate_votes := candidate_percentage * total_votes
noncomputable def rival_votes := rival_percentage * total_votes
noncomputable def votes_lost_by := rival_votes - candidate_votes

theorem candidate_lost_by_2460_votes : votes_lost_by = 2460 := by
  sorry

end candidate_lost_by_2460_votes_l727_727700


namespace nine_x_plus_twenty_seven_y_l727_727634

theorem nine_x_plus_twenty_seven_y (x y : ℤ) (h : 17 * x + 51 * y = 102) : 9 * x + 27 * y = 54 := 
by sorry

end nine_x_plus_twenty_seven_y_l727_727634


namespace multiply_mixed_number_l727_727031

def mixed_to_improper_fraction (n : ℕ) (a b : ℕ) : ℚ :=
  n + a / b

def improper_to_mixed_number (q : ℚ) : ℕ × ℚ :=
  let n := q.natAbs
  let r := q.fract
  (n, r)

theorem multiply_mixed_number (x y n a b : ℕ) (h : y = mixed_to_improper_fraction n a b) :
  x * y = mixed_to_improper_fraction 65 4 5 :=
  sorry

end multiply_mixed_number_l727_727031


namespace range_of_function_l727_727863

theorem range_of_function (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  let y := (Real.cos x)^2 + (Real.sqrt 3 / 2) * (Real.sin (2 * x)) - 1 / 2
  in y ∈ Set.Ioc (-1 / 2) 1 :=
sorry

end range_of_function_l727_727863


namespace count_anagrams_YOANN_l727_727770

/-- A word consisting of 5 letters where the letter 'N' appears twice -/
def has_repeated_letters (word : list char) : Prop :=
  word.length = 5 ∧ 2 ≤ word.count 'N'

/-- The number of distinct anagrams of the word "YOANN" -/
theorem count_anagrams_YOANN :
  has_repeated_letters ['Y', 'O', 'A', 'N', 'N'] →
  let n := 5
  let r := 2
  let permutations := n.factorial / r.factorial
  permutations = 60 :=
by
  intros h
  let n := 5
  let r := 2
  let permutations := n.factorial / r.factorial
  have h1 : permutations = 60 := 
    by sorry -- Proof skipped
  exact h1

end count_anagrams_YOANN_l727_727770


namespace inequality_solution_l727_727774

theorem inequality_solution (x : ℝ) :
  (-4 ≤ x ∧ x < -3 / 2) ↔ (x / 4 ≤ 3 + x ∧ 3 + x < -3 * (1 + x)) :=
by
  sorry

end inequality_solution_l727_727774


namespace find_second_divisor_l727_727703

theorem find_second_divisor
  (N D : ℕ)
  (h1 : ∃ k : ℕ, N = 35 * k + 25)
  (h2 : ∃ m : ℕ, N = D * m + 4) :
  D = 21 :=
sorry

end find_second_divisor_l727_727703


namespace time_per_employee_updating_payroll_records_l727_727545

-- Define the conditions
def minutes_making_coffee : ℕ := 5
def minutes_per_employee_status_update : ℕ := 2
def num_employees : ℕ := 9
def total_morning_routine_minutes : ℕ := 50

-- Define the proof statement encapsulating the problem
theorem time_per_employee_updating_payroll_records :
  (total_morning_routine_minutes - (minutes_making_coffee + minutes_per_employee_status_update * num_employees)) / num_employees = 3 := by
  sorry

end time_per_employee_updating_payroll_records_l727_727545


namespace bathroom_area_l727_727699

def tile_size : ℝ := 0.5 -- Each tile is 0.5 feet

structure Section :=
  (width : ℕ)
  (length : ℕ)

def longer_section : Section := ⟨15, 25⟩
def alcove : Section := ⟨10, 8⟩

def area (s : Section) : ℝ := (s.width * tile_size) * (s.length * tile_size)

theorem bathroom_area :
  area longer_section + area alcove = 113.75 := by
  sorry

end bathroom_area_l727_727699


namespace gp_sum_l727_727407

theorem gp_sum (x : ℕ) (h : (30 + x) / (10 + x) = (60 + x) / (30 + x)) :
  x = 30 ∧ (10 + x) + (30 + x) + (60 + x) + (120 + x) = 340 :=
by {
  sorry
}

end gp_sum_l727_727407


namespace occurrences_of_odot_in_pattern_l727_727638

theorem occurrences_of_odot_in_pattern : 
  let pattern := ["A", "B", "⦿", "⦿"]
  (repeats := 13) -- Number of times the pattern repeats in first 52 symbols
  (full_repeats := pattern.count "⦿" * repeats)
  (53rd_symbol := pattern[52 % 4])

  full_repeats = 26 ∧ 53rd_symbol = "A" → 
  full_repeats = 26 :=
by
  -- Since the 53rd symbol does not add any additional "⦿"
  sorry

end occurrences_of_odot_in_pattern_l727_727638


namespace minimal_abs_diff_l727_727162

theorem minimal_abs_diff (a b : ℕ) (h : a * b - 6 * a + 5 * b = 373) : ∃ (a b : ℕ), abs (a - b) = 31 :=
by
  sorry

end minimal_abs_diff_l727_727162


namespace roots_int_and_coprime_l727_727945

-- Given conditions
variable (a : ℤ) (h_odd : Odd a)
variables (x₁ x₂ : ℂ)
variables (h_roots : x₁^2 + a * x₁ - 1 = 0) (h_roots2 : x₂^2 + a * x₂ - 1 = 0)

-- The statement to be proved
theorem roots_int_and_coprime (n : ℕ) : (x₁^n + x₂^n).re ∈ ℤ ∧ (x₁^(n+1) + x₂^(n+1)).re ∈ ℤ ∧ Int.gcd ( (x₁^n + x₂^n).re.natAbs ) ((x₁^(n+1) + x₂^(n+1)).re.natAbs) = 1 :=
  sorry

end roots_int_and_coprime_l727_727945


namespace n_pow_8_minus_1_divisible_by_480_l727_727265

theorem n_pow_8_minus_1_divisible_by_480 (n : ℤ) (h1 : ¬ (2 ∣ n)) (h2 : ¬ (3 ∣ n)) (h3 : ¬ (5 ∣ n)) : 
  480 ∣ (n^8 - 1) := 
sorry

end n_pow_8_minus_1_divisible_by_480_l727_727265


namespace inequality_proof_l727_727557

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    ((b + c - a)^2) / (a^2 + (b + c)^2) + ((c + a - b)^2) / (b^2 + (c + a)^2) + ((a + b - c)^2) / (c^2 + (a + b)^2) ≥ 3 / 5 :=
  sorry

end inequality_proof_l727_727557


namespace period_length_div_phi_l727_727226

def period_length (m : ℕ) : ℕ := orderOf 10 m

theorem period_length_div_phi {m : ℕ} (h : Nat.gcd m 10 = 1) : 
  period_length m ∣ Nat.totient m := 
by
  sorry

end period_length_div_phi_l727_727226


namespace boat_speed_in_still_water_l727_727529

-- Boat's speed in still water in km/hr
variable (B S : ℝ)

-- Conditions given for the boat's speed along and against the stream
axiom cond1 : B + S = 11
axiom cond2 : B - S = 5

-- Prove that the speed of the boat in still water is 8 km/hr
theorem boat_speed_in_still_water : B = 8 :=
by
  sorry

end boat_speed_in_still_water_l727_727529


namespace third_number_drawn_l727_727368

variable (students : Fin 1000)
variable (sample_size : ℕ)
variable (first_part : Set (Fin 1000))
variable (initial_draw : Fin 20)

-- Define the interval for systematic sampling
def sample_interval : ℕ := 20

-- First part includes numbers 0001 to 0020
def is_in_first_part (n : Fin 1000) : Prop := n.val < 20

-- Define the first part as the set of numbers 0001 to 0020
def first_part_set : Set (Fin 1000) := {n | is_in_first_part n}

-- Function to get the nth drawn number in systematic sampling
def nth_drawn_number (n : ℕ) (start : Fin 1000) : Fin 1000 :=
  ⟨(start.val + n * sample_interval) % 1000, sorry⟩

-- Conditions
axiom initial_draw_in_first_part : initial_draw ∈ first_part_set

-- Main theorem: the 3rd number drawn is 0055
theorem third_number_drawn (h_draw : initial_draw = 15) : nth_drawn_number 2 initial_draw = 55 :=
by
  sorry

end third_number_drawn_l727_727368


namespace perpendicular_to_bisector_of_beta_lies_on_FG_l727_727199

universe u

variables {α : Type u} [linear_ordered_ring α]
variables {A B C F G T : Point α}
variables {β : Angle α}

def incircle_touches_sides (ABC : Triangle α) (F G : Point α) : Prop :=
  incircle ABC ▸ TouchesSide ABC BC F ∧ TouchesSide ABC AB G

def perpendicular_from_A_to_bisector_of_beta (ABC : Triangle α) (A : Point α) : Geometry.Line α :=
  perpendicular A (angle_bisector ABC β)

theorem perpendicular_to_bisector_of_beta_lies_on_FG
  (ABC : Triangle α)
  (h1 : incircle_touches_sides ABC F G)
  (h2 : perpendicular_from_A_to_bisector_of_beta ABC A = T) :
  LiesOn T (line_through F G) :=
sorry

end perpendicular_to_bisector_of_beta_lies_on_FG_l727_727199


namespace range_of_x8_l727_727459

theorem range_of_x8 (x : ℕ → ℝ) (h1 : 0 ≤ x 1 ∧ x 1 ≤ x 2)
  (h_recurrence : ∀ n ≥ 1, x (n+2) = x (n+1) + x n)
  (h_x7 : 1 ≤ x 7 ∧ x 7 ≤ 2) : 
  (21/13 : ℝ) ≤ x 8 ∧ x 8 ≤ (13/4) :=
sorry

end range_of_x8_l727_727459


namespace largest_symmetric_polygon_area_in_triangle_l727_727251

theorem largest_symmetric_polygon_area_in_triangle (T : Triangle) :
  ∃ P : Polygon, is_centrally_symmetric P ∧ inscribed P T ∧ area P = (2 / 3) * area T := sorry

end largest_symmetric_polygon_area_in_triangle_l727_727251


namespace sum_of_15th_and_16th_sets_l727_727488

def set_begin (n : ℕ) : ℕ := 1 + (n * (n - 1)) / 2
def set_end (n : ℕ) : ℕ := set_begin(n) + n - 1
def set_sum (n : ℕ) : ℕ := n * (set_begin(n) + set_end(n)) / 2

theorem sum_of_15th_and_16th_sets : 
  set_sum 15 + set_sum 16 = 3751 := by
  sorry

end sum_of_15th_and_16th_sets_l727_727488


namespace NEMA_is_square_l727_727198

open EuclideanGeometry

structure RightTriangle (A B C : Point) : Prop :=
  (angle_A : angle A B C = 90)

structure Altitude (A D B C : Point) : Prop :=
  (foot : is_foot A D B C)

structure AngleBisector (A E B C : Point) : Prop :=
  (is_angle_bisector : is_angle_bisector A E B C)

structure Square (N E M A : Point) : Prop :=
  (ang_right_NEM : angle N E M = 90)
  (ang_right_EMA : angle E M A = 90)
  (ang_right_MAN : angle M A N = 90)
  (ang_right_ANE : angle A N E = 90)
  (side_NE_eq_EM : dist N E = dist E M)
  (side_EM_eq_MA : dist E M = dist M A)
  (side_MA_eq_AN : dist M A = dist A N)
  (side_AN_eq_NE : dist A N = dist N E)

theorem NEMA_is_square (A B C D E M N : Point)
  (h_triangle : RightTriangle A B C)
  (h_altitude : Altitude A D B C)
  (h_bisector_AE : AngleBisector A E B C)
  (h_bisector_M : AngleBisector B M D A)
  (h_bisector_N : AngleBisector C N D A) :
  Square N E M A :=
sorry

end NEMA_is_square_l727_727198


namespace sqrt_div_four_pow_four_l727_727318

theorem sqrt_div_four_pow_four :
  (sqrt (4^4 + 4^4 + 4^4) / 2 = 8 * sqrt 3) :=
by sorry

end sqrt_div_four_pow_four_l727_727318


namespace alexandra_magazines_l727_727732

theorem alexandra_magazines :
  let friday_magazines := 8
  let saturday_magazines := 12
  let sunday_magazines := 4 * friday_magazines
  let dog_chewed_magazines := 4
  let total_magazines_before_dog := friday_magazines + saturday_magazines + sunday_magazines
  let total_magazines_now := total_magazines_before_dog - dog_chewed_magazines
  total_magazines_now = 48 := by
  sorry

end alexandra_magazines_l727_727732


namespace sqrt_condition_iff_l727_727624

theorem sqrt_condition_iff (x : ℝ) : (∃ y : ℝ, y = (2 * x + 3) ∧ (0 ≤ y)) ↔ (x ≥ -3 / 2) :=
by sorry

end sqrt_condition_iff_l727_727624


namespace slope_range_l727_727276

def ellipse : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧  (x^2 / 4 + y^2 / 3 = 1)}

def A1 : ℝ × ℝ := (-2, 0)
def A2 : ℝ × ℝ := (2, 0)

def slope (P Q : ℝ × ℝ) (h : P ≠ Q) : ℝ := (P.2 - Q.2) / (P.1 - Q.1)

theorem slope_range {x y : ℝ} (hx : x ≠ 2) (hx_neg : x ≠ -2)
  (P_on_ellipse : (x, y) ∈ ellipse)
  (k_PA2_range : -2 ≤ slope (x, y) A2 hx ∧ slope (x, y) A2 hx ≤ -1) :
  (3 / 8 ≤ slope (x, y) A1 hx_neg ∧ slope (x, y) A1 hx_neg ≤ 3 / 4) :=
sorry

end slope_range_l727_727276


namespace total_frogs_each_species_l727_727209

noncomputable def frogs_species_A_LC (ll_A : ℕ) := ll_A - (20 * ll_A / 100)
noncomputable def frogs_species_A_EL (ll_A : ℕ) := ll_A + (30 * ll_A / 100)
noncomputable def frogs_species_B_LC (ll_B : ℕ) := ll_B + (10 * ll_B / 100)
noncomputable def frogs_species_B_EL (ll_B : ℕ) := 2 * ll_B
noncomputable def frogs_species_C_LC (ll_C : ℕ) := ll_C
noncomputable def frogs_species_C_EL (ll_C : ℕ) := ll_C + (50 * ll_C / 100)

theorem total_frogs_each_species (ll_A ll_B ll_C : ℕ) : 
  let total_A := frogs_species_A_LC ll_A + ll_A + frogs_species_A_EL ll_A,
      total_B := frogs_species_B_LC ll_B + ll_B + frogs_species_B_EL ll_B,
      total_C := frogs_species_C_LC ll_C + ll_C + frogs_species_C_EL ll_C in
  ll_A = 45 ∧ ll_B = 35 ∧ ll_C = 25 →
  total_A = 140 ∧ total_B = 144 ∧ total_C = 88 :=
by
  intros,
  sorry

end total_frogs_each_species_l727_727209


namespace incenter_of_triangle_l727_727042

theorem incenter_of_triangle
  {G G₁ G₂ : Circle}
  {A B C W : Point} 
  (h_tangent_1 : externally_tangent G₁ G₂ W)
  (h_tangent_2 : internally_tangent G₁ G W)
  (h_tangent_3 : internally_tangent G₂ G W)
  (h_points_on_circle : A ∈ G ∧ B ∈ G ∧ C ∈ G)
  (h_tangent_BC : common_tangent G₁ G₂ (Line.mk B C))
  (h_transverse_tangent_WA : transverse_tangent G₁ G₂ W (Line.mk W A))
  (h_same_side : same_side W A (Line.mk B C)) :
  incenter_of_triangle W A B C :=
sorry

end incenter_of_triangle_l727_727042


namespace seq_geometric_seq_general_formulas_no_arithmetic_subseq_l727_727219

open Function

-- Given conditions
variable {a : ℕ → ℤ} {b : ℕ → ℤ} {c : ℕ → ℤ} (d q : ℤ) (hnonzero : q ≠ 1)

-- Definitions of the sequences
def an_arithmetic (d : ℤ) (a : ℕ → ℤ) : Prop := 
  ∀ n, a (n + 1) = a n + d

def bn_geometric (q : ℤ) (b : ℕ → ℤ) : Prop := 
  ∀ n, b (n + 1) = b n * q

def cn (a b : ℕ → ℤ) (c : ℕ → ℤ) : Prop :=
  ∀ n, c n = a n + b n

-- Theorem to be proved
theorem seq_geometric_seq (d q : ℤ) (hnonzero : q ≠ 1) {a b c : ℕ → ℤ}
  (h1 : an_arithmetic d a) (h2 : bn_geometric q b) (h3 : cn a b c) :
  (∀ n, (c (n + 1) - c n - d) = b n * (q - 1)) :=
sorry

-- General formula for sequences
theorem general_formulas {a b c : ℕ → ℤ} :
  (a 1 = 1) ∧ (∀ n, a (n + 1) = a n + 3) ∧
  (b 1 = 3) ∧ (∀ n, b (n + 1) = b n * 2) :=
sorry

-- Non-existence of set A
theorem no_arithmetic_subseq (c : ℕ → ℤ) :
  (c 0 = 4) ∧ (c 1 = 10) ∧ (c 2 = 19) ∧ (c 3 = 34) →
  ¬ ∃ (A : Finset ℕ), A.card ≥ 4 ∧ 
        ∃ n1 n2 n3 n4, 
          n1 < n2 < n3 < n4 ∧ 
          ∀ {i < j < k < l}, 2 * c j = c i + c k :=
sorry

end seq_geometric_seq_general_formulas_no_arithmetic_subseq_l727_727219


namespace james_initial_stickers_l727_727203

theorem james_initial_stickers (got : ℕ) (total : ℕ) (initial : ℕ) (h1 : got = 22) (h2 : total = 61) (h3 : total = initial + got) : initial = 39 :=
by
  have h : 61 = initial + 22 := by rw [h2, h1, h3]
  sorry

end james_initial_stickers_l727_727203


namespace population_growth_1990_to_2020_l727_727187

theorem population_growth_1990_to_2020 (x : ℕ)
  (h₁ : ∃ n : ℕ, x = n^2)
  (h₂ : ∃ m : ℕ, x + 180 = m^2 - 16)
  (h₃ : ∃ p : ℕ, x + 360 = p^2)
  (h₄ : ∃ q : ℕ, x + 540 = q^2) :
  let initial_population := x^2
  let final_population := x^2 + 540
  let percent_increase := (final_population - initial_population) * 100 / initial_population in
  percent_increase = 23 := by sorry

end population_growth_1990_to_2020_l727_727187


namespace surface_area_ratio_volume_ratio_l727_727472

-- Define the conditions specified in the problem.
variables (r : ℝ) (h : ℝ) (d : ℝ)
-- Diameter of the sphere is equal to diameter of the base and height of the cylinder.
-- The height and diameter are defined in terms of the radius.
def diameter : ℝ := 2 * r
def height := 2 * r

-- The surface area of the sphere.
def S_sphere : ℝ := 4 * π * r^2

-- The surface area of the cylinder.
def S_cylinder : ℝ := 2 * π * r^2 + 2 * π * r * height

-- The volume of the sphere.
def V_sphere : ℝ := (4 / 3) * π * r^3

-- The volume of the cylinder.
def V_cylinder : ℝ := π * r^2 * height

-- Ratio of the surface area of the sphere to the surface area of the cylinder.
theorem surface_area_ratio : 2 * S_sphere = 3 * S_cylinder := by
  sorry

-- Ratio of the volume of the sphere to the volume of the cylinder.
theorem volume_ratio : 2 * V_sphere = 3 * V_cylinder := by
  sorry

-- Given conditions: diameter = 2 * r, height = 2 * r
example (r : ℝ) : 2 * (4 * π * r^2) = 3 * (2 * π * r^2 + 4 * π * r^2) :=
by sorry

example (r : ℝ) : 2 * ((4 / 3) * π * r^3) = 3 * (2 * π * r^3) :=
by sorry

end surface_area_ratio_volume_ratio_l727_727472


namespace number_of_correct_props_equals_one_l727_727848

def is_line (x : Type) : Prop := -- placeholder for the line type
  ∀ l : x, true

def is_plane (x : Type) : Prop := -- placeholder for the plane type
  ∀ p : x, true

variable {L : Type} {P : Type}
variable [is_line L] [is_plane P]

-- Define lines and planes
variable (a b l : L)
variable (alpha beta gamma : P)

-- Define perpendicular relationships
def perp (x : L) (y : P) : Prop := -- placeholder for perpendicularity relation
  true

-- Define parallel relationships
def parallel (x : L) (y : P) : Prop := -- placeholder for parallel relation
  true

noncomputable def number_of_correct_props : Nat := 
  if (parallel a alpha) && (perp a beta) && (perp alpha beta) then 0 else
  if (parallel a alpha) && (perp a b) && (perp b alpha) then 0 else
  if (parallel a b) && (perp l alpha) && (perp l b) then 1 else
  if (perp alpha gamma) && (perp beta gamma) && (parallel alpha beta) then 0 else 1

theorem number_of_correct_props_equals_one : 
  number_of_correct_props a b l alpha beta gamma = 1 := by
  -- Proof will be inserted here
  sorry

end number_of_correct_props_equals_one_l727_727848


namespace maximal_element_count_of_T_l727_727724

theorem maximal_element_count_of_T {T : Finset ℕ}
  (nonempty_T : T.nonempty)
  (distinct_elements : ∀ x ∈ T, ∀ y ∈ T, x ≠ y → (∀ a ∈ T.erase x, a = y → ¬y ∈ T))
  (mean_is_integer : ∀ y ∈ T, (T.sum - y) % (T.card - 1) = 0)
  (contains_one : 1 ∈ T)
  (largest_element : T.max' nonempty_T = 1764) :
  T.card ≤ 42 :=
sorry

end maximal_element_count_of_T_l727_727724


namespace infinite_points_with_integer_distances_lie_on_line_l727_727979

theorem infinite_points_with_integer_distances_lie_on_line
    (S : Set (EuclideanSpace ℝ 2))
    (H1 : Infinite S)
    (H2 : ∀ {X Y : EuclideanSpace ℝ 2}, X ∈ S ∧ Y ∈ S → (dist X Y) ∈ Int) :
    ∃ L : AffineSubspace ℝ (EuclideanSpace ℝ 2), ∀ X ∈ S, X ∈ L ∧ (AffineDimension L = 1) :=
by
  sorry

end infinite_points_with_integer_distances_lie_on_line_l727_727979


namespace A_is_werewolf_l727_727917

def forest_dwellers := Type
def is_werewolf (dweller : forest_dwellers) : Prop := sorry
def is_knight (dweller : forest_dwellers) : Prop := sorry
def is_liar (dweller : forest_dwellers) : Prop := ¬is_knight dweller

variable (A B C : forest_dwellers)

axiom werewolf_knight_unique : (is_werewolf A ∧ is_knight A) ∨ (is_werewolf B ∧ is_knight B) ∨ (is_werewolf C ∧ is_knight C)
axiom one_werewolf : (is_werewolf A ∧ is_liar B ∧ is_liar C) ∨ (is_werewolf B ∧ is_liar A ∧ is_liar C) ∨ (is_werewolf C ∧ is_liar A ∧ is_liar B)
axiom B_statement : is_knight B → is_werewolf C

theorem A_is_werewolf : is_werewolf A :=
by 
  have h_wk : (is_werewolf A ∧ is_knight A) ∨ (is_werewolf B ∧ is_knight B) ∨ (is_werewolf C ∧ is_knight C) := werewolf_knight_unique
  have h_ow : (is_werewolf A ∧ is_liar B ∧ is_liar C) ∨ (is_werewolf B ∧ is_liar A ∧ is_liar C) ∨ (is_werewolf C ∧ is_liar A ∧ is_liar B) := one_werewolf
  have h_bs : is_knight B → is_werewolf C := B_statement
  -- Further logical deductions to prove A is the werewolf go here.
  sorry

end A_is_werewolf_l727_727917


namespace mul_mixed_number_l727_727010

theorem mul_mixed_number (a b : ℝ) (c : ℚ) (h : c = 9 + 2/5) : 
  7 * c = (65 + 4/5 : ℝ) :=
by 
  simp [h, mul_add, mul_div_cancel_left, rat.cast_add, rat.cast_mul, rat.cast_one]
  sorry

end mul_mixed_number_l727_727010


namespace standard_equation_of_ellipse_max_area_triangle_l727_727840

variables {a b c : ℝ} (h_a_gt_b : a > b > 0)
          (h_minor_axis : b = √3)
          (h_eccentricity : c = a / 2)
          (h_relation : a^2 = b^2 + c^2)

open Real

theorem standard_equation_of_ellipse (C : ℝ → ℝ → Prop) :
  (C x y = (x^2 / 4 + y^2 / 3 = 1)) :=
by
  sorry

theorem max_area_triangle (F₁ F₂ : ℝ × ℝ) (A B : ℝ × ℝ)
  (h_foci : F₁.1 < F₂.1) (h_c : c = 1)
  (h_intersects : ∃ l : ℝ → ℝ, line l ∧ ∀ P ∈ [A, B], C P.1 P.2) :
  (maximum_area (F₁, A, B) = 3) :=
by
  sorry

end standard_equation_of_ellipse_max_area_triangle_l727_727840


namespace trains_clear_time_l727_727684

noncomputable def time_to_clear (length_train1 length_train2 speed_train1 speed_train2 : ℕ) : ℝ :=
  (length_train1 + length_train2) / ((speed_train1 + speed_train2) * 1000 / 3600)

theorem trains_clear_time :
  time_to_clear 121 153 80 65 = 6.803 :=
by
  -- This is a placeholder for the proof
  sorry

end trains_clear_time_l727_727684


namespace bags_of_apples_Dallas_l727_727771

-- Definitions related to the conditions
variables (A : ℕ) (apples_Austin pears_Dallas pears_Austin total_Austin: ℕ)

-- Establishing the conditions
def conditions : Prop :=
  (apples_Austin = A + 6) ∧
  (pears_Dallas = 9) ∧
  (pears_Austin = pears_Dallas - 5) ∧
  (total_Austin = apples_Austin + pears_Austin) ∧
  (total_Austin = 24)

-- Stating the theorem (proof) we need
theorem bags_of_apples_Dallas : conditions A apples_Austin pears_Dallas pears_Austin total_Austin → A = 14 :=
begin
  sorry
end

end bags_of_apples_Dallas_l727_727771


namespace min_diff_x1_x2_l727_727576

def f (x : ℝ) : ℝ := Real.sin ((Real.pi / 2) * x + Real.pi / 3)

theorem min_diff_x1_x2 : ∃ x1 x2 : ℝ, (∀ x : ℝ, f x1 ≤ f x ∧ f x ≤ f x2) ∧ |x1 - x2| = 2 := 
sorry

end min_diff_x1_x2_l727_727576


namespace sum_of_repeating_decimals_l727_727036

-- Defining the given repeating decimals as fractions
def rep_decimal1 : ℚ := 2 / 9
def rep_decimal2 : ℚ := 2 / 99
def rep_decimal3 : ℚ := 2 / 9999

-- Stating the theorem to prove the given sum equals the correct answer
theorem sum_of_repeating_decimals :
  rep_decimal1 + rep_decimal2 + rep_decimal3 = 224422 / 9999 :=
by
  sorry

end sum_of_repeating_decimals_l727_727036


namespace sum_of_first_6_terms_l727_727489

noncomputable def a_n (n : ℕ) : ℚ :=
if n = 0 then 0 else (1 / 2)^(n-1)

theorem sum_of_first_6_terms : 
  (∑ i in Finset.range 6, a_n (i + 1)) = 63 / 32 :=
by
  sorry

end sum_of_first_6_terms_l727_727489


namespace standard_ellipse_equation_slope_value_dot_product_constant_l727_727124

-- Define the given conditions
def is_ellipse (a b : ℝ) : Prop := a > b ∧ b > 0 ∧ ∀ (x y : ℝ), (x^2)/(a^2) + (y^2)/(b^2) = 1
def major_minor_axis_relation (a b : ℝ) : Prop := a = sqrt 3 * b
def triangle_area_condition (a b c : ℝ) : Prop := (1/2) * b * 2 * c = (5 * sqrt 2) / 3

-- Prove the standard equation of the ellipse given the conditions
theorem standard_ellipse_equation (a b c : ℝ) (h1 : is_ellipse a b) 
  (h2 : major_minor_axis_relation a b) (h3 : triangle_area_condition a b c) : 
  (a^2 = 5) ∧ (b^2 = 5/3) :=
sorry

-- Prove the value of slope k given the conditions
theorem slope_value (a b c k : ℝ) (h1 : is_ellipse a b) 
  (h2 : major_minor_axis_relation a b) (h3 : triangle_area_condition a b c) 
  (midpoint_condition : (x -> exists A B : ℝ, y = k (x + 1) ∧ x = -1/2)) : 
  k = sqrt 3 / 3 ∨ k = - sqrt 3 / 3 :=
sorry

-- Prove that the dot product is constant
theorem dot_product_constant (a b c k x₁ x₂ y₁ y₂ : ℝ) (h1 : is_ellipse a b) 
  (h2 : major_minor_axis_relation a b) (h3 : triangle_area_condition a b c) 
  (h_slope : k = sqrt 3 / 3 ∨ k = - sqrt 3 / 3) (h_midpoint : x₁ + x₂ = -1) 
  (point_M : x₁ + y₁ = -7/3) : 
  (x₁ + 7/3) * (x₂ + 7/3) + y₁ * y₂ = 4/9 :=
sorry

end standard_ellipse_equation_slope_value_dot_product_constant_l727_727124


namespace calculate_product_l727_727002

theorem calculate_product :
  7 * (9 + 2/5) = 65 + 4/5 := 
by
  sorry

end calculate_product_l727_727002


namespace max_possible_value_of_s_l727_727568

noncomputable def max_s_value (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y)
  (h_s : ∀ s : ℝ, s = min (min x (y + 1 / x)) (1 / y) → True) : ℝ := 
sqrt 2

theorem max_possible_value_of_s (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y)
  (h_s_min : ∀ s, s = min (min x (y + 1 / x)) (1 / y)) : 
  (∃ s : ℝ, s = sqrt 2 ∧ s = max_s_value x y h_pos_x h_pos_y h_s_min) :=
sorry

end max_possible_value_of_s_l727_727568


namespace side_length_smaller_than_one_l727_727925

theorem side_length_smaller_than_one (n : ℕ) (side_L : ℝ) 
  (h_large : side_L = 100)
  (h_ineq : n = 100000) 
  (squares : ℕ → ℝ × ℝ × ℝ) -- squares indexed by ℕ representing (x, y, side length)
  (h_diagonals : ∀ i j : ℕ, i ≠ j → 
    let (x1, y1, s1) := squares i;
        (x2, y2, s2) := squares j
    in ¬ ((x1 - x2)^2 + (y1 - y2)^2 < (s1/√2)^2 + (s2/√2)^2)) :
∃ k : ℕ, let (_, _, s) := squares k in s < 1 := 
sorry

end side_length_smaller_than_one_l727_727925


namespace total_bottles_l727_727269

/-- 
Prove the total number of new bottles that can eventually be made 
from 3125 initial plastic bottles if each new bottle can be recycled 
and 5 plastic bottles are needed to make one new bottle.
-/
theorem total_bottles (n k : ℕ) (h_n : n = 3125) (h_k : k = 5):
  -- The number of new bottles that can eventually be made is 156.
  let total_cycles := λ n k : ℕ, (n * (1 - (1/k)^5)) / (1 - 1/k) in
  total_cycles n k = 156 :=
by sorry

end total_bottles_l727_727269


namespace mary_initial_green_crayons_l727_727581

theorem mary_initial_green_crayons (g b left : ℕ) (given_green given_blue : ℕ) (initial_blue : ℕ) :
  given_green = 3 →
  given_blue = 1 →
  initial_blue = 8 →
  left = 9 →
  b = initial_blue →
  b - given_blue + g - given_green = left →
  g = 5 :=
by
  intros given_green_three given_blue_one initial_blue_eight left_nine b_is_initial_blue calc_green
  simp [given_green_three, given_blue_one, initial_blue_eight, left_nine, b_is_initial_blue] at calc_green
  linarith

end mary_initial_green_crayons_l727_727581


namespace soldiers_first_side_l727_727902

theorem soldiers_first_side (x : ℤ) (h1 : ∀ s1 : ℤ, s1 = 10)
                           (h2 : ∀ s2 : ℤ, s2 = 8)
                           (h3 : ∀ y : ℤ, y = x - 500)
                           (h4 : (10 * x + 8 * (x - 500)) = 68000) : x = 4000 :=
by
  -- Left blank for Lean to fill in the required proof steps
  sorry

end soldiers_first_side_l727_727902


namespace total_length_of_fence_l727_727721

theorem total_length_of_fence (x : ℝ) (h1 : 2 * x * x = 1250) : 2 * x + 2 * x = 100 :=
by
  sorry

end total_length_of_fence_l727_727721


namespace second_divisor_l727_727706

theorem second_divisor (N k D m : ℤ) (h1 : N = 35 * k + 25) (h2 : N = D * m + 4) : D = 17 := by
  -- Follow conditions from problem
  sorry

end second_divisor_l727_727706


namespace base7_digit_divisibility_l727_727445

-- Define base-7 digit integers
notation "digit" => Fin 7

-- Define conversion from base-7 to base-10 for the form 3dd6_7
def base7_to_base10 (d : digit) : ℤ := 3 * (7^3) + (d:ℤ) * (7^2) + (d:ℤ) * 7 + 6

-- Define the property of being divisible by 13
def is_divisible_by_13 (n : ℤ) : Prop := ∃ k : ℤ, n = 13 * k

-- Formalize the theorem
theorem base7_digit_divisibility (d : digit) :
  is_divisible_by_13 (base7_to_base10 d) ↔ d = 4 :=
sorry

end base7_digit_divisibility_l727_727445


namespace probability_five_heads_in_six_tosses_is_09375_l727_727711

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

noncomputable def probability_exact_heads (n k : ℕ) (p : ℝ) : ℝ :=
  binomial n k * (p^k) * ((1-p)^(n-k))
  
theorem probability_five_heads_in_six_tosses_is_09375 :
  probability_exact_heads 6 5 0.5 = 0.09375 :=
by
  sorry

end probability_five_heads_in_six_tosses_is_09375_l727_727711


namespace inverse_of_A_l727_727426

open Matrix

noncomputable def A : Matrix (Fin 2) (Fin 2) ℚ :=
  matrix.of ![![3, 4], ![-2, 9]]

noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  matrix.of ![![9/35, -4/35], ![2/35, 3/35]]

theorem inverse_of_A :
  A⁻¹ = A_inv := by
  sorry

end inverse_of_A_l727_727426


namespace tangent_min_length_l727_727172

noncomputable def circle_eq (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 8

noncomputable def line_eq (a b x y : ℝ) : Prop := 2 * a * x + b * y + 6 = 0

noncomputable def center := (-1 : ℝ, 2 : ℝ)

theorem tangent_min_length (a b : ℝ) 
  (h1 : line_eq a b (-1) 2) 
  (h2 : a = b + 3) :
  ∃ M : ℝ, M^2 = 10 :=
sorry

end tangent_min_length_l727_727172


namespace increasing_inverse_relation_l727_727628

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry -- This is the inverse function f^-1

theorem increasing_inverse_relation {a b c : ℝ} 
  (h_inc_f : ∀ x y, x < y → f x < f y)
  (h_inc_f_inv : ∀ x y, x < y → f_inv x < f_inv y)
  (h_f3 : f 3 = 0)
  (h_f2 : f 2 = a)
  (h_f_inv2 : f_inv 2 = b)
  (h_f_inv0 : f_inv 0 = c) :
  b > c ∧ c > a := sorry

end increasing_inverse_relation_l727_727628


namespace find_m_for_parallel_lines_l727_727632

noncomputable def parallel_lines_x_plus_1_plus_m_y_eq_2_minus_m_and_m_x_plus_2_y_plus_8_eq_0 (m : ℝ) : Prop :=
  let l1_slope := -(1 + m) / 1
  let l2_slope := -m / 2
  l1_slope = l2_slope

theorem find_m_for_parallel_lines :
  parallel_lines_x_plus_1_plus_m_y_eq_2_minus_m_and_m_x_plus_2_y_plus_8_eq_0 m →
  m = 1 :=
by
  intro h_parallel
  -- Here we would present the proof steps to show that m = 1 under the given conditions.
  sorry

end find_m_for_parallel_lines_l727_727632


namespace salary_relationship_l727_727681

variables (A B C : ℝ)

theorem salary_relationship :
  A + B + C = 10000 ∧ 0.10 * A + 0.15 * B = 0.20 * C →
  A = 6666.67 - (7 / 6) * B :=
begin
  intros h,
  cases h with h1 h2,
  have h3 : C = 10000 - A - B,
  { linarith, },
  rw h3 at h2,
  linarith,
end

end salary_relationship_l727_727681


namespace locus_of_P_is_a_circle_l727_727142

-- Definitions of the problem elements
variables {C C' Γ : Circle} -- Circles C, C', and variable circle Γ
variables {O O' A B M M' N N' P : Point} -- Points involved in the problem

-- Assume centers and intersection points
def centers_and_intersection (C C' : Circle) (O O' A B : Point) [center C = O] [center C' = O'] 
  [intersects_at C C' A] [intersects_at C C' B] : Prop := 
  true

-- The conditions in Lean 4
def problem_conditions (Γ : Circle) (A : Point) (M M' N N' P : Point) : Prop :=
  Γ ∋ A ∧ intersect(Γ, C) = M ∧ intersect(Γ, C') = M' ∧
  line_through A M ∩ C' = N' ∧ line_through A M' ∩ C = N ∧
  intersection (Circle (A, N, N')) Γ = P

-- The theorem stating the locus of P
theorem locus_of_P_is_a_circle 
  (C C' Γ : Circle) (O O' A B M M' N N' P : Point) 
  [centers_and_intersection C C' O O' A B]
  [problem_conditions Γ A M M' N N' P] : 
  ∃ center : Point, ∃ radius : Length, is_circle center radius ∧ centered_at center (midpoint O O') ∧ circle_through center radius A := 
sorry

end locus_of_P_is_a_circle_l727_727142


namespace unique_point_S_and_ratios_l727_727649

noncomputable def centroid (A B C : Point) : Point :=
  (A + B + C) / 3

theorem unique_point_S_and_ratios 
  (A1 A2 A3 : Point) 
  (A : ℕ → Point)
  (h1 : noncollinear A1 A2 A3)
  (h2 : ∀ n ≥ 4, A n = centroid (A (n-3)) (A (n-2)) (A (n-1))) :
  ∃ S : Point, (∀ n ≥ 4, lies_in_interior S (triangle (A (n-3)) (A (n-2)) (A (n-1)))) ∧
  (let T := line_intersection (line A1 A2) (line_through S A3) in 
  (ratio A1 T A2 = 2 / 1) ∧ (ratio T S A3 = 1 / 2)) :=
sorry

end unique_point_S_and_ratios_l727_727649


namespace series1_converges_everywhere_series2_converges_only_at_zero_series3_converges_between_minus1_and_1_series4_converges_between_half_and_three_halves_series5_converges_everywhere_l727_727420

def series1_convergence_region (x : ℝ) : Prop :=
  (∑' n:ℕ, x^(n+1) / (n+1)!) < ⊤

theorem series1_converges_everywhere : ∀ x : ℝ, series1_convergence_region x := sorry

def series2_convergence_region (x : ℝ) : Prop :=
  (∑' n:ℕ, n * (n!) * x ^ (n+1)) < ⊤

theorem series2_converges_only_at_zero : ∀ x : ℝ, series2_convergence_region x ↔ x = 0 := sorry

def series3_convergence_region (x : ℝ) : Prop :=
  (∑' n : ℕ, ((-1)^(n) * x^(n+1) / (n+1)^2)) < ⊤

theorem series3_converges_between_minus1_and_1 : ∀ x : ℝ, series3_convergence_region x ↔ -1 ≤ x ∧ x ≤ 1 := sorry

def series4_convergence_region (x : ℝ) : Prop :=
  (∑' n : ℕ, 2^n * (x - 1)^n) < ⊤

theorem series4_converges_between_half_and_three_halves : ∀ x : ℝ, series4_convergence_region x ↔ (1 / 2) < x ∧ x < (3 / 2) := sorry

def series5_convergence_region (x : ℝ) : Prop :=
  (∑' n : ℕ, x^(n+1) / (n+1)^(n+1)) < ⊤

theorem series5_converges_everywhere : ∀ x : ℝ, series5_convergence_region x := sorry

end series1_converges_everywhere_series2_converges_only_at_zero_series3_converges_between_minus1_and_1_series4_converges_between_half_and_three_halves_series5_converges_everywhere_l727_727420


namespace total_flowers_eaten_l727_727586

theorem total_flowers_eaten (bugs : ℕ) (flowers_per_bug : ℕ) (h_bugs : bugs = 3) (h_flowers_per_bug : flowers_per_bug = 2) :
  (bugs * flowers_per_bug) = 6 :=
by
  sorry

end total_flowers_eaten_l727_727586


namespace max_points_on_circle_at_distance_r_l727_727252

noncomputable def point_is_tangent_to_circle (P : Point) (C : Circle) (r : ℝ) : Prop :=
  ∃ d : ℝ, d = distance P C.center ∧ (d = r - C.radius ∨ d = r + C.radius)

noncomputable theory
open_locale classical

theorem max_points_on_circle_at_distance_r {P : Point} {C : Circle} (r : ℝ)
  (hr : r > 5) (P_outside_C : distance P C.center > C.radius) (C_radius : C.radius = 5) :
  ∃! x : ℕ, x = 1 :=
by
  sorry

end max_points_on_circle_at_distance_r_l727_727252


namespace count_valid_multiples_l727_727881

theorem count_valid_multiples :
  let valid_multiples := { x : ℤ | x >= 100 ∧ x <= 300 ∧ x % 35 = 0 ∧ x % 10 ≠ 0 } in
  valid_multiples.finite.to_finset.card = 3 :=
by sorry

end count_valid_multiples_l727_727881


namespace total_rainfall_2008_l727_727896

theorem total_rainfall_2008 (avg_2007 : ℝ) (diff : ℝ) (months_2008 : ℕ) :
  avg_2007 = 45.2 → diff = 3.5 → months_2008 = 12 → 
  let avg_2008 := avg_2007 + diff in
  let total_rainfall := months_2008 * avg_2008 in
  total_rainfall = 584.4 :=
by {
  intros h1 h2 h3,
  let avg_2008 := avg_2007 + diff,
  rw [h1, h2, h3],
  have h4 : avg_2008 = 45.2 + 3.5, by rw [h1, h2],
  rw h4,
  norm_num,
}

end total_rainfall_2008_l727_727896


namespace cos_diff_quadrant_four_l727_727466

theorem cos_diff_quadrant_four 
  (α : ℝ) 
  (h1 : cos α = 3 / 5) 
  (h2 : α ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi)) :
  cos (α - Real.pi / 4) = -Real.sqrt 2 / 10 :=
by
  sorry

end cos_diff_quadrant_four_l727_727466


namespace question_a_question_b_l727_727561

universe u
variable {α : Type u}

-- Given definitions for the problem setup
def triangle (A B C : α) := true -- Placeholder for triangle context.
def circumcircle (A B C : α) (Ω : α) := true -- Placeholder for circumcircle context.
def orthocenter (A B C H : α) := true -- Placeholder for orthocenter context.
def on_segment (P A B : α) := true -- Placeholder for points on segment.
def equal_lengths (A D E : α) := true -- Placeholder for equal segment lengths.
def parallel_lines (L1 L2 : α) := true -- Placeholder for parallel lines.
def intersects (L1 L2 : α) (P : α) := true -- Placeholder for intersection points.
def circumcircle_of_triangle (A D E ω : α) := true -- Placeholder for triangle circumcircles.

-- Placeholder definitions for lines through B and C parallel to DE meeting Ω again
def parallel_to_segment (A B C D E P Q Ω : α) := true

-- Placeholder definition for circle containing H to show ω passing through H
def circle_contains (ω H : α) := true

-- Definitions for intersection points on ω
def intersects_on_circle (L1 L2 : α) (P ω : α) := true

-- Theorem statements
theorem question_a (A B C Ω H D E P Q ω : α)
  [t : triangle A B C]
  [c : circumcircle A B C Ω]
  [h : orthocenter A B C H]
  [d : on_segment D A B]
  [e : on_segment E A C]
  [adl : equal_lengths A D E]
  [pp : parallel_to_segment A B C D E P Q Ω]
  [circ_tr : circumcircle_of_triangle A D E ω] :
  ∃ X, intersects_on_circle (line P E) (line Q D) X ω := 
sorry

theorem question_b (A B C Ω H D E P Q ω : α)
  [t : triangle A B C]
  [c : circumcircle A B C Ω]
  [h : orthocenter A B C H]
  [d : on_segment D A B]
  [e : on_segment E A C]
  [adl : equal_lengths A D E]
  [pp : parallel_to_segment A B C D E P Q Ω]
  [circ_tr : circumcircle_of_triangle A D E ω]
  [ch : circle_contains ω H] :
  ∃ K, intersects_on_circle (line P D) (line Q E) K ω := 
sorry

end question_a_question_b_l727_727561


namespace sum_of_integers_floor_relationship_l727_727953

theorem sum_of_integers_floor_relationship {p a b : ℕ} (hp : Nat.Prime p) (hp_odd : p % 2 = 1)
    (ha : 0 < a) (hb : 0 < b) (ha_lt_p : a < p) (hb_lt_p : b < p) :
    (a + b = p) ↔ ∀ n : ℕ, (0 < n ∧ n < p) →
    (Nat.floor((2 * a * n : ℝ) / p) + Nat.floor((2 * b * n : ℝ) / p)) % 2 = 1 :=
sorry

end sum_of_integers_floor_relationship_l727_727953


namespace box_volume_correct_l727_727960

-- Define the dimensions of the obelisk
def obelisk_height : ℕ := 15
def base_length : ℕ := 8
def base_width : ℕ := 10

-- Define the dimension and volume goal for the cube-shaped box
def box_side_length : ℕ := obelisk_height
def box_volume : ℕ := box_side_length ^ 3

-- The proof goal
theorem box_volume_correct : box_volume = 3375 := 
by sorry

end box_volume_correct_l727_727960


namespace acute_triangle_to_isosceles_triangles_l727_727591

theorem acute_triangle_to_isosceles_triangles (A B C D : Type) [nonempty A] [nonempty B] [nonempty C] [nonempty D]
  (triangle_ABC : ∀ (α β γ : ℝ), α + β + γ = 180 ∧ α < 90 ∧ β < 90 ∧ γ < 90)
  (angle_condition : ∀ (α : ℝ), ∃ (B C : A → ℝ), ∃ (D : A), 
                         ∃ (Ω₁ Ω₂ Ω₃ : ℝ), 
                         (Ω₁ = α) ∧ (Ω₂ = 2 * α) ∧ (Ω₃ = 180 - (Ω₁ + Ω₂)) ∧ 
                         (Ω₃ < 90) ∧ 
                         ∃ (BD BC DA : ℝ), (BD = BC) ∧ (BD = DA) ∧
                         triangle_ABC α Ω₁ Ω₂) :
  ∃ (acute_angle_triangle : ∃ (B D C : A), B ≠ C ∧ D ≠ A ∧ BD = BC),
  ∃ (isosceles_triangles : ∃ (triangle_BCD : B ≠ D ∧ B ≠ C ∧ D ≠ C ∧  angle_condition α),
                         ∃ (triangle_ABD : ∃ (BD = BC) ∧ ∃ (DA = BD) and angle_condition α)) :=
sorry

end acute_triangle_to_isosceles_triangles_l727_727591


namespace find_integers_l727_727083

noncomputable def S (n : ℕ) : ℚ := (Finset.range (n + 1)).sum (λ i, 1 / (i + 1 : ℚ))

noncomputable def T (n : ℕ) : ℚ := (Finset.range (n + 1)).sum (λ i, S i)

noncomputable def U (n : ℕ) : ℚ := (Finset.range (n + 1)).sum (λ i, T i / (i + 2 : ℚ))

theorem find_integers : 
  ∃ (a b c d : ℕ), 0 < a ∧ a < 1000000 ∧ 0 < b ∧ b < 1000000 ∧
                    0 < c ∧ c < 1000000 ∧ 0 < d ∧ d < 1000000 ∧
                    T 1988 = a * S 1989 - b ∧ 
                    U 1988 = c * S 1989 - d := by
  use 1989, 1988, 1990, 3978
  split
  · repeat { constructor }; norm_num
  · split
    · sorry
    · sorry

end find_integers_l727_727083


namespace gcd_12012_18018_l727_727812

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_12012_18018 : gcd 12012 18018 = 6006 := sorry

end gcd_12012_18018_l727_727812


namespace abs_m_minus_n_eq_five_l727_727228

theorem abs_m_minus_n_eq_five (m n : ℝ) (h₁ : m * n = 6) (h₂ : m + n = 7) : |m - n| = 5 :=
sorry

end abs_m_minus_n_eq_five_l727_727228


namespace coefficient_x3_expansion_l727_727623

open Finset -- To use binomial coefficients and summation

theorem coefficient_x3_expansion (x : ℝ) : 
  (2 + x) ^ 3 = 8 + 12 * x + 6 * x^2 + 1 * x^3 :=
by
  sorry

end coefficient_x3_expansion_l727_727623


namespace players_without_cautions_l727_727398

theorem players_without_cautions (Y N : ℕ) (h1 : Y + N = 11) (h2 : Y = 6) : N = 5 :=
by
  sorry

end players_without_cautions_l727_727398


namespace sin_minus_cos_sqrt_l727_727504

theorem sin_minus_cos_sqrt (θ : ℝ) (b : ℝ) (h₁ : 0 < θ ∧ θ < π / 2) (h₂ : Real.cos (2 * θ) = b) :
  Real.sin θ - Real.cos θ = Real.sqrt (1 - b) :=
sorry

end sin_minus_cos_sqrt_l727_727504


namespace square_nonnegative_for_rat_l727_727675

theorem square_nonnegative_for_rat (x : ℚ) : x^2 ≥ 0 :=
sorry

end square_nonnegative_for_rat_l727_727675


namespace planes_parallel_if_all_lines_parallel_l727_727737

theorem planes_parallel_if_all_lines_parallel (π₁ π₂ : Plane) (h : ∀ (l : Line), l ⊆ π₁ → parallel l π₂) : parallel π₁ π₂ :=
sorry

end planes_parallel_if_all_lines_parallel_l727_727737


namespace speed_difference_maya_kai_l727_727582

def minutes_to_hours (m : ℕ) : ℝ := m / 60

def speed (distance time : ℝ) : ℝ := distance / time

theorem speed_difference_maya_kai 
  (maya_distance : ℝ) (kai_distance : ℝ) (trav_time_minutes : ℕ) 
  (simultaneous_arrival : maya_distance > 0 ∧ kai_distance > 0 ∧ trav_time_minutes > 0):
  let kai_time_hours := minutes_to_hours trav_time_minutes in
  let maya_time_hours := minutes_to_hours trav_time_minutes in
  let kai_speed := speed kai_distance kai_time_hours in
  let maya_speed := speed maya_distance maya_time_hours in
  kai_speed - maya_speed = 20 := by
  sorry

end speed_difference_maya_kai_l727_727582


namespace smallest_angle_in_sector_sequence_l727_727931

theorem smallest_angle_in_sector_sequence :
  ∃ (a_1 : ℕ) (d : ℕ), (∀ n : ℕ, 1 ≤ n ∧ n ≤ 15 → (a_1 + (n - 1) * d : ℕ) ∈ ℕ) ∧
  (∑ n in (finset.range 15), (a_1 + n * d)) = 360 ∧
  (∀ a_b : ℕ, (∀ n : ℕ, 1 ≤ n ∧ n ≤ 15 → (a_b + (n - 1) * d : ℕ) ∈ ℕ) ∧
   ∑ n in (finset.range 15), (a_b + n * d) = 360 → a_1 ≤ a_b) :=
sorry

end smallest_angle_in_sector_sequence_l727_727931


namespace gcd_of_12012_18018_l727_727793

theorem gcd_of_12012_18018 : gcd 12012 18018 = 6006 :=
by
  -- Definitions for conditions
  have h1 : 12012 = 12 * 1001 := by
    sorry
  have h2 : 18018 = 18 * 1001 := by
    sorry
  have h3 : gcd 12 18 = 6 := by
    sorry
  -- Using the conditions to prove the main statement
  rw [h1, h2]
  rw [gcd_mul_right, gcd_mul_right]
  rw [gcd_comm 12 18, h3]
  rw [mul_comm 6 1001]
  sorry

end gcd_of_12012_18018_l727_727793


namespace similar_quadrilaterals_l727_727549

open EuclideanGeometry

variables {A B C D O : Point}

variable (S : Quadrilateral A B C D)
variable (O_inside_S : S.contains O)

def is_feet_of_perpendicular (P Q R : Point) (O : Point) (X : Point) : Prop :=
  right_angle (Line.mk O X) (Line.mk P Q)

variables
  (A1 B1 C1 D1 A2 B2 C2 D2 A3 B3 C3 D3 A4 B4 C4 D4 : Point)
  (H_A1 : is_feet_of_perpendicular A B O A1)
  (H_B1 : is_feet_of_perpendicular B C O B1)
  (H_C1 : is_feet_of_perpendicular C D O C1)
  (H_D1 : is_feet_of_perpendicular D A O D1)
  (H_A2 : is_feet_of_perpendicular A1 B1 O A2)
  (H_B2 : is_feet_of_perpendicular B1 C1 O B2)
  (H_C2 : is_feet_of_perpendicular C1 D1 O C2)
  (H_D2 : is_feet_of_perpendicular D1 A1 O D2)
  (H_A3 : is_feet_of_perpendicular A2 B2 O A3)
  (H_B3 : is_feet_of_perpendicular B2 C2 O B3)
  (H_C3 : is_feet_of_perpendicular C2 D2 O C3)
  (H_D3 : is_feet_of_perpendicular D2 A2 O D3)
  (H_A4 : is_feet_of_perpendicular A3 B3 O A4)
  (H_B4 : is_feet_of_perpendicular B3 C3 O B4)
  (H_C4 : is_feet_of_perpendicular C3 D3 O C4)
  (H_D4 : is_feet_of_perpendicular D3 A3 O D4)

theorem similar_quadrilaterals :
  similar_quadrilateral (Quadrilateral.mk A B C D) (Quadrilateral.mk A4 B4 C4 D4) :=
sorry

end similar_quadrilaterals_l727_727549


namespace cross_product_zero_l727_727884

variables (v w : ℝ × ℝ × ℝ)

theorem cross_product_zero (h : (v.1 - w.1) * (v.2 - w.2) - (v.2 - w.2) * (v.3 - w.3) = (7, -3, 6) ) :
  (2 • v + w) × (2 • v + w) = (0, 0, 0) :=
sorry

end cross_product_zero_l727_727884


namespace roger_has_more_candy_l727_727264

-- Defining the conditions
def sandra_bag1 : Nat := 6
def sandra_bag2 : Nat := 6
def roger_bag1 : Nat := 11
def roger_bag2 : Nat := 3

-- Calculating the total pieces of candy for Sandra and Roger
def total_sandra : Nat := sandra_bag1 + sandra_bag2
def total_roger : Nat := roger_bag1 + roger_bag2

-- Statement of the proof problem
theorem roger_has_more_candy : total_roger - total_sandra = 2 := by
  sorry

end roger_has_more_candy_l727_727264


namespace gcd_of_12012_18018_l727_727792

theorem gcd_of_12012_18018 : gcd 12012 18018 = 6006 :=
by
  -- Definitions for conditions
  have h1 : 12012 = 12 * 1001 := by
    sorry
  have h2 : 18018 = 18 * 1001 := by
    sorry
  have h3 : gcd 12 18 = 6 := by
    sorry
  -- Using the conditions to prove the main statement
  rw [h1, h2]
  rw [gcd_mul_right, gcd_mul_right]
  rw [gcd_comm 12 18, h3]
  rw [mul_comm 6 1001]
  sorry

end gcd_of_12012_18018_l727_727792


namespace price_of_75_cans_l727_727338

/-- The price of 75 cans of a certain brand of soda purchased in 24-can cases,
    given the regular price per can is $0.15 and a 10% discount is applied when
    purchased in 24-can cases, is $10.125.
-/
theorem price_of_75_cans (regular_price : ℝ) (discount : ℝ) (cases_needed : ℕ) (remaining_cans : ℕ) 
  (discounted_price : ℝ) (total_price : ℝ) :
  regular_price = 0.15 →
  discount = 0.10 →
  discounted_price = regular_price - (discount * regular_price) →
  cases_needed = 75 / 24 ∧ remaining_cans = 75 % 24 →
  total_price = (cases_needed * 24 + remaining_cans) * discounted_price →
  total_price = 10.125 :=
by
  sorry

end price_of_75_cans_l727_727338


namespace points_distance_inequality_l727_727563

open Real

theorem points_distance_inequality
  (n : ℕ)
  (points : Fin (n + 1) → ℝ × ℝ)
  (d : ℝ)
  (h_d : 0 < d)
  (h_dist : ∀ i j : Fin (n + 1), i ≠ j → dist (points i) (points j) ≥ d) :
  (∏ i in Finset.range n, dist (points 0) (points (i + 1))) > (d / 3) ^ n * sqrt ((nat.factorial (n + 1)).toReal) :=
sorry

end points_distance_inequality_l727_727563


namespace line_first_quadrant_l727_727475

theorem line_first_quadrant (a b c : ℝ) (h1 : ab > 0) (h2 : ∀ x y : ℝ, (x > 0 ∧ y > 0) → ax + by + c ≠ 0) : ac ≥ 0 :=
by
  sorry

end line_first_quadrant_l727_727475


namespace polynomials_equality_l727_727550

variable {R : Type} [CommRing R] [IsDomain R]

noncomputable def is_integer_coefficient (p : R[X]) : Prop :=
  ∀ n : Nat, p.coeff n ∈ Int

def leading_coefficient_positive (p : R[X]) : Prop :=
  leadingCoeff p > 0

theorem polynomials_equality
  (f g : R[X])
  (hf : is_integer_coefficient f)
  (hg : is_integer_coefficient g)
  (hposf : leading_coefficient_positive f)
  (hposg : leading_coefficient_positive g)
  (hdf : odd (degree f).toNat)
  (hS : {y | ∃ a : ℤ, f.eval ↑a = y} = {y | ∃ a : ℤ, g.eval ↑a = y}) :
  ∃ k : ℤ, ∀ x : R, g.eval x = f.eval (x + k) :=
sorry

end polynomials_equality_l727_727550


namespace median_line_eqn_l727_727625

theorem median_line_eqn (A B C : ℝ × ℝ)
  (hA : A = (3, 7)) (hB : B = (5, -1)) (hC : C = (-2, -5)) :
  ∃ m b : ℝ, (4, -3, -7) = (m, b, 0) :=
by sorry

end median_line_eqn_l727_727625


namespace animal_legs_l727_727965

theorem animal_legs (dogs chickens spiders octopus : Nat) (legs_dog legs_chicken legs_spider legs_octopus : Nat)
  (h1 : dogs = 3)
  (h2 : chickens = 4)
  (h3 : spiders = 2)
  (h4 : octopus = 1)
  (h5 : legs_dog = 4)
  (h6 : legs_chicken = 2)
  (h7 : legs_spider = 8)
  (h8 : legs_octopus = 8) :
  dogs * legs_dog + chickens * legs_chicken + spiders * legs_spider + octopus * legs_octopus = 44 := by
    sorry

end animal_legs_l727_727965


namespace matrix_inverse_exists_and_correct_l727_727430

theorem matrix_inverse_exists_and_correct : 
  let A := matrix ([[3, 4], [-2, 9]] : matrix (fin 2) (fin 2) ℚ)
  let detA := (3 * 9) - (4 * -2)
  detA ≠ 0 →
  matrix.inv A = matrix ([[9/35, -4/35], [2/35, 3/35]] : matrix (fin 2) (fin 2) ℚ) :=
by
  let A := matrix ([[3, 4], [-2, 9]] : matrix (fin 2) (fin 2) ℚ)
  let detA := (3 * 9) - (4 * -2)
  have detA_nz : detA ≠ 0 := by simp [detA]
  have invA := (matrix.inv_of_det_ne_zero _ detA_nz)
  have matrix_inv_eq := (invA A)
  let expected_inv := (matrix ([[9/35, -4/35], [2/35, 3/35]] : matrix (fin 2) (fin 2) ℚ))
  exact matrix_inv_eq = expected_inv
  sorry

end matrix_inverse_exists_and_correct_l727_727430


namespace comp_inter_empty_l727_727160

section
variable {α : Type*} [DecidableEq α]
variable (I M N : Set α)
variable (a b c d e : α)
variable (hI : I = {a, b, c, d, e})
variable (hM : M = {a, c, d})
variable (hN : N = {b, d, e})

theorem comp_inter_empty : 
  (I \ M) ∩ (I \ N) = ∅ :=
by sorry
end

end comp_inter_empty_l727_727160


namespace R_value_one_l727_727956

variable (x y z t : ℝ)
variable (h : x * y * z * t = 1)

def R : ℝ := (1 / (1 + x + x * y + x * y * z)) + 
             (1 / (1 + y + y * z + y * z * t)) + 
             (1 / (1 + z + z * t + z * t * x)) + 
             (1 / (1 + t + t * x + t * x * y))

theorem R_value_one : R x y z t = 1 :=
by
  sorry

end R_value_one_l727_727956


namespace volume_of_pyramid_l727_727982

/--
Rectangle ABCD is the base of pyramid PABCD. Let AB = 10, BC = 6, PA is perpendicular to AB, and PB = 20. 
If PA makes an angle θ = 30° with the diagonal AC of the base, prove the volume of the pyramid PABCD is 200 cubic units.
-/
theorem volume_of_pyramid (AB BC PB : ℝ) (θ : ℝ) (hAB : AB = 10) (hBC : BC = 6)
  (hPB : PB = 20) (hθ : θ = 30) (PA_is_perpendicular_to_AB : true) (PA_makes_angle_with_AC : true) : 
  ∃ V, V = 1 / 3 * (AB * BC) * 10 ∧ V = 200 := 
by
  exists 1 / 3 * (AB * BC) * 10
  sorry

end volume_of_pyramid_l727_727982


namespace max_avg_speed_at_third_hour_l727_727739

variable dist : ℕ → ℕ 
variable (h0 : dist 0 = 0)
variable (h1 : dist 1 = 90)
variable (h2 : dist 2 = 250)
variable (h3 : dist 3 = 480)
variable (h4 : dist 4 = 590)
variable (h5 : dist 5 = 780)
variable (h6 : dist 6 = 900)

theorem max_avg_speed_at_third_hour : 
  ∃ t ∈ {0, 1, 2, 3, 4, 5}, (∀ t' ∈ {0, 1, 2, 3, 4, 5}, (dist (t+1) - dist t) ≥ (dist (t'+1) - dist t')) ∧ (t = 2) := 
by
  sorry

end max_avg_speed_at_third_hour_l727_727739


namespace find_A_l727_727126

noncomputable def A : ℝ :=
  -((2 * Real.sqrt (Real.sqrt 2)) / 3)

theorem find_A :
  3.443 * A = (Real.cot (5 * Real.pi / 4)) - 2 * (Real.sin ((5 * Real.pi / 2) + 
  (1 / 2) * Real.arcsin ((2 * Real.sqrt 2 - 1) / 3))) ^ 2 :=
by
  sorry

end find_A_l727_727126


namespace multiplication_with_mixed_number_l727_727015

-- Define mixed numbers as rational numbers for proper calculation
def mixed_to_rational (whole : ℕ) (num : ℕ) (den : ℕ) : ℚ :=
  whole + num / den

-- 7 * (9 + 2/5)
def lhs : ℚ := 7 * mixed_to_rational 9 2 5

-- 65 + 4/5
def rhs : ℚ := mixed_to_rational 65 4 5

theorem multiplication_with_mixed_number : lhs = rhs := by
  sorry

end multiplication_with_mixed_number_l727_727015


namespace eight_divides_even_integer_l727_727616

theorem eight_divides_even_integer (a n : ℕ) (h1 : even a) 
  (h2 : A = (finset.range (n + 1)).sum (λ i, a^i)) 
  (h3 : ∃ k : ℕ, A = k^2) : 8 ∣ a :=
by sorry

end eight_divides_even_integer_l727_727616


namespace line_circle_no_intersection_ellipse_intersection_l727_727867

theorem line_circle_no_intersection_ellipse_intersection 
  (m n : Real)
  (h_line_circle_no_intersection: ∀ (x y : Real), 
    mx - ny ≠ 4 → x^2 + y^2 ≠ 4)
  : ∃ (L : Real → Real → Prop), 
    (∀ (x y: Real), L x y ↔ ∃ k : Real, y = k * x + n) ∧ 
    (card {p : Real × Real // L p.1 p.2 ∧ (p.1^2 / 9 + p.2^2 / 4 = 1)} = 2) :=
by sorry

end line_circle_no_intersection_ellipse_intersection_l727_727867


namespace function_point_proof_l727_727178

-- Given conditions
def condition (f : ℝ → ℝ) : Prop :=
  f 1 = 3

-- Prove the statement
theorem function_point_proof (f : ℝ → ℝ) (h : condition f) : f (-1) + 1 = 4 :=
by
  -- Adding the conditions here
  sorry -- proof is not required

end function_point_proof_l727_727178


namespace find_k_l727_727087

variables {k : ℝ}
def a : ℝ × ℝ × ℝ := (1, 1, 0)
def b : ℝ × ℝ × ℝ := (-1, 0, 2)

def k_a_plus_b (k : ℝ) : ℝ × ℝ × ℝ :=
  (k - 1, k, 2)

def two_a_minus_b : ℝ × ℝ × ℝ :=
  (3, 2, -2)

theorem find_k
  (h : (k_a_plus_b k).1 * two_a_minus_b.1 +
       (k_a_plus_b k).2 * two_a_minus_b.2 +
       (k_a_plus_b k).3 * two_a_minus_b.3 = 0) :
  k = 7 / 5 :=
sorry

end find_k_l727_727087


namespace sally_cards_sum_l727_727260

theorem sally_cards_sum :
  ∃ (R : Finset ℕ) (B : Finset ℕ) (stack : List ℕ),
    R = {1, 2, 3, 4, 5, 6} ∧
    B = {4, 5, 6, 7, 8} ∧
    (∀ (i : ℕ), i < stack.length - 1 →
      ((i % 2 = 0 → stack[i] ∈ R ∧ stack[i + 1] ∈ B) ∧
      (i % 2 = 1 → stack[i] ∈ B ∧ stack[i + 1] ∈ R))) ∧
    (∀ (i : ℕ), i < stack.length - 1 →
      ((stack[i] ∈ R ∧ stack[i + 1] ∈ B) →
      stack[i + 1] % stack[i] = 0)) ∧
    (∀ (i : ℕ), 0 < i ∧ i < stack.length - 1 →
      (stack[i] ∈ R →
      (stack[i - 1] + stack[i + 1]) % 2 = 0)) ∧
    (stack[4] + stack[5] + stack[6] = 10) := sorry

end sally_cards_sum_l727_727260


namespace factorize_x_cubic_l727_727067

-- Define the function and the condition
def factorize (x : ℝ) : Prop := x^3 - 9 * x = x * (x + 3) * (x - 3)

-- Prove the factorization property
theorem factorize_x_cubic (x : ℝ) : factorize x :=
by
  sorry

end factorize_x_cubic_l727_727067


namespace parallelepiped_face_areas_l727_727626

theorem parallelepiped_face_areas
    (h₁ : ℝ := 2)  -- height corresponding to face area x
    (h₂ : ℝ := 3)  -- height corresponding to face area y
    (h₃ : ℝ := 4)  -- height corresponding to face area z
    (total_surface_area : ℝ := 36) : 
    ∃ (x y z : ℝ), 
    2 * x + 2 * y + 2 * z = total_surface_area ∧
    (∃ V : ℝ, V = h₁ * x ∧ V = h₂ * y ∧ V = h₃ * z) ∧
    x = 108 / 13 ∧ y = 72 / 13 ∧ z = 54 / 13 := 
by 
  sorry

end parallelepiped_face_areas_l727_727626


namespace probability_hits_three_times_exactly_two_consecutive_hits_l727_727719

theorem probability_hits_three_times_exactly_two_consecutive_hits :
  let prob := (4! / 2!) * (1 / 2) ^ 6 in
  prob = (4! / 2!) * (1 / 2) ^ 6 :=
by
  sorry

end probability_hits_three_times_exactly_two_consecutive_hits_l727_727719


namespace rugs_combined_area_l727_727305

/-- 
 Given the conditions where:
 - Three rugs cover a floor area of 140 square meters when overlapped.
 - The area covered by exactly two layers of rug is 24 square meters.
 - The area covered with three layers of rug is 24 square meters.
 We need to prove that the combined area of the three rugs is 212 square meters.
-/
theorem rugs_combined_area 
  (floor_area : ℕ)
  (area_two_layers : ℕ)
  (area_three_layers : ℕ)
  (one_layer_area : ℕ)
  (total_combined_area : ℕ) :
  floor_area = 140 ∧ area_two_layers = 24 ∧ area_three_layers = 24 ∧ 
  one_layer_area = 92 →
  total_combined_area = one_layer_area + (2 * area_two_layers) + (3 * area_three_layers) →
  total_combined_area = 212 :=
begin
  intros h1 h2,
  sorry
end

end rugs_combined_area_l727_727305


namespace cos_double_alpha_half_pi_l727_727108

theorem cos_double_alpha_half_pi (α : ℝ) (h : cos α - sin α = 1 / 5) : 
  cos (2 * α - π / 2) = 24 / 25 := 
  sorry

end cos_double_alpha_half_pi_l727_727108


namespace find_n_l727_727507

theorem find_n (x k m n : ℤ) 
  (h1 : x = 82 * k + 5)
  (h2 : x + n = 41 * m + 18) :
  n = 5 :=
by
  sorry

end find_n_l727_727507


namespace in_first_quadrant_l727_727908

open complex

theorem in_first_quadrant (z : ℂ) (h : z = (2 - I) / (1 - 3 * I)) : 
  z.re > 0 ∧ z.im > 0 :=
by
  have h1 : z = (2 - I) / (1 - 3 * I) := h
  -- Continue from here to prove that z.re > 0 ∧ z.im > 0
  sorry

end in_first_quadrant_l727_727908


namespace Z_real_Z_imaginary_Z_pure_imaginary_l727_727827

-- Definitions

def Z (a : ℝ) : ℂ := (a^2 - 9 : ℝ) + (a^2 - 2 * a - 15 : ℂ)

-- Statement for the proof problems

theorem Z_real (a : ℝ) : 
  (Z a).im = 0 ↔ a = 5 ∨ a = -3 := sorry

theorem Z_imaginary (a : ℝ) : 
  (Z a).re = 0 ↔ a ≠ 5 ∧ a ≠ -3 := sorry

theorem Z_pure_imaginary (a : ℝ) : 
  (Z a).re = 0 ∧ (Z a).im ≠ 0 ↔ a = 3 := sorry

end Z_real_Z_imaginary_Z_pure_imaginary_l727_727827


namespace correct_calculation_l727_727673

theorem correct_calculation : 
  (a b : ℂ) → (hA : ¬ (a + a^2 = a^3)) → (hB : ¬ (a^2 * a^3 = a^6)) → (hC : ¬ ((2 * a^3 * b)^3 = 6 * a^3 * b^3)) → (hD : a^6 / a^4 = a^2) → ∃ (correctOption : Prop), correctOption = (a^6 / a^4 = a^2) :=
by
  intros a b hA hB hC hD
  existsi (a^6 / a^4 = a^2)
  exact hD
  sorry

end correct_calculation_l727_727673


namespace calculate_product_l727_727000

theorem calculate_product :
  7 * (9 + 2/5) = 65 + 4/5 := 
by
  sorry

end calculate_product_l727_727000


namespace steve_final_height_l727_727612

-- Define the initial height of Steve in inches.
def initial_height : ℕ := 5 * 12 + 6

-- Define how many inches Steve grew.
def growth : ℕ := 6

-- Define Steve's final height after growing.
def final_height : ℕ := initial_height + growth

-- The final height should be 72 inches.
theorem steve_final_height : final_height = 72 := by
  -- we don't provide the proof here
  sorry

end steve_final_height_l727_727612


namespace base7_divisibility_l727_727443

/-
Given a base-7 number represented as 3dd6_7, where the first digit is 3, the last digit is 6, 
and the middle two digits are equal to d, prove that the base-7 digit d which makes this 
number divisible by 13 is 5.
-/
theorem base7_divisibility (d : ℕ) (hdig : d ∈ {0, 1, 2, 3, 4, 5, 6}) : 
  (3 * 7^3 + d * 7^2 + d * 7 + 6) % 13 = 0 ↔ d = 5 := 
sorry

end base7_divisibility_l727_727443


namespace sum_of_angles_l727_727525

theorem sum_of_angles (BAC DEF : ℝ)
  (h₁ : is_isosceles BAC 30)
  (h₂ : is_isosceles DEF 40)
  (h₃ : equal_heights BAC DEF)
  (h₄ : support_horizontal_beam BAC DEF)
  : ∠ DAC + ∠ ADE = 145 := by
sorry -- proof goes here

end sum_of_angles_l727_727525


namespace maximize_P_n_k_at_n_20_l727_727654

noncomputable def P_n (n k : ℕ) : ℝ :=
(binomial n k) * ((1/6)^k) * ((5/6)^(n-k))

theorem maximize_P_n_k_at_n_20 : ∃ k : ℕ, (0 ≤ k ∧ k ≤ 20) ∧ ∀ m : ℕ, (0 ≤ m ∧ m ≤ 20) → P_n 20 k ≥ P_n 20 m ∧ k = 3 := 
begin
  sorry
end

end maximize_P_n_k_at_n_20_l727_727654


namespace composite_integer_divisors_l727_727405

theorem composite_integer_divisors (n : ℕ) (k : ℕ) (d : ℕ → ℕ) 
  (h_composite : 1 < n ∧ ¬Prime n)
  (h_divisors : ∀ i, 1 ≤ i ∧ i ≤ k → d i ∣ n)
  (h_distinct : ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → d i < d j)
  (h_range : d 1 = 1 ∧ d k = n)
  (h_ratio : ∀ i, 1 ≤ i ∧ i < k → (d (i + 1) - d i) = (i * (d 2 - d 1))) : n = 6 :=
by sorry

end composite_integer_divisors_l727_727405


namespace solve_for_x_l727_727502

theorem solve_for_x : ∀ x : ℝ, 3 * x - 8 = -2 * x + 17 → x = 5 := 
by {
  intros x h,
  have h1 : 5 * x - 8 = 17, from sorry,
  have h2 : 5 * x = 25, from sorry,
  have h3 : x = 5, from sorry,
  exact h3,
}

end solve_for_x_l727_727502


namespace percentage_HCl_in_added_solution_l727_727349

theorem percentage_HCl_in_added_solution (P V : ℝ) (hV : V ≠ 0) :
  60 * 0.40 + (V * P / 100) = (60 + V) * 0.25 → P = 25 - (900 / V) :=
by
  intro h
  have : 60 * 0.40 = 24 := by norm_num
  rw [this] at h
  sorry

end percentage_HCl_in_added_solution_l727_727349


namespace matrix_inv_correct_l727_727423

open Matrix

noncomputable def matrix_A : Matrix (Fin 2) (Fin 2) ℤ :=
  !![3, 4;
   -2, 9]

noncomputable def matrix_A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  !![
    (9 : ℚ)/35, (-4 : ℚ)/35;
    (2 : ℚ)/35, (3 : ℚ)/35]

theorem matrix_inv_correct : matrix_A⁻¹ = matrix_A_inv := by
  sorry

end matrix_inv_correct_l727_727423


namespace not_real_div_roots_of_quadratic_l727_727118

theorem not_real_div_roots_of_quadratic (b : ℝ) (hb : -2 < b ∧ b < 2) :
  let z1 := -b / 2 + Complex.i * Real.sqrt (4 - b^2) / 2
  let z2 := -b / 2 - Complex.i * Real.sqrt (4 - b^2) / 2
  ¬ (z1 / z2) ∈ ℝ := sorry

end not_real_div_roots_of_quadratic_l727_727118


namespace real_and_equal_roots_l727_727822

theorem real_and_equal_roots (k : ℝ) : (∃ x : ℝ, 3 * x^2 - (k + 1) * x + 2 * k = 0) ∧ (∀ x1 x2 : ℝ, 3 * x1^2 - (k + 1) * x1 + 2 * k = 0 ∧ 3 * x2^2 - (k + 1) * x2 + 2 * k = 0 → x1 = x2) → (k = 11 + 10 * Real.sqrt 6 ∨ k = 11 - 10 * Real.sqrt 6) :=
by
  sorry

end real_and_equal_roots_l727_727822


namespace coordinates_F_l727_727656

-- Definition of point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Reflection over the y-axis
def reflect_y (p : Point) : Point :=
  { x := -p.x, y := p.y }

-- Reflection over the x-axis
def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

-- Original point F
def F : Point := { x := 3, y := 3 }

-- First reflection over the y-axis
def F' := reflect_y F

-- Second reflection over the x-axis
def F'' := reflect_x F'

-- Goal: Coordinates of F'' after both reflections
theorem coordinates_F'' : F'' = { x := -3, y := -3 } :=
by
  -- Proof would go here
  sorry

end coordinates_F_l727_727656


namespace geometric_sequence_common_ratio_l727_727135

variable (m : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Definition of the sum of the first n terms
def sum_first_n_terms (n : ℕ) : ℝ := 3 * 2^n + m

-- Definition of the sequence terms using the sum
def sequence_term (n : ℕ) : ℝ :=
  sum_first_n_terms m (n) - sum_first_n_terms m (n - 1)

-- Definition of the common ratio
def common_ratio (n : ℕ) : ℝ :=
  sum_first_n_terms m (n + 1) - sum_first_n_terms m (n) /
  (sum_first_n_terms m (n) - sum_first_n_terms m (n - 1))

theorem geometric_sequence_common_ratio :
  ∀ (m : ℝ), common_ratio m 1 = 2 := by
  sorry

end geometric_sequence_common_ratio_l727_727135


namespace sin_15_mul_sin_75_l727_727297

theorem sin_15_mul_sin_75 : Real.sin (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 1 / 4 := 
by
  sorry

end sin_15_mul_sin_75_l727_727297


namespace calculate_product_l727_727001

theorem calculate_product :
  7 * (9 + 2/5) = 65 + 4/5 := 
by
  sorry

end calculate_product_l727_727001


namespace points_concyclic_l727_727390

open EuclideanGeometry

-- Define the conditions
variables {Γ₁ Γ₂ : Circle} -- Γ₁ and Γ₂ are two circles
variables {A B C D E F P Q M N : Point} -- Points involved in the problem

-- Assume the conditions
variables (h_intersect : ∃ (A B : Point), A ∈ Γ₁ ∧ A ∈ Γ₂ ∧ B ∈ Γ₁ ∧ B ∈ Γ₂)
variables (h_pass_through_B1 : ∃ (C D : Point), B ∈ Line C D ∧ C ∈ Γ₁ ∧ D ∈ Γ₂)
variables (h_pass_through_B2 : ∃ (E F : Point), B ∈ Line E F ∧ E ∈ Γ₁ ∧ F ∈ Γ₂)
variables (h_intersect_CF : ∃ (P Q : Point), P ∈ Line C F ∧ P ∈ Γ₁ ∧ Q ∈ Line C F ∧ Q ∈ Γ₂)
variables (h_midpoint_arcs : MidpointArc P B M Γ₁ ∧ MidpointArc Q B N Γ₂)
variables (h_equal_segments : dist C D = dist E F)

-- Define the goal
theorem points_concyclic 
  (h_intersect : ∃ (A B : Point), A ∈ Γ₁ ∧ A ∈ Γ₂ ∧ B ∈ Γ₁ ∧ B ∈ Γ₂)
  (h_pass_through_B1 : ∃ (C D : Point), B ∈ Line C D ∧ C ∈ Γ₁ ∧ D ∈ Γ₂)
  (h_pass_through_B2 : ∃ (E F : Point), B ∈ Line E F ∧ E ∈ Γ₁ ∧ F ∈ Γ₂)
  (h_intersect_CF : ∃ (P Q : Point), P ∈ Line C F ∧ P ∈ Γ₁ ∧ Q ∈ Line C F ∧ Q ∈ Γ₂)
  (h_midpoint_arcs : MidpointArc P B M Γ₁ ∧ MidpointArc Q B N Γ₂)
  (h_equal_segments : dist C D = dist E F) : 
  Concyclic C F M N :=
sorry -- proof to be provided

end points_concyclic_l727_727390


namespace bisector_of_given_angle_construct_angle_bisector_l727_727974

-- Define the constructs

variable (α β : ℝ)
variable (A O B P Q M : Point)
variable (a : ℝ) (h_acute : 0 < α ∧ α < π/2)

-- Part (a): The bisector of ∠AOB is OM
def bisector_AOB (A O B : Point) : Line :=
  let P := (translate_parallel O A a)
  let Q := (translate_parallel O B a)
  line_through O (intersection P Q)

theorem bisector_of_given_angle (A O B M : Point) (a : ℝ) :
    M = intersection (translate_parallel O A a) (translate_parallel O B a) →
    bisector_AOB A O B = line_through O M :=
sorry

-- Part (b): Constructing an angle VOC
def construct_angle_VOC (A O B : Point) : Angle :=
  let P := (translate_parallel O B a)
  let M := (intersection P (translate_parallel O A a))
  let N := (translate_parallel M P a)
  angle_through B O (intersection N (other_parallel_through M P))

theorem construct_angle_bisector (A O B M : Point) (a : ℝ) :
    ∠AOB < π/2 →
    M = intersection (translate_parallel O A a) (translate_parallel O B a) →
    bisector (construct_angle_VOC A O B) = ray_through O A :=
sorry

end bisector_of_given_angle_construct_angle_bisector_l727_727974


namespace cloves_used_for_roast_chicken_l727_727238

section
variable (total_cloves : ℕ)
variable (remaining_cloves : ℕ)

theorem cloves_used_for_roast_chicken (h1 : total_cloves = 93) (h2 : remaining_cloves = 7) : total_cloves - remaining_cloves = 86 := 
by 
  have h : total_cloves - remaining_cloves = 93 - 7 := by rw [h1, h2]
  exact h
-- sorry
end

end cloves_used_for_roast_chicken_l727_727238


namespace Kaarel_wins_l727_727603

theorem Kaarel_wins (p : ℕ) (hp : Prime p) (hp_gt3 : p > 3) :
  ∃ (x y a : ℕ), x ∈ Finset.range (p-1) ∧ y ∈ Finset.range (p-1) ∧ a ∈ Finset.range (p-1) ∧ 
  x ≠ y ∧ y ≠ (p - x) ∧ a ≠ x ∧ a ≠ (p - x) ∧ a ≠ y ∧ 
  (x * (p - x) + y * a) % p = 0 :=
sorry

end Kaarel_wins_l727_727603


namespace conversion_correct_l727_727763

-- Define the base 8 number
def base8_number : ℕ := 4 * 8^3 + 5 * 8^2 + 3 * 8^1 + 2 * 8^0

-- Define the target base 10 number
def base10_number : ℕ := 2394

-- The theorem that needs to be proved
theorem conversion_correct : base8_number = base10_number := by
  sorry

end conversion_correct_l727_727763


namespace lunch_break_duration_210_minutes_l727_727236

-- Define variables:
--  m: rate at which Maria works (project/hours)
--  a: combined rate of the three assistants (project/hours)
--  L: duration of the lunch break (hours)

variables (m a L : ℝ)

-- Monday condition: (10 - L)(m + a) = 0.6
def monday_condition : Prop := (10 - L) * (m + a) = 0.6

-- Tuesday condition: (8 - L) * (2 / 3) * a = 0.3
def tuesday_condition : Prop := (8 - L) * (2 / 3) * a = 0.3

-- Wednesday condition: (4 - L) * m = 0.1
def wednesday_condition : Prop := (4 - L) * m = 0.1

-- Final goal is to prove that L in minutes is 210
theorem lunch_break_duration_210_minutes 
  (m : ℝ) (a : ℝ) (L : ℝ) 
  (h1 : monday_condition m a L) 
  (h2 : tuesday_condition m a L) 
  (h3 : wednesday_condition m a L) : 
  (L * 60) = 210 :=
sorry

end lunch_break_duration_210_minutes_l727_727236


namespace remainder_x_squared_div_20_l727_727074

theorem remainder_x_squared_div_20 (x : ℤ) 
  (h1 : 5 * x ≡ 10 [MOD 20])
  (h2 : 6 * x ≡ 12 [MOD 20]) :
  (x^2 ≡ 4 [MOD 20]) :=
by sorry

end remainder_x_squared_div_20_l727_727074


namespace linear_dependent_vectors_l727_727287

theorem linear_dependent_vectors (k : ℤ) :
  (∃ (a b : ℤ), (a ≠ 0 ∨ b ≠ 0) ∧ a * 2 + b * 4 = 0 ∧ a * 3 + b * k = 0) ↔ k = 6 :=
by
  sorry

end linear_dependent_vectors_l727_727287


namespace conversion_base8_to_base10_l727_727760

theorem conversion_base8_to_base10 : 
  (4 * 8^3 + 5 * 8^2 + 3 * 8^1 + 2 * 8^0) = 2394 := by 
  sorry

end conversion_base8_to_base10_l727_727760


namespace decreasing_function_range_l727_727129

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x < 1 then a^x else x * real.log x - a * x^2

theorem decreasing_function_range (a : ℝ) (h_decreasing : ∀ x y : ℝ, x < y → f a y ≤ f a x) :
  a ∈ set.Ico (1/2 : ℝ) 1 :=
sorry

end decreasing_function_range_l727_727129


namespace proof_of_xw_l727_727170

-- Defining the conditions
variables (x w : ℕ)

-- The given conditions
def condition1 := 7 * x = 28
def condition2 := x + w = 9

-- The proof goal
theorem proof_of_xw : condition1 → condition2 → x * w = 20 := 
by 
  sorry

end proof_of_xw_l727_727170


namespace length_ratio_l727_727503

-- Declare the basic definitions and conditions for the problem
variables (Q : Type) [metric_space Q] [inner_product_space ℝ Q] [finite_dimensional ℝ Q]
variables (C D P : Q)
variables (d : ℝ)
variables (O : Q) (PD : ℝ)

-- Assume perpendicular diameters and given lengths
variables (A B : Q) (CD_length : 2 * d = dist C D) 
          (perpendicular : ∀(A B : Q), inner A B = 0)
          (P_on_CQ : P ∈ line_through C Q)
          (angle_CPD : ∠ C P D = real.pi/4)

-- The crucial point to prove is the ratio of lengths PQ and CQ
theorem length_ratio (PQ_length CQ_length : ℝ) : 
  let PQ := dist P Q,
      CQ := dist C Q in
  (PQ / CQ = real.sqrt 5 / 2) :=
sorry

end length_ratio_l727_727503


namespace covered_area_of_two_squares_l727_727611

theorem covered_area_of_two_squares 
  {A B C D E F G H : Point} -- Assume we have points representing vertices
  (hAB : congruent (side_length A B) 12)
  (hCD : congruent (side_length C D) 12)
  (hEF : congruent (side_length E F) 12)
  (hGH : congruent (side_length G H) 12)
  (h_congruent : congruent_square ABCD EFGH)
  (h_position : coincides G A) :
  area_covered_by_two_squares ABCD EFGH = 252 :=
sorry

end covered_area_of_two_squares_l727_727611


namespace tangent_line_at_g_decreasing_interval_and_local_minimum_l727_727485

noncomputable def f (x : ℝ) : ℝ := real.log x
noncomputable def g (x : ℝ) : ℝ := real.log x + x^2 - 3 * x

theorem tangent_line_at (x y : ℝ) (h : (x, y) = (1, 0)) : x - y - 1 = 0 := by
  sorry

theorem g_decreasing_interval_and_local_minimum :
  (∀ x : ℝ, (1 / 2) < x → x < 1 → deriv g x < 0) ∧ g 1 = -2 := by
  sorry

end tangent_line_at_g_decreasing_interval_and_local_minimum_l727_727485


namespace ellipse_general_equation_l727_727137

theorem ellipse_general_equation (x y : ℝ) (α : ℝ) (h1 : x = 5 * Real.cos α) (h2 : y = 3 * Real.sin α) :
  x^2 / 25 + y^2 / 9 = 1 :=
sorry

end ellipse_general_equation_l727_727137


namespace multiply_mixed_number_l727_727032

def mixed_to_improper_fraction (n : ℕ) (a b : ℕ) : ℚ :=
  n + a / b

def improper_to_mixed_number (q : ℚ) : ℕ × ℚ :=
  let n := q.natAbs
  let r := q.fract
  (n, r)

theorem multiply_mixed_number (x y n a b : ℕ) (h : y = mixed_to_improper_fraction n a b) :
  x * y = mixed_to_improper_fraction 65 4 5 :=
  sorry

end multiply_mixed_number_l727_727032


namespace angle_relation_l727_727212

theorem angle_relation 
  (A B C D E : Type)
  [Angle : A → B → ℝ]
  (h1 : Angle A C B = 2 * Angle A B C)
  (h2 : CD / BD = 2)
  (h3 : AD = DE) :
  Angle E C B + 180 = 2 * Angle E B C :=
sorry

end angle_relation_l727_727212


namespace ratio_dvds_to_cds_l727_727895

def total_sold : ℕ := 273
def dvds_sold : ℕ := 168
def cds_sold : ℕ := total_sold - dvds_sold

theorem ratio_dvds_to_cds : (dvds_sold : ℚ) / cds_sold = 8 / 5 := by
  sorry

end ratio_dvds_to_cds_l727_727895


namespace number_of_roots_l727_727233

noncomputable def f : ℝ → ℝ := sorry

axiom f_property : ∀ x, f(x + 1) = f(x) + 1

theorem number_of_roots : 
  (f 0 = 0 ∧ (∃ x : ℝ, ∀ n : ℤ, f (x + n) = x + n)) ∨ 
  (f 0 ≠ 0 ∧ (¬ ∃ x : ℝ, f x = x)) :=
sorry

end number_of_roots_l727_727233


namespace gcd_12012_18018_l727_727815

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_12012_18018 : gcd 12012 18018 = 6006 := sorry

end gcd_12012_18018_l727_727815


namespace log_defined_set_l727_727285

noncomputable def d : ℝ := 1001^(1002^(1005:ℝ):ℝ)

theorem log_defined_set (x : ℝ) : 
  (\log 1008 (\log 1005 (\log 1002 (\log 1001 x))) > 0) ↔ (x > d) :=
sorry

end log_defined_set_l727_727285


namespace digit_2023_in_7_div_26_is_3_l727_727416

-- We define the decimal expansion of 7/26 as a repeating sequence of "269230769"
def repeating_block : string := "269230769"

-- Verify that the 2023rd digit in the sequence is "3"
theorem digit_2023_in_7_div_26_is_3 :
  (repeating_block.str.to_list.nth ((2023 % 9) - 1)).iget = '3' :=
by
  sorry

end digit_2023_in_7_div_26_is_3_l727_727416


namespace sum_of_integers_with_even_product_l727_727513

theorem sum_of_integers_with_even_product (a b : ℤ) (h : ∃ k, a * b = 2 * k) : 
∃ k1 k2, a = 2 * k1 ∨ a = 2 * k1 + 1 ∧ (a + b = 2 * k2 ∨ a + b = 2 * k2 + 1) :=
by
  sorry

end sum_of_integers_with_even_product_l727_727513


namespace work_done_correct_l727_727508

-- Definitions based on conditions
def force : ℝ := 1 -- in Newtons (N)
def stretch_per_force : ℝ := 0.01 -- in meters per Newton (m/N)
def total_stretch : ℝ := 0.06 -- in meters (m)

-- Hooke's Law: Force = k * x, hence k = Force / x
def spring_constant : ℝ := force / stretch_per_force

-- Work done W = 1/2 * k * x^2
def work_done : ℝ := 0.5 * spring_constant * total_stretch^2

-- Statement to prove the work done is 0.18 J
theorem work_done_correct : work_done = 0.18 :=
by
  sorry

end work_done_correct_l727_727508


namespace compare_abc_l727_727936

def a : ℝ := (2 / 5) ^ (3 / 5)
def b : ℝ := (2 / 5) ^ (2 / 5)
def c : ℝ := (3 / 5) ^ (2 / 5)

theorem compare_abc : a < b ∧ b < c :=
by {
  sorry
}

end compare_abc_l727_727936


namespace paintings_per_gallery_l727_727380

theorem paintings_per_gallery (pencils_total: ℕ) (pictures_initial: ℕ) (galleries_new: ℕ) (pencils_per_signing: ℕ) (pencils_per_picture: ℕ) (pencils_for_signature: ℕ) :
  pencils_total = 88 ∧ pictures_initial = 9 ∧ galleries_new = 5 ∧ pencils_per_picture = 4 ∧ pencils_per_signing = 2 → 
  (pencils_total - (galleries_new + 1) * pencils_per_signing) / pencils_per_picture - pictures_initial = galleries_new * 2 :=
by
  intros h,
  cases h with ha hb,
  sorry

end paintings_per_gallery_l727_727380


namespace subset_of_positive_reals_l727_727551

def M := { x : ℝ | x > -1 }

theorem subset_of_positive_reals : {0} ⊆ M :=
by
  sorry

end subset_of_positive_reals_l727_727551


namespace b_15_eq_181_l727_727571

def b : ℕ → ℕ
| 0 := 0
| 1 := 1
| (m + n) := if m > 0 ∧ n > 0 then b m + b n + m * n + m + n else 0

theorem b_15_eq_181 : b 15 = 181 := by
  sorry

end b_15_eq_181_l727_727571


namespace incorrect_statement_d_l727_727597

theorem incorrect_statement_d {P A B C l : Type}
  [Inhabited P] [Inhabited l] -- This ensures P and l have some points
  [line_contains l A] [line_contains l B] [line_contains l C]
  (h_perp : is_perpendicular (segment P B) l)
  (h_dist : ∀ (X : Type), distance P l = segment P ((foot l P) X)) :
  ¬ (distance A (segment P C) = segment A C) :=
by
  -- This is where the proof would go
  sorry

end incorrect_statement_d_l727_727597


namespace cube_surface_area_to_volume_ratio_l727_727637

theorem cube_surface_area_to_volume_ratio (a : ℝ) (h : a = 1) :
    (6 * a ^ 2) / (a ^ 3) = 6 :=
by
  rw [h]
  have h₁ : 6 * 1 ^ 2 = 6 := by norm_num
  have h₂ : 1 ^ 3 = 1 := by norm_num
  rw [h₁, h₂]
  norm_num
  sorry

end cube_surface_area_to_volume_ratio_l727_727637


namespace find_Sn_find_Tn_l727_727959

-- Definition: A sequence {a_n} such that the sum of the first n terms is S_n and S_n + a_n = 1.
def sequence_a (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n + a n = 1

-- Definition: An arithmetic sequence {b_n} with b_1 + b_2 = b_3 and b_3 = 3.
def sequence_b (b : ℕ → ℝ) : Prop :=
  b 1 + b 2 = b 3 ∧ b 3 = 3

-- Theorem 1: Find S_n given the conditions.
theorem find_Sn (S : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_a : sequence_a S a) (h_b : sequence_b b) :
  ∀ n, S n = 1 - (1/2)^n :=
sorry

-- Theorem 2: Find T_n given the conditions.
theorem find_Tn (S : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_a : sequence_a S a) (h_b : sequence_b b) :
  ∀ n, ∑ i in finRange n, a i * b i = 2 - (n + 2) * (1/2)^n :=
sorry

end find_Sn_find_Tn_l727_727959


namespace rita_downstream_speed_l727_727985

-- Definitions from the given conditions
def upstream_speed : ℝ := 3       -- Rita's speed paddling upstream in miles per hour
def total_time : ℝ := 8           -- Total time of the trip in hours
def distance : ℝ := 18            -- Distance to the ice cream parlor in miles

-- The main theorem statement, asserting the speed downstream
theorem rita_downstream_speed : ∃ v : ℝ, 
  let time_upstream := distance / upstream_speed in
  let time_downstream := total_time - time_upstream in
  time_upstream + time_downstream = total_time ∧
  distance / time_downstream = v ∧
  v = 9 :=
by
  -- proof will go here
  sorry

end rita_downstream_speed_l727_727985


namespace Sn_eq_125_over_6_l727_727905

noncomputable def P (n : ℕ+) := (n, (2 / n : ℚ))

noncomputable def line_through (n : ℕ+) := 
  λ (x y : ℚ), y - (2 / n) = -(2 / (n * (n + 1 : ℕ+))) * (x - n)

noncomputable def b (n : ℕ+) : ℚ :=
  let y_intercept := (2 / n) + (2 / (n + 1 : ℕ+)) in
  let x_intercept := 2 * n + 1 in
  (1 / 2) * y_intercept * x_intercept

noncomputable def S (n : ℕ+) : ℚ :=
  (Finset.range n).sum (λ i, b (i + 1 : ℕ+))

theorem Sn_eq_125_over_6 : S 5 = 125 / 6 :=
sorry

end Sn_eq_125_over_6_l727_727905


namespace ceiling_lights_l727_727193

variable (S M L : ℕ)

theorem ceiling_lights (hM : M = 12) (hL : L = 2 * M)
    (hBulbs : S + 2 * M + 3 * L = 118) : S - M = 10 :=
by
  sorry

end ceiling_lights_l727_727193


namespace egg_laying_hen_l727_727746

theorem egg_laying_hen 
  (h : ∀ boxes eggs_per_box, boxes = 315 → eggs_per_box = 6 → boxes * eggs_per_box = 1890)
  (n_hens : 270)
  (n_days : 7) :
  (∃ eggs_per_week, eggs_per_week = 1890) →
  (∃ eggs_per_hen_per_day, eggs_per_hen_per_day = (1890 / (n_hens * n_days))) →
  1 = 1890 / (n_hens * n_days) :=
by {
  assume h1,
  assume h2,
  sorry
}

end egg_laying_hen_l727_727746


namespace gcd_12012_18018_l727_727813

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_12012_18018 : gcd 12012 18018 = 6006 := sorry

end gcd_12012_18018_l727_727813


namespace find_m_of_parallel_find_equation_of_equal_intercepts_l727_727878

def line1 (x y : ℝ) : Prop := 2 * x - y - 3 = 0
def line2 (x y m : ℝ) : Prop := x - m * y + 1 - 3 * m = 0

-- If l_1 is parallel to l_2, find the value of m
theorem find_m_of_parallel (m : ℝ) : 
  (∀ x y : ℝ, line1 x y → line2 x y m → true) →
  m = -1/2 :=
sorry

-- If l_2 has equal intercepts on both coordinate axes, find the equation of l_2
theorem find_equation_of_equal_intercepts (m : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → |1 - 3 * m| * |m| = |3 * m - 1|) →
  line2 = (λ (x y : ℝ), x + y + 4 = 0) ∨ line2 = (λ (x y : ℝ), 3 * x - y = 0) :=
sorry

end find_m_of_parallel_find_equation_of_equal_intercepts_l727_727878


namespace problem_statement_l727_727554

variables {A B C D E F G : Type} [MetricSpace A] {ABC : Triangle A}
variables {Γ : Circle A} {AB AC : Segment A} {AD AE : Length A} {BD CE : Line A}
variables {M N : Type} [MetricSpace M] {DE FG : Line A}

/-- Given the circumcircle of an acute triangle ABC with points D and E on segments AB and AC such that AD = AE.
    The perpendicular bisectors of BD and CE intersect the shorter arcs AB and AC of the circumcircle at points F and G respectively.
    The problem is to prove that the lines DE and FG are either parallel or coincide. -/
theorem problem_statement 
  (hΓ: IsCircumcircleOf Γ ABC) (hD: OnSegment D AB) (hE: OnSegment E AC) (hADAE: AD = AE) 
  (hF: OnPerpendicularBisectorOf F BD) (hG: OnPerpendicularBisectorOf G CE) 
  (intF: IntersectsAtArcs F Γ AB) (intG: IntersectsAtArcs G Γ AC) : 
  ParallelOrCoincide DE FG :=
sorry

end problem_statement_l727_727554


namespace gcd_of_12012_and_18018_l727_727797

theorem gcd_of_12012_and_18018 : Int.gcd 12012 18018 = 6006 := 
by
  -- Here we are assuming the factorization given in the conditions
  have h₁ : 12012 = 12 * 1001 := sorry
  have h₂ : 18018 = 18 * 1001 := sorry
  have gcd_12_18 : Int.gcd 12 18 = 6 := sorry
  -- This sorry will be replaced by the actual proof involving the above conditions to conclude the stated theorem
  sorry

end gcd_of_12012_and_18018_l727_727797


namespace square_area_l727_727643

theorem square_area (side : ℝ) (h : side = 30) : side * side = 900 := by {
  rw h,
  calc
    30 * 30 = 900 : by norm_num
}

end square_area_l727_727643


namespace count_elements_le_20_l727_727105

def a_seq : ℕ → ℕ
| 0     := 1 
| (2 * n) := n + 1
| (2 * n + 1) := a_seq (2 * n) + a_seq (2 * n - 1)

noncomputable def num_elements_le_20 : ℕ := sorry -- placeholder for the result

theorem count_elements_le_20:
  num_elements_le_20 = 24 :=
sorry

end count_elements_le_20_l727_727105


namespace age_of_other_man_l727_727622

/-!
  The average age of 7 men increases by 4 years when two women are included in place of two men.
  One of the men is 26 years old, and the average age of the women is 42.
  Prove that the age of the other man who was replaced is 30 years old.
-/

theorem age_of_other_man 
  (A : ℝ)  -- Denote the average age of the 7 men
  (h1 : 7 * A + 28 = 7 * (A + 4))  -- New average age increases by 4 years
  (h2 : (2 * 42) = 84)  -- Average age of the women is 42
  (h3 : 7 * (A + 4) = 7 * A - M - 26 + (2 * 42))  -- Total age equation after replacement
  (h4 : A - 30 = 0)  -- Given one of the men is 26 and need to find the other
  : M = 30 := 
  sorry  -- The proof is omitted

end age_of_other_man_l727_727622


namespace john_computers_fixed_count_l727_727540

-- Define the problem conditions.
variables (C : ℕ)
variables (unfixable_ratio spare_part_ratio fixable_ratio : ℝ)
variables (fixed_right_away : ℕ)
variables (h1 : unfixable_ratio = 0.20)
variables (h2 : spare_part_ratio = 0.40)
variables (h3 : fixable_ratio = 0.40)
variables (h4 : fixed_right_away = 8)
variables (h5 : fixable_ratio * ↑C = fixed_right_away)

-- The theorem to prove.
theorem john_computers_fixed_count (h1 : C > 0) : C = 20 := by
  sorry

end john_computers_fixed_count_l727_727540


namespace angle_A_triangle_area_l727_727537

variable {A B C a b c : ℝ}

-- Condition 1: Fundamental trigonometric equation
axiom trig_eq : 2 * sin (2 * A) * cos A - sin (3 * A) + sqrt 3 * cos A = sqrt 3

-- Condition 2: Side length of the triangle
axiom a_eq_1 : a = 1

-- Condition 3: Trigonometric relation regarding sine function
axiom sin_rel : sin A + sin (B - C) = 2 * sin (2 * C)

-- Proving angle A is π/3
theorem angle_A : A = π / 3 := by
  sorry

-- Additional conditions to be used in the second part
axiom angle_eq : A = π / 3

-- Proving the area of the triangle
theorem triangle_area : 
  (a = 1) → 
  (sin A + sin (B - C) = 2 * sin (2 * C)) → 
  (A = π / 3) → 
  ∃ S, S = (sqrt 3 / 6) :=
by
  sorry

end angle_A_triangle_area_l727_727537


namespace second_divisor_l727_727705

theorem second_divisor (N k D m : ℤ) (h1 : N = 35 * k + 25) (h2 : N = D * m + 4) : D = 17 := by
  -- Follow conditions from problem
  sorry

end second_divisor_l727_727705


namespace decimal_expansion_2023rd_digit_l727_727414

theorem decimal_expansion_2023rd_digit 
  (x : ℚ) 
  (hx : x = 7 / 26) 
  (decimal_expansion : ℕ → ℕ)
  (hdecimal : ∀ n : ℕ, decimal_expansion n = if n % 12 = 0 
                        then 2 
                        else if n % 12 = 1 
                          then 7 
                          else if n % 12 = 2 
                            then 9 
                            else if n % 12 = 3 
                              then 2 
                              else if n % 12 = 4 
                                then 3 
                                else if n % 12 = 5 
                                  then 0 
                                  else if n % 12 = 6 
                                    then 7 
                                    else if n % 12 = 7 
                                      then 6 
                                      else if n % 12 = 8 
                                        then 9 
                                        else if n % 12 = 9 
                                          then 2 
                                          else if n % 12 = 10 
                                            then 3 
                                            else 0) :
  decimal_expansion 2023 = 0 :=
sorry

end decimal_expansion_2023rd_digit_l727_727414


namespace ratio_alisha_to_todd_is_two_to_one_l727_727150

-- Definitions
def total_gumballs : ℕ := 45
def todd_gumballs : ℕ := 4
def bobby_gumballs (A : ℕ) : ℕ := 4 * A - 5
def remaining_gumballs : ℕ := 6

-- Condition stating Hector's gumball distribution
def hector_gumballs_distribution (A : ℕ) : Prop :=
  todd_gumballs + A + bobby_gumballs A + remaining_gumballs = total_gumballs

-- Definition for the ratio of the gumballs given to Alisha to Todd
def ratio_alisha_todd (A : ℕ) : ℕ × ℕ :=
  (A / 4, todd_gumballs / 4)

-- Theorem stating the problem
theorem ratio_alisha_to_todd_is_two_to_one : ∃ (A : ℕ), hector_gumballs_distribution A → ratio_alisha_todd A = (2, 1) :=
sorry

end ratio_alisha_to_todd_is_two_to_one_l727_727150


namespace log_eq_sqrt_five_l727_727038

open Real

theorem log_eq_sqrt_five (x : ℝ) :
  log 5 x = 1 / 2 ↔ x = sqrt 5 :=
by sorry

end log_eq_sqrt_five_l727_727038


namespace find_saturday_hours_l727_727383

/-
Amanda charges $20.00 per hour to help clean out and organize a person's home.
She has 5 1.5 hours appointments on Monday, a 3-hours appointment on Tuesday and 2 2-hours appointments on Thursday. 
On Saturday, she will spend a certain number of hours at one client's house. She will make $410 this week.
-/

def hourly_rate : ℝ := 20
def monday_appointments : ℕ := 5
def monday_hours_per_appointment : ℝ := 1.5
def tuesday_appointments : ℕ := 1
def tuesday_hours_per_appointment : ℝ := 3
def thursday_appointments : ℕ := 2
def thursday_hours_per_appointment : ℝ := 2
def total_weekly_earnings : ℝ := 410

theorem find_saturday_hours :
  let monday_total_hours := monday_appointments * monday_hours_per_appointment in
  let tuesday_total_hours := tuesday_appointments * tuesday_hours_per_appointment in
  let thursday_total_hours := thursday_appointments * thursday_hours_per_appointment in
  let monday_to_thursday_hours := monday_total_hours + tuesday_total_hours + thursday_total_hours in
  let monday_to_thursday_earnings := monday_to_thursday_hours * hourly_rate in
  let remaining_earnings := total_weekly_earnings - monday_to_thursday_earnings in
  let saturday_hours := remaining_earnings / hourly_rate in
  saturday_hours = 6 := 
by
  sorry

end find_saturday_hours_l727_727383


namespace area_of_shaded_region_l727_727686

-- Defining the necessary elements
variable {r x h : ℝ}  -- r is the radius of the semicircle, x is the distance from O to C, h is the length of CD

-- Condition stating CD is perpendicular to AB and its length is h
axiom perpendicular_CD_AB : true
axiom length_CD : ∃ h : ℝ, true

-- Main theorem statement
theorem area_of_shaded_region (h : ℝ) : (∃ r x : ℝ, r^2 - x^2 = h^2) → (S : ℝ) (S = (π / 4) * h^2) := sorry

end area_of_shaded_region_l727_727686


namespace angle_MHC_l727_727923

variables {A B C H M : Type}

-- Define the angles and conditions
def triangle_ABC (α β γ : ℝ) := α + β + γ = 180
def angle_A (α: ℝ) := α = 100
def angle_B (β: ℝ) := β = 50
def angle_C (γ: ℝ) := γ = 30
def is_altitude (AH BC : Prop) := AH ∧ BC
def is_median (BM AC : Prop) := BM ∧ AC

-- Theorem statement
theorem angle_MHC (α β γ: ℝ) (AH BC BM AC : Prop) : 
  triangle_ABC α β γ →
  angle_A α →
  angle_B β →
  angle_C γ →
  is_altitude AH BC →
  is_median BM AC →
  ∠ M H C = 30 :=
by
  sorry

end angle_MHC_l727_727923


namespace radius_of_kn_l727_727143

theorem radius_of_kn {n : ℕ} (h_pos : 0 < n) : 
  ∃ r : ℝ, ∀ k k1 : ℝ, k = 1 ∧ k1 = 1 ∧ (∀ i, 2 ≤ i → (k + (i-1)^(-2)).sqrt + (k + i^(-2)).sqrt = 2 * (k.sqrt + (i-1)^(-2).sqrt)) → r = (n^(-2) : ℝ) :=
sorry

end radius_of_kn_l727_727143


namespace red_peaches_in_each_basket_l727_727301

theorem red_peaches_in_each_basket :
  ∃ (R : ℕ), (R * 11) + (18 * 11) = 308 ∧ R = 10 :=
by
  use 10
  split
  . calc (10 * 11) + (18 * 11) = 110 + 198 := by simp [mul_comm]
                           ... = 308 := by simp
  . rfl

end red_peaches_in_each_basket_l727_727301


namespace evaluate_expression_at_values_l727_727785

theorem evaluate_expression_at_values 
  (x y z : ℝ) 
  (h1 : x = 2) 
  (h2 : y = -1) 
  (h3 : z = 3) 
  : x^2 + y^2 - 3*z^2 + 2*x*y + 2*y*z - 2*x*z = -44 :=
by 
  subst h1 
  subst h2
  subst h3
  have h : (2 : ℝ)^2 + (-1 : ℝ)^2 - 3*(3 : ℝ)^2 + 2*(2 : ℝ)*(-1 : ℝ) + 2*(-1 : ℝ)*(3 : ℝ) - 2*(2 : ℝ)*(3 : ℝ) = -44,
  by norm_num,
  exact h,

end evaluate_expression_at_values_l727_727785


namespace shortest_chord_length_correct_max_area_triangle_OMN_correct_l727_727097

noncomputable def shortest_chord_length (a : ℝ) (h : 0 < a ∧ a < 1) : ℝ := 2 * Real.sqrt (1 - a^2)

noncomputable def max_area_triangle_OMN (a : ℝ) (h : 0 < a ∧ a < 1) : ℝ :=
  if a ≥ Real.sqrt 2 / 2 then 1 / 2 else a * Real.sqrt (1 - a^2)

theorem shortest_chord_length_correct (a : ℝ) (h : 0 < a ∧ a < 1) :
  ∃ MN : ℝ, MN = shortest_chord_length a h := by
  use shortest_chord_length a h
  sorry

theorem max_area_triangle_OMN_correct (a : ℝ) (h : 0 < a ∧ a < 1) :
  ∃ max_area : ℝ, max_area = max_area_triangle_OMN a h := by
  use max_area_triangle_OMN a h
  sorry

end shortest_chord_length_correct_max_area_triangle_OMN_correct_l727_727097


namespace knight_probability_sum_l727_727647

def num_knights := 30
def chosen_knights := 4

-- Calculate valid placements where no knights are adjacent
def valid_placements : ℕ := 26 * 24 * 22 * 20
-- Calculate total unrestricted placements
def total_placements : ℕ := 26 * 27 * 28 * 29
-- Calculate probability
def P : ℚ := 1 - (valid_placements : ℚ) / total_placements

-- Simplify the fraction P to its lowest terms: 553/1079
def simplified_num := 553
def simplified_denom := 1079

-- Sum of the numerator and denominator of simplified P
def sum_numer_denom := simplified_num + simplified_denom

theorem knight_probability_sum :
  sum_numer_denom = 1632 :=
by
  -- Proof is omitted
  sorry

end knight_probability_sum_l727_727647


namespace proof_problem_l727_727322

-- Define the set and the functions
def my_set : Set ℕ := {0, 1, 2}

def f_b (x : ℝ) := x
def g_b (x : ℝ) := (x ^ 3) ^ (1/3 : ℝ)

-- Define an odd function defined on ℝ
def f_odd (x : ℝ) : ℝ := - f_odd (-x)

-- Theorem stating that options B and D are correct, and options A and C are incorrect
theorem proof_problem :
  (¬ (2 ^ my_set.card - 1 = 8)) ∧
  (∀ x : ℝ, f_b x = g_b x) ∧
  (¬ (∀ f : (ℝ → ℝ), (f 2 < f 3) → ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2)) ∧
  (f_odd 0 = 0) :=
by
  sorry

end proof_problem_l727_727322


namespace find_lambda_l727_727467

variables (E : Type) [add_comm_group E] [module ℝ E]
variables (e1 e2 mn np : E)

-- Non-collinear condition
variables (h_non_collinear : ¬ collinear ℝ ({e1, e2} : set E))

-- Given vectors
variables (h_mn : mn = 2 • e1 - 3 • e2)
variables (h_np : np = λ e1 + 6 • e2)
variables {M N P : E}

-- Collinearity condition
variable (h_collinear : collinear ℝ ({M, N, P} : set E))

theorem find_lambda : λ = -4 :=
by
  sorry

end find_lambda_l727_727467


namespace original_chocolates_l727_727333

theorem original_chocolates (N : ℕ) 
  (H_nuts : N / 2)
  (H_eaten_nuts : 0.8 * (N / 2) = 0.4 * N)
  (H_without_nuts : N / 2)
  (H_eaten_without : 0.5 * (N / 2) = 0.25 * N)
  (H_left : N - (0.4 * N + 0.25 * N) = 28) : 
  N = 80 :=
sorry

end original_chocolates_l727_727333


namespace matrix_inverse_exists_and_correct_l727_727431

theorem matrix_inverse_exists_and_correct : 
  let A := matrix ([[3, 4], [-2, 9]] : matrix (fin 2) (fin 2) ℚ)
  let detA := (3 * 9) - (4 * -2)
  detA ≠ 0 →
  matrix.inv A = matrix ([[9/35, -4/35], [2/35, 3/35]] : matrix (fin 2) (fin 2) ℚ) :=
by
  let A := matrix ([[3, 4], [-2, 9]] : matrix (fin 2) (fin 2) ℚ)
  let detA := (3 * 9) - (4 * -2)
  have detA_nz : detA ≠ 0 := by simp [detA]
  have invA := (matrix.inv_of_det_ne_zero _ detA_nz)
  have matrix_inv_eq := (invA A)
  let expected_inv := (matrix ([[9/35, -4/35], [2/35, 3/35]] : matrix (fin 2) (fin 2) ℚ))
  exact matrix_inv_eq = expected_inv
  sorry

end matrix_inverse_exists_and_correct_l727_727431


namespace max_people_working_no_families_like_singing_l727_727682

variable (total_people : ℕ) (not_working : ℕ) (have_families : ℕ) (like_sing_shower : ℕ)

def working := total_people - not_working
def no_families := total_people - have_families
def max_possible_people := min working (min no_families like_sing_shower)

theorem max_people_working_no_families_like_singing (h_total : total_people = 100)
    (h_not_working : not_working = 50)
    (h_have_families : have_families = 25)
    (h_like_sing_shower : like_sing_shower = 75) :
  max_possible_people total_people not_working have_families like_sing_shower = 50 := 
by
  sorry

end max_people_working_no_families_like_singing_l727_727682


namespace find_m_l727_727844

theorem find_m (m : ℝ) : (∃ (m : ℝ), let A : ℝ × ℝ := (1, -2)
                              let B : ℝ × ℝ := (m, 2)
                              let midpoint := (λ (p1 p2 : ℝ × ℝ), ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)) A B
                              (midpoint.1 + 2 * midpoint.2 - 2 = 0) ) → m = 3 :=
by
  sorry

end find_m_l727_727844


namespace four_digit_numbers_not_multiple_of_3_or_4_l727_727153

theorem four_digit_numbers_not_multiple_of_3_or_4 :
  (finset.Icc 1000 9999).card - 
  ((finset.filter (λ x, x % 3 = 0) (finset.Icc 1000 9999)).card + 
  (finset.filter (λ x, x % 4 = 0) (finset.Icc 1000 9999)).card - 
  (finset.filter (λ x, x % 12 = 0) (finset.Icc 1000 9999)).card) = 4500 :=
by
  sorry

end four_digit_numbers_not_multiple_of_3_or_4_l727_727153


namespace initial_position_is_1963_l727_727388

theorem initial_position_is_1963 :
  ∃ (K_0 : ℤ), (K_0 + (∑ i in range 100, (-1)^(i+1) * (i+1))) = 2013 ∧ K_0 = 1963 :=
begin
  use 1963,
  split,
  {
    -- Prove that 1963 + the sum of the jumps equals 2013
    have h1 : ∑ i in range 100, (-1)^(i+1) * (i+1) = 50,
    { sorry },
    calc 1963 + ∑ i in range 100, (-1)^(i+1) * (i+1)
        = 1963 + 50 : by rw h1
    ... = 2013 : by norm_num,
  },
  {
    -- State that K_0 is 1963, which is the correct answer
    refl,
  },
end

end initial_position_is_1963_l727_727388


namespace four_digit_numbers_not_multiples_of_3_or_4_l727_727154

theorem four_digit_numbers_not_multiples_of_3_or_4 : 
  {x : ℕ | 1000 ≤ x ∧ x ≤ 9999 ∧ ¬ (x % 3 = 0 ∨ x % 4 = 0)}.card = 4500 := 
begin
  sorry
end

end four_digit_numbers_not_multiples_of_3_or_4_l727_727154


namespace purely_imaginary_z_eq_a2_iff_a2_l727_727511

theorem purely_imaginary_z_eq_a2_iff_a2 (a : Real) : 
(∃ (b : Real), a^2 - a - 2 = 0 ∧ a + 1 ≠ 0) → a = 2 :=
by
  sorry

end purely_imaginary_z_eq_a2_iff_a2_l727_727511


namespace coin_value_difference_l727_727984

theorem coin_value_difference :
  ∃ (p n d : ℕ), p + n + d = 3030 ∧ p ≥ 1 ∧ n ≥ 1 ∧ d ≥ 1 ∧
  (let max_val := 1 + 5 * 1 + 10 * (3030 - 1 - 1) in
   let min_val := 3030 + 5 * 1 + 10 * 1 in
   max_val - min_val = 27243) := by
{
  sorry
}

end coin_value_difference_l727_727984


namespace sqrt_720_l727_727993

theorem sqrt_720 : sqrt (720) = 12 * sqrt (5) :=
sorry

end sqrt_720_l727_727993


namespace probability_foma_wait_time_l727_727745

def interval := {x : ℝ // 2 < x ∧ x < 10}
def arrived (x y : interval) := x.val < y.val 

noncomputable def event_happens (x y : ℝ) : Prop := y ≤ x + 4

theorem probability_foma_wait_time :
  (∑(x : interval), ∑(y : interval), arrived x y ∧ event_happens x.val y.val) /
  (∑(x : interval), ∑(y : interval), arrived x y) = 0.75 := 
sorry

end probability_foma_wait_time_l727_727745


namespace length_of_AB_l727_727596

noncomputable def midpoint (x y : ℝ) := (x + y) / 2

theorem length_of_AB 
    (A B C D E F G : ℝ)
    (h1 : C = midpoint A B) 
    (h2 : D = midpoint A C)
    (h3 : E = midpoint A D) 
    (h4 : F = midpoint A E) 
    (h5 : G = midpoint A F)
    (h6 : abs (A - G) = 12) : 
    abs (A - B) = 384 := 
sorry

end length_of_AB_l727_727596


namespace row_swap_works_l727_727433

open Matrix

noncomputable def rowSwapMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![0, 1],
    ![1, 0]
  ]

theorem row_swap_works (a b c d : ℝ) :
  let M := ![
    ![a, b],
    ![c, d]
  ] in (rowSwapMatrix ⬝ M) = ![
    ![c, d],
    ![a, b]
  ] :=
by
  sorry

end row_swap_works_l727_727433


namespace find_PB_l727_727903

variables (A B C D P : Type)
variables [affine_space A] [affine_space B] [affine_space C] [affine_space D] [affine_space P]
variables (AB CD AD BC : line)
variables (CD_perpendicular_AB : perpendicular CD AB)
variables (BC_perpendicular_AD : perpendicular BC AD)
variables (CD_length : length CD = 52)
variables (BC_length : length BC = 35)
variables (line_through_B_perpendicular_to_AD : exists P, perpendicular (line_through B P) AD)
variables (AP_length : length (segment A P) = 10)

theorem find_PB : 
  exists PB : ℝ, PB = 39 :=
by sorry

end find_PB_l727_727903


namespace B_greater_than_A_l727_727410

def A := (54 : ℚ) / (5^7 * 11^4 : ℚ)
def B := (55 : ℚ) / (5^7 * 11^4 : ℚ)

theorem B_greater_than_A : B > A := by
  sorry

end B_greater_than_A_l727_727410


namespace sum_rational_product_rational_sum_and_product_rational_l727_727323

-- 1. Prove that the sum of 3 + sqrt(5) and 7 - sqrt(5) is rational
theorem sum_rational (a b : ℝ) (ha : a = 3 + sqrt 5) (hb : b = 7 - sqrt 5) : ∃ q : ℚ, a + b = q := by
  sorry

-- 2. Prove that the product of sqrt(11) + sqrt(7) and 2 * sqrt(11) - 2 * sqrt(7) is rational
theorem product_rational (a b : ℝ) (ha : a = sqrt 11 + sqrt 7) (hb : b = 2 * sqrt 11 - 2 * sqrt 7) : ∃ q : ℚ, a * b = q := by
  sorry

-- 3. Prove that both the sum and product of 2 + sqrt(3) and 2 - sqrt(3) are rational
theorem sum_and_product_rational (a b : ℝ) (ha : a = 2 + sqrt 3) (hb : b = 2 - sqrt 3) : 
  (∃ q_sum : ℚ, a + b = q_sum) ∧ (∃ q_prod : ℚ, a * b = q_prod) := by
  sorry

end sum_rational_product_rational_sum_and_product_rational_l727_727323


namespace find_p_l727_727715

-- Definitions based on the conditions
def parabola (p : ℝ) : Set (ℝ × ℝ) := { point | point.1^2 = 2 * p * point.2 }
def line_through_focus (p : ℝ) : Set (ℝ × ℝ) := { point | point.2 = point.1 + p / 2 }
def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)
def area_trapezoid (A B : ℝ × ℝ) : ℝ :=
  let D : ℝ × ℝ := (A.1, 0)
  let C : ℝ × ℝ := (B.1, 0)
  (1/2 : ℝ) * ((A.2 + B.2) * (B.1 - A.1))

-- Hypotheses
variable {p : ℝ}
hypothesis (h_p_pos : p > 0)
hypothesis (h_area : area_trapezoid (A : ℝ × ℝ) (B : ℝ × ℝ) = 12 * Real.sqrt 2)

-- Theorem
theorem find_p
  (h_focus : focus p ∈ parabola p)
  (h_A_on_line : A ∈ line_through_focus p)
  (h_A_on_parabola : A ∈ parabola p)
  (h_B_on_line : B ∈ line_through_focus p)
  (h_B_on_parabola : B ∈ parabola p)
  (h_A_not_B : A ≠ B) :
  p = 2 :=
sorry

end find_p_l727_727715


namespace correct_biological_statement_l727_727386

theorem correct_biological_statement
  (A : ¬ (Triploid plants can develop from a fertilized egg))
  (B : ¬ (By observing the morphology of chromosomes during the metaphase of mitosis, the location of gene mutations can be determined))
  (C : During mitosis and meiosis, chromosome structural variations can occur due to the exchange of segments between non-homologous chromosomes)
  (D : ¬ (Low temperatures inhibit the division of chromosome centromeres, causing sister chromatids to not move towards the poles, leading to chromosome doubling)) :
  C := 
sorry

end correct_biological_statement_l727_727386


namespace new_galleries_receive_two_pictures_l727_727373

theorem new_galleries_receive_two_pictures :
  ∀ (total_pencils : ℕ) (orig_gallery_pictures : ℕ) (new_galleries : ℕ) 
    (pencils_per_picture : ℕ) (signature_pencils_per_gallery : ℕ),
    (total_pencils = 88) →
    (orig_gallery_pictures = 9) →
    (new_galleries = 5) →
    (pencils_per_picture = 4) →
    (signature_pencils_per_gallery = 2) →
    let orig_gallery_signature_pencils := signature_pencils_per_gallery in
    let total_galleries := 1 + new_galleries in
    let total_signature_pencils := total_galleries * signature_pencils_per_gallery in
    let total_drawing_pencils := total_pencils - total_signature_pencils in
    let orig_gallery_drawing_pencils := orig_gallery_pictures * pencils_per_picture in
    let new_galleries_drawing_pencils := total_drawing_pencils - orig_gallery_drawing_pencils in
    let total_new_gallery_pictures := new_galleries_drawing_pencils / pencils_per_picture in
    let pictures_per_new_gallery := total_new_gallery_pictures / new_galleries in
    pictures_per_new_gallery = 2 :=
begin
  intros total_pencils orig_gallery_pictures new_galleries pencils_per_picture signature_pencils_per_gallery
          H_total_pencils H_orig_gallery_pictures H_new_galleries H_pencils_per_picture H_signature_pencils_per_gallery,

  let orig_gallery_signature_pencils := signature_pencils_per_gallery,
  let total_galleries := 1 + new_galleries,
  let total_signature_pencils := total_galleries * signature_pencils_per_gallery,
  let total_drawing_pencils := total_pencils - total_signature_pencils,
  let orig_gallery_drawing_pencils := orig_gallery_pictures * pencils_per_picture,
  let new_galleries_drawing_pencils := total_drawing_pencils - orig_gallery_drawing_pencils,
  let total_new_gallery_pictures := new_galleries_drawing_pencils / pencils_per_picture,
  let pictures_per_new_gallery := total_new_gallery_pictures / new_galleries,

  exact sorry
end

end new_galleries_receive_two_pictures_l727_727373


namespace cubes_with_odd_neighbors_in_5x5x5_l727_727449

theorem cubes_with_odd_neighbors_in_5x5x5 (unit_cubes : Fin 125 → ℕ) 
  (neighbors : ∀ (i : Fin 125), Fin 125 → Prop) : ∃ n, n = 62 := 
by
  sorry

end cubes_with_odd_neighbors_in_5x5x5_l727_727449


namespace laptop_price_reduction_l727_727360

-- Conditions definitions
def initial_price (P : ℝ) : ℝ := P
def seasonal_sale (P : ℝ) : ℝ := 0.7 * P
def special_promotion (seasonal_price : ℝ) : ℝ := 0.8 * seasonal_price
def clearance_event (promotion_price : ℝ) : ℝ := 0.9 * promotion_price

-- Proof statement
theorem laptop_price_reduction (P : ℝ) (h1 : seasonal_sale P = 0.7 * P) 
    (h2 : special_promotion (seasonal_sale P) = 0.8 * (seasonal_sale P)) 
    (h3 : clearance_event (special_promotion (seasonal_sale P)) = 0.9 * (special_promotion (seasonal_sale P))) : 
    (initial_price P - clearance_event (special_promotion (seasonal_sale P))) / (initial_price P) = 0.496 := 
by 
  sorry

end laptop_price_reduction_l727_727360


namespace largest_prime_k_l727_727404

-- Define the sequence x
def x : ℕ → ℕ
| 1 := 2
| (n+1) := 2 * (x n) + 1

-- Define the sequence y based on the sequence x
def y (n : ℕ) : ℕ := 2 ^ (x n) - 1

-- Define the main theorem
theorem largest_prime_k : ∀ k, (∀ i, 1 ≤ i ∧ i ≤ k → (Nat.Prime (y i))) → k ≤ 2 := 
by
  sorry

end largest_prime_k_l727_727404


namespace sqrt_720_l727_727996

theorem sqrt_720 : sqrt (720) = 12 * sqrt (5) :=
sorry

end sqrt_720_l727_727996


namespace linear_dependency_k_val_l727_727289

theorem linear_dependency_k_val (k : ℝ) :
  (∃ (c1 c2 : ℝ), (c1 ≠ 0 ∨ c2 ≠ 0) ∧ 2 * c1 + 4 * c2 = 0 ∧ 3 * c1 + k * c2 = 0) ↔ k = 6 :=
by sorry

end linear_dependency_k_val_l727_727289


namespace radius_of_circle_B_l727_727752

-- Define the conditions
structure Circle (center : Type*) (radius : ℝ) :=
(center : center)
(radius : ℝ)

variables {A B C D E : Circle ℝ}
variable {r : ℝ}

-- Conditions given in the problem
axiom tangent_ABC : ∀ (O : Circle ℝ), ∃ (K L M : ℝ), Circle K 2 ∧ Circle L 2 ∧ Circle M 2
axiom tangent_BCD : B.radius = C.radius ∧ B = D
axiom congruent_BC : B.radius = C.radius
axiom radius_A : A.radius = 2
axiom circle_E : A = E ∧ E = D ∧ E.center = D.center

-- Prove the radius of circle B
theorem radius_of_circle_B (r : ℝ) : 
∃ y : ℝ, y = ( -1 + real.sqrt (1 + 2 * r) + (( -1 + real.sqrt (1 + 2 * r))^2 / 4)) / 4 := sorry

end radius_of_circle_B_l727_727752


namespace train_pass_man_approx_9_seconds_l727_727369

noncomputable def time_to_pass_man (train_length : ℕ) (train_speed_man_speed_sum : ℕ) : ℝ :=
  train_length / ((train_speed_man_speed_sum * (5 / 18) : ℝ))

theorem train_pass_man_approx_9_seconds :
  let train_length := 165
  let train_speed := 60
  let man_speed := 6
  let relative_speed := train_speed + man_speed
  time_to_pass_man train_length relative_speed ≈ 9 :=
by
  let train_length := 165
  let train_speed := 60
  let man_speed := 6
  let relative_speed := train_speed + man_speed
  have relative_speed_ms : ℝ := relative_speed * (5 / 18)
  have time_sec : ℝ := train_length / relative_speed_ms
  show time_sec ≈ 9,
  sorry

end train_pass_man_approx_9_seconds_l727_727369


namespace concurrent_lines_l727_727372

theorem concurrent_lines 
  (A B C B' C' H H' : Type) [triangle A B C]
  (circ : circle B C) (CC' : B ∈ circ) (CB' : C' ∈ circ)
  (intersect_Cpr : circle.intersect (line A C B) = C') 
  (intersect_Bpr : circle.intersect (line A B C) = B')
  (orthocenter_H : orthocenter A B C = H)
  (orthocenter_H' : orthocenter A B' C' = H') :
  isConcurrent (line B B') (line C C') (line H H') :=
sorry

end concurrent_lines_l727_727372


namespace addition_of_zeros_resulting_in_perfect_square_l727_727981

theorem addition_of_zeros_resulting_in_perfect_square :
  ∃ n : ℕ, 
    let num := 10^199 - 10^100 in
    n = Nat.floor (10^99 * Real.sqrt 10) ∧
    num < n^2 ∧ n^2 < 10^199 :=
by 
  sorry

end addition_of_zeros_resulting_in_perfect_square_l727_727981


namespace area_lune_correct_l727_727723

/-- Define the diameters of the semicircles -/
def d_smaller := 3
def d_larger := 4

/-- Define the radii of the semicircles -/
def r_smaller : ℚ := d_smaller / 2
def r_larger : ℚ := d_larger / 2

/-- Define the area of a semicircle given its radius -/
def area_semicircle (r : ℚ) : ℚ := (1 / 2) * π * r^2

/-- The area of the smaller semicircle -/
def area_smaller := area_semicircle r_smaller

/-- The area of the larger semicircle -/
def area_larger := area_semicircle r_larger

/-- The area of the sector of the larger semicircle that overlaps with the smaller semicircle -/
def angle_overlap := π * (1 / 3) -- 60 degrees in radians as fraction of 2π
def area_overlap := (angle_overlap / (2 * π)) * area_larger

/-- The area of the lune formed by the smaller semicircle sitting atop the larger semicircle -/
def area_lune := area_smaller - area_overlap

theorem area_lune_correct : area_lune = (11 / 24) * π := by
  sorry

end area_lune_correct_l727_727723


namespace sqrt_720_simplified_l727_727988

theorem sqrt_720_simplified : (sqrt 720 = 12 * sqrt 5) :=
by
  -- The proof is omitted as per the instructions
  sorry

end sqrt_720_simplified_l727_727988


namespace symmetric_bf_contains_median_l727_727922

open EuclideanGeometry

-- Lean 4 statement

theorem symmetric_bf_contains_median (A B C D E F : Point)
    (h_triangle : triangle A B C)
    (h_bisector_bd : angle_bisector B D A C)
    (h_d_on_ac : LineSegment D A C)
    (h_bd_intersects_omega : intersects (LineSegment B D) (circumcircle A B C) B E)
    (h_circle_omega : diameter E D)
    (h_omega_intersects_omega : intersects (circle E D) (circumcircle A B C) E F)
    : symmetric_line (Line B F) (LineSegment B D) contains (median_of_triangle A B C) := 
sorry

end symmetric_bf_contains_median_l727_727922


namespace solution_set_of_inequalities_l727_727948

-- Definitions
def is_even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x

def is_strictly_decreasing (f : ℝ → ℝ) (a b : ℝ) := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- Main Statement
theorem solution_set_of_inequalities
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_periodic : is_periodic f 2)
  (h_decreasing : is_strictly_decreasing f 0 1)
  (h_f_pi : f π = 1)
  (h_f_2pi : f (2 * π) = 2) :
  {x : ℝ | 1 ≤ x ∧ x ≤ 2 ∧ 1 ≤ f x ∧ f x ≤ 2} = {x | π - 2 ≤ x ∧ x ≤ 8 - 2 * π} :=
  sorry

end solution_set_of_inequalities_l727_727948


namespace equidistant_points_count_l727_727577

noncomputable def number_of_equidistant_points (l1 l2 l3 : Line) : ℕ :=
  if ((intersects l2 l1) ∧ (intersects l2 l3) ∧ (is_parallel l3 l1) 
      ∧ (distinct_lines l1 l2 l3) ∧ (coplanar l1 l2 l3))
  then 2
  else 0

theorem equidistant_points_count (l1 l2 l3 : Line) 
  (h_intersects_l2_l1 : intersects l2 l1)
  (h_intersects_l2_l3 : intersects l2 l3)
  (h_is_parallel_l3_l1 : is_parallel l3 l1)
  (h_distinct : distinct_lines l1 l2 l3)
  (h_coplanar : coplanar l1 l2 l3) 
  : number_of_equidistant_points l1 l2 l3 = 2 := by
  sorry

end equidistant_points_count_l727_727577


namespace vector_perpendicular_l727_727890

variables (a b c : EuclideanSpace ℝ (Fin 3))
variables (angle_ab : Real)
variables (norm_a norm_b : ℝ)

-- Given conditions
def condition1 : Prop := angle_ab = 120
def condition2 : Prop := ∥a∥ = 1
def condition3 : Prop := ∥b∥ = 2
def condition4 : Prop := c = a + b

-- Proof statement
theorem vector_perpendicular
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3)
  (h4 : condition4) : c ⬝ a = 0 := sorry

end vector_perpendicular_l727_727890


namespace eval_expression_l727_727784

theorem eval_expression : (8^(-3) ^ 0) + (2 ^ 0) ^ 4 = 2 := by
  sorry

end eval_expression_l727_727784


namespace molecular_weight_4_benzoic_acid_l727_727667

def benzoic_acid_molecular_weight : Float := (7 * 12.01) + (6 * 1.008) + (2 * 16.00)

def molecular_weight_4_moles_benzoic_acid (molecular_weight : Float) : Float := molecular_weight * 4

theorem molecular_weight_4_benzoic_acid :
  molecular_weight_4_moles_benzoic_acid benzoic_acid_molecular_weight = 488.472 :=
by
  unfold molecular_weight_4_moles_benzoic_acid benzoic_acid_molecular_weight
  -- rest of the proof
  sorry

end molecular_weight_4_benzoic_acid_l727_727667


namespace midpoint_check_l727_727666

theorem midpoint_check :
  let A := (10, -8)
      B := (-4, 6)
      M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  in M = (3, -1) ∧ ¬ (M.2 = -2 * M.1 + 1) :=
by
  let A := (10, -8)
  let B := (-4, 6)
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  show M = (3, -1) ∧ ¬ (M.2 = -2 * M.1 + 1)
  sorry

end midpoint_check_l727_727666


namespace salem_size_comparison_l727_727605

theorem salem_size_comparison (S L : ℕ) (hL: L = 58940)
  (hSalem: S - 130000 = 2 * 377050) :
  (S / L = 15) :=
sorry

end salem_size_comparison_l727_727605


namespace aria_cookies_per_day_l727_727081

theorem aria_cookies_per_day 
  (cost_per_cookie : ℕ)
  (total_amount_spent : ℕ)
  (days_in_march : ℕ)
  (h_cost : cost_per_cookie = 19)
  (h_spent : total_amount_spent = 2356)
  (h_days : days_in_march = 31) : 
  (total_amount_spent / cost_per_cookie) / days_in_march = 4 :=
by
  sorry

end aria_cookies_per_day_l727_727081


namespace probability_complement_l727_727047

variables (A B C : Set ℝ)
variables (P : Set ℝ → ℝ)

-- Given conditions
def P_A : ℝ := 0.15
def P_B : ℝ := 0.40
def P_C : ℝ := 0.30
def P_A_and_B : ℝ := 0.06
def P_A_and_C : ℝ := 0.04
def P_B_and_C : ℝ := 0.12
def P_A_and_B_and_C : ℝ := 0.02

-- Definitions to prove
def P_union (A B C : Set ℝ) :=
  P A + P B + P C - P_A_and_B - P_A_and_C - P_B_and_C + P_A_and_B_and_C

def P_complement_union (A B C : Set ℝ) :=
  1 - P_union A B C

-- Theorem to prove
theorem probability_complement (P_A := 0.15)
                               (P_B := 0.40)
                               (P_C := 0.30)
                               (P_A_and_B := 0.06)
                               (P_A_and_C := 0.04)
                               (P_B_and_C := 0.12)
                               (P_A_and_B_and_C := 0.02) :
  P_complement_union A B C = 0.37 :=
by {
  unfold P_complement_union P_union,
  -- Skipping the proof as requested
  sorry
}

end probability_complement_l727_727047


namespace relationship_between_a_b_c_l727_727456

noncomputable def a : ℝ := 0.2 ^ 1.5
noncomputable def b : ℝ := 2.0 ^ 0.1
noncomputable def c : ℝ := 0.2 ^ 1.3

theorem relationship_between_a_b_c : a < c ∧ c < b :=
by
  -- The proof goes here
  sorry

end relationship_between_a_b_c_l727_727456


namespace log_power_equality_l727_727110

variable (a m n : ℝ)

def log_condition_1 := log a 2 = m
def log_condition_2 := log a 3 = n

theorem log_power_equality : log_condition_1 a m n → log_condition_2 a m n → a^(2 * m + n) = 12 := by
  sorry

end log_power_equality_l727_727110


namespace determine_percentage_of_second_mixture_l727_727348

-- Define the given conditions and question
def mixture_problem (P : ℝ) : Prop :=
  ∃ (V1 V2 : ℝ) (A1 A2 A_final : ℝ),
  V1 = 2.5 ∧ A1 = 0.30 ∧
  V2 = 7.5 ∧ A2 = P / 100 ∧
  A_final = 0.45 ∧
  (V1 * A1 + V2 * A2) / (V1 + V2) = A_final

-- State the theorem
theorem determine_percentage_of_second_mixture : mixture_problem 50 := sorry

end determine_percentage_of_second_mixture_l727_727348


namespace range_of_a_l727_727873

theorem range_of_a (a : ℝ) : (-1 < a ∧ a < 5) ↔ (∃ x : ℝ, (1 < x ∧ x < 3) ∧ ((a - 2) < x ∧ x < (a + 2))) :=
by
  split
  { -- Prove that (-1 < a ∧ a < 5) implies (∃ x : ℝ, (1 < x ∧ x < 3) ∧ ((a - 2) < x ∧ x < (a + 2)))
    intro h
    use (2 : ℝ)
    sorry }
  { -- Prove that (∃ x : ℝ, (1 < x ∧ x < 3) ∧ ((a - 2) < x ∧ x < (a + 2))) implies (-1 < a ∧ a < 5)
    intro h
    sorry }

end range_of_a_l727_727873


namespace ratio_of_c_to_b_l727_727078

noncomputable def ratio_c_to_b (a b c : ℝ) : ℝ :=
  c / b

theorem ratio_of_c_to_b (a b c : ℝ) 
  (h1 : a ≠ b)
  (h2 : a ≠ 2 * b)
  (h3 : b ≠ 2 * b)
  (h4 : b ≠ c)
  (h5 : a ≠ c)
  (h6 : segments_lengths : ∀ (l : ℝ), l ∈ {a, a, a, b, b, b, 2*b, c}) :
  ratio_c_to_b a b c = Real.sqrt 2 := 
by 
  sorry

end ratio_of_c_to_b_l727_727078


namespace A_takes_38_4_days_to_finish_alone_l727_727636

-- Definitions for the conditions
def capacity_A (x : ℝ) := 5 * x
def capacity_B (x : ℝ) := 3 * x
def combined_capacity (x : ℝ) := capacity_A x + capacity_B x
def total_work (x : ℝ) := combined_capacity x * 24
def days_A_alone (x : ℝ) := total_work x / (capacity_A x)

-- Theorem statement
theorem A_takes_38_4_days_to_finish_alone (x : ℝ) : 
  days_A_alone x = 38.4 :=
sorry

end A_takes_38_4_days_to_finish_alone_l727_727636


namespace value_of_f_at_1_l727_727130

def f (x : ℝ) (m : ℝ) : ℝ := 4 * x ^ 2 - m * x + 5

theorem value_of_f_at_1 :
  ∀ m, (∀ x, x ≥ -2 → deriv (f x m) ≥ 0) ∧ (∀ x, x ≤ -2 → deriv (f x m) ≤ 0) → f 1 (-16) = 25 :=
by sorry

end value_of_f_at_1_l727_727130


namespace remaining_battery_life_l727_727968

theorem remaining_battery_life (standby_hours : ℝ) (active_hours : ℝ)
  (total_hours : ℝ) (active_usage_hours : ℝ) (remaining_hours : ℝ) :
  (standby_hours = 30) →
  (active_hours = 4) →
  (total_hours = 12) →
  (active_usage_hours = 1.5) →
  (remaining_hours = 8.25) :=
begin
  sorry
end

end remaining_battery_life_l727_727968


namespace average_charge_per_person_l727_727311

theorem average_charge_per_person (x : ℝ) (h : x > 0) :
  (30 * x + 37.5 * x + 32.5 * x) / (2 * x + 5 * x + 13 * x) = 5 :=
by
  have hx : 20 * x ≠ 0 := by linarith
  calc
    (30 * x + 37.5 * x + 32.5 * x) / (2 * x + 5 * x + 13 * x)
      = 100 * x / 20 * x : by ring
      ... = 5 : by nlinarith

end average_charge_per_person_l727_727311


namespace exists_n_for_sin_l727_727601

theorem exists_n_for_sin (x : ℝ) (h : Real.sin x ≠ 0) :
  ∃ n : ℕ, |Real.sin (n * x)| ≥ Real.sqrt 3 / 2 :=
sorry

end exists_n_for_sin_l727_727601


namespace gcd_of_12012_and_18018_l727_727799

theorem gcd_of_12012_and_18018 : Int.gcd 12012 18018 = 6006 := 
by
  -- Here we are assuming the factorization given in the conditions
  have h₁ : 12012 = 12 * 1001 := sorry
  have h₂ : 18018 = 18 * 1001 := sorry
  have gcd_12_18 : Int.gcd 12 18 = 6 := sorry
  -- This sorry will be replaced by the actual proof involving the above conditions to conclude the stated theorem
  sorry

end gcd_of_12012_and_18018_l727_727799


namespace bell_rings_before_geography_l727_727933

-- Define the sequence of classes in the school
inductive Class
| Maths
| History
| Geography
| Science
| Music

-- Define a function that counts the number of bell rings up to (but not including) a specified class.
def bell_rings_up_to : Class → Nat
| Class.Maths := 1
| Class.History := 3
| Class.Geography := 5
| Class.Science := 7
| Class.Music := 9

-- Assertion about the bell rings up to Geography class
theorem bell_rings_before_geography :
  bell_rings_up_to Class.Geography = 5 := 
by
  -- Proof steps omitted
  sorry

end bell_rings_before_geography_l727_727933


namespace base10_to_base4_equivalence_l727_727314

theorem base10_to_base4_equivalence (n : ℕ) : 
  n = 258 → (n.to_digits 4) = [1, 0, 0, 0, 2] :=
by
  intro h
  rw h
  sorry

end base10_to_base4_equivalence_l727_727314


namespace range_of_f_lt_zero_l727_727850

noncomputable def f : ℝ → ℝ := sorry

theorem range_of_f_lt_zero (f_odd : ∀ x : ℝ, f x + f (-x) = 0)
    (f_decreasing_neg : ∀ x y : ℝ, x < y ∧ x < 0 ∧ y < 0 → f x > f y)
    (f_five_zero : f 5 = 0) :
    {x : ℝ | f x < 0} = set.Ioo (-5) 0 ∪ set.Ioi 5 :=
sorry

end range_of_f_lt_zero_l727_727850


namespace smallest_n_l727_727570

-- Define the set of all polynomials in three variables x, y, z with integer coefficients
def A := {f : ℚ[X, Y, Z] | ∀ a b c : ℚ, f.eval a b c ∈ ℤ}

-- Define the subset B of polynomials in A
def B : set (ℚ[X, Y, Z]) :=
  {f : ℚ[X, Y, Z] |
    ∃ P Q R : ℚ[X, Y, Z], 
    (x + y + z) * P + (x*y + y*z + z*x) * Q + x*y*z * R = f}

-- Define the monomial x^i y^j z^k
def monomial (i j k : ℕ) : ℚ[X, Y, Z] := X^i * Y^j * Z^k

-- Define the statement of the theorem
theorem smallest_n (n : ℕ) :
  (∀ i j k, i + j + k < n → monomial i j k ∉ B) ∧
  (∀ i j k, i + j + k ≥ n → monomial i j k ∈ B) → 
  n = 4 :=
sorry

end smallest_n_l727_727570


namespace balls_in_boxes_l727_727299

theorem balls_in_boxes (n : ℕ) (B : Fin n → ℕ)
  (h_moves_1 : ∀ i j : Fin n, i = 0 → 0 < B i → B (i + 1) = B (i + 1) + 1 → B i = B i - 1)
  (h_moves_2 : ∀ i j : Fin n, i = n-1 → 0 < B i → B (i - 1) = B (i + 1) + 1 → B i = B i - 1)
  (h_moves_3 : ∀ k : Fin n, 2 ≤ k ≤ n-1 → 1 < B k → B (k - 1) = B (k - 1) + 1 → B (k + 1) = B (k + 1) + 1 → B k = B k - 2)
  : ∃ f : Fin n → ℕ, (∀ i : Fin n, f i = 1) :=
by
  sorry

end balls_in_boxes_l727_727299


namespace Calculate_AF_l727_727477

open Real

noncomputable def EllipseRightFocusDirectrix (x y : ℝ) : Prop := (x^2 / 2) + y^2 = 1
noncomputable def RightFocus (F : Point) : Prop := EllipseRightFocusDirectrix F.x F.y
noncomputable def RightDirectrix (l : Line) : Prop := -- definition for line corresponding to directrix
noncomputable def PointOnLine (A : Point) (l : Line) : Prop := -- definition for point A on line l
noncomputable def IntersectionAF (AF : Segment) (C : Ellipse) (B : Point) : Prop := -- definition for intersection of AF and C
noncomputable def FAeqThreeFB (FA FB : Vector) : Prop := -- definition for FA = 3 * FB

theorem Calculate_AF
  (x y : ℝ)
  (F A B : Point)
  (l : Line)
  (AF FB : Segment)
  (h1 : EllipseRightFocusDirectrix x y)
  (h2 : RightFocus F)
  (h3 : RightDirectrix l)
  (h4 : PointOnLine A l)
  (h5 : IntersectionAF AF C B)
  (h6 : FAeqThreeFB (\overrightarrow{FA} \overrightarrow{FB}) ) :
  | \overrightarrow{FA}| = sqrt(2) := 
sorry

end Calculate_AF_l727_727477


namespace smallest_n_common_factor_l727_727668

theorem smallest_n_common_factor :
  ∃ n : ℕ, n > 0 ∧ (∀ d : ℕ, d > 1 → d ∣ (11 * n - 4) → d ∣ (8 * n - 5)) ∧ n = 15 :=
by {
  -- Define the conditions as given in the problem
  sorry
}

end smallest_n_common_factor_l727_727668


namespace complex_fraction_in_first_quadrant_l727_727909

-- Define a complex number
def complex_2_minus_i : ℂ := 2 - complex.I
def complex_1_minus_3i : ℂ := 1 - 3 * complex.I

-- Calculate the fraction and its real and imaginary parts
def complex_fraction : ℂ :=
  (complex_2_minus_i * (1 + 3 * complex.I)) /
  (complex_1_minus_3i * (1 + 3 * complex.I))

-- Assert that the complex number is in the first quadrant
theorem complex_fraction_in_first_quadrant :
  complex_fraction.re > 0 ∧ complex_fraction.im > 0 := by
  sorry

end complex_fraction_in_first_quadrant_l727_727909


namespace hyperbola_asymptote_60_deg_l727_727514

theorem hyperbola_asymptote_60_deg 
  (a : ℝ) (h : ∀ x y : ℝ, x^2 / a - y^2 / 9 = 1 → 
    y = sqrt 3 * x ∨ y = - sqrt 3 * x) : 
    a = 3 ∧ ∀ x y : ℝ, x^2 / 3 - y^2 / 9 = 1 := 
by 
  sorry

end hyperbola_asymptote_60_deg_l727_727514


namespace num_squares_lattice_points_no_non_axis_aligned_cube_lattice_points_l727_727343

-- Part (1) in Lean 4
theorem num_squares_lattice_points (n : ℕ) : 
  let lattice_points := {p : ℤ × ℤ | 1 ≤ p.1 ∧ p.1 ≤ n ∧ 1 ≤ p.2 ∧ p.2 ≤ n} in
  ∃ (count : ℕ), count = (n * (n + 1) * (2 * n + 1)) / 6 :=
  sorry

-- Part (2) in Lean 4
theorem no_non_axis_aligned_cube_lattice_points : 
  ¬ ∃ (v1 v2 v3 v4 v5 v6 v7 v8 : ℤ × ℤ × ℤ), 
    (distinct_vertices v1 v2 v3 v4 v5 v6 v7 v8) ∧ 
    (no_edges_parallel_to_axes v1 v2 v3 v4 v5 v6 v7 v8) :=
  sorry

-- Definitions for clarity
def distinct_vertices (v1 v2 v3 v4 v5 v6 v7 v8 : ℤ × ℤ × ℤ) : Prop := 
  v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v1 ≠ v5 ∧ v1 ≠ v6 ∧ v1 ≠ v7 ∧ v1 ≠ v8 ∧
  v2 ≠ v3 ∧ v2 ≠ v4 ∧ v2 ≠ v5 ∧ v2 ≠ v6 ∧ v2 ≠ v7 ∧ v2 ≠ v8 ∧
  v3 ≠ v4 ∧ v3 ≠ v5 ∧ v3 ≠ v6 ∧ v3 ≠ v7 ∧ v3 ≠ v8 ∧
  v4 ≠ v5 ∧ v4 ≠ v6 ∧ v4 ≠ v7 ∧ v4 ≠ v8 ∧
  v5 ≠ v6 ∧ v5 ≠ v7 ∧ v5 ≠ v8 ∧
  v6 ≠ v7 ∧ v6 ≠ v8 ∧
  v7 ≠ v8

def no_edges_parallel_to_axes (v1 v2 v3 v4 v5 v6 v7 v8 : ℤ × ℤ × ℤ) : Prop := 
  true -- substitute with the actual conditions for no edge being parallel to the axes

end num_squares_lattice_points_no_non_axis_aligned_cube_lattice_points_l727_727343


namespace largest_nat_n_inequality_l727_727817

theorem largest_nat_n_inequality (a b c d : ℝ) :
  ∃ n : ℕ, (n+2) * real.sqrt (a^2 + b^2) + (n+1) * real.sqrt (a^2 + c^2) +
  (n+1) * real.sqrt (a^2 + d^2) ≥ n * (a + b + c + d) :=
sorry

end largest_nat_n_inequality_l727_727817


namespace points_collinear_l727_727141

-- Definition of the input circles and the points of intersections of the external tangents:
variable (γ1 γ2 γ3 : Circle)
variable (X Y Z : Point)
variable (external_tangent_intersect : ∀ (C1 C2 : Circle), Point)
-- Assuming conditions as given:
axiom X_is_external_tangent_intersect : external_tangent_intersect γ1 γ2 = X
axiom Y_is_external_tangent_intersect : external_tangent_intersect γ2 γ3 = Y
axiom Z_is_external_tangent_intersect : external_tangent_intersect γ3 γ1 = Z

-- We need to show that X, Y, Z are collinear:
theorem points_collinear : collinear [X, Y, Z] := 
sorry

end points_collinear_l727_727141


namespace simplify_fraction_fraction_c_over_d_l727_727274

-- Define necessary constants and variables
variable (k : ℤ)

/-- Original expression -/
def original_expr := (6 * k + 12 + 3 : ℤ)

/-- Simplified expression -/
def simplified_expr := (2 * k + 5 : ℤ)

/-- The main theorem to prove the equivalent mathematical proof problem -/
theorem simplify_fraction : (original_expr / 3) = simplified_expr :=
by
  sorry

-- The final fraction to prove the answer
theorem fraction_c_over_d : (2 / 5 : ℚ) = 2 / 5 :=
by
  sorry

end simplify_fraction_fraction_c_over_d_l727_727274


namespace london_to_edinburgh_distance_l727_727319

theorem london_to_edinburgh_distance : ∃ D : ℝ, D = 393 ∧ 
  (D / 2 - 200 + 3.5 = 0) ∧
  (0 < 3.5 ∧ 0 < 200) :=
begin
  use 393,
  split,
  { refl, },
  split,
  { norm_num, },
  { split; norm_num, },
end

end london_to_edinburgh_distance_l727_727319


namespace min_chord_length_eq_l727_727854

-- Define the Circle C with center (1, 2) and radius 5
def isCircle (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + (y - 2) ^ 2 = 25

-- Define the Line l parameterized by m
def isLine (m x y : ℝ) : Prop :=
  (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

-- Prove that the minimal chord length intercepted by the circle occurs when the line l is 2x - y - 5 = 0
theorem min_chord_length_eq (x y : ℝ) : 
  (∀ m, isLine m x y → isCircle x y) → isLine 0 x y :=
sorry

end min_chord_length_eq_l727_727854


namespace mul_mixed_number_l727_727003

theorem mul_mixed_number (a b : ℝ) (c : ℚ) (h : c = 9 + 2/5) : 
  7 * c = (65 + 4/5 : ℝ) :=
by 
  simp [h, mul_add, mul_div_cancel_left, rat.cast_add, rat.cast_mul, rat.cast_one]
  sorry

end mul_mixed_number_l727_727003


namespace find_a_b_l727_727567

variable (x y : ℝ)
noncomputable def a : ℝ := Real.sqrt x + Real.sqrt y
noncomputable def b : ℝ := Real.sqrt (x + 2) + Real.sqrt (y + 2)

theorem find_a_b
  (hx : 0 ≤ x)
  (hy : 0 ≤ y)
  (ha_int : a x y ∈ Set.Ico 1 2 → False)
  (hb_int : b x y ∈ Set.Ioc 2 4)
  (h_diff : b x y - a x y ≥ 2)
  (int_a : a x y ∈ ℤ)
  (int_b : b x y ∈ ℤ) :
  a x y = 1 ∧ b x y = 3 :=
by
  sorry

end find_a_b_l727_727567


namespace initial_amount_l727_727206

theorem initial_amount 
  (spend1 spend2 left : ℝ)
  (hspend1 : spend1 = 1.75) 
  (hspend2 : spend2 = 1.25) 
  (hleft : left = 6.00) : 
  spend1 + spend2 + left = 9.00 := 
by
  -- Proof is omitted
  sorry

end initial_amount_l727_727206


namespace vieta_formula_l727_727942

-- Define what it means to be a root of a polynomial
noncomputable def is_root (p : ℝ) (a b c d : ℝ) : Prop :=
  a * p^3 + b * p^2 + c * p + d = 0

-- Setting up the variables and conditions for the polynomial
variables (p q r : ℝ)
variable (a b c d : ℝ)
variable (ha : a = 5)
variable (hb : b = -10)
variable (hc : c = 17)
variable (hd : d = -7)
variable (hp : is_root p a b c d)
variable (hq : is_root q a b c d)
variable (hr : is_root r a b c d)

-- Lean statement to prove the desired equality using Vieta's formulas
theorem vieta_formula : 
  pq + qr + rp = c / a :=
by
  -- Translate the problem into Lean structure
  sorry

end vieta_formula_l727_727942


namespace find_tony_age_l727_727307

variable (y : ℕ)
variable (d : ℕ)

def Tony_day_hours : ℕ := 3
def Tony_hourly_rate (age : ℕ) : ℚ := 0.75 * age
def Tony_days_worked : ℕ := 60
def Tony_total_earnings : ℚ := 945

noncomputable def earnings_before_birthday (age : ℕ) (days : ℕ) : ℚ :=
  Tony_hourly_rate age * Tony_day_hours * days

noncomputable def earnings_after_birthday (age : ℕ) (days : ℕ) : ℚ :=
  Tony_hourly_rate (age + 1) * Tony_day_hours * days

noncomputable def total_earnings (age : ℕ) (days_before : ℕ) : ℚ :=
  (earnings_before_birthday age days_before) +
  (earnings_after_birthday age (Tony_days_worked - days_before))

theorem find_tony_age: ∃ y d : ℕ, total_earnings y d = Tony_total_earnings ∧ y = 6 := by
  sorry

end find_tony_age_l727_727307


namespace find_u_l727_727270

variable (α β γ : ℝ)
variables (q s u : ℝ)

-- The first polynomial has roots α, β, γ
axiom roots_first_poly : ∀ x : ℝ, x^3 + 4 * x^2 + 6 * x - 8 = (x - α) * (x - β) * (x - γ)

-- Sum of the roots α + β + γ = -4
axiom sum_roots_first_poly : α + β + γ = -4

-- Product of the roots αβγ = 8
axiom product_roots_first_poly : α * β * γ = 8

-- The second polynomial has roots α + β, β + γ, γ + α
axiom roots_second_poly : ∀ x : ℝ, x^3 + q * x^2 + s * x + u = (x - (α + β)) * (x - (β + γ)) * (x - (γ + α))

theorem find_u : u = 32 :=
sorry

end find_u_l727_727270


namespace quadrilateral_area_correct_l727_727565

noncomputable def area_of_quadrilateral (n : ℕ) (hn : n > 0) : ℚ :=
  (2 * n^3) / (4 * n^2 - 1)

theorem quadrilateral_area_correct (n : ℕ) (hn : n > 0) :
  ∃ area : ℚ, area = (2 * n^3) / (4 * n^2 - 1) :=
by
  use area_of_quadrilateral n hn
  sorry

end quadrilateral_area_correct_l727_727565


namespace conversion_correct_l727_727764

-- Define the base 8 number
def base8_number : ℕ := 4 * 8^3 + 5 * 8^2 + 3 * 8^1 + 2 * 8^0

-- Define the target base 10 number
def base10_number : ℕ := 2394

-- The theorem that needs to be proved
theorem conversion_correct : base8_number = base10_number := by
  sorry

end conversion_correct_l727_727764


namespace polygon_sides_l727_727361

-- Define the conditions
def side_length : ℝ := 7
def perimeter : ℝ := 42

-- The statement to prove: number of sides is 6
theorem polygon_sides : (perimeter / side_length) = 6 := by
  sorry

end polygon_sides_l727_727361


namespace part1_part2_l727_727573

-- Definition of the function f and relevant propositions
def f (x a : ℝ) : ℝ := x^3 + a*x^2 + a*x

def p (a : ℝ) : Prop := ∀ x : ℝ, f' x a ≥ 0

def q (a m : ℝ) : Prop := |a - 1| ≤ m ∧ m > 0

-- Part 1: Prove p when a = 1
theorem part1 : p 1 := sorry

-- Part 2: Prove the range of m
theorem part2 (m : ℝ) : q a m → p a → q → 0 < m ∧ m < 1 := sorry

end part1_part2_l727_727573


namespace express_c_in_terms_of_a_b_l727_727140

-- Defining the vectors
def vec (x y : ℝ) : ℝ × ℝ := (x, y)

-- Defining the given vectors
def a := vec 1 1
def b := vec 1 (-1)
def c := vec (-1) 2

-- The statement
theorem express_c_in_terms_of_a_b :
  c = (1/2) • a + (-3/2) • b :=
sorry

end express_c_in_terms_of_a_b_l727_727140


namespace combined_average_age_l727_727621

-- Definitions based on given conditions
def num_fifth_graders : ℕ := 28
def avg_age_fifth_graders : ℝ := 10
def num_parents : ℕ := 45
def avg_age_parents : ℝ := 40

-- The statement to prove
theorem combined_average_age : (num_fifth_graders * avg_age_fifth_graders + num_parents * avg_age_parents) / (num_fifth_graders + num_parents) = 28.49 :=
  by
  sorry

end combined_average_age_l727_727621


namespace divisibility_by_24_l727_727254

theorem divisibility_by_24 (n : ℤ) : 24 ∣ n * (n + 2) * (5 * n - 1) * (5 * n + 1) :=
sorry

end divisibility_by_24_l727_727254


namespace number_added_after_division_is_5_l727_727646

noncomputable def number_thought_of : ℕ := 72
noncomputable def result_after_division (n : ℕ) : ℕ := n / 6
noncomputable def final_result (n x : ℕ) : ℕ := result_after_division n + x

theorem number_added_after_division_is_5 :
  ∃ x : ℕ, final_result number_thought_of x = 17 ∧ x = 5 :=
by
  sorry

end number_added_after_division_is_5_l727_727646


namespace exists_points_l727_727229

theorem exists_points (n : ℕ) (h : n ≥ 3) :
  ∃ (points : list (ℝ × ℝ)),
    (∀ i j, i ≠ j → (points.nth i).fst ≠ (points.nth j).fst ∨ (points.nth i).snd ≠ (points.nth j).snd) ∧
    (∀ i j, i ≠ j → (let P_i := points.nth i in let P_j := points.nth j in (P_i.fst - P_j.fst)^2 + (P_i.snd - P_j.snd)^2) ∉ ℚ) ∧
    (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → let P_i := points.nth i in let P_j := points.nth j in let P_k := points.nth k in
    2 * (P_i.fst * (P_j.snd - P_k.snd) + P_j.fst * (P_k.snd - P_i.snd) + P_k.fst * (P_i.snd - P_j.snd)) ∈ ℚ) :=
by {
  sorry
}

end exists_points_l727_727229


namespace calculate_speed_of_stream_l727_727683

noncomputable def speed_of_stream (boat_speed : ℕ) (downstream_distance : ℕ) (upstream_distance : ℕ) : ℕ :=
  let x := (downstream_distance * boat_speed - boat_speed * upstream_distance) / (downstream_distance + upstream_distance)
  x

theorem calculate_speed_of_stream :
  speed_of_stream 20 26 14 = 6 := by
  sorry

end calculate_speed_of_stream_l727_727683


namespace XY_perpendicular_to_median_CM_l727_727308

open EuclideanGeometry

theorem XY_perpendicular_to_median_CM
  {A B C P X Y : Point}
  (hABC_incircle : Circle_incircle A B C)
  (hPA_tangent : TangentToCircle A P)
  (hPB_tangent : TangentToCircle B P)
  (hX_proj : Projection P (Line A C) X)
  (hY_proj : Projection P (Line B C) Y)
  (hM_midpoint : Midpoint (Line A B))
  (hM_C_midpoint : Midpoint (Line A B) (Line C (Image B C))):

  Perpendicular (Line X Y) (Median C) :=
sorry

end XY_perpendicular_to_median_CM_l727_727308


namespace George_must_walk_last_mile_at_15mph_l727_727085

theorem George_must_walk_last_mile_at_15mph :
  ∀ (d1 d2 : ℝ) (s1 s2 usual_speed first_mile_speed second_mile_speed remaining_time required_speed: ℝ),
  d1 = 2 → 
  s1 = 5 →
  d2 = 1 → 
  s2 = 3 →
  first_mile_speed = d2 / s2 → 
  remaining_time = d1 / s1 - first_mile_speed →
  required_speed = d2 / remaining_time →
  required_speed = 15 :=
by
  intros d1 d2 s1 s2 usual_speed first_mile_speed second_mile_speed remaining_time required_speed
  assume h1 h2 h3 h4 h5 h6 h7
  sorry

end George_must_walk_last_mile_at_15mph_l727_727085


namespace induction_proof_base_case_l727_727253

theorem induction_proof (n : ℕ) (h : 0 < n) :
  (∑ k in finset.range (2 * n + 1), (1 : ℝ) / (n + k)) ≥ 1 :=
by sorry

theorem base_case : 
  (1 / 2 + 1 / 3 + 1 / 4) = (1 / 2 + 1 / 3 + 1 / 4) := 
by simp

end induction_proof_base_case_l727_727253


namespace arithmetic_mean_of_laura_scores_l727_727211

-- Given the conditions
def scores : List ℝ := [93, 87, 90, 94, 88, 92]

-- The arithmetic mean of a list of numbers
def arithmetic_mean (l : List ℝ) : ℝ :=
  l.sum / l.length

-- The specific arithmetic mean for Laura's scores
theorem arithmetic_mean_of_laura_scores :
  arithmetic_mean scores = 90.67 := by
  sorry

end arithmetic_mean_of_laura_scores_l727_727211


namespace general_formula_proof_set_of_n_proof_l727_727281

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 + Real.sqrt 2 ∧ (∀ n ≥ 2, (a n - a (n - 1)) * (a n + a (n - 1) - 2 * Real.sqrt n) = 2)

noncomputable def my_sequence := λ n, Real.sqrt n + Real.sqrt (n + 1)

theorem general_formula_proof :
  ∀ a : ℕ → ℝ, sequence a → ∀ n, a n = my_sequence n :=
begin
  sorry
end

theorem set_of_n_proof :
  ∀ a : ℕ → ℝ, sequence a → { n ∈ ℤ | 1021026 ≤ n ∧ n ≤ 1022121 } = { n ∈ ℤ | (Real.floor (my_sequence (n.to_nat)) = 2022) } :=
begin
  sorry
end

end general_formula_proof_set_of_n_proof_l727_727281


namespace determine_M_l727_727408

theorem determine_M : ∃ M : ℕ, 18^3 * 50^3 = 30^3 * M^3 ∧ M = 30 := by
  let M := 30
  use M
  calc
    18^3 * 50^3 = (2 * 3^2)^3 * (2 * 5^2)^3 := by sorry
            ... = 2^3 * (3^2)^3 * 2^3 * (5^2)^3 := by sorry
            ... = 2^6 * 3^6 * 5^6 := by sorry
  calc
    30^3 * M^3 = (2 * 3 * 5)^3 * M^3 := by sorry
             ... = (2^3 * 3^3 * 5^3) * M^3 := by sorry
             ... = 2^6 * 3^6 * 5^6 := by sorry
  show 30^3 * M^3 = 2^6 * 3^6 * 5^6 + 18^3 * 50^3 := by sorry
  calc
    M = 30 := rfl

end determine_M_l727_727408


namespace eq_pow_sum_nonneg_l727_727084

open Real

theorem eq_pow_sum_nonneg (x y : ℝ) (p : ℚ) (hx_nonneg : 0 ≤ x) (hy_nonneg : 0 ≤ y) :
  (x + y)^p = x^p + y^p ↔ 
  (p > 0 ∧ p ≠ 1 ∧ (x = 0 ∨ y = 0)) ∨ 
  (p = 1) :=
sorry

end eq_pow_sum_nonneg_l727_727084


namespace sufficient_condition_l727_727497

variables {ℝ : Type*} [inner_product_space ℝ ℝ]
variables (a b : ℝ) (k : ℝ)

-- Conditions: a and b are unit vectors.
def is_unit_vector (v : ℝ) : Prop := inner v v = 1

-- Definitions directly from the conditions.
def orthogonal (v w : ℝ) : Prop := inner v w = 0

-- The proof statement.
theorem sufficient_condition (a b : ℝ) (k : ℝ) 
  (ha : is_unit_vector a) 
  (hb : is_unit_vector b) :
  orthogonal (a + k • b) (k • a - b) ↔ (k = -1 ∨ orthogonal a b) :=
sorry

end sufficient_condition_l727_727497


namespace matrix_inv_correct_l727_727424

open Matrix

noncomputable def matrix_A : Matrix (Fin 2) (Fin 2) ℤ :=
  !![3, 4;
   -2, 9]

noncomputable def matrix_A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  !![
    (9 : ℚ)/35, (-4 : ℚ)/35;
    (2 : ℚ)/35, (3 : ℚ)/35]

theorem matrix_inv_correct : matrix_A⁻¹ = matrix_A_inv := by
  sorry

end matrix_inv_correct_l727_727424


namespace english_homework_correct_time_l727_727598

-- Define the given conditions as constants
def total_time : ℕ := 180 -- 3 hours in minutes
def math_homework_time : ℕ := 45
def science_homework_time : ℕ := 50
def history_homework_time : ℕ := 25
def special_project_time : ℕ := 30

-- Define the function to compute english homework time
def english_homework_time : ℕ :=
  total_time - (math_homework_time + science_homework_time + history_homework_time + special_project_time)

-- The theorem to show the English homework time is 30 minutes
theorem english_homework_correct_time :
  english_homework_time = 30 :=
  by
    sorry

end english_homework_correct_time_l727_727598


namespace binom_21_15_l727_727109

theorem binom_21_15 (h1 : nat.choose 20 13 = 77520) 
                   (h2 : nat.choose 20 14 = 38760) 
                   (h3 : nat.choose 22 15 = 203490) : 
  nat.choose 21 15 = 87210 := 
by 
  sorry

end binom_21_15_l727_727109


namespace closest_axis_of_symmetry_to_origin_l727_727655

theorem closest_axis_of_symmetry_to_origin :
  ∀ (x : ℝ), g x = 2 * sin (x + 5 * π / 12) →
  axis_of_symmetry x = (π / 12) :=
by
  sorry

end closest_axis_of_symmetry_to_origin_l727_727655


namespace num_elements_intersection_l727_727887

def M : Set ℂ := {z | ∃ (t : ℝ), z = (t / (1 + t) : ℂ) + (1 + t) * Complex.i / t ∧ t ≠ -1 ∧ t ≠ 0}

def N : Set ℂ := {z | ∃ (t : ℝ), z = Real.sqrt 2 * (Real.cos (Real.arcsin t) : ℂ) + (Real.cos (Real.arccos t) : ℂ) * Complex.i ∧ -1 ≤ t ∧ t ≤ 1}

theorem num_elements_intersection : #(M ∩ N) = 0 := by
  sorry

end num_elements_intersection_l727_727887


namespace frank_ryan_problem_ratio_l727_727749

theorem frank_ryan_problem_ratio 
  (bill_problems : ℕ)
  (h1 : bill_problems = 20)
  (ryan_problems : ℕ)
  (h2 : ryan_problems = 2 * bill_problems)
  (frank_problems_per_type : ℕ)
  (h3 : frank_problems_per_type = 30)
  (types : ℕ)
  (h4 : types = 4) : 
  frank_problems_per_type * types / ryan_problems = 3 := by
  sorry

end frank_ryan_problem_ratio_l727_727749


namespace simplify_cos_sum_squared_l727_727266

theorem simplify_cos_sum_squared :
  (cos(42 * real.pi / 180) + cos(102 * real.pi / 180) + 
   cos(114 * real.pi / 180) + cos(174 * real.pi / 180))^2 
   = 3 / 4 := 
by
  sorry

end simplify_cos_sum_squared_l727_727266


namespace dartboard_area_ratio_l727_727401

theorem dartboard_area_ratio
  (side_length : ℝ)
  (h_side_length : side_length = 2)
  (t : ℝ)
  (q : ℝ)
  (h_t : t = (1 / 2) * (1 / (Real.sqrt 2)) * (1 / (Real.sqrt 2)))
  (h_q : q = ((side_length * side_length) - (8 * t)) / 4) :
  q / t = 2 := by
  sorry

end dartboard_area_ratio_l727_727401


namespace part1_part2_l727_727486

noncomputable def f (x a : ℝ) : ℝ := |(x - a)| + |(x + 2)|

-- Part (1)
theorem part1 (x : ℝ) (h : f x 1 ≤ 7) : -4 ≤ x ∧ x ≤ 3 :=
by
  sorry

-- Part (2)
theorem part2 (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 2 * a + 1) : a ≤ 1 :=
by
  sorry

end part1_part2_l727_727486


namespace CI_eq_IK_l727_727690

noncomputable def triangle (A B C : Type) := (A ≠ B ∧ B ≠ C ∧ A ≠ C)

variables {A B C I L K : Type} [triangle A B C]

-- Condition: CL is the angle bisector of ∠C
def angle_bisector (A B C L : Type) : Prop := sorry

-- Condition: CL intersects the circumcircle at K
def intersects_circumcircle (A B C L K : Type) : Prop := sorry

-- Condition: I is the incenter of ΔABC
def is_incenter (A B C I : Type) : Prop := sorry

-- Condition: IL = LK
def lengths_equal (I L K : Type) : Prop := sorry

-- The theorem to prove
theorem CI_eq_IK (h1 : angle_bisector A B C L)
                 (h2 : intersects_circumcircle A B C L K)
                 (h3 : is_incenter A B C I)
                 (h4 : lengths_equal I L K) :
                 sorry :=
sorry

end CI_eq_IK_l727_727690


namespace count_positive_n_l727_727820

def is_factorable (n : ℕ) : Prop :=
  ∃ a b : ℤ, (a + b = -2) ∧ (a * b = - (n:ℤ))

theorem count_positive_n : 
  (∃ (S : Finset ℕ), S.card = 45 ∧ ∀ n ∈ S, (1 ≤ n ∧ n ≤ 2000) ∧ is_factorable n) :=
by
  -- Placeholder for the proof
  sorry

end count_positive_n_l727_727820


namespace value_is_6_l727_727631

-- We know the conditions that the least number which needs an increment is 858
def least_number : ℕ := 858

-- Define the numbers 24, 32, 36, and 54
def num1 : ℕ := 24
def num2 : ℕ := 32
def num3 : ℕ := 36
def num4 : ℕ := 54

-- Define the LCM function to compute the least common multiple
def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

-- Define the LCM of the four numbers
def lcm_all : ℕ := lcm (lcm num1 num2) (lcm num3 num4)

-- Compute the value that needs to be added
def value_to_be_added : ℕ := lcm_all - least_number

-- Prove that this value equals to 6
theorem value_is_6 : value_to_be_added = 6 := by
  -- Proof would go here
  sorry

end value_is_6_l727_727631


namespace arithmetic_sequence_n_value_l727_727906

def arithmetic_seq_nth_term (a1 d n : ℕ) : ℕ :=
  a1 + (n - 1) * d

theorem arithmetic_sequence_n_value :
  ∀ (a1 d n an : ℕ), a1 = 3 → d = 2 → an = 25 → arithmetic_seq_nth_term a1 d n = an → n = 12 :=
by
  intros a1 d n an ha1 hd han h
  sorry

end arithmetic_sequence_n_value_l727_727906


namespace find_original_number_l727_727250

theorem find_original_number :
  ∃ x : ℚ, (5 * (3 * x + 15) = 245) ∧ x = 34 / 3 := by
  sorry

end find_original_number_l727_727250


namespace find_S_l727_727117

variable {R k : ℝ}

theorem find_S (h : |k + R| / |R| = 0) : S = 1 :=
by
  let S := |k + 2*R| / |2*k + R|
  have h1 : k + R = 0 := by sorry
  have h2 : k = -R := by sorry
  sorry

end find_S_l727_727117


namespace part1_part2_prob_l727_727702

noncomputable def calc_distribution : ℕ → list (ℤ × ℚ)
| n := [(-6, 27 / 125), (n - 4, 54 / 125), (2 * n - 2, 36 / 125), (3 * n, 8 / 125)]

noncomputable def E_xi (n : ℕ) : ℚ :=
  (-6) * (27 / 125) + (n - 4) * (54 / 125) + (2 * n - 2) * (36 / 125) + 3 * n * (8 / 125)

theorem part1 (n : ℕ) : (6 * n - 18) / 5 > 0 ↔ n > 3 :=
by sorry

theorem part2_prob (n : ℕ) (hn : n = 4) : 
  let P_C := 1 - (2 / 5) ^ 3,
      P_CD := nat.choose 3 1 * (3 / 5) * (2 / 5) ^ 2
  in P_CD / P_C = 4 / 13 :=
by sorry

end part1_part2_prob_l727_727702


namespace sum_of_middle_ten_terms_in_arithmetic_sequence_l727_727538

theorem sum_of_middle_ten_terms_in_arithmetic_sequence :
  ∃ (d : ℚ), (d = (2 - 1) / 11) → 
  (∑ i in (finset.range 10).map (λ x, x+2), (1 + (i - 1) * d)) = 15 :=
begin
  sorry
end

end sum_of_middle_ten_terms_in_arithmetic_sequence_l727_727538


namespace pentagon_perimeter_even_l727_727534

noncomputable def dist_sq (A B : ℤ × ℤ) : ℤ :=
  (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2

theorem pentagon_perimeter_even (A B C D E : ℤ × ℤ) (h1 : dist_sq A B % 2 = 1) (h2 : dist_sq B C % 2 = 1) 
  (h3 : dist_sq C D % 2 = 1) (h4 : dist_sq D E % 2 = 1) (h5 : dist_sq E A % 2 = 1) : 
  (dist_sq A B + dist_sq B C + dist_sq C D + dist_sq D E + dist_sq E A) % 2 = 0 := 
by 
  sorry

end pentagon_perimeter_even_l727_727534


namespace solution_set_f_x_gt_0_l727_727113

theorem solution_set_f_x_gt_0 (b : ℝ)
  (h_eq : ∀ x : ℝ, (x + 1) * (x - 3) = 0 → b = -2) :
  {x : ℝ | (x - 1)^2 > 0} = {x : ℝ | x ≠ 1} :=
by
  sorry

end solution_set_f_x_gt_0_l727_727113


namespace smallest_positive_integer_ends_in_7_and_divisible_by_5_l727_727669

theorem smallest_positive_integer_ends_in_7_and_divisible_by_5 : 
  ∃ n : ℤ, n > 0 ∧ n % 10 = 7 ∧ n % 5 = 0 ∧ n = 37 := 
by 
  sorry

end smallest_positive_integer_ends_in_7_and_divisible_by_5_l727_727669


namespace probability_odd_dots_is_correct_l727_727969

noncomputable def probability_odd_dots_after_removal : ℚ :=
  let faces : Finset ℕ := Finset.range 6 + 1  -- Faces of the die {1, 2, 3, 4, 5, 6}
  let total_dots := 21  -- Total dots on a standard die
  let choose_2 (n : ℕ) := (n * (n - 1)) / 2  -- Combination formula for choosing 2 from n
  let total_combinations := choose_2 total_dots  -- Total ways to choose 2 dots from 21
  let face_prob (n : ℕ) : ℚ :=
    if n < 2 then 0  -- Less than 2 dots, impossible to remove 2 dots
    else choose_2 n / total_combinations  -- Probability of removing 2 dots from a face with n dots
  let remaining_odd (n : ℕ) : Bool := (n - 2) % 2 = 1  -- Check if (n-2) is odd
  let relevant_faces : Finset ℕ := faces.filter remaining_odd  -- Faces resulting in an odd number of dots
  (1 / faces.card) * ((relevant_faces.to_list.map face_prob).sum)  -- Mean probability across all relevant faces

theorem probability_odd_dots_is_correct : probability_odd_dots_after_removal = 1/60 :=
sorry

end probability_odd_dots_is_correct_l727_727969


namespace sequence_convergence_l727_727842

noncomputable def sum_of_digits_base (n k : ℕ) : ℕ := sorry

noncomputable def next_number (n k : ℕ) : ℕ :=
  n + (k - 1) ^ 2 * sum_of_digits_base n k

def sequence_converges (k : ℕ) (h : k > 5) := ∀ (n : ℕ), ∃ m, ∀ p ≥ m, next_number (iterator next_number p n k) k = 2 * (k - 1) ^ 3

theorem sequence_convergence (k : ℕ) (h : k > 5): ∀ n, ∃ m, ∀ p ≥ m, next_number (iterator next_number p n k) k = 2 * (k - 1) ^ 3 :=
by
  intro n
  sorry

end sequence_convergence_l727_727842


namespace matrix_inverse_exists_and_correct_l727_727429

theorem matrix_inverse_exists_and_correct : 
  let A := matrix ([[3, 4], [-2, 9]] : matrix (fin 2) (fin 2) ℚ)
  let detA := (3 * 9) - (4 * -2)
  detA ≠ 0 →
  matrix.inv A = matrix ([[9/35, -4/35], [2/35, 3/35]] : matrix (fin 2) (fin 2) ℚ) :=
by
  let A := matrix ([[3, 4], [-2, 9]] : matrix (fin 2) (fin 2) ℚ)
  let detA := (3 * 9) - (4 * -2)
  have detA_nz : detA ≠ 0 := by simp [detA]
  have invA := (matrix.inv_of_det_ne_zero _ detA_nz)
  have matrix_inv_eq := (invA A)
  let expected_inv := (matrix ([[9/35, -4/35], [2/35, 3/35]] : matrix (fin 2) (fin 2) ℚ))
  exact matrix_inv_eq = expected_inv
  sorry

end matrix_inverse_exists_and_correct_l727_727429


namespace age_equation_correct_l727_727294

-- Define the ages of the younger and older brothers
variables (x y : ℕ)

-- Conditions from the problem
def condition1 : Prop := x + y = 16
def condition2 : Prop := 2 * (x + 4) = y + 4

-- Prove that the correct equation is provided given the conditions
theorem age_equation_correct : (2 * (x + 4) = y + 4) ↔ condition2 :=
by
  refl

end age_equation_correct_l727_727294


namespace determine_pq_value_l727_727275

noncomputable def p : ℝ → ℝ := λ x => 16 * x
noncomputable def q : ℝ → ℝ := λ x => (x + 4) * (x - 1)

theorem determine_pq_value : (p (-1) / q (-1)) = 8 / 3 := by
  sorry

end determine_pq_value_l727_727275


namespace SamLastPage_l727_727518

theorem SamLastPage (total_pages : ℕ) (Sam_read_time : ℕ) (Lily_read_time : ℕ) (last_page : ℕ) :
  total_pages = 920 ∧ Sam_read_time = 30 ∧ Lily_read_time = 50 → last_page = 575 :=
by
  intros h
  sorry

end SamLastPage_l727_727518


namespace range_of_a3_plus_a9_l727_727196

variable {a_n : ℕ → ℝ}

-- Given condition: in a geometric sequence, a4 * a8 = 9
def geom_seq_condition (a_n : ℕ → ℝ) : Prop :=
  a_n 4 * a_n 8 = 9

-- Theorem statement
theorem range_of_a3_plus_a9 (a_n : ℕ → ℝ) (h : geom_seq_condition a_n) :
  ∃ x y, (x + y = a_n 3 + a_n 9) ∧ (x ≥ 0 ∧ y ≥ 0 ∧ x + y ≥ 6) ∨ (x ≤ 0 ∧ y ≤ 0 ∧ x + y ≤ -6) ∨ (x = 0 ∧ y = 0 ∧ a_n 3 + a_n 9 ∈ (Set.Ici 6 ∪ Set.Iic (-6))) :=
sorry

end range_of_a3_plus_a9_l727_727196


namespace triangle_side_relation_l727_727345

theorem triangle_side_relation (A B C : Type) [inner_product_space ℝ A]
  (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : ∃ u v w : A, u ≠ v ∧ v ≠ w ∧ w ≠ u ∧ dist u v = c ∧ dist u w = b ∧ dist v w = a)
  (h5 : ∃ θ : ℝ, 0 < θ ∧ θ < π ∧ θ = 2 * dual_dist u v w) :
  a^2 = b^2 + b * c := 
sorry

end triangle_side_relation_l727_727345


namespace find_DC_l727_727192

noncomputable def sin_ratio_A : ℝ := 4 / 5
noncomputable def sin_ratio_C : ℝ := 2 / 5
noncomputable def AB : ℝ := 30

theorem find_DC (BD BC DC : ℝ) (h1 : AB = 30) (h2 : sin (Real.arcsin sin_ratio_A) = sin_ratio_A) 
(h3 : sin (Real.arcsin sin_ratio_C) = sin_ratio_C)
(h4 : BD = 24)
(h5 : BC = 60) :
DC = 24 * Real.sqrt(5.25) := 
sorry

#print find_DC

end find_DC_l727_727192


namespace triangle_value_correct_l727_727918

noncomputable def triangle_value (A b area : ℝ) (A_eq : A = 60) (b_eq : b = 1) (area_eq : area = sqrt 3) : ℝ :=
  let c := 4
  let a := sqrt 13
  let sinA := sqrt 3 / 2
  let sinB := a / c * sinA
  let sinC := b / c * sinA
  (a + b + c) / (sinA + sinB + sinC)

theorem triangle_value_correct : triangle_value 60 1 (sqrt 3) 60 rfl 1 rfl (sqrt 3) rfl = (2 * sqrt 39) / 3 := 
  sorry

end triangle_value_correct_l727_727918


namespace expected_value_fair_octahedral_die_l727_727664

theorem expected_value_fair_octahedral_die : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability := (1 / 8 : ℝ)
  let E := ∑ i in outcomes, (probability * i)
  E = 4.5 :=
by
  sorry

end expected_value_fair_octahedral_die_l727_727664


namespace sum_of_squares_of_real_solutions_l727_727439

theorem sum_of_squares_of_real_solutions :
  let solutions := {x : ℝ | x^64 = 4^64} in
  ∑ x in solutions, x^2 = 128 :=
by
  let solutions := {x | x^64 = 4^64}
  let real_solutions := {x | x = 8 ∨ x = -8}
  have all_real_solutions_found : solutions = real_solutions,
  sorry
  calc
    ∑ x in real_solutions, x^2
        = 8^2 + (-8)^2 : by sorry
    ... = 64 + 64       : by norm_num
    ... = 128          : by norm_num

end sum_of_squares_of_real_solutions_l727_727439


namespace number_of_paintings_per_new_gallery_l727_727378

-- Define all the conditions as variables/constants
def pictures_original : Nat := 9
def new_galleries : Nat := 5
def pencils_per_picture : Nat := 4
def pencils_per_exhibition : Nat := 2
def total_pencils : Nat := 88

-- Define the proof problem in Lean
theorem number_of_paintings_per_new_gallery (pictures_original new_galleries pencils_per_picture pencils_per_exhibition total_pencils : Nat) :
(pictures_original = 9) → (new_galleries = 5) → (pencils_per_picture = 4) → (pencils_per_exhibition = 2) → (total_pencils = 88) → 
∃ (pictures_per_gallery : Nat), pictures_per_gallery = 2 :=
by
  intros
  sorry

end number_of_paintings_per_new_gallery_l727_727378


namespace linear_dependency_k_val_l727_727290

theorem linear_dependency_k_val (k : ℝ) :
  (∃ (c1 c2 : ℝ), (c1 ≠ 0 ∨ c2 ≠ 0) ∧ 2 * c1 + 4 * c2 = 0 ∧ 3 * c1 + k * c2 = 0) ↔ k = 6 :=
by sorry

end linear_dependency_k_val_l727_727290


namespace roots_quadratic_expression_value_l727_727116

theorem roots_quadratic_expression_value (m n : ℝ) 
  (h1 : m^2 + 2 * m - 2027 = 0)
  (h2 : n^2 + 2 * n - 2027 = 0) :
  (2 * m - m * n + 2 * n) = 2023 :=
by
  sorry

end roots_quadratic_expression_value_l727_727116


namespace ponchik_ate_62_cubes_l727_727451

def has_odd_neighbors (neighbors : ℕ) : Prop := neighbors % 2 = 1

def cube_positions : List (ℕ × ℕ × ℕ) := 
  List.product (List.finRange 5) (List.product (List.finRange 5) (List.finRange 5))

def count_odd_neighbor_cubes : ℕ :=
  cube_positions.foldr (λ pos acc, 
    let ⟨x, (y, z)⟩ := pos in
    let neighbors := 
      (if x > 0 then 1 else 0) + (if x < 4 then 1 else 0) + 
      (if y > 0 then 1 else 0) + (if y < 4 then 1 else 0) +
      (if z > 0 then 1 else 0) + (if z < 4 then 1 else 0) 
    in if has_odd_neighbors neighbors then acc + 1 else acc) 0

theorem ponchik_ate_62_cubes : count_odd_neighbor_cubes = 62 := by 
  sorry

end ponchik_ate_62_cubes_l727_727451


namespace proposition_1_correct_proposition_2_correct_proposition_3_incorrect_proposition_4_incorrect_l727_727695

variables (P Q : Prop) (x y a b m : ℝ)
variables (A B C : ℝ)
variables (triangle_ABC : Triangle)

open Classical

theorem proposition_1_correct 
  (h1: (P → Q) ↔ (¬Q → ¬P)) :
  P = Q := by sorry

theorem proposition_2_correct :
  (B = 60 ∧ A + 2 * B + C = 180 = (A, B, C).form_arithmetic_sequence) := by sorry

theorem proposition_3_incorrect :
  ¬ ((x > 1 ∧ y > 2) ↔ (x + y > 3 ∧ x * y > 2)) := by sorry

theorem proposition_4_incorrect :
  ¬ ((a < b) ↔ (a * m < b * m) := by sorry
  
end

end proposition_1_correct_proposition_2_correct_proposition_3_incorrect_proposition_4_incorrect_l727_727695


namespace gcd_12012_18018_l727_727814

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_12012_18018 : gcd 12012 18018 = 6006 := sorry

end gcd_12012_18018_l727_727814


namespace part1_part2_part2_part2_l727_727132

section
variable (a : ℝ) (x : ℝ)
def f (x : ℝ) (a : ℝ) := x^2 + 2 * a * x + 3
def g (a : ℝ) : ℝ := 
  if a > 2 then 7 - 4 * a
  else if a < -2 then 7 + 4 * a
  else 3 - a^2

theorem part1 (a : ℝ) : (∀ x ∈ ([-2.0, 2.0] : set ℝ), f x (-1) ∈ set.Icc 2 11) := sorry

theorem part2 (a : ℝ) (h1: a > 2) : g a = 7 - 4 * a := sorry
theorem part2 (a : ℝ) (h2: -2 ≤ a ∧ a ≤ 2) : g a = 3 - a^2 := sorry
theorem part2 (a : ℝ) (h3: a < -2) : g a = 7 + 4 * a := sorry
end

end part1_part2_part2_part2_l727_727132


namespace right_triangle_hypotenuse_l727_727364

theorem right_triangle_hypotenuse (a b c : ℕ) (h1 : a^2 + b^2 = c^2) 
  (h2 : b = c - 1575) (h3 : b < 1991) : c = 1800 :=
sorry

end right_triangle_hypotenuse_l727_727364


namespace curve_is_circle_l727_727790

-- Definition of the curve in polar coordinates
def curve (r θ : ℝ) : Prop :=
  r = 3 * Real.sin θ

-- The theorem to prove
theorem curve_is_circle : ∀ θ : ℝ, ∃ r : ℝ, curve r θ → (∃ c : ℝ × ℝ, ∃ R : ℝ, ∀ p : ℝ × ℝ, (Real.sqrt ((p.1 - c.1) ^ 2 + (p.2 - c.2) ^ 2) = R)) :=
by
  sorry

end curve_is_circle_l727_727790


namespace determine_c15_l727_727758

noncomputable def polynomial_product (c : ℕ → ℕ) : Polynomial ℤ :=
  (List.range 15).foldr (λ k p, p * (1 - Polynomial.C z ^ (k+1)) ^ (c (k+1))) 1

theorem determine_c15 (c : ℕ → ℕ) (h1 : c 15 = 0) :
  polynomial_product c ≡ 1 - 3 * Polynomial.C z [MOD z^20] :=
sorry

end determine_c15_l727_727758


namespace Euleria_pave_odds_l727_727300

open Classical

-- Given conditions
variables (Cities : Finset ℕ)
variable (Roads : Finset (ℕ × ℕ))
variable (Connected : ∀ x y ∈ Cities, ∃ path : List (ℕ × ℕ), 
             (∀ edge ∈ path, edge ∈ Roads) ∧ 
             (∀ edge ∈ path, edge.1 ∈ Cities ∧ edge.2 ∈ Cities) ∧
             (x = List.head! path).1 ∧ (y = List.last path (x, x)).2)
variable (number_of_cities : Cities.card = 1000)

-- Proof problem statement
theorem Euleria_pave_odds :
  ∃ paved_roads : Finset (ℕ × ℕ), 
  (∀ city ∈ Cities, 
    (Finset.filter (λ edge, edge.1 = city ∨ edge.2 = city) paved_roads).card % 2 = 1) :=
sorry

end Euleria_pave_odds_l727_727300


namespace seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths_l727_727019

theorem seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths : 
  (7 : ℚ) * (9 + (2 / 5)) = 65 + (4 / 5) :=
by
  sorry

end seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths_l727_727019


namespace chocolate_cake_cutting_l727_727244

theorem chocolate_cake_cutting (cake : Set ℝ) (chocolates : Set (Set ℝ)) :
  ∃ (choco1 choco2 : Set ℝ), choco1 ∈ chocolates ∧ choco2 ∈ chocolates ∧ 
  choco1 ≠ choco2 ∧ ¬(∀ (cuts : Set (Set ℝ)), 
  (∀ poly ∈ cuts, Convex ℝ poly ∧ (∃ choco ∈ chocolates, choco ⊆ poly)) ∧ 
  ∀ choco ∈ chocolates, ∃ poly ∈ cuts, choco ⊆ poly) :=
begin
  -- cake is a flat square
  assume h_square : Convex ℝ cake,
  -- chocolates are non-touching triangular subsets of the cake
  assume h_chocolates : ∀ choco ∈ chocolates, Convex ℝ choco ∧ subset choco cake ∧ 
                      ∀ choco1 choco2 ∈ chocolates, choco1 ≠ choco2 → disjoint choco1 choco2,
  -- contradiction arises when trying to cut the cake as required
  by_contradiction,
  contradiction,
end


end chocolate_cake_cutting_l727_727244


namespace compare_negative_one_with_abs_of_negative_two_fifths_l727_727754

theorem compare_negative_one_with_abs_of_negative_two_fifths : -1 < | - (2 / 5) | :=
by sorry

end compare_negative_one_with_abs_of_negative_two_fifths_l727_727754


namespace pear_counts_after_events_l727_727736

theorem pear_counts_after_events (Alyssa_picked Nancy_picked Carlos_picked : ℕ) (give_away : ℕ)
  (eat_fraction : ℚ) (share_fraction : ℚ) :
  Alyssa_picked = 42 →
  Nancy_picked = 17 →
  Carlos_picked = 25 →
  give_away = 5 →
  eat_fraction = 0.20 →
  share_fraction = 0.5 →
  ∃ (Alyssa_picked_final Nancy_picked_final Carlos_picked_final : ℕ),
    Alyssa_picked_final = 30 ∧
    Nancy_picked_final = 14 ∧
    Carlos_picked_final = 18 :=
by
  sorry

end pear_counts_after_events_l727_727736


namespace linear_dependency_k_val_l727_727291

theorem linear_dependency_k_val (k : ℝ) :
  (∃ (c1 c2 : ℝ), (c1 ≠ 0 ∨ c2 ≠ 0) ∧ 2 * c1 + 4 * c2 = 0 ∧ 3 * c1 + k * c2 = 0) ↔ k = 6 :=
by sorry

end linear_dependency_k_val_l727_727291


namespace multiplication_with_mixed_number_l727_727012

-- Define mixed numbers as rational numbers for proper calculation
def mixed_to_rational (whole : ℕ) (num : ℕ) (den : ℕ) : ℚ :=
  whole + num / den

-- 7 * (9 + 2/5)
def lhs : ℚ := 7 * mixed_to_rational 9 2 5

-- 65 + 4/5
def rhs : ℚ := mixed_to_rational 65 4 5

theorem multiplication_with_mixed_number : lhs = rhs := by
  sorry

end multiplication_with_mixed_number_l727_727012


namespace find_g3_l727_727617

variable {α : Type} [OrderedCommRing α]

def f (x : α) : α := sorry
def g (x : α) : α := sorry

axiom fg_condition : ∀ x : α, x ≥ 1 → f(g(x)) = x^3
axiom gf_condition : ∀ x : α, x ≥ 1 → g(f(x)) = x^4
axiom g_81 : g(81) = (81 : α)

theorem find_g3 : [g(3)]^4 = 531441 := by
  sorry

end find_g3_l727_727617


namespace number_of_elements_in_M_l727_727082

def operation_star (m n : ℕ) : ℕ :=
  if (m % 2 = 0) = (n % 2 = 0) then m + n else m * n

def is_element_of_M (a b : ℕ) : Prop :=
  b > 0 ∧ b \mathbin※ b = 12 ∧ a > 0

noncomputable def M : set (ℕ × ℕ) :=
  {p | is_element_of_M p.1 p.2}

theorem number_of_elements_in_M : (M: set (ℕ × ℕ)).card = 15 := sorry

end number_of_elements_in_M_l727_727082


namespace evaluate_expression_l727_727279

def operation (x y : ℚ) : ℚ := x^2 / y

theorem evaluate_expression : 
  (operation (operation 3 4) 2) - (operation 3 (operation 4 2)) = 45 / 32 :=
by
  sorry

end evaluate_expression_l727_727279


namespace most_cost_effective_80_oranges_l727_727592

noncomputable def cost_of_oranges (p1 p2 p3 : ℕ) (q1 q2 q3 : ℕ) : ℕ :=
  let cost_per_orange_p1 := p1 / q1
  let cost_per_orange_p2 := p2 / q2
  let cost_per_orange_p3 := p3 / q3
  if cost_per_orange_p3 ≤ cost_per_orange_p2 ∧ cost_per_orange_p3 ≤ cost_per_orange_p1 then
    (80 / q3) * p3
  else if cost_per_orange_p2 ≤ cost_per_orange_p1 then
    (80 / q2) * p2
  else
    (80 / q1) * p1

theorem most_cost_effective_80_oranges :
  cost_of_oranges 35 45 95 6 9 20 = 380 :=
by sorry

end most_cost_effective_80_oranges_l727_727592


namespace similar_triangles_l727_727600

-- Definitions of the given conditions in the problem
variables {P A B C : Type} 
variable [HasAngle P A C] [HasAngle C B P]

-- Define the angles and lengths involved
variable (angleACB : HasAngle.angle P A C = 90)
variable (lengthABC : ∀ {x y : P}, A = B → B = C → x = y)

-- State the theorem to be proven
theorem similar_triangles (h1 : ∀ {x y : P}, x = y → ∃ z, angleACB ∧ lengthABC) :
  ∃ z, Trian.similar P A B ∧ Trian.similar P B C :=
sorry

end similar_triangles_l727_727600


namespace problem_l727_727220

theorem problem (a₅ b₅ a₆ b₆ a₇ b₇ : ℤ) (S₇ S₅ T₆ T₄ : ℤ)
  (h1 : a₅ = b₅)
  (h2 : a₆ = b₆)
  (h3 : S₇ - S₅ = 4 * (T₆ - T₄)) :
  (a₇ + a₅) / (b₇ + b₅) = -1 :=
sorry

end problem_l727_727220


namespace math_problem_l727_727399

theorem math_problem :
  let numerator := (15^4 + 400) * (30^4 + 400) * (45^4 + 400) * (60^4 + 400) * (75^4 + 400)
  let denominator := (5^4 + 400) * (20^4 + 400) * (35^4 + 400) * (50^4 + 400) * (65^4 + 400)
  numerator / denominator = 301 :=
by 
  sorry

end math_problem_l727_727399


namespace min_max_distance_square_vertices_l727_727659

theorem min_max_distance_square_vertices :
  ∃ (dist_min dist_max : ℝ),
    (dist_min = Real.sqrt 2 - 1) ∧
    (dist_max = Real.sqrt 2 + 1) ∧
    (∀ (x y : ℝ), (x - 1) * (x - 1) + y * y = 0 ∧ 
                 ((x - Real.cos (2 * π / 4)).abs = dist_min ∨ 
                  (x - Real.cos (2 * π / 4)).abs = dist_max)) :=
sorry

end min_max_distance_square_vertices_l727_727659


namespace angle_PAQ_eq_angle_PCQ_l727_727213

open EuclideanGeometry

noncomputable def isosceles_triangle (A B C : Point) : Prop :=
  B.dist A = B.dist C

noncomputable def point_on_ray (B P : Point) : Prop :=
  ∃ (t : ℝ) (t_pos : t > 0), P = B + t • (some_direction)

noncomputable def angle_equality (A B P C Q : Point) : Prop :=
  ∠ B A P = ∠ Q C A

theorem angle_PAQ_eq_angle_PCQ 
  {A B C P Q : Point} 
  (h_isosceles : isosceles_triangle A B C)
  (h_on_ray_P : point_on_ray B P)
  (h_on_ray_Q : point_on_ray B Q)
  (h_angle_eq : angle_equality A B P C Q) :
  ∠ P A Q = ∠ P C Q := 
sorry

end angle_PAQ_eq_angle_PCQ_l727_727213


namespace sqrt_720_eq_12_sqrt_5_l727_727997

theorem sqrt_720_eq_12_sqrt_5 : sqrt 720 = 12 * sqrt 5 :=
by
  sorry

end sqrt_720_eq_12_sqrt_5_l727_727997


namespace find_angle_C_l727_727516

noncomputable def angle_C (a b c : ℝ) (h : b^2 + a^2 - c^2 = sqrt 3 * a * b) : ℝ :=
  (real.arccos (sqrt 3 / 2))

theorem find_angle_C (a b c : ℝ) (h : b^2 + a^2 - c^2 = sqrt 3 * a * b) : angle_C a b c h = π / 6 :=
sorry

end find_angle_C_l727_727516


namespace value_of_a_l727_727490

theorem value_of_a (a : ℝ) (A : Set ℝ) (hA : A = {a^2, 1}) (h : 3 ∈ A) : 
  a = Real.sqrt 3 ∨ a = -Real.sqrt 3 :=
by
  sorry

end value_of_a_l727_727490


namespace west_for_200_is_neg_200_l727_727173

-- Given a definition for driving east
def driving_east (d : Int) : Int := d

-- Driving east for 80 km is +80 km
def driving_east_80 : Int := driving_east 80

-- Driving west should be the negative of driving east
def driving_west (d : Int) : Int := -d

-- Driving west for 200 km is -200 km
def driving_west_200 : Int := driving_west 200

-- Theorem to prove the given condition and expected result
theorem west_for_200_is_neg_200 : driving_west_200 = -200 :=
by
  -- Proof step is skipped
  sorry

end west_for_200_is_neg_200_l727_727173


namespace three_digit_sum_of_factorials_is_145_l727_727680

theorem three_digit_sum_of_factorials_is_145 :
  ∀ (x y z : ℕ),
    x < 10 ∧ y < 10 ∧ z < 10 ∧ 100x + 10y + z = x! + y! + z! →
    100x + 10y + z = 145 :=
by
  sorry

end three_digit_sum_of_factorials_is_145_l727_727680


namespace total_enemies_l727_727901

theorem total_enemies (points_per_enemy : ℕ) (points_earned : ℕ) (enemies_left : ℕ) (enemies_defeated : ℕ) :  
  (3 = points_per_enemy) → 
  (12 = points_earned) → 
  (2 = enemies_left) → 
  (points_earned / points_per_enemy = enemies_defeated) → 
  (enemies_defeated + enemies_left = 6) := 
by
  intros
  sorry

end total_enemies_l727_727901


namespace find_AC_length_l727_727911

theorem find_AC_length (AB BC CD DA : ℕ) 
  (hAB : AB = 10) (hBC : BC = 9) (hCD : CD = 19) (hDA : DA = 5) : 
  14 < AC ∧ AC < 19 → AC = 15 := 
by
  sorry

end find_AC_length_l727_727911


namespace bounded_region_area_l727_727630

theorem bounded_region_area (a : ℝ) (h : a > 0) : 
  let f1 := (x + a * y)^2 = 4 * a^2
      f2 := (a * x - y)^2 = a^2 
  in (area of the region bounded by f1 and f2) = 8 * a^2 / (a^2 + 1) :=
sorry

end bounded_region_area_l727_727630


namespace SI_passes_through_N_l727_727231

noncomputable def is_incircle (O I A B C D E F : Point) :=
  inscribed_in_circle A B C O ∧
  incenter I A B C ∧ 
  incircle_tangent_points D E F I A B C

noncomputable def is_foot_perpendicular (D E F S : Point) :=
  foot_of_perpendicular D E F S

noncomputable def antidiametric_point (A N O : Point) (Γ : Circumcircle) :=
  diameter_of_circle Γ A N O

noncomputable def passes_through (S I N : Point) :=
  collinear S I N

theorem SI_passes_through_N (O I A B C D E F S N : Point) (Γ : Circumcircle) :
  is_incircle O I A B C D E F →
  is_foot_perpendicular D E F S →
  antidiametric_point A N O Γ →
  passes_through S I N :=
begin
  sorry
end

end SI_passes_through_N_l727_727231


namespace problem_statement_l727_727879

def a : ℝ × ℝ := (1, 3)
def b (y : ℝ) : ℝ × ℝ := (2, y)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2
def perp (v1 v2 : ℝ × ℝ) : Prop := dot_product v1 v2 = 0
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)
def projection (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (dot_product v1 v2 / (magnitude v2 ^ 2)) • v2

theorem problem_statement (y : ℝ) (h : perp (a.1 + b y, a.2 + b y) a) :
  (∃ θ : ℝ, dot_product a (b y) = magnitude a * magnitude (b y) * real.cos θ ∧ θ = 3 * real.pi / 4) ∧
  (projection a (b y) = (-1, 2)) := sorry

end problem_statement_l727_727879


namespace cubic_polynomial_roots_value_l727_727833

theorem cubic_polynomial_roots_value
  (a b c d : ℝ) 
  (h_cond : a ≠ 0 ∧ d ≠ 0)
  (h_equiv : (a * (1/2)^3 + b * (1/2)^2 + c * (1/2) + d) + (a * (-1/2)^3 + b * (-1/2)^2 + c * (-1/2) + d) = 1000 * d)
  (h_roots : ∃ (x1 x2 x3 : ℝ), a * x1^3 + b * x1^2 + c * x1 + d = 0 ∧ a * x2^3 + b * x2^2 + c * x2 + d = 0 ∧ a * x3^3 + b * x3^2 + c * x3 + d = 0) 
  : (∃ (x1 x2 x3 : ℝ), (1 / (x1 * x2) + 1 / (x2 * x3) + 1 / (x1 * x3) = 1996)) :=
by
  sorry

end cubic_polynomial_roots_value_l727_727833


namespace charlyn_visibility_area_l727_727039

theorem charlyn_visibility_area 
  (side_length : ℝ) (visibility : ℝ) 
  (h_side_length : side_length = 5)
  (h_visibility : visibility = 1) : 
  (Int.round ((side_length + 2 * visibility)^2 - (side_length - 2 * visibility)^2 + real.pi) = 43) := 
sorry

end charlyn_visibility_area_l727_727039


namespace gear_ratio_l727_727526

/-- A system of three interconnected gears P, Q, and R where P and Q interlock and
    R is connected to Q through a secondary mechanism with 90% efficiency. Proves
    the ratio of their angular speeds. -/
theorem gear_ratio (a b c : ℕ) (ω_P ω_Q ω_R : ℝ) 
  (h1 : a * ω_P = b * ω_Q) 
  (h2 : ω_R = 0.9 * ω_Q) : 
  ω_P : ω_Q : ω_R = b : a : 0.9 * a :=
by
  sorry

end gear_ratio_l727_727526


namespace customers_total_l727_727371

theorem customers_total 
  (initial : ℝ) 
  (added_lunch_rush : ℝ) 
  (added_after_lunch_rush : ℝ) :
  initial = 29.0 →
  added_lunch_rush = 20.0 →
  added_after_lunch_rush = 34.0 →
  initial + added_lunch_rush + added_after_lunch_rush = 83.0 :=
by
  intros h1 h2 h3
  sorry

end customers_total_l727_727371


namespace sum_ages_l727_727062

variables (uncle_age eunji_age yuna_age : ℕ)

def EunjiAge (uncle_age : ℕ) := uncle_age - 25
def YunaAge (eunji_age : ℕ) := eunji_age + 3

theorem sum_ages (h_uncle : uncle_age = 41) (h_eunji : EunjiAge uncle_age = eunji_age) (h_yuna : YunaAge eunji_age = yuna_age) :
  eunji_age + yuna_age = 35 :=
sorry

end sum_ages_l727_727062


namespace lcm_9_14_l727_727432

/-- Given the definition of the least common multiple (LCM) and the prime factorizations,
    prove that the LCM of 9 and 14 is 126. -/
theorem lcm_9_14 : Int.lcm 9 14 = 126 := by
  sorry

end lcm_9_14_l727_727432


namespace mul_mixed_number_l727_727007

theorem mul_mixed_number (a b : ℝ) (c : ℚ) (h : c = 9 + 2/5) : 
  7 * c = (65 + 4/5 : ℝ) :=
by 
  simp [h, mul_add, mul_div_cancel_left, rat.cast_add, rat.cast_mul, rat.cast_one]
  sorry

end mul_mixed_number_l727_727007


namespace hyperbola_asymptote_l727_727864

theorem hyperbola_asymptote (m : ℚ) :
  (x y : ℝ) (h : 4 - m ≠ 0 ∧ m - 2 ≠ 0) :
  (∀ x y, (x^2 / (4 - m) + y^2 / (m - 2) = 1) → (y = x / 3 ∨ y = -x / 3)) → m = 7 / 4 :=
by
  intro x y h H
  sorry

end hyperbola_asymptote_l727_727864


namespace mowing_first_l727_727403

open Real

/-- Define the areas of the lawns -/
def areas :=
  let A_D := 3 * A_E
  let A_D := 4 * A_F
  ()

-- Define the mowing rates
def rates :=
  let R_F := (1/4) * R_D
  let R_F := (1/2) * R_E
  ()

-- Define the mowing times
def times (A_D A_E A_F R_D R_E R_F : ℝ) : ℝ × ℝ × ℝ :=
  (A_D / R_D, A_E / R_E, A_F / R_F)

theorem mowing_first 
  (A_D A_E A_F R_D R_E R_F : ℝ)
  (h1 : A_D = 3 * A_E)
  (h2 : A_D = 4 * A_F)
  (h3 : R_F = (1/4) * R_D)
  (h4 : R_F = (1/2) * R_E) :
  let (T_D, T_E, T_F) := times A_D A_E A_F R_D R_E R_F in
  T_E < T_D ∧ T_E < T_F :=
by
  sorry

end mowing_first_l727_727403


namespace smallest_a_divisible_by_65_l727_727946

theorem smallest_a_divisible_by_65 (a : ℤ) 
  (h : ∀ (n : ℤ), (5 * n ^ 13 + 13 * n ^ 5 + 9 * a * n) % 65 = 0) : 
  a = 63 := 
by {
  sorry
}

end smallest_a_divisible_by_65_l727_727946


namespace correct_propositions_l727_727857

-- Defining the propositions as Lean predicates
def prop1 (α : ℝ) (β : ℝ) : Prop :=
  (0 < α ∧ α < 90) ∧ (0 < β ∧ β < 90)

def prop2 (α : ℝ) : Prop :=
  0 < α ∧ α < 90

def prop3 (α β : ℝ) : Prop :=
  ∀ (α β : ℝ), α ≠ β → ¬ (α + 360 = β ∨ β + 360 = α)

def prop4 (α β : ℝ) (k : ℤ) : Prop :=
  β = α + k * 720

/- Defining a function to determine if a point is in the third quadrant.
   This would normally include some geometrical definitions and checks.
-/
def in_third_quadrant {α : ℝ} : Prop :=
  let P := (Real.tan α, Real.cos α) in
  Real.tan α < 0 ∧ Real.cos α < 0

def prop5 (α : ℝ) : Prop :=
  in_third_quadrant α → (180 < α ∧ α < 270)

-- Formulation of correct proposition as a proof problem
theorem correct_propositions (α β : ℝ) (k : ℤ) 
    (p1 : prop1 α β) 
    (p2 : prop2 α) 
    (p3 : prop3 α β) 
    (p4 : prop4 α β k) 
    (p5 : prop5 α) : 
  p4 ∧ p5 :=
begin
  sorry
end

end correct_propositions_l727_727857


namespace range_of_b_l727_727457

def ellipse (x y : ℝ) : Prop := x^2 + 2*y^2 = 3
def line (m b x y : ℝ) : Prop := y = m*x + b

theorem range_of_b (b : ℝ) : (∀ m : ℝ, ∃ x y : ℝ, ellipse x y ∧ line m b x y) ↔ 
  (b ∈ Icc (-Real.sqrt 6 / 2) (Real.sqrt 6 / 2)) :=
sorry

end range_of_b_l727_727457


namespace gcd_12012_18018_l727_727808

theorem gcd_12012_18018 : Int.gcd 12012 18018 = 6006 := 
by
  sorry

end gcd_12012_18018_l727_727808


namespace limit_of_f_at_1_l727_727479

noncomputable def f (x : ℝ) := (x - 1) * Real.exp x

theorem limit_of_f_at_1 :
  filter.tendsto (λ x, (f x - f 1) / (x - 1)) (nhds 1) (nhds Real.exp 1) :=
by
  sorry

end limit_of_f_at_1_l727_727479


namespace triangle_incenter_bisects_bisector_l727_727689

theorem triangle_incenter_bisects_bisector 
    (a b c : ℝ) (ABC : Type) 
    [Nonempty ABC] [IsTriangle ABC] 
    (BC AC AB : ABC → ℝ) 
    (h_arith_prog : BC 0 < AC 0 ∧ AC 0 < AB 0 ∧ (BC 0) + (AB 0) = 2 * AC 0)
    (B B1 O : ABC) 
    (bisector_B : B ≠ B1 → AngleBisector B B1)
    (circum_circle : Circumcircle ABC)
    (intersection : B ≠ B1 → circum_circle.intersect B B1 = true)
    (O_incenter : Incenter O) : 
  O ∈ Segment (B, B1) := by 
{
  sorry,
}

end triangle_incenter_bisects_bisector_l727_727689


namespace g_property_l727_727940

theorem g_property (g : ℝ → ℝ) (h : ∀ x y : ℝ, g x * g y - g (x * y) = 2 * x + 2 * y) :
  let n := 2
  let s := 14 / 3
  n = 2 ∧ s = 14 / 3 ∧ n * s = 28 / 3 :=
by {
  sorry
}

end g_property_l727_727940


namespace alexandra_magazines_l727_727733

theorem alexandra_magazines :
  let friday_magazines := 8
  let saturday_magazines := 12
  let sunday_magazines := 4 * friday_magazines
  let dog_chewed_magazines := 4
  let total_magazines_before_dog := friday_magazines + saturday_magazines + sunday_magazines
  let total_magazines_now := total_magazines_before_dog - dog_chewed_magazines
  total_magazines_now = 48 := by
  sorry

end alexandra_magazines_l727_727733


namespace pq_sum_is_527_l727_727560

theorem pq_sum_is_527 :
  ∃ (p q : ℕ) (hpqgcd : Nat.gcd p q = 1), 
  (p = 239 ∧ q = 288 ∧ p + q = 527) :=
begin
  use 239,
  use 288,
  use Nat.coprime.symm (Nat.coprime_of_dvd (by norm_num) (by norm_num)),
  split,
  { refl, },
  split,
  { refl, },
  { refl, }
end

end pq_sum_is_527_l727_727560


namespace interval_monotonic_increase_l727_727223

open Real

noncomputable def f (x : ℝ) : ℝ :=
  sin x ^ 2 - sqrt 3 * cos x * cos (x + π / 2)

theorem interval_monotonic_increase :
  ∀ x x' ∈ Icc 0 (π / 2), x <= x' → f x <= f x' :=
sorry

end interval_monotonic_increase_l727_727223


namespace number_of_integer_solutions_Q_is_one_l727_727951

def Q (x : ℤ) : ℤ := x^4 + 6 * x^3 + 13 * x^2 + 3 * x - 19

theorem number_of_integer_solutions_Q_is_one : 
    (∃! x : ℤ, ∃ k : ℤ, Q x = k^2) := 
sorry

end number_of_integer_solutions_Q_is_one_l727_727951


namespace find_m_l727_727856

theorem find_m (x y m : ℤ) (h1 : 3 * x + 4 * y = 7) (h2 : 5 * x - 4 * y = m) (h3 : x + y = 0) : m = -63 := by
  sorry

end find_m_l727_727856


namespace K_in_S2_l727_727548

/-- Define the circles with the given conditions -/
structure Circle (α : Type) :=
(center : α)
(radius : ℝ)

variables {α : Type} [metric_space α] {O O₁ O₂ B A K : α}

theorem K_in_S2
  (S : Circle α) (rS : S.radius = 2) 
  (S1 : Circle α) (rS1 : S1.radius = 1) 
  (S2 : Circle α) (rS2 : S2.radius = 1)
  (B : α) (tangent_S_S1 : dist S.center S1.center = 1)
  (A : α) (tangent_S1_S2 : dist S1.center S2.center = 2) 
  (not_tangent_S_S2 : dist S.center S2.center ≠ 1)
  (K : α) (intersection_AB_S : line_through S1.center B ∩ set_of (λ x, dist S.center x = 2) = {K}) :
  dist S2.center K = 1 :=
sorry

end K_in_S2_l727_727548


namespace evaluate_definite_integral_l727_727412

noncomputable def integrand (x : ℝ) : ℝ := x^2 + Real.exp x - (1 / 3)

theorem evaluate_definite_integral : ∫ x in 0..1, integrand x = Real.exp 1 - 1 :=
by 
  sorry

end evaluate_definite_integral_l727_727412


namespace constant_sequence_f5_geometric_sequence_fn_arithmetic_sequence_general_formula_l727_727465

-- Defining the function f(n) and basic sequences for the problem
def f (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  if h : 0 < n then 
    ∑ k in Finset.range n, a (k + 1) * Nat.choose n (k + 1)
  else 
    0

-- Statement for question 1
theorem constant_sequence_f5 : 
  (∀ n, a n = 1) → f a 5 = 31 := by
  sorry

-- Statement for question 2
theorem geometric_sequence_fn :
  (∀ n, a n = 3^(n - 1)) → f a n = (4^n - 1) / 3 := by
  sorry

-- Statement for question 3
theorem arithmetic_sequence_general_formula :
  (∃ a, (∀ n, a n = 1 + (n - 1) * 2 ) ∧ (∀ n > 0, f a n - 1 = (n - 1) * 2^n)) := by
  sorry

end constant_sequence_f5_geometric_sequence_fn_arithmetic_sequence_general_formula_l727_727465


namespace incorrect_option_B_l727_727738

-- Definitions of the given conditions
def optionA (a : ℝ) : Prop := (8 * a = 8 * a)
def optionB (a : ℝ) : Prop := (a - (0.08 * a) = 8 * a)
def optionC (a : ℝ) : Prop := (8 * a = 8 * a)
def optionD (a : ℝ) : Prop := (a * 8 = 8 * a)

-- The statement to be proved
theorem incorrect_option_B (a : ℝ) : 
  optionA a ∧ ¬optionB a ∧ optionC a ∧ optionD a := 
by
  sorry

end incorrect_option_B_l727_727738


namespace multiply_mixed_number_l727_727030

def mixed_to_improper_fraction (n : ℕ) (a b : ℕ) : ℚ :=
  n + a / b

def improper_to_mixed_number (q : ℚ) : ℕ × ℚ :=
  let n := q.natAbs
  let r := q.fract
  (n, r)

theorem multiply_mixed_number (x y n a b : ℕ) (h : y = mixed_to_improper_fraction n a b) :
  x * y = mixed_to_improper_fraction 65 4 5 :=
  sorry

end multiply_mixed_number_l727_727030


namespace integral_x_sub_e_x_l727_727786

open Real

theorem integral_x_sub_e_x : 
  ∫ x in 0..1, (x - exp x) = (3 / 2) - exp 1 :=
by sorry

end integral_x_sub_e_x_l727_727786


namespace convex_g_inequality_l727_727134

noncomputable def g (x : ℝ) : ℝ := x * Real.log x

theorem convex_g_inequality (a b : ℝ) (h : 0 < a ∧ a < b) :
  g a + g b - 2 * g ((a + b) / 2) > 0 := 
sorry

end convex_g_inequality_l727_727134


namespace find_possible_angles_l727_727894

noncomputable def possible_angles (A B C : ℝ) : Prop :=
  ∃ (a b c : ℝ),
    a + b + c = π ∧
    A = π / 6 ∧ -- 30 degrees in radians
    (B = 5 * π / 9 ∧ C = 5 * π / 18 ∨ B = 2 * π / 3 ∧ C = π / 6) -- (100, 50) or (120, 30) degrees in radians

theorem find_possible_angles (A B C : ℝ) :
  -- bisecting angle conditions should be stated, but skipping as they aren't simple to express directly
  -- However, we'll work under the assumption that other conditions and the triangle setup correctly imply this
  (A = 30 * π / 180 ∨ A = 120 * π / 180) →
  possible_angles A B C :=
sorry

end find_possible_angles_l727_727894


namespace largest_n_and_d_l727_727772

def g_1 (x : ℝ) : ℝ := real.sqrt (1 - x)

noncomputable def g : ℕ → ℝ → ℝ
| 1     := g_1
| (n+1) := λ x, g (n+1-1) (real.cbrt ((n+1)^3 - x))

theorem largest_n_and_d :
  let n := 4
  let d := -19619
  (∀ x, (64 - x ∈ set.Icc 27 512) → false) ∧
  (∀ x, (domain (g (n+1) x).nonempty → x = -134217664 ∧ x = d)) :=
begin
  let n := 4,
  let d := -19619,
  sorry
end

end largest_n_and_d_l727_727772


namespace missing_roots_of_polynomial_l727_727509

def polynomial : ℝ → ℝ :=
λ x, 12 * x^5 - 8 * x^4 - 45 * x^3 + 45 * x^2 + 8 * x - 12

theorem missing_roots_of_polynomial :
  (polynomial 1 = 0) ∧
  (polynomial 1.5 = 0) ∧
  (polynomial (-2) = 0) ∧
  (polynomial (2 / 3) = 0) ∧
  (polynomial (-1 / 2) = 0) :=
by
  -- Polynomial evaluation for given roots
  sorry

#eval polynomial 1  -- Should output 0
#eval polynomial 1.5  -- Should output 0
#eval polynomial (-2)  -- Should output 0
#eval polynomial (2 / 3)  -- Should output 0
#eval polynomial (-1 / 2)  -- Should output 0

end missing_roots_of_polynomial_l727_727509


namespace problem_statement_l727_727839

-- Given conditions
noncomputable def S : ℕ → ℝ := sorry
axiom S_3_eq_2 : S 3 = 2
axiom S_6_eq_6 : S 6 = 6

-- Prove that a_{13} + a_{14} + a_{15} = 32
theorem problem_statement : (S 15 - S 12) = 32 :=
by sorry

end problem_statement_l727_727839


namespace convert_base_8_to_10_l727_727768

theorem convert_base_8_to_10 :
  let n := 4532
  let b := 8
  n = 4 * b^3 + 5 * b^2 + 3 * b^1 + 2 * b^0 → 4 * 512 + 5 * 64 + 3 * 8 + 2 * 1 = 2394 :=
by
  sorry

end convert_base_8_to_10_l727_727768


namespace choose_marble_ways_l727_727932

theorem choose_marble_ways {total_marbles : ℕ} {choose_marbles : ℕ} {blue_marbles : ℕ} 
  (h_total : total_marbles = 9) 
  (h_choose : choose_marbles = 4) 
  (h_blue : blue_marbles = 2) 
  : ∃ (ways : ℕ), ways = 91 := 
begin
  have total_ways : (nat.choose total_marbles choose_marbles = 126) := calc
    (nat.choose 9 4) = 126 : by norm_num, 

  have no_blue_ways : (nat.choose (total_marbles - blue_marbles) choose_marbles = 35) := calc
    (nat.choose 7 4) = 35 : by norm_num,

  let ways := total_ways - no_blue_ways,
  use ways,
  simp *,
  exact calc
    126 - 35 = 91 : by norm_num
end

end choose_marble_ways_l727_727932


namespace gcd_of_12012_and_18018_l727_727800

theorem gcd_of_12012_and_18018 : Int.gcd 12012 18018 = 6006 := 
by
  -- Here we are assuming the factorization given in the conditions
  have h₁ : 12012 = 12 * 1001 := sorry
  have h₂ : 18018 = 18 * 1001 := sorry
  have gcd_12_18 : Int.gcd 12 18 = 6 := sorry
  -- This sorry will be replaced by the actual proof involving the above conditions to conclude the stated theorem
  sorry

end gcd_of_12012_and_18018_l727_727800


namespace Lagrange_interpolation_l727_727788

/-- 
  Prove that the Lagrange interpolation polynomial P_2(x) 
  passing through the points (-3, -5), (-1, -11), and (2, 10) 
  is equal to 2x^2 + 5x - 8.
-/
theorem Lagrange_interpolation :
  ∃ P : ℝ[X], 
    (P.eval (-3) = -5) ∧ 
    (P.eval (-1) = -11) ∧ 
    (P.eval 2 = 10) ∧ 
    (P = 2 * X^2 + 5 * X - 8) :=
by
  sorry

end Lagrange_interpolation_l727_727788


namespace male_students_count_l727_727524

theorem male_students_count (x y : ℕ) 
  (h1 : x + y = 100)
  (h2 : y ≤ 9) 
  (h3 : ∀ s, s.card = 10 → (∃ m ∈ s, m < x)): 
  x ≥ 91 := 
sorry

end male_students_count_l727_727524


namespace perp_OA_PQ_ap_squared_eq_two_ad_om_l727_727387

-- Definitions based on the conditions given in the problem statement
variables {ABC : Triangle}
variable {O : Point} -- Center of the circumscribed circle of triangle ABC
variables {A B C D E F P Q M : Point} -- Vertices and markings in the triangle

-- Acute triangle inscribed in a circle with center O
axiom triangle_inscribed (h_inscribed : O ∈ circumcircle ABC) : True

-- Altitudes of the triangle
axiom altitude_AD : is_altitude A D ABC
axiom altitude_BE : is_altitude B E ABC
axiom altitude_CF : is_altitude C F ABC

-- Line EF intersects the circle at points P and Q
axiom line_intersect_circle (h_intersect : E, F, P, Q ∈ circumcircle ABC) : True

-- Statement for part (a)
theorem perp_OA_PQ (h_inscribed : O ∈ circumcircle ABC) 
                    (h_altitude_AD : is_altitude A D ABC) 
                    (h_altitude_BE : is_altitude B E ABC) 
                    (h_altitude_CF : is_altitude C F ABC) 
                    (h_intersect : ∀ point, point ∈ [E, F] ∩ circumcircle ABC → point ∈ [P, Q]) :
                    perpendicular OA PQ := 
by sorry

-- Statement for part (b)
theorem ap_squared_eq_two_ad_om (h_inscribed : O ∈ circumcircle ABC) 
                                 (h_altitude_AD : is_altitude A D ABC) 
                                 (h_altitude_BE : is_altitude B E ABC) 
                                 (h_altitude_CF : is_altitude C F ABC) 
                                 (h_intersect : ∀ point, point ∈ [E, F] ∩ circumcircle ABC → point ∈ [P, Q]) 
                                 (midpoint_M : midpoint M B C) 
                                 (h_perpendicular : perpendicular OA PQ) : 
                                 (AP ^ 2 = 2 * AD * OM) :=
by sorry

end perp_OA_PQ_ap_squared_eq_two_ad_om_l727_727387


namespace smallest_perfect_square_5336100_l727_727670

def smallestPerfectSquareDivisibleBy (a b c d : Nat) (s : Nat) : Prop :=
  ∃ k : Nat, s = k * k ∧ s % a = 0 ∧ s % b = 0 ∧ s % c = 0 ∧ s % d = 0

theorem smallest_perfect_square_5336100 :
  smallestPerfectSquareDivisibleBy 6 14 22 30 5336100 :=
sorry

end smallest_perfect_square_5336100_l727_727670


namespace convert_base_8_to_10_l727_727767

theorem convert_base_8_to_10 :
  let n := 4532
  let b := 8
  n = 4 * b^3 + 5 * b^2 + 3 * b^1 + 2 * b^0 → 4 * 512 + 5 * 64 + 3 * 8 + 2 * 1 = 2394 :=
by
  sorry

end convert_base_8_to_10_l727_727767


namespace exists_zero_in_interval_l727_727422

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) - 2/x

theorem exists_zero_in_interval :
    (f (Real.exp 1 - 1) < 0) →
    (f 2 > 0) →
    (∃ c : ℝ, c ∈ Set.Ioo (Real.exp 1 - 1) 2 ∧ f c = 0) :=
by
  intros h₁ h₂
  have h3 : ContinuousOn f (Set.Icc (Real.exp 1 - 1) 2) := sorry
  exact IntermediateValueTheorem_interval h₁ h₂ h3

end exists_zero_in_interval_l727_727422


namespace g_of_neg2_l727_727046

def g (x : ℚ) : ℚ := (2 * x - 3) / (4 * x + 5)

theorem g_of_neg2 : g (-2) = 7 / 3 := by
  sorry

end g_of_neg2_l727_727046


namespace find_value_of_f_2019_div_2_l727_727115

noncomputable def f : ℝ → ℝ
| x => if 0 < x ∧ x < 1 then 3^x - 1 else
        if -1 < x ∧ x < 0 then -((3^-x) - 1) else 0  

axiom periodic : ∀ x : ℝ, f (x + 2) = f x
axiom odd : ∀ x : ℝ, f (-x) = -f x
axiom interval1 : ∀ x : ℝ, 0 < x ∧ x < 1 → f x = 3^x - 1

theorem find_value_of_f_2019_div_2 : f (2019 / 2) = 1 - Real.sqrt 3 := 
by
  sorry

end find_value_of_f_2019_div_2_l727_727115


namespace roger_has_more_candies_l727_727261

def candies_sandra_bag1 : ℕ := 6
def candies_sandra_bag2 : ℕ := 6
def candies_roger_bag1 : ℕ := 11
def candies_roger_bag2 : ℕ := 3

def total_candies_sandra := candies_sandra_bag1 + candies_sandra_bag2
def total_candies_roger := candies_roger_bag1 + candies_roger_bag2

theorem roger_has_more_candies : (total_candies_roger - total_candies_sandra) = 2 := by
  sorry

end roger_has_more_candies_l727_727261


namespace sum_of_digits_eleven_l727_727889

-- Definitions for the problem conditions
def distinct_digits (p q r : Nat) : Prop :=
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p > 0 ∧ q > 0 ∧ r > 0 ∧ p < 10 ∧ q < 10 ∧ r < 10

def is_two_digit_prime (n : Nat) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n.Prime

def concat_digits (x y : Nat) : Nat :=
  10 * x + y

def problem_conditions (p q r : Nat) : Prop :=
  distinct_digits p q r ∧
  is_two_digit_prime (concat_digits p q) ∧
  is_two_digit_prime (concat_digits p r) ∧
  is_two_digit_prime (concat_digits q r) ∧
  (concat_digits p q) * (concat_digits p r) = 221

-- Lean 4 statement to prove the sum of p, q, r is 11
theorem sum_of_digits_eleven (p q r : Nat) (h : problem_conditions p q r) : p + q + r = 11 :=
sorry

end sum_of_digits_eleven_l727_727889


namespace number_of_non_similar_1200_pointed_stars_l727_727156

def euler_totient (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ m, Nat.gcd m n = 1).card

noncomputable def num_non_similar_1200_pointed_stars : ℕ :=
  euler_totient 1200 / 2

theorem number_of_non_similar_1200_pointed_stars :
  num_non_similar_1200_pointed_stars = 160 :=
by
  sorry

end number_of_non_similar_1200_pointed_stars_l727_727156


namespace part_a_part_b_l727_727934

-- Define the problem setup
variables (ABC : Type) [triangle ABC] (P : Point ABC) (BC AC AB : Line ABC)

-- Parabolas definitions based on the given conditions
def parabola (P: Point ABC) (directrix: Line ABC) : Set (Point ABC) := 
  { Q | dist Q P = dist Q directrix }

def parabola_P_A := parabola P BC
def parabola_P_B := parabola P AC
def parabola_P_C := parabola P AB

noncomputable def intersection_points_P_B_P_C : list (Point ABC) := sorry

-- Part (a)
theorem part_a (Q : Point ABC) (hQ: Q ∈ (parabola_P_B P AC ∩ parabola_P_C P AB)) 
  : same_side P Q AB ∧ same_side P Q AC := 
  sorry

-- Part (b)
theorem part_b (hI: intersection_points_P_B_P_C.length = 2)
  (inter_pts : list Point ABC := intersection_points_P_B_P_C) :
  ∃ concurrency_point : Point ABC, 
  intersect_line_through_pts (inter_pts.nth 0) (inter_pts.nth 1) ∧
  intersect_line_through_pts (other_intersection_pts parabola_P_A parabola_P_C) ∧
  intersect_line_through_pts (other_intersection_pts parabola_P_A parabola_P_B) :=
  sorry

end part_a_part_b_l727_727934


namespace sum_of_rational_roots_of_h_l727_727056

noncomputable def h (x : ℚ) : ℚ := x^3 - 7 * x^2 + 8 * x + 6

theorem sum_of_rational_roots_of_h : (∑ x in {x : ℚ | h x = 0}, x) = 0 := 
sorry

end sum_of_rational_roots_of_h_l727_727056


namespace one_meter_eq_skips_l727_727501

variable (a b c d e f : ℝ)

-- Given conditions as definitions
def hops_from_skips (hops : ℝ) : ℝ := 3 * b / (2 * a) * hops
def jumps_from_hops (jumps : ℝ) : ℝ := 5 * d / (4 * c) * jumps
def meters_from_jumps (meters : ℝ) : ℝ := 6 * e / (7 * f) * meters

theorem one_meter_eq_skips :
  let skips := hops_from_skips (jumps_from_hops (meters_from_jumps 1))
  in skips = 90 * b * d * e / (56 * a * c * f) :=
by sorry

end one_meter_eq_skips_l727_727501


namespace majorization_operations_l727_727977

-- Defining majorization
def majorizes (α β : ℝ × ℝ × ℝ) : Prop :=
  α.1 >= β.1 ∧ (α.1 + α.2 >= β.1 + β.2) ∧ (α.1 + α.2 + α.3 = β.1 + β.2 + β.3)

-- Defining operations
def op1 (α : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (α.1 - 1, α.2 + 1, α.3)

def op2 (α : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (α.1 - 1, α.2, α.3 + 1)

def op3 (α : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (α.1, α.2 - 1, α.3 + 1)

-- Proving that β can be obtained from α using the operations
theorem majorization_operations (α β : ℝ × ℝ × ℝ) :
  (majorizes α β ↔
  ∃ γ, ((γ = op1 α ∨ γ = op2 α ∨ γ = op3 α) ∧ majorizes γ β)) := sorry

end majorization_operations_l727_727977


namespace lines_parallel_iff_a_eq_neg2_l727_727468

def line₁_eq (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y - 1 = 0
def line₂_eq (a : ℝ) (x y : ℝ) : Prop := x + (a + 1) * y - 1 = 0

theorem lines_parallel_iff_a_eq_neg2 (a : ℝ) :
  (∀ x y : ℝ, line₁_eq a x y → line₂_eq a x y) ↔ a = -2 :=
by sorry

end lines_parallel_iff_a_eq_neg2_l727_727468


namespace Steiner_theorem_l727_727464

theorem Steiner_theorem (A B C D P : Point) 
(h1 : ∃ l : Line, ∀ (X : Point), X ∈ {A, B, C, D} → 
  is_foot (drop_perpendicular P (line_through X (next_vertex X {A, B, C, D}))) X l) :
∃ l' : Line, ∀ (X : Point), X ∈ {A, B, C, D} → 
    is_foot (drop_perpendicular (\orthocenter (triangle_using_side X (next_vertex X {A, B, C, D}))) (line_through X (next_vertex X {A, B, C, D}))) X l' :=
sorry

end Steiner_theorem_l727_727464


namespace polynomial_inequality_holds_l727_727957

theorem polynomial_inequality_holds (a : ℝ) : (∀ x : ℝ, x^4 + (a-2)*x^2 + a ≥ 0) ↔ a ≥ 4 - 2 * Real.sqrt 3 := 
by
  sorry

end polynomial_inequality_holds_l727_727957


namespace units_digit_of_A_is_1_l727_727144

-- Definition of A
def A : ℕ := 2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) + 1

-- Main theorem stating that the units digit of A is 1
theorem units_digit_of_A_is_1 : (A % 10) = 1 :=
by 
  -- Given conditions about powers of 3 and their properties in modulo 10
  sorry

end units_digit_of_A_is_1_l727_727144


namespace area_of_section_of_median_l727_727184

-- Define the appropriate structures and properties
structure Triangle := 
  (vertex : Type)
  (hypotenuse : ℝ)
  (angle1 : ℝ)
  (angle2 : ℝ)
  (angle3 : ℝ)

-- State the problem conditions
def givenTriangle : Triangle := 
  { vertex := ℝ,
    hypotenuse := 16,
    angle1 := 45,
    angle2 := 45,
    angle3 := 90 }

-- Define the function that calculates the area
def calculateArea (t : Triangle) : ℝ :=
  let leg := t.hypotenuse / (real.sqrt 2) in
  (1 / 2) * leg * leg

-- Define the proof statement
theorem area_of_section_of_median (t : Triangle) 
  (h : t = givenTriangle) : calculateArea t / 2 = 32 :=
by 
  cases t with
  | mk vertex hypotenuse angle1 angle2 angle3 =>
    rw [h]
    have leg := givenTriangle.hypotenuse / (real.sqrt 2)
    have area := (1 / 2) * leg * leg
    rw [calculateArea, area, mul_div_cancel_left, real.sqrt_nat_eq m' (nat.succ_pos 1)]
    sorry

end area_of_section_of_median_l727_727184


namespace work_done_in_isothermal_process_l727_727245

-- Definitions for the problem
def ideal_monatomic_gas (n : ℕ) := true -- n represents number of moles

def isobaric_process (work_isobaric : ℝ):= 
  ∃ (p : ℝ) (ΔV : ℝ), work_isobaric = p * ΔV

def heat_added_isobaric (ΔU : ℝ) (work_isobaric : ℝ) := 
  let R := 8.314 
  ΔU = (3 / 2) * R * ΔT ∧ Q_isobaric = ΔU + work_isobaric

def isothermal_process (work_isothermal : ℝ) {Q_isothermal : ℝ} := 
  Q_isothermal = work_isothermal

-- The theorem to be proved
theorem work_done_in_isothermal_process 
  (n : ℕ) (work_isobaric : ℝ) (ΔU : ℝ) (ΔT : ℝ) (R : ℝ := 8.314):
  ideal_monatomic_gas n ∧ isobaric_process work_isobaric ∧ heat_added_isobaric ΔU work_isobaric
  ∧ isothermal_process work_isothermal {Q_isobaric := 25} :=
by
  -- Given one mole of the ideal monatomic gas
  have h1 : ideal_monatomic_gas 1 := trivial,
  -- The work done by gas in isobaric process is 10 Joules
  have h2 : isobaric_process 10 := 
    by 
      -- Existence of pressure and volume change for work calculation
      let p := 1
      let ΔV := 10
    _,
  -- The heat added to gas in isobaric process equals the work done
  have h3 : heat_added_isobaric ΔU 10 := by 
    let Q_isobaric := 25
    -- Calculation of heat based on given conditions
    _,
  -- The isothermal process where work done is equal to heat added, which equals heat added in isobaric process
  have h4 : isothermal_process 25 {Q_isothermal := 25} := 
    by
      let Q_isothermal := 25
      let work_isothermal := 25
    _,
  sorry -- Proof is not required, so we end with sorry.

end work_done_in_isothermal_process_l727_727245


namespace circle_problem_solution_l727_727641

noncomputable def liar_count_problem (n : ℕ) (circle : ZMod n → Prop) : ℕ := sorry

theorem circle_problem_solution :
  let n := 22
  let is_knight := λ p circle, ∀ i : ZMod n, 
                     circle i ↔ ¬circle ((i + 10) % n)
  ∃! liars_count, 
    (liars_count = liar_count_problem n (λ i, ¬(is_knight i)) ∧ liar_count_problem n (λ i, ¬(is_knight i)) = 20)

end circle_problem_solution_l727_727641


namespace greatest_value_expression_l727_727952

theorem greatest_value_expression 
  (n : ℕ) 
  (h1 : n ≥ 5) 
  (points : Finₙ → (ℝ × ℝ)) 
  (h2 : ∀ (i j k : Finₙ), i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬ collinear (points i) (points j) (points k)) 
  (erase_point : Π (k : Fin (n-2)), Finₙ) 
  (convex_hull_vertices : Π (i : Fin (n-2)), ℕ) 
  (h3 : convex_hull_vertices ⟨n-3, by linarith⟩ = 3) :
  ∃ (erases : List (Finₙ)), 
    (|convex_hull_vertices ⟨0, by linarith⟩ - convex_hull_vertices ⟨1, by linarith⟩|
    + |convex_hull_vertices ⟨1, by linarith⟩ - convex_hull_vertices ⟨2, by linarith⟩|
    + ...
    + |convex_hull_vertices ⟨n-4, by linarith⟩ - convex_hull_vertices ⟨n-3, by linarith⟩|)
    = 2n - 8 :=
sorry

end greatest_value_expression_l727_727952


namespace solve_for_p_l727_727517

theorem solve_for_p (a b c p t : ℝ) (h1 : a + b + c + p = 360) (h2 : t = 180 - c) : 
  p = 180 - a - b + t :=
by
  sorry

end solve_for_p_l727_727517


namespace multiplication_with_mixed_number_l727_727017

-- Define mixed numbers as rational numbers for proper calculation
def mixed_to_rational (whole : ℕ) (num : ℕ) (den : ℕ) : ℚ :=
  whole + num / den

-- 7 * (9 + 2/5)
def lhs : ℚ := 7 * mixed_to_rational 9 2 5

-- 65 + 4/5
def rhs : ℚ := mixed_to_rational 65 4 5

theorem multiplication_with_mixed_number : lhs = rhs := by
  sorry

end multiplication_with_mixed_number_l727_727017


namespace correct_algorithm_statement_l727_727676

def reversible : Prop := false -- Algorithms are generally not reversible.
def endless : Prop := false -- Algorithms should not run endlessly.
def unique_algo : Prop := false -- Not always one single algorithm for a task.
def simple_convenient : Prop := true -- Algorithms should be simple and convenient.

theorem correct_algorithm_statement : simple_convenient = true :=
by
  sorry

end correct_algorithm_statement_l727_727676


namespace gcd_12012_18018_l727_727811

theorem gcd_12012_18018 : Int.gcd 12012 18018 = 6006 := 
by
  sorry

end gcd_12012_18018_l727_727811


namespace periodic_G_limit_sum_l727_727547

-- Function Definitions and Conditions
variables {T : ℝ} (f : ℝ → ℝ) (F : ℝ → ℝ)
variable [Continuous f]
variable [∀ x : ℝ, ∃ T : ℝ, f (x + T) = f x] -- Periodic condition for f
variable [∀ x : ℝ, F' x = f x] -- F is a primitive (antiderivative) of f

-- Propositions to be Proved
theorem periodic_G (f : ℝ → ℝ) (F : ℝ → ℝ) (T : ℝ) [Continuous f]
  [∀ x : ℝ, ∃ T : ℝ, f (x + T) = f x] [∀ x : ℝ, F' x = f x] :
  ∃ C : ℝ, ∀ x : ℝ, F (x + T) = F x + C :=
sorry

theorem limit_sum (f : ℝ → ℝ) (F : ℝ → ℝ) (T : ℝ) [Continuous f]
  [∀ x : ℝ, ∃ T : ℝ, f (x + T) = f x] [∀ x : ℝ, F' x = f x] :
  (lim n to_top (λ n, ∑ i in range n, F i / (n^2 + i^2))) = (ln 2 / (2 * T) * ∫ (x : ℝ) in 0..T, f x) :=
sorry

end periodic_G_limit_sum_l727_727547


namespace student_tickets_sold_l727_727389

theorem student_tickets_sold (S NS : ℕ) (h1 : S + NS = 150) (h2 : 5 * S + 8 * NS = 930) : S = 90 :=
by
  sorry

end student_tickets_sold_l727_727389


namespace smallest_value_of_a_plus_b_l727_727830

theorem smallest_value_of_a_plus_b :
  ∃ (a b : ℕ), (2^6 * 3^3 * 5^4 = a^b) ∧ (a + b = 602) :=
begin
  use [600, 2],
  split,
  { norm_num },
  { norm_num }
end

end smallest_value_of_a_plus_b_l727_727830


namespace gcd_12012_18018_l727_727807

theorem gcd_12012_18018 : Int.gcd 12012 18018 = 6006 := 
by
  sorry

end gcd_12012_18018_l727_727807


namespace fib_ratio_l727_727214
noncomputable theory

def φ : ℝ := (1 + Real.sqrt 5) / 2

def Fibonacci : ℕ → ℤ
| 0       := 0
| 1       := 1
| (n+2) := Fibonacci n + Fibonacci (n+1)

def a_n (n : ℕ) : ℝ := φ ^ n * (Fibonacci n : ℝ)

-- statement of the problem
theorem fib_ratio : ∃ (A B : ℚ), 
  (a_n 30 + a_n 29) / (a_n 26 + a_n 25) = (A : ℝ) + (B : ℝ) * Real.sqrt 5 ∧
  A + B = 188 := sorry

end fib_ratio_l727_727214


namespace number_of_possible_values_of_S_is_1601_l727_727575

open Finset

def S_possible_values_count : ℕ := 
  let elems := range 1 101 -- {1, 2, ..., 100}
  let min_S := (80 * 81) / 2 -- smallest sum S when subset is {1, 2, ..., 80}
  let max_S := (100 * 101) / 2 - (20 * 21) / 2 -- largest sum S when subset is {21, 22, ..., 100}
  max_S - min_S + 1 -- number of possible values of S

theorem number_of_possible_values_of_S_is_1601 : 
  let elems := range 1 101
  ∀ (A : Finset ℕ) (hA : A.card = 80) (h : A ⊆ elems),
    1601 = S_possible_values_count :=
by
  sorry

end number_of_possible_values_of_S_is_1601_l727_727575


namespace arrangement_of_volunteers_l727_727059

-- The conditions and the question translated to Lean 4
theorem arrangement_of_volunteers: 
  let volunteers := ["Alice", "Bob", "Charlie", "David", "Xiao Li", "Xiao Wang"]
  let areas := ["A", "B", "C", "D"]
  let cond1 := (∀ x, x ∈ volunteers → ¬(x = "Xiao Li" ∧ x = "Xiao Wang"))
  ∃ (arrangements : List (List String)), 
    arrangements.length = 4 ∧
    arrangements[0].length = 1 ∧ arrangements[1].length = 1 ∧
    arrangements[2].length = 2 ∧ arrangements[3].length = 2 ∧
    (∀ x ∈ arrangements, List.Subset x volunteers) ∧
    (∀ v ∈ volunteers, ∃ a ∈ arrangements, v ∈ a) ∧
    cond1 → arrangements.length = 156 :=
begin
  sorry
end

end arrangement_of_volunteers_l727_727059


namespace compute_fraction_at_six_l727_727755

theorem compute_fraction_at_six (x : ℕ) (h : x = 6) : (x^6 - 16 * x^3 + 64) / (x^3 - 8) = 208 := by
  sorry

end compute_fraction_at_six_l727_727755


namespace hyperbola_line_area_l727_727473

noncomputable def hyperbola_eccentricity : ℝ := real.sqrt 3

noncomputable def hyperbola_point : ℝ × ℝ := (real.sqrt 3, 0)

noncomputable def focus_right : ℝ × ℝ := (3, 0)

noncomputable def line_slope : ℝ := real.sqrt 3

noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  x ^ 2 / 3 - y ^ 2 / 6 = 1

noncomputable def line_equation (x y : ℝ) : Prop :=
  y = line_slope * (x - 3)

theorem hyperbola_line_area :
  ∃ (A B : ℝ × ℝ), hyperbola_equation A.1 A.2 ∧ hyperbola_equation B.1 B.2 ∧ line_equation A.1 A.2 ∧ line_equation B.1 B.2 ∧
  1 / 2 * abs (A.1 * B.2) = 36 :=
begin
  sorry
end

end hyperbola_line_area_l727_727473


namespace problem_statement_l727_727050

def f : ℝ → ℝ := sorry

axiom f_condition1 : ∀ x, f''' x > 2 - f x
axiom f_condition2 : f 0 = 6

theorem problem_statement : {x : ℝ | e^x * f x > 2 * e^x + 4} = {x : ℝ | 0 < x} := 
by
  sorry

end problem_statement_l727_727050


namespace mul_mixed_number_l727_727008

theorem mul_mixed_number (a b : ℝ) (c : ℚ) (h : c = 9 + 2/5) : 
  7 * c = (65 + 4/5 : ℝ) :=
by 
  simp [h, mul_add, mul_div_cancel_left, rat.cast_add, rat.cast_mul, rat.cast_one]
  sorry

end mul_mixed_number_l727_727008


namespace qin_jiushao_operations_count_l727_727661

noncomputable def polynomial := λ (x : ℕ), 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x + 1

theorem qin_jiushao_operations_count :
  ∃ (multiplications additions : ℕ), 
    multiplications = 5 ∧ 
    additions = 5 ∧ 
    ∀ x, polynomial x = (((((5*x + 4)*x + 3)*x + 2)*x + 1)*x + 1) :=
by
  sorry

end qin_jiushao_operations_count_l727_727661


namespace initial_amount_l727_727205

theorem initial_amount 
  (spend1 spend2 left : ℝ)
  (hspend1 : spend1 = 1.75) 
  (hspend2 : spend2 = 1.25) 
  (hleft : left = 6.00) : 
  spend1 + spend2 + left = 9.00 := 
by
  -- Proof is omitted
  sorry

end initial_amount_l727_727205


namespace domain_h_l727_727663

def h (x : ℝ) : ℝ := (2 * x - 3) / (x - 5)

theorem domain_h :
  {x : ℝ | ∃ y : ℝ, y = h x} = {x : ℝ | x ≠ 5} :=
sorry

end domain_h_l727_727663


namespace find_parabola_l727_727292

variable (P : ℝ × ℝ)
variable (a b : ℝ)

def parabola1 (P : ℝ × ℝ) (a : ℝ) := P.2^2 = 4 * a * P.1
def parabola2 (P : ℝ × ℝ) (b : ℝ) := P.1^2 = 4 * b * P.2

theorem find_parabola (hP : P = (-2, 4)) :
  (∃ a, parabola1 P a ∧ P.2^2 = -8 * P.1) ∨ 
  (∃ b, parabola2 P b ∧ P.1^2 = P.2) := by
  sorry

end find_parabola_l727_727292


namespace basketball_score_l727_727698

theorem basketball_score (score_game1 : ℕ) (score_game2 : ℕ) (score_game3 : ℕ) (score_game4 : ℕ) (score_total_games8 : ℕ) (score_total_games9 : ℕ) :
  score_game1 = 18 ∧ score_game2 = 22 ∧ score_game3 = 15 ∧ score_game4 = 20 ∧ 
  (score_game1 + score_game2 + score_game3 + score_game4) / 4 < score_total_games8 / 8 ∧ 
  score_total_games9 / 9 > 19 →
  score_total_games9 - score_total_games8 ≥ 21 :=
by
-- proof steps would be provided here based on the given solution
sorry

end basketball_score_l727_727698


namespace angles_same_terminal_side_l727_727419

theorem angles_same_terminal_side (a b : ℝ) (h1 : a = 30) (h2 : b = 390) : 
    a ∈ interval 0 720 ∧ b ∈ interval 0 720 ∧ 
    (∃ k : ℤ, a = -1050 + 360 * k) ∧ 
    (∃ k : ℤ, b = -1050 + 360 * k) :=
by 
  sorry

end angles_same_terminal_side_l727_727419


namespace probability_dice_roll_l727_727967

theorem probability_dice_roll :
  (1 / 2) * (1 / 3) = 1 / 6 :=
by
  -- Here you can add the proof steps if needed
  sorry

end probability_dice_roll_l727_727967


namespace units_digit_sum_powers_of_2_is_5_l727_727587

theorem units_digit_sum_powers_of_2_is_5 :
  let pattern := ∀ (x: ℕ), ∀ (n: ℕ), (x - 1) * (Nat.sum (λ i => x^(n - i)) (range (n + 1))) = x^(n + 1) - 1 in
  let cyclic_units_digits := [2, 4, 8, 6] in
  let units_digit_of_2_power (n : ℕ) := cyclic_units_digits[(n % 4) - 1] in
  (let units_digit := (units_digit_of_2_power 2024 - 1) % 10 in
   units_digit == 5).
sorry

end units_digit_sum_powers_of_2_is_5_l727_727587


namespace triangle_angle_solution_l727_727919

theorem triangle_angle_solution :
  let (a, b, c) := (2, Real.sqrt 2, Real.sqrt 3 + 1) in
  ∃ A B C, 
    A = Real.pi / 4 ∧ 
    B = Real.pi / 6 ∧ 
    C = 7 * Real.pi / 12 ∧ 
    A + B + C = Real.pi ∧ 
    ∀ (a b c : ℝ), a = 2 → b = Real.sqrt 2 → c = Real.sqrt 3 + 1 → 
    cos A = (b^2 + c^2 - a^2) / (2 * b * c) ∧ 
    sin B = b / c * sin A ∧ 
    cos C = cos (Real.pi - (A + B)) :=
by
  sorry

end triangle_angle_solution_l727_727919


namespace value_of_a_when_x_is_5_l727_727506

noncomputable def find_a (x a : ℝ): Prop := 
  2 * x + 3 * a = 4

theorem value_of_a_when_x_is_5 :
  find_a 5 (-2) :=
by
  unfold find_a
  norm_num
  exact eq.refl _ -- adding sorry to allow the rest of the code to defer remainder of proof

end value_of_a_when_x_is_5_l727_727506


namespace triangle_area_is_18_l727_727313

noncomputable def area_of_triangle (y_8 y_2_2x y_2_minus_2x : ℝ) : ℝ :=
  let intersect1 : ℝ × ℝ := (3, 8)
  let intersect2 : ℝ × ℝ := (-3, 8)
  let intersect3 : ℝ × ℝ := (0, 2)
  let base := 3 - -3
  let height := 8 - 2
  (1 / 2 ) * base * height

theorem triangle_area_is_18 : 
  area_of_triangle (8) (2 + 2 * x) (2 - 2 * x) = 18 := 
  by
    sorry

end triangle_area_is_18_l727_727313


namespace prime_div_or_coprime_l727_727566

open Nat

theorem prime_div_or_coprime (p n : ℕ) (hp : Prime p) : p ∣ n ∨ gcd p n = 1 :=
by sorry

end prime_div_or_coprime_l727_727566


namespace solve_for_star_l727_727167

theorem solve_for_star 
  (x : ℝ) 
  (h : 45 - (28 - (37 - (15 - x))) = 58) : 
  x = 19 :=
by
  -- Proof goes here. Currently incomplete, so we use sorry.
  sorry

end solve_for_star_l727_727167


namespace at_least_two_participants_solved_exactly_five_l727_727189

open Nat Real

variable {n : ℕ}  -- Number of participants
variable {pij : ℕ → ℕ → ℕ} -- Number of contestants who correctly answered both the i-th and j-th problems

-- Conditions as definitions in Lean 4
def conditions (n : ℕ) (pij : ℕ → ℕ → ℕ) : Prop :=
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 6 → pij i j > (2 * n) / 5) ∧
  (∀ k, ¬ (∀ i, 1 ≤ i ∧ i ≤ 6 → pij k i = 1))

-- Main theorem statement
theorem at_least_two_participants_solved_exactly_five (n : ℕ) (pij : ℕ → ℕ → ℕ) (h : conditions n pij) : ∃ k₁ k₂, k₁ ≠ k₂ ∧ (∃ i, 1 ≤ i ∧ i ≤ 6 ∧ pij k₁ i = 1) ∧ (∃ i, 1 ≤ i ∧ i ≤ 6 ∧ pij k₂ i = 1) := sorry

end at_least_two_participants_solved_exactly_five_l727_727189


namespace transformed_sin_l727_727483

noncomputable def transformed_function (x : ℝ) : ℝ :=
  sin (-(3 / 2) * x + 2 * π / 3)

theorem transformed_sin :
  ∀ x, transformed_function x = sin (-(3 / 2) * (x - π / 3) + π / 6) :=
begin
  intro x,
  simp [transformed_function],
  sorry
end

end transformed_sin_l727_727483


namespace smaller_solution_of_quadratic_l727_727438

theorem smaller_solution_of_quadratic :
  (∃ x y : ℝ, x ≠ y ∧ (x^2 - 13 * x + 36 = 0) ∧ (y^2 - 13 * y + 36 = 0) ∧ min x y = 4) :=
sorry

end smaller_solution_of_quadratic_l727_727438


namespace smaller_solution_of_quadratic_l727_727437

theorem smaller_solution_of_quadratic :
  (∃ x y : ℝ, x ≠ y ∧ (x^2 - 13 * x + 36 = 0) ∧ (y^2 - 13 * y + 36 = 0) ∧ min x y = 4) :=
sorry

end smaller_solution_of_quadratic_l727_727437


namespace g_neither_even_nor_odd_l727_727539

def g (x : ℝ) : ℝ := 3^(x^2 - 4*x + 3) - |x + 1|

theorem g_neither_even_nor_odd : ¬(∀ x, g(-x) = g(x)) ∧ ¬(∀ x, g(-x) = -g(x)) :=
by 
  sorry

end g_neither_even_nor_odd_l727_727539


namespace circle_center_sum_l727_727776

theorem circle_center_sum (x y : ℝ) (h : (x - 2)^2 + (y + 1)^2 = 15) : x + y = 1 :=
sorry

end circle_center_sum_l727_727776


namespace probability_of_M_probability_of_N_l727_727713

-- Definition for the group of students
def students : List (String × String) := 
  [("A", "M"), ("B", "M"), ("C", "M"), ("X", "F"), ("Y", "F"), ("Z", "F")]

-- The set of all possible choices of 2 students from the list of students
def possible_outcomes := (students.product students).filter (λ p, p.1 ≠ p.2)

-- Event M: Both selected students are female
def event_M := possible_outcomes.filter (λ p, p.1.2 = "F" ∧ p.2.2 = "F")

-- Event N: The selected students are from different grades and consist of one male and one female student
def event_N := possible_outcomes.filter (λ p, p.1.2 ≠ p.2.2 ∧ (p.1.2 = "M" ∨ p.2.2 = "M"))

-- Proving both questions:
theorem probability_of_M : (event_M.length: ℚ) / (possible_outcomes.length: ℚ) = 1 / 5 := by
  sorry

theorem probability_of_N : (event_N.length: ℚ) / (possible_outcomes.length: ℚ) = 2 / 5 := by
  sorry

end probability_of_M_probability_of_N_l727_727713


namespace percentage_increase_first_year_l727_727355

theorem percentage_increase_first_year (P : ℝ) (x : ℝ) :
  (1 + x / 100) * 0.7 = 1.0499999999999998 → x = 50 := 
by
  sorry

end percentage_increase_first_year_l727_727355


namespace team_leader_prize_l727_727618

theorem team_leader_prize 
    (number_of_students : ℕ := 10)
    (number_of_team_members : ℕ := 9)
    (team_member_prize : ℕ := 200)
    (additional_leader_prize : ℕ := 90)
    (total_prize : ℕ)
    (leader_prize : ℕ := total_prize - (number_of_team_members * team_member_prize))
    (average_prize : ℕ := (total_prize + additional_leader_prize) / number_of_students)
: leader_prize = 300 := 
by {
  sorry  -- Proof omitted
}

end team_leader_prize_l727_727618


namespace golden_ratio_expression_evaluation_l727_727272

theorem golden_ratio_expression_evaluation :
  let φ := 2 * Real.sin (Real.pi / 10)
  in (8 * φ^2 * (Real.cos (Real.pi / 10))^2) / (2 - φ) = 2 :=
by
  let φ := 2 * Real.sin (Real.pi / 10)
  calc
  sorry

end golden_ratio_expression_evaluation_l727_727272


namespace female_students_transfer_l727_727324

theorem female_students_transfer (x y z : ℕ) 
  (h1 : ∀ B : ℕ, B = x - 4) 
  (h2 : ∀ C : ℕ, C = x - 5)
  (h3 : ∀ B' : ℕ, B' = x - 4 + y - z)
  (h4 : ∀ C' : ℕ, C' = x + z - 7) 
  (h5 : x - y + 2 = x - 4 + y - z)
  (h6 : x - 4 + y - z = x + z - 7) 
  (h7 : 2 = 2) :
  y = 3 ∧ z = 4 := 
by 
  sorry

end female_students_transfer_l727_727324


namespace parabola_focus_directrix_parabola_chord_length_l727_727493

theorem parabola_focus_directrix (p : ℝ) (h : 2 * p = 6) :
  let focus := (p / 2, 0)
      directrix := -p / 2
  in focus = (3 / 2, 0) ∧ directrix = -3 / 2 :=
by
  sorry

theorem parabola_chord_length (A B : ℝ × ℝ) (p : ℝ) (h : 2 * p = 6)
  (focus := (p / 2, 0)) (line : ℝ × ℝ → ℝ)
  (h_line : ∀ (x : ℝ), line (x, focus.2) = x - p / 2)
  (parabola : ℝ × ℝ → Prop) (h_parabola : ∀ (x y : ℝ), parabola (x, y) ↔ y ^ 2 = 6 * x)
  (intersects : ∀ {x y : ℝ}, parabola (x, y) → line (x, focus.2) = 0 → (x = A.1 ∨ x = B.1))
  (A_eq : A.1 + B.1 = 9) :
  let length_AB := A.1 + B.1 + p
  in length_AB = 12 :=
by
  sorry

end parabola_focus_directrix_parabola_chord_length_l727_727493


namespace work_done_isothermal_l727_727247

-- Definitions based on the conditions
def n : ℕ := 1  -- one mole of gas
def W_isobaric : ℝ := 10  -- work done in the isobaric process in Joules

-- The core thermodynamic relationship for an isothermal process
theorem work_done_isothermal :
  ∀ (Q_isobaric : ℝ),
  Q_isobaric = (W_isobaric : ℝ) → W_isothermal = Q_isobaric → W_isothermal = 25 :=
begin
  intros Q_isobaric h1 h2,
  rw h1 at h2,
  exact h2,
end

end work_done_isothermal_l727_727247


namespace probability_more_heads_than_tails_l727_727168

-- Define the probability of a coin landing heads
def p_head : ℚ := 3 / 5

-- Define the number of total flips
def n_flips : ℕ := 10

-- Binomial probability function
def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

-- Probability of having more heads than tails
def more_heads_than_tails_prob : ℚ :=
  (finset.range (n_flips + 1)).filter (λ k, k > n_flips / 2).sum (λ k, binomial_prob n_flips k p_head)

theorem probability_more_heads_than_tails :
  more_heads_than_tails_prob = sorry :=
by
  -- the statement without the proof
  sorry

end probability_more_heads_than_tails_l727_727168


namespace combined_probability_correct_l727_727384

-- Define the probabilities
def p_truth : ℚ := 1 / 6
def p_shark_given_truth : ℚ := 1 / 8
def p_combined : ℚ := p_truth * p_shark_given_truth

-- Prove that the combined probability is 1/48
theorem combined_probability_correct : p_combined = 1 / 48 :=
by
  -- Assuming the values for p_truth and p_shark_given_truth
  have h1 : p_truth = 1 / 6 := by rfl
  have h2 : p_shark_given_truth = 1 / 8 := by rfl
  -- Calculate the product
  rw [← h1, ← h2, mul_comm]
  sorry

end combined_probability_correct_l727_727384


namespace line_intersects_curve_equal_segments_l727_727122

theorem line_intersects_curve_equal_segments (k m : ℝ)
  (A B C : ℝ × ℝ)
  (hA_curve : A.2 = A.1^3 - 6 * A.1^2 + 13 * A.1 - 8)
  (hB_curve : B.2 = B.1^3 - 6 * B.1^2 + 13 * B.1 - 8)
  (hC_curve : C.2 = C.1^3 - 6 * C.1^2 + 13 * C.1 - 8)
  (h_lineA : A.2 = k * A.1 + m)
  (h_lineB : B.2 = k * B.1 + m)
  (h_lineC : C.2 = k * C.1 + m)
  (h_midpoint : 2 * B.1 = A.1 + C.1 ∧ 2 * B.2 = A.2 + C.2)
  : 2 * k + m = 2 :=
sorry

end line_intersects_curve_equal_segments_l727_727122


namespace tom_and_jerry_same_speed_l727_727391

noncomputable def speed_of_tom (y : ℝ) : ℝ :=
  y^2 - 14*y + 45

noncomputable def speed_of_jerry (y : ℝ) : ℝ :=
  (y^2 - 2*y - 35) / (y - 5)

theorem tom_and_jerry_same_speed (y : ℝ) (h₁ : y ≠ 5) (h₂ : speed_of_tom y = speed_of_jerry y) :
  speed_of_tom y = 6 :=
by
  sorry

end tom_and_jerry_same_speed_l727_727391


namespace car_speed_correct_l727_727610

noncomputable def car_speed (d v_bike t_delay : ℝ) (h1 : v_bike > 0) (h2 : t_delay > 0): ℝ := 2 * v_bike

theorem car_speed_correct:
  ∀ (d v_bike : ℝ) (t_delay : ℝ) (h1 : v_bike > 0) (h2 : t_delay > 0),
    (d / v_bike - t_delay = d / (car_speed d v_bike t_delay h1 h2)) → 
    car_speed d v_bike t_delay h1 h2 = 0.6 :=
by
  intros
  -- The proof would go here
  sorry

end car_speed_correct_l727_727610


namespace total_snakes_at_least_three_l727_727339

theorem total_snakes_at_least_three 
  (total_people : ℕ) (only_dogs : ℕ) (only_cats : ℕ) (cats_and_dogs : ℕ) (cats_dogs_snakes : ℕ) :
  total_people = 89 →
  only_dogs = 15 →
  only_cats = 10 →
  cats_and_dogs = 5 →
  cats_dogs_snakes = 3 →
  ∃ (snakes : ℕ), snakes ≥ 3 :=
by {
  intros ht_t hp_od hp_oc hp_cd hp_cds,
  use cats_dogs_snakes,
  linarith [hp_cds],
  sorry
}

end total_snakes_at_least_three_l727_727339


namespace unique_complex_roots_l727_727406

def equation (k x : ℂ) : Prop :=
  x / (x + 3) + x / (x + 4) = k * x

theorem unique_complex_roots (k : ℂ) : (k = 2 * Complex.I ∨ k = -2 * Complex.I) → 
  ∃ x₁ x₂ : ℂ, x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ x₁ ≠ x₂ ∧ equation k x₁ ∧ equation k x₂ :=
begin
  sorry
end

end unique_complex_roots_l727_727406


namespace original_chocolates_l727_727331

theorem original_chocolates (N : ℕ) 
  (H_nuts : N / 2)
  (H_eaten_nuts : 0.8 * (N / 2) = 0.4 * N)
  (H_without_nuts : N / 2)
  (H_eaten_without : 0.5 * (N / 2) = 0.25 * N)
  (H_left : N - (0.4 * N + 0.25 * N) = 28) : 
  N = 80 :=
sorry

end original_chocolates_l727_727331


namespace conspiracy_exists_l727_727527

-- Basic definitions and conditions given in the problem.
def num_citizens (n : ℕ) : Prop := n > 3
def pair_conspiracy (n : ℕ) : ℕ := n * (n - 1) / 2

theorem conspiracy_exists (n : ℕ) (hn : n > 3) :
  ∃ c : ℕ, c * c ≤ n ∧ sqrt n ≤ c := by
  sorry

end conspiracy_exists_l727_727527


namespace original_number_of_chairs_l727_727169

variable (x y n : ℕ)

-- Conditions as given in the problem:
-- 1. If all the chairs are exchanged for an equivalent number of tables, the difference in monetary value is 320 yuan.
def condition_1 := n * y = n * x - 320 

-- 2. If no extra money is provided, the number of tables acquired would be 5 less.
def condition_2 := n * y = (n - 5) * x

-- 3. The price of 3 tables is 48 yuan less than the price of 5 chairs.
def condition_3 := 3 * x = 5 * y + 48

-- The final theorem we want to prove:
theorem original_number_of_chairs (H1 : condition_1) (H2 : condition_2) (H3 : condition_3) : n = 20 := by
  sorry

end original_number_of_chairs_l727_727169


namespace simplify_expression_l727_727043

theorem simplify_expression :
  (20^4 + 625) * (40^4 + 625) * (60^4 + 625) * (80^4 + 625) /
  (10^4 + 625) * (30^4 + 625) * (50^4 + 625) * (70^4 + 625) = 7 := 
sorry

end simplify_expression_l727_727043


namespace worker_A_left_time_l727_727645

noncomputable def work_rate (n : ℕ) : ℚ := 1 / n

theorem worker_A_left_time :
  let A_work_rate := work_rate 6,
      B_work_rate := work_rate 8,
      C_work_rate := work_rate 10,
      total_hours := 4
  in
  (8 * 60 + 36) = 
  let work_done_by_B := B_work_rate * total_hours,
      work_done_by_C := C_work_rate * total_hours,
      total_work_done_by_B_and_C := work_done_by_B + work_done_by_C,
      total_work_needed := 1,
      work_done_by_A := total_work_needed - total_work_done_by_B_and_C,
      A_hours_worked := work_done_by_A / A_work_rate,
      A_minutes_worked := A_hours_worked * 60
  in
  8 * 60 + nat.floor A_minutes_worked := 516 := -- minutes elapsed since 8 AM
  by sorry

end worker_A_left_time_l727_727645


namespace num_bribe_takers_l727_727929

-- Definitions according to the conditions
def num_members : ℕ := 20
def has_honest_member : Prop := ∃ h : ℕ, h ∈ fin num_members -- There is at least one honest member
def pairwise_bribe_takers (H : fin num_members → Prop) : Prop :=
  ∀ (x y : fin num_members), x ≠ y → (H x ∨ H y) -- Among any two members, at least one is a bribe-taker

-- The main statement to prove
theorem num_bribe_takers (H : fin num_members → Prop) (hH : has_honest_member) (hb : pairwise_bribe_takers H) :
  ∑ i in finset.univ, if (H i) then 1 else 0 = 19 :=
sorry

end num_bribe_takers_l727_727929


namespace guise_hot_dogs_by_wednesday_l727_727146

theorem guise_hot_dogs_by_wednesday :
  let monday_dogs := 10 in
  let tuesday_dogs := monday_dogs + 2 in
  let wednesday_dogs := tuesday_dogs + 2 in
  (monday_dogs + tuesday_dogs + wednesday_dogs) = 36 :=
by
  sorry

end guise_hot_dogs_by_wednesday_l727_727146


namespace prove_k_value_range_x1_x2_squares_l727_727478

noncomputable def quadratic_eq_has_distinct_real_roots (a b c : ℝ) : Prop :=
  (b^2 - 4 * a * c > 0)

theorem prove_k_value :
  ∀ (k : ℝ), (quadratic_eq_has_distinct_real_roots 1 (-2 * (k + 1)) (k^2 + 3)) →
  (∀ x1 x2 : ℝ, x1 + x2 = 2 * (k + 1) → x1 * x2 = k^2 + 3 → 1 / x1 + 1 / x2 = 6 / 7) →
  k = 2 :=
by
  intros k h_distinct h_vieta
  cases h_vieta with x1 x2
  have h_eq : (3 : ℝ) * k ^ 2 - 7 * k + 2 = 0
  sorry

theorem range_x1_x2_squares :
  ∀ (k : ℝ), k > 1 →
  (∀ x1 x2 : ℝ, x1 + x2 = 2 * (k + 1) → x1 * x2 = k^2 + 3
    → x1^2 + x2^2 = 2 * (k + 2)^2 - 10) →
  (x1 x2 : ℝ, x1^2 + x2^2 > 8) :=
by
  intros k h_k_gt1 h_vieta
  cases h_vieta with x1 x2
  have h_expr : x1^2 + x2^2 = 2 * (k + 2)^2 - 10
  sorry

end prove_k_value_range_x1_x2_squares_l727_727478


namespace range_of_a_for_circle_l727_727627
open Real

theorem range_of_a_for_circle (a : ℝ) : 
  (∀ x y: ℝ, (x^2 + y^2 + a*x - a*y + 2 = 0) → 
  (a > 2 ∨ a < -2)) :=
begin
  sorry
end

end range_of_a_for_circle_l727_727627


namespace minimum_value_of_f_l727_727179

-- Define the function
def f (a b x : ℝ) := x^2 + (a + 2) * x + b

-- Condition that ensures the graph is symmetric about x = 1
def symmetric_about_x1 (a : ℝ) : Prop := a + 2 = -2

-- Minimum value of the function f(x) in terms of the constant c
theorem minimum_value_of_f (a b : ℝ) (h : symmetric_about_x1 a) : ∃ c : ℝ, ∀ x : ℝ, f a b x ≥ c :=
by sorry

end minimum_value_of_f_l727_727179


namespace length_of_segment_XY_l727_727657

theorem length_of_segment_XY (PQ QR YZ XY : ℝ)
  (h_similar : ∀ P Q R X Y Z : ℝ, similar PQR XYZ)
  (h_PQ : PQ = 15)
  (h_QR : QR = 25)
  (h_YZ : YZ = 30) : XY = 18 :=
by
  sorry

end length_of_segment_XY_l727_727657


namespace find_b_minus_c_l727_727295

variable (a b c: ℤ)

theorem find_b_minus_c (h1: a - b - c = 1) (h2: a - (b - c) = 13) (h3: (b - c) - a = -9) : b - c = 1 :=
by {
  sorry
}

end find_b_minus_c_l727_727295


namespace ponchik_ate_62_cubes_l727_727450

def has_odd_neighbors (neighbors : ℕ) : Prop := neighbors % 2 = 1

def cube_positions : List (ℕ × ℕ × ℕ) := 
  List.product (List.finRange 5) (List.product (List.finRange 5) (List.finRange 5))

def count_odd_neighbor_cubes : ℕ :=
  cube_positions.foldr (λ pos acc, 
    let ⟨x, (y, z)⟩ := pos in
    let neighbors := 
      (if x > 0 then 1 else 0) + (if x < 4 then 1 else 0) + 
      (if y > 0 then 1 else 0) + (if y < 4 then 1 else 0) +
      (if z > 0 then 1 else 0) + (if z < 4 then 1 else 0) 
    in if has_odd_neighbors neighbors then acc + 1 else acc) 0

theorem ponchik_ate_62_cubes : count_odd_neighbor_cubes = 62 := by 
  sorry

end ponchik_ate_62_cubes_l727_727450


namespace new_galleries_receive_two_pictures_l727_727374

theorem new_galleries_receive_two_pictures :
  ∀ (total_pencils : ℕ) (orig_gallery_pictures : ℕ) (new_galleries : ℕ) 
    (pencils_per_picture : ℕ) (signature_pencils_per_gallery : ℕ),
    (total_pencils = 88) →
    (orig_gallery_pictures = 9) →
    (new_galleries = 5) →
    (pencils_per_picture = 4) →
    (signature_pencils_per_gallery = 2) →
    let orig_gallery_signature_pencils := signature_pencils_per_gallery in
    let total_galleries := 1 + new_galleries in
    let total_signature_pencils := total_galleries * signature_pencils_per_gallery in
    let total_drawing_pencils := total_pencils - total_signature_pencils in
    let orig_gallery_drawing_pencils := orig_gallery_pictures * pencils_per_picture in
    let new_galleries_drawing_pencils := total_drawing_pencils - orig_gallery_drawing_pencils in
    let total_new_gallery_pictures := new_galleries_drawing_pencils / pencils_per_picture in
    let pictures_per_new_gallery := total_new_gallery_pictures / new_galleries in
    pictures_per_new_gallery = 2 :=
begin
  intros total_pencils orig_gallery_pictures new_galleries pencils_per_picture signature_pencils_per_gallery
          H_total_pencils H_orig_gallery_pictures H_new_galleries H_pencils_per_picture H_signature_pencils_per_gallery,

  let orig_gallery_signature_pencils := signature_pencils_per_gallery,
  let total_galleries := 1 + new_galleries,
  let total_signature_pencils := total_galleries * signature_pencils_per_gallery,
  let total_drawing_pencils := total_pencils - total_signature_pencils,
  let orig_gallery_drawing_pencils := orig_gallery_pictures * pencils_per_picture,
  let new_galleries_drawing_pencils := total_drawing_pencils - orig_gallery_drawing_pencils,
  let total_new_gallery_pictures := new_galleries_drawing_pencils / pencils_per_picture,
  let pictures_per_new_gallery := total_new_gallery_pictures / new_galleries,

  exact sorry
end

end new_galleries_receive_two_pictures_l727_727374


namespace solve_A_B_classification_l727_727731

-- Definition of key terms
def monkey : Type := sorry -- define the concept of a monkey
def human : Type := sorry -- define the concept of a human
def knight (x : human) : Prop := sorry -- A knight is a truth-telling human
def liar (x : human) : Prop := sorry -- A liar is a lying human

variable (A B : human)
variable (HA : ¬(∃ x : human, x = A ∨ x = B ∧ (¬(x = A ∧ x = B) → x = monkey)))
variable (HB : ∃ x : human, x = A ∨ x = B ∧ (¬(x = A ∧ x = B) → ¬knight x))

-- Theorem stating the problem
theorem solve_A_B_classification : (liar A ∧ ¬ knight A) ∧ (knight B ∧ ¬ liar B) := 
by sorry

end solve_A_B_classification_l727_727731


namespace percent_germinated_is_31_l727_727440

-- Define given conditions
def seeds_first_plot : ℕ := 300
def seeds_second_plot : ℕ := 200
def germination_rate_first_plot : ℝ := 0.25
def germination_rate_second_plot : ℝ := 0.40

-- Calculate the number of germinated seeds in each plot
def germinated_first_plot : ℝ := germination_rate_first_plot * seeds_first_plot
def germinated_second_plot : ℝ := germination_rate_second_plot * seeds_second_plot

-- Calculate total number of seeds and total number of germinated seeds
def total_seeds : ℕ := seeds_first_plot + seeds_second_plot
def total_germinated : ℝ := germinated_first_plot + germinated_second_plot

-- Prove the percentage of the total number of seeds that germinated
theorem percent_germinated_is_31 :
  ((total_germinated / total_seeds) * 100) = 31 := 
by
  sorry

end percent_germinated_is_31_l727_727440


namespace decimal_expansion_2023rd_digit_l727_727415

theorem decimal_expansion_2023rd_digit 
  (x : ℚ) 
  (hx : x = 7 / 26) 
  (decimal_expansion : ℕ → ℕ)
  (hdecimal : ∀ n : ℕ, decimal_expansion n = if n % 12 = 0 
                        then 2 
                        else if n % 12 = 1 
                          then 7 
                          else if n % 12 = 2 
                            then 9 
                            else if n % 12 = 3 
                              then 2 
                              else if n % 12 = 4 
                                then 3 
                                else if n % 12 = 5 
                                  then 0 
                                  else if n % 12 = 6 
                                    then 7 
                                    else if n % 12 = 7 
                                      then 6 
                                      else if n % 12 = 8 
                                        then 9 
                                        else if n % 12 = 9 
                                          then 2 
                                          else if n % 12 = 10 
                                            then 3 
                                            else 0) :
  decimal_expansion 2023 = 0 :=
sorry

end decimal_expansion_2023rd_digit_l727_727415


namespace problem_statement_l727_727885

theorem problem_statement (g : ℝ → ℝ) (m k : ℝ) (h₀ : ∀ x, g x = 5 * x - 3)
  (h₁ : 0 < k) (h₂ : 0 < m)
  (h₃ : ∀ x, |g x - 2| < k ↔ |x - 1| < m) : m ≤ k / 5 :=
sorry

end problem_statement_l727_727885


namespace evaluate_expression_l727_727394

theorem evaluate_expression : 
  | -3 | + (Real.pi + 1)^0 - (1 / 3)^(-1) = 1 := 
by
  sorry

end evaluate_expression_l727_727394


namespace seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths_l727_727020

theorem seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths : 
  (7 : ℚ) * (9 + (2 / 5)) = 65 + (4 / 5) :=
by
  sorry

end seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths_l727_727020


namespace intersection_complement_eq_empty_l727_727174

open Set

variable {α : Type*} (M N U: Set α)

theorem intersection_complement_eq_empty (h : M ⊆ N) : M ∩ (compl N) = ∅ :=
sorry

end intersection_complement_eq_empty_l727_727174


namespace sin_double_angle_l727_727460

theorem sin_double_angle (α : ℝ) (h : sin (α - π / 4) = 3 / 5) : sin (2 * α) = 7 / 25 := 
by 
  sorry

end sin_double_angle_l727_727460


namespace zero_of_f_zero_of_g_zero_of_h_zero_inequalities_l727_727487

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := Real.exp 1

theorem zero_of_f (x : ℝ) : (Real.exp x + x = 0 ↔ x = a) := sorry
theorem zero_of_g (x : ℝ) : (Real.log x + x = 0 ↔ x = b) := sorry
theorem zero_of_h (x : ℝ) : (Real.log x = 1 ↔ x = c) := by
  simp only [c, Real.exp_log]
  exact eq_self_iff_true _

theorem zero_inequalities : a < b ∧ b < c := sorry

end zero_of_f_zero_of_g_zero_of_h_zero_inequalities_l727_727487


namespace max_b_integer_l727_727665

theorem max_b_integer (b : ℤ) : (∀ x : ℝ, x^2 + (b : ℝ) * x + 20 ≠ -10) → b ≤ 10 :=
by
  sorry

end max_b_integer_l727_727665


namespace cubes_with_odd_neighbors_in_5x5x5_l727_727448

theorem cubes_with_odd_neighbors_in_5x5x5 (unit_cubes : Fin 125 → ℕ) 
  (neighbors : ∀ (i : Fin 125), Fin 125 → Prop) : ∃ n, n = 62 := 
by
  sorry

end cubes_with_odd_neighbors_in_5x5x5_l727_727448


namespace not_equilateral_if_perimeters_related_l727_727926

variables {α β γ : ℝ}

-- Definitions based on conditions given
def semiperimeter (α β γ : ℝ) : ℝ :=
  real.sin α + real.sin β + real.sin γ

def pedal_semiperimeter (α β γ : ℝ) : ℝ :=
  abs (real.sin (2 * α)) + abs (real.sin (2 * β)) + abs (real.sin (2 * γ))

-- Statement of the problem
theorem not_equilateral_if_perimeters_related (α β γ : ℝ) (h1 : semiperimeter α β γ = 2 * pedal_semiperimeter α β γ) : α ≠ β ∨ β ≠ γ ∨ γ ≠ α :=
begin
  sorry
end

end not_equilateral_if_perimeters_related_l727_727926


namespace elon_teslas_l727_727061

theorem elon_teslas (chris_teslas : ℕ) (chris_has_6 : chris_teslas = 6)
  (sam_teslas : ℕ) (sam_calc : sam_teslas = chris_teslas / 2)
  (elon_teslas : ℕ) (elon_calc : elon_teslas = sam_teslas + 10) :
  elon_teslas = 13 :=
by
  rw [chris_has_6, sam_calc, elon_calc]
  simp
  sorry

end elon_teslas_l727_727061


namespace inverse_h_l727_727614

-- Define the functions f, g, and h as given in the conditions
def f (x : ℝ) := 4 * x - 3
def g (x : ℝ) := 3 * x + 2
def h (x : ℝ) := f (g x)

-- State the problem of proving the inverse of h
theorem inverse_h : ∀ x, h⁻¹ (x : ℝ) = (x - 5) / 12 :=
sorry

end inverse_h_l727_727614


namespace transformed_stddev_l727_727869

theorem transformed_stddev (x : List ℝ) (s : ℝ) (hs : s = 8.5) :
  let y := List.map (λ xi, 3 * xi + 5) x
  in StdDev y = 25.5 := sorry

end transformed_stddev_l727_727869


namespace part1_part2_l727_727234
open Real

noncomputable def f (x : ℝ) (m : ℝ) := x^2 - m * log x
noncomputable def h (x : ℝ) (a : ℝ) := x^2 - x + a
noncomputable def k (x : ℝ) (a : ℝ) := x - 2 * log x - a

theorem part1 (x : ℝ) (m : ℝ) (h_pos_x : 1 < x) : 
  (f x m) - (h x 0) ≥ 0 → m ≤ exp 1 :=
sorry

theorem part2 (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x < 2 → k x a < 0) ∧ 
  (k 2 a < 0) ∧ 
  (∀ x, 2 < x ∧ x ≤ 3 → k x a > 0) →
  2 - 2 * log 2 < a ∧ a ≤ 3 - 2 * log 3 :=
sorry

end part1_part2_l727_727234


namespace plane_splits_into_four_regions_l727_727759

def line1 (x : ℝ) : ℝ := 3 * x
def line2 (x : ℝ) : ℝ := (1/3) * x

theorem plane_splits_into_four_regions :
  let regions := set.univ \ (set.range line1 ∪ set.range line2)
  ∃ (regions : set (set ℝ × ℝ)), regions.finite ∧ regions.card = 4 :=
sorry

end plane_splits_into_four_regions_l727_727759


namespace greatest_k_dividing_n_l727_727362

theorem greatest_k_dividing_n (n : ℕ) 
  (h1 : Nat.totient n = 72) 
  (h2 : Nat.totient (3 * n) = 96) : ∃ k : ℕ, 3^k ∣ n ∧ ∀ j : ℕ, 3^j ∣ n → j ≤ 2 := 
by {
  sorry
}

end greatest_k_dividing_n_l727_727362


namespace tree_space_l727_727899

theorem tree_space
  (spaces_between : ℕ = 7)
  (distance_between : ℕ = 20)
  (num_trees : ℕ = 8)
  (total_length : ℕ = 148) :
  ∃ (x : ℕ), 8 * x + 7 * 20 = 148 ∧ x = 1 :=
by
  use 1
  split
  rfl
  sorry

end tree_space_l727_727899


namespace sandwiches_given_to_brother_l727_727604

/-- Ruth prepared some sandwiches. She ate 1 sandwich and gave some sandwiches to her brother. Her first cousin arrived and ate 2 sandwiches. Then her two other cousins arrived and ate 1 sandwich each. There were 3 sandwiches left. Ruth prepared 10 sandwiches. How many sandwiches did Ruth give to her brother? -/
theorem sandwiches_given_to_brother
  (prepared : ℕ) (ruth_ate : ℕ) (first_cousin_ate : ℕ) (two_other_cousins_ate : ℕ) (left : ℕ)
  (h_prepared : prepared = 10) (h_ruth_ate : ruth_ate = 1) 
  (h_first_cousin_ate : first_cousin_ate = 2) (h_two_other_cousins_ate : two_other_cousins_ate = 2)
  (h_left : left = 3) :
  let total_eaten := ruth_ate + first_cousin_ate + two_other_cousins_ate in
  let given_away := prepared - left in
  given_away - total_eaten = 2 :=
by
  sorry

end sandwiches_given_to_brother_l727_727604


namespace intersection_sets_l727_727874

noncomputable section

open Set

variable {X : Type}

def A : Set ℝ := {x : ℝ | (x - 3) * (x + 1) ≤ 0}
def B : Set ℝ := {x : ℝ | 2 * x > 2}

theorem intersection_sets :
  A ∩ B = {x : ℝ | 1 < x ∧ x ≤ 3} := by
sorry

end intersection_sets_l727_727874


namespace C_duty_days_l727_727730

-- Define the conditions
def totalDays := 12
def sumOfDates := (List.range (totalDays + 1)).sum  -- Sum of dates from 1 to 12
def sumPerPerson := sumOfDates / 3

axiom A_on_duty : List Nat := [1, 3, 10, 12]  -- Dates A is on duty
axiom B_on_duty : List (List Nat) := [[8, 9, 2, 7], [8, 9, 4, 5]]  -- Two possibilities for B's duty days
axiom C_says (C : List Nat) : C.sum = sumPerPerson  -- C's duty days sum condition

-- The theorem to be proved
theorem C_duty_days : ∃ C : List Nat, C.sum = sumPerPerson ∧ C = [6, 11] := by
  sorry

end C_duty_days_l727_727730


namespace distance_of_z101_l727_727051

noncomputable theory

-- Define the sequence 
def seq : ℕ → ℂ
| 1 := 0
| (n + 1) := (seq n) ^ 2 - complex.I

-- Define the distance function
def distance_from_origin (z : ℂ) : ℝ :=
  complex.abs z

-- Theorem to prove
theorem distance_of_z101 : distance_from_origin (seq 101) = real.sqrt 2 :=
by sorry

end distance_of_z101_l727_727051


namespace part_a_part_b_l727_727411

noncomputable def triangle_exists (h1 h2 h3 : ℕ) : Prop :=
  ∃ a b c, 2 * a = h1 * (b + c) ∧ 2 * b = h2 * (a + c) ∧ 2 * c = h3 * (a + b)

theorem part_a : ¬ triangle_exists 2 3 6 :=
sorry

theorem part_b : triangle_exists 2 3 5 :=
sorry

end part_a_part_b_l727_727411


namespace max_stones_even_12x12_l727_727519

theorem max_stones_even_12x12
  (grid : Matrix (Fin 12) (Fin 12) Bool) -- grid[i][j] indicates if there's a stone (true) or not (false)
  (even_rows : ∀ i : Fin 12, (∑ j : Fin 12, cond (grid i j)) % 2 = 0)
  (even_cols : ∀ j : Fin 12, (∑ i : Fin 12, cond (grid i j)) % 2 = 0)
  (even_diag1 : ∀ k in range (12 + 12 - 1), 
                  (∑ i in (range 12), if i <= k ∧ (k - i) < 12 then grid i (k - i) else 0) % 2 = 0)
  (even_diag2 : ∀ k in range (12 + 12 - 1), 
                  (∑ i in (range 12), if i <= k ∧ (k - i) < 12 then grid (11 - i) (k - i) else 0) % 2 = 0) :
  ∃ stones, stones ≤ 120 :=
begin
  sorry
end

end max_stones_even_12x12_l727_727519


namespace conjugate_complex_quadrant_l727_727176

theorem conjugate_complex_quadrant :
  let z := (-2 + 3 * Complex.i) / Complex.i in
  let conj_z : Complex := Complex.conj z in
  (0 < conj_z.re ∧ conj_z.im < 0) :=
by
  let z := (-2 + 3 * Complex.i) / Complex.i
  have h : Complex.conj z = 3 - 2 * Complex.i := sorry
  have hz_real : 3 > 0 := by sorry
  have hz_imag : -2 < 0 := by sorry
  exact ⟨hz_real, hz_imag⟩

end conjugate_complex_quadrant_l727_727176


namespace probability_of_b_winning_l727_727613

theorem probability_of_b_winning (P_A_w P_D : ℝ) (h₁ : P_A_w = 0.2) (h₂ : P_D = 0.5) 
  (mutually_exclusive : P_A_w + P_D + P_B_w = 1) : P_B_w = 0.3 :=
by
  -- Using the given conditions
  have h₃ : P_B_w = 1 - (P_A_w + P_D), from sorry,
  rw [h₁, h₂] at h₃,
  norm_num at h₃,
  exact h₃

end probability_of_b_winning_l727_727613


namespace decimal_to_binary_123_l727_727769

theorem decimal_to_binary_123 : 
  (let n := 123 in let method (n : ℕ) : List ℕ := 
    if n = 0 then []
    else (n % 2) :: method (n / 2)
  in method n.reverse) = [1, 1, 1, 1, 0, 1, 1] :=
by
  sorry

end decimal_to_binary_123_l727_727769


namespace find_f_2018_neg_2019_l727_727120

-- Definitions of the given conditions
def f (x : ℝ) : ℝ :=
  if x ∈ set.Icc 0 1 then real.exp x - 1 else f (mod x 2.0)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

theorem find_f_2018_neg_2019 (e : ℝ) :
  (∀ x : ℝ, f (x + 2) = f x) ∧
  (is_even_function f) ∧
  (∀ x : ℝ, x ∈ set.Icc 0 1 → f x = real.exp x - 1) →
  f 2018 + f (-2019) = e - 1 :=
by
sorresponding
:path
-- sorry to skip the proof
requires:

end find_f_2018_neg_2019_l727_727120


namespace solve_for_s_l727_727609

theorem solve_for_s {x : ℝ} (h : 4 * x^2 - 8 * x - 320 = 0) : ∃ s, s = 81 :=
by 
  -- Introduce the conditions and the steps
  sorry

end solve_for_s_l727_727609


namespace circle_radius_l727_727966

theorem circle_radius : 
  ∀ (x y : ℝ), x^2 + y^2 + 12 = 10 * x - 6 * y → ∃ r : ℝ, r = Real.sqrt 22 :=
by
  intros x y h
  -- Additional steps to complete the proof will be added here
  sorry

end circle_radius_l727_727966


namespace tangent_line_equation_at_point_l727_727071

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - 1) - x

theorem tangent_line_equation_at_point :
  ∃ a b c : ℝ, (∀ x y : ℝ, a * x + b * y + c = 0 ↔ (x = 1 → y = -1 → f x = y)) ∧ (a * 1 + b * (-1) + c = 0) :=
by
  sorry

end tangent_line_equation_at_point_l727_727071


namespace selling_price_l727_727365

theorem selling_price (cost_price : ℕ) (profit_percent : ℕ) (selling_price : ℕ) : 
  cost_price = 2400 ∧ profit_percent = 6 → selling_price = 2544 := by
  sorry

end selling_price_l727_727365


namespace probability_red_run_out_first_is_21_over_40_l727_727351

def marbles_in_bag (bag : list ℕ) : Prop :=
  bag.length = 15 ∧ bag.count 0 = 3 ∧ bag.count 1 = 5 ∧ bag.count 2 = 7

noncomputable def probability_red_run_out_first (bag : list ℕ) : ℚ :=
  if marbles_in_bag bag then
    let total_order_combinations := (15.factorial / ((3.factorial) * (5.factorial) * (7.factorial)) : ℚ) in
    let successful_outcomes := (nat.choose 14 4) * (nat.choose 10 6) + (nat.choose 14 6) * (nat.choose 8 4) in
    successful_outcomes / total_order_combinations
  else 0

theorem probability_red_run_out_first_is_21_over_40 (bag : list ℕ) (h : marbles_in_bag bag) :
  probability_red_run_out_first bag = 21 / 40 :=
by sorry

end probability_red_run_out_first_is_21_over_40_l727_727351


namespace pounds_of_fish_to_ship_l727_727961

theorem pounds_of_fish_to_ship (crates_weight : ℕ) (cost_per_crate : ℝ) (total_cost : ℝ) :
  crates_weight = 30 → cost_per_crate = 1.5 → total_cost = 27 → 
  (total_cost / cost_per_crate) * crates_weight = 540 :=
by
  intros h1 h2 h3
  sorry

end pounds_of_fish_to_ship_l727_727961


namespace linear_function_evaluation_l727_727941

theorem linear_function_evaluation (g : ℝ → ℝ) (h_linear : ∀ x y, g(y) - g(x) = (y - x) * (g 1 - g 0)) 
  (h_condition : g 8 - g 3 = 20) : g 15 - g 3 = 48 :=
by
  sorry

end linear_function_evaluation_l727_727941


namespace distinct_factor_order_identical_factor_order_l727_727530

theorem distinct_factor_order :
  (∃ (a b c : ℕ), a * b * c = 1000000 ∧ function.injective (λ m, m)) → (factors_count distinct_order 1000000) = 784 :=
by sorry

theorem identical_factor_order :
  (∃ (a b c : ℕ), a * b * c = 1000000 ∧ ¬ function.injective (λ m, m)) → (factors_count identical_order 1000000) = 139 :=
by sorry

end distinct_factor_order_identical_factor_order_l727_727530


namespace sum_T3n_eq_l727_727138

variable (a : ℕ → ℝ) (S_n : ℕ → ℝ) (b : ℕ → ℝ) (T_n : ℕ → ℝ)

-- Conditions
axiom S_n_def : ∀ n : ℕ, n > 0 → S_n n = 2 * a 2 * n
axiom b_n_def : ∀ n : ℕ, b n = Real.logBase (1/2) (a n + 2)
axiom T_n_def : ∀ n : ℕ, T_n n = ∑ i in Finset.range (n + 1), b i

-- Proof statement
theorem sum_T3n_eq : ∀ n : ℕ, n > 0 → (∑ k in Finset.range n, 1 / T_n (3 * (k + 1))) = -2 * n / (n + 1) :=
by
  intro n hn
  sorry

end sum_T3n_eq_l727_727138


namespace n_divisible_by_40_l727_727201

theorem n_divisible_by_40 {n : ℕ} (h_pos : 0 < n)
  (h1 : ∃ k1 : ℕ, 2 * n + 1 = k1 * k1)
  (h2 : ∃ k2 : ℕ, 3 * n + 1 = k2 * k2) :
  ∃ k : ℕ, n = 40 * k := 
sorry

end n_divisible_by_40_l727_727201


namespace circle_circumference_l727_727635

noncomputable def square_perimeter : ℝ := 28

noncomputable def side_of_square : ℝ := square_perimeter / 4

noncomputable def radius_of_circle : ℝ := side_of_square

noncomputable def circumference_of_circle : ℝ := 2 * Real.pi * radius_of_circle

theorem circle_circumference : circumference_of_circle ≈ 43.98 := 
begin
  -- The exact proof logic goes here.
  sorry
end

end circle_circumference_l727_727635


namespace modulus_of_z_l727_727469

noncomputable def z_modulus (z : ℂ) : ℝ :=
complex.abs z

theorem modulus_of_z (z : ℂ) (i : ℂ) (h_i : i * i = -1) (h : z * (i + 1) = i) : z_modulus z = real.sqrt 2 / 2 :=
by
  -- Properties of imaginary unit and given equation
  have h_modulus : z_modulus z = complex.abs z := rfl
  sorry

end modulus_of_z_l727_727469


namespace number_of_paintings_per_new_gallery_l727_727377

-- Define all the conditions as variables/constants
def pictures_original : Nat := 9
def new_galleries : Nat := 5
def pencils_per_picture : Nat := 4
def pencils_per_exhibition : Nat := 2
def total_pencils : Nat := 88

-- Define the proof problem in Lean
theorem number_of_paintings_per_new_gallery (pictures_original new_galleries pencils_per_picture pencils_per_exhibition total_pencils : Nat) :
(pictures_original = 9) → (new_galleries = 5) → (pencils_per_picture = 4) → (pencils_per_exhibition = 2) → (total_pencils = 88) → 
∃ (pictures_per_gallery : Nat), pictures_per_gallery = 2 :=
by
  intros
  sorry

end number_of_paintings_per_new_gallery_l727_727377


namespace valid_paths_count_l727_727756

-- Define the grid and the prohibited segments
def grid (height width : ℕ) : Type :=
  { p : ℕ × ℕ // p.1 ≤ height ∧ p.2 ≤ width }

def isForbiddenSegment1 (p : ℕ × ℕ) : Prop :=
  p.2 = 3 ∧ 1 ≤ p.1 ∧ p.1 ≤ 3

def isForbiddenSegment2 (p : ℕ × ℕ) : Prop :=
  p.2 = 4 ∧ 2 ≤ p.1 ∧ p.1 ≤ 5

-- Statement of the problem
theorem valid_paths_count : 
  let height := 5 
  let width  := 8 in
  let A := (0, 0) 
  let B := (height, width) 
  count_valid_paths A B height width isForbiddenSegment1 isForbiddenSegment2 = 838 := sorry

end valid_paths_count_l727_727756


namespace work_done_isothermal_l727_727248

-- Definitions based on the conditions
def n : ℕ := 1  -- one mole of gas
def W_isobaric : ℝ := 10  -- work done in the isobaric process in Joules

-- The core thermodynamic relationship for an isothermal process
theorem work_done_isothermal :
  ∀ (Q_isobaric : ℝ),
  Q_isobaric = (W_isobaric : ℝ) → W_isothermal = Q_isobaric → W_isothermal = 25 :=
begin
  intros Q_isobaric h1 h2,
  rw h1 at h2,
  exact h2,
end

end work_done_isothermal_l727_727248


namespace maximum_value_of_k_minus_b_l727_727480

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := Real.log x - a * x + b

theorem maximum_value_of_k_minus_b (b : ℝ) (k : ℝ) (x : ℝ) 
  (h₀ : 0 ≤ b ∧ b ≤ 2) 
  (h₁ : 1 ≤ x ∧ x ≤ Real.exp 1)
  (h₂ : ∀ x ∈ Set.Icc 1 (Real.exp 1), f x 1 b ≥ (k * x - x * Real.log x - 1)) :
  k - b ≤ 0 :=
sorry

end maximum_value_of_k_minus_b_l727_727480


namespace lorraine_initial_brownies_l727_727578

theorem lorraine_initial_brownies (B : ℝ) 
(h1: (0.375 * B - 1 = 5)) : B = 16 := 
sorry

end lorraine_initial_brownies_l727_727578


namespace sequence_arithmetic_progression_l727_727558

theorem sequence_arithmetic_progression (b : ℕ → ℕ) (b1_eq : b 1 = 1) (recurrence : ∀ n, b (n + 2) = b (n + 1) * b n + 1) : b 2 = 1 ↔ 
  ∃ d : ℕ, ∀ n, b (n + 1) - b n = d :=
sorry

end sequence_arithmetic_progression_l727_727558


namespace unique_combinations_bathing_suits_l727_727352

theorem unique_combinations_bathing_suits
  (men_styles : ℕ) (men_sizes : ℕ) (men_colors : ℕ)
  (women_styles : ℕ) (women_sizes : ℕ) (women_colors : ℕ)
  (h_men_styles : men_styles = 5) (h_men_sizes : men_sizes = 3) (h_men_colors : men_colors = 4)
  (h_women_styles : women_styles = 4) (h_women_sizes : women_sizes = 4) (h_women_colors : women_colors = 5) :
  men_styles * men_sizes * men_colors + women_styles * women_sizes * women_colors = 140 :=
by
  sorry

end unique_combinations_bathing_suits_l727_727352


namespace reggie_marbles_l727_727257

/-- Given that Reggie and his friend played 9 games in total,
    Reggie lost 1 game, and they bet 10 marbles per game.
    Prove that Reggie has 70 marbles after all games. -/
theorem reggie_marbles (total_games : ℕ) (lost_games : ℕ) (marbles_per_game : ℕ) (marbles_initial : ℕ) 
  (h_total_games : total_games = 9) (h_lost_games : lost_games = 1) (h_marbles_per_game : marbles_per_game = 10) 
  (h_marbles_initial : marbles_initial = 0) : 
  marbles_initial + (total_games - lost_games) * marbles_per_game - lost_games * marbles_per_game = 70 :=
by
  -- We proved this in the solution steps, but will skip the proof here with sorry.
  sorry

end reggie_marbles_l727_727257


namespace staircase_toothpicks_l727_727741

theorem staircase_toothpicks (a : ℕ) (r : ℕ) (n : ℕ) :
  a = 9 ∧ r = 3 ∧ n = 3 + 4 
  → (a * r ^ 3 + a * r ^ 2 + a * r + a) + (a * r ^ 2 + a * r + a) + (a * r + a) + a = 351 :=
by
  sorry

end staircase_toothpicks_l727_727741


namespace triangle_from_right_triangles_l727_727334

theorem triangle_from_right_triangles :
  ∃ (a b c : ℕ) (t : Triangle), 
  right_triangle 5 12 t.1 ∧
  right_triangle 9 12 t.2 ∧
  (t.1.base + t.2.base = 14) ∧ 
  (t.1.height = t.2.height = 12) ∧ 
  (t.area = 84) :=
by
  sorry

end triangle_from_right_triangles_l727_727334


namespace number_of_pipes_l727_727750

theorem number_of_pipes (d_large d_small: ℝ) (π : ℝ) (h1: d_large = 4) (h2: d_small = 2) : 
  ((π * (d_large / 2)^2) / (π * (d_small / 2)^2) = 4) := 
by
  sorry

end number_of_pipes_l727_727750


namespace minimum_value_expression_l727_727470

theorem minimum_value_expression {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy : x * y = 1) : 
  (x / y + y) * (y / x + x) ≥ 4 :=
sorry

end minimum_value_expression_l727_727470


namespace alexandra_magazines_l727_727734

noncomputable def magazines (bought_on_friday : ℕ) (bought_on_saturday : ℕ) (times_friday : ℕ) (chewed_up : ℕ) : ℕ :=
  bought_on_friday + bought_on_saturday + times_friday * bought_on_friday - chewed_up

theorem alexandra_magazines :
  ∀ (bought_on_friday bought_on_saturday times_friday chewed_up : ℕ),
      bought_on_friday = 8 → 
      bought_on_saturday = 12 → 
      times_friday = 4 → 
      chewed_up = 4 →
      magazines bought_on_friday bought_on_saturday times_friday chewed_up = 48 :=
by
  intros
  sorry

end alexandra_magazines_l727_727734


namespace brass_weight_l727_727282

theorem brass_weight (copper zinc brass : ℝ) (h_ratio : copper / zinc = 3 / 7) (h_zinc : zinc = 70) : brass = 100 :=
by
  sorry

end brass_weight_l727_727282


namespace crayons_count_l727_727594

-- Define the initial number of crayons
def initial_crayons : ℕ := 1453

-- Define the number of crayons given away
def crayons_given_away : ℕ := 563

-- Define the number of crayons lost
def crayons_lost : ℕ := 558

-- Define the final number of crayons left
def final_crayons_left : ℕ := initial_crayons - crayons_given_away - crayons_lost

-- State that the final number of crayons left is 332
theorem crayons_count : final_crayons_left = 332 :=
by
    -- This is where the proof would go, which we're skipping with sorry
    sorry

end crayons_count_l727_727594


namespace solution_l727_727181

theorem solution (x : ℝ) (h : ¬ (x ^ 2 - 5 * x + 4 > 0)) : 1 ≤ x ∧ x ≤ 4 :=
by
  sorry

end solution_l727_727181


namespace integral_combined_integral_solution_l727_727063

noncomputable def integral_sqrt_1_minus_x_squared : ℝ := 
  ∫ x in -1..1, real.sqrt (1 - x^2)

noncomputable def integral_x_cos_x : ℝ := 
  ∫ x in -1..1, x * real.cos x

theorem integral_combined :
  ∫ x in -1..1, (real.sqrt (1 - x^2) + x * real.cos x) = 
  (integral_sqrt_1_minus_x_squared) + (integral_x_cos_x) := 
by sorry

theorem integral_solution :
  ∫ x in -1..1, (real.sqrt (1 - x^2) + x * real.cos x) = 
  real.pi / 2 :=
  by
  have h1 : integral_sqrt_1_minus_x_squared = real.pi / 2,
  { sorry }, -- Given as condition 1
  have h2 : integral_x_cos_x = 0,
  { sorry }, -- Given as condition 2
  rw [integral_combined, h1, h2],
  ring -- Simplifies (π / 2) + 0 to π / 2

end integral_combined_integral_solution_l727_727063


namespace integral_e_f_l727_727944

noncomputable def f (a b x y : ℝ) : ℝ := 
  max ((b^2 * x^2) / (a^2)) ((a^2 * y^2) / (b^2))

theorem integral_e_f (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∫ x in 0..a, ∫ y in 0..b, exp (f a b x y) = (exp (b^2) - 1) / (a * b) :=
by
  sorry

end integral_e_f_l727_727944


namespace marcus_total_baseball_cards_l727_727580

/-- Given conditions:
  1. Marcus initially has 2100 baseball cards.
  2. Marcus receives 750 baseball cards.
  3. Carter's collection of 3040, Marcus receives 12.5% of it which is 380.

  Goal: Prove Marcus ends up with 3230 baseball cards.
-/
theorem marcus_total_baseball_cards : 
  ∀ (initial_cards : ℕ) (cards_given_by_carter : ℕ) (carter_cards : ℕ) (percentage_received : ℝ),
  initial_cards = 2100 → 
  cards_given_by_carter = 750 → 
  carter_cards = 3040 → 
  percentage_received = 12.5 → 
  let total_cards := initial_cards + cards_given_by_carter + (percentage_received / 100 : ℝ) * (carter_cards : ℝ)
  in total_cards = 3230 :=
begin
  sorry
end

end marcus_total_baseball_cards_l727_727580


namespace no_six_digit_starting_with_five_12_digit_square_six_digit_starting_with_one_12_digit_square_smallest_k_for_n_digit_number_square_l727_727340

-- Part (a)
theorem no_six_digit_starting_with_five_12_digit_square : ∀ (x y : ℕ), (5 * 10^5 ≤ x) → (x < 6 * 10^5) → (10^5 ≤ y) → (y < 10^6) → ¬∃ z : ℕ, (10^11 ≤ z) ∧ (z < 10^12) ∧ x * 10^6 + y = z^2 := sorry

-- Part (b)
theorem six_digit_starting_with_one_12_digit_square : ∀ (x y : ℕ), (10^5 ≤ x) → (x < 2 * 10^5) → (10^5 ≤ y) → (y < 10^6) → ∃ z : ℕ, (10^11 ≤ z) ∧ (z < 2 * 10^11) ∧ x * 10^6 + y = z^2 := sorry

-- Part (c)
theorem smallest_k_for_n_digit_number_square : ∀ (n : ℕ), ∃ (k : ℕ), k = n + 1 ∧ ∀ (x : ℕ), (10^(n-1) ≤ x) → (x < 10^n) → ∃ y : ℕ, (10^(n + k - 1) ≤ x * 10^k + y) ∧ (x * 10^k + y) < 10^(n + k) ∧ ∃ z : ℕ, x * 10^k + y = z^2 := sorry

end no_six_digit_starting_with_five_12_digit_square_six_digit_starting_with_one_12_digit_square_smallest_k_for_n_digit_number_square_l727_727340


namespace expression_value_l727_727335

noncomputable def x1 : ℝ := Real.sqrt 1.5
noncomputable def x2 : ℝ := Real.sqrt 0.81
noncomputable def x3 : ℝ := Real.sqrt 1.44
noncomputable def x4 : ℝ := Real.sqrt 0.49

noncomputable def y1 : ℝ := x1 / x2
noncomputable def y2 : ℝ := x3 / x4
noncomputable def y : ℝ := y1 + y2

theorem expression_value : abs(y - 3.075) < 0.001 := 
by 
  -- sorry skips the proof
  sorry

end expression_value_l727_727335


namespace base_b_conversion_l727_727409

theorem base_b_conversion (b : ℝ) (h₁ : 1 * 5^2 + 3 * 5^1 + 2 * 5^0 = 42) (h₂ : 2 * b^2 + 2 * b + 1 = 42) :
  b = (-1 + Real.sqrt 83) / 2 := 
  sorry

end base_b_conversion_l727_727409


namespace min_chord_length_eq_l727_727855

-- Define the Circle C with center (1, 2) and radius 5
def isCircle (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + (y - 2) ^ 2 = 25

-- Define the Line l parameterized by m
def isLine (m x y : ℝ) : Prop :=
  (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

-- Prove that the minimal chord length intercepted by the circle occurs when the line l is 2x - y - 5 = 0
theorem min_chord_length_eq (x y : ℝ) : 
  (∀ m, isLine m x y → isCircle x y) → isLine 0 x y :=
sorry

end min_chord_length_eq_l727_727855


namespace shaded_region_perimeter_l727_727532

theorem shaded_region_perimeter (r : ℝ) (h : r = 12 / Real.pi) :
  3 * (24 / 6) = 12 := 
by
  sorry

end shaded_region_perimeter_l727_727532


namespace divisor_sum_condition_l727_727826

theorem divisor_sum_condition (σ : ℕ → ℕ) (m n : ℕ) (hσm : σ m = ∑ d in divisors m, d) (hσn : σ n = ∑ d in divisors n, d) 
(hσmn : σ (m * n) = ∑ d in divisors (m * n), d) (hm : m ≥ 2) (hn : n ≥ 2) (hmn : m ≥ n) 
(h_eq : (σ m - 1) / (m - 1) = (σ n - 1) / (n - 1) ∧ (σ m - 1) / (m - 1) = (σ (m * n) - 1) / (m * n - 1)) : 
∃ (p : ℕ) (hp : prime p) (e f : ℕ), e ≥ f ∧ m = p ^ e ∧ n = p ^ f := 
sorry

end divisor_sum_condition_l727_727826


namespace simplify_f_value_f_specific_α_value_f_third_quadrant_l727_727096

def f (α : Real) : Real :=
  (sin (π + α) * cos (2 * π - α) * tan (-α)) /
  (tan (-π - α) * cos ((3 * π / 2) + α))

theorem simplify_f (α : Real) : f(α) = -cos(α) := 
sorry

theorem value_f_specific_α : f (-31 * π / 3) = -1 / 2 := 
sorry

theorem value_f_third_quadrant (α : Real) (h1 : sin α = -1 / 5) (h2 : α > π ∧ α < 3 * π / 2) :
  f(α) = 2 * sqrt 5 / 5 := 
sorry

end simplify_f_value_f_specific_α_value_f_third_quadrant_l727_727096


namespace angle_B_l727_727418

section
  -- Define the length of sides and altitude
  variables {A B C H : Type} [LinearOrderedField A] [OrderIso B C A]
  variables (length_AB : B) (length_CH : C)
  variables (BAC : A) (H_c : C = length_AB / 2)

  -- Conditions based on the problem statement
  variables (BAC_eq_75_deg : BAC = 75)
  variables (CH_eq_half_AB : length_CH = length_AB / 2)

  theorem angle_B (h1 : CH_eq_half_AB) (h2 : BAC_eq_75_deg) : (B = 30) := 
  sorry
end

end angle_B_l727_727418


namespace standard_deviation_is_2point5_l727_727620

noncomputable def mean : ℝ := 17.5
noncomputable def given_value : ℝ := 12.5

theorem standard_deviation_is_2point5 :
  ∀ (σ : ℝ), mean - 2 * σ = given_value → σ = 2.5 := by
  sorry

end standard_deviation_is_2point5_l727_727620


namespace sqrt_720_eq_12_sqrt_5_l727_727999

theorem sqrt_720_eq_12_sqrt_5 : sqrt 720 = 12 * sqrt 5 :=
by
  sorry

end sqrt_720_eq_12_sqrt_5_l727_727999


namespace problem_l727_727553

-- Define the set T
def T := { n : ℕ | ∃ j k : ℕ, 0 ≤ j ∧ j < k ∧ k ≤ 39 ∧ n = 2^j + 2^k }

-- Probability of selecting a number divisible by 11
theorem problem (p q : ℕ) (h_rel_prime : Nat.coprime p q) (h_probability : p / q = 1 / 52) : p + q = 53 := by
  sorry

end problem_l727_727553


namespace equal_angles_l727_727242

noncomputable def midpoint (A B : ℝ^3) : ℝ^3 := (A + B) / 2

variables {A B C D O L M N: ℝ^3}

-- Conditions given in the problem
def is_circumcenter (O A B C D : ℝ^3) : Prop := sorry -- circumcenter condition needs to be defined
def are_midpoints (L M N A B C : ℝ^3) : Prop := 
  L = midpoint B C ∧ 
  M = midpoint C A ∧ 
  N = midpoint A B

def equalities (A B C D : ℝ^3) : Prop :=
  (dist A B + dist B C = dist A D + dist C D) ∧
  (dist C B + dist C A = dist B D + dist A D) ∧
  (dist C A + dist A B = dist C D + dist B D)

-- Theorem to prove
theorem equal_angles 
  (h1 : is_circumcenter O A B C D)
  (h2 : are_midpoints L M N A B C)
  (h3 : equalities A B C D) :
  ∠ L O M = 90 ∧ ∠ M O N = 90 ∧ ∠ N O L = 90 := 
sorry

end equal_angles_l727_727242


namespace remainder_of_sum_l727_727962

theorem remainder_of_sum (k j : ℤ) (a b : ℤ) (h₁ : a = 60 * k + 53) (h₂ : b = 45 * j + 17) : ((a + b) % 15) = 5 :=
by
  sorry

end remainder_of_sum_l727_727962


namespace expected_winnings_is_minus_two_l727_727712

-- Definitions for probabilities
def prob_heads : ℝ := 1 / 3
def prob_tails : ℝ := 1 / 6
def prob_edge : ℝ := 1 / 2

-- Definitions for winnings
def win_heads : ℝ := (2 + 3) / 2
def win_tails : ℝ := 4
def loss_edge : ℝ := -7

-- The expected value calculation
def expected_winnings : ℝ :=
  prob_heads * win_heads + prob_tails * win_tails + prob_edge * loss_edge

-- The theorem stating the expected winnings
theorem expected_winnings_is_minus_two : expected_winnings = -2 := by
  sorry

end expected_winnings_is_minus_two_l727_727712


namespace probability_hare_claims_not_hare_then_not_rabbit_l727_727687

noncomputable def probability_hare_given_claims : ℚ := (27 / 59)

theorem probability_hare_claims_not_hare_then_not_rabbit
  (population : ℚ) (hares : ℚ) (rabbits : ℚ)
  (belief_hare_not_hare : ℚ) (belief_hare_not_rabbit : ℚ)
  (belief_rabbit_not_hare : ℚ) (belief_rabbit_not_rabbit : ℚ) :
  population = 1 ∧ hares = 1/2 ∧ rabbits = 1/2 ∧
  belief_hare_not_hare = 1/4 ∧ belief_hare_not_rabbit = 3/4 ∧
  belief_rabbit_not_hare = 2/3 ∧ belief_rabbit_not_rabbit = 1/3 →
  (27 / 59) = probability_hare_given_claims :=
sorry

end probability_hare_claims_not_hare_then_not_rabbit_l727_727687


namespace meaning_of_a2_add_b2_ne_zero_l727_727633

theorem meaning_of_a2_add_b2_ne_zero (a b : ℝ) (h : a^2 + b^2 ≠ 0) : a ≠ 0 ∨ b ≠ 0 :=
by
  sorry

end meaning_of_a2_add_b2_ne_zero_l727_727633


namespace irreducible_complementary_fraction_l727_727927

-- Define the problem parameters
variables {a b : ℕ}

-- Define gcd condition
def gcd_condition (a b : ℕ) := Nat.gcd a b = 1

-- Define the complementary fraction
def complementary_fraction_irreducible (a b : ℕ) := (Nat.gcd (b - a) b) = 1

-- The statement of the problem
theorem irreducible_complementary_fraction :
  ∀ a b : ℕ, gcd_condition a b → complementary_fraction_irreducible a b :=
begin
  sorry
end

end irreducible_complementary_fraction_l727_727927


namespace transform_expression_to_product_l727_727880

variables (a b c d s: ℝ)

theorem transform_expression_to_product
  (h1 : d = a + b + c)
  (h2 : s = (a + b + c + d) / 2) :
    2 * (a^2 * b^2 + a^2 * c^2 + a^2 * d^2 + b^2 * c^2 + b^2 * d^2 + c^2 * d^2) -
    (a^4 + b^4 + c^4 + d^4) + 8 * a * b * c * d = 16 * (s - a) * (s - b) * (s - c) * (s - d) :=
by
  sorry

end transform_expression_to_product_l727_727880


namespace line_intersects_parabola_l727_727136

theorem line_intersects_parabola (k : ℝ) (P A B F M N : ℝ × ℝ)
  (h₀ : P = (-1, 0))
  (h₁ : F = (1, 0))
  (h₂ : ∀ P A B, P ≠ A ∧ P ≠ B)
  (h₃ : ∃ l, l.slope = k ∧ k > 0 ∧ (P.x - A.x) * (P.y - A.y) ≠ 0 ∧ (P.x - B.x) * (P.y - B.y) ≠ 0)
  (h₄ : ∀ (x : ℝ), ∃ (y1 y2 : ℝ), y1 ≠ y2 ∧ y1^2 = 4 * x ∧ y1 = k * (x + 1) ∧ y2 = k * (x + 1))
  (h₅ : ∀ AF BF FM FN, |AF| = x₁ + 1 ∧ |BF| = x₂ + 1 ∧ |FM| = x₂ + 1 ∧ |FN| = x₁ + 1)
  (h₆ : ∀ y1 y2, y1 + y2 = 4 / k ∧ y1 * y2 = 4)
  (h₇ : 18 = (|AF| / |FM|) + (|BF| / |FN|)) :
  k = real.sqrt 5 / 5 :=
sorry

end line_intersects_parabola_l727_727136


namespace seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths_l727_727026

theorem seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths : 
  (7 : ℚ) * (9 + (2 / 5)) = 65 + (4 / 5) :=
by
  sorry

end seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths_l727_727026


namespace no_solution_to_equation_l727_727166

theorem no_solution_to_equation :
  ¬ ∃ x : ℝ, x ≠ 5 ∧ (1 / (x + 5) + 1 / (x - 5) = 1 / (x - 5)) :=
by 
  sorry

end no_solution_to_equation_l727_727166


namespace min_value_of_n_l727_727127

theorem min_value_of_n :
  ∀ (h : ℝ), ∃ n : ℝ, (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → -x^2 + 2 * h * x - h ≤ n) ∧
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ -x^2 + 2 * h * x - h = n) ∧
  n = -1 / 4 := 
by
  sorry

end min_value_of_n_l727_727127


namespace count_non_negative_integers_l727_727589

theorem count_non_negative_integers: 
  (finset.filter (λ x, 0 ≤ x ∧ x < 3) (finset.Icc (-1) 2)).card = 3 :=
by
  sorry

end count_non_negative_integers_l727_727589


namespace invertible_from_c_l727_727221

-- Define the function f
def f (x : ℝ) : ℝ := (x - 3)^2 - 4

-- Define the condition for c and the statement to prove
theorem invertible_from_c (c : ℝ) (h : ∀ x1 x2 : ℝ, c ≤ x1 → c ≤ x2 → f x1 = f x2 → x1 = x2) : c = 3 :=
sorry

end invertible_from_c_l727_727221


namespace jigsaw_puzzle_pieces_l727_727579

theorem jigsaw_puzzle_pieces
  (P : ℝ)
  (h1 : ∃ P, P = 0.90 * P + 0.72 * 0.10 * P + 0.504 * 0.08 * P + 504)
  (h2 : 0.504 * P = 504) :
  P = 1000 :=
by
  sorry

end jigsaw_puzzle_pieces_l727_727579


namespace tangency_condition_l727_727877

variables (a b c : ℝ) (A : ℝ)
variable h₁ : b - c = -3
variable h₂ : a ≠ 0

theorem tangency_condition : (A = 0 ∨ A = 12) :=
by {
  -- We derive this from the conditions given
  sorry
}

end tangency_condition_l727_727877


namespace geometric_sequence_properties_and_sum_l727_727098

open BigOperators

-- Definition of a geometric sequence and its properties
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a n = a 1 * q ^ (n - 1)

-- Definition of the sum of the first n terms of a sequence
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n : ℕ, S n = ∑ i in range n, a i

-- Problem statement in Lean 4
theorem geometric_sequence_properties_and_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) :
  is_geometric_sequence a 2 ∧ a 1 = 1 ∧ a 4 = 8 ∧ sum_of_first_n_terms a S ∧ S n = 63 →
  (∀ m : ℕ, a m = 2 ^ (m - 1)) ∧ n = 6 :=
by {
  sorry
}

end geometric_sequence_properties_and_sum_l727_727098


namespace inverse_of_A_l727_727428

open Matrix

noncomputable def A : Matrix (Fin 2) (Fin 2) ℚ :=
  matrix.of ![![3, 4], ![-2, 9]]

noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  matrix.of ![![9/35, -4/35], ![2/35, 3/35]]

theorem inverse_of_A :
  A⁻¹ = A_inv := by
  sorry

end inverse_of_A_l727_727428


namespace product_of_large_integers_l727_727608

theorem product_of_large_integers :
  ∃ A B : ℤ, A > 10^2009 ∧ B > 10^2009 ∧ A * B = 3^(4^5) + 4^(5^6) :=
by
  sorry

end product_of_large_integers_l727_727608


namespace number_of_integers_satisfying_inequality_l727_727819

theorem number_of_integers_satisfying_inequality :
  ∃ (n : ℕ), (10 < n^2 ∧ n^2 < 99) ∧ (card {n : ℤ | 10 < n^2 ∧ n^2 < 99} = 12) :=
sorry

end number_of_integers_satisfying_inequality_l727_727819


namespace syllogism_l727_727875

variables (Student : Type) 
variable (An_Mengyi : Student)
variable (Class21 : Set Student)
variable (OnlyChild : Student → Prop)

-- Conditions:
axiom P1 : An_Mengyi ∈ Class21
axiom P3 : ∀ s, s ∈ Class21 → OnlyChild s

-- Statement to prove:
theorem syllogism : OnlyChild An_Mengyi :=
by
  apply P3 An_Mengyi
  exact P1

print syllogism -- This will print the theorem header and its type.

end syllogism_l727_727875


namespace k_range_condition_l727_727413

theorem k_range_condition (k : ℝ) :
    (∀ x : ℝ, x^2 - (2 * k - 6) * x + k - 3 > 0) ↔ (3 < k ∧ k < 4) :=
by
  sorry

end k_range_condition_l727_727413


namespace fifth_term_of_sequence_l727_727243

theorem fifth_term_of_sequence :
  let a_n (n : ℕ) := (-1:ℤ)^(n+1) * (n^2 + 1)
  ∃ x : ℤ, a_n 5 * x^5 = 26 * x^5 :=
by
  sorry

end fifth_term_of_sequence_l727_727243


namespace inequality_l727_727986
-- Import the necessary libraries from Mathlib

-- Define the theorem statement
theorem inequality (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := 
by
  sorry

end inequality_l727_727986


namespace detect_counterfeit_coin_l727_727640

theorem detect_counterfeit_coin :
  ∃ (n weighings : ℕ), n = 13 ∧ weighings = 3 ∧
  (∀ coins : fin n → ℤ, (∃ i, ¬coins.are_congruent ∧ weighings = 3 →
    ∃ (f : fin n → bool), 
      (∀ (c : fin n), f c = true ↔ c = i))): true := 
sorry

end detect_counterfeit_coin_l727_727640


namespace min_reciprocal_sum_l727_727106

theorem min_reciprocal_sum (m n a b : ℝ) (h1 : m = 5) (h2 : n = 5) 
  (h3 : m * a + n * b = 1) (h4 : 0 < a) (h5 : 0 < b) : 
  (1 / a + 1 / b) = 20 :=
by 
  sorry

end min_reciprocal_sum_l727_727106


namespace equal_angles_right_triangle_l727_727828

noncomputable
def right_triangle (A B C : Type) [euclidean_space A B C] (hABC : ∠ACB = 90) : Prop :=
∀ M : A, ∃ N : A, is_perpendicular MN AB -> 
∃ angle_MAN angle_MCN : ℝ, 
(angle_MAN = ∠MAN) ∧ 
(angle_MCN = ∠MCN) ∧ 
(angle_MAN = angle_MCN)

theorem equal_angles_right_triangle (A B C M N: Type) [euclidean_space A B C M N] 
(hABC: ∠ACB = 90) 
(hMN: ∃ N, is_perpendicular MN AB):
angle_MAN = angle_MCN :=
begin
  sorry
end

end equal_angles_right_triangle_l727_727828


namespace equidistant_points_l727_727924

variables {A B C D M N : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables [metric_space M] [metric_space N]
variables {P : A × B × C}

def is_foot_of_altitude_from (B A C : Type*) (D : Type*) : Prop :=
  ∃ h, line (B, D) ⊥ line (A, C) ∧ D ∈ line (A, C)

def is_perpendicular (A B N : Type*) : Prop :=
  line (A, N) ⊥ line (A, B)

def equal_length (x y : Type*) : Prop := |x| = |y|

theorem equidistant_points (ABC : Type*)
  (height_BD : is_foot_of_altitude_from B A C D)
  (AN_perp_AB : is_perpendicular A B N)
  (CM_perp_BC : is_perpendicular C B M)
  (AN_eq_DC : equal_length (distance A N) (distance D C))
  (CM_eq_AD : equal_length (distance C M) (distance A D))
  : equal_length (distance B M) (distance B N) :=
begin
  sorry,  -- Proof steps go here
end

end equidistant_points_l727_727924


namespace seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths_l727_727022

theorem seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths : 
  (7 : ℚ) * (9 + (2 / 5)) = 65 + (4 / 5) :=
by
  sorry

end seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths_l727_727022


namespace mul_mixed_number_l727_727006

theorem mul_mixed_number (a b : ℝ) (c : ℚ) (h : c = 9 + 2/5) : 
  7 * c = (65 + 4/5 : ℝ) :=
by 
  simp [h, mul_add, mul_div_cancel_left, rat.cast_add, rat.cast_mul, rat.cast_one]
  sorry

end mul_mixed_number_l727_727006


namespace probability_spade_heart_diamond_l727_727452

-- Condition: Definition of probability functions and a standard deck
def probability_of_first_spade (deck : Finset ℕ) : ℚ := 13 / 52
def probability_of_second_heart (deck : Finset ℕ) (first_card_spade : Prop) : ℚ := 13 / 51
def probability_of_third_diamond (deck : Finset ℕ) (first_card_spade : Prop) (second_card_heart : Prop) : ℚ := 13 / 50

-- Combined probability calculation
def probability_sequence_spade_heart_diamond (deck : Finset ℕ) : ℚ := 
  probability_of_first_spade deck * 
  probability_of_second_heart deck (true) * 
  probability_of_third_diamond deck (true) (true)

-- Lean statement proving the problem
theorem probability_spade_heart_diamond :
  probability_sequence_spade_heart_diamond (Finset.range 52) = 2197 / 132600 :=
by
  -- Proof steps will go here
  sorry

end probability_spade_heart_diamond_l727_727452


namespace cos_2017pi_minus_2alpha_is_half_l727_727845

theorem cos_2017pi_minus_2alpha_is_half (α : ℝ) (hα_quadrant : π / 2 < α ∧ α < π) 
  (htrig : sin α + cos α = sqrt 3 / 2) : 
  cos (2017 * π - 2 * α) = 1 / 2 :=
sorry

end cos_2017pi_minus_2alpha_is_half_l727_727845


namespace find_f1_l727_727128

def f (n : ℕ) : ℕ :=
if n ≥ 100 then n - 3 else f (f (n + 5))

theorem find_f1 : f 1 = 98 :=
sorry

end find_f1_l727_727128


namespace stratified_sampling_third_year_students_l727_727707

theorem stratified_sampling_third_year_students 
  (total_students : ℕ)
  (sample_size : ℕ)
  (ratio_1st : ℕ)
  (ratio_2nd : ℕ)
  (ratio_3rd : ℕ)
  (ratio_4th : ℕ)
  (h1 : total_students = 1000)
  (h2 : sample_size = 200)
  (h3 : ratio_1st = 4)
  (h4 : ratio_2nd = 3)
  (h5 : ratio_3rd = 2)
  (h6 : ratio_4th = 1) :
  (ratio_3rd : ℚ) / (ratio_1st + ratio_2nd + ratio_3rd + ratio_4th : ℚ) * sample_size = 40 :=
by
  sorry

end stratified_sampling_third_year_students_l727_727707


namespace correct_statements_count_l727_727859

noncomputable def f (x : ℝ) : ℝ := |Real.cos x| * Real.sin x

def statement1 : Prop := ∀ x, f (x + Real.pi) = f x
def statement2 : Prop := ∀ x1 x2, |f x1| = |f x2| → ∃ k : ℤ, x1 = x2 + k * Real.pi
def statement3 : Prop := ∀ x1 x2, -Real.pi/4 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ Real.pi/4 → f x1 < f x2
def statement4 : Prop := ∀ x, f (-Real.pi / 2 - x) = f (-Real.pi / 2)

def correctStatement : ℕ :=
  [statement1, statement2, statement3, statement4].count (λ s, s)

theorem correct_statements_count : correctStatement = 1 := sorry

end correct_statements_count_l727_727859


namespace roger_has_more_candies_l727_727262

def candies_sandra_bag1 : ℕ := 6
def candies_sandra_bag2 : ℕ := 6
def candies_roger_bag1 : ℕ := 11
def candies_roger_bag2 : ℕ := 3

def total_candies_sandra := candies_sandra_bag1 + candies_sandra_bag2
def total_candies_roger := candies_roger_bag1 + candies_roger_bag2

theorem roger_has_more_candies : (total_candies_roger - total_candies_sandra) = 2 := by
  sorry

end roger_has_more_candies_l727_727262


namespace collinear_AFP_l727_727943

-- Define the given geometric configuration
variables {A B C D E F P : Type*}

-- Assume the conditions provided in the problem
variables [triangle : Triangle A B C]
variables [collinear : Collinear_points B D E C]
variables [similar_triangles : Similar (Triangle A B C) (Triangle F D E)]
variables [intersection_point : Second_intersection (Circumcircle B E F) (Circumcircle C D F) P]

-- Prove that points A, F, and P are collinear
theorem collinear_AFP : Collinear_points A F P :=
sorry

end collinear_AFP_l727_727943


namespace main_l727_727782

noncomputable def f (x : ℝ) : ℝ := |x - 3| + |x + 1|

lemma part_I :
  {x : ℝ | f x < 6} = set.Ioo (-2 : ℝ) 4 :=
by sorry

lemma part_II :
  {a : ℝ | ∃ x : ℝ, f x < a} = set.Ioi 4 :=
by sorry

-- Main theorem combining both parts
theorem main :
  ({x : ℝ | f x < 6} = set.Ioo (-2 : ℝ) 4) ∧ 
  ({a : ℝ | ∃ x : ℝ, f x < a} = set.Ioi 4) :=
by sorry

end main_l727_727782


namespace shortest_path_l727_727916

-- Define the point structures
structure Point where
  x : ℝ
  y : ℝ

-- Define points and circle properties based on the conditions
def A : Point := {x := 0, y := 0}
def D : Point := {x := 15, y := 20}
def O : Point := {x := 7, y := 9}
def radius : ℝ := 6

-- Circle equation
def circle_eq (p : Point) : ℝ := (p.x - O.x) ^ 2 + (p.y - O.y) ^ 2

-- Shortest path length avoiding the circle
def shortest_path_length : ℝ := 2 * real.sqrt 94 + 2 * real.pi

-- Theorem to prove the shortest path length
theorem shortest_path : ∀ (A D : Point) (radius : ℝ), 
  A = {x := 0, y := 0} → D = {x := 15, y := 20} → radius = 6 → 
  ∃ (O : Point), O = {x := 7, y := 9} ∧ 
  (shortest_path_length = 2 * real.sqrt 94 + 2 * real.pi) := by
  sorry

end shortest_path_l727_727916


namespace tan_pi_plus_alpha_l727_727846

theorem tan_pi_plus_alpha (α : ℝ) (h1 : sin α = -2 / 3) (h2 : ∃ n : ℤ, (2 * n + 1) * π < α ∧ α < (2 * n + 2) * π):
  tan (π + α) = 2 * Real.sqrt 5 / 5 := 
by 
  sorry

end tan_pi_plus_alpha_l727_727846


namespace total_amount_paid_l727_727930

def first_payment : ℕ := 100
def multiplier : ℕ := 3
def months : ℕ := 5

def payment (n : ℕ) : ℕ :=
  match n with
  | 1 => first_payment
  | k + 1 => multiplier * payment k

def total_payment : ℕ :=
  (List.range months).map payment |>.sum

theorem total_amount_paid : total_payment = 12100 :=
  sorry

end total_amount_paid_l727_727930


namespace harkamal_paid_amount_l727_727336

variable (grapesQuantity : ℕ)
variable (grapesRate : ℕ)
variable (mangoesQuantity : ℕ)
variable (mangoesRate : ℕ)

theorem harkamal_paid_amount (h1 : grapesQuantity = 8) (h2 : grapesRate = 70) (h3 : mangoesQuantity = 9) (h4 : mangoesRate = 45) :
  (grapesQuantity * grapesRate + mangoesQuantity * mangoesRate) = 965 := by
  sorry

end harkamal_paid_amount_l727_727336


namespace find_x_l727_727094

def vector_a : ℝ × ℝ × ℝ := (2, -3, 1)
def vector_b (x : ℝ) : ℝ × ℝ × ℝ := (4, -6, x)
def dot_product : (ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ) → ℝ
  | (a1, a2, a3), (b1, b2, b3) => a1 * b1 + a2 * b2 + a3 * b3

theorem find_x (x : ℝ) (h : dot_product vector_a (vector_b x) = 0) : x = -26 :=
by 
  sorry

end find_x_l727_727094


namespace christina_speed_limit_l727_727040

theorem christina_speed_limit :
  ∀ (D total_distance friend_distance : ℝ), 
  total_distance = 210 → 
  friend_distance = 3 * 40 → 
  D = total_distance - friend_distance → 
  D / 3 = 30 :=
by
  intros D total_distance friend_distance 
  intros h1 h2 h3 
  sorry

end christina_speed_limit_l727_727040


namespace number_of_tricycles_correct_l727_727302

/-- Problem setup assumptions -/
def num_bicycles : ℕ := 16
def total_wheels : ℕ := 53
def wheels_per_bicycle : ℕ := 2
def wheels_per_tricycle : ℕ := 3

/-- Main statement to prove: There are exactly 7 tricycles -/
theorem number_of_tricycles_correct : 
  ∃ T : ℕ, num_bicycles * wheels_per_bicycle + T * wheels_per_tricycle = total_wheels ∧ T = 7 :=
by {
  use 7, -- Construct an example with T = 7
  split, -- Split the goal into proving the two parts of the conjunction
  {
    -- Prove that total wheels add up correctly
    rw [num_bicycles, wheels_per_bicycle],
    rw [mul_add, mul_comm, mul_assoc],
    norm_num
  },
  {
    -- Prove that T = 7
    refl
  }
}

end number_of_tricycles_correct_l727_727302


namespace count_integers_satisfy_conditions_l727_727070

def satisfies_conditions (n : ℤ) : Prop :=
  150 < n ∧ n < 300 ∧ ∃ r : ℤ, 0 ≤ r ∧ r ≤ 6 ∧ (n % 7 = r) ∧ (n % 9 = r)

theorem count_integers_satisfy_conditions : 
  (∃ count : ℕ, count = (Finset.card (Finset.filter satisfies_conditions (Finset.Icc 151 299)))) :=
sorry

end count_integers_satisfy_conditions_l727_727070


namespace number_of_solutions_eq_5_l727_727073

noncomputable def fractional_part (x : ℝ) := x - x.floor

theorem number_of_solutions_eq_5 :
  ∃ n : ℤ, ∀ x : ℝ, (x = n + fractional_part x) → (0 ≤ fractional_part x) ∧ (fractional_part x < 1) → 6 * (fractional_part x) ^ 3 + (fractional_part x) ^ 2 + (fractional_part x) + 2 * x = 2018 → 
  5 = ∑ i in finset.Icc (1005 : ℤ) 1009, 1 :=
sorry

end number_of_solutions_eq_5_l727_727073


namespace alexandra_magazines_l727_727735

noncomputable def magazines (bought_on_friday : ℕ) (bought_on_saturday : ℕ) (times_friday : ℕ) (chewed_up : ℕ) : ℕ :=
  bought_on_friday + bought_on_saturday + times_friday * bought_on_friday - chewed_up

theorem alexandra_magazines :
  ∀ (bought_on_friday bought_on_saturday times_friday chewed_up : ℕ),
      bought_on_friday = 8 → 
      bought_on_saturday = 12 → 
      times_friday = 4 → 
      chewed_up = 4 →
      magazines bought_on_friday bought_on_saturday times_friday chewed_up = 48 :=
by
  intros
  sorry

end alexandra_magazines_l727_727735


namespace kendra_fish_count_l727_727544

variable (K : ℕ) -- Number of fish Kendra caught
variable (Ken_fish : ℕ) -- Number of fish Ken brought home

-- Conditions
axiom twice_as_many : Ken_fish = 2 * K - 3
axiom total_fish : K + Ken_fish = 87

-- The theorem we need to prove
theorem kendra_fish_count : K = 30 :=
by
  -- Lean proof goes here
  sorry

end kendra_fish_count_l727_727544


namespace average_investment_per_km_in_scientific_notation_l727_727693

-- Definitions based on the conditions of the problem
def total_investment : ℝ := 29.6 * 10^9
def upgraded_distance : ℝ := 6000

-- A theorem to be proven
theorem average_investment_per_km_in_scientific_notation :
  (total_investment / upgraded_distance) = 4.9 * 10^6 :=
by
  sorry

end average_investment_per_km_in_scientific_notation_l727_727693


namespace group_product_number_l727_727692

theorem group_product_number (a : ℕ) (group_size : ℕ) (interval : ℕ) (fifth_group_product : ℕ) :
  fifth_group_product = a + 4 * interval → fifth_group_product = 94 → group_size = 5 → interval = 20 →
  (a + (1 - 1) * interval + 1 * interval) = 34 :=
by
  intros fifth_group_eq fifth_group_is_94 group_size_is_5 interval_is_20
  -- Missing steps are handled by sorry
  sorry

end group_product_number_l727_727692


namespace circle_equation_l727_727041

theorem circle_equation :
  ∃ (x0 r : ℝ), ((x0 > 0) ∧ (x0^2 + r^2 = x0^2) ∧ ((2 * sqrt 2)^2 = 2 * (r^2 - (1/sqrt 2 * x0)^2))) ∧
  ∀ x y : ℝ, (x - x0)^2 + y^2 = r^2 ↔ (x - 2)^2 + y^2 = 4 :=
begin
  sorry
end

end circle_equation_l727_727041


namespace melanie_cats_l727_727239

theorem melanie_cats (jacob_cats : ℕ) (annie_cats : ℕ) (melanie_cats : ℕ) 
  (h_jacob : jacob_cats = 90)
  (h_annie : annie_cats = jacob_cats / 3)
  (h_melanie : melanie_cats = annie_cats * 2) :
  melanie_cats = 60 := by
  sorry

end melanie_cats_l727_727239


namespace recurring_decimal_to_fraction_l727_727066

theorem recurring_decimal_to_fraction : 
  let x := 5. + (8 / 10) + (8 / 100) + (8 / 1000) + ... 
  in x = 53 / 9 := by
    sorry

end recurring_decimal_to_fraction_l727_727066


namespace Maria_green_towels_l727_727963

-- Definitions
variable (G : ℕ) -- number of green towels

-- Conditions
def initial_towels := G + 21
def final_towels := initial_towels - 34

-- Theorem statement
theorem Maria_green_towels : final_towels = 22 → G = 35 :=
by
  sorry

end Maria_green_towels_l727_727963


namespace fraction_not_covered_l727_727241

theorem fraction_not_covered (h1 : ℝ) (h2 : ℝ)
  (ratio1_width : ℝ) (ratio1_height : ℝ)
  (ratio2_width : ℝ) (ratio2_height : ℝ)
  (h_ratio1 : ratio1_width / ratio1_height = 16 / 9)
  (h_ratio2 : ratio2_width / ratio2_height = 4 / 3)
  (h_width_same : ratio1_width = ratio2_width) :
  (ratio2_height - (ratio1_height * ratio2_width / ratio1_width)) / ratio2_height = 1 / 4 :=
by 
  have h1 : ratio1_width / ratio1_height = 16 / 9 := h_ratio1,
  have h2 : ratio2_width / ratio2_height = 4 / 3 := h_ratio2,
  have h3 : ratio1_width = ratio2_width := h_width_same,
  sorry

end fraction_not_covered_l727_727241


namespace relationship_among_a_b_c_l727_727832

def a := Real.log 2 / Real.log 0.3
def b := Real.log 2 / Real.log 3
def c := 2 ^ 0.3

theorem relationship_among_a_b_c : c > b ∧ b > a := by
  sorry

end relationship_among_a_b_c_l727_727832


namespace find_a_l727_727471

theorem find_a : 
    (∃ a : ℝ, (∃ f : ℕ → ℝ, (∀ r : ℕ, f r = (Nat.choose 5 r) * (a^(5-r)) * (2^r) * 1) ∧ f 4 = 80)) → a = 1 :=
begin
  sorry
end

end find_a_l727_727471


namespace inversely_directly_proportional_l727_727615

theorem inversely_directly_proportional (m n z : ℝ) (x : ℝ) (h₁ : x = 4) (hz₁ : z = 16) (hz₂ : z = 64) (hy : ∃ y : ℝ, y = n * Real.sqrt z) (hx : ∃ m y : ℝ, x = m / y^2)
: x = 1 :=
by
  sorry

end inversely_directly_proportional_l727_727615


namespace inverse_of_A_l727_727427

open Matrix

noncomputable def A : Matrix (Fin 2) (Fin 2) ℚ :=
  matrix.of ![![3, 4], ![-2, 9]]

noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  matrix.of ![![9/35, -4/35], ![2/35, 3/35]]

theorem inverse_of_A :
  A⁻¹ = A_inv := by
  sorry

end inverse_of_A_l727_727427


namespace brother_age_l727_727717

variables (M B : ℕ)

theorem brother_age (h1 : M = B + 12) (h2 : M + 2 = 2 * (B + 2)) : B = 10 := by
  sorry

end brother_age_l727_727717


namespace find_line_equation_l727_727100

noncomputable def l1 : LinearMap ℝ (ℝ × ℝ) ℝ := (2, 1, -6 : ℝ)
noncomputable def l2 : LinearMap ℝ (ℝ × ℝ) ℝ := (4, 2, -5 : ℝ)

noncomputable def segment_length : ℝ := 7 / 2

def point : ℝ × ℝ := (0, -1)

theorem find_line_equation
  (l : ℝ → ℝ → ℝ)
  (intercepts_parallel_l1_l2_with_segment_length : ∀ x y, l x y ∈ [l1, l2] → dist (l1 x y) (l2 x y) = segment_length)
  (passes_through_point : l 0 -1 = 0) :
  (∃ c, ∃ d, ∀ x y, l x y = c * x + d * y + 4) ∨ (∀ x, l x 0 = 0) :=
sorry

end find_line_equation_l727_727100


namespace part1_part2_l727_727183

variables {A B C a b c : ℝ}

-- Condition: sides opposite to angles A, B, and C are a, b, and c respectively and 4b * sin A = sqrt 7 * a
def condition1 : 4 * b * Real.sin A = Real.sqrt 7 * a := sorry

-- Prove that sin B = sqrt 7 / 4
theorem part1 (h : 4 * b * Real.sin A = Real.sqrt 7 * a) :
  Real.sin B = Real.sqrt 7 / 4 := sorry

-- Condition: a, b, and c form an arithmetic sequence with a common difference greater than 0
def condition2 : 2 * b = a + c := sorry

-- Prove that cos A - cos C = sqrt 7 / 2
theorem part2 (h1 : 4 * b * Real.sin A = Real.sqrt 7 * a) (h2 : 2 * b = a + c) :
  Real.cos A - Real.cos C = Real.sqrt 7 / 2 := sorry

end part1_part2_l727_727183


namespace tan_eq_one_eq_1_tan_eq_one_eq_2_l727_727847

variable (α : ℝ)

theorem tan_eq_one_eq_1 : 
  (tan α = 1) → (cos α + 2 * sin α) / (2 * cos α - 3 * sin α) = -3 :=
by sorry

theorem tan_eq_one_eq_2 : 
  (tan α = 1) → sin α ^ 2 + sin (2 * α) = 3 / 2 :=
by sorry

end tan_eq_one_eq_1_tan_eq_one_eq_2_l727_727847


namespace equal_distribution_of_drawings_l727_727607

theorem equal_distribution_of_drawings (total_drawings : ℕ) (neighbors : ℕ) (drawings_per_neighbor : ℕ)
  (h1 : total_drawings = 54)
  (h2 : neighbors = 6)
  (h3 : total_drawings = neighbors * drawings_per_neighbor) :
  drawings_per_neighbor = 9 :=
by
  rw [h1, h2] at h3
  linarith

end equal_distribution_of_drawings_l727_727607


namespace boundary_length_of_divided_square_l727_727725

theorem boundary_length_of_divided_square 
    (area : ℝ) (segments : ℕ) (π : ℝ) [Fact (π = Real.pi)] 
    (h_area : area = 256) (h_segments : segments = 4) :
    Float.round (16 + 8 * π) 1 = 41.1 :=
by
  sorry

end boundary_length_of_divided_square_l727_727725


namespace total_hot_dogs_by_wednesday_l727_727147

theorem total_hot_dogs_by_wednesday :
  ∀ (dogs_on_monday : ℕ) (daily_increase : ℕ),
    dogs_on_monday = 10 →
    daily_increase = 2 →
    let monday := dogs_on_monday in
    let tuesday := monday + daily_increase in
    let wednesday := tuesday + daily_increase in
    let total_by_tuesday := monday + tuesday in
    let total_by_wednesday := total_by_tuesday + wednesday in
    total_by_wednesday = 36 :=
by
  intros dogs_on_monday daily_increase monday_eq increase_eq,
  let monday := dogs_on_monday,
  let tuesday := monday + daily_increase,
  let wednesday := tuesday + daily_increase,
  let total_by_tuesday := monday + tuesday,
  let total_by_wednesday := total_by_tuesday + wednesday,
  sorry

end total_hot_dogs_by_wednesday_l727_727147


namespace vector_dot_product_l727_727088

open Complex

def a : Complex := (1 : ℝ) + (-(2 : ℝ)) * Complex.I
def b : Complex := (-3 : ℝ) + (4 : ℝ) * Complex.I
def c : Complex := (3 : ℝ) + (2 : ℝ) * Complex.I

-- Note: Using real coordinates to simulate vector operations.
theorem vector_dot_product :
  let a_vec := (1, -2)
  let b_vec := (-3, 4)
  let c_vec := (3, 2)
  let linear_combination := (a_vec.1 + 2 * b_vec.1, a_vec.2 + 2 * b_vec.2)
  (linear_combination.1 * c_vec.1 + linear_combination.2 * c_vec.2) = -3 := 
by
  sorry

end vector_dot_product_l727_727088


namespace Pasha_goal_achievable_in_43_moves_l727_727973

-- Define what Pasha's goal is under the specified conditions
theorem Pasha_goal_achievable_in_43_moves :
  ∃ (a : ℕ → ℕ) (moves : ℕ), 
  moves = 43 ∧ 
  ∃ n : ℕ, n = 2017 ∧ 
  (∀ m < moves, ∀ (c d : ℕ) (h₁ : c ≠ d) (h₂ : c < n) (h₃ : d < n),
  ∀ stones, 
    stones = ((list.range n).map a).sum ∧
    stones % n = 0) ∧ 
  (∀ (k : ℕ), k < 43 → 
    ¬ (∀ (c d : ℕ) (h₁ : c ≠ d) (h₂ : c < n) (h₃ : d < n),
    ∀ stones, 
      stones = ((list.range n).map a).sum ∧
      stones % k = 0)) :=
begin
  sorry
end

end Pasha_goal_achievable_in_43_moves_l727_727973


namespace number_of_apple_trees_l727_727256

variable (T : ℕ) -- Declare the number of apple trees as a natural number

-- Define the conditions
def picked_apples := 8 * T
def remaining_apples := 9
def initial_apples := 33

-- The statement to prove Rachel has 3 apple trees
theorem number_of_apple_trees :
  initial_apples - picked_apples + remaining_apples = initial_apples → T = 3 := 
by
  sorry

end number_of_apple_trees_l727_727256


namespace sum_T_base3_l727_727552

def is_three_digit_base3 (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a ∈ {1, 2} ∧ b < 3 ∧ c < 3 ∧ n = a * 3^2 + b * 3 + c

def T : set ℕ := { n | is_three_digit_base3 n }

noncomputable def sum_T : ℕ := ∑ n in T, n

theorem sum_T_base3 : sum_T = 102120 :=
sorry

end sum_T_base3_l727_727552


namespace harmonic_sum_base_case_l727_727976

theorem harmonic_sum_base_case : 1 + 1/2 + 1/3 < 2 := 
sorry

end harmonic_sum_base_case_l727_727976


namespace range_of_a_l727_727164

theorem range_of_a (x : ℝ) (a : ℝ) (h1 : 2 < x) (h2 : a ≤ x + 1 / (x - 2)) : a ≤ 4 := 
sorry

end range_of_a_l727_727164


namespace proof_problem_l727_727104

noncomputable def a (n : ℕ) : ℕ :=
if n = 0 then 1 
else 2 ^ (n - 1)

def b (n : ℕ) : ℤ :=
7 - 3 * n

def sum_seq (n : ℕ) : ℤ :=
∑ i in Finset.range n, (a i + b i)

theorem proof_problem (n : ℕ) :
  ( ∀ n, (a 1 = 1) ∧ (a (n + 1) = 2 * a n) ) →
  (b 1 = a 3 ∧ b 2 = a 1) →
  (∀ n, a n = 2 ^ (n - 1)) ∧
  (∀ n, b n = 7 - 3 * n) ∧
  (sum_seq n = 2^n - 1 + (11 * n - 3 * n^2) / 2) :=
by
  intros h1 h2
  sorry

end proof_problem_l727_727104


namespace expected_potato_harvest_l727_727240

theorem expected_potato_harvest :
  let steps_length_ft := 3
  let length_steps := 18
  let width_steps := 25
  let yield_per_sqft := 3 / 4
  let length_ft := length_steps * steps_length_ft
  let width_ft := width_steps * steps_length_ft
  let area_sqft := length_ft * width_ft
  let expected_yield_pounds := area_sqft * yield_per_sqft
  in
  expected_yield_pounds = 3038 :=
by
  sorry

end expected_potato_harvest_l727_727240


namespace sequence_property_l727_727939

theorem sequence_property (a : ℕ → ℝ)
    (h_rec : ∀ n ≥ 2, a n = a (n - 1) * a (n + 1))
    (h_a1 : a 1 = 1 + Real.sqrt 7)
    (h_1776 : a 1776 = 13 + Real.sqrt 7) :
    a 2009 = -1 + 2 * Real.sqrt 7 := 
    sorry

end sequence_property_l727_727939


namespace lines_do_not_form_triangle_l727_727494

noncomputable def line1 (x y : ℝ) := 3 * x - y + 2 = 0
noncomputable def line2 (x y : ℝ) := 2 * x + y + 3 = 0
noncomputable def line3 (m x y : ℝ) := m * x + y = 0

theorem lines_do_not_form_triangle (m : ℝ) :
  (∃ x y : ℝ, line1 x y ∧ line2 x y) →
  (∀ x y : ℝ, (line1 x y → line3 m x y) ∨ (line2 x y → line3 m x y) ∨ 
    (line1 x y ∧ line2 x y → line3 m x y)) →
  (m = -3 ∨ m = 2 ∨ m = -1) :=
by
  sorry

end lines_do_not_form_triangle_l727_727494


namespace paintings_per_gallery_l727_727381

theorem paintings_per_gallery (pencils_total: ℕ) (pictures_initial: ℕ) (galleries_new: ℕ) (pencils_per_signing: ℕ) (pencils_per_picture: ℕ) (pencils_for_signature: ℕ) :
  pencils_total = 88 ∧ pictures_initial = 9 ∧ galleries_new = 5 ∧ pencils_per_picture = 4 ∧ pencils_per_signing = 2 → 
  (pencils_total - (galleries_new + 1) * pencils_per_signing) / pencils_per_picture - pictures_initial = galleries_new * 2 :=
by
  intros h,
  cases h with ha hb,
  sorry

end paintings_per_gallery_l727_727381


namespace smaller_solution_of_quadratic_eq_l727_727435

theorem smaller_solution_of_quadratic_eq : 
  (exists x y : ℝ, x < y ∧ x^2 - 13 * x + 36 = 0 ∧ y^2 - 13 * y + 36 = 0 ∧ x = 4) :=
by sorry

end smaller_solution_of_quadratic_eq_l727_727435


namespace smaller_inscribed_cube_volume_l727_727367

theorem smaller_inscribed_cube_volume:
  let edge_length := 12
  let sphere_diameter := edge_length
  let smaller_cube_diagonal := sphere_diameter / 2
  let smaller_cube_side := smaller_cube_diagonal / (Real.sqrt 3)
  let smaller_cube_volume := Real.pow (smaller_cube_side) 3
  smaller_cube_volume = 24 * Real.sqrt 3 := by
  sorry

end smaller_inscribed_cube_volume_l727_727367


namespace log_inequalities_l727_727829

theorem log_inequalities (a b : ℝ) (h1 : 0 < b) (h2 : b < 1) (h3 : 0 < a) (h4 : a < π/2) :
  let x := log b (sin a),
      y := log b (cos a),
      z := log b (sin a)
  in x < z ∧ z < y :=
by {
  -- Definitions
  let x := log b (sin a),
  let y := log b (cos a),
  let z := log b (sin a),
  -- Proof would go here
  sorry
}

end log_inequalities_l727_727829


namespace problem_statement_l727_727076

noncomputable def find_n : ℕ :=
  let n : ℕ := 4064255 in
  if h : (∑ k in Finset.range n | ((1 : ℝ) / (Real.sqrt k + Real.sqrt (k + 1)))) = 2015 then n
  else 0

theorem problem_statement :
  ∑ k in Finset.range 4064255 | ((1 : ℝ) / (Real.sqrt k + Real.sqrt (k + 1))) = 2015 :=
sorry

end problem_statement_l727_727076


namespace min_length_GH_l727_727123

theorem min_length_GH :
  let ellipse (x y : ℝ) := (x^2 / 4) + y^2 = 1
  let A := (-2, 0)
  let B := (2, 0)
  ∀ P G H : ℝ × ℝ,
    (P.1^2 / 4 + P.2^2 = 1) →
    P.2 > 0 →
    (G.2 = 3) →
    (H.2 = 3) →
    ∃ k : ℝ, k > 0 ∧ G.1 = 3 / k - 2 ∧ H.1 = -12 * k + 2 →
    |G.1 - H.1| = 8 :=
sorry

end min_length_GH_l727_727123


namespace g_x_plus_2_minus_g_x_l727_727505

def g (x : ℝ) : ℝ := 8^x

theorem g_x_plus_2_minus_g_x (x : ℝ) : g(x + 2) - g(x) = 63 * g(x) := 
by sorry

end g_x_plus_2_minus_g_x_l727_727505


namespace distance_PQ_in_triangle_ABC_l727_727536

theorem distance_PQ_in_triangle_ABC :
  ∀ (A B C P Q: ℝ × ℝ),
  let AB := dist A B in
  let AC := dist A C in
  let BC := dist B C in
  let P := midpoint ℝ A B in
  let Q := point_on_line ℝ A C 1 in
  AB = 4 → AC = 3 → BC = sqrt 37 →
  dist P Q = 2 * sqrt 3 :=
by
  intros A B C P Q AB AC BC PQ HPQ Q T1 T2 T3 
  sorry

# I expect the generated code to pass the import test without the complete proof.

end distance_PQ_in_triangle_ABC_l727_727536


namespace max_imag_part_theta_l727_727385

noncomputable def polynomial := (λ z : ℂ, z^10 - z^8 + z^6 - z^4 + z^2 - 1)

theorem max_imag_part_theta :
  ∀ θ : ℝ, -90 ≤ θ ∧ θ ≤ 90 →
  let z := complex.of_real θ in
  polynomial z = 0 → ∃ θ, θ = 90 :=
by
-- Proof is omitted
sorry

end max_imag_part_theta_l727_727385


namespace train_cross_pole_in_21_seconds_l727_727728

variable (speed_kmph : ℕ) (length_m : ℕ)

def speed_mps (speed_kmph : ℕ) : ℝ := speed_kmph * 1000 / 3600
def time_cross_pole (length_m : ℕ) (speed_kmph : ℕ) : ℝ := length_m / speed_mps speed_kmph

theorem train_cross_pole_in_21_seconds
  (h_speed : speed_kmph = 60)
  (h_length : length_m = 350) :
  time_cross_pole length_m speed_kmph ≈ 21 :=
by
  sorry

end train_cross_pole_in_21_seconds_l727_727728


namespace flower_problem_l727_727584

def totalFlowers (n_rows n_per_row : Nat) : Nat :=
  n_rows * n_per_row

def flowersCut (total percent_cut : Nat) : Nat :=
  total * percent_cut / 100

def flowersRemaining (total cut : Nat) : Nat :=
  total - cut

theorem flower_problem :
  let n_rows := 50
  let n_per_row := 400
  let percent_cut := 60
  let total := totalFlowers n_rows n_per_row
  let cut := flowersCut total percent_cut
  flowersRemaining total cut = 8000 :=
by
  sorry

end flower_problem_l727_727584


namespace dance_relationship_l727_727744

theorem dance_relationship (b g : ℕ) 
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ b → i = 1 → ∃ m, m = 7)
  (h2 : b = g - 6) 
  : 7 + (b - 1) = g := 
by
  sorry

end dance_relationship_l727_727744


namespace total_hot_dogs_by_wednesday_l727_727148

theorem total_hot_dogs_by_wednesday :
  ∀ (dogs_on_monday : ℕ) (daily_increase : ℕ),
    dogs_on_monday = 10 →
    daily_increase = 2 →
    let monday := dogs_on_monday in
    let tuesday := monday + daily_increase in
    let wednesday := tuesday + daily_increase in
    let total_by_tuesday := monday + tuesday in
    let total_by_wednesday := total_by_tuesday + wednesday in
    total_by_wednesday = 36 :=
by
  intros dogs_on_monday daily_increase monday_eq increase_eq,
  let monday := dogs_on_monday,
  let tuesday := monday + daily_increase,
  let wednesday := tuesday + daily_increase,
  let total_by_tuesday := monday + tuesday,
  let total_by_wednesday := total_by_tuesday + wednesday,
  sorry

end total_hot_dogs_by_wednesday_l727_727148


namespace problem1_proof_problem2_proof_l727_727751

-- Problem 1 proof statement
theorem problem1_proof : (-1)^10 * 2 + (-2)^3 / 4 = 0 := 
by
  sorry

-- Problem 2 proof statement
theorem problem2_proof : -24 * (5 / 6 - 4 / 3 + 3 / 8) = 3 :=
by
  sorry

end problem1_proof_problem2_proof_l727_727751


namespace find_a_b_part_two_l727_727861

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x^2

theorem find_a_b :
  let a := 1
  let b := 1
  a = 1 ∧ b = 1 := by
  sorry

theorem part_two (x : ℝ) (hx : x ≥ 0) :
  f(x) > x^2 + 4 * x - 14 :=
by
  sorry

end find_a_b_part_two_l727_727861


namespace max_markable_squares_l727_727316

theorem max_markable_squares (knight_moves_L : ∀ (x y : ℕ), (x, y) ∈ {(2, 1), (1, 2), (-2, -1), (-1, -2), (2, -1), (1, -2), (-2, 1), (-1, 2)}) 
                             (chessboard : Π (i j : ℕ), i < 8 ∧ j < 8 → Prop)
                             (markable_same_color : ∀ (i j : ℕ), chessboard i j → (i + j) % 2 = 0) :
  ∃ (marked_cells : set (ℕ × ℕ)), 
  marked_cells ⊆ {(i, j) | i < 8 ∧ j < 8 ∧ (i + j) % 2 = 0} ∧ 
  (∀ (a b : ℕ × ℕ), a ∈ marked_cells → b ∈ marked_cells → ∃ x y, (x, y) ∈ {(2, 1), (1, 2), (-2, -1), (-1, -2), (2, -1), (1, -2), (-2, 1), (-1, 2)} ∧ 
                  (fst b - fst a, snd b - snd a) = (x, y)) ∧ 
  ∀ (a : ℕ × ℕ), a ∈ marked_cells → ∀ (x y : ℕ), (x, y) ∈ {(4, 4), (4, -4), (-4, 4), (-4, -4)} → 
                  (fst a + x, snd a + y) ∉ marked_cells ∧ 
  card marked_cells = 8 := sorry

end max_markable_squares_l727_727316


namespace range_of_a_l727_727515

def circle_eq (a : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - a)^2 + (p.2 - a)^2 = 8}

def distance_to_origin (p : (ℝ × ℝ)) : ℝ := Real.sqrt (p.1^2 + p.2^2)

theorem range_of_a :
  (∀ (a : ℝ), ∃ p ∈ circle_eq a, distance_to_origin p = Real.sqrt 2) ↔
    ∀ a, (1 ≤ |a| ∧ |a| ≤ 3) :=
by
  sorry

end range_of_a_l727_727515


namespace triathlete_average_speed_l727_727370

noncomputable def harmonic_mean (a b c : ℝ) : ℝ :=
  3 / (1/a + 1/b + 1/c)

theorem triathlete_average_speed :
  let swim_speed := 2
  let bike_speed := 25
  let run_speed := 12
  harmonic_mean swim_speed bike_speed run_speed ≈ 4.8 :=
by
  sorry

end triathlete_average_speed_l727_727370


namespace gcd_12012_18018_l727_727809

theorem gcd_12012_18018 : Int.gcd 12012 18018 = 6006 := 
by
  sorry

end gcd_12012_18018_l727_727809


namespace sqrt_720_l727_727995

theorem sqrt_720 : sqrt (720) = 12 * sqrt (5) :=
sorry

end sqrt_720_l727_727995


namespace multiplication_with_mixed_number_l727_727013

-- Define mixed numbers as rational numbers for proper calculation
def mixed_to_rational (whole : ℕ) (num : ℕ) (den : ℕ) : ℚ :=
  whole + num / den

-- 7 * (9 + 2/5)
def lhs : ℚ := 7 * mixed_to_rational 9 2 5

-- 65 + 4/5
def rhs : ℚ := mixed_to_rational 65 4 5

theorem multiplication_with_mixed_number : lhs = rhs := by
  sorry

end multiplication_with_mixed_number_l727_727013


namespace prime_divisors_of_N_l727_727080

def Graph := ... -- Assuming a suitable graph representation
def is_valid_vertex_assignment := ... -- Predicate for valid vertex assignment
def is_valid_edge_assignment := ... -- Predicate for valid edge assignment

variable (n : ℕ) (G : Graph)
variable (vertex_labels : ... ) -- Assuming appropriate type for vertex labels
variable (edge_labels : ... ) -- Assuming appropriate type for edge labels
variable [finite_graph G]
variable [positive_integer n]
variable [valid_graph G]
variable [valid_vertex_assignment vertex_labels n]
noncomputable def N : ℕ := ... -- Definition of N based on the compatible edge assignments

theorem prime_divisors_of_N (n : ℕ) (G : Graph) (vertex_labels : ...) 
  (h_finite : finite_graph G)
  (h_positive : positive_integer n)
  (h_valid_graph : valid_graph G)
  (h_vertex_assignment : valid_vertex_assignment vertex_labels n)
  (N ≠ 0) : 
  ∀ p : ℕ, p.prime → p ∣ N → p ≤ n :=
sorry

end prime_divisors_of_N_l727_727080


namespace multiples_of_5_count_l727_727442

def f (n : ℕ) : ℕ := 2 * n ^ 5 + 3 * n ^ 4 + 5 * n ^ 3 + 2 * n ^ 2 + 3 * n + 6

def count_multiples_of_5_in_range (a b : ℕ) : ℕ :=
  List.length $ List.filter (λ n, f n % 5 = 0) (List.range' a (b - a + 1))

theorem multiples_of_5_count : count_multiples_of_5_in_range 2 100 = 40 := by
  sorry

end multiples_of_5_count_l727_727442


namespace find_p_q_l727_727886

theorem find_p_q : 
  (∀ x : ℝ, (x - 2) * (x + 1) ∣ (x ^ 5 - x ^ 4 + x ^ 3 - p * x ^ 2 + q * x - 8)) → (p = -1 ∧ q = -10) :=
by
  sorry

end find_p_q_l727_727886


namespace multiply_mixed_number_l727_727027

def mixed_to_improper_fraction (n : ℕ) (a b : ℕ) : ℚ :=
  n + a / b

def improper_to_mixed_number (q : ℚ) : ℕ × ℚ :=
  let n := q.natAbs
  let r := q.fract
  (n, r)

theorem multiply_mixed_number (x y n a b : ℕ) (h : y = mixed_to_improper_fraction n a b) :
  x * y = mixed_to_improper_fraction 65 4 5 :=
  sorry

end multiply_mixed_number_l727_727027


namespace zero_in_interval_l727_727533

noncomputable def f (x : ℝ) : ℝ := exp x + 4 * x - 3

theorem zero_in_interval :
  ∃ x, (1/4 : ℝ) < x ∧ x < (1/2 : ℝ) ∧ f x = 0 :=
sorry

end zero_in_interval_l727_727533


namespace transform_polynomial_to_y_l727_727125

theorem transform_polynomial_to_y (x y : ℝ) (h : y = x + 1/x) :
  (x^6 + x^5 - 5*x^4 + x^3 + x + 1 = 0) → 
  (∃ (y_expr : ℝ), (x * y_expr = 0 ∨ (x = 0 ∧ y_expr = y_expr))) :=
sorry

end transform_polynomial_to_y_l727_727125


namespace sum_of_solutions_l727_727052

noncomputable def given_equation (x : ℝ) : Prop :=
  3 * Real.sin (2 * x) * (Real.sin (2 * x) - Real.sin (4028 * Real.pi / x)) = Real.sin (4 * x) - 1

theorem sum_of_solutions :
  ∑ (x : ℝ) in { x : ℝ | x > 0 ∧ given_equation x }.finset, x = 1080 * Real.pi :=
sorry

end sum_of_solutions_l727_727052


namespace tangent_line_value_of_m_l727_727121

theorem tangent_line_value_of_m:
  (∀ (x : ℝ), x > 0 → deriv (λ x, x^2 - 3 * log x) x = -1 -> (x, x^2 - 3 * log x) ∈ {(x, y) | y = -x + 2}) →
  ∀ m : ℝ, (∀ x : ℝ, x > 0 → y = -x + m) → m = 2 :=
by
  sorry

end tangent_line_value_of_m_l727_727121


namespace simplest_square_root_correct_l727_727320

def simplest_square_root (x : ℝ) : Prop :=
  x = real.sqrt 3

def candidates : list ℝ := [real.sqrt (1 / 3), real.sqrt 3, real.sqrt 0.3, real.sqrt 1]

theorem simplest_square_root_correct : simplest_square_root (real.sqrt 3) :=
by {
  -- Verify the candidates list doesn't contain any simpler form than sqrt(3)
  have h1 : real.sqrt (1 / 3) ≠ real.sqrt 3, sorry,
  have h2 : real.sqrt 0.3 ≠ real.sqrt 3, sorry,
  have h3 : real.sqrt 1 = 1, sorry,
  -- Based on the above, sqrt(3) is concluded as the simplest
  exact rfl
}

end simplest_square_root_correct_l727_727320


namespace min_log_sum_l727_727099

-- Given conditions
variables {a : ℕ → ℝ}
variables {n : ℕ}

-- The sequence is geometric
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (i j k : ℕ) (h1 : i < j) (h2 : j < k), a j ^ 2 = a i * a k

-- The sequence contains positive terms
def positive_terms (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0

-- The sum of logarithms condition
def log_sum_condition (a : ℕ → ℝ) : Prop :=
  ∑ i in finRange 2009, Real.logBase 2 (a i) = 2009

-- The proof problem statement
theorem min_log_sum (a : ℕ → ℝ) (h_geom : is_geometric_sequence a) (h_pos : positive_terms a) (h_log_sum : log_sum_condition a) :
  ∃ b : ℝ, b = a 0 + a 2008 ∧ Real.logBase 2 b ≥ 2 :=
sorry

end min_log_sum_l727_727099


namespace derivative_y_l727_727421

noncomputable def y (x : ℝ) : ℝ := x / (1 - cos x)

theorem derivative_y (x : ℝ) :
  deriv y x = (1 - cos x - x * sin x) / (1 - cos x)^2 :=
by
  sorry  -- Proof placeholder

end derivative_y_l727_727421


namespace population_percentage_5000_to_20000_l727_727720

def pie_chart_distribution : ℕ → ℕ := 
λ n, if n < 5000 then 35 else if n <= 20000 then 40 else 25

theorem population_percentage_5000_to_20000 :
  ∀ n : ℕ, 5000 ≤ n ∧ n ≤ 20000 → pie_chart_distribution n = 40 :=
begin
  intros n hn,
  rw pie_chart_distribution,
  split_ifs,
  { exfalso, linarith, },
  { exact rfl, },
  { exfalso, linarith, },
end

end population_percentage_5000_to_20000_l727_727720


namespace sam_dad_gave_39_nickels_l727_727606

-- Define the initial conditions
def initial_pennies : ℕ := 49
def initial_nickels : ℕ := 24
def given_quarters : ℕ := 31
def dad_given_nickels : ℕ := 63 - initial_nickels

-- Statement to prove
theorem sam_dad_gave_39_nickels 
    (total_nickels_after : ℕ) 
    (initial_nickels : ℕ) 
    (final_nickels : ℕ := total_nickels_after - initial_nickels) : 
    final_nickels = 39 :=
sorry

end sam_dad_gave_39_nickels_l727_727606


namespace c_range_l727_727852

open Real

theorem c_range (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : 1 / a + 1 / b = 1)
  (h2 : 1 / (a + b) + 1 / c = 1) : 1 < c ∧ c ≤ 4 / 3 := 
sorry

end c_range_l727_727852


namespace even_function_f_f_monotonic_increasing_and_inverse_h_function_inequality_and_m_range_l727_727045

theorem even_function_f (f : ℝ → ℝ) (hf : ∀ x, f x = f (-x)) 
  (g : ℝ → ℝ) (hg : ∀ x, g x = -g (-x)) 
  (h : ∀ x, f x + g x = 2^(x + 1)) : 
  f = fun x => 2^x + 1 / 2^x ∧ g = fun x => 2^x - 1 / 2^x :=
by sorry

theorem f_monotonic_increasing_and_inverse (f : ℝ → ℝ) (hf : ∀ x, f x = f (-x)) 
  (hf_expr : f = fun x => 2^x + 1 / 2^x) :
  (∀ x1 x2, 0 ≤ x1 → x1 < x2 → f x1 < f x2) ∧
  (∀ y, y ≥ 2 → f⁻¹ y = 2⁻¹ * log (y + sqrt (y^2 - 4))) :=
by sorry

theorem h_function_inequality_and_m_range (g : ℝ → ℝ) (hf : ∀ x, f x = f (-x))
  (hf_expr : f = fun x => 2^x + 1 / 2^x)
  (hg : ∀ x, g x = -g (-x))
  (hg_expr : g = fun x => 2^x - 1 / 2^x) 
  (h : ℝ → ℝ) (hh : ∀ (x : ℝ), h x = x^2 + 2 * m * x + m^2 - m + 1)
  (ineq : ∀ x ∈ set.Icc 1 2, h (g x) ≥ m^2 - m - 1) : 
  m ≥ -17/12 :=
by sorry

end even_function_f_f_monotonic_increasing_and_inverse_h_function_inequality_and_m_range_l727_727045


namespace moving_circle_trajectory_eq_l727_727462

theorem moving_circle_trajectory_eq (x y : ℝ) :
  let stationary_circle_eq := (x - 5)^2 + (y + 7)^2 = 16
  let moving_circle_radius := 1
  let stationary_circle_center_x := 5
  let stationary_circle_center_y := -7
  let stationary_circle_radius := 4
in ((x - stationary_circle_center_x)^2 + (y + stationary_circle_center_y)^2 = (4 + moving_circle_radius)^2) ↔
   ((x - 5)^2 + (y + 7)^2 = 25) := by
  sorry

end moving_circle_trajectory_eq_l727_727462


namespace inequality_proof_l727_727255

theorem inequality_proof (n : ℕ) (h₁ : n > 0) :
  (1 : ℝ) / (2 * Real.sqrt n) < (finset.prod (finset.range n) (λ k, ((2*k + 1) / (2*(k + 1)))) / 2)  ∧ 
  (finset.prod (finset.range n) (λ k, ((2*k + 1) / (2*(k + 1)))) / 2)  < 1 / Real.sqrt (2*n) := 
sorry

end inequality_proof_l727_727255


namespace parabola_focus_distance_correct_l727_727177

noncomputable def parabola_focus_distance : Prop :=
  ∀ {M : ℝ × ℝ}, (M.1^2 + M.2^2 = 4) → (M.2^2 = 3 * M.1) → 
  let focus := (3 / 4, 0) in 
  real.sqrt ((M.1 - focus.1)^2 + M.2^2) = 7 / 4

-- Here, sorry is added to skip the proof
theorem parabola_focus_distance_correct : parabola_focus_distance :=
  sorry

end parabola_focus_distance_correct_l727_727177


namespace sequence_term_eq_nine_l727_727870

theorem sequence_term_eq_nine (n : ℕ) : (n = 14) ↔ (sqrt (3 * (2 * n - 1)) = 9) := 
  sorry

end sequence_term_eq_nine_l727_727870


namespace intersection_of_parabolas_l727_727590

theorem intersection_of_parabolas
  (a h d : ℝ)
  (ha : a ≠ 0) :
  ∃ (x y : ℝ), x = -3 / 2 ∧ y = a * ((-3/2) - h)^2 + d ∧
    (a * (x - h)^2 + d) = y ∧
    (a * (((x + 3) - h)^2 + d)) = y :=
begin
  sorry
end

end intersection_of_parabolas_l727_727590


namespace race_speeds_l727_727900

theorem race_speeds (x y : ℕ) 
  (h1 : 5 * x + 10 = 5 * y) 
  (h2 : 6 * x = 4 * y) :
  x = 4 ∧ y = 6 :=
by {
  -- Proof will go here, but for now we skip it.
  sorry
}

end race_speeds_l727_727900


namespace pencils_sold_l727_727891

theorem pencils_sold (C S : ℝ) (n : ℝ) 
  (h1 : 12 * C = n * S) (h2 : S = 1.5 * C) : n = 8 := by
  sorry

end pencils_sold_l727_727891


namespace maximum_constant_log2_l727_727055

theorem maximum_constant_log2 (n : ℕ) (x : ℕ → ℝ) (h0 : x 0 = 1) (h1 : x n = 2) (h2 : ∀ i, 1 ≤ i → i ≤ n → x i > 0): 
  ∑ i in Finset.range n, (x (i+1) / x i) ≥ n + Real.log 2 :=
by
  sorry

end maximum_constant_log2_l727_727055


namespace sum_is_zero_l727_727572

theorem sum_is_zero (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : a + b + c = 0) :
  (a / |a|) + (b / |b|) + (c / |c|) + ((a * b * c) / |a * b * c|) = 0 :=
by
  sorry

end sum_is_zero_l727_727572


namespace possible_m_value_l727_727103

variable (a b m t : ℝ)
variable (h_a : a ≠ 0)
variable (h1 : ∃ t, ∀ x, ax^2 - bx ≥ -1 ↔ (x ≤ t - 1 ∨ x ≥ -3 - t))
variable (h2 : a * m^2 - b * m = 2)

theorem possible_m_value : m = 1 :=
sorry

end possible_m_value_l727_727103


namespace smaller_solution_of_quadratic_eq_l727_727436

theorem smaller_solution_of_quadratic_eq : 
  (exists x y : ℝ, x < y ∧ x^2 - 13 * x + 36 = 0 ∧ y^2 - 13 * y + 36 = 0 ∧ x = 4) :=
by sorry

end smaller_solution_of_quadratic_eq_l727_727436


namespace sqrt_720_simplified_l727_727987

theorem sqrt_720_simplified : (sqrt 720 = 12 * sqrt 5) :=
by
  -- The proof is omitted as per the instructions
  sorry

end sqrt_720_simplified_l727_727987


namespace gcd_12012_18018_l727_727803

theorem gcd_12012_18018 : Nat.gcd 12012 18018 = 6006 := sorry

end gcd_12012_18018_l727_727803


namespace part_I_part_II_part_III_l727_727834

noncomputable def f (x : ℝ) : ℝ := x^3 - x

theorem part_I : 
  ∀ x:ℝ, f x = x^3 - x :=
by sorry

theorem part_II : 
  ∃ (x1 x2 : ℝ), x1 ∈ Set.Icc (-1:ℝ) 1 ∧ x2 ∈ Set.Icc (-1:ℝ) 1 ∧ (3 * x1^2 - 1) * (3 * x2^2 - 1) = -1 :=
by sorry

theorem part_III (x_n y_m : ℝ) (hx : x_n ∈ Set.Icc (-1:ℝ) 1) (hy : y_m ∈ Set.Icc (-1:ℝ) 1) : 
  |f x_n - f y_m| < 1 :=
by sorry

end part_I_part_II_part_III_l727_727834


namespace linear_dependent_vectors_l727_727288

theorem linear_dependent_vectors (k : ℤ) :
  (∃ (a b : ℤ), (a ≠ 0 ∨ b ≠ 0) ∧ a * 2 + b * 4 = 0 ∧ a * 3 + b * k = 0) ↔ k = 6 :=
by
  sorry

end linear_dependent_vectors_l727_727288


namespace conversion_base8_to_base10_l727_727762

theorem conversion_base8_to_base10 : 
  (4 * 8^3 + 5 * 8^2 + 3 * 8^1 + 2 * 8^0) = 2394 := by 
  sorry

end conversion_base8_to_base10_l727_727762


namespace exists_n_lcm_lt_l727_727048

theorem exists_n_lcm_lt {p q : ℕ} (hp1 : p > 1) (hq1 : q > 1) (hpq_coprime : Nat.coprime p q) (h_distinct : (q - p).natAbs > 1) : 
  ∃ n : ℕ, n > 0 ∧ Nat.lcm (p + n) (q + n) < Nat.lcm p q := 
by 
  sorry

end exists_n_lcm_lt_l727_727048


namespace equation_of_line_passing_through_point_parallel_to_vector_l727_727876

theorem equation_of_line_passing_through_point_parallel_to_vector 
  (a b : ℝ × ℝ)
  (A : ℝ × ℝ)
  (h₀ : a = (6, 2))
  (h₁ : b = (-4, 1/2))
  (h₂ : A = (3, -1)) :
  let dir_vec := (a.1 + 2 * b.1, a.2 + 2 * b.2) in
  let slope := -dir_vec.2 / dir_vec.1 in
  let line := (2 : ℝ) * (Prod.fst A) + (3 : ℝ) * (Prod.snd A) - (3 : ℝ) = 0 in
  line = (2 : ℝ) * (x : ℝ) + (3 : ℝ) * (y : ℝ) - (3 : ℝ) = 0 :=
by
  sorry

end equation_of_line_passing_through_point_parallel_to_vector_l727_727876


namespace complement_of_intersection_l727_727491

theorem complement_of_intersection (U M N : Set ℕ)
  (hU : U = {1, 2, 3, 4})
  (hM : M = {1, 2, 3})
  (hN : N = {2, 3, 4}) :
  (U \ (M ∩ N)) = {1, 4} :=
by
  rw [hU, hM, hN]
  sorry

end complement_of_intersection_l727_727491


namespace root_approx_interval_l727_727694

noncomputable def f (x : ℝ) : ℝ := x^3 - 22 - x

theorem root_approx_interval :
  ∃ x : ℝ, x ∈ set.Ioo 1 2 ∧ f x = 0 :=
sorry

end root_approx_interval_l727_727694


namespace correct_option_D_l727_727321

noncomputable theory

-- Define the terms and their properties
def term_A : String := "3πxy²"
def term_B : String := "-3x²y"
def term_C : String := "x"
def term_D : String := "-(1/3)xy²"

def coeff_A : ℝ := 3 * real.pi
def coeff_B : ℝ := -3
def coeff_C : ℝ := 1
def coeff_D : ℝ := -(1/3)

def degree (term : String) : ℕ :=
  if term = term_A then 3
  else if term = term_B then 3
  else if term = term_C then 1
  else if term = term_D then 3
  else 0

-- Definition of the correctness for each option
def is_correct (term : String) (coeff : ℝ) (deg : ℕ) := 
  (term = term_A ∧ coeff = coeff_A ∧ deg = 3) ∨
  (term = term_B ∧ coeff = coeff_B ∧ deg = 3) ∨
  (term = term_C ∧ coeff = coeff_C ∧ deg = 1) ∨
  (term = term_D ∧ coeff = coeff_D ∧ deg = 3)

-- Prove that Option D is the correct statement
theorem correct_option_D : 
  is_correct term_D coeff_D 3 := 
sorry

end correct_option_D_l727_727321


namespace polynomial_degree_bound_l727_727950

theorem polynomial_degree_bound (m n k : ℕ) (P : Polynomial ℤ) 
  (hm_pos : 0 < m)
  (hn_pos : 0 < n)
  (hk_pos : 2 ≤ k)
  (hP_odd : ∀ i, P.coeff i % 2 = 1) 
  (h_div : (X - 1) ^ m ∣ P)
  (hm_bound : m ≥ 2 ^ k) :
  n ≥ 2 ^ (k + 1) - 1 := sorry

end polynomial_degree_bound_l727_727950


namespace convert_base_8_to_10_l727_727766

theorem convert_base_8_to_10 :
  let n := 4532
  let b := 8
  n = 4 * b^3 + 5 * b^2 + 3 * b^1 + 2 * b^0 → 4 * 512 + 5 * 64 + 3 * 8 + 2 * 1 = 2394 :=
by
  sorry

end convert_base_8_to_10_l727_727766


namespace winter_sales_correct_l727_727697

-- Define the given conditions as constants or hypotheses
def total_sales : ℕ := 20 -- Total annual sales in million hamburgers
def spring_sales : ℕ := 4 -- Spring sales in million hamburgers
def summer_sales : ℕ := 6 -- Summer sales in million hamburgers
def fall_percentage : ℚ := 0.3 -- Fall percentage of total sales

-- Calculate fall_sales from fall_percentage and total_sales
def fall_sales : ℕ := (fall_percentage * total_sales).toNat

-- Define a variable for winter sales
def winter_sales (total_sales spring_sales summer_sales fall_sales : ℕ) : ℕ :=
  total_sales - spring_sales - summer_sales - fall_sales

-- The theorem statement to prove winter sales given the conditions
theorem winter_sales_correct :
  winter_sales total_sales spring_sales summer_sales fall_sales = 4 := by
  sorry

end winter_sales_correct_l727_727697


namespace range_of_a_l727_727860

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + (1 - a) / 2 * x^2 - x

theorem range_of_a (a : ℝ) (h : a ≠ 1) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Ici 1 ∧ f a x₀ < a / (a - 1)) →
  a ∈ Set.Ioo (-Real.sqrt 2 - 1) (Real.sqrt 2 - 1) ∨ a ∈ Set.Ioi 1 :=
by sorry

end range_of_a_l727_727860


namespace ak_not_perfect_square_l727_727310

theorem ak_not_perfect_square (a b : ℕ → ℤ)
  (h1 : ∀ k, b k = a k + 9)
  (h2 : ∀ k, a (k + 1) = 8 * b k + 8)
  (h3 : ∃ k1 k2, a k1 = 1988 ∧ b k2 = 1988) :
  ∀ k, ¬ ∃ n, a k = n * n :=
by
  sorry

end ak_not_perfect_square_l727_727310


namespace squares_area_sum_l727_727740

theorem squares_area_sum (A B E : Point) (hEAB : Angle E A B = 90) (hBE : BE = 12)
  (hAB : AB = BE / 2) : (AB ^ 2 + AE ^ 2) = 144 :=
by
  have hAB_value : AB = 6 := by rw [hBE, hAB]; norm_num
  have hAE_square : AE ^ 2 = BE ^ 2 - AB ^ 2 := by 
    sorry -- Derive from Pythagorean theorem
  have hBE_value : BE ^ 2 = 12 ^ 2 := by norm_num
  have hAB_square : AB ^ 2 = 6 ^ 2 := by norm_num
  have hAE_square_value : AE ^ 2 = 108 := by
    rw [hBE_value, hAB_square, hAE_square]
    sorry -- Calculation
  have sum_squares : (AB ^ 2 + AE ^ 2) = 36 + 108 := by 
    rw [hAB_square, hAE_square_value]; norm_num
  exact sum_squares

end squares_area_sum_l727_727740


namespace range_of_independent_variable_l727_727283

theorem range_of_independent_variable (x : ℝ) : 
  (∃ y : ℝ, y = 2 * x / (x - 1)) ↔ x ≠ 1 :=
by sorry

end range_of_independent_variable_l727_727283


namespace gcd_of_12012_18018_l727_727795

theorem gcd_of_12012_18018 : gcd 12012 18018 = 6006 :=
by
  -- Definitions for conditions
  have h1 : 12012 = 12 * 1001 := by
    sorry
  have h2 : 18018 = 18 * 1001 := by
    sorry
  have h3 : gcd 12 18 = 6 := by
    sorry
  -- Using the conditions to prove the main statement
  rw [h1, h2]
  rw [gcd_mul_right, gcd_mul_right]
  rw [gcd_comm 12 18, h3]
  rw [mul_comm 6 1001]
  sorry

end gcd_of_12012_18018_l727_727795


namespace seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths_l727_727023

theorem seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths : 
  (7 : ℚ) * (9 + (2 / 5)) = 65 + (4 / 5) :=
by
  sorry

end seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths_l727_727023


namespace second_column_Jia_Zi_same_l727_727065

def hStems : List String := ["Jia", "Yi", "Bing", "Ding", "Wu", "Ji", "Geng", "Xin", "Ren", "Gui"]
def eBranches : List String := ["Zi", "Chou", "Yin", "Mao", "Chen", "Si", "Wu", "Wei", "Shen", "You", "Xu", "Hai"]

noncomputable def lcm_10_12 : ℕ := Nat.lcm 10 12

theorem second_column_Jia_Zi_same (n : ℕ) (h : n = 61) : (hStems ((n-1) % 10) = "Jia") ∧ (eBranches ((n-1) % 12) = "Zi") :=
by
  sorry

end second_column_Jia_Zi_same_l727_727065


namespace solve_m_n_monotonic_f_inequality_f_l727_727858

-- Definition of the function f(x)
def f (m n : ℝ) (x : ℝ) : ℝ := m * x + (1 / (n * x)) + (1 / 2)

-- Given conditions
def cond1 (m n : ℝ) : Prop := f m n 1 = 2
def cond2 (m n : ℝ) : Prop := f m n 2 = 11 / 4

-- Solutions to prove correctness
theorem solve_m_n : ∃ (m n : ℝ), cond1 m n ∧ cond2 m n :=
by
  -- The solution steps are omitted and replaced by sorry
  sorry

def f2 (x : ℝ) : ℝ := x + 1 / (2 * x) + 1 / 2

-- Monotonicity proof
theorem monotonic_f : ∀ x y : ℝ, 1 ≤ x → x < y → f2 x < f2 y :=
by
  -- The solution steps are omitted and replaced by sorry
  sorry

-- Inequality proof
theorem inequality_f (x : ℝ) : f2 (1 + 2 * x^2) > f2 (x^2 - 2 * x + 4) → x < -3 ∨ 1 < x :=
by
  -- The solution steps are omitted and replaced by sorry
  sorry

end solve_m_n_monotonic_f_inequality_f_l727_727858


namespace multiples_of_12_count_l727_727882

theorem multiples_of_12_count : 
  let multiples := { n : ℕ | 10 ≤ 12 * n ∧ 12 * n ≤ 250 }
  in multiples.card = 20 :=
by
  sorry

end multiples_of_12_count_l727_727882


namespace paint_intensity_l727_727268

theorem paint_intensity (I : ℝ) (F : ℝ) (I_initial I_new : ℝ) : 
  I_initial = 50 → I_new = 30 → F = 2 / 3 → I = 20 :=
by
  intros h1 h2 h3
  sorry

end paint_intensity_l727_727268


namespace train_speed_in_kmph_l727_727726

-- Definitions based on the conditions
def train_length : ℝ := 280 -- in meters
def time_to_pass_tree : ℝ := 28 -- in seconds

-- Conversion factor from meters/second to kilometers/hour
def mps_to_kmph : ℝ := 3.6

-- The speed of the train in kilometers per hour
theorem train_speed_in_kmph : (train_length / time_to_pass_tree) * mps_to_kmph = 36 := 
sorry

end train_speed_in_kmph_l727_727726


namespace ellipse_proof_I_slope_proof_II_perpendicular_proof_III_l727_727841

-- Defining ellipse E with given conditions
def ellipse_E (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Conditions for point F and F1
def left_focus_F1 : ℝ × ℝ := (-Real.sqrt(5), 0)
def right_focus_F2 : ℝ × ℝ := (Real.sqrt(5), 0)

-- Given points
def point_Q : ℝ × ℝ := (-2, 0)
def point_M : ℝ × ℝ := (0, 1)

-- Ellipse G definition based on (Ⅰ)
def ellipse_G (x y : ℝ) : Prop := x^2 + y^2 / 4 = 1

-- Definition of line l with slope k passing through point Q and intersecting G
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 2)

-- Midpoint of segment HK on ellipse G when intersected by line l
def midpoint_N (x1 x2 y1 y2 : ℝ) : ℝ × ℝ := ((x1 + x2) / 2, (y1 + y2) / 2)

-- Given ellipse W
def ellipse_W (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Points and perpendicularity for ellipse W
def point_P (m n : ℝ) : Prop := (ellipse_W m n) ∧ (0 < m) ∧ (0 < n)
def point_A (m n : ℝ) : ℝ × ℝ := (-m, -n)
def point_C (m : ℝ) : ℝ × ℝ := (m, 0)

-- Proof for the problems
theorem ellipse_proof_I : 
  (hE : ∀ x y : ℝ, ellipse_E x y ↔ x^2 / 9 + y^2 / 4 = 1) ∧
  (hF1 : left_focus_F1 = (-Real.sqrt(5), 0)) ∧
  (hF2 : right_focus_F2 = (Real.sqrt(5), 0)) ∧
  (hC : ∃ D : ℝ × ℝ, ellipse_E D.1 D.2 ∧
                     ∃ F : ℝ × ℝ, F = (-5 - Real.sqrt(5), 0) / 2 ∧ -- Midpoint condition
                     tangent_to_minor_axis E F D) → 
  ellipse_E = "x^2 / 9 + y^2 / 4 = 1" :=
sorry

theorem slope_proof_II (k : ℝ) :
  (∀ x y : ℝ, ellipse_G x y → 
              line_l k x y → 
              (0 ≤ k ∧ k ≤ (2:ℝ)/3) ∨ 
              k = -4 + 2 * Real.sqrt(5)) :=
sorry

theorem perpendicular_proof_III (m n : ℝ) :
  (hp : ellipse_W m n ∧ 0 < m ∧ 0 < n) ∧ 
  (ha : point_A m n) ∧ 
  (hc : point_C m) →
  ( ∃ A P B : ℝ×ℝ, PA_perp PB A P B) := 
sorry

end ellipse_proof_I_slope_proof_II_perpendicular_proof_III_l727_727841


namespace sum_of_ages_eq_19_l727_727309

theorem sum_of_ages_eq_19 :
  ∃ (a b s : ℕ), (3 * a + 5 + b = s) ∧ (6 * s^2 = 2 * a^2 + 10 * b^2) ∧ (Nat.gcd a (Nat.gcd b s) = 1 ∧ a + b + s = 19) :=
sorry

end sum_of_ages_eq_19_l727_727309


namespace impossible_lattice_path_to_origin_l727_727865

theorem impossible_lattice_path_to_origin :
  let U := {p : ℤ × ℤ | 0 ≤ p.1 ∧ p.1 ≤ 23 ∧ 0 ≤ p.2 ∧ p.2 ≤ 23} in
  ¬(∃ f : ℕ → ℤ × ℤ, (∀ n, f n ∈ U) ∧
    (∀ n m, n ≠ m → f n ≠ f m) ∧
    (∀ n, (f n).fst - (f (n + 1)).fst = 4 ∧ (f n).snd - (f (n + 1)).snd = 5 ∨
          (f n).fst - (f (n + 1)).fst = -4 ∧ (f n).snd - (f (n + 1)).snd = -5 ∨
          (f n).fst - (f (n + 1)).fst = 5 ∧ (f n).snd - (f (n + 1)).snd = 4 ∨
          (f n).fst - (f (n + 1)).fst = -5 ∧ (f n).snd - (f (n + 1)).snd = -4) ∧
    f 0 = (0, 0) ∧ ∃ N, f N = (0, 0)) :=
by {
  intro U,
  sorry
}

end impossible_lattice_path_to_origin_l727_727865


namespace problem1_problem2_problem3_l727_727849

-- Problem 1 proof outline
theorem problem1 
  (a b : ℝ) 
  (l₁ : ℝ → ℝ := λ x, x + a) 
  (l₂ : ℝ → ℝ := λ x, x + b) 
  (A B C D : ℝ × ℝ) 
  (W : ℝ × ℝ → Prop := λ ⟨x, y⟩, x^2 + y^2 = 1) 
  (h_intersect_1 : W (A.1, A.2) ∧ l₁ A.1 = A.2) 
  (h_intersect_2 : W (B.1, B.2) ∧ l₁ B.1 = B.2) 
  (h_intersect_3 : W (C.1, C.2) ∧ l₂ C.1 = C.2) 
  (h_intersect_4 : W (D.1, D.2) ∧ l₂ D.1 = D.2) 
  (h_equal_arcs : some_arc_division_def)
  : a^2 + b^2 = 2 := 
sorry

-- Problem 2 proof outline
theorem problem2 
  (A B C D : ℝ × ℝ)
  (h_l₁ : ∀ x, (A.2 ≠ B.2) → x = (2 * (A.1) - sqrt 10))
  (h_l₂ : ∀ x, (C.2 ≠ D.2) → x = (2 * (C.1) + sqrt 10))
  (W : ℝ × ℝ → Prop := λ ⟨x, y⟩, x^2 + y^2 = 4) 
  (h_intersect_1 : W (A.1, A.2))
  (h_intersect_2 : W (B.1, B.2))
  (h_intersect_3 : W (C.1, C.2))
  (h_intersect_4 : W (D.1, D.2))
  : quadrilateral_is_square A B C D :=
sorry

-- Problem 3 proof outline
theorem problem3
  (A B C D : ℝ × ℝ)
  (W : ℝ × ℝ → Prop := λ ⟨x, y⟩, x^2 / 2 + y^2 = 1)
  (h_vertices : W (A.1, A.2) ∧ W (B.1, B.2) ∧ W (C.1, C.2) ∧ W (D.1, D.2))
  : unique_inscribed_square W ∧ area_of_square = 8 / 3 :=
sorry

end problem1_problem2_problem3_l727_727849


namespace f_leq_binom_l727_727564

-- Define the function f with given conditions
def f (m n : ℕ) : ℕ := if m = 1 ∨ n = 1 then 1 else sorry

-- State the property to be proven
theorem f_leq_binom (m n : ℕ) (h2 : 2 ≤ m) (h2' : 2 ≤ n) :
  f m n ≤ Nat.choose (m + n) n := 
sorry

end f_leq_binom_l727_727564


namespace exterior_angle_DEF_l727_727258

noncomputable def interior_angle (n : ℕ) : ℝ := 180 * (n - 2) / n

def ex1_hexagon := 120 -- Interior angle of a regular hexagon
def ex1_octagon := 135 -- Interior angle of a regular octagon

theorem exterior_angle_DEF :
  ∠DEF = 105 :=
by
  have hex_angle : interior_angle 6 = 120 := by
    dsimp [interior_angle]
    norm_num
  have oct_angle : interior_angle 8 = 135 := by
    dsimp [interior_angle]
    norm_num
  -- Sum of angles around point E
  have sum_angles : 360 = 120 + 135 + ∠DEF := by
    sorry -- To be filled with the calculation logic
  sorry -- Replace this final sorry with the required conclusions using have clauses

end exterior_angle_DEF_l727_727258


namespace fraction_difference_of_squares_l727_727317

theorem fraction_difference_of_squares :
  (175^2 - 155^2) / 20 = 330 :=
by
  -- Proof goes here
  sorry

end fraction_difference_of_squares_l727_727317


namespace problem1_problem2_l727_727461

theorem problem1 (a b m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) (h4 : ab ≤ m) : m ≥ 1/4 :=
sorry

theorem problem2 (a b x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) (h4 : (1/a) + (1/b) ≥ abs(2*x - 1) - abs(x + 1)) : -2 ≤ x ∧ x ≤ 6 :=
sorry

end problem1_problem2_l727_727461


namespace total_students_l727_727521

theorem total_students (ratio_boys : ℕ) (ratio_girls : ℕ) (num_girls : ℕ) 
  (h_ratio : ratio_boys = 8) (h_ratio_girls : ratio_girls = 5) (h_num_girls : num_girls = 175) : 
  ratio_boys * (num_girls / ratio_girls) + num_girls = 455 :=
by
  sorry

end total_students_l727_727521


namespace strawberry_milk_production_probability_l727_727204

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

noncomputable def probability_of_success (p : ℚ) (successes : ℕ) : ℚ :=
  p ^ successes

noncomputable def probability_of_failure (p : ℚ) (failures : ℕ) : ℚ :=
  (1 - p) ^ failures

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  binomial_coefficient n k * probability_of_success p k * probability_of_failure p (n - k)

theorem strawberry_milk_production_probability : 
  binomial_probability 7 5 (3 / 5 : ℚ) = (20412 / 78125 : ℚ) :=
by 
  sorry

end strawberry_milk_production_probability_l727_727204


namespace no_triangular_sides_of_specific_a_b_l727_727743

theorem no_triangular_sides_of_specific_a_b (a b c : ℕ) (h1 : a = 10^100 + 1002) (h2 : b = 1001) (h3 : ∃ n : ℕ, c = n^2) : ¬ (a + b > c ∧ a + c > b ∧ b + c > a) :=
by sorry

end no_triangular_sides_of_specific_a_b_l727_727743


namespace multiplication_difference_is_1242_l727_727359

theorem multiplication_difference_is_1242 (a b c : ℕ) (h1 : a = 138) (h2 : b = 43) (h3 : c = 34) :
  a * b - a * c = 1242 :=
by
  sorry

end multiplication_difference_is_1242_l727_727359


namespace range_of_a_l727_727095

theorem range_of_a (a : ℝ) :
  let f := λ x, x^3 + a * x^2 + (a + 6) * x + 1
  ∃ x₁ x₂ : ℝ, 
    x₁ ≠ x₂ ∧ 
    f' x₁ = 0 ∧ 
    f' x₂ = 0 ↔ (a < -3 ∨ a > 6) :=
by
  let f := λ x, x^3 + a * x^2 + (a + 6) * x + 1
  let f' := λ x, 3 * x^2 + 2 * a * x + (a + 6)
  sorry

end range_of_a_l727_727095


namespace ab_bm_ratio_l727_727921

theorem ab_bm_ratio (A B C M : Point) 
  (h_median : is_median B M A C) 
  (h_angle_abm : ∠ A B M = 40) 
  (h_angle_cbm : ∠ C B M = 70) : AB / BM = 2 := 
sorry

end ab_bm_ratio_l727_727921


namespace multiplication_with_mixed_number_l727_727018

-- Define mixed numbers as rational numbers for proper calculation
def mixed_to_rational (whole : ℕ) (num : ℕ) (den : ℕ) : ℚ :=
  whole + num / den

-- 7 * (9 + 2/5)
def lhs : ℚ := 7 * mixed_to_rational 9 2 5

-- 65 + 4/5
def rhs : ℚ := mixed_to_rational 65 4 5

theorem multiplication_with_mixed_number : lhs = rhs := by
  sorry

end multiplication_with_mixed_number_l727_727018


namespace problem_1_problem_2_problem_3_l727_727789

-- Problem 1
theorem problem_1 (f : ℝ → ℝ) (x : ℝ) (h : f x = x^2 + 2 * x) : f (2 * x + 1) = 4 * x ^ 2 + 8 * x + 3 :=
sorry

-- Problem 2
theorem problem_2 (f : ℝ → ℝ) (x : ℝ) (h : ∀ x : ℝ, x ≥ -1 → f (real.sqrt x - 1) = x + 2 * real.sqrt x) : ∀ x, f x = x^2 + 4 * x + 3 :=
sorry

-- Problem 3
theorem problem_3 (f : ℝ → ℝ) (h : ∀ x : ℝ, f x - 2 * f (1 / x) = 3 * x + 2) : ∀ x, f x = - x - 2 / x - 2 :=
sorry

end problem_1_problem_2_problem_3_l727_727789


namespace chocolates_in_box_l727_727327

theorem chocolates_in_box (chocolates_with_nuts_chances : ℕ) (chocolates_without_nuts_chances : ℕ)
  (eaten_with_nuts_chances : ℝ) (eaten_without_nuts_chances : ℝ) (remaining_chocolates : ℕ)
  (half_chocolates : chocolates_with_nuts_chances = chocolates_without_nuts_chances)
  (eaten_with_nuts_ratio : eaten_with_nuts_chances = 0.80)
  (eaten_without_nuts_ratio : eaten_without_nuts_chances = 0.50)
  (remaining_chocolates_eq : remaining_chocolates = 28) :
  ∃ (total_chocolates : ℕ), total_chocolates = 80 :=
by
  use 80
  sorry

end chocolates_in_box_l727_727327


namespace big_crash_occurrence_l727_727064

open Nat

theorem big_crash_occurrence :
  (∀ t, t % 10 = 0 → ∃ n, t = 10 * n) →
  (∃ m, 240 = 60 * m) →
  (∀ t, 240 % 10 = 0 ∧ ∃ n, 240 = 10 * n) →
  (∀ x, 36 = 12 + 24) →
  (∃ t, t * 12 = 240) →
  (∃ t, t = 20) 
:= by sorry

end big_crash_occurrence_l727_727064


namespace number_of_distinct_models_of_cubes_l727_727296

theorem number_of_distinct_models_of_cubes : 
  let num_rotations := 24
  ∃! (num_ways_to_arrange_vertices : ℕ), 
    num_ways_to_arrange_vertices = Nat.factorial 8 / num_rotations 
    ∧ num_ways_to_arrange_vertices = 1680 :=
begin
  sorry,
end

end number_of_distinct_models_of_cubes_l727_727296


namespace sqrt_720_l727_727992

theorem sqrt_720 : sqrt (720) = 12 * sqrt (5) :=
sorry

end sqrt_720_l727_727992


namespace proof_valid_x_values_l727_727068

noncomputable def valid_x_values (x : ℝ) : Prop :=
  (x^2 + 2*x^3 - 3*x^4) / (x + 2*x^2 - 3*x^3) ≤ 1

theorem proof_valid_x_values :
  {x : ℝ | valid_x_values x} = {x : ℝ | (x < -1) ∨ (x > -1 ∧ x < 0) ∨ (x > 0 ∧ x < 1)} :=
by {
  sorry
}

end proof_valid_x_values_l727_727068


namespace intersecting_areas_equal_remaining_area_l727_727186

noncomputable def circle_radius (R : ℝ) := R
noncomputable def smaller_circle_radius (R : ℝ) := R / 2
noncomputable def original_circle_area (R : ℝ) := π * R^2
noncomputable def smaller_circle_area (R : ℝ) := π * (R / 2)^2

theorem intersecting_areas_equal_remaining_area (R : ℝ) :
  let original_area := original_circle_area R
  let four_smaller_areas := 4 * smaller_circle_area R
  let intersecting_area := (π * R^2 * (π - 2)) / 4 in
  intersecting_area = original_area - four_smaller_areas + intersecting_area :=
by
  sorry

end intersecting_areas_equal_remaining_area_l727_727186


namespace find_second_divisor_l727_727704

theorem find_second_divisor
  (N D : ℕ)
  (h1 : ∃ k : ℕ, N = 35 * k + 25)
  (h2 : ∃ m : ℕ, N = D * m + 4) :
  D = 21 :=
sorry

end find_second_divisor_l727_727704


namespace calculate_a10_l727_727836

-- Define the sequence and the conditions
def sequence (a : ℕ → ℤ) : Prop :=
  ∀ p q : ℕ, 0 < p → 0 < q → a (p + q) = a p + a q

-- Given conditions
def a2_value (a : ℕ → ℤ) : Prop := a 2 = -6

-- The main theorem to prove
theorem calculate_a10 (a : ℕ → ℤ) (seq : sequence a) (a2 : a2_value a) : a 10 = -30 :=
  sorry

end calculate_a10_l727_727836


namespace rearrangements_with_vowels_first_l727_727499
open Nat

theorem rearrangements_with_vowels_first
  (word : String := "COMPLEX")
  (vowels : Set Char := {'O', 'E'})
  (consonants : Set Char := {'C', 'M', 'P', 'L', 'X'}) :
  (∃ n, n = 5! * 2!) ∧ (5! * 2! = 240) :=
by
  have h1 : 5! = 120 := by rfl
  have h2 : 2! = 2 := by rfl
  use 5! * 2!
  rw [h1, h2]
  exact Nat.mul_eq_mul_right 120 2 240 rfl

end rearrangements_with_vowels_first_l727_727499


namespace a2b2c2d2_is_square_l727_727200

structure Point where
  x : ℝ
  y : ℝ

structure Square (a b c d : Point) : Prop where
  side_length : ℝ
  sides_eq : dist a b = side_length ∧ 
             dist b c = side_length ∧ 
             dist c d = side_length ∧ 
             dist d a = side_length
  right_angles : 
    (∠ a b c = π / 2) ∧
    (∠ b c d = π / 2) ∧
    (∠ c d a = π / 2) ∧
    (∠ d a b = π / 2)

noncomputable def dist (p1 p2 : Point) : ℝ := 
  ( (p1.x - p2.x)^2 + (p1.y - p2.y)^2 )^.5

noncomputable def angle (p1 p2 p3 : Point) : ℝ := -- Assume this definition to compute angle between three points

theorem a2b2c2d2_is_square (A B C D O A2 B2 C2 D2: Point)
    (h1: Square A B C D)
    (h2: dist A O = dist O A2 ∧ Square O A A2 A2)
    (h3: dist B O = dist O B2 ∧ Square O B B2 B2)
    (h4: dist C O = dist O C2 ∧ Square O C C2 C2)
    (h5: dist D O = dist O D2 ∧ Square O D D2 D2)
    (h6: ∠ A O B = π / 2 ∨ - (∠ A O B = π / 2))
    (h7: ∠ A O C = π / 2 ∨ - (∠ A O C = π / 2)) :
    Square A2 B2 C2 D2 :=
by sorry

end a2b2c2d2_is_square_l727_727200


namespace trig_problem_l727_727077

theorem trig_problem (α : ℝ) (h : Real.cot α = -(1/5)) :
  2 - 13 * Real.cos (2 * α) + Real.asin (2 * α) = 57 / 5 :=
sorry

end trig_problem_l727_727077


namespace coeff_x4y_in_expansion_l727_727914

theorem coeff_x4y_in_expansion :
  let expr := (x^2 - 2*y - 3)^5
  in coeff_of_term expr (x^4 * y) = -540 :=
by
  sorry

end coeff_x4y_in_expansion_l727_727914


namespace minimize_S_l727_727531

-- Definition of the line equation given a point and slope
def line (a : ℝ) : ℝ → ℝ := λ x, a * x + (2 - a)

-- Definition of the parabola
def parabola : ℝ → ℝ := λ x, x^2

-- Function to calculate the area between the line and the parabola
noncomputable def S (a : ℝ) : ℝ :=
  let x1 := (a - Real.sqrt (a^2 - 4 * a + 8)) / 2 in
  let x2 := (a + Real.sqrt (a^2 - 4 * a + 8)) / 2 in
  (∫ x in x1..x2, (parabola x - line a x))

-- Statement of the problem
theorem minimize_S : ∀ (a : ℝ), 0 ≤ a ∧ a ≤ 6 → S a ≥ 0 → (S a = S 2) :=
by
  sorry

end minimize_S_l727_727531


namespace guise_hot_dogs_by_wednesday_l727_727145

theorem guise_hot_dogs_by_wednesday :
  let monday_dogs := 10 in
  let tuesday_dogs := monday_dogs + 2 in
  let wednesday_dogs := tuesday_dogs + 2 in
  (monday_dogs + tuesday_dogs + wednesday_dogs) = 36 :=
by
  sorry

end guise_hot_dogs_by_wednesday_l727_727145


namespace fewer_seats_on_right_side_l727_727897

-- Definitions based on the conditions
def left_seats := 15
def seats_per_seat := 3
def back_seat_capacity := 8
def total_capacity := 89

-- Statement to prove the problem
theorem fewer_seats_on_right_side : left_seats - (total_capacity - back_seat_capacity - (left_seats * seats_per_seat)) / seats_per_seat = 3 := 
by
  -- proof steps go here
  sorry

end fewer_seats_on_right_side_l727_727897


namespace people_dislike_both_l727_727972

variable (total_people : ℕ)
variable (percent_dislike_tv : ℝ)
variable (percent_dislike_both : ℝ)

theorem people_dislike_both :
  total_people = 1500 →
  percent_dislike_tv = 0.4 →
  percent_dislike_both = 0.15 →
  let number_dislike_tv := percent_dislike_tv * total_people in
  let number_dislike_both := percent_dislike_both * number_dislike_tv in
  number_dislike_both = 90 :=
by
  intros h1 h2 h3
  let number_dislike_tv := percent_dislike_tv * total_people
  let number_dislike_both := percent_dislike_both * number_dislike_tv
  have h4 : number_dislike_tv = 0.4 * 1500, from by { rw [h1, h2], ring }
  have h5 : number_dislike_both = 0.15 * 600, from by { rw [h4, h3], ring }
  rw [h5]
  norm_num
  sorry

end people_dislike_both_l727_727972


namespace find_a_l727_727230

noncomputable def is_pure_imaginary (z : ℂ) : Prop :=
z.re = 0 ∧ z.im ≠ 0

theorem find_a (a : ℝ) (h : is_pure_imaginary ((1 : ℂ) + Complex.I * (1 : ℂ)) * (1 + a * Complex.I)) : a = 1 :=
sorry

end find_a_l727_727230


namespace angle_YRS_45_l727_727912

/-- 
In the diagram, square \(WXYZ\) has sides of length \(6\), and \(\triangle WXF\) is equilateral. 
Line segments \(XF\) and \(WZ\) intersect at \(R\). Point \(S\) is on \(YZ\) such that \(RS\) is perpendicular to \(YZ\). 
Determine the measure of angle \(\angle YRS\).
-/
theorem angle_YRS_45
  (W X Y Z F R S : Type)
  [IsSquare W X Y Z]
  (hWY: Distance W Y = 6)
  (hYZ: Distance Y Z = 6)
  [IsEquilateralTriangle W X F]
  (hRXF: SegmentIntersectionPoint R X F)
  (hWZ : Segment W Z)
  (hS_on_YZ : PointOnSegment S Y Z)
  (hRS_perp_YZ: Perpendicular RS YZ) 
  : MeasureAngle Y R S = 45 :=
  sorry

end angle_YRS_45_l727_727912


namespace integral_evaluation_l727_727057

noncomputable def integral_value : ℝ :=
  ∫ x in 0..(Real.pi / 4), (Real.cos (2 * x) / (Real.cos x + Real.sin x))

theorem integral_evaluation :
  integral_value = Real.sqrt 2 - 1 :=
by
  sorry

end integral_evaluation_l727_727057


namespace tetrahedron_CD_length_l727_727277

theorem tetrahedron_CD_length {a b c d e f : ℕ} (h1 : {a, b, c, d, e, f} = {7, 13, 18, 27, 36, 41}) 
  (h2 : a = 41 ∨ b = 41 ∨ c = 41 ∨ d = 41 ∨ e = 41 ∨ f = 41) :
  (a = 41 → (b = 13 ∨ c = 13 ∨ d = 13 ∨ e = 13 ∨ f = 13)) ∧
  (b = 41 → (a = 13 ∨ c = 13 ∨ d = 13 ∨ e = 13 ∨ f = 13)) ∧
  (c = 41 → (a = 13 ∨ b = 13 ∨ d = 13 ∨ e = 13 ∨ f = 13)) ∧
  (d = 41 → (a = 13 ∨ b = 13 ∨ c = 13 ∨ e = 13 ∨ f = 13)) ∧
  (e = 41 → (a = 13 ∨ b = 13 ∨ c = 13 ∨ d = 13 ∨ f = 13)) ∧
  (f = 41 → (a = 13 ∨ b = 13 ∨ c = 13 ∨ d = 13 ∨ e = 13)) :=
begin
  sorry
end

end tetrahedron_CD_length_l727_727277


namespace analytical_expression_f_range_of_a_l727_727851

-- Definition of the odd function f(x) on ℝ
def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 + 2 * x else
  if x = 0 then 0 else
  -x^2 + 2 * x

-- Part 1: Proving the analytical expression of f(x)
theorem analytical_expression_f :
  ∀ x : ℝ, f(x) = 
  if x > 0 then x^2 + 2 * x else
  if x = 0 then 0 else
  -x^2 + 2 * x := by
  sorry

-- Part 2: Proving the range of a
theorem range_of_a (a : ℝ) :
  f(2^(a + 1)) - f(4^a) ≤ 0 ↔ a ≥ 1 := by
  sorry

end analytical_expression_f_range_of_a_l727_727851


namespace peter_situps_eq_24_l727_727441

noncomputable def situps_peter_did : ℕ :=
  let ratio_peter_greg := 3 / 4
  let situps_greg := 32
  let situps_peter := (3 * situps_greg) / 4
  situps_peter

theorem peter_situps_eq_24 : situps_peter_did = 24 := 
by 
  let h := situps_peter_did
  show h = 24
  sorry

end peter_situps_eq_24_l727_727441


namespace four_digit_numbers_not_multiple_of_3_or_4_l727_727152

theorem four_digit_numbers_not_multiple_of_3_or_4 :
  (finset.Icc 1000 9999).card - 
  ((finset.filter (λ x, x % 3 = 0) (finset.Icc 1000 9999)).card + 
  (finset.filter (λ x, x % 4 = 0) (finset.Icc 1000 9999)).card - 
  (finset.filter (λ x, x % 12 = 0) (finset.Icc 1000 9999)).card) = 4500 :=
by
  sorry

end four_digit_numbers_not_multiple_of_3_or_4_l727_727152


namespace chocolate_count_l727_727329

theorem chocolate_count (C : ℝ) 
  (half_with_nuts : C/2) 
  (half_without_nuts : C/2) 
  (eaten_with_nuts : 0.8 * (C/2)) 
  (eaten_without_nuts : 0.5 * (C/2)) 
  (remaining_chocolates : 0.1 * C + 0.25 * C = 28) : 
  C = 80 :=
by
  sorry

end chocolate_count_l727_727329


namespace equalize_marbles_l727_727337

theorem equalize_marbles (x : ℝ) (h : x > 0) : 
    let m := 0.05 * x
    let percent_of_jarY := (m / (1.10 * x)) * 100
  in percent_of_jarY ≈ 4.545 :=
by
  sorry

end equalize_marbles_l727_727337


namespace gcd_of_12012_18018_l727_727796

theorem gcd_of_12012_18018 : gcd 12012 18018 = 6006 :=
by
  -- Definitions for conditions
  have h1 : 12012 = 12 * 1001 := by
    sorry
  have h2 : 18018 = 18 * 1001 := by
    sorry
  have h3 : gcd 12 18 = 6 := by
    sorry
  -- Using the conditions to prove the main statement
  rw [h1, h2]
  rw [gcd_mul_right, gcd_mul_right]
  rw [gcd_comm 12 18, h3]
  rw [mul_comm 6 1001]
  sorry

end gcd_of_12012_18018_l727_727796


namespace trajectory_equation_l727_727102

-- Define the ellipse E
def ellipse (x y : ℝ) : Prop :=
  (x^2 / 4) + y^2 = 1

-- Define point P on the ellipse
variable (P : ℝ × ℝ) (hP : ellipse P.1 P.2)

-- Define the foci F1 and F2
def F1 : ℝ × ℝ := (-Real.sqrt 3, 0)
def F2 : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define the point Q
variable (Q : ℝ × ℝ)
def satisfies_trajectory (Q P : ℝ × ℝ) : Prop :=
  Q = ((F1.1 + F2.1) - 2 * P.1, (F1.2 + F2.2) - 2 * P.2)

theorem trajectory_equation (P : ℝ × ℝ) (Q : ℝ × ℝ) (hP : ellipse P.1 P.2) (hQ : satisfies_trajectory Q P) :
  (Q.1^2 / 16) + (Q.2^2 / 4) = 1 := sorry

end trajectory_equation_l727_727102


namespace max_omega_l727_727481

noncomputable def f (ω φ x : ℝ) := 2 * Real.sin (ω * x + φ)

theorem max_omega (ω φ : ℝ) (k k' : ℤ) (hω_pos : ω > 0) (hφ1 : 0 < φ)
  (hφ2 : φ < Real.pi / 2) (h1 : f ω φ (-Real.pi / 4) = 0)
  (h2 : ∀ x, f ω φ (Real.pi / 4 - x) = f ω φ (Real.pi / 4 + x))
  (h3 : ∀ x, x ∈ Set.Ioo (Real.pi / 18) (2 * Real.pi / 9) →
    Monotone (f ω φ)) :
  ω = 5 :=
sorry

end max_omega_l727_727481


namespace average_speed_train_l727_727727

theorem average_speed_train (x : ℝ) (h1 : x ≠ 0) :
  let t1 := x / 40
  let t2 := 2 * x / 20
  let t3 := 3 * x / 60
  let total_time := t1 + t2 + t3
  let total_distance := 6 * x
  let average_speed := total_distance / total_time
  average_speed = 240 / 7 := by
  sorry

end average_speed_train_l727_727727


namespace ott_fraction_of_total_money_l727_727583

-- Definitions for the conditions
def Moe_initial_money (x : ℕ) : ℕ := 3 * x
def Loki_initial_money (x : ℕ) : ℕ := 5 * x
def Nick_initial_money (x : ℕ) : ℕ := 4 * x
def Total_initial_money (x : ℕ) : ℕ := Moe_initial_money x + Loki_initial_money x + Nick_initial_money x
def Ott_received_money (x : ℕ) : ℕ := 3 * x

-- Making the statement we want to prove
theorem ott_fraction_of_total_money (x : ℕ) : 
  (Ott_received_money x) / (Total_initial_money x) = 1 / 4 := by
  sorry

end ott_fraction_of_total_money_l727_727583


namespace race_distance_l727_727190

variable (vA vB D : ℝ)
variable (h1 : vA / vB = 1 / 2)
variable (h2 : ∃ D, ∀ vA vB, vB = 2 * vA → (2 * (D - 300) = D - 100))

theorem race_distance (hstart : 300 > 0) (hwin : 100 > 0) : D = 500 :=
  by
  -- A starts with a head start of 300 meters and wins by 100 meters
  have vB_eq : vA / vB = 1 / 2 := h1
  -- Total distance is D
  let D_run_A := D - 300
  let D_run_B := D - 100
  -- B is twice as fast as A
  have speed_condition : vB = 2 * vA := by
    field_simp [vB_eq]
    norm_num
  -- Given speed conditions and distances, set up the race equation 
  exact Eq.symm (calc
    2 * (D - 300) = D - 100 : by 
                      simp [speed_condition]
                      ring
                      -- Now solve for D
                      )

end race_distance_l727_727190


namespace problem1_problem2_l727_727044

noncomputable def expression1 : ℝ := 
  real.sqrt 8 + | real.sqrt (2⁻¹) | - real.pi^0 + (1 / 2)⁻¹

noncomputable def expression2 : ℝ := 
  (2 * real.sqrt 5 - 2 * real.sqrt 3) * (real.sqrt 12 + real.sqrt 20)

theorem problem1 : expression1 = (5 * real.sqrt 2) / 2 + 1 := 
by
  sorry

theorem problem2 : expression2 = 8 := 
by
  sorry

end problem1_problem2_l727_727044


namespace gcd_of_12012_and_18018_l727_727798

theorem gcd_of_12012_and_18018 : Int.gcd 12012 18018 = 6006 := 
by
  -- Here we are assuming the factorization given in the conditions
  have h₁ : 12012 = 12 * 1001 := sorry
  have h₂ : 18018 = 18 * 1001 := sorry
  have gcd_12_18 : Int.gcd 12 18 = 6 := sorry
  -- This sorry will be replaced by the actual proof involving the above conditions to conclude the stated theorem
  sorry

end gcd_of_12012_and_18018_l727_727798


namespace seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths_l727_727024

theorem seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths : 
  (7 : ℚ) * (9 + (2 / 5)) = 65 + (4 / 5) :=
by
  sorry

end seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths_l727_727024


namespace find_100th_positive_term_l727_727773

noncomputable def b (n : ℕ) : ℝ := ∑ k in finset.range (n + 1), Real.cos k

theorem find_100th_positive_term :
  ∃ n : ℕ, n = 628 ∧ (∃ count : ℕ, count = 100 ∧ ∀ i < count, b (628 + i) > 0) := by
  sorry

end find_100th_positive_term_l727_727773


namespace linear_regression_decrease_l727_727101

theorem linear_regression_decrease (x : ℝ) (y : ℝ) (h : y = 2 - 1.5 * x) : 
  y = 2 - 1.5 * (x + 1) -> (y - (2 - 1.5 * (x +1))) = -1.5 :=
by
  sorry

end linear_regression_decrease_l727_727101


namespace triangle_coordinates_l727_727382

theorem triangle_coordinates
  (A B C : ℝ × ℝ)
  (on_parabola : ∀ v ∈ [B, C], v.2 = v.1 ^ 2)
  (A_origin : A = (0, 0))
  (not_parallel : (B.1 ≠ C.1))
  (right_angle_at_A : A.1 = 0 ∧ A.2 = 0)
  (area_ABC : 36 = (1 / 2) * abs(B.1 - C.1) * abs(B.2 - 0))
  (equidistant_from_y_axis : B.1 = -C.1):
  B = (-real.cbrt 36, 36) ∧ C = (real.cbrt 36, 36) :=
by
  sorry

end triangle_coordinates_l727_727382


namespace range_of_a_l727_727222

-- Define the piecewise function
def f (x : ℝ) : ℝ :=
  if x ≥ 0 then log (x + 1) - 1 / (1 + x^2)
  else log (-x + 1) - 1 / (1 + x^2)

-- Define the statement
theorem range_of_a (a : ℝ) : f (a - 2) < f (4 - a^2) → a > 2 ∨ a < -3 ∨ (-1 < a ∧ a < 2) :=
by
  sorry

end range_of_a_l727_727222


namespace member_pays_48_percent_of_SRP_l727_727353

theorem member_pays_48_percent_of_SRP
  (P : ℝ)
  (h₀ : P > 0)
  (basic_discount : ℝ := 0.40)
  (additional_discount : ℝ := 0.20) :
  ((1 - additional_discount) * (1 - basic_discount) * P) / P * 100 = 48 := by
  sorry

end member_pays_48_percent_of_SRP_l727_727353


namespace mul_mixed_number_l727_727009

theorem mul_mixed_number (a b : ℝ) (c : ℚ) (h : c = 9 + 2/5) : 
  7 * c = (65 + 4/5 : ℝ) :=
by 
  simp [h, mul_add, mul_div_cancel_left, rat.cast_add, rat.cast_mul, rat.cast_one]
  sorry

end mul_mixed_number_l727_727009


namespace probability_valid_p_l727_727500

theorem probability_valid_p (p q : ℤ) (h : 1 ≤ p ∧ p ≤ 20) :
  (pq - 6p - 3q = 3) → ∃ p ∈ {4, 6, 10},
  (3 / 20 : ℚ) := 
by 
  sorry

end probability_valid_p_l727_727500


namespace geometric_series_correct_statements_l727_727872

noncomputable def geometric_series_sum (a r : ℝ) (hr : r ≠ 1) : ℝ :=
  a / (1 - r)

theorem geometric_series_correct_statements :
  let s := geometric_series_sum 3 (1/3) (by norm_num)
  in (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs((1/3)^n - 0) < ε) ∧
     (∀ ε > 0, abs(s - 4.5) < ε) ∧
     (∃ l : ℝ, tendsto (λ n, ∑ i in finset.range n, (3 / (3 ^ i))) at_top (𝓝 l)) :=
by {
  let s := geometric_series_sum 3 (1/3) (by norm_num),
  sorry
}

end geometric_series_correct_statements_l727_727872


namespace exists_real_number_x_l727_727232

theorem exists_real_number_x (a : Fin 1998 → ℕ)
  (h1 : ∀ i j : ℕ, 1 ≤ i → i ≤ 1997 → 1 ≤ j → j ≤ 1997 → i + j ≤ 1997 → a ⟨i, Nat.le_of_succ_le_succ i.2⟩ + a ⟨j, Nat.le_of_succ_le_succ j.2⟩ ≤ a ⟨i + j, Nat.le_of_succ_le_succ (Nat.add_le_add i.2 j.2)⟩)
  (h2 : ∀ i j : ℕ, 1 ≤ i → i ≤ 1997 → 1 ≤ j → j ≤ 1997 → i + j ≤ 1997 → a ⟨i + j, Nat.le_of_succ_le_succ (Nat.add_le_add i.2 j.2)⟩ ≤ a ⟨i, Nat.le_of_succ_le_succ i.2⟩ + a ⟨j, Nat.le_of_succ_le_succ j.2⟩ + 1) :
  ∃ x : ℝ, ∀ n : ℕ, 1 ≤ n → n ≤ 1997 → a ⟨n, Nat.le_of_succ_le_succ n.2⟩ = ⌊n * x⌋ := 
sorry

end exists_real_number_x_l727_727232


namespace sqrt_720_eq_12_sqrt_5_l727_727998

theorem sqrt_720_eq_12_sqrt_5 : sqrt 720 = 12 * sqrt 5 :=
by
  sorry

end sqrt_720_eq_12_sqrt_5_l727_727998


namespace four_digit_numbers_not_multiples_of_3_or_4_l727_727155

theorem four_digit_numbers_not_multiples_of_3_or_4 : 
  {x : ℕ | 1000 ≤ x ∧ x ≤ 9999 ∧ ¬ (x % 3 = 0 ∨ x % 4 = 0)}.card = 4500 := 
begin
  sorry
end

end four_digit_numbers_not_multiples_of_3_or_4_l727_727155


namespace excentre_distance_l727_727159

-- Defining the structure and properties of the acute triangle and distances to excentres
variable {A B C I_A I_B I_C : Type} -- vertices and corresponding excentres

-- Main theorem statement
theorem excentre_distance {a b c : ℝ} (h_acute : is_acute_triangle a b c) :
 (∀A I_A, distance A I_A < b + c) ∧ (∀B I_B, distance B I_B < a + c) ∧ (∀C I_C, distance C I_C < a + b) := 
  sorry

end excentre_distance_l727_727159


namespace HA_appears_at_least_once_l727_727158

-- Define the set of letters to be arranged
def letters : List Char := ['A', 'A', 'A', 'H', 'H']

-- Define a function to count the number of ways to arrange letters such that "HA" appears at least once
def countHA(A : List Char) : Nat := sorry

-- The proof problem to establish that there are 9 such arrangements
theorem HA_appears_at_least_once : countHA letters = 9 :=
sorry

end HA_appears_at_least_once_l727_727158


namespace chocolate_count_l727_727328

theorem chocolate_count (C : ℝ) 
  (half_with_nuts : C/2) 
  (half_without_nuts : C/2) 
  (eaten_with_nuts : 0.8 * (C/2)) 
  (eaten_without_nuts : 0.5 * (C/2)) 
  (remaining_chocolates : 0.1 * C + 0.25 * C = 28) : 
  C = 80 :=
by
  sorry

end chocolate_count_l727_727328


namespace perimeter_of_figure_l727_727983

-- Define the conditions given in the problem
structure RectangleFigure where
  bottomRow : ℕ -- number of unit squares in the bottom row
  middleUnitSquare : Bool -- presence of the middle unit square
  sideRectangles : ℕ -- number of 1x2 rectangles on either side of the middle unit square
  bottomRow = 3
  middleUnitSquare = true
  sideRectangles = 2

-- Define a theorem to prove the perimeter of the figure
theorem perimeter_of_figure (fig : RectangleFigure) : 
  (fig.bottomRow = 3) → 
  (fig.middleUnitSquare = true) → 
  (fig.sideRectangles = 2) → 
  (calculate_perimeter fig = 13) := 
by
  sorry

end perimeter_of_figure_l727_727983


namespace remainder_of_squares_mod_8_l727_727569

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ a 2 = 2 ∧ ∀ n : ℕ, 1 ≤ n → a (n + 2) = (a n + a (n + 1)) % 100

theorem remainder_of_squares_mod_8 (a : ℕ → ℕ) (h : sequence a) :
  (∑ i in finset.range 2007, (a (i + 1))^2) % 8 = 1 :=
sorry

end remainder_of_squares_mod_8_l727_727569


namespace new_galleries_receive_two_pictures_l727_727375

theorem new_galleries_receive_two_pictures :
  ∀ (total_pencils : ℕ) (orig_gallery_pictures : ℕ) (new_galleries : ℕ) 
    (pencils_per_picture : ℕ) (signature_pencils_per_gallery : ℕ),
    (total_pencils = 88) →
    (orig_gallery_pictures = 9) →
    (new_galleries = 5) →
    (pencils_per_picture = 4) →
    (signature_pencils_per_gallery = 2) →
    let orig_gallery_signature_pencils := signature_pencils_per_gallery in
    let total_galleries := 1 + new_galleries in
    let total_signature_pencils := total_galleries * signature_pencils_per_gallery in
    let total_drawing_pencils := total_pencils - total_signature_pencils in
    let orig_gallery_drawing_pencils := orig_gallery_pictures * pencils_per_picture in
    let new_galleries_drawing_pencils := total_drawing_pencils - orig_gallery_drawing_pencils in
    let total_new_gallery_pictures := new_galleries_drawing_pencils / pencils_per_picture in
    let pictures_per_new_gallery := total_new_gallery_pictures / new_galleries in
    pictures_per_new_gallery = 2 :=
begin
  intros total_pencils orig_gallery_pictures new_galleries pencils_per_picture signature_pencils_per_gallery
          H_total_pencils H_orig_gallery_pictures H_new_galleries H_pencils_per_picture H_signature_pencils_per_gallery,

  let orig_gallery_signature_pencils := signature_pencils_per_gallery,
  let total_galleries := 1 + new_galleries,
  let total_signature_pencils := total_galleries * signature_pencils_per_gallery,
  let total_drawing_pencils := total_pencils - total_signature_pencils,
  let orig_gallery_drawing_pencils := orig_gallery_pictures * pencils_per_picture,
  let new_galleries_drawing_pencils := total_drawing_pencils - orig_gallery_drawing_pencils,
  let total_new_gallery_pictures := new_galleries_drawing_pencils / pencils_per_picture,
  let pictures_per_new_gallery := total_new_gallery_pictures / new_galleries,

  exact sorry
end

end new_galleries_receive_two_pictures_l727_727375


namespace count_positive_multiples_perfect_squares_l727_727157

theorem count_positive_multiples_perfect_squares :
  {n : ℕ | n < 3000 ∧ ∃ k : ℕ, n = 60 * k^2}.card = 7 :=
by
  sorry

end count_positive_multiples_perfect_squares_l727_727157


namespace area_enclosed_by_eq_l727_727053

noncomputable def abs_rhombus_area (x y : ℝ) : ℝ := (|4 * x| + |3 * y|)

theorem area_enclosed_by_eq : 
  (∀ (x y : ℝ), abs_rhombus_area x y = 12 → False) → 
  let d1 := 2 * (12 / 4) in
  let d2 := 2 * (12 / 3) in 
  area = (1 / 2) * d1 * d2 :=
by
  let d1 := 2 * (12 / 4)
  let d2 := 2 * (12 / 3)
  have area_eq : area = 1 / 2 * d1 * d2
  sorry

end area_enclosed_by_eq_l727_727053


namespace function_increasing_on_interval_l727_727131

def f (x : ℝ) : ℝ := x - Real.sin x

theorem function_increasing_on_interval
  (x1 x2 : ℝ) 
  (h1 : x1 ∈ Icc (-Real.pi / 2) (Real.pi / 2)) 
  (h2 : x2 ∈ Icc (-Real.pi / 2) (Real.pi / 2)) 
  (h3 : x1 < x2) : 
  f x1 < f x2 :=
sorry

end function_increasing_on_interval_l727_727131


namespace original_chocolates_l727_727332

theorem original_chocolates (N : ℕ) 
  (H_nuts : N / 2)
  (H_eaten_nuts : 0.8 * (N / 2) = 0.4 * N)
  (H_without_nuts : N / 2)
  (H_eaten_without : 0.5 * (N / 2) = 0.25 * N)
  (H_left : N - (0.4 * N + 0.25 * N) = 28) : 
  N = 80 :=
sorry

end original_chocolates_l727_727332


namespace triangle_angle_solution_l727_727920

theorem triangle_angle_solution :
  let (a, b, c) := (2, Real.sqrt 2, Real.sqrt 3 + 1) in
  ∃ A B C, 
    A = Real.pi / 4 ∧ 
    B = Real.pi / 6 ∧ 
    C = 7 * Real.pi / 12 ∧ 
    A + B + C = Real.pi ∧ 
    ∀ (a b c : ℝ), a = 2 → b = Real.sqrt 2 → c = Real.sqrt 3 + 1 → 
    cos A = (b^2 + c^2 - a^2) / (2 * b * c) ∧ 
    sin B = b / c * sin A ∧ 
    cos C = cos (Real.pi - (A + B)) :=
by
  sorry

end triangle_angle_solution_l727_727920


namespace units_digit_of_M_l727_727306

-- Define the sequence function
def sequence_step (n : ℕ) : ℕ :=
  n - (Nat.sqrt n)^2

-- Define the sequence length function
def sequence_length (m : ℕ) : ℕ :=
  if m = 1 then 1 else 1 + sequence_length (sequence_step m)

-- The statement to be proved
theorem units_digit_of_M (M : ℕ) :
  (sequence_length M = 6 ∧ ∃ m, sequence_length m = 6 ∧ m <= M) →
  M % 10 = 1 :=
sorry

end units_digit_of_M_l727_727306


namespace expenditure_of_income_l727_727652

def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

def regression_line (income expenditure : List ℝ) (x : ℝ) : ℝ :=
  let b := 0.76
  let x_mean := mean income
  let y_mean := mean expenditure
  let a := y_mean - b * x_mean
  b * x + a

theorem expenditure_of_income (income expenditure : List ℝ) (x : ℝ) : 
  regression_line income expenditure x = 11.8 :=
by 
  let income := [8.2, 8.6, 10.0, 11.3, 11.9]
  let expenditure := [6.2, 7.5, 8.0, 8.5, 9.8]
  let y := regression_line income expenditure 15
  have h : y = 0.76 * 15 + 0.4 := by sorry
  have h2 : y = 11.8 := by sorry
  exact h2

end expenditure_of_income_l727_727652


namespace general_solution_correct_l727_727072

-- Define the inhomogeneous system of linear equations
def inhomogeneous_system : List (List ℚ × ℚ) := [
  ([0, 1, 2, -3, 0], -1),
  ([2, -1, 3, 0, 4], 5),
  ([2, 0, 5, -3, 4], 4)
]

-- Define the general solution of the system
def general_solution_part : Vector ℚ 5 := ⟨[2, -1, 0, 0, 0], by decide⟩
def general_solution_fund1 : Vector ℚ 5 := ⟨[-5/2, -2, 1, 0, 0], by decide⟩
def general_solution_fund2 : Vector ℚ 5 := ⟨[3/2, 3, 0, 1, 0], by decide⟩
def general_solution_fund3 : Vector ℚ 5 := ⟨[-2, 0, 0, 0, 1], by decide⟩

-- Define the general form of the solution
def general_solution (C1 C2 C3 : ℚ) : Vector ℚ 5 :=
  general_solution_part + C1 • general_solution_fund1 +
  C2 • general_solution_fund2 + C3 • general_solution_fund3

-- The exact theorem to be proven
theorem general_solution_correct :
  ∃ (C1 C2 C3 : ℚ),
    ∀ (x : Vector ℚ 5),
      (x ∈ solutions_of_system inhomogeneous_system) ↔
      x = general_solution C1 C2 C3 :=
sorry  -- Proof is omitted

end general_solution_correct_l727_727072


namespace sqrt_720_l727_727994

theorem sqrt_720 : sqrt (720) = 12 * sqrt (5) :=
sorry

end sqrt_720_l727_727994


namespace common_difference_l727_727838

-- Define the conditions
def arithmetic_sequence (nums : List ℝ) : Prop :=
  ∀ (n : ℕ), n < nums.length - 1 → nums.get? n.succ = nums.get? n + some d

constant nums : List ℝ
constant S_total : ℝ
constant S_even : ℝ
constant d : ℝ

-- Given conditions
axiom h1 : nums.length = 20
axiom h2 : S_total = 75
axiom h3 : S_even = 25
axiom h4 : ∑ k in (finset.range nums.length), if k % 2 = 1 then nums.get k else 0 = 75 - 25
axiom h5 : S_even = ∑ k in (finset.range nums.length), if k % 2 = 0 then nums.get k else 0

-- Statement to prove
theorem common_difference : d = -2.5 :=
by sorry

end common_difference_l727_727838


namespace base7_digit_divisibility_l727_727446

-- Define base-7 digit integers
notation "digit" => Fin 7

-- Define conversion from base-7 to base-10 for the form 3dd6_7
def base7_to_base10 (d : digit) : ℤ := 3 * (7^3) + (d:ℤ) * (7^2) + (d:ℤ) * 7 + 6

-- Define the property of being divisible by 13
def is_divisible_by_13 (n : ℤ) : Prop := ∃ k : ℤ, n = 13 * k

-- Formalize the theorem
theorem base7_digit_divisibility (d : digit) :
  is_divisible_by_13 (base7_to_base10 d) ↔ d = 4 :=
sorry

end base7_digit_divisibility_l727_727446


namespace complement_M_N_contains_only_one_integer_l727_727574

noncomputable def M := {x : ℝ | abs x < 2}
noncomputable def N := ({-1, 1} : Set ℝ)
noncomputable def complement_M_N := {x ∈ M | x ∉ N}
noncomputable def count_integers (S : Set ℝ) := (S ∩ Set.range (coe : ℕ → ℝ)).card

theorem complement_M_N_contains_only_one_integer :
  count_integers complement_M_N = 1 :=
  sorry

end complement_M_N_contains_only_one_integer_l727_727574


namespace chocolates_in_box_l727_727326

theorem chocolates_in_box (chocolates_with_nuts_chances : ℕ) (chocolates_without_nuts_chances : ℕ)
  (eaten_with_nuts_chances : ℝ) (eaten_without_nuts_chances : ℝ) (remaining_chocolates : ℕ)
  (half_chocolates : chocolates_with_nuts_chances = chocolates_without_nuts_chances)
  (eaten_with_nuts_ratio : eaten_with_nuts_chances = 0.80)
  (eaten_without_nuts_ratio : eaten_without_nuts_chances = 0.50)
  (remaining_chocolates_eq : remaining_chocolates = 28) :
  ∃ (total_chocolates : ℕ), total_chocolates = 80 :=
by
  use 80
  sorry

end chocolates_in_box_l727_727326


namespace flower_counts_l727_727753

theorem flower_counts (R G Y : ℕ) : (R + G = 62) → (R + Y = 49) → (G + Y = 77) → R = 17 ∧ G = 45 ∧ Y = 32 :=
by
  intros h1 h2 h3
  sorry

end flower_counts_l727_727753


namespace prism_surface_area_volume_eq_l727_727722

theorem prism_surface_area_volume_eq (x : ℝ) (h : 2 * (log 2 x * log 3 x + log 2 x * log 4 x + log 3 x * log 4 x) = log 2 x * log 3 x * log 4 x) : x = 576 := 
  sorry

end prism_surface_area_volume_eq_l727_727722


namespace solution_l727_727775

noncomputable def f (x : ℝ) : ℝ := x + 1

theorem solution (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) :
    f (x * f y) = f (x * y) + x :=
by
  have h : ∀ (z : ℝ) (hz : 0 < z), f z = z + 1 := by
    intro z hz
    unfold f
    sorry
  unfold f
  simp
  sorry

end solution_l727_727775


namespace alex_wins_probability_l727_727543

theorem alex_wins_probability :
  let outcomes := {1, 2, 3, 4, 5, 6} in
  let even_numbers := {2, 4, 6} in
  let P_k_win_first := 3 / 6 in
  let P_k_lose_first := 1 - P_k_win_first in
  let P_a_win_first := 4 / 6 in
  let P_a_win_first_combined := P_k_lose_first * P_a_win_first in
  let P_k_win_second := 5 / 6 in
  let P_k_win_second_combined := (1 - P_a_win_first) * P_k_win_second in
  let P_a_win_second := 4 / 6 in
  let P_a_win_second_combined := (1 - P_k_win_second) * P_a_win_second in
  let P_alex_wins := P_a_win_first_combined + P_a_win_second_combined in
  P_alex_wins = 22 / 27 :=
begin
  sorry
end

end alex_wins_probability_l727_727543


namespace probability_at_least_three_same_l727_727079

theorem probability_at_least_three_same (n : ℕ) (dice : Fin n → Fin 6) (h : n = 5) : 
  (∃ k : ℚ, k = 113 / 648) :=
begin
  sorry,
end

end probability_at_least_three_same_l727_727079


namespace lemonade_percentage_correct_l727_727366
noncomputable def lemonade_percentage (first_lemonade first_carbon second_carbon mixture_carbon first_portion : ℝ) : ℝ :=
  100 - second_carbon

theorem lemonade_percentage_correct :
  let first_lemonade := 20
  let first_carbon := 80
  let second_carbon := 55
  let mixture_carbon := 60
  let first_portion := 19.99999999999997
  lemonade_percentage first_lemonade first_carbon second_carbon mixture_carbon first_portion = 45 :=
by
  -- Proof to be completed.
  sorry

end lemonade_percentage_correct_l727_727366


namespace algebraic_expression_value_l727_727163

theorem algebraic_expression_value (a b : ℝ) (h : a - b = 2 * real.sqrt 3) :
  ( ( (a ^ 2 + b ^ 2) / (2 * a) - b ) * (a / (a - b)) ) = real.sqrt 3 :=
by
  sorry

end algebraic_expression_value_l727_727163


namespace inequality_solution_set_l727_727224

noncomputable def odd_function_foid (f : ℝ → ℝ) : Prop :=
∀ x, f(-x) = -f(x)

noncomputable def even_function_foid (g : ℝ → ℝ) : Prop :=
∀ x, g(-x) = g(x)

theorem inequality_solution_set (f g : ℝ → ℝ)
  (h₁ : odd_function_foid f)
  (h₂ : even_function_foid g)
  (h₃ : ∀ x, x < 0 → (f'(x) * g(x) + f(x) * g'(x)) > 0)
  (h₄ : g(-3) = 0) :
  { x : ℝ | f(x) * g(x) < 0 } = set.Iio (-3) ∪ set.Ioo 0 3 :=
sorry

end inequality_solution_set_l727_727224


namespace segment_lengths_perpendicular_diameters_l727_727185

theorem segment_lengths_perpendicular_diameters 
    (radius : ℝ) (chord_length : ℝ) 
    (h1 : radius = 6)
    (h2 : chord_length = 10) :
    ∃ a b : ℝ, a + b = 2 * radius ∧ a = 6 - real.sqrt 11 ∧ b = 6 + real.sqrt 11 := 
by
  use [6 - real.sqrt 11, 6 + real.sqrt 11]
  split
  . rw [add_comm] -- Prove a + b = 12
    have h : 6 - real.sqrt 11 + 6 + real.sqrt 11 = 12 := 
    calc (6 - real.sqrt 11) + (6 + real.sqrt 11) 
       = (6 + 6) + (- real.sqrt 11 + real.sqrt 11) : by ring
   ... = 12 + 0 : by ring
   ... = 12 : by ring
    exact h
  . split
   . refl -- Prove a = 6 - real.sqrt 11
   . refl -- Prove b = 6 + real.sqrt 11

end segment_lengths_perpendicular_diameters_l727_727185


namespace cos_angle_CAE_l727_727215

/-- Let  ΔABC  be a triangle with  ∠BAC = 90°  and  ∠ABC = 60°. 
    Point  E  is chosen on side  BC  so that  BE : EC = 3 : 2. 
    Compute  cos(∠CAE). -/
theorem cos_angle_CAE (A B C E : Point) 
    (h1: ∠ A B C = 60) 
    (h2: ∠ B A C = 90) 
    (h3: ∠ B C A = 30) 
    (h4: point_on_segment E B C)
    (h5:  ratio_eq (line_segment B E) (line_segment E C) (3/2)) :
  cos (angle C A E) = 5 * sqrt 3 / sqrt 31 :=
sorry

end cos_angle_CAE_l727_727215


namespace total_wet_surface_area_is_correct_l727_727709

noncomputable def wet_surface_area (cistern_length cistern_width water_depth platform_length platform_width platform_height : ℝ) : ℝ :=
  let two_longer_walls := 2 * (cistern_length * water_depth)
  let two_shorter_walls := 2 * (cistern_width * water_depth)
  let area_walls := two_longer_walls + two_shorter_walls
  let area_bottom := cistern_length * cistern_width
  let submerged_height := water_depth - platform_height
  let two_longer_sides_platform := 2 * (platform_length * submerged_height)
  let two_shorter_sides_platform := 2 * (platform_width * submerged_height)
  let area_platform_sides := two_longer_sides_platform + two_shorter_sides_platform
  area_walls + area_bottom + area_platform_sides

theorem total_wet_surface_area_is_correct :
  wet_surface_area 8 4 1.25 1 0.5 0.75 = 63.5 :=
by
  -- The proof goes here
  sorry

end total_wet_surface_area_is_correct_l727_727709


namespace paintings_per_gallery_l727_727379

theorem paintings_per_gallery (pencils_total: ℕ) (pictures_initial: ℕ) (galleries_new: ℕ) (pencils_per_signing: ℕ) (pencils_per_picture: ℕ) (pencils_for_signature: ℕ) :
  pencils_total = 88 ∧ pictures_initial = 9 ∧ galleries_new = 5 ∧ pencils_per_picture = 4 ∧ pencils_per_signing = 2 → 
  (pencils_total - (galleries_new + 1) * pencils_per_signing) / pencils_per_picture - pictures_initial = galleries_new * 2 :=
by
  intros h,
  cases h with ha hb,
  sorry

end paintings_per_gallery_l727_727379


namespace intersection_of_M_and_N_l727_727139

def M : Set ℝ := { x | (x - 3) / (x - 1) ≤ 0 }
def N : Set ℝ := { x | -6 * x^2 + 11 * x - 4 > 0 }

theorem intersection_of_M_and_N : M ∩ N = { x | 1 < x ∧ x < 4 / 3 } :=
by 
  sorry

end intersection_of_M_and_N_l727_727139


namespace find_a10_l727_727853

def arith_seq (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n + d

variables (a : ℕ → ℚ) (d : ℚ)

-- Conditions
def condition1 := a 4 + a 11 = 16  -- translates to a_5 + a_12 = 16
def condition2 := a 6 = 1  -- translates to a_7 = 1
def condition3 := arith_seq a d  -- a is an arithmetic sequence with common difference d

-- The main theorem
theorem find_a10 : condition1 a ∧ condition2 a ∧ condition3 a d → a 9 = 15 := sorry

end find_a10_l727_727853


namespace find_a_values_l727_727069

theorem find_a_values (a : ℝ) (x y : ℝ) (x₁ y₁ x₂ y₂ : ℝ) (n : ℤ) :
  (x^2 + y^2 = 10 * (x * Real.cos (4 * a) + y * Real.sin (4 * a))) ∧ 
  (x^2 + y^2 = 10 * (x * Real.sin a + y * Real.cos a)) ∧ 
  ((x₁ - x₂)^2 + (y₁ - y₂)^2 = 36) → 
  (a = (Real.pi / 10) + (2 * n * Real.pi) / 5 ∨ 
   a = (Real.pi / 10) - (2 * n * Real.pi) / 5 ∨ 
   a = (Real.pi / 10) + (2 * n * Real.pi) / 5 + (2 / 5) * Real.arctan (4 / 3) ∨ 
   a = (Real.pi / 10) + (2 * n * Real.pi) / 5 - (2 / 5) * Real.arctan (4 / 3)) :=
begin
  sorry,
end

end find_a_values_l727_727069


namespace general_term_formula_l727_727629

def seq (n : ℕ) : ℤ :=
match n with
| 0       => 1
| 1       => -3
| 2       => 5
| 3       => -7
| 4       => 9
| (n + 1) => (-1)^(n+1) * (2*n + 1) -- extends indefinitely for general natural number

theorem general_term_formula (n : ℕ) : 
  seq n = (-1)^(n+1) * (2*n-1) :=
sorry

end general_term_formula_l727_727629


namespace chemical_solution_problem_l727_727708

theorem chemical_solution_problem:
  let initial_volume := 60
  let initial_concentration := 0.35
  let additional_volume := 18
  let desired_concentration := 0.5 in
  (initial_volume * initial_concentration + additional_volume) / 
  (initial_volume + additional_volume) = desired_concentration := by
    sorry

end chemical_solution_problem_l727_727708


namespace probability_neither_red_nor_purple_l727_727356

theorem probability_neither_red_nor_purple (total_balls red_balls purple_balls : ℕ) 
  (h_total : total_balls = 100) (h_red : red_balls = 47) (h_purple : purple_balls = 3) :
  (total_balls - (red_balls + purple_balls)) / total_balls = 0.5 := 
  by sorry

end probability_neither_red_nor_purple_l727_727356


namespace seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths_l727_727021

theorem seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths : 
  (7 : ℚ) * (9 + (2 / 5)) = 65 + (4 / 5) :=
by
  sorry

end seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths_l727_727021


namespace derivative_ln_expression_l727_727092

def y (x : ℝ) : ℝ := real.log (1 / real.sqrt (1 + x^2))

theorem derivative_ln_expression (x : ℝ) : deriv y x = - x / (1 + x^2) :=
by
  sorry

end derivative_ln_expression_l727_727092


namespace michael_total_fish_l727_727964

-- Definitions based on conditions
def michael_original_fish : ℕ := 31
def ben_fish_given : ℕ := 18

-- Theorem to prove the total number of fish Michael has now
theorem michael_total_fish : (michael_original_fish + ben_fish_given) = 49 :=
by sorry

end michael_total_fish_l727_727964


namespace functions_with_inverses_l727_727757

-- Define the first function as a linear function
def func_a (x : ℝ) : ℝ := x

-- Define the second function as a parabola
def func_b (x : ℝ) : ℝ := 1 - x^2

-- Define the third function as a semicircle
def func_c (x : ℝ) : ℝ := real.sqrt 9 - x^2

-- Define the fourth function as a piecewise linear function
def func_d (x : ℝ) : ℝ :=
  if x < -2 then -1
  else if x <= 2 then x + 2
  else 3

-- Define the Horizontal Line Test
def hlt (f : ℝ → ℝ) : Prop :=
  ∀ y : ℝ, ∃! x : ℝ, f x = y

-- The theorem to prove which functions have inverses
theorem functions_with_inverses :
  {f : ℝ → ℝ | hlt f} = {func_a, func_d} :=
  sorry

end functions_with_inverses_l727_727757


namespace multiply_mixed_number_l727_727034

def mixed_to_improper_fraction (n : ℕ) (a b : ℕ) : ℚ :=
  n + a / b

def improper_to_mixed_number (q : ℚ) : ℕ × ℚ :=
  let n := q.natAbs
  let r := q.fract
  (n, r)

theorem multiply_mixed_number (x y n a b : ℕ) (h : y = mixed_to_improper_fraction n a b) :
  x * y = mixed_to_improper_fraction 65 4 5 :=
  sorry

end multiply_mixed_number_l727_727034


namespace compare_numbers_l727_727284

theorem compare_numbers :
  (real.log 6 / real.log 0.7) < (0.7 ^ 6) ∧ (0.7 ^ 6) < (6 ^ 0.7) :=
by
  sorry

end compare_numbers_l727_727284


namespace greatest_b_not_in_range_l727_727315

theorem greatest_b_not_in_range (b : ℤ) : 
  (∀ x : ℝ, x^2 + (b : ℝ) * x + 12 ≠ -4) → b ≤ 7 := 
by {
  sorry
}

end greatest_b_not_in_range_l727_727315


namespace difference_max_min_l727_727862

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := x^3 - 3*x^2 - 6*x + c

theorem difference_max_min (c : ℝ) :
  (∃ a b : ℝ, ∀ x : ℝ, f x c = x^3 + 3*a*x^2 + 3*b*x + c) ∧
  (f' 2 = 0) ∧
  (f' 1 = -3) →
  ∃ m M : ℝ, f m c - f M c = 4 :=
by
  sorry

end difference_max_min_l727_727862


namespace volume_ratio_of_smaller_snowball_l727_727354

theorem volume_ratio_of_smaller_snowball (r : ℝ) (k : ℝ) :
  let V₀ := (4/3) * π * r^3
  let S := 4 * π * r^2
  let V_large := (4/3) * π * (2 * r)^3
  let V_large_half := V_large / 2
  let new_r := (V_large_half / ((4/3) * π))^(1/3)
  let reduction := 2*r - new_r
  let remaining_r := r - reduction
  let remaining_V := (4/3) * π * remaining_r^3
  let volume_ratio := remaining_V / V₀ 
  volume_ratio = 1/5 :=
by
  -- Proof goes here
  sorry

end volume_ratio_of_smaller_snowball_l727_727354


namespace maria_pays_correct_amount_l727_727237

def notebookA_cost : ℝ := 3.50
def notebookB_cost : ℝ := 2.25
def notebookC_cost : ℝ := 1.75
def pen_cost : ℝ := 2.00
def highlighters_cost : ℝ := 4.50
def coupon_discount : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05

def total_cost_before_coupon : ℝ :=
  (4 * notebookA_cost) + (3 * notebookB_cost) +
  (3 * notebookC_cost) + (5 * pen_cost) + highlighters_cost

def discounted_cost : ℝ :=
  total_cost_before_coupon * (1 - coupon_discount)

def sales_tax : ℝ :=
  discounted_cost * sales_tax_rate

def total_cost_after_tax : ℝ :=
  discounted_cost + sales_tax

theorem maria_pays_correct_amount :
  total_cost_after_tax = 38.27 :=
by
  rw [total_cost_before_coupon, discounted_cost, sales_tax, total_cost_after_tax]
  have h1 : total_cost_before_coupon = 40.50 := by sorry
  have h2 : discounted_cost = 36.45 := by sorry
  have h3 : sales_tax = 1.82 := by sorry
  rw [h1, h2, h3]
  norm_num

end maria_pays_correct_amount_l727_727237


namespace quadratic_distinct_real_roots_range_l727_727892

open Real

theorem quadratic_distinct_real_roots_range (k : ℝ) :
    (∃ a b c : ℝ, a = k^2 ∧ b = 4 * k - 1 ∧ c = 4 ∧ (b^2 - 4 * a * c > 0) ∧ a ≠ 0) ↔ (k < 1 / 8 ∧ k ≠ 0) :=
by
  sorry

end quadratic_distinct_real_roots_range_l727_727892


namespace q_and_r_are_contrapositives_l727_727512

variables {m n : Prop}

-- Proposition p
def p : Prop := m → n

-- Inverse proposition q
def q : Prop := n → m

-- Negation proposition r
def r : Prop := ¬m → ¬n

theorem q_and_r_are_contrapositives : q = (¬r) :=
by
  sorry

end q_and_r_are_contrapositives_l727_727512


namespace area_of_largest_circle_is_1000_pi_l727_727648

noncomputable def area_largest_circle (r : ℝ) (C1 C2 C3 : E × E) (R : ℝ) (A : ℝ) : Prop :=
  let d := Math.sqrt (40^2 + 20^2) / 2
  in 10 = r ∧ 
      (@dist E _ C1 C2 = 2 * r ∧ 
       @dist E _ C2 C3 = 2 * r ∧ 
       @dist E _ C1 C3 = 4 * r) ∧ 
      20 = 2 * r ∧ 
      10 * Math.sqrt 5 = d ∧ 
      A = π * (10 * Math.sqrt 5)^2

theorem area_of_largest_circle_is_1000_pi :
  ∃ (r : ℝ), ∃ (C1 C2 C3 : E × E), ∃ (R : ℝ), ∃ (A : ℝ),
  area_largest_circle r C1 C2 C3 R A ∧ A = 1000 * π := 
sorry

end area_of_largest_circle_is_1000_pi_l727_727648


namespace vector_magnitude_theorem_l727_727273

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.fst * v.fst + v.snd * v.snd)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.fst * v2.fst + v1.snd * v2.snd

theorem vector_magnitude_theorem: 
  ∀ (a b : ℝ × ℝ),
    (a = (2, 0)) →
    (vector_magnitude b = 1) →
    (dot_product a b = 1) →
    (vector_magnitude (a.1 + 2 * b.1, a.2 + 2 * b.2) = 2 * real.sqrt 3) := 
by
  intros a b ha hb hab
  rw [ha, hb, hab]
  sorry

end vector_magnitude_theorem_l727_727273


namespace max_min_sum_l727_727119

variable {α : Type*} [LinearOrderedField α]

def is_odd_function (g : α → α) : Prop :=
∀ x, g (-x) = - g x

def has_max_min (f : α → α) (M N : α) : Prop :=
  (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ (∀ x, N ≤ f x) ∧ (∃ x₁, f x₁ = N)

theorem max_min_sum (g f : α → α) (M N : α)
  (h_odd : is_odd_function g)
  (h_def : ∀ x, f x = g (x - 2) + 1)
  (h_max_min : has_max_min f M N) :
  M + N = 2 :=
sorry

end max_min_sum_l727_727119


namespace gcd_of_12012_18018_l727_727794

theorem gcd_of_12012_18018 : gcd 12012 18018 = 6006 :=
by
  -- Definitions for conditions
  have h1 : 12012 = 12 * 1001 := by
    sorry
  have h2 : 18018 = 18 * 1001 := by
    sorry
  have h3 : gcd 12 18 = 6 := by
    sorry
  -- Using the conditions to prove the main statement
  rw [h1, h2]
  rw [gcd_mul_right, gcd_mul_right]
  rw [gcd_comm 12 18, h3]
  rw [mul_comm 6 1001]
  sorry

end gcd_of_12012_18018_l727_727794


namespace sum_of_elements_in_A_l727_727458

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

def A : set ℝ := { x | f (f x) = 0 }

theorem sum_of_elements_in_A : ∑ x in {0, 2, 1 - real.sqrt 3, 1 + real.sqrt 3}, x = 4 :=
by {
  have element_set : A = {0, 2, 1 - real.sqrt 3, 1 + real.sqrt 3},
  sorry,
  have sum_elements := set.sum of element_set,
  rw sum_elements,
  calc 0 + 2 + (1 - real.sqrt 3) + (1 + real.sqrt 3)
       = 0 + 2 + 1 - real.sqrt 3 + 1 + real.sqrt 3 : by ring
   ... = 0 + 2 + 1 + 1 : by linarith
   ... = 4 : by norm_num
}

end sum_of_elements_in_A_l727_727458


namespace problem1_problem2_l727_727691

section problem1

variables {α : ℝ} (sin_alpha := 3 / 5)

-- Conditions
-- α is in the second quadrant
-- sin α = 3/5

def cos_alpha (h1 : 1 - (sin_alpha ^ 2) = (4 / 5) ^ 2) : ℝ :=
  -real.sqrt (1 - (sin_alpha ^ 2))

theorem problem1 (h1 : 1 - (sin_alpha ^ 2) = (4 / 5) ^ 2) :
  let cos_alpha := cos_alpha h1 in
  (1 + sin_alpha + cos_alpha + 2 * sin_alpha * cos_alpha) / (1 + sin_alpha + cos_alpha) = -1 / 5 := 
by
  sorry

end problem1


section problem2

variables {α : ℝ}

-- Conditions
-- α is in the second quadrant

theorem problem2 (h : ∀ α, 0 < α ∧ α < π):
  ∀ (sin_alpha cos_alpha : ℝ), 
    cos_alpha = -real.sqrt (1 - sin_alpha ^ 2) →
    sin_alpha ^ 2 + cos_alpha ^ 2 = 1 →
    cos_alpha sqrt ((1 - sin_alpha) / (1 + sin_alpha)) + sin_alpha sqrt ((1 - cos_alpha) / (1 + cos_alpha)) = sin_alpha - cos_alpha :=
by
  sorry

end problem2

end problem1_problem2_l727_727691


namespace evaluate_sqrt_exponent_l727_727783

theorem evaluate_sqrt_exponent : (real.sqrt 4 ^ (1 / 6)) ^ 9 = 8 := by
  sorry

end evaluate_sqrt_exponent_l727_727783


namespace find_14_points_l727_727522

def scores : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

noncomputable def player_scores : List (String × ℕ) :=
  [("Alan", 27), ("Betty", 23), ("Carla", 30), ("Dan", 15), ("Eliza", 21), ("Felix", 24)]

def throws (player : String) : List ℕ := sorry -- to be defined with valid categories of 3 unique scores per player's total.

theorem find_14_points :
  ∃ player, 14 ∈ throws player ∧ 
  (player, 24) ∈ player_scores :=
by
  apply Exists.intro "Felix"
  intro h
  rw <- h at player_scores
  apply Exists.intro "Felix"
  rw <- h
  sorry

end find_14_points_l727_727522


namespace dessert_menu_count_l727_727358

variable (Dessert : Type) [Fintype Dessert]
variable (cake pie ice_cream pudding : Dessert)

noncomputable def totalDessertMenus : Nat :=
  let choices (d y : Dessert) := {d' : Dessert // d' ≠ d ∧ (y = cake → d' ≠ pie ∧ d' ≠ ice_cream) ∧ (y = pudding → d' ≠ pudding) ∧ (y = ice_cream → d' ≠ ice_cream) ∧ (y = pie → d' ≠ pie)}
  let monday := pudding
  let friday := cake
  let day_choices (prev : Option Dessert) (current : Dessert) : List Dessert :=
    match prev with
    | none => [cake, pie, ice_cream, pudding]
    | some prev => [cake, pie, ice_cream, pudding].filter (· ≠ prev)
  let week_choices : List (Option Dessert → Nat) := [
    λ _ => 1,       -- Monday (fixed)
    λ _ => 3,       -- Tuesday
    λ tuesday => if tuesday = cake then 2 else 1,  -- Wednesday (depends on Tuesday)
    λ _ => 3,       -- Thursday
    λ _ => 1,       -- Friday (fixed)
    λ _ => 3,       -- Saturday
    λ saturday => 3 -- Sunday (can't be same as Saturday)
  ]
  List.foldr (λ f acc => acc * f none) 1 week_choices

theorem dessert_menu_count : totalDessertMenus cake pie ice_cream pudding = 151 := 
by {
  sorry
}

end dessert_menu_count_l727_727358


namespace smallest_positive_period_cos_sq_l727_727778

theorem smallest_positive_period_cos_sq (x : ℝ) : ∃ T > 0, (∀ t : ℝ, cos (x + T) ^ 2 = cos x ^ 2) ∧ T = π := 
by sorry

end smallest_positive_period_cos_sq_l727_727778


namespace square_area_720_l727_727588

noncomputable def length_squared {α : Type*} [EuclideanDomain α] (a b : α) := a * a + b * b

theorem square_area_720
  (side x : ℝ)
  (h1 : BE = 20) (h2 : EF = 20) (h3 : FD = 20)
  (h4 : AE = 2 * ED) (h5 : BF = 2 * FC)
  : x * x = 720 :=
by
  let AE := 2/3 * side
  let ED := 1/3 * side
  let BF := 2/3 * side
  let FC := 1/3 * side
  have h6 : length_squared BF EF = BE * BE := sorry
  have h7 : x * x = 720 := sorry
  exact h7

end square_area_720_l727_727588


namespace vertical_asymptotes_l727_727787

-- Define the function
def f (x : ℝ) := (x + 3) / (x^2 - 5*x + 6)

-- Statement of the problem
theorem vertical_asymptotes : 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (∀ x, (x - x1) * (x - x2) = x^2 - 5*x + 6) ∧ 
  (∀ x, f x = (x + 3) / ((x - x1) * (x - x2))) ∧ 
  (x1 ∉ {x | f x = 0}) ∧ (x2 ∉ {x | f x = 0}))
  → 
  ∃ x1 x2 : ℝ, (∀ x ∈ {x1, x2}, (f x) = 0) ∧ x1 ≠ x2 :=
sorry

end vertical_asymptotes_l727_727787


namespace lambda_sum_le_n_l727_727949

theorem lambda_sum_le_n (n : ℕ) (λ : ℕ → ℝ) (θ : ℝ) 
  (h : ∀ θ : ℝ, (∑ i in finset.range n, λ i * real.cos (i.succ * θ)) ≥ -1) : 
  (∑ i in finset.range n, λ i.succ) ≤ n := sorry

end lambda_sum_le_n_l727_727949


namespace fraction_increase_invariance_l727_727888

theorem fraction_increase_invariance (x y : ℝ) :
  (3 * (2 * y)) / (2 * x + 2 * y) = 3 * y / (x + y) :=
by
  sorry

end fraction_increase_invariance_l727_727888


namespace construct_triangle_with_median_l727_727049

-- Define the sides and the median
variable {a b m_c c : ℝ}
-- Assume positive values for sides and median can build a valid triangle
axiom sides_and_median (h1 : a > 0) (h2 : b > 0) (h3 : m_c > 0) : Prop

-- Define the construction target: there exists a triangle with sides a, b, and 
-- a median m_c relating to the third side length.
theorem construct_triangle_with_median (h1 : a > 0) (h2 : b > 0) (h3 : m_c > 0) :
  ∃ (A B C : Type) (triangle : (A • B • C → Prop)), 
  (∃ (AB BC CA : ℝ), AB = a ∧ BC = b ∧ CA = c ∧ median CA BC = m_c) :=
sorry

end construct_triangle_with_median_l727_727049


namespace minimum_base_ten_with_four_octal_digits_l727_727672

theorem minimum_base_ten_with_four_octal_digits : ∃ (n : ℕ), (∀ (k : ℕ), k < n → k.oct.numDigits' < 4) ∧ n.oct.numDigits' = 4 :=
by
  use 512
  split
  { intro k
    intro hk
    -- Here we would show that k < 512 implies k.oct.numDigits’ < 4
    sorry }
  { -- Here we would show that 512.oct.numDigits’ = 4
    sorry }

end minimum_base_ten_with_four_octal_digits_l727_727672


namespace work_done_in_isothermal_process_l727_727246

-- Definitions for the problem
def ideal_monatomic_gas (n : ℕ) := true -- n represents number of moles

def isobaric_process (work_isobaric : ℝ):= 
  ∃ (p : ℝ) (ΔV : ℝ), work_isobaric = p * ΔV

def heat_added_isobaric (ΔU : ℝ) (work_isobaric : ℝ) := 
  let R := 8.314 
  ΔU = (3 / 2) * R * ΔT ∧ Q_isobaric = ΔU + work_isobaric

def isothermal_process (work_isothermal : ℝ) {Q_isothermal : ℝ} := 
  Q_isothermal = work_isothermal

-- The theorem to be proved
theorem work_done_in_isothermal_process 
  (n : ℕ) (work_isobaric : ℝ) (ΔU : ℝ) (ΔT : ℝ) (R : ℝ := 8.314):
  ideal_monatomic_gas n ∧ isobaric_process work_isobaric ∧ heat_added_isobaric ΔU work_isobaric
  ∧ isothermal_process work_isothermal {Q_isobaric := 25} :=
by
  -- Given one mole of the ideal monatomic gas
  have h1 : ideal_monatomic_gas 1 := trivial,
  -- The work done by gas in isobaric process is 10 Joules
  have h2 : isobaric_process 10 := 
    by 
      -- Existence of pressure and volume change for work calculation
      let p := 1
      let ΔV := 10
    _,
  -- The heat added to gas in isobaric process equals the work done
  have h3 : heat_added_isobaric ΔU 10 := by 
    let Q_isobaric := 25
    -- Calculation of heat based on given conditions
    _,
  -- The isothermal process where work done is equal to heat added, which equals heat added in isobaric process
  have h4 : isothermal_process 25 {Q_isothermal := 25} := 
    by
      let Q_isothermal := 25
      let work_isothermal := 25
    _,
  sorry -- Proof is not required, so we end with sorry.

end work_done_in_isothermal_process_l727_727246


namespace seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths_l727_727025

theorem seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths : 
  (7 : ℚ) * (9 + (2 / 5)) = 65 + (4 / 5) :=
by
  sorry

end seven_mul_nine_and_two_fifths_eq_sixty_five_and_four_fifths_l727_727025


namespace ratio_EG_FH_l727_727975

variables (E F G H : Type) [linear_order E] [linear_order F] [linear_order G] [linear_order H]

def length (a b : Type) [linear_order a] [linear_order b] : ℕ

axiom EF_length : length E F = 3
axiom FG_length : length F G = 6
axiom GH_length : length G H = 4
axiom EH_length : length E H = 20

noncomputable def EG_length := length E F + length F G

noncomputable def FH_length := length E H - length E F

theorem ratio_EG_FH : EG_length = 9 → FH_length = 17 → EG_length / FH_length = 9 / 17 :=
begin
  intros hEG hFH,
  rw hEG,
  rw hFH,
  norm_num,
end

end ratio_EG_FH_l727_727975


namespace conversion_base8_to_base10_l727_727761

theorem conversion_base8_to_base10 : 
  (4 * 8^3 + 5 * 8^2 + 3 * 8^1 + 2 * 8^0) = 2394 := by 
  sorry

end conversion_base8_to_base10_l727_727761


namespace gcd_12012_18018_l727_727802

theorem gcd_12012_18018 : Nat.gcd 12012 18018 = 6006 := sorry

end gcd_12012_18018_l727_727802


namespace tan_alpha_neq_sin_cos_trig_identity_l727_727086

theorem tan_alpha_neq (α : ℝ) (h1 : 0 < α) (h2 : α < π) (h3 : sin α + cos α = 1/5) : 
  tan α = -4/3 := 
by sorry

theorem sin_cos_trig_identity (α : ℝ) (h1 : 0 < α) (h2 : α < π) (h3 : sin α + cos α = 1/5): 
  sin α^2 - 3 * sin α * cos α - 4 * cos α^2 = 16 / 25 := 
by sorry

end tan_alpha_neq_sin_cos_trig_identity_l727_727086


namespace multiply_mixed_number_l727_727029

def mixed_to_improper_fraction (n : ℕ) (a b : ℕ) : ℚ :=
  n + a / b

def improper_to_mixed_number (q : ℚ) : ℕ × ℚ :=
  let n := q.natAbs
  let r := q.fract
  (n, r)

theorem multiply_mixed_number (x y n a b : ℕ) (h : y = mixed_to_improper_fraction n a b) :
  x * y = mixed_to_improper_fraction 65 4 5 :=
  sorry

end multiply_mixed_number_l727_727029


namespace tangent_perpendicular_points_l727_727171

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 + x - 3

theorem tangent_perpendicular_points :
  let f' := λ x : ℝ, 6 * x^2 + 1
  (x0 y0 : ℝ) (h1 : f' x0 = 7) (h2 : y0 = f x0) =>
  (x0 = 1 ∧ y0 = 0) ∨ (x0 = -1 ∧ y0 = -6) := 
by
  intro f' x0 y0 h1 h2
  change 6 * x0^2 + 1 = 7 at h1
  have eq1 : x0 = 1 ∨ x0 = -1 := 
    by
      simp at h1
      exact eq_of_eq_of_mul_eq_mul 6 1 1 h1
  cases eq1 with
  | inl hx1 =>
    left
    split ; assumption
  | inr hx2 => 
    right
    split ; assumption
  sorry

end tangent_perpendicular_points_l727_727171


namespace farmer_shipped_67_dozens_l727_727210

def pomelos_in_box (box_type : String) : ℕ :=
  if box_type = "small" then 10 else if box_type = "medium" then 20 else if box_type = "large" then 30 else 0

def total_pomelos_last_week : ℕ := 360

def boxes_this_week (box_type : String) : ℕ :=
  if box_type = "small" then 10 else if box_type = "medium" then 8 else if box_type = "large" then 7 else 0

def damage_boxes (box_type : String) : ℕ :=
  if box_type = "small" then 3 else if box_type = "medium" then 2 else if box_type = "large" then 2 else 0

def loss_percentage (box_type : String) : ℕ :=
  if box_type = "small" then 10 else if box_type = "medium" then 15 else if box_type = "large" then 20 else 0

def total_pomelos_shipped_this_week : ℕ :=
  (boxes_this_week "small") * (pomelos_in_box "small") +
  (boxes_this_week "medium") * (pomelos_in_box "medium") +
  (boxes_this_week "large") * (pomelos_in_box "large")

def total_pomelos_lost_this_week : ℕ :=
  (damage_boxes "small") * (pomelos_in_box "small") * (loss_percentage "small") / 100 +
  (damage_boxes "medium") * (pomelos_in_box "medium") * (loss_percentage "medium") / 100 +
  (damage_boxes "large") * (pomelos_in_box "large") * (loss_percentage "large") / 100

def total_pomelos_shipped_successfully_this_week : ℕ :=
  total_pomelos_shipped_this_week - total_pomelos_lost_this_week

def total_pomelos_for_both_weeks : ℕ :=
  total_pomelos_last_week + total_pomelos_shipped_successfully_this_week

def total_dozens_shipped : ℕ :=
  total_pomelos_for_both_weeks / 12

theorem farmer_shipped_67_dozens :
  total_dozens_shipped = 67 := 
by sorry

end farmer_shipped_67_dozens_l727_727210


namespace gumballs_per_package_correct_l727_727149

-- Define the conditions
def total_gumballs_eaten : ℕ := 20
def number_of_boxes_finished : ℕ := 4

-- Define the target number of gumballs in each package
def gumballs_in_each_package := 5

theorem gumballs_per_package_correct :
  total_gumballs_eaten / number_of_boxes_finished = gumballs_in_each_package :=
by
  sorry

end gumballs_per_package_correct_l727_727149


namespace smallest_A_l727_727271

open Finset

-- Definitions specific to our problem
variables (a : ℕ → ℤ) (S : Finset ℕ)

-- Conditions
def circle_sum_eq_100 : Prop :=
  ∑ i in range 10, a i = 100

def triplet_sum_ge_29 : Prop :=
  ∀ i in range 10, a i + a ((i + 1) % 10) + a ((i + 2) % 10) ≥ 29

def each_number_le_A (A : ℤ) : Prop :=
  ∀ i in range 10, a i ≤ A

-- The main statement we want to prove
theorem smallest_A :
  circle_sum_eq_100 a ∧ triplet_sum_ge_29 a → each_number_le_A a 13 :=
sorry

end smallest_A_l727_727271


namespace max_elements_l727_727463

-- Define the subset S and the conditions
def is_valid_subset (S : Finset ℕ) : Prop :=
  ∀ a b ∈ S, ∃ c ∈ S, Nat.gcd a c = 1 ∧ Nat.gcd b c = 1 ∧
  ∃ c' ∈ S, c' ≠ a ∧ c' ≠ b ∧ Nat.gcd a c' > 1 ∧ Nat.gcd b c' > 1

-- The main theorem stating the size of the subset satisfying the conditions
theorem max_elements (S : Finset ℕ) (hS : is_valid_subset S) (h_nonempty : S.nonempty) (h_subset : ∀ x ∈ S, x ∈ Finset.range 109) :
  S.card ≤ 76 :=
sorry

end max_elements_l727_727463


namespace lim_integral_fn_l727_727825

def fn (n : ℕ) (x : ℝ) : ℝ := Real.arctan (⌊x⌋)

theorem lim_integral_fn : ∀ n : ℕ, (fn n) is RiemannIntegrable ∧ (tendsto (λ n, (1 : ℝ) / (n : ℝ) * ∫(0:ℝ)..(n:ℝ), fn n) atTop (𝓝 (Real.pi / 2))) :=
by
  sorry

end lim_integral_fn_l727_727825


namespace rate_of_interest_l727_727342

theorem rate_of_interest (P A : ℝ) (t : ℝ) (hP : P = 600) (hA : A = 720) (ht : t = 4) : 
    (A - P) / (P * t) = 0.05 :=
by
  rw [hP, hA, ht]
  norm_num
  /- the expected proof steps would validate the arithmetic, though formally skipped here -/
  sorry

end rate_of_interest_l727_727342


namespace difference_C_D_l727_727035

def C : ℤ := (List.sum $ List.map (λ i : ℤ, (2 * i - 1) * 2 * i) (List.range 20)) + 41
def D : ℤ := 1 + (List.sum $ List.map (λ i : ℤ, (2 * i) * (2 * i + 1)) (List.range 20 - 1)) + 40 * 41

theorem difference_C_D : abs (C - D) = 800 :=
by
  sorry

end difference_C_D_l727_727035


namespace crayons_difference_l727_727595

noncomputable def initial_crayons : ℕ := 250
noncomputable def gave_crayons : ℕ := 150
noncomputable def lost_crayons : ℕ := 512
noncomputable def broke_crayons : ℕ := 75
noncomputable def traded_crayons : ℕ := 35

theorem crayons_difference :
  lost_crayons - (gave_crayons + broke_crayons + traded_crayons) = 252 := by
  sorry

end crayons_difference_l727_727595


namespace sqrt_720_simplified_l727_727991

theorem sqrt_720_simplified : (sqrt 720 = 12 * sqrt 5) :=
by
  -- The proof is omitted as per the instructions
  sorry

end sqrt_720_simplified_l727_727991


namespace roger_has_more_candy_l727_727263

-- Defining the conditions
def sandra_bag1 : Nat := 6
def sandra_bag2 : Nat := 6
def roger_bag1 : Nat := 11
def roger_bag2 : Nat := 3

-- Calculating the total pieces of candy for Sandra and Roger
def total_sandra : Nat := sandra_bag1 + sandra_bag2
def total_roger : Nat := roger_bag1 + roger_bag2

-- Statement of the proof problem
theorem roger_has_more_candy : total_roger - total_sandra = 2 := by
  sorry

end roger_has_more_candy_l727_727263


namespace A_inter_B_empty_l727_727492

def Z_plus := { n : ℤ // 0 < n }

def A : Set ℤ := { x | ∃ n : Z_plus, x = 2 * (n.1) - 1 }
def B : Set ℤ := { y | ∃ x ∈ A, y = 3 * x - 1 }

theorem A_inter_B_empty : A ∩ B = ∅ :=
by {
  sorry
}

end A_inter_B_empty_l727_727492


namespace julia_total_food_cost_l727_727207

def parrot_food_cost : ℝ := 4
def rabbit_food_cost : ℝ := 3
def turtle_food_cost : ℝ := 2
def guinea_pig_food_cost : ℝ := 1

def parrot_food_per_week : ℝ := 3.75
def rabbit_food_per_week : ℝ := 4
def turtle_food_per_week : ℝ := 4
def guinea_pig_food_per_week : ℝ := 5

def parrot_weeks : ℝ := 3
def rabbit_weeks : ℝ := 5
def turtle_weeks : ℝ := 2
def guinea_pig_weeks : ℝ := 6

def parrot_total_cost : ℝ := parrot_food_cost * parrot_food_per_week * parrot_weeks
def rabbit_total_cost : ℝ := rabbit_food_cost * rabbit_food_per_week * rabbit_weeks
def turtle_total_cost : ℝ := turtle_food_cost * turtle_food_per_week * turtle_weeks
def guinea_pig_total_cost : ℝ := guinea_pig_food_cost * guinea_pig_food_per_week * guinea_pig_weeks

def total_cost : ℝ := parrot_total_cost + rabbit_total_cost + turtle_total_cost + guinea_pig_total_cost

theorem julia_total_food_cost : total_cost = 151 := 
  by /* proof steps here */
  sorry

end julia_total_food_cost_l727_727207


namespace conversion_correct_l727_727765

-- Define the base 8 number
def base8_number : ℕ := 4 * 8^3 + 5 * 8^2 + 3 * 8^1 + 2 * 8^0

-- Define the target base 10 number
def base10_number : ℕ := 2394

-- The theorem that needs to be proved
theorem conversion_correct : base8_number = base10_number := by
  sorry

end conversion_correct_l727_727765


namespace expected_value_star_player_games_prob_star_player_played_given_win_l727_727651

/-- Conditions -/
def win_prob_star_player : ℚ := 3/4
def win_prob_other_players : ℚ := 1/2

/-- Define the random variable \(X\) representing the number of games won by team A in the first three games,
where the star player \(M\) plays in all three games, and calculate the probabilities. -/
def prob_X_0 : ℚ := 1/16
def prob_X_1 : ℚ := 5/16
def prob_X_2 : ℚ := 7/16
def prob_X_3 : ℚ := 3/16

/-- Define the expected value of the random variable \(X\) based on its probability distribution. -/
def expected_value_X : ℚ := 7/4

/-- Probability that teams A and B played 3 games, team A won with a score of 3:0,
    and M played in the first 3 games. -/
def prob_M_played_given_won_3_0 : ℚ := 9/13

theorem expected_value_star_player_games : 
  ∑ k in finset.range 4, k * (ite (k = 0) prob_X_0 (ite (k = 1) prob_X_1 (ite (k = 2) prob_X_2 (if k = 3 then prob_X_3 else 0)))) = expected_value_X := 
  by sorry

theorem prob_star_player_played_given_win : 
  prob_M_played_given_won_3_0 = 9/13 := 
  by sorry

end expected_value_star_player_games_prob_star_player_played_given_win_l727_727651


namespace problem_solution_l727_727938

def largest_number_of_elements (S : Set ℕ) : Prop :=
  S ⊆ {n | 1 ≤ n ∧ n ≤ 1989} ∧
  (∀ a b ∈ S, a ≠ b → |a - b| ≠ 4) ∧
  (∀ a b ∈ S, a ≠ b → |a - b| ≠ 7) →
  Finset.card S ≤ 905

theorem problem_solution : ∃ S : Set ℕ, largest_number_of_elements S ∧ Finset.card S = 905 :=
sorry

end problem_solution_l727_727938


namespace bookshelf_arrangements_l727_727818

theorem bookshelf_arrangements :
  let math_books := 6
  let english_books := 5
  let valid_arrangements := 2400
  (∃ (math_books : Nat) (english_books : Nat) (valid_arrangements : Nat), 
    math_books = 6 ∧ english_books = 5 ∧ valid_arrangements = 2400) :=
by
  sorry

end bookshelf_arrangements_l727_727818


namespace det_AB_det_A_inv_l727_727161

-- Given conditions
variables (A B : Matrix n n ℝ) (hA : det A = 5) (hB : det B = 7)

-- Theorem statements
theorem det_AB : det (A ⬝ B) = 35 :=
by sorry

theorem det_A_inv : det (A⁻¹) = 1 / 5 :=
by sorry

end det_AB_det_A_inv_l727_727161


namespace smallest_positive_n_l727_727821

noncomputable def smallest_n : ℕ :=
  (1002! ^ 2) + 2004

theorem smallest_positive_n (f : ℤ[X]) :
  (∃ x : ℤ, f.eval x = 2004) ∧ (∃ n : ℕ, n ≠ 2004 ∧ (∀ x : ℤ, f.eval x = n → ∃ p, p.card = 2004 ∧ ∀ y ∈ p, f.eval y = n)) →
  smallest_n = (1002! ^ 2) + 2004 :=
begin
  sorry
end

end smallest_positive_n_l727_727821


namespace john_heroes_on_large_sheets_front_l727_727541

noncomputable def num_pictures_on_large_sheets_front : ℕ :=
  let total_pictures := 20
  let minutes_spent := 75 - 5
  let average_time_per_picture := 5
  let front_pictures := total_pictures / 2
  let x := front_pictures / 3
  2 * x

theorem john_heroes_on_large_sheets_front : num_pictures_on_large_sheets_front = 6 :=
by
  sorry

end john_heroes_on_large_sheets_front_l727_727541


namespace equation_parallel_through_P2_l727_727496

variable {X : Type}
variable [Field X]

-- Define the points P1 and P2
variable (x1 y1 x2 y2 : X)

-- Function representing line l
variable (f : X → X → X)

-- Define the conditions
def on_line (x y : X) : Prop := f x y = 0
def not_on_line (x y : X) : Prop := f x y ≠ 0

-- The main theorem to prove
theorem equation_parallel_through_P2 (hP1 : on_line x1 y1) (hP2 : not_on_line x2 y2) :
  ∀ x y, f x y - f x1 y1 - f x2 y2 = 0 ↔ (∃ k : X, f x y = k ∧ k - f x2 y2 = 0) :=
by
  sorry

end equation_parallel_through_P2_l727_727496


namespace partition_generating_function_l727_727954

noncomputable def p : ℕ → ℕ := λ n, -- p(n) is the number of partitions of n
  sorry

theorem partition_generating_function :
  (∑ n : ℕ, p n * x ^ n) = ∏ k : ℕ in (set.Ici 1), (1 - x ^ k)⁻¹ :=
sorry

end partition_generating_function_l727_727954


namespace product_of_solutions_l727_727434

theorem product_of_solutions : 
  let f := λ x : ℝ, abs (x - 1) = 3 * (abs (x - 1) - 2) in
  (∃ x1 x2 : ℝ, f x1 ∧ f x2 ∧ x1 ≠ x2 ∧ (x1 * x2 = -8)) := by
  sorry

end product_of_solutions_l727_727434


namespace roshini_sweets_cost_correct_l727_727235

noncomputable def roshini_sweet_cost_before_discounts_and_tax : ℝ := 10.54

theorem roshini_sweets_cost_correct (R F1 F2 F3 : ℝ) (h1 : R + F1 + F2 + F3 = 10.54)
    (h2 : R * 0.9 = (10.50 - 9.20) / 1.08)
    (h3 : F1 + F2 + F3 = 3.40 + 4.30 + 1.50) :
    R + F1 + F2 + F3 = roshini_sweet_cost_before_discounts_and_tax :=
by
  sorry

end roshini_sweets_cost_correct_l727_727235


namespace ranking_arrangements_l727_727642

def students := {A, B, C, D, E}

theorem ranking_arrangements : 
  ∃ n : ℕ, n = 18 ∧ ∀ rankings : students → ℕ, 
  (rankings A ≠ 1) ∧ (rankings B = 3) → (list.perm (list.range 1 6) (list.map rankings students)) :=
sorry

end ranking_arrangements_l727_727642


namespace mul_mixed_number_l727_727004

theorem mul_mixed_number (a b : ℝ) (c : ℚ) (h : c = 9 + 2/5) : 
  7 * c = (65 + 4/5 : ℝ) :=
by 
  simp [h, mul_add, mul_div_cancel_left, rat.cast_add, rat.cast_mul, rat.cast_one]
  sorry

end mul_mixed_number_l727_727004


namespace determine_AM_l727_727843

theorem determine_AM (X A Y B M M' P : Point) (BC : Line) (AX AY : Ray)
  (h1 : isAngle X A Y)
  (h2 : onRay B AX)
  (h3 : onRay M AY)
  (h4 : area_eq (triangle B A M) (triangle M' M P))
  (h5 : dist_eq (dist B M) (dist B M'))
  (h6 : perp M' P AY) :
  (dist A M = (4/3) * dist A C ∨ dist A M = (4/5) * dist A C) :=
sorry

end determine_AM_l727_727843


namespace arithmetic_sum_S8_l727_727837

theorem arithmetic_sum_S8 (S : ℕ → ℕ)
  (h_arithmetic : ∀ n, S (n + 1) - S n = S 1 - S 0)
  (h_positive : ∀ n, S n > 0)
  (h_S4 : S 4 = 10)
  (h_S12 : S 12 = 130) : 
  S 8 = 40 :=
sorry

end arithmetic_sum_S8_l727_727837


namespace add_two_integers_l727_727780

/-- If the difference of two positive integers is 5 and their product is 180,
then their sum is 25. -/
theorem add_two_integers {x y : ℕ} (h1: x > y) (h2: x - y = 5) (h3: x * y = 180) : x + y = 25 :=
sorry

end add_two_integers_l727_727780


namespace sum_of_first_mk_terms_l727_727528

open Nat

theorem sum_of_first_mk_terms (a : ℕ → ℝ) (m k : ℕ) (h_m_pos : 0 < m) (h_k_pos : 0 < k) (h_mk : m ≠ k) 
  (h_a_n : ∃ n, a n = 1 / k) (h_a_k : a k = 1 / m) : 
  ∑ i in range (m * k), a i = (m * k + 1) / 2 :=
sorry

end sum_of_first_mk_terms_l727_727528


namespace sum_reciprocal_l727_727639

-- Definition of the problem
theorem sum_reciprocal (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 + y^2 = 4 * x * y) : 
  (1 / x) + (1 / y) = 1 :=
sorry

end sum_reciprocal_l727_727639


namespace exists_kites_greater_than_points_l727_727593

-- Define the problem specifically in terms of points and kites
def kite_condition (kites points : ℕ) : Prop := kites > points

-- We need to state the existence of an arrangement of points into kites so that
-- there are more kites than points.
theorem exists_kites_greater_than_points :
  ∃ points kites, kite_condition kites points :=
begin
  let points := 21,
  let kites := 24,
  use [points, kites],
  unfold kite_condition,
  exact nat.lt_succ_of_le (nat.zero_le points),
  sorry
end

end exists_kites_greater_than_points_l727_727593


namespace unit_vector_parallel_to_a_l727_727454

open Real

def vector_a : ℝ × ℝ := (2, 1)

def magnitude (v : ℝ × ℝ) : ℝ := 
sqrt (v.1^2 + v.2^2)

def normalize (v : ℝ × ℝ) : ℝ × ℝ := 
(v.1 / magnitude v, v.2 / magnitude v)

def rationalize (v : ℝ × ℝ) : ℝ × ℝ :=
match v with
| (x, y) => (x * sqrt 5 / 5, y * sqrt 5 / 5)

theorem unit_vector_parallel_to_a :
  ∃ u : ℝ × ℝ, (u = rationalize (normalize vector_a) ∨ u = -(rationalize (normalize vector_a))) :=
by
  sorry

end unit_vector_parallel_to_a_l727_727454


namespace length_of_longer_leg_of_smallest_triangle_l727_727060

theorem length_of_longer_leg_of_smallest_triangle (hyp_large : ℝ) (H1 : hyp_large = 16) : 
  let short_leg (hyp : ℝ) := hyp / 2,
      long_leg (hyp : ℝ) := (short_leg hyp) * Real.sqrt 3 
  in long_leg (long_leg (long_leg (long_leg (long_leg hyp_large)))) = (9 * Real.sqrt 3) :=
by 
  sorry

end length_of_longer_leg_of_smallest_triangle_l727_727060


namespace sqrt_720_simplified_l727_727990

theorem sqrt_720_simplified : (sqrt 720 = 12 * sqrt 5) :=
by
  -- The proof is omitted as per the instructions
  sorry

end sqrt_720_simplified_l727_727990


namespace harmonic_mean_1999_2001_is_2000_l727_727393

def harmonic_mean (a b : ℕ) : ℚ := (2 * a * b) / (a + b)

theorem harmonic_mean_1999_2001_is_2000 :
  abs (harmonic_mean 1999 2001 - 2000 : ℚ) < 1 := by
  -- Actual proof omitted
  sorry

end harmonic_mean_1999_2001_is_2000_l727_727393


namespace barrel_capacities_l727_727249

theorem barrel_capacities :
  ∃ (x : ℝ), let second_barrel := (3 / 4) * x in
              let third_barrel := (7 / 12) * x in
              third_barrel + 50 = x ∧ x = 120 ∧ second_barrel = 90 ∧ third_barrel = 70 :=
begin
  sorry
end

end barrel_capacities_l727_727249


namespace non_monotonic_piecewise_l727_727482

theorem non_monotonic_piecewise (a : ℝ) (f : ℝ → ℝ)
  (h : ∀ (x t : ℝ),
    (f x = if x ≤ t then (4 * a - 3) * x + (2 * a - 4) else (2 * x^3 - 6 * x)))
  : a ≤ 3 / 4 := 
sorry

end non_monotonic_piecewise_l727_727482


namespace total_sales_l727_727679

theorem total_sales (S : ℝ) (remitted : ℝ) : 
  (∀ S, remitted = S - (0.05 * 10000 + 0.04 * (S - 10000)) → remitted = 31100) → S = 32500 :=
by
  sorry

end total_sales_l727_727679


namespace train_length_l727_727729

theorem train_length :
  ∀ (speed_train_kmph speed_bike_kmph : ℝ)
    (time_seconds : ℝ),
    speed_train_kmph = 100 →
    speed_bike_kmph = 64 →
    time_seconds = 80 →
  let speed_train_mps := speed_train_kmph * 1000 / 3600 in
  let speed_bike_mps := speed_bike_kmph * 1000 / 3600 in
  let relative_speed_mps := speed_train_mps - speed_bike_mps in
  let length_train_m := relative_speed_mps * time_seconds in
  length_train_m = 800 :=
by
  intros
  rename_i speed_train_kmph speed_bike_kmph time_seconds h_train h_bike h_time
  let speed_train_mps := speed_train_kmph * 1000 / 3600
  let speed_bike_mps := speed_bike_kmph * 1000 / 3600
  let relative_speed_mps := speed_train_mps - speed_bike_mps
  let length_train_m := relative_speed_mps * time_seconds
  have h1 : speed_train_mps = 100 * 1000 / 3600, by rw h_train
  have h2 : speed_bike_mps = 64 * 1000 / 3600, by rw h_bike
  have h3 : time_seconds = 80, by rw h_time
  sorry

end train_length_l727_727729


namespace common_elements_lemma_l727_727453

-- Definitions
def is_prime (n : ℕ) : Prop := ∀ m ∣ n, m = 1 ∨ m = n
def r (x p : ℕ) : ℕ := x % p

noncomputable def inverse_mod (x p : ℕ) : ℕ := 
  if h : ∃ y, (x * y) % p = 1 then
    classical.some h 
  else 
    0

-- Conditions
axiom prime_p (p : ℕ) : is_prime p ∧ p > 2023 
axiom primes_less_than_bound (p : ℕ) (h : p > 2023) : 
  ∃ (ps : list ℕ), (∀ n ∈ ps, is_prime n ∧ n < (1/2 * p)^(1/4)) ∧ 
  nat.strongly_sorted (<) ps

-- Main statement
theorem common_elements_lemma (p : ℕ) (h_prime : is_prime p) (h_gt : p > 2023) 
  (ps : list ℕ) (h_ps : ∀ n ∈ ps, is_prime n ∧ n < (1/2 * p)^(1/4)) 
  (h_sorted : nat.strongly_sorted (<) ps) :
  ∀ (a b : ℕ), 0 < a → a < p → 0 < b → b < p →
  let qs := ps.map (λ pi, inverse_mod pi p) in
  (qs.map (λ q, r q p)).to_finset ∩ (qs.map (λ q, r (a * q + b) p)).to_finset ∈ {0, 1, 2, 3} := 
sorry

end common_elements_lemma_l727_727453


namespace max_value_of_expression_l727_727824

noncomputable def expression (x : ℝ) : ℝ :=
  x^6 / (x^10 + 3 * x^8 - 5 * x^6 + 15 * x^4 + 25)

theorem max_value_of_expression : ∃ x : ℝ, (expression x) = 1 / 17 :=
sorry

end max_value_of_expression_l727_727824


namespace find_const_s_l727_727559

noncomputable def g (x : ℝ) (a b c d : ℝ) := (x + 2 * a) * (x + 2 * b) * (x + 2 * c) * (x + 2 * d)

theorem find_const_s (a b c d : ℝ) (p q r s : ℝ) (h1 : 1 + p + q + r + s = 4041)
  (h2 : g 1 a b c d = 1 + p + q + r + s) :
  s = 3584 := 
sorry

end find_const_s_l727_727559


namespace gcd_12012_18018_l727_727816

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_12012_18018 : gcd 12012 18018 = 6006 := sorry

end gcd_12012_18018_l727_727816


namespace clive_money_l727_727396

noncomputable def clive_initial_money : ℝ  :=
  let total_olives := 80
  let olives_per_jar := 20
  let cost_per_jar := 1.5
  let change := 4
  let jars_needed := total_olives / olives_per_jar
  let total_cost := jars_needed * cost_per_jar
  total_cost + change

theorem clive_money (h1 : clive_initial_money = 10) : clive_initial_money = 10 :=
by sorry

end clive_money_l727_727396


namespace arithmetic_seq_8th_term_l727_727312

theorem arithmetic_seq_8th_term (a_1 a_30 : ℚ) (n : ℕ) (h1 : a_1 = 5) (h2 : a_30 = 80) (hn : n = 30) :
  let d := (a_30 - a_1) / (n - 1) in
  a_1 + 7 * d = 670 / 29 :=
by
  sorry

end arithmetic_seq_8th_term_l727_727312


namespace diane_trip_length_l727_727058

-- Define constants and conditions
def first_segment_fraction : ℚ := 1 / 4
def middle_segment_length : ℚ := 24
def last_segment_fraction : ℚ := 1 / 3

def total_trip_length (x : ℚ) : Prop :=
  (1 - first_segment_fraction - last_segment_fraction) * x = middle_segment_length

theorem diane_trip_length : ∃ x : ℚ, total_trip_length x ∧ x = 57.6 := by
  sorry

end diane_trip_length_l727_727058


namespace parabola_rhombus_proof_l727_727280

noncomputable def parabola_rhombus_condition (a b : ℝ) : Prop :=
  let x_intercepts_f : ℝ → ℝ → set ℝ :=
    λ a, {x | x^2 = 2 / a}
  let x_intercepts_g : ℝ → ℝ → set ℝ :=
    λ b, {x | x^2 = 6 / b}
  let vertices := (x_intercepts_f a (2 / a)).union (x_intercepts_g b (6 / b))
  let diagonals := (2 * (sqrt (2 / a)), 2 * (sqrt (6 / b)))
  4 = vertices.card ∧ 24 = (1 / 2) * diagonals.1 * diagonals.2

theorem parabola_rhombus_proof (a b : ℝ) (h : parabola_rhombus_condition a b) : a + b = 6 :=
sorry

end parabola_rhombus_proof_l727_727280


namespace initial_y_percentage_proof_l727_727346

variable (initial_volume : ℝ) (added_volume : ℝ) (initial_percentage_x : ℝ) (result_percentage_x : ℝ)

-- Conditions
def initial_volume_condition : Prop := initial_volume = 80
def added_volume_condition : Prop := added_volume = 20
def initial_percentage_x_condition : Prop := initial_percentage_x = 0.30
def result_percentage_x_condition : Prop := result_percentage_x = 0.44

-- Question
def initial_percentage_y (initial_volume added_volume initial_percentage_x result_percentage_x : ℝ) : ℝ :=
  1 - initial_percentage_x

-- Theorem
theorem initial_y_percentage_proof 
  (h1 : initial_volume_condition initial_volume)
  (h2 : added_volume_condition added_volume)
  (h3 : initial_percentage_x_condition initial_percentage_x)
  (h4 : result_percentage_x_condition result_percentage_x) :
  initial_percentage_y initial_volume added_volume initial_percentage_x result_percentage_x = 0.70 := 
sorry

end initial_y_percentage_proof_l727_727346


namespace count_irrationals_l727_727535

theorem count_irrationals : 
  let s := {22 / 7, 0, -Real.sqrt 2, 2 * Real.pi} 
  in s.to_finset.count (λ x, ¬x.is_rat) = 2 :=
by
  let s := {22 / 7, 0, -Real.sqrt 2, 2 * Real.pi}
  have h1: ¬(22 / 7).is_rat := by sorry
  have h2: (0).is_rat := by sorry
  have h3: ¬(-Real.sqrt 2).is_rat := by sorry
  have h4: ¬(2 * Real.pi).is_rat := by sorry
  exact calc
    {22 / 7, 0, -Real.sqrt 2, 2 * Real.pi}.to_finset.count (λ x, ¬x.is_rat) 
    = 2 : by sorry

end count_irrationals_l727_727535


namespace matrix_inv_correct_l727_727425

open Matrix

noncomputable def matrix_A : Matrix (Fin 2) (Fin 2) ℤ :=
  !![3, 4;
   -2, 9]

noncomputable def matrix_A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  !![
    (9 : ℚ)/35, (-4 : ℚ)/35;
    (2 : ℚ)/35, (3 : ℚ)/35]

theorem matrix_inv_correct : matrix_A⁻¹ = matrix_A_inv := by
  sorry

end matrix_inv_correct_l727_727425


namespace max_median_soda_purchased_l727_727716

theorem max_median_soda_purchased (total_cans : ℕ) (total_customers : ℕ)
  (h1 : total_cans = 310) (h2 : total_customers = 120) :
  ∃ m, (forall_each_customer_buys_at_least_one_can -> median_satisfaction m) :=
begin
  sorry
end

-- Definitions of helper functions or assumptions
def forall_each_customer_buys_at_least_one_can (purchases : list ℕ) : Prop :=
  ∀ p ∈ purchases, p ≥ 1

def median_satisfaction (m : ℕ) : Prop :=
  m = 5

end max_median_soda_purchased_l727_727716


namespace max_internal_reflections_l727_727363

theorem max_internal_reflections (n : ℕ) (angle_CDA : ℕ) (max_reflection_angle : ℕ):
  angle_CDA = 6 ∧ max_reflection_angle = 90 → n ≤ 15 := by
  intro h
  cases h with ha hb
  rw [ha, hb]
  sorry

end max_internal_reflections_l727_727363


namespace mul_mixed_number_l727_727005

theorem mul_mixed_number (a b : ℝ) (c : ℚ) (h : c = 9 + 2/5) : 
  7 * c = (65 + 4/5 : ℝ) :=
by 
  simp [h, mul_add, mul_div_cancel_left, rat.cast_add, rat.cast_mul, rat.cast_one]
  sorry

end mul_mixed_number_l727_727005


namespace problem1_l727_727344

theorem problem1 :
  8^(2/3) - real.sqrt ((real.sqrt 2 - 1)^2) + 2^(1/2) + (1/3)^0 - real.log 100 / real.log 10 = 4 := by
  sorry

end problem1_l727_727344


namespace line_intersects_segment_l727_727866

open Real

theorem line_intersects_segment (a: ℝ) :
  (∃ x y, a * x + y + 2 = 0 ∧ (x, y) ∈ itv (-2, 1) (3, 2)) ↔
  (-∞ < a ∧ a ≤ -4/3) ∨ (3/2 ≤ a ∧ a < ∞) := 
sorry

end line_intersects_segment_l727_727866


namespace fish_count_after_five_years_l727_727653

variable (start_fish : ℕ := 2) (additional_fish : ℕ := 2) (dying_fish : ℕ := 1) (years : ℕ := 5)

theorem fish_count_after_five_years : let total_fish := start_fish + (years * additional_fish) - (years * dying_fish) in
  total_fish = 7 :=
by
  have base_case: start_fish = 2 := by rfl
  have add_fish: additional_fish = 2 := by rfl
  have die_fish: dying_fish = 1 := by rfl
  have years_duration: years = 5 := by rfl
  have total_fish := start_fish + (years * additional_fish) - (years * dying_fish)
  sorry

end fish_count_after_five_years_l727_727653


namespace original_cube_volume_eq_216_l727_727660

theorem original_cube_volume_eq_216 (a : ℕ)
  (h1 : ∀ (a : ℕ), ∃ V_orig V_new : ℕ, 
    V_orig = a^3 ∧ 
    V_new = (a + 1) * (a + 1) * (a - 2) ∧ 
    V_orig = V_new + 10) : 
  a = 6 → a^3 = 216 := 
by
  sorry

end original_cube_volume_eq_216_l727_727660


namespace wombats_count_l727_727037

theorem wombats_count (W : ℕ) (H : 4 * W + 3 = 39) : W = 9 := 
sorry

end wombats_count_l727_727037


namespace coefficient_of_x2_in_expansion_l727_727913

theorem coefficient_of_x2_in_expansion : 
  coeff ((1 - X)^4 * (1 + X)^3) 2 = -3 :=
sorry

end coefficient_of_x2_in_expansion_l727_727913


namespace frog_jump_l727_727303

-- Define the type alias for coordinates on the chessboard
def Coord := (ℕ × ℕ)

-- Define the set L containing all valid coordinates on a 2k x 2k chessboard
def L (k : ℕ) : Set Coord := { p | 1 ≤ p.1 ∧ p.1 ≤ 2 * k ∧ 1 ≤ p.2 ∧ p.2 ≤ 2 * k }

-- Define the function f
def f (k : ℕ) (p : Coord) : Coord :=
  match p with
  | (i, j) => if i % 2 = 1 then
                if j <= k then (i + 1, j + k)
                else (i + 1, j - k)
              else
                if j <= k then (i - 1, j + k)
                else (i - 1, j - k)

-- Define the sets X and O
variable (X O : Set Coord)

-- Conditions on X and O: X and O are subsets of L
variable (k : ℕ)
variable (hX : X ⊆ L k)
variable (hO : O ⊆ L k)

-- Condition that for each (i, j) in X, f(i, j) is in O
variable (hInO : ∀ p ∈ X, f k p ∈ O)

-- Condition that f is injective
variable (hInjective : ∀ p1 p2 ∈ L k, f k p1 = f k p2 → p1 = p2)

-- The theorem to be proven
theorem frog_jump (m n : ℕ) (hX_card : X.card = m) (hO_card : O.card = n) : n ≥ m := by
  sorry

end frog_jump_l727_727303


namespace vector_identity_l727_727978

noncomputable theory

variables {R : Type*} [real R]

structure vec3 (R : Type*) := (i : R) (j : R) (k : R)

variables (a b c: R)
def v := vec3.mk a b c

def i : vec3 R := vec3.mk 1 0 0
def j : vec3 R := vec3.mk 0 1 0
def k : vec3 R := vec3.mk 0 0 1

def cross_product (u v : vec3 R) : vec3 R :=
{ i := u.j * v.k - u.k * v.j,
  j := u.k * v.i - u.i * v.k,
  k := u.i * v.j - u.j * v.i }

def dot_product (u v : vec3 R) : R := 
u.i * v.i + u.j * v.j + u.k * v.k

theorem vector_identity 
  (u v : vec3 R) :
  (i R) × ((v) × (2 • (i R)) + 
  (j R) × ((v) × (3 • (j R))) + 
  (k R) × ((v) × (4 • (k R))) = 0 := sorry

end vector_identity_l727_727978


namespace find_line_through_intersection_and_perpendicular_l727_727791

-- Definitions for the given conditions
def line1 (x y : ℝ) : Prop := 3 * x - 2 * y + 1 = 0
def line2 (x y : ℝ) : Prop := x + 3 * y + 4 = 0
def perpendicular (x y m : ℝ) : Prop := x + 3 * y + 4 = 0 ∧ 3 * x - y + m = 0

theorem find_line_through_intersection_and_perpendicular :
  ∃ m : ℝ, ∃ x y : ℝ, line1 x y ∧ line2 x y ∧ perpendicular x y m → 3 * x - y + 2 = 0 :=
by
  sorry

end find_line_through_intersection_and_perpendicular_l727_727791


namespace range_of_m_l727_727090

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 → (4^x + 2^x - m ≤ 0)) → 
  m ≥ 6 :=
by
  intro h
  have h_max : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → 4^x + 2^x ≤ 6 :=
    sorry
  specialize h 1
  linarith

end range_of_m_l727_727090


namespace quadratic_equation_roots_l727_727835

theorem quadratic_equation_roots (a b c : ℝ) (h_a_nonzero : a ≠ 0) 
  (h_roots : ∀ x, a * x^2 + b * x + c = 0 ↔ x = 1 ∨ x = -1) : 
  a + b + c = 0 ∧ b = 0 :=
by
  -- Using Vieta's formulas and the properties given, we should show:
  -- h_roots means the sum of roots = -(b/a) = 0 → b = 0
  -- and the product of roots = (c/a) = -1/a → c = -a
  -- Substituting these into ax^2 + bx + c = 0 should give us:
  -- a + b + c = 0 → we need to show both parts to complete the proof.
  sorry

end quadratic_equation_roots_l727_727835


namespace exists_m_divisible_by_1988_l727_727935

def f (x : ℕ) : ℕ := 3 * x + 2
def iter_function (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => f (iter_function n x)

theorem exists_m_divisible_by_1988 : ∃ m : ℕ, 1988 ∣ iter_function 100 m :=
by sorry

end exists_m_divisible_by_1988_l727_727935


namespace count_ways_to_express_5050_l727_727937

theorem count_ways_to_express_5050 : ∃ M : ℕ,
  M = Fintype.card {b : Fin 200 × Fin 200 × Fin 200 × Fin 200 // 125 * ↑b.1.1 + 25 * ↑b.1.2 + 5 * ↑b.1.3 + ↑b.2 = 5050} :=
by
  sorry

end count_ways_to_express_5050_l727_727937


namespace digit_2023_in_7_div_26_is_3_l727_727417

-- We define the decimal expansion of 7/26 as a repeating sequence of "269230769"
def repeating_block : string := "269230769"

-- Verify that the 2023rd digit in the sequence is "3"
theorem digit_2023_in_7_div_26_is_3 :
  (repeating_block.str.to_list.nth ((2023 % 9) - 1)).iget = '3' :=
by
  sorry

end digit_2023_in_7_div_26_is_3_l727_727417


namespace number_of_paintings_per_new_gallery_l727_727376

-- Define all the conditions as variables/constants
def pictures_original : Nat := 9
def new_galleries : Nat := 5
def pencils_per_picture : Nat := 4
def pencils_per_exhibition : Nat := 2
def total_pencils : Nat := 88

-- Define the proof problem in Lean
theorem number_of_paintings_per_new_gallery (pictures_original new_galleries pencils_per_picture pencils_per_exhibition total_pencils : Nat) :
(pictures_original = 9) → (new_galleries = 5) → (pencils_per_picture = 4) → (pencils_per_exhibition = 2) → (total_pencils = 88) → 
∃ (pictures_per_gallery : Nat), pictures_per_gallery = 2 :=
by
  intros
  sorry

end number_of_paintings_per_new_gallery_l727_727376


namespace correct_quotient_of_original_division_operation_l727_727898

theorem correct_quotient_of_original_division_operation 
  (incorrect_divisor correct_divisor incorrect_quotient : ℕ)
  (h1 : incorrect_divisor = 102)
  (h2 : correct_divisor = 201)
  (h3 : incorrect_quotient = 753)
  (h4 : ∃ k, k = incorrect_quotient * 3) :
  ∃ q, q = 1146 ∧ (correct_divisor * q = incorrect_divisor * (incorrect_quotient * 3)) :=
by
  sorry

end correct_quotient_of_original_division_operation_l727_727898


namespace chocolates_in_box_l727_727325

theorem chocolates_in_box (chocolates_with_nuts_chances : ℕ) (chocolates_without_nuts_chances : ℕ)
  (eaten_with_nuts_chances : ℝ) (eaten_without_nuts_chances : ℝ) (remaining_chocolates : ℕ)
  (half_chocolates : chocolates_with_nuts_chances = chocolates_without_nuts_chances)
  (eaten_with_nuts_ratio : eaten_with_nuts_chances = 0.80)
  (eaten_without_nuts_ratio : eaten_without_nuts_chances = 0.50)
  (remaining_chocolates_eq : remaining_chocolates = 28) :
  ∃ (total_chocolates : ℕ), total_chocolates = 80 :=
by
  use 80
  sorry

end chocolates_in_box_l727_727325


namespace proof_problem1_proof_problem2_l727_727395

noncomputable def problem1 : ℝ := (√27 - √12) * √(1 / 3)
theorem proof_problem1 : problem1 = 1 := 
by
  sorry

noncomputable def problem2 : ℝ := (√2023 + 1) * (√2023 - 1) + √8 / √2
theorem proof_problem2 : problem2 = 2024 := 
by
  sorry

end proof_problem1_proof_problem2_l727_727395


namespace replaced_person_weight_l727_727188

theorem replaced_person_weight :
  ∀ (avg_weight: ℝ), 
    10 * (avg_weight + 4) - 10 * avg_weight = 110 - 70 :=
by
  intros avg_weight
  sorry

end replaced_person_weight_l727_727188


namespace parallelogram_area_find_perpendicular_vector_l727_727495

noncomputable def pointA : (ℝ × ℝ × ℝ) := (0, 2, 3)
noncomputable def pointB : (ℝ × ℝ × ℝ) := (-2, 1, 6)
noncomputable def pointC : (ℝ × ℝ × ℝ) := (1, -1, 5)

noncomputable def vecAB := (pointB.1 - pointA.1, pointB.2 - pointA.2, pointB.3 - pointA.3)
noncomputable def vecAC := (pointC.1 - pointA.1, pointC.2 - pointA.2, pointC.3 - pointA.3)

noncomputable def crossProd (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

noncomputable def magnitude (u : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (u.1^2 + u.2^2 + u.3^2)

theorem parallelogram_area :
  magnitude (crossProd vecAB vecAC) = Real.sqrt 123 :=
  sorry

theorem find_perpendicular_vector (a : ℝ × ℝ × ℝ) :
  (vecAB.1 * a.1 + vecAB.2 * a.2 + vecAB.3 * a.3 = 0) ∧
  (vecAC.1 * a.1 + vecAC.2 * a.2 + vecAC.3 * a.3 = 0) ∧
  (magnitude a = 3) →
  (a = (1, 1, 1) ∨ a = (-1, -1, -1)) :=
  sorry

end parallelogram_area_find_perpendicular_vector_l727_727495


namespace linear_dependent_vectors_l727_727286

theorem linear_dependent_vectors (k : ℤ) :
  (∃ (a b : ℤ), (a ≠ 0 ∨ b ≠ 0) ∧ a * 2 + b * 4 = 0 ∧ a * 3 + b * k = 0) ↔ k = 6 :=
by
  sorry

end linear_dependent_vectors_l727_727286


namespace triangle_angle_condition_l727_727980

theorem triangle_angle_condition (a b h_3 : ℝ) (A C : ℝ) 
  (h : 1/(h_3^2) = 1/(a^2) + 1/(b^2)) :
  C = 90 ∨ |A - C| = 90 := 
sorry

end triangle_angle_condition_l727_727980


namespace vector_magnitude_rectangle_l727_727904

theorem vector_magnitude_rectangle {A B C D : ℝ} (rectangle_ABCD : true)
  (AB_eq_2 : AB = 2) (BC_eq_1 : BC = 1) :
  abs (AB + AD + AC) = 2 * sqrt 5 :=
sorry

end vector_magnitude_rectangle_l727_727904


namespace gcd_12012_18018_l727_727810

theorem gcd_12012_18018 : Int.gcd 12012 18018 = 6006 := 
by
  sorry

end gcd_12012_18018_l727_727810


namespace no_real_roots_quadratic_l727_727777

theorem no_real_roots_quadratic (a b c : ℝ) (h : a = 1 ∧ b = -4 ∧ c = 8) :
    (a ≠ 0) → (∀ x : ℝ, a * x^2 + b * x + c ≠ 0) :=
by
  sorry

end no_real_roots_quadratic_l727_727777


namespace polynomial_inequality_l727_727947

-- Definition of the polynomial and problem-related variables
noncomputable def polynomial (R : Type) [CommRing R] := R[X]
variables {R : Type} [CommRing R] (f : polynomial R) {n : ℕ} [hn : fact (n ≥ 2)] 
          (real_roots : ∀ x : R, is_root f x → x ∈ ℝ) (x0 : R)

-- Hypotheses
hypothesis (h_f_degree : f.natDegree = n) 
hypothesis (h_deriv_nonzero : f.derivative.eval x0 ≠ 0)

-- Conclusion: The existence of a root x1
theorem polynomial_inequality :
  ∃ x1 : R, is_root f x1 ∧ abs (x0 - x1) ≤ (n : R) * abs (f.eval x0 / f.derivative.eval x0) :=
sorry

end polynomial_inequality_l727_727947


namespace multiplication_with_mixed_number_l727_727014

-- Define mixed numbers as rational numbers for proper calculation
def mixed_to_rational (whole : ℕ) (num : ℕ) (den : ℕ) : ℚ :=
  whole + num / den

-- 7 * (9 + 2/5)
def lhs : ℚ := 7 * mixed_to_rational 9 2 5

-- 65 + 4/5
def rhs : ℚ := mixed_to_rational 65 4 5

theorem multiplication_with_mixed_number : lhs = rhs := by
  sorry

end multiplication_with_mixed_number_l727_727014


namespace time_spent_on_each_piece_l727_727602

def chairs : Nat := 7
def tables : Nat := 3
def total_time : Nat := 40
def total_pieces := chairs + tables
def time_per_piece := total_time / total_pieces

theorem time_spent_on_each_piece : time_per_piece = 4 :=
by
  sorry

end time_spent_on_each_piece_l727_727602


namespace gcd_12012_18018_l727_727806

theorem gcd_12012_18018 : Nat.gcd 12012 18018 = 6006 := sorry

end gcd_12012_18018_l727_727806


namespace finitely_many_cheesy_numbers_l727_727662

def is_cheesy (n : ℕ) : Prop :=
  let digits := n.digits 10
  let len := digits.length
  let sum := digits.sum
  (n : ℚ) / 10 ^ (len - 1) = sum / len

theorem finitely_many_cheesy_numbers : {n : ℕ | is_cheesy n}.finite :=
by
  sorry

end finitely_many_cheesy_numbers_l727_727662


namespace chocolate_count_l727_727330

theorem chocolate_count (C : ℝ) 
  (half_with_nuts : C/2) 
  (half_without_nuts : C/2) 
  (eaten_with_nuts : 0.8 * (C/2)) 
  (eaten_without_nuts : 0.5 * (C/2)) 
  (remaining_chocolates : 0.1 * C + 0.25 * C = 28) : 
  C = 80 :=
by
  sorry

end chocolate_count_l727_727330


namespace rocco_total_usd_l727_727259

def us_quarters := 4 * 8 * 0.25
def canadian_dimes := 6 * 12 * 0.10 * 0.8
def us_nickels := 9 * 10 * 0.05
def euro_cents := 5 * 15 * 0.01 * 1.18
def british_pence := 3 * 20 * 0.01 * 1.4
def japanese_yen := 2 * 10 * 1 * 0.0091
def mexican_pesos := 4 * 5 * 1 * 0.05

def total_usd := us_quarters + canadian_dimes + us_nickels + euro_cents + british_pence + japanese_yen + mexican_pesos

theorem rocco_total_usd : total_usd = 21.167 := by
  simp [us_quarters, canadian_dimes, us_nickels, euro_cents, british_pence, japanese_yen, mexican_pesos]
  sorry

end rocco_total_usd_l727_727259


namespace possible_periods_l727_727093

theorem possible_periods (r n : ℕ) (h_r : r = 5) (h_n : n = 12) : 
  ∃ p : ℕ, p ∈ {1, 2, 3, 4, 6, 12} :=
by
  sorry

end possible_periods_l727_727093


namespace total_animals_correct_l727_727225

def L := 10

def C := 2 * L + 4

def Merry_lambs := L
def Merry_cows := C
def Merry_pigs (P : ℕ) := P
def Brother_lambs := L + 3

def Brother_chickens (R : ℕ) := R * Brother_lambs
def Brother_goats (Q : ℕ) := 2 * Brother_lambs + Q

def Merry_total (P : ℕ) := Merry_lambs + Merry_cows + Merry_pigs P
def Brother_total (R Q : ℕ) := Brother_lambs + Brother_chickens R + Brother_goats Q

def Total_animals (P R Q : ℕ) := Merry_total P + Brother_total R Q

theorem total_animals_correct (P R Q : ℕ) : 
  Total_animals P R Q = 73 + P + R * 13 + Q := by
  sorry

end total_animals_correct_l727_727225


namespace bisect_AO_l727_727195

-- Let ABC be an acute-angled triangle, with AB > AC,
-- M be the midpoint of side BC,
-- P be an interior point such that ∠MAB = ∠PAC,
-- O, O₁, and O₂ be the circumcenters of triangles ABC, ABP, and ACP respectively.

variables {A B C M P O O₁ O₂ : Point}
variables (hABC : is_acute_triangle A B C)
variables (hAB_GT_AC : dist A B > dist A C)
variables (hM_midpoint : midpoint M B C)
variables (hP_condition : ∠ M A B = ∠ P A C)
variables (hO : is_circumcenter O A B C)
variables (hO₁ : is_circumcenter O₁ A B P)
variables (hO₂ : is_circumcenter O₂ A C P)

theorem bisect_AO (hABC : is_acute_triangle A B C)
  (hAB_GT_AC : dist A B > dist A C)
  (hM_midpoint : midpoint M B C)
  (hP_condition : ∠ M A B = ∠ P A C)
  (hO : is_circumcenter O A B C)
  (hO₁ : is_circumcenter O₁ A B P)
  (hO₂ : is_circumcenter O₂ A C P) : 
    bisects (line_through A O) (segment O₁ O₂) :=
sorry

end bisect_AO_l727_727195


namespace lena_more_than_nicole_l727_727546

theorem lena_more_than_nicole (L K N : ℕ) 
  (h1 : L = 23)
  (h2 : 4 * K = L + 7)
  (h3 : K = N - 6) : L - N = 10 := sorry

end lena_more_than_nicole_l727_727546


namespace radius_large_circle_l727_727304

/-- Let R be the radius of the large circle. Assume three circles of radius 2 are externally 
tangent to each other. Two of these circles are internally tangent to the larger circle, 
and the third circle is tangent to the larger circle both internally and externally. 
Prove that the radius of the large circle is 4 + 2 * sqrt 3. -/
theorem radius_large_circle (R : ℝ)
  (h1 : ∃ (C1 C2 C3 : ℝ × ℝ), 
    dist C1 C2 = 4 ∧ dist C2 C3 = 4 ∧ dist C3 C1 = 4 ∧ 
    (∃ (O : ℝ × ℝ), 
      (dist O C1 = R - 2) ∧ 
      (dist O C2 = R - 2) ∧ 
      (dist O C3 = R + 2) ∧ 
      (dist C1 C2 = 4) ∧ (dist C2 C3 = 4) ∧ (dist C3 C1 = 4))):
  R = 4 + 2 * Real.sqrt 3 := 
sorry

end radius_large_circle_l727_727304


namespace boxes_after_unpacking_weigh_twenty_kg_l727_727619

-- conditions
def initial_weight : ℝ := 100
def initial_books_weight : ℝ := 0.99 * initial_weight
def initial_non_books_weight : ℝ := initial_weight - initial_books_weight
def after_unpacking_books_fraction : ℝ := 0.95
def remaining_non_books_weight  : ℝ := initial_non_books_weight
def total_weight_after_unpacking : ℝ := remaining_non_books_weight / (1 - after_unpacking_books_fraction) 

-- question and answer
theorem boxes_after_unpacking_weigh_twenty_kg : total_weight_after_unpacking = 20 := 
by sorry

end boxes_after_unpacking_weigh_twenty_kg_l727_727619


namespace multiply_mixed_number_l727_727033

def mixed_to_improper_fraction (n : ℕ) (a b : ℕ) : ℚ :=
  n + a / b

def improper_to_mixed_number (q : ℚ) : ℕ × ℚ :=
  let n := q.natAbs
  let r := q.fract
  (n, r)

theorem multiply_mixed_number (x y n a b : ℕ) (h : y = mixed_to_improper_fraction n a b) :
  x * y = mixed_to_improper_fraction 65 4 5 :=
  sorry

end multiply_mixed_number_l727_727033


namespace gwen_money_difference_l727_727823

theorem gwen_money_difference:
  let money_from_grandparents : ℕ := 15
  let money_from_uncle : ℕ := 8
  money_from_grandparents - money_from_uncle = 7 :=
by
  sorry

end gwen_money_difference_l727_727823


namespace ratio_of_areas_ACP_BQA_l727_727915

open EuclideanGeometry

-- Define the geometric configuration
variables (A B C D P Q : Point)
  (is_square : square A B C D)
  (is_bisector_CAD : is_angle_bisector A C D P)
  (is_bisector_ABD : is_angle_bisector B A D Q)

-- Define the areas of triangles
def area_triangle (X Y Z : Point) : Real := sorry -- Placeholder for the area function

-- Lean statement for the proof problem
theorem ratio_of_areas_ACP_BQA 
  (h_square : is_square) 
  (h_bisector_CAD : is_bisector_CAD) 
  (h_bisector_ABD : is_bisector_ABD) :
  (area_triangle A C P) / (area_triangle B Q A) = 2 :=
sorry

end ratio_of_areas_ACP_BQA_l727_727915


namespace negation_of_universal_proposition_l727_727278

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ ∃ x : ℝ, |x| + x^2 < 0 :=
by
  sorry

end negation_of_universal_proposition_l727_727278


namespace increasing_intervals_l727_727089

noncomputable def f (x : ℝ) : ℝ := x^3 - (1 / 2) * x^2 - 2 * x + 5

theorem increasing_intervals :
  { x : ℝ | x < -2 / 3 } ∪ { x : ℝ | x > 1 } = { x : ℝ | deriv f x > 0 } :=
by
  sorry

end increasing_intervals_l727_727089


namespace area_sum_of_two_tris_eq_third_l727_727107

noncomputable theory

open EuclideanGeometry

variables {A B C P : Point} 
variables {M1 M2 M3 : Point}
variables (BC AC AB : Line) (ABC : Triangle)

-- Given a triangle ABC and a point P
def is_midpoint (M : Point) (X Y : Point) : Prop :=
  X -ᵥ M = M -ᵥ Y

-- M1, M2, M3 are the midpoints of the sides BC, AC, AB respectively
variables (M1_mid : is_midpoint M1 B C)
variables (M2_mid : is_midpoint M2 A C)
variables (M3_mid : is_midpoint M3 A B)

-- The statement to prove
theorem area_sum_of_two_tris_eq_third :
  let t1 := mk_triangle A M1 P in
  let t2 := mk_triangle B M2 P in
  let t3 := mk_triangle C M3 P in
  area t1 + area t2 = area t3 ∧
  area t2 + area t3 = area t1 ∧
  area t3 + area t1 = area t2 :=
sorry

end area_sum_of_two_tris_eq_third_l727_727107


namespace parallel_lines_slope_l727_727474

theorem parallel_lines_slope {m : ℝ} : 
  (∃ m, (∀ x y : ℝ, 3 * x + 4 * y - 3 = 0 → 6 * x + m * y + 14 = 0)) ↔ m = 8 :=
by
  sorry

end parallel_lines_slope_l727_727474


namespace percent_singles_l727_727781

theorem percent_singles :
  ∀ (total_hits home_runs triples doubles : ℕ),
  total_hits = 50 →
  home_runs = 2 →
  triples = 4 →
  doubles = 10 →
  (total_hits - (home_runs + triples + doubles)) * 100 / total_hits = 68 :=
by
  sorry

end percent_singles_l727_727781


namespace land_percentage_relationship_l727_727298

variable {V : ℝ} -- Total taxable value of all land in the village
variable {x y z : ℝ} -- Percentages of Mr. William's land in types A, B, C

-- Conditions
axiom total_tax_collected : 0.80 * (x / 100 * V) + 0.90 * (y / 100 * V) + 0.95 * (z / 100 * V) = 3840
axiom mr_william_tax : 0.80 * (x / 100 * V) + 0.90 * (y / 100 * V) + 0.95 * (z / 100 * V) = 480

-- Prove the relationship
theorem land_percentage_relationship : (0.80 * x + 0.90 * y + 0.95 * z = 48000 / V) → (x + y + z = 100) := by
  sorry

end land_percentage_relationship_l727_727298


namespace multiply_mixed_number_l727_727028

def mixed_to_improper_fraction (n : ℕ) (a b : ℕ) : ℚ :=
  n + a / b

def improper_to_mixed_number (q : ℚ) : ℕ × ℚ :=
  let n := q.natAbs
  let r := q.fract
  (n, r)

theorem multiply_mixed_number (x y n a b : ℕ) (h : y = mixed_to_improper_fraction n a b) :
  x * y = mixed_to_improper_fraction 65 4 5 :=
  sorry

end multiply_mixed_number_l727_727028


namespace train_crossing_time_l727_727883

theorem train_crossing_time
  (train_length : ℕ)
  (bridge_length : ℕ)
  (train_speed_kmph : ℕ) :
  train_length = 100 →
  bridge_length = 160 →
  train_speed_kmph = 36 →
  (train_length + bridge_length) / ((train_speed_kmph * 1000) / 3600) = 26 := 
by 
intros h1 h2 h3
-- converting train_speed_kmph to mps
have train_speed_mps : ℕ := (train_speed_kmph * 1000) / 3600
-- total distance = train_length + bridge_length
have total_distance : ℕ := train_length + bridge_length
-- time to cross = total_distance / train_speed_mps
have time_to_cross : ℕ := total_distance / train_speed_mps
-- given train_length = 100, bridge_length = 160, and train_speed_kmph = 36
rw [h1, h2, h3] at *
-- now total_distance = 100 + 160 = 260 and train_speed_mps = 10
have train_speed_mps : ℕ := 10
have time_to_cross : ℕ := 260 / 10
rw [h4, h5] at *
-- which gives time_to_cross = 26
have h6 : time_to_cross = 26 := by norm_num
exact h6

end train_crossing_time_l727_727883


namespace xy_sum_value_l727_727091

theorem xy_sum_value (x y : ℝ) (h1 : x^2 + x * y + 2 * y = 10) (h2 : y^2 + x * y + 2 * x = 14) :
  x + y = -6 ∨ x + y = 4 :=
sorry

end xy_sum_value_l727_727091


namespace each_bug_ate_1_5_flowers_l727_727585

-- Define the conditions given in the problem
def bugs : ℝ := 2.0
def flowers : ℝ := 3.0

-- The goal is to prove that the number of flowers each bug ate is 1.5
theorem each_bug_ate_1_5_flowers : (flowers / bugs) = 1.5 :=
by
  sorry

end each_bug_ate_1_5_flowers_l727_727585


namespace dried_grapes_weight_l727_727447

def fresh_grapes_weight : ℝ := 30
def fresh_grapes_water_percentage : ℝ := 0.60
def dried_grapes_water_percentage : ℝ := 0.20

theorem dried_grapes_weight :
  let non_water_content := fresh_grapes_weight * (1 - fresh_grapes_water_percentage)
  let dried_grapes := non_water_content / (1 - dried_grapes_water_percentage)
  dried_grapes = 15 :=
by
  let non_water_content := fresh_grapes_weight * (1 - fresh_grapes_water_percentage)
  let dried_grapes := non_water_content / (1 - dried_grapes_water_percentage)
  show dried_grapes = 15
  sorry

end dried_grapes_weight_l727_727447


namespace find_first_number_in_second_pair_l727_727267

theorem find_first_number_in_second_pair:
  ∃ (a1 b1 a2 b2 a3 b3 a4 b4 a5 b5 : ℕ),
    a1 + b1 - 1 = 12 ∧ 
    a2 + b2 - 1 = 16 ∧ 
    a3 + b3 - 1 = 10 ∧ 
    a4 + b4 - 1 = 14 ∧ 
    a5 + b5 - 1 = 5 ∧ 
    a2 = 8 :=
begin
  sorry
end

end find_first_number_in_second_pair_l727_727267


namespace club_voyager_probability_l727_727397

-- Defining the problem statement in Lean
theorem club_voyager_probability :
  ∑ (W L T : ℕ) in { (W, L, T) | W > L ∧ W + L + T = 5 }
      (multinomial (W, L, T) * (1 / 2)^W * (1 / 2)^L * (1 / 6)^T) =
  1061 / 2304 :=
sorry

end club_voyager_probability_l727_727397


namespace line_y2_not_pass_second_quadrant_l727_727180

theorem line_y2_not_pass_second_quadrant {a b : ℝ} (h1 : a < 0) (h2 : b > 0) :
  ¬∃ x : ℝ, x < 0 ∧ bx + a > 0 :=
by
  sorry

end line_y2_not_pass_second_quadrant_l727_727180


namespace family_chocolate_chip_count_l727_727208

theorem family_chocolate_chip_count
  (batch_cookies : ℕ)
  (total_people : ℕ)
  (batches : ℕ)
  (choco_per_cookie : ℕ)
  (cookie_total : ℕ := batch_cookies * batches)
  (cookies_per_person : ℕ := cookie_total / total_people)
  (choco_per_person : ℕ := cookies_per_person * choco_per_cookie)
  (h1 : batch_cookies = 12)
  (h2 : total_people = 4)
  (h3 : batches = 3)
  (h4 : choco_per_cookie = 2)
  : choco_per_person = 18 := 
by sorry

end family_chocolate_chip_count_l727_727208


namespace blackjack_payout_ratio_l727_727701

theorem blackjack_payout_ratio (total_payout original_bet : ℝ) (h1 : total_payout = 60) (h2 : original_bet = 40):
  total_payout - original_bet = (1 / 2) * original_bet :=
by
  sorry

end blackjack_payout_ratio_l727_727701


namespace gcd_7654321_6789012_l727_727392

theorem gcd_7654321_6789012 : Nat.gcd 7654321 6789012 = 3 := by
  sorry

end gcd_7654321_6789012_l727_727392


namespace minimum_knights_in_tournament_l727_727714

def knights_tournament : Prop :=
  ∃ (N : ℕ), (∀ (x : ℕ), x = N / 4 →
    ∃ (k : ℕ), k = (3 * x - 1) / 7 → N = 20)

theorem minimum_knights_in_tournament : knights_tournament :=
  sorry

end minimum_knights_in_tournament_l727_727714


namespace distribute_stickers_l727_727151

theorem distribute_stickers (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 5) : 
  ∃ C, C = 30 ∧ (C = (multiset.partition (range n) k).length) := 
  by 
    sorry

end distribute_stickers_l727_727151


namespace gcd_12012_18018_l727_727804

theorem gcd_12012_18018 : Nat.gcd 12012 18018 = 6006 := sorry

end gcd_12012_18018_l727_727804


namespace border_collie_catches_ball_in_32_seconds_l727_727747

noncomputable def time_to_catch_ball (v_ball : ℕ) (t_ball : ℕ) (v_collie : ℕ) : ℕ := 
  (v_ball * t_ball) / v_collie

theorem border_collie_catches_ball_in_32_seconds :
  time_to_catch_ball 20 8 5 = 32 :=
by
  sorry

end border_collie_catches_ball_in_32_seconds_l727_727747


namespace segment_length_circumcircle_radius_l727_727688

noncomputable def triangle_contains_median_and_bisectors
  (A B C M D E P : Point)
  (BM : Segment)
  (MD ME : AngleBisector)
  (BP MP : ℝ)
  (conditions : TriangleContainsMedianAndBisectors A B C M D E P BM MD ME BP MP) : Prop :=
  true -- Placeholder for indicating the condition is met

theorem segment_length (A B C M D E P : Point)
  (conditions : TriangleContainsMedianAndBisectors A B C M D E P (Segment.mk A B) (AngleBisector.mk A M D) (AngleBisector.mk C M E) 2 4) :
  SegmentLength (Segment.mk D E) = 8 := 
sorry

theorem circumcircle_radius (A B C M D E P : Point)
  (conditions : TriangleContainsMedianAndBisectors A B C M D E P (Segment.mk A B) (AngleBisector.mk A M D) (AngleBisector.mk C M E) 2 4) :
  CircumcircleRadius (Quadrilateral.mk A D E C) = 2 * Real.sqrt 85 :=
sorry

end segment_length_circumcircle_radius_l727_727688


namespace greatest_number_of_factors_l727_727054

theorem greatest_number_of_factors (b n : ℕ) (hb : 1 ≤ b ∧ b ≤ 20) (hn : n = 10) :
  ∃ k, k = 231 ∧ k = (b^n).factors.card :=
begin
  sorry
end

end greatest_number_of_factors_l727_727054


namespace find_parametric_eq_l727_727197

noncomputable def parametric_eq_line (t : ℝ) : ℝ × ℝ :=
  (⟨ (sqrt 2) / 2 * t, -1 + (sqrt 2) / 2 * t ⟩ : ℝ × ℝ)

theorem find_parametric_eq (t : ℝ) : 
  ∃ l : ℝ × ℝ → Prop, 
  (l (parametric_eq_line t)) ∧
  (∃ a : ℝ, a = 3 ∧ 
  sqrt 2 * (parametric_eq_line t).1 * (cos ((polar_coord (parametric_eq_line t)).2 + π/4)) = 1 ∧ 
  (polar_coord (parametric_eq_line t)).1 = 2 * a * cos ((polar_coord (parametric_eq_line t)).2) ∧ 
  M = (0, -1) ∧ 
  norm(P - Q)^2 = 4 * norm(M - P) * norm(M - Q)) := 
sorry

end find_parametric_eq_l727_727197


namespace height_of_water_l727_727357

-- Definitions of variables
def radius : ℝ := 20
def height : ℝ := 100
def fill_percentage : ℝ := 0.30

-- Height of water in the tank, expressed in the form a * ∛b
-- and prove a + b = 31
theorem height_of_water
  (r : ℝ) (h : ℝ) (fp : ℝ)
  (a b : ℕ) (positive_b : b > 0) (not_cube_rt_b : ∀ k : ℕ, k > 1 → b ≠ k^3) :
  r = radius → h = height → fp = fill_percentage → a * (real.cbrt b) = 30 → a + b = 31 :=
by
  intros _ _ _ _ _ _ _
  sorry

end height_of_water_l727_727357


namespace f_increasing_f_odd_value_of_a_f_range_when_odd_l727_727111

-- Definition of the function
def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

-- (I) Prove that f(x) is an increasing function when a is any real number
theorem f_increasing (a : ℝ) : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂ :=
by sorry

-- (II) Determine the value of a such that f(x) is an odd function
theorem f_odd_value_of_a : ∀ a : ℝ, (∀ x : ℝ, f a (-x) = -f a x) → a = 1 :=
by sorry

-- (III) Given a = 1, find the range of f(x)
theorem f_range_when_odd : set.range (λ x : ℝ, f 1 x) = set.Ioo (-1 : ℝ) (1 : ℝ) :=
by sorry

end f_increasing_f_odd_value_of_a_f_range_when_odd_l727_727111


namespace blue_marbles_in_bag_l727_727350

theorem blue_marbles_in_bag
  (B W : ℕ)
  (h1 : B + W + 9 = 20)
  (h2 : (9 + W).to_rat / 20 = 3 / 4) :
  B = 5 :=
sorry

end blue_marbles_in_bag_l727_727350


namespace avg_two_expressions_l727_727175

theorem avg_two_expressions (a : ℝ) (h : ((2 * a + 16) + (3 * a - 8)) / 2 = 84) : a = 32 := sorry

end avg_two_expressions_l727_727175


namespace inverse_proportional_t_no_linear_function_2k_times_quadratic_function_5_times_l727_727112

-- Proof Problem 1
theorem inverse_proportional_t (t : ℝ) (h1 : 1 ≤ t ∧ t ≤ 2023) : t = 1 :=
sorry

-- Proof Problem 2
theorem no_linear_function_2k_times (k : ℝ) (h_pos : 0 < k) : ¬ ∃ a b : ℝ, (a < b) ∧ (∀ x, a ≤ x ∧ x ≤ b → (2 * k * a ≤ k * x + 2 ∧ k * x + 2 ≤ 2 * k * b)) :=
sorry

-- Proof Problem 3
theorem quadratic_function_5_times (a b : ℝ) (h_ab : a < b) (h_quad : ∀ x, a ≤ x ∧ x ≤ b → (5 * a ≤ x^2 - 4 * x - 7 ∧ x^2 - 4 * x - 7 ≤ 5 * b)) :
  (a = -2 ∧ b = 1) ∨ (a = -(11/5) ∧ b = (9 + Real.sqrt 109) / 2) :=
sorry

end inverse_proportional_t_no_linear_function_2k_times_quadratic_function_5_times_l727_727112


namespace proof_f_f7_eq_2_l727_727476

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def f (x : ℝ) : ℝ := if x < 0 then log (1 - x) / log 2 else -log (1 + x) / log 2 -- defining a piecewise function for it to be defined for all real x

theorem proof_f_f7_eq_2 : 
  is_odd_function f →
  (∀ x : ℝ, x < 0 → f x = log (1 - x) / log 2) →
  f (f 7) = 2 :=
by
  intros is_odd_function_f f_neg_def
  sorry

end proof_f_f7_eq_2_l727_727476


namespace price_of_each_rose_l727_727742

def number_of_roses_started (roses : ℕ) : Prop := roses = 9
def number_of_roses_left (roses : ℕ) : Prop := roses = 4
def amount_earned (money : ℕ) : Prop := money = 35
def selling_price_per_rose (price : ℕ) : Prop := price = 7

theorem price_of_each_rose 
  (initial_roses sold_roses left_roses total_money price_per_rose : ℕ)
  (h1 : number_of_roses_started initial_roses)
  (h2 : number_of_roses_left left_roses)
  (h3 : amount_earned total_money)
  (h4 : initial_roses - left_roses = sold_roses)
  (h5 : total_money / sold_roses = price_per_rose) :
  selling_price_per_rose price_per_rose := 
by
  sorry

end price_of_each_rose_l727_727742


namespace range_of_a_l727_727599

def condition1 (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0

def condition2 (a : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → (3 - 2 * a)^x < (3 - 2 * a)^y

def exclusive_or (p q : Prop) : Prop :=
  (p ∧ ¬q) ∨ (¬p ∧ q)

theorem range_of_a (a : ℝ) :
  exclusive_or (condition1 a) (condition2 a) → (1 ≤ a ∧ a < 2) ∨ a ≤ -2 :=
by
  -- Proof omitted
  sorry

end range_of_a_l727_727599


namespace complex_fraction_in_first_quadrant_l727_727910

-- Define a complex number
def complex_2_minus_i : ℂ := 2 - complex.I
def complex_1_minus_3i : ℂ := 1 - 3 * complex.I

-- Calculate the fraction and its real and imaginary parts
def complex_fraction : ℂ :=
  (complex_2_minus_i * (1 + 3 * complex.I)) /
  (complex_1_minus_3i * (1 + 3 * complex.I))

-- Assert that the complex number is in the first quadrant
theorem complex_fraction_in_first_quadrant :
  complex_fraction.re > 0 ∧ complex_fraction.im > 0 := by
  sorry

end complex_fraction_in_first_quadrant_l727_727910


namespace discount_percentage_l727_727718

-- Definitions from the conditions
def cost_price := 100
def markup_percentage := 40 / 100
def profit_percentage := 19 / 100

-- The question translated to Lean 4 statement format
theorem discount_percentage : 
  let marked_price := cost_price * (1 + markup_percentage) in
  let selling_price := cost_price * (1 + profit_percentage) in
  let discount := marked_price - selling_price in
  (discount / marked_price) * 100 = 15 := 
by 
  sorry

end discount_percentage_l727_727718


namespace bijection_three_progression_no_2003_progression_l727_727227

theorem bijection_three_progression (f : ℕ → ℕ) (bij : Function.Bijective f) :
  ∃ (a0 a1 a2 : ℕ), a0 < a1 ∧ a1 < a2 ∧ a1 - a0 = a2 - a1 ∧ f a0 < f a1 ∧ f a1 < f a2 :=
begin
  sorry
end

theorem no_2003_progression (f : ℕ → ℕ) (bij : Function.Bijective f) :
  ¬(∃ (a : Fin 2003 → ℕ), (∀ i j : Fin 2003, i < j → a i < a j) ∧
  (∀ i, a (Fin.succ i) - a i = a 1 - a 0) ∧ ∀ i j : Fin 2003, i < j → f (a i) < f (a j)) :=
begin
  sorry
end

end bijection_three_progression_no_2003_progression_l727_727227


namespace ratio_d_c_l727_727893

theorem ratio_d_c (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hc : c ≠ 0) 
  (h1 : 8 * x - 5 * y = c) (h2 : 10 * y - 16 * x = d) : d / c = -2 :=
by
  sorry

end ratio_d_c_l727_727893


namespace sibling_discount_is_correct_l727_727658

-- Defining the given conditions
def tuition_per_person : ℕ := 45
def total_cost_with_discount : ℕ := 75

-- Defining the calculation of sibling discount
def sibling_discount : ℕ :=
  let original_cost := 2 * tuition_per_person
  let discount := original_cost - total_cost_with_discount
  discount

-- Statement to prove
theorem sibling_discount_is_correct : sibling_discount = 15 :=
by
  unfold sibling_discount
  simp
  sorry

end sibling_discount_is_correct_l727_727658


namespace find_a_from_inequality_solution_set_l727_727182

theorem find_a_from_inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, (x^2 - a*x + 4 < 0) ↔ (1 < x ∧ x < 4)) -> a = 5 :=
by
  intro h
  sorry

end find_a_from_inequality_solution_set_l727_727182


namespace distance_from_intersection_to_asymptote_l727_727868

theorem distance_from_intersection_to_asymptote (a b : ℝ) (l : ℝ → ℝ) (P : ℝ × ℝ)
    (h1 : a > 0)
    (h2 : ∀ x y, y = l x → y^2 = 8 * a * x)
    (h3 : ∀ x y, y = l x → y = x - 2 * a)
    (h4 : ∀ x, let y := l x in 16 = Real.sqrt (1 + 1) * Real.sqrt (144 * a ^ 2 - 16 * a ^ 2))
    (h5 : let k := Real.sqrt 3 in ∀ x y, y^2 = 8 * a * x ↔ ∃ b c, b = Real.sqrt 3 ∧ (y = k * x - a * k))
    (h6 : C₂: ∀ x y : ℝ, (x^2/a^2) - (y^2/b^2) = 1)
    (h7 : x = -2 → let c := 2 in b = Real.sqrt 3)
  : let d := abs (2 * a) / Real.sqrt (b^2 + a^2) in d = 1 := 
sorry

end distance_from_intersection_to_asymptote_l727_727868


namespace find_sum_of_abcd_l727_727779

theorem find_sum_of_abcd (a b c d : ℚ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 10) :
  a + b + c + d = -26 / 3 :=
sorry

end find_sum_of_abcd_l727_727779


namespace minimizes_sum_of_sequence_l727_727871

noncomputable def sequence (n : ℕ) : ℤ := n^2 - 11 * n - 12

noncomputable def S (n : ℕ) : ℤ := ∑ k in Finset.range n, sequence k

def n_minimizes_S (n : ℕ) : Prop := ∀ m : ℕ, S n ≤ S m

theorem minimizes_sum_of_sequence : n_minimizes_S 11 ∨ n_minimizes_S 12 :=
sorry

end minimizes_sum_of_sequence_l727_727871


namespace least_number_to_subtract_l727_727685

theorem least_number_to_subtract (n : ℕ) (h : n = 42739) : 
    ∃ k, k = 4 ∧ (n - k) % 15 = 0 := by
  sorry

end least_number_to_subtract_l727_727685


namespace minimum_triangle_perimeter_l727_727958

def fractional_part (x : ℚ) : ℚ := x - ⌊x⌋

theorem minimum_triangle_perimeter (l m n : ℕ) (h1 : l > m) (h2 : m > n)
  (h3 : fractional_part (3^l / 10^4) = fractional_part (3^m / 10^4)) 
  (h4 : fractional_part (3^m / 10^4) = fractional_part (3^n / 10^4)) :
   l + m + n = 3003 := 
sorry

end minimum_triangle_perimeter_l727_727958


namespace wrapping_paper_area_l727_727710

variable (l w h : ℝ)
variable (l_gt_w : l > w)

def area_wrapping_paper (l w h : ℝ) : ℝ :=
  3 * (l + w) * h

theorem wrapping_paper_area :
  area_wrapping_paper l w h = 3 * (l + w) * h :=
sorry

end wrapping_paper_area_l727_727710


namespace base7_divisibility_l727_727444

/-
Given a base-7 number represented as 3dd6_7, where the first digit is 3, the last digit is 6, 
and the middle two digits are equal to d, prove that the base-7 digit d which makes this 
number divisible by 13 is 5.
-/
theorem base7_divisibility (d : ℕ) (hdig : d ∈ {0, 1, 2, 3, 4, 5, 6}) : 
  (3 * 7^3 + d * 7^2 + d * 7 + 6) % 13 = 0 ↔ d = 5 := 
sorry

end base7_divisibility_l727_727444


namespace algebra_inequality_l727_727955

theorem algebra_inequality (n : ℕ) (r : ℕ → ℝ) (h : ∀ i, 1 ≤ r i) :
  (∑ j in finset.range n, 1 / (r j + 1)) ≥ n / ((finset.range n).prod (λ j, r j) ^ (1 / n) + 1) :=
sorry

end algebra_inequality_l727_727955


namespace simplify_fraction_l727_727400

variable (x : ℕ)

theorem simplify_fraction (h : x = 3) : (x^10 + 15 * x^5 + 125) / (x^5 + 5) = 248 + 25 / 62 := by
  sorry

end simplify_fraction_l727_727400


namespace math_problem_l727_727455

theorem math_problem
  (a b c : ℤ)
  (h1 : sqrt (a - 3) + sqrt (3 - a) = 0)
  (h2 : cbrt (3 * b - 4) = 2)
  (h3 : c = int.floor (sqrt 6)) :
  a = 3 ∧ b = 4 ∧ c = 2 ∧ sqrt (a + 6 * b - c) = 5 :=
begin
  sorry
end

end math_problem_l727_727455


namespace angle_B44_B45_B43_65_l727_727556

-- Define the points and their relationships
def point : Type := ℝ × ℝ

variables (B : ℕ → point)

-- Define the condition that B_1B_2 = B_1B_3 (isosceles triangle)
def is_isosceles (B1 B2 B3 : point) : Prop :=
  dist B1 B2 = dist B1 B3

-- Define the condition that angle at B1 is 50 degrees
def angle_at_B1_50 (B1 B2 B3 : point) : Prop :=
  ∠ B2 B1 B3 = 50

-- Define the condition that B_{n+3} is the midpoint of B_nB_{n+2}
def is_midpoint (A B C : point) : Prop :=
  C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def B_n_midpoint (B : ℕ → point) : Prop :=
  ∀ n : ℕ, B (n + 3) = ((B n).1 + (B (n + 2)).1) / 2, ((B n).2 + (B (n + 2)).2) / 2

-- Final statement: angle B44B45B43 = 65 degrees
theorem angle_B44_B45_B43_65
  (h_isosceles : is_isosceles (B 1) (B 2) (B 3))
  (h_angle_B1_50 : angle_at_B1_50 (B 1) (B 2) (B 3))
  (h_midpoint : B_n_midpoint B) :
  ∠ B 44 B 45 B 43 = 65 :=
sorry

end angle_B44_B45_B43_65_l727_727556


namespace square_of_other_leg_l727_727191

variable {R : Type} [CommRing R]

theorem square_of_other_leg (a b c : R) (h1 : a^2 + b^2 = c^2) (h2 : c = a + 2) : b^2 = 4 * a + 4 :=
by
  sorry

end square_of_other_leg_l727_727191


namespace clock_hands_angle_seventy_degrees_l727_727075

theorem clock_hands_angle_seventy_degrees (t : ℝ) (h : t ≥ 0 ∧ t ≤ 60):
    let hour_angle := 210 + 30 * (t / 60)
    let minute_angle := 360 * (t / 60)
    let angle := abs (hour_angle - minute_angle)
    (angle = 70 ∨ angle = 290) ↔ (t = 25 ∨ t = 52) :=
by apply sorry

end clock_hands_angle_seventy_degrees_l727_727075


namespace problem_two_l727_727133

def f (x : ℝ) : ℝ := |x + 2| + |x - 2|

theorem problem_two (m n : ℝ) (A : set ℝ) 
  (hA : A = {x | -3 ≤ x ∧ x ≤ 3})
  (hm : m ∈ A) (hn : n ∈ A) 
  : |(1 / 3) * m - (1 / 2) * n| ≤ 5 / 2 := 
sorry

end problem_two_l727_727133


namespace find_f_log4_9_value_l727_727114

-- Define the function and properties
def f (x : ℝ) : ℝ := if x < 0 then 2^x else sorry

lemma even_function (x : ℝ) : f x = f (-x) := sorry

theorem find_f_log4_9_value : f (Real.log 9 / Real.log 4) = 1 / 3 := sorry

end find_f_log4_9_value_l727_727114


namespace gcd_of_12012_and_18018_l727_727801

theorem gcd_of_12012_and_18018 : Int.gcd 12012 18018 = 6006 := 
by
  -- Here we are assuming the factorization given in the conditions
  have h₁ : 12012 = 12 * 1001 := sorry
  have h₂ : 18018 = 18 * 1001 := sorry
  have gcd_12_18 : Int.gcd 12 18 = 6 := sorry
  -- This sorry will be replaced by the actual proof involving the above conditions to conclude the stated theorem
  sorry

end gcd_of_12012_and_18018_l727_727801


namespace vertical_asymptote_l727_727165

def f (x : ℝ) : ℝ := (x^2 + 3 * x + 4) / (x - 5)

theorem vertical_asymptote : ∃ x : ℝ, x = 5 ∧ (f x = (x^2 + 3 * x + 4) / (x - 5)) ∧ ¬ (x^2 + 3 * x + 4 = 0) :=
by
  sorry

end vertical_asymptote_l727_727165


namespace fish_pond_estimate_l727_727194

variable (N : ℕ)
variable (total_first_catch total_second_catch marked_in_first_catch marked_in_second_catch : ℕ)

/-- Estimate the total number of fish in the pond -/
theorem fish_pond_estimate
  (h1 : total_first_catch = 100)
  (h2 : total_second_catch = 120)
  (h3 : marked_in_first_catch = 100)
  (h4 : marked_in_second_catch = 15)
  (h5 : (marked_in_second_catch : ℚ) / total_second_catch = (marked_in_first_catch : ℚ) / N) :
  N = 800 := 
sorry

end fish_pond_estimate_l727_727194


namespace largest_divisor_of_square_divisible_by_24_l727_727644

theorem largest_divisor_of_square_divisible_by_24 (n : ℕ) (h₁ : n > 0) (h₂ : 24 ∣ n^2) (h₃ : ∀ k : ℕ, k ∣ n → k ≤ 8) : n = 24 := 
sorry

end largest_divisor_of_square_divisible_by_24_l727_727644


namespace jude_chairs_expense_correct_l727_727542

def judes_expense 
  (cost_table : ℕ) 
  (cost_plate_set : ℕ) 
  (num_plate_sets : ℕ) 
  (total_cash_given : ℕ) 
  (change_received : ℕ) 
  (total_cash_spent : ℕ) 
  (chairs_expense : ℕ) : Prop := 
  cost_table = 50 ∧ 
  cost_plate_set = 20 ∧ 
  num_plate_sets = 2 ∧ 
  total_cash_given = 130 ∧ 
  change_received = 4 ∧ 
  total_cash_spent = total_cash_given - change_received ∧
  total_cash_spent = 126 ∧
  chairs_expense = total_cash_spent - (cost_table + (cost_plate_set * num_plate_sets)) 

theorem jude_chairs_expense_correct : judes_expense 50 20 2 130 4 126 36 :=
by {
  unfold judes_expense,
  split, refl,
  split, refl,
  split, refl,
  split, refl,
  split, refl,
  split, linarith,
  split, refl,
  linarith,
}

end jude_chairs_expense_correct_l727_727542


namespace Problem_7_6_A_Problem_7_6_B_Problem_7_6_C_Problem_7_6_D_l727_727971

-- A) Proof that there are 7 mines for the first configuration
theorem Problem_7_6_A (field : Matrix (Fin 9) (Fin 6) ℕ) (numbers : List (ℕ × (Fin 9 × Fin 6))) :
  Minesweeper.countMines field numbers = 7 := 
sorry

-- B) Proof that there are 8 mines for the second configuration
theorem Problem_7_6_B (field : Matrix (Fin 9) (Fin 6) ℕ) (numbers : List (ℕ × (Fin 9 × Fin 6))) :
  Minesweeper.countMines field numbers = 8 := 
sorry

-- C) Proof that there are 9 mines for the third configuration
theorem Problem_7_6_C (field : Matrix (Fin 9) (Fin 6) ℕ) (numbers : List (ℕ × (Fin 9 × Fin 6))) :
  Minesweeper.countMines field numbers = 9 :=
sorry

-- D) Proof that there are 10 mines for the fourth configuration
theorem Problem_7_6_D (field : Matrix (Fin 9) (Fin 6) ℕ) (numbers : List (ℕ × (Fin 9 × Fin 6))) :
  Minesweeper.countMines field numbers = 10 := 
sorry

end Problem_7_6_A_Problem_7_6_B_Problem_7_6_C_Problem_7_6_D_l727_727971


namespace quadratic_function_symmetry_l727_727510

theorem quadratic_function_symmetry (b c : ℝ) :
  let f := λ x : ℝ, x^2 + b * x + c,
  ∀ x, f(2) = (2 : ℝ) ∧ f(1) < f(2) < f(4) := by
  sorry

end quadratic_function_symmetry_l727_727510


namespace gcd_12012_18018_l727_727805

theorem gcd_12012_18018 : Nat.gcd 12012 18018 = 6006 := sorry

end gcd_12012_18018_l727_727805


namespace sqrt_720_simplified_l727_727989

theorem sqrt_720_simplified : (sqrt 720 = 12 * sqrt 5) :=
by
  -- The proof is omitted as per the instructions
  sorry

end sqrt_720_simplified_l727_727989


namespace value_of_H_l727_727216

def sqrt (x : ℝ) : ℝ := Real.sqrt x

def x : ℝ := (sqrt (6 + 2 * sqrt 5) + sqrt (6 - 2 * sqrt 5)) / sqrt 20

def H := (1 + x^5 - x^7)^(2012^(3^11))

theorem value_of_H : H = 1 := by
  -- Given that x = (sqrt (6 + 2 * sqrt 5) + sqrt (6 - 2 * sqrt 5)) / sqrt 20
  -- We need to prove that H = 1
  sorry

end value_of_H_l727_727216


namespace multiplication_with_mixed_number_l727_727016

-- Define mixed numbers as rational numbers for proper calculation
def mixed_to_rational (whole : ℕ) (num : ℕ) (den : ℕ) : ℚ :=
  whole + num / den

-- 7 * (9 + 2/5)
def lhs : ℚ := 7 * mixed_to_rational 9 2 5

-- 65 + 4/5
def rhs : ℚ := mixed_to_rational 65 4 5

theorem multiplication_with_mixed_number : lhs = rhs := by
  sorry

end multiplication_with_mixed_number_l727_727016


namespace matrix_multiplication_l727_727555

variables {α : Type*} [AddCommGroup α] [Module ℝ α]

-- Define vectors a, b and matrix N
variables (a b : α) (N : α →ₗ[ℝ] α)

-- Given conditions
axiom ha : N a = (⟨1, 2⟩ : ℝ × ℝ)
axiom hb : N b = (⟨3, -1⟩ : ℝ × ℝ)

-- The theorem to be proved
theorem matrix_multiplication :
  N (2 • a - b) = (⟨-1, 5⟩ : ℝ × ℝ) :=
by
  -- Proof will go here
  sorry

end matrix_multiplication_l727_727555


namespace multiplication_with_mixed_number_l727_727011

-- Define mixed numbers as rational numbers for proper calculation
def mixed_to_rational (whole : ℕ) (num : ℕ) (den : ℕ) : ℚ :=
  whole + num / den

-- 7 * (9 + 2/5)
def lhs : ℚ := 7 * mixed_to_rational 9 2 5

-- 65 + 4/5
def rhs : ℚ := mixed_to_rational 65 4 5

theorem multiplication_with_mixed_number : lhs = rhs := by
  sorry

end multiplication_with_mixed_number_l727_727011


namespace length_of_BC_fraction_l727_727970

-- Definitions:
variables (A B C D : Point)
variables (AB AC BD CD AD BC : Real)

-- Given conditions:
variables (h1 : AB = 3 * BD)
variables (h2 : AC = 5 * CD)
variables (h3 : AD = AB + BD)
variables (h4 : AD = AC + CD)

-- Axiom to prove the goal:
theorem length_of_BC_fraction (h1 : AB = 3 * BD) 
                              (h2 : AC = 5 * CD) 
                              (h3 : AD = AB + BD) 
                              (h4 : AD = AC + CD) 
                              (h5 : AD ≠ 0) :
  BC / AD = 1 / 12 :=
begin
  sorry,
end

end length_of_BC_fraction_l727_727970


namespace xiao_li_estimate_l727_727678

variable (x y z : ℝ)

theorem xiao_li_estimate (h1 : x > y) (h2 : y > 0) (h3 : 0 < z):
    (x + z) + (y - z) = x + y := 
by 
sorry

end xiao_li_estimate_l727_727678


namespace fliers_left_for_next_day_l727_727677

theorem fliers_left_for_next_day (total_fliers : ℕ) (morning_fraction afternoon_fraction : ℚ) :
  morning_fraction = 1 / 10 → afternoon_fraction = 1 / 4 → total_fliers = 2000 →
  let morning_sent := total_fliers * morning_fraction in
  let remaining_after_morning := total_fliers - morning_sent in
  let afternoon_sent := remaining_after_morning * afternoon_fraction in
  let remaining_after_afternoon := remaining_after_morning - afternoon_sent in
  remaining_after_afternoon = 1350 :=
begin
  assume h1 h2 h3,
  have morning_sent : ℚ := total_fliers * morning_fraction,
  have remaining_after_morning : ℚ := total_fliers - morning_sent,
  have afternoon_sent : ℚ := remaining_after_morning * afternoon_fraction,
  have remaining_after_afternoon : ℚ := remaining_after_morning - afternoon_sent,
  sorry
end

end fliers_left_for_next_day_l727_727677


namespace find_angle_l727_727498

def vec_a : ℝ × ℝ := (1, 2)
def vec_b (x : ℝ) : ℝ × ℝ := (3, x)
def vec_c (y : ℝ) : ℝ × ℝ := (2, y)
def vec_m (x : ℝ) := 2 • vec_a - vec_b x
def vec_n (y : ℝ) := vec_a + vec_c y

theorem find_angle (x y : ℝ) (hx : x = 6) (hy : y = -1) :
  let m := vec_m x in
  let n := vec_n y in
  real.angle_of_vectors m n = 3 • real.pi / 4 := sorry

end find_angle_l727_727498


namespace solve_inner_circle_radius_l727_727218

noncomputable def isosceles_trapezoid_radius := 
  let AB := 8
  let BC := 7
  let DA := 7
  let CD := 6
  let radiusA := 4
  let radiusB := 4
  let radiusC := 3
  let radiusD := 3
  let r := (-72 + 60 * Real.sqrt 3) / 26
  r

theorem solve_inner_circle_radius :
  let k := 72
  let m := 60
  let n := 3
  let p := 26
  gcd k p = 1 → -- explicit gcd calculation between k and p 
  (isosceles_trapezoid_radius = (-k + m * Real.sqrt n) / p) ∧ (k + m + n + p = 161) :=
by
  sorry

end solve_inner_circle_radius_l727_727218


namespace households_both_brands_l727_727347

theorem households_both_brands
  (T : ℕ) (N : ℕ) (A : ℕ) (B : ℕ)
  (hT : T = 300) (hN : N = 80) (hA : A = 60) (hB : ∃ X : ℕ, B = 3 * X ∧ T = N + A + B + X) :
  ∃ X : ℕ, X = 40 :=
by
  -- Upon extracting values from conditions, solving for both brand users X = 40
  sorry

end households_both_brands_l727_727347


namespace largest_of_consecutive_even_integers_l727_727293

theorem largest_of_consecutive_even_integers (sum : ℕ) (count : ℕ) (h_sum : sum = 3000) (h_count : count = 20) :
  let y := (sum - (count * 19)) / count in
  y % 2 = 0 →
  y + (2 * (count - 1)) = 169 :=
by
  sorry

end largest_of_consecutive_even_integers_l727_727293


namespace correct_divisor_l727_727523

variable (D X : ℕ)

-- Conditions
def condition1 : Prop := X = D * 24
def condition2 : Prop := X = (D - 12) * 42

theorem correct_divisor (D X : ℕ) (h1 : condition1 D X) (h2 : condition2 D X) : D = 28 := by
  sorry

end correct_divisor_l727_727523


namespace range_of_f_interval_strictly_increasing_range_of_a_l727_727484

noncomputable def f (x : ℝ) : ℝ := sin (2 * x) + 2 * (cos x) ^ 2 + 1

-- Part 1: Prove the range of f(x) is [1, 2 + sqrt 2] on the interval [0, π / 2]
theorem range_of_f : set.range (λ x : ℝ, f x) = set.Icc (1 : ℝ) (2 + real.sqrt 2) := sorry

-- Part 2: Prove f(x) is strictly increasing on [0, π / 8]
theorem interval_strictly_increasing : ∀ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ π / 8 → f x1 < f x2 := sorry

-- Part 3: Prove the range of a satisfying the inequality a · f(x) + 2a ≥ f(x) for all x in [0, π / 2]
theorem range_of_a (a : ℝ) :
  (∀ x ∈ set.Icc 0 (π / 2), a * f x + 2 * a ≥ f x) ↔ a ≥ (3 + real.sqrt 2) / 7 := sorry

end range_of_f_interval_strictly_increasing_range_of_a_l727_727484


namespace time_after_increment_l727_727928

def initial_time := (5, 45, 0)      -- (hours, minutes, seconds)
def time_increment := 10000         -- seconds
def expected_time := (8, 31, 40)    -- (hours, minutes, seconds)

theorem time_after_increment : 
  (calculate_new_time initial_time time_increment) = expected_time :=
by sorry

-- Assuming a function 'calculate_new_time' which takes the initial time and seconds to add,
-- and returns the new time in (hours, minutes, seconds) format.
def calculate_new_time (start_time : (ℕ, ℕ, ℕ)) (seconds_to_add : ℕ) : (ℕ, ℕ, ℕ) := sorry

end time_after_increment_l727_727928


namespace probability_no_prize_l727_727696

theorem probability_no_prize : (1 : ℚ) - (1 : ℚ) / (50 * 50) = 2499 / 2500 :=
by
  sorry

end probability_no_prize_l727_727696
