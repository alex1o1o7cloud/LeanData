import Mathlib

namespace NUMINAMATH_GPT_range_of_y_for_x_gt_2_l1424_142446

theorem range_of_y_for_x_gt_2 :
  (∀ x : ℝ, x > 2 → 0 < 2 / x ∧ 2 / x < 1) :=
by 
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_range_of_y_for_x_gt_2_l1424_142446


namespace NUMINAMATH_GPT_ratio_Raphael_to_Manny_l1424_142447

-- Define the pieces of lasagna each person will eat
def Manny_pieces : ℕ := 1
def Kai_pieces : ℕ := 2
def Lisa_pieces : ℕ := 2
def Aaron_pieces : ℕ := 0
def Total_pieces : ℕ := 6

-- Calculate the remaining pieces for Raphael
def Raphael_pieces : ℕ := Total_pieces - (Manny_pieces + Kai_pieces + Lisa_pieces + Aaron_pieces)

-- Prove that the ratio of Raphael's pieces to Manny's pieces is 1:1
theorem ratio_Raphael_to_Manny : Raphael_pieces = Manny_pieces :=
by
  -- Provide the actual proof logic, but currently leaving it as a placeholder
  sorry

end NUMINAMATH_GPT_ratio_Raphael_to_Manny_l1424_142447


namespace NUMINAMATH_GPT_percentage_born_in_july_l1424_142459

def total_scientists : ℕ := 150
def scientists_born_in_july : ℕ := 15

theorem percentage_born_in_july : (scientists_born_in_july * 100 / total_scientists) = 10 := by
  sorry

end NUMINAMATH_GPT_percentage_born_in_july_l1424_142459


namespace NUMINAMATH_GPT_opposite_of_neg3_squared_l1424_142402

theorem opposite_of_neg3_squared : -(-3^2) = 9 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg3_squared_l1424_142402


namespace NUMINAMATH_GPT_problem_condition_l1424_142436

theorem problem_condition (m : ℝ) :
  (∀ x : ℝ, x ≤ -1 → (m^2 - m) * 4^x - 2^x < 0) → -1 < m ∧ m < 2 :=
sorry

end NUMINAMATH_GPT_problem_condition_l1424_142436


namespace NUMINAMATH_GPT_audio_space_per_hour_l1424_142479

/-
The digital music library holds 15 days of music.
The library occupies 20,000 megabytes of disk space.
The library contains both audio and video files.
Video files take up twice as much space per hour as audio files.
There is an equal number of hours for audio and video.
-/

theorem audio_space_per_hour (total_days : ℕ) (total_space : ℕ) (equal_hours : Prop) (video_space : ℕ → ℕ) 
  (H1 : total_days = 15)
  (H2 : total_space = 20000)
  (H3 : equal_hours)
  (H4 : ∀ x, video_space x = 2 * x) :
  ∃ x, x = 37 :=
by
  sorry

end NUMINAMATH_GPT_audio_space_per_hour_l1424_142479


namespace NUMINAMATH_GPT_find_a_l1424_142442

theorem find_a (a : ℝ) :
  (∀ x : ℝ, ((x^2 - 4 * x + a) + |x - 3| ≤ 5) → x ≤ 3) →
  (∃ x : ℝ, x = 3 ∧ ((x^2 - 4 * x + a) + |x - 3| ≤ 5)) →
  a = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_a_l1424_142442


namespace NUMINAMATH_GPT_area_of_concentric_ring_l1424_142488

theorem area_of_concentric_ring (r_large : ℝ) (r_small : ℝ) 
  (h1 : r_large = 10) 
  (h2 : r_small = 6) : 
  (π * r_large^2 - π * r_small^2) = 64 * π :=
by {
  sorry
}

end NUMINAMATH_GPT_area_of_concentric_ring_l1424_142488


namespace NUMINAMATH_GPT_weight_labels_correct_l1424_142410

-- Noncomputable because we're dealing with theoretical weight comparisons
noncomputable section

-- Defining the weights and their properties
variables {x1 x2 x3 x4 x5 x6 : ℕ}

-- Given conditions as stated
axiom h1 : x1 + x2 + x3 = 6
axiom h2 : x6 = 6
axiom h3 : x1 + x6 < x3 + x5

theorem weight_labels_correct :
  x1 = 1 ∧ x2 = 2 ∧ x3 = 3 ∧ x4 = 4 ∧ x5 = 5 ∧ x6 = 6 :=
sorry

end NUMINAMATH_GPT_weight_labels_correct_l1424_142410


namespace NUMINAMATH_GPT_abs_neg_two_equals_two_l1424_142489

theorem abs_neg_two_equals_two : |(-2 : ℤ)| = 2 := 
by 
  sorry

end NUMINAMATH_GPT_abs_neg_two_equals_two_l1424_142489


namespace NUMINAMATH_GPT_minimum_value_expression_l1424_142458

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ x : ℝ, 
    (∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → 
    x = (a + 3 * c) / (a + 2 * b + c) + (4 * b) / (a + b + 2 * c) - (8 * c) / (a + b + 3 * c)) ∧
    x = -17 + 12 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_minimum_value_expression_l1424_142458


namespace NUMINAMATH_GPT_base_area_of_cuboid_eq_seven_l1424_142403

-- Definitions of the conditions
def volume_of_cuboid : ℝ := 28 -- Volume is 28 cm³
def height_of_cuboid : ℝ := 4  -- Height is 4 cm

-- The theorem statement for the problem
theorem base_area_of_cuboid_eq_seven
  (Volume : ℝ)
  (Height : ℝ)
  (h1 : Volume = 28)
  (h2 : Height = 4) :
  Volume / Height = 7 := by
  sorry

end NUMINAMATH_GPT_base_area_of_cuboid_eq_seven_l1424_142403


namespace NUMINAMATH_GPT_triangle_classification_l1424_142435

def is_obtuse_triangle (a b c : ℕ) : Prop :=
c^2 > a^2 + b^2 ∧ a < b ∧ b < c

def is_right_triangle (a b c : ℕ) : Prop :=
c^2 = a^2 + b^2 ∧ a < b ∧ b < c

def is_acute_triangle (a b c : ℕ) : Prop :=
c^2 < a^2 + b^2 ∧ a < b ∧ b < c

theorem triangle_classification :
    is_acute_triangle 10 12 14 ∧ 
    is_right_triangle 10 24 26 ∧ 
    is_obtuse_triangle 4 6 8 :=
by 
  sorry

end NUMINAMATH_GPT_triangle_classification_l1424_142435


namespace NUMINAMATH_GPT_equilateral_triangle_area_l1424_142469

theorem equilateral_triangle_area (h : ∀ (a : ℝ), a = 2 * Real.sqrt 3) : 
  ∃ (a : ℝ), a = 4 * Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_equilateral_triangle_area_l1424_142469


namespace NUMINAMATH_GPT_uki_total_earnings_l1424_142477

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

end NUMINAMATH_GPT_uki_total_earnings_l1424_142477


namespace NUMINAMATH_GPT_triangle_sum_of_squares_not_right_l1424_142433

noncomputable def is_right_triangle (a b c : ℝ) : Prop := 
  (a^2 + b^2 = c^2) ∨ (b^2 + c^2 = a^2) ∨ (c^2 + a^2 = b^2)

theorem triangle_sum_of_squares_not_right
  (a b r : ℝ) :
  a^2 + b^2 = (2 * r)^2 → ¬ ∃ (c : ℝ), is_right_triangle a b c := 
sorry

end NUMINAMATH_GPT_triangle_sum_of_squares_not_right_l1424_142433


namespace NUMINAMATH_GPT_pens_bought_l1424_142421

-- Define the given conditions
def num_notebooks : ℕ := 10
def cost_per_pen : ℕ := 2
def total_paid : ℕ := 30
def cost_per_notebook : ℕ := 0  -- Assumption that notebooks are free

-- Converted condition that 10N + 2P = 30 and N = 0
def equation (N P : ℕ) : Prop := (10 * N + 2 * P = total_paid)

-- Statement to prove that if notebooks are free, 15 pens were bought
theorem pens_bought (N : ℕ) (P : ℕ) (hN : N = cost_per_notebook) (h : equation N P) : P = 15 :=
by sorry

end NUMINAMATH_GPT_pens_bought_l1424_142421


namespace NUMINAMATH_GPT_trajectory_of_P_l1424_142465
-- Import entire library for necessary definitions and theorems.

-- Define the properties of the conic sections.
def ellipse (x y : ℝ) (n : ℝ) : Prop :=
  x^2 / 4 + y^2 / n = 1

def hyperbola (x y : ℝ) (m : ℝ) : Prop :=
  x^2 / 8 - y^2 / m = 1

-- Define the condition where the conic sections share the same foci.
def shared_foci (n m : ℝ) : Prop :=
  4 - n = 8 + m

-- The main theorem stating the relationship between m and n forming a straight line.
theorem trajectory_of_P : ∀ (n m : ℝ), shared_foci n m → (m + n + 4 = 0) :=
by
  intros n m h
  sorry

end NUMINAMATH_GPT_trajectory_of_P_l1424_142465


namespace NUMINAMATH_GPT_find_m_l1424_142443

theorem find_m : ∃ m : ℝ, 
  (∀ x : ℝ, 0 < x → (m^2 - m - 5) * x^(m - 1) = (m^2 - m - 5) * x^(m - 1) ∧ 
  (m^2 - m - 5) * (m - 1) * x^(m - 2) > 0) → m = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1424_142443


namespace NUMINAMATH_GPT_smallest_whole_number_larger_than_perimeter_l1424_142416

theorem smallest_whole_number_larger_than_perimeter (s : ℝ) (h1 : 7 + 23 > s) (h2 : 7 + s > 23) (h3 : 23 + s > 7) : 
  60 = Int.ceil (7 + 23 + s - 1) :=
by
  sorry

end NUMINAMATH_GPT_smallest_whole_number_larger_than_perimeter_l1424_142416


namespace NUMINAMATH_GPT_circle_area_l1424_142474

open Real

noncomputable def radius_square (x : ℝ) (DE : ℝ) (EF : ℝ) : ℝ :=
  let DE_square := DE^2
  let r_square_1 := x^2 + DE_square
  let product_DE_EF := DE * EF
  let r_square_2 := product_DE_EF + x^2
  r_square_2

theorem circle_area (x : ℝ) (h1 : OE = x) (h2 : DE = 8) (h3 : EF = 4) :
  π * radius_square x 8 4 = 96 * π :=
by
  sorry

end NUMINAMATH_GPT_circle_area_l1424_142474


namespace NUMINAMATH_GPT_refrigerator_volume_unit_l1424_142475

theorem refrigerator_volume_unit (V : ℝ) (u : String) : 
  V = 150 → (u = "Liters" ∨ u = "Milliliters" ∨ u = "Cubic meters") → 
  u = "Liters" :=
by
  intro hV hu
  sorry

end NUMINAMATH_GPT_refrigerator_volume_unit_l1424_142475


namespace NUMINAMATH_GPT_Mira_trips_to_fill_tank_l1424_142484

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

noncomputable def volume_of_cube (a : ℝ) : ℝ :=
  a^3

noncomputable def number_of_trips (cube_side : ℝ) (sphere_diameter : ℝ) : ℕ :=
  let r := sphere_diameter / 2
  let sphere_volume := volume_of_sphere r
  let cube_volume := volume_of_cube cube_side
  Nat.ceil (cube_volume / sphere_volume)

theorem Mira_trips_to_fill_tank : number_of_trips 8 6 = 5 :=
by
  sorry

end NUMINAMATH_GPT_Mira_trips_to_fill_tank_l1424_142484


namespace NUMINAMATH_GPT_div_c_a_l1424_142417

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a / b = 3)
variable (h2 : b / c = 2 / 5)

-- State the theorem to be proven
theorem div_c_a (a b c : ℝ) (h1 : a / b = 3) (h2 : b / c = 2 / 5) : c / a = 5 / 6 := 
by 
  sorry

end NUMINAMATH_GPT_div_c_a_l1424_142417


namespace NUMINAMATH_GPT_first_half_day_wednesday_l1424_142437

theorem first_half_day_wednesday (h1 : ¬(1 : ℕ) = (4 % 7) ∨ 1 % 7 != 0)
  (h2 : ∀ d : ℕ, d ≤ 31 → d % 7 = ((d + 3) % 7)) : 
  ∃ d : ℕ, d = 25 ∧ ∃ W : ℕ → Prop, W d := sorry

end NUMINAMATH_GPT_first_half_day_wednesday_l1424_142437


namespace NUMINAMATH_GPT_solution_to_inequality_l1424_142468

theorem solution_to_inequality :
  { x : ℝ | ((x^2 - 1) / (x - 4)^2) ≥ 0 } = { x : ℝ | x ≤ -1 ∨ (1 ≤ x ∧ x < 4) ∨ x > 4 } := 
sorry

end NUMINAMATH_GPT_solution_to_inequality_l1424_142468


namespace NUMINAMATH_GPT_rachel_milk_correct_l1424_142434

-- Define the initial amount of milk Don has
def don_milk : ℚ := 1 / 5

-- Define the fraction of milk Rachel drinks
def rachel_drinks_fraction : ℚ := 2 / 3

-- Define the total amount of milk Rachel drinks
def rachel_milk : ℚ := rachel_drinks_fraction * don_milk

-- The goal is to prove that Rachel drinks a specific amount of milk
theorem rachel_milk_correct : rachel_milk = 2 / 15 :=
by
  -- The proof would be here
  sorry

end NUMINAMATH_GPT_rachel_milk_correct_l1424_142434


namespace NUMINAMATH_GPT_solve_problem_l1424_142450

noncomputable def problem_statement : ℤ :=
  (-3)^6 / 3^4 - 4^3 * 2^2 + 9^2

theorem solve_problem : problem_statement = -166 :=
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_solve_problem_l1424_142450


namespace NUMINAMATH_GPT_function_symmetry_origin_l1424_142425

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - x

theorem function_symmetry_origin : ∀ x : ℝ, f (-x) = -f x := by
  sorry

end NUMINAMATH_GPT_function_symmetry_origin_l1424_142425


namespace NUMINAMATH_GPT_vector_sum_l1424_142428

def a : ℝ × ℝ := (-2, 3)
def b : ℝ × ℝ := (1, -2)

theorem vector_sum:
  2 • a + b = (-3, 4) :=
by 
  sorry

end NUMINAMATH_GPT_vector_sum_l1424_142428


namespace NUMINAMATH_GPT_radius_I_l1424_142445

noncomputable def radius_O1 : ℝ := 3
noncomputable def radius_O2 : ℝ := 3
noncomputable def radius_O3 : ℝ := 3

axiom O1_O2_tangent : ∀ (O1 O2 : ℝ), O1 + O2 = radius_O1 + radius_O2
axiom O2_O3_tangent : ∀ (O2 O3 : ℝ), O2 + O3 = radius_O2 + radius_O3
axiom O3_O1_tangent : ∀ (O3 O1 : ℝ), O3 + O1 = radius_O3 + radius_O1

axiom I_O1_tangent : ∀ (I O1 : ℝ), I + O1 = radius_O1 + I
axiom I_O2_tangent : ∀ (I O2 : ℝ), I + O2 = radius_O2 + I
axiom I_O3_tangent : ∀ (I O3 : ℝ), I + O3 = radius_O3 + I

theorem radius_I : ∀ (I : ℝ), I = radius_O1 :=
by
  sorry

end NUMINAMATH_GPT_radius_I_l1424_142445


namespace NUMINAMATH_GPT_man_speed_l1424_142418

theorem man_speed (time_in_minutes : ℕ) (distance_in_km : ℕ) 
  (h_time : time_in_minutes = 30) 
  (h_distance : distance_in_km = 5) : 
  (distance_in_km : ℝ) / (time_in_minutes / 60 : ℝ) = 10 :=
by 
  sorry

end NUMINAMATH_GPT_man_speed_l1424_142418


namespace NUMINAMATH_GPT_poultry_count_correct_l1424_142430

noncomputable def total_poultry : ℝ :=
  let hens_total := 40
  let ducks_total := 20
  let geese_total := 10
  let pigeons_total := 30

  -- Calculate males and females
  let hens_males := (2/9) * hens_total
  let hens_females := hens_total - hens_males

  let ducks_males := (1/4) * ducks_total
  let ducks_females := ducks_total - ducks_males

  let geese_males := (3/11) * geese_total
  let geese_females := geese_total - geese_males

  let pigeons_males := (1/2) * pigeons_total
  let pigeons_females := pigeons_total - pigeons_males

  -- Offspring calculations using breeding success rates
  let hens_offspring := (0.85 * hens_females) * 7
  let ducks_offspring := (0.75 * ducks_females) * 9
  let geese_offspring := (0.9 * geese_females) * 5
  let pigeons_pairs := 0.8 * (pigeons_females / 2)
  let pigeons_offspring := pigeons_pairs * 2 * 0.8

  -- Total poultry count
  (hens_total + ducks_total + geese_total + pigeons_total) + (hens_offspring + ducks_offspring + geese_offspring + pigeons_offspring)

theorem poultry_count_correct : total_poultry = 442 := by
  sorry

end NUMINAMATH_GPT_poultry_count_correct_l1424_142430


namespace NUMINAMATH_GPT_work_rate_l1424_142407

theorem work_rate (R_B : ℚ) (R_A : ℚ) (R_total : ℚ) (days : ℚ)
  (h1 : R_A = (1/2) * R_B)
  (h2 : R_B = 1 / 22.5)
  (h3 : R_total = R_A + R_B)
  (h4 : days = 1 / R_total) : 
  days = 15 := 
sorry

end NUMINAMATH_GPT_work_rate_l1424_142407


namespace NUMINAMATH_GPT_billy_raspberry_juice_billy_raspberry_juice_quarts_l1424_142448

theorem billy_raspberry_juice (V : ℚ) (h : V / 12 + 1 = 3) : V = 24 :=
by sorry

theorem billy_raspberry_juice_quarts (V : ℚ) (h : V / 12 + 1 = 3) : V / 4 = 6 :=
by sorry

end NUMINAMATH_GPT_billy_raspberry_juice_billy_raspberry_juice_quarts_l1424_142448


namespace NUMINAMATH_GPT_probability_odd_sum_l1424_142483

-- Definitions based on the conditions
def cards : List ℕ := [1, 2, 3, 4, 5]

def is_odd_sum (a b : ℕ) : Prop := (a + b) % 2 = 1

def combinations (n k : ℕ) : ℕ := (Nat.choose n k)

-- Main statement
theorem probability_odd_sum :
  (combinations 5 2) = 10 → -- Total combinations of 2 cards from 5
  (∃ N, N = 6 ∧ (N:ℚ)/(combinations 5 2) = 3/5) :=
by 
  sorry

end NUMINAMATH_GPT_probability_odd_sum_l1424_142483


namespace NUMINAMATH_GPT_find_a12_l1424_142439

variable (a : ℕ → ℤ)
variable (H1 : a 1 = 1) 
variable (H2 : ∀ m n : ℕ, a (m + n) = a m + a n + m * n)

theorem find_a12 : a 12 = 78 := 
by
  sorry

end NUMINAMATH_GPT_find_a12_l1424_142439


namespace NUMINAMATH_GPT_average_age_add_person_l1424_142415

theorem average_age_add_person (n : ℕ) (h1 : (∀ T, T = n * 14 → (T + 34) / (n + 1) = 16)) : n = 9 :=
by
  sorry

end NUMINAMATH_GPT_average_age_add_person_l1424_142415


namespace NUMINAMATH_GPT_units_digit_sum_even_20_to_80_l1424_142495

theorem units_digit_sum_even_20_to_80 :
  let a := 20
  let d := 2
  let l := 80
  let n := ((l - a) / d) + 1 -- Given by the formula l = a + (n-1)d => n = (l - a) / d + 1
  let sum := (n * (a + l)) / 2
  (sum % 10) = 0 := sorry

end NUMINAMATH_GPT_units_digit_sum_even_20_to_80_l1424_142495


namespace NUMINAMATH_GPT_geom_sequence_sum_first_ten_terms_l1424_142461

noncomputable def geom_sequence_sum (a1 q n : ℕ) : ℕ :=
  a1 * (1 - q^n) / (1 - q)

theorem geom_sequence_sum_first_ten_terms (a : ℕ) (q : ℕ) (h1 : a * (1 + q) = 6) (h2 : a * q^3 * (1 + q) = 48) :
  geom_sequence_sum a q 10 = 2046 :=
sorry

end NUMINAMATH_GPT_geom_sequence_sum_first_ten_terms_l1424_142461


namespace NUMINAMATH_GPT_evaluate_expression_l1424_142464

theorem evaluate_expression :
  ((3.5 / 0.7) * (5 / 3) + (7.2 / 0.36) - ((5 / 3) * (0.75 / 0.25))) = 23.3335 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1424_142464


namespace NUMINAMATH_GPT_independent_and_dependent_variables_l1424_142462

variable (R V : ℝ)

theorem independent_and_dependent_variables (h : V = (4 / 3) * Real.pi * R^3) :
  (∃ R : ℝ, ∀ V : ℝ, V = (4 / 3) * Real.pi * R^3) ∧ (∃ V : ℝ, ∃ R' : ℝ, V = (4 / 3) * Real.pi * R'^3) :=
by
  sorry

end NUMINAMATH_GPT_independent_and_dependent_variables_l1424_142462


namespace NUMINAMATH_GPT_range_arcsin_x_squared_minus_x_l1424_142476

noncomputable def range_of_arcsin : Set ℝ :=
  {x | -Real.arcsin (1 / 4) ≤ x ∧ x ≤ Real.pi / 2}

theorem range_arcsin_x_squared_minus_x :
  ∀ x : ℝ, ∃ y ∈ range_of_arcsin, y = Real.arcsin (x^2 - x) :=
by
  sorry

end NUMINAMATH_GPT_range_arcsin_x_squared_minus_x_l1424_142476


namespace NUMINAMATH_GPT_length_after_haircut_l1424_142420

-- Definitions
def original_length : ℕ := 18
def cut_length : ℕ := 9

-- Target statement to prove
theorem length_after_haircut : original_length - cut_length = 9 :=
by
  -- Simplification and proof
  sorry

end NUMINAMATH_GPT_length_after_haircut_l1424_142420


namespace NUMINAMATH_GPT_convert_to_scientific_notation_l1424_142404

-- Problem statement: convert 120 million to scientific notation and validate the format.
theorem convert_to_scientific_notation :
  120000000 = 1.2 * 10^7 :=
sorry

end NUMINAMATH_GPT_convert_to_scientific_notation_l1424_142404


namespace NUMINAMATH_GPT_train_speed_kmph_l1424_142472

/-- Given that the length of the train is 200 meters and it crosses a pole in 9 seconds,
the speed of the train in km/hr is 80. -/
theorem train_speed_kmph (length : ℝ) (time : ℝ) (length_eq : length = 200) (time_eq : time = 9) : 
  (length / time) * (3600 / 1000) = 80 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_kmph_l1424_142472


namespace NUMINAMATH_GPT_right_triangle_other_angle_l1424_142429

theorem right_triangle_other_angle (a b c : ℝ) 
  (h_triangle_sum : a + b + c = 180) 
  (h_right_angle : a = 90) 
  (h_acute_angle : b = 60) : 
  c = 30 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_other_angle_l1424_142429


namespace NUMINAMATH_GPT_tan_2x_abs_properties_l1424_142423

open Real

theorem tan_2x_abs_properties :
  (∀ x : ℝ, |tan (2 * x)| = |tan (2 * (-x))|) ∧ (∀ x : ℝ, |tan (2 * x)| = |tan (2 * (x + π / 2))|) :=
by
  sorry

end NUMINAMATH_GPT_tan_2x_abs_properties_l1424_142423


namespace NUMINAMATH_GPT_correct_grid_l1424_142498

def A := 8
def B := 6
def C := 4
def D := 2

def grid := [[A, 1, 9],
             [3, 5, D],
             [B, C, 7]]

theorem correct_grid :
  (A + 1 < 12) ∧ (A + 3 < 12) ∧ (1 + 9 < 12) ∧
  (1 + 5 < 12) ∧ (3 + 5 < 12) ∧ (3 + B < 12) ∧
  (5 + D < 12) ∧ (5 + C < 12) ∧ (9 + D < 12) ∧
  (B + C < 12) ∧ (C + 7 < 12) :=
by
  -- This is to provide a sketch dummy theorem, we'd prove each step here  
  sorry

end NUMINAMATH_GPT_correct_grid_l1424_142498


namespace NUMINAMATH_GPT_math_majors_consecutive_probability_l1424_142440

def twelve_people := 12
def math_majors := 5
def physics_majors := 4
def biology_majors := 3

def total_ways := Nat.choose twelve_people math_majors

-- Computes the probability that all five math majors sit in consecutive seats
theorem math_majors_consecutive_probability :
  (12 : ℕ) / (Nat.choose twelve_people math_majors) = 1 / 66 := by
  sorry

end NUMINAMATH_GPT_math_majors_consecutive_probability_l1424_142440


namespace NUMINAMATH_GPT_main_problem_l1424_142452

-- Define the set A
def A (a : ℝ) : Set ℝ :=
  {0, 1, a^2 - 2 * a}

-- Define the main problem as a theorem
theorem main_problem (a : ℝ) (h : a ∈ A a) : a = 1 ∨ a = 3 :=
  sorry

end NUMINAMATH_GPT_main_problem_l1424_142452


namespace NUMINAMATH_GPT_scientific_notation_of_trade_volume_l1424_142478

-- Define the total trade volume
def total_trade_volume : ℕ := 175000000000

-- Define the expected scientific notation result
def expected_result : ℝ := 1.75 * 10^11

-- Theorem stating the problem
theorem scientific_notation_of_trade_volume :
  (total_trade_volume : ℝ) = expected_result := by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_trade_volume_l1424_142478


namespace NUMINAMATH_GPT_bags_needed_l1424_142471

theorem bags_needed (expected_people extra_people extravagant_bags average_bags : ℕ) 
    (h1 : expected_people = 50) 
    (h2 : extra_people = 40) 
    (h3 : extravagant_bags = 10) 
    (h4 : average_bags = 20) : 
    (expected_people + extra_people - (extravagant_bags + average_bags) = 60) :=
by {
  sorry
}

end NUMINAMATH_GPT_bags_needed_l1424_142471


namespace NUMINAMATH_GPT_marshmallow_total_l1424_142470

-- Define the number of marshmallows each kid can hold
def Haley := 8
def Michael := 3 * Haley
def Brandon := Michael / 2

-- Prove the total number of marshmallows held by all three is 44
theorem marshmallow_total : Haley + Michael + Brandon = 44 := by
  sorry

end NUMINAMATH_GPT_marshmallow_total_l1424_142470


namespace NUMINAMATH_GPT_line_passes_through_quadrants_l1424_142413

theorem line_passes_through_quadrants (a b c : ℝ) (hab : a * b < 0) (hbc : b * c < 0) : 
  ∀ (x y : ℝ), (a * x + b * y + c = 0) → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_line_passes_through_quadrants_l1424_142413


namespace NUMINAMATH_GPT_opposite_of_neg2023_l1424_142497

def opposite (x : Int) := -x

theorem opposite_of_neg2023 : opposite (-2023) = 2023 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg2023_l1424_142497


namespace NUMINAMATH_GPT_people_in_club_M_l1424_142408

theorem people_in_club_M (m s z n : ℕ) (h1 : s = 18) (h2 : z = 11) (h3 : m + s + z + n = 60) (h4 : n ≤ 26) : m = 5 :=
sorry

end NUMINAMATH_GPT_people_in_club_M_l1424_142408


namespace NUMINAMATH_GPT_total_snails_and_frogs_l1424_142481

-- Define the number of snails and frogs in the conditions.
def snails : Nat := 5
def frogs : Nat := 2

-- State the problem: proving that the total number of snails and frogs equals 7.
theorem total_snails_and_frogs : snails + frogs = 7 := by
  -- Proof is omitted as the user requested only the statement.
  sorry

end NUMINAMATH_GPT_total_snails_and_frogs_l1424_142481


namespace NUMINAMATH_GPT_sum_of_arithmetic_series_51_to_100_l1424_142451

theorem sum_of_arithmetic_series_51_to_100 :
  let first_term := 51
  let last_term := 100
  let n := (last_term - first_term) + 1
  2 * (n / 2) * (first_term + last_term) / 2 = 3775 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_arithmetic_series_51_to_100_l1424_142451


namespace NUMINAMATH_GPT_remainder_when_sum_divided_by_29_l1424_142405

theorem remainder_when_sum_divided_by_29 (c d : ℤ) (k j : ℤ) 
  (hc : c = 52 * k + 48) 
  (hd : d = 87 * j + 82) : 
  (c + d) % 29 = 22 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_when_sum_divided_by_29_l1424_142405


namespace NUMINAMATH_GPT_solution_unique_2014_l1424_142491

theorem solution_unique_2014 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (2 * x - 2 * y + 1 / z = 1 / 2014) ∧
  (2 * y - 2 * z + 1 / x = 1 / 2014) ∧
  (2 * z - 2 * x + 1 / y = 1 / 2014) →
  x = 2014 ∧ y = 2014 ∧ z = 2014 :=
by
  sorry

end NUMINAMATH_GPT_solution_unique_2014_l1424_142491


namespace NUMINAMATH_GPT_yellow_balls_in_bag_l1424_142492

theorem yellow_balls_in_bag (y : ℕ) (r : ℕ) (P_red : ℚ) (h_r : r = 8) (h_P_red : P_red = 1 / 3) 
  (h_prob : P_red = r / (r + y)) : y = 16 :=
by
  sorry

end NUMINAMATH_GPT_yellow_balls_in_bag_l1424_142492


namespace NUMINAMATH_GPT_intersection_complement_eq_l1424_142466

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {4, 5}

-- Theorem
theorem intersection_complement_eq : (A ∩ (U \ B)) = {2, 3} :=
by 
  sorry

end NUMINAMATH_GPT_intersection_complement_eq_l1424_142466


namespace NUMINAMATH_GPT_rhombus_area_l1424_142482

noncomputable def area_of_rhombus (d1 d2 : ℝ) : ℝ :=
  0.5 * d1 * d2

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 3) (h2 : d2 = 4) : area_of_rhombus d1 d2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_area_l1424_142482


namespace NUMINAMATH_GPT_largest_7_10_triple_l1424_142427

theorem largest_7_10_triple :
  ∃ M : ℕ, (3 * M = Nat.ofDigits 10 (Nat.digits 7 M))
  ∧ (∀ N : ℕ, (3 * N = Nat.ofDigits 10 (Nat.digits 7 N)) → N ≤ M)
  ∧ M = 335 :=
sorry

end NUMINAMATH_GPT_largest_7_10_triple_l1424_142427


namespace NUMINAMATH_GPT_average_of_roots_l1424_142456

theorem average_of_roots (p q : ℝ) (h : ∃ x1 x2 : ℝ, 3*p*x1^2 - 6*p*x1 + q = 0 ∧ 3*p*x2^2 - 6*p*x2 + q = 0 ∧ x1 ≠ x2):
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 3*p*x1^2 - 6*p*x1 + q = 0 ∧ 3*p*x2^2 - 6*p*x2 + q = 0) → 
  (x1 + x2) / 2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_average_of_roots_l1424_142456


namespace NUMINAMATH_GPT_equal_areas_triangle_height_l1424_142494

theorem equal_areas_triangle_height (l b h : ℝ) (hlb : l > b) 
  (H1 : l * b = (1/2) * l * h) : h = 2 * b :=
by 
  -- skipping proof
  sorry

end NUMINAMATH_GPT_equal_areas_triangle_height_l1424_142494


namespace NUMINAMATH_GPT_age_hence_l1424_142419

theorem age_hence (A x : ℕ) (h1 : A = 50)
  (h2 : 5 * (A + x) - 5 * (A - 5) = A) : x = 5 :=
by sorry

end NUMINAMATH_GPT_age_hence_l1424_142419


namespace NUMINAMATH_GPT_solution_to_water_l1424_142449

theorem solution_to_water (A W S T: ℝ) (h1: A = 0.04) (h2: W = 0.02) (h3: S = 0.06) (h4: T = 0.48) :
  (T * (W / S) = 0.16) :=
by
  sorry

end NUMINAMATH_GPT_solution_to_water_l1424_142449


namespace NUMINAMATH_GPT_quadratic_function_order_l1424_142460

theorem quadratic_function_order (a b c : ℝ) (h_neg_a : a < 0) 
  (h_sym : ∀ x, (a * (x + 2)^2 + b * (x + 2) + c) = (a * (2 - x)^2 + b * (2 - x) + c)) :
  (a * (-1992)^2 + b * (-1992) + c) < (a * (1992)^2 + b * (1992) + c) ∧
  (a * (1992)^2 + b * (1992) + c) < (a * (0)^2 + b * (0) + c) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_order_l1424_142460


namespace NUMINAMATH_GPT_saffron_milk_caps_and_milk_caps_in_basket_l1424_142432

structure MushroomBasket :=
  (total : ℕ)
  (saffronMilkCapCount : ℕ)
  (milkCapCount : ℕ)
  (TotalMushrooms : total = 30)
  (SaffronMilkCapCondition : ∀ (selected : Finset ℕ), selected.card = 12 → ∃ i ∈ selected, i < saffronMilkCapCount)
  (MilkCapCondition : ∀ (selected : Finset ℕ), selected.card = 20 → ∃ i ∈ selected, i < milkCapCount)

theorem saffron_milk_caps_and_milk_caps_in_basket
  (basket : MushroomBasket)
  (TotalMushrooms : basket.total = 30)
  (SaffronMilkCapCondition : ∀ (selected : Finset ℕ), selected.card = 12 → ∃ i ∈ selected, i < basket.saffronMilkCapCount)
  (MilkCapCondition : ∀ (selected : Finset ℕ), selected.card = 20 → ∃ i ∈ selected, i < basket.milkCapCount) :
  basket.saffronMilkCapCount = 19 ∧ basket.milkCapCount = 11 :=
sorry

end NUMINAMATH_GPT_saffron_milk_caps_and_milk_caps_in_basket_l1424_142432


namespace NUMINAMATH_GPT_not_prime_sum_l1424_142414

theorem not_prime_sum (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_eq : a * b = c * d) : ¬ Nat.Prime (a + b + c + d) :=
sorry

end NUMINAMATH_GPT_not_prime_sum_l1424_142414


namespace NUMINAMATH_GPT_arithmetic_sequence_26th_term_l1424_142499

theorem arithmetic_sequence_26th_term (a d : ℤ) (h1 : a = 3) (h2 : a + d = 13) (h3 : a + 2 * d = 23) : 
  a + 25 * d = 253 :=
by
  -- specifications for variables a, d, and hypotheses h1, h2, h3
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_26th_term_l1424_142499


namespace NUMINAMATH_GPT_cone_central_angle_l1424_142426

/-- Proof Problem Statement: Given the radius of the base circle of a cone (r) and the slant height of the cone (l),
    prove that the central angle (θ) of the unfolded diagram of the lateral surface of this cone is 120 degrees. -/
theorem cone_central_angle (r l : ℝ) (h_r : r = 10) (h_l : l = 30) : (360 * r) / l = 120 :=
by
  -- The proof steps are omitted
  sorry

end NUMINAMATH_GPT_cone_central_angle_l1424_142426


namespace NUMINAMATH_GPT_train_speed_proof_l1424_142467

theorem train_speed_proof : 
  ∀ (V_A V_B : ℝ) (T_A T_B : ℝ), 
  T_A = 9 ∧ 
  T_B = 4 ∧ 
  V_B = 90 ∧ 
  (V_A / V_B = T_B / T_A) → 
  V_A = 40 := 
by
  intros V_A V_B T_A T_B h
  obtain ⟨hT_A, hT_B, hV_B, hprop⟩ := h
  sorry

end NUMINAMATH_GPT_train_speed_proof_l1424_142467


namespace NUMINAMATH_GPT_biology_books_needed_l1424_142441

-- Define the problem in Lean
theorem biology_books_needed
  (B P Q R F Z₁ Z₂ : ℕ)
  (b p : ℝ)
  (H1 : B ≠ P)
  (H2 : B ≠ Q)
  (H3 : B ≠ R)
  (H4 : B ≠ F)
  (H5 : P ≠ Q)
  (H6 : P ≠ R)
  (H7 : P ≠ F)
  (H8 : Q ≠ R)
  (H9 : Q ≠ F)
  (H10 : R ≠ F)
  (H11 : 0 < B ∧ 0 < P ∧ 0 < Q ∧ 0 < R ∧ 0 < F)
  (H12 : Bb + Pp = Z₁)
  (H13 : Qb + Rp = Z₂)
  (H14 : Fb = Z₁)
  (H15 : Z₂ < Z₁) :
  F = (Q - B) / (P - R) :=
by
  sorry  -- Proof to be provided

end NUMINAMATH_GPT_biology_books_needed_l1424_142441


namespace NUMINAMATH_GPT_fraction_of_automobile_installment_credit_extended_by_finance_companies_l1424_142487

theorem fraction_of_automobile_installment_credit_extended_by_finance_companies
  (total_consumer_credit : ℝ)
  (percentage_auto_credit : ℝ)
  (credit_extended_by_finance_companies : ℝ)
  (total_auto_credit_fraction : percentage_auto_credit = 0.36)
  (total_consumer_credit_value : total_consumer_credit = 475)
  (credit_extended_by_finance_companies_value : credit_extended_by_finance_companies = 57) :
  credit_extended_by_finance_companies / (percentage_auto_credit * total_consumer_credit) = 1 / 3 :=
by
  -- The proof part will go here.
  sorry

end NUMINAMATH_GPT_fraction_of_automobile_installment_credit_extended_by_finance_companies_l1424_142487


namespace NUMINAMATH_GPT_max_value_sqrt_abcd_l1424_142406

theorem max_value_sqrt_abcd (a b c d : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d) (h_sum : a + b + c + d = 1) :
  Real.sqrt (abcd) ^ (1 / 4) + Real.sqrt ((1 - a) * (1 - b) * (1 - c) * (1 - d)) ^ (1 / 4) ≤ 1 := 
sorry

end NUMINAMATH_GPT_max_value_sqrt_abcd_l1424_142406


namespace NUMINAMATH_GPT_passengers_on_ship_l1424_142422

theorem passengers_on_ship : 
  ∀ (P : ℕ), 
    P / 20 + P / 15 + P / 10 + P / 12 + P / 30 + 60 = P → 
    P = 90 :=
by 
  intros P h
  sorry

end NUMINAMATH_GPT_passengers_on_ship_l1424_142422


namespace NUMINAMATH_GPT_magnitude_of_b_is_5_l1424_142486

variable (a b : ℝ × ℝ)
variable (h_a : a = (3, -2))
variable (h_ab : a + b = (0, 2))

theorem magnitude_of_b_is_5 : ‖b‖ = 5 :=
by
  sorry

end NUMINAMATH_GPT_magnitude_of_b_is_5_l1424_142486


namespace NUMINAMATH_GPT_num_individuals_eliminated_l1424_142493

theorem num_individuals_eliminated (pop_size : ℕ) (sample_size : ℕ) :
  (pop_size % sample_size) = 2 :=
by
  -- Given conditions
  let pop_size := 1252
  let sample_size := 50
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_num_individuals_eliminated_l1424_142493


namespace NUMINAMATH_GPT_expand_expression_l1424_142444

variable {R : Type _} [CommRing R] (x : R)

theorem expand_expression :
  (3*x^2 + 7*x + 4) * (5*x - 2) = 15*x^3 + 29*x^2 + 6*x - 8 :=
by
  sorry

end NUMINAMATH_GPT_expand_expression_l1424_142444


namespace NUMINAMATH_GPT_leak_empties_tank_in_10_hours_l1424_142480

theorem leak_empties_tank_in_10_hours :
  (∀ (A L : ℝ), (A = 1/5) → (A - L = 1/10) → (1 / L = 10)) 
  := by
  intros A L hA hAL
  sorry

end NUMINAMATH_GPT_leak_empties_tank_in_10_hours_l1424_142480


namespace NUMINAMATH_GPT_sequence_relation_l1424_142463

theorem sequence_relation (b : ℕ → ℚ) : 
  b 1 = 2 ∧ b 2 = 5 / 11 ∧ (∀ n ≥ 3, b n = b (n-2) * b (n-1) / (3 * b (n-2) - b (n-1)))
  ↔ b 2023 = 5 / 12137 :=
by sorry

end NUMINAMATH_GPT_sequence_relation_l1424_142463


namespace NUMINAMATH_GPT_part1_part2_1_part2_2_l1424_142412

-- Define the operation
def mul_op (x y : ℚ) : ℚ := x ^ 2 - 3 * y + 3

-- Part 1: Prove (-4) * 2 = 13 given the operation definition
theorem part1 : mul_op (-4) 2 = 13 := sorry

-- Part 2.1: Simplify (a - b) * (a - b)^2
theorem part2_1 (a b : ℚ) : mul_op (a - b) ((a - b) ^ 2) = -2 * a ^ 2 - 2 * b ^ 2 + 4 * a * b + 3 := sorry

-- Part 2.2: Find the value of the expression when a = -2 and b = 1/2
theorem part2_2 : mul_op (-2 - 1/2) ((-2 - 1/2) ^ 2) = -13 / 2 := sorry

end NUMINAMATH_GPT_part1_part2_1_part2_2_l1424_142412


namespace NUMINAMATH_GPT_tan_inverse_least_positive_l1424_142431

variables (a b x : ℝ)

-- Condition 1: tan(x) = a / (2*b)
def condition1 : Prop := Real.tan x = a / (2 * b)

-- Condition 2: tan(2*x) = 2*b / (a + 2*b)
def condition2 : Prop := Real.tan (2 * x) = (2 * b) / (a + 2 * b)

-- The theorem stating the least positive value of x is arctan(0)
theorem tan_inverse_least_positive (h1 : condition1 a b x) (h2 : condition2 a b x) : ∃ k : ℝ, Real.arctan k = 0 :=
by
  sorry

end NUMINAMATH_GPT_tan_inverse_least_positive_l1424_142431


namespace NUMINAMATH_GPT_max_x_value_l1424_142400

theorem max_x_value (x y z : ℝ) (h1 : x + y + z = 5) (h2 : xy + xz + yz = 8) : 
  x ≤ 7 / 3 :=
sorry

end NUMINAMATH_GPT_max_x_value_l1424_142400


namespace NUMINAMATH_GPT_actual_diameter_of_tissue_is_0_03_mm_l1424_142455

-- Defining necessary conditions
def magnified_diameter_meters : ℝ := 0.15
def magnification_factor : ℝ := 5000
def meters_to_millimeters : ℝ := 1000

-- Prove that the actual diameter of the tissue is 0.03 millimeters
theorem actual_diameter_of_tissue_is_0_03_mm :
  (magnified_diameter_meters * meters_to_millimeters) / magnification_factor = 0.03 := 
  sorry

end NUMINAMATH_GPT_actual_diameter_of_tissue_is_0_03_mm_l1424_142455


namespace NUMINAMATH_GPT_evaluate_fraction_l1424_142409

theorem evaluate_fraction (a b : ℤ) (h1 : a = 5) (h2 : b = -2) : (5 : ℝ) / (a + b) = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_fraction_l1424_142409


namespace NUMINAMATH_GPT_sequence_geometric_progression_iff_b1_eq_b2_l1424_142490

theorem sequence_geometric_progression_iff_b1_eq_b2 
  (b : ℕ → ℝ) 
  (h0 : ∀ n, b n > 0)
  (h1 : ∀ n, b (n + 2) = 3 * b n * b (n + 1)) :
  (∃ r : ℝ, ∀ n, b (n + 1) = r * b n) ↔ b 1 = b 0 :=
sorry

end NUMINAMATH_GPT_sequence_geometric_progression_iff_b1_eq_b2_l1424_142490


namespace NUMINAMATH_GPT_minimum_mn_l1424_142438

noncomputable def f (x : ℝ) (n m : ℝ) : ℝ := Real.log x - n * x + Real.log m + 1

noncomputable def f' (x : ℝ) (n : ℝ) : ℝ := 1/x - n

theorem minimum_mn (m n x_0 : ℝ) (h_m : m > 1) (h_tangent : 2*x_0 - (f x_0 n m) + 1 = 0) :
  mn = e * ((1/x_0 - 1) ^ 2 - 1) :=
sorry

end NUMINAMATH_GPT_minimum_mn_l1424_142438


namespace NUMINAMATH_GPT_mod_multiplication_l1424_142424

theorem mod_multiplication :
  (176 * 929) % 50 = 4 :=
by
  sorry

end NUMINAMATH_GPT_mod_multiplication_l1424_142424


namespace NUMINAMATH_GPT_carmen_candles_needed_l1424_142453

-- Definitions based on the conditions

def candle_lifespan_1_hour : Nat := 8  -- a candle lasts 8 nights when burned 1 hour each night
def nights_total : Nat := 24  -- total nights

-- Question: How many candles are needed if burned 2 hours a night?

theorem carmen_candles_needed (h : candle_lifespan_1_hour / 2 = 4) :
  nights_total / 4 = 6 := 
  sorry

end NUMINAMATH_GPT_carmen_candles_needed_l1424_142453


namespace NUMINAMATH_GPT_sum_of_perimeters_l1424_142454

theorem sum_of_perimeters (x y : ℝ) (h1 : x^2 + y^2 = 85) (h2 : x^2 - y^2 = 41) :
  4 * (Real.sqrt 63 + Real.sqrt 22) = 4 * (Real.sqrt x^2 + Real.sqrt y^2) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_perimeters_l1424_142454


namespace NUMINAMATH_GPT_evaluate_root_power_l1424_142485

theorem evaluate_root_power : (Real.sqrt (Real.sqrt 9))^12 = 729 := 
by sorry

end NUMINAMATH_GPT_evaluate_root_power_l1424_142485


namespace NUMINAMATH_GPT_trim_area_dodecagon_pie_l1424_142411

theorem trim_area_dodecagon_pie :
  let d := 8 -- diameter of the pie
  let r := d / 2 -- radius of the pie
  let A_circle := π * r^2 -- area of the circle
  let A_dodecagon := 3 * r^2 -- area of the dodecagon
  let A_trimmed := A_circle - A_dodecagon -- area to be trimmed
  let a := 16 -- coefficient of π in A_trimmed
  let b := 48 -- constant term in A_trimmed
  a + b = 64 := 
by 
  sorry

end NUMINAMATH_GPT_trim_area_dodecagon_pie_l1424_142411


namespace NUMINAMATH_GPT_fraction_of_income_from_tips_l1424_142401

variable (S T I : ℝ)

-- Conditions
def tips_are_fraction_of_salary : Prop := T = (3/4) * S
def total_income_is_sum_of_salary_and_tips : Prop := I = S + T

-- Statement to prove
theorem fraction_of_income_from_tips (h1 : tips_are_fraction_of_salary S T) (h2 : total_income_is_sum_of_salary_and_tips S T I) :
  T / I = 3 / 7 := by
  sorry

end NUMINAMATH_GPT_fraction_of_income_from_tips_l1424_142401


namespace NUMINAMATH_GPT_johnny_practice_l1424_142473

variable (P : ℕ) -- Current amount of practice in days
variable (h : P = 40) -- Given condition translating Johnny's practice amount
variable (d : ℕ) -- Additional days needed

theorem johnny_practice : d = 80 :=
by
  have goal : 3 * P = P + d := sorry
  have initial_condition : P = 40 := sorry
  have required : d = 3 * 40 - 40 := sorry
  sorry

end NUMINAMATH_GPT_johnny_practice_l1424_142473


namespace NUMINAMATH_GPT_cosine_135_eq_neg_sqrt_2_div_2_l1424_142457

theorem cosine_135_eq_neg_sqrt_2_div_2 (cos180 : Real := -1) (cos90 : Real := 0) (cos45 : Real := Real.sqrt 2 / 2)
    (theta1 : Real := 180) (theta2 : Real := 45) (theta3 : Real := 135) :
    theta3 = theta1 - theta2 ∧ 
    Real.cos theta1 = cos180 ∧ 
    Real.cos theta2 = cos45 → 
    Real.cos theta3 = -cos45 := by
  sorry

end NUMINAMATH_GPT_cosine_135_eq_neg_sqrt_2_div_2_l1424_142457


namespace NUMINAMATH_GPT_find_a_b_l1424_142496

theorem find_a_b (a b : ℝ)
  (h1 : a < 0)
  (h2 : (-b / a) = -((1 / 2) - (1 / 3)))
  (h3 : (2 / a) = -((1 / 2) * (1 / 3))) : 
  a + b = -14 :=
sorry

end NUMINAMATH_GPT_find_a_b_l1424_142496
