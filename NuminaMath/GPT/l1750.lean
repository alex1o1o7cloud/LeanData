import Mathlib

namespace NUMINAMATH_GPT_combined_age_of_Jane_and_John_in_future_l1750_175043

def Justin_age : ℕ := 26
def Jessica_age_when_Justin_born : ℕ := 6
def James_older_than_Jessica : ℕ := 7
def Julia_younger_than_Justin : ℕ := 8
def Jane_older_than_James : ℕ := 25
def John_older_than_Jane : ℕ := 3
def years_later : ℕ := 12

theorem combined_age_of_Jane_and_John_in_future :
  let Jessica_age := Justin_age + Jessica_age_when_Justin_born
  let James_age := Jessica_age + James_older_than_Jessica
  let Julia_age := Justin_age - Julia_younger_than_Justin
  let Jane_age := James_age + Jane_older_than_James
  let John_age := Jane_age + John_older_than_Jane
  let Jane_age_after_years := Jane_age + years_later
  let John_age_after_years := John_age + years_later
  Jane_age_after_years + John_age_after_years = 155 :=
by
  sorry

end NUMINAMATH_GPT_combined_age_of_Jane_and_John_in_future_l1750_175043


namespace NUMINAMATH_GPT_cos_210_eq_neg_sqrt3_div_2_l1750_175040

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (Real.pi + Real.pi / 6) = -Real.sqrt 3 / 2 := sorry

end NUMINAMATH_GPT_cos_210_eq_neg_sqrt3_div_2_l1750_175040


namespace NUMINAMATH_GPT_find_interval_n_l1750_175006

theorem find_interval_n 
  (n : ℕ) 
  (h1 : n < 500)
  (h2 : (∃ abcde : ℕ, 0 < abcde ∧ abcde < 99999 ∧ n * abcde = 99999))
  (h3 : (∃ uvw : ℕ, 0 < uvw ∧ uvw < 999 ∧ (n + 3) * uvw = 999)) 
  : 201 ≤ n ∧ n ≤ 300 := 
sorry

end NUMINAMATH_GPT_find_interval_n_l1750_175006


namespace NUMINAMATH_GPT_total_cost_of_motorcycle_l1750_175018

-- Definitions from conditions
def total_cost (x : ℝ) := 0.20 * x = 400

-- The theorem to prove
theorem total_cost_of_motorcycle (x : ℝ) (h : total_cost x) : x = 2000 := 
by
  sorry

end NUMINAMATH_GPT_total_cost_of_motorcycle_l1750_175018


namespace NUMINAMATH_GPT_train_speed_is_45_kmph_l1750_175081

noncomputable def speed_of_train_kmph (train_length bridge_length total_time : ℕ) : ℕ :=
  let total_distance := train_length + bridge_length
  let speed_mps := total_distance / total_time
  let speed_kmph := speed_mps * 36 / 10
  speed_kmph

theorem train_speed_is_45_kmph :
  speed_of_train_kmph 150 225 30 = 45 :=
  sorry

end NUMINAMATH_GPT_train_speed_is_45_kmph_l1750_175081


namespace NUMINAMATH_GPT_range_of_values_for_a_l1750_175096

theorem range_of_values_for_a (a : ℝ) :
  (∀ x : ℝ, a * x^2 + a * x - 1 < 0) ↔ (-4 < a ∧ a ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_range_of_values_for_a_l1750_175096


namespace NUMINAMATH_GPT_simplify_expression_l1750_175028

variable (a b : ℤ)

theorem simplify_expression :
  (30 * a + 45 * b) + (15 * a + 40 * b) - (20 * a + 55 * b) + (5 * a - 10 * b) = 30 * a + 20 * b :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1750_175028


namespace NUMINAMATH_GPT_determine_k_l1750_175046

theorem determine_k (a b c k : ℝ) (h : a + b + c = 1) (h_eq : k * (a + bc) = (a + b) * (a + c)) : k = 1 :=
sorry

end NUMINAMATH_GPT_determine_k_l1750_175046


namespace NUMINAMATH_GPT_remaining_paint_fraction_l1750_175049

def initial_paint : ℚ := 1

def paint_day_1 : ℚ := initial_paint - (1/2) * initial_paint
def paint_day_2 : ℚ := paint_day_1 - (1/4) * paint_day_1
def paint_day_3 : ℚ := paint_day_2 - (1/3) * paint_day_2

theorem remaining_paint_fraction : paint_day_3 = 1/4 :=
by
  sorry

end NUMINAMATH_GPT_remaining_paint_fraction_l1750_175049


namespace NUMINAMATH_GPT_number_of_students_in_first_class_l1750_175094

theorem number_of_students_in_first_class 
  (x : ℕ) -- number of students in the first class
  (avg_first_class : ℝ := 50) 
  (num_second_class : ℕ := 50)
  (avg_second_class : ℝ := 60)
  (avg_all_students : ℝ := 56.25)
  (total_avg_eqn : (avg_first_class * x + avg_second_class * num_second_class) / (x + num_second_class) = avg_all_students) : 
  x = 30 :=
by sorry

end NUMINAMATH_GPT_number_of_students_in_first_class_l1750_175094


namespace NUMINAMATH_GPT_new_area_shortening_other_side_l1750_175076

-- Define the dimensions of the original card
def original_length : ℕ := 5
def original_width : ℕ := 7

-- Define the shortened length and the resulting area after shortening one side by 2 inches
def shortened_length_1 := original_length - 2
def new_area_1 : ℕ := shortened_length_1 * original_width
def condition_1 : Prop := new_area_1 = 21

-- Prove that shortening the width by 2 inches results in an area of 25 square inches
theorem new_area_shortening_other_side : condition_1 → (original_length * (original_width - 2) = 25) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_new_area_shortening_other_side_l1750_175076


namespace NUMINAMATH_GPT_g_of_f_at_3_eq_1902_l1750_175009

def f (x : ℤ) : ℤ := x^3 - 2
def g (x : ℤ) : ℤ := 3 * x^2 + x + 2

theorem g_of_f_at_3_eq_1902 : g (f 3) = 1902 := by
  sorry

end NUMINAMATH_GPT_g_of_f_at_3_eq_1902_l1750_175009


namespace NUMINAMATH_GPT_BurjKhalifaHeight_l1750_175067

def SearsTowerHeight : ℕ := 527
def AdditionalHeight : ℕ := 303

theorem BurjKhalifaHeight : (SearsTowerHeight + AdditionalHeight) = 830 :=
by
  sorry

end NUMINAMATH_GPT_BurjKhalifaHeight_l1750_175067


namespace NUMINAMATH_GPT_factorize_quadratic_l1750_175044

theorem factorize_quadratic (x : ℝ) : 2*x^2 - 4*x + 2 = 2*(x-1)^2 :=
by
  sorry

end NUMINAMATH_GPT_factorize_quadratic_l1750_175044


namespace NUMINAMATH_GPT_product_divisible_by_eight_l1750_175001

theorem product_divisible_by_eight (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 96) : 
  8 ∣ n * (n + 1) * (n + 2) := 
sorry

end NUMINAMATH_GPT_product_divisible_by_eight_l1750_175001


namespace NUMINAMATH_GPT_largest_A_l1750_175062

theorem largest_A (A B C : ℕ) (h1 : A = 7 * B + C) (h2 : B = C) : A ≤ 48 :=
  sorry

end NUMINAMATH_GPT_largest_A_l1750_175062


namespace NUMINAMATH_GPT_find_y_l1750_175061

theorem find_y (x y : ℝ) (h : x = 180) (h1 : 0.25 * x = 0.10 * y - 5) : y = 500 :=
by sorry

end NUMINAMATH_GPT_find_y_l1750_175061


namespace NUMINAMATH_GPT_find_height_l1750_175002

-- Definitions from the problem conditions
def Area : ℕ := 442
def width : ℕ := 7
def length : ℕ := 8

-- The statement to prove
theorem find_height (h : ℕ) (H : 2 * length * width + 2 * length * h + 2 * width * h = Area) : h = 11 := 
by
  sorry

end NUMINAMATH_GPT_find_height_l1750_175002


namespace NUMINAMATH_GPT_expenses_neg_of_income_pos_l1750_175074

theorem expenses_neg_of_income_pos :
  ∀ (income expense : Int), income = 5 → expense = -income → expense = -5 :=
by
  intros income expense h_income h_expense
  rw [h_income] at h_expense
  exact h_expense

end NUMINAMATH_GPT_expenses_neg_of_income_pos_l1750_175074


namespace NUMINAMATH_GPT_find_monic_polynomial_of_shifted_roots_l1750_175024

theorem find_monic_polynomial_of_shifted_roots (a b c : ℝ) (h : ∀ x : ℝ, (x - a) * (x - b) * (x - c) = x^3 - 5 * x + 7) : 
  (x : ℝ) → (x - (a - 3)) * (x - (b - 3)) * (x - (c - 3)) = x^3 + 9 * x^2 + 22 * x + 19 :=
by
  -- Proof will be provided here.
  sorry

end NUMINAMATH_GPT_find_monic_polynomial_of_shifted_roots_l1750_175024


namespace NUMINAMATH_GPT_laura_total_miles_per_week_l1750_175097

def round_trip_school : ℕ := 20
def round_trip_supermarket : ℕ := 40
def round_trip_gym : ℕ := 10
def round_trip_friends_house : ℕ := 24

def school_trips_per_week : ℕ := 5
def supermarket_trips_per_week : ℕ := 2
def gym_trips_per_week : ℕ := 3
def friends_house_trips_per_week : ℕ := 1

def total_miles_driven_per_week :=
  round_trip_school * school_trips_per_week +
  round_trip_supermarket * supermarket_trips_per_week +
  round_trip_gym * gym_trips_per_week +
  round_trip_friends_house * friends_house_trips_per_week

theorem laura_total_miles_per_week : total_miles_driven_per_week = 234 :=
by
  sorry

end NUMINAMATH_GPT_laura_total_miles_per_week_l1750_175097


namespace NUMINAMATH_GPT_axis_of_symmetry_r_minus_2s_zero_l1750_175035

/-- 
Prove that if y = x is an axis of symmetry for the curve 
y = (2 * p * x + q) / (r * x - 2 * s) with p, q, r, s nonzero, 
then r - 2s = 0. 
-/
theorem axis_of_symmetry_r_minus_2s_zero
  (p q r s : ℝ) (h_p : p ≠ 0) (h_q : q ≠ 0) (h_r : r ≠ 0) (h_s : s ≠ 0) 
  (h_sym : ∀ (a b : ℝ), (b = (2 * p * a + q) / (r * a - 2 * s)) ↔ (a = (2 * p * b + q) / (r * b - 2 * s))) :
  r - 2 * s = 0 :=
sorry

end NUMINAMATH_GPT_axis_of_symmetry_r_minus_2s_zero_l1750_175035


namespace NUMINAMATH_GPT_find_a_extreme_value_at_2_l1750_175059

noncomputable def f (x : ℝ) (a : ℝ) := (2 / 3) * x^3 + a * x^2

theorem find_a_extreme_value_at_2 (a : ℝ) :
  (∀ x : ℝ, x ≠ 2 -> 0 = 2 * x^2 + 2 * a * x) ->
  (2 * 2^2 + 2 * a * 2 = 0) ->
  a = -2 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_a_extreme_value_at_2_l1750_175059


namespace NUMINAMATH_GPT_minimum_trees_with_at_least_three_types_l1750_175045

theorem minimum_trees_with_at_least_three_types 
    (total_trees : ℕ)
    (birches spruces pines aspens : ℕ)
    (h_total : total_trees = 100)
    (h_any_85 : ∀ (S : Finset ℕ), S.card = 85 → 
                  (∃ (b s p a : ℕ), b ∈ S ∧ s ∈ S ∧ p ∈ S ∧ a ∈ S)) :
  ∃ (n : ℕ), n = 69 ∧ ∀ (T : Finset ℕ), T.card = n → 
                  ∃ (b s p : ℕ), b ∈ T ∧ s ∈ T ∧ p ∈ T :=
  sorry

end NUMINAMATH_GPT_minimum_trees_with_at_least_three_types_l1750_175045


namespace NUMINAMATH_GPT_sphere_to_cube_volume_ratio_l1750_175021

noncomputable def volume_ratio (s : ℝ) : ℝ :=
  let r := s / 4
  let V_s := (4/3:ℝ) * Real.pi * r^3 
  let V_c := s^3
  V_s / V_c

theorem sphere_to_cube_volume_ratio (s : ℝ) (h : s > 0) : volume_ratio s = Real.pi / 48 := by
  sorry

end NUMINAMATH_GPT_sphere_to_cube_volume_ratio_l1750_175021


namespace NUMINAMATH_GPT_smaller_part_volume_l1750_175027

noncomputable def volume_of_smaller_part (a : ℝ) : ℝ :=
  (25 / 144) * (a^3)

theorem smaller_part_volume (a : ℝ) (h_pos : 0 < a) :
  ∃ v : ℝ, v = volume_of_smaller_part a :=
  sorry

end NUMINAMATH_GPT_smaller_part_volume_l1750_175027


namespace NUMINAMATH_GPT_largest_common_term_arith_progressions_l1750_175011

theorem largest_common_term_arith_progressions (a : ℕ) : 
  (∃ n m : ℕ, a = 4 + 5 * n ∧ a = 3 + 9 * m ∧ a < 1000) → a = 984 := by
  -- Proof is not required, so we add sorry.
  sorry

end NUMINAMATH_GPT_largest_common_term_arith_progressions_l1750_175011


namespace NUMINAMATH_GPT_teal_bluish_count_l1750_175042

theorem teal_bluish_count (n G Bg N B : ℕ) (h1 : n = 120) (h2 : G = 80) (h3 : Bg = 35) (h4 : N = 20) :
  B = 55 :=
by
  sorry

end NUMINAMATH_GPT_teal_bluish_count_l1750_175042


namespace NUMINAMATH_GPT_triangles_pentagons_difference_l1750_175053

theorem triangles_pentagons_difference :
  ∃ x y : ℕ, 
  (x + y = 50) ∧ (3 * x + 5 * y = 170) ∧ (x - y = 30) :=
sorry

end NUMINAMATH_GPT_triangles_pentagons_difference_l1750_175053


namespace NUMINAMATH_GPT_value_of_f_neg_2_l1750_175034

section
variable {f : ℝ → ℝ}
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_pos : ∀ x : ℝ, 0 < x → f x = 2 ^ x + 1)

theorem value_of_f_neg_2 (h_odd : ∀ x, f (-x) = -f x) (h_pos : ∀ x, 0 < x → f x = 2^x + 1) :
  f (-2) = -5 :=
by
  sorry
end

end NUMINAMATH_GPT_value_of_f_neg_2_l1750_175034


namespace NUMINAMATH_GPT_negation_propositional_logic_l1750_175080

theorem negation_propositional_logic :
  ¬ (∀ x : ℝ, x^2 + x + 1 < 0) ↔ ∃ x : ℝ, x^2 + x + 1 ≥ 0 :=
by sorry

end NUMINAMATH_GPT_negation_propositional_logic_l1750_175080


namespace NUMINAMATH_GPT_interior_edges_sum_l1750_175031

-- Definitions based on conditions
def frame_width : ℕ := 2
def frame_area : ℕ := 32
def outer_edge_length : ℕ := 8

-- Mathematically equivalent proof problem
theorem interior_edges_sum :
  ∃ (y : ℕ),  (frame_width * 2) * (y - frame_width * 2) = 32 ∧ (outer_edge_length * y - (outer_edge_length - 2 * frame_width) * (y - 2 * frame_width)) = 32 -> 4 + 4 + 0 + 0 = 8 :=
sorry

end NUMINAMATH_GPT_interior_edges_sum_l1750_175031


namespace NUMINAMATH_GPT_adult_ticket_cost_l1750_175072

variable (A : ℝ)

theorem adult_ticket_cost :
  (20 * 6) + (12 * A) = 216 → A = 8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_adult_ticket_cost_l1750_175072


namespace NUMINAMATH_GPT_jet_flight_distance_l1750_175068

-- Setting up the hypotheses and the statement
theorem jet_flight_distance (v d : ℕ) (h1 : d = 4 * (v + 50)) (h2 : d = 5 * (v - 50)) : d = 2000 :=
sorry

end NUMINAMATH_GPT_jet_flight_distance_l1750_175068


namespace NUMINAMATH_GPT_total_material_weight_l1750_175050

def gravel_weight : ℝ := 5.91
def sand_weight : ℝ := 8.11

theorem total_material_weight : gravel_weight + sand_weight = 14.02 := by
  sorry

end NUMINAMATH_GPT_total_material_weight_l1750_175050


namespace NUMINAMATH_GPT_layers_removed_l1750_175075

theorem layers_removed (n : ℕ) (original_volume remaining_volume side_length : ℕ) :
  original_volume = side_length^3 →
  remaining_volume = (side_length - 2 * n)^3 →
  original_volume = 1000 →
  remaining_volume = 512 →
  side_length = 10 →
  n = 1 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_layers_removed_l1750_175075


namespace NUMINAMATH_GPT_sqrt_abs_eq_zero_imp_power_eq_neg_one_l1750_175026

theorem sqrt_abs_eq_zero_imp_power_eq_neg_one (m n : ℤ) (h : (Real.sqrt (m - 2) + abs (n + 3) = 0)) : (m + n) ^ 2023 = -1 := by
  sorry

end NUMINAMATH_GPT_sqrt_abs_eq_zero_imp_power_eq_neg_one_l1750_175026


namespace NUMINAMATH_GPT_area_expression_l1750_175058

noncomputable def overlapping_area (m : ℝ) (h1 : 0 < m) (h2 : m < 4 * Real.sqrt 2) : ℝ :=
if h : m ≤ 2 * Real.sqrt 2 then
  6 - Real.sqrt 2 * m
else
  (1 / 4) * m^2 - 2 * Real.sqrt 2 * m + 8

theorem area_expression (m : ℝ) (h1 : 0 < m) (h2 : m < 4 * Real.sqrt 2) :
  let y := overlapping_area m h1 h2
  (if h : m ≤ 2 * Real.sqrt 2 then y = 6 - Real.sqrt 2 * m
   else y = (1 / 4) * m^2 - 2 * Real.sqrt 2 * m + 8) := 
sorry

end NUMINAMATH_GPT_area_expression_l1750_175058


namespace NUMINAMATH_GPT_prime_triple_l1750_175003

theorem prime_triple (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r)
  (h1 : p ∣ (q * r - 1)) (h2 : q ∣ (p * r - 1)) (h3 : r ∣ (p * q - 1)) :
  (p = 2 ∧ q = 3 ∧ r = 5) ∨ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 3 ∧ q = 2 ∧ r = 5) ∨ (p = 3 ∧ q = 5 ∧ r = 2) ∨ (p = 5 ∧ q = 2 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) :=
sorry

end NUMINAMATH_GPT_prime_triple_l1750_175003


namespace NUMINAMATH_GPT_solve_fraction_equation_l1750_175007

theorem solve_fraction_equation (x : ℝ) (h : (x + 5) / (x - 3) = 4) : x = 17 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_fraction_equation_l1750_175007


namespace NUMINAMATH_GPT_proof_problem_l1750_175048

theorem proof_problem (p q r : ℝ) 
  (h1 : p + q = 20)
  (h2 : p * q = 144) 
  (h3 : q + r = 52) 
  (h4 : 4 * (r + p) = r * p) : 
  r - p = 32 := 
sorry

end NUMINAMATH_GPT_proof_problem_l1750_175048


namespace NUMINAMATH_GPT_solve_x_l1750_175008

theorem solve_x (x : ℝ) (h : (30 * x + 15)^(1/3) = 15) : x = 112 := by
  sorry

end NUMINAMATH_GPT_solve_x_l1750_175008


namespace NUMINAMATH_GPT_correct_option_C_l1750_175030

def number_of_stamps : String := "the number of the stamps"
def number_of_people : String := "a number of people"

def is_singular (subject : String) : Prop := subject = number_of_stamps
def is_plural (subject : String) : Prop := subject = number_of_people

def correct_sentence (verb1 verb2 : String) : Prop :=
  verb1 = "is" ∧ verb2 = "want"

theorem correct_option_C : correct_sentence "is" "want" :=
by
  show correct_sentence "is" "want"
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_correct_option_C_l1750_175030


namespace NUMINAMATH_GPT_root_integer_l1750_175084

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

def is_root (x_0 : ℝ) : Prop := f x_0 = 0

theorem root_integer (x_0 : ℝ) (h : is_root x_0) : Int.floor x_0 = 2 := by
  sorry

end NUMINAMATH_GPT_root_integer_l1750_175084


namespace NUMINAMATH_GPT_wire_length_before_cut_l1750_175066

-- Defining the conditions
def wire_cut (L S : ℕ) : Prop :=
  S = 20 ∧ S = (2 / 5 : ℚ) * L

-- The statement we need to prove
theorem wire_length_before_cut (L S : ℕ) (h : wire_cut L S) : (L + S) = 70 := 
by 
  sorry

end NUMINAMATH_GPT_wire_length_before_cut_l1750_175066


namespace NUMINAMATH_GPT_find_a_l1750_175052

theorem find_a (A B : Real) (b a : Real) (hA : A = 45) (hB : B = 60) (hb : b = Real.sqrt 3) : 
  a = Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_find_a_l1750_175052


namespace NUMINAMATH_GPT_total_team_cost_l1750_175064

-- Define the costs of individual items and the number of players
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.2
def socks_cost : ℝ := 6.8
def num_players : ℕ := 16

-- Define the total cost for equipment for one player
def player_cost : ℝ :=
  jersey_cost + shorts_cost + socks_cost

-- The main theorem stating the total cost for all players
theorem total_team_cost : num_players * player_cost = 752 := by
  sorry

end NUMINAMATH_GPT_total_team_cost_l1750_175064


namespace NUMINAMATH_GPT_rate_of_Y_l1750_175032

noncomputable def rate_X : ℝ := 2
noncomputable def time_to_cross : ℝ := 0.5

theorem rate_of_Y (rate_Y : ℝ) : rate_X * time_to_cross = 1 → rate_Y * time_to_cross = 1 → rate_Y = rate_X :=
by
    intros h_rate_X h_rate_Y
    sorry

end NUMINAMATH_GPT_rate_of_Y_l1750_175032


namespace NUMINAMATH_GPT_original_cone_volume_l1750_175060

theorem original_cone_volume
  (H R h r : ℝ)
  (Vcylinder : ℝ) (Vfrustum : ℝ)
  (cylinder_volume : Vcylinder = π * r^2 * h)
  (frustum_volume : Vfrustum = (1 / 3) * π * (R^2 + R * r + r^2) * (H - h))
  (Vcylinder_value : Vcylinder = 9)
  (Vfrustum_value : Vfrustum = 63) :
  (1 / 3) * π * R^2 * H = 64 :=
by
  sorry

end NUMINAMATH_GPT_original_cone_volume_l1750_175060


namespace NUMINAMATH_GPT_distance_from_A_to_O_is_3_l1750_175029

-- Define polar coordinates with the given conditions
def point_A : ℝ × ℝ := (3, -4)

-- Define the distance function in terms of polar coordinates
def distance_to_pole_O (coords : ℝ × ℝ) : ℝ := coords.1

-- The main theorem to be proved
theorem distance_from_A_to_O_is_3 : distance_to_pole_O point_A = 3 := by
  sorry

end NUMINAMATH_GPT_distance_from_A_to_O_is_3_l1750_175029


namespace NUMINAMATH_GPT_area_inside_S_outside_R_l1750_175063

theorem area_inside_S_outside_R (area_R area_S : ℝ) (h1: area_R = 1 + 3 * Real.sqrt 3) (h2: area_S = 6 * Real.sqrt 3) :
  area_S - area_R = 1 :=
by {
   sorry
}

end NUMINAMATH_GPT_area_inside_S_outside_R_l1750_175063


namespace NUMINAMATH_GPT_polynomial_coefficient_product_identity_l1750_175005

theorem polynomial_coefficient_product_identity (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ)
  (h1 : a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 0)
  (h2 : a_0 - a_1 + a_2 - a_3 + a_4 - a_5 = 32) :
  (a_0 + a_2 + a_4) * (a_1 + a_3 + a_5) = -256 := 
by {
  sorry
}

end NUMINAMATH_GPT_polynomial_coefficient_product_identity_l1750_175005


namespace NUMINAMATH_GPT_work_done_in_a_day_l1750_175022

noncomputable def A : ℕ := sorry
noncomputable def B_days : ℕ := A / 2

theorem work_done_in_a_day (h : 1 / A + 2 / A = 1 / 6) : A = 18 := 
by 
  -- skipping the proof as instructed
  sorry

end NUMINAMATH_GPT_work_done_in_a_day_l1750_175022


namespace NUMINAMATH_GPT_reduced_bucket_fraction_l1750_175055

theorem reduced_bucket_fraction (C : ℝ) (F : ℝ) (h : 25 * F * C = 10 * C) : F = 2 / 5 :=
by sorry

end NUMINAMATH_GPT_reduced_bucket_fraction_l1750_175055


namespace NUMINAMATH_GPT_line_not_in_fourth_quadrant_l1750_175083

-- Let the line be defined as y = 3x + 2
def line_eq (x : ℝ) : ℝ := 3 * x + 2

-- The Fourth quadrant is defined by x > 0 and y < 0
def in_fourth_quadrant (x : ℝ) (y : ℝ) : Prop := x > 0 ∧ y < 0

-- Prove that the line does not intersect the Fourth quadrant
theorem line_not_in_fourth_quadrant : ¬ (∃ x : ℝ, in_fourth_quadrant x (line_eq x)) :=
by
  -- Proof goes here (abstracted)
  sorry

end NUMINAMATH_GPT_line_not_in_fourth_quadrant_l1750_175083


namespace NUMINAMATH_GPT_dacid_weighted_average_l1750_175077

theorem dacid_weighted_average :
  let english := 96
  let mathematics := 95
  let physics := 82
  let chemistry := 87
  let biology := 92
  let weight_english := 0.20
  let weight_mathematics := 0.25
  let weight_physics := 0.15
  let weight_chemistry := 0.25
  let weight_biology := 0.15
  (english * weight_english) + (mathematics * weight_mathematics) +
  (physics * weight_physics) + (chemistry * weight_chemistry) +
  (biology * weight_biology) = 90.8 :=
by
  sorry

end NUMINAMATH_GPT_dacid_weighted_average_l1750_175077


namespace NUMINAMATH_GPT_infinity_gcd_binom_l1750_175054

theorem infinity_gcd_binom {k l : ℕ} : ∃ᶠ m in at_top, m ≥ k ∧ Nat.gcd (Nat.choose m k) l = 1 :=
sorry

end NUMINAMATH_GPT_infinity_gcd_binom_l1750_175054


namespace NUMINAMATH_GPT_volume_range_l1750_175037

theorem volume_range (a b c : ℝ) (h1 : a + b + c = 9)
  (h2 : a * b + b * c + a * c = 24) : 16 ≤ a * b * c ∧ a * b * c ≤ 20 :=
by {
  -- Proof would go here
  sorry
}

end NUMINAMATH_GPT_volume_range_l1750_175037


namespace NUMINAMATH_GPT_inequality_solution_l1750_175070

theorem inequality_solution (x : ℝ) 
  (hx1 : x ≠ 1) 
  (hx2 : x ≠ 2) 
  (hx3 : x ≠ 3) 
  (hx4 : x ≠ 4) :
  (1 / (x - 1) - 3 / (x - 2) + 5 / (x - 3) - 1 / (x - 4) < 1 / 24) ↔ (x ∈ Set.Ioo (-7 : ℝ) 1 ∪ Set.Ioo 3 4) := 
sorry

end NUMINAMATH_GPT_inequality_solution_l1750_175070


namespace NUMINAMATH_GPT_time_to_cut_womans_hair_l1750_175010

theorem time_to_cut_womans_hair 
  (WL : ℕ) (WM : ℕ) (WK : ℕ) (total_time : ℕ) 
  (num_women : ℕ) (num_men : ℕ) (num_kids : ℕ) 
  (men_haircut_time : ℕ) (kids_haircut_time : ℕ) 
  (overall_time : ℕ) :
  men_haircut_time = 15 →
  kids_haircut_time = 25 →
  num_women = 3 →
  num_men = 2 →
  num_kids = 3 →
  overall_time = 255 →
  overall_time = (num_women * WL + num_men * men_haircut_time + num_kids * kids_haircut_time) →
  WL = 50 :=
by
  sorry

end NUMINAMATH_GPT_time_to_cut_womans_hair_l1750_175010


namespace NUMINAMATH_GPT_new_energy_vehicle_price_l1750_175004

theorem new_energy_vehicle_price (x : ℝ) :
  (5000 / (x + 1)) = (5000 * (1 - 0.2)) / x :=
sorry

end NUMINAMATH_GPT_new_energy_vehicle_price_l1750_175004


namespace NUMINAMATH_GPT_div_of_floats_l1750_175078

theorem div_of_floats : (0.2 : ℝ) / (0.005 : ℝ) = 40 := 
by
  sorry

end NUMINAMATH_GPT_div_of_floats_l1750_175078


namespace NUMINAMATH_GPT_hypotenuse_length_l1750_175012

theorem hypotenuse_length
  (a b c : ℝ)
  (h1 : a + b + c = 40)
  (h2 : (1 / 2) * a * b = 24)
  (h3 : a^2 + b^2 = c^2) :
  c = 18.8 :=
by sorry

end NUMINAMATH_GPT_hypotenuse_length_l1750_175012


namespace NUMINAMATH_GPT_range_f_l1750_175039

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 1)

theorem range_f : Set.Ioo 0 1 ∪ {1} = {y : ℝ | ∃ x : ℝ, f x = y} :=
by 
  sorry

end NUMINAMATH_GPT_range_f_l1750_175039


namespace NUMINAMATH_GPT_h_inverse_correct_l1750_175086

noncomputable def f (x : ℝ) := 4 * x + 7
noncomputable def g (x : ℝ) := 3 * x - 2
noncomputable def h (x : ℝ) := f (g x)
noncomputable def h_inv (y : ℝ) := (y + 1) / 12

theorem h_inverse_correct : ∀ x : ℝ, h_inv (h x) = x :=
by
  intro x
  sorry

end NUMINAMATH_GPT_h_inverse_correct_l1750_175086


namespace NUMINAMATH_GPT_heavy_equipment_pay_l1750_175090

theorem heavy_equipment_pay
  (total_workers : ℕ)
  (total_payroll : ℕ)
  (laborers : ℕ)
  (laborer_pay : ℕ)
  (heavy_operator_pay : ℕ)
  (h1 : total_workers = 35)
  (h2 : total_payroll = 3950)
  (h3 : laborers = 19)
  (h4 : laborer_pay = 90)
  (h5 : (total_workers - laborers) * heavy_operator_pay + laborers * laborer_pay = total_payroll) :
  heavy_operator_pay = 140 :=
by
  sorry

end NUMINAMATH_GPT_heavy_equipment_pay_l1750_175090


namespace NUMINAMATH_GPT_sphere_surface_area_l1750_175047

theorem sphere_surface_area
  (a : ℝ)
  (expansion : (1 - 2 * 1 : ℝ)^6 = a)
  (a_value : a = 1) :
  4 * Real.pi * ((Real.sqrt (2^2 + 3^2 + a^2) / 2)^2) = 14 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_sphere_surface_area_l1750_175047


namespace NUMINAMATH_GPT_martha_makes_40_cookies_martha_needs_7_5_cups_l1750_175051

theorem martha_makes_40_cookies :
  (24 / 3) * 5 = 40 :=
by
  sorry

theorem martha_needs_7_5_cups :
  60 / (24 / 3) = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_martha_makes_40_cookies_martha_needs_7_5_cups_l1750_175051


namespace NUMINAMATH_GPT_sample_std_dev_range_same_l1750_175056

noncomputable def sample_std_dev (data : List ℝ) : ℝ := sorry
noncomputable def sample_range (data : List ℝ) : ℝ := sorry

theorem sample_std_dev_range_same (n : ℕ) (c : ℝ) (Hc : c ≠ 0) (x : Fin n → ℝ) :
  sample_std_dev (List.ofFn (λ i => x i)) = sample_std_dev (List.ofFn (λ i => x i + c)) ∧
  sample_range (List.ofFn (λ i => x i)) = sample_range (List.ofFn (λ i => x i + c)) :=
by
  sorry

end NUMINAMATH_GPT_sample_std_dev_range_same_l1750_175056


namespace NUMINAMATH_GPT_celia_receives_correct_amount_of_aranha_l1750_175085

def borboleta_to_tubarao (b : Int) : Int := 3 * b
def tubarao_to_periquito (t : Int) : Int := 2 * t
def periquito_to_aranha (p : Int) : Int := 3 * p
def macaco_to_aranha (m : Int) : Int := 4 * m
def cobra_to_periquito (c : Int) : Int := 3 * c

def celia_stickers_to_aranha (borboleta tubarao cobra periquito macaco : Int) : Int :=
  let borboleta_to_aranha := periquito_to_aranha (tubarao_to_periquito (borboleta_to_tubarao borboleta))
  let tubarao_to_aranha := periquito_to_aranha (tubarao_to_periquito tubarao)
  let cobra_to_aranha := periquito_to_aranha (cobra_to_periquito cobra)
  let periquito_to_aranha := periquito_to_aranha periquito
  let macaco_to_aranha := macaco_to_aranha macaco
  borboleta_to_aranha + tubarao_to_aranha + cobra_to_aranha + periquito_to_aranha + macaco_to_aranha

theorem celia_receives_correct_amount_of_aranha : 
  celia_stickers_to_aranha 4 5 3 6 6 = 171 := 
by
  simp only [celia_stickers_to_aranha, borboleta_to_tubarao, tubarao_to_periquito, periquito_to_aranha, cobra_to_periquito, macaco_to_aranha]
  -- Here we need to perform the arithmetic steps to verify the sum
  sorry -- This is the placeholder for the actual proof

end NUMINAMATH_GPT_celia_receives_correct_amount_of_aranha_l1750_175085


namespace NUMINAMATH_GPT_perfect_square_iff_divisibility_l1750_175098

theorem perfect_square_iff_divisibility (A : ℕ) :
  (∃ d : ℕ, A = d^2) ↔ ∀ n : ℕ, n > 0 → ∃ j : ℕ, 1 ≤ j ∧ j ≤ n ∧ n ∣ (A + j)^2 - A :=
sorry

end NUMINAMATH_GPT_perfect_square_iff_divisibility_l1750_175098


namespace NUMINAMATH_GPT_floor_div_eq_floor_div_l1750_175091

theorem floor_div_eq_floor_div
  (a : ℝ) (n : ℤ) (ha_pos : 0 < a) :
  (⌊⌊a⌋ / n⌋ : ℤ) = ⌊a / n⌋ := 
sorry

end NUMINAMATH_GPT_floor_div_eq_floor_div_l1750_175091


namespace NUMINAMATH_GPT_find_p1_plus_q1_l1750_175038

noncomputable def p (x : ℤ) := x^4 + 14 * x^2 + 1
noncomputable def q (x : ℤ) := x^4 - 14 * x^2 + 1

theorem find_p1_plus_q1 :
  (p 1) + (q 1) = 4 :=
sorry

end NUMINAMATH_GPT_find_p1_plus_q1_l1750_175038


namespace NUMINAMATH_GPT_average_percentage_of_first_20_percent_l1750_175088

theorem average_percentage_of_first_20_percent (X : ℝ) 
  (h1 : 0.20 * X + 0.50 * 60 + 0.30 * 40 = 58) : 
  X = 80 :=
sorry

end NUMINAMATH_GPT_average_percentage_of_first_20_percent_l1750_175088


namespace NUMINAMATH_GPT_throwers_count_l1750_175092

variable (totalPlayers : ℕ) (rightHandedPlayers : ℕ) (nonThrowerLeftHandedFraction nonThrowerRightHandedFraction : ℚ)

theorem throwers_count
  (h1 : totalPlayers = 70)
  (h2 : rightHandedPlayers = 64)
  (h3 : nonThrowerLeftHandedFraction = 1 / 3)
  (h4 : nonThrowerRightHandedFraction = 2 / 3)
  (h5 : nonThrowerLeftHandedFraction + nonThrowerRightHandedFraction = 1) : 
  ∃ T : ℕ, T = 52 := by
  sorry

end NUMINAMATH_GPT_throwers_count_l1750_175092


namespace NUMINAMATH_GPT_sum_six_smallest_multiples_of_12_is_252_l1750_175025

-- Define the six smallest positive distinct multiples of 12
def six_smallest_multiples_of_12 := [12, 24, 36, 48, 60, 72]

-- Define the sum problem
def sum_of_six_smallest_multiples_of_12 : Nat :=
  six_smallest_multiples_of_12.foldr (· + ·) 0

-- Main proof statement
theorem sum_six_smallest_multiples_of_12_is_252 :
  sum_of_six_smallest_multiples_of_12 = 252 :=
by
  sorry

end NUMINAMATH_GPT_sum_six_smallest_multiples_of_12_is_252_l1750_175025


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1750_175095

theorem simplify_and_evaluate (a : ℚ) (h : a = 3) :
  (1 - (a - 2) / (a^2 - 4)) / ((a^2 + a) / (a^2 + 4*a + 4)) = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1750_175095


namespace NUMINAMATH_GPT_line_passes_vertex_parabola_l1750_175065

theorem line_passes_vertex_parabola :
  ∃ (b₁ b₂ : ℚ), (b₁ ≠ b₂) ∧ (∀ b, (b = b₁ ∨ b = b₂) → 
    (∃ x y, y = x + b ∧ y = x^2 + 4 * b^2 ∧ x = 0 ∧ y = 4 * b^2)) :=
by 
  sorry

end NUMINAMATH_GPT_line_passes_vertex_parabola_l1750_175065


namespace NUMINAMATH_GPT_pyramid_volume_l1750_175000

-- Define the given conditions
def regular_octagon (A B C D E F G H : Point) : Prop := sorry
def right_pyramid (P A B C D E F G H : Point) : Prop := sorry
def equilateral_triangle (P A D : Point) (side_length : ℝ) : Prop := sorry

-- Define the specific pyramid problem with all the given conditions
noncomputable def volume_pyramid (P A B C D E F G H : Point) (height : ℝ) (base_area : ℝ) : ℝ :=
  (1 / 3) * base_area * height

-- The main theorem to prove the volume of the pyramid
theorem pyramid_volume (A B C D E F G H P : Point) 
(h1 : regular_octagon A B C D E F G H)
(h2 : right_pyramid P A B C D E F G H)
(h3 : equilateral_triangle P A D 10) :
  volume_pyramid P A B C D E F G H (5 * Real.sqrt 3) (50 * Real.sqrt 3) = 250 := 
sorry

end NUMINAMATH_GPT_pyramid_volume_l1750_175000


namespace NUMINAMATH_GPT_John_has_22_quarters_l1750_175073

variable (q d n : ℕ)

-- Conditions
axiom cond1 : d = q + 3
axiom cond2 : n = q - 6
axiom cond3 : q + d + n = 63

theorem John_has_22_quarters : q = 22 := by
  sorry

end NUMINAMATH_GPT_John_has_22_quarters_l1750_175073


namespace NUMINAMATH_GPT_number_division_l1750_175013

theorem number_division (m k n : ℤ) (h : n = m * k + 1) : n = m * k + 1 :=
by
  exact h

end NUMINAMATH_GPT_number_division_l1750_175013


namespace NUMINAMATH_GPT_price_decrease_proof_l1750_175019

-- Definitions based on the conditions
def original_price (C : ℝ) : ℝ := C
def new_price (C : ℝ) : ℝ := 0.76 * C

theorem price_decrease_proof (C : ℝ) : new_price C = 421.05263157894734 :=
by
  sorry

end NUMINAMATH_GPT_price_decrease_proof_l1750_175019


namespace NUMINAMATH_GPT_adam_spent_on_ferris_wheel_l1750_175069

theorem adam_spent_on_ferris_wheel (t_initial t_left t_price : ℕ) (h1 : t_initial = 13)
  (h2 : t_left = 4) (h3 : t_price = 9) : t_initial - t_left = 9 ∧ (t_initial - t_left) * t_price = 81 := 
by
  sorry

end NUMINAMATH_GPT_adam_spent_on_ferris_wheel_l1750_175069


namespace NUMINAMATH_GPT_only_three_A_l1750_175017

def student := Type
variable (Alan Beth Carlos Diana Eliza : student)

variable (gets_A : student → Prop)

variable (H1 : gets_A Alan → gets_A Beth)
variable (H2 : gets_A Beth → gets_A Carlos)
variable (H3 : gets_A Carlos → gets_A Diana)
variable (H4 : gets_A Diana → gets_A Eliza)
variable (H5 : gets_A Eliza → gets_A Alan)
variable (H6 : ∃ a b c : student, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ gets_A a ∧ gets_A b ∧ gets_A c ∧ ∀ d : student, gets_A d → d = a ∨ d = b ∨ d = c)

theorem only_three_A : gets_A Carlos ∧ gets_A Diana ∧ gets_A Eliza :=
by
  sorry

end NUMINAMATH_GPT_only_three_A_l1750_175017


namespace NUMINAMATH_GPT_bottle_caps_per_visit_l1750_175020

-- Define the given conditions
def total_bottle_caps : ℕ := 25
def number_of_visits : ℕ := 5

-- The statement we want to prove
theorem bottle_caps_per_visit :
  total_bottle_caps / number_of_visits = 5 :=
sorry

end NUMINAMATH_GPT_bottle_caps_per_visit_l1750_175020


namespace NUMINAMATH_GPT_shift_gives_f_l1750_175014

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 6)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x)

theorem shift_gives_f :
  (∀ x, f x = g (x + Real.pi / 3)) :=
  by
  sorry

end NUMINAMATH_GPT_shift_gives_f_l1750_175014


namespace NUMINAMATH_GPT_determinant_transformation_l1750_175099

theorem determinant_transformation (p q r s : ℝ) (h : p * s - q * r = -3) :
    p * (5 * r + 2 * s) - r * (5 * p + 2 * q) = -6 := by
  sorry

end NUMINAMATH_GPT_determinant_transformation_l1750_175099


namespace NUMINAMATH_GPT_determine_a_range_f_l1750_175057

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - (2 / (2 ^ x + 1))

theorem determine_a (a : ℝ) : (∀ x : ℝ, f a (-x) = -f a x) -> a = 1 :=
by
  sorry

theorem range_f (x : ℝ) : (∀ x : ℝ, f 1 (-x) = -f 1 x) -> -1 < f 1 x ∧ f 1 x < 1 :=
by
  sorry

end NUMINAMATH_GPT_determine_a_range_f_l1750_175057


namespace NUMINAMATH_GPT_smaller_consecutive_number_divisibility_l1750_175071

theorem smaller_consecutive_number_divisibility :
  ∃ (m : ℕ), (m < m + 1) ∧ (1 ≤ m ∧ m ≤ 200) ∧ (1 ≤ m + 1 ∧ m + 1 ≤ 200) ∧
              (∀ n, (1 ≤ n ∧ n ≤ 200 ∧ n ≠ m ∧ n ≠ m + 1) → ∃ k, chosen_num = k * n) ∧
              (128 = m) :=
sorry

end NUMINAMATH_GPT_smaller_consecutive_number_divisibility_l1750_175071


namespace NUMINAMATH_GPT_range_of_x_when_y_lt_0_l1750_175093

variable (a b c n m : ℝ)

-- The definition of the quadratic function
def quadratic_function (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions given in the problem
axiom value_at_neg1 : quadratic_function a b c (-1) = 4
axiom value_at_0 : quadratic_function a b c 0 = 0
axiom value_at_1 : quadratic_function a b c 1 = n
axiom value_at_2 : quadratic_function a b c 2 = m
axiom value_at_3 : quadratic_function a b c 3 = 4

-- Proof statement
theorem range_of_x_when_y_lt_0 : ∀ (x : ℝ), quadratic_function a b c x < 0 ↔ 0 < x ∧ x < 2 :=
sorry

end NUMINAMATH_GPT_range_of_x_when_y_lt_0_l1750_175093


namespace NUMINAMATH_GPT_bob_speed_l1750_175041

theorem bob_speed (j_speed : ℝ) (b_headstart : ℝ) (t : ℝ) (j_catches_up : t = 20 / 60 ∧ j_speed = 9 ∧ b_headstart = 1) : 
  ∃ b_speed : ℝ, b_speed = 6 := 
by
  sorry

end NUMINAMATH_GPT_bob_speed_l1750_175041


namespace NUMINAMATH_GPT_abs_sum_lt_abs_sum_of_neg_product_l1750_175023

theorem abs_sum_lt_abs_sum_of_neg_product 
  (a b : ℝ) : ab < 0 ↔ |a + b| < |a| + |b| := 
by 
  sorry

end NUMINAMATH_GPT_abs_sum_lt_abs_sum_of_neg_product_l1750_175023


namespace NUMINAMATH_GPT_work_done_l1750_175016

theorem work_done (m : ℕ) : 18 * 30 = m * 36 → m = 15 :=
by
  intro h  -- assume the equality condition
  have h1 : m = 15 := by
    -- We would solve for m here similarly to the solution given to derive 15
    sorry
  exact h1

end NUMINAMATH_GPT_work_done_l1750_175016


namespace NUMINAMATH_GPT_solve_equation_l1750_175082

theorem solve_equation :
  ∀ x : ℝ, (3 * x^2 / (x - 2) - (3 * x + 4) / 2 + (5 - 9 * x) / (x - 2) + 2 = 0) →
    (x = 2 + Real.sqrt 6 ∨ x = 2 - Real.sqrt 6) :=
by
  intro x h
  -- the proof would go here
  sorry

end NUMINAMATH_GPT_solve_equation_l1750_175082


namespace NUMINAMATH_GPT_average_weight_of_all_girls_l1750_175033

theorem average_weight_of_all_girls (avg1 : ℝ) (n1 : ℕ) (avg2 : ℝ) (n2 : ℕ) :
  avg1 = 50.25 → n1 = 16 → avg2 = 45.15 → n2 = 8 → 
  ((n1 * avg1 + n2 * avg2) / (n1 + n2)) = 48.55 := 
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_average_weight_of_all_girls_l1750_175033


namespace NUMINAMATH_GPT_total_weight_of_fruits_l1750_175015

/-- Define the given conditions in Lean -/
def weight_of_orange_bags (n : ℕ) : ℝ :=
  if n = 12 then 24 else 0

def weight_of_apple_bags (n : ℕ) : ℝ :=
  if n = 8 then 30 else 0

/-- Prove that the total weight of 5 bags of oranges and 4 bags of apples is 25 pounds given the conditions -/
theorem total_weight_of_fruits :
  weight_of_orange_bags 12 / 12 * 5 + weight_of_apple_bags 8 / 8 * 4 = 25 :=
by sorry

end NUMINAMATH_GPT_total_weight_of_fruits_l1750_175015


namespace NUMINAMATH_GPT_sum_of_six_terms_arithmetic_sequence_l1750_175087

theorem sum_of_six_terms_arithmetic_sequence (S : ℕ → ℕ)
    (h1 : S 2 = 2)
    (h2 : S 4 = 10) :
    S 6 = 42 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_six_terms_arithmetic_sequence_l1750_175087


namespace NUMINAMATH_GPT_problem_abcd_eq_14400_l1750_175079

theorem problem_abcd_eq_14400
 (a b c d : ℝ)
 (h1 : a^2 + b^2 + c^2 + d^2 = 762)
 (h2 : a * b + c * d = 260)
 (h3 : a * c + b * d = 365)
 (h4 : a * d + b * c = 244) :
 a * b * c * d = 14400 := 
sorry

end NUMINAMATH_GPT_problem_abcd_eq_14400_l1750_175079


namespace NUMINAMATH_GPT_cost_of_largest_pot_l1750_175036

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

end NUMINAMATH_GPT_cost_of_largest_pot_l1750_175036


namespace NUMINAMATH_GPT_toothpick_grid_l1750_175089

theorem toothpick_grid (l w : ℕ) (h_l : l = 45) (h_w : w = 25) :
  let effective_vertical_lines := l + 1 - (l + 1) / 5
  let effective_horizontal_lines := w + 1 - (w + 1) / 5
  let vertical_toothpicks := effective_vertical_lines * w
  let horizontal_toothpicks := effective_horizontal_lines * l
  let total_toothpicks := vertical_toothpicks + horizontal_toothpicks
  total_toothpicks = 1722 :=
by {
  sorry
}

end NUMINAMATH_GPT_toothpick_grid_l1750_175089
