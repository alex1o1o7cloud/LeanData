import Mathlib

namespace NUMINAMATH_GPT_closest_point_on_plane_exists_l2177_217793

def point_on_plane : Type := {P : ℝ × ℝ × ℝ // ∃ (x y z : ℝ), P = (x, y, z) ∧ 2 * x - 3 * y + 4 * z = 20}

def point_A : ℝ × ℝ × ℝ := (0, 1, -1)

theorem closest_point_on_plane_exists (P : point_on_plane) :
  ∃ (x y z : ℝ), (x, y, z) = (54 / 29, -80 / 29, 83 / 29) := sorry

end NUMINAMATH_GPT_closest_point_on_plane_exists_l2177_217793


namespace NUMINAMATH_GPT_correct_reasoning_l2177_217781

-- Define that every multiple of 9 is a multiple of 3
def multiple_of_9_is_multiple_of_3 : Prop :=
  ∀ n : ℤ, n % 9 = 0 → n % 3 = 0

-- Define that a certain odd number is a multiple of 9
def odd_multiple_of_9 (n : ℤ) : Prop :=
  n % 2 = 1 ∧ n % 9 = 0

-- The goal: Prove that the reasoning process is completely correct
theorem correct_reasoning (H1 : multiple_of_9_is_multiple_of_3)
                          (n : ℤ)
                          (H2 : odd_multiple_of_9 n) : 
                          (n % 3 = 0) :=
by
  -- Explanation of the proof here
  sorry

end NUMINAMATH_GPT_correct_reasoning_l2177_217781


namespace NUMINAMATH_GPT_x_varies_inversely_l2177_217798

theorem x_varies_inversely (y: ℝ) (x: ℝ): (∃ k: ℝ, (∀ y: ℝ, x = k / y ^ 2) ∧ (1 = k / 3 ^ 2)) → x = 0.5625 :=
by
  sorry

end NUMINAMATH_GPT_x_varies_inversely_l2177_217798


namespace NUMINAMATH_GPT_sequence_solution_l2177_217709

theorem sequence_solution (a : ℕ → ℝ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, (2*n - 1) * a (n + 1) = (2*n + 1) * a n) : 
∀ n : ℕ, a n = 2 * n - 1 := 
by
  sorry

end NUMINAMATH_GPT_sequence_solution_l2177_217709


namespace NUMINAMATH_GPT_lisa_total_spoons_l2177_217768

def num_children := 4
def spoons_per_child := 3
def decorative_spoons := 2
def large_spoons := 10
def teaspoons := 15

def total_spoons := num_children * spoons_per_child + decorative_spoons + large_spoons + teaspoons

theorem lisa_total_spoons : total_spoons = 39 := by
  sorry

end NUMINAMATH_GPT_lisa_total_spoons_l2177_217768


namespace NUMINAMATH_GPT_nails_remaining_l2177_217719

theorem nails_remaining (nails_initial : ℕ) (kitchen_fraction : ℚ) (fence_fraction : ℚ) (nails_used_kitchen : ℕ) (nails_remaining_after_kitchen : ℕ) (nails_used_fence : ℕ) (nails_remaining_final : ℕ) 
  (h1 : nails_initial = 400) 
  (h2 : kitchen_fraction = 0.30) 
  (h3 : nails_used_kitchen = kitchen_fraction * nails_initial) 
  (h4 : nails_remaining_after_kitchen = nails_initial - nails_used_kitchen) 
  (h5 : fence_fraction = 0.70) 
  (h6 : nails_used_fence = fence_fraction * nails_remaining_after_kitchen) 
  (h7 : nails_remaining_final = nails_remaining_after_kitchen - nails_used_fence) :
  nails_remaining_final = 84 := by
sorry

end NUMINAMATH_GPT_nails_remaining_l2177_217719


namespace NUMINAMATH_GPT_condition_sufficient_not_necessary_l2177_217730

theorem condition_sufficient_not_necessary (x : ℝ) :
  (0 < x ∧ x < 2) → (x < 2) ∧ ¬((x < 2) → (0 < x ∧ x < 2)) :=
by
  sorry

end NUMINAMATH_GPT_condition_sufficient_not_necessary_l2177_217730


namespace NUMINAMATH_GPT_union_complement_eq_l2177_217780

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 5}

theorem union_complement_eq : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end NUMINAMATH_GPT_union_complement_eq_l2177_217780


namespace NUMINAMATH_GPT_ball_travel_distance_l2177_217729

noncomputable def total_distance : ℝ :=
  200 + (2 * (200 * (1 / 3))) + (2 * (200 * ((1 / 3) ^ 2))) +
  (2 * (200 * ((1 / 3) ^ 3))) + (2 * (200 * ((1 / 3) ^ 4)))

theorem ball_travel_distance :
  total_distance = 397.2 :=
by
  sorry

end NUMINAMATH_GPT_ball_travel_distance_l2177_217729


namespace NUMINAMATH_GPT_cheenu_time_difference_l2177_217754

def cheenu_bike_time_per_mile (distance_bike : ℕ) (time_bike : ℕ) : ℕ := time_bike / distance_bike
def cheenu_walk_time_per_mile (distance_walk : ℕ) (time_walk : ℕ) : ℕ := time_walk / distance_walk
def time_difference (time1 : ℕ) (time2 : ℕ) : ℕ := time2 - time1

theorem cheenu_time_difference 
  (distance_bike : ℕ) (time_bike : ℕ) 
  (distance_walk : ℕ) (time_walk : ℕ) 
  (H_bike : distance_bike = 20) (H_time_bike : time_bike = 80) 
  (H_walk : distance_walk = 8) (H_time_walk : time_walk = 160) :
  time_difference (cheenu_bike_time_per_mile distance_bike time_bike) (cheenu_walk_time_per_mile distance_walk time_walk) = 16 := 
by
  sorry

end NUMINAMATH_GPT_cheenu_time_difference_l2177_217754


namespace NUMINAMATH_GPT_number_of_four_digit_integers_with_digit_sum_nine_l2177_217756

theorem number_of_four_digit_integers_with_digit_sum_nine :
  ∃ (n : ℕ), (n = 165) ∧ (
    ∃ (a b c d : ℕ), 
      1 ≤ a ∧ 
      a + b + c + d = 9 ∧ 
      (1 ≤ a ∧ a ≤ 9) ∧ 
      (0 ≤ b ∧ b ≤ 9) ∧ 
      (0 ≤ c ∧ c ≤ 9) ∧ 
      (0 ≤ d ∧ d ≤ 9)) := 
sorry

end NUMINAMATH_GPT_number_of_four_digit_integers_with_digit_sum_nine_l2177_217756


namespace NUMINAMATH_GPT_probability_both_selected_l2177_217757

theorem probability_both_selected (P_C : ℚ) (P_B : ℚ) (hC : P_C = 4/5) (hB : P_B = 3/5) : 
  ((4/5) * (3/5)) = (12/25) := by
  sorry

end NUMINAMATH_GPT_probability_both_selected_l2177_217757


namespace NUMINAMATH_GPT_islander_parity_l2177_217728

-- Define the concept of knights and liars
def is_knight (x : ℕ) : Prop := x % 2 = 0 -- Knight count is even
def is_liar (x : ℕ) : Prop := ¬(x % 2 = 1) -- Liar count being odd is false, so even

-- Define the total inhabitants on the island and conditions
theorem islander_parity (K L : ℕ) (h₁ : is_knight K) (h₂ : is_liar L) (h₃ : K + L = 2021) : false := sorry

end NUMINAMATH_GPT_islander_parity_l2177_217728


namespace NUMINAMATH_GPT_projection_area_rectangular_board_l2177_217724

noncomputable def projection_area (AB BC NE MN : ℝ) (ABCD_perp_ground : Prop) (E_mid_AB : Prop) (light_at_M : Prop) : ℝ :=
  let width := AB
  let height := BC
  let shadow_width := 5
  (1 / 2) * (width + shadow_width) * height

theorem projection_area_rectangular_board (AB BC NE MN : ℝ) (ABCD_perp_ground : Prop) (E_mid_AB : Prop) (light_at_M : Prop) :
  AB = 3 → BC = 2 → NE = 3 → MN = 5 → projection_area AB BC NE MN ABCD_perp_ground E_mid_AB light_at_M = 8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_projection_area_rectangular_board_l2177_217724


namespace NUMINAMATH_GPT_acute_angle_of_parallelogram_l2177_217733

theorem acute_angle_of_parallelogram
  (a b : ℝ) (h : a < b)
  (parallelogram_division : ∀ x y : ℝ, x + y = a → b = x + 2 * Real.sqrt (x * y) + y) :
  ∃ α : ℝ, α = Real.arcsin ((b / a) - 1) :=
sorry

end NUMINAMATH_GPT_acute_angle_of_parallelogram_l2177_217733


namespace NUMINAMATH_GPT_geom_seq_value_l2177_217765

noncomputable def geom_sequence (a : ℕ → ℝ) (q : ℝ) :=
∀ (n : ℕ), a (n + 1) = a n * q

theorem geom_seq_value
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom_seq : geom_sequence a q)
  (h_a5 : a 5 = 2)
  (h_a6_a8 : a 6 * a 8 = 8) :
  (a 2018 - a 2016) / (a 2014 - a 2012) = 2 :=
sorry

end NUMINAMATH_GPT_geom_seq_value_l2177_217765


namespace NUMINAMATH_GPT_functional_eq_solution_l2177_217763

theorem functional_eq_solution (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (x + f (x + y)) + f (x * y) = x + f (x + y) + y * f x) →
  (∀ x : ℝ, f x = x) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_functional_eq_solution_l2177_217763


namespace NUMINAMATH_GPT_fraction_addition_l2177_217784

theorem fraction_addition : (2 / 5 + 3 / 8) = 31 / 40 :=
by
  sorry

end NUMINAMATH_GPT_fraction_addition_l2177_217784


namespace NUMINAMATH_GPT_max_square_side_length_l2177_217741

theorem max_square_side_length (AC BC : ℝ) (hAC : AC = 3) (hBC : BC = 7) : 
  ∃ s : ℝ, s = 2.1 := by
  sorry

end NUMINAMATH_GPT_max_square_side_length_l2177_217741


namespace NUMINAMATH_GPT_find_f_6_5_l2177_217744

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : is_even_function f
axiom periodic_f : ∀ x, f (x + 4) = f x
axiom f_in_interval : ∀ x, 1 ≤ x ∧ x ≤ 2 → f x = x - 2

theorem find_f_6_5 : f 6.5 = -0.5 := by
  sorry

end NUMINAMATH_GPT_find_f_6_5_l2177_217744


namespace NUMINAMATH_GPT_claire_gerbils_l2177_217732

theorem claire_gerbils (G H : ℕ) (h1 : G + H = 92) (h2 : (1/4 : ℚ) * G + (1/3 : ℚ) * H = 25) : G = 68 :=
sorry

end NUMINAMATH_GPT_claire_gerbils_l2177_217732


namespace NUMINAMATH_GPT_divisible_by_other_l2177_217769

theorem divisible_by_other (y : ℕ) 
  (h1 : y = 20)
  (h2 : y % 4 = 0)
  (h3 : y % 8 ≠ 0) : (∃ n, n ≠ 4 ∧ y % n = 0 ∧ n = 5) :=
by 
  sorry

end NUMINAMATH_GPT_divisible_by_other_l2177_217769


namespace NUMINAMATH_GPT_find_max_sum_of_squares_l2177_217795

open Real

theorem find_max_sum_of_squares 
  (a b c d : ℝ)
  (h1 : a + b = 17)
  (h2 : ab + c + d = 98)
  (h3 : ad + bc = 176)
  (h4 : cd = 105) :
  a^2 + b^2 + c^2 + d^2 ≤ 770 :=
sorry

end NUMINAMATH_GPT_find_max_sum_of_squares_l2177_217795


namespace NUMINAMATH_GPT_floor_difference_l2177_217725

theorem floor_difference (x : ℝ) (h : x = 15.3) : 
  (⌊ x^2 ⌋ - ⌊ x ⌋ * ⌊ x ⌋ + 5) = 14 := 
by
  -- Skipping proof
  sorry

end NUMINAMATH_GPT_floor_difference_l2177_217725


namespace NUMINAMATH_GPT_base_conversion_and_addition_l2177_217758

theorem base_conversion_and_addition :
  let n1 := 2 * (8:ℕ)^2 + 4 * 8^1 + 3 * 8^0
  let d1 := 1 * 4^1 + 3 * 4^0
  let n2 := 2 * 7^2 + 0 * 7^1 + 4 * 7^0
  let d2 := 2 * 5^1 + 3 * 5^0
  n1 / d1 + n2 / d2 = 31 + 51 / 91 := by
  sorry

end NUMINAMATH_GPT_base_conversion_and_addition_l2177_217758


namespace NUMINAMATH_GPT_abs_neg_2023_l2177_217710

theorem abs_neg_2023 : |(-2023)| = 2023 := by
  sorry

end NUMINAMATH_GPT_abs_neg_2023_l2177_217710


namespace NUMINAMATH_GPT_fraction_addition_l2177_217726

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end NUMINAMATH_GPT_fraction_addition_l2177_217726


namespace NUMINAMATH_GPT_find_functional_form_l2177_217723

theorem find_functional_form (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x + y) = 2 * x + f (f y - x)) →
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c :=
by
  sorry

end NUMINAMATH_GPT_find_functional_form_l2177_217723


namespace NUMINAMATH_GPT_diff_of_squares_expression_l2177_217773

theorem diff_of_squares_expression (m n : ℝ) :
  (3 * m + n) * (3 * m - n) = (3 * m)^2 - n^2 :=
by
  sorry

end NUMINAMATH_GPT_diff_of_squares_expression_l2177_217773


namespace NUMINAMATH_GPT_surface_area_is_correct_l2177_217714

structure CubicSolid where
  base_layer : ℕ
  second_layer : ℕ
  third_layer : ℕ
  top_layer : ℕ

def conditions : CubicSolid := ⟨4, 4, 3, 1⟩

theorem surface_area_is_correct : 
  (conditions.base_layer + conditions.second_layer + conditions.third_layer + conditions.top_layer + 7 + 7 + 3 + 3) = 28 := 
  by
  sorry

end NUMINAMATH_GPT_surface_area_is_correct_l2177_217714


namespace NUMINAMATH_GPT_solve_system_of_equations_l2177_217787

theorem solve_system_of_equations 
  (a b c s : ℝ) (x y z : ℝ)
  (h1 : y^2 - z * x = a * (x + y + z)^2)
  (h2 : x^2 - y * z = b * (x + y + z)^2)
  (h3 : z^2 - x * y = c * (x + y + z)^2)
  (h4 : a^2 + b^2 + c^2 - (a * b + b * c + c * a) = a + b + c) :
  (x = 0 ∧ y = 0 ∧ z = 0 ∧ x + y + z = 0) ∨
  ((x + y + z ≠ 0) ∧
   (x = (2 * c - a - b + 1) * s) ∧
   (y = (2 * a - b - c + 1) * s) ∧
   (z = (2 * b - c - a + 1) * s)) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l2177_217787


namespace NUMINAMATH_GPT_determine_g_l2177_217701

theorem determine_g (g : ℝ → ℝ) : (∀ x : ℝ, 4 * x^4 + x^3 - 2 * x + 5 + g x = 2 * x^3 - 7 * x^2 + 4) →
  (∀ x : ℝ, g x = -4 * x^4 + x^3 - 7 * x^2 + 2 * x - 1) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_determine_g_l2177_217701


namespace NUMINAMATH_GPT_depth_notation_l2177_217711

-- Definition of depth and altitude
def above_sea_level (h : ℝ) : Prop := h > 0 
def below_sea_level (h : ℝ) : Prop := h < 0 

-- Given conditions
axiom height_Dabaijing : above_sea_level 9050
axiom depth_Haidou1 : below_sea_level (-10907)

-- Proof goal
theorem depth_notation :
  ∀ (d : ℝ), above_sea_level 9050 → below_sea_level (-d) → d = 10907 :=
by
  intros d _ _
  exact sorry

end NUMINAMATH_GPT_depth_notation_l2177_217711


namespace NUMINAMATH_GPT_price_of_shoes_on_tuesday_is_correct_l2177_217745

theorem price_of_shoes_on_tuesday_is_correct :
  let price_thursday : ℝ := 30
  let price_friday : ℝ := price_thursday * 1.2
  let price_monday : ℝ := price_friday - price_friday * 0.15
  let price_tuesday : ℝ := price_monday - price_monday * 0.1
  price_tuesday = 27.54 := 
by
  sorry

end NUMINAMATH_GPT_price_of_shoes_on_tuesday_is_correct_l2177_217745


namespace NUMINAMATH_GPT_train_distance_l2177_217742

theorem train_distance (t : ℕ) (d : ℕ) (rate : d / t = 1 / 2) (total_time : ℕ) (h : total_time = 90) : ∃ distance : ℕ, distance = 45 := by
  sorry

end NUMINAMATH_GPT_train_distance_l2177_217742


namespace NUMINAMATH_GPT_find_m_collinear_l2177_217722

-- Definition of a point in 2D space
structure Point2D where
  x : ℤ
  y : ℤ

-- Predicate to check if three points are collinear 
def collinear_points (p1 p2 p3 : Point2D) : Prop :=
  (p3.x - p2.x) * (p2.y - p1.y) = (p2.x - p1.x) * (p3.y - p2.y)

-- Given points A, B, and C
def A : Point2D := ⟨2, 3⟩
def B (m : ℤ) : Point2D := ⟨-4, m⟩
def C : Point2D := ⟨-12, -1⟩

-- Theorem stating the value of m such that points A, B, and C are collinear
theorem find_m_collinear : ∃ (m : ℤ), collinear_points A (B m) C ∧ m = 9 / 7 := sorry

end NUMINAMATH_GPT_find_m_collinear_l2177_217722


namespace NUMINAMATH_GPT_sheep_drowned_proof_l2177_217775

def animal_problem_statement (S : ℕ) : Prop :=
  let initial_sheep := 20
  let initial_cows := 10
  let initial_dogs := 14
  let total_animals_made_shore := 35
  let sheep_drowned := S
  let cows_drowned := 2 * S
  let dogs_survived := initial_dogs
  let animals_made_shore := initial_sheep + initial_cows + initial_dogs - (sheep_drowned + cows_drowned)
  30 - 3 * S = 35 - 14

theorem sheep_drowned_proof : ∃ S : ℕ, animal_problem_statement S ∧ S = 3 :=
by
  sorry

end NUMINAMATH_GPT_sheep_drowned_proof_l2177_217775


namespace NUMINAMATH_GPT_expression_divisible_by_17_l2177_217736

theorem expression_divisible_by_17 (n : ℕ) : 
  (6^(2*n) + 2^(n+2) + 12 * 2^n) % 17 = 0 :=
by
  sorry

end NUMINAMATH_GPT_expression_divisible_by_17_l2177_217736


namespace NUMINAMATH_GPT_length_PT_30_l2177_217704

noncomputable def length_PT (PQ QR : ℝ) (angle_QRT : ℝ) (T_on_RS : Prop) : ℝ := 
  if h : PQ = 30 ∧ QR = 15 ∧ angle_QRT = 75 then 30 else 0

theorem length_PT_30 (PQ QR : ℝ) (angle_QRT : ℝ) (T_on_RS : Prop) :
  PQ = 30 → QR = 15 → angle_QRT = 75 → length_PT PQ QR angle_QRT T_on_RS = 30 :=
sorry

end NUMINAMATH_GPT_length_PT_30_l2177_217704


namespace NUMINAMATH_GPT_bumper_cars_initial_count_l2177_217779

variable {X : ℕ}

theorem bumper_cars_initial_count (h : (X - 6) + 3 = 6) : X = 9 := 
by
  sorry

end NUMINAMATH_GPT_bumper_cars_initial_count_l2177_217779


namespace NUMINAMATH_GPT_marites_saves_120_per_year_l2177_217774

def current_internet_speed := 10 -- Mbps
def current_monthly_bill := 20 -- dollars

def monthly_cost_20mbps := current_monthly_bill + 10 -- dollars
def monthly_cost_30mbps := current_monthly_bill * 2 -- dollars

def bundled_cost_20mbps := 80 -- dollars per month
def bundled_cost_30mbps := 90 -- dollars per month

def annual_cost_20mbps := bundled_cost_20mbps * 12 -- dollars per year
def annual_cost_30mbps := bundled_cost_30mbps * 12 -- dollars per year

theorem marites_saves_120_per_year :
  annual_cost_30mbps - annual_cost_20mbps = 120 := 
by
  sorry

end NUMINAMATH_GPT_marites_saves_120_per_year_l2177_217774


namespace NUMINAMATH_GPT_library_science_books_count_l2177_217749

-- Definitions based on the problem conditions
def initial_science_books := 120
def borrowed_books := 40
def returned_books := 15
def books_on_hold := 10
def borrowed_from_other_library := 20
def lost_books := 2
def damaged_books := 1

-- Statement for the proof.
theorem library_science_books_count :
  initial_science_books - borrowed_books + returned_books - books_on_hold + borrowed_from_other_library - lost_books - damaged_books = 102 :=
by
  sorry

end NUMINAMATH_GPT_library_science_books_count_l2177_217749


namespace NUMINAMATH_GPT_equal_number_of_digits_l2177_217718

noncomputable def probability_equal_digits : ℚ := (20 * (9/16)^3 * (7/16)^3)

theorem equal_number_of_digits :
  probability_equal_digits = 3115125 / 10485760 := by
  sorry

end NUMINAMATH_GPT_equal_number_of_digits_l2177_217718


namespace NUMINAMATH_GPT_correct_product_of_0_035_and_3_84_l2177_217738

theorem correct_product_of_0_035_and_3_84 : 
  (0.035 * 3.84 = 0.1344) := sorry

end NUMINAMATH_GPT_correct_product_of_0_035_and_3_84_l2177_217738


namespace NUMINAMATH_GPT_mass_percentage_of_H_in_ascorbic_acid_l2177_217764

-- Definitions based on the problem conditions
def molar_mass_C : ℝ := 12.01
def molar_mass_H : ℝ := 1.01
def molar_mass_O : ℝ := 16.00

def ascorbic_acid_molecular_formula_C : ℝ := 6
def ascorbic_acid_molecular_formula_H : ℝ := 8
def ascorbic_acid_molecular_formula_O : ℝ := 6

noncomputable def ascorbic_acid_molar_mass : ℝ :=
  ascorbic_acid_molecular_formula_C * molar_mass_C + 
  ascorbic_acid_molecular_formula_H * molar_mass_H + 
  ascorbic_acid_molecular_formula_O * molar_mass_O

noncomputable def hydrogen_mass_in_ascorbic_acid : ℝ :=
  ascorbic_acid_molecular_formula_H * molar_mass_H

noncomputable def hydrogen_mass_percentage_in_ascorbic_acid : ℝ :=
  (hydrogen_mass_in_ascorbic_acid / ascorbic_acid_molar_mass) * 100

theorem mass_percentage_of_H_in_ascorbic_acid :
  hydrogen_mass_percentage_in_ascorbic_acid = 4.588 :=
by
  sorry

end NUMINAMATH_GPT_mass_percentage_of_H_in_ascorbic_acid_l2177_217764


namespace NUMINAMATH_GPT_reciprocals_and_opposites_l2177_217748

theorem reciprocals_and_opposites (a b c d : ℝ) (h_ab : a * b = 1) (h_cd : c + d = 0) : 
  (c + d)^2 - a * b = -1 := by
  sorry

end NUMINAMATH_GPT_reciprocals_and_opposites_l2177_217748


namespace NUMINAMATH_GPT_danny_initial_caps_l2177_217752

-- Define the conditions
variables (lostCaps : ℕ) (currentCaps : ℕ)
-- Assume given conditions
axiom lost_caps_condition : lostCaps = 66
axiom current_caps_condition : currentCaps = 25

-- Define the total number of bottle caps Danny had at first
def originalCaps (lostCaps currentCaps : ℕ) : ℕ := lostCaps + currentCaps

-- State the theorem to prove the number of bottle caps Danny originally had is 91
theorem danny_initial_caps : originalCaps lostCaps currentCaps = 91 :=
by
  -- Insert the proof here when available
  sorry

end NUMINAMATH_GPT_danny_initial_caps_l2177_217752


namespace NUMINAMATH_GPT_thabo_books_l2177_217700

variable (P F H : Nat)

theorem thabo_books :
  P > 55 ∧ F = 2 * P ∧ H = 55 ∧ H + P + F = 280 → P - H = 20 :=
by
  sorry

end NUMINAMATH_GPT_thabo_books_l2177_217700


namespace NUMINAMATH_GPT_number_of_solutions_l2177_217786

theorem number_of_solutions : ∃ (s : Finset ℕ), (∀ x ∈ s, 100 ≤ x^2 ∧ x^2 ≤ 200) ∧ s.card = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_of_solutions_l2177_217786


namespace NUMINAMATH_GPT_option_c_correct_l2177_217746

theorem option_c_correct (x y : ℝ) (h : x < y) : -x > -y := 
sorry

end NUMINAMATH_GPT_option_c_correct_l2177_217746


namespace NUMINAMATH_GPT_sum_of_ages_l2177_217713

theorem sum_of_ages (age1 age2 age3 : ℕ) (h : age1 * age2 * age3 = 128) : age1 + age2 + age3 = 18 :=
sorry

end NUMINAMATH_GPT_sum_of_ages_l2177_217713


namespace NUMINAMATH_GPT_f_leq_zero_l2177_217755

noncomputable def f (x a : ℝ) := x * Real.log x - a * x^2 + (2 * a - 1) * x

theorem f_leq_zero (a x : ℝ) (h1 : 1/2 < a) (h2 : a ≤ 1) (hx : 0 < x) :
  f x a ≤ 0 :=
sorry

end NUMINAMATH_GPT_f_leq_zero_l2177_217755


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l2177_217794

theorem quadratic_inequality_solution (x : ℝ) : x^2 + 4 * x > 45 ↔ x < -9 ∨ x > 5 := 
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l2177_217794


namespace NUMINAMATH_GPT_workshop_cost_l2177_217715

theorem workshop_cost
  (x : ℝ)
  (h1 : 0 < x) -- Given the cost must be positive
  (h2 : (x / 4) - 15 = x / 7) :
  x = 140 :=
by
  sorry

end NUMINAMATH_GPT_workshop_cost_l2177_217715


namespace NUMINAMATH_GPT_bob_after_alice_l2177_217789

def race_distance : ℕ := 15
def alice_speed : ℕ := 7
def bob_speed : ℕ := 9

def alice_time : ℕ := alice_speed * race_distance
def bob_time : ℕ := bob_speed * race_distance

theorem bob_after_alice : bob_time - alice_time = 30 := by
  sorry

end NUMINAMATH_GPT_bob_after_alice_l2177_217789


namespace NUMINAMATH_GPT_apple_counts_l2177_217782

theorem apple_counts (x y : ℤ) (h1 : y - x = 2) (h2 : y = 3 * x - 4) : x = 3 ∧ y = 5 := 
by
  sorry

end NUMINAMATH_GPT_apple_counts_l2177_217782


namespace NUMINAMATH_GPT_exceeded_by_600_l2177_217740

noncomputable def ken_collected : ℕ := 600
noncomputable def mary_collected (ken : ℕ) : ℕ := 5 * ken
noncomputable def scott_collected (mary : ℕ) : ℕ := mary / 3
noncomputable def total_collected (ken mary scott : ℕ) : ℕ := ken + mary + scott
noncomputable def goal : ℕ := 4000
noncomputable def exceeded_goal (total goal : ℕ) : ℕ := total - goal

theorem exceeded_by_600 : exceeded_goal (total_collected ken_collected (mary_collected ken_collected) (scott_collected (mary_collected ken_collected))) goal = 600 := by
  sorry

end NUMINAMATH_GPT_exceeded_by_600_l2177_217740


namespace NUMINAMATH_GPT_complementary_angles_of_same_angle_are_equal_l2177_217743

def complementary_angles (α β : ℝ) := α + β = 90 

theorem complementary_angles_of_same_angle_are_equal 
        (θ : ℝ) (α β : ℝ) 
        (h1 : complementary_angles θ α) 
        (h2 : complementary_angles θ β) : 
        α = β := 
by 
  sorry

end NUMINAMATH_GPT_complementary_angles_of_same_angle_are_equal_l2177_217743


namespace NUMINAMATH_GPT_number_of_moles_of_OC_NH2_2_formed_l2177_217770

-- Definition: Chemical reaction condition
def reaction_eqn (x y : ℕ) : Prop := 
  x ≥ 1 ∧ y ≥ 2 ∧ x * 2 = y

-- Theorem: Prove that combining 3 moles of CO2 and 6 moles of NH3 results in 3 moles of OC(NH2)2
theorem number_of_moles_of_OC_NH2_2_formed (x y : ℕ) 
(h₁ : reaction_eqn x y)
(h₂ : x = 3)
(h₃ : y = 6) : 
x =  y / 2 :=
by {
    -- Proof is not provided
    sorry 
}

end NUMINAMATH_GPT_number_of_moles_of_OC_NH2_2_formed_l2177_217770


namespace NUMINAMATH_GPT_ninth_term_arithmetic_sequence_l2177_217777

def first_term : ℚ := 3 / 4
def seventeenth_term : ℚ := 6 / 7

theorem ninth_term_arithmetic_sequence :
  let a1 := first_term
  let a17 := seventeenth_term
  (a1 + a17) / 2 = 45 / 56 := 
sorry

end NUMINAMATH_GPT_ninth_term_arithmetic_sequence_l2177_217777


namespace NUMINAMATH_GPT_unplanted_fraction_l2177_217796

theorem unplanted_fraction (a b hypotenuse : ℕ) (side_length_P : ℚ) 
                          (h1 : a = 5) (h2 : b = 12) (h3 : hypotenuse = 13)
                          (h4 : side_length_P = 5 / 3) : 
                          (side_length_P * side_length_P) / ((a * b) / 2) = 5 / 54 :=
by
  sorry

end NUMINAMATH_GPT_unplanted_fraction_l2177_217796


namespace NUMINAMATH_GPT_toothpick_count_l2177_217712

theorem toothpick_count (height width : ℕ) (h_height : height = 20) (h_width : width = 10) : 
  (21 * width + 11 * height) = 430 :=
by
  sorry

end NUMINAMATH_GPT_toothpick_count_l2177_217712


namespace NUMINAMATH_GPT_find_c_plus_one_div_b_l2177_217706

-- Assume that a, b, and c are positive real numbers such that the given conditions hold.
variables (a b c : ℝ)
variables (habc : a * b * c = 1)
variables (hac : a + 1 / c = 7)
variables (hba : b + 1 / a = 11)

-- The goal is to show that c + 1 / b = 5 / 19.
theorem find_c_plus_one_div_b : c + 1 / b = 5 / 19 :=
by 
  sorry

end NUMINAMATH_GPT_find_c_plus_one_div_b_l2177_217706


namespace NUMINAMATH_GPT_absolute_value_solution_l2177_217797

theorem absolute_value_solution (m : ℤ) (h : abs m = abs (-7)) : m = 7 ∨ m = -7 := by
  sorry

end NUMINAMATH_GPT_absolute_value_solution_l2177_217797


namespace NUMINAMATH_GPT_similar_polygon_area_sum_l2177_217737

theorem similar_polygon_area_sum 
  (t1 t2 a1 a2 b : ℝ)
  (h_ratio: t1 / t2 = a1^2 / a2^2)
  (t3 : ℝ := t1 + t2)
  (h_area_eq : t3 = b^2 * a1^2 / a2^2): 
  b = Real.sqrt (a1^2 + a2^2) :=
by
  sorry

end NUMINAMATH_GPT_similar_polygon_area_sum_l2177_217737


namespace NUMINAMATH_GPT_point_B_coordinates_l2177_217759

def move_up (x y : Int) (units : Int) : Int := y + units
def move_left (x y : Int) (units : Int) : Int := x - units

theorem point_B_coordinates :
  let A : Int × Int := (1, -1)
  let B : Int × Int := (move_left A.1 A.2 3, move_up A.1 A.2 2)
  B = (-2, 1) := 
by
  -- This is where the proof would go, but we omit it with "sorry"
  sorry

end NUMINAMATH_GPT_point_B_coordinates_l2177_217759


namespace NUMINAMATH_GPT_mike_max_marks_l2177_217717

theorem mike_max_marks
  (M : ℝ)
  (h1 : 0.30 * M = 234)
  (h2 : 234 = 212 + 22) : M = 780 := 
sorry

end NUMINAMATH_GPT_mike_max_marks_l2177_217717


namespace NUMINAMATH_GPT_infinite_geometric_series_sum_l2177_217720

theorem infinite_geometric_series_sum (a : ℕ → ℝ) (a1 : a 1 = 1) (r : ℝ) (h : r = 1 / 3) (S : ℝ) (H : S = a 1 / (1 - r)) : S = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_infinite_geometric_series_sum_l2177_217720


namespace NUMINAMATH_GPT_radian_measure_of_sector_l2177_217776

theorem radian_measure_of_sector
  (perimeter : ℝ) (area : ℝ) (radian_measure : ℝ)
  (h1 : perimeter = 8)
  (h2 : area = 4) :
  radian_measure = 2 :=
sorry

end NUMINAMATH_GPT_radian_measure_of_sector_l2177_217776


namespace NUMINAMATH_GPT_area_of_L_equals_22_l2177_217761

-- Define the dimensions of the rectangles
def big_rectangle_length := 8
def big_rectangle_width := 5
def small_rectangle_length := big_rectangle_length - 2
def small_rectangle_width := big_rectangle_width - 2

-- Define the areas
def area_big_rectangle := big_rectangle_length * big_rectangle_width
def area_small_rectangle := small_rectangle_length * small_rectangle_width

-- Define the area of the "L" shape
def area_L := area_big_rectangle - area_small_rectangle

-- State the theorem
theorem area_of_L_equals_22 : area_L = 22 := by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_area_of_L_equals_22_l2177_217761


namespace NUMINAMATH_GPT_find_J_salary_l2177_217734

variable (J F M A : ℝ)

theorem find_J_salary (h1 : (J + F + M + A) / 4 = 8000) (h2 : (F + M + A + 6500) / 4 = 8900) :
  J = 2900 := by
  sorry

end NUMINAMATH_GPT_find_J_salary_l2177_217734


namespace NUMINAMATH_GPT_log_function_domain_l2177_217778

theorem log_function_domain (x : ℝ) : 
  (3 - x > 0) ∧ (x - 1 > 0) ∧ (x - 1 ≠ 1) -> (1 < x ∧ x < 3 ∧ x ≠ 2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_log_function_domain_l2177_217778


namespace NUMINAMATH_GPT_factor_tree_X_value_l2177_217792

theorem factor_tree_X_value :
  let F := 2 * 5
  let G := 7 * 3
  let Y := 7 * F
  let Z := 11 * G
  let X := Y * Z
  X = 16170 := by
sorry

end NUMINAMATH_GPT_factor_tree_X_value_l2177_217792


namespace NUMINAMATH_GPT_minimum_product_value_l2177_217702

-- Problem conditions
def total_stones : ℕ := 40
def b_min : ℕ := 20
def b_max : ℕ := 32

-- Define the product function
def P (b : ℕ) : ℕ := b * (total_stones - b)

-- Goal: Prove the minimum value of P(b) for b in [20, 32] is 256
theorem minimum_product_value : ∃ (b : ℕ), b_min ≤ b ∧ b ≤ b_max ∧ P b = 256 := by
  sorry

end NUMINAMATH_GPT_minimum_product_value_l2177_217702


namespace NUMINAMATH_GPT_number_of_positive_integer_pairs_l2177_217753

theorem number_of_positive_integer_pairs (x y : ℕ) (h : 20 * x + 6 * y = 2006) : 
  ∃ n, n = 34 ∧ ∀ (x y : ℕ), 20 * x + 6 * y = 2006 → 0 < x → 0 < y → 
  (∃ k, x = 3 * k + 1 ∧ y = 331 - 10 * k ∧ 0 ≤ k ∧ k ≤ 33) :=
sorry

end NUMINAMATH_GPT_number_of_positive_integer_pairs_l2177_217753


namespace NUMINAMATH_GPT_sum_of_roots_l2177_217705

theorem sum_of_roots : (x₁ x₂ : ℝ) → (h : 2 * x₁^2 + 6 * x₁ - 1 = 0) → (h₂ : 2 * x₂^2 + 6 * x₂ - 1 = 0) → x₁ + x₂ = -3 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_roots_l2177_217705


namespace NUMINAMATH_GPT_total_distance_between_first_and_fifth_poles_l2177_217721

noncomputable def distance_between_poles (n : ℕ) (d : ℕ) : ℕ :=
  d / n

theorem total_distance_between_first_and_fifth_poles :
  ∀ (n : ℕ) (d : ℕ), (n = 3 ∧ d = 90) → (4 * distance_between_poles n d = 120) :=
by
  sorry

end NUMINAMATH_GPT_total_distance_between_first_and_fifth_poles_l2177_217721


namespace NUMINAMATH_GPT_gcd_of_168_56_224_l2177_217766

theorem gcd_of_168_56_224 : (Nat.gcd 168 56 = 56) ∧ (Nat.gcd 56 224 = 56) ∧ (Nat.gcd 168 224 = 56) :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_168_56_224_l2177_217766


namespace NUMINAMATH_GPT_evaluate_polynomial_at_3_l2177_217727

noncomputable def f (x : ℝ) : ℝ := 0.5 * x^5 + 4 * x^4 - 3 * x^2 + x - 1

theorem evaluate_polynomial_at_3 : f 3 = 1 := by
  sorry

end NUMINAMATH_GPT_evaluate_polynomial_at_3_l2177_217727


namespace NUMINAMATH_GPT_infinite_divisibility_of_2n_plus_n2_by_100_l2177_217762

theorem infinite_divisibility_of_2n_plus_n2_by_100 :
  ∃ᶠ n in at_top, 100 ∣ (2^n + n^2) :=
sorry

end NUMINAMATH_GPT_infinite_divisibility_of_2n_plus_n2_by_100_l2177_217762


namespace NUMINAMATH_GPT_ab_plus_cd_eq_12_l2177_217707

theorem ab_plus_cd_eq_12 (a b c d : ℝ) 
  (h1 : a + b + c = 3) 
  (h2 : a + b + d = -1) 
  (h3 : a + c + d = 8) 
  (h4 : b + c + d = 5) :
  a * b + c * d = 12 := by
  sorry

end NUMINAMATH_GPT_ab_plus_cd_eq_12_l2177_217707


namespace NUMINAMATH_GPT_inequality_proof_l2177_217739

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
    (x^3 / (x^3 + 2 * y^2 * Real.sqrt (z * x))) + 
    (y^3 / (y^3 + 2 * z^2 * Real.sqrt (x * y))) + 
    (z^3 / (z^3 + 2 * x^2 * Real.sqrt (y * z))) ≥ 1 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l2177_217739


namespace NUMINAMATH_GPT_cuboid_surface_area_4_8_6_l2177_217790

noncomputable def cuboid_surface_area (length width height : ℕ) : ℕ :=
  2 * (length * width + length * height + width * height)

theorem cuboid_surface_area_4_8_6 : cuboid_surface_area 4 8 6 = 208 := by
  sorry

end NUMINAMATH_GPT_cuboid_surface_area_4_8_6_l2177_217790


namespace NUMINAMATH_GPT_exam_students_l2177_217735

noncomputable def totalStudents (N : ℕ) (T : ℕ) := T = 70 * N
noncomputable def marksOfExcludedStudents := 5 * 50
noncomputable def remainingStudents (N : ℕ) := N - 5
noncomputable def remainingMarksCondition (N T : ℕ) := (T - marksOfExcludedStudents) / remainingStudents N = 90

theorem exam_students (N : ℕ) (T : ℕ) 
  (h1 : totalStudents N T) 
  (h2 : remainingMarksCondition N T) : 
  N = 10 :=
by 
  sorry

end NUMINAMATH_GPT_exam_students_l2177_217735


namespace NUMINAMATH_GPT_problem_solution_sets_l2177_217767

theorem problem_solution_sets (x y : ℝ) :
  (x^2 * y + y^3 = 2 * x^2 + 2 * y^2 ∧ x * y + 1 = x + y) →
  ( (x = 0 ∧ y = 0) ∨ y = 2 ∨ x = 1 ∨ y = 1 ) :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_sets_l2177_217767


namespace NUMINAMATH_GPT_greatest_difference_four_digit_numbers_l2177_217750

theorem greatest_difference_four_digit_numbers : 
  ∃ (d1 d2 d3 d4 : ℕ), (d1 = 0 ∨ d1 = 3 ∨ d1 = 4 ∨ d1 = 8) ∧ 
                      (d2 = 0 ∨ d2 = 3 ∨ d2 = 4 ∨ d2 = 8) ∧ 
                      (d3 = 0 ∨ d3 = 3 ∨ d3 = 4 ∨ d3 = 8) ∧ 
                      (d4 = 0 ∨ d4 = 3 ∨ d4 = 4 ∨ d4 = 8) ∧ 
                      d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ 
                      d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4 ∧ 
                      (∃ n1 n2, n1 = 1000 * 8 + 100 * 4 + 10 * 3 + 0 ∧ 
                                n2 = 1000 * 3 + 100 * 0 + 10 * 4 + 8 ∧ 
                                n1 - n2 = 5382) :=
by {
  sorry
}

end NUMINAMATH_GPT_greatest_difference_four_digit_numbers_l2177_217750


namespace NUMINAMATH_GPT_quadratic_polynomial_solution_l2177_217716

theorem quadratic_polynomial_solution :
  ∃ a b c : ℚ, 
    (∀ x : ℚ, ax*x + bx + c = 8 ↔ x = -2) ∧ 
    (∀ x : ℚ, ax*x + bx + c = 2 ↔ x = 1) ∧ 
    (∀ x : ℚ, ax*x + bx + c = 10 ↔ x = 3) ∧ 
    a = 6 / 5 ∧ 
    b = -4 / 5 ∧ 
    c = 8 / 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_polynomial_solution_l2177_217716


namespace NUMINAMATH_GPT_union_of_A_and_B_l2177_217760

def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {1, 2}

theorem union_of_A_and_B : A ∪ B = {x | x ≤ 2} := sorry

end NUMINAMATH_GPT_union_of_A_and_B_l2177_217760


namespace NUMINAMATH_GPT_special_case_m_l2177_217771

theorem special_case_m (m : ℝ) :
  (∀ x : ℝ, mx^2 - 4 * x + 3 = 0 → y = mx^2 - 4 * x + 3 → (x = 0 ∧ m = 0) ∨ (x ≠ 0 ∧ m = 4/3)) :=
sorry

end NUMINAMATH_GPT_special_case_m_l2177_217771


namespace NUMINAMATH_GPT_lollipop_distribution_l2177_217731

theorem lollipop_distribution 
  (P1 P2 P_total L x : ℕ) 
  (h1 : P1 = 45) 
  (h2 : P2 = 15) 
  (h3 : L = 12) 
  (h4 : P_total = P1 + P2) 
  (h5 : P_total = 60) : 
  x = 5 := 
by 
  sorry

end NUMINAMATH_GPT_lollipop_distribution_l2177_217731


namespace NUMINAMATH_GPT_c_left_days_before_completion_l2177_217799

-- Definitions for the given conditions
def work_done_by_a_in_one_day := 1 / 30
def work_done_by_b_in_one_day := 1 / 30
def work_done_by_c_in_one_day := 1 / 40
def total_days := 12

-- Proof problem statement (to prove that c left 8 days before the completion)
theorem c_left_days_before_completion :
  ∃ x : ℝ, 
  (12 - x) * (7 / 60) + x * (1 / 15) = 1 → 
  x = 8 := sorry

end NUMINAMATH_GPT_c_left_days_before_completion_l2177_217799


namespace NUMINAMATH_GPT_evaluate_f_g_3_l2177_217747

def g (x : ℝ) := x^3
def f (x : ℝ) := 3 * x - 2

theorem evaluate_f_g_3 : f (g 3) = 79 := by
  sorry

end NUMINAMATH_GPT_evaluate_f_g_3_l2177_217747


namespace NUMINAMATH_GPT_train_length_is_correct_l2177_217708

noncomputable def length_of_train (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  speed_ms * time_s

theorem train_length_is_correct :
  length_of_train 60 12 = 200.04 :=
by 
  -- Here, a proof would be provided, eventually using the definitions and conditions given
  sorry

end NUMINAMATH_GPT_train_length_is_correct_l2177_217708


namespace NUMINAMATH_GPT_stickers_on_first_day_l2177_217791

theorem stickers_on_first_day (s e total : ℕ) (h1 : e = 22) (h2 : total = 61) (h3 : total = s + e) : s = 39 :=
by
  sorry

end NUMINAMATH_GPT_stickers_on_first_day_l2177_217791


namespace NUMINAMATH_GPT_cube_root_neg_eighth_l2177_217785

theorem cube_root_neg_eighth : ∃ x : ℚ, x^3 = -1 / 8 ∧ x = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cube_root_neg_eighth_l2177_217785


namespace NUMINAMATH_GPT_pool_capacity_l2177_217703

variables {T : ℕ} {A B C : ℕ → ℕ}

-- Conditions
def valve_rate_A (T : ℕ) : ℕ := T / 180
def valve_rate_B (T : ℕ) := valve_rate_A T + 60
def valve_rate_C (T : ℕ) := valve_rate_A T + 75

def combined_rate (T : ℕ) := valve_rate_A T + valve_rate_B T + valve_rate_C T

-- Theorem to prove
theorem pool_capacity (T : ℕ) (h1 : combined_rate T = T / 40) : T = 16200 :=
by
  sorry

end NUMINAMATH_GPT_pool_capacity_l2177_217703


namespace NUMINAMATH_GPT_karen_piggy_bank_total_l2177_217751

theorem karen_piggy_bank_total (a r n : ℕ) (h1 : a = 2) (h2 : r = 3) (h3 : n = 7) :
  (a * ((1 - r^n) / (1 - r))) = 2186 := by
  sorry

end NUMINAMATH_GPT_karen_piggy_bank_total_l2177_217751


namespace NUMINAMATH_GPT_interest_rate_proof_l2177_217772

noncomputable def compound_interest_rate (P A : ℝ) (n : ℕ) (r : ℝ) : Prop :=
  A = P * (1 + r)^n

noncomputable def interest_rate (initial  final: ℝ) (years : ℕ) : ℝ := 
  (4: ℝ)^(1/(years: ℝ)) - 1

theorem interest_rate_proof :
  compound_interest_rate 8000 32000 36 (interest_rate 8000 32000 36) ∧
  abs (interest_rate 8000 32000 36 * 100 - 3.63) < 0.01 :=
by
  -- Conditions from the problem for compound interest
  -- Using the formula for interest rate and the condition checks
  sorry

end NUMINAMATH_GPT_interest_rate_proof_l2177_217772


namespace NUMINAMATH_GPT_cyclist_A_speed_l2177_217783

theorem cyclist_A_speed (a b : ℝ) (h1 : b = a + 5)
    (h2 : 80 / a = 120 / b) : a = 10 :=
by
  sorry

end NUMINAMATH_GPT_cyclist_A_speed_l2177_217783


namespace NUMINAMATH_GPT_incorrect_population_growth_statement_l2177_217788

def population_growth_behavior (p: ℝ → ℝ) : Prop :=
(p 0 < p 1) ∧ (∃ t₁ t₂, t₁ < t₂ ∧ (∀ t < t₁, p t < p (t + 1)) ∧
 (∀ t > t₁, (p t < p (t - 1)) ∨ (p t = p (t - 1))))

def stabilizes_at_K (p: ℝ → ℝ) (K: ℝ) : Prop :=
∃ t₀, ∀ t > t₀, p t = K

def K_value_definition (K: ℝ) (environmental_conditions: ℝ → ℝ) : Prop :=
∀ t, environmental_conditions t = K

theorem incorrect_population_growth_statement (p: ℝ → ℝ) (K: ℝ) (environmental_conditions: ℝ → ℝ)
(h1: population_growth_behavior p)
(h2: stabilizes_at_K p K)
(h3: K_value_definition K environmental_conditions) :
(p 0 > p 1) ∨ (¬ (∃ t₁ t₂, t₁ < t₂ ∧ (∀ t < t₁, p t < p (t + 1)) ∧
 (∀ t > t₁, (p t < p (t - 1)) ∨ (p t = p (t - 1))))) :=
sorry

end NUMINAMATH_GPT_incorrect_population_growth_statement_l2177_217788
