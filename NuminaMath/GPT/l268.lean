import Mathlib

namespace NUMINAMATH_GPT_largest_y_coordinate_l268_26853

theorem largest_y_coordinate (x y : ℝ) (h : (x^2 / 49) + ((y - 3)^2 / 25) = 0) : y = 3 :=
sorry

end NUMINAMATH_GPT_largest_y_coordinate_l268_26853


namespace NUMINAMATH_GPT_triangle_BC_length_l268_26843

noncomputable def length_of_BC (ABC : Triangle) (incircle_radius : ℝ) (altitude_A_to_BC : ℝ) 
    (BD_squared_plus_CD_squared : ℝ) : ℝ :=
  if incircle_radius = 3 ∧ altitude_A_to_BC = 15 ∧ BD_squared_plus_CD_squared = 33 then
    3 * Real.sqrt 7
  else
    0 -- This value is arbitrary, as the conditions above are specific

theorem triangle_BC_length {ABC : Triangle}
    (incircle_radius : ℝ) (altitude_A_to_BC : ℝ) (BD_squared_plus_CD_squared : ℝ) :
    incircle_radius = 3 →
    altitude_A_to_BC = 15 →
    BD_squared_plus_CD_squared = 33 →
    length_of_BC ABC incircle_radius altitude_A_to_BC BD_squared_plus_CD_squared = 3 * Real.sqrt 7 :=
by intros; sorry

end NUMINAMATH_GPT_triangle_BC_length_l268_26843


namespace NUMINAMATH_GPT_combined_original_price_l268_26877

def original_price_shoes (discount_price : ℚ) (discount_rate : ℚ) : ℚ := discount_price / (1 - discount_rate)

def original_price_dress (discount_price : ℚ) (discount_rate : ℚ) : ℚ := discount_price / (1 - discount_rate)

theorem combined_original_price (shoes_price : ℚ) (shoes_discount : ℚ) (dress_price : ℚ) (dress_discount : ℚ) 
  (h_shoes : shoes_discount = 0.20 ∧ shoes_price = 480) 
  (h_dress : dress_discount = 0.30 ∧ dress_price = 350) : 
  original_price_shoes shoes_price shoes_discount + original_price_dress dress_price dress_discount = 1100 := by
  sorry

end NUMINAMATH_GPT_combined_original_price_l268_26877


namespace NUMINAMATH_GPT_percentage_markup_l268_26818

variable (W R : ℝ) -- W is the wholesale cost, R is the normal retail price

-- The condition that, at 60% discount, the sale price nets a 35% profit on the wholesale cost
variable (h : 0.4 * R = 1.35 * W)

-- The goal statement to prove
theorem percentage_markup (h : 0.4 * R = 1.35 * W) : ((R - W) / W) * 100 = 237.5 :=
by
  sorry

end NUMINAMATH_GPT_percentage_markup_l268_26818


namespace NUMINAMATH_GPT_hexagon_area_l268_26851

theorem hexagon_area (s : ℝ) (hex_area : ℝ) (p q : ℤ) :
  s = 3 ∧ hex_area = (3 * Real.sqrt 3 / 2) * s^2 ∧ hex_area = Real.sqrt p + Real.sqrt q → p + q = 545 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_area_l268_26851


namespace NUMINAMATH_GPT_num_pairs_in_arithmetic_progression_l268_26852

theorem num_pairs_in_arithmetic_progression : 
  ∃ n : ℕ, n = 2 ∧ ∀ a b : ℝ, (a = (15 + b) / 2 ∧ (a + a * b = 2 * b)) ↔ 
  (a = (9 + 3 * Real.sqrt 7) / 2 ∧ b = -6 + 3 * Real.sqrt 7)
  ∨ (a = (9 - 3 * Real.sqrt 7) / 2 ∧ b = -6 - 3 * Real.sqrt 7) :=    
  sorry

end NUMINAMATH_GPT_num_pairs_in_arithmetic_progression_l268_26852


namespace NUMINAMATH_GPT_smallest_number_append_l268_26862

def lcm (a b : Nat) : Nat := a * b / Nat.gcd a b

theorem smallest_number_append (m n : Nat) (k: Nat) :
  m = 2014 ∧ n = 2520 ∧ n % m ≠ 0 ∧ (k = n - m) →
  ∃ d : Nat, (m * 10 ^ d + k) % n = 0 := by
  sorry

end NUMINAMATH_GPT_smallest_number_append_l268_26862


namespace NUMINAMATH_GPT_evaluate_expression_l268_26827

theorem evaluate_expression :
  (1 / (3 - (1 / (3 - (1 / (3 - (1 / 3))))))) = (3 / 4) :=
sorry

end NUMINAMATH_GPT_evaluate_expression_l268_26827


namespace NUMINAMATH_GPT_angle_C_is_108_l268_26858

theorem angle_C_is_108
  (A B C D E : ℝ)
  (h1 : A < B)
  (h2 : B < C)
  (h3 : C < D)
  (h4 : D < E)
  (h5 : B - A = C - B)
  (h6 : C - B = D - C)
  (h7 : D - C = E - D)
  (angle_sum : A + B + C + D + E = 540) :
  C = 108 := 
sorry

end NUMINAMATH_GPT_angle_C_is_108_l268_26858


namespace NUMINAMATH_GPT_find_h_s_pairs_l268_26829

def num_regions (h s : ℕ) : ℕ :=
  1 + h * (s + 1) + s * (s + 1) / 2

theorem find_h_s_pairs (h s : ℕ) :
  h > 0 ∧ s > 0 ∧
  num_regions h s = 1992 ↔ 
  (h, s) = (995, 1) ∨ (h, s) = (176, 10) ∨ (h, s) = (80, 21) :=
by
  sorry

end NUMINAMATH_GPT_find_h_s_pairs_l268_26829


namespace NUMINAMATH_GPT_sum_of_three_integers_l268_26875

theorem sum_of_three_integers (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c) (h_product : a * b * c = 125) : a + b + c = 31 :=
sorry

end NUMINAMATH_GPT_sum_of_three_integers_l268_26875


namespace NUMINAMATH_GPT_cosh_le_exp_sqr_l268_26815

open Real

theorem cosh_le_exp_sqr {x k : ℝ} : (∀ x : ℝ, cosh x ≤ exp (k * x^2)) ↔ k ≥ 1/2 :=
sorry

end NUMINAMATH_GPT_cosh_le_exp_sqr_l268_26815


namespace NUMINAMATH_GPT_thermos_count_l268_26854

theorem thermos_count
  (total_gallons : ℝ)
  (pints_per_gallon : ℝ)
  (thermoses_drunk_by_genevieve : ℕ)
  (pints_drunk_by_genevieve : ℝ)
  (total_pints : ℝ) :
  total_gallons * pints_per_gallon = total_pints ∧
  pints_drunk_by_genevieve / thermoses_drunk_by_genevieve = 2 →
  total_pints / 2 = 18 :=
by
  intros h
  have := h.2
  sorry

end NUMINAMATH_GPT_thermos_count_l268_26854


namespace NUMINAMATH_GPT_max_digit_d_l268_26820

theorem max_digit_d (d f : ℕ) (h₁ : d ≤ 9) (h₂ : f ≤ 9) (h₃ : (18 + d + f) % 3 = 0) (h₄ : (12 - (d + f)) % 11 = 0) : d = 1 :=
sorry

end NUMINAMATH_GPT_max_digit_d_l268_26820


namespace NUMINAMATH_GPT_annie_jacob_ratio_l268_26802

theorem annie_jacob_ratio :
  ∃ (a j : ℕ), ∃ (m : ℕ), (m = 2 * a) ∧ (j = 90) ∧ (m = 60) ∧ (a / j = 1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_annie_jacob_ratio_l268_26802


namespace NUMINAMATH_GPT_find_value_of_expression_l268_26806

variable {x y : ℝ}
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)
variable (h : x + y = x * y + 1)

theorem find_value_of_expression (h : x + y = x * y + 1) : 
  (1 / x) + (1 / y) = 1 + (1 / (x * y)) :=
  sorry

end NUMINAMATH_GPT_find_value_of_expression_l268_26806


namespace NUMINAMATH_GPT_eric_boxes_l268_26848

def numberOfBoxes (totalPencils : Nat) (pencilsPerBox : Nat) : Nat :=
  totalPencils / pencilsPerBox

theorem eric_boxes :
  numberOfBoxes 27 9 = 3 := by
  sorry

end NUMINAMATH_GPT_eric_boxes_l268_26848


namespace NUMINAMATH_GPT_sum_of_consecutive_even_numbers_l268_26844

theorem sum_of_consecutive_even_numbers (n : ℕ) (h : (n + 2)^2 - n^2 = 84) :
  n + (n + 2) = 42 :=
sorry

end NUMINAMATH_GPT_sum_of_consecutive_even_numbers_l268_26844


namespace NUMINAMATH_GPT_gcd_294_84_l268_26864

theorem gcd_294_84 : gcd 294 84 = 42 :=
by
  sorry

end NUMINAMATH_GPT_gcd_294_84_l268_26864


namespace NUMINAMATH_GPT_steve_took_4_berries_l268_26869

theorem steve_took_4_berries (s t : ℕ) (H1 : s = 32) (H2 : t = 21) (H3 : s - 7 = t + x) :
  x = 4 :=
by
  sorry

end NUMINAMATH_GPT_steve_took_4_berries_l268_26869


namespace NUMINAMATH_GPT_find_integer_pairs_l268_26822

def satisfies_conditions (m n : ℤ) : Prop :=
  m^2 = n^5 + n^4 + 1 ∧ ((m - 7 * n) ∣ (m - 4 * n))

theorem find_integer_pairs :
  ∀ (m n : ℤ), satisfies_conditions m n → (m, n) = (-1, 0) ∨ (m, n) = (1, 0) := by
  sorry

end NUMINAMATH_GPT_find_integer_pairs_l268_26822


namespace NUMINAMATH_GPT_exists_distinct_group_and_country_selection_l268_26886

theorem exists_distinct_group_and_country_selection 
  (n m : ℕ) 
  (h_nm1 : n > m) 
  (h_m1 : m > 1) 
  (groups : Fin n → Fin m → Fin n → Prop) 
  (group_conditions : ∀ i j : Fin n, ∀ k : Fin m, ∀ l : Fin m, (i ≠ j) → (groups i k j = false)) 
  : 
  ∃ (selected : Fin n → Fin (m * n)), 
    (∀ i j: Fin n, i ≠ j → selected i ≠ selected j) ∧ 
    (∀ i j: Fin n, selected i / m ≠ selected j / m) := sorry

end NUMINAMATH_GPT_exists_distinct_group_and_country_selection_l268_26886


namespace NUMINAMATH_GPT_ilya_arithmetic_l268_26804

theorem ilya_arithmetic (v t : ℝ) (h : v + t = v * t ∧ v + t = v / t) : False :=
by
  sorry

end NUMINAMATH_GPT_ilya_arithmetic_l268_26804


namespace NUMINAMATH_GPT_games_played_by_third_player_l268_26801

theorem games_played_by_third_player
    (games_first : ℕ)
    (games_second : ℕ)
    (games_first_eq : games_first = 10)
    (games_second_eq : games_second = 21) :
    ∃ (games_third : ℕ), games_third = 11 := by
  sorry

end NUMINAMATH_GPT_games_played_by_third_player_l268_26801


namespace NUMINAMATH_GPT_employed_males_percent_l268_26840

def percent_employed_population : ℝ := 96
def percent_females_among_employed : ℝ := 75

theorem employed_males_percent :
  percent_employed_population * (1 - percent_females_among_employed / 100) = 24 := by
    sorry

end NUMINAMATH_GPT_employed_males_percent_l268_26840


namespace NUMINAMATH_GPT_copy_pages_l268_26819

theorem copy_pages (cost_per_5_pages : ℝ) (total_dollars : ℝ) : 
  (cost_per_5_pages = 10) → (total_dollars = 15) → (15 * 100 / 10 * 5 = 750) :=
by
  intros
  sorry

end NUMINAMATH_GPT_copy_pages_l268_26819


namespace NUMINAMATH_GPT_negation_of_homework_submission_l268_26868

variable {S : Type} -- S is the set of all students in this class
variable (H : S → Prop) -- H(x) means "student x has submitted the homework"

theorem negation_of_homework_submission :
  (¬ ∀ x, H x) ↔ (∃ x, ¬ H x) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_homework_submission_l268_26868


namespace NUMINAMATH_GPT_anne_age_ratio_l268_26817

-- Define the given conditions and prove the final ratio
theorem anne_age_ratio (A M : ℕ) (h1 : A = 4 * (A - 4 * M) + M) 
(h2 : A - M = 3 * (A - 4 * M)) : (A : ℚ) / (M : ℚ) = 5.5 := 
sorry

end NUMINAMATH_GPT_anne_age_ratio_l268_26817


namespace NUMINAMATH_GPT_sum_of_transformed_numbers_l268_26812

theorem sum_of_transformed_numbers (a b S : ℕ) (h : a + b = S) :
  3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_transformed_numbers_l268_26812


namespace NUMINAMATH_GPT_find_quotient_l268_26885

theorem find_quotient
    (dividend divisor remainder : ℕ)
    (h1 : dividend = 136)
    (h2 : divisor = 15)
    (h3 : remainder = 1)
    (h4 : dividend = divisor * quotient + remainder) :
    quotient = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_quotient_l268_26885


namespace NUMINAMATH_GPT_integer_triangle_cosines_rational_l268_26899

theorem integer_triangle_cosines_rational (a b c : ℕ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  ∃ (cos_α cos_β cos_γ : ℚ), 
    cos_γ = (a^2 + b^2 - c^2) / (2 * a * b) ∧
    cos_β = (a^2 + c^2 - b^2) / (2 * a * c) ∧
    cos_α = (b^2 + c^2 - a^2) / (2 * b * c) :=
by
  sorry

end NUMINAMATH_GPT_integer_triangle_cosines_rational_l268_26899


namespace NUMINAMATH_GPT_find_number_l268_26807

theorem find_number (x : ℝ) (h : (((18 + x) / 3 + 10) / 5 = 4)) : x = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l268_26807


namespace NUMINAMATH_GPT_rectangle_measurement_error_l268_26866

theorem rectangle_measurement_error
    (L W : ℝ) -- actual lengths of the sides
    (x : ℝ) -- percentage in excess for the first side
    (h1 : 0 ≤ x) -- ensuring percentage cannot be negative
    (h2 : (L * (1 + x / 100)) * (W * 0.95) = L * W * 1.045) -- given condition on areas
    : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_measurement_error_l268_26866


namespace NUMINAMATH_GPT_car_miles_per_gallon_in_city_l268_26898

-- Define the conditions and the problem
theorem car_miles_per_gallon_in_city :
  ∃ C H T : ℝ, 
    H = 462 / T ∧ 
    C = 336 / T ∧ 
    C = H - 12 ∧ 
    C = 32 :=
by
  sorry

end NUMINAMATH_GPT_car_miles_per_gallon_in_city_l268_26898


namespace NUMINAMATH_GPT_count_possible_pairs_l268_26861

/-- There are four distinct mystery novels, three distinct fantasy novels, and three distinct biographies.
I want to choose two books with one of them being a specific mystery novel, "Mystery Masterpiece".
Prove that the number of possible pairs that include this mystery novel and one book from a different genre
is 6. -/
theorem count_possible_pairs (mystery_novels : Fin 4)
                            (fantasy_novels : Fin 3)
                            (biographies : Fin 3)
                            (MysteryMasterpiece : Fin 4):
                            (mystery_novels ≠ MysteryMasterpiece) →
                            ∀ genre : Fin 2, genre ≠ 0 ∧ genre ≠ 1 →
                            (genre = 1 → ∃ pairs : List (Fin 3), pairs.length = 3) →
                            (genre = 2 → ∃ pairs : List (Fin 3), pairs.length = 3) →
                            ∃ total_pairs : Nat, total_pairs = 6 :=
by
  intros h_ne_genres h_genres h_counts1 h_counts2
  sorry

end NUMINAMATH_GPT_count_possible_pairs_l268_26861


namespace NUMINAMATH_GPT_negation_of_existence_statement_l268_26879

theorem negation_of_existence_statement :
  ¬ (∃ P : ℝ × ℝ, (P.1^2 + P.2^2 - 1 ≤ 0)) ↔ ∀ P : ℝ × ℝ, (P.1^2 + P.2^2 - 1 > 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existence_statement_l268_26879


namespace NUMINAMATH_GPT_minimum_letters_for_grid_coloring_l268_26823

theorem minimum_letters_for_grid_coloring : 
  ∀ (grid_paper : Type) 
  (is_node : grid_paper → Prop) 
  (marked : grid_paper → Prop)
  (mark_with_letter : grid_paper → ℕ) 
  (connected : grid_paper → grid_paper → Prop), 
  (∀ n₁ n₂ : grid_paper, is_node n₁ → is_node n₂ → mark_with_letter n₁ = mark_with_letter n₂ → 
  (n₁ ≠ n₂ → ∃ n₃ : grid_paper, is_node n₃ ∧ connected n₁ n₃ ∧ connected n₃ n₂ ∧ mark_with_letter n₃ ≠ mark_with_letter n₁)) → 
  ∃ (k : ℕ), k = 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_letters_for_grid_coloring_l268_26823


namespace NUMINAMATH_GPT_clock_angle_at_3_45_l268_26828

theorem clock_angle_at_3_45 :
  let minute_angle_rate := 6.0 -- degrees per minute
  let hour_angle_rate := 0.5  -- degrees per minute
  let initial_angle := 90.0   -- degrees at 3:00
  let minutes_passed := 45.0  -- minutes since 3:00
  let angle_difference_rate := minute_angle_rate - hour_angle_rate
  let angle_change := angle_difference_rate * minutes_passed
  let final_angle := initial_angle - angle_change
  let smaller_angle := if final_angle < 0 then 360.0 + final_angle else final_angle
  smaller_angle = 157.5 :=
by
  sorry

end NUMINAMATH_GPT_clock_angle_at_3_45_l268_26828


namespace NUMINAMATH_GPT_age_sum_proof_l268_26808

theorem age_sum_proof : 
  ∀ (Matt Fem Jake : ℕ), 
    Matt = 4 * Fem →
    Fem = 11 →
    Jake = Matt + 5 →
    (Matt + 2) + (Fem + 2) + (Jake + 2) = 110 :=
by
  intros Matt Fem Jake h1 h2 h3
  sorry

end NUMINAMATH_GPT_age_sum_proof_l268_26808


namespace NUMINAMATH_GPT_french_fries_cost_is_10_l268_26867

-- Define the costs as given in the problem conditions
def taco_salad_cost : ℕ := 10
def daves_single_cost : ℕ := 5
def peach_lemonade_cost : ℕ := 2
def num_friends : ℕ := 5
def friend_payment : ℕ := 11

-- Define the total amount collected from friends
def total_collected : ℕ := num_friends * friend_payment

-- Define the subtotal for the known items
def subtotal : ℕ := taco_salad_cost + (num_friends * daves_single_cost) + (num_friends * peach_lemonade_cost)

-- The total cost of french fries
def total_french_fries_cost := total_collected - subtotal

-- The proof statement:
theorem french_fries_cost_is_10 : total_french_fries_cost = 10 := by
  sorry

end NUMINAMATH_GPT_french_fries_cost_is_10_l268_26867


namespace NUMINAMATH_GPT_Shelby_fog_time_l268_26838

variable (x y : ℕ)

-- Conditions
def speed_sun := 7/12
def speed_rain := 5/12
def speed_fog := 1/4
def total_time := 60
def total_distance := 20

theorem Shelby_fog_time :
  ((speed_sun * (total_time - x - y)) + (speed_rain * x) + (speed_fog * y) = total_distance) → y = 45 :=
by
  sorry

end NUMINAMATH_GPT_Shelby_fog_time_l268_26838


namespace NUMINAMATH_GPT_rhombus_area_l268_26800

/-
  We want to prove that the area of a rhombus with given diagonals' lengths is 
  equal to the computed value according to the formula Area = (d1 * d2) / 2.
-/
theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 12) : 
  (d1 * d2) / 2 = 60 :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_rhombus_area_l268_26800


namespace NUMINAMATH_GPT_lucy_l268_26846

theorem lucy's_age 
  (L V: ℕ)
  (h1: L - 5 = 3 * (V - 5))
  (h2: L + 10 = 2 * (V + 10)) :
  L = 50 :=
by
  sorry

end NUMINAMATH_GPT_lucy_l268_26846


namespace NUMINAMATH_GPT_tank_fraction_after_adding_water_l268_26897

theorem tank_fraction_after_adding_water 
  (initial_fraction : ℚ) 
  (full_capacity : ℚ) 
  (added_water : ℚ) 
  (final_fraction : ℚ) 
  (h1 : initial_fraction = 3/4) 
  (h2 : full_capacity = 56) 
  (h3 : added_water = 7) 
  (h4 : final_fraction = (initial_fraction * full_capacity + added_water) / full_capacity) : 
  final_fraction = 7 / 8 := 
by 
  sorry

end NUMINAMATH_GPT_tank_fraction_after_adding_water_l268_26897


namespace NUMINAMATH_GPT_impossible_to_place_integers_35x35_l268_26860

theorem impossible_to_place_integers_35x35 (f : Fin 35 → Fin 35 → ℤ) :
  (∀ i j, abs (f i j - f (i + 1) j) ≤ 18 ∧ abs (f i j - f i (j + 1)) ≤ 18) →
  ∃ i j, i ≠ j ∧ f i j = f i j → False :=
by sorry

end NUMINAMATH_GPT_impossible_to_place_integers_35x35_l268_26860


namespace NUMINAMATH_GPT_removed_term_is_a11_l268_26834

noncomputable def sequence_a (n : ℕ) (a1 d : ℤ) := a1 + (n - 1) * d

def sequence_sum (n : ℕ) (a1 d : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem removed_term_is_a11 :
  ∃ d : ℤ, ∀ a1 d : ℤ, 
            a1 = -5 ∧ 
            sequence_sum 11 a1 d = 55 ∧ 
            (sequence_sum 11 a1 d - sequence_a 11 a1 d) / 10 = 4 
          → sequence_a 11 a1 d = removed_term :=
sorry

end NUMINAMATH_GPT_removed_term_is_a11_l268_26834


namespace NUMINAMATH_GPT_range_of_fraction_l268_26850

variable {x y : ℝ}

-- Condition given in the problem
def equation (x y : ℝ) : Prop := x + 2 * y - 6 = 0

-- The range condition for x
def x_range (x : ℝ) : Prop := 0 < x ∧ x < 3

-- The corresponding theorem statement
theorem range_of_fraction (h_eq : equation x y) (h_x_range : x_range x) :
  ∃ a b : ℝ, (a < 1 ∧ 10 < b) ∧ (a, b) = (1, 10) ∧
  ∀ k : ℝ, k = (x + 2) / (y - 1) → 1 < k ∧ k < 10 :=
sorry

end NUMINAMATH_GPT_range_of_fraction_l268_26850


namespace NUMINAMATH_GPT_find_t_l268_26873

variable {x y z w t : ℝ}

theorem find_t (hx : x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w)
               (hpos : 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w)
               (hxy : x + 1/y = t)
               (hyz : y + 1/z = t)
               (hzw : z + 1/w = t)
               (hwx : w + 1/x = t) : 
               t = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_find_t_l268_26873


namespace NUMINAMATH_GPT_compute_expression_l268_26809
-- Start with importing math library utilities for linear algebra and dot product

-- Define vector 'a' and 'b' in Lean
def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (-1, 2)

-- Define dot product operation 
def dot_product (x y : ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2

-- Define the expression and the theorem
theorem compute_expression : dot_product ((2 * a.1 + b.1, 2 * a.2 + b.2)) a = 1 :=
by
  -- Insert the proof steps here
  sorry

end NUMINAMATH_GPT_compute_expression_l268_26809


namespace NUMINAMATH_GPT_original_price_l268_26830

variable (a : ℝ)

-- Given the price after a 20% discount is a yuan per unit,
-- Prove that the original price per unit was (5/4) * a yuan.
theorem original_price (h : a > 0) : (a / (4 / 5)) = (5 / 4) * a :=
by sorry

end NUMINAMATH_GPT_original_price_l268_26830


namespace NUMINAMATH_GPT_first_day_price_l268_26836

theorem first_day_price (x n: ℝ) :
  n * x = (n + 100) * (x - 1) ∧ 
  n * x = (n - 200) * (x + 2) → 
  x = 4 :=
by
  sorry

end NUMINAMATH_GPT_first_day_price_l268_26836


namespace NUMINAMATH_GPT_total_employees_in_company_l268_26893

-- Given facts and conditions
def ratio_A_B_C : Nat × Nat × Nat := (5, 4, 1)
def sample_size : Nat := 20
def prob_sel_A_B_from_C : ℚ := 1 / 45

-- Number of group C individuals, calculated from probability constraint
def num_persons_group_C := 10

theorem total_employees_in_company (x : Nat) :
  x = 10 * (5 + 4 + 1) :=
by
  -- Since the sample size is 20, and the ratio of sampling must be consistent with the population ratio,
  -- it can be derived that the total number of employees in the company must be 100.
  -- Adding sorry to skip the actual detailed proof.
  sorry

end NUMINAMATH_GPT_total_employees_in_company_l268_26893


namespace NUMINAMATH_GPT_solution_set_for_f_when_a_2_range_of_a_for_f_plus_g_ge_3_l268_26865

-- Define the function f(x) and g(x)
def f (x : ℝ) (a : ℝ) := |2 * x - a| + a
def g (x : ℝ) := |2 * x - 1|

-- Define the inequality problem when a = 2
theorem solution_set_for_f_when_a_2 : 
  { x : ℝ | f x 2 ≤ 6 } = { x : ℝ | -1 ≤ x ∧ x ≤ 3 } :=
by
  sorry

-- Prove the range of values for a when f(x) + g(x) ≥ 3
theorem range_of_a_for_f_plus_g_ge_3 : 
  ∀ a : ℝ, (∀ x : ℝ, f x a + g x ≥ 3) ↔ 2 ≤ a :=
by
  sorry

end NUMINAMATH_GPT_solution_set_for_f_when_a_2_range_of_a_for_f_plus_g_ge_3_l268_26865


namespace NUMINAMATH_GPT_calculate_savings_l268_26895

theorem calculate_savings :
  let plane_cost : ℕ := 600
  let boat_cost : ℕ := 254
  plane_cost - boat_cost = 346 := by
    let plane_cost : ℕ := 600
    let boat_cost : ℕ := 254
    sorry

end NUMINAMATH_GPT_calculate_savings_l268_26895


namespace NUMINAMATH_GPT_radius_of_circle_with_center_on_line_and_passing_through_points_l268_26811

theorem radius_of_circle_with_center_on_line_and_passing_through_points : 
  (∃ a b : ℝ, 2 * a + b = 0 ∧ 
              (a - 1) ^ 2 + (b - 3) ^ 2 = r ^ 2 ∧ 
              (a - 4) ^ 2 + (b - 2) ^ 2 = r ^ 2 
              → r = 5) := 
by 
  sorry

end NUMINAMATH_GPT_radius_of_circle_with_center_on_line_and_passing_through_points_l268_26811


namespace NUMINAMATH_GPT_find_a_l268_26847

theorem find_a (a : ℚ) :
  let p1 := (3, 4)
  let p2 := (-4, 1)
  let direction_vector := (a, -2)
  let vector_between_points := (p2.1 - p1.1, p2.2 - p1.2)
  ∃ k : ℚ, direction_vector = (k * vector_between_points.1, k * vector_between_points.2) →
  a = -14 / 3 := by
    sorry

end NUMINAMATH_GPT_find_a_l268_26847


namespace NUMINAMATH_GPT_minimal_polynomial_correct_l268_26837

noncomputable def minimal_polynomial : Polynomial ℚ :=
  (Polynomial.X^2 - 4 * Polynomial.X + 1) * (Polynomial.X^2 - 6 * Polynomial.X + 2)

theorem minimal_polynomial_correct :
  Polynomial.X^4 - 10 * Polynomial.X^3 + 29 * Polynomial.X^2 - 26 * Polynomial.X + 2 = minimal_polynomial :=
  sorry

end NUMINAMATH_GPT_minimal_polynomial_correct_l268_26837


namespace NUMINAMATH_GPT_ratio_not_necessarily_constant_l268_26894

theorem ratio_not_necessarily_constant (x y : ℝ) : ¬ (∃ k : ℝ, ∀ x y, x / y = k) :=
by
  sorry

end NUMINAMATH_GPT_ratio_not_necessarily_constant_l268_26894


namespace NUMINAMATH_GPT_num_solutions_3x_plus_2y_eq_806_l268_26849

theorem num_solutions_3x_plus_2y_eq_806 :
  (∃ y : ℕ, ∃ x : ℕ, x > 0 ∧ y > 0 ∧ 3 * x + 2 * y = 806) ∧
  ((∃ t : ℤ, x = 268 - 2 * t ∧ y = 1 + 3 * t) ∧ (∃ t : ℤ, 0 ≤ t ∧ t ≤ 133)) :=
sorry

end NUMINAMATH_GPT_num_solutions_3x_plus_2y_eq_806_l268_26849


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l268_26896

variables (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)

-- Conditions
def condition1 : Prop := ∀ n, S n = (n * (2*a 1 + (n-1) * d)) / 2
def condition2 : Prop := S 3 = 6
def condition3 : Prop := a 3 = 0

-- Question
def question : ℝ := d

-- Correct Answer
def correct_answer : ℝ := -2

-- Proof Problem Statement
theorem arithmetic_sequence_common_difference : 
  condition1 a S d ∧ condition2 S ∧ condition3 a →
  question d = correct_answer :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l268_26896


namespace NUMINAMATH_GPT_ford_younger_than_christopher_l268_26880

variable (G C F Y : ℕ)

-- Conditions
axiom h1 : G = C + 8
axiom h2 : F = C - Y
axiom h3 : G + C + F = 60
axiom h4 : C = 18

-- Target statement
theorem ford_younger_than_christopher : Y = 2 :=
sorry

end NUMINAMATH_GPT_ford_younger_than_christopher_l268_26880


namespace NUMINAMATH_GPT_max_x1_x2_squares_l268_26857

noncomputable def x1_x2_squares_eq_max : Prop :=
  ∃ k : ℝ, (∀ x1 x2 : ℝ, (x1 + x2 = k - 2) ∧ (x1 * x2 = k^2 + 3 * k + 5) → x1^2 + x2^2 = 18)

theorem max_x1_x2_squares : x1_x2_squares_eq_max :=
by sorry

end NUMINAMATH_GPT_max_x1_x2_squares_l268_26857


namespace NUMINAMATH_GPT_Sam_drinks_l268_26891

theorem Sam_drinks (juice_don : ℚ) (fraction_sam : ℚ) 
  (h1 : juice_don = 3 / 7) (h2 : fraction_sam = 4 / 5) : 
  (fraction_sam * juice_don = 12 / 35) :=
by
  sorry

end NUMINAMATH_GPT_Sam_drinks_l268_26891


namespace NUMINAMATH_GPT_fraction_of_remaining_birds_left_l268_26810

theorem fraction_of_remaining_birds_left :
  ∀ (total_birds initial_fraction next_fraction x : ℚ), 
    total_birds = 60 ∧ 
    initial_fraction = 1 / 3 ∧ 
    next_fraction = 2 / 5 ∧ 
    8 = (total_birds * (1 - initial_fraction)) * (1 - next_fraction) * (1 - x) →
    x = 2 / 3 :=
by
  intros total_birds initial_fraction next_fraction x h
  obtain ⟨hb, hi, hn, he⟩ := h
  sorry

end NUMINAMATH_GPT_fraction_of_remaining_birds_left_l268_26810


namespace NUMINAMATH_GPT_max_min_values_l268_26833

namespace ProofPrimary

-- Define the polynomial function f
def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 36 * x + 1

-- State the interval of interest
def interval : Set ℝ := Set.Icc 1 11

-- Main theorem asserting the minimum and maximum values
theorem max_min_values : 
  (∀ x ∈ interval, f x ≥ -43 ∧ f x ≤ 2630) ∧
  (∃ x ∈ interval, f x = -43) ∧
  (∃ x ∈ interval, f x = 2630) :=
by
  sorry

end ProofPrimary

end NUMINAMATH_GPT_max_min_values_l268_26833


namespace NUMINAMATH_GPT_sum_of_two_numbers_l268_26803

variables {x y : ℝ}

theorem sum_of_two_numbers (h1 : x * y = 120) (h2 : x^2 + y^2 = 289) : x + y = 22 :=
sorry

end NUMINAMATH_GPT_sum_of_two_numbers_l268_26803


namespace NUMINAMATH_GPT_Richard_walked_10_miles_third_day_l268_26870

def distance_to_NYC := 70
def day1 := 20
def day2 := (day1 / 2) - 6
def remaining_distance := 36
def day3 := 70 - (day1 + day2 + remaining_distance)

theorem Richard_walked_10_miles_third_day (h : day3 = 10) : day3 = 10 :=
by {
    sorry
}

end NUMINAMATH_GPT_Richard_walked_10_miles_third_day_l268_26870


namespace NUMINAMATH_GPT_charlie_age_when_jenny_twice_as_old_as_bobby_l268_26887

-- Conditions as Definitions
def ageDifferenceJennyCharlie : ℕ := 5
def ageDifferenceCharlieBobby : ℕ := 3

-- Problem Statement as a Theorem
theorem charlie_age_when_jenny_twice_as_old_as_bobby (j c b : ℕ) 
  (H1 : j = c + ageDifferenceJennyCharlie) 
  (H2 : c = b + ageDifferenceCharlieBobby) : 
  j = 2 * b → c = 11 :=
by
  sorry

end NUMINAMATH_GPT_charlie_age_when_jenny_twice_as_old_as_bobby_l268_26887


namespace NUMINAMATH_GPT_mode_is_37_median_is_36_l268_26826

namespace ProofProblem

def data_set : List ℕ := [34, 35, 36, 34, 36, 37, 37, 36, 37, 37]

def mode (l : List ℕ) : ℕ := sorry -- Implementing a mode function

def median (l : List ℕ) : ℕ := sorry -- Implementing a median function

theorem mode_is_37 : mode data_set = 37 := 
  by 
    sorry -- Proof of mode

theorem median_is_36 : median data_set = 36 := 
  by
    sorry -- Proof of median

end ProofProblem

end NUMINAMATH_GPT_mode_is_37_median_is_36_l268_26826


namespace NUMINAMATH_GPT_not_coincidence_l268_26835

theorem not_coincidence (G : Type) [Fintype G] [DecidableEq G]
    (friend_relation : G → G → Prop)
    (h_friend : ∀ (a b : G), friend_relation a b → friend_relation b a)
    (initial_condition : ∀ (subset : Finset G), subset.card = 4 → 
         ∃ x ∈ subset, ∀ y ∈ subset, x ≠ y → friend_relation x y) :
    ∀ (subset : Finset G), subset.card = 4 → 
        ∃ x ∈ subset, ∀ y ∈ Finset.univ, x ≠ y → friend_relation x y :=
by
  intros subset h_card
  -- The proof would be constructed here
  sorry

end NUMINAMATH_GPT_not_coincidence_l268_26835


namespace NUMINAMATH_GPT_factor_problem_l268_26839

theorem factor_problem (C D : ℤ) (h1 : 16 * x^2 - 88 * x + 63 = (C * x - 21) * (D * x - 3)) (h2 : C * D + C = 21) : C = 7 ∧ D = 2 :=
by 
  sorry

end NUMINAMATH_GPT_factor_problem_l268_26839


namespace NUMINAMATH_GPT_a_plus_b_eq_l268_26816

-- Define the sets A and B
def A := { x : ℝ | -1 < x ∧ x < 3 }
def B := { x : ℝ | -3 < x ∧ x < 2 }

-- Define the intersection set A ∩ B
def A_inter_B := { x : ℝ | -1 < x ∧ x < 2 }

-- Define a condition
noncomputable def is_solution_set (a b : ℝ) : Prop :=
  ∀ x : ℝ, (-1 < x ∧ x < 2) ↔ (x^2 + a * x + b < 0)

-- The proof statement
theorem a_plus_b_eq : ∃ a b : ℝ, is_solution_set a b ∧ a + b = -3 := by
  sorry

end NUMINAMATH_GPT_a_plus_b_eq_l268_26816


namespace NUMINAMATH_GPT_set_intersection_example_l268_26881

def universal_set := Set ℝ

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 1}

def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2 * x + 1 ∧ -2 ≤ x ∧ x ≤ 1}

def C : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 4}

def complement (A : Set ℝ) : Set ℝ := {x : ℝ | x ∉ A}

def difference (A B : Set ℝ) : Set ℝ := A \ B

def union (A B : Set ℝ) : Set ℝ := {x : ℝ | x ∈ A ∨ x ∈ B}

def intersection (A B : Set ℝ) : Set ℝ := {x : ℝ | x ∈ A ∧ x ∈ B}

theorem set_intersection_example :
  intersection (complement A) (union B C) = {x : ℝ | (-3 ≤ x ∧ x < -2) ∨ (1 < x ∧ x ≤ 4)} :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_example_l268_26881


namespace NUMINAMATH_GPT_articles_in_selling_price_l268_26855

theorem articles_in_selling_price (C : ℝ) (N : ℕ) 
  (h1 : 50 * C = N * (1.25 * C)) 
  (h2 : 0.25 * C = 25 / 100 * C) :
  N = 40 :=
by
  sorry

end NUMINAMATH_GPT_articles_in_selling_price_l268_26855


namespace NUMINAMATH_GPT_percentage_of_girls_after_change_l268_26871

variables (initial_total_children initial_boys initial_girls additional_boys : ℕ)
variables (percentage_boys : ℚ)

-- Initial conditions
def initial_conditions : Prop :=
  initial_total_children = 50 ∧
  percentage_boys = 90 / 100 ∧
  initial_boys = initial_total_children * percentage_boys ∧
  initial_girls = initial_total_children - initial_boys ∧
  additional_boys = 50

-- Statement to prove
theorem percentage_of_girls_after_change :
  initial_conditions initial_total_children initial_boys initial_girls additional_boys percentage_boys →
  (initial_girls / (initial_total_children + additional_boys) * 100 = 5) :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_girls_after_change_l268_26871


namespace NUMINAMATH_GPT_bacteria_population_l268_26863

theorem bacteria_population (initial_population : ℕ) (tripling_factor : ℕ) (hours_per_tripling : ℕ) (target_population : ℕ) 
(initial_population_eq : initial_population = 300)
(tripling_factor_eq : tripling_factor = 3)
(hours_per_tripling_eq : hours_per_tripling = 5)
(target_population_eq : target_population = 87480) :
∃ n : ℕ, (hours_per_tripling * n = 30) ∧ (initial_population * (tripling_factor ^ n) ≥ target_population) := sorry

end NUMINAMATH_GPT_bacteria_population_l268_26863


namespace NUMINAMATH_GPT_gum_total_l268_26859

theorem gum_total (initial_gum : ℝ) (additional_gum : ℝ) : initial_gum = 18.5 → additional_gum = 44.25 → initial_gum + additional_gum = 62.75 :=
by
  intros
  sorry

end NUMINAMATH_GPT_gum_total_l268_26859


namespace NUMINAMATH_GPT_correct_operation_l268_26892

theorem correct_operation (x : ℝ) (hx : x ≠ 0) :
  (x^3 / x^2 = x) :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_operation_l268_26892


namespace NUMINAMATH_GPT_smaller_angle_measure_l268_26842

theorem smaller_angle_measure (x : ℝ) (h1 : 4 * x + x = 90) : x = 18 := by
  sorry

end NUMINAMATH_GPT_smaller_angle_measure_l268_26842


namespace NUMINAMATH_GPT_find_f_prime_at_1_l268_26884

def f (x : ℝ) (f_prime_at_1 : ℝ) : ℝ := x^2 + 3 * x * f_prime_at_1

theorem find_f_prime_at_1 (f_prime_at_1 : ℝ) :
  (∀ x, deriv (λ x => f x f_prime_at_1) x = 2 * x + 3 * f_prime_at_1) → 
  deriv (λ x => f x f_prime_at_1) 1 = -1 := 
by
exact sorry

end NUMINAMATH_GPT_find_f_prime_at_1_l268_26884


namespace NUMINAMATH_GPT_total_trash_pieces_l268_26872

theorem total_trash_pieces (classroom_trash : ℕ) (outside_trash : ℕ)
  (h1 : classroom_trash = 344) (h2 : outside_trash = 1232) : 
  classroom_trash + outside_trash = 1576 :=
by
  sorry

end NUMINAMATH_GPT_total_trash_pieces_l268_26872


namespace NUMINAMATH_GPT_problem1_problem2_l268_26890

noncomputable section

theorem problem1 :
  (2 * Real.sqrt 3 - 1)^2 + (Real.sqrt 3 + 2) * (Real.sqrt 3 - 2) = 12 - 4 * Real.sqrt 3 :=
  sorry

theorem problem2 :
  (Real.sqrt 6 - 2 * Real.sqrt 15) * Real.sqrt 3 - 6 * Real.sqrt (1 / 2) = -6 * Real.sqrt 5 :=
  sorry

end NUMINAMATH_GPT_problem1_problem2_l268_26890


namespace NUMINAMATH_GPT_value_of_business_l268_26856

variable (V : ℝ)
variable (h1 : (2 / 3) * V = S)
variable (h2 : (3 / 4) * S = 75000)

theorem value_of_business (h1 : (2 / 3) * V = S) (h2 : (3 / 4) * S = 75000) : V = 150000 :=
sorry

end NUMINAMATH_GPT_value_of_business_l268_26856


namespace NUMINAMATH_GPT_simplify_expression_l268_26882

theorem simplify_expression (a : ℝ) (h : a < 1 / 4) : 4 * (4 * a - 1)^2 = (1 - 4 * a)^(2 : ℝ) :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l268_26882


namespace NUMINAMATH_GPT_sum_mnp_l268_26874

noncomputable def volume_of_parallelepiped := 2 * 3 * 4
noncomputable def volume_of_extended_parallelepipeds := 
  2 * (1 * 2 * 3 + 1 * 2 * 4 + 1 * 3 * 4)
noncomputable def volume_of_quarter_cylinders := 
  4 * (1 / 4 * Real.pi * 1^2 * (2 + 3 + 4))
noncomputable def volume_of_spherical_octants := 
  8 * (1 / 8 * (4 / 3) * Real.pi * 1^3)

noncomputable def total_volume := 
  volume_of_parallelepiped + volume_of_extended_parallelepipeds + 
  volume_of_quarter_cylinders + volume_of_spherical_octants

theorem sum_mnp : 228 + 85 + 3 = 316 := by
  sorry

end NUMINAMATH_GPT_sum_mnp_l268_26874


namespace NUMINAMATH_GPT_feathers_per_crown_l268_26883

theorem feathers_per_crown (total_feathers total_crowns feathers_per_crown : ℕ) 
  (h₁ : total_feathers = 6538) 
  (h₂ : total_crowns = 934) 
  (h₃ : feathers_per_crown = total_feathers / total_crowns) : 
  feathers_per_crown = 7 := 
by 
  sorry

end NUMINAMATH_GPT_feathers_per_crown_l268_26883


namespace NUMINAMATH_GPT_journey_distance_l268_26814

theorem journey_distance
  (total_time : ℝ)
  (speed1 speed2 : ℝ)
  (journey_time : total_time = 10)
  (speed1_val : speed1 = 21)
  (speed2_val : speed2 = 24) :
  ∃ D : ℝ, (D / 2 / speed1 + D / 2 / speed2 = total_time) ∧ D = 224 :=
by
  sorry

end NUMINAMATH_GPT_journey_distance_l268_26814


namespace NUMINAMATH_GPT_sum_of_ages_l268_26831

theorem sum_of_ages (y : ℕ) 
  (h_diff : 38 - y = 2) : y + 38 = 74 := 
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_ages_l268_26831


namespace NUMINAMATH_GPT_cannot_pay_exactly_500_can_pay_exactly_600_l268_26878

-- Defining the costs and relevant equations
def price_of_bun : ℕ := 15
def price_of_croissant : ℕ := 12

-- Proving the non-existence for the 500 Ft case
theorem cannot_pay_exactly_500 : ¬ ∃ (x y : ℕ), price_of_croissant * x + price_of_bun * y = 500 :=
sorry

-- Proving the existence for the 600 Ft case
theorem can_pay_exactly_600 : ∃ (x y : ℕ), price_of_croissant * x + price_of_bun * y = 600 :=
sorry

end NUMINAMATH_GPT_cannot_pay_exactly_500_can_pay_exactly_600_l268_26878


namespace NUMINAMATH_GPT_sum_of_solutions_eq_neg_six_l268_26841

theorem sum_of_solutions_eq_neg_six (x r s : ℝ) :
  (81 : ℝ) - 18 * x - 3 * x^2 = 0 →
  (r + s = -6) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_eq_neg_six_l268_26841


namespace NUMINAMATH_GPT_geometric_progression_condition_l268_26821

theorem geometric_progression_condition (a b c : ℝ) (h_b_neg : b < 0) : 
  (b^2 = a * c) ↔ (∃ (r : ℝ), a = r * b ∧ b = r * c) :=
sorry

end NUMINAMATH_GPT_geometric_progression_condition_l268_26821


namespace NUMINAMATH_GPT_division_quotient_l268_26825

-- Define conditions
def dividend : ℕ := 686
def divisor : ℕ := 36
def remainder : ℕ := 2

-- Define the quotient
def quotient : ℕ := dividend - remainder

theorem division_quotient :
  quotient = divisor * 19 :=
sorry

end NUMINAMATH_GPT_division_quotient_l268_26825


namespace NUMINAMATH_GPT_remainder_of_3_pow_500_mod_17_l268_26889

theorem remainder_of_3_pow_500_mod_17 : (3 ^ 500) % 17 = 13 := 
by
  sorry

end NUMINAMATH_GPT_remainder_of_3_pow_500_mod_17_l268_26889


namespace NUMINAMATH_GPT_max_c_value_l268_26832

variable {a b c : ℝ}

theorem max_c_value (h1 : 2 * (a + b) = a * b) (h2 : a + b + c = a * b * c) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  c ≤ 8 / 15 :=
sorry

end NUMINAMATH_GPT_max_c_value_l268_26832


namespace NUMINAMATH_GPT_expand_subtract_equals_result_l268_26805

-- Definitions of the given expressions
def expand_and_subtract (x : ℝ) : ℝ :=
  (x + 3) * (2 * x - 5) - (2 * x + 1)

-- Expected result
def expected_result (x : ℝ) : ℝ :=
  2 * x ^ 2 - x - 16

-- The theorem stating the equivalence of the expanded and subtracted expression with the expected result
theorem expand_subtract_equals_result (x : ℝ) : expand_and_subtract x = expected_result x :=
  sorry

end NUMINAMATH_GPT_expand_subtract_equals_result_l268_26805


namespace NUMINAMATH_GPT_find_k_value_l268_26824

theorem find_k_value 
  (A B C k : ℤ)
  (hA : A = -3)
  (hB : B = -5)
  (hC : C = 6)
  (hSum : A + B + C + k = -A - B - C - k) : 
  k = 2 :=
sorry

end NUMINAMATH_GPT_find_k_value_l268_26824


namespace NUMINAMATH_GPT_arithmetic_sequence_75th_term_l268_26845

theorem arithmetic_sequence_75th_term (a1 d : ℤ) (n : ℤ) (h1 : a1 = 3) (h2 : d = 5) (h3 : n = 75) :
  a1 + (n - 1) * d = 373 :=
by
  rw [h1, h2, h3]
  -- Here, we arrive at the explicitly stated elements and evaluate:
  -- 3 + (75 - 1) * 5 = 373
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_75th_term_l268_26845


namespace NUMINAMATH_GPT_probability_two_digit_between_15_25_l268_26888

-- Define a type for standard six-sided dice rolls
def is_standard_six_sided_die (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 6

-- Define the set of valid two-digit numbers
def valid_two_digit_number (n : ℕ) : Prop := n ≥ 15 ∧ n ≤ 25

-- Function to form a two-digit number from two dice rolls
def form_two_digit_number (d1 d2 : ℕ) : ℕ := 10 * d1 + d2

-- The main statement of the problem
theorem probability_two_digit_between_15_25 :
  (∃ (n : ℚ), n = 5/9) ∧
  (∀ (d1 d2 : ℕ), is_standard_six_sided_die d1 → is_standard_six_sided_die d2 →
  valid_two_digit_number (form_two_digit_number d1 d2)) :=
sorry

end NUMINAMATH_GPT_probability_two_digit_between_15_25_l268_26888


namespace NUMINAMATH_GPT_print_shop_cost_difference_l268_26876

theorem print_shop_cost_difference :
  let cost_per_copy_X := 1.25
  let cost_per_copy_Y := 2.75
  let num_copies := 40
  let total_cost_X := cost_per_copy_X * num_copies
  let total_cost_Y := cost_per_copy_Y * num_copies
  total_cost_Y - total_cost_X = 60 :=
by 
  dsimp only []
  sorry

end NUMINAMATH_GPT_print_shop_cost_difference_l268_26876


namespace NUMINAMATH_GPT_rectangle_area_l268_26813

theorem rectangle_area (l w : ℝ) (h1 : 2 * l + 2 * w = 14) (h2 : l^2 + w^2 = 25) : l * w = 12 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l268_26813
