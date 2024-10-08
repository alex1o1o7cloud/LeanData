import Mathlib

namespace find_point_C_on_z_axis_l28_28597

noncomputable def point_c_condition (C : ℝ × ℝ × ℝ) (A B : ℝ × ℝ × ℝ) : Prop :=
  dist C A = dist C B

theorem find_point_C_on_z_axis :
  ∃ C : ℝ × ℝ × ℝ, C = (0, 0, 1) ∧ point_c_condition C (1, 0, 2) (1, 1, 1) :=
by
  use (0, 0, 1)
  simp [point_c_condition]
  sorry

end find_point_C_on_z_axis_l28_28597


namespace arrangement_count_correct_l28_28567

def num_arrangements_exactly_two_females_next_to_each_other (males : ℕ) (females : ℕ) : ℕ :=
  if males = 4 ∧ females = 3 then 3600 else 0

theorem arrangement_count_correct :
  num_arrangements_exactly_two_females_next_to_each_other 4 3 = 3600 :=
by
  sorry

end arrangement_count_correct_l28_28567


namespace vasya_purchase_l28_28172

theorem vasya_purchase : ∃ x y z w : ℕ, x + y + z + w = 15 ∧ 9 * x + 4 * z = 30 ∧ 2 * y + z = 9 ∧ w = 7 :=
by
  sorry

end vasya_purchase_l28_28172


namespace track_width_l28_28797

theorem track_width (r1 r2 : ℝ) (h : 2 * Real.pi * r1 - 2 * Real.pi * r2 = 20 * Real.pi) : r1 - r2 = 10 := by
  sorry

end track_width_l28_28797


namespace mean_square_sum_l28_28420

theorem mean_square_sum (x y z : ℝ) 
  (h1 : x + y + z = 27)
  (h2 : x * y * z = 216)
  (h3 : x * y + y * z + z * x = 162) : 
  x^2 + y^2 + z^2 = 405 :=
by
  sorry

end mean_square_sum_l28_28420


namespace ratio_of_lengths_l28_28565

theorem ratio_of_lengths (total_length short_length : ℕ)
  (h1 : total_length = 35)
  (h2 : short_length = 10) :
  short_length / (total_length - short_length) = 2 / 5 := by
  -- Proof skipped
  sorry

end ratio_of_lengths_l28_28565


namespace increase_by_thirteen_possible_l28_28331

-- Define the main condition which states the reduction of the original product
def product_increase_by_thirteen (a : Fin 7 → ℕ) : Prop :=
  let P := (List.range 7).map (fun i => a ⟨i, sorry⟩) |>.prod
  let Q := (List.range 7).map (fun i => a ⟨i, sorry⟩ - 3) |>.prod
  Q = 13 * P

-- State the theorem to be proved
theorem increase_by_thirteen_possible : ∃ (a : Fin 7 → ℕ), product_increase_by_thirteen a :=
sorry

end increase_by_thirteen_possible_l28_28331


namespace primes_between_2_and_100_l28_28940

open Nat

theorem primes_between_2_and_100 :
  { p : ℕ | 2 ≤ p ∧ p ≤ 100 ∧ Nat.Prime p } = 
  {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97} :=
by
  sorry

end primes_between_2_and_100_l28_28940


namespace dan_picked_more_apples_l28_28433

-- Define the number of apples picked by Benny and Dan
def apples_picked_by_benny := 2
def apples_picked_by_dan := 9

-- Lean statement to prove the given condition
theorem dan_picked_more_apples :
  apples_picked_by_dan - apples_picked_by_benny = 7 := 
sorry

end dan_picked_more_apples_l28_28433


namespace find_n_l28_28680

def P_X_eq_2 (n : ℕ) : Prop :=
  (3 * n) / ((n + 3) * (n + 2)) = (7 : ℚ) / 30

theorem find_n (n : ℕ) (h : P_X_eq_2 n) : n = 7 :=
by sorry

end find_n_l28_28680


namespace tyler_eggs_in_fridge_l28_28594

def recipe_eggs_for_four : Nat := 2
def people_multiplier : Nat := 2
def eggs_needed : Nat := recipe_eggs_for_four * people_multiplier
def eggs_to_buy : Nat := 1
def eggs_in_fridge : Nat := eggs_needed - eggs_to_buy

theorem tyler_eggs_in_fridge : eggs_in_fridge = 3 := by
  sorry

end tyler_eggs_in_fridge_l28_28594


namespace positive_difference_of_two_numbers_l28_28376

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l28_28376


namespace quadrilateral_area_l28_28867

theorem quadrilateral_area (a b : ℤ) (h1 : a > b) (h2 : b > 0) 
  (h3 : 2 * |a - b| * |a + b| = 32) : a + b = 8 :=
by
  sorry

end quadrilateral_area_l28_28867


namespace M_is_even_l28_28442

def sum_of_digits (n : ℕ) : ℕ := -- Define the digit sum function
  sorry

theorem M_is_even (M : ℕ) (h1 : sum_of_digits M = 100) (h2 : sum_of_digits (5 * M) = 50) : 
  M % 2 = 0 :=
sorry

end M_is_even_l28_28442


namespace ellipse_foci_x_axis_l28_28189

theorem ellipse_foci_x_axis (k : ℝ) : 
  (0 < k ∧ k < 2) ↔ (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ∧ a > b) := 
sorry

end ellipse_foci_x_axis_l28_28189


namespace maximum_t_l28_28056

theorem maximum_t {a b t : ℝ} (ha : 0 < a) (hb : a < b) (ht : b < t)
  (h_condition : b * Real.log a < a * Real.log b) : t ≤ Real.exp 1 :=
sorry

end maximum_t_l28_28056


namespace log_a_less_than_neg_b_minus_one_l28_28312

variable {x : ℝ} (a b : ℝ) (f : ℝ → ℝ)

theorem log_a_less_than_neg_b_minus_one
  (h1 : 0 < a)
  (h2 : ∀ x > 0, f x ≥ f 3)
  (h3 : ∀ x > 0, f x = -3 * Real.log x + a * x^2 + b * x) :
  Real.log a < -b - 1 :=
  sorry

end log_a_less_than_neg_b_minus_one_l28_28312


namespace curve_not_parabola_l28_28768

theorem curve_not_parabola (k : ℝ) : ¬ ∃ a b c t : ℝ, a * t^2 + b * t + c = x^2 + k * y^2 - 1 := sorry

end curve_not_parabola_l28_28768


namespace find_A_l28_28363

def clubsuit (A B : ℝ) : ℝ := 4 * A - 3 * B + 7

theorem find_A (A : ℝ) : clubsuit A 6 = 31 → A = 10.5 :=
by
  intro h
  sorry

end find_A_l28_28363


namespace monthly_salary_is_correct_l28_28860

noncomputable def man's_salary : ℝ :=
  let S : ℝ := 6500
  S

theorem monthly_salary_is_correct (S : ℝ) (h1 : S * 0.20 = S * 0.20) (h2 : S * 0.80 * 1.20 + 260 = S):
  S = man's_salary :=
by sorry

end monthly_salary_is_correct_l28_28860


namespace smartphones_discount_l28_28646

theorem smartphones_discount
  (discount : ℝ)
  (cost_per_iphone : ℝ)
  (total_saving : ℝ)
  (num_people : ℕ)
  (num_iphones : ℕ)
  (total_cost : ℝ)
  (required_num : ℕ) :
  discount = 0.05 →
  cost_per_iphone = 600 →
  total_saving = 90 →
  num_people = 3 →
  num_iphones = 3 →
  total_cost = num_iphones * cost_per_iphone →
  required_num = num_iphones →
  required_num * cost_per_iphone * discount = total_saving →
  required_num = 3 :=
by
  intros
  sorry

end smartphones_discount_l28_28646


namespace negation_of_proposition_l28_28063

theorem negation_of_proposition (x : ℝ) : ¬(∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
sorry

end negation_of_proposition_l28_28063


namespace sqrt_2023_irrational_l28_28479

theorem sqrt_2023_irrational : ¬ ∃ (r : ℚ), r^2 = 2023 := by
  sorry

end sqrt_2023_irrational_l28_28479


namespace mike_total_rose_bushes_l28_28510

-- Definitions based on the conditions
def costPerRoseBush : ℕ := 75
def costPerTigerToothAloe : ℕ := 100
def numberOfRoseBushesForFriend : ℕ := 2
def totalExpenseByMike : ℕ := 500
def numberOfTigerToothAloe : ℕ := 2

-- The total number of rose bushes Mike bought
noncomputable def totalNumberOfRoseBushes : ℕ :=
  let totalSpentOnAloes := numberOfTigerToothAloe * costPerTigerToothAloe
  let amountSpentOnRoseBushes := totalExpenseByMike - totalSpentOnAloes
  let numberOfRoseBushesForMike := amountSpentOnRoseBushes / costPerRoseBush
  numberOfRoseBushesForMike + numberOfRoseBushesForFriend

-- The theorem to prove
theorem mike_total_rose_bushes : totalNumberOfRoseBushes = 6 :=
  by
    sorry

end mike_total_rose_bushes_l28_28510


namespace find_function_l28_28983

theorem find_function (a : ℝ) (f : ℝ → ℝ) (h : ∀ x y, (f x * f y - f (x * y)) / 4 = 2 * x + 2 * y + a) : a = -3 ∧ ∀ x, f x = x + 1 :=
by
  sorry

end find_function_l28_28983


namespace max_statements_true_l28_28850

theorem max_statements_true : ∃ x : ℝ, 
  (0 < x^2 ∧ x^2 < 1 ∨ x^2 > 1) ∧ 
  (-1 < x ∧ x < 0 ∨ 0 < x ∧ x < 1) ∧ 
  (0 < (x - x^3) ∧ (x - x^3) < 1) :=
  sorry

end max_statements_true_l28_28850


namespace find_cd_l28_28109

noncomputable def g (c d : ℝ) (x : ℝ) : ℝ := c * x^3 - 8 * x^2 + d * x - 7

theorem find_cd (c d : ℝ) 
  (h1 : g c d 2 = -7) 
  (h2 : g c d (-1) = -25) : 
  (c, d) = (2, 8) := 
by
  sorry

end find_cd_l28_28109


namespace conic_section_is_ellipse_l28_28014

theorem conic_section_is_ellipse :
  (∃ (x y : ℝ), 3 * x^2 + y^2 - 12 * x - 4 * y + 36 = 0) ∧
  ∀ (x y : ℝ), 3 * x^2 + y^2 - 12 * x - 4 * y + 36 = 0 →
    ((x - 2)^2 / (20 / 3) + (y - 2)^2 / 20 = 1) :=
sorry

end conic_section_is_ellipse_l28_28014


namespace pipe_tank_fill_time_l28_28222

/-- 
Given:
1. Pipe A fills the tank in 2 hours.
2. The leak empties the tank in 4 hours.
Prove: 
The tank is filled in 4 hours when both Pipe A and the leak are working together.
 -/
theorem pipe_tank_fill_time :
  let A := 1 / 2 -- rate at which Pipe A fills the tank (tank per hour)
  let L := 1 / 4 -- rate at which the leak empties the tank (tank per hour)
  let net_rate := A - L -- net rate of filling the tank
  net_rate > 0 → (1 / net_rate) = 4 := 
by
  intros
  sorry

end pipe_tank_fill_time_l28_28222


namespace left_handed_jazz_lovers_count_l28_28931

noncomputable def club_members := 30
noncomputable def left_handed := 11
noncomputable def like_jazz := 20
noncomputable def right_handed_dislike_jazz := 4

theorem left_handed_jazz_lovers_count : 
  ∃ x, x + (left_handed - x) + (like_jazz - x) + right_handed_dislike_jazz = club_members ∧ x = 5 :=
by
  sorry

end left_handed_jazz_lovers_count_l28_28931


namespace cost_per_meter_l28_28589

-- Definitions of the conditions
def length_of_plot : ℕ := 63
def breadth_of_plot : ℕ := length_of_plot - 26
def perimeter_of_plot := 2 * length_of_plot + 2 * breadth_of_plot
def total_cost : ℕ := 5300

-- Statement to prove
theorem cost_per_meter : (total_cost : ℚ) / perimeter_of_plot = 26.5 :=
by sorry

end cost_per_meter_l28_28589


namespace increase_in_green_chameleons_is_11_l28_28813

-- Definitions to encode the problem conditions
def num_green_chameleons_increase : Nat :=
  let sunny_days := 18
  let cloudy_days := 12
  let deltaB := 5
  let delta_A_minus_B := sunny_days - cloudy_days
  delta_A_minus_B + deltaB

-- Assertion to prove
theorem increase_in_green_chameleons_is_11 : num_green_chameleons_increase = 11 := by 
  sorry

end increase_in_green_chameleons_is_11_l28_28813


namespace water_percentage_l28_28482

theorem water_percentage (P : ℕ) : 
  let initial_volume := 300
  let final_volume := initial_volume + 100
  let desired_water_percentage := 70
  let water_added := 100
  let final_water_amount := desired_water_percentage * final_volume / 100
  let current_water_amount := P * initial_volume / 100

  current_water_amount + water_added = final_water_amount → 
  P = 60 :=
by sorry

end water_percentage_l28_28482


namespace mandarin_ducks_total_l28_28858

theorem mandarin_ducks_total : (3 * 2) = 6 := by
  sorry

end mandarin_ducks_total_l28_28858


namespace original_number_eq_0_000032_l28_28355

theorem original_number_eq_0_000032 (x : ℝ) (hx : 0 < x) 
  (h : 10^8 * x = 8 * (1 / x)) : x = 0.000032 :=
sorry

end original_number_eq_0_000032_l28_28355


namespace boat_width_l28_28003

-- Definitions: river width, number of boats, and space between/banks
def river_width : ℝ := 42
def num_boats : ℕ := 8
def space_between : ℝ := 2

-- Prove the width of each boat given the conditions
theorem boat_width : 
  ∃ w : ℝ, 
    8 * w + 7 * space_between + 2 * space_between = river_width ∧
    w = 3 :=
by
  sorry

end boat_width_l28_28003


namespace complement_is_empty_l28_28096

def U : Set ℕ := {1, 3}
def A : Set ℕ := {1, 3}

theorem complement_is_empty : (U \ A) = ∅ := 
by 
  sorry

end complement_is_empty_l28_28096


namespace cars_in_garage_l28_28771

theorem cars_in_garage (c : ℕ) 
  (bicycles : ℕ := 20) 
  (motorcycles : ℕ := 5) 
  (total_wheels : ℕ := 90) 
  (bicycle_wheels : ℕ := 2 * bicycles)
  (motorcycle_wheels : ℕ := 2 * motorcycles)
  (car_wheels : ℕ := 4 * c) 
  (eq : bicycle_wheels + car_wheels + motorcycle_wheels = total_wheels) : 
  c = 10 := 
by 
  sorry

end cars_in_garage_l28_28771


namespace kevin_birth_year_l28_28886

theorem kevin_birth_year (year_first_amc: ℕ) (annual: ∀ n, year_first_amc + n = year_first_amc + n) (age_tenth_amc: ℕ) (year_tenth_amc: ℕ) (year_kevin_took_amc: ℕ) 
  (h_first_amc: year_first_amc = 1988) (h_age_tenth_amc: age_tenth_amc = 13) (h_tenth_amc: year_tenth_amc = year_first_amc + 9) (h_kevin_took_amc: year_kevin_took_amc = year_tenth_amc) :
  year_kevin_took_amc - age_tenth_amc = 1984 :=
by
  sorry

end kevin_birth_year_l28_28886


namespace difference_of_squares_l28_28641

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 18) (h2 : x^2 + y^2 = 180) : |x^2 - y^2| = 108 :=
  sorry

end difference_of_squares_l28_28641


namespace hyperbola_satisfies_conditions_l28_28259

-- Define the equations of the hyperbolas as predicates
def hyperbola_A (x y : ℝ) : Prop := x^2 - (y^2 / 4) = 1
def hyperbola_B (x y : ℝ) : Prop := (x^2 / 4) - y^2 = 1
def hyperbola_C (x y : ℝ) : Prop := (y^2 / 4) - x^2 = 1
def hyperbola_D (x y : ℝ) : Prop := y^2 - (x^2 / 4) = 1

-- Define the conditions on foci and asymptotes
def foci_on_y_axis (h : (ℝ → ℝ → Prop)) : Prop := 
  h = hyperbola_C ∨ h = hyperbola_D

def has_asymptotes (h : (ℝ → ℝ → Prop)) : Prop :=
  ∀ x y, h x y → (y = (1/2) * x ∨ y = -(1/2) * x)

-- The proof statement
theorem hyperbola_satisfies_conditions :
  foci_on_y_axis hyperbola_D ∧ has_asymptotes hyperbola_D ∧ 
    (¬ (foci_on_y_axis hyperbola_A ∧ has_asymptotes hyperbola_A)) ∧ 
    (¬ (foci_on_y_axis hyperbola_B ∧ has_asymptotes hyperbola_B)) ∧ 
    (¬ (foci_on_y_axis hyperbola_C ∧ has_asymptotes hyperbola_C)) := 
by
  sorry

end hyperbola_satisfies_conditions_l28_28259


namespace percentage_cross_pollinated_l28_28472

-- Definitions and known conditions:
variables (F C T : ℕ)
variables (h1 : F + C = 221)
variables (h2 : F = 3 * T / 4)
variables (h3 : T = F + 39 + C)

-- Theorem statement for the percentage of cross-pollinated trees
theorem percentage_cross_pollinated : ((C : ℚ) / T) * 100 = 10 :=
by sorry

end percentage_cross_pollinated_l28_28472


namespace find_constants_for_matrix_condition_l28_28190

noncomputable section

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, 2, 3], ![0, 1, 2], ![1, 0, 1]]

def I : Matrix (Fin 3) (Fin 3) ℝ :=
  1

theorem find_constants_for_matrix_condition :
  ∃ p q r : ℝ, B^3 + p • B^2 + q • B + r • I = 0 :=
by
  use -5, 3, -6
  sorry

end find_constants_for_matrix_condition_l28_28190


namespace find_pairs_s_t_l28_28010

theorem find_pairs_s_t (n : ℤ) (hn : n > 1) : 
  ∃ s t : ℤ, (
    (∀ x : ℝ, x ^ n + s * x = 2007 ∧ x ^ n + t * x = 2008 → 
     (s, t) = (2006, 2007) ∨ (s, t) = (-2008, -2009) ∨ (s, t) = (-2006, -2007))
  ) :=
sorry

end find_pairs_s_t_l28_28010


namespace find_selling_price_l28_28904

variable (SP CP : ℝ)

def original_selling_price (SP CP : ℝ) : Prop :=
  0.9 * SP = CP + 0.08 * CP

theorem find_selling_price (h1 : CP = 17500)
  (h2 : original_selling_price SP CP) : SP = 21000 :=
by
  sorry

end find_selling_price_l28_28904


namespace orchestra_french_horn_players_l28_28875

open Nat

theorem orchestra_french_horn_players :
  ∃ (french_horn_players : ℕ), 
  french_horn_players = 1 ∧
  1 + 6 + 5 + 7 + 1 + french_horn_players = 21 :=
by
  sorry

end orchestra_french_horn_players_l28_28875


namespace spaceship_distance_l28_28667

-- Define the distance variables and conditions
variables (D : ℝ) -- Distance from Earth to Planet X
variable (T : ℝ) -- Total distance traveled by the spaceship

-- Conditions
variables (hx : T = 0.7) -- Total distance traveled is 0.7 light-years
variables (hy : D + 0.1 + 0.1 = T) -- Sum of distances along the path

-- Theorem statement to prove the distance from Earth to Planet X
theorem spaceship_distance (h1 : T = 0.7) (h2 : D + 0.1 + 0.1 = T) : D = 0.5 :=
by
  -- Proof steps would go here
  sorry

end spaceship_distance_l28_28667


namespace toys_sold_week2_l28_28852

-- Define the given conditions
def original_stock := 83
def toys_sold_week1 := 38
def toys_left := 19

-- Define the statement we want to prove
theorem toys_sold_week2 : (original_stock - toys_left) - toys_sold_week1 = 26 :=
by
  sorry

end toys_sold_week2_l28_28852


namespace arithmetic_sequence_sum_equals_product_l28_28045

theorem arithmetic_sequence_sum_equals_product :
  ∃ (a_1 a_2 a_3 : ℤ), (a_2 = a_1 + d) ∧ (a_3 = a_1 + 2 * d) ∧ 
    a_1 ≠ 0 ∧ (a_1 + a_2 + a_3 = a_1 * a_2 * a_3) ∧ 
    (∃ d x : ℤ, x ≠ 0 ∧ d ≠ 0 ∧ 
    ((x = 1 ∧ d = 1) ∨ (x = -3 ∧ d = 1) ∨ (x = 3 ∧ d = -1) ∨ (x = -1 ∧ d = -1))) :=
sorry

end arithmetic_sequence_sum_equals_product_l28_28045


namespace regular_polygon_sides_l28_28029

theorem regular_polygon_sides 
  (A B C : ℝ)
  (h₁ : A + B + C = 180)
  (h₂ : B = 3 * A)
  (h₃ : C = 6 * A) :
  ∃ (n : ℕ), n = 5 :=
by
  sorry

end regular_polygon_sides_l28_28029


namespace total_limes_picked_l28_28516

-- Define the number of limes each person picked
def fred_limes : Nat := 36
def alyssa_limes : Nat := 32
def nancy_limes : Nat := 35
def david_limes : Nat := 42
def eileen_limes : Nat := 50

-- Formal statement of the problem
theorem total_limes_picked : 
  fred_limes + alyssa_limes + nancy_limes + david_limes + eileen_limes = 195 := by
  -- Add proof
  sorry

end total_limes_picked_l28_28516


namespace max_length_third_side_l28_28568

open Real

theorem max_length_third_side (A B C : ℝ) (a b c : ℝ) 
  (h1 : cos (2 * A) + cos (2 * B) + cos (2 * C) = 1)
  (h2 : a = 9) 
  (h3 : b = 12)
  (h4 : a^2 + b^2 = c^2) : 
  c = 15 := 
sorry

end max_length_third_side_l28_28568


namespace rearrange_pairs_l28_28841

theorem rearrange_pairs {a b : ℕ} (hb: b = (2 / 3 : ℚ) * a) (boys_way_museum boys_way_back : ℕ) :
  boys_way_museum = 3 * a ∧ boys_way_back = 4 * b → 
  ∃ c : ℕ, boys_way_museum = 7 * c ∧ b = c := sorry

end rearrange_pairs_l28_28841


namespace g_f_of_3_l28_28297

def f (x : ℝ) : ℝ := x^3 - 4
def g (x : ℝ) : ℝ := 3 * x^2 + 5 * x + 2

theorem g_f_of_3 : g (f 3) = 1704 := by
  sorry

end g_f_of_3_l28_28297


namespace AM_GM_inequality_l28_28178

theorem AM_GM_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_not_all_eq : x ≠ y ∨ y ≠ z ∨ z ≠ x) :
  (x + y) * (y + z) * (z + x) > 8 * x * y * z :=
by
  sorry

end AM_GM_inequality_l28_28178


namespace tank_capacity_ratio_l28_28247

-- Definitions from the problem conditions
def tank1_filled : ℝ := 300
def tank2_filled : ℝ := 450
def tank2_percentage_filled : ℝ := 0.45
def additional_needed : ℝ := 1250

-- Theorem statement
theorem tank_capacity_ratio (C1 C2 : ℝ) 
  (h1 : tank1_filled + tank2_filled + additional_needed = C1 + C2)
  (h2 : tank2_filled = tank2_percentage_filled * C2) : 
  C1 / C2 = 2 :=
by
  sorry

end tank_capacity_ratio_l28_28247


namespace number_of_purchasing_schemes_l28_28378

def total_cost (a : Nat) (b : Nat) : Nat := 8 * a + 10 * b

def valid_schemes : List (Nat × Nat) :=
  [(4, 4), (4, 5), (4, 6), (4, 7),
   (5, 4), (5, 5), (5, 6),
   (6, 4), (6, 5),
   (7, 4)]

theorem number_of_purchasing_schemes : valid_schemes.length = 9 := sorry

end number_of_purchasing_schemes_l28_28378


namespace smallest_n_for_divisibility_l28_28288

theorem smallest_n_for_divisibility (n : ℕ) (h1 : 24 ∣ n^2) (h2 : 1080 ∣ n^3) : n = 120 :=
sorry

end smallest_n_for_divisibility_l28_28288


namespace sufficient_but_not_necessary_l28_28081

theorem sufficient_but_not_necessary (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > 0) :
  (a > b ∧ b > 0 ∧ c > 0) → (a / (a + c) > b / (b + c)) :=
by
  intros
  sorry

end sufficient_but_not_necessary_l28_28081


namespace problem_solution_l28_28671

noncomputable def vector_magnitudes_and_angle 
  (a b : ℝ) (angle_ab : ℝ) (norma normb : ℝ) (k : ℝ) : Prop :=
(a = 4 ∧ b = 8 ∧ angle_ab = 2 * Real.pi / 3 ∧ norma = 4 ∧ normb = 8) →
((norma^2 + normb^2 + 2 * norma * normb * Real.cos angle_ab = 48) ∧
  (16 * k - 32 * k + 16 - 128 = 0))

theorem problem_solution : vector_magnitudes_and_angle 4 8 (2 * Real.pi / 3) 4 8 (-7) := 
by 
  sorry

end problem_solution_l28_28671


namespace path_inequality_l28_28427

theorem path_inequality
  (f : ℕ → ℕ → ℝ) :
  f 1 6 * f 2 5 * f 3 4 + f 1 5 * f 2 4 * f 3 6 + f 1 4 * f 2 6 * f 3 5 ≥
  f 1 6 * f 2 4 * f 3 5 + f 1 5 * f 2 6 * f 3 4 + f 1 4 * f 2 5 * f 3 6 :=
sorry

end path_inequality_l28_28427


namespace abs_lt_two_nec_but_not_suff_l28_28123

theorem abs_lt_two_nec_but_not_suff (x : ℝ) :
  (|x - 1| < 2) → (0 < x ∧ x < 3) ∧ ¬((0 < x ∧ x < 3) → (|x - 1| < 2)) := sorry

end abs_lt_two_nec_but_not_suff_l28_28123


namespace focus_of_parabola_l28_28741

-- Define the equation of the given parabola
def given_parabola (x y : ℝ) : Prop := y = - (1 / 8) * x^2

-- Define the condition for the focus of the parabola
def is_focus (focus : ℝ × ℝ) : Prop := focus = (0, -2)

-- State the theorem
theorem focus_of_parabola : ∃ (focus : ℝ × ℝ), given_parabola x y → is_focus focus :=
by
  -- Placeholder proof
  sorry

end focus_of_parabola_l28_28741


namespace smallest_value_of_N_l28_28480

theorem smallest_value_of_N (l m n : ℕ) (N : ℕ) (h1 : (l-1) * (m-1) * (n-1) = 270) (h2 : N = l * m * n): 
  N = 420 :=
sorry

end smallest_value_of_N_l28_28480


namespace sin_theta_fourth_quadrant_l28_28672

-- Given conditions
variables {θ : ℝ} (h1 : Real.cos θ = 1 / 3) (h2 : 3 * pi / 2 < θ ∧ θ < 2 * pi)

-- Proof statement
theorem sin_theta_fourth_quadrant : Real.sin θ = -2 * Real.sqrt 2 / 3 :=
sorry

end sin_theta_fourth_quadrant_l28_28672


namespace profit_function_simplified_maximize_profit_l28_28520

-- Define the given conditions
def cost_per_product : ℝ := 3
def management_fee_per_product : ℝ := 3
def annual_sales_volume (x : ℝ) : ℝ := (12 - x) ^ 2

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - (cost_per_product + management_fee_per_product)) * annual_sales_volume x

-- Define the bounds for x
def x_bounds (x : ℝ) : Prop := 9 ≤ x ∧ x ≤ 11

-- Prove the profit function in simplified form
theorem profit_function_simplified (x : ℝ) (h : x_bounds x) :
    profit x = x ^ 3 - 30 * x ^ 2 + 288 * x - 864 :=
by
  sorry

-- Prove the maximum profit and the corresponding x value
theorem maximize_profit (x : ℝ) (h : x_bounds x) :
    (∀ y, (∃ x', x_bounds x' ∧ y = profit x') → y ≤ 27) ∧ profit 9 = 27 :=
by
  sorry

end profit_function_simplified_maximize_profit_l28_28520


namespace abs_eq_of_unique_solution_l28_28944

theorem abs_eq_of_unique_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
    (unique_solution : ∃! x : ℝ, a * (x - a) ^ 2 + b * (x - b) ^ 2 = 0) :
    |a| = |b| :=
sorry

end abs_eq_of_unique_solution_l28_28944


namespace person_speed_l28_28352

noncomputable def distance_meters : ℝ := 1080
noncomputable def time_minutes : ℝ := 14
noncomputable def distance_kilometers : ℝ := distance_meters / 1000
noncomputable def time_hours : ℝ := time_minutes / 60
noncomputable def speed_km_per_hour : ℝ := distance_kilometers / time_hours

theorem person_speed :
  abs (speed_km_per_hour - 4.63) < 0.01 :=
by
  -- conditions extracted
  let distance_in_km := distance_meters / 1000
  let time_in_hours := time_minutes / 60
  let speed := distance_in_km / time_in_hours
  -- We expect speed to be approximately 4.63
  sorry 

end person_speed_l28_28352


namespace students_failed_in_english_l28_28495

variable (H : ℝ) (E : ℝ) (B : ℝ) (P : ℝ)

theorem students_failed_in_english
  (hH : H = 34 / 100) 
  (hB : B = 22 / 100)
  (hP : P = 44 / 100)
  (hIE : (1 - P) = H + E - B) :
  E = 44 / 100 := 
sorry

end students_failed_in_english_l28_28495


namespace solve_for_s_l28_28581

theorem solve_for_s (s t : ℚ) (h1 : 7 * s + 8 * t = 150) (h2 : s = 2 * t + 3) : s = 162 / 11 := 
by
  sorry

end solve_for_s_l28_28581


namespace sine_triangle_l28_28552

theorem sine_triangle (a b c : ℝ) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_perimeter : a + b + c ≤ 2 * Real.pi)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (ha_pi : a < Real.pi) (hb_pi : b < Real.pi) (hc_pi : c < Real.pi):
  ∃ (x y z : ℝ), x = Real.sin a ∧ y = Real.sin b ∧ z = Real.sin c ∧ x + y > z ∧ y + z > x ∧ x + z > y :=
by
  sorry

end sine_triangle_l28_28552


namespace parabola_vertex_l28_28631

theorem parabola_vertex (x y : ℝ) : ∀ x y, (y^2 + 8 * y + 2 * x + 11 = 0) → (x = 5 / 2 ∧ y = -4) :=
by
  intro x y h
  sorry

end parabola_vertex_l28_28631


namespace kaleb_lives_left_l28_28013

theorem kaleb_lives_left (initial_lives : ℕ) (lives_lost : ℕ) (remaining_lives : ℕ) :
  initial_lives = 98 → lives_lost = 25 → remaining_lives = initial_lives - lives_lost → remaining_lives = 73 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end kaleb_lives_left_l28_28013


namespace miller_rabin_probability_at_least_half_l28_28845

theorem miller_rabin_probability_at_least_half
  {n : ℕ} (hcomp : ¬Nat.Prime n) (s d : ℕ) (hd_odd : d % 2 = 1) (h_decomp : n - 1 = 2^s * d)
  (a : ℤ) (ha_range : 2 ≤ a ∧ a ≤ n - 2) :
  ∃ P : ℝ, P ≥ 1 / 2 ∧ ∀ a, (2 ≤ a ∧ a ≤ n - 2) → ¬(a^(d * 2^s) % n = 1)
  :=
sorry

end miller_rabin_probability_at_least_half_l28_28845


namespace inequality_for_positive_real_numbers_l28_28337

theorem inequality_for_positive_real_numbers (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a^4 + b^4 + c^4) / (a + b + c) ≥ a * b * c :=
  sorry

end inequality_for_positive_real_numbers_l28_28337


namespace find_triplets_l28_28290

theorem find_triplets (a m n : ℕ) (ha : 0 < a) (hm : 0 < m) (hn : 0 < n) :
  (a^m + 1 ∣ (a + 1)^n) ↔ ((a = 1) ∨ (a = 2 ∧ m = 3 ∧ n ≥ 2)) :=
by
  sorry

end find_triplets_l28_28290


namespace rotameter_gas_phase_measurement_l28_28085

theorem rotameter_gas_phase_measurement
  (liquid_inch_per_lpm : ℝ) (liquid_liter_per_minute : ℝ) (gas_inch_movement_ratio : ℝ) (gas_liter_passed : ℝ) :
  liquid_inch_per_lpm = 2.5 → liquid_liter_per_minute = 60 → gas_inch_movement_ratio = 0.5 → gas_liter_passed = 192 →
  (gas_inch_movement_ratio * liquid_inch_per_lpm * gas_liter_passed / liquid_liter_per_minute) = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end rotameter_gas_phase_measurement_l28_28085


namespace sum_of_possible_values_for_a_l28_28946

-- Define the conditions
variables (a b c d : ℤ)
variables (h1 : a > b) (h2 : b > c) (h3 : c > d)
variables (h4 : a + b + c + d = 52)
variables (differences : finset ℤ)

-- Hypotheses about the pairwise differences
variable (h_diff : differences = {2, 3, 5, 6, 8, 11})
variable (h_ad : a - d = 11)

-- The pairs of differences adding up to 11
variable (h_pairs1 : a - b + b - d = 11)
variable (h_pairs2 : a - c + c - d = 11)

-- The theorem to be proved
theorem sum_of_possible_values_for_a : a = 19 :=
by
-- Implemented variables and conditions correctly, and the proof is outlined.
sorry

end sum_of_possible_values_for_a_l28_28946


namespace sum_first_12_terms_l28_28832

def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

def geometric_mean {α : Type} [Field α] (a b c : α) : Prop :=
b^2 = a * c

def sum_arithmetic_sequence (a : ℕ → ℚ) (n : ℕ) : ℚ :=
n * (a 1 + a n) / 2

theorem sum_first_12_terms 
  (a : ℕ → ℚ)
  (d : ℚ)
  (h1 : arithmetic_sequence a 1)
  (h2 : geometric_mean (a 3) (a 6) (a 11)) :
  sum_arithmetic_sequence a 12 = 96 :=
sorry

end sum_first_12_terms_l28_28832


namespace at_least_one_true_l28_28082

theorem at_least_one_true (p q : Prop) (h : ¬(p ∨ q) = false) : p ∨ q :=
by
  sorry

end at_least_one_true_l28_28082


namespace factor_expression_l28_28970

theorem factor_expression (x : ℝ) : 5 * x * (x - 2) + 9 * (x - 2) = (x - 2) * (5 * x + 9) := 
by 
sorry

end factor_expression_l28_28970


namespace minimum_value_of_function_l28_28942

theorem minimum_value_of_function (x : ℝ) (h : x > 1) : 
  (x + (1 / x) + (16 * x) / (x^2 + 1)) ≥ 8 :=
sorry

end minimum_value_of_function_l28_28942


namespace geometric_sequence_formula_and_sum_l28_28419

theorem geometric_sequence_formula_and_sum (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) 
  (h1 : a 1 = 2) 
  (h2 : ∀ n, a (n+1) = 2 * a n) 
  (h_arith : a 1 = 2 ∧ 2 * (a 3 + 1) = a 1 + a 4)
  (h_b : ∀ n, b n = Nat.log2 (a n)) :
  (∀ n, a n = 2 ^ n) ∧ (S n = (n * (n + 1)) / 2) := 
by 
  sorry

end geometric_sequence_formula_and_sum_l28_28419


namespace government_subsidy_per_hour_l28_28881

-- Given conditions:
def cost_first_employee : ℕ := 20
def cost_second_employee : ℕ := 22
def hours_per_week : ℕ := 40
def weekly_savings : ℕ := 160

-- To prove:
theorem government_subsidy_per_hour (S : ℕ) : S = 2 :=
by
  -- Proof steps go here.
  sorry

end government_subsidy_per_hour_l28_28881


namespace largest_square_in_right_triangle_largest_rectangle_in_right_triangle_l28_28809

theorem largest_square_in_right_triangle (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ s, s = (a * b) / (a + b) := 
sorry

theorem largest_rectangle_in_right_triangle (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ x y, x = a / 2 ∧ y = b / 2 :=
sorry

end largest_square_in_right_triangle_largest_rectangle_in_right_triangle_l28_28809


namespace recycle_cans_l28_28802

theorem recycle_cans (initial_cans : ℕ) (recycle_rate : ℕ) (n1 n2 n3 : ℕ)
  (h1 : initial_cans = 450)
  (h2 : recycle_rate = 5)
  (h3 : n1 = initial_cans / recycle_rate)
  (h4 : n2 = n1 / recycle_rate)
  (h5 : n3 = n2 / recycle_rate)
  (h6 : n3 / recycle_rate = 0) : 
  n1 + n2 + n3 = 111 :=
by
  sorry

end recycle_cans_l28_28802


namespace list_price_of_article_l28_28765

theorem list_price_of_article (P : ℝ) 
  (first_discount second_discount final_price : ℝ)
  (h1 : first_discount = 0.10)
  (h2 : second_discount = 0.08235294117647069)
  (h3 : final_price = 56.16)
  (h4 : P * (1 - first_discount) * (1 - second_discount) = final_price) : P = 68 :=
sorry

end list_price_of_article_l28_28765


namespace trays_from_first_table_l28_28274

-- Definitions based on conditions
def trays_per_trip : ℕ := 4
def trips : ℕ := 3
def trays_from_second_table : ℕ := 2

-- Theorem statement to prove the number of trays picked up from the first table
theorem trays_from_first_table : trays_per_trip * trips - trays_from_second_table = 10 := by
  sorry

end trays_from_first_table_l28_28274


namespace periodic_function_l28_28032

open Real

theorem periodic_function (f : ℝ → ℝ) 
  (h_bound : ∀ x : ℝ, |f x| ≤ 1)
  (h_func_eq : ∀ x : ℝ, f (x + 13 / 42) + f x = f (x + 1 / 6) + f (x + 1 / 7)) : 
  ∀ x : ℝ, f (x + 1) = f x := 
  sorry

end periodic_function_l28_28032


namespace relationship_and_range_max_profit_find_a_l28_28637

noncomputable def functional_relationship (x : ℝ) : ℝ :=
if 40 ≤ x ∧ x ≤ 50 then 5
else if 50 < x ∧ x ≤ 100 then 10 - 0.1 * x
else 0  -- default case to handle x out of range, though ideally this should not occur in the context.

theorem relationship_and_range : 
  ∀ (x : ℝ), (40 ≤ x ∧ x ≤ 100) →
    (functional_relationship x = 
    (if 40 ≤ x ∧ x ≤ 50 then 5 else if 50 < x ∧ x ≤ 100 then 10 - 0.1 * x else 0)) :=
sorry

noncomputable def monthly_profit (x : ℝ) : ℝ :=
(x - 40) * functional_relationship x

theorem max_profit : 
  (∀ x, 40 ≤ x ∧ x ≤ 100 → monthly_profit x ≤ 90) ∧
  (monthly_profit 70 = 90) :=
sorry

noncomputable def donation_profit (x a : ℝ) : ℝ :=
(x - 40 - a) * (10 - 0.1 * x)

theorem find_a (a : ℝ) : 
  (∀ x, x ≤ 70 → donation_profit x a ≤ 78) ∧
  (donation_profit 70 a = 78) → 
  a = 4 :=
sorry

end relationship_and_range_max_profit_find_a_l28_28637


namespace number_of_uncool_parents_l28_28343

variable (total_students cool_dads cool_moms cool_both : ℕ)

theorem number_of_uncool_parents (h1 : total_students = 40)
                                  (h2 : cool_dads = 18)
                                  (h3 : cool_moms = 22)
                                  (h4 : cool_both = 10) :
    total_students - (cool_dads + cool_moms - cool_both) = 10 := by
  sorry

end number_of_uncool_parents_l28_28343


namespace weight_of_b_is_37_l28_28452

variables {a b c : ℝ}

-- Conditions
def average_abc (a b c : ℝ) : Prop := (a + b + c) / 3 = 45
def average_ab (a b : ℝ) : Prop := (a + b) / 2 = 40
def average_bc (b c : ℝ) : Prop := (b + c) / 2 = 46

-- Statement to prove
theorem weight_of_b_is_37 (h1 : average_abc a b c) (h2 : average_ab a b) (h3 : average_bc b c) : b = 37 :=
by {
  sorry
}

end weight_of_b_is_37_l28_28452


namespace income_growth_relation_l28_28201

-- Define all the conditions
def initial_income : ℝ := 1.3
def third_week_income : ℝ := 2
def growth_rate (x : ℝ) : ℝ := (1 + x)^2  -- Compound interest style growth over 2 weeks.

-- Theorem: proving the relationship given the conditions
theorem income_growth_relation (x : ℝ) : initial_income * growth_rate x = third_week_income :=
by
  unfold initial_income third_week_income growth_rate
  sorry  -- Proof not required.

end income_growth_relation_l28_28201


namespace find_remainder_mod_10_l28_28624

def inv_mod_10 (x : ℕ) : ℕ := 
  if x = 1 then 1 
  else if x = 3 then 7 
  else if x = 7 then 3 
  else if x = 9 then 9 
  else 0 -- invalid, not invertible

theorem find_remainder_mod_10 (a b c d : ℕ) 
  (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ d) (hd : d ≠ a) 
  (ha' : a < 10) (hb' : b < 10) (hc' : c < 10) (hd' : d < 10)
  (ha_inv : inv_mod_10 a ≠ 0) (hb_inv : inv_mod_10 b ≠ 0)
  (hc_inv : inv_mod_10 c ≠ 0) (hd_inv : inv_mod_10 d ≠ 0) :
  ((a * b * c + a * b * d + a * c * d + b * c * d) * (inv_mod_10 (a * b * c * d % 10))) % 10 = 0 :=
by
  sorry

end find_remainder_mod_10_l28_28624


namespace bike_sharing_problem_l28_28171

def combinations (n k : ℕ) : ℕ := (Nat.choose n k)

theorem bike_sharing_problem:
  let total_bikes := 10
  let blue_bikes := 4
  let yellow_bikes := 6
  let inspected_bikes := 4
  let way_two_blue := combinations blue_bikes 2 * combinations yellow_bikes 2
  let way_three_blue := combinations blue_bikes 3 * combinations yellow_bikes 1
  let way_four_blue := combinations blue_bikes 4
  way_two_blue + way_three_blue + way_four_blue = 115 :=
by
  sorry

end bike_sharing_problem_l28_28171


namespace evaluate_polynomial_at_minus_two_l28_28217

noncomputable def polynomial (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 - x^2 + 2 * x + 5

theorem evaluate_polynomial_at_minus_two : polynomial (-2) = 5 := by
  sorry

end evaluate_polynomial_at_minus_two_l28_28217


namespace smallest_positive_period_of_f_range_of_a_l28_28227

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)

theorem smallest_positive_period_of_f : (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (T = π) :=
by
  sorry

theorem range_of_a (a : ℝ) : (∀ x, f x ≤ a) → a ≥ Real.sqrt 2 :=
by
  sorry

end smallest_positive_period_of_f_range_of_a_l28_28227


namespace intersect_at_one_point_l28_28477

theorem intersect_at_one_point (a : ℝ) : 
  (a * (4 * 4) + 4 * 4 * 6 = 0) -> a = 2 / (3: ℝ) :=
by sorry

end intersect_at_one_point_l28_28477


namespace geometric_sequence_first_term_l28_28344

theorem geometric_sequence_first_term (a r : ℝ) 
  (h1 : a * r = 18) 
  (h2 : a * r^4 = 1458) : 
  a = 6 := 
by 
  sorry

end geometric_sequence_first_term_l28_28344


namespace determine_true_propositions_l28_28834

def p (x y : ℝ) := x > y → -x < -y
def q (x y : ℝ) := (1/x > 1/y) → x < y

theorem determine_true_propositions (x y : ℝ) :
  (p x y ∨ q x y) ∧ (p x y ∧ ¬ q x y) :=
by
  sorry

end determine_true_propositions_l28_28834


namespace find_N_l28_28574

theorem find_N : ∃ N : ℕ, 36^2 * 72^2 = 12^2 * N^2 ∧ N = 216 :=
by
  sorry

end find_N_l28_28574


namespace sale_in_second_month_l28_28042

-- Define the constants for known sales and average requirement
def sale_first_month : Int := 8435
def sale_third_month : Int := 8855
def sale_fourth_month : Int := 9230
def sale_fifth_month : Int := 8562
def sale_sixth_month : Int := 6991
def average_sale_per_month : Int := 8500
def number_of_months : Int := 6

-- Define the total sales required for six months
def total_sales_required : Int := average_sale_per_month * number_of_months

-- Define the total known sales excluding the second month
def total_known_sales : Int := sale_first_month + sale_third_month + sale_fourth_month + sale_fifth_month + sale_sixth_month

-- The statement to prove: the sale in the second month is 8927
theorem sale_in_second_month : 
  total_sales_required - total_known_sales = 8927 := 
by
  sorry

end sale_in_second_month_l28_28042


namespace Billy_has_10_fish_l28_28981

def Billy_has_fish (Bobby Sarah Tony Billy : ℕ) : Prop :=
  Bobby = 2 * Sarah ∧
  Sarah = Tony + 5 ∧
  Tony = 3 * Billy ∧
  Bobby + Sarah + Tony + Billy = 145

theorem Billy_has_10_fish : ∃ (Billy : ℕ), Billy_has_fish (2 * (3 * Billy + 5)) (3 * Billy + 5) (3 * Billy) Billy ∧ Billy = 10 :=
by
  sorry

end Billy_has_10_fish_l28_28981


namespace contrapositive_l28_28238

variable (Line Circle : Type) (distance : Line → Circle → ℝ) (radius : Circle → ℝ)
variable (is_tangent : Line → Circle → Prop)

-- Original proposition in Lean notation:
def original_proposition (l : Line) (c : Circle) : Prop :=
  distance l c ≠ radius c → ¬ is_tangent l c

-- Contrapositive of the original proposition:
theorem contrapositive (l : Line) (c : Circle) : Prop :=
  is_tangent l c → distance l c = radius c

end contrapositive_l28_28238


namespace speed_of_stream_l28_28232

theorem speed_of_stream (v_d v_u : ℝ) (h_d : v_d = 13) (h_u : v_u = 8) :
  (v_d - v_u) / 2 = 2.5 :=
by
  -- Insert proof steps here
  sorry

end speed_of_stream_l28_28232


namespace wrench_force_inversely_proportional_l28_28057

theorem wrench_force_inversely_proportional (F L : ℝ) (F1 F2 L1 L2 : ℝ) 
    (h1 : F1 = 375) 
    (h2 : L1 = 9) 
    (h3 : L2 = 15) 
    (h4 : ∀ L : ℝ, F * L = F1 * L1) : F2 = 225 :=
by
  sorry

end wrench_force_inversely_proportional_l28_28057


namespace min_value_expr_l28_28784

-- Define the given expression
def given_expr (x : ℝ) : ℝ :=
  (15 - x) * (8 - x) * (15 + x) * (8 + x) + 200

-- Define the minimum value we need to prove
def min_value : ℝ :=
  -6290.25

-- The statement of the theorem
theorem min_value_expr :
  ∃ x : ℝ, ∀ y : ℝ, given_expr y ≥ min_value := by
  sorry

end min_value_expr_l28_28784


namespace simplify_polynomial_l28_28853

theorem simplify_polynomial :
  (2 * x * (4 * x ^ 3 - 3 * x + 1) - 4 * (2 * x ^ 3 - x ^ 2 + 3 * x - 5)) =
  8 * x ^ 4 - 8 * x ^ 3 - 2 * x ^ 2 - 10 * x + 20 :=
by
  sorry

end simplify_polynomial_l28_28853


namespace probability_of_two_red_balls_l28_28413

-- Definitions of quantities
def total_balls := 11
def red_balls := 3
def blue_balls := 4 
def green_balls := 4 
def balls_picked := 2

-- Theorem statement
theorem probability_of_two_red_balls :
  ((red_balls * (red_balls - 1)) / (total_balls * (total_balls - 1) / balls_picked)) = 3 / 55 :=
by
  sorry

end probability_of_two_red_balls_l28_28413


namespace maximum_value_of_a_l28_28557

theorem maximum_value_of_a :
  (∀ x : ℝ, |x - 2| + |x - 8| ≥ a) → a ≤ 6 :=
by
  sorry

end maximum_value_of_a_l28_28557


namespace square_area_l28_28848

theorem square_area (P : ℝ) (hP : P = 32) : ∃ A : ℝ, A = 64 ∧ A = (P / 4) ^ 2 :=
by {
  sorry
}

end square_area_l28_28848


namespace final_cost_l28_28506

-- Definitions of initial conditions
def initial_cart_total : ℝ := 54.00
def discounted_item_original_price : ℝ := 20.00
def discount_rate1 : ℝ := 0.20
def coupon_rate : ℝ := 0.10

-- Prove the final cost after applying discounts
theorem final_cost (initial_cart_total discounted_item_original_price discount_rate1 coupon_rate : ℝ) :
  let discounted_price := discounted_item_original_price * (1 - discount_rate1)
  let total_after_first_discount := initial_cart_total - discounted_price
  let final_total := total_after_first_discount * (1 - coupon_rate)
  final_total = 45.00 :=
by 
  sorry

end final_cost_l28_28506


namespace greatest_x_for_quadratic_inequality_l28_28947

theorem greatest_x_for_quadratic_inequality (x : ℝ) (h : x^2 - 12 * x + 35 ≤ 0) : x ≤ 7 :=
sorry

end greatest_x_for_quadratic_inequality_l28_28947


namespace campers_rowing_afternoon_l28_28475

theorem campers_rowing_afternoon (morning_rowing morning_hiking total : ℕ) 
  (h1 : morning_rowing = 41) 
  (h2 : morning_hiking = 4) 
  (h3 : total = 71) : 
  total - (morning_rowing + morning_hiking) = 26 :=
by
  sorry

end campers_rowing_afternoon_l28_28475


namespace at_least_one_greater_than_one_l28_28996

open Classical

variable (x y : ℝ)

theorem at_least_one_greater_than_one (h : x + y > 2) : x > 1 ∨ y > 1 :=
by
  sorry

end at_least_one_greater_than_one_l28_28996


namespace arithmetic_progression_sum_15_terms_l28_28213

def arithmetic_progression_sum (a₁ d : ℚ) : ℚ :=
  15 * (2 * a₁ + (15 - 1) * d) / 2

def am_prog3_and_9_sum_and_product (a₁ d : ℚ) : Prop :=
  (a₁ + 2 * d) + (a₁ + 8 * d) = 6 ∧ (a₁ + 2 * d) * (a₁ + 8 * d) = 135 / 16

theorem arithmetic_progression_sum_15_terms (a₁ d : ℚ)
  (h : am_prog3_and_9_sum_and_product a₁ d) :
  arithmetic_progression_sum a₁ d = 37.5 ∨ arithmetic_progression_sum a₁ d = 52.5 :=
sorry

end arithmetic_progression_sum_15_terms_l28_28213


namespace smallest_two_digit_multiple_of_3_l28_28815

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n <= 99
def is_multiple_of_3 (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k

theorem smallest_two_digit_multiple_of_3 : ∃ n : ℕ, is_two_digit n ∧ is_multiple_of_3 n ∧ ∀ m : ℕ, is_two_digit m ∧ is_multiple_of_3 m → n <= m :=
sorry

end smallest_two_digit_multiple_of_3_l28_28815


namespace arithmetic_mean_of_fractions_l28_28654

theorem arithmetic_mean_of_fractions :
  let a := 7 / 9
  let b := 5 / 6
  let c := 8 / 9
  2 * b = a + c :=
by
  sorry

end arithmetic_mean_of_fractions_l28_28654


namespace emily_collected_8484_eggs_l28_28686

def number_of_baskets : ℕ := 303
def eggs_per_basket : ℕ := 28
def total_eggs : ℕ := number_of_baskets * eggs_per_basket

theorem emily_collected_8484_eggs : total_eggs = 8484 :=
by
  sorry

end emily_collected_8484_eggs_l28_28686


namespace sum_series_evaluation_l28_28377

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, (if k = 0 then 0 else (2 * k) / (4 : ℝ) ^ k)

theorem sum_series_evaluation : sum_series = 8 / 9 := by
  sorry

end sum_series_evaluation_l28_28377


namespace parabola_passes_through_A_C_l28_28129

theorem parabola_passes_through_A_C : ∃ (a b : ℝ), (2 = a * 1^2 + b * 1 + 1) ∧ (1 = a * 2^2 + b * 2 + 1) :=
by {
  sorry
}

end parabola_passes_through_A_C_l28_28129


namespace chocolate_bar_cost_l28_28669

variable (cost_per_bar num_bars : ℝ)

theorem chocolate_bar_cost (num_scouts smores_per_scout smores_per_bar : ℕ) (total_cost : ℝ)
  (h1 : num_scouts = 15)
  (h2 : smores_per_scout = 2)
  (h3 : smores_per_bar = 3)
  (h4 : total_cost = 15)
  (h5 : num_bars = (num_scouts * smores_per_scout) / smores_per_bar)
  (h6 : total_cost = cost_per_bar * num_bars) :
  cost_per_bar = 1.50 :=
by
  sorry

end chocolate_bar_cost_l28_28669


namespace least_possible_value_l28_28245

theorem least_possible_value (x y : ℝ) : (x + y - 1)^2 + (x * y)^2 ≥ 0 :=
by 
  sorry

end least_possible_value_l28_28245


namespace sum_of_first_five_integers_l28_28856

theorem sum_of_first_five_integers : (1 + 2 + 3 + 4 + 5) = 15 := 
by 
  sorry

end sum_of_first_five_integers_l28_28856


namespace intersection_A_compB_l28_28814

def setA : Set ℤ := {x | (abs (x - 1) < 3)}
def setB : Set ℝ := {x | x^2 + 2 * x - 3 ≥ 0}
def setCompB : Set ℝ := {x | ¬(x^2 + 2 * x - 3 ≥ 0)}

theorem intersection_A_compB :
  { x : ℤ | x ∈ setA ∧ (x:ℝ) ∈ setCompB } = {-1, 0} :=
sorry

end intersection_A_compB_l28_28814


namespace apple_tree_distribution_l28_28353

-- Definition of the problem
noncomputable def paths := 4

-- Definition of the apple tree positions
structure Position where
  x : ℕ -- Coordinate x
  y : ℕ -- Coordinate y

-- Definition of the initial condition: one existing apple tree
def existing_apple_tree : Position := {x := 0, y := 0}

-- Problem: proving the existence of a configuration with three new apple trees
theorem apple_tree_distribution :
  ∃ (p1 p2 p3 : Position),
    (p1 ≠ existing_apple_tree) ∧ (p2 ≠ existing_apple_tree) ∧ (p3 ≠ existing_apple_tree) ∧
    -- Ensure each path has equal number of trees on both sides
    (∃ (path1 path2 : ℕ), 
      -- Horizontal path balance
      path1 = (if p1.x > 0 then 1 else 0) + (if p2.x > 0 then 1 else 0) + (if p3.x > 0 then 1 else 0) + 1 ∧
      path2 = (if p1.x < 0 then 1 else 0) + (if p2.x < 0 then 1 else 0) + (if p3.x < 0 then 1 else 0) ∧
      path1 = path2) ∧
    (∃ (path3 path4 : ℕ), 
      -- Vertical path balance
      path3 = (if p1.y > 0 then 1 else 0) + (if p2.y > 0 then 1 else 0) + (if p3.y > 0 then 1 else 0) + 1 ∧
      path4 = (if p1.y < 0 then 1 else 0) + (if p2.y < 0 then 1 else 0) + (if p3.y < 0 then 1 else 0) ∧
      path3 = path4)
  := by sorry

end apple_tree_distribution_l28_28353


namespace gcd_12345_6789_l28_28790

theorem gcd_12345_6789 : Int.gcd 12345 6789 = 3 :=
by
  sorry

end gcd_12345_6789_l28_28790


namespace sum_of_three_consecutive_odd_integers_l28_28989

theorem sum_of_three_consecutive_odd_integers (a : ℤ) (h₁ : a + (a + 4) = 150) :
  (a + (a + 2) + (a + 4) = 225) :=
by
  sorry

end sum_of_three_consecutive_odd_integers_l28_28989


namespace findAnalyticalExpression_l28_28092

-- Defining the point A as a structure with x and y coordinates
structure Point where
  x : ℝ
  y : ℝ

-- Defining a line as having a slope and y-intercept
structure Line where
  slope : ℝ
  intercept : ℝ

-- Condition: Line 1 is parallel to y = 2x - 3
def line1 : Line := {slope := 2, intercept := -3}

-- Condition: Line 2 passes through point A
def point_A : Point := {x := -2, y := -1}

-- The theorem statement:
theorem findAnalyticalExpression : 
  ∃ b : ℝ, (∀ x : ℝ, (point_A.y = line1.slope * point_A.x + b) → b = 3) ∧ 
            ∀ x : ℝ, (line1.slope * x + b = 2 * x + 3) :=
sorry

end findAnalyticalExpression_l28_28092


namespace triangle_inequalities_l28_28993

theorem triangle_inequalities (a b c h_a h_b h_c : ℝ) (ha_eq : h_a = b * Real.sin (arc_c)) (hb_eq : h_b = a * Real.sin (arc_c)) (hc_eq : h_c = a * Real.sin (arc_b)) (h : a > b) (h2 : b > c) :
  (a + h_a > b + h_b) ∧ (b + h_b > c + h_c) :=
by
  sorry

end triangle_inequalities_l28_28993


namespace trisha_money_left_l28_28800

theorem trisha_money_left
    (meat cost: ℕ) (chicken_cost: ℕ) (veggies_cost: ℕ) (eggs_cost: ℕ) (dog_food_cost: ℕ) 
    (initial_money: ℕ) (total_spent: ℕ) (money_left: ℕ) :
    meat_cost = 17 →
    chicken_cost = 22 →
    veggies_cost = 43 →
    eggs_cost = 5 →
    dog_food_cost = 45 →
    initial_money = 167 →
    total_spent = meat_cost + chicken_cost + veggies_cost + eggs_cost + dog_food_cost →
    money_left = initial_money - total_spent →
    money_left = 35 :=
by
    intros
    sorry

end trisha_money_left_l28_28800


namespace derek_walk_time_l28_28889

theorem derek_walk_time (x : ℕ) :
  (∀ y : ℕ, (y = 9) → (∀ d₁ d₂ : ℕ, (d₁ = 20 ∧ d₂ = 60) →
    (20 * x = d₁ * y + d₂))) → x = 12 :=
by
  intro h
  sorry

end derek_walk_time_l28_28889


namespace new_volume_is_80_gallons_l28_28176

-- Define the original volume
def V_original : ℝ := 5

-- Define the factors by which length, width, and height are increased
def length_factor : ℝ := 2
def width_factor : ℝ := 2
def height_factor : ℝ := 4

-- Define the new volume
def V_new : ℝ := V_original * (length_factor * width_factor * height_factor)

-- Theorem to prove the new volume is 80 gallons
theorem new_volume_is_80_gallons : V_new = 80 := 
by
  -- Proof goes here
  sorry

end new_volume_is_80_gallons_l28_28176


namespace no_intersection_points_l28_28693

theorem no_intersection_points :
  ¬ ∃ x y : ℝ, y = |3 * x + 4| ∧ y = -|2 * x + 1| :=
by
  sorry

end no_intersection_points_l28_28693


namespace lcm_of_denominators_l28_28823

theorem lcm_of_denominators : Nat.lcm (List.foldr Nat.lcm 1 [2, 3, 4, 5, 6, 7]) = 420 :=
by 
  sorry

end lcm_of_denominators_l28_28823


namespace sum_of_reciprocals_l28_28701

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) : (1/x) + (1/y) = 3/8 :=
by
  sorry

end sum_of_reciprocals_l28_28701


namespace inequality_solution_l28_28799

theorem inequality_solution (x : ℝ) :
  (6 * (x ^ 3 - 8) * (Real.sqrt (x ^ 2 + 6 * x + 9)) / ((x ^ 2 + 2 * x + 4) * (x ^ 2 + x - 6)) ≥ x - 2) ↔
  (x ∈ Set.Iic (-4) ∪ Set.Ioo (-3) 2 ∪ Set.Ioo 2 8) := sorry

end inequality_solution_l28_28799


namespace intersection_A_B_l28_28422

-- Define set A
def A : Set ℝ := { y | ∃ x : ℝ, y = Real.log x }

-- Define set B
def B : Set ℝ := { x | ∃ y : ℝ, y = Real.sqrt x }

-- Prove that the intersection of sets A and B is [0, +∞)
theorem intersection_A_B : A ∩ B = { x | 0 ≤ x } :=
by
  sorry

end intersection_A_B_l28_28422


namespace employed_males_percentage_l28_28807

theorem employed_males_percentage (total_population employed employed_as_percent employed_females female_as_percent employed_males employed_males_percentage : ℕ) 
(total_population_eq : total_population = 100)
(employed_eq : employed = employed_as_percent * total_population / 100)
(employed_as_percent_eq : employed_as_percent = 60)
(employed_females_eq : employed_females = female_as_percent * employed / 100)
(female_as_percent_eq : female_as_percent = 25)
(employed_males_eq : employed_males = employed - employed_females)
(employed_males_percentage_eq : employed_males_percentage = employed_males * 100 / total_population) :
employed_males_percentage = 45 :=
sorry

end employed_males_percentage_l28_28807


namespace value_of_a_l28_28048

-- Declare and define the given conditions.
def line1 (y : ℝ) := y = 13
def line2 (x t y : ℝ) := y = 3 * x + t

-- Define the proof statement.
theorem value_of_a (a b t : ℝ) (h1 : line1 b) (h2 : line2 a t b) (ht : t = 1) : a = 4 :=
by
  sorry

end value_of_a_l28_28048


namespace count_four_digit_numbers_without_1_or_4_l28_28870

-- Define a function to check if a digit is allowed (i.e., not 1 or 4)
def allowed_digit (d : ℕ) : Prop := d ≠ 1 ∧ d ≠ 4

-- Function to count four-digit numbers without digits 1 or 4
def count_valid_four_digit_numbers : ℕ :=
  let valid_first_digits := [2, 3, 5, 6, 7, 8, 9]
  let valid_other_digits := [0, 2, 3, 5, 6, 7, 8, 9]
  (valid_first_digits.length) * (valid_other_digits.length ^ 3)

-- The main theorem stating that the number of valid four-digit integers is 3072
theorem count_four_digit_numbers_without_1_or_4 : count_valid_four_digit_numbers = 3072 :=
by
  sorry

end count_four_digit_numbers_without_1_or_4_l28_28870


namespace committee_count_l28_28636

noncomputable def num_acceptable_committees (total_people : ℕ) (committee_size : ℕ) (conditions : List (Set ℕ)) : ℕ := sorry

theorem committee_count :
  num_acceptable_committees 9 5 [ {1, 2}, {3, 4} ] = 41 := sorry

end committee_count_l28_28636


namespace sum_f_1_2021_l28_28874

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom equation_f : ∀ x : ℝ, f (2 + x) + f (2 - x) = 0
axiom interval_f : ∀ x : ℝ, -1 ≤ x ∧ x < 0 → f x = Real.log (1 - x) / Real.log 2

theorem sum_f_1_2021 : (List.sum (List.map f (List.range' 1 2021))) = -1 := sorry

end sum_f_1_2021_l28_28874


namespace multiplication_factor_correct_l28_28333

theorem multiplication_factor_correct (N X : ℝ) (h1 : 98 = abs ((N * X - N / 10) / (N * X)) * 100) : X = 5 := by
  sorry

end multiplication_factor_correct_l28_28333


namespace slips_numbers_exist_l28_28025

theorem slips_numbers_exist (x y z : ℕ) (h₁ : x + y + z = 20) (h₂ : 5 * x + 3 * y = 46) : 
  (x = 4) ∧ (y = 10) ∧ (z = 6) :=
by {
  -- Technically, the actual proving steps should go here, but skipped due to 'sorry'
  sorry
}

end slips_numbers_exist_l28_28025


namespace nine_div_repeating_decimal_l28_28590

noncomputable def repeating_decimal := 1 / 3

theorem nine_div_repeating_decimal : 9 / repeating_decimal = 27 := by
  sorry

end nine_div_repeating_decimal_l28_28590


namespace n_times_s_eq_2023_l28_28571

noncomputable def S := { x : ℝ | x > 0 }

-- Function f: S → ℝ
def f (x : ℝ) : ℝ := sorry

-- Condition: f(x) f(y) = f(xy) + 2023 * (2/x + 2/y + 2022) for all x, y > 0
axiom f_property (x y : ℝ) (hx : x > 0) (hy : y > 0) : f x * f y = f (x * y) + 2023 * (2 / x + 2 / y + 2022)

-- Theorem: Prove n × s = 2023 where n is the number of possible values of f(2) and s is the sum of all possible values of f(2)
theorem n_times_s_eq_2023 (n s : ℕ) : n * s = 2023 :=
sorry

end n_times_s_eq_2023_l28_28571


namespace problem_l28_28348

def vec_a : ℝ × ℝ := (5, 3)
def vec_b : ℝ × ℝ := (1, -2)
def two_vec_b : ℝ × ℝ := (2 * 1, 2 * -2)
def expected_result : ℝ × ℝ := (3, 7)

theorem problem : (vec_a.1 - two_vec_b.1, vec_a.2 - two_vec_b.2) = expected_result :=
by
  sorry

end problem_l28_28348


namespace rationalize_sqrt_5_over_12_l28_28826

theorem rationalize_sqrt_5_over_12 : Real.sqrt (5 / 12) = (Real.sqrt 15) / 6 :=
sorry

end rationalize_sqrt_5_over_12_l28_28826


namespace ratio_green_to_yellow_l28_28262

theorem ratio_green_to_yellow (yellow fish blue fish green fish total fish : ℕ) 
  (h_yellow : yellow = 12)
  (h_blue : blue = yellow / 2)
  (h_total : total = yellow + blue + green)
  (h_aquarium_total : total = 42) : 
  green / yellow = 2 := 
sorry

end ratio_green_to_yellow_l28_28262


namespace area_computation_l28_28284

noncomputable def areaOfBoundedFigure : ℝ :=
  let x (t : ℝ) := 2 * Real.sqrt 2 * Real.cos t
  let y (t : ℝ) := 5 * Real.sqrt 2 * Real.sin t
  let rectArea := 20
  let integral := ∫ t in (3 * Real.pi / 4)..(Real.pi / 4), 
    (5 * Real.sqrt 2 * Real.sin t) * (-2 * Real.sqrt 2 * Real.sin t)
  (integral / 2) - rectArea

theorem area_computation :
  let x (t : ℝ) := 2 * Real.sqrt 2 * Real.cos t
  let y (t : ℝ) := 5 * Real.sqrt 2 * Real.sin t
  let rectArea := 20
  let integral := ∫ t in (3 * Real.pi / 4)..(Real.pi / 4),
    (5 * Real.sqrt 2 * Real.sin t) * (-2 * Real.sqrt 2 * Real.sin t)
  ((integral / 2) - rectArea) = (5 * Real.pi - 10) :=
by
  sorry

end area_computation_l28_28284


namespace find_a_45_l28_28880

theorem find_a_45 (a : ℕ → ℝ) 
  (h0 : a 0 = 11) 
  (h1 : a 1 = 11) 
  (h_rec : ∀ m n : ℕ, a (m + n) = (1 / 2) * (a (2 * m) + a (2 * n)) - (m - n) ^ 2) 
  : a 45 = 1991 :=
sorry

end find_a_45_l28_28880


namespace routes_from_M_to_N_l28_28697

structure Paths where
  -- Specify the paths between nodes
  C_to_N : ℕ
  D_to_N : ℕ
  A_to_C : ℕ
  A_to_D : ℕ
  B_to_N : ℕ
  B_to_A : ℕ
  B_to_C : ℕ
  M_to_B : ℕ
  M_to_A : ℕ

theorem routes_from_M_to_N (p : Paths) : 
  p.C_to_N = 1 → 
  p.D_to_N = 1 →
  p.A_to_C = 1 →
  p.A_to_D = 1 →
  p.B_to_N = 1 →
  p.B_to_A = 1 →
  p.B_to_C = 1 →
  p.M_to_B = 1 →
  p.M_to_A = 1 →
  (p.M_to_B * (p.B_to_N + (p.B_to_A * (p.A_to_C + p.A_to_D)) + p.B_to_C)) + 
  (p.M_to_A * (p.A_to_C + p.A_to_D)) = 6 
:= by
  sorry

end routes_from_M_to_N_l28_28697


namespace train_length_l28_28248

noncomputable def length_of_train (speed_kmph : ℝ) (time_sec : ℝ) (length_platform_m : ℝ) : ℝ :=
  let speed_ms := (speed_kmph * 1000) / 3600
  let distance_covered := speed_ms * time_sec
  distance_covered - length_platform_m

theorem train_length :
  length_of_train 72 25 340.04 = 159.96 := by
  sorry

end train_length_l28_28248


namespace cost_price_computer_table_l28_28417

theorem cost_price_computer_table (C S : ℝ) (hS1 : S = 1.25 * C) (hS2 : S = 1000) : C = 800 :=
by
  sorry

end cost_price_computer_table_l28_28417


namespace divisible_by_11_l28_28450

theorem divisible_by_11 (n : ℤ) : (11 ∣ (n^2001 - n^4)) ↔ (n % 11 = 0 ∨ n % 11 = 1) :=
by
  sorry

end divisible_by_11_l28_28450


namespace angle_triple_supplement_l28_28166

theorem angle_triple_supplement {x : ℝ} (h1 : ∀ y : ℝ, y + (180 - y) = 180) (h2 : x = 3 * (180 - x)) :
  x = 135 :=
by
  sorry

end angle_triple_supplement_l28_28166


namespace complete_pairs_of_socks_l28_28559

def initial_pairs_blue : ℕ := 20
def initial_pairs_green : ℕ := 15
def initial_pairs_red : ℕ := 15

def lost_socks_blue : ℕ := 3
def lost_socks_green : ℕ := 2
def lost_socks_red : ℕ := 2

def donated_socks_blue : ℕ := 10
def donated_socks_green : ℕ := 15
def donated_socks_red : ℕ := 10

def purchased_pairs_blue : ℕ := 5
def purchased_pairs_green : ℕ := 3
def purchased_pairs_red : ℕ := 2

def gifted_pairs_blue : ℕ := 2
def gifted_pairs_green : ℕ := 1

theorem complete_pairs_of_socks : 
  (initial_pairs_blue - 1 - (donated_socks_blue / 2) + purchased_pairs_blue + gifted_pairs_blue) +
  (initial_pairs_green - 1 - (donated_socks_green / 2) + purchased_pairs_green + gifted_pairs_green) +
  (initial_pairs_red - 1 - (donated_socks_red / 2) + purchased_pairs_red) = 43 := by
  sorry

end complete_pairs_of_socks_l28_28559


namespace cats_left_in_store_l28_28467

theorem cats_left_in_store 
  (initial_siamese : ℕ := 25)
  (initial_persian : ℕ := 18)
  (initial_house : ℕ := 12)
  (initial_maine_coon : ℕ := 10)
  (sold_siamese : ℕ := 6)
  (sold_persian : ℕ := 4)
  (sold_maine_coon : ℕ := 3)
  (sold_house : ℕ := 0)
  (remaining_siamese : ℕ := 19)
  (remaining_persian : ℕ := 14)
  (remaining_house : ℕ := 12)
  (remaining_maine_coon : ℕ := 7) : 
  initial_siamese - sold_siamese = remaining_siamese ∧
  initial_persian - sold_persian = remaining_persian ∧
  initial_house - sold_house = remaining_house ∧
  initial_maine_coon - sold_maine_coon = remaining_maine_coon :=
by sorry

end cats_left_in_store_l28_28467


namespace tank_capacity_l28_28497

theorem tank_capacity (T : ℝ) (h : (3 / 4) * T + 7 = (7 / 8) * T) : T = 56 := 
sorry

end tank_capacity_l28_28497


namespace road_repair_equation_l28_28334

variable (x : ℝ) 

-- Original problem conditions
def total_road_length := 150
def extra_repair_per_day := 5
def days_ahead := 5

-- The proof problem to show that the schedule differential equals 5 days ahead
theorem road_repair_equation :
  (total_road_length / x) - (total_road_length / (x + extra_repair_per_day)) = days_ahead :=
sorry

end road_repair_equation_l28_28334


namespace least_number_l28_28228

theorem least_number (n p q r s : ℕ) : 
  (n + p) % 24 = 0 ∧ 
  (n + q) % 32 = 0 ∧ 
  (n + r) % 36 = 0 ∧
  (n + s) % 54 = 0 →
  n = 863 :=
sorry

end least_number_l28_28228


namespace converse_of_prop1_true_l28_28972

theorem converse_of_prop1_true
  (h1 : ∀ {x : ℝ}, x^2 - 3 * x + 2 = 0 → x = 1 ∨ x = 2)
  (h2 : ∀ {x : ℝ}, -2 ≤ x ∧ x < 3 → (x - 2) * (x - 3) ≤ 0)
  (h3 : ∀ {x y : ℝ}, x = 0 ∧ y = 0 → x^2 + y^2 = 0)
  (h4 : ∀ {x y : ℕ}, x > 0 ∧ y > 0 ∧ (x + y) % 2 = 1 → (x % 2 = 1 ∧ y % 2 = 0) ∨ (x % 2 = 0 ∧ y % 2 = 1)) :
  (∀ {x : ℝ}, x = 1 ∨ x = 2 → x^2 - 3 * x + 2 = 0) :=
by
  sorry

end converse_of_prop1_true_l28_28972


namespace second_carpenter_days_l28_28058

theorem second_carpenter_days (x : ℚ) (h1 : 1 / 5 + 1 / x = 1 / 2) : x = 10 / 3 :=
by
  sorry

end second_carpenter_days_l28_28058


namespace fill_tank_time_l28_28664

theorem fill_tank_time 
  (tank_capacity : ℕ) (initial_fill : ℕ) (fill_rate : ℝ) 
  (drain_rate1 : ℝ) (drain_rate2 : ℝ) : 
  tank_capacity = 8000 ∧ initial_fill = 4000 ∧ fill_rate = 0.5 ∧ drain_rate1 = 0.25 ∧ drain_rate2 = 0.1667 
  → (initial_fill + fill_rate * t - (drain_rate1 + drain_rate2) * t) = tank_capacity → t = 48 := sorry

end fill_tank_time_l28_28664


namespace value_of_f_neg_a_l28_28975

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem value_of_f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 0 :=
by
  sorry

end value_of_f_neg_a_l28_28975


namespace fisherman_gets_8_red_snappers_l28_28414

noncomputable def num_red_snappers (R : ℕ) : Prop :=
  let cost_red_snapper := 3
  let cost_tuna := 2
  let num_tunas := 14
  let total_earnings := 52
  (R * cost_red_snapper) + (num_tunas * cost_tuna) = total_earnings

theorem fisherman_gets_8_red_snappers : num_red_snappers 8 :=
by
  sorry

end fisherman_gets_8_red_snappers_l28_28414


namespace find_ordered_pair_l28_28903

open Polynomial

theorem find_ordered_pair (a b : ℝ) :
  (∀ x : ℝ, (((x^3 + a * x^2 + 17 * x + 10 = 0) ∧ (x^3 + b * x^2 + 20 * x + 12 = 0)) → 
  (x = -6 ∧ y = -7))) :=
sorry

end find_ordered_pair_l28_28903


namespace isosceles_triangle_largest_angle_l28_28869

theorem isosceles_triangle_largest_angle (A B C : Type) (α β γ : ℝ)
  (h_iso : α = β) (h_angles : α = 50) (triangle: α + β + γ = 180) : γ = 80 :=
sorry

end isosceles_triangle_largest_angle_l28_28869


namespace domain_of_function_l28_28820

theorem domain_of_function (x : ℝ) : 4 - x ≥ 0 ∧ x ≠ 2 ↔ (x ≤ 4 ∧ x ≠ 2) :=
sorry

end domain_of_function_l28_28820


namespace probability_cd_l28_28285

theorem probability_cd (P_A P_B : ℚ) (h1 : P_A = 1/4) (h2 : P_B = 1/3) :
  (1 - P_A - P_B = 5/12) :=
by
  -- Placeholder for the proof
  sorry

end probability_cd_l28_28285


namespace difference_of_squares_l28_28035

theorem difference_of_squares : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end difference_of_squares_l28_28035


namespace senate_arrangement_l28_28100

def countArrangements : ℕ :=
  let totalSeats : ℕ := 14
  let democrats : ℕ := 6
  let republicans : ℕ := 6
  let independents : ℕ := 2
  -- The calculation for arrangements considering fixed elements, and permutations adjusted for rotation
  12 * (Nat.factorial 10 / 2)

theorem senate_arrangement :
  let totalSeats : ℕ := 14
  let democrats : ℕ := 6
  let republicans : ℕ := 6
  let independents : ℕ := 2
  -- Total ways to arrange the members around the table under the given conditions
  countArrangements = 21772800 :=
by
  sorry

end senate_arrangement_l28_28100


namespace problem_solution_l28_28346

theorem problem_solution (a b c d : ℝ) (h1 : a = 5 * b) (h2 : b = 3 * c) (h3 : c = 6 * d) :
  (a + b * c) / (c + d * b) = (3 * (5 + 6 * d)) / (1 + 3 * d) :=
by
  sorry

end problem_solution_l28_28346


namespace leon_older_than_aivo_in_months_l28_28470

theorem leon_older_than_aivo_in_months
    (jolyn therese aivo leon : ℕ)
    (h1 : jolyn = therese + 2)
    (h2 : therese = aivo + 5)
    (h3 : jolyn = leon + 5) :
    leon = aivo + 2 := 
sorry

end leon_older_than_aivo_in_months_l28_28470


namespace rectangle_length_l28_28810

-- Define the area and width of the rectangle as given
def width : ℝ := 4
def area  : ℝ := 28

-- Prove that the length is 7 cm given the conditions
theorem rectangle_length : ∃ length : ℝ, length = 7 ∧ area = length * width :=
sorry

end rectangle_length_l28_28810


namespace remainder_sum_of_first_eight_primes_div_tenth_prime_l28_28668

theorem remainder_sum_of_first_eight_primes_div_tenth_prime :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) % 29 = 19 :=
by norm_num

end remainder_sum_of_first_eight_primes_div_tenth_prime_l28_28668


namespace alice_bob_task_l28_28806

theorem alice_bob_task (t : ℝ) (h₁ : 1/4 + 1/6 = 5/12) (h₂ : t - 1/2 ≠ 0) :
    (5/12) * (t - 1/2) = 1 :=
sorry

end alice_bob_task_l28_28806


namespace nth_number_in_S_l28_28827

def S : Set ℕ := {n | ∃ k : ℕ, n = 15 * k + 11}

theorem nth_number_in_S (n : ℕ) (hn : n = 127) : ∃ k, 15 * k + 11 = 1901 :=
by
  sorry

end nth_number_in_S_l28_28827


namespace incorrect_statement_D_l28_28588

theorem incorrect_statement_D (a b r : ℝ) (hr : r > 0) :
  ¬ ∀ b < r, ∃ x, (x - a)^2 + (0 - b)^2 = r^2 :=
by 
  sorry

end incorrect_statement_D_l28_28588


namespace abs_ineq_solution_l28_28416

theorem abs_ineq_solution (x : ℝ) :
  (|x - 2| + |x + 1| < 4) ↔ (x ∈ Set.Ioo (-7 / 2) (-1) ∪ Set.Ico (-1) (5 / 2)) := by
  sorry

end abs_ineq_solution_l28_28416


namespace group_allocation_minimizes_time_total_duration_after_transfer_l28_28267

theorem group_allocation_minimizes_time :
  ∃ x y : ℕ,
  x + y = 52 ∧
  (x = 20 ∧ y = 32) ∧
  (min (60 / x) (100 / y) = 25 / 8) := sorry

theorem total_duration_after_transfer (x y x' y' : ℕ) (H : x = 20) (H1 : y = 32) (H2 : x' = x - 6) (H3 : y' = y + 6) :
  min ((100 * (2/5)) / x') ((152 * (2/3)) / y') = 27 / 7 := sorry

end group_allocation_minimizes_time_total_duration_after_transfer_l28_28267


namespace fraction_value_eq_l28_28913

theorem fraction_value_eq : (5 * 8) / 10 = 4 := 
by 
  sorry

end fraction_value_eq_l28_28913


namespace solve_equation_l28_28837

theorem solve_equation (x : ℝ) (h : 2 / x = 1 / (x + 1)) : x = -2 :=
sorry

end solve_equation_l28_28837


namespace matias_fewer_cards_l28_28230

theorem matias_fewer_cards (J M C : ℕ) (h1 : J = M) (h2 : C = 20) (h3 : C + M + J = 48) : C - M = 6 :=
by
-- To be proven
  sorry

end matias_fewer_cards_l28_28230


namespace inequality_proof_l28_28268

variable (a b : ℝ)

theorem inequality_proof (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 4) : 
  (1 / (a^2 + b^2) ≤ 1 / 8) :=
by
  sorry

end inequality_proof_l28_28268


namespace value_of_k_l28_28437

open Nat

def perm (n r : ℕ) : ℕ := factorial n / factorial (n - r)
def comb (n r : ℕ) : ℕ := factorial n / (factorial r * factorial (n - r))

theorem value_of_k : ∃ k : ℕ, perm 32 6 = k * comb 32 6 ∧ k = 720 := by
  use 720
  unfold perm comb
  sorry

end value_of_k_l28_28437


namespace find_value_of_a_l28_28825

theorem find_value_of_a (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
(h_eq : 7 * a^2 + 14 * a * b = a^3 + 2 * a^2 * b) : a = 7 := 
sorry

end find_value_of_a_l28_28825


namespace cos_four_times_arccos_val_l28_28661

theorem cos_four_times_arccos_val : 
  ∀ x : ℝ, x = Real.arccos (1 / 4) → Real.cos (4 * x) = 17 / 32 :=
by
  intro x h
  sorry

end cos_four_times_arccos_val_l28_28661


namespace trip_time_80_minutes_l28_28677

noncomputable def v : ℝ := 1 / 2
noncomputable def speed_highway := 4 * v -- 4 times speed on the highway
noncomputable def time_mountain : ℝ := 20 / v -- Distance on mountain road divided by speed on mountain road
noncomputable def time_highway : ℝ := 80 / speed_highway -- Distance on highway divided by speed on highway
noncomputable def total_time := time_mountain + time_highway

theorem trip_time_80_minutes : total_time = 80 :=
by sorry

end trip_time_80_minutes_l28_28677


namespace car_speed_is_120_l28_28736

theorem car_speed_is_120 (v t : ℝ) (h1 : v > 0) (h2 : t > 0) (h3 : v * t = 75)
  (h4 : 1.5 * v * (t - (12.5 / 60)) = 75) : v = 120 := by
  sorry

end car_speed_is_120_l28_28736


namespace rotation_transforms_and_sums_l28_28746

theorem rotation_transforms_and_sums 
    (D E F D' E' F' : (ℝ × ℝ))
    (hD : D = (0, 0)) (hE : E = (0, 20)) (hF : F = (30, 0)) 
    (hD' : D' = (-26, 23)) (hE' : E' = (-46, 23)) (hF' : F' = (-26, -7))
    (n : ℝ) (x y : ℝ)
    (rotation_condition : 0 < n ∧ n < 180)
    (angle_condition : n = 90) :
    n + x + y = 60.5 :=
by
  have hx : x = -49 := sorry
  have hy : y = 19.5 := sorry
  have hn : n = 90 := sorry
  sorry

end rotation_transforms_and_sums_l28_28746


namespace trapezoid_area_l28_28465

theorem trapezoid_area
  (AD BC AC BD : ℝ)
  (h1 : AD = 24)
  (h2 : BC = 8)
  (h3 : AC = 13)
  (h4 : BD = 5 * Real.sqrt 17) : 
  ∃ (area : ℝ), area = 80 :=
by
  let area := (1 / 2) * (AD + BC) * 5
  existsi area
  sorry

end trapezoid_area_l28_28465


namespace log_exp_identity_l28_28152

theorem log_exp_identity (a : ℝ) (h : a = Real.log 5 / Real.log 4) : 
  (2^a + 2^(-a) = 6 * Real.sqrt 5 / 5) :=
by {
  -- a = log_4 (5) can be rewritten using change-of-base formula: log 5 / log 4
  -- so, it can be used directly in the theorem
  sorry
}

end log_exp_identity_l28_28152


namespace no_integer_solution_xyz_l28_28764

theorem no_integer_solution_xyz : ¬ ∃ (x y z : ℤ),
  x^6 + x^3 + x^3 * y + y = 147^157 ∧
  x^3 + x^3 * y + y^2 + y + z^9 = 157^147 := by
  sorry

end no_integer_solution_xyz_l28_28764


namespace sum_first_seven_terms_geometric_sequence_l28_28719

noncomputable def sum_geometric_sequence (a r : ℚ) (n : ℕ) : ℚ := 
  a * (1 - r^n) / (1 - r)

theorem sum_first_seven_terms_geometric_sequence : 
  sum_geometric_sequence (1/4) (1/4) 7 = 16383 / 49152 := 
by
  sorry

end sum_first_seven_terms_geometric_sequence_l28_28719


namespace total_savings_percentage_l28_28108

theorem total_savings_percentage
  (original_coat_price : ℕ) (original_pants_price : ℕ)
  (coat_discount_percent : ℚ) (pants_discount_percent : ℚ)
  (original_total_price : ℕ) (total_savings : ℕ)
  (savings_percentage : ℚ) :
  original_coat_price = 120 →
  original_pants_price = 60 →
  coat_discount_percent = 0.30 →
  pants_discount_percent = 0.60 →
  original_total_price = original_coat_price + original_pants_price →
  total_savings = original_coat_price * coat_discount_percent + original_pants_price * pants_discount_percent →
  savings_percentage = (total_savings / original_total_price) * 100 →
  savings_percentage = 40 := 
by
  intros
  sorry

end total_savings_percentage_l28_28108


namespace ratio_of_squares_l28_28053

def square_inscribed_triangle_1 (x : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
  a = 6 ∧ b = 8 ∧ c = 10 ∧
  x = 24 / 7

def square_inscribed_triangle_2 (y : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
  a = 6 ∧ b = 8 ∧ c = 10 ∧
  y = 10 / 3

theorem ratio_of_squares (x y : ℝ) 
  (hx : square_inscribed_triangle_1 x) 
  (hy : square_inscribed_triangle_2 y) : 
  x / y = 36 / 35 := 
by sorry

end ratio_of_squares_l28_28053


namespace inequality_trig_l28_28069

theorem inequality_trig 
  (x y z : ℝ) 
  (hx : 0 < x ∧ x < (π / 2)) 
  (hy : 0 < y ∧ y < (π / 2)) 
  (hz : 0 < z ∧ z < (π / 2)) :
  (π / 2) + 2 * (Real.sin x) * (Real.cos y) + 2 * (Real.sin y) * (Real.cos z) > 
  (Real.sin (2 * x)) + (Real.sin (2 * y)) + (Real.sin (2 * z)) :=
by
  sorry  -- The proof is omitted

end inequality_trig_l28_28069


namespace max_abs_sum_eq_two_l28_28390

theorem max_abs_sum_eq_two (x y : ℝ) (h : x^2 + y^2 = 2) : |x| + |y| ≤ 2 :=
by
  sorry

end max_abs_sum_eq_two_l28_28390


namespace dante_initially_has_8_jelly_beans_l28_28319

-- Conditions
def aaron_jelly_beans : ℕ := 5
def bianca_jelly_beans : ℕ := 7
def callie_jelly_beans : ℕ := 8
def dante_jelly_beans_initially (D : ℕ) : Prop := 
  ∀ (D : ℕ), (6 ≤ D - 1 ∧ D - 1 ≤ callie_jelly_beans - 1)

-- Theorem
theorem dante_initially_has_8_jelly_beans :
  ∃ (D : ℕ), (aaron_jelly_beans + 1 = 6) →
             (callie_jelly_beans = 8) →
             dante_jelly_beans_initially D →
             D = 8 := 
by
  sorry

end dante_initially_has_8_jelly_beans_l28_28319


namespace hexagon_side_lengths_l28_28195

theorem hexagon_side_lengths (a b c d e f : ℕ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ f)
(h1: a = 7 ∧ b = 5 ∧ (a + b + c + d + e + f = 38)) : 
(a + b + c + d + e + f = 38 ∧ a + b + c + d + e + f = 7 + 7 + 7 + 7 + 5 + 5) → 
(a + b + c + d + e + f = (4 * 7) + (2 * 5)) :=
sorry

end hexagon_side_lengths_l28_28195


namespace quadratic_inequality_solution_set_l28_28219

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - x - 2 < 0} = {x : ℝ | -1 < x ∧ x < 2} :=
by
  sorry

end quadratic_inequality_solution_set_l28_28219


namespace music_stand_cost_proof_l28_28329

-- Definitions of the constants involved
def flute_cost : ℝ := 142.46
def song_book_cost : ℝ := 7.00
def total_spent : ℝ := 158.35
def music_stand_cost : ℝ := total_spent - (flute_cost + song_book_cost)

-- The statement we need to prove
theorem music_stand_cost_proof : music_stand_cost = 8.89 := 
by
  sorry

end music_stand_cost_proof_l28_28329


namespace combined_length_in_scientific_notation_l28_28660

noncomputable def yards_to_inches (yards : ℝ) : ℝ := yards * 36
noncomputable def inches_to_cm (inches : ℝ) : ℝ := inches * 2.54
noncomputable def feet_to_inches (feet : ℝ) : ℝ := feet * 12

def sports_stadium_length_yards : ℝ := 61
def safety_margin_feet : ℝ := 2
def safety_margin_inches : ℝ := 9

theorem combined_length_in_scientific_notation :
  (inches_to_cm (yards_to_inches sports_stadium_length_yards) +
   (inches_to_cm (feet_to_inches safety_margin_feet + safety_margin_inches)) * 2) = 5.74268 * 10^3 :=
by
  sorry

end combined_length_in_scientific_notation_l28_28660


namespace total_accidents_l28_28018

theorem total_accidents :
  let accidentsA := (75 / 100) * 2500
  let accidentsB := (50 / 80) * 1600
  let accidentsC := (90 / 200) * 1900
  accidentsA + accidentsB + accidentsC = 3730 :=
by
  let accidentsA := (75 / 100) * 2500
  let accidentsB := (50 / 80) * 1600
  let accidentsC := (90 / 200) * 1900
  sorry

end total_accidents_l28_28018


namespace bug_total_distance_l28_28733

theorem bug_total_distance : 
  let start := 3
  let first_point := 9
  let second_point := -4
  let distance_1 := abs (first_point - start)
  let distance_2 := abs (second_point - first_point)
  distance_1 + distance_2 = 19 := 
by
  sorry

end bug_total_distance_l28_28733


namespace two_bags_remainder_l28_28071

-- Given conditions
variables (n : ℕ)

-- Assume n ≡ 8 (mod 11)
def satisfied_mod_condition : Prop := n % 11 = 8

-- Prove that 2n ≡ 5 (mod 11)
theorem two_bags_remainder (h : satisfied_mod_condition n) : (2 * n) % 11 = 5 :=
by 
  unfold satisfied_mod_condition at h
  sorry

end two_bags_remainder_l28_28071


namespace solve_chair_table_fraction_l28_28644

def chair_table_fraction : Prop :=
  ∃ (C T : ℝ), T = 140 ∧ (T + 4 * C = 220) ∧ (C / T = 1 / 7)

theorem solve_chair_table_fraction : chair_table_fraction :=
  sorry

end solve_chair_table_fraction_l28_28644


namespace min_value_a_plus_b_l28_28008

theorem min_value_a_plus_b (a b : ℕ) (h₁ : 79 ∣ (a + 77 * b)) (h₂ : 77 ∣ (a + 79 * b)) : a + b = 193 :=
by
  sorry

end min_value_a_plus_b_l28_28008


namespace complex_number_solution_l28_28587

theorem complex_number_solution (z : ℂ) (i : ℂ) (h1 : i * z = (1 - 2 * i) ^ 2) (h2 : i * i = -1) : z = -4 + 3 * i := by
  sorry

end complex_number_solution_l28_28587


namespace find_number_of_packs_l28_28930

-- Define the cost of a pack of Digimon cards
def cost_pack_digimon : ℝ := 4.45

-- Define the cost of the deck of baseball cards
def cost_deck_baseball : ℝ := 6.06

-- Define the total amount spent
def total_spent : ℝ := 23.86

-- Define the number of packs of Digimon cards Keith bought
def number_of_packs (D : ℝ) : Prop :=
  cost_pack_digimon * D + cost_deck_baseball = total_spent

-- Prove the number of packs is 4
theorem find_number_of_packs : ∃ D, number_of_packs D ∧ D = 4 :=
by
  -- the proof will be inserted here
  sorry

end find_number_of_packs_l28_28930


namespace sea_lions_at_zoo_l28_28438

def ratio_sea_lions_to_penguins (S P : ℕ) : Prop := P = 11 * S / 4
def ratio_sea_lions_to_flamingos (S F : ℕ) : Prop := F = 7 * S / 4
def penguins_more_sea_lions (S P : ℕ) : Prop := P = S + 84
def flamingos_more_penguins (P F : ℕ) : Prop := F = P + 42

theorem sea_lions_at_zoo (S P F : ℕ)
  (h1 : ratio_sea_lions_to_penguins S P)
  (h2 : ratio_sea_lions_to_flamingos S F)
  (h3 : penguins_more_sea_lions S P)
  (h4 : flamingos_more_penguins P F) :
  S = 42 :=
sorry

end sea_lions_at_zoo_l28_28438


namespace max_value_quadratic_l28_28777

theorem max_value_quadratic (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : x * (1 - x) ≤ 1 / 4 :=
by
  sorry

end max_value_quadratic_l28_28777


namespace base_length_of_isosceles_triangle_l28_28038

noncomputable def isosceles_triangle_base_length (height : ℝ) (radius : ℝ) : ℝ :=
  if height = 25 ∧ radius = 8 then 80 / 3 else 0

theorem base_length_of_isosceles_triangle :
  isosceles_triangle_base_length 25 8 = 80 / 3 :=
by
  -- skipping the proof
  sorry

end base_length_of_isosceles_triangle_l28_28038


namespace average_cookies_l28_28374

theorem average_cookies (cookie_counts : List ℕ) (h : cookie_counts = [8, 10, 12, 15, 16, 17, 20]) :
  (cookie_counts.sum : ℚ) / cookie_counts.length = 14 := by
    -- Proof goes here
  sorry

end average_cookies_l28_28374


namespace evaluate_polynomial_103_l28_28454

theorem evaluate_polynomial_103 :
  103 ^ 4 - 4 * 103 ^ 3 + 6 * 103 ^ 2 - 4 * 103 + 1 = 108243216 :=
by
  sorry

end evaluate_polynomial_103_l28_28454


namespace female_students_count_l28_28256

variable (F : ℕ)

theorem female_students_count
    (avg_all_students : ℕ)
    (avg_male_students : ℕ)
    (avg_female_students : ℕ)
    (num_male_students : ℕ)
    (condition1 : avg_all_students = 90)
    (condition2 : avg_male_students = 82)
    (condition3 : avg_female_students = 92)
    (condition4 : num_male_students = 8)
    (condition5 : 8 * 82 + F * 92 = (8 + F) * 90) : 
    F = 32 := 
by 
  sorry

end female_students_count_l28_28256


namespace negation_of_exists_statement_l28_28544

theorem negation_of_exists_statement :
  ¬ (∃ x0 : ℝ, x0 > 0 ∧ x0^2 - 5 * x0 + 6 > 0) ↔ ∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0 :=
by
  sorry

end negation_of_exists_statement_l28_28544


namespace pq_problem_l28_28023

theorem pq_problem
  (p q : ℝ)
  (h1 : ∀ x : ℝ, (x - 7) * (2 * x + 11) = x^2 - 19 * x +  60)
  (h2 : p * q = 7 * (-9))
  (h3 : 7 + (-9) = -16):
  (p - 2) * (q - 2) = -55 :=
by
  sorry

end pq_problem_l28_28023


namespace sum_of_x_and_y_l28_28714

theorem sum_of_x_and_y (x y : ℝ) (h : x^2 + y^2 = 8*x - 10*y + 5) : x + y = -1 := by
  sorry

end sum_of_x_and_y_l28_28714


namespace number_of_classes_l28_28997

-- Define the conditions
def first_term : ℕ := 27
def common_diff : ℤ := -2
def total_students : ℕ := 115

-- Define and prove the main statement
theorem number_of_classes : ∃ n : ℕ, n > 0 ∧ (first_term + (n - 1) * common_diff) * n / 2 = total_students ∧ n = 5 :=
by
  sorry

end number_of_classes_l28_28997


namespace range_of_a_l28_28202

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≥ a → |x - 1| < 1) → (∃ x : ℝ, |x - 1| < 1 ∧ x < a) → a ≤ 0 := 
sorry

end range_of_a_l28_28202


namespace range_of_x_l28_28436

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) + 1

theorem range_of_x (x : ℝ) (h : f (2 * x - 1) + f (4 - x^2) > 2) : x ∈ Set.Ioo (-1 : ℝ) 3 :=
by
  sorry

end range_of_x_l28_28436


namespace rectangle_dimensions_l28_28257

theorem rectangle_dimensions (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h_area : x * y = 36) (h_perimeter : 2 * x + 2 * y = 30) : 
  (x = 12 ∧ y = 3) ∨ (x = 3 ∧ y = 12) :=
by
  sorry

end rectangle_dimensions_l28_28257


namespace gasoline_added_l28_28133

noncomputable def initial_amount (capacity: ℕ) : ℝ :=
  (3 / 4) * capacity

noncomputable def final_amount (capacity: ℕ) : ℝ :=
  (9 / 10) * capacity

theorem gasoline_added (capacity: ℕ) (initial_fraction final_fraction: ℝ) (initial_amount final_amount: ℝ) : 
  capacity = 54 ∧ initial_fraction = 3/4 ∧ final_fraction = 9/10 ∧ 
  initial_amount = initial_fraction * capacity ∧ 
  final_amount = final_fraction * capacity →
  final_amount - initial_amount = 8.1 :=
sorry

end gasoline_added_l28_28133


namespace madeline_water_intake_l28_28715

def water_bottle_capacity : ℕ := 12
def number_of_refills : ℕ := 7
def additional_water_needed : ℕ := 16
def total_water_needed : ℕ := 100

theorem madeline_water_intake : water_bottle_capacity * number_of_refills + additional_water_needed = total_water_needed :=
by
  sorry

end madeline_water_intake_l28_28715


namespace blocks_eaten_correct_l28_28187

def initial_blocks : ℕ := 55
def remaining_blocks : ℕ := 26

-- How many blocks were eaten by the hippopotamus?
def blocks_eaten_by_hippopotamus : ℕ := initial_blocks - remaining_blocks

theorem blocks_eaten_correct :
  blocks_eaten_by_hippopotamus = 29 := by
  sorry

end blocks_eaten_correct_l28_28187


namespace speed_of_man_in_still_water_l28_28144

theorem speed_of_man_in_still_water (v_m v_s : ℝ) (h1 : v_m + v_s = 6.2) (h2 : v_m - v_s = 6) : v_m = 6.1 :=
by
  sorry

end speed_of_man_in_still_water_l28_28144


namespace opposite_of_2023_is_neg_2023_l28_28691

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l28_28691


namespace binary_arithmetic_l28_28932

theorem binary_arithmetic :
  let a := 0b11101
  let b := 0b10011
  let c := 0b101
  (a * b) / c = 0b11101100 :=
by
  sorry

end binary_arithmetic_l28_28932


namespace complex_number_solution_l28_28643

open Complex

theorem complex_number_solution (z : ℂ) (h : (z - 2 * I) / z = 2 + I) :
  im z = -1 ∧ abs z = Real.sqrt 2 ∧ z ^ 6 = -8 * I :=
by
  sorry

end complex_number_solution_l28_28643


namespace simplify_and_evaluate_expression_l28_28175

theorem simplify_and_evaluate_expression (x y : ℝ) (h1 : x = 1 / 2) (h2 : y = 2023) :
  (x + y)^2 + (x + y) * (x - y) - 2 * x^2 = 2023 :=
by
  sorry

end simplify_and_evaluate_expression_l28_28175


namespace prime_pairs_l28_28599

-- Define the predicate to check whether a number is a prime.
def is_prime (n : Nat) : Prop := Nat.Prime n

-- Define the main theorem.
theorem prime_pairs (p q : Nat) (hp : is_prime p) (hq : is_prime q) : 
  (p^3 - q^5 = (p + q)^2) → (p = 7 ∧ q = 3) :=
by
  sorry

end prime_pairs_l28_28599


namespace isosceles_right_triangle_area_l28_28220

theorem isosceles_right_triangle_area (h : ℝ) (l : ℝ) (A : ℝ)
  (h_def : h = 6 * Real.sqrt 2)
  (rel_leg_hypotenuse : h = Real.sqrt 2 * l)
  (area_def : A = 1 / 2 * l * l) :
  A = 18 :=
by
  sorry

end isosceles_right_triangle_area_l28_28220


namespace michael_will_meet_two_times_l28_28456

noncomputable def michael_meetings : ℕ :=
  let michael_speed := 6 -- feet per second
  let pail_distance := 300 -- feet
  let truck_speed := 12 -- feet per second
  let truck_stop_time := 20 -- seconds
  let initial_distance := pail_distance -- feet
  let michael_position (t: ℕ) := michael_speed * t
  let truck_position (cycle: ℕ) := pail_distance * cycle
  let truck_cycle_time := pail_distance / truck_speed + truck_stop_time -- seconds per cycle
  let truck_position_at_time (t: ℕ) := 
    let cycle := t / truck_cycle_time
    let remaining_time := t % truck_cycle_time
    if remaining_time < (pail_distance / truck_speed) then 
      truck_position cycle + truck_speed * remaining_time
    else 
      truck_position cycle + pail_distance
  let distance_between := 
    λ (t: ℕ) => truck_position_at_time t - michael_position t
  let meet_time := 
    λ (t: ℕ) => if distance_between t = 0 then 1 else 0
  let total_meetings := 
    (List.range 300).map meet_time -- estimating within 300 seconds
    |> List.sum
  total_meetings

theorem michael_will_meet_two_times : michael_meetings = 2 :=
  sorry

end michael_will_meet_two_times_l28_28456


namespace number_in_circle_Y_l28_28891

section
variables (a b c d X Y : ℕ)

theorem number_in_circle_Y :
  a + b + X = 30 ∧
  c + d + Y = 30 ∧
  a + b + c + d = 40 ∧
  X + Y + c + b = 40 ∧
  X = 9 → Y = 11 := by
  intros h
  sorry
end

end number_in_circle_Y_l28_28891


namespace line_equation_l28_28122

noncomputable def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let norm_u_sq := u.1 * u.1 + u.2 * u.2
  (dot_uv / norm_u_sq) • u

theorem line_equation :
  ∀ (x y : ℝ), projection (4, 3) (x, y) = (-4, -3) → y = (-4 / 3) * x - 25 / 3 :=
by
  intros x y h
  sorry

end line_equation_l28_28122


namespace tetris_blocks_form_square_l28_28369

-- Definitions of Tetris blocks types
inductive TetrisBlock
| A | B | C | D | E | F | G

open TetrisBlock

-- Definition of a block's ability to form a square
def canFormSquare (block: TetrisBlock) : Prop :=
  block = A ∨ block = B ∨ block = C ∨ block = D ∨ block = G

-- The main theorem statement
theorem tetris_blocks_form_square : ∀ (block : TetrisBlock), canFormSquare block → block = A ∨ block = B ∨ block = C ∨ block = D ∨ block = G := 
by
  intros block h
  exact h

end tetris_blocks_form_square_l28_28369


namespace intersection_is_correct_l28_28634

noncomputable def A : Set ℝ := { x : ℝ | x^2 - x - 2 ≤ 0 }
noncomputable def B : Set ℝ := { x : ℝ | x < 1 }

theorem intersection_is_correct : (A ∩ B) = { x : ℝ | -1 ≤ x ∧ x < 1 } :=
by
  sorry

end intersection_is_correct_l28_28634


namespace segments_count_l28_28365

/--
Given two concentric circles, with chords of the larger circle that are tangent to the smaller circle,
if each chord subtends an angle of 80 degrees at the center, then the number of such segments 
drawn before returning to the starting point is 18.
-/
theorem segments_count (angle_ABC : ℝ) (circumference_angle_sum : ℝ → ℝ) (n m : ℕ) :
  angle_ABC = 80 → 
  circumference_angle_sum angle_ABC = 360 → 
  100 * n = 360 * m → 
  5 * n = 18 * m →
  n = 18 :=
by
  intros h1 h2 h3 h4
  sorry

end segments_count_l28_28365


namespace reflection_line_equation_l28_28793

-- Given condition 1: Original line equation
def original_line (x : ℝ) : ℝ := -2 * x + 7

-- Given condition 2: Reflection line
def reflection_line_x : ℝ := 3

-- Proving statement
theorem reflection_line_equation
  (a b : ℝ)
  (h₁ : a = -(-2))
  (h₂ : original_line 3 = 1)
  (h₃ : 1 = a * 3 + b) :
  2 * a + b = -1 :=
  sorry

end reflection_line_equation_l28_28793


namespace alex_candles_left_l28_28518

theorem alex_candles_left (candles_start used_candles : ℕ) (h1 : candles_start = 44) (h2 : used_candles = 32) :
  candles_start - used_candles = 12 :=
by
  sorry

end alex_candles_left_l28_28518


namespace find_n_such_that_4_pow_n_plus_5_pow_n_is_perfect_square_l28_28049

theorem find_n_such_that_4_pow_n_plus_5_pow_n_is_perfect_square :
  ∃ n : ℕ, (4^n + 5^n) = k^2 ↔ n = 1 :=
by
  sorry

end find_n_such_that_4_pow_n_plus_5_pow_n_is_perfect_square_l28_28049


namespace mod_last_digit_l28_28779

theorem mod_last_digit (N : ℕ) (a b : ℕ) (h : N = 10 * a + b) (hb : b < 10) : 
  N % 10 = b ∧ N % 2 = b % 2 ∧ N % 5 = b % 5 :=
by
  sorry

end mod_last_digit_l28_28779


namespace oil_used_l28_28658

theorem oil_used (total_weight : ℕ) (ratio_oil_peanuts : ℕ) (ratio_total_parts : ℕ) 
  (ratio_peanuts : ℕ) (ratio_parts : ℕ) (peanuts_weight : ℕ) : 
  ratio_oil_peanuts = 2 → 
  ratio_peanuts = 8 → 
  ratio_total_parts = 10 → 
  ratio_parts = 20 →
  peanuts_weight = total_weight / ratio_total_parts →
  total_weight = 20 → 
  2 * peanuts_weight = 4 :=
by sorry

end oil_used_l28_28658


namespace ratio_of_building_heights_l28_28928

theorem ratio_of_building_heights (F_h F_s A_s B_s : ℝ) (hF_h : F_h = 18) (hF_s : F_s = 45)
  (hA_s : A_s = 60) (hB_s : B_s = 72) :
  let h_A := (F_h / F_s) * A_s
  let h_B := (F_h / F_s) * B_s
  (h_A / h_B) = 5 / 6 :=
by
  sorry

end ratio_of_building_heights_l28_28928


namespace mike_baseball_cards_l28_28306

theorem mike_baseball_cards :
  let InitialCards : ℕ := 87
  let BoughtCards : ℕ := 13
  (InitialCards - BoughtCards = 74)
:= by
  sorry

end mike_baseball_cards_l28_28306


namespace fifth_term_of_sequence_l28_28445

theorem fifth_term_of_sequence :
  let a_n (n : ℕ) := (-1:ℤ)^(n+1) * (n^2 + 1)
  ∃ x : ℤ, a_n 5 * x^5 = 26 * x^5 :=
by
  sorry

end fifth_term_of_sequence_l28_28445


namespace cylinder_ratio_max_volume_l28_28421

theorem cylinder_ratio_max_volume 
    (l w : ℝ) 
    (r : ℝ) 
    (h : ℝ)
    (H_perimeter : 2 * l + 2 * w = 12)
    (H_length_circumference : l = 2 * π * r)
    (H_width_height : w = h) :
    (∀ V : ℝ, V = π * r^2 * h) →
    (∀ r : ℝ, r = 2 / π) →
    ((2 * π * r) / h = 2) :=
sorry

end cylinder_ratio_max_volume_l28_28421


namespace functional_equation_solution_l28_28847

open Function

theorem functional_equation_solution :
  ∀ (f g : ℚ → ℚ), 
    (∀ x y : ℚ, f (g x + g y) = f (g x) + y ∧ g (f x + f y) = g (f x) + y) →
    (∃ a b : ℚ, (ab = 1) ∧ (∀ x : ℚ, f x = a * x) ∧ (∀ x : ℚ, g x = b * x)) :=
by
  intros f g h
  sorry

end functional_equation_solution_l28_28847


namespace fifth_iteration_perimeter_l28_28276

theorem fifth_iteration_perimeter :
  let A1_side_length := 1
  let P1 := 3 * A1_side_length
  let P2 := 3 * (A1_side_length * 4 / 3)
  ∀ n : ℕ, P_n = 3 * (4 / 3) ^ (n - 1) →
  P_5 = 3 * (4 / 3) ^ 4 :=
  by sorry

end fifth_iteration_perimeter_l28_28276


namespace solve_for_a_l28_28794

theorem solve_for_a (S P Q R : Type) (a b c d : ℝ) 
  (h1 : a + b + c + d = 360)
  (h2 : ∀ (PSQ : Type), d = 90) :
  a = 270 - b - c :=
by
  sorry

end solve_for_a_l28_28794


namespace translate_line_upwards_l28_28278

theorem translate_line_upwards {x y : ℝ} (h : y = -2 * x + 1) :
  y = -2 * x + 3 := by
  sorry

end translate_line_upwards_l28_28278


namespace rectangular_field_area_l28_28689

noncomputable def length : ℝ := 1.2
noncomputable def width : ℝ := (3/4) * length

theorem rectangular_field_area : (length * width = 1.08) :=
by 
  -- The proof steps would go here
  sorry

end rectangular_field_area_l28_28689


namespace probability_even_sum_is_correct_l28_28461

noncomputable def probability_even_sum : ℚ :=
  let p_even_first := (2 : ℚ) / 5
  let p_odd_first := (3 : ℚ) / 5
  let p_even_second := (1 : ℚ) / 4
  let p_odd_second := (3 : ℚ) / 4

  let p_both_even := p_even_first * p_even_second
  let p_both_odd := p_odd_first * p_odd_second

  p_both_even + p_both_odd

theorem probability_even_sum_is_correct : probability_even_sum = 11 / 20 := by
  sorry

end probability_even_sum_is_correct_l28_28461


namespace squares_with_equal_black_and_white_cells_l28_28418

open Nat

/-- Given a specific coloring of cells in a 5x5 grid, prove that there are
exactly 16 squares that have an equal number of black and white cells. --/
theorem squares_with_equal_black_and_white_cells :
  let gridSize := 5
  let number_of_squares_with_equal_black_and_white_cells := 16
  true := sorry

end squares_with_equal_black_and_white_cells_l28_28418


namespace solve_quadratic_l28_28354

theorem solve_quadratic : ∀ x : ℝ, 2 * x^2 + 5 * x = 0 ↔ x = 0 ∨ x = -5/2 :=
by
  intro x
  sorry

end solve_quadratic_l28_28354


namespace parking_lot_length_l28_28600

theorem parking_lot_length (W : ℝ) (U : ℝ) (A_car : ℝ) (N_cars : ℕ) (H_w : W = 400) (H_u : U = 0.80) (H_Acar : A_car = 10) (H_Ncars : N_cars = 16000) :
  (U * (W * L) = N_cars * A_car) → (L = 500) :=
by
  sorry

end parking_lot_length_l28_28600


namespace curve_crosses_itself_l28_28430

-- Definitions of the parametric equations
def x (t k : ℝ) : ℝ := t^2 + k
def y (t k : ℝ) : ℝ := t^3 - k * t + 5

-- The main theorem statement
theorem curve_crosses_itself (k : ℝ) (ha : ℝ) (hb : ℝ) :
  ha ≠ hb →
  x ha k = x hb k →
  y ha k = y hb k →
  k = 9 ∧ x ha k = 18 ∧ y ha k = 5 :=
by
  sorry

end curve_crosses_itself_l28_28430


namespace square_of_binomial_l28_28735

theorem square_of_binomial (a : ℝ) :
  (∃ b : ℝ, (3 * x + b) ^ 2 = 9 * x^2 - 18 * x + a) ↔ a = 9 :=
by
  sorry

end square_of_binomial_l28_28735


namespace square_perimeter_l28_28665

variable (side : ℕ) (P : ℕ)

theorem square_perimeter (h : side = 19) : P = 4 * side → P = 76 := by
  intro hp
  rw [h] at hp
  norm_num at hp
  exact hp

end square_perimeter_l28_28665


namespace laura_mowing_time_correct_l28_28705

noncomputable def laura_mowing_time : ℝ := 
  let combined_time := 1.71428571429
  let sammy_time := 3
  let combined_rate := 1 / combined_time
  let sammy_rate := 1 / sammy_time
  let laura_rate := combined_rate - sammy_rate
  1 / laura_rate

theorem laura_mowing_time_correct : laura_mowing_time = 4.2 := 
  by
    sorry

end laura_mowing_time_correct_l28_28705


namespace algebraic_expression_value_l28_28318

theorem algebraic_expression_value (x y : ℝ) (h : x + 2 * y + 1 = 3) : 2 * x + 4 * y + 1 = 5 :=
by
  sorry

end algebraic_expression_value_l28_28318


namespace min_value_f_prime_at_2_l28_28536

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 2 * a * x^2 + (1/a) * x

noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 4*a*x + (1/a)

theorem min_value_f_prime_at_2 (a : ℝ) (h : a > 0) : 
  f_prime a 2 >= 12 + 4 * Real.sqrt 2 := 
by
  -- proof will be written here
  sorry

end min_value_f_prime_at_2_l28_28536


namespace inequality_holds_l28_28578

theorem inequality_holds (a b : ℝ) : (6 * a - 3 * b - 3) * (a ^ 2 + a ^ 2 * b - 2 * a ^ 3) ≤ 0 :=
sorry

end inequality_holds_l28_28578


namespace highland_high_students_highland_high_num_both_clubs_l28_28345

theorem highland_high_students (total_students drama_club science_club either_both both_clubs : ℕ)
  (h1 : total_students = 320)
  (h2 : drama_club = 90)
  (h3 : science_club = 140)
  (h4 : either_both = 200) : 
  both_clubs = drama_club + science_club - either_both :=
by
  sorry

noncomputable def num_both_clubs : ℕ :=
if h : 320 = 320 ∧ 90 = 90 ∧ 140 = 140 ∧ 200 = 200
then 90 + 140 - 200
else 0

theorem highland_high_num_both_clubs : num_both_clubs = 30 :=
by
  sorry

end highland_high_students_highland_high_num_both_clubs_l28_28345


namespace pilot_fish_speed_is_30_l28_28546

-- Define the initial conditions
def keanu_speed : ℝ := 20
def shark_initial_speed : ℝ := keanu_speed
def shark_speed_increase_factor : ℝ := 2
def pilot_fish_speed_increase_factor : ℝ := 0.5

-- Calculating final speeds
def shark_final_speed : ℝ := shark_initial_speed * shark_speed_increase_factor
def shark_speed_increase : ℝ := shark_final_speed - shark_initial_speed
def pilot_fish_speed_increase : ℝ := shark_speed_increase * pilot_fish_speed_increase_factor
def pilot_fish_final_speed : ℝ := keanu_speed + pilot_fish_speed_increase

-- The statement to prove
theorem pilot_fish_speed_is_30 : pilot_fish_final_speed = 30 := by
  sorry

end pilot_fish_speed_is_30_l28_28546


namespace average_selections_correct_l28_28120

noncomputable def cars := 18
noncomputable def selections_per_client := 3
noncomputable def clients := 18
noncomputable def total_selections := clients * selections_per_client
noncomputable def average_selections_per_car := total_selections / cars

theorem average_selections_correct :
  average_selections_per_car = 3 :=
by
  sorry

end average_selections_correct_l28_28120


namespace common_ratio_of_geometric_sequence_l28_28766

noncomputable def a_n (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

theorem common_ratio_of_geometric_sequence
  (a1 d : ℝ) (h1 : d ≠ 0)
  (h2 : (a_n a1 d 5) * (a_n a1 d 20) = (a_n a1 d 10) ^ 2) :
  (a_n a1 d 10) / (a_n a1 d 5) = 2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l28_28766


namespace sampling_method_selection_l28_28169

-- Define the sampling methods as data type
inductive SamplingMethod
| SimpleRandomSampling : SamplingMethod
| SystematicSampling : SamplingMethod
| StratifiedSampling : SamplingMethod
| SamplingWithReplacement : SamplingMethod

-- Define our conditions
def basketballs : Nat := 10
def is_random_selection : Bool := true
def no_obvious_stratification : Bool := true

-- The theorem to prove the correct sampling method
theorem sampling_method_selection 
  (b : Nat) 
  (random_selection : Bool) 
  (no_stratification : Bool) : 
  SamplingMethod :=
  if b = 10 ∧ random_selection ∧ no_stratification then SamplingMethod.SimpleRandomSampling 
  else sorry

-- Prove the correct sampling method given our conditions
example : sampling_method_selection basketballs is_random_selection no_obvious_stratification = SamplingMethod.SimpleRandomSampling := 
by
-- skipping the proof here with sorry
sorry

end sampling_method_selection_l28_28169


namespace solutions_diff_squared_l28_28817

theorem solutions_diff_squared (a b : ℝ) (h : 5 * a^2 - 6 * a - 55 = 0 ∧ 5 * b^2 - 6 * b - 55 = 0) :
  (a - b)^2 = 1296 / 25 := by
  sorry

end solutions_diff_squared_l28_28817


namespace weng_hourly_rate_l28_28849

theorem weng_hourly_rate (minutes_worked : ℝ) (earnings : ℝ) (fraction_of_hour : ℝ) 
  (conversion_rate : ℝ) (hourly_rate : ℝ) : 
  minutes_worked = 50 → earnings = 10 → 
  fraction_of_hour = minutes_worked / conversion_rate → 
  conversion_rate = 60 → 
  hourly_rate = earnings / fraction_of_hour → 
  hourly_rate = 12 := by
    sorry

end weng_hourly_rate_l28_28849


namespace distinct_arrangements_balloon_l28_28209

-- Let's define the basic conditions:
def total_letters : Nat := 7
def repeats_l : Nat := 2
def repeats_o : Nat := 2

-- Now let's state the problem.
theorem distinct_arrangements_balloon : 
  (Nat.factorial total_letters) / ((Nat.factorial repeats_l) * (Nat.factorial repeats_o)) = 1260 := 
by
  sorry

end distinct_arrangements_balloon_l28_28209


namespace min_value_frac_sum_l28_28916

theorem min_value_frac_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) : 
  (1 / x) + (4 / y) ≥ 9 :=
sorry

end min_value_frac_sum_l28_28916


namespace find_inequality_solution_l28_28407

theorem find_inequality_solution :
  {x : ℝ | (x + 1) / (x - 2) + (x + 3) / (2 * x + 1) ≤ 2}
  = {x : ℝ | -1 / 2 ≤ x ∧ x ≤ 1 ∨ 2 ≤ x ∧ x ≤ 9} :=
by
  -- The proof steps are omitted.
  sorry

end find_inequality_solution_l28_28407


namespace joe_height_l28_28494

theorem joe_height (S J A : ℝ) (h1 : S + J + A = 180) (h2 : J = 2 * S + 6) (h3 : A = S - 3) : J = 94.5 :=
by 
  -- Lean proof goes here
  sorry

end joe_height_l28_28494


namespace jose_completion_time_l28_28294

noncomputable def rate_jose : ℚ := 1 / 30
noncomputable def rate_jane : ℚ := 1 / 6

theorem jose_completion_time :
  ∀ (J A : ℚ), 
    (J + A = 1 / 5) ∧ (J = rate_jose) ∧ (A = rate_jane) → 
    (1 / J = 30) :=
by
  intros J A h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end jose_completion_time_l28_28294


namespace min_value_fraction_solve_inequality_l28_28951

-- Part 1
theorem min_value_fraction (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (f : ℝ → ℝ)
  (h3 : f 1 = 2) (h4 : ∀ x, f x = a * x^2 + b * x + 1) :
  (a + b = 1) → (∃ z, z = (1 / a + 4 / b) ∧ z = 9) := 
by {
  sorry
}

-- Part 2
theorem solve_inequality (a : ℝ) (x : ℝ) (h1 : b = -a - 1) (f : ℝ → ℝ)
  (h2 : ∀ x, f x = a * x^2 + b * x + 1) :
  (f x ≤ 0) → 
  (if a = 0 then 
      {x | x ≥ 1}
  else if a > 0 then
      if a = 1 then 
          {x | x = 1}
      else if 0 < a ∧ a < 1 then 
          {x | 1 ≤ x ∧ x ≤ 1 / a}
      else 
          {x | 1 / a ≤ x ∧ x ≤ 1}
  else 
      {x | x ≥ 1 ∨ x ≤ 1 / a}) :=
by {
  sorry
}

end min_value_fraction_solve_inequality_l28_28951


namespace smallest_positive_integer_x_l28_28938

theorem smallest_positive_integer_x :
  ∃ x : ℕ, 42 * x + 14 ≡ 4 [MOD 26] ∧ x ≡ 3 [MOD 5] ∧ x = 38 := 
by
  sorry

end smallest_positive_integer_x_l28_28938


namespace max_parts_by_rectangles_l28_28966

theorem max_parts_by_rectangles (n : ℕ) : 
  ∃ S : ℕ, S = 2 * n^2 - 2 * n + 2 :=
by
  sorry

end max_parts_by_rectangles_l28_28966


namespace min_sum_x1_x2_x3_x4_l28_28991

variables (x1 x2 x3 x4 : ℝ)

theorem min_sum_x1_x2_x3_x4 : 
  (x1 + x2 ≥ 12) → 
  (x1 + x3 ≥ 13) → 
  (x1 + x4 ≥ 14) → 
  (x3 + x4 ≥ 22) → 
  (x2 + x3 ≥ 23) → 
  (x2 + x4 ≥ 24) → 
  (x1 + x2 + x3 + x4 = 37) := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end min_sum_x1_x2_x3_x4_l28_28991


namespace rectangle_circle_area_ratio_l28_28371

theorem rectangle_circle_area_ratio (w r : ℝ) (h1 : 2 * 2 * w + 2 * w = 2 * pi * r) :
  ((2 * w) * w) / (pi * r^2) = 2 * pi / 9 :=
by
  sorry

end rectangle_circle_area_ratio_l28_28371


namespace sequence_general_term_l28_28441

theorem sequence_general_term (a : ℕ → ℝ) (h₁ : a 1 = 1) 
  (h₂ : ∀ n, a (n + 1) = 2^n * a n) : 
  ∀ n, a n = 2^((n-1)*n / 2) := sorry

end sequence_general_term_l28_28441


namespace extremum_values_of_function_l28_28243

noncomputable def maxValue := Real.sqrt 2 + 1 / Real.sqrt 2
noncomputable def minValue := -Real.sqrt 2 + 1 / Real.sqrt 2

theorem extremum_values_of_function :
  ∀ x : ℝ, - (Real.sqrt 2) + (1 / Real.sqrt 2) ≤ (Real.sin x + Real.cos x + 1 / Real.sqrt (1 + |Real.sin (2 * x)|)) ∧ 
            (Real.sin x + Real.cos x + 1 / Real.sqrt (1 + |Real.sin (2 * x)|)) ≤ (Real.sqrt 2 + 1 / Real.sqrt 2) := 
by
  sorry

end extremum_values_of_function_l28_28243


namespace sufficient_but_not_necessary_condition_l28_28994

variable (a : ℝ)

theorem sufficient_but_not_necessary_condition :
  (a > 2 → a^2 > 2 * a)
  ∧ (¬(a^2 > 2 * a → a > 2)) := by
  sorry

end sufficient_but_not_necessary_condition_l28_28994


namespace candy_left_l28_28675

theorem candy_left (total_candy : ℕ) (eaten_per_person : ℕ) (number_of_people : ℕ)
  (h_total_candy : total_candy = 68)
  (h_eaten_per_person : eaten_per_person = 4)
  (h_number_of_people : number_of_people = 2) :
  total_candy - (eaten_per_person * number_of_people) = 60 :=
by
  sorry

end candy_left_l28_28675


namespace negation_proof_l28_28153

open Real

theorem negation_proof :
  (¬ ∃ x : ℕ, exp x - x - 1 ≤ 0) ↔ (∀ x : ℕ, exp x - x - 1 > 0) :=
by
  sorry

end negation_proof_l28_28153


namespace johns_outfit_cost_l28_28896

theorem johns_outfit_cost (pants_cost shirt_cost outfit_cost : ℝ)
    (h_pants : pants_cost = 50)
    (h_shirt : shirt_cost = pants_cost + 0.6 * pants_cost)
    (h_outfit : outfit_cost = pants_cost + shirt_cost) :
    outfit_cost = 130 :=
by
  sorry

end johns_outfit_cost_l28_28896


namespace james_selling_price_l28_28782

variable (P : ℝ)  -- Selling price per candy bar

theorem james_selling_price 
  (boxes_sold : ℕ)
  (candy_bars_per_box : ℕ) 
  (cost_price_per_candy_bar : ℝ)
  (total_profit : ℝ)
  (H1 : candy_bars_per_box = 10)
  (H2 : boxes_sold = 5)
  (H3 : cost_price_per_candy_bar = 1)
  (H4 : total_profit = 25)
  (profit_eq : boxes_sold * candy_bars_per_box * (P - cost_price_per_candy_bar) = total_profit)
  : P = 1.5 :=
by 
  sorry

end james_selling_price_l28_28782


namespace corrected_mean_l28_28279

theorem corrected_mean (mean : ℝ) (num_observations : ℕ) 
  (incorrect_observation correct_observation : ℝ)
  (h_mean : mean = 36) (h_num_observations : num_observations = 50)
  (h_incorrect_observation : incorrect_observation = 23) 
  (h_correct_observation : correct_observation = 44)
  : (mean * num_observations + (correct_observation - incorrect_observation)) / num_observations = 36.42 := 
by
  sorry

end corrected_mean_l28_28279


namespace find_number_l28_28487

-- Define given numbers
def a : ℕ := 555
def b : ℕ := 445

-- Define given conditions
def sum : ℕ := a + b
def difference : ℕ := a - b
def quotient : ℕ := 2 * difference
def remainder : ℕ := 30

-- Define the number we're looking for
def number := sum * quotient + remainder

-- The theorem to prove
theorem find_number : number = 220030 := by
  -- Use the let expressions to simplify the calculation for clarity
  let sum := a + b
  let difference := a - b
  let quotient := 2 * difference
  let number := sum * quotient + remainder
  show number = 220030
  -- Placeholder for proof
  sorry

end find_number_l28_28487


namespace cost_per_page_first_time_l28_28404

-- Definitions based on conditions
variables (num_pages : ℕ) (rev_once_pages : ℕ) (rev_twice_pages : ℕ)
variables (rev_cost : ℕ) (total_cost : ℕ)
variables (first_time_cost : ℕ)

-- Conditions
axiom h1 : num_pages = 100
axiom h2 : rev_once_pages = 35
axiom h3 : rev_twice_pages = 15
axiom h4 : rev_cost = 4
axiom h5 : total_cost = 860

-- Proof statement: Prove that the cost per page for the first time a page is typed is $6
theorem cost_per_page_first_time : first_time_cost = 6 :=
sorry

end cost_per_page_first_time_l28_28404


namespace division_value_l28_28408

theorem division_value (n x : ℝ) (h₀ : n = 4.5) (h₁ : (n / x) * 12 = 9) : x = 6 :=
by
  sorry

end division_value_l28_28408


namespace max_value_of_n_l28_28995

theorem max_value_of_n : 
  ∃ n : ℕ, 
    (∀ m : ℕ, m ≤ n → (2 / 3)^(m - 1) * (1 / 3) ≥ 1 / 60) 
      ∧ 
    (∀ k : ℕ, k > n → (2 / 3)^(k - 1) * (1 / 3) < 1 / 60) 
      ∧ 
    n = 8 :=
by
  sorry

end max_value_of_n_l28_28995


namespace train_length_l28_28978

noncomputable def speed_kph := 56  -- speed in km/hr
def time_crossing := 9  -- time in seconds
noncomputable def speed_mps := speed_kph * 1000 / 3600  -- converting km/hr to m/s

theorem train_length : speed_mps * time_crossing = 140 := by
  -- conversion and result approximation
  sorry

end train_length_l28_28978


namespace arithmetic_sequence_150th_term_l28_28028

theorem arithmetic_sequence_150th_term :
  let a₁ := 3
  let d := 4
  a₁ + (150 - 1) * d = 599 :=
by
  sorry

end arithmetic_sequence_150th_term_l28_28028


namespace A_ge_B_l28_28682

def A (a b : ℝ) : ℝ := a^3 + 3 * a^2 * b^2 + 2 * b^2 + 3 * b
def B (a b : ℝ) : ℝ := a^3 - a^2 * b^2 + b^2 + 3 * b

theorem A_ge_B (a b : ℝ) : A a b ≥ B a b := by
  sorry

end A_ge_B_l28_28682


namespace weight_of_11m_rebar_l28_28801

theorem weight_of_11m_rebar (w5m : ℝ) (l5m : ℝ) (l11m : ℝ) 
  (h_w5m : w5m = 15.3) (h_l5m : l5m = 5) (h_l11m : l11m = 11) : 
  (w5m / l5m) * l11m = 33.66 := 
by {
  sorry
}

end weight_of_11m_rebar_l28_28801


namespace polyhedron_space_diagonals_l28_28405

theorem polyhedron_space_diagonals (V E F T Q P : ℕ) (hV : V = 30) (hE : E = 70) (hF : F = 42)
                                    (hT : T = 26) (hQ : Q = 12) (hP : P = 4) : 
  ∃ D : ℕ, D = 321 :=
by
  have total_pairs := (30 * 29) / 2
  have triangular_face_diagonals := 0
  have quadrilateral_face_diagonals := 12 * 2
  have pentagon_face_diagonals := 4 * 5
  have total_face_diagonals := triangular_face_diagonals + quadrilateral_face_diagonals + pentagon_face_diagonals
  have total_edges_and_diagonals := total_pairs - 70 - total_face_diagonals
  use total_edges_and_diagonals
  sorry

end polyhedron_space_diagonals_l28_28405


namespace carlos_meeting_percentage_l28_28879

-- Definitions for the given conditions
def work_day_minutes : ℕ := 10 * 60
def first_meeting_minutes : ℕ := 80
def second_meeting_minutes : ℕ := 3 * first_meeting_minutes
def break_minutes : ℕ := 15
def total_meeting_and_break_minutes : ℕ := first_meeting_minutes + second_meeting_minutes + break_minutes

-- Statement to prove
theorem carlos_meeting_percentage : 
  (total_meeting_and_break_minutes * 100 / work_day_minutes) = 56 := 
by
  sorry

end carlos_meeting_percentage_l28_28879


namespace passing_grade_fraction_l28_28366

theorem passing_grade_fraction (A B C D F : ℚ) (hA : A = 1/4) (hB : B = 1/2) (hC : C = 1/8) (hD : D = 1/12) (hF : F = 1/24) : 
  A + B + C = 7/8 :=
by
  sorry

end passing_grade_fraction_l28_28366


namespace only_n_equal_1_is_natural_number_for_which_2_pow_n_plus_n_pow_2016_is_prime_l28_28934

theorem only_n_equal_1_is_natural_number_for_which_2_pow_n_plus_n_pow_2016_is_prime (n : ℕ) : 
  Prime (2^n + n^2016) ↔ n = 1 := by
  sorry

end only_n_equal_1_is_natural_number_for_which_2_pow_n_plus_n_pow_2016_is_prime_l28_28934


namespace algebraic_expression_value_l28_28400

theorem algebraic_expression_value (p q : ℝ) 
  (h : p * 3^3 + q * 3 + 1 = 2015) : 
  p * (-3)^3 + q * (-3) + 1 = -2013 :=
by 
  sorry

end algebraic_expression_value_l28_28400


namespace third_vertex_l28_28315

/-- Two vertices of a right triangle are located at (4, 3) and (0, 0).
The third vertex of the triangle lies on the positive branch of the x-axis.
Determine the coordinates of the third vertex if the area of the triangle is 24 square units. -/
theorem third_vertex (x : ℝ) (h : x > 0) : 
  (1 / 2 * |x| * 3 = 24) → (x, 0) = (16, 0) :=
by
  intro h_area
  sorry

end third_vertex_l28_28315


namespace initial_reading_times_per_day_l28_28424

-- Definitions based on the conditions

/-- Number of pages Jessy plans to read initially in each session is 6. -/
def session_pages : ℕ := 6

/-- Jessy needs to read 140 pages in one week. -/
def total_pages : ℕ := 140

/-- Jessy reads an additional 2 pages per day to achieve her goal. -/
def additional_daily_pages : ℕ := 2

/-- Days in a week -/
def days_in_week : ℕ := 7

-- Proving Jessy's initial plan for reading times per day
theorem initial_reading_times_per_day (x : ℕ) (h : days_in_week * (session_pages * x + additional_daily_pages) = total_pages) : 
    x = 3 := by
  -- skipping the proof itself
  sorry

end initial_reading_times_per_day_l28_28424


namespace pie_charts_cannot_show_changes_l28_28208

def pie_chart_shows_part_whole (P : Type) := true
def bar_chart_shows_amount (B : Type) := true
def line_chart_shows_amount_and_changes (L : Type) := true

theorem pie_charts_cannot_show_changes (P B L : Type) :
  pie_chart_shows_part_whole P ∧ bar_chart_shows_amount B ∧ line_chart_shows_amount_and_changes L →
  ¬ (pie_chart_shows_part_whole P ∧ ¬ line_chart_shows_amount_and_changes P) :=
by sorry

end pie_charts_cannot_show_changes_l28_28208


namespace factor_27x6_minus_512y6_sum_coeffs_is_152_l28_28300

variable {x y : ℤ}

theorem factor_27x6_minus_512y6_sum_coeffs_is_152 :
  ∃ a b c d e f g h j k : ℤ, 
    (27 * x^6 - 512 * y^6 = (a * x + b * y) * (c * x^2 + d * x * y + e * y^2) * (f * x + g * y) * (h * x^2 + j * x * y + k * y^2)) ∧ 
    (a + b + c + d + e + f + g + h + j + k = 152) := 
sorry

end factor_27x6_minus_512y6_sum_coeffs_is_152_l28_28300


namespace part_a_part_b_part_c_l28_28846

variable (N : ℕ) (r : Fin N → Fin N → ℝ)

-- Part (a)
theorem part_a (h : ∀ (s : Finset (Fin N)), s.card = 5 → (exists pts : s → ℝ × ℝ, ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j)) :
  ∃ pts : Fin N → ℝ × ℝ, ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j :=
sorry

-- Part (b)
theorem part_b (h : ∀ (s : Finset (Fin N)), s.card = 4 → (exists pts : s → ℝ × ℝ, ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j)) :
  ¬ (∃ pts : Fin N → ℝ × ℝ, ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j) :=
sorry

-- Part (c)
theorem part_c (h : ∀ (s : Finset (Fin N)), s.card = 6 → (exists pts : s → ℝ × ℝ × ℝ, ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j)) :
  ∃ (pts : Fin N → ℝ × ℝ × ℝ), ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j :=
sorry

end part_a_part_b_part_c_l28_28846


namespace find_b_l28_28617

theorem find_b
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : ∀ x y : ℝ, (x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1) -> True)
  (h4 : ∃ e, e = (Real.sqrt 5) / 3)
  (h5 : 2 * a = 12) :
  b = 4 :=
by
  sorry

end find_b_l28_28617


namespace percentage_correct_l28_28976

theorem percentage_correct (x : ℕ) (h : x > 0) : 
  (4 * x / (6 * x) * 100 = 200 / 3) :=
by
  sorry

end percentage_correct_l28_28976


namespace factorization_of_x_squared_minus_64_l28_28304

theorem factorization_of_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := 
by 
  sorry

end factorization_of_x_squared_minus_64_l28_28304


namespace probability_f4_positive_l28_28275

theorem probability_f4_positive {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = -f x)
  (h_fn : ∀ x < 0, f x = a + x + Real.logb 2 (-x)) (h_a : a > -4 ∧ a < 5) :
  (1/3 : ℝ) < (2/3 : ℝ) :=
sorry

end probability_f4_positive_l28_28275


namespace irrational_pi_l28_28699

def is_irrational (x : ℝ) : Prop := ¬ ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

theorem irrational_pi : is_irrational π := by
  sorry

end irrational_pi_l28_28699


namespace average_age_group_l28_28019

theorem average_age_group (n : ℕ) (T : ℕ) (h1 : T = n * 14) (h2 : T + 32 = (n + 1) * 15) : n = 17 :=
by
  sorry

end average_age_group_l28_28019


namespace all_terms_are_integers_l28_28539

open Nat

def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 143 ∧ ∀ n ≥ 2, a (n + 1) = 5 * (Finset.range n).sum a / n

theorem all_terms_are_integers (a : ℕ → ℕ) (h : seq a) : ∀ n : ℕ, 1 ≤ n → ∃ k : ℕ, a n = k := 
by
  sorry

end all_terms_are_integers_l28_28539


namespace car_distance_kilometers_l28_28024

theorem car_distance_kilometers (d_amar : ℝ) (d_car : ℝ) (ratio : ℝ) (total_d_amar : ℝ) :
  d_amar = 24 ->
  d_car = 60 ->
  ratio = 2 / 5 ->
  total_d_amar = 880 ->
  (d_car / d_amar) = 5 / 2 ->
  (total_d_amar * 5 / 2) / 1000 = 2.2 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end car_distance_kilometers_l28_28024


namespace A_inter_B_empty_iff_A_union_B_eq_B_iff_l28_28040

open Set

variable (a x : ℝ)

def A (a : ℝ) : Set ℝ := {x | 1 - a ≤ x ∧ x ≤ 1 + a}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

theorem A_inter_B_empty_iff {a : ℝ} :
  (A a ∩ B = ∅) ↔ 0 ≤ a ∧ a ≤ 4 :=
by 
  sorry

theorem A_union_B_eq_B_iff {a : ℝ} :
  (A a ∪ B = B) ↔ a < -4 :=
by
  sorry

end A_inter_B_empty_iff_A_union_B_eq_B_iff_l28_28040


namespace batsman_average_after_17th_inning_l28_28601

theorem batsman_average_after_17th_inning :
  ∃ x : ℤ, (63 + (16 * x) = 17 * (x + 3)) ∧ (x + 3 = 17) :=
by
  sorry

end batsman_average_after_17th_inning_l28_28601


namespace problem_b_l28_28755

theorem problem_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b + a * b = 3) : a + b ≥ 2 :=
sorry

end problem_b_l28_28755


namespace lengthDE_is_correct_l28_28015

noncomputable def triangleBase : ℝ := 12

noncomputable def triangleArea (h : ℝ) : ℝ := (1 / 2) * triangleBase * h

noncomputable def projectedArea (h : ℝ) : ℝ := 0.16 * triangleArea h

noncomputable def lengthDE (h : ℝ) : ℝ := 0.4 * triangleBase

theorem lengthDE_is_correct (h : ℝ) :
  lengthDE h = 4.8 :=
by
  simp [lengthDE, triangleBase, triangleArea, projectedArea]
  sorry

end lengthDE_is_correct_l28_28015


namespace range_of_a_l28_28041

noncomputable def f (x a : ℝ) := (x^2 + a * x + 11) / (x + 1)

theorem range_of_a (a : ℝ) :
  (∀ x : ℕ, x > 0 → f x a ≥ 3) ↔ (a ≥ -8 / 3) :=
by sorry

end range_of_a_l28_28041


namespace find_x_value_l28_28098

-- Define the condition as a hypothesis
def condition (x : ℝ) : Prop := (x / 4) - x - (3 / 6) = 1

-- State the theorem
theorem find_x_value (x : ℝ) (h : condition x) : x = -2 := 
by sorry

end find_x_value_l28_28098


namespace cone_volume_is_correct_l28_28652

theorem cone_volume_is_correct (r l h : ℝ) 
  (h1 : 2 * r = Real.sqrt 2 * l)
  (h2 : π * r * l = 16 * Real.sqrt 2 * π)
  (h3 : h = r) : 
  (1 / 3) * π * r ^ 2 * h = (64 / 3) * π :=
by sorry

end cone_volume_is_correct_l28_28652


namespace science_club_officers_l28_28842

-- Definitions of the problem conditions
def num_members : ℕ := 25
def num_officers : ℕ := 3
def alice : ℕ := 1 -- unique identifier for Alice
def bob : ℕ := 2 -- unique identifier for Bob

-- Main theorem statement
theorem science_club_officers :
  ∃ (ways_to_choose_officers : ℕ), ways_to_choose_officers = 10764 :=
  sorry

end science_club_officers_l28_28842


namespace flagpole_height_l28_28037

theorem flagpole_height (x : ℝ) (h1 : (x + 2)^2 = x^2 + 6^2) : x = 8 := 
by 
  sorry

end flagpole_height_l28_28037


namespace highlighter_count_l28_28712

-- Define the quantities of highlighters.
def pinkHighlighters := 3
def yellowHighlighters := 7
def blueHighlighters := 5

-- Define the total number of highlighters.
def totalHighlighters := pinkHighlighters + yellowHighlighters + blueHighlighters

-- The theorem states that the total number of highlighters is 15.
theorem highlighter_count : totalHighlighters = 15 := by
  -- Proof skipped for now.
  sorry

end highlighter_count_l28_28712


namespace germination_percentage_in_second_plot_l28_28001

theorem germination_percentage_in_second_plot
     (seeds_first_plot : ℕ := 300)
     (seeds_second_plot : ℕ := 200)
     (germination_first_plot : ℕ := 75)
     (total_seeds : ℕ := 500)
     (germination_total : ℕ := 155)
     (x : ℕ := 40) :
  (x : ℕ) = (80 / 2) := by
  -- Provided conditions, skipping the proof part with sorry
  have h1 : 75 = 0.25 * 300 := sorry
  have h2 : 500 = 300 + 200 := sorry
  have h3 : 155 = 0.31 * 500 := sorry
  have h4 : 80 = 155 - 75 := sorry
  have h5 : x = (80 / 2) := sorry
  exact h5

end germination_percentage_in_second_plot_l28_28001


namespace yield_percentage_l28_28670

theorem yield_percentage (d : ℝ) (q : ℝ) (f : ℝ) : d = 12 → q = 150 → f = 100 → (d * f / q) * 100 = 8 :=
by
  intros h_d h_q h_f
  rw [h_d, h_q, h_f]
  sorry

end yield_percentage_l28_28670


namespace problem1_problem2_l28_28859

-- Definitions of the polynomials A and B
def A (x y : ℝ) := x^2 + x * y + 3 * y
def B (x y : ℝ) := x^2 - x * y

-- Problem 1 Statement: 
theorem problem1 (x y : ℝ) (h : (x - 2)^2 + |y + 5| = 0) : 2 * (A x y) - (B x y) = -56 := by
  sorry

-- Problem 2 Statement:
theorem problem2 (x : ℝ) (h : ∀ y, 2 * (A x y) - (B x y) = 0) : x = -2 := by
  sorry

end problem1_problem2_l28_28859


namespace water_current_speed_l28_28325

theorem water_current_speed (v : ℝ) (swimmer_speed : ℝ := 4) (time : ℝ := 3.5) (distance : ℝ := 7) :
  (4 - v) = distance / time → v = 2 := 
by
  sorry

end water_current_speed_l28_28325


namespace jo_page_an_hour_ago_l28_28447

variables (total_pages current_page hours_left : ℕ)
variables (steady_reading_rate : ℕ)
variables (page_an_hour_ago : ℕ)

-- Conditions
def conditions := 
  steady_reading_rate * hours_left = total_pages - current_page ∧
  total_pages = 210 ∧
  current_page = 90 ∧
  hours_left = 4 ∧
  page_an_hour_ago = current_page - steady_reading_rate

-- Theorem to prove that Jo was on page 60 an hour ago
theorem jo_page_an_hour_ago (h : conditions total_pages current_page hours_left steady_reading_rate page_an_hour_ago) : 
  page_an_hour_ago = 60 :=
sorry

end jo_page_an_hour_ago_l28_28447


namespace geometric_sequence_problem_l28_28659

theorem geometric_sequence_problem 
  (a : ℕ → ℝ)
  (h_geo : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n)
  (h1 : a 7 * a 11 = 6)
  (h2 : a 4 + a 14 = 5) :
  ∃ x : ℝ, x = 2 / 3 ∨ x = 3 / 2 := by
  sorry

end geometric_sequence_problem_l28_28659


namespace false_proposition_l28_28900

-- Definitions of the conditions
def p1 := ∃ x0 : ℝ, x0^2 - 2*x0 + 1 ≤ 0
def p2 := ∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → x^2 - 1 ≥ 0

-- Statement to prove
theorem false_proposition : ¬ (¬ p1 ∧ ¬ p2) :=
by sorry

end false_proposition_l28_28900


namespace electricity_price_increase_percentage_l28_28769

noncomputable def old_power_kW : ℝ := 0.8
noncomputable def additional_power_percent : ℝ := 50 / 100
noncomputable def old_price_per_kWh : ℝ := 0.12
noncomputable def cost_for_50_hours : ℝ := 9
noncomputable def total_hours : ℝ := 50
noncomputable def energy_consumed := old_power_kW * total_hours

theorem electricity_price_increase_percentage :
  ∃ P : ℝ, 
    (energy_consumed * P = cost_for_50_hours) ∧
    ((P - old_price_per_kWh) / old_price_per_kWh) * 100 = 87.5 :=
by
  sorry

end electricity_price_increase_percentage_l28_28769


namespace factorial_division_l28_28464

theorem factorial_division (N : Nat) (h : N ≥ 2) : 
  (Nat.factorial (2 * N)) / ((Nat.factorial (N + 2)) * (Nat.factorial (N - 2))) = 
  (List.prod (List.range' (N + 3) (2 * N - (N + 2) + 1))) / (Nat.factorial (N - 1)) :=
sorry

end factorial_division_l28_28464


namespace pentagon_area_calc_l28_28299

noncomputable def pentagon_area : ℝ :=
  let triangle1 := (1 / 2) * 18 * 22
  let triangle2 := (1 / 2) * 30 * 26
  let trapezoid := (1 / 2) * (22 + 30) * 10
  triangle1 + triangle2 + trapezoid

theorem pentagon_area_calc :
  pentagon_area = 848 := by
  sorry

end pentagon_area_calc_l28_28299


namespace chocolates_for_sister_l28_28835
-- Importing necessary library

-- Lean 4 statement of the problem
theorem chocolates_for_sister (S : ℕ) 
  (herself_chocolates_per_saturday : ℕ := 2)
  (birthday_gift_chocolates : ℕ := 10)
  (saturdays_in_month : ℕ := 4)
  (total_chocolates : ℕ := 22) 
  (monthly_chocolates_herself := saturdays_in_month * herself_chocolates_per_saturday) 
  (equation : saturdays_in_month * S + monthly_chocolates_herself + birthday_gift_chocolates = total_chocolates) : 
  S = 1 :=
  sorry

end chocolates_for_sister_l28_28835


namespace geometric_sum_first_six_terms_l28_28181

variable (a_n : ℕ → ℝ)

axiom geometric_seq (r a1 : ℝ) : ∀ n, a_n n = a1 * r ^ (n - 1)
axiom a2_val : a_n 2 = 2
axiom a5_val : a_n 5 = 16

theorem geometric_sum_first_six_terms (S6 : ℝ) : S6 = 1 * (1 - 2^6) / (1 - 2) := by
  sorry

end geometric_sum_first_six_terms_l28_28181


namespace jon_total_cost_l28_28229
-- Import the complete Mathlib library

-- Define the conditions
def MSRP : ℝ := 30
def insurance_rate : ℝ := 0.20
def tax_rate : ℝ := 0.50

-- Calculate intermediate values based on conditions
noncomputable def insurance_cost : ℝ := insurance_rate * MSRP
noncomputable def subtotal_before_tax : ℝ := MSRP + insurance_cost
noncomputable def state_tax : ℝ := tax_rate * subtotal_before_tax
noncomputable def total_cost : ℝ := subtotal_before_tax + state_tax

-- The theorem we need to prove
theorem jon_total_cost : total_cost = 54 := by
  -- Proof is omitted
  sorry

end jon_total_cost_l28_28229


namespace eq_root_count_l28_28425

theorem eq_root_count (p : ℝ) : 
  (∀ x : ℝ, (2 * x^2 - 3 * p * x + 2 * p = 0 → (9 * p^2 - 16 * p = 0))) →
  (∃! p1 p2 : ℝ, (9 * p1^2 - 16 * p1 = 0) ∧ (9 * p2^2 - 16 * p2 = 0) ∧ p1 ≠ p2) :=
sorry

end eq_root_count_l28_28425


namespace vertex_of_parabola_l28_28242

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := (x + 2)^2 - 1

-- Define the vertex point
def vertex : ℝ × ℝ := (-2, -1)

-- The theorem we need to prove
theorem vertex_of_parabola : ∀ x : ℝ, parabola x = (x + 2)^2 - 1 → vertex = (-2, -1) := 
by
  sorry

end vertex_of_parabola_l28_28242


namespace tire_circumference_l28_28115

/-- If a tire rotates at 400 revolutions per minute and the car is traveling at 48 km/h, 
    prove that the circumference of the tire in meters is 2. -/
theorem tire_circumference (speed_kmh : ℕ) (revolutions_per_min : ℕ)
  (h1 : speed_kmh = 48) (h2 : revolutions_per_min = 400) : 
  (circumference : ℕ) = 2 := 
sorry

end tire_circumference_l28_28115


namespace sequence_geometric_l28_28509

theorem sequence_geometric (a : ℕ → ℝ) (r : ℝ) (h1 : ∀ n, a (n + 1) = r * a n) (h2 : a 4 = 2) : a 2 * a 6 = 4 :=
by
  sorry

end sequence_geometric_l28_28509


namespace annual_growth_rate_l28_28549

theorem annual_growth_rate (P₁ P₂ : ℝ) (y : ℕ) (r : ℝ)
  (h₁ : P₁ = 1) 
  (h₂ : P₂ = 1.21)
  (h₃ : y = 2)
  (h_growth : P₂ = P₁ * (1 + r) ^ y) :
  r = 0.1 :=
by {
  sorry
}

end annual_growth_rate_l28_28549


namespace multiple_properties_l28_28783

variables (a b : ℤ)

-- Definitions of the conditions
def is_multiple_of_4 (x : ℤ) : Prop := ∃ k : ℤ, x = 4 * k
def is_multiple_of_8 (x : ℤ) : Prop := ∃ k : ℤ, x = 8 * k

-- Problem statement
theorem multiple_properties (h1 : is_multiple_of_4 a) (h2 : is_multiple_of_8 b) :
  is_multiple_of_4 b ∧ is_multiple_of_4 (a + b) ∧ (∃ k : ℤ, a + b = 2 * k) :=
by
  sorry

end multiple_properties_l28_28783


namespace parabola_intersections_l28_28804

theorem parabola_intersections :
  (∀ x y, (y = 4 * x^2 + 4 * x - 7) ↔ (y = x^2 + 5)) →
  (∃ (points : List (ℝ × ℝ)),
    (points = [(-2, 9), (2, 9)]) ∧
    (∀ p ∈ points, ∃ x, p = (x, x^2 + 5) ∧ y = 4 * x^2 + 4 * x - 7)) :=
by sorry

end parabola_intersections_l28_28804


namespace equation1_solution_equation2_solutions_l28_28221

theorem equation1_solution (x : ℝ) : (x - 2) * (x - 3) = x - 2 → (x = 2 ∨ x = 4) :=
by
  intro h
  have h1 : (x - 2) * (x - 3) - (x - 2) = 0 := by sorry
  have h2 : (x - 2) * (x - 4) = 0 := by sorry
  have h3 : x - 2 = 0 ∨ x - 4 = 0 := by sorry
  cases h3 with
  | inl h4 => left; exact eq_of_sub_eq_zero h4
  | inr h5 => right; exact eq_of_sub_eq_zero h5

theorem equation2_solutions (x : ℝ) : 2 * x^2 - 5 * x + 1 = 0 → (x = (5 + Real.sqrt 17) / 4 ∨ x = (5 - Real.sqrt 17) / 4) :=
by
  intro h
  have h1 : (-5)^2 - 4 * 2 * 1 = 17 := by sorry
  have h2 : 2 * x^2 - 5 * x + 1 = 2 * ((x - (5 + Real.sqrt 17) / 4) * (x - (5 - Real.sqrt 17) / 4)) := by sorry
  have h3 : (x = (5 + Real.sqrt 17) / 4 ∨ x = (5 - Real.sqrt 17) / 4) := by sorry
  exact h3

end equation1_solution_equation2_solutions_l28_28221


namespace smallest_period_of_f_center_of_symmetry_of_f_range_of_f_on_interval_l28_28650

noncomputable def f (x : ℝ) : ℝ := 4 * (Real.sin x)^2 + 4 * (Real.sin x)^2 - (1 + 2)

theorem smallest_period_of_f : ∀ x : ℝ, f (x + π) = f x := 
by sorry

theorem center_of_symmetry_of_f : ∀ k : ℤ, ∃ c : ℝ, ∀ x : ℝ, f (c - x) = f (c + x) := 
by sorry

theorem range_of_f_on_interval : 
  ∃ a b, (∀ x ∈ Set.Icc (-π / 4) (π / 4), f x ∈ Set.Icc a b) ∧ 
          (∀ y, y ∈ Set.Icc 3 5 → ∃ x ∈ Set.Icc (-π / 4) (π / 4), y = f x) := 
by sorry

end smallest_period_of_f_center_of_symmetry_of_f_range_of_f_on_interval_l28_28650


namespace expression_value_zero_l28_28246

variable (x : ℝ)

theorem expression_value_zero (h : x^2 + 3 * x - 3 = 0) : x^3 + 2 * x^2 - 6 * x + 3 = 0 := 
by
  sorry

end expression_value_zero_l28_28246


namespace largest_minus_smallest_l28_28490

-- Define the given conditions
def A : ℕ := 10 * 2 + 9
def B : ℕ := A - 16
def C : ℕ := B * 3

-- Statement to prove
theorem largest_minus_smallest : C - B = 26 := by
  sorry

end largest_minus_smallest_l28_28490


namespace train_length_correct_l28_28078

noncomputable def train_speed_kmph : ℝ := 60
noncomputable def train_time_seconds : ℝ := 15

noncomputable def length_of_train : ℝ :=
  let speed_mps := train_speed_kmph * 1000 / 3600
  speed_mps * train_time_seconds

theorem train_length_correct :
  length_of_train = 250.05 :=
by
  -- Proof goes here
  sorry

end train_length_correct_l28_28078


namespace locus_of_M_is_ellipse_l28_28296

theorem locus_of_M_is_ellipse :
  ∀ (a b : ℝ) (M : ℝ × ℝ),
  a > b → b > 0 → (∃ x y : ℝ, 
  (M = (x, y)) ∧ 
  ∃ (P : ℝ × ℝ),
  (∃ x0 y0 : ℝ, P = (x0, y0) ∧ (x0^2 / a^2 + y0^2 / b^2 = 1)) ∧ 
  P ≠ (a, 0) ∧ P ≠ (-a, 0) ∧
  (∃ t : ℝ, t = (x^2 + y^2 - a^2) / (2 * y)) ∧ 
  (∃ x0 y0 : ℝ, 
    x0 = -x ∧ 
    y0 = 2 * t - y ∧
    x0^2 / a^2 + y0^2 / b^2 = 1)) →
  ∃ (x y : ℝ),
  M = (x, y) ∧ 
  (x^2 / a^2 + y^2 / (a^4 / b^2) = 1) := 
sorry

end locus_of_M_is_ellipse_l28_28296


namespace hens_egg_laying_l28_28548

theorem hens_egg_laying :
  ∀ (hens: ℕ) (price_per_dozen: ℝ) (total_revenue: ℝ) (weeks: ℕ) (total_hens: ℕ),
  hens = 10 →
  price_per_dozen = 3 →
  total_revenue = 120 →
  weeks = 4 →
  total_hens = hens →
  (total_revenue / price_per_dozen / 12) * 12 = 480 →
  (480 / weeks) = 120 →
  (120 / hens) = 12 :=
by sorry

end hens_egg_laying_l28_28548


namespace g_is_even_l28_28136

noncomputable def g (x : ℝ) := 2 ^ (x ^ 2 - 4) - |x|

theorem g_is_even : ∀ x : ℝ, g (-x) = g x :=
by
  sorry

end g_is_even_l28_28136


namespace digit_Q_is_0_l28_28375

theorem digit_Q_is_0 (M N P Q : ℕ) (hM : M < 10) (hN : N < 10) (hP : P < 10) (hQ : Q < 10) 
  (add_eq : 10 * M + N + 10 * P + M = 10 * Q + N) 
  (sub_eq : 10 * M + N - (10 * P + M) = N) : Q = 0 := 
by
  sorry

end digit_Q_is_0_l28_28375


namespace charlie_share_l28_28778

theorem charlie_share (A B C D E : ℝ) (h1 : A = (1/3) * B)
  (h2 : B = (1/2) * C) (h3 : C = 0.75 * D) (h4 : D = 2 * E) 
  (h5 : A + B + C + D + E = 15000) : C = 15000 * (3 / 11) :=
by
  sorry

end charlie_share_l28_28778


namespace find_ABC_l28_28570

-- Define the angles as real numbers in degrees
variables (ABC CBD DBC DBE ABE : ℝ)

-- Assert the given conditions
axiom horz_angle: CBD = 90
axiom DBC_ABC_relation : DBC = ABC + 30
axiom straight_angle: DBE = 180
axiom measure_abe: ABE = 145

-- State the proof problem
theorem find_ABC : ABC = 30 :=
by
  -- Include all steps required to derive the conclusion in the proof
  sorry

end find_ABC_l28_28570


namespace balance_balls_l28_28502

open Real

variables (G B Y W : ℝ)

-- Conditions
def condition1 := (4 * G = 8 * B)
def condition2 := (3 * Y = 6 * B)
def condition3 := (8 * B = 6 * W)

-- Theorem statement
theorem balance_balls 
  (h1 : condition1 G B) 
  (h2 : condition2 Y B) 
  (h3 : condition3 B W) :
  ∃ (B_needed : ℝ), B_needed = 5 * G + 3 * Y + 4 * W ∧ B_needed = 64 / 3 * B :=
sorry

end balance_balls_l28_28502


namespace sum_of_smallest_and_largest_prime_l28_28389

def primes_between (a b : ℕ) : List ℕ := List.filter Nat.Prime (List.range' a (b - a + 1))

def smallest_prime_in_range (a b : ℕ) : ℕ :=
  match primes_between a b with
  | [] => 0
  | h::t => h

def largest_prime_in_range (a b : ℕ) : ℕ :=
  match List.reverse (primes_between a b) with
  | [] => 0
  | h::t => h

theorem sum_of_smallest_and_largest_prime : smallest_prime_in_range 1 50 + largest_prime_in_range 1 50 = 49 := 
by
  -- Let the Lean prover take over from here
  sorry

end sum_of_smallest_and_largest_prime_l28_28389


namespace find_lengths_of_DE_and_HJ_l28_28316

noncomputable def lengths_consecutive_segments (BD DE EF FG GH HJ : ℝ) (BC : ℝ) : Prop :=
  BD = 5 ∧ EF = 11 ∧ FG = 7 ∧ GH = 3 ∧ BC = 29 ∧ BD + DE + EF + FG + GH + HJ = BC ∧ DE = HJ

theorem find_lengths_of_DE_and_HJ (x : ℝ) : lengths_consecutive_segments 5 x 11 7 3 x 29 → x = 1.5 :=
by
  intros h
  sorry

end find_lengths_of_DE_and_HJ_l28_28316


namespace weeks_jake_buys_papayas_l28_28743

theorem weeks_jake_buys_papayas
  (jake_papayas : ℕ)
  (brother_papayas : ℕ)
  (father_papayas : ℕ)
  (total_papayas : ℕ)
  (h1 : jake_papayas = 3)
  (h2 : brother_papayas = 5)
  (h3 : father_papayas = 4)
  (h4 : total_papayas = 48) :
  (total_papayas / (jake_papayas + brother_papayas + father_papayas) = 4) :=
by
  sorry

end weeks_jake_buys_papayas_l28_28743


namespace fraction_diff_l28_28361

open Real

theorem fraction_diff (x y : ℝ) (hx : x = sqrt 5 - 1) (hy : y = sqrt 5 + 1) :
  (1 / x - 1 / y) = 1 / 2 := sorry

end fraction_diff_l28_28361


namespace find_f_2012_l28_28622

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  a * Real.log x / Real.log 2 + b * Real.log x / Real.log 3 + 2

theorem find_f_2012 (a b : ℝ) (h : f (1 / 2012) a b = 5) : f 2012 a b = -1 :=
by
  sorry

end find_f_2012_l28_28622


namespace probability_of_lamps_arrangement_l28_28395

noncomputable def probability_lava_lamps : ℚ :=
  let total_lamps := 8
  let red_lamps := 4
  let blue_lamps := 4
  let total_turn_on := 4
  let left_red_on := 1
  let right_blue_off := 1
  let ways_to_choose_positions := Nat.choose total_lamps red_lamps
  let ways_to_choose_turn_on := Nat.choose total_lamps total_turn_on
  let remaining_positions := total_lamps - left_red_on - right_blue_off
  let remaining_red_lamps := red_lamps - left_red_on
  let remaining_turn_on := total_turn_on - left_red_on
  let arrangements_of_remaining_red := Nat.choose remaining_positions remaining_red_lamps
  let arrangements_of_turn_on :=
    Nat.choose (remaining_positions - right_blue_off) remaining_turn_on
  -- The probability calculation
  (arrangements_of_remaining_red * arrangements_of_turn_on : ℚ) / 
    (ways_to_choose_positions * ways_to_choose_turn_on)

theorem probability_of_lamps_arrangement :
    probability_lava_lamps = 4 / 49 :=
by
  sorry

end probability_of_lamps_arrangement_l28_28395


namespace div_by_72_l28_28216

theorem div_by_72 (x : ℕ) (y : ℕ) (h1 : 0 ≤ x ∧ x ≤ 9) (h2 : x = 4)
    (h3 : 0 ≤ y ∧ y ≤ 9) (h4 : y = 6) : 
    72 ∣ (9834800 + 1000 * x + 10 * y) :=
by 
  sorry

end div_by_72_l28_28216


namespace derivative_of_cos_over_x_l28_28265

open Real

noncomputable def f (x : ℝ) : ℝ := (cos x) / x

theorem derivative_of_cos_over_x (x : ℝ) (h : x ≠ 0) : 
  deriv f x = - (x * sin x + cos x) / (x^2) :=
sorry

end derivative_of_cos_over_x_l28_28265


namespace abby_correct_percentage_l28_28621

-- Defining the scores and number of problems for each test
def score_test1 := 85 / 100
def score_test2 := 75 / 100
def score_test3 := 60 / 100
def score_test4 := 90 / 100

def problems_test1 := 30
def problems_test2 := 50
def problems_test3 := 20
def problems_test4 := 40

-- Define the total number of problems
def total_problems := problems_test1 + problems_test2 + problems_test3 + problems_test4

-- Calculate the number of problems Abby answered correctly on each test
def correct_problems_test1 := score_test1 * problems_test1
def correct_problems_test2 := score_test2 * problems_test2
def correct_problems_test3 := score_test3 * problems_test3
def correct_problems_test4 := score_test4 * problems_test4

-- Calculate the total number of correctly answered problems
def total_correct_problems := correct_problems_test1 + correct_problems_test2 + correct_problems_test3 + correct_problems_test4

-- Calculate the overall percentage of problems answered correctly
def overall_percentage_correct := (total_correct_problems / total_problems) * 100

-- The theorem to be proved
theorem abby_correct_percentage : overall_percentage_correct = 80 := by
  -- Skipping the actual proof
  sorry

end abby_correct_percentage_l28_28621


namespace cone_base_radius_half_l28_28156

theorem cone_base_radius_half :
  let R : ℝ := sorry
  let semicircle_radius : ℝ := 1
  let unfolded_circumference : ℝ := π
  let base_circumference : ℝ := 2 * π * R
  base_circumference = unfolded_circumference -> R = 1 / 2 :=
by
  sorry

end cone_base_radius_half_l28_28156


namespace find_n_l28_28006

-- Definitions for conditions given in the problem
def a₂ (a : ℕ → ℕ) : Prop := a 2 = 3
def consecutive_sum (S : ℕ → ℕ) (n : ℕ) : Prop := ∀ n > 3, S n - S (n - 3) = 51
def total_sum (S : ℕ → ℕ) (n : ℕ) : Prop := S n = 100

-- The main proof problem
theorem find_n (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) 
  (h₁ : a₂ a) (h₂ : consecutive_sum S n) (h₃ : total_sum S n) : n = 10 :=
sorry

end find_n_l28_28006


namespace solve_equation_l28_28927

theorem solve_equation (a b : ℕ) : 
  (a^2 = b * (b + 7) ∧ a ≥ 0 ∧ b ≥ 0) ↔ (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) := by
  sorry

end solve_equation_l28_28927


namespace dictionary_cost_l28_28584

def dinosaur_book_cost : ℕ := 19
def children_cookbook_cost : ℕ := 7
def saved_amount : ℕ := 8
def needed_amount : ℕ := 29

def total_amount_needed := saved_amount + needed_amount
def combined_books_cost := dinosaur_book_cost + children_cookbook_cost

theorem dictionary_cost : total_amount_needed - combined_books_cost = 11 :=
by
  -- proof omitted
  sorry

end dictionary_cost_l28_28584


namespace real_number_solution_l28_28177

theorem real_number_solution : ∃ x : ℝ, x = 3 + 6 / (1 + 6 / x) ∧ x = 3 * Real.sqrt 2 :=
by
  sorry

end real_number_solution_l28_28177


namespace problem_solution_l28_28681

theorem problem_solution (x : ℝ) (h : ∃ (A B : Set ℝ), A = {0, 1, 2, 4, 5} ∧ B = {x-2, x, x+2} ∧ A ∩ B = {0, 2}) : x = 0 :=
sorry

end problem_solution_l28_28681


namespace sequence_count_l28_28317

theorem sequence_count :
  ∃ f : ℕ → ℕ,
    (f 3 = 1) ∧ (f 4 = 1) ∧ (f 5 = 1) ∧ (f 6 = 2) ∧ (f 7 = 2) ∧
    (∀ n, n ≥ 8 → f n = f (n-4) + 2 * f (n-5) + f (n-6)) ∧
    f 15 = 21 :=
by {
  sorry
}

end sequence_count_l28_28317


namespace number_of_episodes_last_season_more_than_others_l28_28730

-- Definitions based on conditions
def episodes_per_other_season : ℕ := 22
def initial_seasons : ℕ := 9
def duration_per_episode : ℚ := 0.5
def total_hours_after_last_season : ℚ := 112

-- Derived definitions based on conditions (not solution steps)
def total_hours_first_9_seasons := initial_seasons * episodes_per_other_season * duration_per_episode
def additional_hours_last_season := total_hours_after_last_season - total_hours_first_9_seasons
def episodes_last_season := additional_hours_last_season / duration_per_episode

-- Proof problem statement
theorem number_of_episodes_last_season_more_than_others : 
  episodes_last_season = episodes_per_other_season + 4 :=
by
  -- Placeholder for the proof
  sorry

end number_of_episodes_last_season_more_than_others_l28_28730


namespace part1_part2_part3_part3_expectation_l28_28854

/-- Conditions setup -/
noncomputable def gameCondition (Aacc Bacc : ℝ) :=
  (Aacc = 0.5) ∧ (Bacc = 0.6)

def scoreDist (X:ℤ) : ℝ :=
  if X = -1 then 0.3
  else if X = 0 then 0.5
  else if X = 1 then 0.2
  else 0

def tieProbability : ℝ := 0.2569

def roundDist (Y:ℤ) : ℝ :=
  if Y = 2 then 0.13
  else if Y = 3 then 0.13
  else if Y = 4 then 0.74
  else 0

def roundExpectation : ℝ := 3.61

/-- Proof Statements -/
theorem part1 (Aacc Bacc : ℝ) (h : gameCondition Aacc Bacc) : 
  ∀ (X : ℤ), scoreDist X = if X = -1 then 0.3 else if X = 0 then 0.5 else if X = 1 then 0.2 else 0 :=
by sorry

theorem part2 (Aacc Bacc : ℝ) (h : gameCondition Aacc Bacc) : 
  tieProbability = 0.2569 :=
by sorry

theorem part3 (Aacc Bacc : ℝ) (h : gameCondition Aacc Bacc) : 
  ∀ (Y : ℤ), roundDist Y = if Y = 2 then 0.13 else if Y = 3 then 0.13 else if Y = 4 then 0.74 else 0 :=
by sorry

theorem part3_expectation (Aacc Bacc : ℝ) (h : gameCondition Aacc Bacc) :
  roundExpectation = 3.61 :=
by sorry

end part1_part2_part3_part3_expectation_l28_28854


namespace count_arithmetic_sequence_terms_l28_28595

theorem count_arithmetic_sequence_terms : 
  ∃ n : ℕ, 
  (∀ k : ℕ, k ≥ 1 → 6 + (k - 1) * 4 = 202 → n = k) ∧ n = 50 :=
by
  sorry

end count_arithmetic_sequence_terms_l28_28595


namespace gcd_1729_1768_l28_28004

theorem gcd_1729_1768 : Int.gcd 1729 1768 = 13 := by
  sorry

end gcd_1729_1768_l28_28004


namespace talent_show_l28_28047

theorem talent_show (B G : ℕ) (h1 : G = B + 22) (h2 : G + B = 34) : G = 28 :=
by
  sorry

end talent_show_l28_28047


namespace balloon_count_l28_28530

-- Conditions
def Fred_balloons : ℕ := 5
def Sam_balloons : ℕ := 6
def Mary_balloons : ℕ := 7
def total_balloons : ℕ := 18

-- Proof statement
theorem balloon_count :
  Fred_balloons + Sam_balloons + Mary_balloons = total_balloons :=
by
  exact Nat.add_assoc 5 6 7 ▸ rfl

end balloon_count_l28_28530


namespace integer_solution_count_l28_28635

theorem integer_solution_count : 
  let condition := ∀ x : ℤ, (x - 2) ^ 2 ≤ 4
  ∃ count : ℕ, count = 5 ∧ (∀ x : ℤ, condition → (0 ≤ x ∧ x ≤ 4)) := 
sorry

end integer_solution_count_l28_28635


namespace wrapping_paper_area_correct_l28_28791

-- Define the length, width, and height of the box
variables (l w h : ℝ)

-- Define the function to calculate the area of the wrapping paper
def wrapping_paper_area (l w h : ℝ) : ℝ := 2 * (l + w + h) ^ 2

-- Statement problem that we need to prove
theorem wrapping_paper_area_correct :
  wrapping_paper_area l w h = 2 * (l + w + h) ^ 2 := 
sorry

end wrapping_paper_area_correct_l28_28791


namespace sum_squares_l28_28950

theorem sum_squares {a b c : ℝ} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) 
  (h5 : a^4 + b^4 + c^4 = a^6 + b^6 + c^6) : 
  a^2 + b^2 + c^2 = 6 / 5 := 
by sorry

end sum_squares_l28_28950


namespace negation_of_proposition_l28_28925

-- Conditions
variable {x : ℝ}

-- The proposition
def proposition : Prop := ∃ x : ℝ, Real.exp x > x

-- The proof problem: proving the negation of the proposition
theorem negation_of_proposition : (¬ proposition) ↔ ∀ x : ℝ, Real.exp x ≤ x := by
  sorry

end negation_of_proposition_l28_28925


namespace range_m_plus_2n_l28_28307

noncomputable def f (x : ℝ) : ℝ := Real.log x - 1 / x
noncomputable def m_value (t : ℝ) : ℝ := 1 / t + 1 / (t ^ 2)

noncomputable def n_value (t : ℝ) : ℝ := Real.log t - 2 / t - 1

noncomputable def g (x : ℝ) : ℝ := (1 / (x ^ 2)) + 2 * Real.log x - (3 / x) - 2

theorem range_m_plus_2n :
  ∀ m n : ℝ, (∃ t > 0, m = m_value t ∧ n = n_value t) →
  (m + 2 * n) ∈ Set.Ici (-2 * Real.log 2 - 4) := by
  sorry

end range_m_plus_2n_l28_28307


namespace find_q_l28_28145

theorem find_q (q : ℤ) (x : ℤ) (y : ℤ) (h1 : x = 55 + 2 * q) (h2 : y = 4 * q + 41) (h3 : x = y) : q = 7 :=
by
  sorry

end find_q_l28_28145


namespace square_root_unique_l28_28356

theorem square_root_unique (x : ℝ) (h1 : x + 3 ≥ 0) (h2 : 2 * x - 6 ≥ 0)
  (h : (x + 3)^2 = (2 * x - 6)^2) :
  x = 1 ∧ (x + 3)^2 = 16 := 
by
  sorry

end square_root_unique_l28_28356


namespace non_neg_int_solutions_l28_28091

theorem non_neg_int_solutions : 
  ∀ (x y : ℕ), 2 * x ^ 2 + 2 * x * y - x + y = 2020 → 
               (x = 0 ∧ y = 2020) ∨ (x = 1 ∧ y = 673) :=
by
  sorry

end non_neg_int_solutions_l28_28091


namespace ones_digit_of_prime_in_arithmetic_sequence_is_one_l28_28728

theorem ones_digit_of_prime_in_arithmetic_sequence_is_one 
  (p q r s : ℕ) 
  (hp : Prime p) 
  (hq : Prime q) 
  (hr : Prime r) 
  (hs : Prime s) 
  (h₁ : p > 10) 
  (h₂ : q = p + 10) 
  (h₃ : r = q + 10) 
  (h₄ : s = r + 10) 
  (h₅ : s > r) 
  (h₆ : r > q) 
  (h₇ : q > p) : 
  p % 10 = 1 :=
sorry

end ones_digit_of_prime_in_arithmetic_sequence_is_one_l28_28728


namespace triangle_probability_is_correct_l28_28986

-- Define the total number of figures
def total_figures : ℕ := 8

-- Define the number of triangles among the figures
def number_of_triangles : ℕ := 3

-- Define the probability function for choosing a triangle
def probability_of_triangle : ℚ := number_of_triangles / total_figures

-- The theorem to be proved
theorem triangle_probability_is_correct :
  probability_of_triangle = 3 / 8 := by
  sorry

end triangle_probability_is_correct_l28_28986


namespace find_fraction_of_ab_l28_28632

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def x := a / b

theorem find_fraction_of_ab (h1 : a ≠ b) (h2 : a / b + (3 * a + 4 * b) / (b + 12 * a) = 2) :
  a / b = (5 - Real.sqrt 19) / 6 :=
sorry

end find_fraction_of_ab_l28_28632


namespace residents_rent_contribution_l28_28620

theorem residents_rent_contribution (x R : ℝ) (hx1 : 10 * x + 88 = R) (hx2 : 10.80 * x = 1.025 * R) :
  R / x = 10.54 :=
by sorry

end residents_rent_contribution_l28_28620


namespace jelly_bean_ratio_l28_28517

theorem jelly_bean_ratio
  (initial_jelly_beans : ℕ)
  (num_people : ℕ)
  (remaining_jelly_beans : ℕ)
  (amount_taken_by_each_of_last_four : ℕ)
  (total_taken_by_last_four : ℕ)
  (total_jelly_beans_taken : ℕ)
  (X : ℕ)
  (ratio : ℕ)
  (h0 : initial_jelly_beans = 8000)
  (h1 : num_people = 10)
  (h2 : remaining_jelly_beans = 1600)
  (h3 : amount_taken_by_each_of_last_four = 400)
  (h4 : total_taken_by_last_four = 4 * amount_taken_by_each_of_last_four)
  (h5 : total_jelly_beans_taken = initial_jelly_beans - remaining_jelly_beans)
  (h6 : X = total_jelly_beans_taken - total_taken_by_last_four)
  (h7 : ratio = X / total_taken_by_last_four)
  : ratio = 3 :=
by sorry

end jelly_bean_ratio_l28_28517


namespace x_y_differ_by_one_l28_28914

theorem x_y_differ_by_one (x y : ℚ) (h : (1 + y) / (x - y) = x) : y = x - 1 :=
by
sorry

end x_y_differ_by_one_l28_28914


namespace train_length_l28_28065

-- Define the given conditions
def train_cross_time : ℕ := 40 -- time in seconds
def train_speed_kmh : ℕ := 144 -- speed in km/h

-- Convert the speed from km/h to m/s
def kmh_to_ms (speed_kmh : ℕ) : ℕ :=
  (speed_kmh * 5) / 18 

def train_speed_ms : ℕ := kmh_to_ms train_speed_kmh

-- Theorem statement
theorem train_length :
  train_speed_ms * train_cross_time = 1600 :=
by
  sorry

end train_length_l28_28065


namespace circle_radius_formula_correct_l28_28687

noncomputable def touch_circles_radius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let Δ := (s - a) * (s - b) * (s - c)
  let numerator := c * Real.sqrt ((s - a) * (s - b) * (s - c))
  let denominator := c * Real.sqrt s + 2 * Real.sqrt ((s - a) * (s - b) * (s - c))
  numerator / denominator

theorem circle_radius_formula_correct (a b c : ℝ) : 
  let s := (a + b + c) / 2
  let Δ := (s - a) * (s - b) * (s - c)
  ∀ (r : ℝ), (r = touch_circles_radius a b c) :=
sorry

end circle_radius_formula_correct_l28_28687


namespace median_of_consecutive_integers_l28_28349

theorem median_of_consecutive_integers (a b : ℤ) (h : a + b = 50) : 
  (a + b) / 2 = 25 := 
by 
  sorry

end median_of_consecutive_integers_l28_28349


namespace triangle_inequality_condition_l28_28359

variable (a b c : ℝ)
variable (α : ℝ) -- angle in radians

-- Define the condition where c must be less than a + b
theorem triangle_inequality_condition : c < a + b := by
  sorry

end triangle_inequality_condition_l28_28359


namespace total_votes_l28_28729

theorem total_votes (V : ℝ) (h1 : 0.70 * V = V - 240) (h2 : 0.30 * V = 240) : V = 800 :=
by
  sorry

end total_votes_l28_28729


namespace pow_ge_double_l28_28180

theorem pow_ge_double (n : ℕ) : 2^n ≥ 2 * n := sorry

end pow_ge_double_l28_28180


namespace smallest_possible_value_l28_28017

theorem smallest_possible_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 / (x : ℝ)) + (1 / (y : ℝ)) = 1 / 15) : x + y = 64 :=
sorry

end smallest_possible_value_l28_28017


namespace problem_statement_l28_28093

theorem problem_statement : ((26.3 * 12 * 20) / 3 + 125 - Real.sqrt 576 = 21141) :=
by
  sorry

end problem_statement_l28_28093


namespace total_boxes_correct_l28_28550

noncomputable def initial_boxes : ℕ := 400
noncomputable def cost_per_box : ℕ := 80 + 165
noncomputable def initial_spent : ℕ := initial_boxes * cost_per_box
noncomputable def donor_amount : ℕ := 4 * initial_spent
noncomputable def additional_boxes : ℕ := donor_amount / cost_per_box
noncomputable def total_boxes : ℕ := initial_boxes + additional_boxes

theorem total_boxes_correct : total_boxes = 2000 := by
  sorry

end total_boxes_correct_l28_28550


namespace solve_fractional_eq_l28_28151

theorem solve_fractional_eq (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ -3) : (1 / x = 6 / (x + 3)) → (x = 0.6) :=
by
  sorry

end solve_fractional_eq_l28_28151


namespace nancy_rose_bracelets_l28_28113

-- Definitions based on conditions
def metal_beads_nancy : ℕ := 40
def pearl_beads_nancy : ℕ := metal_beads_nancy + 20
def total_beads_nancy : ℕ := metal_beads_nancy + pearl_beads_nancy

def crystal_beads_rose : ℕ := 20
def stone_beads_rose : ℕ := 2 * crystal_beads_rose
def total_beads_rose : ℕ := crystal_beads_rose + stone_beads_rose

def number_of_bracelets (total_beads : ℕ) (beads_per_bracelet : ℕ) : ℕ :=
  total_beads / beads_per_bracelet

-- Theorem to be proved
theorem nancy_rose_bracelets : number_of_bracelets (total_beads_nancy + total_beads_rose) 8 = 20 := 
by
  -- Definitions will be expanded here
  sorry

end nancy_rose_bracelets_l28_28113


namespace sum_of_coefficients_l28_28717

-- Defining the given conditions
def vertex : ℝ × ℝ := (5, -4)
def point : ℝ × ℝ := (3, -2)

-- Defining the problem to prove the sum of the coefficients
theorem sum_of_coefficients (a b c : ℝ)
  (h_eq : ∀ y, 5 = a * ((-4) + y)^2 + c)
  (h_pt : 3 = a * ((-4) + (-2))^2 + b * (-2) + c) :
  a + b + c = -15 / 2 :=
sorry

end sum_of_coefficients_l28_28717


namespace problem_ineq_l28_28577

variable {a b c : ℝ}

theorem problem_ineq 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * a^2 * b^2 * c^2 := 
sorry

end problem_ineq_l28_28577


namespace field_area_is_243_l28_28460

noncomputable def field_area (w l : ℝ) (h1 : w = l / 3) (h2 : 2 * (w + l) = 72) : ℝ :=
  w * l

theorem field_area_is_243 (w l : ℝ) (h1 : w = l / 3) (h2 : 2 * (w + l) = 72) : field_area w l h1 h2 = 243 :=
  sorry

end field_area_is_243_l28_28460


namespace solve_for_x_l28_28706

theorem solve_for_x (x y z : ℝ) (h1 : x * y = 8 - 3 * x - 2 * y) 
                                  (h2 : y * z = 8 - 2 * y - 3 * z) 
                                  (h3 : x * z = 35 - 5 * x - 3 * z) : 
  x = 8 :=
sorry

end solve_for_x_l28_28706


namespace probability_of_both_types_probability_distribution_and_expectation_of_X_l28_28411

-- Definitions
def total_zongzi : ℕ := 8
def red_bean_paste_zongzi : ℕ := 2
def date_zongzi : ℕ := 6
def selected_zongzi : ℕ := 3

-- Part 1: The probability of selecting both red bean paste and date zongzi
theorem probability_of_both_types :
  let total_combinations := Nat.choose total_zongzi selected_zongzi
  let one_red_two_date := Nat.choose red_bean_paste_zongzi 1 * Nat.choose date_zongzi 2
  let two_red_one_date := Nat.choose red_bean_paste_zongzi 2 * Nat.choose date_zongzi 1
  (one_red_two_date + two_red_one_date) / total_combinations = 9 / 14 :=
by sorry

-- Part 2: The probability distribution and expectation of X
theorem probability_distribution_and_expectation_of_X :
  let P_X_0 := (Nat.choose red_bean_paste_zongzi 0 * Nat.choose date_zongzi 3) / Nat.choose total_zongzi selected_zongzi
  let P_X_1 := (Nat.choose red_bean_paste_zongzi 1 * Nat.choose date_zongzi 2) / Nat.choose total_zongzi selected_zongzi
  let P_X_2 := (Nat.choose red_bean_paste_zongzi 2 * Nat.choose date_zongzi 1) / Nat.choose total_zongzi selected_zongzi
  P_X_0 = 5 / 14 ∧ P_X_1 = 15 / 28 ∧ P_X_2 = 3 / 28 ∧
  (0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2 = 3 / 4) :=
by sorry

end probability_of_both_types_probability_distribution_and_expectation_of_X_l28_28411


namespace max_min_sum_difference_l28_28409

-- The statement that we need to prove
theorem max_min_sum_difference : 
  ∃ (max_sum min_sum: ℕ), (∀ (RST UVW XYZ : ℕ),
   -- Constraints for Max's and Minnie's sums respectively
   (RST = 100 * 9 + 10 * 6 + 3 ∧ UVW = 100 * 8 + 10 * 5 + 2 ∧ XYZ = 100 * 7 + 10 * 4 + 1 → max_sum = 2556) ∧ 
   (RST = 100 * 1 + 10 * 0 + 6 ∧ UVW = 100 * 2 + 10 * 4 + 7 ∧ XYZ = 100 * 3 + 10 * 5 + 8 → min_sum = 711)) → 
    max_sum - min_sum = 1845 :=
by
  sorry

end max_min_sum_difference_l28_28409


namespace number_of_sides_of_regular_polygon_l28_28253

theorem number_of_sides_of_regular_polygon (n : ℕ) (h : 0 < n) (h_angle : ∀ i, i < n → (2 * n - 4) * 90 / n = 150) : n = 12 :=
sorry

end number_of_sides_of_regular_polygon_l28_28253


namespace taxi_ride_cost_l28_28474

theorem taxi_ride_cost (initial_cost : ℝ) (cost_first_3_miles : ℝ) (rate_first_3_miles : ℝ) (rate_after_3_miles : ℝ) (total_miles : ℝ) (remaining_miles : ℝ) :
  initial_cost = 2.00 ∧ rate_first_3_miles = 0.30 ∧ rate_after_3_miles = 0.40 ∧ total_miles = 8 ∧ total_miles - 3 = remaining_miles →
  initial_cost + 3 * rate_first_3_miles + remaining_miles * rate_after_3_miles = 4.90 :=
sorry

end taxi_ride_cost_l28_28474


namespace friends_count_is_four_l28_28036

def number_of_friends (Melanie Benny Sally Jessica : ℕ) (total_cards : ℕ) : ℕ :=
  4

theorem friends_count_is_four (Melanie Benny Sally Jessica : ℕ) (total_cards : ℕ) (h1 : total_cards = 12) :
  number_of_friends Melanie Benny Sally Jessica total_cards = 4 :=
by
  sorry

end friends_count_is_four_l28_28036


namespace find_value_of_a_l28_28039

-- Define the setting for triangle ABC with sides a, b, c opposite to angles A, B, C respectively
variables {a b c : ℝ}
variables {A B C : ℝ}
variables (h1 : b^2 - c^2 + 2 * a = 0)
variables (h2 : Real.tan C / Real.tan B = 3)

-- Given conditions and conclusion for the proof problem
theorem find_value_of_a 
  (h1 : b^2 - c^2 + 2 * a = 0) 
  (h2 : Real.tan C / Real.tan B = 3) : 
  a = 4 := 
sorry

end find_value_of_a_l28_28039


namespace clock_chime_time_l28_28102

/-- The proven time it takes for a wall clock to strike 12 times at 12 o'clock -/
theorem clock_chime_time :
  (∃ (interval_time : ℝ), (interval_time = 3) ∧ (∃ (time_12_times : ℝ), (time_12_times = interval_time * (12 - 1)) ∧ (time_12_times = 33))) :=
by
  sorry

end clock_chime_time_l28_28102


namespace proof_problem_l28_28298

variable (a b c d x : ℤ)

-- Conditions
def are_opposite (a b : ℤ) : Prop := a + b = 0
def are_reciprocals (c d : ℤ) : Prop := c * d = 1
def largest_negative_integer (x : ℤ) : Prop := x = -1

theorem proof_problem 
  (h1 : are_opposite a b) 
  (h2 : are_reciprocals c d) 
  (h3 : largest_negative_integer x) :
  x^2 - (a + b - c * d)^(2012 : ℕ) + (-c * d)^(2011 : ℕ) = -1 :=
by
  sorry

end proof_problem_l28_28298


namespace find_width_l28_28770

-- Definition of the perimeter of a rectangle
def perimeter (L W : ℝ) : ℝ := 2 * (L + W)

-- The given conditions
def length := 13
def perimeter_value := 50

-- The goal to prove: if the perimeter is 50 and the length is 13, then the width must be 12
theorem find_width :
  ∃ (W : ℝ), perimeter length W = perimeter_value ∧ W = 12 :=
by
  sorry

end find_width_l28_28770


namespace measure_four_liters_impossible_l28_28628

theorem measure_four_liters_impossible (a b c : ℕ) (h1 : a = 12) (h2 : b = 9) (h3 : c = 4) :
  ¬ ∃ x y : ℕ, x * a + y * b = c := 
by
  sorry

end measure_four_liters_impossible_l28_28628


namespace remaining_players_average_points_l28_28698

-- Define the conditions
def total_points : ℕ := 270
def total_players : ℕ := 9
def players_averaged_50 : ℕ := 5
def average_points_50 : ℕ := 50

-- Define the query
theorem remaining_players_average_points :
  (total_points - players_averaged_50 * average_points_50) / (total_players - players_averaged_50) = 5 :=
by
  sorry

end remaining_players_average_points_l28_28698


namespace average_bc_l28_28493

variables (A B C : ℝ)

-- Conditions
def average_abc := (A + B + C) / 3 = 45
def average_ab := (A + B) / 2 = 40
def weight_b := B = 31

-- Proof statement
theorem average_bc (A B C : ℝ) (h_avg_abc : average_abc A B C) (h_avg_ab : average_ab A B) (h_b : weight_b B) :
  (B + C) / 2 = 43 :=
sorry

end average_bc_l28_28493


namespace circle_radius_center_l28_28139

theorem circle_radius_center (x y : ℝ) (h : x^2 + y^2 - 2*x - 2*y - 2 = 0) :
  (∃ a b r, (x - a)^2 + (y - b)^2 = r^2 ∧ a = 1 ∧ b = 1 ∧ r = 2) := 
sorry

end circle_radius_center_l28_28139


namespace positive_perfect_squares_multiples_of_36_lt_10_pow_8_l28_28403

theorem positive_perfect_squares_multiples_of_36_lt_10_pow_8 :
  ∃ (count : ℕ), count = 1666 ∧ 
    (∀ n : ℕ, (n * n < 10^8 → n % 6 = 0) ↔ (n < 10^4 ∧ n % 6 = 0)) :=
sorry

end positive_perfect_squares_multiples_of_36_lt_10_pow_8_l28_28403


namespace find_N_l28_28373

noncomputable def N : ℕ := 1156

-- Condition 1: N is a perfect square
axiom N_perfect_square : ∃ n : ℕ, N = n^2

-- Condition 2: All digits of N are less than 7
axiom N_digits_less_than_7 : ∀ d, d ∈ [1, 1, 5, 6] → d < 7

-- Condition 3: Adding 3 to each digit yields another perfect square
axiom N_plus_3_perfect_square : ∃ m : ℕ, (m^2 = 1156 + 3333)

theorem find_N : N = 1156 :=
by
  -- Proof goes here
  sorry

end find_N_l28_28373


namespace complex_div_eq_i_l28_28111

open Complex

theorem complex_div_eq_i : (1 + I) / (1 - I) = I := by
  sorry

end complex_div_eq_i_l28_28111


namespace sum_log_base_5_divisors_l28_28105

theorem sum_log_base_5_divisors (n : ℕ) (h : n * (n + 1) / 2 = 264) : n = 23 :=
by
  sorry

end sum_log_base_5_divisors_l28_28105


namespace kat_boxing_trainings_per_week_l28_28362

noncomputable def strength_training_hours_per_week : ℕ := 3
noncomputable def boxing_training_hours (x : ℕ) : ℚ := 1.5 * x
noncomputable def total_training_hours : ℕ := 9

theorem kat_boxing_trainings_per_week (x : ℕ) (h : total_training_hours = strength_training_hours_per_week + boxing_training_hours x) : x = 4 :=
by
  sorry

end kat_boxing_trainings_per_week_l28_28362


namespace factor_expression_l28_28451

theorem factor_expression (x y : ℝ) :
  66 * x^5 - 165 * x^9 + 99 * x^5 * y = 33 * x^5 * (2 - 5 * x^4 + 3 * y) :=
by
  sorry

end factor_expression_l28_28451


namespace exponentiation_rule_l28_28159

theorem exponentiation_rule (x : ℝ) : (x^5)^2 = x^10 :=
by {
  sorry
}

end exponentiation_rule_l28_28159


namespace minimum_value_inequality_l28_28140

theorem minimum_value_inequality {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (hxyz : x + y + z = 9) :
  (x^3 + y^3) / (x + y) + (x^3 + z^3) / (x + z) + (y^3 + z^3) / (y + z) ≥ 27 :=
sorry

end minimum_value_inequality_l28_28140


namespace real_part_implies_value_of_a_l28_28455

theorem real_part_implies_value_of_a (a b : ℝ) (h : a = 2 * b) (hb : b = 1) : a = 2 := by
  sorry

end real_part_implies_value_of_a_l28_28455


namespace triangle_obtuse_l28_28054

theorem triangle_obtuse (α β γ : ℝ) 
  (h1 : α ≤ β) (h2 : β < γ) 
  (h3 : α + β + γ = 180) 
  (h4 : α + β < γ) : 
  γ > 90 :=
  sorry

end triangle_obtuse_l28_28054


namespace f_correct_l28_28924

noncomputable def f : ℕ → ℝ
| 0       => 0 -- undefined for 0, start from 1
| (n + 1) => if n = 0 then 1/2 else sorry -- recursion undefined for now

theorem f_correct : ∀ n ≥ 1, f n = (3^(n-1) / (3^(n-1) + 1)) :=
by
  -- Initial conditions
  have h0 : f 1 = 1/2 := sorry
  -- Recurrence relations
  have h1 : ∀ n, n ≥ 1 → f (n + 1) ≥ (3 * f n) / (2 * f n + 1) := sorry
  -- Prove the function form
  sorry

end f_correct_l28_28924


namespace toys_gained_l28_28865

theorem toys_gained
  (sp : ℕ) -- selling price of 18 toys
  (cp_per_toy : ℕ) -- cost price per toy
  (sp_val : sp = 27300) -- given selling price value
  (cp_per_val : cp_per_toy = 1300) -- given cost price per toy value
  : (sp - 18 * cp_per_toy) / cp_per_toy = 3 := by
  -- Conditions of the problem are stated
  -- Proof is omitted with 'sorry'
  sorry

end toys_gained_l28_28865


namespace trigonometric_inequality_1_l28_28501

theorem trigonometric_inequality_1 {n : ℕ} 
  (h1 : 0 < n) (x : ℝ) (h2 : 0 < x) (h3 : x < (Real.pi / (2 * n))) :
  (1 / 2) * (Real.tan x + Real.tan (n * x) - Real.tan ((n - 1) * x)) > (1 / n) * Real.tan (n * x) := 
sorry

end trigonometric_inequality_1_l28_28501


namespace savings_percentage_correct_l28_28653

-- Definitions based on conditions
def food_per_week : ℕ := 100
def num_weeks : ℕ := 4
def rent : ℕ := 1500
def video_streaming : ℕ := 30
def cell_phone : ℕ := 50
def savings : ℕ := 198

-- Total spending calculations based on the conditions
def food_total : ℕ := food_per_week * num_weeks
def total_spending : ℕ := food_total + rent + video_streaming + cell_phone

-- Calculation of the percentage
def savings_percentage (savings total_spending : ℕ) : ℕ :=
  (savings * 100) / total_spending

-- The statement to prove
theorem savings_percentage_correct : savings_percentage savings total_spending = 10 := by
  sorry

end savings_percentage_correct_l28_28653


namespace spherical_to_rectangular_conversion_l28_28382

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_conversion :
  spherical_to_rectangular 4 (Real.pi / 2) (Real.pi / 4) = (0, 2 * Real.sqrt 2, 2 * Real.sqrt 2) :=
by
  sorry

end spherical_to_rectangular_conversion_l28_28382


namespace find_x_if_perpendicular_l28_28110

-- Define the vectors a and b
def vector_a (x : ℝ) : ℝ × ℝ := (x, 3)
def vector_b (x : ℝ) : ℝ × ℝ := (2, x - 5)

-- Define the condition that vectors a and b are perpendicular
def perpendicular (x : ℝ) : Prop :=
  let a := vector_a x
  let b := vector_b x
  a.1 * b.1 + a.2 * b.2 = 0

-- Prove that x = 3 if a and b are perpendicular
theorem find_x_if_perpendicular :
  ∃ x : ℝ, perpendicular x ∧ x = 3 :=
by
  sorry

end find_x_if_perpendicular_l28_28110


namespace problem1_problem2_l28_28707

theorem problem1 (x : ℝ) : (4 * x ^ 2 + 12 * x - 7 ≤ 0) ∧ (a = 0) ∧ (x < -3 ∨ x > 3) → (-7/2 ≤ x ∧ x < -3) := by
  sorry

theorem problem2 (a : ℝ) : (∀ x : ℝ, 4 * x ^ 2 + 12 * x - 7 ≤ 0 → a - 3 ≤ x ∧ x ≤ a + 3) → (-5/2 ≤ a ∧ a ≤ -1/2) := by
  sorry

end problem1_problem2_l28_28707


namespace cubic_expression_identity_l28_28857

theorem cubic_expression_identity (x : ℝ) (hx : x + 1/x = 8) : 
  x^3 + 1/x^3 = 332 :=
sorry

end cubic_expression_identity_l28_28857


namespace problem_solution_l28_28533

-- Define a function to sum the digits of a number.
def sum_digits (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n % 1000) / 100
  let d3 := (n % 100) / 10
  let d4 := n % 10
  d1 + d2 + d3 + d4

-- Define the problem numbers.
def nums : List ℕ := [4272, 4281, 4290, 4311, 4320]

-- Check if the sum of digits is divisible by 9.
def divisible_by_9 (n : ℕ) : Prop :=
  sum_digits n % 9 = 0

-- Main theorem asserting the result.
theorem problem_solution :
  ∃ n ∈ nums, ¬divisible_by_9 n ∧ (n % 100 / 10) * (n % 10) = 14 := by
  sorry

end problem_solution_l28_28533


namespace ratio_meerkats_to_lion_cubs_l28_28752

-- Defining the initial conditions 
def initial_animals : ℕ := 68
def gorillas_sent : ℕ := 6
def hippo_adopted : ℕ := 1
def rhinos_rescued : ℕ := 3
def lion_cubs : ℕ := 8
def final_animal_count : ℕ := 90

-- Calculating the number of meerkats
def animals_before_meerkats : ℕ := initial_animals - gorillas_sent + hippo_adopted + rhinos_rescued + lion_cubs
def meerkats : ℕ := final_animal_count - animals_before_meerkats

-- Proving the ratio of meerkats to lion cubs is 2:1
theorem ratio_meerkats_to_lion_cubs : meerkats / lion_cubs = 2 := by
  -- Placeholder for the proof
  sorry

end ratio_meerkats_to_lion_cubs_l28_28752


namespace meeting_time_l28_28305

/--
The Racing Magic takes 150 seconds to circle the racing track once.
The Charging Bull makes 40 rounds of the track in an hour.
Prove that Racing Magic and Charging Bull meet at the starting point for the second time 
after 300 minutes.
-/
theorem meeting_time (rac_magic_time : ℕ) (chrg_bull_rounds_hour : ℕ)
  (h1 : rac_magic_time = 150) (h2 : chrg_bull_rounds_hour = 40) : 
  ∃ t: ℕ, t = 300 := 
by
  sorry

end meeting_time_l28_28305


namespace questionnaires_drawn_from_D_l28_28051

theorem questionnaires_drawn_from_D (a b c d : ℕ) (A_s B_s C_s D_s: ℕ) (common_diff: ℕ)
  (h1 : a + b + c + d = 1000)
  (h2 : b = a + common_diff)
  (h3 : c = a + 2 * common_diff)
  (h4 : d = a + 3 * common_diff)
  (h5 : A_s = 30 - common_diff)
  (h6 : B_s = 30)
  (h7 : C_s = 30 + common_diff)
  (h8 : D_s = 30 + 2 * common_diff)
  (h9 : A_s + B_s + C_s + D_s = 150)
  : D_s = 60 := sorry

end questionnaires_drawn_from_D_l28_28051


namespace find_units_min_selling_price_l28_28310

-- Definitions for the given conditions
def total_units : ℕ := 160
def cost_A : ℕ := 150
def cost_B : ℕ := 350
def total_cost : ℕ := 36000
def min_profit : ℕ := 11000

-- Part 1: Proving number of units purchased
theorem find_units :
  ∃ x y : ℕ,
    x + y = total_units ∧
    cost_A * x + cost_B * y = total_cost ∧
    y = total_units - x :=
by
  sorry

-- Part 2: Finding the minimum selling price per unit of model A for the profit condition
theorem min_selling_price (t : ℕ) :
  (∃ x y : ℕ,
    x + y = total_units ∧
    cost_A * x + cost_B * y = total_cost ∧
    y = total_units - x) →
  100 * (t - cost_A) + 60 * 2 * (t - cost_A) ≥ min_profit →
  t ≥ 200 :=
by
  sorry

end find_units_min_selling_price_l28_28310


namespace number_of_matches_in_first_set_l28_28022

theorem number_of_matches_in_first_set
  (avg_next_13_matches : ℕ := 15)
  (total_matches : ℕ := 35)
  (avg_all_matches : ℚ := 23.17142857142857)
  (x : ℕ := total_matches - 13) :
  x = 22 := by
  sorry

end number_of_matches_in_first_set_l28_28022


namespace find_the_number_l28_28130

theorem find_the_number (x : ℕ) (h : 18396 * x = 183868020) : x = 9990 :=
by
  sorry

end find_the_number_l28_28130


namespace student_l28_28061

-- Definition of the conditions
def mistaken_calculation (x : ℤ) : ℤ :=
  x + 10

def correct_calculation (x : ℤ) : ℤ :=
  x + 5

-- Theorem statement: Prove that the student's result is 10 more than the correct result
theorem student's_error {x : ℤ} : mistaken_calculation x = correct_calculation x + 5 :=
by
  sorry

end student_l28_28061


namespace equivalent_statements_l28_28888

variables (P Q : Prop)

theorem equivalent_statements : (¬Q → ¬P) ∧ (¬P ∨ Q) ↔ (P → Q) :=
by
  -- Proof goes here
  sorry

end equivalent_statements_l28_28888


namespace fixed_monthly_fee_l28_28313

theorem fixed_monthly_fee (x y : ℝ)
  (h₁ : x + y = 18.70)
  (h₂ : x + 3 * y = 34.10) : x = 11.00 :=
by sorry

end fixed_monthly_fee_l28_28313


namespace cube_red_face_probability_l28_28481

theorem cube_red_face_probability :
  let faces_total := 6
  let red_faces := 3
  let probability_red := red_faces / faces_total
  probability_red = 1 / 2 :=
by
  sorry

end cube_red_face_probability_l28_28481


namespace number_of_groups_is_correct_l28_28269

-- Define the number of students
def number_of_students : ℕ := 16

-- Define the group size
def group_size : ℕ := 4

-- Define the expected number of groups
def expected_number_of_groups : ℕ := 4

-- Prove the expected number of groups when grouping students into groups of four
theorem number_of_groups_is_correct :
  number_of_students / group_size = expected_number_of_groups := by
  sorry

end number_of_groups_is_correct_l28_28269


namespace largest_fraction_l28_28282

theorem largest_fraction (a b c d e : ℝ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) (h5 : d < e) :
  (b + d + e) / (a + c) > max ((a + b + e) / (c + d))
                        (max ((a + d) / (b + e))
                            (max ((b + c) / (a + e)) ((c + e) / (a + b + d)))) := 
sorry

end largest_fraction_l28_28282


namespace binary_multiplication_binary_result_l28_28863

-- Definitions for binary numbers
def bin_11011 : ℕ := 27 -- 11011 in binary is 27 in decimal
def bin_101 : ℕ := 5 -- 101 in binary is 5 in decimal

-- Theorem statement to prove the product of two binary numbers
theorem binary_multiplication : (bin_11011 * bin_101) = 135 := by
  sorry

-- Convert the result back to binary, expected to be 10000111
theorem binary_result : 135 = 8 * 16 + 7 := by
  sorry

end binary_multiplication_binary_result_l28_28863


namespace least_5_digit_divisible_l28_28103

theorem least_5_digit_divisible (n : ℕ) (h1 : n ≥ 10000) (h2 : n < 100000)
  (h3 : 15 ∣ n) (h4 : 12 ∣ n) (h5 : 18 ∣ n) : n = 10080 :=
by
  sorry

end least_5_digit_divisible_l28_28103


namespace find_first_number_l28_28295

theorem find_first_number (x : ℝ) :
  (20 + 40 + 60) / 3 = (x + 70 + 13) / 3 + 9 → x = 10 :=
by
  sorry

end find_first_number_l28_28295


namespace negative_option_is_B_l28_28876

-- Define the options as constants
def optionA : ℤ := -( -2 )
def optionB : ℤ := (-1) ^ 2023
def optionC : ℤ := |(-1) ^ 2|
def optionD : ℤ := (-5) ^ 2

-- Prove that the negative number among the options is optionB
theorem negative_option_is_B : optionB = -1 := 
by
  rw [optionB]
  sorry

end negative_option_is_B_l28_28876


namespace average_height_of_trees_l28_28608

def first_tree_height : ℕ := 1000
def half_tree_height : ℕ := first_tree_height / 2
def last_tree_height : ℕ := first_tree_height + 200

def total_height : ℕ := first_tree_height + 2 * half_tree_height + last_tree_height
def number_of_trees : ℕ := 4
def average_height : ℕ := total_height / number_of_trees

theorem average_height_of_trees :
  average_height = 800 :=
by
  -- This line contains a placeholder proof, the actual proof is omitted.
  sorry

end average_height_of_trees_l28_28608


namespace circle_tangent_to_line_at_parabola_focus_l28_28033

noncomputable def parabola_focus : (ℝ × ℝ) := (2, 0)

def line_eq (p : ℝ × ℝ) : Prop := p.2 = p.1

def circle_eq (center radius : ℝ) (p : ℝ × ℝ) : Prop := 
  (p.1 - center)^2 + p.2^2 = radius

theorem circle_tangent_to_line_at_parabola_focus : 
  ∀ p : ℝ × ℝ, (circle_eq 2 2 p ↔ (line_eq p ∧ p = parabola_focus)) := by
  sorry

end circle_tangent_to_line_at_parabola_focus_l28_28033


namespace temperature_at_night_l28_28125

theorem temperature_at_night 
  (T_morning : ℝ) 
  (T_rise_noon : ℝ) 
  (T_drop_night : ℝ) 
  (h1 : T_morning = 22) 
  (h2 : T_rise_noon = 6) 
  (h3 : T_drop_night = 10) : 
  (T_morning + T_rise_noon - T_drop_night = 18) :=
by 
  sorry

end temperature_at_night_l28_28125


namespace school_fee_l28_28224

theorem school_fee (a b c d e f g h i j k l : ℕ) (h1 : a = 2) (h2 : b = 100) (h3 : c = 1) (h4 : d = 50) (h5 : e = 5) (h6 : f = 20) (h7 : g = 3) (h8 : h = 10) (h9 : i = 4) (h10 : j = 5) (h11 : k = 4 ) (h12 : l = 50) :
  a * b + c * d + e * f + g * h + i * j + 3 * b + k * d + 2 * f + l * h + 6 * j = 980 := sorry

end school_fee_l28_28224


namespace relationship_of_new_stationary_points_l28_28131

noncomputable def g (x : ℝ) : ℝ := Real.sin x
noncomputable def h (x : ℝ) : ℝ := Real.log x
noncomputable def phi (x : ℝ) : ℝ := x^3

noncomputable def g' (x : ℝ) : ℝ := Real.cos x
noncomputable def h' (x : ℝ) : ℝ := 1 / x
noncomputable def phi' (x : ℝ) : ℝ := 3 * x^2

-- Definitions of the new stationary points
noncomputable def new_stationary_point_g (x : ℝ) : Prop := g x = g' x
noncomputable def new_stationary_point_h (x : ℝ) : Prop := h x = h' x
noncomputable def new_stationary_point_phi (x : ℝ) : Prop := phi x = phi' x

theorem relationship_of_new_stationary_points :
  ∃ (a b c : ℝ), (0 < a ∧ a < π) ∧ (1 < b ∧ b < Real.exp 1) ∧ (c ≠ 0) ∧
  new_stationary_point_g a ∧ new_stationary_point_h b ∧ new_stationary_point_phi c ∧
  c > b ∧ b > a :=
by
  sorry

end relationship_of_new_stationary_points_l28_28131


namespace inequality_xyz_l28_28292

theorem inequality_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : 1/x + 1/y + 1/z = 3) : 
  (x - 1) * (y - 1) * (z - 1) ≤ (1/4) * (x * y * z - 1) := 
by 
  sorry

end inequality_xyz_l28_28292


namespace speed_in_still_water_l28_28070

def upstream_speed : ℝ := 25
def downstream_speed : ℝ := 35

theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 30 :=
by
  sorry

end speed_in_still_water_l28_28070


namespace min_distance_from_origin_to_line_l28_28199

open Real

theorem min_distance_from_origin_to_line :
    ∀ x y : ℝ, (3 * x + 4 * y - 4 = 0) -> dist (0, 0) (x, y) = 4 / 5 :=
by
  sorry

end min_distance_from_origin_to_line_l28_28199


namespace factorization_eq1_factorization_eq2_l28_28207

-- Definitions for the given conditions
variables (a b x y m : ℝ)

-- The problem statement as Lean definitions and the goal theorems
def expr1 : ℝ := -6 * a * b + 3 * a^2 + 3 * b^2
def factored1 : ℝ := 3 * (a - b)^2

def expr2 : ℝ := y^2 * (2 - m) + x^2 * (m - 2)
def factored2 : ℝ := (m - 2) * (x + y) * (x - y)

-- Theorem statements for equivalence
theorem factorization_eq1 : expr1 a b = factored1 a b :=
by
  sorry

theorem factorization_eq2 : expr2 x y m = factored2 x y m :=
by
  sorry

end factorization_eq1_factorization_eq2_l28_28207


namespace rationalize_denominator_correct_l28_28435

noncomputable def rationalize_denominator : ℚ :=
  let A := 5
  let B := 49
  let C := 21
  A + B + C

theorem rationalize_denominator_correct :
  (let A := 5
   let B := 49
   let C := 21
   (A + B + C) = 75) :=
by
  sorry

end rationalize_denominator_correct_l28_28435


namespace complement_union_eq_l28_28749

open Set

noncomputable def U : Set ℕ := {0, 1, 2, 3, 4, 5}
noncomputable def A : Set ℕ := {1, 2, 4}
noncomputable def B : Set ℕ := {2, 3, 5}

theorem complement_union_eq:
  compl A ∪ B = {0, 2, 3, 5} :=
by
  sorry

end complement_union_eq_l28_28749


namespace range_of_alpha_plus_beta_l28_28623

theorem range_of_alpha_plus_beta (α β : ℝ) (h1 : 0 < α - β) (h2 : α - β < π) (h3 : 0 < α + 2 * β) (h4 : α + 2 * β < π) :
  0 < α + β ∧ α + β < π :=
sorry

end range_of_alpha_plus_beta_l28_28623


namespace product_remainder_l28_28184

theorem product_remainder
    (a b c : ℕ)
    (h₁ : a % 36 = 16)
    (h₂ : b % 36 = 8)
    (h₃ : c % 36 = 24) :
    (a * b * c) % 36 = 12 := 
    by
    sorry

end product_remainder_l28_28184


namespace perpendicular_lines_a_value_l28_28128

theorem perpendicular_lines_a_value :
  (∃ (a : ℝ), ∀ (x y : ℝ), (3 * y + x + 5 = 0) ∧ (4 * y + a * x + 3 = 0) → a = -12) :=
by
  sorry

end perpendicular_lines_a_value_l28_28128


namespace productivity_after_repair_l28_28547

-- Define the initial productivity and the increase factor.
def original_productivity : ℕ := 10
def increase_factor : ℝ := 1.5

-- Define the expected productivity after the improvement.
def expected_productivity : ℝ := 25

-- The theorem we need to prove.
theorem productivity_after_repair :
  original_productivity * (1 + increase_factor) = expected_productivity := by
  sorry

end productivity_after_repair_l28_28547


namespace not_always_product_greater_l28_28949

-- Define the premise and the conclusion
theorem not_always_product_greater (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b < 1) : a * b < a :=
sorry

end not_always_product_greater_l28_28949


namespace compute_expression_in_terms_of_k_l28_28020

-- Define the main theorem to be proven, with all conditions directly translated to Lean statements.
theorem compute_expression_in_terms_of_k
  (x y : ℝ)
  (h : (x^2 + y^2) / (x^2 - y^2) + (x^2 - y^2) / (x^2 + y^2) = k) :
    (x^8 + y^8) / (x^8 - y^8) - (x^8 - y^8) / (x^8 + y^8) = ((k - 2)^2 * (k + 2)^2) / (4 * k * (k^2 + 4)) :=
by
  sorry

end compute_expression_in_terms_of_k_l28_28020


namespace minimum_value_l28_28280

theorem minimum_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 1) :
  (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x)) ≥ 9 / 4 :=
by sorry

end minimum_value_l28_28280


namespace sum_of_fractions_l28_28163

theorem sum_of_fractions : 
  (1 / 10) + (2 / 10) + (3 / 10) + (4 / 10) + (10 / 10) + (11 / 10) + (15 / 10) + (20 / 10) + (25 / 10) + (50 / 10) = 14.1 :=
by sorry

end sum_of_fractions_l28_28163


namespace max_checkers_on_chessboard_l28_28101

theorem max_checkers_on_chessboard (n : ℕ) : 
  ∃ k : ℕ, k = 2 * n * (n / 2) := sorry

end max_checkers_on_chessboard_l28_28101


namespace problem_set_equiv_l28_28718

def positive_nats (x : ℕ) : Prop := x > 0

def problem_set : Set ℕ := {x | positive_nats x ∧ x - 3 < 2}

theorem problem_set_equiv : problem_set = {1, 2, 3, 4} :=
by 
  sorry

end problem_set_equiv_l28_28718


namespace remainder_17_pow_49_mod_5_l28_28828

theorem remainder_17_pow_49_mod_5 : (17^49) % 5 = 2 :=
by
  sorry

end remainder_17_pow_49_mod_5_l28_28828


namespace sum_in_base_b_l28_28368

-- Definitions needed to articulate the problem
def base_b_value (n : ℕ) (b : ℕ) : ℕ :=
  match n with
  | 12 => b + 2
  | 15 => b + 5
  | 16 => b + 6
  | 3146 => 3 * b^3 + 1 * b^2 + 4 * b + 6
  | _  => 0

def s_in_base_b (b : ℕ) : ℕ :=
  base_b_value 12 b + base_b_value 15 b + base_b_value 16 b

theorem sum_in_base_b (b : ℕ) (h : (base_b_value 12 b) * (base_b_value 15 b) * (base_b_value 16 b) = base_b_value 3146 b) :
  s_in_base_b b = 44 := by
  sorry

end sum_in_base_b_l28_28368


namespace prob_union_of_mutually_exclusive_l28_28068

-- Let's denote P as a probability function
variable {Ω : Type} (P : Set Ω → ℝ)

-- Define the mutually exclusive condition
def mutually_exclusive (A B : Set Ω) : Prop :=
  (A ∩ B) = ∅

-- State the theorem that we want to prove
theorem prob_union_of_mutually_exclusive (A B : Set Ω) 
  (h : mutually_exclusive A B) : P (A ∪ B) = P A + P B :=
sorry

end prob_union_of_mutually_exclusive_l28_28068


namespace find_function_f_l28_28553

theorem find_function_f
  (f : ℝ → ℝ)
  (H : ∀ x y, f x ^ 2 + f y ^ 2 = f (x + y) ^ 2) :
  ∀ x, f x = 0 := 
by 
  sorry

end find_function_f_l28_28553


namespace sum_of_areas_l28_28929

def base_width : ℕ := 3
def lengths : List ℕ := [1, 8, 27, 64, 125, 216]
def area (w l : ℕ) : ℕ := w * l
def total_area : ℕ := (lengths.map (area base_width)).sum

theorem sum_of_areas : total_area = 1323 := 
by sorry

end sum_of_areas_l28_28929


namespace min_girls_in_class_l28_28301

theorem min_girls_in_class : ∃ d : ℕ, 20 - d ≤ 2 * (d + 1) ∧ d ≥ 6 := by
  sorry

end min_girls_in_class_l28_28301


namespace john_books_per_day_l28_28160

theorem john_books_per_day (books_per_week := 2) (weeks := 6) (total_books := 48) :
  (total_books / (books_per_week * weeks) = 4) :=
by
  sorry

end john_books_per_day_l28_28160


namespace segment_length_in_meters_l28_28027

-- Conditions
def inch_to_meters : ℝ := 500
def segment_length_in_inches : ℝ := 7.25

-- Theorem to prove
theorem segment_length_in_meters : segment_length_in_inches * inch_to_meters = 3625 := by
  sorry

end segment_length_in_meters_l28_28027


namespace markers_per_box_l28_28725

theorem markers_per_box (original_markers new_boxes total_markers : ℕ) 
    (h1 : original_markers = 32) (h2 : new_boxes = 6) (h3 : total_markers = 86) : 
    total_markers - original_markers = new_boxes * 9 :=
by sorry

end markers_per_box_l28_28725


namespace salary_increase_l28_28225

theorem salary_increase (S : ℝ) (P : ℝ) (H0 : P > 0 )  
  (saved_last_year : ℝ := 0.10 * S)
  (salary_this_year : ℝ := S * (1 + P / 100))
  (saved_this_year : ℝ := 0.15 * salary_this_year)
  (H1 : saved_this_year = 1.65 * saved_last_year) :
  P = 10 :=
by
  sorry

end salary_increase_l28_28225


namespace tree_growth_factor_l28_28740

theorem tree_growth_factor 
  (initial_total : ℕ) 
  (initial_maples : ℕ) 
  (initial_lindens : ℕ) 
  (spring_total : ℕ) 
  (autumn_total : ℕ)
  (initial_maple_percentage : initial_maples = 3 * initial_total / 5)
  (spring_maple_percentage : initial_maples = spring_total / 5)
  (autumn_maple_percentage : initial_maples * 2 = autumn_total * 3 / 5) :
  autumn_total = 6 * initial_total :=
sorry

end tree_growth_factor_l28_28740


namespace Marty_combination_count_l28_28974

theorem Marty_combination_count :
  let num_colors := 4
  let num_methods := 3
  num_colors * num_methods = 12 :=
by
  let num_colors := 4
  let num_methods := 3
  sorry

end Marty_combination_count_l28_28974


namespace boat_crossing_time_l28_28839

theorem boat_crossing_time :
  ∀ (width_of_river speed_of_current speed_of_boat : ℝ),
  width_of_river = 1.5 →
  speed_of_current = 8 →
  speed_of_boat = 10 →
  (width_of_river / (Real.sqrt (speed_of_boat ^ 2 - speed_of_current ^ 2)) * 60) = 15 :=
by
  intros width_of_river speed_of_current speed_of_boat h1 h2 h3
  sorry

end boat_crossing_time_l28_28839


namespace proof_sum_of_drawn_kinds_l28_28272

def kindsGrains : Nat := 40
def kindsVegetableOils : Nat := 10
def kindsAnimalFoods : Nat := 30
def kindsFruitsAndVegetables : Nat := 20
def totalKindsFood : Nat := kindsGrains + kindsVegetableOils + kindsAnimalFoods + kindsFruitsAndVegetables
def sampleSize : Nat := 20
def samplingRatio : Nat := sampleSize / totalKindsFood

def numKindsVegetableOilsDrawn : Nat := kindsVegetableOils / 5
def numKindsFruitsAndVegetablesDrawn : Nat := kindsFruitsAndVegetables / 5
def sumVegetableOilsAndFruitsAndVegetablesDrawn : Nat := numKindsVegetableOilsDrawn + numKindsFruitsAndVegetablesDrawn

theorem proof_sum_of_drawn_kinds : sumVegetableOilsAndFruitsAndVegetablesDrawn = 6 := by
  have h1 : totalKindsFood = 100 := by rfl
  have h2 : samplingRatio = 1 / 5 := by
    calc
      sampleSize / totalKindsFood
      _ = 20 / 100 := rfl
      _ = 1 / 5 := by norm_num
  have h3 : numKindsVegetableOilsDrawn = 2 := by
    calc
      kindsVegetableOils / 5
      _ = 10 / 5 := rfl
      _ = 2 := by norm_num
  have h4 : numKindsFruitsAndVegetablesDrawn = 4 := by
    calc
      kindsFruitsAndVegetables / 5
      _ = 20 / 5 := rfl
      _ = 4 := by norm_num
  calc
    sumVegetableOilsAndFruitsAndVegetablesDrawn
    _ = numKindsVegetableOilsDrawn + numKindsFruitsAndVegetablesDrawn := rfl
    _ = 2 + 4 := by rw [h3, h4]
    _ = 6 := by norm_num

end proof_sum_of_drawn_kinds_l28_28272


namespace geometric_sequence_sum_l28_28007

theorem geometric_sequence_sum (n : ℕ) (S : ℕ → ℚ) (a : ℚ) :
  (∀ n, S n = (1 / 2) * 3^(n + 1) - a) →
  S 1 - (S 2 - S 1)^2 = (S 2 - S 1) * (S 3 - S 2) →
  a = 3 / 2 :=
by
  intros hSn hgeo
  sorry

end geometric_sequence_sum_l28_28007


namespace inequality_sqrt_l28_28573

open Real

theorem inequality_sqrt (x y : ℝ) :
  (sqrt (x^2 - 2*x*y) > sqrt (1 - y^2)) ↔ 
    ((x - y > 1 ∧ -1 < y ∧ y < 1) ∨ (x - y < -1 ∧ -1 < y ∧ y < 1)) :=
by
  sorry

end inequality_sqrt_l28_28573


namespace square_area_is_4802_l28_28165

-- Condition: the length of the diagonal of the square is 98 meters.
def diagonal (d : ℝ) := d = 98

-- Goal: Prove that the area of the square field is 4802 square meters.
theorem square_area_is_4802 (d : ℝ) (h : diagonal d) : ∃ (A : ℝ), A = 4802 := 
by sorry

end square_area_is_4802_l28_28165


namespace solve_for_b_l28_28840

theorem solve_for_b (a b : ℤ) (h1 : 3 * a + 2 = 5) (h2 : b - 4 * a = 2) : b = 6 :=
by
  -- proof goes here
  sorry

end solve_for_b_l28_28840


namespace probability_one_even_dice_l28_28535

noncomputable def probability_exactly_one_even (p : ℚ) : Prop :=
  ∃ (n : ℕ), (p = (4 * (1/2)^4 )) ∧ (n = 1) → p = 1/4

theorem probability_one_even_dice : probability_exactly_one_even (1/4) :=
by
  unfold probability_exactly_one_even
  sorry

end probability_one_even_dice_l28_28535


namespace correct_coefficient_l28_28429

-- Definitions based on given conditions
def isMonomial (expr : String) : Prop := true

def coefficient (expr : String) : ℚ :=
  if expr = "-a/3" then -1/3 else 0

-- Statement to prove
theorem correct_coefficient : coefficient "-a/3" = -1/3 :=
by
  sorry

end correct_coefficient_l28_28429


namespace waiting_time_boarding_l28_28143

noncomputable def time_taken_uber_to_house : ℕ := 10
noncomputable def time_taken_uber_to_airport : ℕ := 5 * time_taken_uber_to_house
noncomputable def time_taken_bag_check : ℕ := 15
noncomputable def time_taken_security : ℕ := 3 * time_taken_bag_check
noncomputable def total_process_time : ℕ := 180
noncomputable def remaining_time : ℕ := total_process_time - (time_taken_uber_to_house + time_taken_uber_to_airport + time_taken_bag_check + time_taken_security)
noncomputable def time_before_takeoff (B : ℕ) := 2 * B

theorem waiting_time_boarding : ∃ B : ℕ, B + time_before_takeoff B = remaining_time ∧ B = 20 := 
by 
  sorry

end waiting_time_boarding_l28_28143


namespace quadratic_root_and_coefficient_l28_28194

theorem quadratic_root_and_coefficient (k : ℝ) :
  (∃ x : ℝ, 5 * x^2 + k * x - 6 = 0 ∧ x = 2) →
  (∃ x₁ : ℝ, (5 * x₁^2 + k * x₁ - 6 = 0 ∧ x₁ ≠ 2) ∧ x₁ = -3/5 ∧ k = -7) :=
by
  sorry

end quadratic_root_and_coefficient_l28_28194


namespace perpendicular_line_slope_l28_28031

theorem perpendicular_line_slope (a : ℝ) :
  let M := (0, -1)
  let N := (2, 3)
  let k_MN := (N.2 - M.2) / (N.1 - M.1)
  k_MN * (-a / 2) = -1 → a = 1 :=
by
  intros M N k_MN H
  let M := (0, -1)
  let N := (2, 3)
  let k_MN := (N.2 - M.2) / (N.1 - M.1)
  sorry

end perpendicular_line_slope_l28_28031


namespace solve_for_y_l28_28124

theorem solve_for_y (y : ℚ) (h : |5 * y - 6| = 0) : y = 6 / 5 :=
by 
  sorry

end solve_for_y_l28_28124


namespace min_number_of_gennadys_l28_28821

theorem min_number_of_gennadys (a b v g : ℕ) (h_a : a = 45) (h_b : b = 122) (h_v : v = 27)
    (h_needed_g : g = 49) :
    (b - 1) - (a + v) = g :=
by
  -- We include sorry because we are focusing on the statement, not the proof itself.
  sorry

end min_number_of_gennadys_l28_28821


namespace hotel_accommodation_l28_28401

theorem hotel_accommodation :
  ∃ (arrangements : ℕ), arrangements = 27 :=
by
  -- problem statement
  let triple_room := 1
  let double_room := 1
  let single_room := 1
  let adults := 3
  let children := 2
  
  -- use the given conditions and properties of combinations to calculate arrangements
  sorry

end hotel_accommodation_l28_28401


namespace initial_people_in_castle_l28_28426

theorem initial_people_in_castle (P : ℕ) (provisions : ℕ → ℕ → ℕ) :
  (provisions P 90) - (provisions P 30) = provisions (P - 100) 90 ↔ P = 300 :=
by
  sorry

end initial_people_in_castle_l28_28426


namespace product_of_two_two_digit_numbers_greater_than_40_is_four_digit_l28_28666

-- Define the condition: both numbers are two-digit numbers greater than 40
def is_two_digit_and_greater_than_40 (n : ℕ) : Prop :=
  40 < n ∧ n < 100

-- Define the problem statement
theorem product_of_two_two_digit_numbers_greater_than_40_is_four_digit
  (a b : ℕ) (ha : is_two_digit_and_greater_than_40 a) (hb : is_two_digit_and_greater_than_40 b) :
  1000 ≤ a * b ∧ a * b < 10000 :=
by
  sorry

end product_of_two_two_digit_numbers_greater_than_40_is_four_digit_l28_28666


namespace min_cos_for_sqrt_l28_28466

theorem min_cos_for_sqrt (x : ℝ) (h : 2 * Real.cos x - 1 ≥ 0) : Real.cos x ≥ 1 / 2 := 
by
  sorry

end min_cos_for_sqrt_l28_28466


namespace tan_420_eq_sqrt3_l28_28905

theorem tan_420_eq_sqrt3 : Real.tan (420 * Real.pi / 180) = Real.sqrt 3 := 
by 
  -- Additional mathematical justification can go here.
  sorry

end tan_420_eq_sqrt3_l28_28905


namespace x_sq_y_sq_value_l28_28838

theorem x_sq_y_sq_value (x y : ℝ) 
  (h1 : x + y = 25) 
  (h2 : x^2 + y^2 = 169) 
  (h3 : x^3 * y^3 + y^3 * x^3 = 243) :
  x^2 * y^2 = 51984 := 
by 
  -- Proof to be added
  sorry

end x_sq_y_sq_value_l28_28838


namespace greatest_possible_integer_radius_l28_28612

theorem greatest_possible_integer_radius (r : ℕ) (h : ∀ (A : ℝ), A = Real.pi * (r : ℝ)^2 → A < 75 * Real.pi) : r ≤ 8 :=
by sorry

end greatest_possible_integer_radius_l28_28612


namespace vector_definition_l28_28943

-- Definition of a vector's characteristics
def hasCharacteristics (vector : Type) := ∃ (magnitude : ℝ) (direction : ℂ), true

-- The statement to prove: a vector is defined by having both magnitude and direction
theorem vector_definition (vector : Type) : hasCharacteristics vector := 
sorry

end vector_definition_l28_28943


namespace investment_ratio_l28_28521

variable (x : ℝ)
variable (p q t : ℝ)

theorem investment_ratio (h1 : 7 * p = 5 * q) (h2 : (7 * p * 8) / (5 * q * t) = 7 / 10) : t = 16 :=
by
  sorry

end investment_ratio_l28_28521


namespace power_inequality_l28_28196

theorem power_inequality 
( a b : ℝ )
( h1 : 0 < a )
( h2 : 0 < b )
( h3 : a ^ 1999 + b ^ 2000 ≥ a ^ 2000 + b ^ 2001 ) :
  a ^ 2000 + b ^ 2000 ≤ 2 :=
sorry

end power_inequality_l28_28196


namespace walt_total_interest_l28_28737

noncomputable def interest_8_percent (P_8 R_8 : ℝ) : ℝ :=
  P_8 * R_8

noncomputable def remaining_amount (P_total P_8 : ℝ) : ℝ :=
  P_total - P_8

noncomputable def interest_9_percent (P_9 R_9 : ℝ) : ℝ :=
  P_9 * R_9

noncomputable def total_interest (I_8 I_9 : ℝ) : ℝ :=
  I_8 + I_9

theorem walt_total_interest :
  let P_8 := 4000
  let R_8 := 0.08
  let P_total := 9000
  let R_9 := 0.09
  let I_8 := interest_8_percent P_8 R_8
  let P_9 := remaining_amount P_total P_8
  let I_9 := interest_9_percent P_9 R_9
  let I_total := total_interest I_8 I_9
  I_total = 770 := 
by
  sorry

end walt_total_interest_l28_28737


namespace plane_equation_l28_28226

variable (a b c : ℝ)
variable (ha : a ≠ 0)
variable (hb : b ≠ 0)
variable (hc : c ≠ 0)

theorem plane_equation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ x y z : ℝ, (x / a + y / b + z / c = 1) :=
sorry

end plane_equation_l28_28226


namespace Jimin_weight_l28_28203

variable (T J : ℝ)

theorem Jimin_weight (h1 : T - J = 4) (h2 : T + J = 88) : J = 42 :=
sorry

end Jimin_weight_l28_28203


namespace simple_random_sampling_correct_statements_l28_28926

theorem simple_random_sampling_correct_statements :
  let N : ℕ := 10
  -- Conditions for simple random sampling
  let is_finite (N : ℕ) := N > 0
  let is_non_sequential (N : ℕ) := N > 0 -- represents sampling does not require sequential order
  let without_replacement := true
  let equal_probability := true
  -- Verification
  (is_finite N) ∧ 
  (¬ is_non_sequential N) ∧ 
  without_replacement ∧ 
  equal_probability = true :=
by
  sorry

end simple_random_sampling_correct_statements_l28_28926


namespace woman_l28_28655

-- Define the variables and given conditions
variables (W S X : ℕ)
axiom s_eq : S = 27
axiom sum_eq : W + S = 84
axiom w_eq : W = 2 * S + X

theorem woman's_age_more_years : X = 3 :=
by
  -- Proof goes here
  sorry

end woman_l28_28655


namespace william_probability_l28_28114

def probability_of_correct_answer (p : ℚ) (q : ℚ) (n : ℕ) : ℚ :=
  1 - q^n

theorem william_probability :
  let p := 1 / 5
  let q := 4 / 5
  let n := 6
  probability_of_correct_answer p q n = 11529 / 15625 :=
by
  let p := 1 / 5
  let q := 4 / 5
  let n := 6
  unfold probability_of_correct_answer
  sorry

end william_probability_l28_28114


namespace sale_price_per_bearing_before_bulk_discount_l28_28753

-- Define the given conditions
def machines : ℕ := 10
def ball_bearings_per_machine : ℕ := 30
def total_ball_bearings : ℕ := machines * ball_bearings_per_machine

def normal_cost_per_bearing : ℝ := 1
def total_normal_cost : ℝ := total_ball_bearings * normal_cost_per_bearing

def bulk_discount : ℝ := 0.20
def sale_savings : ℝ := 120

-- The theorem we need to prove
theorem sale_price_per_bearing_before_bulk_discount (P : ℝ) :
  total_normal_cost - (total_ball_bearings * P * (1 - bulk_discount)) = sale_savings → 
  P = 0.75 :=
by sorry

end sale_price_per_bearing_before_bulk_discount_l28_28753


namespace smallest_number_in_systematic_sample_l28_28776

theorem smallest_number_in_systematic_sample (n m x : ℕ) (products : Finset ℕ) :
  n = 80 ∧ m = 5 ∧ products = Finset.range n ∧ x = 42 ∧ x ∈ products ∧ (∃ k : ℕ, x = (n / m) * k + 10) → 10 ∈ products :=
by
  sorry

end smallest_number_in_systematic_sample_l28_28776


namespace one_eq_a_l28_28392

theorem one_eq_a (x y z a : ℝ) (h₁: x + y + z = a) (h₂: 1/x + 1/y + 1/z = 1/a) :
  x = a ∨ y = a ∨ z = a :=
  sorry

end one_eq_a_l28_28392


namespace a_squared_plus_b_squared_equals_61_l28_28909

theorem a_squared_plus_b_squared_equals_61 (a b : ℝ) (h1 : a + b = -9) (h2 : a = 30 / b) : a^2 + b^2 = 61 :=
sorry

end a_squared_plus_b_squared_equals_61_l28_28909


namespace not_suitable_for_storing_l28_28011

-- Define the acceptable temperature range conditions for storing dumplings
def acceptable_range (t : ℤ) : Prop :=
  -20 ≤ t ∧ t ≤ -16

-- Define the specific temperatures under consideration
def temp_A : ℤ := -17
def temp_B : ℤ := -18
def temp_C : ℤ := -19
def temp_D : ℤ := -22

-- Define a theorem stating that temp_D is not in the acceptable range
theorem not_suitable_for_storing (t : ℤ) (h : t = temp_D) : ¬ acceptable_range t :=
by {
  sorry
}

end not_suitable_for_storing_l28_28011


namespace find_math_marks_l28_28158

theorem find_math_marks
  (e p c b : ℕ)
  (n : ℕ)
  (a : ℚ)
  (M : ℕ) :
  e = 96 →
  p = 82 →
  c = 87 →
  b = 92 →
  n = 5 →
  a = 90.4 →
  (a * n = (e + p + c + b + M)) →
  M = 95 :=
by intros
   sorry

end find_math_marks_l28_28158


namespace number_of_factors_in_224_l28_28021

def smallest_is_half_largest (n1 n2 : ℕ) : Prop :=
  n1 * 2 = n2

theorem number_of_factors_in_224 :
  ∃ n1 n2 n3 : ℕ, n1 * n2 * n3 = 224 ∧ smallest_is_half_largest (min n1 (min n2 n3)) (max n1 (max n2 n3)) ∧
    (if h : n1 < n2 ∧ n1 < n3 then
      if h2 : n2 < n3 then 
        smallest_is_half_largest n1 n3 
        else 
        smallest_is_half_largest n1 n2 
    else if h : n2 < n1 ∧ n2 < n3 then 
      if h2 : n1 < n3 then 
        smallest_is_half_largest n2 n3 
        else 
        smallest_is_half_largest n2 n1 
    else 
      if h2 : n1 < n2 then 
        smallest_is_half_largest n3 n2 
        else 
        smallest_is_half_largest n3 n1) = true ∧ 
    (if h : n1 < n2 ∧ n1 < n3 then
       if h2 : n2 < n3 then 
         n1 * n2 * n3 
         else 
         n1 * n3 * n2 
     else if h : n2 < n1 ∧ n2 < n3 then 
       if h2 : n1 < n3 then 
         n2 * n1 * n3
         else 
         n2 * n3 * n1 
     else 
       if h2 : n1 < n2 then 
         n3 * n1 * n2 
         else 
         n3 * n2 * n1) = 224 := sorry

end number_of_factors_in_224_l28_28021


namespace symmetric_about_origin_implies_odd_l28_28917

variable {F : Type} [Field F] (f : F → F)
variable (x : F)

theorem symmetric_about_origin_implies_odd (H : ∀ x, f (-x) = -f x) : f x + f (-x) = 0 := 
by 
  sorry

end symmetric_about_origin_implies_odd_l28_28917


namespace probability_of_yellow_l28_28833

-- Definitions of the given conditions
def red_jelly_beans := 4
def green_jelly_beans := 8
def yellow_jelly_beans := 9
def blue_jelly_beans := 5
def total_jelly_beans := red_jelly_beans + green_jelly_beans + yellow_jelly_beans + blue_jelly_beans

-- Theorem statement
theorem probability_of_yellow :
  (yellow_jelly_beans : ℚ) / total_jelly_beans = 9 / 26 :=
by
  sorry

end probability_of_yellow_l28_28833


namespace calculation_correct_l28_28264

theorem calculation_correct : 469111 * 9999 = 4690428889 := 
by sorry

end calculation_correct_l28_28264


namespace some_number_value_l28_28968

theorem some_number_value (some_number : ℝ) (h : (some_number * 14) / 100 = 0.045388) :
  some_number = 0.3242 :=
sorry

end some_number_value_l28_28968


namespace ratio_of_turtles_l28_28920

noncomputable def initial_turtles_owen : ℕ := 21
noncomputable def initial_turtles_johanna : ℕ := initial_turtles_owen - 5
noncomputable def turtles_johanna_after_month : ℕ := initial_turtles_johanna / 2
noncomputable def turtles_owen_after_month : ℕ := 50 - turtles_johanna_after_month

theorem ratio_of_turtles (a b : ℕ) (h1 : a = 21) (h2 : b = 5) (h3 : initial_turtles_owen = a) (h4 : initial_turtles_johanna = initial_turtles_owen - b) 
(h5 : turtles_johanna_after_month = initial_turtles_johanna / 2) (h6 : turtles_owen_after_month = 50 - turtles_johanna_after_month) : 
turtles_owen_after_month / initial_turtles_owen = 2 := by
  sorry

end ratio_of_turtles_l28_28920


namespace mike_total_spent_l28_28754

noncomputable def total_spent_by_mike (food_cost wallet_cost shirt_cost shoes_cost belt_cost 
  discounted_shirt_cost discounted_shoes_cost discounted_belt_cost : ℝ) : ℝ :=
  food_cost + wallet_cost + discounted_shirt_cost + discounted_shoes_cost + discounted_belt_cost

theorem mike_total_spent :
  let food_cost := 30
  let wallet_cost := food_cost + 60
  let shirt_cost := wallet_cost / 3
  let shoes_cost := 2 * wallet_cost
  let belt_cost := shoes_cost - 45
  let discounted_shirt_cost := shirt_cost - (0.2 * shirt_cost)
  let discounted_shoes_cost := shoes_cost - (0.15 * shoes_cost)
  let discounted_belt_cost := belt_cost - (0.1 * belt_cost)
  total_spent_by_mike food_cost wallet_cost shirt_cost shoes_cost belt_cost
    discounted_shirt_cost discounted_shoes_cost discounted_belt_cost = 418.50 := by
  sorry

end mike_total_spent_l28_28754


namespace math_proof_problem_l28_28606

noncomputable def sum_of_distinct_squares (a b c : ℕ) : ℕ :=
3 * ((a^2 + b^2 + c^2 : ℕ))

theorem math_proof_problem (a b c : ℕ)
  (h1 : a + b + c = 27)
  (h2 : Nat.gcd a b + Nat.gcd b c + Nat.gcd c a = 11) :
  sum_of_distinct_squares a b c = 2274 :=
sorry

end math_proof_problem_l28_28606


namespace blue_die_prime_yellow_die_power_2_probability_l28_28808

def prime_numbers : Finset ℕ := {2, 3, 5, 7}

def powers_of_2 : Finset ℕ := {1, 2, 4, 8}

def total_outcomes : ℕ := 8 * 8

def successful_outcomes : ℕ := prime_numbers.card * powers_of_2.card

def probability (x y : Finset ℕ) : ℚ := (x.card * y.card) / (total_outcomes : ℚ)

theorem blue_die_prime_yellow_die_power_2_probability :
  probability prime_numbers powers_of_2 = 1 / 4 :=
by
  sorry

end blue_die_prime_yellow_die_power_2_probability_l28_28808


namespace time_correct_l28_28923

theorem time_correct {t : ℝ} (h : 0 < t ∧ t < 60) :
  |6 * (t + 5) - (90 + 0.5 * (t - 4))| = 180 → t = 43 := by
  sorry

end time_correct_l28_28923


namespace find_coordinates_of_Q_l28_28937

theorem find_coordinates_of_Q (x y : ℝ) (P : ℝ × ℝ) (hP : P = (1, 2))
    (perp : x + 2 * y = 0) (length : x^2 + y^2 = 5) :
    (x, y) = (-2, 1) :=
by
  -- Proof should go here
  sorry

end find_coordinates_of_Q_l28_28937


namespace problem_statement_l28_28830

noncomputable def a : ℕ := by
  -- The smallest positive two-digit multiple of 3
  let a := Finset.range 100 \ Finset.range 10
  let multiples := a.filter (λ n => n % 3 = 0)
  exact multiples.min' ⟨12, sorry⟩

noncomputable def b : ℕ := by
  -- The smallest positive three-digit multiple of 4
  let b := Finset.range 1000 \ Finset.range 100
  let multiples := b.filter (λ n => n % 4 = 0)
  exact multiples.min' ⟨100, sorry⟩

theorem problem_statement : a + b = 112 := by
  sorry

end problem_statement_l28_28830


namespace findDivisor_l28_28212

def addDivisorProblem : Prop :=
  ∃ d : ℕ, ∃ n : ℕ, n = 172835 + 21 ∧ d ∣ n ∧ d = 21

theorem findDivisor : addDivisorProblem :=
by
  sorry

end findDivisor_l28_28212


namespace smallest_positive_period_f_max_value_f_interval_min_value_f_interval_l28_28440

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.sin x - Real.cos x) + 1

theorem smallest_positive_period_f : ∃ k > 0, ∀ x, f (x + k) = f x := 
sorry

theorem max_value_f_interval : ∃ x ∈ Set.Icc (Real.pi / 8) (3 * Real.pi / 4), f x = Real.sqrt 2 :=
sorry

theorem min_value_f_interval : ∃ x ∈ Set.Icc (Real.pi / 8) (3 * Real.pi / 4), f x = -1 :=
sorry

end smallest_positive_period_f_max_value_f_interval_min_value_f_interval_l28_28440


namespace chord_constant_sum_l28_28135

theorem chord_constant_sum (d : ℝ) (h : d = 1/2) :
  ∀ A B : ℝ × ℝ, (A.2 = A.1^2) → (B.2 = B.1^2) →
  (∃ m : ℝ, A.2 = m * A.1 + d ∧ B.2 = m * B.1 + d) →
  (∃ D : ℝ × ℝ, D = (0, d) ∧ (∃ s : ℝ,
    s = (1 / ((A.1 - D.1)^2 + (A.2 - D.2)^2) + 1 / ((B.1 - D.1)^2 + (B.2 - D.2)^2)) ∧ s = 4)) :=
by 
  sorry

end chord_constant_sum_l28_28135


namespace total_cost_chairs_l28_28649

def living_room_chairs : Nat := 3
def kitchen_chairs : Nat := 6
def dining_room_chairs : Nat := 8
def outdoor_patio_chairs : Nat := 12

def living_room_price : Nat := 75
def kitchen_price : Nat := 50
def dining_room_price : Nat := 100
def outdoor_patio_price : Nat := 60

theorem total_cost_chairs : 
  living_room_chairs * living_room_price + 
  kitchen_chairs * kitchen_price + 
  dining_room_chairs * dining_room_price + 
  outdoor_patio_chairs * outdoor_patio_price = 2045 := by
  sorry

end total_cost_chairs_l28_28649


namespace probability_not_passing_l28_28812

noncomputable def probability_of_passing : ℚ := 4 / 7

theorem probability_not_passing (h : probability_of_passing = 4 / 7) : 1 - probability_of_passing = 3 / 7 :=
by
  sorry

end probability_not_passing_l28_28812


namespace simplify_and_evaluate_l28_28468

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 2 - 1) : 
  (1 - 1 / (a + 1)) * ((a^2 + 2 * a + 1) / a) = Real.sqrt 2 := 
by {
  sorry
}

end simplify_and_evaluate_l28_28468


namespace part1_part2_l28_28760

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := sorry

theorem part2 (a : ℝ) : 
  (∀ x, f x a > -a) ↔ (a > -3/2) := sorry

end part1_part2_l28_28760


namespace determinant_of_2x2_matrix_l28_28626

theorem determinant_of_2x2_matrix :
  let a := 2
  let b := 4
  let c := 1
  let d := 3
  a * d - b * c = 2 := by
  sorry

end determinant_of_2x2_matrix_l28_28626


namespace blue_socks_count_l28_28286

-- Defining the total number of socks
def total_socks : ℕ := 180

-- Defining the number of white socks as two thirds of the total socks
def white_socks : ℕ := (2 * total_socks) / 3

-- Defining the number of blue socks as the difference between total socks and white socks
def blue_socks : ℕ := total_socks - white_socks

-- The theorem to prove
theorem blue_socks_count : blue_socks = 60 := by
  sorry

end blue_socks_count_l28_28286


namespace candy_problem_l28_28099

theorem candy_problem (N a S : ℕ) 
  (h1 : ∀ i : ℕ, i < N → a = S - 7 - a)
  (h2 : ∀ i : ℕ, i < N → a > 1)
  (h3 : S = N * a) : 
  S = 21 :=
by
  sorry

end candy_problem_l28_28099


namespace point_on_parabola_l28_28663

def parabola (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem point_on_parabola : parabola (1/2) = 0 := 
by sorry

end point_on_parabola_l28_28663


namespace max_value_of_trig_function_l28_28742

theorem max_value_of_trig_function : 
  ∀ x, 3 * Real.sin x + 4 * Real.cos x ≤ 5 := sorry


end max_value_of_trig_function_l28_28742


namespace age_of_b_l28_28688

variable (a b c d : ℕ)
variable (h1 : a = b + 2)
variable (h2 : b = 2 * c)
variable (h3 : d = b / 2)
variable (h4 : a + b + c + d = 44)

theorem age_of_b : b = 14 :=
by 
  sorry

end age_of_b_l28_28688


namespace mass_percentages_correct_l28_28357

noncomputable def mass_percentage_of_Ba (x y : ℝ) : ℝ :=
  ( ((x / 175.323) * 137.327 + (y / 153.326) * 137.327) / (x + y) ) * 100

noncomputable def mass_percentage_of_F (x y : ℝ) : ℝ :=
  ( ((x / 175.323) * (2 * 18.998)) / (x + y) ) * 100

noncomputable def mass_percentage_of_O (x y : ℝ) : ℝ :=
  ( ((y / 153.326) * 15.999) / (x + y) ) * 100

theorem mass_percentages_correct (x y : ℝ) :
  ∃ (Ba F O : ℝ), 
    Ba = mass_percentage_of_Ba x y ∧
    F = mass_percentage_of_F x y ∧
    O = mass_percentage_of_O x y :=
sorry

end mass_percentages_correct_l28_28357


namespace find_k_l28_28513

theorem find_k (k b : ℤ) (h1 : -x^2 - (k + 10) * x - b = -(x - 2) * (x - 4))
  (h2 : b = 8) : k = -16 :=
sorry

end find_k_l28_28513


namespace polynomial_remainder_l28_28469

theorem polynomial_remainder (p : Polynomial ℝ) :
  (p.eval 2 = 3) → (p.eval 3 = 9) → ∃ q : Polynomial ℝ, p = (Polynomial.X - 2) * (Polynomial.X - 3) * q + (6 * Polynomial.X - 9) :=
by
  sorry

end polynomial_remainder_l28_28469


namespace proposition_p_proposition_q_l28_28358

theorem proposition_p : ∅ ≠ ({∅} : Set (Set Empty)) := by
  sorry

theorem proposition_q (A : Set ℕ) (B : Set (Set ℕ)) (hA : A = {1, 2})
    (hB : B = {x | x ⊆ A}) : A ∈ B := by
  sorry

end proposition_p_proposition_q_l28_28358


namespace height_percentage_difference_l28_28320

theorem height_percentage_difference
  (h_B h_A : ℝ)
  (hA_def : h_A = h_B * 0.55) :
  ((h_B - h_A) / h_A) * 100 = 81.82 := by 
  sorry

end height_percentage_difference_l28_28320


namespace trig_identity_proof_l28_28398

theorem trig_identity_proof 
  (α : ℝ) 
  (h1 : Real.sin (4 * α) = 2 * Real.sin (2 * α) * Real.cos (2 * α))
  (h2 : Real.cos (4 * α) = Real.cos (2 * α) ^ 2 - Real.sin (2 * α) ^ 2) : 
  (1 - 2 * Real.sin (2 * α) ^ 2) / (1 - Real.sin (4 * α)) = 
  (1 + Real.tan (2 * α)) / (1 - Real.tan (2 * α)) := 
by 
  sorry

end trig_identity_proof_l28_28398


namespace range_of_m_l28_28511

theorem range_of_m (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (∀ x y : ℝ, 0 < x → 0 < y → (2 * y / x + 9 * x / (2 * y) ≥ m^2 + m))
  ↔ (-3 ≤ m ∧ m ≤ 2) := sorry

end range_of_m_l28_28511


namespace initial_amount_l28_28952

theorem initial_amount (P : ℝ) :
  (P * 1.0816 - P * 1.08 = 3.0000000000002274) → P = 1875.0000000001421 :=
by
  sorry

end initial_amount_l28_28952


namespace minimum_candies_to_identify_coins_l28_28720

-- Set up the problem: define the relevant elements.
inductive Coin : Type
| C1 : Coin
| C2 : Coin
| C3 : Coin
| C4 : Coin
| C5 : Coin

def values : List ℕ := [1, 2, 5, 10, 20]

-- Statement of the problem in Lean 4, no means to identify which is which except through purchases and change from vending machine.
theorem minimum_candies_to_identify_coins : ∃ n : ℕ, n = 4 :=
by
  -- Skipping the proof
  sorry

end minimum_candies_to_identify_coins_l28_28720


namespace expression_equals_neg_one_l28_28185

theorem expression_equals_neg_one (a b c : ℝ) (h : a + b + c = 0) :
  (|a| / a) + (|b| / b) + (|c| / c) + (|a * b| / (a * b)) + (|a * c| / (a * c)) + (|b * c| / (b * c)) + (|a * b * c| / (a * b * c)) = -1 :=
  sorry

end expression_equals_neg_one_l28_28185


namespace smallest_four_digit_divisible_by_4_and_5_l28_28206

theorem smallest_four_digit_divisible_by_4_and_5 : 
  ∃ n, (n % 4 = 0) ∧ (n % 5 = 0) ∧ 1000 ≤ n ∧ n < 10000 ∧ 
  ∀ m, (m % 4 = 0) ∧ (m % 5 = 0) ∧ 1000 ≤ m ∧ m < 10000 → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_4_and_5_l28_28206


namespace relationship_among_a_b_c_l28_28050

-- Defining the properties and conditions of the function
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- Defining the function f based on the condition
noncomputable def f (x m : ℝ) : ℝ := 2 ^ |x - m| - 1

-- Defining the constants a, b, c
noncomputable def a : ℝ := f (Real.log 3 / Real.log 0.5) 0
noncomputable def b : ℝ := f (Real.log 5 / Real.log 2) 0
noncomputable def c : ℝ := f 0 0

-- The theorem stating the relationship among a, b, and c
theorem relationship_among_a_b_c : c < a ∧ a < b := by
  sorry

end relationship_among_a_b_c_l28_28050


namespace total_puppies_is_74_l28_28878

-- Define the number of puppies adopted each week based on the given conditions
def number_of_puppies_first_week : Nat := 20
def number_of_puppies_second_week : Nat := 2 * number_of_puppies_first_week / 5
def number_of_puppies_third_week : Nat := 2 * number_of_puppies_second_week
def number_of_puppies_fourth_week : Nat := number_of_puppies_first_week + 10

-- Define the total number of puppies
def total_number_of_puppies : Nat :=
  number_of_puppies_first_week + number_of_puppies_second_week + number_of_puppies_third_week + number_of_puppies_fourth_week

-- Proof statement: Prove that the total number of puppies is 74
theorem total_puppies_is_74 : total_number_of_puppies = 74 := by
  sorry

end total_puppies_is_74_l28_28878


namespace seeds_in_first_plot_l28_28170

theorem seeds_in_first_plot (x : ℕ) (h1 : 0 < x)
  (h2 : 200 = 200)
  (h3 : 0.25 * (x : ℝ) = 0.25 * (x : ℝ))
  (h4 : 0.35 * 200 = 70)
  (h5 : (0.25 * (x : ℝ) + 70) / (x + 200) = 0.29) :
  x = 300 :=
by sorry

end seeds_in_first_plot_l28_28170


namespace problem_statement_l28_28476

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 2) :
  (1 < b ∧ b < 2) ∧ (ab < 1) :=
by
  sorry

end problem_statement_l28_28476


namespace sum_is_402_3_l28_28964

def sum_of_numbers := 3 + 33 + 333 + 33.3

theorem sum_is_402_3 : sum_of_numbers = 402.3 := by
  sorry

end sum_is_402_3_l28_28964


namespace find_angle_A_l28_28954

variable (a b c : ℝ)
variable (A : ℝ)

axiom triangle_ABC : a = Real.sqrt 3 ∧ b = 1 ∧ c = 2

theorem find_angle_A : a = Real.sqrt 3 ∧ b = 1 ∧ c = 2 → A = Real.pi / 3 :=
by
  intro h
  sorry

end find_angle_A_l28_28954


namespace inequality_solution_l28_28386

theorem inequality_solution (x : ℝ) :
  (∀ y : ℝ, (0 < y) → (4 * (x^2 * y^2 + x * y^3 + 4 * y^2 + 4 * x * y) / (x + y) > 3 * x^2 * y + y)) ↔ (1 < x) :=
by
  sorry

end inequality_solution_l28_28386


namespace time_to_paint_one_house_l28_28522

theorem time_to_paint_one_house (houses : ℕ) (total_time_hours : ℕ) (total_time_minutes : ℕ) 
  (minutes_per_hour : ℕ) (h1 : houses = 9) (h2 : total_time_hours = 3) 
  (h3 : minutes_per_hour = 60) (h4 : total_time_minutes = total_time_hours * minutes_per_hour) : 
  (total_time_minutes / houses) = 20 :=
by
  sorry

end time_to_paint_one_house_l28_28522


namespace interest_rate_is_six_percent_l28_28311

noncomputable def amount : ℝ := 1120
noncomputable def principal : ℝ := 979.0209790209791
noncomputable def time_years : ℝ := 2 + 2 / 5

noncomputable def total_interest (A P: ℝ) : ℝ := A - P

noncomputable def interest_rate_per_annum (I P T: ℝ) : ℝ := I / (P * T) * 100

theorem interest_rate_is_six_percent :
  interest_rate_per_annum (total_interest amount principal) principal time_years = 6 := 
by
  sorry

end interest_rate_is_six_percent_l28_28311


namespace suff_not_necessary_no_real_solutions_l28_28255

theorem suff_not_necessary_no_real_solutions :
  ∀ m : ℝ, |m| < 1 → (m : ℝ)^2 < 4 ∧ ∃ x, x^2 - m * x + 1 = 0 →
  ∀ a b : ℝ, (a = 1) ∧ (b = -m) ∧ (c = 1) → (b^2 - 4 * a * c) < 0 ∧ (m > -2) ∧ (m < 2) :=
by
  sorry

end suff_not_necessary_no_real_solutions_l28_28255


namespace f_2015_l28_28674

def f : ℤ → ℤ := sorry

axiom f1 : f 1 = 1
axiom f2 : f 2 = 0
axiom functional_eq (x y : ℤ) : f (x + y) = f x * f (1 - y) + f (1 - x) * f y

theorem f_2015 : f 2015 = 1 ∨ f 2015 = -1 :=
sorry

end f_2015_l28_28674


namespace smallest_k_l28_28174

-- Definitions used in the conditions
def poly1 (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1
def poly2 (z : ℂ) (k : ℕ) : ℂ := z^k - 1

-- Lean 4 statement for the problem
theorem smallest_k (k : ℕ) (hk : k = 120) :
  ∀ z : ℂ, poly1 z ∣ poly2 z k :=
sorry

end smallest_k_l28_28174


namespace equilibrium_force_l28_28609

def f1 : ℝ × ℝ := (-2, -1)
def f2 : ℝ × ℝ := (-3, 2)
def f3 : ℝ × ℝ := (4, -3)
def expected_f4 : ℝ × ℝ := (1, 2)

theorem equilibrium_force :
  (1, 2) = -(f1 + f2 + f3) := 
by
  sorry

end equilibrium_force_l28_28609


namespace four_digit_numbers_no_5s_8s_l28_28127

def count_valid_four_digit_numbers : Nat :=
  let thousand_place := 7  -- choices: 1, 2, 3, 4, 6, 7, 9
  let other_places := 8  -- choices: 0, 1, 2, 3, 4, 6, 7, 9
  thousand_place * other_places * other_places * other_places

theorem four_digit_numbers_no_5s_8s : count_valid_four_digit_numbers = 3584 :=
by
  rfl

end four_digit_numbers_no_5s_8s_l28_28127


namespace combination_problem_l28_28775

theorem combination_problem (x : ℕ) (hx_pos : 0 < x) (h_comb : Nat.choose 9 x = Nat.choose 9 (2 * x + 3)) : x = 2 :=
by {
  sorry
}

end combination_problem_l28_28775


namespace integer_solutions_l28_28774

theorem integer_solutions (t : ℤ) : 
  ∃ x y : ℤ, 5 * x - 7 * y = 3 ∧ x = 7 * t - 12 ∧ y = 5 * t - 9 :=
by
  sorry

end integer_solutions_l28_28774


namespace range_of_a_l28_28087

theorem range_of_a (a : ℝ) :
  (¬ ∃ x₀ : ℝ, x₀^2 - a*x₀ + 2 < 0) ↔ (a^2 ≤ 8) :=
by
  sorry

end range_of_a_l28_28087


namespace father_son_age_problem_l28_28523

theorem father_son_age_problem
  (F S Y : ℕ)
  (h1 : F = 3 * S)
  (h2 : F = 45)
  (h3 : F + Y = 2 * (S + Y)) :
  Y = 15 :=
sorry

end father_son_age_problem_l28_28523


namespace problem_statement_l28_28060

-- Let's define the conditions
def num_blue_balls : ℕ := 8
def num_green_balls : ℕ := 7
def total_balls : ℕ := num_blue_balls + num_green_balls

-- Function to calculate combinations (binomial coefficients)
def combination (n r : ℕ) : ℕ :=
  n.choose r

-- Specific combinations for this problem
def blue_ball_ways : ℕ := combination num_blue_balls 3
def green_ball_ways : ℕ := combination num_green_balls 2
def total_ways : ℕ := combination total_balls 5

-- The number of favorable outcomes
def favorable_outcomes : ℕ := blue_ball_ways * green_ball_ways

-- The probability
def probability : ℚ := favorable_outcomes / total_ways

-- The theorem stating our result
theorem problem_statement : probability = 1176/3003 := by
  sorry

end problem_statement_l28_28060


namespace field_dimensions_l28_28137

theorem field_dimensions (W L : ℕ) (h1 : L = 2 * W) (h2 : 2 * L + 2 * W = 600) : W = 100 ∧ L = 200 :=
sorry

end field_dimensions_l28_28137


namespace lowest_price_for_16_oz_butter_l28_28877

-- Define the constants
def price_single_16_oz_package : ℝ := 7
def price_8_oz_package : ℝ := 4
def price_4_oz_package : ℝ := 2
def discount_4_oz_package : ℝ := 0.5

-- Calculate the discounted price for a 4 oz package
def discounted_price_4_oz_package : ℝ := price_4_oz_package * discount_4_oz_package

-- Calculate the total price for two discounted 4 oz packages
def total_price_two_discounted_4_oz_packages : ℝ := 2 * discounted_price_4_oz_package

-- Calculate the total price using the 8 oz package and two discounted 4 oz packages
def total_price_using_coupon : ℝ := price_8_oz_package + total_price_two_discounted_4_oz_packages

-- State the property to prove
theorem lowest_price_for_16_oz_butter :
  min price_single_16_oz_package total_price_using_coupon = 6 :=
sorry

end lowest_price_for_16_oz_butter_l28_28877


namespace scholarship_amount_l28_28789

-- Definitions
def tuition_per_semester : ℕ := 22000
def parents_contribution : ℕ := tuition_per_semester / 2
def work_hours : ℕ := 200
def hourly_wage : ℕ := 10
def work_earnings : ℕ := work_hours * hourly_wage
def remaining_tuition : ℕ := tuition_per_semester - parents_contribution - work_earnings

-- Theorem to prove the scholarship amount
theorem scholarship_amount (S : ℕ) (h : 3 * S = remaining_tuition) : S = 3000 :=
by
  sorry

end scholarship_amount_l28_28789


namespace profit_percentage_l28_28605

theorem profit_percentage (SP CP : ℤ) (h_SP : SP = 1170) (h_CP : CP = 975) :
  ((SP - CP : ℤ) * 100) / CP = 20 :=
by 
  sorry

end profit_percentage_l28_28605


namespace number_of_groups_l28_28583

-- Define constants
def new_players : Nat := 48
def returning_players : Nat := 6
def players_per_group : Nat := 6

-- Define the theorem to be proven
theorem number_of_groups :
  (new_players + returning_players) / players_per_group = 9 :=
by
  sorry

end number_of_groups_l28_28583


namespace sqrt_one_fourth_l28_28534

theorem sqrt_one_fourth :
  {x : ℚ | x^2 = 1/4} = {1/2, -1/2} :=
by sorry

end sqrt_one_fourth_l28_28534


namespace percentage_of_failed_candidates_l28_28155

theorem percentage_of_failed_candidates :
  let total_candidates := 2000
  let girls := 900
  let boys := total_candidates - girls
  let boys_passed := 32 / 100 * boys
  let girls_passed := 32 / 100 * girls
  let total_passed := boys_passed + girls_passed
  let total_failed := total_candidates - total_passed
  let percentage_failed := (total_failed / total_candidates) * 100
  percentage_failed = 68 :=
by
  -- Proof goes here
  sorry

end percentage_of_failed_candidates_l28_28155


namespace quadrilateral_EFGH_l28_28887

variable {EF FG GH HE EH : ℤ}

theorem quadrilateral_EFGH (h1 : EF = 6) (h2 : FG = 18) (h3 : GH = 6) (h4 : HE = 10) (h5 : 12 < EH) (h6 : EH < 24) : EH = 12 := 
sorry

end quadrilateral_EFGH_l28_28887


namespace largest_4_digit_divisible_by_35_l28_28751

theorem largest_4_digit_divisible_by_35 : ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ (n % 35 = 0) ∧ (∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999) ∧ (m % 35 = 0) → m ≤ n) ∧ n = 9985 := 
by sorry

end largest_4_digit_divisible_by_35_l28_28751


namespace female_students_in_sample_l28_28157

/-- In a high school, there are 500 male students and 400 female students in the first grade. 
    If a random sample of size 45 is taken from the students of this grade using stratified sampling by gender, 
    the number of female students in the sample is 20. -/
theorem female_students_in_sample 
  (num_male : ℕ) (num_female : ℕ) (sample_size : ℕ)
  (h_male : num_male = 500)
  (h_female : num_female = 400)
  (h_sample : sample_size = 45)
  (total_students : ℕ := num_male + num_female)
  (sample_ratio : ℚ := sample_size / total_students) :
  num_female * sample_ratio = 20 := 
sorry

end female_students_in_sample_l28_28157


namespace remainder_7_pow_2010_l28_28360

theorem remainder_7_pow_2010 :
  (7 ^ 2010) % 100 = 49 := 
by 
  sorry

end remainder_7_pow_2010_l28_28360


namespace domain_of_reciprocal_shifted_function_l28_28385

def domain_of_function (x : ℝ) : Prop :=
  x ≠ 1

theorem domain_of_reciprocal_shifted_function : 
  ∀ x : ℝ, (∃ y : ℝ, y = 1 / (x - 1)) ↔ domain_of_function x :=
by 
  sorry

end domain_of_reciprocal_shifted_function_l28_28385


namespace molecular_weight_one_mole_of_AlOH3_l28_28458

variable (MW_7_moles : ℕ) (MW : ℕ)

theorem molecular_weight_one_mole_of_AlOH3 (h : MW_7_moles = 546) : MW = 78 :=
by
  sorry

end molecular_weight_one_mole_of_AlOH3_l28_28458


namespace determine_b_l28_28367

theorem determine_b (b : ℝ) : (∀ x1 x2 : ℝ, x1^2 - x2^2 = 7 → x1 * x2 = 12 → x1 + x2 = b) → (b = 7 ∨ b = -7) := 
by {
  -- Proof needs to be provided
  sorry
}

end determine_b_l28_28367


namespace tap_B_fill_time_l28_28781

theorem tap_B_fill_time :
  ∃ t : ℝ, 
    (3 * 10 + (12 / t) * 10 = 36) →
    t = 20 :=
by
  sorry

end tap_B_fill_time_l28_28781


namespace solver_inequality_l28_28084

theorem solver_inequality (x : ℝ) :
  (2 * x - 1 ≥ x + 2) ∧ (x + 5 < 4 * x - 1) → (x ≥ 3) :=
by
  intro h
  sorry

end solver_inequality_l28_28084


namespace find_tuition_l28_28397

def tuition_problem (T : ℝ) : Prop :=
  75 = T + (T - 15)

theorem find_tuition (T : ℝ) (h : tuition_problem T) : T = 45 :=
by
  sorry

end find_tuition_l28_28397


namespace range_of_a_l28_28197

def p (x : ℝ) : Prop := 1 / 2 ≤ x ∧ x ≤ 1

def q (x a : ℝ) : Prop := (x - a) * (x - a - 1) ≤ 0

theorem range_of_a (a x : ℝ) 
  (hp : ∀ x, ¬ (1 / 2 ≤ x ∧ x ≤ 1) → (x < 1 / 2 ∨ x > 1))
  (hq : ∀ x, ¬ ((x - a) * (x - a - 1) ≤ 0) → (x < a ∨ x > a + 1))
  (h : ∀ x, (q x a) → (p x)) :
  0 ≤ a ∧ a ≤ 1 / 2 := 
sorry

end range_of_a_l28_28197


namespace mushroom_distribution_l28_28627

-- Define the total number of mushrooms
def total_mushrooms : ℕ := 120

-- Define the number of girls
def number_of_girls : ℕ := 5

-- Auxiliary function to represent each girl receiving pattern
def mushrooms_received (n :ℕ) (total : ℕ) : ℝ :=
  (n + 20) + 0.04 * (total - (n + 20))

-- Define the equality function to check distribution condition
def equal_distribution (girls : ℕ) (total : ℕ) : Prop :=
  ∀ i j : ℕ, i < girls → j < girls → mushrooms_received i total = mushrooms_received j total

-- Main proof statement about the total mushrooms and number of girls following the distribution
theorem mushroom_distribution :
  total_mushrooms = 120 ∧ number_of_girls = 5 ∧ equal_distribution number_of_girls total_mushrooms := 
by 
  sorry

end mushroom_distribution_l28_28627


namespace vertical_asymptotes_sum_l28_28939

theorem vertical_asymptotes_sum : 
  let f (x : ℝ) := (6 * x^2 + 1) / (4 * x^2 + 6 * x + 3)
  let den := 4 * x^2 + 6 * x + 3
  let p := -(3 / 2)
  let q := -(1 / 2)
  (den = 0) → (p + q = -2) :=
by
  sorry

end vertical_asymptotes_sum_l28_28939


namespace tangent_line_is_x_minus_y_eq_zero_l28_28998

theorem tangent_line_is_x_minus_y_eq_zero : 
  ∀ (f : ℝ → ℝ) (x y : ℝ), 
  f x = x^3 - 2 * x → 
  (x, y) = (1, 1) → 
  (∃ (m : ℝ), m = 3 * (1:ℝ)^2 - 2 ∧ (y - 1) = m * (x - 1)) → 
  x - y = 0 :=
by
  intros f x y h_func h_point h_tangent
  sorry

end tangent_line_is_x_minus_y_eq_zero_l28_28998


namespace max_profit_30000_l28_28610

noncomputable def max_profit (type_A : ℕ) (type_B : ℕ) : ℝ := 
  10000 * type_A + 5000 * type_B

theorem max_profit_30000 :
  ∃ (type_A type_B : ℕ), 
  (4 * type_A + 1 * type_B ≤ 10) ∧
  (18 * type_A + 15 * type_B ≤ 66) ∧
  max_profit type_A type_B = 30000 :=
sorry

end max_profit_30000_l28_28610


namespace stock_worth_l28_28585

theorem stock_worth (X : ℝ)
  (H1 : 0.2 * X * 0.1 = 0.02 * X)  -- 20% of stock at 10% profit given in condition.
  (H2 : 0.8 * X * 0.05 = 0.04 * X) -- Remaining 80% of stock at 5% loss given in condition.
  (H3 : 0.04 * X - 0.02 * X = 400) -- Overall loss incurred is Rs. 400.
  : X = 20000 := 
sorry

end stock_worth_l28_28585


namespace solution_l28_28073

namespace ProofProblem

variables (a b : ℝ)

def five_times_a_minus_b_eq_60 := 5 * a - b = 60
def six_times_a_plus_b_lt_90 := 6 * a + b < 90

theorem solution (h1 : five_times_a_minus_b_eq_60 a b) (h2 : six_times_a_plus_b_lt_90 a b) :
  a < 150 / 11 ∧ b < 8.18 :=
sorry

end ProofProblem

end solution_l28_28073


namespace combined_salaries_ABC_E_l28_28615

-- Definitions for the conditions
def salary_D : ℝ := 7000
def avg_salary_ABCDE : ℝ := 8200

-- Defining the combined salary proof
theorem combined_salaries_ABC_E : (A B C E : ℝ) → 
  (A + B + C + D + E = 5 * avg_salary_ABCDE ∧ D = salary_D) → 
  (A + B + C + E = 34000) := 
sorry

end combined_salaries_ABC_E_l28_28615


namespace find_smallest_value_of_sum_of_squares_l28_28106
noncomputable def smallest_value (x y z : ℚ) := x^2 + y^2 + z^2

theorem find_smallest_value_of_sum_of_squares :
  ∃ (x y z : ℚ), (x + 4) * (y - 4) = 0 ∧ 3 * z - 2 * y = 5 ∧ smallest_value x y z = 457 / 9 :=
by
  sorry

end find_smallest_value_of_sum_of_squares_l28_28106


namespace sum_of_dimensions_l28_28541

theorem sum_of_dimensions
  (X Y Z : ℝ)
  (h1 : X * Y = 24)
  (h2 : X * Z = 48)
  (h3 : Y * Z = 72) :
  X + Y + Z = 22 := 
sorry

end sum_of_dimensions_l28_28541


namespace symmetric_about_line_periodic_function_l28_28988

section
variable {α : Type*} [LinearOrderedField α]

-- First proof problem
theorem symmetric_about_line (f : α → α) (a : α) (h : ∀ x, f (a + x) = f (a - x)) : 
  ∀ x, f (2 * a - x) = f x :=
sorry

-- Second proof problem
theorem periodic_function (f : α → α) (a b : α) (ha : a ≠ b)
  (hsymm_a : ∀ x, f (2 * a - x) = f x)
  (hsymm_b : ∀ x, f (2 * b - x) = f x) : 
  ∃ p, p ≠ 0 ∧ ∀ x, f (x + p) = f x :=
sorry
end

end symmetric_about_line_periodic_function_l28_28988


namespace cupcakes_per_package_l28_28330

theorem cupcakes_per_package
  (packages : ℕ) (total_left : ℕ) (cupcakes_eaten : ℕ) (initial_packages : ℕ) (cupcakes_per_package : ℕ)
  (h1 : initial_packages = 3)
  (h2 : cupcakes_eaten = 5)
  (h3 : total_left = 7)
  (h4 : packages = initial_packages * cupcakes_per_package - cupcakes_eaten)
  (h5 : packages = total_left) : 
  cupcakes_per_package = 4 := 
by
  sorry

end cupcakes_per_package_l28_28330


namespace phil_books_remaining_pages_l28_28572

/-- We define the initial number of books and the number of pages per book. -/
def initial_books : Nat := 10
def pages_per_book : Nat := 100
def lost_books : Nat := 2

/-- The goal is to find the total number of pages Phil has left after losing 2 books. -/
theorem phil_books_remaining_pages : (initial_books - lost_books) * pages_per_book = 800 := by 
  -- The proof will go here
  sorry

end phil_books_remaining_pages_l28_28572


namespace largest_number_l28_28651

noncomputable def a : ℝ := 8.12331
noncomputable def b : ℝ := 8.123 + 3 / 10000 * ∑' n, 1 / (10 : ℝ)^n
noncomputable def c : ℝ := 8.12 + 331 / 100000 * ∑' n, 1 / (1000 : ℝ)^n
noncomputable def d : ℝ := 8.1 + 2331 / 1000000 * ∑' n, 1 / (10000 : ℝ)^n
noncomputable def e : ℝ := 8 + 12331 / 100000 * ∑' n, 1 / (10000 : ℝ)^n

theorem largest_number : (b > a) ∧ (b > c) ∧ (b > d) ∧ (b > e) := by
  sorry

end largest_number_l28_28651


namespace stones_in_10th_pattern_l28_28235

def stones_in_nth_pattern (n : ℕ) : ℕ :=
n * (3 * n - 1) / 2 + 1

theorem stones_in_10th_pattern : stones_in_nth_pattern 10 = 145 :=
by
  sorry

end stones_in_10th_pattern_l28_28235


namespace total_apples_count_l28_28287

-- Definitions based on conditions
def red_apples := 16
def green_apples := red_apples + 12
def total_apples := green_apples + red_apples

-- Statement to prove
theorem total_apples_count : total_apples = 44 := 
by
  sorry

end total_apples_count_l28_28287


namespace points_lie_on_line_l28_28788

theorem points_lie_on_line (t : ℝ) (ht : t ≠ 0) :
  let x := (2 * t + 2) / t
  let y := (2 * t - 2) / t
  x + y = 4 :=
by
  let x := (2 * t + 2) / t
  let y := (2 * t - 2) / t
  sorry

end points_lie_on_line_l28_28788


namespace min_value_of_sum_of_squares_l28_28236

theorem min_value_of_sum_of_squares (a b c : ℕ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) 
  (h : a^2 + b^2 - c = 2022) : 
  a^2 + b^2 + c^2 = 2034 ∧ a = 27 ∧ b = 36 ∧ c = 3 := 
sorry

end min_value_of_sum_of_squares_l28_28236


namespace number_exceeds_by_35_l28_28554

theorem number_exceeds_by_35 (x : ℤ) (h : x = (3 / 8 : ℚ) * x + 35) : x = 56 :=
by
  sorry

end number_exceeds_by_35_l28_28554


namespace largest_three_digit_in_pascal_triangle_l28_28822

-- Define Pascal's triangle and binomial coefficient
def pascal (n k : ℕ) : ℕ := Nat.choose n k

-- State the theorem about the first appearance of the number 999 in Pascal's triangle
theorem largest_three_digit_in_pascal_triangle :
  ∃ (n : ℕ), n = 1000 ∧ ∃ (k : ℕ), pascal n k = 999 :=
sorry

end largest_three_digit_in_pascal_triangle_l28_28822


namespace johns_overall_average_speed_l28_28328

open Real

noncomputable def johns_average_speed (scooter_time_min : ℝ) (scooter_speed_mph : ℝ) 
    (jogging_time_min : ℝ) (jogging_speed_mph : ℝ) : ℝ :=
  let scooter_time_hr := scooter_time_min / 60
  let jogging_time_hr := jogging_time_min / 60
  let distance_scooter := scooter_speed_mph * scooter_time_hr
  let distance_jogging := jogging_speed_mph * jogging_time_hr
  let total_distance := distance_scooter + distance_jogging
  let total_time := scooter_time_hr + jogging_time_hr
  total_distance / total_time

theorem johns_overall_average_speed :
  johns_average_speed 40 20 60 6 = 11.6 :=
by
  sorry

end johns_overall_average_speed_l28_28328


namespace complex_sum_magnitude_eq_three_l28_28598

open Complex

theorem complex_sum_magnitude_eq_three (a b c : ℂ) 
    (h1 : abs a = 1) 
    (h2 : abs b = 1) 
    (h3 : abs c = 1) 
    (h4 : a^3 / (b * c) + b^3 / (a * c) + c^3 / (a * b) = -3) : 
    abs (a + b + c) = 3 := 
sorry

end complex_sum_magnitude_eq_three_l28_28598


namespace square_side_length_exists_l28_28602

-- Define the dimensions of the tile
structure Tile where
  width : Nat
  length : Nat

-- Define the specific tile used in the problem
def given_tile : Tile :=
  { width := 16, length := 24 }

-- Define the condition of forming a square using 6 tiles
def forms_square_with_6_tiles (tile : Tile) (side_length : Nat) : Prop :=
  (2 * tile.length = side_length) ∧ (3 * tile.width = side_length)

-- Problem statement requiring proof
theorem square_side_length_exists : forms_square_with_6_tiles given_tile 48 :=
  sorry

end square_side_length_exists_l28_28602


namespace find_k_l28_28324

theorem find_k (k : ℕ) : (1 / 2) ^ 18 * (1 / 81) ^ k = 1 / 18 ^ 18 → k = 0 := by
  intro h
  sorry

end find_k_l28_28324


namespace power_expression_result_l28_28462

theorem power_expression_result : (-2)^2004 + (-2)^2005 = -2^2004 :=
by
  sorry

end power_expression_result_l28_28462


namespace total_ribbon_length_l28_28747

theorem total_ribbon_length (a b c d e f g h i : ℝ) 
  (H : a + b + c + d + e + f + g + h + i = 62) : 
  1.5 * (a + b + c + d + e + f + g + h + i) = 93 :=
by
  sorry

end total_ribbon_length_l28_28747


namespace twenty_fifty_yuan_bills_unique_l28_28851

noncomputable def twenty_fifty_yuan_bills (x y : ℕ) : Prop :=
  x + y = 260 ∧ 20 * x + 50 * y = 100 * 100

theorem twenty_fifty_yuan_bills_unique (x y : ℕ) (h : twenty_fifty_yuan_bills x y) :
  x = 100 ∧ y = 160 :=
by
  sorry

end twenty_fifty_yuan_bills_unique_l28_28851


namespace prove_min_period_and_max_value_l28_28266

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 - (Real.cos x)^2

theorem prove_min_period_and_max_value :
  (∀ x : ℝ, f (x + π) = f x) ∧ (∀ y : ℝ, y ≤ f y) :=
by
  -- Proof goes here
  sorry

end prove_min_period_and_max_value_l28_28266


namespace range_of_m_inequality_a_b_l28_28043

def f (x : ℝ) : ℝ := |2 * x + 1| + |2 * x - 2|

theorem range_of_m (m : ℝ) : (∀ x, f x ≥ |m - 1|) → -2 ≤ m ∧ m ≤ 4 :=
sorry

theorem inequality_a_b (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a^2 + b^2 = 2) : 
  a + b ≥ 2 * a * b :=
sorry

end range_of_m_inequality_a_b_l28_28043


namespace initial_children_count_l28_28640

theorem initial_children_count (passed retake : ℝ) (h_passed : passed = 105.0) (h_retake : retake = 593) : 
    passed + retake = 698 := 
by
  sorry

end initial_children_count_l28_28640


namespace henry_kombucha_bottles_l28_28619

theorem henry_kombucha_bottles :
  ∀ (monthly_bottles: ℕ) (cost_per_bottle refund_rate: ℝ) (months_in_year total_bottles_in_year: ℕ),
  (monthly_bottles = 15) →
  (cost_per_bottle = 3.0) →
  (refund_rate = 0.10) →
  (months_in_year = 12) →
  (total_bottles_in_year = monthly_bottles * months_in_year) →
  (total_refund = refund_rate * total_bottles_in_year) →
  (bottles_bought_with_refund = total_refund / cost_per_bottle) →
  bottles_bought_with_refund = 6 :=
by
  intros monthly_bottles cost_per_bottle refund_rate months_in_year total_bottles_in_year
  sorry

end henry_kombucha_bottles_l28_28619


namespace fraction_students_above_eight_l28_28432

theorem fraction_students_above_eight (total_students S₈ : ℕ) (below_eight_percent : ℝ)
    (num_below_eight : total_students * below_eight_percent = 10) 
    (total_equals : total_students = 50) 
    (students_eight : S₈ = 24) :
    (total_students - (total_students * below_eight_percent + S₈)) / S₈ = 2 / 3 := 
by 
  -- Solution steps can go here 
  sorry

end fraction_students_above_eight_l28_28432


namespace consecutive_integer_product_sum_l28_28540

theorem consecutive_integer_product_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 :=
sorry

end consecutive_integer_product_sum_l28_28540


namespace problem1_problem2_l28_28026

-- Problem (1)
theorem problem1 (a : ℕ → ℤ) (h1 : a 1 = 4) (h2 : ∀ n, a n = a (n + 1) + 3) : a 10 = -23 :=
by {
  sorry
}

-- Problem (2)
theorem problem2 (a : ℕ → ℚ) (h1 : a 6 = (1 / 4)) (h2 : ∃ d : ℚ, ∀ n, 1 / a n = 1 / a 1 + (n - 1) * d) : 
  ∀ n, a n = (4 / (3 * n - 2)) :=
by {
  sorry
}

end problem1_problem2_l28_28026


namespace temperature_decrease_l28_28341

theorem temperature_decrease (initial : ℤ) (decrease : ℤ) : initial = -3 → decrease = 6 → initial - decrease = -9 :=
by
  intros
  sorry

end temperature_decrease_l28_28341


namespace train_length_equals_750_l28_28971

theorem train_length_equals_750
  (L : ℕ) -- length of the train in meters
  (v : ℕ) -- speed of the train in m/s
  (t : ℕ) -- time in seconds
  (h1 : v = 25) -- speed is 25 m/s
  (h2 : t = 60) -- time is 60 seconds
  (h3 : 2 * L = v * t) -- total distance covered by the train is 2L (train and platform) and equals speed * time
  : L = 750 := 
sorry

end train_length_equals_750_l28_28971


namespace maria_travel_fraction_l28_28412

theorem maria_travel_fraction (x : ℝ) (total_distance : ℝ)
  (h1 : ∀ d1 d2, d1 + d2 = total_distance)
  (h2 : total_distance = 360)
  (h3 : ∃ d1 d2 d3, d1 = 360 * x ∧ d2 = (1 / 4) * (360 - 360 * x) ∧ d3 = 135)
  (h4 : d1 + d2 + d3 = total_distance)
  : x = 1 / 2 :=
by
  sorry

end maria_travel_fraction_l28_28412


namespace gigi_remaining_batches_l28_28372

variable (f b1 tf remaining_batches : ℕ)
variable (f_pos : 0 < f)
variable (batches_nonneg : 0 ≤ b1)
variable (t_f_pos : 0 < tf)
variable (h_f : f = 2)
variable (h_b1 : b1 = 3)
variable (h_tf : tf = 20)

theorem gigi_remaining_batches (h : remaining_batches = (tf - (f * b1)) / f) : remaining_batches = 7 := by
  sorry

end gigi_remaining_batches_l28_28372


namespace ordered_pairs_1944_l28_28088

theorem ordered_pairs_1944 :
  ∃ n : ℕ, (∀ x y : ℕ, (x * y = 1944 ↔ x > 0 ∧ y > 0)) → n = 24 :=
by
  sorry

end ordered_pairs_1944_l28_28088


namespace lines_intersect_at_3_6_l28_28708

theorem lines_intersect_at_3_6 (c d : ℝ) 
  (h1 : 3 = 2 * 6 + c) 
  (h2 : 6 = 2 * 3 + d) : 
  c + d = -9 := by 
  sorry

end lines_intersect_at_3_6_l28_28708


namespace dimes_max_diff_l28_28892

-- Definitions and conditions
def num_coins (a b c : ℕ) : Prop := a + b + c = 120
def coin_values (a b c : ℕ) : Prop := 5 * a + 10 * b + 50 * c = 1050
def dimes_difference (a1 a2 b1 b2 c1 c2 : ℕ) : Prop := num_coins a1 b1 c1 ∧ num_coins a2 b2 c2 ∧ coin_values a1 b1 c1 ∧ coin_values a2 b2 c2 ∧ a1 = a2 ∧ c1 = c2

-- Theorem statement
theorem dimes_max_diff : ∃ (a b1 b2 c : ℕ), dimes_difference a a b1 b2 c c ∧ b1 - b2 = 90 :=
by sorry

end dimes_max_diff_l28_28892


namespace smallest_w_l28_28044

def fact_936 : ℕ := 2^3 * 3^1 * 13^1

theorem smallest_w (w : ℕ) (h_w_pos : 0 < w) :
  (2^5 ∣ 936 * w) ∧ (3^3 ∣ 936 * w) ∧ (12^2 ∣ 936 * w) → w = 36 :=
by
  sorry

end smallest_w_l28_28044


namespace hair_growth_l28_28379

-- Define the length of Isabella's hair initially and the growth
def initial_length : ℕ := 18
def growth : ℕ := 4

-- Define the final length of the hair after growth
def final_length (initial_length : ℕ) (growth : ℕ) : ℕ := initial_length + growth

-- State the theorem that the final length is 22 inches
theorem hair_growth : final_length initial_length growth = 22 := 
by
  sorry

end hair_growth_l28_28379


namespace sin_300_eq_neg_sqrt3_div_2_l28_28933

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l28_28933


namespace xiaolin_distance_l28_28862

theorem xiaolin_distance (speed : ℕ) (time : ℕ) (distance : ℕ)
    (h1 : speed = 80) (h2 : time = 28) : distance = 2240 :=
by
  have h3 : distance = time * speed := by sorry
  rw [h1, h2] at h3
  exact h3

end xiaolin_distance_l28_28862


namespace segment_length_R_R_l28_28002

theorem segment_length_R_R' :
  let R := (-4, 1)
  let R' := (-4, -1)
  let distance : ℝ := Real.sqrt ((R'.1 - R.1)^2 + (R'.2 - R.2)^2)
  distance = 2 :=
by
  sorry

end segment_length_R_R_l28_28002


namespace train_speed_in_kmph_l28_28999

variable (L V : ℝ) -- L is the length of the train in meters, and V is the speed of the train in m/s.

-- Conditions given in the problem
def crosses_platform_in_30_seconds : Prop := L + 200 = V * 30
def crosses_man_in_20_seconds : Prop := L = V * 20

-- Length of the platform
def platform_length : ℝ := 200

-- The proof problem: Prove the speed of the train is 72 km/h
theorem train_speed_in_kmph 
  (h1 : crosses_man_in_20_seconds L V) 
  (h2 : crosses_platform_in_30_seconds L V) : 
  V * 3.6 = 72 := 
by 
  sorry

end train_speed_in_kmph_l28_28999


namespace ratio_lena_kevin_after_5_more_l28_28525

variables (L K N : ℕ)

def lena_initial_candy : ℕ := 16
def lena_gets_more : ℕ := 5
def kevin_candy_less_than_nicole : ℕ := 4
def lena_more_than_nicole : ℕ := 5

theorem ratio_lena_kevin_after_5_more
  (lena_initial : L = lena_initial_candy)
  (lena_to_multiple_of_kevin : L + lena_gets_more = K * 3) 
  (kevin_less_than_nicole : K = N - kevin_candy_less_than_nicole)
  (lena_more_than_nicole_condition : L = N + lena_more_than_nicole) :
  (L + lena_gets_more) / K = 3 :=
sorry

end ratio_lena_kevin_after_5_more_l28_28525


namespace define_interval_l28_28104

theorem define_interval (x : ℝ) : 
  (0 < x + 2) → (0 < 5 - x) → (-2 < x ∧ x < 5) :=
by
  intros h1 h2
  sorry

end define_interval_l28_28104


namespace gcd_lcm_product_l28_28492

theorem gcd_lcm_product (a b : ℕ) (h₁ : a = 24) (h₂ : b = 36) :
  Nat.gcd a b * Nat.lcm a b = 864 :=
by
  rw [h₁, h₂]
  unfold Nat.gcd Nat.lcm
  sorry

end gcd_lcm_product_l28_28492


namespace sally_cards_l28_28831

theorem sally_cards (x : ℕ) (h1 : 27 + x + 20 = 88) : x = 41 := by
  sorry

end sally_cards_l28_28831


namespace binomial_prime_div_l28_28012

theorem binomial_prime_div {p : ℕ} {m : ℕ} (hp : Nat.Prime p) (hm : 0 < m) : (Nat.choose (p ^ m) p - p ^ (m - 1)) % p ^ m = 0 := 
  sorry

end binomial_prime_div_l28_28012


namespace calculation_error_l28_28555

def percentage_error (actual expected : ℚ) : ℚ :=
  (actual - expected) / expected * 100

theorem calculation_error :
  let correct_result := (5 / 3) * 3
  let incorrect_result := (5 / 3) / 3
  percentage_error incorrect_result correct_result = 88.89 := by
  sorry

end calculation_error_l28_28555


namespace ali_babas_cave_min_moves_l28_28738

theorem ali_babas_cave_min_moves : 
  ∀ (counters : Fin 28 → Fin 2018) (decrease_by : ℕ → Fin 28 → ℕ),
    (∀ n, n < 28 → decrease_by n ≤ 2017) → 
    (∃ (k : ℕ), k ≤ 11 ∧ 
      ∀ n, (n < 28 → decrease_by (k - n) n = 0)) :=
sorry

end ali_babas_cave_min_moves_l28_28738


namespace sum_of_all_possible_k_values_l28_28586

theorem sum_of_all_possible_k_values (j k : ℕ) (h : 1 / j + 1 / k = 1 / 4) : 
  (∃ j k : ℕ, (j > 0 ∧ k > 0) ∧ (1 / j + 1 / k = 1 / 4) ∧ (k = 8 ∨ k = 12 ∨ k = 20)) →
  (8 + 12 + 20 = 40) :=
by
  sorry

end sum_of_all_possible_k_values_l28_28586


namespace length_of_GH_l28_28805

theorem length_of_GH (AB FE CD : ℕ) (side_large side_second side_third side_small : ℕ) 
  (h1 : AB = 11) (h2 : FE = 13) (h3 : CD = 5)
  (h4 : side_large = side_second + AB)
  (h5 : side_second = side_third + CD)
  (h6 : side_third = side_small + FE) :
  GH = 29 :=
by
  -- Proof steps would follow here based on the problem's solution
  -- Using the given conditions and transformations.
  sorry

end length_of_GH_l28_28805


namespace simplify_trig_identity_l28_28478

theorem simplify_trig_identity (α : ℝ) :
  (Real.cos (Real.pi / 3 + α) + Real.sin (Real.pi / 6 + α)) = Real.cos α :=
by
  sorry

end simplify_trig_identity_l28_28478


namespace find_x_range_l28_28149

variable (f : ℝ → ℝ)

def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)

def decreasing_on_nonnegative (f : ℝ → ℝ) :=
  ∀ x1 x2 : ℝ, x1 ≥ 0 → x2 ≥ 0 → x1 ≠ x2 → (f x2 - f x1) / (x2 - x1) < 0

theorem find_x_range (f : ℝ → ℝ)
  (h1 : even_function f)
  (h2 : decreasing_on_nonnegative f)
  (h3 : f (1/3) = 3/4)
  (h4 : ∀ x : ℝ, 4 * f (Real.logb (1/8) x) > 3) :
  ∀ x : ℝ, (1/2 < x ∧ x < 2) ↔ True := sorry

end find_x_range_l28_28149


namespace solve_system_eqs_l28_28551

theorem solve_system_eqs : 
    ∃ (x y z : ℚ), 
    4 * x - 3 * y + z = -10 ∧
    3 * x + 5 * y - 2 * z = 8 ∧
    x - 2 * y + 7 * z = 5 ∧
    x = -51 / 61 ∧ 
    y = 378 / 61 ∧ 
    z = 728 / 61 := by
  sorry

end solve_system_eqs_l28_28551


namespace find_angle_l28_28885

theorem find_angle (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by 
  sorry

end find_angle_l28_28885


namespace smallest_value_N_l28_28957

theorem smallest_value_N (l m n N : ℕ) (h1 : (l - 1) * (m - 1) * (n - 1) = 143) (h2 : N = l * m * n) :
  N = 336 :=
sorry

end smallest_value_N_l28_28957


namespace marys_total_cards_l28_28223

def initial_cards : ℕ := 18
def torn_cards : ℕ := 8
def cards_from_fred : ℕ := 26
def cards_bought_by_mary : ℕ := 40

theorem marys_total_cards :
  initial_cards - torn_cards + cards_from_fred + cards_bought_by_mary = 76 :=
by
  sorry

end marys_total_cards_l28_28223


namespace anticipated_margin_l28_28322

noncomputable def anticipated_profit_margin (original_purchase_price : ℝ) (decrease_percentage : ℝ) (profit_margin_increase : ℝ) (selling_price : ℝ) : ℝ :=
original_purchase_price * (1 + profit_margin_increase / 100)

theorem anticipated_margin (x : ℝ) (original_purchase_price_decrease : ℝ := 0.064) (profit_margin_increase : ℝ := 8) (selling_price : ℝ) :
  selling_price = original_purchase_price * (1 + x / 100) ∧ selling_price = (1 - original_purchase_price_decrease) * (1 + (x + profit_margin_increase) / 100) →
  true :=
by
  sorry

end anticipated_margin_l28_28322


namespace quadratics_root_k_value_l28_28428

theorem quadratics_root_k_value :
  (∀ k : ℝ, (∀ x : ℝ, x^2 + k * x + 6 = 0 → (x = 2 ∨ ∃ x1 : ℝ, x1 * 2 = 6 ∧ x1 + 2 = k)) → 
  (x = 2 → ∃ x1 : ℝ, x1 = 3 ∧ k = -5)) := 
sorry

end quadratics_root_k_value_l28_28428


namespace smallest_number_of_students_l28_28722

-- Define the structure of the problem
def unique_row_configurations (n : ℕ) : Prop :=
  (∀ k : ℕ, k ∣ n → k < 10) → ∃ divs : Finset ℕ, divs.card = 9 ∧ ∀ d ∈ divs, d ∣ n ∧ (∀ d' ∈ divs, d ≠ d') 

-- The main statement to be proven in Lean 4
theorem smallest_number_of_students : ∃ n : ℕ, unique_row_configurations n ∧ n = 36 :=
by
  sorry

end smallest_number_of_students_l28_28722


namespace minimum_value_m_l28_28089

noncomputable def f (x : ℝ) (phi : ℝ) : ℝ :=
  Real.sin (2 * x + phi)

theorem minimum_value_m (phi : ℝ) (m : ℝ) (h1 : |phi| < Real.pi / 2)
  (h2 : ∃ x ∈ Set.Icc 0 (Real.pi / 2), f x (Real.pi / 6) ≤ m) :
  m = -1 / 2 :=
by
  sorry

end minimum_value_m_l28_28089


namespace sales_tax_per_tire_l28_28431

def cost_per_tire : ℝ := 7
def number_of_tires : ℕ := 4
def final_total_cost : ℝ := 30

theorem sales_tax_per_tire :
  (final_total_cost - number_of_tires * cost_per_tire) / number_of_tires = 0.5 :=
sorry

end sales_tax_per_tire_l28_28431


namespace program_output_for_six_l28_28439

-- Define the factorial function
def factorial : ℕ → ℕ
  | 0       => 1
  | (n + 1) => (n + 1) * factorial n

-- The theorem we want to prove
theorem program_output_for_six : factorial 6 = 720 := by
  sorry

end program_output_for_six_l28_28439


namespace smallest_portion_is_2_l28_28116

theorem smallest_portion_is_2 (a d : ℝ) (h1 : 5 * a = 120) (h2 : 3 * a + 3 * d = 7 * (2 * a - 3 * d)) : a - 2 * d = 2 :=
by sorry

end smallest_portion_is_2_l28_28116


namespace max_n_value_l28_28335

noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x)

theorem max_n_value (m : ℝ) (x_i : ℕ → ℝ) (n : ℕ) (h1 : ∀ i, i < n → f (x_i i) / (x_i i) = m)
  (h2 : ∀ i, i < n → -2 * Real.pi ≤ x_i i ∧ x_i i ≤ 2 * Real.pi) :
  n ≤ 12 :=
sorry

end max_n_value_l28_28335


namespace find_percentage_l28_28121

variable (x p : ℝ)
variable (h1 : 0.25 * x = (p / 100) * 1500 - 20)
variable (h2 : x = 820)

theorem find_percentage : p = 15 :=
by
  sorry

end find_percentage_l28_28121


namespace number_of_hens_l28_28251

theorem number_of_hens
    (H C : ℕ) -- Hens and Cows
    (h1 : H + C = 44) -- Condition 1: The number of heads
    (h2 : 2 * H + 4 * C = 128) -- Condition 2: The number of feet
    : H = 24 :=
by
  sorry

end number_of_hens_l28_28251


namespace megan_folders_l28_28772

theorem megan_folders (initial_files deleted_files files_per_folder : ℕ) (h1 : initial_files = 237)
    (h2 : deleted_files = 53) (h3 : files_per_folder = 12) :
    let remaining_files := initial_files - deleted_files
    let total_folders := (remaining_files / files_per_folder) + 1
    total_folders = 16 := 
by
  sorry

end megan_folders_l28_28772


namespace range_of_a_l28_28107

theorem range_of_a (a : ℝ) : 
  (∃! x : ℤ, 4 - 2 * x ≥ 0 ∧ (1 / 2 : ℝ) * x - a > 0) ↔ -1 ≤ a ∧ a < -0.5 :=
by
  sorry

end range_of_a_l28_28107


namespace general_term_a_general_term_b_l28_28911

def arithmetic_sequence (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) :=
∀ n, a_n n = n ∧ S_n n = (n^2 + n) / 2

def sequence_b (b_n : ℕ → ℝ) (T_n : ℕ → ℝ) :=
  (b_n 1 = 1/2) ∧
  (∀ n, b_n (n+1) = (n+1) / n * b_n n) ∧ 
  (∀ n, b_n n = n / 2) ∧ 
  (∀ n, T_n n = (n^2 + n) / 4) ∧ 
  (∀ m, m = 1 → T_n m = 1/2)

-- Arithmetic sequence {a_n}
theorem general_term_a (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : a 2 = 2) (h2 : S 5 = 15) :
  arithmetic_sequence a S := sorry

-- Sequence {b_n}
theorem general_term_b (b : ℕ → ℝ) (T : ℕ → ℝ) (h1 : b 1 = 1/2) (h2 : ∀ n, b (n+1) = (n+1) / n * b n) :
  sequence_b b T := sorry

end general_term_a_general_term_b_l28_28911


namespace Sam_age_proof_l28_28237

-- Define the conditions (Phoebe's current age, Raven's age relation, Sam's age definition)
def Phoebe_current_age : ℕ := 10
def Raven_in_5_years (R : ℕ) : Prop := R + 5 = 4 * (Phoebe_current_age + 5)
def Sam_age (R : ℕ) : ℕ := 2 * ((R + 3) - (Phoebe_current_age + 3))

-- The proof statement for Sam's current age
theorem Sam_age_proof (R : ℕ) (h : Raven_in_5_years R) : Sam_age R = 90 := by
  sorry

end Sam_age_proof_l28_28237


namespace ratio_of_surface_areas_l28_28147

theorem ratio_of_surface_areas (s : ℝ) :
  let cube_surface_area := 6 * s^2
  let tetrahedron_edge := s * Real.sqrt 2
  let tetrahedron_face_area := (Real.sqrt 3 / 4) * (tetrahedron_edge)^2
  let tetrahedron_surface_area := 4 * tetrahedron_face_area
  (cube_surface_area / tetrahedron_surface_area) = Real.sqrt 3 :=
by
  let cube_surface_area := 6 * s^2
  let tetrahedron_edge := s * Real.sqrt 2
  let tetrahedron_face_area := (Real.sqrt 3 / 4) * (tetrahedron_edge)^2
  let tetrahedron_surface_area := 4 * tetrahedron_face_area
  show (cube_surface_area / tetrahedron_surface_area) = Real.sqrt 3
  sorry

end ratio_of_surface_areas_l28_28147


namespace b_is_some_even_number_l28_28941

noncomputable def factorable_b (b : ℤ) : Prop :=
  ∃ (m n p q : ℤ), 
    (m * p = 15 ∧ n * q = 15) ∧ 
    (b = m * q + n * p)

theorem b_is_some_even_number (b : ℤ) 
  (h : factorable_b b) : ∃ k : ℤ, b = 2 * k := 
by
  sorry

end b_is_some_even_number_l28_28941


namespace xiaoming_comprehensive_score_l28_28704

theorem xiaoming_comprehensive_score :
  ∀ (a b c d : ℝ),
  a = 92 → b = 90 → c = 88 → d = 95 →
  (0.4 * a + 0.3 * b + 0.2 * c + 0.1 * d) = 90.9 :=
by
  intros a b c d ha hb hc hd
  simp [ha, hb, hc, hd]
  norm_num
  done

end xiaoming_comprehensive_score_l28_28704


namespace boatworks_total_canoes_l28_28614

theorem boatworks_total_canoes : 
  let jan := 5
  let feb := 3 * jan
  let mar := 3 * feb
  let apr := 3 * mar
  jan + feb + mar + apr = 200 := 
by 
  sorry

end boatworks_total_canoes_l28_28614


namespace min_days_to_sun_l28_28545

def active_days_for_level (N : ℕ) : ℕ :=
  N * (N + 4)

def days_needed_for_upgrade (current_days future_days : ℕ) : ℕ :=
  future_days - current_days

theorem min_days_to_sun (current_level future_level : ℕ) :
  current_level = 9 →
  future_level = 16 →
  days_needed_for_upgrade (active_days_for_level current_level) (active_days_for_level future_level) = 203 :=
by
  intros h1 h2
  rw [h1, h2, active_days_for_level, active_days_for_level]
  sorry

end min_days_to_sun_l28_28545


namespace total_crayons_l28_28309
-- Import the whole Mathlib to ensure all necessary components are available

-- Definitions of the number of crayons each person has
def Billy_crayons : ℕ := 62
def Jane_crayons : ℕ := 52
def Mike_crayons : ℕ := 78
def Sue_crayons : ℕ := 97

-- Theorem stating the total number of crayons is 289
theorem total_crayons : (Billy_crayons + Jane_crayons + Mike_crayons + Sue_crayons) = 289 := by
  sorry

end total_crayons_l28_28309


namespace P_at_10_l28_28700

-- Define the main properties of the polynomial
variable (P : ℤ → ℤ)
axiom quadratic (a b c : ℤ) : (∀ n : ℤ, P n = a * n^2 + b * n + c) 

-- Conditions for the polynomial
axiom int_coefficients : ∃ (a b c : ℤ), ∀ n : ℤ, P n = a * n^2 + b * n + c
axiom relatively_prime (n : ℤ) (hn : 0 < n) : Int.gcd (P n) n = 1 ∧ Int.gcd (P (P n)) n = 1
axiom P_at_3 : P 3 = 89

-- The main theorem to prove
theorem P_at_10 : P 10 = 859 := by sorry

end P_at_10_l28_28700


namespace vector_parallel_and_on_line_l28_28364

noncomputable def is_point_on_line (x y t : ℝ) : Prop :=
  x = 5 * t + 3 ∧ y = 2 * t + 4

noncomputable def is_parallel (a b c d : ℝ) : Prop :=
  ∃ k : ℝ, a = k * c ∧ b = k * d

theorem vector_parallel_and_on_line :
  ∃ (a b t : ℝ), 
      (a = (5 * t + 3) - 1) ∧ (b = (2 * t + 4) - 1) ∧ 
      is_parallel a b 3 2 ∧ is_point_on_line (5 * t + 3) (2 * t + 4) t := 
by
  use (33 / 4), (11 / 2), (5 / 4)
  sorry

end vector_parallel_and_on_line_l28_28364


namespace sum_first_11_terms_l28_28164

-- Define the arithmetic sequence and sum formula
def arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ :=
  a1 + (n - 1) * d

def sum_arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a1 + (n - 1) * d) / 2

-- Conditions given
variables (a1 d : ℤ)
axiom condition : (a1 + d) + (a1 + 9 * d) = 4

-- Proof statement
theorem sum_first_11_terms : sum_arithmetic_sequence a1 d 11 = 22 :=
by
  -- Placeholder for the actual proof
  sorry

end sum_first_11_terms_l28_28164


namespace jogger_ahead_engine_l28_28302

-- Define the given constants for speed and length
def jogger_speed : ℝ := 2.5 -- in m/s
def train_speed : ℝ := 12.5 -- in m/s
def train_length : ℝ := 120 -- in meters
def passing_time : ℝ := 40 -- in seconds

-- Define the target distance
def jogger_ahead : ℝ := 280 -- in meters

-- Lean 4 statement to prove the jogger is 280 meters ahead of the train's engine
theorem jogger_ahead_engine :
  passing_time * (train_speed - jogger_speed) - train_length = jogger_ahead :=
by
  sorry

end jogger_ahead_engine_l28_28302


namespace not_buy_either_l28_28694

-- Definitions
variables (n T C B : ℕ)
variables (h_n : n = 15)
variables (h_T : T = 9)
variables (h_C : C = 7)
variables (h_B : B = 3)

-- Theorem statement
theorem not_buy_either (n T C B : ℕ) (h_n : n = 15) (h_T : T = 9) (h_C : C = 7) (h_B : B = 3) :
  n - (T - B) - (C - B) - B = 2 :=
sorry

end not_buy_either_l28_28694


namespace james_vacuuming_hours_l28_28173

/-- James spends some hours vacuuming and 3 times as long on the rest of his chores. 
    He spends 12 hours on his chores in total. -/
theorem james_vacuuming_hours (V : ℝ) (h : V + 3 * V = 12) : V = 3 := 
sorry

end james_vacuuming_hours_l28_28173


namespace log_equation_l28_28090

theorem log_equation :
  (3 / (Real.log 1000^4 / Real.log 8)) + (4 / (Real.log 1000^4 / Real.log 9)) = 3 :=
by
  sorry

end log_equation_l28_28090


namespace final_height_of_helicopter_total_fuel_consumed_l28_28503

noncomputable def height_changes : List Float := [4.1, -2.3, 1.6, -0.9, 1.1]

def total_height_change (changes : List Float) : Float :=
  changes.foldl (λ acc x => acc + x) 0

theorem final_height_of_helicopter :
  total_height_change height_changes = 3.6 :=
by
  sorry

noncomputable def fuel_consumption (changes : List Float) : Float :=
  changes.foldl (λ acc x => if x > 0 then acc + 5 * x else acc + 3 * -x) 0

theorem total_fuel_consumed :
  fuel_consumption height_changes = 43.6 :=
by
  sorry

end final_height_of_helicopter_total_fuel_consumed_l28_28503


namespace smaller_area_l28_28836

theorem smaller_area (x y : ℝ) 
  (h1 : x + y = 900)
  (h2 : y - x = (1 / 5) * (x + y) / 2) :
  x = 405 :=
sorry

end smaller_area_l28_28836


namespace solve_inequality_l28_28723

noncomputable def inequality_statement (x : ℝ) : Prop :=
  2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 20

theorem solve_inequality (x : ℝ) :
  (x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5) →
  (inequality_statement x ↔ (x < -3 ∨ (-2 < x ∧ x < 2) ∨ (3 < x ∧ x < 4) ∨ (5 < x ∧ x < 7) ∨ 8 < x)) :=
by sorry

end solve_inequality_l28_28723


namespace green_tiles_in_50th_row_l28_28142

-- Conditions
def tiles_in_row (n : ℕ) : ℕ := 2 * n - 1

def green_tiles_in_row (n : ℕ) : ℕ := (tiles_in_row n - 1) / 2

-- Prove the number of green tiles in the 50th row
theorem green_tiles_in_50th_row : green_tiles_in_row 50 = 49 :=
by
  -- Placeholder proof
  sorry

end green_tiles_in_50th_row_l28_28142


namespace top_width_is_76_l28_28607

-- Definitions of the conditions
def bottom_width : ℝ := 4
def area : ℝ := 10290
def depth : ℝ := 257.25

-- The main theorem to prove that the top width equals 76 meters
theorem top_width_is_76 (x : ℝ) (h : 10290 = 1/2 * (x + 4) * 257.25) : x = 76 :=
by {
  sorry
}

end top_width_is_76_l28_28607


namespace g_recursion_relation_l28_28453

noncomputable def g (n : ℕ) : ℝ :=
  (3 + 2 * Real.sqrt 3) / 6 * ((2 + Real.sqrt 3) / 2)^n +
  (3 - 2 * Real.sqrt 3) / 6 * ((2 - Real.sqrt 3) / 2)^n

theorem g_recursion_relation (n : ℕ) : g (n + 1) - 2 * g n + g (n - 1) = 0 :=
  sorry

end g_recursion_relation_l28_28453


namespace B_k_largest_at_45_l28_28948

def B_k (k : ℕ) : ℝ := (Nat.choose 500 k) * (0.1)^k

theorem B_k_largest_at_45 : ∀ k : ℕ, k = 45 → ∀ m : ℕ, m ≠ 45 → B_k 45 > B_k m :=
by
  intro k h_k m h_m
  sorry

end B_k_largest_at_45_l28_28948


namespace max_x_satisfying_ineq_l28_28062

theorem max_x_satisfying_ineq : ∃ (x : ℤ), (x ≤ 1 ∧ ∀ (y : ℤ), (y > x → y > 1) ∧ (y ≤ 1 → (y : ℚ) / 3 + 7 / 4 < 9 / 4)) := 
by
  sorry

end max_x_satisfying_ineq_l28_28062


namespace find_number_l28_28240

theorem find_number (x number : ℝ) (h₁ : 5 - (5 / x) = number + (4 / x)) (h₂ : x = 9) : number = 4 :=
by
  subst h₂
  -- proof steps
  sorry

end find_number_l28_28240


namespace sum_of_D_coordinates_l28_28076

-- Definition of the midpoint condition
def is_midpoint (N C D : ℝ × ℝ) : Prop :=
  N.1 = (C.1 + D.1) / 2 ∧ N.2 = (C.2 + D.2) / 2

-- Given points
def N : ℝ × ℝ := (5, -1)
def C : ℝ × ℝ := (11, 10)

-- Statement of the problem
theorem sum_of_D_coordinates :
  ∃ D : ℝ × ℝ, is_midpoint N C D ∧ (D.1 + D.2 = -13) :=
  sorry

end sum_of_D_coordinates_l28_28076


namespace chess_tournament_rounds_needed_l28_28406

theorem chess_tournament_rounds_needed
  (num_players : ℕ)
  (num_games_per_round : ℕ)
  (H1 : num_players = 20)
  (H2 : num_games_per_round = 10) :
  (num_players * (num_players - 1)) / num_games_per_round = 38 :=
by
  sorry

end chess_tournament_rounds_needed_l28_28406


namespace profit_of_150_cents_requires_120_oranges_l28_28504

def cost_price_per_orange := 15 / 4  -- cost price per orange in cents
def selling_price_per_orange := 30 / 6  -- selling price per orange in cents
def profit_per_orange := selling_price_per_orange - cost_price_per_orange  -- profit per orange in cents
def required_oranges_to_make_profit := 150 / profit_per_orange  -- number of oranges to get 150 cents of profit

theorem profit_of_150_cents_requires_120_oranges :
  required_oranges_to_make_profit = 120 :=
by
  -- the actual proof will follow here
  sorry

end profit_of_150_cents_requires_120_oranges_l28_28504


namespace n_multiple_of_40_and_infinite_solutions_l28_28767

theorem n_multiple_of_40_and_infinite_solutions 
  (n : ℤ)
  (h1 : ∃ k₁ : ℤ, 2 * n + 1 = k₁^2)
  (h2 : ∃ k₂ : ℤ, 3 * n + 1 = k₂^2)
  : ∃ (m : ℤ), n = 40 * m ∧ ∃ (seq : ℕ → ℤ), 
    (∀ i : ℕ, ∃ k₁ k₂ : ℤ, (2 * (seq i) + 1 = k₁^2) ∧ (3 * (seq i) + 1 = k₂^2) ∧ 
     (i ≠ 0 → seq i ≠ seq (i - 1))) :=
by sorry

end n_multiple_of_40_and_infinite_solutions_l28_28767


namespace cosine_function_range_l28_28563

theorem cosine_function_range : 
  (∀ x ∈ Set.Icc (-Real.pi / 6) (2 * Real.pi / 3), -1/2 ≤ Real.cos x ∧ Real.cos x ≤ 1) ∧
  (∃ a ∈ Set.Icc (-Real.pi / 6) (2 * Real.pi / 3), Real.cos a = 1) ∧
  (∃ b ∈ Set.Icc (-Real.pi / 6) (2 * Real.pi / 3), Real.cos b = -1/2) :=
by
  sorry

end cosine_function_range_l28_28563


namespace min_packs_for_126_cans_l28_28965

-- Definition of pack sizes
def pack_sizes : List ℕ := [15, 18, 36]

-- The given total cans of soda
def total_cans : ℕ := 126

-- The minimum number of packs needed to buy exactly 126 cans of soda
def min_packs_needed (total : ℕ) (packs : List ℕ) : ℕ :=
  -- Function definition to calculate the minimum packs needed
  -- This function needs to be implemented or proven
  sorry

-- The proof that the minimum number of packs needed to buy exactly 126 cans of soda is 4
theorem min_packs_for_126_cans : min_packs_needed total_cans pack_sizes = 4 :=
  -- Proof goes here
  sorry

end min_packs_for_126_cans_l28_28965


namespace largest_divisor_of_product_of_five_consecutive_integers_l28_28711

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ (d : ℤ), d = 60 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l28_28711


namespace find_angle_E_l28_28873

def trapezoid_angles (E H F G : ℝ) : Prop :=
  E + H = 180 ∧ E = 3 * H ∧ G = 4 * F

theorem find_angle_E (E H F G : ℝ) 
  (h1 : E + H = 180)
  (h2 : E = 3 * H)
  (h3 : G = 4 * F) : 
  E = 135 := by
    sorry

end find_angle_E_l28_28873


namespace coyote_time_lemma_l28_28198

theorem coyote_time_lemma (coyote_speed darrel_speed : ℝ) (catch_up_time t : ℝ) 
  (h1 : coyote_speed = 15) (h2 : darrel_speed = 30) (h3 : catch_up_time = 1) (h4 : darrel_speed * catch_up_time = coyote_speed * t) :
  t = 2 :=
by
  sorry

end coyote_time_lemma_l28_28198


namespace min_diff_f_l28_28592

def f (x : ℝ) := 2017 * x ^ 2 - 2018 * x + 2019 * 2020

theorem min_diff_f (t : ℝ) : 
  let f_max := max (f t) (f (t + 2))
  let f_min := min (f t) (f (t + 2))
  (f_max - f_min) ≥ 2017 :=
sorry

end min_diff_f_l28_28592


namespace scientific_notation_120_million_l28_28958

theorem scientific_notation_120_million :
  120000000 = 1.2 * 10^7 :=
by
  sorry

end scientific_notation_120_million_l28_28958


namespace distance_between_points_l28_28792

theorem distance_between_points :
  let x1 := 2
  let y1 := -2
  let x2 := 8
  let y2 := 8
  let dist := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  dist = Real.sqrt 136 :=
by
  -- Proof to be filled in here.
  sorry

end distance_between_points_l28_28792


namespace carpet_needed_l28_28231

/-- A rectangular room with dimensions 15 feet by 9 feet has a non-carpeted area occupied by 
a table with dimensions 3 feet by 2 feet. We want to prove that the number of square yards 
of carpet needed to cover the rest of the floor is 15. -/
theorem carpet_needed
  (room_length : ℝ) (room_width : ℝ) (table_length : ℝ) (table_width : ℝ)
  (h_room : room_length = 15) (h_room_width : room_width = 9)
  (h_table : table_length = 3) (h_table_width : table_width = 2) : 
  (⌈(((room_length * room_width) - (table_length * table_width)) / 9 : ℝ)⌉ = 15) := 
by
  sorry

end carpet_needed_l28_28231


namespace find_y_l28_28639

theorem find_y :
  ∃ (x y : ℤ), (x - 5) / 7 = 7 ∧ (x - y) / 10 = 3 ∧ y = 24 :=
by
  sorry

end find_y_l28_28639


namespace total_toes_on_bus_l28_28402

-- Definitions based on conditions
def toes_per_hand_hoopit : Nat := 3
def hands_per_hoopit : Nat := 4
def number_of_hoopits_on_bus : Nat := 7

def toes_per_hand_neglart : Nat := 2
def hands_per_neglart : Nat := 5
def number_of_neglarts_on_bus : Nat := 8

-- We need to prove that the total number of toes on the bus is 164
theorem total_toes_on_bus :
  (toes_per_hand_hoopit * hands_per_hoopit * number_of_hoopits_on_bus) +
  (toes_per_hand_neglart * hands_per_neglart * number_of_neglarts_on_bus) = 164 :=
by sorry

end total_toes_on_bus_l28_28402


namespace geometric_series_first_term_l28_28234

theorem geometric_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) (h_ratio : r = 1/4) (h_sum : S = 80) (h_series : S = a / (1 - r)) :
  a = 60 :=
by
  sorry

end geometric_series_first_term_l28_28234


namespace boys_in_class_l28_28066

-- Define the conditions given in the problem
def ratio_boys_to_girls (boys girls : ℕ) : Prop := boys = 4 * (boys + girls) / 7 ∧ girls = 3 * (boys + girls) / 7
def total_students (boys girls : ℕ) : Prop := boys + girls = 49

-- Define the statement to be proved
theorem boys_in_class (boys girls : ℕ) (h1 : ratio_boys_to_girls boys girls) (h2 : total_students boys girls) : boys = 28 :=
by
  sorry

end boys_in_class_l28_28066


namespace find_m_l28_28260

open Nat

theorem find_m (m : ℕ) (h_pos : 0 < m) 
  (a : ℕ := Nat.choose (2 * m) m) 
  (b : ℕ := Nat.choose (2 * m + 1) m)
  (h_eq : 13 * a = 7 * b) : 
  m = 6 :=
by
  sorry

end find_m_l28_28260


namespace internal_diagonal_cubes_l28_28211

-- Define the dimensions of the rectangular solid
def x_dimension : ℕ := 168
def y_dimension : ℕ := 350
def z_dimension : ℕ := 390

-- Define the GCD calculations for the given dimensions
def gcd_xy : ℕ := Nat.gcd x_dimension y_dimension
def gcd_yz : ℕ := Nat.gcd y_dimension z_dimension
def gcd_zx : ℕ := Nat.gcd z_dimension x_dimension
def gcd_xyz : ℕ := Nat.gcd (Nat.gcd x_dimension y_dimension) z_dimension

-- Define a statement that the internal diagonal passes through a certain number of cubes
theorem internal_diagonal_cubes :
  x_dimension + y_dimension + z_dimension - gcd_xy - gcd_yz - gcd_zx + gcd_xyz = 880 :=
by
  -- Configuration of conditions and proof skeleton with sorry
  sorry

end internal_diagonal_cubes_l28_28211


namespace fountains_fill_pool_together_l28_28977

-- Define the times in hours for each fountain to fill the pool
def time_fountain1 : ℚ := 5 / 2  -- 2.5 hours
def time_fountain2 : ℚ := 15 / 4 -- 3.75 hours

-- Define the rates at which each fountain can fill the pool
def rate_fountain1 : ℚ := 1 / time_fountain1
def rate_fountain2 : ℚ := 1 / time_fountain2

-- Calculate the combined rate
def combined_rate : ℚ := rate_fountain1 + rate_fountain2

-- Define the time for both fountains working together to fill the pool
def combined_time : ℚ := 1 / combined_rate

-- Prove that the combined time is indeed 1.5 hours
theorem fountains_fill_pool_together : combined_time = 3 / 2 := by
  sorry

end fountains_fill_pool_together_l28_28977


namespace tree_leaves_remaining_after_three_weeks_l28_28745

theorem tree_leaves_remaining_after_three_weeks :
  let initial_leaves := 1000
  let leaves_shed_first_week := (2 / 5 : ℝ) * initial_leaves
  let leaves_remaining_after_first_week := initial_leaves - leaves_shed_first_week
  let leaves_shed_second_week := (4 / 10 : ℝ) * leaves_remaining_after_first_week
  let leaves_remaining_after_second_week := leaves_remaining_after_first_week - leaves_shed_second_week
  let leaves_shed_third_week := (3 / 4 : ℝ) * leaves_shed_second_week
  let leaves_remaining_after_third_week := leaves_remaining_after_second_week - leaves_shed_third_week
  leaves_remaining_after_third_week = 180 :=
by
  sorry

end tree_leaves_remaining_after_three_weeks_l28_28745


namespace smallest_integer_not_expressible_in_form_l28_28214

theorem smallest_integer_not_expressible_in_form :
  ∀ (n : ℕ), (0 < n ∧ (∀ a b c d : ℕ, n ≠ (2^a - 2^b) / (2^c - 2^d))) ↔ n = 11 :=
by
  sorry

end smallest_integer_not_expressible_in_form_l28_28214


namespace range_of_a_l28_28273

noncomputable def A : Set ℝ := Set.Ico 1 5 -- A = [1, 5)
noncomputable def B (a : ℝ) : Set ℝ := Set.Iio a -- B = (-∞, a)

theorem range_of_a (a : ℝ) (h : A ⊆ B a) : 5 ≤ a :=
sorry

end range_of_a_l28_28273


namespace solve_for_x_l28_28271

theorem solve_for_x : ∀ (x : ℝ), (x ≠ 3) → ((x - 3) / (x + 2) + (3 * x - 6) / (x - 3) = 2) → x = 1 / 2 := 
by
  intros x hx h
  sorry

end solve_for_x_l28_28271


namespace order_of_trig_values_l28_28884

noncomputable def a := Real.sin (Real.sin (2008 * Real.pi / 180))
noncomputable def b := Real.sin (Real.cos (2008 * Real.pi / 180))
noncomputable def c := Real.cos (Real.sin (2008 * Real.pi / 180))
noncomputable def d := Real.cos (Real.cos (2008 * Real.pi / 180))

theorem order_of_trig_values : b < a ∧ a < d ∧ d < c :=
by
  sorry

end order_of_trig_values_l28_28884


namespace swimming_speed_l28_28486

theorem swimming_speed (v : ℝ) (water_speed : ℝ) (distance : ℝ) (time : ℝ) 
  (h1 : water_speed = 2) 
  (h2 : distance = 14) 
  (h3 : time = 3.5) 
  (h4 : distance = (v - water_speed) * time) : 
  v = 6 := 
by
  sorry

end swimming_speed_l28_28486


namespace sum_of_reciprocals_l28_28647

noncomputable def roots (p q r : ℂ) : Prop := 
  p ^ 3 - p + 1 = 0 ∧ q ^ 3 - q + 1 = 0 ∧ r ^ 3 - r + 1 = 0

theorem sum_of_reciprocals (p q r : ℂ) (h : roots p q r) : 
  (1 / (p + 2)) + (1 / (q + 2)) + (1 / (r + 2)) = - (10 / 13) := by 
  sorry

end sum_of_reciprocals_l28_28647


namespace derivative_of_gx_eq_3x2_l28_28448

theorem derivative_of_gx_eq_3x2 (f : ℝ → ℝ) : (∀ x : ℝ, f x = (x + 1) * (x^2 - x + 1)) → (∀ x : ℝ, deriv f x = 3 * x^2) :=
by
  intro h
  sorry

end derivative_of_gx_eq_3x2_l28_28448


namespace charles_nickels_l28_28676

theorem charles_nickels :
  ∀ (num_pennies num_cents penny_value nickel_value n : ℕ),
  num_pennies = 6 →
  num_cents = 21 →
  penny_value = 1 →
  nickel_value = 5 →
  (num_cents - num_pennies * penny_value) / nickel_value = n →
  n = 3 :=
by
  intros num_pennies num_cents penny_value nickel_value n hnum_pennies hnum_cents hpenny_value hnickel_value hn
  sorry

end charles_nickels_l28_28676


namespace part_a_solution_part_b_solution_l28_28263

-- Part (a)
theorem part_a_solution (x y : ℝ) :
  x^2 + y^2 - 4*x + 6*y + 13 = 0 ↔ (x = 2 ∧ y = -3) :=
sorry

-- Part (b)
theorem part_b_solution (x y : ℝ) :
  xy - 1 = x - y ↔ ((x = 1 ∨ y = 1) ∨ (x ≠ 1 ∧ y ≠ 1)) :=
sorry

end part_a_solution_part_b_solution_l28_28263


namespace find_n_l28_28992

variable {a : ℕ → ℝ}  -- Defining the sequence

-- Defining the conditions:
def a1 : Prop := a 1 = 1 / 3
def a2_plus_a5 : Prop := a 2 + a 5 = 4
def a_n_eq_33 (n : ℕ) : Prop := a n = 33

theorem find_n (n : ℕ) : a 1 = 1 / 3 → (a 2 + a 5 = 4) → (a n = 33) → n = 50 := 
by 
  intros h1 h2 h3 
  -- the complete proof can be done here
  sorry

end find_n_l28_28992


namespace geometric_sequence_general_term_l28_28690

variable (a : ℕ → ℝ)
variable (n : ℕ)

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n
  
theorem geometric_sequence_general_term 
  (h_geo : is_geometric_sequence a)
  (h_a3 : a 3 = 3)
  (h_a10 : a 10 = 384) :
  a n = 3 * 2^(n-3) :=
by sorry

end geometric_sequence_general_term_l28_28690


namespace euler_conjecture_counter_example_l28_28218

theorem euler_conjecture_counter_example :
  ∃ (n : ℕ), 133^5 + 110^5 + 84^5 + 27^5 = n^5 ∧ n = 144 :=
by
  sorry

end euler_conjecture_counter_example_l28_28218


namespace at_least_one_false_l28_28446

theorem at_least_one_false (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : ¬ ((a + b < c + d) ∧ ((a + b) * (c + d) < a * b + c * d) ∧ ((a + b) * c * d < a * b * (c + d))) :=
  by
  sorry

end at_least_one_false_l28_28446


namespace false_converse_of_vertical_angles_l28_28381

theorem false_converse_of_vertical_angles
  (P : Prop) (Q : Prop) (V : ∀ {A B C D : Type}, (A = B ∧ C = D) → P) (C1 : P → Q) :
  ¬ (Q → P) :=
sorry

end false_converse_of_vertical_angles_l28_28381


namespace determine_m_type_l28_28241

theorem determine_m_type (m : ℝ) :
  ((m^2 + 2*m - 8 = 0) ↔ (m = -4)) ∧
  ((m^2 - 2*m = 0) ↔ (m = 0 ∨ m = 2)) ∧
  ((m^2 - 2*m ≠ 0) ↔ (m ≠ 0 ∧ m ≠ 2)) :=
by sorry

end determine_m_type_l28_28241


namespace soup_adult_feeding_l28_28261

theorem soup_adult_feeding (cans_of_soup : ℕ) (cans_for_children : ℕ) (feeding_ratio : ℕ) 
  (children : ℕ) (adults : ℕ) :
  feeding_ratio = 4 → cans_of_soup = 10 → children = 20 →
  cans_for_children = (children / feeding_ratio) → 
  adults = feeding_ratio * (cans_of_soup - cans_for_children) →
  adults = 20 :=
by
  intros h1 h2 h3 h4 h5
  -- proof goes here
  sorry

end soup_adult_feeding_l28_28261


namespace remaining_painting_time_l28_28182

-- Define the conditions
def total_rooms : ℕ := 10
def hours_per_room : ℕ := 8
def rooms_painted : ℕ := 8

-- Define what we want to prove
theorem remaining_painting_time : (total_rooms - rooms_painted) * hours_per_room = 16 :=
by
  -- Here is where you would provide the proof
  sorry

end remaining_painting_time_l28_28182


namespace sofa_love_seat_cost_l28_28785

theorem sofa_love_seat_cost (love_seat_cost : ℕ) (sofa_cost : ℕ) 
    (h₁ : love_seat_cost = 148) (h₂ : sofa_cost = 2 * love_seat_cost) :
    love_seat_cost + sofa_cost = 444 := 
by
  sorry

end sofa_love_seat_cost_l28_28785


namespace find_t_l28_28945

-- Define the logarithm base 3 function
noncomputable def log_base_3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- Given Condition
def condition (t : ℝ) : Prop := 4 * log_base_3 t = log_base_3 (4 * t) + 2

-- Theorem stating if the given condition holds, then t must be 6
theorem find_t (t : ℝ) (ht : condition t) : t = 6 := 
by
  sorry

end find_t_l28_28945


namespace ratio_Nicolai_to_Charliz_l28_28347

-- Definitions based on conditions
def Haylee_guppies := 3 * 12
def Jose_guppies := Haylee_guppies / 2
def Charliz_guppies := Jose_guppies / 3
def Total_guppies := 84
def Nicolai_guppies := Total_guppies - (Haylee_guppies + Jose_guppies + Charliz_guppies)

-- Proof statement
theorem ratio_Nicolai_to_Charliz : Nicolai_guppies / Charliz_guppies = 4 := by
  sorry

end ratio_Nicolai_to_Charliz_l28_28347


namespace count_more_blue_l28_28757

-- Definitions derived from the provided conditions
variables (total_people more_green both neither : ℕ)
variable (more_blue : ℕ)

-- Condition 1: There are 150 people in total
axiom total_people_def : total_people = 150

-- Condition 2: 90 people believe that teal is "more green"
axiom more_green_def : more_green = 90

-- Condition 3: 35 people believe it is both "more green" and "more blue"
axiom both_def : both = 35

-- Condition 4: 25 people think that teal is neither "more green" nor "more blue"
axiom neither_def : neither = 25


-- Theorem statement
theorem count_more_blue (total_people more_green both neither more_blue : ℕ) 
  (total_people_def : total_people = 150)
  (more_green_def : more_green = 90)
  (both_def : both = 35)
  (neither_def : neither = 25) :
  more_blue = 70 :=
by
  sorry

end count_more_blue_l28_28757


namespace find_point_P_l28_28283

theorem find_point_P :
  ∃ (P : ℝ × ℝ), P.1 = 1 ∧ P.2 = 0 ∧ 
  (P.2 = P.1^4 - P.1) ∧
  (∃ m, m = 4 * P.1^3 - 1 ∧ m = 3) :=
by
  sorry

end find_point_P_l28_28283


namespace real_roots_of_cubic_equation_l28_28233

theorem real_roots_of_cubic_equation : 
  ∃ (S : Finset ℝ), (∀ x ∈ S, (x^3 - 2 * x + 1)^2 = 9) ∧ S.card = 2 := 
by
  sorry

end real_roots_of_cubic_equation_l28_28233


namespace compare_fractions_l28_28528

theorem compare_fractions :
  (111110 / 111111) < (333331 / 333334) ∧ (333331 / 333334) < (222221 / 222223) :=
by
  sorry

end compare_fractions_l28_28528


namespace sum_fractions_eq_l28_28973

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l28_28973


namespace total_amount_spent_l28_28491

namespace KeithSpending

def speakers_cost : ℝ := 136.01
def cd_player_cost : ℝ := 139.38
def tires_cost : ℝ := 112.46
def total_cost : ℝ := 387.85

theorem total_amount_spent : speakers_cost + cd_player_cost + tires_cost = total_cost :=
by sorry

end KeithSpending

end total_amount_spent_l28_28491


namespace arithmetic_mean_18_27_45_l28_28922

theorem arithmetic_mean_18_27_45 : 
  (18 + 27 + 45) / 3 = 30 :=
by
  -- skipping proof
  sorry

end arithmetic_mean_18_27_45_l28_28922


namespace statement_l28_28763

variable {f : ℝ → ℝ}

-- Condition 1: f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Condition 2: f(x-2) = -f(x) for all x
def satisfies_periodicity (f : ℝ → ℝ) : Prop := ∀ x, f (x - 2) = -f x

-- Condition 3: f is decreasing on [0, 2]
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

-- The proof statement
theorem statement (h1 : is_odd_function f) (h2 : satisfies_periodicity f) (h3 : is_decreasing_on f 0 2) :
  f 5 < f 4 ∧ f 4 < f 3 :=
sorry

end statement_l28_28763


namespace t_minus_s_equals_neg_17_25_l28_28148

noncomputable def t : ℝ := (60 + 30 + 20 + 5 + 5) / 5
noncomputable def s : ℝ := (60 * (60 / 120) + 30 * (30 / 120) + 20 * (20 / 120) + 5 * (5 / 120) + 5 * (5 / 120))
noncomputable def t_minus_s : ℝ := t - s

theorem t_minus_s_equals_neg_17_25 : t_minus_s = -17.25 := by
  sorry

end t_minus_s_equals_neg_17_25_l28_28148


namespace bloodPressureFriday_l28_28505

def bloodPressureSunday : ℕ := 120
def bpChangeMonday : ℤ := 20
def bpChangeTuesday : ℤ := -30
def bpChangeWednesday : ℤ := -25
def bpChangeThursday : ℤ := 15
def bpChangeFriday : ℤ := 30

theorem bloodPressureFriday : bloodPressureSunday + bpChangeMonday + bpChangeTuesday + bpChangeWednesday + bpChangeThursday + bpChangeFriday = 130 := by {
  -- Placeholder for the proof
  sorry
}

end bloodPressureFriday_l28_28505


namespace complement_of_union_l28_28963

def U : Set ℝ := Set.univ
def A : Set ℝ := { x | (x - 2) * (x + 1) ≤ 0 }
def B : Set ℝ := { x | 0 ≤ x ∧ x < 3 }

theorem complement_of_union :
  Set.compl (A ∪ B) = { x : ℝ | x < -1 } ∪ { x | x ≥ 3 } := by
  sorry

end complement_of_union_l28_28963


namespace parabola_tangent_xsum_l28_28871

theorem parabola_tangent_xsum
  (p : ℝ) (hp : p > 0) 
  (X_A X_B X_M : ℝ) 
  (hxM_line : ∃ y, y = -2 * p ∧ y = -2 * p)
  (hxA_tangent : ∃ y, y = (X_A / p) * (X_A - X_M) - 2 * p)
  (hxB_tangent : ∃ y, y = (X_B / p) * (X_B - X_M) - 2 * p) :
  2 * X_M = X_A + X_B :=
by
  sorry

end parabola_tangent_xsum_l28_28871


namespace smallest_b_value_l28_28118

variable {a b c d : ℝ}

-- Definitions based on conditions
def is_arithmetic_series (a b c : ℝ) (d : ℝ) : Prop :=
  a = b - d ∧ c = b + d

def abc_product (a b c : ℝ) : Prop :=
  a * b * c = 216

theorem smallest_b_value (a b c d : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (arith_series : is_arithmetic_series a b c d)
  (abc_216 : abc_product a b c) : 
  b ≥ 6 :=
by
  sorry

end smallest_b_value_l28_28118


namespace divisible_by_91_l28_28685

def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 202020
  | _ => -- Define the sequence here, ensuring it constructs the number properly with inserted '2's
    sorry -- this might be a more complex function to define

theorem divisible_by_91 (n : ℕ) : 91 ∣ a n :=
  sorry

end divisible_by_91_l28_28685


namespace greatest_perimeter_l28_28990

theorem greatest_perimeter :
  ∀ (y : ℤ), (y > 4 ∧ y < 20 / 3) → (∃ p : ℤ, p = y + 4 * y + 20 ∧ p = 50) :=
by
  sorry

end greatest_perimeter_l28_28990


namespace solve_x_in_equation_l28_28338

theorem solve_x_in_equation : ∃ (x : ℤ), 24 - 4 * 2 = 3 + x ∧ x = 13 :=
by
  use 13
  sorry

end solve_x_in_equation_l28_28338


namespace baseball_card_decrease_l28_28696

theorem baseball_card_decrease (V : ℝ) (hV : V > 0) (x : ℝ) :
  (1 - x / 100) * (1 - 0.30) = 1 - 0.44 -> x = 20 :=
by {
  -- proof omitted 
  sorry
}

end baseball_card_decrease_l28_28696


namespace number_of_monkeys_l28_28498

theorem number_of_monkeys (X : ℕ) : 
  10 * 10 = 10 →
  1 * 1 = 1 →
  1 * 70 / 10 = 7 →
  (X / 7) = X / 7 :=
by
  intros h1 h2 h3
  sorry

end number_of_monkeys_l28_28498


namespace isosceles_triangle_sides_l28_28537

/-
  Given: 
  - An isosceles triangle with a perimeter of 60 cm.
  - The intersection point of the medians lies on the inscribed circle.
  Prove:
  - The sides of the triangle are 25 cm, 25 cm, and 10 cm.
-/

theorem isosceles_triangle_sides (AB BC AC : ℝ) 
  (h1 : AB = BC)
  (h2 : AB + BC + AC = 60) 
  (h3 : ∃ r : ℝ, r > 0 ∧ 6 * r = AC ∧ 3 * r * AC = 30 * r) :
  AB = 25 ∧ BC = 25 ∧ AC = 10 :=
sorry

end isosceles_triangle_sides_l28_28537


namespace find_cost_price_l28_28818

theorem find_cost_price (SP PP : ℝ) (hSP : SP = 600) (hPP : PP = 25) : 
  ∃ CP : ℝ, CP = 480 := 
by
  sorry

end find_cost_price_l28_28818


namespace ants_movement_impossible_l28_28721

theorem ants_movement_impossible (initial_positions final_positions : Fin 3 → ℝ × ℝ) :
  initial_positions 0 = (0,0) ∧ initial_positions 1 = (0,1) ∧ initial_positions 2 = (1,0) →
  final_positions 0 = (-1,0) ∧ final_positions 1 = (0,1) ∧ final_positions 2 = (1,0) →
  (∀ t : ℕ, ∃ m : Fin 3, 
    ∀ i : Fin 3, (i ≠ m → ∃ k l : ℝ, 
      (initial_positions i).2 - l * (initial_positions i).1 = 0 ∧ 
      ∀ (p : ℕ → ℝ × ℝ), p 0 = initial_positions i ∧ p t = final_positions i → 
      (p 0).1 + k * (p 0).2 = 0)) →
  false :=
by 
  sorry

end ants_movement_impossible_l28_28721


namespace total_annual_salary_excluding_turban_l28_28370

-- Let X be the total amount of money Gopi gives as salary for one year, excluding the turban.
variable (X : ℝ)

-- Condition: The servant leaves after 9 months and receives Rs. 60 plus the turban.
variable (received_money : ℝ)
variable (turban_price : ℝ)

-- Condition values:
axiom received_money_condition : received_money = 60
axiom turban_price_condition : turban_price = 30

-- Question: Prove that X equals 90.
theorem total_annual_salary_excluding_turban :
  3/4 * (X + turban_price) = 90 :=
sorry

end total_annual_salary_excluding_turban_l28_28370


namespace sum_of_numbers_l28_28582

theorem sum_of_numbers (a : ℝ) (n : ℕ) (h : a = 5.3) (hn : n = 8) : (a * n) = 42.4 :=
sorry

end sum_of_numbers_l28_28582


namespace arithmetic_mean_l28_28308

theorem arithmetic_mean (a b : ℚ) (h1 : a = 3/7) (h2 : b = 5/9) :
  (a + b) / 2 = 31/63 := 
by 
  sorry

end arithmetic_mean_l28_28308


namespace triangle_area_l28_28281

theorem triangle_area (a c : ℝ) (B : ℝ) (h_a : a = 7) (h_c : c = 5) (h_B : B = 120 * Real.pi / 180) : 
  (1 / 2 * a * c * Real.sin B) = 35 * Real.sqrt 3 / 4 := by
  sorry

end triangle_area_l28_28281


namespace find_d_l28_28727

theorem find_d
  (a b c d : ℝ)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_c_pos : c > 0)
  (h_d_pos : d > 0)
  (h_max : a * 1 + d = 5)
  (h_min : a * (-1) + d = -3) :
  d = 1 := 
sorry

end find_d_l28_28727


namespace no_integer_solutions_l28_28138

theorem no_integer_solutions (x y z : ℤ) (h : ¬ (x = 0 ∧ y = 0 ∧ z = 0)) : 2 * x^4 + y^4 ≠ 7 * z^4 :=
sorry

end no_integer_solutions_l28_28138


namespace percentage_decrease_increase_l28_28618

theorem percentage_decrease_increase (S : ℝ) (x : ℝ) (h : S > 0) :
  S * (1 - x / 100) * (1 + x / 100) = S * (64 / 100) → x = 6 :=
by
  sorry

end percentage_decrease_increase_l28_28618


namespace remainder_of_eggs_is_2_l28_28075

-- Define the number of eggs each person has
def david_eggs : ℕ := 45
def emma_eggs : ℕ := 52
def fiona_eggs : ℕ := 25

-- Define total eggs and remainder function
def total_eggs : ℕ := david_eggs + emma_eggs + fiona_eggs
def remainder (a b : ℕ) : ℕ := a % b

-- Prove that the remainder of total eggs divided by 10 is 2
theorem remainder_of_eggs_is_2 : remainder total_eggs 10 = 2 := by
  sorry

end remainder_of_eggs_is_2_l28_28075


namespace minimum_value_of_f_l28_28692

noncomputable def f (x : ℝ) : ℝ := x^2 / (x - 5)

theorem minimum_value_of_f : ∃ (x : ℝ), x > 5 ∧ f x = 20 :=
by
  use 10
  sorry

end minimum_value_of_f_l28_28692


namespace problem_statement_l28_28340

theorem problem_statement
  (a b : ℝ)
  (ha : a = Real.sqrt 2 + 1)
  (hb : b = Real.sqrt 2 - 1) :
  a^2 - a * b + b^2 = 5 :=
sorry

end problem_statement_l28_28340


namespace complex_purely_imaginary_condition_l28_28239

theorem complex_purely_imaginary_condition (a : ℝ) :
  (a = 1 → (a - 1) * (a + 2) + (a + 3) * Complex.I.im = (0 : ℝ)) ∧
  ¬(a = 1 ∧ ¬a = -2 → (a - 1) * (a + 2) + (a + 3) * Complex.I.im = (0 : ℝ)) :=
  sorry

end complex_purely_imaginary_condition_l28_28239


namespace overall_gain_percent_l28_28488

variables (C_A S_A C_B S_B : ℝ)

def cost_price_A (n : ℝ) : ℝ := n * C_A
def selling_price_A (n : ℝ) : ℝ := n * S_A

def cost_price_B (n : ℝ) : ℝ := n * C_B
def selling_price_B (n : ℝ) : ℝ := n * S_B

theorem overall_gain_percent :
  (selling_price_A 25 = cost_price_A 50) →
  (selling_price_B 30 = cost_price_B 60) →
  ((S_A - C_A) / C_A * 100 = 100) ∧ ((S_B - C_B) / C_B * 100 = 100) :=
by
  sorry

end overall_gain_percent_l28_28488


namespace hawks_points_l28_28739

def touchdowns : ℕ := 3
def points_per_touchdown : ℕ := 7
def total_points (t : ℕ) (p : ℕ) : ℕ := t * p

theorem hawks_points : total_points touchdowns points_per_touchdown = 21 :=
by
  -- Proof will go here
  sorry

end hawks_points_l28_28739


namespace total_crayons_l28_28459

-- We're given the conditions
def crayons_per_child : ℕ := 6
def number_of_children : ℕ := 12

-- We need to prove the total number of crayons.
theorem total_crayons (c : ℕ := crayons_per_child) (n : ℕ := number_of_children) : (c * n) = 72 := by
  sorry

end total_crayons_l28_28459


namespace largest_divisor_of_n_squared_l28_28052

theorem largest_divisor_of_n_squared (n : ℕ) (h_pos : 0 < n) (h_div : ∀ d, d ∣ n^2 → d = 900) : 900 ∣ n^2 :=
by sorry

end largest_divisor_of_n_squared_l28_28052


namespace opening_night_ticket_price_l28_28895

theorem opening_night_ticket_price :
  let matinee_customers := 32
  let evening_customers := 40
  let opening_night_customers := 58
  let matinee_price := 5
  let evening_price := 7
  let popcorn_price := 10
  let total_revenue := 1670
  let total_customers := matinee_customers + evening_customers + opening_night_customers
  let popcorn_customers := total_customers / 2
  let total_matinee_revenue := matinee_customers * matinee_price
  let total_evening_revenue := evening_customers * evening_price
  let total_popcorn_revenue := popcorn_customers * popcorn_price
  let known_revenue := total_matinee_revenue + total_evening_revenue + total_popcorn_revenue
  let opening_night_revenue := total_revenue - known_revenue
  let opening_night_price := opening_night_revenue / opening_night_customers
  opening_night_price = 10 := by
  sorry

end opening_night_ticket_price_l28_28895


namespace meals_given_away_l28_28560

def initial_meals_colt_and_curt : ℕ := 113
def additional_meals_sole_mart : ℕ := 50
def remaining_meals : ℕ := 78
def total_initial_meals : ℕ := initial_meals_colt_and_curt + additional_meals_sole_mart
def given_away_meals (total : ℕ) (remaining : ℕ) : ℕ := total - remaining

theorem meals_given_away : given_away_meals total_initial_meals remaining_meals = 85 :=
by
  sorry

end meals_given_away_l28_28560


namespace full_seasons_already_aired_l28_28303

variable (days_until_premiere : ℕ)
variable (episodes_per_day : ℕ)
variable (episodes_per_season : ℕ)

theorem full_seasons_already_aired (h_days : days_until_premiere = 10)
                                  (h_episodes_day : episodes_per_day = 6)
                                  (h_episodes_season : episodes_per_season = 15) :
  (days_until_premiere * episodes_per_day) / episodes_per_season = 4 := by
  sorry

end full_seasons_already_aired_l28_28303


namespace B_grazed_months_l28_28326

-- Define the conditions
variables (A_cows B_cows C_cows D_cows : ℕ)
variables (A_months B_months C_months D_months : ℕ)
variables (A_rent total_rent : ℕ)

-- Given conditions
def A_condition := (A_cows = 24 ∧ A_months = 3)
def B_condition := (B_cows = 10)
def C_condition := (C_cows = 35 ∧ C_months = 4)
def D_condition := (D_cows = 21 ∧ D_months = 3)
def A_rent_condition := (A_rent = 720)
def total_rent_condition := (total_rent = 3250)

-- Define cow-months calculation
def cow_months (cows months : ℕ) : ℕ := cows * months

-- Define cost per cow-month
def cost_per_cow_month (rent cow_months : ℕ) : ℕ := rent / cow_months

-- Define B's months of grazing proof problem
theorem B_grazed_months
  (A_cows_months : cow_months 24 3 = 72)
  (B_cows := 10)
  (C_cows_months : cow_months 35 4 = 140)
  (D_cows_months : cow_months 21 3 = 63)
  (A_rent_condition : A_rent = 720)
  (total_rent_condition : total_rent = 3250) :
  ∃ (B_months : ℕ), 10 * B_months = 50 ∧ B_months = 5 := sorry

end B_grazed_months_l28_28326


namespace books_initially_l28_28953

theorem books_initially (A B : ℕ) (h1 : A = 3) (h2 : B = (A + 2) + 2) : B = 7 :=
by
  -- Using the given facts, we need to show B = 7
  sorry

end books_initially_l28_28953


namespace second_number_value_l28_28132

def first_number := ℚ
def second_number := ℚ

variables (x y : ℚ)

/-- Given conditions: 
      (1) \( \frac{1}{5}x = \frac{5}{8}y \)
      (2) \( x + 35 = 4y \)
    Prove that \( y = 40 \) 
-/
theorem second_number_value (h1 : (1/5 : ℚ) * x = (5/8 : ℚ) * y) (h2 : x + 35 = 4 * y) : 
  y = 40 :=
sorry

end second_number_value_l28_28132


namespace searchlight_revolutions_l28_28380

theorem searchlight_revolutions (p : ℝ) (r : ℝ) (t : ℝ) 
  (h1 : p = 0.6666666666666667) 
  (h2 : t = 10) 
  (h3 : p = (60 / r - t) / (60 / r)) : 
  r = 2 :=
by sorry

end searchlight_revolutions_l28_28380


namespace post_height_l28_28959

theorem post_height 
  (circumference : ℕ) 
  (rise_per_circuit : ℕ) 
  (travel_distance : ℕ)
  (circuits : ℕ := travel_distance / circumference) 
  (total_rise : ℕ := circuits * rise_per_circuit) 
  (c : circumference = 3)
  (r : rise_per_circuit = 4)
  (t : travel_distance = 9) :
  total_rise = 12 := by
  sorry

end post_height_l28_28959


namespace solution_set_l28_28323

def f (x m : ℝ) : ℝ := |x - 1| - |x - m|

theorem solution_set (x : ℝ) :
  (f x 2) ≥ 1 ↔ x ≥ 2 :=
sorry

end solution_set_l28_28323


namespace one_fourths_in_one_eighth_l28_28906

theorem one_fourths_in_one_eighth : (1 / 8) / (1 / 4) = 1 / 2 := 
by 
  sorry

end one_fourths_in_one_eighth_l28_28906


namespace johns_hats_cost_l28_28074

theorem johns_hats_cost 
  (weeks : ℕ)
  (days_in_week : ℕ)
  (cost_per_hat : ℕ) 
  (h : weeks = 2 ∧ days_in_week = 7 ∧ cost_per_hat = 50) 
  : (weeks * days_in_week * cost_per_hat) = 700 :=
by
  sorry

end johns_hats_cost_l28_28074


namespace inscribed_quadrilateral_circle_eq_radius_l28_28383

noncomputable def inscribed_circle_condition (AB CD AD BC : ℝ) : Prop :=
  AB + CD = AD + BC

noncomputable def equal_radius_condition (r₁ r₂ r₃ r₄ : ℝ) : Prop :=
  r₁ = r₃ ∨ r₄ = r₂

theorem inscribed_quadrilateral_circle_eq_radius 
  (AB CD AD BC r₁ r₂ r₃ r₄ : ℝ)
  (h_inscribed_circle: inscribed_circle_condition AB CD AD BC)
  (h_four_circles: ∀ i, (i = 1 ∨ i = 2 ∨ i = 3 ∨ i = 4) → ∃ (r : ℝ), r = rᵢ): 
  equal_radius_condition r₁ r₂ r₃ r₄ :=
by {
  sorry
}

end inscribed_quadrilateral_circle_eq_radius_l28_28383


namespace no_real_intersection_l28_28059

theorem no_real_intersection (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, x * f y = y * f x) 
  (h2 : f 1 = -1) : ¬∃ x : ℝ, f x = x^2 + 1 :=
by
  sorry

end no_real_intersection_l28_28059


namespace negation_of_universal_prop_l28_28761

theorem negation_of_universal_prop:
  (¬ (∀ x : ℝ, x ^ 3 - x ≥ 0)) ↔ (∃ x : ℝ, x ^ 3 - x < 0) := 
by 
sorry

end negation_of_universal_prop_l28_28761


namespace sets_equivalence_l28_28872

theorem sets_equivalence :
  (∀ M N, (M = {(3, 2)} ∧ N = {(2, 3)} → M ≠ N) ∧
          (M = {4, 5} ∧ N = {5, 4} → M = N) ∧
          (M = {1, 2} ∧ N = {(1, 2)} → M ≠ N) ∧
          (M = {(x, y) | x + y = 1} ∧ N = {y | ∃ x, x + y = 1} → M ≠ N)) :=
by sorry

end sets_equivalence_l28_28872


namespace find_x_value_l28_28773

def average_eq_condition (x : ℝ) : Prop :=
  (5050 + x) / 101 = 50 * (x + 1)

theorem find_x_value : ∃ x : ℝ, average_eq_condition x ∧ x = 0 :=
by
  use 0
  sorry

end find_x_value_l28_28773


namespace total_money_l28_28562

variable (A B C : ℕ)

theorem total_money
  (h1 : A + C = 250)
  (h2 : B + C = 450)
  (h3 : C = 100) :
  A + B + C = 600 := by
  sorry

end total_money_l28_28562


namespace chess_positions_after_one_move_each_l28_28508

def number_of_chess_positions (initial_positions : ℕ) (pawn_moves : ℕ) (knight_moves : ℕ) (active_pawns : ℕ) (active_knights : ℕ) : ℕ :=
  let pawn_move_combinations := active_pawns * pawn_moves
  let knight_move_combinations := active_knights * knight_moves
  pawn_move_combinations + knight_move_combinations

theorem chess_positions_after_one_move_each :
  number_of_chess_positions 1 2 2 8 2 * number_of_chess_positions 1 2 2 8 2 = 400 :=
by
  sorry

end chess_positions_after_one_move_each_l28_28508


namespace ratio_expression_l28_28604

-- Given conditions: X : Y : Z = 3 : 2 : 6
def ratio (X Y Z : ℚ) : Prop := X / Y = 3 / 2 ∧ Y / Z = 2 / 6

-- The expression to be evaluated
def expr (X Y Z : ℚ) : ℚ := (4 * X + 3 * Y) / (5 * Z - 2 * X)

-- The proof problem itself
theorem ratio_expression (X Y Z : ℚ) (h : ratio X Y Z) : expr X Y Z = 3 / 4 := by
  sorry

end ratio_expression_l28_28604


namespace generatrix_length_of_cone_l28_28192

theorem generatrix_length_of_cone (r l : ℝ) (π : ℝ) (sqrt : ℝ → ℝ) 
  (hx : r = sqrt 2)
  (h_baseline_length : 2 * π * r = π * l) : 
  l = 2 * sqrt 2 :=
by
  sorry

end generatrix_length_of_cone_l28_28192


namespace difference_between_second_and_third_levels_l28_28912

def total_parking_spots : ℕ := 400
def first_level_open_spots : ℕ := 58
def second_level_open_spots : ℕ := first_level_open_spots + 2
def fourth_level_open_spots : ℕ := 31
def total_full_spots : ℕ := 186

def total_open_spots : ℕ := total_parking_spots - total_full_spots

def third_level_open_spots : ℕ := 
  total_open_spots - (first_level_open_spots + second_level_open_spots + fourth_level_open_spots)

def difference_open_spots : ℕ := third_level_open_spots - second_level_open_spots

theorem difference_between_second_and_third_levels : difference_open_spots = 5 :=
sorry

end difference_between_second_and_third_levels_l28_28912


namespace propositions_correct_l28_28512

variable {R : Type} [LinearOrderedField R] {A B : Set R}

theorem propositions_correct :
  (¬ ∃ x : R, x^2 + x + 1 = 0) ∧
  (¬ (∃ x : R, x + 1 ≤ 2) → ∀ x : R, x + 1 > 2) ∧
  (∀ x : R, x ∈ A ∩ B → x ∈ A) ∧
  (∀ x : R, x > 3 → x^2 > 9 ∧ ∃ y : R, y^2 > 9 ∧ y < 3) :=
by
  sorry

end propositions_correct_l28_28512


namespace shaded_area_l28_28844

noncomputable def squareArea (a : ℝ) : ℝ := a * a

theorem shaded_area {s : ℝ} (h1 : squareArea s = 1) (h2 : s / s = 2) : 
  ∃ (shaded : ℝ), shaded = 1 / 3 :=
by
  sorry

end shaded_area_l28_28844


namespace area_of_grey_part_l28_28168

theorem area_of_grey_part :
  let area1 := 8 * 10
  let area2 := 12 * 9
  let area_black := 37
  let area_white := 43
  area2 - area_white = 65 :=
by
  let area1 := 8 * 10
  let area2 := 12 * 9
  let area_black := 37
  let area_white := 43
  have : area2 - area_white = 65 := by sorry
  exact this

end area_of_grey_part_l28_28168


namespace remainder_37_remainder_73_l28_28731

theorem remainder_37 (N : ℕ) (k : ℕ) (h : N = 1554 * k + 131) : N % 37 = 20 := sorry

theorem remainder_73 (N : ℕ) (k : ℕ) (h : N = 1554 * k + 131) : N % 73 = 58 := sorry

end remainder_37_remainder_73_l28_28731


namespace expected_winnings_l28_28908

theorem expected_winnings (roll_1_2: ℝ) (roll_3_4: ℝ) (roll_5_6: ℝ) (p1_2 p3_4 p5_6: ℝ) :
    roll_1_2 = 2 →
    roll_3_4 = 4 →
    roll_5_6 = -6 →
    p1_2 = 1 / 8 →
    p3_4 = 1 / 4 →
    p5_6 = 1 / 8 →
    (2 * p1_2 + 2 * p1_2 + 4 * p3_4 + 4 * p3_4 + roll_5_6 * p5_6 + roll_5_6 * p5_6) = 1 := by
  intros
  sorry

end expected_winnings_l28_28908


namespace noncongruent_triangles_count_l28_28077

/-- Prove that the number of noncongruent integer-sided triangles 
with positive area and perimeter less than 20, 
which are neither equilateral, isosceles, nor right triangles, is 15. -/
theorem noncongruent_triangles_count : 
  ∃ n : ℕ, 
  (∀ (a b c : ℕ) (h : a ≤ b ∧ b ≤ c),
    a + b + c < 20 ∧ a + b > c ∧ a^2 + b^2 ≠ c^2 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c → n ≥ 15) :=
sorry

end noncongruent_triangles_count_l28_28077


namespace translate_function_down_l28_28531

theorem translate_function_down 
  (f : ℝ → ℝ) 
  (a k : ℝ) 
  (h : ∀ x, f x = a * x) 
  : ∀ x, (f x - k) = a * x - k :=
by
  sorry

end translate_function_down_l28_28531


namespace number_of_intersections_l28_28638

theorem number_of_intersections (x y : ℝ) :
  (x^2 + y^2 = 16) ∧ (x = 4) → (x = 4 ∧ y = 0) :=
by {
  sorry
}

end number_of_intersections_l28_28638


namespace twelve_integers_divisible_by_eleven_l28_28864

theorem twelve_integers_divisible_by_eleven (a : Fin 12 → ℤ) : 
  ∃ (i j : Fin 12), i ≠ j ∧ 11 ∣ (a i - a j) :=
by
  sorry

end twelve_integers_divisible_by_eleven_l28_28864


namespace probability_complement_B_probability_union_A_B_l28_28899

variable (Ω : Type) [MeasurableSpace Ω] {P : ProbabilityMeasure Ω}
variable (A B : Set Ω)

theorem probability_complement_B
  (hB : P B = 1 / 3) : P Bᶜ = 2 / 3 :=
by
  sorry

theorem probability_union_A_B
  (hA : P A = 1 / 2) (hB : P B = 1 / 3) : P (A ∪ B) ≤ 5 / 6 :=
by
  sorry

end probability_complement_B_probability_union_A_B_l28_28899


namespace terminating_decimal_count_l28_28716

theorem terminating_decimal_count : ∃ n, n = 23 ∧ (∀ k, 1 ≤ k ∧ k ≤ 499 → (∃ m, k = 21 * m)) :=
by
  sorry

end terminating_decimal_count_l28_28716


namespace find_a1_l28_28204

theorem find_a1 (a b : ℕ → ℝ) (h1 : ∀ n ≥ 1, a (n + 1) + b (n + 1) = (a n + b n) / 2) 
  (h2 : ∀ n ≥ 1, a (n + 1) * b (n + 1) = (a n * b n) ^ (1/2)) 
  (hb2016 : b 2016 = 1) (ha1_pos : a 1 > 0) :
  a 1 = 2^2015 :=
sorry

end find_a1_l28_28204


namespace monkey_climbing_distance_l28_28112

theorem monkey_climbing_distance
  (x : ℝ)
  (h1 : ∀ t : ℕ, t % 2 = 0 → t ≠ 0 → x - 3 > 0) -- condition (2,4)
  (h2 : ∀ t : ℕ, t % 2 = 1 → x > 0) -- condition (5)
  (h3 : 18 * (x - 3) + x = 60) -- condition (6)
  : x = 6 :=
sorry

end monkey_climbing_distance_l28_28112


namespace total_hotdogs_brought_l28_28514

-- Define the number of hotdogs brought by the first and second neighbors based on given conditions.

def first_neighbor_hotdogs : Nat := 75
def second_neighbor_hotdogs : Nat := first_neighbor_hotdogs - 25

-- Prove that the total hotdogs brought by the neighbors equals 125.
theorem total_hotdogs_brought :
  first_neighbor_hotdogs + second_neighbor_hotdogs = 125 :=
by
  -- statement only, proof not required
  sorry

end total_hotdogs_brought_l28_28514


namespace find_total_students_l28_28724

variables (x X : ℕ)
variables (x_percent_students : ℕ) (total_students : ℕ)
variables (boys_fraction : ℝ)

-- Provided Conditions
axiom a1 : x_percent_students = 120
axiom a2 : boys_fraction = 0.30
axiom a3 : total_students = X

-- The theorem we need to prove
theorem find_total_students (a1 : 120 = x_percent_students) 
                            (a2 : boys_fraction = 0.30) 
                            (a3 : total_students = X) : 
  120 = (x / 100) * (boys_fraction * total_students) :=
sorry

end find_total_students_l28_28724


namespace original_cost_of_tshirt_l28_28861

theorem original_cost_of_tshirt
  (backpack_cost : ℕ := 10)
  (cap_cost : ℕ := 5)
  (total_spent_after_discount : ℕ := 43)
  (discount : ℕ := 2)
  (tshirt_cost_before_discount : ℕ) :
  total_spent_after_discount + discount - (backpack_cost + cap_cost) = tshirt_cost_before_discount :=
by
  sorry

end original_cost_of_tshirt_l28_28861


namespace find_m_values_l28_28713

theorem find_m_values {m : ℝ} :
  (∀ x : ℝ, mx^2 + (m+2) * x + (1 / 2) * m + 1 = 0 → x = 0) 
  ↔ (m = 0 ∨ m = 2 ∨ m = -2) :=
by sorry

end find_m_values_l28_28713


namespace total_number_of_coins_l28_28393

variable (nickels dimes total_value : ℝ)
variable (total_nickels : ℕ)

def value_of_nickel : ℝ := 0.05
def value_of_dime : ℝ := 0.10

theorem total_number_of_coins :
  total_value = 3.50 → total_nickels = 30 → total_value = total_nickels * value_of_nickel + dimes * value_of_dime → 
  total_nickels + dimes = 50 :=
by
  intros h_total_value h_total_nickels h_value_equation
  sorry

end total_number_of_coins_l28_28393


namespace businessmen_drink_neither_l28_28258

theorem businessmen_drink_neither (n c t b : ℕ) 
  (h_n : n = 30) 
  (h_c : c = 15) 
  (h_t : t = 13) 
  (h_b : b = 7) : 
  n - (c + t - b) = 9 := 
  by
  sorry

end businessmen_drink_neither_l28_28258


namespace billy_unknown_lap_time_l28_28005

theorem billy_unknown_lap_time :
  ∀ (time_first_5_laps time_next_3_laps time_last_lap time_margaret total_time_billy : ℝ) (lap_time_unknown : ℝ),
    time_first_5_laps = 2 ∧
    time_next_3_laps = 4 ∧
    time_last_lap = 2.5 ∧
    time_margaret = 10 ∧
    total_time_billy = time_margaret - 0.5 →
    (time_first_5_laps + time_next_3_laps + time_last_lap + lap_time_unknown = total_time_billy) →
    lap_time_unknown = 1 :=
by
  sorry

end billy_unknown_lap_time_l28_28005


namespace harry_terry_difference_l28_28471

theorem harry_terry_difference :
  let H := 12 - (3 * 4)
  let T := 12 - (3 * 4) -- Correcting Terry's mistake
  H - T = 0 := by
  sorry

end harry_terry_difference_l28_28471


namespace solve_equation_l28_28388

theorem solve_equation : ∃ x : ℤ, (x - 15) / 3 = (3 * x + 11) / 8 ∧ x = -153 := 
by
  use -153
  sorry

end solve_equation_l28_28388


namespace f_five_eq_three_f_three_x_inv_f_243_l28_28499

-- Define the function f satisfying the given conditions.
def f (x : ℕ) : ℕ :=
  if x = 5 then 3
  else if x = 15 then 9
  else if x = 45 then 27
  else if x = 135 then 81
  else if x = 405 then 243
  else 0

-- Define the condition f(5) = 3
theorem f_five_eq_three : f 5 = 3 := rfl

-- Define the condition f(3x) = 3f(x) for all x
theorem f_three_x (x : ℕ) : f (3 * x) = 3 * f x :=
sorry

-- Prove that f⁻¹(243) = 405.
theorem inv_f_243 : f (405) = 243 :=
by sorry

-- Concluding the proof statement using the concluded theorems.
example : f (405) = 243 :=
by apply inv_f_243

end f_five_eq_three_f_three_x_inv_f_243_l28_28499


namespace tiffany_reading_homework_pages_l28_28321

theorem tiffany_reading_homework_pages 
  (math_pages : ℕ)
  (problems_per_page : ℕ)
  (total_problems : ℕ)
  (reading_pages : ℕ)
  (H1 : math_pages = 6)
  (H2 : problems_per_page = 3)
  (H3 : total_problems = 30)
  (H4 : reading_pages = (total_problems - math_pages * problems_per_page) / problems_per_page) 
  : reading_pages = 4 := 
sorry

end tiffany_reading_homework_pages_l28_28321


namespace min_d_value_l28_28748

noncomputable def minChordLength (a : ℝ) : ℝ :=
  let P1 := (Real.arcsin a, Real.arcsin a)
  let P2 := (Real.arccos a, -Real.arccos a)
  let d_sq := 2 * ((Real.arcsin a)^2 + (Real.arccos a)^2)
  Real.sqrt d_sq

theorem min_d_value {a : ℝ} (h₁ : a ∈ Set.Icc (-1) 1) : 
  ∃ d : ℝ, d = minChordLength a ∧ d ≥ (π / 2) :=
sorry

end min_d_value_l28_28748


namespace rose_price_vs_carnation_price_l28_28566

variable (x y : ℝ)

theorem rose_price_vs_carnation_price
  (h1 : 3 * x + 2 * y > 8)
  (h2 : 2 * x + 3 * y < 7) :
  x > 2 * y :=
sorry

end rose_price_vs_carnation_price_l28_28566


namespace kendra_packs_l28_28967

/-- Kendra has some packs of pens. Tony has 2 packs of pens. There are 3 pens in each pack. 
Kendra and Tony decide to keep two pens each and give the remaining pens to their friends 
one pen per friend. They give pens to 14 friends. Prove that Kendra has 4 packs of pens. --/
theorem kendra_packs : ∀ (kendra_pens tony_pens pens_per_pack pens_kept pens_given friends : ℕ),
  tony_pens = 2 →
  pens_per_pack = 3 →
  pens_kept = 2 →
  pens_given = 14 →
  tony_pens * pens_per_pack - pens_kept + kendra_pens - pens_kept = pens_given →
  kendra_pens / pens_per_pack = 4 :=
by
  intros kendra_pens tony_pens pens_per_pack pens_kept pens_given friends
  intro h1
  intro h2
  intro h3
  intro h4
  intro h5
  sorry

end kendra_packs_l28_28967


namespace Kevin_ends_with_54_cards_l28_28146

/-- Kevin starts with 7 cards and finds another 47 cards. 
    This theorem proves that Kevin ends with 54 cards. -/
theorem Kevin_ends_with_54_cards :
  let initial_cards := 7
  let found_cards := 47
  initial_cards + found_cards = 54 := 
by
  let initial_cards := 7
  let found_cards := 47
  sorry

end Kevin_ends_with_54_cards_l28_28146


namespace sequence_explicit_formula_l28_28543

noncomputable def sequence_a : ℕ → ℝ
| 0     => 0  -- Not used, but needed for definition completeness
| 1     => 3
| (n+1) => n / (n + 1) * sequence_a n

theorem sequence_explicit_formula (n : ℕ) (h : n ≠ 0) :
  sequence_a n = 3 / n :=
by sorry

end sequence_explicit_formula_l28_28543


namespace toilet_paper_squares_per_roll_l28_28866

theorem toilet_paper_squares_per_roll
  (trips_per_day : ℕ)
  (squares_per_trip : ℕ)
  (num_rolls : ℕ)
  (supply_days : ℕ)
  (total_squares : ℕ)
  (squares_per_roll : ℕ)
  (h1 : trips_per_day = 3)
  (h2 : squares_per_trip = 5)
  (h3 : num_rolls = 1000)
  (h4 : supply_days = 20000)
  (h5 : total_squares = trips_per_day * squares_per_trip * supply_days)
  (h6 : squares_per_roll = total_squares / num_rolls) :
  squares_per_roll = 300 :=
by sorry

end toilet_paper_squares_per_roll_l28_28866


namespace chess_or_basketball_students_l28_28915

-- Definitions based on the conditions
def percentage_likes_basketball : ℝ := 0.4
def percentage_likes_chess : ℝ := 0.1
def total_students : ℕ := 250

-- Main statement to prove
theorem chess_or_basketball_students : 
  (percentage_likes_basketball + percentage_likes_chess) * total_students = 125 :=
by
  sorry

end chess_or_basketball_students_l28_28915


namespace cos_54_deg_l28_28979

-- Define cosine function
noncomputable def cos_deg (θ : ℝ) : ℝ := Real.cos (θ * Real.pi / 180)

-- The main theorem statement
theorem cos_54_deg : cos_deg 54 = (-1 + Real.sqrt 5) / 4 :=
  sorry

end cos_54_deg_l28_28979


namespace sum_min_max_z_l28_28984

theorem sum_min_max_z (x y : ℝ) 
  (h1 : x - y - 2 ≥ 0) 
  (h2 : x - 5 ≤ 0) 
  (h3 : y + 2 ≥ 0) :
  ∃ (z_min z_max : ℝ), z_min = 2 ∧ z_max = 34 ∧ z_min + z_max = 36 :=
by
  sorry

end sum_min_max_z_l28_28984


namespace equivalent_single_discount_l28_28625

variable (original_price : ℝ)
variable (first_discount : ℝ)
variable (second_discount : ℝ)

-- Conditions
def sale_price (p : ℝ) (d : ℝ) : ℝ := p * (1 - d)

def final_price (p : ℝ) (d1 d2 : ℝ) : ℝ :=
  let sale1 := sale_price p d1
  sale_price sale1 d2

-- Prove the equivalent single discount is as described
theorem equivalent_single_discount :
  original_price = 30 → first_discount = 0.2 → second_discount = 0.25 →
  (1 - final_price original_price first_discount second_discount / original_price) * 100 = 40 :=
by
  intros
  sorry

end equivalent_single_discount_l28_28625


namespace leila_cakes_monday_l28_28829

def number_of_cakes_monday (m : ℕ) : Prop :=
  let cakes_friday := 9
  let cakes_saturday := 3 * m
  let total_cakes := m + cakes_friday + cakes_saturday
  total_cakes = 33

theorem leila_cakes_monday : ∃ m : ℕ, number_of_cakes_monday m ∧ m = 6 :=
by 
  -- We propose that the number of cakes she ate on Monday, denoted as m, is 6.
  -- We need to prove that this satisfies the given conditions.
  -- This line is a placeholder for the proof.
  sorry

end leila_cakes_monday_l28_28829


namespace russian_players_pairing_probability_l28_28868

theorem russian_players_pairing_probability :
  let total_players := 10
  let russian_players := 4
  (russian_players * (russian_players - 1)) / (total_players * (total_players - 1)) * 
  ((russian_players - 2) * (russian_players - 3)) / ((total_players - 2) * (total_players - 3)) = 1 / 21 :=
by
  sorry

end russian_players_pairing_probability_l28_28868


namespace find_P_Q_sum_l28_28662

theorem find_P_Q_sum (P Q : ℤ) 
  (h : ∃ b c : ℤ, x^2 + 3 * x + 2 ∣ x^4 + P * x^2 + Q 
    ∧ b + 3 = 0 
    ∧ c + 3 * b + 6 = P 
    ∧ 3 * c + 2 * b = 0 
    ∧ 2 * c = Q): 
  P + Q = 3 := 
sorry

end find_P_Q_sum_l28_28662


namespace choir_row_lengths_l28_28796

theorem choir_row_lengths : 
  ∃ s : Finset ℕ, (∀ d ∈ s, d ∣ 90 ∧ 6 ≤ d ∧ d ≤ 15) ∧ s.card = 4 := by
  sorry

end choir_row_lengths_l28_28796


namespace count_triangles_l28_28980

-- Assuming the conditions are already defined and given as parameters  
-- Let's define a proposition to prove the solution

noncomputable def total_triangles_in_figure : ℕ := 68

-- Create the theorem statement:
theorem count_triangles : total_triangles_in_figure = 68 := 
by
  sorry

end count_triangles_l28_28980


namespace tournament_trio_l28_28384

theorem tournament_trio
  (n : ℕ)
  (h_n : n ≥ 3)
  (match_result : Fin n → Fin n → Prop)
  (h1 : ∀ i j : Fin n, i ≠ j → (match_result i j ∨ match_result j i))
  (h2 : ∀ i : Fin n, ∃ j : Fin n, match_result i j)
:
  ∃ (A B C : Fin n), match_result A B ∧ match_result B C ∧ match_result C A :=
by
  sorry

end tournament_trio_l28_28384


namespace solve_system_of_equations_l28_28086

theorem solve_system_of_equations : ∃ (x y : ℝ), 4 * x + y = 6 ∧ 3 * x - y = 1 ∧ x = 1 ∧ y = 2 :=
by
  existsi (1 : ℝ)
  existsi (2 : ℝ)
  sorry

end solve_system_of_equations_l28_28086


namespace find_a3_minus_b3_l28_28935

theorem find_a3_minus_b3 (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 47) : a^3 - b^3 = 322 :=
by
  sorry

end find_a3_minus_b3_l28_28935


namespace triangle_perimeter_l28_28254

theorem triangle_perimeter (x : ℕ) (h_odd : x % 2 = 1) (h_range : 1 < x ∧ x < 5) : 2 + 3 + x = 8 :=
by
  sorry

end triangle_perimeter_l28_28254


namespace no_solution_intervals_l28_28270

theorem no_solution_intervals (a : ℝ) :
  (a < -17 ∨ a > 0) → ¬∃ x : ℝ, 7 * |x - 4 * a| + |x - a^2| + 6 * x - 3 * a = 0 :=
by
  sorry

end no_solution_intervals_l28_28270


namespace numbers_not_as_difference_of_squares_l28_28150

theorem numbers_not_as_difference_of_squares :
  {n : ℕ | ¬ ∃ x y : ℕ, x^2 - y^2 = n} = {1, 4} ∪ {4*k + 2 | k : ℕ} :=
by sorry

end numbers_not_as_difference_of_squares_l28_28150


namespace meena_cookies_left_l28_28210

-- Define the given conditions in terms of Lean definitions
def total_cookies_baked := 5 * 12
def cookies_sold_to_stone := 2 * 12
def cookies_bought_by_brock := 7
def cookies_bought_by_katy := 2 * cookies_bought_by_brock

-- Define the total cookies sold
def total_cookies_sold := cookies_sold_to_stone + cookies_bought_by_brock + cookies_bought_by_katy

-- Define the number of cookies left
def cookies_left := total_cookies_baked - total_cookies_sold

-- Prove that the number of cookies left is 15
theorem meena_cookies_left : cookies_left = 15 := by
  -- The proof is omitted (sorry is used to skip proof)
  sorry

end meena_cookies_left_l28_28210


namespace range_of_a_l28_28205

open Set

def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2 * x - 8 > 0

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, p a x → q x) →
  ({ x : ℝ | p a x } ⊆ { x : ℝ | q x }) →
  a ≤ -4 ∨ a ≥ 2 ∨ a = 0 :=
by
  sorry

end range_of_a_l28_28205


namespace chair_arrangements_48_l28_28648

theorem chair_arrangements_48 :
  ∃ (n : ℕ), n = 8 ∧ (∀ (r c : ℕ), r * c = 48 → 2 ≤ r ∧ 2 ≤ c) := 
sorry

end chair_arrangements_48_l28_28648


namespace remainder_sum_div_11_l28_28683

theorem remainder_sum_div_11 :
  ((100001 + 100002 + 100003 + 100004 + 100005 + 100006 + 100007 + 100008 + 100009 + 100010) % 11) = 10 :=
by
  sorry

end remainder_sum_div_11_l28_28683


namespace quadratic_transformation_l28_28532

theorem quadratic_transformation (y m n : ℝ) 
  (h1 : 2 * y^2 - 2 = 4 * y) 
  (h2 : (y - m)^2 = n) : 
  (m - n)^2023 = -1 := 
  sorry

end quadratic_transformation_l28_28532


namespace hotel_assignment_l28_28529

noncomputable def numberOfWaysToAssignFriends (rooms friends : ℕ) : ℕ :=
  if rooms = 5 ∧ friends = 6 then 7200 else 0

theorem hotel_assignment : numberOfWaysToAssignFriends 5 6 = 7200 :=
by 
  -- This is the condition already matched in the noncomputable function defined above.
  sorry

end hotel_assignment_l28_28529


namespace expected_total_cost_of_removing_blocks_l28_28960

/-- 
  There are six blocks in a row labeled 1 through 6, each with weight 1.
  Two blocks x ≤ y are connected if for all x ≤ z ≤ y, block z has not been removed.
  While there is at least one block remaining, a block is chosen uniformly at random and removed.
  The cost of removing a block is the sum of the weights of the blocks that are connected to it.
  Prove that the expected total cost of removing all blocks is 163 / 10.
-/
theorem expected_total_cost_of_removing_blocks : (6:ℚ) + 5 + 8/3 + 3/2 + 4/5 + 1/3 = 163 / 10 := sorry

end expected_total_cost_of_removing_blocks_l28_28960


namespace triangle_perimeter_sqrt_l28_28969

theorem triangle_perimeter_sqrt :
  let a := Real.sqrt 8
  let b := Real.sqrt 18
  let c := Real.sqrt 32
  a + b + c = 9 * Real.sqrt 2 :=
by
  sorry

end triangle_perimeter_sqrt_l28_28969


namespace traveling_zoo_l28_28762

theorem traveling_zoo (x y : ℕ) (h1 : x + y = 36) (h2 : 4 * x + 6 * y = 100) : x = 14 ∧ y = 22 :=
by {
  sorry
}

end traveling_zoo_l28_28762


namespace water_fee_relationship_xiao_qiangs_water_usage_l28_28684

variable (x y : ℝ)
variable (H1 : x > 10)
variable (H2 : y = 3 * x - 8)

theorem water_fee_relationship : y = 3 * x - 8 := 
  by 
    exact H2

theorem xiao_qiangs_water_usage : y = 67 → x = 25 :=
  by
    intro H
    have H_eq : 67 = 3 * x - 8 := by 
      rw [←H2, H]
    linarith

end water_fee_relationship_xiao_qiangs_water_usage_l28_28684


namespace ratio_of_Y_share_l28_28527

theorem ratio_of_Y_share (total_profit share_diff X_share Y_share : ℝ) 
(h1 : total_profit = 700) (h2 : share_diff = 140) 
(h3 : X_share + Y_share = 700) (h4 : X_share - Y_share = 140) : 
Y_share / total_profit = 2 / 5 :=
sorry

end ratio_of_Y_share_l28_28527


namespace real_number_value_of_m_pure_imaginary_value_of_m_l28_28898

def is_real (z : ℂ) : Prop := z.im = 0
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem real_number_value_of_m (m : ℝ) : 
  is_real ((m^2 + 2 * m - 8) + (m^2 - 2 * m) * I) ↔ (m = 0 ∨ m = 2) := 
by sorry

theorem pure_imaginary_value_of_m (m : ℝ) : 
  is_pure_imaginary ((m^2 + 2 * m - 8) + (m^2 - 2 * m) * I) ↔ (m = -4) := 
by sorry

end real_number_value_of_m_pure_imaginary_value_of_m_l28_28898


namespace simplify_expression_l28_28434

variable (m n : ℝ)

theorem simplify_expression : -2 * (m - n) = -2 * m + 2 * n := by
  sorry

end simplify_expression_l28_28434


namespace count_two_digit_integers_congruent_to_2_mod_4_l28_28141

theorem count_two_digit_integers_congruent_to_2_mod_4 : 
  ∃ n : ℕ, (∀ x : ℕ, 10 ≤ x ∧ x ≤ 99 → x % 4 = 2 → x = 4 * k + 2) ∧ n = 23 := 
by
  sorry

end count_two_digit_integers_congruent_to_2_mod_4_l28_28141


namespace intersection_point_exists_correct_line_l28_28387

noncomputable def line1 (x y : ℝ) : Prop := 2 * x - 3 * y + 2 = 0
noncomputable def line2 (x y : ℝ) : Prop := 3 * x - 4 * y - 2 = 0
noncomputable def parallel_line (x y : ℝ) : Prop := 4 * x - 2 * y + 7 = 0
noncomputable def target_line (x y : ℝ) : Prop := 2 * x - y - 18 = 0

theorem intersection_point_exists (x y : ℝ) : line1 x y ∧ line2 x y → (x = 14 ∧ y = 10) := 
by sorry

theorem correct_line (x y : ℝ) : 
  (∃ (x y : ℝ), line1 x y ∧ line2 x y) ∧ parallel_line x y 
  → target_line x y :=
by sorry

end intersection_point_exists_correct_line_l28_28387


namespace tank_capacity_is_780_l28_28613

noncomputable def tank_capacity : ℕ := 
  let fill_rate_A := 40
  let fill_rate_B := 30
  let drain_rate_C := 20
  let cycle_minutes := 3
  let total_minutes := 48
  let net_fill_per_cycle := fill_rate_A + fill_rate_B - drain_rate_C
  let total_cycles := total_minutes / cycle_minutes
  let total_fill := total_cycles * net_fill_per_cycle
  let final_capacity := total_fill - drain_rate_C -- Adjust for the last minute where C opens
  final_capacity

theorem tank_capacity_is_780 : tank_capacity = 780 := by
  unfold tank_capacity
  -- Proof steps to be filled in
  sorry

end tank_capacity_is_780_l28_28613


namespace class_scores_mean_l28_28538

theorem class_scores_mean 
  (F S : ℕ) (Rf Rs : ℚ)
  (hF : F = 90)
  (hS : S = 75)
  (hRatio : Rf / Rs = 2 / 3) :
  (F * (2/3 * Rs) + S * Rs) / (2/3 * Rs + Rs) = 81 := by
    sorry

end class_scores_mean_l28_28538


namespace batteries_on_flashlights_l28_28179

variable (b_flashlights b_toys b_controllers b_total : ℕ)

theorem batteries_on_flashlights :
  b_toys = 15 → 
  b_controllers = 2 → 
  b_total = 19 → 
  b_total = b_flashlights + b_toys + b_controllers → 
  b_flashlights = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end batteries_on_flashlights_l28_28179


namespace john_climbs_9_flights_l28_28645

variable (fl : Real := 10)  -- Each flight of stairs is 10 feet
variable (step_height_inches : Real := 18)  -- Each step is 18 inches
variable (steps : Nat := 60)  -- John climbs 60 steps

theorem john_climbs_9_flights :
  (steps * (step_height_inches / 12) / fl = 9) :=
by
  sorry

end john_climbs_9_flights_l28_28645


namespace ceil_square_range_count_l28_28154

theorem ceil_square_range_count (x : ℝ) (h : ⌈x⌉ = 12) : 
  ∃ n : ℕ, n = 23 ∧ (∀ y : ℝ, 11 < y ∧ y ≤ 12 → ⌈y^2⌉ = n) := 
sorry

end ceil_square_range_count_l28_28154


namespace solve_inequality_l28_28561

theorem solve_inequality (x : ℝ) : (x^2 + 7 * x < 8) ↔ x ∈ (Set.Ioo (-8 : ℝ) 1) := by
  sorry

end solve_inequality_l28_28561


namespace circumcircle_equation_l28_28515

theorem circumcircle_equation :
  ∃ (a b r : ℝ), 
    (∀ {x y : ℝ}, (x, y) = (2, 2) → (x - a)^2 + (y - b)^2 = r^2) ∧
    (∀ {x y : ℝ}, (x, y) = (5, 3) → (x - a)^2 + (y - b)^2 = r^2) ∧
    (∀ {x y : ℝ}, (x, y) = (3, -1) → (x - a)^2 + (y - b)^2 = r^2) ∧
    ((x - 4)^2 + (y - 1)^2 = 5) :=
sorry

end circumcircle_equation_l28_28515


namespace find_nat_numbers_for_divisibility_l28_28936

theorem find_nat_numbers_for_divisibility :
  ∃ (a b : ℕ), (7^3 ∣ a^2 + a * b + b^2) ∧ (¬ 7 ∣ a) ∧ (¬ 7 ∣ b) ∧ (a = 1) ∧ (b = 18) := by
  sorry

end find_nat_numbers_for_divisibility_l28_28936


namespace line_intersects_parabola_at_one_point_l28_28987

theorem line_intersects_parabola_at_one_point (k : ℝ) :
    (∃ y : ℝ, x = -3 * y^2 - 4 * y + 7) ↔ (x = k) := by
  sorry

end line_intersects_parabola_at_one_point_l28_28987


namespace negation_of_proposition_l28_28215

theorem negation_of_proposition :
  (¬∃ x₀ ∈ Set.Ioo 0 (π/2), Real.cos x₀ > Real.sin x₀) ↔ ∀ x ∈ Set.Ioo 0 (π / 2), Real.cos x ≤ Real.sin x :=
by
  sorry

end negation_of_proposition_l28_28215


namespace percentage_of_acid_in_original_mixture_l28_28484

theorem percentage_of_acid_in_original_mixture
  (a w : ℚ)
  (h1 : a / (a + w + 2) = 18 / 100)
  (h2 : (a + 2) / (a + w + 4) = 30 / 100) :
  (a / (a + w)) * 100 = 29 := 
sorry

end percentage_of_acid_in_original_mixture_l28_28484


namespace rhombus_diagonal_length_l28_28134

theorem rhombus_diagonal_length
  (d1 d2 A : ℝ)
  (h1 : d1 = 20)
  (h2 : A = 250)
  (h3 : A = (d1 * d2) / 2) :
  d2 = 25 :=
by
  sorry

end rhombus_diagonal_length_l28_28134


namespace counterexample_to_proposition_l28_28119

theorem counterexample_to_proposition (x y : ℤ) (h1 : x = -1) (h2 : y = -2) : x > y ∧ ¬ (x^2 > y^2) := by
  sorry

end counterexample_to_proposition_l28_28119


namespace mean_of_remaining_two_numbers_l28_28919

/-- 
Given seven numbers:
a = 1870, b = 1995, c = 2020, d = 2026, e = 2110, f = 2124, g = 2500
and the condition that the mean of five of these numbers is 2100,
prove that the mean of the remaining two numbers is 2072.5.
-/
theorem mean_of_remaining_two_numbers :
  let a := 1870
  let b := 1995
  let c := 2020
  let d := 2026
  let e := 2110
  let f := 2124
  let g := 2500
  a + b + c + d + e + f + g = 14645 →
  (a + b + c + d + e + f + g) = 14645 →
  (a + b + c + d + e) / 5 = 2100 →
  (f + g) / 2 = 2072.5 :=
by
  let a := 1870
  let b := 1995
  let c := 2020
  let d := 2026
  let e := 2110
  let f := 2124
  let g := 2500
  sorry

end mean_of_remaining_two_numbers_l28_28919


namespace speedster_convertibles_count_l28_28907

-- Definitions of conditions
def total_inventory (T : ℕ) : Prop := (T / 3) = 60
def number_of_speedsters (T S : ℕ) : Prop := S = (2 / 3) * T
def number_of_convertibles (S C : ℕ) : Prop := C = (4 / 5) * S

-- Primary statement to prove
theorem speedster_convertibles_count (T S C : ℕ) (h1 : total_inventory T) (h2 : number_of_speedsters T S) (h3 : number_of_convertibles S C) : C = 96 :=
by
  -- Conditions and given values are defined
  sorry

end speedster_convertibles_count_l28_28907


namespace largest_fraction_l28_28519

theorem largest_fraction :
  let f1 := (2 : ℚ) / 3
  let f2 := (3 : ℚ) / 4
  let f3 := (2 : ℚ) / 5
  let f4 := (11 : ℚ) / 15
  f2 > f1 ∧ f2 > f3 ∧ f2 > f4 :=
by
  sorry

end largest_fraction_l28_28519


namespace f_neg_expression_l28_28657

noncomputable def f : ℝ → ℝ :=
  λ x => if x > 0 then x^2 - 2*x + 3 else sorry

-- Define f by cases: for x > 0 and use the property of odd functions to conclude the expression for x < 0.

theorem f_neg_expression (x : ℝ) (h : x < 0) : f x = -x^2 - 2*x - 3 :=
by
  sorry

end f_neg_expression_l28_28657


namespace distance_between_points_l28_28579

theorem distance_between_points (A B : ℝ) (dA : |A| = 2) (dB : |B| = 7) : |A - B| = 5 ∨ |A - B| = 9 := 
by
  sorry

end distance_between_points_l28_28579


namespace inverse_geometric_sequence_l28_28580

-- Define that a, b, c form a geometric sequence
def geometric_sequence (a b c : ℝ) := b^2 = a * c

-- Define the theorem: if b^2 = a * c, then a, b, c form a geometric sequence
theorem inverse_geometric_sequence (a b c : ℝ) (h : b^2 = a * c) : geometric_sequence a b c :=
by
  sorry

end inverse_geometric_sequence_l28_28580


namespace compute_fraction_eq_2410_l28_28500

theorem compute_fraction_eq_2410 (x : ℕ) (hx : x = 7) : 
  (x^8 + 18 * x^4 + 81) / (x^4 + 9) = 2410 := 
by
  -- proof steps go here
  sorry

end compute_fraction_eq_2410_l28_28500


namespace percent_of_x_l28_28457

theorem percent_of_x
  (x y z : ℝ)
  (h1 : 0.45 * z = 1.20 * y)
  (h2 : z = 2 * x) :
  y = 0.75 * x :=
sorry

end percent_of_x_l28_28457


namespace reduce_to_original_l28_28350

theorem reduce_to_original (x : ℝ) (factor : ℝ) (original : ℝ) :
  original = x → factor = 1/1000 → x * factor = 0.0169 :=
by
  intros h1 h2
  sorry

end reduce_to_original_l28_28350


namespace total_treats_value_l28_28336

noncomputable def hotel_per_night := 4000
noncomputable def nights := 2
noncomputable def car_value := 30000
noncomputable def house_value := 4 * car_value
noncomputable def total_value := hotel_per_night * nights + car_value + house_value

theorem total_treats_value : total_value = 158000 :=
by
  sorry

end total_treats_value_l28_28336


namespace max_g_value_l28_28803

def g : Nat → Nat
| n => if n < 15 then n + 15 else g (n - 6)

theorem max_g_value : ∀ n, g n ≤ 29 := by
  sorry

end max_g_value_l28_28803


namespace number_of_classes_l28_28616

theorem number_of_classes (max_val : ℕ) (min_val : ℕ) (class_interval : ℕ) (range : ℕ) (num_classes : ℕ) :
  max_val = 169 → min_val = 143 → class_interval = 3 → range = max_val - min_val → num_classes = (range + 2) / class_interval + 1 :=
sorry

end number_of_classes_l28_28616


namespace inequality_solution_l28_28083

theorem inequality_solution 
  (a b c d e f : ℕ) 
  (h1 : a * d * f > b * c * f)
  (h2 : c * f * b > d * e * b) 
  (h3 : a * f - b * e = 1) 
  : d ≥ b + f := by
  -- Proof goes here
  sorry

end inequality_solution_l28_28083


namespace parabola_point_distance_condition_l28_28117

theorem parabola_point_distance_condition (k : ℝ) (p : ℝ) (h_p_gt_0 : p > 0) (focus : ℝ × ℝ) (vertex : ℝ × ℝ) :
  vertex = (0, 0) → focus = (0, p/2) → (k^2 = -2 * p * (-2)) → dist (k, -2) focus = 4 → k = 4 ∨ k = -4 :=
by
  sorry

end parabola_point_distance_condition_l28_28117


namespace percentage_increase_Sakshi_Tanya_l28_28633

def efficiency_Sakshi : ℚ := 1 / 5
def efficiency_Tanya : ℚ := 1 / 4
def percentage_increase_in_efficiency (eff_Sakshi eff_Tanya : ℚ) : ℚ :=
  ((eff_Tanya - eff_Sakshi) / eff_Sakshi) * 100

theorem percentage_increase_Sakshi_Tanya :
  percentage_increase_in_efficiency efficiency_Sakshi efficiency_Tanya = 25 :=
by
  sorry

end percentage_increase_Sakshi_Tanya_l28_28633


namespace peak_valley_usage_l28_28630

-- Define the electricity rate constants
def normal_rate : ℝ := 0.5380
def peak_rate : ℝ := 0.5680
def valley_rate : ℝ := 0.2880

-- Define the total consumption and the savings
def total_consumption : ℝ := 200
def savings : ℝ := 16.4

-- Define the theorem to prove the peak and off-peak usage
theorem peak_valley_usage :
  ∃ (x y : ℝ), x + y = total_consumption ∧ peak_rate * x + valley_rate * y = total_consumption * normal_rate - savings ∧ x = 120 ∧ y = 80 :=
by
  sorry

end peak_valley_usage_l28_28630


namespace problem_statement_l28_28485

variable {x y : ℤ}

def is_multiple_of_5 (n : ℤ) : Prop := ∃ m : ℤ, n = 5 * m
def is_multiple_of_10 (n : ℤ) : Prop := ∃ m : ℤ, n = 10 * m

theorem problem_statement (hx : is_multiple_of_5 x) (hy : is_multiple_of_10 y) :
  (is_multiple_of_5 (x + y)) ∧ (x + y ≥ 15) :=
sorry

end problem_statement_l28_28485


namespace no_rectangle_from_five_distinct_squares_l28_28339

theorem no_rectangle_from_five_distinct_squares (q1 q2 q3 q4 q5 : ℝ) 
  (h1 : q1 < q2) 
  (h2 : q2 < q3) 
  (h3 : q3 < q4) 
  (h4 : q4 < q5) : 
  ¬∃(a b: ℝ), a * b = 5 ∧ a = q1 + q2 + q3 + q4 + q5 := sorry

end no_rectangle_from_five_distinct_squares_l28_28339


namespace even_function_f_l28_28394

-- Problem statement:
-- Given that f is an even function and that for x < 0, f(x) = x^2 - 1/x,
-- prove that f(1) = 2.

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then x^2 - 1/x else 0

theorem even_function_f {f : ℝ → ℝ} (h_even : ∀ x, f x = f (-x))
  (h_neg_def : ∀ x, x < 0 → f x = x^2 - 1/x) : f 1 = 2 :=
by
  -- Proof body (to be completed)
  sorry

end even_function_f_l28_28394


namespace number_of_hens_l28_28759

theorem number_of_hens (H C : ℕ) (h1 : H + C = 44) (h2 : 2 * H + 4 * C = 128) : H = 24 :=
by
  sorry

end number_of_hens_l28_28759


namespace part1_part2_l28_28795

/-- Part (1) -/
theorem part1 (a : ℝ) (p : ∀ x : ℝ, x^2 - a*x + 4 > 0) (q : ∀ x y : ℝ, (0 < x ∧ x < y) → x^a < y^a) : 
  0 < a ∧ a < 4 :=
sorry

/-- Part (2) -/
theorem part2 (a : ℝ) (p_iff: ∀ x : ℝ, x^2 - a*x + 4 > 0 ↔ -4 < a ∧ a < 4)
  (q_iff: ∀ x y : ℝ, (0 < x ∧ x < y) ↔ x^a < y^a ∧ a > 0) (hp : ∃ x : ℝ, ¬(x^2 - a*x + 4 > 0))
  (hq : ∀ x y : ℝ, (x^a < y^a) → (0 < x ∧ x < y)) : 
  (a >= 4) ∨ (-4 < a ∧ a <= 0) :=
sorry

end part1_part2_l28_28795


namespace platform_length_l28_28009

theorem platform_length (train_length : ℕ) (tree_cross_time : ℕ) (platform_cross_time : ℕ) (platform_length : ℕ)
  (h_train_length : train_length = 1200)
  (h_tree_cross_time : tree_cross_time = 120)
  (h_platform_cross_time : platform_cross_time = 160)
  (h_speed_calculation : (train_length / tree_cross_time = 10))
  : (train_length + platform_length) / 10 = platform_cross_time → platform_length = 400 :=
sorry

end platform_length_l28_28009


namespace inequality_proof_l28_28558

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 / (2 * x) + 1 / (2 * y) + 1 / (2 * z)) > 
  (1 / (y + z) + 1 / (z + x) + 1 / (x + y)) :=
  by
    let a := y + z
    let b := z + x
    let c := x + y
    have x_def : x = (a + c - b) / 2 := sorry
    have y_def : y = (a + b - c) / 2 := sorry
    have z_def : z = (b + c - a) / 2 := sorry
    sorry

end inequality_proof_l28_28558


namespace some_number_value_l28_28351

theorem some_number_value (some_number : ℝ): 
  (∀ n : ℝ, (n / some_number) * (n / 80) = 1 → n = 40) → some_number = 80 :=
by
  sorry

end some_number_value_l28_28351


namespace max_value_f_on_interval_l28_28678

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem max_value_f_on_interval : 
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, (∀ y ∈ Set.Icc (-1 : ℝ) 1, f y ≤ f x) ∧ f x = Real.exp 1 - 1 :=
sorry

end max_value_f_on_interval_l28_28678


namespace area_of_roof_l28_28332

-- Definitions and conditions
def length (w : ℝ) := 4 * w
def difference_eq (l w : ℝ) := l - w = 39
def area (l w : ℝ) := l * w

-- Theorem statement
theorem area_of_roof (w l : ℝ) (h_length : l = length w) (h_diff : difference_eq l w) : area l w = 676 :=
by
  sorry

end area_of_roof_l28_28332


namespace marble_distribution_l28_28786

theorem marble_distribution (x : ℝ) (h : 49 = (3 * x + 2) + (x + 1) + (2 * x - 1) + x) :
  (3 * x + 2 = 22) ∧ (x + 1 = 8) ∧ (2 * x - 1 = 12) ∧ (x = 7) :=
by
  sorry

end marble_distribution_l28_28786


namespace negation_of_universal_l28_28750

theorem negation_of_universal (P : ∀ x : ℝ, x^2 > 0) : ¬ ( ∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 :=
by 
  sorry

end negation_of_universal_l28_28750


namespace no_nat_triplet_square_l28_28756

theorem no_nat_triplet_square (m n k : ℕ) : ¬ (∃ a b c : ℕ, m^2 + n + k = a^2 ∧ n^2 + k + m = b^2 ∧ k^2 + m + n = c^2) :=
by sorry

end no_nat_triplet_square_l28_28756


namespace evaluate_expression_l28_28449

theorem evaluate_expression : 6 - 8 * (9 - 4^2) * 5 = 286 := by
  sorry

end evaluate_expression_l28_28449


namespace find_second_cert_interest_rate_l28_28726

theorem find_second_cert_interest_rate
  (initial_investment : ℝ := 12000)
  (first_term_months : ℕ := 8)
  (first_interest_rate : ℝ := 8 / 100)
  (second_term_months : ℕ := 10)
  (final_amount : ℝ := 13058.40)
  : ∃ s : ℝ, (s = 3.984) := sorry

end find_second_cert_interest_rate_l28_28726


namespace maximum_value_expression_l28_28591

theorem maximum_value_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 = 9) : 
  (a - b)^2 + (b - c)^2 + (c - a)^2 ≤ 27 :=
sorry

end maximum_value_expression_l28_28591


namespace more_tvs_sold_l28_28843

variable (T x : ℕ)

theorem more_tvs_sold (h1 : T + x = 327) (h2 : T + 3 * x = 477) : x = 75 := by
  sorry

end more_tvs_sold_l28_28843


namespace total_potatoes_l28_28072

theorem total_potatoes (cooked_potatoes : ℕ) (time_per_potato : ℕ) (remaining_time : ℕ) (H1 : cooked_potatoes = 7) (H2 : time_per_potato = 5) (H3 : remaining_time = 45) : (cooked_potatoes + (remaining_time / time_per_potato) = 16) :=
by
  sorry

end total_potatoes_l28_28072


namespace test_two_categorical_features_l28_28291

-- Definitions based on the problem conditions
def is_testing_method (method : String) : Prop :=
  method = "Three-dimensional bar chart" ∨
  method = "Two-dimensional bar chart" ∨
  method = "Contour bar chart" ∨
  method = "Independence test"

noncomputable def correct_method : String :=
  "Independence test"

-- Theorem statement based on the problem and solution
theorem test_two_categorical_features :
  ∀ m : String, is_testing_method m → m = correct_method :=
by
  sorry

end test_two_categorical_features_l28_28291


namespace smallest_n_l28_28046

theorem smallest_n 
    (h1 : ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * r = b ∧ b * r = c ∧ 7 * n + 1 = a + b + c)
    (h2 : ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x * s = y ∧ y * s = z ∧ 8 * n + 1 = x + y + z) :
    n = 22 :=
sorry

end smallest_n_l28_28046


namespace a_investment_l28_28569

theorem a_investment (B C total_profit A_share: ℝ) (hB: B = 7200) (hC: C = 9600) (htotal_profit: total_profit = 9000) 
  (hA_share: A_share = 1125) : ∃ x : ℝ, (A_share / total_profit) = (x / (x + B + C)) ∧ x = 2400 := 
by
  use 2400
  sorry

end a_investment_l28_28569


namespace sum_of_roots_l28_28079

theorem sum_of_roots (b : ℝ) (x : ℝ) (y : ℝ) :
  (x^2 - b * x + 20 = 0) ∧ (y^2 - b * y + 20 = 0) ∧ (x * y = 20) -> (x + y = b) := 
by
  sorry

end sum_of_roots_l28_28079


namespace partial_fraction_sum_zero_l28_28893

variable {A B C D E : ℝ}
variable {x : ℝ}

theorem partial_fraction_sum_zero (h : 
  (1:ℝ) / ((x-1)*x*(x+1)*(x+2)*(x+3)) = 
  A / (x-1) + B / x + C / (x+1) + D / (x+2) + E / (x+3)) : 
  A + B + C + D + E = 0 :=
by sorry

end partial_fraction_sum_zero_l28_28893


namespace total_surface_area_of_cube_l28_28642

theorem total_surface_area_of_cube : 
  ∀ (s : Real), 
  (12 * s = 36) → 
  (s * Real.sqrt 3 = 3 * Real.sqrt 3) → 
  6 * s^2 = 54 := 
by
  intros s h1 h2
  sorry

end total_surface_area_of_cube_l28_28642


namespace rectangle_dimensions_l28_28816

theorem rectangle_dimensions (x y : ℝ) (h1 : y = 2 * x) (h2 : 2 * (x + y) = 2 * (x * y)) :
  (x = 3 / 2) ∧ (y = 3) := by
  sorry

end rectangle_dimensions_l28_28816


namespace rhombus_side_length_l28_28703

/-
  Define the length of the rhombus diagonal and the area of the rhombus.
-/
def diagonal1 : ℝ := 20
def area : ℝ := 480

/-
  The theorem states that given these conditions, the length of each side of the rhombus is 26 m.
-/
theorem rhombus_side_length (d1 d2 : ℝ) (A : ℝ) (h1 : d1 = diagonal1) (h2 : A = area):
  2 * 26 * 26 * 2 = A * 2 * 2 + (d1 / 2) * (d1 / 2) :=
sorry

end rhombus_side_length_l28_28703


namespace add_least_number_l28_28191

theorem add_least_number (n : ℕ) (h1 : n = 1789) (h2 : ∃ k : ℕ, 5 * k = n + 11) (h3 : ∃ j : ℕ, 6 * j = n + 11) (h4 : ∃ m : ℕ, 4 * m = n + 11) (h5 : ∃ l : ℕ, 11 * l = n + 11) : 11 = 11 :=
by
  sorry

end add_least_number_l28_28191


namespace two_and_four_digit_singular_numbers_six_digit_singular_number_exists_twenty_digit_singular_number_at_most_ten_singular_numbers_with_100_digits_exists_thirty_digit_singular_number_l28_28327

def is_singular_number (n : ℕ) (num : ℕ) : Prop :=
  let first_n_digits := num / 10^n;
  let last_n_digits := num % 10^n;
  (num > 0) ∧
  (first_n_digits > 0) ∧
  (last_n_digits > 0) ∧
  (first_n_digits < 10^n) ∧
  (last_n_digits < 10^n) ∧
  (num = first_n_digits * 10^n + last_n_digits) ∧
  (∃ k, num = k^2) ∧
  (∃ k, first_n_digits = k^2) ∧
  (∃ k, last_n_digits = k^2)

-- (1) Prove that 49 is a two-digit singular number and 1681 is a four-digit singular number
theorem two_and_four_digit_singular_numbers :
  is_singular_number 1 49 ∧ is_singular_number 2 1681 :=
sorry

-- (2) Prove that 256036 is a six-digit singular number
theorem six_digit_singular_number :
  is_singular_number 3 256036 :=
sorry

-- (3) Prove the existence of a 20-digit singular number
theorem exists_twenty_digit_singular_number :
  ∃ num, is_singular_number 10 num :=
sorry

-- (4) Prove that there are at most 10 singular numbers with 100 digits
theorem at_most_ten_singular_numbers_with_100_digits :
  ∃! n, n <= 10 ∧ ∀ num, num < 10^100 → is_singular_number 50 num → num < 10 ∧ num > 0 :=
sorry

-- (5) Prove the existence of a 30-digit singular number
theorem exists_thirty_digit_singular_number :
  ∃ num, is_singular_number 15 num :=
sorry

end two_and_four_digit_singular_numbers_six_digit_singular_number_exists_twenty_digit_singular_number_at_most_ten_singular_numbers_with_100_digits_exists_thirty_digit_singular_number_l28_28327


namespace train_length_l28_28161

variable (L V : ℝ)

-- Given conditions
def condition1 : Prop := V = L / 24
def condition2 : Prop := V = (L + 650) / 89

theorem train_length : condition1 L V → condition2 L V → L = 240 := by
  intro h1 h2
  sorry

end train_length_l28_28161


namespace quadratic_inequality_solution_range_l28_28564

theorem quadratic_inequality_solution_range (m : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 + m * x + 2 > 0) ↔ m > -3 := 
sorry

end quadratic_inequality_solution_range_l28_28564


namespace fraction_subtraction_l28_28780

theorem fraction_subtraction (x y : ℝ) (h : x / y = 3 / 2) : (x - y) / y = 1 / 2 := 
by 
  sorry

end fraction_subtraction_l28_28780


namespace problem_solution_l28_28732

variables (a b : ℝ)
variables (h1 : a > 0) (h2 : b > 0)
variables (h3 : 3 * log 101 ((1030301 - a - b) / (3 * a * b)) = 3 - 2 * log 101 (a * b))

theorem problem_solution : 101 - (a)^(1/3) - (b)^(1/3) = 0 :=
by
  sorry

end problem_solution_l28_28732


namespace geom_seq_root_product_l28_28064

theorem geom_seq_root_product
  (a : ℕ → ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * a 1)
  (h_root1 : 3 * (a 1)^2 + 7 * a 1 - 9 = 0)
  (h_root10 : 3 * (a 10)^2 + 7 * a 10 - 9 = 0) :
  a 4 * a 7 = -3 := 
by
  sorry

end geom_seq_root_product_l28_28064


namespace nickels_eq_100_l28_28314

variables (P D N Q H DollarCoins : ℕ)

def conditions :=
  D = P + 10 ∧
  N = 2 * D ∧
  Q = 4 ∧
  P = 10 * Q ∧
  H = Q + 5 ∧
  DollarCoins = 3 * H ∧
  (P + 10 * D + 5 * N + 25 * Q + 50 * H + 100 * DollarCoins = 2000)

theorem nickels_eq_100 (h : conditions P D N Q H DollarCoins) : N = 100 :=
by {
  sorry
}

end nickels_eq_100_l28_28314


namespace pow_comparison_l28_28496

theorem pow_comparison : 2^700 > 5^300 :=
by sorry

end pow_comparison_l28_28496


namespace find_particular_number_l28_28787

variable (x : ℝ)

theorem find_particular_number (h : 0.46 + x = 0.72) : x = 0.26 :=
sorry

end find_particular_number_l28_28787


namespace solve_system_l28_28798

theorem solve_system:
  ∃ (x y : ℝ), (26 * x^2 + 42 * x * y + 17 * y^2 = 10 ∧ 10 * x^2 + 18 * x * y + 8 * y^2 = 6) ↔
  (x = -1 ∧ y = 2) ∨ (x = -11 ∧ y = 14) ∨ (x = 11 ∧ y = -14) ∨ (x = 1 ∧ y = -2) :=
by
  sorry

end solve_system_l28_28798


namespace common_difference_of_sequence_l28_28507

variable (a : ℕ → ℚ)

def is_arithmetic_sequence (a : ℕ → ℚ) :=
  ∃ d : ℚ, ∀ n m : ℕ, a n = a m + d * (n - m)

theorem common_difference_of_sequence 
  (h : a 2015 = a 2013 + 6) 
  (ha : is_arithmetic_sequence a) :
  ∃ d : ℚ, d = 3 :=
by
  sorry

end common_difference_of_sequence_l28_28507


namespace piece_length_is_111_l28_28985

-- Define the conditions
axiom condition1 : ∃ (x : ℤ), 9 * x ≤ 1000
axiom condition2 : ∃ (x : ℤ), 9 * x ≤ 1100

-- State the problem: Prove that the length of each piece is 111 centimeters
theorem piece_length_is_111 (x : ℤ) (h1 : 9 * x ≤ 1000) (h2 : 9 * x ≤ 1100) : x = 111 :=
by sorry

end piece_length_is_111_l28_28985


namespace jasmine_percent_after_addition_l28_28883

-- Variables definition based on the problem
def original_volume : ℕ := 90
def original_jasmine_percent : ℚ := 0.05
def added_jasmine : ℕ := 8
def added_water : ℕ := 2

-- Total jasmine amount calculation in original solution
def original_jasmine_amount : ℚ := original_jasmine_percent * original_volume

-- New total jasmine amount after addition
def new_jasmine_amount : ℚ := original_jasmine_amount + added_jasmine

-- New total volume calculation after addition
def new_total_volume : ℕ := original_volume + added_jasmine + added_water

-- New jasmine percent in the solution
def new_jasmine_percent : ℚ := (new_jasmine_amount / new_total_volume) * 100

-- The proof statement
theorem jasmine_percent_after_addition : new_jasmine_percent = 12.5 :=
by
  sorry

end jasmine_percent_after_addition_l28_28883


namespace min_side_length_is_isosceles_l28_28855

-- Let a denote the side length BC
-- Let b denote the side length AB
-- Let c denote the side length AC

theorem min_side_length_is_isosceles (α : ℝ) (S : ℝ) (a b c : ℝ) :
  (a^2 = b^2 + c^2 - 2 * b * c * Real.cos α ∧ S = 0.5 * b * c * Real.sin α) →
  a = Real.sqrt (((b - c)^2 + (4 * S * (1 - Real.cos α)) / Real.sin α)) →
  b = c :=
by
  intros h1 h2
  sorry

end min_side_length_is_isosceles_l28_28855


namespace seating_arrangements_l28_28126

theorem seating_arrangements (n : ℕ) (max_capacity : ℕ) 
  (h_n : n = 6) (h_max : max_capacity = 4) :
  ∃ k : ℕ, k = 50 :=
by
  sorry

end seating_arrangements_l28_28126


namespace smallest_7_heavy_three_digit_number_l28_28423

def is_7_heavy (n : ℕ) : Prop := n % 7 > 4

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

theorem smallest_7_heavy_three_digit_number :
  ∃ n : ℕ, is_three_digit n ∧ is_7_heavy n ∧ (∀ m : ℕ, is_three_digit m ∧ is_7_heavy m → n ≤ m) ∧
  n = 103 := 
by
  sorry

end smallest_7_heavy_three_digit_number_l28_28423


namespace arithmetic_sequence_geometric_sum_l28_28758

theorem arithmetic_sequence_geometric_sum (a₁ a₂ d : ℕ) (h₁ : d ≠ 0) 
    (h₂ : (2 * a₁ + d)^2 = a₁ * (4 * a₁ + 6 * d)) :
    a₂ = 3 * a₁ :=
by
  sorry

end arithmetic_sequence_geometric_sum_l28_28758


namespace cells_at_end_of_9th_day_l28_28200

def initial_cells : ℕ := 4
def split_ratio : ℕ := 3
def total_days : ℕ := 9
def days_per_split : ℕ := 3

def num_terms : ℕ := total_days / days_per_split

noncomputable def number_of_cells (initial_cells split_ratio num_terms : ℕ) : ℕ :=
  initial_cells * split_ratio ^ (num_terms - 1)

theorem cells_at_end_of_9th_day :
  number_of_cells initial_cells split_ratio num_terms = 36 :=
by
  sorry

end cells_at_end_of_9th_day_l28_28200


namespace isosceles_triangle_perimeter_l28_28277

variable (a b c : ℝ) (h_iso : a = b ∨ a = c ∨ b = c) (h_a : a = 6) (h_b : b = 6) (h_c : c = 3)
variable (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a)

theorem isosceles_triangle_perimeter : a + b + c = 15 :=
by 
  -- Given definitions and triangle inequality
  have h_valid : a = 6 ∧ b = 6 ∧ c = 3 := ⟨h_a, h_b, h_c⟩
  sorry

end isosceles_triangle_perimeter_l28_28277


namespace min_fence_length_l28_28603

theorem min_fence_length (w l F: ℝ) (h1: l = 2 * w) (h2: 2 * w^2 ≥ 500) : F = 96 :=
by sorry

end min_fence_length_l28_28603


namespace Li_Minghui_should_go_to_supermarket_B_for_300_yuan_cost_equal_for_500_yuan_l28_28982

def cost_supermarket_A (x : ℝ) : ℝ :=
  200 + 0.8 * (x - 200)

def cost_supermarket_B (x : ℝ) : ℝ :=
  100 + 0.85 * (x - 100)

theorem Li_Minghui_should_go_to_supermarket_B_for_300_yuan :
  cost_supermarket_B 300 < cost_supermarket_A 300 := by
  sorry

theorem cost_equal_for_500_yuan :
  cost_supermarket_A 500 = cost_supermarket_B 500 := by
  sorry

end Li_Minghui_should_go_to_supermarket_B_for_300_yuan_cost_equal_for_500_yuan_l28_28982


namespace Jean_money_l28_28526

theorem Jean_money (x : ℝ) (h1 : 3 * x + x = 76): 
  3 * x = 57 := 
by
  sorry

end Jean_money_l28_28526


namespace bridge_length_is_115_meters_l28_28921

noncomputable def length_of_bridge (length_of_train : ℝ) (speed_km_per_hr : ℝ) (time_to_pass : ℝ) : ℝ :=
  let speed_m_per_s := speed_km_per_hr * (1000 / 3600)
  let total_distance := speed_m_per_s * time_to_pass
  total_distance - length_of_train

theorem bridge_length_is_115_meters :
  length_of_bridge 300 35 42.68571428571429 = 115 :=
by
  -- Here the proof has to show the steps for converting speed and calculating distances
  sorry

end bridge_length_is_115_meters_l28_28921


namespace arithmetic_sequence_l28_28955

noncomputable def a_n (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

theorem arithmetic_sequence (a1 d : ℝ) (h_d : d ≠ 0) 
  (h1 : a1 + (a1 + 2 * d) = 8) 
  (h2 : (a1 + d) * (a1 + 8 * d) = (a1 + 3 * d) * (a1 + 3 * d)) :
  a_n a1 d 5 = 13 := 
by 
  sorry

end arithmetic_sequence_l28_28955


namespace julia_total_food_cost_l28_28067

-- Definitions based on conditions
def weekly_total_cost : ℕ := 30
def rabbit_weeks : ℕ := 5
def rabbit_food_cost : ℕ := 12
def parrot_weeks : ℕ := 3
def parrot_food_cost : ℕ := weekly_total_cost - rabbit_food_cost

-- Proof statement
theorem julia_total_food_cost : 
  rabbit_weeks * rabbit_food_cost + parrot_weeks * parrot_food_cost = 114 := 
by 
  sorry

end julia_total_food_cost_l28_28067


namespace donut_selection_l28_28734

-- Lean statement for the proof problem
theorem donut_selection (n k : ℕ) (h1 : n = 5) (h2 : k = 4) : (n + k - 1).choose (k - 1) = 56 :=
by
  rw [h1, h2]
  sorry

end donut_selection_l28_28734


namespace total_cost_of_trick_decks_l28_28410

theorem total_cost_of_trick_decks (cost_per_deck: ℕ) (victor_decks: ℕ) (friend_decks: ℕ) (total_spent: ℕ) : 
  cost_per_deck = 8 → victor_decks = 6 → friend_decks = 2 → total_spent = cost_per_deck * victor_decks + cost_per_deck * friend_decks → total_spent = 64 :=
by 
  sorry

end total_cost_of_trick_decks_l28_28410


namespace three_digit_sum_permutations_l28_28902

theorem three_digit_sum_permutations (a b c : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ 9) (h₃ : 1 ≤ b) (h₄ : b ≤ 9) (h₅ : 1 ≤ c) (h₆ : c ≤ 9)
  (h₇ : n = 100 * a + 10 * b + c)
  (h₈ : 222 * (a + b + c) - n = 1990) :
  n = 452 :=
by
  sorry

end three_digit_sum_permutations_l28_28902


namespace inequality_solution_l28_28961

theorem inequality_solution (x : ℝ) : 
  3 - 1 / (3 * x + 4) < 5 ↔ x < -4 / 3 ∨ -3 / 2 < x := 
sorry

end inequality_solution_l28_28961


namespace number_of_sides_l28_28399

theorem number_of_sides (P s : ℝ) (hP : P = 108) (hs : s = 12) : P / s = 9 :=
by sorry

end number_of_sides_l28_28399


namespace rice_amount_previously_l28_28918

variables (P X : ℝ) (hP : P > 0) (h : 0.8 * P * 50 = P * X)

theorem rice_amount_previously (hP : P > 0) (h : 0.8 * P * 50 = P * X) : X = 40 := 
by 
  sorry

end rice_amount_previously_l28_28918


namespace volume_of_given_tetrahedron_l28_28396

noncomputable def volume_of_tetrahedron (radius : ℝ) (total_length : ℝ) : ℝ := 
  let R := radius
  let L := total_length
  let a := (2 * Real.sqrt 33) / 3
  let V := (a^3 * Real.sqrt 2) / 12
  V

theorem volume_of_given_tetrahedron :
  volume_of_tetrahedron (Real.sqrt 22 / 2) (8 * Real.pi) = 48 := 
  sorry

end volume_of_given_tetrahedron_l28_28396


namespace value_at_2007_l28_28293

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom symmetric_property (x : ℝ) : f (2 + x) = f (2 - x)
axiom specific_value : f (-3) = -2

theorem value_at_2007 : f 2007 = -2 :=
sorry

end value_at_2007_l28_28293


namespace seconds_in_8_point_5_minutes_l28_28710

def minutesToSeconds (minutes : ℝ) : ℝ := minutes * 60

theorem seconds_in_8_point_5_minutes : minutesToSeconds 8.5 = 510 := 
by
  sorry

end seconds_in_8_point_5_minutes_l28_28710


namespace solve_inequality_l28_28811

theorem solve_inequality (a : ℝ) :
  (a > 0 → ∀ x : ℝ, (12 * x^2 - a * x - a^2 < 0 ↔ -a / 4 < x ∧ x < a / 3)) ∧
  (a = 0 → ∀ x : ℝ, ¬ (12 * x^2 - a * x - a^2 < 0)) ∧ 
  (a < 0 → ∀ x : ℝ, (12 * x^2 - a * x - a^2 < 0 ↔ a / 3 < x ∧ x < -a / 4)) :=
by
  sorry

end solve_inequality_l28_28811


namespace abs_nested_expression_l28_28252

theorem abs_nested_expression : 
  abs (abs (-abs (-2 + 3) - 2) + 2) = 5 :=
by
  sorry

end abs_nested_expression_l28_28252


namespace john_reading_time_l28_28593

theorem john_reading_time:
  let weekday_hours_moses := 1.5
  let weekday_rate_moses := 30
  let saturday_hours_moses := 2
  let saturday_rate_moses := 40
  let pages_moses := 450
  let weekday_hours_rest := 1.5
  let weekday_rate_rest := 45
  let saturday_hours_rest := 2.5
  let saturday_rate_rest := 60
  let pages_rest := 2350
  let weekdays_per_week := 5
  let saturdays_per_week := 1
  let total_pages_per_week_moses := (weekday_hours_moses * weekday_rate_moses * weekdays_per_week) + 
                                    (saturday_hours_moses * saturday_rate_moses * saturdays_per_week)
  let total_pages_per_week_rest := (weekday_hours_rest * weekday_rate_rest * weekdays_per_week) + 
                                   (saturday_hours_rest * saturday_rate_rest * saturdays_per_week)
  let weeks_moses := (pages_moses / total_pages_per_week_moses).ceil
  let weeks_rest := (pages_rest / total_pages_per_week_rest).ceil
  let total_weeks := weeks_moses + weeks_rest
  total_weeks = 7 :=
by
  -- placeholders for the proof steps.
  sorry

end john_reading_time_l28_28593


namespace max_d_77733e_divisible_by_33_l28_28034

open Int

theorem max_d_77733e_divisible_by_33 : ∃ d e : ℕ, 
  (7 * 100000 + d * 10000 + 7 * 1000 + 3 * 100 + 3 * 10 + e) % 33 = 0 ∧ 
  (d ≤ 9) ∧ (e ≤ 9) ∧ 
  (∀ d' e', ((7 * 100000 + d' * 10000 + 7 * 1000 + 3 * 100 + 3 * 10 + e') % 33 = 0 ∧ d' ≤ 9 ∧ e' ≤ 9 → d' ≤ d)) 
  := ⟨6, 0, by sorry⟩

end max_d_77733e_divisible_by_33_l28_28034


namespace p_minus_q_l28_28695

-- Define the given equation as a predicate.
def eqn (x : ℝ) : Prop := (3*x - 9) / (x*x + 3*x - 18) = x + 3

-- Define the values p and q as distinct solutions.
def p_and_q (p q : ℝ) : Prop := eqn p ∧ eqn q ∧ p ≠ q ∧ p > q

theorem p_minus_q {p q : ℝ} (h : p_and_q p q) : p - q = 2 := sorry

end p_minus_q_l28_28695


namespace largest_three_digit_int_l28_28249

theorem largest_three_digit_int (n : ℕ) (h1 : 100 ≤ n ∧ n ≤ 999) (h2 : 75 * n ≡ 225 [MOD 300]) : n = 999 :=
sorry

end largest_three_digit_int_l28_28249


namespace george_boxes_l28_28193

-- Define the problem conditions and the question's expected outcome.
def total_blocks : ℕ := 12
def blocks_per_box : ℕ := 6
def expected_num_boxes : ℕ := 2

-- The proof statement that needs to be proved: George has the expected number of boxes.
theorem george_boxes : total_blocks / blocks_per_box = expected_num_boxes := 
  sorry

end george_boxes_l28_28193


namespace square_tiles_count_l28_28186

theorem square_tiles_count 
  (h s : ℕ)
  (total_tiles : h + s = 30)
  (total_edges : 6 * h + 4 * s = 128) : 
  s = 26 :=
by
  sorry

end square_tiles_count_l28_28186


namespace problem_solution_l28_28524

theorem problem_solution : (3127 - 2972) ^ 3 / 343 = 125 := by
  sorry

end problem_solution_l28_28524


namespace intersection_A_B_l28_28894

def A : Set ℤ := {x | abs x < 2}
def B : Set ℤ := {-1, 0, 1, 2, 3}

theorem intersection_A_B : A ∩ B = {-1, 0, 1} := by
  sorry

end intersection_A_B_l28_28894


namespace arcsin_range_l28_28956

theorem arcsin_range (α : ℝ ) (x : ℝ ) (h₁ : x = Real.cos α) (h₂ : -Real.pi / 4 ≤ α ∧ α ≤ 3 * Real.pi / 4) : 
-Real.pi / 4 ≤ Real.arcsin x ∧ Real.arcsin x ≤ Real.pi / 2 :=
sorry

end arcsin_range_l28_28956


namespace total_canoes_built_l28_28901

theorem total_canoes_built (boats_jan : ℕ) (h : boats_jan = 5)
    (boats_feb : ℕ) (h1 : boats_feb = boats_jan * 3)
    (boats_mar : ℕ) (h2 : boats_mar = boats_feb * 3)
    (boats_apr : ℕ) (h3 : boats_apr = boats_mar * 3) :
  boats_jan + boats_feb + boats_mar + boats_apr = 200 :=
sorry

end total_canoes_built_l28_28901


namespace friend_spent_13_50_l28_28391

noncomputable def amount_you_spent : ℝ := 
  let x := (22 - 5) / 2
  x

noncomputable def amount_friend_spent (x : ℝ) : ℝ := 
  x + 5

theorem friend_spent_13_50 :
  ∃ x : ℝ, (x + (x + 5) = 22) ∧ (x + 5 = 13.5) :=
by
  sorry

end friend_spent_13_50_l28_28391


namespace current_short_trees_l28_28097

-- Definitions of conditions in a)
def tall_trees : ℕ := 44
def short_trees_planted : ℕ := 57
def total_short_trees_after_planting : ℕ := 98

-- Statement to prove the question == answer given conditions
theorem current_short_trees (S : ℕ) (h : S + short_trees_planted = total_short_trees_after_planting) : S = 41 :=
by
  -- Proof would go here
  sorry

end current_short_trees_l28_28097


namespace max_sum_of_inequalities_l28_28444

theorem max_sum_of_inequalities (x y : ℝ) (h1 : 4 * x + 3 * y ≤ 10) (h2 : 3 * x + 5 * y ≤ 11) :
  x + y ≤ 31 / 11 :=
sorry

end max_sum_of_inequalities_l28_28444


namespace min_integer_solution_l28_28188

theorem min_integer_solution (x : ℤ) (h1 : 3 - x > 0) (h2 : (4 * x / 3 : ℚ) + 3 / 2 > -(x / 6)) : x = 0 := by
  sorry

end min_integer_solution_l28_28188


namespace units_digit_of_m_squared_plus_3_to_the_m_l28_28824

def m : ℕ := 2021^3 + 3^2021

theorem units_digit_of_m_squared_plus_3_to_the_m 
  (hm : m = 2021^3 + 3^2021) : 
  ((m^2 + 3^m) % 10) = 7 := 
by 
  -- Here you would input the proof steps, however, we skip it now with sorry.
  sorry

end units_digit_of_m_squared_plus_3_to_the_m_l28_28824


namespace coefficient_of_x3y0_l28_28483

def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

def f (m n : ℕ) : ℕ :=
  binomial_coeff 6 m * binomial_coeff 4 n

theorem coefficient_of_x3y0 :
  f 3 0 = 20 :=
by
  sorry

end coefficient_of_x3y0_l28_28483


namespace find_rate_of_interest_l28_28094

variable (P : ℝ) (R : ℝ) (T : ℕ := 2)

-- Condition for Simple Interest (SI = Rs. 660 for 2 years)
def simple_interest :=
  P * R * ↑T / 100 = 660

-- Condition for Compound Interest (CI = Rs. 696.30 for 2 years)
def compound_interest :=
  P * ((1 + R / 100) ^ T - 1) = 696.30

-- We need to prove that R = 11
theorem find_rate_of_interest (P : ℝ) (h1 : simple_interest P R) (h2 : compound_interest P R) : 
  R = 11 := by
  sorry

end find_rate_of_interest_l28_28094


namespace distance_BC_in_circle_l28_28489

theorem distance_BC_in_circle
    (r : ℝ) (A B C : ℝ × ℝ)
    (h_radius : r = 10)
    (h_diameter : dist A B = 2 * r)
    (h_chord : dist A C = 12) :
    dist B C = 16 := by
  sorry

end distance_BC_in_circle_l28_28489


namespace maximum_area_rhombus_l28_28744

theorem maximum_area_rhombus 
    (x₀ y₀ k : ℝ)
    (h1 : 2 ≤ x₀ ∧ x₀ ≤ 4)
    (h2 : y₀ = k / x₀)
    (h3 : ∀ x > 0, ∃ y, y = k / x) :
    (∀ (x₀ : ℝ), 2 ≤ x₀ ∧ x₀ ≤ 4 → ∃ (S : ℝ), S = 3 * (Real.sqrt 2 / 2 * x₀^2) → S ≤ 24 * Real.sqrt 2) :=
by
  sorry

end maximum_area_rhombus_l28_28744


namespace derivative_sqrt_l28_28167

/-- The derivative of the function y = sqrt x is 1 / (2 * sqrt x) -/
theorem derivative_sqrt (x : ℝ) (h : 0 < x) : (deriv (fun x => Real.sqrt x) x) = 1 / (2 * Real.sqrt x) :=
sorry

end derivative_sqrt_l28_28167


namespace line_direction_vector_correct_l28_28576

theorem line_direction_vector_correct :
  ∃ (A B C : ℝ), (A = 2 ∧ B = -3 ∧ C = 1) ∧ 
  ∃ (v w : ℝ), (v = A ∧ w = B) :=
by
  sorry

end line_direction_vector_correct_l28_28576


namespace range_of_m_l28_28596

noncomputable def G (x m : ℝ) : ℝ := (8 * x^2 + 24 * x + 5 * m) / 8

theorem range_of_m (G_is_square : ∃ c d, ∀ x, G x m = (c * x + d) ^ 2) : 3 < m ∧ m < 4 :=
by
  sorry

end range_of_m_l28_28596


namespace diff_cubes_square_of_squares_l28_28542

theorem diff_cubes_square_of_squares {x y : ℤ} (h1 : (x + 1) ^ 3 - x ^ 3 = y ^ 2) :
  ∃ (a b : ℤ), y = a ^ 2 + b ^ 2 ∧ a = b + 1 :=
sorry

end diff_cubes_square_of_squares_l28_28542


namespace sufficient_not_necessary_condition_l28_28962

variable (x y : ℝ)

theorem sufficient_not_necessary_condition (h : x + y ≤ 1) : x ≤ 1/2 ∨ y ≤ 1/2 := 
  sorry

end sufficient_not_necessary_condition_l28_28962


namespace isabel_earned_l28_28890

theorem isabel_earned :
  let bead_necklace_price := 4
  let gemstone_necklace_price := 8
  let bead_necklace_count := 3
  let gemstone_necklace_count := 3
  let sales_tax_rate := 0.05
  let discount_rate := 0.10

  let total_cost_before_tax := bead_necklace_count * bead_necklace_price + gemstone_necklace_count * gemstone_necklace_price
  let sales_tax := total_cost_before_tax * sales_tax_rate
  let total_cost_after_tax := total_cost_before_tax + sales_tax
  let discount := total_cost_after_tax * discount_rate
  let final_amount_earned := total_cost_after_tax - discount

  final_amount_earned = 34.02 :=
by {
  sorry
}

end isabel_earned_l28_28890


namespace maximum_price_for_360_skewers_price_for_1920_profit_l28_28611

-- Define the number of skewers sold as a function of the price
def skewers_sold (price : ℝ) : ℝ := 300 + 60 * (10 - price)

-- Define the profit as a function of the price
def profit (price : ℝ) : ℝ := (skewers_sold price) * (price - 3)

-- Maximum price for selling at least 360 skewers per day
theorem maximum_price_for_360_skewers (price : ℝ) (h : skewers_sold price ≥ 360) : price ≤ 9 :=
by {
    sorry
}

-- Price to achieve a profit of 1920 yuan per day with price constraint
theorem price_for_1920_profit (price : ℝ) (h₁ : profit price = 1920) (h₂ : price ≤ 8) : price = 7 :=
by {
    sorry
}

end maximum_price_for_360_skewers_price_for_1920_profit_l28_28611


namespace anna_grams_l28_28080

-- Definitions based on conditions
def gary_grams : ℕ := 30
def gary_cost_per_gram : ℝ := 15
def anna_cost_per_gram : ℝ := 20
def combined_cost : ℝ := 1450

-- Statement to prove
theorem anna_grams : (combined_cost - (gary_grams * gary_cost_per_gram)) / anna_cost_per_gram = 50 :=
by 
  sorry

end anna_grams_l28_28080


namespace find_f_2000_l28_28575

variable (f : ℕ → ℕ)
variable (x : ℕ)

axiom initial_condition : f 0 = 1
axiom recurrence_relation : ∀ x, f (x + 2) = f x + 4 * x + 2

theorem find_f_2000 : f 2000 = 3998001 :=
by
  sorry

end find_f_2000_l28_28575


namespace jason_initial_quarters_l28_28709

theorem jason_initial_quarters (q_d q_n q_i : ℕ) (h1 : q_d = 25) (h2 : q_n = 74) :
  q_i = q_n - q_d → q_i = 49 :=
by
  sorry

end jason_initial_quarters_l28_28709


namespace Tamara_is_95_inches_l28_28473

/- Defining the basic entities: Kim's height (K), Tamara's height, Gavin's height -/
def Kim_height (K : ℝ) := K
def Tamara_height (K : ℝ) := 3 * K - 4
def Gavin_height (K : ℝ) := 2 * K + 6

/- Combined height equation -/
def combined_height (K : ℝ) := (Tamara_height K) + (Kim_height K) + (Gavin_height K) = 200

/- Given that Kim's height satisfies the combined height condition,
   proving that Tamara's height is 95 inches -/
theorem Tamara_is_95_inches (K : ℝ) (h : combined_height K) : Tamara_height K = 95 :=
by
  sorry

end Tamara_is_95_inches_l28_28473


namespace proof_problem_l28_28342

-- Definitions of the sets U, A, B
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 6}
def B : Set ℕ := {2, 3, 4}

-- The complement of B with respect to U
def complement_U_B : Set ℕ := U \ B

-- The intersection of A and the complement of B with respect to U
def intersection_A_complement_U_B : Set ℕ := A ∩ complement_U_B

-- The statement we want to prove
theorem proof_problem : intersection_A_complement_U_B = {1, 6} :=
by
  sorry

end proof_problem_l28_28342


namespace polygon_properties_l28_28183

def interior_angle_sum (n : ℕ) : ℝ :=
  (n - 2) * 180

def exterior_angle_sum : ℝ :=
  360

theorem polygon_properties (n : ℕ) (h : interior_angle_sum n = 3 * exterior_angle_sum + 180) :
  n = 9 ∧ interior_angle_sum n / n = 140 :=
by
  sorry

end polygon_properties_l28_28183


namespace josh_bottle_caps_l28_28443

/--
Suppose:
1. 7 bottle caps weigh exactly one ounce.
2. Josh's entire bottle cap collection weighs 18 pounds exactly.
3. There are 16 ounces in 1 pound.
We aim to show that Josh has 2016 bottle caps in his collection.
-/
theorem josh_bottle_caps :
  (7 : ℕ) * (1 : ℕ) = (7 : ℕ) → 
  (18 : ℕ) * (16 : ℕ) = (288 : ℕ) →
  (288 : ℕ) * (7 : ℕ) = (2016 : ℕ) :=
by
  intros h1 h2;
  exact sorry

end josh_bottle_caps_l28_28443


namespace parabola_focus_distance_l28_28244

noncomputable def distance_to_focus (p : ℝ) (M : ℝ × ℝ) : ℝ :=
  let focus := (p, 0)
  let (x1, y1) := M
  let (x2, y2) := focus
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) + p

theorem parabola_focus_distance
  (M : ℝ × ℝ) (p : ℝ)
  (hM : M = (2, 2))
  (hp : p = 1) :
  distance_to_focus p M = Real.sqrt 5 + 1 :=
by
  sorry

end parabola_focus_distance_l28_28244


namespace odd_number_divides_3n_plus_1_l28_28629

theorem odd_number_divides_3n_plus_1 (n : ℕ) (h₁ : n % 2 = 1) (h₂ : n ∣ 3^n + 1) : n = 1 :=
by
  sorry

end odd_number_divides_3n_plus_1_l28_28629


namespace polynomial_rewrite_l28_28000

theorem polynomial_rewrite :
  ∃ (a b c d e f : ℤ), 
  (2401 * x^4 + 16 = (a * x + b) * (c * x^3 + d * x^2 + e * x + f)) ∧
  (a + b + c + d + e + f = 274) :=
sorry

end polynomial_rewrite_l28_28000


namespace smallest_n_45_l28_28910

def is_perfect_square (x : ℕ) : Prop :=
  ∃ k : ℕ, x = k * k

def is_perfect_cube (x : ℕ) : Prop :=
  ∃ m : ℕ, x = m * m * m

theorem smallest_n_45 :
  ∃ n : ℕ, n > 0 ∧ (is_perfect_square (5 * n)) ∧ (is_perfect_cube (3 * n)) ∧ ∀ m : ℕ, (m > 0 ∧ (is_perfect_square (5 * m)) ∧ (is_perfect_cube (3 * m))) → n ≤ m :=
sorry

end smallest_n_45_l28_28910


namespace triangle_equilateral_l28_28016

variable {a b c : ℝ}

theorem triangle_equilateral (h : a^2 + 2 * b^2 = 2 * b * (a + c) - c^2) : a = b ∧ b = c := by
  sorry

end triangle_equilateral_l28_28016


namespace part1_part2_l28_28556

noncomputable section
def g1 (x : ℝ) : ℝ := Real.log x

noncomputable def f (t : ℝ) : ℝ := 
  if g1 t = t then 1 else sorry  -- Assuming g1(x) = t has exactly one root.

theorem part1 (t : ℝ) : f t = 1 :=
by sorry

def g2 (x : ℝ) (a : ℝ) : ℝ := 
  if x ≤ 0 then x else -x^2 + 2*a*x + a

theorem part2 (a : ℝ) (h : ∃ t : ℝ, f (t + 2) > f t) : a > 1 :=
by sorry

end part1_part2_l28_28556


namespace express_b_c_range_a_not_monotonic_l28_28095

noncomputable def f (a b c x : ℝ) : ℝ := (a * x^2 + b * x + c) * Real.exp (-x)
noncomputable def f' (a b c x : ℝ) : ℝ := 
    (a * x^2 + b * x + c) * (-Real.exp (-x)) + (2 * a * x + b) * Real.exp (-x)

theorem express_b_c (a : ℝ) : 
    (∃ b c : ℝ, f a b c 0 = 2 * a ∧ f' a b c 0 = Real.pi / 4) → 
    (∃ b c : ℝ, b = 1 + 2 * a ∧ c = 2 * a) := 
sorry

noncomputable def g (a x : ℝ) : ℝ := -a * x^2 - x + 1

theorem range_a_not_monotonic (a : ℝ) : 
    (¬ (∀ x y : ℝ, x ∈ Set.Ici (1 / 2) → y ∈ Set.Ici (1 / 2) → x < y → g a x ≤ g a y)) → 
    (-1 / 4 < a ∧ a < 2) := 
sorry

end express_b_c_range_a_not_monotonic_l28_28095


namespace total_shaded_area_is_2pi_l28_28702

theorem total_shaded_area_is_2pi (sm_radius large_radius : ℝ) 
  (h_sm_radius : sm_radius = 1) 
  (h_large_radius : large_radius = 2) 
  (sm_circle_area large_circle_area total_shaded_area : ℝ) 
  (h_sm_circle_area : sm_circle_area = π * sm_radius^2) 
  (h_large_circle_area : large_circle_area = π * large_radius^2) 
  (h_total_shaded_area : total_shaded_area = large_circle_area - 2 * sm_circle_area) :
  total_shaded_area = 2 * π :=
by
  -- Proof goes here
  sorry

end total_shaded_area_is_2pi_l28_28702


namespace mohan_least_cookies_l28_28415

theorem mohan_least_cookies :
  ∃ b : ℕ, 
    b % 6 = 5 ∧
    b % 8 = 3 ∧
    b % 9 = 6 ∧
    b = 59 :=
by
  sorry

end mohan_least_cookies_l28_28415


namespace harvey_sold_17_steaks_l28_28819

variable (initial_steaks : ℕ) (steaks_left_after_first_sale : ℕ) (steaks_sold_in_second_sale : ℕ)

noncomputable def total_steaks_sold (initial_steaks steaks_left_after_first_sale steaks_sold_in_second_sale : ℕ) : ℕ :=
  (initial_steaks - steaks_left_after_first_sale) + steaks_sold_in_second_sale

theorem harvey_sold_17_steaks :
  initial_steaks = 25 →
  steaks_left_after_first_sale = 12 →
  steaks_sold_in_second_sale = 4 →
  total_steaks_sold initial_steaks steaks_left_after_first_sale steaks_sold_in_second_sale = 17 :=
by
  intros
  sorry

end harvey_sold_17_steaks_l28_28819


namespace converse_inverse_contrapositive_l28_28656

-- The original statement
def original_statement (x y : ℕ) : Prop :=
  (x + y = 5) → (x = 3 ∧ y = 2)

-- Converse of the original statement
theorem converse (x y : ℕ) : (x = 3 ∧ y = 2) → (x + y = 5) :=
by
  sorry

-- Inverse of the original statement
theorem inverse (x y : ℕ) : (x + y ≠ 5) → (x ≠ 3 ∨ y ≠ 2) :=
by
  sorry

-- Contrapositive of the original statement
theorem contrapositive (x y : ℕ) : (x ≠ 3 ∨ y ≠ 2) → (x + y ≠ 5) :=
by
  sorry

end converse_inverse_contrapositive_l28_28656


namespace sqrt_xyz_ge_sqrt_x_add_sqrt_y_add_sqrt_z_l28_28463

open Real

theorem sqrt_xyz_ge_sqrt_x_add_sqrt_y_add_sqrt_z (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y * z ≥ x * y + y * z + z * x) :
  sqrt (x * y * z) ≥ sqrt x + sqrt y + sqrt z :=
by
  sorry

end sqrt_xyz_ge_sqrt_x_add_sqrt_y_add_sqrt_z_l28_28463


namespace probability_xi_l28_28679

noncomputable def xi_distribution (k : ℕ) : ℚ :=
  if h : k > 0 then 1 / (2 : ℚ)^k else 0

theorem probability_xi (h : ∀ k : ℕ, k > 0 → xi_distribution k = 1 / (2 : ℚ)^k) :
  (xi_distribution 3 + xi_distribution 4) = 3 / 16 :=
by
  sorry

end probability_xi_l28_28679


namespace weight_of_purple_ring_l28_28897

noncomputable section

def orange_ring_weight : ℝ := 0.08333333333333333
def white_ring_weight : ℝ := 0.4166666666666667
def total_weight : ℝ := 0.8333333333

theorem weight_of_purple_ring :
  total_weight - orange_ring_weight - white_ring_weight = 0.3333333333 :=
by
  -- We'll place the statement here, leave out the proof for skipping.
  sorry

end weight_of_purple_ring_l28_28897


namespace gcd_2024_2048_l28_28030

theorem gcd_2024_2048 : Nat.gcd 2024 2048 = 8 := by
  sorry

end gcd_2024_2048_l28_28030


namespace Jamie_earns_10_per_hour_l28_28882

noncomputable def JamieHourlyRate (days_per_week : ℕ) (hours_per_day : ℕ) (weeks : ℕ) (total_earnings : ℕ) : ℕ :=
  let total_hours := days_per_week * hours_per_day * weeks
  total_earnings / total_hours

theorem Jamie_earns_10_per_hour :
  JamieHourlyRate 2 3 6 360 = 10 := by
  sorry

end Jamie_earns_10_per_hour_l28_28882


namespace smallest_angle_half_largest_l28_28289

open Real

-- Statement of the problem
theorem smallest_angle_half_largest (a b c : ℝ) (α β γ : ℝ)
  (h_sides : a = 4 ∧ b = 5 ∧ c = 6)
  (h_angles : α < β ∧ β < γ)
  (h_cos_alpha : cos α = (b^2 + c^2 - a^2) / (2 * b * c))
  (h_cos_gamma : cos γ = (a^2 + b^2 - c^2) / (2 * a * b)) :
  2 * α = γ := 
sorry

end smallest_angle_half_largest_l28_28289


namespace line_passes_through_circle_center_l28_28162

theorem line_passes_through_circle_center
  (a : ℝ)
  (h_line : ∀ (x y : ℝ), 3 * x + y + a = 0 → (x, y) = (-1, 2))
  (h_circle : ∀ (x y : ℝ), x^2 + y^2 + 2 * x - 4 * y = 0 → (x, y) = (-1, 2)) :
  a = 1 :=
by
  sorry

end line_passes_through_circle_center_l28_28162


namespace value_of_k_l28_28673

theorem value_of_k :
  ∀ (k : ℝ), (∃ m : ℝ, m = 4/5 ∧ (21 - (-5)) / (k - 3) = m) →
  k = 35.5 :=
by
  intros k hk
  -- Here hk is the proof that the line through (3, -5) and (k, 21) has the same slope as 4/5
  sorry

end value_of_k_l28_28673


namespace probability_two_slate_rocks_l28_28250

theorem probability_two_slate_rocks 
    (n_slate : ℕ) (n_pumice : ℕ) (n_granite : ℕ)
    (h_slate : n_slate = 12)
    (h_pumice : n_pumice = 16)
    (h_granite : n_granite = 8) :
    (n_slate / (n_slate + n_pumice + n_granite)) * ((n_slate - 1) / (n_slate + n_pumice + n_granite - 1)) = 11 / 105 :=
by
    sorry

end probability_two_slate_rocks_l28_28250


namespace republicans_in_house_l28_28055

theorem republicans_in_house (D R : ℕ) (h1 : D + R = 434) (h2 : R = D + 30) : R = 232 :=
by sorry

end republicans_in_house_l28_28055
