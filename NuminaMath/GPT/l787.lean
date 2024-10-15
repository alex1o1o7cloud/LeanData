import Mathlib

namespace NUMINAMATH_GPT_diving_competition_score_l787_78783

theorem diving_competition_score 
  (scores : List ℝ)
  (h : scores = [7.5, 8.0, 9.0, 6.0, 8.8])
  (degree_of_difficulty : ℝ)
  (hd : degree_of_difficulty = 3.2) :
  let sorted_scores := scores.erase 9.0 |>.erase 6.0
  let remaining_sum := sorted_scores.sum
  remaining_sum * degree_of_difficulty = 77.76 :=
by
  sorry

end NUMINAMATH_GPT_diving_competition_score_l787_78783


namespace NUMINAMATH_GPT_probability_ephraim_fiona_same_heads_as_keiko_l787_78772

/-- Define a function to calculate the probability that Keiko, Ephraim, and Fiona get the same number of heads. -/
def probability_same_heads : ℚ :=
  let total_outcomes := (2^2) * (2^3) * (2^3)
  let successful_outcomes := 13
  successful_outcomes / total_outcomes

/-- Theorem stating the problem condition and expected probability. -/
theorem probability_ephraim_fiona_same_heads_as_keiko
  (h_keiko : ℕ := 2) -- Keiko tosses two coins
  (h_ephraim : ℕ := 3) -- Ephraim tosses three coins
  (h_fiona : ℕ := 3) -- Fiona tosses three coins
  -- Expected probability that both Ephraim and Fiona get the same number of heads as Keiko
  : probability_same_heads = 13 / 256 :=
sorry

end NUMINAMATH_GPT_probability_ephraim_fiona_same_heads_as_keiko_l787_78772


namespace NUMINAMATH_GPT_ratio_of_time_l787_78779

theorem ratio_of_time (T_A T_B : ℝ) (h1 : T_A = 8) (h2 : 1 / T_A + 1 / T_B = 0.375) :
  T_B / T_A = 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_of_time_l787_78779


namespace NUMINAMATH_GPT_common_ratio_of_geometric_series_l787_78721

-- Definitions of the first two terms of the geometric series
def term1 : ℚ := 4 / 7
def term2 : ℚ := -8 / 3

-- Theorem to prove the common ratio
theorem common_ratio_of_geometric_series : (term2 / term1 = -14 / 3) := by
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_series_l787_78721


namespace NUMINAMATH_GPT_candle_remaining_length_l787_78706

-- Define the initial length of the candle and the burn rate
def initial_length : ℝ := 20
def burn_rate : ℝ := 5

-- Define the remaining length function
def remaining_length (t : ℝ) : ℝ := initial_length - burn_rate * t

-- Prove the relationship between time and remaining length for the given range of time
theorem candle_remaining_length (t : ℝ) (ht: 0 ≤ t ∧ t ≤ 4) : remaining_length t = 20 - 5 * t :=
by
  dsimp [remaining_length]
  sorry

end NUMINAMATH_GPT_candle_remaining_length_l787_78706


namespace NUMINAMATH_GPT_even_function_a_value_l787_78797

theorem even_function_a_value (a : ℝ) :
  (∀ x : ℝ, (x^2 + a * x - 1) = ((-x)^2 + a * (-x) - 1)) ↔ a = 0 :=
by
  sorry

end NUMINAMATH_GPT_even_function_a_value_l787_78797


namespace NUMINAMATH_GPT_betty_honey_oats_problem_l787_78741

theorem betty_honey_oats_problem
  (o h : ℝ)
  (h_condition1 : o ≥ 8 + h / 3)
  (h_condition2 : o ≤ 3 * h) :
  h ≥ 3 :=
sorry

end NUMINAMATH_GPT_betty_honey_oats_problem_l787_78741


namespace NUMINAMATH_GPT_find_a1_l787_78718

variable (a : ℕ → ℚ) (d : ℚ)
variable (S : ℕ → ℚ)
variable (h_seq : ∀ n, a (n + 1) = a n + d)
variable (h_diff : d ≠ 0)
variable (h_prod : (a 2) * (a 3) = (a 4) * (a 5))
variable (h_sum : S 4 = 27)
variable (h_sum_def : ∀ n, S n = n * (a 1 + a n) / 2)

theorem find_a1 : a 1 = 135 / 8 := by
  sorry

end NUMINAMATH_GPT_find_a1_l787_78718


namespace NUMINAMATH_GPT_initial_games_l787_78798

theorem initial_games (X : ℕ) (h1 : X + 31 - 105 = 6) : X = 80 :=
by
  sorry

end NUMINAMATH_GPT_initial_games_l787_78798


namespace NUMINAMATH_GPT_quadratic_interval_solution_l787_78748

open Set

def quadratic_function (x : ℝ) : ℝ := x^2 + 5 * x + 6

theorem quadratic_interval_solution :
  {x : ℝ | 6 ≤ quadratic_function x ∧ quadratic_function x ≤ 12} = {x | -6 ≤ x ∧ x ≤ -5} ∪ {x | 0 ≤ x ∧ x ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_quadratic_interval_solution_l787_78748


namespace NUMINAMATH_GPT_third_side_length_l787_78714

theorem third_side_length (a b : ℝ) (h : (a - 3) ^ 2 + |b - 4| = 0) :
  ∃ x : ℝ, (a = 3 ∧ b = 4) ∧ (x = 5 ∨ x = Real.sqrt 7) :=
by
  sorry

end NUMINAMATH_GPT_third_side_length_l787_78714


namespace NUMINAMATH_GPT_speed_of_first_train_is_correct_l787_78759

-- Define the lengths of the trains
def length_train1 : ℕ := 110
def length_train2 : ℕ := 200

-- Define the speed of the second train in kmph
def speed_train2 : ℕ := 65

-- Define the time they take to clear each other in seconds
def time_clear_seconds : ℚ := 7.695936049253991

-- Define the speed of the first train
def speed_train1 : ℚ :=
  let time_clear_hours : ℚ := time_clear_seconds / 3600
  let total_distance_km : ℚ := (length_train1 + length_train2) / 1000
  let relative_speed_kmph : ℚ := total_distance_km / time_clear_hours 
  relative_speed_kmph - speed_train2

-- The proof problem is to show that the speed of the first train is 80.069 kmph
theorem speed_of_first_train_is_correct : speed_train1 = 80.069 := by
  sorry

end NUMINAMATH_GPT_speed_of_first_train_is_correct_l787_78759


namespace NUMINAMATH_GPT_general_term_sum_bn_l787_78751

noncomputable def S (n : ℕ) : ℕ := 2 * n^2 + 2 * n
noncomputable def a (n : ℕ) : ℕ := 4 * n
noncomputable def b (n : ℕ) : ℕ := 2 ^ (4 * n)
noncomputable def T (n : ℕ) : ℝ := (16 / 15) * (16^n - 1)

theorem general_term (n : ℕ) (h1 : S n = 2 * n^2 + 2 * n) 
    (h2 : S (n-1) = 2 * (n-1)^2 + 2 * (n-1))
    (h3 : n ≥ 1) : a n = 4 * n :=
by sorry

theorem sum_bn (n : ℕ) (h : ∀ n, (b n, a n) = ((2 ^ (4 * n)), 4 * n)) : 
    T n = (16 / 15) * (16^n - 1) :=
by sorry

end NUMINAMATH_GPT_general_term_sum_bn_l787_78751


namespace NUMINAMATH_GPT_sum_of_areas_of_circles_l787_78790

noncomputable def radius (n : ℕ) : ℝ :=
  3 / 3^n

noncomputable def area (n : ℕ) : ℝ :=
  Real.pi * (radius n)^2

noncomputable def total_area : ℝ :=
  ∑' n, area n

theorem sum_of_areas_of_circles:
  total_area = (9 * Real.pi) / 8 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_areas_of_circles_l787_78790


namespace NUMINAMATH_GPT_sum_of_roots_of_quadratic_l787_78747

noncomputable def x1_x2_roots_properties : Prop :=
  ∃ x₁ x₂ : ℝ, (x₁ + x₂ = 3) ∧ (x₁ * x₂ = -4)

theorem sum_of_roots_of_quadratic :
  ∃ x₁ x₂ : ℝ, (x₁ * x₂ = -4) → (x₁ + x₂ = 3) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_of_quadratic_l787_78747


namespace NUMINAMATH_GPT_ball_distance_traveled_l787_78752

noncomputable def total_distance (a1 d n : ℕ) : ℕ :=
  n * (a1 + a1 + (n-1) * d) / 2

theorem ball_distance_traveled : 
  total_distance 8 5 20 = 1110 :=
by
  sorry

end NUMINAMATH_GPT_ball_distance_traveled_l787_78752


namespace NUMINAMATH_GPT_find_integer_pairs_l787_78745

def is_perfect_square (x : ℤ) : Prop :=
  ∃ k : ℤ, k * k = x

theorem find_integer_pairs (m n : ℤ) :
  (is_perfect_square (m^2 + 4 * n) ∧ is_perfect_square (n^2 + 4 * m)) ↔
  (∃ a : ℤ, (m = 0 ∧ n = a^2) ∨ (m = a^2 ∧ n = 0) ∨ (m = -4 ∧ n = -4) ∨ (m = -5 ∧ n = -6) ∨ (m = -6 ∧ n = -5)) :=
by
  sorry

end NUMINAMATH_GPT_find_integer_pairs_l787_78745


namespace NUMINAMATH_GPT_volume_of_one_wedge_l787_78787

theorem volume_of_one_wedge 
  (circumference : ℝ)
  (h : circumference = 15 * Real.pi) 
  (radius : ℝ) 
  (volume : ℝ) 
  (wedge_volume : ℝ) 
  (h_radius : radius = 7.5)
  (h_volume : volume = (4 / 3) * Real.pi * radius^3)
  (h_wedge_volume : wedge_volume = volume / 5)
  : wedge_volume = 112.5 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_volume_of_one_wedge_l787_78787


namespace NUMINAMATH_GPT_tax_diminished_by_20_percent_l787_78799

theorem tax_diminished_by_20_percent
(T C : ℝ) 
(hT : T > 0) 
(hC : C > 0) 
(X : ℝ) 
(h_increased_consumption : ∀ (T C : ℝ), (C * 1.15) = C + 0.15 * C)
(h_decrease_revenue : T * (1 - X / 100) * C * 1.15 = T * C * 0.92) :
X = 20 := 
sorry

end NUMINAMATH_GPT_tax_diminished_by_20_percent_l787_78799


namespace NUMINAMATH_GPT_download_time_correct_l787_78771

-- Define the given conditions
def total_size : ℕ := 880
def downloaded : ℕ := 310
def speed : ℕ := 3

-- Calculate the remaining time to download
def time_remaining : ℕ := (total_size - downloaded) / speed

-- Theorem statement that needs to be proven
theorem download_time_correct : time_remaining = 190 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_download_time_correct_l787_78771


namespace NUMINAMATH_GPT_common_tangent_exists_l787_78711

theorem common_tangent_exists:
  ∃ (a b c : ℕ), (a + b + c = 11) ∧
  ( ∀ (x y : ℝ),
      (y = x^2 + 12/5) ∧ 
      (x = y^2 + 99/10) ∧ 
      (a*x + b*y = c) ∧ 
      0 < a ∧ 0 < b ∧ 0 < c ∧ 
      Int.gcd (Int.gcd a b) c = 1
  ) := 
by
  sorry

end NUMINAMATH_GPT_common_tangent_exists_l787_78711


namespace NUMINAMATH_GPT_traveling_cost_l787_78730

def area_road_length_parallel (length width : ℕ) := width * length

def area_road_breadth_parallel (length width : ℕ) := width * length

def area_intersection (width : ℕ) := width * width

def total_area_of_roads  (length breadth width : ℕ) : ℕ :=
  (area_road_length_parallel length width) + (area_road_breadth_parallel breadth width) - area_intersection width

def cost_of_traveling_roads (total_area_of_roads cost_per_sq_m : ℕ) := total_area_of_roads * cost_per_sq_m

theorem traveling_cost
  (length breadth width cost_per_sq_m : ℕ)
  (h_length : length = 80)
  (h_breadth : breadth = 50)
  (h_width : width = 10)
  (h_cost_per_sq_m : cost_per_sq_m = 3)
  : cost_of_traveling_roads (total_area_of_roads length breadth width) cost_per_sq_m = 3600 :=
by
  sorry

end NUMINAMATH_GPT_traveling_cost_l787_78730


namespace NUMINAMATH_GPT_number_of_boys_selected_l787_78788

theorem number_of_boys_selected {boys girls selections : ℕ} 
  (h_boys : boys = 11) (h_girls : girls = 10) (h_selections : selections = 6600) : 
  ∃ (k : ℕ), k = 2 :=
sorry

end NUMINAMATH_GPT_number_of_boys_selected_l787_78788


namespace NUMINAMATH_GPT_max_residents_per_apartment_l787_78733

theorem max_residents_per_apartment (total_floors : ℕ) (floors_with_6_apts : ℕ) (floors_with_5_apts : ℕ)
  (rooms_per_6_floors : ℕ) (rooms_per_5_floors : ℕ) (max_residents : ℕ) : 
  total_floors = 12 ∧ floors_with_6_apts = 6 ∧ floors_with_5_apts = 6 ∧ 
  rooms_per_6_floors = 6 ∧ rooms_per_5_floors = 5 ∧ max_residents = 264 → 
  264 / (6 * 6 + 6 * 5) = 4 := sorry

end NUMINAMATH_GPT_max_residents_per_apartment_l787_78733


namespace NUMINAMATH_GPT_cyclic_inequality_l787_78728

theorem cyclic_inequality (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) :
  (x + y) * Real.sqrt (y + z) * Real.sqrt (z + x) + (y + z) * Real.sqrt (z + x) * Real.sqrt (x + y) + (z + x) * Real.sqrt (x + y) * Real.sqrt (y + z) ≥ 4 * (x * y + y * z + z * x) :=
by
  sorry

end NUMINAMATH_GPT_cyclic_inequality_l787_78728


namespace NUMINAMATH_GPT_cos_double_angle_l787_78784

theorem cos_double_angle (α : ℝ) (h : ‖(Real.cos α, Real.sqrt 2 / 2)‖ = Real.sqrt 3 / 2) : Real.cos (2 * α) = -1 / 2 :=
sorry

end NUMINAMATH_GPT_cos_double_angle_l787_78784


namespace NUMINAMATH_GPT_range_of_x_l787_78764

theorem range_of_x (m : ℝ) (x : ℝ) (h : 0 < m ∧ m ≤ 5) : 
  (x^2 + (2 * m - 1) * x > 4 * x + 2 * m - 4) ↔ (x < -6 ∨ x > 4) := 
sorry

end NUMINAMATH_GPT_range_of_x_l787_78764


namespace NUMINAMATH_GPT_initial_speed_is_sixty_l787_78734

variable (D T : ℝ)

-- Condition: Two-thirds of the distance is covered in one-third of the total time.
def two_thirds_distance_in_one_third_time (V : ℝ) : Prop :=
  (2 * D / 3) / V = T / 3

-- Condition: The remaining distance is covered at 15 kmph.
def remaining_distance_at_fifteen_kmph : Prop :=
  (D / 3) / 15 = T - T / 3

-- Given that 30T = D from simplification in the solution.
def distance_time_relationship : Prop :=
  D = 30 * T

-- Prove that the initial speed V is 60 kmph.
theorem initial_speed_is_sixty (V : ℝ) (h1 : two_thirds_distance_in_one_third_time D T V) (h2 : remaining_distance_at_fifteen_kmph D T) (h3 : distance_time_relationship D T) : V = 60 := 
  sorry

end NUMINAMATH_GPT_initial_speed_is_sixty_l787_78734


namespace NUMINAMATH_GPT_initially_planned_days_l787_78700

-- Definitions of the conditions
def total_work_initial (x : ℕ) : ℕ := 50 * x
def total_work_with_reduction (x : ℕ) : ℕ := 25 * (x + 20)

-- The main theorem
theorem initially_planned_days :
  ∀ (x : ℕ), total_work_initial x = total_work_with_reduction x → x = 20 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_initially_planned_days_l787_78700


namespace NUMINAMATH_GPT_original_price_of_petrol_l787_78742

variable (P : ℝ)

theorem original_price_of_petrol (h : 0.9 * (200 / P - 200 / (0.9 * P)) = 5) : 
  (P = 20 / 4.5) :=
sorry

end NUMINAMATH_GPT_original_price_of_petrol_l787_78742


namespace NUMINAMATH_GPT_number_of_boys_l787_78778

def initial_girls : ℕ := 706
def new_girls : ℕ := 418
def total_pupils : ℕ := 1346
def total_girls := initial_girls + new_girls

theorem number_of_boys : 
  total_pupils = total_girls + 222 := 
by
  sorry

end NUMINAMATH_GPT_number_of_boys_l787_78778


namespace NUMINAMATH_GPT_joan_football_games_l787_78770

theorem joan_football_games (games_this_year games_last_year total_games: ℕ)
  (h1 : games_this_year = 4)
  (h2 : games_last_year = 9)
  (h3 : total_games = games_this_year + games_last_year) :
  total_games = 13 := 
by
  sorry

end NUMINAMATH_GPT_joan_football_games_l787_78770


namespace NUMINAMATH_GPT_a1_greater_than_floor_2n_over_3_l787_78767

theorem a1_greater_than_floor_2n_over_3
  (n : ℕ)
  (a : ℕ → ℕ)
  (h1 : ∀ i j : ℕ, i < j → i ≤ n ∧ j ≤ n → a i < a j)
  (h2 : ∀ i j : ℕ, i ≠ j → i ≤ n ∧ j ≤ n → lcm (a i) (a j) > 2 * n)
  (h_max : ∀ i : ℕ, i ≤ n → a i ≤ 2 * n) :
  a 1 > (2 * n) / 3 :=
by
  sorry

end NUMINAMATH_GPT_a1_greater_than_floor_2n_over_3_l787_78767


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l787_78754

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 - 3 * x - 4 = 0) -> (x = 4 ∨ x = -1) ∧ ¬(x = 4 ∨ x = -1 -> x = 4) :=
by sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l787_78754


namespace NUMINAMATH_GPT_gdp_scientific_notation_l787_78780

theorem gdp_scientific_notation (trillion : ℕ) (five_year_growth : ℝ) (gdp : ℝ) :
  trillion = 10^12 ∧ 1 ≤ gdp / 10^14 ∧ gdp / 10^14 < 10 ∧ gdp = 121 * 10^12 → gdp = 1.21 * 10^14
:= by
  sorry

end NUMINAMATH_GPT_gdp_scientific_notation_l787_78780


namespace NUMINAMATH_GPT_distance_from_y_axis_l787_78753

theorem distance_from_y_axis (P : ℝ × ℝ) (x : ℝ) (hx : P = (x, -9)) 
  (h : (abs (P.2) = 1/2 * abs (P.1))) :
  abs x = 18 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_y_axis_l787_78753


namespace NUMINAMATH_GPT_count_consecutive_sequences_l787_78735

def consecutive_sequences (n : ℕ) : ℕ :=
  if n = 15 then 270 else 0

theorem count_consecutive_sequences : consecutive_sequences 15 = 270 :=
by
  sorry

end NUMINAMATH_GPT_count_consecutive_sequences_l787_78735


namespace NUMINAMATH_GPT_square_tiles_count_l787_78781

theorem square_tiles_count (a b : ℕ) (h1 : a + b = 25) (h2 : 3 * a + 4 * b = 84) : b = 9 := by
  sorry

end NUMINAMATH_GPT_square_tiles_count_l787_78781


namespace NUMINAMATH_GPT_find_base_number_l787_78776

-- Define the base number
def base_number (x : ℕ) (k : ℕ) : Prop := x ^ k > 4 ^ 22

-- State the theorem based on the problem conditions
theorem find_base_number : ∃ x : ℕ, ∀ k : ℕ, (k = 8) → (base_number x k) → (x = 64) :=
by sorry

end NUMINAMATH_GPT_find_base_number_l787_78776


namespace NUMINAMATH_GPT_rate_of_current_l787_78713

theorem rate_of_current (speed_boat_still_water : ℕ) (time_hours : ℚ) (distance_downstream : ℚ)
    (h_speed_boat_still_water : speed_boat_still_water = 20)
    (h_time_hours : time_hours = 15 / 60)
    (h_distance_downstream : distance_downstream = 6.25) :
    ∃ c : ℚ, distance_downstream = (speed_boat_still_water + c) * time_hours ∧ c = 5 :=
by
    sorry

end NUMINAMATH_GPT_rate_of_current_l787_78713


namespace NUMINAMATH_GPT_larger_number_is_55_l787_78765

theorem larger_number_is_55 (x y : ℤ) (h1 : x + y = 70) (h2 : x = 3 * y + 10) (h3 : y = 15) : x = 55 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_is_55_l787_78765


namespace NUMINAMATH_GPT_denomination_is_100_l787_78774

-- Define the initial conditions
def num_bills : ℕ := 8
def total_savings : ℕ := 800

-- Define the denomination of the bills
def denomination_bills (num_bills : ℕ) (total_savings : ℕ) : ℕ := 
  total_savings / num_bills

-- The theorem stating the denomination is $100
theorem denomination_is_100 :
  denomination_bills num_bills total_savings = 100 := by
  sorry

end NUMINAMATH_GPT_denomination_is_100_l787_78774


namespace NUMINAMATH_GPT_geometric_sequence_sum_l787_78712

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a n = a 1 * q ^ n

theorem geometric_sequence_sum (h : geometric_sequence a 2) (h_sum : a 1 + a 2 = 3) :
  a 4 + a 5 = 24 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l787_78712


namespace NUMINAMATH_GPT_min_value_of_reciprocal_sum_l787_78701

variable (m n : ℝ)

theorem min_value_of_reciprocal_sum (hmn : m * n > 0) (h_line : m + n = 2) :
  (1 / m + 1 / n = 2) :=
sorry

end NUMINAMATH_GPT_min_value_of_reciprocal_sum_l787_78701


namespace NUMINAMATH_GPT_rectangular_to_polar_l787_78775

theorem rectangular_to_polar : 
  ∃ r θ, r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ (r, θ) = (3 * Real.sqrt 2, 7 * Real.pi / 4) := 
by
  sorry

end NUMINAMATH_GPT_rectangular_to_polar_l787_78775


namespace NUMINAMATH_GPT_roberts_total_sales_l787_78794

theorem roberts_total_sales 
  (basic_salary : ℝ := 1250) 
  (commission_rate : ℝ := 0.10) 
  (savings_rate : ℝ := 0.20) 
  (monthly_expenses : ℝ := 2888) 
  (S : ℝ) : S = 23600 :=
by
  have total_earnings := basic_salary + commission_rate * S
  have used_for_expenses := (1 - savings_rate) * total_earnings
  have expenses_eq : used_for_expenses = monthly_expenses := sorry
  have expense_calc : (1 - savings_rate) * (basic_salary + commission_rate * S) = monthly_expenses := sorry
  have simplify_eq : 0.80 * (1250 + 0.10 * S) = 2888 := sorry
  have open_eq : 1000 + 0.08 * S = 2888 := sorry
  have isolate_S : 0.08 * S = 1888 := sorry
  have solve_S : S = 1888 / 0.08 := sorry
  have final_S : S = 23600 := sorry
  exact final_S

end NUMINAMATH_GPT_roberts_total_sales_l787_78794


namespace NUMINAMATH_GPT_correct_calculation_l787_78744

-- Definitions of the equations
def option_A (a : ℝ) : Prop := a + 2 * a = 3 * a^2
def option_B (a b : ℝ) : Prop := (a^2 * b)^3 = a^6 * b^3
def option_C (a : ℝ) (m : ℕ) : Prop := (a^m)^2 = a^(m+2)
def option_D (a : ℝ) : Prop := a^3 * a^2 = a^6

-- The theorem that states option B is correct and others are incorrect
theorem correct_calculation (a b : ℝ) (m : ℕ) : 
  ¬ option_A a ∧ 
  option_B a b ∧ 
  ¬ option_C a m ∧ 
  ¬ option_D a :=
by sorry

end NUMINAMATH_GPT_correct_calculation_l787_78744


namespace NUMINAMATH_GPT_lions_min_games_for_90_percent_wins_l787_78773

theorem lions_min_games_for_90_percent_wins : 
  ∀ N : ℕ, (N ≥ 26) ↔ 1 + N ≥ (9 * (4 + N)) / 10 := 
by 
  sorry

end NUMINAMATH_GPT_lions_min_games_for_90_percent_wins_l787_78773


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_l787_78755

theorem right_triangle_hypotenuse (x : ℝ) (h : x^2 = 3^2 + 5^2) : x = Real.sqrt 34 :=
by sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_l787_78755


namespace NUMINAMATH_GPT_annette_miscalculation_l787_78708

theorem annette_miscalculation :
  let x := 6
  let y := 3
  let x' := 5
  let y' := 4
  x' - y' = 1 :=
by
  let x := 6
  let y := 3
  let x' := 5
  let y' := 4
  sorry

end NUMINAMATH_GPT_annette_miscalculation_l787_78708


namespace NUMINAMATH_GPT_num_four_digit_integers_with_at_least_one_4_or_7_l787_78720

def count_four_digit_integers_with_4_or_7 : ℕ := 5416

theorem num_four_digit_integers_with_at_least_one_4_or_7 :
  let all_four_digit_integers := 9000
  let valid_digits_first := 7
  let valid_digits := 8
  let integers_without_4_or_7 := valid_digits_first * valid_digits * valid_digits * valid_digits
  all_four_digit_integers - integers_without_4_or_7 = count_four_digit_integers_with_4_or_7 :=
by
  -- Using known values from the problem statement
  let all_four_digit_integers := 9000
  let valid_digits_first := 7
  let valid_digits := 8
  let integers_without_4_or_7 := valid_digits_first * valid_digits * valid_digits * valid_digits
  show all_four_digit_integers - integers_without_4_or_7 = count_four_digit_integers_with_4_or_7
  sorry

end NUMINAMATH_GPT_num_four_digit_integers_with_at_least_one_4_or_7_l787_78720


namespace NUMINAMATH_GPT_area_of_triangle_formed_by_line_and_axes_l787_78704

-- Definition of the line equation condition
def line_eq (x y : ℝ) : Prop := 2 * x - 5 * y - 10 = 0

-- Statement of the problem to prove
theorem area_of_triangle_formed_by_line_and_axes :
  (∃ x y : ℝ, line_eq x y ∧ x = 0 ∧ y = -2) ∧
  (∃ x y : ℝ, line_eq x y ∧ x = 5 ∧ y = 0) →
  let base : ℝ := 5
  let height : ℝ := 2
  let area := (1 / 2) * base * height
  area = 5 := 
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_formed_by_line_and_axes_l787_78704


namespace NUMINAMATH_GPT_find_values_of_A_l787_78766

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem find_values_of_A (A B C : ℕ) :
  sum_of_digits A = B ∧
  sum_of_digits B = C ∧
  A + B + C = 60 →
  (A = 44 ∨ A = 50 ∨ A = 47) :=
by
  sorry

end NUMINAMATH_GPT_find_values_of_A_l787_78766


namespace NUMINAMATH_GPT_periodic_odd_function_example_l787_78738

open Real

def periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x
def odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem periodic_odd_function_example (f : ℝ → ℝ) 
  (h_odd : odd f) 
  (h_periodic : periodic f 2) : 
  f 1 + f 4 + f 7 = 0 := 
sorry

end NUMINAMATH_GPT_periodic_odd_function_example_l787_78738


namespace NUMINAMATH_GPT_factorization_25x2_minus_155x_minus_150_l787_78724

theorem factorization_25x2_minus_155x_minus_150 :
  ∃ (a b : ℤ), (a + b) * 5 = -155 ∧ a * b = -150 ∧ a + 2 * b = 27 :=
by
  sorry

end NUMINAMATH_GPT_factorization_25x2_minus_155x_minus_150_l787_78724


namespace NUMINAMATH_GPT_find_number_l787_78760

theorem find_number (n : ℤ) (h : 7 * n - 15 = 2 * n + 10) : n = 5 :=
sorry

end NUMINAMATH_GPT_find_number_l787_78760


namespace NUMINAMATH_GPT_dinner_customers_l787_78739

theorem dinner_customers 
    (breakfast : ℕ)
    (lunch : ℕ)
    (total_friday : ℕ)
    (H : breakfast = 73)
    (H1 : lunch = 127)
    (H2 : total_friday = 287) :
  (breakfast + lunch + D = total_friday) → D = 87 := by
  sorry

end NUMINAMATH_GPT_dinner_customers_l787_78739


namespace NUMINAMATH_GPT_n_squared_divisible_by_144_l787_78763

theorem n_squared_divisible_by_144 (n : ℕ) (h1 : 0 < n) (h2 : ∃ t : ℕ, t = 12 ∧ ∀ d : ℕ, d ∣ n → d ≤ t) : 144 ∣ n^2 :=
sorry

end NUMINAMATH_GPT_n_squared_divisible_by_144_l787_78763


namespace NUMINAMATH_GPT_x_lt_1_iff_x_abs_x_lt_1_l787_78791

theorem x_lt_1_iff_x_abs_x_lt_1 (x : ℝ) : x < 1 ↔ x * |x| < 1 :=
sorry

end NUMINAMATH_GPT_x_lt_1_iff_x_abs_x_lt_1_l787_78791


namespace NUMINAMATH_GPT_avg_visitors_per_day_correct_l787_78716

-- Define the given conditions
def avg_sundays : Nat := 540
def avg_other_days : Nat := 240
def num_days : Nat := 30
def sundays_in_month : Nat := 5
def other_days_in_month : Nat := 25

-- Define the total visitors calculation
def total_visitors := (sundays_in_month * avg_sundays) + (other_days_in_month * avg_other_days)

-- Define the average visitors per day calculation
def avg_visitors_per_day := total_visitors / num_days

-- State the proof problem
theorem avg_visitors_per_day_correct : avg_visitors_per_day = 290 :=
by
  sorry

end NUMINAMATH_GPT_avg_visitors_per_day_correct_l787_78716


namespace NUMINAMATH_GPT_arithmetic_sequence_ninth_term_l787_78702

theorem arithmetic_sequence_ninth_term :
  ∃ a d : ℤ, (a + 2 * d = 23) ∧ (a + 5 * d = 29) ∧ (a + 8 * d = 35) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_ninth_term_l787_78702


namespace NUMINAMATH_GPT_championship_outcome_count_l787_78757

theorem championship_outcome_count (students championships : ℕ) (h_students : students = 8) (h_championships : championships = 3) : students ^ championships = 512 := by
  rw [h_students, h_championships]
  norm_num

end NUMINAMATH_GPT_championship_outcome_count_l787_78757


namespace NUMINAMATH_GPT_express_y_in_terms_of_y_l787_78769

variable (x : ℝ)

theorem express_y_in_terms_of_y (y : ℝ) (h : 2 * x - y = 3) : y = 2 * x - 3 :=
sorry

end NUMINAMATH_GPT_express_y_in_terms_of_y_l787_78769


namespace NUMINAMATH_GPT_find_z_l787_78703

theorem find_z
  (z : ℝ)
  (h : (1 : ℝ) • (2 : ℝ) + 4 • (-1 : ℝ) + z • (3 : ℝ) = 6) :
  z = 8 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_find_z_l787_78703


namespace NUMINAMATH_GPT_smallest_k_l787_78795

theorem smallest_k (k : ℕ) (h : 201 ≡ 9 [MOD 24]) : k = 1 := by
  sorry

end NUMINAMATH_GPT_smallest_k_l787_78795


namespace NUMINAMATH_GPT_lattice_points_condition_l787_78732

/-- A lattice point is a point on the plane with integer coordinates. -/
structure LatticePoint :=
  (x : ℤ)
  (y : ℤ)

/-- A triangle in the plane with three vertices and at least two lattice points inside. -/
structure Triangle :=
  (A B C : LatticePoint)
  (lattice_points_inside : List LatticePoint)
  (lattice_points_nonempty : lattice_points_inside.length ≥ 2)

noncomputable def exists_lattice_points (T : Triangle) : Prop :=
∃ (X Y : LatticePoint) (hX : X ∈ T.lattice_points_inside) (hY : Y ∈ T.lattice_points_inside), 
  ((∃ (V : LatticePoint), V = T.A ∨ V = T.B ∨ V = T.C ∧ ∃ (k : ℤ), (k : ℝ) * (Y.x - X.x) = (V.x - X.x) ∧ (k : ℝ) * (Y.y - X.y) = (V.y - X.y)) ∨
  (∃ (l m n : ℝ), l * (Y.x - X.x) = m * (T.A.x - T.B.x) ∧ l * (Y.y - X.y) = m * (T.A.y - T.B.y) ∨ l * (Y.x - X.x) = n * (T.B.x - T.C.x) ∧ l * (Y.y - X.y) = n * (T.B.y - T.C.y) ∨ l * (Y.x - X.x) = m * (T.C.x - T.A.x) ∧ l * (Y.y - X.y) = m * (T.C.y - T.A.y)))

theorem lattice_points_condition (T : Triangle) : exists_lattice_points T :=
sorry

end NUMINAMATH_GPT_lattice_points_condition_l787_78732


namespace NUMINAMATH_GPT_percentage_reduction_l787_78789

theorem percentage_reduction :
  let P := 60
  let R := 45
  (900 / R) - (900 / P) = 5 →
  (P - R) / P * 100 = 25 :=
by 
  intros P R h
  have h1 : R = 45 := rfl
  have h2 : P = 60 := sorry
  rw [h1] at h
  rw [h2]
  sorry -- detailed steps to be filled in the proof

end NUMINAMATH_GPT_percentage_reduction_l787_78789


namespace NUMINAMATH_GPT_find_m_l787_78756

theorem find_m (x1 x2 m : ℝ) (h1 : x1 + x2 = -3) (h2 : x1 * x2 = m) (h3 : 1 / x1 + 1 / x2 = 1) : m = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l787_78756


namespace NUMINAMATH_GPT_domain_of_function_l787_78723

theorem domain_of_function :
  {x : ℝ | x ≥ -1 ∧ x ≠ 1 / 2} =
  {x : ℝ | 2 * x - 1 ≠ 0 ∧ x + 1 ≥ 0} :=
by {
  sorry
}

end NUMINAMATH_GPT_domain_of_function_l787_78723


namespace NUMINAMATH_GPT_color_triangle_vertices_no_same_color_l787_78709

-- Define the colors and the vertices
inductive Color | red | green | blue | yellow
inductive Vertex | A | B | C 

-- Define a function that counts ways to color the triangle given constraints
def count_valid_colorings (colors : List Color) (vertices : List Vertex) : Nat := 
  -- There are 4 choices for the first vertex, 3 for the second, 2 for the third
  4 * 3 * 2

-- The theorem we want to prove
theorem color_triangle_vertices_no_same_color : count_valid_colorings [Color.red, Color.green, Color.blue, Color.yellow] [Vertex.A, Vertex.B, Vertex.C] = 24 := by
  sorry

end NUMINAMATH_GPT_color_triangle_vertices_no_same_color_l787_78709


namespace NUMINAMATH_GPT_num_handshakes_ten_women_l787_78796

def num_handshakes (n : ℕ) : ℕ :=
(n * (n - 1)) / 2

theorem num_handshakes_ten_women :
  num_handshakes 10 = 45 :=
by
  sorry

end NUMINAMATH_GPT_num_handshakes_ten_women_l787_78796


namespace NUMINAMATH_GPT_determinant_inequality_l787_78722

theorem determinant_inequality (x : ℝ) (h : 2 * x - (3 - x) > 0) : 3 * x - 3 > 0 := 
by
  sorry

end NUMINAMATH_GPT_determinant_inequality_l787_78722


namespace NUMINAMATH_GPT_norris_money_left_l787_78727

-- Define the amounts saved each month
def september_savings : ℕ := 29
def october_savings : ℕ := 25
def november_savings : ℕ := 31

-- Define the total savings
def total_savings : ℕ := september_savings + october_savings + november_savings

-- Define the amount spent on the online game
def amount_spent : ℕ := 75

-- Define the remaining money
def money_left : ℕ := total_savings - amount_spent

-- The theorem stating the problem and the solution
theorem norris_money_left : money_left = 10 := by
  sorry

end NUMINAMATH_GPT_norris_money_left_l787_78727


namespace NUMINAMATH_GPT_Andy_solves_correct_number_of_problems_l787_78725

-- Define the problem boundaries
def first_problem : ℕ := 80
def last_problem : ℕ := 125

-- The goal is to prove that Andy solves 46 problems given the range
theorem Andy_solves_correct_number_of_problems : (last_problem - first_problem + 1) = 46 :=
by
  sorry

end NUMINAMATH_GPT_Andy_solves_correct_number_of_problems_l787_78725


namespace NUMINAMATH_GPT_georgie_entry_exit_ways_l787_78746

-- Defining the conditions
def castle_windows : Nat := 8
def non_exitable_windows : Nat := 2

-- Defining the problem
theorem georgie_entry_exit_ways (total_windows : Nat) (blocked_exits : Nat) (entry_windows : Nat) : 
  total_windows = castle_windows → blocked_exits = non_exitable_windows → 
  entry_windows = castle_windows →
  (entry_windows * (total_windows - 1 - blocked_exits) = 40) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_georgie_entry_exit_ways_l787_78746


namespace NUMINAMATH_GPT_rectangle_area_l787_78749

theorem rectangle_area (y : ℝ) (w : ℝ) (h : w > 0) (h_diag : y^2 = 10 * w^2) : 
  (3 * w)^2 * w = 3 * (y^2 / 10) :=
by sorry

end NUMINAMATH_GPT_rectangle_area_l787_78749


namespace NUMINAMATH_GPT_exactly_two_succeed_probability_l787_78719

/-- Define the probabilities of three independent events -/
def P1 : ℚ := 1 / 2
def P2 : ℚ := 1 / 3
def P3 : ℚ := 3 / 4

/-- Define the probability that exactly two out of the three people successfully decrypt the password -/
def prob_exactly_two_succeed : ℚ := P1 * P2 * (1 - P3) + P1 * (1 - P2) * P3 + (1 - P1) * P2 * P3

theorem exactly_two_succeed_probability :
  prob_exactly_two_succeed = 5 / 12 :=
sorry

end NUMINAMATH_GPT_exactly_two_succeed_probability_l787_78719


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l787_78762

   theorem necessary_but_not_sufficient (a : ℝ) : a^2 > a → (a > 1) :=
   by {
     sorry
   }
   
end NUMINAMATH_GPT_necessary_but_not_sufficient_l787_78762


namespace NUMINAMATH_GPT_education_budget_l787_78793

-- Definitions of the conditions
def total_budget : ℕ := 32 * 10^6  -- 32 million
def policing_budget : ℕ := total_budget / 2
def public_spaces_budget : ℕ := 4 * 10^6  -- 4 million

-- The theorem statement
theorem education_budget :
  total_budget - (policing_budget + public_spaces_budget) = 12 * 10^6 :=
by
  sorry

end NUMINAMATH_GPT_education_budget_l787_78793


namespace NUMINAMATH_GPT_isosceles_right_triangle_area_l787_78758

theorem isosceles_right_triangle_area (a : ℝ) (h : ℝ) (p : ℝ) 
  (h_triangle : h = a * Real.sqrt 2) 
  (hypotenuse_is_16 : h = 16) :
  (1 / 2) * a * a = 64 := 
by
  -- Skip the proof as per guidelines
  sorry

end NUMINAMATH_GPT_isosceles_right_triangle_area_l787_78758


namespace NUMINAMATH_GPT_abc_sum_is_17_l787_78726

noncomputable def A := 3
noncomputable def B := 5
noncomputable def C := 9

theorem abc_sum_is_17 (A B C : ℕ) (h1 : 100 * A + 10 * B + C = 359) (h2 : 4 * (100 * A + 10 * B + C) = 1436)
  (h3 : A ≠ B) (h4 : B ≠ C) (h5 : A ≠ C) : A + B + C = 17 :=
by
  sorry

end NUMINAMATH_GPT_abc_sum_is_17_l787_78726


namespace NUMINAMATH_GPT_shifted_parabola_eq_l787_78785

theorem shifted_parabola_eq :
  ∀ x, (∃ y, y = 2 * (x - 3)^2 + 2) →
       (∃ y, y = 2 * (x + 0)^2 + 4) :=
by sorry

end NUMINAMATH_GPT_shifted_parabola_eq_l787_78785


namespace NUMINAMATH_GPT_textbook_order_total_cost_l787_78715

theorem textbook_order_total_cost :
  let english_quantity := 35
  let geography_quantity := 35
  let mathematics_quantity := 20
  let science_quantity := 30
  let english_price := 7.50
  let geography_price := 10.50
  let mathematics_price := 12.00
  let science_price := 9.50
  (english_quantity * english_price + geography_quantity * geography_price + mathematics_quantity * mathematics_price + science_quantity * science_price = 1155.00) :=
by sorry

end NUMINAMATH_GPT_textbook_order_total_cost_l787_78715


namespace NUMINAMATH_GPT_minimum_value_of_GP_l787_78729

theorem minimum_value_of_GP (a : ℕ → ℝ) (h : ∀ n, 0 < a n) (h_prod : a 2 * a 10 = 9) :
  a 5 + a 7 = 6 :=
by
  -- proof steps will be filled in here
  sorry

end NUMINAMATH_GPT_minimum_value_of_GP_l787_78729


namespace NUMINAMATH_GPT_largest_prime_factor_of_4620_l787_78768

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m ≤ n / m → ¬ (m ∣ n)

def prime_factors (n : ℕ) : List ℕ :=
  -- assumes a well-defined function that generates the prime factor list
  -- this is a placeholder function for demonstrating purposes
  sorry

def largest_prime_factor (l : List ℕ) : ℕ :=
  l.foldr max 0

theorem largest_prime_factor_of_4620 : largest_prime_factor (prime_factors 4620) = 11 :=
by
  sorry

end NUMINAMATH_GPT_largest_prime_factor_of_4620_l787_78768


namespace NUMINAMATH_GPT_red_basket_fruit_count_l787_78743

-- Defining the basket counts
def blue_basket_bananas := 12
def blue_basket_apples := 4
def blue_basket_fruits := blue_basket_bananas + blue_basket_apples
def red_basket_fruits := blue_basket_fruits / 2

-- Statement of the proof problem
theorem red_basket_fruit_count : red_basket_fruits = 8 := by
  sorry

end NUMINAMATH_GPT_red_basket_fruit_count_l787_78743


namespace NUMINAMATH_GPT_area_difference_l787_78731

theorem area_difference (d : ℝ) (r : ℝ) (ratio : ℝ) (h1 : d = 10) (h2 : ratio = 2) (h3 : r = 5) :
  (π * r^2 - ((d^2 / (ratio^2 + 1)).sqrt * (2 * d^2 / (ratio^2 + 1)).sqrt)) = 38.5 :=
by
  sorry

end NUMINAMATH_GPT_area_difference_l787_78731


namespace NUMINAMATH_GPT_right_triangle_segment_ratio_l787_78750

-- Definitions of the triangle sides and hypotenuse
def right_triangle (AB BC : ℝ) : Prop :=
  AB/BC = 4/3

def hypotenuse (AB BC AC : ℝ) : Prop :=
  AC^2 = AB^2 + BC^2

def perpendicular_segment_ratio (AD CD : ℝ) : Prop :=
  AD / CD = 9/16

-- Final statement of the problem
theorem right_triangle_segment_ratio
  (AB BC AC AD CD : ℝ)
  (h1 : right_triangle AB BC)
  (h2 : hypotenuse AB BC AC)
  (h3 : perpendicular_segment_ratio AD CD) :
  CD / AD = 16/9 := sorry

end NUMINAMATH_GPT_right_triangle_segment_ratio_l787_78750


namespace NUMINAMATH_GPT_dodecahedron_edges_l787_78710

noncomputable def regular_dodecahedron := Type

def faces : regular_dodecahedron → ℕ := λ _ => 12
def edges_per_face : regular_dodecahedron → ℕ := λ _ => 5
def shared_edges : regular_dodecahedron → ℕ := λ _ => 2

theorem dodecahedron_edges (d : regular_dodecahedron) :
  (faces d * edges_per_face d) / shared_edges d = 30 :=
by
  sorry

end NUMINAMATH_GPT_dodecahedron_edges_l787_78710


namespace NUMINAMATH_GPT_xyz_inequality_l787_78777

theorem xyz_inequality (x y z : ℝ) (hx : 0 ≤ x) (hx' : x ≤ 1) (hy : 0 ≤ y) (hy' : y ≤ 1) (hz : 0 ≤ z) (hz' : z ≤ 1) :
  (x^2 / (1 + x + x*y*z) + y^2 / (1 + y + x*y*z) + z^2 / (1 + z + x*y*z) ≤ 1) :=
sorry

end NUMINAMATH_GPT_xyz_inequality_l787_78777


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l787_78737

theorem hyperbola_eccentricity
  (a b m : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (PA_perpendicular_to_l2 : (b/a * m) / (m + a) * (-b/a) = -1)
  (PB_parallel_to_l2 : (b/a * m) / (m - a) = -b/a) :
  (∃ e, e = 2) :=
by sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l787_78737


namespace NUMINAMATH_GPT_find_a_for_perpendicular_lines_l787_78736

theorem find_a_for_perpendicular_lines (a : ℝ) 
    (h_perpendicular : 2 * a + (-1) * (3 - a) = 0) :
    a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_for_perpendicular_lines_l787_78736


namespace NUMINAMATH_GPT_total_spent_is_64_l787_78707

/-- Condition 1: The cost of each deck is 8 dollars -/
def deck_cost : ℕ := 8

/-- Condition 2: Tom bought 3 decks -/
def tom_decks : ℕ := 3

/-- Condition 3: Tom's friend bought 5 decks -/
def friend_decks : ℕ := 5

/-- Total amount spent by Tom and his friend -/
def total_amount_spent : ℕ := (tom_decks * deck_cost) + (friend_decks * deck_cost)

/-- Proof statement: Prove that total amount spent is 64 -/
theorem total_spent_is_64 : total_amount_spent = 64 := by
  sorry

end NUMINAMATH_GPT_total_spent_is_64_l787_78707


namespace NUMINAMATH_GPT_min_vertical_segment_length_l787_78717

noncomputable def vertical_segment_length (x : ℝ) : ℝ :=
  abs (|x| - (-x^2 - 4*x - 3))

theorem min_vertical_segment_length :
  ∃ x : ℝ, vertical_segment_length x = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_min_vertical_segment_length_l787_78717


namespace NUMINAMATH_GPT_num_pos_pairs_l787_78761

theorem num_pos_pairs (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : m^2 + 3 * n < 40) :
  ∃ k : ℕ, k = 45 :=
by {
  -- Additional setup and configuration if needed
  -- ...
  sorry
}

end NUMINAMATH_GPT_num_pos_pairs_l787_78761


namespace NUMINAMATH_GPT_solve_for_x_values_for_matrix_l787_78792

def matrix_equals_neg_two (x : ℝ) : Prop :=
  let a := 3 * x
  let b := x
  let c := 4
  let d := 2 * x
  (a * b - c * d = -2)

theorem solve_for_x_values_for_matrix : 
  ∃ (x : ℝ), matrix_equals_neg_two x ↔ (x = (4 + Real.sqrt 10) / 3 ∨ x = (4 - Real.sqrt 10) / 3) :=
sorry

end NUMINAMATH_GPT_solve_for_x_values_for_matrix_l787_78792


namespace NUMINAMATH_GPT_find_other_number_l787_78740

theorem find_other_number (a b : ℕ) (h_lcm : Nat.lcm a b = 9240) (h_gcd : Nat.gcd a b = 33) (h_a : a = 231) : b = 1320 :=
sorry

end NUMINAMATH_GPT_find_other_number_l787_78740


namespace NUMINAMATH_GPT_compare_a_b_c_l787_78786

noncomputable def a : ℝ := Real.log (Real.sqrt 2)
noncomputable def b : ℝ := (Real.log 3) / 3
noncomputable def c : ℝ := 1 / Real.exp 1

theorem compare_a_b_c : a < b ∧ b < c := by
  -- Proof will be done here
  sorry

end NUMINAMATH_GPT_compare_a_b_c_l787_78786


namespace NUMINAMATH_GPT_waiter_customers_l787_78705

-- Define initial conditions
def initial_customers : ℕ := 47
def customers_left : ℕ := 41
def new_customers : ℕ := 20

-- Calculate remaining customers after some left
def remaining_customers : ℕ := initial_customers - customers_left

-- Calculate the total customers after getting new ones
def total_customers : ℕ := remaining_customers + new_customers

-- State the theorem to prove the final total customers
theorem waiter_customers : total_customers = 26 := by
  -- We include sorry for the proof placeholder
  sorry

end NUMINAMATH_GPT_waiter_customers_l787_78705


namespace NUMINAMATH_GPT_triangle_classification_l787_78782

theorem triangle_classification 
  (a b c : ℝ) 
  (h : (b^2 + a^2) * (b - a) = b * c^2 - a * c^2) : 
  (a = b ∨ a^2 + b^2 = c^2) :=
by sorry

end NUMINAMATH_GPT_triangle_classification_l787_78782
