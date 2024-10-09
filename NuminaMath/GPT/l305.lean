import Mathlib

namespace geometric_sequence_seventh_term_l305_30558

theorem geometric_sequence_seventh_term
  (a r : ℝ)
  (h1 : a * r^4 = 16)
  (h2 : a * r^10 = 4) :
  a * r^6 = 4 * (2^(2/3)) :=
by
  sorry

end geometric_sequence_seventh_term_l305_30558


namespace triangle_area_is_12_5_l305_30530

structure Point :=
  (x : ℝ)
  (y : ℝ)

def M : Point := ⟨5, 0⟩
def N : Point := ⟨0, 5⟩
noncomputable def P (x y : ℝ) (h : x + y = 8) : Point := ⟨x, y⟩

noncomputable def area_triangle (A B C : Point) : ℝ :=
  (1 / 2) * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

theorem triangle_area_is_12_5 (x y : ℝ) (h : x + y = 8) :
  area_triangle M N (P x y h) = 12.5 :=
sorry

end triangle_area_is_12_5_l305_30530


namespace sum_series_75_to_99_l305_30519

theorem sum_series_75_to_99 : 
  let a := 75
  let l := 99
  let n := l - a + 1
  let s := n * (a + l) / 2
  s = 2175 :=
by
  sorry

end sum_series_75_to_99_l305_30519


namespace distinct_pairwise_products_l305_30521

theorem distinct_pairwise_products
  (n a b c d : ℕ) (h_distinct: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_bounds: n^2 < a ∧ a < b ∧ b < c ∧ c < d ∧ d < (n+1)^2) :
  (a * b ≠ a * c ∧ a * b ≠ a * d ∧ a * b ≠ b * c ∧ a * b ≠ b * d ∧ a * b ≠ c * d) ∧
  (a * c ≠ a * d ∧ a * c ≠ b * c ∧ a * c ≠ b * d ∧ a * c ≠ c * d) ∧
  (a * d ≠ b * c ∧ a * d ≠ b * d ∧ a * d ≠ c * d) ∧
  (b * c ≠ b * d ∧ b * c ≠ c * d) ∧
  (b * d ≠ c * d) :=
sorry

end distinct_pairwise_products_l305_30521


namespace min_value_ineq_l305_30584

theorem min_value_ineq (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 1) : 
  (∀ z : ℝ, z = (4 / x + 1 / y) → z ≥ 9) :=
by
  sorry

end min_value_ineq_l305_30584


namespace probability_of_winning_at_least_10_rubles_l305_30582

-- Definitions based on conditions
def total_tickets : ℕ := 100
def win_20_rubles_tickets : ℕ := 5
def win_15_rubles_tickets : ℕ := 10
def win_10_rubles_tickets : ℕ := 15
def win_2_rubles_tickets : ℕ := 25
def win_nothing_tickets : ℕ := total_tickets - (win_20_rubles_tickets + win_15_rubles_tickets + win_10_rubles_tickets + win_2_rubles_tickets)

-- Probability calculations
def prob_win_20_rubles : ℚ := win_20_rubles_tickets / total_tickets
def prob_win_15_rubles : ℚ := win_15_rubles_tickets / total_tickets
def prob_win_10_rubles : ℚ := win_10_rubles_tickets / total_tickets

-- Prove the probability of winning at least 10 rubles
theorem probability_of_winning_at_least_10_rubles : 
  prob_win_20_rubles + prob_win_15_rubles + prob_win_10_rubles = 0.30 := by
  sorry

end probability_of_winning_at_least_10_rubles_l305_30582


namespace inequality_for_natural_n_l305_30523

theorem inequality_for_natural_n (n : ℕ) : (2 * n + 1) ^ n ≥ (2 * n) ^ n + (2 * n - 1) ^ n :=
by
  sorry

end inequality_for_natural_n_l305_30523


namespace cannot_form_right_triangle_l305_30589

theorem cannot_form_right_triangle (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) :
  a^2 + b^2 ≠ c^2 :=
by
  rw [h1, h2, h3]
  sorry

end cannot_form_right_triangle_l305_30589


namespace radiator_water_fraction_l305_30536

theorem radiator_water_fraction :
  let initial_volume := 20
  let replacement_volume := 5
  let fraction_remaining_per_replacement := (initial_volume - replacement_volume) / initial_volume
  fraction_remaining_per_replacement^4 = 81 / 256 := by
  let initial_volume := 20
  let replacement_volume := 5
  let fraction_remaining_per_replacement := (initial_volume - replacement_volume) / initial_volume
  sorry

end radiator_water_fraction_l305_30536


namespace carpet_size_l305_30574

def length := 5
def width := 2
def area := length * width

theorem carpet_size : area = 10 := by
  sorry

end carpet_size_l305_30574


namespace min_distance_between_curves_l305_30529

noncomputable def distance_between_intersections : ℝ :=
  let f (x : ℝ) := (2 * x + 1) - (x + Real.log x)
  let f' (x : ℝ) := 1 - 1 / x
  let minimum_distance :=
    if hs : 1 < 1 then 2 else
    if hs : 1 > 1 then 2 else
    2
  minimum_distance

theorem min_distance_between_curves : distance_between_intersections = 2 :=
by
  sorry

end min_distance_between_curves_l305_30529


namespace units_digit_m_squared_plus_3_to_m_l305_30579

theorem units_digit_m_squared_plus_3_to_m (m : ℕ) (h : m = 2021^2 + 3^2021) : (m^2 + 3^m) % 10 = 7 :=
by
  sorry

end units_digit_m_squared_plus_3_to_m_l305_30579


namespace shaded_area_l305_30524

-- Defining the conditions
def total_area_of_grid : ℕ := 38
def base_of_triangle : ℕ := 12
def height_of_triangle : ℕ := 4

-- Using the formula for the area of a right triangle
def area_of_unshaded_triangle : ℕ := (base_of_triangle * height_of_triangle) / 2

-- The goal: Prove the area of the shaded region
theorem shaded_area : total_area_of_grid - area_of_unshaded_triangle = 14 :=
by
  sorry

end shaded_area_l305_30524


namespace black_squares_count_l305_30598

def checkerboard_size : Nat := 32
def total_squares : Nat := checkerboard_size * checkerboard_size
def black_squares (n : Nat) : Nat := n / 2

theorem black_squares_count : black_squares total_squares = 512 := by
  let n := total_squares
  show black_squares n = 512
  sorry

end black_squares_count_l305_30598


namespace ball_cost_l305_30551

theorem ball_cost (C x y : ℝ)
  (H1 :  x = 1/3 * (C/2 + y + 5) )
  (H2 :  y = 1/4 * (C/2 + x + 5) )
  (H3 :  C/2 + x + y + 5 = C ) : C = 20 := 
by
  sorry

end ball_cost_l305_30551


namespace find_N_value_l305_30522

variable (a b N : ℚ)
variable (h1 : a + 2 * b = N)
variable (h2 : a * b = 4)
variable (h3 : 2 / a + 1 / b = 1.5)

theorem find_N_value : N = 6 :=
by
  sorry

end find_N_value_l305_30522


namespace cube_volume_l305_30515

theorem cube_volume (d : ℝ) (h : d = 6 * Real.sqrt 2) : 
  ∃ v : ℝ, v = 48 * Real.sqrt 6 := by
  let s := d / Real.sqrt 3
  let volume := s ^ 3
  use volume
  /- Proof of the volume calculation is omitted. -/
  sorry

end cube_volume_l305_30515


namespace teresa_marks_ratio_l305_30500

theorem teresa_marks_ratio (science music social_studies total_marks physics_ratio : ℝ) 
  (h_science : science = 70)
  (h_music : music = 80)
  (h_social_studies : social_studies = 85)
  (h_total_marks : total_marks = 275)
  (h_physics : science + music + social_studies + physics_ratio * music = total_marks) :
  physics_ratio = 1 / 2 :=
by
  subst h_science
  subst h_music
  subst h_social_studies
  subst h_total_marks
  have : 70 + 80 + 85 + physics_ratio * 80 = 275 := h_physics
  linarith

end teresa_marks_ratio_l305_30500


namespace additional_coins_needed_l305_30506

def num_friends : Nat := 15
def current_coins : Nat := 105

def total_coins_needed (n : Nat) : Nat :=
  n * (n + 1) / 2
  
theorem additional_coins_needed :
  let coins_needed := total_coins_needed num_friends
  let additional_coins := coins_needed - current_coins
  additional_coins = 15 :=
by
  sorry

end additional_coins_needed_l305_30506


namespace ratio_a3_a2_l305_30553

theorem ratio_a3_a2 (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) (x : ℝ)
  (h : (1 - 2 * x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) :
  a_3 / a_2 = -2 :=
sorry

end ratio_a3_a2_l305_30553


namespace length_of_uncovered_side_l305_30576

theorem length_of_uncovered_side (L W : ℕ) (h1 : L * W = 680) (h2 : 2 * W + L = 74) : L = 40 :=
sorry

end length_of_uncovered_side_l305_30576


namespace weight_of_A_l305_30514

theorem weight_of_A (A B C D E : ℝ) 
  (h1 : (A + B + C) / 3 = 84) 
  (h2 : (A + B + C + D) / 4 = 80) 
  (h3 : (B + C + D + E) / 4 = 79) 
  (h4 : E = D + 7): 
  A = 79 := by
  have h5 : A + B + C = 252 := by
    linarith [h1]
  have h6 : A + B + C + D = 320 := by
    linarith [h2]
  have h7 : B + C + D + E = 316 := by
    linarith [h3]
  have hD : D = 68 := by
    linarith [h5, h6]
  have hE : E = 75 := by
    linarith [hD, h4]
  have hBC : B + C = 252 - A := by
    linarith [h5]
  have : 252 - A + 68 + 75 = 316 := by
    linarith [h7, hBC, hD, hE]
  linarith

end weight_of_A_l305_30514


namespace sqrt_sum_ineq_l305_30507

open Real

theorem sqrt_sum_ineq (a b c d : ℝ) (h : a ≥ 0) (h1 : b ≥ 0) (h2 : c ≥ 0) (h3 : d ≥ 0)
  (h4 : a + b + c + d = 4) : 
  sqrt (a + b + c) + sqrt (b + c + d) + sqrt (c + d + a) + sqrt (d + a + b) ≥ 6 :=
sorry

end sqrt_sum_ineq_l305_30507


namespace multiply_24_99_l305_30538

theorem multiply_24_99 : 24 * 99 = 2376 :=
by
  sorry

end multiply_24_99_l305_30538


namespace area_of_curves_l305_30525

noncomputable def enclosed_area : ℝ :=
  ∫ x in (0:ℝ)..1, (Real.sqrt x - x^2)

theorem area_of_curves :
  enclosed_area = 1 / 3 :=
sorry

end area_of_curves_l305_30525


namespace correct_proposition_l305_30511

variables {Point Line Plane : Type}
variables (m n : Line) (α β γ : Plane)

-- Conditions
axiom perpendicular (m : Line) (α : Plane) : Prop
axiom parallel (n : Line) (α : Plane) : Prop

-- Specific conditions given
axiom m_perp_α : perpendicular m α
axiom n_par_α : parallel n α

-- Statement to prove
theorem correct_proposition : perpendicular m n := sorry

end correct_proposition_l305_30511


namespace james_sells_boxes_l305_30531

theorem james_sells_boxes (profit_per_candy_bar : ℝ) (total_profit : ℝ) 
                          (candy_bars_per_box : ℕ) (x : ℕ)
                          (h1 : profit_per_candy_bar = 1.5 - 1)
                          (h2 : total_profit = 25)
                          (h3 : candy_bars_per_box = 10) 
                          (h4 : total_profit = (x * candy_bars_per_box) * profit_per_candy_bar) :
                          x = 5 :=
by
  sorry

end james_sells_boxes_l305_30531


namespace total_pies_eaten_l305_30533

variable (Adam Bill Sierra : ℕ)

axiom condition1 : Adam = Bill + 3
axiom condition2 : Sierra = 2 * Bill
axiom condition3 : Sierra = 12

theorem total_pies_eaten : Adam + Bill + Sierra = 27 :=
by
  -- Sorry is used to skip the proof
  sorry

end total_pies_eaten_l305_30533


namespace smallest_n_satisfying_conditions_l305_30527

variable (n : ℕ)
variable (h1 : 100 ≤ n ∧ n < 1000)
variable (h2 : (n + 7) % 6 = 0)
variable (h3 : (n - 5) % 9 = 0)

theorem smallest_n_satisfying_conditions : n = 113 := by
  sorry

end smallest_n_satisfying_conditions_l305_30527


namespace sum_to_12_of_7_chosen_l305_30563

theorem sum_to_12_of_7_chosen (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}) (T : Finset ℕ) (hT1 : T ⊆ S) (hT2 : T.card = 7) :
  ∃ (a b : ℕ), a ∈ T ∧ b ∈ T ∧ a ≠ b ∧ a + b = 12 :=
by
  sorry

end sum_to_12_of_7_chosen_l305_30563


namespace BobsFruitDrinkCost_l305_30545

theorem BobsFruitDrinkCost 
  (AndySpent : ℕ)
  (BobSpent : ℕ)
  (AndySodaCost : ℕ)
  (AndyHamburgerCost : ℕ)
  (BobSandwichCost : ℕ)
  (FruitDrinkCost : ℕ) :
  AndySpent = 5 ∧ AndySodaCost = 1 ∧ AndyHamburgerCost = 2 ∧ 
  AndySpent = BobSpent ∧ 
  BobSandwichCost = 3 ∧ 
  FruitDrinkCost = BobSpent - BobSandwichCost →
  FruitDrinkCost = 2 := by
  sorry

end BobsFruitDrinkCost_l305_30545


namespace determine_xyz_l305_30588

theorem determine_xyz (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 35)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) : 
  x * y * z = 23 / 3 := 
by { sorry }

end determine_xyz_l305_30588


namespace arithmetic_sequence_thm_l305_30571

theorem arithmetic_sequence_thm
  (a : ℕ → ℝ)
  (h1 : a 1 + a 4 + a 7 = 48)
  (h2 : a 2 + a 5 + a 8 = 40)
  (d : ℝ)
  (h3 : ∀ n, a (n + 1) = a n + d) :
  a 3 + a 6 + a 9 = 32 :=
by {
  sorry
}

end arithmetic_sequence_thm_l305_30571


namespace base_b_sum_correct_l305_30559

def sum_double_digit_numbers (b : ℕ) : ℕ :=
  (b * (b - 1) * (b ^ 2 - b + 1)) / 2

def base_b_sum (b : ℕ) : ℕ :=
  b ^ 2 + 12 * b + 5

theorem base_b_sum_correct : ∃ b : ℕ, sum_double_digit_numbers b = base_b_sum b ∧ b = 15 :=
by
  sorry

end base_b_sum_correct_l305_30559


namespace stock_price_rise_l305_30581

theorem stock_price_rise {P : ℝ} (h1 : P > 0)
    (h2007 : P * 1.20 = 1.20 * P)
    (h2008 : 1.20 * P * 0.75 = P * 0.90)
    (hCertainYear : P * 1.17 = P * 0.90 * (1 + 30 / 100)) :
  30 = 30 :=
by sorry

end stock_price_rise_l305_30581


namespace place_mat_length_l305_30560

theorem place_mat_length (r : ℝ) (n : ℕ) (w : ℝ) (x : ℝ) (inner_touch : Bool)
  (h1 : r = 4)
  (h2 : n = 6)
  (h3 : w = 1)
  (h4 : inner_touch = true)
  : x = (3 * Real.sqrt 7 - Real.sqrt 3) / 2 :=
sorry

end place_mat_length_l305_30560


namespace convert_degrees_to_radians_l305_30510

theorem convert_degrees_to_radians (θ : ℝ) (h : θ = -630) : θ * (Real.pi / 180) = -7 * Real.pi / 2 := by
  sorry

end convert_degrees_to_radians_l305_30510


namespace sculpture_cost_in_CNY_l305_30547

theorem sculpture_cost_in_CNY (USD_to_NAD USD_to_CNY cost_NAD : ℝ) :
  USD_to_NAD = 8 → USD_to_CNY = 5 → cost_NAD = 160 → (cost_NAD * (1 / USD_to_NAD) * USD_to_CNY) = 100 :=
by
  intros h1 h2 h3
  sorry

end sculpture_cost_in_CNY_l305_30547


namespace vector_computation_equiv_l305_30543

variables (u v w : ℤ × ℤ)

def vector_expr (u v w : ℤ × ℤ) :=
  2 • u + 4 • v - 3 • w

theorem vector_computation_equiv :
  u = (3, -5) →
  v = (-1, 6) →
  w = (2, -4) →
  vector_expr u v w = (-4, 26) :=
by
  intros hu hv hw
  rw [hu, hv, hw]
  dsimp [vector_expr]
  -- The actual proof goes here, but we use 'sorry' to skip it.
  sorry

end vector_computation_equiv_l305_30543


namespace QT_value_l305_30517

noncomputable def find_QT (PQ RS PT : ℝ) : ℝ :=
  let tan_gamma := (RS / PQ)
  let QT := (RS / tan_gamma) - PT
  QT

theorem QT_value :
  let PQ := 45
  let RS := 75
  let PT := 15
  find_QT PQ RS PT = 210 := by
  sorry

end QT_value_l305_30517


namespace root_in_interval_iff_a_outside_range_l305_30542

theorem root_in_interval_iff_a_outside_range (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Ioo (-1 : ℝ) 1 ∧ a * x + 1 = 0) ↔ (a < -1 ∨ a > 1) :=
by
  sorry

end root_in_interval_iff_a_outside_range_l305_30542


namespace abe_age_sum_is_31_l305_30526

-- Define the present age of Abe
def abe_present_age : ℕ := 19

-- Define Abe's age 7 years ago
def abe_age_7_years_ago : ℕ := abe_present_age - 7

-- Define the sum of Abe's present age and his age 7 years ago
def abe_age_sum : ℕ := abe_present_age + abe_age_7_years_ago

-- Prove that the sum is 31
theorem abe_age_sum_is_31 : abe_age_sum = 31 := 
by 
  sorry

end abe_age_sum_is_31_l305_30526


namespace value_of_a_plus_b_l305_30520

variables (a b c d x : ℕ)

theorem value_of_a_plus_b : (b + c = 9) → (c + d = 3) → (a + d = 8) → (a + b = x) → x = 14 :=
by
  intros h1 h2 h3 h4
  sorry

end value_of_a_plus_b_l305_30520


namespace gcd_840_1764_l305_30540

def a : ℕ := 840
def b : ℕ := 1764

theorem gcd_840_1764 : Nat.gcd a b = 84 := by
  -- Proof omitted
  sorry

end gcd_840_1764_l305_30540


namespace find_c_l305_30555

theorem find_c (x : ℝ) (c : ℝ) (h1: 3 * x + 6 = 0) (h2: c * x + 15 = 3) : c = 6 := 
by
  sorry

end find_c_l305_30555


namespace log_equation_solution_l305_30535

theorem log_equation_solution (x : ℝ) (h : Real.log x + Real.log (x + 4) = Real.log (2 * x + 8)) : x = 2 :=
sorry

end log_equation_solution_l305_30535


namespace slope_of_l_l305_30505

noncomputable def C (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 4 * Real.sin θ)
noncomputable def l (t α : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, 2 + t * Real.sin α)

theorem slope_of_l
  (α θ₁ θ₂ t₁ t₂ : ℝ)
  (h_midpoint : (C θ₁).fst + (C θ₂).fst = 1 + (t₁ + t₂) * Real.cos α ∧ 
                (C θ₁).snd + (C θ₂).snd = 2 + (t₁ + t₂) * Real.sin α) :
  Real.tan α = -2 :=
by
  sorry

end slope_of_l_l305_30505


namespace percent_increase_sales_l305_30503

-- Define constants for sales
def sales_last_year : ℕ := 320
def sales_this_year : ℕ := 480

-- Define the percent increase formula
def percent_increase (old_value new_value : ℕ) : ℚ :=
  ((new_value - old_value) / old_value) * 100

-- Prove the percent increase from last year to this year is 50%
theorem percent_increase_sales : percent_increase sales_last_year sales_this_year = 50 := by
  sorry

end percent_increase_sales_l305_30503


namespace unequal_numbers_l305_30512

theorem unequal_numbers {k : ℚ} (h : 3 * (1 : ℚ) + 7 * (1 : ℚ) + 2 * k = 0) (d : (7^2 : ℚ) - 4 * 3 * 2 * k = 0) : 
    (3 : ℚ) ≠ (7 : ℚ) ∧ (3 : ℚ) ≠ k ∧ (7 : ℚ) ≠ k :=
by
  -- adding sorry for skipping proof
  sorry

end unequal_numbers_l305_30512


namespace oranges_in_bin_after_changes_l305_30534

def initial_oranges := 31
def thrown_away_oranges := 9
def new_oranges := 38

theorem oranges_in_bin_after_changes : 
  initial_oranges - thrown_away_oranges + new_oranges = 60 := by
  sorry

end oranges_in_bin_after_changes_l305_30534


namespace total_ticket_cost_is_correct_l305_30516

-- Definitions based on the conditions provided
def child_ticket_cost : ℝ := 4.25
def adult_ticket_cost : ℝ := child_ticket_cost + 3.50
def senior_ticket_cost : ℝ := adult_ticket_cost - 1.75

def number_adult_tickets : ℕ := 2
def number_child_tickets : ℕ := 4
def number_senior_tickets : ℕ := 1

def total_ticket_cost_before_discount : ℝ := 
  number_adult_tickets * adult_ticket_cost + 
  number_child_tickets * child_ticket_cost + 
  number_senior_tickets * senior_ticket_cost

def total_tickets : ℕ := number_adult_tickets + number_child_tickets + number_senior_tickets
def discount : ℝ := if total_tickets >= 5 then 3.0 else 0.0

def total_ticket_cost_after_discount : ℝ := total_ticket_cost_before_discount - discount

-- The proof statement: proving the total ticket cost after the discount is $35.50
theorem total_ticket_cost_is_correct : total_ticket_cost_after_discount = 35.50 := by
  -- Note: The exact solution is omitted and replaced with sorry to denote where the proof would be.
  sorry

end total_ticket_cost_is_correct_l305_30516


namespace gcd_of_factorials_l305_30587

-- Define factorials
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define 7!
def seven_factorial : ℕ := factorial 7

-- Define (11! / 4!)
def eleven_div_four_factorial : ℕ := factorial 11 / factorial 4

-- GCD function based on prime factorization (though a direct gcd function also exists, we follow the steps)
def prime_factorization_gcd (a b : ℕ) : ℕ := Nat.gcd a b

-- Proof statement
theorem gcd_of_factorials : prime_factorization_gcd seven_factorial eleven_div_four_factorial = 5040 := by
  sorry

end gcd_of_factorials_l305_30587


namespace netSalePrice_correct_l305_30575

-- Definitions for item costs and fees
def purchaseCostA : ℝ := 650
def handlingFeeA : ℝ := 0.02 * purchaseCostA
def totalCostA : ℝ := purchaseCostA + handlingFeeA

def purchaseCostB : ℝ := 350
def restockingFeeB : ℝ := 0.03 * purchaseCostB
def totalCostB : ℝ := purchaseCostB + restockingFeeB

def purchaseCostC : ℝ := 400
def transportationFeeC : ℝ := 0.015 * purchaseCostC
def totalCostC : ℝ := purchaseCostC + transportationFeeC

-- Desired profit percentages
def profitPercentageA : ℝ := 0.40
def profitPercentageB : ℝ := 0.25
def profitPercentageC : ℝ := 0.30

-- Net sale prices for achieving the desired profit percentages
def netSalePriceA : ℝ := totalCostA + (profitPercentageA * totalCostA)
def netSalePriceB : ℝ := totalCostB + (profitPercentageB * totalCostB)
def netSalePriceC : ℝ := totalCostC + (profitPercentageC * totalCostC)

-- Expected values
def expectedNetSalePriceA : ℝ := 928.20
def expectedNetSalePriceB : ℝ := 450.63
def expectedNetSalePriceC : ℝ := 527.80

-- Theorem to prove the net sale prices match the expected values
theorem netSalePrice_correct :
  netSalePriceA = expectedNetSalePriceA ∧
  netSalePriceB = expectedNetSalePriceB ∧
  netSalePriceC = expectedNetSalePriceC :=
by
  unfold netSalePriceA netSalePriceB netSalePriceC totalCostA totalCostB totalCostC
         handlingFeeA restockingFeeB transportationFeeC
  sorry

end netSalePrice_correct_l305_30575


namespace length_of_each_train_l305_30561

noncomputable def length_of_train : ℝ := 
  let speed_fast := 46 -- in km/hr
  let speed_slow := 36 -- in km/hr
  let relative_speed := speed_fast - speed_slow -- 10 km/hr
  let relative_speed_km_per_sec := relative_speed / 3600.0 -- converting to km/sec
  let time_sec := 18.0 -- time in seconds
  let distance_km := relative_speed_km_per_sec * time_sec -- calculates distance in km
  distance_km * 1000.0 -- converts to meters

theorem length_of_each_train : length_of_train = 50 :=
  by
    sorry

end length_of_each_train_l305_30561


namespace correct_percentage_is_500_over_7_l305_30537

-- Given conditions
variable (x : ℕ)
def total_questions : ℕ := 7 * x
def missed_questions : ℕ := 2 * x

-- Definition of the fraction and percentage calculation
def correct_fraction : ℚ := (total_questions x - missed_questions x : ℕ) / total_questions x
def correct_percentage : ℚ := correct_fraction x * 100

-- The theorem to prove
theorem correct_percentage_is_500_over_7 : correct_percentage x = 500 / 7 :=
by
  -- Proof goes here
  sorry

end correct_percentage_is_500_over_7_l305_30537


namespace point_on_line_l_is_necessary_and_sufficient_for_pa_perpendicular_pb_l305_30573

theorem point_on_line_l_is_necessary_and_sufficient_for_pa_perpendicular_pb
  (x1 x2 : ℝ) : 
  (x1 * x2 / 4 = -1) ↔ ((x1 / 2) * (x2 / 2) = -1) :=
by sorry

end point_on_line_l_is_necessary_and_sufficient_for_pa_perpendicular_pb_l305_30573


namespace corrections_needed_l305_30556

-- Define the corrected statements
def corrected_statements : List String :=
  ["A = 50", "B = A", "x = 1", "y = 2", "z = 3", "INPUT“How old are you?”;x",
   "INPUT x", "PRINT“A+B=”;C", "PRINT“Good-bye!”"]

-- Define the function to check if the statement is correctly formatted
def is_corrected (statement : String) : Prop :=
  statement ∈ corrected_statements

-- Lean theorem statement to prove each original incorrect statement should be correctly formatted
theorem corrections_needed (s : String) (incorrect : s ∈ ["A = B = 50", "x = 1, y = 2, z = 3", 
  "INPUT“How old are you”x", "INPUT, x", "PRINT A+B=;C", "PRINT Good-bye!"]) :
  ∃ t : String, is_corrected t :=
by 
  sorry

end corrections_needed_l305_30556


namespace rank_from_last_l305_30539

theorem rank_from_last (total_students : ℕ) (rank_from_top : ℕ) (rank_from_last : ℕ) : 
  total_students = 35 → 
  rank_from_top = 14 → 
  rank_from_last = (total_students - rank_from_top + 1) → 
  rank_from_last = 22 := 
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end rank_from_last_l305_30539


namespace total_boxes_stacked_l305_30590

/-- Definitions used in conditions --/
def box_width : ℕ := 1
def box_length : ℕ := 1
def land_width : ℕ := 44
def land_length : ℕ := 35
def first_day_layers : ℕ := 7
def second_day_layers : ℕ := 3

/-- Theorem stating the number of boxes stacked in two days --/
theorem total_boxes_stacked : first_day_layers * (land_width * land_length) + second_day_layers * (land_width * land_length) = 15400 := by
  sorry

end total_boxes_stacked_l305_30590


namespace melanie_batches_l305_30564

theorem melanie_batches (total_brownies_given: ℕ)
                        (brownies_per_batch: ℕ)
                        (fraction_bake_sale: ℚ)
                        (fraction_container: ℚ)
                        (remaining_brownies_given: ℕ) :
                        brownies_per_batch = 20 →
                        fraction_bake_sale = 3/4 →
                        fraction_container = 3/5 →
                        total_brownies_given = 20 →
                        (remaining_brownies_given / (brownies_per_batch * (1 - fraction_bake_sale) * (1 - fraction_container))) = 10 :=
by
  sorry

end melanie_batches_l305_30564


namespace final_salt_concentration_is_25_l305_30562

-- Define the initial conditions
def original_solution_weight : ℝ := 100
def original_salt_concentration : ℝ := 0.10
def added_salt_weight : ℝ := 20

-- Define the amount of salt in the original solution
def original_salt_weight := original_solution_weight * original_salt_concentration

-- Define the total amount of salt after adding pure salt
def total_salt_weight := original_salt_weight + added_salt_weight

-- Define the total weight of the new solution
def new_solution_weight := original_solution_weight + added_salt_weight

-- Define the final salt concentration
noncomputable def final_salt_concentration := (total_salt_weight / new_solution_weight) * 100

-- Prove the final salt concentration equals 25%
theorem final_salt_concentration_is_25 : final_salt_concentration = 25 :=
by
  sorry

end final_salt_concentration_is_25_l305_30562


namespace paige_team_total_players_l305_30569

theorem paige_team_total_players 
    (total_points : ℕ)
    (paige_points : ℕ)
    (other_points_per_player : ℕ)
    (other_players : ℕ) :
    total_points = paige_points + other_points_per_player * other_players →
    (other_players + 1) = 6 :=
by
  intros h
  sorry

end paige_team_total_players_l305_30569


namespace power_function_zeros_l305_30595

theorem power_function_zeros :
  ∃ f : ℝ → ℝ, (∀ x : ℝ, f x = x ^ 3) ∧ (f 2 = 8) ∧ (∀ y : ℝ, (f y - y = 0) ↔ (y = 0 ∨ y = 1 ∨ y = -1)) := by
  sorry

end power_function_zeros_l305_30595


namespace find_roots_of_g_l305_30513

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 - a*x - b
noncomputable def g (x : ℝ) (a b : ℝ) : ℝ := b*x^2 - a*x - 1

theorem find_roots_of_g :
  (∀ a b : ℝ, f 2 a b = 0 ∧ f 3 a b = 0 → ∃ (x1 x2 : ℝ), g x1 a b = 0 ∧ g x2 a b = 0 ∧
    (x1 = -1/2 ∨ x1 = -1/3) ∧ (x2 = -1/2 ∨ x2 = -1/3) ∧ x1 ≠ x2) :=
by
  sorry

end find_roots_of_g_l305_30513


namespace how_many_fewer_runs_did_E_score_l305_30578

-- Define the conditions
variables (a b c d e : ℕ)
variable (h1 : 5 * 36 = 180)
variable (h2 : d = e + 5)
variable (h3 : e = 20)
variable (h4 : b = d + e)
variable (h5 : b + c = 107)
variable (h6 : a + b + c + d + e = 180)

-- Specification to be proved
theorem how_many_fewer_runs_did_E_score :
  a - e = 8 :=
by {
  sorry
}

end how_many_fewer_runs_did_E_score_l305_30578


namespace probability_A_and_B_same_last_hour_l305_30580
open Classical

-- Define the problem conditions
def attraction_count : ℕ := 6
def total_scenarios : ℕ := attraction_count * attraction_count
def favorable_scenarios : ℕ := attraction_count

-- Define the probability calculation
def probability_same_attraction : ℚ := favorable_scenarios / total_scenarios

-- The proof problem statement
theorem probability_A_and_B_same_last_hour : 
  probability_same_attraction = 1 / 6 :=
sorry

end probability_A_and_B_same_last_hour_l305_30580


namespace find_abc_l305_30565

theorem find_abc (a b c : ℤ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : (a-1) * (b-1) * (c-1) ∣ a * b * c - 1) :
    (a, b, c) = (3, 5, 15) ∨ (a, b, c) = (2, 4, 8) :=
by
  sorry

end find_abc_l305_30565


namespace cappuccino_cost_l305_30592

theorem cappuccino_cost 
  (total_order_cost drip_price espresso_price latte_price syrup_price cold_brew_price total_other_cost : ℝ)
  (h1 : total_order_cost = 25)
  (h2 : drip_price = 2 * 2.25)
  (h3 : espresso_price = 3.50)
  (h4 : latte_price = 2 * 4.00)
  (h5 : syrup_price = 0.50)
  (h6 : cold_brew_price = 2 * 2.50)
  (h7 : total_other_cost = drip_price + espresso_price + latte_price + syrup_price + cold_brew_price) :
  total_order_cost - total_other_cost = 3.50 := 
by
  sorry

end cappuccino_cost_l305_30592


namespace x_plus_reciprocal_eq_two_implies_x_pow_12_eq_one_l305_30591

theorem x_plus_reciprocal_eq_two_implies_x_pow_12_eq_one {x : ℝ} (h : x + 1 / x = 2) : x^12 = 1 :=
by
  -- The proof will go here, but it is omitted.
  sorry

end x_plus_reciprocal_eq_two_implies_x_pow_12_eq_one_l305_30591


namespace stickers_total_l305_30546

theorem stickers_total (yesterday_packs : ℕ) (increment_packs : ℕ) (today_packs : ℕ) (total_packs : ℕ) :
  yesterday_packs = 15 → increment_packs = 10 → today_packs = yesterday_packs + increment_packs → total_packs = yesterday_packs + today_packs → total_packs = 40 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2] at h3
  rw [h1, h3] at h4
  exact h4

end stickers_total_l305_30546


namespace james_tylenol_intake_per_day_l305_30509

variable (hours_in_day : ℕ := 24) 
variable (tablets_per_dose : ℕ := 2) 
variable (mg_per_tablet : ℕ := 375)
variable (hours_per_dose : ℕ := 6)

theorem james_tylenol_intake_per_day :
  (tablets_per_dose * mg_per_tablet) * (hours_in_day / hours_per_dose) = 3000 := by
  sorry

end james_tylenol_intake_per_day_l305_30509


namespace eggs_for_husband_is_correct_l305_30597

-- Define the conditions
def eggs_per_child : Nat := 2
def num_children : Nat := 4
def eggs_for_herself : Nat := 2
def total_eggs_per_year : Nat := 3380
def days_per_week : Nat := 5
def weeks_per_year : Nat := 52

-- Define the total number of eggs Lisa makes for her husband per year
def eggs_for_husband : Nat :=
  total_eggs_per_year - 
  (num_children * eggs_per_child + eggs_for_herself) * (days_per_week * weeks_per_year)

-- Prove the main statement
theorem eggs_for_husband_is_correct : eggs_for_husband = 780 := by
  sorry

end eggs_for_husband_is_correct_l305_30597


namespace trains_clear_each_other_in_12_seconds_l305_30572

noncomputable def length_train1 : ℕ := 137
noncomputable def length_train2 : ℕ := 163
noncomputable def speed_train1_kmph : ℕ := 42
noncomputable def speed_train2_kmph : ℕ := 48

noncomputable def kmph_to_mps (v : ℕ) : ℚ := v * (5 / 18)
noncomputable def total_distance : ℕ := length_train1 + length_train2
noncomputable def relative_speed_kmph : ℕ := speed_train1_kmph + speed_train2_kmph
noncomputable def relative_speed_mps : ℚ := kmph_to_mps relative_speed_kmph

theorem trains_clear_each_other_in_12_seconds :
  (total_distance : ℚ) / relative_speed_mps = 12 := by
  sorry

end trains_clear_each_other_in_12_seconds_l305_30572


namespace right_triangle_exists_l305_30550

theorem right_triangle_exists (a b c d : ℕ) (h1 : ab = cd) (h2 : a + b = c - d) : 
  ∃ (x y z : ℕ), x * y / 2 = ab ∧ x^2 + y^2 = z^2 :=
sorry

end right_triangle_exists_l305_30550


namespace find_principal_amount_l305_30518

theorem find_principal_amount (P r : ℝ) (h1 : 720 = P * (1 + 2 * r)) (h2 : 1020 = P * (1 + 7 * r)) : P = 600 :=
by sorry

end find_principal_amount_l305_30518


namespace monotonicity_and_range_of_a_l305_30577

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2 * x + a * Real.log x

theorem monotonicity_and_range_of_a (a : ℝ) (t : ℝ) (ht : t ≥ 1) :
  (∀ x, x > 0 → f x a ≥ f t a - 3) → a ≤ 2 := 
sorry

end monotonicity_and_range_of_a_l305_30577


namespace more_trees_in_ahmeds_orchard_l305_30549

-- Given conditions
def ahmed_orange_trees : ℕ := 8
def hassan_apple_trees : ℕ := 1
def hassan_orange_trees : ℕ := 2
def ahmed_apple_trees : ℕ := 4 * hassan_apple_trees
def ahmed_total_trees : ℕ := ahmed_orange_trees + ahmed_apple_trees
def hassan_total_trees : ℕ := hassan_apple_trees + hassan_orange_trees

-- Statement to be proven
theorem more_trees_in_ahmeds_orchard : ahmed_total_trees - hassan_total_trees = 9 :=
by
  sorry

end more_trees_in_ahmeds_orchard_l305_30549


namespace weight_of_grapes_l305_30557

theorem weight_of_grapes :
  ∀ (weight_of_fruits weight_of_apples weight_of_oranges weight_of_strawberries weight_of_grapes : ℕ),
  weight_of_fruits = 10 →
  weight_of_apples = 3 →
  weight_of_oranges = 1 →
  weight_of_strawberries = 3 →
  weight_of_fruits = weight_of_apples + weight_of_oranges + weight_of_strawberries + weight_of_grapes →
  weight_of_grapes = 3 :=
by
  intros
  sorry

end weight_of_grapes_l305_30557


namespace identify_nearly_regular_polyhedra_l305_30568

structure Polyhedron :=
  (faces : ℕ)
  (edges : ℕ)
  (vertices : ℕ)

def nearlyRegularPolyhedra : List Polyhedron :=
  [ 
    ⟨8, 12, 6⟩,   -- Properties of Tetrahedron-octahedron intersection
    ⟨14, 24, 12⟩, -- Properties of Cuboctahedron
    ⟨32, 60, 30⟩  -- Properties of Dodecahedron-Icosahedron
  ]

theorem identify_nearly_regular_polyhedra :
  nearlyRegularPolyhedra = [
    ⟨8, 12, 6⟩,  -- Tetrahedron-octahedron intersection
    ⟨14, 24, 12⟩, -- Cuboctahedron
    ⟨32, 60, 30⟩  -- Dodecahedron-icosahedron intersection
  ] :=
by
  sorry

end identify_nearly_regular_polyhedra_l305_30568


namespace arithmetic_and_geometric_mean_l305_30532

theorem arithmetic_and_geometric_mean (a b : ℝ) (h1 : a + b = 40) (h2 : a * b = 100) : a^2 + b^2 = 1400 := by
  sorry

end arithmetic_and_geometric_mean_l305_30532


namespace time_to_pass_l305_30548
-- Import the Mathlib library

-- Define the lengths of the trains
def length_train1 := 150 -- meters
def length_train2 := 150 -- meters

-- Define the speeds of the trains in km/h
def speed_train1_kmh := 95 -- km/h
def speed_train2_kmh := 85 -- km/h

-- Convert speeds to m/s
def speed_train1_ms := (speed_train1_kmh * 1000) / 3600 -- meters per second
def speed_train2_ms := (speed_train2_kmh * 1000) / 3600 -- meters per second

-- Calculate the relative speed in m/s (since they move in opposite directions, the relative speed is additive)
def relative_speed_ms := speed_train1_ms + speed_train2_ms -- meters per second

-- Calculate the total distance to be covered (sum of the lengths of the trains)
def total_length := length_train1 + length_train2 -- meters

-- State the theorem: the time taken for the trains to pass each other
theorem time_to_pass :
  total_length / relative_speed_ms = 6 := by
  sorry

end time_to_pass_l305_30548


namespace quadratic_roots_quadratic_roots_one_quadratic_roots_two_l305_30594

open scoped Classical

variables {p : Type*} [Field p] {a b c x : p}

theorem quadratic_roots (h_a : a ≠ 0) :
  (¬ ∃ y : p, y^2 = b^2 - 4 * a * c) → ∀ x : p, ¬ a * x^2 + b * x + c = 0 :=
by sorry

theorem quadratic_roots_one (h_a : a ≠ 0) :
  (b^2 - 4 * a * c = 0) → ∃ x : p, a * x^2 + b * x + c = 0 ∧ ∀ y : p, a * y^2 + b * y + c = 0 → y = x :=
by sorry

theorem quadratic_roots_two (h_a : a ≠ 0) :
  (∃ y : p, y^2 = b^2 - 4 * a * c) ∧ (b^2 - 4 * a * c ≠ 0) → ∃ x1 x2 : p, x1 ≠ x2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 :=
by sorry

end quadratic_roots_quadratic_roots_one_quadratic_roots_two_l305_30594


namespace no_infinite_arithmetic_progression_l305_30566

open Classical

variable {R : Type*} [LinearOrderedField R]

noncomputable def f (x : R) : R := sorry

theorem no_infinite_arithmetic_progression
  (f_strict_inc : ∀ x y : R, 0 < x ∧ 0 < y → x < y → f x < f y)
  (f_convex : ∀ x y : R, 0 < x ∧ 0 < y → f ((x + y) / 2) < (f x + f y) / 2) :
  ∀ a : ℕ → R, (∀ n : ℕ, a n = f n) → ¬(∃ d : R, ∀ k : ℕ, a (k + 1) - a k = d) :=
sorry

end no_infinite_arithmetic_progression_l305_30566


namespace two_digit_square_difference_l305_30504

-- Define the problem in Lean
theorem two_digit_square_difference :
  ∃ (X Y : ℕ), (10 ≤ X ∧ X ≤ 99) ∧ (10 ≤ Y ∧ Y ≤ 99) ∧ (X > Y) ∧
  (∃ (t : ℕ), (1 ≤ t ∧ t ≤ 9) ∧ (X^2 - Y^2 = 100 * t)) :=
sorry

end two_digit_square_difference_l305_30504


namespace max_prime_difference_l305_30508

theorem max_prime_difference (a b c d : ℕ) 
  (p1 : Prime a) (p2 : Prime b) (p3 : Prime c) (p4 : Prime d)
  (p5 : Prime (a + b + c + 18 + d)) (p6 : Prime (a + b + c + 18 - d))
  (p7 : Prime (b + c)) (p8 : Prime (c + d))
  (h1 : a + b + c = 2010) (h2 : a ≠ 3) (h3 : b ≠ 3) (h4 : c ≠ 3) (h5 : d ≠ 3) (h6 : d ≤ 50)
  (distinct_primes : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ (a + b + c + 18 + d)
                    ∧ a ≠ (a + b + c + 18 - d) ∧ a ≠ (b + c) ∧ a ≠ (c + d)
                    ∧ b ≠ c ∧ b ≠ d ∧ b ≠ (a + b + c + 18 + d)
                    ∧ b ≠ (a + b + c + 18 - d) ∧ b ≠ (b + c) ∧ b ≠ (c + d)
                    ∧ c ≠ d ∧ c ≠ (a + b + c + 18 + d)
                    ∧ c ≠ (a + b + c + 18 - d) ∧ c ≠ (b + c) ∧ c ≠ (c + d)
                    ∧ d ≠ (a + b + c + 18 + d) ∧ d ≠ (a + b + c + 18 - d)
                    ∧ d ≠ (b + c) ∧ d ≠ (c + d)
                    ∧ (a + b + c + 18 + d) ≠ (a + b + c + 18 - d)
                    ∧ (a + b + c + 18 + d) ≠ (b + c) ∧ (a + b + c + 18 + d) ≠ (c + d)
                    ∧ (a + b + c + 18 - d) ≠ (b + c) ∧ (a + b + c + 18 - d) ≠ (c + d)
                    ∧ (b + c) ≠ (c + d)) :
  ∃ max_diff : ℕ, max_diff = 2067 := sorry

end max_prime_difference_l305_30508


namespace determinant_condition_l305_30552

variable (p q r s : ℝ)

theorem determinant_condition (h: p * s - q * r = 5) :
  p * (5 * r + 4 * s) - r * (5 * p + 4 * q) = 20 :=
by
  sorry

end determinant_condition_l305_30552


namespace find_a7_l305_30567

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

def Sn_for_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

theorem find_a7 (h_arith : arithmetic_sequence a)
  (h_sum_property : Sn_for_arithmetic_sequence a S)
  (h1 : a 2 + a 5 = 4)
  (h2 : S 7 = 21) :
  a 7 = 9 :=
sorry

end find_a7_l305_30567


namespace greatest_divisor_l305_30596

theorem greatest_divisor (n : ℕ) (h1 : 3461 % n = 23) (h2 : 4783 % n = 41) : n = 2 := by {
  sorry
}

end greatest_divisor_l305_30596


namespace starting_player_wins_by_taking_2_white_first_l305_30570

-- Define initial setup
def initial_blue_balls : ℕ := 15
def initial_white_balls : ℕ := 12

-- Define conditions of the game
def can_take_blue_balls (n : ℕ) : Prop := n % 3 = 0
def can_take_white_balls (n : ℕ) : Prop := n % 2 = 0
def player_win_condition (blue white : ℕ) : Prop := 
  (blue = 0 ∧ white = 0)

-- Define the game strategy to establish and maintain the ratio 3/2
def maintain_ratio (blue white : ℕ) : Prop := blue * 2 = white * 3

-- Prove that the starting player should take 2 white balls first to ensure winning
theorem starting_player_wins_by_taking_2_white_first :
  (can_take_white_balls 2) →
  maintain_ratio initial_blue_balls (initial_white_balls - 2) →
  ∀ (blue white : ℕ), player_win_condition blue white :=
by
  intros h_take_white h_maintain_ratio blue white
  sorry

end starting_player_wins_by_taking_2_white_first_l305_30570


namespace beta_max_success_ratio_l305_30583

-- Define Beta's score conditions
variables (a b c d : ℕ)
def beta_score_conditions :=
  (0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) ∧
  (a * 25 < b * 9) ∧
  (c * 25 < d * 17) ∧
  (b + d = 600)

-- Define Beta's success ratio
def beta_success_ratio :=
  (a + c) / 600

theorem beta_max_success_ratio :
  beta_score_conditions a b c d →
  beta_success_ratio a c ≤ 407 / 600 :=
sorry

end beta_max_success_ratio_l305_30583


namespace negation_of_statement_l305_30585

theorem negation_of_statement (h: ∀ x : ℝ, |x| + x^2 ≥ 0) :
  ¬ (∀ x : ℝ, |x| + x^2 ≥ 0) ↔ ∃ x : ℝ, |x| + x^2 < 0 :=
by
  sorry

end negation_of_statement_l305_30585


namespace smallest_five_consecutive_even_sum_320_l305_30554

theorem smallest_five_consecutive_even_sum_320 : ∃ (a b c d e : ℤ), a + b + c + d + e = 320 ∧ (∀ i j : ℤ, (i = a ∨ i = b ∨ i = c ∨ i = d ∨ i = e) → (j = a ∨ j = b ∨ j = c ∨ j = d ∨ j = e) → (i = j + 2 ∨ i = j - 2 ∨ i = j)) ∧ (a ≤ b ∧ a ≤ c ∧ a ≤ d ∧ a ≤ e) ∧ a = 60 :=
by
  sorry

end smallest_five_consecutive_even_sum_320_l305_30554


namespace div_relation_l305_30541

variables (a b c : ℚ)

theorem div_relation (h1 : a / b = 3) (h2 : b / c = 1 / 2) : c / a = 2 / 3 :=
by
  -- proof to be filled in
  sorry

end div_relation_l305_30541


namespace solve_root_equation_l305_30593

noncomputable def sqrt4 (x : ℝ) : ℝ := x^(1/4)

theorem solve_root_equation (x : ℝ) :
  sqrt4 (43 - 2 * x) + sqrt4 (39 + 2 * x) = 4 ↔ x = 21 ∨ x = -13.5 :=
by
  sorry

end solve_root_equation_l305_30593


namespace tom_purchased_8_kg_of_apples_l305_30501

noncomputable def number_of_apples_purchased (price_per_kg_apple : ℤ) (price_per_kg_mango : ℤ) (kg_mangoes : ℤ) (total_paid : ℤ) : ℤ :=
  let total_cost_mangoes := price_per_kg_mango * kg_mangoes
  total_paid - total_cost_mangoes / price_per_kg_apple

theorem tom_purchased_8_kg_of_apples : 
  number_of_apples_purchased 70 65 9 1145 = 8 := 
by {
  -- Expand the definitions and simplify
  sorry
}

end tom_purchased_8_kg_of_apples_l305_30501


namespace price_of_72_cans_l305_30544

def regular_price_per_can : ℝ := 0.60
def discount_percentage : ℝ := 0.20
def total_price : ℝ := 34.56

theorem price_of_72_cans (discounted_price_per_can : ℝ) (number_of_cans : ℕ)
  (H1 : discounted_price_per_can = regular_price_per_can - (discount_percentage * regular_price_per_can))
  (H2 : number_of_cans = total_price / discounted_price_per_can) :
  total_price = number_of_cans * discounted_price_per_can := by
  sorry

end price_of_72_cans_l305_30544


namespace power_point_relative_to_circle_l305_30599

noncomputable def circle_power (a b R x1 y1 : ℝ) : ℝ :=
  (x1 - a) ^ 2 + (y1 - b) ^ 2 - R ^ 2

theorem power_point_relative_to_circle (a b R x1 y1 : ℝ) :
  (x1 - a) ^ 2 + (y1 - b) ^ 2 - R ^ 2 = circle_power a b R x1 y1 := by
  unfold circle_power
  sorry

end power_point_relative_to_circle_l305_30599


namespace f_2_value_l305_30528

variable (a b : ℝ)
def f (x : ℝ) : ℝ := a * x^3 + b * x - 4

theorem f_2_value :
  (f a b (-2)) = 2 → (f a b 2) = -10 :=
by
  intro h
  -- Provide the solution steps here, starting with simplifying the equation. Sorry for now
  sorry

end f_2_value_l305_30528


namespace triangle_side_ratio_range_l305_30586

theorem triangle_side_ratio_range (A B C a b c : ℝ) (h1 : A + 4 * B = 180) (h2 : C = 3 * B) (h3 : 0 < B ∧ B < 45) 
  (h4 : a / b = Real.sin (4 * B) / Real.sin B) : 
  1 < a / b ∧ a / b < 3 := 
sorry

end triangle_side_ratio_range_l305_30586


namespace no_magpies_left_l305_30502

theorem no_magpies_left (initial_magpies killed_magpies : ℕ) (fly_away : Prop):
  initial_magpies = 40 → killed_magpies = 6 → fly_away → ∀ M : ℕ, M = 0 :=
by
  intro h0 h1 h2
  sorry

end no_magpies_left_l305_30502
