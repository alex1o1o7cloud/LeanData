import Mathlib

namespace solve_equation_l1239_123915

theorem solve_equation : ∀ x : ℝ, (2 / (x + 5) = 1 / (3 * x)) → x = 1 :=
by
  intro x
  intro h
  -- The proof would go here
  sorry

end solve_equation_l1239_123915


namespace mean_equality_l1239_123948

theorem mean_equality (x : ℝ) 
  (h : (7 + 9 + 23) / 3 = (16 + x) / 2) : 
  x = 10 := 
sorry

end mean_equality_l1239_123948


namespace sqrt_meaningful_range_l1239_123909

theorem sqrt_meaningful_range (x : ℝ): x + 2 ≥ 0 ↔ x ≥ -2 := by
  sorry

end sqrt_meaningful_range_l1239_123909


namespace georgia_coughs_5_times_per_minute_l1239_123911

-- Definitions
def georgia_coughs_per_minute (G : ℕ) := true
def robert_coughs_per_minute (G : ℕ) := 2 * G
def total_coughs (G : ℕ) := 20 * (G + 2 * G) = 300

-- Theorem to prove
theorem georgia_coughs_5_times_per_minute (G : ℕ) 
  (h1 : georgia_coughs_per_minute G) 
  (h2 : robert_coughs_per_minute G = 2 * G) 
  (h3 : total_coughs G) : G = 5 := 
sorry

end georgia_coughs_5_times_per_minute_l1239_123911


namespace problem_I_problem_II_l1239_123989

theorem problem_I (a b p : ℝ) (F_2 M : ℝ × ℝ)
(h1 : a > b) (h2 : b > 0) (h3 : p > 0)
(h4 : (F_2.1)^2 / a^2 + (F_2.2)^2 / b^2 = 1)
(h5 : M.2^2 = 2 * p * M.1)
(h6 : M.1 = abs (M.2 - F_2.2) - 1)
(h7 : (|F_2.1 - 1|) = 5 / 2) :
    p = 2 ∧ ∃ f : ℝ × ℝ, (f.1)^2 / 9 + (f.2)^2 / 8 = 1 := sorry

theorem problem_II (k m x_0 : ℝ) 
(h8 : k ≠ 0) 
(h9 : m ≠ 0) 
(h10 : km = 1) 
(h11: ∃ A B : ℝ × ℝ, A ≠ B ∧
    (A.2 = k * A.1 + m ∧ B.2 = k * B.1 + m) ∧
    ((A.1)^2 / 9 + (A.2)^2 / 8 = 1) ∧
    ((B.1)^2 / 9 + (B.2)^2 / 8 = 1) ∧
    (x_0 = (A.1 + B.1) / 2)) :
  -1 < x_0 ∧ x_0 < 0 := sorry

end problem_I_problem_II_l1239_123989


namespace solve_equation_l1239_123988

theorem solve_equation (x : ℝ) :
  (16 * x - x^2) / (x + 2) * (x + (16 - x) / (x + 2)) = 60 → x = 4 := by
  sorry

end solve_equation_l1239_123988


namespace batsman_total_score_eq_120_l1239_123990

/-- A batsman's runs calculation including boundaries, sixes, and running between wickets. -/
def batsman_runs_calculation (T : ℝ) : Prop :=
  let runs_from_boundaries := 5 * 4
  let runs_from_sixes := 5 * 6
  let runs_from_total := runs_from_boundaries + runs_from_sixes
  let runs_from_running := 0.5833333333333334 * T
  T = runs_from_total + runs_from_running

theorem batsman_total_score_eq_120 :
  ∃ T : ℝ, batsman_runs_calculation T ∧ T = 120 :=
sorry

end batsman_total_score_eq_120_l1239_123990


namespace number_of_people_liking_at_least_one_activity_l1239_123941

def total_people := 200
def people_like_books := 80
def people_like_songs := 60
def people_like_movies := 30
def people_like_books_and_songs := 25
def people_like_books_and_movies := 15
def people_like_songs_and_movies := 20
def people_like_all_three := 10

theorem number_of_people_liking_at_least_one_activity :
  total_people = 200 →
  people_like_books = 80 →
  people_like_songs = 60 →
  people_like_movies = 30 →
  people_like_books_and_songs = 25 →
  people_like_books_and_movies = 15 →
  people_like_songs_and_movies = 20 →
  people_like_all_three = 10 →
  (people_like_books + people_like_songs + people_like_movies -
   people_like_books_and_songs - people_like_books_and_movies -
   people_like_songs_and_movies + people_like_all_three) = 120 := sorry

end number_of_people_liking_at_least_one_activity_l1239_123941


namespace work_completion_days_l1239_123917

theorem work_completion_days (A B C : ℕ) (A_rate B_rate C_rate : ℚ) :
  A_rate = 1 / 30 → B_rate = 1 / 55 → C_rate = 1 / 45 →
  1 / (A_rate + B_rate + C_rate) = 55 / 4 :=
by
  intro hA hB hC
  rw [hA, hB, hC]
  sorry

end work_completion_days_l1239_123917


namespace typesetter_times_l1239_123973

theorem typesetter_times (α β γ : ℝ) (h1 : 1 / β - 1 / α = 10)
                                        (h2 : 1 / β - 1 / γ = 6)
                                        (h3 : 9 * (α + β) = 10 * (β + γ)) :
    α = 1 / 20 ∧ β = 1 / 30 ∧ γ = 1 / 24 :=
by {
  sorry
}

end typesetter_times_l1239_123973


namespace alberto_spent_more_l1239_123919

noncomputable def alberto_total_before_discount : ℝ := 2457 + 374 + 520
noncomputable def alberto_discount : ℝ := 0.05 * alberto_total_before_discount
noncomputable def alberto_total_after_discount : ℝ := alberto_total_before_discount - alberto_discount

noncomputable def samara_total_before_tax : ℝ := 25 + 467 + 79 + 150
noncomputable def samara_tax : ℝ := 0.07 * samara_total_before_tax
noncomputable def samara_total_after_tax : ℝ := samara_total_before_tax + samara_tax

noncomputable def amount_difference : ℝ := alberto_total_after_discount - samara_total_after_tax

theorem alberto_spent_more : amount_difference = 2411.98 :=
by
  sorry

end alberto_spent_more_l1239_123919


namespace xy_not_z_probability_l1239_123933

theorem xy_not_z_probability :
  let P_X := (1 : ℝ) / 4
  let P_Y := (1 : ℝ) / 3
  let P_not_Z := (3 : ℝ) / 8
  let P := P_X * P_Y * P_not_Z
  P = (1 : ℝ) / 32 :=
by
  -- Definitions based on problem conditions
  let P_X := (1 : ℝ) / 4
  let P_Y := (1 : ℝ) / 3
  let P_not_Z := (3 : ℝ) / 8

  -- Calculate the combined probability
  let P := P_X * P_Y * P_not_Z
  
  -- Check equality with 1/32
  have h : P = (1 : ℝ) / 32 := by sorry
  exact h

end xy_not_z_probability_l1239_123933


namespace find_solutions_l1239_123935

def system_solutions (x y z : ℝ) : Prop :=
  (x + 1) * y * z = 12 ∧
  (y + 1) * z * x = 4 ∧
  (z + 1) * x * y = 4

theorem find_solutions :
  ∃ (x y z : ℝ), system_solutions x y z ∧ ((x = 2 ∧ y = -2 ∧ z = -2) ∨ (x = 1/3 ∧ y = 3 ∧ z = 3)) :=
by
  sorry

end find_solutions_l1239_123935


namespace water_required_l1239_123925

-- Definitions based on the conditions
def balanced_equation : Prop := ∀ (NH4Cl H2O NH4OH HCl : ℕ), NH4Cl + H2O = NH4OH + HCl

-- New problem with the conditions translated into Lean
theorem water_required 
  (h_eq : balanced_equation)
  (n : ℕ)
  (m : ℕ)
  (mole_NH4Cl : n = 2 * m)
  (mole_H2O : m = 2) :
  n = m :=
by
  sorry

end water_required_l1239_123925


namespace min_AB_distance_l1239_123900

theorem min_AB_distance : 
  ∀ (A B : ℝ × ℝ), 
  A ≠ B → 
  ((∃ (m : ℝ), A.2 = m * (A.1 - 1) + 1 ∧ B.2 = m * (B.1 - 1) + 1) ∧ 
    ((A.1 - 2)^2 + (A.2 - 3)^2 = 9) ∧ 
    ((B.1 - 2)^2 + (B.2 - 3)^2 = 9)) → 
  dist A B = 4 :=
sorry

end min_AB_distance_l1239_123900


namespace saved_percentage_is_correct_l1239_123943

def rent : ℝ := 5000
def milk : ℝ := 1500
def groceries : ℝ := 4500
def education : ℝ := 2500
def petrol : ℝ := 2000
def miscellaneous : ℝ := 5200
def amount_saved : ℝ := 2300

noncomputable def total_expenses : ℝ :=
  rent + milk + groceries + education + petrol + miscellaneous

noncomputable def total_salary : ℝ :=
  total_expenses + amount_saved

noncomputable def percentage_saved : ℝ :=
  (amount_saved / total_salary) * 100

theorem saved_percentage_is_correct :
  percentage_saved = 8.846 := by
  sorry

end saved_percentage_is_correct_l1239_123943


namespace payment_n_amount_l1239_123947

def payment_m_n (m n : ℝ) : Prop :=
  m + n = 550 ∧ m = 1.2 * n

theorem payment_n_amount : ∃ n : ℝ, ∀ m : ℝ, payment_m_n m n → n = 250 :=
by
  sorry

end payment_n_amount_l1239_123947


namespace chosen_number_is_121_l1239_123952

theorem chosen_number_is_121 (x : ℤ) (h : 2 * x - 140 = 102) : x = 121 := 
by 
  sorry

end chosen_number_is_121_l1239_123952


namespace number_of_girls_l1239_123912

theorem number_of_girls (n : ℕ) (A : ℝ) 
    (h1 : A = (n * (A + 1) + 55 - 80) / n) : n = 25 :=
by 
  sorry

end number_of_girls_l1239_123912


namespace total_legos_156_l1239_123987

def pyramid_bottom_legos (side_length : Nat) : Nat := side_length * side_length
def pyramid_second_level_legos (length : Nat) (width : Nat) : Nat := length * width
def pyramid_third_level_legos (side_length : Nat) : Nat :=
  let total_legos := (side_length * (side_length + 1)) / 2
  total_legos - 3  -- Subtracting 3 Legos for the corners

def pyramid_fourth_level_legos : Nat := 1

def total_pyramid_legos : Nat :=
  pyramid_bottom_legos 10 +
  pyramid_second_level_legos 8 6 +
  pyramid_third_level_legos 4 +
  pyramid_fourth_level_legos

theorem total_legos_156 : total_pyramid_legos = 156 := by
  sorry

end total_legos_156_l1239_123987


namespace find_abc_value_l1239_123966

variable (a b c : ℝ)
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
variable (h4 : a * b = 30)
variable (h5 : b * c = 54)
variable (h6 : c * a = 45)

theorem find_abc_value : a * b * c = 270 := by
  sorry

end find_abc_value_l1239_123966


namespace differentiable_difference_constant_l1239_123923

variable {R : Type*} [AddCommGroup R] [Module ℝ R]

theorem differentiable_difference_constant (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) (hg : Differentiable ℝ g) 
  (h : ∀ x, fderiv ℝ f x = fderiv ℝ g x) : 
  ∃ C : ℝ, ∀ x, f x - g x = C := 
sorry

end differentiable_difference_constant_l1239_123923


namespace find_incorrect_option_l1239_123927

-- The given conditions from the problem
def incomes : List ℝ := [2, 2.5, 2.5, 2.5, 3, 3, 3, 3, 3, 4, 4, 5, 5, 9, 13]
def mean_incorrect : Prop := (incomes.sum / incomes.length) = 4
def option_incorrect : Prop := ¬ mean_incorrect

-- The goal is to prove that the statement about the mean being 4 is incorrect
theorem find_incorrect_option : option_incorrect := by
  sorry

end find_incorrect_option_l1239_123927


namespace hungarian_olympiad_problem_l1239_123928

-- Define the function A_n as given in the problem
def A (n : ℕ) : ℕ := 5^n + 2 * 3^(n - 1) + 1

-- State the theorem to be proved
theorem hungarian_olympiad_problem (n : ℕ) (h : 0 < n) : 8 ∣ A n :=
by
  sorry

end hungarian_olympiad_problem_l1239_123928


namespace tetrahedron_volume_correct_l1239_123985

noncomputable def tetrahedron_volume (AB : ℝ) (area_ABC : ℝ) (area_ABD : ℝ) (angle_ABD_ABC : ℝ) : ℝ :=
  let h_ABD := (2 * area_ABD) / AB
  let h := h_ABD * Real.sin angle_ABD_ABC
  (1 / 3) * area_ABC * h

theorem tetrahedron_volume_correct:
  tetrahedron_volume 3 15 12 (Real.pi / 6) = 20 :=
by
  sorry

end tetrahedron_volume_correct_l1239_123985


namespace f_is_odd_l1239_123942

open Real

noncomputable def f (x : ℝ) (n : ℕ) : ℝ :=
  (1 + sin x)^(2 * n) - (1 - sin x)^(2 * n)

theorem f_is_odd (n : ℕ) (h : n > 0) : ∀ x : ℝ, f (-x) n = -f x n :=
by
  intros x
  -- Proof goes here
  sorry

end f_is_odd_l1239_123942


namespace calculate_total_feet_in_garden_l1239_123901

-- Define the entities in the problem
def dogs := 6
def feet_per_dog := 4

def ducks := 2
def feet_per_duck := 2

-- Define the total number of feet in the garden
def total_feet_in_garden : Nat :=
  (dogs * feet_per_dog) + (ducks * feet_per_duck)

-- Theorem to state the total number of feet in the garden
theorem calculate_total_feet_in_garden :
  total_feet_in_garden = 28 :=
by
  sorry

end calculate_total_feet_in_garden_l1239_123901


namespace sum_of_circle_areas_l1239_123937

theorem sum_of_circle_areas (r s t : ℝ) 
  (h1 : r + s = 6) 
  (h2 : r + t = 8) 
  (h3 : s + t = 10) : 
  π * r^2 + π * s^2 + π * t^2 = 56 * π := 
by 
  sorry

end sum_of_circle_areas_l1239_123937


namespace power_of_a_l1239_123929

theorem power_of_a (a : ℝ) (h : 1 / a = -1) : a ^ 2023 = -1 := sorry

end power_of_a_l1239_123929


namespace bridge_height_at_distance_l1239_123949

theorem bridge_height_at_distance :
  (∃ (a : ℝ), ∀ (x : ℝ), (x = 25) → (a * x^2 + 25 = 0)) →
  (∀ (x : ℝ), (x = 10) → (-1/25 * x^2 + 25 = 21)) :=
by
  intro h1
  intro x h2
  have h : 625 * (-1 / 25) * (-1 / 25) = -25 := sorry
  sorry

end bridge_height_at_distance_l1239_123949


namespace determine_scores_l1239_123999

variables {M Q S K : ℕ}

theorem determine_scores (h1 : Q > M ∨ K > M) 
                          (h2 : M ≠ K) 
                          (h3 : S ≠ Q) 
                          (h4 : S ≠ M) : 
  (Q, S, M) = (Q, S, M) :=
by
  -- We state the theorem as true
  sorry

end determine_scores_l1239_123999


namespace number_of_ways_to_place_letters_l1239_123954

-- Define the number of letters and mailboxes
def num_letters : Nat := 3
def num_mailboxes : Nat := 5

-- Define the function to calculate the number of ways to place the letters into mailboxes
def count_ways (n : Nat) (m : Nat) : Nat := m ^ n

-- The theorem to prove
theorem number_of_ways_to_place_letters :
  count_ways num_letters num_mailboxes = 5 ^ 3 :=
by
  sorry

end number_of_ways_to_place_letters_l1239_123954


namespace rectangle_sides_l1239_123974

theorem rectangle_sides (a b : ℝ) (h₁ : a < b) (h₂ : a * b = 2 * (a + b)) : a < 4 ∧ b > 4 :=
sorry

end rectangle_sides_l1239_123974


namespace blake_lollipops_count_l1239_123934

theorem blake_lollipops_count (lollipop_cost : ℕ) (choc_cost_per_pack : ℕ) 
  (chocolate_packs : ℕ) (total_paid : ℕ) (change_received : ℕ) 
  (total_spent : ℕ) (total_choc_cost : ℕ) (remaining_amount : ℕ) 
  (lollipop_count : ℕ) : 
  lollipop_cost = 2 →
  choc_cost_per_pack = 4 * lollipop_cost →
  chocolate_packs = 6 →
  total_paid = 6 * 10 →
  change_received = 4 →
  total_spent = total_paid - change_received →
  total_choc_cost = chocolate_packs * choc_cost_per_pack →
  remaining_amount = total_spent - total_choc_cost →
  lollipop_count = remaining_amount / lollipop_cost →
  lollipop_count = 4 := 
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end blake_lollipops_count_l1239_123934


namespace Lexie_age_proof_l1239_123976

variables (L B S : ℕ)

def condition1 : Prop := L = B + 6
def condition2 : Prop := S = 2 * L
def condition3 : Prop := S - B = 14

theorem Lexie_age_proof (h1 : condition1 L B) (h2 : condition2 S L) (h3 : condition3 S B) : L = 8 :=
by
  sorry

end Lexie_age_proof_l1239_123976


namespace draw_contains_chinese_book_l1239_123946

theorem draw_contains_chinese_book
  (total_books : ℕ)
  (chinese_books : ℕ)
  (math_books : ℕ)
  (drawn_books : ℕ)
  (h_total : total_books = 12)
  (h_chinese : chinese_books = 10)
  (h_math : math_books = 2)
  (h_drawn : drawn_books = 3) :
  ∃ n, n ≥ 1 ∧ n ≤ drawn_books ∧ n * (chinese_books/total_books) > 1 := 
  sorry

end draw_contains_chinese_book_l1239_123946


namespace sufficient_not_necessary_l1239_123962

noncomputable def f (x a : ℝ) := x^2 - 2*a*x + 1

def no_real_roots (a : ℝ) : Prop := 4*a^2 - 4 < 0

def non_monotonic_interval (a m : ℝ) : Prop := m < a ∧ a < m + 3

def A := {a : ℝ | -1 < a ∧ a < 1}
def B (m : ℝ) := {a : ℝ | m < a ∧ a < m + 3}

theorem sufficient_not_necessary (x : ℝ) (m : ℝ) :
  (x ∈ A → x ∈ B m) → (A ⊆ B m) ∧ (exists a : ℝ, a ∈ B m ∧ a ∉ A) →
  -2 ≤ m ∧ m ≤ -1 := by 
  sorry

end sufficient_not_necessary_l1239_123962


namespace find_c_l1239_123978

theorem find_c (c : ℝ) (h : ∀ x, 2 < x ∧ x < 6 → -x^2 + c * x + 8 > 0) : c = 8 := 
by
  sorry

end find_c_l1239_123978


namespace exists_multiple_of_power_of_2_with_non_zero_digits_l1239_123977

theorem exists_multiple_of_power_of_2_with_non_zero_digits (n : ℕ) (hn : n ≥ 1) :
  ∃ a : ℕ, (∀ d ∈ a.digits 10, d = 1 ∨ d = 2) ∧ 2^n ∣ a :=
by
  sorry

end exists_multiple_of_power_of_2_with_non_zero_digits_l1239_123977


namespace sandy_painting_area_l1239_123967

theorem sandy_painting_area :
  let wall_height := 10
  let wall_length := 15
  let painting_height := 3
  let painting_length := 5
  let wall_area := wall_height * wall_length
  let painting_area := painting_height * painting_length
  let area_to_paint := wall_area - painting_area
  area_to_paint = 135 := 
by 
  sorry

end sandy_painting_area_l1239_123967


namespace miranda_heels_cost_l1239_123939

theorem miranda_heels_cost (months_saved : ℕ) (savings_per_month : ℕ) (gift_from_sister : ℕ) 
  (h1 : months_saved = 3) (h2 : savings_per_month = 70) (h3 : gift_from_sister = 50) : 
  months_saved * savings_per_month + gift_from_sister = 260 := 
by
  sorry

end miranda_heels_cost_l1239_123939


namespace percentage_decrease_l1239_123970

-- Define the condition given in the problem
def is_increase (pct : ℤ) : Prop := pct > 0
def is_decrease (pct : ℤ) : Prop := pct < 0

-- The main proof statement
theorem percentage_decrease (pct : ℤ) (h : pct = -10) : is_decrease pct :=
by
  sorry

end percentage_decrease_l1239_123970


namespace train_crosses_bridge_in_12_4_seconds_l1239_123956

noncomputable def train_crossing_bridge_time (length_train : ℝ) (speed_train_kmph : ℝ) (length_bridge : ℝ) : ℝ :=
  let speed_train_mps := speed_train_kmph * (1000 / 3600)
  let total_distance := length_train + length_bridge
  total_distance / speed_train_mps

theorem train_crosses_bridge_in_12_4_seconds :
  train_crossing_bridge_time 110 72 138 = 12.4 :=
by
  sorry

end train_crosses_bridge_in_12_4_seconds_l1239_123956


namespace middle_of_three_consecutive_integers_is_60_l1239_123932

theorem middle_of_three_consecutive_integers_is_60 (n : ℤ)
    (h : (n - 1) + n + (n + 1) = 180) : n = 60 := by
  sorry

end middle_of_three_consecutive_integers_is_60_l1239_123932


namespace fraction_six_power_l1239_123916

theorem fraction_six_power (n : ℕ) (hyp : n = 6 ^ 2024) : n / 6 = 6 ^ 2023 :=
by sorry

end fraction_six_power_l1239_123916


namespace arithmetic_formula_geometric_formula_comparison_S_T_l1239_123921

noncomputable def a₁ : ℕ := 16
noncomputable def d : ℤ := -3

def a_n (n : ℕ) : ℤ := -3 * (n : ℤ) + 19
def b_n (n : ℕ) : ℤ := 4^(3 - n)

def S_n (n : ℕ) : ℚ := (-3 * (n : ℚ)^2 + 35 * n) / 2
def T_n (n : ℕ) : ℤ := -n^2 + 3 * n

theorem arithmetic_formula (n : ℕ) : a_n n = -3 * n + 19 :=
sorry

theorem geometric_formula (n : ℕ) : b_n n = 4^(3 - n) :=
sorry

theorem comparison_S_T (n : ℕ) :
  if n = 29 then S_n n = (T_n n : ℚ)
  else if n < 29 then S_n n > (T_n n : ℚ)
  else S_n n < (T_n n : ℚ) :=
sorry

end arithmetic_formula_geometric_formula_comparison_S_T_l1239_123921


namespace john_cards_l1239_123907

theorem john_cards (C : ℕ) (h1 : 15 * 2 + C * 2 = 70) : C = 20 :=
by
  sorry

end john_cards_l1239_123907


namespace inequality_system_solution_l1239_123996

theorem inequality_system_solution (x : ℝ) : x + 1 > 0 → x - 3 > 0 → x > 3 :=
by
  intros h1 h2
  sorry

end inequality_system_solution_l1239_123996


namespace decorations_cost_correct_l1239_123913

def cost_of_decorations (num_tables : ℕ) (cost_tablecloth per_tablecloth : ℕ) (num_place_settings per_table : ℕ) (cost_place_setting per_setting : ℕ) (num_roses per_centerpiece : ℕ) (cost_rose per_rose : ℕ) (num_lilies per_centerpiece : ℕ) (cost_lily per_lily : ℕ) : ℕ :=
  let cost_roses := cost_rose * num_roses
  let cost_lilies := cost_lily * num_lilies
  let cost_settings := cost_place_setting * num_place_settings
  let cost_per_table := cost_roses + cost_lilies + cost_settings + cost_tablecloth
  num_tables * cost_per_table

theorem decorations_cost_correct :
  cost_of_decorations 20 25 4 10 10 5 15 4 = 3500 :=
by
  sorry

end decorations_cost_correct_l1239_123913


namespace percent_defective_units_shipped_for_sale_l1239_123982

variable (total_units : ℕ)
variable (defective_units_percentage : ℝ := 0.08)
variable (shipped_defective_units_percentage : ℝ := 0.05)

theorem percent_defective_units_shipped_for_sale :
  defective_units_percentage * shipped_defective_units_percentage * 100 = 0.4 :=
by
  sorry

end percent_defective_units_shipped_for_sale_l1239_123982


namespace expressions_equal_l1239_123957

variable (a b c : ℝ)

theorem expressions_equal (h : a + 2 * b + 2 * c = 0) : a + 2 * b * c = (a + 2 * b) * (a + 2 * c) := 
by 
  sorry

end expressions_equal_l1239_123957


namespace initial_necklaces_count_l1239_123940

theorem initial_necklaces_count (N : ℕ) 
  (h1 : N - 13 = 37) : 
  N = 50 := 
by
  sorry

end initial_necklaces_count_l1239_123940


namespace find_second_number_l1239_123959

theorem find_second_number
  (first_number : ℕ)
  (second_number : ℕ)
  (h1 : first_number = 45)
  (h2 : first_number / second_number = 5) : second_number = 9 :=
by
  -- Proof goes here
  sorry

end find_second_number_l1239_123959


namespace not_perfect_square_of_divisor_l1239_123936

theorem not_perfect_square_of_divisor (n d : ℕ) (hn : 0 < n) (hd : d ∣ 2 * n^2) :
  ¬ ∃ x : ℕ, n^2 + d = x^2 :=
by
  sorry

end not_perfect_square_of_divisor_l1239_123936


namespace r_investment_time_l1239_123994

variables (P Q R Profit_p Profit_q Profit_r Tp Tq Tr : ℕ)
variables (h1 : P / Q = 7 / 5)
variables (h2 : Q / R = 5 / 4)
variables (h3 : Profit_p / Profit_q = 7 / 10)
variables (h4 : Profit_p / Profit_r = 7 / 8)
variables (h5 : Tp = 2)
variables (h6 : Tq = t)

theorem r_investment_time (t : ℕ) :
  ∃ Tr : ℕ, Tr = 4 :=
sorry

end r_investment_time_l1239_123994


namespace find_c_l1239_123998

theorem find_c (a b c d : ℕ) (h1 : 8 = 4 * a / 100) (h2 : 4 = d * a / 100) (h3 : 8 = d * b / 100) (h4 : c = b / a) : 
  c = 2 := 
by
  sorry

end find_c_l1239_123998


namespace giant_exponent_modulo_result_l1239_123980

theorem giant_exponent_modulo_result :
  (7 ^ (7 ^ (7 ^ 7))) % 500 = 343 :=
by sorry

end giant_exponent_modulo_result_l1239_123980


namespace range_of_a_l1239_123965

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - a*x + 2*a > 0) → 0 < a ∧ a < 8 := 
sorry

end range_of_a_l1239_123965


namespace first_five_terms_series_l1239_123972

theorem first_five_terms_series (a : ℕ → ℚ) (h : ∀ n, a n = 1 / (n * (n + 1))) :
  (a 1 = 1 / 2) ∧
  (a 2 = 1 / 6) ∧
  (a 3 = 1 / 12) ∧
  (a 4 = 1 / 20) ∧
  (a 5 = 1 / 30) :=
by
  sorry

end first_five_terms_series_l1239_123972


namespace calculate_expression_l1239_123961

def f (x : ℕ) : ℕ := x^2 - 3*x + 4
def g (x : ℕ) : ℕ := 2*x + 1

theorem calculate_expression : f (g 3) - g (f 3) = 23 := by
  sorry

end calculate_expression_l1239_123961


namespace single_burger_cost_l1239_123953

theorem single_burger_cost
  (total_cost : ℝ)
  (total_hamburgers : ℕ)
  (double_burgers : ℕ)
  (cost_double_burger : ℝ)
  (remaining_cost : ℝ)
  (single_burgers : ℕ)
  (cost_single_burger : ℝ) :
  total_cost = 64.50 ∧
  total_hamburgers = 50 ∧
  double_burgers = 29 ∧
  cost_double_burger = 1.50 ∧
  remaining_cost = total_cost - (double_burgers * cost_double_burger) ∧
  single_burgers = total_hamburgers - double_burgers ∧
  cost_single_burger = remaining_cost / single_burgers →
  cost_single_burger = 1.00 :=
by
  sorry

end single_burger_cost_l1239_123953


namespace trigonometric_expression_l1239_123975

theorem trigonometric_expression
  (α : ℝ)
  (h1 : Real.sin α = 3 / 5)
  (h2 : α ∈ Set.Ioo (π / 2) π) :
  (Real.cos (2 * α) / (Real.sqrt 2 * Real.sin (α + π / 4))) = -7 / 5 := 
sorry

end trigonometric_expression_l1239_123975


namespace cuboid_third_face_area_l1239_123905

-- Problem statement in Lean
theorem cuboid_third_face_area (l w h : ℝ) (A₁ A₂ V : ℝ) 
  (hw1 : l * w = 120)
  (hw2 : w * h = 60)
  (hw3 : l * w * h = 720) : 
  l * h = 72 :=
sorry

end cuboid_third_face_area_l1239_123905


namespace fill_tank_in_12_minutes_l1239_123931

theorem fill_tank_in_12_minutes (rate1 rate2 rate_out : ℝ) 
  (h1 : rate1 = 1 / 18) (h2 : rate2 = 1 / 20) (h_out : rate_out = 1 / 45) : 
  12 = 1 / (rate1 + rate2 - rate_out) :=
by
  -- sorry will be replaced with the actual proof.
  sorry

end fill_tank_in_12_minutes_l1239_123931


namespace max_ab_min_fraction_l1239_123930

-- Question 1: Maximum value of ab
theorem max_ab (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 3 * a + 7 * b = 10) : ab ≤ 25/21 := sorry

-- Question 2: Minimum value of (3/a + 7/b)
theorem min_fraction (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 3 * a + 7 * b = 10) : 3/a + 7/b ≥ 10 := sorry

end max_ab_min_fraction_l1239_123930


namespace evaluate_expression_l1239_123964

theorem evaluate_expression (a b c : ℝ) (h : a / (30 - a) + b / (70 - b) + c / (55 - c) = 8) : 
  6 / (30 - a) + 14 / (70 - b) + 11 / (55 - c) = 2.2 :=
by 
  sorry

end evaluate_expression_l1239_123964


namespace students_use_red_color_l1239_123903

theorem students_use_red_color
  (total_students : ℕ)
  (students_use_green : ℕ)
  (students_use_both : ℕ)
  (total_students_eq : total_students = 70)
  (students_use_green_eq : students_use_green = 52)
  (students_use_both_eq : students_use_both = 38) :
  ∃ (students_use_red : ℕ), students_use_red = 56 :=
by
  -- We will skip the proof part as specified
  sorry

end students_use_red_color_l1239_123903


namespace quadratic_reciprocal_sum_l1239_123926

theorem quadratic_reciprocal_sum :
  ∃ (x1 x2 : ℝ), (x1^2 - 5 * x1 + 4 = 0) ∧ (x2^2 - 5 * x2 + 4 = 0) ∧ (x1 ≠ x2) ∧ (x1 + x2 = 5) ∧ (x1 * x2 = 4) ∧ (1 / x1 + 1 / x2 = 5 / 4) :=
sorry

end quadratic_reciprocal_sum_l1239_123926


namespace integral_value_l1239_123910

theorem integral_value : ∫ x in (1:ℝ)..(2:ℝ), (x^2 + 1) / x = (3 / 2) + Real.log 2 :=
by sorry

end integral_value_l1239_123910


namespace find_a_share_l1239_123968

noncomputable def total_investment (a b c : ℕ) : ℕ :=
  a + b + c

noncomputable def total_profit (b_share total_inv b_inv : ℕ) : ℕ :=
  b_share * total_inv / b_inv

noncomputable def a_share (a_inv total_inv total_pft : ℕ) : ℕ :=
  a_inv * total_pft / total_inv

theorem find_a_share
  (a_inv b_inv c_inv b_share : ℕ)
  (h1 : a_inv = 7000)
  (h2 : b_inv = 11000)
  (h3 : c_inv = 18000)
  (h4 : b_share = 880) :
  a_share a_inv (total_investment a_inv b_inv c_inv) (total_profit b_share (total_investment a_inv b_inv c_inv) b_inv) = 560 := 
by
  sorry

end find_a_share_l1239_123968


namespace sum_of_first_seven_primes_mod_eighth_prime_l1239_123955

theorem sum_of_first_seven_primes_mod_eighth_prime :
  (2 + 3 + 5 + 7 + 11 + 13 + 17) % 19 = 1 :=
by
  sorry

end sum_of_first_seven_primes_mod_eighth_prime_l1239_123955


namespace induction_step_divisibility_l1239_123995

theorem induction_step_divisibility {x y : ℤ} (k : ℕ) (h : ∀ n, n = 2*k - 1 → (x^n + y^n) % (x+y) = 0) :
  (x^(2*k+1) + y^(2*k+1)) % (x+y) = 0 :=
sorry

end induction_step_divisibility_l1239_123995


namespace loss_percentage_is_20_l1239_123960

-- Define necessary conditions
def CP : ℕ := 2000
def gain_percent : ℕ := 6
def SP_new : ℕ := CP + ((gain_percent * CP) / 100)
def increase : ℕ := 520

-- Define the selling price condition
def SP : ℕ := SP_new - increase

-- Define the loss percentage condition
def loss_percent : ℕ := ((CP - SP) * 100) / CP

-- Prove the loss percentage is 20%
theorem loss_percentage_is_20 : loss_percent = 20 :=
by sorry

end loss_percentage_is_20_l1239_123960


namespace muffin_cost_is_correct_l1239_123984

variable (M : ℝ)

def total_original_cost (muffin_cost : ℝ) : ℝ := 3 * muffin_cost + 1.45

def discounted_cost (original_cost : ℝ) : ℝ := 0.85 * original_cost

def kevin_paid (discounted_price : ℝ) : Prop := discounted_price = 3.70

theorem muffin_cost_is_correct (h : discounted_cost (total_original_cost M) = 3.70) : M = 0.97 :=
  by
  sorry

end muffin_cost_is_correct_l1239_123984


namespace bumper_car_rides_correct_l1239_123993

def tickets_per_ride : ℕ := 7
def total_tickets : ℕ := 63
def ferris_wheel_rides : ℕ := 5

def tickets_for_bumper_cars : ℕ :=
  total_tickets - ferris_wheel_rides * tickets_per_ride

def bumper_car_rides : ℕ :=
  tickets_for_bumper_cars / tickets_per_ride

theorem bumper_car_rides_correct : bumper_car_rides = 4 :=
by
  sorry

end bumper_car_rides_correct_l1239_123993


namespace num_solutions_of_system_eq_two_l1239_123945

theorem num_solutions_of_system_eq_two : 
  (∃ n : ℕ, n = 2 ∧ ∀ (x y : ℝ), 
    5 * y - 3 * x = 15 ∧ x^2 + y^2 ≤ 16 ↔ 
    (x, y) = ((-90 + Real.sqrt 31900) / 68, 3 * ((-90 + Real.sqrt 31900) / 68) / 5 + 3) ∨ 
    (x, y) = ((-90 - Real.sqrt 31900) / 68, 3 * ((-90 - Real.sqrt 31900) / 68) / 5 + 3)) :=
sorry

end num_solutions_of_system_eq_two_l1239_123945


namespace yield_difference_correct_l1239_123902

noncomputable def tomato_yield (initial : ℝ) (growth_rate : ℝ) : ℝ := initial * (1 + growth_rate / 100)
noncomputable def corn_yield (initial : ℝ) (growth_rate : ℝ) : ℝ := initial * (1 + growth_rate / 100)
noncomputable def onion_yield (initial : ℝ) (growth_rate : ℝ) : ℝ := initial * (1 + growth_rate / 100)
noncomputable def carrot_yield (initial : ℝ) (growth_rate : ℝ) : ℝ := initial * (1 + growth_rate / 100)

theorem yield_difference_correct :
  let tomato_initial := 2073
  let corn_initial := 4112
  let onion_initial := 985
  let carrot_initial := 6250
  let tomato_growth := 12
  let corn_growth := 15
  let onion_growth := 8
  let carrot_growth := 10
  let tomato_total := tomato_yield tomato_initial tomato_growth
  let corn_total := corn_yield corn_initial corn_growth
  let onion_total := onion_yield onion_initial onion_growth
  let carrot_total := carrot_yield carrot_initial carrot_growth
  let highest_yield := max (max tomato_total corn_total) (max onion_total carrot_total)
  let lowest_yield := min (min tomato_total corn_total) (min onion_total carrot_total)
  highest_yield - lowest_yield = 5811.2 := by
  sorry

end yield_difference_correct_l1239_123902


namespace reduced_price_per_dozen_bananas_l1239_123958

noncomputable def original_price (P : ℝ) := P
noncomputable def reduced_price_one_banana (P : ℝ) := 0.60 * P
noncomputable def number_bananas_original (P : ℝ) := 40 / P
noncomputable def number_bananas_reduced (P : ℝ) := 40 / (0.60 * P)
noncomputable def difference_bananas (P : ℝ) := (number_bananas_reduced P) - (number_bananas_original P)

theorem reduced_price_per_dozen_bananas 
  (P : ℝ) 
  (h1 : difference_bananas P = 67) 
  (h2 : P = 16 / 40.2) :
  12 * reduced_price_one_banana P = 2.856 :=
sorry

end reduced_price_per_dozen_bananas_l1239_123958


namespace bert_money_left_l1239_123979

theorem bert_money_left
  (initial_amount : ℝ)
  (spent_hardware_store_fraction : ℝ)
  (amount_spent_dry_cleaners : ℝ)
  (spent_grocery_store_fraction : ℝ)
  (final_amount : ℝ) :
  initial_amount = 44 →
  spent_hardware_store_fraction = 1/4 →
  amount_spent_dry_cleaners = 9 →
  spent_grocery_store_fraction = 1/2 →
  final_amount = initial_amount - (spent_hardware_store_fraction * initial_amount) - amount_spent_dry_cleaners - (spent_grocery_store_fraction * (initial_amount - (spent_hardware_store_fraction * initial_amount) - amount_spent_dry_cleaners)) →
  final_amount = 12 :=
by
  sorry

end bert_money_left_l1239_123979


namespace gretel_hansel_salary_difference_l1239_123906

theorem gretel_hansel_salary_difference :
  let hansel_initial_salary := 30000
  let hansel_raise_percentage := 10
  let gretel_initial_salary := 30000
  let gretel_raise_percentage := 15
  let hansel_new_salary := hansel_initial_salary + (hansel_raise_percentage / 100 * hansel_initial_salary)
  let gretel_new_salary := gretel_initial_salary + (gretel_raise_percentage / 100 * gretel_initial_salary)
  gretel_new_salary - hansel_new_salary = 1500 := sorry

end gretel_hansel_salary_difference_l1239_123906


namespace correct_sequence_l1239_123920

def step1 := "Collect the admission ticket"
def step2 := "Register"
def step3 := "Written and computer-based tests"
def step4 := "Photography"

theorem correct_sequence : [step2, step4, step1, step3] = ["Register", "Photography", "Collect the admission ticket", "Written and computer-based tests"] :=
by
  sorry

end correct_sequence_l1239_123920


namespace solve_equation_l1239_123986

theorem solve_equation (n : ℝ) :
  (3 - 2 * n) / (n + 2) + (3 * n - 9) / (3 - 2 * n) = 2 ↔ 
  n = (25 + Real.sqrt 13) / 18 ∨ n = (25 - Real.sqrt 13) / 18 :=
by
  sorry

end solve_equation_l1239_123986


namespace sum_of_ages_l1239_123924

-- Define Henry's and Jill's present ages
def Henry_age : ℕ := 23
def Jill_age : ℕ := 17

-- Define the condition that 11 years ago, Henry was twice the age of Jill
def condition_11_years_ago : Prop := (Henry_age - 11) = 2 * (Jill_age - 11)

-- Theorem statement: sum of Henry's and Jill's present ages is 40
theorem sum_of_ages : Henry_age + Jill_age = 40 :=
by
  -- Placeholder for proof
  sorry

end sum_of_ages_l1239_123924


namespace problem_statement_l1239_123950

variable {A B C D E F H : Point}
variable {a b c : ℝ}

-- Assume the conditions
variable (h_triangle : Triangle A B C)
variable (h_acute : AcuteTriangle h_triangle)
variable (h_altitudes : AltitudesIntersectAt h_triangle H A D B E C F)
variable (h_sides : Sides h_triangle BC a AC b AB c)

-- Statement to prove
theorem problem_statement : AH * AD + BH * BE + CH * CF = 1/2 * (a^2 + b^2 + c^2) :=
sorry

end problem_statement_l1239_123950


namespace jordan_walk_distance_l1239_123922

theorem jordan_walk_distance
  (d t : ℝ)
  (flat_speed uphill_speed walk_speed : ℝ)
  (total_time : ℝ)
  (h1 : flat_speed = 18)
  (h2 : uphill_speed = 6)
  (h3 : walk_speed = 4)
  (h4 : total_time = 3)
  (h5 : d / (3 * 18) + d / (3 * 6) + d / (3 * 4) = total_time) :
  t = 6.6 :=
by
  -- Proof goes here
  sorry

end jordan_walk_distance_l1239_123922


namespace cougar_sleep_hours_l1239_123914

-- Definitions
def total_sleep_hours (C Z : Nat) : Prop :=
  C + Z = 70

def zebra_cougar_difference (C Z : Nat) : Prop :=
  Z = C + 2

-- Theorem statement
theorem cougar_sleep_hours :
  ∃ C : Nat, ∃ Z : Nat, zebra_cougar_difference C Z ∧ total_sleep_hours C Z ∧ C = 34 :=
sorry

end cougar_sleep_hours_l1239_123914


namespace equilateral_triangle_l1239_123971

theorem equilateral_triangle
  (a b c : ℝ) (α β γ : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b)
  (h7 : α + β + γ = π)
  (h8 : a = 2 * Real.sin α)
  (h9 : b = 2 * Real.sin β)
  (h10 : c = 2 * Real.sin γ)
  (h11 : a * (1 - 2 * Real.cos α) + b * (1 - 2 * Real.cos β) + c * (1 - 2 * Real.cos γ) = 0) :
  a = b ∧ b = c :=
sorry

end equilateral_triangle_l1239_123971


namespace shortest_altitude_l1239_123997

theorem shortest_altitude (a b c : ℕ) (h1 : a = 12) (h2 : b = 16) (h3 : c = 20) (h4 : a^2 + b^2 = c^2) : ∃ x, x = 9.6 :=
by
  sorry

end shortest_altitude_l1239_123997


namespace donny_cost_of_apples_l1239_123944

def cost_of_apples (small_cost medium_cost big_cost : ℝ) (n_small n_medium n_big : ℕ) : ℝ := 
  n_small * small_cost + n_medium * medium_cost + n_big * big_cost

theorem donny_cost_of_apples :
  cost_of_apples 1.5 2 3 6 6 8 = 45 :=
by
  sorry

end donny_cost_of_apples_l1239_123944


namespace priyas_fathers_age_l1239_123951

-- Define Priya's age P and her father's age F
variables (P F : ℕ)

-- Define the conditions
def conditions : Prop :=
  F - P = 31 ∧ P + F = 53

-- Define the theorem to be proved
theorem priyas_fathers_age (h : conditions P F) : F = 42 :=
sorry

end priyas_fathers_age_l1239_123951


namespace BothNormal_l1239_123908

variable (Normal : Type) (Person : Type) (MrA MrsA : Person)
variables (isNormal : Person → Prop)

-- Conditions given in the problem
axiom MrA_statement : ∀ p : Person, p = MrsA → isNormal MrA → isNormal MrsA
axiom MrsA_statement : ∀ p : Person, p = MrA → isNormal MrsA → isNormal MrA

-- Question (translated to proof problem): 
-- prove that Mr. A and Mrs. A are both normal persons
theorem BothNormal : isNormal MrA ∧ isNormal MrsA := 
  by 
    sorry -- proof is omitted

end BothNormal_l1239_123908


namespace children_got_off_bus_l1239_123963

theorem children_got_off_bus (initial : ℕ) (got_on : ℕ) (after : ℕ) : Prop :=
  initial = 22 ∧ got_on = 40 ∧ after = 2 → initial + got_on - 60 = after


end children_got_off_bus_l1239_123963


namespace greatest_whole_number_solution_l1239_123983

theorem greatest_whole_number_solution :
  ∃ (x : ℕ), (5 * x - 4 < 3 - 2 * x) ∧ ∀ (y : ℕ), (5 * y - 4 < 3 - 2 * y) → y ≤ x ∧ x = 0 :=
by
  sorry

end greatest_whole_number_solution_l1239_123983


namespace fraction_zero_l1239_123991

theorem fraction_zero (x : ℝ) (h : x ≠ 1) (h₁ : (x + 1) / (x - 1) = 0) : x = -1 :=
sorry

end fraction_zero_l1239_123991


namespace equation_of_parallel_line_l1239_123938

noncomputable def line_parallel_and_intercept (m : ℝ) : Prop :=
  (∃ x y : ℝ, x + y + 2 = 0) ∧ (∃ z : ℝ, 3*z + m = 0)

theorem equation_of_parallel_line {m : ℝ} :
  line_parallel_and_intercept m ↔ (∃ x y : ℝ, x + y + 2 = 0) ∨ (∃ x y : ℝ, x + y - 2 = 0) :=
by
  sorry

end equation_of_parallel_line_l1239_123938


namespace min_value_range_l1239_123918

theorem min_value_range:
  ∀ (x m n : ℝ), 
    (y = (3 * x + 2) / (x - 1)) → 
    (∀ x ∈ Set.Ioo m n, y ≥ 3 + 5 / (x - 1)) → 
    (y = 8) → 
    n = 2 → 
    (1 ≤ m ∧ m < 2) := by
  sorry

end min_value_range_l1239_123918


namespace find_m_l1239_123981

-- Define the conditions
variables {m x1 x2 : ℝ}

-- Given the equation x^2 + mx - 1 = 0 has roots x1 and x2:
-- The sum of the roots x1 + x2 is -m, and the product of the roots x1 * x2 is -1.
-- Furthermore, given that 1/x1 + 1/x2 = -3,
-- Prove that m = -3.

theorem find_m :
  (x1 + x2 = -m) →
  (x1 * x2 = -1) →
  (1 / x1 + 1 / x2 = -3) →
  m = -3 := by
  intros hSum hProd hRecip
  sorry

end find_m_l1239_123981


namespace smallest_value_of_diff_l1239_123904

-- Definitions of the side lengths from the conditions
def XY (x : ℝ) := x + 6
def XZ (x : ℝ) := 4 * x - 1
def YZ (x : ℝ) := x + 10

-- Conditions derived from the problem
noncomputable def valid_x (x : ℝ) := x > 5 / 3 ∧ x < 11 / 3

-- The proof statement
theorem smallest_value_of_diff : 
  ∀ (x : ℝ), valid_x x → (YZ x - XY x) = 4 :=
by
  intros x hx
  -- Proof goes here
  sorry

end smallest_value_of_diff_l1239_123904


namespace does_not_pass_through_third_quadrant_l1239_123992

noncomputable def f (a b x : ℝ) : ℝ := a^x + b - 1

theorem does_not_pass_through_third_quadrant (a b : ℝ) (h_a : 0 < a ∧ a < 1) (h_b : 0 < b ∧ b < 1) :
  ¬ ∃ x, f a b x < 0 ∧ x < 0 := sorry

end does_not_pass_through_third_quadrant_l1239_123992


namespace sum_lent_correct_l1239_123969

noncomputable section

-- Define the principal amount (sum lent)
def P : ℝ := 4464.29

-- Define the interest rate per annum
def R : ℝ := 12.0

-- Define the time period in years
def T : ℝ := 12.0

-- Define the interest after 12 years (using the initial conditions and results)
def I : ℝ := 1.44 * P

-- Define the interest given as "2500 less than double the sum lent" condition
def I_condition : ℝ := 2 * P - 2500

-- Theorem stating the sum lent is the given value P
theorem sum_lent_correct : P = 4464.29 :=
by
  -- Placeholder for the proof
  sorry

end sum_lent_correct_l1239_123969
