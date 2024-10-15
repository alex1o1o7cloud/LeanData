import Mathlib

namespace NUMINAMATH_GPT_largest_invertible_interval_l548_54822

def g (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 4

theorem largest_invertible_interval (x : ℝ) (hx : x = 2) : 
  ∃ I : Set ℝ, (I = Set.univ ∩ {y | y ≥ 3 / 2}) ∧ ∀ y ∈ I, g y = 3 * (y - 3 / 2) ^ 2 - 11 / 4 ∧ g y ∈ I ∧ Function.Injective (g ∘ (fun z => z : I → ℝ)) :=
sorry

end NUMINAMATH_GPT_largest_invertible_interval_l548_54822


namespace NUMINAMATH_GPT_Gunther_typing_correct_l548_54827

def GuntherTypingProblem : Prop :=
  let first_phase := (160 * (120 / 3))
  let second_phase := (200 * (180 / 3))
  let third_phase := (50 * 60)
  let fourth_phase := (140 * (90 / 3))
  let total_words := first_phase + second_phase + third_phase + fourth_phase
  total_words = 26200

theorem Gunther_typing_correct : GuntherTypingProblem := by
  sorry

end NUMINAMATH_GPT_Gunther_typing_correct_l548_54827


namespace NUMINAMATH_GPT_sticks_per_chair_l548_54858

-- defining the necessary parameters and conditions
def sticksPerTable := 9
def sticksPerStool := 2
def sticksPerHour := 5
def chairsChopped := 18
def tablesChopped := 6
def stoolsChopped := 4
def hoursKeptWarm := 34

-- calculation of total sticks needed
def totalSticksNeeded := sticksPerHour * hoursKeptWarm

-- the main theorem to prove the number of sticks a chair makes
theorem sticks_per_chair (C : ℕ) : (chairsChopped * C) + (tablesChopped * sticksPerTable) + (stoolsChopped * sticksPerStool) = totalSticksNeeded → C = 6 := by
  sorry

end NUMINAMATH_GPT_sticks_per_chair_l548_54858


namespace NUMINAMATH_GPT_negation_proposition_l548_54859

theorem negation_proposition :
  (¬ (∀ x : ℝ, x ≥ 0)) ↔ (∃ x : ℝ, x < 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_l548_54859


namespace NUMINAMATH_GPT_max_value_expr_max_l548_54852

noncomputable def max_value_expr (x : ℝ) : ℝ :=
  (x^2 + 3 - (x^4 + 9).sqrt) / x

theorem max_value_expr_max (x : ℝ) (hx : 0 < x) :
  max_value_expr x ≤ (6 * (6:ℝ).sqrt) / (6 + 3 * (2:ℝ).sqrt) :=
sorry

end NUMINAMATH_GPT_max_value_expr_max_l548_54852


namespace NUMINAMATH_GPT_grid_cut_990_l548_54823

theorem grid_cut_990 (grid : Matrix (Fin 1000) (Fin 1000) (Fin 2)) :
  (∃ (rows_to_remove : Finset (Fin 1000)), rows_to_remove.card = 990 ∧ 
   ∀ col : Fin 1000, ∃ row ∈ (Finset.univ \ rows_to_remove), grid row col = 1) ∨
  (∃ (cols_to_remove : Finset (Fin 1000)), cols_to_remove.card = 990 ∧ 
   ∀ row : Fin 1000, ∃ col ∈ (Finset.univ \ cols_to_remove), grid row col = 0) :=
sorry

end NUMINAMATH_GPT_grid_cut_990_l548_54823


namespace NUMINAMATH_GPT_sum_of_numbers_l548_54845

theorem sum_of_numbers (a b c : ℝ) (h1 : 2 * a + b = 46) (h2 : b + 2 * c = 53) (h3 : 2 * c + a = 29) :
  a + b + c = 48.8333 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l548_54845


namespace NUMINAMATH_GPT_people_born_in_country_l548_54899

-- Define the conditions
def people_immigrated : ℕ := 16320
def new_people_total : ℕ := 106491

-- Define the statement to be proven
theorem people_born_in_country (people_born : ℕ) (h : people_born = new_people_total - people_immigrated) : 
    people_born = 90171 :=
  by
    -- This is where we would provide the proof, but we use sorry to skip the proof.
    sorry

end NUMINAMATH_GPT_people_born_in_country_l548_54899


namespace NUMINAMATH_GPT_minimum_candies_l548_54876

theorem minimum_candies (students : ℕ) (N : ℕ) (k : ℕ) : 
  students = 25 → 
  N = 25 * k → 
  (∀ n, 1 ≤ n → n ≤ students → ∃ m, n * k + m ≤ N) → 
  600 ≤ N := 
by
  intros hs hn hd
  sorry

end NUMINAMATH_GPT_minimum_candies_l548_54876


namespace NUMINAMATH_GPT_CD_expression_l548_54897

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C A1 B1 C1 D : V)
variables (a b c : V)

-- Given conditions
axiom AB_eq_a : A - B = a
axiom AC_eq_b : A - C = b
axiom AA1_eq_c : A - A1 = c
axiom midpoint_D : D = (1/2) • (B1 + C1)

-- We need to show
theorem CD_expression : C - D = (1/2) • a - (1/2) • b + c :=
sorry

end NUMINAMATH_GPT_CD_expression_l548_54897


namespace NUMINAMATH_GPT_total_distance_traveled_l548_54853

variable (vm vr t d_up d_down : ℝ)
variable (H_river_speed : vr = 3)
variable (H_row_speed : vm = 6)
variable (H_time : t = 1)

theorem total_distance_traveled (H_upstream : d_up = vm - vr) 
                                (H_downstream : d_down = vm + vr) 
                                (total_time : d_up / (vm - vr) + d_down / (vm + vr) = t) : 
                                2 * (d_up + d_down) = 4.5 := 
                                by
  sorry

end NUMINAMATH_GPT_total_distance_traveled_l548_54853


namespace NUMINAMATH_GPT_tailor_cut_difference_l548_54839

def dress_silk_cut : ℝ := 0.75
def dress_satin_cut : ℝ := 0.60
def dress_chiffon_cut : ℝ := 0.55
def pants_cotton_cut : ℝ := 0.50
def pants_polyester_cut : ℝ := 0.45

theorem tailor_cut_difference :
  (dress_silk_cut + dress_satin_cut + dress_chiffon_cut) - (pants_cotton_cut + pants_polyester_cut) = 0.95 :=
by
  sorry

end NUMINAMATH_GPT_tailor_cut_difference_l548_54839


namespace NUMINAMATH_GPT_no_infinite_subset_of_natural_numbers_l548_54895

theorem no_infinite_subset_of_natural_numbers {
  S : Set ℕ 
} (hS_infinite : S.Infinite) :
  ¬ (∀ a b : ℕ, a ∈ S → b ∈ S → a^2 - a * b + b^2 ∣ (a * b)^2) :=
sorry

end NUMINAMATH_GPT_no_infinite_subset_of_natural_numbers_l548_54895


namespace NUMINAMATH_GPT_first_chapter_length_l548_54885

theorem first_chapter_length (total_pages : ℕ) (second_chapter_pages : ℕ) (third_chapter_pages : ℕ)
  (h : total_pages = 125) (h2 : second_chapter_pages = 35) (h3 : third_chapter_pages  = 24) :
  total_pages - second_chapter_pages - third_chapter_pages = 66 :=
by
  -- Construct the proof using the provided conditions
  sorry

end NUMINAMATH_GPT_first_chapter_length_l548_54885


namespace NUMINAMATH_GPT_equality_of_a_and_b_l548_54806

theorem equality_of_a_and_b
  (a b : ℕ)
  (ha : 0 < a)
  (hb : 0 < b)
  (h : 4 * a * b - 1 ∣ (4 * a ^ 2 - 1) ^ 2) : a = b := 
sorry

end NUMINAMATH_GPT_equality_of_a_and_b_l548_54806


namespace NUMINAMATH_GPT_maximum_mass_difference_l548_54896

theorem maximum_mass_difference (m1 m2 : ℝ) (h1 : 19.7 ≤ m1 ∧ m1 ≤ 20.3) (h2 : 19.7 ≤ m2 ∧ m2 ≤ 20.3) :
  abs (m1 - m2) ≤ 0.6 :=
by
  sorry

end NUMINAMATH_GPT_maximum_mass_difference_l548_54896


namespace NUMINAMATH_GPT_C_is_a_liar_l548_54869

def is_knight_or_liar (P : Prop) : Prop :=
P = true ∨ P = false

variable (A B C : Prop)

-- A, B and C can only be true (knight) or false (liar)
axiom a1 : is_knight_or_liar A
axiom a2 : is_knight_or_liar B
axiom a3 : is_knight_or_liar C

-- A says "B is a liar", meaning if A is a knight, B is a liar, and if A is a liar, B is a knight
axiom a4 : A = true → B = false
axiom a5 : A = false → B = true

-- B says "A and C are of the same type", meaning if B is a knight, A and C are of the same type, otherwise they are not
axiom a6 : B = true → (A = C)
axiom a7 : B = false → (A ≠ C)

-- Prove that C is a liar
theorem C_is_a_liar : C = false :=
by
  sorry

end NUMINAMATH_GPT_C_is_a_liar_l548_54869


namespace NUMINAMATH_GPT_cost_price_is_925_l548_54810

-- Definitions for the conditions
def SP : ℝ := 1110
def profit_percentage : ℝ := 0.20

-- Theorem to prove that the cost price is 925
theorem cost_price_is_925 (CP : ℝ) (h : SP = (CP * (1 + profit_percentage))) : CP = 925 := 
by sorry

end NUMINAMATH_GPT_cost_price_is_925_l548_54810


namespace NUMINAMATH_GPT_cookie_cost_l548_54824

theorem cookie_cost
  (classes3 : ℕ) (students_per_class3 : ℕ)
  (classes4 : ℕ) (students_per_class4 : ℕ)
  (classes5 : ℕ) (students_per_class5 : ℕ)
  (hamburger_cost : ℝ) (carrot_cost : ℝ) (total_lunch_cost : ℝ) (cookie_cost : ℝ)
  (h1 : classes3 = 5) (h2 : students_per_class3 = 30)
  (h3 : classes4 = 4) (h4 : students_per_class4 = 28)
  (h5 : classes5 = 4) (h6 : students_per_class5 = 27)
  (h7 : hamburger_cost = 2.10) (h8 : carrot_cost = 0.50)
  (h9 : total_lunch_cost = 1036):
  ((classes3 * students_per_class3) + (classes4 * students_per_class4) + (classes5 * students_per_class5)) * (cookie_cost + hamburger_cost + carrot_cost) = total_lunch_cost → 
  cookie_cost = 0.20 := 
by 
  sorry

end NUMINAMATH_GPT_cookie_cost_l548_54824


namespace NUMINAMATH_GPT_find_function_range_of_a_l548_54819

variables (a b : ℝ) (f : ℝ → ℝ) 

-- Given: f(x) = ax + b where a ≠ 0 
--        f(2x + 1) = 4x + 1
-- Prove: f(x) = 2x - 1
theorem find_function (h1 : ∀ x, f (2 * x + 1) = 4 * x + 1) : 
  ∃ a b, a = 2 ∧ b = -1 ∧ ∀ x, f x = a * x + b :=
by sorry

-- Given: A = {x | a - 1 < x < 2a +1 }
--        B = {x | 1 < f(x) < 3 }
--        B ⊆ A
-- Prove: 1/2 ≤ a ≤ 2
theorem range_of_a (Hf : ∀ x, f x = 2 * x - 1) (Hsubset: ∀ x, 1 < f x ∧ f x < 3 → a - 1 < x ∧ x < 2 * a + 1) :
  1 / 2 ≤ a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_GPT_find_function_range_of_a_l548_54819


namespace NUMINAMATH_GPT_olympics_year_zodiac_l548_54829

-- Define the list of zodiac signs
def zodiac_cycle : List String :=
  ["rat", "ox", "tiger", "rabbit", "dragon", "snake", "horse", "goat", "monkey", "rooster", "dog", "pig"]

-- Function to compute the zodiac sign for a given year
def zodiac_sign (start_year : ℕ) (year : ℕ) : String :=
  let index := (year - start_year) % 12
  zodiac_cycle.getD index "unknown"

-- Proof statement: the zodiac sign of the year 2008 is "rabbit"
theorem olympics_year_zodiac :
  zodiac_sign 1 2008 = "rabbit" :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_olympics_year_zodiac_l548_54829


namespace NUMINAMATH_GPT_problem_solution_l548_54888

theorem problem_solution : (6 * 7 * 8 * 9 * 10) / (6 + 7 + 8 + 9 + 10) = 756 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l548_54888


namespace NUMINAMATH_GPT_calculate_sin_product_l548_54800

theorem calculate_sin_product (α β : ℝ) (h1 : Real.sin (α + β) = 0.2) (h2 : Real.cos (α - β) = 0.3) :
  Real.sin (α + π/4) * Real.sin (β + π/4) = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_calculate_sin_product_l548_54800


namespace NUMINAMATH_GPT_tommy_initial_balloons_l548_54855

theorem tommy_initial_balloons (initial_balloons balloons_added total_balloons : ℝ)
  (h1 : balloons_added = 34.5)
  (h2 : total_balloons = 60.75)
  (h3 : total_balloons = initial_balloons + balloons_added) :
  initial_balloons = 26.25 :=
by sorry

end NUMINAMATH_GPT_tommy_initial_balloons_l548_54855


namespace NUMINAMATH_GPT_smallest_x_abs_eq_15_l548_54856

theorem smallest_x_abs_eq_15 :
  ∃ x : ℝ, (|x - 8| = 15) ∧ ∀ y : ℝ, (|y - 8| = 15) → y ≥ x :=
sorry

end NUMINAMATH_GPT_smallest_x_abs_eq_15_l548_54856


namespace NUMINAMATH_GPT_square_of_105_l548_54840

theorem square_of_105 : 105^2 = 11025 := by
  sorry

end NUMINAMATH_GPT_square_of_105_l548_54840


namespace NUMINAMATH_GPT_richard_older_than_david_l548_54843

theorem richard_older_than_david
  (R D S : ℕ)   -- ages of Richard, David, Scott
  (x : ℕ)       -- the number of years Richard is older than David
  (h1 : R = D + x)
  (h2 : D = S + 8)
  (h3 : R + 8 = 2 * (S + 8))
  (h4 : D = 14) : 
  x = 6 := sorry

end NUMINAMATH_GPT_richard_older_than_david_l548_54843


namespace NUMINAMATH_GPT_joey_needs_figures_to_cover_cost_l548_54880

-- Definitions based on conditions
def cost_sneakers : ℕ := 92
def earnings_per_lawn : ℕ := 8
def lawns : ℕ := 3
def earnings_per_hour : ℕ := 5
def work_hours : ℕ := 10
def price_per_figure : ℕ := 9

-- Total earnings from mowing lawns
def earnings_lawns := lawns * earnings_per_lawn
-- Total earnings from job
def earnings_job := work_hours * earnings_per_hour
-- Total earnings from both
def total_earnings := earnings_lawns + earnings_job
-- Remaining amount to cover the cost
def remaining_amount := cost_sneakers - total_earnings

-- Correct answer based on the problem statement
def collectible_figures_needed := remaining_amount / price_per_figure

-- Lean 4 statement to prove the requirement
theorem joey_needs_figures_to_cover_cost :
  collectible_figures_needed = 2 := by
  sorry

end NUMINAMATH_GPT_joey_needs_figures_to_cover_cost_l548_54880


namespace NUMINAMATH_GPT_find_alpha_angle_l548_54846

theorem find_alpha_angle :
  ∃ α : ℝ, (7 * α + 8 * α + 45) = 180 ∧ α = 9 :=
by 
  sorry

end NUMINAMATH_GPT_find_alpha_angle_l548_54846


namespace NUMINAMATH_GPT_peter_spent_on_repairs_l548_54849

variable (C : ℝ)

def repairs_cost (C : ℝ) := 0.10 * C

def profit (C : ℝ) := 1.20 * C - C

theorem peter_spent_on_repairs :
  ∀ C, profit C = 1100 → repairs_cost C = 550 :=
by
  intro C
  sorry

end NUMINAMATH_GPT_peter_spent_on_repairs_l548_54849


namespace NUMINAMATH_GPT_meet_time_same_departure_meet_time_staggered_departure_catch_up_time_same_departure_l548_54847

-- Distance between locations A and B
def distance : ℝ := 448

-- Speed of the slow train
def slow_speed : ℝ := 60

-- Speed of the fast train
def fast_speed : ℝ := 80

-- Problem 1: Prove the two trains meet 3.2 hours after the fast train departs (both trains heading towards each other, departing at the same time)
theorem meet_time_same_departure : 
  (slow_speed + fast_speed) * 3.2 = distance :=
by
  sorry

-- Problem 2: Prove the two trains meet 3 hours after the fast train departs (slow train departs 28 minutes before the fast train)
theorem meet_time_staggered_departure : 
  (slow_speed * (28/60) + (slow_speed + fast_speed) * 3) = distance :=
by
  sorry

-- Problem 3: Prove the fast train catches up to the slow train 22.4 hours after departure (both trains heading in the same direction, departing at the same time)
theorem catch_up_time_same_departure : 
  (fast_speed - slow_speed) * 22.4 = distance :=
by
  sorry

end NUMINAMATH_GPT_meet_time_same_departure_meet_time_staggered_departure_catch_up_time_same_departure_l548_54847


namespace NUMINAMATH_GPT_list_length_eq_12_l548_54851

-- Define a list of numbers in the sequence
def seq : List ℝ := [1.5, 5.5, 9.5, 13.5, 17.5, 21.5, 25.5, 29.5, 33.5, 37.5, 41.5, 45.5]

-- Define the theorem that states the number of elements in the sequence
theorem list_length_eq_12 : seq.length = 12 := 
by 
  -- Proof here
  sorry

end NUMINAMATH_GPT_list_length_eq_12_l548_54851


namespace NUMINAMATH_GPT_range_a_l548_54814

theorem range_a (a : ℝ) :
  (∀ x : ℝ, (0 < x ∧ x ≤ 2) → x^2 - 2 * a * x + 1 ≥ 0) → a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_a_l548_54814


namespace NUMINAMATH_GPT_fraction_product_equals_l548_54893

def frac1 := 7 / 4
def frac2 := 8 / 14
def frac3 := 9 / 6
def frac4 := 10 / 25
def frac5 := 28 / 21
def frac6 := 15 / 45
def frac7 := 32 / 16
def frac8 := 50 / 100

theorem fraction_product_equals : 
  (frac1 * frac2 * frac3 * frac4 * frac5 * frac6 * frac7 * frac8) = (4 / 5) := 
by
  sorry

end NUMINAMATH_GPT_fraction_product_equals_l548_54893


namespace NUMINAMATH_GPT_inequality_system_solution_l548_54866

theorem inequality_system_solution (x : ℝ) :
  (x + 7) / 3 ≤ x + 3 ∧ 2 * (x + 1) < x + 3 ↔ -1 ≤ x ∧ x < 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_system_solution_l548_54866


namespace NUMINAMATH_GPT_silvia_last_play_without_breach_l548_54833

theorem silvia_last_play_without_breach (N : ℕ) : 
  36 * N < 2000 ∧ 72 * N ≥ 2000 ↔ N = 28 :=
by
  sorry

end NUMINAMATH_GPT_silvia_last_play_without_breach_l548_54833


namespace NUMINAMATH_GPT_tan_alpha_value_tan_beta_value_sum_angles_l548_54872

open Real

noncomputable def tan_alpha (α : ℝ) : ℝ := sin α / cos α
noncomputable def tan_beta (β : ℝ) : ℝ := sin β / cos β

def conditions (α β : ℝ) :=
  α > 0 ∧ α < π / 2 ∧ β > 0 ∧ β < π / 2 ∧ 
  sin α = 1 / sqrt 10 ∧ tan β = 1 / 7

theorem tan_alpha_value (α β : ℝ) (h : conditions α β) : tan_alpha α = 1 / 3 := sorry

theorem tan_beta_value (α β : ℝ) (h : conditions α β) : tan_beta β = 1 / 7 := sorry

theorem sum_angles (α β : ℝ) (h : conditions α β) : 2 * α + β = π / 4 := sorry

end NUMINAMATH_GPT_tan_alpha_value_tan_beta_value_sum_angles_l548_54872


namespace NUMINAMATH_GPT_find_a6_l548_54825

-- Defining the conditions of the problem
def a1 := 2
def S3 := 12

-- Defining the necessary arithmetic sequence properties
def Sn (a1 d : ℕ) (n : ℕ) : ℕ := n * (2 * a1 + (n - 1) * d) / 2
def an (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

-- Proof statement in Lean
theorem find_a6 (d : ℕ) (a1_val S3_val : ℕ) (h1 : a1_val = 2) (h2 : S3_val = 12) 
    (h3 : 3 * (2 * a1_val + (3 - 1) * d) / 2 = S3_val) : an a1_val d 6 = 12 :=
by 
  -- omitted proof
  sorry

end NUMINAMATH_GPT_find_a6_l548_54825


namespace NUMINAMATH_GPT_exact_time_between_9_10_l548_54868

theorem exact_time_between_9_10
  (t : ℝ)
  (h1 : 0 ≤ t ∧ t < 60)
  (h2 : |6 * (t + 5) - (270 + 0.5 * (t - 2))| = 180) :
  t = 10 + 3 / 4 :=
sorry

end NUMINAMATH_GPT_exact_time_between_9_10_l548_54868


namespace NUMINAMATH_GPT_binomial_expansion_calculation_l548_54842

theorem binomial_expansion_calculation :
  102^5 - 5 * 102^4 + 10 * 102^3 - 10 * 102^2 + 5 * 102 - 1 = 101^5 :=
by
  sorry

end NUMINAMATH_GPT_binomial_expansion_calculation_l548_54842


namespace NUMINAMATH_GPT_abs_sum_lt_ineq_l548_54873

theorem abs_sum_lt_ineq (x : ℝ) (a : ℝ) (h₀ : 0 < a) : 
  (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ (1 < a) :=
by
  sorry

end NUMINAMATH_GPT_abs_sum_lt_ineq_l548_54873


namespace NUMINAMATH_GPT_find_digits_l548_54841

theorem find_digits (x y z : ℕ) (hx : x ≤ 9) (hy : y ≤ 9) (hz : z ≤ 9)
    (h_eq : (10*x+5) * (300 + 10*y + z) = 7850) : x = 2 ∧ y = 1 ∧ z = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_digits_l548_54841


namespace NUMINAMATH_GPT_length_of_football_field_l548_54863

theorem length_of_football_field :
  ∃ x : ℝ, (4 * x + 500 = 1172) ∧ x = 168 :=
by
  use 168
  simp
  sorry

end NUMINAMATH_GPT_length_of_football_field_l548_54863


namespace NUMINAMATH_GPT_find_equation_of_ellipse_C_l548_54801

def equation_of_ellipse_C (a b : ℝ) : Prop :=
  ∀ x y, (x^2 / a^2 + y^2 / b^2 = 1)

theorem find_equation_of_ellipse_C :
  ∀ (a b : ℝ), (a = 2) → (b = 1) →
  (equation_of_ellipse_C a b) →
  equation_of_ellipse_C 2 1 :=
by
  intros a b ha hb h
  sorry

end NUMINAMATH_GPT_find_equation_of_ellipse_C_l548_54801


namespace NUMINAMATH_GPT_words_lost_due_to_prohibition_l548_54894

-- Define the conditions given in the problem.
def number_of_letters := 64
def forbidden_letter := 7
def total_one_letter_words := number_of_letters
def total_two_letter_words := number_of_letters * number_of_letters

-- Define the forbidden letter loss calculation.
def one_letter_words_lost := 1
def two_letter_words_lost := number_of_letters + number_of_letters - 1

-- Define the total words lost calculation.
def total_words_lost := one_letter_words_lost + two_letter_words_lost

-- State the theorem to prove the number of words lost is 128.
theorem words_lost_due_to_prohibition : total_words_lost = 128 :=
by sorry

end NUMINAMATH_GPT_words_lost_due_to_prohibition_l548_54894


namespace NUMINAMATH_GPT_equation1_solution_equation2_solution_l548_54807

theorem equation1_solution (x : ℝ) : (x - 4)^2 - 9 = 0 ↔ (x = 7 ∨ x = 1) := 
sorry

theorem equation2_solution (x : ℝ) : (x + 1)^3 = -27 ↔ (x = -4) := 
sorry

end NUMINAMATH_GPT_equation1_solution_equation2_solution_l548_54807


namespace NUMINAMATH_GPT_area_transformed_region_l548_54883

-- Define the transformation matrix
def matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 1], ![4, 3]]

-- Define the area of region T
def area_T := 6

-- The statement we want to prove: the area of T' is 30.
theorem area_transformed_region :
  let det := matrix.det
  area_T * det = 30 :=
by
  sorry

end NUMINAMATH_GPT_area_transformed_region_l548_54883


namespace NUMINAMATH_GPT_smallest_positive_angle_same_terminal_side_l548_54881

theorem smallest_positive_angle_same_terminal_side 
  (k : ℤ) : ∃ α : ℝ, 0 < α ∧ α < 360 ∧ -2002 = α + k * 360 ∧ α = 158 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_angle_same_terminal_side_l548_54881


namespace NUMINAMATH_GPT_find_third_divisor_l548_54803

theorem find_third_divisor 
  (h1 : ∃ (n : ℕ), n = 1014 - 3 ∧ n % 12 = 0 ∧ n % 16 = 0 ∧ n % 21 = 0 ∧ n % 28 = 0) 
  (h2 : 1011 - 3 = 1008) : 
  (∃ d, d = 3 ∧ 1008 % d = 0 ∧ 1008 % 12 = 0 ∧ 1008 % 16 = 0 ∧ 1008 % 21 = 0 ∧ 1008 % 28 = 0) :=
sorry

end NUMINAMATH_GPT_find_third_divisor_l548_54803


namespace NUMINAMATH_GPT_possible_distance_between_houses_l548_54850

variable (d : ℝ)

theorem possible_distance_between_houses (h_d1 : 1 ≤ d) (h_d2 : d ≤ 5) : 1 ≤ d ∧ d ≤ 5 :=
by
  exact ⟨h_d1, h_d2⟩

end NUMINAMATH_GPT_possible_distance_between_houses_l548_54850


namespace NUMINAMATH_GPT_expressionEquals243_l548_54865

noncomputable def calculateExpression : ℕ :=
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 *
  (1 / 19683) * 59049

theorem expressionEquals243 : calculateExpression = 243 := by
  sorry

end NUMINAMATH_GPT_expressionEquals243_l548_54865


namespace NUMINAMATH_GPT_ratio_of_selling_prices_l548_54877

variable (CP : ℝ)
def SP1 : ℝ := CP * 1.6
def SP2 : ℝ := CP * 0.8

theorem ratio_of_selling_prices : SP2 / SP1 = 1 / 2 := 
by sorry

end NUMINAMATH_GPT_ratio_of_selling_prices_l548_54877


namespace NUMINAMATH_GPT_least_xy_l548_54891

theorem least_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / (3 * y) = 1 / 9) : x * y = 108 :=
by
  sorry

end NUMINAMATH_GPT_least_xy_l548_54891


namespace NUMINAMATH_GPT_fractional_part_inequality_l548_54826

noncomputable def frac (z : ℝ) : ℝ := z - ⌊z⌋

theorem fractional_part_inequality (x y : ℝ) : frac (x + y) ≤ frac x + frac y := 
sorry

end NUMINAMATH_GPT_fractional_part_inequality_l548_54826


namespace NUMINAMATH_GPT_smallest_X_divisible_by_15_l548_54821

theorem smallest_X_divisible_by_15 (T : ℕ) (h_pos : T > 0) (h_digits : ∀ (d : ℕ), d ∈ (Nat.digits 10 T) → d = 0 ∨ d = 1)
  (h_div15 : T % 15 = 0) : ∃ X : ℕ, X = T / 15 ∧ X = 74 :=
sorry

end NUMINAMATH_GPT_smallest_X_divisible_by_15_l548_54821


namespace NUMINAMATH_GPT_sum_coefficients_l548_54864

theorem sum_coefficients (a : ℤ) (f : ℤ → ℤ) :
  f x = (1 - 2 * x)^7 ∧ a_0 = f 0 ∧ a_1_plus_a_7 = f 1 - f 0 
→ a_1_plus_a_7 = -2 :=
by sorry

end NUMINAMATH_GPT_sum_coefficients_l548_54864


namespace NUMINAMATH_GPT_least_number_to_add_l548_54816

theorem least_number_to_add (LCM : ℕ) (a : ℕ) (x : ℕ) :
  LCM = 23 * 29 * 31 →
  a = 1076 →
  x = LCM - a →
  (a + x) % LCM = 0 :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_add_l548_54816


namespace NUMINAMATH_GPT_rent_budget_l548_54871

variables (food_per_week : ℝ) (weekly_food_budget : ℝ) (video_streaming : ℝ)
          (cell_phone : ℝ) (savings : ℝ) (rent : ℝ)
          (total_spending : ℝ)

-- Conditions
def food_budget := food_per_week * 4 = weekly_food_budget
def video_streaming_budget := video_streaming = 30
def cell_phone_budget := cell_phone = 50
def savings_budget := savings = 0.1 * total_spending
def savings_amount := savings = 198

-- Prove
theorem rent_budget (h1 : food_budget food_per_week weekly_food_budget)
                    (h2 : video_streaming_budget video_streaming)
                    (h3 : cell_phone_budget cell_phone)
                    (h4 : savings_budget savings total_spending)
                    (h5 : savings_amount savings) :
  rent = 1500 :=
sorry

end NUMINAMATH_GPT_rent_budget_l548_54871


namespace NUMINAMATH_GPT_jill_and_emily_total_peaches_l548_54828

-- Define each person and their conditions
variables (Steven Jake Jill Maria Emily : ℕ)

-- Given conditions
def steven_has_peaches : Steven = 14 := sorry
def jake_has_fewer_than_steven : Jake = Steven - 6 := sorry
def jake_has_more_than_jill : Jake = Jill + 3 := sorry
def maria_has_twice_jake : Maria = 2 * Jake := sorry
def emily_has_fewer_than_maria : Emily = Maria - 9 := sorry

-- The theorem statement combining the conditions and the required result
theorem jill_and_emily_total_peaches (Steven Jake Jill Maria Emily : ℕ)
  (h1 : Steven = 14) 
  (h2 : Jake = Steven - 6) 
  (h3 : Jake = Jill + 3) 
  (h4 : Maria = 2 * Jake) 
  (h5 : Emily = Maria - 9) : 
  Jill + Emily = 12 := 
sorry

end NUMINAMATH_GPT_jill_and_emily_total_peaches_l548_54828


namespace NUMINAMATH_GPT_find_largest_n_l548_54867

theorem find_largest_n 
  (a : ℕ → ℕ) (b : ℕ → ℕ)
  (x y : ℕ)
  (h_a1 : a 1 = 1)
  (h_b1 : b 1 = 1)
  (h_arith_a : ∀ n : ℕ, a n = 1 + (n - 1) * x)
  (h_arith_b : ∀ n : ℕ, b n = 1 + (n - 1) * y)
  (h_order : x ≤ y)
  (h_product : ∃ n : ℕ, a n * b n = 4021) :
  ∃ n : ℕ, a n * b n = 4021 ∧ n ≤ 11 := 
by
  sorry

end NUMINAMATH_GPT_find_largest_n_l548_54867


namespace NUMINAMATH_GPT_smallest_possible_value_l548_54818

open Nat

theorem smallest_possible_value (c d : ℕ) (hc : c > d) (hc_pos : 0 < c) (hd_pos : 0 < d) (odd_cd : ¬Even (c + d)) :
  (∃ (y : ℚ), y > 0 ∧ y = (c + d : ℚ) / (c - d) + (c - d : ℚ) / (c + d) ∧ y = 10 / 3) :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_value_l548_54818


namespace NUMINAMATH_GPT_tangent_line_to_parabola_parallel_l548_54870

theorem tangent_line_to_parabola_parallel (m : ℝ) :
  ∀ (x y : ℝ), (y = x^2) → (2*x - y + m = 0 → m = -1) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_to_parabola_parallel_l548_54870


namespace NUMINAMATH_GPT_brownies_in_pan_l548_54887

theorem brownies_in_pan : 
    ∀ (pan_length pan_width brownie_length brownie_width : ℕ), 
    pan_length = 24 -> 
    pan_width = 20 -> 
    brownie_length = 3 -> 
    brownie_width = 2 -> 
    (pan_length * pan_width) / (brownie_length * brownie_width) = 80 := 
by
  intros pan_length pan_width brownie_length brownie_width h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_brownies_in_pan_l548_54887


namespace NUMINAMATH_GPT_triangle_third_side_l548_54836

theorem triangle_third_side (a b c : ℝ) (h1 : a = 5) (h2 : b = 7) (h3 : 2 < c ∧ c < 12) : c = 6 :=
sorry

end NUMINAMATH_GPT_triangle_third_side_l548_54836


namespace NUMINAMATH_GPT_solve_xyz_l548_54830

def is_solution (x y z : ℕ) : Prop :=
  x * y + y * z + z * x = 2 * (x + y + z)

theorem solve_xyz (x y z : ℕ) :
  is_solution x y z ↔ (x = 1 ∧ y = 2 ∧ z = 4) ∨
                     (x = 1 ∧ y = 4 ∧ z = 2) ∨
                     (x = 2 ∧ y = 1 ∧ z = 4) ∨
                     (x = 2 ∧ y = 4 ∧ z = 1) ∨
                     (x = 2 ∧ y = 2 ∧ z = 2) ∨
                     (x = 4 ∧ y = 1 ∧ z = 2) ∨
                     (x = 4 ∧ y = 2 ∧ z = 1) := sorry

end NUMINAMATH_GPT_solve_xyz_l548_54830


namespace NUMINAMATH_GPT_goals_scored_by_each_l548_54817

theorem goals_scored_by_each (total_goals : ℕ) (percentage : ℕ) (two_players_goals : ℕ) (each_player_goals : ℕ)
  (H1 : total_goals = 300)
  (H2 : percentage = 20)
  (H3 : two_players_goals = (percentage * total_goals) / 100)
  (H4 : two_players_goals / 2 = each_player_goals) :
  each_player_goals = 30 := by
  sorry

end NUMINAMATH_GPT_goals_scored_by_each_l548_54817


namespace NUMINAMATH_GPT_fraction_to_decimal_l548_54832

theorem fraction_to_decimal : (7 / 32 : ℚ) = 0.21875 := 
by {
  sorry
}

end NUMINAMATH_GPT_fraction_to_decimal_l548_54832


namespace NUMINAMATH_GPT_maximum_fraction_l548_54857

theorem maximum_fraction (A B : ℕ) (h1 : A ≠ B) (h2 : 0 < A ∧ A < 1000) (h3 : 0 < B ∧ B < 1000) :
  ∃ (A B : ℕ), (A = 500) ∧ (B = 499) ∧ (A ≠ B) ∧ (0 < A ∧ A < 1000) ∧ (0 < B ∧ B < 1000) ∧ (A - B = 1) ∧ (A + B = 999) ∧ (499 / 500 = 0.998) := sorry

end NUMINAMATH_GPT_maximum_fraction_l548_54857


namespace NUMINAMATH_GPT_Jean_spots_l548_54811

/--
Jean the jaguar has a total of 60 spots.
Half of her spots are located on her upper torso.
One-third of the spots are located on her back and hindquarters.
Jean has 30 spots on her upper torso.
Prove that Jean has 10 spots located on her sides.
-/
theorem Jean_spots (TotalSpots UpperTorsoSpots BackHindquartersSpots SidesSpots : ℕ)
  (h_half : UpperTorsoSpots = TotalSpots / 2)
  (h_back : BackHindquartersSpots = TotalSpots / 3)
  (h_total_upper : UpperTorsoSpots = 30)
  (h_total : TotalSpots = 60) :
  SidesSpots = 10 :=
by
  sorry

end NUMINAMATH_GPT_Jean_spots_l548_54811


namespace NUMINAMATH_GPT_domain_of_sqrt_log_l548_54862

noncomputable def domain_of_function : Set ℝ := 
  {x : ℝ | (-Real.sqrt 2) ≤ x ∧ x < -1 ∨ 1 < x ∧ x ≤ Real.sqrt 2}

theorem domain_of_sqrt_log : ∀ x : ℝ, 
  (∃ y : ℝ, y = Real.sqrt (Real.log (x^2 - 1) / Real.log (1/2)) ∧ 
  y ≥ 0) ↔ x ∈ domain_of_function := 
by
  sorry

end NUMINAMATH_GPT_domain_of_sqrt_log_l548_54862


namespace NUMINAMATH_GPT_largest_integer_less_than_100_leaving_remainder_4_l548_54802

theorem largest_integer_less_than_100_leaving_remainder_4 (n : ℕ) (h1 : n < 100) (h2 : n % 7 = 4) : n = 95 := 
sorry

end NUMINAMATH_GPT_largest_integer_less_than_100_leaving_remainder_4_l548_54802


namespace NUMINAMATH_GPT_students_in_sixth_level_l548_54838

theorem students_in_sixth_level (S : ℕ)
  (h1 : ∃ S₄ : ℕ, S₄ = 4 * S)
  (h2 : ∃ S₇ : ℕ, S₇ = 2 * (4 * S))
  (h3 : S + 4 * S + 2 * (4 * S) = 520) :
  S = 40 :=
by
  sorry

end NUMINAMATH_GPT_students_in_sixth_level_l548_54838


namespace NUMINAMATH_GPT_line_equation_exists_l548_54854

theorem line_equation_exists 
  (a b : ℝ) 
  (ha_pos: a > 0)
  (hb_pos: b > 0)
  (h_area: 1 / 2 * a * b = 2) 
  (h_diff: a - b = 3 ∨ b - a = 3) : 
  (∀ x y : ℝ, (x + 4 * y = 4 ∧ (x / a + y / b = 1)) ∨ (4 * x + y = 4 ∧ (x / a + y / b = 1))) :=
sorry

end NUMINAMATH_GPT_line_equation_exists_l548_54854


namespace NUMINAMATH_GPT_large_box_times_smaller_box_l548_54860

noncomputable def large_box_volume (width length height : ℕ) : ℕ := width * length * height

noncomputable def small_box_volume (width length height : ℕ) : ℕ := width * length * height

theorem large_box_times_smaller_box :
  let large_volume := large_box_volume 30 20 5
  let small_volume := small_box_volume 6 4 1
  large_volume / small_volume = 125 :=
by
  let large_volume := large_box_volume 30 20 5
  let small_volume := small_box_volume 6 4 1
  show large_volume / small_volume = 125
  sorry

end NUMINAMATH_GPT_large_box_times_smaller_box_l548_54860


namespace NUMINAMATH_GPT_sum_of_prime_factors_143_l548_54815

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_of_prime_factors_143 : 
  ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ 143 = p1 * p2 ∧ p1 + p2 = 24 := by
  sorry

end NUMINAMATH_GPT_sum_of_prime_factors_143_l548_54815


namespace NUMINAMATH_GPT_N_eq_P_l548_54874

def N : Set ℝ := {x | ∃ n : ℤ, x = (n : ℝ) / 2 - 1 / 3}
def P : Set ℝ := {x | ∃ p : ℤ, x = (p : ℝ) / 2 + 1 / 6}

theorem N_eq_P : N = P :=
  sorry

end NUMINAMATH_GPT_N_eq_P_l548_54874


namespace NUMINAMATH_GPT_greatest_remainder_le_11_l548_54890

noncomputable def greatest_remainder (x : ℕ) : ℕ := x % 12

theorem greatest_remainder_le_11 (x : ℕ) (h : x % 12 ≠ 0) : greatest_remainder x = 11 :=
by
  sorry

end NUMINAMATH_GPT_greatest_remainder_le_11_l548_54890


namespace NUMINAMATH_GPT_drawings_on_last_page_l548_54886

theorem drawings_on_last_page :
  let n_notebooks := 10 
  let p_pages := 50
  let d_original := 5
  let d_new := 8
  let total_drawings := n_notebooks * p_pages * d_original
  let total_pages_new := total_drawings / d_new
  let filled_complete_pages := 6 * p_pages
  let drawings_on_last_page := total_drawings - filled_complete_pages * d_new - 40 * d_new
  drawings_on_last_page == 4 :=
  sorry

end NUMINAMATH_GPT_drawings_on_last_page_l548_54886


namespace NUMINAMATH_GPT_speed_of_current_11_00448_l548_54813

/-- 
  The speed at which a man can row a boat in still water is 25 kmph.
  He takes 7.999360051195905 seconds to cover 80 meters downstream.
  Prove that the speed of the current is 11.00448 km/h.
-/
theorem speed_of_current_11_00448 :
  let speed_in_still_water_kmph := 25
  let distance_m := 80
  let time_s := 7.999360051195905
  (distance_m / time_s) * 3600 / 1000 - speed_in_still_water_kmph = 11.00448 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_current_11_00448_l548_54813


namespace NUMINAMATH_GPT_pencils_remaining_l548_54884

variable (initial_pencils : ℝ) (pencils_given : ℝ)

theorem pencils_remaining (h1 : initial_pencils = 56.0) 
                          (h2 : pencils_given = 9.5) 
                          : initial_pencils - pencils_given = 46.5 :=
by 
  sorry

end NUMINAMATH_GPT_pencils_remaining_l548_54884


namespace NUMINAMATH_GPT_productivity_increase_l548_54812

theorem productivity_increase (a b : ℝ) : (7 / 8) * (1 + 20 / 100) = 1.05 :=
by
  sorry

end NUMINAMATH_GPT_productivity_increase_l548_54812


namespace NUMINAMATH_GPT_find_x_squared_plus_y_squared_l548_54848

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x * y = 12) (h2 : x^2 * y + x * y^2 + x + y = 120) :
  x^2 + y^2 = 10344 / 169 :=
by
  sorry

end NUMINAMATH_GPT_find_x_squared_plus_y_squared_l548_54848


namespace NUMINAMATH_GPT_solids_with_triangular_front_view_l548_54892

-- Definitions based on given conditions
inductive Solid
  | TriangularPyramid
  | SquarePyramid
  | TriangularPrism
  | SquarePrism
  | Cone
  | Cylinder

def can_have_triangular_front_view : Solid → Prop
  | Solid.TriangularPyramid => true
  | Solid.SquarePyramid => true
  | Solid.TriangularPrism => true
  | Solid.SquarePrism => false
  | Solid.Cone => true
  | Solid.Cylinder => false

-- Theorem statement
theorem solids_with_triangular_front_view :
  {s : Solid | can_have_triangular_front_view s} = 
  {Solid.TriangularPyramid, Solid.SquarePyramid, Solid.TriangularPrism, Solid.Cone} :=
by
  sorry

end NUMINAMATH_GPT_solids_with_triangular_front_view_l548_54892


namespace NUMINAMATH_GPT_exponent_division_l548_54875

theorem exponent_division :
  (1000 ^ 7) / (10 ^ 17) = 10 ^ 4 := 
  sorry

end NUMINAMATH_GPT_exponent_division_l548_54875


namespace NUMINAMATH_GPT_cost_of_schools_renovation_plans_and_min_funding_l548_54879

-- Define costs of Type A and Type B schools
def cost_A : ℝ := 60
def cost_B : ℝ := 85

-- Initial conditions given in the problem
axiom initial_condition_1 : cost_A + 2 * cost_B = 230
axiom initial_condition_2 : 2 * cost_A + cost_B = 205

-- Variables for number of Type A and Type B schools to renovate
variables (x : ℕ) (y : ℕ)
-- Total schools to renovate
axiom total_schools : x + y = 6

-- National and local finance constraints
axiom national_finance_max : 60 * x + 85 * y ≤ 380
axiom local_finance_min : 10 * x + 15 * y ≥ 70

-- Proving the cost of one Type A and one Type B school
theorem cost_of_schools : cost_A = 60 ∧ cost_B = 85 := 
by {
  sorry
}

-- Proving the number of renovation plans and the least funding plan
theorem renovation_plans_and_min_funding :
  ∃ x y, (x + y = 6) ∧ 
         (10 * x + 15 * y ≥ 70) ∧ 
         (60 * x + 85 * y ≤ 380) ∧ 
         (x = 2 ∧ y = 4 ∨ x = 3 ∧ y = 3 ∨ x = 4 ∧ y = 2) ∧ 
         (∀ (a b : ℕ), (a + b = 6) ∧ 
                       (10 * a + 15 * b ≥ 70) ∧ 
                       (60 * a + 85 * b ≤ 380) → 
                       60 * a + 85 * b ≥ 410) :=
by {
  sorry
}

end NUMINAMATH_GPT_cost_of_schools_renovation_plans_and_min_funding_l548_54879


namespace NUMINAMATH_GPT_least_5_digit_divisible_by_12_15_18_l548_54837

theorem least_5_digit_divisible_by_12_15_18 : 
  ∃ n, n >= 10000 ∧ n < 100000 ∧ (180 ∣ n) ∧ n = 10080 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_least_5_digit_divisible_by_12_15_18_l548_54837


namespace NUMINAMATH_GPT_treadmill_time_saved_l548_54861

theorem treadmill_time_saved:
  let monday_speed := 6
  let tuesday_speed := 4
  let wednesday_speed := 5
  let thursday_speed := 6
  let friday_speed := 3
  let distance := 3 
  let daily_times : List ℚ := 
    [distance/monday_speed, distance/tuesday_speed, distance/wednesday_speed, distance/thursday_speed, distance/friday_speed]
  let total_time := (daily_times.map (λ t => t)).sum
  let total_distance := 5 * distance 
  let uniform_speed := 5 
  let uniform_time := total_distance / uniform_speed 
  let time_difference := total_time - uniform_time 
  let time_in_minutes := time_difference * 60 
  time_in_minutes = 21 := 
by 
  sorry

end NUMINAMATH_GPT_treadmill_time_saved_l548_54861


namespace NUMINAMATH_GPT_find_speeds_and_circumference_l548_54809

variable (Va Vb : ℝ)
variable (l : ℝ)

axiom smaller_arc_condition : 10 * (Va + Vb) = 150
axiom larger_arc_condition : 14 * (Va + Vb) = l - 150
axiom travel_condition : l / Va = 90 / Vb 

theorem find_speeds_and_circumference :
  Va = 12 ∧ Vb = 3 ∧ l = 360 := by
  sorry

end NUMINAMATH_GPT_find_speeds_and_circumference_l548_54809


namespace NUMINAMATH_GPT_seated_students_count_l548_54878

theorem seated_students_count :
  ∀ (S T standing_students total_attendees : ℕ),
    T = 30 →
    standing_students = 25 →
    total_attendees = 355 →
    total_attendees = S + T + standing_students →
    S = 300 :=
by
  intros S T standing_students total_attendees hT hStanding hTotalAttendees hEquation
  sorry

end NUMINAMATH_GPT_seated_students_count_l548_54878


namespace NUMINAMATH_GPT_lower_limit_of_range_l548_54820

theorem lower_limit_of_range (x y : ℝ) (hx1 : 3 < x) (hx2 : x < 8) (hx3 : y < x) (hx4 : x < 10) (hx5 : x = 7) : 3 < y ∧ y ≤ 7 :=
by
  sorry

end NUMINAMATH_GPT_lower_limit_of_range_l548_54820


namespace NUMINAMATH_GPT_complement_union_l548_54834

open Set

variable (U M N : Set ℕ)

def complement_U (A : Set ℕ) : Set ℕ := { x | x ∈ U ∧ x ∉ A }

theorem complement_union (hU : U = {0, 1, 2, 3, 4, 5, 6})
                          (hM : M = {1, 3, 5})
                          (hN : N = {2, 4, 6}) :
  (complement_U U M) ∪ (complement_U U N) = {0, 1, 2, 3, 4, 5, 6} :=
by 
  sorry

end NUMINAMATH_GPT_complement_union_l548_54834


namespace NUMINAMATH_GPT_number_of_numbers_l548_54882

theorem number_of_numbers (n S : ℕ) 
  (h1 : (S + 26) / n = 15)
  (h2 : (S + 36) / n = 16)
  : n = 10 :=
sorry

end NUMINAMATH_GPT_number_of_numbers_l548_54882


namespace NUMINAMATH_GPT_circle_area_l548_54808

noncomputable def pointA : ℝ × ℝ := (2, 7)
noncomputable def pointB : ℝ × ℝ := (8, 5)

def is_tangent_with_intersection_on_x_axis (A B C : ℝ × ℝ) : Prop :=
  ∃ R : ℝ, ∃ r : ℝ, ∀ M : ℝ × ℝ, dist M C = R → dist A M = r ∧ dist B M = r

theorem circle_area (A B : ℝ × ℝ) (hA : A = (2, 7)) (hB : B = (8, 5))
    (h : ∃ C : ℝ × ℝ, is_tangent_with_intersection_on_x_axis A B C) 
    : ∃ R : ℝ, π * R^2 = 12.5 * π := 
sorry

end NUMINAMATH_GPT_circle_area_l548_54808


namespace NUMINAMATH_GPT_chromium_percentage_in_new_alloy_l548_54831

noncomputable def percentage_chromium_new_alloy (w1 w2 p1 p2 : ℝ) : ℝ :=
  ((p1 * w1 + p2 * w2) / (w1 + w2)) * 100

theorem chromium_percentage_in_new_alloy :
  percentage_chromium_new_alloy 15 35 0.12 0.10 = 10.6 :=
by
  sorry

end NUMINAMATH_GPT_chromium_percentage_in_new_alloy_l548_54831


namespace NUMINAMATH_GPT_gcd_solution_l548_54805

theorem gcd_solution {m n : ℕ} (hm : m > 0) (hn : n > 0) (h : Nat.gcd m n = 10) : Nat.gcd (12 * m) (18 * n) = 60 := 
sorry

end NUMINAMATH_GPT_gcd_solution_l548_54805


namespace NUMINAMATH_GPT_g_six_l548_54889

noncomputable def g : ℝ → ℝ := sorry

axiom g_func_eq (x y : ℝ) : g (x - y) = g x * g y
axiom g_nonzero (x : ℝ) : g x ≠ 0
axiom g_double (x : ℝ) : g (2 * x) = g x ^ 2
axiom g_value : g 6 = 1

theorem g_six : g 6 = 1 := by
  exact g_value

end NUMINAMATH_GPT_g_six_l548_54889


namespace NUMINAMATH_GPT_tan_five_pi_over_four_eq_one_l548_54835

theorem tan_five_pi_over_four_eq_one : Real.tan (5 * Real.pi / 4) = 1 :=
by sorry

end NUMINAMATH_GPT_tan_five_pi_over_four_eq_one_l548_54835


namespace NUMINAMATH_GPT_Chandler_saves_enough_l548_54844

theorem Chandler_saves_enough (total_cost gift_money weekly_earnings : ℕ)
  (h_cost : total_cost = 550)
  (h_gift : gift_money = 130)
  (h_weekly : weekly_earnings = 18) : ∃ x : ℕ, (130 + 18 * x) >= 550 ∧ x = 24 := 
by
  sorry

end NUMINAMATH_GPT_Chandler_saves_enough_l548_54844


namespace NUMINAMATH_GPT_solution_set_quadratic_inequality_l548_54898

theorem solution_set_quadratic_inequality :
  { x : ℝ | x^2 + 3 * x - 4 < 0 } = { x : ℝ | -4 < x ∧ x < 1 } :=
sorry

end NUMINAMATH_GPT_solution_set_quadratic_inequality_l548_54898


namespace NUMINAMATH_GPT_kenya_peanut_count_l548_54804

-- Define the number of peanuts Jose has
def jose_peanuts : ℕ := 85

-- Define the number of additional peanuts Kenya has more than Jose
def additional_peanuts : ℕ := 48

-- Define the number of peanuts Kenya has
def kenya_peanuts : ℕ := jose_peanuts + additional_peanuts

-- Theorem to prove the number of peanuts Kenya has
theorem kenya_peanut_count : kenya_peanuts = 133 := by
  sorry

end NUMINAMATH_GPT_kenya_peanut_count_l548_54804
