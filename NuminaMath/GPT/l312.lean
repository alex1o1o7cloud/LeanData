import Mathlib

namespace angle_between_foci_l312_312482

variables (P F1 F2 : ℝ × ℝ)
-- Ellipse centered at origin with semi-major axis 4 and semi-minor axis 3
def on_ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 16 + P.2^2 / 9 = 1)

-- Foci of the ellipse at (-2√7, 0) and (2√7, 0)
def is_focus (F : ℝ × ℝ) : Prop :=
  F = (-2 * Real.sqrt 7, 0) ∨ F = (2 * Real.sqrt 7, 0)

-- Given condition |PF1| * |PF2| = 12
def product_of_distances (P F1 F2 : ℝ × ℝ) : Prop :=
  (Real.dist P F1) * (Real.dist P F2) = 12

-- Proof statement
theorem angle_between_foci (h1: on_ellipse P) (h2: is_focus F1) (h3: is_focus F2) (h4: product_of_distances P F1 F2) :
  ∃ θ : ℝ, θ = 60 ∧ cos θ = (40 - 28) / (2 * 12) :=
sorry

end angle_between_foci_l312_312482


namespace cylinder_volume_ratio_l312_312530

theorem cylinder_volume_ratio (h1 h2 r1 r2 V1 V2 : ℝ)
  (h1_eq : h1 = 9)
  (h2_eq : h2 = 6)
  (circumference1_eq : 2 * π * r1 = 6)
  (circumference2_eq : 2 * π * r2 = 9)
  (V1_eq : V1 = π * r1^2 * h1)
  (V2_eq : V2 = π * r2^2 * h2)
  (V1_calculated : V1 = 81 / π)
  (V2_calculated : V2 = 243 / (4 * π)) :
  (max V1 V2) / (min V1 V2) = 3 / 4 :=
by
  sorry

end cylinder_volume_ratio_l312_312530


namespace polynomial_real_roots_abs_c_geq_2_l312_312809

-- Definition of the polynomial P(x)
def P (x : ℝ) (a b c : ℝ) : ℝ := x^6 + a*x^5 + b*x^4 + c*x^3 + b*x^2 + a*x + 1

-- Statement of the problem: Given that P(x) has six distinct real roots, prove |c| ≥ 2
theorem polynomial_real_roots_abs_c_geq_2 (a b c : ℝ) :
  (∃ r1 r2 r3 r4 r5 r6 : ℝ, r1 ≠ r2 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r1 ≠ r5 ∧ r1 ≠ r6 ∧
                           r2 ≠ r3 ∧ r2 ≠ r4 ∧ r2 ≠ r5 ∧ r2 ≠ r6 ∧
                           r3 ≠ r4 ∧ r3 ≠ r5 ∧ r3 ≠ r6 ∧
                           r4 ≠ r5 ∧ r4 ≠ r6 ∧
                           r5 ≠ r6 ∧
                           P r1 a b c = 0 ∧ P r2 a b c = 0 ∧ P r3 a b c = 0 ∧
                           P r4 a b c = 0 ∧ P r5 a b c = 0 ∧ P r6 a b c = 0) →
  |c| ≥ 2 := by
  sorry

end polynomial_real_roots_abs_c_geq_2_l312_312809


namespace increasing_interval_l312_312047

noncomputable def y (x : ℝ) : ℝ := x * Real.cos x - Real.sin x

def is_monotonic_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x1 x2, a < x1 ∧ x1 < x2 ∧ x2 < b → f x1 < f x2

theorem increasing_interval :
  is_monotonic_increasing y π (2 * π) :=
by
  -- Proof would go here
  sorry

end increasing_interval_l312_312047


namespace shelves_fit_l312_312171

-- Define the total space of the room for the library
def totalSpace : ℕ := 400

-- Define the space each bookshelf takes up
def spacePerBookshelf : ℕ := 80

-- Define the reserved space for desk and walking area
def reservedSpace : ℕ := 160

-- Define the space available for bookshelves
def availableSpace : ℕ := totalSpace - reservedSpace

-- Define the number of bookshelves that can fit in the available space
def numberOfBookshelves : ℕ := availableSpace / spacePerBookshelf

-- The theorem stating the number of bookshelves Jonas can fit in the room
theorem shelves_fit : numberOfBookshelves = 3 := by
  -- We can defer the proof as we only need the statement for now
  sorry

end shelves_fit_l312_312171


namespace sin_70_eq_1_minus_2k_squared_l312_312132

theorem sin_70_eq_1_minus_2k_squared (k : ℝ) (h : Real.sin 10 = k) : Real.sin 70 = 1 - 2 * k^2 := 
by
  sorry

end sin_70_eq_1_minus_2k_squared_l312_312132


namespace reflected_ray_eqn_l312_312140

theorem reflected_ray_eqn (P : ℝ × ℝ)
  (incident_ray : ∀ x : ℝ, P.2 = 2 * P.1 + 1)
  (reflection_line : P.2 = P.1) :
  P.1 - 2 * P.2 - 1 = 0 :=
sorry

end reflected_ray_eqn_l312_312140


namespace circle_problems_satisfy_conditions_l312_312457

noncomputable def circle1_center_x := 11
noncomputable def circle1_center_y := 8
noncomputable def circle1_radius_squared := 87

noncomputable def circle2_center_x := 14
noncomputable def circle2_center_y := -3
noncomputable def circle2_radius_squared := 168

theorem circle_problems_satisfy_conditions :
  (∀ x y, (x-11)^2 + (y-8)^2 = 87 ∨ (x-14)^2 + (y+3)^2 = 168) := sorry

end circle_problems_satisfy_conditions_l312_312457


namespace factor_expression_l312_312279

theorem factor_expression (x : ℝ) : 5 * x^2 * (x - 2) - 9 * (x - 2) = (x - 2) * (5 * x^2 - 9) :=
sorry

end factor_expression_l312_312279


namespace problem_equivalent_final_answer_l312_312011

noncomputable def a := 12
noncomputable def b := 27
noncomputable def c := 6

theorem problem_equivalent :
  2 * Real.sqrt 3 + (2 / Real.sqrt 3) + 3 * Real.sqrt 2 + (3 / Real.sqrt 2) = (a * Real.sqrt 3 + b * Real.sqrt 2) / c :=
  sorry

theorem final_answer :
  a + b + c = 45 :=
  by
    unfold a b c
    simp
    done

end problem_equivalent_final_answer_l312_312011


namespace quadratic_function_inequality_l312_312303

variable (a x x₁ x₂ : ℝ)

def f (x : ℝ) := a * x^2 + 2 * a * x + 4

theorem quadratic_function_inequality
  (h₀ : 0 < a) (h₁ : a < 3)
  (h₂ : x₁ + x₂ = 0)
  (h₃ : x₁ < x₂) :
  f a x₁ < f a x₂ := 
sorry

end quadratic_function_inequality_l312_312303


namespace relay_race_total_time_l312_312995

noncomputable def mary_time (susan_time : ℕ) : ℕ := 2 * susan_time
noncomputable def susan_time (jen_time : ℕ) : ℕ := jen_time + 10
def jen_time : ℕ := 30
noncomputable def tiffany_time (mary_time : ℕ) : ℕ := mary_time - 7

theorem relay_race_total_time :
  let mary_time := mary_time (susan_time jen_time)
  let susan_time := susan_time jen_time
  let tiffany_time := tiffany_time mary_time
  mary_time + susan_time + jen_time + tiffany_time = 223 := by
  sorry

end relay_race_total_time_l312_312995


namespace ellipse_eccentricity_m_l312_312756

theorem ellipse_eccentricity_m (m : ℝ) (e : ℝ) (h1 : ∀ x y : ℝ, x^2 / m + y^2 = 1) (h2 : e = Real.sqrt 3 / 2) :
  m = 4 ∨ m = 1 / 4 :=
by sorry

end ellipse_eccentricity_m_l312_312756


namespace tan_sum_formula_l312_312904

open Real

theorem tan_sum_formula (α : ℝ) (hα : π < α ∧ α < 3 * π / 2) (h_cos_2α : cos (2 * α) = -3 / 5) :
  tan (π / 4 + 2 * α) = -1 / 7 :=
by
  -- Insert the proof here
  sorry

end tan_sum_formula_l312_312904


namespace fill_time_difference_correct_l312_312086

-- Define the time to fill one barrel in normal conditions
def normal_fill_time_per_barrel : ℕ := 3

-- Define the time to fill one barrel with a leak
def leak_fill_time_per_barrel : ℕ := 5

-- Define the number of barrels to fill
def barrels_to_fill : ℕ := 12

-- Define the time to fill 12 barrels in normal conditions
def normal_fill_time : ℕ := normal_fill_time_per_barrel * barrels_to_fill

-- Define the time to fill 12 barrels with a leak
def leak_fill_time : ℕ := leak_fill_time_per_barrel * barrels_to_fill

-- Define the time difference
def time_difference : ℕ := leak_fill_time - normal_fill_time

theorem fill_time_difference_correct : time_difference = 24 := by
  sorry

end fill_time_difference_correct_l312_312086


namespace number_of_possible_values_of_b_l312_312491

theorem number_of_possible_values_of_b
  (b : ℕ) 
  (h₁ : 4 ∣ b) 
  (h₂ : b ∣ 24)
  (h₃ : 0 < b) :
  { n : ℕ | 4 ∣ n ∧ n ∣ 24 ∧ 0 < n }.card = 4 :=
sorry

end number_of_possible_values_of_b_l312_312491


namespace equation1_unique_solutions_equation2_unique_solutions_l312_312037

noncomputable def solve_equation1 : ℝ → Prop :=
fun x => x ^ 2 - 4 * x + 1 = 0

noncomputable def solve_equation2 : ℝ → Prop :=
fun x => 2 * x ^ 2 - 3 * x + 1 = 0

theorem equation1_unique_solutions :
  ∀ x, solve_equation1 x ↔ (x = 2 + Real.sqrt 3) ∨ (x = 2 - Real.sqrt 3) := by
  sorry

theorem equation2_unique_solutions :
  ∀ x, solve_equation2 x ↔ (x = 1) ∨ (x = 1 / 2) := by
  sorry

end equation1_unique_solutions_equation2_unique_solutions_l312_312037


namespace minimal_hair_loss_l312_312395

theorem minimal_hair_loss (cards : Fin 100 → ℕ)
    (sum_sage1 : ℕ)
    (communicate_card_numbers : List ℕ)
    (communicate_sum : ℕ) :
    (∀ i : Fin 100, (communicate_card_numbers.contains (cards i))) →
    communicate_sum = sum_sage1 →
    sum_sage1 = List.sum communicate_card_numbers →
    communicate_card_numbers.length = 100 →
    ∃ (minimal_loss : ℕ), minimal_loss = 101 := by
  sorry

end minimal_hair_loss_l312_312395


namespace hip_hop_final_percentage_is_39_l312_312506

noncomputable def hip_hop_percentage (total_songs percentage_country: ℝ):
  ℝ :=
  let percentage_non_country := 1 - percentage_country
  let original_ratio_hip_hop := 0.65
  let original_ratio_pop := 0.35
  let total_non_country := original_ratio_hip_hop + original_ratio_pop
  let hip_hop_percentage := original_ratio_hip_hop / total_non_country * percentage_non_country
  hip_hop_percentage

theorem hip_hop_final_percentage_is_39 (total_songs : ℕ) :
  hip_hop_percentage total_songs 0.40 = 0.39 :=
by
  sorry

end hip_hop_final_percentage_is_39_l312_312506


namespace hypotenuse_is_correct_l312_312059

noncomputable def hypotenuse_of_right_triangle (a b : ℕ) : ℝ :=
  Real.sqrt (a^2 + b^2)

theorem hypotenuse_is_correct :
  hypotenuse_of_right_triangle 140 210 = 70 * Real.sqrt 13 :=
by
  sorry

end hypotenuse_is_correct_l312_312059


namespace expand_binomials_l312_312384

theorem expand_binomials :
  (a + 2 * b) ^ 3 = a ^ 3 + 6 * a ^ 2 * b + 12 * a * b ^ 2 + 8 * b ^ 3 ∧
  (5 * a - b) ^ 3 = 125 * a ^ 3 - 75 * a ^ 2 * b + 15 * a * b ^ 2 - b ^ 3 ∧
  (2 * a + 3 * b) ^ 3 = 8 * a ^ 3 + 36 * a ^ 2 * b + 54 * a * b ^ 2 + 27 * b ^ 3 ∧
  (m ^ 3 - n ^ 2) ^ 3 = m ^ 9 - 3 * m ^ 6 * n ^ 2 + 3 * m ^ 3 * n ^ 4 - n ^ 6 :=
by
  sorry

end expand_binomials_l312_312384


namespace line_does_not_pass_third_quadrant_l312_312309

theorem line_does_not_pass_third_quadrant (a b c x y : ℝ) (h_ac : a * c < 0) (h_bc : b * c < 0) :
  ¬(x < 0 ∧ y < 0 ∧ a * x + b * y + c = 0) :=
sorry

end line_does_not_pass_third_quadrant_l312_312309


namespace work_day_percentage_l312_312476

theorem work_day_percentage 
  (work_day_hours : ℕ) 
  (first_meeting_minutes : ℕ) 
  (second_meeting_factor : ℕ) 
  (h_work_day : work_day_hours = 10) 
  (h_first_meeting : first_meeting_minutes = 60) 
  (h_second_meeting_factor : second_meeting_factor = 2) :
  ((first_meeting_minutes + second_meeting_factor * first_meeting_minutes) / (work_day_hours * 60) : ℚ) * 100 = 30 :=
sorry

end work_day_percentage_l312_312476


namespace mean_temperature_correct_l312_312041

-- Define the condition (temperatures)
def temperatures : List Int :=
  [-6, -3, -3, -4, 2, 4, 1]

-- Define the total number of days
def num_days : ℕ := 7

-- Define the expected mean temperature
def expected_mean : Rat := (-6 : Int) / (7 : Int)

-- State the theorem that we need to prove
theorem mean_temperature_correct :
  (temperatures.sum : Rat) / (num_days : Rat) = expected_mean := 
by
  sorry

end mean_temperature_correct_l312_312041


namespace ratio_bee_eaters_leopards_l312_312104

variables (s f l c a t e r : ℕ)

-- Define the conditions from the problem.
def conditions : Prop :=
  s = 100 ∧
  f = 80 ∧
  l = 20 ∧
  c = s / 2 ∧
  a = 2 * (f + l) ∧
  t = 670 ∧
  e = t - (s + f + l + c + a)

-- The theorem statement proving the ratio.
theorem ratio_bee_eaters_leopards (h : conditions s f l c a t e) : r = (e / l) := by
  sorry

end ratio_bee_eaters_leopards_l312_312104


namespace polygon_perpendiculars_length_l312_312887

noncomputable def RegularPolygon := { n : ℕ // n ≥ 3 }

structure Perpendiculars (P : RegularPolygon) (i : ℕ) :=
  (d_i     : ℝ)
  (d_i_minus_1 : ℝ)
  (d_i_plus_1 : ℝ)
  (line_crosses_interior : Bool)

theorem polygon_perpendiculars_length {P : RegularPolygon} {i : ℕ}
  (hyp : Perpendiculars P i) :
  hyp.d_i = if hyp.line_crosses_interior 
            then hyp.d_i_minus_1 + hyp.d_i_plus_1 
            else abs (hyp.d_i_minus_1 - hyp.d_i_plus_1) :=
sorry

end polygon_perpendiculars_length_l312_312887


namespace dogs_prevent_wolf_escape_l312_312078

theorem dogs_prevent_wolf_escape
  (wolf_speed dog_speed : ℝ)
  (at_center: True)
  (dogs_at_vertices: True)
  (wolf_all_over_field: True)
  (dogs_on_perimeter: True)
  (wolf_handles_one_dog: ∀ (d : ℕ), d = 1 → True)
  (wolf_handles_two_dogs: ∀ (d : ℕ), d = 2 → False)
  (dog_faster_than_wolf: dog_speed = 1.5 * wolf_speed) : 
  ∀ (wolf_position : ℝ × ℝ) (boundary_position : ℝ × ℝ), 
  wolf_position != boundary_position → dog_speed > wolf_speed → 
  False := 
by sorry

end dogs_prevent_wolf_escape_l312_312078


namespace library_books_l312_312940

/-- Last year, the school library purchased 50 new books. 
    This year, it purchased 3 times as many books. 
    If the library had 100 books before it purchased new books last year,
    prove that the library now has 300 books in total. -/
theorem library_books (initial_books : ℕ) (last_year_books : ℕ) (multiplier : ℕ)
  (h1 : initial_books = 100) (h2 : last_year_books = 50) (h3 : multiplier = 3) :
  initial_books + last_year_books + (multiplier * last_year_books) = 300 := 
sorry

end library_books_l312_312940


namespace both_players_score_same_points_l312_312463

theorem both_players_score_same_points :
  let P_A_score := 0.5 
  let P_B_score := 0.8 
  let P_A_miss := 1 - P_A_score
  let P_B_miss := 1 - P_B_score
  let P_both_miss := P_A_miss * P_B_miss
  let P_both_score := P_A_score * P_B_score
  let P_same_points := P_both_miss + P_both_score
  P_same_points = 0.5 := 
by {
  -- Actual proof should be here
  sorry
}

end both_players_score_same_points_l312_312463


namespace percentage_increase_after_lawnmower_l312_312551

-- Definitions from conditions
def initial_daily_yards := 8
def weekly_yards_after_lawnmower := 84
def days_in_week := 7

-- Problem statement
theorem percentage_increase_after_lawnmower : 
  ((weekly_yards_after_lawnmower / days_in_week - initial_daily_yards) / initial_daily_yards) * 100 = 50 := 
by 
  sorry

end percentage_increase_after_lawnmower_l312_312551


namespace find_square_number_divisible_by_three_between_90_and_150_l312_312416

theorem find_square_number_divisible_by_three_between_90_and_150 :
  ∃ x : ℕ, 90 < x ∧ x < 150 ∧ ∃ y : ℕ, x = y * y ∧ 3 ∣ x ∧ x = 144 := 
by 
  sorry

end find_square_number_divisible_by_three_between_90_and_150_l312_312416


namespace total_cost_for_tickets_l312_312999

-- Define the known quantities
def students : Nat := 20
def teachers : Nat := 3
def ticket_cost : Nat := 5

-- Define the total number of people
def total_people : Nat := students + teachers

-- Define the total cost
def total_cost : Nat := total_people * ticket_cost

-- Prove that the total cost is $115
theorem total_cost_for_tickets : total_cost = 115 := by
  -- Sorry is used here to skip the proof
  sorry

end total_cost_for_tickets_l312_312999


namespace total_embroidery_time_l312_312266

-- Defining the constants as given in the problem
def stitches_per_minute : ℕ := 4
def stitches_per_flower : ℕ := 60
def stitches_per_unicorn : ℕ := 180
def stitches_per_godzilla : ℕ := 800
def num_flowers : ℕ := 50
def num_unicorns : ℕ := 3
def num_godzillas : ℕ := 1 -- Implicitly from the problem statement

-- Total time calculation as a Lean theorem
theorem total_embroidery_time : 
  (stitches_per_godzilla * num_godzillas + 
   stitches_per_unicorn * num_unicorns + 
   stitches_per_flower * num_flowers) / stitches_per_minute = 1085 := 
by
  sorry

end total_embroidery_time_l312_312266


namespace number_of_bowls_l312_312846

noncomputable theory
open Classical

theorem number_of_bowls (n : ℕ) 
  (h1 : 8 * 12 = 6 * n) : n = 16 := 
by
  sorry

end number_of_bowls_l312_312846


namespace bailey_rawhide_bones_l312_312550

variable (dog_treats : ℕ) (chew_toys : ℕ) (total_items : ℕ)
variable (credit_cards : ℕ) (items_per_card : ℕ)

theorem bailey_rawhide_bones :
  (dog_treats = 8) →
  (chew_toys = 2) →
  (credit_cards = 4) →
  (items_per_card = 5) →
  (total_items = credit_cards * items_per_card) →
  (total_items - (dog_treats + chew_toys) = 10) :=
by
  intros
  sorry

end bailey_rawhide_bones_l312_312550


namespace number_of_bowls_l312_312844

-- Let n be the number of bowls on the table.
variable (n : ℕ)

-- Condition 1: There are n bowls, and each contain some grapes.
-- Condition 2: Adding 8 grapes to each of 12 specific bowls increases the average number of grapes in all bowls by 6.
-- Let's formalize the condition given in the problem
theorem number_of_bowls (h1 : 12 * 8 = 96) (h2 : 6 * n = 96) : n = 16 :=
by
  -- omitting the proof here
  sorry

end number_of_bowls_l312_312844


namespace solve_quadratic_l312_312496

theorem solve_quadratic (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : ∀ x : ℝ, x^2 + c*x + d = 0 ↔ x = c ∨ x = d) : (c, d) = (1, -2) :=
sorry

end solve_quadratic_l312_312496


namespace gcd_m_n_l312_312630

noncomputable def m : ℕ := 5 * 11111111
noncomputable def n : ℕ := 111111111

theorem gcd_m_n : gcd m n = 11111111 := by
  sorry

end gcd_m_n_l312_312630


namespace find_m_value_l312_312149

theorem find_m_value (m : ℝ) (A : Set ℝ) (h₁ : A = {0, m, m^2 - 3 * m + 2}) (h₂ : 2 ∈ A) : m = 3 :=
by
  sorry

end find_m_value_l312_312149


namespace quadratic_roots_diff_by_2_l312_312111

theorem quadratic_roots_diff_by_2 (q : ℝ) (hq : 0 < q) :
  (∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 - r2 = 2 ∨ r2 - r1 = 2) ∧ r1 ^ 2 + (2 * q - 1) * r1 + q = 0 ∧ r2 ^ 2 + (2 * q - 1) * r2 + q = 0) ↔
  q = 1 + (Real.sqrt 7) / 2 :=
sorry

end quadratic_roots_diff_by_2_l312_312111


namespace layla_earnings_l312_312351

def rate_donaldsons : ℕ := 15
def bonus_donaldsons : ℕ := 5
def hours_donaldsons : ℕ := 7
def rate_merck : ℕ := 18
def discount_merck : ℝ := 0.10
def hours_merck : ℕ := 6
def rate_hille : ℕ := 20
def bonus_hille : ℕ := 10
def hours_hille : ℕ := 3
def rate_johnson : ℕ := 22
def flat_rate_johnson : ℕ := 80
def hours_johnson : ℕ := 4
def rate_ramos : ℕ := 25
def bonus_ramos : ℕ := 20
def hours_ramos : ℕ := 2

def donaldsons_earnings := rate_donaldsons * hours_donaldsons + bonus_donaldsons
def merck_earnings := rate_merck * hours_merck - (rate_merck * hours_merck * discount_merck : ℝ)
def hille_earnings := rate_hille * hours_hille + bonus_hille
def johnson_earnings := rate_johnson * hours_johnson
def ramos_earnings := rate_ramos * hours_ramos + bonus_ramos

noncomputable def total_earnings : ℝ :=
  donaldsons_earnings + merck_earnings + hille_earnings + johnson_earnings + ramos_earnings

theorem layla_earnings : total_earnings = 435.2 :=
by
  sorry

end layla_earnings_l312_312351


namespace find_r_over_s_at_0_l312_312202

noncomputable def r (x : ℝ) : ℝ := -3 * (x + 1) * (x - 2)
noncomputable def s (x : ℝ) : ℝ := (x + 1) * (x - 3)

theorem find_r_over_s_at_0 : (r 0) / (s 0) = 2 := by
  sorry

end find_r_over_s_at_0_l312_312202


namespace absolute_difference_of_integers_l312_312654

theorem absolute_difference_of_integers (x y : ℤ) (h1 : (x + y) / 2 = 15) (h2 : Int.sqrt (x * y) + 6 = 15) : |x - y| = 24 :=
  sorry

end absolute_difference_of_integers_l312_312654


namespace find_c_l312_312200

theorem find_c (x c : ℝ) (h1 : 3 * x + 8 = 5) (h2 : c * x - 15 = -3) : c = -12 := 
by
  -- Equations and conditions
  have h1 : 3 * x + 8 = 5 := h1
  have h2 : c * x - 15 = -3 := h2
  -- The proof script would go here
  sorry

end find_c_l312_312200


namespace function_additive_of_tangential_property_l312_312806

open Set

variable {f : ℝ → ℝ}

def is_tangential_quadrilateral_sides (a b c d : ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ (a + c = b + d)

theorem function_additive_of_tangential_property
  (h : ∀ (a b c d : ℝ), is_tangential_quadrilateral_sides a b c d → f (a + b + c + d) = f a + f b + f c + f d) :
  ∀ (x y : ℝ), 0 < x → 0 < y → f (x + y) = f x + f y :=
by
  sorry

end function_additive_of_tangential_property_l312_312806


namespace simplify_expression_l312_312036

variable {x y z : ℝ}

theorem simplify_expression (h : x^2 - y^2 ≠ 0) (hx : x ≠ 0) (hz : z ≠ 0) :
  (x^2 - y^2)⁻¹ * (x⁻¹ - z⁻¹) = (z - x) * x⁻¹ * z⁻¹ * (x^2 - y^2)⁻¹ := by
  sorry

end simplify_expression_l312_312036


namespace larger_number_l312_312214

theorem larger_number (x y : ℕ) (h1 : x + y = 28) (h2 : x - y = 4) : max x y = 16 := by
  sorry

end larger_number_l312_312214


namespace x_squared_plus_y_squared_l312_312918

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 10) : x^2 + y^2 = 29 := 
by
  sorry

end x_squared_plus_y_squared_l312_312918


namespace find_n_largest_binomial_coefficient_term_constant_term_in_expansion_l312_312746

noncomputable def sum_binomial_first_three_terms (n : ℕ) : ℕ :=
  nat.choose n 0 + nat.choose n 1 + nat.choose n 2

theorem find_n (n : ℕ) (hn1 : sum_binomial_first_three_terms n = 22) : n = 6 := by
  sorry

theorem largest_binomial_coefficient_term (n : ℕ) (hn : n = 6) : 
  (nat.choose n 3) * 2^6 * (2^3) * x^(3 / 2) = 1280 * x^(3 / 2) := by
  sorry

theorem constant_term_in_expansion (n : ℕ) (hn : n = 6) :
  (nat.choose n 4) * 2^6 = 960 := by
  sorry

end find_n_largest_binomial_coefficient_term_constant_term_in_expansion_l312_312746


namespace parallel_line_through_P_perpendicular_line_through_P_l312_312428

-- Define point P
def P := (-4, 2)

-- Define line l
def l (x y : ℝ) := 3 * x - 2 * y - 7 = 0

-- Define the equation of the line parallel to l that passes through P
def parallel_line (x y : ℝ) := 3 * x - 2 * y + 16 = 0

-- Define the equation of the line perpendicular to l that passes through P
def perpendicular_line (x y : ℝ) := 2 * x + 3 * y + 2 = 0

-- Theorem 1: Prove that parallel_line is the equation of the line passing through P and parallel to l
theorem parallel_line_through_P :
  ∀ (x y : ℝ), 
    (parallel_line x y → x = -4 ∧ y = 2) :=
sorry

-- Theorem 2: Prove that perpendicular_line is the equation of the line passing through P and perpendicular to l
theorem perpendicular_line_through_P :
  ∀ (x y : ℝ), 
    (perpendicular_line x y → x = -4 ∧ y = 2) :=
sorry

end parallel_line_through_P_perpendicular_line_through_P_l312_312428


namespace triangles_from_sticks_l312_312487

theorem triangles_from_sticks (a1 a2 a3 a4 a5 a6 : ℕ) (h_diff: a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5 ∧ a1 ≠ a6 
∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ a5 ∧ a2 ≠ a6 
∧ a3 ≠ a4 ∧ a3 ≠ a5 ∧ a3 ≠ a6 
∧ a4 ≠ a5 ∧ a4 ≠ a6 
∧ a5 ≠ a6) (h_order: a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5 ∧ a5 < a6) : 
  (a1 + a3 > a5 ∧ a1 + a5 > a3 ∧ a3 + a5 > a1) ∧ 
  (a2 + a4 > a6 ∧ a2 + a6 > a4 ∧ a4 + a6 > a2) :=
by
  sorry

end triangles_from_sticks_l312_312487


namespace initial_oranges_l312_312914

theorem initial_oranges (left_oranges taken_oranges : ℕ) (h1 : left_oranges = 25) (h2 : taken_oranges = 35) : 
  left_oranges + taken_oranges = 60 := 
by 
  sorry

end initial_oranges_l312_312914


namespace combined_height_of_trees_l312_312337

noncomputable def growth_rate_A (weeks : ℝ) : ℝ := (weeks / 2) * 50
noncomputable def growth_rate_B (weeks : ℝ) : ℝ := (weeks / 3) * 70
noncomputable def growth_rate_C (weeks : ℝ) : ℝ := (weeks / 4) * 90
noncomputable def initial_height_A : ℝ := 200
noncomputable def initial_height_B : ℝ := 150
noncomputable def initial_height_C : ℝ := 250
noncomputable def total_weeks : ℝ := 16
noncomputable def total_growth_A := growth_rate_A total_weeks
noncomputable def total_growth_B := growth_rate_B total_weeks
noncomputable def total_growth_C := growth_rate_C total_weeks
noncomputable def final_height_A := initial_height_A + total_growth_A
noncomputable def final_height_B := initial_height_B + total_growth_B
noncomputable def final_height_C := initial_height_C + total_growth_C
noncomputable def final_combined_height := final_height_A + final_height_B + final_height_C

theorem combined_height_of_trees :
  final_combined_height = 1733.33 := by
  sorry

end combined_height_of_trees_l312_312337


namespace brian_tape_needed_l312_312410

-- Define lengths and number of each type of box
def long_side_15_30 := 32
def short_side_15_30 := 17
def num_15_30 := 5

def side_40_40 := 42
def num_40_40 := 2

def long_side_20_50 := 52
def short_side_20_50 := 22
def num_20_50 := 3

-- Calculate the total tape required
def total_tape : Nat :=
  (num_15_30 * (long_side_15_30 + 2 * short_side_15_30)) +
  (num_40_40 * (3 * side_40_40)) +
  (num_20_50 * (long_side_20_50 + 2 * short_side_20_50))

-- Proof statement
theorem brian_tape_needed : total_tape = 870 := by
  sorry

end brian_tape_needed_l312_312410


namespace sum_of_ten_numbers_l312_312315

theorem sum_of_ten_numbers (average count : ℝ) (h_avg : average = 5.3) (h_count : count = 10) : 
  average * count = 53 :=
by
  sorry

end sum_of_ten_numbers_l312_312315


namespace sum_diff_reciprocals_equals_zero_l312_312440

theorem sum_diff_reciprocals_equals_zero
  (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : (1 / (a + 1)) + (1 / (a - 1)) + (1 / (b + 1)) + (1 / (b - 1)) = 0) :
  (a + b) - (1 / a + 1 / b) = 0 :=
by
  sorry

end sum_diff_reciprocals_equals_zero_l312_312440


namespace initial_concentration_is_27_l312_312662

-- Define given conditions
variables (m m_c : ℝ) -- initial mass of solution and salt
variables (x : ℝ) -- initial percentage concentration of salt
variables (h1 : m_c = (x / 100) * m) -- initial concentration definition
variables (h2 : m > 0) (h3 : x > 0) -- non-zero positive mass and concentration

theorem initial_concentration_is_27 (h_evaporated : (m / 5) * 2 * (x / 100) = m_c) 
  (h_new_concentration : (x + 3) = (m_c * 100) / (9 * m / 10)) 
  : x = 27 :=
by
  sorry

end initial_concentration_is_27_l312_312662


namespace prob_2022_2023_l312_312777

theorem prob_2022_2023 (n : ℤ) (h : (n - 2022)^2 + (2023 - n)^2 = 1) : (n - 2022) * (2023 - n) = 0 :=
sorry

end prob_2022_2023_l312_312777


namespace dogs_in_kennel_l312_312967

theorem dogs_in_kennel (C D : ℕ) (h1 : C = D - 8) (h2 : C * 4 = 3 * D) : D = 32 :=
sorry

end dogs_in_kennel_l312_312967


namespace correct_q_solution_l312_312488

noncomputable def solve_q (n m q : ℕ) : Prop :=
  (7 / 8 : ℚ) = (n / 96 : ℚ) ∧
  (7 / 8 : ℚ) = ((m + n) / 112 : ℚ) ∧
  (7 / 8 : ℚ) = ((q - m) / 144 : ℚ) ∧
  n = 84 ∧
  m = 14 →
  q = 140

theorem correct_q_solution : ∃ (q : ℕ), solve_q 84 14 q :=
by sorry

end correct_q_solution_l312_312488


namespace sum_first_9_terms_arithmetic_sequence_l312_312935

noncomputable def sum_of_first_n_terms (a_1 d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a_1 + (n - 1) * d) / 2

def arithmetic_sequence_term (a_1 d : ℤ) (n : ℕ) : ℤ :=
  a_1 + (n - 1) * d

theorem sum_first_9_terms_arithmetic_sequence :
  ∃ a_1 d : ℤ, (a_1 + arithmetic_sequence_term a_1 d 4 + arithmetic_sequence_term a_1 d 7 = 39) ∧
               (arithmetic_sequence_term a_1 d 3 + arithmetic_sequence_term a_1 d 6 + arithmetic_sequence_term a_1 d 9 = 27) ∧
               (sum_of_first_n_terms a_1 d 9 = 99) :=
by
  sorry

end sum_first_9_terms_arithmetic_sequence_l312_312935


namespace number_of_five_digit_numbers_with_at_least_one_zero_l312_312443

-- Definitions for the conditions
def total_five_digit_numbers : ℕ := 90000
def five_digit_numbers_with_no_zeros : ℕ := 59049

-- Theorem stating that the number of 5-digit numbers with at least one zero is 30,951
theorem number_of_five_digit_numbers_with_at_least_one_zero : 
    total_five_digit_numbers - five_digit_numbers_with_no_zeros = 30951 :=
by
  sorry

end number_of_five_digit_numbers_with_at_least_one_zero_l312_312443


namespace bus_costs_unique_min_buses_cost_A_l312_312860

-- Defining the main conditions
def condition1 (x y : ℕ) : Prop := x + 2 * y = 300
def condition2 (x y : ℕ) : Prop := 2 * x + y = 270

-- Part 1: Proving individual bus costs
theorem bus_costs_unique (x y : ℕ) (h1 : condition1 x y) (h2 : condition2 x y) :
  x = 80 ∧ y = 110 := 
by 
  sorry

-- Part 2: Minimum buses of type A and total cost constraint
def total_buses := 10
def total_cost (x y a : ℕ) : Prop := 
  x * a + y * (total_buses - a) ≤ 1000

theorem min_buses_cost_A (x y : ℕ) (hx : x = 80) (hy : y = 110) :
  ∃ a cost, total_cost x y a ∧ a >= 4 ∧ cost = x * 4 + y * (total_buses - 4) ∧ cost = 980 :=
by
  sorry

end bus_costs_unique_min_buses_cost_A_l312_312860


namespace dave_more_than_derek_l312_312558

def derek_initial : ℕ := 40
def derek_spent_on_self1 : ℕ := 14
def derek_spent_on_dad : ℕ := 11
def derek_spent_on_self2 : ℕ := 5

def dave_initial : ℕ := 50
def dave_spent_on_mom : ℕ := 7

def derek_remaining : ℕ := derek_initial - (derek_spent_on_self1 + derek_spent_on_dad + derek_spent_on_self2)
def dave_remaining : ℕ := dave_initial - dave_spent_on_mom

theorem dave_more_than_derek : dave_remaining - derek_remaining = 33 :=
by
  -- The proof goes here
  sorry

end dave_more_than_derek_l312_312558


namespace number_of_bowls_l312_312822

theorem number_of_bowls (n : ℕ) (h : 8 * 12 = 96) (avg_increase : 6 * n = 96) : n = 16 :=
by {
  sorry
}

end number_of_bowls_l312_312822


namespace distance_from_home_to_high_school_l312_312466

theorem distance_from_home_to_high_school 
  (total_mileage track_distance d : ℝ)
  (h_total_mileage : total_mileage = 10)
  (h_track : track_distance = 4)
  (h_eq : d + d + track_distance = total_mileage) :
  d = 3 :=
by sorry

end distance_from_home_to_high_school_l312_312466


namespace binomial_param_exact_l312_312399

variable (ξ : ℕ → ℝ) (n : ℕ) (p : ℝ)

-- Define the conditions: expectation and variance
axiom expectation_eq : n * p = 3
axiom variance_eq : n * p * (1 - p) = 2

-- Statement to prove
theorem binomial_param_exact (h1 : n * p = 3) (h2 : n * p * (1 - p) = 2) : p = 1 / 3 :=
by
  rw [expectation_eq] at h2
  sorry

end binomial_param_exact_l312_312399


namespace probability_correct_l312_312857

/-- 
The set of characters in "HMMT2005".
-/
def characters : List Char := ['H', 'M', 'M', 'T', '2', '0', '0', '5']

/--
The number of ways to choose 4 positions out of 8.
-/
def choose_4_from_8 : ℕ := Nat.choose 8 4

/-- 
The factorial of an integer n.
-/
def factorial : ℕ → ℕ
| 0     => 1
| (n+1) => (n+1) * factorial n

/-- 
The number of ways to arrange "HMMT".
-/
def arrangements_hmmt : ℕ := choose_4_from_8 * (factorial 4 / factorial 2)

/-- 
The number of ways to arrange "2005".
-/
def arrangements_2005 : ℕ := choose_4_from_8 * (factorial 4 / factorial 2)

/-- 
The number of arrangements where both "HMMT" and "2005" appear.
-/
def arrangements_both : ℕ := choose_4_from_8

/-- 
The total number of possible arrangements of "HMMT2005".
-/
def total_arrangements : ℕ := factorial 8 / (factorial 2 * factorial 2)

/-- 
The number of desirable arrangements using inclusion-exclusion.
-/
def desirable_arrangements : ℕ := arrangements_hmmt + arrangements_2005 - arrangements_both

/-- 
The probability of being able to read either "HMMT" or "2005" 
in a random arrangement of "HMMT2005".
-/
def probability : ℚ := (desirable_arrangements : ℚ) / (total_arrangements : ℚ)

/-- 
Prove that the computed probability is equal to 23/144.
-/
theorem probability_correct : probability = 23 / 144 := sorry

end probability_correct_l312_312857


namespace eval_expr_l312_312741

theorem eval_expr : (3^3)^2 = 729 := 
by
  sorry

end eval_expr_l312_312741


namespace quadratic_root_value_l312_312451

theorem quadratic_root_value (m : ℝ) :
  ∃ m, (∀ x, x^2 - m * x - 3 = 0 → x = -2) → m = -1/2 :=
by
  sorry

end quadratic_root_value_l312_312451


namespace prime_p_geq_7_div_240_l312_312031

theorem prime_p_geq_7_div_240 (p : ℕ) (hp : Nat.Prime p) (hge7 : p ≥ 7) : 240 ∣ p^4 - 1 := 
sorry

end prime_p_geq_7_div_240_l312_312031


namespace intersect_x_axis_iff_k_le_4_l312_312437

theorem intersect_x_axis_iff_k_le_4 (k : ℝ) :
  (∃ x : ℝ, (k-3) * x^2 + 2 * x + 1 = 0) ↔ k ≤ 4 :=
sorry

end intersect_x_axis_iff_k_le_4_l312_312437


namespace planted_fraction_l312_312568

theorem planted_fraction (a b : ℕ) (hypotenuse : ℚ) (distance_to_hypotenuse : ℚ) (x : ℚ)
  (h_triangle : a = 5 ∧ b = 12 ∧ hypotenuse = 13)
  (h_distance : distance_to_hypotenuse = 3)
  (h_x : x = 39 / 17)
  (h_square_area : x^2 = 1521 / 289)
  (total_area : ℚ) (planted_area : ℚ)
  (h_total_area : total_area = 30)
  (h_planted_area : planted_area = 7179 / 289) :
  planted_area / total_area = 2393 / 2890 :=
by
  sorry

end planted_fraction_l312_312568


namespace additional_carpet_needed_l312_312169

-- Define the given conditions as part of the hypothesis:
def carpetArea : ℕ := 18
def roomLength : ℕ := 4
def roomWidth : ℕ := 20

-- The theorem we want to prove:
theorem additional_carpet_needed : (roomLength * roomWidth - carpetArea) = 62 := by
  sorry

end additional_carpet_needed_l312_312169


namespace evaluate_power_l312_312705

theorem evaluate_power : (3^3)^2 = 729 := 
by 
  sorry

end evaluate_power_l312_312705


namespace value_of_X_l312_312598

def M := 2007 / 3
def N := M / 3
def X := M - N

theorem value_of_X : X = 446 := by
  sorry

end value_of_X_l312_312598


namespace complete_square_formula_D_l312_312067

-- Definitions of polynomial multiplications
def poly_A (a b : ℝ) : ℝ := (a - b) * (a + b)
def poly_B (a b : ℝ) : ℝ := -((a + b) * (b - a))
def poly_C (a b : ℝ) : ℝ := (a + b) * (b - a)
def poly_D (a b : ℝ) : ℝ := (a - b) * (b - a)

theorem complete_square_formula_D (a b : ℝ) : 
  poly_D a b = -(a - b)*(a - b) :=
by sorry

end complete_square_formula_D_l312_312067


namespace initial_number_of_girls_l312_312353

theorem initial_number_of_girls (n : ℕ) (A : ℝ) 
  (h1 : (n + 1) * (A + 3) - 70 = n * A + 94) :
  n = 8 :=
by {
  sorry
}

end initial_number_of_girls_l312_312353


namespace parabola_focus_l312_312199

noncomputable def parabola_focus_coordinates (a : ℝ) : ℝ × ℝ :=
  if a ≠ 0 then (0, 1 / (4 * a)) else (0, 0)

theorem parabola_focus {x y : ℝ} (a : ℝ) (h : a = 2) (h_eq : y = a * x^2) :
  parabola_focus_coordinates a = (0, 1 / 8) :=
by sorry

end parabola_focus_l312_312199


namespace min_value_inequality_l312_312763

theorem min_value_inequality (a b : ℝ) (h : a * b = 1) : 4 * a^2 + 9 * b^2 ≥ 12 :=
by sorry

end min_value_inequality_l312_312763


namespace parabola_translation_l312_312368

theorem parabola_translation :
  ∀ f g : ℝ → ℝ,
    (∀ x, f x = - (x - 1) ^ 2) →
    (∀ x, g x = f (x - 1) + 2) →
    ∀ x, g x = - (x - 2) ^ 2 + 2 :=
by
  -- Add the proof steps here if needed
  sorry

end parabola_translation_l312_312368


namespace cultural_festival_recommendation_schemes_l312_312976

theorem cultural_festival_recommendation_schemes :
  (∃ (females : Finset ℕ) (males : Finset ℕ),
    females.card = 3 ∧ males.card = 2 ∧
    ∃ (dance : Finset ℕ) (singing : Finset ℕ) (instruments : Finset ℕ),
      dance.card = 2 ∧ dance ⊆ females ∧
      singing.card = 2 ∧ singing ∩ females ≠ ∅ ∧
      instruments.card = 1 ∧ instruments ⊆ males ∧
      (females ∪ males).card = 5) → 
  ∃ (recommendation_schemes : ℕ), recommendation_schemes = 18 :=
by
  sorry

end cultural_festival_recommendation_schemes_l312_312976


namespace shopkeeper_profit_percent_l312_312874

theorem shopkeeper_profit_percent (cost_price profit : ℝ) (h1 : cost_price = 960) (h2 : profit = 40) : 
  (profit / cost_price) * 100 = 4.17 :=
by
  sorry

end shopkeeper_profit_percent_l312_312874


namespace f_2010_plus_f_2011_l312_312906

-- Definition of f being an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Conditions in Lean 4
variables (f : ℝ → ℝ)

axiom f_odd : odd_function f
axiom f_symmetry : ∀ x, f (1 + x) = f (1 - x)
axiom f_1 : f 1 = 2

-- The theorem to be proved
theorem f_2010_plus_f_2011 : f (2010) + f (2011) = -2 :=
by
  sorry

end f_2010_plus_f_2011_l312_312906


namespace problem_1_problem_2_l312_312342

noncomputable def f (x a : ℝ) : ℝ := abs x + 2 * abs (x - a)

theorem problem_1 (x : ℝ) : (f x 1 ≤ 4) ↔ (- 2 / 3 ≤ x ∧ x ≤ 2) := 
sorry

theorem problem_2 (a : ℝ) : (∀ x : ℝ, f x a ≥ 4) ↔ (4 ≤ a) := 
sorry

end problem_1_problem_2_l312_312342


namespace probability_heart_or_king_l312_312239

theorem probability_heart_or_king (cards hearts kings : ℕ) (prob_non_heart_king : ℚ) 
    (prob_two_non_heart_king : ℚ) : 
    cards = 52 → hearts = 13 → kings = 4 → 
    prob_non_heart_king = 36 / 52 → prob_two_non_heart_king = (36 / 52) ^ 2 → 
    1 - prob_two_non_heart_king = 88 / 169 :=
by
  intros h_cards h_hearts h_kings h_prob_non_heart_king h_prob_two_non_heart_king
  sorry

end probability_heart_or_king_l312_312239


namespace solve_system_of_equations_l312_312953

theorem solve_system_of_equations (a1 a2 a3 a4 : ℝ) (h_distinct : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)
  (x1 x2 x3 x4 : ℝ)
  (h1 : |a1 - a2| * x2 + |a1 - a3| * x3 + |a1 - a4| * x4 = 1)
  (h2 : |a2 - a1| * x1 + |a2 - a3| * x3 + |a2 - a4| * x4 = 1)
  (h3 : |a3 - a1| * x1 + |a3 - a2| * x2 + |a3 - a4| * x4 = 1)
  (h4 : |a4 - a1| * x1 + |a4 - a2| * x2 + |a4 - a3| * x3 = 1) :
  x1 = 1 / (a4 - a1) ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 1 / (a4 - a1) := 
sorry

end solve_system_of_equations_l312_312953


namespace square_field_diagonal_l312_312982

theorem square_field_diagonal (a : ℝ) (d : ℝ) (h : a^2 = 800) : d = 40 :=
by
  sorry

end square_field_diagonal_l312_312982


namespace sum_of_distances_parabola_circle_l312_312628

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem sum_of_distances_parabola_circle :
  let focus := (0, 1/8 : ℝ)
  let p1 := (-14, 392)
  let p2 := (-1, 2)
  let p3 := (6.5, 84.5)
  let p4 := (8.5, 144.5)
  distance focus p1 + distance focus p2 + distance focus p3 + distance focus p4 = 23065.6875 :=
by
  sorry

end sum_of_distances_parabola_circle_l312_312628


namespace dot_product_vec_a_vec_b_l312_312576

def vec_a : ℝ × ℝ := (-1, 2)
def vec_b : ℝ × ℝ := (1, 2)

theorem dot_product_vec_a_vec_b : vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2 = 3 := by
  sorry

end dot_product_vec_a_vec_b_l312_312576


namespace john_age_l312_312325

/-
Problem statement:
John is 24 years younger than his dad. The sum of their ages is 68 years.
We need to prove that John is 22 years old.
-/

theorem john_age:
  ∃ (j d : ℕ), (j = d - 24 ∧ j + d = 68) → j = 22 :=
by
  sorry

end john_age_l312_312325


namespace pyramid_cross_section_distance_l312_312212

theorem pyramid_cross_section_distance
  (area1 area2 : ℝ) (distance : ℝ)
  (h1 : area1 = 100 * Real.sqrt 3) 
  (h2 : area2 = 225 * Real.sqrt 3) 
  (h3 : distance = 5) : 
  ∃ h : ℝ, h = 15 :=
by
  sorry

end pyramid_cross_section_distance_l312_312212


namespace Dave_has_more_money_than_Derek_l312_312555

def Derek_initial := 40
def Derek_expense1 := 14
def Derek_expense2 := 11
def Derek_expense3 := 5
def Derek_remaining := Derek_initial - Derek_expense1 - Derek_expense2 - Derek_expense3

def Dave_initial := 50
def Dave_expense := 7
def Dave_remaining := Dave_initial - Dave_expense

def money_difference := Dave_remaining - Derek_remaining

theorem Dave_has_more_money_than_Derek : money_difference = 33 := by sorry

end Dave_has_more_money_than_Derek_l312_312555


namespace average_marks_first_class_l312_312417

theorem average_marks_first_class (A : ℝ) :
  let students_class1 := 55
  let students_class2 := 48
  let avg_class2 := 58
  let avg_all := 59.067961165048544
  let total_students := 103
  let total_marks := avg_all * total_students
  total_marks = (A * students_class1) + (avg_class2 * students_class2) 
  → A = 60 :=
by
  sorry

end average_marks_first_class_l312_312417


namespace complementary_angles_difference_l312_312204

theorem complementary_angles_difference :
  ∃ (θ₁ θ₂ : ℝ), θ₁ + θ₂ = 90 ∧ 5 * θ₁ = 3 * θ₂ ∧ abs (θ₁ - θ₂) = 22.5 :=
by
  sorry

end complementary_angles_difference_l312_312204


namespace compute_c_minus_d_cubed_l312_312469

-- define c as the number of positive multiples of 12 less than 60
def c : ℕ := Finset.card (Finset.filter (λ n => 12 ∣ n) (Finset.range 60))

-- define d as the number of positive integers less than 60 and a multiple of both 3 and 4
def d : ℕ := Finset.card (Finset.filter (λ n => 12 ∣ n) (Finset.range 60))

theorem compute_c_minus_d_cubed : (c - d)^3 = 0 := by
  -- since c and d are computed the same way, (c - d) = 0
  -- hence, (c - d)^3 = 0^3 = 0
  sorry

end compute_c_minus_d_cubed_l312_312469


namespace evaluate_three_cubed_squared_l312_312711

theorem evaluate_three_cubed_squared : (3^3)^2 = 729 :=
by
  -- Given the property of exponents
  have h : (forall (a m n : ℕ), (a^m)^n = a^(m * n)) := sorry,
  -- Now prove the statement using the given property
  calc
    (3^3)^2 = 3^(3 * 2) : by rw [h 3 3 2]
          ... = 3^6       : by norm_num
          ... = 729       : by norm_num

end evaluate_three_cubed_squared_l312_312711


namespace number_of_bowls_l312_312839

theorem number_of_bowls (n : ℕ) :
  (∀ (b : ℕ), b > 0) →
  (∀ (a : ℕ), ∃ (k : ℕ), true) →
  (8 * 12 = 96) →
  (6 * n = 96) →
  n = 16 :=
by
  intros h1 h2 h3 h4
  sorry

end number_of_bowls_l312_312839


namespace imaginary_part_z_l312_312629

open Complex

theorem imaginary_part_z : (im ((i - 1) / (i + 1))) = 1 :=
by
  -- The proof goes here, but it can be marked with sorry for now
  sorry

end imaginary_part_z_l312_312629


namespace unique_digit_sum_is_21_l312_312789

theorem unique_digit_sum_is_21
  (Y E M T : ℕ)
  (YE ME : ℕ)
  (HT0 : YE = 10 * Y + E)
  (HT1 : ME = 10 * M + E)
  (H1 : YE * ME = 999)
  (H2 : Y ≠ E)
  (H3 : Y ≠ M)
  (H4 : Y ≠ T)
  (H5 : E ≠ M)
  (H6 : E ≠ T)
  (H7 : M ≠ T)
  (H8 : Y < 10)
  (H9 : E < 10)
  (H10 : M < 10)
  (H11 : T < 10) :
  Y + E + M + T = 21 :=
sorry

end unique_digit_sum_is_21_l312_312789


namespace geometric_sequence_first_term_l312_312357

open Real Nat

theorem geometric_sequence_first_term (a r : ℝ)
  (h1 : a * r^4 = (7! : ℝ))
  (h2 : a * r^7 = (8! : ℝ)) : a = 315 := by
  sorry

end geometric_sequence_first_term_l312_312357


namespace metal_beams_per_panel_l312_312532

theorem metal_beams_per_panel (panels sheets_per_panel rods_per_sheet rods_needed beams_per_panel rods_per_beam : ℕ)
    (h1 : panels = 10)
    (h2 : sheets_per_panel = 3)
    (h3 : rods_per_sheet = 10)
    (h4 : rods_needed = 380)
    (h5 : rods_per_beam = 4)
    (h6 : beams_per_panel = 2) :
    (panels * sheets_per_panel * rods_per_sheet + panels * beams_per_panel * rods_per_beam = rods_needed) :=
by
  sorry

end metal_beams_per_panel_l312_312532


namespace triangle_angle_contradiction_l312_312381

theorem triangle_angle_contradiction (α β γ : ℝ) (h1 : α + β + γ = 180) (h2 : α > 60) (h3 : β > 60) (h4 : γ > 60) : false :=
sorry

end triangle_angle_contradiction_l312_312381


namespace shortest_segment_length_l312_312998

theorem shortest_segment_length :
  let total_length := 1
  let red_dot := 0.618
  let yellow_dot := total_length - red_dot  -- yellow_dot is at the same point after fold
  let first_cut := red_dot  -- Cut the strip at the red dot
  let remaining_strip := red_dot
  let distance_between_red_and_yellow := total_length - 2 * yellow_dot
  let second_cut := distance_between_red_and_yellow
  let shortest_segment := remaining_strip - 2 * distance_between_red_and_yellow
  shortest_segment = 0.146 :=
by
  sorry

end shortest_segment_length_l312_312998


namespace nutmeg_amount_l312_312336

def amount_of_cinnamon : ℝ := 0.6666666666666666
def difference_cinnamon_nutmeg : ℝ := 0.16666666666666666

theorem nutmeg_amount (x : ℝ) 
  (h1 : amount_of_cinnamon = x + difference_cinnamon_nutmeg) : 
  x = 0.5 :=
by 
  sorry

end nutmeg_amount_l312_312336


namespace number_of_bowls_l312_312847

noncomputable theory
open Classical

theorem number_of_bowls (n : ℕ) 
  (h1 : 8 * 12 = 6 * n) : n = 16 := 
by
  sorry

end number_of_bowls_l312_312847


namespace sequence_general_term_l312_312620

theorem sequence_general_term :
  ∀ n : ℕ, n > 0 → (∀ a: ℕ → ℝ,  a 1 = 4 ∧ (∀ n: ℕ, n > 0 → a (n + 1) = (3 * a n + 2) / (a n + 4))
  → a n = (2 ^ (n - 1) + 5 ^ (n - 1)) / (5 ^ (n - 1) - 2 ^ (n - 1))) :=
by
  sorry

end sequence_general_term_l312_312620


namespace f_increasing_f_at_2_solve_inequality_l312_312773

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_add (a b : ℝ) : f (a + b) = f a + f b - 1
axiom f_pos (x : ℝ) (h : x > 0) : f x > 1
axiom f_at_4 : f 4 = 5

theorem f_increasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 :=
sorry

theorem f_at_2 : f 2 = 3 :=
sorry

theorem solve_inequality (m : ℝ) : f (3 * m^2 - m - 2) < 3 ↔ -1 < m ∧ m < 4 / 3 :=
sorry

end f_increasing_f_at_2_solve_inequality_l312_312773


namespace unique_very_set_on_line_l312_312216

def very_set (S : Finset (ℝ × ℝ)) : Prop :=
  ∀ X ∈ S, ∃ (r : ℝ), 
  ∀ Y ∈ S, Y ≠ X → ∃ Z ∈ S, Z ≠ X ∧ r * r = dist X Y * dist X Z

theorem unique_very_set_on_line (n : ℕ) (A B : ℝ × ℝ) (S1 S2 : Finset (ℝ × ℝ))
  (h : 2 ≤ n) (hA1 : A ∈ S1) (hB1 : B ∈ S1) (hA2 : A ∈ S2) (hB2 : B ∈ S2)
  (hS1 : S1.card = n) (hS2 : S2.card = n) (hV1 : very_set S1) (hV2 : very_set S2) :
  S1 = S2 := 
sorry

end unique_very_set_on_line_l312_312216


namespace gcd_lcm_8951_4267_l312_312217

theorem gcd_lcm_8951_4267 :
  gcd 8951 4267 = 1 ∧ lcm 8951 4267 = 38212917 :=
by
  sorry

end gcd_lcm_8951_4267_l312_312217


namespace volume_of_tetrahedron_l312_312566

theorem volume_of_tetrahedron (A B C D : Type) 
    (area_ABC : ℝ) (area_BCD : ℝ) (BC : ℝ) (angle : ℝ)
    (h1 : area_ABC = 150)
    (h2 : area_BCD = 90)
    (h3 : BC = 10)
    (h4 : angle = π / 4) :
    ∃ V : ℝ, V = 450 * real.sqrt 2 :=
begin
  sorry
end

end volume_of_tetrahedron_l312_312566


namespace spinning_class_frequency_l312_312623

/--
We define the conditions given in the problem:
- duration of each class in hours,
- calorie burn rate per minute,
- total calories burned per week.
We then state that the number of classes James attends per week is equal to 3.
-/
def class_duration_hours : ℝ := 1.5
def calories_per_minute : ℝ := 7
def total_calories_per_week : ℝ := 1890

theorem spinning_class_frequency :
  total_calories_per_week / (class_duration_hours * 60 * calories_per_minute) = 3 :=
by
  sorry

end spinning_class_frequency_l312_312623


namespace max_f_value_l312_312115

noncomputable def f (x : ℝ) : ℝ := min (3 * x + 1) (min (- (4 / 3) * x + 3) ((1 / 3) * x + 9))

theorem max_f_value : ∃ x : ℝ, f x = 31 / 13 :=
by 
  sorry

end max_f_value_l312_312115


namespace evaluate_exp_power_l312_312717

theorem evaluate_exp_power : (3^3)^2 = 729 := 
by {
  sorry
}

end evaluate_exp_power_l312_312717


namespace question_d_not_true_l312_312899

variable {a b c d : ℚ}

theorem question_d_not_true (h : a * b = c * d) : (a + 1) / (c + 1) ≠ (d + 1) / (b + 1) := 
sorry

end question_d_not_true_l312_312899


namespace distinct_ordered_pairs_l312_312587

theorem distinct_ordered_pairs (a b : ℕ) (h : a + b = 40) (ha : a > 0) (hb : b > 0) :
  ∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 39 ∧ ∀ p ∈ pairs, p.1 + p.2 = 40 := 
sorry

end distinct_ordered_pairs_l312_312587


namespace unique_fraction_satisfying_condition_l312_312112

theorem unique_fraction_satisfying_condition : ∃! (x y : ℕ), Nat.gcd x y = 1 ∧ y ≠ 0 ∧ (x + 1) * 5 * y = (y + 1) * 6 * x :=
by
  sorry

end unique_fraction_satisfying_condition_l312_312112


namespace probability_of_exact_r_successes_is_correct_l312_312142

open Probability

noncomputable def probability_of_exact_r_successes (n r : ℕ) (p : ℝ) (h1 : 0 <= r ∧ r <= n) (h2 : 0 <= p ∧ p <= 1) : ℝ := 
  (nat.choose (n-1) (r-1)) * (p^r) * ((1 - p)^(n - r))

theorem probability_of_exact_r_successes_is_correct (n r : ℕ) (p : ℝ) (h1 : 0 <= r ∧ r <= n) (h2 : 0 <= p ∧ p <= 1) :
  probability_of_exact_r_successes n r p h1 h2 = (nat.choose (n-1) (r-1)) * (p^r) * ((1 - p)^(n - r)) :=
sorry

end probability_of_exact_r_successes_is_correct_l312_312142


namespace average_mark_excluded_students_l312_312655

variables (N A E A_R A_E : ℕ)

theorem average_mark_excluded_students:
    N = 56 → A = 80 → E = 8 → A_R = 90 →
    N * A = E * A_E + (N - E) * A_R →
    A_E = 20 :=
by
  intros hN hA hE hAR hEquation
  rw [hN, hA, hE, hAR] at hEquation
  have h : 4480 = 8 * A_E + 4320 := hEquation
  sorry

end average_mark_excluded_students_l312_312655


namespace circle_area_difference_l312_312412

/-- 
Prove that the area of the circle with radius r1 = 30 inches is 675π square inches greater than 
the area of the circle with radius r2 = 15 inches.
-/
theorem circle_area_difference (r1 r2 : ℝ) (h1 : r1 = 30) (h2 : r2 = 15) :
  π * r1^2 - π * r2^2 = 675 * π := 
by {
  -- Placeholders to indicate where the proof would go
  sorry 
}

end circle_area_difference_l312_312412


namespace number_of_apples_remaining_l312_312592

def blue_apples : ℕ := 5
def yellow_apples : ℕ := 2 * blue_apples
def total_apples_before_giving_away : ℕ := blue_apples + yellow_apples
def apples_given_to_son : ℕ := total_apples_before_giving_away / 5
def apples_remaining : ℕ := total_apples_before_giving_away - apples_given_to_son

theorem number_of_apples_remaining : apples_remaining = 12 :=
by
  sorry

end number_of_apples_remaining_l312_312592


namespace fill_time_difference_correct_l312_312087

-- Define the time to fill one barrel in normal conditions
def normal_fill_time_per_barrel : ℕ := 3

-- Define the time to fill one barrel with a leak
def leak_fill_time_per_barrel : ℕ := 5

-- Define the number of barrels to fill
def barrels_to_fill : ℕ := 12

-- Define the time to fill 12 barrels in normal conditions
def normal_fill_time : ℕ := normal_fill_time_per_barrel * barrels_to_fill

-- Define the time to fill 12 barrels with a leak
def leak_fill_time : ℕ := leak_fill_time_per_barrel * barrels_to_fill

-- Define the time difference
def time_difference : ℕ := leak_fill_time - normal_fill_time

theorem fill_time_difference_correct : time_difference = 24 := by
  sorry

end fill_time_difference_correct_l312_312087


namespace option_B_is_perfect_square_option_C_is_perfect_square_option_E_is_perfect_square_l312_312383

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- Definitions of the given options as natural numbers
def A := 3^3 * 4^4 * 5^5
def B := 3^4 * 4^5 * 5^6
def C := 3^6 * 4^4 * 5^6
def D := 3^5 * 4^6 * 5^5
def E := 3^6 * 4^6 * 5^4

-- Lean statements for each option being a perfect square
theorem option_B_is_perfect_square : is_perfect_square B := sorry
theorem option_C_is_perfect_square : is_perfect_square C := sorry
theorem option_E_is_perfect_square : is_perfect_square E := sorry

end option_B_is_perfect_square_option_C_is_perfect_square_option_E_is_perfect_square_l312_312383


namespace james_out_of_pocket_l312_312464

-- Definitions based on conditions
def old_car_value : ℝ := 20000
def old_car_sold_for : ℝ := 0.80 * old_car_value
def new_car_sticker_price : ℝ := 30000
def new_car_bought_for : ℝ := 0.90 * new_car_sticker_price

-- Question and proof statement
def amount_out_of_pocket : ℝ := new_car_bought_for - old_car_sold_for

theorem james_out_of_pocket : amount_out_of_pocket = 11000 := by
  sorry

end james_out_of_pocket_l312_312464


namespace n_power_2020_plus_4_composite_l312_312639

theorem n_power_2020_plus_4_composite {n : ℕ} (h : n > 1) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n^2020 + 4 = a * b := 
by
  sorry

end n_power_2020_plus_4_composite_l312_312639


namespace words_on_each_page_l312_312388

theorem words_on_each_page (p : ℕ) (h1 : p ≤ 120) (h2 : 150 * p % 221 = 210) : p = 48 :=
sorry

end words_on_each_page_l312_312388


namespace fraction_comparison_l312_312109

theorem fraction_comparison (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x^2 - y^2) / (x - y) > (x^2 + y^2) / (x + y) :=
by
  sorry

end fraction_comparison_l312_312109


namespace prove_f_neg_a_l312_312426

noncomputable def f (x : ℝ) : ℝ := x + 1/x - 1

theorem prove_f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = -4 :=
by
  sorry

end prove_f_neg_a_l312_312426


namespace max_value_of_abs_asinx_plus_b_l312_312133

theorem max_value_of_abs_asinx_plus_b 
  (a b c : ℝ) 
  (h : ∀ x : ℝ, |a * (Real.cos x)^2 + b * Real.sin x + c| ≤ 1) : 
  ∃ M, M = 2 ∧ ∀ x : ℝ, |a * Real.sin x + b| ≤ M :=
by
  use 2
  sorry

end max_value_of_abs_asinx_plus_b_l312_312133


namespace sum_of_coefficients_of_parabolas_kite_formed_l312_312270

theorem sum_of_coefficients_of_parabolas_kite_formed (a b : ℝ) 
  (h1 : ∃ (x : ℝ), y = ax^2 - 4)
  (h2 : ∃ (y : ℝ), y = 6 - bx^2)
  (h3 : (a > 0) ∧ (b > 0) ∧ (ax^2 - 4 = 0) ∧ (6 - bx^2 = 0))
  (h4 : kite_area = 18) :
  a + b = 125/36 := 
by sorry

end sum_of_coefficients_of_parabolas_kite_formed_l312_312270


namespace correct_calculation_l312_312063

theorem correct_calculation :
  (∀ (x y : ℝ), -3 * x - 3 * x ≠ 0) ∧
  (∀ (x : ℝ), x - 4 * x ≠ -3) ∧
  (∀ (x : ℝ), 2 * x + 3 * x^2 ≠ 5 * x^3) ∧
  (∀ (x y : ℝ), -4 * x * y + 3 * x * y = -x * y) :=
by
  sorry

end correct_calculation_l312_312063


namespace smallest_pythagorean_sum_square_l312_312597

theorem smallest_pythagorean_sum_square (p q r : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (h : p^2 + q^2 = r^2) :
  ∃ (k : ℤ), k = 4 ∧ (p + q + r)^2 ≥ k :=
by
  sorry

end smallest_pythagorean_sum_square_l312_312597


namespace polynomial_roots_l312_312418

theorem polynomial_roots : (∃ x : ℝ, (4 * x ^ 4 + 11 * x ^ 3 - 37 * x ^ 2 + 18 * x = 0) ↔ (x = 0 ∨ x = 1 / 2 ∨ x = 3 / 2 ∨ x = -6)) :=
by 
  sorry

end polynomial_roots_l312_312418


namespace xy_squared_sum_l312_312920

theorem xy_squared_sum {x y : ℝ} (h1 : (x + y)^2 = 49) (h2 : x * y = 10) : x^2 + y^2 = 29 :=
by
  sorry

end xy_squared_sum_l312_312920


namespace cost_per_mile_l312_312231

theorem cost_per_mile (x : ℝ) (daily_fee : ℝ) (daily_budget : ℝ) (max_miles : ℝ)
  (h1 : daily_fee = 50)
  (h2 : daily_budget = 88)
  (h3 : max_miles = 190)
  (h4 : daily_budget = daily_fee + x * max_miles) :
  x = 0.20 :=
by
  sorry

end cost_per_mile_l312_312231


namespace number_of_apples_remaining_l312_312593

def blue_apples : ℕ := 5
def yellow_apples : ℕ := 2 * blue_apples
def total_apples_before_giving_away : ℕ := blue_apples + yellow_apples
def apples_given_to_son : ℕ := total_apples_before_giving_away / 5
def apples_remaining : ℕ := total_apples_before_giving_away - apples_given_to_son

theorem number_of_apples_remaining : apples_remaining = 12 :=
by
  sorry

end number_of_apples_remaining_l312_312593


namespace min_ab_value_l312_312761

theorem min_ab_value (a b : Real) (h_a : 1 < a) (h_b : 1 < b)
  (h_geom_seq : ∀ (x₁ x₂ x₃ : Real), x₁ = (1/4) * Real.log a → x₂ = 1/4 → x₃ = Real.log b →  x₂^2 = x₁ * x₃) : 
  a * b ≥ Real.exp 1 := by
  sorry

end min_ab_value_l312_312761


namespace art_of_passing_through_walls_l312_312603

theorem art_of_passing_through_walls (n : ℕ) :
  (2 * Real.sqrt (2 / 3) = Real.sqrt (2 * (2 / 3))) ∧
  (3 * Real.sqrt (3 / 8) = Real.sqrt (3 * (3 / 8))) ∧
  (4 * Real.sqrt (4 / 15) = Real.sqrt (4 * (4 / 15))) ∧
  (5 * Real.sqrt (5 / 24) = Real.sqrt (5 * (5 / 24))) →
  8 * Real.sqrt (8 / n) = Real.sqrt (8 * (8 / n)) →
  n = 63 :=
by
  sorry

end art_of_passing_through_walls_l312_312603


namespace inequality_condition_l312_312490

variables {a b c : ℝ} {x : ℝ}

theorem inequality_condition (h : a * a + b * b < c * c) : ∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0 :=
sorry

end inequality_condition_l312_312490


namespace steven_set_aside_9_grapes_l312_312954

-- Define the conditions based on the problem statement
def total_seeds_needed : ℕ := 60
def average_seeds_per_apple : ℕ := 6
def average_seeds_per_pear : ℕ := 2
def average_seeds_per_grape : ℕ := 3
def apples_set_aside : ℕ := 4
def pears_set_aside : ℕ := 3
def additional_seeds_needed : ℕ := 3

-- Calculate the number of seeds from apples and pears
def seeds_from_apples : ℕ := apples_set_aside * average_seeds_per_apple
def seeds_from_pears : ℕ := pears_set_aside * average_seeds_per_pear

-- Calculate the number of seeds that Steven already has from apples and pears
def seeds_from_apples_and_pears : ℕ := seeds_from_apples + seeds_from_pears

-- Calculate the remaining seeds needed from grapes
def seeds_needed_from_grapes : ℕ := total_seeds_needed - seeds_from_apples_and_pears - additional_seeds_needed

-- Calculate the number of grapes set aside
def grapes_set_aside : ℕ := seeds_needed_from_grapes / average_seeds_per_grape

theorem steven_set_aside_9_grapes : grapes_set_aside = 9 :=
by 
  sorry

end steven_set_aside_9_grapes_l312_312954


namespace intersection_points_relation_l312_312764

noncomputable def num_intersections (k : ℕ) : ℕ :=
  k * (k - 1) / 2

theorem intersection_points_relation (k : ℕ) :
  num_intersections (k + 1) = num_intersections k + k := by
sorry

end intersection_points_relation_l312_312764


namespace evaluate_three_star_twostar_one_l312_312602

def operator_star (a b : ℕ) : ℕ :=
  a^b - b^a

theorem evaluate_three_star_twostar_one : operator_star 3 (operator_star 2 1) = 2 := 
  by
    sorry

end evaluate_three_star_twostar_one_l312_312602


namespace solve_for_a_l312_312578

theorem solve_for_a (a : ℤ) (h1 : 0 ≤ a) (h2 : a ≤ 13) (h3 : 13 ∣ 51^2016 - a) : a = 1 :=
by {
  sorry
}

end solve_for_a_l312_312578


namespace reward_function_conditions_l312_312096

theorem reward_function_conditions :
  (∀ x : ℝ, (10 ≤ x ∧ x ≤ 1000) → (y = x / 150 + 2 → y ≤ 90 ∧ y ≤ x / 5) → False) ∧
  (∃ a : ℕ, (∀ x : ℝ, (10 ≤ x ∧ x ≤ 1000) → (y = (10 * x - 3 * a) / (x + 2) → y ≤ 9 ∧ y ≤ x / 5)) ∧ (a = 328)) :=
by
  sorry

end reward_function_conditions_l312_312096


namespace product_of_roots_of_quartic_polynomial_l312_312470

theorem product_of_roots_of_quartic_polynomial :
  (∀ x : ℝ, (3 * x^4 - 8 * x^3 + x^2 - 10 * x - 24 = 0) → x = p ∨ x = q ∨ x = r ∨ x = s) →
  (p * q * r * s = -8) :=
by
  intros
  -- proof goes here
  sorry

end product_of_roots_of_quartic_polynomial_l312_312470


namespace range_of_b_l312_312958

theorem range_of_b (x b : ℝ) (hb : b > 0) : 
  (∃ x : ℝ, |x - 2| + |x + 1| < b) ↔ b > 3 :=
by
  sorry

end range_of_b_l312_312958


namespace speed_on_way_home_l312_312185

theorem speed_on_way_home (d : ℝ) (v_up : ℝ) (v_avg : ℝ) (v_home : ℝ) 
  (h1 : v_up = 110) 
  (h2 : v_avg = 91)
  (h3 : 91 = (2 * d) / (d / 110 + d / v_home)) : 
  v_home = 10010 / 129 := 
sorry

end speed_on_way_home_l312_312185


namespace part1_part2_l312_312343

-- Part 1
theorem part1 (x y : ℝ) 
  (h1 : x + 2 * y = 9) 
  (h2 : 2 * x + y = 6) :
  (x - y = -3) ∧ (x + y = 5) :=
sorry

-- Part 2
theorem part2 (x y : ℝ) 
  (h1 : x + 2 = 5) 
  (h2 : y - 1 = 4) :
  x = 3 ∧ y = 5 :=
sorry

end part1_part2_l312_312343


namespace exponentiation_calculation_l312_312691

theorem exponentiation_calculation : 3000 * (3000 ^ 3000) ^ 2 = 3000 ^ 6001 := by
  sorry

end exponentiation_calculation_l312_312691


namespace max_area_ABC_l312_312176

noncomputable def q (p : ℝ) : ℝ := p^2 - 7*p + 10

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - A.2 * B.1 - B.2 * C.1 - C.2 * A.1)

theorem max_area_ABC : ∃ p : ℝ, 2 ≤ p ∧ p ≤ 5 ∧ 
  triangle_area (2, 0) (5, 4) (p, q p) = 0.536625 := sorry

end max_area_ABC_l312_312176


namespace external_angle_bisector_lengths_l312_312256

noncomputable def f_a (a b c : ℝ) : ℝ := 4 * Real.sqrt 3
noncomputable def f_b (b : ℝ) : ℝ := 6 / Real.sqrt 7
noncomputable def f_c (a b c : ℝ) : ℝ := 4 * Real.sqrt 3

theorem external_angle_bisector_lengths (a b c : ℝ) 
  (ha : a = 5 - Real.sqrt 7)
  (hb : b = 6)
  (hc : c = 5 + Real.sqrt 7) :
  f_a a b c = 4 * Real.sqrt 3 ∧
  f_b b = 6 / Real.sqrt 7 ∧
  f_c a b c = 4 * Real.sqrt 3 := by
  sorry

end external_angle_bisector_lengths_l312_312256


namespace number_of_bowls_l312_312843

-- Let n be the number of bowls on the table.
variable (n : ℕ)

-- Condition 1: There are n bowls, and each contain some grapes.
-- Condition 2: Adding 8 grapes to each of 12 specific bowls increases the average number of grapes in all bowls by 6.
-- Let's formalize the condition given in the problem
theorem number_of_bowls (h1 : 12 * 8 = 96) (h2 : 6 * n = 96) : n = 16 :=
by
  -- omitting the proof here
  sorry

end number_of_bowls_l312_312843


namespace number_of_bowls_l312_312850

noncomputable theory
open Classical

theorem number_of_bowls (n : ℕ) 
  (h1 : 8 * 12 = 6 * n) : n = 16 := 
by
  sorry

end number_of_bowls_l312_312850


namespace smallest_solution_equation_l312_312419

noncomputable def equation (x : ℝ) : ℝ :=
  (3*x / (x-3)) + ((3*x^2 - 45) / x) + 3

theorem smallest_solution_equation : 
  ∃ x : ℝ, equation x = 14 ∧ x = (1 - Real.sqrt 649) / 12 :=
sorry

end smallest_solution_equation_l312_312419


namespace broth_for_third_l312_312545

theorem broth_for_third (b : ℚ) (h : b = 6 + 3/4) : b / 3 = 2 + 1/4 := by
  sorry

end broth_for_third_l312_312545


namespace probability_bernardo_larger_l312_312689

-- Define the sets from which Bernardo and Silvia are picking numbers
def set_B : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def set_S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the function to calculate the probability as described in the problem statement
def bernardo_larger_probability : ℚ := sorry -- The step by step calculations will be inserted here

-- Main theorem stating what needs to be proved
theorem probability_bernardo_larger : bernardo_larger_probability = 61 / 80 := 
sorry

end probability_bernardo_larger_l312_312689


namespace find_x_y_sum_l312_312472

variable {x y : ℝ}

theorem find_x_y_sum (h₁ : (x-1)^3 + 1997 * (x-1) = -1) (h₂ : (y-1)^3 + 1997 * (y-1) = 1) : 
  x + y = 2 := 
by
  sorry

end find_x_y_sum_l312_312472


namespace regular_polygon_num_sides_l312_312002

def diag_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem regular_polygon_num_sides (n : ℕ) (h : diag_formula n = 20) : n = 8 :=
by
  sorry

end regular_polygon_num_sides_l312_312002


namespace distinct_pairs_count_l312_312586

theorem distinct_pairs_count : 
  (∃ (s : Finset (ℕ × ℕ)), (∀ p ∈ s, ∃ (a b : ℕ), 1 ≤ a ∧ 1 ≤ b ∧ a + b = 40 ∧ p = (a, b)) ∧ s.card = 39) := sorry

end distinct_pairs_count_l312_312586


namespace number_of_bowls_l312_312830

theorem number_of_bowls (n : ℕ) (h1 : 12 * 8 = 96) (h2 : 6 * n = 96) : n = 16 :=
by
  -- equations from the conditions
  have h3 : 96 = 96 := by sorry
  exact sorry

end number_of_bowls_l312_312830


namespace sally_balance_fraction_l312_312642

variable (G : ℝ) (x : ℝ)
-- spending limit on gold card is G
-- spending limit on platinum card is 2G
-- Balance on platinum card is G/2
-- After transfer, 0.5833333333333334 portion of platinum card remains unspent

theorem sally_balance_fraction
  (h1 : (5/12) * 2 * G = G / 2 + x * G) : x = 1 / 3 :=
by
  sorry

end sally_balance_fraction_l312_312642


namespace reciprocal_relation_l312_312778

theorem reciprocal_relation (x : ℝ) (h : 1 / (x + 3) = 2) : 1 / (x + 5) = 2 / 5 := 
by
  sorry

end reciprocal_relation_l312_312778


namespace general_term_l312_312907

noncomputable def S : ℕ → ℤ
| n => 3 * n ^ 2 - 2 * n + 1

def a : ℕ → ℤ
| 0 => 2  -- Since sequences often start at n=1 and MATLAB indexing starts at 0.
| 1 => 2
| (n+2) => 6 * (n + 2) - 5

theorem general_term (n : ℕ) : 
  a n = if n = 1 then 2 else 6 * n - 5 :=
by sorry

end general_term_l312_312907


namespace sufficient_but_not_necessary_l312_312993

theorem sufficient_but_not_necessary (a : ℝ) (h1 : a > 0) (h2 : |a| > 0 → a > 0 ∨ a < 0) : 
  (a > 0 → |a| > 0) ∧ (¬(|a| > 0 → a > 0)) := 
by
  sorry

end sufficient_but_not_necessary_l312_312993


namespace no_value_of_n_l312_312471

noncomputable def t1 (n : ℕ) : ℚ :=
3 * n * (n + 2)

noncomputable def t2 (n : ℕ) : ℚ :=
(3 * n^2 + 19 * n) / 2

theorem no_value_of_n (n : ℕ) (h : n > 0) : t1 n ≠ t2 n :=
by {
  sorry
}

end no_value_of_n_l312_312471


namespace min_distance_circle_tangent_l312_312775

theorem min_distance_circle_tangent
  (P : ℝ × ℝ)
  (hP: 3 * P.1 + 4 * P.2 = 11) :
  ∃ d : ℝ, d = 11 / 5 := 
sorry

end min_distance_circle_tangent_l312_312775


namespace simplify_expr1_simplify_expr2_l312_312346

-- (1) Simplify the expression: 3a(a+1) - (3+a)(3-a) - (2a-1)^2 == 7a - 10
theorem simplify_expr1 (a : ℝ) : 
  3 * a * (a + 1) - (3 + a) * (3 - a) - (2 * a - 1) ^ 2 = 7 * a - 10 :=
sorry

-- (2) Simplify the expression: ((x^2 - 2x + 4) / (x - 1) + 2 - x) / (x^2 + 4x + 4) / (1 - x) == -2 / (x + 2)^2
theorem simplify_expr2 (x : ℝ) (h : x ≠ 1) (h1 : x ≠ 0) : 
  (((x^2 - 2 * x + 4) / (x - 1) + 2 - x) / ((x^2 + 4 * x + 4) / (1 - x))) = -2 / (x + 2)^2 :=
sorry

end simplify_expr1_simplify_expr2_l312_312346


namespace expected_value_T_l312_312195

def boys_girls_expected_value (M N : ℕ) : ℚ :=
  2 * ((M / (M + N : ℚ)) * (N / (M + N - 1 : ℚ)))

theorem expected_value_T (M N : ℕ) (hM : M = 10) (hN : N = 10) :
  boys_girls_expected_value M N = 20 / 19 :=
by 
  rw [hM, hN]
  sorry

end expected_value_T_l312_312195


namespace arithmetic_sequence_correct_l312_312965

-- Define the conditions
def last_term_eq_num_of_terms (a l n : Int) : Prop := l = n
def common_difference (d : Int) : Prop := d = 5
def sum_of_sequence (n a S : Int) : Prop :=
  S = n * (2 * a + (n - 1) * 5) / 2

-- The target arithmetic sequence
def seq : List Int := [-7, -2, 3]
def first_term : Int := -7
def num_terms : Int := 3
def sum_of_seq : Int := -6

-- Proof statement
theorem arithmetic_sequence_correct :
  last_term_eq_num_of_terms first_term seq.length num_terms ∧
  common_difference 5 ∧
  sum_of_sequence seq.length first_term sum_of_seq →
  seq = [-7, -2, 3] :=
sorry

end arithmetic_sequence_correct_l312_312965


namespace probability_of_sum_multiple_of_4_l312_312370

noncomputable def prob_sum_multiple_of_4 : ℚ := 
  let n_success := (3 : ℚ) + 5 + 1
  let n_total := (6 : ℚ) * 6
  n_success / n_total

theorem probability_of_sum_multiple_of_4 (success_count total_count : ℕ) 
  (H_success: success_count = 9) (H_total: total_count = 36) 
  (p : ℚ := (success_count : ℚ) / (total_count : ℚ)) :
  p = 1/4 :=
by 
  simp [H_success, H_total, p, prob_sum_multiple_of_4]
  sorry

end probability_of_sum_multiple_of_4_l312_312370


namespace find_m_value_l312_312148

theorem find_m_value (m : ℝ) (A : Set ℝ) (h₁ : A = {0, m, m^2 - 3 * m + 2}) (h₂ : 2 ∈ A) : m = 3 :=
by
  sorry

end find_m_value_l312_312148


namespace train_speed_identification_l312_312546

-- Define the conditions
def train_length : ℕ := 300
def crossing_time : ℕ := 30

-- Define the speed calculation
def calculate_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

-- The target theorem stating the speed of the train
theorem train_speed_identification : calculate_speed train_length crossing_time = 10 := 
by 
  sorry

end train_speed_identification_l312_312546


namespace simplify_sqrt_expression_l312_312647

theorem simplify_sqrt_expression (a b : ℕ) (h_a : a = 5) (h_b : b = 3) :
  (sqrt (a * b) * sqrt ((b ^ 3) * (a ^ 3)) = 225) :=
by
  rw [h_a, h_b]
  sorry

end simplify_sqrt_expression_l312_312647


namespace solve_for_b_l312_312776

theorem solve_for_b (a b c m : ℚ) (h : m = c * a * b / (a - b)) : b = (m * a) / (m + c * a) :=
by
  sorry

end solve_for_b_l312_312776


namespace work_completion_days_l312_312098

theorem work_completion_days 
  (x : ℕ) 
  (h1 : ∀ t : ℕ, t = x → A_work_rate = 2 * (1 / t))
  (h2 : A_and_B_work_together : ∀ d : ℕ, d = 4 → A_B_combined_rate = 1 / d) :
  x = 12 := 
sorry

end work_completion_days_l312_312098


namespace number_of_bowls_l312_312821

theorem number_of_bowls (n : ℕ) (h : 8 * 12 = 96) (avg_increase : 6 * n = 96) : n = 16 :=
by {
  sorry
}

end number_of_bowls_l312_312821


namespace total_snakes_l312_312863

def People (n : ℕ) : Prop := n = 59
def OnlyDogs (n : ℕ) : Prop := n = 15
def OnlyCats (n : ℕ) : Prop := n = 10
def OnlyCatsAndDogs (n : ℕ) : Prop := n = 5
def CatsDogsSnakes (n : ℕ) : Prop := n = 3

theorem total_snakes (n_people n_dogs n_cats n_catsdogs n_catdogsnsnakes : ℕ)
  (h_people : People n_people) 
  (h_onlyDogs : OnlyDogs n_dogs)
  (h_onlyCats : OnlyCats n_cats)
  (h_onlyCatsAndDogs : OnlyCatsAndDogs n_catsdogs)
  (h_catsDogsSnakes : CatsDogsSnakes n_catdogsnsnakes) :
  n_catdogsnsnakes >= 3 :=
by
  -- Proof goes here
  sorry

end total_snakes_l312_312863


namespace goods_train_speed_l312_312537

theorem goods_train_speed :
  ∀ (length_train length_platform time : ℝ),
    length_train = 250.0416 →
    length_platform = 270 →
    time = 26 →
    (length_train + length_platform) / time = 20 :=
by
  intros length_train length_platform time H_train H_platform H_time
  rw [H_train, H_platform, H_time]
  norm_num
  sorry

end goods_train_speed_l312_312537


namespace repeating_decimal_to_fraction_l312_312891

theorem repeating_decimal_to_fraction (x : ℚ) (h : x = 0.3 + 56 / 9900) : x = 3969 / 11100 := 
sorry

end repeating_decimal_to_fraction_l312_312891


namespace solve_for_a_l312_312272

def E (a b c : ℝ) : ℝ := a * b^2 + c

theorem solve_for_a (a : ℝ) : E a 3 2 = E a 5 3 ↔ a = -1/16 :=
by
  sorry

end solve_for_a_l312_312272


namespace bowling_ball_weight_l312_312653

theorem bowling_ball_weight (b c : ℕ) (h1 : 10 * b = 5 * c) (h2 : 3 * c = 120) : b = 20 := by
  sorry

end bowling_ball_weight_l312_312653


namespace days_of_earning_l312_312814

theorem days_of_earning (T D d : ℕ) (hT : T = 165) (hD : D = 33) (h : d = T / D) :
  d = 5 :=
by sorry

end days_of_earning_l312_312814


namespace midpoint_of_segment_l312_312122

theorem midpoint_of_segment (a b : ℝ) : (a + b) / 2 = (a + b) / 2 :=
sorry

end midpoint_of_segment_l312_312122


namespace parabola_vertex_origin_directrix_xaxis_point_1_neg_sqrt2_l312_312123

noncomputable def parabola_equation (x y : ℝ) : Prop :=
  y^2 = 2 * x

theorem parabola_vertex_origin_directrix_xaxis_point_1_neg_sqrt2 :
  parabola_equation 1 (-Real.sqrt 2) :=
by
  sorry

end parabola_vertex_origin_directrix_xaxis_point_1_neg_sqrt2_l312_312123


namespace escher_consecutive_probability_l312_312479

open Classical

noncomputable def probability_Escher_consecutive (total_pieces escher_pieces: ℕ): ℚ :=
  if total_pieces < escher_pieces then 0 else (Nat.factorial (total_pieces - escher_pieces) * Nat.factorial escher_pieces) / Nat.factorial (total_pieces - 1)

theorem escher_consecutive_probability :
  probability_Escher_consecutive 12 4 = 1 / 41 :=
by
  sorry

end escher_consecutive_probability_l312_312479


namespace valid_q_values_l312_312922

theorem valid_q_values (q : ℕ) (h : q > 0) :
  q = 3 ∨ q = 4 ∨ q = 9 ∨ q = 28 ↔ ((5 * q + 40) / (3 * q - 8)) * (3 * q - 8) = 5 * q + 40 :=
by
  sorry

end valid_q_values_l312_312922


namespace degrees_of_remainder_is_correct_l312_312668

noncomputable def degrees_of_remainder (P D : Polynomial ℤ) : Finset ℕ :=
  if D.degree = 3 then {0, 1, 2} else ∅

theorem degrees_of_remainder_is_correct
(P : Polynomial ℤ) :
  degrees_of_remainder P (Polynomial.C 3 * Polynomial.X^3 - Polynomial.C 5 * Polynomial.X^2 + Polynomial.C 2 * Polynomial.X - Polynomial.C 4) = {0, 1, 2} :=
by
  -- Proof omitted
  sorry

end degrees_of_remainder_is_correct_l312_312668


namespace total_amount_is_2500_l312_312245

noncomputable def total_amount_divided (P1 : ℝ) (annual_income : ℝ) : ℝ :=
  let P2 := 2500 - P1
  let income_from_P1 := (5 / 100) * P1
  let income_from_P2 := (6 / 100) * P2
  income_from_P1 + income_from_P2

theorem total_amount_is_2500 : 
  (total_amount_divided 2000 130) = 130 :=
by
  sorry

end total_amount_is_2500_l312_312245


namespace evaluate_expression_l312_312278

theorem evaluate_expression : (1 - (1 / 4)) / (1 - (1 / 5)) = 15 / 16 :=
by 
  sorry

end evaluate_expression_l312_312278


namespace pizza_slices_correct_l312_312624

-- Definitions based on conditions
def john_slices : Nat := 3
def sam_slices : Nat := 2 * john_slices
def eaten_slices : Nat := john_slices + sam_slices
def remaining_slices : Nat := 3
def total_slices : Nat := eaten_slices + remaining_slices

-- The statement to be proven.
theorem pizza_slices_correct : total_slices = 12 := by
  sorry

end pizza_slices_correct_l312_312624


namespace total_books_now_l312_312939

-- Defining the conditions
def books_initial := 100
def books_last_year := 50
def multiplier_this_year := 3

-- Proving the number of books now
theorem total_books_now : 
  let books_after_last_year := books_initial + books_last_year in
  let books_this_year := books_last_year * multiplier_this_year in
  let total_books := books_after_last_year + books_this_year in
  total_books = 300 := 
by
  sorry

end total_books_now_l312_312939


namespace sqrt_fraction_expression_eq_one_l312_312810

theorem sqrt_fraction_expression_eq_one :
  (Real.sqrt (9 / 4) - Real.sqrt (4 / 9) + 1 / 6) = 1 := 
by
  sorry

end sqrt_fraction_expression_eq_one_l312_312810


namespace remainder_when_divided_by_6_eq_5_l312_312090

theorem remainder_when_divided_by_6_eq_5 (k : ℕ) (hk1 : k % 5 = 2) (hk2 : k < 41) (hk3 : k % 7 = 3) : k % 6 = 5 :=
sorry

end remainder_when_divided_by_6_eq_5_l312_312090


namespace mike_eggs_basket_l312_312637

theorem mike_eggs_basket : ∃ k : ℕ, (30 % k = 0) ∧ (42 % k = 0) ∧ k ≥ 4 ∧ (30 / k) ≥ 3 ∧ (42 / k) ≥ 3 ∧ k = 6 := 
by
  -- skipping the proof
  sorry

end mike_eggs_basket_l312_312637


namespace find_natural_numbers_l312_312748

theorem find_natural_numbers (n : ℕ) (p q : ℕ) (hp : p.Prime) (hq : q.Prime)
  (h : q = p + 2) (h1 : (2^n + p).Prime) (h2 : (2^n + q).Prime) :
    n = 1 ∨ n = 3 :=
by
  sorry

end find_natural_numbers_l312_312748


namespace range_3a_2b_l312_312762

theorem range_3a_2b (a b : ℝ) (h : a^2 + b^2 = 4) : 
  -2 * Real.sqrt 13 ≤ 3 * a + 2 * b ∧ 3 * a + 2 * b ≤ 2 * Real.sqrt 13 := 
by 
  sorry

end range_3a_2b_l312_312762


namespace area_of_absolute_value_sum_eq_9_l312_312517

theorem area_of_absolute_value_sum_eq_9 :
  (∃ (area : ℝ), (|x| + |3 * y| = 9) → area = 54) := 
sorry

end area_of_absolute_value_sum_eq_9_l312_312517


namespace percentage_taken_l312_312081

theorem percentage_taken (P : ℝ) (h : (P / 100) * 150 - 40 = 50) : P = 60 :=
by
  sorry

end percentage_taken_l312_312081


namespace evaluate_three_cubed_squared_l312_312713

theorem evaluate_three_cubed_squared : (3^3)^2 = 729 :=
by
  -- Given the property of exponents
  have h : (forall (a m n : ℕ), (a^m)^n = a^(m * n)) := sorry,
  -- Now prove the statement using the given property
  calc
    (3^3)^2 = 3^(3 * 2) : by rw [h 3 3 2]
          ... = 3^6       : by norm_num
          ... = 729       : by norm_num

end evaluate_three_cubed_squared_l312_312713


namespace negation_of_exists_leq_l312_312355

theorem negation_of_exists_leq (
  P : ∃ x : ℝ, x^2 - 2 * x + 4 ≤ 0
) : ∀ x : ℝ, x^2 - 2 * x + 4 > 0 :=
sorry

end negation_of_exists_leq_l312_312355


namespace find_a_l312_312497

open Real

-- Definition of regression line
def regression_line (x : ℝ) : ℝ := 12.6 * x + 0.6

-- Data points for x and y
def x_values : List ℝ := [2, 3, 3.5, 4.5, 7]
def y_values : List ℝ := [26, 38, 43, 60]

-- Proof statement
theorem find_a (a : ℝ) (hx : x_values = [2, 3, 3.5, 4.5, 7])
  (hy : y_values ++ [a] = [26, 38, 43, 60, a]) : a = 88 :=
  sorry

end find_a_l312_312497


namespace countNegativeValues_l312_312422

-- Define the condition that sqrt(x + 122) is a positive integer
noncomputable def isPositiveInteger (n : ℤ) (x : ℤ) : Prop :=
  ∃ n : ℤ, (n > 0) ∧ (x + 122 = n * n)

-- Define the condition that x is negative
def isNegative (x : ℤ) : Prop :=
  x < 0

-- Prove the number of different negative values of x such that sqrt(x + 122) is a positive integer is 11
theorem countNegativeValues :
  ∃ x_set : Finset ℤ, (∀ x ∈ x_set, isNegative x ∧ isPositiveInteger x (x + 122)) ∧ x_set.card = 11 :=
sorry

end countNegativeValues_l312_312422


namespace sequence_values_l312_312619

theorem sequence_values (x y z : ℚ) :
  (∀ n : ℕ, x = 1 ∧ y = 9 / 8 ∧ z = 5 / 4) :=
by
  sorry

end sequence_values_l312_312619


namespace right_triangle_conditions_l312_312927

theorem right_triangle_conditions (A B C : ℝ) (a b c : ℝ):
  (C = 90) ∨ (A + B = C) ∨ (a/b = 3/4 ∧ a/c = 3/5 ∧ b/c = 4/5) →
  (a^2 + b^2 = c^2) ∨ (A + B + C = 180) → 
  (C = 90 ∧ a^2 + b^2 = c^2) :=
sorry

end right_triangle_conditions_l312_312927


namespace clotheslines_per_house_l312_312458

/-- There are a total of 11 children and 20 adults.
Each child has 4 items of clothing on the clotheslines.
Each adult has 3 items of clothing on the clotheslines.
Each clothesline can hold 2 items of clothing.
All of the clotheslines are full.
There are 26 houses on the street.
Show that the number of clotheslines per house is 2. -/
theorem clotheslines_per_house :
  (11 * 4 + 20 * 3) / 2 / 26 = 2 :=
by
  sorry

end clotheslines_per_house_l312_312458


namespace range_of_a_l312_312159

theorem range_of_a (a : ℝ) : ¬ (∃ x : ℝ, a * x^2 + 2 * a * x + 2 < 0) ↔ 0 ≤ a ∧ a ≤ 2 := 
by 
  sorry

end range_of_a_l312_312159


namespace more_calories_per_dollar_l312_312465

-- The conditions given in the problem as definitions
def price_burritos : ℕ := 6
def price_burgers : ℕ := 8
def calories_per_burrito : ℕ := 120
def calories_per_burger : ℕ := 400
def num_burritos : ℕ := 10
def num_burgers : ℕ := 5

-- The theorem stating the mathematically equivalent proof problem
theorem more_calories_per_dollar : 
  (num_burgers * calories_per_burger / price_burgers) - (num_burritos * calories_per_burrito / price_burritos) = 50 :=
by
  sorry

end more_calories_per_dollar_l312_312465


namespace regular_polygon_num_sides_l312_312001

def diag_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem regular_polygon_num_sides (n : ℕ) (h : diag_formula n = 20) : n = 8 :=
by
  sorry

end regular_polygon_num_sides_l312_312001


namespace t_shirt_cost_l312_312698

theorem t_shirt_cost (n_tshirts : ℕ) (total_cost : ℝ) (cost_per_tshirt : ℝ)
  (h1 : n_tshirts = 25)
  (h2 : total_cost = 248) :
  cost_per_tshirt = 9.92 :=
by
  sorry

end t_shirt_cost_l312_312698


namespace evaluate_power_l312_312704

theorem evaluate_power : (3^3)^2 = 729 := 
by 
  sorry

end evaluate_power_l312_312704


namespace prism_closed_polygonal_chain_impossible_l312_312981

theorem prism_closed_polygonal_chain_impossible
  (lateral_edges : ℕ)
  (base_edges : ℕ)
  (total_edges : ℕ)
  (h_lateral : lateral_edges = 171)
  (h_base : base_edges = 171)
  (h_total : total_edges = 513)
  (h_total_sum : total_edges = 2 * base_edges + lateral_edges) :
  ¬ (∃ f : Fin 513 → (ℝ × ℝ × ℝ), (f 513 = f 0) ∧
    ∀ i, ( f (i + 1) - f i = (1, 0, 0) ∨ f (i + 1) - f i = (0, 1, 0) ∨ f (i + 1) - f i = (0, 0, 1) ∨ f (i + 1) - f i = (0, 0, -1) )) :=
by
  sorry

end prism_closed_polygonal_chain_impossible_l312_312981


namespace find_value_of_p_l312_312045

theorem find_value_of_p (p : ℝ) :
  (∀ x y, (x = 0 ∧ y = -2) → y = p*x^2 + 5*x + p) ∧
  (∀ x y, (x = 1/2 ∧ y = 0) → y = p*x^2 + 5*x + p) ∧
  (∀ x y, (x = 2 ∧ y = 0) → y = p*x^2 + 5*x + p) →
  p = -2 :=
by
  sorry

end find_value_of_p_l312_312045


namespace simplify_and_evaluate_correct_l312_312649

noncomputable def simplify_and_evaluate (x y : ℚ) : ℚ :=
  3 * (x^2 - 2 * x * y) - (3 * x^2 - 2 * y + 2 * (x * y + y))

theorem simplify_and_evaluate_correct : 
  simplify_and_evaluate (-1 / 2 : ℚ) (-3 : ℚ) = -12 := by
  sorry

end simplify_and_evaluate_correct_l312_312649


namespace sandy_total_money_l312_312033

def half_dollar_value := 0.5
def quarter_value := 0.25
def dime_value := 0.1
def nickel_value := 0.05
def dollar_value := 1.0

def monday_total := 12 * half_dollar_value + 5 * quarter_value + 10 * dime_value
def tuesday_total := 8 * half_dollar_value + 15 * quarter_value + 5 * dime_value
def wednesday_total := 3 * dollar_value + 4 * half_dollar_value + 10 * quarter_value + 7 * nickel_value
def thursday_total := 5 * dollar_value + 6 * half_dollar_value + 8 * quarter_value + 5 * dime_value + 12 * nickel_value
def friday_total := 2 * dollar_value + 7 * half_dollar_value + 20 * nickel_value + 25 * dime_value

def total_amount := monday_total + tuesday_total + wednesday_total + thursday_total + friday_total

theorem sandy_total_money : total_amount = 44.45 := by
  sorry

end sandy_total_money_l312_312033


namespace find_m_value_l312_312150

theorem find_m_value (m : ℝ) (h1 : 2 ∈ ({0, m, m^2 - 3 * m + 2} : set ℝ)) : m = 3 :=
sorry

end find_m_value_l312_312150


namespace min_value_expression_l312_312796

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : y = Real.sqrt x) :
  ∃ c, c = 2 ∧ ∀ u v : ℝ, 0 < u → v = Real.sqrt u → (u^2 + v^4) / (u * v^2) = c :=
by
  sorry

end min_value_expression_l312_312796


namespace pet_store_customers_buy_different_pets_l312_312542

theorem pet_store_customers_buy_different_pets :
  let puppies := 20
  let kittens := 10
  let hamsters := 12
  let rabbits := 5
  let customers := 4
  (puppies * kittens * hamsters * rabbits * Nat.factorial customers = 288000) := 
by
  sorry

end pet_store_customers_buy_different_pets_l312_312542


namespace liquid_in_cylinders_l312_312817

theorem liquid_in_cylinders (n : ℕ) (a : ℝ) (h1 : 2 ≤ n) :
  (∃ x : ℕ → ℝ, ∀ (k : ℕ), (1 ≤ k ∧ k ≤ n) → 
    (if k = 1 then 
      x k = a * n * (n - 2) / (n - 1) ^ 2 
    else if k = 2 then 
      x k = a * (n^2 - 2*n + 2) / (n - 1) ^ 2 
    else 
      x k = a)) :=
sorry

end liquid_in_cylinders_l312_312817


namespace ratio_of_areas_l312_312975

theorem ratio_of_areas (n : ℕ) (r s : ℕ) (square_area : ℕ) (triangle_adf_area : ℕ)
  (h_square_area : square_area = 4)
  (h_triangle_adf_area : triangle_adf_area = n * square_area)
  (h_triangle_sim : s = 8 / r)
  (h_r_eq_n : r = n):
  (s / square_area) = 2 / n :=
by
  sorry

end ratio_of_areas_l312_312975


namespace prime_division_l312_312634

-- Definitions used in conditions
variables {p q : ℕ}

-- We assume p and q are prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n
def divides (a b : ℕ) : Prop := ∃ k, b = k * a

-- The problem states
theorem prime_division 
  (hp : is_prime p) 
  (hq : is_prime q) 
  (hdiv : divides q (3^p - 2^p)) 
  : p ∣ (q - 1) :=
sorry

end prime_division_l312_312634


namespace sin_theta_add_pi_over_3_l312_312759

theorem sin_theta_add_pi_over_3 (θ : ℝ) (h : Real.cos (π / 6 - θ) = 2 / 3) : 
  Real.sin (θ + π / 3) = 2 / 3 :=
sorry

end sin_theta_add_pi_over_3_l312_312759


namespace quadratic_expression_min_value_l312_312139

noncomputable def min_value_quadratic_expression (x y z : ℝ) : ℝ :=
(x + 5) ^ 2 + (y - 1) ^ 2 + (z + 3) ^ 2

theorem quadratic_expression_min_value :
  ∃ x y z : ℝ, x - 2 * y + 2 * z = 5 ∧ min_value_quadratic_expression x y z = 36 :=
sorry

end quadratic_expression_min_value_l312_312139


namespace shelves_fit_l312_312170

-- Define the total space of the room for the library
def totalSpace : ℕ := 400

-- Define the space each bookshelf takes up
def spacePerBookshelf : ℕ := 80

-- Define the reserved space for desk and walking area
def reservedSpace : ℕ := 160

-- Define the space available for bookshelves
def availableSpace : ℕ := totalSpace - reservedSpace

-- Define the number of bookshelves that can fit in the available space
def numberOfBookshelves : ℕ := availableSpace / spacePerBookshelf

-- The theorem stating the number of bookshelves Jonas can fit in the room
theorem shelves_fit : numberOfBookshelves = 3 := by
  -- We can defer the proof as we only need the statement for now
  sorry

end shelves_fit_l312_312170


namespace number_of_bowls_l312_312831

theorem number_of_bowls (n : ℕ) (h1 : 12 * 8 = 96) (h2 : ∀ t : ℕ, t = 6 * n -> t = 96) : n = 16 := by
  sorry

end number_of_bowls_l312_312831


namespace gcd_polynomial_l312_312432

theorem gcd_polynomial (b : ℤ) (h : 1820 ∣ b) : Int.gcd (b^2 + 11 * b + 28) (b + 6) = 2 := 
sorry

end gcd_polynomial_l312_312432


namespace father_son_speed_ratio_l312_312084

theorem father_son_speed_ratio
  (F S t : ℝ)
  (distance_hallway : ℝ)
  (distance_meet_from_father : ℝ)
  (H1 : distance_hallway = 16)
  (H2 : distance_meet_from_father = 12)
  (H3 : 12 = F * t)
  (H4 : 4 = S * t)
  : F / S = 3 := by
  sorry

end father_son_speed_ratio_l312_312084


namespace range_of_m_l312_312900

theorem range_of_m (f g : ℝ → ℝ) (m : ℝ) :
  (∀ x1 : ℝ, 0 < x1 ∧ x1 < 3 / 2 → ∃ x2 : ℝ, 0 < x2 ∧ x2 < 3 / 2 ∧ f x1 > g x2) →
  (∀ x : ℝ, f x = -x + x * Real.log x + m) →
  (∀ x : ℝ, g x = -3 * Real.exp x / (3 + 4 * x ^ 2)) →
  m > 1 - 3 / 4 * Real.sqrt (Real.exp 1) :=
by
  sorry

end range_of_m_l312_312900


namespace downstream_speed_l312_312870

-- Definitions based on the conditions
def V_m : ℝ := 50 -- speed of the man in still water
def V_upstream : ℝ := 45 -- speed of the man when rowing upstream

-- The statement to prove
theorem downstream_speed : ∃ (V_s V_downstream : ℝ), V_upstream = V_m - V_s ∧ V_downstream = V_m + V_s ∧ V_downstream = 55 := 
by
  sorry

end downstream_speed_l312_312870


namespace part_I_part_II_l312_312026

def f (x : ℝ) : ℝ := abs (2 * x - 7) + 1

def g (x : ℝ) : ℝ := abs (2 * x - 7) - 2 * abs (x - 1) + 1

theorem part_I :
  {x : ℝ | f x ≤ x} = {x : ℝ | (8 / 3) ≤ x ∧ x ≤ 6} := sorry

theorem part_II (a : ℝ) :
  (∃ x : ℝ, g x ≤ a) → a ≥ -4 := sorry

end part_I_part_II_l312_312026


namespace chocolate_bar_weight_l312_312549

theorem chocolate_bar_weight :
  let square_weight := 6
  let triangles_count := 16
  let squares_count := 32
  let triangle_weight := square_weight / 2
  let total_square_weight := squares_count * square_weight
  let total_triangles_weight := triangles_count * triangle_weight
  total_square_weight + total_triangles_weight = 240 := 
by
  sorry

end chocolate_bar_weight_l312_312549


namespace least_subtraction_divisible_l312_312854

theorem least_subtraction_divisible (n : ℕ) (h : n = 3830) (lcm_val : ℕ) (hlcm : lcm_val = Nat.lcm (Nat.lcm 3 7) 11) 
(largest_multiple : ℕ) (h_largest : largest_multiple = (n / lcm_val) * lcm_val) :
  ∃ x : ℕ, x = n - largest_multiple ∧ x = 134 := 
by
  sorry

end least_subtraction_divisible_l312_312854


namespace direction_vector_of_line_l312_312869

theorem direction_vector_of_line : 
  ∃ v : ℝ × ℝ, 
  (∀ x y : ℝ, 2 * y + x = 3 → v = (-2, -1)) :=
by
  sorry

end direction_vector_of_line_l312_312869


namespace instantaneous_velocity_at_1_l312_312253

noncomputable def S (t : ℝ) : ℝ := t^2 + 2 * t

theorem instantaneous_velocity_at_1 : (deriv S 1) = 4 :=
by 
  -- The proof is left as an exercise
  sorry

end instantaneous_velocity_at_1_l312_312253


namespace dave_more_than_derek_l312_312560

def derek_initial : ℕ := 40
def derek_spent_on_self1 : ℕ := 14
def derek_spent_on_dad : ℕ := 11
def derek_spent_on_self2 : ℕ := 5

def dave_initial : ℕ := 50
def dave_spent_on_mom : ℕ := 7

def derek_remaining : ℕ := derek_initial - (derek_spent_on_self1 + derek_spent_on_dad + derek_spent_on_self2)
def dave_remaining : ℕ := dave_initial - dave_spent_on_mom

theorem dave_more_than_derek : dave_remaining - derek_remaining = 33 :=
by
  -- The proof goes here
  sorry

end dave_more_than_derek_l312_312560


namespace triangle_equilateral_if_abs_eq_zero_l312_312425

theorem triangle_equilateral_if_abs_eq_zero (a b c : ℝ) (h : abs (a - b) + abs (b - c) = 0) : a = b ∧ b = c :=
by
  sorry

end triangle_equilateral_if_abs_eq_zero_l312_312425


namespace apples_left_is_correct_l312_312594

-- Definitions for the conditions
def blue_apples : ℕ := 5
def yellow_apples : ℕ := 2 * blue_apples
def total_apples : ℕ := blue_apples + yellow_apples
def apples_given_to_son : ℚ := 1 / 5 * total_apples
def apples_left : ℚ := total_apples - apples_given_to_son

-- The main statement to be proven
theorem apples_left_is_correct : apples_left = 12 := by
  sorry

end apples_left_is_correct_l312_312594


namespace distribution_properties_l312_312248

theorem distribution_properties (m d j s k : ℝ) (h1 : True)
  (h2 : True)
  (h3 : True)
  (h4 : 68 ≤ 100 ∧ 68 ≥ 0) -- 68% being a valid percentage
  : j = 84 ∧ s = s ∧ k = k :=
by
  -- sorry is used to highlight the proof is not included
  sorry

end distribution_properties_l312_312248


namespace range_f_subset_interval_l312_312292

-- Define the function f on real numbers
def f : ℝ → ℝ := sorry

-- The given condition for all real numbers x and y such that x > y
axiom condition (x y : ℝ) (h : x > y) : (f x)^2 ≤ f y

-- The main theorem that needs to be proven
theorem range_f_subset_interval : ∀ x, 0 ≤ f x ∧ f x ≤ 1 := 
by
  intro x
  apply And.intro
  -- Proof for 0 ≤ f x
  sorry
  -- Proof for f x ≤ 1
  sorry

end range_f_subset_interval_l312_312292


namespace find_y_interval_l312_312120

open Real

theorem find_y_interval {y : ℝ}
  (hy_nonzero : y ≠ 0)
  (h_denominator_nonzero : 1 + 3 * y - 4 * y^2 ≠ 0) :
  (y^2 + 9 * y - 1 = 0) →
  (∀ y, y ∈ Set.Icc (-(9 + sqrt 85)/2) (-(9 - sqrt 85)/2) \ {y | y = 0 ∨ 1 + 3 * y - 4 * y^2 = 0} ↔
  (y * (3 - 3 * y))/(1 + 3 * y - 4 * y^2) ≤ 1) :=
by
  sorry

end find_y_interval_l312_312120


namespace range_of_a_l312_312138

def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + (a - 1)*x - 1 < 0
def r (a : ℝ) : Prop := -1 ≤ a ∧ a ≤ 1 ∨ a > 3

theorem range_of_a (a : ℝ) (h : (p a ∨ q a) ∧ ¬(p a ∧ q a)) : r a := 
by sorry

end range_of_a_l312_312138


namespace birds_count_214_l312_312276

def two_legged_birds_count (b m i : Nat) : Prop :=
  b + m + i = 300 ∧ 2 * b + 4 * m + 3 * i = 686 → b = 214

theorem birds_count_214 (b m i : Nat) : two_legged_birds_count b m i :=
by
  sorry

end birds_count_214_l312_312276


namespace ratio_of_areas_l312_312347

theorem ratio_of_areas (s : ℝ) (h_s_pos : 0 < s) :
  let small_triangle_area := (s^2 * Real.sqrt 3) / 4
  let total_small_triangles_area := 6 * small_triangle_area
  let large_triangle_side := 6 * s
  let large_triangle_area := (large_triangle_side^2 * Real.sqrt 3) / 4
  total_small_triangles_area / large_triangle_area = 1 / 6 :=
by
  let small_triangle_area := (s^2 * Real.sqrt 3) / 4
  let total_small_triangles_area := 6 * small_triangle_area
  let large_triangle_side := 6 * s
  let large_triangle_area := (large_triangle_side^2 * Real.sqrt 3) / 4
  sorry
 
end ratio_of_areas_l312_312347


namespace arithmetic_sequence_a15_l312_312435

theorem arithmetic_sequence_a15 
  (a : ℕ → ℤ) 
  (h_arith_seq : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h1 : a 3 + a 13 = 20)
  (h2 : a 2 = -2) :
  a 15 = 24 := 
by
  sorry

end arithmetic_sequence_a15_l312_312435


namespace average_sum_problem_l312_312313

theorem average_sum_problem (avg : ℝ) (n : ℕ) (h_avg : avg = 5.3) (h_n : n = 10) : ∃ sum : ℝ, sum = avg * n ∧ sum = 53 :=
by
  sorry

end average_sum_problem_l312_312313


namespace quadratic_roots_identity_l312_312970

theorem quadratic_roots_identity
  (a b c : ℝ)
  (x1 x2 : ℝ)
  (hx1 : x1 = Real.sin (42 * Real.pi / 180))
  (hx2 : x2 = Real.sin (48 * Real.pi / 180))
  (hx2_trig_identity : x2 = Real.cos (42 * Real.pi / 180))
  (hroots : ∀ x, a * x^2 + b * x + c = 0 ↔ (x = x1 ∨ x = x2)) :
  b^2 = a^2 + 2 * a * c :=
by
  sorry

end quadratic_roots_identity_l312_312970


namespace number_of_bowls_l312_312845

-- Let n be the number of bowls on the table.
variable (n : ℕ)

-- Condition 1: There are n bowls, and each contain some grapes.
-- Condition 2: Adding 8 grapes to each of 12 specific bowls increases the average number of grapes in all bowls by 6.
-- Let's formalize the condition given in the problem
theorem number_of_bowls (h1 : 12 * 8 = 96) (h2 : 6 * n = 96) : n = 16 :=
by
  -- omitting the proof here
  sorry

end number_of_bowls_l312_312845


namespace train_crosses_pole_in_l312_312166

noncomputable def train_crossing_time (length : ℝ) (speed_km_hr : ℝ) : ℝ :=
  let speed_m_s := speed_km_hr * (5.0 / 18.0)
  length / speed_m_s

theorem train_crosses_pole_in : train_crossing_time 175 180 = 3.5 :=
by
  -- Proof would be here, but for now, it is omitted.
  sorry

end train_crosses_pole_in_l312_312166


namespace power_of_power_evaluation_l312_312739

theorem power_of_power_evaluation : (3^3)^2 = 729 := 
by
  -- Replace this with the actual proof
  sorry

end power_of_power_evaluation_l312_312739


namespace surface_area_of_each_smaller_cube_l312_312454

theorem surface_area_of_each_smaller_cube
  (L : ℝ) (l : ℝ)
  (h1 : 6 * L^2 = 600)
  (h2 : 125 * l^3 = L^3) :
  6 * l^2 = 24 := by
  sorry

end surface_area_of_each_smaller_cube_l312_312454


namespace volume_of_water_overflow_l312_312536

-- Definitions based on given conditions
def mass_of_ice : ℝ := 50
def density_of_fresh_ice : ℝ := 0.9
def density_of_salt_ice : ℝ := 0.95
def density_of_fresh_water : ℝ := 1
def density_of_salt_water : ℝ := 1.03

-- Theorem statement corresponding to the problem
theorem volume_of_water_overflow
  (m : ℝ := mass_of_ice) 
  (rho_n : ℝ := density_of_fresh_ice) 
  (rho_c : ℝ := density_of_salt_ice) 
  (rho_fw : ℝ := density_of_fresh_water) 
  (rho_sw : ℝ := density_of_salt_water) :
  ∃ (ΔV : ℝ), ΔV = 2.63 :=
by
  sorry

end volume_of_water_overflow_l312_312536


namespace find_f_l312_312769

def f (x : ℝ) : ℝ := 3 * x + 2

theorem find_f (x : ℝ) : f x = 3 * x + 2 :=
  sorry

end find_f_l312_312769


namespace pulley_weight_l312_312397

theorem pulley_weight (M g : ℝ) (hM_pos : 0 < M) (F : ℝ := 50) :
  (g ≠ 0) → (M * g = 100) :=
by
  sorry

end pulley_weight_l312_312397


namespace candy_distribution_l312_312799

-- Define the required parameters and conditions.
def num_distinct_candies : ℕ := 9
def num_bags : ℕ := 3

-- The result that we need to prove
theorem candy_distribution :
  (3 ^ num_distinct_candies) - 3 * (2 ^ (num_distinct_candies - 1) - 2) = 18921 := by
  sorry

end candy_distribution_l312_312799


namespace john_new_earnings_after_raise_l312_312019

-- Definition of original earnings and raise percentage
def original_earnings : ℝ := 50
def raise_percentage : ℝ := 0.50

-- Calculate raise amount and new earnings after raise
def raise_amount : ℝ := raise_percentage * original_earnings
def new_earnings : ℝ := original_earnings + raise_amount

-- Math proof problem: Prove new earnings after raise equals $75
theorem john_new_earnings_after_raise : new_earnings = 75 := by
  sorry

end john_new_earnings_after_raise_l312_312019


namespace adults_eat_one_third_l312_312161

theorem adults_eat_one_third (n c k : ℕ) (hn : n = 120) (hc : c = 4) (hk : k = 20) :
  ((n - c * k) / n : ℚ) = 1 / 3 :=
by
  sorry

end adults_eat_one_third_l312_312161


namespace isosceles_triangle_sides_l312_312787

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

end isosceles_triangle_sides_l312_312787


namespace number_of_bowls_l312_312836

theorem number_of_bowls (n : ℕ) :
  (∀ (b : ℕ), b > 0) →
  (∀ (a : ℕ), ∃ (k : ℕ), true) →
  (8 * 12 = 96) →
  (6 * n = 96) →
  n = 16 :=
by
  intros h1 h2 h3 h4
  sorry

end number_of_bowls_l312_312836


namespace james_total_riding_time_including_rest_stop_l312_312622

theorem james_total_riding_time_including_rest_stop :
  let distance1 := 40 -- miles
  let speed1 := 16 -- miles per hour
  let distance2 := 40 -- miles
  let speed2 := 20 -- miles per hour
  let rest_stop := 20 -- minutes
  let rest_stop_in_hours := rest_stop / 60 -- convert to hours
  let time1 := distance1 / speed1 -- time for the first part
  let time2 := distance2 / speed2 -- time for the second part
  let total_time := time1 + rest_stop_in_hours + time2 -- total time including rest
  total_time = 4.83 :=
by
  sorry

end james_total_riding_time_including_rest_stop_l312_312622


namespace cos_2alpha_l312_312923

theorem cos_2alpha (α : ℝ) (h : Real.cos (Real.pi / 2 + α) = (1 : ℝ) / 3) : 
  Real.cos (2 * α) = (7 : ℝ) / 9 := 
by
  sorry

end cos_2alpha_l312_312923


namespace distance_walked_hazel_l312_312913

theorem distance_walked_hazel (x : ℝ) (h : x + 2 * x = 6) : x = 2 :=
sorry

end distance_walked_hazel_l312_312913


namespace feeding_sequences_count_l312_312091

def num_feeding_sequences (num_pairs : ℕ) : ℕ :=
  num_pairs * num_pairs.pred * num_pairs.pred * num_pairs.pred.pred *
  num_pairs.pred.pred * num_pairs.pred.pred.pred * num_pairs.pred.pred.pred *
  1 * 1

theorem feeding_sequences_count (num_pairs : ℕ) (h : num_pairs = 5) :
  num_feeding_sequences num_pairs = 5760 := 
by
  rw [h]
  unfold num_feeding_sequences
  norm_num
  sorry

end feeding_sequences_count_l312_312091


namespace find_max_a_l312_312298

def f (a x : ℝ) := a * x^3 - x

theorem find_max_a (a : ℝ) (h : ∃ t : ℝ, |f a (t + 2) - f a t| ≤ 2 / 3) :
  a ≤ 4 / 3 :=
sorry

end find_max_a_l312_312298


namespace number_of_bowls_l312_312824

theorem number_of_bowls (n : ℕ) (h : 8 * 12 = 96) (avg_increase : 6 * n = 96) : n = 16 :=
by {
  sorry
}

end number_of_bowls_l312_312824


namespace average_typed_words_per_minute_l312_312354

def rudy_wpm := 64
def joyce_wpm := 76
def gladys_wpm := 91
def lisa_wpm := 80
def mike_wpm := 89
def num_team_members := 5

theorem average_typed_words_per_minute : 
  (rudy_wpm + joyce_wpm + gladys_wpm + lisa_wpm + mike_wpm) / num_team_members = 80 := 
by
  sorry

end average_typed_words_per_minute_l312_312354


namespace power_of_power_evaluation_l312_312738

theorem power_of_power_evaluation : (3^3)^2 = 729 := 
by
  -- Replace this with the actual proof
  sorry

end power_of_power_evaluation_l312_312738


namespace opposite_of_neg_six_l312_312360

theorem opposite_of_neg_six : -(-6) = 6 := 
by
  sorry

end opposite_of_neg_six_l312_312360


namespace last_two_digits_7_pow_2011_l312_312184

noncomputable def pow_mod_last_two_digits (n : ℕ) : ℕ :=
  (7^n) % 100

theorem last_two_digits_7_pow_2011 : pow_mod_last_two_digits 2011 = 43 :=
by
  sorry

end last_two_digits_7_pow_2011_l312_312184


namespace total_earnings_correct_l312_312073

-- Definitions for the conditions
def price_per_bracelet := 5
def price_for_two_bracelets := 8
def initial_bracelets := 30
def earnings_from_selling_at_5_each := 60

-- Variables to store intermediate calculations
def bracelets_sold_at_5_each := earnings_from_selling_at_5_each / price_per_bracelet
def remaining_bracelets := initial_bracelets - bracelets_sold_at_5_each
def pairs_sold_at_8_each := remaining_bracelets / 2
def earnings_from_pairs := pairs_sold_at_8_each * price_for_two_bracelets
def total_earnings := earnings_from_selling_at_5_each + earnings_from_pairs

-- The theorem stating that Zayne made $132 in total
theorem total_earnings_correct :
  total_earnings = 132 :=
sorry

end total_earnings_correct_l312_312073


namespace seq_equality_iff_initial_equality_l312_312912

variable {α : Type*} [AddGroup α]

-- Definition of sequences and their differences
def sequence_diff (u : ℕ → α) (v : ℕ → α) : Prop := ∀ n, (u (n+1) - u n) = (v (n+1) - v n)

-- Main theorem statement
theorem seq_equality_iff_initial_equality (u v : ℕ → α) :
  sequence_diff u v → (∀ n, u n = v n) ↔ (u 1 = v 1) :=
by
  sorry

end seq_equality_iff_initial_equality_l312_312912


namespace taxi_fare_proportional_l312_312685

theorem taxi_fare_proportional (fare_50 : ℝ) (distance_50 distance_70 : ℝ) (proportional : Prop) (h_fare_50 : fare_50 = 120) (h_distance_50 : distance_50 = 50) (h_distance_70 : distance_70 = 70) :
  fare_70 = 168 :=
by {
  sorry
}

end taxi_fare_proportional_l312_312685


namespace lock_probability_l312_312767

/-- The probability of correctly guessing the last digit of a three-digit combination lock,
given that the first two digits are correctly set and each digit ranges from 0 to 9. -/
theorem lock_probability : 
  ∀ (d1 d2 : ℕ), 
  (0 ≤ d1 ∧ d1 < 10) ∧ (0 ≤ d2 ∧ d2 < 10) →
  (0 ≤ d3 ∧ d3 < 10) → 
  (1/10 : ℝ) = (1 : ℝ) / (10 : ℝ) :=
by
  sorry

end lock_probability_l312_312767


namespace number_of_bowls_l312_312834

theorem number_of_bowls (n : ℕ) (h1 : 12 * 8 = 96) (h2 : ∀ t : ℕ, t = 6 * n -> t = 96) : n = 16 := by
  sorry

end number_of_bowls_l312_312834


namespace probability_of_negative_product_l312_312386

def m : Set ℤ := {-6, -5, -4, -3, -2, -1}
def t : Set ℤ := {-5, -4, -3, -2, -1, 0, 1, 2}

-- Define the count of negative elements in set m
def count_neg_m : ℤ := 6

-- Define the count of positive elements in set t
def count_pos_t : ℤ := 3

-- Define the total combinations count
def total_combinations : ℤ := (Set.cardinal m).toInt * (Set.cardinal t).toInt -- cardinal function returns cardinality, convert cardinal to int

-- Define the favorable combinations count
def favorable_combinations : ℤ := count_neg_m * count_pos_t

-- Statement to prove
theorem probability_of_negative_product : 
    (favorable_combinations : ℚ) / (total_combinations : ℚ) = 3 / 8 := 
sorry

end probability_of_negative_product_l312_312386


namespace kernel_count_in_final_bag_l312_312801

namespace PopcornKernelProblem

def percentage_popped (popped total : ℕ) : ℤ := ((popped : ℤ) * 100) / (total : ℤ)

def first_bag_percentage := percentage_popped 60 75
def second_bag_percentage := percentage_popped 42 50
def final_bag_percentage (x : ℕ) : ℤ := percentage_popped 82 x

theorem kernel_count_in_final_bag :
  (first_bag_percentage + second_bag_percentage + final_bag_percentage 100) / 3 = 82 := 
sorry

end PopcornKernelProblem

end kernel_count_in_final_bag_l312_312801


namespace total_amount_of_money_l312_312865

def one_rupee_note_value := 1
def five_rupee_note_value := 5
def ten_rupee_note_value := 10

theorem total_amount_of_money (n : ℕ) 
  (h : 3 * n = 90) : n * one_rupee_note_value + n * five_rupee_note_value + n * ten_rupee_note_value = 480 :=
by
  sorry

end total_amount_of_money_l312_312865


namespace xy_squared_sum_l312_312921

theorem xy_squared_sum {x y : ℝ} (h1 : (x + y)^2 = 49) (h2 : x * y = 10) : x^2 + y^2 = 29 :=
by
  sorry

end xy_squared_sum_l312_312921


namespace monotonicity_condition_l312_312146

open Real

-- Define the given function f(x)
noncomputable def f (x a : ℝ) := x - (2 / x) + a * (2 - log x)

-- Define the derivative of f(x)
noncomputable def f' (x a : ℝ) := (1:ℝ) + (2 / x^2) - (a / x)

-- Prove that if f(x) is monotonically decreasing in the interval (1, 2), then a ≥ 3
theorem monotonicity_condition (a : ℝ) (h: ∀ x ∈ Ioo (1:ℝ) (2:ℝ), f'(x, a) ≤ 0) : 3 ≤ a :=
by
  have h1 := h 1 (by simp)
  have h2 := h 2 (by simp)
  sorry

end monotonicity_condition_l312_312146


namespace not_consecutive_l312_312510

theorem not_consecutive (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) : 
  ¬ (∃ n : ℕ, (2023 + a - b = n ∧ 2023 + b - c = n + 1 ∧ 2023 + c - a = n + 2) ∨ 
    (2023 + a - b = n ∧ 2023 + b - c = n - 1 ∧ 2023 + c - a = n - 2)) :=
by
  sorry

end not_consecutive_l312_312510


namespace num_of_possible_values_b_l312_312492

-- Define the positive divisors set of 24
def divisors_of_24 := {n : ℕ | n > 0 ∧ 24 % n = 0}

-- Define the subset where 4 is a factor of each element
def divisors_of_24_div_by_4 := {b : ℕ | b ∈ divisors_of_24 ∧ b % 4 = 0}

-- Prove the number of positive values b where 4 is a factor of b and b is a divisor of 24 is 4
theorem num_of_possible_values_b : finset.card (finset.filter (λ b, b % 4 = 0) (finset.filter (λ n, 24 % n = 0) (finset.range 25))) = 4 :=
by
  sorry

end num_of_possible_values_b_l312_312492


namespace evaluate_exponent_l312_312733

theorem evaluate_exponent : (3^3)^2 = 729 := by
  sorry

end evaluate_exponent_l312_312733


namespace count_two_digit_numbers_with_unit_7_lt_50_l312_312915

def is_two_digit_nat (n : ℕ) : Prop := n ≥ 10 ∧ n < 100
def has_unit_digit_7 (n : ℕ) : Prop := n % 10 = 7
def less_than_50 (n : ℕ) : Prop := n < 50

theorem count_two_digit_numbers_with_unit_7_lt_50 : 
  ∃ (s : Finset ℕ), 
    (∀ n ∈ s, is_two_digit_nat n ∧ has_unit_digit_7 n ∧ less_than_50 n) ∧ s.card = 4 := 
by
  sorry

end count_two_digit_numbers_with_unit_7_lt_50_l312_312915


namespace base_angle_of_isosceles_triangle_l312_312156

-- Definitions based on the problem conditions
def is_isosceles_triangle (A B C: ℝ) := (A = B) ∨ (B = C) ∨ (C = A)
def angle_sum_triangle (A B C: ℝ) := A + B + C = 180

-- The main theorem we want to prove
theorem base_angle_of_isosceles_triangle (A B C: ℝ)
(h1: is_isosceles_triangle A B C)
(h2: A = 50 ∨ B = 50 ∨ C = 50):
C = 50 ∨ C = 65 :=
by
  sorry

end base_angle_of_isosceles_triangle_l312_312156


namespace right_isosceles_triangle_areas_l312_312486

theorem right_isosceles_triangle_areas :
  let A := (5 * 5) / 2
  let B := (12 * 12) / 2
  let C := (13 * 13) / 2
  A + B = C :=
by
  let A := (5 * 5) / 2
  let B := (12 * 12) / 2
  let C := (13 * 13) / 2
  sorry

end right_isosceles_triangle_areas_l312_312486


namespace correct_calculation_l312_312064

theorem correct_calculation :
  (∀ (x y : ℝ), -3 * x - 3 * x ≠ 0) ∧
  (∀ (x : ℝ), x - 4 * x ≠ -3) ∧
  (∀ (x : ℝ), 2 * x + 3 * x^2 ≠ 5 * x^3) ∧
  (∀ (x y : ℝ), -4 * x * y + 3 * x * y = -x * y) :=
by
  sorry

end correct_calculation_l312_312064


namespace population_weight_of_500_students_l312_312367

-- Definitions
def number_of_students : ℕ := 500
def number_of_selected_students : ℕ := 60

-- Conditions
def condition1 := number_of_students = 500
def condition2 := number_of_selected_students = 60

-- Statement
theorem population_weight_of_500_students : 
  condition1 → condition2 → 
  (∃ p, p = "the weight of the 500 students") := by
  intros _ _
  existsi "the weight of the 500 students"
  rfl

end population_weight_of_500_students_l312_312367


namespace rate_of_drawing_barbed_wire_per_meter_l312_312498

-- Defining the given conditions as constants
def area : ℝ := 3136
def gates_width_total : ℝ := 2 * 1
def total_cost : ℝ := 666

-- Constants used for intermediary calculations
def side_length : ℝ := Real.sqrt area
def perimeter : ℝ := 4 * side_length
def barbed_wire_length : ℝ := perimeter - gates_width_total

-- Formulating the theorem
theorem rate_of_drawing_barbed_wire_per_meter : 
    (total_cost / barbed_wire_length) = 3 := 
by
  sorry

end rate_of_drawing_barbed_wire_per_meter_l312_312498


namespace all_lights_on_l312_312196

def light_on (n : ℕ) : Prop := sorry

axiom light_rule_1 (k : ℕ) (hk: light_on k): light_on (2 * k) ∧ light_on (2 * k + 1)
axiom light_rule_2 (k : ℕ) (hk: ¬ light_on k): ¬ light_on (4 * k + 1) ∧ ¬ light_on (4 * k + 3)
axiom light_2023_on : light_on 2023

theorem all_lights_on (n : ℕ) (hn : n < 2023) : light_on n :=
by sorry

end all_lights_on_l312_312196


namespace quadratic_roots_difference_l312_312941

theorem quadratic_roots_difference (a b : ℝ) :
  (5 * a^2 - 30 * a + 45 = 0) ∧ (5 * b^2 - 30 * b + 45 = 0) → (a - b)^2 = 0 :=
by
  sorry

end quadratic_roots_difference_l312_312941


namespace find_c_l312_312449

theorem find_c (a c : ℝ) (h1 : x^2 + 80 * x + c = (x + a)^2) (h2 : 2 * a = 80) : c = 1600 :=
sorry

end find_c_l312_312449


namespace alley_width_theorem_l312_312014

noncomputable def width_of_alley (a k h : ℝ) (h₁ : k = a / 2) (h₂ : h = a * (Real.sqrt 2) / 2) : ℝ :=
  Real.sqrt ((a * (Real.sqrt 2) / 2)^2 + (a / 2)^2)

theorem alley_width_theorem (a k h w : ℝ)
  (h₁ : k = a / 2)
  (h₂ : h = a * (Real.sqrt 2) / 2)
  (h₃ : w = width_of_alley a k h h₁ h₂) :
  w = (Real.sqrt 3) * a / 2 :=
by
  sorry

end alley_width_theorem_l312_312014


namespace number_of_bowls_l312_312832

theorem number_of_bowls (n : ℕ) (h1 : 12 * 8 = 96) (h2 : ∀ t : ℕ, t = 6 * n -> t = 96) : n = 16 := by
  sorry

end number_of_bowls_l312_312832


namespace min_benches_l312_312680
-- Import the necessary library

-- Defining the problem in Lean statement
theorem min_benches (N : ℕ) :
  (∀ a c : ℕ, (8 * N = a) ∧ (12 * N = c) ∧ (a = c)) → N = 6 :=
by
  sorry

end min_benches_l312_312680


namespace part1_part2_l312_312105

-- Part 1
theorem part1 : (9 / 4) ^ (1 / 2) - (-2.5) ^ 0 - (8 / 27) ^ (2 / 3) + (3 / 2) ^ (-2) = 1 / 2 := 
by sorry

-- Part 2
theorem part2 (lg : ℝ → ℝ) -- Assuming a hypothetical lg function for demonstration
  (lg_prop1 : lg 10 = 1)
  (lg_prop2 : ∀ x y, lg (x * y) = lg x + lg y) :
  (lg 5) ^ 2 + lg 2 * lg 50 = 1 := 
by sorry

end part1_part2_l312_312105


namespace opposite_of_neg_six_is_six_l312_312361

theorem opposite_of_neg_six_is_six : ∃ x, -6 + x = 0 ∧ x = 6 := by
  use 6
  split
  · rfl
  · rfl

end opposite_of_neg_six_is_six_l312_312361


namespace ellipse_x_intersection_l312_312882

theorem ellipse_x_intersection 
  (F₁ F₂ : ℝ × ℝ)
  (origin : ℝ × ℝ)
  (x_intersect : ℝ × ℝ)
  (h₁ : F₁ = (0, 3))
  (h₂ : F₂ = (4, 0))
  (h₃ : origin = (0, 0))
  (h₄ : ∀ P : ℝ × ℝ, (dist P F₁ + dist P F₂ = 7) ↔ (P = origin ∨ P = x_intersect))
  : x_intersect = (56 / 11, 0) := sorry

end ellipse_x_intersection_l312_312882


namespace total_students_is_17_l312_312162

def total_students_in_class (students_liking_both_baseball_football : ℕ)
                             (students_only_baseball : ℕ)
                             (students_only_football : ℕ)
                             (students_liking_basketball_as_well : ℕ)
                             (students_liking_basketball_and_football_only : ℕ)
                             (students_liking_all_three : ℕ)
                             (students_liking_none : ℕ) : ℕ :=
  students_liking_both_baseball_football -
  students_liking_all_three +
  students_only_baseball +
  students_only_football +
  students_liking_basketball_and_football_only +
  students_liking_all_three +
  students_liking_none +
  (students_liking_basketball_as_well -
   (students_liking_all_three +
    students_liking_basketball_and_football_only))

theorem total_students_is_17 :
    total_students_in_class 7 3 4 2 1 2 5 = 17 :=
by sorry

end total_students_is_17_l312_312162


namespace problem_a_b_n_l312_312295

theorem problem_a_b_n (a b n : ℕ) (h : ∀ k : ℕ, k ≠ 0 → (b - k) ∣ (a - k^n)) : a = b^n := 
sorry

end problem_a_b_n_l312_312295


namespace eval_expr_l312_312744

theorem eval_expr : (3^3)^2 = 729 := 
by
  sorry

end eval_expr_l312_312744


namespace abc_equality_l312_312007

theorem abc_equality (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
                      (h : a^3 + b^3 + c^3 - 3 * a * b * c = 0) : a = b ∧ b = c :=
by
  sorry

end abc_equality_l312_312007


namespace eval_exp_l312_312726

theorem eval_exp : (3^3)^2 = 729 := sorry

end eval_exp_l312_312726


namespace probability_heart_or_king_l312_312241

theorem probability_heart_or_king (cards hearts kings : ℕ) (prob_non_heart_king : ℚ) 
    (prob_two_non_heart_king : ℚ) : 
    cards = 52 → hearts = 13 → kings = 4 → 
    prob_non_heart_king = 36 / 52 → prob_two_non_heart_king = (36 / 52) ^ 2 → 
    1 - prob_two_non_heart_king = 88 / 169 :=
by
  intros h_cards h_hearts h_kings h_prob_non_heart_king h_prob_two_non_heart_king
  sorry

end probability_heart_or_king_l312_312241


namespace sum_of_digits_l312_312006

variables {a b c d : ℕ}

theorem sum_of_digits (h1 : ∀ (x y z w : ℕ), x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w)
                      (h2 : c + a = 10)
                      (h3 : b + c = 9)
                      (h4 : a + d = 10) :
  a + b + c + d = 18 :=
sorry

end sum_of_digits_l312_312006


namespace tan_alpha_plus_beta_l312_312297

open Real

theorem tan_alpha_plus_beta (A alpha beta : ℝ) (h1 : sin alpha = A * sin (alpha + beta)) (h2 : abs A > 1) :
  tan (alpha + beta) = sin beta / (cos beta - A) :=
by
  sorry

end tan_alpha_plus_beta_l312_312297


namespace tire_price_l312_312094

theorem tire_price {p : ℤ} (h : 4 * p + 1 = 421) : p = 105 :=
sorry

end tire_price_l312_312094


namespace find_a_l312_312294

theorem find_a (a : ℝ) (h : ∃ x, x = 3 ∧ x^2 + a * x + a - 1 = 0) : a = -2 :=
sorry

end find_a_l312_312294


namespace scientific_notation_correct_l312_312936

-- Define the problem conditions
def original_number : ℝ := 6175700

-- Define the expected output in scientific notation
def scientific_notation_representation (x : ℝ) : Prop :=
  x = 6.1757 * 10^6

-- The theorem to prove
theorem scientific_notation_correct : scientific_notation_representation original_number :=
by sorry

end scientific_notation_correct_l312_312936


namespace other_divisor_l312_312571

theorem other_divisor (x : ℕ) (h1 : 261 % 37 = 2) (h2 : 261 % x = 2) (h3 : 259 = 261 - 2) :
  ∃ x : ℕ, 259 % 37 = 0 ∧ 259 % x = 0 ∧ x = 7 :=
by
  sorry

end other_divisor_l312_312571


namespace certain_event_white_balls_l312_312382

open Finset

theorem certain_event_white_balls :
  ∀ (box : Finset ℕ) (n : ℕ) (h_box : box.card = 5) 
  (h_all_white : ∀ x ∈ box, x < 5 ∧ x ≥ 0),
  ∃ (draw : Finset ℕ), draw.card = 2 ∧ draw ⊆ box :=
by
  sorry

end certain_event_white_balls_l312_312382


namespace series_proof_l312_312329

noncomputable def series_sum (a b : ℝ) : ℝ :=
  ∑' (n : ℕ), a / (b ^ (n + 1))

noncomputable def transformed_series_sum (a b : ℝ) : ℝ :=
  ∑' (n : ℕ), a / ((a + 2 * b) ^ (n + 1))

theorem series_proof (a b : ℝ)
  (h1 : series_sum a b = 7)
  (h2 : a = 7 * (b - 1)) :
  transformed_series_sum a b = 7 * (b - 1) / (9 * b - 8) :=
by sorry

end series_proof_l312_312329


namespace production_increase_l312_312678

theorem production_increase (h1 : ℝ) (h2 : ℝ) (h3 : h1 = 0.75) (h4 : h2 = 0.5) :
  (h1 + h2 - 1) = 0.25 := by
  sorry

end production_increase_l312_312678


namespace production_days_l312_312574

theorem production_days (n : ℕ) (P : ℕ)
  (h1 : P = 40 * n)
  (h2 : (P + 90) / (n + 1) = 45) :
  n = 9 :=
by
  sorry

end production_days_l312_312574


namespace not_p_suff_not_q_l312_312029

theorem not_p_suff_not_q (x : ℝ) :
  ¬(|x| ≥ 1) → ¬(x^2 + x - 6 ≥ 0) :=
sorry

end not_p_suff_not_q_l312_312029


namespace friend_decks_l312_312516

noncomputable def cost_per_deck : ℕ := 8
noncomputable def victor_decks : ℕ := 6
noncomputable def total_amount_spent : ℕ := 64

theorem friend_decks :
  ∃ x : ℕ, (victor_decks * cost_per_deck) + (x * cost_per_deck) = total_amount_spent ∧ x = 2 :=
by
  sorry

end friend_decks_l312_312516


namespace gift_card_value_l312_312268

def latte_cost : ℝ := 3.75
def croissant_cost : ℝ := 3.50
def daily_treat_cost : ℝ := latte_cost + croissant_cost
def weekly_treat_cost : ℝ := daily_treat_cost * 7

def cookie_cost : ℝ := 1.25
def total_cookie_cost : ℝ := cookie_cost * 5

def total_spent : ℝ := weekly_treat_cost + total_cookie_cost
def remaining_balance : ℝ := 43.00

theorem gift_card_value : (total_spent + remaining_balance) = 100 := 
by sorry

end gift_card_value_l312_312268


namespace compute_nested_f_l312_312330

def f(x : ℤ) : ℤ := x^2 - 4 * x + 3

theorem compute_nested_f : f (f (f (f (f (f 2))))) = f 1179395 := 
  sorry

end compute_nested_f_l312_312330


namespace sum_of_other_endpoint_coordinates_l312_312048

theorem sum_of_other_endpoint_coordinates :
  ∃ (x y: ℤ), (8 + x) / 2 = 6 ∧ y / 2 = -10 ∧ x + y = -16 :=
by
  sorry

end sum_of_other_endpoint_coordinates_l312_312048


namespace charles_whistles_l312_312643

theorem charles_whistles (S : ℕ) (C : ℕ) (h1 : S = 223) (h2 : S = C + 95) : C = 128 :=
by
  sorry

end charles_whistles_l312_312643


namespace perfect_square_polynomial_l312_312153

theorem perfect_square_polynomial (m : ℝ) :
  (∃ (f : ℝ → ℝ), ∀ x : ℝ, x^2 + 2*(m-3)*x + 25 = f x * f x) ↔ (m = 8 ∨ m = -2) :=
by
  sorry

end perfect_square_polynomial_l312_312153


namespace social_logistics_turnover_scientific_notation_l312_312013

noncomputable def total_social_logistics_turnover_2022 : ℝ := 347.6 * (10 ^ 12)

theorem social_logistics_turnover_scientific_notation :
  total_social_logistics_turnover_2022 = 3.476 * (10 ^ 14) :=
by
  sorry

end social_logistics_turnover_scientific_notation_l312_312013


namespace banana_cost_is_2_l312_312959

noncomputable def bananas_cost (B : ℝ) : Prop :=
  let cost_oranges : ℝ := 10 * 1.5
  let total_cost : ℝ := 25
  let cost_bananas : ℝ := total_cost - cost_oranges
  let num_bananas : ℝ := 5
  B = cost_bananas / num_bananas

theorem banana_cost_is_2 : bananas_cost 2 :=
by
  unfold bananas_cost
  sorry

end banana_cost_is_2_l312_312959


namespace polynomial_solution_l312_312750

noncomputable def p (x : ℝ) : ℝ := (7 / 4) * x^2 + 1

theorem polynomial_solution :
  (∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 2) ∧ p 2 = 8 :=
by
  sorry

end polynomial_solution_l312_312750


namespace points_per_correct_answer_l312_312613

theorem points_per_correct_answer (x : ℕ) : 
  let total_questions := 30
  let points_deducted_per_incorrect := 5
  let total_score := 325
  let correct_answers := 19
  let incorrect_answers := total_questions - correct_answers
  let points_lost_from_incorrect := incorrect_answers * points_deducted_per_incorrect
  let score_from_correct := correct_answers * x
  (score_from_correct - points_lost_from_incorrect = total_score) → x = 20 :=
by {
  sorry
}

end points_per_correct_answer_l312_312613


namespace find_x_l312_312223

theorem find_x (x y : ℕ) 
  (h1 : 3^x * 4^y = 59049) 
  (h2 : x - y = 10) : 
  x = 10 := 
by 
  sorry

end find_x_l312_312223


namespace jenny_change_l312_312320

/-!
## Problem statement

Jenny is printing 7 copies of her 25-page essay. It costs $0.10 to print one page.
She also buys 7 pens, each costing $1.50. If she pays with $40, calculate the change she should get.
-/

def cost_per_page : ℝ := 0.10
def pages_per_copy : ℕ := 25
def num_copies : ℕ := 7
def cost_per_pen : ℝ := 1.50
def num_pens : ℕ := 7
def amount_paid : ℝ := 40.0

def total_pages : ℕ := num_copies * pages_per_copy

def cost_printing : ℝ := total_pages * cost_per_page
def cost_pens : ℝ := num_pens * cost_per_pen

def total_cost : ℝ := cost_printing + cost_pens

theorem jenny_change : amount_paid - total_cost = 12 := by
  -- proof here
  sorry

end jenny_change_l312_312320


namespace together_finish_work_in_10_days_l312_312529

theorem together_finish_work_in_10_days (x_days y_days : ℕ) (hx : x_days = 15) (hy : y_days = 30) :
  let x_rate := 1 / (x_days : ℚ)
  let y_rate := 1 / (y_days : ℚ)
  let combined_rate := x_rate + y_rate
  let total_days := 1 / combined_rate
  total_days = 10 :=
by
  sorry

end together_finish_work_in_10_days_l312_312529


namespace probability_sum_of_dice_eq_3_l312_312858

theorem probability_sum_of_dice_eq_3 : 
  ∀ (a b c : ℕ), (1 ≤ a ∧ a ≤ 6) → (1 ≤ b ∧ b ≤ 6) → (1 ≤ c ∧ c ≤ 6) → 
  (a + b + c = 3) → 
  prob_event (λ (a b c : ℕ), a + b + c = 3) = 1 / 216 :=
by
  sorry

end probability_sum_of_dice_eq_3_l312_312858


namespace line_circle_separate_l312_312186

def point_inside_circle (x0 y0 a : ℝ) : Prop :=
  x0^2 + y0^2 < a^2

def not_center_of_circle (x0 y0 : ℝ) : Prop :=
  x0^2 + y0^2 ≠ 0

theorem line_circle_separate (x0 y0 a : ℝ) (h1 : point_inside_circle x0 y0 a) (h2 : a > 0) (h3 : not_center_of_circle x0 y0) :
  ∀ (x y : ℝ), ¬ (x0 * x + y0 * y = a^2 ∧ x^2 + y^2 = a^2) :=
by
  sorry

end line_circle_separate_l312_312186


namespace solve_for_x_l312_312154

theorem solve_for_x (x : ℝ) (h : 3 * x - 8 = 4 * x + 5) : x = -13 :=
by 
  sorry

end solve_for_x_l312_312154


namespace proof_find_C_proof_find_cos_A_l312_312575

noncomputable def find_C {a b c : ℝ} {B : ℝ} (h1 : 2 * c * (Real.cos B) = 2 * a - Real.sqrt 3 * b) : Prop :=
  ∃ (C : ℝ), 0 < C ∧ C < Real.pi ∧ C = Real.pi / 6

noncomputable def find_cos_A {a b c : ℝ} {B : ℝ} (h1 : 2 * c * (Real.cos B) = 2 * a - Real.sqrt 3 * b) (h2 : Real.cos B = 2 / 3) : Prop :=
  ∃ (A : ℝ), Real.cos A = (Real.sqrt 5 - 2 * Real.sqrt 3) / 6

theorem proof_find_C (a b c B : ℝ) (h1 : 2 * c * (Real.cos B) = 2 * a - Real.sqrt 3 * b) : find_C h1 :=
  sorry

theorem proof_find_cos_A (a b c B : ℝ) (h1 : 2 * c * (Real.cos B) = 2 * a - Real.sqrt 3 * b) (h2 : Real.cos B = 2 / 3) : find_cos_A h1 h2 :=
  sorry

end proof_find_C_proof_find_cos_A_l312_312575


namespace multiplicative_inverse_exists_and_is_correct_l312_312333

theorem multiplicative_inverse_exists_and_is_correct :
  ∃ N : ℤ, N > 0 ∧ (123456 * 171717) * N % 1000003 = 1 :=
sorry

end multiplicative_inverse_exists_and_is_correct_l312_312333


namespace probability_at_least_one_heart_or_king_l312_312238
   
   noncomputable def probability_non_favorable : ℚ := 81 / 169

   theorem probability_at_least_one_heart_or_king :
     1 - probability_non_favorable = 88 / 169 := 
   sorry
   
end probability_at_least_one_heart_or_king_l312_312238


namespace real_number_solution_pure_imaginary_solution_zero_solution_l312_312753

noncomputable def real_number_condition (m : ℝ) : Prop :=
  m^2 - 3 * m + 2 = 0

noncomputable def pure_imaginary_condition (m : ℝ) : Prop :=
  (2 * m^2 - 3 * m - 2 = 0) ∧ ¬(m^2 - 3 * m + 2 = 0)

noncomputable def zero_condition (m : ℝ) : Prop :=
  (2 * m^2 - 3 * m - 2 = 0) ∧ (m^2 - 3 * m + 2 = 0)

theorem real_number_solution (m : ℝ) : real_number_condition m ↔ (m = 1 ∨ m = 2) := 
sorry

theorem pure_imaginary_solution (m : ℝ) : pure_imaginary_condition m ↔ (m = -1 / 2) :=
sorry

theorem zero_solution (m : ℝ) : zero_condition m ↔ (m = 2) :=
sorry

end real_number_solution_pure_imaginary_solution_zero_solution_l312_312753


namespace factorization_of_expression_l312_312893

theorem factorization_of_expression (m : ℝ) : 2 * m^2 - 18 = 2 * (m + 3) * (m - 3) :=
by 
  sorry

end factorization_of_expression_l312_312893


namespace slope_of_line_l312_312218

theorem slope_of_line {x y : ℝ} (hx : x ≠ 0) (hy : y ≠ 0) (h : 5 / x + 4 / y = 0) :
  ∃ x₁ x₂ y₁ y₂, (5 / x₁ + 4 / y₁ = 0) ∧ (5 / x₂ + 4 / y₂ = 0) ∧ 
  (y₂ - y₁) / (x₂ - x₁) = -4 / 5 :=
sorry

end slope_of_line_l312_312218


namespace consultation_session_probability_l312_312249

noncomputable def consultation_probability : ℝ :=
  let volume_cube := 3 * 3 * 3
  let volume_valid := 9 - 2 * (1/3 * 2.25 * 1.5)
  volume_valid / volume_cube

theorem consultation_session_probability : consultation_probability = 1 / 4 :=
by
  sorry

end consultation_session_probability_l312_312249


namespace volume_of_tetrahedron_l312_312567

theorem volume_of_tetrahedron 
(angle_ABC_BCD : Real := 45 * Real.pi / 180)
(area_ABC : Real := 150)
(area_BCD : Real := 90)
(length_BC : Real := 10) :
  let h := 2 * area_BCD / length_BC
  let height_perpendicular := h * Real.sin angle_ABC_BCD
  let volume := (1 / 3 : Real) * area_ABC * height_perpendicular
  volume = 450 * Real.sqrt 2 :=
by
  sorry

end volume_of_tetrahedron_l312_312567


namespace vertical_asymptote_sum_l312_312415

theorem vertical_asymptote_sum :
  ∀ x y : ℝ, (4 * x^2 + 8 * x + 3 = 0) → (4 * y^2 + 8 * y + 3 = 0) → x ≠ y → x + y = -2 :=
by
  sorry

end vertical_asymptote_sum_l312_312415


namespace Marcus_walking_speed_l312_312338

def bath_time : ℕ := 20  -- in minutes
def blow_dry_time : ℕ := bath_time / 2  -- in minutes
def trail_distance : ℝ := 3  -- in miles
def total_dog_time : ℕ := 60  -- in minutes

theorem Marcus_walking_speed :
  let walking_time := total_dog_time - (bath_time + blow_dry_time)
  let walking_time_hours := (walking_time:ℝ) / 60
  (trail_distance / walking_time_hours) = 6 := by
  sorry

end Marcus_walking_speed_l312_312338


namespace max_m_value_real_roots_interval_l312_312356

theorem max_m_value_real_roots_interval :
  (∃ x ∈ (Set.Icc 0 1), x^3 - 3 * x - m = 0) → m ≤ 0 :=
by
  sorry 

end max_m_value_real_roots_interval_l312_312356


namespace find_natural_number_n_l312_312749

def is_terminating_decimal (x : ℚ) : Prop :=
  ∃ (a b : ℕ), x = (a / b) ∧ (∃ (m n : ℕ), b = 2 ^ m * 5 ^ n)

theorem find_natural_number_n (n : ℕ) (h₁ : is_terminating_decimal (1 / n)) (h₂ : is_terminating_decimal (1 / (n + 1))) : n = 4 :=
by sorry

end find_natural_number_n_l312_312749


namespace square_field_area_l312_312042

noncomputable def area_of_square_field(speed_kph : ℝ) (time_hrs : ℝ) : ℝ :=
  let speed_mps := (speed_kph * 1000) / 3600
  let distance := speed_mps * (time_hrs * 3600)
  let side_length := distance / Real.sqrt 2
  side_length ^ 2

theorem square_field_area 
  (speed_kph : ℝ := 2.4)
  (time_hrs : ℝ := 3.0004166666666667) :
  area_of_square_field speed_kph time_hrs = 25939764.41 := 
by 
  -- This is a placeholder for the proof. 
  sorry

end square_field_area_l312_312042


namespace flowers_per_bouquet_l312_312027

noncomputable def num_flowers_per_bouquet (total_flowers wilted_flowers bouquets : ℕ) : ℕ :=
  (total_flowers - wilted_flowers) / bouquets

theorem flowers_per_bouquet : num_flowers_per_bouquet 53 18 5 = 7 := by
  sorry

end flowers_per_bouquet_l312_312027


namespace number_of_bowls_l312_312829

theorem number_of_bowls (n : ℕ) (h1 : 12 * 8 = 96) (h2 : 6 * n = 96) : n = 16 :=
by
  -- equations from the conditions
  have h3 : 96 = 96 := by sorry
  exact sorry

end number_of_bowls_l312_312829


namespace value_to_subtract_l312_312158

variable (x y : ℝ)

theorem value_to_subtract (h1 : (x - 5) / 7 = 7) (h2 : (x - y) / 8 = 6) : y = 6 := by
  sorry

end value_to_subtract_l312_312158


namespace no_valid_base_l312_312561

theorem no_valid_base (b : ℤ) (n : ℤ) : b^2 + 2*b + 2 ≠ n^2 := by
  sorry

end no_valid_base_l312_312561


namespace problem_a_problem_b_unique_solution_l312_312524

-- Problem (a)

theorem problem_a (a b c n : ℤ) (hnat : 0 ≤ n) (h : a * n^2 + b * n + c = 0) : n ∣ c :=
sorry

-- Problem (b)

theorem problem_b_unique_solution : ∀ n : ℕ, n^5 - 2 * n^4 - 7 * n^2 - 7 * n + 3 = 0 → n = 3 :=
sorry

end problem_a_problem_b_unique_solution_l312_312524


namespace plane_equation_l312_312251

theorem plane_equation 
  (s t : ℝ)
  (x y z : ℝ)
  (parametric_plane : ℝ → ℝ → ℝ × ℝ × ℝ)
  (plane_eq : ℝ × ℝ × ℝ → Prop) :
  parametric_plane s t = (2 + 2 * s - t, 1 + 2 * s, 4 - 3 * s + t) →
  plane_eq (x, y, z) ↔ 2 * x - 5 * y + 2 * z - 7 = 0 :=
by
  sorry

end plane_equation_l312_312251


namespace smallest_angle_measure_l312_312508

-- Define the conditions
def is_spherical_triangle (a b c : ℝ) : Prop :=
  a + b + c > 180 ∧ a + b + c < 540

def angles (k : ℝ) : Prop :=
  let a := 3 * k
  let b := 4 * k
  let c := 5 * k
  is_spherical_triangle a b c ∧ a + b + c = 270

-- Statement of the theorem
theorem smallest_angle_measure (k : ℝ) (h : angles k) : 3 * k = 67.5 :=
sorry

end smallest_angle_measure_l312_312508


namespace intersection_A_B_l312_312130

def A := {y : ℝ | ∃ x : ℝ, y = 2^x}
def B := {y : ℝ | ∃ x : ℝ, y = -x^2 + 2}
def Intersection := {y : ℝ | 0 < y ∧ y ≤ 2}

theorem intersection_A_B :
  (A ∩ B) = Intersection :=
by
  sorry

end intersection_A_B_l312_312130


namespace total_colored_hangers_l312_312605

theorem total_colored_hangers (pink green : ℕ) :
  pink = 7 →
  green = 4 →
  let blue := green - 1 in
  let yellow := blue - 1 in
  pink + green + blue + yellow = 16 :=
by
  intros hp hg
  let blue := green - 1
  let yellow := blue - 1
  sorry

end total_colored_hangers_l312_312605


namespace evaluate_exp_power_l312_312720

theorem evaluate_exp_power : (3^3)^2 = 729 := 
by {
  sorry
}

end evaluate_exp_power_l312_312720


namespace function_parallel_l312_312190

theorem function_parallel {x y : ℝ} (h : y = -2 * x + 1) : 
    ∀ {a : ℝ}, y = -2 * a + 3 -> y = -2 * x + 1 := by
    sorry

end function_parallel_l312_312190


namespace fenced_yard_area_l312_312246

theorem fenced_yard_area :
  let yard := 20 * 18
  let cutout1 := 3 * 3
  let cutout2 := 4 * 2
  yard - cutout1 - cutout2 = 343 := by
  let yard := 20 * 18
  let cutout1 := 3 * 3
  let cutout2 := 4 * 2
  have h : yard - cutout1 - cutout2 = 343 := sorry
  exact h

end fenced_yard_area_l312_312246


namespace number_of_bowls_l312_312835

theorem number_of_bowls (n : ℕ) (h1 : 12 * 8 = 96) (h2 : ∀ t : ℕ, t = 6 * n -> t = 96) : n = 16 := by
  sorry

end number_of_bowls_l312_312835


namespace cookie_total_l312_312938

-- Definitions of the conditions
def rows_large := 5
def rows_medium := 4
def rows_small := 6
def cookies_per_row_large := 6
def cookies_per_row_medium := 7
def cookies_per_row_small := 8
def number_of_trays := 4
def extra_row_large_first_tray := 1
def total_large_cookies := rows_large * cookies_per_row_large * number_of_trays + extra_row_large_first_tray * cookies_per_row_large
def total_medium_cookies := rows_medium * cookies_per_row_medium * number_of_trays
def total_small_cookies := rows_small * cookies_per_row_small * number_of_trays

-- Theorem to prove the total number of cookies is 430
theorem cookie_total : 
  total_large_cookies + total_medium_cookies + total_small_cookies = 430 :=
by
  -- Proof is omitted
  sorry

end cookie_total_l312_312938


namespace root_value_l312_312157

theorem root_value (a : ℝ) (h: 3 * a^2 - 4 * a + 1 = 0) : 6 * a^2 - 8 * a + 5 = 3 := 
by 
  sorry

end root_value_l312_312157


namespace probability_heart_or_king_l312_312233

theorem probability_heart_or_king :
  let total_cards := 52
  let hearts := 13
  let kings := 4
  let overlap := 1
  let unique_hearts_or_kings := hearts + kings - overlap
  let non_hearts_or_kings := total_cards - unique_hearts_or_kings
  let p_non_heart_or_king := (non_hearts_or_kings : ℚ) / (total_cards : ℚ)
  let p_non_heart_or_king_twice := p_non_heart_or_king * p_non_heart_or_king
  let p_at_least_one_heart_or_king := 1 - p_non_heart_or_king_twice
  p_at_least_one_heart_or_king = 88 / 169 :=
by
  have total_cards := 52
  have hearts := 13
  have kings := 4
  have overlap := 1
  have unique_hearts_or_kings := hearts + kings - overlap
  have non_hearts_or_kings := total_cards - unique_hearts_or_kings
  have p_non_heart_or_king := (non_hearts_or_kings : ℚ) / (total_cards : ℚ)
  have p_non_heart_or_king_twice := p_non_heart_or_king * p_non_heart_or_king
  have p_at_least_one_heart_or_king := 1 - p_non_heart_or_king_twice
  show p_at_least_one_heart_or_king = 88 / 169
  sorry

end probability_heart_or_king_l312_312233


namespace jason_nickels_is_52_l312_312350

theorem jason_nickels_is_52 (n q : ℕ) (h1 : 5 * n + 10 * q = 680) (h2 : q = n - 10) : n = 52 :=
sorry

end jason_nickels_is_52_l312_312350


namespace length_of_FD_l312_312616

/-- In a square of side length 8 cm, point E is located on side AD,
2 cm from A and 6 cm from D. Point F lies on side CD such that folding
the square so that C coincides with E creates a crease along GF. 
Prove that the length of segment FD is 7/4 cm. -/
theorem length_of_FD (x : ℝ) (h_square : ∀ (A B C D : ℝ), A = 8 ∧ B = 8 ∧ C = 8 ∧ D = 8)
    (h_AE : ∀ (A E : ℝ), A - E = 2) (h_ED : ∀ (E D : ℝ), E - D = 6)
    (h_pythagorean : ∀ (x : ℝ), (8 - x)^2 = x^2 + 6^2) : x = 7/4 :=
by
  sorry

end length_of_FD_l312_312616


namespace max_entanglements_l312_312534

theorem max_entanglements (a b : ℕ) (h1 : a < b) (h2 : a < 1000) (h3 : b < 1000) :
  ∃ n ≤ 9, ∀ k, k ≤ n → ∃ a' b' : ℕ, (b' - a' = b - a - 2^k) :=
by sorry

end max_entanglements_l312_312534


namespace election_votes_total_l312_312612

theorem election_votes_total 
  (winner_votes : ℕ) (opponent1_votes opponent2_votes opponent3_votes : ℕ)
  (excess1 excess2 excess3 : ℕ)
  (h1 : winner_votes = opponent1_votes + excess1)
  (h2 : winner_votes = opponent2_votes + excess2)
  (h3 : winner_votes = opponent3_votes + excess3)
  (votes_winner : winner_votes = 195)
  (votes_opponent1 : opponent1_votes = 142)
  (votes_opponent2 : opponent2_votes = 116)
  (votes_opponent3 : opponent3_votes = 90)
  (he1 : excess1 = 53)
  (he2 : excess2 = 79)
  (he3 : excess3 = 105) :
  winner_votes + opponent1_votes + opponent2_votes + opponent3_votes = 543 :=
by sorry

end election_votes_total_l312_312612


namespace businessmen_drink_one_type_l312_312102

def total_businessmen : ℕ := 35
def coffee_drinkers : ℕ := 18
def tea_drinkers : ℕ := 15
def juice_drinkers : ℕ := 8
def coffee_and_tea_drinkers : ℕ := 6
def tea_and_juice_drinkers : ℕ := 4
def coffee_and_juice_drinkers : ℕ := 3
def all_three_drinkers : ℕ := 2

theorem businessmen_drink_one_type : 
  coffee_drinkers - coffee_and_tea_drinkers - coffee_and_juice_drinkers + all_three_drinkers +
  tea_drinkers - coffee_and_tea_drinkers - tea_and_juice_drinkers + all_three_drinkers +
  juice_drinkers - tea_and_juice_drinkers - coffee_and_juice_drinkers + all_three_drinkers = 21 := 
sorry

end businessmen_drink_one_type_l312_312102


namespace monotonic_increasing_range_l312_312304

noncomputable def f (x a : ℝ) : ℝ := (Real.exp x) * (x + a) / x

theorem monotonic_increasing_range (a : ℝ) :
  (∀ x : ℝ, x > 0 → (∀ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x1 < x2 → f x1 a ≤ f x2 a)) ↔ -4 ≤ a ∧ a ≤ 0 :=
sorry

end monotonic_increasing_range_l312_312304


namespace find_multiple_l312_312405

-- Definitions and given conditions
def total_seats : ℤ := 387
def first_class_seats : ℤ := 77

-- The statement we need to prove
theorem find_multiple (m : ℤ) :
  (total_seats = first_class_seats + (m * first_class_seats + 2)) → m = 4 :=
by
  sorry

end find_multiple_l312_312405


namespace total_carpets_l312_312052

theorem total_carpets 
(house1 : ℕ) 
(house2 : ℕ) 
(house3 : ℕ) 
(house4 : ℕ) 
(h1 : house1 = 12) 
(h2 : house2 = 20) 
(h3 : house3 = 10) 
(h4 : house4 = 2 * house3) : 
  house1 + house2 + house3 + house4 = 62 := 
by 
  -- The proof will be inserted here
  sorry

end total_carpets_l312_312052


namespace probability_heart_or_king_l312_312240

theorem probability_heart_or_king (cards hearts kings : ℕ) (prob_non_heart_king : ℚ) 
    (prob_two_non_heart_king : ℚ) : 
    cards = 52 → hearts = 13 → kings = 4 → 
    prob_non_heart_king = 36 / 52 → prob_two_non_heart_king = (36 / 52) ^ 2 → 
    1 - prob_two_non_heart_king = 88 / 169 :=
by
  intros h_cards h_hearts h_kings h_prob_non_heart_king h_prob_two_non_heart_king
  sorry

end probability_heart_or_king_l312_312240


namespace width_of_smaller_cuboids_is_4_l312_312152

def length_smaller_cuboid := 5
def height_smaller_cuboid := 3
def length_larger_cuboid := 16
def width_larger_cuboid := 10
def height_larger_cuboid := 12
def num_smaller_cuboids := 32

theorem width_of_smaller_cuboids_is_4 :
  ∃ W : ℝ, W = 4 ∧ (length_smaller_cuboid * W * height_smaller_cuboid) * num_smaller_cuboids = 
            length_larger_cuboid * width_larger_cuboid * height_larger_cuboid :=
by
  sorry

end width_of_smaller_cuboids_is_4_l312_312152


namespace isosceles_triangle_solution_l312_312788

noncomputable def isosceles_sides : (a b : ℝ) → (a = 25) ∧ (b = 10) → (sides : list ℝ)
| a b ⟨h₁, h₂⟩ := [a, a, b]

theorem isosceles_triangle_solution
(perimeter : ℝ) (iso_condition : ∃ a b, a = 25 ∧ b = 10 ∧ 2 * a + b = perimeter)
(intersect_uninscribed_boundary : ∀ (O M : ℝ) (ratio : ℝ), O = 2 / 3 * M) : 
sides 
       ∃ a b, (a = 25) ∧ (b = 10) ∧ (perimeter = 60)  :=
begin
  sorry
end

end isosceles_triangle_solution_l312_312788


namespace sum_local_values_2345_l312_312372

theorem sum_local_values_2345 : 
  let n := 2345
  let digit_2_value := 2000
  let digit_3_value := 300
  let digit_4_value := 40
  let digit_5_value := 5
  digit_2_value + digit_3_value + digit_4_value + digit_5_value = n := 
by
  sorry

end sum_local_values_2345_l312_312372


namespace probability_card_10_and_spade_l312_312369

theorem probability_card_10_and_spade : 
  let deck_size := 52 
  let num_10s := 4 
  let num_spades := 13
  let first_card_10_prob := num_10s / deck_size
  let second_card_spade_prob_after_10 := (num_spades / (deck_size - 1))
  let total_prob := first_card_10_prob * second_card_spade_prob_after_10
  total_prob = 12 / 663 :=
by
  sorry

end probability_card_10_and_spade_l312_312369


namespace geometric_sequence_a1_l312_312302

theorem geometric_sequence_a1 (a1 a2 a3 S3 : ℝ) (q : ℝ)
  (h1 : S3 = a1 + (1 / 2) * a2)
  (h2 : a3 = (1 / 4))
  (h3 : S3 = a1 * (1 + q + q^2))
  (h4 : a2 = a1 * q)
  (h5 : a3 = a1 * q^2) :
  a1 = 1 :=
sorry

end geometric_sequence_a1_l312_312302


namespace inequality_proof_l312_312178

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) : 
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l312_312178


namespace arithmetic_sequence_sum_l312_312137

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h0 : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2) 
  (h1 : S 10 = 12)
  (h2 : S 20 = 17) :
  S 30 = 15 := by
  sorry

end arithmetic_sequence_sum_l312_312137


namespace primes_div_order_l312_312632

theorem primes_div_order (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h : q ∣ 3^p - 2^p) : p ∣ q - 1 :=
sorry

end primes_div_order_l312_312632


namespace average_speed_to_SF_l312_312232

theorem average_speed_to_SF (v d : ℝ) (h1 : d ≠ 0) (h2 : v ≠ 0) :
  (2 * d / ((d / v) + (2 * d / v)) = 34) → v = 51 :=
by
  -- proof goes here
  sorry

end average_speed_to_SF_l312_312232


namespace x_sq_plus_3x_eq_1_l312_312770

theorem x_sq_plus_3x_eq_1 (x : ℝ) (h : (x^2 + 3*x)^2 + 2*(x^2 + 3*x) - 3 = 0) : x^2 + 3*x = 1 :=
sorry

end x_sq_plus_3x_eq_1_l312_312770


namespace vertical_angles_always_equal_l312_312068

theorem vertical_angles_always_equal (a b : ℝ) (h : a = b) : 
  (∀ θ1 θ2, θ1 + θ2 = 180 ∧ θ1 = a ∧ θ2 = b → θ1 = θ2) :=
by 
  intro θ1 θ2 
  intro h 
  sorry

end vertical_angles_always_equal_l312_312068


namespace probability_heart_or_king_l312_312235

theorem probability_heart_or_king :
  let total_cards := 52
  let hearts := 13
  let kings := 4
  let overlap := 1
  let unique_hearts_or_kings := hearts + kings - overlap
  let non_hearts_or_kings := total_cards - unique_hearts_or_kings
  let p_non_heart_or_king := (non_hearts_or_kings : ℚ) / (total_cards : ℚ)
  let p_non_heart_or_king_twice := p_non_heart_or_king * p_non_heart_or_king
  let p_at_least_one_heart_or_king := 1 - p_non_heart_or_king_twice
  p_at_least_one_heart_or_king = 88 / 169 :=
by
  have total_cards := 52
  have hearts := 13
  have kings := 4
  have overlap := 1
  have unique_hearts_or_kings := hearts + kings - overlap
  have non_hearts_or_kings := total_cards - unique_hearts_or_kings
  have p_non_heart_or_king := (non_hearts_or_kings : ℚ) / (total_cards : ℚ)
  have p_non_heart_or_king_twice := p_non_heart_or_king * p_non_heart_or_king
  have p_at_least_one_heart_or_king := 1 - p_non_heart_or_king_twice
  show p_at_least_one_heart_or_king = 88 / 169
  sorry

end probability_heart_or_king_l312_312235


namespace value_of_a2_l312_312779

theorem value_of_a2 (a0 a1 a2 a3 a4 : ℝ) (x : ℝ) 
  (h : x^4 = a0 + a1 * (x - 2) + a2 * (x - 2)^2 + a3 * (x - 2)^3 + a4 * (x - 2)^4) :
  a2 = 24 :=
sorry

end value_of_a2_l312_312779


namespace find_x_l312_312062

variables (a b c d x y : ℚ)

noncomputable def modified_fraction (a b x y : ℚ) := (a + x) / (b + y)

theorem find_x (h1 : a ≠ b) (h2 : b ≠ 0) (h3 : modified_fraction a b x y = c / d) :
  x = (b * c - a * d + y * c) / d :=
by
  sorry

end find_x_l312_312062


namespace cost_of_orange_juice_l312_312888

theorem cost_of_orange_juice (total_money : ℕ) (bread_qty : ℕ) (orange_qty : ℕ) 
  (bread_cost : ℕ) (money_left : ℕ) (total_spent : ℕ) (orange_cost : ℕ) 
  (h1 : total_money = 86) (h2 : bread_qty = 3) (h3 : orange_qty = 3) 
  (h4 : bread_cost = 3) (h5 : money_left = 59) :
  (total_money - money_left - bread_qty * bread_cost) / orange_qty = 6 :=
by
  have h6 : total_spent = total_money - money_left := by sorry
  have h7 : total_spent - bread_qty * bread_cost = orange_qty * orange_cost := by sorry
  have h8 : orange_cost = 6 := by sorry
  exact sorry

end cost_of_orange_juice_l312_312888


namespace combined_work_time_l312_312075

noncomputable def work_time_first_worker : ℤ := 5
noncomputable def work_time_second_worker : ℤ := 4

theorem combined_work_time :
  (1 / (1 / work_time_first_worker + 1 / work_time_second_worker)) = 20 / 9 :=
by
  unfold work_time_first_worker work_time_second_worker
  -- The detailed reasoning and computation would go here
  sorry

end combined_work_time_l312_312075


namespace polygon_sides_l312_312926

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 > 2970) :
  n = 19 :=
by
  sorry

end polygon_sides_l312_312926


namespace find_a_l312_312631

theorem find_a (x y z a : ℝ) (h1 : ∃ k : ℝ, x = 3 * k ∧ y = 4 * k ∧ z = 7 * k) 
              (h2 : x + y + z = 70) 
              (h3 : y = 15 * a - 5) : 
  a = 5 / 3 := 
by sorry

end find_a_l312_312631


namespace quad_inequality_necessary_but_not_sufficient_l312_312226

def quad_inequality (x : ℝ) : Prop := x^2 - x - 6 > 0
def less_than_negative_five (x : ℝ) : Prop := x < -5

theorem quad_inequality_necessary_but_not_sufficient :
  (∀ x : ℝ, less_than_negative_five x → quad_inequality x) ∧ 
  (∃ x : ℝ, quad_inequality x ∧ ¬ less_than_negative_five x) :=
by
  sorry

end quad_inequality_necessary_but_not_sufficient_l312_312226


namespace age_solution_l312_312513

noncomputable def age_problem : Prop :=
  ∃ (m s x : ℕ),
  (m - 3 = 2 * (s - 3)) ∧
  (m - 5 = 3 * (s - 5)) ∧
  (m + x) * 2 = 3 * (s + x) ∧
  x = 1

theorem age_solution : age_problem :=
  by
    sorry

end age_solution_l312_312513


namespace find_number_l312_312192

-- Assume the necessary definitions and conditions
variable (x : ℝ)

-- Sixty-five percent of the number is 21 less than four-fifths of the number
def condition := 0.65 * x = 0.8 * x - 21

-- Final proof goal: We need to prove that the number x is 140
theorem find_number (h : condition x) : x = 140 := by
  sorry

end find_number_l312_312192


namespace triangle_angle_properties_l312_312614

theorem triangle_angle_properties
  (a b : ℕ)
  (h₁ : a = 45)
  (h₂ : b = 70) :
  ∃ (c : ℕ), a + b + c = 180 ∧ c = 65 ∧ max (max a b) c = 70 := by
  sorry

end triangle_angle_properties_l312_312614


namespace number_of_bowls_l312_312828

theorem number_of_bowls (n : ℕ) (h1 : 12 * 8 = 96) (h2 : 6 * n = 96) : n = 16 :=
by
  -- equations from the conditions
  have h3 : 96 = 96 := by sorry
  exact sorry

end number_of_bowls_l312_312828


namespace point_relationship_l312_312483

def quadratic_function (x : ℝ) (c : ℝ) : ℝ :=
  -(x - 1) ^ 2 + c

noncomputable def y1_def (c : ℝ) : ℝ := quadratic_function (-3) c
noncomputable def y2_def (c : ℝ) : ℝ := quadratic_function (-1) c
noncomputable def y3_def (c : ℝ) : ℝ := quadratic_function 5 c

theorem point_relationship (c : ℝ) :
  y2_def c > y1_def c ∧ y1_def c = y3_def c :=
by
  sorry

end point_relationship_l312_312483


namespace at_least_one_neg_l312_312371

theorem at_least_one_neg (a b c d : ℝ) (h1 : a + b = 1) (h2 : c + d = 1) (h3 : ac + bd > 1) : 
  a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0 :=
sorry

end at_least_one_neg_l312_312371


namespace grandfather_7_times_older_after_8_years_l312_312538

theorem grandfather_7_times_older_after_8_years :
  ∃ x : ℕ, ∀ (g_age ng_age : ℕ), 50 < g_age ∧ g_age < 90 ∧ g_age = 31 * ng_age → g_age + x = 7 * (ng_age + x) → x = 8 :=
by
  sorry

end grandfather_7_times_older_after_8_years_l312_312538


namespace find_t_l312_312947

-- Define the utility function based on hours of reading and playing basketball
def utility (reading_hours : ℝ) (basketball_hours : ℝ) : ℝ :=
  reading_hours * basketball_hours

-- Define the conditions for Wednesday and Thursday utilities
def wednesday_utility (t : ℝ) : ℝ :=
  t * (10 - t)

def thursday_utility (t : ℝ) : ℝ :=
  (3 - t) * (t + 4)

-- The main theorem stating the equivalence of utilities implies t = 3
theorem find_t (t : ℝ) (h : wednesday_utility t = thursday_utility t) : t = 3 :=
by
  -- Skip proof with sorry
  sorry

end find_t_l312_312947


namespace solve_quadratics_l312_312193

theorem solve_quadratics (x : ℝ) :
  (x^2 - 7 * x - 18 = 0 → x = 9 ∨ x = -2) ∧
  (4 * x^2 + 1 = 4 * x → x = 1/2) :=
by
  sorry

end solve_quadratics_l312_312193


namespace words_per_page_l312_312391

theorem words_per_page (p : ℕ) (h1 : 150 * p ≡ 210 [MOD 221]) (h2 : p ≤ 120) : p = 195 := by
  sorry

end words_per_page_l312_312391


namespace probability_one_of_two_sheep_selected_l312_312972

theorem probability_one_of_two_sheep_selected :
  let sheep := ["Happy", "Pretty", "Lazy", "Warm", "Boiling"]
  let total_ways := Nat.choose 5 2
  let favorable_ways := (Nat.choose 2 1) * (Nat.choose 3 1)
  let probability := favorable_ways / total_ways
  probability = 3 / 5 :=
by
  let sheep := ["Happy", "Pretty", "Lazy", "Warm", "Boiling"]
  let total_ways := Nat.choose 5 2
  let favorable_ways := (Nat.choose 2 1) * (Nat.choose 3 1)
  let probability := favorable_ways / total_ways
  sorry

end probability_one_of_two_sheep_selected_l312_312972


namespace perpendicular_line_through_P_l312_312963

open Real

/-- Define the point P as (-1, 3) -/
def P : ℝ × ℝ := (-1, 3)

/-- Define the line equation -/
def line1 (x y : ℝ) : Prop := x + 2 * y - 3 = 0

/-- Define the perpendicular line equation -/
def perpendicular_line (x y : ℝ) : Prop := 2 * x - y + 5 = 0

/-- The theorem stating that P lies on the perpendicular line to the given line -/
theorem perpendicular_line_through_P : ∀ x y, P = (x, y) → line1 x y → perpendicular_line x y :=
by
  sorry

end perpendicular_line_through_P_l312_312963


namespace find_smallest_n_l312_312286

open Matrix Complex

noncomputable def rotation_matrix := ![
  ![Real.sqrt 2 / 2, -Real.sqrt 2 / 2],
  ![Real.sqrt 2 / 2, Real.sqrt 2 / 2]
]

def I_2 := (1 : Matrix (Fin 2) (Fin 2) ℝ)

theorem find_smallest_n (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (hA : A = rotation_matrix) : 
  ∃ (n : ℕ), 0 < n ∧ A ^ n = I_2 ∧ ∀ m : ℕ, 0 < m ∧ m < n → A ^ m ≠ I_2 :=
by {
  sorry
}

end find_smallest_n_l312_312286


namespace apples_left_is_correct_l312_312595

-- Definitions for the conditions
def blue_apples : ℕ := 5
def yellow_apples : ℕ := 2 * blue_apples
def total_apples : ℕ := blue_apples + yellow_apples
def apples_given_to_son : ℚ := 1 / 5 * total_apples
def apples_left : ℚ := total_apples - apples_given_to_son

-- The main statement to be proven
theorem apples_left_is_correct : apples_left = 12 := by
  sorry

end apples_left_is_correct_l312_312595


namespace convert_base_7_to_base_10_l312_312929

theorem convert_base_7_to_base_10 (n : ℕ) (h : n = 6 * 7^2 + 5 * 7^1 + 3 * 7^0) : n = 332 := by
  sorry

end convert_base_7_to_base_10_l312_312929


namespace value_of_f_at_7_l312_312300

theorem value_of_f_at_7
  (f : ℝ → ℝ)
  (h_even : ∀ x, f x = f (-x))
  (h_periodic : ∀ x, f (x + 4) = f x)
  (h_definition : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2) :
  f 7 = 2 :=
by
  -- Proof will be filled here
  sorry

end value_of_f_at_7_l312_312300


namespace inequality_solution_l312_312194

theorem inequality_solution :
  {x : ℝ | (x - 1) * (x - 4) * (x - 5)^2 / ((x - 3) * (x^2 - 9)) > 0} = { x : ℝ | -3 < x ∧ x < 3 } :=
sorry

end inequality_solution_l312_312194


namespace number_of_bowls_l312_312849

noncomputable theory
open Classical

theorem number_of_bowls (n : ℕ) 
  (h1 : 8 * 12 = 6 * n) : n = 16 := 
by
  sorry

end number_of_bowls_l312_312849


namespace find_m_find_A_inter_CUB_l312_312025

-- Definitions of sets A and B given m
def A (m : ℤ) : Set ℤ := {-4, 2 * m - 1, m ^ 2}
def B (m : ℤ) : Set ℤ := {9, m - 5, 1 - m}

-- Define the universal set U
def U : Set ℤ := Set.univ

-- First part: Prove that m = -3
theorem find_m (m : ℤ) : A m ∩ B m = {9} → m = -3 := sorry

-- Condition that m = -3 is true
def m_val : ℤ := -3

-- Second part: Prove A ∩ C_U B = {-4, -7}
theorem find_A_inter_CUB: A m_val ∩ (U \ B m_val) = {-4, -7} := sorry

end find_m_find_A_inter_CUB_l312_312025


namespace number_of_bowls_l312_312833

theorem number_of_bowls (n : ℕ) (h1 : 12 * 8 = 96) (h2 : ∀ t : ℕ, t = 6 * n -> t = 96) : n = 16 := by
  sorry

end number_of_bowls_l312_312833


namespace intersection_M_N_l312_312626

open Set

def M := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def N := {x : ℝ | x > 1}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | 1 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_M_N_l312_312626


namespace value_at_zero_eq_sixteen_l312_312747

-- Define the polynomial P(x)
def P (x : ℚ) : ℚ := x ^ 4 - 20 * x ^ 2 + 16

-- Theorem stating the value of P(0)
theorem value_at_zero_eq_sixteen :
  P 0 = 16 :=
by
-- We know the polynomial P(x) is x^4 - 20x^2 + 16
-- When x = 0, P(0) = 0^4 - 20 * 0^2 + 16 = 16
sorry

end value_at_zero_eq_sixteen_l312_312747


namespace pablo_books_read_l312_312638

noncomputable def pages_per_book : ℕ := 150
noncomputable def cents_per_page : ℕ := 1
noncomputable def cost_of_candy : ℕ := 1500    -- $15 in cents
noncomputable def leftover_money : ℕ := 300    -- $3 in cents
noncomputable def total_money := cost_of_candy + leftover_money
noncomputable def earnings_per_book := pages_per_book * cents_per_page

theorem pablo_books_read : total_money / earnings_per_book = 12 := by
  sorry

end pablo_books_read_l312_312638


namespace common_difference_arithmetic_sequence_l312_312015

variable (n d : ℝ) (a : ℝ := 7 - 2 * d) (an : ℝ := 37) (Sn : ℝ := 198)

theorem common_difference_arithmetic_sequence :
  7 + (n - 3) * d = 37 ∧ 
  396 = n * (44 - 2 * d) ∧
  Sn = n / 2 * (a + an) →
  (∃ d : ℝ, 7 + (n - 3) * d = 37 ∧ 396 = n * (44 - 2 * d)) :=
by
  sorry

end common_difference_arithmetic_sequence_l312_312015


namespace number_of_bowls_l312_312841

-- Let n be the number of bowls on the table.
variable (n : ℕ)

-- Condition 1: There are n bowls, and each contain some grapes.
-- Condition 2: Adding 8 grapes to each of 12 specific bowls increases the average number of grapes in all bowls by 6.
-- Let's formalize the condition given in the problem
theorem number_of_bowls (h1 : 12 * 8 = 96) (h2 : 6 * n = 96) : n = 16 :=
by
  -- omitting the proof here
  sorry

end number_of_bowls_l312_312841


namespace quadratic_distinct_zeros_range_l312_312009

theorem quadratic_distinct_zeros_range (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 - (k+1)*x1 + k + 4 = 0 ∧ x2^2 - (k+1)*x2 + k + 4 = 0)
  ↔ k ∈ (Set.Iio (-3) ∪ Set.Ioi 5) :=
by
  sorry

end quadratic_distinct_zeros_range_l312_312009


namespace find_x_value_l312_312308

theorem find_x_value (a b c x y z : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : xy / (x + y) = a) (h5 : xz / (x + z) = b) (h6 : yz / (y + z) = c)
  (h7 : x + y + z = abc) : 
  x = (2 * a * b * c) / (a * b + b * c + a * c) :=
sorry

end find_x_value_l312_312308


namespace price_per_foot_of_fence_l312_312856

theorem price_per_foot_of_fence (area : ℝ) (total_cost : ℝ) (side_length : ℝ) (perimeter : ℝ) (price_per_foot : ℝ) 
  (h1 : area = 289) (h2 : total_cost = 3672) (h3 : side_length = Real.sqrt area) (h4 : perimeter = 4 * side_length) (h5 : price_per_foot = total_cost / perimeter) :
  price_per_foot = 54 := by
  sorry

end price_per_foot_of_fence_l312_312856


namespace setB_is_correct_l312_312757

def setA : Set ℤ := {1, 0, -1, 2}
def setB : Set ℤ := { y | ∃ x ∈ setA, y = Int.natAbs x }

theorem setB_is_correct : setB = {0, 1, 2} := by
  sorry

end setB_is_correct_l312_312757


namespace evaluate_exp_power_l312_312718

theorem evaluate_exp_power : (3^3)^2 = 729 := 
by {
  sorry
}

end evaluate_exp_power_l312_312718


namespace vertex_on_x_axis_l312_312971

theorem vertex_on_x_axis (d : ℝ) : 
  (∃ x : ℝ, x^2 - 6 * x + d = 0) ↔ d = 9 :=
by
  sorry

end vertex_on_x_axis_l312_312971


namespace problem_1_part1_problem_1_part2_problem_2_l312_312290

open Real

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) + 2 + 2 * cos (x) ^ 2

theorem problem_1_part1 : (∃ T > 0, ∀ x, f (x + T) = f x) := sorry

theorem problem_1_part2 : (∀ k : ℤ, ∀ x ∈ Set.Icc (k * π - π / 3) (k * π + π / 6), ∀ y ∈ Set.Icc (k * π - π / 3) (k * π + π / 6), x < y → f x > f y) := sorry

noncomputable def S_triangle (A B C : ℝ) (a b c : ℝ) : ℝ := 1 / 2 * b * c * sin A

theorem problem_2 :
  ∀ (A B C a b c : ℝ), f A = 4 → b = 1 → S_triangle A B C a b c = sqrt 3 / 2 →
    a^2 = b^2 + c^2 - 2 * b * c * cos A → a = sqrt 3 := sorry

end problem_1_part1_problem_1_part2_problem_2_l312_312290


namespace triangle_inequality_expression_non_negative_l312_312992

theorem triangle_inequality_expression_non_negative
  (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
  (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^4 + b^4 + c^4 - 2 * (a^2 * b^2 + a^2 * c^2 + b^2 * c^2) + a^2 * b * c + b^2 * a * c + c^2 * a * b ≥ 0 :=
sorry

end triangle_inequality_expression_non_negative_l312_312992


namespace opposite_of_neg_six_l312_312359

theorem opposite_of_neg_six : -(-6) = 6 := 
by
  sorry

end opposite_of_neg_six_l312_312359


namespace prime_division_l312_312635

-- Definitions used in conditions
variables {p q : ℕ}

-- We assume p and q are prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n
def divides (a b : ℕ) : Prop := ∃ k, b = k * a

-- The problem states
theorem prime_division 
  (hp : is_prime p) 
  (hq : is_prime q) 
  (hdiv : divides q (3^p - 2^p)) 
  : p ∣ (q - 1) :=
sorry

end prime_division_l312_312635


namespace christian_age_in_eight_years_l312_312699

theorem christian_age_in_eight_years (b c : ℕ)
  (h1 : c = 2 * b)
  (h2 : b + 8 = 40) :
  c + 8 = 72 :=
sorry

end christian_age_in_eight_years_l312_312699


namespace Jenny_original_number_l312_312168

theorem Jenny_original_number (y : ℝ) (h : 10 * (y / 2 - 6) = 70) : y = 26 :=
by
  sorry

end Jenny_original_number_l312_312168


namespace ellipse_foci_coordinates_l312_312656

theorem ellipse_foci_coordinates :
  ∀ (x y : ℝ),
  (x^2 / 9 + y^2 / 5 = 1) →
  (x, y) = (2, 0) ∨ (x, y) = (-2, 0) :=
by
  sorry

end ellipse_foci_coordinates_l312_312656


namespace total_garbage_collected_correct_l312_312335

def Lizzie_group_collected : ℕ := 387
def other_group_collected : ℕ := Lizzie_group_collected - 39
def total_garbage_collected : ℕ := Lizzie_group_collected + other_group_collected

theorem total_garbage_collected_correct :
  total_garbage_collected = 735 :=
sorry

end total_garbage_collected_correct_l312_312335


namespace smallest_common_multiple_l312_312219

theorem smallest_common_multiple (n : ℕ) (h8 : n % 8 = 0) (h15 : n % 15 = 0) : n = 120 :=
sorry

end smallest_common_multiple_l312_312219


namespace g_at_3_value_l312_312310

theorem g_at_3_value (c d : ℝ) (g : ℝ → ℝ) 
  (h1 : g 1 = 7)
  (h2 : g 2 = 11)
  (h3 : ∀ x : ℝ, g x = c * x + d * x + 3) : 
  g 3 = 15 :=
by
  sorry

end g_at_3_value_l312_312310


namespace pollen_mass_in_scientific_notation_l312_312949

theorem pollen_mass_in_scientific_notation : 
  ∃ c n : ℝ, 0.0000037 = c * 10^n ∧ 1 ≤ c ∧ c < 10 ∧ c = 3.7 ∧ n = -6 :=
sorry

end pollen_mass_in_scientific_notation_l312_312949


namespace relay_race_total_time_l312_312994

noncomputable def mary_time (susan_time : ℕ) : ℕ := 2 * susan_time
noncomputable def susan_time (jen_time : ℕ) : ℕ := jen_time + 10
def jen_time : ℕ := 30
noncomputable def tiffany_time (mary_time : ℕ) : ℕ := mary_time - 7

theorem relay_race_total_time :
  let mary_time := mary_time (susan_time jen_time)
  let susan_time := susan_time jen_time
  let tiffany_time := tiffany_time mary_time
  mary_time + susan_time + jen_time + tiffany_time = 223 := by
  sorry

end relay_race_total_time_l312_312994


namespace calculate_expression_l312_312694

theorem calculate_expression : 3 * Real.sqrt 2 - abs (Real.sqrt 2 - Real.sqrt 3) = 4 * Real.sqrt 2 - Real.sqrt 3 :=
  by sorry

end calculate_expression_l312_312694


namespace A_and_B_together_finish_in_ten_days_l312_312085

-- Definitions of conditions
def B_daily_work := 1 / 15
def A_daily_work := B_daily_work / 2
def combined_daily_work := A_daily_work + B_daily_work

-- The theorem to be proved
theorem A_and_B_together_finish_in_ten_days : 1 / combined_daily_work = 10 := 
  by 
    sorry

end A_and_B_together_finish_in_ten_days_l312_312085


namespace square_garden_dimensions_and_area_increase_l312_312543

def original_length : ℝ := 60
def original_width : ℝ := 20

def original_area : ℝ := original_length * original_width
def original_perimeter : ℝ := 2 * (original_length + original_width)

theorem square_garden_dimensions_and_area_increase
    (L : ℝ := 60) (W : ℝ := 20)
    (orig_area : ℝ := L * W)
    (orig_perimeter : ℝ := 2 * (L + W))
    (square_side_length : ℝ := orig_perimeter / 4)
    (new_area : ℝ := square_side_length * square_side_length)
    (area_increase : ℝ := new_area - orig_area) :
    square_side_length = 40 ∧ area_increase = 400 :=
by {sorry}

end square_garden_dimensions_and_area_increase_l312_312543


namespace max_volume_of_pyramid_l312_312790

theorem max_volume_of_pyramid
  (a b c : ℝ)
  (h1 : a + b + c = 9)
  (h2 : ∀ (α β : ℝ), α = 30 ∧ β = 45)
  : ∃ V, V = (9 * Real.sqrt 2) / 4 ∧ V = (1/6) * (Real.sqrt 2 / 2) * a * b * c :=
by
  sorry

end max_volume_of_pyramid_l312_312790


namespace minimal_period_of_sum_l312_312986

theorem minimal_period_of_sum (A B : ℝ)
  (hA : ∃ p : ℕ, p = 6 ∧ (∃ (x : ℝ) (l : ℕ), A = x / (10 ^ l * (10 ^ p - 1))))
  (hB : ∃ p : ℕ, p = 12 ∧ (∃ (y : ℝ) (m : ℕ), B = y / (10 ^ m * (10 ^ p - 1)))) :
  ∃ p : ℕ, p = 12 ∧ (∃ (z : ℝ) (n : ℕ), A + B = z / (10 ^ n * (10 ^ p - 1))) :=
sorry

end minimal_period_of_sum_l312_312986


namespace complex_eq_l312_312301

open Complex

theorem complex_eq (z : ℂ) (i : ℂ) (hz : z = 1 + I) :
  z * conj(z) + abs (conj (z)) - 1 = sqrt 2 + 1 :=
by
  sorry

end complex_eq_l312_312301


namespace problem1_problem2_l312_312781

variable {A B C a b c : ℝ}

-- Problem (1)
theorem problem1 (h : b * (1 - 2 * Real.cos A) = 2 * a * Real.cos B) : b = 2 * c := 
sorry

-- Problem (2)
theorem problem2 (a_eq : a = 1) (tanA_eq : Real.tan A = 2 * Real.sqrt 2) (b_eq_c : b = 2 * c): 
  Real.sqrt (c^2 * (1 - (Real.cos (A + B)))) = 2 * Real.sqrt 2 * b :=
sorry

end problem1_problem2_l312_312781


namespace number_of_bowls_l312_312823

theorem number_of_bowls (n : ℕ) (h : 8 * 12 = 96) (avg_increase : 6 * n = 96) : n = 16 :=
by {
  sorry
}

end number_of_bowls_l312_312823


namespace range_of_quadratic_function_l312_312562

variable (x : ℝ)
def quadratic_function (x : ℝ) : ℝ := x^2 - 2 * x + 2

theorem range_of_quadratic_function :
  (∀ y : ℝ, (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ y = quadratic_function x) ↔ (1 ≤ y ∧ y ≤ 5)) :=
by
  sorry

end range_of_quadratic_function_l312_312562


namespace find_alpha_l312_312813

noncomputable def isochronous_growth (k α x₁ x₂ y₁ y₂ : ℝ) : Prop :=
  y₁ = k * x₁^α ∧
  y₂ = k * x₂^α ∧
  x₂ = 16 * x₁ ∧
  y₂ = 8 * y₁

theorem find_alpha (k x₁ x₂ y₁ y₂ : ℝ) (h : isochronous_growth k (3/4) x₁ x₂ y₁ y₂) : 3/4 = 3/4 :=
by 
  sorry

end find_alpha_l312_312813


namespace range_of_a_l312_312429

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, a*x^2 - 3*x + 2 = 0) ∧ 
  (∀ x y : ℝ, a*x^2 - 3*x + 2 = 0 ∧ a*y^2 - 3*y + 2 = 0 → x = y) 
  ↔ (a = 0 ∨ a = 9 / 8) := by sorry

end range_of_a_l312_312429


namespace trapezoid_area_l312_312877

theorem trapezoid_area 
  (a b c : ℝ)
  (h_a : a = 5)
  (h_b : b = 15)
  (h_c : c = 13)
  : (1 / 2) * (a + b) * (Real.sqrt (c ^ 2 - ((b - a) / 2) ^ 2)) = 120 := by
  sorry

end trapezoid_area_l312_312877


namespace correct_option_l312_312296

variables {ξ : Type} [random_variable ξ]
variables {α β : Plane} {u v : vector}

/-- Definition for Variance -/
def D (x : ξ) : ℝ := sorry -- Placeholder for variance definition

/-- Given conditions -/
axiom D_xi_eq_1 : D(ξ) = 1
axiom dot_product_perpendicular : (u • v) = 0 → perpendicular α β

/-- Proposition definitions -/
def p := D(2 * ξ + 1) = 2
def q := (u • v) = 0 → perpendicular α β

/-- Correct option -/
theorem correct_option : (¬ p) ∧ q := 
by
  sorry

end correct_option_l312_312296


namespace factor_expression_l312_312700

theorem factor_expression (b : ℝ) :
  (8 * b^4 - 100 * b^3 + 14 * b^2) - (3 * b^4 - 10 * b^3 + 14 * b^2) = 5 * b^3 * (b - 18) :=
by
  sorry

end factor_expression_l312_312700


namespace sin_18_eq_sin_18_sin_54_eq_sin_36_sin_72_eq_l312_312889

-- Part 1: Prove that sin 18° = ( √5 - 1 ) / 4
theorem sin_18_eq : Real.sin (Real.pi / 10) = (Real.sqrt 5 - 1) / 4 := sorry

-- Part 2: Given sin 18° = ( √5 - 1 ) / 4, prove that sin 18° * sin 54° = 1 / 4
theorem sin_18_sin_54_eq :
  Real.sin (Real.pi / 10) = (Real.sqrt 5 - 1) / 4 → 
  Real.sin (Real.pi / 10) * Real.sin (3 * Real.pi / 10) = 1 / 4 := sorry

-- Part 3: Given sin 18° = ( √5 - 1 ) / 4, prove that sin 36° * sin 72° = √5 / 4
theorem sin_36_sin_72_eq :
  Real.sin (Real.pi / 10) = (Real.sqrt 5 - 1) / 4 → 
  Real.sin (Real.pi / 5) * Real.sin (2 * Real.pi / 5) = Real.sqrt 5 / 4 := sorry

end sin_18_eq_sin_18_sin_54_eq_sin_36_sin_72_eq_l312_312889


namespace evaluate_three_cubed_squared_l312_312710

theorem evaluate_three_cubed_squared : (3^3)^2 = 729 :=
by
  -- Given the property of exponents
  have h : (forall (a m n : ℕ), (a^m)^n = a^(m * n)) := sorry,
  -- Now prove the statement using the given property
  calc
    (3^3)^2 = 3^(3 * 2) : by rw [h 3 3 2]
          ... = 3^6       : by norm_num
          ... = 729       : by norm_num

end evaluate_three_cubed_squared_l312_312710


namespace find_number_of_spiders_l312_312925

theorem find_number_of_spiders (S : ℕ) (h1 : (1 / 2) * S = 5) : S = 10 := sorry

end find_number_of_spiders_l312_312925


namespace Derrick_yard_length_l312_312116

variables (Alex_yard Derrick_yard Brianne_yard Carla_yard Derek_yard : ℝ)

-- Given conditions as hypotheses
theorem Derrick_yard_length :
  (Alex_yard = Derrick_yard / 2) →
  (Brianne_yard = 6 * Alex_yard) →
  (Carla_yard = 3 * Brianne_yard + 5) →
  (Derek_yard = Carla_yard / 2 - 10) →
  (Brianne_yard = 30) →
  Derrick_yard = 10 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end Derrick_yard_length_l312_312116


namespace sin_of_cos_in_third_quadrant_ratio_of_trig_functions_l312_312676

-- Proof for Problem 1
theorem sin_of_cos_in_third_quadrant (α : ℝ) 
  (hcos : Real.cos α = -4 / 5)
  (hquad : π < α ∧ α < 3 * π / 2) :
  Real.sin α = -3 / 5 :=
by
  sorry

-- Proof for Problem 2
theorem ratio_of_trig_functions (α : ℝ) 
  (htan : Real.tan α = -3) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 7 / 2 :=
by
  sorry

end sin_of_cos_in_third_quadrant_ratio_of_trig_functions_l312_312676


namespace Peter_bought_4_notebooks_l312_312948

theorem Peter_bought_4_notebooks :
  (let green_notebooks := 2
   let black_notebook := 1
   let pink_notebook := 1
   green_notebooks + black_notebook + pink_notebook = 4) :=
by sorry

end Peter_bought_4_notebooks_l312_312948


namespace work_completion_l312_312988

theorem work_completion (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : a + b = 1/10) (h2 : a = 1/14) : a + b = 1/10 := 
by {
  sorry
}

end work_completion_l312_312988


namespace complement_union_intersection_l312_312430

open Set

def A : Set ℝ := {x | 3 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 2 < x ∧ x < 9}

theorem complement_union_intersection :
  (compl (A ∪ B) = {x | x ≤ 2 ∨ 9 ≤ x}) ∧
  (compl (A ∩ B) = {x | x < 3 ∨ 5 ≤ x}) :=
by
  sorry

end complement_union_intersection_l312_312430


namespace product_three_consecutive_integers_divisible_by_six_l312_312627

theorem product_three_consecutive_integers_divisible_by_six
  (n : ℕ) (h_pos : 0 < n) : ∃ k : ℕ, (n - 1) * n * (n + 1) = 6 * k :=
by sorry

end product_three_consecutive_integers_divisible_by_six_l312_312627


namespace largest_number_is_A_l312_312703

noncomputable def numA : ℝ := 4.25678
noncomputable def numB : ℝ := 4.2567777 -- repeating 7
noncomputable def numC : ℝ := 4.25676767 -- repeating 67
noncomputable def numD : ℝ := 4.25675675 -- repeating 567
noncomputable def numE : ℝ := 4.25672567 -- repeating 2567

theorem largest_number_is_A : numA > numB ∧ numA > numC ∧ numA > numD ∧ numA > numE := by
  sorry

end largest_number_is_A_l312_312703


namespace sum_reciprocals_roots_l312_312573

theorem sum_reciprocals_roots :
  (∃ p q : ℝ, p + q = 10 ∧ p * q = 3) →
  (∃ p q : ℝ, p ≠ 0 ∧ q ≠ 0 → (1 / p) + (1 / q) = 10 / 3) :=
by
  sorry

end sum_reciprocals_roots_l312_312573


namespace sum_of_ten_numbers_l312_312316

theorem sum_of_ten_numbers (average count : ℝ) (h_avg : average = 5.3) (h_count : count = 10) : 
  average * count = 53 :=
by
  sorry

end sum_of_ten_numbers_l312_312316


namespace rug_area_correct_l312_312990

def floor_length : ℕ := 10
def floor_width : ℕ := 8
def strip_width : ℕ := 2

def adjusted_length : ℕ := floor_length - 2 * strip_width
def adjusted_width : ℕ := floor_width - 2 * strip_width

def area_floor : ℕ := floor_length * floor_width
def area_rug : ℕ := adjusted_length * adjusted_width

theorem rug_area_correct : area_rug = 24 := by
  sorry

end rug_area_correct_l312_312990


namespace value_of_a_12_l312_312902

variable {a : ℕ → ℝ} (h1 : a 6 + a 10 = 20) (h2 : a 4 = 2)

theorem value_of_a_12 : a 12 = 18 :=
by
  sorry

end value_of_a_12_l312_312902


namespace lex_apples_l312_312946

theorem lex_apples (A : ℕ) (h1 : A / 5 < 100) (h2 : A = (A / 5) + ((A / 5) + 9) + 42) : A = 85 :=
by
  sorry

end lex_apples_l312_312946


namespace munchausen_forest_l312_312617

theorem munchausen_forest (E B : ℕ) (h : B = 10 * E) : B > E := by sorry

end munchausen_forest_l312_312617


namespace senior_tickets_count_l312_312979

theorem senior_tickets_count (A S : ℕ) 
  (h1 : A + S = 510)
  (h2 : 21 * A + 15 * S = 8748) :
  S = 327 :=
sorry

end senior_tickets_count_l312_312979


namespace problem1_problem2_l312_312695

theorem problem1 (a b : ℝ) :
  5 * a * b^2 - 2 * a^2 * b + 3 * a * b^2 - a^2 * b - 4 * a * b^2 = 4 * a * b^2 - 3 * a^2 * b := 
by sorry

theorem problem2 (m n : ℝ) :
  -5 * m * n^2 - (2 * m^2 * n - 2 * (m^2 * n - 2 * m * n^2)) = -9 * m * n^2 := 
by sorry

end problem1_problem2_l312_312695


namespace midpoint_translation_l312_312035

theorem midpoint_translation (x1 y1 x2 y2 tx ty mx my : ℤ) 
  (hx1 : x1 = 1) (hy1 : y1 = 3) (hx2 : x2 = 5) (hy2 : y2 = -7)
  (htx : tx = 3) (hty : ty = -4)
  (hmx : mx = (x1 + x2) / 2 + tx) (hmy : my = (y1 + y2) / 2 + ty) : 
  mx = 6 ∧ my = -6 :=
by
  sorry

end midpoint_translation_l312_312035


namespace determine_f_4_l312_312022

theorem determine_f_4 (f g : ℝ → ℝ)
  (h1 : ∀ x y z : ℝ, f (x^2 + y * f z) = x * g x + z * g y)
  (h2 : ∀ x : ℝ, g x = 2 * x) :
  f 4 = 32 :=
sorry

end determine_f_4_l312_312022


namespace evaluate_power_l312_312708

theorem evaluate_power : (3^3)^2 = 729 := 
by 
  sorry

end evaluate_power_l312_312708


namespace value_of_x_l312_312131

theorem value_of_x (x y : ℕ) (h1 : x / y = 3) (h2 : y = 25) : x = 75 := by
  sorry

end value_of_x_l312_312131


namespace don_raise_l312_312275

variable (D R : ℝ)

theorem don_raise 
  (h1 : R = 0.08 * D)
  (h2 : 840 = 0.08 * 10500)
  (h3 : (D + R) - (10500 + 840) = 540) : 
  R = 880 :=
by sorry

end don_raise_l312_312275


namespace construct_triangle_given_side_and_medians_l312_312515

theorem construct_triangle_given_side_and_medians
  (AB : ℝ) (m_a m_b : ℝ)
  (h1 : AB > 0) (h2 : m_a > 0) (h3 : m_b > 0) :
  ∃ (A B C : ℝ × ℝ),
    (∃ G : ℝ × ℝ, 
      dist A B = AB ∧ 
      dist A G = (2 / 3) * m_a ∧
      dist B G = (2 / 3) * m_b ∧ 
      dist G (midpoint ℝ A C) = m_b / 3 ∧ 
      dist G (midpoint ℝ B C) = m_a / 3) :=
sorry

end construct_triangle_given_side_and_medians_l312_312515


namespace evaluate_exponent_l312_312731

theorem evaluate_exponent : (3^3)^2 = 729 := by
  sorry

end evaluate_exponent_l312_312731


namespace intersection_setA_setB_l312_312774

def setA := {x : ℝ | |x| < 1}
def setB := {x : ℝ | x^2 - 2 * x ≤ 0}

theorem intersection_setA_setB :
  {x : ℝ | 0 ≤ x ∧ x < 1} = setA ∩ setB :=
by
  sorry

end intersection_setA_setB_l312_312774


namespace number_of_bowls_l312_312838

theorem number_of_bowls (n : ℕ) :
  (∀ (b : ℕ), b > 0) →
  (∀ (a : ℕ), ∃ (k : ℕ), true) →
  (8 * 12 = 96) →
  (6 * n = 96) →
  n = 16 :=
by
  intros h1 h2 h3 h4
  sorry

end number_of_bowls_l312_312838


namespace sum_of_remainders_l312_312377

theorem sum_of_remainders (a b c : ℕ) (h1 : a % 15 = 11) (h2 : b % 15 = 13) (h3 : c % 15 = 14) : (a + b + c) % 15 = 8 :=
by
  sorry

end sum_of_remainders_l312_312377


namespace number_of_bowls_l312_312848

noncomputable theory
open Classical

theorem number_of_bowls (n : ℕ) 
  (h1 : 8 * 12 = 6 * n) : n = 16 := 
by
  sorry

end number_of_bowls_l312_312848


namespace gg1_eq_13_l312_312273

def g (n : ℕ) : ℕ :=
if n < 3 then n^2 + 1
else if n < 6 then 2 * n + 3
else 4 * n - 2

theorem gg1_eq_13 : g (g (g 1)) = 13 :=
by
  sorry

end gg1_eq_13_l312_312273


namespace evaluate_power_l312_312706

theorem evaluate_power : (3^3)^2 = 729 := 
by 
  sorry

end evaluate_power_l312_312706


namespace river_depth_ratio_l312_312164

-- Definitions based on the conditions
def depthMidMay : ℝ := 5
def increaseMidJune : ℝ := 10
def depthMidJune : ℝ := depthMidMay + increaseMidJune
def depthMidJuly : ℝ := 45

-- The theorem based on the question and correct answer
theorem river_depth_ratio : depthMidJuly / depthMidJune = 3 := by 
  -- Proof skipped for illustration purposes
  sorry

end river_depth_ratio_l312_312164


namespace complementary_three_card_sets_l312_312868

-- Definitions for the problem conditions
inductive Shape | circle | square | triangle | star
inductive Color | red | blue | green | yellow
inductive Shade | light | medium | dark | very_dark

-- Definition of a Card as a combination of shape, color, shade
structure Card :=
(shape : Shape)
(color : Color)
(shade : Shade)

-- Definition of a set being complementary
def is_complementary (c1 c2 c3 : Card) : Prop :=
  ((c1.shape = c2.shape ∧ c2.shape = c3.shape) ∨ (c1.shape ≠ c2.shape ∧ c2.shape ≠ c3.shape ∧ c1.shape ≠ c3.shape)) ∧
  ((c1.color = c2.color ∧ c2.color = c3.color) ∨ (c1.color ≠ c2.color ∧ c2.color ≠ c3.color ∧ c1.color ≠ c3.color)) ∧
  ((c1.shade = c2.shade ∧ c2.shade = c3.shade) ∨ (c1.shade ≠ c2.shade ∧ c2.shade ≠ c3.shade ∧ c1.shade ≠ c3.shade))

-- Definition of the problem statement
def complementary_three_card_sets_count : Nat :=
  360

-- The theorem to be proved
theorem complementary_three_card_sets : ∃ (n : Nat), n = complementary_three_card_sets_count :=
  by
    use 360
    sorry

end complementary_three_card_sets_l312_312868


namespace Alyssa_total_spent_l312_312258

-- define the amounts spent on grapes and cherries
def costGrapes: ℝ := 12.08
def costCherries: ℝ := 9.85

-- define the total cost based on the given conditions
def totalCost: ℝ := costGrapes + costCherries

-- prove that the total cost equals 21.93
theorem Alyssa_total_spent:
  totalCost = 21.93 := 
  by
  -- proof to be completed
  sorry

end Alyssa_total_spent_l312_312258


namespace difference_in_areas_l312_312326

def S1 (x y : ℝ) : Prop :=
  Real.log (3 + x ^ 2 + y ^ 2) / Real.log 2 ≤ 2 + Real.log (x + y) / Real.log 2

def S2 (x y : ℝ) : Prop :=
  Real.log (3 + x ^ 2 + y ^ 2) / Real.log 2 ≤ 3 + Real.log (x + y) / Real.log 2

theorem difference_in_areas : 
  let area_S1 := π * 1 ^ 2
  let area_S2 := π * (Real.sqrt 13) ^ 2
  area_S2 - area_S1 = 12 * π :=
by
  sorry

end difference_in_areas_l312_312326


namespace max_value_expression_correct_l312_312969

noncomputable def max_value_expression (a b c d : ℝ) : ℝ :=
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_value_expression_correct :
  ∃ a b c d : ℝ, a ∈ Set.Icc (-13.5) 13.5 ∧ b ∈ Set.Icc (-13.5) 13.5 ∧ 
                  c ∈ Set.Icc (-13.5) 13.5 ∧ d ∈ Set.Icc (-13.5) 13.5 ∧ 
                  max_value_expression a b c d = 756 := 
sorry

end max_value_expression_correct_l312_312969


namespace scientific_notation_of_trade_volume_l312_312512

-- Define the total trade volume
def total_trade_volume : ℕ := 175000000000

-- Define the expected scientific notation result
def expected_result : ℝ := 1.75 * 10^11

-- Theorem stating the problem
theorem scientific_notation_of_trade_volume :
  (total_trade_volume : ℝ) = expected_result := by
  sorry

end scientific_notation_of_trade_volume_l312_312512


namespace number_of_men_in_first_group_l312_312652

-- Define the conditions and the proof problem
theorem number_of_men_in_first_group (M : ℕ) 
  (h1 : ∀ t : ℝ, 22 * t = M) 
  (h2 : ∀ t' : ℝ, 18 * 17.11111111111111 = t') :
  M = 14 := 
by
  sorry

end number_of_men_in_first_group_l312_312652


namespace tiger_distance_traveled_l312_312684

theorem tiger_distance_traveled :
  let distance1 := 25 * 1
  let distance2 := 35 * 2
  let distance3 := 20 * 1.5
  let distance4 := 10 * 1
  let distance5 := 50 * 0.5
  distance1 + distance2 + distance3 + distance4 + distance5 = 160 := by
sorry

end tiger_distance_traveled_l312_312684


namespace population_decreases_l312_312851

theorem population_decreases (P_0 : ℝ) (k : ℝ) (n : ℕ) (hP0 : P_0 > 0) (hk : -1 < k ∧ k < 0) : 
  P_0 * (1 + k)^n * k < 0 → P_0 * (1 + k)^(n + 1) < P_0 * (1 + k)^n := by
  sorry

end population_decreases_l312_312851


namespace sum_remainder_l312_312379

theorem sum_remainder (p q r : ℕ) (hp : p % 15 = 11) (hq : q % 15 = 13) (hr : r % 15 = 14) : 
  (p + q + r) % 15 = 8 :=
by
  sorry

end sum_remainder_l312_312379


namespace number_of_people_who_bought_1_balloon_l312_312079

-- Define the variables and the main theorem statement
variables (x1 x2 x3 x4 : ℕ)

theorem number_of_people_who_bought_1_balloon : 
  (x1 + x2 + x3 + x4 = 101) → 
  (x1 + 2 * x2 + 3 * x3 + 4 * x4 = 212) →
  (x4 = x2 + 13) → 
  x1 = 52 :=
by
  intros h1 h2 h3
  sorry

end number_of_people_who_bought_1_balloon_l312_312079


namespace max_area_rect_l312_312205

noncomputable def maximize_area (l w : ℕ) : ℕ :=
  l * w

theorem max_area_rect (l w: ℕ) (hl_even : l % 2 = 0) (h_perim : 2*l + 2*w = 40) :
  maximize_area l w = 100 :=
by
  sorry 

end max_area_rect_l312_312205


namespace base8_addition_example_l312_312126

theorem base8_addition_example :
  ∀ (a b c : ℕ), a = 245 ∧ b = 174 ∧ c = 354 → a + b + c = 1015 :=
by
  sorry

end base8_addition_example_l312_312126


namespace supplemental_tank_time_l312_312690

-- Define the given conditions as assumptions
def primary_tank_time : Nat := 2
def total_time_needed : Nat := 8
def supplemental_tanks : Nat := 6
def additional_time_needed : Nat := total_time_needed - primary_tank_time

-- Define the theorem to prove
theorem supplemental_tank_time :
  additional_time_needed / supplemental_tanks = 1 :=
by
  -- Here we would provide the proof, but it is omitted with "sorry"
  sorry

end supplemental_tank_time_l312_312690


namespace quadratic_inequality_empty_solution_set_l312_312010

theorem quadratic_inequality_empty_solution_set (a b c : ℝ) (hₐ : a ≠ 0) :
  (∀ x : ℝ, a * x^2 + b * x + c < 0 → False) ↔ (a > 0 ∧ (b^2 - 4 * a * c) ≤ 0) :=
by 
  sorry

end quadratic_inequality_empty_solution_set_l312_312010


namespace advance_agency_fees_eq_8280_l312_312374

-- Conditions
variables (Commission GivenFees Incentive AdvanceAgencyFees : ℝ)
-- Given values
variables (h_comm : Commission = 25000) 
          (h_given : GivenFees = 18500) 
          (h_incent : Incentive = 1780)

-- The problem statement to prove
theorem advance_agency_fees_eq_8280 
    (h_comm : Commission = 25000) 
    (h_given : GivenFees = 18500) 
    (h_incent : Incentive = 1780)
    : AdvanceAgencyFees = 26780 - GivenFees :=
by
  sorry

end advance_agency_fees_eq_8280_l312_312374


namespace number_of_bowls_l312_312827

theorem number_of_bowls (n : ℕ) (h1 : 12 * 8 = 96) (h2 : 6 * n = 96) : n = 16 :=
by
  -- equations from the conditions
  have h3 : 96 = 96 := by sorry
  exact sorry

end number_of_bowls_l312_312827


namespace avg_speed_l312_312209

variable (d1 d2 t1 t2 : ℕ)

-- Conditions
def distance_first_hour : ℕ := 80
def distance_second_hour : ℕ := 40
def time_first_hour : ℕ := 1
def time_second_hour : ℕ := 1

-- Ensure that total distance and total time are defined correctly from conditions
def total_distance : ℕ := distance_first_hour + distance_second_hour
def total_time : ℕ := time_first_hour + time_second_hour

-- Theorem to prove the average speed
theorem avg_speed : total_distance / total_time = 60 := by
  sorry

end avg_speed_l312_312209


namespace tan_domain_correct_l312_312807

noncomputable def domain_tan : Set ℝ := {x | ∃ k : ℤ, x ≠ k * Real.pi + 3 * Real.pi / 4}

def is_domain_correct : Prop :=
  ∀ x : ℝ, x ∈ domain_tan ↔ (∃ k : ℤ, x ≠ k * Real.pi + 3 * Real.pi / 4)

-- Statement of the problem in Lean 4
theorem tan_domain_correct : is_domain_correct :=
  sorry

end tan_domain_correct_l312_312807


namespace present_number_of_teachers_l312_312528

theorem present_number_of_teachers (S T : ℕ) (h1 : S = 50 * T) (h2 : S + 50 = 25 * (T + 5)) : T = 3 := 
by 
  sorry

end present_number_of_teachers_l312_312528


namespace convert_to_general_form_l312_312271

theorem convert_to_general_form (x : ℝ) :
  5 * x^2 - 2 * x = 3 * (x + 1) ↔ 5 * x^2 - 5 * x - 3 = 0 :=
by
  sorry

end convert_to_general_form_l312_312271


namespace problem_1_problem_2_problem_3_l312_312886

section MathProblems

variable (a b c m n x y : ℝ)
-- Problem 1
theorem problem_1 :
  (-6 * a^2 * b^5 * c) / (-2 * a * b^2)^2 = (3/2) * b * c := sorry

-- Problem 2
theorem problem_2 :
  (-3 * m - 2 * n) * (3 * m + 2 * n) = -9 * m^2 - 12 * m * n - 4 * n^2 := sorry

-- Problem 3
theorem problem_3 :
  ((x - 2 * y)^2 - (x - 2 * y) * (x + 2 * y)) / (2 * y) = -2 * x + 4 * y := sorry

end MathProblems

end problem_1_problem_2_problem_3_l312_312886


namespace common_ratio_of_geometric_sequence_l312_312317

-- Define positive geometric sequence a_n with common ratio q
def geometric_sequence (a q : ℝ) (n : ℕ) : ℝ := a * q^n

-- Define the relevant conditions
variable {a q : ℝ}
variable (h1 : a * q^4 + 2 * a * q^2 * q^6 + a * q^4 * q^8 = 16)
variable (h2 : (a * q^4 + a * q^8) / 2 = 4)
variable (pos_q : q > 0)

-- Define the goal: proving the common ratio q is sqrt(2)
theorem common_ratio_of_geometric_sequence : q = Real.sqrt 2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l312_312317


namespace calculate_principal_l312_312285

theorem calculate_principal
  (I : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
  (hI : I = 8625)
  (hR : R = 50 / 3)
  (hT : T = 3 / 4)
  (hInterest : I = (P * (R / 100) * T)) :
  P = 6900000 := by
  sorry

end calculate_principal_l312_312285


namespace system_of_equations_solution_l312_312650

theorem system_of_equations_solution (x y : ℤ) 
  (h1 : x^2 + x * y + y^2 = 37) 
  (h2 : x^4 + x^2 * y^2 + y^4 = 481) : 
  (x = 3 ∧ y = 4) ∨ (x = 4 ∧ y = 3) ∨ (x = -3 ∧ y = -4) ∨ (x = -4 ∧ y = -3) := 
by sorry

end system_of_equations_solution_l312_312650


namespace total_rectangles_l312_312127

-- Definitions
def horizontal_lines : ℕ := 5
def vertical_lines : ℕ := 4
def exclude_line_pair: ℕ := 1
def total_combinations (n m : ℕ) : ℕ := Nat.choose n m

-- Statement
theorem total_rectangles (h_lines : ℕ) (v_lines : ℕ) 
  (exclude_pair : ℕ) (valid_h_comb : ℕ) (valid_v_comb : ℕ) :
  h_lines = horizontal_lines →
  v_lines = vertical_lines →
  exclude_pair = exclude_line_pair →
  valid_h_comb = total_combinations 5 2 - exclude_pair →
  valid_v_comb = total_combinations 4 2 →
  valid_h_comb * valid_v_comb = 54 :=
by intros; sorry

end total_rectangles_l312_312127


namespace UPOMB_position_l312_312215

-- Define the set of letters B, M, O, P, and U
def letters : List Char := ['B', 'M', 'O', 'P', 'U']

-- Define the word UPOMB
def word := "UPOMB"

-- Define a function that calculates the factorial of a number
def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define a function to calculate the position of a word in the alphabetical permutations of a list of characters
def word_position (w : String) (chars : List Char) : Nat :=
  let rec aux (w : List Char) (remaining : List Char) : Nat :=
    match w with
    | [] => 1
    | c :: cs =>
      let before_count := remaining.filter (· < c) |>.length
      let rest_count := factorial (remaining.length - 1)
      before_count * rest_count + aux cs (remaining.erase c)
  aux w.data chars

-- The desired theorem statement
theorem UPOMB_position : word_position word letters = 119 := by
  sorry

end UPOMB_position_l312_312215


namespace at_least_one_heart_or_king_l312_312243

-- Define the conditions
def total_cards := 52
def hearts := 13
def kings := 4
def king_of_hearts := 1
def cards_hearts_or_kings := hearts + kings - king_of_hearts

-- Calculate probabilities based on the above conditions
def probability_not_heart_or_king := 
  1 - (cards_hearts_or_kings / total_cards)

def probability_neither_heart_nor_king :=
  (probability_not_heart_or_king) ^ 2

def probability_at_least_one_heart_or_king :=
  1 - probability_neither_heart_nor_king

-- State the theorem to be proved
theorem at_least_one_heart_or_king : 
  probability_at_least_one_heart_or_king = (88 / 169) :=
by
  sorry

end at_least_one_heart_or_king_l312_312243


namespace sum_first_five_terms_arith_seq_l312_312811

theorem sum_first_five_terms_arith_seq (a : ℕ → ℤ)
  (h4 : a 4 = 3) (h5 : a 5 = 7) (h6 : a 6 = 11) :
  a 1 + a 2 + a 3 + a 4 + a 5 = -5 :=
by
  sorry

end sum_first_five_terms_arith_seq_l312_312811


namespace convex_polygon_obtuse_sum_l312_312445
open Int

def convex_polygon_sides (n : ℕ) (S : ℕ) : Prop :=
  180 * (n - 2) = 3000 + S ∧ (S = 60 ∨ S = 240)

theorem convex_polygon_obtuse_sum (n : ℕ) (hn : 3 ≤ n) :
  (∃ S, convex_polygon_sides n S) ↔ (n = 19 ∨ n = 20) :=
by
  sorry

end convex_polygon_obtuse_sum_l312_312445


namespace total_area_three_plots_l312_312875

variable (x y z A : ℝ)

theorem total_area_three_plots :
  (x = (2 / 5) * A) →
  (z = x - 16) →
  (y = (9 / 8) * z) →
  (A = x + y + z) →
  A = 96 :=
by
  intros h1 h2 h3 h4
  sorry

end total_area_three_plots_l312_312875


namespace option_d_correct_l312_312065

theorem option_d_correct (x y : ℝ) : -4 * x * y + 3 * x * y = -1 * x * y := 
by {
  sorry
}

end option_d_correct_l312_312065


namespace fifth_stack_33_l312_312489

def cups_in_fifth_stack (a d : ℕ) : ℕ :=
a + 4 * d

theorem fifth_stack_33 
  (a : ℕ) 
  (d : ℕ) 
  (h_first_stack : a = 17) 
  (h_pattern : d = 4) : 
  cups_in_fifth_stack a d = 33 := by
  sorry

end fifth_stack_33_l312_312489


namespace investment_ratio_l312_312660

theorem investment_ratio (P Q : ℝ) (h1 : (P * 5) / (Q * 9) = 7 / 9) : P / Q = 7 / 5 :=
by sorry

end investment_ratio_l312_312660


namespace ratio_AH_HD_triangle_l312_312937

theorem ratio_AH_HD_triangle (BC AC : ℝ) (angleC : ℝ) (H AD HD : ℝ) 
  (hBC : BC = 4) (hAC : AC = 3 * Real.sqrt 2) (hAngleC : angleC = 45) 
  (hAD : AD = 3) (hHD : HD = 1) : 
  (AH / HD) = 2 :=
by
  sorry

end ratio_AH_HD_triangle_l312_312937


namespace sum_symmetry_l312_312023

def f (x : ℝ) : ℝ :=
  x^2 * (1 - x)^2

theorem sum_symmetry :
  f (1/7) - f (2/7) + f (3/7) - f (4/7) + f (5/7) - f (6/7) = 0 :=
by
  sorry

end sum_symmetry_l312_312023


namespace find_initial_balance_l312_312503

-- Define the initial balance
variable (X : ℝ)

-- Conditions
def balance_tripled (X : ℝ) : ℝ := 3 * X
def balance_after_withdrawal (X : ℝ) : ℝ := balance_tripled X - 250

-- The problem statement to prove
theorem find_initial_balance (h : balance_after_withdrawal X = 950) : X = 400 :=
by
  sorry

end find_initial_balance_l312_312503


namespace angle_is_20_l312_312584

theorem angle_is_20 (x : ℝ) (h : 180 - x = 2 * (90 - x) + 20) : x = 20 :=
by
  sorry

end angle_is_20_l312_312584


namespace expected_total_rain_l312_312548

theorem expected_total_rain :
  let p_sun := 0.30
  let p_rain5 := 0.30
  let p_rain12 := 0.40
  let rain_sun := 0
  let rain_rain5 := 5
  let rain_rain12 := 12
  let days := 6
  let E_rain := p_sun * rain_sun + p_rain5 * rain_rain5 + p_rain12 * rain_rain12
  E_rain * days = 37.8 :=
by
  -- Proof omitted
  sorry

end expected_total_rain_l312_312548


namespace inequality_proof_l312_312942

variable {x y : ℝ}

theorem inequality_proof (hx : 0 < x) (hy : 0 < y) (hxy : x > y) :
    2 * x + 1 / (x ^ 2 - 2 * x * y + y ^ 2) ≥ 2 * y + 3 := 
  sorry

end inequality_proof_l312_312942


namespace number_of_red_dresses_l312_312323

-- Define context for Jane's dress shop problem
def dresses_problem (R B : Nat) : Prop :=
  R + B = 200 ∧ B = R + 34

-- Prove that the number of red dresses (R) should be 83
theorem number_of_red_dresses : ∃ R B : Nat, dresses_problem R B ∧ R = 83 :=
by
  sorry

end number_of_red_dresses_l312_312323


namespace emails_received_l312_312322

variable (x y : ℕ)

theorem emails_received (h1 : 3 + 6 = 9) (h2 : x + y + 9 = 10) : x + y = 1 := by
  sorry

end emails_received_l312_312322


namespace sum_of_first_9_terms_is_27_l312_312618

noncomputable def a_n (n : ℕ) : ℝ := sorry -- Definition for the geometric sequence
noncomputable def b_n (n : ℕ) : ℝ := sorry -- Definition for the arithmetic sequence

axiom a_geo_seq : ∃ r : ℝ, ∀ n : ℕ, a_n (n + 1) = a_n n * r
axiom b_ari_seq : ∃ d : ℝ, ∀ n : ℕ, b_n (n + 1) = b_n n + d
axiom a5_eq_3 : 3 * a_n 5 - a_n 3 * a_n 7 = 0
axiom b5_eq_a5 : b_n 5 = a_n 5

noncomputable def S_9 := (1 / 2) * 9 * (b_n 1 + b_n 9)

theorem sum_of_first_9_terms_is_27 : S_9 = 27 := by
  sorry

end sum_of_first_9_terms_is_27_l312_312618


namespace abs_inequality_solution_l312_312751

theorem abs_inequality_solution (x : ℝ) : 
  3 ≤ |x - 3| ∧ |x - 3| ≤ 7 ↔ (-4 ≤ x ∧ x ≤ 0) ∨ (6 ≤ x ∧ x ≤ 10) := 
by {
  sorry
}

end abs_inequality_solution_l312_312751


namespace binomial_log_inequality_l312_312644

theorem binomial_log_inequality (n : ℤ) :
  n * Real.log 2 ≤ Real.log (Nat.choose (2 * n.natAbs) n.natAbs) ∧ 
  Real.log (Nat.choose (2 * n.natAbs) n.natAbs) ≤ n * Real.log 4 :=
by sorry

end binomial_log_inequality_l312_312644


namespace quadratic_vertex_problem_l312_312505

/-- 
    Given a quadratic equation y = ax^2 + bx + c, where (2, -3) 
    is the vertex of the parabola and it passes through (0, 1), 
    prove that a - b + c = 6. 
-/
theorem quadratic_vertex_problem 
    (a b c : ℤ)
    (h : ∀ x : ℝ, y = a * (x - 2)^2 - 3)
    (h_point : y = 1)
    (h_passes_through_origin : y = a * (0 - 2)^2 - 3) :
    a - b + c = 6 :=
sorry

end quadratic_vertex_problem_l312_312505


namespace greatest_integer_b_l312_312284

theorem greatest_integer_b (b : ℤ) : (∀ x : ℝ, x^2 + (b : ℝ) * x + 7 ≠ 0) → b ≤ 5 :=
by sorry

end greatest_integer_b_l312_312284


namespace average_stamps_collected_per_day_l312_312179

open Nat

-- Define an arithmetic sequence
def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  a + d * (n - 1)

-- Define the sum of the first n terms of an arithmetic sequence
def sum_arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Given conditions
def a := 10
def d := 10
def n := 7

-- Prove that the average number of stamps collected over 7 days is 40
theorem average_stamps_collected_per_day : 
  sum_arithmetic_sequence a d n / n = 40 := 
by
  sorry

end average_stamps_collected_per_day_l312_312179


namespace no_rational_roots_of_odd_l312_312645

theorem no_rational_roots_of_odd (m n : ℤ) (hm : m % 2 = 1) (hn : n % 2 = 1) : ¬ ∃ x : ℚ, x^2 + 2 * m * x + 2 * n = 0 :=
sorry

end no_rational_roots_of_odd_l312_312645


namespace find_positive_integer_pairs_l312_312281

theorem find_positive_integer_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (2 * a^2 = 3 * b^3) ↔ ∃ d : ℕ, 0 < d ∧ a = 18 * d^3 ∧ b = 6 * d^2 :=
by
  sorry

end find_positive_integer_pairs_l312_312281


namespace percentage_increase_l312_312441

theorem percentage_increase (N M P : ℝ) (h : M = N * (1 + P / 100)) : ((M - N) / N) * 100 = P :=
by
  sorry

end percentage_increase_l312_312441


namespace units_digit_2749_987_l312_312307

def mod_units_digit (base : ℕ) (exp : ℕ) : ℕ :=
  (base % 10)^(exp % 2) % 10

theorem units_digit_2749_987 : mod_units_digit 2749 987 = 9 := 
by 
  sorry

end units_digit_2749_987_l312_312307


namespace evaluate_fraction_l312_312468

open Complex

theorem evaluate_fraction (a b : ℂ) (h : a ≠ 0 ∧ b ≠ 0) (h_eq : a^2 - a*b + b^2 = 0) : 
  (a^6 + b^6) / (a + b)^6 = 1 / 18 := by
  sorry

end evaluate_fraction_l312_312468


namespace number_of_rice_packets_l312_312853

theorem number_of_rice_packets
  (initial_balance : ℤ) 
  (price_per_rice_packet : ℤ)
  (num_wheat_flour_packets : ℤ) 
  (price_per_wheat_flour_packet : ℤ)
  (price_soda : ℤ) 
  (remaining_balance : ℤ)
  (spent : ℤ)
  (eqn : initial_balance - (price_per_rice_packet * 2 + num_wheat_flour_packets * price_per_wheat_flour_packet + price_soda) = remaining_balance) :
  price_per_rice_packet * 2 + num_wheat_flour_packets * price_per_wheat_flour_packet + price_soda = spent 
    → initial_balance - spent = remaining_balance
    → 2 = 2 :=
by 
  sorry

end number_of_rice_packets_l312_312853


namespace misty_is_three_times_smaller_l312_312183

-- Define constants representing the favorite numbers of Misty and Glory
def G : ℕ := 450
def total_sum : ℕ := 600

-- Define Misty's favorite number in terms of the total sum and Glory's favorite number
def M : ℕ := total_sum - G

-- The main theorem stating that Misty's favorite number is 3 times smaller than Glory's favorite number
theorem misty_is_three_times_smaller : G / M = 3 := by
  -- Sorry placeholder indicating the need for further proof
  sorry

end misty_is_three_times_smaller_l312_312183


namespace weavers_in_first_group_l312_312039

theorem weavers_in_first_group :
  (∃ W : ℕ, (W * 4 = 4) ∧ (12 * 12 = 36) ∧ (4 / (W * 4) = 36 / (12 * 12))) -> (W = 4) :=
by
  sorry

end weavers_in_first_group_l312_312039


namespace minimize_cost_l312_312083

noncomputable def total_cost (x : ℝ) : ℝ := (16000000 / x) + 40000 * x

theorem minimize_cost : ∃ (x : ℝ), x > 0 ∧ (∀ y > 0, total_cost x ≤ total_cost y) ∧ x = 20 := 
sorry

end minimize_cost_l312_312083


namespace average_excluding_highest_lowest_l312_312197

-- Define the conditions
def batting_average : ℚ := 59
def innings : ℕ := 46
def highest_score : ℕ := 156
def score_difference : ℕ := 150
def lowest_score : ℕ := highest_score - score_difference

-- Prove the average excluding the highest and lowest innings is 58
theorem average_excluding_highest_lowest :
  let total_runs := batting_average * innings
  let runs_excluding := total_runs - highest_score - lowest_score
  let effective_innings := innings - 2
  runs_excluding / effective_innings = 58 := by
  -- Insert proof here
  sorry

end average_excluding_highest_lowest_l312_312197


namespace eval_exp_l312_312722

theorem eval_exp : (3^3)^2 = 729 := sorry

end eval_exp_l312_312722


namespace min_abc_sum_l312_312905

theorem min_abc_sum (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 2010) : 
  a + b + c ≥ 78 := 
sorry

end min_abc_sum_l312_312905


namespace expected_value_bound_l312_312473

noncomputable def xi : ℝ → ℝ := sorry -- definition of xi as a nonnegative random variable
noncomputable def zeta : ℝ → ℝ := sorry -- definition of zeta as a nonnegative random variable
def indicator (p : Prop) [Decidable p] := if p then 1 else 0

axiom cond1 (x : ℝ) (hx : x > 0) :
  ℝ := sorry -- P(xi ≥ x) ≤ x⁻¹ E[zeta * indicator(xi ≥ x)]

theorem expected_value_bound (p: ℝ) (hp: p > 1) :
  ℝ := sorry -- E[xi^p] ≤ (p/(p-1))^p E[zeta^p]

end expected_value_bound_l312_312473


namespace set_listing_method_l312_312439

theorem set_listing_method :
  {x : ℤ | -3 < 2 * x - 1 ∧ 2 * x - 1 < 5} = {0, 1, 2} :=
by
  sorry

end set_listing_method_l312_312439


namespace solution_to_system_of_equations_l312_312651

theorem solution_to_system_of_equations :
  ∃ x y : ℤ, 4 * x - 3 * y = 11 ∧ 2 * x + y = 13 ∧ x = 5 ∧ y = 3 :=
by
  sorry

end solution_to_system_of_equations_l312_312651


namespace vision_data_decimal_l312_312973

-- Definition of the vision problem
noncomputable def vision_method_relation (L V : ℝ) : Prop :=
  L = 5 + Real.log10 V

-- The student's five-point recording vision data
def student_L : ℝ := 4.8

-- The vision data in the decimal system to be proven
def correct_V : ℝ := 0.6

-- The theorem stating the equivalence problem
theorem vision_data_decimal : vision_method_relation student_L correct_V :=
sorry

end vision_data_decimal_l312_312973


namespace systematic_sampling_40th_number_l312_312610

open Nat

theorem systematic_sampling_40th_number (N n : ℕ) (sample_size_eq : n = 50) (total_students_eq : N = 1000) (k_def : k = N / n) (first_number : ℕ) (first_number_eq : first_number = 15) : 
  first_number + k * 39 = 795 := by
  sorry

end systematic_sampling_40th_number_l312_312610


namespace volume_of_tetrahedron_equals_450_sqrt_2_l312_312565

-- Given conditions
variables {A B C D : Point}
variables (areaABC areaBCD : ℝ) (BC : ℝ) (angleABC_BCD : ℝ)

-- The specific values for the conditions
axiom h_areaABC : areaABC = 150
axiom h_areaBCD : areaBCD = 90
axiom h_BC : BC = 10
axiom h_angleABC_BCD : angleABC_BCD = π / 4  -- 45 degrees in radians

-- Definition of the volume to be proven
def volume_tetrahedron (A B C D : Point) : ℝ :=
  (1 / 3) * areaABC * (18 * real.sin angleABC_BCD)

-- Final proof statement
theorem volume_of_tetrahedron_equals_450_sqrt_2 :
  volume_tetrahedron A B C D = 450 * real.sqrt 2 :=
by 
  -- Preliminary setup, add the relevant properties and results
  sorry

end volume_of_tetrahedron_equals_450_sqrt_2_l312_312565


namespace eval_expr_l312_312745

theorem eval_expr : (3^3)^2 = 729 := 
by
  sorry

end eval_expr_l312_312745


namespace max_enclosed_area_perimeter_160_length_twice_width_l312_312344

theorem max_enclosed_area_perimeter_160_length_twice_width 
  (W L : ℕ) 
  (h1 : 2 * (L + W) = 160) 
  (h2 : L = 2 * W) : 
  L * W = 1352 := 
sorry

end max_enclosed_area_perimeter_160_length_twice_width_l312_312344


namespace divisible_by_13_l312_312328

theorem divisible_by_13 (a : ℤ) (h₀ : 0 ≤ a) (h₁ : a ≤ 13) : (51^2015 + a) % 13 = 0 → a = 1 :=
by
  sorry

end divisible_by_13_l312_312328


namespace rhombus_area_of_square_l312_312446

theorem rhombus_area_of_square (h : ∀ (c : ℝ), c = 96) : ∃ (a : ℝ), a = 288 := 
by
  sorry

end rhombus_area_of_square_l312_312446


namespace racetrack_circumference_diff_l312_312398

theorem racetrack_circumference_diff (d_inner d_outer width : ℝ) 
(h1 : d_inner = 55) (h2 : width = 15) (h3 : d_outer = d_inner + 2 * width) : 
  (π * d_outer - π * d_inner) = 30 * π :=
by
  sorry

end racetrack_circumference_diff_l312_312398


namespace total_milks_taken_l312_312128

def total_milks (chocolateMilk strawberryMilk regularMilk : Nat) : Nat :=
  chocolateMilk + strawberryMilk + regularMilk

theorem total_milks_taken :
  total_milks 2 15 3 = 20 :=
by
  sorry

end total_milks_taken_l312_312128


namespace find_quotient_l312_312784

def dividend : ℕ := 55053
def divisor : ℕ := 456
def remainder : ℕ := 333

theorem find_quotient (Q : ℕ) (h : dividend = (divisor * Q) + remainder) : Q = 120 := by
  sorry

end find_quotient_l312_312784


namespace pipe_B_fill_time_l312_312054

theorem pipe_B_fill_time
  (rate_A : ℝ)
  (rate_B : ℝ)
  (t : ℝ)
  (h_rate_A : rate_A = 2 / 75)
  (h_rate_B : rate_B = 1 / t)
  (h_fill_total : 9 * (rate_A + rate_B) + 21 * rate_A = 1) :
  t = 45 := 
sorry

end pipe_B_fill_time_l312_312054


namespace verify_quadratic_solution_l312_312944

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def quadratic_roots : Prop :=
  ∃ (p q : ℕ) (x1 x2 : ℤ), is_prime p ∧ is_prime q ∧ 
  (x1 + x2 = -(p : ℤ)) ∧ (x1 * x2 = (3 * q : ℤ)) ∧ x1 < 0 ∧ x2 < 0 ∧ 
  ((p = 7 ∧ q = 2) ∨ (p = 5 ∧ q = 2))

theorem verify_quadratic_solution : quadratic_roots :=
  by {
    sorry
  }

end verify_quadratic_solution_l312_312944


namespace largest_prime_divisor_l312_312274

theorem largest_prime_divisor : ∃ p : ℕ, Nat.Prime p ∧ p ∣ (17^2 + 60^2) ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ (17^2 + 60^2) → q ≤ p :=
  sorry

end largest_prime_divisor_l312_312274


namespace product_sum_abcd_e_l312_312060

-- Define the individual numbers
def a : ℕ := 12
def b : ℕ := 25
def c : ℕ := 52
def d : ℕ := 21
def e : ℕ := 32

-- Define the sum of the numbers a, b, c, and d
def sum_abcd : ℕ := a + b + c + d

-- Prove that multiplying the sum by e equals 3520
theorem product_sum_abcd_e : sum_abcd * e = 3520 := by
  sorry

end product_sum_abcd_e_l312_312060


namespace power_of_power_evaluation_l312_312735

theorem power_of_power_evaluation : (3^3)^2 = 729 := 
by
  -- Replace this with the actual proof
  sorry

end power_of_power_evaluation_l312_312735


namespace geometric_sequence_ratio_l312_312581

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ)
  (hq_pos : 0 < q)
  (h_geom : ∀ n, a (n + 1) = a n * q) 
  (h_arith : 2 * (1/2) * a 2 = 3 * a 0 + 2 * a 1) :
  (a 10 + a 12) / (a 7 + a 9) = 27 :=
sorry

end geometric_sequence_ratio_l312_312581


namespace evaluate_three_cubed_squared_l312_312714

theorem evaluate_three_cubed_squared : (3^3)^2 = 729 :=
by
  -- Given the property of exponents
  have h : (forall (a m n : ℕ), (a^m)^n = a^(m * n)) := sorry,
  -- Now prove the statement using the given property
  calc
    (3^3)^2 = 3^(3 * 2) : by rw [h 3 3 2]
          ... = 3^6       : by norm_num
          ... = 729       : by norm_num

end evaluate_three_cubed_squared_l312_312714


namespace find_C_l312_312175

noncomputable def A := {x : ℝ | x^2 - 5 * x + 6 = 0}
noncomputable def B (a : ℝ) := {x : ℝ | a * x - 6 = 0}
def C := {a : ℝ | (A ∪ (B a)) = A}

theorem find_C : C = {0, 2, 3} := by
  sorry

end find_C_l312_312175


namespace net_change_in_onions_l312_312034

-- Definitions for the given conditions
def onions_added_by_sara : ℝ := 4.5
def onions_taken_by_sally : ℝ := 5.25
def onions_added_by_fred : ℝ := 9.75

-- Statement of the problem to be proved
theorem net_change_in_onions : 
  onions_added_by_sara - onions_taken_by_sally + onions_added_by_fred = 9 := 
by
  sorry -- hint that proof is required

end net_change_in_onions_l312_312034


namespace quadratic_function_series_sum_l312_312174

open Real

noncomputable def P (x : ℝ) : ℝ := 6 * x^2 - 3 * x + 7

theorem quadratic_function_series_sum :
  (∀ (x : ℝ), 0 < x ∧ x < 1 →
    (∑' n, P n * x^n) = (16 * x^2 - 11 * x + 7) / (1 - x)^3) :=
sorry

end quadratic_function_series_sum_l312_312174


namespace enclosed_area_abs_eq_54_l312_312518

theorem enclosed_area_abs_eq_54 :
  (∃ (x y : ℝ), abs x + abs (3 * y) = 9) → True := 
by
  sorry

end enclosed_area_abs_eq_54_l312_312518


namespace helen_made_56_pies_l312_312481

theorem helen_made_56_pies (pinky_pies total_pies : ℕ) (h_pinky : pinky_pies = 147) (h_total : total_pies = 203) :
  (total_pies - pinky_pies) = 56 :=
by
  sorry

end helen_made_56_pies_l312_312481


namespace words_per_page_l312_312390

theorem words_per_page (p : ℕ) (h1 : 150 * p ≡ 210 [MOD 221]) (h2 : p ≤ 120) : p = 195 := by
  sorry

end words_per_page_l312_312390


namespace regular_polygon_num_sides_l312_312005

def diag_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem regular_polygon_num_sides (n : ℕ) (h : diag_formula n = 20) : n = 8 :=
by
  sorry

end regular_polygon_num_sides_l312_312005


namespace points_in_quadrants_l312_312509

theorem points_in_quadrants (x y : ℝ) (h₁ : y > 3 * x) (h₂ : y > 6 - x) : 
  (0 <= x ∧ 0 <= y) ∨ (x <= 0 ∧ 0 <= y) :=
by
  sorry

end points_in_quadrants_l312_312509


namespace margie_change_l312_312180

theorem margie_change (n_sold n_cost n_paid : ℕ) (h1 : n_sold = 3) (h2 : n_cost = 50) (h3 : n_paid = 500) : 
  n_paid - (n_sold * n_cost) = 350 := by
  sorry

end margie_change_l312_312180


namespace problem_ab_cd_l312_312780

theorem problem_ab_cd
    (a b c d : ℝ)
    (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
    (habcd : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
    (h1 : (a^2012 - c^2012) * (a^2012 - d^2012) = 2012)
    (h2 : (b^2012 - c^2012) * (b^2012 - d^2012) = 2012) :
  (ab)^2012 - (cd)^2012 = -2012 := 
sorry

end problem_ab_cd_l312_312780


namespace sum_A_C_l312_312257

theorem sum_A_C (A B C : ℝ) (h1 : A + B + C = 500) (h2 : B + C = 340) (h3 : C = 40) : A + C = 200 :=
by
  sorry

end sum_A_C_l312_312257


namespace find_angle_A_find_tan_C_l312_312431

-- Import necessary trigonometric identities and basic Lean setup
open Real

-- First statement: Given the dot product condition, find angle A
theorem find_angle_A (A : ℝ) (h1 : cos A + sqrt 3 * sin A = 1) :
  A = 2 * π / 3 := 
sorry

-- Second statement: Given the trigonometric condition, find tan C
theorem find_tan_C (B C : ℝ)
  (h1 : 1 + sin (2 * B) = 2 * (cos B ^ 2 - sin B ^ 2))
  (h2 : B + C = π) :
  tan C = (5 * sqrt 3 - 6) / 3 := 
sorry

end find_angle_A_find_tan_C_l312_312431


namespace coefficient_of_y_l312_312129

theorem coefficient_of_y (x y a : ℝ) (h1 : 7 * x + y = 19) (h2 : x + a * y = 1) (h3 : 2 * x + y = 5) : a = 3 :=
sorry

end coefficient_of_y_l312_312129


namespace sum_largest_and_second_smallest_l312_312221

-- Define the list of numbers
def numbers : List ℕ := [10, 11, 12, 13, 14]

-- Define a predicate to get the largest number
def is_largest (n : ℕ) : Prop := ∀ x ∈ numbers, x ≤ n

-- Define a predicate to get the second smallest number
def is_second_smallest (n : ℕ) : Prop :=
  ∃ a b, (a ∈ numbers ∧ b ∈ numbers ∧ a < b ∧ b < n ∧ ∀ x ∈ numbers, (x < a ∨ x > b))

-- The main goal: To prove that the sum of the largest number and the second smallest number is 25
theorem sum_largest_and_second_smallest : 
  ∃ l s, is_largest l ∧ is_second_smallest s ∧ l + s = 25 := 
sorry

end sum_largest_and_second_smallest_l312_312221


namespace find_integers_correct_l312_312121

noncomputable def find_integers (a b c d : ℤ) : Prop :=
  a + b + c = 6 ∧ a + b + d = 7 ∧ a + c + d = 8 ∧ b + c + d = 9

theorem find_integers_correct (a b c d : ℤ) (h : find_integers a b c d) : a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 :=
by
  sorry

end find_integers_correct_l312_312121


namespace probability_at_least_one_heart_or_king_l312_312237
   
   noncomputable def probability_non_favorable : ℚ := 81 / 169

   theorem probability_at_least_one_heart_or_king :
     1 - probability_non_favorable = 88 / 169 := 
   sorry
   
end probability_at_least_one_heart_or_king_l312_312237


namespace find_lawn_length_l312_312544

theorem find_lawn_length
  (width_lawn : ℕ)
  (road_width : ℕ)
  (cost_total : ℕ)
  (cost_per_sqm : ℕ)
  (total_area_roads : ℕ)
  (area_roads_length : ℕ)
  (area_roads_breadth : ℕ)
  (length_lawn : ℕ) :
  width_lawn = 60 →
  road_width = 10 →
  cost_total = 3600 →
  cost_per_sqm = 3 →
  total_area_roads = cost_total / cost_per_sqm →
  area_roads_length = road_width * length_lawn →
  area_roads_breadth = road_width * (width_lawn - road_width) →
  total_area_roads = area_roads_length + area_roads_breadth →
  length_lawn = 70 :=
by
  intros h_width_lawn h_road_width h_cost_total h_cost_per_sqm h_total_area_roads h_area_roads_length h_area_roads_breadth h_total_area_roads_eq
  sorry

end find_lawn_length_l312_312544


namespace algebraic_expression_value_l312_312136

theorem algebraic_expression_value 
  (x1 x2 : ℝ)
  (h1 : x1^2 - x1 - 2022 = 0)
  (h2 : x2^2 - x2 - 2022 = 0) :
  x1^3 - 2022 * x1 + x2^2 = 4045 :=
by 
  sorry

end algebraic_expression_value_l312_312136


namespace eval_expr_l312_312743

theorem eval_expr : (3^3)^2 = 729 := 
by
  sorry

end eval_expr_l312_312743


namespace mary_total_spent_l312_312797

-- The conditions given in the problem
def cost_berries : ℝ := 11.08
def cost_apples : ℝ := 14.33
def cost_peaches : ℝ := 9.31

-- The theorem to prove the total cost
theorem mary_total_spent : cost_berries + cost_apples + cost_peaches = 34.72 := 
by
  sorry

end mary_total_spent_l312_312797


namespace nonnegative_difference_roots_eq_12_l312_312855

theorem nonnegative_difference_roots_eq_12 :
  ∀ (x : ℝ), (x^2 + 40 * x + 300 = -64) →
  ∃ (r₁ r₂ : ℝ), (x^2 + 40 * x + 364 = 0) ∧ 
  (r₁ = -26 ∧ r₂ = -14)
  ∧ (|r₁ - r₂| = 12) :=
by
  sorry

end nonnegative_difference_roots_eq_12_l312_312855


namespace papers_left_l312_312107

def total_papers_bought : ℕ := 20
def pictures_drawn_today : ℕ := 6
def pictures_drawn_yesterday_before_work : ℕ := 6
def pictures_drawn_yesterday_after_work : ℕ := 6

theorem papers_left :
  total_papers_bought - (pictures_drawn_today + pictures_drawn_yesterday_before_work + pictures_drawn_yesterday_after_work) = 2 := 
by 
  sorry

end papers_left_l312_312107


namespace problem_statement_l312_312261

theorem problem_statement : 15 * 35 + 50 * 15 - 5 * 15 = 1200 := by
  sorry

end problem_statement_l312_312261


namespace max_value_of_sums_l312_312332

noncomputable def max_of_sums (a b c d : ℝ) : ℝ :=
  a^4 + b^4 + c^4 + d^4

theorem max_value_of_sums (a b c d : ℝ) (h : a^3 + b^3 + c^3 + d^3 = 4) :
  max_of_sums a b c d ≤ 16 :=
sorry

end max_value_of_sums_l312_312332


namespace number_of_bowls_l312_312826

theorem number_of_bowls (n : ℕ) (h1 : 12 * 8 = 96) (h2 : 6 * n = 96) : n = 16 :=
by
  -- equations from the conditions
  have h3 : 96 = 96 := by sorry
  exact sorry

end number_of_bowls_l312_312826


namespace at_least_one_heart_or_king_l312_312242

-- Define the conditions
def total_cards := 52
def hearts := 13
def kings := 4
def king_of_hearts := 1
def cards_hearts_or_kings := hearts + kings - king_of_hearts

-- Calculate probabilities based on the above conditions
def probability_not_heart_or_king := 
  1 - (cards_hearts_or_kings / total_cards)

def probability_neither_heart_nor_king :=
  (probability_not_heart_or_king) ^ 2

def probability_at_least_one_heart_or_king :=
  1 - probability_neither_heart_nor_king

-- State the theorem to be proved
theorem at_least_one_heart_or_king : 
  probability_at_least_one_heart_or_king = (88 / 169) :=
by
  sorry

end at_least_one_heart_or_king_l312_312242


namespace gibraltar_initial_population_stable_l312_312507
-- Import necessary libraries

-- Define constants based on conditions
def full_capacity := 300 * 4
def initial_population := (full_capacity / 3) - 100
def population := 300 -- This is the final answer we need to validate

-- The main theorem to prove
theorem gibraltar_initial_population_stable : initial_population = population :=
by 
  -- Proof is skipped as requested
  sorry

end gibraltar_initial_population_stable_l312_312507


namespace sum_tens_units_11_pow_2010_l312_312984

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

def units_digit (n : ℕ) : ℕ :=
  n % 10

def sum_tens_units_digits (n : ℕ) : ℕ :=
  tens_digit n + units_digit n

theorem sum_tens_units_11_pow_2010 :
  sum_tens_units_digits (11 ^ 2010) = 1 :=
sorry

end sum_tens_units_11_pow_2010_l312_312984


namespace solve_inequality_l312_312504

-- Define the function satisfying the given conditions
def f (x : ℝ) : ℝ := sorry

axiom f_functional_eq : ∀ (x y : ℝ), f (x / y) = f x - f y
axiom f_not_zero : ∀ x : ℝ, f x ≠ 0
axiom f_positive : ∀ x : ℝ, x > 1 → f x > 0

-- Define the theorem that proves the inequality given the conditions
theorem solve_inequality (x : ℝ) :
  f x + f (x + 1/2) < 0 ↔ x ∈ (Set.Ioo ( (1 - Real.sqrt 17) / 4 ) 0) ∪ (Set.Ioo 0 ( (1 + Real.sqrt 17) / 4 )) :=
by
  sorry

end solve_inequality_l312_312504


namespace base7_to_base10_l312_312113

theorem base7_to_base10 : 6 * 7^3 + 4 * 7^2 + 2 * 7^1 + 3 * 7^0 = 2271 := by
  sorry

end base7_to_base10_l312_312113


namespace min_frac_sum_l312_312427

open Real

noncomputable def minValue (m n : ℝ) : ℝ := 1 / m + 2 / n

theorem min_frac_sum (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) :
  minValue m n = 3 + 2 * sqrt 2 := by
  sorry

end min_frac_sum_l312_312427


namespace calc_expression_l312_312263

theorem calc_expression : 
  (Real.sqrt 16 - 4 * (Real.sqrt 2) / 2 + abs (- (Real.sqrt 3 * Real.sqrt 6)) + (-1) ^ 2023) = 
  (3 + Real.sqrt 2) :=
by
  sorry

end calc_expression_l312_312263


namespace sides_of_triangle_l312_312227

variable (a b c : ℝ)

theorem sides_of_triangle (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_ineq : (a^2 + b^2 + c^2)^2 > 2*(a^4 + b^4 + c^4)) :
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) :=
  sorry

end sides_of_triangle_l312_312227


namespace clara_total_points_l312_312607

-- Define the constants
def percentage_three_point_shots : ℝ := 0.25
def points_per_successful_three_point_shot : ℝ := 3
def percentage_two_point_shots : ℝ := 0.40
def points_per_successful_two_point_shot : ℝ := 2
def total_attempts : ℕ := 40

-- Define the function to calculate the total score
def total_score (x y : ℕ) : ℝ :=
  (percentage_three_point_shots * points_per_successful_three_point_shot) * x +
  (percentage_two_point_shots * points_per_successful_two_point_shot) * y

-- The proof statement
theorem clara_total_points (x y : ℕ) (h : x + y = total_attempts) : 
  total_score x y = 32 :=
by
  -- This is a placeholder for the actual proof
  sorry

end clara_total_points_l312_312607


namespace calculator_display_exceeds_1000_after_three_presses_l312_312229

-- Define the operation of pressing the squaring key
def square_key (n : ℕ) : ℕ := n * n

-- Define the initial display number
def initial_display : ℕ := 3

-- Prove that after pressing the squaring key 3 times, the display is greater than 1000.
theorem calculator_display_exceeds_1000_after_three_presses : 
  square_key (square_key (square_key initial_display)) > 1000 :=
by
  sorry

end calculator_display_exceeds_1000_after_three_presses_l312_312229


namespace words_on_each_page_l312_312389

theorem words_on_each_page (p : ℕ) (h1 : p ≤ 120) (h2 : 150 * p % 221 = 210) : p = 48 :=
sorry

end words_on_each_page_l312_312389


namespace sqrt_product_eq_225_l312_312646

theorem sqrt_product_eq_225 : (Real.sqrt (5 * 3) * Real.sqrt (3 ^ 3 * 5 ^ 3) = 225) :=
by
  sorry

end sqrt_product_eq_225_l312_312646


namespace find_y_when_x_is_7_l312_312980

theorem find_y_when_x_is_7 (x y : ℝ) (h1 : x * y = 200) (h2 : x = 7) : y = 200 / 7 :=
by
  sorry

end find_y_when_x_is_7_l312_312980


namespace joe_fish_times_sam_l312_312340

-- Define the number of fish Sam has
def sam_fish : ℕ := 7

-- Define the number of fish Harry has
def harry_fish : ℕ := 224

-- Define the number of times Joe has as many fish as Sam
def joe_times_sam (x : ℕ) : Prop :=
  4 * (sam_fish * x) = harry_fish

-- The theorem to prove Joe has 8 times as many fish as Sam
theorem joe_fish_times_sam : ∃ x, joe_times_sam x ∧ x = 8 :=
by
  sorry

end joe_fish_times_sam_l312_312340


namespace city_of_archimedes_schools_l312_312615

noncomputable def numberOfSchools : ℕ := 32

theorem city_of_archimedes_schools :
  ∃ n : ℕ, (∀ s : Set ℕ, s = {45, 68, 113} →
  (∀ x ∈ s, x > 1 → 4 * n = x + 1 → (2 * n ≤ x ∧ 2 * n + 1 ≥ x) ))
  ∧ n = numberOfSchools :=
sorry

end city_of_archimedes_schools_l312_312615


namespace arithmetic_progression_sum_l312_312459

noncomputable def a (n : ℕ) (a1 d : ℤ) : ℤ := a1 + (n - 1) * d
noncomputable def S (n : ℕ) (a1 d : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem arithmetic_progression_sum
    (a1 d : ℤ)
    (h : a 9 a1 d = a 12 a1 d / 2 + 3) :
  S 11 a1 d = 66 := 
by 
  sorry

end arithmetic_progression_sum_l312_312459


namespace sum_of_squares_of_roots_l312_312110

theorem sum_of_squares_of_roots :
  ∀ (x₁ x₂ : ℝ), (∀ a b c : ℝ, (a ≠ 0) →
  6 * x₁ ^ 2 + 5 * x₁ - 4 = 0 ∧ 6 * x₂ ^ 2 + 5 * x₂ - 4 = 0 →
  x₁ ^ 2 + x₂ ^ 2 = 73 / 36) :=
by
  sorry

end sum_of_squares_of_roots_l312_312110


namespace evaluate_three_cubed_squared_l312_312715

theorem evaluate_three_cubed_squared : (3^3)^2 = 729 :=
by
  -- Given the property of exponents
  have h : (forall (a m n : ℕ), (a^m)^n = a^(m * n)) := sorry,
  -- Now prove the statement using the given property
  calc
    (3^3)^2 = 3^(3 * 2) : by rw [h 3 3 2]
          ... = 3^6       : by norm_num
          ... = 729       : by norm_num

end evaluate_three_cubed_squared_l312_312715


namespace buddy_met_boy_students_l312_312665

theorem buddy_met_boy_students (total_students : ℕ) (girl_students : ℕ) (boy_students : ℕ) (h1 : total_students = 123) (h2 : girl_students = 57) : boy_students = 66 :=
by
  sorry

end buddy_met_boy_students_l312_312665


namespace no_solution_for_squares_l312_312187

theorem no_solution_for_squares (x y : ℤ) (hx : x > 0) (hy : y > 0) :
  ¬ ∃ k m : ℤ, x^2 + y + 2 = k^2 ∧ y^2 + 4 * x = m^2 :=
sorry

end no_solution_for_squares_l312_312187


namespace num_valid_constants_m_l312_312791

theorem num_valid_constants_m : 
  ∃ (m1 m2 : ℝ), 
  m1 ≠ m2 ∧ 
  (∃ (a b c d : ℝ), 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
    (1 / 2) * abs (2 * c) * abs (2 * d) = 12 ∧ 
    (c / (2 * d) = 2 ∧ 8 = m1 ∨ 2 * c / d = 8) ∧ 
    (c / (2 * d) = (1 / 2) ∧ (1 / 2) = m2 ∨ 2 * c / d = 2)) ∧
  (∀ (m : ℝ), 
    (m = m1 ∨ m = m2) →
    ∃ (a b c d : ℝ), 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
    (1 / 2) * abs (2 * c) * abs (2 * d) = 12 ∧ 
    (c / (2 * d) = 2 ∨ 2 * c / d = 8) ∧ 
    (c / (2 * d) = (1 / 2) ∨ 2 * c / d = 2)) :=
sorry

end num_valid_constants_m_l312_312791


namespace cost_of_new_game_l312_312641

theorem cost_of_new_game (initial_money : ℕ) (money_left : ℕ) (toy_cost : ℕ) (toy_count : ℕ)
  (h_initial : initial_money = 68) (h_toy_cost : toy_cost = 7) (h_toy_count : toy_count = 3) 
  (h_money_left : money_left = toy_count * toy_cost) :
  initial_money - money_left = 47 :=
by {
  sorry
}

end cost_of_new_game_l312_312641


namespace probability_of_one_defective_l312_312143

theorem probability_of_one_defective :
  (2 : ℕ) ≤ 5 → (0 : ℕ) ≤ 2 → (0 : ℕ) ≤ 3 →
  let total_outcomes := Nat.choose 5 2
  let favorable_outcomes := Nat.choose 3 1 * Nat.choose 2 1
  let probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)
  probability = (3 / 5 : ℚ) :=
by
  intros h1 h2 h3
  let total_outcomes := Nat.choose 5 2
  let favorable_outcomes := Nat.choose 3 1 * Nat.choose 2 1
  let probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)
  have : total_outcomes = 10 := by sorry
  have : favorable_outcomes = 6 := by sorry
  have : probability = (6 / 10 : ℚ) := by sorry
  have : (6 / 10 : ℚ) = (3 / 5 : ℚ) := by sorry
  exact this

end probability_of_one_defective_l312_312143


namespace gcd_pow_sub_one_l312_312058

theorem gcd_pow_sub_one (m n : ℕ) (h1 : m = 2^2024 - 1) (h2 : n = 2^2000 - 1) : Nat.gcd m n = 2^24 - 1 := 
by
  sorry

end gcd_pow_sub_one_l312_312058


namespace prob_hits_exactly_two_expectation_total_score_l312_312364

namespace ShooterProblem

-- Events and conditions
def hits_target_A : Event := sorry
def hits_target_B : Event := sorry

-- Probabilities
def prob_hits_A : ℝ := 3/4
def prob_hits_B : ℝ := 2/3

-- The first query: Probability of hitting exactly two shots
/-- The probability that the shooter hits exactly two shots is 5/16 -/
theorem prob_hits_exactly_two 
  (h_ind_A1_A2 : Independent (hits_target_A 1) (hits_target_A 2)) 
  (h_ind_A_B : Independent (hits_target_A) (hits_target_B)) :
  P (hits_target_A 1 ∧ hits_target_A 2 ∧ ¬hits_target_B ∨ 
     hits_target_A 1 ∧ ¬hits_target_A 2 ∧ hits_target_B ∨ 
     ¬hits_target_A 1 ∧ hits_target_A 2 ∧ hits_target_B) = 5/16 :=
sorry

-- The second query: Mathematical expectation of shooter's total score
def score (a1 a2 b : Bool) : ℝ :=
  (if a1 then 1 else 0) + (if a2 then 1 else 0) + (if b then 2 else 0)

/-- The expectation of the shooter's total score is 11/4 -/
theorem expectation_total_score :
  E[X] = 11/4 :=
sorry

end ShooterProblem

end prob_hits_exactly_two_expectation_total_score_l312_312364


namespace find_m_value_l312_312151

theorem find_m_value (m : ℝ) (h1 : 2 ∈ ({0, m, m^2 - 3 * m + 2} : set ℝ)) : m = 3 :=
sorry

end find_m_value_l312_312151


namespace find_d_in_polynomial_l312_312024

theorem find_d_in_polynomial 
  (a b c d : ℤ) 
  (x1 x2 x3 x4 : ℤ)
  (roots_neg : x1 < 0 ∧ x2 < 0 ∧ x3 < 0 ∧ x4 < 0)
  (h_poly : ∀ x, 
    (x + x1) * (x + x2) * (x + x3) * (x + x4) = 
    x^4 + a * x^3 + b * x^2 + c * x + d)
  (h_sum_eq : a + b + c + d = 2009) :
  d = (x1 * x2 * x3 * x4) :=
by
  sorry

end find_d_in_polynomial_l312_312024


namespace value_at_one_positive_l312_312433

-- Define the conditions
variable {f : ℝ → ℝ} 

-- f is a monotonically increasing function
def monotone_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- f is an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement: proving that f(1) > 0
theorem value_at_one_positive (h1 : monotone_increasing f) (h2 : odd_function f) : f 1 > 0 :=
sorry

end value_at_one_positive_l312_312433


namespace marts_income_percentage_of_juans_l312_312527

variable (T J M : Real)
variable (h1 : M = 1.60 * T)
variable (h2 : T = 0.40 * J)

theorem marts_income_percentage_of_juans : M = 0.64 * J :=
by
  sorry

end marts_income_percentage_of_juans_l312_312527


namespace car_grid_probability_l312_312800

theorem car_grid_probability:
  let m := 11
  let n := 48
  100 * m + n = 1148 := by
  sorry

end car_grid_probability_l312_312800


namespace temperature_difference_l312_312658

variable (highest_temp : ℤ)
variable (lowest_temp : ℤ)

theorem temperature_difference : 
  highest_temp = 2 ∧ lowest_temp = -8 → (highest_temp - lowest_temp = 10) := by
  sorry

end temperature_difference_l312_312658


namespace probability_at_least_one_heart_or_king_l312_312236
   
   noncomputable def probability_non_favorable : ℚ := 81 / 169

   theorem probability_at_least_one_heart_or_king :
     1 - probability_non_favorable = 88 / 169 := 
   sorry
   
end probability_at_least_one_heart_or_king_l312_312236


namespace change_in_expression_l312_312012

variables (x b : ℝ) (hb : 0 < b)

theorem change_in_expression : (b * x)^2 - 5 - (x^2 - 5) = (b^2 - 1) * x^2 :=
by sorry

end change_in_expression_l312_312012


namespace evaluate_exponent_l312_312729

theorem evaluate_exponent : (3^3)^2 = 729 := by
  sorry

end evaluate_exponent_l312_312729


namespace divisible_digit_B_l312_312056

-- Define the digit type as natural numbers within the range 0 to 9.
def digit := {n : ℕ // n <= 9}

-- Define what it means for a number to be even.
def even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Define what it means for a number to be divisible by 3.
def divisible_by_3 (n : ℕ) : Prop := ∃ k, n = 3 * k

-- Define our problem in Lean as properties of the digit B.
theorem divisible_digit_B (B : digit) (h_even : even B.1) (h_div_by_3 : divisible_by_3 (14 + B.1)) : B.1 = 4 :=
sorry

end divisible_digit_B_l312_312056


namespace circumscribed_steiner_ellipse_inscribed_steiner_ellipse_l312_312895

variable {α β γ : ℝ}

/-- The equation of the circumscribed Steiner ellipse in barycentric coordinates -/
theorem circumscribed_steiner_ellipse (h : α + β + γ = 1) :
  β * γ + α * γ + α * β = 0 :=
sorry

/-- The equation of the inscribed Steiner ellipse in barycentric coordinates -/
theorem inscribed_steiner_ellipse (h : α + β + γ = 1) :
  2 * β * γ + 2 * α * γ + 2 * α * β = α^2 + β^2 + γ^2 :=
sorry

end circumscribed_steiner_ellipse_inscribed_steiner_ellipse_l312_312895


namespace circle_equation_and_tangent_lines_l312_312082

theorem circle_equation_and_tangent_lines :
  (∃ (a b r : ℝ), (a = 3) ∧ (b = 4) ∧ (r = 5) ∧ 
  (∀ (x y : ℝ), (x - a)^2 + (y - b)^2 = r^2) ∧
  (y = - (3/4) * x ∨ (x + y + 5 * real.sqrt 2 - 7 = 0) ∨ (x + y - 5 * real.sqrt 2 - 7 = 0))) :=
begin
  sorry
end

end circle_equation_and_tangent_lines_l312_312082


namespace x_squared_plus_y_squared_l312_312917

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 10) : x^2 + y^2 = 29 := 
by
  sorry

end x_squared_plus_y_squared_l312_312917


namespace calculate_product1_calculate_square_l312_312692

theorem calculate_product1 : 100.2 * 99.8 = 9999.96 :=
by
  sorry

theorem calculate_square : 103^2 = 10609 :=
by
  sorry

end calculate_product1_calculate_square_l312_312692


namespace triangle_square_ratio_l312_312093

theorem triangle_square_ratio (s_t s_s : ℕ) (h : 3 * s_t = 4 * s_s) : (s_t : ℚ) / s_s = 4 / 3 := by
  sorry

end triangle_square_ratio_l312_312093


namespace cuboid_volume_l312_312224

variable (length width height : ℕ)

-- Conditions given in the problem
def cuboid_edges := (length = 2) ∧ (width = 5) ∧ (height = 8)

-- Mathematically equivalent statement to be proved
theorem cuboid_volume : cuboid_edges length width height → length * width * height = 80 := by
  sorry

end cuboid_volume_l312_312224


namespace find_complex_number_l312_312864

open Complex

theorem find_complex_number (z : ℂ) (hz : z + Complex.abs z = Complex.ofReal 2 + 8 * Complex.I) : 
z = -15 + 8 * Complex.I := by sorry

end find_complex_number_l312_312864


namespace christmas_trees_in_each_box_l312_312277

theorem christmas_trees_in_each_box
  (T : ℕ)
  (pieces_of_tinsel_in_each_box : ℕ := 4)
  (snow_globes_in_each_box : ℕ := 5)
  (total_boxes : ℕ := 12)
  (total_decorations : ℕ := 120)
  (decorations_per_box : ℕ := pieces_of_tinsel_in_each_box + T + snow_globes_in_each_box)
  (total_decorations_distributed : ℕ := total_boxes * decorations_per_box) :
  total_decorations_distributed = total_decorations → T = 1 := by
  sorry

end christmas_trees_in_each_box_l312_312277


namespace necessarily_positive_b_plus_3c_l312_312485

theorem necessarily_positive_b_plus_3c 
  (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 3) : 
  b + 3 * c > 0 := 
sorry

end necessarily_positive_b_plus_3c_l312_312485


namespace spending_on_clothes_transport_per_month_l312_312673

noncomputable def monthly_spending_on_clothes_transport (S : ℝ) : ℝ :=
  0.2 * S

theorem spending_on_clothes_transport_per_month :
  ∃ (S : ℝ), (monthly_spending_on_clothes_transport S = 1584) ∧
             (12 * S - (12 * 0.6 * S + 12 * monthly_spending_on_clothes_transport S) = 19008) :=
by
  sorry

end spending_on_clothes_transport_per_month_l312_312673


namespace depth_of_pond_l312_312933

theorem depth_of_pond (L W V D : ℝ) (hL : L = 20) (hW : W = 10) (hV : V = 1000) (hV_formula : V = L * W * D) : D = 5 := by
  -- at this point, you could start the proof which involves deriving D from hV and hV_formula using arithmetic rules.
  sorry

end depth_of_pond_l312_312933


namespace error_in_area_l312_312526

theorem error_in_area (s : ℝ) (h : s > 0) :
  let s_measured := 1.02 * s
  let A_actual := s^2
  let A_measured := s_measured^2
  let error := (A_measured - A_actual) / A_actual * 100
  error = 4.04 := by
  sorry

end error_in_area_l312_312526


namespace tan_of_cos_alpha_l312_312758

open Real

theorem tan_of_cos_alpha (α : ℝ) (h1 : cos α = 3 / 5) (h2 : -π < α ∧ α < 0) : tan α = -4 / 3 :=
sorry

end tan_of_cos_alpha_l312_312758


namespace fraction_sum_l312_312260

theorem fraction_sum :
  (3 / 30 : ℝ) + (5 / 300) + (7 / 3000) = 0.119 := by
  sorry

end fraction_sum_l312_312260


namespace union_complements_eq_l312_312816

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem union_complements_eq :
  U = {0, 1, 3, 5, 6, 8} →
  A = {1, 5, 8} →
  B = {2} →
  (U \ A) ∪ B = {0, 2, 3, 6} :=
by
  intros hU hA hB
  rw [hU, hA, hB]
  -- Prove that (U \ A) ∪ B = {0, 2, 3, 6}
  sorry

end union_complements_eq_l312_312816


namespace dog_group_division_l312_312957

theorem dog_group_division:
  let total_dogs := 12
  let group1_size := 4
  let group2_size := 5
  let group3_size := 3
  let Rocky_in_group1 := true
  let Bella_in_group2 := true
  (total_dogs == 12 ∧ group1_size == 4 ∧ group2_size == 5 ∧ group3_size == 3 ∧ Rocky_in_group1 ∧ Bella_in_group2) →
  (∃ ways: ℕ, ways = 4200)
  :=
  sorry

end dog_group_division_l312_312957


namespace sum_remainder_l312_312378

theorem sum_remainder (p q r : ℕ) (hp : p % 15 = 11) (hq : q % 15 = 13) (hr : r % 15 = 14) : 
  (p + q + r) % 15 = 8 :=
by
  sorry

end sum_remainder_l312_312378


namespace find_b_l312_312055

theorem find_b (a b : ℤ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 1) : b = 1 :=
by {
  -- Proof will be filled in here
  sorry
}

end find_b_l312_312055


namespace possible_values_b_count_l312_312494

theorem possible_values_b_count :
    {b : ℤ | b > 0 ∧ 4 ∣ b ∧ b ∣ 24}.to_finset.card = 4 :=
sorry

end possible_values_b_count_l312_312494


namespace lemonade_glasses_l312_312792

theorem lemonade_glasses (total_lemons : ℝ) (lemons_per_glass : ℝ) (glasses : ℝ) :
  total_lemons = 18.0 → lemons_per_glass = 2.0 → glasses = total_lemons / lemons_per_glass → glasses = 9 :=
by
  intro h_total_lemons h_lemons_per_glass h_glasses
  sorry

end lemonade_glasses_l312_312792


namespace eval_exp_l312_312723

theorem eval_exp : (3^3)^2 = 729 := sorry

end eval_exp_l312_312723


namespace probability_of_pink_flower_is_five_over_nine_l312_312363

-- Definitions as per the conditions
def flowersInBagA := 9
def pinkFlowersInBagA := 3
def flowersInBagB := 9
def pinkFlowersInBagB := 7
def probChoosingBag := (1:ℚ) / 2

-- Definition of the probabilities
def probPinkFromA := pinkFlowersInBagA / flowersInBagA
def probPinkFromB := pinkFlowersInBagB / flowersInBagB

-- Total probability calculation using the law of total probability
def probPink := probPinkFromA * probChoosingBag + probPinkFromB * probChoosingBag

-- Statement to be proved
theorem probability_of_pink_flower_is_five_over_nine : probPink = (5:ℚ) / 9 := 
by
  sorry

end probability_of_pink_flower_is_five_over_nine_l312_312363


namespace p_arithmetic_square_root_l312_312950

theorem p_arithmetic_square_root {p : ℕ} (hp : p ≠ 2) (a : ℤ) (ha : a ≠ 0) :
  (∃ x1 x2 : ℤ, x1^2 = a ∧ x2^2 = a ∧ x1 ≠ x2) ∨ ¬ (∃ x : ℤ, x^2 = a) :=
  sorry

end p_arithmetic_square_root_l312_312950


namespace range_of_a_l312_312659

open Set

def p (a : ℝ) := ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0
def q (a : ℝ) := ∀ x : ℝ, x ∈ (Icc 1 2) → x^2 ≥ a

theorem range_of_a (a : ℝ) : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a ∈ (Ioo 1 2 ∪ Iic (-2)) :=
by sorry

end range_of_a_l312_312659


namespace composite_rate_proof_l312_312057

noncomputable def composite_rate (P A : ℝ) (T : ℕ) (X Y Z : ℝ) (R : ℝ) : Prop :=
  let factor := (1 + X / 100) * (1 + Y / 100) * (1 + Z / 100)
  1.375 = factor ∧ (A = P * (1 + R / 100) ^ T)

theorem composite_rate_proof :
  composite_rate 4000 5500 3 X Y Z 11.1 :=
by sorry

end composite_rate_proof_l312_312057


namespace total_colored_hangers_l312_312604

theorem total_colored_hangers (pink_hangers green_hangers : ℕ) (h1 : pink_hangers = 7) (h2 : green_hangers = 4)
  (blue_hangers yellow_hangers : ℕ) (h3 : blue_hangers = green_hangers - 1) (h4 : yellow_hangers = blue_hangers - 1) :
  pink_hangers + green_hangers + blue_hangers + yellow_hangers = 16 :=
by
  sorry

end total_colored_hangers_l312_312604


namespace total_revenue_correct_l312_312511

def sections := 5
def seats_per_section_1_4 := 246
def seats_section_5 := 314
def ticket_price_1_4 := 15
def ticket_price_5 := 20

theorem total_revenue_correct :
  4 * seats_per_section_1_4 * ticket_price_1_4 + seats_section_5 * ticket_price_5 = 21040 :=
by
  sorry

end total_revenue_correct_l312_312511


namespace range_of_a_l312_312908

def p (x : ℝ) : Prop := (1/2 ≤ x ∧ x ≤ 1)

def q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, p x → q x a) ∧ (∃ x : ℝ, q x a ∧ ¬ p x) → 
  (0 ≤ a ∧ a ≤ 1/2) :=
by
  sorry

end range_of_a_l312_312908


namespace total_clothes_washed_l312_312265

theorem total_clothes_washed (cally_white_shirts : ℕ) (cally_colored_shirts : ℕ) (cally_shorts : ℕ) (cally_pants : ℕ) 
                             (danny_white_shirts : ℕ) (danny_colored_shirts : ℕ) (danny_shorts : ℕ) (danny_pants : ℕ) 
                             (total_clothes : ℕ)
                             (hcally : cally_white_shirts = 10 ∧ cally_colored_shirts = 5 ∧ cally_shorts = 7 ∧ cally_pants = 6)
                             (hdanny : danny_white_shirts = 6 ∧ danny_colored_shirts = 8 ∧ danny_shorts = 10 ∧ danny_pants = 6)
                             (htotal : total_clothes = 58) : 
  cally_white_shirts + cally_colored_shirts + cally_shorts + cally_pants + 
  danny_white_shirts + danny_colored_shirts + danny_shorts + danny_pants = total_clothes := 
by {
  sorry
}

end total_clothes_washed_l312_312265


namespace determine_d_l312_312318

theorem determine_d (m n d : ℝ) (p : ℝ) (hp : p = 0.6666666666666666) 
  (h1 : m = 3 * n + 5) (h2 : m + d = 3 * (n + p) + 5) : d = 2 :=
by {
  sorry
}

end determine_d_l312_312318


namespace acute_not_greater_than_right_l312_312879

-- Definitions for conditions
def is_right_angle (α : ℝ) : Prop := α = 90
def is_acute_angle (α : ℝ) : Prop := α < 90

-- Statement to be proved
theorem acute_not_greater_than_right (α : ℝ) (h1 : is_right_angle 90) (h2 : is_acute_angle α) : ¬ (α > 90) :=
by
    sorry

end acute_not_greater_than_right_l312_312879


namespace tree_height_end_of_third_year_l312_312547

theorem tree_height_end_of_third_year (h : ℝ) : 
    (∃ h0 h3 h6 : ℝ, 
      h3 = h0 * 3^3 ∧ 
      h6 = h3 * 2^3 ∧ 
      h6 = 1458) → h3 = 182.25 :=
by sorry

end tree_height_end_of_third_year_l312_312547


namespace dave_more_than_derek_l312_312559

def derek_initial : ℕ := 40
def derek_spent_on_self1 : ℕ := 14
def derek_spent_on_dad : ℕ := 11
def derek_spent_on_self2 : ℕ := 5

def dave_initial : ℕ := 50
def dave_spent_on_mom : ℕ := 7

def derek_remaining : ℕ := derek_initial - (derek_spent_on_self1 + derek_spent_on_dad + derek_spent_on_self2)
def dave_remaining : ℕ := dave_initial - dave_spent_on_mom

theorem dave_more_than_derek : dave_remaining - derek_remaining = 33 :=
by
  -- The proof goes here
  sorry

end dave_more_than_derek_l312_312559


namespace find_b_coefficients_l312_312436

theorem find_b_coefficients (x : ℝ) (b₁ b₂ b₃ b₄ : ℝ) :
  x^4 = (x + 1)^4 + b₁ * (x + 1)^3 + b₂ * (x + 1)^2 + b₃ * (x + 1) + b₄ →
  b₁ = -4 ∧ b₂ = 6 ∧ b₃ = -4 ∧ b₄ = 1 := by
  sorry

end find_b_coefficients_l312_312436


namespace determine_d_value_l312_312563

noncomputable def Q (d : ℚ) (x : ℚ) : ℚ := x^3 + 3 * x^2 + d * x + 8

theorem determine_d_value (d : ℚ) : x - 3 ∣ Q d x → d = -62 / 3 := by
  sorry

end determine_d_value_l312_312563


namespace baker_cakes_total_l312_312407

-- Define the variables corresponding to the conditions
def cakes_sold : ℕ := 145
def cakes_left : ℕ := 72

-- State the theorem to prove that the total number of cakes made is 217
theorem baker_cakes_total : cakes_sold + cakes_left = 217 := 
by 
-- The proof is omitted according to the instructions
sorry

end baker_cakes_total_l312_312407


namespace loss_percentage_l312_312861

-- Definitions of cost price (C) and selling price (S)
def cost_price : ℤ := sorry
def selling_price : ℤ := sorry

-- Given condition: Cost price of 40 articles equals selling price of 25 articles
axiom condition : 40 * cost_price = 25 * selling_price

-- Statement to prove: The merchant made a loss of 20%
theorem loss_percentage (C S : ℤ) (h : 40 * C = 25 * S) : 
  ((S - C) * 100) / C = -20 := 
sorry

end loss_percentage_l312_312861


namespace sampling_methods_used_l312_312609

-- Definitions based on problem conditions
def TotalHouseholds : Nat := 2000
def FarmerHouseholds : Nat := 1800
def WorkerHouseholds : Nat := 100
def IntellectualHouseholds : Nat := TotalHouseholds - FarmerHouseholds - WorkerHouseholds
def SampleSize : Nat := 40

-- The statement of the proof problem
theorem sampling_methods_used
  (N : Nat := TotalHouseholds)
  (F : Nat := FarmerHouseholds)
  (W : Nat := WorkerHouseholds)
  (I : Nat := IntellectualHouseholds)
  (S : Nat := SampleSize)
:
  (1 ∈ [1, 2, 3]) ∧ (2 ∈ [1, 2, 3]) ∧ (3 ∈ [1, 2, 3]) :=
by
  -- Add the proof here
  sorry

end sampling_methods_used_l312_312609


namespace find_interest_rate_l312_312103

-- Define the given conditions
def initial_investment : ℝ := 2200
def additional_investment : ℝ := 1099.9999999999998
def total_investment : ℝ := initial_investment + additional_investment
def desired_income : ℝ := 0.06 * total_investment
def income_from_additional_investment : ℝ := 0.08 * additional_investment
def income_from_initial_investment (r : ℝ) : ℝ := initial_investment * r

-- State the proof problem
theorem find_interest_rate (r : ℝ) 
    (h : desired_income = income_from_additional_investment + income_from_initial_investment r) :
    r = 0.05 :=
sorry

end find_interest_rate_l312_312103


namespace additional_time_due_to_leak_l312_312089

theorem additional_time_due_to_leak 
  (normal_time_per_barrel : ℕ)
  (leak_time_per_barrel : ℕ)
  (barrels : ℕ)
  (normal_duration : normal_time_per_barrel = 3)
  (leak_duration : leak_time_per_barrel = 5)
  (barrels_needed : barrels = 12) :
  (leak_time_per_barrel * barrels - normal_time_per_barrel * barrels) = 24 := 
by
  sorry

end additional_time_due_to_leak_l312_312089


namespace probability_correct_l312_312564

def elenaNameLength : Nat := 5
def markNameLength : Nat := 4
def juliaNameLength : Nat := 5
def totalCards : Nat := elenaNameLength + markNameLength + juliaNameLength

-- Without replacement, drawing three cards from 14 cards randomly
def probabilityThreeDifferentSources : ℚ := 
  (elenaNameLength / totalCards) * (markNameLength / (totalCards - 1)) * (juliaNameLength / (totalCards - 2))

def totalPermutations : Nat := 6  -- EMJ, EJM, MEJ, MJE, JEM, JME

def requiredProbability : ℚ := totalPermutations * probabilityThreeDifferentSources

theorem probability_correct :
  requiredProbability = 25 / 91 := by
  sorry

end probability_correct_l312_312564


namespace probability_heart_or_king_l312_312234

theorem probability_heart_or_king :
  let total_cards := 52
  let hearts := 13
  let kings := 4
  let overlap := 1
  let unique_hearts_or_kings := hearts + kings - overlap
  let non_hearts_or_kings := total_cards - unique_hearts_or_kings
  let p_non_heart_or_king := (non_hearts_or_kings : ℚ) / (total_cards : ℚ)
  let p_non_heart_or_king_twice := p_non_heart_or_king * p_non_heart_or_king
  let p_at_least_one_heart_or_king := 1 - p_non_heart_or_king_twice
  p_at_least_one_heart_or_king = 88 / 169 :=
by
  have total_cards := 52
  have hearts := 13
  have kings := 4
  have overlap := 1
  have unique_hearts_or_kings := hearts + kings - overlap
  have non_hearts_or_kings := total_cards - unique_hearts_or_kings
  have p_non_heart_or_king := (non_hearts_or_kings : ℚ) / (total_cards : ℚ)
  have p_non_heart_or_king_twice := p_non_heart_or_king * p_non_heart_or_king
  have p_at_least_one_heart_or_king := 1 - p_non_heart_or_king_twice
  show p_at_least_one_heart_or_king = 88 / 169
  sorry

end probability_heart_or_king_l312_312234


namespace sum_of_numbers_in_ratio_l312_312977

theorem sum_of_numbers_in_ratio (x : ℝ) (h1 : 8 * x - 3 * x = 20) : 3 * x + 8 * x = 44 :=
by
  sorry

end sum_of_numbers_in_ratio_l312_312977


namespace g_is_increasing_l312_312812

open Real

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * x + a

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x + (a / x) - 2 * a

theorem g_is_increasing (a : ℝ) (h : a < 1) :
  ∀ x y : ℝ, 1 < x → 1 < y → x < y → g x a < g y a := by
  sorry

end g_is_increasing_l312_312812


namespace bricks_for_wall_l312_312671

theorem bricks_for_wall
  (wall_length : ℕ) (wall_height : ℕ) (wall_width : ℕ)
  (brick_length : ℕ) (brick_height : ℕ) (brick_width : ℕ)
  (L_eq : wall_length = 600) (H_eq : wall_height = 400) (W_eq : wall_width = 2050)
  (l_eq : brick_length = 30) (h_eq : brick_height = 12) (w_eq : brick_width = 10)
  : (wall_length * wall_height * wall_width) / (brick_length * brick_height * brick_width) = 136667 :=
by
  sorry

end bricks_for_wall_l312_312671


namespace zero_in_M_l312_312909

-- Define the set M
def M : Set ℕ := {0, 1, 2}

-- State the theorem to be proved
theorem zero_in_M : 0 ∈ M := 
  sorry

end zero_in_M_l312_312909


namespace nail_pierces_one_cardboard_only_l312_312345

/--
Seryozha cut out two identical figures from cardboard. He placed them overlapping
at the bottom of a rectangular box. The bottom turned out to be completely covered. 
A nail was driven into the center of the bottom. Prove that it is possible for the 
nail to pierce one cardboard piece without piercing the other.
-/
theorem nail_pierces_one_cardboard_only 
  (identical_cardboards : Prop)
  (overlapping : Prop)
  (fully_covered_bottom : Prop)
  (nail_center : Prop) 
  : ∃ (layout : Prop), layout ∧ nail_center → nail_pierces_one :=
sorry

end nail_pierces_one_cardboard_only_l312_312345


namespace dropped_test_score_l312_312672

theorem dropped_test_score (A B C D : ℕ) 
  (h1 : A + B + C + D = 280) 
  (h2 : A + B + C = 225) : 
  D = 55 := 
by sorry

end dropped_test_score_l312_312672


namespace power_of_power_evaluation_l312_312736

theorem power_of_power_evaluation : (3^3)^2 = 729 := 
by
  -- Replace this with the actual proof
  sorry

end power_of_power_evaluation_l312_312736


namespace number_of_possible_values_l312_312493

theorem number_of_possible_values (b : ℕ) (hb4 : 4 ∣ b) (hb24 : b ∣ 24) (hpos : 0 < b) : ∃ n, n = 4 :=
by
  sorry

end number_of_possible_values_l312_312493


namespace ratio_proof_l312_312447

theorem ratio_proof (a b c d : ℝ) (h1 : a / b = 20) (h2 : c / b = 5) (h3 : c / d = 1 / 8) : 
  a / d = 1 / 2 :=
by
  sorry

end ratio_proof_l312_312447


namespace train_length_l312_312402

theorem train_length (L : ℕ) 
  (h_tree : L / 120 = L / 200 * 200) 
  (h_platform : (L + 800) / 200 = L / 120) : 
  L = 1200 :=
by
  sorry

end train_length_l312_312402


namespace quadrilateral_area_ABCDEF_l312_312640

theorem quadrilateral_area_ABCDEF :
  ∀ (A B C D E : Type)
  (AC CD AE : ℝ) 
  (angle_ABC angle_ACD : ℝ),
  angle_ABC = 90 ∧
  angle_ACD = 90 ∧
  AC = 20 ∧
  CD = 30 ∧
  AE = 5 →
  ∃ S : ℝ, S = 360 :=
by
  sorry

end quadrilateral_area_ABCDEF_l312_312640


namespace evaluate_exp_power_l312_312716

theorem evaluate_exp_power : (3^3)^2 = 729 := 
by {
  sorry
}

end evaluate_exp_power_l312_312716


namespace pond_volume_extraction_l312_312016

/--
  Let length (l), width (w), and depth (h) be dimensions of a pond.
  Given:
  l = 20,
  w = 10,
  h = 5,
  Prove that the volume of the soil extracted from the pond is 1000 cubic meters.
-/
theorem pond_volume_extraction (l w h : ℕ) (hl : l = 20) (hw : w = 10) (hh : h = 5) :
  l * w * h = 1000 :=
  by
    sorry

end pond_volume_extraction_l312_312016


namespace range_of_a_l312_312134

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h_def : ∀ x, f x = a * Real.log x + 1/2 * x^2)
  (h_ineq : ∀ x1 x2 : ℝ, x1 ≠ x2 → 0 < x1 → 0 < x2 → (f x1 - f x2) / (x1 - x2) > 4) : a > 4 :=
sorry

end range_of_a_l312_312134


namespace tan_x_neg7_l312_312289

theorem tan_x_neg7 (x : ℝ) (h1 : Real.sin (x + π / 4) = 3 / 5) (h2 : Real.sin (x - π / 4) = 4 / 5) : 
  Real.tan x = -7 :=
sorry

end tan_x_neg7_l312_312289


namespace carrie_pays_199_27_l312_312697

noncomputable def carrie_payment : ℝ :=
  let shirts := 8 * 12
  let pants := 4 * 25
  let jackets := 4 * 75
  let skirts := 3 * 30
  let shoes := 2 * 50
  let shirts_discount := 0.20 * shirts
  let jackets_discount := 0.20 * jackets
  let skirts_discount := 0.10 * skirts
  let total_cost := shirts + pants + jackets + skirts + shoes
  let discounted_cost := (shirts - shirts_discount) + (pants) + (jackets - jackets_discount) + (skirts - skirts_discount) + shoes
  let mom_payment := 2 / 3 * discounted_cost
  let carrie_payment := discounted_cost - mom_payment
  carrie_payment

theorem carrie_pays_199_27 : carrie_payment = 199.27 :=
by
  sorry

end carrie_pays_199_27_l312_312697


namespace valid_pin_count_l312_312455

def total_pins : ℕ := 10^5

def restricted_pins (seq : List ℕ) : ℕ :=
  if seq = [3, 1, 4, 1] then 10 else 0

def valid_pins (seq : List ℕ) : ℕ :=
  total_pins - restricted_pins seq

theorem valid_pin_count :
  valid_pins [3, 1, 4, 1] = 99990 :=
by
  sorry

end valid_pin_count_l312_312455


namespace val_total_money_l312_312852

theorem val_total_money : 
  ∀ (nickels_initial dimes_initial nickels_found : ℕ),
    nickels_initial = 20 →
    dimes_initial = 3 * nickels_initial →
    nickels_found = 2 * nickels_initial →
    (nickels_initial * 5 + dimes_initial * 10 + nickels_found * 5) / 100 = 9 :=
by
  intros nickels_initial dimes_initial nickels_found h1 h2 h3
  sorry

end val_total_money_l312_312852


namespace shortest_remaining_side_l312_312683

theorem shortest_remaining_side (a b c : ℝ) (h₁ : a = 5) (h₂ : c = 13) (h₃ : a^2 + b^2 = c^2) : b = 12 :=
by
  rw [h₁, h₂] at h₃
  sorry

end shortest_remaining_side_l312_312683


namespace geoff_total_spending_l312_312424

def price_day1 : ℕ := 60
def pairs_day1 : ℕ := 2
def price_per_pair_day1 : ℕ := price_day1 / pairs_day1

def multiplier_day2 : ℕ := 3
def price_per_pair_day2 : ℕ := price_per_pair_day1 * 3 / 2
def discount_day2 : Real := 0.10
def cost_before_discount_day2 : ℕ := multiplier_day2 * price_per_pair_day2
def cost_after_discount_day2 : Real := cost_before_discount_day2 * (1 - discount_day2)

def multiplier_day3 : ℕ := 5
def price_per_pair_day3 : ℕ := price_per_pair_day1 * 2
def sales_tax_day3 : Real := 0.08
def cost_before_tax_day3 : ℕ := multiplier_day3 * price_per_pair_day3
def cost_after_tax_day3 : Real := cost_before_tax_day3 * (1 + sales_tax_day3)

def total_cost : Real := price_day1 + cost_after_discount_day2 + cost_after_tax_day3

theorem geoff_total_spending : total_cost = 505.50 := by
  sorry

end geoff_total_spending_l312_312424


namespace equivalence_l312_312484

-- Non-computable declaration to avoid the computational complexity.
noncomputable def is_isosceles_right_triangle (x₁ x₂ : Complex) : Prop :=
  x₂ = x₁ * Complex.I ∨ x₁ = x₂ * Complex.I

-- Definition of the polynomial roots condition.
def roots_form_isosceles_right_triangle (a b : Complex) : Prop :=
  ∃ x₁ x₂ : Complex,
    x₁ + x₂ = -a ∧
    x₁ * x₂ = b ∧
    is_isosceles_right_triangle x₁ x₂

-- Main theorem statement that matches the mathematical equivalency.
theorem equivalence (a b : Complex) : a^2 = 2*b ∧ b ≠ 0 ↔ roots_form_isosceles_right_triangle a b :=
sorry

end equivalence_l312_312484


namespace bicycle_speed_l312_312247

theorem bicycle_speed
  (dist : ℝ := 15) -- Distance between the school and the museum
  (bus_factor : ℝ := 1.5) -- Bus speed is 1.5 times the bicycle speed
  (time_diff : ℝ := 1 / 4) -- Bicycle students leave 1/4 hour earlier
  (x : ℝ) -- Speed of bicycles
  (h : (dist / x) - (dist / (bus_factor * x)) = time_diff) :
  x = 20 :=
sorry

end bicycle_speed_l312_312247


namespace log_geometric_sequence_l312_312930

theorem log_geometric_sequence :
  ∀ (a : ℕ → ℝ), (∀ n, 0 < a n) → (∃ r : ℝ, ∀ n, a (n + 1) = a n * r) →
  a 2 * a 18 = 16 → Real.logb 2 (a 10) = 2 :=
by
  intros a h_positive h_geometric h_condition
  sorry

end log_geometric_sequence_l312_312930


namespace g_eq_g_g_l312_312331

noncomputable def g (x : ℝ) : ℝ := x^2 - 4 * x + 1

theorem g_eq_g_g (x : ℝ) : 
  g (g x) = g x ↔ x = 2 + Real.sqrt ((11 + 2 * Real.sqrt 21) / 2) 
             ∨ x = 2 - Real.sqrt ((11 + 2 * Real.sqrt 21) / 2) 
             ∨ x = 2 + Real.sqrt ((11 - 2 * Real.sqrt 21) / 2) 
             ∨ x = 2 - Real.sqrt ((11 - 2 * Real.sqrt 21) / 2) := 
by
  sorry

end g_eq_g_g_l312_312331


namespace Megan_finish_all_problems_in_8_hours_l312_312755

theorem Megan_finish_all_problems_in_8_hours :
  ∀ (math_problems spelling_problems problems_per_hour : ℕ),
    math_problems = 36 →
    spelling_problems = 28 →
    problems_per_hour = 8 →
    (math_problems + spelling_problems) / problems_per_hour = 8 :=
by
  intros
  sorry

end Megan_finish_all_problems_in_8_hours_l312_312755


namespace evaluate_exponent_l312_312730

theorem evaluate_exponent : (3^3)^2 = 729 := by
  sorry

end evaluate_exponent_l312_312730


namespace twenty_five_point_zero_six_million_in_scientific_notation_l312_312100

theorem twenty_five_point_zero_six_million_in_scientific_notation :
  (25.06e6 : ℝ) = 2.506 * 10^7 :=
by
  -- The proof would go here, but we use sorry to skip the proof.
  sorry

end twenty_five_point_zero_six_million_in_scientific_notation_l312_312100


namespace pyramid_prism_sum_l312_312460

-- Definitions based on conditions
structure Prism :=
  (vertices : ℕ)
  (edges : ℕ)
  (faces : ℕ)

-- The initial cylindrical-prism object
noncomputable def initial_prism : Prism :=
  { vertices := 8,
    edges := 10,
    faces := 5 }

-- Structure for Pyramid Addition
structure PyramidAddition :=
  (new_vertices : ℕ)
  (new_edges : ℕ)
  (new_faces : ℕ)

noncomputable def pyramid_addition : PyramidAddition := 
  { new_vertices := 1,
    new_edges := 4,
    new_faces := 4 }

-- Function to add pyramid to the prism
noncomputable def add_pyramid (prism : Prism) (pyramid : PyramidAddition) : Prism :=
  { vertices := prism.vertices + pyramid.new_vertices,
    edges := prism.edges + pyramid.new_edges,
    faces := prism.faces - 1 + pyramid.new_faces }

-- The resulting prism after adding the pyramid
noncomputable def resulting_prism := add_pyramid initial_prism pyramid_addition

-- Proof problem statement
theorem pyramid_prism_sum : 
  resulting_prism.vertices + resulting_prism.edges + resulting_prism.faces = 31 :=
by sorry

end pyramid_prism_sum_l312_312460


namespace maximize_container_volume_l312_312254

theorem maximize_container_volume :
  ∃ x : ℝ, 0 < x ∧ x < 24 ∧ ∀ y : ℝ, 0 < y ∧ y < 24 → 
  ( (48 - 2 * x)^2 * x ≥ (48 - 2 * y)^2 * y ) ∧ x = 8 :=
sorry

end maximize_container_volume_l312_312254


namespace prob_three_primes_in_four_rolls_l312_312409

-- Define the basic properties
def is_prime (n : ℕ) : Prop := 
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11

def prob_prime : ℚ := 5 / 12
def prob_not_prime : ℚ := 7 / 12

def choose (n k : ℕ) : ℕ :=
  nat.factorial n / (nat.factorial k * nat.factorial (n - k))

lemma binom_coefficient : choose 4 3 = 4 := 
  by simp [choose, nat.factorial]

theorem prob_three_primes_in_four_rolls : 
  (4 * ((prob_prime ^ 3) * prob_not_prime) = (875 / 5184)) :=
  by tidy; sorry -- The actual proof is omitted

end prob_three_primes_in_four_rolls_l312_312409


namespace find_angle_sum_l312_312021

theorem find_angle_sum (c d : ℝ) (hc : 0 < c ∧ c < π/2) (hd : 0 < d ∧ d < π/2)
    (h1 : 4 * (Real.cos c)^2 + 3 * (Real.sin d)^2 = 1)
    (h2 : 4 * Real.sin (2 * c) = 3 * Real.cos (2 * d)) :
    2 * c + 3 * d = π / 2 :=
by
  sorry

end find_angle_sum_l312_312021


namespace sum_primes_less_than_20_l312_312373

theorem sum_primes_less_than_20 : (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 := sorry

end sum_primes_less_than_20_l312_312373


namespace isosceles_triangle_sides_l312_312786

theorem isosceles_triangle_sides (A B C : Type) [plus : A → A → A] [le : A → A → Prop] [mul : A → A → A] [div : A → A → A] [zero : A] [one : A]
  (triangle_iso : ∀ (x y z : A), plus x y z = 60 → le (plus x z) y → le (plus y z) x → le (plus x y) z → x = y)
  (per_60 : plus (plus A B) C = 60)
  (medians_inter_inscribed : A → A → A → A)
  (centroid_on_incircle : (medians_inter_inscribed A B C) = Inscribed_circle)
  (A : A)
  (B : A)
  (C : A) : 
  A = 25 ∧ B = 25 ∧ C = 10 := 
sorry

end isosceles_triangle_sides_l312_312786


namespace buratino_loss_l312_312688

def buratino_dollars_lost (x y : ℕ) : ℕ := 5 * y - 3 * x

theorem buratino_loss :
  ∃ (x y : ℕ), x + y = 50 ∧ 3 * y - 2 * x = 0 ∧ buratino_dollars_lost x y = 10 :=
by {
  sorry
}

end buratino_loss_l312_312688


namespace area_gray_region_correct_l312_312267

def center_C : ℝ × ℝ := (3, 5)
def radius_C : ℝ := 3
def center_D : ℝ × ℝ := (9, 5)
def radius_D : ℝ := 3

noncomputable def area_gray_region : ℝ :=
  let rectangle_area := (center_D.1 - center_C.1) * (center_C.2 - (center_C.2 - radius_C))
  let sector_area := (1 / 4) * radius_C ^ 2 * Real.pi
  rectangle_area - 2 * sector_area

theorem area_gray_region_correct :
  area_gray_region = 18 - 9 / 2 * Real.pi :=
by
  sorry

end area_gray_region_correct_l312_312267


namespace cross_product_example_l312_312283

def vector_cross (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := 
  (u.2.1 * v.2.2 - u.2.2 * v.2.1, 
   u.2.2 * v.1 - u.1 * v.2.2, 
   u.1 * v.1 - u.2.1 * v.1)
   
theorem cross_product_example : 
  vector_cross (4, 3, -7) (2, 0, 5) = (15, -34, -6) :=
by
  -- The proof will go here
  sorry

end cross_product_example_l312_312283


namespace positive_integer_pairs_l312_312282

theorem positive_integer_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  a^b = b^(a^2) ↔ (a = 1 ∧ b = 1) ∨ (a = 2 ∧ b = 16) ∨ (a = 3 ∧ b = 27) :=
by sorry

end positive_integer_pairs_l312_312282


namespace num_ways_to_distribute_7_balls_in_4_boxes_l312_312591

def num_ways_to_distribute_balls (balls boxes : ℕ) : ℕ :=
  -- Implement the function to calculate the number of ways here, but we'll keep it as a placeholder for now.
  sorry

theorem num_ways_to_distribute_7_balls_in_4_boxes : 
  num_ways_to_distribute_balls 7 4 = 3 := 
sorry

end num_ways_to_distribute_7_balls_in_4_boxes_l312_312591


namespace redistribution_amount_l312_312754

theorem redistribution_amount
    (earnings : Fin 5 → ℕ)
    (h : earnings = ![18, 22, 30, 35, 45]) :
    (earnings 4 - ((earnings 0 + earnings 1 + earnings 2 + earnings 3 + earnings 4) / 5)) = 15 :=
by
  sorry

end redistribution_amount_l312_312754


namespace distinct_pairs_count_l312_312585

theorem distinct_pairs_count : 
  (∃ (s : Finset (ℕ × ℕ)), (∀ p ∈ s, ∃ (a b : ℕ), 1 ≤ a ∧ 1 ≤ b ∧ a + b = 40 ∧ p = (a, b)) ∧ s.card = 39) := sorry

end distinct_pairs_count_l312_312585


namespace distance_from_home_to_high_school_l312_312467

theorem distance_from_home_to_high_school 
  (total_mileage track_distance d : ℝ)
  (h_total_mileage : total_mileage = 10)
  (h_track : track_distance = 4)
  (h_eq : d + d + track_distance = total_mileage) :
  d = 3 :=
by sorry

end distance_from_home_to_high_school_l312_312467


namespace correct_operation_l312_312521

-- Define the conditions as hypotheses
variable (a : ℝ)

-- A: \(a^2 \cdot a = a^3\)
def condition_A : Prop := a^2 * a = a^3

-- B: \((a^3)^3 = a^6\)
def condition_B : Prop := (a^3)^3 = a^6

-- C: \(a^3 + a^3 = a^5\)
def condition_C : Prop := a^3 + a^3 = a^5

-- D: \(a^6 \div a^2 = a^3\)
def condition_D : Prop := a^6 / a^2 = a^3

-- Proof that only condition A is correct:
theorem correct_operation : condition_A a ∧ ¬condition_B a ∧ ¬condition_C a ∧ ¬condition_D a :=
by
  sorry  -- Actual proofs would go here

end correct_operation_l312_312521


namespace logs_per_tree_is_75_l312_312890

-- Definitions
def logsPerDay : Nat := 5

def totalDays : Nat := 30 + 31 + 31 + 28

def totalLogs (burnRate : Nat) (days : Nat) : Nat :=
  burnRate * days

def treesNeeded : Nat := 8

def logsPerTree (totalLogs : Nat) (numTrees : Nat) : Nat :=
  totalLogs / numTrees

-- Theorem statement to prove the number of logs per tree
theorem logs_per_tree_is_75 : logsPerTree (totalLogs logsPerDay totalDays) treesNeeded = 75 :=
  by
  sorry

end logs_per_tree_is_75_l312_312890


namespace triangle_sides_inequality_l312_312596

theorem triangle_sides_inequality
  {a b c : ℝ} (h₁ : a + b + c = 1) (h₂ : a > 0) (h₃ : b > 0) (h₄ : c > 0)
  (h₅ : a + b > c) (h₆ : a + c > b) (h₇ : b + c > a) :
  a^2 + b^2 + c^2 + 4 * a * b * c < 1 / 2 :=
by
  -- We would place the proof here if it were required
  sorry

end triangle_sides_inequality_l312_312596


namespace completion_days_for_B_l312_312866

-- Conditions
def A_completion_days := 20
def B_completion_days (x : ℕ) := x
def project_completion_days := 20
def A_work_days := project_completion_days - 10
def B_work_days := project_completion_days
def A_work_rate := 1 / A_completion_days
def B_work_rate (x : ℕ) := 1 / B_completion_days x
def combined_work_rate (x : ℕ) := A_work_rate + B_work_rate x
def A_project_completed := A_work_days * A_work_rate
def B_project_remaining (x : ℕ) := 1 - A_project_completed
def B_project_completion (x : ℕ) := B_work_days * B_work_rate x

-- Proof statement
theorem completion_days_for_B (x : ℕ) 
  (h : B_project_completion x = B_project_remaining x ∧ combined_work_rate x > 0) :
  x = 40 :=
sorry

end completion_days_for_B_l312_312866


namespace interior_angle_second_quadrant_l312_312771

theorem interior_angle_second_quadrant (α : ℝ) (h1 : 0 < α ∧ α < π) (h2 : Real.sin α * Real.tan α < 0) : 
  π / 2 < α ∧ α < π :=
by
  sorry

end interior_angle_second_quadrant_l312_312771


namespace part1_part2_l312_312135

-- Definition of the function f
def f (a x : ℝ) : ℝ := |a * x - 2| - |x + 2|

-- First Proof Statement: Inequality for a = 2
theorem part1 : ∀ x : ℝ, - (1 : ℝ) / 3 ≤ x ∧ x ≤ 5 → f 2 x ≤ 1 :=
by
  sorry

-- Second Proof Statement: Range for a such that -4 ≤ f(x) ≤ 4 for all x ∈ ℝ
theorem part2 : ∀ x : ℝ, -4 ≤ f a x ∧ f a x ≤ 4 ↔ a = 1 ∨ a = -1 :=
by
  sorry

end part1_part2_l312_312135


namespace avg_age_assist_coaches_l312_312500

-- Define the conditions given in the problem

def total_members := 50
def avg_age_total := 22
def girls := 30
def boys := 15
def coaches := 5
def avg_age_girls := 18
def avg_age_boys := 20
def head_coaches := 3
def assist_coaches := 2
def avg_age_head_coaches := 30

-- Define the target theorem to prove
theorem avg_age_assist_coaches : 
  (avg_age_total * total_members - avg_age_girls * girls - avg_age_boys * boys - avg_age_head_coaches * head_coaches) / assist_coaches = 85 := 
  by
    sorry

end avg_age_assist_coaches_l312_312500


namespace orthocenter_ABC_l312_312590

structure Point2D :=
  (x : ℝ)
  (y : ℝ)

def A : Point2D := ⟨5, -1⟩
def B : Point2D := ⟨4, -8⟩
def C : Point2D := ⟨-4, -4⟩

def isOrthocenter (H : Point2D) (A B C : Point2D) : Prop := sorry  -- Define this properly according to the geometric properties in actual formalization.

theorem orthocenter_ABC : ∃ H : Point2D, isOrthocenter H A B C ∧ H = ⟨3, -5⟩ := 
by 
  sorry  -- Proof omitted

end orthocenter_ABC_l312_312590


namespace other_x_intercept_l312_312881

-- Define the foci of the ellipse
def F1 : ℝ × ℝ := (0, 3)
def F2 : ℝ × ℝ := (4, 0)

-- Define the property of the ellipse where the sum of distances to the foci is constant
def ellipse_property (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in
  Real.sqrt (x^2 + (y - 3)^2) + Real.sqrt ((x - 4)^2 + y^2) = 7

-- Define the point on x-axis for intersection
def is_x_intercept (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in y = 0

-- Full property to be proved: the other point of intersection with the x-axis
theorem other_x_intercept : ∃ (P : ℝ × ℝ), ellipse_property P ∧ is_x_intercept P ∧ P = (56 / 11, 0) := by
  sorry

end other_x_intercept_l312_312881


namespace relay_race_time_l312_312996

theorem relay_race_time (M S J T : ℕ) 
(hJ : J = 30)
(hS : S = J + 10)
(hM : M = 2 * S)
(hT : T = M - 7) : 
M + S + J + T = 223 :=
by sorry

end relay_race_time_l312_312996


namespace hall_volume_l312_312989

theorem hall_volume (l w : ℕ) (h : ℕ) 
    (cond1 : l = 18)
    (cond2 : w = 9)
    (cond3 : (2 * l * w) = (2 * l * h + 2 * w * h)) : 
    (l * w * h = 972) :=
by
  rw [cond1, cond2] at cond3
  have h_eq : h = 324 / 54 := sorry
  rw [h_eq]
  norm_num
  sorry

end hall_volume_l312_312989


namespace monotonically_decreasing_interval_l312_312141

noncomputable def f (x : ℝ) : ℝ :=
  (2 * Real.exp 2) * Real.exp (x - 2) - 2 * x + 1/2 * x^2

theorem monotonically_decreasing_interval :
  ∀ x : ℝ, x < 0 → ((2 * Real.exp x - 2 + x) < 0) :=
by
  sorry

end monotonically_decreasing_interval_l312_312141


namespace second_fisherman_more_fish_l312_312783

-- Defining the conditions
def total_days : ℕ := 213
def first_fisherman_rate : ℕ := 3
def second_fisherman_rate1 : ℕ := 1
def second_fisherman_rate2 : ℕ := 2
def second_fisherman_rate3 : ℕ := 4
def days_rate1 : ℕ := 30
def days_rate2 : ℕ := 60
def days_rate3 : ℕ := total_days - (days_rate1 + days_rate2)

-- Calculating the total number of fish caught by both fishermen
def total_fish_first_fisherman : ℕ := first_fisherman_rate * total_days
def total_fish_second_fisherman : ℕ := (second_fisherman_rate1 * days_rate1) + 
                                        (second_fisherman_rate2 * days_rate2) + 
                                        (second_fisherman_rate3 * days_rate3)

-- Theorem stating the difference in the number of fish caught
theorem second_fisherman_more_fish : (total_fish_second_fisherman - total_fish_first_fisherman) = 3 := 
by
  sorry

end second_fisherman_more_fish_l312_312783


namespace sovereign_states_upper_bound_l312_312038

theorem sovereign_states_upper_bound (n : ℕ) (k : ℕ) : 
  (∃ (lines : ℕ) (border_stop_moving : Prop) (countries_disappear : Prop)
     (create_un : Prop) (total_countries : ℕ),
        (lines = n)
        ∧ (border_stop_moving = true)
        ∧ (countries_disappear = true)
        ∧ (create_un = true)
        ∧ (total_countries = k)) 
  → k ≤ (n^3 + 5*n) / 6 + 1 := 
sorry

end sovereign_states_upper_bound_l312_312038


namespace cookies_on_ninth_plate_l312_312661

-- Define the geometric sequence
def cookies_on_plate (n : ℕ) : ℕ :=
  2 * 2^(n - 1)

-- State the theorem
theorem cookies_on_ninth_plate : cookies_on_plate 9 = 512 :=
by
  sorry

end cookies_on_ninth_plate_l312_312661


namespace intersection_is_2_to_inf_l312_312911

-- Define the set A
def setA (x : ℝ) : Prop :=
 x > 1

-- Define the set B
def setB (y : ℝ) : Prop :=
 ∃ x : ℝ, y = Real.sqrt (x^2 + 2*x + 5)

-- Define the intersection of A and B
def setIntersection : Set ℝ :=
{ y | setA y ∧ setB y }

-- Statement to prove the intersection
theorem intersection_is_2_to_inf : setIntersection = { y | y ≥ 2 } :=
sorry -- Proof is omitted

end intersection_is_2_to_inf_l312_312911


namespace value_of_f_at_13_over_2_l312_312579

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_of_f_at_13_over_2
  (h1 : ∀ x : ℝ , f (-x) = -f (x))
  (h2 : ∀ x : ℝ , f (x - 2) = f (x + 2))
  (h3 : ∀ x : ℝ, 0 < x ∧ x < 2 → f (x) = -x^2) :
  f (13 / 2) = 9 / 4 :=
sorry

end value_of_f_at_13_over_2_l312_312579


namespace work_completion_time_l312_312097

theorem work_completion_time (x : ℕ) (h1 : ∀ B, ∀ A, A = 2 * B) (h2 : (1/x + 1/(2*x)) * 4 = 1) : x = 12 := 
sorry

end work_completion_time_l312_312097


namespace largest_integer_2010_divides_2010_factorial_square_l312_312701

noncomputable def largest_integer_dividing_factorial_square (n : ℕ) : ℕ :=
  let prime_factors_2010 := [2, 3, 5, 67]
  let k := prime_factors_2010.map (λ p, (Nat.floor (2010/ p)) + (Nat.floor (2010/ (p * p))))
  2 * k.foldr Nat.add 0

theorem largest_integer_2010_divides_2010_factorial_square : largest_integer_dividing_factorial_square 2010 = 60 :=
by sorry

end largest_integer_2010_divides_2010_factorial_square_l312_312701


namespace xy_squared_sum_l312_312919

theorem xy_squared_sum {x y : ℝ} (h1 : (x + y)^2 = 49) (h2 : x * y = 10) : x^2 + y^2 = 29 :=
by
  sorry

end xy_squared_sum_l312_312919


namespace arctan_sum_eq_pi_div_two_l312_312413

theorem arctan_sum_eq_pi_div_two : Real.arctan (3 / 7) + Real.arctan (7 / 3) = Real.pi / 2 := 
sorry

end arctan_sum_eq_pi_div_two_l312_312413


namespace total_pages_written_l312_312480

-- Define the conditions
def timeMon : ℕ := 60  -- Minutes on Monday
def rateMon : ℕ := 30  -- Minutes per page on Monday

def timeTue : ℕ := 45  -- Minutes on Tuesday
def rateTue : ℕ := 15  -- Minutes per page on Tuesday

def pagesWed : ℕ := 5  -- Pages written on Wednesday

-- Function to compute pages written based on time and rate
def pages_written (time rate : ℕ) : ℕ := time / rate

-- Define the theorem to be proved
theorem total_pages_written :
  pages_written timeMon rateMon + pages_written timeTue rateTue + pagesWed = 10 :=
sorry

end total_pages_written_l312_312480


namespace equal_roots_h_l312_312583

theorem equal_roots_h (h : ℝ) : (∀ x : ℝ, 3 * x^2 - 4 * x + (h / 3) = 0) -> h = 4 :=
by 
  sorry

end equal_roots_h_l312_312583


namespace part_a_part_b_l312_312020

-- Step d: Lean statements for the proof problems
theorem part_a (p : ℕ) (hp : Nat.Prime p) : ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ (a^2 + b^2 + 2018) % p = 0 :=
by {
  sorry
}

theorem part_b (p : ℕ) (hp : Nat.Prime p) : (∃ a b : ℕ, 0 < a ∧ 0 < b ∧ (a^2 + b^2 + 2018) % p = 0 ∧ a % p ≠ 0 ∧ b % p ≠ 0) ↔ p ≠ 3 :=
by {
  sorry
}

end part_a_part_b_l312_312020


namespace skiing_ratio_l312_312820

theorem skiing_ratio (S : ℕ) (H1 : 4000 ≤ 12000) (H2 : S + 4000 = 12000) : S / 4000 = 2 :=
by {
  sorry
}

end skiing_ratio_l312_312820


namespace souvenir_cost_l312_312885

def total_souvenirs : ℕ := 1000
def total_cost : ℝ := 220
def unknown_souvenirs : ℕ := 400
def known_cost : ℝ := 0.20

theorem souvenir_cost :
  ∃ x : ℝ, x = 0.25 ∧ total_cost = unknown_souvenirs * x + (total_souvenirs - unknown_souvenirs) * known_cost :=
by
  sorry

end souvenir_cost_l312_312885


namespace simplify_neg_expression_l312_312520

variable (a b c : ℝ)

theorem simplify_neg_expression : 
  - (a - (b - c)) = -a + b - c :=
sorry

end simplify_neg_expression_l312_312520


namespace find_f_4_l312_312299

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_4 : (∀ x : ℝ, f (x / 2 - 1) = 2 * x + 3) → f 4 = 23 :=
by
  sorry

end find_f_4_l312_312299


namespace parabola_focus_l312_312961

theorem parabola_focus (a : ℝ) (h : a ≠ 0) : ∃ q : ℝ, q = 1/(4*a) ∧ (0, q) = (0, 1/(4*a)) :=
by
  sorry

end parabola_focus_l312_312961


namespace student_solved_correctly_l312_312401

-- Problem conditions as definitions
def sums_attempted : Nat := 96

def sums_correct (x : Nat) : Prop :=
  let sums_wrong := 3 * x
  x + sums_wrong = sums_attempted

-- Lean statement to prove
theorem student_solved_correctly (x : Nat) (h : sums_correct x) : x = 24 :=
  sorry

end student_solved_correctly_l312_312401


namespace probability_above_80_probability_passing_l312_312931

theorem probability_above_80 {P : ℝ → ℝ} 
  (H1 : P(90 < real.pos) = 0.18)
  (H2 : ∀ x, 80 ≤ x ∧ x ≤ 89 → P(x) = 0.51)
  (H3 : ∀ x, 70 ≤ x ∧ x ≤ 79 → P(x) = 0.15)
  (H4 : ∀ x, 60 ≤ x ∧ x ≤ 69 → P(x) = 0.09) :
  P(80 < real.pos) = 0.69 :=
by sorry

theorem probability_passing {P : ℝ → ℝ}
  (H1 : P(90 < real.pos) = 0.18)
  (H2 : ∀ x, 80 ≤ x ∧ x ≤ 89 → P(x) = 0.51)
  (H3 : ∀ x, 70 ≤ x ∧ x ≤ 79 → P(x) = 0.15)
  (H4 : ∀ x, 60 ≤ x ∧ x ≤ 69 → P(x) = 0.09) :
  P(60 < real.pos) = 0.93 :=
by sorry

end probability_above_80_probability_passing_l312_312931


namespace number_of_bowls_l312_312840

theorem number_of_bowls (n : ℕ) :
  (∀ (b : ℕ), b > 0) →
  (∀ (a : ℕ), ∃ (k : ℕ), true) →
  (8 * 12 = 96) →
  (6 * n = 96) →
  n = 16 :=
by
  intros h1 h2 h3 h4
  sorry

end number_of_bowls_l312_312840


namespace number_of_bookshelves_l312_312173

def total_space : ℕ := 400
def reserved_space : ℕ := 160
def shelf_space : ℕ := 80

theorem number_of_bookshelves : (total_space - reserved_space) / shelf_space = 3 := by
  sorry

end number_of_bookshelves_l312_312173


namespace no_real_roots_of_quadratic_l312_312423

theorem no_real_roots_of_quadratic (k : ℝ) (h : k ≠ 0) : ¬ ∃ x : ℝ, x^2 + k * x + 2 * k^2 = 0 :=
by sorry

end no_real_roots_of_quadratic_l312_312423


namespace min_vases_required_l312_312974

theorem min_vases_required (carnations roses tulips lilies : ℕ)
  (flowers_in_A flowers_in_B flowers_in_C : ℕ) 
  (total_flowers : ℕ) 
  (h_carnations : carnations = 10) 
  (h_roses : roses = 25) 
  (h_tulips : tulips = 15) 
  (h_lilies : lilies = 20)
  (h_flowers_in_A : flowers_in_A = 4) 
  (h_flowers_in_B : flowers_in_B = 6) 
  (h_flowers_in_C : flowers_in_C = 8)
  (h_total_flowers : total_flowers = carnations + roses + tulips + lilies) :
  total_flowers = 70 → 
  (exists vases_A vases_B vases_C : ℕ, 
    vases_A = 0 ∧ 
    vases_B = 1 ∧ 
    vases_C = 8 ∧ 
    total_flowers = vases_A * flowers_in_A + vases_B * flowers_in_B + vases_C * flowers_in_C) :=
by
  intros
  sorry

end min_vases_required_l312_312974


namespace problem_l312_312589

def f (x : ℝ) := 5 * x^3

theorem problem : f 2012 + f (-2012) = 0 := 
by
  sorry

end problem_l312_312589


namespace negation_of_universal_prop_l312_312903

-- Define the proposition p
def p : Prop := ∀ x : ℝ, Real.sin x ≤ 1

-- Define the negation of p
def neg_p : Prop := ∃ x : ℝ, Real.sin x > 1

-- The theorem stating the equivalence
theorem negation_of_universal_prop : ¬p ↔ neg_p := 
by sorry

end negation_of_universal_prop_l312_312903


namespace ellipse_x_intercept_l312_312880

variable (P : ℝ × ℝ) (F1 : ℝ × ℝ) (F2 : ℝ × ℝ)
variable (x : ℝ)

-- Given conditions
def focuses : F1 = (0, 3) ∧ F2 = (4, 0) := sorry

def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  dist P F1 + dist P F2 = 8

def x_intercept_on_x_axis (x : ℝ) : Prop := 
  x ≥ 0 ∧ point_on_ellipse (x, 0)

-- Question translation into Lean statement
theorem ellipse_x_intercept : 
  focuses ∧ x_intercept_on_x_axis x → x = 55/16 := by
  intros
  sorry

end ellipse_x_intercept_l312_312880


namespace engineer_progress_l312_312099

theorem engineer_progress (x : ℕ) : 
  ∀ (road_length_in_km : ℝ) 
    (total_days : ℕ) 
    (initial_men : ℕ) 
    (completed_work_in_km : ℝ) 
    (additional_men : ℕ) 
    (new_total_men : ℕ) 
    (remaining_work_in_km : ℝ) 
    (remaining_days : ℕ),
    road_length_in_km = 10 → 
    total_days = 300 → 
    initial_men = 30 → 
    completed_work_in_km = 2 → 
    additional_men = 30 → 
    new_total_men = 60 → 
    remaining_work_in_km = 8 → 
    remaining_days = total_days - x →
  (4 * (total_days - x) = 8 * x) →
  x = 100 :=
by
  intros road_length_in_km total_days initial_men completed_work_in_km additional_men new_total_men remaining_work_in_km remaining_days
  intros h1 h2 h3 h4 h5 h6 h7 h8 h_eqn
  -- Proof
  sorry

end engineer_progress_l312_312099


namespace jenny_change_l312_312321

def cost_per_page : ℝ := 0.10
def pages_per_essay : ℝ := 25
def num_essays : ℝ := 7
def cost_per_pen : ℝ := 1.50
def num_pens : ℝ := 7
def amount_paid : ℝ := 40.00

theorem jenny_change : 
  let cost_of_printing := num_essays * pages_per_essay * cost_per_page in
  let cost_of_pens := num_pens * cost_per_pen in
  let total_cost := cost_of_printing + cost_of_pens in
  amount_paid - total_cost = 12.00 :=
by
  -- Definitions
  let cost_of_printing := num_essays * pages_per_essay * cost_per_page
  let cost_of_pens := num_pens * cost_per_pen
  let total_cost := cost_of_printing + cost_of_pens

  -- Proof
  sorry

end jenny_change_l312_312321


namespace new_avg_weight_l312_312501

theorem new_avg_weight (A B C D E : ℝ) (h1 : (A + B + C) / 3 = 84) (h2 : A = 78) 
(h3 : (B + C + D + E) / 4 = 79) (h4 : E = D + 6) : 
(A + B + C + D) / 4 = 80 :=
by
  sorry

end new_avg_weight_l312_312501


namespace find_AM_l312_312804

-- Definitions (conditions)
variables {A M B : ℝ}
variable  (collinear : A ≤ M ∧ M ≤ B ∨ B ≤ M ∧ M ≤ A ∨ A ≤ B ∧ B ≤ M)
          (h1 : abs (M - A) = 2 * abs (M - B)) 
          (h2 : abs (A - B) = 6)

-- Proof problem statement
theorem find_AM : (abs (M - A) = 4) ∨ (abs (M - A) = 12) :=
by 
  sorry

end find_AM_l312_312804


namespace probability_three_primes_l312_312408

def prime_between_one_and_twelve := {2, 3, 5, 7, 11}

theorem probability_three_primes :
  (∃ (p : ℚ), p = (3500 : ℚ) / 20736) :=
begin
  -- conditions
  let faces := 12,
  let primes_count := 5,
  let dice := 4,
  let successes := 3,

  -- probabilities
  let p_prime := primes_count / faces,
  let p_not_prime := 1 - p_prime,
  let binom_coeff := nat.choose dice successes,

  -- calculation
  let probability := binom_coeff * p_prime^successes * p_not_prime^(dice - successes),
  
  -- desired outcome
  use probability,
  norm_cast,
  norm_num,
  simp,
end

end probability_three_primes_l312_312408


namespace part1_part2_min_part2_max_part3_l312_312145

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - 2 * a / x - 3 * Real.log x

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := a + 2 * a / (x^2) - 3 / x

theorem part1 (a : ℝ) : f' a 1 = 0 -> a = 1 := sorry

noncomputable def f1 (x : ℝ) : ℝ := x - 2 / x - 3 * Real.log x

noncomputable def f1' (x : ℝ) : ℝ := 1 + 2 / (x^2) - 3 / x

theorem part2_min (h_a : 1 = 1) : 
    ∀ (x : ℝ), (1 ≤ x) ∧ (x ≤ Real.exp 1) -> 
    (f1 2 <= f1 x) := sorry

theorem part2_max (h_a : 1 = 1) : 
    ∀ (x : ℝ), (1 ≤ x) ∧ (x ≤ Real.exp 1) ->
    (f1 x <= f1 1) := sorry

theorem part3 (a : ℝ) : 
    (∀ (x : ℝ), x > 0 -> f' a x ≥ 0) -> a ≥ (3 * Real.sqrt 2) / 4 := sorry

end part1_part2_min_part2_max_part3_l312_312145


namespace total_earnings_correct_l312_312072

-- Definitions for the conditions
def price_per_bracelet := 5
def price_for_two_bracelets := 8
def initial_bracelets := 30
def earnings_from_selling_at_5_each := 60

-- Variables to store intermediate calculations
def bracelets_sold_at_5_each := earnings_from_selling_at_5_each / price_per_bracelet
def remaining_bracelets := initial_bracelets - bracelets_sold_at_5_each
def pairs_sold_at_8_each := remaining_bracelets / 2
def earnings_from_pairs := pairs_sold_at_8_each * price_for_two_bracelets
def total_earnings := earnings_from_selling_at_5_each + earnings_from_pairs

-- The theorem stating that Zayne made $132 in total
theorem total_earnings_correct :
  total_earnings = 132 :=
sorry

end total_earnings_correct_l312_312072


namespace contradiction_assumption_l312_312859

theorem contradiction_assumption (x y : ℝ) (h1 : x > y) : ¬ (x^3 ≤ y^3) := 
by
  sorry

end contradiction_assumption_l312_312859


namespace solve_for_b_l312_312046

theorem solve_for_b :
  (∀ (x y : ℝ), 4 * y - 3 * x + 2 = 0) →
  (∀ (x y : ℝ), 2 * y + b * x - 1 = 0) →
  (∃ b : ℝ, b = 8 / 3) := 
by
  sorry

end solve_for_b_l312_312046


namespace problem1_solution_problem2_solution_l312_312438

def f (c a b : ℝ) (x : ℝ) : ℝ := |(c * x + a)| + |(c * x - b)|
def g (c : ℝ) (x : ℝ) : ℝ := |(x - 2)| + c

noncomputable def sol_set_eq1 := {x : ℝ | -1 / 2 ≤ x ∧ x ≤ 3 / 2}
noncomputable def range_a_eq2 := {a : ℝ | a ≤ -2 ∨ a ≥ 0}

-- Problem (1)
theorem problem1_solution : ∀ (x : ℝ), f 2 1 3 x - 4 = 0 ↔ x ∈ sol_set_eq1 := 
by
  intro x
  sorry -- Proof to be filled in

-- Problem (2)
theorem problem2_solution : 
  ∀ x_1 : ℝ, ∃ x_2 : ℝ, g 1 x_2 = f 1 0 1 x_1 ↔ a ∈ range_a_eq2 :=
by
  intro x_1
  sorry -- Proof to be filled in

end problem1_solution_problem2_solution_l312_312438


namespace x_squared_plus_y_squared_l312_312916

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 10) : x^2 + y^2 = 29 := 
by
  sorry

end x_squared_plus_y_squared_l312_312916


namespace train_length_is_150_meters_l312_312681

def train_speed_kmph : ℝ := 68
def man_speed_kmph : ℝ := 8
def passing_time_sec : ℝ := 8.999280057595392

noncomputable def length_of_train : ℝ :=
  let relative_speed_kmph := train_speed_kmph - man_speed_kmph
  let relative_speed_mps := (relative_speed_kmph * 1000) / 3600
  relative_speed_mps * passing_time_sec

theorem train_length_is_150_meters (train_speed_kmph man_speed_kmph passing_time_sec : ℝ) :
  train_speed_kmph = 68 → man_speed_kmph = 8 → passing_time_sec = 8.999280057595392 →
  length_of_train = 150 :=
by
  intros h1 h2 h3
  simp [length_of_train, h1, h2, h3]
  sorry

end train_length_is_150_meters_l312_312681


namespace boat_travel_l312_312228

theorem boat_travel (T_against T_with : ℝ) (V_b D V_c : ℝ) 
  (hT_against : T_against = 10) 
  (hT_with : T_with = 6) 
  (hV_b : V_b = 12)
  (hD1 : D = (V_b - V_c) * T_against)
  (hD2 : D = (V_b + V_c) * T_with) :
  V_c = 3 ∧ D = 90 :=
by
  sorry

end boat_travel_l312_312228


namespace evaluate_exp_power_l312_312719

theorem evaluate_exp_power : (3^3)^2 = 729 := 
by {
  sorry
}

end evaluate_exp_power_l312_312719


namespace constant_value_l312_312901

noncomputable def find_constant (p q : ℚ) (h : p / q = 4 / 5) : ℚ :=
    let C := 0.5714285714285714 - (2 * q - p) / (2 * q + p)
    C

theorem constant_value (p q : ℚ) (h : p / q = 4 / 5) :
    find_constant p q h = 0.14285714285714285 := by
    sorry

end constant_value_l312_312901


namespace investment_total_amount_l312_312884

noncomputable def compoundedInvestment (principal : ℝ) (rate : ℝ) (tax : ℝ) (years : ℕ) : ℝ :=
let yearlyNetInterest := principal * rate * (1 - tax)
let rec calculate (year : ℕ) (accumulated : ℝ) : ℝ :=
  if year = 0 then accumulated else
    let newPrincipal := accumulated + yearlyNetInterest
    calculate (year - 1) newPrincipal
calculate years principal

theorem investment_total_amount :
  let finalAmount := compoundedInvestment 15000 0.05 0.10 4
  round finalAmount = 17607 :=
by
  sorry

end investment_total_amount_l312_312884


namespace money_left_in_wallet_l312_312663

def initial_amount := 106
def spent_supermarket := 31
def spent_showroom := 49

theorem money_left_in_wallet : initial_amount - spent_supermarket - spent_showroom = 26 := by
  sorry

end money_left_in_wallet_l312_312663


namespace number_of_subsets_sum_of_min_and_max_is_11_l312_312305

/-- Prove the number of nonempty subsets of {1, 2, ..., 10} that have the property
    that the sum of their largest and smallest element is 11 is 341. --/
theorem number_of_subsets_sum_of_min_and_max_is_11 :
  let s := {1, 2, ..., 10} in
  ∃ n, (∀ (t : Finset ℕ), t ⊆ s → t.Nonempty → (t.min' t.nonempty_witness + t.max' t.nonempty_witness = 11) → ∑ (k = 0) ^ 4, 4 ^ k = 341) := 
by
  sorry

end number_of_subsets_sum_of_min_and_max_is_11_l312_312305


namespace vacation_total_cost_l312_312167

def plane_ticket_cost (per_person_cost : ℕ) (num_people : ℕ) : ℕ :=
  num_people * per_person_cost

def hotel_stay_cost (per_person_per_day_cost : ℕ) (num_people : ℕ) (num_days : ℕ) : ℕ :=
  num_people * per_person_per_day_cost * num_days

def total_vacation_cost (plane_ticket_cost : ℕ) (hotel_stay_cost : ℕ) : ℕ :=
  plane_ticket_cost + hotel_stay_cost

theorem vacation_total_cost :
  let per_person_plane_ticket_cost := 24
  let per_person_hotel_cost := 12
  let num_people := 2
  let num_days := 3
  let plane_cost := plane_ticket_cost per_person_plane_ticket_cost num_people
  let hotel_cost := hotel_stay_cost per_person_hotel_cost num_people num_days
  total_vacation_cost plane_cost hotel_cost = 120 := by
  sorry

end vacation_total_cost_l312_312167


namespace eval_exp_l312_312725

theorem eval_exp : (3^3)^2 = 729 := sorry

end eval_exp_l312_312725


namespace find_A_l312_312288

variable {a b : ℝ}

theorem find_A (h : (5 * a + 3 * b)^2 = (5 * a - 3 * b)^2 + A) : A = 60 * a * b :=
sorry

end find_A_l312_312288


namespace arithmetic_sequence_sum_l312_312934

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (a4_eq_3 : a 4 = 3) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 21 :=
by
  sorry

end arithmetic_sequence_sum_l312_312934


namespace wire_cut_l312_312080

theorem wire_cut (x : ℝ) (h₁ : x + (5/2) * x = 14) : x = 4 :=
by
  sorry

end wire_cut_l312_312080


namespace circle_area_l312_312043

/-!

# Problem: Prove that the area of the circle defined by the equation \( x^2 + y^2 - 2x + 4y + 1 = 0 \) is \( 4\pi \).
-/

theorem circle_area : 
  (∃ x y : ℝ, x^2 + y^2 - 2*x + 4*y + 1 = 0) →
  ∃ (A : ℝ), A = 4 * Real.pi := 
by
  sorry

end circle_area_l312_312043


namespace chickens_on_farm_are_120_l312_312819

-- Given conditions
def Number_of_hens : ℕ := 52
def Difference_hens_roosters : ℕ := 16

-- Define the number of roosters based on the conditions
def Number_of_roosters : ℕ := Number_of_hens + Difference_hens_roosters

-- The total number of chickens is the sum of hens and roosters
def Total_number_of_chickens : ℕ := Number_of_hens + Number_of_roosters

-- Prove that the total number of chickens is 120
theorem chickens_on_farm_are_120 : Total_number_of_chickens = 120 := by
  -- leave this part unimplemented for proof.
  -- The steps would involve computing the values based on definitions
  sorry

end chickens_on_farm_are_120_l312_312819


namespace fish_population_l312_312677

theorem fish_population (x : ℕ) : 
  (1: ℝ) / 45 = (100: ℝ) / ↑x -> x = 1125 :=
by
  sorry

end fish_population_l312_312677


namespace additional_discount_percentage_l312_312250

theorem additional_discount_percentage
  (MSRP : ℝ)
  (p : ℝ)
  (d : ℝ)
  (sale_price : ℝ)
  (H1 : MSRP = 45.0)
  (H2 : p = 0.30)
  (H3 : d = MSRP - (p * MSRP))
  (H4 : d = 31.50)
  (H5 : sale_price = 25.20) :
  sale_price = d - (0.20 * d) :=
by
  sorry

end additional_discount_percentage_l312_312250


namespace k_greater_than_inv_e_l312_312291

theorem k_greater_than_inv_e (k : ℝ) (x : ℝ) (hx_pos : 0 < x) (hcond : k * (Real.exp (k * x) + 1) - (1 + (1 / x)) * Real.log x > 0) : 
  k > 1 / Real.exp 1 :=
sorry

end k_greater_than_inv_e_l312_312291


namespace find_k_l312_312144

theorem find_k (k r s : ℝ) 
  (h1 : r + s = -k) 
  (h2 : r * s = 12) 
  (h3 : (r + 7) + (s + 7) = k) : 
  k = 7 := by 
  sorry

end find_k_l312_312144


namespace rational_points_coloring_l312_312341

def is_rational_point (p : ℚ × ℚ) : Prop :=
∃ (a b c d : ℤ), b > 0 ∧ d > 0 ∧ Int.gcd a b = 1 ∧ Int.gcd c d = 1 ∧ p = (a / b, c / d)

theorem rational_points_coloring (n : ℕ) (hn : 0 < n) : 
  ∃ (coloring : ℚ × ℚ → ℕ), 
  (∀ (p : ℚ × ℚ), is_rational_point p → coloring p < n) ∧
  (∀ (p1 p2 : ℚ × ℚ), is_rational_point p1 → is_rational_point p2 → p1 ≠ p2 → 
    ∃ q : ℚ × ℚ, is_rational_point q ∧ q ∈ segment ℚ p1 p2 ∧ 
    (∀ k : ℕ, k < n → q ≠ p1 ∧ q ≠ p2 → coloring q = k)) :=
sorry

end rational_points_coloring_l312_312341


namespace intercept_form_l312_312666

theorem intercept_form (x y : ℝ) : 2 * x - 3 * y - 4 = 0 ↔ x / 2 + y / (-4/3) = 1 := sorry

end intercept_form_l312_312666


namespace all_are_knights_l312_312053

-- Definitions for inhabitants as either knights or knaves
inductive Inhabitant
| Knight : Inhabitant
| Knave : Inhabitant

open Inhabitant

-- Functions that determine if an inhabitant is a knight or a knave
def is_knight (x : Inhabitant) : Prop :=
  x = Knight

def is_knave (x : Inhabitant) : Prop :=
  x = Knave

-- Given conditions
axiom A : Inhabitant
axiom B : Inhabitant
axiom C : Inhabitant

axiom statement_A : is_knight A → is_knight B
axiom statement_B : is_knight B → (is_knight A → is_knight C)

-- The proof goal
theorem all_are_knights : is_knight A ∧ is_knight B ∧ is_knight C := by
  sorry

end all_are_knights_l312_312053


namespace parabola_opening_downwards_l312_312293

theorem parabola_opening_downwards (a : ℝ) :
  (∀ x, 0 < x ∧ x < 3 → ax^2 - 2 * a * x + 3 > 0) → -1 < a ∧ a < 0 :=
by 
  intro h
  sorry

end parabola_opening_downwards_l312_312293


namespace sum_of_prime_factors_1729728_l312_312983

def prime_factors_sum (n : ℕ) : ℕ := 
  -- Suppose that a function defined to calculate the sum of distinct prime factors
  -- In a practical setting, you would define this function or use an existing library
  sorry 

theorem sum_of_prime_factors_1729728 : prime_factors_sum 1729728 = 36 :=
by {
  -- Proof would go here
  sorry
}

end sum_of_prime_factors_1729728_l312_312983


namespace triangle_area_l312_312406

variables {k : ℝ}
variables {A B C D P Q N M O : EuclideanSpace ℝ (Fin 3)}

-- Definition of a parallelogram
def is_parallelogram (A B C D : EuclideanSpace ℝ (Fin 3)) :=
  collinear ({A, B, C}) ∧ collinear ({B, C, D}) ∧ collinear ({C, D, A}) ∧ collinear ({D, A, B})

-- Area of parallelogram ABCD
def area_of_parallelogram (A B C D : EuclideanSpace ℝ (Fin 3)) : ℝ := k

-- Midpoints
def is_midpoint (N : EuclideanSpace ℝ (Fin 3)) (X Y : EuclideanSpace ℝ (Fin 3)) :=
  2 • N = X + Y

-- Conditions based on the problem
def problem_conditions (A B C D P Q N M O : EuclideanSpace ℝ (Fin 3)) : Prop :=
  is_parallelogram A B C D ∧
  is_midpoint N B C ∧
  ∃ λ : ℝ, P = A + λ • (B - A) ∧ ∃ μ : ℝ, M = 1 / 2 • (A + D) ∧
  ∃ ν : ℝ, Q = A + ν • (B - A) ∧
  ∃ ξ : ℝ, O = A + ξ • (P - A)

-- Question to prove
theorem triangle_area (conditions : problem_conditions A B C D P Q N M O) : 
  area_of_triangle Q P O = (9 / 8) * k := 
sorry

end triangle_area_l312_312406


namespace zayne_total_revenue_l312_312070

-- Defining the constants and conditions
def price_per_bracelet := 5
def deal_price := 8
def initial_bracelets := 30
def revenue_from_five_dollar_sales := 60

-- Calculating number of bracelets sold for $5 each
def bracelets_sold_five_dollars := revenue_from_five_dollar_sales / price_per_bracelet

-- Calculating remaining bracelets after selling some for $5 each
def remaining_bracelets := initial_bracelets - bracelets_sold_five_dollars

-- Calculating number of pairs sold at two for $8
def pairs_sold := remaining_bracelets / 2

-- Calculating revenue from selling pairs
def revenue_from_deal_sales := pairs_sold * deal_price

-- Total revenue calculation
def total_revenue := revenue_from_five_dollar_sales + revenue_from_deal_sales

-- Theorem to prove the total revenue is $132
theorem zayne_total_revenue : total_revenue = 132 := by
  sorry

end zayne_total_revenue_l312_312070


namespace eval_expr_l312_312742

theorem eval_expr : (3^3)^2 = 729 := 
by
  sorry

end eval_expr_l312_312742


namespace work_days_of_a_l312_312076

variable (da wa wb wc : ℕ)
variable (hcp : 3 * wc = 5 * wa)
variable (hbw : 4 * wc = 5 * wb)
variable (hwc : wc = 100)
variable (hear : 60 * da + 9 * 80 + 4 * 100 = 1480)

theorem work_days_of_a : da = 6 :=
by
  sorry

end work_days_of_a_l312_312076


namespace speed_of_man_l312_312531

theorem speed_of_man :
  let L := 500 -- Length of the train in meters
  let t := 29.997600191984642 -- Time in seconds
  let V_train_kmh := 63 -- Speed of train in km/hr
  let V_train := (63 * 1000) / 3600 -- Speed of train converted to m/s
  let V_relative := L / t -- Relative speed of train w.r.t man
  
  V_train - V_relative = 0.833 := by
  sorry

end speed_of_man_l312_312531


namespace relay_race_time_l312_312997

theorem relay_race_time (M S J T : ℕ) 
(hJ : J = 30)
(hS : S = J + 10)
(hM : M = 2 * S)
(hT : T = M - 7) : 
M + S + J + T = 223 :=
by sorry

end relay_race_time_l312_312997


namespace smallest_ab_41503_539_l312_312752

noncomputable def find_smallest_ab : (ℕ × ℕ) :=
  let a := 41503
  let b := 539
  (a, b)

theorem smallest_ab_41503_539 (a b : ℕ) (h : 7 * a^3 = 11 * b^5) (ha : a > 0) (hb : b > 0) :
  (a = 41503 ∧ b = 539) :=
  by
    -- Add sorry to skip the proof
    sorry

end smallest_ab_41503_539_l312_312752


namespace circumradius_of_sector_l312_312092

noncomputable def R_circumradius (θ : ℝ) (r : ℝ) := r / (2 * Real.sin (θ / 2))

theorem circumradius_of_sector (r : ℝ) (θ : ℝ) (hθ : θ = 120) (hr : r = 8) :
  R_circumradius θ r = (8 * Real.sqrt 3) / 3 :=
by
  rw [hθ, hr, R_circumradius]
  sorry

end circumradius_of_sector_l312_312092


namespace complement_union_complement_intersection_complementA_intersect_B_l312_312910

def setA (x : ℝ) : Prop := 3 ≤ x ∧ x < 7
def setB (x : ℝ) : Prop := 2 < x ∧ x < 10

theorem complement_union (x : ℝ) : ¬(setA x ∨ setB x) ↔ x ≤ 2 ∨ x ≥ 10 := sorry

theorem complement_intersection (x : ℝ) : ¬(setA x ∧ setB x) ↔ x < 3 ∨ x ≥ 7 := sorry

theorem complementA_intersect_B (x : ℝ) : (¬setA x ∧ setB x) ↔ (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10) := sorry

end complement_union_complement_intersection_complementA_intersect_B_l312_312910


namespace a_plus_b_eq_11_l312_312964

noncomputable def f (a b x : ℝ) : ℝ := x^3 + 3 * a * x^2 + b * x + a^2

theorem a_plus_b_eq_11 (a b : ℝ) 
  (h1 : ∀ x, f a b x ≤ f a b (-1))
  (h2 : f a b (-1) = 0) 
  : a + b = 11 :=
sorry

end a_plus_b_eq_11_l312_312964


namespace eval_exp_l312_312727

theorem eval_exp : (3^3)^2 = 729 := sorry

end eval_exp_l312_312727


namespace opposite_of_neg_six_is_six_l312_312362

theorem opposite_of_neg_six_is_six : ∃ x, -6 + x = 0 ∧ x = 6 := by
  use 6
  split
  · rfl
  · rfl

end opposite_of_neg_six_is_six_l312_312362


namespace min_birthday_employees_wednesday_l312_312385

theorem min_birthday_employees_wednesday :
  ∀ (employees : ℕ) (n : ℕ), 
  employees = 50 → 
  n ≥ 1 →
  ∃ (x : ℕ), 6 * x + (x + n) = employees ∧ x + n ≥ 8 :=
by
  sorry

end min_birthday_employees_wednesday_l312_312385


namespace number_of_balls_l312_312932

noncomputable def totalBalls (frequency : ℚ) (yellowBalls : ℕ) : ℚ :=
  yellowBalls / frequency

theorem number_of_balls (h : totalBalls 0.3 6 = 20) : true :=
by
  sorry

end number_of_balls_l312_312932


namespace friend_spent_more_than_you_l312_312522

-- Define the total amount spent by both
def total_spent : ℤ := 19

-- Define the amount spent by your friend
def friend_spent : ℤ := 11

-- Define the amount spent by you
def you_spent : ℤ := total_spent - friend_spent

-- Define the difference in spending
def difference_in_spending : ℤ := friend_spent - you_spent

-- Prove that the difference in spending is $3
theorem friend_spent_more_than_you : difference_in_spending = 3 :=
by
  sorry

end friend_spent_more_than_you_l312_312522


namespace desks_built_by_carpenters_l312_312674

theorem desks_built_by_carpenters (h : 2 * 2.5 * r ≥ 2 * r) : 4 * 5 * r ≥ 8 * r :=
by
  sorry

end desks_built_by_carpenters_l312_312674


namespace additional_telephone_lines_l312_312928

def telephone_lines_increase : ℕ :=
  let lines_six_digits := 9 * 10^5
  let lines_seven_digits := 9 * 10^6
  lines_seven_digits - lines_six_digits

theorem additional_telephone_lines : telephone_lines_increase = 81 * 10^5 :=
by
  sorry

end additional_telephone_lines_l312_312928


namespace primes_div_order_l312_312633

theorem primes_div_order (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h : q ∣ 3^p - 2^p) : p ∣ q - 1 :=
sorry

end primes_div_order_l312_312633


namespace largest_possible_gcd_l312_312050

theorem largest_possible_gcd (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 221) : ∃ d, Nat.gcd a b = d ∧ d = 17 :=
sorry

end largest_possible_gcd_l312_312050


namespace initial_violet_balloons_l312_312793

-- Define initial conditions and variables
def red_balloons := 4
def violet_balloons_lost := 3
def current_violet_balloons := 4

-- Define the theorem we want to prove
theorem initial_violet_balloons (red_balloons : ℕ) (violet_balloons_lost : ℕ) (current_violet_balloons : ℕ) : 
  red_balloons = 4 → violet_balloons_lost = 3 → current_violet_balloons = 4 → (current_violet_balloons + violet_balloons_lost) = 7 :=
by
  intros
  sorry

end initial_violet_balloons_l312_312793


namespace paving_rate_l312_312358

theorem paving_rate
  (length : ℝ) (width : ℝ) (total_cost : ℝ)
  (h_length : length = 5.5)
  (h_width : width = 3.75)
  (h_total_cost : total_cost = 16500) :
  total_cost / (length * width) = 800 := by
  sorry

end paving_rate_l312_312358


namespace hyperbola_standard_equation_correct_l312_312125

-- Define the initial values given in conditions
def a : ℝ := 12
def b : ℝ := 5
def c : ℝ := 4

-- Define the hyperbola equation form based on conditions and focal properties
noncomputable def hyperbola_standard_equation : Prop :=
  let a2 := (8 / 5)
  let b2 := (72 / 5)
  (∀ x y : ℝ, y^2 / a2 - x^2 / b2 = 1)

-- State the final problem as a theorem
theorem hyperbola_standard_equation_correct :
  ∀ x y : ℝ, y^2 / (8 / 5) - x^2 / (72 / 5) = 1 :=
by
  sorry

end hyperbola_standard_equation_correct_l312_312125


namespace smallest_d_value_l312_312252

theorem smallest_d_value : 
  ∃ d : ℝ, (d ≥ 0) ∧ (dist (0, 0) (4 * Real.sqrt 5, d + 5) = 4 * d) ∧ ∀ d' : ℝ, (d' ≥ 0) ∧ (dist (0, 0) (4 * Real.sqrt 5, d' + 5) = 4 * d') → (3 ≤ d') → d = 3 := 
by
  sorry

end smallest_d_value_l312_312252


namespace simplify_and_evaluate_expr_l312_312191

theorem simplify_and_evaluate_expr 
  (x : ℝ) 
  (h : x = 1/2) : 
  (2 * x - 1) ^ 2 - (3 * x + 1) * (3 * x - 1) + 5 * x * (x - 1) = -5 / 2 := 
by
  sorry

end simplify_and_evaluate_expr_l312_312191


namespace cricket_game_initial_overs_l312_312456

theorem cricket_game_initial_overs
    (run_rate_initial : ℝ)
    (run_rate_remaining : ℝ)
    (remaining_overs : ℕ)
    (target_score : ℝ)
    (initial_overs : ℕ) :
    run_rate_initial = 3.2 →
    run_rate_remaining = 5.25 →
    remaining_overs = 40 →
    target_score = 242 →
    initial_overs = 10 :=
by
  intros h1 h2 h3 h4
  sorry

end cricket_game_initial_overs_l312_312456


namespace average_speed_is_55_l312_312119

theorem average_speed_is_55 
  (initial_reading : ℕ) (final_reading : ℕ) (time_hours : ℕ)
  (H1 : initial_reading = 15951) 
  (H2 : final_reading = 16061)
  (H3 : time_hours = 2) : 
  (final_reading - initial_reading) / time_hours = 55 :=
by
  sorry

end average_speed_is_55_l312_312119


namespace parallel_planes_imply_l312_312580

variable {Point Line Plane : Type}

-- Definitions of parallelism and perpendicularity between lines and planes
variables {parallel_perpendicular : Line → Plane → Prop}
variables {parallel_lines : Line → Line → Prop}
variables {parallel_planes : Plane → Plane → Prop}

-- Given conditions
variable {m n : Line}
variable {α β : Plane}

-- Conditions
axiom m_parallel_n : parallel_lines m n
axiom m_perpendicular_α : parallel_perpendicular m α
axiom n_perpendicular_β : parallel_perpendicular n β

-- The statement to be proven
theorem parallel_planes_imply (m_parallel_n : parallel_lines m n)
  (m_perpendicular_α : parallel_perpendicular m α)
  (n_perpendicular_β : parallel_perpendicular n β) :
  parallel_planes α β :=
sorry

end parallel_planes_imply_l312_312580


namespace max_groups_l312_312188

def eggs : ℕ := 20
def marbles : ℕ := 6
def eggs_per_group : ℕ := 5
def marbles_per_group : ℕ := 2

def groups_of_eggs := eggs / eggs_per_group
def groups_of_marbles := marbles / marbles_per_group

theorem max_groups (h1 : eggs = 20) (h2 : marbles = 6) 
                    (h3 : eggs_per_group = 5) (h4 : marbles_per_group = 2) : 
                    min (groups_of_eggs) (groups_of_marbles) = 3 :=
by
  sorry

end max_groups_l312_312188


namespace number_of_divisors_of_24_divisible_by_4_l312_312495

theorem number_of_divisors_of_24_divisible_by_4 :
  (∃ b, 4 ∣ b ∧ b ∣ 24 ∧ 0 < b) → (finset.card (finset.filter (λ b, 4 ∣ b) (finset.filter (λ b, b ∣ 24) (finset.range 25))) = 4) :=
by
  sorry

end number_of_divisors_of_24_divisible_by_4_l312_312495


namespace tangent_ellipse_hyperbola_l312_312808

theorem tangent_ellipse_hyperbola (n : ℝ) :
  (∀ x y : ℝ, x^2 + 9 * y^2 = 9 ∧ x^2 - n * (y - 1)^2 = 1) → n = 2 :=
by
  intro h
  sorry

end tangent_ellipse_hyperbola_l312_312808


namespace ellipse_x_intersection_l312_312883

open Real

def F1 : Point := (0, 3)
def F2 : Point := (4, 0)

theorem ellipse_x_intersection :
  {P : Point | dist P F1 + dist P F2 = 8} ∧ (P = (x, 0)) → P = (45 / 8, 0) :=
by
  sorry

end ellipse_x_intersection_l312_312883


namespace jellybean_addition_l312_312366

-- Definitions related to the problem
def initial_jellybeans : ℕ := 37
def removed_jellybeans_initial : ℕ := 15
def added_jellybeans (x : ℕ) : ℕ := x
def removed_jellybeans_again : ℕ := 4
def final_jellybeans : ℕ := 23

-- Prove that the number of jellybeans added back (x) is 5
theorem jellybean_addition (x : ℕ) 
  (h1 : initial_jellybeans - removed_jellybeans_initial + added_jellybeans x - removed_jellybeans_again = final_jellybeans) : 
  x = 5 :=
sorry

end jellybean_addition_l312_312366


namespace train_speed_l312_312222

theorem train_speed (train_length : ℕ) (cross_time : ℕ) (speed : ℕ) 
  (h_train_length : train_length = 300)
  (h_cross_time : cross_time = 10)
  (h_speed_eq : speed = train_length / cross_time) : 
  speed = 30 :=
by
  sorry

end train_speed_l312_312222


namespace regular_polygon_num_sides_l312_312003

def diag_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem regular_polygon_num_sides (n : ℕ) (h : diag_formula n = 20) : n = 8 :=
by
  sorry

end regular_polygon_num_sides_l312_312003


namespace two_a_plus_two_b_plus_two_c_l312_312448

variable (a b c : ℝ)

-- Defining the conditions as the hypotheses
def condition1 : Prop := b + c = 15 - 4 * a
def condition2 : Prop := a + c = -18 - 4 * b
def condition3 : Prop := a + b = 10 - 4 * c

-- The theorem to prove
theorem two_a_plus_two_b_plus_two_c (h1 : condition1 a b c) (h2 : condition2 a b c) (h3 : condition3 a b c) :
  2 * a + 2 * b + 2 * c = 7 / 3 :=
by
  sorry

end two_a_plus_two_b_plus_two_c_l312_312448


namespace mean_equality_l312_312968

-- Define the mean calculation
def mean (a b c : ℕ) : ℚ := (a + b + c) / 3

-- The given conditions
theorem mean_equality (z : ℕ) (y : ℕ) (hz : z = 24) :
  mean 8 15 21 = mean 16 z y → y = 4 :=
by
  sorry

end mean_equality_l312_312968


namespace area_of_T_prime_l312_312795

-- Given conditions
def AreaBeforeTransformation : ℝ := 9

def TransformationMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 4],![-2, 5]]

def AreaAfterTransformation (M : Matrix (Fin 2) (Fin 2) ℝ) (area_before : ℝ) : ℝ :=
  (M.det) * area_before

-- Problem statement
theorem area_of_T_prime : 
  AreaAfterTransformation TransformationMatrix AreaBeforeTransformation = 207 :=
by
  sorry

end area_of_T_prime_l312_312795


namespace perimeter_triangle_ABC_is_correct_l312_312768

noncomputable def semicircle_perimeter_trianlge_ABC : ℝ :=
  let BE := (1 : ℝ)
  let EF := (24 : ℝ)
  let FC := (3 : ℝ)
  let BC := BE + EF + FC
  let r := EF / 2
  let x := 71.5
  let AB := x + BE
  let AC := x + FC
  AB + BC + AC

theorem perimeter_triangle_ABC_is_correct : semicircle_perimeter_trianlge_ABC = 175 := by
  sorry

end perimeter_triangle_ABC_is_correct_l312_312768


namespace min_moves_to_checkerboard_l312_312365

noncomputable def minimum_moves_checkerboard (n : ℕ) : ℕ :=
if n = 6 then 18
else 0

theorem min_moves_to_checkerboard :
  minimum_moves_checkerboard 6 = 18 :=
by sorry

end min_moves_to_checkerboard_l312_312365


namespace men_took_dip_l312_312349

theorem men_took_dip 
  (tank_length : ℝ) (tank_breadth : ℝ) (water_rise_cm : ℝ) (man_displacement : ℝ)
  (H1 : tank_length = 40) (H2 : tank_breadth = 20) (H3 : water_rise_cm = 25) (H4 : man_displacement = 4) :
  let water_rise_m := water_rise_cm / 100
  let total_volume_displaced := tank_length * tank_breadth * water_rise_m
  let number_of_men := total_volume_displaced / man_displacement
  number_of_men = 50 :=
by
  sorry

end men_took_dip_l312_312349


namespace find_numbers_l312_312540

theorem find_numbers (x y a : ℕ) (h1 : x = 6 * y - a) (h2 : x + y = 38) : 7 * x = 228 - a → y = 38 - x :=
by
  sorry

end find_numbers_l312_312540


namespace base_of_first_term_l312_312461

-- Define the necessary conditions
def equation (x s : ℝ) : Prop :=
  x^16 * 25^s = 5 * 10^16

-- The proof goal
theorem base_of_first_term (x s : ℝ) (h : equation x s) : x = 2 / 5 :=
by
  sorry

end base_of_first_term_l312_312461


namespace third_number_in_sequence_l312_312815

theorem third_number_in_sequence (n : ℕ) (h_sum : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 63) : n + 2 = 8 :=
by
  -- the proof would be written here
  sorry

end third_number_in_sequence_l312_312815


namespace math_problem_l312_312924

-- Definition of ⊕
def opp (a b : ℝ) : ℝ := a * b + a - b

-- Definition of ⊗
def tensor (a b : ℝ) : ℝ := (a * b) + a - b

theorem math_problem (a b : ℝ) :
  opp a b + tensor (b - a) b = b^2 - b := 
by
  sorry

end math_problem_l312_312924


namespace find_N_l312_312669

-- Define the problem parameters
def certain_value : ℝ := 0
def x : ℝ := 10

-- Define the main statement to be proved
theorem find_N (N : ℝ) : 3 * x = (N - x) + certain_value → N = 40 :=
  by sorry

end find_N_l312_312669


namespace evaluate_exponent_l312_312728

theorem evaluate_exponent : (3^3)^2 = 729 := by
  sorry

end evaluate_exponent_l312_312728


namespace total_income_l312_312871

variable (I : ℝ)

/-- A person distributed 20% of his income to his 3 children each. -/
def distributed_children (I : ℝ) : ℝ := 3 * 0.20 * I

/-- He deposited 30% of his income to his wife's account. -/
def deposited_wife (I : ℝ) : ℝ := 0.30 * I

/-- The total percentage of his income that was given away is 90%. -/
def total_given_away (I : ℝ) : ℝ := distributed_children I + deposited_wife I 

/-- The remaining income after giving away 90%. -/
def remaining_income (I : ℝ) : ℝ := I - total_given_away I

/-- He donated 5% of the remaining income to the orphan house. -/
def donated_orphan_house (remaining : ℝ) : ℝ := 0.05 * remaining

/-- Finally, he has $40,000 left, which is 95% of the remaining income. -/
def final_amount (remaining : ℝ) : ℝ := 0.95 * remaining

theorem total_income (I : ℝ) (h : final_amount (remaining_income I) = 40000) :
  I = 421052.63 := 
  sorry

end total_income_l312_312871


namespace zayne_total_revenue_l312_312071

-- Defining the constants and conditions
def price_per_bracelet := 5
def deal_price := 8
def initial_bracelets := 30
def revenue_from_five_dollar_sales := 60

-- Calculating number of bracelets sold for $5 each
def bracelets_sold_five_dollars := revenue_from_five_dollar_sales / price_per_bracelet

-- Calculating remaining bracelets after selling some for $5 each
def remaining_bracelets := initial_bracelets - bracelets_sold_five_dollars

-- Calculating number of pairs sold at two for $8
def pairs_sold := remaining_bracelets / 2

-- Calculating revenue from selling pairs
def revenue_from_deal_sales := pairs_sold * deal_price

-- Total revenue calculation
def total_revenue := revenue_from_five_dollar_sales + revenue_from_deal_sales

-- Theorem to prove the total revenue is $132
theorem zayne_total_revenue : total_revenue = 132 := by
  sorry

end zayne_total_revenue_l312_312071


namespace cards_value_1_count_l312_312255

/-- There are 4 different suits in a deck of cards containing a total of 52 cards.
  Each suit has 13 cards numbered from 1 to 13.
  Feifei draws 2 hearts, 3 spades, 4 diamonds, and 5 clubs.
  The sum of the face values of these 14 cards is exactly 35.
  Prove that 4 of these cards have a face value of 1. -/
theorem cards_value_1_count :
  ∃ (hearts spades diamonds clubs : List ℕ),
  hearts.length = 2 ∧ spades.length = 3 ∧ diamonds.length = 4 ∧ clubs.length = 5 ∧
  (∀ v, v ∈ hearts → v ∈ List.range 13) ∧ 
  (∀ v, v ∈ spades → v ∈ List.range 13) ∧
  (∀ v, v ∈ diamonds → v ∈ List.range 13) ∧
  (∀ v, v ∈ clubs → v ∈ List.range 13) ∧
  (hearts.sum + spades.sum + diamonds.sum + clubs.sum = 35) ∧
  ((hearts ++ spades ++ diamonds ++ clubs).count 1 = 4) := sorry

end cards_value_1_count_l312_312255


namespace ordered_triples_54000_l312_312444

theorem ordered_triples_54000 : 
  ∃ (count : ℕ), 
  count = 16 ∧ 
  ∀ (a b c : ℕ), 
  0 < a → 0 < b → 0 < c → a^4 * b^2 * c = 54000 → 
  count = 16 := 
sorry

end ordered_triples_54000_l312_312444


namespace find_x_for_parallel_l312_312475

-- Definitions for vector components and parallel condition.
def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, -3)

def parallel (v w : ℝ × ℝ) : Prop := v.1 * w.2 = v.2 * w.1

theorem find_x_for_parallel :
  ∃ x : ℝ, parallel a (b x) ∧ x = -3 / 2 :=
by
  -- The statement to be proven
  sorry

end find_x_for_parallel_l312_312475


namespace least_number_divisible_by_23_l312_312519

theorem least_number_divisible_by_23 (n d : ℕ) (h_n : n = 1053) (h_d : d = 23) : ∃ x : ℕ, (n + x) % d = 0 ∧ x = 5 := by
  sorry

end least_number_divisible_by_23_l312_312519


namespace q_simplified_l312_312943

noncomputable def q (a b c x : ℝ) : ℝ :=
  (x + a)^4 / ((a - b) * (a - c)) +
  (x + b)^4 / ((b - a) * (b - c)) +
  (x + c)^4 / ((c - a) * (c - b)) - 3 * x * (
      1 / ((a - b) * (a - c)) + 
      1 / ((b - a) * (b - c)) +
      1 / ((c - a) * (c - b))
  )

theorem q_simplified (a b c x : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
  q a b c x = a^2 + b^2 + c^2 + 4*x^2 - 4*(a + b + c)*x + 12*x :=
sorry

end q_simplified_l312_312943


namespace sylvia_carla_together_time_l312_312803

-- Define the conditions
def sylviaRate := 1 / 45
def carlaRate := 1 / 30

-- Define the combined work rate and the time taken to complete the job together
def combinedRate := sylviaRate + carlaRate
def timeTogether := 1 / combinedRate

-- Theorem stating the desired result
theorem sylvia_carla_together_time : timeTogether = 18 := by
  sorry

end sylvia_carla_together_time_l312_312803


namespace zinc_copper_mixture_weight_l312_312387

theorem zinc_copper_mixture_weight (Z C : ℝ) (h1 : Z / C = 9 / 11) (h2 : Z = 31.5) : Z + C = 70 := by
  sorry

end zinc_copper_mixture_weight_l312_312387


namespace parabola_vertex_sum_l312_312280

theorem parabola_vertex_sum (p q r : ℝ)
  (h1 : ∃ a : ℝ, ∀ x y : ℝ, y = a * (x - 3)^2 + 4 → y = p * x^2 + q * x + r)
  (h2 : ∀ y1 : ℝ, y1 = p * (1 : ℝ)^2 + q * (1 : ℝ) + r → y1 = 10)
  (h3 : ∀ y2 : ℝ, y2 = p * (-1 : ℝ)^2 + q * (-1 : ℝ) + r → y2 = 14) :
  p + q + r = 10 :=
sorry

end parabola_vertex_sum_l312_312280


namespace eval_exp_l312_312724

theorem eval_exp : (3^3)^2 = 729 := sorry

end eval_exp_l312_312724


namespace number_of_bowls_l312_312825

theorem number_of_bowls (n : ℕ) (h : 8 * 12 = 96) (avg_increase : 6 * n = 96) : n = 16 :=
by {
  sorry
}

end number_of_bowls_l312_312825


namespace potatoes_leftover_l312_312867

-- Define the necessary conditions
def fries_per_potato : ℕ := 25
def total_potatoes : ℕ := 15
def fries_needed : ℕ := 200

-- Prove the goal
theorem potatoes_leftover : total_potatoes - (fries_needed / fries_per_potato) = 7 :=
sorry

end potatoes_leftover_l312_312867


namespace area_of_white_square_l312_312181

theorem area_of_white_square
  (face_area : ℕ)
  (total_surface_area : ℕ)
  (blue_paint_area : ℕ)
  (faces : ℕ)
  (area_of_white_square : ℕ) :
  face_area = 12 * 12 →
  total_surface_area = 6 * face_area →
  blue_paint_area = 432 →
  faces = 6 →
  area_of_white_square = face_area - (blue_paint_area / faces) →
  area_of_white_square = 72 :=
by
  sorry

end area_of_white_square_l312_312181


namespace vector_parallel_and_on_line_l312_312553

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

end vector_parallel_and_on_line_l312_312553


namespace max_min_x2_minus_xy_plus_y2_l312_312177

theorem max_min_x2_minus_xy_plus_y2 (x y: ℝ) (h : |5 * x + y| + |5 * x - y| = 20) : 
  3 ≤ x^2 - x * y + y^2 ∧ x^2 - x * y + y^2 ≤ 124 := 
sorry

end max_min_x2_minus_xy_plus_y2_l312_312177


namespace rainfall_difference_correct_l312_312621

def rainfall_difference (monday_rain : ℝ) (tuesday_rain : ℝ) : ℝ :=
  monday_rain - tuesday_rain

theorem rainfall_difference_correct : rainfall_difference 0.9 0.2 = 0.7 :=
by
  simp [rainfall_difference]
  sorry

end rainfall_difference_correct_l312_312621


namespace minimum_value_f_minimum_value_abc_l312_312474

noncomputable def f (x : ℝ) : ℝ := abs (x - 4) + abs (x - 3)

theorem minimum_value_f : ∃ m : ℝ, m = 1 ∧ ∀ x : ℝ, f x ≥ m := 
by
  let m := 1
  existsi m
  sorry

theorem minimum_value_abc (a b c : ℝ) (h : a + 2 * b + 3 * c = 1) : ∃ n : ℝ, n = 1/14 ∧ a^2 + b^2 + c^2 ≥ n :=
by
  let n := 1 / 14
  existsi n
  sorry

end minimum_value_f_minimum_value_abc_l312_312474


namespace sum_of_remainders_l312_312376

theorem sum_of_remainders (a b c : ℕ) (h1 : a % 15 = 11) (h2 : b % 15 = 13) (h3 : c % 15 = 14) : (a + b + c) % 15 = 8 :=
by
  sorry

end sum_of_remainders_l312_312376


namespace sum_of_other_endpoint_l312_312206

theorem sum_of_other_endpoint (x y : ℝ) :
  (10, -6) = ((x + 12) / 2, (y + 4) / 2) → x + y = -8 :=
by
  sorry

end sum_of_other_endpoint_l312_312206


namespace geometric_mean_of_1_and_4_l312_312201

theorem geometric_mean_of_1_and_4 :
  ∃ a : ℝ, a^2 = 4 ∧ (a = 2 ∨ a = -2) :=
by
  sorry

end geometric_mean_of_1_and_4_l312_312201


namespace evaluate_exponent_l312_312732

theorem evaluate_exponent : (3^3)^2 = 729 := by
  sorry

end evaluate_exponent_l312_312732


namespace length_of_rest_of_body_l312_312478

theorem length_of_rest_of_body (h : ℝ) (legs : ℝ) (head : ℝ) (rest_of_body : ℝ) :
  h = 60 → legs = (1 / 3) * h → head = (1 / 4) * h → rest_of_body = h - (legs + head) → rest_of_body = 25 := by
  sorry

end length_of_rest_of_body_l312_312478


namespace max_sum_of_roots_l312_312147

theorem max_sum_of_roots (a b : ℝ) (h_a : a ≠ 0) (m : ℝ) :
  (∀ x : ℝ, (2 * x ^ 2 - 5 * x + m = 0) → 25 - 8 * m ≥ 0) →
  (∃ s, s = -5 / 2) → m = 25 / 8 :=
by
  sorry

end max_sum_of_roots_l312_312147


namespace total_distance_l312_312074

/--
A man completes a journey in 30 hours. He travels the first half of the journey at the rate of 20 km/hr and 
the second half at the rate of 10 km/hr. Prove that the total journey is 400 km.
-/
theorem total_distance (D : ℝ) (h : D / 40 + D / 20 = 30) :
  D = 400 :=
sorry

end total_distance_l312_312074


namespace problem1_problem2_l312_312765

noncomputable def f (x a b c : ℝ) : ℝ := abs (x + a) + abs (x - b) + c

theorem problem1 (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c)
  (h₃ : ∃ x, f x a b c = 4) : a + b + c = 4 :=
sorry

theorem problem2 (a b c : ℝ) (h : a + b + c = 4) : (1 / a) + (1 / b) + (1 / c) ≥ 9 / 4 :=
sorry

end problem1_problem2_l312_312765


namespace square_of_binomial_l312_312985

theorem square_of_binomial (a b : ℝ) : 
  (a - 5 * b)^2 = a^2 - 10 * a * b + 25 * b^2 :=
by
  sorry

end square_of_binomial_l312_312985


namespace meeting_distance_from_top_l312_312324

section

def total_distance : ℝ := 12
def uphill_distance : ℝ := 6
def downhill_distance : ℝ := 6
def john_start_time : ℝ := 0.25
def john_uphill_speed : ℝ := 12
def john_downhill_speed : ℝ := 18
def jenny_uphill_speed : ℝ := 14
def jenny_downhill_speed : ℝ := 21

theorem meeting_distance_from_top : 
  ∃ (d : ℝ), d = 6 - 14 * ((0.25) + 6 / 14 - (1 / 2) - (6 - 18 * ((1 / 2) + d / 18))) / 14 ∧ d = 45 / 32 :=
sorry

end

end meeting_distance_from_top_l312_312324


namespace solve_for_x_l312_312599

theorem solve_for_x (x : ℝ) (h₀ : x^2 - 2 * x = 0) (h₁ : x ≠ 0) : x = 2 :=
sorry

end solve_for_x_l312_312599


namespace red_fraction_is_three_fifths_l312_312606

noncomputable def fraction_of_red_marbles (x : ℕ) : ℚ := 
  let blue_marbles := (2 / 3 : ℚ) * x
  let red_marbles := x - blue_marbles
  let new_red_marbles := 3 * red_marbles
  let new_total_marbles := blue_marbles + new_red_marbles
  new_red_marbles / new_total_marbles

theorem red_fraction_is_three_fifths (x : ℕ) (hx : x ≠ 0) : fraction_of_red_marbles x = 3 / 5 :=
by {
  sorry
}

end red_fraction_is_three_fifths_l312_312606


namespace max_diff_units_digit_l312_312210

theorem max_diff_units_digit (n : ℕ) (h1 : n = 850 ∨ n = 855) : ∃ d, d = 5 :=
by 
  sorry

end max_diff_units_digit_l312_312210


namespace exercise_l312_312044

theorem exercise (x y z : ℝ)
  (h1 : x + y + z = 30)
  (h2 : x * y * z = 343)
  (h3 : 1/x + 1/y + 1/z = 3/5) : x^2 + y^2 + z^2 = 488.4 :=
sorry

end exercise_l312_312044


namespace cos_angle_MJL_l312_312611

theorem cos_angle_MJL (JKL_angle : ∠J K L = 60)
                      (JNM_angle : ∠J N M = 30) :
  cos (∠M J L) = 5 / 8 := 
by 
  sorry

end cos_angle_MJL_l312_312611


namespace most_likely_num_acceptable_bearings_l312_312687

/--
Let X be a normally distributed random variable with standard deviation σ = 0.4 mm.
The deviation X from the design size is considered acceptable if |X| ≤ 0.77 mm.
We take a sample of 100 bearings.
Prove that the most likely number of acceptable bearings in the sample is 95.
-/
theorem most_likely_num_acceptable_bearings :
  ∀ (X : ℝ → ℝ), is_normal_dist X 0 0.4 →
  (∀ x, X x ∈ set.Icc (-0.77) 0.77) →
  let n := 100 in
  let p := 0.9464 in
  n * p ≈ 95 :=
begin
  sorry
end

end most_likely_num_acceptable_bearings_l312_312687


namespace sam_found_seashells_l312_312805

def seashells_given : Nat := 18
def seashells_left : Nat := 17
def seashells_found : Nat := seashells_given + seashells_left

theorem sam_found_seashells : seashells_found = 35 := by
  sorry

end sam_found_seashells_l312_312805


namespace max_contribution_l312_312600

theorem max_contribution (n : ℕ) (total : ℝ) (min_contribution : ℝ)
  (h1 : n = 12) (h2 : total = 20) (h3 : min_contribution = 1)
  (h4 : ∀ i : ℕ, i < n → min_contribution ≤ min_contribution) :
  ∃ max_contrib : ℝ, max_contrib = 9 :=
by
  sorry

end max_contribution_l312_312600


namespace power_of_power_evaluation_l312_312734

theorem power_of_power_evaluation : (3^3)^2 = 729 := 
by
  -- Replace this with the actual proof
  sorry

end power_of_power_evaluation_l312_312734


namespace find_number_l312_312539

-- Definitions and conditions for the problem
def N_div_7 (N R_1 : ℕ) : ℕ := (N / 7) * 7 + R_1
def N_div_11 (N R_2 : ℕ) : ℕ := (N / 11) * 11 + R_2
def N_div_13 (N R_3 : ℕ) : ℕ := (N / 13) * 13 + R_3

theorem find_number 
  (N a b c R_1 R_2 R_3 : ℕ) 
  (hN7 : N = 7 * a + R_1)
  (hN11 : N = 11 * b + R_2)
  (hN13 : N = 13 * c + R_3)
  (hQ : a + b + c = 21)
  (hR : R_1 + R_2 + R_3 = 21)
  (hR1_lt : R_1 < 7)
  (hR2_lt : R_2 < 11)
  (hR3_lt : R_3 < 13) : 
  N = 74 :=
sorry

end find_number_l312_312539


namespace toms_expense_l312_312211

def cost_per_square_foot : ℝ := 5
def square_feet_per_seat : ℝ := 12
def number_of_seats : ℝ := 500
def partner_coverage : ℝ := 0.40

def total_square_feet : ℝ := square_feet_per_seat * number_of_seats
def land_cost : ℝ := cost_per_square_foot * total_square_feet
def construction_cost : ℝ := 2 * land_cost
def total_cost : ℝ := land_cost + construction_cost
def tom_coverage_percentage : ℝ := 1 - partner_coverage
def toms_share : ℝ := tom_coverage_percentage * total_cost

theorem toms_expense :
  toms_share = 54000 :=
by
  sorry

end toms_expense_l312_312211


namespace Dave_has_more_money_than_Derek_l312_312556

def Derek_initial := 40
def Derek_expense1 := 14
def Derek_expense2 := 11
def Derek_expense3 := 5
def Derek_remaining := Derek_initial - Derek_expense1 - Derek_expense2 - Derek_expense3

def Dave_initial := 50
def Dave_expense := 7
def Dave_remaining := Dave_initial - Dave_expense

def money_difference := Dave_remaining - Derek_remaining

theorem Dave_has_more_money_than_Derek : money_difference = 33 := by sorry

end Dave_has_more_money_than_Derek_l312_312556


namespace expected_waiting_time_for_passenger_l312_312392

noncomputable def expected_waiting_time (arrival_time: ℕ) : ℝ := 
  if arrival_time < 20 * 60
    then (20 / 60) * (40 / 2) + (40 / 60) * (45) -- from 8:00 to 9:00 AM
    else 60 + 45 -- from 9:00 to 10:00 AM

theorem expected_waiting_time_for_passenger :
  expected_waiting_time (20 * 60) = 53.3 :=
sorry

end expected_waiting_time_for_passenger_l312_312392


namespace constant_term_of_expansion_eq_80_l312_312760

theorem constant_term_of_expansion_eq_80 (a : ℝ) (h1 : 0 < a)
  (h2 : (∑ k in Finset.range 6, Nat.choose 5 k * real.pow a k * real.pow (2.5 : ℝ) (10 - 2.5 * k : ℝ)) = 80) :
  a = 2 :=
by
  sorry

end constant_term_of_expansion_eq_80_l312_312760


namespace original_quantity_l312_312872

theorem original_quantity (x : ℕ) : 
  (532 * x - 325 * x = 1065430) -> x = 5148 := 
by
  intro h
  sorry

end original_quantity_l312_312872


namespace problem1_problem2_l312_312675

/-- Problem 1: Prove the solution to the system of equations is x = 1/2 and y = 5 -/
theorem problem1 (x y : ℚ) (h1 : 2 * x - y = -4) (h2 : 4 * x - 5 * y = -23) : 
  x = 1 / 2 ∧ y = 5 := 
sorry

/-- Problem 2: Prove the value of the expression (x-3y)^{2} - (2x+y)(y-2x) when x = 2 and y = -1 is 40 -/
theorem problem2 (x y : ℚ) (h1 : x = 2) (h2 : y = -1) : 
  (x - 3 * y) ^ 2 - (2 * x + y) * (y - 2 * x) = 40 := 
sorry

end problem1_problem2_l312_312675


namespace min_value_inequality_l312_312898

theorem min_value_inequality (a b c : ℝ) (h : a + b + c = 3) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 / (a + b) + 1 / c) ≥ 4 / 3 :=
sorry

end min_value_inequality_l312_312898


namespace asparagus_spears_needed_l312_312552

def BridgetteGuests : Nat := 84
def AlexGuests : Nat := (2 * BridgetteGuests) / 3
def TotalGuests : Nat := BridgetteGuests + AlexGuests
def ExtraPlates : Nat := 10
def TotalPlates : Nat := TotalGuests + ExtraPlates
def VegetarianPercent : Nat := 20
def LargePortionPercent : Nat := 10
def VegetarianMeals : Nat := (VegetarianPercent * TotalGuests) / 100
def LargePortionMeals : Nat := (LargePortionPercent * TotalGuests) / 100
def RegularMeals : Nat := TotalGuests - (VegetarianMeals + LargePortionMeals)
def AsparagusPerRegularMeal : Nat := 8
def AsparagusPerVegetarianMeal : Nat := 6
def AsparagusPerLargePortionMeal : Nat := 12

theorem asparagus_spears_needed : 
  RegularMeals * AsparagusPerRegularMeal + 
  VegetarianMeals * AsparagusPerVegetarianMeal + 
  LargePortionMeals * AsparagusPerLargePortionMeal = 1120 := by
  sorry

end asparagus_spears_needed_l312_312552


namespace linear_equation_solution_l312_312208

theorem linear_equation_solution (m : ℝ) (x : ℝ) (h : |m| - 2 = 1) (h_ne : m ≠ 3) :
  (2 * m - 6) * x^(|m|-2) = m^2 ↔ x = -(3/4) :=
by
  sorry

end linear_equation_solution_l312_312208


namespace simplify_expression_l312_312601

theorem simplify_expression (x y : ℝ) (P Q : ℝ) (hP : P = 2 * x + 3 * y) (hQ : Q = 3 * x + 2 * y) :
  ((P + Q) / (P - Q)) - ((P - Q) / (P + Q)) = (24 * x ^ 2 + 52 * x * y + 24 * y ^ 2) / (5 * x * y - 5 * y ^ 2) :=
by
  sorry

end simplify_expression_l312_312601


namespace lines_perpendicular_iff_l312_312951

/-- Given two lines y = k₁ x + l₁ and y = k₂ x + l₂, 
    which are not parallel to the coordinate axes,
    these lines are perpendicular if and only if k₁ * k₂ = -1. -/
theorem lines_perpendicular_iff 
  (k₁ k₂ l₁ l₂ : ℝ) (h1 : k₁ ≠ 0) (h2 : k₂ ≠ 0) :
  (∀ x, k₁ * x + l₁ = k₂ * x + l₂) <-> k₁ * k₂ = -1 :=
sorry

end lines_perpendicular_iff_l312_312951


namespace max_value_l312_312319

def a_n (n : ℕ) : ℤ := -2 * (n : ℤ)^2 + 29 * (n : ℤ) + 3

theorem max_value : ∃ n : ℕ, a_n n = 108 ∧ ∀ m : ℕ, a_n m ≤ 108 := by
  sorry

end max_value_l312_312319


namespace find_y_l312_312327

def diamond (a b : ℝ) : ℝ := a * b + 3 * b - a

theorem find_y : ∃ y : ℝ, diamond 4 y = 44 ∧ y = 48 / 7 :=
by
  sorry

end find_y_l312_312327


namespace eval_expr_l312_312740

theorem eval_expr : (3^3)^2 = 729 := 
by
  sorry

end eval_expr_l312_312740


namespace simplify_and_evaluate_correct_l312_312648

noncomputable def simplify_and_evaluate (x y : ℚ) : ℚ :=
  3 * (x^2 - 2 * x * y) - (3 * x^2 - 2 * y + 2 * (x * y + y))

theorem simplify_and_evaluate_correct : 
  simplify_and_evaluate (-1 / 2 : ℚ) (-3 : ℚ) = -12 := by
  sorry

end simplify_and_evaluate_correct_l312_312648


namespace distance_between_parallel_lines_l312_312657

theorem distance_between_parallel_lines :
  let A := 3
  let B := 2
  let C1 := -1
  let C2 := 1 / 2
  let d := |C2 - C1| / Real.sqrt (A^2 + B^2)
  d = 3 / Real.sqrt 13 :=
by
  -- Proof goes here
  sorry

end distance_between_parallel_lines_l312_312657


namespace find_c_l312_312702

open Real

-- Definition of the quadratic expression in question
def expr (x y c : ℝ) : ℝ := 5 * x^2 - 8 * c * x * y + (4 * c^2 + 3) * y^2 - 5 * x - 5 * y + 7

-- The theorem to prove that the minimum value of this expression being 0 over all (x, y) implies c = 4
theorem find_c :
  (∀ x y : ℝ, expr x y c ≥ 0) → (∃ x y : ℝ, expr x y c = 0) → c = 4 := 
by 
  sorry

end find_c_l312_312702


namespace james_music_listening_hours_l312_312017

theorem james_music_listening_hours (BPM : ℕ) (beats_per_week : ℕ) (hours_per_day : ℕ) 
  (h1 : BPM = 200) 
  (h2 : beats_per_week = 168000) 
  (h3 : hours_per_day * 200 * 60 * 7 = beats_per_week) : 
  hours_per_day = 2 := 
by
  sorry

end james_music_listening_hours_l312_312017


namespace find_x_l312_312679
-- Import all necessary libraries

-- Define the conditions
variables (x : ℝ) (log5x log6x log15x : ℝ)

-- Assume the edge lengths of the prism are logs with different bases
def edge_lengths (x : ℝ) (log5x log6x log15x : ℝ) : Prop :=
  log5x = Real.logb 5 x ∧ log6x = Real.logb 6 x ∧ log15x = Real.logb 15 x

-- Define the ratio of Surface Area to Volume
def ratio_SA_to_V (x : ℝ) (log5x log6x log15x : ℝ) : Prop :=
  let SA := 2 * (log5x * log6x + log5x * log15x + log6x * log15x)
  let V  := log5x * log6x * log15x
  SA / V = 10

-- Prove the value of x
theorem find_x (h1 : edge_lengths x log5x log6x log15x) (h2 : ratio_SA_to_V x log5x log6x log15x) :
  x = Real.rpow 450 (1/5) := 
sorry

end find_x_l312_312679


namespace distance_focus_asymptote_l312_312772

noncomputable def focus := (Real.sqrt 6 / 2, 0)
def asymptote (x y : ℝ) := x - Real.sqrt 2 * y = 0
def hyperbola (x y : ℝ) := x^2 - 2 * y^2 = 1

theorem distance_focus_asymptote :
  let d := (Real.sqrt 6 / 2, 0)
  let A := 1
  let B := -Real.sqrt 2
  let C := 0
  let numerator := abs (A * d.1 + B * d.2 + C)
  let denominator := Real.sqrt (A^2 + B^2)
  numerator / denominator = Real.sqrt 2 / 2 :=
sorry

end distance_focus_asymptote_l312_312772


namespace bob_hair_growth_time_l312_312203

theorem bob_hair_growth_time (initial_length final_length growth_rate monthly_to_yearly_conversion : ℝ) 
  (initial_cut : initial_length = 6) 
  (current_length : final_length = 36) 
  (growth_per_month : growth_rate = 0.5) 
  (months_in_year : monthly_to_yearly_conversion = 12) : 
  (final_length - initial_length) / (growth_rate * monthly_to_yearly_conversion) = 5 :=
by
  sorry

end bob_hair_growth_time_l312_312203


namespace avg_abc_43_l312_312502

variables (A B C : ℝ)

def avg_ab (A B : ℝ) : Prop := (A + B) / 2 = 40
def avg_bc (B C : ℝ) : Prop := (B + C) / 2 = 43
def weight_b (B : ℝ) : Prop := B = 37

theorem avg_abc_43 (A B C : ℝ) (h1 : avg_ab A B) (h2 : avg_bc B C) (h3 : weight_b B) :
  (A + B + C) / 3 = 43 :=
by
  sorry

end avg_abc_43_l312_312502


namespace divisible_by_xyz_l312_312945

/-- 
Prove that the expression K = (x+y+z)^5 - (-x+y+z)^5 - (x-y+z)^5 - (x+y-z)^5 
is divisible by each of x, y, z.
-/
theorem divisible_by_xyz (x y z : ℝ) :
  ∃ t : ℝ, (x + y + z)^5 - (-x + y + z)^5 - (x - y + z)^5 - (x + y - z)^5 = t * x * y * z :=
by
  -- Proof to be provided
  sorry

end divisible_by_xyz_l312_312945


namespace new_commission_percentage_l312_312400

theorem new_commission_percentage
  (fixed_salary : ℝ)
  (total_sales : ℝ)
  (sales_threshold : ℝ)
  (previous_commission_rate : ℝ)
  (additional_earnings : ℝ)
  (prev_commission : ℝ)
  (extra_sales : ℝ)
  (new_commission : ℝ)
  (new_remuneration : ℝ) :
  fixed_salary = 1000 →
  total_sales = 12000 →
  sales_threshold = 4000 →
  previous_commission_rate = 0.05 →
  additional_earnings = 600 →
  prev_commission = previous_commission_rate * total_sales →
  extra_sales = total_sales - sales_threshold →
  new_remuneration = fixed_salary + new_commission * extra_sales →
  new_remuneration = prev_commission + additional_earnings →
  new_commission = 2.5 / 100 :=
by
  intros
  sorry

end new_commission_percentage_l312_312400


namespace cylinder_volume_l312_312434

theorem cylinder_volume (r h : ℝ) (hrh : 2 * Real.pi * r * h = 100 * Real.pi) (h_diag : 4 * r^2 + h^2 = 200) :
  Real.pi * r^2 * h = 250 * Real.pi :=
sorry

end cylinder_volume_l312_312434


namespace max_n_l312_312572

noncomputable def prod := 160 * 170 * 180 * 190

theorem max_n : ∃ n : ℕ, n = 30499 ∧ n^2 ≤ prod := by
  sorry

end max_n_l312_312572


namespace total_items_bought_l312_312160

def total_money : ℝ := 40
def sandwich_cost : ℝ := 5
def chip_cost : ℝ := 2
def soft_drink_cost : ℝ := 1.5

/-- Ike and Mike spend their total money on sandwiches, chips, and soft drinks.
  We want to prove that the total number of items bought (sandwiches, chips, and soft drinks)
  is equal to 8. -/
theorem total_items_bought :
  ∃ (s c d : ℝ), (sandwich_cost * s + chip_cost * c + soft_drink_cost * d ≤ total_money) ∧
  (∀x : ℝ, sandwich_cost * s ≤ total_money) ∧ ((s + c + d) = 8) :=
by {
  sorry
}

end total_items_bought_l312_312160


namespace find_rate_l312_312225

noncomputable def SI := 200
noncomputable def P := 800
noncomputable def T := 4

theorem find_rate : ∃ R : ℝ, SI = (P * R * T) / 100 ∧ R = 6.25 :=
by sorry

end find_rate_l312_312225


namespace probability_event_l312_312978

-- Define the problem as a probability statement within appropriate mathematical context
theorem probability_event {x y : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  measure_theory.measure_space.probability_space (set_of (λ (p : ℝ × ℝ), 2 * p.1 - p.2 < 0)) = 1 / 4 :=
sorry

end probability_event_l312_312978


namespace min_value_of_f_l312_312450

noncomputable def f (x : ℝ) : ℝ := 2 + 3 * x + 4 / (x - 1)

theorem min_value_of_f :
  (∀ x : ℝ, x > 1 → f x ≥ (5 + 4 * Real.sqrt 3)) ∧
  (f (1 + 2 * Real.sqrt 3 / 3) = 5 + 4 * Real.sqrt 3) :=
by
  sorry

end min_value_of_f_l312_312450


namespace angle_B_in_geometric_progression_l312_312165

theorem angle_B_in_geometric_progression 
  {A B C a b c : ℝ} 
  (hSum : A + B + C = Real.pi)
  (hGeo : A = B / 2)
  (hGeo2 : C = 2 * B)
  (hSide : b^2 - a^2 = a * c)
  : B = 2 * Real.pi / 7 := 
by
  sorry

end angle_B_in_geometric_progression_l312_312165


namespace helen_oranges_l312_312442

def initial_oranges := 9
def oranges_from_ann := 29
def oranges_taken_away := 14

def final_oranges (initial : Nat) (add : Nat) (taken : Nat) : Nat :=
  initial + add - taken

theorem helen_oranges :
  final_oranges initial_oranges oranges_from_ann oranges_taken_away = 24 :=
by
  sorry

end helen_oranges_l312_312442


namespace power_of_power_evaluation_l312_312737

theorem power_of_power_evaluation : (3^3)^2 = 729 := 
by
  -- Replace this with the actual proof
  sorry

end power_of_power_evaluation_l312_312737


namespace probability_interval_l312_312207

theorem probability_interval (P_A P_B p : ℝ) (hP_A : P_A = 2 / 3) (hP_B : P_B = 3 / 5) :
  4 / 15 ≤ p ∧ p ≤ 3 / 5 := sorry

end probability_interval_l312_312207


namespace simplify_expression_l312_312952

variable (a b : ℝ)

theorem simplify_expression (h1 : a ≠ 0) (h2 : b ≠ 0) :
  (a ^ (7 / 3) - 2 * a ^ (5 / 3) * b ^ (2 / 3) + a * b ^ (4 / 3)) / 
  (a ^ (5 / 3) - a ^ (4 / 3) * b ^ (1 / 3) - a * b ^ (2 / 3) + a ^ (2 / 3) * b) / 
  a ^ (1 / 3) =
  a ^ (1 / 3) + b ^ (1 / 3) :=
sorry

end simplify_expression_l312_312952


namespace number_of_bowls_l312_312837

theorem number_of_bowls (n : ℕ) :
  (∀ (b : ℕ), b > 0) →
  (∀ (a : ℕ), ∃ (k : ℕ), true) →
  (8 * 12 = 96) →
  (6 * n = 96) →
  n = 16 :=
by
  intros h1 h2 h3 h4
  sorry

end number_of_bowls_l312_312837


namespace regular_polygon_num_sides_l312_312000

def diag_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem regular_polygon_num_sides (n : ℕ) (h : diag_formula n = 20) : n = 8 :=
by
  sorry

end regular_polygon_num_sides_l312_312000


namespace sum_of_fractions_removal_l312_312118

theorem sum_of_fractions_removal :
  (1 / 3 + 1 / 6 + 1 / 9 + 1 / 12 + 1 / 15 + 1 / 18 + 1 / 21) 
  - (1 / 12 + 1 / 21) = 3 / 4 := 
 by sorry

end sum_of_fractions_removal_l312_312118


namespace f_decreasing_on_interval_l312_312030

open Set Filter

-- Define the function f and the interval (1, +∞)
def f (x : ℝ) : ℝ := 2*x / (x - 1)

-- The theorem we want to prove
theorem f_decreasing_on_interval (x : ℝ) (h : 1 < x) : ∀ x1 x2 : ℝ, 1 < x1 → 1 < x2 → x1 < x2 → f x1 > f x2 :=
by 
  admit

end f_decreasing_on_interval_l312_312030


namespace ramu_profit_percent_l312_312077

theorem ramu_profit_percent
  (cost_of_car : ℕ)
  (cost_of_repairs : ℕ)
  (selling_price : ℕ)
  (total_cost : ℕ := cost_of_car + cost_of_repairs)
  (profit : ℕ := selling_price - total_cost)
  (profit_percent : ℚ := ((profit : ℚ) / total_cost) * 100)
  (h1 : cost_of_car = 42000)
  (h2 : cost_of_repairs = 15000)
  (h3 : selling_price = 64900) :
  profit_percent = 13.86 :=
by
  sorry

end ramu_profit_percent_l312_312077


namespace solid_is_cone_l312_312008

-- Definitions for the conditions
structure Solid where
  front_view : Type
  side_view : Type
  top_view : Type

def is_isosceles_triangle (shape : Type) : Prop := sorry
def is_circle (shape : Type) : Prop := sorry

-- Define the solid based on the given conditions
noncomputable def my_solid : Solid := {
  front_view := sorry,
  side_view := sorry,
  top_view := sorry
}

-- Prove that the solid is a cone given the provided conditions
theorem solid_is_cone (s : Solid) : 
  is_isosceles_triangle s.front_view → 
  is_isosceles_triangle s.side_view → 
  is_circle s.top_view → 
  s = my_solid :=
by
  sorry

end solid_is_cone_l312_312008


namespace steve_speed_ratio_l312_312962

variable (distance : ℝ)
variable (total_time : ℝ)
variable (speed_back : ℝ)
variable (speed_to : ℝ)

noncomputable def speed_ratio (distance : ℝ) (total_time : ℝ) (speed_back : ℝ) : ℝ := 
  let time_to := total_time - distance / speed_back
  let speed_to := distance / time_to
  speed_back / speed_to

theorem steve_speed_ratio (h1 : distance = 10) (h2 : total_time = 6) (h3 : speed_back = 5) :
  speed_ratio distance total_time speed_back = 2 := by
  sorry

end steve_speed_ratio_l312_312962


namespace combined_marbles_l312_312259

def Rhonda_marbles : ℕ := 80
def Amon_marbles : ℕ := Rhonda_marbles + 55

theorem combined_marbles : Amon_marbles + Rhonda_marbles = 215 :=
by
  sorry

end combined_marbles_l312_312259


namespace alchemy_value_l312_312966

def letter_values : List Int :=
  [3, 2, 1, 0, -1, -2, -3, -2, -1, 0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1,
  0, 1, 2, 3]

def char_value (c : Char) : Int :=
  letter_values.getD ((c.toNat - 'A'.toNat) % 13) 0

def word_value (s : String) : Int :=
  s.toList.map char_value |>.sum

theorem alchemy_value :
  word_value "ALCHEMY" = 8 :=
by
  sorry

end alchemy_value_l312_312966


namespace fourth_group_students_l312_312051

theorem fourth_group_students (total_students group1 group2 group3 group4 : ℕ)
  (h_total : total_students = 24)
  (h_group1 : group1 = 5)
  (h_group2 : group2 = 8)
  (h_group3 : group3 = 7)
  (h_groups_sum : group1 + group2 + group3 + group4 = total_students) :
  group4 = 4 :=
by
  -- Proof will go here
  sorry

end fourth_group_students_l312_312051


namespace sum_remainder_l312_312380

theorem sum_remainder (p q r : ℕ) (hp : p % 15 = 11) (hq : q % 15 = 13) (hr : r % 15 = 14) : 
  (p + q + r) % 15 = 8 :=
by
  sorry

end sum_remainder_l312_312380


namespace invitations_per_package_l312_312696

theorem invitations_per_package (total_friends : ℕ) (total_packs : ℕ) (invitations_per_pack : ℕ) 
  (h1 : total_friends = 10) (h2 : total_packs = 5)
  (h3 : invitations_per_pack * total_packs = total_friends) : 
  invitations_per_pack = 2 :=
by
  sorry

end invitations_per_package_l312_312696


namespace exists_five_integers_sum_fifth_powers_no_four_integers_sum_fifth_powers_l312_312525

theorem exists_five_integers_sum_fifth_powers (A B C D E : ℤ) : 
  ∃ (A B C D E : ℤ), 2018 = A^5 + B^5 + C^5 + D^5 + E^5 :=
  by
    sorry

theorem no_four_integers_sum_fifth_powers (A B C D : ℤ) : 
  ¬ ∃ (A B C D : ℤ), 2018 = A^5 + B^5 + C^5 + D^5 :=
  by
    sorry

end exists_five_integers_sum_fifth_powers_no_four_integers_sum_fifth_powers_l312_312525


namespace chef_michel_total_pies_l312_312108

theorem chef_michel_total_pies
  (shepherd_pie_pieces : ℕ)
  (chicken_pot_pie_pieces : ℕ)
  (shepherd_pie_customers : ℕ)
  (chicken_pot_pie_customers : ℕ)
  (H1 : shepherd_pie_pieces = 4)
  (H2 : chicken_pot_pie_pieces = 5)
  (H3 : shepherd_pie_customers = 52)
  (H4 : chicken_pot_pie_customers = 80) :
  (shepherd_pie_customers / shepherd_pie_pieces) + (chicken_pot_pie_customers / chicken_pot_pie_pieces) = 29 :=
by
  sorry

end chef_michel_total_pies_l312_312108


namespace length_of_tunnel_l312_312682

theorem length_of_tunnel (time : ℝ) (speed : ℝ) (train_length : ℝ) (total_distance : ℝ) (tunnel_length : ℝ) 
  (h1 : time = 30) (h2 : speed = 100 / 3) (h3 : train_length = 400) (h4 : total_distance = speed * time) 
  (h5 : tunnel_length = total_distance - train_length) : 
  tunnel_length = 600 :=
by
  sorry

end length_of_tunnel_l312_312682


namespace squido_oysters_l312_312514

theorem squido_oysters (S C : ℕ) (h1 : C ≥ 2 * S) (h2 : S + C = 600) : S = 200 :=
sorry

end squido_oysters_l312_312514


namespace sum_of_remainders_l312_312375

theorem sum_of_remainders (a b c : ℕ) (h1 : a % 15 = 11) (h2 : b % 15 = 13) (h3 : c % 15 = 14) : (a + b + c) % 15 = 8 :=
by
  sorry

end sum_of_remainders_l312_312375


namespace comic_books_stacking_order_l312_312339

-- Definitions of the conditions
def num_spiderman_books : ℕ := 6
def num_archie_books : ℕ := 5
def num_garfield_books : ℕ := 4

-- Calculations of factorials
def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

-- Grouping and order calculation
def ways_to_arrange_group_books : ℕ :=
  factorial num_spiderman_books *
  factorial num_archie_books *
  factorial num_garfield_books

def num_groups : ℕ := 3

def ways_to_arrange_groups : ℕ :=
  factorial num_groups

def total_ways_to_stack_books : ℕ :=
  ways_to_arrange_group_books * ways_to_arrange_groups

-- Theorem stating the total number of different orders
theorem comic_books_stacking_order :
  total_ways_to_stack_books = 12441600 :=
by
  sorry

end comic_books_stacking_order_l312_312339


namespace total_wheels_in_both_garages_l312_312101

/-- Each cycle type has a different number of wheels. --/
def wheels_per_cycle (cycle_type: String) : ℕ :=
  if cycle_type = "bicycle" then 2
  else if cycle_type = "tricycle" then 3
  else if cycle_type = "unicycle" then 1
  else if cycle_type = "quadracycle" then 4
  else 0

/-- Define the counts of each type of cycle in each garage. --/
def garage1_counts := [("bicycle", 5), ("tricycle", 6), ("unicycle", 9), ("quadracycle", 3)]
def garage2_counts := [("bicycle", 2), ("tricycle", 1), ("unicycle", 3), ("quadracycle", 4)]

/-- Total steps for the calculation --/
def wheels_in_garage (garage_counts: List (String × ℕ)) (missing_wheels_unicycles: ℕ) : ℕ :=
  List.foldl (λ acc (cycle_count: String × ℕ) => 
              acc + (if cycle_count.1 = "unicycle" then (cycle_count.2 * wheels_per_cycle cycle_count.1 - missing_wheels_unicycles) 
                     else (cycle_count.2 * wheels_per_cycle cycle_count.1))) 0 garage_counts

/-- The total number of wheels in both garages. --/
def total_wheels : ℕ := wheels_in_garage garage1_counts 0 + wheels_in_garage garage2_counts 3

/-- Prove that the total number of wheels in both garages is 72. --/
theorem total_wheels_in_both_garages : total_wheels = 72 :=
  by sorry

end total_wheels_in_both_garages_l312_312101


namespace common_difference_l312_312163

noncomputable def a : ℕ := 3
noncomputable def an : ℕ := 28
noncomputable def Sn : ℕ := 186

theorem common_difference (d : ℚ) (n : ℕ) (h1 : an = a + (n-1) * d) (h2 : Sn = n * (a + an) / 2) : d = 25 / 11 :=
sorry

end common_difference_l312_312163


namespace john_exactly_three_green_marbles_l312_312625

-- Definitions based on the conditions
def total_marbles : ℕ := 15
def green_marbles : ℕ := 8
def purple_marbles : ℕ := 7
def trials : ℕ := 7
def green_prob : ℚ := 8 / 15
def purple_prob : ℚ := 7 / 15
def binom_coeff : ℕ := Nat.choose 7 3 

-- Theorem Statement
theorem john_exactly_three_green_marbles :
  (binom_coeff : ℚ) * (green_prob^3 * purple_prob^4) = 8604112 / 15946875 :=
by
  sorry

end john_exactly_three_green_marbles_l312_312625


namespace tg_ctg_sum_l312_312894

theorem tg_ctg_sum (x : Real) 
  (h : Real.cos x ≠ 0 ∧ Real.sin x ≠ 0 ∧ 1 / Real.cos x - 1 / Real.sin x = 4 * Real.sqrt 3) :
  (Real.sin x / Real.cos x + Real.cos x / Real.sin x = 8 ∨ Real.sin x / Real.cos x + Real.cos x / Real.sin x = -6) :=
sorry

end tg_ctg_sum_l312_312894


namespace bill_miles_sunday_l312_312862

variables (B : ℕ)
def miles_ran_Bill_Saturday := B
def miles_ran_Bill_Sunday := B + 4
def miles_ran_Julia_Sunday := 2 * (B + 4)
def total_miles_ran := miles_ran_Bill_Saturday + miles_ran_Bill_Sunday + miles_ran_Julia_Sunday

theorem bill_miles_sunday (h1 : total_miles_ran B = 32) : 
  miles_ran_Bill_Sunday B = 9 := 
by sorry

end bill_miles_sunday_l312_312862


namespace citizen_income_l312_312670

theorem citizen_income (tax_paid : ℝ) (base_income : ℝ) (base_rate excess_rate : ℝ) (income : ℝ) 
  (h1 : 0 < base_income) (h2 : base_rate * base_income = 4400) (h3 : tax_paid = 8000)
  (h4 : excess_rate = 0.20) (h5 : base_rate = 0.11)
  (h6 : tax_paid = base_rate * base_income + excess_rate * (income - base_income)) :
  income = 58000 :=
sorry

end citizen_income_l312_312670


namespace rectangle_semicircle_problem_l312_312189

/--
Rectangle ABCD and a semicircle with diameter AB are coplanar and have nonoverlapping interiors.
Let R denote the region enclosed by the semicircle and the rectangle.
Line ℓ meets the semicircle, segment AB, and segment CD at distinct points P, V, and S, respectively.
Line ℓ divides region R into two regions with areas in the ratio 3:1.
Suppose that AV = 120, AP = 180, and VB = 240.
Prove the length of DA = 90 * sqrt(6).
-/
theorem rectangle_semicircle_problem (DA : ℝ) (AV AP VB : ℝ) (h₁ : AV = 120) (h₂ : AP = 180) (h₃ : VB = 240) :
  DA = 90 * Real.sqrt 6 := by
  sorry

end rectangle_semicircle_problem_l312_312189


namespace valid_triangle_side_l312_312878

theorem valid_triangle_side (x : ℕ) (h_pos : 0 < x) (h1 : x + 6 > 15) (h2 : 21 > x) :
  10 ≤ x ∧ x ≤ 20 :=
by {
  sorry
}

end valid_triangle_side_l312_312878


namespace proof_MrLalandeInheritance_l312_312798

def MrLalandeInheritance : Nat := 18000
def initialPayment : Nat := 3000
def monthlyInstallment : Nat := 2500
def numInstallments : Nat := 6

theorem proof_MrLalandeInheritance :
  initialPayment + numInstallments * monthlyInstallment = MrLalandeInheritance := 
by 
  sorry

end proof_MrLalandeInheritance_l312_312798


namespace valid_number_count_is_300_l312_312287

-- Define the set of digits
def digits : List ℕ := [0, 1, 2, 3, 4, 5, 6]

-- Define the set of odd digits
def odd_digits : List ℕ := [1, 3, 5]

-- Define a function to count valid four-digit numbers
noncomputable def count_valid_numbers : ℕ :=
  (odd_digits.length * (digits.length - 2) * (digits.length - 2) * (digits.length - 3))

-- State the theorem
theorem valid_number_count_is_300 : count_valid_numbers = 300 :=
  sorry

end valid_number_count_is_300_l312_312287


namespace div30k_929260_l312_312155

theorem div30k_929260 (k : ℕ) (h : 30^k ∣ 929260) : 3^k - k^3 = 1 := by
  sorry

end div30k_929260_l312_312155


namespace binary_addition_is_correct_l312_312404

theorem binary_addition_is_correct :
  (0b101101 + 0b1011 + 0b11001 + 0b1110101 + 0b1111) = 0b10010001 :=
by sorry

end binary_addition_is_correct_l312_312404


namespace distinct_integer_roots_iff_l312_312569

theorem distinct_integer_roots_iff (a : ℤ) :
  (∃ x y : ℤ, x ≠ y ∧ 2 * x^2 - a * x + 2 * a = 0 ∧ 2 * y^2 - a * y + 2 * a = 0) ↔ a = -2 ∨ a = 18 :=
by
  sorry

end distinct_integer_roots_iff_l312_312569


namespace solve_linear_system_l312_312311

theorem solve_linear_system (x y z : ℝ) 
  (h1 : y + z = 20 - 4 * x)
  (h2 : x + z = -10 - 4 * y)
  (h3 : x + y = 14 - 4 * z)
  : 2 * x + 2 * y + 2 * z = 8 :=
by
  sorry

end solve_linear_system_l312_312311


namespace sue_library_inventory_l312_312956

theorem sue_library_inventory :
  let initial_books := 15
  let initial_movies := 6
  let returned_books := 8
  let returned_movies := initial_movies / 3
  let borrowed_more_books := 9
  let current_books := initial_books - returned_books + borrowed_more_books
  let current_movies := initial_movies - returned_movies
  current_books + current_movies = 20 :=
by
  -- no implementation provided
  sorry

end sue_library_inventory_l312_312956


namespace sum_of_infinite_series_l312_312420

theorem sum_of_infinite_series :
  ∑' n, (1 : ℝ) / ((2 * n + 1)^2 - (2 * n - 1)^2) * ((1 : ℝ) / (2 * n - 1)^2 - (1 : ℝ) / (2 * n + 1)^2) = 1 :=
sorry

end sum_of_infinite_series_l312_312420


namespace ratio_area_rectangle_triangle_l312_312061

noncomputable def area_rectangle (L W : ℝ) : ℝ :=
  L * W

noncomputable def area_triangle (L W : ℝ) : ℝ :=
  (1 / 2) * L * W

theorem ratio_area_rectangle_triangle (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  area_rectangle L W / area_triangle L W = 2 :=
by
  -- sorry will be replaced by the actual proof
  sorry

end ratio_area_rectangle_triangle_l312_312061


namespace evaluate_three_cubed_squared_l312_312712

theorem evaluate_three_cubed_squared : (3^3)^2 = 729 :=
by
  -- Given the property of exponents
  have h : (forall (a m n : ℕ), (a^m)^n = a^(m * n)) := sorry,
  -- Now prove the statement using the given property
  calc
    (3^3)^2 = 3^(3 * 2) : by rw [h 3 3 2]
          ... = 3^6       : by norm_num
          ... = 729       : by norm_num

end evaluate_three_cubed_squared_l312_312712


namespace cost_price_for_one_meter_l312_312523

variable (meters_sold : Nat) (selling_price : Nat) (loss_per_meter : Nat) (total_cost_price : Nat)
variable (cost_price_per_meter : Rat)

theorem cost_price_for_one_meter (h1 : meters_sold = 200)
                                  (h2 : selling_price = 12000)
                                  (h3 : loss_per_meter = 12)
                                  (h4 : total_cost_price = selling_price + loss_per_meter * meters_sold)
                                  (h5 : cost_price_per_meter = total_cost_price / meters_sold) :
  cost_price_per_meter = 72 := by
  sorry

end cost_price_for_one_meter_l312_312523


namespace larger_number_l312_312213

theorem larger_number (x y : ℕ) (h1 : x + y = 28) (h2 : x - y = 4) : max x y = 16 := by
  sorry

end larger_number_l312_312213


namespace florist_sold_roses_l312_312535

theorem florist_sold_roses (x : ℕ) (h1 : 5 - x + 34 = 36) : x = 3 :=
by sorry

end florist_sold_roses_l312_312535


namespace competition_result_l312_312782

def fishing_season_days : ℕ := 213
def first_fisherman_rate : ℕ := 3
def second_fisherman_rate1 : ℕ := 1
def second_fisherman_rate2 : ℕ := 2
def second_fisherman_rate3 : ℕ := 4
def first_period_days : ℕ := 30
def second_period_days : ℕ := 60

theorem competition_result :
  let first_fisherman_total := fishing_season_days * first_fisherman_rate in
  let second_fisherman_total := 
    (second_fisherman_rate1 * first_period_days) +
    (second_fisherman_rate2 * second_period_days) +
    (second_fisherman_rate3 * (fishing_season_days - (first_period_days + second_period_days))) in
  second_fisherman_total - first_fisherman_total = 3 :=
by
  sorry

end competition_result_l312_312782


namespace combined_average_l312_312960

-- Given Conditions
def num_results_1 : ℕ := 30
def avg_results_1 : ℝ := 20
def num_results_2 : ℕ := 20
def avg_results_2 : ℝ := 30
def num_results_3 : ℕ := 25
def avg_results_3 : ℝ := 40

-- Helper Definitions
def total_sum_1 : ℝ := num_results_1 * avg_results_1
def total_sum_2 : ℝ := num_results_2 * avg_results_2
def total_sum_3 : ℝ := num_results_3 * avg_results_3
def total_sum_all : ℝ := total_sum_1 + total_sum_2 + total_sum_3
def total_number_results : ℕ := num_results_1 + num_results_2 + num_results_3

-- Problem Statement
theorem combined_average : 
  (total_sum_all / (total_number_results:ℝ)) = 29.33 := 
by 
  sorry

end combined_average_l312_312960


namespace lateral_surface_area_truncated_cone_l312_312049

theorem lateral_surface_area_truncated_cone :
  let r := 1
  let R := 4
  let h := 4
  let l := Real.sqrt ((R - r)^2 + h^2)
  let S := Real.pi * (r + R) * l
  S = 25 * Real.pi :=
by
  sorry

end lateral_surface_area_truncated_cone_l312_312049


namespace solve_first_system_solve_second_system_l312_312348

theorem solve_first_system :
  (exists x y : ℝ, 3 * x + 2 * y = 6 ∧ y = x - 2) ->
  (∃ (x y : ℝ), x = 2 ∧ y = 0) := by
  sorry

theorem solve_second_system :
  (exists m n : ℝ, m + 2 * n = 7 ∧ -3 * m + 5 * n = 1) ->
  (∃ (m n : ℝ), m = 3 ∧ n = 2) := by
  sorry

end solve_first_system_solve_second_system_l312_312348


namespace johns_total_pay_l312_312018

-- Define the given conditions
def lastYearBonus : ℝ := 10000
def CAGR : ℝ := 0.05
def numYears : ℕ := 1
def projectsCompleted : ℕ := 8
def bonusPerProject : ℝ := 2000
def thisYearSalary : ℝ := 200000

-- Define the calculation for the first part of the bonus using the CAGR formula
def firstPartBonus (presentValue : ℝ) (growthRate : ℝ) (years : ℕ) : ℝ :=
  presentValue * (1 + growthRate)^years

-- Define the calculation for the second part of the bonus
def secondPartBonus (numProjects : ℕ) (bonusPerProject : ℝ) : ℝ :=
  numProjects * bonusPerProject

-- Define the total pay calculation
def totalPay (salary : ℝ) (bonus1 : ℝ) (bonus2 : ℝ) : ℝ :=
  salary + bonus1 + bonus2

-- The proof statement, given the conditions, prove the total pay is $226,500
theorem johns_total_pay : totalPay thisYearSalary (firstPartBonus lastYearBonus CAGR numYears) (secondPartBonus projectsCompleted bonusPerProject) = 226500 := 
by
  -- insert proof here
  sorry

end johns_total_pay_l312_312018


namespace math_problem_l312_312264

noncomputable def problem : Real :=
  (2 * Real.sqrt 2 - 1) ^ 2 + (1 + Real.sqrt 5) * (1 - Real.sqrt 5)

theorem math_problem :
  problem = 5 - 4 * Real.sqrt 2 :=
by
  sorry

end math_problem_l312_312264


namespace minimum_value_ineq_l312_312896

theorem minimum_value_ineq (x : ℝ) (hx : 0 < x) :
  3 * Real.sqrt x + 4 / x ≥ 4 * Real.sqrt 2 :=
by
  sorry

end minimum_value_ineq_l312_312896


namespace carson_gold_stars_l312_312106

theorem carson_gold_stars (yesterday_stars today_total_stars earned_today : ℕ) 
  (h1 : yesterday_stars = 6) 
  (h2 : today_total_stars = 15) 
  (h3 : earned_today = today_total_stars - yesterday_stars) 
  : earned_today = 9 :=
sorry

end carson_gold_stars_l312_312106


namespace original_price_of_shoes_l312_312306

theorem original_price_of_shoes (P : ℝ) (h1 : 0.25 * P = 51) : P = 204 := 
by 
  sorry

end original_price_of_shoes_l312_312306


namespace initial_total_fish_l312_312032

def total_days (weeks : ℕ) : ℕ := weeks * 7
def fish_added (rate : ℕ) (days : ℕ) : ℕ := rate * days
def initial_fish (final_count : ℕ) (added : ℕ) : ℕ := final_count - added

theorem initial_total_fish {final_goldfish final_koi rate_goldfish rate_koi days init_goldfish init_koi : ℕ}
    (h_final_goldfish : final_goldfish = 200)
    (h_final_koi : final_koi = 227)
    (h_rate_goldfish : rate_goldfish = 5)
    (h_rate_koi : rate_koi = 2)
    (h_days : days = total_days 3)
    (h_init_goldfish : init_goldfish = initial_fish final_goldfish (fish_added rate_goldfish days))
    (h_init_koi : init_koi = initial_fish final_koi (fish_added rate_koi days)) :
    init_goldfish + init_koi = 280 :=
by
    sorry -- skipping the proof

end initial_total_fish_l312_312032


namespace Dave_has_more_money_than_Derek_l312_312557

def Derek_initial := 40
def Derek_expense1 := 14
def Derek_expense2 := 11
def Derek_expense3 := 5
def Derek_remaining := Derek_initial - Derek_expense1 - Derek_expense2 - Derek_expense3

def Dave_initial := 50
def Dave_expense := 7
def Dave_remaining := Dave_initial - Dave_expense

def money_difference := Dave_remaining - Derek_remaining

theorem Dave_has_more_money_than_Derek : money_difference = 33 := by sorry

end Dave_has_more_money_than_Derek_l312_312557


namespace semicircle_radius_l312_312991

theorem semicircle_radius (π : ℝ) (P : ℝ) (r : ℝ) (hπ : π ≠ 0) (hP : P = 162) (hPerimeter : P = π * r + 2 * r) : r = 162 / (π + 2) :=
by
  sorry

end semicircle_radius_l312_312991


namespace rate_of_drawing_barbed_wire_is_correct_l312_312499

noncomputable def rate_of_drawing_barbed_wire (area cost: ℝ) (gate_width barbed_wire_extension: ℝ) : ℝ :=
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let total_barbed_wire := (perimeter - 2 * gate_width) + 4 * barbed_wire_extension
  cost / total_barbed_wire

theorem rate_of_drawing_barbed_wire_is_correct :
  rate_of_drawing_barbed_wire 3136 666 1 3 = 2.85 :=
by
  sorry

end rate_of_drawing_barbed_wire_is_correct_l312_312499


namespace not_prime_for_all_n_ge_2_l312_312897

theorem not_prime_for_all_n_ge_2 (n : ℕ) (hn : n ≥ 2) : ¬ Prime (2 * (n^3 + n + 1)) := 
by
  sorry

end not_prime_for_all_n_ge_2_l312_312897


namespace fraction_to_decimal_17_625_l312_312421

def fraction_to_decimal (num : ℕ) (den : ℕ) : ℚ := num / den

theorem fraction_to_decimal_17_625 : fraction_to_decimal 17 625 = 272 / 10000 := by
  sorry

end fraction_to_decimal_17_625_l312_312421


namespace chess_game_problem_l312_312802

-- Mathematical definitions based on the conditions
def petr_wins : ℕ := 6
def petr_draws : ℕ := 2
def karel_points : ℤ := 9
def points_for_win : ℕ := 3
def points_for_loss : ℕ := 2
def points_for_draw : ℕ := 0

-- Defining the final statement to prove
theorem chess_game_problem :
    ∃ (total_games : ℕ) (leader : String), total_games = 15 ∧ leader = "Karel" := 
by
  -- Placeholder for proof
  sorry

end chess_game_problem_l312_312802


namespace alyssa_games_this_year_l312_312686

theorem alyssa_games_this_year : 
    ∀ (X: ℕ), 
    (13 + X + 15 = 39) → 
    X = 11 := 
by
  intros X h
  have h₁ : 13 + 15 = 28 := by norm_num
  have h₂ : X + 28 = 39 := by linarith
  have h₃ : X = 11 := by linarith
  exact h₃

end alyssa_games_this_year_l312_312686


namespace profit_per_meter_is_25_l312_312876

def sell_price : ℕ := 8925
def cost_price_per_meter : ℕ := 80
def meters_sold : ℕ := 85
def total_cost_price : ℕ := cost_price_per_meter * meters_sold
def total_profit : ℕ := sell_price - total_cost_price
def profit_per_meter : ℕ := total_profit / meters_sold

theorem profit_per_meter_is_25 : profit_per_meter = 25 := by
  sorry

end profit_per_meter_is_25_l312_312876


namespace evaluate_exp_power_l312_312721

theorem evaluate_exp_power : (3^3)^2 = 729 := 
by {
  sorry
}

end evaluate_exp_power_l312_312721


namespace at_least_one_heart_or_king_l312_312244

-- Define the conditions
def total_cards := 52
def hearts := 13
def kings := 4
def king_of_hearts := 1
def cards_hearts_or_kings := hearts + kings - king_of_hearts

-- Calculate probabilities based on the above conditions
def probability_not_heart_or_king := 
  1 - (cards_hearts_or_kings / total_cards)

def probability_neither_heart_nor_king :=
  (probability_not_heart_or_king) ^ 2

def probability_at_least_one_heart_or_king :=
  1 - probability_neither_heart_nor_king

-- State the theorem to be proved
theorem at_least_one_heart_or_king : 
  probability_at_least_one_heart_or_king = (88 / 169) :=
by
  sorry

end at_least_one_heart_or_king_l312_312244


namespace computers_built_per_month_l312_312533

theorem computers_built_per_month (days_in_month : ℕ) (hours_per_day : ℕ) (computers_per_interval : ℚ) (intervals_per_hour : ℕ)
    (h_days : days_in_month = 28) (h_hours : hours_per_day = 24) (h_computers : computers_per_interval = 2.25) (h_intervals : intervals_per_hour = 2) :
    days_in_month * hours_per_day * intervals_per_hour * computers_per_interval = 3024 :=
by
  -- We would give the proof here, but it's omitted as per instructions.
  sorry

end computers_built_per_month_l312_312533


namespace dollars_sum_l312_312664

theorem dollars_sum : 
  (5 / 8 : ℝ) + (2 / 5) = 1.025 :=
by
  sorry

end dollars_sum_l312_312664


namespace minnie_more_than_week_l312_312554

-- Define the variables and conditions
variable (M : ℕ) -- number of horses Minnie mounts per day
variable (mickey_daily : ℕ) -- number of horses Mickey mounts per day

axiom mickey_daily_formula : mickey_daily = 2 * M - 6
axiom mickey_total_per_week : mickey_daily * 7 = 98
axiom days_in_week : 7 = 7

-- Theorem: Minnie mounts 3 more horses per day than there are days in a week
theorem minnie_more_than_week (M : ℕ) 
  (h1 : mickey_daily = 2 * M - 6)
  (h2 : mickey_daily * 7 = 98)
  (h3 : 7 = 7) :
  M - 7 = 3 := 
sorry

end minnie_more_than_week_l312_312554


namespace sum_of_n_values_l312_312667

theorem sum_of_n_values : ∃ n1 n2 : ℚ, (abs (3 * n1 - 4) = 5) ∧ (abs (3 * n2 - 4) = 5) ∧ n1 + n2 = 8 / 3 :=
by
  sorry

end sum_of_n_values_l312_312667


namespace quadratic_inequality_l312_312462

theorem quadratic_inequality (a b c d x1 x2 x3 x4 : ℝ)
  (h1 : x1 + x2 = -a) 
  (h2 : x1 * x2 = b)
  (h3 : x3 + x4 = -c)
  (h4 : x3 * x4 = d)
  (h5 : b > d)
  (h6 : b > 0)
  (h7 : d > 0) :
  a^2 - c^2 > b - d :=
by
  sorry

end quadratic_inequality_l312_312462


namespace average_sum_problem_l312_312314

theorem average_sum_problem (avg : ℝ) (n : ℕ) (h_avg : avg = 5.3) (h_n : n = 10) : ∃ sum : ℝ, sum = avg * n ∧ sum = 53 :=
by
  sorry

end average_sum_problem_l312_312314


namespace total_cost_is_103_l312_312393

-- Base cost of the plan is 20 dollars
def base_cost : ℝ := 20

-- Cost per text message in dollars
def cost_per_text : ℝ := 0.10

-- Cost per minute over 25 hours in dollars
def cost_per_minute_over_limit : ℝ := 0.15

-- Number of text messages sent
def text_messages : ℕ := 200

-- Total hours talked
def hours_talked : ℝ := 32

-- Free minutes (25 hours)
def free_minutes : ℝ := 25 * 60

-- Calculating the extra minutes talked
def extra_minutes : ℝ := (hours_talked * 60) - free_minutes

-- Total cost
def total_cost : ℝ :=
  base_cost +
  (text_messages * cost_per_text) +
  (extra_minutes * cost_per_minute_over_limit)

-- Proving that the total cost is 103 dollars
theorem total_cost_is_103 : total_cost = 103 := by
  sorry

end total_cost_is_103_l312_312393


namespace correct_expression_l312_312220

variable (a b : ℝ)

theorem correct_expression : (∃ x, x = 3 * a + b^2) ∧ 
    (x = (3 * a + b)^2 ∨ x = 3 * (a + b)^2 ∨ x = 3 * a + b^2 ∨ x = (a + 3 * b)^2) → 
    x = 3 * a + b^2 := by sorry

end correct_expression_l312_312220


namespace union_A_B_inter_A_B_C_U_union_A_B_C_U_inter_A_B_C_U_A_C_U_B_union_C_U_A_C_U_B_inter_C_U_A_C_U_B_l312_312636

def U : Set ℕ := { x | 1 ≤ x ∧ x < 9 }
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5, 6}
def C (S : Set ℕ) : Set ℕ := U \ S

theorem union_A_B : A ∪ B = {1, 2, 3, 4, 5, 6} := 
by {
  -- proof here
  sorry
}

theorem inter_A_B : A ∩ B = {3} := 
by {
  -- proof here
  sorry
}

theorem C_U_union_A_B : C (A ∪ B) = {7, 8} := 
by {
  -- proof here
  sorry
}

theorem C_U_inter_A_B : C (A ∩ B) = {1, 2, 4, 5, 6, 7, 8} := 
by {
  -- proof here
  sorry
}

theorem C_U_A : C A = {4, 5, 6, 7, 8} := 
by {
  -- proof here
  sorry
}

theorem C_U_B : C B = {1, 2, 7, 8} := 
by {
  -- proof here
  sorry
}

theorem union_C_U_A_C_U_B : C A ∪ C B = {1, 2, 4, 5, 6, 7, 8} := 
by {
  -- proof here
  sorry
}

theorem inter_C_U_A_C_U_B : C A ∩ C B = {7, 8} := 
by {
  -- proof here
  sorry
}

end union_A_B_inter_A_B_C_U_union_A_B_C_U_inter_A_B_C_U_A_C_U_B_union_C_U_A_C_U_B_inter_C_U_A_C_U_B_l312_312636


namespace range_of_m_l312_312577

theorem range_of_m (a m : ℝ) (h_a_neg : a < 0) (y1 y2 : ℝ)
  (hA : y1 = a * m^2 - 4 * a * m)
  (hB : y2 = 4 * a * m^2 - 8 * a * m)
  (hA_above : y1 > -3 * a)
  (hB_above : y2 > -3 * a)
  (hy1_gt_y2 : y1 > y2) :
  4 / 3 < m ∧ m < 3 / 2 :=
sorry

end range_of_m_l312_312577


namespace problem_statement_l312_312411

theorem problem_statement : (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / Nat.factorial 8 = 1 := by
  sorry

end problem_statement_l312_312411


namespace evaluate_power_l312_312707

theorem evaluate_power : (3^3)^2 = 729 := 
by 
  sorry

end evaluate_power_l312_312707


namespace sodium_chloride_moles_produced_l312_312124

theorem sodium_chloride_moles_produced (NaOH HCl NaCl : ℕ) : 
    (NaOH = 3) → (HCl = 3) → NaCl = 3 :=
by
  intro hNaOH hHCl
  -- Placeholder for actual proof
  sorry

end sodium_chloride_moles_produced_l312_312124


namespace incorrect_statement_A_l312_312987

-- Define the statements based on conditions
def statementA : String := "INPUT \"MATH=\"; a+b+c"
def statementB : String := "PRINT \"MATH=\"; a+b+c"
def statementC : String := "a=b+c"
def statementD : String := "a=b-c"

-- Define a function to check if a statement is valid syntax
noncomputable def isValidSyntax : String → Prop :=
  λ stmt => 
    stmt = statementB ∨ stmt = statementC ∨ stmt = statementD

-- The proof problem
theorem incorrect_statement_A : ¬ isValidSyntax statementA :=
  sorry

end incorrect_statement_A_l312_312987


namespace sum_of_coefficients_evaluated_l312_312414

theorem sum_of_coefficients_evaluated 
  (x y : ℤ) (h1 : x = 2) (h2 : y = -1)
  : (3 * x + 4 * y)^9 + (2 * x - 5 * y)^9 = 387420501 := 
by
  rw [h1, h2]
  sorry

end sum_of_coefficients_evaluated_l312_312414


namespace probability_of_intersecting_diagonals_l312_312269

-- Define a regular dodecagon
def regular_dodecagon := Finset (Fin 12)

-- Define number of diagonals
def num_diagonals := (regular_dodecagon.card.choose 2) - 12

-- Define number of pairs of diagonals
def num_pairs_diagonals := (num_diagonals.choose 2)

-- Define number of intersecting diagonals
def num_intersecting_diagonals := (regular_dodecagon.card.choose 4)

-- Define the probability that two randomly chosen diagonals intersect inside the dodecagon
def intersection_probability := (num_intersecting_diagonals : ℝ) / (num_pairs_diagonals : ℝ)

theorem probability_of_intersecting_diagonals :
  intersection_probability = (495 / 1431 : ℝ) :=
sorry

end probability_of_intersecting_diagonals_l312_312269


namespace ratio_Raphael_to_Manny_l312_312477

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

end ratio_Raphael_to_Manny_l312_312477


namespace y_sum_equals_three_l312_312334

noncomputable def sum_of_y_values (solutions : List (ℝ × ℝ × ℝ)) : ℝ :=
  solutions.foldl (fun acc (_, y, _) => acc + y) 0

theorem y_sum_equals_three (solutions : List (ℝ × ℝ × ℝ))
  (h1 : ∀ (x y z : ℝ), (x, y, z) ∈ solutions → x + y * z = 5)
  (h2 : ∀ (x y z : ℝ), (x, y, z) ∈ solutions → y + x * z = 8)
  (h3 : ∀ (x y z : ℝ), (x, y, z) ∈ solutions → z + x * y = 12) :
  sum_of_y_values solutions = 3 := sorry

end y_sum_equals_three_l312_312334


namespace probability_divisible_by_3_of_prime_digit_two_digit_numbers_l312_312312

open Nat

def is_prime_digit (d : ℕ) : Prop := d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def valid_two_digit_numbers : List ℕ := 
  [22, 23, 25, 27, 32, 33, 35, 37, 52, 53, 55, 57, 72, 73, 75, 77]

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

theorem probability_divisible_by_3_of_prime_digit_two_digit_numbers :
  let favorable := valid_two_digit_numbers.filter is_divisible_by_3
  (favorable.length : ℚ) / (valid_two_digit_numbers.length : ℚ) = 5 / 16
:= sorry

end probability_divisible_by_3_of_prime_digit_two_digit_numbers_l312_312312


namespace books_taken_out_on_monday_l312_312818

-- Define total number of books initially
def total_books_init := 336

-- Define books taken out on Monday
variable (x : ℕ)

-- Define books brought back on Tuesday
def books_brought_back := 22

-- Define books present after Tuesday
def books_after_tuesday := 234

-- Theorem statement
theorem books_taken_out_on_monday :
  total_books_init - x + books_brought_back = books_after_tuesday → x = 124 :=
by sorry

end books_taken_out_on_monday_l312_312818


namespace time_to_reach_ship_l312_312873

/-- The scuba diver's descent problem -/

def rate_of_descent : ℕ := 35  -- in feet per minute
def depth_of_ship : ℕ := 3500  -- in feet

theorem time_to_reach_ship : depth_of_ship / rate_of_descent = 100 := by
  sorry

end time_to_reach_ship_l312_312873


namespace ellipse_equation_l312_312570

open Real

theorem ellipse_equation (x y : ℝ) (h₁ : (- sqrt 15) = x) (h₂ : (5 / 2) = y)
  (h₃ : ∃ (a b : ℝ), (a > b) ∧ (b > 0) ∧ (a^2 = b^2 + 5) 
  ∧ b^2 = 20 ∧ a^2 = 25) :
  (x^2 / 20 + y^2 / 25 = 1) :=
sorry

end ellipse_equation_l312_312570


namespace third_term_binomial_coefficient_l312_312198

theorem third_term_binomial_coefficient :
  (∃ m : ℕ, m = 4 ∧ ∃ k : ℕ, k = 2 ∧ Nat.choose m k = 6) :=
by
  sorry

end third_term_binomial_coefficient_l312_312198


namespace tangerines_in_basket_l312_312182

/-- Let n be the initial number of tangerines in the basket. -/
theorem tangerines_in_basket
  (n : ℕ)
  (c1 : ∃ m : ℕ, m = 10) -- Minyoung ate 10 tangerines from the basket initially
  (c2 : ∃ k : ℕ, k = 6)  -- An hour later, Minyoung ate 6 more tangerines
  (c3 : n = 10 + 6)      -- The basket was empty after these were eaten
  : n = 16 := sorry

end tangerines_in_basket_l312_312182


namespace find_number_l312_312541

theorem find_number (x : ℤ) (h : 16 * x = 32) : x = 2 :=
sorry

end find_number_l312_312541


namespace range_of_m_l312_312452

noncomputable def f (x : ℝ) (m : ℝ) := 9^x - m * 3^x - 3

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, f x m = 9^x - m * 3^x - 3) →
  (∀ x : ℝ, f (-x) m = - f(x) m) →
  -2 ≤ m :=
sorry

end range_of_m_l312_312452


namespace covered_digits_l312_312394

def four_digit_int (n : ℕ) : Prop := n ≥ 1000 ∧ n < 10000

theorem covered_digits (a b c : ℕ) (n1 n2 n3 : ℕ) :
  four_digit_int n1 → four_digit_int n2 → four_digit_int n3 →
  n1 + n2 + n3 = 10126 →
  (n1 % 10 = 3 ∧ n2 % 10 = 7 ∧ n3 % 10 = 6) →
  (n1 / 10 % 10 = 4 ∧ n2 / 10 % 10 = a ∧ n3 / 10 % 10 = 2) →
  (n1 / 100 % 10 = 2 ∧ n2 / 100 % 10 = 1 ∧ n3 / 100 % 10 = c) →
  (n1 / 1000 = 1 ∧ n2 / 1000 = 2 ∧ n3 / 1000 = b) →
  (a = 5 ∧ b = 6 ∧ c = 7) := 
sorry

end covered_digits_l312_312394


namespace volume_of_rectangular_solid_l312_312766

theorem volume_of_rectangular_solid (a b c : ℝ) (h1 : a * b = Real.sqrt 2) (h2 : b * c = Real.sqrt 3) (h3 : c * a = Real.sqrt 6) : a * b * c = Real.sqrt 6 :=
sorry

end volume_of_rectangular_solid_l312_312766


namespace evaluate_power_l312_312709

theorem evaluate_power : (3^3)^2 = 729 := 
by 
  sorry

end evaluate_power_l312_312709


namespace initially_caught_and_tagged_fish_l312_312608

theorem initially_caught_and_tagged_fish (N T : ℕ) (hN : N = 800) (h_ratio : 2 / 40 = T / N) : T = 40 :=
by
  have hN : N = 800 := hN
  have h_ratio : 2 / 40 = T / 800 := by rw [hN] at h_ratio; exact h_ratio
  sorry

end initially_caught_and_tagged_fish_l312_312608


namespace misread_weight_l312_312352

theorem misread_weight (n : ℕ) (average_incorrect : ℚ) (average_correct : ℚ) (corrected_weight : ℚ) (incorrect_total correct_total diff : ℚ)
  (h1 : n = 20)
  (h2 : average_incorrect = 58.4)
  (h3 : average_correct = 59)
  (h4 : corrected_weight = 68)
  (h5 : incorrect_total = n * average_incorrect)
  (h6 : correct_total = n * average_correct)
  (h7 : diff = correct_total - incorrect_total)
  (h8 : diff = corrected_weight - x) : x = 56 := 
sorry

end misread_weight_l312_312352


namespace integer_equality_condition_l312_312117

theorem integer_equality_condition
  (x y z : ℤ)
  (h : x * (x - y) + y * (y - z) + z * (z - x) = 0) :
  x = y ∧ y = z :=
sorry

end integer_equality_condition_l312_312117


namespace speed_of_stream_l312_312396

theorem speed_of_stream (v : ℝ) (h_still : ∀ (d : ℝ), d / (3 - v) = 2 * d / (3 + v)) : v = 1 :=
by
  sorry

end speed_of_stream_l312_312396


namespace most_stable_scores_l312_312955

structure StudentScores :=
  (average : ℝ)
  (variance : ℝ)

def studentA : StudentScores := { average := 132, variance := 38 }
def studentB : StudentScores := { average := 132, variance := 10 }
def studentC : StudentScores := { average := 132, variance := 26 }

theorem most_stable_scores :
  studentB.variance < studentA.variance ∧ studentB.variance < studentC.variance :=
by 
  sorry

end most_stable_scores_l312_312955


namespace product_of_roots_l312_312582

-- Let x₁ and x₂ be roots of the quadratic equation x^2 + x - 1 = 0
theorem product_of_roots (x₁ x₂ : ℝ) (h₁ : x₁^2 + x₁ - 1 = 0) (h₂ : x₂^2 + x₂ - 1 = 0) :
  x₁ * x₂ = -1 :=
sorry

end product_of_roots_l312_312582


namespace error_in_step_one_l312_312069

theorem error_in_step_one : 
  ∃ a b c d : ℝ, 
    (a * (x + 1) - b = c * (x - 2)) = (3 * (x + 1) - 6 = 2 * (x - 2)) → 
    a ≠ 3 ∨ b ≠ 6 ∨ c ≠ 2 := 
by
  sorry

end error_in_step_one_l312_312069


namespace proof_equiv_expression_l312_312262

variable (x y : ℝ)

def P : ℝ := x^2 + y^2
def Q : ℝ := x^2 - y^2

theorem proof_equiv_expression :
  ( (P x y)^2 + (Q x y)^2 ) / ( (P x y)^2 - (Q x y)^2 ) - 
  ( (P x y)^2 - (Q x y)^2 ) / ( (P x y)^2 + (Q x y)^2 ) = 
  (x^4 - y^4) / (x^2 * y^2) :=
by
  sorry

end proof_equiv_expression_l312_312262


namespace money_distribution_l312_312403

variable (A B C : ℕ)

theorem money_distribution :
  A + B + C = 500 →
  B + C = 360 →
  C = 60 →
  A + C = 200 :=
by
  intros h1 h2 h3
  sorry

end money_distribution_l312_312403


namespace regular_polygon_num_sides_l312_312004

def diag_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem regular_polygon_num_sides (n : ℕ) (h : diag_formula n = 20) : n = 8 :=
by
  sorry

end regular_polygon_num_sides_l312_312004


namespace range_of_m_local_odd_function_l312_312453

def is_local_odd_function (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f (-x) = -f x

noncomputable def f (x m : ℝ) : ℝ :=
  9^x - m * 3^x - 3

theorem range_of_m_local_odd_function :
  (∀ m : ℝ, is_local_odd_function (λ x => f x m) ↔ m ∈ Set.Ici (-2)) :=
by
  sorry

end range_of_m_local_odd_function_l312_312453


namespace at_least_one_prob_better_option_l312_312040

-- Definitions based on the conditions in a)

def player_A_prelim := 1 / 2
def player_B_prelim := 1 / 3
def player_C_prelim := 1 / 2

def final_round := 1 / 3

def prelim_prob_A := player_A_prelim * final_round
def prelim_prob_B := player_B_prelim * final_round
def prelim_prob_C := player_C_prelim * final_round

def prob_none := (1 - prelim_prob_A) * (1 - prelim_prob_B) * (1 - prelim_prob_C)

def prob_at_least_one := 1 - prob_none

-- Question 1 statement

theorem at_least_one_prob :
  prob_at_least_one = 31 / 81 :=
sorry

-- Definitions based on the reward options in the conditions

def option_1_lottery_prob := 1 / 3
def option_1_reward := 600
def option_1_expected_value := 600 * 3 * (1 / 3)

def option_2_prelim_reward := 100
def option_2_final_reward := 400

-- Expected values calculation for Option 2

def option_2_expected_value :=
  (300 * (1 / 6) + 600 * (5 / 12) + 900 * (1 / 3) + 1200 * (1 / 12))

-- Question 2 statement

theorem better_option :
  option_1_expected_value < option_2_expected_value :=
sorry

end at_least_one_prob_better_option_l312_312040


namespace additional_time_due_to_leak_l312_312088

theorem additional_time_due_to_leak 
  (normal_time_per_barrel : ℕ)
  (leak_time_per_barrel : ℕ)
  (barrels : ℕ)
  (normal_duration : normal_time_per_barrel = 3)
  (leak_duration : leak_time_per_barrel = 5)
  (barrels_needed : barrels = 12) :
  (leak_time_per_barrel * barrels - normal_time_per_barrel * barrels) = 24 := 
by
  sorry

end additional_time_due_to_leak_l312_312088


namespace number_of_bookshelves_l312_312172

def total_space : ℕ := 400
def reserved_space : ℕ := 160
def shelf_space : ℕ := 80

theorem number_of_bookshelves : (total_space - reserved_space) / shelf_space = 3 := by
  sorry

end number_of_bookshelves_l312_312172


namespace distinct_ordered_pairs_l312_312588

theorem distinct_ordered_pairs (a b : ℕ) (h : a + b = 40) (ha : a > 0) (hb : b > 0) :
  ∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 39 ∧ ∀ p ∈ pairs, p.1 + p.2 = 40 := 
sorry

end distinct_ordered_pairs_l312_312588


namespace factorization_of_expression_l312_312892

theorem factorization_of_expression (m : ℝ) : 2 * m^2 - 18 = 2 * (m + 3) * (m - 3) :=
by 
  sorry

end factorization_of_expression_l312_312892


namespace winner_percentage_l312_312785

theorem winner_percentage (V_winner V_margin V_total : ℕ) (h_winner: V_winner = 806) (h_margin: V_margin = 312) (h_total: V_total = V_winner + (V_winner - V_margin)) :
  ((V_winner: ℚ) / V_total) * 100 = 62 := by
  sorry

end winner_percentage_l312_312785


namespace find_image_point_l312_312095

noncomputable def lens_equation (t f k : ℝ) : Prop :=
  (1 / k) + (1 / t) = (1 / f)

theorem find_image_point
  (O F T T_star K_star K : ℝ)
  (OT OTw OTw_star FK : ℝ)
  (OT_eq : OT = OTw)
  (OTw_star_eq : OTw_star = OT)
  (similarity_condition : ∀ (CTw_star OF : ℝ), CTw_star / OF = (CTw_star + OK) / OK)
  : lens_equation OTw FK K :=
sorry

end find_image_point_l312_312095


namespace calculate_expr_l312_312693

theorem calculate_expr : (125 : ℝ)^(2/3) * 2 = 50 := sorry

end calculate_expr_l312_312693


namespace probability_exactly_3_tails_l312_312794

noncomputable def binomial_probability_3_tails : ℚ :=
  let n := 8
  let k := 3
  let p := 3/5
  let q := 2/5
  (nat.choose n k : ℚ) * p^k * q^(n-k)

theorem probability_exactly_3_tails : 
  binomial_probability_3_tails = 48624 / 390625 := 
by
  -- Expected result: prob_exactly_3_tails = 48624 / 390625
  sorry

end probability_exactly_3_tails_l312_312794


namespace average_minutes_per_player_is_2_l312_312028

def total_player_footage := 130 + 145 + 85 + 60 + 180
def total_additional_content := 120 + 90 + 30
def pause_transition_time := 15 * (5 + 3) -- 5 players + game footage + interviews + opening/closing scenes - 1
def total_film_time := total_player_footage + total_additional_content + pause_transition_time
def number_of_players := 5
def average_seconds_per_player := total_player_footage / number_of_players
def average_minutes_per_player := average_seconds_per_player / 60

theorem average_minutes_per_player_is_2 :
  average_minutes_per_player = 2 := by
  -- Proof goes here.
  sorry

end average_minutes_per_player_is_2_l312_312028


namespace A_days_to_complete_job_l312_312230

noncomputable def time_for_A (x : ℝ) (work_left : ℝ) : ℝ :=
  let work_rate_A := 1 / x
  let work_rate_B := 1 / 30
  let combined_work_rate := work_rate_A + work_rate_B
  let completed_work := 4 * combined_work_rate
  let fraction_work_left := 1 - completed_work
  fraction_work_left

theorem A_days_to_complete_job : ∃ x : ℝ, time_for_A x 0.6 = 0.6 ∧ x = 15 :=
by {
  use 15,
  sorry
}

end A_days_to_complete_job_l312_312230


namespace number_of_bowls_l312_312842

-- Let n be the number of bowls on the table.
variable (n : ℕ)

-- Condition 1: There are n bowls, and each contain some grapes.
-- Condition 2: Adding 8 grapes to each of 12 specific bowls increases the average number of grapes in all bowls by 6.
-- Let's formalize the condition given in the problem
theorem number_of_bowls (h1 : 12 * 8 = 96) (h2 : 6 * n = 96) : n = 16 :=
by
  -- omitting the proof here
  sorry

end number_of_bowls_l312_312842


namespace option_d_correct_l312_312066

theorem option_d_correct (x y : ℝ) : -4 * x * y + 3 * x * y = -1 * x * y := 
by {
  sorry
}

end option_d_correct_l312_312066


namespace daisy_milk_problem_l312_312114

theorem daisy_milk_problem (total_milk : ℝ) (kids_percentage : ℝ) (remaining_milk : ℝ) (used_milk : ℝ) :
  total_milk = 16 →
  kids_percentage = 0.75 →
  remaining_milk = total_milk * (1 - kids_percentage) →
  used_milk = 2 →
  (used_milk / remaining_milk) * 100 = 50 :=
by
  intros _ _ _ _ 
  sorry

end daisy_milk_problem_l312_312114
