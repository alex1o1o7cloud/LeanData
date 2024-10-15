import Mathlib

namespace NUMINAMATH_GPT_find_a_perpendicular_lines_l1068_106862

theorem find_a_perpendicular_lines 
  (a : ℤ)
  (l1 : ∀ x y : ℤ, a * x + 4 * y + 7 = 0)
  (l2 : ∀ x y : ℤ, 2 * x - 3 * y - 1 = 0) : 
  (∃ a : ℤ, a = 6) :=
by sorry

end NUMINAMATH_GPT_find_a_perpendicular_lines_l1068_106862


namespace NUMINAMATH_GPT_sum_of_integers_l1068_106827

theorem sum_of_integers (numbers : List ℕ) (h1 : numbers.Nodup) 
(h2 : ∃ a b, (a ≠ b ∧ a * b = 16 ∧ a ∈ numbers ∧ b ∈ numbers)) 
(h3 : ∃ c d, (c ≠ d ∧ c * d = 225 ∧ c ∈ numbers ∧ d ∈ numbers)) :
  numbers.sum = 44 :=
sorry

end NUMINAMATH_GPT_sum_of_integers_l1068_106827


namespace NUMINAMATH_GPT_red_light_probability_l1068_106892

theorem red_light_probability :
  let red_duration := 30
  let yellow_duration := 5
  let green_duration := 40
  let total_duration := red_duration + yellow_duration + green_duration
  let probability_of_red := (red_duration:ℝ) / total_duration
  probability_of_red = 2 / 5 := by
    sorry

end NUMINAMATH_GPT_red_light_probability_l1068_106892


namespace NUMINAMATH_GPT_distance_Bella_to_Galya_l1068_106884

theorem distance_Bella_to_Galya (D_B D_V D_G : ℕ) (BV VG : ℕ)
  (hD_B : D_B = 700)
  (hD_V : D_V = 600)
  (hD_G : D_G = 650)
  (hBV : BV = 100)
  (hVG : VG = 50)
  : BV + VG = 150 := by
  sorry

end NUMINAMATH_GPT_distance_Bella_to_Galya_l1068_106884


namespace NUMINAMATH_GPT_least_value_divisibility_l1068_106849

theorem least_value_divisibility : ∃ (x : ℕ), (23 * x) % 3 = 0  ∧ (∀ y : ℕ, ((23 * y) % 3 = 0 → x ≤ y)) := 
  sorry

end NUMINAMATH_GPT_least_value_divisibility_l1068_106849


namespace NUMINAMATH_GPT_six_hundred_sixes_not_square_l1068_106816

theorem six_hundred_sixes_not_square : 
  ∀ (n : ℕ), (n = 66666666666666666666666666666666666666666666666666666666666 -- continued 600 times
  ∨ n = 66666666666666666666666666666666666666666666666666666666666 -- continued with some zeros
  ) → ¬ (∃ k : ℕ, k * k = n) := 
by
  sorry

end NUMINAMATH_GPT_six_hundred_sixes_not_square_l1068_106816


namespace NUMINAMATH_GPT_expression_equivalence_l1068_106826

theorem expression_equivalence :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128) = 5^256 - 4^256 :=
by sorry

end NUMINAMATH_GPT_expression_equivalence_l1068_106826


namespace NUMINAMATH_GPT_annie_weeks_off_sick_l1068_106806

-- Define the conditions and the question
def weekly_hours_chess : ℕ := 2
def weekly_hours_drama : ℕ := 8
def weekly_hours_glee : ℕ := 3
def semester_weeks : ℕ := 12
def total_hours_before_midterms : ℕ := 52

-- Define the proof problem
theorem annie_weeks_off_sick :
  let total_weekly_hours := weekly_hours_chess + weekly_hours_drama + weekly_hours_glee
  let attended_weeks := total_hours_before_midterms / total_weekly_hours
  semester_weeks - attended_weeks = 8 :=
by
  -- Automatically prove by computation of above assumptions.
  sorry

end NUMINAMATH_GPT_annie_weeks_off_sick_l1068_106806


namespace NUMINAMATH_GPT_can_form_isosceles_triangle_with_given_sides_l1068_106850

-- Define a structure for the sides of a triangle
structure Triangle (α : Type _) :=
  (a b c : α)

-- Define the predicate for the triangle inequality
def triangle_inequality {α : Type _} [LinearOrder α] [Add α] (t : Triangle α) : Prop :=
  t.a + t.b > t.c ∧ t.a + t.c > t.b ∧ t.b + t.c > t.a

-- Define the predicate for an isosceles triangle
def is_isosceles {α : Type _} [DecidableEq α] (t : Triangle α) : Prop :=
  t.a = t.b ∨ t.a = t.c ∨ t.b = t.c

-- Define the main theorem which checks if the given sides can form an isosceles triangle
theorem can_form_isosceles_triangle_with_given_sides
  (t : Triangle ℕ)
  (h_tri : triangle_inequality t)
  (h_iso : is_isosceles t) :
  t = ⟨2, 2, 1⟩ :=
  sorry

end NUMINAMATH_GPT_can_form_isosceles_triangle_with_given_sides_l1068_106850


namespace NUMINAMATH_GPT_nylon_needed_is_192_l1068_106868

-- Define the required lengths for the collars
def nylon_needed_for_dog_collar : ℕ := 18
def nylon_needed_for_cat_collar : ℕ := 10

-- Define the number of collars needed
def number_of_dog_collars : ℕ := 9
def number_of_cat_collars : ℕ := 3

-- Define the total nylon needed
def total_nylon_needed : ℕ :=
  (nylon_needed_for_dog_collar * number_of_dog_collars) + (nylon_needed_for_cat_collar * number_of_cat_collars)

-- State the theorem we need to prove
theorem nylon_needed_is_192 : total_nylon_needed = 192 := 
  by
    -- Simplification to match the complete statement for completeness
    sorry

end NUMINAMATH_GPT_nylon_needed_is_192_l1068_106868


namespace NUMINAMATH_GPT_slope_of_line_l1068_106828

theorem slope_of_line (m : ℤ) (hm : (3 * m - 6) / (1 + m) = 12) : m = -2 := 
sorry

end NUMINAMATH_GPT_slope_of_line_l1068_106828


namespace NUMINAMATH_GPT_volume_rotation_l1068_106803

theorem volume_rotation
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (a b : ℝ)
  (h₁ : ∀ (x : ℝ), f x = x^3)
  (h₂ : ∀ (x : ℝ), g x = x^(1/2))
  (h₃ : a = 0)
  (h₄ : b = 1):
  ∫ x in a..b, π * ((g x)^2 - (f x)^2) = 5 * π / 14 :=
by
  sorry

end NUMINAMATH_GPT_volume_rotation_l1068_106803


namespace NUMINAMATH_GPT_largest_angle_in_ratio_triangle_l1068_106820

theorem largest_angle_in_ratio_triangle (x : ℝ) (h1 : 3 * x + 4 * x + 5 * x = 180) : 
  5 * (180 / (3 + 4 + 5)) = 75 := by
  sorry

end NUMINAMATH_GPT_largest_angle_in_ratio_triangle_l1068_106820


namespace NUMINAMATH_GPT_vanessa_score_l1068_106832

theorem vanessa_score (total_points team_score other_players_avg_score: ℝ) : 
  total_points = 72 ∧ team_score = 7 ∧ other_players_avg_score = 4.5 → 
  ∃ vanessa_points: ℝ, vanessa_points = 40.5 :=
by
  sorry

end NUMINAMATH_GPT_vanessa_score_l1068_106832


namespace NUMINAMATH_GPT_fish_game_teams_l1068_106867

noncomputable def number_of_possible_teams (n : ℕ) : ℕ := 
  if n = 6 then 5 else sorry

theorem fish_game_teams : number_of_possible_teams 6 = 5 := by
  unfold number_of_possible_teams
  rfl

end NUMINAMATH_GPT_fish_game_teams_l1068_106867


namespace NUMINAMATH_GPT_faye_scored_47_pieces_l1068_106879

variable (X : ℕ) -- X is the number of pieces of candy Faye scored on Halloween.

-- Definitions based on the conditions
def initial_candy_count (X : ℕ) : ℕ := X - 25
def after_sister_gave_40 (X : ℕ) : ℕ := initial_candy_count X + 40
def current_candy_count (X : ℕ) : ℕ := after_sister_gave_40 X

-- Theorem to prove the number of pieces of candy Faye scored on Halloween
theorem faye_scored_47_pieces (h : current_candy_count X = 62) : X = 47 :=
by
  sorry

end NUMINAMATH_GPT_faye_scored_47_pieces_l1068_106879


namespace NUMINAMATH_GPT_minimum_photos_needed_l1068_106856

theorem minimum_photos_needed 
  (total_photos : ℕ) 
  (photos_IV : ℕ)
  (photos_V : ℕ) 
  (photos_VI : ℕ) 
  (photos_VII : ℕ) 
  (photos_I_III : ℕ) 
  (H : total_photos = 130)
  (H_IV : photos_IV = 35)
  (H_V : photos_V = 30)
  (H_VI : photos_VI = 25)
  (H_VII : photos_VII = 20)
  (H_I_III : photos_I_III = total_photos - (photos_IV + photos_V + photos_VI + photos_VII)) :
  77 = 77 :=
by
  sorry

end NUMINAMATH_GPT_minimum_photos_needed_l1068_106856


namespace NUMINAMATH_GPT_range_of_a_l1068_106843

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - 3 < 0 → a < x) → a ≤ -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1068_106843


namespace NUMINAMATH_GPT_find_divisor_l1068_106883

variable (x y : ℝ)
variable (h1 : (x - 5) / 7 = 7)
variable (h2 : (x - 2) / y = 4)

theorem find_divisor (x y : ℝ) (h1 : (x - 5) / 7 = 7) (h2 : (x - 2) / y = 4) : y = 13 := by
  sorry

end NUMINAMATH_GPT_find_divisor_l1068_106883


namespace NUMINAMATH_GPT_sale_price_of_trouser_l1068_106840

theorem sale_price_of_trouser : (100 - 0.70 * 100) = 30 := by
  sorry

end NUMINAMATH_GPT_sale_price_of_trouser_l1068_106840


namespace NUMINAMATH_GPT_solve_for_x_l1068_106863

theorem solve_for_x (x : ℝ) (h : (2 * x + 7) / 7 = 13) : x = 42 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1068_106863


namespace NUMINAMATH_GPT_hallie_reads_121_pages_on_fifth_day_l1068_106875

-- Definitions for the given conditions.
def book_length : ℕ := 480
def pages_day_one : ℕ := 63
def pages_day_two : ℕ := 95 -- Rounded from 94.5
def pages_day_three : ℕ := 115
def pages_day_four : ℕ := 86 -- Rounded from 86.25

-- Total pages read from day one to day four
def pages_read_first_four_days : ℕ :=
  pages_day_one + pages_day_two + pages_day_three + pages_day_four

-- Conclusion: the number of pages read on the fifth day.
def pages_day_five : ℕ := book_length - pages_read_first_four_days

-- Proof statement: Hallie reads 121 pages on the fifth day.
theorem hallie_reads_121_pages_on_fifth_day :
  pages_day_five = 121 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_hallie_reads_121_pages_on_fifth_day_l1068_106875


namespace NUMINAMATH_GPT_find_xyz_ratio_l1068_106835

theorem find_xyz_ratio (a b c x y z : ℝ) 
  (h1 : x / a + y / b + z / c = 2) 
  (h2 : a^2 / x^2 + b^2 / y^2 + c^2 / z^2 = 1) : 
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 9 :=
by sorry

end NUMINAMATH_GPT_find_xyz_ratio_l1068_106835


namespace NUMINAMATH_GPT_factorize_polynomial_l1068_106865

theorem factorize_polynomial (a b : ℝ) : 
  a^3 * b - 9 * a * b = a * b * (a + 3) * (a - 3) :=
by sorry

end NUMINAMATH_GPT_factorize_polynomial_l1068_106865


namespace NUMINAMATH_GPT_problem1_problem2_l1068_106893

theorem problem1 (x : ℝ) : (x + 3) * (x - 1) ≤ 0 ↔ -3 ≤ x ∧ x ≤ 1 :=
sorry

theorem problem2 (x : ℝ) : (x + 2) * (x - 3) > 0 ↔ x < -2 ∨ x > 3 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1068_106893


namespace NUMINAMATH_GPT_no_intersection_curves_l1068_106802

theorem no_intersection_curves (k : ℕ) (hn : k > 0) 
  (h_intersection : ∀ x y : ℝ, ¬(x^2 + y^2 = k^2 ∧ x * y = k)) : 
  k = 1 := 
sorry

end NUMINAMATH_GPT_no_intersection_curves_l1068_106802


namespace NUMINAMATH_GPT_water_evaporation_weight_l1068_106885

noncomputable def initial_weight : ℝ := 200
noncomputable def initial_salt_concentration : ℝ := 0.05
noncomputable def final_salt_concentration : ℝ := 0.08

theorem water_evaporation_weight (W_final : ℝ) (evaporation_weight : ℝ) 
  (h1 : W_final = 10 / final_salt_concentration) 
  (h2 : evaporation_weight = initial_weight - W_final) : 
  evaporation_weight = 75 :=
by
  sorry

end NUMINAMATH_GPT_water_evaporation_weight_l1068_106885


namespace NUMINAMATH_GPT_mileage_in_scientific_notation_l1068_106860

noncomputable def scientific_notation_of_mileage : Prop :=
  let mileage := 42000
  mileage = 4.2 * 10^4

theorem mileage_in_scientific_notation :
  scientific_notation_of_mileage :=
by
  sorry

end NUMINAMATH_GPT_mileage_in_scientific_notation_l1068_106860


namespace NUMINAMATH_GPT_geometric_sequence_a_eq_2_l1068_106872

theorem geometric_sequence_a_eq_2 (a : ℝ) (h1 : ¬ a = 0) (h2 : (2 * a) ^ 2 = 8 * a) : a = 2 :=
by {
  sorry -- Proof not required, only the statement.
}

end NUMINAMATH_GPT_geometric_sequence_a_eq_2_l1068_106872


namespace NUMINAMATH_GPT_dot_product_a_b_l1068_106808

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 1)

theorem dot_product_a_b : (a.1 * b.1 + a.2 * b.2) = -1 := by
  sorry

end NUMINAMATH_GPT_dot_product_a_b_l1068_106808


namespace NUMINAMATH_GPT_slices_per_pizza_l1068_106836

-- Definitions based on the conditions
def num_pizzas : Nat := 3
def total_cost : Nat := 72
def cost_per_5_slices : Nat := 10

-- To find the number of slices per pizza
theorem slices_per_pizza (num_pizzas : Nat) (total_cost : Nat) (cost_per_5_slices : Nat): 
  (total_cost / num_pizzas) / (cost_per_5_slices / 5) = 12 :=
by
  sorry

end NUMINAMATH_GPT_slices_per_pizza_l1068_106836


namespace NUMINAMATH_GPT_minimum_gumballs_needed_l1068_106859

/-- Alex wants to buy at least 150 gumballs,
    and have exactly 14 gumballs left after dividing evenly among 17 people.
    Determine the minimum number of gumballs Alex should buy. -/
theorem minimum_gumballs_needed (n : ℕ) (h1 : n ≥ 150) (h2 : n % 17 = 14) : n = 150 :=
sorry

end NUMINAMATH_GPT_minimum_gumballs_needed_l1068_106859


namespace NUMINAMATH_GPT_simplify_expression_l1068_106861

theorem simplify_expression :
  ((45 * 2^10) / (15 * 2^5) * 5) = 480 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1068_106861


namespace NUMINAMATH_GPT_find_x_plus_y_l1068_106858

-- Define the vectors
def vector_a : ℝ × ℝ := (1, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -2)
def vector_c (y : ℝ) : ℝ × ℝ := (-1, y)

-- Define the conditions
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0
def parallel (v1 v2 : ℝ × ℝ) : Prop := ∃ k : ℝ, v2.1 = k * v1.1 ∧ v2.2 = k * v1.2

-- State the theorem
theorem find_x_plus_y (x y : ℝ)
  (h1 : perpendicular vector_a (vector_b x))
  (h2 : parallel vector_a (vector_c y)) :
  x + y = 1 :=
sorry

end NUMINAMATH_GPT_find_x_plus_y_l1068_106858


namespace NUMINAMATH_GPT_tenured_professors_percentage_l1068_106842

noncomputable def percentage_tenured (W M T TM : ℝ) := W = 0.69 ∧ (1 - W) = M ∧ (M * 0.52) = TM ∧ (W + T - TM) = 0.90 → T = 0.7512

-- Define the mathematical entities
variables (W M T TM : ℝ)

-- The main statement
theorem tenured_professors_percentage : percentage_tenured W M T TM := by
  sorry

end NUMINAMATH_GPT_tenured_professors_percentage_l1068_106842


namespace NUMINAMATH_GPT_knight_moves_equal_n_seven_l1068_106824

def knight_moves (n : ℕ) : ℕ := sorry -- Function to calculate the minimum number of moves for a knight.

theorem knight_moves_equal_n_seven :
  ∀ {n : ℕ}, n = 7 →
    knight_moves n = knight_moves n := by
  -- Conditions: Position on standard checkerboard 
  -- and the knight moves described above.
  sorry

end NUMINAMATH_GPT_knight_moves_equal_n_seven_l1068_106824


namespace NUMINAMATH_GPT_probability_multiple_4_or_15_l1068_106844

-- Definitions of natural number range and a set of multiples
def first_30_nat_numbers : Finset ℕ := Finset.range 30
def multiples_of (n : ℕ) (s : Finset ℕ) : Finset ℕ := s.filter (λ x => x % n = 0)

-- Conditions
def multiples_of_4 := multiples_of 4 first_30_nat_numbers
def multiples_of_15 := multiples_of 15 first_30_nat_numbers

-- Proof that probability of selecting a multiple of 4 or 15 is 3 / 10
theorem probability_multiple_4_or_15 : 
  let favorable_outcomes := (multiples_of_4 ∪ multiples_of_15).card
  let total_outcomes := first_30_nat_numbers.card
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  -- correct answer based on the computation
  sorry

end NUMINAMATH_GPT_probability_multiple_4_or_15_l1068_106844


namespace NUMINAMATH_GPT_root_of_quadratic_eq_when_C_is_3_l1068_106829

-- Define the quadratic equation and the roots we are trying to prove
def quadratic_eq (C : ℝ) (x : ℝ) := 3 * x^2 - 6 * x + C = 0

-- Set the constant C to 3
def C : ℝ := 3

-- State the theorem that proves the root of the equation when C=3 is x=1
theorem root_of_quadratic_eq_when_C_is_3 : quadratic_eq C 1 :=
by
  -- Skip the detailed proof
  sorry

end NUMINAMATH_GPT_root_of_quadratic_eq_when_C_is_3_l1068_106829


namespace NUMINAMATH_GPT_original_faculty_members_correct_l1068_106822

noncomputable def original_faculty_members : ℝ := 282

theorem original_faculty_members_correct:
  ∃ F : ℝ, (0.6375 * F = 180) ∧ (F = original_faculty_members) :=
by
  sorry

end NUMINAMATH_GPT_original_faculty_members_correct_l1068_106822


namespace NUMINAMATH_GPT_func_above_x_axis_l1068_106833

theorem func_above_x_axis (a : ℝ) :
  (∀ x : ℝ, (x^4 + 4*x^3 + a*x^2 - 4*x + 1) > 0) ↔ a > 2 :=
sorry

end NUMINAMATH_GPT_func_above_x_axis_l1068_106833


namespace NUMINAMATH_GPT_equivalence_l1068_106819

theorem equivalence (a b c : ℝ) (h : a + c = 2 * b) : a^2 + 8 * b * c = (2 * b + c)^2 := 
by 
  sorry

end NUMINAMATH_GPT_equivalence_l1068_106819


namespace NUMINAMATH_GPT_average_of_second_set_of_two_numbers_l1068_106854

theorem average_of_second_set_of_two_numbers
  (S : ℝ)
  (avg1 avg2 avg3 : ℝ)
  (h1 : S = 6 * 3.95)
  (h2 : avg1 = 3.4)
  (h3 : avg3 = 4.6) :
  (S - (2 * avg1) - (2 * avg3)) / 2 = 3.85 :=
by
  sorry

end NUMINAMATH_GPT_average_of_second_set_of_two_numbers_l1068_106854


namespace NUMINAMATH_GPT_recover_original_sequence_l1068_106886

theorem recover_original_sequence :
  ∃ (a d : ℤ),
    [a, a + d, a + 2 * d, a + 3 * d, a + 4 * d, a + 5 * d] = [113, 125, 137, 149, 161, 173] :=
by
  sorry

end NUMINAMATH_GPT_recover_original_sequence_l1068_106886


namespace NUMINAMATH_GPT_solve_equation_l1068_106899

theorem solve_equation (x : ℝ) (h : x ≠ 1) : 
  (3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1) → 
  x = -4 ∨ x = -2 :=
by 
  sorry

end NUMINAMATH_GPT_solve_equation_l1068_106899


namespace NUMINAMATH_GPT_pen_price_l1068_106821

theorem pen_price (p : ℝ) (h : 30 = 10 * p + 10 * (p / 2)) : p = 2 :=
sorry

end NUMINAMATH_GPT_pen_price_l1068_106821


namespace NUMINAMATH_GPT_transformed_curve_l1068_106869

theorem transformed_curve (x y : ℝ) :
  (y * Real.cos x + 2 * y - 1 = 0) →
  (y - 1) * Real.sin x + 2 * y - 3 = 0 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_transformed_curve_l1068_106869


namespace NUMINAMATH_GPT_set_A_is_2_3_l1068_106805

noncomputable def A : Set ℤ := { x : ℤ | 3 / (x - 1) > 1 }

theorem set_A_is_2_3 : A = {2, 3} :=
by
  sorry

end NUMINAMATH_GPT_set_A_is_2_3_l1068_106805


namespace NUMINAMATH_GPT_hyperbola_standard_equation_l1068_106845

theorem hyperbola_standard_equation (a b : ℝ) (x y : ℝ)
  (H₁ : 2 * a = 2) -- length of the real axis is 2
  (H₂ : y = 2 * x) -- one of its asymptote equations
  : y^2 - 4 * x^2 = 1 :=
sorry

end NUMINAMATH_GPT_hyperbola_standard_equation_l1068_106845


namespace NUMINAMATH_GPT_remy_gallons_l1068_106847

noncomputable def gallons_used (R : ℝ) : ℝ :=
  let remy := 3 * R + 1
  let riley := (R + remy) - 2
  let ronan := riley / 2
  R + remy + riley + ronan

theorem remy_gallons : ∃ R : ℝ, gallons_used R = 60 ∧ (3 * R + 1) = 18.85 :=
by
  sorry

end NUMINAMATH_GPT_remy_gallons_l1068_106847


namespace NUMINAMATH_GPT_people_off_second_eq_8_l1068_106888

-- Initial number of people on the bus
def initial_people := 50

-- People who got off at the first stop
def people_off_first := 15

-- People who got on at the second stop
def people_on_second := 2

-- People who got off at the second stop (unknown, let's call it x)
variable (x : ℕ)

-- People who got off at the third stop
def people_off_third := 4

-- People who got on at the third stop
def people_on_third := 3

-- Number of people on the bus after the third stop
def people_after_third := 28

-- Equation formed by given conditions
def equation := initial_people - people_off_first - x + people_on_second - people_off_third + people_on_third = people_after_third

-- Goal: Prove the equation with given conditions results in x = 8
theorem people_off_second_eq_8 : equation x → x = 8 := by
  sorry

end NUMINAMATH_GPT_people_off_second_eq_8_l1068_106888


namespace NUMINAMATH_GPT_dry_grapes_weight_l1068_106831

theorem dry_grapes_weight (W_fresh : ℝ) (W_dry : ℝ) (P_water_fresh : ℝ) (P_water_dry : ℝ) :
  W_fresh = 40 → P_water_fresh = 0.80 → P_water_dry = 0.20 → W_dry = 10 := 
by 
  intros hWf hPwf hPwd 
  sorry

end NUMINAMATH_GPT_dry_grapes_weight_l1068_106831


namespace NUMINAMATH_GPT_unfair_coin_probability_l1068_106807

theorem unfair_coin_probability (P : ℕ → ℝ) :
  let heads := 3/4
  let initial_condition := P 0 = 1
  let recurrence_relation := ∀n, P (n + 1) = 3 / 4 * (1 - P n) + 1 / 4 * P n
  recurrence_relation →
  initial_condition →
  P 40 = 1 / 2 * (1 + (1 / 2) ^ 40) :=
by
  sorry

end NUMINAMATH_GPT_unfair_coin_probability_l1068_106807


namespace NUMINAMATH_GPT_cos_180_eq_neg1_sin_180_eq_0_l1068_106800

theorem cos_180_eq_neg1 : Real.cos (180 * Real.pi / 180) = -1 := sorry
theorem sin_180_eq_0 : Real.sin (180 * Real.pi / 180) = 0 := sorry

end NUMINAMATH_GPT_cos_180_eq_neg1_sin_180_eq_0_l1068_106800


namespace NUMINAMATH_GPT_proof_expr1_l1068_106864

noncomputable def expr1 : ℝ :=
  (Real.sin (65 * Real.pi / 180) + Real.sin (15 * Real.pi / 180) * Real.sin (10 * Real.pi / 180)) / 
  (Real.sin (25 * Real.pi / 180) - Real.cos (15 * Real.pi / 180) * Real.cos (80 * Real.pi / 180))

theorem proof_expr1 : expr1 = 2 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_GPT_proof_expr1_l1068_106864


namespace NUMINAMATH_GPT_total_pears_picked_l1068_106813

def pears_Alyssa : ℕ := 42
def pears_Nancy : ℕ := 17

theorem total_pears_picked : pears_Alyssa + pears_Nancy = 59 :=
by sorry

end NUMINAMATH_GPT_total_pears_picked_l1068_106813


namespace NUMINAMATH_GPT_total_hiking_distance_l1068_106811

def saturday_distance : ℝ := 8.2
def sunday_distance : ℝ := 1.6
def total_distance (saturday_distance sunday_distance : ℝ) : ℝ := saturday_distance + sunday_distance

theorem total_hiking_distance :
  total_distance saturday_distance sunday_distance = 9.8 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_total_hiking_distance_l1068_106811


namespace NUMINAMATH_GPT_symmetry_condition_l1068_106887

theorem symmetry_condition 
  (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) : 
  (∀ a b : ℝ, b = 2 * a → (∃ y, y = (p * (b/2) + 2*q) / (r * (b/2) + 2*s) ∧  b = 2*(y/2) )) → 
  p + r = 0 :=
by
  sorry

end NUMINAMATH_GPT_symmetry_condition_l1068_106887


namespace NUMINAMATH_GPT_monotonically_increasing_l1068_106895

variable {R : Type} [LinearOrderedField R]

def f (x : R) : R := 3 * x + 1

theorem monotonically_increasing : ∀ x₁ x₂ : R, x₁ < x₂ → f x₁ < f x₂ :=
by
  intro x₁ x₂ h
 -- this is where the proof would go
  sorry

end NUMINAMATH_GPT_monotonically_increasing_l1068_106895


namespace NUMINAMATH_GPT_smallest_portion_is_two_l1068_106870

theorem smallest_portion_is_two (a1 a2 a3 a4 a5 : ℕ) (d : ℕ) (h1 : a1 = a3 - 2 * d) (h2 : a2 = a3 - d) (h3 : a4 = a3 + d) (h4 : a5 = a3 + 2 * d) (h5 : a1 + a2 + a3 + a4 + a5 = 120) (h6 : a3 + a4 + a5 = 7 * (a1 + a2)) : a1 = 2 :=
by sorry

end NUMINAMATH_GPT_smallest_portion_is_two_l1068_106870


namespace NUMINAMATH_GPT_star_comm_star_distrib_over_add_star_special_case_star_no_identity_star_not_assoc_l1068_106814

def star (x y : ℤ) := (x + 2) * (y + 2) - 2

-- Statement A: commutativity
theorem star_comm : ∀ x y : ℤ, star x y = star y x := 
by sorry

-- Statement B: distributivity over addition
theorem star_distrib_over_add : ¬(∀ x y z : ℤ, star x (y + z) = star x y + star x z) :=
by sorry

-- Statement C: special case
theorem star_special_case : ¬(∀ x : ℤ, star (x - 2) (x + 2) = star x x - 2) :=
by sorry

-- Statement D: identity element
theorem star_no_identity : ¬(∃ e : ℤ, ∀ x : ℤ, star x e = x ∧ star e x = x) :=
by sorry

-- Statement E: associativity
theorem star_not_assoc : ¬(∀ x y z : ℤ, star (star x y) z = star x (star y z)) :=
by sorry

end NUMINAMATH_GPT_star_comm_star_distrib_over_add_star_special_case_star_no_identity_star_not_assoc_l1068_106814


namespace NUMINAMATH_GPT_parallelogram_height_l1068_106846

theorem parallelogram_height (A B H : ℝ) 
    (h₁ : A = 96) 
    (h₂ : B = 12) 
    (h₃ : A = B * H) :
  H = 8 := 
by {
  sorry
}

end NUMINAMATH_GPT_parallelogram_height_l1068_106846


namespace NUMINAMATH_GPT_train_speed_l1068_106880

noncomputable def train_length : ℝ := 65 -- length of the train in meters
noncomputable def time_to_pass : ℝ := 6.5 -- time to pass the telegraph post in seconds
noncomputable def speed_conversion_factor : ℝ := 18 / 5 -- conversion factor from m/s to km/h

theorem train_speed (h_length : train_length = 65) (h_time : time_to_pass = 6.5) :
  (train_length / time_to_pass) * speed_conversion_factor = 36 :=
by
  simp [h_length, h_time, train_length, time_to_pass, speed_conversion_factor]
  sorry

end NUMINAMATH_GPT_train_speed_l1068_106880


namespace NUMINAMATH_GPT_value_of_y_l1068_106818

theorem value_of_y (y : ℝ) (h : (y / 5) / 3 = 5 / (y / 3)) : y = 15 ∨ y = -15 :=
by
  sorry

end NUMINAMATH_GPT_value_of_y_l1068_106818


namespace NUMINAMATH_GPT_selected_numbers_satisfy_conditions_l1068_106823

theorem selected_numbers_satisfy_conditions :
  ∃ (nums : Finset ℕ), 
  nums = {6, 34, 35, 51, 55, 77} ∧
  (∀ (a b c : ℕ), a ∈ nums → b ∈ nums → c ∈ nums → a ≠ b → a ≠ c → b ≠ c → 
    gcd a b = 1 ∨ gcd b c = 1 ∨ gcd c a = 1) ∧
  (∀ (x y z : ℕ), x ∈ nums → y ∈ nums → z ∈ nums → x ≠ y → x ≠ z → y ≠ z → 
    gcd x y ≠ 1 ∨ gcd y z ≠ 1 ∨ gcd z x ≠ 1) := 
sorry

end NUMINAMATH_GPT_selected_numbers_satisfy_conditions_l1068_106823


namespace NUMINAMATH_GPT_ellipse_sum_l1068_106841

theorem ellipse_sum (F1 F2 : ℝ × ℝ) (h k a b : ℝ) 
  (hf1 : F1 = (0, 0)) (hf2 : F2 = (6, 0))
  (h_eqn : ∀ P : ℝ × ℝ, dist P F1 + dist P F2 = 10) :
  h + k + a + b = 12 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_sum_l1068_106841


namespace NUMINAMATH_GPT_cubic_coeff_relationship_l1068_106838

theorem cubic_coeff_relationship (a b c d u v w : ℝ) 
  (h_eq : a * (u^3) + b * (u^2) + c * u + d = 0)
  (h_vieta1 : u + v + w = -(b / a)) 
  (h_vieta2 : u * v + u * w + v * w = c / a) 
  (h_vieta3 : u * v * w = -d / a) 
  (h_condition : u + v = u * v) :
  (c + d) * (b + c + d) = a * d :=
by 
  sorry

end NUMINAMATH_GPT_cubic_coeff_relationship_l1068_106838


namespace NUMINAMATH_GPT_simplify_expression_l1068_106852

theorem simplify_expression :
  ( ( (11 / 4) / (11 / 10 + 10 / 3) ) / ( 5 / 2 - ( 4 / 3 ) ) ) /
  ( ( 5 / 7 ) - ( ( (13 / 6 + 9 / 2) * 3 / 8 ) / (11 / 4 - 3 / 2) ) )
  = - (35 / 9) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1068_106852


namespace NUMINAMATH_GPT_regular_polygon_exterior_angle_l1068_106896

theorem regular_polygon_exterior_angle (n : ℕ) (h : 60 * n = 360) : n = 6 :=
sorry

end NUMINAMATH_GPT_regular_polygon_exterior_angle_l1068_106896


namespace NUMINAMATH_GPT_part_I_part_II_l1068_106817

open Set

-- Define the sets A and B
def A : Set ℝ := { x | 1 < x ∧ x < 2 }
def B (a : ℝ) : Set ℝ := { x | 2 * a - 1 < x ∧ x < 2 * a + 1 }

-- Part (Ⅰ): Given A ⊆ B, prove that 1/2 ≤ a ≤ 1
theorem part_I (a : ℝ) : A ⊆ B a → (1 / 2 ≤ a ∧ a ≤ 1) :=
by sorry

-- Part (Ⅱ): Given A ∩ B = ∅, prove that a ≥ 3/2 or a ≤ 0
theorem part_II (a : ℝ) : A ∩ B a = ∅ → (a ≥ 3 / 2 ∨ a ≤ 0) :=
by sorry

end NUMINAMATH_GPT_part_I_part_II_l1068_106817


namespace NUMINAMATH_GPT_orvin_balloons_l1068_106871

def regular_price : ℕ := 2
def total_money_initial := 42 * regular_price
def pair_cost := regular_price + (regular_price / 2)
def pairs := total_money_initial / pair_cost
def balloons_from_sale := pairs * 2

def extra_money : ℕ := 18
def price_per_additional_balloon := 2 * regular_price
def additional_balloons := extra_money / price_per_additional_balloon
def greatest_number_of_balloons := balloons_from_sale + additional_balloons

theorem orvin_balloons (pairs balloons_from_sale additional_balloons greatest_number_of_balloons : ℕ) :
  pairs * 2 = 56 →
  additional_balloons = 4 →
  greatest_number_of_balloons = 60 :=
by
  sorry

end NUMINAMATH_GPT_orvin_balloons_l1068_106871


namespace NUMINAMATH_GPT_fred_speed_5_mph_l1068_106830

theorem fred_speed_5_mph (F : ℝ) (h1 : 50 = 25 + 25) (h2 : 25 / 5 = 5) (h3 : 25 / F = 5) : 
  F = 5 :=
by
  -- Since Fred's speed makes meeting with Sam in the same time feasible
  sorry

end NUMINAMATH_GPT_fred_speed_5_mph_l1068_106830


namespace NUMINAMATH_GPT_arc_length_solution_l1068_106851

variable (r : ℝ) (α : ℝ)

theorem arc_length_solution (h1 : r = 8) (h2 : α = 5 * Real.pi / 3) : 
    r * α = 40 * Real.pi / 3 := 
by 
    sorry

end NUMINAMATH_GPT_arc_length_solution_l1068_106851


namespace NUMINAMATH_GPT_no_integer_in_interval_l1068_106890

theorem no_integer_in_interval (n : ℕ) : ¬ ∃ k : ℤ, 
  (n ≠ 0 ∧ (n * Real.sqrt 2 - 1 / (3 * n) < k) ∧ (k < n * Real.sqrt 2 + 1 / (3 * n))) := 
sorry

end NUMINAMATH_GPT_no_integer_in_interval_l1068_106890


namespace NUMINAMATH_GPT_desk_length_l1068_106855

theorem desk_length (width perimeter length : ℤ) (h1 : width = 9) (h2 : perimeter = 46) (h3 : perimeter = 2 * (length + width)) : length = 14 :=
by
  rw [h1, h2] at h3
  sorry

end NUMINAMATH_GPT_desk_length_l1068_106855


namespace NUMINAMATH_GPT_length_AB_of_parabola_l1068_106801

theorem length_AB_of_parabola (x1 x2 : ℝ)
  (h : x1 + x2 = 6) :
  abs (x1 + x2 + 2) = 8 := by
  sorry

end NUMINAMATH_GPT_length_AB_of_parabola_l1068_106801


namespace NUMINAMATH_GPT_domain_of_g_x_l1068_106877

theorem domain_of_g_x :
  ∀ x, (x ≤ 6 ∧ x ≥ -19) ↔ -19 ≤ x ∧ x ≤ 6 :=
by 
  -- Statement only, no proof
  sorry

end NUMINAMATH_GPT_domain_of_g_x_l1068_106877


namespace NUMINAMATH_GPT_value_of_4_ampersand_neg3_l1068_106881

-- Define the operation '&'
def ampersand (x y : Int) : Int :=
  x * (y + 2) + x * y

-- State the theorem
theorem value_of_4_ampersand_neg3 : ampersand 4 (-3) = -16 :=
by
  sorry

end NUMINAMATH_GPT_value_of_4_ampersand_neg3_l1068_106881


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1068_106891

theorem problem1 : -15 + (-23) - 26 - (-15) = -49 := 
by sorry

theorem problem2 : (- (1 / 2) + (2 / 3) - (1 / 4)) * (-24) = 2 := 
by sorry

theorem problem3 : -24 / (-6) * (- (1 / 4)) = -1 := 
by sorry

theorem problem4 : -1 ^ 2024 - (-2) ^ 3 - 3 ^ 2 + 2 / (2 / 3 * (3 / 2)) = 5 / 2 := 
by sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1068_106891


namespace NUMINAMATH_GPT_real_part_of_complex_l1068_106897

theorem real_part_of_complex (a : ℝ) (h : a^2 + 2 * a - 15 = 0 ∧ a + 5 ≠ 0) : a = 3 :=
by sorry

end NUMINAMATH_GPT_real_part_of_complex_l1068_106897


namespace NUMINAMATH_GPT_four_c_plus_d_l1068_106834

theorem four_c_plus_d (c d : ℝ) (h1 : 2 * c = -6) (h2 : c^2 - d = 1) : 4 * c + d = -4 :=
by
  sorry

end NUMINAMATH_GPT_four_c_plus_d_l1068_106834


namespace NUMINAMATH_GPT_factory_production_system_l1068_106876

theorem factory_production_system (x y : ℕ) (h1 : x + y = 95)
    (h2 : 8*x - 22*y = 0) :
    16*x - 22*y = 0 :=
by
  sorry

end NUMINAMATH_GPT_factory_production_system_l1068_106876


namespace NUMINAMATH_GPT_adam_earnings_correct_l1068_106825

def total_earnings (lawns_mowed lawns_to_mow : ℕ) (lawn_pay : ℕ)
                   (cars_washed cars_to_wash : ℕ) (car_pay_euros : ℕ) (euro_to_dollar : ℝ)
                   (dogs_walked dogs_to_walk : ℕ) (dog_pay_pesos : ℕ) (peso_to_dollar : ℝ) : ℝ :=
  let lawn_earnings := lawns_mowed * lawn_pay
  let car_earnings := (cars_washed * car_pay_euros : ℝ) * euro_to_dollar
  let dog_earnings := (dogs_walked * dog_pay_pesos : ℝ) * peso_to_dollar
  lawn_earnings + car_earnings + dog_earnings

theorem adam_earnings_correct :
  total_earnings 4 12 9 4 6 10 1.1 3 4 50 0.05 = 87.5 :=
by
  sorry

end NUMINAMATH_GPT_adam_earnings_correct_l1068_106825


namespace NUMINAMATH_GPT_earnings_difference_l1068_106873

-- Definitions:
def investments_ratio := (3, 4, 5)
def return_ratio := (6, 5, 4)
def total_earnings := 5800

-- Target statement:
theorem earnings_difference (x y : ℝ)
  (h_investment_ratio : investments_ratio = (3, 4, 5))
  (h_return_ratio : return_ratio = (6, 5, 4))
  (h_total_earnings : (3 * x * 6 * y) / 100 + (4 * x * 5 * y) / 100 + (5 * x * 4 * y) / 100 = total_earnings) :
  ((4 * x * 5 * y) / 100 - (3 * x * 6 * y) / 100) = 200 := 
by
  sorry

end NUMINAMATH_GPT_earnings_difference_l1068_106873


namespace NUMINAMATH_GPT_unique_positive_integer_solution_l1068_106853

theorem unique_positive_integer_solution (n p : ℕ) (x y : ℕ) :
  (x + p * y = n ∧ x + y = p^2 ∧ x > 0 ∧ y > 0) ↔ 
  (p > 1 ∧ (p - 1) ∣ (n - 1) ∧ ∀ k : ℕ, n ≠ p^k ∧ ∃! t : ℕ × ℕ, (t.1 + p * t.2 = n ∧ t.1 + t.2 = p^2 ∧ t.1 > 0 ∧ t.2 > 0)) :=
by
  sorry

end NUMINAMATH_GPT_unique_positive_integer_solution_l1068_106853


namespace NUMINAMATH_GPT_mohan_cookies_l1068_106857

theorem mohan_cookies :
  ∃ (a : ℕ), 
    (a % 6 = 5) ∧ 
    (a % 7 = 3) ∧ 
    (a % 9 = 7) ∧ 
    (a % 11 = 10) ∧ 
    (a = 1817) :=
sorry

end NUMINAMATH_GPT_mohan_cookies_l1068_106857


namespace NUMINAMATH_GPT_factorize_expression_l1068_106889

theorem factorize_expression (m : ℝ) : 2 * m^2 - 8 = 2 * (m + 2) * (m - 2) :=
sorry

end NUMINAMATH_GPT_factorize_expression_l1068_106889


namespace NUMINAMATH_GPT_find_prime_pairs_l1068_106815

def is_prime (n : ℕ) := n ≥ 2 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def has_prime_root (m n : ℕ) : Prop :=
  ∃ (p: ℕ), is_prime p ∧ (p * p - m * p - n = 0)

theorem find_prime_pairs :
  ∀ (m n : ℕ), (is_prime m ∧ is_prime n) → has_prime_root m n → (m, n) = (2, 3) :=
by sorry

end NUMINAMATH_GPT_find_prime_pairs_l1068_106815


namespace NUMINAMATH_GPT_truck_stops_l1068_106874

variable (a : ℕ → ℕ)
variable (sum_1 : ℕ)
variable (sum_2 : ℕ)

-- Definition for the first sequence with a common difference of -10
def first_sequence : ℕ → ℕ
| 0       => 40
| (n + 1) => first_sequence n - 10

-- Definition for the second sequence with a common difference of -5
def second_sequence : ℕ → ℕ 
| 0       => 10
| (n + 1) => second_sequence n - 5

-- Summing the first sequence elements before the condition change:
def sum_first_sequence : ℕ → ℕ 
| 0       => 40
| (n + 1) => sum_first_sequence n + first_sequence (n + 1)

-- Summing the second sequence elements after the condition change:
def sum_second_sequence : ℕ → ℕ 
| 0       => second_sequence 0
| (n + 1) => sum_second_sequence n + second_sequence (n + 1)

-- Final sum of distances
def total_distance : ℕ :=
  sum_first_sequence 3 + sum_second_sequence 1

theorem truck_stops (sum_1 sum_2 : ℕ) (h1 : sum_1 = sum_first_sequence 3)
 (h2 : sum_2 = sum_second_sequence 1) : 
  total_distance = 115 := by
  sorry


end NUMINAMATH_GPT_truck_stops_l1068_106874


namespace NUMINAMATH_GPT_distribute_pencils_l1068_106837

theorem distribute_pencils (number_of_pencils : ℕ) (number_of_people : ℕ)
  (h_pencils : number_of_pencils = 2) (h_people : number_of_people = 5) :
  number_of_distributions = 15 := by
  sorry

end NUMINAMATH_GPT_distribute_pencils_l1068_106837


namespace NUMINAMATH_GPT_find_m_l1068_106839

-- Define the hyperbola equation
def hyperbola1 (x y : ℝ) (m : ℝ) : Prop := (x^3 / m) - (y^2 / 3) = 1
def hyperbola2 (x y : ℝ) : Prop := (x^3 / 8) - (y^2 / 4) = 1

-- Define the condition for eccentricity equivalence
def same_eccentricity (m : ℝ) : Prop :=
  let e1_sq := 1 + (4 / 2^2)
  let e2_sq := 1 + (3 / m)
  e1_sq = e2_sq

-- The main theorem statement
theorem find_m (m : ℝ) : hyperbola1 x y m → hyperbola2 x y → same_eccentricity m → m = 6 :=
by
  -- Proof can be skipped with sorry to satisfy the statement-only requirement
  sorry

end NUMINAMATH_GPT_find_m_l1068_106839


namespace NUMINAMATH_GPT_sheena_sewing_weeks_l1068_106810

theorem sheena_sewing_weeks (sew_time : ℕ) (bridesmaids : ℕ) (sewing_per_week : ℕ) 
    (h_sew_time : sew_time = 12) (h_bridesmaids : bridesmaids = 5) (h_sewing_per_week : sewing_per_week = 4) : 
    (bridesmaids * sew_time) / sewing_per_week = 15 := 
  by sorry

end NUMINAMATH_GPT_sheena_sewing_weeks_l1068_106810


namespace NUMINAMATH_GPT_sales_on_same_days_l1068_106804

-- Definitions representing the conditions
def bookstore_sales_days : List ℕ := [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
def toy_store_sales_days : List ℕ := [2, 9, 16, 23, 30]

-- Lean statement to prove the number of common sale days
theorem sales_on_same_days : (bookstore_sales_days ∩ toy_store_sales_days).length = 2 :=
by sorry

end NUMINAMATH_GPT_sales_on_same_days_l1068_106804


namespace NUMINAMATH_GPT_total_sections_formed_l1068_106866

theorem total_sections_formed (boys girls : ℕ) (hb : boys = 408) (hg : girls = 264) :
  let gcd := Nat.gcd boys girls
  let boys_sections := boys / gcd
  let girls_sections := girls / gcd
  boys_sections + girls_sections = 28 := 
by
  -- Note: this will assert the theorem, but the proof is omitted with sorry.
  sorry

end NUMINAMATH_GPT_total_sections_formed_l1068_106866


namespace NUMINAMATH_GPT_line_b_parallel_or_in_plane_l1068_106882

def Line : Type := sorry    -- Placeholder for the type of line
def Plane : Type := sorry   -- Placeholder for the type of plane

def is_parallel (a b : Line) : Prop := sorry             -- Predicate for parallel lines
def is_parallel_to_plane (l : Line) (p : Plane) : Prop := sorry   -- Predicate for a line being parallel to a plane
def lies_in_plane (l : Line) (p : Plane) : Prop := sorry          -- Predicate for a line lying in a plane

theorem line_b_parallel_or_in_plane (a b : Line) (α : Plane) 
  (h1 : is_parallel a b) 
  (h2 : is_parallel_to_plane a α) : 
  is_parallel_to_plane b α ∨ lies_in_plane b α :=
sorry

end NUMINAMATH_GPT_line_b_parallel_or_in_plane_l1068_106882


namespace NUMINAMATH_GPT_reptile_house_animal_multiple_l1068_106809

theorem reptile_house_animal_multiple (R F x : ℕ) (hR : R = 16) (hF : F = 7) (hCond : R = x * F - 5) : x = 3 := by
  sorry

end NUMINAMATH_GPT_reptile_house_animal_multiple_l1068_106809


namespace NUMINAMATH_GPT_sheets_in_a_bundle_l1068_106894

variable (B : ℕ) -- Denotes the number of sheets in a bundle

-- Conditions
variable (NumBundles NumBunches NumHeaps : ℕ)
variable (SheetsPerBunch SheetsPerHeap TotalSheets : ℕ)

-- Definitions of given conditions
def numBundles := 3
def numBunches := 2
def numHeaps := 5
def sheetsPerBunch := 4
def sheetsPerHeap := 20
def totalSheets := 114

-- Theorem to prove
theorem sheets_in_a_bundle :
  3 * B + 2 * sheetsPerBunch + 5 * sheetsPerHeap = totalSheets → B = 2 := by
  intro h
  sorry

end NUMINAMATH_GPT_sheets_in_a_bundle_l1068_106894


namespace NUMINAMATH_GPT_find_multiplicand_l1068_106848

theorem find_multiplicand (m : ℕ) 
( h : 32519 * m = 325027405 ) : 
m = 9995 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_multiplicand_l1068_106848


namespace NUMINAMATH_GPT_max_daily_sales_revenue_l1068_106898

noncomputable def P (t : ℕ) : ℕ :=
  if 1 ≤ t ∧ t ≤ 24 then t + 2 else if 25 ≤ t ∧ t ≤ 30 then 100 - t else 0

noncomputable def Q (t : ℕ) : ℕ :=
  if 1 ≤ t ∧ t ≤ 30 then 40 - t else 0

noncomputable def y (t : ℕ) : ℕ :=
  P t * Q t

theorem max_daily_sales_revenue :
  ∃ t : ℕ, 1 ≤ t ∧ t ≤ 30 ∧ y t = 115 :=
sorry

end NUMINAMATH_GPT_max_daily_sales_revenue_l1068_106898


namespace NUMINAMATH_GPT_equal_cost_sharing_l1068_106878

variable (X Y Z : ℝ)
variable (h : X < Y ∧ Y < Z)

theorem equal_cost_sharing :
  ∃ (amount : ℝ), amount = (Y + Z - 2 * X) / 3 := 
sorry

end NUMINAMATH_GPT_equal_cost_sharing_l1068_106878


namespace NUMINAMATH_GPT_real_m_of_complex_product_l1068_106812

-- Define the conditions that m is a real number and (m^2 + i)(1 - mi) is a real number
def is_real (z : ℂ) : Prop := z.im = 0
def cplx_eq (m : ℝ) : ℂ := (⟨m^2, 1⟩ : ℂ) * (⟨1, -m⟩ : ℂ)

theorem real_m_of_complex_product (m : ℝ) : is_real (cplx_eq m) ↔ m = 1 :=
by
  sorry

end NUMINAMATH_GPT_real_m_of_complex_product_l1068_106812
