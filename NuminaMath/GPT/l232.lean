import Mathlib

namespace roots_sum_and_product_l232_232299

theorem roots_sum_and_product (k p : ℝ) (hk : (k / 3) = 9) (hp : (p / 3) = 10) : k + p = 57 := by
  sorry

end roots_sum_and_product_l232_232299


namespace solve_equation_l232_232567

theorem solve_equation (x a b : ℝ) (h : x^2 - 6*x + 11 = 27) (sol_a : a = 8) (sol_b : b = -2) :
  3 * a - 2 * b = 28 :=
by
  sorry

end solve_equation_l232_232567


namespace ratio_of_men_to_women_l232_232108

theorem ratio_of_men_to_women (C W M : ℕ) 
  (hC : C = 30) 
  (hW : W = 3 * C) 
  (hTotal : M + W + C = 300) : 
  M / W = 2 :=
by
  sorry

end ratio_of_men_to_women_l232_232108


namespace A_finishes_work_in_8_days_l232_232471

theorem A_finishes_work_in_8_days 
  (A_work B_work W : ℝ) 
  (h1 : 4 * A_work + 6 * B_work = W)
  (h2 : (A_work + B_work) * 4.8 = W) :
  A_work = W / 8 :=
by
  -- We should provide the proof here, but we will use "sorry" for now.
  sorry

end A_finishes_work_in_8_days_l232_232471


namespace goods_train_crossing_time_l232_232173

def speed_kmh : ℕ := 72
def train_length_m : ℕ := 230
def platform_length_m : ℕ := 290

noncomputable def crossing_time_seconds (speed_kmh train_length_m platform_length_m : ℕ) : ℕ :=
  let distance_m := train_length_m + platform_length_m
  let speed_ms := speed_kmh * 1000 / 3600
  distance_m / speed_ms

theorem goods_train_crossing_time :
  crossing_time_seconds speed_kmh train_length_m platform_length_m = 26 :=
by
  -- The proof should be filled in here
  sorry

end goods_train_crossing_time_l232_232173


namespace triangle_transform_same_l232_232805

def Point := ℝ × ℝ

def reflect_x (p : Point) : Point :=
(p.1, -p.2)

def rotate_180 (p : Point) : Point :=
(-p.1, -p.2)

def reflect_y (p : Point) : Point :=
(-p.1, p.2)

def transform (p : Point) : Point :=
reflect_y (rotate_180 (reflect_x p))

theorem triangle_transform_same (A B C : Point) :
A = (2, 1) → B = (4, 1) → C = (2, 3) →
(transform A = (2, 1) ∧ transform B = (4, 1) ∧ transform C = (2, 3)) :=
by
  intros
  sorry

end triangle_transform_same_l232_232805


namespace find_M_l232_232193

theorem find_M (a b c M : ℝ) (h1 : a + b + c = 120) (h2 : a - 9 = M) (h3 : b + 9 = M) (h4 : 9 * c = M) : 
  M = 1080 / 19 :=
by sorry

end find_M_l232_232193


namespace negation_of_universal_proposition_l232_232270

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x^2 + 2*x + 2 > 0) ↔ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0 :=
by
  sorry

end negation_of_universal_proposition_l232_232270


namespace nickys_running_pace_l232_232596

theorem nickys_running_pace (head_start : ℕ) (pace_cristina : ℕ) (time_nicky : ℕ) (distance_meet : ℕ) :
  head_start = 12 →
  pace_cristina = 5 →
  time_nicky = 30 →
  distance_meet = (pace_cristina * (time_nicky - head_start)) →
  (distance_meet / time_nicky = 3) :=
by
  intros h_start h_pace_c h_time_n d_meet
  sorry

end nickys_running_pace_l232_232596


namespace brenda_sally_track_length_l232_232399

theorem brenda_sally_track_length
  (c d : ℝ) 
  (h1 : c / 4 * 3 = d) 
  (h2 : d - 120 = 0.75 * c - 120) 
  (h3 : 0.75 * c + 60 <= 1.25 * c - 180) 
  (h4 : (c - 120 + 0.25 * c - 60) = 1.25 * c - 180):
  c = 766.67 :=
sorry

end brenda_sally_track_length_l232_232399


namespace simplify_999_times_neg13_simplify_complex_expr_correct_division_calculation_l232_232517

-- Part 1: Proving the simplified form of arithmetic operations
theorem simplify_999_times_neg13 : 999 * (-13) = -12987 := by
  sorry

theorem simplify_complex_expr :
  999 * (118 + 4 / 5) + 333 * (-3 / 5) - 999 * (18 + 3 / 5) = 99900 := by
  sorry

-- Part 2: Proving the correct calculation of division
theorem correct_division_calculation : 6 / (-1 / 2 + 1 / 3) = -36 := by
  sorry

end simplify_999_times_neg13_simplify_complex_expr_correct_division_calculation_l232_232517


namespace min_ab_value_l232_232273

theorem min_ab_value (a b : Real) (h_a : 1 < a) (h_b : 1 < b)
  (h_geom_seq : ∀ (x₁ x₂ x₃ : Real), x₁ = (1/4) * Real.log a → x₂ = 1/4 → x₃ = Real.log b →  x₂^2 = x₁ * x₃) : 
  a * b ≥ Real.exp 1 := by
  sorry

end min_ab_value_l232_232273


namespace geometric_progression_common_ratio_l232_232484

/--
If \( a_1, a_2, a_3 \) are terms of an arithmetic progression with common difference \( d \neq 0 \),
and the products \( a_1 a_2, a_2 a_3, a_3 a_1 \) form a geometric progression,
then the common ratio of this geometric progression is \(-2\).
-/
theorem geometric_progression_common_ratio (a₁ a₂ a₃ d : ℝ) (h₀ : d ≠ 0) (h₁ : a₂ = a₁ + d)
  (h₂ : a₃ = a₁ + 2 * d) (h₃ : (a₂ * a₃) / (a₁ * a₂) = (a₃ * a₁) / (a₂ * a₃)) :
  (a₂ * a₃) / (a₁ * a₂) = -2 :=
by
  sorry

end geometric_progression_common_ratio_l232_232484


namespace find_a_l232_232934

theorem find_a (f : ℝ → ℝ)
  (h : ∀ x : ℝ, x < 2 → a - 3 * x > 0) :
  a = 6 :=
by sorry

end find_a_l232_232934


namespace lightest_height_is_135_l232_232137

-- Definitions based on the problem conditions
def heights_in_ratio (a b c d : ℕ) : Prop :=
  ∃ x : ℕ, a = 3 * x ∧ b = 4 * x ∧ c = 5 * x ∧ d = 6 * x

def height_condition (a c d : ℕ) : Prop :=
  d + a = c + 180

-- Lean statement describing the proof problem
theorem lightest_height_is_135 :
  ∀ (a b c d : ℕ),
  heights_in_ratio a b c d →
  height_condition a c d →
  a = 135 :=
by
  intro a b c d
  intro h_in_ratio h_condition
  sorry

end lightest_height_is_135_l232_232137


namespace distribute_books_l232_232043

theorem distribute_books : 
  let total_ways := 4^5
  let subtract_one_student_none := 4 * 3^5
  let add_two_students_none := 6 * 2^5
  total_ways - subtract_one_student_none + add_two_students_none = 240 :=
by
  -- Definitions based on conditions in a)
  let total_ways := 4^5
  let subtract_one_student_none := 4 * 3^5
  let add_two_students_none := 6 * 2^5

  -- The final calculation
  have h : total_ways - subtract_one_student_none + add_two_students_none = 240 := by sorry
  exact h

end distribute_books_l232_232043


namespace range_of_a_in_triangle_l232_232986

open Real

noncomputable def law_of_sines_triangle (A B C : ℝ) (a b c : ℝ) :=
  sin A / a = sin B / b ∧ sin B / b = sin C / c

theorem range_of_a_in_triangle (b : ℝ) (B : ℝ) (a : ℝ) (h1 : b = 2) (h2 : B = pi / 4) (h3 : true) :
  2 < a ∧ a < 2 * sqrt 2 :=
by
  sorry

end range_of_a_in_triangle_l232_232986


namespace probability_of_D_l232_232229

theorem probability_of_D (P_A P_B P_C P_D : ℚ) (hA : P_A = 1/4) (hB : P_B = 1/3) (hC : P_C = 1/6) 
  (hSum : P_A + P_B + P_C + P_D = 1) : P_D = 1/4 := 
by
  sorry

end probability_of_D_l232_232229


namespace opposite_face_of_x_l232_232232

theorem opposite_face_of_x 
    (A D F B E x : Prop) 
    (h1 : x → (A ∧ D ∧ F))
    (h2 : x → B)
    (h3 : E → D ∧ ¬x) : B := 
sorry

end opposite_face_of_x_l232_232232


namespace positional_relationship_of_circles_l232_232773

theorem positional_relationship_of_circles 
  (m n : ℝ)
  (h1 : ∃ (x y : ℝ), x^2 - 10 * x + n = 0 ∧ y^2 - 10 * y + n = 0 ∧ x = 2 ∧ y = m) :
  n = 2 * m ∧ m = 8 → 16 > 2 + 8 :=
by
  sorry

end positional_relationship_of_circles_l232_232773


namespace quadratic_intersect_x_axis_l232_232140

theorem quadratic_intersect_x_axis (a : ℝ) : (∃ x : ℝ, a * x^2 + 4 * x + 1 = 0) ↔ (a ≤ 4 ∧ a ≠ 0) :=
by
  sorry

end quadratic_intersect_x_axis_l232_232140


namespace percent_other_birds_is_31_l232_232832

noncomputable def initial_hawk_percentage : ℝ := 0.30
noncomputable def initial_paddyfield_warbler_percentage : ℝ := 0.25
noncomputable def initial_kingfisher_percentage : ℝ := 0.10
noncomputable def initial_hp_k_total : ℝ := initial_hawk_percentage + initial_paddyfield_warbler_percentage + initial_kingfisher_percentage

noncomputable def migrated_hawk_percentage : ℝ := 0.8 * initial_hawk_percentage
noncomputable def migrated_kingfisher_percentage : ℝ := 2 * initial_kingfisher_percentage
noncomputable def migrated_hp_k_total : ℝ := migrated_hawk_percentage + initial_paddyfield_warbler_percentage + migrated_kingfisher_percentage

noncomputable def other_birds_percentage : ℝ := 1 - migrated_hp_k_total

theorem percent_other_birds_is_31 : other_birds_percentage = 0.31 := sorry

end percent_other_birds_is_31_l232_232832


namespace shaded_area_eq_l232_232066

theorem shaded_area_eq : 
  let side := 8 
  let radius := 3 
  let square_area := side * side
  let sector_area := (1 / 4) * Real.pi * (radius * radius)
  let four_sectors_area := 4 * sector_area
  let triangle_area := (1 / 2) * radius * radius
  let four_triangles_area := 4 * triangle_area
  let shaded_area := square_area - four_sectors_area - four_triangles_area
  shaded_area = 64 - 9 * Real.pi - 18 :=
by
  sorry

end shaded_area_eq_l232_232066


namespace condition_p_neither_sufficient_nor_necessary_l232_232015

theorem condition_p_neither_sufficient_nor_necessary
  (x : ℝ) :
  (1/x ≤ 1 → x^2 - 2 * x ≥ 0) = false ∧ 
  (x^2 - 2 * x ≥ 0 → 1/x ≤ 1) = false := 
by 
  sorry

end condition_p_neither_sufficient_nor_necessary_l232_232015


namespace find_interest_rate_l232_232330

theorem find_interest_rate (P r : ℝ) 
  (h1 : 460 = P * (1 + 3 * r)) 
  (h2 : 560 = P * (1 + 8 * r)) : 
  r = 0.05 :=
by
  sorry

end find_interest_rate_l232_232330


namespace range_of_m_l232_232092

/-- The point (m^2, m) is within the planar region defined by x - 3y + 2 > 0. 
    Find the range of m. -/
theorem range_of_m {m : ℝ} : (m^2 - 3 * m + 2 > 0) ↔ (m < 1 ∨ m > 2) := 
by 
  sorry

end range_of_m_l232_232092


namespace cost_of_apples_l232_232621

theorem cost_of_apples 
  (total_cost : ℕ)
  (cost_bananas : ℕ)
  (cost_bread : ℕ)
  (cost_milk : ℕ)
  (cost_apples : ℕ)
  (h1 : total_cost = 42)
  (h2 : cost_bananas = 12)
  (h3 : cost_bread = 9)
  (h4 : cost_milk = 7)
  (h5 : total_cost = cost_bananas + cost_bread + cost_milk + cost_apples) :
  cost_apples = 14 :=
by
  sorry

end cost_of_apples_l232_232621


namespace cost_of_drill_bits_l232_232652

theorem cost_of_drill_bits (x : ℝ) (h1 : 5 * x + 0.10 * (5 * x) = 33) : x = 6 :=
sorry

end cost_of_drill_bits_l232_232652


namespace find_x_l232_232687

theorem find_x :
    ∃ x : ℚ, (1/7 + 7/x = 15/x + 1/15) ∧ x = 105 := by
  sorry

end find_x_l232_232687


namespace balls_of_yarn_per_sweater_l232_232869

-- Define the conditions as constants
def cost_per_ball := 6
def sell_price_per_sweater := 35
def total_gain := 308
def number_of_sweaters := 28

-- Define a function that models the total gain given the number of balls of yarn per sweater.
def total_gain_formula (x : ℕ) : ℕ :=
  number_of_sweaters * (sell_price_per_sweater - cost_per_ball * x)

-- State the theorem which proves the number of balls of yarn per sweater
theorem balls_of_yarn_per_sweater (x : ℕ) (h : total_gain_formula x = total_gain): x = 4 :=
sorry

end balls_of_yarn_per_sweater_l232_232869


namespace expression_eval_l232_232011

theorem expression_eval (a b c d : ℝ) :
  a * b + c - d = a * (b + c - d) :=
sorry

end expression_eval_l232_232011


namespace third_side_integer_lengths_l232_232376

theorem third_side_integer_lengths (a b : Nat) (h1 : a = 8) (h2 : b = 11) : 
  ∃ n, n = 15 :=
by
  sorry

end third_side_integer_lengths_l232_232376


namespace func_passes_through_1_2_l232_232468

-- Given conditions
variable (a : ℝ) (x : ℝ) (y : ℝ)
variable (h1 : 0 < a) (h2 : a ≠ 1)

-- Definition of the function
noncomputable def func (x : ℝ) : ℝ := a^(x-1) + 1

-- Proof statement
theorem func_passes_through_1_2 : func a 1 = 2 :=
by
  -- proof goes here
  sorry

end func_passes_through_1_2_l232_232468


namespace intercepted_segments_length_l232_232836

theorem intercepted_segments_length {a b c x : ℝ} 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : x = a * b * c / (a * b + b * c + c * a)) : 
  x = a * b * c / (a * b + b * c + c * a) :=
by sorry

end intercepted_segments_length_l232_232836


namespace n_squared_plus_d_not_perfect_square_l232_232022

theorem n_squared_plus_d_not_perfect_square (n d : ℕ) (h1 : n > 0)
  (h2 : d > 0) (h3 : d ∣ 2 * n^2) : ¬ ∃ x : ℕ, n^2 + d = x^2 := 
sorry

end n_squared_plus_d_not_perfect_square_l232_232022


namespace inverse_function_solution_l232_232077

noncomputable def f (a b x : ℝ) := 2 / (a * x + b)

theorem inverse_function_solution (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : f a b 2 = 1 / 2) : b = 1 - 2 * a :=
by
  -- Assuming the inverse function condition means f(2) should be evaluated.
  sorry

end inverse_function_solution_l232_232077


namespace shortest_side_of_triangle_l232_232684

noncomputable def triangle_shortest_side (AB : ℝ) (AD : ℝ) (DB : ℝ) (radius : ℝ) : ℝ :=
  let x := 6
  let y := 5
  2 * y

theorem shortest_side_of_triangle :
  let AB := 16
  let AD := 7
  let DB := 9
  let radius := 5
  AB = AD + DB →
  (AD = 7) ∧ (DB = 9) ∧ (radius = 5) →
  triangle_shortest_side AB AD DB radius = 10 :=
by
  intros h1 h2
  -- proof goes here
  sorry

end shortest_side_of_triangle_l232_232684


namespace arrangment_ways_basil_tomato_l232_232088

theorem arrangment_ways_basil_tomato (basil_plants tomato_plants : Finset ℕ) 
  (hb : basil_plants.card = 5) 
  (ht : tomato_plants.card = 3) 
  : (∃ total_ways : ℕ, total_ways = 4320) :=
by
  sorry

end arrangment_ways_basil_tomato_l232_232088


namespace angle_B_is_pi_over_3_l232_232600

theorem angle_B_is_pi_over_3
  (A B C : ℝ) (a b c : ℝ)
  (h_triangle : a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2)
  (h_sin_ratios : ∃ k > 0, a = 5*k ∧ b = 7*k ∧ c = 8*k) :
  B = π / 3 := 
by
  sorry

end angle_B_is_pi_over_3_l232_232600


namespace probability_one_of_two_sheep_selected_l232_232806

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

end probability_one_of_two_sheep_selected_l232_232806


namespace count_integers_P_leq_0_l232_232485

def P(x : ℤ) : ℤ := 
  (x - 1^3) * (x - 2^3) * (x - 3^3) * (x - 4^3) * (x - 5^3) *
  (x - 6^3) * (x - 7^3) * (x - 8^3) * (x - 9^3) * (x - 10^3) *
  (x - 11^3) * (x - 12^3) * (x - 13^3) * (x - 14^3) * (x - 15^3) *
  (x - 16^3) * (x - 17^3) * (x - 18^3) * (x - 19^3) * (x - 20^3) *
  (x - 21^3) * (x - 22^3) * (x - 23^3) * (x - 24^3) * (x - 25^3) *
  (x - 26^3) * (x - 27^3) * (x - 28^3) * (x - 29^3) * (x - 30^3) *
  (x - 31^3) * (x - 32^3) * (x - 33^3) * (x - 34^3) * (x - 35^3) *
  (x - 36^3) * (x - 37^3) * (x - 38^3) * (x - 39^3) * (x - 40^3) *
  (x - 41^3) * (x - 42^3) * (x - 43^3) * (x - 44^3) * (x - 45^3) *
  (x - 46^3) * (x - 47^3) * (x - 48^3) * (x - 49^3) * (x - 50^3)

theorem count_integers_P_leq_0 : 
  ∃ n : ℕ, n = 15650 ∧ ∀ k : ℤ, (P k ≤ 0) → (n = 15650) :=
by sorry

end count_integers_P_leq_0_l232_232485


namespace min_value_fraction_l232_232074

theorem min_value_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) : 
  (4 / x + 9 / y) ≥ 25 :=
sorry

end min_value_fraction_l232_232074


namespace pounds_of_beef_l232_232754

theorem pounds_of_beef (meals_price : ℝ) (total_sales : ℝ) (meat_per_meal : ℝ) (relationship : ℝ) (total_meat_used : ℝ) (beef_pounds : ℝ) :
  (total_sales = 400) → (meals_price = 20) → (meat_per_meal = 1.5) → (relationship = 0.5) → (20 * meals_price = total_sales) → (total_meat_used = 30) →
  (beef_pounds + beef_pounds * relationship = total_meat_used) → beef_pounds = 20 :=
by
  intros
  sorry

end pounds_of_beef_l232_232754


namespace percent_freshmen_psychology_majors_l232_232240

-- Define the total number of students in our context
def total_students : ℕ := 100

-- Define what 80% of total students being freshmen means
def freshmen (total : ℕ) : ℕ := 8 * total / 10

-- Define what 60% of freshmen being in the school of liberal arts means
def freshmen_in_liberal_arts (total : ℕ) : ℕ := 6 * freshmen total / 10

-- Define what 50% of freshmen in the school of liberal arts being psychology majors means
def freshmen_psychology_majors (total : ℕ) : ℕ := 5 * freshmen_in_liberal_arts total / 10

theorem percent_freshmen_psychology_majors :
  (freshmen_psychology_majors total_students : ℝ) / total_students * 100 = 24 :=
by
  sorry

end percent_freshmen_psychology_majors_l232_232240


namespace solve_trig_eq_l232_232606

open Real -- Open real number structure

theorem solve_trig_eq (x : ℝ) :
  (sin x)^2 + (sin (2 * x))^2 + (sin (3 * x))^2 = 2 ↔ 
  (∃ n : ℤ, x = π / 4 + (π * n) / 2)
  ∨ (∃ n : ℤ, x = π / 2 + π * n)
  ∨ (∃ n : ℤ, x = π / 6 + π * n ∨ x = -π / 6 + π * n) := by sorry

end solve_trig_eq_l232_232606


namespace fraction_division_l232_232383

theorem fraction_division : (3 / 4) / (2 / 5) = 15 / 8 := by
  sorry

end fraction_division_l232_232383


namespace cost_of_flowers_cost_function_minimum_cost_l232_232544

-- Define the costs in terms of yuan
variables (n m : ℕ) -- n is the cost of one lily, m is the cost of one carnation.

-- Define the conditions
axiom cost_condition1 : 2 * n + m = 14
axiom cost_condition2 : 3 * m = 2 * n + 2

-- Prove the cost of one carnation and one lily
theorem cost_of_flowers : n = 5 ∧ m = 4 :=
by {
  sorry
}

-- Variables for the second part
variables (w x : ℕ) -- w is the total cost, x is the number of carnations.

-- Define the conditions
axiom total_condition : 11 = 2 + x + (11 - x)
axiom min_lilies_condition : 11 - x ≥ 2

-- State the relationship between w and x
theorem cost_function : w = 55 - x :=
by {
  sorry
}

-- Prove the minimum cost
theorem minimum_cost : ∃ x, (x ≤ 9 ∧  w = 46) :=
by {
  sorry
}

end cost_of_flowers_cost_function_minimum_cost_l232_232544


namespace pencils_cost_l232_232846

theorem pencils_cost (A B : ℕ) (C D : ℕ) (r : ℚ) : 
    A * 20 = 3200 → B * 20 = 960 → (A / B = 3200 / 960) → (A = 160) → (B = 48) → (C = 3200) → (D = 960) → 160 * 960 / 48 = 3200 :=
by
sorry

end pencils_cost_l232_232846


namespace prime_numbers_r_s_sum_l232_232582

theorem prime_numbers_r_s_sum (p q r s : ℕ) (hp : Fact (Nat.Prime p)) (hq : Fact (Nat.Prime q)) 
  (hr : Fact (Nat.Prime r)) (hs : Fact (Nat.Prime s)) (h1 : p < q) (h2 : q < r) (h3 : r < s) 
  (eqn : p * q * r * s + 1 = 4^(p + q)) : r + s = 274 :=
by
  sorry

end prime_numbers_r_s_sum_l232_232582


namespace peter_and_susan_dollars_l232_232608

theorem peter_and_susan_dollars :
  (2 / 5 : Real) + (1 / 4 : Real) = 0.65 := 
by
  sorry

end peter_and_susan_dollars_l232_232608


namespace condition_necessary_but_not_sufficient_l232_232864

noncomputable def S_n (a_1 d : ℝ) (n : ℕ) : ℝ :=
  n * a_1 + (n * (n - 1) / 2) * d

theorem condition_necessary_but_not_sufficient (a_1 d : ℝ) :
  (∀ n : ℕ, S_n a_1 d (n + 1) > S_n a_1 d n) ↔ (a_1 + d > 0) :=
sorry

end condition_necessary_but_not_sufficient_l232_232864


namespace greater_number_l232_232313

theorem greater_number (x y : ℕ) (h1 : x + y = 22) (h2 : x - y = 4) : x = 13 := 
by sorry

end greater_number_l232_232313


namespace cone_radius_l232_232395

theorem cone_radius
  (l : ℝ) (CSA : ℝ) (π : ℝ) (r : ℝ)
  (h_l : l = 15)
  (h_CSA : CSA = 141.3716694115407)
  (h_pi : π = Real.pi) :
  r = 3 :=
by
  sorry

end cone_radius_l232_232395


namespace g_432_l232_232928

theorem g_432 (g : ℕ → ℤ)
  (h_mul : ∀ x y : ℕ, 0 < x → 0 < y → g (x * y) = g x + g y)
  (h8 : g 8 = 21)
  (h18 : g 18 = 26) :
  g 432 = 47 :=
  sorry

end g_432_l232_232928


namespace circumference_of_flower_bed_l232_232842

noncomputable def square_garden_circumference (a p s r C : ℝ) : Prop :=
  a = s^2 ∧
  p = 4 * s ∧
  a = 2 * p + 14.25 ∧
  r = s / 4 ∧
  C = 2 * Real.pi * r

theorem circumference_of_flower_bed (a p s r : ℝ) (h : square_garden_circumference a p s r (4.75 * Real.pi)) : 
  ∃ C, square_garden_circumference a p s r C ∧ C = 4.75 * Real.pi :=
sorry

end circumference_of_flower_bed_l232_232842


namespace pat_initial_stickers_l232_232406

def initial_stickers (s : ℕ) : ℕ := s  -- Number of stickers Pat had on the first day of the week

def stickers_earned : ℕ := 22  -- Stickers earned during the week

def stickers_end_week (s : ℕ) : ℕ := initial_stickers s + stickers_earned  -- Stickers at the end of the week

theorem pat_initial_stickers (s : ℕ) (h : stickers_end_week s = 61) : s = 39 :=
by
  sorry

end pat_initial_stickers_l232_232406


namespace train_stops_for_10_minutes_per_hour_l232_232828

-- Define the conditions
def speed_excluding_stoppages : ℕ := 48 -- in kmph
def speed_including_stoppages : ℕ := 40 -- in kmph

-- Define the question as proving the train stops for 10 minutes per hour
theorem train_stops_for_10_minutes_per_hour :
  (speed_excluding_stoppages - speed_including_stoppages) * 60 / speed_excluding_stoppages = 10 :=
by
  sorry

end train_stops_for_10_minutes_per_hour_l232_232828


namespace solve_abs_eq_l232_232116

theorem solve_abs_eq (x : ℝ) : 
  (|x - 4| + 3 * x = 12) ↔ (x = 4) :=
by
  sorry

end solve_abs_eq_l232_232116


namespace find_whole_number_N_l232_232588

theorem find_whole_number_N (N : ℕ) (h1 : 6.75 < (N / 4 : ℝ)) (h2 : (N / 4 : ℝ) < 7.25) : N = 28 := 
by 
  sorry

end find_whole_number_N_l232_232588


namespace smallest_constant_l232_232981

theorem smallest_constant (D : ℝ) :
  (∀ (x y : ℝ), x^2 + 2*y^2 + 5 ≥ D*(2*x + 3*y) + 4) → D ≤ Real.sqrt (8 / 17) :=
by
  intros
  sorry

end smallest_constant_l232_232981


namespace term_largest_binomial_coeff_constant_term_in_expansion_l232_232925

theorem term_largest_binomial_coeff {n : ℕ} (h : n = 8) :
  ∃ (k : ℕ) (coeff : ℤ), coeff * x ^ k = 1120 * x^4 :=
by
  sorry

theorem constant_term_in_expansion :
  ∃ (const : ℤ), const = 1280 :=
by
  sorry

end term_largest_binomial_coeff_constant_term_in_expansion_l232_232925


namespace B_pow_48_l232_232449

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 0, 0],
  ![0, 0, 1],
  ![0, -1, 0]
]

theorem B_pow_48 :
  B^48 = ![
    ![0, 0, 0],
    ![0, 1, 0],
    ![0, 0, 1]
  ] := by sorry

end B_pow_48_l232_232449


namespace largest_integer_n_such_that_n_squared_minus_11n_plus_28_is_negative_l232_232038

theorem largest_integer_n_such_that_n_squared_minus_11n_plus_28_is_negative :
  ∃ (n : ℤ), (4 < n) ∧ (n < 7) ∧ (n = 6) :=
by
  sorry

end largest_integer_n_such_that_n_squared_minus_11n_plus_28_is_negative_l232_232038


namespace shopkeeper_loss_percentage_l232_232267

theorem shopkeeper_loss_percentage
  (total_stock_value : ℝ)
  (overall_loss : ℝ)
  (first_part_percentage : ℝ)
  (first_part_profit_percentage : ℝ)
  (remaining_part_loss : ℝ)
  (total_worth_first_part : ℝ)
  (first_part_profit : ℝ)
  (remaining_stock_value : ℝ)
  (remaining_stock_loss : ℝ)
  (loss_percentage : ℝ) :
  total_stock_value = 16000 →
  overall_loss = 400 →
  first_part_percentage = 0.10 →
  first_part_profit_percentage = 0.20 →
  total_worth_first_part = total_stock_value * first_part_percentage →
  first_part_profit = total_worth_first_part * first_part_profit_percentage →
  remaining_stock_value = total_stock_value * (1 - first_part_percentage) →
  remaining_stock_loss = overall_loss + first_part_profit →
  loss_percentage = (remaining_stock_loss / remaining_stock_value) * 100 →
  loss_percentage = 5 :=
by intros; sorry

end shopkeeper_loss_percentage_l232_232267


namespace calculate_price_l232_232044

-- Define variables for prices
def sugar_price_in_terms_of_salt (T : ℝ) : ℝ := 2 * T
def rice_price_in_terms_of_salt (T : ℝ) : ℝ := 3 * T
def apple_price : ℝ := 1.50
def pepper_price : ℝ := 1.25

-- Define pricing conditions
def condition_1 (T : ℝ) : Prop :=
  5 * (sugar_price_in_terms_of_salt T) + 3 * T + 2 * (rice_price_in_terms_of_salt T) + 3 * apple_price + 4 * pepper_price = 35

def condition_2 (T : ℝ) : Prop :=
  4 * (sugar_price_in_terms_of_salt T) + 2 * T + 1 * (rice_price_in_terms_of_salt T) + 2 * apple_price + 3 * pepper_price = 24

-- Define final price calculation with discounts
def total_price (T : ℝ) : ℝ :=
  8 * (sugar_price_in_terms_of_salt T) * 0.9 +
  5 * T +
  (rice_price_in_terms_of_salt T + 3 * (rice_price_in_terms_of_salt T - 0.5)) +
  -- adding two free apples to the count
  5 * apple_price +
  6 * pepper_price

-- Main theorem to prove
theorem calculate_price (T : ℝ) (h1 : condition_1 T) (h2 : condition_2 T) :
  total_price T = 55.64 :=
sorry -- proof omitted

end calculate_price_l232_232044


namespace find_fourth_digit_l232_232926

theorem find_fourth_digit (a b c d : ℕ) (h : 0 ≤ a ∧ a < 8 ∧ 0 ≤ b ∧ b < 8 ∧ 0 ≤ c ∧ c < 8 ∧ 0 ≤ d ∧ d < 8)
  (h_eq : 511 * a + 54 * b - 92 * c - 999 * d = 0) : d = 6 :=
by
  sorry

end find_fourth_digit_l232_232926


namespace intersection_is_correct_l232_232902

def A : Set ℝ := { x | x * (x - 2) < 0 }
def B : Set ℝ := { x | Real.log x > 0 }

theorem intersection_is_correct : A ∩ B = { x | 1 < x ∧ x < 2 } := by
  sorry

end intersection_is_correct_l232_232902


namespace rectangle_area_l232_232653

theorem rectangle_area {AB AC BC : ℕ} (hAB : AB = 15) (hAC : AC = 17)
  (hRightTriangle : AC * AC = AB * AB + BC * BC) : AB * BC = 120 := by
  sorry

end rectangle_area_l232_232653


namespace max_value_of_h_l232_232443

noncomputable def f (x : ℝ) : ℝ := -x + 3
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def h (x : ℝ) : ℝ := min (f x) (g x)

theorem max_value_of_h : ∃ x : ℝ, h x = 1 :=
by
  sorry

end max_value_of_h_l232_232443


namespace math_problem_l232_232117

theorem math_problem (a b : ℕ) (ha : a = 45) (hb : b = 15) :
  (a + b)^2 - 3 * (a^2 + b^2 - 2 * a * b) = 900 :=
by
  sorry

end math_problem_l232_232117


namespace clock_overlap_24_hours_l232_232003

theorem clock_overlap_24_hours (hour_rotations : ℕ) (minute_rotations : ℕ) 
  (h_hour_rotations: hour_rotations = 2) 
  (h_minute_rotations: minute_rotations = 24) : 
  ∃ (overlaps : ℕ), overlaps = 22 := 
by 
  sorry

end clock_overlap_24_hours_l232_232003


namespace revenue_from_full_price_tickets_l232_232324

theorem revenue_from_full_price_tickets (f h p : ℝ) (total_tickets : f + h = 160) (total_revenue : f * p + h * (p / 2) = 2400) :
  f * p = 960 :=
sorry

end revenue_from_full_price_tickets_l232_232324


namespace no_symmetry_line_for_exponential_l232_232030

theorem no_symmetry_line_for_exponential : ¬ ∃ l : ℝ → ℝ, ∀ x : ℝ, (2 ^ x) = l (2 ^ (2 * l x - x)) := 
sorry

end no_symmetry_line_for_exponential_l232_232030


namespace apples_and_pears_weight_l232_232699

theorem apples_and_pears_weight (apples pears : ℕ) 
    (h_apples : apples = 240) 
    (h_pears : pears = 3 * apples) : 
    apples + pears = 960 := 
  by
  sorry

end apples_and_pears_weight_l232_232699


namespace age_difference_between_Mandy_and_sister_l232_232941

variable (Mandy_age Brother_age Sister_age : ℕ)

-- Given conditions
def Mandy_is_3_years_old : Mandy_age = 3 := by sorry
def Brother_is_4_times_older : Brother_age = 4 * Mandy_age := by sorry
def Sister_is_5_years_younger_than_brother : Sister_age = Brother_age - 5 := by sorry

-- Prove the question
theorem age_difference_between_Mandy_and_sister :
  Mandy_age = 3 ∧ Brother_age = 4 * Mandy_age ∧ Sister_age = Brother_age - 5 → Sister_age - Mandy_age = 4 := 
by 
  sorry

end age_difference_between_Mandy_and_sister_l232_232941


namespace solution_set_M_minimum_value_expr_l232_232272

-- Define the function f(x)
def f (x : ℝ) : ℝ := abs (x + 1) - 2 * abs (x - 2)

-- Proof problem (1): Prove that the solution set M of the inequality f(x) ≥ -1 is {x | 2/3 ≤ x ≤ 6}.
theorem solution_set_M : 
  { x : ℝ | f x ≥ -1 } = { x : ℝ | 2/3 ≤ x ∧ x ≤ 6 } :=
sorry

-- Define the requirement for t and the expression to minimize
noncomputable def t : ℝ := 6
noncomputable def expr (a b c : ℝ) : ℝ := 1 / (2 * a + b) + 1 / (2 * a + c)

-- Proof problem (2): Given t = 6 and 4a + b + c = 6, prove that the minimum value of expr is 2/3.
theorem minimum_value_expr (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : 4 * a + b + c = t) :
  expr a b c ≥ 2/3 :=
sorry

end solution_set_M_minimum_value_expr_l232_232272


namespace polyhedron_volume_formula_l232_232821

noncomputable def polyhedron_volume (H S1 S2 S3 : ℝ) : ℝ :=
  (1 / 6) * H * (S1 + S2 + 4 * S3)

theorem polyhedron_volume_formula 
  (H S1 S2 S3 : ℝ)
  (bases_parallel_planes : Prop)
  (lateral_faces_trapezoids_parallelograms_or_triangles : Prop)
  (H_distance : Prop) 
  (S1_area_base : Prop) 
  (S2_area_base : Prop) 
  (S3_area_cross_section : Prop) : 
  polyhedron_volume H S1 S2 S3 = (1 / 6) * H * (S1 + S2 + 4 * S3) :=
sorry

end polyhedron_volume_formula_l232_232821


namespace positive_difference_of_complementary_angles_l232_232311

-- Define the conditions
variables (a b : ℝ)
variable (h1 : a + b = 90)
variable (h2 : 5 * b = a)

-- Define the theorem we are proving
theorem positive_difference_of_complementary_angles (a b : ℝ) (h1 : a + b = 90) (h2 : 5 * b = a) :
  |a - b| = 60 :=
by
  sorry

end positive_difference_of_complementary_angles_l232_232311


namespace data_transmission_time_l232_232020

def packet_size : ℕ := 256
def num_packets : ℕ := 100
def transmission_rate : ℕ := 200
def total_data : ℕ := num_packets * packet_size
def transmission_time_in_seconds : ℚ := total_data / transmission_rate
def transmission_time_in_minutes : ℚ := transmission_time_in_seconds / 60

theorem data_transmission_time :
  transmission_time_in_minutes = 2 :=
  sorry

end data_transmission_time_l232_232020


namespace arithmetic_geometric_sequence_relation_l232_232593

theorem arithmetic_geometric_sequence_relation 
  (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (hA : ∀ n: ℕ, a (n + 1) - a n = a 1) 
  (hG : ∀ n: ℕ, b (n + 1) / b n = b 1) 
  (h1 : a 1 = b 1) 
  (h11 : a 11 = b 11) 
  (h_pos : 0 < a 1 ∧ 0 < a 11 ∧ 0 < b 11 ∧ 0 < b 1) :
  a 6 ≥ b 6 := sorry

end arithmetic_geometric_sequence_relation_l232_232593


namespace big_al_bananas_l232_232249

theorem big_al_bananas (total_bananas : ℕ) (a : ℕ)
  (h : total_bananas = 150)
  (h1 : a + (a + 7) + (a + 14) + (a + 21) + (a + 28) = total_bananas) :
  a + 14 = 30 :=
by
  -- Using the given conditions to prove the statement
  sorry

end big_al_bananas_l232_232249


namespace value_of_f_m_minus_1_pos_l232_232798

variable (a m : ℝ)
variable (f : ℝ → ℝ)
variable (a_pos : a > 0)
variable (fm_neg : f m < 0)
variable (f_def : ∀ x, f x = x^2 - x + a)

theorem value_of_f_m_minus_1_pos : f (m - 1) > 0 :=
by
  sorry

end value_of_f_m_minus_1_pos_l232_232798


namespace positiveDifferenceEquation_l232_232050

noncomputable def positiveDifference (x y : ℝ) : ℝ := |y - x|

theorem positiveDifferenceEquation (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 10) :
  positiveDifference x y = 60 / 7 :=
by
  sorry

end positiveDifferenceEquation_l232_232050


namespace num_black_balls_l232_232440

theorem num_black_balls 
  (R W B : ℕ) 
  (R_eq : R = 30) 
  (prob_white : (W : ℝ) / 100 = 0.47) 
  (total_balls : R + W + B = 100) : B = 23 := 
by 
  sorry

end num_black_balls_l232_232440


namespace evaluate_expression_l232_232211

theorem evaluate_expression : (20^40) / (40^20) = 10^20 := by
  sorry

end evaluate_expression_l232_232211


namespace spending_required_for_free_shipping_l232_232427

def shampoo_cost : ℕ := 10
def conditioner_cost : ℕ := 10
def lotion_cost : ℕ := 6
def shampoo_count : ℕ := 1
def conditioner_count : ℕ := 1
def lotion_count : ℕ := 3
def additional_spending_needed : ℕ := 12
def current_spending : ℕ := (shampoo_cost * shampoo_count) + (conditioner_cost * conditioner_count) + (lotion_cost * lotion_count)

theorem spending_required_for_free_shipping : current_spending + additional_spending_needed = 50 := by
  sorry

end spending_required_for_free_shipping_l232_232427


namespace least_xy_value_l232_232817

theorem least_xy_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / (3 * y) = 1 / 6) : x * y = 90 :=
by
  sorry

end least_xy_value_l232_232817


namespace ball_bounce_l232_232102

theorem ball_bounce :
  ∃ b : ℕ, 324 * (3 / 4) ^ b < 40 ∧ b = 8 :=
by
  have : (3 / 4 : ℝ) < 1 := by norm_num
  have h40_324 : (40 : ℝ) / 324 = 10 / 81 := by norm_num
  sorry

end ball_bounce_l232_232102


namespace initial_ratio_milk_water_l232_232853

theorem initial_ratio_milk_water (M W : ℕ) (h1 : M + W = 165) (h2 : ∀ W', W' = W + 66 → M * 4 = 3 * W') : M / gcd M W = 3 ∧ W / gcd M W = 2 :=
by
  -- Proof here
  sorry

end initial_ratio_milk_water_l232_232853


namespace dealer_gross_profit_l232_232681

variable (purchase_price : ℝ) (markup_rate : ℝ) (gross_profit : ℝ)

def desk_problem (purchase_price : ℝ) (markup_rate : ℝ) (gross_profit : ℝ) : Prop :=
  ∀ (S : ℝ), S = purchase_price + markup_rate * S → gross_profit = S - purchase_price

theorem dealer_gross_profit : desk_problem 150 0.5 150 :=
by 
  sorry

end dealer_gross_profit_l232_232681


namespace smallest_sum_a_b_l232_232085

theorem smallest_sum_a_b :
  ∃ (a b : ℕ), (7 * b - 4 * a = 3) ∧ a > 7 ∧ b > 7 ∧ a + b = 24 :=
by
  sorry

end smallest_sum_a_b_l232_232085


namespace zarnin_staffing_l232_232721

theorem zarnin_staffing (n total unsuitable : ℕ) (unsuitable_factor : ℕ) (job_openings : ℕ)
  (h1 : total = 30) 
  (h2 : unsuitable_factor = 2 / 3) 
  (h3 : unsuitable = unsuitable_factor * total) 
  (h4 : n = total - unsuitable)
  (h5 : job_openings = 5) :
  (n - 0) * (n - 1) * (n - 2) * (n - 3) * (n - 4) = 30240 := by
    sorry

end zarnin_staffing_l232_232721


namespace adam_earnings_per_lawn_l232_232071

theorem adam_earnings_per_lawn (total_lawns : ℕ) (forgot_lawns : ℕ) (total_earnings : ℕ) :
  total_lawns = 12 →
  forgot_lawns = 8 →
  total_earnings = 36 →
  (total_earnings / (total_lawns - forgot_lawns)) = 9 :=
by
  intros h1 h2 h3
  sorry

end adam_earnings_per_lawn_l232_232071


namespace rhombus_area_l232_232068

theorem rhombus_area (s : ℝ) (d1 d2 : ℝ) (h1 : s = Real.sqrt 145) (h2 : abs (d1 - d2) = 10) : 
  (1/2) * d1 * d2 = 100 :=
sorry

end rhombus_area_l232_232068


namespace students_count_l232_232767

theorem students_count :
  ∃ S : ℕ, (S + 4) % 9 = 0 ∧ S = 23 :=
by
  sorry

end students_count_l232_232767


namespace total_cost_is_18_l232_232644

-- Definitions based on the conditions
def cost_soda : ℕ := 1
def cost_3_sodas := 3 * cost_soda
def cost_soup := cost_3_sodas
def cost_2_soups := 2 * cost_soup
def cost_sandwich := 3 * cost_soup
def total_cost := cost_3_sodas + cost_2_soups + cost_sandwich

-- The proof statement
theorem total_cost_is_18 : total_cost = 18 := by
  -- proof will go here
  sorry

end total_cost_is_18_l232_232644


namespace number_of_people_per_van_l232_232655

theorem number_of_people_per_van (num_students : ℕ) (num_adults : ℕ) (num_vans : ℕ) (total_people : ℕ) (people_per_van : ℕ) :
  num_students = 40 →
  num_adults = 14 →
  num_vans = 6 →
  total_people = num_students + num_adults →
  people_per_van = total_people / num_vans →
  people_per_van = 9 :=
by
  intros h_students h_adults h_vans h_total h_div
  sorry

end number_of_people_per_van_l232_232655


namespace domain_of_f_l232_232458

noncomputable def f (x : ℝ) : ℝ := (x^4 - 4*x^3 + 6*x^2 - 4*x + 1) / (x^2 - 9)

theorem domain_of_f :
  ∀ x : ℝ, (x ≠ 3 ∧ x ≠ -3) ↔ ((x < -3) ∨ (-3 < x ∧ x < 3) ∨ (x > 3)) :=
by
  sorry

end domain_of_f_l232_232458


namespace minimum_value_inequality_l232_232182

def minimum_value_inequality_problem : Prop :=
∀ (a b : ℝ), (0 < a) → (0 < b) → (a + 3 * b = 1) → (1 / a + 1 / (3 * b)) = 4

theorem minimum_value_inequality : minimum_value_inequality_problem :=
sorry

end minimum_value_inequality_l232_232182


namespace inequality_equivalence_l232_232219

theorem inequality_equivalence (x : ℝ) : 
  (x + 2) / (x - 1) ≥ 0 ↔ (x + 2) * (x - 1) ≥ 0 :=
sorry

end inequality_equivalence_l232_232219


namespace distance_between_stations_l232_232965

/-- Two trains start at the same time from two stations and proceed towards each other.
    Train 1 travels at 20 km/hr.
    Train 2 travels at 25 km/hr.
    When they meet, Train 2 has traveled 55 km more than Train 1.
    Prove that the distance between the two stations is 495 km. -/
theorem distance_between_stations : ∃ x t : ℕ, 20 * t = x ∧ 25 * t = x + 55 ∧ 2 * x + 55 = 495 :=
by {
  sorry
}

end distance_between_stations_l232_232965


namespace cost_to_fill_half_of_CanB_l232_232689

theorem cost_to_fill_half_of_CanB (r h : ℝ) (C_cost : ℝ) (VC VB : ℝ) 
(h1 : VC = 2 * VB) 
(h2 : VB = Real.pi * r^2 * h) 
(h3 : VC = Real.pi * (2 * r)^2 * (h / 2)) 
(h4 : C_cost = 16):
  C_cost / 4 = 4 :=
by
  sorry

end cost_to_fill_half_of_CanB_l232_232689


namespace larger_number_l232_232425

theorem larger_number (x y : ℝ) (h₁ : x + y = 45) (h₂ : x - y = 7) : x = 26 :=
by
  sorry

end larger_number_l232_232425


namespace transform_eq_l232_232586

theorem transform_eq (x y : ℝ) (h : 5 * x - 6 * y = 4) : 
  y = (5 / 6) * x - (2 / 3) :=
  sorry

end transform_eq_l232_232586


namespace two_rows_arrangement_person_A_not_head_tail_arrangement_girls_together_arrangement_boys_not_adjacent_arrangement_l232_232340

-- Define the number of boys and girls
def boys : ℕ := 2
def girls : ℕ := 3
def total_people : ℕ := boys + girls

-- Define assumptions about arrangements
def arrangements_in_two_rows : ℕ := sorry
def arrangements_with_person_A_not_head_tail : ℕ := sorry
def arrangements_with_girls_together : ℕ := sorry
def arrangements_with_boys_not_adjacent : ℕ := sorry

-- State the mathematical equivalence proof problems
theorem two_rows_arrangement : arrangements_in_two_rows = 60 := 
  sorry

theorem person_A_not_head_tail_arrangement : arrangements_with_person_A_not_head_tail = 72 := 
  sorry

theorem girls_together_arrangement : arrangements_with_girls_together = 36 := 
  sorry

theorem boys_not_adjacent_arrangement : arrangements_with_boys_not_adjacent = 72 := 
  sorry

end two_rows_arrangement_person_A_not_head_tail_arrangement_girls_together_arrangement_boys_not_adjacent_arrangement_l232_232340


namespace second_number_value_l232_232222

theorem second_number_value
  (a b : ℝ)
  (h1 : a * (a - 6) = 7)
  (h2 : b * (b - 6) = 7)
  (h3 : a ≠ b)
  (h4 : a + b = 6) :
  b = 7 := by
sorry

end second_number_value_l232_232222


namespace graph_shift_proof_l232_232213

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)
noncomputable def h (x : ℝ) : ℝ := g (x + Real.pi / 8)

theorem graph_shift_proof : ∀ x, h x = f x := by
  sorry

end graph_shift_proof_l232_232213


namespace upstream_speed_l232_232891

-- Speed of the man in still water
def V_m : ℕ := 32

-- Speed of the man rowing downstream
def V_down : ℕ := 42

-- Speed of the stream
def V_s : ℕ := V_down - V_m

-- Speed of the man rowing upstream
def V_up : ℕ := V_m - V_s

theorem upstream_speed (V_m : ℕ) (V_down : ℕ) (V_s : ℕ) (V_up : ℕ) : 
  V_m = 32 → 
  V_down = 42 → 
  V_s = V_down - V_m → 
  V_up = V_m - V_s → 
  V_up = 22 := 
by intros; 
   repeat {sorry}

end upstream_speed_l232_232891


namespace negation_of_proposition_l232_232873

theorem negation_of_proposition (x y : ℝ): (x + y > 0 → x > 0 ∧ y > 0) ↔ ¬ ((x + y ≤ 0) → (x ≤ 0 ∨ y ≤ 0)) :=
by sorry

end negation_of_proposition_l232_232873


namespace fraction_zero_implies_a_eq_neg2_l232_232287

theorem fraction_zero_implies_a_eq_neg2 (a : ℝ) (h : (a^2 - 4) / (a - 2) = 0) (h2 : a ≠ 2) : a = -2 :=
sorry

end fraction_zero_implies_a_eq_neg2_l232_232287


namespace solve_nat_eqn_l232_232964

theorem solve_nat_eqn (n k l m : ℕ) (hl : l > 1) 
  (h_eq : (1 + n^k)^l = 1 + n^m) : (n, k, l, m) = (2, 1, 2, 3) := 
sorry

end solve_nat_eqn_l232_232964


namespace sum_of_ages_l232_232710

-- Definition of the ages based on the intervals and the youngest child's age.
def youngest_age : ℕ := 6
def second_youngest_age : ℕ := youngest_age + 2
def middle_age : ℕ := youngest_age + 4
def second_oldest_age : ℕ := youngest_age + 6
def oldest_age : ℕ := youngest_age + 8

-- The theorem stating the total sum of the ages of the children, given the conditions.
theorem sum_of_ages :
  youngest_age + second_youngest_age + middle_age + second_oldest_age + oldest_age = 50 :=
by sorry

end sum_of_ages_l232_232710


namespace problem1_statement_problem2_statement_l232_232677

-- Defining the sets A and B
def set_A (x : ℝ) := 2*x^2 - 7*x + 3 ≤ 0
def set_B (x a : ℝ) := x + a < 0

-- Problem 1: Intersection of A and B when a = -2
def question1 (x : ℝ) : Prop := set_A x ∧ set_B x (-2)

-- Problem 2: Range of a for A ∩ B = A
def question2 (a : ℝ) : Prop := ∀ x, set_A x → set_B x a

theorem problem1_statement :
  ∀ x, question1 x ↔ x >= 1/2 ∧ x < 2 :=
by sorry

theorem problem2_statement :
  ∀ a, (∀ x, set_A x → set_B x a) ↔ a < -3 :=
by sorry

end problem1_statement_problem2_statement_l232_232677


namespace triangle_two_solutions_range_of_a_l232_232235

noncomputable def range_of_a (a b : ℝ) (A : ℝ) : Prop :=
b * Real.sin A < a ∧ a < b

theorem triangle_two_solutions_range_of_a (a : ℝ) (A : ℝ := Real.pi / 6) (b : ℝ := 2) :
  range_of_a a b A ↔ 1 < a ∧ a < 2 := by
sorry

end triangle_two_solutions_range_of_a_l232_232235


namespace cone_water_fill_percentage_l232_232764

noncomputable def volumeFilledPercentage (h r : ℝ) : ℝ :=
  let original_cone_volume := (1 / 3) * Real.pi * r^2 * h
  let water_cone_volume := (1 / 3) * Real.pi * ((2 / 3) * r)^2 * ((2 / 3) * h)
  let ratio := water_cone_volume / original_cone_volume
  ratio * 100


theorem cone_water_fill_percentage (h r : ℝ) :
  volumeFilledPercentage h r = 29.6296 :=
by
  sorry

end cone_water_fill_percentage_l232_232764


namespace sum_of_squares_of_coeffs_l232_232701

   theorem sum_of_squares_of_coeffs :
     let expr := 3 * (X^3 - 4 * X^2 + X) - 5 * (X^3 + 2 * X^2 - 5 * X + 3)
     let simplified_expr := -2 * X^3 - 22 * X^2 + 28 * X - 15
     let coefficients := [-2, -22, 28, -15]
     (coefficients.map (λ a => a^2)).sum = 1497 := 
   by 
     -- expending, simplifying and summing up the coefficients 
     sorry
   
end sum_of_squares_of_coeffs_l232_232701


namespace percent_equality_l232_232932

theorem percent_equality :
  (1 / 4 : ℝ) * 100 = (10 / 100 : ℝ) * 250 :=
by
  sorry

end percent_equality_l232_232932


namespace max_sum_of_circle_eq_eight_l232_232906

noncomputable def max_sum_of_integer_solutions (r : ℕ) : ℕ :=
  if r = 6 then 8 else 0

theorem max_sum_of_circle_eq_eight 
  (h1 : ∃ (x y : ℤ), (x - 1)^2 + (y - 1)^2 = 36 ∧ (r : ℕ) = 6) :
  max_sum_of_integer_solutions r = 8 := 
by
  sorry

end max_sum_of_circle_eq_eight_l232_232906


namespace square_area_l232_232572

theorem square_area (s : ℝ) (h : s = 12) : s * s = 144 :=
by
  rw [h]
  norm_num

end square_area_l232_232572


namespace red_ball_probability_correct_l232_232266

theorem red_ball_probability_correct (R B : ℕ) (hR : R = 3) (hB : B = 3) :
  (R / (R + B) : ℚ) = 1 / 2 := by
  sorry

end red_ball_probability_correct_l232_232266


namespace fgh_supermarkets_in_us_more_than_canada_l232_232094

theorem fgh_supermarkets_in_us_more_than_canada
  (total_supermarkets : ℕ)
  (us_supermarkets : ℕ)
  (canada_supermarkets : ℕ)
  (h1 : total_supermarkets = 70)
  (h2 : us_supermarkets = 42)
  (h3 : us_supermarkets + canada_supermarkets = total_supermarkets):
  us_supermarkets - canada_supermarkets = 14 :=
by
  sorry

end fgh_supermarkets_in_us_more_than_canada_l232_232094


namespace half_angle_quadrant_second_quadrant_l232_232142

theorem half_angle_quadrant_second_quadrant
  (θ : Real)
  (h1 : π < θ ∧ θ < 3 * π / 2) -- θ is in the third quadrant
  (h2 : Real.cos (θ / 2) < 0) : -- cos (θ / 2) < 0
  π / 2 < θ / 2 ∧ θ / 2 < π := -- θ / 2 is in the second quadrant
sorry

end half_angle_quadrant_second_quadrant_l232_232142


namespace max_val_z_lt_2_l232_232850

-- Definitions for the variables and constraints
variable {x y m : ℝ}
variable (h1 : y ≥ x) (h2 : y ≤ m * x) (h3 : x + y ≤ 1) (h4 : m > 1)

-- Theorem statement
theorem max_val_z_lt_2 (h1 : y ≥ x) (h2 : y ≤ m * x) (h3 : x + y ≤ 1) (h4 : m > 1) : 
  (∀ x y, y ≥ x → y ≤ m * x → x + y ≤ 1 → x + m * y < 2) ↔ 1 < m ∧ m < 1 + Real.sqrt 2 :=
sorry

end max_val_z_lt_2_l232_232850


namespace aaron_guesses_correctly_l232_232890

noncomputable def P_H : ℝ := 2 / 3
noncomputable def P_T : ℝ := 1 / 3
noncomputable def P_G_H : ℝ := 2 / 3
noncomputable def P_G_T : ℝ := 1 / 3

noncomputable def p : ℝ := P_H * P_G_H + P_T * P_G_T

theorem aaron_guesses_correctly :
  9000 * p = 5000 :=
by
  sorry

end aaron_guesses_correctly_l232_232890


namespace tennis_tournament_matches_l232_232459

theorem tennis_tournament_matches (num_players : ℕ) (total_days : ℕ) (rest_days : ℕ)
  (num_matches_per_day : ℕ) (matches_per_player : ℕ)
  (h1 : num_players = 10)
  (h2 : total_days = 9)
  (h3 : rest_days = 1)
  (h4 : num_matches_per_day = 5)
  (h5 : matches_per_player = 1)
  : (num_players * (num_players - 1) / 2) - (num_matches_per_day * (total_days - rest_days)) = 40 :=
by
  sorry

end tennis_tournament_matches_l232_232459


namespace trajectory_no_intersection_distance_AB_l232_232319

variable (M : Type) [MetricSpace M]

-- Point M on the plane
variable (M : ℝ × ℝ)

-- Given conditions
def condition1 (M : ℝ × ℝ) : Prop := 
  (Real.sqrt ((M.1 - 8)^2 + M.2^2) = 2 * Real.sqrt ((M.1 - 2)^2 + M.2^2))

-- 1. Proving the trajectory C of M
theorem trajectory (M : ℝ × ℝ) (h : condition1 M) : M.1^2 + M.2^2 = 16 :=
by
  sorry

-- 2. Range of values for k such that y = kx - 5 does not intersect trajectory C
theorem no_intersection (k : ℝ) : 
  (∀ (x y : ℝ), x^2 + y^2 = 16 → y ≠ k * x - 5) ↔ (-3 / 4 < k ∧ k < 3 / 4) :=
by
  sorry

-- 3. Distance between intersection points A and B of given circles
def intersection_condition (x y : ℝ) : Prop :=
  (x^2 + y^2 = 16) ∧ (x^2 + y^2 - 8 * x - 8 * y + 16 = 0)

theorem distance_AB (A B : ℝ × ℝ) (hA : intersection_condition A.1 A.2) (hB : intersection_condition B.1 B.2) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 2 :=
by
  sorry

end trajectory_no_intersection_distance_AB_l232_232319


namespace total_cost_with_discount_and_tax_l232_232892

theorem total_cost_with_discount_and_tax
  (sandwich_cost : ℝ := 2.44)
  (soda_cost : ℝ := 0.87)
  (num_sandwiches : ℕ := 2)
  (num_sodas : ℕ := 4)
  (discount : ℝ := 0.15)
  (tax_rate : ℝ := 0.09) : 
  (num_sandwiches * sandwich_cost * (1 - discount) + num_sodas * soda_cost) * (1 + tax_rate) = 8.32 :=
by
  sorry

end total_cost_with_discount_and_tax_l232_232892


namespace PTA_money_left_l232_232204

theorem PTA_money_left (initial_savings : ℝ) (spent_on_supplies : ℝ) (spent_on_food : ℝ) :
  initial_savings = 400 →
  spent_on_supplies = initial_savings / 4 →
  spent_on_food = (initial_savings - spent_on_supplies) / 2 →
  (initial_savings - spent_on_supplies - spent_on_food) = 150 :=
by
  intro initial_savings_eq
  intro spent_on_supplies_eq
  intro spent_on_food_eq
  sorry

end PTA_money_left_l232_232204


namespace car_speed_first_hour_l232_232557

theorem car_speed_first_hour
  (x : ℕ)
  (speed_second_hour : ℕ := 80)
  (average_speed : ℕ := 90)
  (total_time : ℕ := 2)
  (h : average_speed = (x + speed_second_hour) / total_time) :
  x = 100 :=
by
  sorry

end car_speed_first_hour_l232_232557


namespace rationalize_denominator_l232_232461

theorem rationalize_denominator :
  let A := -12
  let B := 7
  let C := 9
  let D := 13
  let E := 5
  (4 * Real.sqrt 7 + 3 * Real.sqrt 13) ≠ 0 →
  B < D →
  ∀ (x : ℝ), x = (3 : ℝ) / (4 * Real.sqrt 7 + 3 * Real.sqrt 13) →
    A + B + C + D + E = 22 := 
by
  intros
  -- Provide the actual theorem statement here
  sorry

end rationalize_denominator_l232_232461


namespace swimming_speed_in_still_water_l232_232500

/-- The speed (in km/h) of a man swimming in still water given the speed of the water current
    and the time taken to swim a certain distance against the current. -/
theorem swimming_speed_in_still_water (v : ℝ) (speed_water : ℝ) (time : ℝ) (distance : ℝ) 
  (h1 : speed_water = 12) (h2 : time = 5) (h3 : distance = 40)
  (h4 : time = distance / (v - speed_water)) : v = 20 :=
by
  sorry

end swimming_speed_in_still_water_l232_232500


namespace bobby_gasoline_left_l232_232289

theorem bobby_gasoline_left
  (initial_gasoline : ℕ) (supermarket_distance : ℕ) 
  (travel_distance : ℕ) (turn_back_distance : ℕ)
  (trip_fuel_efficiency : ℕ) : 
  initial_gasoline = 12 →
  supermarket_distance = 5 →
  travel_distance = 6 →
  turn_back_distance = 2 →
  trip_fuel_efficiency = 2 →
  ∃ remaining_gasoline,
    remaining_gasoline = initial_gasoline - 
    ((supermarket_distance * 2 + 
    turn_back_distance * 2 + 
    travel_distance) / trip_fuel_efficiency) ∧ 
    remaining_gasoline = 2 :=
by sorry

end bobby_gasoline_left_l232_232289


namespace triple_divisor_sum_6_l232_232533

-- Summarize the definition of the divisor sum function excluding the number itself
def divisorSumExcluding (n : ℕ) : ℕ :=
  (Finset.filter (λ x => x ≠ n) (Finset.range (n + 1))).sum id

-- This is the main statement that we need to prove
theorem triple_divisor_sum_6 : divisorSumExcluding (divisorSumExcluding (divisorSumExcluding 6)) = 6 := 
by sorry

end triple_divisor_sum_6_l232_232533


namespace commute_time_abs_diff_l232_232255

theorem commute_time_abs_diff (x y : ℝ) 
  (h1 : (x + y + 10 + 11 + 9)/5 = 10) 
  (h2 : (1/5 : ℝ) * ((x - 10)^2 + (y - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (9 - 10)^2) = 2) : 
  |x - y| = 4 :=
by
  sorry

end commute_time_abs_diff_l232_232255


namespace problem1_problem2_l232_232310

variable {A B C a b c : ℝ}

-- Problem (1)
theorem problem1 (h : b * (1 - 2 * Real.cos A) = 2 * a * Real.cos B) : b = 2 * c := 
sorry

-- Problem (2)
theorem problem2 (a_eq : a = 1) (tanA_eq : Real.tan A = 2 * Real.sqrt 2) (b_eq_c : b = 2 * c): 
  Real.sqrt (c^2 * (1 - (Real.cos (A + B)))) = 2 * Real.sqrt 2 * b :=
sorry

end problem1_problem2_l232_232310


namespace range_of_a_l232_232841

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
    (if x1 ≤ 1 then (-x1^2 + a*x1)
     else (a*x1 - 1)) = 
    (if x2 ≤ 1 then (-x2^2 + a*x2)
     else (a*x2 - 1))) → a < 2 :=
sorry

end range_of_a_l232_232841


namespace smallest_three_digit_n_l232_232815

theorem smallest_three_digit_n (n : ℕ) (h_pos : 100 ≤ n) (h_below : n ≤ 999) 
  (cond1 : n % 9 = 2) (cond2 : n % 6 = 4) : n = 118 :=
by {
  sorry
}

end smallest_three_digit_n_l232_232815


namespace prime_squared_mod_six_l232_232087

theorem prime_squared_mod_six (p : ℕ) (hp1 : p > 5) (hp2 : Nat.Prime p) : (p ^ 2) % 6 = 1 :=
sorry

end prime_squared_mod_six_l232_232087


namespace friends_for_picnic_only_l232_232107

theorem friends_for_picnic_only (M MP MG G PG A P : ℕ) 
(h1 : M + MP + MG + A = 10)
(h2 : G + MG + A = 5)
(h3 : MP = 4)
(h4 : MG = 2)
(h5 : PG = 0)
(h6 : A = 2)
(h7 : M + P + G + MP + MG + PG + A = 31) : 
    P = 20 := by {
  sorry
}

end friends_for_picnic_only_l232_232107


namespace max_marks_test_l232_232571

theorem max_marks_test (M : ℝ) : 
  (0.30 * M = 80 + 100) -> 
  M = 600 :=
by 
  sorry

end max_marks_test_l232_232571


namespace problem_l232_232206

theorem problem :
  ∀ (x y a b : ℝ), 
  |x + y| + |x - y| = 2 → 
  a > 0 → 
  b > 0 → 
  ∀ z : ℝ, 
  z = 4 * a * x + b * y → 
  (∀ (x y : ℝ), |x + y| + |x - y| = 2 → 4 * a * x + b * y ≤ 1) →
  (1 = 4 * a * 1 + b * 1) →
  (1 = 4 * a * (-1) + b * 1) →
  (1 = 4 * a * (-1) + b * (-1)) →
  (1 = 4 * a * 1 + b * (-1)) →
  ∀ a b : ℝ, a > 0 → b > 0 → (1 = 4 * a + b) →
  (a = 1 / 6 ∧ b = 1 / 3) → 
  (1 / a + 1 / b = 9) :=
by
  sorry

end problem_l232_232206


namespace cyclic_cosine_inequality_l232_232977

theorem cyclic_cosine_inequality
  (α β γ : ℝ)
  (hα : 0 ≤ α ∧ α ≤ π / 2)
  (hβ : 0 ≤ β ∧ β ≤ π / 2)
  (hγ : 0 ≤ γ ∧ γ ≤ π / 2)
  (cos_sum : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) :
  2 ≤ (1 + Real.cos α ^ 2) ^ 2 * (Real.sin α) ^ 4
       + (1 + Real.cos β ^ 2) ^ 2 * (Real.sin β) ^ 4
       + (1 + Real.cos γ ^ 2) ^ 2 * (Real.sin γ) ^ 4 ∧
    (1 + Real.cos α ^ 2) ^ 2 * (Real.sin α) ^ 4
       + (1 + Real.cos β ^ 2) ^ 2 * (Real.sin β) ^ 4
       + (1 + Real.cos γ ^ 2) ^ 2 * (Real.sin γ) ^ 4
      ≤ (1 + Real.cos α ^ 2) * (1 + Real.cos β ^ 2) * (1 + Real.cos γ ^ 2) :=
by 
  sorry

end cyclic_cosine_inequality_l232_232977


namespace find_fraction_l232_232996

variable (a b c : ℝ)
variable (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
variable (h1 : (a + b + c) / (a + b - c) = 7)
variable (h2 : (a + b + c) / (a + c - b) = 1.75)

theorem find_fraction : (a + b + c) / (b + c - a) = 3.5 := 
by {
  sorry
}

end find_fraction_l232_232996


namespace root_difference_geom_prog_l232_232078

theorem root_difference_geom_prog
  (x1 x2 x3 : ℝ)
  (h1 : 8 * x1^3 - 22 * x1^2 + 15 * x1 - 2 = 0)
  (h2 : 8 * x2^3 - 22 * x2^2 + 15 * x2 - 2 = 0)
  (h3 : 8 * x3^3 - 22 * x3^2 + 15 * x3 - 2 = 0)
  (geom_prog : ∃ (a r : ℝ), x1 = a / r ∧ x2 = a ∧ x3 = a * r) :
  |x3 - x1| = 33 / 14 :=
by
  sorry

end root_difference_geom_prog_l232_232078


namespace union_A_B_complement_intersection_A_B_l232_232303

-- Define universal set U
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := { x | -5 ≤ x ∧ x ≤ -1 }

-- Define set B
def B : Set ℝ := { x | x ≥ -4 }

-- Prove A ∪ B = [-5, +∞)
theorem union_A_B : A ∪ B = { x : ℝ | -5 ≤ x } :=
by {
  sorry
}

-- Prove complement of A ∩ B with respect to U = (-∞, -4) ∪ (-1, +∞)
theorem complement_intersection_A_B : U \ (A ∩ B) = { x : ℝ | x < -4 } ∪ { x : ℝ | x > -1 } :=
by {
  sorry
}

end union_A_B_complement_intersection_A_B_l232_232303


namespace intersection_of_A_and_B_l232_232268

def setA : Set ℝ := {y | ∃ x : ℝ, y = 2 * x}
def setB : Set ℝ := {y | ∃ x : ℝ, y = x ^ 2}

theorem intersection_of_A_and_B : setA ∩ setB = {y | y ≥ 0} :=
by
  sorry

end intersection_of_A_and_B_l232_232268


namespace value_of_z_sub_y_add_x_l232_232796

-- Represent 312 in base 3
def base3_representation : List ℕ := [1, 0, 1, 2, 1, 0] -- 312 in base 3 is 101210

-- Define x, y, z
def x : ℕ := (base3_representation.count 0)
def y : ℕ := (base3_representation.count 1)
def z : ℕ := (base3_representation.count 2)

-- Proposition to be proved
theorem value_of_z_sub_y_add_x : z - y + x = 2 := by
  sorry

end value_of_z_sub_y_add_x_l232_232796


namespace num_arrangement_options_l232_232711

def competition_events := ["kicking shuttlecocks", "jumping rope", "tug-of-war", "pushing the train", "multi-person multi-foot"]

def is_valid_arrangement (arrangement : List String) : Prop :=
  arrangement.length = 5 ∧
  arrangement.getLast? = some "tug-of-war" ∧
  arrangement.get? 0 ≠ some "multi-person multi-foot"

noncomputable def count_valid_arrangements : ℕ :=
  let positions := ["kicking shuttlecocks", "jumping rope", "pushing the train"]
  3 * positions.permutations.length

theorem num_arrangement_options : count_valid_arrangements = 18 :=
by
  sorry

end num_arrangement_options_l232_232711


namespace combined_seq_20th_term_l232_232558

def arithmetic_seq (a : ℕ) (d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d
def geometric_seq (g : ℕ) (r : ℕ) (n : ℕ) : ℕ := g * r^(n - 1)

theorem combined_seq_20th_term :
  let a := 3
  let d := 4
  let g := 2
  let r := 2
  let n := 20
  arithmetic_seq a d n + geometric_seq g r n = 1048655 :=
by 
  sorry

end combined_seq_20th_term_l232_232558


namespace find_f_l232_232839

theorem find_f 
  (h_vertex : ∃ (d e f : ℝ), ∀ x, y = d * (x - 3)^2 - 5 ∧ y = d * x^2 + e * x + f)
  (h_point : y = d * (4 - 3)^2 - 5) 
  (h_value : y = -3) :
  ∃ f, f = 13 :=
sorry

end find_f_l232_232839


namespace train_length_is_150_meters_l232_232629

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

end train_length_is_150_meters_l232_232629


namespace solve_for_n_l232_232412

theorem solve_for_n (n : ℕ) : (8 ^ n) * (8 ^ n) * (8 ^ n) * (8 ^ n) = 64 ^ 4 → n = 2 :=
by 
  intro h
  sorry

end solve_for_n_l232_232412


namespace simplify_fraction_l232_232778

theorem simplify_fraction (a b : ℝ) :
  ( (3 * b) / (2 * a^2) )^3 = 27 * b^3 / (8 * a^6) :=
by
  sorry

end simplify_fraction_l232_232778


namespace wrapping_paper_needed_l232_232854

-- Define the conditions as variables in Lean
def wrapping_paper_first := 3.5
def wrapping_paper_second := (2 / 3) * wrapping_paper_first
def wrapping_paper_third := wrapping_paper_second + 0.5 * wrapping_paper_second
def wrapping_paper_fourth := wrapping_paper_first + wrapping_paper_second
def wrapping_paper_fifth := wrapping_paper_third - 0.25 * wrapping_paper_third

-- Define the total wrapping paper needed
def total_wrapping_paper := wrapping_paper_first + wrapping_paper_second + wrapping_paper_third + wrapping_paper_fourth + wrapping_paper_fifth

-- Statement to prove the final equivalence
theorem wrapping_paper_needed : 
  total_wrapping_paper = 17.79 := 
sorry  -- Proof is omitted

end wrapping_paper_needed_l232_232854


namespace total_animals_l232_232364

variable (rats chihuahuas : ℕ)
variable (h1 : rats = 60)
variable (h2 : rats = 6 * chihuahuas)

theorem total_animals (rats : ℕ) (chihuahuas : ℕ) (h1 : rats = 60) (h2 : rats = 6 * chihuahuas) : rats + chihuahuas = 70 := by
  sorry

end total_animals_l232_232364


namespace discount_difference_is_correct_l232_232436

-- Define the successive discounts in percentage
def discount1 : ℝ := 0.25
def discount2 : ℝ := 0.15
def discount3 : ℝ := 0.10

-- Define the store's claimed discount
def claimed_discount : ℝ := 0.45

-- Calculate the true discount
def true_discount : ℝ := 1 - ((1 - discount1) * (1 - discount2) * (1 - discount3))

-- Calculate the difference between the true discount and the claimed discount
def discount_difference : ℝ := claimed_discount - true_discount

-- State the theorem to be proved
theorem discount_difference_is_correct : discount_difference = 2.375 / 100 := by
  sorry

end discount_difference_is_correct_l232_232436


namespace prob_neither_defective_l232_232575

-- Definitions for the conditions
def totalPens : ℕ := 8
def defectivePens : ℕ := 2
def nonDefectivePens : ℕ := totalPens - defectivePens
def selectedPens : ℕ := 2

-- Theorem statement for the probability that neither of the two selected pens is defective
theorem prob_neither_defective : 
  (nonDefectivePens / totalPens) * ((nonDefectivePens - 1) / (totalPens - 1)) = 15 / 28 := 
  sorry

end prob_neither_defective_l232_232575


namespace quadratic_equation_in_one_variable_l232_232969

-- Definitions for each condition
def equation_A (x : ℝ) : Prop := x^2 = -1
def equation_B (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0
def equation_C (x : ℝ) : Prop := 2 * (x + 1)^2 = (Real.sqrt 2 * x - 1)^2
def equation_D (x : ℝ) : Prop := x + 1 / x = 1

-- Main theorem statement
theorem quadratic_equation_in_one_variable (x : ℝ) :
  equation_A x ∧ ¬(∃ a b c, equation_B a b c x ∧ a ≠ 0) ∧ ¬equation_C x ∧ ¬equation_D x :=
  sorry

end quadratic_equation_in_one_variable_l232_232969


namespace largest_possible_perimeter_l232_232735

theorem largest_possible_perimeter
  (a b c : ℕ)
  (h1 : a > 2 ∧ b > 2 ∧ c > 2)  -- sides are greater than 2
  (h2 : a = c ∨ b = c ∨ a = b)  -- at least two polygons are congruent
  (h3 : (a - 2) * (b - 2) = 8 ∨ (a - 2) * (c - 2) = 8 ∨ (b - 2) * (c - 2) = 8)  -- possible factorizations
  (h4 : (a - 2) + (b - 2) + (c - 2) = 12)  -- sum of interior angles at A is 360 degrees
  : 2 * a + 2 * b + 2 * c - 6 ≤ 21 :=
sorry

end largest_possible_perimeter_l232_232735


namespace correct_option_l232_232886

theorem correct_option :
  (∀ (a b : ℝ),  3 * a^2 * b - 4 * b * a^2 = -a^2 * b) ∧
  ¬(1 / 7 * (-7) + (-1 / 7) * 7 = 1) ∧
  ¬((-3 / 5)^2 = 9 / 5) ∧
  ¬(∀ (a b : ℝ), 3 * a + 5 * b = 8 * a * b) :=
by
  sorry

end correct_option_l232_232886


namespace slope_product_is_neg_one_l232_232697

noncomputable def slope_product (m n : ℝ) : ℝ := m * n

theorem slope_product_is_neg_one 
  (m n : ℝ)
  (eqn1 : ∀ x, ∃ y, y = m * x)
  (eqn2 : ∀ x, ∃ y, y = n * x)
  (angle : ∃ θ1 θ2 : ℝ, θ1 = θ2 + π / 4)
  (neg_reciprocal : m = -1 / n):
  slope_product m n = -1 := 
sorry

end slope_product_is_neg_one_l232_232697


namespace base_b_square_l232_232261

-- Given that 144 in base b can be written as b^2 + 4b + 4 in base 10,
-- prove that it is a perfect square if and only if b > 4

theorem base_b_square (b : ℕ) (h : b > 4) : ∃ k : ℕ, b^2 + 4 * b + 4 = k^2 := by
  sorry

end base_b_square_l232_232261


namespace mono_increasing_m_value_l232_232477

theorem mono_increasing_m_value (m : ℝ) :
  (∀ x : ℝ, 0 ≤ 3 * x ^ 2 + 4 * x + m) → (m ≥ 4 / 3) :=
by
  intro h
  sorry

end mono_increasing_m_value_l232_232477


namespace abs_z_bounds_l232_232739

open Complex

theorem abs_z_bounds (z : ℂ) (h : abs (z + 1/z) = 1) : 
  (Real.sqrt 5 - 1) / 2 ≤ abs z ∧ abs z ≤ (Real.sqrt 5 + 1) / 2 := 
sorry

end abs_z_bounds_l232_232739


namespace length_of_AB_l232_232069

noncomputable def parabola_intersection (x1 x2 : ℝ) (y1 y2 : ℝ) : ℝ :=
|x1 - x2|

theorem length_of_AB : 
  ∀ (x1 x2 y1 y2 : ℝ),
    (x1 + x2 = 6) →
    (A = (x1, y1)) →
    (B = (x2, y2)) →
    (y1^2 = 4 * x1) →
    (y2^2 = 4 * x2) →
    parabola_intersection x1 x2 y1 y2 = 8 :=
by
  sorry

end length_of_AB_l232_232069


namespace average_goals_per_game_l232_232347

theorem average_goals_per_game
  (slices_per_pizza : ℕ := 12)
  (total_pizzas : ℕ := 6)
  (total_games : ℕ := 8)
  (total_slices : ℕ := total_pizzas * slices_per_pizza)
  (total_goals : ℕ := total_slices)
  (average_goals : ℕ := total_goals / total_games) :
  average_goals = 9 :=
by
  sorry

end average_goals_per_game_l232_232347


namespace lacy_correct_percentage_is_80_l232_232100

-- Define the total number of problems
def total_problems (x : ℕ) : ℕ := 5 * x + 10

-- Define the number of problems Lacy missed
def problems_missed (x : ℕ) : ℕ := x + 2

-- Define the number of problems Lacy answered correctly
def problems_answered (x : ℕ) : ℕ := total_problems x - problems_missed x

-- Define the fraction of problems Lacy answered correctly
def fraction_answered_correctly (x : ℕ) : ℚ :=
  (problems_answered x : ℚ) / (total_problems x : ℚ)

-- The main theorem to prove the percentage of problems correctly answered is 80%
theorem lacy_correct_percentage_is_80 (x : ℕ) : 
  fraction_answered_correctly x = 4 / 5 := 
by 
  sorry

end lacy_correct_percentage_is_80_l232_232100


namespace solve_for_x_l232_232732

theorem solve_for_x (x : ℝ) (h : (4/7) * (2/5) * x = 8) : x = 35 :=
sorry

end solve_for_x_l232_232732


namespace sum_is_odd_square_expression_is_odd_l232_232316

theorem sum_is_odd_square_expression_is_odd (a b c : ℤ) (h : (a + b + c) % 2 = 1) : 
  (a^2 + b^2 - c^2 + 2 * a * b) % 2 = 1 :=
sorry

end sum_is_odd_square_expression_is_odd_l232_232316


namespace larry_daily_dog_time_l232_232496

-- Definitions from the conditions
def half_hour_in_minutes : ℕ := 30
def twice_a_day (minutes : ℕ) : ℕ := 2 * minutes
def one_fifth_hour_in_minutes : ℕ := 60 / 5

-- Hypothesis resulting from the conditions
def time_walking_and_playing : ℕ := twice_a_day half_hour_in_minutes
def time_feeding : ℕ := one_fifth_hour_in_minutes

-- The theorem to prove
theorem larry_daily_dog_time : time_walking_and_playing + time_feeding = 72 := by
  show time_walking_and_playing + time_feeding = 72
  sorry

end larry_daily_dog_time_l232_232496


namespace set_relationship_l232_232284

def set_M : Set ℚ := {x : ℚ | ∃ m : ℤ, x = m + 1/6}
def set_N : Set ℚ := {x : ℚ | ∃ n : ℤ, x = n/2 - 1/3}
def set_P : Set ℚ := {x : ℚ | ∃ p : ℤ, x = p/2 + 1/6}

theorem set_relationship : set_M ⊆ set_N ∧ set_N = set_P := by
  sorry

end set_relationship_l232_232284


namespace trig_evaluation_trig_identity_value_l232_232882

-- Problem 1: Prove the trigonometric evaluation
theorem trig_evaluation :
  (Real.cos (9 * Real.pi / 4)) + (Real.tan (-Real.pi / 4)) + (Real.sin (21 * Real.pi)) = (Real.sqrt 2 / 2) - 1 :=
by
  sorry

-- Problem 2: Prove the value given the trigonometric identity
theorem trig_identity_value (θ : ℝ) (h : Real.sin θ = 2 * Real.cos θ) :
  (Real.sin θ ^ 2 + 2 * Real.sin θ * Real.cos θ) / (2 * Real.sin θ ^ 2 - Real.cos θ ^ 2) = 8 / 7 :=
by
  sorry

end trig_evaluation_trig_identity_value_l232_232882


namespace correct_fraction_l232_232488

theorem correct_fraction (x y : ℕ) (h1 : 480 * 5 / 6 = 480 * x / y + 250) : x / y = 5 / 16 :=
by
  sorry

end correct_fraction_l232_232488


namespace min_value_of_squares_l232_232336

theorem min_value_of_squares (a b c t : ℝ) (h : a + b + c = t) : 
  a^2 + b^2 + c^2 ≥ t^2 / 3 ∧ (∃ (a' b' c' : ℝ), a' = b' ∧ b' = c' ∧ a' + b' + c' = t ∧ a'^2 + b'^2 + c'^2 = t^2 / 3) := 
by
  sorry

end min_value_of_squares_l232_232336


namespace hyperbola_same_foci_as_ellipse_eccentricity_two_l232_232672

theorem hyperbola_same_foci_as_ellipse_eccentricity_two
  (a b c e : ℝ)
  (ellipse_eq : ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (a = 5 ∧ b = 3 ∧ c = 4))
  (eccentricity_eq : e = 2) :
  ∃ x y : ℝ, (x^2 / (c / e)^2 - y^2 / (c^2 - (c / e)^2) = 1) ↔ (x^2 / 4 - y^2 / 12 = 1) :=
by
  sorry

end hyperbola_same_foci_as_ellipse_eccentricity_two_l232_232672


namespace zero_polynomial_is_solution_l232_232256

noncomputable def polynomial_zero (p : Polynomial ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → (p.eval x)^2 + (p.eval (1/x))^2 = (p.eval (x^2)) * (p.eval (1/(x^2))) → p = 0

theorem zero_polynomial_is_solution : ∀ p : Polynomial ℝ, (∀ x : ℝ, x ≠ 0 → (p.eval x)^2 + (p.eval (1/x))^2 = (p.eval (x^2)) * (p.eval (1/(x^2)))) → p = 0 :=
by
  sorry

end zero_polynomial_is_solution_l232_232256


namespace molecular_weight_of_one_mole_l232_232753

theorem molecular_weight_of_one_mole 
  (total_weight : ℝ) (n_moles : ℝ) (mw_per_mole : ℝ)
  (h : total_weight = 792) (h2 : n_moles = 9) 
  (h3 : total_weight = n_moles * mw_per_mole) 
  : mw_per_mole = 88 :=
by
  sorry

end molecular_weight_of_one_mole_l232_232753


namespace carol_twice_as_cathy_l232_232059

-- Define variables for the number of cars each person owns
variables (C L S Ca x : ℕ)

-- Define conditions based on the problem statement
def lindsey_cars := L = C + 4
def susan_cars := S = Ca - 2
def carol_cars := Ca = 2 * x
def total_cars := C + L + S + Ca = 32
def cathy_cars := C = 5

-- State the theorem to prove
theorem carol_twice_as_cathy : 
  lindsey_cars C L ∧ 
  susan_cars S Ca ∧ 
  carol_cars Ca x ∧ 
  total_cars C L S Ca ∧ 
  cathy_cars C
  → x = 5 :=
by
  sorry

end carol_twice_as_cathy_l232_232059


namespace total_food_for_guinea_pigs_l232_232788

-- Definitions of the food consumption for each guinea pig
def first_guinea_pig_food : ℕ := 2
def second_guinea_pig_food : ℕ := 2 * first_guinea_pig_food
def third_guinea_pig_food : ℕ := second_guinea_pig_food + 3

-- Statement to prove the total food required
theorem total_food_for_guinea_pigs : 
  first_guinea_pig_food + second_guinea_pig_food + third_guinea_pig_food = 13 := by
  sorry

end total_food_for_guinea_pigs_l232_232788


namespace negation_proposition_l232_232090

theorem negation_proposition:
  (¬ ∃ x : ℝ, x^2 + 2 * x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2 * x + 5 ≠ 0) :=
by sorry

end negation_proposition_l232_232090


namespace parabola_functions_eq_l232_232733

noncomputable def f (x : ℝ) (b : ℝ) (c : ℝ) : ℝ := x^2 + b * x + c
noncomputable def g (x : ℝ) (c : ℝ) (b : ℝ) : ℝ := x^2 + c * x + b

theorem parabola_functions_eq : ∀ (x₁ x₂ : ℝ), 
  (∃ t : ℝ, (f t b c = g t c b) ∧ (t = 1)) → 
    (f x₁ 2 (-3) = x₁^2 + 2 * x₁ - 3) ∧ (g x₂ (-3) 2 = x₂^2 - 3 * x₂ + 2) :=
sorry

end parabola_functions_eq_l232_232733


namespace necessary_and_sufficient_condition_l232_232705

theorem necessary_and_sufficient_condition (t : ℝ) :
  ((t + 1) * (1 - |t|) > 0) ↔ (t < 1 ∧ t ≠ -1) :=
by
  sorry

end necessary_and_sufficient_condition_l232_232705


namespace new_person_weight_l232_232988

noncomputable def weight_increase (n : ℕ) (avg_increase : ℝ) : ℝ := n * avg_increase

theorem new_person_weight 
  (n : ℕ) (avg_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) 
  (weight_eqn : weight_increase n avg_increase = new_weight - old_weight) : 
  new_weight = 87.5 :=
by
  have n := 9
  have avg_increase := 2.5
  have old_weight := 65
  have weight_increase := 9 * 2.5
  have weight_eqn := weight_increase = 87.5 - 65
  sorry

end new_person_weight_l232_232988


namespace divisor_is_31_l232_232147

-- Definition of the conditions.
def condition1 (x : ℤ) : Prop :=
  ∃ k : ℤ, x = 62 * k + 7

def condition2 (x y : ℤ) : Prop :=
  ∃ m : ℤ, x + 11 = y * m + 18

-- Main statement asserting the divisor y.
theorem divisor_is_31 (x y : ℤ) (h₁ : condition1 x) (h₂ : condition2 x y) : y = 31 :=
sorry

end divisor_is_31_l232_232147


namespace range_of_a_l232_232129

open Set

variable {a x : ℝ}

def A (a : ℝ) : Set ℝ := {x | abs (x - a) < 1}
def B : Set ℝ := {x | 1 < x ∧ x < 5}

theorem range_of_a (h : A a ∩ B = ∅) : a ≤ 0 ∨ a ≥ 6 := 
by 
  sorry

end range_of_a_l232_232129


namespace min_additional_matchsticks_needed_l232_232480

-- Define the number of matchsticks in a 3x7 grid
def matchsticks_in_3x7_grid : Nat := 4 * 7 + 3 * 8

-- Define the number of matchsticks in a 5x5 grid
def matchsticks_in_5x5_grid : Nat := 6 * 5 + 6 * 5

-- Define the minimum number of additional matchsticks required
def additional_matchsticks (matchsticks_in_3x7_grid matchsticks_in_5x5_grid : Nat) : Nat :=
  matchsticks_in_5x5_grid - matchsticks_in_3x7_grid

theorem min_additional_matchsticks_needed :
  additional_matchsticks matchsticks_in_3x7_grid matchsticks_in_5x5_grid = 8 :=
by 
  unfold additional_matchsticks matchsticks_in_3x7_grid matchsticks_in_5x5_grid
  sorry

end min_additional_matchsticks_needed_l232_232480


namespace perpendicular_line_theorem_l232_232138

-- Mathematical definitions used in the condition.
def Line := Type
def Plane := Type

variables {l m : Line} {π : Plane}

-- Given the predicate that a line is perpendicular to another line on the plane
def is_perpendicular (l m : Line) (π : Plane) : Prop :=
sorry -- Definition of perpendicularity in Lean (abstracted here)

-- Given condition: l is perpendicular to the projection of m on plane π
axiom projection_of_oblique (m : Line) (π : Plane) : Line

-- The Perpendicular Line Theorem
theorem perpendicular_line_theorem (h : is_perpendicular l (projection_of_oblique m π) π) : is_perpendicular l m π :=
sorry

end perpendicular_line_theorem_l232_232138


namespace range_of_m_l232_232589

theorem range_of_m (m x1 x2 : ℝ) (h1 : x1 < 0) (h2 : 0 < x2) (h3 : (1 - 2 * m) / x1 < (1 - 2 * m) / x2) : m < 1 / 2 :=
sorry

end range_of_m_l232_232589


namespace radius_of_smaller_circle_l232_232958

theorem radius_of_smaller_circle (A1 : ℝ) (r1 r2 : ℝ) (h1 : π * r2^2 = 4 * A1)
    (h2 : r2 = 4) : r1 = 2 :=
by
  sorry

end radius_of_smaller_circle_l232_232958


namespace reciprocal_of_neg_three_l232_232262

theorem reciprocal_of_neg_three : (1 / (-3 : ℝ)) = (-1 / 3) := by
  sorry

end reciprocal_of_neg_three_l232_232262


namespace rectangle_area_from_perimeter_l232_232746

theorem rectangle_area_from_perimeter
  (a : ℝ)
  (shorter_side := 12 * a)
  (longer_side := 22 * a)
  (P := 2 * (shorter_side + longer_side))
  (hP : P = 102) :
  (shorter_side * longer_side = 594) := by
  sorry

end rectangle_area_from_perimeter_l232_232746


namespace range_of_m_l232_232128

theorem range_of_m {x y : ℝ} (hx : 0 < x) (hy : 0 < y)
  (h_cond : 1/x + 4/y = 1) : 
  (∃ x y, 0 < x ∧ 0 < y ∧ 1/x + 4/y = 1 ∧ x + y/4 < m^2 + 3 * m) ↔
  (m < -4 ∨ 1 < m) := 
sorry

end range_of_m_l232_232128


namespace ab_value_l232_232281

theorem ab_value (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 * b^2 + a^2 * b^3 = 20) : ab = 2 ∨ ab = -2 :=
by
  sorry

end ab_value_l232_232281


namespace ninth_day_skate_time_l232_232143

-- Define the conditions
def first_4_days_skate_time : ℕ := 4 * 70
def second_4_days_skate_time : ℕ := 4 * 100
def total_days : ℕ := 9
def average_minutes_per_day : ℕ := 100

-- Define the theorem stating that Gage must skate 220 minutes on the ninth day to meet the average
theorem ninth_day_skate_time : 
  let total_minutes_needed := total_days * average_minutes_per_day
  let current_skate_time := first_4_days_skate_time + second_4_days_skate_time
  total_minutes_needed - current_skate_time = 220 := 
by
  -- Placeholder for the proof
  sorry

end ninth_day_skate_time_l232_232143


namespace MichelangeloCeilingPainting_l232_232097

theorem MichelangeloCeilingPainting (total_ceiling week1_ceiling next_week_fraction : ℕ) 
  (a1 : total_ceiling = 28) 
  (a2 : week1_ceiling = 12) 
  (a3 : total_ceiling - (week1_ceiling + next_week_fraction * week1_ceiling) = 13) : 
  next_week_fraction = 1 / 4 := 
by 
  sorry

end MichelangeloCeilingPainting_l232_232097


namespace solve_for_m_l232_232968

theorem solve_for_m : 
  ∀ m : ℝ, (3 * (-2) + 5 = -2 - m) → m = -1 :=
by
  intros m h
  sorry

end solve_for_m_l232_232968


namespace intersection_N_complement_M_l232_232547

def U : Set ℝ := Set.univ
def M : Set ℝ := {x : ℝ | x < -2 ∨ x > 2}
def CU_M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x : ℝ | (1 - x) / (x - 3) > 0}

theorem intersection_N_complement_M :
  N ∩ CU_M = {x : ℝ | 1 < x ∧ x ≤ 2} :=
sorry

end intersection_N_complement_M_l232_232547


namespace complete_square_l232_232522

theorem complete_square {x : ℝ} (h : x^2 + 10 * x - 3 = 0) : (x + 5)^2 = 28 :=
sorry

end complete_square_l232_232522


namespace negation_of_exists_proposition_l232_232678

theorem negation_of_exists_proposition :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) :=
sorry

end negation_of_exists_proposition_l232_232678


namespace find_x_l232_232438

def vec_a : ℝ × ℝ × ℝ := (-2, 1, 3)
def vec_b (x : ℝ) : ℝ × ℝ × ℝ := (1, x, -1)

theorem find_x (x : ℝ) (h : (-2) * 1 + 1 * x + 3 * (-1) = 0) : x = 5 :=
by
  sorry

end find_x_l232_232438


namespace intersection_point_of_lines_l232_232265

theorem intersection_point_of_lines :
  ∃ (x y : ℝ), x + 2 * y - 4 = 0 ∧ 2 * x - y + 2 = 0 ∧ (x, y) = (0, 2) :=
by
  sorry

end intersection_point_of_lines_l232_232265


namespace add_base_12_l232_232291

def a_in_base_10 := 10
def b_in_base_10 := 11
def c_base := 12

theorem add_base_12 : 
  let a := 10
  let b := 11
  (3 * c_base ^ 2 + 12 * c_base + 5) + (2 * c_base ^ 2 + a * c_base + b) = 6 * c_base ^ 2 + 3 * c_base + 4 :=
by
  sorry

end add_base_12_l232_232291


namespace algebraic_expression_value_l232_232801

theorem algebraic_expression_value (x : ℝ) (h : 2 * x^2 + 3 * x + 7 = 8) : 4 * x^2 + 6 * x - 9 = -7 :=
by
  sorry

end algebraic_expression_value_l232_232801


namespace three_digit_integer_equal_sum_factorials_l232_232401

open Nat

theorem three_digit_integer_equal_sum_factorials :
  ∃ (a b c : ℕ), a = 1 ∧ b = 4 ∧ c = 5 ∧ 100 * a + 10 * b + c = a.factorial + b.factorial + c.factorial :=
by
  use 1, 4, 5
  simp
  sorry

end three_digit_integer_equal_sum_factorials_l232_232401


namespace product_of_areas_square_of_volume_l232_232826

-- Declare the original dimensions and volume
variables (a b c : ℝ)
def V := a * b * c

-- Declare the areas of the new box
def area_bottom := (a + 2) * (b + 2)
def area_side := (b + 2) * (c + 2)
def area_front := (c + 2) * (a + 2)

-- Final theorem to prove
theorem product_of_areas_square_of_volume :
  (area_bottom a b) * (area_side b c) * (area_front c a) = V a b c ^ 2 :=
sorry

end product_of_areas_square_of_volume_l232_232826


namespace total_weight_of_peppers_l232_232331

theorem total_weight_of_peppers
  (green_peppers : ℝ) 
  (red_peppers : ℝ)
  (h_green : green_peppers = 0.33)
  (h_red : red_peppers = 0.33) :
  green_peppers + red_peppers = 0.66 := 
by
  sorry

end total_weight_of_peppers_l232_232331


namespace candy_days_l232_232749

theorem candy_days (neighbor_candy older_sister_candy candy_per_day : ℝ) 
  (h1 : neighbor_candy = 11.0) 
  (h2 : older_sister_candy = 5.0) 
  (h3 : candy_per_day = 8.0) : 
  ((neighbor_candy + older_sister_candy) / candy_per_day) = 2.0 := 
by 
  sorry

end candy_days_l232_232749


namespace length_of_train_l232_232758

theorem length_of_train (V L : ℝ) (h1 : L = V * 18) (h2 : L + 250 = V * 33) : L = 300 :=
by
  sorry

end length_of_train_l232_232758


namespace candy_difference_l232_232647

theorem candy_difference (frankie_candies : ℕ) (max_candies : ℕ) (h1 : frankie_candies = 74) (h2 : max_candies = 92) : max_candies - frankie_candies = 18 := by
  sorry

end candy_difference_l232_232647


namespace solve_for_x_l232_232123

theorem solve_for_x (x : ℝ) (h : 3 / (x + 2) = 2 / (x - 1)) : x = 7 :=
sorry

end solve_for_x_l232_232123


namespace distance_between_A_and_B_l232_232714

theorem distance_between_A_and_B 
  (d : ℝ)
  (h1 : ∀ (t : ℝ), (t = 2 * (t / 2)) → t = 200) 
  (h2 : ∀ (t : ℝ), 100 = d - (t / 2 + 50))
  (h3 : ∀ (t : ℝ), d = 2 * (d - 60)): 
  d = 300 :=
sorry

end distance_between_A_and_B_l232_232714


namespace negation_proposition_equiv_l232_232054

variable (m : ℤ)

theorem negation_proposition_equiv :
  (¬ ∃ x : ℤ, x^2 + x + m < 0) ↔ (∀ x : ℤ, x^2 + x + m ≥ 0) :=
by
  sorry

end negation_proposition_equiv_l232_232054


namespace ln_abs_x_minus_a_even_iff_a_zero_l232_232243

theorem ln_abs_x_minus_a_even_iff_a_zero (a : ℝ) : 
  (∀ x : ℝ, Real.log (|x - a|) = Real.log (|(-x) - a|)) ↔ a = 0 :=
sorry

end ln_abs_x_minus_a_even_iff_a_zero_l232_232243


namespace simplify_and_evaluate_expression_l232_232046

theorem simplify_and_evaluate_expression (x : ℝ) (hx : x = 4) :
  (1 / (x + 2) + 1) / ((x^2 + 6 * x + 9) / (x^2 - 4)) = 2 / 7 :=
by
  sorry

end simplify_and_evaluate_expression_l232_232046


namespace line_equation_of_intersection_points_l232_232466

theorem line_equation_of_intersection_points (x y : ℝ) :
  (x^2 + y^2 - 6*x - 7 = 0) ∧ (x^2 + y^2 - 6*y - 27 = 0) → (3*x - 3*y = 10) :=
by
  sorry

end line_equation_of_intersection_points_l232_232466


namespace extraMaterialNeeded_l232_232502

-- Box dimensions
def smallBoxLength (a : ℝ) : ℝ := a
def smallBoxWidth (b : ℝ) : ℝ := 1.5 * b
def smallBoxHeight (c : ℝ) : ℝ := c

def largeBoxLength (a : ℝ) : ℝ := 1.5 * a
def largeBoxWidth (b : ℝ) : ℝ := 2 * b
def largeBoxHeight (c : ℝ) : ℝ := 2 * c

-- Volume calculations
def volumeSmallBox (a b c : ℝ) : ℝ := a * (1.5 * b) * c
def volumeLargeBox (a b c : ℝ) : ℝ := (1.5 * a) * (2 * b) * (2 * c)

-- Surface area calculations
def surfaceAreaSmallBox (a b c : ℝ) : ℝ := 2 * (a * (1.5 * b)) + 2 * (a * c) + 2 * ((1.5 * b) * c)
def surfaceAreaLargeBox (a b c : ℝ) : ℝ := 2 * ((1.5 * a) * (2 * b)) + 2 * ((1.5 * a) * (2 * c)) + 2 * ((2 * b) * (2 * c))

-- Proof statement
theorem extraMaterialNeeded (a b c : ℝ) :
  (volumeSmallBox a b c = 1.5 * a * b * c) ∧ (volumeLargeBox a b c = 6 * a * b * c) ∧ 
  (surfaceAreaLargeBox a b c - surfaceAreaSmallBox a b c = 3 * a * b + 4 * a * c + 5 * b * c) :=
by
  sorry

end extraMaterialNeeded_l232_232502


namespace rhombus_side_length_l232_232080

variable {L S : ℝ}

theorem rhombus_side_length (hL : 0 ≤ L) (hS : 0 ≤ S) :
  (∃ m : ℝ, m = 1 / 2 * Real.sqrt (L^2 - 4 * S)) :=
sorry

end rhombus_side_length_l232_232080


namespace problem_I_problem_II_l232_232756

open Set

variable (a x : ℝ)

def p : Prop := ∀ x ∈ Icc (1 : ℝ) 2, x^2 - a ≥ 0
def q : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem problem_I (hp : p a) : a ≤ 1 :=
  sorry

theorem problem_II (hpq : ¬ (p a ∧ q a)) : a ∈ Ioo (-2 : ℝ) (1 : ℝ) ∪ Ioi 1 :=
  sorry

end problem_I_problem_II_l232_232756


namespace part1_part2_l232_232177

noncomputable def f (x k : ℝ) : ℝ := (x ^ 2 + k * x + 1) / (x ^ 2 + 1)

theorem part1 (k : ℝ) (h : k = -4) : ∃ x > 0, f x k = -1 :=
  by sorry -- Proof goes here

theorem part2 (k : ℝ) : (∀ (x1 x2 x3 : ℝ), (0 < x1) → (0 < x2) → (0 < x3) → 
  ∃ a b c, a = f x1 k ∧ b = f x2 k ∧ c = f x3 k ∧ 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)) ↔ (-1 ≤ k ∧ k ≤ 2) :=
  by sorry -- Proof goes here

end part1_part2_l232_232177


namespace remainder_when_divided_by_x_plus_2_l232_232702

-- Define the polynomial q(x)
def q (M N D x : ℝ) : ℝ := M * x^4 + N * x^2 + D * x - 5

-- Define the given conditions
def cond1 (M N D : ℝ) : Prop := q M N D 2 = 15

-- The theorem statement we want to prove
theorem remainder_when_divided_by_x_plus_2 (M N D : ℝ) (h1 : cond1 M N D) : q M N D (-2) = 15 :=
sorry

end remainder_when_divided_by_x_plus_2_l232_232702


namespace divisor_of_4k2_minus_1_squared_iff_even_l232_232174

-- Define the conditions
variable (k : ℕ) (h_pos : 0 < k)

-- Define the theorem
theorem divisor_of_4k2_minus_1_squared_iff_even :
  ∃ n : ℕ, (8 * k * n - 1) ∣ (4 * k ^ 2 - 1) ^ 2 ↔ Even k :=
by { sorry }

end divisor_of_4k2_minus_1_squared_iff_even_l232_232174


namespace raft_time_l232_232301

-- Defining the conditions
def distance_between_villages := 1 -- unit: ed (arbitrary unit)

def steamboat_time := 1 -- unit: hours
def motorboat_time := 3 / 4 -- unit: hours
def motorboat_speed_ratio := 2 -- Motorboat speed is twice steamboat speed in still water

-- Speed equations with the current
def steamboat_speed_with_current (v_s v_c : ℝ) := v_s + v_c = 1 -- unit: ed/hr
def motorboat_speed_with_current (v_s v_c : ℝ) := 2 * v_s + v_c = 4 / 3 -- unit: ed/hr

-- Goal: Prove the time it takes for the raft to travel the same distance downstream
theorem raft_time : ∃ t : ℝ, t = 90 :=
by
  -- Definitions
  let v_s := 1 / 3 -- Speed of the steamboat in still water (derived)
  let v_c := 2 / 3 -- Speed of the current (derived)
  let raft_speed := v_c -- Raft speed equals the speed of the current
  
  -- Calculate the time for the raft to travel the distance
  let raft_time := distance_between_villages / raft_speed
  
  -- Convert time to minutes
  let raft_time_minutes := raft_time * 60
  
  -- Prove the raft time is 90 minutes
  existsi raft_time_minutes
  exact sorry

end raft_time_l232_232301


namespace sequence_statements_correct_l232_232763

theorem sequence_statements_correct (S : ℕ → ℝ) (a : ℕ → ℝ) (T : ℕ → ℝ) 
(h_S_nonzero : ∀ n, n > 0 → S n ≠ 0)
(h_S_T_relation : ∀ n, n > 0 → S n + T n = S n * T n) :
  (a 1 = 2) ∧ (∀ n, n > 0 → T n - T (n - 1) = 1) ∧ (∀ n, n > 0 → S n = (n + 1) / n) :=
by
  sorry

end sequence_statements_correct_l232_232763


namespace flour_price_increase_l232_232897

theorem flour_price_increase (x : ℝ) (hx : x > 0) :
  (9600 / (1.5 * x) - 6000 / x = 0.4) :=
by 
  sorry

end flour_price_increase_l232_232897


namespace determine_c_l232_232298

noncomputable def fib (n : ℕ) : ℕ :=
match n with
| 0     => 0
| 1     => 1
| (n+2) => fib (n+1) + fib n

theorem determine_c (c d : ℤ) (h1 : ∃ s : ℂ, s^2 - s - 1 = 0 ∧ (c : ℂ) * s^19 + (d : ℂ) * s^18 + 1 = 0) : 
  c = 1597 :=
by
  sorry

end determine_c_l232_232298


namespace find_n_l232_232290

-- Definitions and conditions
def painted_total_faces (n : ℕ) : ℕ := 6 * n^2
def total_faces_of_unit_cubes (n : ℕ) : ℕ := 6 * n^3
def fraction_of_red_faces (n : ℕ) : ℚ := (painted_total_faces n : ℚ) / (total_faces_of_unit_cubes n : ℚ)

-- Statement to be proven
theorem find_n (n : ℕ) (h : fraction_of_red_faces n = 1 / 4) : n = 4 :=
by
  sorry

end find_n_l232_232290


namespace pooja_speed_l232_232344

theorem pooja_speed (v : ℝ) 
  (roja_speed : ℝ := 5)
  (distance : ℝ := 32)
  (time : ℝ := 4)
  (h : distance = (roja_speed + v) * time) : v = 3 :=
by
  sorry

end pooja_speed_l232_232344


namespace count_multiples_of_70_in_range_200_to_500_l232_232559

theorem count_multiples_of_70_in_range_200_to_500 : 
  ∃! count, count = 5 ∧ (∀ n, 200 ≤ n ∧ n ≤ 500 ∧ (n % 70 = 0) ↔ n = 210 ∨ n = 280 ∨ n = 350 ∨ n = 420 ∨ n = 490) :=
by
  sorry

end count_multiples_of_70_in_range_200_to_500_l232_232559


namespace ratio_of_a_to_b_and_c_l232_232745

theorem ratio_of_a_to_b_and_c (A B C : ℝ) (h1 : A = 160) (h2 : A + B + C = 400) (h3 : B = (2/3) * (A + C)) :
  A / (B + C) = 2 / 3 :=
by
  sorry

end ratio_of_a_to_b_and_c_l232_232745


namespace equation_of_parallel_line_passing_through_point_l232_232302

variable (x y : ℝ)

def is_point_on_line (x_val y_val : ℝ) (a b c : ℝ) : Prop := a * x_val + b * y_val + c = 0

def is_parallel (slope1 slope2 : ℝ) : Prop := slope1 = slope2

theorem equation_of_parallel_line_passing_through_point :
  (is_point_on_line (-1) 3 1 (-2) 7) ∧ (is_parallel (1 / 2) (1 / 2)) → (∀ x y, is_point_on_line x y 1 (-2) 7) :=
by
  sorry

end equation_of_parallel_line_passing_through_point_l232_232302


namespace m_range_l232_232215

noncomputable def G (x : ℝ) (m : ℝ) : ℝ := (8 * x^2 + 22 * x + 5 * m) / 8

theorem m_range (m : ℝ) : 2.5 ≤ m ∧ m ≤ 3.5 ↔ m = 121 / 40 := by
  sorry

end m_range_l232_232215


namespace find_particular_number_l232_232812

theorem find_particular_number (x : ℝ) (h : 4 * x * 25 = 812) : x = 8.12 :=
by sorry

end find_particular_number_l232_232812


namespace find_annual_interest_rate_l232_232465

theorem find_annual_interest_rate 
  (TD : ℝ) (FV : ℝ) (T : ℝ) (expected_R: ℝ)
  (hTD : TD = 189)
  (hFV : FV = 1764)
  (hT : T = 9 / 12)
  (hExpected : expected_R = 16) : 
  ∃ R : ℝ, 
  (TD = (FV - (FV - TD)) * R * T / 100) ∧ 
  R = expected_R := 
by 
  sorry

end find_annual_interest_rate_l232_232465


namespace trigonometric_expression_evaluation_l232_232041

theorem trigonometric_expression_evaluation
  (α : ℝ)
  (h1 : Real.tan α = -3 / 4) :
  (3 * Real.sin (α / 2) ^ 2 + 
   2 * Real.sin (α / 2) * Real.cos (α / 2) + 
   Real.cos (α / 2) ^ 2 - 2) / 
  (Real.sin (π / 2 + α) * Real.tan (-3 * π + α) + 
   Real.cos (6 * π - α)) = -7 := 
by 
  sorry
  -- This will skip the proof and ensure the Lean code can be built successfully.

end trigonometric_expression_evaluation_l232_232041


namespace math_problem_l232_232860

theorem math_problem (a b : ℝ) :
  (a^2 - 1) * (b^2 - 1) ≥ 0 → a^2 + b^2 - 1 - a^2 * b^2 ≤ 0 :=
by
  sorry

end math_problem_l232_232860


namespace hypotenuse_length_l232_232012

theorem hypotenuse_length (a b c : ℝ) (h₀ : a^2 + b^2 + c^2 = 1800) (h₁ : c^2 = a^2 + b^2) : c = 30 :=
by
  sorry

end hypotenuse_length_l232_232012


namespace intersection_complement_U_l232_232663

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

def B_complement_U : Set ℕ := U \ B

theorem intersection_complement_U (hU : U = {1, 3, 5, 7}) 
                                  (hA : A = {3, 5}) 
                                  (hB : B = {1, 3, 7}) : 
  A ∩ (B_complement_U U B) = {5} := by
  sorry

end intersection_complement_U_l232_232663


namespace sum_of_coordinates_A_l232_232742

theorem sum_of_coordinates_A (a b : ℝ)
  (h1 : a ≠ 0)
  (h2 : ∃ x y : ℝ, y = a * x + 4 ∧ y = 2 * x + b ∧ y = (a / 2) * x + 8) :
  ∃ y : ℝ, y = 13 ∨ y = 20 :=
by
  sorry

end sum_of_coordinates_A_l232_232742


namespace evaluate_polynomial_at_2_l232_232848

def polynomial (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + x^2 + 2 * x + 3

theorem evaluate_polynomial_at_2 : polynomial 2 = 67 := by
  sorry

end evaluate_polynomial_at_2_l232_232848


namespace no_sol_for_frac_eq_l232_232922

theorem no_sol_for_frac_eq (x y : ℕ) (h : x > 1) : ¬ (y^5 + 1 = (x^7 - 1) / (x - 1)) :=
sorry

end no_sol_for_frac_eq_l232_232922


namespace planes_parallel_if_perpendicular_to_same_line_l232_232861

variables {Point : Type} {Line : Type} {Plane : Type} 

-- Definitions and conditions
noncomputable def is_parallel (α β : Plane) : Prop := sorry
noncomputable def is_perpendicular (l : Line) (α : Plane) : Prop := sorry

variables (l1 : Line) (α β : Plane)

theorem planes_parallel_if_perpendicular_to_same_line
  (h1 : is_perpendicular l1 α)
  (h2 : is_perpendicular l1 β) : is_parallel α β := 
sorry

end planes_parallel_if_perpendicular_to_same_line_l232_232861


namespace imag_part_of_complex_squared_is_2_l232_232124

-- Define the complex number 1 + i
def complex_num := (1 : ℂ) + (Complex.I : ℂ)

-- Define the squared value of the complex number
def complex_squared := complex_num ^ 2

-- Define the imaginary part of the squared value
def imag_part := complex_squared.im

-- State the theorem
theorem imag_part_of_complex_squared_is_2 : imag_part = 2 := sorry

end imag_part_of_complex_squared_is_2_l232_232124


namespace multiples_7_not_14_less_350_l232_232233

theorem multiples_7_not_14_less_350 : 
  ∃ n : ℕ, n = 25 ∧ (∀ k : ℕ, k < 350 → (k % 7 = 0 ∧ k % 14 ≠ 0 → k ∈ {7 * m | m : ℕ}) ∨ (k % 14 = 0 → k ∉ {7 * m | m : ℕ})) := 
sorry

end multiples_7_not_14_less_350_l232_232233


namespace polynomial_coefficients_l232_232382

noncomputable def a : ℝ := 15
noncomputable def b : ℝ := -198
noncomputable def c : ℝ := 1

theorem polynomial_coefficients :
  (∀ x₁ x₂ x₃ : ℝ, 
    (x₁ + x₂ + x₃ = 0) ∧ 
    (x₁ * x₂ + x₂ * x₃ + x₃ * x₁ = -3) ∧ 
    (x₁ * x₂ * x₃ = -1) → 
    (a = 15) ∧ 
    (b = -198) ∧ 
    (c = 1)) := 
by sorry

end polynomial_coefficients_l232_232382


namespace total_people_on_bus_l232_232942

def students_left := 42
def students_right := 38
def students_back := 5
def students_aisle := 15
def teachers := 2
def bus_driver := 1

theorem total_people_on_bus : students_left + students_right + students_back + students_aisle + teachers + bus_driver = 103 :=
by
  sorry

end total_people_on_bus_l232_232942


namespace quadratic_has_two_real_roots_l232_232179

theorem quadratic_has_two_real_roots (k : ℝ) (h1 : k ≠ 0) (h2 : 4 - 12 * k ≥ 0) : 0 < k ∧ k ≤ 1 / 3 :=
sorry

end quadratic_has_two_real_roots_l232_232179


namespace Tonya_spent_on_brushes_l232_232196

section
variable (total_spent : ℝ)
variable (cost_canvases : ℝ)
variable (cost_paints : ℝ)
variable (cost_easel : ℝ)
variable (cost_brushes : ℝ)

def Tonya_total_spent : Prop := total_spent = 90.0
def Cost_of_canvases : Prop := cost_canvases = 40.0
def Cost_of_paints : Prop := cost_paints = cost_canvases / 2
def Cost_of_easel : Prop := cost_easel = 15.0
def Cost_of_brushes : Prop := cost_brushes = total_spent - (cost_canvases + cost_paints + cost_easel)

theorem Tonya_spent_on_brushes : Tonya_total_spent total_spent →
  Cost_of_canvases cost_canvases →
  Cost_of_paints cost_paints cost_canvases →
  Cost_of_easel cost_easel →
  Cost_of_brushes cost_brushes total_spent cost_canvases cost_paints cost_easel →
  cost_brushes = 15.0 := by
  intro h_total_spent h_cost_canvases h_cost_paints h_cost_easel h_cost_brushes
  rw [Tonya_total_spent, Cost_of_canvases, Cost_of_paints, Cost_of_easel, Cost_of_brushes] at *
  sorry
end

end Tonya_spent_on_brushes_l232_232196


namespace intersection_A_notB_l232_232917

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A according to the given condition
def A : Set ℝ := { x | |x - 1| > 1 }

-- Define set B according to the given condition
def B : Set ℝ := { x | (x - 1) * (x - 4) > 0 }

-- Define the complement of set B in U
def notB : Set ℝ := { x | 1 ≤ x ∧ x ≤ 4 }

-- Lean statement to prove A ∩ notB = { x | 2 < x ∧ x ≤ 4 }
theorem intersection_A_notB :
  A ∩ notB = { x | 2 < x ∧ x ≤ 4 } :=
sorry

end intersection_A_notB_l232_232917


namespace add_neg_two_eq_zero_l232_232541

theorem add_neg_two_eq_zero :
  (-2) + 2 = 0 :=
by
  sorry

end add_neg_two_eq_zero_l232_232541


namespace profit_in_may_highest_monthly_profit_and_max_value_l232_232251

def f (x : ℕ) : ℕ :=
  if 1 ≤ x ∧ x ≤ 6 then 12 * x + 28 else 200 - 14 * x

theorem profit_in_may :
  f 5 = 88 :=
by sorry

theorem highest_monthly_profit_and_max_value :
  ∃ x, 1 ≤ x ∧ x ≤ 12 ∧ f x = 102 :=
by sorry

end profit_in_may_highest_monthly_profit_and_max_value_l232_232251


namespace solve_equation_l232_232002

theorem solve_equation :
  ∀ x : ℝ, x ≠ 1 → x ≠ 2 → (x + 1) / (x - 1) = 1 / (x - 2) + 1 → x = 3 := by
  sorry

end solve_equation_l232_232002


namespace hardey_fitness_center_ratio_l232_232403

theorem hardey_fitness_center_ratio
  (f m : ℕ)
  (avg_female_weight : ℕ := 140)
  (avg_male_weight : ℕ := 180)
  (avg_overall_weight : ℕ := 160)
  (h1 : avg_female_weight * f + avg_male_weight * m = avg_overall_weight * (f + m)) :
  f = m :=
by
  sorry

end hardey_fitness_center_ratio_l232_232403


namespace total_cost_eq_898_80_l232_232515

theorem total_cost_eq_898_80 (M R F : ℝ)
  (h1 : 10 * M = 24 * R)
  (h2 : 6 * F = 2 * R)
  (h3 : F = 21) :
  4 * M + 3 * R + 5 * F = 898.80 :=
by
  sorry

end total_cost_eq_898_80_l232_232515


namespace males_band_not_orchestra_l232_232797

/-- Define conditions as constants -/
def total_females_band := 150
def total_males_band := 130
def total_females_orchestra := 140
def total_males_orchestra := 160
def females_both := 90
def males_both := 80
def total_students_either := 310

/-- The number of males in the band who are NOT in the orchestra -/
theorem males_band_not_orchestra : total_males_band - males_both = 50 := by
  sorry

end males_band_not_orchestra_l232_232797


namespace divisibility_by_3_divisibility_by_4_l232_232994

-- Proof that 5n^2 + 10n + 8 is divisible by 3 if and only if n ≡ 2 (mod 3)
theorem divisibility_by_3 (n : ℤ) : (5 * n^2 + 10 * n + 8) % 3 = 0 ↔ n % 3 = 2 := 
    sorry

-- Proof that 5n^2 + 10n + 8 is divisible by 4 if and only if n ≡ 0 (mod 2)
theorem divisibility_by_4 (n : ℤ) : (5 * n^2 + 10 * n + 8) % 4 = 0 ↔ n % 2 = 0 :=
    sorry

end divisibility_by_3_divisibility_by_4_l232_232994


namespace total_journey_time_l232_232184

theorem total_journey_time
  (river_speed : ℝ)
  (boat_speed_still_water : ℝ)
  (distance_upstream : ℝ)
  (total_journey_time : ℝ) :
  river_speed = 2 → 
  boat_speed_still_water = 6 → 
  distance_upstream = 48 → 
  total_journey_time = (distance_upstream / (boat_speed_still_water - river_speed) + distance_upstream / (boat_speed_still_water + river_speed)) → 
  total_journey_time = 18 := 
by
  intros h1 h2 h3 h4
  sorry

end total_journey_time_l232_232184


namespace simplify_and_evaluate_l232_232079

theorem simplify_and_evaluate (a : ℤ) (h : a = -2) : 
  (1 - (1 / (a + 1))) / ((a^2 - 2*a + 1) / (a^2 - 1)) = (2 / 3) :=
by
  sorry

end simplify_and_evaluate_l232_232079


namespace find_a_b_l232_232341

noncomputable def f (a b x : ℝ) : ℝ := a * x + b
noncomputable def g (x : ℝ) : ℝ := 3 * x - 6

theorem find_a_b (a b : ℝ) (h : ∀ x : ℝ, g (f a b x) = 4 * x + 7) :
  a + b = 17 / 3 :=
by
  sorry

end find_a_b_l232_232341


namespace solve_quadratic_l232_232070

theorem solve_quadratic :
  ∀ x : ℝ, x * (x - 2) + x - 2 = 0 ↔ (x = 2 ∨ x = -1) :=
by
  sorry

end solve_quadratic_l232_232070


namespace solve_r_minus_s_l232_232115

noncomputable def r := 20
noncomputable def s := 4

theorem solve_r_minus_s
  (h1 : r^2 - 24 * r + 80 = 0)
  (h2 : s^2 - 24 * s + 80 = 0)
  (h3 : r > s) : r - s = 16 :=
by
  sorry

end solve_r_minus_s_l232_232115


namespace update_year_l232_232983

def a (n : ℕ) : ℕ :=
  if n ≤ 7 then 2 * n + 2 else 16 * (5 / 4) ^ (n - 7)

noncomputable def S (n : ℕ) : ℕ :=
  if n ≤ 7 then n^2 + 3 * n else 80 * ((5 / 4) ^ (n - 7)) - 10

noncomputable def avg_maintenance_cost (n : ℕ) : ℚ :=
  (S n : ℚ) / n

theorem update_year (n : ℕ) (h : avg_maintenance_cost n > 12) : n = 9 :=
  by
  sorry

end update_year_l232_232983


namespace verify_calculations_l232_232304

theorem verify_calculations (m n x y a b : ℝ) :
  (2 * m - 3 * n) ^ 2 = 4 * m ^ 2 - 12 * m * n + 9 * n ^ 2 ∧
  (-x + y) ^ 2 = x ^ 2 - 2 * x * y + y ^ 2 ∧
  (a + 2 * b) * (a - 2 * b) = a ^ 2 - 4 * b ^ 2 ∧
  (-2 * x ^ 2 * y ^ 2) ^ 3 / (- x * y) ^ 3 ≠ -2 * x ^ 3 * y ^ 3 :=
by
  sorry

end verify_calculations_l232_232304


namespace max_value_arithmetic_sequence_l232_232312

theorem max_value_arithmetic_sequence
  (a : ℕ → ℝ)
  (a1 d : ℝ)
  (h1 : a 1 = a1)
  (h_diff : ∀ n : ℕ, a (n + 1) = a n + d)
  (ha1_pos : a1 > 0)
  (hd_pos : d > 0)
  (h1_2 : a1 + (a1 + d) ≤ 60)
  (h2_3 : (a1 + d) + (a1 + 2 * d) ≤ 100) :
  5 * a1 + (a1 + 4 * d) ≤ 200 :=
sorry

end max_value_arithmetic_sequence_l232_232312


namespace problem_l232_232104

noncomputable def f (ω x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

theorem problem
  (ω : ℝ) 
  (hω : ω > 0)
  (hab : Real.sqrt (4 + (Real.pi ^ 2) / (ω ^ 2)) = 2 * Real.sqrt 2) :
  f ω 1 = Real.sqrt 3 / 2 := 
sorry

end problem_l232_232104


namespace complement_union_l232_232706

open Set

def set_A : Set ℝ := {x | x ≤ 0}
def set_B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

theorem complement_union (A B : Set ℝ) (hA : A = set_A) (hB : B = set_B) :
  (univ \ (A ∪ B) = {x | 1 < x}) := by
  rw [hA, hB]
  sorry

end complement_union_l232_232706


namespace elvis_ralph_matchsticks_l232_232929

/-- 
   Elvis and Ralph are making square shapes with matchsticks from a box containing 
   50 matchsticks. Elvis makes 4-matchstick squares and Ralph makes 8-matchstick 
   squares. If Elvis makes 5 squares and Ralph makes 3, prove the number of matchsticks 
   left in the box is 6. 
-/
def matchsticks_left_in_box
  (initial_matchsticks : ℕ)
  (elvis_squares : ℕ)
  (elvis_matchsticks : ℕ)
  (ralph_squares : ℕ)
  (ralph_matchsticks : ℕ)
  (elvis_squares_count : ℕ)
  (ralph_squares_count : ℕ) : ℕ :=
  initial_matchsticks - (elvis_squares_count * elvis_matchsticks + ralph_squares_count * ralph_matchsticks)

theorem elvis_ralph_matchsticks : matchsticks_left_in_box 50 4 5 8 3 = 6 := 
  sorry

end elvis_ralph_matchsticks_l232_232929


namespace total_handshakes_l232_232999

theorem total_handshakes (total_people : ℕ) (first_meeting_people : ℕ) (second_meeting_new_people : ℕ) (common_people : ℕ)
  (total_people_is : total_people = 12)
  (first_meeting_people_is : first_meeting_people = 7)
  (second_meeting_new_people_is : second_meeting_new_people = 5)
  (common_people_is : common_people = 2)
  (first_meeting_handshakes : ℕ := (first_meeting_people * (first_meeting_people - 1)) / 2)
  (second_meeting_handshakes: ℕ := (first_meeting_people * (first_meeting_people - 1)) / 2 - (common_people * (common_people - 1)) / 2):
  first_meeting_handshakes + second_meeting_handshakes = 41 := 
sorry

end total_handshakes_l232_232999


namespace prove_inequality_l232_232357

theorem prove_inequality (x : ℝ) (h : 3 * x^2 + x - 8 < 0) : -2 < x ∧ x < 4 / 3 :=
sorry

end prove_inequality_l232_232357


namespace calculate_number_l232_232169

theorem calculate_number (tens ones tenths hundredths : ℝ) 
  (h_tens : tens = 21) 
  (h_ones : ones = 8) 
  (h_tenths : tenths = 5) 
  (h_hundredths : hundredths = 34) :
  tens * 10 + ones * 1 + tenths * 0.1 + hundredths * 0.01 = 218.84 :=
by
  sorry

end calculate_number_l232_232169


namespace max_unmarried_women_l232_232260

theorem max_unmarried_women (total_people : ℕ) (frac_women : ℚ) (frac_married : ℚ)
  (h_total : total_people = 80) (h_frac_women : frac_women = 1 / 4) (h_frac_married : frac_married = 3 / 4) :
  ∃ (max_unmarried_women : ℕ), max_unmarried_women = 20 :=
by
  -- The proof will be filled here
  sorry

end max_unmarried_women_l232_232260


namespace geom_seq_sum_eqn_l232_232543

theorem geom_seq_sum_eqn (n : ℕ) (a : ℚ) (r : ℚ) (S_n : ℚ) : 
  a = 1/3 → r = 1/3 → S_n = 80/243 → S_n = a * (1 - r^n) / (1 - r) → n = 5 :=
by
  intros ha hr hSn hSum
  sorry

end geom_seq_sum_eqn_l232_232543


namespace roman_numeral_calculation_l232_232225

def I : ℕ := 1
def V : ℕ := 5
def X : ℕ := 10
def L : ℕ := 50
def C : ℕ := 100
def D : ℕ := 500
def M : ℕ := 1000

theorem roman_numeral_calculation : 2 * M + 5 * L + 7 * X + 9 * I = 2329 := by
  sorry

end roman_numeral_calculation_l232_232225


namespace fries_remaining_time_l232_232722

def recommendedTime : ℕ := 5 * 60
def timeInOven : ℕ := 45
def remainingTime : ℕ := recommendedTime - timeInOven

theorem fries_remaining_time : remainingTime = 255 :=
by
  sorry

end fries_remaining_time_l232_232722


namespace palindromic_condition_l232_232883

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem palindromic_condition (m n : ℕ) :
  is_palindrome (2^n + 2^m + 1) ↔ (m ≤ 9 ∨ n ≤ 9) :=
sorry

end palindromic_condition_l232_232883


namespace purchase_gifts_and_have_money_left_l232_232718

/-
  We start with 5000 forints in our pocket to buy gifts, visiting three stores.
  In each store, we find a gift that we like and purchase it if we have enough money. 
  The prices in each store are independently 1000, 1500, or 2000 forints, each with a probability of 1/3. 
  What is the probability that we can purchase gifts from all three stores 
  and still have money left (i.e., the total expenditure is at most 4500 forints)?
-/

def giftProbability (totalForints : ℕ) (prices : List ℕ) : ℚ :=
  let outcomes := prices |>.product prices |>.product prices
  let favorable := outcomes.filter (λ ((p1, p2), p3) => p1 + p2 + p3 <= totalForints)
  favorable.length / outcomes.length

theorem purchase_gifts_and_have_money_left :
  giftProbability 4500 [1000, 1500, 2000] = 17 / 27 :=
sorry

end purchase_gifts_and_have_money_left_l232_232718


namespace total_students_l232_232779

theorem total_students (groups students_per_group : ℕ) (h : groups = 6) (k : students_per_group = 5) :
  groups * students_per_group = 30 := 
by
  sorry

end total_students_l232_232779


namespace choosing_officers_l232_232305

noncomputable def total_ways_to_choose_officers (members : List String) (boys : ℕ) (girls : ℕ) : ℕ :=
  let total_members := boys + girls
  let president_choices := total_members
  let vice_president_choices := boys - 1 + girls - 1
  let remaining_members := total_members - 2
  president_choices * vice_president_choices * remaining_members

theorem choosing_officers (members : List String) (boys : ℕ) (girls : ℕ) :
  boys = 15 → girls = 15 → members.length = 30 → total_ways_to_choose_officers members boys girls = 11760 :=
by
  intros hboys hgirls htotal
  rw [hboys, hgirls]
  sorry

end choosing_officers_l232_232305


namespace average_score_for_entire_class_l232_232823

theorem average_score_for_entire_class (n x y : ℕ) (a b : ℝ) (hn : n = 100) (hx : x = 70) (hy : y = 30) (ha : a = 0.65) (hb : b = 0.95) :
    ((x * a + y * b) / n) = 0.74 := by
  sorry

end average_score_for_entire_class_l232_232823


namespace mod_remainder_1287_1499_l232_232320

theorem mod_remainder_1287_1499 : (1287 * 1499) % 300 = 213 := 
by 
  sorry

end mod_remainder_1287_1499_l232_232320


namespace avg_speed_including_stoppages_l232_232723

theorem avg_speed_including_stoppages (speed_without_stoppages : ℝ) (stoppage_time_per_hour : ℝ) 
  (h₁ : speed_without_stoppages = 60) (h₂ : stoppage_time_per_hour = 0.5) : 
  (speed_without_stoppages * (1 - stoppage_time_per_hour)) / 1 = 30 := 
  by 
  sorry

end avg_speed_including_stoppages_l232_232723


namespace inequality_holds_l232_232728

theorem inequality_holds (a b : ℝ) (h : a < b) (h₀ : b < 0) : - (1 / a) < - (1 / b) :=
sorry

end inequality_holds_l232_232728


namespace students_not_enrolled_in_biology_class_l232_232962

theorem students_not_enrolled_in_biology_class (total_students : ℕ) (percent_biology : ℕ) 
  (h1 : total_students = 880) (h2 : percent_biology = 35) : 
  total_students - (percent_biology * total_students / 100) = 572 := by
  sorry

end students_not_enrolled_in_biology_class_l232_232962


namespace vec_same_direction_l232_232585

theorem vec_same_direction (k : ℝ) : (k = 2) ↔ ∃ m : ℝ, m > 0 ∧ (k, 2) = (m * 1, m * 1) :=
by
  sorry

end vec_same_direction_l232_232585


namespace trigonometric_identity_l232_232165

theorem trigonometric_identity (α : ℝ) 
  (h : Real.tan (π / 4 + α) = 1) : 
  (2 * Real.sin α + Real.cos α) / (3 * Real.cos α - Real.sin α) = 1 / 3 :=
by
  sorry

end trigonometric_identity_l232_232165


namespace sunzi_oranges_l232_232510

theorem sunzi_oranges :
  ∃ (a : ℕ), ( 5 * a + 10 * 3 = 60 ) ∧ ( ∀ n, n = 0 → a = 6 ) :=
by
  sorry

end sunzi_oranges_l232_232510


namespace likes_spinach_not_music_lover_l232_232520

universe u

variable (Person : Type u)
variable (likes_spinach is_pearl_diver is_music_lover : Person → Prop)

theorem likes_spinach_not_music_lover :
  (∃ x, likes_spinach x ∧ ¬ is_pearl_diver x) →
  (∀ x, is_music_lover x → (is_pearl_diver x ∨ ¬ likes_spinach x)) →
  (∀ x, (¬ is_pearl_diver x → is_music_lover x) ∨ (is_pearl_diver x → ¬ is_music_lover x)) →
  (∀ x, likes_spinach x → ¬ is_music_lover x) :=
by
  sorry

end likes_spinach_not_music_lover_l232_232520


namespace first_player_always_wins_l232_232952

theorem first_player_always_wins (A B : ℤ) (hA : A ≠ 0) (hB : B ≠ 0) : A + B + 1998 = 0 → 
  (∃ (a b c : ℤ), (a = A ∨ a = B ∨ a = 1998) ∧ (b = A ∨ b = B ∨ b = 1998) ∧ (c = A ∨ c = B ∨ c = 1998) ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  (∃ (r1 r2 : ℚ), r1 ≠ r2 ∧ r1 * r1 * a + r1 * b + c = 0 ∧ r2 * r2 * a + r2 * b + c = 0)) :=
sorry

end first_player_always_wins_l232_232952


namespace product_of_roots_l232_232047

open Real

theorem product_of_roots : (sqrt (Real.exp (1 / 4 * log (16)))) * (sqrt (Real.exp (1 / 6 * log (64)))) = 4 :=
by
  -- sorry is used to bypass the actual proof implementation
  sorry

end product_of_roots_l232_232047


namespace hcf_of_two_numbers_l232_232766

theorem hcf_of_two_numbers (A B H L : ℕ) (h1 : A * B = 1800) (h2 : L = 200) (h3 : A * B = H * L) : H = 9 :=
by
  sorry

end hcf_of_two_numbers_l232_232766


namespace unique_integer_sum_squares_l232_232634

theorem unique_integer_sum_squares (n : ℤ) (h : ∃ d1 d2 d3 d4 : ℕ, d1 * d2 * d3 * d4 = n ∧ n = d1*d1 + d2*d2 + d3*d3 + d4*d4) : n = 42 := 
sorry

end unique_integer_sum_squares_l232_232634


namespace proof_age_gladys_l232_232761

-- Definitions of ages
def age_gladys : ℕ := 30
def age_lucas : ℕ := 5
def age_billy : ℕ := 10

-- Conditions
def condition1 : Prop := age_gladys = 2 * (age_billy + age_lucas)
def condition2 : Prop := age_gladys = 3 * age_billy
def condition3 : Prop := age_lucas + 3 = 8

-- Theorem to prove the correct age of Gladys
theorem proof_age_gladys (G L B : ℕ)
  (h1 : G = 2 * (B + L))
  (h2 : G = 3 * B)
  (h3 : L + 3 = 8) :
  G = 30 :=
sorry

end proof_age_gladys_l232_232761


namespace false_disjunction_implies_both_false_l232_232346

theorem false_disjunction_implies_both_false (p q : Prop) (h : ¬ (p ∨ q)) : ¬ p ∧ ¬ q :=
sorry

end false_disjunction_implies_both_false_l232_232346


namespace product_eq_one_l232_232605

theorem product_eq_one (a b c : ℝ) (h1 : a^2 + 2 = b^4) (h2 : b^2 + 2 = c^4) (h3 : c^2 + 2 = a^4) : 
  (a^2 - 1) * (b^2 - 1) * (c^2 - 1) = 1 :=
sorry

end product_eq_one_l232_232605


namespace students_taller_than_Yoongi_l232_232161

theorem students_taller_than_Yoongi {n total shorter : ℕ} (h1 : total = 20) (h2 : shorter = 11) : n = 8 :=
by
  sorry

end students_taller_than_Yoongi_l232_232161


namespace school_accomodation_proof_l232_232594

theorem school_accomodation_proof
  (total_classrooms : ℕ) 
  (fraction_classrooms_45 : ℕ) 
  (fraction_classrooms_38 : ℕ)
  (fraction_classrooms_32 : ℕ)
  (fraction_classrooms_25 : ℕ)
  (desks_45 : ℕ)
  (desks_38 : ℕ)
  (desks_32 : ℕ)
  (desks_25 : ℕ)
  (student_capacity_limit : ℕ) :
  total_classrooms = 50 ->
  fraction_classrooms_45 = (3 / 10) * total_classrooms -> 
  fraction_classrooms_38 = (1 / 4) * total_classrooms -> 
  fraction_classrooms_32 = (1 / 5) * total_classrooms -> 
  fraction_classrooms_25 = (total_classrooms - fraction_classrooms_45 - fraction_classrooms_38 - fraction_classrooms_32) ->
  desks_45 = 15 * 45 -> 
  desks_38 = 12 * 38 -> 
  desks_32 = 10 * 32 -> 
  desks_25 = fraction_classrooms_25 * 25 -> 
  student_capacity_limit = 1800 -> 
  fraction_classrooms_45 * 45 +
  fraction_classrooms_38 * 38 +
  fraction_classrooms_32 * 32 + 
  fraction_classrooms_25 * 25 = 1776 + sorry
  :=
sorry

end school_accomodation_proof_l232_232594


namespace fixed_point_of_family_of_lines_l232_232927

theorem fixed_point_of_family_of_lines :
  ∀ (m : ℝ), ∃ (x y : ℝ), (2 * x - m * y + 1 - 3 * m = 0) ∧ (x = -1 / 2) ∧ (y = -3) :=
by
  intro m
  use -1 / 2, -3
  constructor
  · sorry
  constructor
  · rfl
  · rfl

end fixed_point_of_family_of_lines_l232_232927


namespace fraction_sum_l232_232415

theorem fraction_sum : (3 / 8) + (9 / 12) + (5 / 6) = 47 / 24 := by
  sorry

end fraction_sum_l232_232415


namespace number_of_women_bathing_suits_correct_l232_232414

def men_bathing_suits : ℕ := 14797
def total_bathing_suits : ℕ := 19766

def women_bathing_suits : ℕ :=
  total_bathing_suits - men_bathing_suits

theorem number_of_women_bathing_suits_correct :
  women_bathing_suits = 19669 := by
  -- proof goes here
  sorry

end number_of_women_bathing_suits_correct_l232_232414


namespace find_x_y_l232_232880

theorem find_x_y (a n x y : ℕ) (hx4 : 1000 ≤ x ∧ x < 10000) (hy4 : 1000 ≤ y ∧ y < 10000) 
  (h_yx : y > x) (h_y : y = a * 10 ^ n) 
  (h_sum : (x / 1000) + ((x % 1000) / 100) = 5 * a) 
  (ha : a = 2) (hn : n = 3) :
  x = 1990 ∧ y = 2000 := 
by 
  sorry

end find_x_y_l232_232880


namespace ava_planted_9_trees_l232_232583

theorem ava_planted_9_trees
  (L : ℕ)
  (hAva : ∀ L, Ava = L + 3)
  (hTotal : L + (L + 3) = 15) : 
  Ava = 9 :=
by
  sorry

end ava_planted_9_trees_l232_232583


namespace maximal_value_6tuple_l232_232908

theorem maximal_value_6tuple :
  ∀ (a b c d e f : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧ f ≥ 0 ∧ 
  a + b + c + d + e + f = 6 → 
  a * b * c + b * c * d + c * d * e + d * e * f + e * f * a + f * a * b ≤ 8 ∧ 
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 2 ∧
  ((a, b, c, d, e, f) = (0, 0, t, 2, 2, 2 - t) ∨ 
   (a, b, c, d, e, f) = (0, t, 2, 2 - t, 0, 0) ∨ 
   (a, b, c, d, e, f) = (t, 2, 2 - t, 0, 0, 0) ∨ 
   (a, b, c, d, e, f) = (2, 2 - t, 0, 0, 0, t) ∨
   (a, b, c, d, e, f) = (2 - t, 0, 0, 0, t, 2) ∨
   (a, b, c, d, e, f) = (0, 0, 0, t, 2, 2 - t))) := 
sorry

end maximal_value_6tuple_l232_232908


namespace compute_floor_expression_l232_232370

theorem compute_floor_expression : 
  (Int.floor (↑(2025^3) / (2023 * 2024 : ℤ) - ↑(2023^3) / (2024 * 2025 : ℤ)) = 8) := 
sorry

end compute_floor_expression_l232_232370


namespace cos_alpha_value_l232_232045

theorem cos_alpha_value (θ α : Real) (P : Real × Real)
  (hP : P = (-3/5, 4/5))
  (hθ : θ = Real.arccos (-3/5))
  (hαθ : α = θ - Real.pi / 3) :
  Real.cos α = (4 * Real.sqrt 3 - 3) / 10 := 
by 
  sorry

end cos_alpha_value_l232_232045


namespace least_number_to_add_l232_232781

theorem least_number_to_add (x : ℕ) : (1056 + x) % 28 = 0 ↔ x = 4 :=
by sorry

end least_number_to_add_l232_232781


namespace ab_gt_c_l232_232435

theorem ab_gt_c {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (h : 1 / a + 4 / b = 1) (hc : c < 9) : a + b > c :=
sorry

end ab_gt_c_l232_232435


namespace part_1_part_2_l232_232372

-- Definitions for sets M and N
def M : Set ℝ := {x | -2 < x ∧ x < 3}
def N (m : ℝ) : Set ℝ := {x | x ≥ m}

-- Proof problem 1: Prove that if M ∪ N = N, then m ≤ -2
theorem part_1 (m : ℝ) : (M ∪ N m = N m) → m ≤ -2 :=
by sorry

-- Proof problem 2: Prove that if M ∩ N = ∅, then m ≥ 3
theorem part_2 (m : ℝ) : (M ∩ N m = ∅) → m ≥ 3 :=
by sorry

end part_1_part_2_l232_232372


namespace part1_3_neg5_is_pair_part1_neg2_4_is_not_pair_part2_find_n_part3_find_k_l232_232236

def is_equation_number_pair (a b : ℝ) : Prop :=
  ∀ x : ℝ, (x = 1 / (a + b) ↔ a / x + 1 = b)

theorem part1_3_neg5_is_pair : is_equation_number_pair 3 (-5) :=
sorry

theorem part1_neg2_4_is_not_pair : ¬ is_equation_number_pair (-2) 4 :=
sorry

theorem part2_find_n (n : ℝ) : is_equation_number_pair n (3 - n) ↔ n = 1 / 2 :=
sorry

theorem part3_find_k (m k : ℝ) (hm : m ≠ -1) (hm0 : m ≠ 0) (hk1 : k ≠ 1) :
  is_equation_number_pair (m - k) k → k = (m^2 + 1) / (m + 1) :=
sorry

end part1_3_neg5_is_pair_part1_neg2_4_is_not_pair_part2_find_n_part3_find_k_l232_232236


namespace trigonometric_identity_l232_232666

variable {a b c A B C : ℝ}

theorem trigonometric_identity (h1 : 2 * c^2 - 2 * a^2 = b^2) 
  (cos_A : ℝ) (cos_C : ℝ) 
  (h_cos_A : cos_A = (b^2 + c^2 - a^2) / (2 * b * c))
  (h_cos_C : cos_C = (a^2 + b^2 - c^2) / (2 * a * b)) :
  2 * c * cos_A - 2 * a * cos_C = b := 
sorry

end trigonometric_identity_l232_232666


namespace diagonal_angle_with_plane_l232_232654

theorem diagonal_angle_with_plane (α : ℝ) {a : ℝ} 
  (h_square: a > 0)
  (θ : ℝ := Real.arcsin ((Real.sin α) / Real.sqrt 2)): 
  ∃ (β : ℝ), β = θ :=
sorry

end diagonal_angle_with_plane_l232_232654


namespace current_babysitter_hourly_rate_l232_232624

-- Define variables
def new_babysitter_hourly_rate := 12
def extra_charge_per_scream := 3
def hours_hired := 6
def number_of_screams := 2
def cost_difference := 18

-- Define the total cost calculations
def new_babysitter_total_cost :=
  new_babysitter_hourly_rate * hours_hired + extra_charge_per_scream * number_of_screams

def current_babysitter_total_cost :=
  new_babysitter_total_cost + cost_difference

theorem current_babysitter_hourly_rate :
  current_babysitter_total_cost / hours_hired = 16 := by
  sorry

end current_babysitter_hourly_rate_l232_232624


namespace Z_is_1_5_decades_younger_l232_232814

theorem Z_is_1_5_decades_younger (X Y Z : ℝ) (h : X + Y = Y + Z + 15) : (X - Z) / 10 = 1.5 :=
by
  sorry

end Z_is_1_5_decades_younger_l232_232814


namespace negation_equivalence_l232_232133

variables (x : ℝ)

def is_irrational (x : ℝ) : Prop := ¬ ∃ (q : ℚ), ↑q = x

def has_rational_square (x : ℝ) : Prop := ∃ (q : ℚ), ↑q * ↑q = x * x

def proposition := ∃ (x : ℝ), is_irrational x ∧ has_rational_square x

theorem negation_equivalence :
  (¬ proposition) ↔ ∀ (x : ℝ), is_irrational x → ¬ has_rational_square x :=
by sorry

end negation_equivalence_l232_232133


namespace min_x_plus_4y_min_value_l232_232724

noncomputable def min_x_plus_4y (x y: ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(2 * y) = 1) : ℝ :=
  x + 4 * y

theorem min_x_plus_4y_min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(2 * y) = 1) :
  min_x_plus_4y x y hx hy h = 3 + 2 * Real.sqrt 2 :=
sorry

end min_x_plus_4y_min_value_l232_232724


namespace fraction_simplification_l232_232000

theorem fraction_simplification (a b d : ℝ) (h : a^2 + d^2 - b^2 + 2 * a * d ≠ 0) :
  (a^2 + b^2 + d^2 + 2 * b * d) / (a^2 + d^2 - b^2 + 2 * a * d) = (a^2 + (b + d)^2) / ((a + d)^2 + a^2 - b^2) :=
sorry

end fraction_simplification_l232_232000


namespace log8_512_is_3_l232_232106

def log_base_8_of_512 : Prop :=
  ∀ (log8 : ℝ → ℝ),
    (log8 8 = 1 / 3 * log8 2) →
    (log8 512 = 9 * log8 2) →
    log8 8 = 3 → log8 512 = 3

theorem log8_512_is_3 : log_base_8_of_512 :=
by
  intros log8 H1 H2 H3
  -- here you would normally provide the detailed steps to solve this.
  -- however, we directly proclaim the result due to the proof being non-trivial.
  sorry

end log8_512_is_3_l232_232106


namespace find_angle_A_find_area_l232_232493

noncomputable def triangle_area (a b c : ℝ) (A : ℝ) : ℝ :=
  0.5 * b * c * Real.sin A

theorem find_angle_A (a b c A : ℝ)
  (h1: ∀ x, 4 * Real.cos x * Real.sin (x - π/6) ≤ 4 * Real.cos A * Real.sin (A - π/6))
  (h2: a = b^2 + c^2 - 2 * b * c * Real.cos A) : 
  A = π / 3 := by
  sorry

theorem find_area (a b c : ℝ)
  (A : ℝ) (hA : A = π / 3)
  (ha : a = Real.sqrt 7) (hb : b = 2) 
  : triangle_area a b c A = (3 * Real.sqrt 3) / 2 := by
  sorry

end find_angle_A_find_area_l232_232493


namespace secretary_worked_longest_l232_232492

theorem secretary_worked_longest
  (h1 : ∀ (x : ℕ), 3 * x + 5 * x + 7 * x + 11 * x = 2080)
  (h2 : ∀ (a b c d : ℕ), a = 3 * x ∧ b = 5 * x ∧ c = 7 * x ∧ d = 11 * x → d = 11 * x):
  ∃ y : ℕ, y = 880 :=
by
  sorry

end secretary_worked_longest_l232_232492


namespace train_crossing_time_l232_232909

theorem train_crossing_time 
  (train_length : ℕ) 
  (train_speed_kmph : ℕ) 
  (conversion_factor : ℚ := 1000/3600) 
  (train_speed_mps : ℚ := train_speed_kmph * conversion_factor) :
  train_length = 100 →
  train_speed_kmph = 72 →
  train_speed_mps = 20 →
  train_length / train_speed_mps = 5 :=
by
  intros
  sorry

end train_crossing_time_l232_232909


namespace sixth_inequality_l232_232479

theorem sixth_inequality :
  (1 + 1/2^2 + 1/3^2 + 1/4^2 + 1/5^2 + 1/6^2 + 1/7^2) < 13/7 :=
  sorry

end sixth_inequality_l232_232479


namespace initial_roses_l232_232322

theorem initial_roses (R : ℕ) (initial_orchids : ℕ) (current_orchids : ℕ) (current_roses : ℕ) (added_orchids : ℕ) (added_roses : ℕ) :
  initial_orchids = 84 →
  current_orchids = 91 →
  current_roses = 14 →
  added_orchids = current_orchids - initial_orchids →
  added_roses = added_orchids →
  (R + added_roses = current_roses) →
  R = 7 :=
by
  sorry

end initial_roses_l232_232322


namespace trig_expression_value_quadratic_roots_l232_232120

theorem trig_expression_value :
  (Real.tan (Real.pi / 6))^2 + 2 * Real.sin (Real.pi / 4) - 2 * Real.cos (Real.pi / 3) = (3 * Real.sqrt 2 - 2) / 3 := by
  sorry

theorem quadratic_roots :
  (∀ x : ℝ, 2 * x^2 + 4 * x + 1 = 0 ↔ x = (-2 + Real.sqrt 2) / 2 ∨ x = (-2 - Real.sqrt 2) / 2) := by
  sorry

end trig_expression_value_quadratic_roots_l232_232120


namespace Bran_remaining_payment_l232_232607

theorem Bran_remaining_payment :
  let tuition_fee : ℝ := 90
  let job_income_per_month : ℝ := 15
  let scholarship_percentage : ℝ := 0.30
  let months : ℕ := 3
  let scholarship_amount : ℝ := tuition_fee * scholarship_percentage
  let remaining_after_scholarship : ℝ := tuition_fee - scholarship_amount
  let total_job_income : ℝ := job_income_per_month * months
  let amount_to_pay : ℝ := remaining_after_scholarship - total_job_income
  amount_to_pay = 18 := sorry

end Bran_remaining_payment_l232_232607


namespace max_profit_l232_232584

-- Definitions based on conditions from the problem
def L1 (x : ℕ) : ℤ := -5 * (x : ℤ)^2 + 900 * (x : ℤ) - 16000
def L2 (x : ℕ) : ℤ := 300 * (x : ℤ) - 2000
def total_vehicles := 110
def total_profit (x : ℕ) : ℤ := L1 x + L2 (total_vehicles - x)

-- Statement of the problem
theorem max_profit :
  ∃ x y : ℕ, x + y = 110 ∧ x ≥ 0 ∧ y ≥ 0 ∧
  (L1 x + L2 y = 33000 ∧
   (∀ z w : ℕ, z + w = 110 ∧ z ≥ 0 ∧ w ≥ 0 → L1 z + L2 w ≤ 33000)) :=
sorry

end max_profit_l232_232584


namespace range_of_a_l232_232317
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := |x - 1| + |2 * x - a|

theorem range_of_a (a : ℝ)
  (h : ∀ x : ℝ, f x a ≥ (1 / 4) * a ^ 2 + 1) : -2 ≤ a ∧ a ≤ 0 :=
by sorry

end range_of_a_l232_232317


namespace kimberly_total_skittles_l232_232183

def initial_skittles : ℝ := 7.5
def skittles_eaten : ℝ := 2.25
def skittles_given : ℝ := 1.5
def promotion_skittles : ℝ := 3.75
def oranges_bought : ℝ := 18
def exchange_oranges : ℝ := 6
def exchange_skittles : ℝ := 10.5

theorem kimberly_total_skittles :
  initial_skittles - skittles_eaten - skittles_given + promotion_skittles + exchange_skittles = 18 := by
  sorry

end kimberly_total_skittles_l232_232183


namespace estate_area_correct_l232_232127

-- Define the basic parameters given in the problem
def scale : ℝ := 500  -- 500 miles per inch
def width_on_map : ℝ := 5  -- 5 inches
def height_on_map : ℝ := 3  -- 3 inches

-- Define actual dimensions based on the scale
def actual_width : ℝ := width_on_map * scale  -- actual width in miles
def actual_height : ℝ := height_on_map * scale  -- actual height in miles

-- Define the expected actual area of the estate
def actual_area : ℝ := 3750000  -- actual area in square miles

-- The main theorem to prove
theorem estate_area_correct :
  (actual_width * actual_height) = actual_area := by
  sorry

end estate_area_correct_l232_232127


namespace find_speed_of_first_train_l232_232782

noncomputable def relative_speed (length1 length2 : ℕ) (time_seconds : ℝ) : ℝ :=
  let total_length_km := (length1 + length2) / 1000
  let time_hours := time_seconds / 3600
  total_length_km / time_hours

theorem find_speed_of_first_train
  (length1 : ℕ)   -- Length of the first train in meters
  (length2 : ℕ)   -- Length of the second train in meters
  (speed2 : ℝ)    -- Speed of the second train in km/h
  (time_seconds : ℝ)  -- Time in seconds to be clear from each other
  (correct_speed1 : ℝ)  -- Correct speed of the first train in km/h
  (h_length1 : length1 = 160)
  (h_length2 : length2 = 280)
  (h_speed2 : speed2 = 30)
  (h_time_seconds : time_seconds = 21.998240140788738)
  (h_correct_speed1 : correct_speed1 = 41.98) :
  relative_speed length1 length2 time_seconds = speed2 + correct_speed1 :=
by
  sorry

end find_speed_of_first_train_l232_232782


namespace angle_measure_l232_232885

variable (x : ℝ)

noncomputable def is_supplement (x : ℝ) : Prop := 180 - x = 3 * (90 - x) - 60

theorem angle_measure : is_supplement x → x = 15 :=
by
  sorry

end angle_measure_l232_232885


namespace petya_coin_difference_20_l232_232668

-- Definitions for the problem conditions
variables (n k : ℕ) -- n: number of 5-ruble coins Petya has, k: number of 2-ruble coins Petya has

-- Condition: Petya has 60 rubles more than Vanya
def petya_has_60_more (n k : ℕ) : Prop := (5 * n + 2 * k = 5 * k + 2 * n + 60)

-- Theorem to prove Petya has 20 more 5-ruble coins than 2-ruble coins
theorem petya_coin_difference_20 (n k : ℕ) (h : petya_has_60_more n k) : n - k = 20 :=
sorry

end petya_coin_difference_20_l232_232668


namespace polar_to_cartesian_2_pi_over_6_l232_232508

theorem polar_to_cartesian_2_pi_over_6 :
  let r : ℝ := 2
  let θ : ℝ := (Real.pi / 6)
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y) = (Real.sqrt 3, 1) := by
    -- Initialize the constants and their values
    let r := 2
    let θ := Real.pi / 6
    let x := r * Real.cos θ
    let y := r * Real.sin θ
    -- Placeholder for the actual proof
    sorry

end polar_to_cartesian_2_pi_over_6_l232_232508


namespace power_function_decreasing_l232_232114

theorem power_function_decreasing (m : ℝ) (x : ℝ) (hx : x > 0) :
  (m^2 - 2*m - 2 = 1) ∧ (-4*m - 2 < 0) → m = 3 :=
by
  sorry

end power_function_decreasing_l232_232114


namespace part_1_property_part_2_property_part_3_geometric_l232_232431

-- Defining properties
def prop1 (a : ℕ → ℕ) (i j m: ℕ) : Prop := i > j ∧ (a i)^2 / (a j) = a m
def prop2 (a : ℕ → ℕ) (n k l: ℕ) : Prop := n ≥ 3 ∧ k > l ∧ (a n) = (a k)^2 / (a l)

-- Part I: Sequence {a_n = n} check for property 1
theorem part_1_property (a : ℕ → ℕ) (h : ∀ n, a n = n) : ¬∃ i j m, prop1 a i j m := by
  sorry

-- Part II: Sequence {a_n = 2^(n-1)} check for property 1 and 2
theorem part_2_property (a : ℕ → ℕ) (h : ∀ n, a n = 2^(n-1)) : 
  (∀ i j, ∃ m, prop1 a i j m) ∧ (∀ n k l, prop2 a n k l) := by
  sorry

-- Part III: Increasing sequence that satisfies both properties is a geometric sequence
theorem part_3_geometric (a : ℕ → ℕ) (h_inc : ∀ n m, n < m → a n < a m) 
  (h_prop1 : ∀ i j, i > j → ∃ m, prop1 a i j m)
  (h_prop2 : ∀ n, n ≥ 3 → ∃ k l, k > l ∧ (a n) = (a k)^2 / (a l)) : 
  ∃ r, ∀ n, a (n + 1) = r * a n := by
  sorry

end part_1_property_part_2_property_part_3_geometric_l232_232431


namespace oscar_leap_more_than_piper_hop_l232_232019

noncomputable def difference_leap_hop : ℝ :=
let number_of_poles := 51
let total_distance := 7920 -- in feet
let Elmer_strides_per_gap := 44
let Oscar_leaps_per_gap := 15
let Piper_hops_per_gap := 22
let number_of_gaps := number_of_poles - 1
let Elmer_total_strides := Elmer_strides_per_gap * number_of_gaps
let Oscar_total_leaps := Oscar_leaps_per_gap * number_of_gaps
let Piper_total_hops := Piper_hops_per_gap * number_of_gaps
let Elmer_stride_length := total_distance / Elmer_total_strides
let Oscar_leap_length := total_distance / Oscar_total_leaps
let Piper_hop_length := total_distance / Piper_total_hops
Oscar_leap_length - Piper_hop_length

theorem oscar_leap_more_than_piper_hop :
  difference_leap_hop = 3.36 := by
  sorry

end oscar_leap_more_than_piper_hop_l232_232019


namespace quadratic_equation_nonzero_coefficient_l232_232636

theorem quadratic_equation_nonzero_coefficient (m : ℝ) : 
  m - 1 ≠ 0 ↔ m ≠ 1 :=
by
  sorry

end quadratic_equation_nonzero_coefficient_l232_232636


namespace inequality_greater_sqrt_two_l232_232323

theorem inequality_greater_sqrt_two (x y : ℝ) (hx : x > Real.sqrt 2) (hy : y > Real.sqrt 2) : 
  x^4 - x^3 * y + x^2 * y^2 - x * y^3 + y^4 > x^2 + y^2 := 
by 
  sorry

end inequality_greater_sqrt_two_l232_232323


namespace find_m_l232_232734

open Set

def A : Set ℕ := {1, 3, 5}
def B (m : ℕ) : Set ℕ := {1, m}
def C (m : ℕ) : Set ℕ := {1, m}

theorem find_m (m : ℕ) (h : A ∩ B m = C m) : m = 3 ∨ m = 5 :=
sorry

end find_m_l232_232734


namespace scientific_notation_470M_l232_232151

theorem scientific_notation_470M :
  (470000000 : ℝ) = 4.7 * 10^8 :=
sorry

end scientific_notation_470M_l232_232151


namespace calculate_expression_l232_232751

-- Theorem statement for the provided problem
theorem calculate_expression :
  ((18 ^ 15 / 18 ^ 14)^3 * 8 ^ 3) / 4 ^ 5 = 2916 := by
  sorry

end calculate_expression_l232_232751


namespace cos_largest_angle_value_l232_232803

noncomputable def cos_largest_angle (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) : ℝ :=
  (a * a + b * b - c * c) / (2 * a * b)

theorem cos_largest_angle_value : cos_largest_angle 2 3 4 (by rfl) (by rfl) (by rfl) = -1 / 4 := 
sorry

end cos_largest_angle_value_l232_232803


namespace total_cards_is_56_l232_232006

-- Let n be the number of Pokemon cards each person has
def n : Nat := 14

-- Let k be the number of people
def k : Nat := 4

-- Total number of Pokemon cards
def total_cards : Nat := n * k

-- Prove that the total number of Pokemon cards is 56
theorem total_cards_is_56 : total_cards = 56 := by
  sorry

end total_cards_is_56_l232_232006


namespace johns_total_cost_l232_232244

-- Definitions for the prices and quantities
def price_shirt : ℝ := 15.75
def price_tie : ℝ := 9.40
def quantity_shirts : ℕ := 3
def quantity_ties : ℕ := 2

-- Definition for the total cost calculation
def total_cost (price_shirt price_tie : ℝ) (quantity_shirts quantity_ties : ℕ) : ℝ :=
  (price_shirt * quantity_shirts) + (price_tie * quantity_ties)

-- Theorem stating the total cost calculation for John's purchase
theorem johns_total_cost : total_cost price_shirt price_tie quantity_shirts quantity_ties = 66.05 :=
by
  sorry

end johns_total_cost_l232_232244


namespace complement_intersection_l232_232101

open Set

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def M : Set Int := {-1, 0, 1, 3}
def N : Set Int := {-2, 0, 2, 3}

theorem complement_intersection :
  ((U \ M) ∩ N = {-2, 2}) :=
by sorry

end complement_intersection_l232_232101


namespace estimate_passed_students_l232_232696

-- Definitions for the given conditions
def total_papers_in_city : ℕ := 5000
def papers_selected : ℕ := 400
def papers_passed : ℕ := 360

-- The theorem stating the problem in Lean
theorem estimate_passed_students : 
    (5000:ℕ) * ((360:ℕ) / (400:ℕ)) = (4500:ℕ) :=
by
  -- Providing a trivial sorry to skip the proof.
  sorry

end estimate_passed_students_l232_232696


namespace total_population_of_Springfield_and_Greenville_l232_232280

theorem total_population_of_Springfield_and_Greenville :
  let Springfield := 482653
  let diff := 119666
  let Greenville := Springfield - diff
  Springfield + Greenville = 845640 := by
  sorry

end total_population_of_Springfield_and_Greenville_l232_232280


namespace contracting_arrangements_1680_l232_232063

def num_contracting_arrangements (n a b c d : ℕ) : ℕ :=
  Nat.choose n a * Nat.choose (n - a) b * Nat.choose (n - a - b) c

theorem contracting_arrangements_1680 : num_contracting_arrangements 8 3 1 2 2 = 1680 := by
  unfold num_contracting_arrangements
  simp
  sorry

end contracting_arrangements_1680_l232_232063


namespace negation_of_proposition_l232_232602

theorem negation_of_proposition:
  (∀ x : ℝ, x ≥ 0 → x - 2 > 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x - 2 ≤ 0) := 
sorry

end negation_of_proposition_l232_232602


namespace set_intersection_complement_l232_232673

theorem set_intersection_complement (U M N : Set ℤ)
  (hU : U = {0, -1, -2, -3, -4})
  (hM : M = {0, -1, -2})
  (hN : N = {0, -3, -4}) :
  (U \ M) ∩ N = {-3, -4} :=
by
  sorry

end set_intersection_complement_l232_232673


namespace feathers_to_cars_ratio_l232_232604

theorem feathers_to_cars_ratio (initial_feathers : ℕ) (final_feathers : ℕ) (cars_dodged : ℕ)
  (h₁ : initial_feathers = 5263) (h₂ : final_feathers = 5217) (h₃ : cars_dodged = 23) :
  (initial_feathers - final_feathers) / cars_dodged = 2 :=
by
  sorry

end feathers_to_cars_ratio_l232_232604


namespace power_of_square_l232_232855

variable {R : Type*} [CommRing R] (a : R)

theorem power_of_square (a : R) : (3 * a^2)^2 = 9 * a^4 :=
by sorry

end power_of_square_l232_232855


namespace percentage_of_women_employees_l232_232361

variable (E W M : ℝ)

-- Introduce conditions
def total_employees_are_married : Prop := 0.60 * E = (1 / 3) * M + 0.6842 * W
def total_employees_count : Prop := W + M = E
def percentage_of_women : Prop := W = 0.7601 * E

-- State the theorem to prove
theorem percentage_of_women_employees :
  total_employees_are_married E W M ∧ total_employees_count E W M → percentage_of_women E W :=
by sorry

end percentage_of_women_employees_l232_232361


namespace attention_index_proof_l232_232337

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
  if 0 ≤ x ∧ x ≤ 10 then 100 * a ^ (x / 10) - 60
  else if 10 < x ∧ x ≤ 20 then 340
  else if 20 < x ∧ x ≤ 40 then 640 - 15 * x
  else 0

theorem attention_index_proof (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f 5 a = 140) :
  a = 4 ∧ f 5 4 > f 35 4 ∧ (5 ≤ (x : ℝ) ∧ x ≤ 100 / 3 → f x 4 ≥ 140) :=
by
  sorry

end attention_index_proof_l232_232337


namespace simplify_expression_l232_232454

theorem simplify_expression (a c b : ℝ) (h1 : a > c) (h2 : c ≥ 0) (h3 : b > 0) :
  (a * b^2 * (1 / (a + c)^2 + 1 / (a - c)^2) = a - b) → (2 * a * b = a^2 - c^2) :=
by
  sorry

end simplify_expression_l232_232454


namespace cost_equation_l232_232489

variables (x y z : ℝ)

theorem cost_equation (h1 : 2 * x + y + 3 * z = 24) (h2 : 3 * x + 4 * y + 2 * z = 36) : x + y + z = 12 := by
  -- proof steps would go here, but are omitted as per instruction
  sorry

end cost_equation_l232_232489


namespace relationship_between_abc_l232_232271

-- Definitions based on the conditions
def a : ℕ := 3^44
def b : ℕ := 4^33
def c : ℕ := 5^22

-- The theorem to prove the relationship a > b > c
theorem relationship_between_abc : a > b ∧ b > c := by
  sorry

end relationship_between_abc_l232_232271


namespace exceeding_speed_limit_percentages_overall_exceeding_speed_limit_percentage_l232_232126

theorem exceeding_speed_limit_percentages
  (percentage_A : ℕ) (percentage_B : ℕ) (percentage_C : ℕ)
  (H_A : percentage_A = 30)
  (H_B : percentage_B = 20)
  (H_C : percentage_C = 25) :
  percentage_A = 30 ∧ percentage_B = 20 ∧ percentage_C = 25 := by
  sorry

theorem overall_exceeding_speed_limit_percentage
  (percentage_A percentage_B percentage_C : ℕ)
  (H_A : percentage_A = 30)
  (H_B : percentage_B = 20)
  (H_C : percentage_C = 25) :
  (percentage_A + percentage_B + percentage_C) / 3 = 25 := by
  sorry

end exceeding_speed_limit_percentages_overall_exceeding_speed_limit_percentage_l232_232126


namespace find_m_l232_232008

theorem find_m : ∃ m : ℤ, 2^5 - 7 = 3^3 + m ∧ m = -2 :=
by
  use -2
  sorry

end find_m_l232_232008


namespace no_values_less_than_180_l232_232296

/-- Given that w and n are positive integers less than 180 
    such that w % 13 = 2 and n % 8 = 5, 
    prove that there are no such values for w and n. -/
theorem no_values_less_than_180 (w n : ℕ) (hw : w < 180) (hn : n < 180) 
  (h1 : w % 13 = 2) (h2 : n % 8 = 5) : false :=
by
  sorry

end no_values_less_than_180_l232_232296


namespace count_valid_n_decomposition_l232_232679

theorem count_valid_n_decomposition : 
  ∃ (count : ℕ), count = 108 ∧ 
  ∀ (a b c n : ℕ), 
    8 * a + 88 * b + 888 * c = 8000 → 
    0 ≤ b ∧ b ≤ 90 → 
    0 ≤ c ∧ c ≤ 9 → 
    n = a + 2 * b + 3 * c → 
    n < 1000 :=
sorry

end count_valid_n_decomposition_l232_232679


namespace find_monday_temperature_l232_232342

theorem find_monday_temperature
  (M T W Th F : ℤ)
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (h3 : F = 35) :
  M = 43 :=
by
  sorry

end find_monday_temperature_l232_232342


namespace maria_payment_l232_232863

noncomputable def calculate_payment : ℝ :=
  let regular_price := 15
  let first_discount := 0.40 * regular_price
  let after_first_discount := regular_price - first_discount
  let holiday_discount := 0.10 * after_first_discount
  let after_holiday_discount := after_first_discount - holiday_discount
  after_holiday_discount + 2

theorem maria_payment : calculate_payment = 10.10 :=
by
  sorry

end maria_payment_l232_232863


namespace largest_sum_faces_l232_232862

theorem largest_sum_faces (a b c d e f : ℕ)
  (h_ab : a + b ≤ 7) (h_ac : a + c ≤ 7) (h_ad : a + d ≤ 7) (h_ae : a + e ≤ 7) (h_af : a + f ≤ 7)
  (h_bc : b + c ≤ 7) (h_bd : b + d ≤ 7) (h_be : b + e ≤ 7) (h_bf : b + f ≤ 7)
  (h_cd : c + d ≤ 7) (h_ce : c + e ≤ 7) (h_cf : c + f ≤ 7)
  (h_de : d + e ≤ 7) (h_df : d + f ≤ 7)
  (h_ef : e + f ≤ 7) :
  ∃ x y z, 
  ((x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e ∨ x = f) ∧ 
   (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e ∨ y = f) ∧ 
   (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e ∨ z = f)) ∧ 
  (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧
  (x + y ≤ 7) ∧ (y + z ≤ 7) ∧ (x + z ≤ 7) ∧
  (x + y + z = 9) :=
sorry

end largest_sum_faces_l232_232862


namespace FB_length_correct_l232_232209

-- Define a structure for the problem context
structure Triangle (A B C D E F : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] where
  AB : ℝ
  CD : ℝ
  AE : ℝ
  altitude_CD : C -> (A -> B -> Prop)  -- CD is an altitude to AB
  altitude_AE : E -> (B -> C -> Prop)  -- AE is an altitude to BC
  angle_bisector_AF : F -> (B -> C -> Prop)  -- AF is the angle bisector of ∠BAC intersecting BC at F
  intersect_AF_BC_at_F : (F -> B -> Prop)  -- AF intersects BC at F

noncomputable def length_of_FB (A B C D E F : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] 
  (t : Triangle A B C D E F) : ℝ := 
  2  -- From given conditions and conclusion

-- The main theorem to prove
theorem FB_length_correct (A B C D E F : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] 
  (t : Triangle A B C D E F) : 
  t.AB = 8 ∧ t.CD = 3 ∧ t.AE = 4 → length_of_FB A B C D E F t = 2 :=
by
  intro h
  obtain ⟨AB_eq, CD_eq, AE_eq⟩ := h
  sorry

end FB_length_correct_l232_232209


namespace ratio_of_segments_of_hypotenuse_l232_232473

theorem ratio_of_segments_of_hypotenuse (k : Real) :
  let AB := 3 * k
  let BC := 2 * k
  let AC := Real.sqrt (AB^2 + BC^2)
  ∃ D : Real, 
    let BD := (2 / 3) * D
    let AD := (4 / 9) * D
    let CD := D
    ∀ AD CD, AD / CD = 4 / 9 :=
by
  sorry

end ratio_of_segments_of_hypotenuse_l232_232473


namespace find_k_l232_232362

def f (x : ℤ) : ℤ := 3*x^2 - 2*x + 4
def g (x : ℤ) (k : ℤ) : ℤ := x^2 - k * x - 6

theorem find_k : 
  ∃ k : ℤ, f 10 - g 10 k = 10 ∧ k = -18 :=
by 
  sorry

end find_k_l232_232362


namespace find_p_l232_232410

-- Definitions and conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_valid_configuration (p q s r : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime s ∧ is_prime r ∧ 
  1 < p ∧ p < q ∧ q < s ∧ p + q + s = r

-- The theorem statement
theorem find_p (p q s r : ℕ) (h : is_valid_configuration p q s r) : p = 2 :=
by
  sorry

end find_p_l232_232410


namespace total_green_and_yellow_peaches_in_basket_l232_232708

def num_red_peaches := 5
def num_yellow_peaches := 14
def num_green_peaches := 6

theorem total_green_and_yellow_peaches_in_basket :
  num_yellow_peaches + num_green_peaches = 20 :=
by
  sorry

end total_green_and_yellow_peaches_in_basket_l232_232708


namespace dime_probability_l232_232628

theorem dime_probability (dime_value quarter_value : ℝ) (dime_worth quarter_worth total_coins: ℕ) :
  dime_value = 0.10 ∧
  quarter_value = 0.25 ∧
  dime_worth = 10 ∧
  quarter_worth = 4 ∧
  total_coins = 14 →
  (dime_worth / total_coins : ℝ) = 5 / 7 :=
by
  sorry

end dime_probability_l232_232628


namespace unique_solution_l232_232157

variables {x y z : ℝ}

def equation1 (x y z : ℝ) : Prop :=
  (x^2 + x*y + y^2) * (y^2 + y*z + z^2) * (z^2 + z*x + x^2) = x*y*z

def equation2 (x y z : ℝ) : Prop :=
  (x^4 + x^2*y^2 + y^4) * (y^4 + y^2*z^2 + z^4) * (z^4 + z^2*x^2 + x^4) = x^3*y^3*z^3

theorem unique_solution :
  equation1 x y z ∧ equation2 x y z → x = 1/3 ∧ y = 1/3 ∧ z = 1/3 :=
by
  sorry

end unique_solution_l232_232157


namespace pos_diff_is_multiple_of_9_l232_232494

theorem pos_diff_is_multiple_of_9 
  (q r : ℕ) 
  (h_qr : 10 ≤ q ∧ q < 100 ∧ 10 ≤ r ∧ r < 100 ∧ (q % 10) * 10 + (q / 10) = r)
  (h_max_diff : q - r = 63) : 
  ∃ k : ℕ, q - r = 9 * k :=
by
  sorry

end pos_diff_is_multiple_of_9_l232_232494


namespace find_z_value_l232_232667

-- We will define the variables and the given condition
variables {x y z : ℝ}

-- Translate the given condition into Lean
def given_condition (x y z : ℝ) : Prop := (1 / x^2 - 1 / y^2) = (1 / z)

-- State the theorem to prove
theorem find_z_value (x y z : ℝ) (h : given_condition x y z) : 
  z = (x^2 * y^2) / (y^2 - x^2) :=
sorry

end find_z_value_l232_232667


namespace winning_percentage_is_70_l232_232643

def percentage_of_votes (P : ℝ) : Prop :=
  ∃ (P : ℝ), (7 * P - 7 * (100 - P) = 280 ∧ 0 ≤ P ∧ P ≤ 100)

theorem winning_percentage_is_70 :
  percentage_of_votes 70 :=
by
  sorry

end winning_percentage_is_70_l232_232643


namespace integer_x_cubed_prime_l232_232150

theorem integer_x_cubed_prime (x : ℕ) : 
  (∃ p : ℕ, Prime p ∧ (2^x + x^2 + 25 = p^3)) → x = 6 :=
by
  sorry

end integer_x_cubed_prime_l232_232150


namespace flower_shop_sold_bouquets_l232_232639

theorem flower_shop_sold_bouquets (roses_per_bouquet : ℕ) (daisies_per_bouquet : ℕ) 
  (rose_bouquets_sold : ℕ) (daisy_bouquets_sold : ℕ) (total_flowers_sold : ℕ)
  (h1 : roses_per_bouquet = 12) (h2 : rose_bouquets_sold = 10) 
  (h3 : daisy_bouquets_sold = 10) (h4 : total_flowers_sold = 190) : 
  (rose_bouquets_sold + daisy_bouquets_sold) = 20 :=
by sorry

end flower_shop_sold_bouquets_l232_232639


namespace chess_tournament_games_l232_232339

def games_played (n : ℕ) : ℕ := n * (n - 1) / 2

theorem chess_tournament_games (n : ℕ) (h : n = 19) : games_played n = 171 :=
by
  rw [h]
  sorry

end chess_tournament_games_l232_232339


namespace disjunction_of_p_and_q_l232_232056

-- Define the propositions p and q
variable (p q : Prop)

-- Assume that p is true and q is false
theorem disjunction_of_p_and_q (h1 : p) (h2 : ¬q) : p ∨ q := 
sorry

end disjunction_of_p_and_q_l232_232056


namespace intersection_eq_l232_232444

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | x^2 - x ≤ 0}

theorem intersection_eq : A ∩ B = {0, 1} := by
  sorry

end intersection_eq_l232_232444


namespace regular_polygon_sides_160_l232_232052

theorem regular_polygon_sides_160 (n : ℕ) 
  (h1 : n ≥ 3) 
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ n → (interior_angle : ℝ) = 160) : 
  n = 18 :=
by
  sorry

end regular_polygon_sides_160_l232_232052


namespace athletes_meeting_time_and_overtakes_l232_232166

-- Define the constants for the problem
noncomputable def track_length : ℕ := 400
noncomputable def speed1 : ℕ := 155
noncomputable def speed2 : ℕ := 200
noncomputable def speed3 : ℕ := 275

-- The main theorem for the problem statement
theorem athletes_meeting_time_and_overtakes :
  ∃ (t : ℚ) (n_overtakes : ℕ), 
  (t = 80 / 3) ∧
  (n_overtakes = 13) ∧
  (∀ n : ℕ, n * (track_length / 45) = t) ∧
  (∀ k : ℕ, k * (track_length / 120) = t) ∧
  (∀ m : ℕ, m * (track_length / 75) = t) := 
sorry

end athletes_meeting_time_and_overtakes_l232_232166


namespace max_n_for_factoring_l232_232740

theorem max_n_for_factoring (n : ℤ) :
  (∃ A B : ℤ, (5 * B + A = n) ∧ (A * B = 90)) → n = 451 :=
by
  sorry

end max_n_for_factoring_l232_232740


namespace length_of_other_side_l232_232390

-- Defining the conditions
def roofs := 3
def sides_per_roof := 2
def length_of_one_side := 40 -- measured in feet
def shingles_per_square_foot := 8
def total_shingles := 38400

-- The proof statement
theorem length_of_other_side : 
    ∃ (L : ℕ), (total_shingles / shingles_per_square_foot / roofs / sides_per_roof = 40 * L) ∧ L = 20 :=
by
  sorry

end length_of_other_side_l232_232390


namespace not_car_probability_l232_232462

-- Defining the probabilities of taking different modes of transportation.
def P_train : ℝ := 0.5
def P_car : ℝ := 0.2
def P_plane : ℝ := 0.3

-- Defining the event that these probabilities are for mutually exclusive events
axiom mutually_exclusive_events : P_train + P_car + P_plane = 1

-- Statement of the theorem to prove
theorem not_car_probability : P_train + P_plane = 0.8 := 
by 
  -- Use the definitions and axiom provided
  sorry

end not_car_probability_l232_232462


namespace monthly_profit_10000_daily_profit_15000_maximize_profit_l232_232368

noncomputable def price_increase (c p: ℕ) (x: ℕ) : ℕ := c + x - p
noncomputable def sales_volume (s d: ℕ) (x: ℕ) : ℕ := s - d * x
noncomputable def monthly_profit (price cost volume: ℕ) : ℕ := (price - cost) * volume
noncomputable def monthly_profit_equation (x: ℕ) : ℕ := (40 + x - 30) * (600 - 10 * x)

theorem monthly_profit_10000 (x: ℕ) : monthly_profit_equation x = 10000 ↔ x = 10 ∨ x = 40 :=
by sorry

theorem daily_profit_15000 (x: ℕ) : ¬∃ x, monthly_profit_equation x = 15000 :=
by sorry

theorem maximize_profit (x p y: ℕ) : (∀ x, monthly_profit (40 + x) 30 (600 - 10 * x) ≤ y) ∧ y = 12250 ∧ x = 65 :=
by sorry

end monthly_profit_10000_daily_profit_15000_maximize_profit_l232_232368


namespace inequality_proof_l232_232136

theorem inequality_proof (a b c d : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : d > 0)
    (h_cond : 2 * (a + b + c + d) ≥ a * b * c * d) : (a^2 + b^2 + c^2 + d^2) ≥ (a * b * c * d) :=
by
  sorry

end inequality_proof_l232_232136


namespace right_triangle_area_l232_232191

theorem right_triangle_area (a b c : ℝ) (h₁ : a = 24) (h₂ : c = 26) (h₃ : a^2 + b^2 = c^2) : 
  (1 / 2) * a * b = 120 :=
by
  sorry

end right_triangle_area_l232_232191


namespace marked_price_l232_232590

theorem marked_price (initial_price : ℝ) (discount_percent : ℝ) (profit_margin_percent : ℝ) (final_discount_percent : ℝ) (marked_price : ℝ) :
  initial_price = 40 → 
  discount_percent = 0.25 → 
  profit_margin_percent = 0.50 → 
  final_discount_percent = 0.10 → 
  marked_price = 50 := by
  sorry

end marked_price_l232_232590


namespace solve_inequality_l232_232034

theorem solve_inequality (x : ℝ) (h : 3 * x + 4 ≠ 0) :
  3 - 1 / (3 * x + 4) < 5 ↔ -3 / 2 < x :=
by
  sorry

end solve_inequality_l232_232034


namespace sum_of_squares_and_product_l232_232396

open Real

theorem sum_of_squares_and_product (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h1 : x^2 + y^2 = 325) (h2 : x * y = 120) :
    x + y = Real.sqrt 565 := by
  sorry

end sum_of_squares_and_product_l232_232396


namespace percentage_difference_l232_232967

theorem percentage_difference (x y : ℝ) (h : x = 3 * y) : ((x - y) / x) * 100 = 66.67 :=
by
  sorry

end percentage_difference_l232_232967


namespace circle_radius_of_diameter_l232_232577

theorem circle_radius_of_diameter (d : ℝ) (h : d = 22) : d / 2 = 11 :=
by
  sorry

end circle_radius_of_diameter_l232_232577


namespace correct_equation_l232_232973

variable (x : ℤ)
variable (cost_of_chickens : ℤ)

-- Condition 1: If each person contributes 9 coins, there will be an excess of 11 coins.
def condition1 : Prop := 9 * x - cost_of_chickens = 11

-- Condition 2: If each person contributes 6 coins, there will be a shortage of 16 coins.
def condition2 : Prop := 6 * x - cost_of_chickens = -16

-- The goal is to prove the correct equation given the conditions.
theorem correct_equation (h1 : condition1 (x) (cost_of_chickens)) (h2 : condition2 (x) (cost_of_chickens)) :
  9 * x - 11 = 6 * x + 16 :=
sorry

end correct_equation_l232_232973


namespace product_of_factors_l232_232007

theorem product_of_factors : (2.1 * (53.2 - 0.2) = 111.3) := by
  sorry

end product_of_factors_l232_232007


namespace change_received_l232_232385

def cost_per_banana_cents : ℕ := 30
def cost_per_banana_dollars : ℝ := 0.30
def number_of_bananas : ℕ := 5
def total_paid_dollars : ℝ := 10.00

def total_cost (cost_per_banana_dollars : ℝ) (number_of_bananas : ℕ) : ℝ :=
  cost_per_banana_dollars * number_of_bananas

theorem change_received :
  total_paid_dollars - total_cost cost_per_banana_dollars number_of_bananas = 8.50 :=
by
  sorry

end change_received_l232_232385


namespace post_tax_income_correct_l232_232130

noncomputable def worker_a_pre_tax_income : ℝ :=
  80 * 30 + 50 * 30 * 1.20 + 35 * 30 * 1.50 + (35 * 30 * 1.50) * 0.05

noncomputable def worker_b_pre_tax_income : ℝ :=
  90 * 25 + 45 * 25 * 1.25 + 40 * 25 * 1.45 + (40 * 25 * 1.45) * 0.05

noncomputable def worker_c_pre_tax_income : ℝ :=
  70 * 35 + 40 * 35 * 1.15 + 60 * 35 * 1.60 + (60 * 35 * 1.60) * 0.05

noncomputable def worker_a_post_tax_income : ℝ := 
  worker_a_pre_tax_income * 0.85 - 200

noncomputable def worker_b_post_tax_income : ℝ := 
  worker_b_pre_tax_income * 0.82 - 250

noncomputable def worker_c_post_tax_income : ℝ := 
  worker_c_pre_tax_income * 0.80 - 300

theorem post_tax_income_correct :
  worker_a_post_tax_income = 4775.69 ∧ 
  worker_b_post_tax_income = 3996.57 ∧ 
  worker_c_post_tax_income = 5770.40 :=
by {
  sorry
}

end post_tax_income_correct_l232_232130


namespace cakes_served_during_lunch_l232_232804

theorem cakes_served_during_lunch (T D L : ℕ) (h1 : T = 15) (h2 : D = 9) : L = T - D → L = 6 :=
by
  intros h
  rw [h1, h2] at h
  exact h

end cakes_served_during_lunch_l232_232804


namespace total_space_after_compaction_correct_l232_232566

noncomputable def problem : Prop :=
  let num_small_cans := 50
  let num_large_cans := 50
  let small_can_size := 20
  let large_can_size := 40
  let small_can_compaction := 0.30
  let large_can_compaction := 0.40
  let small_cans_compacted := num_small_cans * small_can_size * small_can_compaction
  let large_cans_compacted := num_large_cans * large_can_size * large_can_compaction
  let total_space_after_compaction := small_cans_compacted + large_cans_compacted
  total_space_after_compaction = 1100

theorem total_space_after_compaction_correct :
  problem :=
  by
    unfold problem
    sorry

end total_space_after_compaction_correct_l232_232566


namespace pen_price_first_day_l232_232912

theorem pen_price_first_day (x y : ℕ) 
  (h1 : x * y = (x - 1) * (y + 100)) 
  (h2 : x * y = (x + 2) * (y - 100)) : x = 4 :=
by
  sorry

end pen_price_first_day_l232_232912


namespace distance_on_map_is_correct_l232_232690

-- Define the parameters
def time_hours : ℝ := 1.5
def speed_mph : ℝ := 60
def map_scale_inches_per_mile : ℝ := 0.05555555555555555

-- Define the computation of actual distance and distance on the map
def actual_distance_miles : ℝ := speed_mph * time_hours
def distance_on_map_inches : ℝ := actual_distance_miles * map_scale_inches_per_mile

-- Theorem statement
theorem distance_on_map_is_correct :
  distance_on_map_inches = 5 :=
by 
  sorry

end distance_on_map_is_correct_l232_232690


namespace line_intersects_y_axis_at_0_6_l232_232286

theorem line_intersects_y_axis_at_0_6 : ∃ y : ℝ, 4 * y + 3 * (0 : ℝ) = 24 ∧ (0, y) = (0, 6) :=
by
  use 6
  simp
  sorry

end line_intersects_y_axis_at_0_6_l232_232286


namespace probability_kwoes_non_intersect_breads_l232_232300

-- Define the total number of ways to pick 3 points from 7
def total_combinations : ℕ := Nat.choose 7 3

-- Define the number of ways to pick 3 consecutive points from 7
def favorable_combinations : ℕ := 7

-- Define the probability of non-intersection
def non_intersection_probability : ℚ := favorable_combinations / total_combinations

-- Assert the final required probability
theorem probability_kwoes_non_intersect_breads :
  non_intersection_probability = 1 / 5 :=
by
  sorry

end probability_kwoes_non_intersect_breads_l232_232300


namespace complement_M_in_U_l232_232162

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x | ∃ y : ℝ, y = Real.sqrt (1 - x)}

-- State the theorem to prove that the complement of M in U is (1, +∞)
theorem complement_M_in_U :
  (U \ M) = {x | 1 < x} :=
by
  sorry

end complement_M_in_U_l232_232162


namespace true_discount_is_36_l232_232055

noncomputable def calc_true_discount (BD SD : ℝ) : ℝ := BD / (1 + BD / SD)

theorem true_discount_is_36 :
  let BD := 42
  let SD := 252
  calc_true_discount BD SD = 36 := 
by
  -- proof here
  sorry

end true_discount_is_36_l232_232055


namespace least_value_expression_l232_232829

theorem least_value_expression (x y : ℝ) : 
  (x^2 * y + x * y^2 - 1)^2 + (x + y)^2 ≥ 1 :=
sorry

end least_value_expression_l232_232829


namespace packing_heights_difference_l232_232707

-- Definitions based on conditions
def diameter := 8   -- Each pipe has a diameter of 8 cm
def num_pipes := 160 -- Each crate contains 160 pipes

-- Heights of the crates based on the given packing methods
def height_crate_A := 128 -- Calculated height for Crate A

noncomputable def height_crate_B := 8 + 60 * Real.sqrt 3 -- Calculated height for Crate B

-- Positive difference in the total heights of the two packings
noncomputable def delta_height := height_crate_A - height_crate_B

-- The goal to prove
theorem packing_heights_difference :
  delta_height = 120 - 60 * Real.sqrt 3 :=
sorry

end packing_heights_difference_l232_232707


namespace claire_photos_l232_232152

theorem claire_photos (C : ℕ) (h1 : 3 * C = C + 20) : C = 10 :=
sorry

end claire_photos_l232_232152


namespace num_pairs_satisfying_eq_l232_232159

theorem num_pairs_satisfying_eq :
  ∃ n : ℕ, (n = 256) ∧ (∀ x y : ℤ, x^2 + x * y = 30000000 → true) :=
sorry

end num_pairs_satisfying_eq_l232_232159


namespace rahul_share_is_100_l232_232914

-- Definitions of the conditions
def rahul_rate := 1/3
def rajesh_rate := 1/2
def total_payment := 250

-- Definition of their work rate when they work together
def combined_rate := rahul_rate + rajesh_rate

-- Definition of the total value of the work done in one day when both work together
noncomputable def combined_work_value := total_payment / combined_rate

-- Definition of Rahul's share for the work done in one day
noncomputable def rahul_share := rahul_rate * combined_work_value

-- The theorem we need to prove
theorem rahul_share_is_100 : rahul_share = 100 := by
  sorry

end rahul_share_is_100_l232_232914


namespace eval_floor_neg_sqrt_l232_232199

theorem eval_floor_neg_sqrt : (Int.floor (-Real.sqrt (64 / 9)) = -3) := sorry

end eval_floor_neg_sqrt_l232_232199


namespace hyperbola_real_axis_length_l232_232649

theorem hyperbola_real_axis_length (x y : ℝ) :
  x^2 - y^2 / 9 = 1 → 2 = 2 :=
by
  sorry

end hyperbola_real_axis_length_l232_232649


namespace johns_out_of_pocket_expense_l232_232363

-- Define the conditions given in the problem
def old_system_cost : ℤ := 250
def old_system_trade_in_value : ℤ := (80 * old_system_cost) / 100
def new_system_initial_cost : ℤ := 600
def new_system_discount : ℤ := (25 * new_system_initial_cost) / 100
def new_system_final_cost : ℤ := new_system_initial_cost - new_system_discount

-- Define the amount of money that came out of John's pocket
def out_of_pocket_expense : ℤ := new_system_final_cost - old_system_trade_in_value

-- State the theorem that needs to be proven
theorem johns_out_of_pocket_expense : out_of_pocket_expense = 250 := by
  sorry

end johns_out_of_pocket_expense_l232_232363


namespace no_carry_consecutive_pairs_l232_232693

/-- Consider the range of integers {2000, 2001, ..., 3000}. 
    We determine that the number of pairs of consecutive integers in this range such that their addition requires no carrying is 729. -/
theorem no_carry_consecutive_pairs : 
  ∀ (n : ℕ), (2000 ≤ n ∧ n < 3000) ∧ ((n + 1) ≤ 3000) → 
  ∃ (count : ℕ), count = 729 := 
sorry

end no_carry_consecutive_pairs_l232_232693


namespace common_ratio_l232_232497

theorem common_ratio (a1 a2 a3 : ℚ) (S3 q : ℚ)
  (h1 : a3 = 3 / 2)
  (h2 : S3 = 9 / 2)
  (h3 : a1 + a2 + a3 = S3)
  (h4 : a1 = a3 / q^2)
  (h5 : a2 = a3 / q):
  q = 1 ∨ q = -1/2 :=
by sorry

end common_ratio_l232_232497


namespace alex_original_seat_l232_232579

-- We define a type for seats
inductive Seat where
  | s1 | s2 | s3 | s4 | s5 | s6
  deriving DecidableEq, Inhabited

open Seat

-- Define the initial conditions and movements
def initial_seats : (Fin 6 → Seat) := ![s1, s2, s3, s4, s5, s6]

def move_bella (s : Seat) : Seat :=
  match s with
  | s1 => s2
  | s2 => s3
  | s3 => s4
  | s4 => s5
  | s5 => s6
  | s6 => s1  -- invalid movement for the problem context, can handle separately

def move_coral (s : Seat) : Seat :=
  match s with
  | s1 => s6  -- two seats left from s1 wraps around to s6
  | s2 => s1
  | s3 => s2
  | s4 => s3
  | s5 => s4
  | s6 => s5

-- Dan and Eve switch seats among themselves
def switch_dan_eve (s : Seat) : Seat :=
  match s with
  | s3 => s4
  | s4 => s3
  | _ => s  -- all other positions remain the same

def move_finn (s : Seat) : Seat :=
  match s with
  | s1 => s2
  | s2 => s3
  | s3 => s4
  | s4 => s5
  | s5 => s6
  | s6 => s1  -- invalid movement for the problem context, can handle separately

-- Define the final seat for Alex
def alex_final_seat : Seat := s6  -- Alex returns to one end seat

-- Define a theorem for the proof of Alex's original seat being Seat.s1
theorem alex_original_seat :
  ∃ (original_seat : Seat), original_seat = s1 :=
  sorry

end alex_original_seat_l232_232579


namespace min_value_proof_l232_232469

noncomputable def min_value (x y : ℝ) : ℝ := 1 / x + 1 / (2 * y)

theorem min_value_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + 2 * y = 1) :
  min_value x y = 4 :=
sorry

end min_value_proof_l232_232469


namespace min_weights_needed_l232_232475

theorem min_weights_needed :
  ∃ (weights : List ℕ), (∀ m : ℕ, 1 ≤ m ∧ m ≤ 100 → ∃ (left right : List ℕ), m = (left.sum - right.sum)) ∧ weights.length = 5 :=
sorry

end min_weights_needed_l232_232475


namespace cost_of_rice_l232_232894

-- Define the cost variables
variables (E R K : ℝ)

-- State the conditions as assumptions
def conditions (E R K : ℝ) : Prop :=
  (E = R) ∧
  (K = (2 / 3) * E) ∧
  (2 * K = 48)

-- State the theorem to be proven
theorem cost_of_rice (E R K : ℝ) (h : conditions E R K) : R = 36 :=
by
  sorry

end cost_of_rice_l232_232894


namespace martha_butterflies_total_l232_232358

theorem martha_butterflies_total
  (B : ℕ) (Y : ℕ) (black : ℕ)
  (h1 : B = 4)
  (h2 : Y = B / 2)
  (h3 : black = 5) :
  B + Y + black = 11 :=
by {
  -- skip proof 
  sorry 
}

end martha_butterflies_total_l232_232358


namespace roses_left_unsold_l232_232004

def price_per_rose : ℕ := 4
def initial_roses : ℕ := 13
def total_earned : ℕ := 36

theorem roses_left_unsold : (initial_roses - (total_earned / price_per_rose) = 4) :=
by
  sorry

end roses_left_unsold_l232_232004


namespace repeating_decimal_to_fraction_l232_232844

noncomputable def repeating_decimal_solution : ℚ := 7311 / 999

theorem repeating_decimal_to_fraction (x : ℚ) (h : x = 7 + 318 / 999) : x = repeating_decimal_solution := 
by
  sorry

end repeating_decimal_to_fraction_l232_232844


namespace complex_division_l232_232889

-- Define the imaginary unit 'i'
def i := Complex.I

-- Define the complex numbers as described in the problem
def num := Complex.mk 3 (-1)
def denom := Complex.mk 1 (-1)
def expected := Complex.mk 2 1

-- State the theorem to prove that the complex division is as expected
theorem complex_division : (num / denom) = expected :=
by
  sorry

end complex_division_l232_232889


namespace sum_of_roots_eq_zero_l232_232843

theorem sum_of_roots_eq_zero :
  ∀ (x : ℝ), x^2 - 7 * |x| + 6 = 0 → (∃ a b c d : ℝ, a + b + c + d = 0) :=
by
  sorry

end sum_of_roots_eq_zero_l232_232843


namespace diamond_evaluation_l232_232049

def diamond (a b : ℝ) : ℝ := (a + b) * (a - b)

theorem diamond_evaluation : diamond 2 (diamond 3 (diamond 1 4)) = -46652 :=
  by
  sorry

end diamond_evaluation_l232_232049


namespace factor_expression_l232_232353

theorem factor_expression (x : ℝ) : 
  5 * x * (x + 2) + 9 * (x + 2) = (x + 2) * (5 * x + 9) :=
by sorry

end factor_expression_l232_232353


namespace mean_value_z_l232_232574

theorem mean_value_z (z : ℚ) (h : (7 + 10 + 23) / 3 = (18 + z) / 2) : z = 26 / 3 :=
by
  sorry

end mean_value_z_l232_232574


namespace measure_angle_WYZ_l232_232884

def angle_XYZ : ℝ := 45
def angle_XYW : ℝ := 15

theorem measure_angle_WYZ : angle_XYZ - angle_XYW = 30 := by
  sorry

end measure_angle_WYZ_l232_232884


namespace solve_for_x_l232_232771

theorem solve_for_x (x y : ℤ) (h1 : x + y = 14) (h2 : x - y = 60) : x = 37 := by
  sorry

end solve_for_x_l232_232771


namespace total_turtles_in_lake_l232_232825

theorem total_turtles_in_lake
  (female_percent : ℝ) (male_with_stripes_fraction : ℝ) 
  (babies_with_stripes : ℝ) (adults_percentage : ℝ) : 
  female_percent = 0.6 → 
  male_with_stripes_fraction = 1/4 →
  babies_with_stripes = 4 →
  adults_percentage = 0.6 →
  ∃ (total_turtles : ℕ), total_turtles = 100 :=
  by
  -- Step-by-step proof to be filled here
  sorry

end total_turtles_in_lake_l232_232825


namespace clock_hand_speed_ratio_l232_232980

theorem clock_hand_speed_ratio :
  (360 / 720 : ℝ) / (360 / 60 : ℝ) = (2 / 24 : ℝ) := by
    sorry

end clock_hand_speed_ratio_l232_232980


namespace part1_part2_part3_l232_232356

-- Definition of the function
def linear_function (m : ℝ) (x : ℝ) : ℝ :=
  (2 * m + 1) * x + m - 3

-- Part 1: If the graph passes through the origin
theorem part1 (h : linear_function m 0 = 0) : m = 3 :=
by {
  sorry
}

-- Part 2: If the graph is parallel to y = 3x - 3
theorem part2 (h : ∀ x, linear_function m x = 3 * x - 3 → 2 * m + 1 = 3) : m = 1 :=
by {
  sorry
}

-- Part 3: If the graph intersects the y-axis below the x-axis
theorem part3 (h_slope : 2 * m + 1 ≠ 0) (h_intercept : m - 3 < 0) : m < 3 ∧ m ≠ -1 / 2 :=
by {
  sorry
}

end part1_part2_part3_l232_232356


namespace domain_of_f_l232_232561

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - x)

theorem domain_of_f :
  {x : ℝ | f x = Real.log (x^2 - x)} = {x : ℝ | x < 0 ∨ x > 1} :=
sorry

end domain_of_f_l232_232561


namespace solve_for_x_l232_232528

noncomputable def x : ℚ := 45^2 / (7 - (3 / 4))

theorem solve_for_x : x = 324 := by
  sorry

end solve_for_x_l232_232528


namespace find_solutions_l232_232093

theorem find_solutions (x y : Real) :
    (x = 1 ∧ y = 2) ∨
    (x = 1 ∧ y = 0) ∨
    (x = -4 ∧ y = 6) ∨
    (x = -5 ∧ y = 2) ∨
    (x = -3 ∧ y = 0) ↔
    x^2 + x*y + y^2 + 2*x - 3*y - 3 = 0 := by
  sorry

end find_solutions_l232_232093


namespace surface_area_cone_first_octant_surface_area_sphere_inside_cylinder_surface_area_cylinder_inside_sphere_l232_232367

-- First Problem:
theorem surface_area_cone_first_octant :
  ∃ (surface_area : ℝ), 
    (∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 4 ∧ z^2 = 2*x*y) → surface_area = 16 :=
sorry

-- Second Problem:
theorem surface_area_sphere_inside_cylinder (R : ℝ) :
  ∃ (surface_area : ℝ), 
    (∀ (x y z : ℝ), x^2 + y^2 + z^2 = R^2 ∧ x^2 + y^2 = R*x) → surface_area = 2 * R^2 * (π - 2) :=
sorry

-- Third Problem:
theorem surface_area_cylinder_inside_sphere (R : ℝ) :
  ∃ (surface_area : ℝ), 
    (∀ (x y z : ℝ), x^2 + y^2 = R*x ∧ x^2 + y^2 + z^2 = R^2) → surface_area = 4 * R^2 :=
sorry

end surface_area_cone_first_octant_surface_area_sphere_inside_cylinder_surface_area_cylinder_inside_sphere_l232_232367


namespace diameter_of_circumscribed_circle_l232_232951

noncomputable def circumscribed_circle_diameter (a : ℝ) (A : ℝ) : ℝ :=
  a / Real.sin A

theorem diameter_of_circumscribed_circle :
  circumscribed_circle_diameter 15 (Real.pi / 4) = 15 * Real.sqrt 2 :=
by
  sorry

end diameter_of_circumscribed_circle_l232_232951


namespace find_f_of_half_l232_232084

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_of_half : (∀ x : ℝ, f (Real.logb 4 x) = x) → f (1 / 2) = 2 :=
by
  intros h
  have h1 := h (4 ^ (1 / 2))
  sorry

end find_f_of_half_l232_232084


namespace simplify_333_div_9999_mul_99_l232_232943

theorem simplify_333_div_9999_mul_99 :
  (333 / 9999) * 99 = 37 / 101 :=
by
  -- Sorry for skipping proof
  sorry

end simplify_333_div_9999_mul_99_l232_232943


namespace rigid_motion_pattern_l232_232660

-- Define the types of transformations
inductive Transformation
| rotation : ℝ → Transformation -- rotation by an angle
| translation : ℝ → Transformation -- translation by a distance
| reflection_across_m : Transformation -- reflection across line m
| reflection_perpendicular_to_m : ℝ → Transformation -- reflective across line perpendicular to m at a point

-- Define the problem statement conditions
def pattern_alternates (line_m : ℝ → ℝ) : Prop := sorry -- This should define the alternating pattern of equilateral triangles and squares along line m

-- Problem statement in Lean
theorem rigid_motion_pattern (line_m : ℝ → ℝ) (p : Transformation → Prop)
    (h1 : p (Transformation.rotation 180)) -- 180-degree rotation is a valid transformation for the pattern
    (h2 : ∀ d, p (Transformation.translation d)) -- any translation by pattern unit length is a valid transformation
    (h3 : p Transformation.reflection_across_m) -- reflection across line m is a valid transformation
    (h4 : ∀ x, p (Transformation.reflection_perpendicular_to_m x)) -- reflection across any perpendicular line is a valid transformation
    : ∃ t : Finset Transformation, t.card = 4 ∧ ∀ t_val, t_val ∈ t → p t_val ∧ t_val ≠ Transformation.rotation 0 := 
sorry

end rigid_motion_pattern_l232_232660


namespace ball_hits_ground_l232_232795

theorem ball_hits_ground (t : ℝ) :
  (∃ t, -(16 * t^2) + 32 * t + 30 = 0 ∧ t = 1 + (Real.sqrt 46) / 4) :=
sorry

end ball_hits_ground_l232_232795


namespace hens_count_l232_232181

theorem hens_count (H C : ℕ) (h1 : H + C = 48) (h2 : 2 * H + 4 * C = 136) : H = 28 := by
  sorry

end hens_count_l232_232181


namespace greatest_possible_x_for_equation_l232_232716

theorem greatest_possible_x_for_equation :
  ∃ x, (x = (9 : ℝ) / 5) ∧ 
  ((5 * x - 20) / (4 * x - 5))^2 + ((5 * x - 20) / (4 * x - 5)) = 20 := by
  sorry

end greatest_possible_x_for_equation_l232_232716


namespace percentage_republicans_vote_X_l232_232935

theorem percentage_republicans_vote_X (R : ℝ) (P_R : ℝ) :
  (3 * R * P_R + 2 * R * 0.15) - (3 * R * (1 - P_R) + 2 * R * 0.85) = 0.019999999999999927 * (3 * R + 2 * R) →
  P_R = 4.1 / 6 :=
by
  intro h
  sorry

end percentage_republicans_vote_X_l232_232935


namespace range_of_m_l232_232744

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 1 then 2^x + 1 else 1 - Real.log (x) / Real.log 2

-- The problem is to find the range of m such that f(1 - m^2) > f(2m - 2). We assert the range of m as given in the correct answer.
theorem range_of_m : {m : ℝ | f (1 - m^2) > f (2 * m - 2)} = 
  {m : ℝ | -3 < m ∧ m < 1} ∪ {m : ℝ | m > 3 / 2} :=
sorry

end range_of_m_l232_232744


namespace dave_trips_l232_232391

/-- Dave can only carry 9 trays at a time. -/
def trays_per_trip := 9

/-- Number of trays Dave has to pick up from one table. -/
def trays_from_table1 := 17

/-- Number of trays Dave has to pick up from another table. -/
def trays_from_table2 := 55

/-- Total number of trays Dave has to pick up. -/
def total_trays := trays_from_table1 + trays_from_table2

/-- The number of trips Dave will make. -/
def number_of_trips := total_trays / trays_per_trip

theorem dave_trips :
  number_of_trips = 8 :=
sorry

end dave_trips_l232_232391


namespace rightmost_three_digits_of_3_pow_1987_l232_232335

theorem rightmost_three_digits_of_3_pow_1987 :
  3^1987 % 2000 = 187 :=
by sorry

end rightmost_three_digits_of_3_pow_1987_l232_232335


namespace necessity_of_A_for_B_l232_232460

variables {a b h : ℝ}

def PropA (a b h : ℝ) : Prop := |a - b| < 2 * h
def PropB (a b h : ℝ) : Prop := |a - 1| < h ∧ |b - 1| < h

theorem necessity_of_A_for_B (h_pos : 0 < h) : 
  (∀ a b, PropB a b h → PropA a b h) ∧ ¬ (∀ a b, PropA a b h → PropB a b h) :=
by sorry

end necessity_of_A_for_B_l232_232460


namespace max_value_sum_seq_l232_232402

theorem max_value_sum_seq : 
  ∃ a1 a2 a3 a4 : ℝ, 
    a1 = 0 ∧ 
    |a2| = |a1 - 1| ∧ 
    |a3| = |a2 - 1| ∧ 
    |a4| = |a3 - 1| ∧ 
    a1 + a2 + a3 + a4 = 2 := 
by 
  sorry

end max_value_sum_seq_l232_232402


namespace mobot_coloring_six_colorings_l232_232264

theorem mobot_coloring_six_colorings (n m : ℕ) (h : n ≥ 3 ∧ m ≥ 3) :
  (∃ mobot, mobot = (1, 1)) ↔ (∃ colorings : ℕ, colorings = 6) :=
sorry

end mobot_coloring_six_colorings_l232_232264


namespace min_cost_per_student_is_80_l232_232218

def num_students : ℕ := 48
def swims_per_student : ℕ := 8
def cost_per_card : ℕ := 240
def cost_per_bus : ℕ := 40

def total_swims : ℕ := num_students * swims_per_student

def min_cost_per_student : ℕ :=
  let n := 8
  let c := total_swims / n
  let total_cost := cost_per_card * n + cost_per_bus * c
  total_cost / num_students

theorem min_cost_per_student_is_80 :
  min_cost_per_student = 80 :=
sorry

end min_cost_per_student_is_80_l232_232218


namespace students_in_class_l232_232790

theorem students_in_class (n m f r u : ℕ) (cond1 : 20 < n ∧ n < 30)
  (cond2 : f = 2 * m) (cond3 : n = m + f)
  (cond4 : r = 3 * u - 1) (cond5 : r + u = n) :
  n = 27 :=
sorry

end students_in_class_l232_232790


namespace more_than_half_remains_l232_232659

def cubic_block := { n : ℕ // n > 0 }

noncomputable def total_cubes (b : cubic_block) : ℕ := b.val ^ 3

noncomputable def outer_layer_cubes (b : cubic_block) : ℕ := 6 * (b.val ^ 2) - 12 * b.val + 8

noncomputable def remaining_cubes (b : cubic_block) : ℕ := total_cubes b - outer_layer_cubes b

theorem more_than_half_remains (b : cubic_block) (h : b.val = 10) : remaining_cubes b > total_cubes b / 2 :=
by
  sorry

end more_than_half_remains_l232_232659


namespace opposite_number_of_2_eq_neg2_abs_val_eq_2_iff_eq_2_or_neg2_l232_232326

theorem opposite_number_of_2_eq_neg2 : -2 = -2 := by
  sorry

theorem abs_val_eq_2_iff_eq_2_or_neg2 (x : ℝ) : abs x = 2 ↔ x = 2 ∨ x = -2 := by
  sorry

end opposite_number_of_2_eq_neg2_abs_val_eq_2_iff_eq_2_or_neg2_l232_232326


namespace max_value_of_linear_function_l232_232192

theorem max_value_of_linear_function :
  ∀ (x : ℝ), -3 ≤ x ∧ x ≤ 3 → y = 5 / 3 * x + 2 → ∃ (y_max : ℝ), y_max = 7 ∧ ∀ (x' : ℝ), -3 ≤ x' ∧ x' ≤ 3 → 5 / 3 * x' + 2 ≤ y_max :=
by
  intro x interval_x function_y
  sorry

end max_value_of_linear_function_l232_232192


namespace sin_alpha_cos_alpha_l232_232009

theorem sin_alpha_cos_alpha (α : ℝ) (h : Real.sin (3 * Real.pi - α) = -2 * Real.sin (Real.pi / 2 + α)) : 
  Real.sin α * Real.cos α = -2 / 5 :=
by
  sorry

end sin_alpha_cos_alpha_l232_232009


namespace amount_of_cocoa_powder_given_by_mayor_l232_232601

def total_cocoa_powder_needed : ℕ := 306
def cocoa_powder_still_needed : ℕ := 47

def cocoa_powder_given_by_mayor : ℕ :=
  total_cocoa_powder_needed - cocoa_powder_still_needed

theorem amount_of_cocoa_powder_given_by_mayor :
  cocoa_powder_given_by_mayor = 259 := by
  sorry

end amount_of_cocoa_powder_given_by_mayor_l232_232601


namespace emily_irises_after_addition_l232_232550

theorem emily_irises_after_addition
  (initial_roses : ℕ)
  (added_roses : ℕ)
  (ratio_irises_roses : ℕ)
  (ratio_roses_irises : ℕ)
  (h_ratio : ratio_irises_roses = 3 ∧ ratio_roses_irises = 7)
  (h_initial_roses : initial_roses = 35)
  (h_added_roses : added_roses = 30) :
  ∃ irises_after_addition : ℕ, irises_after_addition = 27 :=
  by
    sorry

end emily_irises_after_addition_l232_232550


namespace smallest_positive_integer_l232_232799

theorem smallest_positive_integer (n : ℕ) (h₁ : n > 1) (h₂ : n % 2 = 1) (h₃ : n % 3 = 1) (h₄ : n % 4 = 1) (h₅ : n % 5 = 1) : n = 61 :=
by
  sorry

end smallest_positive_integer_l232_232799


namespace bowling_ball_weight_l232_232428

-- Define the weights of the bowling balls and canoes
variables (b c : ℝ)

-- Conditions provided by the problem statement
axiom eq1 : 8 * b = 4 * c
axiom eq2 : 3 * c = 108

-- Prove that one bowling ball weighs 18 pounds
theorem bowling_ball_weight : b = 18 :=
by
  sorry

end bowling_ball_weight_l232_232428


namespace race_distance_l232_232314

variable (speed_cristina speed_nicky head_start time_nicky : ℝ)

theorem race_distance
  (h1 : speed_cristina = 5)
  (h2 : speed_nicky = 3)
  (h3 : head_start = 12)
  (h4 : time_nicky = 30) :
  let time_cristina := time_nicky - head_start
  let distance_nicky := speed_nicky * time_nicky
  let distance_cristina := speed_cristina * time_cristina
  distance_nicky = 90 ∧ distance_cristina = 90 :=
by
  sorry

end race_distance_l232_232314


namespace other_diagonal_of_rhombus_l232_232736

noncomputable def calculate_other_diagonal (area d1 : ℝ) : ℝ :=
  (area * 2) / d1

theorem other_diagonal_of_rhombus {a1 a2 : ℝ} (area_eq : a1 = 21.46) (d1_eq : a2 = 7.4) : calculate_other_diagonal a1 a2 = 5.8 :=
by
  rw [area_eq, d1_eq]
  norm_num
  -- The next step would involve proving that (21.46 * 2) / 7.4 = 5.8 in a formal proof.
  sorry

end other_diagonal_of_rhombus_l232_232736


namespace root_in_interval_k_eq_2_l232_232345

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 5

theorem root_in_interval_k_eq_2
  (k : ℤ)
  (h1 : 0 < f 2)
  (h2 : Real.log 2 + 2 * 2 - 5 < 0)
  (h3 : Real.log 3 + 2 * 3 - 5 > 0) 
  (h4 : f (k : ℝ) * f (k + 1 : ℝ) < 0) :
  k = 2 := 
sorry

end root_in_interval_k_eq_2_l232_232345


namespace P_is_sufficient_but_not_necessary_for_Q_l232_232818

def P (x : ℝ) : Prop := (2 * x - 3)^2 < 1
def Q (x : ℝ) : Prop := x * (x - 3) < 0

theorem P_is_sufficient_but_not_necessary_for_Q : 
  (∀ x, P x → Q x) ∧ (∃ x, Q x ∧ ¬ P x) :=
by
  sorry

end P_is_sufficient_but_not_necessary_for_Q_l232_232818


namespace binary_to_base4_representation_l232_232630

def binary_to_base4 (n : ℕ) : ℕ :=
  -- Assuming implementation that converts binary number n to its base 4 representation 
  sorry

theorem binary_to_base4_representation :
  binary_to_base4 0b10110110010 = 23122 :=
by sorry

end binary_to_base4_representation_l232_232630


namespace freshmen_count_l232_232992

theorem freshmen_count (n : ℕ) (h1 : n < 600) (h2 : n % 17 = 16) (h3 : n % 19 = 18) : n = 322 := 
by 
  sorry

end freshmen_count_l232_232992


namespace magnitude_of_sum_l232_232036

variables (a b : ℝ × ℝ)
variables (h1 : a.1 * b.1 + a.2 * b.2 = 0)
variables (h2 : a = (4, 3))
variables (h3 : (b.1 ^ 2 + b.2 ^ 2) = 1)

theorem magnitude_of_sum (a b : ℝ × ℝ) (h1 : a.1 * b.1 + a.2 * b.2 = 0) 
  (h2 : a = (4, 3)) (h3 : (b.1 ^ 2 + b.2 ^ 2) = 1) : 
  (a.1 + 2 * b.1) ^ 2 + (a.2 + 2 * b.2) ^ 2 = 29 :=
by sorry

end magnitude_of_sum_l232_232036


namespace angle_ZAX_pentagon_triangle_common_vertex_l232_232279

theorem angle_ZAX_pentagon_triangle_common_vertex :
  let n_pentagon := 5
  let n_triangle := 3
  let internal_angle_pentagon := (n_pentagon - 2) * 180 / n_pentagon
  let internal_angle_triangle := 60
  let common_angle_A := 360 - (internal_angle_pentagon + internal_angle_pentagon + internal_angle_triangle + internal_angle_triangle) / 2
  common_angle_A = 192 := by
  let n_pentagon := 5
  let n_triangle := 3
  let internal_angle_pentagon := (n_pentagon - 2) * 180 / n_pentagon
  let internal_angle_triangle := 60
  let common_angle_A := 360 - (internal_angle_pentagon + internal_angle_pentagon + internal_angle_triangle + internal_angle_triangle) / 2
  sorry

end angle_ZAX_pentagon_triangle_common_vertex_l232_232279


namespace coloring_count_in_3x3_grid_l232_232620

theorem coloring_count_in_3x3_grid (n m : ℕ) (h1 : n = 3) (h2 : m = 3) : 
  ∃ count : ℕ, count = 15 ∧ ∀ (cells : Finset (Fin n × Fin m)),
  (cells.card = 3 ∧ ∀ (c1 c2 : Fin n × Fin m), c1 ∈ cells → c2 ∈ cells → c1 ≠ c2 → 
  (c1.fst ≠ c2.fst ∧ c1.snd ≠ c2.snd)) → cells.card ∣ count :=
sorry

end coloring_count_in_3x3_grid_l232_232620


namespace building_floors_l232_232237

-- Define the properties of the staircases
def staircaseA_steps : Nat := 104
def staircaseB_steps : Nat := 117
def staircaseC_steps : Nat := 156

-- The problem asks us to show the number of floors, which is the gcd of the steps of all staircases 
theorem building_floors :
  Nat.gcd (Nat.gcd staircaseA_steps staircaseB_steps) staircaseC_steps = 13 :=
by
  sorry

end building_floors_l232_232237


namespace hare_wins_by_10_meters_l232_232991

def speed_tortoise := 3 -- meters per minute
def speed_hare_sprint := 12 -- meters per minute
def speed_hare_walk := 1 -- meters per minute
def time_total := 50 -- minutes
def time_hare_sprint := 10 -- minutes
def time_hare_walk := time_total - time_hare_sprint -- minutes

def distance_tortoise := speed_tortoise * time_total -- meters
def distance_hare := (speed_hare_sprint * time_hare_sprint) + (speed_hare_walk * time_hare_walk) -- meters

theorem hare_wins_by_10_meters : (distance_hare - distance_tortoise) = 10 := by
  -- Proof would go here
  sorry

end hare_wins_by_10_meters_l232_232991


namespace no_such_prime_pair_l232_232879

open Prime

theorem no_such_prime_pair :
  ∀ (p q : ℕ), Prime p → Prime q → (p > 5) → (q > 5) →
  (p * q) ∣ ((5^p - 2^p) * (5^q - 2^q)) → false :=
by
  intros p q hp hq hp_gt5 hq_gt5 hdiv
  sorry

end no_such_prime_pair_l232_232879


namespace egg_laying_hens_l232_232037

theorem egg_laying_hens (total_chickens : ℕ) (num_roosters : ℕ) (non_egg_laying_hens : ℕ)
  (h1 : total_chickens = 325)
  (h2 : num_roosters = 28)
  (h3 : non_egg_laying_hens = 20) :
  total_chickens - num_roosters - non_egg_laying_hens = 277 :=
by sorry

end egg_laying_hens_l232_232037


namespace least_y_solution_l232_232918

theorem least_y_solution :
  (∃ y : ℝ, 3 * y^2 + 5 * y + 2 = 4 ∧ ∀ z : ℝ, 3 * z^2 + 5 * z + 2 = 4 → y ≤ z) →
  ∃ y : ℝ, y = -2 :=
by
  sorry

end least_y_solution_l232_232918


namespace tetrahedron_perpendicular_distances_inequalities_l232_232789

section Tetrahedron

variables {R : Type*} [LinearOrderedField R]

variables {S_A S_B S_C S_D V d_A d_B d_C d_D h_A h_B h_C h_D : R}

/-- Given areas and perpendicular distances of a tetrahedron, prove inequalities involving these parameters. -/
theorem tetrahedron_perpendicular_distances_inequalities 
  (h1 : S_A * d_A + S_B * d_B + S_C * d_C + S_D * d_D = 3 * V) : 
  (min h_A (min h_B (min h_C h_D)) ≤ d_A + d_B + d_C + d_D) ∧ 
  (d_A + d_B + d_C + d_D ≤ max h_A (max h_B (max h_C h_D))) ∧ 
  (d_A * d_B * d_C * d_D ≤ 81 * V ^ 4 / (256 * S_A * S_B * S_C * S_D)) :=
sorry

end Tetrahedron

end tetrahedron_perpendicular_distances_inequalities_l232_232789


namespace valid_triples_l232_232163

theorem valid_triples (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hxy : x ∣ (y + 1)) (hyz : y ∣ (z + 1)) (hzx : z ∣ (x + 1)) :
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ 
  (x = 1 ∧ y = 1 ∧ z = 2) ∨ 
  (x = 1 ∧ y = 2 ∧ z = 3) :=
sorry

end valid_triples_l232_232163


namespace sum_of_arithmetic_sequence_l232_232961

-- Define the conditions
def is_arithmetic_sequence (first_term last_term : ℕ) (terms : ℕ) : Prop :=
  ∃ (a l : ℕ) (n : ℕ), a = first_term ∧ l = last_term ∧ n = terms ∧ n > 1

-- State the theorem
theorem sum_of_arithmetic_sequence (a l n : ℕ) (h_arith: is_arithmetic_sequence 5 41 10):
  n = 10 ∧ a = 5 ∧ l = 41 → (n * (a + l) / 2) = 230 :=
by
  intros h
  sorry

end sum_of_arithmetic_sequence_l232_232961


namespace problem_part1_problem_part2_problem_part3_l232_232252

open Set

-- Define the universal set U
def U : Set ℝ := Set.univ 

-- Define sets A and B within the universal set U
def A : Set ℝ := { x | 0 < x ∧ x ≤ 2 }
def B : Set ℝ := { x | x < -3 ∨ x > 1 }

-- Define the complements of A and B within U
def complement_A : Set ℝ := U \ A
def complement_B : Set ℝ := U \ B

-- Define the results as goals to be proved
theorem problem_part1 : A ∩ B = { x | 1 < x ∧ x ≤ 2 } := 
by
  sorry

theorem problem_part2 : complement_A ∩ complement_B = { x | -3 ≤ x ∧ x ≤ 0 } :=
by
  sorry

theorem problem_part3 : U \ (A ∪ B) = { x | -3 ≤ x ∧ x ≤ 0 } :=
by
  sorry

end problem_part1_problem_part2_problem_part3_l232_232252


namespace percentage_of_hexagon_area_is_closest_to_17_l232_232960

noncomputable def tiling_area_hexagon_percentage : Real :=
  let total_area := 2 * 3
  let square_area := 1 * 1 
  let squares_count := 5 -- Adjusted count from 8 to fit total area properly
  let square_total_area := squares_count * square_area
  let hexagon_area := total_area - square_total_area
  let percentage := (hexagon_area / total_area) * 100
  percentage

theorem percentage_of_hexagon_area_is_closest_to_17 :
  abs (tiling_area_hexagon_percentage - 17) < 1 :=
sorry

end percentage_of_hexagon_area_is_closest_to_17_l232_232960


namespace second_player_can_ensure_symmetry_l232_232989

def is_symmetric (seq : List ℕ) : Prop :=
  seq.reverse = seq

def swap_digits (seq : List ℕ) (i j : ℕ) : List ℕ :=
  if h : i < seq.length ∧ j < seq.length then
    seq.mapIdx (λ k x => if k = i then seq.get ⟨j, h.2⟩ 
                        else if k = j then seq.get ⟨i, h.1⟩ 
                        else x)
  else seq

theorem second_player_can_ensure_symmetry (seq : List ℕ) (h : seq.length = 1999) :
  (∃ swappable_seq : List ℕ, is_symmetric swappable_seq) :=
by
  sorry

end second_player_can_ensure_symmetry_l232_232989


namespace inequality_has_real_solution_l232_232540

variable {f : ℝ → ℝ}

theorem inequality_has_real_solution (h : ∃ x : ℝ, f x > 0) : 
    (∃ x : ℝ, f x > 0) :=
by
  sorry

end inequality_has_real_solution_l232_232540


namespace lower_limit_of_arun_weight_l232_232411

-- Given conditions for Arun's weight
variables (W : ℝ)
variables (avg_val : ℝ)

-- Define the conditions
def arun_weight_condition_1 := W < 72
def arun_weight_condition_2 := 60 < W ∧ W < 70
def arun_weight_condition_3 := W ≤ 67
def arun_weight_avg := avg_val = 66

-- The math proof problem statement
theorem lower_limit_of_arun_weight 
  (h1: arun_weight_condition_1 W) 
  (h2: arun_weight_condition_2 W) 
  (h3: arun_weight_condition_3 W) 
  (h4: arun_weight_avg avg_val) :
  ∃ (lower_limit : ℝ), lower_limit = 65 :=
sorry

end lower_limit_of_arun_weight_l232_232411


namespace least_number_remainder_l232_232495

theorem least_number_remainder (n : ℕ) (h1 : n % 34 = 4) (h2 : n % 5 = 4) : n = 174 :=
  sorry

end least_number_remainder_l232_232495


namespace students_remaining_l232_232727

theorem students_remaining (students_showed_up : ℕ) (students_checked_out : ℕ) (students_left : ℕ) :
  students_showed_up = 16 → students_checked_out = 7 → students_left = students_showed_up - students_checked_out → students_left = 9 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end students_remaining_l232_232727


namespace cos_double_angle_l232_232433

theorem cos_double_angle (theta : ℝ) (h : Real.sin (Real.pi - theta) = 1 / 3) : Real.cos (2 * theta) = 7 / 9 := by
  sorry

end cos_double_angle_l232_232433


namespace smaller_solution_of_quadratic_l232_232148

theorem smaller_solution_of_quadratic :
  ∀ x : ℝ, x^2 + 17 * x - 72 = 0 → x = -24 ∨ x = 3 :=
by sorry

end smaller_solution_of_quadratic_l232_232148


namespace Sandra_brought_20_pairs_l232_232591

-- Definitions for given conditions
variable (S : ℕ) -- S for Sandra's pairs of socks
variable (C : ℕ) -- C for Lisa's cousin's pairs of socks

-- Conditions translated into Lean definitions
def initial_pairs : ℕ := 12
def mom_pairs : ℕ := 3 * initial_pairs + 8 -- Lisa's mom brought 8 more than three times the number of pairs Lisa started with
def cousin_pairs (S : ℕ) : ℕ := S / 5       -- Lisa's cousin brought one-fifth the number of pairs that Sandra bought
def total_pairs (S : ℕ) : ℕ := initial_pairs + S + cousin_pairs S + mom_pairs -- Total pairs of socks Lisa ended up with

-- The theorem to prove
theorem Sandra_brought_20_pairs (h : total_pairs S = 80) : S = 20 :=
by
  sorry

end Sandra_brought_20_pairs_l232_232591


namespace no_square_ends_with_four_identical_digits_except_0_l232_232398

theorem no_square_ends_with_four_identical_digits_except_0 (n : ℤ) :
  ¬ (∃ k : ℕ, (1 ≤ k ∧ k < 10) ∧ (n^2 % 10000 = k * 1111)) :=
by {
  sorry
}

end no_square_ends_with_four_identical_digits_except_0_l232_232398


namespace second_to_last_digit_of_n_squared_plus_2n_l232_232640
open Nat

theorem second_to_last_digit_of_n_squared_plus_2n (n : ℕ) (h : (n^2 + 2 * n) % 10 = 4) : ((n^2 + 2 * n) / 10) % 10 = 2 :=
  sorry

end second_to_last_digit_of_n_squared_plus_2n_l232_232640


namespace equivalent_proof_problem_l232_232242

variables {a b c d e : ℚ}

theorem equivalent_proof_problem
  (h1 : 3 * a + 4 * b + 6 * c + 8 * d + 10 * e = 55)
  (h2 : 4 * (d + c + e) = b)
  (h3 : 4 * b + 2 * c = a)
  (h4 : c - 2 = d)
  (h5 : d + 1 = e) : 
  a * b * c * d * e = -1912397372 / 78364164096 := 
sorry

end equivalent_proof_problem_l232_232242


namespace percentage_increase_l232_232617

variables (a b x m : ℝ) (p : ℝ)
variables (h1 : a / b = 4 / 5)
variables (h2 : x = a + (p / 100) * a)
variables (h3 : m = b - 0.6 * b)
variables (h4 : m / x = 0.4)

theorem percentage_increase (a_pos : 0 < a) (b_pos : 0 < b) : p = 25 :=
by sorry

end percentage_increase_l232_232617


namespace compare_negatives_l232_232423

theorem compare_negatives : -3.3 < -3.14 :=
sorry

end compare_negatives_l232_232423


namespace tea_bags_number_l232_232455

theorem tea_bags_number (n : ℕ) (h1 : 2 * n ≤ 41) (h2 : 41 ≤ 3 * n) (h3 : 2 * n ≤ 58) (h4 : 58 ≤ 3 * n) : n = 20 :=
by
  sorry

end tea_bags_number_l232_232455


namespace total_bill_first_month_l232_232910

theorem total_bill_first_month (F C : ℝ) 
  (h1 : F + C = 50) 
  (h2 : F + 2 * C = 76) 
  (h3 : 2 * C = 2 * C) : 
  F + C = 50 := by
  sorry

end total_bill_first_month_l232_232910


namespace side_length_range_l232_232539

-- Define the inscribed circle diameter condition
def inscribed_circle_diameter (d : ℝ) (cir_diameter : ℝ) := cir_diameter = 1

-- Define inscribed square side condition
def inscribed_square_side (d side : ℝ) :=
  ∃ (triangle_ABC : Type) (AB AC BC : triangle_ABC → ℝ), 
    side = d ∧
    side < 1

-- Define the main theorem: The side length of the inscribed square lies within given bounds
theorem side_length_range (d : ℝ) :
  inscribed_circle_diameter d 1 → inscribed_square_side d d → (4/5) ≤ d ∧ d < 1 :=
by
  intros h1 h2
  sorry

end side_length_range_l232_232539


namespace find_A_l232_232001

theorem find_A (A B : ℕ) (hcfAB lcmAB : ℕ)
  (hcf_cond : Nat.gcd A B = hcfAB)
  (lcm_cond : Nat.lcm A B = lcmAB)
  (B_val : B = 169)
  (hcf_val : hcfAB = 13)
  (lcm_val : lcmAB = 312) :
  A = 24 :=
by 
  sorry

end find_A_l232_232001


namespace geom_seq_sum_is_15_l232_232439

theorem geom_seq_sum_is_15 (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 = 1) (hq : q = -2) (h_geom : ∀ n, a (n + 1) = a n * q) :
  a 1 + |a 2| + a 3 + |a 4| = 15 :=
by
  sorry

end geom_seq_sum_is_15_l232_232439


namespace cats_needed_to_catch_100_mice_in_time_l232_232318

-- Define the context and given conditions
def cats_mice_catch_time (cats mice minutes : ℕ) : Prop :=
  cats = 5 ∧ mice = 5 ∧ minutes = 5

-- Define the goal
theorem cats_needed_to_catch_100_mice_in_time :
  cats_mice_catch_time 5 5 5 → (∃ t : ℕ, t = 500) :=
by
  intro h
  sorry

end cats_needed_to_catch_100_mice_in_time_l232_232318


namespace consecutive_integers_sum_l232_232171

theorem consecutive_integers_sum (x : ℤ) (h : x * (x + 1) = 440) : x + (x + 1) = 43 :=
by sorry

end consecutive_integers_sum_l232_232171


namespace simplify_and_evaluate_l232_232180

theorem simplify_and_evaluate (x : ℤ) (h : x = 2) :
  (2 * x + 1) ^ 2 - (x + 3) * (x - 3) = 30 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_l232_232180


namespace min_value_fraction_l232_232464

theorem min_value_fraction (x y : ℝ) (hx : x > 1) (hy : y > 1) : 
  (∃c, (c = 8) ∧ (∀z w : ℝ, z > 1 → w > 1 → ((z^3 / (w - 1) + w^3 / (z - 1)) ≥ c))) :=
by 
  sorry

end min_value_fraction_l232_232464


namespace time_for_5x5_grid_l232_232516

-- Definitions based on the conditions
def total_length_3x7 : ℕ := 4 * 7 + 8 * 3
def time_for_3x7 : ℕ := 26
def time_per_unit_length : ℚ := time_for_3x7 / total_length_3x7
def total_length_5x5 : ℕ := 6 * 5 + 6 * 5
def expected_time_for_5x5 : ℚ := total_length_5x5 * time_per_unit_length

-- Theorem statement to prove the total time for 5x5 grid
theorem time_for_5x5_grid : expected_time_for_5x5 = 30 := by
  sorry

end time_for_5x5_grid_l232_232516


namespace calculate_total_marks_l232_232523

theorem calculate_total_marks 
  (total_questions : ℕ) 
  (correct_answers : ℕ) 
  (marks_per_correct : ℕ) 
  (marks_per_wrong : ℤ) 
  (total_attempted : total_questions = 60) 
  (correct_attempted : correct_answers = 44)
  (marks_per_correct_is_4 : marks_per_correct = 4)
  (marks_per_wrong_is_neg1 : marks_per_wrong = -1) : 
  total_questions * marks_per_correct - (total_questions - correct_answers) * (abs marks_per_wrong) = 160 := 
by 
  sorry

end calculate_total_marks_l232_232523


namespace smallest_possible_difference_l232_232529

theorem smallest_possible_difference :
  ∃ (x y z : ℕ), 
    x + y + z = 1801 ∧ x < y ∧ y ≤ z ∧ x + y > z ∧ y + z > x ∧ z + x > y ∧ (y - x = 1) := 
by
  sorry

end smallest_possible_difference_l232_232529


namespace upper_limit_l232_232149

noncomputable def upper_limit_Arun (w : ℝ) (X : ℝ) : Prop :=
  (w > 66 ∧ w < X) ∧ (w > 60 ∧ w < 70) ∧ (w ≤ 69) ∧ ((66 + X) / 2 = 68)

theorem upper_limit (w : ℝ) (X : ℝ) (h : upper_limit_Arun w X) : X = 69 :=
by sorry

end upper_limit_l232_232149


namespace problem_I_problem_II_l232_232470

-- Problem (I)
theorem problem_I (a b : ℝ) (h1 : a = 1) (h2 : b = 1) :
  { x : ℝ | |2*x + a| + |2*x - 2*b| + 3 > 8 } = 
  { x : ℝ | x < -1 ∨ x > 1.5 } := by
  sorry

-- Problem (II)
theorem problem_II (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : ∀ x : ℝ, |2*x + a| + |2*x - 2*b| + 3 ≥ 5) :
  (1 / a + 1 / b) = (3 + 2 * Real.sqrt 2) / 2 := by
  sorry

end problem_I_problem_II_l232_232470


namespace ratio_of_volume_to_surface_area_l232_232132

def volume_of_shape (num_cubes : ℕ) : ℕ :=
  -- Volume is simply the number of unit cubes
  num_cubes

def surface_area_of_shape : ℕ :=
  -- Surface area calculation given in the problem and solution
  12  -- edge cubes (4 cubes) with 3 exposed faces each
  + 16  -- side middle cubes (4 cubes) with 4 exposed faces each
  + 1  -- top face of the central cube in the bottom layer
  + 5  -- middle cube in the column with 5 exposed faces
  + 6  -- top cube in the column with all 6 faces exposed

theorem ratio_of_volume_to_surface_area
  (num_cubes : ℕ)
  (h1 : num_cubes = 9) :
  (volume_of_shape num_cubes : ℚ) / (surface_area_of_shape : ℚ) = 9 / 40 :=
by
  sorry

end ratio_of_volume_to_surface_area_l232_232132


namespace decreasing_direct_proportion_l232_232060

theorem decreasing_direct_proportion (k : ℝ) (h : ∀ x1 x2 : ℝ, x1 < x2 → k * x1 > k * x2) : k < 0 :=
by
  sorry

end decreasing_direct_proportion_l232_232060


namespace percentage_discount_of_retail_price_l232_232694

theorem percentage_discount_of_retail_price {wp rp sp discount : ℝ} (h1 : wp = 99) (h2 : rp = 132) (h3 : sp = wp + 0.20 * wp) (h4 : discount = (rp - sp) / rp * 100) : discount = 10 := 
by 
  sorry

end percentage_discount_of_retail_price_l232_232694


namespace remainder_when_divided_by_2_is_0_l232_232422

theorem remainder_when_divided_by_2_is_0 (n : ℕ)
  (h1 : ∃ r, n % 2 = r)
  (h2 : n % 7 = 5)
  (h3 : ∃ p, p = 5 ∧ (n + p) % 10 = 0) :
  n % 2 = 0 :=
by
  -- skipping the proof steps; hence adding sorry
  sorry

end remainder_when_divided_by_2_is_0_l232_232422


namespace solve_for_t_l232_232946

theorem solve_for_t (t : ℝ) (h1 : 60 * t + 80 * ((10 : ℝ)/3 - t) = 220) 
  (h2 : 0 ≤ t) : 60 * t + 80 * ((10 : ℝ)/3 - t) = 220 :=
by
  sorry

end solve_for_t_l232_232946


namespace derivative_of_f_l232_232976

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / x

theorem derivative_of_f (x : ℝ) (hx : x ≠ 0) : deriv f x = ((x * Real.exp x - Real.exp x) / (x * x)) :=
by
  sorry

end derivative_of_f_l232_232976


namespace square_plot_area_l232_232642

theorem square_plot_area (cost_per_foot total_cost : ℕ) (hcost_per_foot : cost_per_foot = 60) (htotal_cost : total_cost = 4080) :
  ∃ (A : ℕ), A = 289 :=
by
  have h : 4 * 60 * 17 = 4080 := by rfl
  have s : 17 = 4080 / (4 * 60) := by sorry
  use 17 ^ 2
  have hsquare : 17 ^ 2 = 289 := by rfl
  exact hsquare

end square_plot_area_l232_232642


namespace volume_ratio_inscribed_circumscribed_sphere_regular_tetrahedron_l232_232978

theorem volume_ratio_inscribed_circumscribed_sphere_regular_tetrahedron (R r : ℝ) (h : r = R / 3) : 
  (4/3 * π * r^3) / (4/3 * π * R^3) = 1 / 27 :=
by
  sorry

end volume_ratio_inscribed_circumscribed_sphere_regular_tetrahedron_l232_232978


namespace problem_y_values_l232_232859

theorem problem_y_values (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 54) :
  ∃ y : ℝ, (y = (x - 3)^2 * (x + 4) / (3 * x - 4)) ∧ (y = 7.5 ∨ y = 4.5) := by
sorry

end problem_y_values_l232_232859


namespace number_of_problems_l232_232876

/-- Given the conditions of the problem, prove that the number of problems I did is exactly 140.-/
theorem number_of_problems (p t : ℕ) (h1 : p > 12) (h2 : p * t = (p + 6) * (t - 3)) : p * t = 140 :=
by
  sorry

end number_of_problems_l232_232876


namespace percentage_error_is_94_l232_232599

theorem percentage_error_is_94 (x : ℝ) (hx : 0 < x) :
  let correct_result := 4 * x
  let error_result := x / 4
  let error := |correct_result - error_result|
  let percentage_error := (error / correct_result) * 100
  percentage_error = 93.75 := by
    sorry

end percentage_error_is_94_l232_232599


namespace parabola_focus_coordinates_l232_232947

theorem parabola_focus_coordinates :
  ∃ (focus : ℝ × ℝ), focus = (0, 1 / 18) ∧ 
    ∃ (p : ℝ), y = 9 * x^2 → x^2 = 4 * p * y ∧ p = 1 / 18 :=
by
  sorry

end parabola_focus_coordinates_l232_232947


namespace quartic_polynomial_eval_l232_232787

noncomputable def f (x : ℝ) : ℝ := sorry  -- f is a monic quartic polynomial

theorem quartic_polynomial_eval (h_monic: true)
    (h1 : f (-1) = -1)
    (h2 : f 2 = -4)
    (h3 : f (-3) = -9)
    (h4 : f 4 = -16) : f 1 = 23 :=
sorry

end quartic_polynomial_eval_l232_232787


namespace carrie_bought_t_shirts_l232_232048

theorem carrie_bought_t_shirts (total_spent : ℝ) (cost_each : ℝ) (n : ℕ) 
    (h_total : total_spent = 199) (h_cost : cost_each = 9.95) 
    (h_eq : n = total_spent / cost_each) : n = 20 := 
by
sorry

end carrie_bought_t_shirts_l232_232048


namespace ratio_boys_to_girls_l232_232378

theorem ratio_boys_to_girls (g b : ℕ) (h1 : g + b = 30) (h2 : b = g + 3) : 
  (b : ℚ) / g = 16 / 13 := 
by 
  sorry

end ratio_boys_to_girls_l232_232378


namespace hanoi_moves_correct_l232_232474

def hanoi_moves (n : ℕ) : ℕ :=
  if n = 0 then 0 else 2 * hanoi_moves (n - 1) + 1

theorem hanoi_moves_correct (n : ℕ) : hanoi_moves n = 2^n - 1 := by
  sorry

end hanoi_moves_correct_l232_232474


namespace find_missing_number_l232_232109

theorem find_missing_number 
  (x : ℝ) (y : ℝ)
  (h1 : (12 + x + 42 + 78 + 104) / 5 = 62)
  (h2 : (128 + 255 + y + 1023 + x) / 5 = 398.2) :
  y = 511 := 
sorry

end find_missing_number_l232_232109


namespace computation_result_l232_232768

-- Define the vectors and scalar multiplications
def v1 : ℤ × ℤ := (3, -9)
def v2 : ℤ × ℤ := (2, -7)
def v3 : ℤ × ℤ := (-1, 4)

noncomputable def result : ℤ × ℤ := 
  let scalar_mult (m : ℤ) (v : ℤ × ℤ) : ℤ × ℤ := (m * v.1, m * v.2)
  scalar_mult 5 v1 - scalar_mult 3 v2 + scalar_mult 2 v3

-- The main theorem
theorem computation_result : result = (7, -16) :=
  by 
    -- Skip the proof as required
    sorry

end computation_result_l232_232768


namespace sum_remainder_l232_232028

theorem sum_remainder (m : ℤ) : ((9 - m) + (m + 5)) % 8 = 6 :=
by
  sorry

end sum_remainder_l232_232028


namespace math_proof_problem_l232_232521

noncomputable def f (x : ℝ) := Real.log (Real.sin x) * Real.log (Real.cos x)

def domain (k : ℤ) : Set ℝ := { x | 2 * k * Real.pi < x ∧ x < 2 * k * Real.pi + Real.pi / 2 }

def is_even_shifted : Prop :=
  ∀ x, f (x + Real.pi / 4) = f (- (x + Real.pi / 4))

def has_unique_maximum : Prop :=
  ∃! x, 0 < x ∧ x < Real.pi / 2 ∧ ∀ y, 0 < y ∧ y < Real.pi / 2 → f y ≤ f x

theorem math_proof_problem (k : ℤ) :
  (∀ x, x ∈ domain k → f x ∈ domain k) ∧
  ¬ (∀ x, f (-x) = f x) ∧
  is_even_shifted ∧
  has_unique_maximum :=
by
  sorry

end math_proof_problem_l232_232521


namespace find_divisor_l232_232481

theorem find_divisor : exists d : ℕ, 
  (∀ x : ℕ, x ≥ 10 ∧ x ≤ 1000000 → x % d = 0) ∧ 
  (10 + 999990 * d/111110 = 1000000) ∧
  d = 9 := by
  sorry

end find_divisor_l232_232481


namespace part_I_part_II_part_III_l232_232685

noncomputable def f (x : ℝ) := x / (x^2 - 1)

-- (I) Prove that f(2) = 2/3.
theorem part_I : f 2 = 2 / 3 :=
by sorry

-- (II) Prove that f(x) is decreasing on the interval (-1, 1).
theorem part_II : ∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → x2 < 1 → f x1 > f x2 :=
by sorry

-- (III) Prove that f(x) is an odd function.
theorem part_III : ∀ x : ℝ, f (-x) = -f x :=
by sorry

end part_I_part_II_part_III_l232_232685


namespace equation_solution_l232_232223

theorem equation_solution (x : ℝ) (h : (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 2)) : x = 9 :=
by
  sorry

end equation_solution_l232_232223


namespace problem_statement_l232_232794

theorem problem_statement (f : ℝ → ℝ) (a b : ℝ) (h_f : ∀ x : ℝ, f x = x^2 + x + 1) 
  (h_a : a > 0) (h_b : b > 0) :
  (∀ x : ℝ, |x - 1| < b → |f x - 3| < a) ↔ b ≤ a / 3 :=
sorry

end problem_statement_l232_232794


namespace average_salary_excluding_manager_l232_232948

theorem average_salary_excluding_manager
    (A : ℝ)
    (manager_salary : ℝ)
    (total_employees : ℕ)
    (salary_increase : ℝ)
    (h1 : total_employees = 24)
    (h2 : manager_salary = 4900)
    (h3 : salary_increase = 100)
    (h4 : 24 * A + manager_salary = 25 * (A + salary_increase)) :
    A = 2400 := by
  sorry

end average_salary_excluding_manager_l232_232948


namespace eventually_one_student_answers_yes_l232_232598

-- Conditions and Definitions
variable (a b r₁ r₂ : ℕ)
variable (h₁ : r₁ ≠ r₂)   -- r₁ and r₂ are distinct
variable (h₂ : r₁ = a + b ∨ r₂ = a + b) -- One of r₁ or r₂ is the sum a + b
variable (h₃ : a > 0) -- a is a positive integer
variable (h₄ : b > 0) -- b is a positive integer

theorem eventually_one_student_answers_yes (a b r₁ r₂ : ℕ) (h₁ : r₁ ≠ r₂) (h₂ : r₁ = a + b ∨ r₂ = a + b) (h₃ : a > 0) (h₄ : b > 0) :
  ∃ n : ℕ, (∃ c : ℕ, (r₁ = c + b ∨ r₂ = c + b) ∧ (c = a ∨ c ≤ r₁ ∨ c ≤ r₂)) ∨ 
  (∃ c : ℕ, (r₁ = a + c ∨ r₂ = a + c) ∧ (c = b ∨ c ≤ r₁ ∨ c ≤ r₂)) :=
sorry

end eventually_one_student_answers_yes_l232_232598


namespace find_functions_l232_232770

noncomputable def pair_of_functions_condition (f g : ℝ → ℝ) : Prop :=
∀ x y : ℝ, g (f (x + y)) = f x + (2 * x + y) * g y

theorem find_functions (f g : ℝ → ℝ) :
  pair_of_functions_condition f g →
  (∃ c d : ℝ, ∀ x : ℝ, f x = c * (x + d)) :=
sorry

end find_functions_l232_232770


namespace women_fraction_l232_232432

/-- In a room with 100 people, 1/4 of whom are married, the maximum number of unmarried women is 40.
    We need to prove that the fraction of women in the room is 2/5. -/
theorem women_fraction (total_people : ℕ) (married_fraction : ℚ) (unmarried_women : ℕ) (W : ℚ) 
  (h1 : total_people = 100) 
  (h2 : married_fraction = 1 / 4) 
  (h3 : unmarried_women = 40) 
  (hW : W = 2 / 5) : 
  W = 2 / 5 := 
by
  sorry

end women_fraction_l232_232432


namespace analytical_expression_of_f_range_of_f_on_interval_l232_232525

noncomputable def f (x : ℝ) (a c : ℝ) : ℝ := a * x^3 + c * x

theorem analytical_expression_of_f
  (a c : ℝ)
  (h1 : a > 0)
  (h2 : ∀ x, f x a c = a * x^3 + c * x) 
  (h3 : 3 * a + c = -6)
  (h4 : ∀ x, (3 * a * x ^ 2 + c) ≥ -12) :
    a = 2 ∧ c = -12 :=
by
  sorry

theorem range_of_f_on_interval
  (h1 : ∃ a c, a = 2 ∧ c = -12)
  (h2 : ∀ x, f x 2 (-12) = 2 * x^3 - 12 * x)
  :
    Set.range (fun x => f x 2 (-12)) = Set.Icc (-8 * Real.sqrt 2) (8 * Real.sqrt 2) :=
by
  sorry

end analytical_expression_of_f_range_of_f_on_interval_l232_232525


namespace repeating_block_length_five_sevenths_l232_232407

theorem repeating_block_length_five_sevenths : 
  ∃ n : ℕ, (∃ k : ℕ, (5 * 10^k - 5) % 7 = 0) ∧ n = 6 :=
sorry

end repeating_block_length_five_sevenths_l232_232407


namespace arithmetic_sequence_n_15_l232_232394

theorem arithmetic_sequence_n_15 (a : ℕ → ℤ) (n : ℕ)
  (h₁ : a 3 = 5)
  (h₂ : a 2 + a 5 = 12)
  (h₃ : a n = 29) :
  n = 15 :=
sorry

end arithmetic_sequence_n_15_l232_232394


namespace polynomial_value_at_minus_2_l232_232227

-- Define the polynomial f(x)
def f (x : ℤ) := x^6 - 5 * x^5 + 6 * x^4 + x^2 + 3 * x + 2

-- Define the evaluation point
def x_val : ℤ := -2

-- State the theorem we want to prove
theorem polynomial_value_at_minus_2 : f x_val = 320 := 
by sorry

end polynomial_value_at_minus_2_l232_232227


namespace symmetrical_ring_of_polygons_l232_232580

theorem symmetrical_ring_of_polygons (m n : ℕ) (hn : n ≥ 7) (hm : m ≥ 3) 
  (condition1 : ∀ p1 p2 : ℕ, p1 ≠ p2 → n = 1) 
  (condition2 : ∀ p : ℕ, p * (n - 2) = 4) 
  (condition3 : ∀ p : ℕ, 2 * m - (n - 2) = 4) :
  ∃ k, (k = 6) :=
by
  -- This block is only a placeholder. The actual proof would go here.
  sorry

end symmetrical_ring_of_polygons_l232_232580


namespace parking_spaces_in_the_back_l232_232442

theorem parking_spaces_in_the_back
  (front_spaces : ℕ)
  (cars_parked : ℕ)
  (half_back_filled : ℕ → ℚ)
  (spaces_available : ℕ)
  (B : ℕ)
  (h1 : front_spaces = 52)
  (h2 : cars_parked = 39)
  (h3 : half_back_filled B = B / 2)
  (h4 : spaces_available = 32) :
  B = 38 :=
by
  -- Here you can provide the proof steps.
  sorry

end parking_spaces_in_the_back_l232_232442


namespace expression_value_l232_232355

theorem expression_value (a b c : ℕ) (h1 : 25^a * 5^(2*b) = 5^6) (h2 : 4^b / 4^c = 4) : a^2 + a * b + 3 * c = 6 := by
  sorry

end expression_value_l232_232355


namespace find_percentage_l232_232453

theorem find_percentage (P : ℕ) (h: (P / 100) * 180 - (1 / 3) * (P / 100) * 180 = 18) : P = 15 :=
sorry

end find_percentage_l232_232453


namespace total_money_spent_l232_232146

def cost_life_journey_cd : ℕ := 100
def cost_day_life_cd : ℕ := 50
def cost_when_rescind_cd : ℕ := 85
def number_of_cds_each : ℕ := 3

theorem total_money_spent :
  number_of_cds_each * cost_life_journey_cd +
  number_of_cds_each * cost_day_life_cd +
  number_of_cds_each * cost_when_rescind_cd = 705 :=
sorry

end total_money_spent_l232_232146


namespace solve_floor_fractional_l232_232325

noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

noncomputable def fractional_part (x : ℝ) : ℝ :=
  x - floor x

theorem solve_floor_fractional (x : ℝ) :
  floor x * fractional_part x = 2019 * x ↔ x = 0 ∨ x = -1 / 2020 :=
by
  sorry

end solve_floor_fractional_l232_232325


namespace probability_product_divisible_by_four_l232_232923

open Finset

theorem probability_product_divisible_by_four :
  (∃ (favorable_pairs total_pairs : ℕ), favorable_pairs = 70 ∧ total_pairs = 190 ∧ favorable_pairs / total_pairs = 7 / 19) := 
sorry

end probability_product_divisible_by_four_l232_232923


namespace sand_exchange_impossible_to_achieve_l232_232692

-- Let G and P be the initial weights of gold and platinum sand, respectively
def initial_G : ℕ := 1 -- 1 kg
def initial_P : ℕ := 1 -- 1 kg

-- Initial values for g and p
def initial_g : ℕ := 1001
def initial_p : ℕ := 1001

-- Daily reduction of either g or p
axiom decrease_g_or_p (g p : ℕ) : g > 1 ∨ p > 1 → (g = g - 1 ∨ p = p - 1) ∧ (g ≥ 1) ∧ (p ≥ 1)

-- Final condition: after 2000 days, g and p both equal to 1
axiom final_g_p_after_2000_days : ∀ (g p : ℕ), (g = initial_g - 2000) ∧ (p = initial_p - 2000) → g = 1 ∧ p = 1

-- State of the system, defined as S = G * p + P * g
def S (G P g p : ℕ) : ℕ := G * p + P * g

-- Prove that after 2000 days, the banker cannot have at least 2 kg of each type of sand
theorem sand_exchange_impossible_to_achieve (G P g p : ℕ) (h : G = initial_G) (h1 : P = initial_P) 
  (h2 : g = initial_g) (h3 : p = initial_p) : 
  ∀ (d : ℕ), (d = 2000) → (g = 1) ∧ (p = 1) 
    → (S G P g p < 4) :=
by
  sorry

end sand_exchange_impossible_to_achieve_l232_232692


namespace rectangle_area_with_circles_touching_l232_232483

theorem rectangle_area_with_circles_touching
  (r : ℝ)
  (radius_pos : r = 3)
  (short_side : ℝ)
  (long_side : ℝ)
  (dim_rect : short_side = 2 * r ∧ long_side = 4 * r) :
  short_side * long_side = 72 :=
by
  sorry

end rectangle_area_with_circles_touching_l232_232483


namespace rectangle_width_l232_232650

-- Define the conditions
def length := 6
def area_triangle := 60
def area_ratio := 2/5

-- The theorem: proving that the width of the rectangle is 4 cm
theorem rectangle_width (w : ℝ) (A_triangle : ℝ) (len : ℝ) 
  (ratio : ℝ) (h1 : A_triangle = 60) (h2 : len = 6) (h3 : ratio = 2 / 5) 
  (h4 : (len * w) / A_triangle = ratio) : 
  w = 4 := 
by 
  sorry

end rectangle_width_l232_232650


namespace converse_of_statement_l232_232039

variables (a b : ℝ)

theorem converse_of_statement :
  (a + b ≤ 2) → (a ≤ 1 ∨ b ≤ 1) :=
by
  sorry

end converse_of_statement_l232_232039


namespace t_shirts_per_package_l232_232638

theorem t_shirts_per_package (total_tshirts : ℕ) (packages : ℕ) (tshirts_per_package : ℕ) :
  total_tshirts = 70 → packages = 14 → tshirts_per_package = total_tshirts / packages → tshirts_per_package = 5 :=
by
  sorry

end t_shirts_per_package_l232_232638


namespace sum_areas_of_tangent_circles_l232_232490

theorem sum_areas_of_tangent_circles : 
  ∃ r s t : ℝ, 
    (r + s = 6) ∧ 
    (r + t = 8) ∧ 
    (s + t = 10) ∧ 
    (π * (r^2 + s^2 + t^2) = 36 * π) :=
by
  sorry

end sum_areas_of_tangent_circles_l232_232490


namespace convert_to_base_k_l232_232669

noncomputable def base_k_eq (k : ℕ) : Prop :=
  4 * k + 4 = 36

theorem convert_to_base_k :
  ∃ k : ℕ, base_k_eq k ∧ (67 / k^2 % k^2 % k = 1 ∧ 67 / k % k = 0 ∧ 67 % k = 3) :=
sorry

end convert_to_base_k_l232_232669


namespace number_of_three_digit_numbers_with_5_and_7_l232_232172

def isThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def containsDigit (n : ℕ) (d : ℕ) : Prop := d ∈ (n.digits 10) 
def hasAtLeastOne5andOne7 (n : ℕ) : Prop := containsDigit n 5 ∧ containsDigit n 7
def totalThreeDigitNumbersWith5and7 : ℕ := 50

theorem number_of_three_digit_numbers_with_5_and_7 :
  ∃ n : ℕ, isThreeDigitNumber n ∧ hasAtLeastOne5andOne7 n → n = 50 := sorry

end number_of_three_digit_numbers_with_5_and_7_l232_232172


namespace real_solutions_to_system_l232_232343

theorem real_solutions_to_system (x y : ℝ) (h1 : x^3 + y^3 = 1) (h2 : x^4 + y^4 = 1) :
  (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) :=
sorry

end real_solutions_to_system_l232_232343


namespace simplify_expression_l232_232548

theorem simplify_expression (z : ℝ) : (3 - 5 * z^2) - (4 * z^2 + 2 * z - 5) = 8 - 9 * z^2 - 2 * z :=
by
  sorry

end simplify_expression_l232_232548


namespace second_person_time_l232_232971

theorem second_person_time (x : ℝ) (h1 : ∀ t : ℝ, t = 3) 
(h2 : (1/3 + 1/x) = 5/12) : x = 12 := 
by sorry

end second_person_time_l232_232971


namespace sin_double_angle_ratio_l232_232017

theorem sin_double_angle_ratio (α : ℝ) (h : Real.sin α = 3 * Real.cos α) : 
  Real.sin (2 * α) / (Real.cos α)^2 = 6 :=
by 
  sorry

end sin_double_angle_ratio_l232_232017


namespace quadratic_factors_l232_232686

-- Define the quadratic polynomial
def quadratic (b c x : ℝ) : ℝ := x^2 + b * x + c

-- Define the roots
def root1 : ℝ := -2
def root2 : ℝ := 3

-- Theorem: If the quadratic equation has roots -2 and 3, then it factors as (x + 2)(x - 3)
theorem quadratic_factors (b c : ℝ) (h1 : quadratic b c root1 = 0) (h2 : quadratic b c root2 = 0) :
  ∀ x : ℝ, quadratic b c x = (x + 2) * (x - 3) :=
by
  sorry

end quadratic_factors_l232_232686


namespace alan_tickets_l232_232527

variables (A M : ℕ)

def condition1 := A + M = 150
def condition2 := M = 5 * A - 6

theorem alan_tickets : A = 26 :=
by
  have h1 : condition1 A M := sorry
  have h2 : condition2 A M := sorry
  sorry

end alan_tickets_l232_232527


namespace polynomial_identity_l232_232531

theorem polynomial_identity
  (x : ℝ)
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ)
  (h : (2*x + 1)^6 = a_0*x^6 + a_1*x^5 + a_2*x^4 + a_3*x^3 + a_4*x^2 + a_5*x + a_6) :
  (a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = 729)
  ∧ (a_1 + a_3 + a_5 = 364)
  ∧ (a_2 + a_4 = 300) := sorry

end polynomial_identity_l232_232531


namespace Nina_money_l232_232023

theorem Nina_money : ∃ (M : ℝ) (W : ℝ), M = 10 * W ∧ M = 14 * (W - 3) ∧ M = 105 :=
by
  sorry

end Nina_money_l232_232023


namespace tree_height_increase_fraction_l232_232576

theorem tree_height_increase_fraction :
  ∀ (initial_height annual_increase : ℝ) (additional_years₄ additional_years₆ : ℕ),
    initial_height = 4 →
    annual_increase = 0.4 →
    additional_years₄ = 4 →
    additional_years₆ = 6 →
    ((initial_height + annual_increase * additional_years₆) - (initial_height + annual_increase * additional_years₄)) / (initial_height + annual_increase * additional_years₄) = 1 / 7 :=
by
  sorry

end tree_height_increase_fraction_l232_232576


namespace arithmetic_sequence_sum_l232_232416

theorem arithmetic_sequence_sum {a : ℕ → ℝ} (d a1 : ℝ)
  (h_arith: ∀ n, a n = a1 + (n - 1) * d)
  (h_condition: a 3 + a 8 = 10) :
  3 * a 5 + a 7 = 20 :=
by {
  sorry
}

end arithmetic_sequence_sum_l232_232416


namespace largest_angle_in_pentagon_l232_232373

-- Define the angles of the pentagon
variables (C D E : ℝ) 

-- Given conditions
def is_pentagon (A B C D E : ℝ) : Prop :=
  A = 75 ∧ B = 95 ∧ D = C + 10 ∧ E = 2 * C + 20 ∧ A + B + C + D + E = 540

-- Prove that the measure of the largest angle is 190°
theorem largest_angle_in_pentagon (C D E : ℝ) : 
  is_pentagon 75 95 C D E → max 75 (max 95 (max C (max (C + 10) (2 * C + 20)))) = 190 :=
by 
  sorry

end largest_angle_in_pentagon_l232_232373


namespace problem_trapezoid_l232_232569

noncomputable def ratio_of_areas (AB CD : ℝ) (h : ℝ) (ratio : ℝ) :=
  let area_trapezoid := (AB + CD) * h / 2
  let area_triangle_AZW := (4 * h) / 15
  ratio = area_triangle_AZW / area_trapezoid

theorem problem_trapezoid :
  ratio_of_areas 2 5 h (8 / 105) :=
by
  sorry

end problem_trapezoid_l232_232569


namespace margie_change_l232_232259

theorem margie_change :
  let num_apples := 5
  let cost_per_apple := 0.30
  let discount := 0.10
  let amount_paid := 10.00
  let total_cost := num_apples * cost_per_apple
  let discounted_cost := total_cost * (1 - discount)
  let change_received := amount_paid - discounted_cost
  change_received = 8.65 := sorry

end margie_change_l232_232259


namespace union_sets_l232_232959

theorem union_sets :
  let A := { x : ℝ | x^2 - x - 2 < 0 }
  let B := { x : ℝ | x > -2 ∧ x < 0 }
  A ∪ B = { x : ℝ | x > -2 ∧ x < 2 } :=
by
  sorry

end union_sets_l232_232959


namespace rectangle_circumference_15pi_l232_232808

noncomputable def rectangle_diagonal (a b : ℝ) : ℝ := 
  Real.sqrt (a ^ 2 + b ^ 2)

noncomputable def circumference_of_circle (d : ℝ) : ℝ := 
  Real.pi * d
  
theorem rectangle_circumference_15pi :
  let a := 9
  let b := 12
  let diagonal := rectangle_diagonal a b
  circumference_of_circle diagonal = 15 * Real.pi :=
by 
  sorry

end rectangle_circumference_15pi_l232_232808


namespace Ivan_cannot_cut_off_all_heads_l232_232122

-- Defining the number of initial heads
def initial_heads : ℤ := 100

-- Effect of the first sword: Removes 21 heads
def first_sword_effect : ℤ := 21

-- Effect of the second sword: Removes 4 heads and adds 2006 heads
def second_sword_effect : ℤ := 2006 - 4

-- Proving Ivan cannot reduce the number of heads to zero
theorem Ivan_cannot_cut_off_all_heads :
  (∀ n : ℤ, n % 7 = initial_heads % 7 → n ≠ 0) :=
by
  sorry

end Ivan_cannot_cut_off_all_heads_l232_232122


namespace sara_initial_quarters_l232_232365

theorem sara_initial_quarters (borrowed quarters_current : ℕ) (q_initial : ℕ) :
  quarters_current = 512 ∧ quarters_borrowed = 271 → q_initial = 783 :=
by
  sorry

end sara_initial_quarters_l232_232365


namespace contrapositive_proof_l232_232898

variable {p q : Prop}

theorem contrapositive_proof : (p → q) ↔ (¬q → ¬p) :=
  by sorry

end contrapositive_proof_l232_232898


namespace B_finishes_job_in_37_5_days_l232_232156

variable (eff_A eff_B eff_C : ℝ)
variable (effA_eq_half_effB : eff_A = (1 / 2) * eff_B)
variable (effB_eq_two_thirds_effC : eff_B = (2 / 3) * eff_C)
variable (job_in_15_days : 15 * (eff_A + eff_B + eff_C) = 1)

theorem B_finishes_job_in_37_5_days :
  (1 / eff_B) = 37.5 :=
by
  sorry

end B_finishes_job_in_37_5_days_l232_232156


namespace polynomial_inequality_holds_l232_232915

theorem polynomial_inequality_holds (a : ℝ) : (∀ x : ℝ, x^4 + (a-2)*x^2 + a ≥ 0) ↔ a ≥ 4 - 2 * Real.sqrt 3 := 
by
  sorry

end polynomial_inequality_holds_l232_232915


namespace georgie_guacamole_servings_l232_232682

-- Define the conditions
def avocados_needed_per_serving : Nat := 3
def initial_avocados : Nat := 5
def additional_avocados : Nat := 4

-- State the target number of servings Georgie can make
def total_avocados := initial_avocados + additional_avocados
def guacamole_servings := total_avocados / avocados_needed_per_serving

-- Lean 4 statement asserting the number of servings equals 3
theorem georgie_guacamole_servings : guacamole_servings = 3 := by
  sorry

end georgie_guacamole_servings_l232_232682


namespace ratio_of_auto_finance_companies_credit_l232_232145

theorem ratio_of_auto_finance_companies_credit
    (total_consumer_credit : ℝ)
    (percent_auto_installment_credit : ℝ)
    (credit_by_auto_finance_companies : ℝ)
    (total_auto_credit : ℝ)
    (hc1 : total_consumer_credit = 855)
    (hc2 : percent_auto_installment_credit = 0.20)
    (hc3 : credit_by_auto_finance_companies = 57)
    (htotal_auto_credit : total_auto_credit = percent_auto_installment_credit * total_consumer_credit) :
    (credit_by_auto_finance_companies / total_auto_credit) = (1 / 3) := 
by
  sorry

end ratio_of_auto_finance_companies_credit_l232_232145


namespace total_distance_is_20_l232_232713

noncomputable def total_distance_walked (x : ℝ) : ℝ :=
  let flat_distance := 4 * x
  let uphill_time := (2 / 3) * (5 - x)
  let uphill_distance := 3 * uphill_time
  let downhill_time := (1 / 3) * (5 - x)
  let downhill_distance := 6 * downhill_time
  flat_distance + uphill_distance + downhill_distance

theorem total_distance_is_20 :
  ∃ x : ℝ, x >= 0 ∧ x <= 5 ∧ total_distance_walked x = 20 :=
by
  -- The existence proof is omitted (hence the sorry)
  sorry

end total_distance_is_20_l232_232713


namespace man_l232_232413

theorem man's_speed_with_current (v c : ℝ) (h1 : c = 4.3) (h2 : v - c = 12.4) : v + c = 21 :=
by {
  sorry
}

end man_l232_232413


namespace toll_for_18_wheel_truck_l232_232867

-- Define the conditions
def wheels_per_axle : Nat := 2
def total_wheels : Nat := 18
def toll_formula (x : Nat) : ℝ := 1.5 + 0.5 * (x - 2)

-- Calculate number of axles from the number of wheels
def number_of_axles := total_wheels / wheels_per_axle

-- Target statement: The toll for the given truck
theorem toll_for_18_wheel_truck : toll_formula number_of_axles = 5.0 := by
  sorry

end toll_for_18_wheel_truck_l232_232867


namespace find_f_of_3_l232_232626

theorem find_f_of_3 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2 * x + 1) = x^2 - 2 * x) : f 3 = -1 :=
by 
  sorry

end find_f_of_3_l232_232626


namespace surface_area_of_larger_prism_l232_232354

def volume_of_brick := 288
def number_of_bricks := 11
def target_surface_area := 1368

theorem surface_area_of_larger_prism
    (vol: ℕ := volume_of_brick)
    (num: ℕ := number_of_bricks)
    (target: ℕ := target_surface_area)
    (exists_a_b_h : ∃ (a b h : ℕ), a = 12 ∧ b = 8 ∧ h = 3)
    (large_prism_dimensions : ∃ (L W H : ℕ), L = 24 ∧ W = 12 ∧ H = 11):
    2 * (24 * 12 + 24 * 11 + 12 * 11) = target :=
by
  sorry

end surface_area_of_larger_prism_l232_232354


namespace water_evaporation_correct_l232_232903

noncomputable def water_evaporation_each_day (initial_water: ℝ) (percentage_evaporated: ℝ) (days: ℕ) : ℝ :=
  let total_evaporated := (percentage_evaporated / 100) * initial_water
  total_evaporated / days

theorem water_evaporation_correct :
  water_evaporation_each_day 10 6 30 = 0.02 := by
  sorry

end water_evaporation_correct_l232_232903


namespace total_bills_proof_l232_232277

variable (a : ℝ) (total_may : ℝ) (total_june_may_june : ℝ)

-- The total bill in May is 140 yuan.
def total_bill_may (a : ℝ) := 140

-- The water bill increases by 10% in June.
def water_bill_june (a : ℝ) := 1.1 * a

-- The electricity bill in May.
def electricity_bill_may (a : ℝ) := 140 - a

-- The electricity bill increases by 20% in June.
def electricity_bill_june (a : ℝ) := (140 - a) * 1.2

-- Total electricity bills in June.
def total_electricity_june (a : ℝ) := (140 - a) + 0.2 * (140 - a)

-- Total water and electricity bills in June.
def total_water_electricity_june (a : ℝ) := 1.1 * a + 168 - 1.2 * a

-- Total water and electricity bills for May and June.
def total_water_electricity_may_june (a : ℝ) := a + (1.1 * a) + (140 - a) + ((140 - a) * 1.2)

-- When a = 40, the total water and electricity bills for May and June.
theorem total_bills_proof : ∀ a : ℝ, a = 40 → total_water_electricity_may_june a = 304 := 
by
  intros a ha
  rw [ha]
  sorry

end total_bills_proof_l232_232277


namespace arithmetic_square_root_of_nine_l232_232297

theorem arithmetic_square_root_of_nine :
  ∃ x : ℝ, x^2 = 9 ∧ x = 3 :=
by
  sorry

end arithmetic_square_root_of_nine_l232_232297


namespace initial_books_eq_41_l232_232332

-- Definitions and conditions
def books_sold : ℕ := 33
def books_added : ℕ := 2
def books_remaining : ℕ := 10

-- Proof problem
theorem initial_books_eq_41 (B : ℕ) (h : B - books_sold + books_added = books_remaining) : B = 41 :=
by
  sorry

end initial_books_eq_41_l232_232332


namespace tallest_vs_shortest_height_difference_l232_232840

-- Define the heights of the trees
def pine_tree_height := 12 + 4/5
def birch_tree_height := 18 + 1/2
def maple_tree_height := 14 + 3/5

-- Calculate improper fractions
def pine_tree_improper := 64 / 5
def birch_tree_improper := 41 / 2  -- This is 82/4 but not simplified here
def maple_tree_improper := 73 / 5

-- Calculate height difference
def height_difference := (82 / 4) - (64 / 5)

-- The statement that needs to be proven
theorem tallest_vs_shortest_height_difference : height_difference = 7 + 7 / 10 :=
by 
  sorry

end tallest_vs_shortest_height_difference_l232_232840


namespace tank_fish_count_l232_232809

theorem tank_fish_count (total_fish blue_fish : ℕ) 
  (h1 : blue_fish = total_fish / 3)
  (h2 : 10 * 2 = blue_fish) : 
  total_fish = 60 :=
sorry

end tank_fish_count_l232_232809


namespace ellipse_condition_l232_232067

theorem ellipse_condition (m : ℝ) :
  (m > 0) ∧ (2 * m - 1 > 0) ∧ (m ≠ 2 * m - 1) ↔ (m > 1/2) ∧ (m ≠ 1) :=
by
  sorry

end ellipse_condition_l232_232067


namespace arithmetic_seq_common_diff_l232_232381

theorem arithmetic_seq_common_diff
  (a : ℕ → ℝ) (d : ℝ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_d_nonzero : d ≠ 0)
  (h_a1 : a 1 = 1)
  (h_geomet : (a 3) ^ 2 = a 1 * a 13) :
  d = 2 :=
by
  sorry

end arithmetic_seq_common_diff_l232_232381


namespace sufficient_condition_for_inequality_l232_232513

theorem sufficient_condition_for_inequality (a : ℝ) : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) → a ≥ 5 :=
by
  sorry

end sufficient_condition_for_inequality_l232_232513


namespace problem_statement_l232_232463

structure Pricing :=
  (price_per_unit_1 : ℕ) (threshold_1 : ℕ)
  (price_per_unit_2 : ℕ) (threshold_2 : ℕ)
  (price_per_unit_3 : ℕ)

def cost (units : ℕ) (pricing : Pricing) : ℕ :=
  let t1 := pricing.threshold_1
  let t2 := pricing.threshold_2
  let p1 := pricing.price_per_unit_1
  let p2 := pricing.price_per_unit_2
  let p3 := pricing.price_per_unit_3
  if units ≤ t1 then units * p1
  else if units ≤ t2 then t1 * p1 + (units - t1) * p2
  else t1 * p1 + (t2 - t1) * p2 + (units - t2) * p3 

def units_given_cost (c : ℕ) (pricing : Pricing) : ℕ :=
  let t1 := pricing.threshold_1
  let t2 := pricing.threshold_2
  let p1 := pricing.price_per_unit_1
  let p2 := pricing.price_per_unit_2
  let p3 := pricing.price_per_unit_3
  if c ≤ t1 * p1 then c / p1
  else if c ≤ t1 * p1 + (t2 - t1) * p2 then t1 + (c - t1 * p1) / p2
  else t2 + (c - t1 * p1 - (t2 - t1) * p2) / p3

def double_eleven_case (total_units total_cost : ℕ) (x_units : ℕ) (pricing : Pricing) : ℕ :=
  let y_units := total_units - x_units
  let case1_cost := cost x_units pricing + cost y_units pricing
  if case1_cost = total_cost then (x_units, y_units).fst
  else sorry

theorem problem_statement (pricing : Pricing):
  (cost 120 pricing = 420) ∧ 
  (cost 260 pricing = 868) ∧
  (units_given_cost 740 pricing = 220) ∧
  (double_eleven_case 400 1349 290 pricing = 290)
  := sorry

end problem_statement_l232_232463


namespace first_generation_tail_length_l232_232704

theorem first_generation_tail_length
  (length_first_gen : ℝ)
  (H : (1.25:ℝ) * (1.25:ℝ) * length_first_gen = 25) :
  length_first_gen = 16 := by
  sorry

end first_generation_tail_length_l232_232704


namespace three_pow_124_mod_7_l232_232207

theorem three_pow_124_mod_7 : (3^124) % 7 = 4 := by
  sorry

end three_pow_124_mod_7_l232_232207


namespace ram_leela_piggy_bank_l232_232253

theorem ram_leela_piggy_bank (final_amount future_deposits weeks: ℕ) 
  (initial_deposit common_diff: ℕ) (total_deposits : ℕ) 
  (h_total : total_deposits = (weeks * (initial_deposit + (initial_deposit + (weeks - 1) * common_diff)) / 2)) 
  (h_final : final_amount = 1478) 
  (h_weeks : weeks = 52) 
  (h_future_deposits : future_deposits = total_deposits) 
  (h_initial_deposit : initial_deposit = 1) 
  (h_common_diff : common_diff = 1) 
  : final_amount - future_deposits = 100 :=
sorry

end ram_leela_piggy_bank_l232_232253


namespace additional_money_needed_l232_232784

theorem additional_money_needed :
  let total_budget := 500
  let budget_dresses := 300
  let budget_shoes := 150
  let budget_accessories := 50
  let extra_fraction := 2 / 5
  let discount_rate := 0.15
  let total_without_discount := 
    budget_dresses * (1 + extra_fraction) +
    budget_shoes * (1 + extra_fraction) +
    budget_accessories * (1 + extra_fraction)
  let discounted_total := total_without_discount * (1 - discount_rate)
  discounted_total > total_budget :=
sorry

end additional_money_needed_l232_232784


namespace paint_cans_used_l232_232418

theorem paint_cans_used (init_rooms : ℕ) (lost_cans : ℕ) (remaining_rooms : ℕ) (final_rooms : ℕ) :
  init_rooms = 50 → lost_cans = 5 → remaining_rooms = 40 → final_rooms = 40 → 
  remaining_rooms / (lost_cans / (init_rooms - remaining_rooms)) = 20 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  sorry

end paint_cans_used_l232_232418


namespace money_bounds_l232_232144

   theorem money_bounds (a b : ℝ) (h₁ : 4 * a + 2 * b > 110) (h₂ : 2 * a + 3 * b = 105) : a > 15 ∧ b < 25 :=
   by
     sorry
   
end money_bounds_l232_232144


namespace roots_of_polynomial_l232_232737

theorem roots_of_polynomial : 
  (∀ x : ℝ, (x^3 - 6*x^2 + 11*x - 6) * (x - 2) = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3) :=
by
  intro x
  sorry

end roots_of_polynomial_l232_232737


namespace cylinder_volume_increase_l232_232631

theorem cylinder_volume_increase (r h : ℝ) (V : ℝ) : 
  V = π * r^2 * h → 
  (3 * h) * (2 * r)^2 * π = 12 * V := by
    sorry

end cylinder_volume_increase_l232_232631


namespace simple_interest_time_l232_232811

-- Definitions based on given conditions
def SI : ℝ := 640           -- Simple interest
def P : ℝ := 4000           -- Principal
def R : ℝ := 8              -- Rate
def T : ℝ := 2              -- Time in years (correct answer to be proved)

-- Theorem statement
theorem simple_interest_time :
  SI = (P * R * T) / 100 := 
by 
  sorry

end simple_interest_time_l232_232811


namespace property_related_only_to_temperature_l232_232447

-- The conditions given in the problem
def solubility_of_ammonia_gas (T P : Prop) : Prop := T ∧ P
def ion_product_of_water (T : Prop) : Prop := T
def oxidizing_property_of_pp (T C A : Prop) : Prop := T ∧ C ∧ A
def degree_of_ionization_of_acetic_acid (T C : Prop) : Prop := T ∧ C

-- The statement to prove
theorem property_related_only_to_temperature
  (T P C A : Prop)
  (H1 : solubility_of_ammonia_gas T P)
  (H2 : ion_product_of_water T)
  (H3 : oxidizing_property_of_pp T C A)
  (H4 : degree_of_ionization_of_acetic_acid T C) :
  ∃ T, ion_product_of_water T ∧
        ¬solubility_of_ammonia_gas T P ∧
        ¬oxidizing_property_of_pp T C A ∧
        ¬degree_of_ionization_of_acetic_acid T C :=
by
  sorry

end property_related_only_to_temperature_l232_232447


namespace isabel_games_problem_l232_232950

noncomputable def prime_sum : ℕ := 83 + 89 + 97

theorem isabel_games_problem (initial_games : ℕ) (X : ℕ) (H1 : initial_games = 90) (H2 : X = prime_sum) : X > initial_games :=
by 
  sorry

end isabel_games_problem_l232_232950


namespace find_possible_values_l232_232472

noncomputable def complex_values (x y : ℂ) : Prop :=
  (x^2 + y^2) / (x + y) = 4 ∧ (x^4 + y^4) / (x^3 + y^3) = 2

theorem find_possible_values (x y : ℂ) (h : complex_values x y) :
  ∃ z : ℂ, z = (x^6 + y^6) / (x^5 + y^5) ∧ (z = 10 + 2 * Real.sqrt 17 ∨ z = 10 - 2 * Real.sqrt 17) :=
sorry

end find_possible_values_l232_232472


namespace batsman_average_after_12_innings_l232_232189

theorem batsman_average_after_12_innings
  (score_12th: ℕ) (increase_avg: ℕ) (initial_innings: ℕ) (final_innings: ℕ) 
  (initial_avg: ℕ) (final_avg: ℕ) :
  score_12th = 48 ∧ increase_avg = 2 ∧ initial_innings = 11 ∧ final_innings = 12 ∧
  final_avg = initial_avg + increase_avg ∧
  12 * final_avg = initial_innings * initial_avg + score_12th →
  final_avg = 26 :=
by 
  sorry

end batsman_average_after_12_innings_l232_232189


namespace annual_sales_profit_relationship_and_maximum_l232_232187

def cost_per_unit : ℝ := 6
def selling_price (x : ℝ) := x > 6
def sales_volume (u : ℝ) := u * 10000
def proportional_condition (x u : ℝ) := (585 / 8) - u = 2 * (x - 21 / 4) ^ 2
def sales_volume_condition : Prop := proportional_condition 10 28

theorem annual_sales_profit_relationship_and_maximum (x u y : ℝ) 
    (hx : selling_price x) 
    (hu : proportional_condition x u) 
    (hs : sales_volume_condition) :
    (y = (-2 * x^3 + 33 * x^2 - 108 * x - 108)) ∧ 
    (x = 9 → y = 135) := 
sorry

end annual_sales_profit_relationship_and_maximum_l232_232187


namespace limit_of_sequence_l232_232703

theorem limit_of_sequence (a_n : ℕ → ℝ) (a : ℝ) :
  (∀ n : ℕ, a_n n = (2 * (n ^ 3)) / ((n ^ 3) - 2)) →
  a = 2 →
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a_n n - a| < ε) :=
by
  intros h1 h2 ε hε
  sorry

end limit_of_sequence_l232_232703


namespace total_distance_journey_l232_232555

def miles_driven : ℕ := 384
def miles_remaining : ℕ := 816

theorem total_distance_journey :
  miles_driven + miles_remaining = 1200 :=
by
  sorry

end total_distance_journey_l232_232555


namespace solve_equation_l232_232530

theorem solve_equation : 
  ∀ x : ℝ, 
  (((15 * x - x^2) / (x + 2)) * (x + (15 - x) / (x + 2)) = 54) → (x = 9 ∨ x = -1) :=
by
  sorry

end solve_equation_l232_232530


namespace age_problem_l232_232868

theorem age_problem 
  (P R J M : ℕ)
  (h1 : P = 1 / 2 * R)
  (h2 : R = J + 7)
  (h3 : J + 12 = 3 * P)
  (h4 : M = J + 17)
  (h5 : M = 2 * R + 4) : 
  P = 5 ∧ R = 10 ∧ J = 3 ∧ M = 24 :=
by sorry

end age_problem_l232_232868


namespace paper_clips_in_two_cases_l232_232847

-- Define the conditions
variables (c b : ℕ)

-- Define the theorem statement
theorem paper_clips_in_two_cases (c b : ℕ) : 
    2 * c * b * 400 = 2 * c * b * 400 :=
by
  sorry

end paper_clips_in_two_cases_l232_232847


namespace mat_radius_increase_l232_232743

theorem mat_radius_increase (C1 C2 : ℝ) (h1 : C1 = 40) (h2 : C2 = 50) :
  let r1 := C1 / (2 * Real.pi)
  let r2 := C2 / (2 * Real.pi)
  (r2 - r1) = 5 / Real.pi := by
  sorry

end mat_radius_increase_l232_232743


namespace translation_proof_l232_232514

-- Define the points and the translation process
def point_A : ℝ × ℝ := (-1, 0)
def point_B : ℝ × ℝ := (1, 2)
def point_C : ℝ × ℝ := (1, -2)

-- Translation from point A to point C
def translation_vector : ℝ × ℝ :=
  (point_C.1 - point_A.1, point_C.2 - point_A.2)

-- Define point D using the translation vector applied to point B
def point_D : ℝ × ℝ :=
  (point_B.1 + translation_vector.1, point_B.2 + translation_vector.2)

-- Statement to prove point D has the expected coordinates
theorem translation_proof : 
  point_D = (3, 0) :=
by 
  -- The exact proof is omitted, presented here for completion
  sorry

end translation_proof_l232_232514


namespace min_points_in_symmetric_set_l232_232953

theorem min_points_in_symmetric_set (T : Set (ℝ × ℝ)) (h1 : ∀ {a b : ℝ}, (a, b) ∈ T → (a, -b) ∈ T)
                                      (h2 : ∀ {a b : ℝ}, (a, b) ∈ T → (-a, b) ∈ T)
                                      (h3 : ∀ {a b : ℝ}, (a, b) ∈ T → (-b, -a) ∈ T)
                                      (h4 : (1, 4) ∈ T) : 
    ∃ (S : Finset (ℝ × ℝ)), 
          (∀ p ∈ S, p ∈ T) ∧
          (∀ q ∈ T, ∃ p ∈ S, q = (p.1, p.2) ∨ q = (p.1, -p.2) ∨ q = (-p.1, p.2) ∨ q = (-p.1, -p.2) ∨ q = (-p.2, -p.1) ∨ q = (-p.2, p.1) ∨ q = (p.2, p.1) ∨ q = (p.2, -p.1)) ∧
          S.card = 8 := sorry

end min_points_in_symmetric_set_l232_232953


namespace minimum_value_of_quadratic_l232_232170

theorem minimum_value_of_quadratic (p q : ℝ) (hp : 0 < p) (hq : 0 < q) : 
  ∃ x : ℝ, x = - (p + q) / 2 ∧ ∀ y : ℝ, (y^2 + p*y + q*y) ≥ ((- (p + q) / 2)^2 + p*(- (p + q) / 2) + q*(- (p + q) / 2)) := by
  sorry

end minimum_value_of_quadratic_l232_232170


namespace inequality_problem_l232_232348

theorem inequality_problem {a b c : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b + b * c + c * a = a + b + c) :
  a^2 + b^2 + c^2 + 2 * a * b * c ≥ 5 :=
sorry

end inequality_problem_l232_232348


namespace brandon_skittles_final_l232_232997
-- Conditions
def brandon_initial_skittles := 96
def brandon_lost_skittles := 9

-- Theorem stating the question and answer
theorem brandon_skittles_final : brandon_initial_skittles - brandon_lost_skittles = 87 := 
by
  -- Proof steps go here
  sorry

end brandon_skittles_final_l232_232997


namespace determine_a_l232_232198

theorem determine_a (a : ℝ) :
  (∃ (x y : ℝ), (|y - 10| + |x + 3| - 2) * (x^2 + y^2 - 6) = 0 ∧ (x + 3)^2 + (y - 5)^2 = a) →
  (a = 49 ∨ a = 40 - 4 * Real.sqrt 51) :=
by sorry

end determine_a_l232_232198


namespace inverse_contrapositive_l232_232536

theorem inverse_contrapositive (a b c : ℝ) (h : a > b → a + c > b + c) :
  a + c ≤ b + c → a ≤ b :=
sorry

end inverse_contrapositive_l232_232536


namespace not_sum_of_squares_of_form_4m_plus_3_l232_232985

theorem not_sum_of_squares_of_form_4m_plus_3 (n m : ℤ) (h : n = 4 * m + 3) : 
  ¬ ∃ a b : ℤ, n = a^2 + b^2 :=
by
  sorry

end not_sum_of_squares_of_form_4m_plus_3_l232_232985


namespace condition_sufficient_but_not_necessary_l232_232308

theorem condition_sufficient_but_not_necessary (x y : ℝ) :
  (|x| + |y| ≤ 1 → x^2 + y^2 ≤ 1) ∧ (x^2 + y^2 ≤ 1 → ¬ (|x| + |y| ≤ 1)) :=
sorry

end condition_sufficient_but_not_necessary_l232_232308


namespace triangle_area_difference_l232_232437

-- Definitions based on given lengths and right angles.
def GH : ℝ := 5
def HI : ℝ := 7
def FG : ℝ := 9

-- Note: Right angles are implicitly used in the area calculations and do not need to be represented directly in Lean.
-- Define areas for triangles involved.
def area_FGH : ℝ := 0.5 * FG * GH
def area_GHI : ℝ := 0.5 * GH * HI
def area_FHI : ℝ := 0.5 * FG * HI

-- Define areas of the triangles FGJ and HJI using variables.
variable (x y z : ℝ)
axiom area_FGJ : x = area_FHI - z
axiom area_HJI : y = area_GHI - z

-- The main proof statement involving the difference.
theorem triangle_area_difference : (x - y) = 14 := by
  sorry

end triangle_area_difference_l232_232437


namespace length_of_yellow_line_l232_232309

theorem length_of_yellow_line
  (w1 w2 w3 w4 : ℝ) (path_width : ℝ) (middle_line_dist : ℝ)
  (h1 : w1 = 40) (h2 : w2 = 10) (h3 : w3 = 20) (h4 : w4 = 30) (h5 : path_width = 5) (h6 : middle_line_dist = 2.5) :
  w1 - path_width * middle_line_dist/2 + w2 + w3 + w4 - path_width * middle_line_dist/2 = 95 :=
by sorry

end length_of_yellow_line_l232_232309


namespace gain_percent_is_80_l232_232512

theorem gain_percent_is_80 (C S : ℝ) (h : 81 * C = 45 * S) : ((S - C) / C) * 100 = 80 :=
by
  sorry

end gain_percent_is_80_l232_232512


namespace speed_of_stream_l232_232955

theorem speed_of_stream (v : ℝ) (canoe_speed : ℝ) 
  (upstream_speed_condition : canoe_speed - v = 3) 
  (downstream_speed_condition : canoe_speed + v = 12) :
  v = 4.5 := 
by 
  sorry

end speed_of_stream_l232_232955


namespace problem_statement_eq_l232_232998

variable (x y : ℝ)

def dollar (a b : ℝ) : ℝ := (a - b) ^ 2

theorem problem_statement_eq :
  dollar ((x + y) ^ 2) ((y + x) ^ 2) = 0 := by
  sorry

end problem_statement_eq_l232_232998


namespace cannot_be_written_as_square_l232_232524

theorem cannot_be_written_as_square (A B : ℤ) : 
  99999 + 111111 * Real.sqrt 3 ≠ (A + B * Real.sqrt 3) ^ 2 :=
by
  -- Here we would provide the actual mathematical proof
  sorry

end cannot_be_written_as_square_l232_232524


namespace smallest_integer_to_make_1008_perfect_square_l232_232837

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem smallest_integer_to_make_1008_perfect_square : ∃ k : ℕ, k > 0 ∧ 
  (∀ m : ℕ, m > 0 → (is_perfect_square (1008 * m) → m ≥ k)) ∧ is_perfect_square (1008 * k) :=
by
  sorry

end smallest_integer_to_make_1008_perfect_square_l232_232837


namespace symmetry_P_over_xOz_l232_232919

-- Definition for the point P and the plane xOz
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def P : Point3D := { x := 2, y := 3, z := 4 }

def symmetry_over_xOz_plane (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

theorem symmetry_P_over_xOz : symmetry_over_xOz_plane P = { x := 2, y := -3, z := 4 } :=
by
  -- The proof is omitted.
  sorry

end symmetry_P_over_xOz_l232_232919


namespace partitions_distinct_parts_eq_odd_parts_l232_232717

def num_partitions_into_distinct_parts (n : ℕ) : ℕ := sorry
def num_partitions_into_odd_parts (n : ℕ) : ℕ := sorry

theorem partitions_distinct_parts_eq_odd_parts (n : ℕ) :
  num_partitions_into_distinct_parts n = num_partitions_into_odd_parts n :=
  sorry

end partitions_distinct_parts_eq_odd_parts_l232_232717


namespace nebraska_license_plate_increase_l232_232556

open Nat

theorem nebraska_license_plate_increase :
  let old_plates : ℕ := 26 * 10^3
  let new_plates : ℕ := 26^2 * 10^4
  new_plates / old_plates = 260 :=
by
  -- Definitions based on conditions
  let old_plates : ℕ := 26 * 10^3
  let new_plates : ℕ := 26^2 * 10^4
  -- Assertion to prove
  show new_plates / old_plates = 260
  sorry

end nebraska_license_plate_increase_l232_232556


namespace probability_of_choosing_A_l232_232234

def P (n : ℕ) : ℝ :=
  if n = 0 then 1 else 0.5 + 0.5 * (-0.2)^(n-1)

theorem probability_of_choosing_A (n : ℕ) :
  P n = if n = 0 then 1 else 0.5 + 0.5 * (-0.2)^(n-1) := 
by {
  sorry
}

end probability_of_choosing_A_l232_232234


namespace zoe_distance_more_than_leo_l232_232276

theorem zoe_distance_more_than_leo (d t s : ℝ)
  (maria_driving_time : ℝ := t + 2)
  (maria_speed : ℝ := s + 15)
  (zoe_driving_time : ℝ := t + 3)
  (zoe_speed : ℝ := s + 20)
  (leo_distance : ℝ := s * t)
  (maria_distance : ℝ := (s + 15) * (t + 2))
  (zoe_distance : ℝ := (s + 20) * (t + 3))
  (maria_leo_distance_diff : ℝ := 110)
  (h1 : maria_distance = leo_distance + maria_leo_distance_diff)
  : zoe_distance - leo_distance = 180 :=
by
  sorry

end zoe_distance_more_than_leo_l232_232276


namespace cafeteria_students_count_l232_232938

def total_students : ℕ := 90

def initial_in_cafeteria : ℕ := total_students * 2 / 3

def initial_outside : ℕ := total_students / 3

def ran_inside : ℕ := initial_outside / 3

def ran_outside : ℕ := 3

def net_change_in_cafeteria : ℕ := ran_inside - ran_outside

def final_in_cafeteria : ℕ := initial_in_cafeteria + net_change_in_cafeteria

theorem cafeteria_students_count : final_in_cafeteria = 67 := 
by
  sorry

end cafeteria_students_count_l232_232938


namespace inequality_of_distinct_positives_l232_232203

variable {a b c : ℝ}

theorem inequality_of_distinct_positives (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
(habc : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  2 * (a^3 + b^3 + c^3) > a^2 * (b + c) + b^2 * (a + c) + c^2 * (a + b) :=
by
  sorry

end inequality_of_distinct_positives_l232_232203


namespace range_of_a_l232_232371

open Real

noncomputable def C1 (t a : ℝ) : ℝ × ℝ := (2 * t + 2 * a, -t)
noncomputable def C2 (θ : ℝ) : ℝ × ℝ := (2 * cos θ, 2 + 2 * sin θ)

theorem range_of_a {a : ℝ} :
  (∃ (t θ : ℝ), C1 t a = C2 θ) ↔ 2 - sqrt 5 ≤ a ∧ a ≤ 2 + sqrt 5 :=
by 
  sorry

end range_of_a_l232_232371


namespace find_m_l232_232820

theorem find_m (m : ℝ) :
  (∃ x a : ℝ, |x - 1| - |x + m| ≥ a ∧ a ≤ 5) ↔ (m = 4 ∨ m = -6) :=
by
  sorry

end find_m_l232_232820


namespace strawberry_rows_l232_232611

theorem strawberry_rows (yield_per_row total_harvest : ℕ) (h1 : yield_per_row = 268) (h2 : total_harvest = 1876) :
  total_harvest / yield_per_row = 7 := 
by 
  sorry

end strawberry_rows_l232_232611


namespace union_sets_l232_232352

theorem union_sets (M N : Set ℝ) (hM : M = {x | -3 < x ∧ x < 1}) (hN : N = {x | x ≤ -3}) :
  M ∪ N = {x | x < 1} :=
sorry

end union_sets_l232_232352


namespace fred_current_money_l232_232888

-- Conditions
def initial_amount_fred : ℕ := 19
def earned_amount_fred : ℕ := 21

-- Question and Proof
theorem fred_current_money : initial_amount_fred + earned_amount_fred = 40 :=
by sorry

end fred_current_money_l232_232888


namespace gain_percent_calculation_l232_232283

noncomputable def gain_percent (C S : ℝ) : ℝ :=
  (S - C) / C * 100

theorem gain_percent_calculation (C S : ℝ) (h : 50 * C = 46 * S) : 
  gain_percent C S = 100 / 11.5 :=
by
  sorry

end gain_percent_calculation_l232_232283


namespace solve_picnic_problem_l232_232834

def picnic_problem : Prop :=
  ∃ (M W A C : ℕ), 
    M = W + 80 ∧ 
    A = C + 80 ∧ 
    M + W = A ∧ 
    A + C = 240 ∧ 
    M = 120

theorem solve_picnic_problem : picnic_problem :=
  sorry

end solve_picnic_problem_l232_232834


namespace evaluate_g_at_3_l232_232366

def g : ℝ → ℝ := fun x => x^2 - 3 * x + 2

theorem evaluate_g_at_3 : g 3 = 2 := by
  sorry

end evaluate_g_at_3_l232_232366


namespace hyperbola_with_foci_condition_l232_232772

theorem hyperbola_with_foci_condition (k : ℝ) :
  ( ∀ x y : ℝ, (x^2 / (k + 3)) + (y^2 / (k + 2)) = 1 → ∀ x y : ℝ, (x^2 / (k + 3)) + (y^2 / (k + 2)) = 1 ∧ (k + 3 > 0 ∧ k + 2 < 0) ) ↔ (-3 < k ∧ k < -2) :=
sorry

end hyperbola_with_foci_condition_l232_232772


namespace second_pipe_filling_time_l232_232777

theorem second_pipe_filling_time (T : ℝ) :
  (∃ T : ℝ, (1 / 8 + 1 / T = 1 / 4.8) ∧ T = 12) :=
by
  sorry

end second_pipe_filling_time_l232_232777


namespace find_sixth_number_l232_232153

theorem find_sixth_number (A : ℕ → ℤ) 
  (h1 : (1 / 11 : ℚ) * (A 1 + A 2 + A 3 + A 4 + A 5 + A 6 + A 7 + A 8 + A 9 + A 10 + A 11) = 60)
  (h2 : (1 / 6 : ℚ) * (A 1 + A 2 + A 3 + A 4 + A 5 + A 6) = 88)
  (h3 : (1 / 6 : ℚ) * (A 6 + A 7 + A 8 + A 9 + A 10 + A 11) = 65) :
  A 6 = 258 :=
sorry

end find_sixth_number_l232_232153


namespace length_of_other_train_l232_232901

def speed1 := 90 -- speed in km/hr
def speed2 := 90 -- speed in km/hr
def length_train1 := 1.10 -- length in km
def crossing_time := 40 -- time in seconds

theorem length_of_other_train : 
  ∀ s1 s2 l1 t l2 : ℝ,
  s1 = 90 → s2 = 90 → l1 = 1.10 → t = 40 → 
  ((s1 + s2) / 3600 * t - l1 = l2) → 
  l2 = 0.90 :=
by
  intros s1 s2 l1 t l2 hs1 hs2 hl1 ht hdist
  sorry

end length_of_other_train_l232_232901


namespace relationship_m_n_k_l_l232_232380

-- Definitions based on the conditions
variables (m n k l : ℕ)

-- Condition: Number of teachers (m), Number of students (n)
-- Each teacher teaches exactly k students
-- Any pair of students has exactly l common teachers

theorem relationship_m_n_k_l (h1 : 0 < m) (h2 : 0 < n) (h3 : 0 < k) (h4 : 0 < l)
  (hk : k * (k - 1) / 2 = k * (k - 1) / 2) (hl : n * (n - 1) / 2 = n * (n - 1) / 2) 
  (h5 : m * (k * (k - 1)) = (n * (n - 1)) * l) :
  m * k * (k - 1) = n * (n - 1) * l :=
by 
  sorry

end relationship_m_n_k_l_l232_232380


namespace monotonicity_of_f_abs_f_diff_ge_four_abs_diff_l232_232167

noncomputable def f (a x : ℝ) : ℝ := (a + 1) * Real.log x + a * x^2 + 1

theorem monotonicity_of_f {a : ℝ} (x : ℝ) (hx : 0 < x) :
  (f a x) = (f a x) := sorry

theorem abs_f_diff_ge_four_abs_diff {a x1 x2: ℝ} (ha : a ≤ -2) (hx1 : 0 < x1) (hx2 : 0 < x2) :
  |f a x1 - f a x2| ≥ 4 * |x1 - x2| := sorry

end monotonicity_of_f_abs_f_diff_ge_four_abs_diff_l232_232167


namespace vertex_in_fourth_quadrant_l232_232614

theorem vertex_in_fourth_quadrant (m : ℝ) (h : m < 0) : 
  (0 < -m) ∧ (-1 < 0) :=
by
  sorry

end vertex_in_fourth_quadrant_l232_232614


namespace alice_probability_multiple_of_4_l232_232208

noncomputable def probability_one_multiple_of_4 (choices : ℕ) : ℚ :=
  let p_not_multiple_of_4 : ℚ := 45 / 60
  let p_all_not_multiple_of_4 : ℚ := p_not_multiple_of_4 ^ choices
  1 - p_all_not_multiple_of_4

theorem alice_probability_multiple_of_4 :
  probability_one_multiple_of_4 3 = 37 / 64 :=
by
  sorry

end alice_probability_multiple_of_4_l232_232208


namespace abs_diff_of_two_numbers_l232_232072

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x * y = 320) : |x - y| = 4 := 
by
  sorry

end abs_diff_of_two_numbers_l232_232072


namespace initial_minutes_under_plan_A_l232_232404

theorem initial_minutes_under_plan_A (x : ℕ) (planA_initial : ℝ) (planA_rate : ℝ) (planB_rate : ℝ) (call_duration : ℕ) :
  planA_initial = 0.60 ∧ planA_rate = 0.06 ∧ planB_rate = 0.08 ∧ call_duration = 3 ∧
  (planA_initial + planA_rate * (call_duration - x) = planB_rate * call_duration) →
  x = 9 := 
by
  intros h
  obtain ⟨h1, h2, h3, h4, heq⟩ := h
  -- Skipping the proof
  sorry

end initial_minutes_under_plan_A_l232_232404


namespace total_fish_weight_is_25_l232_232562

-- Define the conditions and the problem
def num_trout : ℕ := 4
def weight_trout : ℝ := 2
def num_catfish : ℕ := 3
def weight_catfish : ℝ := 1.5
def num_bluegills : ℕ := 5
def weight_bluegill : ℝ := 2.5

-- Calculate the total weight of each type of fish
def total_weight_trout : ℝ := num_trout * weight_trout
def total_weight_catfish : ℝ := num_catfish * weight_catfish
def total_weight_bluegills : ℝ := num_bluegills * weight_bluegill

-- Calculate the total weight of all fish
def total_weight_fish : ℝ := total_weight_trout + total_weight_catfish + total_weight_bluegills

-- Statement to be proved
theorem total_fish_weight_is_25 : total_weight_fish = 25 := by
  sorry

end total_fish_weight_is_25_l232_232562


namespace units_digit_7_pow_3_pow_5_l232_232175

theorem units_digit_7_pow_3_pow_5 : ∀ (n : ℕ), n % 4 = 3 → ∀ k, 7 ^ k ≡ 3 [MOD 10] :=
by 
    sorry

end units_digit_7_pow_3_pow_5_l232_232175


namespace increasing_intervals_l232_232384

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x - Real.pi / 3)

theorem increasing_intervals :
  ∀ x : ℝ, x ∈ Set.Icc (-Real.pi) 0 →
    (f x > f (x - ε) ∧ f x < f (x + ε) ∧ x ∈ Set.Icc (-Real.pi) (-7 * Real.pi / 12) ∪ Set.Icc (-Real.pi / 12) 0) :=
sorry

end increasing_intervals_l232_232384


namespace accurate_to_hundreds_place_l232_232822

def rounded_number : ℝ := 8.80 * 10^4

theorem accurate_to_hundreds_place
  (n : ℝ) (h : n = rounded_number) : 
  exists (d : ℤ), n = d * 100 ∧ |round n - n| < 50 :=
sorry

end accurate_to_hundreds_place_l232_232822


namespace dolls_total_l232_232482

theorem dolls_total (V S A : ℕ) 
  (hV : V = 20) 
  (hS : S = 2 * V)
  (hA : A = 2 * S) 
  : A + S + V = 140 := 
by 
  sorry

end dolls_total_l232_232482


namespace length_of_one_string_l232_232338

theorem length_of_one_string (total_length : ℕ) (num_strings : ℕ) (h_total_length : total_length = 98) (h_num_strings : num_strings = 7) : total_length / num_strings = 14 := by
  sorry

end length_of_one_string_l232_232338


namespace system_solution_l232_232535

noncomputable def x1 : ℝ := 55 / Real.sqrt 91
noncomputable def y1 : ℝ := 18 / Real.sqrt 91
noncomputable def x2 : ℝ := -55 / Real.sqrt 91
noncomputable def y2 : ℝ := -18 / Real.sqrt 91

theorem system_solution (x y : ℝ) (h1 : x^2 = 4 * y^2 + 19) (h2 : x * y + 2 * y^2 = 18) :
  (x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2) :=
sorry

end system_solution_l232_232535


namespace framing_needed_l232_232334

def orig_width : ℕ := 5
def orig_height : ℕ := 7
def border_width : ℕ := 3
def doubling_factor : ℕ := 2
def inches_per_foot : ℕ := 12

-- Define the new dimensions after doubling
def new_width := orig_width * doubling_factor
def new_height := orig_height * doubling_factor

-- Define the dimensions after adding the border
def final_width := new_width + 2 * border_width
def final_height := new_height + 2 * border_width

-- Calculate the perimeter in inches
def perimeter := 2 * (final_width + final_height)

-- Convert perimeter to feet and round up if necessary
def framing_feet := (perimeter + inches_per_foot - 1) / inches_per_foot

theorem framing_needed : framing_feet = 6 := by
  sorry

end framing_needed_l232_232334


namespace second_man_start_time_l232_232024

theorem second_man_start_time (P Q : Type) (departure_time_P departure_time_Q meeting_time arrival_time_P arrival_time_Q : ℕ) 
(distance speed : ℝ) (first_man_speed second_man_speed : ℕ → ℝ)
(h1 : departure_time_P = 6) 
(h2 : arrival_time_Q = 10) 
(h3 : arrival_time_P = 12) 
(h4 : meeting_time = 9) 
(h5 : ∀ t, 0 ≤ t ∧ t ≤ 4 → first_man_speed t = distance / 4)
(h6 : ∀ t, second_man_speed t = distance / 4)
(h7 : ∀ t, second_man_speed t * (meeting_time - t) = (3 * distance / 4))
: departure_time_Q = departure_time_P :=
by 
  sorry

end second_man_start_time_l232_232024


namespace find_cd_l232_232201

def g (c d x : ℝ) : ℝ := c * x^3 - 4 * x^2 + d * x - 7

theorem find_cd :
  let c := -1 / 3
  let d := 28 / 3
  g c d 2 = -7 ∧ g c d (-1) = -20 :=
by sorry

end find_cd_l232_232201


namespace tinas_extra_earnings_l232_232075

def price_per_candy_bar : ℕ := 2
def marvins_candy_bars_sold : ℕ := 35
def tinas_candy_bars_sold : ℕ := 3 * marvins_candy_bars_sold

def marvins_earnings : ℕ := marvins_candy_bars_sold * price_per_candy_bar
def tinas_earnings : ℕ := tinas_candy_bars_sold * price_per_candy_bar

theorem tinas_extra_earnings : tinas_earnings - marvins_earnings = 140 := by
  sorry

end tinas_extra_earnings_l232_232075


namespace Peter_can_always_ensure_three_distinct_real_roots_l232_232878

noncomputable def cubic_has_three_distinct_real_roots (b d : ℝ) : Prop :=
∃ (a : ℝ), ∃ (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ 
  (r1 * r2 * r3 = -a) ∧ (r1 + r2 + r3 = -b) ∧ (r1 * r2 + r2 * r3 + r3 * r1 = -d)

theorem Peter_can_always_ensure_three_distinct_real_roots (b d : ℝ) :
  cubic_has_three_distinct_real_roots b d :=
sorry

end Peter_can_always_ensure_three_distinct_real_roots_l232_232878


namespace isosceles_triangle_perimeter_l232_232099

noncomputable def is_isosceles_triangle (a b c : ℕ) : Prop :=
  a = b ∨ b = c ∨ c = a

theorem isosceles_triangle_perimeter {a b c : ℕ} (h1 : is_isosceles_triangle a b c) (h2 : a = 3 ∨ a = 6)
  (h3 : b = 3 ∨ b = 6) (h4 : c = 3 ∨ c = 6) (h5 : a + b + c = 15) : a + b + c = 15 :=
by
  sorry

end isosceles_triangle_perimeter_l232_232099


namespace total_earnings_l232_232452

def phone_repair_cost : ℕ := 11
def laptop_repair_cost : ℕ := 15
def computer_repair_cost : ℕ := 18

def num_phone_repairs : ℕ := 5
def num_laptop_repairs : ℕ := 2
def num_computer_repairs : ℕ := 2

theorem total_earnings :
  phone_repair_cost * num_phone_repairs
  + laptop_repair_cost * num_laptop_repairs
  + computer_repair_cost * num_computer_repairs = 121 := by
  sorry

end total_earnings_l232_232452


namespace comparison_of_a_and_c_l232_232592

variable {α : Type _} [LinearOrderedField α]

theorem comparison_of_a_and_c (a b c : α) (h1 : a > b) (h2 : (a - b) * (b - c) * (c - a) > 0) : a > c :=
sorry

end comparison_of_a_and_c_l232_232592


namespace find_x_y_l232_232581

theorem find_x_y (x y : ℝ)
  (h1 : (x - 1) ^ 2003 + 2002 * (x - 1) = -1)
  (h2 : (y - 2) ^ 2003 + 2002 * (y - 2) = 1) :
  x + y = 3 :=
sorry

end find_x_y_l232_232581


namespace pentagon_diagonals_l232_232760

def number_of_sides_pentagon : ℕ := 5
def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem pentagon_diagonals : number_of_diagonals number_of_sides_pentagon = 5 := by
  sorry

end pentagon_diagonals_l232_232760


namespace meeting_success_probability_l232_232700

noncomputable def meeting_probability : ℝ :=
  let totalVolume := 1.5 ^ 3
  let z_gt_x_y := (1.5 * 1.5 * 1.5) / 3
  let assistants_leave := 2 * ((1.5 * 0.5 / 2) / 3 * 0.5)
  let effectiveVolume := z_gt_x_y - assistants_leave
  let probability := effectiveVolume / totalVolume
  probability

theorem meeting_success_probability :
  meeting_probability = 8 / 27 := by
  sorry

end meeting_success_probability_l232_232700


namespace scientific_notation_of_858_million_l232_232194

theorem scientific_notation_of_858_million :
  858000000 = 8.58 * 10 ^ 8 :=
sorry

end scientific_notation_of_858_million_l232_232194


namespace B_is_empty_l232_232526

def A : Set ℤ := {0}
def B : Set ℤ := {x | x > 8 ∧ x < 5}
def C : Set ℕ := {x | x - 1 = 0}
def D : Set ℤ := {x | x > 4}

theorem B_is_empty : B = ∅ := by
  sorry

end B_is_empty_l232_232526


namespace cubics_identity_l232_232838

variable (a b c x y z : ℝ)

theorem cubics_identity (X Y Z : ℝ)
  (h1 : X = a * x + b * y + c * z)
  (h2 : Y = a * y + b * z + c * x)
  (h3 : Z = a * z + b * x + c * y) :
  X^3 + Y^3 + Z^3 - 3 * X * Y * Z = 
  (x^3 + y^3 + z^3 - 3 * x * y * z) * (a^3 + b^3 + c^3 - 3 * a * b * c) :=
sorry

end cubics_identity_l232_232838


namespace polygon_RS_ST_sum_l232_232419

theorem polygon_RS_ST_sum
  (PQ RS ST: ℝ)
  (PQ_eq : PQ = 10)
  (QR_eq : QR = 7)
  (TU_eq : TU = 6)
  (polygon_area : PQ * QR = 70)
  (PQRSTU_area : 70 = 70) :
  RS + ST = 80 :=
by
  sorry

end polygon_RS_ST_sum_l232_232419


namespace ratio_of_new_time_to_previous_time_l232_232691

-- Given conditions
def distance : ℕ := 288
def initial_time : ℕ := 6
def new_speed : ℕ := 32

-- Question: Prove the ratio of the new time to the previous time is 3:2
theorem ratio_of_new_time_to_previous_time :
  (distance / new_speed) / initial_time = 3 / 2 :=
by
  sorry

end ratio_of_new_time_to_previous_time_l232_232691


namespace min_tip_percentage_l232_232785

noncomputable def meal_cost : ℝ := 37.25
noncomputable def total_paid : ℝ := 40.975
noncomputable def tip_percentage (P : ℝ) : Prop := P > 0 ∧ P < 15 ∧ (meal_cost + (P/100) * meal_cost = total_paid)

theorem min_tip_percentage : ∃ P : ℝ, tip_percentage P ∧ P = 10 := by
  sorry

end min_tip_percentage_l232_232785


namespace Greatest_Percentage_Difference_l232_232835

def max_percentage_difference (B W P : ℕ) : ℕ :=
  ((max B (max W P) - min B (min W P)) * 100) / (min B (min W P))

def January_B : ℕ := 6
def January_W : ℕ := 4
def January_P : ℕ := 5

def February_B : ℕ := 7
def February_W : ℕ := 5
def February_P : ℕ := 6

def March_B : ℕ := 7
def March_W : ℕ := 7
def March_P : ℕ := 7

def April_B : ℕ := 5
def April_W : ℕ := 6
def April_P : ℕ := 7

def May_B : ℕ := 3
def May_W : ℕ := 4
def May_P : ℕ := 2

theorem Greatest_Percentage_Difference :
  max_percentage_difference May_B May_W May_P >
  max (max_percentage_difference January_B January_W January_P)
      (max (max_percentage_difference February_B February_W February_P)
           (max (max_percentage_difference March_B March_W March_P)
                (max_percentage_difference April_B April_W April_P))) :=
by
  sorry

end Greatest_Percentage_Difference_l232_232835


namespace Sn_eq_S9_l232_232112

-- Definition of the arithmetic sequence sum formula.
def Sn (n a1 d : ℕ) : ℕ := (n * a1) + (n * (n - 1) / 2 * d)

theorem Sn_eq_S9 (a1 d : ℕ) (h1 : Sn 3 a1 d = 9) (h2 : Sn 6 a1 d = 36) : Sn 9 a1 d = 81 := by
  sorry

end Sn_eq_S9_l232_232112


namespace rational_with_smallest_absolute_value_is_zero_l232_232762

theorem rational_with_smallest_absolute_value_is_zero (r : ℚ) :
  (forall r : ℚ, |r| ≥ 0) →
  (forall r : ℚ, r ≠ 0 → |r| > 0) →
  |r| = 0 ↔ r = 0 := sorry

end rational_with_smallest_absolute_value_is_zero_l232_232762


namespace infinitely_many_digitally_divisible_integers_l232_232221

theorem infinitely_many_digitally_divisible_integers :
  ∀ n : ℕ, ∃ k : ℕ, k = (10 ^ (3 ^ n) - 1) / 9 ∧ (3 ^ n ∣ k) :=
by
  sorry

end infinitely_many_digitally_divisible_integers_l232_232221


namespace inequality_proof_l232_232441

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : -b > 0) (h3 : a > -b) (h4 : c < 0) : 
  a * (1 - c) > b * (c - 1) :=
sorry

end inequality_proof_l232_232441


namespace arithmetic_sequence_nth_term_l232_232446

theorem arithmetic_sequence_nth_term (x n : ℕ) (a1 a2 a3 : ℚ) (a_n : ℕ) :
  a1 = 3 * x - 5 ∧ a2 = 7 * x - 17 ∧ a3 = 4 * x + 3 ∧ a_n = 4033 →
  n = 641 :=
by sorry

end arithmetic_sequence_nth_term_l232_232446


namespace digit_A_in_comb_60_15_correct_l232_232224

-- Define the combination function
def comb (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The main theorem we want to prove
theorem digit_A_in_comb_60_15_correct : 
  ∃ (A : ℕ), (660 * 10^9 + A * 10^8 + B * 10^7 + 5 * 10^6 + A * 10^4 + 640 * 10^1 + A) = comb 60 15 ∧ A = 6 :=
by
  sorry

end digit_A_in_comb_60_15_correct_l232_232224


namespace company_spends_less_l232_232467

noncomputable def total_spending_reduction_in_dollars : ℝ :=
  let magazine_initial_cost := 840.00
  let online_resources_initial_cost_gbp := 960.00
  let exchange_rate := 1.40
  let mag_cut_percentage := 0.30
  let online_cut_percentage := 0.20

  let magazine_cost_cut := magazine_initial_cost * mag_cut_percentage
  let online_resource_cost_cut_gbp := online_resources_initial_cost_gbp * online_cut_percentage
  
  let new_magazine_cost := magazine_initial_cost - magazine_cost_cut
  let new_online_resource_cost_gbp := online_resources_initial_cost_gbp - online_resource_cost_cut_gbp

  let online_resources_initial_cost := online_resources_initial_cost_gbp * exchange_rate
  let new_online_resource_cost := new_online_resource_cost_gbp * exchange_rate

  let mag_cut_amount := magazine_initial_cost - new_magazine_cost
  let online_cut_amount := online_resources_initial_cost - new_online_resource_cost
  
  mag_cut_amount + online_cut_amount

theorem company_spends_less : total_spending_reduction_in_dollars = 520.80 :=
by
  sorry

end company_spends_less_l232_232467


namespace appropriate_investigation_method_l232_232774

theorem appropriate_investigation_method
  (volume_of_investigation_large : Prop)
  (no_need_for_comprehensive_investigation : Prop) :
  (∃ (method : String), method = "sampling investigation") :=
by
  sorry

end appropriate_investigation_method_l232_232774


namespace earliest_year_for_mismatched_pairs_l232_232176

def num_pairs (year : ℕ) : ℕ := 2 ^ (year - 2013)

def mismatched_pairs (pairs : ℕ) : ℕ := pairs * (pairs - 1)

theorem earliest_year_for_mismatched_pairs (year : ℕ) (h : year ≥ 2013) :
  (∃ pairs, (num_pairs year = pairs) ∧ (mismatched_pairs pairs ≥ 500)) → year = 2018 :=
by
  sorry

end earliest_year_for_mismatched_pairs_l232_232176


namespace find_value_of_a_l232_232786

variables (a : ℚ)

-- Definitions based on the conditions
def Brian_has_mar_bles : ℚ := 3 * a
def Caden_original_mar_bles : ℚ := 4 * Brian_has_mar_bles a
def Daryl_original_mar_bles : ℚ := 2 * Caden_original_mar_bles a
def Caden_after_give_10 : ℚ := Caden_original_mar_bles a - 10
def Daryl_after_receive_10 : ℚ := Daryl_original_mar_bles a + 10

-- Together Caden and Daryl now have 190 marbles
def together_mar_bles : ℚ := Caden_after_give_10 a + Daryl_after_receive_10 a

theorem find_value_of_a : together_mar_bles a = 190 → a = 95 / 18 :=
by
  sorry

end find_value_of_a_l232_232786


namespace geometric_sequence_example_l232_232984

theorem geometric_sequence_example
  (a : ℕ → ℝ)
  (h1 : ∀ n, 0 < a n)
  (h2 : ∃ r, ∀ n, a (n + 1) = r * a n)
  (h3 : Real.log (a 2) / Real.log 2 + Real.log (a 8) / Real.log 2 = 1) :
  a 3 * a 7 = 2 :=
sorry

end geometric_sequence_example_l232_232984


namespace problem1_l232_232245

theorem problem1 (f g : ℝ → ℝ) (m : ℝ) :
  (∀ x, 2 * |x + 3| ≥ m - 2 * |x + 7|) →
  (m ≤ 20) :=
by
  sorry

end problem1_l232_232245


namespace candle_burning_problem_l232_232158

theorem candle_burning_problem (burn_time_per_night_1h : ∀ n : ℕ, n = 8) 
                                (nightly_burn_rate : ∀ h : ℕ, h / 2 = 4) 
                                (total_nights : ℕ) 
                                (two_hour_nightly_burn : ∀ t : ℕ, t = 24) 
                                : ∃ candles : ℕ, candles = 6 := 
by {
  sorry
}

end candle_burning_problem_l232_232158


namespace mary_saw_total_snakes_l232_232715

theorem mary_saw_total_snakes :
  let breedingBalls := 3
  let snakesPerBall := 8
  let pairsOfSnakes := 6
  let snakesPerPair := 2
  let totalSnakes := breedingBalls * snakesPerBall + pairsOfSnakes * snakesPerPair
  totalSnakes = 36 :=
by
  /- Definitions -/ 
  let breedingBalls := 3
  let snakesPerBall := 8
  let pairsOfSnakes := 6
  let snakesPerPair := 2
  let totalSnakes := breedingBalls * snakesPerBall + pairsOfSnakes * snakesPerPair
  /- Main proof statement -/
  show totalSnakes = 36
  sorry

end mary_saw_total_snakes_l232_232715


namespace tan_add_pi_div_four_sine_cosine_ratio_l232_232197

-- Definition of the tangent function and trigonometric identities
variable {α : ℝ}

-- Given condition: tan(α) = 2
axiom tan_alpha_eq_2 : Real.tan α = 2

-- Problem 1: Prove that tan(α + π/4) = -3
theorem tan_add_pi_div_four : Real.tan ( α + Real.pi / 4 ) = -3 :=
by
  sorry

-- Problem 2: Prove that (6 * sin(α) + cos(α)) / (3 * sin(α) - cos(α)) = 13 / 5
theorem sine_cosine_ratio : 
  ( 6 * Real.sin α + Real.cos α ) / ( 3 * Real.sin α - Real.cos α ) = 13 / 5 :=
by
  sorry

end tan_add_pi_div_four_sine_cosine_ratio_l232_232197


namespace how_many_necklaces_given_away_l232_232937

-- Define the initial conditions
def initial_necklaces := 50
def broken_necklaces := 3
def bought_necklaces := 5
def final_necklaces := 37

-- Define the question proof statement
theorem how_many_necklaces_given_away : 
  (initial_necklaces - broken_necklaces + bought_necklaces - final_necklaces) = 15 :=
by sorry

end how_many_necklaces_given_away_l232_232937


namespace binom_10_3_l232_232032

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l232_232032


namespace find_z_value_l232_232424

theorem find_z_value (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0)
  (h1 : x = 2 + 1 / z)
  (h2 : z = 3 + 1 / x) : 
  z = (3 + Real.sqrt 15) / 2 :=
by
  sorry

end find_z_value_l232_232424


namespace reciprocal_of_neg_five_l232_232979

theorem reciprocal_of_neg_five : (1 / (-5 : ℝ)) = -1 / 5 := 
by
  sorry

end reciprocal_of_neg_five_l232_232979


namespace geom_seq_seventh_term_l232_232573

theorem geom_seq_seventh_term (a r : ℝ) (n : ℕ) (h1 : a = 2) (h2 : r^8 * a = 32) :
  a * r^6 = 128 :=
by
  sorry

end geom_seq_seventh_term_l232_232573


namespace kayak_rental_cost_l232_232720

theorem kayak_rental_cost (F : ℝ) (C : ℝ) (h1 : ∀ t : ℝ, C = F + 5 * t)
  (h2 : C = 30) : C = 45 :=
sorry

end kayak_rental_cost_l232_232720


namespace complement_intersection_l232_232393

open Set

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem complement_intersection :
  (U \ M) ∩ N = {3} :=
sorry

end complement_intersection_l232_232393


namespace original_recipe_pasta_l232_232238

noncomputable def pasta_per_person (total_pasta : ℕ) (total_people : ℕ) : ℚ :=
  total_pasta / total_people

noncomputable def original_pasta (pasta_per_person : ℚ) (people_served : ℕ) : ℚ :=
  pasta_per_person * people_served

theorem original_recipe_pasta (total_pasta : ℕ) (total_people : ℕ) (people_served : ℕ) (required_pasta : ℚ) :
  total_pasta = 10 → total_people = 35 → people_served = 7 → required_pasta = 2 →
  pasta_per_person total_pasta total_people * people_served = required_pasta :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end original_recipe_pasta_l232_232238


namespace greatest_N_exists_l232_232155

def is_condition_satisfied (N : ℕ) (xs : Fin N → ℤ) : Prop :=
  ∀ i j : Fin N, i ≠ j → ¬ (1111 ∣ ((xs i) * (xs i) - (xs i) * (xs j)))

theorem greatest_N_exists : ∃ N : ℕ, (∀ M : ℕ, (∀ xs : Fin M → ℤ, is_condition_satisfied M xs → M ≤ N)) ∧ N = 1000 :=
by
  sorry

end greatest_N_exists_l232_232155


namespace parabola_distance_to_focus_l232_232095

theorem parabola_distance_to_focus (P : ℝ × ℝ) (y_axis_dist : ℝ) (hx : P.1 = 4) (hy : P.2 ^ 2 = 32) :
  (P.1 - 2) ^ 2 + P.2 ^ 2 = 36 :=
by {
  sorry
}

end parabola_distance_to_focus_l232_232095


namespace distance_between_houses_l232_232865

theorem distance_between_houses (d d_JS d_QS : ℝ) (h1 : d_JS = 3) (h2 : d_QS = 1) :
  (2 ≤ d ∧ d ≤ 4) → d = 3 :=
sorry

end distance_between_houses_l232_232865


namespace batsman_average_after_12th_innings_l232_232295

theorem batsman_average_after_12th_innings (A : ℕ) (total_runs_11 : ℕ) (total_runs_12 : ℕ ) : 
  total_runs_11 = 11 * A → 
  total_runs_12 = total_runs_11 + 55 → 
  (total_runs_12 / 12 = A + 1) → 
  (A + 1) = 44 := 
by
  intros h1 h2 h3
  sorry

end batsman_average_after_12th_innings_l232_232295


namespace handshake_count_l232_232113

theorem handshake_count (n_twins: ℕ) (n_triplets: ℕ)
  (twin_pairs: ℕ) (triplet_groups: ℕ)
  (handshakes_twin : ∀ (x: ℕ), x = (n_twins - 2))
  (handshakes_triplet : ∀ (y: ℕ), y = (n_triplets - 3))
  (handshakes_cross_twins : ∀ (z: ℕ), z = 3*n_triplets / 4)
  (handshakes_cross_triplets : ∀ (w: ℕ), w = n_twins / 4) :
  2 * (n_twins * (n_twins -1 -1) / 2 + n_triplets * (n_triplets - 1 - 1) / 2 + n_twins * (3*n_triplets / 4) + n_triplets * (n_twins / 4)) / 2 = 804 := 
sorry

end handshake_count_l232_232113


namespace gel_pen_price_ratio_l232_232282

variable (x y b g T : ℝ)

-- Conditions from the problem
def condition1 : Prop := T = x * b + y * g
def condition2 : Prop := (x + y) * g = 4 * T
def condition3 : Prop := (x + y) * b = (1 / 2) * T

theorem gel_pen_price_ratio (h1 : condition1 x y b g T) (h2 : condition2 x y g T) (h3 : condition3 x y b T) :
  g = 8 * b :=
sorry

end gel_pen_price_ratio_l232_232282


namespace certain_event_l232_232073

-- Definitions of the events
def event1 : Prop := ∀ (P : ℝ), P ≠ 20.0
def event2 : Prop := ∀ (x : ℤ), x ≠ 105 ∧ x ≤ 100
def event3 : Prop := ∃ (r : ℝ), 0 ≤ r ∧ r ≤ 1 ∧ ¬(r = 0 ∨ r = 1)
def event4 (a b : ℝ) : Prop := ∃ (area : ℝ), area = a * b

-- Statement to prove that event4 is the only certain event
theorem certain_event (a b : ℝ) : (event4 a b) := 
by
  sorry

end certain_event_l232_232073


namespace equidistant_point_l232_232456

theorem equidistant_point (x y : ℝ) :
  (abs x = abs y) → (abs x = abs (x + y - 3) / (Real.sqrt 2)) → x = 1.5 :=
by {
  -- proof omitted
  sorry
}

end equidistant_point_l232_232456


namespace trade_and_unification_effects_l232_232615

theorem trade_and_unification_effects :
  let country_A_corn := 8
  let country_B_eggplants := 18
  let country_B_corn := 12
  let country_A_eggplants := 10
  
  -- Part (a): Absolute and comparative advantages
  (country_B_corn > country_A_corn) ∧ (country_B_eggplants > country_A_eggplants) ∧
  let opportunity_cost_A_eggplants := country_A_corn / country_A_eggplants
  let opportunity_cost_A_corn := country_A_eggplants / country_A_corn
  let opportunity_cost_B_eggplants := country_B_corn / country_B_eggplants
  let opportunity_cost_B_corn := country_B_eggplants / country_B_corn
  (opportunity_cost_B_eggplants < opportunity_cost_A_eggplants) ∧ (opportunity_cost_A_corn < opportunity_cost_B_corn) ∧

  -- Part (b): Volumes produced and consumed with trade
  let price := 1
  let income_A := country_A_corn * price
  let income_B := country_B_eggplants * price
  let consumption_A_eggplants := income_A / price / 2
  let consumption_A_corn := country_A_corn / 2
  let consumption_B_corn := income_B / price / 2
  let consumption_B_eggplants := country_B_eggplants / 2
  (consumption_A_eggplants = 4) ∧ (consumption_A_corn = 4) ∧
  (consumption_B_corn = 9) ∧ (consumption_B_eggplants = 9) ∧

  -- Part (c): Volumes after unification without trade
  let unified_eggplants := 18 - (1.5 * 4)
  let unified_corn := 8 + 4
  let total_unified_eggplants := unified_eggplants
  let total_unified_corn := unified_corn
  (total_unified_eggplants = 12) ∧ (total_unified_corn = 12) ->
  
  total_unified_eggplants = 12 ∧ total_unified_corn = 12 ∧
  (total_unified_eggplants < (consumption_A_eggplants + consumption_B_eggplants)) ∧
  (total_unified_corn < (consumption_A_corn + consumption_B_corn))
:= by
  -- Proof omitted
  sorry

end trade_and_unification_effects_l232_232615


namespace teams_worked_together_days_l232_232278

noncomputable def first_team_rate : ℝ := 1 / 12
noncomputable def second_team_rate : ℝ := 1 / 9
noncomputable def first_team_days : ℕ := 5
noncomputable def total_work : ℝ := 1
noncomputable def work_first_team_alone := first_team_rate * first_team_days

theorem teams_worked_together_days (x : ℝ) : work_first_team_alone + (first_team_rate + second_team_rate) * x = total_work → x = 3 := 
by
  sorry

end teams_worked_together_days_l232_232278


namespace mass_of_circle_is_one_l232_232975

variable (x y z : ℝ)

theorem mass_of_circle_is_one (h1 : 3 * y = 2 * x)
                              (h2 : 2 * y = x + 1)
                              (h3 : 5 * z = x + y)
                              (h4 : true) : z = 1 :=
sorry

end mass_of_circle_is_one_l232_232975


namespace range_of_m_l232_232712

-- Definitions of propositions
def is_circle (m : ℝ) : Prop :=
  ∃ x y : ℝ, (x - m)^2 + y^2 = 2 * m - m^2 ∧ 2 * m - m^2 > 0

def is_hyperbola_eccentricity_in_interval (m : ℝ) : Prop :=
  1 < Real.sqrt (1 + m / 5) ∧ Real.sqrt (1 + m / 5) < 2

-- Proving the main statement
theorem range_of_m (m : ℝ) (h1 : is_circle m ∨ is_hyperbola_eccentricity_in_interval m)
  (h2 : ¬ (is_circle m ∧ is_hyperbola_eccentricity_in_interval m)) : 2 ≤ m ∧ m < 15 :=
sorry

end range_of_m_l232_232712


namespace oliver_dishes_count_l232_232816

def total_dishes : ℕ := 42
def mango_salsa_dishes : ℕ := 5
def fresh_mango_dishes : ℕ := total_dishes / 6
def mango_jelly_dishes : ℕ := 2
def strawberry_dishes : ℕ := 3
def pineapple_dishes : ℕ := 5
def kiwi_dishes : ℕ := 4
def mango_dishes_oliver_picks_out : ℕ := 3

def total_mango_dishes : ℕ := mango_salsa_dishes + fresh_mango_dishes + mango_jelly_dishes
def mango_dishes_oliver_wont_eat : ℕ := total_mango_dishes - mango_dishes_oliver_picks_out
def max_strawberry_pineapple_dishes : ℕ := strawberry_dishes

def dishes_left_for_oliver : ℕ := total_dishes - mango_dishes_oliver_wont_eat - max_strawberry_pineapple_dishes

theorem oliver_dishes_count : dishes_left_for_oliver = 28 := 
by 
  sorry

end oliver_dishes_count_l232_232816


namespace n_times_s_l232_232089

noncomputable def f (x : ℝ) : ℝ := sorry

theorem n_times_s : (f 0 = 0 ∨ f 0 = 1) ∧
  (∀ (y : ℝ), f 0 = 0 → False) ∧
  (∀ (x y : ℝ), f x * f y - f (x * y) = x^2 + y^2) → 
  let n : ℕ := if f 0 = 0 then 1 else 1
  let s : ℝ := if f 0 = 0 then 0 else 1
  n * s = 1 :=
by
  sorry

end n_times_s_l232_232089


namespace number_is_37_5_l232_232214

theorem number_is_37_5 (y : ℝ) (h : 0.4 * y = 15) : y = 37.5 :=
sorry

end number_is_37_5_l232_232214


namespace value_of_4k_minus_1_l232_232578

theorem value_of_4k_minus_1 (k x y : ℝ)
  (h1 : x + y - 5 * k = 0)
  (h2 : x - y - 9 * k = 0)
  (h3 : 2 * x + 3 * y = 6) :
  4 * k - 1 = 2 :=
  sorry

end value_of_4k_minus_1_l232_232578


namespace john_age_proof_l232_232913

theorem john_age_proof (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end john_age_proof_l232_232913


namespace solve_fractional_eq_l232_232241

theorem solve_fractional_eq (x : ℝ) (hx1 : x ≠ 1 / 3) (hx2 : x ≠ -3) :
  (3 * x + 2) / (3 * x * x + 8 * x - 3) = (3 * x) / (3 * x - 1) ↔ 
  (x = -1 + (Real.sqrt 15) / 3) ∨ (x = -1 - (Real.sqrt 15) / 3) := 
by 
  sorry

end solve_fractional_eq_l232_232241


namespace ellipse_equation_l232_232695

theorem ellipse_equation (a b : ℝ) (A : ℝ × ℝ)
  (hA : A = (-3, 1.75))
  (he : 0.75 = Real.sqrt (a^2 - b^2) / a) 
  (hcond : (Real.sqrt (a^2 - b^2) / a) = 0.75) :
  (16 = a^2) ∧ (7 = b^2) :=
by
  have h1 : A = (-3, 1.75) := hA
  have h2 : Real.sqrt (a^2 - b^2) / a = 0.75 := hcond
  sorry

end ellipse_equation_l232_232695


namespace solve_eq_l232_232051

-- Defining the condition
def eq_condition (x : ℝ) : Prop := (x - 3) ^ 2 = x ^ 2 - 9

-- The statement we need to prove
theorem solve_eq (x : ℝ) (h : eq_condition x) : x = 3 :=
by
  sorry

end solve_eq_l232_232051


namespace projects_count_minimize_time_l232_232377

-- Define the conditions as given in the problem
def total_projects := 15
def energy_transfer_condition (x y : ℕ) : Prop := x = 2 * y - 3

-- Define question 1 as a proof problem
theorem projects_count (x y : ℕ) (h1 : x + y = total_projects) (h2 : energy_transfer_condition x y) :
  x = 9 ∧ y = 6 :=
by
  sorry

-- Define conditions for question 2
def average_time (energy_transfer_time leaping_gate_time : ℕ) (m n total_time : ℕ) : Prop :=
  total_time = 6 * m + 8 * n

-- Define additional conditions needed for Question 2 regarding time
theorem minimize_time (m n total_time : ℕ)
  (h1 : m + n = 10)
  (h2 : 10 - m > n)
  (h3 : average_time 6 8 m n total_time)
  (h4 : m = 6) :
  total_time = 68 :=
by
  sorry

end projects_count_minimize_time_l232_232377


namespace least_positive_integer_added_to_575_multiple_4_l232_232813

theorem least_positive_integer_added_to_575_multiple_4 :
  ∃ n : ℕ, n > 0 ∧ (575 + n) % 4 = 0 ∧ 
           ∀ m : ℕ, (m > 0 ∧ (575 + m) % 4 = 0) → n ≤ m := by
  sorry

end least_positive_integer_added_to_575_multiple_4_l232_232813


namespace solution_set_inequality_l232_232995

theorem solution_set_inequality (x : ℝ) : 4 * x < 3 * x + 2 → x < 2 :=
by
  intro h
  -- Add actual proof here, but for now; we use sorry
  sorry

end solution_set_inequality_l232_232995


namespace men_in_first_group_l232_232350

theorem men_in_first_group (M : ℕ) (h1 : 20 * 30 * (480 / (20 * 30)) = 480) (h2 : M * 15 * (120 / (M * 15)) = 120) :
  M = 10 :=
by sorry

end men_in_first_group_l232_232350


namespace fraction_addition_l232_232025

variable {a b : ℚ}
variable (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a / b = 3 / 4)

theorem fraction_addition (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a / b = 3 / 4) : (a + b) / b = 7 / 4 :=
  sorry

end fraction_addition_l232_232025


namespace parallelogram_angle_l232_232627

theorem parallelogram_angle (a b : ℝ) (h1 : a + b = 180) (h2 : a = b + 50) : b = 65 :=
by
  -- Proof would go here, but we're adding a placeholder
  sorry

end parallelogram_angle_l232_232627


namespace probability_without_order_knowledge_correct_probability_with_order_knowledge_correct_l232_232565

def TianJi_top {α : Type} [LinearOrder α] (a1 a2 : α) (b1 : α) : Prop :=
  a2 < b1 ∧ b1 < a1

def TianJi_middle {α : Type} [LinearOrder α] (a3 a2 : α) (b2 : α) : Prop :=
  a3 < b2 ∧ b2 < a2

def TianJi_bottom {α : Type} [LinearOrder α] (a3 : α) (b3 : α) : Prop :=
  b3 < a3

def without_order_knowledge_probability (a1 a2 a3 b1 b2 b3 : ℕ) 
  (h_top : TianJi_top a1 a2 b1) 
  (h_middle : TianJi_middle a3 a2 b2) 
  (h_bottom : TianJi_bottom a3 b3) : ℚ :=
  -- Formula for the probability of Tian Ji winning without knowing the order
  1 / 6

theorem probability_without_order_knowledge_correct (a1 a2 a3 b1 b2 b3 : ℕ) 
  (h_top : TianJi_top a1 a2 b1) 
  (h_middle : TianJi_middle a3 a2 b2) 
  (h_bottom : TianJi_bottom a3 b3) : 
  without_order_knowledge_probability a1 a2 a3 b1 b2 b3 h_top h_middle h_bottom = 1 / 6 :=
sorry

def with_order_knowledge_probability (a1 a2 a3 b1 b2 b3 : ℕ) 
  (h_top : TianJi_top a1 a2 b1) 
  (h_middle : TianJi_middle a3 a2 b2) 
  (h_bottom : TianJi_bottom a3 b3) : ℚ :=
  -- Formula for the probability of Tian Ji winning with specific group knowledge
  1 / 2

theorem probability_with_order_knowledge_correct (a1 a2 a3 b1 b2 b3 : ℕ) 
  (h_top : TianJi_top a1 a2 b1) 
  (h_middle : TianJi_middle a3 a2 b2) 
  (h_bottom : TianJi_bottom a3 b3) : 
  with_order_knowledge_probability a1 a2 a3 b1 b2 b3 h_top h_middle h_bottom = 1 / 2 :=
sorry

end probability_without_order_knowledge_correct_probability_with_order_knowledge_correct_l232_232565


namespace trajectory_equation_line_slope_is_constant_l232_232275

/-- Definitions for points A, B, and the moving point P -/ 
def pointA : ℝ × ℝ := (-2, 0)
def pointB : ℝ × ℝ := (2, 0)

/-- The condition that the product of the slopes is -3/4 -/
def slope_condition (P : ℝ × ℝ) : Prop :=
  let k_PA := P.2 / (P.1 + 2)
  let k_PB := P.2 / (P.1 - 2)
  k_PA * k_PB = -3 / 4

/-- The trajectory equation as a theorem to be proved -/
theorem trajectory_equation (P : ℝ × ℝ) (h : slope_condition P) : 
  P.2 ≠ 0 ∧ (P.1^2 / 4 + P.2^2 / 3 = 1) := 
sorry

/-- Additional conditions for the line l and points M, N -/ 
def line_l (k m : ℝ) (x : ℝ) : ℝ := k * x + m
def intersect_conditions (P M N : ℝ × ℝ) (k m : ℝ) : Prop :=
  (M.2 = line_l k m M.1) ∧ (N.2 = line_l k m N.1) ∧ 
  (P ≠ M ∧ P ≠ N) ∧ ((P.1 = 1) ∧ (P.2 = 3 / 2)) ∧ 
  (let k_PM := (M.2 - P.2) / (M.1 - P.1)
  let k_PN := (N.2 - P.2) / (N.1 - P.1)
  k_PM + k_PN = 0)

/-- The theorem to prove that the slope of line l is 1/2 -/
theorem line_slope_is_constant (P M N : ℝ × ℝ) (k m : ℝ) 
  (h1 : slope_condition P) 
  (h2 : intersect_conditions P M N k m) : 
  k = 1 / 2 := 
sorry

end trajectory_equation_line_slope_is_constant_l232_232275


namespace max_non_div_by_3_l232_232294

theorem max_non_div_by_3 (s : Finset ℕ) (h_len : s.card = 7) (h_prod : 3 ∣ s.prod id) : 
  ∃ n, n ≤ 6 ∧ ∀ x ∈ s, ¬ (3 ∣ x) → n = 6 :=
sorry

end max_non_div_by_3_l232_232294


namespace max_profit_correctness_l232_232726

noncomputable def daily_purchase_max_profit := 
  let purchase_price := 4.2
  let selling_price := 6
  let return_price := 1.2
  let days_sold_10kg := 10
  let days_sold_6kg := 20
  let days_in_month := 30
  let profit_function (x : ℝ) := 
    10 * x * (selling_price - purchase_price) + 
    days_sold_6kg * 6 * (selling_price - purchase_price) + 
    days_sold_6kg * (x - 6) * (return_price - purchase_price)
  (6, profit_function 6)

theorem max_profit_correctness : daily_purchase_max_profit = (6, 324) :=
  sorry

end max_profit_correctness_l232_232726


namespace arrangements_of_6_books_l232_232168

theorem arrangements_of_6_books : ∃ (n : ℕ), n = 720 ∧ n = Nat.factorial 6 :=
by
  use 720
  constructor
  · rfl
  · sorry

end arrangements_of_6_books_l232_232168


namespace sum_first_60_natural_numbers_l232_232603

theorem sum_first_60_natural_numbers : (60 * (60 + 1)) / 2 = 1830 := by
  sorry

end sum_first_60_natural_numbers_l232_232603


namespace leah_coins_value_l232_232819

theorem leah_coins_value : 
  ∃ (p n : ℕ), p + n = 15 ∧ n + 1 = p ∧ 5 * n + 1 * p = 43 := 
by
  sorry

end leah_coins_value_l232_232819


namespace total_size_of_game_is_880_l232_232164

-- Define the initial amount already downloaded
def initialAmountDownloaded : ℕ := 310

-- Define the download speed after the connection slows (in MB per minute)
def downloadSpeed : ℕ := 3

-- Define the remaining download time (in minutes)
def remainingDownloadTime : ℕ := 190

-- Define the total additional data to be downloaded in the remaining time (speed * time)
def additionalDataDownloaded : ℕ := downloadSpeed * remainingDownloadTime

-- Define the total size of the game as the sum of initial and additional data downloaded
def totalSizeOfGame : ℕ := initialAmountDownloaded + additionalDataDownloaded

-- State the theorem to prove
theorem total_size_of_game_is_880 : totalSizeOfGame = 880 :=
by 
  -- We provide no proof here; 'sorry' indicates an unfinished proof.
  sorry

end total_size_of_game_is_880_l232_232164


namespace price_difference_l232_232610

-- Definitions of conditions
def market_price : ℝ := 15400
def initial_sales_tax_rate : ℝ := 0.076
def new_sales_tax_rate : ℝ := 0.0667
def discount_rate : ℝ := 0.05
def handling_fee : ℝ := 200

-- Calculation of original sales tax
def original_sales_tax_amount : ℝ := market_price * initial_sales_tax_rate
-- Calculation of price after discount
def discount_amount : ℝ := market_price * discount_rate
def price_after_discount : ℝ := market_price - discount_amount
-- Calculation of new sales tax
def new_sales_tax_amount : ℝ := price_after_discount * new_sales_tax_rate
-- Calculation of total price with new sales tax and handling fee
def total_price_new : ℝ := price_after_discount + new_sales_tax_amount + handling_fee
-- Calculation of original total price with handling fee
def original_total_price : ℝ := market_price + original_sales_tax_amount + handling_fee

-- Expected difference in total cost
def expected_difference : ℝ := 964.60

-- Lean 4 statement to prove the difference
theorem price_difference :
  original_total_price - total_price_new = expected_difference :=
by
  sorry

end price_difference_l232_232610


namespace smallest_number_satisfying_conditions_l232_232949

theorem smallest_number_satisfying_conditions :
  ∃ b : ℕ, b ≡ 3 [MOD 5] ∧ b ≡ 2 [MOD 4] ∧ b ≡ 2 [MOD 6] ∧ b = 38 := 
by
  sorry

end smallest_number_satisfying_conditions_l232_232949


namespace final_price_set_l232_232328

variable (c ch s : ℕ)
variable (dc dtotal : ℚ)
variable (p_final : ℚ)

def coffee_price : ℕ := 6
def cheesecake_price : ℕ := 10
def sandwich_price : ℕ := 8
def coffee_discount : ℚ := 0.25 * 6
def final_discount : ℚ := 3

theorem final_price_set :
  p_final = (coffee_price - coffee_discount) + cheesecake_price + sandwich_price - final_discount :=
by
  sorry

end final_price_set_l232_232328


namespace emails_received_in_afternoon_l232_232487

theorem emails_received_in_afternoon (A : ℕ) 
  (h1 : 4 + (A - 3) = 9) : 
  A = 8 :=
by
  sorry

end emails_received_in_afternoon_l232_232487


namespace cube_root_squared_l232_232016

noncomputable def solve_for_x (x : ℝ) : Prop :=
  (x^(1/3))^2 = 81 → x = 729

theorem cube_root_squared (x : ℝ) :
  solve_for_x x :=
by
  sorry

end cube_root_squared_l232_232016


namespace range_of_m_l232_232386

open Set

noncomputable def A (m : ℝ) : Set (ℝ × ℝ) := 
  {p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ x^2 + m * x - y + 2 = 0} 

noncomputable def B : Set (ℝ × ℝ) := 
  {p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ x - y + 1 = 0}

theorem range_of_m (m : ℝ) : (A m ∩ B ≠ ∅) → (m ≤ -1 ∨ m ≥ 3) := 
sorry

end range_of_m_l232_232386


namespace max_students_l232_232457

open BigOperators

def seats_in_row (i : ℕ) : ℕ := 8 + 2 * i

def max_students_in_row (i : ℕ) : ℕ := 4 + i

def total_max_students : ℕ := ∑ i in Finset.range 15, max_students_in_row (i + 1)

theorem max_students (condition1 : true) : total_max_students = 180 :=
by
  sorry

end max_students_l232_232457


namespace largest_n_for_factored_polynomial_l232_232131

theorem largest_n_for_factored_polynomial :
  ∃ (n : ℤ), (∀ (A B : ℤ), 3 * A * B = 108 → n = 3 * B + A) ∧ n = 325 :=
by 
  sorry

end largest_n_for_factored_polynomial_l232_232131


namespace find_integers_a_b_c_l232_232830

theorem find_integers_a_b_c :
  ∃ (a b c : ℤ), (∀ (x : ℤ), (x - a) * (x - 8) + 4 = (x + b) * (x + c)) ∧ 
  (a = 20 ∨ a = 29) :=
 by {
      sorry 
}

end find_integers_a_b_c_l232_232830


namespace reality_show_duration_l232_232618

variable (x : ℕ)

theorem reality_show_duration :
  (5 * x + 10 = 150) → (x = 28) :=
by
  intro h
  sorry

end reality_show_duration_l232_232618


namespace bugs_initial_count_l232_232613

theorem bugs_initial_count (B : ℝ) 
  (h_spray : ∀ (b : ℝ), b * 0.8 = b * (4 / 5)) 
  (h_spiders : ∀ (s : ℝ), s * 7 = 12 * 7) 
  (h_initial_spray_spiders : ∀ (b : ℝ), b * 0.8 - (12 * 7) = 236) 
  (h_final_bugs : 320 / 0.8 = 400) : 
  B = 400 :=
sorry

end bugs_initial_count_l232_232613


namespace intersection_A_B_intersection_CR_A_B_l232_232507

noncomputable def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x < 7}
noncomputable def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}
noncomputable def CR_A : Set ℝ := {x : ℝ | x < 3} ∪ {x : ℝ | 7 ≤ x}

theorem intersection_A_B :
  A ∩ B = {x : ℝ | 3 ≤ x ∧ x < 7} :=
by
  sorry

theorem intersection_CR_A_B :
  CR_A ∩ B = ({x : ℝ | 2 < x ∧ x < 3} ∪ {x : ℝ | 7 ≤ x ∧ x < 10}) :=
by
  sorry

end intersection_A_B_intersection_CR_A_B_l232_232507


namespace inv_f_zero_l232_232765

noncomputable def f (a b x : Real) : Real := 1 / (2 * a * x + 3 * b)

theorem inv_f_zero (a b : Real) (ha : a ≠ 0) (hb : b ≠ 0) : f a b (1 / (3 * b)) = 0 :=
by 
  sorry

end inv_f_zero_l232_232765


namespace minimum_value_expr_l232_232186

theorem minimum_value_expr (x : ℝ) (h : x > 2) :
  ∃ y, y = (x^2 - 6 * x + 8) / (2 * x - 4) ∧ y = -1/2 := sorry

end minimum_value_expr_l232_232186


namespace correct_multiplication_value_l232_232257

theorem correct_multiplication_value (N : ℝ) (x : ℝ) : 
  (0.9333333333333333 = (N * x - N / 5) / (N * x)) → 
  x = 3 := 
by 
  sorry

end correct_multiplication_value_l232_232257


namespace arithmetic_sequence_25th_term_l232_232226

theorem arithmetic_sequence_25th_term (a1 a2 : ℕ) (d : ℕ) (n : ℕ) (h1 : a1 = 2) (h2 : a2 = 5) (h3 : d = a2 - a1) (h4 : n = 25) :
  a1 + (n - 1) * d = 74 :=
by
  sorry

end arithmetic_sequence_25th_term_l232_232226


namespace total_hamburgers_menu_l232_232389

def meat_patties_choices := 4
def condiment_combinations := 2 ^ 9

theorem total_hamburgers_menu :
  meat_patties_choices * condiment_combinations = 2048 :=
by
  sorry

end total_hamburgers_menu_l232_232389


namespace walking_times_relationship_l232_232982

theorem walking_times_relationship (x : ℝ) (h : x > 0) :
  (15 / x) - (15 / (x + 1)) = 1 / 2 :=
sorry

end walking_times_relationship_l232_232982


namespace find_f_3_l232_232476

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_3 (hf : ∀ y > 0, f ( (4 * y + 1) / (y + 1) ) = 1 / y) : f 3 = 0.5 :=
by
  sorry

end find_f_3_l232_232476


namespace expressions_positive_l232_232564

-- Definitions based on given conditions
def A := 2.5
def B := -0.8
def C := -2.2
def D := 1.1
def E := -3.1

-- The Lean statement to prove the necessary expressions are positive numbers.

theorem expressions_positive :
  (B + C) / E = 0.97 ∧
  B * D - A * C = 4.62 ∧
  C / (A * B) = 1.1 :=
by
  -- Assuming given conditions and steps to prove the theorem.
  sorry

end expressions_positive_l232_232564


namespace loss_percentage_l232_232729

theorem loss_percentage (CP SP : ℝ) (h_CP : CP = 1300) (h_SP : SP = 1040) :
  ((CP - SP) / CP) * 100 = 20 :=
by
  sorry

end loss_percentage_l232_232729


namespace compare_powers_l232_232160

theorem compare_powers :
  let a := 5 ^ 140
  let b := 3 ^ 210
  let c := 2 ^ 280
  c < a ∧ a < b := by
  -- Proof omitted
  sorry

end compare_powers_l232_232160


namespace range_of_abs_2z_minus_1_l232_232595

open Complex

theorem range_of_abs_2z_minus_1
  (z : ℂ)
  (h : abs (z + 2 - I) = 1) :
  abs (2 * z - 1) ∈ Set.Icc (Real.sqrt 29 - 2) (Real.sqrt 29 + 2) :=
sorry

end range_of_abs_2z_minus_1_l232_232595


namespace choose_copresidents_l232_232545

theorem choose_copresidents (total_members : ℕ) (departments : ℕ) (members_per_department : ℕ) 
    (h1 : total_members = 24) (h2 : departments = 4) (h3 : members_per_department = 6) :
    ∃ ways : ℕ, ways = 54 :=
by
  sorry

end choose_copresidents_l232_232545


namespace determine_p_l232_232537

-- Define the quadratic equation
def quadratic_eq (p x : ℝ) : ℝ := 3 * x^2 - 5 * (p - 1) * x + (p^2 + 2)

-- Define the conditions for the roots x1 and x2
def conditions (p x1 x2 : ℝ) : Prop :=
  quadratic_eq p x1 = 0 ∧
  quadratic_eq p x2 = 0 ∧
  x1 + 4 * x2 = 14

-- Define the theorem to prove the correct values of p
theorem determine_p (p : ℝ) (x1 x2 : ℝ) :
  conditions p x1 x2 → p = 742 / 127 ∨ p = 4 :=
by
  sorry

end determine_p_l232_232537


namespace men_l232_232065

namespace WagesProblem

def men_women_boys_equivalence (man woman boy : ℕ) : Prop :=
  9 * man = woman ∧ woman = 7 * boy

def total_earnings (man woman boy earnings : ℕ) : Prop :=
  (9 * man + woman + woman) = earnings ∧ earnings = 216

theorem men's_wages (man woman boy : ℕ) (h1 : men_women_boys_equivalence man woman boy) (h2 : total_earnings man woman 7 216) : 9 * man = 72 :=
sorry

end WagesProblem

end men_l232_232065


namespace initial_cows_l232_232408

theorem initial_cows {D C : ℕ}
  (h1 : C = 2 * D)
  (h2 : 161 = (3 * C) / 4 + D / 4) :
  C = 184 :=
by
  sorry

end initial_cows_l232_232408


namespace expression_result_l232_232096

-- We define the mixed number fractions as conditions
def mixed_num_1 := 2 + 1 / 2         -- 2 1/2
def mixed_num_2 := 3 + 1 / 3         -- 3 1/3
def mixed_num_3 := 4 + 1 / 4         -- 4 1/4
def mixed_num_4 := 1 + 1 / 6         -- 1 1/6

-- Here are their improper fractions
def improper_fraction_1 := 5 / 2     -- (2 + 1/2) converted to improper fraction
def improper_fraction_2 := 10 / 3    -- (3 + 1/3) converted to improper fraction
def improper_fraction_3 := 17 / 4    -- (4 + 1/4) converted to improper fraction
def improper_fraction_4 := 7 / 6     -- (1 + 1/6) converted to improper fraction

-- Define the problematic expression
def expression := (improper_fraction_1 - improper_fraction_2)^2 / (improper_fraction_3 + improper_fraction_4)

-- Statement of the simplified result
theorem expression_result : expression = 5 / 39 :=
by
  sorry

end expression_result_l232_232096


namespace annika_total_distance_l232_232587

/--
Annika hikes at a constant rate of 12 minutes per kilometer. She has hiked 2.75 kilometers
east from the start of a hiking trail when she realizes that she has to be back at the start
of the trail in 51 minutes. Prove that the total distance Annika hiked east is 3.5 kilometers.
-/
theorem annika_total_distance :
  (hike_rate : ℝ) = 12 → 
  (initial_distance_east : ℝ) = 2.75 → 
  (total_time : ℝ) = 51 → 
  (total_distance_east : ℝ) = 3.5 :=
by 
  intro hike_rate initial_distance_east total_time 
  sorry

end annika_total_distance_l232_232587


namespace swim_ratio_l232_232375

theorem swim_ratio
  (V_m : ℝ) (h1 : V_m = 4.5)
  (V_s : ℝ) (h2 : V_s = 1.5)
  (V_u : ℝ) (h3 : V_u = V_m - V_s)
  (V_d : ℝ) (h4 : V_d = V_m + V_s)
  (T_u T_d : ℝ) (h5 : T_u / T_d = V_d / V_u) :
  T_u / T_d = 2 :=
by {
  sorry
}

end swim_ratio_l232_232375


namespace laundry_loads_l232_232974

-- Conditions
def wash_time_per_load : ℕ := 45 -- in minutes
def dry_time_per_load : ℕ := 60 -- in minutes
def total_time : ℕ := 14 -- in hours

theorem laundry_loads (L : ℕ) 
  (h1 : total_time = 14)
  (h2 : total_time * 60 = L * (wash_time_per_load + dry_time_per_load)) :
  L = 8 :=
by
  sorry

end laundry_loads_l232_232974


namespace cubic_trinomial_degree_l232_232833

theorem cubic_trinomial_degree (n : ℕ) (P : ℕ → ℕ →  ℕ → Prop) : 
  (P n 5 4) → n = 3 := 
  sorry

end cubic_trinomial_degree_l232_232833


namespace annual_decrease_rate_l232_232420

theorem annual_decrease_rate :
  ∀ (P₀ P₂ : ℕ) (t : ℕ) (rate : ℝ),
    P₀ = 20000 → P₂ = 12800 → t = 2 → P₂ = P₀ * (1 - rate) ^ t → rate = 0.2 :=
by
sorry

end annual_decrease_rate_l232_232420


namespace intersection_M_N_l232_232228

def M : Set ℝ := { y | ∃ x, y = 2^x ∧ x > 0 }
def N : Set ℝ := { y | ∃ z, y = Real.log z ∧ z ∈ M }

theorem intersection_M_N : M ∩ N = { y | y > 1 } := sorry

end intersection_M_N_l232_232228


namespace find_parameters_l232_232040

theorem find_parameters (s h : ℝ) :
  (∀ (x y t : ℝ), (x = s + 3 * t) ∧ (y = 2 + h * t) ∧ (y = 5 * x - 7)) → (s = 9 / 5 ∧ h = 15) :=
by
  sorry

end find_parameters_l232_232040


namespace alex_score_l232_232769

theorem alex_score 
    (n : ℕ) -- number of students
    (avg_19 : ℕ) -- average score of first 19 students
    (avg_20 : ℕ) -- average score of all 20 students
    (h_n : n = 20) -- number of students is 20
    (h_avg_19 : avg_19 = 75) -- average score of first 19 students is 75
    (h_avg_20 : avg_20 = 76) -- average score of all 20 students is 76
  : ∃ alex_score : ℕ, alex_score = 95 := 
by
    sorry

end alex_score_l232_232769


namespace three_boys_in_shop_at_same_time_l232_232609

-- Definitions for the problem conditions
def boys : Type := Fin 7  -- Representing the 7 boys
def visits : Type := Fin 3  -- Each boy makes 3 visits

-- A structure representing a visit by a boy
structure Visit := (boy : boys) (visit_num : visits)

-- Meeting condition: Every pair of boys meets at the shop
def meets_at_shop (v1 v2 : Visit) : Prop :=
  v1.boy ≠ v2.boy  -- Ensure it's not the same boy (since we assume each pair meets)

-- The theorem to be proven
theorem three_boys_in_shop_at_same_time :
  ∃ (v1 v2 v3 : Visit), v1.boy ≠ v2.boy ∧ v2.boy ≠ v3.boy ∧ v1.boy ≠ v3.boy :=
sorry

end three_boys_in_shop_at_same_time_l232_232609


namespace remainder_7_mul_11_pow_24_plus_2_pow_24_mod_12_l232_232793

theorem remainder_7_mul_11_pow_24_plus_2_pow_24_mod_12 :
  (7 * 11 ^ 24 + 2 ^ 24) % 12 = 11 := by
sorry

end remainder_7_mul_11_pow_24_plus_2_pow_24_mod_12_l232_232793


namespace number_of_primes_in_interval_35_to_44_l232_232135

/--
The number of prime numbers in the interval [35, 44] is 3.
-/
theorem number_of_primes_in_interval_35_to_44 : 
  (Finset.filter Nat.Prime (Finset.Icc 35 44)).card = 3 := 
by
  sorry

end number_of_primes_in_interval_35_to_44_l232_232135


namespace aang_caught_7_fish_l232_232635

theorem aang_caught_7_fish (A : ℕ) (h_avg : (A + 5 + 12) / 3 = 8) : A = 7 :=
by
  sorry

end aang_caught_7_fish_l232_232635


namespace investment_C_l232_232623

-- Definitions of the given conditions
def investment_A : ℝ := 6300
def investment_B : ℝ := 4200
def total_profit : ℝ := 12700
def profit_A : ℝ := 3810

-- Defining the total investment, including C's investment
noncomputable def investment_total_including_C (C : ℝ) : ℝ := investment_A + investment_B + C

-- Proving the correct investment for C under the given conditions
theorem investment_C (C : ℝ) :
  (investment_A / investment_total_including_C C) = (profit_A / total_profit) → 
  C = 10500 :=
by
  -- Placeholder for the actual proof
  sorry

end investment_C_l232_232623


namespace number_one_number_two_number_three_number_four_number_five_number_six_number_seven_number_eight_number_nine_number_ten_number_eleven_number_twelve_number_thirteen_number_fourteen_number_fifteen_number_sixteen_number_seventeen_l232_232083

section FiveFives

def five : ℕ := 5

-- Definitions for each number 1 to 17 using five fives.
def one : ℕ := (five / five) * (five / five)
def two : ℕ := (five / five) + (five / five)
def three : ℕ := (five * five - five) / five
def four : ℕ := (five - five / five) * (five / five)
def five_num : ℕ := five + (five - five) * (five / five)
def six : ℕ := five + (five + five) / (five + five)
def seven : ℕ := five + (five * five - five^2) / five
def eight : ℕ := (five + five + five) / five + five
def nine : ℕ := five + (five - five / five)
def ten : ℕ := five + five
def eleven : ℕ := (55 - 55 / five) / five
def twelve : ℕ := five * (five - five / five) / five
def thirteen : ℕ := (five * five - five - five) / five + five
def fourteen : ℕ := five + five + five - (five / five)
def fifteen : ℕ := five + five + five
def sixteen : ℕ := five + five + five + (five / five)
def seventeen : ℕ := five + five + five + ((five / five) + (five / five))

-- Proof statements to be provided
theorem number_one : one = 1 := sorry
theorem number_two : two = 2 := sorry
theorem number_three : three = 3 := sorry
theorem number_four : four = 4 := sorry
theorem number_five : five_num = 5 := sorry
theorem number_six : six = 6 := sorry
theorem number_seven : seven = 7 := sorry
theorem number_eight : eight = 8 := sorry
theorem number_nine : nine = 9 := sorry
theorem number_ten : ten = 10 := sorry
theorem number_eleven : eleven = 11 := sorry
theorem number_twelve : twelve = 12 := sorry
theorem number_thirteen : thirteen = 13 := sorry
theorem number_fourteen : fourteen = 14 := sorry
theorem number_fifteen : fifteen = 15 := sorry
theorem number_sixteen : sixteen = 16 := sorry
theorem number_seventeen : seventeen = 17 := sorry

end FiveFives

end number_one_number_two_number_three_number_four_number_five_number_six_number_seven_number_eight_number_nine_number_ten_number_eleven_number_twelve_number_thirteen_number_fourteen_number_fifteen_number_sixteen_number_seventeen_l232_232083


namespace fraction_of_girls_correct_l232_232872

-- Define the total number of students in each school
def total_greenwood : ℕ := 300
def total_maplewood : ℕ := 240

-- Define the ratios of boys to girls
def ratio_boys_girls_greenwood := (3, 2)
def ratio_boys_girls_maplewood := (3, 4)

-- Define the number of boys and girls at Greenwood Middle School
def boys_greenwood (x : ℕ) : ℕ := 3 * x
def girls_greenwood (x : ℕ) : ℕ := 2 * x

-- Define the number of boys and girls at Maplewood Middle School
def boys_maplewood (y : ℕ) : ℕ := 3 * y
def girls_maplewood (y : ℕ) : ℕ := 4 * y

-- Define the total fractions
def total_girls (x y : ℕ) : ℚ := (girls_greenwood x + girls_maplewood y)
def total_students : ℚ := (total_greenwood + total_maplewood)

-- Main theorem to prove the fraction of girls at the event
theorem fraction_of_girls_correct (x y : ℕ)
  (h1 : 5 * x = total_greenwood)
  (h2 : 7 * y = total_maplewood) :
  (total_girls x y) / total_students = 5 / 7 :=
by
  sorry

end fraction_of_girls_correct_l232_232872


namespace center_of_tangent_circle_lies_on_hyperbola_l232_232501

open Real

def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4

def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 6*y + 24 = 0

noncomputable def locus_of_center : Set (ℝ × ℝ) :=
  {P | ∃ (r : ℝ), ∀ (x1 y1 x2 y2 : ℝ), circle1 x1 y1 ∧ circle2 x2 y2 → 
    dist P (x1, y1) = r + 2 ∧ dist P (x2, y2) = r + 1}

theorem center_of_tangent_circle_lies_on_hyperbola :
  ∀ P : ℝ × ℝ, P ∈ locus_of_center → ∃ (a b : ℝ) (F1 F2 : ℝ × ℝ), ∀ Q : ℝ × ℝ,
    dist Q F1 - dist Q F2 = 1 ∧ 
    dist F1 F2 = 5 ∧
    P ∈ {Q | dist Q F1 - dist Q F2 = 1} :=
sorry

end center_of_tangent_circle_lies_on_hyperbola_l232_232501


namespace ellipse_properties_l232_232018

-- Define the ellipse E with its given properties
def is_ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define properties related to the intersection points and lines
def intersects (l : ℝ → ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  l (-1) = 0 ∧ 
  is_ellipse x₁ (l x₁) ∧ 
  is_ellipse x₂ (l x₂) ∧ 
  y₁ = l x₁ ∧ 
  y₂ = l x₂

def perpendicular_lines (l1 l2 : ℝ → ℝ) : Prop :=
  ∀ x, l1 x * l2 x = -1

-- Define the main theorem
theorem ellipse_properties :
  (∀ (x y : ℝ), is_ellipse x y) →
  (∀ (l1 l2 : ℝ → ℝ) 
     (A B C D : ℝ × ℝ),
      intersects l1 A.1 A.2 B.1 B.2 → 
      intersects l2 C.1 C.2 D.1 D.2 → 
      perpendicular_lines l1 l2 → 
      12 * (|A.1 - B.1| + |C.1 - D.1|) = 7 * |A.1 - B.1| * |C.1 - D.1|) :=
by 
  sorry

end ellipse_properties_l232_232018


namespace eighth_binomial_term_l232_232741

theorem eighth_binomial_term :
  let n := 10
  let a := 2 * x
  let b := 1
  let k := 7
  (Nat.choose n k) * (a ^ k) * (b ^ (n - k)) = 960 * (x ^ 3) := by
  sorry

end eighth_binomial_term_l232_232741


namespace geometric_sequence_b_value_l232_232731

theorem geometric_sequence_b_value (b : ℝ) 
  (h1 : ∃ r : ℝ, 30 * r = b ∧ b * r = 9 / 4)
  (h2 : b > 0) : b = 3 * Real.sqrt 30 :=
by
  sorry

end geometric_sequence_b_value_l232_232731


namespace interest_earned_l232_232748

theorem interest_earned (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ) : 
  P = 2000 → r = 0.05 → n = 5 → 
  A = P * (1 + r)^n → 
  A - P = 552.56 :=
by
  intro hP hr hn hA
  rw [hP, hr, hn] at hA
  sorry

end interest_earned_l232_232748


namespace margaret_time_is_10_minutes_l232_232648

variable (time_billy_first_5_laps : ℕ)
variable (time_billy_next_3_laps : ℕ)
variable (time_billy_next_lap : ℕ)
variable (time_billy_final_lap : ℕ)
variable (time_difference : ℕ)

def billy_total_time := time_billy_first_5_laps + time_billy_next_3_laps + time_billy_next_lap + time_billy_final_lap

def margaret_total_time := billy_total_time + time_difference

theorem margaret_time_is_10_minutes :
  time_billy_first_5_laps = 120 ∧
  time_billy_next_3_laps = 240 ∧
  time_billy_next_lap = 60 ∧
  time_billy_final_lap = 150 ∧
  time_difference = 30 →
  margaret_total_time = 600 :=
by 
  sorry

end margaret_time_is_10_minutes_l232_232648


namespace carly_dog_count_l232_232111

theorem carly_dog_count (total_nails : ℕ) (three_legged_dogs : ℕ) (total_dogs : ℕ) 
  (h1 : total_nails = 164) 
  (h2 : three_legged_dogs = 3) 
  (h3 : total_dogs * 4 - three_legged_dogs = 41 - 3 * three_legged_dogs) 
  : total_dogs = 11 :=
sorry

end carly_dog_count_l232_232111


namespace factorial_expression_evaluation_l232_232619

theorem factorial_expression_evaluation : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + 5 * Nat.factorial 5 = 5760 := 
by 
  sorry

end factorial_expression_evaluation_l232_232619


namespace complex_sum_series_l232_232933

theorem complex_sum_series (ω : ℂ) (h1 : ω ^ 7 = 1) (h2 : ω ≠ 1) :
  ω ^ 16 + ω ^ 18 + ω ^ 20 + ω ^ 22 + ω ^ 24 + ω ^ 26 + ω ^ 28 + ω ^ 30 + 
  ω ^ 32 + ω ^ 34 + ω ^ 36 + ω ^ 38 + ω ^ 40 + ω ^ 42 + ω ^ 44 + ω ^ 46 +
  ω ^ 48 + ω ^ 50 + ω ^ 52 + ω ^ 54 = -1 :=
sorry

end complex_sum_series_l232_232933


namespace price_of_adult_ticket_l232_232042

theorem price_of_adult_ticket (total_payment : ℕ) (child_price : ℕ) (difference : ℕ) (children : ℕ) (adults : ℕ) (A : ℕ)
  (h1 : total_payment = 720) 
  (h2 : child_price = 8) 
  (h3 : difference = 25) 
  (h4 : children = 15)
  (h5 : adults = children + difference)
  (h6 : total_payment = children * child_price + adults * A) :
  A = 15 :=
by
  sorry

end price_of_adult_ticket_l232_232042


namespace find_values_l232_232429

theorem find_values (x : ℝ) (h : 2 * Real.cos x - 5 * Real.sin x = 3) :
  3 * Real.sin x + 2 * Real.cos x = ( -21 + 13 * Real.sqrt 145 ) / 58 ∨
  3 * Real.sin x + 2 * Real.cos x = ( -21 - 13 * Real.sqrt 145 ) / 58 := sorry

end find_values_l232_232429


namespace incircle_excircle_relation_l232_232612

variables {α : Type*} [LinearOrderedField α]

-- Defining the area expressions and radii
def area_inradius (a b c r : α) : α := (a + b + c) * r / 2
def area_exradius1 (a b c r1 : α) : α := (b + c - a) * r1 / 2
def area_exradius2 (a b c r2 : α) : α := (a + c - b) * r2 / 2
def area_exradius3 (a b c r3 : α) : α := (a + b - c) * r3 / 2

theorem incircle_excircle_relation (a b c r r1 r2 r3 Q : α) 
  (h₁ : Q = area_inradius a b c r)
  (h₂ : Q = area_exradius1 a b c r1)
  (h₃ : Q = area_exradius2 a b c r2)
  (h₄ : Q = area_exradius3 a b c r3) :
  1 / r = 1 / r1 + 1 / r2 + 1 / r3 :=
by 
  sorry

end incircle_excircle_relation_l232_232612


namespace probability_of_selecting_cooking_l232_232747

def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

theorem probability_of_selecting_cooking : (favorable_outcomes : ℚ) / total_courses = 1 / 4 := 
by 
  sorry

end probability_of_selecting_cooking_l232_232747


namespace ratio_brothers_sisters_boys_ratio_brothers_sisters_girls_l232_232671

variables (x y k t : ℕ)

theorem ratio_brothers_sisters_boys (h1 : (x+1) / y = k) (h2 : x / (y+1) = t) :
  (x / (y+1)) = t := 
by simp [h2]

theorem ratio_brothers_sisters_girls (h1 : (x+1) / y = k) (h2 : x / (y+1) = t) :
  ((x+1) / y) = k := 
by simp [h1]

#check ratio_brothers_sisters_boys    -- Just for verification
#check ratio_brothers_sisters_girls   -- Just for verification

end ratio_brothers_sisters_boys_ratio_brothers_sisters_girls_l232_232671


namespace total_area_to_paint_proof_l232_232121

def barn_width : ℝ := 15
def barn_length : ℝ := 20
def barn_height : ℝ := 8
def door_width : ℝ := 3
def door_height : ℝ := 7
def window_width : ℝ := 2
def window_height : ℝ := 4

noncomputable def wall_area (width length height : ℝ) : ℝ := 2 * (width * height + length * height)
noncomputable def door_area (width height : ℝ) (num: ℕ) : ℝ := width * height * num
noncomputable def window_area (width height : ℝ) (num: ℕ) : ℝ := width * height * num

noncomputable def total_area_to_paint : ℝ := 
  let total_wall_area := wall_area barn_width barn_length barn_height
  let total_door_area := door_area door_width door_height 2
  let total_window_area := window_area window_width window_height 3
  let net_wall_area := total_wall_area - total_door_area - total_window_area
  let ceiling_floor_area := barn_width * barn_length * 2
  net_wall_area * 2 + ceiling_floor_area

theorem total_area_to_paint_proof : total_area_to_paint = 1588 := by
  sorry

end total_area_to_paint_proof_l232_232121


namespace boys_tried_out_l232_232139

theorem boys_tried_out (B : ℕ) (girls : ℕ) (called_back : ℕ) (not_cut : ℕ) (total_tryouts : ℕ) 
  (h1 : girls = 39)
  (h2 : called_back = 26)
  (h3 : not_cut = 17)
  (h4 : total_tryouts = girls + B)
  (h5 : total_tryouts = called_back + not_cut) : 
  B = 4 := 
by
  sorry

end boys_tried_out_l232_232139


namespace norma_cards_lost_l232_232875

theorem norma_cards_lost (original_cards : ℕ) (current_cards : ℕ) (cards_lost : ℕ)
  (h1 : original_cards = 88) (h2 : current_cards = 18) :
  original_cards - current_cards = cards_lost →
  cards_lost = 70 := by
  sorry

end norma_cards_lost_l232_232875


namespace circle_condition_k_l232_232930

theorem circle_condition_k (k : ℝ) : 
  (∃ (h : ℝ), (x^2 + y^2 - 2*x + 6*y + k = 0)) → k < 10 :=
by
  sorry

end circle_condition_k_l232_232930


namespace interval_f_has_two_roots_l232_232852

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x^2 + a * x

theorem interval_f_has_two_roots (a : ℝ) : (∀ x : ℝ, f a x = 0 → ∃ u v : ℝ, u ≠ v ∧ f a u = 0 ∧ f a v = 0) ↔ 0 < a ∧ a < 1 / 8 := 
sorry

end interval_f_has_two_roots_l232_232852


namespace family_members_before_baby_l232_232329

theorem family_members_before_baby 
  (n T : ℕ)
  (h1 : T = 17 * n)
  (h2 : (T + 3 * n + 2) / (n + 1) = 17)
  (h3 : 2 = 2) : n = 5 :=
sorry

end family_members_before_baby_l232_232329


namespace subtract_base3_sum_eq_result_l232_232141

theorem subtract_base3_sum_eq_result :
  let a := 10 -- interpreted as 10_3
  let b := 1101 -- interpreted as 1101_3
  let c := 2102 -- interpreted as 2102_3
  let d := 212 -- interpreted as 212_3
  let sum := 1210 -- interpreted as the base 3 sum of a + b + c
  let result := 1101 -- interpreted as the final base 3 result
  sum - d = result :=
by sorry

end subtract_base3_sum_eq_result_l232_232141


namespace problem_statement_l232_232775

variable (a b : ℝ) (f : ℝ → ℝ)
variable (h1 : ∀ x > 0, f x = Real.log x / Real.log 3)
variable (h2 : b = 9 * a)

theorem problem_statement : f a - f b = -2 := by
  sorry

end problem_statement_l232_232775


namespace more_customers_after_lunch_rush_l232_232755

-- Definitions for conditions
def initial_customers : ℝ := 29.0
def added_customers : ℝ := 20.0
def total_customers : ℝ := 83.0

-- The number of additional customers that came in after the lunch rush
def additional_customers (initial additional total : ℝ) : ℝ :=
  total - (initial + additional)

-- Statement to prove
theorem more_customers_after_lunch_rush :
  additional_customers initial_customers added_customers total_customers = 34.0 :=
by
  sorry

end more_customers_after_lunch_rush_l232_232755


namespace simplify_expression_l232_232676

variable (a b : ℝ)

theorem simplify_expression (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a ≠ b) :
  (3 * (a^2 + a * b + b^2) / (4 * (a + b))) * (2 * (a^2 - b^2) / (9 * (a^3 - b^3))) = 
  1 / 6 := 
by
  -- Placeholder for proof steps
  sorry

end simplify_expression_l232_232676


namespace volume_of_region_l232_232315

theorem volume_of_region : 
  ∀ (x y z : ℝ),
  abs (x + y + z) + abs (x - y + z) ≤ 10 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 → 
  ∃ V : ℝ, V = 62.5 :=
  sorry

end volume_of_region_l232_232315


namespace vicente_total_spent_l232_232359

def kilograms_of_rice := 5
def cost_per_kilogram_of_rice := 2
def pounds_of_meat := 3
def cost_per_pound_of_meat := 5

def total_spent := kilograms_of_rice * cost_per_kilogram_of_rice + pounds_of_meat * cost_per_pound_of_meat

theorem vicente_total_spent : total_spent = 25 := 
by
  sorry -- Proof would go here

end vicente_total_spent_l232_232359


namespace total_seeds_correct_l232_232911

def seeds_per_bed : ℕ := 6
def flower_beds : ℕ := 9
def total_seeds : ℕ := seeds_per_bed * flower_beds

theorem total_seeds_correct : total_seeds = 54 := by
  sorry

end total_seeds_correct_l232_232911


namespace bird_wings_l232_232616

theorem bird_wings (birds wings_per_bird : ℕ) (h1 : birds = 13) (h2 : wings_per_bird = 2) : birds * wings_per_bird = 26 := by
  sorry

end bird_wings_l232_232616


namespace isosceles_triangle_l232_232752

def shape_of_triangle (A B C : Real) (h : 2 * Real.sin A * Real.cos B = Real.sin C) : Prop :=
  A = B

theorem isosceles_triangle {A B C : Real} (h : 2 * Real.sin A * Real.cos B = Real.sin C) :
  shape_of_triangle A B C h := 
  sorry

end isosceles_triangle_l232_232752


namespace average_speed_x_to_z_l232_232845

theorem average_speed_x_to_z 
  (d : ℝ)
  (h1 : d > 0)
  (distance_xy : ℝ := 2 * d)
  (distance_yz : ℝ := d)
  (speed_xy : ℝ := 100)
  (speed_yz : ℝ := 75)
  (total_distance : ℝ := distance_xy + distance_yz)
  (time_xy : ℝ := distance_xy / speed_xy)
  (time_yz : ℝ := distance_yz / speed_yz)
  (total_time : ℝ := time_xy + time_yz) :
  total_distance / total_time = 90 :=
by
  sorry

end average_speed_x_to_z_l232_232845


namespace find_theta_l232_232374

open Real

theorem find_theta (theta : ℝ) : sin theta = -1/3 ∧ -π < theta ∧ theta < -π / 2 ↔ theta = -π - arcsin (-1 / 3) :=
by
  sorry

end find_theta_l232_232374


namespace simplify_expression_l232_232931

theorem simplify_expression (x : ℝ) : x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3 * x^3 - 5 * x^2 + 12 * x + 2 := by
  sorry

end simplify_expression_l232_232931


namespace average_speed_is_correct_l232_232709

namespace CyclistTrip

-- Define the trip parameters
def distance_north := 10 -- kilometers
def speed_north := 15 -- kilometers per hour
def rest_time := 10 / 60 -- hours
def distance_south := 10 -- kilometers
def speed_south := 20 -- kilometers per hour

-- The total trip distance
def total_distance := distance_north + distance_south -- kilometers

-- Calculate the time for each segment
def time_north := distance_north / speed_north -- hours
def time_south := distance_south / speed_south -- hours

-- Total time for the trip
def total_time := time_north + rest_time + time_south -- hours

-- Calculate the average speed
def average_speed := total_distance / total_time -- kilometers per hour

theorem average_speed_is_correct : average_speed = 15 := by
  sorry

end CyclistTrip

end average_speed_is_correct_l232_232709


namespace num_integer_pairs_satisfying_m_plus_n_eq_mn_l232_232285

theorem num_integer_pairs_satisfying_m_plus_n_eq_mn : 
  ∃ (m n : ℤ), (m + n = m * n) ∧ ∀ (m n : ℤ), (m + n = m * n) → 
  (m = 0 ∧ n = 0) ∨ (m = 2 ∧ n = 2) :=
by
  sorry

end num_integer_pairs_satisfying_m_plus_n_eq_mn_l232_232285


namespace youtube_likes_l232_232792

theorem youtube_likes (L D : ℕ) 
  (h1 : D = (1 / 2 : ℝ) * L + 100)
  (h2 : D + 1000 = 2600) : 
  L = 3000 := 
by
  sorry

end youtube_likes_l232_232792


namespace simplify_expression_l232_232062

theorem simplify_expression : 1 - (1 / (1 + Real.sqrt 2)) + (1 / (1 - Real.sqrt 2)) = 1 - 2 * Real.sqrt 2 := by
  sorry

end simplify_expression_l232_232062


namespace total_beads_in_necklace_l232_232776

noncomputable def amethyst_beads : ℕ := 7
noncomputable def amber_beads : ℕ := 2 * amethyst_beads
noncomputable def turquoise_beads : ℕ := 19
noncomputable def total_beads : ℕ := amethyst_beads + amber_beads + turquoise_beads

theorem total_beads_in_necklace : total_beads = 40 := by
  sorry

end total_beads_in_necklace_l232_232776


namespace carol_blocks_l232_232750

theorem carol_blocks (initial_blocks lost_blocks final_blocks : ℕ) 
  (h_initial : initial_blocks = 42) 
  (h_lost : lost_blocks = 25) : 
  final_blocks = initial_blocks - lost_blocks → final_blocks = 17 := by
  sorry

end carol_blocks_l232_232750


namespace vacation_trip_l232_232597

theorem vacation_trip (airbnb_cost : ℕ) (car_rental_cost : ℕ) (share_per_person : ℕ) (total_people : ℕ) :
  airbnb_cost = 3200 → car_rental_cost = 800 → share_per_person = 500 → airbnb_cost + car_rental_cost / share_per_person = 8 :=
by
  intros h1 h2 h3
  sorry

end vacation_trip_l232_232597


namespace parallel_vectors_m_value_l232_232010

theorem parallel_vectors_m_value :
  ∀ (m : ℝ), (∀ k : ℝ, (1 : ℝ) = k * m ∧ (-2) = k * (-1)) -> m = (1 / 2) :=
by
  intros m h
  sorry

end parallel_vectors_m_value_l232_232010


namespace max_abs_f_le_f0_f1_l232_232058

noncomputable def f (a b x : ℝ) : ℝ := 3 * a * x^2 - 2 * (a + b) * x + b

theorem max_abs_f_le_f0_f1 (a b : ℝ) (h : 0 < a) (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) :
  |f a b x| ≤ max (|f a b 0|) (|f a b 1|) :=
sorry

end max_abs_f_le_f0_f1_l232_232058


namespace spider_moves_away_from_bee_l232_232966

noncomputable def bee : ℝ × ℝ := (14, 5)
noncomputable def spider_line (x : ℝ) : ℝ := -3 * x + 25
noncomputable def perpendicular_line (x : ℝ) : ℝ := (1 / 3) * x + 14 / 3

theorem spider_moves_away_from_bee : ∃ (c d : ℝ), 
  (d = spider_line c) ∧ (d = perpendicular_line c) ∧ c + d = 13.37 := 
sorry

end spider_moves_away_from_bee_l232_232966


namespace total_action_figures_l232_232664

def action_figures_per_shelf : ℕ := 11
def number_of_shelves : ℕ := 4

theorem total_action_figures : action_figures_per_shelf * number_of_shelves = 44 := by
  sorry

end total_action_figures_l232_232664


namespace find_unknown_rate_l232_232369

def blankets_cost (num : ℕ) (rate : ℕ) (discount_tax : ℕ) (is_discount : Bool) : ℕ :=
  if is_discount then rate * (100 - discount_tax) / 100 * num
  else (rate * (100 + discount_tax) / 100) * num

def total_cost := blankets_cost 3 100 10 true +
                  blankets_cost 4 150 0 false +
                  blankets_cost 3 200 20 false

def avg_cost (total : ℕ) (num : ℕ) : ℕ :=
  total / num

theorem find_unknown_rate
  (unknown_rate : ℕ)
  (h1 : total_cost + 2 * unknown_rate = 1800)
  (h2 : avg_cost (total_cost + 2 * unknown_rate) 12 = 150) :
  unknown_rate = 105 :=
by
  sorry

end find_unknown_rate_l232_232369


namespace recycling_money_l232_232035

theorem recycling_money (cans_per_unit : ℕ) (payment_per_unit_cans : ℝ) 
  (newspapers_per_unit : ℕ) (payment_per_unit_newspapers : ℝ) 
  (total_cans : ℕ) (total_newspapers : ℕ) : 
  cans_per_unit = 12 → payment_per_unit_cans = 0.50 → 
  newspapers_per_unit = 5 → payment_per_unit_newspapers = 1.50 → 
  total_cans = 144 → total_newspapers = 20 → 
  (total_cans / cans_per_unit) * payment_per_unit_cans + 
  (total_newspapers / newspapers_per_unit) * payment_per_unit_newspapers = 12 := 
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  done

end recycling_money_l232_232035


namespace rectangle_ratio_l232_232552

theorem rectangle_ratio (s y x : ℝ) 
  (inner_square_area outer_square_area : ℝ) 
  (h1 : inner_square_area = s^2)
  (h2 : outer_square_area = 9 * inner_square_area)
  (h3 : outer_square_area = (3 * s)^2)
  (h4 : s + 2 * y = 3 * s)
  (h5 : x + y = 3 * s)
  : x / y = 2 := 
by
  -- Proof steps will go here
  sorry

end rectangle_ratio_l232_232552


namespace pie_eating_contest_l232_232247

theorem pie_eating_contest :
  (7 / 8 : ℚ) - (5 / 6 : ℚ) = (1 / 24 : ℚ) :=
sorry

end pie_eating_contest_l232_232247


namespace smallest_checkered_rectangle_area_l232_232417

def even (n: ℕ) : Prop := n % 2 = 0

-- Both figure types are present and areas of these types are 1 and 2 respectively
def isValidPieceComposition (a b : ℕ) : Prop :=
  ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ m * 1 + n * 2 = a * b

theorem smallest_checkered_rectangle_area :
  ∀ a b : ℕ, even a → even b → isValidPieceComposition a b → a * b ≥ 40 := 
by
  intro a b a_even b_even h_valid
  sorry

end smallest_checkered_rectangle_area_l232_232417


namespace number_of_family_members_l232_232945

-- Define the number of legs for each type of animal.
def bird_legs : ℕ := 2
def dog_legs : ℕ := 4
def cat_legs : ℕ := 4

-- Define the number of animals.
def birds : ℕ := 4
def dogs : ℕ := 3
def cats : ℕ := 18

-- Define the total number of legs of all animals.
def total_animal_feet : ℕ := birds * bird_legs + dogs * dog_legs + cats * cat_legs

-- Define the total number of heads of all animals.
def total_animal_heads : ℕ := birds + dogs + cats

-- Main theorem: If the total number of feet in the house is 74 more than the total number of heads, find the number of family members.
theorem number_of_family_members (F : ℕ) (h : total_animal_feet + 2 * F = total_animal_heads + F + 74) : F = 7 :=
by
  sorry

end number_of_family_members_l232_232945


namespace initial_number_of_men_l232_232916

theorem initial_number_of_men (n : ℕ) (A : ℕ)
  (h1 : 2 * n = 16)
  (h2 : 60 - 44 = 16)
  (h3 : 60 = 2 * 30)
  (h4 : 44 = 21 + 23) :
  n = 8 :=
by
  sorry

end initial_number_of_men_l232_232916


namespace xy_y_sq_eq_y_sq_3y_12_l232_232881

variable (x y : ℝ)

theorem xy_y_sq_eq_y_sq_3y_12 (h : x * (x + y) = x^2 + 3 * y + 12) : 
  x * y + y^2 = y^2 + 3 * y + 12 := 
sorry

end xy_y_sq_eq_y_sq_3y_12_l232_232881


namespace snowfall_rate_in_Hamilton_l232_232430

theorem snowfall_rate_in_Hamilton 
  (initial_depth_Kingston : ℝ := 12.1)
  (rate_Kingston : ℝ := 2.6)
  (initial_depth_Hamilton : ℝ := 18.6)
  (duration : ℕ := 13)
  (final_depth_equal : initial_depth_Kingston + rate_Kingston * duration = initial_depth_Hamilton + duration * x)
  (x : ℝ) :
  x = 2.1 :=
sorry

end snowfall_rate_in_Hamilton_l232_232430


namespace expr_is_irreducible_fraction_l232_232877

def a : ℚ := 3 / 2015
def b : ℚ := 11 / 2016

noncomputable def expr : ℚ := 
  (6 + a) * (8 + b) - (11 - a) * (3 - b) - 12 * a

theorem expr_is_irreducible_fraction : expr = 11 / 112 := by
  sorry

end expr_is_irreducible_fraction_l232_232877


namespace range_of_a_l232_232637

theorem range_of_a {a : ℝ} : (∀ x : ℝ, (x^2 + 2 * (a + 1) * x + a^2 - 1 = 0) → (x = 0 ∨ x = -4)) → (a = 1 ∨ a ≤ -1) := 
by {
  sorry
}

end range_of_a_l232_232637


namespace planting_equation_l232_232905

def condition1 (x : ℕ) : ℕ := 5 * x + 3
def condition2 (x : ℕ) : ℕ := 6 * x - 4

theorem planting_equation (x : ℕ) : condition1 x = condition2 x := by
  sorry

end planting_equation_l232_232905


namespace gianna_saved_for_365_days_l232_232503

-- Define the total amount saved and the amount saved each day
def total_amount_saved : ℕ := 14235
def amount_saved_each_day : ℕ := 39

-- Define the problem statement to prove the number of days saved
theorem gianna_saved_for_365_days :
  (total_amount_saved / amount_saved_each_day) = 365 :=
sorry

end gianna_saved_for_365_days_l232_232503


namespace weeds_in_rice_l232_232212

-- Define the conditions
def total_weight_of_rice := 1536
def sample_size := 224
def weeds_in_sample := 28

-- State the main proof
theorem weeds_in_rice (total_rice : ℕ) (sample_size : ℕ) (weeds_sample : ℕ) 
  (H1 : total_rice = total_weight_of_rice) (H2 : sample_size = sample_size) (H3 : weeds_sample = weeds_in_sample) :
  total_rice * weeds_sample / sample_size = 192 := 
by
  -- Evidence of calculations and external assumptions, translated initial assumptions into mathematical format
  sorry

end weeds_in_rice_l232_232212


namespace find_y_l232_232349

theorem find_y (x y : ℝ) (h₁ : x = 51) (h₂ : x^3 * y - 2 * x^2 * y + x * y = 51000) : y = 2 / 5 := by
  sorry

end find_y_l232_232349


namespace find_bags_l232_232518

theorem find_bags (x : ℕ) : 10 + x + 7 = 20 → x = 3 :=
by
  sorry

end find_bags_l232_232518


namespace abs_eq_five_l232_232665

theorem abs_eq_five (x : ℝ) : |x| = 5 → (x = 5 ∨ x = -5) :=
by
  sorry

end abs_eq_five_l232_232665


namespace range_of_a_l232_232807

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - x
noncomputable def g (x : ℝ) : ℝ := Real.log x
noncomputable def h (a x : ℝ) : ℝ := f a x - g x
noncomputable def k (x : ℝ) : ℝ := (Real.log x + x) / x^2

theorem range_of_a (a : ℝ) (h_zero : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ h a x₁ = 0 ∧ h a x₂ = 0) :
  0 < a ∧ a < 1 :=
sorry

end range_of_a_l232_232807


namespace area_change_factor_l232_232920

theorem area_change_factor (k b : ℝ) (hk : 0 < k) (hb : 0 < b) :
  let S1 := (b * b) / (2 * k)
  let S2 := (b * b) / (16 * k)
  S1 / S2 = 8 :=
by
  sorry

end area_change_factor_l232_232920


namespace maximum_value_of_d_l232_232645

theorem maximum_value_of_d (a b c d : ℝ) 
  (h₁ : a + b + c + d = 10)
  (h₂ : ab + ac + ad + bc + bd + cd = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end maximum_value_of_d_l232_232645


namespace squares_expression_l232_232972

theorem squares_expression (a : ℕ) : 
  a^2 + 5*a + 7 = (a+3) * (a+2)^2 + (a+2) * 1^2 := 
by
  sorry

end squares_expression_l232_232972


namespace remainder_geometric_series_sum_l232_232849

/-- Define the sum of the geometric series. --/
def geometric_series_sum (n : ℕ) : ℕ :=
  (13^(n+1) - 1) / 12

/-- The given geometric series. --/
def series_sum := geometric_series_sum 1004

/-- Define the modulo operation. --/
def mod_op (a b : ℕ) := a % b

/-- The main statement to prove. --/
theorem remainder_geometric_series_sum :
  mod_op series_sum 1000 = 1 :=
sorry

end remainder_geometric_series_sum_l232_232849


namespace problem1_problem2_l232_232205

theorem problem1 (α : ℝ) (hα : Real.tan α = 2) :
    (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3 := 
sorry

theorem problem2 (α : ℝ) (hα : Real.tan α = 2) :
    (Real.sin (↑(π/2) + α) * Real.cos (↑(5*π/2) - α) * Real.tan (↑(-π) + α)) / 
    (Real.tan (↑(7*π) - α) * Real.sin (↑π + α)) = Real.cos α := 
sorry

end problem1_problem2_l232_232205


namespace find_number_l232_232057

variable (x : ℝ)

theorem find_number : ((x * 5) / 2.5 - 8 * 2.25 = 5.5) -> x = 11.75 :=
by
  intro h
  sorry

end find_number_l232_232057


namespace factor_by_resultant_is_three_l232_232379

theorem factor_by_resultant_is_three
  (x : ℕ) (f : ℕ) (h1 : x = 7)
  (h2 : (2 * x + 9) * f = 69) :
  f = 3 :=
sorry

end factor_by_resultant_is_three_l232_232379


namespace selling_price_eq_l232_232780

theorem selling_price_eq (cp sp L : ℕ) (h_cp: cp = 47) (h_L : L = cp - 40) (h_profit_loss_eq : sp - cp = L) :
  sp = 54 :=
by
  sorry

end selling_price_eq_l232_232780


namespace abs_diff_squares_l232_232021

theorem abs_diff_squares (a b : ℤ) (ha : a = 103) (hb : b = 97) : |a^2 - b^2| = 1200 :=
by
  sorry

end abs_diff_squares_l232_232021


namespace int_pairs_satisfy_eq_l232_232738

theorem int_pairs_satisfy_eq (x y : ℤ) : (x^2 = y^2 + 2 * y + 13) ↔ ((x = 4 ∧ y = 1) ∨ (x = -4 ∧ y = -5)) :=
by 
  sorry

end int_pairs_satisfy_eq_l232_232738


namespace solution_set_of_inequality_l232_232939

def f : Int → Int
| -1 => -1
| 0 => -1
| 1 => 1
| _ => 0 -- Assuming this default as table provided values only for -1, 0, 1

def g : Int → Int
| -1 => 1
| 0 => 1
| 1 => -1
| _ => 0 -- Assuming this default as table provided values only for -1, 0, 1

theorem solution_set_of_inequality :
  {x | f (g x) > 0} = { -1, 0 } :=
by
  sorry

end solution_set_of_inequality_l232_232939


namespace calculate_expression_correct_l232_232421

theorem calculate_expression_correct :
  ( (6 + (7 / 8) - (2 + (1 / 2))) * (1 / 4) + (3 + (23 / 24) + 1 + (2 / 3)) / 4 ) / 2.5 = 1 := 
by 
  sorry

end calculate_expression_correct_l232_232421


namespace nonagon_diagonals_not_parallel_l232_232405

theorem nonagon_diagonals_not_parallel (n : ℕ) (h : n = 9) : 
  ∃ k : ℕ, k = 18 ∧ 
    ∀ v₁ v₂, v₁ ≠ v₂ → (n : ℕ).choose 2 = 27 → 
    (v₂ - v₁) % n ≠ 4 ∧ (v₂ - v₁) % n ≠ n-4 :=
by
  sorry

end nonagon_diagonals_not_parallel_l232_232405


namespace place_numbers_in_table_l232_232957

theorem place_numbers_in_table (nums : Fin 100 → ℝ) (h_distinct : Function.Injective nums) :
  ∃ (table : Fin 10 → Fin 10 → ℝ),
    (∀ i j, table i j = nums ⟨10 * i + j, sorry⟩) ∧
    (∀ i j k l, (i, j) ≠ (k, l) → (i = k ∧ (j = l + 1 ∨ j = l - 1) ∨ j = l ∧ (i = k + 1 ∨ i = k - 1)) →
      |table i j - table k l| ≠ 1) := sorry  -- Proof omitted

end place_numbers_in_table_l232_232957


namespace x_one_minus_f_eq_one_l232_232570

noncomputable def x : ℝ := (1 + Real.sqrt 2) ^ 500
noncomputable def n : ℤ := Int.floor x
noncomputable def f : ℝ := x - n

theorem x_one_minus_f_eq_one : x * (1 - f) = 1 :=
by
  sorry

end x_one_minus_f_eq_one_l232_232570


namespace statement_correctness_l232_232990

def correct_statements := [4, 8]
def incorrect_statements := [1, 2, 3, 5, 6, 7]

theorem statement_correctness :
  correct_statements = [4, 8] ∧ incorrect_statements = [1, 2, 3, 5, 6, 7] :=
  by sorry

end statement_correctness_l232_232990


namespace shape_is_plane_l232_232081

-- Define cylindrical coordinates
structure CylindricalCoord :=
  (r : ℝ) (theta : ℝ) (z : ℝ)

-- Define the condition
def condition (c : ℝ) (coord : CylindricalCoord) : Prop :=
  coord.z = c

-- The shape is described as a plane
def is_plane : Prop := ∀ (coord1 coord2 : CylindricalCoord), (coord1.z = coord2.z)

theorem shape_is_plane (c : ℝ) : 
  (∀ coord : CylindricalCoord, condition c coord) ↔ is_plane :=
by 
  sorry

end shape_is_plane_l232_232081


namespace find_N_product_l232_232154

variables (M L : ℤ) (N : ℤ)

theorem find_N_product
  (h1 : M = L + N)
  (h2 : M + 3 = (L + N + 3))
  (h3 : L - 5 = L - 5)
  (h4 : |(L + N + 3) - (L - 5)| = 4) :
  N = -4 ∨ N = -12 → (-4 * -12) = 48 :=
by sorry

end find_N_product_l232_232154


namespace find_x_l232_232670

theorem find_x (x y : ℝ) (h1 : y = 1 / (2 * x + 2)) (h2 : y = 2) : x = -3 / 4 :=
by
  sorry

end find_x_l232_232670


namespace same_remainder_division_l232_232134

theorem same_remainder_division {a m b : ℤ} (r c k : ℤ) 
  (ha : a = b * c + r) (hm : m = b * k + r) : b ∣ (a - m) :=
by
  sorry

end same_remainder_division_l232_232134


namespace son_work_time_l232_232870

theorem son_work_time :
  let M := (1 : ℚ) / 7
  let combined_rate := (1 : ℚ) / 3
  let S := combined_rate - M
  1 / S = 5.25 :=  
by
  sorry

end son_work_time_l232_232870


namespace problem1_problem2_l232_232936

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a| + |3 * x - 1|

-- Part (1) statement
theorem problem1 (x : ℝ) : f x (-1) ≤ 1 ↔ (1/4 ≤ x ∧ x ≤ 1/2) :=
by
    sorry

-- Part (2) statement
theorem problem2 (x a : ℝ) (h : 1/4 ≤ x ∧ x ≤ 1) : f x a ≤ |3 * x + 1| ↔ -7/3 ≤ a ∧ a ≤ 1 :=
by
    sorry

end problem1_problem2_l232_232936


namespace Marty_painting_combinations_l232_232683

theorem Marty_painting_combinations :
  let parts_of_room := 2
  let colors := 5
  let methods := 3
  (parts_of_room * colors * methods) = 30 := 
by
  let parts_of_room := 2
  let colors := 5
  let methods := 3
  show (parts_of_room * colors * methods) = 30
  sorry

end Marty_painting_combinations_l232_232683


namespace range_of_a_l232_232954

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (hf : ∀ x : ℝ, f x = a*x^3 + Real.log x) :
  (∃ x : ℝ, x > 0 ∧ (deriv f x = 0)) → a < 0 :=
by
  sorry

end range_of_a_l232_232954


namespace gerbils_left_l232_232759

theorem gerbils_left (initial count sold : ℕ) (h_initial : count = 85) (h_sold : sold = 69) : 
  count - sold = 16 := 
by 
  sorry

end gerbils_left_l232_232759


namespace inequality_proof_l232_232633

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) : 
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000 / 9 :=
by
  sorry

end inequality_proof_l232_232633


namespace find_n_correct_l232_232924

noncomputable def find_n : Prop :=
  ∃ n : ℕ, 0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * (Real.pi / 180)) = Real.cos (317 * (Real.pi / 180)) → n = 43

theorem find_n_correct : find_n :=
  sorry

end find_n_correct_l232_232924


namespace number_of_n_values_l232_232831

-- Definition of sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ := 
  (n.digits 10).sum

-- The main statement to prove
theorem number_of_n_values : 
  ∃ M, M = 8 ∧ ∀ n : ℕ, (n + sum_of_digits n + sum_of_digits (sum_of_digits n) = 2010) → M = 8 :=
by
  sorry

end number_of_n_values_l232_232831


namespace largest_m_for_negative_integral_solutions_l232_232563

theorem largest_m_for_negative_integral_solutions :
  ∃ m : ℕ, (∀ p q : ℤ, 10 * p * p + (-m) * p + 560 = 0 ∧ p < 0 ∧ q < 0 ∧ p * q = 56 → m ≤ 570) ∧ m = 570 :=
sorry

end largest_m_for_negative_integral_solutions_l232_232563


namespace snail_max_distance_300_meters_l232_232327
-- Import required library

-- Define the problem statement
theorem snail_max_distance_300_meters 
  (n : ℕ) (left_turns : ℕ) (right_turns : ℕ) 
  (total_distance : ℕ)
  (h1 : n = 300)
  (h2 : left_turns = 99)
  (h3 : right_turns = 200)
  (h4 : total_distance = n) : 
  ∃ d : ℝ, d = 100 * Real.sqrt 2 :=
by
  sorry

end snail_max_distance_300_meters_l232_232327


namespace pears_value_l232_232195

-- Condition: 3/4 of 12 apples is equivalent to 6 pears
def apples_to_pears (a p : ℕ) : Prop := (3 / 4) * a = 6 * p

-- Target: 1/3 of 9 apples is equivalent to 2 pears
def target_equiv : Prop := (1 / 3) * 9 = 2

theorem pears_value (a p : ℕ) (h : apples_to_pears 12 6) : target_equiv := by
  sorry

end pears_value_l232_232195


namespace power_function_increasing_m_eq_2_l232_232307

theorem power_function_increasing_m_eq_2 (m : ℝ) :
  (∀ x > 0, (m^2 - m - 1) * x^m > 0) → m = 2 :=
by
  sorry

end power_function_increasing_m_eq_2_l232_232307


namespace contradiction_example_l232_232478

theorem contradiction_example 
  (a b c : ℝ) 
  (h : (a - 1) * (b - 1) * (c - 1) > 0) : 
  (1 < a) ∨ (1 < b) ∨ (1 < c) :=
by
  sorry

end contradiction_example_l232_232478


namespace yasmine_chocolate_beverage_l232_232426

theorem yasmine_chocolate_beverage :
  ∃ (m s : ℕ), (∀ k : ℕ, k > 0 → (∃ n : ℕ, 4 * n = 7 * k) → (m, s) = (7 * k, 4 * k)) ∧
  (2 * 7 * 1 + 1.4 * 4 * 1) = 19.6 := by
sorry

end yasmine_chocolate_beverage_l232_232426


namespace candy_necklaces_left_l232_232200

theorem candy_necklaces_left (total_packs : ℕ) (candy_per_pack : ℕ) 
  (opened_packs : ℕ) (candy_necklaces : ℕ)
  (h1 : total_packs = 9) 
  (h2 : candy_per_pack = 8) 
  (h3 : opened_packs = 4)
  (h4 : candy_necklaces = total_packs * candy_per_pack) :
  (total_packs - opened_packs) * candy_per_pack = 40 :=
by
  sorry

end candy_necklaces_left_l232_232200


namespace jeremy_money_ratio_l232_232783

theorem jeremy_money_ratio :
  let cost_computer := 3000
  let cost_accessories := 0.10 * cost_computer
  let money_left := 2700
  let total_spent := cost_computer + cost_accessories
  let money_before_purchase := total_spent + money_left
  (money_before_purchase / cost_computer) = 2 := by
  sorry

end jeremy_money_ratio_l232_232783


namespace problem1_problem2_l232_232551

def M (x : ℝ) : Prop := (x + 5) / (x - 8) ≥ 0

def N (x : ℝ) (a : ℝ) : Prop := a - 1 ≤ x ∧ x ≤ a + 1

theorem problem1 : ∀ (x : ℝ), (M x ∨ (N x 9)) ↔ (x ≤ -5 ∨ x ≥ 8) :=
by
  sorry

theorem problem2 : ∀ (a : ℝ), (∀ (x : ℝ), N x a → M x) ↔ (a ≤ -6 ∨ 9 < a) :=
by
  sorry

end problem1_problem2_l232_232551


namespace james_pays_660_for_bed_and_frame_l232_232451

theorem james_pays_660_for_bed_and_frame :
  let bed_frame_price := 75
  let bed_price := 10 * bed_frame_price
  let total_price_before_discount := bed_frame_price + bed_price
  let discount := 0.20 * total_price_before_discount
  let final_price := total_price_before_discount - discount
  final_price = 660 := 
by
  sorry

end james_pays_660_for_bed_and_frame_l232_232451


namespace age_problem_l232_232511

-- Define the ages of a, b, and c
variables (a b c : ℕ)

-- State the conditions
theorem age_problem (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 22) : b = 8 :=
by
  sorry

end age_problem_l232_232511


namespace handshakes_count_l232_232053

-- Define the number of people
def num_people : ℕ := 10

-- Define a function to calculate the number of handshakes
noncomputable def num_handshakes (n : ℕ) : ℕ :=
  (n - 1) * n / 2

-- The main statement to be proved
theorem handshakes_count : num_handshakes num_people = 45 := by
  -- Proof will be filled in here
  sorry

end handshakes_count_l232_232053


namespace temperature_difference_l232_232827

theorem temperature_difference (T_high T_low : ℤ) (h_high : T_high = 11) (h_low : T_low = -11) :
  T_high - T_low = 22 := by
  sorry

end temperature_difference_l232_232827


namespace compare_diff_functions_l232_232491

variable {R : Type*} [LinearOrderedField R]
variable {f g : R → R}
variable (h_fg : ∀ x, f' x > g' x)
variable {x1 x2 : R}

theorem compare_diff_functions (h : x1 < x2) : f x1 - f x2 < g x1 - g x2 :=
  sorry

end compare_diff_functions_l232_232491


namespace dice_probability_l232_232856

-- The context that there are three six-sided dice
def total_outcomes : ℕ := 6 * 6 * 6

-- Function to count the number of favorable outcomes where two dice sum to the third
def favorable_outcomes : ℕ :=
  let sum_cases := [1, 2, 3, 4, 5]
  sum_cases.sum
  -- sum_cases is [1, 2, 3, 4, 5] each mapping to the number of ways to form that sum with a third die

theorem dice_probability : 
  (favorable_outcomes * 3) / total_outcomes = 5 / 24 := 
by 
  -- to prove: the probability that the values on two dice sum to the value on the remaining die is 5/24
  sorry

end dice_probability_l232_232856


namespace distance_between_homes_l232_232086

def speed (name : String) : ℝ :=
  if name = "Maxwell" then 4
  else if name = "Brad" then 6
  else 0

def meeting_time : ℝ := 4

def delay : ℝ := 1

def distance_covered (name : String) : ℝ :=
  if name = "Maxwell" then speed name * meeting_time
  else if name = "Brad" then speed name * (meeting_time - delay)
  else 0

def total_distance : ℝ :=
  distance_covered "Maxwell" + distance_covered "Brad"

theorem distance_between_homes : total_distance = 34 :=
by
  -- proof goes here
  sorry

end distance_between_homes_l232_232086


namespace range_of_m_l232_232506

variable (m : ℝ)

def p : Prop := m + 1 ≤ 0
def q : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem range_of_m (h : ¬ (p m ∧ q m)) : m ≤ -2 ∨ m > -1 := 
by
  sorry

end range_of_m_l232_232506


namespace gcd_80_36_l232_232250

theorem gcd_80_36 : Nat.gcd 80 36 = 4 := by
  -- Using the method of successive subtraction algorithm
  sorry

end gcd_80_36_l232_232250


namespace greatest_large_chips_l232_232091

def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, 2 ≤ a ∧ 2 ≤ b ∧ n = a * b

theorem greatest_large_chips (s l : ℕ) (c : ℕ) (hc : is_composite c) (h : s + l = 60) (hs : s = l + c) :
  l ≤ 28 :=
sorry

end greatest_large_chips_l232_232091


namespace negation_of_p_l232_232321

open Real

-- Define the original proposition p
def p : Prop := ∀ x : ℝ, exp x > log x

-- Theorem stating that the negation of p is as described
theorem negation_of_p : ¬p ↔ ∃ x : ℝ, exp x ≤ log x :=
by
  sorry

end negation_of_p_l232_232321


namespace f_x1_plus_f_x2_always_greater_than_zero_l232_232641

theorem f_x1_plus_f_x2_always_greater_than_zero
  {f : ℝ → ℝ}
  (h1 : ∀ x, f (-x) = -f (x + 2))
  (h2 : ∀ x > 1, ∀ y > 1, x < y → f y < f x)
  (h3 : ∃ x₁ x₂ : ℝ, 1 + x₁ * x₂ < x₁ + x₂ ∧ x₁ + x₂ < 2) :
  ∀ x₁ x₂ : ℝ, (1 + x₁ * x₂ < x₁ + x₂ ∧ x₁ + x₂ < 2) → f x₁ + f x₂ > 0 := by
  sorry

end f_x1_plus_f_x2_always_greater_than_zero_l232_232641


namespace intercept_sum_l232_232486

theorem intercept_sum {x y : ℝ} 
  (h : y - 3 = -3 * (x - 5)) 
  (hx : x = 6) 
  (hy : y = 18) 
  (intercept_sum_eq : x + y = 24) : 
  x + y = 24 :=
by
  sorry

end intercept_sum_l232_232486


namespace sequence_is_increasing_l232_232661

-- Define the sequence recurrence property
def sequence_condition (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = 3

-- The theorem statement
theorem sequence_is_increasing (a : ℕ → ℤ) (h : sequence_condition a) : 
  ∀ n : ℕ, a n < a (n + 1) :=
by
  unfold sequence_condition at h
  intro n
  specialize h n
  sorry

end sequence_is_increasing_l232_232661


namespace directrix_of_parabola_l232_232757

-- Define the equation of the parabola and what we need to prove
def parabola_equation (x : ℝ) : ℝ := 2 * x^2 + 6

-- Theorem stating the directrix of the given parabola
theorem directrix_of_parabola :
  ∀ x : ℝ, y = parabola_equation x → y = 47 / 8 := 
by
  sorry

end directrix_of_parabola_l232_232757


namespace boxes_given_to_mom_l232_232505

theorem boxes_given_to_mom 
  (sophie_boxes : ℕ) 
  (donuts_per_box : ℕ) 
  (donuts_to_sister : ℕ) 
  (donuts_left_for_her : ℕ) 
  (H1 : sophie_boxes = 4) 
  (H2 : donuts_per_box = 12) 
  (H3 : donuts_to_sister = 6) 
  (H4 : donuts_left_for_her = 30)
  : sophie_boxes * donuts_per_box - donuts_to_sister - donuts_left_for_her = donuts_per_box := 
by
  sorry

end boxes_given_to_mom_l232_232505


namespace arrange_x_y_z_l232_232292

theorem arrange_x_y_z (x : ℝ) (hx : 0.9 < x ∧ x < 1) :
  let y := x^(1/x)
  let z := x^y
  x < z ∧ z < y :=
by
  let y := x^(1/x)
  let z := x^y
  have : 0.9 < x ∧ x < 1 := hx
  sorry

end arrange_x_y_z_l232_232292


namespace geq_solution_l232_232851

def geom_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ (a (n+1) / a n) = (a 1 / a 0)

theorem geq_solution
  (a : ℕ → ℝ)
  (h_seq : geom_seq a)
  (h_cond : a 0 * a 2 + 2 * a 1 * a 3 + a 1 * a 5 = 9) :
  a 1 + a 3 = 3 :=
sorry

end geq_solution_l232_232851


namespace incorrect_conclusions_l232_232013

theorem incorrect_conclusions
  (h1 : ∃ (y x : ℝ), (¬∃ a b : ℝ, a < 0 ∧ y = a * x + b) ∧ ∃ a b : ℝ, y = 2.347 * x - 6.423)
  (h2 : ∃ (y x : ℝ), (∃ a b : ℝ, a < 0 ∧ y = a * x + b) ∧ y = -3.476 * x + 5.648)
  (h3 : ∃ (y x : ℝ), (∃ a b : ℝ, a > 0 ∧ y = a * x + b) ∧ y = 5.437 * x + 8.493)
  (h4 : ∃ (y x : ℝ), (¬∃ a b : ℝ, a > 0 ∧ y = a * x + b) ∧ y = -4.326 * x - 4.578) :
  (∃ (y x : ℝ), y = 2.347 * x - 6.423 ∧ (¬∃ a b : ℝ, a < 0 ∧ y = a * x + b)) ∧
  (∃ (y x : ℝ), y = -4.326 * x - 4.578 ∧ (¬∃ a b : ℝ, a > 0 ∧ y = a * x + b)) :=
by {
  sorry
}

end incorrect_conclusions_l232_232013


namespace larger_number_is_1634_l232_232033

theorem larger_number_is_1634 (L S : ℤ) (h1 : L - S = 1365) (h2 : L = 6 * S + 20) : L = 1634 := 
sorry

end larger_number_is_1634_l232_232033


namespace least_integer_solution_l232_232657

theorem least_integer_solution (x : ℤ) : (∀ y : ℤ, |2 * y + 9| <= 20 → x ≤ y) ↔ x = -14 := by
  sorry

end least_integer_solution_l232_232657


namespace number_of_female_students_l232_232288

theorem number_of_female_students 
  (average_all : ℝ)
  (num_males : ℝ) 
  (average_males : ℝ)
  (average_females : ℝ) 
  (h_avg_all : average_all = 88)
  (h_num_males : num_males = 15)
  (h_avg_males : average_males = 80)
  (h_avg_females : average_females = 94) :
  ∃ F : ℝ, 1200 + 94 * F = 88 * (15 + F) ∧ F = 20 :=
by
  use 20
  sorry

end number_of_female_students_l232_232288


namespace first_set_broken_percent_l232_232202

-- Defining some constants
def firstSetTotal : ℕ := 50
def secondSetTotal : ℕ := 60
def secondSetBrokenPercent : ℕ := 20
def totalBrokenMarbles : ℕ := 17

-- Define the function that calculates broken marbles from percentage
def brokenMarbles (percent marbles : ℕ) : ℕ := (percent * marbles) / 100

-- Theorem statement
theorem first_set_broken_percent :
  ∃ (x : ℕ), brokenMarbles x firstSetTotal + brokenMarbles secondSetBrokenPercent secondSetTotal = totalBrokenMarbles ∧ x = 10 :=
by
  sorry

end first_set_broken_percent_l232_232202


namespace range_of_a_l232_232306

noncomputable def f (x : ℝ) : ℝ := Real.log (2 + 3 * x) - (3 / 2) * x^2
noncomputable def f' (x : ℝ) : ℝ := (3 / (2 + 3 * x)) - 3 * x
noncomputable def valid_range (a : ℝ) : Prop := 
∀ x : ℝ, (1 / 6) ≤ x ∧ x ≤ (1 / 3) → |a - Real.log x| + Real.log (f' x + 3 * x) > 0

theorem range_of_a : { a : ℝ | valid_range a } = { a : ℝ | a ≠ Real.log (1 / 3) } := 
sorry

end range_of_a_l232_232306


namespace find_n_l232_232900

-- Defining the conditions given in the problem
def condition_eq (n : ℝ) : Prop :=
  10 * 1.8 - (n * 1.5 / 0.3) = 50

-- Stating the goal: Prove that the number n is -6.4
theorem find_n : condition_eq (-6.4) :=
by
  -- Proof is omitted
  sorry

end find_n_l232_232900


namespace xy_difference_l232_232061

theorem xy_difference (x y : ℝ) (h1 : x + y = 6) (h2 : x^2 - y^2 = 12) : x - y = 2 := by
  sorry

end xy_difference_l232_232061


namespace certain_number_l232_232110

theorem certain_number (x : ℝ) (h : 4 * x = 200) : x = 50 :=
by
  sorry

end certain_number_l232_232110


namespace different_colors_probability_l232_232907

-- Definitions of the chips in the bag
def purple_chips := 7
def green_chips := 6
def orange_chips := 5
def total_chips := purple_chips + green_chips + orange_chips

-- Calculating probabilities for drawing chips of different colors and ensuring the final probability of different colors is correct
def probability_different_colors : ℚ :=
  let P := purple_chips
  let G := green_chips
  let O := orange_chips
  let T := total_chips
  (P / T) * ((G + O) / T) + (G / T) * ((P + O) / T) + (O / T) * ((P + G) / T)

theorem different_colors_probability : probability_different_colors = (107 / 162) := by
  sorry

end different_colors_probability_l232_232907


namespace gaeun_taller_than_nana_l232_232546

def nana_height_m : ℝ := 1.618
def gaeun_height_cm : ℝ := 162.3
def nana_height_cm : ℝ := nana_height_m * 100

theorem gaeun_taller_than_nana : gaeun_height_cm - nana_height_cm = 0.5 := by
  sorry

end gaeun_taller_than_nana_l232_232546


namespace plants_same_height_after_54_years_l232_232993

noncomputable def h1 (t : ℝ) : ℝ := 44 + (3 / 2) * t
noncomputable def h2 (t : ℝ) : ℝ := 80 + (5 / 6) * t

theorem plants_same_height_after_54_years :
  ∃ t : ℝ, h1 t = h2 t :=
by
  use 54
  sorry

end plants_same_height_after_54_years_l232_232993


namespace proof_problem_l232_232656

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set A as {y | y = 2^x, x ∈ ℝ}
def A : Set ℝ := {y | ∃ x : ℝ, y = 2^x}

-- Define the set B as {x ∈ ℤ | x^2 - 4 ≤ 0}
def B : Set ℤ := {x | x ∈ Set.Icc (-2 : ℤ) 2}

-- Define the complement of A relative to U (universal set)
def CU_A : Set ℝ := {x | x ≤ 0}

-- Define the proposition to be proved
theorem proof_problem :
  (CU_A ∩ (Set.image (coe : ℤ → ℝ) B)) = {-2.0, 1.0, 0.0} :=
by 
  sorry

end proof_problem_l232_232656


namespace problem_solution_l232_232504

noncomputable def S : ℝ :=
  1 / (4 - Real.sqrt 15) - 1 / (Real.sqrt 15 - Real.sqrt 14) + 1 / (Real.sqrt 14 - Real.sqrt 13) - 1 / (Real.sqrt 13 - 3)

theorem problem_solution : S = (13 / 4) + (3 / 4) * Real.sqrt 13 :=
by sorry

end problem_solution_l232_232504


namespace intersection_of_function_and_inverse_l232_232351

theorem intersection_of_function_and_inverse (m : ℝ) :
  (∀ x y : ℝ, y = Real.sqrt (x - m) ↔ x = y^2 + m) →
  (∃ x : ℝ, Real.sqrt (x - m) = x) ↔ (m ≤ 1 / 4) :=
by
  sorry

end intersection_of_function_and_inverse_l232_232351


namespace intersection_points_of_circle_and_line_l232_232519

theorem intersection_points_of_circle_and_line :
  (∃ y, (4, y) ∈ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 25}) → 
  ∃ s : Finset (ℝ × ℝ), s.card = 2 ∧ ∀ p ∈ s, (p.1 = 4 ∧ (p.1 ^ 2 + p.2 ^ 2 = 25)) :=
by
  sorry

end intersection_points_of_circle_and_line_l232_232519


namespace sale_price_of_sarees_l232_232970

theorem sale_price_of_sarees 
  (P : ℝ) 
  (d1 d2 d3 d4 tax_rate : ℝ) 
  (P_initial : P = 510) 
  (d1_val : d1 = 0.12) 
  (d2_val : d2 = 0.15) 
  (d3_val : d3 = 0.20) 
  (d4_val : d4 = 0.10) 
  (tax_val : tax_rate = 0.10) :
  let discount_step (price discount : ℝ) := price * (1 - discount)
  let tax_step (price tax_rate : ℝ) := price * (1 + tax_rate)
  let P1 := discount_step P d1
  let P2 := discount_step P1 d2
  let P3 := discount_step P2 d3
  let P4 := discount_step P3 d4
  let final_price := tax_step P4 tax_rate
  abs (final_price - 302.13) < 0.01 := 
sorry

end sale_price_of_sarees_l232_232970


namespace upper_side_length_l232_232239

variable (L U h : ℝ)

-- Given conditions
def condition1 : Prop := U = L - 6
def condition2 : Prop := 72 = (1 / 2) * (L + U) * 8
def condition3 : Prop := h = 8

-- The length of the upper side of the trapezoid
theorem upper_side_length (h : h = 8) (c1 : U = L - 6) (c2 : 72 = (1 / 2) * (L + U) * 8) : U = 6 := 
by
  sorry

end upper_side_length_l232_232239


namespace cost_per_page_of_notebooks_l232_232450

-- Define the conditions
def notebooks : Nat := 2
def pages_per_notebook : Nat := 50
def cost_in_dollars : Nat := 5

-- Define the conversion constants
def dollars_to_cents : Nat := 100

-- Define the correct answer
def expected_cost_per_page := 5

-- State the theorem to prove the cost per page
theorem cost_per_page_of_notebooks :
  let total_pages := notebooks * pages_per_notebook
  let total_cost_in_cents := cost_in_dollars * dollars_to_cents
  let cost_per_page := total_cost_in_cents / total_pages
  cost_per_page = expected_cost_per_page :=
by
  -- Skip the proof with sorry
  sorry

end cost_per_page_of_notebooks_l232_232450


namespace statement_B_not_true_l232_232509

def diamondsuit (x y : ℝ) : ℝ := 2 * |(x - y)| + 1

theorem statement_B_not_true : ¬ (∀ x y : ℝ, 3 * diamondsuit x y = 3 * diamondsuit (2 * x) (2 * y)) :=
sorry

end statement_B_not_true_l232_232509


namespace value_of_E_l232_232263

variable {D E F : ℕ}

theorem value_of_E (h1 : D + E + F = 16) (h2 : F + D + 1 = 16) (h3 : E - 1 = D) : E = 1 :=
sorry

end value_of_E_l232_232263


namespace mutually_exclusive_scoring_l232_232333

-- Define conditions as types
def shoots_twice : Prop := true
def scoring_at_least_once : Prop :=
  ∃ (shot1 shot2 : Bool), shot1 || shot2
def not_scoring_both_times : Prop :=
  ∀ (shot1 shot2 : Bool), ¬(shot1 && shot2)

-- Statement of the problem: Prove the events are mutually exclusive.
theorem mutually_exclusive_scoring :
  shoots_twice → (scoring_at_least_once → not_scoring_both_times → false) :=
by
  intro h_shoots_twice
  intro h_scoring_at_least_once
  intro h_not_scoring_both_times
  sorry

end mutually_exclusive_scoring_l232_232333


namespace tangent_point_x_coordinate_l232_232791

-- Define the function representing the curve.
def curve (x : ℝ) : ℝ := x^2 + 1

-- Define the derivative of the curve.
def derivative (x : ℝ) : ℝ := 2 * x

-- The statement to be proved.
theorem tangent_point_x_coordinate (x : ℝ) (h : derivative x = 4) : x = 2 :=
sorry

end tangent_point_x_coordinate_l232_232791


namespace scientific_notation_correct_l232_232963

noncomputable def scientific_notation (x : ℝ) : ℝ × ℤ :=
  let a := x * 10^9
  (a, -9)

theorem scientific_notation_correct :
  scientific_notation 0.000000007 = (7, -9) :=
by
  sorry

end scientific_notation_correct_l232_232963


namespace otimes_computation_l232_232400

-- Definition of ⊗ given m
def otimes (a b m : ℕ) : ℚ := (m * a + b) / (2 * a * b)

-- The main theorem we need to prove
theorem otimes_computation (m : ℕ) (h : otimes 1 4 m = otimes 2 3 m) :
  otimes 3 4 6 = 11 / 12 :=
sorry

end otimes_computation_l232_232400


namespace initial_processing_capacity_l232_232031

variable (x y z : ℕ)

-- Conditions
def initial_condition : Prop := x * y = 38880
def after_modernization : Prop := (x + 3) * z = 44800
def capacity_increased : Prop := y < z
def minimum_machines : Prop := x ≥ 20

-- Prove that the initial daily processing capacity y is 1215
theorem initial_processing_capacity
  (h1 : initial_condition x y)
  (h2 : after_modernization x z)
  (h3 : capacity_increased y z)
  (h4 : minimum_machines x) :
  y = 1215 := by
  sorry

end initial_processing_capacity_l232_232031


namespace binom_two_formula_l232_232560

def binom (n k : ℕ) : ℕ :=
  n.choose k

-- Formalizing the conditions
variable (n : ℕ)
variable (h : n ≥ 2)

-- Stating the problem mathematically in Lean
theorem binom_two_formula :
  binom n 2 = n * (n - 1) / 2 := by
  sorry

end binom_two_formula_l232_232560


namespace product_mb_gt_one_l232_232662

theorem product_mb_gt_one (m b : ℝ) (hm : m = 3 / 4) (hb : b = 2) : m * b = 3 / 2 := by
  sorry

end product_mb_gt_one_l232_232662


namespace no_solution_ineq_positive_exponents_l232_232392

theorem no_solution_ineq (m : ℝ) (h : m < 6) : ¬∃ x : ℝ, |x + 1| + |x - 5| ≤ m := 
sorry

theorem positive_exponents (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h_neq : a ≠ b) : a^a * b^b - a^b * b^a > 0 := 
sorry

end no_solution_ineq_positive_exponents_l232_232392


namespace remainder_product_mod_5_l232_232397

theorem remainder_product_mod_5 (a b c : ℕ) (h_a : a % 5 = 2) (h_b : b % 5 = 3) (h_c : c % 5 = 4) :
  (a * b * c) % 5 = 4 := 
by
  sorry

end remainder_product_mod_5_l232_232397


namespace find_divisor_l232_232622

theorem find_divisor (X : ℕ) (h12 : 12 ∣ (1020 - 12)) (h24 : 24 ∣ (1020 - 12)) (h48 : 48 ∣ (1020 - 12)) (h56 : 56 ∣ (1020 - 12)) :
  X = 63 :=
sorry

end find_divisor_l232_232622


namespace cheryl_initial_mms_l232_232178

theorem cheryl_initial_mms (lunch_mms : ℕ) (dinner_mms : ℕ) (sister_mms : ℕ) (total_mms : ℕ) 
  (h1 : lunch_mms = 7) (h2 : dinner_mms = 5) (h3 : sister_mms = 13) (h4 : total_mms = lunch_mms + dinner_mms + sister_mms) : 
  total_mms = 25 := 
by 
  rw [h1, h2, h3] at h4
  exact h4

end cheryl_initial_mms_l232_232178


namespace system_solution_l232_232940

theorem system_solution (x y : ℤ) (h1 : x + y = 1) (h2 : 2*x + y = 5) : x = 4 ∧ y = -3 :=
by {
  sorry
}

end system_solution_l232_232940


namespace tangent_line_parabola_l232_232698

theorem tangent_line_parabola (d : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + d → ∃! x y, y^2 = 12 * x) → d = 3 := 
by
  intro h
  -- Here, "h" would be our hypothesis where we assume the line is tangent to the parabola
  sorry

end tangent_line_parabola_l232_232698


namespace initial_pieces_count_l232_232658

theorem initial_pieces_count (people : ℕ) (pieces_per_person : ℕ) (leftover_pieces : ℕ) :
  people = 6 → pieces_per_person = 7 → leftover_pieces = 3 → people * pieces_per_person + leftover_pieces = 45 :=
by
  intros h_people h_pieces_per_person h_leftover_pieces
  sorry

end initial_pieces_count_l232_232658


namespace solve_inequality_min_value_F_l232_232549

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) - abs (x + 1)
def m := 3    -- Arbitrary constant, m + n = 7 implies n = 4
def n := 4

-- First statement: Solve the inequality f(x) ≥ (m + n)x
theorem solve_inequality (x : ℝ) : f x ≥ (m + n) * x ↔ x ≤ 0 := by
  sorry

noncomputable def F (x y : ℝ) : ℝ := max (abs (x^2 - 4 * y + m)) (abs (y^2 - 2 * x + n))

-- Second statement: Find the minimum value of F
theorem min_value_F (x y : ℝ) : (F x y) ≥ 1 ∧ (∃ x y, (F x y) = 1) := by
  sorry

end solve_inequality_min_value_F_l232_232549


namespace fg_of_minus_three_l232_232210

-- Definitions of the functions f and g
def f (x : ℤ) : ℤ := 2 * x - 1
def g (x : ℤ) : ℤ := x * x + 4

-- The theorem to prove
theorem fg_of_minus_three : f (g (-3)) = 25 := by
  sorry

end fg_of_minus_three_l232_232210


namespace tangent_line_to_parabola_l232_232866

theorem tangent_line_to_parabola (l : ℝ → ℝ) (y : ℝ) (x : ℝ)
  (passes_through_P : l (-2) = 0)
  (intersects_once : ∃! x, (l x)^2 = 8*x) :
  (l = fun x => 0) ∨ (l = fun x => x + 2) ∨ (l = fun x => -x - 2) :=
sorry

end tangent_line_to_parabola_l232_232866


namespace solution_set_of_inequality_min_value_of_expression_l232_232014

def f (x : ℝ) : ℝ := |x + 1| - |2 * x - 2|

-- (I) Prove that the solution set of the inequality f(x) ≥ x - 1 is [0, 2]
theorem solution_set_of_inequality 
  (x : ℝ) : f x ≥ x - 1 ↔ 0 ≤ x ∧ x ≤ 2 := 
sorry

-- (II) Given the maximum value m of f(x) is 2 and a + b + c = 2, prove the minimum value of b^2/a + c^2/b + a^2/c is 2
theorem min_value_of_expression
  (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : a + b + c = 2) :
  b^2 / a + c^2 / b + a^2 / c ≥ 2 :=
sorry

end solution_set_of_inequality_min_value_of_expression_l232_232014


namespace slope_angle_of_perpendicular_line_l232_232387

theorem slope_angle_of_perpendicular_line (h : ∀ x, x = (π / 3)) : ∀ θ, θ = (π / 2) := 
by 
  -- Placeholder for the proof
  sorry

end slope_angle_of_perpendicular_line_l232_232387


namespace more_than_3000_students_l232_232688

-- Define the conditions
def students_know_secret (n : ℕ) : ℕ :=
  3 ^ (n - 1)

-- Define the statement to prove
theorem more_than_3000_students : ∃ n : ℕ, students_know_secret n > 3000 ∧ n = 9 := by
  sorry

end more_than_3000_students_l232_232688


namespace max_area_rectangle_l232_232553

theorem max_area_rectangle (P : ℝ) (x : ℝ) (h1 : P = 40) (h2 : 6 * x = P) : 
  2 * (x ^ 2) = 800 / 9 :=
by
  sorry

end max_area_rectangle_l232_232553


namespace union_sets_S_T_l232_232220

open Set Int

def S : Set Int := { s : Int | ∃ n : Int, s = 2 * n + 1 }
def T : Set Int := { t : Int | ∃ n : Int, t = 4 * n + 1 }

theorem union_sets_S_T : S ∪ T = S := 
by sorry

end union_sets_S_T_l232_232220


namespace delta_max_success_ratio_l232_232027

theorem delta_max_success_ratio :
  ∃ (x y z w : ℕ),
  (0 < x ∧ x < (7 * y) / 12) ∧
  (0 < z ∧ z < (5 * w) / 8) ∧
  (y + w = 600) ∧
  (35 * x + 28 * z < 4200) ∧
  (x + z = 150) ∧ 
  (x + z) / 600 = 1 / 4 :=
by sorry

end delta_max_success_ratio_l232_232027


namespace parallelogram_area_l232_232956

theorem parallelogram_area (d : ℝ) (h : ℝ) (α : ℝ) (h_d : d = 30) (h_h : h = 20) : 
  ∃ A : ℝ, A = d * h ∧ A = 600 :=
by
  sorry

end parallelogram_area_l232_232956


namespace units_digit_of_square_l232_232871

theorem units_digit_of_square (a b : ℕ) (h₁ : (10 * a + b) ^ 2 % 100 / 10 = 7) : b = 6 :=
sorry

end units_digit_of_square_l232_232871


namespace initial_velocity_l232_232026

noncomputable def displacement (t : ℝ) : ℝ := 3 * t - t^2

theorem initial_velocity :
  (deriv displacement 0) = 3 :=
by
  sorry

end initial_velocity_l232_232026


namespace divides_p_minus_one_l232_232625

theorem divides_p_minus_one {p a b : ℕ} {n : ℕ} 
  (hp : p ≥ 3) 
  (prime_p : Nat.Prime p )
  (gcd_ab : Nat.gcd a b = 1)
  (hdiv : p ∣ (a ^ (2 ^ n) + b ^ (2 ^ n))) : 
  2 ^ (n + 1) ∣ p - 1 := 
sorry

end divides_p_minus_one_l232_232625


namespace problem1_range_of_x_problem2_value_of_a_l232_232802

open Set

-- Definition of the function f(x)
def f (x a : ℝ) : ℝ := |x + 3| + |x - a|

-- Problem 1
theorem problem1_range_of_x (a : ℝ) (h : a = 4) (h_eq : ∀ x : ℝ, f x a = 7 ↔ x ∈ Icc (-3 : ℝ) 4) :
  ∀ x : ℝ, f x 4 = 7 ↔ x ∈ Icc (-3 : ℝ) 4 := by
  sorry

-- Problem 2
theorem problem2_value_of_a (h₁ : ∀ x : ℝ, x ∈ {x : ℝ | f x 4 ≥ 6} ↔ x ≤ -4 ∨ x ≥ 2) :
  f x a ≥ 6 ↔  x ≤ -4 ∨ x ≥ 2 :=
  by
  sorry

end problem1_range_of_x_problem2_value_of_a_l232_232802


namespace range_of_f_when_a_0_range_of_a_for_three_zeros_l232_232005

noncomputable def f_part1 (x : ℝ) : ℝ :=
if h : x ≤ 0 then 2 ^ x else x ^ 2

theorem range_of_f_when_a_0 : Set.range f_part1 = {y : ℝ | 0 < y} := by
  sorry

noncomputable def f_part2 (a : ℝ) (x : ℝ) : ℝ :=
if h : x ≤ 0 then 2 ^ x - a else x ^ 2 - 3 * a * x + a

def discriminant (a : ℝ) (x : ℝ) : ℝ := (3 * a) ^ 2 - 4 * a

theorem range_of_a_for_three_zeros (a : ℝ) :
  (∀ x : ℝ, f_part2 a x = 0) → (4 / 9 < a ∧ a ≤ 1) := by
  sorry

end range_of_f_when_a_0_range_of_a_for_three_zeros_l232_232005


namespace arithmetic_sequence_term_l232_232448

theorem arithmetic_sequence_term (a : ℕ → ℕ) (h1 : a 2 = 2) (h2 : a 3 = 4) : a 10 = 18 :=
by
  sorry

end arithmetic_sequence_term_l232_232448


namespace number_of_ways_to_divide_friends_l232_232568

theorem number_of_ways_to_divide_friends :
  (4 ^ 8 = 65536) := by
  -- result obtained through the multiplication principle
  sorry

end number_of_ways_to_divide_friends_l232_232568


namespace x_ge_3_is_necessary_but_not_sufficient_for_x_gt_3_l232_232896

theorem x_ge_3_is_necessary_but_not_sufficient_for_x_gt_3 :
  (∀ x : ℝ, x > 3 → x ≥ 3) ∧ (∃ x : ℝ, x ≥ 3 ∧ ¬ (x > 3)) :=
by
  sorry

end x_ge_3_is_necessary_but_not_sufficient_for_x_gt_3_l232_232896


namespace set_intersection_l232_232254

open Set Real

theorem set_intersection (A : Set ℝ) (hA : A = {-1, 0, 1}) (B : Set ℝ) (hB : B = {y | ∃ x ∈ A, y = cos (π * x)}) :
  A ∩ B = {-1, 1} :=
by
  rw [hA, hB]
  -- remaining proof should go here
  sorry

end set_intersection_l232_232254


namespace number_condition_l232_232248

theorem number_condition (x : ℝ) (h : 45 - 3 * x^2 = 12) : x = Real.sqrt 11 ∨ x = -Real.sqrt 11 :=
sorry

end number_condition_l232_232248


namespace james_bought_dirt_bikes_l232_232498

variable (D : ℕ)

-- Definitions derived from conditions
def cost_dirt_bike := 150
def cost_off_road_vehicle := 300
def registration_fee := 25
def num_off_road_vehicles := 4
def total_paid := 1825

-- Auxiliary definitions
def total_cost_dirt_bike := cost_dirt_bike + registration_fee
def total_cost_off_road_vehicle := cost_off_road_vehicle + registration_fee
def total_cost_off_road_vehicles := num_off_road_vehicles * total_cost_off_road_vehicle
def total_cost_dirt_bikes := total_paid - total_cost_off_road_vehicles

-- The final statement we need to prove
theorem james_bought_dirt_bikes : D = total_cost_dirt_bikes / total_cost_dirt_bike ↔ D = 3 := by
  sorry

end james_bought_dirt_bikes_l232_232498


namespace triangle_height_l232_232858

theorem triangle_height (base : ℝ) (height : ℝ) (area : ℝ)
  (h_base : base = 8) (h_area : area = 16) (h_area_formula : area = (base * height) / 2) :
  height = 4 :=
by
  sorry

end triangle_height_l232_232858


namespace forum_members_l232_232445

theorem forum_members (M : ℕ)
  (h1 : ∀ q a, a = 3 * q)
  (h2 : ∀ h d, q = 3 * h * d)
  (h3 : 24 * (M * 3 * (24 + 3 * 72)) = 57600) : M = 200 :=
by
  sorry

end forum_members_l232_232445


namespace eqn_has_real_root_in_interval_l232_232217

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) + x - 3

theorem eqn_has_real_root_in_interval (k : ℤ) :
  (∃ (x : ℝ), x > k ∧ x < (k + 1) ∧ f x = 0) → k = 2 :=
by
  sorry

end eqn_has_real_root_in_interval_l232_232217


namespace inequality_proof_l232_232944

theorem inequality_proof
  (a b c d : ℝ) 
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d)
  (h_cond : a * b + b * c + c * d + d * a = 1) :
  (a^3 / (b + c + d) + b^3 / (c + d + a) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1 / 3) :=
by {
  sorry
}

end inequality_proof_l232_232944


namespace lines_parallel_if_perpendicular_to_same_plane_l232_232118

variables {Line : Type} {Plane : Type}
variable (a b : Line)
variable (α : Plane)

-- Conditions 
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry -- Definition for line perpendicular to plane
def lines_parallel (l1 l2 : Line) : Prop := sorry -- Definition for lines parallel

-- Theorem Statement
theorem lines_parallel_if_perpendicular_to_same_plane :
  line_perpendicular_to_plane a α →
  line_perpendicular_to_plane b α →
  lines_parallel a b :=
sorry

end lines_parallel_if_perpendicular_to_same_plane_l232_232118


namespace lanies_salary_l232_232538

variables (hours_worked_per_week : ℚ) (hourly_rate : ℚ)

namespace Lanie
def salary (fraction_of_weekly_hours : ℚ) : ℚ :=
  (fraction_of_weekly_hours * hours_worked_per_week) * hourly_rate

theorem lanies_salary : 
  hours_worked_per_week = 40 ∧
  hourly_rate = 15 ∧
  fraction_of_weekly_hours = 4 / 5 →
  salary fraction_of_weekly_hours = 480 :=
by
  -- Proof steps go here
  sorry
end Lanie

end lanies_salary_l232_232538


namespace original_number_l232_232532

theorem original_number (x : ℝ) (h : 20 = 0.4 * (x - 5)) : x = 55 :=
sorry

end original_number_l232_232532


namespace proj_vector_correct_l232_232125

open Real

noncomputable def vector_proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot := u.1 * v.1 + u.2 * v.2
  let mag_sq := v.1 * v.1 + v.2 * v.2
  (dot / mag_sq) • v

theorem proj_vector_correct :
  vector_proj ⟨3, -1⟩ ⟨4, -6⟩ = ⟨18 / 13, -27 / 13⟩ :=
  sorry

end proj_vector_correct_l232_232125


namespace only_statement_4_is_correct_l232_232921

-- Defining conditions for input/output statement correctness
def INPUT_statement_is_correct (s : String) : Prop :=
  s = "INPUT x=, 2"

def PRINT_statement_is_correct (s : String) : Prop :=
  s = "PRINT 20, 4"

-- List of statements
def statement_1 := "INPUT a; b; c"
def statement_2 := "PRINT a=1"
def statement_3 := "INPUT x=2"
def statement_4 := "PRINT 20, 4"

-- Predicate for correctness of statements
def statement_is_correct (s : String) : Prop :=
  (s = statement_4) ∧
  ¬(s = statement_1 ∨ s = statement_2 ∨ s = statement_3)

-- Theorem to prove that only statement 4 is correct
theorem only_statement_4_is_correct :
  ∀ s : String, (statement_is_correct s) ↔ (s = statement_4) :=
by
  intros s
  sorry

end only_statement_4_is_correct_l232_232921


namespace equal_vectors_implies_collinear_l232_232730

-- Definitions for vectors and their properties
variables {V : Type*} [AddCommGroup V] [Module ℝ V]

def collinear (u v : V) : Prop := ∃ (a : ℝ), v = a • u 

def equal_vectors (u v : V) : Prop := u = v

theorem equal_vectors_implies_collinear (u v : V)
  (h : equal_vectors u v) : collinear u v :=
by sorry

end equal_vectors_implies_collinear_l232_232730


namespace winding_clock_available_time_l232_232725

theorem winding_clock_available_time
    (minute_hand_restriction_interval: ℕ := 5) -- Each interval the minute hand restricts
    (hour_hand_restriction_interval: ℕ := 60) -- Each interval the hour hand restricts
    (intervals_per_12_hours: ℕ := 2) -- Number of restricted intervals in each 12-hour cycle
    (minutes_in_day: ℕ := 24 * 60) -- Total minutes in 24 hours
    : (minutes_in_day - ((minute_hand_restriction_interval * intervals_per_12_hours * 12) + 
                         (hour_hand_restriction_interval * intervals_per_12_hours * 2))) = 1080 :=
by
  -- Skipping the proof steps
  sorry

end winding_clock_available_time_l232_232725


namespace expression_undefined_l232_232857

theorem expression_undefined (a : ℝ) : (a = 2 ∨ a = -2) ↔ (a^2 - 4 = 0) :=
by sorry

end expression_undefined_l232_232857


namespace integer_values_abc_l232_232646

theorem integer_values_abc (a b c : ℤ) :
  1 < a ∧ a < b ∧ b < c ∧ (a - 1) * (b - 1) * (c - 1) ∣ (a * b * c - 1) →
  (a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15) :=
by
  sorry

end integer_values_abc_l232_232646


namespace elements_in_set_C_l232_232274

-- Definitions and main theorem
variables (C D : Finset ℕ)  -- Define sets C and D as finite sets of natural numbers
open BigOperators    -- Opens notation for finite sums

-- Given conditions as premises
def condition1 (c d : ℕ) : Prop := c = 3 * d
def condition2 (C D : Finset ℕ) : Prop := (C ∪ D).card = 4500
def condition3 (C D : Finset ℕ) : Prop := (C ∩ D).card = 1200

-- Theorem statement to be proven
theorem elements_in_set_C (c d : ℕ) (h1 : condition1 c d)
  (h2 : ∀ (C D : Finset ℕ), condition2 C D)
  (h3 : ∀ (C D : Finset ℕ), condition3 C D) :
  c = 4275 :=
sorry  -- proof to be completed

end elements_in_set_C_l232_232274


namespace product_of_roots_l232_232887

theorem product_of_roots :
  let a := 18
  let b := 45
  let c := -500
  let prod_roots := c / a
  prod_roots = -250 / 9 := 
by
  -- Define coefficients
  let a := 18
  let c := -500

  -- Calculate product of roots
  let prod_roots := c / a

  -- Statement to prove
  have : prod_roots = -250 / 9 := sorry
  exact this

-- Adding sorry since the proof is not required according to the problem statement.

end product_of_roots_l232_232887


namespace find_a_in_geometric_sequence_l232_232103

theorem find_a_in_geometric_sequence (S : ℕ → ℝ) (a : ℝ) :
  (∀ n, S n = 3^(n+1) + a) →
  (∃ a, ∀ n, S n = 3^(n+1) + a ∧ (18 : ℝ) ^ 2 = (S 1 - (S 1 - S 2)) * (S 2 - S 3) → a = -3) := 
by
  sorry

end find_a_in_geometric_sequence_l232_232103


namespace product_of_d_l232_232904

theorem product_of_d (d1 d2 : ℕ) (h1 : ∃ k1 : ℤ, 49 - 12 * d1 = k1^2)
  (h2 : ∃ k2 : ℤ, 49 - 12 * d2 = k2^2) (h3 : 0 < d1) (h4 : 0 < d2)
  (h5 : d1 ≠ d2) : d1 * d2 = 8 := 
sorry

end product_of_d_l232_232904


namespace simplify_3_375_to_fraction_l232_232499

def simplified_fraction_of_3_375 : ℚ := 3.375

theorem simplify_3_375_to_fraction : simplified_fraction_of_3_375 = 27 / 8 := 
by
  sorry

end simplify_3_375_to_fraction_l232_232499


namespace root_at_neg_x0_l232_232360

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom f_odd : ∀ x, f (-x) = -f x
axiom x0_root : ∃ x0, f x0 = Real.exp x0

-- Theorem
theorem root_at_neg_x0 : 
  (∃ x0, (f (-x0) * Real.exp (-x0) + 1 = 0))
  → (∃ x0, (f x0 * Real.exp x0 + 1 = 0)) := 
sorry

end root_at_neg_x0_l232_232360


namespace sum_of_midpoints_eq_15_l232_232098

theorem sum_of_midpoints_eq_15 (a b c d : ℝ) (h : a + b + c + d = 15) :
  (a + b) / 2 + (b + c) / 2 + (c + d) / 2 + (d + a) / 2 = 15 :=
by sorry

end sum_of_midpoints_eq_15_l232_232098


namespace remaining_amount_is_16_l232_232719

-- Define initial amount of money Sam has.
def initial_amount : ℕ := 79

-- Define cost per book.
def cost_per_book : ℕ := 7

-- Define the number of books.
def number_of_books : ℕ := 9

-- Define the total cost of books.
def total_cost : ℕ := cost_per_book * number_of_books

-- Define the remaining amount of money after buying the books.
def remaining_amount : ℕ := initial_amount - total_cost

-- Prove the remaining amount is 16 dollars.
theorem remaining_amount_is_16 : remaining_amount = 16 := by
  rfl

end remaining_amount_is_16_l232_232719


namespace find_missing_part_l232_232105

variable (x y : ℚ) -- Using rationals as the base field for generality.

theorem find_missing_part :
  2 * x * (-3 * x^2 * y) = -6 * x^3 * y := 
by
  sorry

end find_missing_part_l232_232105


namespace train_length_l232_232680

theorem train_length
  (V L : ℝ)
  (h1 : L = V * 18)
  (h2 : L + 350 = V * 39) :
  L = 300 := 
by
  sorry

end train_length_l232_232680


namespace store_revenue_is_1210_l232_232388

noncomputable def shirt_price : ℕ := 10
noncomputable def jeans_price : ℕ := 2 * shirt_price
noncomputable def jacket_price : ℕ := 3 * jeans_price
noncomputable def discounted_jacket_price : ℕ := jacket_price - (jacket_price / 10)

noncomputable def total_revenue : ℕ :=
  20 * shirt_price + 10 * jeans_price + 15 * discounted_jacket_price

theorem store_revenue_is_1210 :
  total_revenue = 1210 :=
by
  sorry

end store_revenue_is_1210_l232_232388


namespace factorial_division_l232_232434

theorem factorial_division :
  (Nat.factorial 4) / (Nat.factorial (4 - 3)) = 24 :=
by
  sorry

end factorial_division_l232_232434


namespace units_digit_fraction_l232_232076

theorem units_digit_fraction :
  (30 * 31 * 32 * 33 * 34 * 35) % 10 = (2500) % 10 → 
  ((30 * 31 * 32 * 33 * 34 * 35) / 2500) % 10 = 1 := 
by 
  intro h
  sorry

end units_digit_fraction_l232_232076


namespace smallest_divisible_1_to_10_l232_232269

open Nat

def is_divisible_by_all (n : ℕ) (s : List ℕ) : Prop :=
  ∀ x ∈ s, x ∣ n

theorem smallest_divisible_1_to_10 : ∃ n, is_divisible_by_all n [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ∧ n = 2520 := by
  sorry

end smallest_divisible_1_to_10_l232_232269


namespace inscribed_circle_radius_l232_232185

theorem inscribed_circle_radius :
  ∀ (a b c : ℝ), a = 3 → b = 6 → c = 18 → (∃ (r : ℝ), (1 / r) = (1 / a) + (1 / b) + (1 / c) + 4 * Real.sqrt ((1 / (a * b)) + (1 / (a * c)) + (1 / (b * c))) ∧ r = 9 / (5 + 6 * Real.sqrt 3)) :=
by
  intros a b c h₁ h₂ h₃
  sorry

end inscribed_circle_radius_l232_232185


namespace total_weight_peppers_l232_232216

def weight_green_peppers : ℝ := 0.3333333333333333
def weight_red_peppers : ℝ := 0.3333333333333333

theorem total_weight_peppers : weight_green_peppers + weight_red_peppers = 0.6666666666666666 := 
by sorry

end total_weight_peppers_l232_232216


namespace translation_symmetric_y_axis_phi_l232_232064

theorem translation_symmetric_y_axis_phi :
  ∀ (f : ℝ → ℝ) (φ : ℝ),
    (∀ x : ℝ, f x = Real.sin (2 * x + π / 6)) →
    (0 < φ ∧ φ ≤ π / 2) →
    (∀ x, Real.sin (2 * (x + φ) + π / 6) = Real.sin (2 * (-x + φ) + π / 6)) →
    φ = π / 6 :=
by
  intros f φ f_def φ_bounds symmetry
  sorry

end translation_symmetric_y_axis_phi_l232_232064


namespace count_integers_between_3250_and_3500_with_increasing_digits_l232_232987

theorem count_integers_between_3250_and_3500_with_increasing_digits :
  ∃ n : ℕ, n = 20 ∧
    (∀ x : ℕ, 3250 ≤ x ∧ x ≤ 3500 →
      ∀ (d1 d2 d3 d4 : ℕ),
        d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧
        (x = d1 * 1000 + d2 * 100 + d3 * 10 + d4) →
        (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4)) :=
  sorry

end count_integers_between_3250_and_3500_with_increasing_digits_l232_232987


namespace rowing_downstream_speed_l232_232674

-- Define the given conditions
def V_u : ℝ := 60  -- speed upstream in kmph
def V_s : ℝ := 75  -- speed in still water in kmph

-- Define the problem statement
theorem rowing_downstream_speed : ∃ (V_d : ℝ), V_s = (V_u + V_d) / 2 ∧ V_d = 90 :=
by
  sorry

end rowing_downstream_speed_l232_232674


namespace f_2016_plus_f_2015_l232_232824

theorem f_2016_plus_f_2015 (f : ℝ → ℝ) 
  (H1 : ∀ x, f (-x) = -f x) -- Odd function property
  (H2 : ∀ x, f (x + 1) = f (-x + 1)) -- Even function property for f(x+1)
  (H3 : f 1 = 1) : 
  f 2016 + f 2015 = -1 :=
sorry

end f_2016_plus_f_2015_l232_232824


namespace line_through_point_equal_intercepts_l232_232029

theorem line_through_point_equal_intercepts (P : ℝ × ℝ) (x y a : ℝ) (k : ℝ) 
  (hP : P = (2, 3))
  (hx : x / a + y / a = 1 ∨ (P.fst * k - P.snd = 0)) :
  (x + y - 5 = 0 ∨ 3 * P.fst - 2 * P.snd = 0) := by
  sorry

end line_through_point_equal_intercepts_l232_232029


namespace brick_height_l232_232899

def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

theorem brick_height 
  (l : ℝ) (w : ℝ) (SA : ℝ) (h : ℝ) 
  (surface_area_eq : surface_area l w h = SA)
  (length_eq : l = 10)
  (width_eq : w = 4)
  (surface_area_given : SA = 164) :
  h = 3 :=
by
  sorry

end brick_height_l232_232899


namespace back_wheel_revolutions_calculation_l232_232810

def front_wheel_radius : ℝ := 3
def back_wheel_radius : ℝ := 0.5
def gear_ratio : ℝ := 2
def front_wheel_revolutions : ℕ := 50

noncomputable def back_wheel_revolutions (front_wheel_radius back_wheel_radius gear_ratio : ℝ) (front_wheel_revolutions : ℕ) : ℝ :=
  let front_circumference := 2 * Real.pi * front_wheel_radius
  let distance_traveled := front_circumference * front_wheel_revolutions
  let back_circumference := 2 * Real.pi * back_wheel_radius
  distance_traveled / back_circumference * gear_ratio

theorem back_wheel_revolutions_calculation :
  back_wheel_revolutions front_wheel_radius back_wheel_radius gear_ratio front_wheel_revolutions = 600 :=
sorry

end back_wheel_revolutions_calculation_l232_232810


namespace pirates_divide_coins_l232_232675

theorem pirates_divide_coins (N : ℕ) (hN : 220 ≤ N ∧ N ≤ 300) :
  ∃ n : ℕ, 
    (N - 2 - (N - 2) / 3 - 2 - (2 * ((N - 2) / 3 - (2 * ((N - 2) / 3) / 3)) / 3) - 
    2 - (2 * (((N - 2) / 3 - (2 * ((N - 2) / 3) / 3)) / 3)) / 3) / 3 = 84 := 
sorry

end pirates_divide_coins_l232_232675


namespace sale_in_fifth_month_condition_l232_232230

theorem sale_in_fifth_month_condition 
  (sale1 sale2 sale3 sale4 sale6 : ℕ)
  (avg_sale : ℕ)
  (n_months : ℕ)
  (total_sales : ℕ)
  (first_four_sales_and_sixth : ℕ) :
  sale1 = 6435 → 
  sale2 = 6927 → 
  sale3 = 6855 → 
  sale4 = 7230 → 
  sale6 = 6791 → 
  avg_sale = 6800 → 
  n_months = 6 → 
  total_sales = avg_sale * n_months → 
  first_four_sales_and_sixth = sale1 + sale2 + sale3 + sale4 + sale6 → 
  ∃ sale5, sale5 = total_sales - first_four_sales_and_sixth ∧ sale5 = 6562 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end sale_in_fifth_month_condition_l232_232230


namespace not_sum_of_squares_l232_232231

def P (x y : ℝ) : ℝ := 4 + x^2 * y^4 + x^4 * y^2 - 3 * x^2 * y^2

theorem not_sum_of_squares (P : ℝ → ℝ → ℝ) : 
  (¬ ∃ g₁ g₂ : ℝ → ℝ → ℝ, ∀ x y : ℝ, P x y = g₁ x y * g₁ x y + g₂ x y * g₂ x y) :=
  by
  {
    -- By contradiction proof as outlined in the example problem
    sorry
  }

end not_sum_of_squares_l232_232231


namespace exists_number_between_70_and_80_with_gcd_10_l232_232554

theorem exists_number_between_70_and_80_with_gcd_10 :
  ∃ n : ℕ, 70 ≤ n ∧ n ≤ 80 ∧ Nat.gcd 30 n = 10 :=
sorry

end exists_number_between_70_and_80_with_gcd_10_l232_232554


namespace band_total_earnings_l232_232893

variables (earnings_per_gig_per_member : ℕ)
variables (number_of_members : ℕ)
variables (number_of_gigs : ℕ)

theorem band_total_earnings :
  earnings_per_gig_per_member = 20 →
  number_of_members = 4 →
  number_of_gigs = 5 →
  earnings_per_gig_per_member * number_of_members * number_of_gigs = 400 :=
by
  intros
  sorry

end band_total_earnings_l232_232893


namespace range_of_a_l232_232082

theorem range_of_a (h : ¬ ∃ x : ℝ, x^2 + (a-1) * x + 1 ≤ 0) : -1 < a ∧ a < 3 :=
sorry

end range_of_a_l232_232082


namespace cannot_eat_166_candies_l232_232409

-- Define parameters for sandwiches and candies equations
def sandwiches_eq (x y z : ℕ) := x + 2 * y + 3 * z = 100
def candies_eq (x y z : ℕ) := 3 * x + 4 * y + 5 * z = 166

theorem cannot_eat_166_candies (x y z : ℕ) : ¬ (sandwiches_eq x y z ∧ candies_eq x y z) :=
by {
  -- Proof will show impossibility of (x, y, z) as nonnegative integers solution
  sorry
}

end cannot_eat_166_candies_l232_232409


namespace ones_digit_of_sum_is_0_l232_232632

-- Define the integer n
def n : ℕ := 2012

-- Define the ones digit function
def ones_digit (x : ℕ) : ℕ := x % 10

-- Define the power function mod 10
def power_mod_10 (d a : ℕ) : ℕ := (d^a) % 10

-- Define the sequence sum for ones digits
def seq_sum_mod_10 (m : ℕ) : ℕ :=
  Finset.sum (Finset.range m) (λ k => power_mod_10 (k+1) n)

-- Define the final sum mod 10 considering the repeating cycle and sum
def total_ones_digit_sum (a b : ℕ) : ℕ :=
  let cycle_sum := Finset.sum (Finset.range 10) (λ k => power_mod_10 (k+1) n)
  let s := cycle_sum * (a / 10) + Finset.sum (Finset.range b) (λ k => power_mod_10 (k+1) n)
  s % 10

-- Prove that the ones digit of the sum is 0
theorem ones_digit_of_sum_is_0 : total_ones_digit_sum n (n % 10) = 0 :=
sorry

end ones_digit_of_sum_is_0_l232_232632


namespace cos_double_angle_l232_232293

theorem cos_double_angle
  {x : ℝ}
  (h : Real.sin x = -2 / 3) :
  Real.cos (2 * x) = 1 / 9 :=
by
  sorry

end cos_double_angle_l232_232293


namespace time_spent_on_road_l232_232800

theorem time_spent_on_road (Total_time_hours Stop1_minutes Stop2_minutes Stop3_minutes : ℕ) 
  (h1: Total_time_hours = 13) 
  (h2: Stop1_minutes = 25) 
  (h3: Stop2_minutes = 10) 
  (h4: Stop3_minutes = 25) : 
  Total_time_hours - (Stop1_minutes + Stop2_minutes + Stop3_minutes) / 60 = 12 :=
by
  sorry

end time_spent_on_road_l232_232800


namespace factorize_m_minimize_ab_find_abc_l232_232119

-- Problem 1: Factorization
theorem factorize_m (m : ℝ) : m^2 - 6 * m + 5 = (m - 1) * (m - 5) :=
sorry

-- Problem 2: Minimization
theorem minimize_ab (a b : ℝ) (h1 : (a - 2)^2 ≥ 0) (h2 : (b + 5)^2 ≥ 0) :
  ∃ (a b : ℝ), (a - 2)^2 + (b + 5)^2 + 4 = 4 ∧ a = 2 ∧ b = -5 :=
sorry

-- Problem 3: Value of a + b + c
theorem find_abc (a b c : ℝ) (h1 : a - b = 8) (h2 : a * b + c^2 - 4 * c + 20 = 0) :
  a + b + c = 2 :=
sorry

end factorize_m_minimize_ab_find_abc_l232_232119


namespace intersection_P_Q_l232_232651

def set_P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def set_Q : Set ℝ := {x | (x - 1) ^ 2 ≤ 4}

theorem intersection_P_Q :
  {x | x ∈ set_P ∧ x ∈ set_Q} = {x : ℝ | 1 ≤ x ∧ x ≤ 3} := by
  sorry

end intersection_P_Q_l232_232651


namespace snowman_volume_l232_232246

theorem snowman_volume
  (r1 r2 r3 : ℝ)
  (volume : ℝ)
  (h1 : r1 = 1)
  (h2 : r2 = 4)
  (h3 : r3 = 6)
  (h_volume : volume = (4.0 / 3.0) * Real.pi * (r1 ^ 3 + r2 ^ 3 + r3 ^ 3)) :
  volume = (1124.0 / 3.0) * Real.pi :=
by
  sorry

end snowman_volume_l232_232246


namespace final_value_l232_232895

noncomputable def f : ℕ → ℝ := sorry

axiom f_mul_add (a b : ℕ) : f (a + b) = f a * f b
axiom f_one : f 1 = 2

theorem final_value : 
  (f 1)^2 + f 2 / f 1 + (f 2)^2 + f 4 / f 3 + (f 3)^2 + f 6 / f 5 + (f 4)^2 + f 8 / f 7 = 16 := 
sorry

end final_value_l232_232895


namespace zero_point_exists_in_interval_l232_232874

noncomputable def f (x : ℝ) : ℝ := x + 2^x

theorem zero_point_exists_in_interval :
  ∃ x : ℝ, -1 < x ∧ x < 0 ∧ f x = 0 :=
by
  existsi -0.5 -- This is not a formal proof; the existi -0.5 is just for example purposes
  sorry

end zero_point_exists_in_interval_l232_232874


namespace greg_age_is_16_l232_232534

-- Define the ages of Cindy, Jan, Marcia, and Greg based on the conditions
def cindy_age : ℕ := 5
def jan_age : ℕ := cindy_age + 2
def marcia_age : ℕ := 2 * jan_age
def greg_age : ℕ := marcia_age + 2

-- Theorem statement: Prove that Greg's age is 16
theorem greg_age_is_16 : greg_age = 16 :=
by
  -- Proof would go here
  sorry

end greg_age_is_16_l232_232534


namespace total_books_arithmetic_sequence_l232_232258

theorem total_books_arithmetic_sequence :
  ∃ (n : ℕ) (a₁ a₂ aₙ d S : ℤ), 
    n = 11 ∧
    a₁ = 32 ∧
    a₂ = 29 ∧
    aₙ = 2 ∧
    d = -3 ∧
    S = (n * (a₁ + aₙ)) / 2 ∧
    S = 187 :=
by sorry

end total_books_arithmetic_sequence_l232_232258


namespace printer_task_total_pages_l232_232542

theorem printer_task_total_pages
  (A B : ℕ)
  (h1 : 1 / A + 1 / B = 1 / 24)
  (h2 : 1 / A = 1 / 60)
  (h3 : B = A + 6) :
  60 * A = 720 := by
  sorry

end printer_task_total_pages_l232_232542


namespace son_age_is_9_l232_232190

-- Definitions for the conditions in the problem
def son_age (S F : ℕ) : Prop := S = (1 / 4 : ℝ) * F - 1
def father_age (S F : ℕ) : Prop := F = 5 * S - 5

-- Main statement of the equivalent problem
theorem son_age_is_9 : ∃ S F : ℕ, son_age S F ∧ father_age S F ∧ S = 9 :=
by
  -- We will leave the proof as an exercise
  sorry

end son_age_is_9_l232_232190


namespace bug_probability_nine_moves_l232_232188

noncomputable def bug_cube_probability (moves : ℕ) : ℚ := sorry

/-- 
The probability that after exactly 9 moves, a bug starting at one vertex of a cube 
and moving randomly along the edges will have visited every vertex exactly once and 
revisited one vertex once more. 
-/
theorem bug_probability_nine_moves : bug_cube_probability 9 = 16 / 6561 := by
  sorry

end bug_probability_nine_moves_l232_232188
