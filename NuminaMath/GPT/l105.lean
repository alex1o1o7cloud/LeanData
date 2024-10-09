import Mathlib

namespace exists_multiple_of_n_with_ones_l105_10577

theorem exists_multiple_of_n_with_ones (n : ℤ) (hn1 : n ≥ 1) (hn2 : Int.gcd n 10 = 1) :
  ∃ k : ℕ, n ∣ (10^k - 1) / 9 :=
by sorry

end exists_multiple_of_n_with_ones_l105_10577


namespace sin_alpha_of_point_P_l105_10536

theorem sin_alpha_of_point_P (α : ℝ) 
  (h1 : ∃ P : ℝ × ℝ, P = (Real.cos (π / 3), 1) ∧ P = (Real.cos α, Real.sin α) ) :
  Real.sin α = (2 * Real.sqrt 5) / 5 := by
  sorry

end sin_alpha_of_point_P_l105_10536


namespace part1_part2_l105_10564

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (x^2 + 6)

theorem part1 (k : ℝ) :
  (∀ x : ℝ, (f x > k) ↔ (x < -3 ∨ x > -2)) ↔ k = -2/5 :=
by
  sorry

theorem part2 (t : ℝ) :
  (∀ x : ℝ, (x > 0) → (f x ≤ t)) ↔ t ∈ (Set.Ici (Real.sqrt 6 / 6)) :=
by
  sorry

end part1_part2_l105_10564


namespace find_analytical_expression_of_f_l105_10561

-- Define the function f satisfying the condition
def f (x : ℝ) : ℝ := sorry

-- Lean 4 theorem statement
theorem find_analytical_expression_of_f :
  (∀ x : ℝ, f (x + 1) = x^2 + 2*x + 2) → (∀ x : ℝ, f x = x^2 + 1) :=
by
  -- The initial f definition and theorem statement are created
  -- The proof is omitted since the focus is on translating the problem
  sorry

end find_analytical_expression_of_f_l105_10561


namespace total_handshakes_l105_10542

theorem total_handshakes :
  let gremlins := 20
  let imps := 20
  let sprites := 10
  let handshakes_gremlins := gremlins * (gremlins - 1) / 2
  let handshakes_gremlins_imps := gremlins * imps
  let handshakes_imps_sprites := imps * sprites
  handshakes_gremlins + handshakes_gremlins_imps + handshakes_imps_sprites = 790 :=
by
  sorry

end total_handshakes_l105_10542


namespace concert_revenue_l105_10548

-- Defining the conditions
def ticket_price_adult : ℕ := 26
def ticket_price_child : ℕ := ticket_price_adult / 2
def attendees_adults : ℕ := 183
def attendees_children : ℕ := 28

-- Defining the total revenue calculation based on the conditions
def total_revenue : ℕ :=
  attendees_adults * ticket_price_adult +
  attendees_children * ticket_price_child

-- The theorem to prove the total revenue
theorem concert_revenue : total_revenue = 5122 := by
  sorry

end concert_revenue_l105_10548


namespace min_omega_l105_10502

noncomputable def f (ω φ : ℝ) (x : ℝ) := 2 * Real.sin (ω * x + φ)

theorem min_omega (ω φ : ℝ) (hω : ω > 0)
  (h_sym : ∀ x : ℝ, f ω φ (2 * (π / 3) - x) = f ω φ x)
  (h_val : f ω φ (π / 12) = 0) :
  ω = 2 :=
sorry

end min_omega_l105_10502


namespace find_a_in_triangle_l105_10579

variable (a b c B : ℝ)

theorem find_a_in_triangle (h1 : b = Real.sqrt 3) (h2 : c = 3) (h3 : B = 30) :
    a = 2 * Real.sqrt 3 := by
  sorry

end find_a_in_triangle_l105_10579


namespace average_price_per_book_l105_10574

-- Definitions of the conditions
def books_shop1 := 65
def cost_shop1 := 1480
def books_shop2 := 55
def cost_shop2 := 920

-- Definition of total values
def total_books := books_shop1 + books_shop2
def total_cost := cost_shop1 + cost_shop2

-- Proof statement
theorem average_price_per_book : (total_cost / total_books) = 20 := by
  sorry

end average_price_per_book_l105_10574


namespace alcohol_percentage_after_adding_water_l105_10599

variables (initial_volume : ℕ) (initial_percentage : ℕ) (added_volume : ℕ)
def initial_alcohol_volume := initial_volume * initial_percentage / 100
def final_volume := initial_volume + added_volume
def final_percentage := initial_alcohol_volume * 100 / final_volume

theorem alcohol_percentage_after_adding_water :
  initial_volume = 15 →
  initial_percentage = 20 →
  added_volume = 5 →
  final_percentage = 15 := by
sorry

end alcohol_percentage_after_adding_water_l105_10599


namespace servings_in_box_l105_10511

def totalCereal : ℕ := 18
def servingSize : ℕ := 2

theorem servings_in_box : totalCereal / servingSize = 9 := by
  sorry

end servings_in_box_l105_10511


namespace sum_of_decimals_l105_10529

theorem sum_of_decimals : 5.46 + 2.793 + 3.1 = 11.353 := by
  sorry

end sum_of_decimals_l105_10529


namespace max_min_values_l105_10585

open Real

noncomputable def circle_condition (x y : ℝ) :=
  (x - 3) ^ 2 + (y - 3) ^ 2 = 6

theorem max_min_values (x y : ℝ) (hx : circle_condition x y) :
  ∃ k k' d d', 
    k = 3 + 2 * sqrt 2 ∧
    k' = 3 - 2 * sqrt 2 ∧
    k = y / x ∧
    k' = y / x ∧
    d = sqrt ((x - 2) ^ 2 + y ^ 2) ∧
    d' = sqrt ((x - 2) ^ 2 + y ^ 2) ∧
    d = sqrt (10) + sqrt (6) ∧
    d' = sqrt (10) - sqrt (6) :=
sorry

end max_min_values_l105_10585


namespace gcd_f_x_l105_10591

def f (x : ℤ) : ℤ := (5 * x + 3) * (11 * x + 2) * (14 * x + 7) * (3 * x + 8)

theorem gcd_f_x (x : ℤ) (hx : x % 3456 = 0) : Int.gcd (f x) x = 48 := by
  sorry

end gcd_f_x_l105_10591


namespace find_b_exists_l105_10534

theorem find_b_exists (N : ℕ) (hN : N ≠ 1) : ∃ (a c d : ℕ), a > 1 ∧ c > 1 ∧ d > 1 ∧
  (N : ℝ) ^ (1/a + 1/(a*4) + 1/(a*4*c) + 1/(a*4*c*d)) = (N : ℝ) ^ (37/48) :=
by
  sorry

end find_b_exists_l105_10534


namespace croissants_left_l105_10593

-- Definitions based on conditions
def total_croissants : ℕ := 17
def vegans : ℕ := 3
def allergic_to_chocolate : ℕ := 2
def any_type : ℕ := 2
def guests : ℕ := 7
def plain_needed : ℕ := vegans + allergic_to_chocolate
def plain_baked : ℕ := plain_needed
def choc_baked : ℕ := total_croissants - plain_baked

-- Assuming choc_baked > plain_baked as given
axiom croissants_greater_condition : choc_baked > plain_baked

-- Theorem to prove
theorem croissants_left (total_croissants vegans allergic_to_chocolate any_type guests : ℕ) 
    (plain_needed plain_baked choc_baked : ℕ) 
    (croissants_greater_condition : choc_baked > plain_baked) : 
    (choc_baked - guests + any_type) = 3 := 
by sorry

end croissants_left_l105_10593


namespace counting_unit_of_0_75_l105_10559

def decimal_places (n : ℝ) : ℕ := 
  by sorry  -- Assume this function correctly calculates the number of decimal places of n

def counting_unit (n : ℝ) : ℝ :=
  by sorry  -- Assume this function correctly determines the counting unit based on decimal places

theorem counting_unit_of_0_75 : counting_unit 0.75 = 0.01 :=
  by sorry


end counting_unit_of_0_75_l105_10559


namespace trigonometric_identity_proof_l105_10583

theorem trigonometric_identity_proof (α : ℝ) :
  3.3998 * (Real.cos α) ^ 4 - 4 * (Real.cos α) ^ 3 - 8 * (Real.cos α) ^ 2 + 3 * Real.cos α + 1 =
  -2 * Real.sin (7 * α / 2) * Real.sin (α / 2) :=
by
  sorry

end trigonometric_identity_proof_l105_10583


namespace intersection_of_domains_l105_10575

def A_domain : Set ℝ := { x : ℝ | 4 - x^2 ≥ 0 }
def B_domain : Set ℝ := { x : ℝ | 1 - x > 0 }

theorem intersection_of_domains :
  (A_domain ∩ B_domain) = { x : ℝ | -2 ≤ x ∧ x < 1 } :=
by
  sorry

end intersection_of_domains_l105_10575


namespace angle_BAC_is_105_or_35_l105_10554

-- Definitions based on conditions
def arcAB : ℝ := 110
def arcAC : ℝ := 40
def arcBC_major : ℝ := 360 - (arcAB + arcAC)
def arcBC_minor : ℝ := arcAB - arcAC

-- The conjecture: proving that the inscribed angle ∠BAC is 105° or 35° given the conditions.
theorem angle_BAC_is_105_or_35
  (h1 : 0 < arcAB ∧ arcAB < 360)
  (h2 : 0 < arcAC ∧ arcAC < 360)
  (h3 : arcAB + arcAC < 360) :
  (arcBC_major / 2 = 105) ∨ (arcBC_minor / 2 = 35) :=
  sorry

end angle_BAC_is_105_or_35_l105_10554


namespace Chrysler_Building_floors_l105_10517

variable (C L : ℕ)

theorem Chrysler_Building_floors :
  (C = L + 11) → (C + L = 35) → (C = 23) :=
by
  intro h1 h2
  sorry

end Chrysler_Building_floors_l105_10517


namespace domain_of_function_l105_10539

theorem domain_of_function :
  ∀ x, (x - 2 > 0) ∧ (3 - x ≥ 0) ↔ 2 < x ∧ x ≤ 3 :=
by 
  intros x 
  simp only [and_imp, gt_iff_lt, sub_lt_iff_lt_add, sub_nonneg, le_iff_eq_or_lt, add_comm]
  exact sorry

end domain_of_function_l105_10539


namespace vector_on_line_l105_10551

theorem vector_on_line (t : ℝ) (x y : ℝ) : 
  (x = 3 * t + 1) → (y = 2 * t + 3) → 
  ∃ t, (∃ x y, (x = 3 * t + 1) ∧ (y = 2 * t + 3) ∧ (x = 23 / 2) ∧ (y = 10)) :=
  by
  sorry

end vector_on_line_l105_10551


namespace fly_travel_time_to_opposite_vertex_l105_10592

noncomputable def cube_side_length (a : ℝ) := 
  a

noncomputable def fly_travel_time_base := 4 -- minutes

noncomputable def fly_speed (a : ℝ) := 
  4 * a / fly_travel_time_base

noncomputable def space_diagonal_length (a : ℝ) := 
  a * Real.sqrt 3

theorem fly_travel_time_to_opposite_vertex (a : ℝ) : 
  fly_speed a ≠ 0 -> 
  space_diagonal_length a / fly_speed a = Real.sqrt 3 :=
by
  intro h
  sorry

end fly_travel_time_to_opposite_vertex_l105_10592


namespace solution_opposite_numbers_l105_10528

theorem solution_opposite_numbers (x y : ℤ) (h1 : 2 * x + 3 * y - 4 = 0) (h2 : x = -y) : x = -4 ∧ y = 4 :=
by
  sorry

end solution_opposite_numbers_l105_10528


namespace minimum_cups_needed_l105_10546

theorem minimum_cups_needed (container_capacity cup_capacity : ℕ) (h1 : container_capacity = 980) (h2 : cup_capacity = 80) : 
  Nat.ceil (container_capacity / cup_capacity : ℚ) = 13 :=
by
  sorry

end minimum_cups_needed_l105_10546


namespace proof_problem_l105_10540

noncomputable def g (x : ℝ) : ℝ := 2^(2*x - 1) + x - 1

theorem proof_problem
  (x1 x2 : ℝ)
  (h1 : g x1 = 0)  -- x1 is the root of the equation
  (h2 : 2 * x2 - 1 = 0)  -- x2 is the zero point of f(x) = 2x - 1
  : |x1 - x2| ≤ 1/4 :=
sorry

end proof_problem_l105_10540


namespace stream_current_rate_l105_10518

theorem stream_current_rate (r c : ℝ) (h1 : 20 / (r + c) + 6 = 20 / (r - c)) (h2 : 20 / (3 * r + c) + 1.5 = 20 / (3 * r - c)) 
  : c = 3 :=
  sorry

end stream_current_rate_l105_10518


namespace parallel_x_axis_implies_conditions_l105_10525

variable (a b : ℝ)

theorem parallel_x_axis_implies_conditions (h1 : (5, a) ≠ (b, -2)) (h2 : (5, -2) = (5, a)) : a = -2 ∧ b ≠ 5 :=
sorry

end parallel_x_axis_implies_conditions_l105_10525


namespace least_three_digit_multiple_of_8_l105_10572

theorem least_three_digit_multiple_of_8 : 
  ∃ n : ℕ, n >= 100 ∧ n < 1000 ∧ (n % 8 = 0) ∧ 
  (∀ m : ℕ, m >= 100 ∧ m < 1000 ∧ (m % 8 = 0) → n ≤ m) ∧ n = 104 :=
sorry

end least_three_digit_multiple_of_8_l105_10572


namespace no_integer_solutions_l105_10531

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ), x^1988 + y^1988 + z^1988 = 7^1990 := by
  sorry

end no_integer_solutions_l105_10531


namespace starting_number_l105_10566

theorem starting_number (n : ℕ) (h1 : n % 11 = 3) (h2 : (n + 11) % 11 = 3) (h3 : (n + 22) % 11 = 3) 
  (h4 : (n + 33) % 11 = 3) (h5 : (n + 44) % 11 = 3) (h6 : n + 44 ≤ 50) : n = 3 := 
sorry

end starting_number_l105_10566


namespace divisibility_check_l105_10568

variable (d : ℕ) (h1 : d % 2 = 1) (h2 : d % 5 ≠ 0)
variable (δ : ℕ) (h3 : ∃ m : ℕ, 10 * δ + 1 = m * d)
variable (N : ℕ)

def last_digit (N : ℕ) : ℕ := N % 10
def remove_last_digit (N : ℕ) : ℕ := N / 10

theorem divisibility_check (h4 : ∃ N' u : ℕ, N = 10 * N' + u ∧ N = N' * 10 + u ∧ N' = remove_last_digit N ∧ u = last_digit N)
  (N' : ℕ) (u : ℕ) (N1 : ℕ) (h5 : N1 = N' - δ * u) :
  d ∣ N1 → d ∣ N := by
  sorry

end divisibility_check_l105_10568


namespace inequality_proof_l105_10513

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y + y * z + z * x = 1) :
  x * y * z * (x + y) * (y + z) * (z + x) ≥ (1 - x^2) * (1 - y^2) * (1 - z^2) :=
by
  sorry

end inequality_proof_l105_10513


namespace find_m_l105_10596

noncomputable def M : Set ℝ := {x | -1 < x ∧ x < 2}
noncomputable def N (m : ℝ) : Set ℝ := {x | x*x - m*x < 0}
noncomputable def M_inter_N (m : ℝ) : Set ℝ := {x | 0 < x ∧ x < 1}

theorem find_m (m : ℝ) (h : M ∩ (N m) = M_inter_N m) : m = 1 :=
by sorry

end find_m_l105_10596


namespace max_y_difference_eq_l105_10569

theorem max_y_difference_eq (x y p q : ℤ) (hp : 0 < p) (hq : 0 < q)
  (h : x * y = p * x + q * y) : y - x = (p - 1) * (q + 1) :=
sorry

end max_y_difference_eq_l105_10569


namespace min_coins_for_less_than_1_dollar_l105_10594

theorem min_coins_for_less_than_1_dollar :
  ∃ (p n q h : ℕ), 1*p + 5*n + 25*q + 50*h ≥ 1 ∧ 1*p + 5*n + 25*q + 50*h < 100 ∧ p + n + q + h = 8 :=
by 
  sorry

end min_coins_for_less_than_1_dollar_l105_10594


namespace smallest_k_remainder_2_l105_10550

theorem smallest_k_remainder_2 (k : ℕ) :
  k > 1 ∧
  k % 13 = 2 ∧
  k % 7 = 2 ∧
  k % 3 = 2 →
  k = 275 :=
by sorry

end smallest_k_remainder_2_l105_10550


namespace always_negative_l105_10588

noncomputable def f (x : ℝ) : ℝ := 
  Real.log (Real.sqrt (x ^ 2 + 1) - x) - Real.sin x

theorem always_negative (a b : ℝ) (ha : a ∈ Set.Ioo (-Real.pi/2) (Real.pi/2))
                     (hb : b ∈ Set.Ioo (-Real.pi/2) (Real.pi/2))
                     (hab : a + b ≠ 0) : 
  (f a + f b) / (a + b) < 0 := 
sorry

end always_negative_l105_10588


namespace part1_part2_l105_10571

def f (x a : ℝ) : ℝ := |x + a| + |x - a^2|

theorem part1 (x : ℝ) : f x 1 ≥ 4 ↔ x ≤ -2 ∨ x ≥ 2 := sorry

theorem part2 (m : ℝ) : (∀ x : ℝ, ∃ a : ℝ, -1 < a ∧ a < 3 ∧ m < f x a) ↔ m < 12 := sorry

end part1_part2_l105_10571


namespace average_score_l105_10553

variable (K M : ℕ) (E : ℕ)

theorem average_score (h1 : (K + M) / 2 = 86) (h2 : E = 98) :
  (K + M + E) / 3 = 90 :=
by
  sorry

end average_score_l105_10553


namespace complex_problem_proof_l105_10519

open Complex

noncomputable def z : ℂ := (1 - I)^2 + 1 + 3 * I

theorem complex_problem_proof : z = 1 + I ∧ abs (z - 2 * I) = Real.sqrt 2 ∧ (∀ a b : ℝ, (z^2 + a + b = 1 - I) → (a = -3 ∧ b = 4)) := 
by
  have h1 : z = (1 - I)^2 + 1 + 3 * I := rfl
  have h2 : z = 1 + I := sorry
  have h3 : abs (z - 2 * I) = Real.sqrt 2 := sorry
  have h4 : (∀ a b : ℝ, (z^2 + a + b = 1 - I) → (a = -3 ∧ b = 4)) := sorry
  exact ⟨h2, h3, h4⟩

end complex_problem_proof_l105_10519


namespace derivative_correct_l105_10527

noncomputable def f (x : ℝ) : ℝ := 
  (1 / (2 * Real.sqrt 2)) * (Real.sin (Real.log x) - (Real.sqrt 2 - 1) * Real.cos (Real.log x)) * x^(Real.sqrt 2 + 1)

noncomputable def df (x : ℝ) : ℝ := 
  (x^(Real.sqrt 2)) / (2 * Real.sqrt 2) * (2 * Real.cos (Real.log x) - Real.sqrt 2 * Real.cos (Real.log x) + 2 * Real.sqrt 2 * Real.sin (Real.log x))

theorem derivative_correct (x : ℝ) (hx : 0 < x) :
  deriv f x = df x := by
  sorry

end derivative_correct_l105_10527


namespace total_number_of_subjects_l105_10545

-- Definitions from conditions
def average_marks_5_subjects (total_marks : ℕ) : Prop :=
  74 * 5 = total_marks

def marks_in_last_subject (marks : ℕ) : Prop :=
  marks = 74

def total_average_marks (n : ℕ) (total_marks : ℕ) : Prop :=
  74 * n = total_marks

-- Lean 4 statement
theorem total_number_of_subjects (n total_marks total_marks_5 last_subject_marks : ℕ)
  (h1 : total_average_marks n total_marks)
  (h2 : average_marks_5_subjects total_marks_5)
  (h3 : marks_in_last_subject last_subject_marks)
  (h4 : total_marks = total_marks_5 + last_subject_marks) :
  n = 6 :=
sorry

end total_number_of_subjects_l105_10545


namespace paint_cost_l105_10595

theorem paint_cost {width height : ℕ} (price_per_quart coverage_area : ℕ) (total_cost : ℕ) :
  width = 5 → height = 4 → price_per_quart = 2 → coverage_area = 4 → total_cost = 20 :=
by
  intros h1 h2 h3 h4
  have area_one_side : ℕ := width * height
  have total_area : ℕ := 2 * area_one_side
  have quarts_needed : ℕ := total_area / coverage_area
  have cost : ℕ := quarts_needed * price_per_quart
  sorry

end paint_cost_l105_10595


namespace specific_clothing_choice_probability_l105_10520

noncomputable def probability_of_specific_clothing_choice : ℚ :=
  let total_clothing := 4 + 5 + 6
  let total_ways_to_choose_3 := Nat.choose 15 3
  let ways_to_choose_specific_3 := 4 * 5 * 6
  let probability := ways_to_choose_specific_3 / total_ways_to_choose_3
  probability

theorem specific_clothing_choice_probability :
  probability_of_specific_clothing_choice = 24 / 91 :=
by
  -- proof here 
  sorry

end specific_clothing_choice_probability_l105_10520


namespace fraction_numerator_greater_than_denominator_l105_10552

theorem fraction_numerator_greater_than_denominator (x : ℝ) :
  (4 * x + 2 > 8 - 3 * x) ↔ (6 / 7 < x ∧ x ≤ 3) :=
by
  sorry

end fraction_numerator_greater_than_denominator_l105_10552


namespace time_for_A_to_complete_work_l105_10524

-- Defining the work rates and the condition
def workRateA (a : ℕ) : ℚ := 1 / a
def workRateB : ℚ := 1 / 12
def workRateC : ℚ := 1 / 24
def combinedWorkRate (a : ℕ) : ℚ := workRateA a + workRateB + workRateC
def togetherWorkRate : ℚ := 1 / 4

-- Stating the theorem
theorem time_for_A_to_complete_work : 
  ∃ (a : ℕ), combinedWorkRate a = togetherWorkRate ∧ a = 8 :=
by
  sorry

end time_for_A_to_complete_work_l105_10524


namespace second_coloring_book_pictures_l105_10590

-- Let P1 be the number of pictures in the first coloring book.
def P1 := 23

-- Let P2 be the number of pictures in the second coloring book.
variable (P2 : Nat)

-- Let colored_pics be the number of pictures Rachel colored.
def colored_pics := 44

-- Let remaining_pics be the number of pictures Rachel still has to color.
def remaining_pics := 11

-- Total number of pictures in both coloring books.
def total_pics := colored_pics + remaining_pics

theorem second_coloring_book_pictures :
  P2 = total_pics - P1 :=
by
  -- We need to prove that P2 = 32.
  sorry

end second_coloring_book_pictures_l105_10590


namespace remainder_equality_l105_10505

variables (A B D : ℕ) (S S' s s' : ℕ)

theorem remainder_equality 
  (h1 : A > B) 
  (h2 : (A + 3) % D = S) 
  (h3 : (B - 2) % D = S') 
  (h4 : ((A + 3) * (B - 2)) % D = s) 
  (h5 : (S * S') % D = s') : 
  s = s' := 
sorry

end remainder_equality_l105_10505


namespace num_ways_distribute_balls_l105_10562

-- Definition of the combinatorial problem
def indistinguishableBallsIntoBoxes : ℕ := 11

-- Main theorem statement
theorem num_ways_distribute_balls : indistinguishableBallsIntoBoxes = 11 := by
  sorry

end num_ways_distribute_balls_l105_10562


namespace div_neg_forty_five_l105_10532

theorem div_neg_forty_five : (-40 / 5) = -8 :=
by
  sorry

end div_neg_forty_five_l105_10532


namespace solution_of_loginequality_l105_10589

-- Define the conditions as inequalities
def condition1 (x : ℝ) : Prop := 2 * x - 1 > 0
def condition2 (x : ℝ) : Prop := -x + 5 > 0
def condition3 (x : ℝ) : Prop := 2 * x - 1 > -x + 5

-- Define the final solution set
def solution_set (x : ℝ) : Prop := (2 < x) ∧ (x < 5)

-- The theorem stating that under the given conditions, the solution set holds
theorem solution_of_loginequality (x : ℝ) : condition1 x ∧ condition2 x ∧ condition3 x → solution_set x :=
by
  intro h
  sorry

end solution_of_loginequality_l105_10589


namespace find_A_l105_10504

def heartsuit (A B : ℤ) : ℤ := 4 * A + A * B + 3 * B + 6

theorem find_A (A : ℤ) : heartsuit A 3 = 75 ↔ A = 60 / 7 := sorry

end find_A_l105_10504


namespace never_attains_95_l105_10578

def dihedral_angle_condition (α β : ℝ) : Prop :=
  0 < α ∧ 0 < β ∧ α + β < 90

theorem never_attains_95 (α β : ℝ) (h : dihedral_angle_condition α β) :
  α + β ≠ 95 :=
by
  sorry

end never_attains_95_l105_10578


namespace evaluate_M_l105_10541

noncomputable def M : ℝ := 
  (Real.sqrt (Real.sqrt 7 + 3) + Real.sqrt (Real.sqrt 7 - 3)) / Real.sqrt (Real.sqrt 7 + 2) - Real.sqrt (5 - 2 * Real.sqrt 6)

theorem evaluate_M : M = (1 + Real.sqrt 3 + Real.sqrt 5 + 3 * Real.sqrt 2) / 3 :=
by
  sorry

end evaluate_M_l105_10541


namespace sector_area_150_degrees_l105_10503

def sector_area (radius : ℝ) (central_angle : ℝ) : ℝ :=
  0.5 * radius^2 * central_angle

theorem sector_area_150_degrees (r : ℝ) (angle_rad : ℝ) (h1 : r = Real.sqrt 3) (h2 : angle_rad = (5 * Real.pi) / 6) : 
  sector_area r angle_rad = (5 * Real.pi) / 4 :=
by
  simp [sector_area, h1, h2]
  sorry

end sector_area_150_degrees_l105_10503


namespace Roger_years_to_retire_l105_10598

noncomputable def Peter : ℕ := 12
noncomputable def Robert : ℕ := Peter - 4
noncomputable def Mike : ℕ := Robert - 2
noncomputable def Tom : ℕ := 2 * Robert
noncomputable def Roger : ℕ := Peter + Tom + Robert + Mike

theorem Roger_years_to_retire :
  Roger = 42 → 50 - Roger = 8 := by
sorry

end Roger_years_to_retire_l105_10598


namespace cars_already_parked_l105_10567

-- Define the levels and their parking spaces based on given conditions
def first_level_spaces : Nat := 90
def second_level_spaces : Nat := first_level_spaces + 8
def third_level_spaces : Nat := second_level_spaces + 12
def fourth_level_spaces : Nat := third_level_spaces - 9

-- Compute total spaces in the garage
def total_spaces : Nat := first_level_spaces + second_level_spaces + third_level_spaces + fourth_level_spaces

-- Define the available spaces for more cars
def available_spaces : Nat := 299

-- Prove the number of cars already parked
theorem cars_already_parked : total_spaces - available_spaces = 100 :=
by
  exact Nat.sub_eq_of_eq_add sorry -- Fill in with the actual proof step

end cars_already_parked_l105_10567


namespace total_calories_in_jerrys_breakfast_l105_10565

theorem total_calories_in_jerrys_breakfast :
  let pancakes := 7 * 120
  let bacon := 3 * 100
  let orange_juice := 2 * 300
  let cereal := 1 * 200
  let chocolate_muffin := 1 * 350
  pancakes + bacon + orange_juice + cereal + chocolate_muffin = 2290 :=
by
  -- Proof omitted
  sorry

end total_calories_in_jerrys_breakfast_l105_10565


namespace planting_area_l105_10560

variable (x : ℝ)

def garden_length := x + 2
def garden_width := 4
def path_width := 1

def effective_garden_length := garden_length x - 2 * path_width
def effective_garden_width := garden_width - 2 * path_width

theorem planting_area : effective_garden_length x * effective_garden_width = 2 * x := by
  simp [garden_length, garden_width, path_width, effective_garden_length, effective_garden_width]
  sorry

end planting_area_l105_10560


namespace farmer_rows_of_tomatoes_l105_10507

def num_rows (total_tomatoes yield_per_plant plants_per_row : ℕ) : ℕ :=
  (total_tomatoes / yield_per_plant) / plants_per_row

theorem farmer_rows_of_tomatoes (total_tomatoes yield_per_plant plants_per_row : ℕ)
    (ht : total_tomatoes = 6000)
    (hy : yield_per_plant = 20)
    (hp : plants_per_row = 10) :
    num_rows total_tomatoes yield_per_plant plants_per_row = 30 := 
by
  sorry

end farmer_rows_of_tomatoes_l105_10507


namespace maximum_area_of_flower_bed_l105_10512

-- Definitions based on conditions
def length_of_flower_bed : ℝ := 150
def total_fencing : ℝ := 450

-- Question reframed as a proof statement
theorem maximum_area_of_flower_bed :
  ∀ (w : ℝ), 2 * w + length_of_flower_bed = total_fencing → (length_of_flower_bed * w = 22500) :=
by
  intro w h
  sorry

end maximum_area_of_flower_bed_l105_10512


namespace work_done_in_one_day_l105_10533

theorem work_done_in_one_day (A_days B_days : ℝ) (hA : A_days = 6) (hB : B_days = A_days / 2) : 
  (1 / A_days + 1 / B_days) = 1 / 2 := by
  sorry

end work_done_in_one_day_l105_10533


namespace math_problem_l105_10537

theorem math_problem :
  2.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 5000 := 
by
  sorry

end math_problem_l105_10537


namespace calories_burned_l105_10516

/-- 
  The football coach makes his players run up and down the bleachers 60 times. 
  Each time they run up and down, they encounter 45 stairs. 
  The first half of the staircase has 20 stairs and every stair burns 3 calories, 
  while the second half has 25 stairs burning 4 calories each. 
  Prove that each player burns 9600 calories during this exercise.
--/
theorem calories_burned (n_stairs_first_half : ℕ) (calories_first_half : ℕ) 
  (n_stairs_second_half : ℕ) (calories_second_half : ℕ) (n_trips : ℕ) 
  (total_calories : ℕ) :
  n_stairs_first_half = 20 → calories_first_half = 3 → 
  n_stairs_second_half = 25 → calories_second_half = 4 → 
  n_trips = 60 → total_calories = 
  (n_stairs_first_half * calories_first_half + n_stairs_second_half * calories_second_half) * n_trips →
  total_calories = 9600 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end calories_burned_l105_10516


namespace ivan_total_money_l105_10538

-- Define values of the coins
def penny_value : ℝ := 0.01
def dime_value : ℝ := 0.1
def nickel_value : ℝ := 0.05
def quarter_value : ℝ := 0.25

-- Define number of each type of coin in each piggy bank
def first_piggybank_pennies := 100
def first_piggybank_dimes := 50
def first_piggybank_nickels := 20
def first_piggybank_quarters := 10

def second_piggybank_pennies := 150
def second_piggybank_dimes := 30
def second_piggybank_nickels := 40
def second_piggybank_quarters := 15

def third_piggybank_pennies := 200
def third_piggybank_dimes := 60
def third_piggybank_nickels := 10
def third_piggybank_quarters := 20

-- Calculate the total value of each piggy bank
def first_piggybank_value : ℝ :=
  (first_piggybank_pennies * penny_value) +
  (first_piggybank_dimes * dime_value) +
  (first_piggybank_nickels * nickel_value) +
  (first_piggybank_quarters * quarter_value)

def second_piggybank_value : ℝ :=
  (second_piggybank_pennies * penny_value) +
  (second_piggybank_dimes * dime_value) +
  (second_piggybank_nickels * nickel_value) +
  (second_piggybank_quarters * quarter_value)

def third_piggybank_value : ℝ :=
  (third_piggybank_pennies * penny_value) +
  (third_piggybank_dimes * dime_value) +
  (third_piggybank_nickels * nickel_value) +
  (third_piggybank_quarters * quarter_value)

-- Calculate the total amount of money Ivan has
def total_value : ℝ :=
  first_piggybank_value + second_piggybank_value + third_piggybank_value

-- The theorem to prove
theorem ivan_total_money :
  total_value = 33.25 :=
by
  sorry

end ivan_total_money_l105_10538


namespace integer_solutions_l105_10515

def satisfies_equation (x y : ℤ) : Prop := x^2 = y^2 * (x + y^4 + 2 * y^2)

theorem integer_solutions :
  {p : ℤ × ℤ | satisfies_equation p.1 p.2} = { (0, 0), (12, 2), (-8, 2) } :=
by sorry

end integer_solutions_l105_10515


namespace tommy_nickels_l105_10543

-- Definitions of given conditions
def pennies (quarters : Nat) : Nat := 10 * quarters  -- Tommy has 10 times as many pennies as quarters
def dimes (pennies : Nat) : Nat := pennies + 10      -- Tommy has 10 more dimes than pennies
def nickels (dimes : Nat) : Nat := 2 * dimes         -- Tommy has twice as many nickels as dimes

theorem tommy_nickels (quarters : Nat) (P : Nat) (D : Nat) (N : Nat) 
  (h1 : quarters = 4) 
  (h2 : P = pennies quarters) 
  (h3 : D = dimes P) 
  (h4 : N = nickels D) : 
  N = 100 := 
by
  -- sorry allows us to skip the proof
  sorry

end tommy_nickels_l105_10543


namespace proportion_solution_l105_10586

theorem proportion_solution (x : ℝ) (h : 0.75 / x = 5 / 8) : x = 1.2 :=
by
  sorry

end proportion_solution_l105_10586


namespace vertex_of_parabola_l105_10535

theorem vertex_of_parabola :
  (∃ x y : ℝ, y = -3*x^2 + 6*x + 1 ∧ (x, y) = (1, 4)) :=
sorry

end vertex_of_parabola_l105_10535


namespace robotics_club_neither_l105_10500

theorem robotics_club_neither (n c e b neither : ℕ) (h1 : n = 80) (h2 : c = 50) (h3 : e = 40) (h4 : b = 25) :
  neither = n - (c - b + e - b + b) :=
by 
  rw [h1, h2, h3, h4]
  sorry

end robotics_club_neither_l105_10500


namespace last_digit_1993_2002_plus_1995_2002_l105_10547

theorem last_digit_1993_2002_plus_1995_2002 :
  (1993 ^ 2002 + 1995 ^ 2002) % 10 = 4 :=
by sorry

end last_digit_1993_2002_plus_1995_2002_l105_10547


namespace nested_fraction_value_l105_10587

theorem nested_fraction_value : 
  let expr := 1 / (3 - (1 / (3 - (1 / (3 - (1 / (3 - (1 / 3))))))))
  expr = 21 / 55 :=
by 
  sorry

end nested_fraction_value_l105_10587


namespace intersection_complement_is_singleton_l105_10555

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {1, 2, 5}

theorem intersection_complement_is_singleton : (U \ M) ∩ N = {1} := by
  sorry

end intersection_complement_is_singleton_l105_10555


namespace second_vote_difference_l105_10508

-- Define the total number of members
def total_members : ℕ := 300

-- Define the votes for and against in the initial vote
structure votes_initial :=
  (a : ℕ) (b : ℕ) (h : a + b = total_members) (rejected : b > a)

-- Define the votes for and against in the second vote
structure votes_second :=
  (a' : ℕ) (b' : ℕ) (h : a' + b' = total_members)

-- Define the margin and condition of passage by three times the margin
def margin (vi : votes_initial) : ℕ := vi.b - vi.a

def passage_by_margin (vi : votes_initial) (vs : votes_second) : Prop :=
  vs.a' - vs.b' = 3 * margin vi

-- Define the condition that a' is 7/6 times b
def proportion (vs : votes_second) (vi : votes_initial) : Prop :=
  vs.a' = (7 * vi.b) / 6

-- The final proof statement
theorem second_vote_difference (vi : votes_initial) (vs : votes_second)
  (h_margin : passage_by_margin vi vs)
  (h_proportion : proportion vs vi) :
  vs.a' - vi.a = 55 :=
by
  sorry  -- This is where the proof would go

end second_vote_difference_l105_10508


namespace calculate_final_speed_l105_10556

noncomputable def final_speed : ℝ :=
  let v1 : ℝ := (150 * 1.60934 * 1000) / 3600
  let v2 : ℝ := (170 * 1000) / 3600
  let v_decreased : ℝ := v1 - v2
  let a : ℝ := (500000 * 0.01) / 60
  v_decreased + a * (30 * 60)

theorem calculate_final_speed : final_speed = 150013.45 :=
by
  sorry

end calculate_final_speed_l105_10556


namespace age_of_youngest_child_l105_10580

theorem age_of_youngest_child
  (total_bill : ℝ)
  (mother_charge : ℝ)
  (child_charge_per_year : ℝ)
  (children_total_years : ℝ)
  (twins_age : ℕ)
  (youngest_child_age : ℕ)
  (h_total_bill : total_bill = 13.00)
  (h_mother_charge : mother_charge = 6.50)
  (h_child_charge_per_year : child_charge_per_year = 0.65)
  (h_children_bill : total_bill - mother_charge = children_total_years * child_charge_per_year)
  (h_children_age : children_total_years = 10)
  (h_youngest_child : youngest_child_age = 10 - 2 * twins_age) :
  youngest_child_age = 2 ∨ youngest_child_age = 4 :=
by
  sorry

end age_of_youngest_child_l105_10580


namespace each_girl_brought_2_cups_l105_10558

-- Here we define all the given conditions
def total_students : ℕ := 30
def num_boys : ℕ := 10
def cups_per_boy : ℕ := 5
def total_cups : ℕ := 90

-- Define the conditions as Lean definitions
def num_girls : ℕ := total_students - num_boys -- From condition 3
def cups_by_boys : ℕ := num_boys * cups_per_boy
def cups_by_girls : ℕ := total_cups - cups_by_boys

-- Define the final question and expected answer
def cups_per_girl : ℕ := cups_by_girls / num_girls

-- Final problem statement to prove
theorem each_girl_brought_2_cups :
  cups_per_girl = 2 :=
by
  have h1 : num_girls = 20 := by sorry
  have h2 : cups_by_boys = 50 := by sorry
  have h3 : cups_by_girls = 40 := by sorry
  have h4 : cups_per_girl = 2 := by sorry
  exact h4

end each_girl_brought_2_cups_l105_10558


namespace quadratic_no_real_roots_l105_10521

theorem quadratic_no_real_roots (m : ℝ) : (4 + 4 * m < 0) → (m < -1) :=
by
  intro h
  linarith

end quadratic_no_real_roots_l105_10521


namespace unique_primes_solution_l105_10510

theorem unique_primes_solution (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) : 
    p + q^2 = r^4 ↔ (p = 7 ∧ q = 3 ∧ r = 2) := 
by
  sorry

end unique_primes_solution_l105_10510


namespace jaclyn_constant_term_l105_10581

variable {R : Type*} [CommRing R] (P Q : Polynomial R)

theorem jaclyn_constant_term (hP : P.leadingCoeff = 1) (hQ : Q.leadingCoeff = 1)
  (deg_P : P.degree = 4) (deg_Q : Q.degree = 4)
  (constant_terms_eq : P.coeff 0 = Q.coeff 0)
  (coeff_z_eq : P.coeff 1 = Q.coeff 1)
  (product_eq : P * Q = Polynomial.C 1 * 
    Polynomial.C 1 * Polynomial.C 1 * Polynomial.C (-1) *
    Polynomial.C 1) :
  Jaclyn's_constant_term = 3 :=
sorry

end jaclyn_constant_term_l105_10581


namespace floyd_infinite_jumps_l105_10522

def sum_of_digits (n: Nat) : Nat := 
  n.digits 10 |>.sum 

noncomputable def jumpable (a b: Nat) : Prop := 
  b > a ∧ b ≤ 2 * a 

theorem floyd_infinite_jumps :
  ∃ f : ℕ → ℕ, 
    (∀ n : ℕ, jumpable (f n) (f (n + 1))) ∧
    (∀ m n : ℕ, m ≠ n → sum_of_digits (f m) ≠ sum_of_digits (f n)) :=
sorry

end floyd_infinite_jumps_l105_10522


namespace problem_l105_10526

theorem problem 
  (k a b c : ℝ)
  (h1 : (3 : ℝ)^2 - 7 * 3 + k = 0)
  (h2 : (a : ℝ)^2 - 7 * a + k = 0)
  (h3 : (b : ℝ)^2 - 8 * b + (k + 1) = 0)
  (h4 : (c : ℝ)^2 - 8 * c + (k + 1) = 0) :
  a + b * c = 17 := sorry

end problem_l105_10526


namespace friends_came_over_later_l105_10582

def original_friends : ℕ := 4
def total_people : ℕ := 7

theorem friends_came_over_later : (total_people - original_friends = 3) :=
sorry

end friends_came_over_later_l105_10582


namespace date_behind_D_correct_l105_10549

noncomputable def date_behind_B : ℕ := sorry
noncomputable def date_behind_E : ℕ := date_behind_B + 2
noncomputable def date_behind_F : ℕ := date_behind_B + 15
noncomputable def date_behind_D : ℕ := sorry

theorem date_behind_D_correct :
  date_behind_B + date_behind_D = date_behind_E + date_behind_F := sorry

end date_behind_D_correct_l105_10549


namespace real_distance_between_cities_l105_10506

-- Condition: the map distance between Goteborg and Jonkoping
def map_distance_cm : ℝ := 88

-- Condition: the map scale
def map_scale_km_per_cm : ℝ := 15

-- The real distance to be proven
theorem real_distance_between_cities :
  (map_distance_cm * map_scale_km_per_cm) = 1320 := by
  sorry

end real_distance_between_cities_l105_10506


namespace largest_t_value_l105_10563

theorem largest_t_value : 
  ∃ t : ℝ, 
    (∃ s : ℝ, s > 0 ∧ t = 3 ∧
    ∀ u : ℝ, 
      (u = 3 →
        (15 * u^2 - 40 * u + 18) / (4 * u - 3) + 3 * u = 4 * u + 2 ∧
        u ≤ 3) ∧
      (u ≠ 3 → 
        (15 * u^2 - 40 * u + 18) / (4 * u - 3) + 3 * u = 4 * u + 2 → 
        u ≤ 3)) :=
sorry

end largest_t_value_l105_10563


namespace problem_statement_l105_10509

noncomputable def AB2_AC2_BC2_eq_4 (l m n k : ℝ) : Prop :=
  let D := (l+k, 0, 0)
  let E := (0, m+k, 0)
  let F := (0, 0, n+k)
  let AB_sq := 4 * (n+k)^2
  let AC_sq := 4 * (m+k)^2
  let BC_sq := 4 * (l+k)^2
  AB_sq + AC_sq + BC_sq = 4 * ((l+k)^2 + (m+k)^2 + (n+k)^2)

theorem problem_statement (l m n k : ℝ) : 
  AB2_AC2_BC2_eq_4 l m n k :=
by
  sorry

end problem_statement_l105_10509


namespace intersection_eq_l105_10576

-- Define sets A and B
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {0, 1, 2}

-- The theorem to be proved
theorem intersection_eq : A ∩ B = {2} := 
by sorry

end intersection_eq_l105_10576


namespace crayons_given_proof_l105_10573

def initial_crayons : ℕ := 110
def total_lost_crayons : ℕ := 412
def more_lost_than_given : ℕ := 322

def G : ℕ := 45 -- This is the given correct answer to prove.

theorem crayons_given_proof :
  ∃ G : ℕ, (G + (G + more_lost_than_given)) = total_lost_crayons ∧ G = 45 :=
by
  sorry

end crayons_given_proof_l105_10573


namespace train_speed_kmph_l105_10514

theorem train_speed_kmph (length : ℝ) (time : ℝ) (speed_conversion : ℝ) (speed_kmph : ℝ) :
  length = 100.008 → time = 4 → speed_conversion = 3.6 →
  speed_kmph = (length / time) * speed_conversion → speed_kmph = 90.0072 :=
by
  sorry

end train_speed_kmph_l105_10514


namespace students_wearing_blue_lipstick_l105_10570

theorem students_wearing_blue_lipstick
  (total_students : ℕ)
  (half_students_wore_lipstick : total_students / 2 = 180)
  (red_fraction : ℚ)
  (pink_fraction : ℚ)
  (purple_fraction : ℚ)
  (green_fraction : ℚ)
  (students_wearing_red : red_fraction * 180 = 45)
  (students_wearing_pink : pink_fraction * 180 = 60)
  (students_wearing_purple : purple_fraction * 180 = 30)
  (students_wearing_green : green_fraction * 180 = 15)
  (total_red_fraction : red_fraction = 1 / 4)
  (total_pink_fraction : pink_fraction = 1 / 3)
  (total_purple_fraction : purple_fraction = 1 / 6)
  (total_green_fraction : green_fraction = 1 / 12) :
  (180 - (45 + 60 + 30 + 15) = 30) :=
by sorry

end students_wearing_blue_lipstick_l105_10570


namespace volume_of_cube_l105_10544

theorem volume_of_cube (d : ℝ) (h : d = 5 * Real.sqrt 3) : ∃ (V : ℝ), V = 125 := by
  sorry

end volume_of_cube_l105_10544


namespace problem_discussion_organization_l105_10584

theorem problem_discussion_organization 
    (students : Fin 20 → Finset (Fin 20))
    (problems : Fin 20 → Finset (Fin 20))
    (h1 : ∀ s, (students s).card = 2)
    (h2 : ∀ p, (problems p).card = 2)
    (h3 : ∀ s p, s ∈ problems p ↔ p ∈ students s) : 
    ∃ (discussion : Fin 20 → Fin 20), 
        (∀ s, discussion s ∈ students s) ∧ 
        (Finset.univ.image discussion).card = 20 :=
by
  -- proof goes here
  sorry

end problem_discussion_organization_l105_10584


namespace brian_tape_needed_l105_10530

-- Definitions of conditions
def tape_needed_for_box (short_side: ℕ) (long_side: ℕ) : ℕ := 
  2 * short_side + long_side

def total_tape_needed (num_short_long_boxes: ℕ) (short_side: ℕ) (long_side: ℕ) (num_square_boxes: ℕ) (side: ℕ) : ℕ := 
  (num_short_long_boxes * tape_needed_for_box short_side long_side) + (num_square_boxes * 3 * side)

-- Theorem statement
theorem brian_tape_needed : total_tape_needed 5 15 30 2 40 = 540 := 
by 
  sorry

end brian_tape_needed_l105_10530


namespace range_omega_l105_10597

noncomputable def f (ω x : ℝ) := Real.cos (ω * x + Real.pi / 6)

theorem range_omega (ω : ℝ) (hω : ω > 0) :
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi → -1 ≤ f ω x ∧ f ω x ≤ Real.sqrt 3 / 2) →
  ω ∈ Set.Icc (5 / 6) (5 / 3) :=
  sorry

end range_omega_l105_10597


namespace number_of_undeveloped_sections_l105_10523

def undeveloped_sections (total_area section_area : ℕ) : ℕ :=
  total_area / section_area

theorem number_of_undeveloped_sections :
  undeveloped_sections 7305 2435 = 3 :=
by
  unfold undeveloped_sections
  exact rfl

end number_of_undeveloped_sections_l105_10523


namespace modulus_of_z_l105_10557

open Complex

theorem modulus_of_z 
  (z : ℂ) 
  (h : (1 - I) * z = 2 * I) : 
  abs z = Real.sqrt 2 := 
sorry

end modulus_of_z_l105_10557


namespace initial_amount_is_825_l105_10501

theorem initial_amount_is_825 (P R : ℝ) 
    (h1 : 956 = P * (1 + 3 * R / 100))
    (h2 : 1055 = P * (1 + 3 * (R + 4) / 100)) : 
    P = 825 := 
by 
  sorry

end initial_amount_is_825_l105_10501
