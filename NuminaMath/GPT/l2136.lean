import Mathlib

namespace concentration_of_salt_solution_l2136_213627

-- Conditions:
def total_volume : ℝ := 1 + 0.25
def concentration_of_mixture : ℝ := 0.15
def volume_of_salt_solution : ℝ := 0.25

-- Expression for the concentration of the salt solution used, $C$:
theorem concentration_of_salt_solution (C : ℝ) :
  (volume_of_salt_solution * (C / 100)) = (total_volume * concentration_of_mixture) → C = 75 := by
  sorry

end concentration_of_salt_solution_l2136_213627


namespace math_proof_problem_l2136_213685

variables {a b c d e f k : ℝ}

theorem math_proof_problem 
  (h1 : a + b + c = d + e + f)
  (h2 : a^2 + b^2 + c^2 = d^2 + e^2 + f^2)
  (h3 : a^3 + b^3 + c^3 ≠ d^3 + e^3 + f^3) :
  (a + b + c + (d + k) + (e + k) + (f + k) = d + e + f + (a + k) + (b + k) + (c + k) ∧
   a^2 + b^2 + c^2 + (d + k)^2 + (e + k)^2 + (f + k)^2 = d^2 + e^2 + f^2 + (a + k)^2 + (b + k)^2 + (c + k)^2 ∧
   a^3 + b^3 + c^3 + (d + k)^3 + (e + k)^3 + (f + k)^3 = d^3 + e^3 + f^3 + (a + k)^3 + (b + k)^3 + (c + k)^3) 
   ∧ 
  (a^4 + b^4 + c^4 + (d + k)^4 + (e + k)^4 + (f + k)^4 ≠ d^4 + e^4 + f^4 + (a + k)^4 + (b + k)^4 + (c + k)^4) := 
  sorry

end math_proof_problem_l2136_213685


namespace angles_equal_l2136_213629

theorem angles_equal (A B C : ℝ) (h1 : A + B + C = Real.pi) (h2 : Real.sin A = 2 * Real.cos B * Real.sin C) : B = C :=
by sorry

end angles_equal_l2136_213629


namespace loom_weaving_rate_l2136_213622

noncomputable def total_cloth : ℝ := 27
noncomputable def total_time : ℝ := 210.9375

theorem loom_weaving_rate :
  (total_cloth / total_time) = 0.128 :=
by
  sorry

end loom_weaving_rate_l2136_213622


namespace total_area_of_removed_triangles_l2136_213632

theorem total_area_of_removed_triangles (a b : ℝ)
  (square_side : ℝ := 16)
  (triangle_hypotenuse : ℝ := 8)
  (isosceles_right_triangle : a = b ∧ a^2 + b^2 = triangle_hypotenuse^2) :
  4 * (1 / 2 * a * b) = 64 :=
by
  -- Sketch of the proof:
  -- From the isosceles right triangle property and Pythagorean theorem,
  -- a^2 + b^2 = 8^2 ⇒ 2 * a^2 = 64 ⇒ a^2 = 32 ⇒ a = b = 4√2
  -- The area of one triangle is (1/2) * a * b = 16
  -- Total area of four such triangles is 4 * 16 = 64
  sorry

end total_area_of_removed_triangles_l2136_213632


namespace roots_equal_and_real_l2136_213673

theorem roots_equal_and_real (a c : ℝ) (h : 32 - 4 * a * c = 0) :
  ∃ x : ℝ, x = (2 * Real.sqrt 2) / a := 
by sorry

end roots_equal_and_real_l2136_213673


namespace fragment_probability_l2136_213650

noncomputable def probability_fragment_in_21_digit_code : ℚ :=
  (12 * 10^11 - 30) / 10^21

theorem fragment_probability:
  ∀ (code : Fin 10 → Fin 21 → Fin 10),
  (∃ (i : Fin 12), ∀ (j : Fin 10), code (i + j) = j) → 
  probability_fragment_in_21_digit_code = (12 * 10^11 - 30) / 10^21 :=
sorry

end fragment_probability_l2136_213650


namespace evaluate_expression_l2136_213678

theorem evaluate_expression :
  (3 ^ 4 * 5 ^ 2 * 7 ^ 3 * 11) / (7 * 11 ^ 2) = 9025 :=
by 
  sorry

end evaluate_expression_l2136_213678


namespace exists_integers_greater_than_N_l2136_213679

theorem exists_integers_greater_than_N (N : ℝ) : 
  ∃ (x1 x2 x3 x4 : ℤ), (x1 > N) ∧ (x2 > N) ∧ (x3 > N) ∧ (x4 > N) ∧ 
  (x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4 = x1 * x2 * x3 + x1 * x2 * x4 + x1 * x3 * x4 + x2 * x3 * x4) := 
sorry

end exists_integers_greater_than_N_l2136_213679


namespace sum_of_six_terms_l2136_213615

variable {a : ℕ → ℝ} {q : ℝ}

/-- Given conditions:
* a is a decreasing geometric sequence with ratio q
-/
def is_decreasing_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

theorem sum_of_six_terms
  (h_geo : is_decreasing_geometric_sequence a q)
  (h_decreasing : 0 < q ∧ q < 1)
  (h_a1 : 0 < a 1)
  (h_a1a3 : a 1 * a 3 = 1)
  (h_a2a4 : a 2 + a 4 = 5 / 4) :
  (a 1 * (1 - q^6) / (1 - q)) = 63 / 16 := by
  sorry

end sum_of_six_terms_l2136_213615


namespace prime_sum_l2136_213694

theorem prime_sum (m n : ℕ) (hm : Prime m) (hn : Prime n) (h : 5 * m + 7 * n = 129) :
  m + n = 19 ∨ m + n = 25 := by
  sorry

end prime_sum_l2136_213694


namespace cube_edge_length_l2136_213676

theorem cube_edge_length
  (length_base : ℝ) (width_base : ℝ) (rise_level : ℝ) (volume_displaced : ℝ) (volume_cube : ℝ) (edge_length : ℝ)
  (h_base : length_base = 20) (h_width : width_base = 15) (h_rise : rise_level = 3.3333333333333335)
  (h_volume_displaced : volume_displaced = length_base * width_base * rise_level)
  (h_volume_cube : volume_cube = volume_displaced)
  (h_edge_length_eq : volume_cube = edge_length ^ 3)
  : edge_length = 10 :=
by
  sorry

end cube_edge_length_l2136_213676


namespace complement_set_l2136_213652

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x | x^2 - 4 ≤ 0}

-- Define the complement of M in U
def complement_M_in_U : Set ℝ := {x | x < -2 ∨ x > 2}

-- The mathematical proof to be stated
theorem complement_set :
  U \ M = complement_M_in_U := sorry

end complement_set_l2136_213652


namespace ratio_paperback_fiction_to_nonfiction_l2136_213619

-- Definitions
def total_books := 160
def hardcover_nonfiction := 25
def paperback_nonfiction := hardcover_nonfiction + 20
def paperback_fiction := total_books - hardcover_nonfiction - paperback_nonfiction

-- Theorem statement
theorem ratio_paperback_fiction_to_nonfiction : paperback_fiction / paperback_nonfiction = 2 :=
by
  -- proof details would go here
  sorry

end ratio_paperback_fiction_to_nonfiction_l2136_213619


namespace Dacid_weighted_average_l2136_213696

noncomputable def DacidMarks := 86 * 3 + 85 * 4 + 92 * 4 + 87 * 3 + 95 * 3 + 89 * 2 + 75 * 1
noncomputable def TotalCreditHours := 3 + 4 + 4 + 3 + 3 + 2 + 1
noncomputable def WeightedAverageMarks := (DacidMarks : ℝ) / (TotalCreditHours : ℝ)

theorem Dacid_weighted_average :
  WeightedAverageMarks = 88.25 :=
sorry

end Dacid_weighted_average_l2136_213696


namespace inequality_solution_set_minimum_value_mn_squared_l2136_213690

noncomputable def f (x : ℝ) := |x - 2| + |x + 1|

theorem inequality_solution_set : 
  (∀ x, f x > 7 ↔ x > 4 ∨ x < -3) :=
by sorry

theorem minimum_value_mn_squared (m n : ℝ) (hm : n > 0) (hmin : ∀ x, f x ≥ m + n) :
  m^2 + n^2 = 9 / 2 ∧ m = 3 / 2 ∧ n = 3 / 2 :=
by sorry

end inequality_solution_set_minimum_value_mn_squared_l2136_213690


namespace complete_the_square_l2136_213677

theorem complete_the_square (x : ℝ) : 
    (x^2 - 2 * x - 5 = 0) -> (x - 1)^2 = 6 :=
by sorry

end complete_the_square_l2136_213677


namespace constant_term_binomial_expansion_l2136_213684

theorem constant_term_binomial_expansion :
  ∀ (x : ℝ), ((2 / x) + x) ^ 4 = 24 :=
by
  sorry

end constant_term_binomial_expansion_l2136_213684


namespace smallest_x_value_l2136_213686

theorem smallest_x_value :
  ∃ x, (x ≠ 9) ∧ (∀ y, (y ≠ 9) → ((x^2 - x - 72) / (x - 9) = 3 / (x + 6)) → x ≤ y) ∧ x = -9 :=
by
  sorry

end smallest_x_value_l2136_213686


namespace subset_B_of_A_l2136_213600

def A : Set ℕ := {2, 0, 3}
def B : Set ℕ := {2, 3}

theorem subset_B_of_A : B ⊆ A :=
by
  sorry

end subset_B_of_A_l2136_213600


namespace yule_log_surface_area_increase_l2136_213687

theorem yule_log_surface_area_increase :
  let h := 10
  let d := 5
  let r := d / 2
  let n := 9
  let initial_surface_area := 2 * Real.pi * r * h + 2 * Real.pi * r^2
  let slice_height := h / n
  let slice_surface_area := 2 * Real.pi * r * slice_height + 2 * Real.pi * r^2
  let total_surface_area_slices := n * slice_surface_area
  let delta_surface_area := total_surface_area_slices - initial_surface_area
  delta_surface_area = 100 * Real.pi :=
by
  sorry

end yule_log_surface_area_increase_l2136_213687


namespace number_of_cells_after_9_days_l2136_213633

theorem number_of_cells_after_9_days : 
  let initial_cells := 4 
  let doubling_period := 3 
  let total_duration := 9 
  ∀ cells_after_9_days, cells_after_9_days = initial_cells * 2^(total_duration / doubling_period) 
  → cells_after_9_days = 32 :=
by
  sorry

end number_of_cells_after_9_days_l2136_213633


namespace truncated_cone_volume_l2136_213608

theorem truncated_cone_volume :
  let R := 10
  let r := 5
  let h_t := 10
  let V_large := (1/3:Real) * Real.pi * (R^2) * (20)
  let V_small := (1/3:Real) * Real.pi * (r^2) * (10)
  (V_large - V_small) = (1750/3) * Real.pi :=
by
  sorry

end truncated_cone_volume_l2136_213608


namespace domain_tan_x_plus_pi_over_3_l2136_213651

open Real Set

theorem domain_tan_x_plus_pi_over_3 :
  ∀ x : ℝ, ¬ (∃ k : ℤ, x = k * π + π / 6) ↔ x ∈ {x : ℝ | ¬ ∃ k : ℤ, x = k * π + π / 6} :=
by {
  sorry
}

end domain_tan_x_plus_pi_over_3_l2136_213651


namespace area_outside_two_small_squares_l2136_213688

theorem area_outside_two_small_squares (L S : ℝ) (hL : L = 9) (hS : S = 4) :
  let large_square_area := L^2
  let small_square_area := S^2
  let combined_small_squares_area := 2 * small_square_area
  large_square_area - combined_small_squares_area = 49 :=
by
  sorry

end area_outside_two_small_squares_l2136_213688


namespace Alex_final_silver_tokens_l2136_213638

variable (x y : ℕ)

def final_red_tokens (x y : ℕ) : ℕ := 90 - 3 * x + 2 * y
def final_blue_tokens (x y : ℕ) : ℕ := 65 + 2 * x - 4 * y
def silver_tokens (x y : ℕ) : ℕ := x + y

theorem Alex_final_silver_tokens (h1 : final_red_tokens x y < 3)
                                 (h2 : final_blue_tokens x y < 4) :
  silver_tokens x y = 67 := 
sorry

end Alex_final_silver_tokens_l2136_213638


namespace parallelogram_perimeter_l2136_213606

theorem parallelogram_perimeter 
  (EF FG EH : ℝ)
  (hEF : EF = 40) (hFG : FG = 30) (hEH : EH = 50) : 
  2 * (EF + FG) = 140 := 
by 
  rw [hEF, hFG]
  norm_num

end parallelogram_perimeter_l2136_213606


namespace vampire_needs_7_gallons_per_week_l2136_213697

-- Define conditions given in the problem
def pints_per_person : ℕ := 2
def people_per_day : ℕ := 4
def days_per_week : ℕ := 7
def pints_per_gallon : ℕ := 8

-- Prove the vampire needs 7 gallons of blood per week to survive
theorem vampire_needs_7_gallons_per_week :
  (pints_per_person * people_per_day * days_per_week) / pints_per_gallon = 7 := 
by 
  sorry

end vampire_needs_7_gallons_per_week_l2136_213697


namespace B_days_to_complete_job_alone_l2136_213681

theorem B_days_to_complete_job_alone (x : ℝ) : 
  (1 / 15 + 1 / x) * 4 = 0.4666666666666667 → x = 20 :=
by
  intro h
  -- Note: The proof is omitted as we only need the statement here.
  sorry

end B_days_to_complete_job_alone_l2136_213681


namespace plates_to_remove_l2136_213612

-- Definitions based on the problem conditions
def number_of_plates : ℕ := 38
def weight_per_plate : ℕ := 10
def acceptable_weight : ℕ := 320

-- Theorem to prove
theorem plates_to_remove (initial_weight := number_of_plates * weight_per_plate) 
  (excess_weight := initial_weight - acceptable_weight) 
  (plates_to_remove := excess_weight / weight_per_plate) :
  plates_to_remove = 6 :=
by
  sorry

end plates_to_remove_l2136_213612


namespace dad_vacuum_time_l2136_213691

theorem dad_vacuum_time (x : ℕ) (h1 : 2 * x + 5 = 27) (h2 : x + (2 * x + 5) = 38) :
  (2 * x + 5) = 27 := by
  sorry

end dad_vacuum_time_l2136_213691


namespace imo1965_cmo6511_l2136_213634

theorem imo1965_cmo6511 (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2 * Real.pi) :
  2 * Real.cos x ≤ |(Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x)))| ∧
  |(Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x)))| ≤ Real.sqrt 2 ↔
  ((Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 2) ∨ (3 * Real.pi / 2 ≤ x ∧ x ≤ 7 * Real.pi / 4)) :=
sorry

end imo1965_cmo6511_l2136_213634


namespace cobbler_hours_per_day_l2136_213672

-- Defining some conditions based on our problem statement
def cobbler_rate : ℕ := 3  -- pairs of shoes per hour
def friday_hours : ℕ := 3  -- number of hours worked on Friday
def friday_pairs : ℕ := cobbler_rate * friday_hours  -- pairs mended on Friday
def weekly_pairs : ℕ := 105  -- total pairs mended in a week
def mon_thu_pairs : ℕ := weekly_pairs - friday_pairs  -- pairs mended from Monday to Thursday
def mon_thu_hours : ℕ := mon_thu_pairs / cobbler_rate  -- total hours worked from Monday to Thursday

-- Thm statement: If a cobbler works h hours daily from Mon to Thu, then h = 8 implies total = 105 pairs
theorem cobbler_hours_per_day (h : ℕ) : (4 * h = mon_thu_hours) ↔ (h = 8) :=
by
  sorry

end cobbler_hours_per_day_l2136_213672


namespace math_homework_pages_l2136_213655

-- Define Rachel's total pages, math homework pages, and reading homework pages
def total_pages : ℕ := 13
def reading_homework : ℕ := sorry
def math_homework (r : ℕ) : ℕ := r + 3

-- State the main theorem that needs to be proved
theorem math_homework_pages :
  ∃ r : ℕ, r + (math_homework r) = total_pages ∧ (math_homework r) = 8 :=
by {
  sorry
}

end math_homework_pages_l2136_213655


namespace must_be_nonzero_l2136_213630

noncomputable def Q (a b c d : ℝ) : ℝ → ℝ :=
  λ x => x^5 + a * x^4 + b * x^3 + c * x^2 + d * x

theorem must_be_nonzero (a b c d : ℝ)
  (h_roots : ∃ p q r s : ℝ, (∀ y : ℝ, Q a b c d y = 0 → y = 0 ∨ y = -1 ∨ y = p ∨ y = q ∨ y = r ∨ y = s) ∧ p ≠ 0 ∧ p ≠ -1 ∧ q ≠ 0 ∧ q ≠ -1 ∧ r ≠ 0 ∧ r ≠ -1 ∧ s ≠ 0 ∧ s ≠ -1)
  (h_distinct : (∀ x₁ x₂ : ℝ, Q a b c d x₁ = 0 ∧ Q a b c d x₂ = 0 → x₁ ≠ x₂ ∨ x₁ = x₂) → False)
  (h_f_zero : Q a b c d 0 = 0) :
  d ≠ 0 := by
  sorry

end must_be_nonzero_l2136_213630


namespace parabola_directrix_l2136_213666

theorem parabola_directrix (x y : ℝ) : 
    (x^2 = (1/2) * y) -> (y = -1/8) :=
sorry

end parabola_directrix_l2136_213666


namespace find_f_3_l2136_213628

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_3 : 
  (∀ (x : ℝ), x ≠ 0 → 27 * f (-x) / x - x^2 * f (1 / x) = - 2 * x^2) →
  f 3 = 2 :=
sorry

end find_f_3_l2136_213628


namespace subset_singleton_zero_A_l2136_213614

def A : Set ℝ := {x | x > -3}

theorem subset_singleton_zero_A : {0} ⊆ A := 
by
  sorry  -- Proof is not required

end subset_singleton_zero_A_l2136_213614


namespace find_f500_l2136_213626

variable (f : ℕ → ℕ)
variable (h : ∀ x y : ℕ, f (x * y) = f x + f y)
variable (h₁ : f 10 = 16)
variable (h₂ : f 40 = 24)

theorem find_f500 : f 500 = 44 :=
sorry

end find_f500_l2136_213626


namespace find_y_l2136_213649

theorem find_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 18) : y = 5 :=
sorry

end find_y_l2136_213649


namespace cherries_count_l2136_213699

theorem cherries_count (b s r c : ℝ) 
  (h1 : b + s + r + c = 360)
  (h2 : s = 2 * b)
  (h3 : r = 4 * s)
  (h4 : c = 2 * r) : 
  c = 640 / 3 :=
by 
  sorry

end cherries_count_l2136_213699


namespace total_packs_l2136_213646

noncomputable def robyn_packs : ℕ := 16
noncomputable def lucy_packs : ℕ := 19

theorem total_packs : robyn_packs + lucy_packs = 35 := by
  sorry

end total_packs_l2136_213646


namespace find_g_l2136_213640

noncomputable def g (x : ℝ) : ℝ := sorry

axiom functional_equation : ∀ x y : ℝ, (g x * g y - g (x * y)) / 4 = x + y + 4

theorem find_g :
  (∀ x : ℝ, g x = x + 5) ∨ (∀ x : ℝ, g x = -x - 3) := 
by
  sorry

end find_g_l2136_213640


namespace no_valid_n_lt_200_l2136_213683

noncomputable def roots_are_consecutive (n m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * (k + 1) ∧ n = 2 * k + 1

theorem no_valid_n_lt_200 :
  ¬∃ n m : ℕ, n < 200 ∧
              m % 4 = 0 ∧
              ∃ t : ℕ, t^2 = m ∧
              roots_are_consecutive n m := 
by
  sorry

end no_valid_n_lt_200_l2136_213683


namespace molecular_weight_BaCl2_l2136_213631

theorem molecular_weight_BaCl2 (mw8 : ℝ) (n : ℝ) (h : mw8 = 1656) : (mw8 / n = 207) ↔ n = 8 := 
by
  sorry

end molecular_weight_BaCl2_l2136_213631


namespace count_even_numbers_between_500_and_800_l2136_213693

theorem count_even_numbers_between_500_and_800 :
  let a := 502
  let d := 2
  let last_term := 798
  ∃ n, a + (n - 1) * d = last_term ∧ n = 149 :=
by
  sorry

end count_even_numbers_between_500_and_800_l2136_213693


namespace sum_of_fraction_parts_l2136_213637

theorem sum_of_fraction_parts (x : ℝ) (hx : x = 0.45) : 
  (∃ (a b : ℕ), x = a / b ∧ Nat.gcd a b = 1 ∧ a + b = 16) :=
by
  sorry

end sum_of_fraction_parts_l2136_213637


namespace evaluate_f_at_3_l2136_213643

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  x^7 + a * x^5 + b * x - 5

theorem evaluate_f_at_3 (a b : ℝ)
  (h : f (-3) a b = 5) : f 3 a b = -15 :=
by
  sorry

end evaluate_f_at_3_l2136_213643


namespace competition_end_time_is_5_35_am_l2136_213670

def start_time : Nat := 15 * 60  -- 3:00 p.m. in minutes
def duration : Nat := 875  -- competition duration in minutes
def end_time : Nat := (start_time + duration) % (24 * 60)  -- competition end time in minutes

theorem competition_end_time_is_5_35_am :
  end_time = 5 * 60 + 35 :=  -- 5:35 a.m. in minutes
sorry

end competition_end_time_is_5_35_am_l2136_213670


namespace vector_magnitude_difference_l2136_213669

-- Defining the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Statement to prove that the magnitude of the difference of vectors a and b is 5
theorem vector_magnitude_difference : ‖a - b‖ = 5 := 
sorry -- Proof omitted

end vector_magnitude_difference_l2136_213669


namespace DollOutfit_l2136_213609

variables (VeraDress OlyaCoat VeraCoat NinaCoat : Prop)
axiom FirstAnswer : (VeraDress ∧ ¬OlyaCoat) ∨ (¬VeraDress ∧ OlyaCoat)
axiom SecondAnswer : (VeraCoat ∧ ¬NinaCoat) ∨ (¬VeraCoat ∧ NinaCoat)
axiom OnlyOneTrueFirstAnswer : (VeraDress ∨ OlyaCoat) ∧ ¬(VeraDress ∧ OlyaCoat)
axiom OnlyOneTrueSecondAnswer : (VeraCoat ∨ NinaCoat) ∧ ¬(VeraCoat ∧ NinaCoat)

theorem DollOutfit :
  VeraDress ∧ NinaCoat ∧ ¬OlyaCoat ∧ ¬VeraCoat ∧ ¬NinaCoat :=
sorry

end DollOutfit_l2136_213609


namespace white_mice_count_l2136_213695

variable (T W B : ℕ) -- Declare variables T (total), W (white), B (brown)

def W_condition := W = (2 / 3) * T  -- White mice condition
def B_condition := B = 7           -- Brown mice condition
def T_condition := T = W + B       -- Total mice condition

theorem white_mice_count : W = 14 :=
by
  sorry  -- Proof to be filled in

end white_mice_count_l2136_213695


namespace part_a_l2136_213662

theorem part_a {d m b : ℕ} (h_d : d = 41) (h_m : m = 28) (h_b : b = 15) :
    d - b + m - b + b = 54 :=
  by sorry

end part_a_l2136_213662


namespace students_not_taken_test_l2136_213674

theorem students_not_taken_test 
  (num_enrolled : ℕ) 
  (answered_q1 : ℕ) 
  (answered_q2 : ℕ) 
  (answered_both : ℕ) 
  (H_num_enrolled : num_enrolled = 40) 
  (H_answered_q1 : answered_q1 = 30) 
  (H_answered_q2 : answered_q2 = 29) 
  (H_answered_both : answered_both = 29) : 
  num_enrolled - (answered_q1 + answered_q2 - answered_both) = 10 :=
by {
  sorry
}

end students_not_taken_test_l2136_213674


namespace tax_budget_level_correct_l2136_213618

-- Definitions for tax types and their corresponding budget levels
inductive TaxType where
| property_tax_organizations : TaxType
| federal_tax : TaxType
| profit_tax_organizations : TaxType
| tax_subjects_RF : TaxType
| transport_collecting : TaxType
deriving DecidableEq

inductive BudgetLevel where
| federal_budget : BudgetLevel
| subjects_RF_budget : BudgetLevel
deriving DecidableEq

def tax_to_budget_level : TaxType → BudgetLevel
| TaxType.property_tax_organizations => BudgetLevel.subjects_RF_budget
| TaxType.federal_tax => BudgetLevel.federal_budget
| TaxType.profit_tax_organizations => BudgetLevel.subjects_RF_budget
| TaxType.tax_subjects_RF => BudgetLevel.subjects_RF_budget
| TaxType.transport_collecting => BudgetLevel.subjects_RF_budget

theorem tax_budget_level_correct :
  tax_to_budget_level TaxType.property_tax_organizations = BudgetLevel.subjects_RF_budget ∧
  tax_to_budget_level TaxType.federal_tax = BudgetLevel.federal_budget ∧
  tax_to_budget_level TaxType.profit_tax_organizations = BudgetLevel.subjects_RF_budget ∧
  tax_to_budget_level TaxType.tax_subjects_RF = BudgetLevel.subjects_RF_budget ∧
  tax_to_budget_level TaxType.transport_collecting = BudgetLevel.subjects_RF_budget :=
by
  sorry

end tax_budget_level_correct_l2136_213618


namespace two_digit_sum_reverse_l2136_213617

theorem two_digit_sum_reverse (a b : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ 9)
    (h₃ : 0 ≤ b) (h₄ : b ≤ 9)
    (h₅ : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
    (10 * a + b) + (10 * b + a) = 99 := 
by
  sorry

end two_digit_sum_reverse_l2136_213617


namespace prob_truth_same_time_l2136_213658

theorem prob_truth_same_time (pA pB : ℝ) (hA : pA = 0.85) (hB : pB = 0.60) :
  pA * pB = 0.51 :=
by
  rw [hA, hB]
  norm_num

end prob_truth_same_time_l2136_213658


namespace no_rational_multiples_pi_tan_sum_two_l2136_213689

theorem no_rational_multiples_pi_tan_sum_two (x y : ℚ) (hx : 0 < x * π ∧ x * π < y * π ∧ y * π < π / 2) (hxy : Real.tan (x * π) + Real.tan (y * π) = 2) : False :=
sorry

end no_rational_multiples_pi_tan_sum_two_l2136_213689


namespace arcsin_one_half_l2136_213620

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  -- Conditions
  have h1 : -Real.pi / 2 ≤ Real.pi / 6 ∧ Real.pi / 6 ≤ Real.pi / 2 := by
    -- Proof the range of pi/6 is within [-pi/2, pi/2]
    sorry
  have h2 : ∀ x, Real.sin x = 1 / 2 → x = Real.pi / 6 := by
    -- Proof sin(pi/6) = 1 / 2
    sorry
  show Real.arcsin (1 / 2) = Real.pi / 6
  -- Proof arcsin(1/2) = pi/6 based on the above conditions
  sorry

end arcsin_one_half_l2136_213620


namespace A_equals_half_C_equals_half_l2136_213645

noncomputable def A := 2 * Real.sin (75 * Real.pi / 180) * Real.cos (75 * Real.pi / 180)
noncomputable def C := Real.sin (45 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) - Real.cos (45 * Real.pi / 180) * Real.sin (15 * Real.pi / 180)

theorem A_equals_half : A = 1 / 2 := 
by
  sorry

theorem C_equals_half : C = 1 / 2 := 
by
  sorry

end A_equals_half_C_equals_half_l2136_213645


namespace more_pups_than_adult_dogs_l2136_213656

def number_of_huskies := 5
def number_of_pitbulls := 2
def number_of_golden_retrievers := 4
def pups_per_husky := 3
def pups_per_pitbull := 3
def additional_pups_per_golden_retriever := 2
def pups_per_golden_retriever := pups_per_husky + additional_pups_per_golden_retriever

def total_pups := (number_of_huskies * pups_per_husky) + (number_of_pitbulls * pups_per_pitbull) + (number_of_golden_retrievers * pups_per_golden_retriever)
def total_adult_dogs := number_of_huskies + number_of_pitbulls + number_of_golden_retrievers

theorem more_pups_than_adult_dogs : (total_pups - total_adult_dogs) = 30 :=
by
  -- proof steps, which we will skip
  sorry

end more_pups_than_adult_dogs_l2136_213656


namespace product_of_roots_l2136_213613

theorem product_of_roots : ∃ (x : ℕ), x = 45 ∧ (∃ a b c : ℕ, a ^ 3 = 27 ∧ b ^ 4 = 81 ∧ c ^ 2 = 25 ∧ x = a * b * c) := 
sorry

end product_of_roots_l2136_213613


namespace cupboard_cost_price_l2136_213641

theorem cupboard_cost_price
  (C : ℝ)
  (h1 : ∀ (S : ℝ), S = 0.84 * C) -- Vijay sells a cupboard at 84% of the cost price.
  (h2 : ∀ (S_new : ℝ), S_new = 1.16 * C) -- If Vijay got Rs. 1200 more, he would have made a profit of 16%.
  (h3 : ∀ (S_new S : ℝ), S_new - S = 1200) -- The difference between new selling price and original selling price is Rs. 1200.
  : C = 3750 := 
sorry -- Proof is not required.

end cupboard_cost_price_l2136_213641


namespace prime_polynomial_l2136_213671

theorem prime_polynomial (n : ℕ) (h1 : 2 ≤ n)
  (h2 : ∀ k : ℕ, k ≤ Nat.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) :
  ∀ k : ℕ, k ≤ n - 2 → Nat.Prime (k^2 + k + n) :=
sorry

end prime_polynomial_l2136_213671


namespace totalNumberOfCrayons_l2136_213682

def numOrangeCrayons (numBoxes : ℕ) (crayonsPerBox : ℕ) : ℕ :=
  numBoxes * crayonsPerBox

def numBlueCrayons (numBoxes : ℕ) (crayonsPerBox : ℕ) : ℕ :=
  numBoxes * crayonsPerBox

def numRedCrayons (numBoxes : ℕ) (crayonsPerBox : ℕ) : ℕ :=
  numBoxes * crayonsPerBox

theorem totalNumberOfCrayons :
  numOrangeCrayons 6 8 + numBlueCrayons 7 5 + numRedCrayons 1 11 = 94 :=
by
  sorry

end totalNumberOfCrayons_l2136_213682


namespace sum_of_solutions_l2136_213657

theorem sum_of_solutions (x y : ℝ) (h1 : y = 9) (h2 : x^2 + y^2 = 225) : 2 * x = 0 :=
by
  sorry

end sum_of_solutions_l2136_213657


namespace valentines_left_l2136_213603

theorem valentines_left (initial valentines_to_children valentines_to_neighbors valentines_to_coworkers : ℕ) (h_initial : initial = 30)
  (h1 : valentines_to_children = 8) (h2 : valentines_to_neighbors = 5) (h3 : valentines_to_coworkers = 3) : initial - (valentines_to_children + valentines_to_neighbors + valentines_to_coworkers) = 14 := by
  sorry

end valentines_left_l2136_213603


namespace minimize_rental_cost_l2136_213663

def travel_agency (x y : ℕ) : ℕ := 1600 * x + 2400 * y

theorem minimize_rental_cost :
    ∃ (x y : ℕ), (x + y ≤ 21) ∧ (y ≤ x + 7) ∧ (36 * x + 60 * y = 900) ∧ 
    (∀ (a b : ℕ), (a + b ≤ 21) ∧ (b ≤ a + 7) ∧ (36 * a + 60 * b = 900) → travel_agency a b ≥ travel_agency x y) ∧
    travel_agency x y = 36800 :=
sorry

end minimize_rental_cost_l2136_213663


namespace greatest_number_of_fruit_baskets_l2136_213625

def number_of_oranges : ℕ := 18
def number_of_pears : ℕ := 27
def number_of_bananas : ℕ := 12

theorem greatest_number_of_fruit_baskets :
  Nat.gcd (Nat.gcd number_of_oranges number_of_pears) number_of_bananas = 3 :=
by
  sorry

end greatest_number_of_fruit_baskets_l2136_213625


namespace minimum_pencils_needed_l2136_213601

theorem minimum_pencils_needed (red_pencils blue_pencils : ℕ) (total_pencils : ℕ) 
  (h_red : red_pencils = 7) (h_blue : blue_pencils = 4) (h_total : total_pencils = red_pencils + blue_pencils) :
  (∃ n : ℕ, n = 8 ∧ n ≤ total_pencils ∧ (∀ m : ℕ, m < 8 → (m < red_pencils ∨ m < blue_pencils))) :=
by
  sorry

end minimum_pencils_needed_l2136_213601


namespace divisible_by_five_l2136_213664

theorem divisible_by_five {x y z : ℤ} (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) : 
  5 ∣ ((x - y)^5 + (y - z)^5 + (z - x)^5) :=
sorry

end divisible_by_five_l2136_213664


namespace arithmetic_series_remainder_l2136_213680

-- Define the sequence parameters
def a : ℕ := 2
def l : ℕ := 12
def d : ℕ := 1
def n : ℕ := (l - a) / d + 1

-- Define the sum of the arithmetic series
def S : ℕ := n * (a + l) / 2

-- The final theorem statement
theorem arithmetic_series_remainder : S % 9 = 5 := 
by sorry

end arithmetic_series_remainder_l2136_213680


namespace time_after_seconds_l2136_213692

def initial_time : Nat × Nat × Nat := (4, 45, 0)
def seconds_to_add : Nat := 12345
def final_time : Nat × Nat × Nat := (8, 30, 45)

theorem time_after_seconds (h : initial_time = (4, 45, 0) ∧ seconds_to_add = 12345) : 
  ∃ (h' : Nat × Nat × Nat), h' = final_time := by
  sorry

end time_after_seconds_l2136_213692


namespace initial_machines_count_l2136_213610

theorem initial_machines_count (M : ℕ) (h1 : M * 8 = 8 * 1) (h2 : 72 * 6 = 12 * 2) : M = 64 :=
by
  sorry

end initial_machines_count_l2136_213610


namespace managers_salary_l2136_213661

-- Definitions based on conditions
def avg_salary_50_employees : ℝ := 2000
def num_employees : ℕ := 50
def new_avg_salary : ℝ := 2150
def num_employees_with_manager : ℕ := 51

-- Condition statement: The manager's salary such that when added, average salary increases as given.
theorem managers_salary (M : ℝ) :
  (num_employees * avg_salary_50_employees + M) / num_employees_with_manager = new_avg_salary →
  M = 9650 := sorry

end managers_salary_l2136_213661


namespace quotient_with_zero_in_middle_l2136_213668

theorem quotient_with_zero_in_middle : 
  ∃ (op : ℕ → ℕ → ℕ), 
  (op = Nat.add ∧ ((op 6 4) / 3).digits 10 = [3, 0, 3]) := 
by 
  sorry

end quotient_with_zero_in_middle_l2136_213668


namespace seats_scientific_notation_l2136_213698

theorem seats_scientific_notation : 
  (13000 = 1.3 * 10^4) := 
by 
  sorry 

end seats_scientific_notation_l2136_213698


namespace sum_num_den_252_l2136_213616

theorem sum_num_den_252 (h : (252 : ℤ) / 100 = (63 : ℤ) / 25) : 63 + 25 = 88 :=
by
  sorry

end sum_num_den_252_l2136_213616


namespace triangle_third_side_length_l2136_213605

theorem triangle_third_side_length 
  (a b c : ℝ) (ha : a = 7) (hb : b = 11) (hc : c = 3) :
  (4 < c ∧ c < 18) → c ≠ 3 :=
by
  sorry

end triangle_third_side_length_l2136_213605


namespace negation_proposition_equiv_l2136_213660

open Classical

variable (R : Type) [OrderedRing R] (a x : R)

theorem negation_proposition_equiv :
  (¬ ∃ a : R, ∃ x : R, a * x^2 + 1 = 0) ↔ (∀ a : R, ∀ x : R, a * x^2 + 1 ≠ 0) :=
by
  sorry

end negation_proposition_equiv_l2136_213660


namespace degree_polynomial_is_13_l2136_213635

noncomputable def degree_polynomial (a b c d e f g h j : ℝ) : ℕ :=
  (7 + 4 + 2)

theorem degree_polynomial_is_13 (a b c d e f g h j : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0) (hf : f ≠ 0) (hg : g ≠ 0) (hh : h ≠ 0) (hj : j ≠ 0) : 
  degree_polynomial a b c d e f g h j = 13 :=
by
  rfl

end degree_polynomial_is_13_l2136_213635


namespace problem_1_problem_2_l2136_213659

def f (x : ℝ) : ℝ := |(1 - 2 * x)| - |(1 + x)|

theorem problem_1 :
  {x | f x ≥ 4} = {x | x ≤ -2 ∨ x ≥ 6} :=
sorry

theorem problem_2 (a : ℝ) : 
  (∀ x : ℝ, a^2 + 2 * a + |(1 + x)| > f x) → (a < -3 ∨ a > 1) :=
sorry

end problem_1_problem_2_l2136_213659


namespace parabola_directrix_equation_l2136_213665

theorem parabola_directrix_equation (x y a : ℝ) : 
  (x^2 = 4 * y) → (a = 1) → (y = -a) := by
  intro h1 h2
  rw [h2] -- given a = 1
  sorry

end parabola_directrix_equation_l2136_213665


namespace triangle_proof_l2136_213653

-- Declare a structure for a triangle with given conditions
structure TriangleABC :=
  (a b c : ℝ) -- sides opposite to angles A, B, and C
  (A B C : ℝ) -- angles A, B, and C
  (R : ℝ) -- circumcircle radius
  (r : ℝ := 3) -- inradius is given as 3
  (area : ℝ := 6) -- area of the triangle is 6
  (h1 : a * Real.cos A + b * Real.cos B + c * Real.cos C = R / 3) -- given condition
  (h2 : ∀ a b c A B C, a * Real.sin A + b * Real.sin B + c * Real.sin C = 2 * area / (a+b+c)) -- implied area condition

-- Define the theorem using the above conditions
theorem triangle_proof (t : TriangleABC) :
  t.a + t.b + t.c = 4 ∧
  (Real.sin (2 * t.A) + Real.sin (2 * t.B) + Real.sin (2 * t.C)) = 1/3 ∧
  t.R = 6 :=
by
  sorry

end triangle_proof_l2136_213653


namespace number_of_valid_pairings_l2136_213602

-- Definition for the problem
def validPairingCount (n : ℕ) (k: ℕ) : ℕ :=
  sorry -- Calculating the valid number of pairings is deferred

-- The problem statement to be proven:
theorem number_of_valid_pairings : validPairingCount 12 3 = 14 :=
sorry

end number_of_valid_pairings_l2136_213602


namespace unique_four_digit_square_l2136_213654

theorem unique_four_digit_square (n : ℕ) : 
  1000 ≤ n ∧ n < 10000 ∧ 
  (n % 10 = (n / 10) % 10) ∧ 
  ((n / 100) % 10 = (n / 1000) % 10) ∧ 
  (∃ k : ℕ, n = k^2) ↔ n = 7744 := 
by 
  sorry

end unique_four_digit_square_l2136_213654


namespace avg_consecutive_integers_l2136_213642

theorem avg_consecutive_integers (a : ℝ) (b : ℝ) 
  (h₁ : b = (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5)) / 6) :
  (a + 5) = (b + (b + 1) + (b + 2) + (b + 3) + (b + 4) + (b + 5)) / 6 :=
by sorry

end avg_consecutive_integers_l2136_213642


namespace average_temp_tues_to_fri_l2136_213636

theorem average_temp_tues_to_fri (T W Th : ℕ) 
  (h1: (42 + T + W + Th) / 4 = 48) 
  (mon: 42 = 42) 
  (fri: 10 = 10) :
  (T + W + Th + 10) / 4 = 40 := by
  sorry

end average_temp_tues_to_fri_l2136_213636


namespace largest_3_digit_sum_l2136_213623

theorem largest_3_digit_sum : ∃ A B : ℕ, A ≠ B ∧ A < 10 ∧ B < 10 ∧ 100 ≤ 111 * A + 12 * B ∧ 111 * A + 12 * B = 996 := by
  sorry

end largest_3_digit_sum_l2136_213623


namespace min_amount_for_free_shipping_l2136_213621

def book1 : ℝ := 13.00
def book2 : ℝ := 15.00
def book3 : ℝ := 10.00
def book4 : ℝ := 10.00
def discount_rate : ℝ := 0.25
def shipping_threshold : ℝ := 9.00

def total_cost_before_discount : ℝ := book1 + book2 + book3 + book4
def discount_amount : ℝ := book1 * discount_rate + book2 * discount_rate
def total_cost_after_discount : ℝ := total_cost_before_discount - discount_amount

theorem min_amount_for_free_shipping : total_cost_after_discount + shipping_threshold = 50.00 :=
by
  sorry

end min_amount_for_free_shipping_l2136_213621


namespace trig_identity_l2136_213648

theorem trig_identity (A : ℝ) (h : Real.cos (π + A) = -1/2) : Real.sin (π / 2 + A) = 1/2 :=
by 
sorry

end trig_identity_l2136_213648


namespace find_value_of_m_l2136_213639

open Real

theorem find_value_of_m (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 2) : m = sqrt 10 := by
  sorry

end find_value_of_m_l2136_213639


namespace inequality_proof_l2136_213624

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≥ a^2 + c^2) : 
  (a * f - c * d)^2 ≤ (a * e - b * d)^2 + (b * f - c * e)^2 := 
by 
  sorry

end inequality_proof_l2136_213624


namespace arithmetic_sequence_sum_l2136_213644

variable {a : ℕ → ℕ}

noncomputable def is_arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (h_arith : is_arithmetic_seq a) (h_a5 : a 5 = 2) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 2 * 9 :=
by
  sorry

end arithmetic_sequence_sum_l2136_213644


namespace perimeter_remaining_shape_l2136_213607

theorem perimeter_remaining_shape (length width square1 square2 : ℝ) 
  (H_len : length = 50) (H_width : width = 20) 
  (H_sq1 : square1 = 12) (H_sq2 : square2 = 4) : 
  2 * (length + width) + 4 * (square1 + square2) = 204 :=
by 
  rw [H_len, H_width, H_sq1, H_sq2]
  sorry

end perimeter_remaining_shape_l2136_213607


namespace use_six_threes_to_get_100_use_five_threes_to_get_100_l2136_213604

theorem use_six_threes_to_get_100 : 100 = (333 / 3) - (33 / 3) :=
by
  -- proof steps go here
  sorry

theorem use_five_threes_to_get_100 : 100 = (33 * 3) + (3 / 3) :=
by
  -- proof steps go here
  sorry

end use_six_threes_to_get_100_use_five_threes_to_get_100_l2136_213604


namespace parabola_y_intercepts_l2136_213611

theorem parabola_y_intercepts : 
  (∀ y : ℝ, 3 * y^2 - 6 * y + 1 = 0) → (∃ y1 y2 : ℝ, y1 ≠ y2) :=
by sorry

end parabola_y_intercepts_l2136_213611


namespace choco_delight_remainder_l2136_213667

theorem choco_delight_remainder (m : ℕ) (h : m % 7 = 5) : (4 * m) % 7 = 6 := 
by 
  sorry

end choco_delight_remainder_l2136_213667


namespace starting_current_ratio_l2136_213647

theorem starting_current_ratio (running_current : ℕ) (units : ℕ) (total_current : ℕ)
    (h1 : running_current = 40) 
    (h2 : units = 3) 
    (h3 : total_current = 240) 
    (h4 : total_current = running_current * (units * starter_ratio)) :
    starter_ratio = 2 := 
sorry

end starting_current_ratio_l2136_213647


namespace quadratic_has_distinct_real_roots_l2136_213675

theorem quadratic_has_distinct_real_roots :
  ∃ (x y : ℝ), x ≠ y ∧ (x^2 - 3 * x - 1 = 0) ∧ (y^2 - 3 * y - 1 = 0) :=
by {
  sorry
}

end quadratic_has_distinct_real_roots_l2136_213675
