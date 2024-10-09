import Mathlib

namespace sin_pi_six_minus_alpha_eq_one_third_cos_two_answer_l1567_156790

theorem sin_pi_six_minus_alpha_eq_one_third_cos_two_answer
  (α : ℝ) (h1 : Real.sin (π / 6 - α) = 1 / 3) :
  2 * Real.cos (π / 6 + α / 2) ^ 2 - 1 = 1 / 3 := by
  sorry

end sin_pi_six_minus_alpha_eq_one_third_cos_two_answer_l1567_156790


namespace haley_marbles_l1567_156753

theorem haley_marbles (m : ℕ) (k : ℕ) (h1 : k = 2) (h2 : m = 28) : m / k = 14 :=
by sorry

end haley_marbles_l1567_156753


namespace neither_directly_nor_inversely_proportional_A_D_l1567_156716

-- Definitions for the equations where y is neither directly nor inversely proportional to x
def equationA (x y : ℝ) : Prop := x^2 + x * y = 0
def equationD (x y : ℝ) : Prop := 4 * x + y^2 = 7

-- Definition for direct or inverse proportionality
def isDirectlyProportional (x y : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ y = k * x
def isInverselyProportional (x y : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ x * y = k

-- Proposition that y is neither directly nor inversely proportional to x for equations A and D
theorem neither_directly_nor_inversely_proportional_A_D (x y : ℝ) :
  equationA x y ∧ equationD x y ∧ ¬isDirectlyProportional x y ∧ ¬isInverselyProportional x y :=
by sorry

end neither_directly_nor_inversely_proportional_A_D_l1567_156716


namespace proof_f_prime_at_2_l1567_156730

noncomputable def f_prime (x : ℝ) (f_prime_2 : ℝ) : ℝ :=
  2 * x + 2 * f_prime_2 - (1 / x)

theorem proof_f_prime_at_2 :
  ∃ (f_prime_2 : ℝ), f_prime 2 f_prime_2 = -7 / 2 :=
by
  sorry

end proof_f_prime_at_2_l1567_156730


namespace cupcake_frosting_l1567_156715

theorem cupcake_frosting :
  (let cagney_rate := (1 : ℝ) / 24
   let lacey_rate := (1 : ℝ) / 40
   let sammy_rate := (1 : ℝ) / 30
   let total_time := 12 * 60
   let combined_rate := cagney_rate + lacey_rate + sammy_rate
   total_time * combined_rate = 72) :=
by 
   -- Proof goes here
   sorry

end cupcake_frosting_l1567_156715


namespace natural_number_increased_by_one_l1567_156706

theorem natural_number_increased_by_one (a : ℕ) 
  (h : (a + 1) ^ 2 - a ^ 2 = 1001) : 
  a = 500 := 
sorry

end natural_number_increased_by_one_l1567_156706


namespace c_ge_a_plus_b_sin_half_C_l1567_156767

-- Define a triangle with sides a, b, and c opposite to angles A, B, and C respectively, with C being the angle at vertex C
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a)
  (angles_positive : 0 < A ∧ 0 < B ∧ 0 < C)
  (angles_sum : A + B + C = π)

namespace TriangleProveInequality

open Triangle

theorem c_ge_a_plus_b_sin_half_C (t : Triangle) :
  t.c ≥ (t.a + t.b) * Real.sin (t.C / 2) := sorry

end TriangleProveInequality

end c_ge_a_plus_b_sin_half_C_l1567_156767


namespace sold_on_saturday_l1567_156746

-- Define all the conditions provided in the question
def amount_sold_thursday : ℕ := 210
def amount_sold_friday : ℕ := 2 * amount_sold_thursday
def amount_sold_sunday (S : ℕ) : ℕ := (S / 2)
def total_planned_sold : ℕ := 500
def excess_sold : ℕ := 325

-- Total sold is the sum of sold amounts from Thursday to Sunday
def total_sold (S : ℕ) : ℕ := amount_sold_thursday + amount_sold_friday + S + amount_sold_sunday S

-- The theorem to prove
theorem sold_on_saturday : ∃ S : ℕ, total_sold S = total_planned_sold + excess_sold ∧ S = 130 :=
by
  sorry

end sold_on_saturday_l1567_156746


namespace intersect_graphs_exactly_four_l1567_156757

theorem intersect_graphs_exactly_four (A : ℝ) (hA : 0 < A) :
  (∃ x y : ℝ, y = A * x^2 ∧ x^2 + 2 * y^2 = A + 3) ↔ (∀ x1 y1 x2 y2 : ℝ, (y1 = A * x1^2 ∧ x1^2 + 2 * y1^2 = A + 3) ∧ (y2 = A * x2^2 ∧ x2^2 + 2 * y2^2 = A + 3) → (x1, y1) ≠ (x2, y2)) :=
by
  sorry

end intersect_graphs_exactly_four_l1567_156757


namespace area_of_triangle_formed_by_tangency_points_l1567_156740

theorem area_of_triangle_formed_by_tangency_points :
  let r1 := 1
  let r2 := 3
  let r3 := 5
  let O1O2 := r1 + r2
  let O2O3 := r2 + r3
  let O1O3 := r1 + r3
  let s := (O1O2 + O2O3 + O1O3) / 2
  let A := Real.sqrt (s * (s - O1O2) * (s - O2O3) * (s - O1O3))
  let r := A / s
  r^2 = 5 / 3 := 
by
  sorry

end area_of_triangle_formed_by_tangency_points_l1567_156740


namespace fraction_addition_l1567_156704

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l1567_156704


namespace elena_bread_max_flour_l1567_156789

variable (butter_per_cup_flour butter sugar_per_cup_flour sugar : ℕ)
variable (available_butter available_sugar : ℕ)

def max_flour (butter_per_cup_flour butter sugar_per_cup_flour sugar : ℕ)
  (available_butter available_sugar : ℕ) : ℕ :=
  min (available_butter * sugar / butter_per_cup_flour) (available_sugar * butter / sugar_per_cup_flour)

theorem elena_bread_max_flour : 
  max_flour 3 4 2 5 24 30 = 32 := sorry

end elena_bread_max_flour_l1567_156789


namespace nail_polish_count_l1567_156769

-- Definitions from conditions
def K : ℕ := 25
def H : ℕ := K + 8
def Ka : ℕ := K - 6
def L : ℕ := 2 * K
def S : ℕ := 13 + 10  -- Since 25 / 2 = 12.5, rounded to 13 for practical purposes

-- Statement to prove
def T : ℕ := H + Ka + L + S

theorem nail_polish_count : T = 125 := by
  sorry

end nail_polish_count_l1567_156769


namespace common_root_unique_k_l1567_156781

theorem common_root_unique_k (k : ℝ) (x : ℝ) 
  (h₁ : x^2 + k * x - 12 = 0) 
  (h₂ : 3 * x^2 - 8 * x - 3 * k = 0) 
  : k = 1 :=
sorry

end common_root_unique_k_l1567_156781


namespace largest_number_is_b_l1567_156799

noncomputable def a := 0.935
noncomputable def b := 0.9401
noncomputable def c := 0.9349
noncomputable def d := 0.9041
noncomputable def e := 0.9400

theorem largest_number_is_b : b > a ∧ b > c ∧ b > d ∧ b > e :=
by
  -- proof can be filled in here
  sorry

end largest_number_is_b_l1567_156799


namespace max_gold_coins_l1567_156796

theorem max_gold_coins (n k : ℕ) 
  (h1 : n = 8 * k + 4)
  (h2 : n < 150) : 
  n = 148 :=
by
  sorry

end max_gold_coins_l1567_156796


namespace find_m_l1567_156712

theorem find_m (x1 x2 m : ℝ) (h_eq : ∀ x, x^2 + x + m = 0 → (x = x1 ∨ x = x2))
  (h_abs : |x1| + |x2| = 3)
  (h_sum : x1 + x2 = -1)
  (h_prod : x1 * x2 = m) :
  m = -2 :=
sorry

end find_m_l1567_156712


namespace average_marks_increase_ratio_l1567_156700

theorem average_marks_increase_ratio
  (T : ℕ)  -- The correct total marks of the class
  (n : ℕ)  -- The number of pupils in the class
  (h_n : n = 16) (wrong_mark : ℕ) (correct_mark : ℕ)  -- The wrong and correct marks
  (h_wrong : wrong_mark = 73) (h_correct : correct_mark = 65) :
  (8 : ℚ) / T = (wrong_mark - correct_mark : ℚ) / n * (n / T) :=
by
  sorry

end average_marks_increase_ratio_l1567_156700


namespace sum_first_12_terms_l1567_156733

-- Defining the basic sequence recurrence relation
def seq (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) + (-1 : ℝ) ^ n * a n = 2 * (n : ℝ) - 1

-- Theorem statement: Sum of the first 12 terms of the given sequence is 78
theorem sum_first_12_terms (a : ℕ → ℝ) (h : seq a) : 
  (Finset.range 12).sum a = 78 := 
sorry

end sum_first_12_terms_l1567_156733


namespace find_b_collinear_points_l1567_156727

theorem find_b_collinear_points :
  ∃ b : ℚ, 4 * 11 - 6 * (-3 * b + 4) = 5 * (b + 3) - 1 * 4 ∧ b = 11 / 26 :=
by
  sorry

end find_b_collinear_points_l1567_156727


namespace cider_production_l1567_156739

theorem cider_production (gd_pint : ℕ) (pl_pint : ℕ) (gs_pint : ℕ) (farmhands : ℕ) (gd_rate : ℕ) (pl_rate : ℕ) (gs_rate : ℕ) (work_hours : ℕ) 
  (gd_total : ℕ) (pl_total : ℕ) (gs_total : ℕ) (gd_ratio : ℕ) (pl_ratio : ℕ) (gs_ratio : ℕ) 
  (gd_pint_val : gd_pint = 20) (pl_pint_val : pl_pint = 40) (gs_pint_val : gs_pint = 30)
  (farmhands_val : farmhands = 6) (gd_rate_val : gd_rate = 120) (pl_rate_val : pl_rate = 240) (gs_rate_val : gs_rate = 180) 
  (work_hours_val : work_hours = 5) 
  (gd_total_val : gd_total = farmhands * work_hours * gd_rate) 
  (pl_total_val : pl_total = farmhands * work_hours * pl_rate) 
  (gs_total_val : gs_total = farmhands * work_hours * gs_rate) 
  (gd_ratio_val : gd_ratio = 1) (pl_ratio_val : pl_ratio = 2) (gs_ratio_val : gs_ratio = 3/2) 
  (ratio_condition : gd_total / gd_ratio = pl_total / pl_ratio ∧ pl_total / pl_ratio = gs_total / gs_ratio) : 
  (gd_total / gd_pint) = 180 := 
sorry

end cider_production_l1567_156739


namespace four_identical_pairwise_differences_l1567_156723

theorem four_identical_pairwise_differences (a : Fin 20 → ℕ) (h_distinct : Function.Injective a) (h_lt_70 : ∀ i, a i < 70) :
  ∃ d, ∃ (f g : Fin 20 × Fin 20), f ≠ g ∧ (a f.1 - a f.2 = d) ∧ (a g.1 - a g.2 = d) ∧
  ∃ (f1 f2 : Fin 20 × Fin 20), (f1 ≠ f ∧ f1 ≠ g) ∧ (f2 ≠ f ∧ f2 ≠ g) ∧ (a f1.1 - a f1.2 = d) ∧ (a f2.1 - a f2.2 = d) ∧
  (a f1.1 - a f1.2 = d) ∧ (a f2.1 - a f2.2 = d) ∧
  ∃ (f3 : Fin 20 × Fin 20), (f3 ≠ f ∧ f3 ≠ g ∧ f3 ≠ f1 ∧ f3 ≠ f2) ∧ (a f3.1 - a f3.2 = d) := 
sorry

end four_identical_pairwise_differences_l1567_156723


namespace range_of_a_l1567_156732

variable {x a : ℝ}

theorem range_of_a (h1 : x < 0) (h2 : 2 ^ x - a = 1 / (x - 1)) : 0 < a ∧ a < 2 :=
sorry

end range_of_a_l1567_156732


namespace prob_square_l1567_156795

def total_figures := 10
def num_squares := 3
def num_circles := 4
def num_triangles := 3

theorem prob_square : (num_squares : ℚ) / total_figures = 3 / 10 :=
by
  rw [total_figures, num_squares]
  exact sorry

end prob_square_l1567_156795


namespace marikas_father_age_twice_in_2036_l1567_156744

theorem marikas_father_age_twice_in_2036 :
  ∃ (x : ℕ), (10 + x = 2006 + x) ∧ (50 + x = 2 * (10 + x)) ∧ (2006 + x = 2036) :=
by
  sorry

end marikas_father_age_twice_in_2036_l1567_156744


namespace sum_of_angles_is_360_l1567_156701

-- Let's define the specific angles within our geometric figure
variables (A B C D F G : ℝ)

-- Define a condition stating that these angles form a quadrilateral inside a geometric figure, such that their sum is valid
def angles_form_quadrilateral (A B C D F G : ℝ) : Prop :=
  (A + B + C + D + F + G = 360)

-- Finally, we declare the theorem we want to prove
theorem sum_of_angles_is_360 (A B C D F G : ℝ) (h : angles_form_quadrilateral A B C D F G) : A + B + C + D + F + G = 360 :=
  h


end sum_of_angles_is_360_l1567_156701


namespace total_students_in_class_l1567_156776

def total_students (H : Nat) (hands_per_student : Nat) (consider_teacher : Nat) : Nat :=
  (H / hands_per_student) + consider_teacher

theorem total_students_in_class (H : Nat) (hands_per_student : Nat) (consider_teacher : Nat) 
  (H_eq : H = 20) (hands_per_student_eq : hands_per_student = 2) (consider_teacher_eq : consider_teacher = 1) : 
  total_students H hands_per_student consider_teacher = 11 := by
  sorry

end total_students_in_class_l1567_156776


namespace fruit_basket_l1567_156745

-- Define the quantities and their relationships
variables (O A B P : ℕ)

-- State the conditions
def condition1 : Prop := A = O - 2
def condition2 : Prop := B = 3 * A
def condition3 : Prop := P = B / 2
def condition4 : Prop := O + A + B + P = 28

-- State the theorem
theorem fruit_basket (h1 : condition1 O A) (h2 : condition2 A B) (h3 : condition3 B P) (h4 : condition4 O A B P) : O = 6 :=
sorry

end fruit_basket_l1567_156745


namespace tangent_slope_at_point_x_eq_1_l1567_156725

noncomputable def curve (x : ℝ) : ℝ := x^3 - 4 * x
noncomputable def curve_derivative (x : ℝ) : ℝ := 3 * x^2 - 4

theorem tangent_slope_at_point_x_eq_1 : curve_derivative 1 = -1 :=
by {
  -- This is just the theorem statement, no proof is required as per the instructions.
  sorry
}

end tangent_slope_at_point_x_eq_1_l1567_156725


namespace cos_angle_subtraction_l1567_156749

open Real

theorem cos_angle_subtraction (A B : ℝ) (h1 : sin A + sin B = 3 / 2) (h2 : cos A + cos B = 1) :
  cos (A - B) = 5 / 8 :=
sorry

end cos_angle_subtraction_l1567_156749


namespace gardenia_to_lilac_ratio_l1567_156764

-- Defining sales of flowers
def lilacs_sold : Nat := 10
def roses_sold : Nat := 3 * lilacs_sold
def total_flowers_sold : Nat := 45
def gardenias_sold : Nat := total_flowers_sold - (roses_sold + lilacs_sold)

-- The ratio of gardenias to lilacs as a fraction
def ratio_gardenias_to_lilacs (gardenias lilacs : Nat) : Rat := gardenias / lilacs

-- Stating the theorem to prove
theorem gardenia_to_lilac_ratio :
  ratio_gardenias_to_lilacs gardenias_sold lilacs_sold = 1 / 2 :=
by
  sorry

end gardenia_to_lilac_ratio_l1567_156764


namespace determine_dresses_and_shoes_colors_l1567_156711

variables (dress_color shoe_color : String → String)
variables (Tamara Valya Lida : String)

-- Conditions
def condition_1 : Prop := ∀ x : String, x ≠ Tamara → dress_color x ≠ shoe_color x
def condition_2 : Prop := shoe_color Valya = "white"
def condition_3 : Prop := dress_color Lida ≠ "red" ∧ shoe_color Lida ≠ "red"
def condition_4 : Prop := ∀ x : String, dress_color x ∈ ["white", "red", "blue"] ∧ shoe_color x ∈ ["white", "red", "blue"]

-- Desired conclusion
def conclusion : Prop :=
  dress_color Valya = "blue" ∧ shoe_color Valya = "white" ∧
  dress_color Lida = "white" ∧ shoe_color Lida = "blue" ∧
  dress_color Tamara = "red" ∧ shoe_color Tamara = "red"

theorem determine_dresses_and_shoes_colors
  (Tamara Valya Lida : String)
  (h1 : condition_1 dress_color shoe_color Tamara)
  (h2 : condition_2 shoe_color Valya)
  (h3 : condition_3 dress_color shoe_color Lida)
  (h4 : condition_4 dress_color shoe_color) :
  conclusion dress_color shoe_color Valya Lida Tamara :=
sorry

end determine_dresses_and_shoes_colors_l1567_156711


namespace arithmetic_sequence_fifth_term_l1567_156774

theorem arithmetic_sequence_fifth_term (x y : ℝ) (h1 : x = 2) (h2 : y = 1) :
    let a1 := x^2 + y^2
    let a2 := x^2 - y^2
    let a3 := x^2 * y^2
    let a4 := x^2 / y^2
    let d := a2 - a1
    let a5 := a4 + d
    a5 = 2 := by
  sorry

end arithmetic_sequence_fifth_term_l1567_156774


namespace brass_players_count_l1567_156771

def marching_band_size : ℕ := 110
def woodwinds (b : ℕ) : ℕ := 2 * b
def percussion (w : ℕ) : ℕ := 4 * w
def total_members (b : ℕ) : ℕ := b + woodwinds b + percussion (woodwinds b)

theorem brass_players_count : ∃ b : ℕ, total_members b = marching_band_size ∧ b = 10 :=
by
  sorry

end brass_players_count_l1567_156771


namespace molecular_weight_chlorous_acid_l1567_156735

def weight_H : ℝ := 1.01
def weight_Cl : ℝ := 35.45
def weight_O : ℝ := 16.00

def molecular_weight_HClO2 := (1 * weight_H) + (1 * weight_Cl) + (2 * weight_O)

theorem molecular_weight_chlorous_acid : molecular_weight_HClO2 = 68.46 := 
  by
    sorry

end molecular_weight_chlorous_acid_l1567_156735


namespace sum_volumes_spheres_l1567_156755

theorem sum_volumes_spheres (l : ℝ) (h_l : l = 2) : 
  ∑' (n : ℕ), (4 / 3) * π * ((1 / (3 ^ (n + 1))) ^ 3) = (2 * π / 39) :=
by
  sorry

end sum_volumes_spheres_l1567_156755


namespace solve_fractional_equation_l1567_156710

theorem solve_fractional_equation (x : ℝ) (h : x ≠ 1) : 
  (3 * x + 6) / (x^2 + 6 * x - 7) = (3 - x) / (x - 1) ↔ x = -5 ∨ x = 3 :=
sorry

end solve_fractional_equation_l1567_156710


namespace poly_a_c_sum_l1567_156724

theorem poly_a_c_sum {a b c d : ℝ} (f g : ℝ → ℝ)
  (hf : ∀ x, f x = x^2 + a * x + b)
  (hg : ∀ x, g x = x^2 + c * x + d)
  (hv_f_root_g : g (-a / 2) = 0)
  (hv_g_root_f : f (-c / 2) = 0)
  (f_min : ∀ x, f x ≥ -25)
  (g_min : ∀ x, g x ≥ -25)
  (f_g_intersect : f 50 = -25 ∧ g 50 = -25) : a + c = -101 :=
by
  sorry

end poly_a_c_sum_l1567_156724


namespace inequality_a_over_b_gt_a_plus_c_over_b_plus_d_gt_c_over_d_l1567_156705

theorem inequality_a_over_b_gt_a_plus_c_over_b_plus_d_gt_c_over_d
  (a b c d : ℚ) 
  (h1 : a * d > b * c) 
  (h2 : (a : ℚ) / b > (c : ℚ) / d) : 
  (a / b > (a + c) / (b + d)) ∧ ((a + c) / (b + d) > c / d) :=
by 
  sorry

end inequality_a_over_b_gt_a_plus_c_over_b_plus_d_gt_c_over_d_l1567_156705


namespace carols_weight_l1567_156742

variables (a c : ℝ)

theorem carols_weight (h1 : a + c = 220) (h2 : c - a = c / 3 + 10) : c = 138 :=
by
  sorry

end carols_weight_l1567_156742


namespace x729_minus_inverse_l1567_156703

theorem x729_minus_inverse (x : ℂ) (h : x - x⁻¹ = 2 * Complex.I) : x ^ 729 - x⁻¹ ^ 729 = 2 * Complex.I := 
by 
  sorry

end x729_minus_inverse_l1567_156703


namespace max_product_of_two_integers_with_sum_180_l1567_156794

theorem max_product_of_two_integers_with_sum_180 :
  ∃ x y : ℤ, (x + y = 180) ∧ (x * y = 8100) := by
  sorry

end max_product_of_two_integers_with_sum_180_l1567_156794


namespace find_m_l1567_156785

theorem find_m (m : ℝ) : (243 : ℝ)^(1/3) = (3 : ℝ)^m → m = 5 / 3 :=
by
  sorry

end find_m_l1567_156785


namespace quadratic_inequality_l1567_156714

theorem quadratic_inequality (a x1 x2 : ℝ) (h_eq : x1 ^ 2 - a * x1 + a = 0) (h_eq' : x2 ^ 2 - a * x2 + a = 0) :
  x1^2 + x2^2 ≥ 2 * (x1 + x2) :=
sorry

end quadratic_inequality_l1567_156714


namespace fraction_to_decimal_l1567_156772

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 :=
by
  sorry

end fraction_to_decimal_l1567_156772


namespace unit_digit_of_square_l1567_156759

theorem unit_digit_of_square (n : ℤ) (h : (n^2 / 10) % 10 = 7) : (n^2 % 10) = 6 := sorry

end unit_digit_of_square_l1567_156759


namespace distance_between_points_l1567_156729

theorem distance_between_points : 
  let p1 := (0, 24)
  let p2 := (10, 0)
  dist p1 p2 = 26 := 
by
  sorry

end distance_between_points_l1567_156729


namespace mixture_price_correct_l1567_156737

noncomputable def priceOfMixture (x y : ℝ) (P : ℝ) : Prop :=
  P = (3.10 * x + 3.60 * y) / (x + y)

theorem mixture_price_correct {x y : ℝ} (h_proportion : x / y = 7 / 3) : priceOfMixture x (3 / 7 * x) 3.25 :=
by
  sorry

end mixture_price_correct_l1567_156737


namespace average_weight_of_class_l1567_156787

theorem average_weight_of_class (n_boys n_girls : ℕ) (avg_weight_boys avg_weight_girls : ℝ)
    (h_boys : n_boys = 5) (h_girls : n_girls = 3)
    (h_avg_weight_boys : avg_weight_boys = 60) (h_avg_weight_girls : avg_weight_girls = 50) :
    (n_boys * avg_weight_boys + n_girls * avg_weight_girls) / (n_boys + n_girls) = 56.25 := 
by
  sorry

end average_weight_of_class_l1567_156787


namespace science_club_members_neither_l1567_156708

theorem science_club_members_neither {S B C : ℕ} (total : S = 60) (bio : B = 40) (chem : C = 35) (both : ℕ := 25) :
    S - ((B - both) + (C - both) + both) = 10 :=
by
  sorry

end science_club_members_neither_l1567_156708


namespace trig_expression_zero_l1567_156765

theorem trig_expression_zero (α : ℝ) (h : Real.tan α = 2) : 
  2 * (Real.sin α)^2 - 3 * (Real.sin α) * (Real.cos α) - 2 * (Real.cos α)^2 = 0 := 
by
  sorry

end trig_expression_zero_l1567_156765


namespace root_expression_of_cubic_l1567_156780

theorem root_expression_of_cubic :
  ∀ a b c : ℝ, (a^3 - 2*a - 2 = 0) ∧ (b^3 - 2*b - 2 = 0) ∧ (c^3 - 2*c - 2 = 0)
    → a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = -6 := 
by 
  sorry

end root_expression_of_cubic_l1567_156780


namespace binom_multiplication_l1567_156786

open BigOperators

noncomputable def choose_and_multiply (n k m l : ℕ) : ℕ :=
  Nat.choose n k * Nat.choose m l

theorem binom_multiplication : choose_and_multiply 10 3 8 3 = 6720 := by
  sorry

end binom_multiplication_l1567_156786


namespace married_men_fraction_l1567_156783

-- define the total number of women
def W : ℕ := 7

-- define the number of single women
def single_women (W : ℕ) : ℕ := 3

-- define the probability of picking a single woman
def P_s : ℚ := single_women W / W

-- define number of married women
def married_women (W : ℕ) : ℕ := W - single_women W

-- define number of married men
def married_men (W : ℕ) : ℕ := married_women W

-- define total number of people
def total_people (W : ℕ) : ℕ := W + married_men W

-- define fraction of married men
def married_men_ratio (W : ℕ) : ℚ := married_men W / total_people W

-- theorem to prove that the ratio is 4/11
theorem married_men_fraction : married_men_ratio W = 4 / 11 := 
by 
  sorry

end married_men_fraction_l1567_156783


namespace real_imaginary_part_above_x_axis_polynomial_solutions_l1567_156766

-- Question 1: For what values of the real number m is (m^2 - 2m - 15) > 0
theorem real_imaginary_part_above_x_axis (m : ℝ) : 
  (m^2 - 2 * m - 15 > 0) ↔ (m < -3 ∨ m > 5) :=
sorry

-- Question 2: For what values of the real number m does 2m^2 + 3m - 4=0?
theorem polynomial_solutions (m : ℝ) : 
  (2 * m^2 + 3 * m - 4 = 0) ↔ (m = -3 ∨ m = 2) :=
sorry

end real_imaginary_part_above_x_axis_polynomial_solutions_l1567_156766


namespace tan_alpha_eq_neg_one_l1567_156773

theorem tan_alpha_eq_neg_one (α : ℝ) (h1 : |Real.sin α| = |Real.cos α|)
    (h2 : π / 2 < α ∧ α < π) : Real.tan α = -1 :=
sorry

end tan_alpha_eq_neg_one_l1567_156773


namespace necessary_and_sufficient_condition_l1567_156784

theorem necessary_and_sufficient_condition (a b : ℝ) (ha : a < 0) (hb : b < 0) :
  (a > b) ↔ (a - 1/a > b - 1/b) :=
sorry

end necessary_and_sufficient_condition_l1567_156784


namespace pages_in_second_chapter_l1567_156754

theorem pages_in_second_chapter
  (total_pages : ℕ)
  (first_chapter_pages : ℕ)
  (second_chapter_pages : ℕ)
  (h1 : total_pages = 93)
  (h2 : first_chapter_pages = 60)
  (h3: second_chapter_pages = total_pages - first_chapter_pages) :
  second_chapter_pages = 33 :=
by
  sorry

end pages_in_second_chapter_l1567_156754


namespace solution_in_range_for_fraction_l1567_156720

theorem solution_in_range_for_fraction (a : ℝ) : 
  (∃ x : ℝ, (2 * x + a) / (x + 1) = 1 ∧ x < 0) ↔ (a > 1 ∧ a ≠ 2) :=
by
  sorry

end solution_in_range_for_fraction_l1567_156720


namespace sum_of_eight_numbers_l1567_156756

theorem sum_of_eight_numbers (a b c d e f g h : ℕ) 
  (h1 : (a + b) * (c + d) * (e + f) * (g + h) = 330) : 
  a + b + c + d + e + f + g + h = 21 :=
sorry

end sum_of_eight_numbers_l1567_156756


namespace sparrow_population_decline_l1567_156778

theorem sparrow_population_decline {P : ℕ} (initial_year : ℕ) (initial_population : ℕ) (decrease_by_half : ∀ year, year ≥ initial_year →  init_population * (1 / (2 ^ (year - initial_year))) < init_population / 20) :
  ∃ year, year ≥ initial_year + 5 ∧ init_population * (1 / (2 ^ (year - initial_year))) < init_population / 20 :=
by
  sorry

end sparrow_population_decline_l1567_156778


namespace monotonic_decreasing_interval_l1567_156760

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 2 * Real.log x

theorem monotonic_decreasing_interval :
  ∀ x ∈ Set.Ioo 0 (Real.sqrt 3 / 3), (deriv f x) < 0 :=
by
  sorry

end monotonic_decreasing_interval_l1567_156760


namespace rectangle_area_increase_l1567_156762

theorem rectangle_area_increase (L W : ℝ) (h1: L > 0) (h2: W > 0) :
   let original_area := L * W
   let new_length := 1.20 * L
   let new_width := 1.20 * W
   let new_area := new_length * new_width
   let percentage_increase := ((new_area - original_area) / original_area) * 100
   percentage_increase = 44 :=
by
  sorry

end rectangle_area_increase_l1567_156762


namespace binary_multiplication_addition_l1567_156777

-- Define the binary representation of the given numbers
def b1101 : ℕ := 0b1101
def b111 : ℕ := 0b111
def b1011 : ℕ := 0b1011
def b1011010 : ℕ := 0b1011010

-- State the theorem
theorem binary_multiplication_addition :
  (b1101 * b111 + b1011) = b1011010 := 
sorry

end binary_multiplication_addition_l1567_156777


namespace inequality_proof_l1567_156751

theorem inequality_proof (a b : ℝ) (c : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : c < 0) :
  (c / a > c / b) ∧ (a^c < b^c) ∧ (Real.log (a - c) / Real.log b > Real.log (b - c) / Real.log a) := 
sorry

end inequality_proof_l1567_156751


namespace finite_solutions_to_equation_l1567_156793

theorem finite_solutions_to_equation :
  ∃ n : ℕ, ∀ (a b c : ℕ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ b ∧ b ≤ c ∧ (1 / (a:ℝ) + 1 / (b:ℝ) + 1 / (c:ℝ) = 1 / 1983) →
    a ≤ n ∧ b ≤ n ∧ c ≤ n :=
sorry

end finite_solutions_to_equation_l1567_156793


namespace sum_largest_three_digit_multiple_of_4_smallest_four_digit_multiple_of_3_l1567_156702

theorem sum_largest_three_digit_multiple_of_4_smallest_four_digit_multiple_of_3 :
  let largestThreeDigitMultipleOf4 := 996
  let smallestFourDigitMultipleOf3 := 1002
  largestThreeDigitMultipleOf4 + smallestFourDigitMultipleOf3 = 1998 :=
by
  sorry

end sum_largest_three_digit_multiple_of_4_smallest_four_digit_multiple_of_3_l1567_156702


namespace single_discount_equivalence_l1567_156722

noncomputable def original_price : ℝ := 50
noncomputable def discount1 : ℝ := 0.15
noncomputable def discount2 : ℝ := 0.10
noncomputable def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)
noncomputable def effective_discount_price := 
  apply_discount (apply_discount original_price discount1) discount2
noncomputable def effective_discount :=
  (original_price - effective_discount_price) / original_price

theorem single_discount_equivalence :
  effective_discount = 0.235 := by
  sorry

end single_discount_equivalence_l1567_156722


namespace jan_paid_amount_l1567_156748

def number_of_roses (dozens : Nat) : Nat := dozens * 12

def total_cost (number_of_roses : Nat) (cost_per_rose : Nat) : Nat := number_of_roses * cost_per_rose

def discounted_price (total_cost : Nat) (discount_percentage : Nat) : Nat := total_cost * discount_percentage / 100

theorem jan_paid_amount :
  let dozens := 5
  let cost_per_rose := 6
  let discount_percentage := 80
  number_of_roses dozens = 60 →
  total_cost (number_of_roses dozens) cost_per_rose = 360 →
  discounted_price (total_cost (number_of_roses dozens) cost_per_rose) discount_percentage = 288 :=
by
  intros
  sorry

end jan_paid_amount_l1567_156748


namespace circle_equation_correct_l1567_156763

theorem circle_equation_correct (x y : ℝ) :
  let h : ℝ := -2
  let k : ℝ := 2
  let r : ℝ := 5
  ((x - h)^2 + (y - k)^2 = r^2) ↔ ((x + 2)^2 + (y - 2)^2 = 25) :=
by
  sorry

end circle_equation_correct_l1567_156763


namespace pam_number_of_bags_l1567_156738

-- Definitions of the conditions
def apples_in_geralds_bag : Nat := 40
def pam_bags_ratio : Nat := 3
def total_pam_apples : Nat := 1200

-- Problem statement (Theorem)
theorem pam_number_of_bags :
  Pam_bags == total_pam_apples / (pam_bags_ratio * apples_in_geralds_bag) :=
by 
  sorry

end pam_number_of_bags_l1567_156738


namespace distance_of_route_l1567_156747

-- Define the conditions
def round_trip_time : ℝ := 1 -- in hours
def avg_speed : ℝ := 3 -- in miles per hour
def return_speed : ℝ := 6.000000000000002 -- in miles per hour

-- Problem statement to prove
theorem distance_of_route : 
  ∃ (D : ℝ), 
  2 * D = avg_speed * round_trip_time ∧ 
  D = 1.5 := 
by
  sorry

end distance_of_route_l1567_156747


namespace min_value_expression_l1567_156791

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b + a * c = 4) :
  ∃ m, m = 4 ∧ m ≤ 2 / a + 2 / (b + c) + 8 / (a + b + c) :=
by
  sorry

end min_value_expression_l1567_156791


namespace right_angle_sides_of_isosceles_right_triangle_l1567_156719

def is_on_line (x y a b c : ℝ) : Prop := a * x + b * y + c = 0

def is_perpendicular (a1 b1 a2 b2 : ℝ) : Prop := a1 * a2 + b1 * b2 = 0

theorem right_angle_sides_of_isosceles_right_triangle
  (C : ℝ × ℝ)
  (hyp_line : ℝ → ℝ → Prop)
  (side_AC side_BC : ℝ → ℝ → Prop)
  (H1 : C = (3, -2))
  (H2 : hyp_line = is_on_line 3 (-1) 2)
  (H3 : side_AC = is_on_line 2 1 (-4))
  (H4 : side_BC = is_on_line 1 (-2) (-7))
  (H5 : ∃ x y, side_BC (3) y ∧ side_AC x (-2)) :
  side_AC = is_on_line 2 1 (-4) ∧ side_BC = is_on_line 1 (-2) (-7) :=
by
  sorry

end right_angle_sides_of_isosceles_right_triangle_l1567_156719


namespace problem_statement_l1567_156792

theorem problem_statement (a b : ℕ) (ha : a = 55555) (hb : b = 66666) :
  55554 * 55559 * 55552 - 55556 * 55551 * 55558 =
  66665 * 66670 * 66663 - 66667 * 66662 * 66669 := 
by
  sorry

end problem_statement_l1567_156792


namespace range_of_m_l1567_156782

theorem range_of_m (x y m : ℝ) 
  (h1 : 3 * x + y = 3 * m + 1)
  (h2 : x + 2 * y = 3)
  (h3 : 2 * x - y < 1) : 
  m < 1 := 
sorry

end range_of_m_l1567_156782


namespace sum_a_b_l1567_156713

theorem sum_a_b (a b : ℝ) (h₁ : 2 = a + b) (h₂ : 6 = a + b / 9) : a + b = 2 :=
by
  sorry

end sum_a_b_l1567_156713


namespace square_side_length_square_area_l1567_156779

theorem square_side_length 
  (d : ℝ := 4) : (s : ℝ) = 2 * Real.sqrt 2 :=
  sorry

theorem square_area 
  (s : ℝ := 2 * Real.sqrt 2) : (A : ℝ) = 8 :=
  sorry

end square_side_length_square_area_l1567_156779


namespace scientific_calculators_ordered_l1567_156788

variables (x y : ℕ)

theorem scientific_calculators_ordered :
  (10 * x + 57 * y = 1625) ∧ (x + y = 45) → x = 20 :=
by
  -- proof goes here
  sorry

end scientific_calculators_ordered_l1567_156788


namespace find_tan_theta_l1567_156717

open Real

theorem find_tan_theta (θ : ℝ) (h1 : sin θ + cos θ = 7 / 13) (h2 : 0 < θ ∧ θ < π) :
  tan θ = -12 / 5 :=
sorry

end find_tan_theta_l1567_156717


namespace sum_of_perimeters_l1567_156718

theorem sum_of_perimeters (A1 A2 : ℝ) (h1 : A1 + A2 = 145) (h2 : A1 - A2 = 25) :
  4 * Real.sqrt 85 + 4 * Real.sqrt 60 = 4 * Real.sqrt A1 + 4 * Real.sqrt A2 :=
by
  sorry

end sum_of_perimeters_l1567_156718


namespace remainder_sum_mult_3_zero_mod_18_l1567_156721

theorem remainder_sum_mult_3_zero_mod_18
  (p q r s : ℕ)
  (hp : p % 18 = 8)
  (hq : q % 18 = 11)
  (hr : r % 18 = 14)
  (hs : s % 18 = 15) :
  3 * (p + q + r + s) % 18 = 0 :=
by
  sorry

end remainder_sum_mult_3_zero_mod_18_l1567_156721


namespace first_digit_base9_650_l1567_156736

theorem first_digit_base9_650 : ∃ d : ℕ, 
  d = 8 ∧ (∃ k : ℕ, 650 = d * 9^2 + k ∧ k < 9^2) :=
by {
  sorry
}

end first_digit_base9_650_l1567_156736


namespace number_of_true_propositions_l1567_156752

variable (x : ℝ)

def original_proposition (x : ℝ) : Prop := (x = 5) → (x^2 - 8 * x + 15 = 0)
def converse_proposition (x : ℝ) : Prop := (x^2 - 8 * x + 15 = 0) → (x = 5)
def inverse_proposition (x : ℝ) : Prop := (x ≠ 5) → (x^2 - 8 * x + 15 ≠ 0)
def contrapositive_proposition (x : ℝ) : Prop := (x^2 - 8 * x + 15 ≠ 0) → (x ≠ 5)

theorem number_of_true_propositions : 
  (original_proposition x ∧ contrapositive_proposition x) ∧
  ¬(converse_proposition x) ∧ ¬(inverse_proposition x) ↔ true := sorry

end number_of_true_propositions_l1567_156752


namespace union_of_A_B_l1567_156734

open Set

variable {α : Type*} [LinearOrder α]

def A : Set ℝ := { x | -1 < x ∧ x < 2 }
def B : Set ℝ := { x | 1 < x ∧ x < 3 }

theorem union_of_A_B : A ∪ B = { x | -1 < x ∧ x < 3 } :=
sorry

end union_of_A_B_l1567_156734


namespace avg_height_of_class_is_168_6_l1567_156775

noncomputable def avgHeightClass : ℕ → ℕ → ℕ → ℕ → ℚ :=
  λ n₁ h₁ n₂ h₂ => (n₁ * h₁ + n₂ * h₂) / (n₁ + n₂)

theorem avg_height_of_class_is_168_6 :
  avgHeightClass 40 169 10 167 = 168.6 := 
by 
  sorry

end avg_height_of_class_is_168_6_l1567_156775


namespace union_of_M_and_N_l1567_156768

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1, 2} :=
by
  sorry

end union_of_M_and_N_l1567_156768


namespace oliver_cards_l1567_156728

variable {MC AB BG : ℕ}

theorem oliver_cards : 
  (BG = 48) → 
  (BG = 3 * AB) → 
  (MC = 2 * AB) → 
  MC = 32 := 
by 
  intros h1 h2 h3
  sorry

end oliver_cards_l1567_156728


namespace main_theorem_l1567_156761

variable (x y z : ℝ) (h₀ : 0 < x ∧ 0 < y ∧ 0 < z) (h₁ : x^3 + y^3 + z^3 = 1)

theorem main_theorem (h₀ : 0 < x ∧ 0 < y ∧ 0 < z) (h₁ : x^3 + y^3 + z^3 = 1):
  (x^2 / (1 - x^2)) + (y^2 / (1 - y^2)) + (z^2 / (1 - z^2)) ≥ (3 * Real.sqrt 3) / 2 := 
by
  sorry

end main_theorem_l1567_156761


namespace equal_red_B_black_C_l1567_156743

theorem equal_red_B_black_C (a : ℕ) (h_even : a % 2 = 0) :
  ∃ (x y k j l i : ℕ), x + y = a ∧ y + i + j = a ∧ i + k = y ∧ k + j = x ∧ i = k := 
  sorry

end equal_red_B_black_C_l1567_156743


namespace unique_n_value_l1567_156797

theorem unique_n_value :
  ∃ (n : ℕ), n > 0 ∧ (∃ k : ℕ, k > 0 ∧ k < 10 ∧ 111 * k = (n * (n + 1) / 2)) ∧ ∀ (m : ℕ), m > 0 → (∃ j : ℕ, j > 0 ∧ j < 10 ∧ 111 * j = (m * (m + 1) / 2)) → m = 36 :=
by
  sorry

end unique_n_value_l1567_156797


namespace quadrilateral_ratio_l1567_156731

theorem quadrilateral_ratio (AB CD AD BC IA IB IC ID : ℝ)
  (h_tangential : AB + CD = AD + BC)
  (h_IA : IA = 5)
  (h_IB : IB = 7)
  (h_IC : IC = 4)
  (h_ID : ID = 9) :
  AB / CD = 35 / 36 :=
by
  -- Proof will be provided here
  sorry

end quadrilateral_ratio_l1567_156731


namespace ratio_of_roots_l1567_156741

theorem ratio_of_roots 
  (a b c : ℝ) 
  (h : a * b * c ≠ 0)
  (x1 x2 : ℝ) 
  (root1 : x1 = 2022 * x2) 
  (root2 : a * x1 ^ 2 + b * x1 + c = 0) 
  (root3 : a * x2 ^ 2 + b * x2 + c = 0) : 
  2023 * a * c / b ^ 2 = 2022 / 2023 :=
by
  sorry

end ratio_of_roots_l1567_156741


namespace remainder_5_7_9_6_3_5_mod_7_l1567_156750

theorem remainder_5_7_9_6_3_5_mod_7 : (5^7 + 9^6 + 3^5) % 7 = 5 :=
by sorry

end remainder_5_7_9_6_3_5_mod_7_l1567_156750


namespace total_population_expr_l1567_156758

-- Definitions of the quantities
variables (b g t : ℕ)

-- Conditions
axiom boys_as_girls : b = 3 * g
axiom girls_as_teachers : g = 9 * t

-- Theorem to prove
theorem total_population_expr : b + g + t = 37 * b / 27 :=
by
  sorry

end total_population_expr_l1567_156758


namespace wheel_distance_covered_l1567_156707

noncomputable def diameter : ℝ := 15
noncomputable def revolutions : ℝ := 11.210191082802547
noncomputable def pi : ℝ := Real.pi -- or you can use the approximate value if required: 3.14159
noncomputable def circumference : ℝ := pi * diameter
noncomputable def distance_covered : ℝ := circumference * revolutions

theorem wheel_distance_covered :
  distance_covered = 528.316820577 := 
by
  unfold distance_covered
  unfold circumference
  unfold diameter
  unfold revolutions
  norm_num
  sorry

end wheel_distance_covered_l1567_156707


namespace smallest_positive_integer_N_l1567_156709

theorem smallest_positive_integer_N :
  ∃ N : ℕ, N > 0 ∧ (N % 7 = 5) ∧ (N % 8 = 6) ∧ (N % 9 = 7) ∧ (∀ M : ℕ, M > 0 ∧ (M % 7 = 5) ∧ (M % 8 = 6) ∧ (M % 9 = 7) → N ≤ M) :=
sorry

end smallest_positive_integer_N_l1567_156709


namespace pole_length_after_cut_l1567_156798

theorem pole_length_after_cut (original_length : ℝ) (percentage_retained : ℝ) : 
  original_length = 20 → percentage_retained = 0.7 → 
  original_length * percentage_retained = 14 :=
by
  intros h0 h1
  rw [h0, h1]
  norm_num

end pole_length_after_cut_l1567_156798


namespace mersenne_prime_condition_l1567_156770

theorem mersenne_prime_condition (a n : ℕ) (h_a : 1 < a) (h_n : 1 < n) (h_prime : Prime (a ^ n - 1)) : a = 2 ∧ Prime n :=
by
  sorry

end mersenne_prime_condition_l1567_156770


namespace hindi_speaking_children_l1567_156726

-- Condition Definitions
def total_children : ℕ := 90
def percent_only_english : ℝ := 0.25
def percent_only_hindi : ℝ := 0.15
def percent_only_spanish : ℝ := 0.10
def percent_english_hindi : ℝ := 0.20
def percent_english_spanish : ℝ := 0.15
def percent_hindi_spanish : ℝ := 0.10
def percent_all_three : ℝ := 0.05

-- Question translated to a Lean statement
theorem hindi_speaking_children :
  (percent_only_hindi + percent_english_hindi + percent_hindi_spanish + percent_all_three) * total_children = 45 :=
by
  sorry

end hindi_speaking_children_l1567_156726
