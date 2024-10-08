import Mathlib

namespace frank_completes_book_in_three_days_l39_39980

-- Define the total number of pages in a book
def total_pages : ℕ := 249

-- Define the number of pages Frank reads per day
def pages_per_day : ℕ := 83

-- Define the number of days Frank needs to finish a book
def days_to_finish_book (total_pages pages_per_day : ℕ) : ℕ :=
  total_pages / pages_per_day

-- Theorem statement to prove that Frank finishes a book in 3 days
theorem frank_completes_book_in_three_days : days_to_finish_book total_pages pages_per_day = 3 := 
by {
  -- Proof goes here
  sorry
}

end frank_completes_book_in_three_days_l39_39980


namespace min_photos_needed_to_ensure_conditions_l39_39856

noncomputable def min_photos (girls boys : ℕ) : ℕ :=
  33

theorem min_photos_needed_to_ensure_conditions (girls boys : ℕ)
  (hgirls : girls = 4) (hboys : boys = 8) :
  min_photos girls boys = 33 := 
sorry

end min_photos_needed_to_ensure_conditions_l39_39856


namespace bamboo_middle_node_capacity_l39_39526

def capacities_form_arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n + d

theorem bamboo_middle_node_capacity :
  ∃ (a : ℕ → ℚ) (d : ℚ), 
    capacities_form_arithmetic_sequence a d ∧ 
    (a 1 + a 2 + a 3 = 4) ∧
    (a 6 + a 7 + a 8 + a 9 = 3) ∧
    (a 5 = 67 / 66) :=
  sorry

end bamboo_middle_node_capacity_l39_39526


namespace mass_percentage_iodine_neq_662_l39_39581

theorem mass_percentage_iodine_neq_662 (atomic_mass_Al : ℝ) (atomic_mass_I : ℝ) (molar_mass_AlI3 : ℝ) :
  atomic_mass_Al = 26.98 ∧ atomic_mass_I = 126.90 ∧ molar_mass_AlI3 = ((1 * atomic_mass_Al) + (3 * atomic_mass_I)) →
  (3 * atomic_mass_I / molar_mass_AlI3 * 100) ≠ 6.62 :=
by
  sorry

end mass_percentage_iodine_neq_662_l39_39581


namespace total_votes_l39_39277

theorem total_votes (V : ℕ) (h1 : ∃ c : ℕ, c = 84) (h2 : ∃ m : ℕ, m = 476) (h3 : ∃ d : ℕ, d = ((84 * V - 16 * V) / 100)) : 
  V = 700 := 
by 
  sorry 

end total_votes_l39_39277


namespace ratio_of_horses_to_cows_l39_39848

/-- Let H and C be the initial number of horses and cows respectively.
Given that:
1. (H - 15) / (C + 15) = 7 / 3,
2. H - 15 = C + 75,
prove that the initial ratio of horses to cows is 4:1. -/
theorem ratio_of_horses_to_cows (H C : ℕ) 
  (h1 : (H - 15 : ℚ) / (C + 15 : ℚ) = 7 / 3)
  (h2 : H - 15 = C + 75) :
  H / C = 4 :=
by
  sorry

end ratio_of_horses_to_cows_l39_39848


namespace symmetric_axis_of_quadratic_fn_l39_39330

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 + 8 * x + 9 

-- State the theorem that the axis of symmetry for the quadratic function y = x^2 + 8x + 9 is x = -4
theorem symmetric_axis_of_quadratic_fn : ∃ h : ℝ, h = -4 ∧ ∀ x, quadratic_function x = quadratic_function (2 * h - x) :=
by sorry

end symmetric_axis_of_quadratic_fn_l39_39330


namespace match_foci_of_parabola_and_hyperbola_l39_39933

noncomputable def focus_of_parabola (a : ℝ) : ℝ :=
a / 4

noncomputable def foci_of_hyperbola : Set ℝ :=
{2, -2}

theorem match_foci_of_parabola_and_hyperbola (a : ℝ) :
  focus_of_parabola a ∈ foci_of_hyperbola ↔ a = 8 ∨ a = -8 :=
by
  -- This is the placeholder for the proof.
  sorry

end match_foci_of_parabola_and_hyperbola_l39_39933


namespace probability_defective_is_three_tenths_l39_39704

open Classical

noncomputable def probability_of_defective_product (total_products defective_products: ℕ) : ℝ :=
  (defective_products * 1.0) / (total_products * 1.0)

theorem probability_defective_is_three_tenths :
  probability_of_defective_product 10 3 = 3 / 10 := by
  sorry

end probability_defective_is_three_tenths_l39_39704


namespace tens_digit_2015_pow_2016_minus_2017_l39_39081

theorem tens_digit_2015_pow_2016_minus_2017 :
  (2015^2016 - 2017) % 100 = 8 := 
sorry

end tens_digit_2015_pow_2016_minus_2017_l39_39081


namespace xyz_value_l39_39334

variable (x y z : ℝ)

theorem xyz_value :
  (x + y + z) * (x*y + x*z + y*z) = 36 →
  x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 24 →
  x * y * z = 4 :=
by
  intros h1 h2
  sorry

end xyz_value_l39_39334


namespace tangent_parallel_l39_39031

noncomputable def f (x : ℝ) := x^4 - x

theorem tangent_parallel (P : ℝ × ℝ) (hP : P = (1, 0)) :
  (∃ x y : ℝ, P = (x, y) ∧ (fderiv ℝ f x) 1 = 3 / 1) ↔ P = (1, 0) :=
by
  sorry

end tangent_parallel_l39_39031


namespace smallest_positive_integer_n_l39_39937

theorem smallest_positive_integer_n (n : ℕ) (cube : Finset (Fin 8)) :
    (∀ (coloring : Finset (Fin 8)), 
      coloring.card = n → 
      ∃ (v : Fin 8), 
        (∀ (adj : Finset (Fin 8)), adj.card = 3 → adj ⊆ cube → v ∈ adj → adj ⊆ coloring)) 
    ↔ n = 5 := 
by
  sorry

end smallest_positive_integer_n_l39_39937


namespace vector_c_expression_l39_39128

-- Define the vectors a, b, c
def vector_a : ℤ × ℤ := (1, 2)
def vector_b : ℤ × ℤ := (-1, 1)
def vector_c : ℤ × ℤ := (1, 5)

-- Define the addition of vectors in ℤ × ℤ
def vec_add (v1 v2 : ℤ × ℤ) : ℤ × ℤ := (v1.1 + v2.1, v1.2 + v2.2)

-- Define the scalar multiplication of vectors in ℤ × ℤ
def scalar_mul (k : ℤ) (v : ℤ × ℤ) : ℤ × ℤ := (k * v.1, k * v.2)

-- Given the conditions
def condition1 := vector_a = (1, 2)
def condition2 := vec_add vector_a vector_b = (0, 3)

-- The goal is to prove that vector_c = 2 * vector_a + vector_b
theorem vector_c_expression : vec_add (scalar_mul 2 vector_a) vector_b = vector_c := by
  sorry

end vector_c_expression_l39_39128


namespace solve_for_x_l39_39814

theorem solve_for_x:
  ∀ (x : ℝ), (x + 10) / (x - 4) = (x - 3) / (x + 6) → x = -(48 / 23) :=
by
  sorry

end solve_for_x_l39_39814


namespace count_even_numbers_l39_39280

theorem count_even_numbers (a b : ℕ) (h1 : a > 300) (h2 : b ≤ 600) (h3 : ∀ n, 300 < n ∧ n ≤ 600 → n % 2 = 0) : 
  ∃ c : ℕ, c = 150 :=
by
  sorry

end count_even_numbers_l39_39280


namespace purity_of_alloy_l39_39884

theorem purity_of_alloy (w1 w2 : ℝ) (p1 p2 : ℝ) (h_w1 : w1 = 180) (h_p1 : p1 = 920) (h_w2 : w2 = 100) (h_p2 : p2 = 752) : 
  let a := w1 * (p1 / 1000) + w2 * (p2 / 1000)
  let b := w1 + w2
  let p_result := (a / b) * 1000
  p_result = 860 :=
by
  sorry

end purity_of_alloy_l39_39884


namespace hugo_probability_l39_39960

noncomputable def P_hugo_first_roll_seven_given_win (P_Hugo_wins : ℚ) (P_first_roll_seven : ℚ)
  (P_all_others_roll_less_than_seven : ℚ) : ℚ :=
(P_first_roll_seven * P_all_others_roll_less_than_seven) / P_Hugo_wins

theorem hugo_probability :
  let P_Hugo_wins := (1 : ℚ) / 4
  let P_first_roll_seven := (1 : ℚ) / 8
  let P_all_others_roll_less_than_seven := (27 : ℚ) / 64
  P_hugo_first_roll_seven_given_win P_Hugo_wins P_first_roll_seven P_all_others_roll_less_than_seven = (27 : ℚ) / 128 :=
by
  sorry

end hugo_probability_l39_39960


namespace intersecting_lines_l39_39545

theorem intersecting_lines (x y : ℝ) : x ^ 2 - y ^ 2 = 0 ↔ (y = x ∨ y = -x) := by
  sorry

end intersecting_lines_l39_39545


namespace remainder_when_divided_by_15_l39_39005

theorem remainder_when_divided_by_15 (N : ℤ) (k : ℤ) 
  (h : N = 45 * k + 31) : (N % 15) = 1 := by
  sorry

end remainder_when_divided_by_15_l39_39005


namespace circle_radius_9_l39_39854

theorem circle_radius_9 (k : ℝ) : 
  (∀ x y : ℝ, 2 * x^2 + 20 * x + 3 * y^2 + 18 * y - k = 81) → 
  (k = 94) :=
by
  sorry

end circle_radius_9_l39_39854


namespace negation_proposition_false_l39_39949

variable (a : ℝ)

theorem negation_proposition_false : ¬ (∃ a : ℝ, a ≤ 2 ∧ a^2 ≥ 4) :=
sorry

end negation_proposition_false_l39_39949


namespace problem_1_problem_2_l39_39776

def f (x : ℝ) : ℝ := 2 * x + 1
def g (x : ℝ) : ℝ := 2 * x - 1

theorem problem_1 (x : ℝ) : (g x ≥ abs (x - 1)) ↔ (x ≥ 2/3) :=
by
  sorry

theorem problem_2 (c : ℝ) : (∀ x, abs (g x) - c ≥ abs (x - 1)) → (c ≤ -1/2) :=
by
  sorry

end problem_1_problem_2_l39_39776


namespace geometric_sequence_common_ratio_l39_39659

-- Define a sequence as a list of real numbers
def seq : List ℚ := [8, -20, 50, -125]

-- Define the common ratio of a geometric sequence
def common_ratio (l : List ℚ) : ℚ := l.head! / l.tail!.head!

-- The theorem to prove the common ratio is -5/2
theorem geometric_sequence_common_ratio :
  common_ratio seq = -5 / 2 :=
by
  sorry

end geometric_sequence_common_ratio_l39_39659


namespace total_stickers_l39_39225

theorem total_stickers :
  (20.0 : ℝ) + (26.0 : ℝ) + (20.0 : ℝ) + (6.0 : ℝ) + (58.0 : ℝ) = 130.0 := by
  sorry

end total_stickers_l39_39225


namespace geese_left_in_the_field_l39_39981

theorem geese_left_in_the_field 
  (initial_geese : ℕ) 
  (geese_flew_away : ℕ) 
  (geese_joined : ℕ)
  (h1 : initial_geese = 372)
  (h2 : geese_flew_away = 178)
  (h3 : geese_joined = 57) :
  initial_geese - geese_flew_away + geese_joined = 251 := by
  sorry

end geese_left_in_the_field_l39_39981


namespace derivative_at_x_equals_1_l39_39947

variable (x : ℝ)
def y : ℝ := (x + 1) * (x - 1)

theorem derivative_at_x_equals_1 : deriv y 1 = 2 :=
by
  sorry

end derivative_at_x_equals_1_l39_39947


namespace length_of_AD_l39_39009

theorem length_of_AD (AB BC AC AD DC : ℝ)
    (h1 : AB = BC)
    (h2 : AD = 2 * DC)
    (h3 : AC = AD + DC)
    (h4 : AC = 27) : AD = 18 := 
by
  sorry

end length_of_AD_l39_39009


namespace chocolate_bars_in_large_box_l39_39747

theorem chocolate_bars_in_large_box : 
  let small_boxes := 19 
  let bars_per_small_box := 25 
  let total_bars := small_boxes * bars_per_small_box 
  total_bars = 475 := by 
  -- declarations and assumptions
  let small_boxes : ℕ := 19 
  let bars_per_small_box : ℕ := 25 
  let total_bars : ℕ := small_boxes * bars_per_small_box 
  sorry

end chocolate_bars_in_large_box_l39_39747


namespace area_of_vegetable_patch_l39_39511

theorem area_of_vegetable_patch : ∃ (a b : ℕ), 
  (2 * (a + b) = 24 ∧ b = 3 * a + 2 ∧ (6 * (a + 1)) * (6 * (b + 1)) = 576) :=
sorry

end area_of_vegetable_patch_l39_39511


namespace sum_of_fractions_correct_l39_39860

def sum_of_fractions : ℚ := (4 / 3) + (8 / 9) + (18 / 27) + (40 / 81) + (88 / 243) - 5

theorem sum_of_fractions_correct : sum_of_fractions = -305 / 243 := by
  sorry -- proof to be provided

end sum_of_fractions_correct_l39_39860


namespace inequality_proof_l39_39182

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + 9 * y + 3 * z) * (x + 4 * y + 2 * z) * (2 * x + 12 * y + 9 * z) ≥ 1029 * x * y * z :=
by
  sorry

end inequality_proof_l39_39182


namespace train_length_l39_39153

theorem train_length (speed : ℝ) (time : ℝ) (h1 : speed = 300)
  (h2 : time = 33) : (speed * 1000 / 3600) * time = 2750 := by
  sorry

end train_length_l39_39153


namespace arc_length_l39_39111

-- Define the radius and central angle
def radius : ℝ := 10
def central_angle : ℝ := 240

-- Theorem to prove the arc length is (40 * π) / 3
theorem arc_length (r : ℝ) (n : ℝ) (h_r : r = radius) (h_n : n = central_angle) : 
  (n * π * r) / 180 = (40 * π) / 3 :=
by
  -- Proof omitted
  sorry

end arc_length_l39_39111


namespace point_to_real_l39_39921

-- Condition: Real numbers correspond one-to-one with points on the number line.
def real_numbers_correspond (x : ℝ) : Prop :=
  ∃ (p : ℝ), p = x

-- Condition: Any real number can be represented by a point on the number line.
def represent_real_by_point (x : ℝ) : Prop :=
  real_numbers_correspond x

-- Condition: Conversely, any point on the number line represents a real number.
def point_represents_real (p : ℝ) : Prop :=
  ∃ (x : ℝ), x = p

-- Condition: The number represented by any point on the number line is either a rational number or an irrational number.
def rational_or_irrational (p : ℝ) : Prop :=
  (∃ q : ℚ, (q : ℝ) = p) ∨ (¬∃ q : ℚ, (q : ℝ) = p)

theorem point_to_real (p : ℝ) : represent_real_by_point p ∧ point_represents_real p ∧ rational_or_irrational p → real_numbers_correspond p :=
by sorry

end point_to_real_l39_39921


namespace total_books_l39_39737

def sam_books := 110
def joan_books := 102
def tom_books := 125
def alice_books := 97

theorem total_books : sam_books + joan_books + tom_books + alice_books = 434 :=
by
  sorry

end total_books_l39_39737


namespace polynomial_inequality_solution_l39_39504

theorem polynomial_inequality_solution :
  { x : ℝ | x * (x - 5) * (x - 10)^2 > 0 } = { x : ℝ | 0 < x ∧ x < 5 ∨ 10 < x } :=
by
  sorry

end polynomial_inequality_solution_l39_39504


namespace sum_and_product_of_roots_l39_39767

-- Define the equation in terms of |x|
def equation (x : ℝ) : ℝ := |x|^3 - |x|^2 - 6 * |x| + 8

-- Lean statement to prove the sum and product of the roots
theorem sum_and_product_of_roots :
  (∀ x, equation x = 0 → (∃ L : List ℝ, L.sum = 0 ∧ L.prod = 16 ∧ ∀ y ∈ L, equation y = 0)) := 
sorry

end sum_and_product_of_roots_l39_39767


namespace moores_law_transistors_l39_39024

-- Define the initial conditions
def initial_transistors : ℕ := 500000
def doubling_period : ℕ := 2 -- in years
def transistors_doubling (n : ℕ) : ℕ := initial_transistors * 2^n

-- Calculate the number of doubling events from 1995 to 2010
def years_spanned : ℕ := 15
def number_of_doublings : ℕ := years_spanned / doubling_period

-- Expected number of transistors in 2010
def expected_transistors_in_2010 : ℕ := 64000000

theorem moores_law_transistors :
  transistors_doubling number_of_doublings = expected_transistors_in_2010 :=
sorry

end moores_law_transistors_l39_39024


namespace sin_cos_product_l39_39892

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 2 * Real.cos x) : Real.sin x * Real.cos x = 2 / 5 := by
  sorry

end sin_cos_product_l39_39892


namespace mass_percentage_of_H_in_H2O_is_11_19_l39_39219

def mass_of_hydrogen : Float := 1.008
def mass_of_oxygen : Float := 16.00
def mass_of_H2O : Float := 2 * mass_of_hydrogen + mass_of_oxygen
def mass_percentage_hydrogen : Float :=
  (2 * mass_of_hydrogen / mass_of_H2O) * 100

theorem mass_percentage_of_H_in_H2O_is_11_19 :
  mass_percentage_hydrogen = 11.19 :=
  sorry

end mass_percentage_of_H_in_H2O_is_11_19_l39_39219


namespace share_difference_3600_l39_39925

theorem share_difference_3600 (x : ℕ) (p q r : ℕ) (h1 : p = 3 * x) (h2 : q = 7 * x) (h3 : r = 12 * x) (h4 : r - q = 4500) : q - p = 3600 := by
  sorry

end share_difference_3600_l39_39925


namespace sum_of_cubes_application_l39_39542

theorem sum_of_cubes_application : 
  ¬ ((a+1) * (a^2 - a + 1) = a^3 + 1) :=
by
  sorry

end sum_of_cubes_application_l39_39542


namespace range_of_m_necessary_condition_range_of_m_not_sufficient_condition_l39_39013

-- Problem I Statement
theorem range_of_m_necessary_condition (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 8 * x - 20 ≤ 0) → (1 - m^2 ≤ x ∧ x ≤ 1 + m^2)) →
  (-Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3) :=
by sorry

-- Problem II Statement
theorem range_of_m_not_sufficient_condition (m : ℝ) : 
  (∀ x : ℝ, ¬(x^2 - 8 * x - 20 ≤ 0) → ¬(1 - m^2 ≤ x ∧ x ≤ 1 + m^2)) →
  (m ≤ -3 ∨ m ≥ 3) :=
by sorry

end range_of_m_necessary_condition_range_of_m_not_sufficient_condition_l39_39013


namespace contrapositive_of_real_roots_l39_39486

theorem contrapositive_of_real_roots (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0 :=
sorry

end contrapositive_of_real_roots_l39_39486


namespace find_m_l39_39838

noncomputable def union_sets (A B : Set ℝ) : Set ℝ :=
  {x | x ∈ A ∨ x ∈ B}

theorem find_m :
  ∀ (m : ℝ),
    (A = {1, 2 ^ m}) →
    (B = {0, 2}) →
    (union_sets A B = {0, 1, 2, 8}) →
    m = 3 :=
by
  intros m hA hB hUnion
  sorry

end find_m_l39_39838


namespace max_workers_l39_39314

theorem max_workers (S a n : ℕ) (h1 : n > 0) (h2 : S > 0) (h3 : a > 0)
  (h4 : (S:ℚ) / (a * n) > (3 * S:ℚ) / (a * (n + 5))) :
  2 * n + 5 = 9 := 
by
  sorry

end max_workers_l39_39314


namespace probability_cond_satisfied_l39_39154

-- Define the floor and log conditions
def cond1 (x : ℝ) : Prop := ⌊Real.log x / Real.log 2 + 1⌋ = ⌊Real.log x / Real.log 2⌋
def cond2 (x : ℝ) : Prop := ⌊Real.log (2 * x) / Real.log 2 + 1⌋ = ⌊Real.log (2 * x) / Real.log 2⌋
def valid_interval (x : ℝ) : Prop := 0 < x ∧ x < 1

-- Main theorem stating the proof problem
theorem probability_cond_satisfied : 
  (∀ (x : ℝ), valid_interval x → cond1 x → cond2 x → x ∈ Set.Icc (0.25:ℝ) 0.5) → 
  (0.5 - 0.25) / 1 = 1 / 4 := 
by
  -- Proof omitted
  sorry

end probability_cond_satisfied_l39_39154


namespace find_cos_alpha_l39_39782

theorem find_cos_alpha (α : ℝ) (h : (1 - Real.cos α) / Real.sin α = 3) : Real.cos α = -4/5 :=
by
  sorry

end find_cos_alpha_l39_39782


namespace mandy_gets_15_pieces_l39_39799

def initial_pieces : ℕ := 75
def michael_takes (pieces : ℕ) : ℕ := pieces / 3
def paige_takes (pieces : ℕ) : ℕ := (pieces - michael_takes pieces) / 2
def ben_takes (pieces : ℕ) : ℕ := 2 * (pieces - michael_takes pieces - paige_takes pieces) / 5
def mandy_takes (pieces : ℕ) : ℕ := pieces - michael_takes pieces - paige_takes pieces - ben_takes pieces

theorem mandy_gets_15_pieces :
  mandy_takes initial_pieces = 15 :=
by
  sorry

end mandy_gets_15_pieces_l39_39799


namespace intersection_result_l39_39685

open Set

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := { x : ℝ | abs (x - 1) > 2 }

-- Define set B
def B : Set ℝ := { x : ℝ | -x^2 + 6 * x - 8 > 0 }

-- Define the complement of A in U
def compl_A : Set ℝ := U \ A

-- Define the intersection of compl_A and B
def inter_complA_B : Set ℝ := compl_A ∩ B

-- Prove that the intersection is equal to the given set
theorem intersection_result : inter_complA_B = { x : ℝ | 2 < x ∧ x ≤ 3 } :=
by
  sorry

end intersection_result_l39_39685


namespace number_of_real_solutions_l39_39064

theorem number_of_real_solutions (x : ℝ) (n : ℤ) : 
  (3 : ℝ) * x^2 - 27 * (n : ℝ) + 29 = 0 → n = ⌊x⌋ →  ∃! x, (3 : ℝ) * x^2 - 27 * (⌊x⌋ : ℝ) + 29 = 0 := 
sorry

end number_of_real_solutions_l39_39064


namespace smallest_number_when_diminished_by_7_is_divisible_l39_39697

-- Variables for divisors
def divisor1 : Nat := 12
def divisor2 : Nat := 16
def divisor3 : Nat := 18
def divisor4 : Nat := 21
def divisor5 : Nat := 28

-- The smallest number x which, when diminished by 7, is divisible by the divisors.
theorem smallest_number_when_diminished_by_7_is_divisible (x : Nat) : 
  (x - 7) % divisor1 = 0 ∧ 
  (x - 7) % divisor2 = 0 ∧ 
  (x - 7) % divisor3 = 0 ∧ 
  (x - 7) % divisor4 = 0 ∧ 
  (x - 7) % divisor5 = 0 → 
  x = 1015 := 
sorry

end smallest_number_when_diminished_by_7_is_divisible_l39_39697


namespace house_assignment_l39_39673

theorem house_assignment (n : ℕ) (assign : Fin n → Fin n) (pref : Fin n → Fin n → Fin n → Prop) :
  (∀ (p : Fin n), ∃ (better_assign : Fin n → Fin n),
    (∃ q, pref p (assign p) (better_assign p) ∧ pref q (assign q) (better_assign p) ∧ better_assign q ≠ assign q)
  ) → (∃ p, pref p (assign p) (assign p))
:= sorry

end house_assignment_l39_39673


namespace speed_for_remaining_distance_l39_39208

theorem speed_for_remaining_distance
  (t_total : ℝ) (v1 : ℝ) (d_total : ℝ)
  (t_total_def : t_total = 1.4)
  (v1_def : v1 = 4)
  (d_total_def : d_total = 5.999999999999999) :
  ∃ v2 : ℝ, v2 = 5 := 
by
  sorry

end speed_for_remaining_distance_l39_39208


namespace months_to_save_l39_39589

/-- The grandfather saves 530 yuan from his pension every month. -/
def savings_per_month : ℕ := 530

/-- The price of the smartphone is 2000 yuan. -/
def smartphone_price : ℕ := 2000

/-- The number of months needed to save enough money to buy the smartphone. -/
def months_needed : ℕ := smartphone_price / savings_per_month

/-- Proof that the number of months needed is 4. -/
theorem months_to_save : months_needed = 4 :=
by
  sorry

end months_to_save_l39_39589


namespace combined_garden_area_l39_39861

-- Definitions for the sizes and counts of the gardens.
def Mancino_gardens : ℕ := 4
def Marquita_gardens : ℕ := 3
def Matteo_gardens : ℕ := 2
def Martina_gardens : ℕ := 5

def Mancino_garden_area : ℕ := 16 * 5
def Marquita_garden_area : ℕ := 8 * 4
def Matteo_garden_area : ℕ := 12 * 6
def Martina_garden_area : ℕ := 10 * 3

-- The total combined area to be proven.
def total_area : ℕ :=
  (Mancino_gardens * Mancino_garden_area) +
  (Marquita_gardens * Marquita_garden_area) +
  (Matteo_gardens * Matteo_garden_area) +
  (Martina_gardens * Martina_garden_area)

-- Proof statement for the combined area.
theorem combined_garden_area : total_area = 710 :=
by sorry

end combined_garden_area_l39_39861


namespace num_distinct_log_values_l39_39899

-- Defining the set of numbers
def number_set : Set ℕ := {1, 2, 3, 4, 6, 9}

-- Define a function to count distinct logarithmic values
noncomputable def distinct_log_values (s : Set ℕ) : ℕ := 
  -- skipped, assume the implementation is done correctly
  sorry 

theorem num_distinct_log_values : distinct_log_values number_set = 17 :=
by
  sorry

end num_distinct_log_values_l39_39899


namespace no_preimage_for_p_gt_1_l39_39722

def f (x : ℝ) : ℝ := -x^2 + 2 * x

theorem no_preimage_for_p_gt_1 (P : ℝ) (hP : P > 1) : ¬ ∃ x : ℝ, f x = P :=
sorry

end no_preimage_for_p_gt_1_l39_39722


namespace first_reduction_percentage_l39_39297

theorem first_reduction_percentage (P : ℝ) (x : ℝ) (h : 0.30 * (1 - x / 100) * P = 0.225 * P) : x = 25 :=
by
  sorry

end first_reduction_percentage_l39_39297


namespace zaim_larger_part_l39_39437

theorem zaim_larger_part (x y : ℕ) (h_sum : x + y = 20) (h_prod : x * y = 96) : max x y = 12 :=
by
  -- The proof goes here
  sorry

end zaim_larger_part_l39_39437


namespace negation_equiv_l39_39091

-- Define the initial proposition
def initial_proposition (x : ℝ) : Prop :=
  x^2 - x + 1 > 0

-- Define the negation of the initial proposition
def negated_proposition : Prop :=
  ∃ x₀ : ℝ, x₀^2 - x₀ + 1 ≤ 0

-- The statement asserting the negation equivalence
theorem negation_equiv :
  (¬ ∀ x : ℝ, initial_proposition x) ↔ negated_proposition :=
by sorry

end negation_equiv_l39_39091


namespace general_term_of_sequence_l39_39379

theorem general_term_of_sequence (n : ℕ) :
  ∃ (a : ℕ → ℚ),
    a 1 = 1 / 2 ∧ 
    a 2 = -2 ∧ 
    a 3 = 9 / 2 ∧ 
    a 4 = -8 ∧ 
    a 5 = 25 / 2 ∧ 
    ∀ n, a n = (-1) ^ (n + 1) * (n ^ 2 / 2) := 
by
  sorry

end general_term_of_sequence_l39_39379


namespace investment_A_l39_39307

-- Define constants B and C's investment values, C's share, and total profit.
def B_investment : ℕ := 8000
def C_investment : ℕ := 9000
def C_share : ℕ := 36000
def total_profit : ℕ := 88000

-- Problem statement to prove
theorem investment_A (A_investment : ℕ) : 
  (A_investment + B_investment + C_investment = 17000) → 
  (C_investment * total_profit = C_share * (A_investment + B_investment + C_investment)) →
  A_investment = 5000 :=
by 
  intros h1 h2
  sorry

end investment_A_l39_39307


namespace max_reflections_max_reflections_example_l39_39502

-- Definition of the conditions
def angle_cda := 10  -- angle in degrees
def max_angle := 90  -- practical limit for angle of reflections

-- Given that the angle of incidence after n reflections is 10n degrees,
-- prove that the largest possible n is 9 before exceeding practical limits.
theorem max_reflections (n : ℕ) (h₁ : angle_cda = 10) (h₂ : max_angle = 90) :
  10 * n ≤ 90 :=
by sorry

-- Specific case instantiating n = 9
theorem max_reflections_example : (10 : ℕ) * 9 ≤ 90 := max_reflections 9 rfl rfl

end max_reflections_max_reflections_example_l39_39502


namespace can_cross_all_rivers_and_extra_material_l39_39093

-- Definitions for river widths, bridge length, and additional material.
def river1_width : ℕ := 487
def river2_width : ℕ := 621
def river3_width : ℕ := 376
def bridge_length : ℕ := 295
def additional_material : ℕ := 1020

-- Calculations for material needed for each river.
def material_needed_for_river1 : ℕ := river1_width - bridge_length
def material_needed_for_river2 : ℕ := river2_width - bridge_length
def material_needed_for_river3 : ℕ := river3_width - bridge_length

-- Total material needed to cross all three rivers.
def total_material_needed : ℕ := material_needed_for_river1 + material_needed_for_river2 + material_needed_for_river3

-- The main theorem statement to prove.
theorem can_cross_all_rivers_and_extra_material :
  total_material_needed <= additional_material ∧ (additional_material - total_material_needed = 421) := 
by 
  sorry

end can_cross_all_rivers_and_extra_material_l39_39093


namespace find_arrays_l39_39347

theorem find_arrays (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a ∣ b * c * d - 1 ∧ b ∣ a * c * d - 1 ∧ c ∣ a * b * d - 1 ∧ d ∣ a * b * c - 1 →
  (a = 2 ∧ b = 3 ∧ c = 7 ∧ d = 11) ∨
  (a = 2 ∧ b = 3 ∧ c = 11 ∧ d = 13) := by
  sorry

end find_arrays_l39_39347


namespace sabrina_fraction_books_second_month_l39_39550

theorem sabrina_fraction_books_second_month (total_books : ℕ) (pages_per_book : ℕ) (books_first_month : ℕ) (pages_total_read : ℕ)
  (h_total_books : total_books = 14)
  (h_pages_per_book : pages_per_book = 200)
  (h_books_first_month : books_first_month = 4)
  (h_pages_total_read : pages_total_read = 1000) :
  let total_pages := total_books * pages_per_book
  let pages_first_month := books_first_month * pages_per_book
  let pages_remaining := total_pages - pages_first_month
  let books_remaining := total_books - books_first_month
  let pages_read_first_month := total_pages - pages_total_read
  let pages_read_second_month := pages_read_first_month - pages_first_month
  let books_second_month := pages_read_second_month / pages_per_book
  let fraction_books := books_second_month / books_remaining
  fraction_books = 1 / 2 :=
by
  sorry

end sabrina_fraction_books_second_month_l39_39550


namespace subtraction_problem_digits_sum_l39_39231

theorem subtraction_problem_digits_sum :
  ∃ (K L M N : ℕ), K < 10 ∧ L < 10 ∧ M < 10 ∧ N < 10 ∧ 
  ((6000 + K * 100 + 0 + L) - (900 + N * 10 + 4) = 2011) ∧ 
  (K + L + M + N = 17) :=
by
  sorry

end subtraction_problem_digits_sum_l39_39231


namespace stratified_sampling_groupD_l39_39460

-- Definitions for the conditions
def totalDistrictCount : ℕ := 38
def groupADistrictCount : ℕ := 4
def groupBDistrictCount : ℕ := 10
def groupCDistrictCount : ℕ := 16
def groupDDistrictCount : ℕ := 8
def numberOfCitiesToSelect : ℕ := 9

-- Define stratified sampling calculation with a floor function or rounding
noncomputable def numberSelectedFromGroupD : ℕ := (groupDDistrictCount * numberOfCitiesToSelect) / totalDistrictCount

-- The theorem to prove 
theorem stratified_sampling_groupD : numberSelectedFromGroupD = 2 := by
  sorry -- This is where the proof would go

end stratified_sampling_groupD_l39_39460


namespace range_of_m_l39_39967

theorem range_of_m (x m : ℝ) (h1 : (x ≥ 0) ∧ (x ≠ 1) ∧ (x = (6 - m) / 4)) :
    m ≤ 6 ∧ m ≠ 2 :=
by
  sorry

end range_of_m_l39_39967


namespace largest_cannot_be_sum_of_two_composites_l39_39177

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def cannot_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∀ a b : ℕ, is_composite a → is_composite b → a + b ≠ n

theorem largest_cannot_be_sum_of_two_composites :
  ∀ n, n > 11 → ¬ cannot_be_sum_of_two_composites n := 
by {
  sorry
}

end largest_cannot_be_sum_of_two_composites_l39_39177


namespace maximum_value_of_m_solve_inequality_l39_39544

theorem maximum_value_of_m (a b : ℝ) (h : a ≠ 0) : 
  ∃ m : ℝ, (∀ a b : ℝ, a ≠ 0 → |a + b| + |a - b| ≥ m * |a|) ∧ (m = 2) :=
by
  use 2
  sorry

theorem solve_inequality (x : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x - 2| ≤ 2 → (1/2 ≤ x ∧ x ≤ 5/2)) :=
by
  sorry

end maximum_value_of_m_solve_inequality_l39_39544


namespace combined_weight_loss_l39_39710

theorem combined_weight_loss :
  let aleesia_loss_per_week := 1.5
  let aleesia_weeks := 10
  let alexei_loss_per_week := 2.5
  let alexei_weeks := 8
  (aleesia_loss_per_week * aleesia_weeks) + (alexei_loss_per_week * alexei_weeks) = 35 := by
sorry

end combined_weight_loss_l39_39710


namespace number_of_three_star_reviews_l39_39603

theorem number_of_three_star_reviews:
  ∀ (x : ℕ),
  (6 * 5 + 7 * 4 + 1 * 2 + x * 3) / 18 = 4 →
  x = 4 :=
by
  intros x H
  sorry  -- Placeholder for the proof

end number_of_three_star_reviews_l39_39603


namespace Ivan_walk_time_l39_39779

variables (u v : ℝ) (T t : ℝ)

-- Define the conditions
def condition1 : Prop := T = 10 * v / u
def condition2 : Prop := T + 70 = t
def condition3 : Prop := v * t = u * T + v * (t - T + 70)

-- Problem statement: Given the conditions, prove T = 80
theorem Ivan_walk_time (h1 : condition1 u v T) (h2 : condition2 T t) (h3 : condition3 u v T t) : 
  T = 80 := by
  sorry

end Ivan_walk_time_l39_39779


namespace problem1_problem2_l39_39157

-- Definitions for Problem 1
def cond1 (x t : ℝ) : Prop := |2 * x + t| - t ≤ 8
def sol_set1 (x : ℝ) : Prop := -5 ≤ x ∧ x ≤ 4

theorem problem1 {t : ℝ} : (∀ x, cond1 x t → sol_set1 x) → t = 1 :=
sorry

-- Definitions for Problem 2
def cond2 (x y z : ℝ) : Prop := x^2 + (1 / 4) * y^2 + (1 / 9) * z^2 = 2

theorem problem2 {x y z : ℝ} : cond2 x y z → x + y + z ≤ 2 * Real.sqrt 7 :=
sorry

end problem1_problem2_l39_39157


namespace voter_ratio_l39_39677

theorem voter_ratio (Vx Vy : ℝ) (hx : 0.72 * Vx + 0.36 * Vy = 0.60 * (Vx + Vy)) : Vx = 2 * Vy :=
by
sorry

end voter_ratio_l39_39677


namespace Danielle_has_6_rooms_l39_39946

axiom Danielle_rooms : ℕ
axiom Heidi_rooms : ℕ
axiom Grant_rooms : ℕ

axiom Heidi_has_3_times_Danielle : Heidi_rooms = 3 * Danielle_rooms
axiom Grant_has_1_9_Heidi : Grant_rooms = Heidi_rooms / 9
axiom Grant_has_2_rooms : Grant_rooms = 2

theorem Danielle_has_6_rooms : Danielle_rooms = 6 :=
by {
  -- proof steps would go here
  sorry
}

end Danielle_has_6_rooms_l39_39946


namespace log_comparison_l39_39002

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def log4 (x : ℝ) : ℝ := Real.log x / Real.log 4
noncomputable def log6 (x : ℝ) : ℝ := Real.log x / Real.log 6

theorem log_comparison :
  let a := log2 6
  let b := log4 12
  let c := log6 18
  a > b ∧ b > c :=
by 
  sorry

end log_comparison_l39_39002


namespace smallest_x_fraction_floor_l39_39085

theorem smallest_x_fraction_floor (x : ℝ) (h : ⌊x⌋ / x = 7 / 8) : x = 48 / 7 :=
sorry

end smallest_x_fraction_floor_l39_39085


namespace squirrels_acorns_l39_39644

theorem squirrels_acorns (squirrels : ℕ) (total_collected : ℕ) (acorns_needed_per_squirrel : ℕ) (total_needed : ℕ) (acorns_still_needed : ℕ) : 
  squirrels = 5 → 
  total_collected = 575 → 
  acorns_needed_per_squirrel = 130 → 
  total_needed = squirrels * acorns_needed_per_squirrel →
  acorns_still_needed = total_needed - total_collected →
  acorns_still_needed / squirrels = 15 :=
by
  sorry

end squirrels_acorns_l39_39644


namespace eliot_account_balance_l39_39852

variable (A E : ℝ)

theorem eliot_account_balance (h1 : A - E = (1/12) * (A + E)) (h2 : 1.10 * A = 1.20 * E + 20) : 
  E = 200 := 
by 
  sorry

end eliot_account_balance_l39_39852


namespace general_term_min_S9_and_S10_sum_b_seq_l39_39853

-- Definitions for the arithmetic sequence {a_n}
def a_seq (n : ℕ) : ℤ := 2 * ↑n - 20

-- Conditions provided in the problem
def cond1 : Prop := a_seq 4 = -12
def cond2 : Prop := a_seq 8 = -4

-- The sum of the first n terms S_n of the arithmetic sequence {a_n}
def S_n (n : ℕ) : ℤ := n * (a_seq 1 + a_seq n) / 2

-- Definitions for the new sequence {b_n}
def b_seq (n : ℕ) : ℤ := 2^n - 20

-- The sum of the first n terms of the new sequence {b_n}
def T_n (n : ℕ) : ℤ := (2^(n + 1) - 2) - 20 * n

-- Lean 4 theorem statements
theorem general_term (h1 : cond1) (h2 : cond2) : ∀ n : ℕ, a_seq n = 2 * ↑n - 20 :=
sorry

theorem min_S9_and_S10 (h1 : cond1) (h2 : cond2) : S_n 9 = -90 ∧ S_n 10 = -90 :=
sorry

theorem sum_b_seq (n : ℕ) : ∀ k : ℕ, (k < n) → T_n k = (2^(k+1) - 20 * k - 2) :=
sorry

end general_term_min_S9_and_S10_sum_b_seq_l39_39853


namespace solve_system_of_equations_l39_39489

theorem solve_system_of_equations :
  ∃ (x y z : ℝ),
    (x^2 + y^2 + 8 * x - 6 * y = -20) ∧
    (x^2 + z^2 + 8 * x + 4 * z = -10) ∧
    (y^2 + z^2 - 6 * y + 4 * z = 0) ∧
    ((x = -3 ∧ y = 1 ∧ z = 1) ∨
     (x = -3 ∧ y = 1 ∧ z = -5) ∨
     (x = -3 ∧ y = 5 ∧ z = 1) ∨
     (x = -3 ∧ y = 5 ∧ z = -5) ∨
     (x = -5 ∧ y = 1 ∧ z = 1) ∨
     (x = -5 ∧ y = 1 ∧ z = -5) ∨
     (x = -5 ∧ y = 5 ∧ z = 1) ∨
     (x = -5 ∧ y = 5 ∧ z = -5)) :=
sorry

end solve_system_of_equations_l39_39489


namespace value_of_f_l39_39547

def f (x z : ℕ) (y : ℕ) : ℕ := 2 * x^2 + y - z

theorem value_of_f (y : ℕ) (h1 : f 2 3 y = 100) : f 5 7 y = 138 := by
  sorry

end value_of_f_l39_39547


namespace unique_solution_l39_39144

theorem unique_solution (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (eq1 : x * y + y * z + z * x = 12) (eq2 : x * y * z = 2 + x + y + z) :
  x = 2 ∧ y = 2 ∧ z = 2 :=
by 
  sorry

end unique_solution_l39_39144


namespace range_of_a_l39_39329

theorem range_of_a {a : ℝ} (h1 : ∀ x : ℝ, x - a ≥ 0 → 2 * x - 10 < 0) :
  3 < a ∧ a ≤ 4 :=
by
  sorry

end range_of_a_l39_39329


namespace sum_of_roots_of_equation_l39_39054

theorem sum_of_roots_of_equation :
  (∀ x : ℝ, (x + 3) * (x - 4) = 22 → ∃ a b : ℝ, (x - a) * (x - b) = 0 ∧ a + b = 1) :=
by
  sorry

end sum_of_roots_of_equation_l39_39054


namespace age_of_b_l39_39606

variable (A B C : ℕ)

-- Conditions
def avg_abc : Prop := A + B + C = 78
def avg_ac : Prop := A + C = 58

-- Question: Prove that B = 20
theorem age_of_b (h1 : avg_abc A B C) (h2 : avg_ac A C) : B = 20 := 
by sorry

end age_of_b_l39_39606


namespace basketball_success_rate_l39_39420

theorem basketball_success_rate (p : ℝ) (h : 1 - p^2 = 16 / 25) : p = 3 / 5 :=
sorry

end basketball_success_rate_l39_39420


namespace expression_greater_than_m_l39_39845

theorem expression_greater_than_m (m : ℚ) : m + 2 > m :=
by sorry

end expression_greater_than_m_l39_39845


namespace sin_value_l39_39374

theorem sin_value (α : ℝ) (hα : 0 < α ∧ α < π) 
  (hcos : Real.cos (α + Real.pi / 6) = -3 / 5) : 
  Real.sin (2 * α + Real.pi / 12) = -17 * Real.sqrt 2 / 50 := 
sorry

end sin_value_l39_39374


namespace smallest_solution_l39_39599

theorem smallest_solution (x : ℕ) (h1 : 6 * x ≡ 17 [MOD 31]) (h2 : x ≡ 3 [MOD 7]) : x = 24 := 
by 
  sorry

end smallest_solution_l39_39599


namespace problem_1_problem_2_l39_39175
-- Import the entire Mathlib library.

-- Problem (1)
theorem problem_1 (x y : ℝ) (h1 : |x - 3 * y| < 1 / 2) (h2 : |x + 2 * y| < 1 / 6) : |x| < 3 / 10 :=
sorry

-- Problem (2)
theorem problem_2 (x y : ℝ) : x^4 + 16 * y^4 ≥ 2 * x^3 * y + 8 * x * y^3 :=
sorry

end problem_1_problem_2_l39_39175


namespace find_x_l39_39035

noncomputable def series_sum (x : ℝ) : ℝ :=
  ∑' n : ℕ, (2 * n + 1) * x^n

theorem find_x (x : ℝ) (H : series_sum x = 16) : 
  x = (33 - Real.sqrt 129) / 32 :=
by
  sorry

end find_x_l39_39035


namespace corey_gave_more_books_l39_39102

def books_given_by_mike : ℕ := 10
def total_books_received_by_lily : ℕ := 35
def books_given_by_corey : ℕ := total_books_received_by_lily - books_given_by_mike
def difference_in_books (a b : ℕ) : ℕ := a - b

theorem corey_gave_more_books :
  difference_in_books books_given_by_corey books_given_by_mike = 15 := by
sorry

end corey_gave_more_books_l39_39102


namespace sin_cos_relationship_l39_39275

theorem sin_cos_relationship (α : ℝ) (h1 : Real.pi / 2 < α) (h2 : α < Real.pi) : 
  Real.sin α - Real.cos α > 1 :=
sorry

end sin_cos_relationship_l39_39275


namespace matrix_power_eigenvector_l39_39919

section MatrixEigen
variable (B : Matrix (Fin 2) (Fin 2) ℝ) (v : Fin 2 → ℝ)

theorem matrix_power_eigenvector (h : B.mulVec ![3, -1] = ![-12, 4]) :
  (B ^ 5).mulVec ![3, -1] = ![-3072, 1024] := 
  sorry
end MatrixEigen

end matrix_power_eigenvector_l39_39919


namespace most_efficient_packing_l39_39678

theorem most_efficient_packing :
  ∃ box_size, 
  (box_size = 3 ∨ box_size = 6 ∨ box_size = 9) ∧ 
  (∀ q ∈ [21, 18, 15, 12, 9], q % box_size = 0) ∧
  box_size = 3 :=
by
  sorry

end most_efficient_packing_l39_39678


namespace sum_lengths_AMC_l39_39119

theorem sum_lengths_AMC : 
  let length_A := 2 * (Real.sqrt 2) + 2
  let length_M := 3 + 3 + 2 * (Real.sqrt 2)
  let length_C := 3 + 3 + 2
  length_A + length_M + length_C = 13 + 4 * (Real.sqrt 2)
  := by
  sorry

end sum_lengths_AMC_l39_39119


namespace find_a_l39_39183

theorem find_a (A B : Set ℝ) (a : ℝ)
  (hA : A = {1, 2})
  (hB : B = {a, a^2 + 1})
  (hUnion : A ∪ B = {0, 1, 2}) :
  a = 0 :=
sorry

end find_a_l39_39183


namespace best_model_l39_39965

theorem best_model (R1 R2 R3 R4 : ℝ) :
  R1 = 0.78 → R2 = 0.85 → R3 = 0.61 → R4 = 0.31 →
  (R2 = max R1 (max R2 (max R3 R4))) :=
by
  intros hR1 hR2 hR3 hR4
  sorry

end best_model_l39_39965


namespace train_passing_time_correct_l39_39164

noncomputable def train_passing_time (L1 L2 : ℕ) (S1 S2 : ℕ) : ℝ :=
  let S1_mps := S1 * (1000 / 3600)
  let S2_mps := S2 * (1000 / 3600)
  let relative_speed := S1_mps + S2_mps
  let total_length := L1 + L2
  total_length / relative_speed

theorem train_passing_time_correct :
  train_passing_time 105 140 45 36 = 10.89 := by
  sorry

end train_passing_time_correct_l39_39164


namespace total_spend_on_four_games_l39_39900

noncomputable def calculate_total_spend (batman_price : ℝ) (superman_price : ℝ)
                                        (batman_discount : ℝ) (superman_discount : ℝ)
                                        (tax_rate : ℝ) (game1_price : ℝ) (game2_price : ℝ) : ℝ :=
  let batman_discounted_price := batman_price - batman_discount * batman_price
  let superman_discounted_price := superman_price - superman_discount * superman_price
  let batman_price_after_tax := batman_discounted_price + tax_rate * batman_discounted_price
  let superman_price_after_tax := superman_discounted_price + tax_rate * superman_discounted_price
  batman_price_after_tax + superman_price_after_tax + game1_price + game2_price

theorem total_spend_on_four_games :
  calculate_total_spend 13.60 5.06 0.10 0.05 0.08 7.25 12.50 = 38.16 :=
by sorry

end total_spend_on_four_games_l39_39900


namespace xy_maximum_value_l39_39608

theorem xy_maximum_value (x y : ℝ) (h : 3 * (x^2 + y^2) = x + 2 * y) : x - 2 * y ≤ 2 / 3 :=
sorry

end xy_maximum_value_l39_39608


namespace side_length_of_square_l39_39257

theorem side_length_of_square (d : ℝ) (h_d : d = 2 * Real.sqrt 2) : 
  ∃ s : ℝ, d = s * Real.sqrt 2 ∧ s = 2 := by
  sorry

end side_length_of_square_l39_39257


namespace product_of_y_values_l39_39611

theorem product_of_y_values :
  (∀ (x y : ℤ), x ^ 3 + y ^ 2 - 3 * y + 1 < 0 ∧ 3 * x ^ 3 - y ^ 2 + 3 * y > 0 → (y = 1 ∨ y = 2)) →
  (∀ (x y₁ x' y₂ : ℤ), (x, y₁) ≠ (x', y₂) → x = x' ∨ y₁ ≠ y₂) →
  (∀ (x y : ℤ), (x ^ 3 + y ^ 2 - 3 * y + 1 < 0 ∧ 3 * x ^ 3 - y ^ 2 + 3 * y > 0 → y = 1 ∨ y = 2) →
    (∃ (y₁ y₂ : ℤ), y₁ = 1 ∧ y₂ = 2 ∧ y₁ * y₂ = 2)) :=
by {
  sorry
}

end product_of_y_values_l39_39611


namespace fraction_value_l39_39325

theorem fraction_value
  (x y z : ℝ)
  (h1 : x / 2 = y / 3)
  (h2 : y / 3 = z / 5)
  (h3 : 2 * x + y ≠ 0) :
  (x + y - 3 * z) / (2 * x + y) = -10 / 7 := by
  -- Add sorry to skip the proof.
  sorry

end fraction_value_l39_39325


namespace equivalent_problem_l39_39118

variable {x y : Real}

theorem equivalent_problem 
  (h1 : (x + y)^2 = 81) 
  (h2 : x * y = 15) :
  (x - y)^2 = 21 ∧ (x + y) * (x - y) = Real.sqrt 1701 :=
by
  sorry

end equivalent_problem_l39_39118


namespace words_per_page_l39_39639

theorem words_per_page (p : ℕ) (h1 : p ≤ 120) (h2 : 154 * p % 221 = 207) : p = 100 :=
sorry

end words_per_page_l39_39639


namespace Sue_waited_in_NY_l39_39541

-- Define the conditions as constants and assumptions
def T_NY_SF : ℕ := 24
def T_total : ℕ := 58
def T_NO_NY : ℕ := (3 * T_NY_SF) / 4

-- Define the waiting time
def T_wait : ℕ := T_total - T_NO_NY - T_NY_SF

-- Theorem stating the problem
theorem Sue_waited_in_NY :
  T_wait = 16 :=
by
  -- Implicitly using the given conditions
  sorry

end Sue_waited_in_NY_l39_39541


namespace team_total_points_l39_39692

theorem team_total_points (T : ℕ) (h1 : ∃ x : ℕ, x = T / 6)
    (h2 : (T + (92 - 85)) / 6 = 84) : T = 497 := 
by sorry

end team_total_points_l39_39692


namespace team_a_games_played_l39_39151

theorem team_a_games_played (a b: ℕ) (hA_wins : 3 * a = 4 * wins_A)
(hB_wins : 2 * b = 3 * wins_B)
(hB_more_wins : wins_B = wins_A + 8)
(hB_more_loss : b - wins_B = a - wins_A + 8) :
  a = 192 := 
by
  sorry

end team_a_games_played_l39_39151


namespace compute_H_five_times_l39_39694

def H (x : ℝ) : ℝ := x^2 - 2 * x - 1

theorem compute_H_five_times : H (H (H (H (H 2)))) = -1 := by
  sorry

end compute_H_five_times_l39_39694


namespace olivia_quarters_left_l39_39321

-- Define the initial condition and action condition as parameters
def initial_quarters : ℕ := 11
def quarters_spent : ℕ := 4
def quarters_left : ℕ := initial_quarters - quarters_spent

-- The theorem to state the result
theorem olivia_quarters_left : quarters_left = 7 := by
  sorry

end olivia_quarters_left_l39_39321


namespace expand_polynomial_expression_l39_39766

theorem expand_polynomial_expression (x : ℝ) : 
  (x + 6) * (x + 8) * (x - 3) = x^3 + 11 * x^2 + 6 * x - 144 :=
by
  sorry

end expand_polynomial_expression_l39_39766


namespace option_e_is_perfect_square_l39_39827

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem option_e_is_perfect_square :
  is_perfect_square (4^10 * 5^5 * 6^10) :=
sorry

end option_e_is_perfect_square_l39_39827


namespace n1_prime_n2_not_prime_l39_39419

def n1 := 1163
def n2 := 16424
def N := 19101112
def N_eq : N = n1 * n2 := by decide

theorem n1_prime : Prime n1 := 
sorry

theorem n2_not_prime : ¬ Prime n2 :=
sorry

end n1_prime_n2_not_prime_l39_39419


namespace no_solution_eqn_l39_39156

theorem no_solution_eqn : ∀ x : ℝ, x ≠ -11 ∧ x ≠ -8 ∧ x ≠ -12 ∧ x ≠ -7 →
  ¬ (1 / (x + 11) + 1 / (x + 8) = 1 / (x + 12) + 1 / (x + 7)) :=
by
  intros x h
  sorry

end no_solution_eqn_l39_39156


namespace toy_selling_price_l39_39106

theorem toy_selling_price (x : ℝ) (units_sold : ℝ) (profit_per_day : ℝ) : 
  (units_sold = 200 + 20 * (80 - x)) → 
  (profit_per_day = (x - 60) * units_sold) → 
  profit_per_day = 2500 → 
  x ≤ 60 * 1.4 → 
  x = 65 :=
by
  intros h1 h2 h3 h4
  sorry

end toy_selling_price_l39_39106


namespace major_axis_length_l39_39241

theorem major_axis_length (r : ℝ) (minor_axis : ℝ) (major_axis : ℝ) 
  (h1 : r = 2) 
  (h2 : minor_axis = 2 * r) 
  (h3 : major_axis = minor_axis + 0.75 * minor_axis) : 
  major_axis = 7 := 
by 
  sorry

end major_axis_length_l39_39241


namespace frog_probability_l39_39346

noncomputable def frog_escape_prob (P : ℕ → ℚ) : Prop :=
  P 0 = 0 ∧
  P 11 = 1 ∧
  (∀ N, 0 < N ∧ N < 11 → 
    P N = (N + 1) / 12 * P (N - 1) + (1 - (N + 1) / 12) * P (N + 1)) ∧
  P 2 = 72 / 167

theorem frog_probability : ∃ P : ℕ → ℚ, frog_escape_prob P :=
sorry

end frog_probability_l39_39346


namespace symmetric_circle_eq_l39_39517

theorem symmetric_circle_eq :
  (∃ f : ℝ → ℝ → Prop, (∀ x y, f x y ↔ (x - 2)^2 + (y + 1)^2 = 1)) →
  (∃ line : ℝ → ℝ → Prop, (∀ x y, line x y ↔ x - y + 3 = 0)) →
  (∃ eq : ℝ → ℝ → Prop, (∀ x y, eq x y ↔ (x - 4)^2 + (y - 5)^2 = 1)) :=
by
  sorry

end symmetric_circle_eq_l39_39517


namespace ratio_of_areas_l39_39670

variables (s : ℝ)

def side_length_square := s
def longer_side_rect := 1.2 * s
def shorter_side_rect := 0.8 * s

noncomputable def area_rectangle := longer_side_rect s * shorter_side_rect s
noncomputable def area_triangle := (1 / 2) * (longer_side_rect s * shorter_side_rect s)

theorem ratio_of_areas :
  (area_triangle s) / (area_rectangle s) = 1 / 2 :=
by
  sorry

end ratio_of_areas_l39_39670


namespace pocket_money_calculation_l39_39436

theorem pocket_money_calculation
  (a b c d e : ℝ)
  (h1 : (a + b + c + d + e) / 5 = 2300)
  (h2 : (a + b) / 2 = 3000)
  (h3 : (b + c) / 2 = 2100)
  (h4 : (c + d) / 2 = 2750)
  (h5 : a = b + 800) :
  d = 3900 :=
by
  sorry

end pocket_money_calculation_l39_39436


namespace functional_eq_zero_function_l39_39369

theorem functional_eq_zero_function (f : ℝ → ℝ) (k : ℝ) (h : ∀ x y : ℝ, f (f x + f y + k * x * y) = x * f y + y * f x) : 
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end functional_eq_zero_function_l39_39369


namespace maximum_delta_value_l39_39166

-- Definition of the sequence a 
def a (n : ℕ) : ℕ := 1 + n^3

-- Definition of δ_n as the gcd of consecutive terms in the sequence a
def delta (n : ℕ) : ℕ := Nat.gcd (a (n + 1)) (a n)

-- Main theorem statement
theorem maximum_delta_value : ∃ n, delta n = 7 :=
by
  -- Insert the proof later
  sorry

end maximum_delta_value_l39_39166


namespace find_f_l39_39948

-- Define the conditions as hypotheses
def cond1 (f : ℕ) (p : ℕ) : Prop := f + p = 75
def cond2 (f : ℕ) (p : ℕ) : Prop := (f + p) + p = 143

-- The theorem stating that given the conditions, f must be 7
theorem find_f (f p : ℕ) (h1 : cond1 f p) (h2 : cond2 f p) : f = 7 := 
  by
  sorry

end find_f_l39_39948


namespace determine_b_for_constant_remainder_l39_39336

theorem determine_b_for_constant_remainder (b : ℚ) :
  ∃ r : ℚ, ∀ x : ℚ,  (12 * x^3 - 9 * x^2 + b * x + 8) / (3 * x^2 - 4 * x + 2) = r ↔ b = -4 / 3 :=
by sorry

end determine_b_for_constant_remainder_l39_39336


namespace garden_perimeter_is_48_l39_39283

def square_garden_perimeter (pond_area garden_remaining_area : ℕ) : ℕ :=
  let garden_area := pond_area + garden_remaining_area
  let side_length := Int.natAbs (Int.sqrt garden_area)
  4 * side_length

theorem garden_perimeter_is_48 :
  square_garden_perimeter 20 124 = 48 :=
  by
  sorry

end garden_perimeter_is_48_l39_39283


namespace increase_in_sold_items_l39_39825

variable (P N M : ℝ)
variable (discounted_price := 0.9 * P)
variable (increased_total_income := 1.17 * P * N)

theorem increase_in_sold_items (h: 0.9 * P * M = increased_total_income):
  M = 1.3 * N :=
  by sorry

end increase_in_sold_items_l39_39825


namespace students_per_class_l39_39874

theorem students_per_class :
  let buns_per_package := 8
  let packages := 30
  let buns_per_student := 2
  let classes := 4
  (packages * buns_per_package) / (buns_per_student * classes) = 30 :=
by
  sorry

end students_per_class_l39_39874


namespace find_n_l39_39936

theorem find_n 
  (N : ℕ) 
  (hn : ¬ (N = 0))
  (parts_inv_prop : ∀ k, 1 ≤ k → k ≤ n → N / (k * (k + 1)) = x / (n * (n + 1))) 
  (smallest_part : (N : ℝ) / 400 = N / (n * (n + 1))) : 
  n = 20 :=
sorry

end find_n_l39_39936


namespace probability_same_gender_l39_39390

theorem probability_same_gender :
  let males := 3
  let females := 2
  let total := males + females
  let total_ways := Nat.choose total 2
  let male_ways := Nat.choose males 2
  let female_ways := Nat.choose females 2
  let same_gender_ways := male_ways + female_ways
  let probability := (same_gender_ways : ℚ) / total_ways
  probability = 2 / 5 :=
by
  sorry

end probability_same_gender_l39_39390


namespace intersection_with_x_axis_l39_39381

theorem intersection_with_x_axis (t : ℝ) (x y : ℝ) 
  (h1 : x = -2 + 5 * t) 
  (h2 : y = 1 - 2 * t) 
  (h3 : y = 0) : x = 1 / 2 := 
by 
  sorry

end intersection_with_x_axis_l39_39381


namespace fraction_is_five_over_nine_l39_39092

theorem fraction_is_five_over_nine (f k t : ℝ) (h1 : t = f * (k - 32)) (h2 : t = 50) (h3 : k = 122) : f = 5 / 9 :=
by
  sorry

end fraction_is_five_over_nine_l39_39092


namespace solve_for_x_l39_39480

theorem solve_for_x (x : ℤ) : (16 : ℝ) ^ (3 * x - 5) = ((1 : ℝ) / 4) ^ (2 * x + 6) → x = -1 / 2 :=
by
  sorry

end solve_for_x_l39_39480


namespace symmetric_points_origin_a_plus_b_l39_39911

theorem symmetric_points_origin_a_plus_b (a b : ℤ) 
  (h1 : a + 3 * b = 5)
  (h2 : a + 2 * b = -3) :
  a + b = -11 :=
by
  sorry

end symmetric_points_origin_a_plus_b_l39_39911


namespace smallest_area_right_triangle_l39_39867

theorem smallest_area_right_triangle (a b : ℕ) (h₁ : a = 4) (h₂ : b = 5) : 
  ∃ c, (c = 6 ∧ ∀ (x y : ℕ) (h₃ : x = 4 ∨ y = 4) (h₄ : x = 5 ∨ y = 5), c ≤ (x * y / 2)) :=
by {
  sorry
}

end smallest_area_right_triangle_l39_39867


namespace math_problem_l39_39525

variable {a b c : ℝ}

theorem math_problem
  (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (c_nonzero : c ≠ 0)
  (h : a + b + c = -a * b * c) :
  (a^2 * b^2 / ((a^2 + b * c) * (b^2 + a * c)) +
  a^2 * c^2 / ((a^2 + b * c) * (c^2 + a * b)) +
  b^2 * c^2 / ((b^2 + a * c) * (c^2 + a * b))) = 1 :=
by
  sorry

end math_problem_l39_39525


namespace find_a_l39_39772

noncomputable def quadratic_inequality_solution (a b : ℝ) : Prop :=
  a * ((-1/2) * (1/3)) * 20 = 20 ∧
  a < 0 ∧
  (-b / (2 * a)) = (-1 / 2 + 1 / 3)

theorem find_a (a b : ℝ) (h : quadratic_inequality_solution a b) : a = -12 :=
  sorry

end find_a_l39_39772


namespace max_dinners_for_7_people_max_dinners_for_8_people_l39_39270

def max_dinners_with_new_neighbors (n : ℕ) : ℕ :=
  if n = 7 ∨ n = 8 then 3 else 0

theorem max_dinners_for_7_people : max_dinners_with_new_neighbors 7 = 3 := sorry

theorem max_dinners_for_8_people : max_dinners_with_new_neighbors 8 = 3 := sorry

end max_dinners_for_7_people_max_dinners_for_8_people_l39_39270


namespace student_correct_answers_l39_39662

theorem student_correct_answers (C I : ℕ) (h1 : C + I = 100) (h2 : C - 2 * I = 79) : C = 93 :=
by
  sorry

end student_correct_answers_l39_39662


namespace find_a_l39_39078

variable (U : Set ℝ) (A : Set ℝ) (a : ℝ)

theorem find_a (hU_def : U = {2, 3, a^2 - a - 1})
               (hA_def : A = {2, 3})
               (h_compl : U \ A = {1}) :
  a = -1 ∨ a = 2 := 
sorry

end find_a_l39_39078


namespace geometric_sequence_common_ratio_l39_39688

theorem geometric_sequence_common_ratio (a_1 q : ℝ) 
  (h1 : a_1 * q^2 = 9) 
  (h2 : a_1 * (1 + q) + 9 = 27) : 
  q = 1 ∨ q = -1/2 := 
by
  sorry

end geometric_sequence_common_ratio_l39_39688


namespace largest_n_satisfying_inequality_l39_39617

theorem largest_n_satisfying_inequality :
  ∃ n : ℕ, n ≥ 1 ∧ n^(6033) < 2011^(2011) ∧ ∀ m : ℕ, m > n → m^(6033) ≥ 2011^(2011) :=
sorry

end largest_n_satisfying_inequality_l39_39617


namespace infinite_squares_of_form_l39_39718

theorem infinite_squares_of_form (k : ℕ) (hk : k > 0) : ∃ᶠ n in at_top, ∃ m : ℕ, n * 2^k - 7 = m^2 := sorry

end infinite_squares_of_form_l39_39718


namespace shaded_area_of_square_with_quarter_circles_l39_39619

theorem shaded_area_of_square_with_quarter_circles :
  let side_len : ℝ := 12
  let square_area := side_len * side_len
  let radius := side_len / 2
  let total_circle_area := 4 * (π * radius^2 / 4)
  let shaded_area := square_area - total_circle_area
  shaded_area = 144 - 36 * π := 
by
  sorry

end shaded_area_of_square_with_quarter_circles_l39_39619


namespace sum_of_square_roots_of_consecutive_odd_numbers_l39_39185

theorem sum_of_square_roots_of_consecutive_odd_numbers :
  (Real.sqrt 1 + Real.sqrt (1 + 3) + Real.sqrt (1 + 3 + 5) + Real.sqrt (1 + 3 + 5 + 7) + Real.sqrt (1 + 3 + 5 + 7 + 9)) = 15 :=
by
  sorry

end sum_of_square_roots_of_consecutive_odd_numbers_l39_39185


namespace average_shifted_data_is_7_l39_39194

variable (x1 x2 x3 : ℝ)

theorem average_shifted_data_is_7 (h : (x1 + x2 + x3) / 3 = 5) : 
  ((x1 + 2) + (x2 + 2) + (x3 + 2)) / 3 = 7 :=
by
  sorry

end average_shifted_data_is_7_l39_39194


namespace Q_div_P_eq_10_over_3_l39_39870

noncomputable def solve_Q_over_P (P Q : ℤ) :=
  (Q / P = 10 / 3)

theorem Q_div_P_eq_10_over_3 (P Q : ℤ) (x : ℝ) :
  (∀ x, x ≠ 3 → x ≠ 4 → (P / (x + 3) + Q / (x^2 - 10 * x + 16) = (x^2 - 6 * x + 18) / (x^3 - 7 * x^2 + 14 * x - 48))) →
  solve_Q_over_P P Q :=
sorry

end Q_div_P_eq_10_over_3_l39_39870


namespace delta_ratio_l39_39590

theorem delta_ratio 
  (Δx : ℝ) (Δy : ℝ) 
  (y_new : ℝ := (1 + Δx)^2 + 1)
  (y_old : ℝ := 1^2 + 1)
  (Δy_def : Δy = y_new - y_old) :
  Δy / Δx = 2 + Δx :=
by
  sorry

end delta_ratio_l39_39590


namespace upper_left_region_l39_39482

theorem upper_left_region (t : ℝ) : (2 - 2 * t + 4 ≤ 0) → (t ≤ 3) :=
by
  sorry

end upper_left_region_l39_39482


namespace smallest_positive_integer_exists_l39_39666

theorem smallest_positive_integer_exists :
  ∃ (n : ℕ), n > 0 ∧ (∃ (k m : ℕ), n = 5 * k + 3 ∧ n = 12 * m) ∧ n = 48 :=
by
  sorry

end smallest_positive_integer_exists_l39_39666


namespace line_through_midpoint_of_ellipse_l39_39535

theorem line_through_midpoint_of_ellipse:
  (∀ x y : ℝ, (x - 4)^2 + (y - 2)^2 = (1/36) * ((9 * 4) + 36 * (1 / 4)) → (1 + 2 * (y - 2) / (x - 4) = 0)) →
  (x - 8) + 2 * (y - 4) = 0 :=
by
  sorry

end line_through_midpoint_of_ellipse_l39_39535


namespace range_of_m_l39_39844

variable {f : ℝ → ℝ}

def is_decreasing (f : ℝ → ℝ) := ∀ x y, x < y → f x > f y

theorem range_of_m (hf_dec : is_decreasing f) (hf_odd : ∀ x, f (-x) = -f x) 
  (h : ∀ m, f (m - 1) + f (2 * m - 1) > 0) : ∀ m, m < 2 / 3 :=
by
  sorry

end range_of_m_l39_39844


namespace sum_last_two_digits_l39_39763

theorem sum_last_two_digits (a b : ℕ) (ha : a = 7) (hb : b = 13) :
  (a ^ 30 + b ^ 30) % 100 = 0 := 
by
  sorry

end sum_last_two_digits_l39_39763


namespace horner_method_multiplications_additions_count_l39_39349

-- Define the polynomial function
def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 - 2 * x^2 + 4 * x - 6

-- Define the property we want to prove
theorem horner_method_multiplications_additions_count : 
  ∃ (multiplications additions : ℕ), multiplications = 4 ∧ additions = 4 := 
by
  sorry

end horner_method_multiplications_additions_count_l39_39349


namespace toll_for_18_wheel_truck_l39_39717

-- Definitions
def total_wheels : ℕ := 18
def front_axle_wheels : ℕ := 2
def rear_axle_wheels_per_axle : ℕ := 4
def toll_formula (x : ℕ) : ℝ := 0.50 + 0.50 * (x - 2)

-- Theorem statement
theorem toll_for_18_wheel_truck : 
  ∃ t : ℝ, t = 2.00 ∧
  ∃ x : ℕ, x = (1 + ((total_wheels - front_axle_wheels) / rear_axle_wheels_per_axle)) ∧
  t = toll_formula x := 
by
  -- Proof to be provided
  sorry

end toll_for_18_wheel_truck_l39_39717


namespace m_n_value_l39_39386

theorem m_n_value (m n : ℝ)
  (h1 : m * (-1/2)^2 + n * (-1/2) - 1/m < 0)
  (h2 : m * 2^2 + n * 2 - 1/m < 0)
  (h3 : m < 0)
  (h4 : (-1/2 + 2 = -n/m))
  (h5 : (-1/2) * 2 = -1/m^2) :
  m - n = -5/2 :=
sorry

end m_n_value_l39_39386


namespace lineup_count_l39_39549

theorem lineup_count (n k : ℕ) (h : n = 13) (k_eq : k = 4) : (n.choose k) = 715 := by
  sorry

end lineup_count_l39_39549


namespace animal_sighting_ratio_l39_39635

theorem animal_sighting_ratio
  (jan_sightings : ℕ)
  (feb_sightings : ℕ)
  (march_sightings : ℕ)
  (total_sightings : ℕ)
  (h1 : jan_sightings = 26)
  (h2 : feb_sightings = 3 * jan_sightings)
  (h3 : total_sightings = jan_sightings + feb_sightings + march_sightings)
  (h4 : total_sightings = 143) :
  (march_sightings : ℚ) / (feb_sightings : ℚ) = 1 / 2 :=
by
  sorry

end animal_sighting_ratio_l39_39635


namespace max_playground_area_l39_39773

/-- Mara is setting up a fence around a rectangular playground with given constraints.
    We aim to prove that the maximum area the fence can enclose is 10000 square feet. --/
theorem max_playground_area (l w : ℝ) 
  (h1 : 2 * l + 2 * w = 400) 
  (h2 : l ≥ 100) 
  (h3 : w ≥ 50) : 
  l * w ≤ 10000 :=
sorry

end max_playground_area_l39_39773


namespace perfume_price_reduction_l39_39790

theorem perfume_price_reduction : 
  let original_price := 1200
  let increased_price := original_price * (1 + 0.10)
  let final_price := increased_price * (1 - 0.15)
  original_price - final_price = 78 := 
by
  sorry

end perfume_price_reduction_l39_39790


namespace root_of_quadratic_l39_39222

theorem root_of_quadratic (x m : ℝ) (h : x = -1 ∧ x^2 + m*x - 1 = 0) : m = 0 :=
sorry

end root_of_quadratic_l39_39222


namespace distance_between_points_l39_39990

theorem distance_between_points (A B : ℝ) (hA : |A| = 2) (hB : |B| = 7) :
  |A - B| = 5 ∨ |A - B| = 9 := 
sorry

end distance_between_points_l39_39990


namespace tree_count_in_yard_l39_39012

-- Definitions from conditions
def yard_length : ℕ := 350
def tree_distance : ℕ := 14

-- Statement of the theorem
theorem tree_count_in_yard : (yard_length / tree_distance) + 1 = 26 := by
  sorry

end tree_count_in_yard_l39_39012


namespace math_proof_problem_l39_39291

theorem math_proof_problem : 
  (325 - Real.sqrt 125) / 425 = 65 - 5 := 
by sorry

end math_proof_problem_l39_39291


namespace circle_radius_l39_39583

theorem circle_radius (x y : ℝ) :
  x^2 + 2 * x + y^2 = 0 → 1 = 1 :=
by sorry

end circle_radius_l39_39583


namespace probability_of_rolling_5_is_1_over_9_l39_39117

def num_sides_dice : ℕ := 6

def favorable_combinations : List (ℕ × ℕ) :=
[(1, 4), (2, 3), (3, 2), (4, 1)]

def total_combinations : ℕ :=
num_sides_dice * num_sides_dice

def favorable_count : ℕ := favorable_combinations.length

def probability_rolling_5 : ℚ :=
favorable_count / total_combinations

theorem probability_of_rolling_5_is_1_over_9 :
  probability_rolling_5 = 1 / 9 :=
sorry

end probability_of_rolling_5_is_1_over_9_l39_39117


namespace w_janous_conjecture_l39_39076

theorem w_janous_conjecture (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (z^2 - x^2) / (x + y) + (x^2 - y^2) / (y + z) + (y^2 - z^2) / (z + x) ≥ 0 :=
by
  sorry

end w_janous_conjecture_l39_39076


namespace tan_alpha_eq_m_over_3_and_tan_alpha_plus_pi_over_4_eq_2_over_m_imp_m_l39_39509

theorem tan_alpha_eq_m_over_3_and_tan_alpha_plus_pi_over_4_eq_2_over_m_imp_m (m : ℝ) (α : ℝ)
  (h1 : Real.tan α = m / 3)
  (h2 : Real.tan (α + Real.pi / 4) = 2 / m) :
  m = -6 ∨ m = 1 :=
sorry

end tan_alpha_eq_m_over_3_and_tan_alpha_plus_pi_over_4_eq_2_over_m_imp_m_l39_39509


namespace desired_line_equation_exists_l39_39879

theorem desired_line_equation_exists :
  ∃ (a b c : ℝ), (a * 0 + b * 1 + c = 0) ∧
  (x - 3 * y + 10 = 0) ∧
  (2 * x + y - 8 = 0) ∧
  (a * x + b * y + c = 0) :=
by
  sorry

end desired_line_equation_exists_l39_39879


namespace sum_eq_expected_l39_39634

noncomputable def complex_sum : Complex :=
  12 * Complex.exp (Complex.I * 3 * Real.pi / 13) + 12 * Complex.exp (Complex.I * 6 * Real.pi / 13)

noncomputable def expected_value : Complex :=
  24 * Real.cos (Real.pi / 13) * Complex.exp (Complex.I * 9 * Real.pi / 26)

theorem sum_eq_expected :
  complex_sum = expected_value :=
by
  sorry

end sum_eq_expected_l39_39634


namespace number_of_terms_l39_39850

variable {α : Type} [LinearOrderedField α]

def sum_of_arithmetic_sequence (a₁ aₙ d : α) (n : ℕ) : α :=
  n * (a₁ + aₙ) / 2

theorem number_of_terms (a₁ aₙ : α) (d : α) (n : ℕ)
  (h₀ : 4 * (2 * a₁ + 3 * d) / 2 = 21)
  (h₁ : 4 * (2 * aₙ - 3 * d) / 2 = 67)
  (h₂ : sum_of_arithmetic_sequence a₁ aₙ d n = 286) :
  n = 26 :=
sorry

end number_of_terms_l39_39850


namespace proportion_correct_l39_39593

theorem proportion_correct (m n : ℤ) (h : 6 * m = 7 * n) (hn : n ≠ 0) : (m : ℚ) / 7 = n / 6 :=
by sorry

end proportion_correct_l39_39593


namespace engineering_student_max_marks_l39_39131

/-- 
If an engineering student has to secure 36% marks to pass, and he gets 130 marks but fails by 14 marks, 
then the maximum number of marks is 400.
-/
theorem engineering_student_max_marks (M : ℝ) (passing_percentage : ℝ) (marks_obtained : ℝ) (marks_failed_by : ℝ) (pass_marks : ℝ) :
  passing_percentage = 0.36 →
  marks_obtained = 130 →
  marks_failed_by = 14 →
  pass_marks = marks_obtained + marks_failed_by →
  pass_marks = passing_percentage * M →
  M = 400 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end engineering_student_max_marks_l39_39131


namespace find_number_of_partners_l39_39714

noncomputable def law_firm_partners (P A : ℕ) : Prop :=
  (P / A = 3 / 97) ∧ (P / (A + 130) = 1 / 58)

theorem find_number_of_partners (P A : ℕ) (h : law_firm_partners P A) : P = 5 :=
  sorry

end find_number_of_partners_l39_39714


namespace intersection_points_count_l39_39866

theorem intersection_points_count : 
  ∃ (x1 y1 x2 y2 : ℝ), 
  (x1 - ⌊x1⌋)^2 + (y1 - 1)^2 = x1 - ⌊x1⌋ ∧ 
  y1 = 1/5 * x1 + 1 ∧ 
  (x2 - ⌊x2⌋)^2 + (y2 - 1)^2 = x2 - ⌊x2⌋ ∧ 
  y2 = 1/5 * x2 + 1 ∧ 
  (x1, y1) ≠ (x2, y2) :=
sorry

end intersection_points_count_l39_39866


namespace trigonometric_identity_l39_39294

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 3) :
  (Real.sin (3 * Real.pi / 2 + θ) + 2 * Real.cos (Real.pi - θ)) /
  (Real.sin (Real.pi / 2 - θ) - Real.sin (Real.pi - θ)) = -3 / 2 :=
by
  sorry

end trigonometric_identity_l39_39294


namespace dot_product_result_l39_39992

open Real

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (-1, 2)

def scale_vec (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def add_vec (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem dot_product_result :
  dot_product (add_vec (scale_vec 2 a) b) a = 6 :=
by
  sorry

end dot_product_result_l39_39992


namespace round_table_legs_l39_39276

theorem round_table_legs:
  ∀ (chairs tables disposed chairs_legs tables_legs : ℕ) (total_legs : ℕ),
  chairs = 80 →
  chairs_legs = 5 →
  tables = 20 →
  disposed = 40 * chairs / 100 →
  total_legs = 300 →
  total_legs - (chairs - disposed) * chairs_legs = tables * tables_legs →
  tables_legs = 3 :=
by 
  intros chairs tables disposed chairs_legs tables_legs total_legs
  sorry

end round_table_legs_l39_39276


namespace family_reunion_kids_l39_39691

theorem family_reunion_kids (adults : ℕ) (tables : ℕ) (people_per_table : ℕ) 
  (h_adults : adults = 123) (h_tables : tables = 14) 
  (h_people_per_table : people_per_table = 12) :
  (tables * people_per_table - adults) = 45 :=
by
  sorry

end family_reunion_kids_l39_39691


namespace train_lengths_equal_l39_39520

theorem train_lengths_equal (v_fast v_slow : ℝ) (t : ℝ) (L : ℝ)  
  (h1 : v_fast = 46) 
  (h2 : v_slow = 36) 
  (h3 : t = 36.00001) : 
  2 * L = (v_fast - v_slow) / 3600 * t → L = 1800.0005 := 
by
  sorry

end train_lengths_equal_l39_39520


namespace simplify_expression_l39_39039

theorem simplify_expression (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -3) :
  (3 * x^2 + 2 * x) / ((x - 1) * (x + 3)) - (5 * x + 3) / ((x - 1) * (x + 3))
  = 3 * (x^2 - x - 1) / ((x - 1) * (x + 3)) :=
by
  sorry

end simplify_expression_l39_39039


namespace radius_of_given_circle_is_eight_l39_39575

noncomputable def radius_of_circle (diameter : ℝ) : ℝ := diameter / 2

theorem radius_of_given_circle_is_eight :
  radius_of_circle 16 = 8 :=
by
  sorry

end radius_of_given_circle_is_eight_l39_39575


namespace bill_earnings_l39_39859

theorem bill_earnings
  (milk_total : ℕ)
  (fraction : ℚ)
  (milk_to_butter_ratio : ℕ)
  (milk_to_sour_cream_ratio : ℕ)
  (butter_price_per_gallon : ℚ)
  (sour_cream_price_per_gallon : ℚ)
  (whole_milk_price_per_gallon : ℚ)
  (milk_for_butter : ℚ)
  (milk_for_sour_cream : ℚ)
  (remaining_milk : ℚ)
  (total_earnings : ℚ) :
  milk_total = 16 →
  fraction = 1/4 →
  milk_to_butter_ratio = 4 →
  milk_to_sour_cream_ratio = 2 →
  butter_price_per_gallon = 5 →
  sour_cream_price_per_gallon = 6 →
  whole_milk_price_per_gallon = 3 →
  milk_for_butter = milk_total * fraction / milk_to_butter_ratio →
  milk_for_sour_cream = milk_total * fraction / milk_to_sour_cream_ratio →
  remaining_milk = milk_total - 2 * (milk_total * fraction) →
  total_earnings = (remaining_milk * whole_milk_price_per_gallon) + 
                   (milk_for_sour_cream * sour_cream_price_per_gallon) + 
                   (milk_for_butter * butter_price_per_gallon) →
  total_earnings = 41 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end bill_earnings_l39_39859


namespace arithmetic_seq_75th_term_difference_l39_39684

theorem arithmetic_seq_75th_term_difference :
  ∃ (d : ℝ), 300 * (50 + d) = 15000 ∧ -30 / 299 ≤ d ∧ d ≤ 30 / 299 ∧
  let L := 50 - 225 * (30 / 299)
  let G := 50 + 225 * (30 / 299)
  G - L = 13500 / 299 := by
sorry

end arithmetic_seq_75th_term_difference_l39_39684


namespace sticky_strips_used_l39_39636

theorem sticky_strips_used 
  (total_decorations : ℕ) 
  (nails_used : ℕ) 
  (decorations_hung_with_nails_fraction : ℚ) 
  (decorations_hung_with_thumbtacks_fraction : ℚ) 
  (nails_used_eq : nails_used = 50)
  (decorations_hung_with_nails_fraction_eq : decorations_hung_with_nails_fraction = 2/3)
  (decorations_hung_with_thumbtacks_fraction_eq : decorations_hung_with_thumbtacks_fraction = 2/5)
  (total_decorations_eq : total_decorations = nails_used / decorations_hung_with_nails_fraction)
  : (total_decorations - nails_used - decorations_hung_with_thumbtacks_fraction * (total_decorations - nails_used)) = 15 := 
by {
  sorry
}

end sticky_strips_used_l39_39636


namespace train_crossing_time_l39_39680

-- Definitions for conditions
def train_length : ℝ := 100 -- train length in meters
def train_speed_kmh : ℝ := 90 -- train speed in km/hr
def train_speed_mps : ℝ := 25 -- train speed in m/s after conversion

-- Lean 4 statement to prove the time taken for the train to cross the electric pole is 4 seconds
theorem train_crossing_time : (train_length / train_speed_mps) = 4 := by
  sorry

end train_crossing_time_l39_39680


namespace joe_cut_kids_hair_l39_39476

theorem joe_cut_kids_hair
  (time_women minutes_women count_women : ℕ)
  (time_men minutes_men count_men : ℕ)
  (time_kid minutes_kid : ℕ)
  (total_minutes: ℕ) : 
  minutes_women = 50 → 
  minutes_men = 15 →
  minutes_kid = 25 →
  count_women = 3 →
  count_men = 2 →
  total_minutes = 255 →
  (count_women * minutes_women + count_men * minutes_men + time_kid * minutes_kid) = total_minutes →
  time_kid = 3 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  -- Proof is not provided, hence stating sorry.
  sorry

end joe_cut_kids_hair_l39_39476


namespace base_triangle_not_equilateral_l39_39375

-- Define the lengths of the lateral edges
def SA := 1
def SB := 2
def SC := 4

-- Main theorem: the base triangle is not equilateral
theorem base_triangle_not_equilateral 
  (a : ℝ)
  (equilateral : a = a)
  (triangle_inequality1 : SA + SB > a)
  (triangle_inequality2 : SA + a > SC) : 
  a ≠ a :=
by 
  sorry

end base_triangle_not_equilateral_l39_39375


namespace trig_inequality_l39_39868

noncomputable def a : ℝ := Real.sin (31 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (58 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (32 * Real.pi / 180)

theorem trig_inequality : c > b ∧ b > a := by
  sorry

end trig_inequality_l39_39868


namespace least_non_lucky_multiple_of_7_correct_l39_39808

def is_lucky (n : ℕ) : Prop :=
  n % (n.digits 10).sum = 0

def least_non_lucky_multiple_of_7 : ℕ :=
  14

theorem least_non_lucky_multiple_of_7_correct : 
  ¬ is_lucky 14 ∧ ∀ m, m < 14 → m % 7 = 0 → ¬ ¬ is_lucky m :=
by
  sorry

end least_non_lucky_multiple_of_7_correct_l39_39808


namespace garden_area_increase_l39_39309

theorem garden_area_increase : 
  let length_old := 60
  let width_old := 20
  let perimeter := 2 * (length_old + width_old)
  let side_new := perimeter / 4
  let area_old := length_old * width_old
  let area_new := side_new * side_new
  area_new - area_old = 400 :=
by
  sorry

end garden_area_increase_l39_39309


namespace binomial_coefficient_multiple_of_4_l39_39274

theorem binomial_coefficient_multiple_of_4 :
  ∃ (S : Finset ℕ), (∀ k ∈ S, 0 ≤ k ∧ k ≤ 2014 ∧ (Nat.choose 2014 k) % 4 = 0) ∧ S.card = 991 :=
sorry

end binomial_coefficient_multiple_of_4_l39_39274


namespace number_subtracted_l39_39161

-- Define the variables x and y
variable (x y : ℝ)

-- Define the conditions
def condition1 := 6 * x - y = 102
def condition2 := x = 40

-- Define the theorem to prove
theorem number_subtracted (h1 : condition1 x y) (h2 : condition2 x) : y = 138 :=
sorry

end number_subtracted_l39_39161


namespace reinforcement_left_after_days_l39_39687

theorem reinforcement_left_after_days
  (initial_men : ℕ) (initial_days : ℕ) (remaining_days : ℕ) (men_left : ℕ)
  (remaining_men : ℕ) (x : ℕ) :
  initial_men = 400 ∧
  initial_days = 31 ∧
  remaining_days = 8 ∧
  men_left = initial_men - remaining_men ∧
  remaining_men = 200 ∧
  400 * 31 - 400 * x = 200 * 8 →
  x = 27 :=
by
  intros h
  sorry

end reinforcement_left_after_days_l39_39687


namespace value_of_A_cos_alpha_plus_beta_l39_39601

noncomputable def f (A x : ℝ) : ℝ := A * Real.cos (x / 4 + Real.pi / 6)

theorem value_of_A {A : ℝ}
  (h1 : f A (Real.pi / 3) = Real.sqrt 2) :
  A = 2 := 
by
  sorry

theorem cos_alpha_plus_beta {α β : ℝ}
  (hαβ1 : 0 ≤ α ∧ α ≤ Real.pi / 2)
  (hαβ2 : 0 ≤ β ∧ β ≤ Real.pi / 2)
  (h2 : f 2 (4*α + 4*Real.pi/3) = -30 / 17)
  (h3 : f 2 (4*β - 2*Real.pi/3) = 8 / 5) :
  Real.cos (α + β) = -13 / 85 :=
by
  sorry

end value_of_A_cos_alpha_plus_beta_l39_39601


namespace pascal_family_min_children_l39_39855

-- We define the conditions b >= 3 and g >= 2
def b_condition (b : ℕ) : Prop := b >= 3
def g_condition (g : ℕ) : Prop := g >= 2

-- We state that the smallest number of children given these conditions is 5
theorem pascal_family_min_children (b g : ℕ) (hb : b_condition b) (hg : g_condition g) : b + g = 5 :=
sorry

end pascal_family_min_children_l39_39855


namespace no_prime_for_equation_l39_39360

theorem no_prime_for_equation (x k : ℕ) (p : ℕ) (h_prime : p.Prime) (h_eq : x^5 + 2 * x + 3 = p^k) : False := 
sorry

end no_prime_for_equation_l39_39360


namespace hakeem_artichoke_dip_l39_39869

theorem hakeem_artichoke_dip 
(total_money : ℝ)
(cost_per_artichoke : ℝ)
(artichokes_per_dip : ℕ)
(dip_per_three_artichokes : ℕ)
(h : total_money = 15)
(h₁ : cost_per_artichoke = 1.25)
(h₂ : artichokes_per_dip = 3)
(h₃ : dip_per_three_artichokes = 5) : 
total_money / cost_per_artichoke * (dip_per_three_artichokes / artichokes_per_dip) = 20 := 
sorry

end hakeem_artichoke_dip_l39_39869


namespace part_a_part_b_part_c_l39_39978

def op (a b : ℕ) : ℕ := a ^ b + b ^ a

theorem part_a (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : op a b = op b a :=
by
  dsimp [op]
  rw [add_comm]

theorem part_b (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : ¬ (op a (op b c) = op (op a b) c) :=
by
  -- example counter: a = 2, b = 2, c = 2 
  -- 2 ^ (2^2 + 2^2) + (2^2 + 2^2) ^ 2 ≠ (2^2 + 2 ^ 2) ^ 2 + 8 ^ 2
  sorry

theorem part_c (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : ¬ (op (op a b) (op b c) = op (op b a) (op c b)) :=
by
  -- example counter: a = 2, b = 3, c = 2 
  -- This will involve specific calculations showing the inequality.
  sorry

end part_a_part_b_part_c_l39_39978


namespace typist_original_salary_l39_39959

theorem typist_original_salary (x : ℝ) (h : (x * 1.10 * 0.95 = 4180)) : x = 4000 :=
by sorry

end typist_original_salary_l39_39959


namespace new_boarders_day_scholars_ratio_l39_39253

theorem new_boarders_day_scholars_ratio
  (initial_boarders : ℕ)
  (initial_day_scholars : ℕ)
  (ratio_boarders_day_scholars : ℕ → ℕ → Prop)
  (additional_boarders : ℕ)
  (new_boarders : ℕ)
  (new_ratio : ℕ → ℕ → Prop)
  (r1 r2 : ℕ)
  (h1 : ratio_boarders_day_scholars 7 16)
  (h2 : initial_boarders = 560)
  (h3 : initial_day_scholars = 1280)
  (h4 : additional_boarders = 80)
  (h5 : new_boarders = initial_boarders + additional_boarders)
  (h6 : new_ratio new_boarders initial_day_scholars) :
  new_ratio r1 r2 → r1 = 1 ∧ r2 = 2 :=
by {
    sorry
}

end new_boarders_day_scholars_ratio_l39_39253


namespace minimum_of_a_plus_b_l39_39657

theorem minimum_of_a_plus_b {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 4/b = 1) : a + b ≥ 9 :=
by sorry

end minimum_of_a_plus_b_l39_39657


namespace max_value_x_div_y_l39_39033

variables {x y a b : ℝ}

theorem max_value_x_div_y (h1 : x ≥ y) (h2 : y > 0) (h3 : 0 ≤ a) (h4 : a ≤ x) (h5 : 0 ≤ b) (h6 : b ≤ y) 
  (h7 : (x - a)^2 + (y - b)^2 = x^2 + b^2) (h8 : x^2 + b^2 = y^2 + a^2) :
  x / y ≤ (2 * Real.sqrt 3) / 3 :=
sorry

end max_value_x_div_y_l39_39033


namespace share_of_B_l39_39797

noncomputable def B_share (B_investment A_investment C_investment D_investment total_profit : ℝ) : ℝ :=
  (B_investment / (A_investment + B_investment + C_investment + D_investment)) * total_profit

theorem share_of_B (B_investment total_profit : ℝ) (hA : A_investment = 3 * B_investment) 
  (hC : C_investment = (3 / 2) * B_investment) 
  (hD : D_investment = (3 / 2) * B_investment) 
  (h_profit : total_profit = 19900) :
  B_share B_investment A_investment C_investment D_investment total_profit = 2842.86 :=
by
  rw [B_share, hA, hC, hD, h_profit]
  sorry

end share_of_B_l39_39797


namespace tammy_weekly_distance_l39_39822

-- Define the conditions.
def track_length : ℕ := 50
def loops_per_day : ℕ := 10
def days_in_week : ℕ := 7

-- Using the conditions, prove the total distance per week is 3500 meters.
theorem tammy_weekly_distance : (track_length * loops_per_day * days_in_week) = 3500 := by
  sorry

end tammy_weekly_distance_l39_39822


namespace bananas_to_oranges_equivalence_l39_39252

theorem bananas_to_oranges_equivalence :
  (3 / 4 : ℚ) * 16 = 12 ->
  (2 / 5 : ℚ) * 10 = 4 :=
by
  intros h
  sorry

end bananas_to_oranges_equivalence_l39_39252


namespace cylinder_height_in_hemisphere_l39_39968

theorem cylinder_height_in_hemisphere 
  (R : ℝ) (r : ℝ) (h : ℝ) 
  (hemisphere_radius : R = 7) 
  (cylinder_radius : r = 3) 
  (height_eq : h = 2 * Real.sqrt 10) : 
  R^2 = h^2 + r^2 :=
by
  have : h = 2 * Real.sqrt 10 := height_eq
  have : R = 7 := hemisphere_radius
  have : r = 3 := cylinder_radius
  sorry

end cylinder_height_in_hemisphere_l39_39968


namespace race_length_l39_39725

noncomputable def solve_race_length (a b c d : ℝ) : Prop :=
  (d > 0) →
  (d / a = (d - 40) / b) →
  (d / b = (d - 30) / c) →
  (d / a = (d - 65) / c) →
  d = 240

theorem race_length : ∃ (d : ℝ), solve_race_length a b c d :=
by
  use 240
  sorry

end race_length_l39_39725


namespace range_of_a_l39_39429

theorem range_of_a (a : ℝ) (h : a > 0) : (∀ x : ℝ, x > 0 → 9 * x + a^2 / x ≥ a^2 + 8) → 2 ≤ a ∧ a ≤ 4 :=
by
  intros h1
  sorry

end range_of_a_l39_39429


namespace text_messages_relationship_l39_39955

theorem text_messages_relationship (l x : ℕ) (h_l : l = 111) (h_combined : l + x = 283) : x = l + 61 :=
by sorry

end text_messages_relationship_l39_39955


namespace parabolas_intersect_at_points_l39_39057

theorem parabolas_intersect_at_points :
  ∃ (x y : ℝ), (y = 3 * x^2 - 5 * x + 1 ∧ y = 4 * x^2 + 3 * x + 1) ↔ ((x = 0 ∧ y = 1) ∨ (x = -8 ∧ y = 233)) := 
sorry

end parabolas_intersect_at_points_l39_39057


namespace original_price_after_discount_l39_39716

theorem original_price_after_discount (a x : ℝ) (h : 0.7 * x = a) : x = (10 / 7) * a := 
sorry

end original_price_after_discount_l39_39716


namespace cost_of_first_book_l39_39944

-- Define the initial amount of money Shelby had.
def initial_amount : ℕ := 20

-- Define the cost of the second book.
def cost_of_second_book : ℕ := 4

-- Define the cost of one poster.
def cost_of_poster : ℕ := 4

-- Define the number of posters bought.
def num_posters : ℕ := 2

-- Define the total cost that Shelby had to spend on posters.
def total_cost_of_posters : ℕ := num_posters * cost_of_poster

-- Define the total amount spent on books and posters.
def total_spent (X : ℕ) : ℕ := X + cost_of_second_book + total_cost_of_posters

-- Prove that the cost of the first book is 8 dollars.
theorem cost_of_first_book (X : ℕ) (h : total_spent X = initial_amount) : X = 8 :=
by
  sorry

end cost_of_first_book_l39_39944


namespace central_angle_of_sector_l39_39633

theorem central_angle_of_sector (r l θ : ℝ) 
  (h1 : 2 * r + l = 8) 
  (h2 : (1 / 2) * l * r = 4) 
  (h3 : θ = l / r) : θ = 2 := 
sorry

end central_angle_of_sector_l39_39633


namespace seashells_at_end_of_month_l39_39756

-- Given conditions as definitions
def initial_seashells : ℕ := 50
def increase_per_week : ℕ := 20

-- Define function to calculate seashells in the nth week
def seashells_in_week (n : ℕ) : ℕ :=
  initial_seashells + n * increase_per_week

-- Lean statement to prove the number of seashells in the jar at the end of four weeks is 130
theorem seashells_at_end_of_month : seashells_in_week 4 = 130 :=
by
  sorry

end seashells_at_end_of_month_l39_39756


namespace abs_neg_2023_l39_39612

theorem abs_neg_2023 : abs (-2023) = 2023 := 
by
  sorry

end abs_neg_2023_l39_39612


namespace coefficient_of_y_l39_39561

theorem coefficient_of_y (x y a : ℝ) (h1 : 7 * x + y = 19) (h2 : x + a * y = 1) (h3 : 2 * x + y = 5) : a = 3 :=
sorry

end coefficient_of_y_l39_39561


namespace middle_circle_radius_l39_39432

theorem middle_circle_radius 
  (r1 r3 : ℝ) 
  (geometric_sequence: ∃ r2 : ℝ, r2 ^ 2 = r1 * r3) 
  (r1_val : r1 = 5) 
  (r3_val : r3 = 20) 
  : ∃ r2 : ℝ, r2 = 10 := 
by
  sorry

end middle_circle_radius_l39_39432


namespace range_of_function_l39_39904

theorem range_of_function :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → 2 ≤ x^2 - 2 * x + 3 ∧ x^2 - 2 * x + 3 ≤ 6) :=
by {
  sorry
}

end range_of_function_l39_39904


namespace no_3_digit_even_sum_27_l39_39040

/-- Predicate for a 3-digit number -/
def is_3_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- Predicate for an even number -/
def is_even (n : ℕ) : Prop := n % 2 = 0

/-- Function to compute the digit sum of a number -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- Theorem: There are no 3-digit numbers with a digit sum of 27 that are even -/
theorem no_3_digit_even_sum_27 : 
  ∀ n : ℕ, is_3_digit n → digit_sum n = 27 → is_even n → false :=
by
  sorry

end no_3_digit_even_sum_27_l39_39040


namespace find_integer_solutions_l39_39103

theorem find_integer_solutions (x y : ℤ) :
  8 * x^2 * y^2 + x^2 + y^2 = 10 * x * y ↔
  (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) := 
by 
  sorry

end find_integer_solutions_l39_39103


namespace percentage_of_students_receiving_certificates_l39_39015

theorem percentage_of_students_receiving_certificates
  (boys girls : ℕ)
  (pct_boys pct_girls : ℕ)
  (h_boys : boys = 30)
  (h_girls : girls = 20)
  (h_pct_boys : pct_boys = 30)
  (h_pct_girls : pct_girls = 40)
  :
  (pct_boys * boys + pct_girls * girls) / (100 * (boys + girls)) * 100 = 34 :=
by
  sorry

end percentage_of_students_receiving_certificates_l39_39015


namespace books_count_l39_39566

theorem books_count (Tim_books Total_books Mike_books : ℕ) (h1 : Tim_books = 22) (h2 : Total_books = 42) : Mike_books = 20 :=
by
  sorry

end books_count_l39_39566


namespace algebraic_expression_eq_five_l39_39727

theorem algebraic_expression_eq_five (a b : ℝ)
  (h₁ : a^2 - a = 1)
  (h₂ : b^2 - b = 1) :
  3 * a^2 + 2 * b^2 - 3 * a - 2 * b = 5 :=
by
  sorry

end algebraic_expression_eq_five_l39_39727


namespace thirteen_pow_2023_mod_1000_l39_39355

theorem thirteen_pow_2023_mod_1000 :
  (13^2023) % 1000 = 99 :=
sorry

end thirteen_pow_2023_mod_1000_l39_39355


namespace geometric_sequence_y_value_l39_39878

theorem geometric_sequence_y_value (q : ℝ) 
  (h1 : 2 * q^4 = 18) 
  : 2 * (q^2) = 6 := by
  sorry

end geometric_sequence_y_value_l39_39878


namespace quadratic_inequality_solution_l39_39204

theorem quadratic_inequality_solution 
  (a b : ℝ) 
  (h1 : (∀ x : ℝ, x^2 + a * x + b > 0 → (x < 3 ∨ x > 1))) :
  ∀ x : ℝ, a * x + b < 0 → x > 3 / 4 := 
by 
  sorry

end quadratic_inequality_solution_l39_39204


namespace min_brilliant_triple_product_l39_39239

theorem min_brilliant_triple_product :
  ∃ a b c : ℕ, a > b ∧ b > c ∧ Prime a ∧ Prime b ∧ Prime c ∧ (a = b + 2 * c) ∧ (∃ k : ℕ, (a + b + c) = k^2) ∧ (a * b * c = 35651) :=
by
  sorry

end min_brilliant_triple_product_l39_39239


namespace fifth_friend_paid_40_l39_39914

-- Defining the conditions given in the problem
variables {a b c d e : ℝ}
variables (h1 : a = (1/3) * (b + c + d + e))
variables (h2 : b = (1/4) * (a + c + d + e))
variables (h3 : c = (1/5) * (a + b + d + e))
variables (h4 : d = (1/6) * (a + b + c + e))
variables (h5 : a + b + c + d + e = 120)

-- Proving that the amount paid by the fifth friend is $40
theorem fifth_friend_paid_40 : e = 40 :=
by
  sorry  -- Proof to be provided

end fifth_friend_paid_40_l39_39914


namespace parabola_passes_through_point_l39_39404

theorem parabola_passes_through_point {x y : ℝ} (h_eq : y = (1/2) * x^2 - 2) :
  (x = 2 ∧ y = 0) :=
by
  sorry

end parabola_passes_through_point_l39_39404


namespace paint_usage_correct_l39_39902

-- Define the parameters representing paint usage and number of paintings
def largeCanvasPaint : Nat := 3
def smallCanvasPaint : Nat := 2
def largePaintings : Nat := 3
def smallPaintings : Nat := 4

-- Define the total paint used
def totalPaintUsed : Nat := largeCanvasPaint * largePaintings + smallCanvasPaint * smallPaintings

-- Prove that total paint used is 17 ounces
theorem paint_usage_correct : totalPaintUsed = 17 :=
  by
    sorry

end paint_usage_correct_l39_39902


namespace shift_line_down_4_units_l39_39089

theorem shift_line_down_4_units :
  ∀ (x : ℝ), y = - (3 / 4) * x → (y - 4 = - (3 / 4) * x - 4) := by
  sorry

end shift_line_down_4_units_l39_39089


namespace courtyard_width_l39_39260

def width_of_courtyard (w : ℝ) : Prop :=
  28 * 100 * 100 * w = 13788 * 22 * 12

theorem courtyard_width :
  ∃ w : ℝ, width_of_courtyard w ∧ abs (w - 13.012) < 0.001 :=
by
  sorry

end courtyard_width_l39_39260


namespace geometric_sequence_sum_l39_39952

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) 
  (h_pos : ∀ n, 0 < a n) 
  (h_a1 : a 1 = 3)
  (h_sum_first_three : a 1 + a 2 + a 3 = 21) :
  a 4 + a 5 + a 6 = 168 := 
sorry

end geometric_sequence_sum_l39_39952


namespace average_last_three_l39_39499

theorem average_last_three (a b c d e f g : ℝ) 
  (h1 : (a + b + c + d + e + f + g) / 7 = 65) 
  (h2 : (a + b + c + d) / 4 = 60) : 
  (e + f + g) / 3 = 71.67 :=
by
  sorry

end average_last_three_l39_39499


namespace pyramid_surface_area_and_volume_l39_39585

def s := 8
def PF := 15

noncomputable def FM := s / 2
noncomputable def PM := Real.sqrt (PF^2 + FM^2)
noncomputable def baseArea := s^2
noncomputable def lateralAreaTriangle := (1 / 2) * s * PM
noncomputable def totalSurfaceArea := baseArea + 4 * lateralAreaTriangle
noncomputable def volume := (1 / 3) * baseArea * PF

theorem pyramid_surface_area_and_volume :
  totalSurfaceArea = 64 + 16 * Real.sqrt 241 ∧
  volume = 320 :=
by
  sorry

end pyramid_surface_area_and_volume_l39_39585


namespace remainder_2519_div_6_l39_39048

theorem remainder_2519_div_6 : ∃ q r, 2519 = 6 * q + r ∧ 0 ≤ r ∧ r < 6 ∧ r = 5 := 
by
  sorry

end remainder_2519_div_6_l39_39048


namespace inequality_a6_b6_l39_39332

theorem inequality_a6_b6 (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 
  a^6 + b^6 ≥ ab * (a^4 + b^4) :=
sorry

end inequality_a6_b6_l39_39332


namespace exchange_candies_l39_39627

-- Define the problem conditions and calculate the required values
def chocolates := 7
def caramels := 9
def exchange := 5

-- Combinatorial function to calculate binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem exchange_candies (h1 : chocolates = 7) (h2 : caramels = 9) (h3 : exchange = 5) :
  binomial chocolates exchange * binomial caramels exchange = 2646 := by
  sorry

end exchange_candies_l39_39627


namespace greatest_divisor_form_p_plus_1_l39_39594

theorem greatest_divisor_form_p_plus_1 (n : ℕ) (hn : 0 < n):
  (∀ p : ℕ, Nat.Prime p → p % 3 = 2 → ¬ (p ∣ n) → 6 ∣ (p + 1)) ∧
  (∀ d : ℕ, (∀ p : ℕ, Nat.Prime p → p % 3 = 2 → ¬ (p ∣ n) → d ∣ (p + 1)) → d ≤ 6) :=
by {
  sorry
}

end greatest_divisor_form_p_plus_1_l39_39594


namespace correct_value_l39_39705

theorem correct_value (x : ℚ) (h : x + 7/5 = 81/20) : x - 7/5 = 5/4 :=
sorry

end correct_value_l39_39705


namespace number_of_terms_before_4_appears_l39_39778

-- Define the parameters of the arithmetic sequence
def first_term : ℤ := 100
def common_difference : ℤ := -4
def nth_term (n : ℕ) : ℤ := first_term + common_difference * (n - 1)

-- Problem: Prove that the number of terms before the number 4 appears in this sequence is 24.
theorem number_of_terms_before_4_appears :
  ∃ n : ℕ, nth_term n = 4 ∧ n - 1 = 24 := 
by
  sorry

end number_of_terms_before_4_appears_l39_39778


namespace least_positive_integer_l39_39288

theorem least_positive_integer (n : ℕ) : 
  (530 + n) % 4 = 0 → n = 2 :=
by {
  sorry
}

end least_positive_integer_l39_39288


namespace peter_walks_more_time_l39_39481

-- Define the total distance Peter has to walk
def total_distance : ℝ := 2.5

-- Define the distance Peter has already walked
def distance_walked : ℝ := 1.0

-- Define Peter's walking pace in minutes per mile
def walking_pace : ℝ := 20.0

-- Prove that Peter has to walk 30 more minutes to reach the grocery store
theorem peter_walks_more_time : walking_pace * (total_distance - distance_walked) = 30 :=
by
  sorry

end peter_walks_more_time_l39_39481


namespace bin101_to_decimal_l39_39427

-- Define the binary representation of 101 (base 2)
def bin101 : ℕ := 1 * 2^2 + 0 * 2^1 + 1 * 2^0

-- State the theorem that asserts the decimal value of 101 (base 2) is 5
theorem bin101_to_decimal : bin101 = 5 := by
  sorry

end bin101_to_decimal_l39_39427


namespace digit_appears_in_3n_l39_39246

-- Define a function to check if a digit is in a number
def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ k : ℕ, n / 10^k % 10 = d

-- Define the statement that n does not contain the digits 1, 2, or 9
def does_not_contain_1_2_9 (n : ℕ) : Prop :=
  ¬ (contains_digit n 1 ∨ contains_digit n 2 ∨ contains_digit n 9)

theorem digit_appears_in_3n (n : ℕ) (hn : 1 ≤ n) (h : does_not_contain_1_2_9 n) :
  contains_digit (3 * n) 1 ∨ contains_digit (3 * n) 2 ∨ contains_digit (3 * n) 9 :=
by
  sorry

end digit_appears_in_3n_l39_39246


namespace cos_seven_pi_over_four_proof_l39_39749

def cos_seven_pi_over_four : Prop := (Real.cos (7 * Real.pi / 4) = 1 / Real.sqrt 2)

theorem cos_seven_pi_over_four_proof : cos_seven_pi_over_four :=
by
  sorry

end cos_seven_pi_over_four_proof_l39_39749


namespace remainder_division_l39_39387

def polynomial (x : ℝ) : ℝ := x^4 - 4 * x^2 + 7 * x - 8

theorem remainder_division : polynomial 3 = 58 :=
by
  sorry

end remainder_division_l39_39387


namespace minimize_expression_l39_39362

theorem minimize_expression (x y : ℝ) (k : ℝ) (h : k = -1) : (xy + k)^2 + (x - y)^2 ≥ 0 ∧ (∀ x y : ℝ, (xy + k)^2 + (x - y)^2 = 0 ↔ k = -1) := 
by {
  sorry
}

end minimize_expression_l39_39362


namespace sequence_property_l39_39477

theorem sequence_property (x : ℝ) (a : ℕ → ℝ) (h : ∀ n, a n = 1 + x ^ (n + 1) + x ^ (n + 2)) (h_given : (a 2) ^ 2 = (a 1) * (a 3)) :
  ∀ n ≥ 3, (a n) ^ 2 = (a (n - 1)) * (a (n + 1)) :=
by
  intros n hn
  sorry

end sequence_property_l39_39477


namespace combined_payment_is_correct_l39_39638

-- Define the conditions for discounts
def discount_scheme (amount : ℕ) : ℕ :=
  if amount ≤ 100 then amount
  else if amount ≤ 300 then (amount * 90) / 100
  else (amount * 80) / 100

-- Given conditions for Wang Bo's purchases
def first_purchase := 80
def second_purchase_with_discount_applied := 252

-- Two possible original amounts for the second purchase
def possible_second_purchases : Set ℕ :=
  { x | discount_scheme x = second_purchase_with_discount_applied }

-- Total amount to be considered for combined buys with discounts
def total_amount_paid := {x + first_purchase | x ∈ possible_second_purchases}

-- discount applied on the combined amount
def discount_applied_amount (combined : ℕ) : ℕ :=
  discount_scheme combined

-- Prove the combined amount is either 288 or 316
theorem combined_payment_is_correct :
  ∃ combined ∈ total_amount_paid, discount_applied_amount combined = 288 ∨ discount_applied_amount combined = 316 :=
sorry

end combined_payment_is_correct_l39_39638


namespace farmer_land_area_l39_39620

theorem farmer_land_area
  (A : ℝ)
  (h1 : A / 3 + A / 4 + A / 5 + 26 = A) : A = 120 :=
sorry

end farmer_land_area_l39_39620


namespace inf_geometric_mean_gt_3_inf_geometric_mean_le_2_l39_39120

variables {x y g : ℝ}
variables (hx : 0 < x) (hy : 0 < y)
variable (hg : g = Real.sqrt (x * y))

theorem inf_geometric_mean_gt_3 :
  g ≥ 3 → (1 / Real.sqrt (1 + x) + 1 / Real.sqrt (1 + y) ≥ 2 / Real.sqrt (1 + g)) :=
by
  sorry

theorem inf_geometric_mean_le_2 :
  g ≤ 2 → (1 / Real.sqrt (1 + x) + 1 / Real.sqrt (1 + y) ≤ 2 / Real.sqrt (1 + g)) :=
by
  sorry

end inf_geometric_mean_gt_3_inf_geometric_mean_le_2_l39_39120


namespace total_money_raised_l39_39622

def tickets_sold : ℕ := 25
def ticket_price : ℕ := 2
def num_15_donations : ℕ := 2
def donation_15_amount : ℕ := 15
def donation_20_amount : ℕ := 20

theorem total_money_raised : 
  tickets_sold * ticket_price + num_15_donations * donation_15_amount + donation_20_amount = 100 := 
by sorry

end total_money_raised_l39_39622


namespace expected_worth_of_coin_flip_l39_39621

theorem expected_worth_of_coin_flip :
  let p_heads := 2 / 3
  let p_tails := 1 / 3
  let gain_heads := 5
  let loss_tails := -9
  (p_heads * gain_heads) + (p_tails * loss_tails) = 1 / 3 :=
by
  -- Proof will be here
  sorry

end expected_worth_of_coin_flip_l39_39621


namespace intersection_points_l39_39742

noncomputable def y1 := 2*((7 + Real.sqrt 61)/2)^2 - 3*((7 + Real.sqrt 61)/2) + 1
noncomputable def y2 := 2*((7 - Real.sqrt 61)/2)^2 - 3*((7 - Real.sqrt 61)/2) + 1

theorem intersection_points :
  ∃ (x y : ℝ), (y = 2*x^2 - 3*x + 1) ∧ (y = x^2 + 4*x + 4) ∧
                ((x = (7 + Real.sqrt 61)/2 ∧ y = y1) ∨
                 (x = (7 - Real.sqrt 61)/2 ∧ y = y2)) :=
by
  sorry

end intersection_points_l39_39742


namespace value_of_leftover_coins_l39_39507

def quarters_per_roll : ℕ := 30
def dimes_per_roll : ℕ := 40

def ana_quarters : ℕ := 95
def ana_dimes : ℕ := 183

def ben_quarters : ℕ := 104
def ben_dimes : ℕ := 219

def leftover_quarters : ℕ := (ana_quarters + ben_quarters) % quarters_per_roll
def leftover_dimes : ℕ := (ana_dimes + ben_dimes) % dimes_per_roll

def dollar_value (quarters dimes : ℕ) : ℝ := quarters * 0.25 + dimes * 0.10

theorem value_of_leftover_coins : 
  dollar_value leftover_quarters leftover_dimes = 6.95 := 
  sorry

end value_of_leftover_coins_l39_39507


namespace maximize_a_minus_b_plus_c_l39_39096

noncomputable def f (a b c x : ℝ) : ℝ := a * Real.cos x + b * Real.cos (2 * x) + c * Real.cos (3 * x)

theorem maximize_a_minus_b_plus_c
  {a b c : ℝ}
  (h : ∀ x : ℝ, f a b c x ≥ -1) :
  a - b + c ≤ 1 :=
sorry

end maximize_a_minus_b_plus_c_l39_39096


namespace remainder_product_mod_five_l39_39258

-- Define the conditions as congruences
def num1 : ℕ := 14452
def num2 : ℕ := 15652
def num3 : ℕ := 16781

-- State the main theorem using the conditions and the given problem
theorem remainder_product_mod_five : 
  (num1 % 5 = 2) → 
  (num2 % 5 = 2) → 
  (num3 % 5 = 1) → 
  ((num1 * num2 * num3) % 5 = 4) :=
by
  intros
  sorry

end remainder_product_mod_five_l39_39258


namespace Heracles_age_l39_39512

variable (A H : ℕ)

theorem Heracles_age :
  (A = H + 7) →
  (A + 3 = 2 * H) →
  H = 10 :=
by
  sorry

end Heracles_age_l39_39512


namespace last_digit_of_a2009_div_a2006_is_6_l39_39426
open Nat

def ratio_difference_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 2) * a n = (a (n + 1)) ^ 2 + d * a (n + 1)

theorem last_digit_of_a2009_div_a2006_is_6
  (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 1)
  (h3 : a 3 = 2)
  (d : ℕ)
  (h4 : ratio_difference_sequence a d) :
  (a 2009 / a 2006) % 10 = 6 :=
by
  sorry

end last_digit_of_a2009_div_a2006_is_6_l39_39426


namespace fraction_transform_l39_39976

theorem fraction_transform {x : ℤ} :
  (537 - x : ℚ) / (463 + x) = 1 / 9 ↔ x = 437 := by
sorry

end fraction_transform_l39_39976


namespace convert_444_quinary_to_octal_l39_39207

def quinary_to_decimal (n : ℕ) : ℕ :=
  let d2 := (n / 100) * 25
  let d1 := ((n % 100) / 10) * 5
  let d0 := (n % 10)
  d2 + d1 + d0

def decimal_to_octal (n : ℕ) : ℕ :=
  let r2 := (n / 64)
  let n2 := (n % 64)
  let r1 := (n2 / 8)
  let r0 := (n2 % 8)
  r2 * 100 + r1 * 10 + r0

theorem convert_444_quinary_to_octal :
  decimal_to_octal (quinary_to_decimal 444) = 174 := by
  sorry

end convert_444_quinary_to_octal_l39_39207


namespace necessary_but_not_sufficient_l39_39862

theorem necessary_but_not_sufficient {a b c d : ℝ} (hcd : c > d) : 
  (a - c > b - d) → (a > b) ∧ ¬((a > b) → (a - c > b - d)) :=
by
  sorry

end necessary_but_not_sufficient_l39_39862


namespace total_amount_shared_l39_39023

theorem total_amount_shared (a b c : ℝ)
  (h1 : a = 1/3 * (b + c))
  (h2 : b = 2/7 * (a + c))
  (h3 : a = b + 20) : 
  a + b + c = 720 :=
by
  sorry

end total_amount_shared_l39_39023


namespace alyssa_kittens_l39_39176

theorem alyssa_kittens (original_kittens given_away: ℕ) (h1: original_kittens = 8) (h2: given_away = 4) :
  original_kittens - given_away = 4 :=
by
  sorry

end alyssa_kittens_l39_39176


namespace true_proposition_among_options_l39_39265

theorem true_proposition_among_options :
  (∀ (x y : ℝ), (x > |y|) → (x > y)) ∧
  (¬ (∀ (x : ℝ), (x > 1) → (x^2 > 1))) ∧
  (¬ (∀ (x : ℤ), (x = 1) → (x^2 + x - 2 = 0))) ∧
  (¬ (∀ (x : ℝ), (x^2 > 0) → (x > 1))) :=
by
  sorry

end true_proposition_among_options_l39_39265


namespace part_one_extreme_value_part_two_max_k_l39_39272

noncomputable def f (x : ℝ) (k : ℝ) : ℝ :=
  x * Real.log x - k * (x - 1)

theorem part_one_extreme_value :
  ∃ x : ℝ, x > 0 ∧ ∀ y > 0, f y 1 ≥ f x 1 ∧ f x 1 = 0 := 
  sorry

theorem part_two_max_k :
  ∀ x : ℝ, ∃ k : ℕ, (1 < x) -> (f x (k:ℝ) + x > 0) ∧ k = 3 :=
  sorry

end part_one_extreme_value_part_two_max_k_l39_39272


namespace thursday_loaves_baked_l39_39741

theorem thursday_loaves_baked (wednesday friday saturday sunday monday : ℕ) (p1 : wednesday = 5) (p2 : friday = 10) (p3 : saturday = 14) (p4 : sunday = 19) (p5 : monday = 25) : 
  ∃ thursday : ℕ, thursday = 11 := 
by 
  sorry

end thursday_loaves_baked_l39_39741


namespace problem1_l39_39410

theorem problem1 (x : ℝ) (hx : x > 0) : (x + 1/x = 2) ↔ (x = 1) :=
by
  sorry

end problem1_l39_39410


namespace new_average_marks_l39_39637

theorem new_average_marks
  (orig_avg : ℕ) (num_papers : ℕ)
  (add_geography : ℕ) (add_history : ℕ)
  (H_orig_avg : orig_avg = 63)
  (H_num_papers : num_papers = 11)
  (H_add_geography : add_geography = 20)
  (H_add_history : add_history = 2) :
  (orig_avg * num_ppapers + add_geography + add_history) / num_papers = 65 :=
by
  -- Here would be the proof steps
  sorry

end new_average_marks_l39_39637


namespace fixed_monthly_charge_l39_39121

-- Given conditions
variable (F C_J : ℕ)
axiom january_bill : F + C_J = 46
axiom february_bill : F + 2 * C_J = 76

-- Proof problem
theorem fixed_monthly_charge : F = 16 :=
by
  sorry

end fixed_monthly_charge_l39_39121


namespace both_fifth_and_ninth_terms_are_20_l39_39609

def sequence_a (n : ℕ) : ℕ := n^2 - 14 * n + 65

theorem both_fifth_and_ninth_terms_are_20 : sequence_a 5 = 20 ∧ sequence_a 9 = 20 := 
by
  sorry

end both_fifth_and_ninth_terms_are_20_l39_39609


namespace problem_inequality_sol1_problem_inequality_sol2_l39_39016

def f (x a : ℝ) : ℝ := x^2 - 2 * a * x - (2 * a + 2)

theorem problem_inequality_sol1 (a x : ℝ) :
  (a > -3 / 2 ∧ (x > 2 * a + 2 ∨ x < -1)) ∨
  (a = -3 / 2 ∧ x ≠ -1) ∨
  (a < -3 / 2 ∧ (x > -1 ∨ x < 2 * a + 2)) ↔
  f x a > x :=
sorry

theorem problem_inequality_sol2 (a : ℝ) :
  (∀ x : ℝ, x > -1 → f x a + 3 ≥ 0) ↔
  a ≤ Real.sqrt 2 - 1 :=
sorry

end problem_inequality_sol1_problem_inequality_sol2_l39_39016


namespace infinite_alternating_parity_l39_39913

theorem infinite_alternating_parity (m : ℕ) : ∃ᶠ n in at_top, 
  ∀ i < m, ((5^n / 10^i) % 2) ≠ (((5^n / 10^(i+1)) % 10) % 2) :=
sorry

end infinite_alternating_parity_l39_39913


namespace greatest_integer_x_l39_39339

theorem greatest_integer_x (x : ℤ) : 
  (∃ n : ℤ, (x^2 + 4*x + 10) = n * (x - 4)) → x ≤ 46 := 
by
  sorry

end greatest_integer_x_l39_39339


namespace parallel_lines_slope_l39_39086

theorem parallel_lines_slope (a : ℝ) :
  (∀ x y : ℝ, ax + 3 * y + 1 = 0 → 2 * x + (a + 1) * y + 1 = 0) →
  a = -3 :=
by
  sorry

end parallel_lines_slope_l39_39086


namespace total_cost_one_each_l39_39453

theorem total_cost_one_each (x y z : ℝ)
  (h1 : 3 * x + 7 * y + z = 6.3)
  (h2 : 4 * x + 10 * y + z = 8.4) :
  x + y + z = 2.1 :=
  sorry

end total_cost_one_each_l39_39453


namespace average_age_of_team_is_23_l39_39463

noncomputable def average_age_team (A : ℝ) : Prop :=
  let captain_age := 27
  let wicket_keeper_age := 28
  let team_size := 11
  let remaining_players := team_size - 2
  let remaining_average_age := A - 1
  11 * A = 55 + 9 * (A - 1)

theorem average_age_of_team_is_23 : average_age_team 23 := by
  sorry

end average_age_of_team_is_23_l39_39463


namespace cookies_per_person_l39_39985

variable (x y z : ℕ)
variable (h_pos_z : z ≠ 0) -- Ensure z is not zero to avoid division by zero

theorem cookies_per_person (h_cookies : x * y / z = 35) : 35 / 5 = 7 := by
  sorry

end cookies_per_person_l39_39985


namespace apple_count_l39_39805

/--
Given the total number of apples in several bags, where each bag contains either 12 or 6 apples,
and knowing that the total number of apples is between 70 and 80 inclusive,
prove that the total number of apples can only be 72 or 78.
-/
theorem apple_count (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : ∃ k : ℕ, n = 6 * k) : n = 72 ∨ n = 78 :=
by {
  sorry
}

end apple_count_l39_39805


namespace no_common_perfect_squares_l39_39970

theorem no_common_perfect_squares (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ¬ (∃ m n : ℕ, a^2 + 4 * b = m^2 ∧ b^2 + 4 * a = n^2) :=
by
  sorry

end no_common_perfect_squares_l39_39970


namespace at_least_one_not_less_than_two_l39_39267

theorem at_least_one_not_less_than_two
  (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) :
  (2 ≤ (y / x + y / z)) ∨ (2 ≤ (z / x + z / y)) ∨ (2 ≤ (x / z + x / y)) :=
sorry

end at_least_one_not_less_than_two_l39_39267


namespace inverse_proportion_function_has_m_value_l39_39915

theorem inverse_proportion_function_has_m_value
  (k : ℝ)
  (h1 : 2 * -3 = k)
  {m : ℝ}
  (h2 : 6 = k / m) :
  m = -1 :=
by
  sorry

end inverse_proportion_function_has_m_value_l39_39915


namespace percentage_difference_l39_39443

variable (x y : ℝ)
variable (hxy : x = 6 * y)

theorem percentage_difference : ((x - y) / x) * 100 = 83.33 := by
  sorry

end percentage_difference_l39_39443


namespace minimum_x_value_l39_39565

theorem minimum_x_value
  (sales_jan_may june_sales x : ℝ)
  (h_sales_jan_may : sales_jan_may = 38.6)
  (h_june_sales : june_sales = 5)
  (h_total_sales_condition : sales_jan_may + june_sales + 2 * june_sales * (1 + x / 100) + 2 * june_sales * (1 + x / 100)^2 ≥ 70) :
  x = 20 := by
  sorry

end minimum_x_value_l39_39565


namespace mod_multiplication_example_l39_39214

theorem mod_multiplication_example :
  (98 % 75) * (202 % 75) % 75 = 71 :=
by
  have h1 : 98 % 75 = 23 := by sorry
  have h2 : 202 % 75 = 52 := by sorry
  have h3 : 1196 % 75 = 71 := by sorry
  exact h3

end mod_multiplication_example_l39_39214


namespace min_value_l39_39837

theorem min_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 3) : 
  ∃ c : ℝ, (c = 3 / 4) ∧ (∀ (a b c : ℝ), a = x ∧ b = y ∧ c = z → 
    (1/(a + 3*b) + 1/(b + 3*c) + 1/(c + 3*a)) ≥ c) :=
sorry

end min_value_l39_39837


namespace shorter_side_ratio_l39_39137

variable {x y : ℝ}
variables (h1 : x < y)
variables (h2 : x + y - Real.sqrt (x^2 + y^2) = 1/2 * y)

theorem shorter_side_ratio (h1 : x < y) (h2 : x + y - Real.sqrt (x^2 + y^2) = 1 / 2 * y) : x / y = 3 / 4 := 
sorry

end shorter_side_ratio_l39_39137


namespace elephant_entry_duration_l39_39865

theorem elephant_entry_duration
  (initial_elephants : ℕ)
  (exodus_duration : ℕ)
  (leaving_rate : ℕ)
  (entering_rate : ℕ)
  (final_elephants : ℕ)
  (h_initial : initial_elephants = 30000)
  (h_exodus_duration : exodus_duration = 4)
  (h_leaving_rate : leaving_rate = 2880)
  (h_entering_rate : entering_rate = 1500)
  (h_final : final_elephants = 28980) :
  (final_elephants - (initial_elephants - (exodus_duration * leaving_rate))) / entering_rate = 7 :=
by
  sorry

end elephant_entry_duration_l39_39865


namespace eval_expression_l39_39530

theorem eval_expression : (825 * 825) - (824 * 826) = 1 := by
  sorry

end eval_expression_l39_39530


namespace original_polynomial_l39_39630

theorem original_polynomial {x y : ℝ} (P : ℝ) :
  P - (-x^2 * y) = 3 * x^2 * y - 2 * x * y - 1 → P = 2 * x^2 * y - 2 * x * y - 1 :=
sorry

end original_polynomial_l39_39630


namespace trig_identity_l39_39266

theorem trig_identity (α : ℝ) (h : Real.tan α = 4) : 
  (1 + Real.cos (2 * α) + 8 * Real.sin α ^ 2) / Real.sin (2 * α) = 65 / 4 :=
by
  sorry

end trig_identity_l39_39266


namespace x_intercept_of_line_is_six_l39_39883

theorem x_intercept_of_line_is_six : ∃ x : ℝ, (∃ y : ℝ, y = 0) ∧ (2*x - 4*y = 12) ∧ x = 6 :=
by {
  sorry
}

end x_intercept_of_line_is_six_l39_39883


namespace find_angle_x_l39_39313

theorem find_angle_x (x : ℝ) (α : ℝ) (β : ℝ) (γ : ℝ)
  (h₁ : α = 45)
  (h₂ : β = 3 * x)
  (h₃ : γ = x)
  (h₄ : α + β + γ = 180) :
  x = 33.75 :=
sorry

end find_angle_x_l39_39313


namespace solution_of_abs_eq_l39_39655

theorem solution_of_abs_eq (x : ℝ) : |x - 5| = 3 * x + 6 → x = -1 / 4 :=
by
  sorry

end solution_of_abs_eq_l39_39655


namespace initial_puppies_correct_l39_39418

def initial_puppies (total_puppies_after: ℝ) (bought_puppies: ℝ) : ℝ :=
  total_puppies_after - bought_puppies

theorem initial_puppies_correct : initial_puppies (4.2 * 5.0) 3.0 = 18.0 := by
  sorry

end initial_puppies_correct_l39_39418


namespace function_has_local_minimum_at_zero_l39_39301

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2 * x + 2) / (2 * (x - 1))

def is_local_minimum (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y, abs (y - x) < ε → f x ≤ f y

theorem function_has_local_minimum_at_zero :
  -4 < 0 ∧ 0 < 1 ∧ is_local_minimum f 0 := 
sorry

end function_has_local_minimum_at_zero_l39_39301


namespace magician_trick_success_l39_39341

theorem magician_trick_success {n : ℕ} (T_pos : ℕ) (deck_size : ℕ := 52) (discard_count : ℕ := 51):
  (T_pos = 1 ∨ T_pos = deck_size) → ∃ strategy : Type, ∀ spectator_choice : ℕ, (spectator_choice ≤ deck_size) → 
                          ((T_pos = 1 → ∃ k, k ≤ deck_size ∧ k ≠ T_pos ∧ discard_count - k = deck_size - 1)
                          ∧ (T_pos = deck_size → ∃ k, k ≤ deck_size ∧ k ≠ T_pos ∧ discard_count - k = deck_size - 1)) :=
sorry

end magician_trick_success_l39_39341


namespace correct_answer_l39_39746

def P : Set ℝ := {1, 2, 3}
def Q : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}

theorem correct_answer : P ∩ Q ⊆ P := by
  sorry

end correct_answer_l39_39746


namespace common_divisors_4n_7n_l39_39472

theorem common_divisors_4n_7n (n : ℕ) (h1 : n < 50) 
    (h2 : (Nat.gcd (4 * n + 5) (7 * n + 6) > 1)) :
    n = 7 ∨ n = 18 ∨ n = 29 ∨ n = 40 := 
  sorry

end common_divisors_4n_7n_l39_39472


namespace lucille_house_difference_l39_39006

def height_lucille : ℕ := 80
def height_neighbor1 : ℕ := 70
def height_neighbor2 : ℕ := 99

def average_height (h1 h2 h3 : ℕ) : ℕ := (h1 + h2 + h3) / 3

def difference (h_average h_actual : ℕ) : ℕ := h_average - h_actual

theorem lucille_house_difference :
  difference (average_height height_lucille height_neighbor1 height_neighbor2) height_lucille = 3 :=
by
  unfold difference
  unfold average_height
  sorry

end lucille_house_difference_l39_39006


namespace triangle_inequality_l39_39036

noncomputable def area_triangle (a b c : ℝ) : ℝ := sorry -- Definition of area, but implementation is not required.

theorem triangle_inequality (a b c : ℝ) (S_triangle : ℝ):
  1 - (8 * ((a - b) ^ 2 + (b - c) ^ 2 + (c - a) ^ 2) / (a + b + c) ^ 2)
  ≤ 432 * S_triangle ^ 2 / (a + b + c) ^ 4
  ∧ 432 * S_triangle ^ 2 / (a + b + c) ^ 4
  ≤ 1 - (2 * ((a - b) ^ 2 + (b - c) ^ 2 + (c - a) ^ 2) / (a + b + c) ^ 2) :=
sorry -- Proof is omitted

end triangle_inequality_l39_39036


namespace trajectory_eq_of_moving_point_Q_l39_39661

-- Define the conditions and the correct answer
theorem trajectory_eq_of_moving_point_Q 
(a b : ℝ) (h : a > b) (h_pos : b > 0)
(P Q : ℝ × ℝ)
(h_ellipse : (P.1^2) / (a^2) + (P.2^2) / (b^2) = 1)
(h_Q : Q = (P.1 * 2, P.2 * 2)) :
  (Q.1^2) / (4 * a^2) + (Q.2^2) / (4 * b^2) = 1 :=
by 
  sorry

end trajectory_eq_of_moving_point_Q_l39_39661


namespace equilateral_triangle_area_l39_39490

theorem equilateral_triangle_area (perimeter : ℝ) (h1 : perimeter = 120) :
  ∃ A : ℝ, A = 400 * Real.sqrt 3 ∧
    (∃ s : ℝ, s = perimeter / 3 ∧ A = (Real.sqrt 3 / 4) * (s ^ 2)) :=
by
  sorry

end equilateral_triangle_area_l39_39490


namespace range_of_m_l39_39409

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |2009 * x + 1| ≥ |m - 1| - 2) → -1 ≤ m ∧ m ≤ 3 :=
by
  intro h
  sorry

end range_of_m_l39_39409


namespace sum_first_15_terms_l39_39700

-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

-- Define the sum of the first n terms of the sequence
def sum_first_n_terms (a d : ℤ) (n : ℕ) : ℤ := n * (2 * a + (n - 1) * d) / 2

-- Conditions
def a_7 := 1
def a_9 := 5

-- Prove that S_15 = 45
theorem sum_first_15_terms : 
  ∃ (a d : ℤ), 
    (arithmetic_sequence a d 7 = a_7) ∧ 
    (arithmetic_sequence a d 9 = a_9) ∧ 
    (sum_first_n_terms a d 15 = 45) :=
sorry

end sum_first_15_terms_l39_39700


namespace minutes_watched_on_Thursday_l39_39969

theorem minutes_watched_on_Thursday 
  (n_total : ℕ) (n_Mon : ℕ) (n_Tue : ℕ) (n_Wed : ℕ) (n_Fri : ℕ) (n_weekend : ℕ)
  (h_total : n_total = 352)
  (h_Mon : n_Mon = 138)
  (h_Tue : n_Tue = 0)
  (h_Wed : n_Wed = 0)
  (h_Fri : n_Fri = 88)
  (h_weekend : n_weekend = 105) :
  n_total - (n_Mon + n_Tue + n_Wed + n_Fri + n_weekend) = 21 := by
  sorry

end minutes_watched_on_Thursday_l39_39969


namespace original_hourly_wage_l39_39983

theorem original_hourly_wage 
  (daily_wage_increase : ∀ W : ℝ, 1.60 * W + 10 = 45)
  (work_hours : ℝ := 8) : 
  ∃ W_hourly : ℝ, W_hourly = 2.73 :=
by 
  have W : ℝ := (45 - 10) / 1.60 
  have W_hourly : ℝ := W / work_hours
  use W_hourly 
  sorry

end original_hourly_wage_l39_39983


namespace cost_of_second_batch_l39_39783

theorem cost_of_second_batch
  (C_1 C_2 : ℕ)
  (quantity_ratio cost_increase: ℕ) 
  (H1 : C_1 = 3000) 
  (H2 : C_2 = 9600) 
  (H3 : quantity_ratio = 3) 
  (H4 : cost_increase = 1)
  : (∃ x : ℕ, C_1 / x = C_2 / (x + cost_increase) / quantity_ratio) ∧ 
    (C_2 / (C_1 / 15 + cost_increase) / 3 = 16) :=
by
  sorry

end cost_of_second_batch_l39_39783


namespace faye_initial_books_l39_39440

theorem faye_initial_books (X : ℕ) (h : (X - 3) + 48 = 79) : X = 34 :=
sorry

end faye_initial_books_l39_39440


namespace find_C_plus_D_l39_39518

theorem find_C_plus_D (C D : ℝ) (h : ∀ x : ℝ, (Cx - 20) / (x^2 - 3 * x - 10) = D / (x + 2) + 4 / (x - 5)) :
  C + D = 4.7 :=
sorry

end find_C_plus_D_l39_39518


namespace centered_hexagonal_seq_l39_39044

def is_centered_hexagonal (a : ℕ) : Prop :=
  ∃ n : ℕ, a = 3 * n^2 - 3 * n + 1

def are_sequences (a b c d : ℕ) : Prop :=
  (b = 2 * a - 1) ∧ (d = c^2) ∧ (a + b = c + d)

theorem centered_hexagonal_seq (a : ℕ) :
  (∃ b c d, are_sequences a b c d) ↔ is_centered_hexagonal a :=
sorry

end centered_hexagonal_seq_l39_39044


namespace James_selling_percentage_l39_39505

def James_selling_percentage_proof : Prop :=
  ∀ (total_cost original_price return_cost extra_item bought_price out_of_pocket sold_amount : ℝ),
    total_cost = 3000 →
    return_cost = 700 + 500 →
    extra_item = 500 * 1.2 →
    bought_price = 100 →
    out_of_pocket = 2020 →
    sold_amount = out_of_pocket - (total_cost - return_cost + bought_price) →
    sold_amount / extra_item * 100 = 20

theorem James_selling_percentage : James_selling_percentage_proof :=
by
  sorry

end James_selling_percentage_l39_39505


namespace find_g_l39_39964

-- Define given functions and terms
def f1 (x : ℝ) := 7 * x^4 - 4 * x^3 + 2 * x - 5
def f2 (x : ℝ) := 5 * x^3 - 3 * x^2 + 4 * x - 1
def g (x : ℝ) := -7 * x^4 + 9 * x^3 - 3 * x^2 + 2 * x + 4

-- Theorem to prove that g(x) satisfies the given condition
theorem find_g : ∀ x : ℝ, f1 x + g x = f2 x :=
by 
  -- Alternatively: Proof is required here
  sorry

end find_g_l39_39964


namespace ratio_of_sequence_l39_39001

variables (a b c : ℝ)

-- Condition 1: arithmetic sequence
def arithmetic_sequence : Prop := 2 * b = a + c

-- Condition 2: geometric sequence
def geometric_sequence : Prop := c^2 = a * b

-- Theorem stating the ratio of a:b:c
theorem ratio_of_sequence (h1 : arithmetic_sequence a b c) (h2 : geometric_sequence a b c) : 
  (a = 4 * b) ∧ (c = -2 * b) :=
sorry

end ratio_of_sequence_l39_39001


namespace smallest_integer_a_l39_39757

theorem smallest_integer_a (a : ℤ) (b : ℤ) (h1 : a < 21) (h2 : 20 ≤ b) (h3 : b < 31) (h4 : (a : ℝ) / b < 2 / 3) : 13 < a :=
sorry

end smallest_integer_a_l39_39757


namespace part_a_part_b_l39_39894

/-- Two equally skilled chess players with p = 0.5, q = 0.5. -/
def p : ℝ := 0.5
def q : ℝ := 0.5

-- Definition for binomial coefficient
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

-- Binomial distribution
def P (n k : ℕ) : ℝ := (binomial_coeff n k) * (p^k) * (q^(n-k))

/-- Prove that the probability of winning one out of two games is greater than the probability of winning two out of four games -/
theorem part_a : (P 2 1) > (P 4 2) := sorry

/-- Prove that the probability of winning at least two out of four games is greater than the probability of winning at least three out of five games -/
theorem part_b : (P 4 2 + P 4 3 + P 4 4) > (P 5 3 + P 5 4 + P 5 5) := sorry

end part_a_part_b_l39_39894


namespace find_g_zero_l39_39079

noncomputable def g (x : ℝ) : ℝ := sorry  -- fourth-degree polynomial

-- Conditions
axiom cond1 : |g 1| = 16
axiom cond2 : |g 3| = 16
axiom cond3 : |g 4| = 16
axiom cond4 : |g 5| = 16
axiom cond5 : |g 6| = 16
axiom cond6 : |g 7| = 16

-- statement to prove
theorem find_g_zero : |g 0| = 54 := 
by sorry

end find_g_zero_l39_39079


namespace while_loop_output_correct_do_while_loop_output_correct_l39_39356

def while_loop (a : ℕ) (i : ℕ) : List (ℕ × ℕ) :=
  (List.range (7 - i)).map (λ n => (i + n, a + n + 1))

def do_while_loop (x : ℕ) (i : ℕ) : List (ℕ × ℕ) :=
  (List.range (10 - i + 1)).map (λ n => (i + n, x + (n + 1) * 10))

theorem while_loop_output_correct : while_loop 2 1 = [(1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8)] := 
sorry

theorem do_while_loop_output_correct : do_while_loop 100 1 = [(1, 110), (2, 120), (3, 130), (4, 140), (5, 150), (6, 160), (7, 170), (8, 180), (9, 190), (10, 200)] :=
sorry

end while_loop_output_correct_do_while_loop_output_correct_l39_39356


namespace compute_g3_l39_39769

def g (x : ℤ) : ℤ := 7 * x - 3

theorem compute_g3: g (g (g 3)) = 858 :=
by
  sorry

end compute_g3_l39_39769


namespace rectangle_area_l39_39158

variable (l w : ℕ)

-- Conditions
def length_eq_four_times_width (l w : ℕ) : Prop := l = 4 * w
def perimeter_eq_200 (l w : ℕ) : Prop := 2 * l + 2 * w = 200

-- Question: Prove the area is 1600 square centimeters
theorem rectangle_area (h1 : length_eq_four_times_width l w) (h2 : perimeter_eq_200 l w) :
  l * w = 1600 := by
  sorry

end rectangle_area_l39_39158


namespace total_students_playing_one_sport_l39_39333

noncomputable def students_playing_at_least_one_sport (total_students B S Ba C B_S B_Ba B_C S_Ba C_S C_Ba B_C_S: ℕ) : ℕ :=
  B + S + Ba + C - B_S - B_Ba - B_C - S_Ba - C_S - C_Ba + B_C_S

theorem total_students_playing_one_sport : 
  students_playing_at_least_one_sport 200 50 60 35 80 10 15 20 25 30 5 10 = 130 := by
  sorry

end total_students_playing_one_sport_l39_39333


namespace vector_arithmetic_l39_39786

theorem vector_arithmetic (a b : ℝ × ℝ)
    (h₀ : a = (3, 5))
    (h₁ : b = (-2, 1)) :
    a - (2 : ℝ) • b = (7, 3) :=
sorry

end vector_arithmetic_l39_39786


namespace students_walk_fraction_l39_39052

theorem students_walk_fraction
  (school_bus_fraction : ℚ := 1/3)
  (car_fraction : ℚ := 1/5)
  (bicycle_fraction : ℚ := 1/8) :
  (1 - (school_bus_fraction + car_fraction + bicycle_fraction) = 41/120) :=
by
  sorry

end students_walk_fraction_l39_39052


namespace max_value_x_y_squared_l39_39804

theorem max_value_x_y_squared (x y : ℝ) (h : 3 * (x^3 + y^3) = x + y^2) : x + y^2 ≤ 1/3 :=
sorry

end max_value_x_y_squared_l39_39804


namespace david_more_pushups_l39_39097

theorem david_more_pushups (d z : ℕ) (h1 : d = 51) (h2 : d + z = 53) : d - z = 49 := by
  sorry

end david_more_pushups_l39_39097


namespace digits_right_of_decimal_l39_39752

theorem digits_right_of_decimal : 
  ∃ n : ℕ, (3^6 : ℚ) / ((6^4 : ℚ) * 625) = 9 * 10^(-4 : ℤ) ∧ n = 4 := 
by 
  sorry

end digits_right_of_decimal_l39_39752


namespace fraction_of_shaded_area_l39_39478

theorem fraction_of_shaded_area (total_length total_width : ℕ) (total_area : ℕ)
  (quarter_fraction half_fraction : ℚ)
  (h1 : total_length = 15) 
  (h2 : total_width = 20)
  (h3 : total_area = total_length * total_width)
  (h4 : quarter_fraction = 1 / 4)
  (h5 : half_fraction = 1 / 2) :
  (half_fraction * quarter_fraction * total_area) / total_area = 1 / 8 :=
by
  sorry

end fraction_of_shaded_area_l39_39478


namespace arrangement_is_correct_l39_39745

-- Definition of adjacency in a 3x3 matrix.
def adjacent (i j k l : Nat) : Prop := 
  (i = k ∧ j = l + 1) ∨ (i = k ∧ j = l - 1) ∨ -- horizontal adjacency
  (i = k + 1 ∧ j = l) ∨ (i = k - 1 ∧ j = l) ∨ -- vertical adjacency
  (i = k + 1 ∧ j = l + 1) ∨ (i = k - 1 ∧ j = l - 1) ∨ -- diagonal adjacency
  (i = k + 1 ∧ j = l - 1) ∨ (i = k - 1 ∧ j = l + 1)   -- diagonal adjacency

-- Definition to check if two numbers share a common divisor greater than 1.
def coprime (x y : Nat) : Prop := Nat.gcd x y = 1

-- The arrangement of the numbers in the 3x3 grid.
def grid := 
  [ [8, 9, 10],
    [5, 7, 11],
    [6, 13, 12] ]

-- Ensure adjacents numbers are coprime
def correctArrangement :=
  (coprime grid[0][0] grid[0][1]) ∧ (coprime grid[0][1] grid[0][2]) ∧
  (coprime grid[1][0] grid[1][1]) ∧ (coprime grid[1][1] grid[1][2]) ∧
  (coprime grid[2][0] grid[2][1]) ∧ (coprime grid[2][1] grid[2][2]) ∧
  (adjacent 0 0 1 1 → coprime grid[0][0] grid[1][1]) ∧
  (adjacent 0 2 1 1 → coprime grid[0][2] grid[1][1]) ∧
  (adjacent 1 0 2 1 → coprime grid[1][0] grid[2][1]) ∧
  (adjacent 1 2 2 1 → coprime grid[1][2] grid[2][1]) ∧
  (coprime grid[0][1] grid[1][0]) ∧ (coprime grid[0][1] grid[1][2]) ∧
  (coprime grid[2][1] grid[1][0]) ∧ (coprime grid[2][1] grid[1][2])

-- Statement to be proven
theorem arrangement_is_correct : correctArrangement := 
  sorry

end arrangement_is_correct_l39_39745


namespace repeating_decimal_rational_representation_l39_39587

theorem repeating_decimal_rational_representation :
  (0.12512512512512514 : ℝ) = (125 / 999 : ℝ) :=
sorry

end repeating_decimal_rational_representation_l39_39587


namespace factor_expression_l39_39819

theorem factor_expression (x : ℝ) :
  (12 * x^3 + 45 * x - 3) - (-3 * x^3 + 5 * x - 2) = 5 * x * (3 * x^2 + 8) - 1 :=
by
  sorry

end factor_expression_l39_39819


namespace math_equivalence_problem_l39_39060

theorem math_equivalence_problem :
  (2^2 + 92 * 3^2) * (4^2 + 92 * 5^2) = 1388^2 + 92 * 2^2 :=
by
  sorry

end math_equivalence_problem_l39_39060


namespace find_difference_l39_39553

theorem find_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 :=
by
  sorry

end find_difference_l39_39553


namespace total_animals_sighted_l39_39768

theorem total_animals_sighted (lions_saturday elephants_saturday buffaloes_sunday leopards_sunday rhinos_monday warthogs_monday : ℕ)
(hlions_saturday : lions_saturday = 3)
(helephants_saturday : elephants_saturday = 2)
(hbuffaloes_sunday : buffaloes_sunday = 2)
(hleopards_sunday : leopards_sunday = 5)
(hrhinos_monday : rhinos_monday = 5)
(hwarthogs_monday : warthogs_monday = 3) :
  lions_saturday + elephants_saturday + buffaloes_sunday + leopards_sunday + rhinos_monday + warthogs_monday = 20 :=
by
  -- This is where the proof will be, but we are skipping the proof here.
  sorry

end total_animals_sighted_l39_39768


namespace seats_needed_l39_39028

-- Definitions based on the problem's condition
def children : ℕ := 58
def children_per_seat : ℕ := 2

-- Theorem statement to prove
theorem seats_needed : children / children_per_seat = 29 :=
by
  sorry

end seats_needed_l39_39028


namespace lcm_of_numbers_l39_39030

-- Define the conditions given in the problem
def ratio (a b : ℕ) : Prop := 7 * b = 13 * a
def hcf_23 (a b : ℕ) : Prop := Nat.gcd a b = 23

-- Main statement to prove
theorem lcm_of_numbers (a b : ℕ) (h_ratio : ratio a b) (h_hcf : hcf_23 a b) : Nat.lcm a b = 2093 := by
  sorry

end lcm_of_numbers_l39_39030


namespace Moscow_Olympiad_1958_problem_l39_39422

theorem Moscow_Olympiad_1958_problem :
  ∀ n : ℤ, 1155 ^ 1958 + 34 ^ 1958 ≠ n ^ 2 := 
by 
  sorry

end Moscow_Olympiad_1958_problem_l39_39422


namespace age_of_youngest_child_l39_39391

theorem age_of_youngest_child (x : ℕ) 
  (h : x + (x + 4) + (x + 8) + (x + 12) + (x + 16) + (x + 20) + (x + 24) = 112) :
  x = 4 :=
sorry

end age_of_youngest_child_l39_39391


namespace miles_to_add_per_week_l39_39178

theorem miles_to_add_per_week :
  ∀ (initial_miles_per_week : ℝ) 
    (percentage_increase : ℝ) 
    (total_days : ℕ) 
    (days_in_week : ℕ),
    initial_miles_per_week = 100 →
    percentage_increase = 0.2 →
    total_days = 280 →
    days_in_week = 7 →
    ((initial_miles_per_week * (1 + percentage_increase)) - initial_miles_per_week) / (total_days / days_in_week) = 3 :=
by
  intros initial_miles_per_week percentage_increase total_days days_in_week
  intros h1 h2 h3 h4
  sorry

end miles_to_add_per_week_l39_39178


namespace total_working_days_l39_39450

variables (x a b c : ℕ)

-- Given conditions
axiom bus_morning : b + c = 6
axiom bus_afternoon : a + c = 18
axiom train_commute : a + b = 14

-- Proposition to prove
theorem total_working_days : x = a + b + c → x = 19 :=
by
  -- Placeholder for Lean's automatic proof generation
  sorry

end total_working_days_l39_39450


namespace area_of_rectangle_is_432_l39_39961

/-- Define the width of the rectangle --/
def width : ℕ := 12

/-- Define the length of the rectangle, which is three times the width --/
def length : ℕ := 3 * width

/-- The area of the rectangle is length multiplied by width --/
def area : ℕ := length * width

/-- Proof problem: the area of the rectangle is 432 square meters --/
theorem area_of_rectangle_is_432 :
  area = 432 :=
sorry

end area_of_rectangle_is_432_l39_39961


namespace prove_total_number_of_apples_l39_39417

def avg_price (light_price heavy_price : ℝ) (light_proportion heavy_proportion : ℝ) : ℝ :=
  light_proportion * light_price + heavy_proportion * heavy_price

def weighted_avg_price (prices proportions : List ℝ) : ℝ :=
  (List.map (λ ⟨p, prop⟩ => p * prop) (List.zip prices proportions)).sum

noncomputable def total_num_apples (total_earnings weighted_price : ℝ) : ℝ :=
  total_earnings / weighted_price

theorem prove_total_number_of_apples : 
  let light_proportion := 0.6
  let heavy_proportion := 0.4
  let prices := [avg_price 0.4 0.6 light_proportion heavy_proportion, 
                 avg_price 0.1 0.15 light_proportion heavy_proportion,
                 avg_price 0.25 0.35 light_proportion heavy_proportion,
                 avg_price 0.15 0.25 light_proportion heavy_proportion,
                 avg_price 0.2 0.3 light_proportion heavy_proportion,
                 avg_price 0.05 0.1 light_proportion heavy_proportion]
  let proportions := [0.4, 0.2, 0.15, 0.1, 0.1, 0.05]
  let weighted_avg := weighted_avg_price prices proportions
  total_num_apples 120 weighted_avg = 392 :=
by
  sorry

end prove_total_number_of_apples_l39_39417


namespace inequality_inequality_holds_l39_39614

theorem inequality_inequality_holds (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * z)) + (y^3 / (y^3 + 2 * z^2 * x)) + (z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_inequality_holds_l39_39614


namespace gcf_and_multiples_l39_39729

theorem gcf_and_multiples (a b gcf : ℕ) : 
  (a = 90) → (b = 135) → gcd a b = gcf → 
  (gcf = 45) ∧ (45 % gcf = 0) ∧ (90 % gcf = 0) ∧ (135 % gcf = 0) := 
by
  intros ha hb hgcf
  rw [ha, hb] at hgcf
  sorry

end gcf_and_multiples_l39_39729


namespace find_definite_integers_l39_39963

theorem find_definite_integers (n d e f : ℕ) (h₁ : n = d + Int.sqrt (e + Int.sqrt f)) 
    (h₂: ∀ x : ℝ, x = d + Int.sqrt (e + Int.sqrt f) → 
        (4 / (x - 4) + 6 / (x - 6) + 18 / (x - 18) + 20 / (x - 20) = x^2 - 12 * x - 5))
        : d + e + f = 76 :=
sorry

end find_definite_integers_l39_39963


namespace intersection_points_polar_coords_l39_39328

theorem intersection_points_polar_coords :
  (∀ (x y : ℝ), ((x - 4)^2 + (y - 5)^2 = 25 ∧ (x^2 + y^2 - 2*y = 0)) →
  (∃ (ρ θ : ℝ), ρ > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 
    ((x, y) = (ρ * Real.cos θ, ρ * Real.sin θ)) ∧
    ((ρ = 2 ∧ θ = Real.pi / 2) ∨ (ρ = Real.sqrt 2 ∧ θ = Real.pi / 4)))) :=
sorry

end intersection_points_polar_coords_l39_39328


namespace definite_integral_l39_39624

open Real

theorem definite_integral : ∫ x in (0 : ℝ)..(π / 2), (x + sin x) = π^2 / 8 + 1 :=
by
  sorry

end definite_integral_l39_39624


namespace correct_line_equation_l39_39821

theorem correct_line_equation :
  ∃ (c : ℝ), (∀ (x y : ℝ), 2 * x - 3 * y + 4 = 0 → 2 * x - 3 * y + c = 0 ∧ 2 * (-1) - 3 * 2 + c = 0) ∧ c = 8 :=
by
  use 8
  sorry

end correct_line_equation_l39_39821


namespace triangle_side_b_eq_l39_39428

   variable (a b c : Real) (A B C : Real)
   variable (cos_A sin_A : Real)
   variable (area : Real)
   variable (π : Real := Real.pi)

   theorem triangle_side_b_eq :
     cos_A = 1 / 3 →
     B = π / 6 →
     a = 4 * Real.sqrt 2 →
     sin_A = 2 * Real.sqrt 2 / 3 →
     b = (a * sin_B / sin_A) →
     b = 3 := sorry
   
end triangle_side_b_eq_l39_39428


namespace matrix_eq_l39_39205

open Matrix

def matA : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 3], ![4, 2]]
def matI : Matrix (Fin 2) (Fin 2) ℤ := 1

theorem matrix_eq (A : Matrix (Fin 2) (Fin 2) ℤ)
  (hA : A = ![![1, 3], ![4, 2]]) :
  A ^ 7 = 9936 * A ^ 2 + 12400 * 1 :=
  by
    sorry

end matrix_eq_l39_39205


namespace find_smallest_angle_l39_39738

theorem find_smallest_angle (x : ℝ) (h1 : Real.tan (2 * x) + Real.tan (3 * x) = 1) :
  x = 9 * Real.pi / 180 :=
by
  sorry

end find_smallest_angle_l39_39738


namespace questions_answered_second_half_l39_39543

theorem questions_answered_second_half :
  ∀ (q1 q2 p s : ℕ), q1 = 3 → p = 3 → s = 15 → s = (q1 + q2) * p → q2 = 2 :=
by
  intros q1 q2 p s hq1 hp hs h_final_score
  -- proofs go here, but we skip them
  sorry

end questions_answered_second_half_l39_39543


namespace number_of_yellow_highlighters_l39_39497

-- Definitions based on the given conditions
def total_highlighters : Nat := 12
def pink_highlighters : Nat := 6
def blue_highlighters : Nat := 4

-- Statement to prove the question equals the correct answer given the conditions
theorem number_of_yellow_highlighters : 
  ∃ y : Nat, y = total_highlighters - (pink_highlighters + blue_highlighters) := 
by
  -- TODO: The proof will be filled in here
  sorry

end number_of_yellow_highlighters_l39_39497


namespace amount_of_CaCO3_required_l39_39721

-- Define the balanced chemical reaction
def balanced_reaction (CaCO3 HCl CaCl2 CO2 H2O : ℕ) : Prop :=
  CaCO3 + 2 * HCl = CaCl2 + CO2 + H2O

-- Define the required conditions
def conditions (HCl_req CaCl2_req CO2_req H2O_req : ℕ) : Prop :=
  HCl_req = 4 ∧ CaCl2_req = 2 ∧ CO2_req = 2 ∧ H2O_req = 2

-- The main theorem to be proved
theorem amount_of_CaCO3_required :
  ∃ (CaCO3_req : ℕ), conditions 4 2 2 2 ∧ balanced_reaction CaCO3_req 4 2 2 2 ∧ CaCO3_req = 2 :=
by 
  sorry

end amount_of_CaCO3_required_l39_39721


namespace solve_angle_CBO_l39_39877

theorem solve_angle_CBO 
  (BAO CAO : ℝ) (CBO ABO : ℝ) (ACO BCO : ℝ) (AOC : ℝ) 
  (h1 : BAO = CAO) 
  (h2 : CBO = ABO) 
  (h3 : ACO = BCO) 
  (h4 : AOC = 110) 
  : CBO = 20 :=
by
  sorry

end solve_angle_CBO_l39_39877


namespace find_other_number_l39_39962

theorem find_other_number 
  (a b : ℕ)
  (h_lcm : Nat.lcm a b = 5040)
  (h_gcd : Nat.gcd a b = 24)
  (h_a : a = 240) : b = 504 := by
  sorry

end find_other_number_l39_39962


namespace length_of_bridge_is_80_l39_39569

-- Define the given constants
def length_of_train : ℕ := 280
def speed_of_train : ℕ := 18
def time_to_cross : ℕ := 20

-- Define the distance traveled by the train in the given time
def distance_traveled : ℕ := speed_of_train * time_to_cross

-- Define the length of the bridge from the given distance traveled
def length_of_bridge := distance_traveled - length_of_train

-- The theorem to prove the length of the bridge is 80 meters
theorem length_of_bridge_is_80 :
  length_of_bridge = 80 := by
  sorry

end length_of_bridge_is_80_l39_39569


namespace product_of_solutions_of_x_squared_eq_49_l39_39890

theorem product_of_solutions_of_x_squared_eq_49 : 
  (∀ x, x^2 = 49 → x = 7 ∨ x = -7) → (7 * (-7) = -49) :=
by
  intros
  sorry

end product_of_solutions_of_x_squared_eq_49_l39_39890


namespace least_possible_area_l39_39733

variable (x y : ℝ) (n : ℤ)

-- Conditions
def is_integer (x : ℝ) := ∃ k : ℤ, x = k
def is_half_integer (y : ℝ) := ∃ n : ℤ, y = n + 0.5

-- Problem statement in Lean 4
theorem least_possible_area (h1 : is_integer x) (h2 : is_half_integer y)
(h3 : 2 * (x + y) = 150) : ∃ A, A = 0 :=
sorry

end least_possible_area_l39_39733


namespace freds_change_l39_39665

theorem freds_change (ticket_cost : ℝ) (num_tickets : ℕ) (borrowed_movie_cost : ℝ) (total_paid : ℝ) 
  (h_ticket_cost : ticket_cost = 5.92) 
  (h_num_tickets : num_tickets = 2) 
  (h_borrowed_movie_cost : borrowed_movie_cost = 6.79) 
  (h_total_paid : total_paid = 20) : 
  total_paid - (num_tickets * ticket_cost + borrowed_movie_cost) = 1.37 := 
by 
  sorry

end freds_change_l39_39665


namespace max_students_divide_equal_pen_pencil_l39_39954

theorem max_students_divide_equal_pen_pencil : Nat.gcd 2500 1575 = 25 := 
by
  sorry

end max_students_divide_equal_pen_pencil_l39_39954


namespace range_of_a_l39_39940

variable (a x : ℝ)

theorem range_of_a (h : x - 5 = -3 * a) (hx_neg : x < 0) : a > 5 / 3 :=
by {
  sorry
}

end range_of_a_l39_39940


namespace minuend_calculation_l39_39730

theorem minuend_calculation (subtrahend difference : ℕ) (h : subtrahend + difference + 300 = 600) :
  300 = 300 :=
sorry

end minuend_calculation_l39_39730


namespace find_original_six_digit_number_l39_39939

theorem find_original_six_digit_number (N x y : ℕ) (h1 : N = 10 * x + y) (h2 : N - x = 654321) (h3 : 0 ≤ y ∧ y ≤ 9) :
  N = 727023 :=
sorry

end find_original_six_digit_number_l39_39939


namespace combined_population_correct_l39_39974

theorem combined_population_correct (W PP LH N : ℕ) 
  (hW : W = 900)
  (hPP : PP = 7 * W)
  (hLH : LH = 2 * W + 600)
  (hN : N = 3 * (PP - W)) :
  PP + LH + N = 24900 :=
by
  sorry

end combined_population_correct_l39_39974


namespace shooter_scores_l39_39243

theorem shooter_scores
    (x y z : ℕ)
    (hx : x + y + z > 11)
    (hscore: 8 * x + 9 * y + 10 * z = 100) :
    (x + y + z = 12) ∧ ((x = 10 ∧ y = 0 ∧ z = 2) ∨ (x = 9 ∧ y = 2 ∧ z = 1) ∨ (x = 8 ∧ y = 4 ∧ z = 0)) :=
by
  sorry

end shooter_scores_l39_39243


namespace smallest_k_l39_39698

theorem smallest_k (k: ℕ) : k > 1 ∧ (k % 23 = 1) ∧ (k % 7 = 1) ∧ (k % 3 = 1) → k = 484 :=
sorry

end smallest_k_l39_39698


namespace correct_calculation_l39_39843

theorem correct_calculation (a : ℕ) :
  ¬ (a^3 + a^4 = a^7) ∧
  ¬ (2 * a - a = 2) ∧
  2 * a + a = 3 * a ∧
  ¬ (a^4 - a^3 = a) :=
by
  sorry

end correct_calculation_l39_39843


namespace inequality_solution_l39_39568

theorem inequality_solution (x : ℝ) : (3 * x + 4 ≥ 4 * x) ∧ (2 * (x - 1) + x > 7) ↔ (3 < x ∧ x ≤ 4) := 
by 
  sorry

end inequality_solution_l39_39568


namespace possible_denominators_count_l39_39424

variable (a b c : ℕ)
-- Conditions
def is_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9
def no_two_zeros (a b c : ℕ) : Prop := ¬(a = 0 ∧ b = 0) ∧ ¬(b = 0 ∧ c = 0) ∧ ¬(a = 0 ∧ c = 0)
def none_is_eight (a b c : ℕ) : Prop := a ≠ 8 ∧ b ≠ 8 ∧ c ≠ 8

-- Theorem
theorem possible_denominators_count : 
  is_digit a ∧ is_digit b ∧ is_digit c ∧ no_two_zeros a b c ∧ none_is_eight a b c →
  ∃ denoms : Finset ℕ, denoms.card = 7 ∧ ∀ d ∈ denoms, 999 % d = 0 :=
by
  sorry

end possible_denominators_count_l39_39424


namespace preimage_of_3_1_is_2_half_l39_39445

def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 2 * p.2, p.1 - 2 * p.2)

theorem preimage_of_3_1_is_2_half :
  (∃ x y : ℝ, f (x, y) = (3, 1) ∧ (x = 2 ∧ y = 1/2)) :=
by
  sorry

end preimage_of_3_1_is_2_half_l39_39445


namespace jellybeans_count_l39_39818

noncomputable def jellybeans_initial (y: ℝ) (n: ℕ) : ℝ :=
  y / (0.7 ^ n)

theorem jellybeans_count (y x: ℝ) (n: ℕ) (h: y = 24) (h2: n = 3) :
  x = 70 :=
by
  apply sorry

end jellybeans_count_l39_39818


namespace days_worked_prove_l39_39906

/-- Work rate of A is 1/15 work per day -/
def work_rate_A : ℚ := 1/15

/-- Work rate of B is 1/20 work per day -/
def work_rate_B : ℚ := 1/20

/-- Combined work rate of A and B -/
def combined_work_rate : ℚ := work_rate_A + work_rate_B

/-- Fraction of work left after some days -/
def fraction_work_left : ℚ := 8/15

/-- Fraction of work completed after some days -/
def fraction_work_completed : ℚ := 1 - fraction_work_left

/-- Number of days A and B worked together -/
def days_worked_together : ℚ := fraction_work_completed / combined_work_rate

theorem days_worked_prove : 
    days_worked_together = 4 := 
by 
    sorry

end days_worked_prove_l39_39906


namespace derivative_at_x₀_l39_39996

-- Define the function y = (x - 2)^2
def f (x : ℝ) : ℝ := (x - 2) ^ 2

-- Define the point of interest
def x₀ : ℝ := 1

-- State the problem and the correct answer
theorem derivative_at_x₀ : (deriv f x₀) = -2 := by
  sorry

end derivative_at_x₀_l39_39996


namespace find_a_b_l39_39882

noncomputable def log (base x : ℝ) : ℝ := Real.log x / Real.log base

theorem find_a_b (a b : ℝ) (x : ℝ) (h : 5 * (log a x) ^ 2 + 2 * (log b x) ^ 2 = (10 * (Real.log x) ^ 2 / (Real.log a * Real.log b)) + (Real.log x) ^ 2) :
  b = a ^ (2 / (5 + Real.sqrt 17)) ∨ b = a ^ (2 / (5 - Real.sqrt 17)) :=
sorry

end find_a_b_l39_39882


namespace determine_cubic_coeffs_l39_39707

-- Define the cubic function f(x)
def cubic_function (a b c x : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

-- Define the expression f(f(x) + x)
def composition_expression (a b c x : ℝ) : ℝ :=
  cubic_function a b c (cubic_function a b c x + x)

-- Given that the fraction of the compositions equals the given polynomial
def given_fraction_equals_polynomial (a b c : ℝ) : Prop :=
  ∀ x : ℝ, (composition_expression a b c x) / (cubic_function a b c x) = x^3 + 2023 * x^2 + 1776 * x + 2010

-- Prove that this implies specific values of a, b, and c
theorem determine_cubic_coeffs (a b c : ℝ) :
  given_fraction_equals_polynomial a b c →
  (a = 2022 ∧ b = 1776 ∧ c = 2010) :=
by
  sorry

end determine_cubic_coeffs_l39_39707


namespace lcm_9_16_21_eq_1008_l39_39032

theorem lcm_9_16_21_eq_1008 : Nat.lcm (Nat.lcm 9 16) 21 = 1008 := by
  sorry

end lcm_9_16_21_eq_1008_l39_39032


namespace domino_cover_grid_l39_39180

-- Definitions representing the conditions:
def isPositive (n : ℕ) : Prop := n > 0
def divides (a b : ℕ) : Prop := ∃ k, b = k * a
def canCoverWithDominos (n k : ℕ) : Prop := ∀ i j, (i < n) → (j < n) → (∃ r, i = r * k ∨ j = r * k)

-- The hypothesis: n and k are positive integers
axiom n : ℕ
axiom k : ℕ
axiom n_positive : isPositive n
axiom k_positive : isPositive k

-- The main theorem
theorem domino_cover_grid (n k : ℕ) (n_positive : isPositive n) (k_positive : isPositive k) :
  canCoverWithDominos n k ↔ divides k n := by
  sorry

end domino_cover_grid_l39_39180


namespace checkerboard_probability_l39_39216

def total_squares (n : ℕ) : ℕ :=
  n * n

def perimeter_squares (n : ℕ) : ℕ :=
  4 * n - 4

def non_perimeter_squares (n : ℕ) : ℕ :=
  total_squares n - perimeter_squares n

def probability_non_perimeter_square (n : ℕ) : ℚ :=
  non_perimeter_squares n / total_squares n

theorem checkerboard_probability :
  probability_non_perimeter_square 10 = 16 / 25 :=
by
  sorry

end checkerboard_probability_l39_39216


namespace chris_current_age_l39_39759

def praveens_age_after_10_years (P : ℝ) : ℝ := P + 10
def praveens_age_3_years_back (P : ℝ) : ℝ := P - 3

def praveens_age_condition (P : ℝ) : Prop :=
  praveens_age_after_10_years P = 3 * praveens_age_3_years_back P

def chris_age (P : ℝ) : ℝ := (P - 4) - 2

theorem chris_current_age (P : ℝ) (h₁ : praveens_age_condition P) :
  chris_age P = 3.5 :=
sorry

end chris_current_age_l39_39759


namespace circumcircle_diameter_of_triangle_l39_39406

theorem circumcircle_diameter_of_triangle (a b c : ℝ) (A B C : ℝ) 
  (h_a : a = 1) 
  (h_B : B = π/4) 
  (h_area : (1/2) * a * c * Real.sin B = 2) : 
  (2 * b = 5 * Real.sqrt 2) := 
sorry

end circumcircle_diameter_of_triangle_l39_39406


namespace broken_shells_count_l39_39950

-- Definitions from conditions
def total_perfect_shells := 17
def non_spiral_perfect_shells := 12
def extra_broken_spiral_shells := 21

-- Derived definitions
def perfect_spiral_shells : ℕ := total_perfect_shells - non_spiral_perfect_shells
def broken_spiral_shells : ℕ := perfect_spiral_shells + extra_broken_spiral_shells
def broken_shells : ℕ := 2 * broken_spiral_shells

-- The theorem to be proved
theorem broken_shells_count : broken_shells = 52 := by
  sorry

end broken_shells_count_l39_39950


namespace coefficient_of_x_l39_39045

theorem coefficient_of_x : 
  let expr := 2 * (x - 5) + 5 * (8 - 3 * x^2 + 6 * x) - 9 * (3 * x - 2)
  ∃ (a b c : ℝ), expr = a * x^2 + b * x + c ∧ b = 5 := by
    let expr := 2 * (x - 5) + 5 * (8 - 3 * x^2 + 6 * x) - 9 * (3 * x - 2)
    exact sorry

end coefficient_of_x_l39_39045


namespace square_side_length_l39_39179

theorem square_side_length (A : ℝ) (s : ℝ) (h : A = s^2) (hA : A = 144) : s = 12 :=
by 
  -- sorry is used to skip the proof
  sorry

end square_side_length_l39_39179


namespace find_x_coord_of_N_l39_39287

theorem find_x_coord_of_N
  (M N : ℝ × ℝ)
  (hM : M = (3, -5))
  (hN : N = (x, 2))
  (parallel : M.1 = N.1) :
  x = 3 :=
sorry

end find_x_coord_of_N_l39_39287


namespace find_a_l39_39298

-- Define the variables and conditions
variable (a x y : ℤ)

-- Given conditions
def x_value := (x = 2)
def y_value := (y = 1)
def equation := (a * x - y = 3)

-- The theorem to prove
theorem find_a : x_value x → y_value y → equation a x y → a = 2 :=
by
  intros
  sorry

end find_a_l39_39298


namespace tangent_parallel_coordinates_l39_39461

theorem tangent_parallel_coordinates :
  (∃ (x1 y1 x2 y2 : ℝ), 
    (y1 = x1^3 - 2) ∧ (y2 = x2^3 - 2) ∧ 
    ((3 * x1^2 = 3) ∧ (3 * x2^2 = 3)) ∧ 
    ((x1 = 1 ∧ y1 = -1) ∧ (x2 = -1 ∧ y2 = -3))) :=
sorry

end tangent_parallel_coordinates_l39_39461


namespace sum_of_three_numbers_l39_39600

theorem sum_of_three_numbers (a b c : ℝ) (h1 : a + b = 35) (h2 : b + c = 54) (h3 : c + a = 58) : 
  a + b + c = 73.5 :=
by
  sorry -- Proof is omitted

end sum_of_three_numbers_l39_39600


namespace quadratic_m_value_l39_39770

theorem quadratic_m_value (m : ℕ) :
  (∃ x : ℝ, x^(m + 1) - (m + 1) * x - 2 = 0) →
  m + 1 = 2 →
  m = 1 :=
by {
  sorry
}

end quadratic_m_value_l39_39770


namespace determinant_in_terms_of_roots_l39_39788

noncomputable def determinant_3x3 (a b c : ℝ) : ℝ :=
  (1 + a) * ((1 + b) * (1 + c) - 1) - 1 * (1 + c) + (1 + b) * 1

theorem determinant_in_terms_of_roots (a b c s p q : ℝ)
  (h1 : a + b + c = -s)
  (h2 : a * b + a * c + b * c = p)
  (h3 : a * b * c = -q) :
  determinant_3x3 a b c = -q + p - s :=
by
  sorry

end determinant_in_terms_of_roots_l39_39788


namespace system_solutions_l39_39591

theorem system_solutions (x a : ℝ) (h1 : a = -3*x^2 + 5*x - 2) (h2 : (x + 2) * a = 4 * (x^2 - 1)) (hx : x ≠ -2) :
  (x = 0 ∧ a = -2) ∨ (x = 1 ∧ a = 0) ∨ (x = -8/3 ∧ a = -110/3) :=
  sorry

end system_solutions_l39_39591


namespace range_of_a_l39_39240

theorem range_of_a (a : ℝ) : 
  (∀ (x : ℝ) (θ : ℝ), 0 ≤ θ ∧ θ ≤ π / 2 → 
    (x + 3 + 2 * (Real.sin θ) * (Real.cos θ))^2 + (x + a * (Real.sin θ) + a * (Real.cos θ))^2 ≥ 1 / 8) → 
  a ≥ 7 / 2 ∨ a ≤ Real.sqrt 6 :=
sorry

end range_of_a_l39_39240


namespace range_of_m_l39_39640

-- Definitions based on the problem conditions
def f (x : ℝ) : ℝ := x^2 - x + 1

-- Define the interval
def interval (x : ℝ) : Prop := x ≥ -1 ∧ x ≤ 2

-- Prove the range of m
theorem range_of_m (m : ℝ) : (∀ x : ℝ, interval x → f x > 2 * x + m) ↔ m < - 5 / 4 :=
by
  -- This is the theorem statement, hence the proof starts here
  sorry

end range_of_m_l39_39640


namespace inequality_abc_l39_39595

theorem inequality_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1) :
  (1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b))) ≥ 3 / 2 :=
by
  sorry

end inequality_abc_l39_39595


namespace problem_solution_l39_39889

-- Definition of the geometric sequence and the arithmetic condition
def geometric_sequence (a : ℕ → ℕ) (q : ℕ) := ∀ n, a (n + 1) = q * a n
def arithmetic_condition (a : ℕ → ℕ) := 2 * (a 3 + 1) = a 2 + a 4

-- Definitions used in the proof
def a_n (n : ℕ) : ℕ := 2^(n-1)
def b_n (n : ℕ) := a_n n + n
def S_5 := b_n 1 + b_n 2 + b_n 3 + b_n 4 + b_n 5

-- Proof statement
theorem problem_solution : 
  (∃ a : ℕ → ℕ, geometric_sequence a 2 ∧ arithmetic_condition a ∧ a 1 = 1 ∧ (∀ n, a n = 2^(n-1))) ∧
  S_5 = 46 :=
by
  sorry

end problem_solution_l39_39889


namespace find_C_l39_39184

theorem find_C (A B C : ℕ) 
  (h1 : A + B + C = 300) 
  (h2 : A + C = 200) 
  (h3 : B + C = 350) : 
  C = 250 := 
  by sorry

end find_C_l39_39184


namespace train_length_l39_39261

noncomputable def jogger_speed_kmh : ℝ := 9
noncomputable def train_speed_kmh : ℝ := 45
noncomputable def head_start : ℝ := 270
noncomputable def passing_time : ℝ := 39

noncomputable def kmh_to_ms (speed: ℝ) : ℝ := speed * (1000 / 3600)

theorem train_length (l : ℝ) 
  (v_j := kmh_to_ms jogger_speed_kmh)
  (v_t := kmh_to_ms train_speed_kmh)
  (d_h := head_start)
  (t := passing_time) :
  l = 120 :=
by 
  sorry

end train_length_l39_39261


namespace total_red_marbles_l39_39392

theorem total_red_marbles (jessica_marbles sandy_marbles alex_marbles : ℕ) (dozen : ℕ)
  (h_jessica : jessica_marbles = 3 * dozen)
  (h_sandy : sandy_marbles = 4 * jessica_marbles)
  (h_alex : alex_marbles = jessica_marbles + 2 * dozen)
  (h_dozen : dozen = 12) :
  jessica_marbles + sandy_marbles + alex_marbles = 240 :=
by
  sorry

end total_red_marbles_l39_39392


namespace molly_age_l39_39572

theorem molly_age
  (avg_age : ℕ)
  (hakimi_age : ℕ)
  (jared_age : ℕ)
  (molly_age : ℕ)
  (h1 : avg_age = 40)
  (h2 : hakimi_age = 40)
  (h3 : jared_age = hakimi_age + 10)
  (h4 : 3 * avg_age = hakimi_age + jared_age + molly_age) :
  molly_age = 30 :=
by
  sorry

end molly_age_l39_39572


namespace sufficient_but_not_necessary_condition_l39_39723

def p (x : ℝ) := x^2 + x - 2 > 0
def q (x a : ℝ) := x > a

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x, q x a → p x) ∧ (∃ x, ¬q x a ∧ p x) → a ∈ Set.Ici 1 :=
by
  sorry

end sufficient_but_not_necessary_condition_l39_39723


namespace determine_parity_of_f_l39_39826

def parity_of_f (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = 0

theorem determine_parity_of_f (f : ℝ → ℝ) :
  (∀ x y : ℝ, x + y ≠ 0 → f (x * y) = (f x + f y) / (x + y)) →
  parity_of_f f :=
sorry

end determine_parity_of_f_l39_39826


namespace number_of_non_congruent_triangles_l39_39384

theorem number_of_non_congruent_triangles :
  ∃ q : ℕ, q = 3 ∧ 
    (∀ (a b : ℕ), (a ≤ 2 ∧ 2 ≤ b) → (a + 2 > b) ∧ (a + b > 2) ∧ (2 + b > a) →
    (q = 3)) :=
by
  sorry

end number_of_non_congruent_triangles_l39_39384


namespace total_profit_l39_39924

noncomputable def profit_x (P : ℕ) : ℕ := 3 * P
noncomputable def profit_y (P : ℕ) : ℕ := 2 * P

theorem total_profit
  (P_x P_y : ℕ)
  (h_ratio : P_x = 3 * (P_y / 2))
  (h_diff : P_x - P_y = 100) :
  P_x + P_y = 500 :=
by
  sorry

end total_profit_l39_39924


namespace find_x_range_l39_39613

-- Given definition for a decreasing function
def is_decreasing (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x > f y

-- The main theorem to prove
theorem find_x_range (f : ℝ → ℝ) (h_decreasing : is_decreasing f) :
  {x : ℝ | f (|1 / x|) < f 1} = {x | -1 < x ∧ x < 0} ∪ {x | 0 < x ∧ x < 1} :=
sorry

end find_x_range_l39_39613


namespace total_length_segments_l39_39485

noncomputable def segment_length (rect_horizontal_1 rect_horizontal_2 rect_vertical left_segment : ℕ) :=
  let total_length := rect_horizontal_1 + rect_horizontal_2 + rect_vertical
  total_length - 8 + left_segment

theorem total_length_segments
  (rect_horizontal_1 rect_horizontal_2 rect_vertical left_segment total_left : ℕ)
  (h1 : rect_horizontal_1 = 10)
  (h2 : rect_horizontal_2 = 3)
  (h3 : rect_vertical = 12)
  (h4 : left_segment = 8)
  (h5 : total_left = 19)
  : segment_length rect_horizontal_1 rect_horizontal_2 rect_vertical left_segment = total_left :=
sorry

end total_length_segments_l39_39485


namespace money_left_after_purchase_l39_39109

def initial_toonies : Nat := 4
def value_per_toonie : Nat := 2
def total_coins : Nat := 10
def value_per_loonie : Nat := 1
def frappuccino_cost : Nat := 3

def toonies_value : Nat := initial_toonies * value_per_toonie
def loonies : Nat := total_coins - initial_toonies
def loonies_value : Nat := loonies * value_per_loonie
def initial_total : Nat := toonies_value + loonies_value
def remaining_money : Nat := initial_total - frappuccino_cost

theorem money_left_after_purchase : remaining_money = 11 := by
  sorry

end money_left_after_purchase_l39_39109


namespace empty_set_a_gt_nine_over_eight_singleton_set_a_values_at_most_one_element_set_a_range_l39_39496

noncomputable def A (a : ℝ) : Set ℝ := { x | a*x^2 - 3*x + 2 = 0 }

theorem empty_set_a_gt_nine_over_eight (a : ℝ) : A a = ∅ ↔ a > 9 / 8 :=
by
  sorry

theorem singleton_set_a_values (a : ℝ) : (∃ x, A a = {x}) ↔ (a = 0 ∨ a = 9 / 8) :=
by
  sorry

theorem at_most_one_element_set_a_range (a : ℝ) : (∀ x y, x ∈ A a → y ∈ A a → x = y) →
  (A a = ∅ ∨ ∃ x, A a = {x}) ↔ (a = 0 ∨ a ≥ 9 / 8) :=
by
  sorry

end empty_set_a_gt_nine_over_eight_singleton_set_a_values_at_most_one_element_set_a_range_l39_39496


namespace count_divisors_of_100000_l39_39359

theorem count_divisors_of_100000 : 
  ∃ n : ℕ, n = 36 ∧ ∀ k : ℕ, (k ∣ 100000) → ∃ (i j : ℕ), 0 ≤ i ∧ i ≤ 5 ∧ 0 ≤ j ∧ j ≤ 5 ∧ k = 2^i * 5^j := by
  sorry

end count_divisors_of_100000_l39_39359


namespace left_vertex_of_ellipse_l39_39345

theorem left_vertex_of_ellipse : 
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ (∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1) ∧
  (∀ (x y : ℝ), x^2 + y^2 - 6*x + 8 = 0 ∧ x = a - 5) ∧
  2 * b = 8 → left_vertex = (-5, 0) :=
sorry

end left_vertex_of_ellipse_l39_39345


namespace black_white_ratio_l39_39378

theorem black_white_ratio 
  (x y : ℕ) 
  (h1 : (y - 1) * 7 = x * 9) 
  (h2 : y * 5 = (x - 1) * 7) : 
  y - x = 7 := 
by 
  sorry

end black_white_ratio_l39_39378


namespace metallic_sheet_dimension_l39_39573

theorem metallic_sheet_dimension :
  ∃ w : ℝ, (∀ (h := 8) (l := 40) (v := 2688),
    v = (w - 2 * h) * (l - 2 * h) * h) → w = 30 :=
by sorry

end metallic_sheet_dimension_l39_39573


namespace observations_decrement_l39_39236

theorem observations_decrement (n : ℤ) (h_n_pos : n > 0) : 200 - 15 = 185 :=
by
  sorry

end observations_decrement_l39_39236


namespace fraction_bad_teams_leq_l39_39173

variable (teams total_teams : ℕ) (b : ℝ)

-- Given conditions
variable (cond₁ : total_teams = 18)
variable (cond₂ : teams = total_teams / 2)
variable (cond₃ : ∀ (rb_teams : ℕ), rb_teams ≠ 10 → rb_teams ≤ teams)

theorem fraction_bad_teams_leq (H : 18 * b ≤ teams) : b ≤ 1 / 2 :=
sorry

end fraction_bad_teams_leq_l39_39173


namespace students_count_inconsistent_l39_39695

-- Define the conditions
variables (total_students boys_more_than_girls : ℤ)

-- Define the main theorem: The computed number of girls is not an integer
theorem students_count_inconsistent 
  (h1 : total_students = 3688) 
  (h2 : boys_more_than_girls = 373) 
  : ¬ ∃ x : ℤ, 2 * x + boys_more_than_girls = total_students := 
by
  sorry

end students_count_inconsistent_l39_39695


namespace units_digit_of_33_pow_33_mul_22_pow_22_l39_39038

theorem units_digit_of_33_pow_33_mul_22_pow_22 :
  (33 ^ (33 * (22 ^ 22))) % 10 = 1 :=
sorry

end units_digit_of_33_pow_33_mul_22_pow_22_l39_39038


namespace correct_integer_with_7_divisors_l39_39893

theorem correct_integer_with_7_divisors (n : ℕ) (p : ℕ) (h_prime : Prime p) 
  (h_3_divisors : ∃ (d : ℕ), d = 3 ∧ n = p^2) : n = 4 :=
by
-- Proof omitted
sorry

end correct_integer_with_7_divisors_l39_39893


namespace largest_angle_in_triangle_l39_39676

theorem largest_angle_in_triangle (A B C : ℝ) 
  (h_sum : A + B = 126) 
  (h_diff : B = A + 40) 
  (h_triangle : A + B + C = 180) : max A (max B C) = 83 := 
by
  sorry

end largest_angle_in_triangle_l39_39676


namespace remainder_130_div_k_l39_39245

theorem remainder_130_div_k (k : ℕ) (h_positive : k > 0)
  (h_remainder : 84 % (k*k) = 20) : 
  130 % k = 2 := 
by sorry

end remainder_130_div_k_l39_39245


namespace total_cost_shorts_tshirt_boots_shinguards_l39_39760

variable (x : ℝ)

-- Definitions provided in the problem statement.
def cost_shorts : ℝ := x
def cost_shorts_and_tshirt : ℝ := 2 * x
def cost_shorts_and_boots : ℝ := 5 * x
def cost_shorts_and_shinguards : ℝ := 3 * x

-- The proof goal to verify:
theorem total_cost_shorts_tshirt_boots_shinguards : 
  (cost_shorts x + (cost_shorts_and_tshirt x - cost_shorts x) + 
   (cost_shorts_and_boots x - cost_shorts x) + 
   (cost_shorts_and_shinguards x - cost_shorts x)) = 8 * x := by 
  sorry

end total_cost_shorts_tshirt_boots_shinguards_l39_39760


namespace complex_multiplication_l39_39674

theorem complex_multiplication (a b c d : ℤ) (i : ℂ) (hi : i^2 = -1) : 
  ((3 : ℂ) - 4 * i) * ((-7 : ℂ) + 6 * i) = (3 : ℂ) + 46 * i := 
  by
    sorry

end complex_multiplication_l39_39674


namespace tan_diff_eqn_l39_39927

theorem tan_diff_eqn (α : ℝ) (h : Real.tan α = 2) : Real.tan (α - 3 * Real.pi / 4) = -3 := 
by 
  sorry

end tan_diff_eqn_l39_39927


namespace radioactive_decay_minimum_years_l39_39312

noncomputable def min_years (a : ℝ) (n : ℕ) : Prop :=
  (a * (1 - 3 / 4) ^ n ≤ a * 1 / 100)

theorem radioactive_decay_minimum_years (a : ℝ) (h : 0 < a) : ∃ n : ℕ, min_years a n ∧ n = 4 :=
by {
  sorry
}

end radioactive_decay_minimum_years_l39_39312


namespace second_number_is_twenty_two_l39_39163

theorem second_number_is_twenty_two (x y : ℕ) 
  (h1 : x + y = 33) 
  (h2 : y = 2 * x) : 
  y = 22 :=
by
  sorry

end second_number_is_twenty_two_l39_39163


namespace sum_of_interior_angles_6_find_n_from_300_degrees_l39_39271

-- Definitions and statement for part 1:
def sum_of_interior_angles (n : ℕ) : ℕ :=
  (n - 2) * 180

theorem sum_of_interior_angles_6 :
  sum_of_interior_angles 6 = 720 := 
by
  sorry

-- Definitions and statement for part 2:
def find_n_from_angles (angle : ℕ) : ℕ := 
  (angle / 180) + 2

theorem find_n_from_300_degrees :
  find_n_from_angles 900 = 7 :=
by
  sorry

end sum_of_interior_angles_6_find_n_from_300_degrees_l39_39271


namespace complement_of_M_in_U_l39_39193

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 4}

theorem complement_of_M_in_U : U \ M = {3, 5, 6} := by
  sorry

end complement_of_M_in_U_l39_39193


namespace complement_union_l39_39155

open Set

def S : Set ℝ := { x | x > -2 }
def T : Set ℝ := { x | x^2 + 3*x - 4 ≤ 0 }

theorem complement_union :
  (compl S) ∪ T = { x : ℝ | x ≤ 1 } :=
sorry

end complement_union_l39_39155


namespace circle_center_coordinates_l39_39251

theorem circle_center_coordinates :
  ∃ (h k : ℝ), (x^2 + y^2 - 2*x + 4*y = 0) → (x - h)^2 + (y + k)^2 = 5 :=
sorry

end circle_center_coordinates_l39_39251


namespace rectangle_area_k_l39_39731

theorem rectangle_area_k (d : ℝ) (length width : ℝ) (h_ratio : length / width = 5 / 2)
  (h_diag : (length ^ 2 + width ^ 2) = d ^ 2) :
  ∃ (k : ℝ), k = 10 / 29 ∧ length * width = k * d ^ 2 := by
  sorry

end rectangle_area_k_l39_39731


namespace min_distance_sq_l39_39338

theorem min_distance_sq (x y : ℝ) (h : x - y - 1 = 0) : (x - 2) ^ 2 + (y - 2) ^ 2 = 1 / 2 :=
sorry

end min_distance_sq_l39_39338


namespace simplify_inv_sum_l39_39972

variables {x y z : ℝ}

theorem simplify_inv_sum (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x⁻¹ + y⁻¹ + z⁻¹)⁻¹ = xyz / (yz + xz + xy) :=
by
  sorry

end simplify_inv_sum_l39_39972


namespace domain_condition_implies_m_range_range_condition_implies_m_range_l39_39491

noncomputable def f (x m : ℝ) : ℝ := Real.log (x^2 - 2 * m * x + m + 2)

def condition1 (m : ℝ) : Prop :=
  ∀ x : ℝ, (x^2 - 2 * m * x + m + 2 > 0)

def condition2 (m : ℝ) : Prop :=
  ∃ y : ℝ, (∀ x : ℝ, y = Real.log (x^2 - 2 * m * x + m + 2))

theorem domain_condition_implies_m_range (m : ℝ) :
  condition1 m → -1 < m ∧ m < 2 :=
sorry

theorem range_condition_implies_m_range (m : ℝ) :
  condition2 m → (m ≤ -1 ∨ m ≥ 2) :=
sorry

end domain_condition_implies_m_range_range_condition_implies_m_range_l39_39491


namespace sarah_marry_age_l39_39354

/-- Sarah is 9 years old. -/
def Sarah_age : ℕ := 9

/-- Sarah's name has 5 letters. -/
def Sarah_name_length : ℕ := 5

/-- The game's rule is to add the number of letters in the player's name 
    to twice the player's age. -/
def game_rule (name_length age : ℕ) : ℕ :=
  name_length + 2 * age

/-- Prove that Sarah will get married at the age of 23. -/
theorem sarah_marry_age : game_rule Sarah_name_length Sarah_age = 23 := 
  sorry

end sarah_marry_age_l39_39354


namespace simplify_and_evaluate_expression_l39_39529

-- Define the conditions
def a := 2
def b := -1

-- State the theorem
theorem simplify_and_evaluate_expression : 
  ((2 * a + 3 * b) * (2 * a - 3 * b) - (2 * a - b) ^ 2 - 3 * a * b) / (-b) = -12 := by
  -- Placeholder for the proof
  sorry

end simplify_and_evaluate_expression_l39_39529


namespace milk_concentration_l39_39897

variable {V_initial V_removed V_total : ℝ}

theorem milk_concentration (h1 : V_initial = 20) (h2 : V_removed = 2) (h3 : V_total = 20) :
    (V_initial - V_removed) / V_total * 100 = 90 := 
by 
  sorry

end milk_concentration_l39_39897


namespace evaluate_expression_l39_39150

theorem evaluate_expression : (2 * (-1) + 3) * (2 * (-1) - 3) - ((-1) - 1) * ((-1) + 5) = 3 := by
  sorry

end evaluate_expression_l39_39150


namespace emma_bank_account_balance_l39_39353

def initial_amount : ℝ := 230
def withdrawn_amount : ℝ := 60
def deposit_amount : ℝ := 2 * withdrawn_amount
def final_amount : ℝ := initial_amount - withdrawn_amount + deposit_amount

theorem emma_bank_account_balance : final_amount = 290 := 
by 
  -- Definitions have already been stated; the proof is not required
  sorry

end emma_bank_account_balance_l39_39353


namespace math_problem_l39_39259

theorem math_problem :
  (1 / (1 / (1 / (1 / (3 + 2 : ℝ) - 1 : ℝ) - 1 : ℝ) - 1 : ℝ) - 1 : ℝ) = -13 / 9 :=
by
  -- proof goes here
  sorry

end math_problem_l39_39259


namespace ratio_men_to_women_l39_39172

theorem ratio_men_to_women
  (W M : ℕ)      -- W is the number of women, M is the number of men
  (avg_height_all : ℕ) (avg_height_female : ℕ) (avg_height_male : ℕ)
  (h1 : avg_height_all = 180)
  (h2 : avg_height_female = 170)
  (h3 : avg_height_male = 182) 
  (h_avg : (170 * W + 182 * M) / (W + M) = 180) :
  M = 5 * W :=
by
  sorry

end ratio_men_to_women_l39_39172


namespace find_balcony_seat_cost_l39_39357

-- Definitions based on conditions
variable (O B : ℕ) -- Number of orchestra tickets and cost of balcony ticket
def orchestra_ticket_cost : ℕ := 12
def total_tickets : ℕ := 370
def total_cost : ℕ := 3320
def tickets_difference : ℕ := 190

-- Lean statement to prove the cost of a balcony seat
theorem find_balcony_seat_cost :
  (2 * O + tickets_difference = total_tickets) ∧
  (orchestra_ticket_cost * O + B * (O + tickets_difference) = total_cost) →
  B = 8 :=
by
  sorry

end find_balcony_seat_cost_l39_39357


namespace spending_ratio_l39_39625

theorem spending_ratio 
  (lisa_tshirts : Real)
  (lisa_jeans : Real)
  (lisa_coats : Real)
  (carly_tshirts : Real)
  (carly_jeans : Real)
  (carly_coats : Real)
  (total_spent : Real)
  (hl1 : lisa_tshirts = 40)
  (hl2 : lisa_jeans = lisa_tshirts / 2)
  (hl3 : lisa_coats = 2 * lisa_tshirts)
  (hc1 : carly_tshirts = lisa_tshirts / 4)
  (hc2 : carly_coats = lisa_coats / 4)
  (htotal : total_spent = lisa_tshirts + lisa_jeans + lisa_coats + carly_tshirts + carly_jeans + carly_coats)
  (h_total_spent_val : total_spent = 230) :
  carly_jeans = 3 * lisa_jeans :=
by
  -- Placeholder for theorem's proof
  sorry

end spending_ratio_l39_39625


namespace relationship_between_products_l39_39134

variable {a₁ a₂ b₁ b₂ : ℝ}

theorem relationship_between_products (h₁ : a₁ < a₂) (h₂ : b₁ < b₂) : a₁ * b₁ + a₂ * b₂ > a₁ * b₂ + a₂ * b₁ := 
sorry

end relationship_between_products_l39_39134


namespace cos_double_angle_l39_39650

theorem cos_double_angle (y0 : ℝ) (h : (1 / 3)^2 + y0^2 = 1) : 
  Real.cos (2 * Real.arccos (1 / 3)) = -7 / 9 := 
by
  sorry

end cos_double_angle_l39_39650


namespace paint_cost_of_cube_l39_39019

theorem paint_cost_of_cube (cost_per_kg : ℕ) (coverage_per_kg : ℕ) (side_length : ℕ) (total_cost : ℕ) 
  (h1 : cost_per_kg = 20)
  (h2 : coverage_per_kg = 15)
  (h3 : side_length = 5)
  (h4 : total_cost = 200) : 
  (6 * side_length^2 / coverage_per_kg) * cost_per_kg = total_cost :=
by
  sorry

end paint_cost_of_cube_l39_39019


namespace length_ab_square_l39_39295

theorem length_ab_square (s a : ℝ) (h_square : s = 2 * a) (h_area : 3000 = 1/2 * (s + (s - 2 * a)) * s) : 
  s = 20 * Real.sqrt 15 :=
by
  sorry

end length_ab_square_l39_39295


namespace find_x_such_that_ceil_mul_x_eq_168_l39_39108

theorem find_x_such_that_ceil_mul_x_eq_168 (x : ℝ) (h_pos : x > 0)
  (h_eq : ⌈x⌉ * x = 168) (h_ceil: ⌈x⌉ - 1 < x ∧ x ≤ ⌈x⌉) :
  x = 168 / 13 :=
by
  sorry

end find_x_such_that_ceil_mul_x_eq_168_l39_39108


namespace responses_needed_l39_39951

theorem responses_needed (p : ℝ) (q : ℕ) (r : ℕ) : 
  p = 0.6 → q = 370 → r = 222 → 
  q * p = r := 
by
  intros hp hq hr
  rw [hp, hq] 
  sorry

end responses_needed_l39_39951


namespace probability_one_red_ball_distribution_of_X_l39_39066

-- Definitions of probabilities
def C (n k : ℕ) : ℕ := Nat.choose n k

def P_one_red_ball : ℚ := (C 2 1 * C 3 2 : ℚ) / C 5 3

#check (1 : ℚ)
#check (3 : ℚ)
#check (5 : ℚ)
def X_distribution (i : ℕ) : ℚ :=
  if i = 0 then (C 3 3 : ℚ) / C 5 3
  else if i = 1 then (C 2 1 * C 3 2 : ℚ) / C 5 3
  else if i = 2 then (C 2 2 * C 3 1 : ℚ) / C 5 3
  else 0

-- Statement to prove
theorem probability_one_red_ball : 
  P_one_red_ball = 3 / 5 := 
sorry

theorem distribution_of_X :
  Π i, (i = 0 → X_distribution i = 1 / 10) ∧
       (i = 1 → X_distribution i = 3 / 5) ∧
       (i = 2 → X_distribution i = 3 / 10) :=
sorry

end probability_one_red_ball_distribution_of_X_l39_39066


namespace determine_beta_l39_39815

-- Define a structure for angles in space
structure Angle where
  measure : ℝ

-- Define the conditions
def alpha : Angle := ⟨30⟩
def parallel_sides (a b : Angle) : Prop := true  -- Simplification for the example, should be defined properly for general case

-- The theorem to be proved
theorem determine_beta (α β : Angle) (h1 : α = Angle.mk 30) (h2 : parallel_sides α β) : β = Angle.mk 30 ∨ β = Angle.mk 150 := by
  sorry

end determine_beta_l39_39815


namespace janice_total_earnings_l39_39107

-- Defining the working conditions as constants
def days_per_week : ℕ := 5  -- Janice works 5 days a week
def earning_per_day : ℕ := 30  -- Janice earns $30 per day
def overtime_earning_per_shift : ℕ := 15  -- Janice earns $15 per overtime shift
def overtime_shifts : ℕ := 3  -- Janice works three overtime shifts

-- Defining Janice's total earnings for the week
def total_earnings : ℕ := (days_per_week * earning_per_day) + (overtime_shifts * overtime_earning_per_shift)

-- Statement to prove that Janice's total earnings are $195
theorem janice_total_earnings : total_earnings = 195 :=
by
  -- The proof is omitted.
  sorry

end janice_total_earnings_l39_39107


namespace min_value_geom_seq_l39_39343

theorem min_value_geom_seq (a : ℕ → ℝ) (r m n : ℕ) (h_geom : ∃ r, ∀ i, a (i + 1) = a i * r)
  (h_ratio : r = 2) (h_a_m : 4 * a 1 = a m) :
  ∃ (m n : ℕ), (m + n = 6) → (1 / m + 4 / n) = 3 / 2 :=
by 
  sorry

end min_value_geom_seq_l39_39343


namespace minimum_trains_needed_l39_39273

theorem minimum_trains_needed (n : ℕ) (h : 50 * n >= 645) : n = 13 :=
by
  sorry

end minimum_trains_needed_l39_39273


namespace find_k_l39_39689

theorem find_k (k : ℝ) : 
  (∃ (x y : ℝ), 2 * x + 3 * y + 8 = 0 ∧ x - y - 1 = 0 ∧ x + k * y = 0) → k = -1 / 2 :=
by 
  sorry

end find_k_l39_39689


namespace find_fourth_number_l39_39607

theorem find_fourth_number (x y : ℝ) (h1 : 0.25 / x = 2 / y) (h2 : x = 0.75) : y = 6 :=
by
  sorry

end find_fourth_number_l39_39607


namespace jenny_problem_l39_39534

def round_to_nearest_ten (n : ℤ) : ℤ :=
  if n % 10 < 5 then n - (n % 10) else n + (10 - n % 10)

theorem jenny_problem : round_to_nearest_ten (58 + 29) = 90 := 
by
  sorry

end jenny_problem_l39_39534


namespace scientific_notation_of_56_point_5_million_l39_39548

-- Definitions based on conditions
def million : ℝ := 10^6
def number_in_millions : ℝ := 56.5 * million

-- Statement to be proved
theorem scientific_notation_of_56_point_5_million : 
  number_in_millions = 5.65 * 10^7 :=
sorry

end scientific_notation_of_56_point_5_million_l39_39548


namespace total_spending_l39_39140

-- Conditions used as definitions
def price_pants : ℝ := 110.00
def discount_pants : ℝ := 0.30
def number_of_pants : ℕ := 4

def price_socks : ℝ := 60.00
def discount_socks : ℝ := 0.30
def number_of_socks : ℕ := 2

-- Lean 4 statement to prove the total spending
theorem total_spending :
  (number_of_pants : ℝ) * (price_pants * (1 - discount_pants)) +
  (number_of_socks : ℝ) * (price_socks * (1 - discount_socks)) = 392.00 :=
by
  sorry

end total_spending_l39_39140


namespace eval_expression_l39_39604

-- Definitions based on the conditions and problem statement
def x (b : ℕ) : ℕ := b + 9

-- The theorem to prove
theorem eval_expression (b : ℕ) : x b - b + 5 = 14 := by
    sorry

end eval_expression_l39_39604


namespace fibonacci_150_mod_7_l39_39875

def fibonacci_mod_7 : Nat → Nat
| 0 => 0
| 1 => 1
| n + 2 => (fibonacci_mod_7 (n + 1) + fibonacci_mod_7 n) % 7

theorem fibonacci_150_mod_7 : fibonacci_mod_7 150 = 1 := 
by sorry

end fibonacci_150_mod_7_l39_39875


namespace cuboid_second_edge_l39_39226

variable (x : ℝ)

theorem cuboid_second_edge (h1 : 4 * x * 6 = 96) : x = 4 := by
  sorry

end cuboid_second_edge_l39_39226


namespace angelina_speed_l39_39465

theorem angelina_speed (v : ℝ) (h1 : 840 / v - 40 = 240 / v) :
  2 * v = 30 :=
by
  sorry

end angelina_speed_l39_39465


namespace root_reciprocals_identity_l39_39806

noncomputable def cubic_roots (a b c : ℝ) : Prop :=
  (a + b + c = 12) ∧ (a * b + b * c + c * a = 20) ∧ (a * b * c = -5)

theorem root_reciprocals_identity (a b c : ℝ) (h : cubic_roots a b c) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) = 20.8 :=
by
  sorry

end root_reciprocals_identity_l39_39806


namespace max_m_plus_n_l39_39546

noncomputable def quadratic_function (x : ℝ) : ℝ := -x^2 + 3

theorem max_m_plus_n (m n : ℝ) (h : n = quadratic_function m) : m + n ≤ 13/4 :=
sorry

end max_m_plus_n_l39_39546


namespace smallest_is_C_l39_39986

def A : ℚ := 1/2
def B : ℚ := 9/10
def C : ℚ := 2/5

theorem smallest_is_C : min (min A B) C = C := 
by
  sorry

end smallest_is_C_l39_39986


namespace cos_alpha_beta_half_l39_39399

open Real

theorem cos_alpha_beta_half (α β : ℝ)
  (h1 : cos (α - β / 2) = -1 / 3)
  (h2 : sin (α / 2 - β) = 1 / 4)
  (h3 : 3 * π / 2 < α ∧ α < 2 * π)
  (h4 : π / 2 < β ∧ β < π) :
  cos ((α + β) / 2) = -(2 * sqrt 2 + sqrt 15) / 12 :=
by
  sorry

end cos_alpha_beta_half_l39_39399


namespace least_positive_24x_16y_l39_39122

theorem least_positive_24x_16y (x y : ℤ) : ∃ a : ℕ, a > 0 ∧ a = 24 * x + 16 * y ∧ ∀ b : ℕ, b = 24 * x + 16 * y → b > 0 → b ≥ a :=
sorry

end least_positive_24x_16y_l39_39122


namespace arithmetic_sequence_a12_bound_l39_39255

theorem arithmetic_sequence_a12_bound (a_1 d : ℤ) (h8 : a_1 + 7 * d ≥ 15) (h9 : a_1 + 8 * d ≤ 13) : 
  a_1 + 11 * d ≤ 7 :=
by
  sorry

end arithmetic_sequence_a12_bound_l39_39255


namespace roots_of_quadratic_l39_39320

theorem roots_of_quadratic (m n : ℝ) (h₁ : m + n = -2) (h₂ : m * n = -2022) (h₃ : ∀ x, x^2 + 2 * x - 2022 = 0 → x = m ∨ x = n) :
  m^2 + 3 * m + n = 2020 :=
sorry

end roots_of_quadratic_l39_39320


namespace simplify_polynomials_l39_39090

theorem simplify_polynomials :
  (4 * q ^ 4 + 2 * p ^ 3 - 7 * p + 8) + (3 * q ^ 4 - 2 * p ^ 3 + 3 * p ^ 2 - 5 * p + 6) =
  7 * q ^ 4 + 3 * p ^ 2 - 12 * p + 14 :=
by
  sorry

end simplify_polynomials_l39_39090


namespace kevin_stone_count_l39_39495

theorem kevin_stone_count :
  ∃ (N : ℕ), (∀ (n k : ℕ), 2007 = 9 * n + 11 * k → N = 20) := 
sorry

end kevin_stone_count_l39_39495


namespace sum_first_12_terms_l39_39335

def a (n : ℕ) : ℕ :=
  if n % 2 = 1 then 2 ^ (n - 1) else 2 * n - 1

def S (n : ℕ) : ℕ := 
  (Finset.range n).sum a

theorem sum_first_12_terms : S 12 = 1443 :=
by
  sorry

end sum_first_12_terms_l39_39335


namespace division_yields_square_l39_39681

theorem division_yields_square (a b : ℕ) (hab : ab + 1 ∣ a^2 + b^2) :
  ∃ m : ℕ, m^2 = (a^2 + b^2) / (ab + 1) :=
sorry

end division_yields_square_l39_39681


namespace radius_of_spheres_in_cube_l39_39863

noncomputable def sphere_radius (sides: ℝ) (spheres: ℕ) (tangent_pairs: ℕ) (tangent_faces: ℕ): ℝ :=
  if sides = 2 ∧ spheres = 10 ∧ tangent_pairs = 2 ∧ tangent_faces = 3 then 0.5 else 0

theorem radius_of_spheres_in_cube : sphere_radius 2 10 2 3 = 0.5 :=
by
  -- This is the main theorem that states the radius of each sphere given the problem conditions.
  sorry

end radius_of_spheres_in_cube_l39_39863


namespace find_pairs_l39_39835

noncomputable def pairs_of_real_numbers (α β : ℝ) := 
  ∀ x y z w : ℝ, 0 < x → 0 < y → 0 < z → 0 < w →
    (x + y^2 + z^3 + w^6 ≥ α * (x * y * z * w)^β)

theorem find_pairs (α β : ℝ) :
  (∃ x y z w : ℝ, 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w ∧
    (x + y^2 + z^3 + w^6 = α * (x * y * z * w)^β))
  →
  pairs_of_real_numbers α β :=
sorry

end find_pairs_l39_39835


namespace total_earnings_l39_39942

-- Definitions based on conditions
def bead_necklaces : ℕ := 7
def gem_necklaces : ℕ := 3
def cost_per_necklace : ℕ := 9

-- The main theorem to prove
theorem total_earnings : (bead_necklaces + gem_necklaces) * cost_per_necklace = 90 :=
by
  sorry

end total_earnings_l39_39942


namespace Sadie_l39_39238

theorem Sadie's_homework_problems (T : ℝ) 
  (h1 : 0.40 * T = A) 
  (h2 : 0.5 * A = 28) 
  : T = 140 := 
by
  sorry

end Sadie_l39_39238


namespace A_2013_eq_neg_1007_l39_39112

def A (n : ℕ) : ℤ :=
  (-1)^n * ((n + 1) / 2)

theorem A_2013_eq_neg_1007 : A 2013 = -1007 :=
by
  sorry

end A_2013_eq_neg_1007_l39_39112


namespace sphere_volume_l39_39758

theorem sphere_volume (r : ℝ) (h1 : 4 * π * r^2 = 256 * π) : 
  (4 / 3) * π * r^3 = (2048 / 3) * π :=
by
  sorry

end sphere_volume_l39_39758


namespace no_intersection_points_l39_39651

-- Define the absolute value functions
def f1 (x : ℝ) : ℝ := abs (3 * x + 6)
def f2 (x : ℝ) : ℝ := -abs (4 * x - 3)

-- State the theorem
theorem no_intersection_points : ∀ x y : ℝ, f1 x = y ∧ f2 x = y → false := by
  sorry

end no_intersection_points_l39_39651


namespace factorial_fraction_integer_l39_39536

open Nat

theorem factorial_fraction_integer (m n : ℕ) : 
  ∃ k : ℕ, k = (2 * m).factorial * (2 * n).factorial / (m.factorial * n.factorial * (m + n).factorial) := 
sorry

end factorial_fraction_integer_l39_39536


namespace chocolates_per_student_class_7B_l39_39254

theorem chocolates_per_student_class_7B :
  (∃ (x : ℕ), 9 * x < 288 ∧ 10 * x > 300 ∧ x = 31) :=
by
  use 31
  -- proof steps omitted here
  sorry

end chocolates_per_student_class_7B_l39_39254


namespace bob_calories_l39_39521

-- conditions
def slices : ℕ := 8
def half_slices (slices : ℕ) : ℕ := slices / 2
def calories_per_slice : ℕ := 300
def total_calories (half_slices : ℕ) (calories_per_slice : ℕ) : ℕ := half_slices * calories_per_slice

-- proof problem
theorem bob_calories : total_calories (half_slices slices) calories_per_slice = 1200 := by
  sorry

end bob_calories_l39_39521


namespace max_value_of_quadratic_l39_39459

theorem max_value_of_quadratic :
  ∃ (x : ℝ), ∀ (y : ℝ), -3 * y^2 + 18 * y - 5 ≤ -3 * x^2 + 18 * x - 5 ∧ -3 * x^2 + 18 * x - 5 = 22 :=
sorry

end max_value_of_quadratic_l39_39459


namespace quadrilateral_area_l39_39560

theorem quadrilateral_area (d h1 h2 : ℝ) (hd : d = 30) (hh1 : h1 = 10) (hh2 : h2 = 6) :
  (1 / 2 * d * (h1 + h2) = 240) := by
  sorry

end quadrilateral_area_l39_39560


namespace range_of_x_l39_39065

theorem range_of_x (x : ℝ) (h1 : 2 ≤ |x - 5|) (h2 : |x - 5| ≤ 10) (h3 : 0 < x) : 
  (0 < x ∧ x ≤ 3) ∨ (7 ≤ x ∧ x ≤ 15) := 
sorry

end range_of_x_l39_39065


namespace full_price_tickets_revenue_l39_39646

theorem full_price_tickets_revenue (f h d p : ℕ) 
  (h1 : f + h + d = 200) 
  (h2 : f * p + h * (p / 2) + d * (2 * p) = 5000) 
  (h3 : p = 50) : 
  f * p = 4500 :=
by
  sorry

end full_price_tickets_revenue_l39_39646


namespace smallest_c_value_l39_39331

theorem smallest_c_value :
  ∃ a b c : ℕ, a * b * c = 3990 ∧ a + b + c = 56 ∧ a > 0 ∧ b > 0 ∧ c > 0 :=
by {
  -- Skipping proof as instructed
  sorry
}

end smallest_c_value_l39_39331


namespace a7_of_arithmetic_seq_l39_39918

-- Defining the arithmetic sequence
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) := ∀ n, a (n + 1) = a n + d

theorem a7_of_arithmetic_seq (a : ℕ → ℤ) (d : ℤ) 
  (h_arith : arithmetic_seq a d) 
  (h_a4 : a 4 = 5) 
  (h_a5_a6 : a 5 + a 6 = 11) : 
  a 7 = 6 :=
by
  sorry

end a7_of_arithmetic_seq_l39_39918


namespace average_marks_l39_39414

theorem average_marks {n : ℕ} (h1 : 5 * 74 + 104 = n * 79) : n = 6 :=
by
  sorry

end average_marks_l39_39414


namespace complementary_implies_right_triangle_l39_39557

theorem complementary_implies_right_triangle (A B C : ℝ) (h : A + B = 90 ∧ A + B + C = 180) :
  C = 90 :=
by
  sorry

end complementary_implies_right_triangle_l39_39557


namespace projected_increase_l39_39289

theorem projected_increase (R : ℝ) (P : ℝ) 
  (h1 : ∃ P, ∀ (R : ℝ), 0.9 * R = 0.75 * (R + (P / 100) * R)) 
  (h2 : ∀ (R : ℝ), R > 0) :
  P = 20 :=
by
  sorry

end projected_increase_l39_39289


namespace calculate_milk_and_oil_l39_39905

theorem calculate_milk_and_oil (q_f div_f milk_p oil_p : ℕ) (portions q_m q_o : ℕ) :
  q_f = 1050 ∧ div_f = 350 ∧ milk_p = 70 ∧ oil_p = 30 ∧
  portions = q_f / div_f ∧
  q_m = portions * milk_p ∧
  q_o = portions * oil_p →
  q_m = 210 ∧ q_o = 90 := by
  sorry

end calculate_milk_and_oil_l39_39905


namespace initial_number_of_eggs_l39_39755

theorem initial_number_of_eggs (eggs_taken harry_eggs eggs_left initial_eggs : ℕ)
    (h1 : harry_eggs = 5)
    (h2 : eggs_left = 42)
    (h3 : initial_eggs = eggs_left + harry_eggs) : 
    initial_eggs = 47 := by
  sorry

end initial_number_of_eggs_l39_39755


namespace find_algebraic_expression_l39_39302

-- Definitions as per the conditions
variable (a b : ℝ)

-- Given condition
def given_condition (σ : ℝ) : Prop := σ * (2 * a * b) = 4 * a^2 * b

-- The statement to prove
theorem find_algebraic_expression (σ : ℝ) (h : given_condition a b σ) : σ = 2 * a := 
sorry

end find_algebraic_expression_l39_39302


namespace find_angle_A_find_perimeter_l39_39629

-- Given problem conditions as Lean definitions
def triangle_sides (a b c : ℝ) : Prop :=
  ∃ B : ℝ, c = a * (Real.cos B + Real.sqrt 3 * Real.sin B)

def triangle_area (S a : ℝ) : Prop :=
  S = Real.sqrt 3 / 4 ∧ a = 1

-- Prove angle A
theorem find_angle_A (a b c S : ℝ) (hc : triangle_sides a b c) (ha : triangle_area S a) :
  ∃ A : ℝ, A = Real.pi / 6 := 
sorry

-- Prove perimeter
theorem find_perimeter (a b c S : ℝ) (hc : triangle_sides a b c) (ha : triangle_area S a) :
  ∃ P : ℝ, P = Real.sqrt 3 + 2 := 
sorry

end find_angle_A_find_perimeter_l39_39629


namespace smallest_positive_period_of_f_symmetry_center_of_f_range_of_f_in_interval_l39_39304

open Real

noncomputable def a (x : ℝ) : ℝ × ℝ := (5 * sqrt 3 * cos x, cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (sin x, 2 * cos x)

noncomputable def f (x : ℝ) : ℝ := 
  let a_dot_b := (a x).1 * (b x).1 + (a x).2 * (b x).2
  let b_norm_sq := (b x).1 ^ 2 + (b x).2 ^ 2
  a_dot_b + b_norm_sq + 3 / 2

theorem smallest_positive_period_of_f :
  ∀ x, f (x + π) = f x := sorry

theorem symmetry_center_of_f :
  ∃ k : ℤ, ∀ x, f x = 5 ↔ x = (-π / 12 + k * (π / 2) : ℝ) := sorry

theorem range_of_f_in_interval :
  ∀ x, (π / 6 ≤ x ∧ x ≤ π / 2) → (5 / 2 ≤ f x ∧ f x ≤ 10) := sorry

end smallest_positive_period_of_f_symmetry_center_of_f_range_of_f_in_interval_l39_39304


namespace peter_work_days_l39_39971

variable (W M P : ℝ)
variable (h1 : M + P = W / 20) -- Combined rate of Matt and Peter
variable (h2 : 12 * (W / 20) + 14 * P = W) -- Work done by Matt and Peter for 12 days + Peter's remaining work

theorem peter_work_days :
  P = W / 35 :=
by
  sorry

end peter_work_days_l39_39971


namespace traveler_journey_possible_l39_39885

structure Archipelago (Island : Type) :=
  (n : ℕ)
  (fare : Island → Island → ℝ)
  (unique_ferry : ∀ i j : Island, i ≠ j → fare i j ≠ fare j i)
  (distinct_fares : ∀ i j k l: Island, i ≠ j ∧ k ≠ l → fare i j ≠ fare k l)
  (connected : ∀ i j : Island, i ≠ j → fare i j = fare j i)

theorem traveler_journey_possible {Island : Type} (arch : Archipelago Island) :
  ∃ (t : Island) (seq : List (Island × Island)), -- there exists a starting island and a sequence of journeys
    seq.length = arch.n - 1 ∧                   -- length of the sequence is n-1
    (∀ i j, (i, j) ∈ seq → j ≠ i ∧ arch.fare i j < arch.fare j i) := -- fare decreases with each journey
sorry

end traveler_journey_possible_l39_39885


namespace brendan_fish_caught_afternoon_l39_39527

theorem brendan_fish_caught_afternoon (morning_fish : ℕ) (thrown_fish : ℕ) (dads_fish : ℕ) (total_fish : ℕ) :
  morning_fish = 8 → thrown_fish = 3 → dads_fish = 13 → total_fish = 23 → 
  (morning_fish - thrown_fish) + dads_fish + brendan_afternoon_catch = total_fish → 
  brendan_afternoon_catch = 5 :=
by
  intros morning_fish_eq thrown_fish_eq dads_fish_eq total_fish_eq fish_sum_eq
  sorry

end brendan_fish_caught_afternoon_l39_39527


namespace friends_meet_first_time_at_4pm_l39_39973

def lcm_four_times (a b c d : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm a b) (Nat.lcm c d)

def first_meeting_time (start_time_minutes: ℕ) (lap_anna lap_stephanie lap_james lap_carlos: ℕ) : ℕ :=
  start_time_minutes + lcm_four_times lap_anna lap_stephanie lap_james lap_carlos

theorem friends_meet_first_time_at_4pm :
  first_meeting_time 600 5 8 9 12 = 960 :=
by
  -- where 600 represents 10:00 AM in minutes since midnight and 960 represents 4:00 PM
  sorry

end friends_meet_first_time_at_4pm_l39_39973


namespace gdp_scientific_notation_l39_39832

theorem gdp_scientific_notation : 
  (33.5 * 10^12 = 3.35 * 10^13) := 
by
  sorry

end gdp_scientific_notation_l39_39832


namespace atomic_number_R_l39_39586

noncomputable def atomic_number_Pb := 82
def electron_shell_difference := 32

def same_group_atomic_number 
  (atomic_number_Pb : ℕ) 
  (electron_shell_difference : ℕ) : 
  ℕ := 
  atomic_number_Pb + electron_shell_difference

theorem atomic_number_R (R : ℕ) : 
  same_group_atomic_number atomic_number_Pb electron_shell_difference = 114 := 
by
  sorry

end atomic_number_R_l39_39586


namespace calculate_value_l39_39124

theorem calculate_value : (2 / 3 : ℝ)^0 + Real.log 2 + Real.log 5 = 2 :=
by 
  sorry

end calculate_value_l39_39124


namespace sum_first_n_terms_l39_39303

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_first_n_terms
  (a : ℕ → ℝ)
  (h_seq : arithmetic_sequence a)
  (h_a2a4 : a 2 + a 4 = 8)
  (h_common_diff : ∀ n : ℕ, a (n + 1) = a n + 2) :
  ∃ S_n : ℕ → ℝ, ∀ n : ℕ, S_n n = n^2 - n :=
by 
  sorry

end sum_first_n_terms_l39_39303


namespace transaction_gain_per_year_l39_39578

noncomputable def principal : ℝ := 9000
noncomputable def time : ℝ := 2
noncomputable def rate_lending : ℝ := 6
noncomputable def rate_borrowing : ℝ := 4

noncomputable def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * R * T / 100

noncomputable def total_interest_earned := simple_interest principal rate_lending time
noncomputable def total_interest_paid := simple_interest principal rate_borrowing time

noncomputable def total_gain := total_interest_earned - total_interest_paid
noncomputable def gain_per_year := total_gain / 2

theorem transaction_gain_per_year : gain_per_year = 180 :=
by
  sorry

end transaction_gain_per_year_l39_39578


namespace gcd_polynomial_eval_l39_39316

theorem gcd_polynomial_eval (b : ℤ) (h : ∃ (k : ℤ), b = 570 * k) :
  Int.gcd (4 * b ^ 3 + b ^ 2 + 5 * b + 95) b = 95 := by
  sorry

end gcd_polynomial_eval_l39_39316


namespace problem_statement_l39_39218

   noncomputable def g (x : ℝ) : ℝ := (3 * x + 4) / (x + 3)

   def T := {y : ℝ | ∃ (x : ℝ), x ≥ 0 ∧ y = g x}

   theorem problem_statement :
     (∃ N, (∀ y ∈ T, y ≤ N) ∧ N = 3 ∧ N ∉ T) ∧
     (∃ n, (∀ y ∈ T, y ≥ n) ∧ n = 4/3 ∧ n ∈ T) :=
   by
     sorry
   
end problem_statement_l39_39218


namespace sin_squared_identity_l39_39833

theorem sin_squared_identity :
  1 - 2 * (Real.sin (105 * Real.pi / 180))^2 = - (Real.sqrt 3) / 2 :=
by sorry

end sin_squared_identity_l39_39833


namespace probability_XOXOX_l39_39237

theorem probability_XOXOX (n_X n_O n_total : ℕ) (h_total : n_X + n_O = n_total)
  (h_X : n_X = 3) (h_O : n_O = 2) (h_total' : n_total = 5) :
  (1 / ↑(Nat.choose n_total n_X)) = (1 / 10) :=
by
  sorry

end probability_XOXOX_l39_39237


namespace min_box_height_l39_39203

noncomputable def height_of_box (x : ℝ) := x + 4

def surface_area (x : ℝ) : ℝ := 2 * x^2 + 4 * x * (x + 4)

theorem min_box_height (x h : ℝ) (h₁ : h = height_of_box x) (h₂ : surface_area x ≥ 130) : h ≥ 25 / 3 :=
by sorry

end min_box_height_l39_39203


namespace ratio_songs_kept_to_deleted_l39_39690

theorem ratio_songs_kept_to_deleted (initial_songs deleted_songs kept_songs : ℕ) 
  (h_initial : initial_songs = 54) (h_deleted : deleted_songs = 9) (h_kept : kept_songs = initial_songs - deleted_songs) :
  (kept_songs : ℚ) / (deleted_songs : ℚ) = 5 / 1 :=
by
  sorry

end ratio_songs_kept_to_deleted_l39_39690


namespace hyperbola_eccentricity_is_4_l39_39152

noncomputable def hyperbola_eccentricity (a b c : ℝ) (h_eq1 : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (h_eq2 : ∀ y : ℝ, y^2 = 16 * (4 : ℝ))
  (h_focus : c = 4)
: ℝ := c / a

theorem hyperbola_eccentricity_is_4 (a b c : ℝ)
  (h_eq1 : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (h_eq2 : ∀ y : ℝ, y^2 = 16 * (4 : ℝ))
  (h_focus : c = 4)
  (h_c2 : c^2 = a^2 + b^2)
  (h_bc : b^2 = a^2 * (c^2 / a^2 - 1))
: hyperbola_eccentricity a b c h_eq1 h_eq2 h_focus = 4 := by
  sorry

end hyperbola_eccentricity_is_4_l39_39152


namespace solution_set_of_inequality_l39_39663

open Set Real

theorem solution_set_of_inequality :
  {x : ℝ | sqrt (x + 3) > 3 - x} = {x : ℝ | 1 < x} ∪ {x : ℝ | x ≥ 3} := by
  sorry

end solution_set_of_inequality_l39_39663


namespace test_question_count_l39_39810

theorem test_question_count :
  ∃ (x y : ℕ), x + y = 30 ∧ 5 * x + 10 * y = 200 ∧ x = 20 :=
by
  sorry

end test_question_count_l39_39810


namespace four_digit_numbers_count_l39_39567

theorem four_digit_numbers_count : (3:ℕ) ^ 4 = 81 := by
  sorry

end four_digit_numbers_count_l39_39567


namespace min_a_value_l39_39466

theorem min_a_value 
  (a x y : ℤ) 
  (h1 : x - y^2 = a) 
  (h2 : y - x^2 = a) 
  (h3 : x ≠ y) 
  (h4 : |x| ≤ 10) : 
  a = -111 :=
sorry

end min_a_value_l39_39466


namespace product_of_cubes_eq_l39_39920

theorem product_of_cubes_eq :
  ( (3^3 - 1) / (3^3 + 1) ) * 
  ( (4^3 - 1) / (4^3 + 1) ) * 
  ( (5^3 - 1) / (5^3 + 1) ) * 
  ( (6^3 - 1) / (6^3 + 1) ) * 
  ( (7^3 - 1) / (7^3 + 1) ) * 
  ( (8^3 - 1) / (8^3 + 1) ) 
  = 73 / 256 :=
by
  sorry

end product_of_cubes_eq_l39_39920


namespace integral_x_squared_l39_39540

theorem integral_x_squared:
  ∫ x in (0:ℝ)..(1:ℝ), x^2 = 1/3 :=
by
  sorry

end integral_x_squared_l39_39540


namespace cube_surface_area_l39_39508

noncomputable def total_surface_area_of_cube (Q : ℝ) : ℝ :=
  8 * Q * Real.sqrt 3 / 3

theorem cube_surface_area (Q : ℝ) (h : Q > 0) :
  total_surface_area_of_cube Q = 8 * Q * Real.sqrt 3 / 3 :=
sorry

end cube_surface_area_l39_39508


namespace total_candies_l39_39004

-- Condition definitions
def lindaCandies : ℕ := 34
def chloeCandies : ℕ := 28

-- Proof statement to show their total candies
theorem total_candies : lindaCandies + chloeCandies = 62 := 
by
  sorry

end total_candies_l39_39004


namespace increasing_interval_of_f_on_0_pi_l39_39671

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 4)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.cos (2 * x - Real.pi / 4)

theorem increasing_interval_of_f_on_0_pi {ω : ℝ} (hω : ω > 0)
  (h_symmetry : ∀ x, f ω x = g x) :
  {x : ℝ | 0 ≤ x ∧ x ≤ Real.pi ∧ ∀ x1 x2, (0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ Real.pi) → f ω x1 < f ω x2} = 
  {x : ℝ | 0 ≤ x ∧ x ≤ Real.pi / 8} :=
sorry

end increasing_interval_of_f_on_0_pi_l39_39671


namespace find_prices_maximize_profit_l39_39631

-- Definition of conditions
def sales_eq1 (m n : ℝ) : Prop := 150 * m + 100 * n = 1450
def sales_eq2 (m n : ℝ) : Prop := 200 * m + 50 * n = 1100

def profit_function (x : ℕ) : ℝ := -2 * x + 1500
def range_x (x : ℕ) : Prop := 375 ≤ x ∧ x ≤ 500

-- Theorem to prove the unit prices
theorem find_prices : ∃ m n : ℝ, sales_eq1 m n ∧ sales_eq2 m n ∧ m = 3 ∧ n = 10 := 
sorry

-- Theorem to prove the profit function and maximum profit
theorem maximize_profit : ∃ (x : ℕ) (W : ℝ), range_x x ∧ W = profit_function x ∧ W = 750 :=
sorry

end find_prices_maximize_profit_l39_39631


namespace tax_on_clothing_l39_39494

variable (T : ℝ)
variable (c : ℝ := 0.45 * T)
variable (f : ℝ := 0.45 * T)
variable (o : ℝ := 0.10 * T)
variable (x : ℝ)
variable (t_c : ℝ := x / 100 * c)
variable (t_f : ℝ := 0)
variable (t_o : ℝ := 0.10 * o)
variable (t : ℝ := 0.0325 * T)

theorem tax_on_clothing :
  t_c + t_o = t → x = 5 :=
by
  sorry

end tax_on_clothing_l39_39494


namespace license_plate_count_l39_39104

-- Define the conditions as constants
def even_digit_count : Nat := 5
def consonant_count : Nat := 20
def vowel_count : Nat := 6

-- Define the problem as a theorem to prove
theorem license_plate_count : even_digit_count * consonant_count * vowel_count * consonant_count = 12000 := 
by
  -- The proof is not required, so we leave it as sorry
  sorry

end license_plate_count_l39_39104


namespace num_female_fox_terriers_l39_39615

def total_dogs : Nat := 2012
def total_female_dogs : Nat := 1110
def total_fox_terriers : Nat := 1506
def male_shih_tzus : Nat := 202

theorem num_female_fox_terriers :
    ∃ (female_fox_terriers: Nat), 
        female_fox_terriers = total_fox_terriers - (total_dogs - total_female_dogs - male_shih_tzus) := by
    sorry

end num_female_fox_terriers_l39_39615


namespace simplify_expression_l39_39385

theorem simplify_expression (y : ℝ) : 5 * y + 7 * y + 8 * y = 20 * y :=
by
  sorry

end simplify_expression_l39_39385


namespace find_sum_of_cubes_l39_39149

noncomputable def roots_of_polynomial := 
  ∃ a b c : ℝ, 
    (6 * a^3 + 500 * a + 1001 = 0) ∧ 
    (6 * b^3 + 500 * b + 1001 = 0) ∧ 
    (6 * c^3 + 500 * c + 1001 = 0)

theorem find_sum_of_cubes (a b c : ℝ) 
  (h : roots_of_polynomial) : 
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 500.5 := 
sorry

end find_sum_of_cubes_l39_39149


namespace lab_tech_ratio_l39_39654

theorem lab_tech_ratio (U T C : ℕ) (hU : U = 12) (hC : C = 6 * U) (hT : T = (C + U) / 14) :
  (T : ℚ) / U = 1 / 2 :=
by
  sorry

end lab_tech_ratio_l39_39654


namespace range_of_a_l39_39642

-- Define the problem statement in Lean 4
theorem range_of_a (a : ℝ) : (∀ x : ℝ, ((x^2 - (a-1)*x + 1) > 0)) → (-1 < a ∧ a < 3) :=
by
  intro h
  sorry -- Proof to be filled in

end range_of_a_l39_39642


namespace percent_decrease_l39_39791

theorem percent_decrease (original_price sale_price : ℝ) 
  (h_original: original_price = 100) 
  (h_sale: sale_price = 75) : 
  (original_price - sale_price) / original_price * 100 = 25 :=
by
  sorry

end percent_decrease_l39_39791


namespace find_length_of_PB_l39_39244

theorem find_length_of_PB
  (PA : ℝ) -- Define PA
  (h_PA : PA = 4) -- Condition PA = 4
  (PB : ℝ) -- Define PB
  (PT : ℝ) -- Define PT
  (h_PT : PT = PB - 2 * PA) -- Condition PT = PB - 2 * PA
  (h_power_of_a_point : PA * PB = PT^2) -- Condition PA * PB = PT^2
  : PB = 16 :=
sorry

end find_length_of_PB_l39_39244


namespace perpendicular_vectors_X_value_l39_39667

open Real

-- Define vectors a and b, and their perpendicularity condition
def vector_a (x : ℝ) : ℝ × ℝ := (x, x + 1)
def vector_b : ℝ × ℝ := (1, 2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The theorem statement
theorem perpendicular_vectors_X_value (x : ℝ) 
  (h : dot_product (vector_a x) vector_b = 0) : 
    x = -2 / 3 :=
by sorry

end perpendicular_vectors_X_value_l39_39667


namespace find_dot_AP_BC_l39_39841

-- Defining the lengths of the sides of the triangle.
def length_AB : ℝ := 13
def length_BC : ℝ := 14
def length_CA : ℝ := 15

-- Defining the provided dot product conditions at point P.
def dot_BP_CA : ℝ := 18
def dot_CP_BA : ℝ := 32

-- The target is to prove the final dot product.
theorem find_dot_AP_BC :
  ∃ (AP BC : ℝ), BC = 14 → dot_BP_CA = 18 → dot_CP_BA = 32 → (AP * BC = 14) :=
by
  -- proof goes here
  sorry

end find_dot_AP_BC_l39_39841


namespace problem_part1_problem_part2_problem_part3_l39_39473

noncomputable def given_quadratic (x : ℝ) (m : ℝ) : ℝ := 2 * x^2 - (Real.sqrt 3 + 1) * x + m

noncomputable def sin_cos_eq_quadratic_roots (θ m : ℝ) : Prop := 
  let sinθ := Real.sin θ
  let cosθ := Real.cos θ
  given_quadratic sinθ m = 0 ∧ given_quadratic cosθ m = 0

theorem problem_part1 (θ : ℝ) (h : 0 < θ ∧ θ < 2 * Real.pi) (Hroots : sin_cos_eq_quadratic_roots θ m) : 
  (Real.sin θ / (1 - Real.cos θ) + Real.cos θ / (1 - Real.tan θ)) = (3 + 5 * Real.sqrt 3) / 4 :=
sorry

theorem problem_part2 {θ : ℝ} (h : 0 < θ ∧ θ < 2 * Real.pi) (Hroots : sin_cos_eq_quadratic_roots θ m) : 
  m = Real.sqrt 3 / 4 :=
sorry

theorem problem_part3 (m : ℝ) (sinθ1 cosθ1 sinθ2 cosθ2 : ℝ) (θ1 θ2 : ℝ)
  (H1 : sinθ1 = Real.sqrt 3 / 2 ∧ cosθ1 = 1 / 2 ∧ θ1 = Real.pi / 3)
  (H2 : sinθ2 = 1 / 2 ∧ cosθ2 = Real.sqrt 3 / 2 ∧ θ2 = Real.pi / 6) : 
  ∃ θ, sin_cos_eq_quadratic_roots θ m ∧ 
       (Real.sin θ = sinθ1 ∧ Real.cos θ = cosθ1 ∨ Real.sin θ = sinθ2 ∧ Real.cos θ = cosθ2) :=
sorry

end problem_part1_problem_part2_problem_part3_l39_39473


namespace intersection_M_N_l39_39802

noncomputable def set_M : Set ℚ := {α | ∃ k : ℤ, α = k * 90 - 36}
noncomputable def set_N : Set ℚ := {α | -180 < α ∧ α < 180}

theorem intersection_M_N : set_M ∩ set_N = {-36, 54, 144, -126} := by
  sorry

end intersection_M_N_l39_39802


namespace triangle_geometric_sequence_sine_rule_l39_39020

noncomputable def sin60 : Real := Real.sqrt 3 / 2

theorem triangle_geometric_sequence_sine_rule 
  {a b c : Real} 
  {A B C : Real} 
  (h1 : a / b = b / c) 
  (h2 : A = 60 * Real.pi / 180) :
  b * Real.sin B / c = Real.sqrt 3 / 2 :=
by
  sorry

end triangle_geometric_sequence_sine_rule_l39_39020


namespace nonneg_reals_inequality_l39_39250

theorem nonneg_reals_inequality (a b c d : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d) (h₄ : a^2 + b^2 + c^2 + d^2 = 1) : 
  a + b + c + d - 1 ≥ 16 * a * b * c * d := 
by 
  sorry

end nonneg_reals_inequality_l39_39250


namespace xyz_identity_l39_39840

theorem xyz_identity (x y z : ℝ) (h : x + y + z = x * y * z) :
  x * (1 - y^2) * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) = 4 * x * y * z :=
by
  sorry

end xyz_identity_l39_39840


namespace circles_tangent_area_l39_39195

noncomputable def triangle_area (r1 r2 r3 : ℝ) := 
  let d1 := r1 + r2
  let d2 := r2 + r3
  let d3 := r1 + r3
  let s := (d1 + d2 + d3) / 2
  (s * (s - d1) * (s - d2) * (s - d3)).sqrt

theorem circles_tangent_area :
  let r1 := 5
  let r2 := 12
  let r3 := 13
  let area := triangle_area r1 r2 r3 / (4 * (r1 + r2 + r3)).sqrt
  area = 120 / 25 := 
by 
  sorry

end circles_tangent_area_l39_39195


namespace triangle_perimeter_l39_39764

theorem triangle_perimeter
  (a b : ℕ) (c : ℕ) 
  (h_side1 : a = 3)
  (h_side2 : b = 4)
  (h_third_side : c^2 - 13 * c + 40 = 0)
  (h_valid_triangle : (a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :
  a + b + c = 12 :=
by {
  sorry
}

end triangle_perimeter_l39_39764


namespace michelle_gas_left_l39_39761

def gasLeft (initialGas: ℝ) (usedGas: ℝ) : ℝ :=
  initialGas - usedGas

theorem michelle_gas_left :
  gasLeft 0.5 0.3333333333333333 = 0.1666666666666667 :=
by
  -- proof goes here
  sorry

end michelle_gas_left_l39_39761


namespace johnny_distance_walked_l39_39570

theorem johnny_distance_walked
  (dist_q_to_y : ℕ) (matthew_rate : ℕ) (johnny_rate : ℕ) (time_diff : ℕ) (johnny_walked : ℕ):
  dist_q_to_y = 45 →
  matthew_rate = 3 →
  johnny_rate = 4 →
  time_diff = 1 →
  (∃ t: ℕ, johnny_walked = johnny_rate * t 
            ∧ dist_q_to_y = matthew_rate * (t + time_diff) + johnny_walked) →
  johnny_walked = 24 := by
  sorry

end johnny_distance_walked_l39_39570


namespace population_reaches_target_l39_39817

def initial_year : ℕ := 2020
def initial_population : ℕ := 450
def growth_period : ℕ := 25
def growth_factor : ℕ := 3
def target_population : ℕ := 10800

theorem population_reaches_target : ∃ (year : ℕ), year - initial_year = 3 * growth_period ∧ (initial_population * growth_factor ^ 3) >= target_population := by
  sorry

end population_reaches_target_l39_39817


namespace total_telephone_bill_second_month_l39_39210

variable (F C : ℝ)

-- Elvin's total telephone bill for January is 40 dollars
axiom january_bill : F + C = 40

-- The charge for calls in the second month is twice the charge for calls in January
axiom second_month_call_charge : ∃ C2, C2 = 2 * C

-- Proof that the total telephone bill for the second month is 40 + C
theorem total_telephone_bill_second_month : 
  ∃ S, S = F + 2 * C ∧ S = 40 + C :=
sorry

end total_telephone_bill_second_month_l39_39210


namespace abc_sum_l39_39792

def f (x : Int) (a b c : Nat) : Int :=
  if x > 0 then a * x + 4
  else if x = 0 then a * b
  else b * x + c

theorem abc_sum :
  ∃ a b c : Nat, 
  f 3 a b c = 7 ∧ 
  f 0 a b c = 6 ∧ 
  f (-3) a b c = -15 ∧ 
  a + b + c = 10 :=
by
  sorry

end abc_sum_l39_39792


namespace restaurant_customer_problem_l39_39708

theorem restaurant_customer_problem (x y z : ℕ) 
  (h1 : x = 2 * z)
  (h2 : y = x - 3)
  (h3 : 3 + x + y - z = 8) :
  x = 6 ∧ y = 3 ∧ z = 3 ∧ (x + y = 9) :=
by
  sorry

end restaurant_customer_problem_l39_39708


namespace part2_l39_39457

noncomputable def f (a x : ℝ) : ℝ := Real.log x - a * x

theorem part2 (x1 x2 : ℝ) (h1 : x1 < x2) (h2 : f 1 x1 = f 1 x2) : x1 + x2 > 2 := by
  have f_x1 := h2
  sorry

end part2_l39_39457


namespace number_of_lattice_points_l39_39171

theorem number_of_lattice_points (A B : ℝ) (h : B - A = 10) :
  ∃ n, n = 10 ∨ n = 11 :=
sorry

end number_of_lattice_points_l39_39171


namespace find_B_age_l39_39199

variable (a b c : ℕ)

def problem_conditions : Prop :=
  a = b + 2 ∧ b = 2 * c ∧ a + b + c = 22

theorem find_B_age (h : problem_conditions a b c) : b = 8 :=
by {
  sorry
}

end find_B_age_l39_39199


namespace triangle_leg_ratio_l39_39372

theorem triangle_leg_ratio :
  ∀ (a b : ℝ) (h₁ : a = 4) (h₂ : b = 2 * Real.sqrt 5),
    ((a / b) = (2 * Real.sqrt 5) / 5) :=
by
  intros a b h₁ h₂
  sorry

end triangle_leg_ratio_l39_39372


namespace lowest_possible_students_l39_39682

-- Definitions based on conditions
def isDivisibleBy (n m : ℕ) : Prop := n % m = 0

def canBeDividedIntoTeams (num_students num_teams : ℕ) : Prop := isDivisibleBy num_students num_teams

-- Theorem statement for the lowest possible number of students
theorem lowest_possible_students (n : ℕ) : 
  (canBeDividedIntoTeams n 8) ∧ (canBeDividedIntoTeams n 12) → n = 24 := by
  sorry

end lowest_possible_students_l39_39682


namespace marla_errand_total_time_l39_39917

theorem marla_errand_total_time :
  let d1 := 20 -- Driving to son's school
  let b := 30  -- Taking a bus to the grocery store
  let s := 15  -- Shopping at the grocery store
  let w := 10  -- Walking to the gas station
  let g := 5   -- Filling up gas
  let r := 25  -- Riding a bicycle to the school
  let p := 70  -- Attending parent-teacher night
  let c := 30  -- Catching up with a friend at a coffee shop
  let sub := 40-- Taking the subway home
  let d2 := 20 -- Driving home
  d1 + b + s + w + g + r + p + c + sub + d2 = 265 := by
  sorry

end marla_errand_total_time_l39_39917


namespace solve_x_l39_39192

theorem solve_x (x : ℝ) (h : (x / 3) / 5 = 5 / (x / 3)) : x = 15 ∨ x = -15 :=
by sorry

end solve_x_l39_39192


namespace remainder_division_l39_39648

theorem remainder_division (L S R : ℕ) (h1 : L - S = 1325) (h2 : L = 1650) (h3 : L = 5 * S + R) : 
  R = 25 :=
sorry

end remainder_division_l39_39648


namespace spent_on_video_game_l39_39197

def saved_September : ℕ := 30
def saved_October : ℕ := 49
def saved_November : ℕ := 46
def money_left : ℕ := 67
def total_saved := saved_September + saved_October + saved_November

theorem spent_on_video_game : total_saved - money_left = 58 := by
  -- proof steps go here
  sorry

end spent_on_video_game_l39_39197


namespace sum_of_third_terms_arithmetic_progressions_l39_39146

theorem sum_of_third_terms_arithmetic_progressions
  (a : ℕ → ℕ) (b : ℕ → ℕ)
  (d1 d2 : ℕ)
  (h1 : ∃ d1 : ℕ, ∀ n : ℕ, a (n + 1) = a 1 + n * d1)
  (h2 : ∃ d2 : ℕ, ∀ n : ℕ, b (n + 1) = b 1 + n * d2)
  (h3 : a 1 + b 1 = 7)
  (h4 : a 5 + b 5 = 35) :
  a 3 + b 3 = 21 :=
by
  sorry

end sum_of_third_terms_arithmetic_progressions_l39_39146


namespace value_of_x2_plus_9y2_l39_39754

theorem value_of_x2_plus_9y2 (x y : ℝ) (h1 : x + 3 * y = 9) (h2 : x * y = -15) : x^2 + 9 * y^2 = 171 :=
sorry

end value_of_x2_plus_9y2_l39_39754


namespace speed_of_car_in_second_hour_l39_39014

noncomputable def speed_in_first_hour : ℝ := 90
noncomputable def average_speed : ℝ := 82.5
noncomputable def total_time : ℝ := 2

theorem speed_of_car_in_second_hour : 
  ∃ (speed_in_second_hour : ℝ), 
  (speed_in_first_hour + speed_in_second_hour) / total_time = average_speed ∧ 
  speed_in_first_hour = 90 ∧ 
  average_speed = 82.5 → 
  speed_in_second_hour = 75 :=
by 
  sorry

end speed_of_car_in_second_hour_l39_39014


namespace hours_per_toy_l39_39500

-- Defining the conditions
def toys_produced (hours: ℕ) : ℕ := 40 
def hours_worked : ℕ := 80

-- Theorem: If a worker makes 40 toys in 80 hours, then it takes 2 hours to make one toy.
theorem hours_per_toy : (hours_worked / toys_produced hours_worked) = 2 :=
by
  sorry

end hours_per_toy_l39_39500


namespace hyperbola_center_l39_39999

theorem hyperbola_center (x y : ℝ) :
  ( ∃ (h k : ℝ), ∀ (x y : ℝ), (4 * x - 8)^2 / 9^2 - (5 * y - 15)^2 / 7^2 = 1 → (h, k) = (2, 3) ) :=
by
  existsi 2
  existsi 3
  intros x y h
  sorry

end hyperbola_center_l39_39999


namespace find_length_DE_l39_39383

theorem find_length_DE (AB AC BC : ℝ) (angleA : ℝ) 
                         (DE DF EF : ℝ) (angleD : ℝ) :
  AB = 9 → AC = 11 → BC = 7 →
  angleA = 60 → DE = 3 → DF = 5.5 → EF = 2.5 →
  angleD = 60 →
  DE = 9 * 2.5 / 7 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end find_length_DE_l39_39383


namespace at_least_one_ge_two_l39_39206

theorem at_least_one_ge_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    max (a + 1/b) (max (b + 1/c) (c + 1/a)) ≥ 2 :=
sorry

end at_least_one_ge_two_l39_39206


namespace exact_fraction_difference_l39_39132

theorem exact_fraction_difference :
  let x := (8:ℚ) / 11
  let y := (18:ℚ) / 25 
  x - y = (2:ℚ) / 275 :=
by
  -- Definitions from conditions: x = 0.\overline{72} and y = 0.72
  let x := (8:ℚ) / 11
  let y := (18:ℚ) / 25 
  -- Goal is to prove the exact fraction difference
  show x - y = (2:ℚ) / 275
  sorry

end exact_fraction_difference_l39_39132


namespace power_mod_remainder_l39_39891

theorem power_mod_remainder 
  (h1 : 7^2 % 17 = 15)
  (h2 : 15 % 17 = -2 % 17)
  (h3 : 2^4 % 17 = -1 % 17)
  (h4 : 1011 % 2 = 1) :
  7^2023 % 17 = 12 := 
  sorry

end power_mod_remainder_l39_39891


namespace probability_Cecilia_rolls_4_given_win_l39_39037

noncomputable def P_roll_Cecilia_4_given_win : ℚ :=
  let P_C1_4 := 1/6
  let P_W_C := 1/5
  let P_W_C_given_C1_4 := (4/6)^4
  let P_C1_4_and_W_C := P_C1_4 * P_W_C_given_C1_4
  let P_C1_4_given_W_C := P_C1_4_and_W_C / P_W_C
  P_C1_4_given_W_C

theorem probability_Cecilia_rolls_4_given_win :
  P_roll_Cecilia_4_given_win = 256 / 1555 :=
by 
  -- Here the proof would go, but we include sorry for now.
  sorry

end probability_Cecilia_rolls_4_given_win_l39_39037


namespace smaller_two_digit_product_l39_39811

-- Statement of the problem
theorem smaller_two_digit_product (a b : ℕ) (h : a * b = 4536) (h_a : 10 ≤ a ∧ a < 100) (h_b : 10 ≤ b ∧ b < 100) : a = 21 ∨ b = 21 := 
by {
  sorry -- proof not needed
}

end smaller_two_digit_product_l39_39811


namespace k_positive_first_third_quadrants_l39_39169

theorem k_positive_first_third_quadrants (k : ℝ) (hk : k ≠ 0) :
  (∀ x : ℝ, (x > 0 → k*x > 0) ∧ (x < 0 → k*x < 0)) → k > 0 :=
by
  sorry

end k_positive_first_third_quadrants_l39_39169


namespace pet_store_animals_left_l39_39538

theorem pet_store_animals_left (initial_birds initial_puppies initial_cats initial_spiders initial_snakes : ℕ)
  (donation_fraction snakes_share_sold birds_sold puppies_adopted cats_transferred kittens_brought : ℕ)
  (spiders_loose spiders_captured : ℕ)
  (H_initial_birds : initial_birds = 12)
  (H_initial_puppies : initial_puppies = 9)
  (H_initial_cats : initial_cats = 5)
  (H_initial_spiders : initial_spiders = 15)
  (H_initial_snakes : initial_snakes = 8)
  (H_donation_fraction : donation_fraction = 25)
  (H_snakes_share_sold : snakes_share_sold = (donation_fraction * initial_snakes) / 100)
  (H_birds_sold : birds_sold = initial_birds / 2)
  (H_puppies_adopted : puppies_adopted = 3)
  (H_cats_transferred : cats_transferred = 4)
  (H_kittens_brought : kittens_brought = 2)
  (H_spiders_loose : spiders_loose = 7)
  (H_spiders_captured : spiders_captured = 5) :
  (initial_snakes - snakes_share_sold) + (initial_birds - birds_sold) + 
  (initial_puppies - puppies_adopted) + (initial_cats - cats_transferred + kittens_brought) + 
  (initial_spiders - (spiders_loose - spiders_captured)) = 34 := 
by 
  sorry

end pet_store_animals_left_l39_39538


namespace first_stack_height_l39_39468

theorem first_stack_height (x : ℕ) (h1 : x + (x + 2) + (x - 3) + (x + 2) = 21) : x = 5 :=
by
  sorry

end first_stack_height_l39_39468


namespace find_zero_function_l39_39984

noncomputable def satisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x ^ 714 + y) = f (x ^ 2019) + f (y ^ 122)

theorem find_zero_function (f : ℝ → ℝ) (h : satisfiesCondition f) :
  ∀ x : ℝ, f x = 0 :=
sorry

end find_zero_function_l39_39984


namespace equation_of_line_AB_l39_39292

noncomputable def circle_center : ℝ × ℝ := (1, 0)  -- center of the circle (x-1)^2 + y^2 = 1
noncomputable def circle_radius : ℝ := 1          -- radius of the circle (x-1)^2 + y^2 = 1
noncomputable def point_P : ℝ × ℝ := (3, 1)       -- point P(3,1)

theorem equation_of_line_AB :
  ∃ (AB : ℝ → ℝ → Prop),
    (∀ x y, AB x y ↔ (2 * x + y - 3 = 0)) := sorry

end equation_of_line_AB_l39_39292


namespace sum_digits_increment_l39_39046

def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_digits_increment (n : ℕ) (h : sum_digits n = 1365) : 
  sum_digits (n + 1) = 1360 :=
by
  sorry

end sum_digits_increment_l39_39046


namespace complex_number_is_purely_imaginary_l39_39059

theorem complex_number_is_purely_imaginary (a : ℂ) : 
  (a^2 - a - 2 = 0) ∧ (a^2 - 3*a + 2 ≠ 0) ↔ a = -1 :=
by 
  sorry

end complex_number_is_purely_imaginary_l39_39059


namespace smallest_positive_solution_l39_39168

theorem smallest_positive_solution :
  ∃ x : ℝ, x > 0 ∧ (x ^ 4 - 50 * x ^ 2 + 576 = 0) ∧ (∀ y : ℝ, y > 0 ∧ y ^ 4 - 50 * y ^ 2 + 576 = 0 → x ≤ y) ∧ x = 3 * Real.sqrt 2 :=
sorry

end smallest_positive_solution_l39_39168


namespace workers_in_first_group_l39_39458

-- Define the first condition: Some workers collect 48 kg of cotton in 4 days
def cotton_collected_by_W_workers_in_4_days (W : ℕ) : ℕ := 48

-- Define the second condition: 9 workers collect 72 kg of cotton in 2 days
def cotton_collected_by_9_workers_in_2_days : ℕ := 72

-- Define the rate of cotton collected per worker per day for both scenarios
def rate_per_worker_first_group (W : ℕ) : ℕ :=
cotton_collected_by_W_workers_in_4_days W / (W * 4)

def rate_per_worker_second_group : ℕ :=
cotton_collected_by_9_workers_in_2_days / (9 * 2)

-- Given the rates are the same for both groups, prove W = 3
theorem workers_in_first_group (W : ℕ) (h : rate_per_worker_first_group W = rate_per_worker_second_group) : W = 3 :=
sorry

end workers_in_first_group_l39_39458


namespace sum_of_triangulars_iff_sum_of_squares_l39_39455

-- Definitions of triangular numbers and sums of squares
def isTriangular (n : ℕ) : Prop := ∃ k, n = k * (k + 1) / 2
def isSumOfTwoTriangulars (m : ℕ) : Prop := ∃ x y, m = (x * (x + 1) / 2) + (y * (y + 1) / 2)
def isSumOfTwoSquares (n : ℕ) : Prop := ∃ a b, n = a * a + b * b

-- Main theorem statement
theorem sum_of_triangulars_iff_sum_of_squares (m : ℕ) (h_pos : 0 < m) : 
  isSumOfTwoTriangulars m ↔ isSumOfTwoSquares (4 * m + 1) :=
sorry

end sum_of_triangulars_iff_sum_of_squares_l39_39455


namespace equation_represents_point_l39_39479

theorem equation_represents_point (a b x y : ℝ) :
  x^2 + y^2 + 2 * a * x + 2 * b * y + a^2 + b^2 = 0 ↔ x = -a ∧ y = -b := 
by sorry

end equation_represents_point_l39_39479


namespace part1_part2_part3_l39_39213

section ShoppingMall

variable (x y a b : ℝ)
variable (cpaA spaA cpaB spaB : ℝ)
variable (n total_y yuan : ℝ)

-- Conditions given in the problem
def cost_price_A := 160
def selling_price_A := 220
def cost_price_B := 120
def selling_price_B := 160
def total_clothing := 100
def min_A_clothing := 60
def max_budget := 15000
def discount_diff := 4
def max_profit_with_discount := 4950

-- Definitions applied from conditions
def profit_per_piece_A := selling_price_A - cost_price_A
def profit_per_piece_B := selling_price_B - cost_price_B

-- Question 1: Functional relationship between y and x
theorem part1 : 
  (∀ (x : ℝ), x ≥ 0 → x ≤ total_clothing → 
  y = profit_per_piece_A * x + profit_per_piece_B * (total_clothing - x)) →
  y = 20 * x + 4000 := 
sorry

-- Question 2: Maximum profit under given cost constraints
theorem part2 : 
  (min_A_clothing ≤ x ∧ x ≤ 75 ∧ 
  (cost_price_A * x + cost_price_B * (total_clothing - x) ≤ max_budget)) →
  y = 20 * 75 + 4000 → 
  y = 5500 :=
sorry

-- Question 3: Determine a under max profit condition
theorem part3 : 
  (a - b = discount_diff ∧ 0 < a ∧ a < 20 ∧ 
  (20 - a) * 75 + 4000 + 100 * a - 400 = max_profit_with_discount) →
  a = 9 :=
sorry

end ShoppingMall

end part1_part2_part3_l39_39213


namespace correct_judgment_l39_39058

def P := Real.pi < 2
def Q := Real.pi > 3

theorem correct_judgment : (P ∨ Q) ∧ ¬P := by
  sorry

end correct_judgment_l39_39058


namespace largest_whole_number_l39_39396

theorem largest_whole_number (x : ℕ) : 8 * x < 120 → x ≤ 14 :=
by
  intro h
  -- prove the main statement here
  sorry

end largest_whole_number_l39_39396


namespace cos2alpha_plus_sin2alpha_l39_39571

def point_angle_condition (x y : ℝ) (r : ℝ) (α : ℝ) : Prop :=
  x = -3 ∧ y = 4 ∧ r = 5 ∧ x^2 + y^2 = r^2

theorem cos2alpha_plus_sin2alpha (α : ℝ) (x y r : ℝ)
  (h : point_angle_condition x y r α) : 
  (Real.cos (2 * α) + Real.sin (2 * α)) = -31/25 :=
by
  sorry

end cos2alpha_plus_sin2alpha_l39_39571


namespace problem_one_problem_two_l39_39444

-- Problem 1
theorem problem_one : -9 + 5 * (-6) - 18 / (-3) = -33 :=
by
  sorry

-- Problem 2
theorem problem_two : ((-3/4) - (5/8) + (9/12)) * (-24) + (-8) / (2/3) = -6 :=
by
  sorry

end problem_one_problem_two_l39_39444


namespace square_equiv_l39_39080

theorem square_equiv (x : ℝ) : 
  (7 - (x^3 - 49)^(1/3))^2 = 
  49 - 14 * (x^3 - 49)^(1/3) + ((x^3 - 49)^(1/3))^2 := 
by 
  sorry

end square_equiv_l39_39080


namespace computer_price_after_9_years_l39_39982

theorem computer_price_after_9_years 
  (initial_price : ℝ) (decrease_factor : ℝ) (years : ℕ) 
  (initial_price_eq : initial_price = 8100)
  (decrease_factor_eq : decrease_factor = 1 - 1/3)
  (years_eq : years = 9) :
  initial_price * (decrease_factor ^ (years / 3)) = 2400 := 
by
  sorry

end computer_price_after_9_years_l39_39982


namespace base_length_of_parallelogram_l39_39070

theorem base_length_of_parallelogram (area : ℝ) (base altitude : ℝ)
  (h1 : area = 98)
  (h2 : altitude = 2 * base) :
  base = 7 :=
by
  sorry

end base_length_of_parallelogram_l39_39070


namespace liza_phone_bill_eq_70_l39_39135

theorem liza_phone_bill_eq_70 (initial_balance rent payment paycheck electricity internet final_balance phone_bill : ℝ)
  (h1 : initial_balance = 800)
  (h2 : rent = 450)
  (h3 : paycheck = 1500)
  (h4 : electricity = 117)
  (h5 : internet = 100)
  (h6 : final_balance = 1563)
  (h_balance_before_phone_bill : initial_balance - rent + paycheck - (electricity + internet) = 1633)
  (h_final_balance_def : 1633 - phone_bill = final_balance) :
  phone_bill = 70 := sorry

end liza_phone_bill_eq_70_l39_39135


namespace sphere_cube_volume_ratio_l39_39170

theorem sphere_cube_volume_ratio (d a : ℝ) (h_d : d = 12) (h_a : a = 6) :
  let r := d / 2
  let V_sphere := (4 / 3) * π * r^3
  let V_cube := a^3
  V_sphere / V_cube = (4 * π) / 3 :=
by
  sorry

end sphere_cube_volume_ratio_l39_39170


namespace factor_quadratic_expression_l39_39462

theorem factor_quadratic_expression (a b : ℤ) (h: 25 * -198 = -4950 ∧ a + b = -195 ∧ a * b = -4950) : a + 2 * b = -420 :=
sorry

end factor_quadratic_expression_l39_39462


namespace neither_sufficient_nor_necessary_condition_l39_39750

theorem neither_sufficient_nor_necessary_condition
  (a1 b1 c1 a2 b2 c2 : ℝ) (h1 : a1 ≠ 0) (h2 : b1 ≠ 0) (h3 : c1 ≠ 0)
  (h4 : a2 ≠ 0) (h5 : b2 ≠ 0) (h6 : c2 ≠ 0) :
  (a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2) ↔
  ¬(∀ x, a1 * x^2 + b1 * x + c1 > 0 ↔ a2 * x^2 + b2 * x + c2 > 0) :=
sorry

end neither_sufficient_nor_necessary_condition_l39_39750


namespace volume_of_fifth_section_l39_39991

theorem volume_of_fifth_section
  (a : ℕ → ℚ)
  (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)  -- Arithmetic sequence constraint
  (h_sum_top_four : a 0 + a 1 + a 2 + a 3 = 3)  -- Sum of the top four sections
  (h_sum_bottom_three : a 6 + a 7 + a 8 = 4)  -- Sum of the bottom three sections
  : a 4 = 67 / 66 := sorry

end volume_of_fifth_section_l39_39991


namespace right_triangle_inequality_l39_39221

theorem right_triangle_inequality (a b c : ℝ) (h : a^2 + b^2 = c^2) :
  a^4 + b^4 < c^4 :=
by
  sorry

end right_triangle_inequality_l39_39221


namespace exists_n_consecutive_non_prime_or_prime_power_l39_39061

theorem exists_n_consecutive_non_prime_or_prime_power (n : ℕ) (h : n > 0) :
  ∃ (seq : Fin n → ℕ), (∀ i, ¬ (Nat.Prime (seq i)) ∧ ¬ (∃ p k : ℕ, p.Prime ∧ k > 1 ∧ seq i = p ^ k)) :=
by
  sorry

end exists_n_consecutive_non_prime_or_prime_power_l39_39061


namespace rahul_meena_work_together_l39_39022

theorem rahul_meena_work_together (days_rahul : ℚ) (days_meena : ℚ) (combined_days : ℚ) :
  days_rahul = 5 ∧ days_meena = 10 → combined_days = 10 / 3 :=
by
  intros h
  sorry

end rahul_meena_work_together_l39_39022


namespace part1_part2_l39_39532

theorem part1 (x : ℝ) (m : ℝ) :
  (∀ x : ℝ, |x + 2| + |x - 4| - m ≥ 0) ↔ m ≤ 6 :=
sorry

theorem part2 (a b : ℝ) (n : ℝ) :
  n = 6 → (a > 0 ∧ b > 0 ∧ (4 / (a + 5 * b)) + (1 / (3 * a + 2 * b)) = 1) → (4 * a + 7 * b) ≥ 9 :=
sorry

end part1_part2_l39_39532


namespace range_of_k_l39_39441

noncomputable def f (x k : ℝ) : ℝ := 2^x + 3*x - k

theorem range_of_k (k : ℝ) :
  (∃ x : ℝ, 1 ≤ x ∧ x < 2 ∧ f x k = 0) ↔ 5 ≤ k ∧ k < 10 :=
by sorry

end range_of_k_l39_39441


namespace handshake_problem_l39_39400

noncomputable def total_handshakes (num_companies : ℕ) (repr_per_company : ℕ) : ℕ :=
    let total_people := num_companies * repr_per_company
    let possible_handshakes_per_person := total_people - repr_per_company
    (total_people * possible_handshakes_per_person) / 2

theorem handshake_problem : total_handshakes 4 4 = 96 :=
by
  sorry

end handshake_problem_l39_39400


namespace cosine_of_half_pi_minus_double_alpha_l39_39412

theorem cosine_of_half_pi_minus_double_alpha (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.tan α = 2) :
  Real.cos (π / 2 - 2 * α) = 4 / 5 :=
sorry

end cosine_of_half_pi_minus_double_alpha_l39_39412


namespace total_distance_total_distance_alt_l39_39310

variable (D : ℝ) -- declare the variable for the total distance

-- defining the conditions
def speed_walking : ℝ := 4 -- speed in km/hr when walking
def speed_running : ℝ := 8 -- speed in km/hr when running
def total_time : ℝ := 3.75 -- total time in hours

-- proving that D = 10 given the conditions
theorem total_distance 
    (h1 : D / (2 * speed_walking) + D / (2 * speed_running) = total_time) : 
    D = 10 := 
sorry

-- Alternative theorem version declaring variables directly
theorem total_distance_alt
    (speed_walking speed_running total_time : ℝ) -- declaring variables
    (D : ℝ) -- the total distance
    (h1 : D / (2 * speed_walking) + D / (2 * speed_running) = total_time)
    (hw : speed_walking = 4)
    (hr : speed_running = 8)
    (ht : total_time = 3.75) : 
    D = 10 := 
sorry

end total_distance_total_distance_alt_l39_39310


namespace matthews_contribution_l39_39256

theorem matthews_contribution 
  (total_cost : ℝ) (yen_amount : ℝ) (conversion_rate : ℝ)
  (h1 : total_cost = 18)
  (h2 : yen_amount = 2500)
  (h3 : conversion_rate = 140) :
  (total_cost - (yen_amount / conversion_rate)) = 0.143 :=
by sorry

end matthews_contribution_l39_39256


namespace taller_tree_height_l39_39995

theorem taller_tree_height
  (h : ℕ)
  (h_shorter_ratio : h - 16 = (3 * h) / 4) : h = 64 := by
  sorry

end taller_tree_height_l39_39995


namespace andrew_daily_work_hours_l39_39998

theorem andrew_daily_work_hours (total_hours : ℝ) (days : ℝ) (h1 : total_hours = 7.5) (h2 : days = 3) : total_hours / days = 2.5 :=
by
  rw [h1, h2]
  norm_num

end andrew_daily_work_hours_l39_39998


namespace parabola_equation_l39_39871

theorem parabola_equation (x y : ℝ)
    (focus : x = 1 ∧ y = -2)
    (directrix : 5 * x + 2 * y = 10) :
    4 * x^2 - 20 * x * y + 25 * y^2 + 158 * x + 156 * y + 16 = 0 := 
by
  -- use the given conditions and intermediate steps to derive the final equation
  sorry

end parabola_equation_l39_39871


namespace birds_total_distance_l39_39493

-- Define the speeds of the birds
def eagle_speed : ℕ := 15
def falcon_speed : ℕ := 46
def pelican_speed : ℕ := 33
def hummingbird_speed : ℕ := 30

-- Define the flying time for each bird
def flying_time : ℕ := 2

-- Calculate the total distance flown by all birds
def total_distance_flown : ℕ := (eagle_speed * flying_time) +
                                 (falcon_speed * flying_time) +
                                 (pelican_speed * flying_time) +
                                 (hummingbird_speed * flying_time)

-- The goal is to prove that the total distance flown by all birds is 248 miles
theorem birds_total_distance : total_distance_flown = 248 := by
  -- Proof here
  sorry

end birds_total_distance_l39_39493


namespace solve_eq1_solve_eq2_l39_39217

-- Define the first equation
def eq1 (x : ℚ) : Prop := x / (x - 1) = 3 / (2*x - 2) - 2

-- Define the valid solution for the first equation
def sol1 : ℚ := 7 / 6

-- Theorem for the first equation
theorem solve_eq1 : eq1 sol1 :=
by
  sorry

-- Define the second equation
def eq2 (x : ℚ) : Prop := (5*x + 2) / (x^2 + x) = 3 / (x + 1)

-- Theorem for the second equation: there is no valid solution
theorem solve_eq2 : ¬ ∃ x : ℚ, eq2 x :=
by
  sorry

end solve_eq1_solve_eq2_l39_39217


namespace athletes_camp_duration_l39_39672

theorem athletes_camp_duration
  (h : ℕ)
  (initial_athletes : ℕ := 300)
  (rate_leaving : ℕ := 28)
  (rate_entering : ℕ := 15)
  (hours_entering : ℕ := 7)
  (difference : ℕ := 7) :
  300 - 28 * h + 15 * 7 = 300 + 7 → h = 4 :=
by
  sorry

end athletes_camp_duration_l39_39672


namespace eq_pow_four_l39_39664

theorem eq_pow_four (a b : ℝ) (h : a = b + 1) : a^4 = b^4 → a = 1/2 ∧ b = -1/2 :=
by
  sorry

end eq_pow_four_l39_39664


namespace coin_pile_problem_l39_39898

theorem coin_pile_problem (x y z : ℕ) (h1 : 2 * (x - y) = 16) (h2 : 2 * y - z = 16) (h3 : 2 * z - x + y = 16) :
  x = 22 ∧ y = 14 ∧ z = 12 :=
by
  sorry

end coin_pile_problem_l39_39898


namespace temperature_on_friday_l39_39017

theorem temperature_on_friday 
  (M T W Th F : ℤ) 
  (h1 : (M + T + W + Th) / 4 = 48) 
  (h2 : (T + W + Th + F) / 4 = 46) 
  (h3 : M = 43) : 
  F = 35 := 
by
  sorry

end temperature_on_friday_l39_39017


namespace probability_same_color_l39_39467

-- Definitions for the conditions
def blue_balls : Nat := 8
def yellow_balls : Nat := 5
def total_balls : Nat := blue_balls + yellow_balls

def prob_two_balls_same_color : ℚ :=
  (blue_balls/total_balls) * (blue_balls/total_balls) + (yellow_balls/total_balls) * (yellow_balls/total_balls)

-- Lean statement to be proved
theorem probability_same_color : prob_two_balls_same_color = 89 / 169 :=
by
  -- The proof is omitted as per the instruction
  sorry

end probability_same_color_l39_39467


namespace first_player_wins_the_game_l39_39564

-- Define the game state with 1992 stones and rules for taking stones
structure GameState where
  stones : Nat

-- Game rule: Each player can take a number of stones that is a divisor of the number of stones the 
-- opponent took on the previous turn
def isValidMove (prevMove: Nat) (currentMove: Nat) : Prop :=
  currentMove > 0 ∧ prevMove % currentMove = 0

-- The first player can take any number of stones but not all at once on their first move
def isFirstMoveValid (move: Nat) : Prop :=
  move > 0 ∧ move < 1992

-- Define the initial state of the game with 1992 stones
def initialGameState : GameState := { stones := 1992 }

-- Definition of optimal play leading to the first player's victory
def firstPlayerWins (s : GameState) : Prop :=
  s.stones = 1992 →
  ∃ move: Nat, isFirstMoveValid move ∧
  ∃ nextState: GameState, nextState.stones = s.stones - move ∧ 
  -- The first player wins with optimal strategy
  sorry

-- Theorem statement in Lean 4 equivalent to the math problem
theorem first_player_wins_the_game :
  firstPlayerWins initialGameState :=
  sorry

end first_player_wins_the_game_l39_39564


namespace slope_reciprocal_and_a_bounds_l39_39281

theorem slope_reciprocal_and_a_bounds (x : ℝ) (f g : ℝ → ℝ) 
    (h1 : ∀ x, f x = Real.log x - a * (x - 1)) 
    (h2 : ∀ x, g x = Real.exp x) :
    ((∀ k₁ k₂, (∃ x₁, k₁ = deriv f x₁) ∧ (∃ x₂, k₂ = deriv g x₂) ∧ k₁ * k₂ = 1) 
    ↔ (Real.exp 1 - 1) / Real.exp 1 < a ∧ a < (Real.exp 2 - 1) / Real.exp 1 ∨ a = 0) :=
by
  sorry

end slope_reciprocal_and_a_bounds_l39_39281


namespace lower_limit_brother_l39_39887

variable (W B : Real)

-- Arun's opinion
def aruns_opinion := 66 < W ∧ W < 72

-- Brother's opinion
def brothers_opinion := B < W ∧ W < 70

-- Mother's opinion
def mothers_opinion := W ≤ 69

-- Given the average probable weight of Arun which is 68 kg
def average_weight := (69 + (max 66 B)) / 2 = 68

theorem lower_limit_brother (h₁ : aruns_opinion W) (h₂ : brothers_opinion W B) (h₃ : mothers_opinion W) (h₄ : average_weight B) :
  B = 67 := sorry

end lower_limit_brother_l39_39887


namespace points_per_touchdown_l39_39598

theorem points_per_touchdown (P : ℕ) (games : ℕ) (touchdowns_per_game : ℕ) (two_point_conversions : ℕ) (two_point_conversion_value : ℕ) (total_points : ℕ) :
  touchdowns_per_game = 4 →
  games = 15 →
  two_point_conversions = 6 →
  two_point_conversion_value = 2 →
  total_points = (4 * P * 15 + 6 * two_point_conversion_value) →
  total_points = 372 →
  P = 6 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end points_per_touchdown_l39_39598


namespace tank_capacity_l39_39138

theorem tank_capacity (C : ℝ) (h1 : 1/4 * C + 180 = 3/4 * C) : C = 360 :=
sorry

end tank_capacity_l39_39138


namespace intersection_P_Q_l39_39626

def P : Set ℤ := { x | -4 ≤ x ∧ x ≤ 2 }

def Q : Set ℤ := { x | -3 < x ∧ x < 1 }

theorem intersection_P_Q : P ∩ Q = {-2, -1, 0} :=
by
  sorry

end intersection_P_Q_l39_39626


namespace percentage_difference_l39_39235

variable {P Q : ℝ}

theorem percentage_difference (P Q : ℝ) : (100 * (Q - P)) / Q = ((Q - P) / Q) * 100 :=
by
  sorry

end percentage_difference_l39_39235


namespace xiaopangs_score_is_16_l39_39324

-- Define the father's score
def fathers_score : ℕ := 48

-- Define Xiaopang's score in terms of father's score
def xiaopangs_score (fathers_score : ℕ) : ℕ := fathers_score / 2 - 8

-- The theorem to prove that Xiaopang's score is 16
theorem xiaopangs_score_is_16 : xiaopangs_score fathers_score = 16 := 
by
  sorry

end xiaopangs_score_is_16_l39_39324


namespace six_digit_number_consecutive_evens_l39_39703

theorem six_digit_number_consecutive_evens :
  ∃ n : ℕ,
    287232 = (2 * n - 2) * (2 * n) * (2 * n + 2) ∧
    287232 / 100000 = 2 ∧
    287232 % 10 = 2 :=
by
  sorry

end six_digit_number_consecutive_evens_l39_39703


namespace intersection_ST_l39_39719

def S : Set ℝ := { x : ℝ | x < -5 } ∪ { x : ℝ | x > 5 }
def T : Set ℝ := { x : ℝ | -7 < x ∧ x < 3 }

theorem intersection_ST : S ∩ T = { x : ℝ | -7 < x ∧ x < -5 } := 
by 
  sorry

end intersection_ST_l39_39719


namespace michael_needs_flour_l39_39421

-- Define the given conditions
def total_flour : ℕ := 8
def measuring_cup : ℚ := 1/4
def scoops_to_remove : ℕ := 8

-- Prove the amount of flour Michael needs is 6 cups
theorem michael_needs_flour : 
  (total_flour - (scoops_to_remove * measuring_cup)) = 6 := 
by
  sorry

end michael_needs_flour_l39_39421


namespace c_share_l39_39989

theorem c_share (S : ℝ) (b_share_per_rs c_share_per_rs : ℝ)
  (h1 : S = 246)
  (h2 : b_share_per_rs = 0.65)
  (h3 : c_share_per_rs = 0.40) :
  (c_share_per_rs * S) = 98.40 :=
by sorry

end c_share_l39_39989


namespace triangle_leg_length_l39_39793

theorem triangle_leg_length (perimeter_square : ℝ)
                            (base_triangle : ℝ)
                            (area_equality : ∃ (side_square : ℝ) (height_triangle : ℝ),
                                4 * side_square = perimeter_square ∧
                                side_square * side_square = (1/2) * base_triangle * height_triangle)
                            : ∃ (y : ℝ), y = 22.5 :=
by
  -- Placeholder proof
  sorry

end triangle_leg_length_l39_39793


namespace min_value_quadratic_l39_39743

theorem min_value_quadratic :
  ∃ (x y : ℝ), (∀ (a b : ℝ), (3*a^2 + 4*a*b + 2*b^2 - 6*a - 8*b + 6 ≥ 0)) ∧ 
  (3*x^2 + 4*x*y + 2*y^2 - 6*x - 8*y + 6 = 0) := 
sorry

end min_value_quadratic_l39_39743


namespace min_value_of_sequence_l39_39351

theorem min_value_of_sequence :
  ∃ (a : ℕ → ℤ), a 1 = 0 ∧ (∀ n : ℕ, n ≥ 2 → |a n| = |a (n - 1) + 1|) ∧ (a 1 + a 2 + a 3 + a 4 = -2) :=
by
  sorry

end min_value_of_sequence_l39_39351


namespace arithmetic_sequence_a4_is_5_l39_39537

variable (a : ℕ → ℕ)

-- Arithmetic sequence property
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n m k : ℕ, n < m ∧ m < k → 2 * a m = a n + a k

-- Given condition
axiom sum_third_and_fifth : a 3 + a 5 = 10

-- Prove that a_4 = 5
theorem arithmetic_sequence_a4_is_5
  (h : is_arithmetic_sequence a) : a 4 = 5 := by
  sorry

end arithmetic_sequence_a4_is_5_l39_39537


namespace fraction_bounds_l39_39744

theorem fraction_bounds (n : ℕ) (h : 0 < n) : (1 : ℚ) / 2 ≤ n / (n + 1 : ℚ) ∧ n / (n + 1 : ℚ) < 1 :=
by
  sorry

end fraction_bounds_l39_39744


namespace sin_beta_value_l39_39618

theorem sin_beta_value (alpha beta : ℝ) (h1 : 0 < alpha) (h2 : alpha < beta) (h3 : beta < π / 2)
  (h4 : Real.sin alpha = 3 / 5) (h5 : Real.cos (alpha - beta) = 12 / 13) : Real.sin beta = 56 / 65 := by
  sorry

end sin_beta_value_l39_39618


namespace equal_saturdays_and_sundays_l39_39317

theorem equal_saturdays_and_sundays (start_day : ℕ) (h : start_day < 7) :
  ∃! d, (d < 7 ∧ ((d + 2) % 7 = 0 → (d = 5))) :=
by
  sorry

end equal_saturdays_and_sundays_l39_39317


namespace longest_side_similar_triangle_l39_39471

theorem longest_side_similar_triangle (a b c : ℝ) (p : ℝ) (h₀ : a = 8) (h₁ : b = 15) (h₂ : c = 17) (h₃ : a^2 + b^2 = c^2) (h₄ : p = 160) :
  ∃ x : ℝ, (8 * x) + (15 * x) + (17 * x) = p ∧ 17 * x = 68 :=
by
  sorry

end longest_side_similar_triangle_l39_39471


namespace sin_17pi_over_6_l39_39813

theorem sin_17pi_over_6 : Real.sin (17 * Real.pi / 6) = 1 / 2 :=
by
  sorry

end sin_17pi_over_6_l39_39813


namespace find_floors_l39_39784

theorem find_floors (a b : ℕ) 
  (h1 : 3 * a + 4 * b = 25)
  (h2 : 2 * a + 3 * b = 18) : 
  a = 3 ∧ b = 4 := 
sorry

end find_floors_l39_39784


namespace prove_value_of_expressions_l39_39350

theorem prove_value_of_expressions (a b : ℕ) 
  (h₁ : 2^a = 8^b) 
  (h₂ : a + 2 * b = 5) : 
  2^a + 8^b = 16 := 
by 
  -- proof steps go here
  sorry

end prove_value_of_expressions_l39_39350


namespace maximum_value_of_expression_l39_39228

theorem maximum_value_of_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a * b = 1) :
  (1 / (a + 9 * b) + 1 / (9 * a + b)) ≤ 5 / 24 := sorry

end maximum_value_of_expression_l39_39228


namespace distinct_real_roots_range_l39_39938

def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem distinct_real_roots_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (x^2 - 2 * x + a = 0) ∧ (y^2 - 2 * y + a = 0))
  ↔ a < 1 := 
by
  sorry

end distinct_real_roots_range_l39_39938


namespace Greg_more_than_Sharon_l39_39373

-- Define the harvest amounts
def Greg_harvest : ℝ := 0.4
def Sharon_harvest : ℝ := 0.1

-- Show that Greg harvested 0.3 more acres than Sharon
theorem Greg_more_than_Sharon : Greg_harvest - Sharon_harvest = 0.3 := by
  sorry

end Greg_more_than_Sharon_l39_39373


namespace cubes_difference_l39_39389

theorem cubes_difference (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 25) : a^3 - b^3 = 99 := 
sorry

end cubes_difference_l39_39389


namespace percentage_increase_overtime_rate_l39_39247

theorem percentage_increase_overtime_rate :
  let regular_rate := 16
  let regular_hours_limit := 30
  let total_earnings := 760
  let total_hours_worked := 40
  let overtime_rate := 28 -- This is calculated as $280/10 from the solution.
  let increase_in_hourly_rate := overtime_rate - regular_rate
  let percentage_increase := (increase_in_hourly_rate / regular_rate) * 100
  percentage_increase = 75 :=
by {
  sorry
}

end percentage_increase_overtime_rate_l39_39247


namespace spherical_to_rectangular_coordinates_l39_39709

-- Define the given conditions
variable (ρ : ℝ) (θ : ℝ) (φ : ℝ)
variable (hρ : ρ = 6) (hθ : θ = 7 * Real.pi / 4) (hφ : φ = Real.pi / 2)

-- Convert spherical coordinates (ρ, θ, φ) to rectangular coordinates (x, y, z) and prove the values
theorem spherical_to_rectangular_coordinates :
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  x = 3 * Real.sqrt 2 ∧ y = -3 * Real.sqrt 2 ∧ z = 0 :=
by
  sorry

end spherical_to_rectangular_coordinates_l39_39709


namespace dogs_on_mon_wed_fri_l39_39726

def dogs_on_tuesday : ℕ := 12
def dogs_on_thursday : ℕ := 9
def pay_per_dog : ℕ := 5
def total_earnings : ℕ := 210

theorem dogs_on_mon_wed_fri :
  ∃ (d : ℕ), d = 21 ∧ d * pay_per_dog = total_earnings - (dogs_on_tuesday + dogs_on_thursday) * pay_per_dog :=
by 
  sorry

end dogs_on_mon_wed_fri_l39_39726


namespace new_ratio_l39_39034

theorem new_ratio (J: ℝ) (F: ℝ) (F_new: ℝ): 
  J = 59.99999999999997 → 
  F / J = 3 / 2 → 
  F_new = F + 10 → 
  F_new / J = 5 / 3 :=
by
  intros hJ hF hF_new
  sorry

end new_ratio_l39_39034


namespace cos_240_degree_l39_39148

theorem cos_240_degree : Real.cos (240 * Real.pi / 180) = -1/2 :=
by
  sorry

end cos_240_degree_l39_39148


namespace find_last_number_l39_39438

-- Definitions for the conditions
def avg_first_three (A B C : ℕ) : ℕ := (A + B + C) / 3
def avg_last_three (B C D : ℕ) : ℕ := (B + C + D) / 3
def sum_first_last (A D : ℕ) : ℕ := A + D

-- Proof problem statement
theorem find_last_number (A B C D : ℕ) 
  (h1 : avg_first_three A B C = 6)
  (h2 : avg_last_three B C D = 5)
  (h3 : sum_first_last A D = 11) : D = 4 :=
sorry

end find_last_number_l39_39438


namespace num_quarters_l39_39413

theorem num_quarters (n q : ℕ) (avg_initial avg_new : ℕ) 
  (h1 : avg_initial = 10) 
  (h2 : avg_new = 12) 
  (h3 : avg_initial * n + 10 = avg_new * (n + 1)) :
  q = 1 :=
by {
  sorry
}

end num_quarters_l39_39413


namespace money_left_l39_39706

noncomputable def initial_amount : ℕ := 100
noncomputable def cost_roast : ℕ := 17
noncomputable def cost_vegetables : ℕ := 11

theorem money_left (init_amt cost_r cost_v : ℕ) 
  (h1 : init_amt = 100)
  (h2 : cost_r = 17)
  (h3 : cost_v = 11) : init_amt - (cost_r + cost_v) = 72 := by
  sorry

end money_left_l39_39706


namespace count_equilateral_triangles_in_hexagonal_lattice_l39_39248

-- Definitions based on conditions in problem (hexagonal lattice setup)
def hexagonal_lattice (dist : ℕ) : Prop :=
  -- Define properties of the points in hexagonal lattice
  -- Placeholder for actual structure defining the hexagon and surrounding points
  sorry

def equilateral_triangles (n : ℕ) : Prop :=
  -- Define a method to count equilateral triangles in the given lattice setup
  sorry

-- Theorem stating that 10 equilateral triangles can be formed in the lattice
theorem count_equilateral_triangles_in_hexagonal_lattice (dist : ℕ) (h : dist = 1 ∨ dist = 2) :
  equilateral_triangles 10 :=
by
  -- Proof to be completed
  sorry

end count_equilateral_triangles_in_hexagonal_lattice_l39_39248


namespace find_middle_side_length_l39_39977

theorem find_middle_side_length (a b c : ℕ) (h1 : a + b + c = 2022) (h2 : c - b = 1) (h3 : b - a = 2) :
  b = 674 := 
by
  -- The proof goes here, but we skip it using sorry.
  sorry

end find_middle_side_length_l39_39977


namespace green_paint_quarts_l39_39232

theorem green_paint_quarts (blue green white : ℕ) (h_ratio : 3 = blue ∧ 2 = green ∧ 4 = white) 
  (h_white_paint : white = 12) : green = 6 := 
by
  sorry

end green_paint_quarts_l39_39232


namespace exists_positive_int_n_l39_39416

theorem exists_positive_int_n (p a k : ℕ) 
  (hp : Nat.Prime p) (ha : 0 < a) (hk1 : p^a < k) (hk2 : k < 2 * p^a) :
  ∃ n : ℕ, n < p^(2 * a) ∧ (Nat.choose n k ≡ n [MOD p^a]) ∧ (n ≡ k [MOD p^a]) :=
sorry

end exists_positive_int_n_l39_39416


namespace find_x_l39_39344

namespace MathProof

variables {a b x : ℝ}
variables (h1 : a > 0) (h2 : b > 0)

theorem find_x (h3 : (a^2)^(2 * b) = a^b * x^b) : x = a^3 :=
by sorry

end MathProof

end find_x_l39_39344


namespace baseball_card_difference_l39_39162

theorem baseball_card_difference (marcus_cards carter_cards : ℕ) (h1 : marcus_cards = 210) (h2 : carter_cards = 152) : marcus_cards - carter_cards = 58 :=
by {
    --skip the proof
    sorry
}

end baseball_card_difference_l39_39162


namespace slope_of_intersection_line_l39_39488

theorem slope_of_intersection_line 
    (x y : ℝ)
    (h1 : x^2 + y^2 - 6*x + 4*y - 20 = 0)
    (h2 : x^2 + y^2 - 2*x - 6*y + 10 = 0) :
    ∃ m : ℝ, m = 0.4 := 
sorry

end slope_of_intersection_line_l39_39488


namespace maximize_take_home_pay_l39_39483

-- Define the tax system condition
def tax (y : ℝ) : ℝ := y^3

-- Define the take-home pay condition
def take_home_pay (y : ℝ) : ℝ := 100 * y^2 - tax y

-- The theorem to prove the maximum take-home pay is achieved at a specific income level
theorem maximize_take_home_pay : 
  ∃ y : ℝ, take_home_pay y = 100 * 50^2 - 50^3 := sorry

end maximize_take_home_pay_l39_39483


namespace compound_oxygen_atoms_l39_39487

theorem compound_oxygen_atoms (H C O : Nat) (mw : Nat) (H_weight C_weight O_weight : Nat) 
  (h_H : H = 2)
  (h_C : C = 1)
  (h_mw : mw = 62)
  (h_H_weight : H_weight = 1)
  (h_C_weight : C_weight = 12)
  (h_O_weight : O_weight = 16)
  : O = 3 :=
by
  sorry

end compound_oxygen_atoms_l39_39487


namespace solve_n_m_equation_l39_39645

theorem solve_n_m_equation : 
  ∃ (n m : ℤ), n^4 - 2*n^2 = m^2 + 38 ∧ ((n, m) = (3, 5) ∨ (n, m) = (3, -5) ∨ (n, m) = (-3, 5) ∨ (n, m) = (-3, -5)) :=
by { sorry }

end solve_n_m_equation_l39_39645


namespace max_sum_x_y_l39_39408

theorem max_sum_x_y (x y : ℝ) (h1 : x^2 + y^2 = 7) (h2 : x^3 + y^3 = 10) : x + y ≤ 4 :=
sorry

end max_sum_x_y_l39_39408


namespace triangle_angle_extension_l39_39322

theorem triangle_angle_extension :
  ∀ (BAC ABC BCA CDB DBC : ℝ),
  180 = BAC + ABC + BCA →
  CDB = BAC + ABC →
  DBC = BAC + BCA →
  (CDB + DBC) / (BAC + ABC) = 2 :=
by
  intros BAC ABC BCA CDB DBC h1 h2 h3
  sorry

end triangle_angle_extension_l39_39322


namespace product_power_conjecture_calculate_expression_l39_39574

-- Conjecture Proof
theorem product_power_conjecture (a b : ℂ) (n : ℕ) : (a * b)^n = (a^n) * (b^n) :=
sorry

-- Calculation Proof
theorem calculate_expression : 
  ((-0.125 : ℂ)^2022) * ((2 : ℂ)^2021) * ((4 : ℂ)^2020) = (1 / 32 : ℂ) :=
sorry

end product_power_conjecture_calculate_expression_l39_39574


namespace hyperbola_equation_l39_39202

theorem hyperbola_equation (h : ∃ (x y : ℝ), y = 1 / 2 * x) (p : (2, 2) ∈ {p : ℝ × ℝ | ((p.snd)^2 / 3) - ((p.fst)^2 / 12) = 1}) :
  ∀ (x y : ℝ), (y^2 / 3 - x^2 / 12 = 1) ↔ (∃ (a b : ℝ), y = a * x ∧ b * y = x ^ 2) :=
sorry

end hyperbola_equation_l39_39202


namespace max_blocks_fit_in_box_l39_39361

def box_dimensions : ℕ × ℕ × ℕ := (4, 6, 2)
def block_dimensions : ℕ × ℕ × ℕ := (3, 2, 1)
def block_volume := 6
def box_volume := 48

theorem max_blocks_fit_in_box (box_dimensions : ℕ × ℕ × ℕ)
    (block_dimensions : ℕ × ℕ × ℕ) : 
  (box_volume / block_volume = 8) := 
by
  sorry

end max_blocks_fit_in_box_l39_39361


namespace max_log_sum_l39_39934

open Real

theorem max_log_sum (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : x + 4 * y = 40) :
  log x + log y ≤ 2 :=
sorry

end max_log_sum_l39_39934


namespace lucas_fraction_of_money_left_l39_39669

theorem lucas_fraction_of_money_left (m p n : ℝ) (h1 : (1 / 4) * m = (1 / 2) * n * p) :
  (m - n * p) / m = 1 / 2 :=
by 
  -- Sorry is used to denote that we are skipping the proof
  sorry

end lucas_fraction_of_money_left_l39_39669


namespace number_div_mult_l39_39411

theorem number_div_mult (n : ℕ) (h : n = 4) : (n / 6) * 12 = 8 :=
by
  sorry

end number_div_mult_l39_39411


namespace solution_system_equations_l39_39801

theorem solution_system_equations :
  ∀ (x y : ℝ) (k n : ℤ),
    (4 * (Real.cos x) ^ 2 - 4 * Real.cos x * (Real.cos (6 * x)) ^ 2 + (Real.cos (6 * x)) ^ 2 = 0) ∧
    (Real.sin x = Real.cos y) →
    (∃ k n : ℤ, (x = (Real.pi / 3) + 2 * Real.pi * k ∧ y = (Real.pi / 6) + 2 * Real.pi * n) ∨
                 (x = (Real.pi / 3) + 2 * Real.pi * k ∧ y = -(Real.pi / 6) + 2 * Real.pi * n) ∨
                 (x = -(Real.pi / 3) + 2 * Real.pi * k ∧ y = (5 * Real.pi / 6) + 2 * Real.pi * n) ∨
                 (x = -(Real.pi / 3) + 2 * Real.pi * k ∧ y = -(5 * Real.pi / 6) + 2 * Real.pi * n)) :=
by
  sorry

end solution_system_equations_l39_39801


namespace find_all_possible_f_l39_39099

-- Noncomputability is needed here since we cannot construct a function 
-- like f deterministically via computation due to the nature of the problem.
noncomputable def functional_equation_solution (f : ℕ → ℕ) := 
  (∀ a b : ℕ, f a + f b ∣ 2 * (a + b - 1)) → 
  (∀ x : ℕ, f x = 1) ∨ (∀ x : ℕ, f x = 2 * x - 1)

-- Statement of the mathematically equivalent proof problem.
theorem find_all_possible_f (f : ℕ → ℕ) : functional_equation_solution f := 
sorry

end find_all_possible_f_l39_39099


namespace problem_statement_l39_39136

theorem problem_statement (x : ℝ) (h : x + 1/x = 3) : (x - 3)^4 + 81 / (x - 3)^4 = 63 :=
by
  sorry

end problem_statement_l39_39136


namespace correct_option_D_l39_39713

theorem correct_option_D (y : ℝ): 
  3 * y^2 - 2 * y^2 = y^2 :=
by
  sorry

end correct_option_D_l39_39713


namespace find_multiplication_value_l39_39715

-- Define the given conditions
def student_chosen_number : ℤ := 63
def subtracted_value : ℤ := 142
def result_after_subtraction : ℤ := 110

-- Define the value he multiplied the number by
def multiplication_value (x : ℤ) : Prop := 
  (student_chosen_number * x) - subtracted_value = result_after_subtraction

-- Statement to prove that the value he multiplied the number by is 4
theorem find_multiplication_value : 
  ∃ x : ℤ, multiplication_value x ∧ x = 4 :=
by 
  -- Placeholder for the actual proof
  sorry

end find_multiplication_value_l39_39715


namespace number_of_pupils_in_class_l39_39056

-- Defining the conditions
def wrongMark : ℕ := 79
def correctMark : ℕ := 45
def averageIncreasedByHalf : ℕ := 2  -- Condition representing average increased by half

-- The goal is to prove the number of pupils is 68
theorem number_of_pupils_in_class (n S : ℕ) (h1 : wrongMark = 79) (h2 : correctMark = 45)
(h3 : averageIncreasedByHalf = 2) 
(h4 : S + (wrongMark - correctMark) = (3 / 2) * S) :
  n = 68 :=
  sorry

end number_of_pupils_in_class_l39_39056


namespace problem_arith_seq_l39_39748

variables {a : ℕ → ℝ} (d : ℝ)
def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem problem_arith_seq (h_arith : is_arithmetic_sequence a) 
  (h_condition : a 1 + a 6 + a 11 = 3) 
  : a 3 + a 9 = 2 :=
sorry

end problem_arith_seq_l39_39748


namespace original_number_l39_39363

def is_three_digit (n : ℕ) : Prop := n >= 100 ∧ n < 1000

def permutations_sum (a b c : ℕ) : ℕ :=
  let abc := 100 * a + 10 * b + c
  let acb := 100 * a + 10 * c + b
  let bac := 100 * b + 10 * a + c
  let bca := 100 * b + 10 * c + a
  let cab := 100 * c + 10 * a + b
  let cba := 100 * c + 10 * b + a
  abc + acb + bac + bca + cab + cba

theorem original_number (abc : ℕ) (a b c : ℕ) :
  is_three_digit abc →
  abc = 100 * a + 10 * b + c →
  permutations_sum a b c = 3194 →
  abc = 358 :=
by
  sorry

end original_number_l39_39363


namespace rank_matA_l39_39072

def matA : Matrix (Fin 4) (Fin 5) ℤ :=
  ![![5, 7, 12, 48, -14],
    ![9, 16, 24, 98, -31],
    ![14, 24, 25, 146, -45],
    ![11, 12, 24, 94, -25]]

theorem rank_matA : Matrix.rank matA = 3 :=
by
  sorry

end rank_matA_l39_39072


namespace log_simplification_l39_39539

open Real

theorem log_simplification (a b d e z y : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) (he : e ≠ 0)
  (ha : a ≠ 0) (hz : z ≠ 0) (hy : y ≠ 0) :
  log (a / b) + log (b / e) + log (e / d) - log (az / dy) = log (dy / z) :=
by
  sorry

end log_simplification_l39_39539


namespace range_of_m_real_roots_l39_39067

theorem range_of_m_real_roots (m : ℝ) : 
  (∀ x : ℝ, ∃ k l : ℝ, k = 2*x ∧ l = m - x^2 ∧ k^2 - 4*l ≥ 0) ↔ m ≤ 1 := 
sorry

end range_of_m_real_roots_l39_39067


namespace expression_constant_for_large_x_l39_39506

theorem expression_constant_for_large_x (x : ℝ) (h : x ≥ 4 / 7) : 
  -4 * x + |4 - 7 * x| - |1 - 3 * x| + 4 = 1 :=
by
  sorry

end expression_constant_for_large_x_l39_39506


namespace degree_of_angle_C_l39_39849

theorem degree_of_angle_C 
  (A B C : ℝ) 
  (h1 : A = 4 * x) 
  (h2 : B = 4 * x) 
  (h3 : C = 7 * x) 
  (h_sum : A + B + C = 180) : 
  C = 84 := 
by 
  sorry

end degree_of_angle_C_l39_39849


namespace first_train_cross_time_is_10_seconds_l39_39042

-- Definitions based on conditions
def length_of_train := 120 -- meters
def time_second_train_cross_telegraph_post := 15 -- seconds
def distance_cross_each_other := 240 -- meters
def time_cross_each_other := 12 -- seconds

-- The speed of the second train
def speed_second_train := length_of_train / time_second_train_cross_telegraph_post -- m/s

-- The relative speed of both trains when crossing each other
def relative_speed := distance_cross_each_other / time_cross_each_other -- m/s

-- The speed of the first train
def speed_first_train := relative_speed - speed_second_train -- m/s

-- The time taken by the first train to cross the telegraph post
def time_first_train_cross_telegraph_post := length_of_train / speed_first_train -- seconds

-- Proof statement
theorem first_train_cross_time_is_10_seconds :
  time_first_train_cross_telegraph_post = 10 := by
  sorry

end first_train_cross_time_is_10_seconds_l39_39042


namespace herd_total_cows_l39_39831

noncomputable def total_cows (n : ℕ) : Prop :=
  let fraction_first_son := 1 / 3
  let fraction_second_son := 1 / 5
  let fraction_third_son := 1 / 9
  let fraction_combined := fraction_first_son + fraction_second_son + fraction_third_son
  let fraction_fourth_son := 1 - fraction_combined
  let cows_fourth_son := 11
  fraction_fourth_son * n = cows_fourth_son

theorem herd_total_cows : ∃ n : ℕ, total_cows n ∧ n = 31 :=
by
  existsi 31
  sorry

end herd_total_cows_l39_39831


namespace graph_depicts_one_line_l39_39909

theorem graph_depicts_one_line {x y : ℝ} :
  (x - 1) ^ 2 * (x + y - 2) = (y - 1) ^ 2 * (x + y - 2) →
  ∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b :=
by
  intros h
  sorry

end graph_depicts_one_line_l39_39909


namespace shorter_diagonal_of_rhombus_l39_39242

variable (d s : ℝ)  -- d for shorter diagonal, s for the side length of the rhombus

theorem shorter_diagonal_of_rhombus 
  (h1 : ∀ (s : ℝ), s = 39)
  (h2 : ∀ (a b : ℝ), a^2 + b^2 = s^2)
  (h3 : ∀ (d a : ℝ), (d / 2)^2 + a^2 = 39^2)
  (h4 : 72 / 2 = 36)
  : d = 30 := 
by 
  sorry

end shorter_diagonal_of_rhombus_l39_39242


namespace inheritance_amount_l39_39165

def federalTax (x : ℝ) : ℝ := 0.25 * x
def remainingAfterFederalTax (x : ℝ) : ℝ := x - federalTax x
def stateTax (x : ℝ) : ℝ := 0.15 * remainingAfterFederalTax x
def totalTaxes (x : ℝ) : ℝ := federalTax x + stateTax x

theorem inheritance_amount (x : ℝ) (h : totalTaxes x = 15000) : x = 41379 :=
by
  sorry

end inheritance_amount_l39_39165


namespace sum_of_faces_edges_vertices_l39_39696

def rectangular_prism_faces : ℕ := 6
def rectangular_prism_edges : ℕ := 12
def rectangular_prism_vertices : ℕ := 8

theorem sum_of_faces_edges_vertices : 
  rectangular_prism_faces + rectangular_prism_edges + rectangular_prism_vertices = 26 := 
by
  sorry

end sum_of_faces_edges_vertices_l39_39696


namespace remainder_div_82_l39_39299

theorem remainder_div_82 (x : ℤ) (h : ∃ k : ℤ, x + 17 = 41 * k + 22) : (x % 82 = 5) :=
by
  sorry

end remainder_div_82_l39_39299


namespace order_of_magnitude_l39_39953

noncomputable def a : Real := 70.3
noncomputable def b : Real := 70.2
noncomputable def c : Real := Real.log 0.3

theorem order_of_magnitude : a > b ∧ b > c :=
by
  sorry

end order_of_magnitude_l39_39953


namespace backyard_area_l39_39901

theorem backyard_area {length width : ℝ} 
  (h1 : 30 * length = 1500) 
  (h2 : 12 * (2 * (length + width)) = 1500) : 
  length * width = 625 :=
by
  sorry

end backyard_area_l39_39901


namespace bus_driver_hours_l39_39211

theorem bus_driver_hours (h : ℕ) (regular_rate : ℕ) (extra_rate1 : ℕ) (extra_rate2 : ℕ) (total_earnings : ℕ)
  (h1 : regular_rate = 14)
  (h2 : extra_rate1 = (14 + (14 * 35 / 100)))
  (h3: extra_rate2 = (14 + (14 * 75 / 100)))
  (h4: total_earnings = 1230)
  (h5: total_earnings = 40 * regular_rate + 10 * extra_rate1 + (h - 50) * extra_rate2)
  (condition : 50 < h) :
  h = 69 :=
by
  sorry

end bus_driver_hours_l39_39211


namespace investment_amount_l39_39160

theorem investment_amount (A_investment B_investment total_profit A_share : ℝ)
  (hA_investment : A_investment = 100)
  (hB_investment_months : B_investment > 0)
  (h_total_profit : total_profit = 100)
  (h_A_share : A_share = 50)
  (h_conditions : A_share / total_profit = (A_investment * 12) / ((A_investment * 12) + (B_investment * 6))) :
  B_investment = 200 :=
by {
  sorry
}

end investment_amount_l39_39160


namespace find_linear_function_l39_39987

theorem find_linear_function (a m : ℝ) : 
  (∀ x y : ℝ, (x, y) = (-2, -3) ∨ (x, y) = (-1, -1) ∨ (x, y) = (0, m) ∨ (x, y) = (1, 3) ∨ (x, y) = (a, 5) → 
  y = 2 * x + 1) → 
  (m = 1 ∧ a = 2) :=
by
  sorry

end find_linear_function_l39_39987


namespace min_value_of_reciprocal_sum_l39_39864

variable (m n : ℝ)
variable (a : ℝ × ℝ := (m, 1))
variable (b : ℝ × ℝ := (4 - n, 2))

theorem min_value_of_reciprocal_sum
  (h1 : m > 0) (h2 : n > 0)
  (h3 : a.1 * b.2 = a.2 * b.1) :
  (1/m + 8/n) = 9/2 :=
sorry

end min_value_of_reciprocal_sum_l39_39864


namespace value_of_expression_l39_39007

theorem value_of_expression (x y : ℝ) (h1 : 4 * x + y = 20) (h2 : x + 4 * y = 16) : 
  17 * x ^ 2 + 20 * x * y + 17 * y ^ 2 = 656 :=
sorry

end value_of_expression_l39_39007


namespace minimum_value_l39_39142

theorem minimum_value (x : ℝ) (h : x > -3) : 2 * x + (1 / (x + 3)) ≥ 2 * Real.sqrt 2 - 6 :=
sorry

end minimum_value_l39_39142


namespace arith_seq_ratio_l39_39029

variable {S T : ℕ → ℚ}

-- Conditions
def is_arith_seq_sum (S : ℕ → ℚ) (a : ℕ → ℚ) :=
  ∀ n, S n = n * (2 * a 1 + (n - 1) * a n) / 2

def ratio_condition (S T : ℕ → ℚ) :=
  ∀ n, S n / T n = (2 * n - 1) / (3 * n + 2)

-- Main theorem
theorem arith_seq_ratio
  (a b : ℕ → ℚ)
  (h1 : is_arith_seq_sum S a)
  (h2 : is_arith_seq_sum T b)
  (h3 : ratio_condition S T)
  : a 7 / b 7 = 25 / 41 :=
sorry

end arith_seq_ratio_l39_39029


namespace interval_of_n_l39_39596

noncomputable def divides (a b : ℕ) : Prop := ∃ k, b = k * a

theorem interval_of_n (n : ℕ) (hn : 0 < n ∧ n < 2000)
  (h1 : divides n 9999)
  (h2 : divides (n + 4) 999999) :
  801 ≤ n ∧ n ≤ 1200 :=
sorry

end interval_of_n_l39_39596


namespace initial_payment_mr_dubois_l39_39931

-- Definition of the given conditions
def total_cost_of_car : ℝ := 13380
def monthly_payment : ℝ := 420
def number_of_months : ℝ := 19

-- Calculate the total amount paid in monthly installments
def total_amount_paid_in_installments : ℝ := monthly_payment * number_of_months

-- Statement of the theorem we want to prove
theorem initial_payment_mr_dubois :
  total_cost_of_car - total_amount_paid_in_installments = 5400 :=
by
  sorry

end initial_payment_mr_dubois_l39_39931


namespace tim_income_percentage_less_than_juan_l39_39798

variables (M T J : ℝ)

theorem tim_income_percentage_less_than_juan 
  (h1 : M = 1.60 * T)
  (h2 : M = 0.80 * J) : 
  100 - 100 * (T / J) = 50 :=
by
  sorry

end tim_income_percentage_less_than_juan_l39_39798


namespace perpendicular_vectors_l39_39262

theorem perpendicular_vectors (b : ℚ) : 
  (4 * b - 15 = 0) → (b = 15 / 4) :=
by
  intro h
  sorry

end perpendicular_vectors_l39_39262


namespace prob_snow_both_days_l39_39551

-- Definitions for the conditions
def prob_snow_monday : ℚ := 40 / 100
def prob_snow_tuesday : ℚ := 30 / 100

def independent_events (A B : Prop) : Prop := true  -- A placeholder definition of independence

-- The proof problem: 
theorem prob_snow_both_days : 
  independent_events (prob_snow_monday = 0.40) (prob_snow_tuesday = 0.30) →
  prob_snow_monday * prob_snow_tuesday = 0.12 := 
by 
  sorry

end prob_snow_both_days_l39_39551


namespace geometric_sequence_sum_l39_39652

theorem geometric_sequence_sum (a₁ q : ℝ) (h1 : q ≠ 1)
    (hS2 : (a₁ * (1 - q^2)) / (1 - q) = 1)
    (hS4 : (a₁ * (1 - q^4)) / (1 - q) = 3) :
    (a₁ * (1 - q^8)) / (1 - q) = 15 := 
by
  sorry

end geometric_sequence_sum_l39_39652


namespace fifth_term_arithmetic_seq_l39_39660

theorem fifth_term_arithmetic_seq (a d : ℤ) 
  (h10th : a + 9 * d = 23) 
  (h11th : a + 10 * d = 26) 
  : a + 4 * d = 8 :=
sorry

end fifth_term_arithmetic_seq_l39_39660


namespace negation_cube_of_every_odd_is_odd_l39_39658

def odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

def cube (n : ℤ) : ℤ := n * n * n

def cube_of_odd_is_odd (n : ℤ) : Prop := odd n → odd (cube n)

theorem negation_cube_of_every_odd_is_odd :
  ¬ (∀ n : ℤ, odd n → odd (cube n)) ↔ ∃ n : ℤ, odd n ∧ ¬ odd (cube n) :=
sorry

end negation_cube_of_every_odd_is_odd_l39_39658


namespace robinson_family_children_count_l39_39069

theorem robinson_family_children_count 
  (m : ℕ) -- mother's age
  (f : ℕ) (f_age : f = 50) -- father's age is 50
  (x : ℕ) -- number of children
  (y : ℕ) -- average age of children
  (h1 : (m + 50 + x * y) / (2 + x) = 22)
  (h2 : (m + x * y) / (1 + x) = 18) :
  x = 6 := 
sorry

end robinson_family_children_count_l39_39069


namespace smallest_lcm_of_4digit_multiples_of_5_l39_39775

theorem smallest_lcm_of_4digit_multiples_of_5 :
  ∃ m n : ℕ, (1000 ≤ m) ∧ (m ≤ 9999) ∧ (1000 ≤ n) ∧ (n ≤ 9999) ∧ (Nat.gcd m n = 5) ∧ (Nat.lcm m n = 201000) := 
sorry

end smallest_lcm_of_4digit_multiples_of_5_l39_39775


namespace trigonometric_identity_l39_39452

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3 :=
sorry

end trigonometric_identity_l39_39452


namespace ratio_of_sides_l39_39910

open Real

variable (s y x : ℝ)

-- Assuming the rectangles and squares conditions
def condition1 := 4 * (x * y) + s * s = 9 * (s * s)
def condition2 := x = 2 * s
def condition3 := y = s

-- Stating the theorem
theorem ratio_of_sides (h1 : condition1 s y x) (h2 : condition2 s x) (h3 : condition3 s y) :
  x / y = 2 := by
  sorry

end ratio_of_sides_l39_39910


namespace problem1_problem2_l39_39555

section problems

variables (m n a b : ℕ)
variables (h1 : 4 ^ m = a) (h2 : 8 ^ n = b)

theorem problem1 : 2 ^ (2 * m + 3 * n) = a * b :=
sorry

theorem problem2 : 2 ^ (4 * m - 6 * n) = a ^ 2 / b ^ 2 :=
sorry

end problems

end problem1_problem2_l39_39555


namespace part_a_part_b_l39_39935

theorem part_a (k : ℕ) : ∃ (a : ℕ → ℕ), (∀ i, i ≤ k → a i > 0) ∧ (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → a i < a j) ∧ (∀ i j, 1 ≤ i ∧ i ≠ j ∧ i ≤ k ∧ j ≤ k → (a i - a j) ∣ a i) :=
sorry

theorem part_b : ∃ C > 0, ∀ a : ℕ → ℕ, (∀ k : ℕ, (∀ i j, 1 ≤ i ∧ i ≠ j ∧ i ≤ k ∧ j ≤ k → (a i - a j) ∣ a i) → a 1 > (k : ℕ) ^ (C * k : ℕ)) :=
sorry

end part_a_part_b_l39_39935


namespace initial_workers_l39_39088

/--
In a factory, some workers were employed, and then 25% more workers have just been hired.
There are now 1065 employees in the factory. Prove that the number of workers initially employed is 852.
-/
theorem initial_workers (x : ℝ) (h1 : x + 0.25 * x = 1065) : x = 852 :=
sorry

end initial_workers_l39_39088


namespace original_cost_111_l39_39945

theorem original_cost_111 (P : ℝ) (h1 : 0.76 * P * 0.90 = 760) : P = 111 :=
by sorry

end original_cost_111_l39_39945


namespace cost_of_7_enchiladas_and_6_tacos_l39_39736

theorem cost_of_7_enchiladas_and_6_tacos (e t : ℝ) 
  (h₁ : 4 * e + 5 * t = 5.00) 
  (h₂ : 6 * e + 3 * t = 5.40) : 
  7 * e + 6 * t = 7.47 := 
sorry

end cost_of_7_enchiladas_and_6_tacos_l39_39736


namespace f_has_exactly_one_zero_point_a_range_condition_l39_39994

noncomputable def f (x : ℝ) : ℝ := - (1 / 2) * Real.log x + 2 / (x + 1)

theorem f_has_exactly_one_zero_point :
  ∃! x : ℝ, 1 < x ∧ x < Real.exp 2 ∧ f x = 0 := sorry

theorem a_range_condition (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (1 / Real.exp 1) 1 → ∀ t : ℝ, t ∈ Set.Icc (1 / 2) 2 → f x ≥ t^3 - t^2 - 2 * a * t + 2) → a ≥ 5 / 4 := sorry

end f_has_exactly_one_zero_point_a_range_condition_l39_39994


namespace zoey_holidays_in_a_year_l39_39442

-- Given conditions as definitions
def holidays_per_month : ℕ := 2
def months_in_a_year : ℕ := 12

-- Definition of the total holidays in a year
def total_holidays_in_year : ℕ := holidays_per_month * months_in_a_year

-- Proof statement
theorem zoey_holidays_in_a_year : total_holidays_in_year = 24 := 
by
  sorry

end zoey_holidays_in_a_year_l39_39442


namespace equal_potatoes_l39_39720

theorem equal_potatoes (total_potatoes : ℕ) (total_people : ℕ) (h_potatoes : total_potatoes = 24) (h_people : total_people = 3) :
  (total_potatoes / total_people) = 8 :=
by {
  sorry
}

end equal_potatoes_l39_39720


namespace gingerbread_percentage_red_hats_l39_39470

def total_gingerbread_men (n_red_hats : ℕ) (n_blue_boots : ℕ) (n_both : ℕ) : ℕ :=
  n_red_hats + n_blue_boots - n_both

def percentage_with_red_hats (n_red_hats : ℕ) (total : ℕ) : ℕ :=
  (n_red_hats * 100) / total

theorem gingerbread_percentage_red_hats 
  (n_red_hats : ℕ) (n_blue_boots : ℕ) (n_both : ℕ)
  (h_red_hats : n_red_hats = 6)
  (h_blue_boots : n_blue_boots = 9)
  (h_both : n_both = 3) : 
  percentage_with_red_hats n_red_hats (total_gingerbread_men n_red_hats n_blue_boots n_both) = 50 := by
  sorry

end gingerbread_percentage_red_hats_l39_39470


namespace smallest_integer_condition_l39_39643

def is_not_prime (n : Nat) : Prop := ¬ Nat.Prime n

def is_not_square (n : Nat) : Prop :=
  ∀ m : Nat, m * m ≠ n

def has_no_prime_factor_less_than (n k : Nat) : Prop :=
  ∀ p : Nat, Nat.Prime p → p < k → ¬ (p ∣ n)

theorem smallest_integer_condition :
  ∃ n : Nat, n > 0 ∧ is_not_prime n ∧ is_not_square n ∧ has_no_prime_factor_less_than n 70 ∧ n = 5183 :=
by {
  sorry
}

end smallest_integer_condition_l39_39643


namespace purchased_only_A_l39_39926

-- Definitions for the conditions
def total_B (x : ℕ) := x + 500
def total_A (y : ℕ) := 2 * y

-- Question formulated in Lean 4
theorem purchased_only_A : 
  ∃ C : ℕ, (∀ x y : ℕ, 2 * x = 500 → y = total_B x → 2 * y = total_A y → C = total_A y - 500) ∧ C = 1000 :=
  sorry

end purchased_only_A_l39_39926


namespace square_lawn_area_l39_39580

theorem square_lawn_area (map_scale : ℝ) (map_edge_length_cm : ℝ) (actual_edge_length_m : ℝ) (actual_area_m2 : ℝ) 
  (h1 : map_scale = 1 / 5000) 
  (h2 : map_edge_length_cm = 4) 
  (h3 : actual_edge_length_m = (map_edge_length_cm / map_scale) / 100)
  (h4 : actual_area_m2 = actual_edge_length_m^2)
  : actual_area_m2 = 400 := 
by 
  sorry

end square_lawn_area_l39_39580


namespace find_b_over_a_l39_39896

variables {a b c : ℝ}
variables {b₃ b₇ b₁₁ : ℝ}

-- Conditions
def roots_of_quadratic (a b c b₃ b₁₁ : ℝ) : Prop :=
  ∃ p q, p + q = -b / a ∧ p * q = c / a ∧ (p = b₃ ∨ p = b₁₁) ∧ (q = b₃ ∨ q = b₁₁)

def middle_term_value (b₇ : ℝ) : Prop :=
  b₇ = 3

-- The statement to be proved
theorem find_b_over_a
  (h1 : roots_of_quadratic a b c b₃ b₁₁)
  (h2 : middle_term_value b₇)
  (h3 : b₃ + b₁₁ = 2 * b₇) :
  b / a = -6 :=
sorry

end find_b_over_a_l39_39896


namespace molecular_weight_of_complex_compound_l39_39588

def molecular_weight (n : ℕ) (N_w : ℝ) (o : ℕ) (O_w : ℝ) (h : ℕ) (H_w : ℝ) (p : ℕ) (P_w : ℝ) : ℝ :=
  (n * N_w) + (o * O_w) + (h * H_w) + (p * P_w)

theorem molecular_weight_of_complex_compound :
  molecular_weight 2 14.01 5 16.00 3 1.01 1 30.97 = 142.02 :=
by
  sorry

end molecular_weight_of_complex_compound_l39_39588


namespace equal_constants_l39_39498

theorem equal_constants (a b : ℝ) :
  (∃ᶠ n in at_top, ⌊a * n + b⌋ ≥ ⌊a + b * n⌋) →
  (∃ᶠ m in at_top, ⌊a + b * m⌋ ≥ ⌊a * m + b⌋) →
  a = b :=
by
  sorry

end equal_constants_l39_39498


namespace exists_triangle_free_not_4_colorable_l39_39649

/-- Define a graph as a structure with vertices and edges. -/
structure Graph (V : Type*) :=
  (adj : V → V → Prop)
  (symm : ∀ x y, adj x y → adj y x)
  (irreflexive : ∀ x, ¬adj x x)

/-- A definition of triangle-free graph. -/
def triangle_free {V : Type*} (G : Graph V) : Prop :=
  ∀ (a b c : V), G.adj a b → G.adj b c → G.adj c a → false

/-- A definition that a graph cannot be k-colored. -/
def not_k_colorable {V : Type*} (G : Graph V) (k : ℕ) : Prop :=
  ¬∃ (f : V → ℕ), (∀ (v : V), f v < k) ∧ (∀ (v w : V), G.adj v w → f v ≠ f w)

/-- There exists a triangle-free graph that is not 4-colorable. -/
theorem exists_triangle_free_not_4_colorable : ∃ (V : Type*) (G : Graph V), triangle_free G ∧ not_k_colorable G 4 := 
sorry

end exists_triangle_free_not_4_colorable_l39_39649


namespace strudel_price_l39_39975

def initial_price := 80
def first_increment (P0 : ℕ) := P0 * 3 / 2
def second_increment (P1 : ℕ) := P1 * 3 / 2
def final_price (P2 : ℕ) := P2 / 2

theorem strudel_price (P0 : ℕ) (P1 : ℕ) (P2 : ℕ) (Pf : ℕ)
  (h0 : P0 = initial_price)
  (h1 : P1 = first_increment P0)
  (h2 : P2 = second_increment P1)
  (hf : Pf = final_price P2) :
  Pf = 90 :=
sorry

end strudel_price_l39_39975


namespace quotient_sum_40_5_l39_39907

theorem quotient_sum_40_5 : (40 + 5) / 5 = 9 := by
  sorry

end quotient_sum_40_5_l39_39907


namespace supplement_comp_greater_l39_39475

theorem supplement_comp_greater {α β : ℝ} (h : α + β = 90) : 180 - α = β + 90 :=
by
  sorry

end supplement_comp_greater_l39_39475


namespace HunterScoreIs45_l39_39957

variable (G J H : ℕ)
variable (h1 : G = J + 10)
variable (h2 : J = 2 * H)
variable (h3 : G = 100)

theorem HunterScoreIs45 : H = 45 := by
  sorry

end HunterScoreIs45_l39_39957


namespace max_sum_after_swap_l39_39423

section
variables (a1 a2 a3 b1 b2 b3 c1 c2 c3 : ℕ)
  (h1 : 100 * a1 + 10 * b1 + c1 + 100 * a2 + 10 * b2 + c2 + 100 * a3 + 10 * b3 + c3 = 2019)
  (h2 : 1 ≤ a1 ∧ a1 ≤ 9 ∧ 0 ≤ b1 ∧ b1 ≤ 9 ∧ 0 ≤ c1 ∧ c1 ≤ 9)
  (h3 : 1 ≤ a2 ∧ a2 ≤ 9 ∧ 0 ≤ b2 ∧ b2 ≤ 9 ∧ 0 ≤ c2 ∧ c2 ≤ 9)
  (h4 : 1 ≤ a3 ∧ a3 ≤ 9 ∧ 0 ≤ b3 ∧ b3 ≤ 9 ∧ 0 ≤ c3 ∧ c3 ≤ 9)

theorem max_sum_after_swap : 100 * c1 + 10 * b1 + a1 + 100 * c2 + 10 * b2 + a2 + 100 * c3 + 10 * b3 + a3 ≤ 2118 := 
  sorry

end

end max_sum_after_swap_l39_39423


namespace fraction_product_l39_39100

theorem fraction_product :
  (5 / 8) * (7 / 9) * (11 / 13) * (3 / 5) * (17 / 19) * (8 / 15) = 14280 / 1107000 :=
by sorry

end fraction_product_l39_39100


namespace find_q_value_l39_39018

theorem find_q_value (q : ℚ) (x y : ℚ) (hx : x = 5 - q) (hy : y = 3*q - 1) : x = 3*y → q = 4/5 :=
by
  sorry

end find_q_value_l39_39018


namespace complex_quadrant_l39_39794

open Complex

theorem complex_quadrant 
  (z : ℂ) 
  (h : (1 - I) ^ 2 / z = 1 + I) :
  z = -1 - I :=
by
  sorry

end complex_quadrant_l39_39794


namespace largest_value_of_c_l39_39903

noncomputable def g (x : ℝ) : ℝ := x^2 + 3 * x + 1

theorem largest_value_of_c :
  ∃ (c : ℝ), (∀ (x d : ℝ), |x - 1| ≤ d ∧ 0 < d ∧ 0 < c → |g x - 1| ≤ c) ∧ (∀ (c' : ℝ), (∀ (x d : ℝ), |x - 1| ≤ d ∧ 0 < d ∧ 0 < c' → |g x - 1| ≤ c') → c' ≤ c) :=
sorry

end largest_value_of_c_l39_39903


namespace triangle_is_isosceles_l39_39026

theorem triangle_is_isosceles
  (A B C : ℝ)
  (h_triangle : A + B + C = π)
  (h_condition : 2 * Real.cos B * Real.sin C = Real.sin A) :
  B = C :=
sorry

end triangle_is_isosceles_l39_39026


namespace correct_average_l39_39816

theorem correct_average 
(n : ℕ) (avg1 avg2 avg3 : ℝ): 
  n = 10 
  → avg1 = 40.2 
  → avg2 = avg1
  → avg3 = avg1
  → avg1 = avg3 :=
by 
  intros hn h_avg1 h_avg2 h_avg3
  sorry

end correct_average_l39_39816


namespace find_m_l39_39401

-- Definitions for the sets A and B
def A (m : ℝ) : Set ℝ := {3, 4, 4 * m - 4}
def B (m : ℝ) : Set ℝ := {3, m^2}

-- Problem statement
theorem find_m {m : ℝ} (h : B m ⊆ A m) : m = -2 :=
sorry

end find_m_l39_39401


namespace least_not_lucky_multiple_of_6_l39_39787

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

theorem least_not_lucky_multiple_of_6 : ∃ k : ℕ, k > 0 ∧ k % 6 = 0 ∧ ¬ is_lucky k ∧ ∀ m : ℕ, m > 0 ∧ m % 6 = 0 ∧ ¬ is_lucky m → k ≤ m :=
  sorry

end least_not_lucky_multiple_of_6_l39_39787


namespace find_angle_BAC_l39_39073

-- Definitions and Hypotheses
variables (A B C P : Type) (AP PC AB AC : Real) (angle_BPC : Real)

-- Hypotheses
-- AP = PC
-- AB = AC
-- angle BPC = 120 
axiom AP_eq_PC : AP = PC
axiom AB_eq_AC : AB = AC
axiom angle_BPC_eq_120 : angle_BPC = 120

-- Theorem
theorem find_angle_BAC (AP_eq_PC : AP = PC) (AB_eq_AC : AB = AC) (angle_BPC_eq_120 : angle_BPC = 120) : angle_BAC = 60 :=
sorry

end find_angle_BAC_l39_39073


namespace determine_time_l39_39771

variable (g a V_0 V S t : ℝ)

def velocity_eq : Prop := V = (g + a) * t + V_0
def displacement_eq : Prop := S = 1 / 2 * (g + a) * t^2 + V_0 * t

theorem determine_time (h1 : velocity_eq g a V_0 V t) (h2 : displacement_eq g a V_0 S t) :
  t = 2 * S / (V + V_0) := 
sorry

end determine_time_l39_39771


namespace problem_1_problem_2_l39_39198

noncomputable def f (x a : ℝ) : ℝ := abs (x + a) + abs (x - 2)

-- (1) Prove that, given f(x) and a = -3, the solution set for f(x) ≥ 3 is (-∞, 1] ∪ [4, +∞)
theorem problem_1 (x : ℝ) : 
  (∃ (a : ℝ), a = -3 ∧ f x a ≥ 3) ↔ (x ≤ 1 ∨ x ≥ 4) :=
sorry

-- (2) Prove that for f(x) to be ≥ 3 for all x, the range of a is a ≥ 1 or a ≤ -5
theorem problem_2 : 
  (∀ (x : ℝ), f x a ≥ 3) ↔ (a ≥ 1 ∨ a ≤ -5) :=
sorry

end problem_1_problem_2_l39_39198


namespace B_won_third_four_times_l39_39993

noncomputable def first_place := 5
noncomputable def second_place := 2
noncomputable def third_place := 1

structure ContestantScores :=
  (A_score : ℕ)
  (B_score : ℕ)
  (C_score : ℕ)

def competition_results (A B C : ContestantScores) (a b c : ℕ) : Prop :=
  A.A_score = 26 ∧ B.B_score = 11 ∧ C.C_score = 11 ∧ 1 = 1 ∧ -- B won first place once is synonymous to holding true
  a > b ∧ b > c ∧ a = 5 ∧ b = 2 ∧ c = 1

theorem B_won_third_four_times :
  ∃ (A B C : ContestantScores), competition_results A B C first_place second_place third_place → 
  B.B_score = 4 * third_place + first_place := 
sorry

end B_won_third_four_times_l39_39993


namespace proof_solution_l39_39474

def proof_problem : Prop :=
  ∀ (s c p d : ℝ), 
  4 * s + 8 * c + p + 2 * d = 5.00 → 
  5 * s + 11 * c + p + 3 * d = 6.50 → 
  s + c + p + d = 1.50

theorem proof_solution : proof_problem :=
  sorry

end proof_solution_l39_39474


namespace pattern_formula_l39_39830

theorem pattern_formula (n : ℤ) : n * (n + 2) = (n + 1) ^ 2 - 1 := 
by sorry

end pattern_formula_l39_39830


namespace inv_f_of_neg3_l39_39368

def f (x : Real) : Real := 5 - 2 * x

theorem inv_f_of_neg3 : f⁻¹ (-3) = 4 :=
by
  sorry

end inv_f_of_neg3_l39_39368


namespace probability_P_plus_S_mod_7_correct_l39_39284

noncomputable def probability_P_plus_S_mod_7 : ℚ :=
  let n := 60
  let total_ways := (n * (n - 1)) / 2
  let num_special_pairs := total_ways - ((52 * 51) / 2)
  num_special_pairs / total_ways

theorem probability_P_plus_S_mod_7_correct :
  probability_P_plus_S_mod_7 = 148 / 590 :=
by
  rw [probability_P_plus_S_mod_7]
  sorry

end probability_P_plus_S_mod_7_correct_l39_39284


namespace range_of_m_l39_39129

theorem range_of_m (m : ℝ) : (-6 < m ∧ m < 2) ↔ ∃ x : ℝ, |x - m| + |x + 2| < 4 :=
by sorry

end range_of_m_l39_39129


namespace matt_assignment_problems_l39_39308

theorem matt_assignment_problems (P : ℕ) (h : 5 * P - 2 * P = 60) : P = 20 :=
by
  sorry

end matt_assignment_problems_l39_39308


namespace combined_proposition_range_l39_39263

def p (a : ℝ) : Prop := ∀ x ∈ ({1, 2} : Set ℝ), 3 * x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem combined_proposition_range (a : ℝ) : 
  (p a ∧ q a) ↔ (a ≤ -2 ∨ (1 ≤ a ∧ a ≤ 3)) := 
  sorry

end combined_proposition_range_l39_39263


namespace triangle_inequality_l39_39712

theorem triangle_inequality
  (A B C : ℝ)
  (hA : 0 < A)
  (hB : 0 < B)
  (hC : 0 < C)
  (hABC : A + B + C = Real.pi) :
  Real.sin (3 * A / 2) + Real.sin (3 * B / 2) + Real.sin (3 * C / 2) ≤
  Real.cos ((A - B) / 2) + Real.cos ((B - C) / 2) + Real.cos ((C - A) / 2) :=
by
  sorry

end triangle_inequality_l39_39712


namespace smallest_bob_number_l39_39724

-- Definitions and conditions
def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def prime_factors (n : ℕ) : Set ℕ := { p | is_prime p ∧ p ∣ n }

def alice_number := 36
def bob_number (m : ℕ) : Prop := prime_factors alice_number ⊆ prime_factors m

-- Proof problem statement
theorem smallest_bob_number :
  ∃ m, bob_number m ∧ m = 6 :=
sorry

end smallest_bob_number_l39_39724


namespace cube_expression_l39_39796

theorem cube_expression (a : ℝ) (h : (a + 1/a)^2 = 5) : a^3 + 1/a^3 = 2 * Real.sqrt 5 :=
by
  sorry

end cube_expression_l39_39796


namespace sum_of_box_weights_l39_39125

theorem sum_of_box_weights (heavy_box_weight : ℚ) (difference : ℚ) 
  (h1 : heavy_box_weight = 14 / 15) (h2 : difference = 1 / 10) :
  heavy_box_weight + (heavy_box_weight - difference) = 53 / 30 := 
  by
  sorry

end sum_of_box_weights_l39_39125


namespace polynomial_equivalence_l39_39095

variable (x : ℝ) -- Define variable x

-- Define the expressions.
def expr1 := (3 * x^2 + 5 * x + 8) * (x + 2)
def expr2 := (x + 2) * (x^2 + 5 * x - 72)
def expr3 := (4 * x - 15) * (x + 2) * (x + 6)

-- Define the expression to be proved.
def original_expr := expr1 - expr2 + expr3
def simplified_expr := 6 * x^3 + 21 * x^2 + 18 * x

-- The theorem to prove the equivalence of the original and simplified expressions.
theorem polynomial_equivalence : original_expr = simplified_expr :=
by sorry -- proof to be filled in

end polynomial_equivalence_l39_39095


namespace months_decreasing_l39_39407

noncomputable def stock_decrease (m : ℕ) : Prop :=
  2 * m + 2 * 8 = 18

theorem months_decreasing (m : ℕ) (h : stock_decrease m) : m = 1 :=
by
  exact sorry

end months_decreasing_l39_39407


namespace sector_COD_area_ratio_l39_39434

-- Define the given angles
def angle_AOC : ℝ := 30
def angle_DOB : ℝ := 45
def angle_AOB : ℝ := 180

-- Define the full circle angle
def full_circle_angle : ℝ := 360

-- Calculate the angle COD
def angle_COD : ℝ := angle_AOB - angle_AOC - angle_DOB

-- State the ratio of the area of sector COD to the area of the circle
theorem sector_COD_area_ratio :
  angle_COD / full_circle_angle = 7 / 24 := by
  sorry

end sector_COD_area_ratio_l39_39434


namespace largest_multiple_of_18_with_8_and_0_digits_l39_39395

theorem largest_multiple_of_18_with_8_and_0_digits :
  ∃ m : ℕ, (∀ d ∈ (m.digits 10), d = 8 ∨ d = 0) ∧ (m % 18 = 0) ∧ (m = 8888888880) ∧ (m / 18 = 493826048) :=
by sorry

end largest_multiple_of_18_with_8_and_0_digits_l39_39395


namespace inequality_proof_l39_39364

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)
variable (habc : a * b * c = 1)

theorem inequality_proof :
  (a + 1 / b)^2 + (b + 1 / c)^2 + (c + 1 / a)^2 ≥ 3 * (a + b + c + 1) :=
by
  sorry

end inequality_proof_l39_39364


namespace redistribution_l39_39834

/-
Given:
- b = (12 / 13) * a
- c = (2 / 3) * b
- Person C will contribute 9 dollars based on the amount each person spent

Prove:
- Person C gives 6 dollars to Person A.
- Person C gives 3 dollars to Person B.
-/

theorem redistribution (a b c : ℝ) (h1 : b = (12 / 13) * a) (h2 : c = (2 / 3) * b) : 
  ∃ (x y : ℝ), x + y = 9 ∧ x = 6 ∧ y = 3 :=
by
  sorry

end redistribution_l39_39834


namespace min_points_dodecahedron_min_points_icosahedron_l39_39702

-- Definitions for the dodecahedron
def dodecahedron_faces : ℕ := 12
def vertices_per_face_dodecahedron : ℕ := 3

-- Prove the minimum number of points to mark each face of a dodecahedron
theorem min_points_dodecahedron (n : ℕ) (h : 3 * n >= dodecahedron_faces) : n >= 4 :=
sorry

-- Definitions for the icosahedron
def icosahedron_faces : ℕ := 20
def icosahedron_vertices : ℕ := 12

-- Prove the minimum number of points to mark each face of an icosahedron
theorem min_points_icosahedron (n : ℕ) (h : n >= 6) : n = 6 :=
sorry

end min_points_dodecahedron_min_points_icosahedron_l39_39702


namespace reimbursement_diff_l39_39820

/-- Let Tom, Emma, and Harry share equally the costs for a group activity.
- Tom paid $95
- Emma paid $140
- Harry paid $165
If Tom and Emma are to reimburse Harry to ensure all expenses are shared equally,
prove that e - t = -45 where e is the amount Emma gives Harry and t is the amount Tom gives Harry.
-/
theorem reimbursement_diff :
  let tom_paid := 95
  let emma_paid := 140
  let harry_paid := 165
  let total_cost := tom_paid + emma_paid + harry_paid
  let equal_share := total_cost / 3
  let t := equal_share - tom_paid
  let e := equal_share - emma_paid
  e - t = -45 :=
by {
  sorry
}

end reimbursement_diff_l39_39820


namespace work_done_isothermal_l39_39958

variable (n : ℕ) (R T : ℝ) (P DeltaV : ℝ)

-- Definitions based on the conditions
def isobaric_work (P DeltaV : ℝ) := P * DeltaV

noncomputable def isobaric_heat (P DeltaV : ℝ) : ℝ :=
  (5 / 2) * P * DeltaV

noncomputable def isothermal_work (Q_iso : ℝ) : ℝ := Q_iso

theorem work_done_isothermal :
  ∃ (n R : ℝ) (P DeltaV : ℝ),
    isobaric_work P DeltaV = 20 ∧
    isothermal_work (isobaric_heat P DeltaV) = 50 :=
by 
  sorry

end work_done_isothermal_l39_39958


namespace inequality_solution_set_inequality_range_of_a_l39_39115

theorem inequality_solution_set (a : ℝ) (x : ℝ) (h : a = -8) :
  (|x - 3| + |x + 2| ≤ |a + 1|) ↔ (-3 ≤ x ∧ x ≤ 4) :=
by sorry

theorem inequality_range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x + 2| ≤ |a + 1|) ↔ (a ≤ -6 ∨ a ≥ 4) :=
by sorry

end inequality_solution_set_inequality_range_of_a_l39_39115


namespace monotonically_decreasing_iff_l39_39847

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then (3 * a - 2) * x + 6 * a - 1 else a^x

theorem monotonically_decreasing_iff (a : ℝ) : 
  (∀ x y : ℝ, x ≤ y → f a y ≤ f a x) ↔ (3/8 ≤ a ∧ a < 2/3) :=
sorry

end monotonically_decreasing_iff_l39_39847


namespace tom_initial_money_l39_39740

-- Defining the given values
def super_nintendo_value : ℝ := 150
def store_percentage : ℝ := 0.80
def nes_price : ℝ := 160
def game_value : ℝ := 30
def change_received : ℝ := 10

-- Calculate the credit received for the Super Nintendo
def credit_received := store_percentage * super_nintendo_value

-- Calculate the remaining amount Tom needs to pay for the NES after using the credit
def remaining_amount := nes_price - credit_received

-- Calculate the total amount Tom needs to pay, including the game value
def total_amount_needed := remaining_amount + game_value

-- Proving that the initial money Tom gave is $80
theorem tom_initial_money : total_amount_needed + change_received = 80 :=
by
    sorry

end tom_initial_money_l39_39740


namespace bananas_to_pears_l39_39402

theorem bananas_to_pears:
  (∀ b a o p : ℕ, 
    6 * b = 4 * a → 
    5 * a = 3 * o → 
    4 * o = 7 * p → 
    36 * b = 28 * p) :=
by
  intros b a o p h1 h2 h3
  -- We need to prove 36 * b = 28 * p under the given conditions
  sorry

end bananas_to_pears_l39_39402


namespace convert_10203_base4_to_base10_l39_39456

def base4_to_base10 (n : ℕ) (d₀ d₁ d₂ d₃ d₄ : ℕ) : ℕ :=
  d₄ * 4^4 + d₃ * 4^3 + d₂ * 4^2 + d₁ * 4^1 + d₀ * 4^0

theorem convert_10203_base4_to_base10 :
  base4_to_base10 10203 3 0 2 0 1 = 291 :=
by
  -- proof goes here
  sorry

end convert_10203_base4_to_base10_l39_39456


namespace original_square_side_length_l39_39739

theorem original_square_side_length (a : ℕ) (initial_thickness final_thickness : ℕ) (side_length_reduction_factor thickness_doubling_factor : ℕ) (s : ℕ) :
  a = 3 →
  final_thickness = 16 →
  initial_thickness = 1 →
  side_length_reduction_factor = 16 →
  thickness_doubling_factor = 16 →
  s * s = side_length_reduction_factor * a * a →
  s = 12 :=
by
  intros ha hfinal_thickness hin_initial_thickness hside_length_reduction_factor hthickness_doubling_factor h_area_equiv
  sorry

end original_square_side_length_l39_39739


namespace complement_set_solution_l39_39388

open Set Real

theorem complement_set_solution :
  let M := {x : ℝ | (1 + x) / (1 - x) > 0}
  compl M = {x : ℝ | x ≤ -1 ∨ x ≥ 1} :=
by
  sorry

end complement_set_solution_l39_39388


namespace responses_needed_l39_39264

-- Define the given conditions
def rate : ℝ := 0.80
def num_mailed : ℕ := 375

-- Statement to prove
theorem responses_needed :
  rate * num_mailed = 300 := by
  sorry

end responses_needed_l39_39264


namespace greatest_possible_value_of_median_l39_39021

-- Given conditions as definitions
variables (k m r s t : ℕ)

-- condition 1: The average (arithmetic mean) of the 5 integers is 10
def avg_is_10 : Prop := k + m + r + s + t = 50

-- condition 2: The integers are in a strictly increasing order
def increasing_order : Prop := k < m ∧ m < r ∧ r < s ∧ s < t

-- condition 3: t is 20
def t_is_20 : Prop := t = 20

-- The main statement to prove
theorem greatest_possible_value_of_median : 
  avg_is_10 k m r s t → 
  increasing_order k m r s t → 
  t_is_20 t → 
  r = 13 :=
by
  intros
  sorry

end greatest_possible_value_of_median_l39_39021


namespace total_hours_l39_39531

variable (K : ℕ) (P : ℕ) (M : ℕ)

-- Conditions:
axiom h1 : P = 2 * K
axiom h2 : P = (1 / 3 : ℝ) * M
axiom h3 : M = K + 105

-- Goal: Proving the total number of hours is 189
theorem total_hours : K + P + M = 189 := by
  sorry

end total_hours_l39_39531


namespace nancy_age_l39_39224

variable (n g : ℕ)

theorem nancy_age (h1 : g = 10 * n) (h2 : g - n = 45) : n = 5 :=
by
  sorry

end nancy_age_l39_39224


namespace cos_225_eq_l39_39285

noncomputable def angle_in_radians (deg : ℝ) : ℝ := deg * (Real.pi / 180)

theorem cos_225_eq :
  Real.cos (angle_in_radians 225) = - (Real.sqrt 2) / 2 := 
  sorry

end cos_225_eq_l39_39285


namespace tom_profit_calculation_l39_39139

theorem tom_profit_calculation :
  let flour_needed := 500
  let flour_per_bag := 50
  let flour_bag_cost := 20
  let salt_needed := 10
  let salt_cost_per_pound := 0.2
  let promotion_cost := 1000
  let tickets_sold := 500
  let ticket_price := 20

  let flour_bags := flour_needed / flour_per_bag
  let cost_flour := flour_bags * flour_bag_cost
  let cost_salt := salt_needed * salt_cost_per_pound
  let total_expenses := cost_flour + cost_salt + promotion_cost
  let total_revenue := tickets_sold * ticket_price

  let profit := total_revenue - total_expenses

  profit = 8798 := by
  sorry

end tom_profit_calculation_l39_39139


namespace arthur_muffins_l39_39220

variable (arthur_baked : ℕ)
variable (james_baked : ℕ := 1380)
variable (times_as_many : ℕ := 12)

theorem arthur_muffins : arthur_baked * times_as_many = james_baked -> arthur_baked = 115 := by
  sorry

end arthur_muffins_l39_39220


namespace non_parallel_lines_implies_unique_solution_l39_39397

variable (a1 b1 c1 a2 b2 c2 : ℝ)

def system_of_equations (x y : ℝ) := a1 * x + b1 * y = c1 ∧ a2 * x + b2 * y = c2

def lines_not_parallel := a1 * b2 ≠ a2 * b1

theorem non_parallel_lines_implies_unique_solution :
  lines_not_parallel a1 b1 a2 b2 → ∃! (x y : ℝ), system_of_equations a1 b1 c1 a2 b2 c2 x y :=
sorry

end non_parallel_lines_implies_unique_solution_l39_39397


namespace find_alpha_l39_39435

noncomputable def isochronous_growth (k α x₁ x₂ y₁ y₂ : ℝ) : Prop :=
  y₁ = k * x₁^α ∧
  y₂ = k * x₂^α ∧
  x₂ = 16 * x₁ ∧
  y₂ = 8 * y₁

theorem find_alpha (k x₁ x₂ y₁ y₂ : ℝ) (h : isochronous_growth k (3/4) x₁ x₂ y₁ y₂) : 3/4 = 3/4 :=
by 
  sorry

end find_alpha_l39_39435


namespace find_triples_l39_39189

theorem find_triples (x p n : ℕ) (hp : Nat.Prime p) :
  2 * x * (x + 5) = p^n + 3 * (x - 1) →
  (x = 2 ∧ p = 5 ∧ n = 2) ∨ (x = 0 ∧ p = 3 ∧ n = 1) :=
by
  sorry

end find_triples_l39_39189


namespace num_distinct_sums_of_three_distinct_elements_l39_39923

noncomputable def arith_seq_sum_of_three_distinct : Nat :=
  let a (i : Nat) : Nat := 3 * i + 1
  let lower_bound := 21
  let upper_bound := 129
  (upper_bound - lower_bound) / 3 + 1

theorem num_distinct_sums_of_three_distinct_elements : arith_seq_sum_of_three_distinct = 37 := by
  -- We are skipping the proof by using sorry
  sorry

end num_distinct_sums_of_three_distinct_elements_l39_39923


namespace scientific_notation_248000_l39_39908

theorem scientific_notation_248000 : (248000 : Float) = 2.48 * 10^5 := 
sorry

end scientific_notation_248000_l39_39908


namespace max_min_value_sum_l39_39290

noncomputable def f (x : ℝ) : ℝ := (x^2 - 4*x) * Real.sin (x - 2) + x + 1

theorem max_min_value_sum (M m : ℝ) 
  (hM : ∀ x ∈ Set.Icc (-1 : ℝ) 5, f x ≤ M)
  (hm : ∀ x ∈ Set.Icc (-1 : ℝ) 5, f x ≥ m)
  (hM_max : ∃ x ∈ Set.Icc (-1 : ℝ) 5, f x = M)
  (hm_min : ∃ x ∈ Set.Icc (-1 : ℝ) 5, f x = m)
  : M + m = 6 :=
sorry

end max_min_value_sum_l39_39290


namespace queenie_total_earnings_l39_39105

-- Define the conditions
def daily_wage : ℕ := 150
def overtime_wage_per_hour : ℕ := 5
def days_worked : ℕ := 5
def overtime_hours : ℕ := 4

-- Define the main problem
theorem queenie_total_earnings : 
  (daily_wage * days_worked + overtime_wage_per_hour * overtime_hours) = 770 :=
by
  sorry

end queenie_total_earnings_l39_39105


namespace car_X_travel_distance_l39_39641

def car_distance_problem (speed_X speed_Y : ℝ) (delay : ℝ) : ℝ :=
  let t := 7 -- duration in hours computed in the provided solution
  speed_X * t

theorem car_X_travel_distance
  (speed_X speed_Y : ℝ) (delay : ℝ)
  (h_speed_X : speed_X = 35) (h_speed_Y : speed_Y = 39) (h_delay : delay = 48 / 60) :
  car_distance_problem speed_X speed_Y delay = 245 :=
by
  rw [h_speed_X, h_speed_Y, h_delay]
  -- compute the given car distance problem using the values provided
  sorry

end car_X_travel_distance_l39_39641


namespace side_length_of_square_l39_39774

theorem side_length_of_square (s : ℝ) (h : s^2 = 100) : s = 10 := 
sorry

end side_length_of_square_l39_39774


namespace trigonometric_identity_l39_39051

theorem trigonometric_identity (α : ℝ) (h : Real.sin (α + Real.pi / 6) = 1 / 3) :
  Real.cos (2 * α - 2 * Real.pi / 3) = -7 / 9 :=
  sorry

end trigonometric_identity_l39_39051


namespace steve_num_nickels_l39_39098

-- Definitions for the conditions
def num_nickels (N : ℕ) : Prop :=
  ∃ D Q : ℕ, D = N + 4 ∧ Q = D + 3 ∧ 5 * N + 10 * D + 25 * Q + 5 = 380

-- Statement of the problem
theorem steve_num_nickels : num_nickels 4 :=
sorry

end steve_num_nickels_l39_39098


namespace number_greater_than_neg_one_by_two_l39_39876

/-- Theorem: The number that is greater than -1 by 2 is 1. -/
theorem number_greater_than_neg_one_by_two : -1 + 2 = 1 :=
by
  sorry

end number_greater_than_neg_one_by_two_l39_39876


namespace units_digit_smallest_n_l39_39522

theorem units_digit_smallest_n (n : ℕ) (h1 : 7 * n ≥ 10^2015) (h2 : 7 * (n - 1) < 10^2015) : (n % 10) = 6 :=
sorry

end units_digit_smallest_n_l39_39522


namespace isabel_piggy_bank_l39_39200

theorem isabel_piggy_bank:
  ∀ (initial_amount spent_on_toy spent_on_book remaining_amount : ℕ),
  initial_amount = 204 →
  spent_on_toy = initial_amount / 2 →
  remaining_amount = initial_amount - spent_on_toy →
  spent_on_book = remaining_amount / 2 →
  remaining_amount - spent_on_book = 51 :=
by
  sorry

end isabel_piggy_bank_l39_39200


namespace Lacy_correct_percentage_l39_39956

def problems_exam (y : ℕ) := 10 * y
def problems_section1 (y : ℕ) := 6 * y
def problems_section2 (y : ℕ) := 4 * y
def missed_section1 (y : ℕ) := 2 * y
def missed_section2 (y : ℕ) := y
def solved_section1 (y : ℕ) := problems_section1 y - missed_section1 y
def solved_section2 (y : ℕ) := problems_section2 y - missed_section2 y
def total_solved (y : ℕ) := solved_section1 y + solved_section2 y
def percent_correct (y : ℕ) := (total_solved y : ℚ) / (problems_exam y) * 100

theorem Lacy_correct_percentage (y : ℕ) : percent_correct y = 70 := by
  -- Proof would go here
  sorry

end Lacy_correct_percentage_l39_39956


namespace exists_pair_sum_ends_with_last_digit_l39_39126

theorem exists_pair_sum_ends_with_last_digit (a : ℕ → ℕ) (h_distinct: ∀ i j, (i ≠ j) → a i ≠ a j) (h_range: ∀ i, a i < 10) : ∀ (n : ℕ), n < 10 → ∃ i j, (i ≠ j) ∧ (a i + a j) % 10 = n % 10 :=
by sorry

end exists_pair_sum_ends_with_last_digit_l39_39126


namespace divide_angle_into_parts_l39_39454

-- Definitions based on the conditions
def given_angle : ℝ := 19

/-- 
Theorem: An angle of 19 degrees can be divided into 19 equal parts using a compass and a ruler,
and each part will measure 1 degree.
-/
theorem divide_angle_into_parts (angle : ℝ) (n : ℕ) (h1 : angle = given_angle) (h2 : n = 19) : angle / n = 1 :=
by
  -- Proof to be filled out
  sorry

end divide_angle_into_parts_l39_39454


namespace student_correct_numbers_l39_39562

theorem student_correct_numbers (x y : ℕ) 
  (h1 : (10 * x + 5) * y = 4500)
  (h2 : (10 * x + 3) * y = 4380) : 
  (10 * x + 5 = 75 ∧ y = 60) :=
by 
  sorry

end student_correct_numbers_l39_39562


namespace find_father_age_l39_39063

variable (M F : ℕ)

noncomputable def age_relation_1 : Prop := M = (2 / 5) * F
noncomputable def age_relation_2 : Prop := M + 5 = (1 / 2) * (F + 5)

theorem find_father_age (h1 : age_relation_1 M F) (h2 : age_relation_2 M F) : F = 25 := by
  sorry

end find_father_age_l39_39063


namespace mandy_total_cost_after_discount_l39_39469

-- Define the conditions
def packs_black_shirts : ℕ := 6
def packs_yellow_shirts : ℕ := 8
def packs_green_socks : ℕ := 5

def items_per_pack_black_shirts : ℕ := 7
def items_per_pack_yellow_shirts : ℕ := 4
def items_per_pack_green_socks : ℕ := 5

def cost_per_pack_black_shirts : ℕ := 25
def cost_per_pack_yellow_shirts : ℕ := 15
def cost_per_pack_green_socks : ℕ := 10

def discount_rate : ℚ := 0.10

-- Calculate the total number of each type of item
def total_black_shirts : ℕ := packs_black_shirts * items_per_pack_black_shirts
def total_yellow_shirts : ℕ := packs_yellow_shirts * items_per_pack_yellow_shirts
def total_green_socks : ℕ := packs_green_socks * items_per_pack_green_socks

-- Calculate the total cost before discount
def total_cost_before_discount : ℕ :=
  (packs_black_shirts * cost_per_pack_black_shirts) +
  (packs_yellow_shirts * cost_per_pack_yellow_shirts) +
  (packs_green_socks * cost_per_pack_green_socks)

-- Calculate the total cost after discount
def discount_amount : ℚ := discount_rate * total_cost_before_discount
def total_cost_after_discount : ℚ := total_cost_before_discount - discount_amount

-- Problem to prove: Total cost after discount is $288
theorem mandy_total_cost_after_discount : total_cost_after_discount = 288 := by
  sorry

end mandy_total_cost_after_discount_l39_39469


namespace number_square_roots_l39_39780

theorem number_square_roots (a x : ℤ) (h1 : x = (2 * a + 3) ^ 2) (h2 : x = (a - 18) ^ 2) : x = 169 :=
by 
  sorry

end number_square_roots_l39_39780


namespace inequality_solution_l39_39087

theorem inequality_solution (x : ℝ) : 3 * x^2 - x > 9 ↔ x < -3 ∨ x > 1 := by
  sorry

end inequality_solution_l39_39087


namespace shirts_sold_l39_39523

theorem shirts_sold (S : ℕ) (H_total : 69 = 7 * 7 + 5 * S) : S = 4 :=
by
  sorry -- Placeholder for the proof

end shirts_sold_l39_39523


namespace profit_percentage_correct_l39_39922

noncomputable def cost_price : ℝ := 47.50
noncomputable def selling_price : ℝ := 70
noncomputable def list_price : ℝ := selling_price / 0.95
noncomputable def profit : ℝ := selling_price - cost_price
noncomputable def profit_percentage : ℝ := (profit / cost_price) * 100

theorem profit_percentage_correct :
  abs (profit_percentage - 47.37) < 0.01 := sorry

end profit_percentage_correct_l39_39922


namespace jimin_class_students_l39_39003

theorem jimin_class_students 
    (total_distance : ℝ)
    (interval_distance : ℝ)
    (h1 : total_distance = 242)
    (h2 : interval_distance = 5.5) :
    (total_distance / interval_distance) + 1 = 45 :=
by sorry

end jimin_class_students_l39_39003


namespace div_val_is_2_l39_39000

theorem div_val_is_2 (x : ℤ) (h : 5 * x = 100) : x / 10 = 2 :=
by 
  sorry

end div_val_is_2_l39_39000


namespace cosine_of_tangent_line_at_e_l39_39514

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem cosine_of_tangent_line_at_e :
  let θ := Real.arctan 2
  Real.cos θ = Real.sqrt (1 / 5) := by
  sorry

end cosine_of_tangent_line_at_e_l39_39514


namespace find_p_l39_39800

theorem find_p (p : ℝ) (h1 : (1/2) * 15 * (3 + 15) - ((1/2) * 3 * (15 - p) + (1/2) * 15 * p) = 40) : 
  p = 12.0833 :=
by sorry

end find_p_l39_39800


namespace range_of_omega_l39_39366

noncomputable def function_with_highest_points (ω : ℝ) (x : ℝ) : ℝ :=
  2 * Real.sin (ω * x + Real.pi / 4)

theorem range_of_omega (ω : ℝ) (hω : ω > 0)
  (h : ∀ x ∈ Set.Icc 0 1, 2 * Real.sin (ω * x + Real.pi / 4) = 2) :
  Set.Icc (17 * Real.pi / 4) (25 * Real.pi / 4) :=
by
  sorry

end range_of_omega_l39_39366


namespace find_original_number_l39_39348

noncomputable def three_digit_number (d e f : ℕ) := 100 * d + 10 * e + f

/-- Given conditions and the sum S, determine the original three-digit number -/
theorem find_original_number (S : ℕ) (d e f : ℕ) (h1 : 0 ≤ d ∧ d ≤ 9)
  (h2 : 0 ≤ e ∧ e ≤ 9) (h3 : 0 ≤ f ∧ f ≤ 9) (h4 : S = 4321) :
  three_digit_number d e f = 577 :=
sorry


end find_original_number_l39_39348


namespace katya_attached_squares_perimeter_l39_39212

theorem katya_attached_squares_perimeter :
  let p1 := 100 -- Perimeter of the larger square
  let p2 := 40  -- Perimeter of the smaller square
  let s1 := p1 / 4 -- Side length of the larger square
  let s2 := p2 / 4 -- Side length of the smaller square
  let combined_perimeter_without_internal_sides := p1 + p2
  let actual_perimeter := combined_perimeter_without_internal_sides - 2 * s2
  actual_perimeter = 120 :=
by
  sorry

end katya_attached_squares_perimeter_l39_39212


namespace number_of_juniors_twice_seniors_l39_39886

variable (j s : ℕ)

theorem number_of_juniors_twice_seniors
  (h1 : (3 / 7 : ℝ) * j = (6 / 7 : ℝ) * s) : j = 2 * s := 
sorry

end number_of_juniors_twice_seniors_l39_39886


namespace quadratic_equation_equivalence_l39_39041

theorem quadratic_equation_equivalence
  (a_0 a_1 a_2 : ℝ)
  (r s : ℝ)
  (h_roots : a_0 + a_1 * r + a_2 * r^2 = 0 ∧ a_0 + a_1 * s + a_2 * s^2 = 0)
  (h_a2_nonzero : a_2 ≠ 0) :
  (∀ x, a_0 ≠ 0 ↔ a_0 + a_1 * x + a_2 * x^2 = a_0 * (1 - x / r) * (1 - x / s)) :=
sorry

end quadratic_equation_equivalence_l39_39041


namespace upsilon_value_l39_39701

theorem upsilon_value (Upsilon : ℤ) (h : 5 * (-3) = Upsilon - 3) : Upsilon = -12 :=
by
  sorry

end upsilon_value_l39_39701


namespace simplify_and_evaluate_expression_l39_39318

theorem simplify_and_evaluate_expression :
  (1 - 2 / (Real.tan (Real.pi / 3) - 1 + 1)) / 
  ((Real.tan (Real.pi / 3) - 1) ^ 2 - 2 * (Real.tan (Real.pi / 3) - 1) + 1) / 
  ((Real.tan (Real.pi / 3) - 1) ^ 2 - (Real.tan (Real.pi / 3) - 1)) = 
  (3 - Real.sqrt 3) / 3 :=
sorry

end simplify_and_evaluate_expression_l39_39318


namespace insects_remaining_l39_39610

-- Define the initial counts of spiders, ants, and ladybugs
def spiders : ℕ := 3
def ants : ℕ := 12
def ladybugs : ℕ := 8

-- Define the number of ladybugs that flew away
def ladybugs_flew_away : ℕ := 2

-- Prove the total number of remaining insects in the playground
theorem insects_remaining : (spiders + ants + ladybugs - ladybugs_flew_away) = 21 := by
  -- Expand the definitions and compute the result
  sorry

end insects_remaining_l39_39610


namespace intersection_is_correct_l39_39447

def A : Set ℕ := {1, 2, 4, 6, 8}
def B : Set ℕ := {x | ∃ k ∈ A, x = 2 * k}

theorem intersection_is_correct : A ∩ B = {2, 4, 8} :=
by
  sorry

end intersection_is_correct_l39_39447


namespace initial_men_in_hostel_l39_39789

theorem initial_men_in_hostel (x : ℕ) (h1 : 36 * x = 45 * (x - 50)) : x = 250 := 
  sorry

end initial_men_in_hostel_l39_39789


namespace problem_statement_l39_39071

theorem problem_statement (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) 
  (h_eq : x + y + z = 1/x + 1/y + 1/z) : 
  x + y + z ≥ Real.sqrt ((x * y + 1) / 2) + Real.sqrt ((y * z + 1) / 2) + Real.sqrt ((z * x + 1) / 2) :=
by
  sorry

end problem_statement_l39_39071


namespace solve_equation_l39_39997

theorem solve_equation : ∀ (x : ℝ), x ≠ -3 → x ≠ 3 → 
  (x / (x + 3) + 6 / (x^2 - 9) = 1 / (x - 3)) → x = 1 :=
by
  intros x hx1 hx2 h
  sorry

end solve_equation_l39_39997


namespace cut_rectangle_to_square_l39_39842

theorem cut_rectangle_to_square (a b : ℕ) (h₁ : a = 16) (h₂ : b = 9) :
  ∃ (s : ℕ), s * s = a * b ∧ s = 12 :=
by {
  sorry
}

end cut_rectangle_to_square_l39_39842


namespace part1_part2_l39_39602

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def b : ℝ × ℝ := (3, -Real.sqrt 3)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

noncomputable def f (x : ℝ) : ℝ := dot_product (a x) b

theorem part1 (hx : x ∈ Set.Icc 0 Real.pi) (h_perp : dot_product (a x) b = 0) : x = 5 * Real.pi / 6 :=
sorry

theorem part2 (hx : x ∈ Set.Icc 0 Real.pi) :
  (f x ≤ 2 * Real.sqrt 3) ∧ (f x = 2 * Real.sqrt 3 → x = 0) ∧
  (f x ≥ -2 * Real.sqrt 3) ∧ (f x = -2 * Real.sqrt 3 → x = 5 * Real.pi / 6) :=
sorry

end part1_part2_l39_39602


namespace good_apples_count_l39_39365

theorem good_apples_count (total_apples : ℕ) (rotten_percentage : ℝ) (good_apples : ℕ) (h1 : total_apples = 75) (h2 : rotten_percentage = 0.12) :
  good_apples = (1 - rotten_percentage) * total_apples := by
  sorry

end good_apples_count_l39_39365


namespace min_ab_value_l39_39405

theorem min_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (1 / a) + (4 / b) = 1) : ab ≥ 16 :=
by
  sorry

end min_ab_value_l39_39405


namespace sufficient_not_necessary_condition_l39_39403

-- Definitions of propositions p and q
def p (x : ℝ) : Prop := abs (x + 1) ≤ 4
def q (x : ℝ) : Prop := x^2 < 5 * x - 6

-- Definitions of negations of p and q
def not_p (x : ℝ) : Prop := x < -5 ∨ x > 3
def not_q (x : ℝ) : Prop := x ≤ 2 ∨ x ≥ 3

-- The theorem to prove
theorem sufficient_not_necessary_condition (x : ℝ) :
  (¬ p x → ¬ q x) ∧ (¬ q x → ¬ p x → False) := 
by
  sorry

end sufficient_not_necessary_condition_l39_39403


namespace total_apples_picked_l39_39836

-- Definitions based on conditions from part a)
def mike_apples : ℝ := 7.5
def nancy_apples : ℝ := 3.2
def keith_apples : ℝ := 6.1
def olivia_apples : ℝ := 12.4
def thomas_apples : ℝ := 8.6

-- The theorem we need to prove
theorem total_apples_picked : mike_apples + nancy_apples + keith_apples + olivia_apples + thomas_apples = 37.8 := by
    sorry

end total_apples_picked_l39_39836


namespace radius_of_circle_l39_39577

variable {O : Type*} [MetricSpace O]

def distance_near : ℝ := 1
def distance_far : ℝ := 7
def diameter : ℝ := distance_near + distance_far

theorem radius_of_circle (P : O) (r : ℝ) (h1 : distance_near = 1) (h2 : distance_far = 7) :
  r = diameter / 2 :=
by
  -- Proof would go here 
  sorry

end radius_of_circle_l39_39577


namespace sticks_at_20_l39_39679

-- Define the sequence of sticks used at each stage
def sticks (n : ℕ) : ℕ :=
  if n = 1 then 5
  else if n ≤ 10 then 5 + 3 * (n - 1)
  else 32 + 4 * (n - 11)

-- Prove that the number of sticks at the 20th stage is 68
theorem sticks_at_20 : sticks 20 = 68 := by
  sorry

end sticks_at_20_l39_39679


namespace golden_ratio_minus_one_binary_l39_39824

theorem golden_ratio_minus_one_binary (n : ℕ → ℕ) (h_n : ∀ i, 1 ≤ n i)
  (h_incr : ∀ i, n i ≤ n (i + 1)): 
  (∀ k ≥ 4, n k ≤ 2^(k - 1) - 2) := 
by
  sorry

end golden_ratio_minus_one_binary_l39_39824


namespace original_triangle_area_l39_39094

theorem original_triangle_area (A_orig A_new : ℝ) (h1 : A_new = 256) (h2 : A_new = 16 * A_orig) : A_orig = 16 :=
by
  sorry

end original_triangle_area_l39_39094


namespace f_neg_l39_39693

-- Define the function f and its properties
noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then x^2 - 2*x else sorry

-- Define the property of f being an odd function
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- Define the property of f for non-negative x
axiom f_nonneg : ∀ x : ℝ, x ≥ 0 → f x = x^2 - 2*x

-- The theorem to be proven
theorem f_neg : ∀ x : ℝ, x < 0 → f x = -x^2 - 2*x := by
  sorry

end f_neg_l39_39693


namespace bacteria_growth_final_count_l39_39047

theorem bacteria_growth_final_count (initial_count : ℕ) (t : ℕ) 
(h1 : initial_count = 10) 
(h2 : t = 7) 
(h3 : ∀ n : ℕ, (n * 60) = t * 60 → 2 ^ n = 128) : 
(initial_count * 2 ^ t) = 1280 := 
by
  sorry

end bacteria_growth_final_count_l39_39047


namespace gcd_1021_2729_l39_39279

theorem gcd_1021_2729 : Int.gcd 1021 2729 = 1 :=
by
  sorry

end gcd_1021_2729_l39_39279


namespace equilibrium_problems_l39_39123

-- Definition of equilibrium constant and catalyst relations

def q1 := False -- Any concentration of substances in equilibrium constant
def q2 := False -- Catalysts changing equilibrium constant
def q3 := False -- No shift if equilibrium constant doesn't change
def q4 := False -- ΔH > 0 if K decreases with increasing temperature
def q5 := True  -- Stoichiometric differences affecting equilibrium constants
def q6 := True  -- Equilibrium shift not necessarily changing equilibrium constant
def q7 := True  -- Extent of reaction indicated by both equilibrium constant and conversion rate

-- The theorem includes our problem statements

theorem equilibrium_problems :
  q1 = False ∧ q2 = False ∧ q3 = False ∧
  q4 = False ∧ q5 = True ∧ q6 = True ∧ q7 = True := by
  sorry

end equilibrium_problems_l39_39123


namespace intervals_of_monotonicity_l39_39305

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos (x + Real.pi / 3)

theorem intervals_of_monotonicity :
  (∀ k : ℤ, (∀ x, x ∈ Set.Icc (Real.pi / 12 + k * Real.pi) (7 * Real.pi / 12 + k * Real.pi) → (f x ≤ f (7 * Real.pi / 12 + k * Real.pi)))) ∧
  (∀ k : ℤ, (∀ x, x ∈ Set.Icc (-5 * Real.pi / 12 + k * Real.pi) (Real.pi / 12 + k * Real.pi) → (f x ≥ f (Real.pi / 12 + k * Real.pi)))) ∧
  (f (Real.pi / 2) = -Real.sqrt 3) ∧
  (f (Real.pi / 12) = 1 - Real.sqrt 3 / 2) := sorry

end intervals_of_monotonicity_l39_39305


namespace votes_for_winning_candidate_l39_39311

-- Define the variables and conditions
variable (V : ℝ) -- Total number of votes
variable (W : ℝ) -- Votes for the winner

-- Condition 1: The winner received 75% of the votes
axiom winner_votes: W = 0.75 * V

-- Condition 2: The winner won by 500 votes
axiom win_by_500: W - 0.25 * V = 500

-- The statement we want to prove
theorem votes_for_winning_candidate : W = 750 :=
by sorry

end votes_for_winning_candidate_l39_39311


namespace find_p_q_sum_p_plus_q_l39_39533

noncomputable def probability_third_six : ℚ :=
  have fair_die_prob_two_sixes := (1 / 6) * (1 / 6)
  have biased_die_prob_two_sixes := (2 / 3) * (2 / 3)
  have total_prob_two_sixes := (1 / 2) * fair_die_prob_two_sixes + (1 / 2) * biased_die_prob_two_sixes
  have prob_fair_given_two_sixes := fair_die_prob_two_sixes / total_prob_two_sixes
  have prob_biased_given_two_sixes := biased_die_prob_two_sixes / total_prob_two_sixes
  let prob_third_six :=
    prob_fair_given_two_sixes * (1 / 6) +
    prob_biased_given_two_sixes * (2 / 3)
  prob_third_six

theorem find_p_q_sum : 
  probability_third_six = 65 / 102 :=
by sorry

theorem p_plus_q : 
  65 + 102 = 167 :=
by sorry

end find_p_q_sum_p_plus_q_l39_39533


namespace B_value_l39_39187

theorem B_value (A B : Nat) (hA : A < 10) (hB : B < 10) (h_div99 : (100000 * A + 10000 + 1000 * 5 + 100 * B + 90 + 4) % 99 = 0) :
  B = 3 :=
by
  -- skipping the proof
  sorry

end B_value_l39_39187


namespace trigonometric_identity_l39_39880

open Real

theorem trigonometric_identity (θ : ℝ) (h : tan θ = 2) :
  (sin θ * (1 + sin (2 * θ))) / (sqrt 2 * cos (θ - π / 4)) = 6 / 5 :=
by
  sorry

end trigonometric_identity_l39_39880


namespace MikeSalaryNow_l39_39851

-- Definitions based on conditions
def FredSalary  := 1000   -- Fred's salary five months ago
def MikeSalaryFiveMonthsAgo := 10 * FredSalary  -- Mike's salary five months ago
def SalaryIncreasePercent := 40 / 100  -- 40 percent salary increase
def SalaryIncrease := SalaryIncreasePercent * MikeSalaryFiveMonthsAgo  -- Increase in Mike's salary

-- Statement to be proved
theorem MikeSalaryNow : MikeSalaryFiveMonthsAgo + SalaryIncrease = 14000 :=
by
  -- Proof is skipped
  sorry

end MikeSalaryNow_l39_39851


namespace prob_only_one_success_first_firing_is_correct_prob_all_success_after_both_firings_is_correct_l39_39839

noncomputable def prob_first_firing_A : ℚ := 4 / 5
noncomputable def prob_first_firing_B : ℚ := 3 / 4
noncomputable def prob_first_firing_C : ℚ := 2 / 3

noncomputable def prob_second_firing : ℚ := 3 / 5

noncomputable def prob_only_one_success_first_firing :=
  prob_first_firing_A * (1 - prob_first_firing_B) * (1 - prob_first_firing_C) +
  (1 - prob_first_firing_A) * prob_first_firing_B * (1 - prob_first_firing_C) +
  (1 - prob_first_firing_A) * (1 - prob_first_firing_B) * prob_first_firing_C

theorem prob_only_one_success_first_firing_is_correct :
  prob_only_one_success_first_firing = 3 / 20 :=
by sorry

noncomputable def prob_success_after_both_firings_A := prob_first_firing_A * prob_second_firing
noncomputable def prob_success_after_both_firings_B := prob_first_firing_B * prob_second_firing
noncomputable def prob_success_after_both_firings_C := prob_first_firing_C * prob_second_firing

noncomputable def prob_all_success_after_both_firings :=
  prob_success_after_both_firings_A * prob_success_after_both_firings_B * prob_success_after_both_firings_C

theorem prob_all_success_after_both_firings_is_correct :
  prob_all_success_after_both_firings = 54 / 625 :=
by sorry

end prob_only_one_success_first_firing_is_correct_prob_all_success_after_both_firings_is_correct_l39_39839


namespace small_triangle_area_ratio_l39_39315

theorem small_triangle_area_ratio (a b n : ℝ) 
  (h₀ : a > 0) 
  (h₁ : b > 0) 
  (h₂ : n > 0) 
  (h₃ : ∃ (r s : ℝ), r > 0 ∧ s > 0 ∧ (1/2) * a * r = n * a * b ∧ s = (b^2) / (2 * n * b)) :
  (b^2 / (4 * n)) / (a * b) = 1 / (4 * n) :=
by sorry

end small_triangle_area_ratio_l39_39315


namespace power_boat_travel_time_l39_39116

theorem power_boat_travel_time {r p t : ℝ} (h1 : r > 0) (h2 : p > 0) 
  (h3 : (p + r) * t + (p - r) * (9 - t) = 9 * r) : t = 4.5 :=
by
  sorry

end power_boat_travel_time_l39_39116


namespace sum_difference_l39_39143

def arithmetic_series_sum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

def set_A_sum : ℕ :=
  arithmetic_series_sum 42 2 25

def set_B_sum : ℕ :=
  arithmetic_series_sum 62 2 25

theorem sum_difference :
  set_B_sum - set_A_sum = 500 :=
by
  sorry

end sum_difference_l39_39143


namespace elective_course_schemes_l39_39300

theorem elective_course_schemes : Nat.choose 4 2 = 6 := by
  sorry

end elective_course_schemes_l39_39300


namespace exists_n_prime_divides_exp_sum_l39_39524

theorem exists_n_prime_divides_exp_sum (p : ℕ) [Fact (Nat.Prime p)] : 
  ∃ n : ℕ, p ∣ (2^n + 3^n + 6^n - 1) :=
by
  sorry

end exists_n_prime_divides_exp_sum_l39_39524


namespace find_x_when_y_3_l39_39110

variable (y x k : ℝ)

axiom h₁ : x = k / (y ^ 2)
axiom h₂ : y = 9 → x = 0.1111111111111111
axiom y_eq_3 : y = 3

theorem find_x_when_y_3 : y = 3 → x = 1 :=
by
  sorry

end find_x_when_y_3_l39_39110


namespace shampoo_duration_l39_39632

theorem shampoo_duration
  (rose_shampoo : ℚ := 1/3)
  (jasmine_shampoo : ℚ := 1/4)
  (daily_usage : ℚ := 1/12) :
  (rose_shampoo + jasmine_shampoo) / daily_usage = 7 := 
by
  sorry

end shampoo_duration_l39_39632


namespace point_in_second_quadrant_l39_39858

def is_in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

def problem_points : List (ℝ × ℝ) :=
  [(1, -2), (2, 1), (-2, -1), (-1, 2)]

theorem point_in_second_quadrant :
  ∃ (p : ℝ × ℝ), p ∈ problem_points ∧ is_in_second_quadrant p.1 p.2 := by
  use (-1, 2)
  sorry

end point_in_second_quadrant_l39_39858


namespace least_number_subtracted_divisible_l39_39377

theorem least_number_subtracted_divisible (n : ℕ) (d : ℕ) (h : n = 1234567) (k : d = 37) :
  n % d = 13 :=
by 
  rw [h, k]
  sorry

end least_number_subtracted_divisible_l39_39377


namespace doubled_money_is_1_3_l39_39268

-- Define the amounts of money Alice and Bob have
def alice_money := (2 : ℚ) / 5
def bob_money := (1 : ℚ) / 4

-- Define the total money before doubling
def total_money_before_doubling := alice_money + bob_money

-- Define the total money after doubling
def total_money_after_doubling := 2 * total_money_before_doubling

-- State the proposition to prove
theorem doubled_money_is_1_3 : total_money_after_doubling = 1.3 := by
  -- The proof will be filled in here
  sorry

end doubled_money_is_1_3_l39_39268


namespace sequence_geometric_l39_39053

theorem sequence_geometric (a : ℕ → ℝ) (n : ℕ)
  (h1 : a 1 = 1)
  (h_geom : ∀ k : ℕ, a (k + 1) - a k = (1 / 3) ^ k) :
  a n = (3 / 2) * (1 - (1 / 3) ^ n) :=
by
  sorry

end sequence_geometric_l39_39053


namespace three_monotonic_intervals_iff_a_lt_zero_l39_39579

-- Definition of the function f
def f (a x : ℝ) : ℝ := a * x^3 + x

-- Definition of the first derivative of f
def f' (a x : ℝ) : ℝ := 3 * a * x^2 + 1

-- Main statement: Prove that f(x) has exactly three monotonic intervals if and only if a < 0.
theorem three_monotonic_intervals_iff_a_lt_zero (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f' a x1 = 0 ∧ f' a x2 = 0) ↔ a < 0 :=
by
  sorry

end three_monotonic_intervals_iff_a_lt_zero_l39_39579


namespace euro_operation_example_l39_39050

def euro_operation (x y : ℕ) : ℕ := 3 * x * y

theorem euro_operation_example : euro_operation 3 (euro_operation 4 5) = 540 :=
by sorry

end euro_operation_example_l39_39050


namespace isosceles_triangle_perimeter_l39_39326

theorem isosceles_triangle_perimeter (a b : ℝ) (h₁ : 2 * a - 3 * b + 5 = 0) (h₂ : 2 * a + 3 * b - 13 = 0) :
  ∃ p : ℝ, p = 7 ∨ p = 8 :=
sorry

end isosceles_triangle_perimeter_l39_39326


namespace appropriate_speech_length_l39_39912

def speech_length_min := 20
def speech_length_max := 40
def speech_rate := 120

theorem appropriate_speech_length 
  (min_words := speech_length_min * speech_rate) 
  (max_words := speech_length_max * speech_rate) : 
  ∀ n : ℕ, n >= min_words ∧ n <= max_words ↔ (n = 2500 ∨ n = 3800 ∨ n = 4600) := 
by 
  sorry

end appropriate_speech_length_l39_39912


namespace lines_intersect_l39_39563

noncomputable def line1 (t : ℚ) : ℚ × ℚ :=
  (2 + 3 * t, 2 - 4 * t)

noncomputable def line2 (u : ℚ) : ℚ × ℚ :=
  (4 + 5 * u, -6 + 3 * u)

theorem lines_intersect :
  ∃ (t u : ℚ), line1 t = line2 u ∧ line1 t = (160 / 29, -160 / 29) :=
by
  sorry

end lines_intersect_l39_39563


namespace no_valid_n_values_l39_39516

theorem no_valid_n_values :
  ¬ ∃ n : ℕ, (100 ≤ n / 4 ∧ n / 4 ≤ 999) ∧ (100 ≤ 4 * n ∧ 4 * n ≤ 999) :=
by
  sorry

end no_valid_n_values_l39_39516


namespace divides_polynomial_difference_l39_39027

def P (a b c d x : ℤ) : ℤ := a * x^3 + b * x^2 + c * x + d

theorem divides_polynomial_difference (a b c d x y : ℤ) (hxneqy : x ≠ y) :
  (x - y) ∣ (P a b c d x - P a b c d y) :=
by
  sorry

end divides_polynomial_difference_l39_39027


namespace cart_total_distance_l39_39371

-- Definitions for the conditions
def first_section_distance := (15/2) * (8 + (8 + 14 * 10))
def second_section_distance := (15/2) * (148 + (148 + 14 * 6))

-- Combining both distances
def total_distance := first_section_distance + second_section_distance

-- Statement to be proved
theorem cart_total_distance:
  total_distance = 4020 :=
by
  sorry

end cart_total_distance_l39_39371


namespace sum_zero_inv_sum_zero_a_plus_d_zero_l39_39181

theorem sum_zero_inv_sum_zero_a_plus_d_zero 
  (a b c d : ℝ) (h1 : a ≤ b ∧ b ≤ c ∧ c ≤ d) 
  (h2 : a + b + c + d = 0) 
  (h3 : 1/a + 1/b + 1/c + 1/d = 0) :
  a + d = 0 := 
  sorry

end sum_zero_inv_sum_zero_a_plus_d_zero_l39_39181


namespace cone_slant_height_correct_l39_39881

noncomputable def cone_slant_height (r : ℝ) : ℝ := 4 * r

theorem cone_slant_height_correct (r : ℝ) (h₁ : π * r^2 + π * r * cone_slant_height r = 5 * π)
  (h₂ : 2 * π * r = (1/4) * 2 * π * cone_slant_height r) : cone_slant_height r = 4 :=
by
  sorry

end cone_slant_height_correct_l39_39881


namespace clea_ride_down_time_l39_39293

theorem clea_ride_down_time (c s d : ℝ) (h1 : d = 70 * c) (h2 : d = 28 * (c + s)) :
  (d / s) = 47 := by
  sorry

end clea_ride_down_time_l39_39293


namespace sahil_selling_price_l39_39582

noncomputable def sales_tax : ℝ := 0.10 * 18000
noncomputable def initial_cost_with_tax : ℝ := 18000 + sales_tax

noncomputable def broken_part_cost : ℝ := 3000
noncomputable def software_update_cost : ℝ := 4000
noncomputable def total_repair_cost : ℝ := broken_part_cost + software_update_cost
noncomputable def service_tax_on_repair : ℝ := 0.05 * total_repair_cost
noncomputable def total_repair_cost_with_tax : ℝ := total_repair_cost + service_tax_on_repair

noncomputable def transportation_charges : ℝ := 1500
noncomputable def total_cost_before_depreciation : ℝ := initial_cost_with_tax + total_repair_cost_with_tax + transportation_charges

noncomputable def depreciation_first_year : ℝ := 0.15 * total_cost_before_depreciation
noncomputable def value_after_first_year : ℝ := total_cost_before_depreciation - depreciation_first_year

noncomputable def depreciation_second_year : ℝ := 0.15 * value_after_first_year
noncomputable def value_after_second_year : ℝ := value_after_first_year - depreciation_second_year

noncomputable def profit : ℝ := 0.50 * value_after_second_year
noncomputable def selling_price : ℝ := value_after_second_year + profit

theorem sahil_selling_price : selling_price = 31049.44 := by
  sorry

end sahil_selling_price_l39_39582


namespace copper_content_range_l39_39828

theorem copper_content_range (x2 : ℝ) (y : ℝ) (h1 : 0 ≤ x2) (h2 : x2 ≤ 4 / 9) (hy : y = 0.4 + 0.075 * x2) : 
  40 ≤ 100 * y ∧ 100 * y ≤ 130 / 3 :=
by { sorry }

end copper_content_range_l39_39828


namespace expression_of_y_l39_39167

theorem expression_of_y (x y : ℝ) (h : x - y / 2 = 1) : y = 2 * x - 2 :=
sorry

end expression_of_y_l39_39167


namespace price_of_scooter_l39_39323

-- Assume upfront_payment and percentage_upfront are given
def upfront_payment : ℝ := 240
def percentage_upfront : ℝ := 0.20

noncomputable
def total_price (upfront_payment : ℝ) (percentage_upfront : ℝ) : ℝ :=
  (upfront_payment / percentage_upfront)

theorem price_of_scooter : total_price upfront_payment percentage_upfront = 1200 :=
  by
    sorry

end price_of_scooter_l39_39323


namespace fraction_addition_l39_39439

theorem fraction_addition : (1 / 3) + (5 / 12) = 3 / 4 := 
sorry

end fraction_addition_l39_39439


namespace parabola_hyperbola_intersection_l39_39147

open Real

theorem parabola_hyperbola_intersection (p : ℝ) (hp : p > 0)
  (h_hyperbola : ∀ x y, (x^2 / 4 - y^2 = 1) → (y = 2*x ∨ y = -2*x))
  (h_parabola_directrix : ∀ y, (x^2 = 2 * p * y) → (x = -p/2)) 
  (h_area_triangle : (1/2) * (p/2) * (2 * p) = 1) :
  p = sqrt 2 := sorry

end parabola_hyperbola_intersection_l39_39147


namespace fraction_addition_l39_39683

theorem fraction_addition :
  (5 / (8 / 13) + 4 / 7) = (487 / 56) := by
  sorry

end fraction_addition_l39_39683


namespace condition_I_condition_II_l39_39628

noncomputable def f (x a : ℝ) : ℝ := |x - a|

-- Condition (I) proof problem
theorem condition_I (x : ℝ) (a : ℝ) (h : a = 1) :
  f x a ≥ 4 - |x - 1| ↔ (x ≤ -1 ∨ x ≥ 3) :=
by sorry

-- Condition (II) proof problem
theorem condition_II (a : ℝ) (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_f : ∀ x, f x a ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2)
    (h_eq : 1/m + 1/(2*n) = a) : mn ≥ 2 :=
by sorry

end condition_I_condition_II_l39_39628


namespace proof_problem_l39_39352

theorem proof_problem (x : ℕ) (h : 320 / (x + 26) = 4) : x = 54 := 
by 
  sorry

end proof_problem_l39_39352


namespace greatest_y_value_l39_39084

theorem greatest_y_value (x y : ℤ) (h : x * y + 7 * x + 2 * y = -8) : y ≤ -1 :=
by
  sorry

end greatest_y_value_l39_39084


namespace domain_correct_l39_39278

noncomputable def domain_function (x : ℝ) : Prop :=
  (4 * x - 3 > 0) ∧ (Real.log (4 * x - 3) / Real.log 0.5 > 0)

theorem domain_correct : {x : ℝ | domain_function x} = {x : ℝ | (3 / 4 : ℝ) < x ∧ x < 1} :=
by
  sorry

end domain_correct_l39_39278


namespace passes_to_left_l39_39932

theorem passes_to_left (total_passes right_passes center_passes left_passes : ℕ)
  (h_total : total_passes = 50)
  (h_right : right_passes = 2 * left_passes)
  (h_center : center_passes = left_passes + 2)
  (h_sum : left_passes + right_passes + center_passes = total_passes) :
  left_passes = 12 := 
by
  sorry

end passes_to_left_l39_39932


namespace hyperbola_equation_l39_39049

-- Definitions of the conditions
def is_asymptote_1 (y x : ℝ) : Prop :=
  y = 2 * x

def is_asymptote_2 (y x : ℝ) : Prop :=
  y = -2 * x

def passes_through_focus (x y : ℝ) : Prop :=
  x = 1 ∧ y = 0

-- The statement to be proved
theorem hyperbola_equation :
  (∀ x y : ℝ, passes_through_focus x y → x^2 - (y^2 / 4) = 1) :=
sorry

end hyperbola_equation_l39_39049


namespace smaller_square_area_percentage_l39_39296

noncomputable def area_percentage_of_smaller_square :=
  let side_length_large_square : ℝ := 4
  let area_large_square := side_length_large_square ^ 2
  let side_length_smaller_square := side_length_large_square / 5
  let area_smaller_square := side_length_smaller_square ^ 2
  (area_smaller_square / area_large_square) * 100
theorem smaller_square_area_percentage :
  area_percentage_of_smaller_square = 4 := 
sorry

end smaller_square_area_percentage_l39_39296


namespace machinery_spent_correct_l39_39943

def raw_materials : ℝ := 3000
def total_amount : ℝ := 5714.29
def cash (total : ℝ) : ℝ := 0.30 * total
def machinery_spent (total : ℝ) (raw : ℝ) : ℝ := total - raw - cash total

theorem machinery_spent_correct :
  machinery_spent total_amount raw_materials = 1000 := 
  by
    sorry

end machinery_spent_correct_l39_39943


namespace multiply_then_divide_eq_multiply_l39_39367

theorem multiply_then_divide_eq_multiply (x : ℚ) :
  (x * (2 / 5)) / (3 / 7) = x * (14 / 15) :=
by
  sorry

end multiply_then_divide_eq_multiply_l39_39367


namespace chocolate_and_gum_l39_39340

/--
Kolya says that two chocolate bars are more expensive than five gum sticks, 
while Sasha claims that three chocolate bars are more expensive than eight gum sticks. 
When this was checked, only one of them was right. Is it true that seven chocolate bars 
are more expensive than nineteen gum sticks?
-/
theorem chocolate_and_gum (c g : ℝ) (hk : 2 * c > 5 * g) (hs : 3 * c > 8 * g) (only_one_correct : ¬((2 * c > 5 * g) ∧ (3 * c > 8 * g)) ∧ (2 * c > 5 * g ∨ 3 * c > 8 * g)) : 7 * c < 19 * g :=
by
  sorry

end chocolate_and_gum_l39_39340


namespace count_integers_within_range_l39_39083

theorem count_integers_within_range : 
  ∃ (count : ℕ), count = 57 ∧ ∀ n : ℤ, -5.5 * Real.pi ≤ n ∧ n ≤ 12.5 * Real.pi → n ≥ -17 ∧ n ≤ 39 :=
by
  sorry

end count_integers_within_range_l39_39083


namespace surplus_by_end_of_week_is_14_estimated_monthly_income_is_1860_l39_39269

-- Given conditions
def income_per_day : List Int := [65, 68, 50, 66, 50, 75, 74]
def expenditure_per_day : List Int := [-60, -64, -63, -58, -60, -64, -65]

-- Part 1: Proving the surplus by the end of the week is 14 yuan
theorem surplus_by_end_of_week_is_14 :
  List.sum income_per_day + List.sum expenditure_per_day = 14 :=
by
  sorry

-- Part 2: Proving the estimated income needed per month to maintain normal expenses is 1860 yuan
theorem estimated_monthly_income_is_1860 :
  (List.sum (List.map Int.natAbs expenditure_per_day) / 7) * 30 = 1860 :=
by
  sorry

end surplus_by_end_of_week_is_14_estimated_monthly_income_is_1860_l39_39269


namespace solution_for_m_exactly_one_solution_l39_39068

theorem solution_for_m_exactly_one_solution (m : ℚ) : 
  (∀ x : ℚ, (x - 3) / (m * x + 4) = 2 * x → 
            (2 * m * x^2 + 7 * x + 3 = 0)) →
  (49 - 24 * m = 0) → 
  m = 49 / 24 :=
by
  intro h1 h2
  sorry

end solution_for_m_exactly_one_solution_l39_39068


namespace six_digit_number_unique_solution_l39_39559

theorem six_digit_number_unique_solution
    (a b c d e f : ℕ)
    (hN : (N : ℕ) = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f)
    (hM : (M : ℕ) = 100000 * d + 10000 * e + 1000 * f + 100 * a + 10 * b + c)
    (h_eq : 7 * N = 6 * M) :
    N = 461538 :=
by
  sorry

end six_digit_number_unique_solution_l39_39559


namespace probability_both_visible_l39_39888

noncomputable def emma_lap_time : ℕ := 100
noncomputable def ethan_lap_time : ℕ := 75
noncomputable def start_time : ℕ := 0
noncomputable def photo_start_minute : ℕ := 12 * 60 -- converted to seconds
noncomputable def photo_end_minute : ℕ := 13 * 60 -- converted to seconds
noncomputable def photo_visible_angle : ℚ := 1 / 3

theorem probability_both_visible :
  ∀ start_time photo_start_minute photo_end_minute emma_lap_time ethan_lap_time photo_visible_angle,
  start_time = 0 →
  photo_start_minute = 12 * 60 →
  photo_end_minute = 13 * 60 →
  emma_lap_time = 100 →
  ethan_lap_time = 75 →
  photo_visible_angle = 1 / 3 →
  (∃ t, photo_start_minute ≤ t ∧ t < photo_end_minute ∧
        (t % emma_lap_time ≤ (photo_visible_angle * emma_lap_time) / 2 ∨
         t % emma_lap_time ≥ emma_lap_time - (photo_visible_angle * emma_lap_time) / 2) ∧
        (t % ethan_lap_time ≤ (photo_visible_angle * ethan_lap_time) / 2 ∨
         t % ethan_lap_time ≥ ethan_lap_time - (photo_visible_angle * ethan_lap_time) / 2)) ↔
  true :=
sorry

end probability_both_visible_l39_39888


namespace triangle_area_l39_39393

-- Define the given conditions
def perimeter : ℝ := 60
def inradius : ℝ := 2.5

-- Prove the area of the triangle using the given inradius and perimeter
theorem triangle_area (p : ℝ) (r : ℝ) (h1 : p = 60) (h2 : r = 2.5) :
  (r * (p / 2)) = 75 := 
by
  rw [h1, h2]
  sorry

end triangle_area_l39_39393


namespace ratio_unit_price_brand_x_to_brand_y_l39_39503

-- Definitions based on the conditions in the problem
def volume_brand_y (v : ℝ) := v
def price_brand_y (p : ℝ) := p
def volume_brand_x (v : ℝ) := 1.3 * v
def price_brand_x (p : ℝ) := 0.85 * p
noncomputable def unit_price (volume : ℝ) (price : ℝ) := price / volume

-- Theorems to prove the ratio of unit price of Brand X to Brand Y is 17/26
theorem ratio_unit_price_brand_x_to_brand_y (v p : ℝ) (hv : v ≠ 0) (hp : p ≠ 0) : 
  (unit_price (volume_brand_x v) (price_brand_x p)) / (unit_price (volume_brand_y v) (price_brand_y p)) = 17 / 26 := by
  sorry

end ratio_unit_price_brand_x_to_brand_y_l39_39503


namespace total_loaves_served_l39_39431

-- Given conditions
def wheat_bread := 0.5
def white_bread := 0.4

-- Proof that total loaves served is 0.9
theorem total_loaves_served : wheat_bread + white_bread = 0.9 :=
by sorry

end total_loaves_served_l39_39431


namespace speed_of_stream_l39_39398

theorem speed_of_stream
  (V S : ℝ)
  (h1 : 27 = 9 * (V - S))
  (h2 : 81 = 9 * (V + S)) :
  S = 3 :=
by
  sorry

end speed_of_stream_l39_39398


namespace compare_A_B_C_l39_39215

-- Define the expressions A, B, and C
def A : ℚ := (2010 / 2009) + (2010 / 2011)
def B : ℚ := (2010 / 2011) + (2012 / 2011)
def C : ℚ := (2011 / 2010) + (2011 / 2012)

-- The statement asserting A is the greatest
theorem compare_A_B_C : A > B ∧ A > C := by
  sorry

end compare_A_B_C_l39_39215


namespace abc_divisibility_l39_39233

theorem abc_divisibility (a b c : ℕ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) : 
  (a - 1) * (b - 1) * (c - 1) ∣ (a * b * c - 1) ↔ (a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15) :=
by {
  sorry  -- proof to be filled in
}

end abc_divisibility_l39_39233


namespace count_birds_l39_39190

theorem count_birds (b m c : ℕ) (h1 : b + m + c = 300) (h2 : 2 * b + 4 * m + 3 * c = 708) : b = 192 := 
sorry

end count_birds_l39_39190


namespace polynomial_identity_solution_l39_39753

theorem polynomial_identity_solution (P : Polynomial ℝ) :
  (∀ x : ℝ, x * P.eval (x - 1) = (x - 2) * P.eval x) ↔ (∃ a : ℝ, P = Polynomial.C a * (Polynomial.X ^ 2 - Polynomial.X)) :=
by
  sorry

end polynomial_identity_solution_l39_39753


namespace age_of_20th_student_l39_39358

theorem age_of_20th_student (avg_age_20 : ℕ) (avg_age_9 : ℕ) (avg_age_10 : ℕ) :
  (avg_age_20 = 20) →
  (avg_age_9 = 11) →
  (avg_age_10 = 24) →
  let T := 20 * avg_age_20
  let T1 := 9 * avg_age_9
  let T2 := 10 * avg_age_10
  let T19 := T1 + T2
  let age_20th := T - T19
  (age_20th = 61) :=
by
  intros h1 h2 h3
  let T := 20 * avg_age_20
  let T1 := 9 * avg_age_9
  let T2 := 10 * avg_age_10
  let T19 := T1 + T2
  let age_20th := T - T19
  sorry

end age_of_20th_student_l39_39358


namespace right_triangle_third_angle_l39_39519

-- Define the problem
def sum_of_angles_in_triangle (a b c : ℝ) : Prop := a + b + c = 180

-- Define the given angles
def is_right_angle (a : ℝ) : Prop := a = 90
def given_angle (b : ℝ) : Prop := b = 25

-- Define the third angle
def third_angle (a b c : ℝ) : Prop := a + b + c = 180

-- The theorem to prove 
theorem right_triangle_third_angle : ∀ (a b c : ℝ), 
  is_right_angle a → given_angle b → third_angle a b c → c = 65 :=
by
  intros a b c ha hb h_triangle
  sorry

end right_triangle_third_angle_l39_39519


namespace Katy_jellybeans_l39_39319

variable (Matt Matilda Steve Katy : ℕ)

def jellybean_relationship (Matt Matilda Steve Katy : ℕ) : Prop :=
  (Matt = 10 * Steve) ∧
  (Matilda = Matt / 2) ∧
  (Steve = 84) ∧
  (Katy = 3 * Matilda) ∧
  (Katy = Matt / 2)

theorem Katy_jellybeans : ∃ Katy, jellybean_relationship Matt Matilda Steve Katy ∧ Katy = 1260 := by
  sorry

end Katy_jellybeans_l39_39319


namespace group_weight_problem_l39_39008

theorem group_weight_problem (n : ℕ) (avg_weight_increase : ℕ) (weight_diff : ℕ) (total_weight_increase : ℕ) 
  (h1 : avg_weight_increase = 3) (h2 : weight_diff = 75 - 45) (h3 : total_weight_increase = avg_weight_increase * n)
  (h4 : total_weight_increase = weight_diff) : n = 10 := by
  sorry

end group_weight_problem_l39_39008


namespace ratio_of_sums_l39_39711

theorem ratio_of_sums (p q r u v w : ℝ) 
  (h1 : p > 0) (h2 : q > 0) (h3 : r > 0) (h4 : u > 0) (h5 : v > 0) (h6 : w > 0)
  (h7 : p^2 + q^2 + r^2 = 49) (h8 : u^2 + v^2 + w^2 = 64)
  (h9 : p * u + q * v + r * w = 56) : 
  (p + q + r) / (u + v + w) = 7 / 8 :=
by
  sorry

end ratio_of_sums_l39_39711


namespace smallest_coprime_to_210_l39_39728

theorem smallest_coprime_to_210 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 210 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Nat.gcd y 210 = 1 → y ≥ x :=
by
  sorry

end smallest_coprime_to_210_l39_39728


namespace Frank_days_to_finish_book_l39_39807

theorem Frank_days_to_finish_book (pages_per_day : ℕ) (total_pages : ℕ) (h1 : pages_per_day = 22) (h2 : total_pages = 12518) : total_pages / pages_per_day = 569 := by
  sorry

end Frank_days_to_finish_book_l39_39807


namespace faster_train_speed_l39_39584

theorem faster_train_speed
  (slower_train_speed : ℝ := 60) -- speed of the slower train in km/h
  (length_train1 : ℝ := 1.10) -- length of the slower train in km
  (length_train2 : ℝ := 0.9) -- length of the faster train in km
  (cross_time_sec : ℝ := 47.99999999999999) -- crossing time in seconds
  (cross_time : ℝ := cross_time_sec / 3600) -- crossing time in hours
  (total_distance : ℝ := length_train1 + length_train2) -- total distance covered
  (relative_speed : ℝ := total_distance / cross_time) -- relative speed
  (faster_train_speed : ℝ := relative_speed - slower_train_speed) -- speed of the faster train
  : faster_train_speed = 90 :=
by
  sorry

end faster_train_speed_l39_39584


namespace optimal_sampling_methods_l39_39230

/-
We define the conditions of the problem.
-/
def households := 500
def high_income_households := 125
def middle_income_households := 280
def low_income_households := 95
def sample_households := 100

def soccer_players := 12
def sample_soccer_players := 3

/-
We state the goal as a theorem.
-/
theorem optimal_sampling_methods :
  (sample_households == 100) ∧
  (sample_soccer_players == 3) ∧
  (high_income_households + middle_income_households + low_income_households == households) →
  ("stratified" = "stratified" ∧ "random" = "random") :=
by
  -- Sorry to skip the proof
  sorry

end optimal_sampling_methods_l39_39230


namespace min_value_problem_l39_39930

theorem min_value_problem (a b c d : ℝ) (h : a + 2*b + 3*c + 4*d = 12) : 
    a^2 + b^2 + c^2 + d^2 >= 24 / 5 := 
by
  sorry

end min_value_problem_l39_39930


namespace math_problem_l39_39430

variable {x y z : ℝ}
variable (hx : x > 0) (hy : y > 0) (hz : z > 0)
variable (h : x^2 + y^2 + z^2 = 1)

theorem math_problem : 
  (x / (1 - x^2)) + (y / (1 - y^2)) + (z / (1 - z^2)) ≥ 3 * Real.sqrt 3 / 2 :=
sorry

end math_problem_l39_39430


namespace not_a_fraction_l39_39734

axiom x : ℝ
axiom a : ℝ
axiom b : ℝ

noncomputable def A := 1 / (x^2)
noncomputable def B := (b + 3) / a
noncomputable def C := (x^2 - 1) / (x + 1)
noncomputable def D := (2 / 7) * a

theorem not_a_fraction : ¬ (D = A) ∧ ¬ (D = B) ∧ ¬ (D = C) :=
by 
  sorry

end not_a_fraction_l39_39734


namespace chickens_cheaper_than_eggs_l39_39415

-- Define the initial costs of the chickens
def initial_cost_chicken1 : ℝ := 25
def initial_cost_chicken2 : ℝ := 30
def initial_cost_chicken3 : ℝ := 22
def initial_cost_chicken4 : ℝ := 35

-- Define the weekly feed costs for the chickens
def weekly_feed_cost_chicken1 : ℝ := 1.50
def weekly_feed_cost_chicken2 : ℝ := 1.30
def weekly_feed_cost_chicken3 : ℝ := 1.10
def weekly_feed_cost_chicken4 : ℝ := 0.90

-- Define the weekly egg production for the chickens
def weekly_egg_prod_chicken1 : ℝ := 4
def weekly_egg_prod_chicken2 : ℝ := 3
def weekly_egg_prod_chicken3 : ℝ := 5
def weekly_egg_prod_chicken4 : ℝ := 2

-- Define the cost of a dozen eggs at the store
def cost_per_dozen_eggs : ℝ := 2

-- Define total initial costs, total weekly feed cost, and weekly savings
def total_initial_cost : ℝ := initial_cost_chicken1 + initial_cost_chicken2 + initial_cost_chicken3 + initial_cost_chicken4
def total_weekly_feed_cost : ℝ := weekly_feed_cost_chicken1 + weekly_feed_cost_chicken2 + weekly_feed_cost_chicken3 + weekly_feed_cost_chicken4
def weekly_savings : ℝ := cost_per_dozen_eggs

-- Define the condition for the number of weeks (W) when the chickens become cheaper
def breakeven_weeks : ℝ := 40

theorem chickens_cheaper_than_eggs (W : ℕ) :
  total_initial_cost + W * total_weekly_feed_cost = W * weekly_savings :=
sorry

end chickens_cheaper_than_eggs_l39_39415


namespace compare_neg_fractions_and_neg_values_l39_39223

theorem compare_neg_fractions_and_neg_values :
  (- (3 : ℚ) / 4 > - (4 : ℚ) / 5) ∧ (-(-3 : ℤ) > -|(3 : ℤ)|) :=
by
  apply And.intro
  sorry
  sorry

end compare_neg_fractions_and_neg_values_l39_39223


namespace range_of_a_l39_39492

noncomputable def quadratic_inequality_solution_set (a : ℝ) : Prop :=
∀ x : ℝ, a * x^2 + a * x - 4 < 0

theorem range_of_a :
  {a : ℝ | quadratic_inequality_solution_set a} = {a | -16 < a ∧ a ≤ 0} := 
sorry

end range_of_a_l39_39492


namespace M_intersect_P_l39_39765

noncomputable def M : Set ℝ := { y | ∃ x : ℝ, y = x^2 + 1 }
noncomputable def P : Set ℝ := { y | ∃ x : ℝ, y = Real.log x }

theorem M_intersect_P :
  M ∩ P = { y | y ≥ 1 } :=
sorry

end M_intersect_P_l39_39765


namespace find_all_functions_l39_39433

theorem find_all_functions 
  (f : ℤ → ℝ)
  (h1 : ∀ m n : ℤ, m < n → f m < f n)
  (h2 : ∀ m n : ℤ, ∃ k : ℤ, f m - f n = f k) :
  ∃ a t : ℝ, a > 0 ∧ (∀ n : ℤ, f n = a * (n + t)) :=
sorry

end find_all_functions_l39_39433


namespace joan_games_attended_l39_39327
-- Mathematical definitions based on the provided conditions

def total_games_played : ℕ := 864
def games_missed_by_Joan : ℕ := 469

-- Theorem statement
theorem joan_games_attended : total_games_played - games_missed_by_Joan = 395 :=
by
  -- Proof omitted
  sorry

end joan_games_attended_l39_39327


namespace total_cost_of_books_l39_39528

theorem total_cost_of_books (C1 C2 : ℝ) 
  (hC1 : C1 = 268.33)
  (h_selling_prices_equal : 0.85 * C1 = 1.19 * C2) :
  C1 + C2 = 459.15 :=
by
  -- placeholder for the proof
  sorry

end total_cost_of_books_l39_39528


namespace total_balloons_l39_39647

-- Define the number of balloons Alyssa, Sandy, and Sally have.
def alyssa_balloons : ℕ := 37
def sandy_balloons : ℕ := 28
def sally_balloons : ℕ := 39

-- Theorem stating that the total number of balloons is 104.
theorem total_balloons : alyssa_balloons + sandy_balloons + sally_balloons = 104 :=
by
  -- Proof is omitted for the purpose of this task.
  sorry

end total_balloons_l39_39647


namespace hybrids_with_full_headlights_l39_39249

-- Definitions for the conditions
def total_cars : ℕ := 600
def percentage_hybrids : ℕ := 60
def percentage_one_headlight : ℕ := 40

-- The proof statement
theorem hybrids_with_full_headlights :
  (percentage_hybrids * total_cars) / 100 - (percentage_one_headlight * (percentage_hybrids * total_cars) / 100) / 100 = 216 :=
by
  sorry

end hybrids_with_full_headlights_l39_39249


namespace find_fourth_number_l39_39201

variables (A B C D E F : ℝ)

theorem find_fourth_number
  (h1 : A + B + C + D + E + F = 180)
  (h2 : A + B + C + D = 100)
  (h3 : D + E + F = 105) :
  D = 25 :=
by
  sorry

end find_fourth_number_l39_39201


namespace find_ratio_of_sums_l39_39425

variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n, S n = n * (a 1 + a n) / 2

def ratio_condition (a : ℕ → ℝ) :=
  a 6 / a 5 = 9 / 11

theorem find_ratio_of_sums (seq : ∃ d, arithmetic_sequence a d)
    (sum_prop : sum_first_n_terms S a)
    (ratio_prop : ratio_condition a) :
  S 11 / S 9 = 1 :=
sorry

end find_ratio_of_sums_l39_39425


namespace sqrt_of_9_is_3_l39_39376

theorem sqrt_of_9_is_3 {x : ℝ} (h₁ : x * x = 9) (h₂ : x ≥ 0) : x = 3 := sorry

end sqrt_of_9_is_3_l39_39376


namespace cos_double_angle_sub_pi_six_l39_39979

variable (α : ℝ)
variable (h1 : 0 < α ∧ α < π / 3)
variable (h2 : Real.sin (α + π / 6) = 2 * Real.sqrt 5 / 5)

theorem cos_double_angle_sub_pi_six :
  Real.cos (2 * α - π / 6) = 4 / 5 :=
by
  sorry

end cos_double_angle_sub_pi_six_l39_39979


namespace betty_oranges_l39_39394

theorem betty_oranges (boxes: ℕ) (oranges_per_box: ℕ) (h1: boxes = 3) (h2: oranges_per_box = 8) : boxes * oranges_per_box = 24 :=
by
  -- proof omitted
  sorry

end betty_oranges_l39_39394


namespace convert_base_10_to_base_8_l39_39605

theorem convert_base_10_to_base_8 (n : ℕ) (n_eq : n = 3275) : 
  n = 3275 → ∃ (a b c d : ℕ), (a * 8^3 + b * 8^2 + c * 8^1 + d * 8^0 = 6323) :=
by 
  sorry

end convert_base_10_to_base_8_l39_39605


namespace solve_derivative_equation_l39_39451

theorem solve_derivative_equation :
  (∃ n : ℤ, ∀ x,
    x = 2 * n * Real.pi ∨
    x = 2 * n * Real.pi - 2 * Real.arctan (3 / 5)) :=
by
  sorry

end solve_derivative_equation_l39_39451


namespace min_deliveries_to_cover_cost_l39_39552

theorem min_deliveries_to_cover_cost (cost_per_van earnings_per_delivery gasoline_cost_per_delivery : ℕ) (h1 : cost_per_van = 4500) (h2 : earnings_per_delivery = 15 ) (h3 : gasoline_cost_per_delivery = 5) : 
  ∃ d : ℕ, 10 * d ≥ cost_per_van ∧ ∀ x : ℕ, x < d → 10 * x < cost_per_van :=
by
  use 450
  sorry

end min_deliveries_to_cover_cost_l39_39552


namespace solve_equation_l39_39795

theorem solve_equation (x : ℝ) (h : x ≠ -2) : (x^2 + x + 1) / (x + 2) = x + 1 → x = -1 / 2 := 
by
  intro h1
  sorry

end solve_equation_l39_39795


namespace required_percentage_to_pass_l39_39113

-- Definitions based on conditions
def obtained_marks : ℕ := 175
def failed_by : ℕ := 56
def max_marks : ℕ := 700
def pass_marks : ℕ := obtained_marks + failed_by

-- Theorem stating the required percentage to pass
theorem required_percentage_to_pass : 
  (pass_marks : ℚ) / max_marks * 100 = 33 := 
by 
  sorry

end required_percentage_to_pass_l39_39113


namespace remainder_of_789987_div_8_l39_39055

theorem remainder_of_789987_div_8 : (789987 % 8) = 3 := by
  sorry

end remainder_of_789987_div_8_l39_39055


namespace true_value_of_product_l39_39872

theorem true_value_of_product (a b c : ℕ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  let product := (100 * a + 10 * b + c) * (100 * b + 10 * c + a) * (100 * c + 10 * a + b)
  product = 2342355286 → (product % 10 = 6) → product = 328245326 :=
by
  sorry

end true_value_of_product_l39_39872


namespace maximum_f_l39_39074

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def f (p : ℝ) : ℝ :=
  binomial_coefficient 20 2 * p^2 * (1 - p)^18

theorem maximum_f :
  ∃ p_0 : ℝ, 0 < p_0 ∧ p_0 < 1 ∧ f p = f (0.1) := sorry

end maximum_f_l39_39074


namespace prob_exactly_M_laws_in_concept_expected_laws_in_concept_l39_39513

section Anchuria
variables (K N M : ℕ) (p : ℝ)

-- Part (a): Define P_M as the binomial probability distribution result
def probability_exactly_M_laws : ℝ :=
  (Nat.choose K M : ℝ) * (1 - (1 - p) ^ N) ^ M * ((1 - p) ^ N) ^ (K - M)

-- Part (a): Prove the result for the probability that exactly M laws are included in the Concept
theorem prob_exactly_M_laws_in_concept :
  probability_exactly_M_laws K N M p =
  (Nat.choose K M : ℝ) * (1 - (1 - p) ^ N) ^ M * ((1 - p) ^ N) ^ (K - M) := 
sorry

-- Part (b): Define the expected number of laws included in the Concept
def expected_number_of_laws : ℝ :=
  K * (1 - (1 - p) ^ N)

-- Part (b): Prove the result for the expected number of laws included in the Concept
theorem expected_laws_in_concept :
  expected_number_of_laws K N p = K * (1 - (1 - p) ^ N) :=
sorry
end Anchuria

end prob_exactly_M_laws_in_concept_expected_laws_in_concept_l39_39513


namespace proof_problem_l39_39653

noncomputable def problem_statement (a b : ℝ) : Prop :=
  (∀ x, (a * x^2 + b * x + 2 > 0) ↔ (x ∈ Set.Ioo (-1/2 : ℝ) (1/3 : ℝ))) 

theorem proof_problem (a b : ℝ) (h : problem_statement a b) : a + b = -14 :=
sorry

end proof_problem_l39_39653


namespace river_flow_volume_l39_39196

theorem river_flow_volume (depth width : ℝ) (flow_rate_kmph : ℝ) :
  depth = 3 → width = 36 → flow_rate_kmph = 2 → 
  (depth * width) * (flow_rate_kmph * 1000 / 60) = 3599.64 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end river_flow_volume_l39_39196


namespace resized_height_l39_39145

-- Define original dimensions
def original_width : ℝ := 4.5
def original_height : ℝ := 3

-- Define new width
def new_width : ℝ := 13.5

-- Define new height to be proven
def new_height : ℝ := 9

-- Theorem statement
theorem resized_height :
  (new_width / original_width) * original_height = new_height :=
by
  -- The statement that equates the new height calculated proportionately to 9
  sorry

end resized_height_l39_39145


namespace Ron_eats_24_pickle_slices_l39_39011

theorem Ron_eats_24_pickle_slices : 
  ∀ (pickle_slices_Sammy Tammy Ron : ℕ), 
    pickle_slices_Sammy = 15 → 
    Tammy = 2 * pickle_slices_Sammy → 
    Ron = Tammy - (20 * Tammy / 100) → 
    Ron = 24 := by
  intros pickle_slices_Sammy Tammy Ron h_sammy h_tammy h_ron
  sorry

end Ron_eats_24_pickle_slices_l39_39011


namespace share_difference_l39_39809

theorem share_difference 
  (S : ℝ) -- Total sum of money
  (A B C D : ℝ) -- Shares of a, b, c, d respectively
  (h_proportion : A = 5 / 14 * S)
  (h_proportion : B = 2 / 14 * S)
  (h_proportion : C = 4 / 14 * S)
  (h_proportion : D = 3 / 14 * S)
  (h_d_share : D = 1500) :
  C - D = 500 :=
sorry

end share_difference_l39_39809


namespace acute_angle_sum_l39_39227

theorem acute_angle_sum (n : ℕ) (hn : n ≥ 4) (M m: ℕ) 
  (hM : M = 3) (hm : m = 0) : M + m = 3 := 
by 
  sorry

end acute_angle_sum_l39_39227


namespace price_of_70_cans_l39_39501

noncomputable def discounted_price (regular_price : ℝ) (discount_percent : ℝ) : ℝ :=
  regular_price * (1 - discount_percent / 100)

noncomputable def total_price (regular_price : ℝ) (discount_percent : ℝ) (total_cans : ℕ) (cans_per_case : ℕ) : ℝ :=
  let price_per_can := discounted_price regular_price discount_percent
  let full_cases := total_cans / cans_per_case
  let remaining_cans := total_cans % cans_per_case
  full_cases * cans_per_case * price_per_can + remaining_cans * price_per_can

theorem price_of_70_cans :
  total_price 0.55 25 70 24 = 28.875 :=
by
  sorry

end price_of_70_cans_l39_39501


namespace ratio_of_walkway_to_fountain_l39_39114

theorem ratio_of_walkway_to_fountain (n s d : ℝ) (h₀ : n = 10) (h₁ : n^2 * s^2 = 0.40 * (n*s + 2*n*d)^2) : 
  d / s = 1 / 3.44 := 
sorry

end ratio_of_walkway_to_fountain_l39_39114


namespace intersection_setA_setB_l39_39623

noncomputable def setA : Set ℝ := { x : ℝ | abs (x - 1) < 2 }
noncomputable def setB : Set ℝ := { x : ℝ | (x - 2) / (x + 4) < 0 }

theorem intersection_setA_setB : 
  (setA ∩ setB) = { x : ℝ | -1 < x ∧ x < 2 } :=
by
  sorry

end intersection_setA_setB_l39_39623


namespace right_triangle_hypotenuse_length_l39_39916

theorem right_triangle_hypotenuse_length (a b c : ℕ) (h₁ : a = 5) (h₂ : b = 12)
  (h₃ : c^2 = a^2 + b^2) : c = 13 :=
by
  -- We should provide the actual proof here, but we'll use sorry for now.
  sorry

end right_triangle_hypotenuse_length_l39_39916


namespace total_kids_played_l39_39928

def kids_played_week (monday tuesday wednesday thursday: ℕ): ℕ :=
  let friday := thursday + (thursday * 20 / 100)
  let saturday := friday - (friday * 30 / 100)
  let sunday := 2 * monday
  monday + tuesday + wednesday + thursday + friday + saturday + sunday

theorem total_kids_played : 
  kids_played_week 15 18 25 30 = 180 :=
by
  sorry

end total_kids_played_l39_39928


namespace sparrows_on_fence_l39_39446

-- Define the number of sparrows initially on the fence
def initial_sparrows : ℕ := 2

-- Define the number of sparrows that joined later
def additional_sparrows : ℕ := 4

-- Define the number of sparrows that flew away
def sparrows_flew_away : ℕ := 3

-- Define the final number of sparrows on the fence
def final_sparrows : ℕ := initial_sparrows + additional_sparrows - sparrows_flew_away

-- Prove that the final number of sparrows on the fence is 3
theorem sparrows_on_fence : final_sparrows = 3 := by
  sorry

end sparrows_on_fence_l39_39446


namespace barbara_total_cost_l39_39382

-- Definitions based on the given conditions
def steak_cost_per_pound : ℝ := 15.00
def steak_quantity : ℝ := 4.5
def chicken_cost_per_pound : ℝ := 8.00
def chicken_quantity : ℝ := 1.5

def expected_total_cost : ℝ := 42.00

-- The main proposition we need to prove
theorem barbara_total_cost :
  steak_cost_per_pound * steak_quantity + chicken_cost_per_pound * chicken_quantity = expected_total_cost :=
by
  sorry

end barbara_total_cost_l39_39382


namespace problem_statement_l39_39592

theorem problem_statement (a b : ℤ) (h : |a + 5| + (b - 2) ^ 2 = 0) : (a + b) ^ 2010 = 3 ^ 2010 :=
by
  sorry

end problem_statement_l39_39592


namespace find_value_of_fraction_l39_39556

theorem find_value_of_fraction (a b c d: ℕ) (h1: a = 4 * b) (h2: b = 3 * c) (h3: c = 5 * d) : 
  (a * c) / (b * d) = 20 :=
by
  sorry

end find_value_of_fraction_l39_39556


namespace kittens_count_l39_39675

def initial_kittens : ℕ := 8
def additional_kittens : ℕ := 2
def total_kittens : ℕ := 10

theorem kittens_count : initial_kittens + additional_kittens = total_kittens := by
  -- Proof will go here
  sorry

end kittens_count_l39_39675


namespace eq_proof_l39_39186

noncomputable def S_even : ℚ := 28
noncomputable def S_odd : ℚ := 24

theorem eq_proof : ( (S_even / S_odd - S_odd / S_even) * 2 ) = (13 / 21) :=
by
  sorry

end eq_proof_l39_39186


namespace probability_of_8_or_9_ring_l39_39558

theorem probability_of_8_or_9_ring (p10 p9 p8 : ℝ) (h1 : p10 = 0.3) (h2 : p9 = 0.3) (h3 : p8 = 0.2) :
  p9 + p8 = 0.5 :=
by
  sorry

end probability_of_8_or_9_ring_l39_39558


namespace ratio_of_amounts_l39_39966

theorem ratio_of_amounts
    (initial_cents : ℕ)
    (given_to_peter_cents : ℕ)
    (remaining_nickels : ℕ)
    (nickel_value : ℕ := 5)
    (nickels_initial := initial_cents / nickel_value)
    (nickels_to_peter := given_to_peter_cents / nickel_value)
    (nickels_remaining := nickels_initial - nickels_to_peter)
    (nickels_given_to_randi := nickels_remaining - remaining_nickels)
    (cents_to_randi := nickels_given_to_randi * nickel_value)
    (cents_initial : initial_cents = 95)
    (cents_peter : given_to_peter_cents = 25)
    (nickels_left : remaining_nickels = 4)
    :
    (cents_to_randi / given_to_peter_cents) = 2 :=
by
  sorry

end ratio_of_amounts_l39_39966


namespace harmonic_mean_closest_to_one_l39_39082

-- Define the given conditions a = 1/4 and b = 2048
def a : ℚ := 1 / 4
def b : ℚ := 2048

-- Define the harmonic mean of two numbers
def harmonic_mean (x y : ℚ) : ℚ := 2 * x * y / (x + y)

-- State the theorem proving the harmonic mean is closest to 1
theorem harmonic_mean_closest_to_one : abs (harmonic_mean a b - 1) < 1 :=
sorry

end harmonic_mean_closest_to_one_l39_39082


namespace min_packs_needed_l39_39062

theorem min_packs_needed (P8 P15 P30 : ℕ) (h: P8 * 8 + P15 * 15 + P30 * 30 = 120) : P8 + P15 + P30 = 4 :=
by
  sorry

end min_packs_needed_l39_39062


namespace inequality_solution_l39_39127

theorem inequality_solution (x : ℝ) :
  (x + 2) / (x^2 + 3 * x + 10) ≥ 0 ↔ x ≥ -2 := sorry

end inequality_solution_l39_39127


namespace sum_series_eq_three_l39_39380

theorem sum_series_eq_three : 
  ∑' (k : ℕ), (k^2 : ℝ) / (2^k : ℝ) = 3 := sorry

end sum_series_eq_three_l39_39380


namespace price_of_each_bottle_is_3_l39_39686

/-- Each bottle of iced coffee has 6 servings. -/
def servings_per_bottle : ℕ := 6

/-- Tricia drinks half a container (bottle) a day. -/
def daily_consumption_rate : ℕ := servings_per_bottle / 2

/-- Number of days in 2 weeks. -/
def duration_days : ℕ := 14

/-- Number of servings Tricia consumes in 2 weeks. -/
def total_servings : ℕ := daily_consumption_rate * duration_days

/-- Number of bottles needed to get the total servings. -/
def bottles_needed : ℕ := total_servings / servings_per_bottle

/-- The total cost of the bottles is $21. -/
def total_cost : ℕ := 21

/-- The price per bottle is the total cost divided by the number of bottles. -/
def price_per_bottle : ℕ := total_cost / bottles_needed

/-- The price of each bottle is $3. -/
theorem price_of_each_bottle_is_3 : price_per_bottle = 3 :=
by
  -- We assume the necessary steps and mathematical verifications have been done.
  sorry

end price_of_each_bottle_is_3_l39_39686


namespace problem1_simplification_problem2_simplification_l39_39812

theorem problem1_simplification : (3 / Real.sqrt 3 - (Real.sqrt 3) ^ 2 - Real.sqrt 27 + (abs (Real.sqrt 3 - 2))) = -1 - 3 * Real.sqrt 3 :=
  by
    sorry

theorem problem2_simplification (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 2) :
  ((x + 2) / (x ^ 2 - 2 * x) - (x - 1) / (x ^ 2 - 4 * x + 4)) / ((x - 4) / x) = 1 / (x - 2) ^ 2 :=
  by
    sorry

end problem1_simplification_problem2_simplification_l39_39812


namespace four_digit_number_divisibility_l39_39448

theorem four_digit_number_divisibility : ∃ x : ℕ, 
  (let n := 1000 + x * 100 + 50 + x; 
   ∃ k₁ k₂ : ℤ, (n = 36 * k₁) ∧ ((10 * 5 + x) = 4 * k₂) ∧ ((2 * x + 6) % 9 = 0)) :=
sorry

end four_digit_number_divisibility_l39_39448


namespace number_of_pages_in_chunk_l39_39510

-- Conditions
def first_page : Nat := 213
def last_page : Nat := 312

-- Define the property we need to prove
theorem number_of_pages_in_chunk : last_page - first_page + 1 = 100 := by
  -- skipping the proof
  sorry

end number_of_pages_in_chunk_l39_39510


namespace son_age_l39_39077

theorem son_age (S M : ℕ) (h1 : M = S + 30) (h2 : M + 2 = 2 * (S + 2)) : S = 28 := 
by
  -- The proof can be filled in here.
  sorry

end son_age_l39_39077


namespace olivia_initial_money_l39_39337

theorem olivia_initial_money (spent_supermarket : ℕ) (spent_showroom : ℕ) (left_money : ℕ) (initial_money : ℕ) :
  spent_supermarket = 31 → spent_showroom = 49 → left_money = 26 → initial_money = spent_supermarket + spent_showroom + left_money → initial_money = 106 :=
by
  intros h_supermarket h_showroom h_left h_initial 
  rw [h_supermarket, h_showroom, h_left] at h_initial
  exact h_initial

end olivia_initial_money_l39_39337


namespace square_root_of_16_is_pm_4_l39_39846

theorem square_root_of_16_is_pm_4 : { x : ℝ | x^2 = 16 } = {4, -4} :=
sorry

end square_root_of_16_is_pm_4_l39_39846


namespace a4_minus_b4_l39_39668

theorem a4_minus_b4 (a b : ℝ) (h1 : a - b = 1) (h2 : a^2 - b^2 = -1) : a^4 - b^4 = -1 := by
  sorry

end a4_minus_b4_l39_39668


namespace xiao_ming_final_score_l39_39576

theorem xiao_ming_final_score :
  let speech_image := 9
  let content := 8
  let effectiveness := 8
  let weight_speech_image := 0.3
  let weight_content := 0.4
  let weight_effectiveness := 0.3
  (speech_image * weight_speech_image +
   content * weight_content +
   effectiveness * weight_effectiveness) = 8.3 :=
by
  let speech_image := 9
  let content := 8
  let effectiveness := 8
  let weight_speech_image := 0.3
  let weight_content := 0.4
  let weight_effectiveness := 0.3
  sorry

end xiao_ming_final_score_l39_39576


namespace find_positive_integers_l39_39286

theorem find_positive_integers (x y : ℕ) (hx : x > 0) (hy : y > 0) : x^2 - Nat.factorial y = 2019 ↔ x = 45 ∧ y = 3 :=
by
  sorry

end find_positive_integers_l39_39286


namespace number_neither_9_nice_nor_10_nice_500_l39_39174

def is_k_nice (N k : ℕ) : Prop := ∃ a : ℕ, a > 0 ∧ (∃ m : ℕ, N = (k * m) + 1)

def count_k_nice (N k : ℕ) : ℕ :=
  (N - 1) / k + 1

def count_neither_9_nice_nor_10_nice (N : ℕ) : ℕ :=
  let count_9_nice := count_k_nice N 9
  let count_10_nice := count_k_nice N 10
  let lcm_9_10 := 90  -- lcm of 9 and 10
  let count_both := count_k_nice N lcm_9_10
  N - (count_9_nice + count_10_nice - count_both)

theorem number_neither_9_nice_nor_10_nice_500 : count_neither_9_nice_nor_10_nice 500 = 400 :=
  sorry

end number_neither_9_nice_nor_10_nice_500_l39_39174


namespace shirt_price_after_discount_l39_39803

/-- Given a shirt with an initial cost price of $20 and a profit margin of 30%, 
    and a sale discount of 50%, prove that the final sale price of the shirt is $13. -/
theorem shirt_price_after_discount
  (cost_price : ℝ)
  (profit_margin : ℝ)
  (discount : ℝ)
  (selling_price : ℝ)
  (final_price : ℝ)
  (h_cost : cost_price = 20)
  (h_profit_margin : profit_margin = 0.30)
  (h_discount : discount = 0.50)
  (h_selling_price : selling_price = cost_price + profit_margin * cost_price)
  (h_final_price : final_price = selling_price - discount * selling_price) :
  final_price = 13 := 
  sorry

end shirt_price_after_discount_l39_39803


namespace total_days_2001_2005_l39_39829

theorem total_days_2001_2005 : 
  let is_leap_year (y : ℕ) := y % 4 = 0 ∧ (y % 100 ≠ 0 ∨ y % 400 = 0)
  let days_in_year (y : ℕ) := if is_leap_year y then 366 else 365 
  (days_in_year 2001) + (days_in_year 2002) + (days_in_year 2003) + (days_in_year 2004) + (days_in_year 2005) = 1461 :=
by
  sorry

end total_days_2001_2005_l39_39829


namespace sandy_balloons_l39_39342

def balloons_problem (A S T : ℕ) : ℕ :=
  T - (A + S)

theorem sandy_balloons : balloons_problem 37 39 104 = 28 := by
  sorry

end sandy_balloons_l39_39342


namespace find_constants_l39_39554

-- Define constants and the problem
variables (C D Q : Type) [AddCommGroup Q] [Module ℝ Q]
variables (CQ QD : ℝ) (h_ratio : CQ = 3 * QD / 5)

-- Define the conjecture we want to prove
theorem find_constants (t u : ℝ) (h_t : t = 5 / (3 + 5)) (h_u : u = 3 / (3 + 5)) :
  (CQ = 3 * QD / 5) → 
  (t * CQ + u * QD = (5 / 8) * CQ + (3 / 8) * QD) :=
sorry

end find_constants_l39_39554


namespace unique_solution_for_a_l39_39484

def system_has_unique_solution (a : ℝ) (x y : ℝ) : Prop :=
(x^2 + y^2 + 2 * x ≤ 1) ∧ (x - y + a = 0)

theorem unique_solution_for_a (a x y : ℝ) :
  (system_has_unique_solution 3 x y ∨ system_has_unique_solution (-1) x y)
  ∧ (((a = 3) → (x, y) = (-2, 1)) ∨ ((a = -1) → (x, y) = (0, -1))) :=
sorry

end unique_solution_for_a_l39_39484


namespace sale_price_of_sarees_after_discounts_l39_39464

theorem sale_price_of_sarees_after_discounts :
  let original_price := 400.0
  let discount_1 := 0.15
  let discount_2 := 0.08
  let discount_3 := 0.07
  let discount_4 := 0.10
  let price_after_first_discount := original_price * (1 - discount_1)
  let price_after_second_discount := price_after_first_discount * (1 - discount_2)
  let price_after_third_discount := price_after_second_discount * (1 - discount_3)
  let final_price := price_after_third_discount * (1 - discount_4)
  final_price = 261.81 := by
    -- Sorry is used to skip the proof
    sorry

end sale_price_of_sarees_after_discounts_l39_39464


namespace elena_novel_pages_l39_39777

theorem elena_novel_pages
  (days_vacation : ℕ)
  (pages_first_two_days : ℕ)
  (pages_next_three_days : ℕ)
  (pages_last_day : ℕ)
  (h1 : days_vacation = 6)
  (h2 : pages_first_two_days = 2 * 42)
  (h3 : pages_next_three_days = 3 * 35)
  (h4 : pages_last_day = 15) :
  pages_first_two_days + pages_next_three_days + pages_last_day = 204 := by
  sorry

end elena_novel_pages_l39_39777


namespace angle_B_lt_90_l39_39159

theorem angle_B_lt_90 {a b c : ℝ} (h_arith : b = (a + c) / 2) (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) : 
  ∃ (A B C : ℝ), B < 90 :=
sorry

end angle_B_lt_90_l39_39159


namespace complex_power_of_sum_l39_39130

theorem complex_power_of_sum (i : ℂ) (hi : i^2 = -1) : (1 + i)^2 = 2 * i :=
by
  sorry

end complex_power_of_sum_l39_39130


namespace oil_depth_solution_l39_39895

theorem oil_depth_solution
  (length diameter surface_area : ℝ)
  (h : ℝ)
  (h_length : length = 12)
  (h_diameter : diameter = 4)
  (h_surface_area : surface_area = 24)
  (r : ℝ := diameter / 2)
  (c : ℝ := surface_area / length) :
  (h = 2 - Real.sqrt 3 ∨ h = 2 + Real.sqrt 3) :=
by
  sorry

end oil_depth_solution_l39_39895


namespace parallel_sufficient_not_necessary_l39_39449

def line := Type
def parallel (l1 l2 : line) : Prop := sorry
def in_plane (l : line) : Prop := sorry

theorem parallel_sufficient_not_necessary (a β : line) :
  (parallel a β → ∃ γ, in_plane γ ∧ parallel a γ) ∧
  ¬( (∃ γ, in_plane γ ∧ parallel a γ) → parallel a β ) :=
by sorry

end parallel_sufficient_not_necessary_l39_39449


namespace cuboid_volume_l39_39370

theorem cuboid_volume (a b c : ℝ) (h1 : a * b = 3) (h2 : b * c = 5) (h3 : a * c = 15) : a * b * c = 15 :=
sorry

end cuboid_volume_l39_39370


namespace no_integer_solutions_quadratic_l39_39234

theorem no_integer_solutions_quadratic (n : ℤ) (s : ℕ) (pos_odd_s : s % 2 = 1) :
  ¬ ∃ x : ℤ, x^2 - 16 * n * x + 7^s = 0 :=
sorry

end no_integer_solutions_quadratic_l39_39234


namespace arithmetic_square_root_16_l39_39025

theorem arithmetic_square_root_16 : ∀ x : ℝ, x ≥ 0 → x^2 = 16 → x = 4 :=
by
  intro x hx h
  sorry

end arithmetic_square_root_16_l39_39025


namespace shadow_taller_pot_length_l39_39616

-- Definitions based on the conditions a)
def height_shorter_pot : ℕ := 20
def shadow_shorter_pot : ℕ := 10
def height_taller_pot : ℕ := 40

-- The proof problem
theorem shadow_taller_pot_length : 
  ∃ (S2 : ℕ), (height_shorter_pot / shadow_shorter_pot = height_taller_pot / S2) ∧ S2 = 20 :=
sorry

end shadow_taller_pot_length_l39_39616


namespace rental_property_key_count_l39_39785

def number_of_keys (complexes apartments_per_complex keys_per_lock locks_per_apartment : ℕ) : ℕ :=
  complexes * apartments_per_complex * keys_per_lock * locks_per_apartment

theorem rental_property_key_count : 
  number_of_keys 2 12 3 1 = 72 := by
  sorry

end rental_property_key_count_l39_39785


namespace shaded_region_area_l39_39229

section

-- Define points and shapes
structure point := (x : ℝ) (y : ℝ)
def square_side_length : ℝ := 40
def square_area : ℝ := square_side_length * square_side_length

-- Points defining the square and triangles within it
def point_O : point := ⟨0, 0⟩
def point_A : point := ⟨15, 0⟩
def point_B : point := ⟨40, 25⟩
def point_C : point := ⟨40, 40⟩
def point_D1 : point := ⟨25, 40⟩
def point_E : point := ⟨0, 15⟩

-- Function to calculate the area of a triangle given base and height
def triangle_area (base height : ℝ) : ℝ := 0.5 * base * height

-- Areas of individual triangles
def triangle1_area : ℝ := triangle_area 15 15
def triangle2_area : ℝ := triangle_area 25 25
def triangle3_area : ℝ := triangle_area 15 15

-- Total area of the triangles
def total_triangles_area : ℝ := triangle1_area + triangle2_area + triangle3_area

-- Shaded area calculation
def shaded_area : ℝ := square_area - total_triangles_area

-- Statement of the theorem to be proven
theorem shaded_region_area : shaded_area = 1062.5 := by sorry

end

end shaded_region_area_l39_39229


namespace spherical_to_rectangular_l39_39306

theorem spherical_to_rectangular
  (ρ θ φ : ℝ)
  (ρ_eq : ρ = 10)
  (θ_eq : θ = 5 * Real.pi / 4)
  (φ_eq : φ = Real.pi / 4) :
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z) = (-5, -5, 5 * Real.sqrt 2) :=
by
  sorry

end spherical_to_rectangular_l39_39306


namespace tangent_line_parabola_l39_39282

theorem tangent_line_parabola (k : ℝ) (tangent : ∀ y : ℝ, ∃ x : ℝ, 4 * x + 3 * y + k = 0 ∧ y^2 = 12 * x) : 
  k = 27 / 4 :=
sorry

end tangent_line_parabola_l39_39282


namespace sum_of_two_numbers_l39_39781

theorem sum_of_two_numbers (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h1 : x * y = 12) (h2 : 1 / x = 3 * (1 / y)) : x + y = 8 :=
by
  sorry

end sum_of_two_numbers_l39_39781


namespace total_baseball_cards_l39_39010

/-- 
Given that you have 5 friends and each friend gets 91 baseball cards, 
prove that the total number of baseball cards you have is 455.
-/
def baseball_cards (f c : Nat) (t : Nat) : Prop :=
  (t = f * c)

theorem total_baseball_cards:
  ∀ (f c t : Nat), f = 5 → c = 91 → t = 455 → baseball_cards f c t :=
by
  intros f c t hf hc ht
  sorry

end total_baseball_cards_l39_39010


namespace find_positives_xyz_l39_39133

theorem find_positives_xyz (x y z : ℕ) (h : x > 0 ∧ y > 0 ∧ z > 0)
    (heq : (1 : ℚ)/x + (1 : ℚ)/y + (1 : ℚ)/z = 4 / 5) :
    (x = 2 ∧ y = 4 ∧ z = 20) ∨ (x = 2 ∧ y = 5 ∧ z = 10) :=
by
  sorry

-- This theorem states that there are only two sets of positive integers (x, y, z)
-- that satisfy the equation (1/x) + (1/y) + (1/z) = 4/5, specifically:
-- (2, 4, 20) and (2, 5, 10).

end find_positives_xyz_l39_39133


namespace problem_l39_39188

universe u

def U : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {x ∈ U | x ≠ 0} -- Placeholder, B itself is a generic subset of U
def A : Set ℕ := {x ∈ U | x = 3 ∨ x = 5 ∨ x = 9}

noncomputable def C_U (B : Set ℕ) : Set ℕ := {x ∈ U | ¬ (x ∈ B)}

axiom h1 : A ∩ B = {3, 5}
axiom h2 : A ∩ C_U B = {9}

theorem problem : A = {3, 5, 9} :=
by
  sorry

end problem_l39_39188


namespace kids_from_lawrence_county_go_to_camp_l39_39941

theorem kids_from_lawrence_county_go_to_camp : 
  (1201565 - 590796 = 610769) := 
by
  sorry

end kids_from_lawrence_county_go_to_camp_l39_39941


namespace fraction_shaded_area_l39_39043

theorem fraction_shaded_area (l w : ℕ) (h_l : l = 15) (h_w : w = 20)
  (h_qtr : (1 / 4: ℝ) * (l * w) = 75) (h_shaded : (1 / 5: ℝ) * 75 = 15) :
  (15 / (l * w): ℝ) = 1 / 20 :=
by
  sorry

end fraction_shaded_area_l39_39043


namespace find_x_l39_39873

theorem find_x (x y z : ℝ) 
  (h1 : x + y + z = 150)
  (h2 : x + 10 = y - 10)
  (h3 : x + 10 = 3 * z) :
  x = 380 / 7 := 
  sorry

end find_x_l39_39873


namespace equation_is_true_l39_39209

theorem equation_is_true :
  10 * 6 - (9 - 3) * 2 = 48 :=
by
  sorry

end equation_is_true_l39_39209


namespace proposition_holds_n_2019_l39_39101

theorem proposition_holds_n_2019 (P: ℕ → Prop) 
  (H1: ∀ k : ℕ, k > 0 → ¬ P (k + 1) → ¬ P k) 
  (H2: P 2018) : 
  P 2019 :=
by 
  sorry

end proposition_holds_n_2019_l39_39101


namespace negation_of_exists_l39_39597

theorem negation_of_exists (x : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - x + 1 > 0) :=
by
  sorry

end negation_of_exists_l39_39597


namespace leah_total_coin_value_l39_39751

variable (p n : ℕ) -- Let p be the number of pennies and n be the number of nickels

-- Leah has 15 coins consisting of pennies and nickels
axiom coin_count : p + n = 15

-- If she had three more nickels, she would have twice as many pennies as nickels
axiom conditional_equation : p = 2 * (n + 3)

-- We want to prove that the total value of Leah's coins in cents is 27
theorem leah_total_coin_value : 5 * n + p = 27 := by
  sorry

end leah_total_coin_value_l39_39751


namespace num_positive_divisors_36_l39_39075

theorem num_positive_divisors_36 :
  let n := 36
  let d := (2 + 1) * (2 + 1)
  d = 9 :=
by
  sorry

end num_positive_divisors_36_l39_39075


namespace inequality_proof_l39_39656

theorem inequality_proof
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h : a * b + b * c + c * a = 1) :
  (1 / (a^2 + 1)) + (1 / (b^2 + 1)) + (1 / (c^2 + 1)) ≤ 9 / 4 :=
by
  sorry

end inequality_proof_l39_39656


namespace largest_stickers_per_page_l39_39191

theorem largest_stickers_per_page :
  Nat.gcd (Nat.gcd 1050 1260) 945 = 105 := 
sorry

end largest_stickers_per_page_l39_39191


namespace conditional_without_else_l39_39823

def if_then_else_statement (s: String) : Prop :=
  (s = "IF—THEN" ∨ s = "IF—THEN—ELSE")

theorem conditional_without_else : if_then_else_statement "IF—THEN" :=
  sorry

end conditional_without_else_l39_39823


namespace fish_cost_l39_39988

theorem fish_cost (F P : ℝ) (h1 : 4 * F + 2 * P = 530) (h2 : 7 * F + 3 * P = 875) : F = 80 := 
by
  sorry

end fish_cost_l39_39988


namespace jerry_pool_time_l39_39732

variables (J : ℕ) -- Denote the time Jerry was in the pool

-- Conditions
def Elaine_time := 2 * J -- Elaine stayed in the pool for twice as long as Jerry
def George_time := (2 / 3) * J -- George could only stay in the pool for one-third as long as Elaine
def Kramer_time := 0 -- Kramer did not find the pool

-- Combined total time
def total_time : ℕ := J + Elaine_time J + George_time J + Kramer_time

-- Theorem stating that J = 3 given the combined total time of 11 minutes
theorem jerry_pool_time (h : total_time J = 11) : J = 3 :=
by
  sorry

end jerry_pool_time_l39_39732


namespace not_divisible_1998_minus_1_by_1000_minus_1_l39_39141

theorem not_divisible_1998_minus_1_by_1000_minus_1 (m : ℕ) : ¬ (1000^m - 1 ∣ 1998^m - 1) :=
sorry

end not_divisible_1998_minus_1_by_1000_minus_1_l39_39141


namespace set_points_quadrants_l39_39857

theorem set_points_quadrants (x y : ℝ) :
  (y > 3 * x) ∧ (y > 5 - 2 * x) → 
  (y > 0 ∧ x > 0) ∨ (y > 0 ∧ x < 0) :=
by 
  sorry

end set_points_quadrants_l39_39857


namespace law_firm_associates_l39_39515

def percentage (total: ℕ) (part: ℕ): ℕ := part * 100 / total

theorem law_firm_associates (total: ℕ) (second_year: ℕ) (first_year: ℕ) (more_than_two_years: ℕ):
  percentage total more_than_two_years = 50 →
  percentage total second_year = 25 →
  first_year = more_than_two_years - second_year →
  percentage total first_year = 25 →
  percentage total (total - first_year) = 75 :=
by
  intros h1 h2 h3 h4
  sorry

end law_firm_associates_l39_39515


namespace size_relationship_l39_39762

noncomputable def a : ℝ := 1 + Real.sqrt 7
noncomputable def b : ℝ := Real.sqrt 3 + Real.sqrt 5
noncomputable def c : ℝ := 4

theorem size_relationship : a < b ∧ b < c := by
  sorry

end size_relationship_l39_39762


namespace divides_343_l39_39699

theorem divides_343 
  (x y z : ℕ) 
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (h : 7 ∣ (x + 6 * y) * (2 * x + 5 * y) * (3 * x + 4 * y)) :
  343 ∣ (x + 6 * y) * (2 * x + 5 * y) * (3 * x + 4 * y) :=
by sorry

end divides_343_l39_39699


namespace triangle_perimeter_l39_39929

def ellipse (x y : ℝ) := x^2 / 4 + y^2 / 2 = 1

def foci_distance (c : ℝ) := c = Real.sqrt 2

theorem triangle_perimeter {x y : ℝ} (A : ellipse x y) (F1 F2 : ℝ)
  (hF1 : F1 = -Real.sqrt 2) (hF2 : F2 = Real.sqrt 2) :
  |(x - F1)| + |(x - F2)| = 4 + 2 * Real.sqrt 2 :=
sorry

end triangle_perimeter_l39_39929


namespace smallest_y_of_arithmetic_sequence_l39_39735

theorem smallest_y_of_arithmetic_sequence
  (x y z d : ℝ)
  (h_arith_series_x : x = y - d)
  (h_arith_series_z : z = y + d)
  (h_positive_x : x > 0)
  (h_positive_y : y > 0)
  (h_positive_z : z > 0)
  (h_product : x * y * z = 216) : y = 6 :=
sorry

end smallest_y_of_arithmetic_sequence_l39_39735
