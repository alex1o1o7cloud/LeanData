import Mathlib

namespace NUMINAMATH_GPT_find_x_plus_y_l1594_159415

theorem find_x_plus_y (x y : ℝ) (h1 : |x| - x + y = 13) (h2 : x - |y| + y = 7) : x + y = 20 := 
by
  sorry

end NUMINAMATH_GPT_find_x_plus_y_l1594_159415


namespace NUMINAMATH_GPT_sandy_books_from_second_shop_l1594_159498

noncomputable def books_from_second_shop (books_first: ℕ) (cost_first: ℕ) (cost_second: ℕ) (avg_price: ℕ): ℕ :=
  let total_cost := cost_first + cost_second
  let total_books := books_first + (total_cost / avg_price) - books_first
  total_cost / avg_price - books_first

theorem sandy_books_from_second_shop :
  books_from_second_shop 65 1380 900 19 = 55 :=
by
  sorry

end NUMINAMATH_GPT_sandy_books_from_second_shop_l1594_159498


namespace NUMINAMATH_GPT_solve_complex_addition_l1594_159458

noncomputable def complex_addition : Prop :=
  let i := Complex.I
  let z1 := 3 - 5 * i
  let z2 := -1 + 12 * i
  let result := 2 + 7 * i
  z1 + z2 = result

theorem solve_complex_addition :
  complex_addition :=
by
  sorry

end NUMINAMATH_GPT_solve_complex_addition_l1594_159458


namespace NUMINAMATH_GPT_ratio_of_areas_l1594_159441

-- Definitions based on the conditions given
def square_side_length : ℕ := 48
def rectangle_width : ℕ := 56
def rectangle_height : ℕ := 63

-- Areas derived from the definitions
def square_area := square_side_length * square_side_length
def rectangle_area := rectangle_width * rectangle_height

-- Lean statement to prove the ratio of areas
theorem ratio_of_areas :
  (square_area : ℚ) / rectangle_area = 2 / 3 := 
sorry

end NUMINAMATH_GPT_ratio_of_areas_l1594_159441


namespace NUMINAMATH_GPT_num_2_edge_paths_l1594_159483

-- Let T be a tetrahedron with vertices connected such that each vertex has exactly 3 edges.
-- Prove that the number of distinct 2-edge paths from a starting vertex P to an ending vertex Q is 3.

def tetrahedron : Type := ℕ -- This is a simplified representation of vertices

noncomputable def edges (a b : tetrahedron) : Prop := true -- Each pair of distinct vertices is an edge in a tetrahedron

theorem num_2_edge_paths (P Q : tetrahedron) (hP : P ≠ Q) : 
  -- There are 3 distinct 2-edge paths from P to Q  
  ∃ (paths : Finset (tetrahedron × tetrahedron)), 
    paths.card = 3 ∧ 
    ∀ (p : tetrahedron × tetrahedron), p ∈ paths → 
      edges P p.1 ∧ edges p.1 p.2 ∧ p.2 = Q :=
by 
  sorry

end NUMINAMATH_GPT_num_2_edge_paths_l1594_159483


namespace NUMINAMATH_GPT_find_a_l1594_159430

theorem find_a (x y z a : ℝ) (h1 : 2 * x^2 + 3 * y^2 + 6 * z^2 = a) (h2 : a > 0) (h3 : ∀ x y z : ℝ, 2 * x^2 + 3 * y^2 + 6 * z^2 = a → (x + y + z) ≤ 1) :
  a = 1 := 
sorry

end NUMINAMATH_GPT_find_a_l1594_159430


namespace NUMINAMATH_GPT_dante_eggs_l1594_159456

theorem dante_eggs (E F : ℝ) (h1 : F = E / 2) (h2 : F + E = 90) : E = 60 :=
by
  sorry

end NUMINAMATH_GPT_dante_eggs_l1594_159456


namespace NUMINAMATH_GPT_PropA_impl_PropB_not_PropB_impl_PropA_l1594_159444

variable {x : ℝ}

def PropA (x : ℝ) : Prop := abs (x - 1) < 5
def PropB (x : ℝ) : Prop := abs (abs x - 1) < 5

theorem PropA_impl_PropB : PropA x → PropB x :=
by sorry

theorem not_PropB_impl_PropA : ¬(PropB x → PropA x) :=
by sorry

end NUMINAMATH_GPT_PropA_impl_PropB_not_PropB_impl_PropA_l1594_159444


namespace NUMINAMATH_GPT_evaluate_expression_l1594_159462

theorem evaluate_expression :
  let a := 17
  let b := 19
  let c := 23
  let numerator1 := 136 * (1 / b - 1 / c) + 361 * (1 / c - 1 / a) + 529 * (1 / a - 1 / b)
  let denominator := a * (1 / b - 1 / c) + b * (1 / c - 1 / a) + c * (1 / a - 1 / b)
  let numerator2 := 144 * (1 / b - 1 / c) + 400 * (1 / c - 1 / a) + 576 * (1 / a - 1 / b)
  (numerator1 / denominator) * (numerator2 / denominator) = 3481 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1594_159462


namespace NUMINAMATH_GPT_train_pass_platform_time_l1594_159400

theorem train_pass_platform_time (l v t : ℝ) (h1 : v = l / t) (h2 : l > 0) (h3 : t > 0) :
  ∃ T : ℝ, T = 3.5 * t := by
  sorry

end NUMINAMATH_GPT_train_pass_platform_time_l1594_159400


namespace NUMINAMATH_GPT_f_five_l1594_159416

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = - f x
axiom f_one : f 1 = 1 / 2
axiom functional_equation : ∀ x : ℝ, f (x + 2) = f x + f 2

theorem f_five : f 5 = 5 / 2 :=
by sorry

end NUMINAMATH_GPT_f_five_l1594_159416


namespace NUMINAMATH_GPT_even_function_a_value_l1594_159414

theorem even_function_a_value (a : ℝ) (f : ℝ → ℝ) 
  (h_def : ∀ x, f x = (x + 1) * (x - a))
  (h_even : ∀ x, f x = f (-x)) : a = -1 :=
by
  sorry

end NUMINAMATH_GPT_even_function_a_value_l1594_159414


namespace NUMINAMATH_GPT_D_double_prime_coordinates_l1594_159490

-- The coordinates of points A, B, C, D as given in the problem
def A : (ℝ × ℝ) := (3, 6)
def B : (ℝ × ℝ) := (5, 10)
def C : (ℝ × ℝ) := (7, 6)
def D : (ℝ × ℝ) := (5, 2)

-- Reflection across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
def D' : ℝ × ℝ := reflect_x D

-- Translate the point (x, y) by (dx, dy)
def translate (p : ℝ × ℝ) (dx dy : ℝ) : ℝ × ℝ := (p.1 + dx, p.2 + dy)

-- Reflect across the line y = x
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Combined translation and reflection across y = x + 2
def reflect_y_eq_x_plus_2 (p : ℝ × ℝ) : ℝ × ℝ :=
  let p_translated := translate p 0 (-2)
  let p_reflected := reflect_y_eq_x p_translated
  translate p_reflected 0 2

def D'' : ℝ × ℝ := reflect_y_eq_x_plus_2 D'

theorem D_double_prime_coordinates : D'' = (-4, 7) := by
  sorry

end NUMINAMATH_GPT_D_double_prime_coordinates_l1594_159490


namespace NUMINAMATH_GPT_magnitude_of_a_plus_b_l1594_159421

open Real

noncomputable def magnitude (x y : ℝ) : ℝ :=
  sqrt (x^2 + y^2)

theorem magnitude_of_a_plus_b (m : ℝ) (a b : ℝ × ℝ)
  (h₁ : a = (m+2, 1))
  (h₂ : b = (1, -2*m))
  (h₃ : (a.1 * b.1 + a.2 * b.2 = 0)) :
  magnitude (a.1 + b.1) (a.2 + b.2) = sqrt 34 :=
by
  sorry

end NUMINAMATH_GPT_magnitude_of_a_plus_b_l1594_159421


namespace NUMINAMATH_GPT_initial_bananas_proof_l1594_159423

noncomputable def initial_bananas_per_child (total_children : ℕ) (absent_children : ℕ) (extra_bananas : ℕ) : ℕ :=
  (extra_bananas * (total_children - absent_children)) / (total_children - extra_bananas)

theorem initial_bananas_proof
  (total_children : ℕ)
  (absent_children : ℕ)
  (extra_bananas : ℕ)
  (h_total : total_children = 640)
  (h_absent : absent_children = 320)
  (h_extra : extra_bananas = 2) : initial_bananas_per_child total_children absent_children extra_bananas = 2 :=
by
  sorry

end NUMINAMATH_GPT_initial_bananas_proof_l1594_159423


namespace NUMINAMATH_GPT_max_value_of_m_l1594_159437

theorem max_value_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - 8 > 0 → x < m) → m = -2 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_m_l1594_159437


namespace NUMINAMATH_GPT_circle_area_sum_l1594_159448

theorem circle_area_sum (x y z : ℕ) (A₁ A₂ A₃ total_area : ℕ) (h₁ : A₁ = 6) (h₂ : A₂ = 15) 
  (h₃ : A₃ = 83) (h₄ : total_area = 220) (hx : x = 4) (hy : y = 2) (hz : z = 2) :
  A₁ * x + A₂ * y + A₃ * z = total_area := by
  sorry

end NUMINAMATH_GPT_circle_area_sum_l1594_159448


namespace NUMINAMATH_GPT_horner_evaluation_l1594_159408

-- Define the polynomial function
def f (x : ℝ) : ℝ := 4 * x^4 + 3 * x^3 - 6 * x^2 + x - 1

-- The theorem that we need to prove
theorem horner_evaluation : f (-1) = -5 :=
  by
  -- This is the statement without the proof steps
  sorry

end NUMINAMATH_GPT_horner_evaluation_l1594_159408


namespace NUMINAMATH_GPT_local_tax_deduction_in_cents_l1594_159404

def aliciaHourlyWageInDollars : ℝ := 25
def taxDeductionRate : ℝ := 0.02
def aliciaHourlyWageInCents := aliciaHourlyWageInDollars * 100

theorem local_tax_deduction_in_cents :
  taxDeductionRate * aliciaHourlyWageInCents = 50 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_local_tax_deduction_in_cents_l1594_159404


namespace NUMINAMATH_GPT_quadratic_csq_l1594_159431

theorem quadratic_csq (x q t : ℝ) (h : 9 * x^2 - 36 * x - 81 = 0) (hq : q = -2) (ht : t = 13) :
  q + t = 11 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_csq_l1594_159431


namespace NUMINAMATH_GPT_determine_k_completed_square_l1594_159472

theorem determine_k_completed_square (x : ℝ) :
  ∃ (a h k : ℝ), a * (x - h)^2 + k = x^2 - 7 * x ∧ k = -49/4 := sorry

end NUMINAMATH_GPT_determine_k_completed_square_l1594_159472


namespace NUMINAMATH_GPT_cube_faces_one_third_blue_l1594_159419

theorem cube_faces_one_third_blue (n : ℕ) (h1 : ∃ n, n > 0 ∧ (6 * n^2) / (6 * n^3) = 1 / 3) : n = 3 := by
  sorry

end NUMINAMATH_GPT_cube_faces_one_third_blue_l1594_159419


namespace NUMINAMATH_GPT_ball_highest_point_at_l1594_159442

noncomputable def h (a b t : ℝ) : ℝ := a * t^2 + b * t

theorem ball_highest_point_at (a b : ℝ) :
  (h a b 3 = h a b 7) →
  t = 4.9 :=
by
  sorry

end NUMINAMATH_GPT_ball_highest_point_at_l1594_159442


namespace NUMINAMATH_GPT_compare_neg_fractions_l1594_159426

theorem compare_neg_fractions :
  - (10 / 11 : ℤ) > - (11 / 12 : ℤ) :=
sorry

end NUMINAMATH_GPT_compare_neg_fractions_l1594_159426


namespace NUMINAMATH_GPT_household_waste_per_day_l1594_159420

theorem household_waste_per_day (total_waste_4_weeks : ℝ) (h : total_waste_4_weeks = 30.8) : 
  (total_waste_4_weeks / 4 / 7) = 1.1 :=
by
  sorry

end NUMINAMATH_GPT_household_waste_per_day_l1594_159420


namespace NUMINAMATH_GPT_selling_price_l1594_159491

theorem selling_price (cost_price profit_percentage : ℝ) (h_cost : cost_price = 250) (h_profit : profit_percentage = 0.60) :
  cost_price + profit_percentage * cost_price = 400 := sorry

end NUMINAMATH_GPT_selling_price_l1594_159491


namespace NUMINAMATH_GPT_find_x_value_l1594_159435

theorem find_x_value (x y : ℝ) (hx : x ≠ 0) (h1 : x / 3 = y^3) (h2 : x / 6 = 3 * y) : x = 18 * Real.sqrt 6 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_value_l1594_159435


namespace NUMINAMATH_GPT_find_subtracted_value_l1594_159485

theorem find_subtracted_value (N : ℤ) (V : ℤ) (h1 : N = 740) (h2 : N / 4 - V = 10) : V = 175 :=
by
  sorry

end NUMINAMATH_GPT_find_subtracted_value_l1594_159485


namespace NUMINAMATH_GPT_solve_for_x_l1594_159478

theorem solve_for_x (x : ℝ) (h : x + 3 * x = 500 - (4 * x + 5 * x)) : x = 500 / 13 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l1594_159478


namespace NUMINAMATH_GPT_britney_has_more_chickens_l1594_159440

theorem britney_has_more_chickens :
  let susie_rhode_island_reds := 11
  let susie_golden_comets := 6
  let britney_rhode_island_reds := 2 * susie_rhode_island_reds
  let britney_golden_comets := susie_golden_comets / 2
  let susie_total := susie_rhode_island_reds + susie_golden_comets
  let britney_total := britney_rhode_island_reds + britney_golden_comets
  britney_total - susie_total = 8 := by
    sorry

end NUMINAMATH_GPT_britney_has_more_chickens_l1594_159440


namespace NUMINAMATH_GPT_mrs_jackson_decorations_l1594_159418

theorem mrs_jackson_decorations (boxes decorations_in_each_box decorations_used : Nat) 
  (h1 : boxes = 4) 
  (h2 : decorations_in_each_box = 15) 
  (h3 : decorations_used = 35) :
  boxes * decorations_in_each_box - decorations_used = 25 := 
  by
  sorry

end NUMINAMATH_GPT_mrs_jackson_decorations_l1594_159418


namespace NUMINAMATH_GPT_problem_l1594_159496

variable (p q : Prop)

theorem problem (h₁ : ¬ p) (h₂ : ¬ (p ∧ q)) : ¬ (p ∨ q) := sorry

end NUMINAMATH_GPT_problem_l1594_159496


namespace NUMINAMATH_GPT_count_with_consecutive_ones_l1594_159463

noncomputable def countValidIntegers : ℕ := 512
noncomputable def invalidCount : ℕ := 89

theorem count_with_consecutive_ones :
  countValidIntegers - invalidCount = 423 :=
by
  sorry

end NUMINAMATH_GPT_count_with_consecutive_ones_l1594_159463


namespace NUMINAMATH_GPT_vecMA_dotProduct_vecBA_range_l1594_159493

-- Define the conditions
def pointM : ℝ × ℝ := (1, 0)

def onEllipse (p : ℝ × ℝ) : Prop := (p.1^2 / 4 + p.2^2 = 1)

def vecMA (A : ℝ × ℝ) := (A.1 - pointM.1, A.2 - pointM.2)
def vecMB (B : ℝ × ℝ) := (B.1 - pointM.1, B.2 - pointM.2)
def vecBA (A B : ℝ × ℝ) := (A.1 - B.1, A.2 - B.2)

def dotProduct (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the statement
theorem vecMA_dotProduct_vecBA_range (A B : ℝ × ℝ) (α : ℝ) :
  onEllipse A → onEllipse B → dotProduct (vecMA A) (vecMB B) = 0 → 
  A = (2 * Real.cos α, Real.sin α) → 
  (2/3 ≤ dotProduct (vecMA A) (vecBA A B) ∧ dotProduct (vecMA A) (vecBA A B) ≤ 9) :=
sorry

end NUMINAMATH_GPT_vecMA_dotProduct_vecBA_range_l1594_159493


namespace NUMINAMATH_GPT_allen_blocks_l1594_159476

def blocks_per_color : Nat := 7
def colors_used : Nat := 7

theorem allen_blocks : (blocks_per_color * colors_used) = 49 :=
by
  sorry

end NUMINAMATH_GPT_allen_blocks_l1594_159476


namespace NUMINAMATH_GPT_tan_2theta_sin_cos_fraction_l1594_159433

variable {θ : ℝ} (h : (2 - Real.tan θ) / (1 + Real.tan θ) = 1)

-- Part (I)
theorem tan_2theta (h : (2 - Real.tan θ) / (1 + Real.tan θ) = 1) : Real.tan (2 * θ) = 4 / 3 :=
by sorry

-- Part (II)
theorem sin_cos_fraction (h : (2 - Real.tan θ) / (1 + Real.tan θ) = 1) : 
  (Real.sin θ + Real.cos θ) / (Real.cos θ - 3 * Real.sin θ) = -3 :=
by sorry

end NUMINAMATH_GPT_tan_2theta_sin_cos_fraction_l1594_159433


namespace NUMINAMATH_GPT_tan_80_l1594_159468

theorem tan_80 (m : ℝ) (h : Real.cos (100 * Real.pi / 180) = m) :
    Real.tan (80 * Real.pi / 180) = Real.sqrt (1 - m^2) / -m :=
by
  sorry

end NUMINAMATH_GPT_tan_80_l1594_159468


namespace NUMINAMATH_GPT_investment_period_more_than_tripling_l1594_159480

theorem investment_period_more_than_tripling (r : ℝ) (multiple : ℝ) (n : ℕ) 
  (h_r: r = 0.341) (h_multiple: multiple > 3) :
  (1 + r)^n ≥ multiple → n = 4 :=
by
  sorry

end NUMINAMATH_GPT_investment_period_more_than_tripling_l1594_159480


namespace NUMINAMATH_GPT_ratio_of_boxes_sold_l1594_159405

-- Definitions for conditions
variables (T W Tu : ℕ)

-- Define the conditions as hypotheses
def conditions : Prop :=
  W = 2 * T ∧
  Tu = 2 * W ∧
  T = 1200

-- The statement to prove the ratio Tu / W = 2
theorem ratio_of_boxes_sold (T W Tu : ℕ) (h : conditions T W Tu) :
  Tu / W = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_boxes_sold_l1594_159405


namespace NUMINAMATH_GPT_find_ratio_l1594_159492

variable (a b : ℕ → ℕ)
variable (S T : ℕ → ℕ)

-- Given conditions
axiom sum_arithmetic_a (n : ℕ) : S n = n / 2 * (a 1 + a n)
axiom sum_arithmetic_b (n : ℕ) : T n = n / 2 * (b 1 + b n)
axiom sum_ratios (n : ℕ) : S n / T n = (2 * n + 1) / (3 * n + 2)

-- The proof problem
theorem find_ratio : (a 3 + a 11 + a 19) / (b 7 + b 15) = 129 / 130 := 
sorry

end NUMINAMATH_GPT_find_ratio_l1594_159492


namespace NUMINAMATH_GPT_sequences_recurrence_relation_l1594_159453

theorem sequences_recurrence_relation 
    (a : ℕ → ℕ) (b : ℕ → ℕ) (c : ℕ → ℕ)
    (h1 : a 1 = 1) (h2 : b 1 = 3) (h3 : c 1 = 2)
    (ha : ∀ i : ℕ, a (i + 1) = a i + c i - b i + 2)
    (hb : ∀ i : ℕ, b (i + 1) = (3 * c i - a i + 5) / 2)
    (hc : ∀ i : ℕ, c (i + 1) = 2 * a i + 2 * b i - 3) : 
    (∀ n, a n = 2^(n-1)) ∧ (∀ n, b n = 2^n + 1) ∧ (∀ n, c n = 3 * 2^(n-1) - 1) := 
sorry

end NUMINAMATH_GPT_sequences_recurrence_relation_l1594_159453


namespace NUMINAMATH_GPT_arctan_3_4_add_arctan_4_3_is_pi_div_2_l1594_159499

noncomputable def arctan_add (a b : ℝ) : ℝ :=
  Real.arctan a + Real.arctan b

theorem arctan_3_4_add_arctan_4_3_is_pi_div_2 :
  arctan_add (3 / 4) (4 / 3) = Real.pi / 2 :=
sorry

end NUMINAMATH_GPT_arctan_3_4_add_arctan_4_3_is_pi_div_2_l1594_159499


namespace NUMINAMATH_GPT_find_number_l1594_159409

theorem find_number (x : ℕ) (h : x / 3 = 3) : x = 9 :=
sorry

end NUMINAMATH_GPT_find_number_l1594_159409


namespace NUMINAMATH_GPT_sarah_monthly_payment_l1594_159460

noncomputable def monthly_payment (loan_amount : ℝ) (down_payment : ℝ) (years : ℝ) : ℝ :=
  let financed_amount := loan_amount - down_payment
  let months := years * 12
  financed_amount / months

theorem sarah_monthly_payment : monthly_payment 46000 10000 5 = 600 := by
  sorry

end NUMINAMATH_GPT_sarah_monthly_payment_l1594_159460


namespace NUMINAMATH_GPT_alpha_beta_square_inequality_l1594_159403

theorem alpha_beta_square_inequality
  (α β : ℝ)
  (h1 : α ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h2 : β ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h3 : α * Real.sin α - β * Real.sin β > 0) :
  α^2 > β^2 :=
by
  sorry

end NUMINAMATH_GPT_alpha_beta_square_inequality_l1594_159403


namespace NUMINAMATH_GPT_fraction_product_l1594_159482

theorem fraction_product :
  (2 / 3) * (3 / 4) * (5 / 6) * (6 / 7) * (8 / 9) = 80 / 63 :=
by sorry

end NUMINAMATH_GPT_fraction_product_l1594_159482


namespace NUMINAMATH_GPT_geometric_sequence_general_term_l1594_159411

noncomputable def a_n (n : ℕ) : ℝ := 1 * (2:ℝ)^(n-1)

theorem geometric_sequence_general_term : 
  ∀ (n : ℕ), 
  (∀ (n : ℕ), 0 < a_n n) ∧ a_n 1 = 1 ∧ (a_n 1 + a_n 2 + a_n 3 = 7) → 
  a_n n = 2^(n-1) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_general_term_l1594_159411


namespace NUMINAMATH_GPT_probability_in_smaller_spheres_l1594_159494

theorem probability_in_smaller_spheres 
    (R r : ℝ)
    (h_eq : ∀ (R r : ℝ), R + r = 4 * r)
    (vol_eq : ∀ (R r : ℝ), (4/3) * π * r^3 * 5 = (4/3) * π * R^3 * (5/27)) :
    P = 0.2 := by
  sorry

end NUMINAMATH_GPT_probability_in_smaller_spheres_l1594_159494


namespace NUMINAMATH_GPT_solve_eq_log_base_l1594_159407

theorem solve_eq_log_base (x : ℝ) : (9 : ℝ)^(x+8) = (10 : ℝ)^x → x = Real.logb (10 / 9) ((9 : ℝ)^8) := by
  intro h
  sorry

end NUMINAMATH_GPT_solve_eq_log_base_l1594_159407


namespace NUMINAMATH_GPT_problem_solution_l1594_159443

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
if x < 0 then 3^x else m - x^2

def p (m : ℝ) : Prop :=
∃ x, f m x = 0

def q (m : ℝ) : Prop :=
m = 1 / 9 → f m (f m (-1)) = 0

theorem problem_solution :
  ¬ (∃ m, m < 0 ∧ p m) ∧ q (1 / 9) :=
by 
  sorry

end NUMINAMATH_GPT_problem_solution_l1594_159443


namespace NUMINAMATH_GPT_measure_of_angle_x_l1594_159466

-- Given conditions
def angle_ABC : ℝ := 120
def angle_BAD : ℝ := 31
def angle_BDA (x : ℝ) : Prop := x + 60 + 31 = 180 

-- Statement to prove
theorem measure_of_angle_x : 
  ∃ x : ℝ, angle_BDA x → x = 89 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_angle_x_l1594_159466


namespace NUMINAMATH_GPT_balls_in_boxes_l1594_159465

theorem balls_in_boxes:
  ∃ (x y z : ℕ), 
  x + y + z = 320 ∧ 
  6 * x + 11 * y + 15 * z = 1001 ∧
  x > 0 ∧ y > 0 ∧ z > 0 :=
by
  sorry

end NUMINAMATH_GPT_balls_in_boxes_l1594_159465


namespace NUMINAMATH_GPT_no_all_blue_possible_l1594_159402

-- Define initial counts of chameleons
def initial_red : ℕ := 25
def initial_green : ℕ := 12
def initial_blue : ℕ := 8

-- Define the invariant condition
def invariant (r g : ℕ) : Prop := (r - g) % 3 = 1

-- Define the main theorem statement
theorem no_all_blue_possible : ¬∃ r g, r = 0 ∧ g = 0 ∧ invariant r g :=
by {
  sorry
}

end NUMINAMATH_GPT_no_all_blue_possible_l1594_159402


namespace NUMINAMATH_GPT_chinese_chess_draw_probability_l1594_159424

theorem chinese_chess_draw_probability (pMingNotLosing : ℚ) (pDongLosing : ℚ) : 
    pMingNotLosing = 3/4 → 
    pDongLosing = 1/2 → 
    (pMingNotLosing - (1 - pDongLosing)) = 1/4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_chinese_chess_draw_probability_l1594_159424


namespace NUMINAMATH_GPT_zero_in_A_l1594_159413

-- Define the set A
def A : Set ℝ := { x | x * (x - 2) = 0 }

-- State the theorem
theorem zero_in_A : 0 ∈ A :=
by {
  -- Skipping the actual proof with "sorry"
  sorry
}

end NUMINAMATH_GPT_zero_in_A_l1594_159413


namespace NUMINAMATH_GPT_rectangle_area_l1594_159473

theorem rectangle_area (l w : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 250) : l * w = 2500 :=
  sorry

end NUMINAMATH_GPT_rectangle_area_l1594_159473


namespace NUMINAMATH_GPT_negation_if_proposition_l1594_159454

variable (a b : Prop)

theorem negation_if_proposition (a b : Prop) : ¬ (a → b) = a ∧ ¬b := 
sorry

end NUMINAMATH_GPT_negation_if_proposition_l1594_159454


namespace NUMINAMATH_GPT_problem1_correct_problem2_correct_l1594_159428

noncomputable def problem1 : Real :=
  2 * Real.sqrt (2 / 3) - 3 * Real.sqrt (3 / 2) + Real.sqrt 24

theorem problem1_correct : problem1 = (7 * Real.sqrt 6) / 6 := by
  sorry

noncomputable def problem2 : Real :=
  Real.sqrt (25 / 2) + Real.sqrt 32 - Real.sqrt 18 - (Real.sqrt 2 - 1)^2

theorem problem2_correct : problem2 = (11 * Real.sqrt 2) / 2 - 3 := by
  sorry

end NUMINAMATH_GPT_problem1_correct_problem2_correct_l1594_159428


namespace NUMINAMATH_GPT_slope_of_perpendicular_line_l1594_159477

theorem slope_of_perpendicular_line 
  (x1 y1 x2 y2 : ℤ)
  (h : x1 = 3 ∧ y1 = -4 ∧ x2 = -6 ∧ y2 = 2) : 
∃ m : ℚ, m = 3/2 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_perpendicular_line_l1594_159477


namespace NUMINAMATH_GPT_sum_of_products_of_two_at_a_time_l1594_159439

-- Given conditions
variables (a b c : ℝ)
axiom sum_of_squares : a^2 + b^2 + c^2 = 252
axiom sum_of_numbers : a + b + c = 22

-- The goal
theorem sum_of_products_of_two_at_a_time : a * b + b * c + c * a = 116 :=
sorry

end NUMINAMATH_GPT_sum_of_products_of_two_at_a_time_l1594_159439


namespace NUMINAMATH_GPT_segments_form_quadrilateral_l1594_159417

theorem segments_form_quadrilateral (a d : ℝ) (h_pos : a > 0 ∧ d > 0) (h_sum : 4 * a + 6 * d = 3) : 
  (∃ s1 s2 s3 s4 : ℝ, s1 + s2 + s3 > s4 ∧ s1 + s2 + s4 > s3 ∧ s1 + s3 + s4 > s2 ∧ s2 + s3 + s4 > s1) :=
sorry

end NUMINAMATH_GPT_segments_form_quadrilateral_l1594_159417


namespace NUMINAMATH_GPT_total_coffee_consumed_l1594_159484

def Ivory_hourly_coffee := 2
def Kimberly_hourly_coffee := Ivory_hourly_coffee
def Brayan_hourly_coffee := 4
def Raul_hourly_coffee := Brayan_hourly_coffee / 2
def duration_hours := 10

theorem total_coffee_consumed :
  (Brayan_hourly_coffee * duration_hours) + 
  (Ivory_hourly_coffee * duration_hours) + 
  (Kimberly_hourly_coffee * duration_hours) + 
  (Raul_hourly_coffee * duration_hours) = 100 :=
by sorry

end NUMINAMATH_GPT_total_coffee_consumed_l1594_159484


namespace NUMINAMATH_GPT_smallest_triangle_perimeter_l1594_159474

theorem smallest_triangle_perimeter :
  ∃ (y : ℕ), (y % 2 = 0) ∧ (y < 17) ∧ (y > 3) ∧ (7 + 10 + y = 21) :=
by
  sorry

end NUMINAMATH_GPT_smallest_triangle_perimeter_l1594_159474


namespace NUMINAMATH_GPT_circle_radius_l1594_159470
open Real

theorem circle_radius (d : ℝ) (h_diam : d = 24) : d / 2 = 12 :=
by
  -- The proof will be here
  sorry

end NUMINAMATH_GPT_circle_radius_l1594_159470


namespace NUMINAMATH_GPT_find_f_three_l1594_159447

noncomputable def f : ℝ → ℝ := sorry

axiom additivity (a b : ℝ) : f (a + b) = f a + f b
axiom f_two : f 2 = 3

theorem find_f_three : f 3 = 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_f_three_l1594_159447


namespace NUMINAMATH_GPT_parabola_distance_l1594_159481

theorem parabola_distance (p : ℝ) : 
  (∃ p: ℝ, y^2 = 10*x ∧ 2*p = 10) → p = 5 :=
by
  sorry

end NUMINAMATH_GPT_parabola_distance_l1594_159481


namespace NUMINAMATH_GPT_find_point_of_intersection_l1594_159427
noncomputable def point_of_intersection_curve_line : Prop :=
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x^y = y^x ∧ y = x ∧ x = Real.exp 1 ∧ y = Real.exp 1

theorem find_point_of_intersection : point_of_intersection_curve_line :=
sorry

end NUMINAMATH_GPT_find_point_of_intersection_l1594_159427


namespace NUMINAMATH_GPT_volume_original_cone_l1594_159432

-- Given conditions
def V_cylinder : ℝ := 21
def V_truncated_cone : ℝ := 91

-- To prove: The volume of the original cone is 94.5
theorem volume_original_cone : 
    (∃ (H R h r : ℝ), (π * r^2 * h = V_cylinder) ∧ (1 / 3 * π * (R^2 + R * r + r^2) * (H - h) = V_truncated_cone)) →
    (1 / 3 * π * R^2 * H = 94.5) :=
by
  sorry

end NUMINAMATH_GPT_volume_original_cone_l1594_159432


namespace NUMINAMATH_GPT_sum_of_money_proof_l1594_159461

noncomputable def total_sum (A B C : ℝ) : ℝ := A + B + C

theorem sum_of_money_proof (A B C : ℝ) (h1 : B = 0.65 * A) (h2 : C = 0.40 * A) (h3 : C = 64) : total_sum A B C = 328 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_money_proof_l1594_159461


namespace NUMINAMATH_GPT_brick_width_l1594_159487

/-- Let dimensions of the wall be 700 cm (length), 600 cm (height), and 22.5 cm (thickness).
    Let dimensions of each brick be 25 cm (length), W cm (width), and 6 cm (height).
    Given that 5600 bricks are required to build the wall, prove that the width of each brick is 11.25 cm. -/
theorem brick_width (W : ℝ)
  (h_wall_dimensions : 700 = 700) (h_wall_height : 600 = 600) (h_wall_thickness : 22.5 = 22.5)
  (h_brick_length : 25 = 25) (h_brick_height : 6 = 6) (h_num_bricks : 5600 = 5600)
  (h_wall_volume : 700 * 600 * 22.5 = 9450000)
  (h_brick_volume : 25 * W * 6 = 9450000 / 5600) :
  W = 11.25 :=
sorry

end NUMINAMATH_GPT_brick_width_l1594_159487


namespace NUMINAMATH_GPT_solve_tetrahedron_side_length_l1594_159459

noncomputable def side_length_of_circumscribing_tetrahedron (r : ℝ) (tangent_spheres : ℕ) (radius_spheres_equal : ℝ) : ℝ := 
  if h : r = 1 ∧ tangent_spheres = 4 then
    2 + 2 * Real.sqrt 6
  else
    0

theorem solve_tetrahedron_side_length :
  side_length_of_circumscribing_tetrahedron 1 4 1 = 2 + 2 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_solve_tetrahedron_side_length_l1594_159459


namespace NUMINAMATH_GPT_kittens_more_than_twice_puppies_l1594_159425

-- Define the number of puppies
def num_puppies : ℕ := 32

-- Define the number of kittens
def num_kittens : ℕ := 78

-- Define the problem statement
theorem kittens_more_than_twice_puppies :
  num_kittens = 2 * num_puppies + 14 :=
by sorry

end NUMINAMATH_GPT_kittens_more_than_twice_puppies_l1594_159425


namespace NUMINAMATH_GPT_simplify_expression_l1594_159429

theorem simplify_expression (x y : ℝ) :  3 * x + 5 * x + 7 * x + 2 * y = 15 * x + 2 * y := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l1594_159429


namespace NUMINAMATH_GPT_product_abc_l1594_159455

theorem product_abc (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h : a * b^3 = 180) : a * b * c = 60 * c := 
sorry

end NUMINAMATH_GPT_product_abc_l1594_159455


namespace NUMINAMATH_GPT_worker_total_pay_l1594_159469

def regular_rate : ℕ := 10
def total_surveys : ℕ := 50
def cellphone_surveys : ℕ := 35
def non_cellphone_surveys := total_surveys - cellphone_surveys
def higher_rate := regular_rate + (30 * regular_rate / 100)

def pay_non_cellphone_surveys := non_cellphone_surveys * regular_rate
def pay_cellphone_surveys := cellphone_surveys * higher_rate

def total_pay := pay_non_cellphone_surveys + pay_cellphone_surveys

theorem worker_total_pay : total_pay = 605 := by
  sorry

end NUMINAMATH_GPT_worker_total_pay_l1594_159469


namespace NUMINAMATH_GPT_rectangle_area_l1594_159449

theorem rectangle_area (l : ℝ) (w : ℝ) (h_l : l = 15) (h_ratio : (2 * l + 2 * w) / w = 5) : (l * w) = 150 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l1594_159449


namespace NUMINAMATH_GPT_exists_positive_M_l1594_159438

open Set

noncomputable def f (x : ℝ) : ℝ := sorry

theorem exists_positive_M 
  (h₁ : ∀ x ∈ Ioo (0 : ℝ) 1, f x > 0)
  (h₂ : ∀ x ∈ Ioo (0 : ℝ) 1, f (2 * x / (1 + x^2)) = 2 * f x) :
  ∃ M > 0, ∀ x ∈ Ioo (0 : ℝ) 1, f x ≤ M :=
sorry

end NUMINAMATH_GPT_exists_positive_M_l1594_159438


namespace NUMINAMATH_GPT_interest_rate_determination_l1594_159450

-- Problem statement
theorem interest_rate_determination (P r : ℝ) :
  (50 = P * r * 2) ∧ (51.25 = P * ((1 + r) ^ 2 - 1)) → r = 0.05 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_interest_rate_determination_l1594_159450


namespace NUMINAMATH_GPT_edric_hourly_rate_l1594_159446

-- Define conditions
def edric_monthly_salary : ℝ := 576
def edric_weekly_hours : ℝ := 8 * 6 -- 48 hours
def average_weeks_per_month : ℝ := 4.33
def edric_monthly_hours : ℝ := edric_weekly_hours * average_weeks_per_month -- Approx 207.84 hours

-- Define the expected result
def edric_expected_hourly_rate : ℝ := 2.77

-- Proof statement
theorem edric_hourly_rate :
  edric_monthly_salary / edric_monthly_hours = edric_expected_hourly_rate :=
by
  sorry

end NUMINAMATH_GPT_edric_hourly_rate_l1594_159446


namespace NUMINAMATH_GPT_expected_value_is_minus_one_fifth_l1594_159471

-- Define the parameters given in the problem
def p_heads := 2 / 5
def p_tails := 3 / 5
def win_heads := 4
def loss_tails := -3

-- Calculate the expected value for heads and tails
def expected_heads := p_heads * win_heads
def expected_tails := p_tails * loss_tails

-- The theorem stating that the expected value is -1/5
theorem expected_value_is_minus_one_fifth :
  expected_heads + expected_tails = -1 / 5 :=
by
  -- The proof can be filled in here
  sorry

end NUMINAMATH_GPT_expected_value_is_minus_one_fifth_l1594_159471


namespace NUMINAMATH_GPT_no_divisor_form_24k_20_l1594_159488

theorem no_divisor_form_24k_20 (n : ℕ) : ¬ ∃ k : ℕ, 24 * k + 20 ∣ 3^n + 1 :=
sorry

end NUMINAMATH_GPT_no_divisor_form_24k_20_l1594_159488


namespace NUMINAMATH_GPT_rainfall_ratio_l1594_159412

theorem rainfall_ratio (R_1 R_2 : ℕ) (h1 : R_1 + R_2 = 25) (h2 : R_2 = 15) : R_2 / R_1 = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_rainfall_ratio_l1594_159412


namespace NUMINAMATH_GPT_pairs_nat_eq_l1594_159467

theorem pairs_nat_eq (n k : ℕ) :
  (n + 1) ^ k = n! + 1 ↔ (n, k) = (1, 1) ∨ (n, k) = (2, 1) ∨ (n, k) = (4, 2) :=
by
  sorry

end NUMINAMATH_GPT_pairs_nat_eq_l1594_159467


namespace NUMINAMATH_GPT_exists_real_solution_real_solution_specific_values_l1594_159486

theorem exists_real_solution (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) : 
  ∃ x : ℝ, (a * b^x)^(x + 1) = c :=
sorry

theorem real_solution_specific_values  (a b c : ℝ) (ha : a = 3) (hb : b = 4) (hc : c = 5) : 
  ∃ x : ℝ, (a * b^x)^(x + 1) = c :=
sorry

end NUMINAMATH_GPT_exists_real_solution_real_solution_specific_values_l1594_159486


namespace NUMINAMATH_GPT_fraction_savings_on_makeup_l1594_159401

theorem fraction_savings_on_makeup (savings : ℝ) (sweater_cost : ℝ) (makeup_cost : ℝ) (h_savings : savings = 80) (h_sweater : sweater_cost = 20) (h_makeup : makeup_cost = savings - sweater_cost) : makeup_cost / savings = 3 / 4 := by
  sorry

end NUMINAMATH_GPT_fraction_savings_on_makeup_l1594_159401


namespace NUMINAMATH_GPT_arithmetic_sequence_ninth_term_l1594_159452

theorem arithmetic_sequence_ninth_term (a d : ℤ)
  (h1 : a + 2 * d = 5)
  (h2 : a + 5 * d = 11) :
  a + 8 * d = 17 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_ninth_term_l1594_159452


namespace NUMINAMATH_GPT_determine_right_triangle_l1594_159475

-- Definitions based on conditions
def condition_A (A B C : ℝ) : Prop := A^2 + B^2 = C^2
def condition_B (A B C : ℝ) : Prop := A^2 - B^2 = C^2
def condition_C (A B C : ℝ) : Prop := A + B = C
def condition_D (A B C : ℝ) : Prop := A / B = 3 / 4 ∧ B / C = 4 / 5

-- Problem statement: D cannot determine that triangle ABC is a right triangle
theorem determine_right_triangle (A B C : ℝ) : ¬ condition_D A B C :=
by sorry

end NUMINAMATH_GPT_determine_right_triangle_l1594_159475


namespace NUMINAMATH_GPT_solve_for_D_d_Q_R_l1594_159434

theorem solve_for_D_d_Q_R (D d Q R : ℕ) 
    (h1 : D = d * Q + R) 
    (h2 : d * Q = 135) 
    (h3 : R = 2 * d) : 
    D = 165 ∧ d = 15 ∧ Q = 9 ∧ R = 30 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_D_d_Q_R_l1594_159434


namespace NUMINAMATH_GPT_geese_problem_l1594_159497

theorem geese_problem 
  (G : ℕ)  -- Total number of geese in the original V formation
  (T : ℕ)  -- Number of geese that flew up from the trees to join the new V formation
  (h1 : G / 2 + T = 12)  -- Final number of geese flying in the V formation was 12 
  (h2 : T = G / 2)  -- Number of geese that flew out from the trees is the same as the number of geese that landed initially
: T = 6 := 
sorry

end NUMINAMATH_GPT_geese_problem_l1594_159497


namespace NUMINAMATH_GPT_negation_proposition_l1594_159406

theorem negation_proposition :
  ¬(∃ x_0 : ℝ, x_0^2 + x_0 - 2 < 0) ↔ ∀ x_0 : ℝ, x_0^2 + x_0 - 2 ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_l1594_159406


namespace NUMINAMATH_GPT_part1_part2_l1594_159451

def f (x : ℝ) : ℝ :=
  abs (x - 1) + 2 * abs (x + 5)

theorem part1 : ∀ x, f x < 10 ↔ (x > -19 / 3 ∧ x ≤ -5) ∨ (-5 < x ∧ x < -1) :=
  sorry

theorem part2 (a b x : ℝ) (ha : abs a < 3) (hb : abs b < 3) :
  abs (a + b) + abs (a - b) < f x :=
  sorry

end NUMINAMATH_GPT_part1_part2_l1594_159451


namespace NUMINAMATH_GPT_fraction_evaluation_l1594_159495

theorem fraction_evaluation :
  (1 / 2 + 1 / 3) / (3 / 7 - 1 / 5) = 175 / 48 :=
by
  sorry

end NUMINAMATH_GPT_fraction_evaluation_l1594_159495


namespace NUMINAMATH_GPT_office_person_count_l1594_159422

theorem office_person_count
    (N : ℕ)
    (avg_age_all : ℕ)
    (num_5 : ℕ)
    (avg_age_5 : ℕ)
    (num_9 : ℕ)
    (avg_age_9 : ℕ)
    (age_15th : ℕ)
    (h1 : avg_age_all = 15)
    (h2 : num_5 = 5)
    (h3 : avg_age_5 = 14)
    (h4 : num_9 = 9)
    (h5 : avg_age_9 = 16)
    (h6 : age_15th = 86)
    (h7 : 15 * N = (num_5 * avg_age_5) + (num_9 * avg_age_9) + age_15th) :
    N = 20 :=
by
    -- Proof will be provided here
    sorry

end NUMINAMATH_GPT_office_person_count_l1594_159422


namespace NUMINAMATH_GPT_general_equation_l1594_159445

theorem general_equation (n : ℤ) : 
    ∀ (a b : ℤ), 
    (a = 2 ∧ b = 6) ∨ (a = 5 ∧ b = 3) ∨ (a = 7 ∧ b = 1) ∨ (a = 10 ∧ b = -2) → 
    (a / (a - 4) + b / (b - 4) = 2) →
    (n / (n - 4) + (8 - n) / ((8 - n) - 4) = 2) :=
by
  intros a b h_cond h_eq
  sorry

end NUMINAMATH_GPT_general_equation_l1594_159445


namespace NUMINAMATH_GPT_susan_cars_fewer_than_carol_l1594_159479

theorem susan_cars_fewer_than_carol 
  (Lindsey_cars Carol_cars Susan_cars Cathy_cars : ℕ)
  (h1 : Lindsey_cars = Cathy_cars + 4)
  (h2 : Susan_cars < Carol_cars)
  (h3 : Carol_cars = 2 * Cathy_cars)
  (h4 : Cathy_cars = 5)
  (h5 : Cathy_cars + Carol_cars + Lindsey_cars + Susan_cars = 32) :
  Carol_cars - Susan_cars = 2 :=
sorry

end NUMINAMATH_GPT_susan_cars_fewer_than_carol_l1594_159479


namespace NUMINAMATH_GPT_real_solutions_of_polynomial_l1594_159457

theorem real_solutions_of_polynomial (b : ℝ) :
  b < -4 → ∃! x : ℝ, x^3 - b * x^2 - 4 * b * x + b^2 - 4 = 0 :=
by
  sorry

end NUMINAMATH_GPT_real_solutions_of_polynomial_l1594_159457


namespace NUMINAMATH_GPT_cost_of_dinner_l1594_159436

theorem cost_of_dinner (x : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (total_cost : ℝ) : 
  tax_rate = 0.09 → tip_rate = 0.18 → total_cost = 36.90 → 
  1.27 * x = 36.90 → x = 29 :=
by
  intros htr htt htc heq
  rw [←heq] at htc
  sorry

end NUMINAMATH_GPT_cost_of_dinner_l1594_159436


namespace NUMINAMATH_GPT_restaurant_vegetarian_dishes_l1594_159489

theorem restaurant_vegetarian_dishes (n : ℕ) : 
    5 ≥ 2 → 200 < Nat.choose 5 2 * Nat.choose n 2 → n ≥ 7 :=
by
  intros h_combinations h_least
  sorry

end NUMINAMATH_GPT_restaurant_vegetarian_dishes_l1594_159489


namespace NUMINAMATH_GPT_circle_radius_5_l1594_159410

theorem circle_radius_5 (k : ℝ) : 
  (∃ x y : ℝ, x^2 + 14 * x + y^2 + 8 * y - k = 0) ↔ k = -40 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_5_l1594_159410


namespace NUMINAMATH_GPT_adam_and_simon_distance_l1594_159464

theorem adam_and_simon_distance :
  ∀ (t : ℝ), (10 * t)^2 + (12 * t)^2 = 16900 → t = 65 / Real.sqrt 61 :=
by
  sorry

end NUMINAMATH_GPT_adam_and_simon_distance_l1594_159464
