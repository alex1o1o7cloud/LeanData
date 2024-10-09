import Mathlib

namespace vasya_cuts_larger_area_l1091_109187

noncomputable def E_Vasya_square_area : ℝ :=
  (1/6) * (1^2) + (1/6) * (2^2) + (1/6) * (3^2) + (1/6) * (4^2) + (1/6) * (5^2) + (1/6) * (6^2)

noncomputable def E_Asya_rectangle_area : ℝ :=
  (3.5 * 3.5)

theorem vasya_cuts_larger_area :
  E_Vasya_square_area > E_Asya_rectangle_area :=
  by
    sorry

end vasya_cuts_larger_area_l1091_109187


namespace math_problem_l1091_109199

theorem math_problem :
  let result := 83 - 29
  let final_sum := result + 58
  let rounded := if final_sum % 10 < 5 then final_sum - final_sum % 10 else final_sum + (10 - final_sum % 10)
  rounded = 110 := by
  sorry

end math_problem_l1091_109199


namespace asymptotes_of_hyperbola_l1091_109116

variable (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
variable (h3 : (1 + b^2 / a^2) = (6 / 4))

theorem asymptotes_of_hyperbola :
  ∃ (m : ℝ), m = b / a ∧ (m = Real.sqrt 2 / 2) ∧ ∀ x : ℝ, (y = m*x) ∨ (y = -m*x) :=
by
  sorry

end asymptotes_of_hyperbola_l1091_109116


namespace original_height_l1091_109180

theorem original_height (h : ℝ) (h_rebound : ∀ n : ℕ, h / (4/3)^(n+1) > 0) (total_distance : ∀ h : ℝ, h*(1 + 1.5 + 1.5*(0.75) + 1.5*(0.75)^2 + 1.5*(0.75)^3 + (0.75)^4) = 305) :
  h = 56.3 := 
sorry

end original_height_l1091_109180


namespace mirror_area_proof_l1091_109182

-- Definitions of conditions
def outer_width := 100
def outer_height := 70
def frame_width := 15
def mirror_width := outer_width - 2 * frame_width -- 100 - 2 * 15 = 70
def mirror_height := outer_height - 2 * frame_width -- 70 - 2 * 15 = 40

-- Statement of the proof problem
theorem mirror_area_proof : 
  (mirror_width * mirror_height) = 2800 := 
by
  sorry

end mirror_area_proof_l1091_109182


namespace shoes_sold_first_week_eq_100k_l1091_109194

-- Define variables for purchase price and total revenue
def purchase_price : ℝ := 180
def total_revenue : ℝ := 216

-- Define markups
def first_week_markup : ℝ := 1.25
def remaining_markup : ℝ := 1.16

-- Define the conditions
theorem shoes_sold_first_week_eq_100k (x y : ℝ) 
  (h1 : x + y = purchase_price) 
  (h2 : first_week_markup * x + remaining_markup * y = total_revenue) :
  first_week_markup * x = 100  := 
sorry

end shoes_sold_first_week_eq_100k_l1091_109194


namespace sector_area_proof_l1091_109198

-- Define variables for the central angle, arc length, and derived radius
variables (θ L : ℝ) (r A: ℝ)

-- Define the conditions given in the problem
def central_angle_condition : Prop := θ = 2
def arc_length_condition : Prop := L = 4
def radius_condition : Prop := r = L / θ

-- Define the formula for the area of the sector
def area_of_sector_condition : Prop := A = (1 / 2) * r^2 * θ

-- The theorem that needs to be proved
theorem sector_area_proof :
  central_angle_condition θ ∧ arc_length_condition L ∧ radius_condition θ L r ∧ area_of_sector_condition r θ A → A = 4 :=
by
  sorry

end sector_area_proof_l1091_109198


namespace no_integer_solution_l1091_109189

theorem no_integer_solution (x : ℤ) : ¬ (x + 12 > 15 ∧ -3 * x > -9) :=
by {
  sorry
}

end no_integer_solution_l1091_109189


namespace find_other_factor_l1091_109149

theorem find_other_factor (n : ℕ) (hn : n = 75) :
    ( ∃ k, k = 25 ∧ ∃ m, (k * 3^3 * m = 75 * 2^5 * 6^2 * 7^3) ) :=
by
  sorry

end find_other_factor_l1091_109149


namespace circular_pipes_equivalence_l1091_109185

/-- Determine how many circular pipes with an inside diameter 
of 2 inches are required to carry the same amount of water as 
one circular pipe with an inside diameter of 8 inches. -/
theorem circular_pipes_equivalence 
  (d_small d_large : ℝ)
  (h1 : d_small = 2)
  (h2 : d_large = 8) :
  (d_large / 2) ^ 2 / (d_small / 2) ^ 2 = 16 :=
by
  sorry

end circular_pipes_equivalence_l1091_109185


namespace abs_sum_factors_l1091_109119

theorem abs_sum_factors (a b c d : ℤ) : 
  (6 * x ^ 2 + x - 12 = (a * x + b) * (c * x + d)) →
  (|a| + |b| + |c| + |d| = 12) :=
by
  intros h
  sorry

end abs_sum_factors_l1091_109119


namespace product_of_abc_l1091_109114

noncomputable def abc_product (a b c : ℝ) : ℝ :=
  a * b * c

theorem product_of_abc (a b c m : ℝ) 
    (h1 : a + b + c = 300)
    (h2 : m = 5 * a)
    (h3 : m = b + 14)
    (h4 : m = c - 14) : 
    abc_product a b c = 664500 :=
by sorry

end product_of_abc_l1091_109114


namespace molecular_weight_of_Y_l1091_109129

def molecular_weight_X : ℝ := 136
def molecular_weight_C6H8O7 : ℝ := 192
def moles_C6H8O7 : ℝ := 5

def total_mass_reactants := molecular_weight_X + moles_C6H8O7 * molecular_weight_C6H8O7

theorem molecular_weight_of_Y :
  total_mass_reactants = 1096 := by
  sorry

end molecular_weight_of_Y_l1091_109129


namespace find_pairs_l1091_109105

theorem find_pairs :
  { (m, n) : ℕ × ℕ | (m > 0) ∧ (n > 0) ∧ (m^2 - n ∣ m + n^2)
      ∧ (n^2 - m ∣ n + m^2) } = { (2, 2), (3, 3), (1, 2), (2, 1), (3, 2), (2, 3) } :=
sorry

end find_pairs_l1091_109105


namespace at_least_502_friendly_numbers_l1091_109191

def friendly (a : ℤ) : Prop :=
  ∃ (m n : ℤ), m > 0 ∧ n > 0 ∧ (m^2 + n) * (n^2 + m) = a * (m - n)^3

theorem at_least_502_friendly_numbers :
  ∃ S : Finset ℤ, (∀ a ∈ S, friendly a) ∧ 502 ≤ S.card ∧ ∀ x ∈ S, 1 ≤ x ∧ x ≤ 2012 :=
by
  sorry

end at_least_502_friendly_numbers_l1091_109191


namespace percent_other_sales_l1091_109127

-- Define the given conditions
def s_brushes : ℝ := 0.45
def s_paints : ℝ := 0.28

-- Define the proof goal in Lean
theorem percent_other_sales :
  1 - (s_brushes + s_paints) = 0.27 := by
-- Adding the conditions to the proof environment
  sorry

end percent_other_sales_l1091_109127


namespace commencement_addresses_sum_l1091_109130

noncomputable def addresses (S H L : ℕ) := 40

theorem commencement_addresses_sum
  (S H L : ℕ)
  (h1 : S = 12)
  (h2 : S = 2 * H)
  (h3 : L = S + 10) :
  S + H + L = addresses S H L :=
by
  sorry

end commencement_addresses_sum_l1091_109130


namespace negation_equivalence_l1091_109106

-- Definition of the original proposition
def proposition (x : ℝ) : Prop := x > 1 → Real.log x > 0

-- Definition of the negated proposition
def negation (x : ℝ) : Prop := ¬ (x > 1 → Real.log x > 0)

-- The mathematically equivalent proof problem as Lean statement
theorem negation_equivalence (x : ℝ) : 
  (¬ (x > 1 → Real.log x > 0)) ↔ (x ≤ 1 → Real.log x ≤ 0) := 
by 
  sorry

end negation_equivalence_l1091_109106


namespace product_pattern_l1091_109107

theorem product_pattern (a b : ℕ) (h1 : b < 10) (h2 : 10 - b < 10) :
    (10 * a + b) * (10 * a + (10 - b)) = 100 * a * (a + 1) + b * (10 - b) :=
by
  sorry

end product_pattern_l1091_109107


namespace find_d_l1091_109146

theorem find_d (d : ℤ) :
  (∀ x : ℤ, 6 * x^3 + 19 * x^2 + d * x - 15 = 0) ->
  d = -32 :=
by
  sorry

end find_d_l1091_109146


namespace find_salary_B_l1091_109140

def salary_A : ℕ := 8000
def salary_C : ℕ := 11000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000
def avg_salary : ℕ := 8000

theorem find_salary_B (S_B : ℕ) :
  (salary_A + S_B + salary_C + salary_D + salary_E) / 5 = avg_salary ↔ S_B = 5000 := by
  sorry

end find_salary_B_l1091_109140


namespace relationship_between_abc_l1091_109171

theorem relationship_between_abc (u v a b c : ℝ)
  (h1 : u - v = a) 
  (h2 : u^2 - v^2 = b)
  (h3 : u^3 - v^3 = c) : 
  3 * b ^ 2 + a ^ 4 = 4 * a * c :=
sorry

end relationship_between_abc_l1091_109171


namespace negation_proposition_l1091_109156

theorem negation_proposition (a b : ℝ) :
  (a * b ≠ 0) → (a ≠ 0 ∧ b ≠ 0) :=
by
  sorry

end negation_proposition_l1091_109156


namespace circumradius_of_triangle_l1091_109131

theorem circumradius_of_triangle (a b c : ℝ) (h₁ : a = 8) (h₂ : b = 10) (h₃ : c = 14) : 
  R = (35 * Real.sqrt 2) / 3 :=
by
  sorry

end circumradius_of_triangle_l1091_109131


namespace logan_buys_15_pounds_of_corn_l1091_109150

theorem logan_buys_15_pounds_of_corn (c b : ℝ) 
    (h1 : 1.20 * c + 0.60 * b = 27) 
    (h2 : b + c = 30) : 
    c = 15.0 :=
by
  sorry

end logan_buys_15_pounds_of_corn_l1091_109150


namespace square_area_is_8_point_0_l1091_109102

theorem square_area_is_8_point_0 (A B C D E F : ℝ) 
    (h_square : E + F = 4)
    (h_diag : 1 + 2 + 1 = 4) : 
    ∃ (s : ℝ), s^2 = 8 :=
by
  sorry

end square_area_is_8_point_0_l1091_109102


namespace find_angle_BDC_l1091_109167

theorem find_angle_BDC
  (CAB CAD DBA DBC : ℝ)
  (h1 : CAB = 40)
  (h2 : CAD = 30)
  (h3 : DBA = 75)
  (h4 : DBC = 25) :
  ∃ BDC : ℝ, BDC = 45 :=
by
  sorry

end find_angle_BDC_l1091_109167


namespace card_selection_l1091_109159

noncomputable def count_ways := 438400

theorem card_selection :
  let decks := 2
  let total_cards := 52 * decks
  let suits := 4
  let non_royal_count := 10 * decks
  let royal_count := 3 * decks
  let non_royal_options := non_royal_count * decks
  let royal_options := royal_count * decks
  1 * (non_royal_options)^4 + (suits.choose 1) * royal_options * (non_royal_options)^3 + (suits.choose 2) * (royal_options)^2 * (non_royal_options)^2 = count_ways :=
sorry

end card_selection_l1091_109159


namespace children_got_off_bus_l1091_109193

-- Conditions
def original_number_of_children : ℕ := 43
def children_left_on_bus : ℕ := 21

-- Definition of the number of children who got off the bus
def children_got_off : ℕ := original_number_of_children - children_left_on_bus

-- Theorem stating the number of children who got off the bus
theorem children_got_off_bus : children_got_off = 22 :=
by
  -- This is to indicate where the proof would go
  sorry

end children_got_off_bus_l1091_109193


namespace well_depth_and_rope_length_l1091_109169

theorem well_depth_and_rope_length (h x : ℝ) : 
  (h / 3 = x + 4) ∧ (h / 4 = x + 1) :=
sorry

end well_depth_and_rope_length_l1091_109169


namespace find_x_l1091_109197

theorem find_x (x : ℝ) : (x * 16) / 100 = 0.051871999999999995 → x = 0.3242 := by
  intro h
  sorry

end find_x_l1091_109197


namespace no_such_n_exists_l1091_109178

-- Definition of the sum of the digits function s(n)
def s (n : ℕ) : ℕ := n.digits 10 |> List.sum

-- Statement of the proof problem
theorem no_such_n_exists : ¬ ∃ n : ℕ, n * s n = 20222022 :=
by
  -- argument based on divisibility rules as presented in the problem
  sorry

end no_such_n_exists_l1091_109178


namespace greatest_number_of_sets_l1091_109186

-- We define the number of logic and visual puzzles.
def n_logic : ℕ := 18
def n_visual : ℕ := 9

-- The theorem states that the greatest number of identical sets Mrs. Wilson can create is the GCD of 18 and 9.
theorem greatest_number_of_sets : gcd n_logic n_visual = 9 := by
  sorry

end greatest_number_of_sets_l1091_109186


namespace number_of_divisors_of_36_l1091_109184

theorem number_of_divisors_of_36 : Nat.totient 36 = 9 := 
by sorry

end number_of_divisors_of_36_l1091_109184


namespace seventh_term_geometric_seq_l1091_109196

theorem seventh_term_geometric_seq (a r : ℝ) (h_pos: 0 < r) (h_fifth: a * r^4 = 16) (h_ninth: a * r^8 = 4) : a * r^6 = 8 := by
  sorry

end seventh_term_geometric_seq_l1091_109196


namespace solution_set_of_inequality_l1091_109172

theorem solution_set_of_inequality (x : ℝ) : x^2 - 5 * |x| + 6 < 0 ↔ (-3 < x ∧ x < -2) ∨ (2 < x ∧ x < 3) :=
  sorry

end solution_set_of_inequality_l1091_109172


namespace ellipse_foci_x_axis_l1091_109133

theorem ellipse_foci_x_axis (a b : ℝ) (h : ∀ x y : ℝ, a * x^2 + b * y^2 = 1) : 0 < a ∧ a < b :=
sorry

end ellipse_foci_x_axis_l1091_109133


namespace sqrt_x_minus_1_domain_l1091_109158

theorem sqrt_x_minus_1_domain (x : ℝ) : (∃ y, y^2 = x - 1) → (x ≥ 1) := 
by 
  sorry

end sqrt_x_minus_1_domain_l1091_109158


namespace find_m_n_sum_l1091_109110

noncomputable def point (x : ℝ) (y : ℝ) := (x, y)

def center_line (P : ℝ × ℝ) : Prop := P.1 - P.2 - 2 = 0

def on_circle (C : ℝ × ℝ) (P : ℝ × ℝ) (r : ℝ) : Prop := 
  (P.1 - C.1)^2 + (P.2 - C.2)^2 = r^2

def circles_intersect (A B C D : ℝ × ℝ) (r₁ r₂ : ℝ) : Prop :=
  on_circle A C r₁ ∧ on_circle A D r₂ ∧ on_circle B C r₁ ∧ on_circle B D r₂

theorem find_m_n_sum 
  (A : ℝ × ℝ) (m n : ℝ)
  (C D : ℝ × ℝ)
  (r₁ r₂ : ℝ)
  (H1 : A = point 1 3)
  (H2 : circles_intersect A (point m n) C D r₁ r₂)
  (H3 : center_line C ∧ center_line D) :
  m + n = 4 :=
sorry

end find_m_n_sum_l1091_109110


namespace equation_one_solution_equation_two_solution_l1091_109113

-- Define the conditions and prove the correctness of solutions to the equations
theorem equation_one_solution (x : ℝ) (h : 3 / (x - 2) = 9 / x) : x = 3 :=
by
  sorry

theorem equation_two_solution (x : ℝ) (h : x / (x + 1) = 2 * x / (3 * x + 3) - 1) : x = -3 / 4 :=
by
  sorry

end equation_one_solution_equation_two_solution_l1091_109113


namespace lena_glued_friends_pictures_l1091_109123

-- Define the conditions
def clippings_per_friend : ℕ := 3
def glue_per_clipping : ℕ := 6
def total_glue : ℕ := 126

-- Define the proof problem statement
theorem lena_glued_friends_pictures : 
    ∃ (F : ℕ), F * (clippings_per_friend * glue_per_clipping) = total_glue ∧ F = 7 := 
by
  sorry

end lena_glued_friends_pictures_l1091_109123


namespace complex_number_simplification_l1091_109154

theorem complex_number_simplification (i : ℂ) (hi : i^2 = -1) : i - (1 / i) = 2 * i :=
by
  sorry

end complex_number_simplification_l1091_109154


namespace B_necessary_not_sufficient_for_A_l1091_109100

def A (x : ℝ) : Prop := 0 < x ∧ x < 5
def B (x : ℝ) : Prop := |x - 2| < 3

theorem B_necessary_not_sufficient_for_A (x : ℝ) :
  (A x → B x) ∧ (∃ x, B x ∧ ¬ A x) :=
by
  sorry

end B_necessary_not_sufficient_for_A_l1091_109100


namespace check_independence_and_expected_value_l1091_109139

noncomputable def contingency_table (students: ℕ) (pct_75 : ℕ) (pct_less10 : ℕ) (num_75_10 : ℕ) : (ℕ × ℕ) × (ℕ × ℕ) :=
  let num_75 := students * pct_75 / 100
  let num_less10 := students * pct_less10 / 100
  let num_75_less10 := num_75 - num_75_10
  let num_not75 := students - num_75
  let num_not75_less10 := num_less10 - num_75_less10
  let num_not75_10 := num_not75 - num_not75_less10
  ((num_not75_less10, num_75_less10), (num_not75_10, num_75_10))

noncomputable def chi_square_statistic (a b c d : ℕ) (n: ℕ) : ℚ :=
  (n * ((a * d - b * c)^2)) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem check_independence_and_expected_value :
  let students := 500
  let pct_75 := 30
  let pct_less10 := 50
  let num_75_10 := 100
  let ((num_not75_less10, num_75_less10), (num_not75_10, num_75_10)) := contingency_table students pct_75 pct_less10 num_75_10
  let chi2 := chi_square_statistic num_not75_less10 num_75_less10 num_not75_10 num_75_10 students
  let critical_value := 10.828
  let p0 := 1 / 84
  let p1 := 3 / 14
  let p2 := 15 / 28
  let p3 := 5 / 21
  let expected_x := 0 * p0 + 1 * p1 + 2 * p2 + 3 * p3
  (chi2 > critical_value) ∧ (expected_x = 2) :=
by 
  sorry

end check_independence_and_expected_value_l1091_109139


namespace max_profit_l1091_109155

noncomputable def revenue (x : ℝ) : ℝ := 
  if (0 < x ∧ x ≤ 10) then 13.5 - (1 / 30) * x^2 
  else if (x > 10) then (168 / x) - (2000 / (3 * x^2)) 
  else 0

noncomputable def cost (x : ℝ) : ℝ := 
  20 + 5.4 * x

noncomputable def profit (x : ℝ) : ℝ := revenue x * x - cost x

theorem max_profit : 
  ∃ (x : ℝ), 0 < x ∧ x ≤ 10 ∧ (profit x = 8.1 * x - (1 / 30) * x^3 - 20) ∧ 
    (∀ (y : ℝ), 0 < y ∧ y ≤ 10 → profit y ≤ profit 9) ∧ 
    ∀ (z : ℝ), z > 10 → profit z ≤ profit 9 :=
by
  sorry

end max_profit_l1091_109155


namespace trajectory_of_point_M_l1091_109166

theorem trajectory_of_point_M (a x y : ℝ) (h: 0 < a) (A B M : ℝ × ℝ)
    (hA : A = (x, 0)) (hB : B = (0, y)) (hAB_length : Real.sqrt (x^2 + y^2) = 2 * a)
    (h_ratio : ∃ k, k ≠ 0 ∧ ∃ k', k' ≠ 0 ∧ A = k • M + k' • B ∧ (k + k' = 1) ∧ (k / k' = 1 / 2)) :
    (x / (4 / 3 * a))^2 + (y / (2 / 3 * a))^2 = 1 :=
sorry

end trajectory_of_point_M_l1091_109166


namespace problem1_problem2_l1091_109120

-- Definitions related to the given problem
def polar_curve (ρ θ : ℝ) : Prop :=
  ρ^2 = 9 / (Real.cos θ^2 + 9 * Real.sin θ^2)

def standard_curve (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 = 1

-- Proving the standard equation of the curve
theorem problem1 (ρ θ : ℝ) (h : polar_curve ρ θ) : ∃ x y, standard_curve x y :=
  sorry

-- Proving the perpendicular condition and its consequence
theorem problem2 (ρ1 ρ2 α : ℝ)
  (hA : polar_curve ρ1 α)
  (hB : polar_curve ρ2 (α + π/2))
  (perpendicular : ∀ (A B : (ℝ × ℝ)), A ≠ B → A.1 * B.1 + A.2 * B.2 = 0) :
  (1 / ρ1^2) + (1 / ρ2^2) = 10 / 9 :=
  sorry

end problem1_problem2_l1091_109120


namespace find_amount_with_R_l1091_109144

variable (P_amount Q_amount R_amount : ℝ)
variable (total_amount : ℝ) (r_has_twothirds : Prop)

noncomputable def amount_with_R (total_amount : ℝ) : ℝ :=
  let R_amount := 2 / 3 * (total_amount - R_amount)
  R_amount

theorem find_amount_with_R (P_amount Q_amount R_amount : ℝ) (total_amount : ℝ)
  (h_total : total_amount = 5000)
  (h_two_thirds : R_amount = 2 / 3 * (P_amount + Q_amount)) :
  R_amount = 2000 := by sorry

end find_amount_with_R_l1091_109144


namespace find_x_l1091_109179

noncomputable def h (x : ℚ) : ℚ :=
  (5 * ((x - 2) / 3) - 3)

theorem find_x : h (19/2) = 19/2 :=
by
  sorry

end find_x_l1091_109179


namespace john_unanswered_questions_l1091_109145

theorem john_unanswered_questions :
  ∃ (c w u : ℕ), (30 + 4 * c - w = 84) ∧ (5 * c + 2 * u = 93) ∧ (c + w + u = 30) ∧ (u = 9) :=
by
  sorry

end john_unanswered_questions_l1091_109145


namespace arithmetic_mean_of_17_29_45_64_l1091_109174

theorem arithmetic_mean_of_17_29_45_64 : (17 + 29 + 45 + 64) / 4 = 38.75 := by
  sorry

end arithmetic_mean_of_17_29_45_64_l1091_109174


namespace fraction_red_marbles_l1091_109124

theorem fraction_red_marbles (x : ℕ) (h : x > 0) :
  let blue := (2/3 : ℚ) * x
  let red := (1/3 : ℚ) * x
  let new_red := 3 * red
  let new_total := blue + new_red
  new_red / new_total = (3/5 : ℚ) := by
  sorry

end fraction_red_marbles_l1091_109124


namespace remainder_of_powers_l1091_109181

theorem remainder_of_powers (n1 n2 n3 : ℕ) : (9^6 + 8^8 + 7^9) % 7 = 2 :=
by
  sorry

end remainder_of_powers_l1091_109181


namespace average_calculation_l1091_109190

def average_two (a b : ℚ) : ℚ := (a + b) / 2
def average_three (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem average_calculation :
  average_three (average_three 2 4 1) (average_two 3 2) 5 = 59 / 18 :=
by
  sorry

end average_calculation_l1091_109190


namespace find_radius_of_sphere_l1091_109137

def radius_of_sphere_equal_to_cylinder_area (r : ℝ) (h : ℝ) (d : ℝ) : Prop :=
  (4 * Real.pi * r^2 = 2 * Real.pi * ((d / 2) * h))

theorem find_radius_of_sphere : ∃ r : ℝ, radius_of_sphere_equal_to_cylinder_area r 6 6 ∧ r = 3 :=
by
  sorry

end find_radius_of_sphere_l1091_109137


namespace charcoal_drawings_count_l1091_109101

-- Defining the conditions
def total_drawings : Nat := 25
def colored_pencil_drawings : Nat := 14
def blending_marker_drawings : Nat := 7

-- Defining the target value for charcoal drawings
def charcoal_drawings : Nat := total_drawings - (colored_pencil_drawings + blending_marker_drawings)

-- The theorem we need to prove
theorem charcoal_drawings_count : charcoal_drawings = 4 :=
by
  -- Lean proof goes here, but since we skip the proof, we'll just use 'sorry'
  sorry

end charcoal_drawings_count_l1091_109101


namespace factorable_iff_m_eq_2_l1091_109160

theorem factorable_iff_m_eq_2 (m : ℤ) :
  (∃ (A B C D : ℤ), (x y : ℤ) -> (x^2 + 2*x*y + 2*x + m*y + 2*m = (x + A*y + B) * (x + C*y + D))) ↔ m = 2 :=
sorry

end factorable_iff_m_eq_2_l1091_109160


namespace karl_present_salary_l1091_109111

def original_salary : ℝ := 20000
def reduction_percentage : ℝ := 0.10
def increase_percentage : ℝ := 0.10

theorem karl_present_salary :
  let reduced_salary := original_salary * (1 - reduction_percentage)
  let present_salary := reduced_salary * (1 + increase_percentage)
  present_salary = 19800 :=
by
  sorry

end karl_present_salary_l1091_109111


namespace maria_needs_more_cartons_l1091_109183

theorem maria_needs_more_cartons
  (total_needed : ℕ)
  (strawberries : ℕ)
  (blueberries : ℕ)
  (already_has : ℕ)
  (more_needed : ℕ)
  (h1 : total_needed = 21)
  (h2 : strawberries = 4)
  (h3 : blueberries = 8)
  (h4 : already_has = strawberries + blueberries)
  (h5 : more_needed = total_needed - already_has) :
  more_needed = 9 :=
by sorry

end maria_needs_more_cartons_l1091_109183


namespace total_cost_of_repair_l1091_109170

theorem total_cost_of_repair (hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) (H1 : hours = 2) (H2 : hourly_rate = 75) (H3 : part_cost = 150) :
  hours * hourly_rate + part_cost = 300 := 
by
  sorry

end total_cost_of_repair_l1091_109170


namespace car_round_trip_time_l1091_109195

theorem car_round_trip_time
  (d_AB : ℝ) (v_AB_downhill : ℝ) (v_BA_uphill : ℝ)
  (h_d_AB : d_AB = 75.6)
  (h_v_AB_downhill : v_AB_downhill = 33.6)
  (h_v_BA_uphill : v_BA_uphill = 25.2) :
  d_AB / v_AB_downhill + d_AB / v_BA_uphill = 5.25 := by
  sorry

end car_round_trip_time_l1091_109195


namespace eq_x_add_q_l1091_109147

theorem eq_x_add_q (x q : ℝ) (h1 : abs (x - 5) = q) (h2 : x > 5) : x + q = 5 + 2*q :=
by {
  sorry
}

end eq_x_add_q_l1091_109147


namespace geo_seq_second_term_l1091_109168

theorem geo_seq_second_term (b r : Real) 
  (h1 : 280 * r = b) 
  (h2 : b * r = 90 / 56) 
  (h3 : b > 0) 
  : b = 15 * Real.sqrt 2 := 
by 
  sorry

end geo_seq_second_term_l1091_109168


namespace exists_p_for_q_l1091_109126

noncomputable def sqrt_56 : ℝ := Real.sqrt 56
noncomputable def sqrt_58 : ℝ := Real.sqrt 58

theorem exists_p_for_q (q : ℕ) (hq : q > 0) (hq_ne_1 : q ≠ 1) (hq_ne_3 : q ≠ 3) :
  ∃ p : ℤ, sqrt_56 < (p : ℝ) / q ∧ (p : ℝ) / q < sqrt_58 :=
by sorry

end exists_p_for_q_l1091_109126


namespace henry_walks_distance_l1091_109112

noncomputable def gym_distance : ℝ := 3

noncomputable def walk_factor : ℝ := 2 / 3

noncomputable def c_limit_position : ℝ := 1.5

noncomputable def d_limit_position : ℝ := 2.5

theorem henry_walks_distance :
  abs (c_limit_position - d_limit_position) = 1 := by
  sorry

end henry_walks_distance_l1091_109112


namespace eval_expr_l1091_109148

theorem eval_expr : (3^3)^2 = 729 := 
by
  sorry

end eval_expr_l1091_109148


namespace common_factor_of_polynomial_l1091_109128

noncomputable def polynomial_common_factor (m : ℤ) : ℤ :=
  let polynomial := 2 * m^3 - 8 * m
  let common_factor := 2 * m
  common_factor  -- We're stating that the common factor is 2 * m

-- The theorem to verify that the common factor of each term in the polynomial is 2m
theorem common_factor_of_polynomial (m : ℤ) : 
  polynomial_common_factor m = 2 * m := by
  sorry

end common_factor_of_polynomial_l1091_109128


namespace line_equation_l1091_109151

theorem line_equation (k : ℝ) (x1 y1 : ℝ) (P : x1 = 1 ∧ y1 = -1) (angle_slope : k = Real.tan (135 * Real.pi / 180)) : 
  ∃ (a b : ℝ), a = -1 ∧ b = -1 ∧ (y1 = k * x1 + b) ∧ (y1 = a * x1 + b) :=
by
  sorry

end line_equation_l1091_109151


namespace number_913n_divisible_by_18_l1091_109103

theorem number_913n_divisible_by_18 (n : ℕ) (h1 : 9130 % 2 = 0) (h2 : (9 + 1 + 3 + n) % 9 = 0) : n = 8 :=
by
  sorry

end number_913n_divisible_by_18_l1091_109103


namespace juniors_score_l1091_109134

/-- Mathematical proof problem stated in Lean 4 -/
theorem juniors_score 
  (total_students : ℕ) 
  (juniors seniors : ℕ)
  (junior_score senior_avg total_avg : ℝ)
  (h_total_students : total_students > 0)
  (h_juniors : juniors = total_students / 10)
  (h_seniors : seniors = (total_students * 9) / 10)
  (h_total_avg : total_avg = 84)
  (h_senior_avg : senior_avg = 83)
  (h_junior_score_same : ∀ j : ℕ, j < juniors → ∃ s : ℝ, s = junior_score)
  :
  junior_score = 93 :=
by
  sorry

end juniors_score_l1091_109134


namespace money_left_over_l1091_109143

theorem money_left_over 
  (num_books : ℕ) 
  (price_per_book : ℝ) 
  (num_records : ℕ) 
  (price_per_record : ℝ) 
  (total_books : num_books = 200) 
  (book_price : price_per_book = 1.5) 
  (total_records : num_records = 75) 
  (record_price : price_per_record = 3) :
  (num_books * price_per_book - num_records * price_per_record) = 75 :=
by 
  -- calculation
  sorry

end money_left_over_l1091_109143


namespace find_natural_number_l1091_109153

theorem find_natural_number (n : ℕ) (k : ℤ) (h : 2^n + 3 = k^2) : n = 0 :=
sorry

end find_natural_number_l1091_109153


namespace find_unknown_number_l1091_109136

theorem find_unknown_number :
  ∃ (x : ℝ), (786 * x) / 30 = 1938.8 → x = 74 :=
by 
  sorry

end find_unknown_number_l1091_109136


namespace parabola_c_value_l1091_109118

theorem parabola_c_value (b c : ℝ) 
  (h1 : 6 = 2^2 + 2 * b + c) 
  (h2 : 20 = 4^2 + 4 * b + c) : 
  c = 0 :=
by {
  -- We state that we're skipping the proof
  sorry
}

end parabola_c_value_l1091_109118


namespace total_difference_in_cards_l1091_109157

theorem total_difference_in_cards (cards_chris : ℕ) (cards_charlie : ℕ) (cards_diana : ℕ) (cards_ethan : ℕ)
  (h_chris : cards_chris = 18)
  (h_charlie : cards_charlie = 32)
  (h_diana : cards_diana = 25)
  (h_ethan : cards_ethan = 40) :
  (cards_charlie - cards_chris) + (cards_diana - cards_chris) + (cards_ethan - cards_chris) = 43 := by
  sorry

end total_difference_in_cards_l1091_109157


namespace find_abc_l1091_109135

-- Definitions based on given conditions
variables (a b c : ℝ)
variable (h1 : a * b = 30 * (3 ^ (1/3)))
variable (h2 : a * c = 42 * (3 ^ (1/3)))
variable (h3 : b * c = 18 * (3 ^ (1/3)))

-- Formal statement of the proof problem
theorem find_abc : a * b * c = 90 * Real.sqrt 3 :=
by
  sorry

end find_abc_l1091_109135


namespace rectangle_area_l1091_109162

theorem rectangle_area (a b c d : ℝ) 
  (ha : a = 4) 
  (hb : b = 4) 
  (hc : c = 4) 
  (hd : d = 1) :
  ∃ E F G H : ℝ,
    (E = 0 ∧ F = 3 ∧ G = 4 ∧ H = 0) →
    (a + b + c + d) = 10 :=
by
  intros
  sorry

end rectangle_area_l1091_109162


namespace sum_of_repeating_decimals_l1091_109108

noncomputable def repeating_decimal_sum : ℚ :=
  let x := 1 / 3
  let y := 7 / 99
  let z := 8 / 999
  x + y + z

theorem sum_of_repeating_decimals :
  repeating_decimal_sum = 418 / 999 :=
by
  sorry

end sum_of_repeating_decimals_l1091_109108


namespace point_on_xoz_plane_l1091_109141

def Point := ℝ × ℝ × ℝ

def lies_on_plane_xoz (p : Point) : Prop :=
  p.2 = 0

theorem point_on_xoz_plane :
  lies_on_plane_xoz (-2, 0, 3) :=
by
  sorry

end point_on_xoz_plane_l1091_109141


namespace nancy_carrots_l1091_109165

-- Definitions based on the conditions
def initial_carrots := 12
def carrots_to_cook := 2
def new_carrot_seeds := 5
def growth_factor := 3
def kept_carrots := 10
def poor_quality_ratio := 3

-- Calculate new carrots grown from seeds
def new_carrots := new_carrot_seeds * growth_factor

-- Total carrots after new ones are added
def total_carrots := kept_carrots + new_carrots

-- Calculate poor quality carrots (integer part only)
def poor_quality_carrots := total_carrots / poor_quality_ratio

-- Calculate good quality carrots
def good_quality_carrots := total_carrots - poor_quality_carrots

-- Statement to prove
theorem nancy_carrots : good_quality_carrots = 17 :=
by
  sorry -- proof is not required

end nancy_carrots_l1091_109165


namespace inequality_div_two_l1091_109173

theorem inequality_div_two (a b : ℝ) (h : a > b) : (a / 2) > (b / 2) :=
sorry

end inequality_div_two_l1091_109173


namespace p_suff_but_not_nec_q_l1091_109175

variable (p q : Prop)

-- Given conditions: ¬p is a necessary but not sufficient condition for ¬q.
def neg_p_nec_but_not_suff_neg_q : Prop :=
  (¬q → ¬p) ∧ ¬(¬p → ¬q)

-- Concluding statement: p is a sufficient but not necessary condition for q.
theorem p_suff_but_not_nec_q 
  (h : neg_p_nec_but_not_suff_neg_q p q) : (p → q) ∧ ¬(q → p) := 
sorry

end p_suff_but_not_nec_q_l1091_109175


namespace largest_z_l1091_109122

theorem largest_z (x y z : ℝ) 
  (h1 : x + y + z = 5)  
  (h2 : x * y + y * z + x * z = 3) 
  : z ≤ 13 / 3 := sorry

end largest_z_l1091_109122


namespace no_positive_integers_exist_l1091_109163

theorem no_positive_integers_exist 
  (a b c d : ℕ) 
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (c_pos : 0 < c) 
  (d_pos : 0 < d)
  (h₁ : a * b = c * d)
  (p : ℕ) 
  (hp : Nat.Prime p)
  (h₂ : a + b + c + d = p) : 
  False := 
by
  sorry

end no_positive_integers_exist_l1091_109163


namespace average_playtime_l1091_109117

-- Definitions based on conditions
def h_w := 2 -- Hours played on Wednesday
def h_t := 2 -- Hours played on Thursday
def h_f := h_w + 3 -- Hours played on Friday (3 hours more than Wednesday)

-- Statement to prove
theorem average_playtime :
  (h_w + h_t + h_f) / 3 = 3 := by
  sorry

end average_playtime_l1091_109117


namespace angle_bisector_slope_l1091_109142

-- Definitions of the conditions
def line1_slope := 2
def line2_slope := 4

-- The proof statement: Prove that the slope of the angle bisector is -12/7
theorem angle_bisector_slope : (line1_slope + line2_slope + Real.sqrt (line1_slope^2 + line2_slope^2 + 2 * line1_slope * line2_slope)) / 
                               (1 - line1_slope * line2_slope) = -12/7 :=
by
  sorry

end angle_bisector_slope_l1091_109142


namespace car_distribution_l1091_109104

theorem car_distribution :
  let total_cars := 5650000
  let first_supplier := 1000000
  let second_supplier := first_supplier + 500000
  let third_supplier := first_supplier + second_supplier
  let fourth_supplier := (total_cars - (first_supplier + second_supplier + third_supplier)) / 2
  let fifth_supplier := fourth_supplier
  fourth_supplier = 325000 ∧ fifth_supplier = 325000 := 
by 
  sorry

end car_distribution_l1091_109104


namespace sufficient_but_not_necessary_condition_l1091_109138

theorem sufficient_but_not_necessary_condition (a : ℝ) (h : ∀ x : ℝ, x > a → x > 2 ∧ ¬(x > 2 → x > a)) : a > 2 :=
sorry

end sufficient_but_not_necessary_condition_l1091_109138


namespace find_abc_l1091_109164

theorem find_abc (a b c : ℕ) (k : ℕ) 
  (h1 : a = 2 * k) 
  (h2 : b = 3 * k) 
  (h3 : c = 4 * k) 
  (h4 : k ≠ 0)
  (h5 : 2 * a - b + c = 10) : 
  a = 4 ∧ b = 6 ∧ c = 8 :=
sorry

end find_abc_l1091_109164


namespace arithmetic_sequence_a8_l1091_109109

/-- In an arithmetic sequence with the given sum of terms, prove the value of a_8 is 14. -/
theorem arithmetic_sequence_a8 (a : ℕ → ℕ) (d : ℕ) (h1 : ∀ (n : ℕ), a (n+1) = a n + d)
    (h2 : a 2 + a 7 + a 8 + a 9 + a 14 = 70) : a 8 = 14 :=
  sorry

end arithmetic_sequence_a8_l1091_109109


namespace fraction_equivalence_l1091_109125

theorem fraction_equivalence : 
    (1 / 4 - 1 / 6) / (1 / 3 - 1 / 4) = 1 := by sorry

end fraction_equivalence_l1091_109125


namespace ascetic_height_l1091_109115

theorem ascetic_height (h m : ℝ) (x : ℝ) (hx : h * (m + 1) = (x + h)^2 + (m * h)^2) : x = h * m / (m + 2) :=
sorry

end ascetic_height_l1091_109115


namespace one_minus_repeating_three_l1091_109121

theorem one_minus_repeating_three : ∀ b : ℚ, b = 1 / 3 → 1 - b = 2 / 3 :=
by
  intro b hb
  rw [hb]
  norm_num

end one_minus_repeating_three_l1091_109121


namespace hexagon_ratio_l1091_109192

theorem hexagon_ratio 
  (hex_area : ℝ)
  (rs_bisects_area : ∃ (a b : ℝ), a + b = hex_area / 2 ∧ ∃ (x r s : ℝ), x = 4 ∧ r * s = (hex_area / 2 - 1))
  : ∀ (XR RS : ℝ), XR = RS → XR / RS = 1 :=
by
  sorry

end hexagon_ratio_l1091_109192


namespace expected_number_of_adjacent_black_pairs_l1091_109161

theorem expected_number_of_adjacent_black_pairs :
  let total_cards := 52
  let black_cards := 26
  let adjacent_probability := (black_cards - 1) / (total_cards - 1)
  let expected_per_black_card := black_cards * adjacent_probability / total_cards
  let expected_total := black_cards * adjacent_probability
  expected_total = 650 / 51 := 
by
  let total_cards := 52
  let black_cards := 26
  let adjacent_probability := (black_cards - 1) / (total_cards - 1)
  let expected_total := black_cards * adjacent_probability
  sorry

end expected_number_of_adjacent_black_pairs_l1091_109161


namespace joan_lost_balloons_l1091_109132

theorem joan_lost_balloons :
  let initial_balloons := 9
  let current_balloons := 7
  let balloons_lost := initial_balloons - current_balloons
  balloons_lost = 2 :=
by
  sorry

end joan_lost_balloons_l1091_109132


namespace average_height_of_60_students_l1091_109152

theorem average_height_of_60_students :
  (35 * 22 + 25 * 18) / 60 = 20.33 := 
sorry

end average_height_of_60_students_l1091_109152


namespace intersection_of_sets_l1091_109176

theorem intersection_of_sets :
  let A := {x : ℝ | -2 < x ∧ x < 1}
  let B := {x : ℝ | x < 0 ∨ x > 3}
  ∀ x, (x ∈ A ∧ x ∈ B) ↔ (-2 < x ∧ x < 0) :=
by
  let A := {x : ℝ | -2 < x ∧ x < 1}
  let B := {x : ℝ | x < 0 ∨ x > 3}
  intro x
  sorry

end intersection_of_sets_l1091_109176


namespace solve_for_y_l1091_109188


theorem solve_for_y (b y : ℝ) (h : b ≠ 0) :
    Matrix.det ![
        ![y + b, y, y],
        ![y, y + b, y],
        ![y, y, y + b]] = 0 → y = -b := by
  sorry

end solve_for_y_l1091_109188


namespace fish_remaining_l1091_109177

def fish_caught_per_hour := 7
def hours_fished := 9
def fish_lost := 15

theorem fish_remaining : 
  (fish_caught_per_hour * hours_fished - fish_lost) = 48 :=
by
  sorry

end fish_remaining_l1091_109177
