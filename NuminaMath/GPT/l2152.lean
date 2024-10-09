import Mathlib

namespace dagger_simplified_l2152_215286

def dagger (m n p q : ℚ) : ℚ := (m^2) * p * (q / n)

theorem dagger_simplified :
  dagger (5:ℚ) (9:ℚ) (4:ℚ) (6:ℚ) = (200:ℚ) / (3:ℚ) :=
by
  sorry

end dagger_simplified_l2152_215286


namespace find_b_l2152_215287

theorem find_b {a b : ℝ} (h₁ : 2 * 2 + b = 1 - 2 * a) (h₂ : -2 * 2 + b = -15 + 2 * a) : 
  b = -7 := sorry

end find_b_l2152_215287


namespace fraction_grades_C_l2152_215291

def fraction_grades_A (students : ℕ) : ℕ := (1 / 5) * students
def fraction_grades_B (students : ℕ) : ℕ := (1 / 4) * students
def num_grades_D : ℕ := 5
def total_students : ℕ := 100

theorem fraction_grades_C :
  (total_students - (fraction_grades_A total_students + fraction_grades_B total_students + num_grades_D)) / total_students = 1 / 2 :=
by
  sorry

end fraction_grades_C_l2152_215291


namespace solution_for_x_l2152_215212

theorem solution_for_x (x : ℝ) : x^2 - x - 1 = (x + 1)^0 → x = 2 :=
by
  intro h
  have h_simp : x^2 - x - 1 = 1 := by simp [h]
  sorry

end solution_for_x_l2152_215212


namespace cohen_saw_1300_fish_eater_birds_l2152_215294

theorem cohen_saw_1300_fish_eater_birds :
  let day1 := 300
  let day2 := 2 * day1
  let day3 := day2 - 200
  day1 + day2 + day3 = 1300 :=
by
  sorry

end cohen_saw_1300_fish_eater_birds_l2152_215294


namespace not_right_triangle_l2152_215235

theorem not_right_triangle (a b c : ℝ) (h : a / b = 1 / 2 ∧ b / c = 2 / 3) :
  ¬(a^2 = b^2 + c^2) :=
by sorry

end not_right_triangle_l2152_215235


namespace euler_conjecture_disproof_l2152_215225

theorem euler_conjecture_disproof :
    ∃ (n : ℕ), 133^4 + 110^4 + 56^4 = n^4 ∧ n = 143 :=
by {
  use 143,
  sorry
}

end euler_conjecture_disproof_l2152_215225


namespace candy_received_l2152_215292

theorem candy_received (pieces_eaten : ℕ) (piles : ℕ) (pieces_per_pile : ℕ) 
  (h_eaten : pieces_eaten = 12) (h_piles : piles = 4) (h_pieces_per_pile : pieces_per_pile = 5) :
  pieces_eaten + piles * pieces_per_pile = 32 := 
by
  sorry

end candy_received_l2152_215292


namespace geometric_sequence_sum_l2152_215234

theorem geometric_sequence_sum (a_n : ℕ → ℝ) (q : ℝ) (h1 : q = 2) (h2 : a_n 1 + a_n 3 = 5) :
  a_n 3 + a_n 5 = 20 :=
by
  -- The proof would go here, but it is not required for this task.
  sorry

end geometric_sequence_sum_l2152_215234


namespace line_quadrant_relationship_l2152_215253

theorem line_quadrant_relationship
  (a b c : ℝ)
  (passes_first_second_fourth : ∀ x y : ℝ, (a * x + b * y + c = 0) → ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0))) :
  (a * b > 0) ∧ (b * c < 0) :=
sorry

end line_quadrant_relationship_l2152_215253


namespace elodie_rats_l2152_215213

-- Define the problem conditions as hypotheses
def E (H : ℕ) : ℕ := H + 10
def K (H : ℕ) : ℕ := 3 * (E H + H)

-- The goal is to prove E = 30 given the conditions
theorem elodie_rats (H : ℕ) (h1 : E (H := H) + H + K (H := H) = 200) : E H = 30 :=
by
  sorry

end elodie_rats_l2152_215213


namespace sum_of_star_tips_l2152_215283

theorem sum_of_star_tips :
  let n := 9
  let alpha := 80  -- in degrees
  let total := n * alpha
  total = 720 := by sorry

end sum_of_star_tips_l2152_215283


namespace least_number_divisible_remainder_l2152_215289

theorem least_number_divisible_remainder (n : ℕ) (h1 : n % 34 = 4) (h2 : n % 5 = 4) : n = 174 := 
sorry

end least_number_divisible_remainder_l2152_215289


namespace number_of_intersections_l2152_215243

def line₁ (x y : ℝ) := 2 * x - 3 * y + 6 = 0
def line₂ (x y : ℝ) := 5 * x + 2 * y - 10 = 0
def line₃ (x y : ℝ) := x - 2 * y + 1 = 0
def line₄ (x y : ℝ) := 3 * x - 4 * y + 8 = 0

theorem number_of_intersections : 
  ∃! (p₁ p₂ p₃ : ℝ × ℝ),
    (line₁ p₁.1 p₁.2 ∨ line₂ p₁.1 p₁.2) ∧ (line₃ p₁.1 p₁.2 ∨ line₄ p₁.1 p₁.2) ∧
    (line₁ p₂.1 p₂.2 ∨ line₂ p₂.1 p₂.2) ∧ (line₃ p₂.1 p₂.2 ∨ line₄ p₂.1 p₂.2) ∧
    (line₁ p₃.1 p₃.2 ∨ line₂ p₃.1 p₃.2) ∧ (line₃ p₃.1 p₃.2 ∨ line₄ p₃.1 p₃.2) ∧
    p₁ ≠ p₂ ∧ p₂ ≠ p₃ ∧ p₁ ≠ p₃ := 
sorry

end number_of_intersections_l2152_215243


namespace laboratory_painting_area_laboratory_paint_needed_l2152_215219

section
variable (l w h excluded_area : ℝ)
variable (paint_per_sqm : ℝ)

def painting_area (l w h excluded_area : ℝ) : ℝ :=
  let total_area := (l * w + w * h + h * l) * 2 - (l * w)
  total_area - excluded_area

def paint_needed (painting_area paint_per_sqm : ℝ) : ℝ :=
  painting_area * paint_per_sqm

theorem laboratory_painting_area :
  painting_area 12 8 6 28.4 = 307.6 :=
by
  simp [painting_area, *]
  norm_num

theorem laboratory_paint_needed :
  paint_needed 307.6 0.2 = 61.52 :=
by
  simp [paint_needed, *]
  norm_num

end

end laboratory_painting_area_laboratory_paint_needed_l2152_215219


namespace Richard_Orlando_ratio_l2152_215276

def Jenny_cards : ℕ := 6
def Orlando_more_cards : ℕ := 2
def Total_cards : ℕ := 38

theorem Richard_Orlando_ratio :
  let Orlando_cards := Jenny_cards + Orlando_more_cards
  let Richard_cards := Total_cards - (Jenny_cards + Orlando_cards)
  let ratio := Richard_cards / Orlando_cards
  ratio = 3 :=
by
  sorry

end Richard_Orlando_ratio_l2152_215276


namespace translation_preserves_parallel_and_equal_length_l2152_215262

theorem translation_preserves_parallel_and_equal_length
    (A B C D : ℝ)
    (after_translation : (C - A) = (D - B))
    (connecting_parallel : C - A = D - B) :
    (C - A = D - B) ∧ (C - A = D - B) :=
by
  sorry

end translation_preserves_parallel_and_equal_length_l2152_215262


namespace simplify_sqrt_l2152_215205

-- Define the domain and main trigonometric properties
open Real

noncomputable def simplify_expression (x : ℝ) : ℝ :=
  sqrt (1 - 2 * sin x * cos x)

-- Define the main theorem with given conditions
theorem simplify_sqrt {x : ℝ} (h1 : (5 / 4) * π < x) (h2 : x < (3 / 2) * π) (h3 : cos x > sin x) :
  simplify_expression x = cos x - sin x :=
  sorry

end simplify_sqrt_l2152_215205


namespace inequality_correct_l2152_215272

theorem inequality_correct (a b : ℝ) (h1 : a > b) (h2 : b > 0) : (1 / a) < (1 / b) :=
sorry

end inequality_correct_l2152_215272


namespace bianca_picture_books_shelves_l2152_215275

theorem bianca_picture_books_shelves (total_shelves : ℕ) (books_per_shelf : ℕ) (mystery_shelves : ℕ) (total_books : ℕ) :
  books_per_shelf = 8 →
  mystery_shelves = 5 →
  total_books = 72 →
  total_shelves = (total_books - (mystery_shelves * books_per_shelf)) / books_per_shelf →
  total_shelves = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end bianca_picture_books_shelves_l2152_215275


namespace frosting_sugar_l2152_215265

-- Define the conditions as constants
def total_sugar : ℝ := 0.8
def cake_sugar : ℝ := 0.2

-- The theorem stating that the sugar required for the frosting is 0.6 cups
theorem frosting_sugar : total_sugar - cake_sugar = 0.6 := by
  sorry

end frosting_sugar_l2152_215265


namespace find_integer_pairs_l2152_215260

theorem find_integer_pairs (x y : ℤ) (h_xy : x ≤ y) (h_eq : (1 : ℚ)/x + (1 : ℚ)/y = 1/4) :
  (x, y) = (5, 20) ∨ (x, y) = (6, 12) ∨ (x, y) = (8, 8) ∨ (x, y) = (-4, 2) ∨ (x, y) = (-12, 3) :=
sorry

end find_integer_pairs_l2152_215260


namespace robis_savings_in_january_l2152_215210

theorem robis_savings_in_january (x : ℕ) (h: (x + (x + 4) + (x + 8) + (x + 12) + (x + 16) + (x + 20) = 126)) : x = 11 := 
by {
  -- By simplification, the lean equivalent proof would include combining like
  -- terms and solving the resulting equation. For now, we'll use sorry.
  sorry
}

end robis_savings_in_january_l2152_215210


namespace problem_l2152_215246

noncomputable def f(x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x + 1
noncomputable def f_prime(x : ℝ) (a b : ℝ) : ℝ := 3 * a * x^2 + b

theorem problem (a b : ℝ) 
  (h₁ : f_prime 1 a b = 4) 
  (h₂ : f 1 a b = 3) : 
  a + b = 2 :=
sorry

end problem_l2152_215246


namespace diff_set_Q_minus_P_l2152_215211

def P (x : ℝ) : Prop := 1 - (2 / x) < 0
def Q (x : ℝ) : Prop := |x - 2| < 1
def diff_set (P Q : ℝ → Prop) (x : ℝ) : Prop := Q x ∧ ¬ P x

theorem diff_set_Q_minus_P :
  ∀ x : ℝ, diff_set Q P x ↔ (2 ≤ x ∧ x < 3) :=
by
  sorry

end diff_set_Q_minus_P_l2152_215211


namespace larger_number_is_72_l2152_215278

theorem larger_number_is_72 (a b : ℕ) (h1 : 5 * b = 6 * a) (h2 : b - a = 12) : b = 72 :=
by
  sorry

end larger_number_is_72_l2152_215278


namespace proof_problem_l2152_215241

theorem proof_problem (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a^2 / (b - c)^2 + b^2 / (c - a)^2 + c^2 / (a - b)^2 = 0 :=
sorry

end proof_problem_l2152_215241


namespace shape_described_by_constant_phi_is_cone_l2152_215200

-- Definition of spherical coordinates
-- (ρ, θ, φ) where ρ is the radial distance,
-- θ is the azimuthal angle, and φ is the polar angle.
structure SphericalCoordinates :=
  (ρ : ℝ)
  (θ : ℝ)
  (φ : ℝ)

-- The condition that φ is equal to a constant d
def satisfies_condition (p : SphericalCoordinates) (d : ℝ) : Prop :=
  p.φ = d

-- The main theorem to prove
theorem shape_described_by_constant_phi_is_cone (d : ℝ) :
  ∃ (S : Set SphericalCoordinates), (∀ p ∈ S, satisfies_condition p d) ∧
  (∀ p, satisfies_condition p d → ∃ ρ θ, p = ⟨ρ, θ, d⟩) ∧
  (∀ ρ θ, ρ > 0 → θ ∈ [0, 2 * Real.pi] → SphericalCoordinates.mk ρ θ d ∈ S) :=
sorry

end shape_described_by_constant_phi_is_cone_l2152_215200


namespace x_is_perfect_square_l2152_215299

theorem x_is_perfect_square (x y : ℕ) (hx_pos : x > 0) (hy_pos : y > 0) (hdiv : 2 * x * y ∣ x^2 + y^2 - x) : ∃ (n : ℕ), x = n^2 :=
by
  sorry

end x_is_perfect_square_l2152_215299


namespace only_function_l2152_215280

def divides (a b : ℕ) : Prop := ∃ k, b = k * a

def satisfies_condition (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, divides (f m + f n) (m + n)

theorem only_function (f : ℕ → ℕ) (h : satisfies_condition f) : f = id :=
by
  -- Proof goes here.
  sorry

end only_function_l2152_215280


namespace total_votes_cast_l2152_215236

theorem total_votes_cast (F A T : ℕ) (h1 : F = A + 70) (h2 : A = 2 * T / 5) (h3 : T = F + A) : T = 350 :=
by
  sorry

end total_votes_cast_l2152_215236


namespace selection_schemes_correct_l2152_215242

-- Define the problem parameters
def number_of_selection_schemes (persons : ℕ) (cities : ℕ) (persons_cannot_visit : ℕ) : ℕ :=
  let choices_for_paris := persons - persons_cannot_visit
  let remaining_people := persons - 1
  choices_for_paris * remaining_people * (remaining_people - 1) * (remaining_people - 2)

-- Define the example constants
def total_people : ℕ := 6
def total_cities : ℕ := 4
def cannot_visit_paris : ℕ := 2

-- The statement to be proved
theorem selection_schemes_correct : 
  number_of_selection_schemes total_people total_cities cannot_visit_paris = 240 := by
  sorry

end selection_schemes_correct_l2152_215242


namespace sqrt_sum_equality_l2152_215237

theorem sqrt_sum_equality :
  (Real.sqrt (9 - 6 * Real.sqrt 2) + Real.sqrt (9 + 6 * Real.sqrt 2) = 2 * Real.sqrt 6) :=
by
  sorry

end sqrt_sum_equality_l2152_215237


namespace find_number_l2152_215244

theorem find_number (x : ℤ) (h : 5 * (x - 12) = 40) : x = 20 := 
by
  sorry

end find_number_l2152_215244


namespace composite_prop_true_l2152_215271

def p : Prop := ∀ (x : ℝ), x > 0 → x + (1/(2*x)) ≥ 1

def q : Prop := ∀ (x : ℝ), x > 1 → (x^2 + 2*x - 3 > 0)

theorem composite_prop_true : p ∨ q :=
by
  sorry

end composite_prop_true_l2152_215271


namespace sequence_geometric_sum_bn_l2152_215269

theorem sequence_geometric (a : ℕ → ℕ) (h_recurrence : ∀ n ≥ 2, (a n)^2 = (a (n - 1)) * (a (n + 1)))
  (h_init1 : a 2 + 2 * a 1 = 4) (h_init2 : (a 3)^2 = a 5) : 
  (∀ n, a n = 2^n) :=
by sorry

theorem sum_bn (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ)
  (h_recurrence : ∀ n ≥ 2, (a n)^2 = (a (n - 1)) * (a (n + 1)))
  (h_init1 : a 2 + 2 * a 1 = 4) (h_init2 : (a 3)^2 = a 5) 
  (h_gen : ∀ n, a n = 2^n) (h_bn : ∀ n, b n = n * a n) :
  (∀ n, S n = (n-1) * 2^(n+1) + 2) :=
by sorry

end sequence_geometric_sum_bn_l2152_215269


namespace find_integers_a_b_c_l2152_215279

theorem find_integers_a_b_c :
  ∃ a b c : ℤ, ((x - a) * (x - 12) + 1 = (x + b) * (x + c)) ∧ 
  ((b + 12) * (c + 12) = 1 → ((b = -11 ∧ c = -11) → a = 10) ∧ 
  ((b = -13 ∧ c = -13) → a = 14)) :=
by
  sorry

end find_integers_a_b_c_l2152_215279


namespace residents_ticket_price_l2152_215214

theorem residents_ticket_price
  (total_attendees : ℕ)
  (resident_count : ℕ)
  (non_resident_price : ℝ)
  (total_revenue : ℝ)
  (R : ℝ)
  (h1 : total_attendees = 586)
  (h2 : resident_count = 219)
  (h3 : non_resident_price = 17.95)
  (h4 : total_revenue = 9423.70)
  (total_residents_pay : ℝ := resident_count * R)
  (total_non_residents_pay : ℝ := (total_attendees - resident_count) * non_resident_price)
  (h5 : total_revenue = total_residents_pay + total_non_residents_pay) :
  R = 12.95 := by
  sorry

end residents_ticket_price_l2152_215214


namespace not_possible_one_lies_other_not_l2152_215264

-- Variable definitions: Jean is lying (J), Pierre is lying (P)
variable (J P : Prop)

-- Conditions from the problem
def Jean_statement : Prop := P → J
def Pierre_statement : Prop := P → J

-- Theorem statement
theorem not_possible_one_lies_other_not (h1 : Jean_statement J P) (h2 : Pierre_statement J P) : ¬ ((J ∨ ¬ J) ∧ (P ∨ ¬ P) ∧ ((J ∧ ¬ P) ∨ (¬ J ∧ P))) :=
by
  sorry

end not_possible_one_lies_other_not_l2152_215264


namespace logarithmic_inequality_l2152_215227

theorem logarithmic_inequality (a : ℝ) (h : a > 1) : 
  1 / 2 + 1 / Real.log a ≥ 1 := 
sorry

end logarithmic_inequality_l2152_215227


namespace cost_to_feed_turtles_l2152_215270

theorem cost_to_feed_turtles :
  (∀ (weight : ℚ), weight > 0 → (food_per_half_pound_ounces = 1) →
    (total_weight_pounds = 30) →
    (jar_contents_ounces = 15 ∧ jar_cost = 2) →
    (total_cost_dollars = 8)) :=
by
  sorry

variables
  (food_per_half_pound_ounces : ℚ := 1)
  (total_weight_pounds : ℚ := 30)
  (jar_contents_ounces : ℚ := 15)
  (jar_cost : ℚ := 2)
  (total_cost_dollars : ℚ := 8)

-- Assuming all variables needed to state the theorem exist and are meaningful
/-!
  Given:
  - Each turtle needs 1 ounce of food per 1/2 pound of body weight
  - Total turtles' weight is 30 pounds
  - Each jar of food contains 15 ounces and costs $2

  Prove:
  - The total cost to feed Sylvie's turtles is $8
-/

end cost_to_feed_turtles_l2152_215270


namespace n_in_S_implies_n2_in_S_l2152_215233

def S (n : ℕ) : Prop :=
  ∃ (a b c d e f : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧
  a ≥ b ∧ c ≥ d ∧ e ≥ f ∧
  n - 1 = a^2 + b^2 ∧ n = c^2 + d^2 ∧ n + 1 = e^2 + f^2

theorem n_in_S_implies_n2_in_S (n : ℕ) (h : S n) : S (n^2) :=
  sorry

end n_in_S_implies_n2_in_S_l2152_215233


namespace probability_to_buy_ticket_l2152_215290

def p : ℝ := 0.1
def q : ℝ := 0.9
def initial_money : ℝ := 20
def target_money : ℝ := 45
def ticket_cost : ℝ := 10
def prize : ℝ := 30

noncomputable def equation_lhs : ℝ := p^2 * (1 + 2 * q)
noncomputable def equation_rhs : ℝ := 1 - 2 * p * q^2

noncomputable def x2 : ℝ := equation_lhs / equation_rhs

theorem probability_to_buy_ticket : x2 = 0.033 := sorry

end probability_to_buy_ticket_l2152_215290


namespace euler_disproven_conjecture_solution_l2152_215201

theorem euler_disproven_conjecture_solution : 
  ∃ (n : ℕ), n^5 = 133^5 + 110^5 + 84^5 + 27^5 ∧ n = 144 :=
by
  use 144
  have h : 144^5 = 133^5 + 110^5 + 84^5 + 27^5 := sorry
  exact ⟨h, rfl⟩

end euler_disproven_conjecture_solution_l2152_215201


namespace find_g2_l2152_215229

variable {R : Type*} [Nonempty R] [Field R]

-- Define the function g
def g (x : R) : R := sorry

-- Given conditions
axiom condition1 : ∀ x y : R, x * g y = 2 * y * g x
axiom condition2 : g 10 = 5

-- The statement to be proved
theorem find_g2 : g 2 = 2 :=
by
  sorry

end find_g2_l2152_215229


namespace sum_of_solutions_eq_minus_2_l2152_215281

-- Defining the equation and the goal
theorem sum_of_solutions_eq_minus_2 (x1 x2 : ℝ) (floor : ℝ → ℤ) (h1 : floor (3 * x1 + 1) = 2 * x1 - 1 / 2)
(h2 : floor (3 * x2 + 1) = 2 * x2 - 1 / 2) :
  x1 + x2 = -2 :=
sorry

end sum_of_solutions_eq_minus_2_l2152_215281


namespace possible_values_of_product_l2152_215261

theorem possible_values_of_product 
  (P_A P_B P_C P_D P_E : ℕ)
  (H1 : P_A = P_B + P_C + P_D + P_E)
  (H2 : ∃ n1 n2 n3 n4, 
          ((P_B = n1 * (n1 + 1)) ∨ (P_B = n2 * (n2 + 1) * (n2 + 2)) ∨ 
           (P_B = n3 * (n3 + 1) * (n3 + 2) * (n3 + 3)) ∨ (P_B = n4 * (n4 + 1) * (n4 + 2) * (n4 + 3) * (n4 + 4))) ∧
          ∃ m1 m2 m3 m4, 
          ((P_C = m1 * (m1 + 1)) ∨ (P_C = m2 * (m2 + 1) * (m2 + 2)) ∨ 
           (P_C = m3 * (m3 + 1) * (m3 + 2) * (m3 + 3)) ∨ (P_C = m4 * (m4 + 1) * (m4 + 2) * (m4 + 3) * (m4 + 4))) ∧
          ∃ o1 o2 o3 o4, 
          ((P_D = o1 * (o1 + 1)) ∨ (P_D = o2 * (o2 + 1) * (o2 + 2)) ∨ 
           (P_D = o3 * (o3 + 1) * (o3 + 2) * (o3 + 3)) ∨ (P_D = o4 * (o4 + 1) * (o4 + 2) * (o4 + 3) * (o4 + 4))) ∧
          ∃ p1 p2 p3 p4, 
          ((P_E = p1 * (p1 + 1)) ∨ (P_E = p2 * (p2 + 1) * (p2 + 2)) ∨ 
           (P_E = p3 * (p3 + 1) * (p3 + 2) * (p3 + 3)) ∨ (P_E = p4 * (p4 + 1) * (p4 + 2) * (p4 + 3) * (p4 + 4))) ∧ 
          ∃ q1 q2 q3 q4, 
          ((P_A = q1 * (q1 + 1)) ∨ (P_A = q2 * (q2 + 1) * (q2 + 2)) ∨ 
           (P_A = q3 * (q3 + 1) * (q3 + 2) * (q3 + 3)) ∨ (P_A = q4 * (q4 + 1) * (q4 + 2) * (q4 + 3) * (q4 + 4)))) :
  P_A = 6 ∨ P_A = 24 :=
by sorry

end possible_values_of_product_l2152_215261


namespace total_ways_to_buy_l2152_215249

-- Definitions:
def h : ℕ := 9  -- number of headphones
def m : ℕ := 13 -- number of mice
def k : ℕ := 5  -- number of keyboards
def s_km : ℕ := 4 -- number of sets of keyboard and mouse
def s_hm : ℕ := 5 -- number of sets of headphones and mouse

-- Theorem stating the number of ways to buy three items is 646.
theorem total_ways_to_buy : (s_km * h) + (s_hm * k) + (h * m * k) = 646 := by
  -- Placeholder proof using sorry.
  sorry

end total_ways_to_buy_l2152_215249


namespace min_AB_dot_CD_l2152_215215

theorem min_AB_dot_CD (a b : ℝ) (h1 : 0 <= (a - 1)^2 + (b - 3 / 2)^2 - 13/4) :
  ∃ (a b : ℝ), (a-1)^2 + (b - 3 / 2)^2 - 13/4 = 0 :=
by
  sorry

end min_AB_dot_CD_l2152_215215


namespace multiplication_more_than_subtraction_l2152_215202

def x : ℕ := 22

def multiplication_result : ℕ := 3 * x
def subtraction_result : ℕ := 62 - x
def difference : ℕ := multiplication_result - subtraction_result

theorem multiplication_more_than_subtraction : difference = 26 :=
by
  sorry

end multiplication_more_than_subtraction_l2152_215202


namespace completing_the_square_l2152_215256

theorem completing_the_square (x m n : ℝ) 
  (h : x^2 - 6 * x = 1) 
  (hm : (x - m)^2 = n) : 
  m + n = 13 :=
sorry

end completing_the_square_l2152_215256


namespace fifth_plot_difference_l2152_215239

-- Define the dimensions of the plots
def plot_width (n : Nat) : Nat := 3 + 2 * (n - 1)
def plot_length (n : Nat) : Nat := 4 + 3 * (n - 1)

-- Define the number of tiles in a plot
def tiles_in_plot (n : Nat) : Nat := plot_width n * plot_length n

-- The main theorem to prove the required difference
theorem fifth_plot_difference :
  tiles_in_plot 5 - tiles_in_plot 4 = 59 := sorry

end fifth_plot_difference_l2152_215239


namespace aeroplane_distance_l2152_215226

theorem aeroplane_distance
  (speed : ℝ) (time : ℝ) (distance : ℝ)
  (h1 : speed = 590)
  (h2 : time = 8)
  (h3 : distance = speed * time) :
  distance = 4720 :=
by {
  -- The proof will contain the steps to show that distance = 4720
  sorry
}

end aeroplane_distance_l2152_215226


namespace fraction_of_recipe_l2152_215230

theorem fraction_of_recipe 
  (recipe_sugar recipe_milk recipe_flour : ℚ)
  (have_sugar have_milk have_flour : ℚ)
  (h1 : recipe_sugar = 3/4) (h2 : recipe_milk = 2/3) (h3 : recipe_flour = 3/8)
  (h4 : have_sugar = 2/4) (h5 : have_milk = 1/2) (h6 : have_flour = 1/4) : 
  (min ((have_sugar / recipe_sugar)) (min ((have_milk / recipe_milk)) (have_flour / recipe_flour)) = 2/3) := 
by sorry

end fraction_of_recipe_l2152_215230


namespace inequality_of_abc_l2152_215204

variable {a b c : ℝ}

theorem inequality_of_abc 
    (h : 0 < a ∧ 0 < b ∧ 0 < c)
    (h₁ : abc * (a + b + c) = ab + bc + ca) :
    5 * (a + b + c) ≥ 7 + 8 * abc :=
sorry

end inequality_of_abc_l2152_215204


namespace unique_n_divides_2_pow_n_minus_1_l2152_215203

theorem unique_n_divides_2_pow_n_minus_1 (n : ℕ) (h : n ∣ 2^n - 1) : n = 1 :=
sorry

end unique_n_divides_2_pow_n_minus_1_l2152_215203


namespace meat_pie_cost_l2152_215232

variable (total_farthings : ℕ) (farthings_per_pfennig : ℕ) (remaining_pfennigs : ℕ)

def total_pfennigs (total_farthings farthings_per_pfennig : ℕ) : ℕ :=
  total_farthings / farthings_per_pfennig

def pie_cost (total_farthings farthings_per_pfennig remaining_pfennigs : ℕ) : ℕ :=
  total_pfennigs total_farthings farthings_per_pfennig - remaining_pfennigs

theorem meat_pie_cost
  (h1 : total_farthings = 54)
  (h2 : farthings_per_pfennig = 6)
  (h3 : remaining_pfennigs = 7) :
  pie_cost total_farthings farthings_per_pfennig remaining_pfennigs = 2 :=
by
  sorry

end meat_pie_cost_l2152_215232


namespace distance_between_chords_l2152_215220

theorem distance_between_chords (R : ℝ) (AB CD : ℝ) (d : ℝ) : 
  R = 25 → AB = 14 → CD = 40 → (d = 39 ∨ d = 9) :=
by intros; sorry

end distance_between_chords_l2152_215220


namespace probability_blue_point_l2152_215209

-- Definitions of the random points
def is_random_point (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 2

-- Definition of the condition for the probability problem
def condition (x y : ℝ) : Prop :=
  x < y ∧ y < 3 * x

-- Statement of the theorem
theorem probability_blue_point (x y : ℝ) (h1 : is_random_point x) (h2 : is_random_point y) :
  ∃ p : ℝ, (p = 1 / 3) ∧ (∃ (hx : x < y) (hy : y < 3 * x), x ≤ 2 ∧ 0 ≤ x ∧ y ≤ 2 ∧ 0 ≤ y) :=
by
  sorry

end probability_blue_point_l2152_215209


namespace symmetric_about_line_l2152_215222

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x - 2)
noncomputable def g (x a : ℝ) : ℝ := f (x + a)

theorem symmetric_about_line (a : ℝ) : (∀ x, g x a = x + 1) ↔ a = 0 :=
by sorry

end symmetric_about_line_l2152_215222


namespace geometric_series_sum_l2152_215224

theorem geometric_series_sum :
  let a := 6
  let r := - (2 / 5)
  s = ∑' n, (a * r ^ n) ↔ s = 30 / 7 :=
sorry

end geometric_series_sum_l2152_215224


namespace correct_operation_l2152_215221

theorem correct_operation (a : ℝ) : 2 * (a^2) * a = 2 * (a^3) := by sorry

end correct_operation_l2152_215221


namespace number_of_footballs_is_3_l2152_215250

-- Define the variables and conditions directly from the problem

-- Let F be the cost of one football and S be the cost of one soccer ball
variable (F S : ℝ)

-- Condition 1: Some footballs and 1 soccer ball cost 155 dollars
variable (number_of_footballs : ℝ)
variable (H1 : F * number_of_footballs + S = 155)

-- Condition 2: 2 footballs and 3 soccer balls cost 220 dollars
variable (H2 : 2 * F + 3 * S = 220)

-- Condition 3: The cost of one soccer ball is 50 dollars
variable (H3 : S = 50)

-- Theorem: Prove that the number of footballs in the first set is 3
theorem number_of_footballs_is_3 (H1 H2 H3 : Prop) :
  number_of_footballs = 3 := by
  sorry

end number_of_footballs_is_3_l2152_215250


namespace pollutant_decay_l2152_215293

noncomputable def p (t : ℝ) (p0 : ℝ) := p0 * 2^(-t / 30)

theorem pollutant_decay : 
  ∃ p0 : ℝ, p0 = 300 ∧ p 60 p0 = 75 * Real.log 2 := 
by
  sorry

end pollutant_decay_l2152_215293


namespace area_of_region_l2152_215238

-- Definitions from the problem's conditions.
def equation (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 10*y = -9

-- Statement of the theorem.
theorem area_of_region : 
  ∃ (area : ℝ), (∀ x y : ℝ, equation x y → True) ∧ area = 32 * Real.pi :=
by
  sorry

end area_of_region_l2152_215238


namespace license_plate_combinations_l2152_215263

def number_of_license_plates : ℕ :=
  10^5 * 26^3 * 20

theorem license_plate_combinations :
  number_of_license_plates = 35152000000 := by
  -- Here's where the proof would go
  sorry

end license_plate_combinations_l2152_215263


namespace probability_diagonals_intersect_hexagon_l2152_215295

theorem probability_diagonals_intersect_hexagon:
  let n : ℕ := 6
  let total_diagonals := (n * (n - 3)) / 2 -- Total number of diagonals in a convex polygon
  let total_pairs := (total_diagonals * (total_diagonals - 1)) / 2 -- Total number of ways to choose 2 diagonals
  let non_principal_intersections := 3 * 6 -- Each of 6 non-principal diagonals intersects 3 others
  let principal_intersections := 4 * 3 -- Each of 3 principal diagonals intersects 4 others
  let total_intersections := (non_principal_intersections + principal_intersections) / 2 -- Correcting for double-counting
  let probability := total_intersections / total_pairs -- Probability of intersection inside the hexagon
  probability = 5 / 12 := by
  let n : ℕ := 6
  let total_diagonals := (n * (n - 3)) / 2
  let total_pairs := (total_diagonals * (total_diagonals - 1)) / 2
  let non_principal_intersections := 3 * 6
  let principal_intersections := 4 * 3
  let total_intersections := (non_principal_intersections + principal_intersections) / 2
  let probability := total_intersections / total_pairs
  have h : total_diagonals = 9 := by norm_num
  have h_pairs : total_pairs = 36 := by norm_num
  have h_intersections : total_intersections = 15 := by norm_num
  have h_prob : probability = 5 / 12 := by norm_num
  exact h_prob

end probability_diagonals_intersect_hexagon_l2152_215295


namespace total_eggs_l2152_215298

-- Define the number of eggs eaten in each meal
def breakfast_eggs : ℕ := 2
def lunch_eggs : ℕ := 3
def dinner_eggs : ℕ := 1

-- Prove the total number of eggs eaten is 6
theorem total_eggs : breakfast_eggs + lunch_eggs + dinner_eggs = 6 :=
by
  sorry

end total_eggs_l2152_215298


namespace quadratic_roots_l2152_215297

theorem quadratic_roots (r s : ℝ) (A : ℝ) (B : ℝ) (C : ℝ) (p q : ℝ) 
  (h1 : A = 3) (h2 : B = 4) (h3 : C = 5) 
  (h4 : r + s = -B / A) (h5 : rs = C / A) 
  (h6 : 4 * rs = q) :
  p = 56 / 9 :=
by 
  -- We assume the correct answer is given as we skip the proof details here.
  sorry

end quadratic_roots_l2152_215297


namespace smallest_N_for_percentages_l2152_215247

theorem smallest_N_for_percentages 
  (N : ℕ) 
  (h1 : ∃ N, ∀ f ∈ [1/10, 2/5, 1/5, 3/10], ∃ k : ℕ, N * f = k) :
  N = 10 := 
by
  sorry

end smallest_N_for_percentages_l2152_215247


namespace smallest_difference_l2152_215252

theorem smallest_difference (a b : ℤ) (h1 : a + b < 11) (h2 : a > 6) : a - b = 4 :=
by
  sorry

end smallest_difference_l2152_215252


namespace total_owed_proof_l2152_215251

-- Define initial conditions
def initial_owed : ℕ := 20
def borrowed : ℕ := 8

-- Define the total amount owed
def total_owed : ℕ := initial_owed + borrowed

-- Prove the statement
theorem total_owed_proof : total_owed = 28 := 
by 
  -- Proof is omitted with sorry
  sorry

end total_owed_proof_l2152_215251


namespace cost_price_eq_560_l2152_215258

variables (C SP1 SP2 : ℝ)
variables (h1 : SP1 = 0.79 * C) (h2 : SP2 = SP1 + 140) (h3 : SP2 = 1.04 * C)

theorem cost_price_eq_560 : C = 560 :=
by 
  sorry

end cost_price_eq_560_l2152_215258


namespace teachers_per_grade_correct_l2152_215216

def fifth_graders : ℕ := 109
def sixth_graders : ℕ := 115
def seventh_graders : ℕ := 118
def parents_per_grade : ℕ := 2
def number_of_grades : ℕ := 3
def buses : ℕ := 5
def seats_per_bus : ℕ := 72

-- Total number of students
def total_students : ℕ := fifth_graders + sixth_graders + seventh_graders

-- Total number of parents
def total_parents : ℕ := parents_per_grade * number_of_grades

-- Total number of seats available on the buses
def total_seats : ℕ := buses * seats_per_bus

-- Seats left for teachers
def seats_for_teachers : ℕ := total_seats - total_students - total_parents

-- Teachers per grade
def teachers_per_grade : ℕ := seats_for_teachers / number_of_grades

theorem teachers_per_grade_correct : teachers_per_grade = 4 := sorry

end teachers_per_grade_correct_l2152_215216


namespace eq_determines_ratio_l2152_215231

theorem eq_determines_ratio (a b x y : ℝ) (h : a * x^3 + b * x^2 * y + b * x * y^2 + a * y^3 = 0) :
  ∃ t : ℝ, t = x / y ∧ (a * t^3 + b * t^2 + b * t + a = 0) :=
sorry

end eq_determines_ratio_l2152_215231


namespace problem_1_problem_2_l2152_215268

-- Definitions for the sets A and B:

def set_A : Set ℝ := { x | x^2 - x - 12 ≤ 0 }
def set_B (m : ℝ) : Set ℝ := { x | 2 * m - 1 < x ∧ x < 1 + m }

-- Problem 1: When m = -2, find A ∪ B
theorem problem_1 : set_A ∪ set_B (-2) = { x | -5 < x ∧ x ≤ 4 } :=
sorry

-- Problem 2: If A ∩ B = B, find the range of the real number m
theorem problem_2 : (∀ x, x ∈ set_B m → x ∈ set_A) ↔ m ≥ -1 :=
sorry

end problem_1_problem_2_l2152_215268


namespace bricklayer_team_size_l2152_215255

/-- Problem: Prove the number of bricklayers in the team -/
theorem bricklayer_team_size
  (x : ℕ)
  (h1 : 432 = (432 * (x - 4) / x) + 9 * (x - 4)) :
  x = 16 :=
sorry

end bricklayer_team_size_l2152_215255


namespace phone_answered_within_two_rings_l2152_215266

def probability_of_first_ring : ℝ := 0.5
def probability_of_second_ring : ℝ := 0.3
def probability_of_within_two_rings : ℝ := 0.8

theorem phone_answered_within_two_rings :
  probability_of_first_ring + probability_of_second_ring = probability_of_within_two_rings :=
by
  sorry

end phone_answered_within_two_rings_l2152_215266


namespace find_x_l2152_215240

/-
If two minus the reciprocal of (3 - x) equals the reciprocal of (2 + x), 
then x equals (1 + sqrt(15)) / 2 or (1 - sqrt(15)) / 2.
-/
theorem find_x (x : ℝ) :
  (2 - (1 / (3 - x)) = (1 / (2 + x))) → 
  (x = (1 + Real.sqrt 15) / 2 ∨ x = (1 - Real.sqrt 15) / 2) :=
by 
  sorry

end find_x_l2152_215240


namespace new_average_doubled_l2152_215282

theorem new_average_doubled
  (average : ℕ)
  (num_students : ℕ)
  (h_avg : average = 45)
  (h_num_students : num_students = 30)
  : (2 * average * num_students / num_students) = 90 := by
  sorry

end new_average_doubled_l2152_215282


namespace tart_fill_l2152_215296

theorem tart_fill (cherries blueberries total : ℚ) (h_cherries : cherries = 0.08) (h_blueberries : blueberries = 0.75) (h_total : total = 0.91) :
  total - (cherries + blueberries) = 0.08 :=
by
  sorry

end tart_fill_l2152_215296


namespace boat_distance_against_stream_l2152_215274

/-- 
  Given:
  1. The boat goes 13 km along the stream in one hour.
  2. The speed of the boat in still water is 11 km/hr.

  Prove:
  The distance the boat goes against the stream in one hour is 9 km.
-/
theorem boat_distance_against_stream (v_s : ℝ) (distance_along_stream time : ℝ) (v_still : ℝ) :
  distance_along_stream = 13 ∧ time = 1 ∧ v_still = 11 ∧ (v_still + v_s) = 13 → 
  (v_still - v_s) * time = 9 := by
  sorry

end boat_distance_against_stream_l2152_215274


namespace craft_store_pricing_maximize_daily_profit_l2152_215218

theorem craft_store_pricing (profit_per_item marked_price cost_price : ℝ)
  (h₁ : profit_per_item = marked_price - cost_price)
  (h₂ : 8 * 0.85 * marked_price + 12 * (marked_price - 35) = 20 * cost_price)
  : cost_price = 155 ∧ marked_price = 200 := 
sorry

theorem maximize_daily_profit (profit_per_item cost_price marked_price : ℝ)
  (h₁ : profit_per_item = marked_price - cost_price)
  (h₃ : ∀ p : ℝ, (100 + 4 * (200 - p)) * (p - cost_price) ≤ 4900)
  : p = 190 ∧ daily_profit = 4900 :=
sorry

end craft_store_pricing_maximize_daily_profit_l2152_215218


namespace find_m_when_power_function_decreasing_l2152_215228

theorem find_m_when_power_function_decreasing :
  ∃ m : ℝ, (m^2 - 2 * m - 2 = 1) ∧ (-4 * m - 2 < 0) ∧ (m = 3) :=
by
  sorry

end find_m_when_power_function_decreasing_l2152_215228


namespace cost_of_adult_ticket_l2152_215207

theorem cost_of_adult_ticket
    (child_ticket_cost : ℝ)
    (total_tickets : ℕ)
    (total_receipts : ℝ)
    (adult_tickets_sold : ℕ)
    (A : ℝ)
    (child_tickets_sold : ℕ := total_tickets - adult_tickets_sold)
    (total_revenue_adult : ℝ := adult_tickets_sold * A)
    (total_revenue_child : ℝ := child_tickets_sold * child_ticket_cost) :
    child_ticket_cost = 4 →
    total_tickets = 130 →
    total_receipts = 840 →
    adult_tickets_sold = 90 →
    total_revenue_adult + total_revenue_child = total_receipts →
    A = 7.56 :=
by
  intros
  sorry

end cost_of_adult_ticket_l2152_215207


namespace commentator_mistake_l2152_215217

def round_robin_tournament : Prop :=
  ∀ (x y : ℝ),
    x + 2 * x + 13 * y = 105 ∧ x < y ∧ y < 2 * x → False

theorem commentator_mistake : round_robin_tournament :=
  by {
    sorry
  }

end commentator_mistake_l2152_215217


namespace min_value_of_M_l2152_215206

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)

noncomputable def M : ℝ :=
  (Real.rpow (a / (b + c)) (1 / 4)) + (Real.rpow (b / (c + a)) (1 / 4)) + (Real.rpow (c / (b + a)) (1 / 4)) +
  Real.sqrt ((b + c) / a) + Real.sqrt ((a + c) / b) + Real.sqrt ((a + b) / c)

theorem min_value_of_M : M a b c = 3 * Real.sqrt 2 + (3 * Real.rpow 8 (1 / 4)) / 2 := sorry

end min_value_of_M_l2152_215206


namespace least_prime_P_with_integer_roots_of_quadratic_l2152_215267

theorem least_prime_P_with_integer_roots_of_quadratic :
  ∃ P : ℕ, P.Prime ∧ (∃ m : ℤ,  m^2 = 12 * P + 60) ∧ P = 7 :=
by
  sorry

end least_prime_P_with_integer_roots_of_quadratic_l2152_215267


namespace problem1_l2152_215277

theorem problem1 (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y > 2) : 
    (1 + x) / y < 2 ∨ (1 + y) / x < 2 := 
sorry

end problem1_l2152_215277


namespace complex_transformation_l2152_215248

open Complex

theorem complex_transformation :
  let z := -1 + (7 : ℂ) * I
  let rotation := (1 / 2 + (Real.sqrt 3) / 2 * I)
  let dilation := 2
  (z * rotation * dilation = -22 - ((Real.sqrt 3) - 7) * I) :=
by
  sorry

end complex_transformation_l2152_215248


namespace chelsea_guaranteed_victory_l2152_215245

noncomputable def minimum_bullseye_shots_to_win (k : ℕ) (n : ℕ) : ℕ :=
  if (k + 5 * n + 500 > k + 930) then n else sorry

theorem chelsea_guaranteed_victory (k : ℕ) :
  minimum_bullseye_shots_to_win k 87 = 87 :=
by
  sorry

end chelsea_guaranteed_victory_l2152_215245


namespace gamin_difference_calculation_l2152_215273

def largest_number : ℕ := 532
def smallest_number : ℕ := 406
def difference : ℕ := 126

theorem gamin_difference_calculation : largest_number - smallest_number = difference :=
by
  -- The solution proves that the difference between the largest and smallest numbers is 126.
  sorry

end gamin_difference_calculation_l2152_215273


namespace zeros_in_square_of_nines_l2152_215257

def num_zeros (n : ℕ) (m : ℕ) : ℕ :=
  -- Count the number of zeros in the decimal representation of m
sorry

theorem zeros_in_square_of_nines :
  num_zeros 6 ((10^6 - 1)^2) = 5 :=
sorry

end zeros_in_square_of_nines_l2152_215257


namespace not_perfect_square_l2152_215284

theorem not_perfect_square (p : ℕ) (hp : Nat.Prime p) : ¬ ∃ t : ℕ, 7 * p + 3^p - 4 = t^2 :=
sorry

end not_perfect_square_l2152_215284


namespace triangle_is_isosceles_l2152_215208

theorem triangle_is_isosceles
  (A B C : ℝ)
  (h_sum_angles : A + B + C = π)
  (h_condition : (Real.sin A + Real.sin B) * (Real.cos A + Real.cos B) = 2 * Real.sin C) :
  A = B :=
sorry

end triangle_is_isosceles_l2152_215208


namespace x_is_half_l2152_215254

theorem x_is_half (w x y : ℝ) 
  (h1 : 6 / w + 6 / x = 6 / y) 
  (h2 : w * x = y) 
  (h3 : (w + x) / 2 = 0.5) : x = 0.5 :=
sorry

end x_is_half_l2152_215254


namespace part1_part2_l2152_215259

-- Definitions from condition part
def f (a x : ℝ) := a * x^2 + (1 + a) * x + a

-- Part (1) Statement
theorem part1 (a : ℝ) : 
  (a ≥ -1/3) → (∀ x : ℝ, f a x ≥ 0) :=
sorry

-- Part (2) Statement
theorem part2 (a : ℝ) : 
  (a > 0) → 
  (∀ x : ℝ, f a x < a - 1) → 
  ((0 < a ∧ a < 1) → (-1/a < x ∧ x < -1) ∨ 
   (a = 1) → False ∨
   (a > 1) → (-1 < x ∧ x < -1/a)) :=
sorry

end part1_part2_l2152_215259


namespace domain_of_function_l2152_215288

theorem domain_of_function :
  ∀ x : ℝ, ⌊x^2 - 8 * x + 18⌋ ≠ 0 :=
sorry

end domain_of_function_l2152_215288


namespace find_number_l2152_215223

theorem find_number (x : ℝ) (h : 0.65 * x = 0.8 * x - 21) : x = 140 := by
  sorry

end find_number_l2152_215223


namespace geometric_sequence_product_l2152_215285

theorem geometric_sequence_product 
  (a : ℕ → ℝ) (r : ℝ)
  (h_geom : ∀ n, a (n+1) = a n * r)
  (h_pos : ∀ n, a n > 0)
  (h_log_sum : Real.log (a 3) + Real.log (a 8) + Real.log (a 13) = 6) :
  a 1 * a 15 = 10000 := 
sorry

end geometric_sequence_product_l2152_215285
