import Mathlib

namespace slope_parallel_to_line_l223_223626

theorem slope_parallel_to_line (x y : ℝ) (h : 3 * x - 6 * y = 15) :
  (∃ m, (∀ b, y = m * x + b) ∧ (∀ k, k ≠ m → ¬ 3 * x - 6 * (k * x + b) = 15)) →
  ∃ p, p = 1/2 :=
sorry

end slope_parallel_to_line_l223_223626


namespace even_red_points_l223_223550

theorem even_red_points {P : Fin 6 → ℝ^3} 
  (h_pos : ∀ i j k l m n : Fin 6, Finset.card ({i, j, k, l, m, n} : Finset (Fin 6)) = 6)
  (h_general : ∀ (A B C D : Fin 6), affine_independent ℝ (λ i, P i))
  : ∃ n_red : ℕ, n_red % 2 = 0 :=
  sorry

end even_red_points_l223_223550


namespace shooting_enthusiast_l223_223680

variables {P : ℝ} -- Declare P as a real number

-- Define the conditions where X follows a binomial distribution
def binomial_variance (n : ℕ) (p : ℝ) :=
  n * p * (1 - p)

-- State the theorem in Lean 4
theorem shooting_enthusiast (h : binomial_variance 3 P = 3 / 4) : 
  P = 1 / 2 :=
by
  sorry -- Proof goes here

end shooting_enthusiast_l223_223680


namespace inequality_solution_l223_223949

theorem inequality_solution 
  (x : ℝ)
  (h₁ : x ≠ 2)
  (h₂ : x ≠ 3)
  (h₃ : x ≠ 4)
  (h₄ : x ≠ 5)
  (h₅ : x ∈ ((Set.Ioo Float.neg_infty (-2)) ∪ Set.Ioo (-1) 1 ∪ Set.Ioo 2 3 ∪ Set.Ioo 4 6 ∪ Set.Ioo 7 Float.infty)) :
  (2 / (x - 2)) - (5 / (x - 3)) + (5 / (x - 4)) - (2 / (x - 5)) < (1 / 24) := 
by 
  sorry

end inequality_solution_l223_223949


namespace contradiction_proof_l223_223245

theorem contradiction_proof :
  ∀ (a b c d : ℝ),
    a + b = 1 →
    c + d = 1 →
    ac + bd > 1 →
    (a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0) :=
by
  sorry

end contradiction_proof_l223_223245


namespace vacation_books_l223_223851

-- Define the number of mystery, fantasy, and biography novels.
def num_mystery : ℕ := 3
def num_fantasy : ℕ := 4
def num_biography : ℕ := 3

-- Define the condition that we want to choose three books with no more than one from each genre.
def num_books_to_choose : ℕ := 3
def max_books_per_genre : ℕ := 1

-- The number of ways to choose one book from each genre
def num_combinations (m f b : ℕ) : ℕ :=
  m * f * b

-- Prove that the number of possible sets of books is 36
theorem vacation_books : num_combinations num_mystery num_fantasy num_biography = 36 := by
  sorry

end vacation_books_l223_223851


namespace total_cost_after_discounts_l223_223300

-- Definition of the cost function with applicable discounts
def pencil_cost (price: ℝ) (count: ℕ) (discount_threshold: ℕ) (discount_rate: ℝ) :=
  let initial_cost := count * price
  if count > discount_threshold then
    initial_cost - (initial_cost * discount_rate)
  else initial_cost

def pen_cost (price: ℝ) (count: ℕ) (discount_threshold: ℕ) (discount_rate: ℝ) :=
  let initial_cost := count * price
  if count > discount_threshold then
    initial_cost - (initial_cost * discount_rate)
  else initial_cost

-- The statement to be proved
theorem total_cost_after_discounts :
  let pencil_price := 2.50
  let pen_price := 3.50
  let pencil_count := 38
  let pen_count := 56
  let pencil_discount_threshold := 30
  let pencil_discount_rate := 0.10
  let pen_discount_threshold := 50
  let pen_discount_rate := 0.15
  let total_cost := pencil_cost pencil_price pencil_count pencil_discount_threshold pencil_discount_rate
                   + pen_cost pen_price pen_count pen_discount_threshold pen_discount_rate
  total_cost = 252.10 := 
by 
  sorry

end total_cost_after_discounts_l223_223300


namespace circle_symmetric_point_l223_223795

theorem circle_symmetric_point (a b : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + a * x - 2 * y + b = 0 → x = 2 ∧ y = 1) ∧
  (∀ x y : ℝ, (x, y) ∈ { (px, py) | px = 2 ∧ py = 1 ∨ x + y - 1 = 0 } → x^2 + y^2 + a * x - 2 * y + b = 0) →
  a = 0 ∧ b = -3 := 
by {
  sorry
}

end circle_symmetric_point_l223_223795


namespace six_people_acquaintance_or_strangers_l223_223157

theorem six_people_acquaintance_or_strangers (p : Fin 6 → Prop) :
  ∃ (A B C : Fin 6), (p A ∧ p B ∧ p C) ∨ (¬p A ∧ ¬p B ∧ ¬p C) :=
sorry

end six_people_acquaintance_or_strangers_l223_223157


namespace triangle_similar_l223_223694

-- Define the points in the convex pentagon ABCDE
variables (A B C D E O M N : Type*)

-- Assuming that ABC and CDE are equilateral
variable [isEquilateral △A B C]
variable [isEquilateral △C D E]

-- Assuming that O is the centroid of △A B C
variable [isCentroid O △A B C]

-- Assuming that M is the midpoint of segment BD, N is the midpoint of segment AE
variable [isMidpoint M B D]
variable [isMidpoint N A E]

-- Proposition that we need to prove
theorem triangle_similar (h1: isEquilateral △A B C)
                         (h2: isEquilateral △C D E)
                         (h3: isCentroid O △A B C)
                         (h4: isMidpoint M B D)
                         (h5: isMidpoint N A E) :
  triangleSimilarity (△O M E) (△O N D) := 
sorry

end triangle_similar_l223_223694


namespace min_chord_length_l223_223377

theorem min_chord_length 
  (m : ℝ)
  (circle_eq : ∀ x y, (x - 3)^2 + y^2 = 25)
  (line_eq : ∀ x y, (m + 1) * x + (m - 1) * y - 2 = 0) 
  : ∃ a b : ℝ, a = 4 * real.sqrt 5 := by
  sorry

end min_chord_length_l223_223377


namespace pile_splitting_l223_223111

theorem pile_splitting (single_stone_piles : ℕ) :
  ∃ (final_heap_size : ℕ), 
    (∀ heap_size ≤ single_stone_piles, heap_size > 0 → (heap_size * 2) ≥ heap_size) ∧ (final_heap_size = single_stone_piles) :=
by
  sorry

end pile_splitting_l223_223111


namespace angle_C1OC2_l223_223235

-- Definitions based on conditions from part a)
def three_lines_forming_equal_angles (O : Point) (l1 l2 l3 : Line) : Prop :=
  angle_between_lines l1 l2 = angle_between_lines l2 l3 ∧
  angle_between_lines l2 l3 = angle_between_lines l3 l1

def point_on_line (A : Point) (l : Line) : Prop :=
  -- This would be defined properly according to how points and lines are defined in the library
  sorry

def intersection (l1 l2 : Line) : Point :=
  -- This would return the intersection point of two lines
  sorry

-- Problem restated in Lean
theorem angle_C1OC2 (O A1 A2 B1 B2 C1 C2 : Point) (l1 l2 l3 : Line) :
  three_lines_forming_equal_angles O l1 l2 l3 →
  point_on_line A1 l1 →
  point_on_line A2 l1 →
  point_on_line B1 l2 →
  point_on_line B2 l2 →
  C1 = intersection (line_through A1 B1) (line_through A2 B2) →
  point_on_line C1 l3 →
  C2 = intersection (line_through A1 B2) (line_through A2 B1) →
  angle_between_points O C1 C2 = 90 :=
by sorry

end angle_C1OC2_l223_223235


namespace inequality_proof_l223_223497

theorem inequality_proof (a b : ℝ) (n : ℕ) (x : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ge_two : 2 ≤ n) (h_pos_x : 0 < x) (h_ineq : x ^ n ≤ a * x + b) :
  x < Real.root (n - 1) (2 * a) + Real.root n (2 * b) := 
sorry

end inequality_proof_l223_223497


namespace product_of_prs_eq_60_l223_223440

theorem product_of_prs_eq_60 (p r s : ℕ) (h1 : 3 ^ p + 3 ^ 5 = 270) (h2 : 2 ^ r + 46 = 94) (h3 : 6 ^ s + 5 ^ 4 = 1560) :
  p * r * s = 60 :=
  sorry

end product_of_prs_eq_60_l223_223440


namespace unique_arrangements_mississippi_l223_223746

open scoped Nat

theorem unique_arrangements_mississippi : 
  let n := 11
  let n_M := 1
  let n_I := 4
  let n_S := 4
  let n_P := 2
  (n.fact / (n_M.fact * n_I.fact * n_S.fact * n_P.fact)) = 34650 :=
  by
  sorry

end unique_arrangements_mississippi_l223_223746


namespace mutually_exclusive_not_complementary_l223_223282

-- Definitions based on conditions in the problem
def total_balls : ℕ := 6
def red_balls : ℕ := 3
def white_balls : ℕ := 2
def black_balls : ℕ := 1

-- Event A: At least one white ball
def at_least_one_white_ball (events : set ℕ) : Prop :=
  ∃ (n : ℕ), n ∈ events ∧ n ≥ 1

-- Event B: One red ball and one black ball
def one_red_one_black (events : set ℕ) : Prop :=
  ∃ (r b : ℕ), r ∈ events ∧ b ∈ events ∧ r = 1 ∧ b = 1

-- The final proof problem statement in Lean 4
theorem mutually_exclusive_not_complementary :
(EX : set ℕ) → 
(at_least_one_white_ball EX) ∧ (one_red_one_black EX)
→ ...
-- [Mutually exclusive, but not complementary logic goes here]
sorry

end mutually_exclusive_not_complementary_l223_223282


namespace min_chord_length_l223_223378

theorem min_chord_length 
  (m : ℝ)
  (circle_eq : ∀ x y, (x - 3)^2 + y^2 = 25)
  (line_eq : ∀ x y, (m + 1) * x + (m - 1) * y - 2 = 0) 
  : ∃ a b : ℝ, a = 4 * real.sqrt 5 := by
  sorry

end min_chord_length_l223_223378


namespace sum_of_fractions_l223_223228

theorem sum_of_fractions : (1/2 + 1/2 + 1/3 + 1/3 + 1/3) = 2 :=
by
  -- Proof goes here
  sorry

end sum_of_fractions_l223_223228


namespace find_eccentricity_l223_223373

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
  -- c^2 = a^2 + b^2
  let c := Real.sqrt (a^2 + b^2)
  -- eccentricity e = c / a
  in c / a

theorem find_eccentricity 
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : (let c := Real.sqrt (a^2 + b^2) in (Real.sqrt (7) / 2) = c / a)) :
  hyperbola_eccentricity a b ha hb = (Real.sqrt (7) / 2) := 
by 
  sorry

end find_eccentricity_l223_223373


namespace find_E_equals_2023_l223_223533

noncomputable def proof : Prop :=
  ∃ a b c : ℝ, a ≠ b ∧ (a^2 * (b + c) = 2023) ∧ (b^2 * (c + a) = 2023) ∧ (c^2 * (a + b) = 2023)

theorem find_E_equals_2023 : proof :=
by
  sorry

end find_E_equals_2023_l223_223533


namespace find_n_l223_223869

theorem find_n (x : ℝ) (hx : x > 0) (h : x / n + x / 25 = 0.24000000000000004 * x) : n = 5 :=
sorry

end find_n_l223_223869


namespace symmetric_graph_value_l223_223518

-- Definitions from conditions
def f (x : ℝ) : ℝ := Real.log (x + 3) / Real.log 2

-- Being symmetric about y=x implies g is the inverse of f
def g (x : ℝ) : ℝ := sorry -- Define formally as the inverse of f (if details about g are missing)

-- The statement to prove
theorem symmetric_graph_value : f 1 + g 1 = 2 := by
sorry

end symmetric_graph_value_l223_223518


namespace number_of_true_propositions_l223_223602

variable {a b c : ℝ}

theorem number_of_true_propositions :
  (2 = (if (a > b → a * c ^ 2 > b * c ^ 2) then 1 else 0) +
       (if (a * c ^ 2 > b * c ^ 2 → a > b) then 1 else 0) +
       (if (¬(a * c ^ 2 > b * c ^ 2) → ¬(a > b)) then 1 else 0) +
       (if (¬(a > b) → ¬(a * c ^ 2 > b * c ^ 2)) then 1 else 0)) :=
sorry

end number_of_true_propositions_l223_223602


namespace kolya_sheets_exceed_500_l223_223897

theorem kolya_sheets_exceed_500 :
  ∃ k : ℕ, (10 + k * (k + 1) / 2 > 500) :=
sorry

end kolya_sheets_exceed_500_l223_223897


namespace arithmetic_sequence_general_term_sum_of_inverse_l223_223386

noncomputable def a_n (n : ℕ) : ℝ := 4 * n + 1

noncomputable def S_n (n : ℕ) : ℝ := n * (a_n 1 + a_n n) / 2

noncomputable def T_n (n : ℕ) : ℝ := ∑ k in Finset.range n, (1 / (S_n (k + 1) - (k + 1)))

theorem arithmetic_sequence_general_term (n : ℕ) : a_n n = 4 * n + 1 := 
by sorry

theorem sum_of_inverse (n : ℕ) : T_n n = n / (2 * (n + 1)) := 
by sorry

end arithmetic_sequence_general_term_sum_of_inverse_l223_223386


namespace min_recolor_edges_l223_223142

open SimpleGraph

noncomputable def min_recolor_edges_for_connected (n : ℕ) (h : n ≥ 3) : ℕ :=
  ⌊n / 3⌋

theorem min_recolor_edges (n : ℕ) (h : n ≥ 3) :
  ∃ k, ∀ (G : SimpleGraph (fin n)), (∀ e, G.edge_coloring (fin 3)) ->
    (∃ recolor : fin k -> G.edge_set, (G.recolor_edges (fin 3) recolor).is_connected) ↔
    k = min_recolor_edges_for_connected n h :=
sorry

end min_recolor_edges_l223_223142


namespace angle_between_lines_l223_223617

-- Definitions of the problem conditions
variables (O A B C : Type) [IsPoint O] [IsPoint A] [IsPoint B] [IsPoint C]
variables (circle_center : O) (tangent_point1 : A) (tangent_point2 : B) (intersection_point : C)
variables (line1 : Line) (line2 : Line)

-- Conditions
axiom tangent1 : Tangent line1 circle_center tangent_point1
axiom tangent2 : Tangent line2 circle_center tangent_point2
axiom intersection : Intersects line1 line2 intersection_point
axiom angle_ABO : ∠(A, B, O) = 40°

-- Conclusion
theorem angle_between_lines : ∠(A, C, B) = 80° :=
sorry

end angle_between_lines_l223_223617


namespace simplify_expression_l223_223177

theorem simplify_expression (y : ℝ) : (3 * y^4)^5 = 243 * y^20 :=
sorry

end simplify_expression_l223_223177


namespace equivalent_proof_triangle_maximum_area_l223_223889

noncomputable def given_triangle (a b c : ℝ) (A B C : ℝ) (h1 : angle A B C) (h2 : angle B C A) (h3 : angle C A B) 
  (condition : 2 * a * Real.sin A = (2 * b + c) * Real.sin B + (2 * c + b) * Real.sin C) : Prop :=
  A = 2 * Real.pi / 3 ∧ (∀ (a = 2 * Real.sqrt 3), ∃ (S : ℝ), S = Real.sqrt 3)

theorem equivalent_proof_triangle_maximum_area :
  ∀ (a b c : ℝ) (A B C : ℝ),
  (triangle a b c A B C) → 
  (2 * a * Real.sin A = (2 * b + c) * Real.sin B + (2 * c + b) * Real.sin C) → 
  A = 2 * Real.pi / 3 ∧ 
  (a = 2 * Real.sqrt 3 → (∀ (S : ℝ), S ≤ Real.sqrt 3)) :=
by
  -- Here would go the detailed proof: sorry
  sorry

end equivalent_proof_triangle_maximum_area_l223_223889


namespace number_of_ordered_quadruples_l223_223061

theorem number_of_ordered_quadruples :
  let n := (finset.range 50).powerset.filter (λ s, s.card = 4 ∧ ∑ i in s, 2 * (i + 1) - 1 = 100).card in
  (n : ℝ) / 100 = 208.25 :=
sorry

end number_of_ordered_quadruples_l223_223061


namespace total_doctors_and_nurses_l223_223612

theorem total_doctors_and_nurses
    (ratio_doctors_nurses : ℕ -> ℕ -> Prop)
    (num_nurses : ℕ)
    (h₁ : ratio_doctors_nurses 2 3)
    (h₂ : num_nurses = 150) :
    ∃ num_doctors total_doctors_nurses, 
    (total_doctors_nurses = num_doctors + num_nurses) 
    ∧ (num_doctors / num_nurses = 2 / 3) 
    ∧ total_doctors_nurses = 250 := 
by
  sorry

end total_doctors_and_nurses_l223_223612


namespace find_5a_plus_5b_l223_223332

noncomputable def g (x : ℝ) : ℝ := 5 * x - 4
noncomputable def f (a b x : ℝ) : ℝ := a * x + b
noncomputable def f_inv (a b x : ℝ) : ℝ := g x + 3

theorem find_5a_plus_5b (a b : ℝ) (h_inverse : ∀ x, f_inv a b (f a b x) = x) : 5 * a + 5 * b = 2 :=
by
  sorry

end find_5a_plus_5b_l223_223332


namespace jim_miles_driven_l223_223495

theorem jim_miles_driven (total_journey: ℕ) (remaining_miles: ℕ) (h1: total_journey = 1200) (h2: remaining_miles = 816) : total_journey - remaining_miles = 384 :=
by {
  rw [h1, h2],
  norm_num,
}

end jim_miles_driven_l223_223495


namespace max_sin_a_l223_223913

theorem max_sin_a (a b : ℝ) (h : sin (a + b) = sin a + sin b) : 
  sin a ≤ 1 :=
by
  sorry

end max_sin_a_l223_223913


namespace find_a_l223_223406

def f (x a : ℝ) : ℝ := 2^x - a * 2^(-x)

theorem find_a (a : ℝ) (h : ∀ x : ℝ, f (-x) a = -f x a) : a = 1 := by
  sorry

end find_a_l223_223406


namespace heaps_combination_preserve_similarity_split_stones_into_similar_heaps_l223_223116

def initial_heaps (n : ℕ) : list ℕ := list.repeat 1 n

def combine_heaps (heaps : list ℕ) : list ℕ :=
  if heaps.length ≥ 2 then
    let min1 := list.minimum heaps,
        heaps' := list.erase heaps min1,
        min2 := list.minimum heaps'
    in
    if min1 ≤ min2 then
      (min1 + min2) :: list.erase heaps' min2
    else
      heaps
  else
    heaps

theorem heaps_combination_preserve_similarity (heaps : list ℕ) (h : ∀ x ∈ heaps, x = 1) :
  ∀ combined_heaps, combined_heaps = combine_heaps heaps →
  ∀ x y ∈ combined_heaps, x ≤ y → x + y ≤ 2 * y :=
sorry

theorem split_stones_into_similar_heaps (n : ℕ) :
  ∃ combined_heaps : list ℕ, ∀ x y ∈ combined_heaps, x ≤ y → x + y ≤ 2 * y :=
sorry

end heaps_combination_preserve_similarity_split_stones_into_similar_heaps_l223_223116


namespace geoff_spending_proof_l223_223773

theorem geoff_spending_proof (M : ℕ) 
  (pairs_monday : ℕ := 2) 
  (spend_monday : ℕ := M) 
  (spend_tuesday : ℕ := 4 * M) 
  (spend_wednesday : ℕ := 5 * M) 
  (total_spent : ℕ := 600) : 
  M = 60 :=
  by
    have spending_equation : spend_monday + spend_tuesday + spend_wednesday = total_spent := by
      rw [spend_monday, spend_tuesday, spend_wednesday]
      exact add_assoc (M) (4 * M) (5 * M).symm
    have simplified_equation : 10 * M = 600 := by
      rw [spend_monday, spend_tuesday, spend_wednesday] at spending_equation
      linarith
    sorry

end geoff_spending_proof_l223_223773


namespace calculate_I_l223_223033

def V : ℂ := 2 + 2 * Complex.i
def Z : ℂ := 3 - 4 * Complex.i
def correct_I : ℂ := -2 / 25 + (14 / 25) * Complex.i

theorem calculate_I : (V = Z * correct_I) :=
by
  sorry

end calculate_I_l223_223033


namespace angle_C_max_area_l223_223659

-- Definitions for conditions
variables {R a b A B C : ℝ}
variables (α β γ : Prop) -- Propositions representing the variables

-- Condition 1: A circle with radius R is externally tangent to triangle ABC
def circle_tangent_triangle : Prop := sorry

-- Condition 2: 2R(\sin^2 A - \sin^2 C) = (\sqrt{3}a - b)\sin B
def main_condition : Prop := 
  2 * R * ((Real.sin A) ^ 2 - (Real.sin C) ^ 2) = (Real.sqrt 3 * a - b) * Real.sin B

-- The conditions encoded as hypotheses
axiom h1 : circle_tangent_triangle α
axiom h2 : main_condition β

-- Conclusion 1: C = \pi / 6
theorem angle_C (hyp1 : α) (hyp2 : β) : C = Real.pi / 6 := sorry

-- Conclusion 2: Maximum area of triangle ABC is (\sqrt{3}+2)/4 * R^2
theorem max_area (hyp1 : α) (hyp2 : β) :  ∃ (S_max : ℝ), S_max = (Real.sqrt 3 + 2) / 4 * R^2 := sorry

end angle_C_max_area_l223_223659


namespace split_stones_l223_223082

theorem split_stones (n : ℕ) :
  ∃ (heaps : list ℕ), (∀ h ∈ heaps, 1 ≤ h ∧ h ≤ n) ∧ (∀ i j, i ≠ j → (i < heaps.length ∧ j < heaps.length → heaps.nth i ≤ 2 * heaps.nth j)) :=
sorry

end split_stones_l223_223082


namespace split_into_similar_heaps_l223_223101

noncomputable def similar_sizes (x y : ℕ) : Prop :=
  x ≤ 2 * y

theorem split_into_similar_heaps (n : ℕ) (h : n > 0) : 
  ∃ f : ℕ → ℕ, (∀ k, k < n → similar_sizes (f (k + 1)) (f k)) ∧ f (n - 1) = n := by
  sorry

end split_into_similar_heaps_l223_223101


namespace PR_perpendicular_QS_l223_223646

variables (A B C D P Q R S : Point)
variable (circle : Circle)

-- Conditions
axiom midpoint_arc_AB_R : isMidpoint (arc A B circle) R
axiom midpoint_arc_CD_S : isMidpoint (arc C D circle) S
axiom point_P_on_circle : onCircle P circle
axiom point_Q_on_circle : onCircle Q circle
axiom line_PR : Line P R
axiom line_QS : Line Q S

-- Proof Statement
theorem PR_perpendicular_QS : perpendicular (line P R) (line Q S) :=
sorry

end PR_perpendicular_QS_l223_223646


namespace isabel_total_problems_l223_223047

theorem isabel_total_problems
  (math_pages : ℕ)
  (reading_pages : ℕ)
  (problems_per_page : ℕ)
  (h1 : math_pages = 2)
  (h2 : reading_pages = 4)
  (h3 : problems_per_page = 5) :
  (math_pages + reading_pages) * problems_per_page = 30 :=
by
  sorry

end isabel_total_problems_l223_223047


namespace max_dot_product_l223_223214

-- Definition of regular octagon vertices and vector 
def is_regular_octagon (A : ℕ → (ℝ × ℝ)) : Prop :=
  ∀ i, norm (A ((i + 1) % 8) - A i) = 1

-- Given side length of the octagon is 1
def side_length_one (A : ℕ → (ℝ × ℝ)) : Prop :=
  ∀ i, norm (A ((i + 1) % 8) - A i) = 1

-- Points Ai and Aj are represented as A_i and A_j in the function
-- The vector A1A2 is aligned along the x-axis
noncomputable def vector_A1_A2 := (1, 0 : ℝ × ℝ)

-- Definition of the dot product of vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.fst * v.fst + u.snd * v.snd

-- The maximum dot product value to be shown
theorem max_dot_product (A : ℕ → (ℝ × ℝ)) (h₀ : is_regular_octagon A) (h₁ : side_length_one A) :
  ∃ i j, dot_product (A j - A i) vector_A1_A2 = (ℝ.sqrt 2) + 1 :=
sorry

end max_dot_product_l223_223214


namespace slower_pipe_time_l223_223237

-- Define the rates of the pipes according to the problem's conditions
def rate_pipe2 : ℝ := 1 / (36 * 7) -- Rate of the second (slower) pipe in tanks per minute
def rate_pipe1 : ℝ := 4 * rate_pipe2 -- Rate of the first pipe
def rate_pipe3 : ℝ := 2 * rate_pipe2 -- Rate of the third pipe

-- Combined rate when all three pipes are working together
def combined_rate : ℝ := rate_pipe1 + rate_pipe2 + rate_pipe3

-- Time taken by the second pipe to fill the tank alone
def time_pipe2 : ℝ := 1 / rate_pipe2

-- Leak rate in tanks per minute (converted from 5 liters per hour, assuming tank capacity is needed)
def leak_rate : ℝ := 5 / 60 / T -- T is the tank capacity in liters (to be specified)

-- Theorem to be proven
theorem slower_pipe_time :
  combined_rate * 36 = 1 ∧ time_pipe2 = 252 :=
by
  sorry

end slower_pipe_time_l223_223237


namespace find_f_9_over_2_l223_223926

noncomputable def f (x : ℝ) : ℝ := 
  if x ∈ set.Ioc 1 2 then -2 * x^2 + 2
  else if x ∈ set.Ioc (-2) (-1) then -2 * (x+4)^2 + 2
  else -2 * (x - (4 * ⌊x / 4⌋))^2 + 2

theorem find_f_9_over_2 :
  (f (9 / 2) = 5 / 2) :=
by
  sorry

end find_f_9_over_2_l223_223926


namespace solve_for_x_l223_223437

theorem solve_for_x (x : ℝ) (h : 9 / (1 + 4 / x) = 1) : x = 1 / 2 :=
by
  sorry

end solve_for_x_l223_223437


namespace range_of_c_minus_b_l223_223792

theorem range_of_c_minus_b (B : ℝ) (h1 : 0 < B) (h2 : B < π/4) : 
  (√2 / 2) < 1 / (√2 * Real.sin (B + π / 4)) ∧ 1 / (√2 * Real.sin (B + π / 4)) < 1 :=
by
  sorry

end range_of_c_minus_b_l223_223792


namespace f_f_x_eq_5_has_two_solutions_l223_223515

def f (x : ℝ) : ℝ := if x ≤ 3 then -x + 2 else 3 * x - 7

theorem f_f_x_eq_5_has_two_solutions : 
  ∃ (s : Finset ℝ), ∀ x, f (f x) = 5 ↔ x ∈ s ∧ s.card = 2 :=
by
  sorry

end f_f_x_eq_5_has_two_solutions_l223_223515


namespace yang_hui_problem_solution_l223_223887

theorem yang_hui_problem_solution (x : ℕ) (h : x * (x - 1) = 650) : x * (x - 1) = 650 :=
by
  exact h

end yang_hui_problem_solution_l223_223887


namespace quadrilateral_area_l223_223938

theorem quadrilateral_area :
  let A := (0, 0)
  let B := (0, 4)
  let C := (3, 0)
  let D := (3, 3)
  ∀ A B C D : (ℝ × ℝ),
    A = (0, 0) ∧
    B = (0, 4) ∧
    C = (3, 0) ∧
    D = (3, 3) →
    area_of_quadrilateral A B C D = 7.5 := sorry

end quadrilateral_area_l223_223938


namespace find_LCM_Xiaofang_l223_223637

-- Define relevant notation for two-digit numbers
def two_digit (x y : ℕ) : ℕ := 10 * x + y

-- Define the LCM function (assuming existence of an appropriate LCM function in Lean's Mathlib)
noncomputable def LCM (a b : ℕ) : ℕ := sorry

-- Variables representing the digits, with the restriction that they are between 0-9
variables (A B C D : ℕ)
variable (hAB_CD : LCM (two_digit A B) (two_digit C D) = 1.75 * LCM (two_digit B A) (two_digit D C))

-- Statement to prove
theorem find_LCM_Xiaofang :
  LCM (two_digit A B) (two_digit C D) = 252 :=
by {
  -- Provided conditions
  exact hAB_CD,
  sorry
}

end find_LCM_Xiaofang_l223_223637


namespace find_k_l223_223834

theorem find_k (k : ℝ) (h : ∃ z : ℂ, z^3 + ↑(2 * (k - 1)) * z^2 + 9 * z + ↑(5 * (k - 1)) = 0 ∧ complex.abs z = real.sqrt 5) :
  k = 3 ∨ k = -1 :=
sorry

end find_k_l223_223834


namespace average_percentage_of_popped_kernels_l223_223148

theorem average_percentage_of_popped_kernels (k1 k2 k3 p1 p2 p3 : ℕ) (h1 : k1 = 75) (h2 : k2 = 50) (h3 : k3 = 100)
    (h1_pop : p1 = 60) (h2_pop : p2 = 42) (h3_pop : p3 = 82) :
    ((p1 / (k1 : ℝ) + p2 / (k2 : ℝ) + p3 / (k3 : ℝ)) / 3) * 100 = 82 :=
by
  -- The proportion for each bag
  have prop1 : p1 / (k1 : ℝ) = 60 / 75 := by rw [h1, h1_pop]
  have prop2 : p2 / (k2 : ℝ) = 42 / 50 := by rw [h2, h2_pop]
  have prop3 : p3 / (k3 : ℝ) = 82 / 100 := by rw [h3, h3_pop]
  -- Sum the proportions
  have total_props : (p1 / (k1 : ℝ) + p2 / (k2 : ℝ) + p3 / (k3 : ℝ)) = 0.8 + 0.84 + 0.82 := by
    rw [prop1, prop2, prop3]
  -- Calculating the average proportion
  have avg_prop : ((p1 / (k1 : ℝ) + p2 / (k2 : ℝ) + p3 / (k3 : ℝ)) / 3) = 0.82 := by
    rw [total_props]
  -- Finally multiply the average by 100 to get the percentage
  have avg_percentage : ((p1 / (k1 : ℝ) + p2 / (k2 : ℝ) + p3 / (k3 : ℝ)) / 3) * 100 = 82 := by
    rw [avg_prop]
    norm_num
  exact avg_percentage

end average_percentage_of_popped_kernels_l223_223148


namespace opposite_of_5_is_neg_5_l223_223597

def opposite_number (x y : ℤ) : Prop := x + y = 0

theorem opposite_of_5_is_neg_5 : opposite_number 5 (-5) := by
  sorry

end opposite_of_5_is_neg_5_l223_223597


namespace carA_catches_up_with_carB_at_150_km_l223_223711

-- Definitions representing the problem's conditions
variable (t_A t_B v_A v_B : ℝ)
variable (distance_A_B : ℝ := 300)
variable (time_diff_start : ℝ := 1)
variable (time_diff_end : ℝ := 1)

-- Assumptions representing the problem's conditions
axiom speed_carA : v_A = distance_A_B / t_A
axiom speed_carB : v_B = distance_A_B / (t_A + 2)
axiom time_relation : t_B = t_A + 2
axiom time_diff_starting : t_A = t_B - 2

-- The statement to be proven: car A catches up with car B 150 km from city B
theorem carA_catches_up_with_carB_at_150_km :
  ∃ t₀ : ℝ, v_A * t₀ = v_B * (t₀ + time_diff_start) ∧ (distance_A_B - v_A * t₀ = 150) :=
sorry

end carA_catches_up_with_carB_at_150_km_l223_223711


namespace problem_statement_l223_223692

-- Define the necessary and sufficient conditions
def necessary_but_not_sufficient (P Q : Prop) : Prop :=
  (Q → P) ∧ (¬ (P → Q))

-- Specific propositions in this scenario
def x_conditions (x : ℝ) : Prop := x^2 - 2 * x - 3 = 0
def x_equals_3 (x : ℝ) : Prop := x = 3

-- Prove the given problem statement
theorem problem_statement (x : ℝ) : necessary_but_not_sufficient (x_conditions x) (x_equals_3 x) :=
  sorry

end problem_statement_l223_223692


namespace min_distance_point_curve_to_line_l223_223860

noncomputable def curve (x : ℝ) : ℝ := x^2 - Real.log x

def line (x y : ℝ) : Prop := x - y - 2 = 0

theorem min_distance_point_curve_to_line :
  ∀ (P : ℝ × ℝ), 
  curve P.1 = P.2 →
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 2 :=
by
  sorry

end min_distance_point_curve_to_line_l223_223860


namespace fraction_of_water_lost_l223_223619

theorem fraction_of_water_lost (pipe1_rate pipe2_rate total_fill_time effective_fill_time : ℚ)
  (h_pipe1 : pipe1_rate = 1 / 20)
  (h_pipe2 : pipe2_rate = 1 / 30)
  (h_total_fill : total_fill_time = 1 / 16) :
  ∃ L : ℚ, (1 - L) * (pipe1_rate + pipe2_rate) = total_fill_time ∧ L = 1 / 4 :=
by
  sorry

end fraction_of_water_lost_l223_223619


namespace julies_balls_after_1729_steps_l223_223049

-- Define the process described
def increment_base_8 (n : ℕ) : List ℕ := 
by
  if n = 0 then
    exact [0]
  else
    let rec loop (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n = 0 then acc
      else loop (n / 8) (n % 8 :: acc)
    exact loop n []

-- Define the total number of balls after 'steps' steps
def julies_total_balls (steps : ℕ) : ℕ :=
by 
  exact (increment_base_8 steps).sum

theorem julies_balls_after_1729_steps : julies_total_balls 1729 = 7 :=
by
  sorry

end julies_balls_after_1729_steps_l223_223049


namespace convert_base_1729_to_6_l223_223719

-- Define the base 10 number and the base to convert to
def n : ℕ := 1729
def b : ℕ := 6
def expected_result : ℕ := 120001

-- State the problem: Prove that the base 6 representation of 1729 is 120001
theorem convert_base_1729_to_6 : nat_to_base n b = expected_result := 
by sorry

end convert_base_1729_to_6_l223_223719


namespace perpendicular_lines_l223_223844

-- Define necessary variables and assumptions
variables {Point Line Plane : Type} 
variables (m n : Line) (α β : Plane)

-- Definitions for geometric relations (parallelism and perpendicularity)
class Perpendicular (x y : Type) : Prop := (perp : x ⊥ y)
notation x " ⊥ " y := Perpendicular.perp x y
class Parallel (x y : Type) : Prop := (par : x || y)
notation x " || " y := Parallel.par x y

-- Given conditions
variables (hmα : m ⊥ α) (hnβ : n ⊥ β) (hαβ : α ⊥ β)

-- Proof goal
theorem perpendicular_lines (m n : Line) (α β : Plane) :
  m ⊥ α → n ⊥ β → α ⊥ β → m ⊥ n :=
by
  intro hmα hnβ hαβ
  sorry

end perpendicular_lines_l223_223844


namespace optionB_correct_optionC_correct_l223_223041

open Real

noncomputable def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  A > 0 ∧ B > 0 ∧ C > 0

theorem optionB_correct (A B C a b c : ℝ) :
  triangle A B C a b c →
  0 < tan A * tan B ∧ tan A * tan B < 1 →
  A + B + C ∈ Icc 0 π ∧ (π - (A + B)) < π ∧ (π - (A + B)) > π / 2 :=
  sorry

theorem optionC_correct (A B C a b c : ℝ) :
  triangle A B C a b c →
  cos (A - B) * cos (B - C) * cos (C - A) = 1 →
  A = B ∧ B = C ∧ C = A :=
  sorry

end optionB_correct_optionC_correct_l223_223041


namespace permutations_mississippi_l223_223754

theorem permutations_mississippi : 
  let total_letters := 11
  let m_count := 1
  let i_count := 4
  let s_count := 4
  let p_count := 2
  (Nat.factorial total_letters / (Nat.factorial m_count * Nat.factorial i_count * Nat.factorial s_count * Nat.factorial p_count)) = 34650 := 
by
  sorry

end permutations_mississippi_l223_223754


namespace solution_l223_223966

theorem solution (m : ℝ) 
    (f g : ℝ → ℝ)
    (h1 : ∀ x, f(x) = x^3 - 3*x^2 + m)
    (h2 : ∀ x, g(x) = x^3 - 3*x^2 + 5*m) 
    (h3 : 3 * f 3 = g 3) : 
    m = 0 :=
by {
    sorry
}

end solution_l223_223966


namespace reciprocal_of_neg_2023_l223_223978

theorem reciprocal_of_neg_2023 : (1 : ℝ) / (-2023) = -(1 / 2023) :=
by 
  sorry

end reciprocal_of_neg_2023_l223_223978


namespace total_CDs_in_stores_l223_223470

def shelvesA := 5
def racksPerShelfA := 7
def cdsPerRackA := 8

def shelvesB := 4
def racksPerShelfB := 6
def cdsPerRackB := 7

def totalCDsA := shelvesA * racksPerShelfA * cdsPerRackA
def totalCDsB := shelvesB * racksPerShelfB * cdsPerRackB

def totalCDs := totalCDsA + totalCDsB

theorem total_CDs_in_stores :
  totalCDs = 448 := 
by 
  sorry

end total_CDs_in_stores_l223_223470


namespace max_sin_a_l223_223912

theorem max_sin_a (a b : ℝ) (h : sin (a + b) = sin a + sin b) : 
  sin a ≤ 1 :=
by
  sorry

end max_sin_a_l223_223912


namespace sum_of_interior_edges_l223_223297

noncomputable def interior_edge_sum (outer_length : ℝ) (wood_width : ℝ) (frame_area : ℝ) : ℝ := 
  let outer_width := (frame_area + 3 * (outer_length - 2 * wood_width) * 4) / outer_length
  let inner_length := outer_length - 2 * wood_width
  let inner_width := outer_width - 2 * wood_width
  2 * inner_length + 2 * inner_width

theorem sum_of_interior_edges :
  interior_edge_sum 7 2 34 = 9 := by
  sorry

end sum_of_interior_edges_l223_223297


namespace point_on_line_in_plane_l223_223442

variables {A : Type} {a : set A} {α : set a}

theorem point_on_line_in_plane (hA : A ∈ a) (ha : a ⊆ α) : A ∈ a ∧ a ⊆ α := 
by
  split
  assumption
  assumption

end point_on_line_in_plane_l223_223442


namespace toothpick_pattern_15th_stage_l223_223362

theorem toothpick_pattern_15th_stage :
  let a₁ := 5
  let d := 3
  let n := 15
  a₁ + (n - 1) * d = 47 :=
by
  sorry

end toothpick_pattern_15th_stage_l223_223362


namespace geometric_seq_a2_value_l223_223026

variable {a : ℝ}

def Sn (n : ℕ) := a * 3^n - 2

def a2 := Sn 2 - Sn 1

theorem geometric_seq_a2_value 
  (h : ∀ n : ℕ, a ≠ 0 → Sn n = a * 3^n - 2) : 
  a_2 = 12 :=
sorry

end geometric_seq_a2_value_l223_223026


namespace expression_divisible_by_10_l223_223037

noncomputable def is_divisible_by_10 (n : ℤ) : Prop :=
  n % 10 = 0

theorem expression_divisible_by_10 (x y : ℤ) (h_y : y % 4 = 2) :
  is_divisible_by_10 (215^x + 342^y - 113^2) :=
by
  sorry

end expression_divisible_by_10_l223_223037


namespace even_digits_in_base7_403_l223_223353

def base7_representation (n : ℕ) : list ℕ :=
  let rec aux (n : ℕ) (acc : list ℕ) :=
    if n = 0 then acc
    else aux (n / 7) ((n % 7) :: acc)
  in aux n []

def count_even_digits (digits : list ℕ) : ℕ :=
  digits.countp (λ d, d % 2 = 0)

theorem even_digits_in_base7_403 : count_even_digits (base7_representation 403) = 1 := by
  sorry

end even_digits_in_base7_403_l223_223353


namespace ratio_proof_l223_223432

theorem ratio_proof (a b c d : ℝ) (h1 : b = 3 * a) (h2 : c = 4 * b) (h3 : d = 2 * b - a) :
  (a + b + d) / (b + c + d) = 9 / 20 :=
by sorry

end ratio_proof_l223_223432


namespace relay_race_arrangements_l223_223453

-- Definitions reflecting the conditions from part a)
def num_people := 5
def num_selected := 4
def runners := ['A', 'B', 'C', 'D', 'E']

-- Using these sets directly in conditions reflecting the choice constraints
def first_runner_choices := ['A', 'B', 'C']
def last_runner_choices := ['A', 'B']

-- The proof statement
theorem relay_race_arrangements : 
  let total_arrangements := 
    (first_runner_choices.card * last_runner_choices.card * (nat.factorial (num_selected - 2))) // 2 +
    (2 * nat.factorial (num_selected - 1))
  in total_arrangements = 24 := 
by 
  sorry

end relay_race_arrangements_l223_223453


namespace combine_heaps_l223_223128

def heaps_similar (x y : ℕ) : Prop :=
  x ≤ 2 * y ∧ y ≤ 2 * x

theorem combine_heaps (n : ℕ) : 
  ∃ f : ℕ → ℕ, 
  f 0 = n ∧
  ∀ k, k < n → (∃ i j, i + j = k ∧ heaps_similar (f i) (f j)) ∧ 
  (∃ k, f k = n) :=
by
  sorry

end combine_heaps_l223_223128


namespace a_n_formula_T_n_sum_inequality_l223_223790

variable (S : ℕ → ℚ) (a b T : ℕ → ℚ)

noncomputable theory 

-- Define the properties for sequence {a_n}
axiom a1 : a 1 = 1
axiom S_n_property : ∀ n, 2 * n * S (n + 1) - 2 * (n + 1) * S n = n^2 + n

-- Prove the general term formula for {a_n}
theorem a_n_formula (n : ℕ) :
  ∀ n, a n = n :=
sorry

-- Define the sequence {b_n} and the first n terms sum {T_n}
def b_n (n : ℕ) : ℚ := n / (2 * (n + 3) * S n)
def T_n (n : ℕ) : ℚ := ∑ i in finset.range n, b_n i

-- Prove the sum of the first n terms of {b_n}
theorem T_n_sum (n : ℕ) :
  T n = (5 / 12) - (2 * n + 5) / (2 * (n + 2) * (n + 3)) :=
sorry

-- Prove the inequality ∑_{k=2}^n 1/a_k^3 < 1/4 for n ≥ 2
theorem inequality (n : ℕ) (h : n ≥ 2) :
  ∑ k in finset.range n \ finset.range 2, (1 / a k ^ 3) < 1 / 4 :=
sorry

end a_n_formula_T_n_sum_inequality_l223_223790


namespace angle_B1KB2_l223_223465

-- Definitions and conditions based on the problem statement
variables (A B C B1 C1 B2 C2 K : Type) 
variables [Triangle ABC] -- Representation of a triangle with angles sum to 180°
variables [Angle (A B C) = 35] -- Angle A as 35 degrees
variables [Altitude (B1 B) (A C)] -- BB1 is an altitude
variables [Altitude (C1 C) (A B)] -- CC1 is an altitude
variables [Midpoint B2 A C] -- B2 is the midpoint of AC
variables [Midpoint C2 A B] -- C2 is the midpoint of AB
variables [Intersection K (B1 C2) (C1 B2)] -- Intersection point K of B1C2 and C1B2

-- Statement to be proved: angle B1KB2 is 75 degrees
theorem angle_B1KB2 : Angle (B1 K B2) = 75 :=
sorry

end angle_B1KB2_l223_223465


namespace max_unique_coin_sums_l223_223669

def coin_values : List ℕ := [1, 1, 1, 5, 5, 10, 10, 50]

def possible_sums : Finset ℕ := (Finset.filter (λ x, x ≠ 0)
 (Finset.map (Function.uncurry (+))
  (Finset.product (Finset.fromList coin_values) (Finset.fromList coin_values))))

theorem max_unique_coin_sums : possible_sums.card = 9 := by sorry

end max_unique_coin_sums_l223_223669


namespace expand_expression_l223_223757

theorem expand_expression (x : ℝ) : 16 * (2 * x + 5) = 32 * x + 80 :=
by
  sorry

end expand_expression_l223_223757


namespace find_cylinder_height_l223_223663

noncomputable def cylinder_radius : ℝ := 3
noncomputable def lateral_surface_area (h : ℝ) : ℝ := 6 * real.pi * h
noncomputable def total_surface_area (h : ℝ) : ℝ := 6 * real.pi * h + 2 * real.pi * (cylinder_radius ^ 2)

theorem find_cylinder_height
  (h : ℝ)
  (cond : lateral_surface_area h = 0.5 * total_surface_area h) :
  h = 3 :=
sorry

end find_cylinder_height_l223_223663


namespace union_A_B_range_of_a_l223_223003

-- Definitions of sets A, B, and C
def A : Set ℝ := { x | 3 ≤ x ∧ x ≤ 9 }
def B : Set ℝ := { x | 2 < x ∧ x < 5 }
def C (a : ℝ) : Set ℝ := { x | x > a }

-- Problem 1: Proving A ∪ B = { x | 2 < x ≤ 9 }
theorem union_A_B : A ∪ B = { x | 2 < x ∧ x ≤ 9 } :=
sorry

-- Problem 2: Proving the range of 'a' given B ∩ C = ∅
theorem range_of_a (a : ℝ) (h : B ∩ C a = ∅) : a ≥ 5 :=
sorry

end union_A_B_range_of_a_l223_223003


namespace value_of_A_l223_223924

theorem value_of_A : 
  let A := 15 * Real.tan (Real.pi / 4 - Real.pi / 180 * 1) * Real.tan (Real.pi / 4) * Real.tan (Real.pi / 4 + Real.pi / 180 * 1) in
  A = 15 :=
by 
  -- Use Lean statements and trigonometric identities here to show that
  -- A = 15 * tan(44) * tan(45) * tan(46) simplifies to 15.
  -- The proof steps are not included because the task requires only the statement.
  sorry

end value_of_A_l223_223924


namespace arrange_MISSISSIPPI_l223_223733

theorem arrange_MISSISSIPPI : 
  let total_letters := 11
  let count_I := 4
  let count_S := 4
  let count_P := 2
  let arrangements := 11.factorial / (count_I.factorial * count_S.factorial * count_P.factorial)
  in arrangements = 34650 :=
by
  have h1 : 11.factorial = 39916800 := by sorry
  have h2 : 4.factorial = 24 := by sorry
  have h3 : 2.factorial = 2 := by sorry
  have h4 : arrangements = 34650 := by sorry
  exact h4

end arrange_MISSISSIPPI_l223_223733


namespace car_A_catches_up_l223_223705

variables (v_A v_B t_A : ℝ)

-- The conditions of the problem
def distance : ℝ := 300
def time_car_B : ℝ := t_A + 2
def distance_eq_A : Prop := distance = v_A * t_A
def distance_eq_B : Prop := distance = v_B * time_car_B

-- The final proof problem: Car A catches up with car B 150 kilometers away from city B.
theorem car_A_catches_up (t_A > 0) (v_A > 0) (v_B > 0) :
  distance_eq_A ∧ distance_eq_B → 
  ∃ d : ℝ, d = 150 := 
sorry

end car_A_catches_up_l223_223705


namespace proposition_only_A_l223_223645

def is_proposition (statement : String) : Prop := sorry

def statement_A : String := "Red beans grow in the southern country"
def statement_B : String := "They sprout several branches in spring"
def statement_C : String := "I hope you pick more"
def statement_D : String := "For these beans symbolize longing"

theorem proposition_only_A :
  is_proposition statement_A ∧
  ¬is_proposition statement_B ∧
  ¬is_proposition statement_C ∧
  ¬is_proposition statement_D := 
sorry

end proposition_only_A_l223_223645


namespace machines_initially_producing_shirts_l223_223164

theorem machines_initially_producing_shirts :
  (∀ m : ℕ, (m * 2 = 32) ↔ m = 16) :=
by {
  -- Let the rate of production per machine be r.
  assume m : ℕ,
  -- Given that 8 machines produce 160 shirts in 10 minutes.
  have h1 : 8 * (160 / (8 * 10)) = 32, 
  { 
    calc
      8 * (160 / (8 * 10)) = 8 * 2 : by { simp, norm_num, }
      ... = 32 : by { refl, },
  },
  -- Simplify the above equation: 8 * r = 32.
  have h2 : 8 * 2 = 32, 
  { 
    calc
      8 * 2 = 16 : by { norm_num, }
      ... = 32 : by { refl, },
  },
  -- Divide by 2 on both sides: m = 16
  have m_value : m = 8, { 
    calc
      m = 32 / 2 : by { norm_num, },
  },
  -- Replace with machines.
  split,
  assume h,
  {
    have : 8 * (160 / (8 * 10)) = 32, from h,
    exact 16,
  },
  assume h : m = 16,
  {
    have h : 8 * 2 = 32,
    {
      calc
        m = 32 / 2 : by { norm_num, },
    },
    exact h  
  }
}

end machines_initially_producing_shirts_l223_223164


namespace trapezoid_circle_tangent_radius_l223_223501

theorem trapezoid_circle_tangent_radius 
  (AB CD : ℝ) (BC DA : ℝ) (r : ℝ)
  (h_trapezoid : AB = 6 ∧ BC = 5 ∧ DA = 5 ∧ CD = 4)
  (h_circles_radii : ∀ (A B C D : ℝ), (dist A B) = 3 ∧ (dist C D) = 2)
  (h_radius_eq : r = (-(60 : ℝ) + 48 * Real.sqrt 3) / 23) :
  let k : ℕ := 60
  let m : ℕ := 48
  let n : ℕ := 3
  let p : ℕ := 23 in
  k + m + n + p = 134 := by
sorry

end trapezoid_circle_tangent_radius_l223_223501


namespace stones_equal_piles_l223_223955

theorem stones_equal_piles (n k : ℕ) (hdiv : ∃ nk_piles : Fin n → ℕ, (∑ i, nk_piles i) = n * k)
  (hoperation : ∀ (x y : ℕ), ∃ (x' y' : ℕ), x' = 2 * x ∧ y' = y - x) : 
  (∃ f : Fin n → ℕ, (∀ i, f i = k)) ↔ ∃ m : ℕ, k = 2 ^ m :=
by
  sorry

end stones_equal_piles_l223_223955


namespace solve_inequality_l223_223553

theorem solve_inequality :
  ∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 →
  (2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 15) ↔
  (x < 1 ∨ (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ (6 < x ∧ x < 8) ∨ (10 < x)).

end solve_inequality_l223_223553


namespace exists_centroid_l223_223499

structure Tree (V E : Type) :=
(vertices : set V)
(edges : set E)
(is_tree : ∀ {u v : V}, u ≠ v → (∃ (p : List V), p.head = u ∧ p.last = v ∧ ∀ i ∈ List.zip (p.init) (p.tail), (i.1, i.2) ∈ edges))

def is_centroid {V : Type} (A : Tree V (V × V)) (v : V) : Prop :=
∀ u ∈ A.vertices, u ≠ v → (∀ C ∈ (A.remove_vertex v).connected_components, C.size ≤ A.vertices.size / 2)

theorem exists_centroid (V E : Type) [fintype V] (A : Tree V (V × V)) :
    ∃ v ∈ A.vertices, is_centroid A v := sorry

end exists_centroid_l223_223499


namespace combined_weight_of_three_new_people_l223_223190

theorem combined_weight_of_three_new_people 
  (W : ℝ) 
  (h_avg_increase : (W + 80) / 20 = W / 20 + 4) 
  (h_replaced_weights : 60 + 75 + 85 = 220) : 
  220 + 80 = 300 :=
by
  sorry

end combined_weight_of_three_new_people_l223_223190


namespace value_of_x_minus_y_squared_l223_223857

theorem value_of_x_minus_y_squared (x y : ℝ) (hx : x^2 = 4) (hy : y^2 = 9) : 
  ((x - y)^2 = 1) ∨ ((x - y)^2 = 25) :=
sorry

end value_of_x_minus_y_squared_l223_223857


namespace sum_of_three_consecutive_odd_integers_l223_223629

theorem sum_of_three_consecutive_odd_integers (n : ℤ) 
  (h1 : n + (n + 4) = 130) 
  (h2 : n % 2 = 1) : 
  n + (n + 2) + (n + 4) = 195 := 
by
  sorry

end sum_of_three_consecutive_odd_integers_l223_223629


namespace fraction_simplification_l223_223178

theorem fraction_simplification : 
  (320 / 18) * (9 / 144) * (4 / 5) = 1 / 2 :=
by sorry

end fraction_simplification_l223_223178


namespace length_DC_l223_223759

theorem length_DC
  (ABCD_trapezoid : True)
  (AB_parallel_DC : True)
  (AB_eq_7 : True)
  (BC_eq_2_sqrt_5 : True)
  (angle_BCD_30 : True)
  (angle_CDA_45 : True) :
  (DC = 2 * sqrt 15 + 7 + 7 * sqrt 2 / 2) :=
  sorry

end length_DC_l223_223759


namespace sum_of_234_and_142_in_base_4_l223_223202

theorem sum_of_234_and_142_in_base_4 :
  (234 + 142) = 376 ∧ (376 + 0) = 256 * 1 + 64 * 1 + 16 * 3 + 4 * 2 + 1 * 0 :=
by sorry

end sum_of_234_and_142_in_base_4_l223_223202


namespace increasing_iff_positive_slope_l223_223812

variable {α β : Type}
variable [LinearOrder α] [OrderedAddCommMonoid β] {f : α → β}
variable {D : Set α}
variable {m n : α} [Interval : Set.Icc m n]

theorem increasing_iff_positive_slope (hD : Set.Icc m n ⊆ D) (h : ∀ x₁ x₂ ∈ Set.Ioo m n, x₁ ≠ x₂ → ((f x₁ - f x₂) / (x₁ - x₂) > (0:β))) :
  ∀ x₁ x₂ ∈ Set.Ioo m n, x₁ ≠ x₂ ↔ f x₁ > f x₂ :=
  sorry

end increasing_iff_positive_slope_l223_223812


namespace swept_lines_parabola_l223_223797

noncomputable def parabolaRegion : set (ℝ × ℝ) := 
  {p | ∃ x : ℝ, p = (x, x^2 / 4 + 1)}

theorem swept_lines_parabola {t : ℝ} :
  ∀ p : ℝ × ℝ, (∃ t : ℝ, p.1 = 1 + t ∧ p.2 = 1 + t ∧ p.1 = -1 + t ∧ p.2 = 1 - t) →
  p ∈ {p : ℝ × ℝ | p.2 ≤ p.1^2 / 4 + 1} :=
sorry

end swept_lines_parabola_l223_223797


namespace segment_AB_length_l223_223473

-- Define the conditions for the problem
def P := (1 : ℝ, 0 : ℝ)
def theta := ℝ
def t := ℝ
def line_l (t : ℝ) : ℝ × ℝ := (1 + 1/2 * t, (Real.sqrt 3 / 2) * t)
def ellipse_C (theta : ℝ) : ℝ × ℝ := (Real.cos theta, 2 * Real.sin theta)
def cartesian_eq_ellipse_C (x y : ℝ) := x^2 + (y^2 / 4) = 1

-- Translate the question to a proof statement
theorem segment_AB_length :
  ∃ (t1 t2 : ℝ), (line_l t1, line_l t2) ∈ ellipse_C theta ∧
  |t1 - t2| = 16/7 := 
sorry

end segment_AB_length_l223_223473


namespace cross_section_area_of_pyramid_l223_223984

-- Define the problem's known conditions and question
theorem cross_section_area_of_pyramid
  (a : ℝ)
  (α : ℝ)
  (hα : α ≤ Real.arctan (Real.sqrt 2 / 2)) :
  let area := (a^2 / 8) * (Real.cos α) * (Real.cot α) * (Real.tan (2 * α)) in
  cross_section_area_of_pyramid a α = area := 
sorry

end cross_section_area_of_pyramid_l223_223984


namespace same_range_of_shifted_functions_l223_223198

variable {X Y : Type} (f : X → Y)

theorem same_range_of_shifted_functions (f : X → Y) : 
    (range (λ x: X, f (x - 1))) = (range (λ x: X, f (x + 1))) :=
sorry

end same_range_of_shifted_functions_l223_223198


namespace combined_savings_after_four_weeks_l223_223537

-- Definitions based on problem conditions
def hourly_wage : ℕ := 10
def daily_hours : ℕ := 10
def days_per_week : ℕ := 5
def weeks : ℕ := 4

def robby_saving_ratio : ℚ := 2/5
def jaylene_saving_ratio : ℚ := 3/5
def miranda_saving_ratio : ℚ := 1/2

-- Definitions derived from the conditions
def daily_earnings : ℕ := hourly_wage * daily_hours
def total_working_days : ℕ := days_per_week * weeks
def monthly_earnings : ℕ := daily_earnings * total_working_days

def robby_savings : ℚ := robby_saving_ratio * monthly_earnings
def jaylene_savings : ℚ := jaylene_saving_ratio * monthly_earnings
def miranda_savings : ℚ := miranda_saving_ratio * monthly_earnings

def total_savings : ℚ := robby_savings + jaylene_savings + miranda_savings

-- The main theorem to prove
theorem combined_savings_after_four_weeks :
  total_savings = 3000 := by sorry

end combined_savings_after_four_weeks_l223_223537


namespace parabola_standard_equation_l223_223503

theorem parabola_standard_equation (p : ℝ) (hp : 0 < p) :
  (∀ D E : ℝ × ℝ, D = (2, 2 * real.sqrt p) →
    E = (2, -2 * real.sqrt p) →
    let O := (0, 0) in
    let OD := (2 - 0, 2 * real.sqrt p - 0) in
    let OE := (2 - 0, -2 * real.sqrt p - 0) in
    OD.1 * OE.1 + OD.2 * OE.2 = 0) →
  (∀ x y : ℝ, y ^ 2 = 2 * p * x →
  y ^ 2 = 2 * x) :=
by
  intros h1 h2 x y h3
  sorry

end parabola_standard_equation_l223_223503


namespace perimeter_of_square_l223_223975

-- Defining the context and proving the equivalence.
theorem perimeter_of_square (x y : ℕ) (h : Nat.gcd x y = 3) (area : ℕ) :
  let lcm_xy := Nat.lcm x y
  let side_length := Real.sqrt (20 * lcm_xy)
  let perimeter := 4 * side_length
  perimeter = 24 * Real.sqrt 5 :=
by
  let lcm_xy := Nat.lcm x y
  let side_length := Real.sqrt (20 * lcm_xy)
  let perimeter := 4 * side_length
  sorry

end perimeter_of_square_l223_223975


namespace minimum_area_triangle_line_MN_fixed_point_l223_223796

-- Definitions of conditions
variables {p m n k1 k2 λ : ℝ} (h_p : p > 0)
def parabola (x y : ℝ) := y^2 = 2 * p * x
def point_E := (m, n)
def line_through_E (k : ℝ) (x y : ℝ) := x = (1/k) * (y - n) + m

-- Problem 1: Minimum area of triangle EMN
theorem minimum_area_triangle (h_n : n = 0) (h_k1k2 : k1 * k2 = -1) :
  ∃ S, S = p^2 :=
sorry

-- Problem 2: Line MN passes through a fixed point
theorem line_MN_fixed_point (h_lambda : k1 + k2 = λ) (h_lambda_ne_zero : λ ≠ 0) :
  ∃ x_fixed y_fixed, x_fixed = m - n / λ ∧ y_fixed = p / λ ∧ 
  ∀ M N, midpoint M N := sorry

end minimum_area_triangle_line_MN_fixed_point_l223_223796


namespace total_time_for_steps_l223_223312

noncomputable def time_spent : ℕ := 
  let t1 := 45 -- Time for the first step
  let t2 := (2 / 3 : ℚ) * t1 -- Time for the second step
  let t3 := t1 + t2 -- Time for the third step
  let t4 := (3 / 2 : ℚ) * t1 -- Time for the fourth step
  let t5 := (t2 + t4) - 20 -- Time for the fifth step
  t1 + t2 + t3 + t4 + t5

theorem total_time_for_steps : time_spent = 295 := by
  -- The numerical steps from solution are omitted
  sorry

end total_time_for_steps_l223_223312


namespace frequency_of_fourth_group_relative_frequency_of_fourth_group_l223_223233

open Real

def sample_capacity : ℕ := 50
def freq1 : ℕ := 8
def freq2 : ℕ := 11
def freq3 : ℕ := 10
def freq5 : ℕ := 9

theorem frequency_of_fourth_group :
  let x := sample_capacity - (freq1 + freq2 + freq3 + freq5)
  in x = 12 :=
by
  sorry

theorem relative_frequency_of_fourth_group :
  let x := sample_capacity - (freq1 + freq2 + freq3 + freq5)
  in (x : ℝ) / sample_capacity = 0.24 :=
by
  sorry

end frequency_of_fourth_group_relative_frequency_of_fourth_group_l223_223233


namespace exponent_of_2_in_f3_div_g3_l223_223270

def f (n : ℕ) : ℕ := (List.range' 4 (n^2 - 4 + 1)).foldl (· * ·) 1

def g (n : ℕ) : ℕ := (List.range' 1 (n - 1 + 1)).foldl (λ prod x => prod * x^2) 1

theorem exponent_of_2_in_f3_div_g3 : (Nat.factorization (f 3 / g 3)).get 2 = 4 := by
  sorry

end exponent_of_2_in_f3_div_g3_l223_223270


namespace find_x_find_x_plus_y_l223_223009

-- Define the given vectors and the norm condition
def a (x : ℝ) : ℝ × ℝ × ℝ := (2, 4, x)
def b (y : ℝ) : ℝ × ℝ × ℝ := (2, y, 2)

-- Define the norm condition for vector a
def norm_condition (x : ℝ) : Prop :=
  real.sqrt (2^2 + 4^2 + x^2) = 6

-- Define the condition for vectors a and b to be parallel
def parallel_condition (x y : ℝ) : Prop :=
  (2 / 2 = y / 4) ∧ (y / 4 = x / 2)

-- The Lean 4 statements representing the proof problems
theorem find_x (x : ℝ) (h : norm_condition x) :
  x = 4 ∨ x = -4 :=
sorry

theorem find_x_plus_y (x y : ℝ) (h1 : parallel_condition x y) (h2 : x = 4) (h3 : y = 2) :
  x + y = 6 :=
sorry

end find_x_find_x_plus_y_l223_223009


namespace Aleesia_weeks_l223_223689

theorem Aleesia_weeks (w : ℕ) :
  (1.5 * w) + (2.5 * 8) = 35 → w = 10 :=
by
  intro h
  sorry

end Aleesia_weeks_l223_223689


namespace cheesecake_factory_savings_l223_223541

noncomputable def combined_savings : ℕ := 3000

theorem cheesecake_factory_savings :
  let hourly_wage := 10
  let daily_hours := 10
  let working_days := 5
  let weekly_hours := daily_hours * working_days
  let weekly_salary := weekly_hours * hourly_wage
  let robby_savings := (2/5 : ℚ) * weekly_salary
  let jaylen_savings := (3/5 : ℚ) * weekly_salary
  let miranda_savings := (1/2 : ℚ) * weekly_salary
  let combined_weekly_savings := robby_savings + jaylen_savings + miranda_savings
  4 * combined_weekly_savings = combined_savings :=
by
  sorry

end cheesecake_factory_savings_l223_223541


namespace value_of_b_minus_a_l223_223601

theorem value_of_b_minus_a (a b : ℕ) (h1 : a * b = 2 * (a + b) + 1) (h2 : b = 7) : b - a = 4 :=
by
  sorry

end value_of_b_minus_a_l223_223601


namespace plane_through_O_maps_to_itself_plane_not_containing_O_maps_to_sphere_sphere_containing_O_maps_to_plane_l223_223272

-- Define the center and radius of the inversion sphere.
variable (O : Point)
variable (R : Real)

-- Define Plane and Sphere types and the inversion transformation.
structure Plane where
  normal : Vector
  point_on_plane : Point

structure Sphere where
  center : Point
  radius : Real

def inversion (O : Point) (R : Real) (P : Point) : Point :=
  sorry -- The specific transformation formula would be here

-- a) Prove that a plane passing through the center O maps to itself under inversion.
theorem plane_through_O_maps_to_itself (plane : Plane) (h : plane.point_on_plane = O) :
  inversion O R plane = plane := sorry

-- b) Prove that a plane not containing the center O maps to a sphere passing through O under inversion.
theorem plane_not_containing_O_maps_to_sphere (plane : Plane) (h : plane.point_on_plane ≠ O) :
  ∃ sphere : Sphere, inversion O R plane = sphere ∧ sphere.center = O := sorry

-- c) Prove that a sphere containing the center O maps to a plane not containing O under inversion.
theorem sphere_containing_O_maps_to_plane (sphere : Sphere) (h : sphere.center = O) :
  ∃ plane : Plane, inversion O R sphere = plane ∧ plane.point_on_plane ≠ O := sorry

end plane_through_O_maps_to_itself_plane_not_containing_O_maps_to_sphere_sphere_containing_O_maps_to_plane_l223_223272


namespace find_a_values_l223_223584

theorem find_a_values (a : ℝ) : 
  (∃ x : ℝ, (a * x^2 + (a - 3) * x + 1 = 0)) ∧ 
  (∀ x1 x2 : ℝ, (a * x1^2 + (a - 3) * x1 + 1 = 0 ∧ a * x2^2 + (a - 3) * x2 + 1 = 0 → x1 = x2)) 
  ↔ a = 0 ∨ a = 1 ∨ a = 9 :=
sorry

end find_a_values_l223_223584


namespace evaluate_at_3_l223_223779

def f (x : ℕ) : ℕ := x ^ 2

theorem evaluate_at_3 : f 3 = 9 :=
by
  sorry

end evaluate_at_3_l223_223779


namespace cross_section_area_proof_l223_223592

noncomputable def cross_section_area (H α : ℝ) : ℝ :=
  (1 / 2) * H^2 * real.sqrt 3 * real.cot α * real.sqrt (1 + 16 * (real.cot α)^2)

theorem cross_section_area_proof (H α : ℝ) :
  let area := (1 / 2) * H^2 * real.sqrt 3 * real.cot α * real.sqrt (1 + 16 * (real.cot α)^2)
  in area =
     (1 / 2) * H^2 * real.sqrt 3 * real.cot α * real.sqrt (1 + 16 * (real.cot α)^2) := 
by
  sorry

end cross_section_area_proof_l223_223592


namespace find_smallest_n_l223_223766

def smallest_n_ensuring_relatively_prime (S : Set ℕ) (n : ℕ) : ℕ :=
  if ∃ T ⊆ S, T.card = n ∧ ∀ x y ∈ T, Nat.gcd x y ≠ 1 then
    ⊤
  else
    n

theorem find_smallest_n : smallest_n_ensuring_relatively_prime {x | 1 ≤ x ∧ x ≤ 2004} 1003 = 1003 :=
by
  sorry

end find_smallest_n_l223_223766


namespace complicated_expression_value_l223_223768

noncomputable def complicated_expression : ℝ :=
  1 / (3 + 1 / (3 + 1 / (3 - 1 / (3 + 1 / (2 * (3 + 2 / (5 * Real.sqrt 2)))))))

theorem complicated_expression_value :
  complicated_expression ≈ 0.29645 :=
by
  sorry

end complicated_expression_value_l223_223768


namespace max_sin_a_l223_223907

theorem max_sin_a (a b : ℝ) (h : Real.sin (a + b) = Real.sin a + Real.sin b) : Real.sin a ≤ 1 :=
by
  sorry

end max_sin_a_l223_223907


namespace score_difference_average_score_sports_elite_title_l223_223871

def passing_score : ℤ := 140
def scores : List ℤ := [-25, 17, 23, 0, -39, -11, 9, 34]
def total_points (scores : List ℤ) : ℤ := (scores.filter (λ x, x > 0)).sum * 2 - (scores.filter (λ x, x < 0)).sum

theorem score_difference : 
  List.maximum scores - List.minimum scores = 73 := 
by sorry

theorem average_score (scores : List ℤ) : 
  (passing_score:ℤ) + scores.sum / 8 = 141 := 
by sorry

theorem sports_elite_title (scores : List ℤ) : 
  total_points scores < 100 :=
by sorry

end score_difference_average_score_sports_elite_title_l223_223871


namespace volume_of_tetrahedron_l223_223500

variables {a b c x y z : ℕ}

def mutually_perpendicular (a b c : ℕ) : Prop :=
  true -- Placeholder for actual perpendicular condition

def area_of_triangles (a b c : ℕ) (x y z : ℕ) : Prop :=
  x = (1/2) * a * b ∧ y = (1/2) * b * c ∧ z = (1/2) * c * a

theorem volume_of_tetrahedron (a b c x y z : ℕ)
  (h_perp : mutually_perpendicular a b c)
  (h_area : area_of_triangles a b c x y z) :
  volume a b c x y z = 8 * x * y * z / (a * b * c) :=
sorry

end volume_of_tetrahedron_l223_223500


namespace find_angle_l223_223464

open Classical

variable {α : Type*} [LinearOrderedField α]

structure Triangle (α : Type*) [LinearOrderedField α] :=
(A B C : α × α)

structure Point (α : Type*) [LinearOrderedField α] :=
(x : α) (y : α)

def midpoint {α : Type*} [LinearOrderedField α] (P Q : Point α) : Point α :=
⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2⟩

def is_acute (T : Triangle α) : Prop :=
(T.A.1 < 90) ∧ (T.B.1 < 90) ∧ (T.C.1 < 90)

def altitude {α : Type*} [LinearOrderedField α] (P Q : Point α) (R : Point α) : Point α := sorry

noncomputable def intersection {α : Type*} [LinearOrderedField α] (l1 l2 : Point α → Prop) : Point α := sorry

theorem find_angle
  (T : Triangle α)
  (acute_T : is_acute T)
  (angle_A : T.A.1 = 35)
  (B1 : Point α)
  (C1 : Point α)
  (B2 : Point α := midpoint T.A T.C)
  (C2 : Point α := midpoint T.A T.B)
  (K : Point α := intersection (line B1 C2) (line C1 B2))
  (alt_BB1 : B1 = altitude T.B T.C T.A)
  (alt_CC1 : C1 = altitude T.C T.B T.A) :
  angle B1 K B2 = 75 :=
sorry

end find_angle_l223_223464


namespace tangent_lines_l223_223368

noncomputable def curve1 (x : ℝ) : ℝ := 2 * x ^ 2 - 5
noncomputable def curve2 (x : ℝ) : ℝ := x ^ 2 - 3 * x + 5

theorem tangent_lines :
  (∃ (m₁ b₁ m₂ b₂ : ℝ), 
    (∀ x y, y = -20 * x - 55 ∨ y = -13 * x - 20 ∨ y = 8 * x - 13 ∨ y = x + 1) ∧ 
    (
      (m₁ = 4 * 2 ∧ b₁ = 3) ∨ 
      (m₁ = 2 * -5 - 3 ∧ b₁ = 45) ∨
      (m₂ = 4 * -5 ∧ b₂ = 45) ∨
      (m₂ = 2 * 2 - 3 ∧ b₂ = 3)
    )) :=
sorry

end tangent_lines_l223_223368


namespace correct_statements_l223_223810

/-- Define the quadratic function and related entities -/
def quadratic_function (x : ℕ) : ℕ := x^2 - 2 * x + 1

/-- Point N lies on the curve y = x^2 - 2x + 1 and x_n = n -/
structure PointN (n : ℕ) where
  x : ℕ := n
  y : ℕ := quadratic_function n

/-- Define A_n and B_n -/
def A (n : ℕ) : ℕ := n + (n + 1)^2

def B (n : ℕ) : ℕ := (A n) % 10

/-- Define correctness of given statements -/
def correctness_of_statements :=
  let st_2 := (1 + 2 + 3 + 4 = ((1 - 1)^2 - 1^2 + 2^2 - 3^2 + 4^2)) in
  let st_3 := (∑ n in Finset.range 2022, 1 / (A (n + 1))) = 2022 / 2023 in
  st_2 ∧ st_3
  
/-- Main theorem to prove correctness of statements 2 and 3 -/
theorem correct_statements : correctness_of_statements := by
  sorry

end correct_statements_l223_223810


namespace ratio_c_b_l223_223479

theorem ratio_c_b {A B C a b c : ℝ} (hA : A = 2 * real.pi / 3) (h_a_squared : a^2 = 2 * b * c + 3 * c^2) :
  c / b = 1 / 2 :=
sorry

end ratio_c_b_l223_223479


namespace trapezoid_area_l223_223999

variable (x y : ℝ)

def condition1 : Prop := abs (y - 3 * x) ≥ abs (2 * y + x) ∧ -1 ≤ y - 3 ∧ y - 3 ≤ 1

def condition2 : Prop := (2 * y + y - y + 3 * x) * (2 * y + x + y - 3 * x) ≤ 0 ∧ 2 ≤ y ∧ y ≤ 4

theorem trapezoid_area (h1 : condition1 x y) (h2 : condition2 x y) :
  let A := (3, 2)
  let B := (-1/2, 2)
  let C := (-1, 4)
  let D := (6, 4)
  let S := (1/2) * (2 * (7 + 3.5))
  S = 10.5 :=
sorry

end trapezoid_area_l223_223999


namespace f_eight_l223_223823

noncomputable def f : ℝ → ℝ := sorry -- Defining the function without implementing it here

axiom f_x_neg {x : ℝ} (hx : x < 0) : f x = Real.log (-x) + x
axiom f_symmetric {x : ℝ} (hx : -Real.exp 1 ≤ x ∧ x ≤ Real.exp 1) : f (-x) = -f x
axiom f_periodic {x : ℝ} (hx : x > 1) : f (x + 2) = f x

theorem f_eight : f 8 = 2 - Real.log 2 := 
by
  sorry

end f_eight_l223_223823


namespace smallest_positive_integer_n_l223_223510

noncomputable def a : Real := Real.pi / 2023

def f (n : ℕ) : Real :=
  2 * (Finset.range (n + 1)).sum (λ k, Real.cos ((k+1)^2 * a) * Real.sin ((k+1) * a))

theorem smallest_positive_integer_n (n : ℕ) (h : ∀ m < n, ¬ 2 * (Finset.range (m + 1)).sum (λ k, Real.cos ((k+1)^2 * a) * Real.sin ((k+1) * a)).is_integer)
: n = 289 :=
  sorry

end smallest_positive_integer_n_l223_223510


namespace election_cases_l223_223660

def candidates : Finset String := {"Jungkook", "Jimin", "Yoongi"}

theorem election_cases (h : candidates.card = 3) : 
  let pres_choices := 3,
      vp_choices := 2
  in pres_choices * vp_choices = 6 :=
by
  sorry

end election_cases_l223_223660


namespace hens_and_cows_total_feet_l223_223292

noncomputable def total_feet (H C : ℕ) : ℕ :=
  (H * 2) + (C * 4)

theorem hens_and_cows_total_feet :
  ∀ (H C : ℕ), H = 24 ∧ H + C = 48 → total_feet H C = 144 :=
by
  intros H C hc
  cases hc with h1 h2
  subst h1
  have hc2 : C = 48 - 24 := by linarith
  rw hc2
  exact rfl


end hens_and_cows_total_feet_l223_223292


namespace sum_of_transformed_numbers_l223_223226

-- Define the variables and condition
variables (x y S : ℝ)
hypothesis (h1 : x + y = S)

-- State the theorem: the sum of the final two numbers
theorem sum_of_transformed_numbers : 3 * (x + 5) + 3 * (y + 5) = 3 * S + 30 :=
by
  sorry

end sum_of_transformed_numbers_l223_223226


namespace area_of_quadrilateral_formed_by_dividing_unit_square_into_5_equal_parts_l223_223341

noncomputable def divide_square (n : ℕ) : fin n → ℝ := 
  λ k, k / (n : ℝ)

noncomputable def points_of_square_division (n : ℕ) : list (ℝ × ℝ) :=
  [ (divide_square n 2, 0),
    (1, divide_square n 2),
    (1 - divide_square n 2, 1),
    (0, 1 - divide_square n 2) ]

theorem area_of_quadrilateral_formed_by_dividing_unit_square_into_5_equal_parts :
  let E := (2 / 5, 0)
  let F := (1, 2 / 5)
  let G := (1 - 2 / 5, 1)
  let H := (0, 1 - 2 / 5) 
  (abs ((E.1 - G.1) * (F.2 - H.2) - (E.2 - G.2) * (F.1 - H.1)) / 2) = 9 / 29 :=
sorry

end area_of_quadrilateral_formed_by_dividing_unit_square_into_5_equal_parts_l223_223341


namespace inclination_angle_line_l223_223968

theorem inclination_angle_line (x y : ℝ) : x - y + 3 = 0 → real.angle (1, - 1) = 45 :=
begin
  sorry
end

end inclination_angle_line_l223_223968


namespace fill_time_with_leak_l223_223154

theorem fill_time_with_leak (A L : ℝ) (hA : A = 1 / 5) (hL : L = 1 / 10) :
  1 / (A - L) = 10 :=
by 
  sorry

end fill_time_with_leak_l223_223154


namespace sin_max_value_l223_223909

open Real

theorem sin_max_value (a b : ℝ) (h : sin (a + b) = sin a + sin b) :
  sin a ≤ 1 :=
by
  sorry

end sin_max_value_l223_223909


namespace greatest_prime_factor_147_l223_223625

theorem greatest_prime_factor_147 : ∀ (n : ℕ), n = 147 → ∃ (p : ℕ), prime p ∧ p ∣ n ∧ ∀ q, prime q ∧ q ∣ n → q ≤ p :=
by
  intros n h
  rw h at *
  have h₁ : 147 = 3 * 7 * 7 := by norm_num
  have h₂ : prime 7 := by norm_num
  unfold prime at h₂
  use 7
  split
  apply h₂
  split
  use 21
  split
  norm_num
  intros q hq
  cases hq with hq₁ hq₂
  cases hq₂ with k hk
  rw ← hk at hq₁
  norm_num at hq₁,
  exact hq₁.left,
  exact sorry

end greatest_prime_factor_147_l223_223625


namespace regions_divided_by_chords_l223_223508

open Function

theorem regions_divided_by_chords (P : Finset ℤ) (hP : P.card = 20) 
  (h_non_concurrent : ∀ {a b c d : Finset ℤ}, {a, b, c, d} ⊆ P → ({a, b}, {c, d} ∈ P ∧ {a, c}, {b, d} ∈ P)
    → IsLinearIndep ℝ ![(a : ℝ), (b : ℝ)] ([(c : ℝ), (d : ℝ)])) :
  let V := 20 + (Finset.card (Finset.powersetLen 4 P))
      E := (20 * 21 + 4 * (Finset.card (Finset.powersetLen 4 P))) / 2
  in (E - V + 2 - 1) = 5036 :=
by
  sorry

end regions_divided_by_chords_l223_223508


namespace correct_cd_value_l223_223632

noncomputable def repeating_decimal (c d : ℕ) : ℝ :=
  1 + c / 10.0 + d / 100.0 + (c * 10 + d) / 990.0

theorem correct_cd_value (c d : ℕ) (h : (c = 9) ∧ (d = 9)) : 90 * (repeating_decimal 9 9 - (1 + 9 / 10.0 + 9 / 100.0)) = 0.9 :=
by
  sorry

end correct_cd_value_l223_223632


namespace pile_splitting_l223_223108

theorem pile_splitting (single_stone_piles : ℕ) :
  ∃ (final_heap_size : ℕ), 
    (∀ heap_size ≤ single_stone_piles, heap_size > 0 → (heap_size * 2) ≥ heap_size) ∧ (final_heap_size = single_stone_piles) :=
by
  sorry

end pile_splitting_l223_223108


namespace interval_decreasing_l223_223407

open Real

noncomputable def phi_value (A : ℝ) (hA : A > 0) : ℝ :=
  if P : (P = (-1 : ℝ, 1 : ℝ)) ∧ (0 < ?m_1) ∧ (?m_1 < π)
    then ∃ (φ : ℝ), φ = 3 * π / 4
    else 0

theorem interval_decreasing (A : ℝ) (hA : A > 0) :
  ∃ I : set ℝ, ∀ x ∈ I, f(x) = A * sin(2 * x + 3 * π / 4) ∧ f'(x) < 0 ∧
  (I = {x : ℝ | ∃ k : ℤ, -π / 8 + k * π ≤ x ∧ x ≤ 3 * π / 8 + k * π}) :=
sorry

end interval_decreasing_l223_223407


namespace warriors_won_40_games_l223_223206

variable (H F W K R S : ℕ)

-- Conditions as given in the problem
axiom hawks_won_more_games_than_falcons : H > F
axiom knights_won_more_than_30 : K > 30
axiom warriors_won_more_than_knights_but_fewer_than_royals : W > K ∧ W < R
axiom squires_tied_with_falcons : S = F

-- The proof statement
theorem warriors_won_40_games : W = 40 :=
sorry

end warriors_won_40_games_l223_223206


namespace geom_series_first_term_l223_223223

theorem geom_series_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 30)
  (h2 : a^2 / (1 - r^2) = 120) : 
  a = 120 / 17 :=
by
  sorry

end geom_series_first_term_l223_223223


namespace possible_U_values_l223_223621

def validU_and_valid_sum (U ЛАН ДЭ : ℕ) : Prop :=
  (U * (ЛАН + ДЭ) = 2020) ∧ (U > 0) ∧ (U < 10) ∧ (ЛАН < 1000) ∧ (ДЭ < 100)

theorem possible_U_values : 
  ∃ (U : ℕ), (U = 2 ∨ U = 5) ∧ ∃ (ЛАН ДЭ : ℕ), validU_and_valid_sum U ЛАН ДЭ :=
by sorry

end possible_U_values_l223_223621


namespace sum_of_roots_eq_a_plus_b_l223_223854

theorem sum_of_roots_eq_a_plus_b (a b : ℝ) 
  (h : ∀ x : ℝ, x^2 - (a + b) * x + (ab + 1) = 0 → (x = a ∨ x = b)) :
  a + b = a + b :=
by sorry

end sum_of_roots_eq_a_plus_b_l223_223854


namespace tree_2008_coordinates_l223_223674

def x (n: ℕ) : ℕ := 
  if n = 1 then 
    1 
  else 
    x (n - 1) + 1 - 5 * (n - 1) / 5 + 5 * (n - 2) / 5

def y (n : ℕ) : ℕ := 
  if n = 1 then 
    1 
  else 
    y (n - 1) + (n - 1) / 5 - (n - 2) / 5

theorem tree_2008_coordinates :
  x 2008 = 3 ∧ y 2008 = 402 :=
by 
  sorry

end tree_2008_coordinates_l223_223674


namespace sum_of_binomial_coefficients_l223_223805

noncomputable theory

def binomial_coeff (n k : ℕ) : ℕ :=
  nat.choose n k

def calc_integral : ℝ :=
  ∫ x with x in 0..2, (1 - 3 * x ^ 2) + 4

def is_coeff_third_term (coeff : ℕ) : Prop :=
  ∃ n, binomial_coeff n 2 = coeff

theorem sum_of_binomial_coefficients :
  ∫ x with x in 0..2, (1 - 3 * x ^ 2) + 4 = (14 / 3 : ℝ) →
  is_coeff_third_term 15 →
  ∑ (k : ℕ) in finset.range (n + 1), binomial_coeff 6 k = 64 :=
sorry

end sum_of_binomial_coefficients_l223_223805


namespace inequality_proof_l223_223158

theorem inequality_proof (x1 x2 y1 y2 z1 z2 : ℝ) 
  (hx1 : x1 > 0) (hx2 : x2 > 0) (hy1 : y1 > 0) (hy2 : y2 > 0)
  (hx1y1_pos : x1 * y1 - z1^2 > 0) (hx2y2_pos : x2 * y2 - z2^2 > 0) :
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2)^2) ≤ 
    1 / (x1 * y1 - z1^2) + 1 / (x2 * y2 - z2^2) :=
by
  sorry

end inequality_proof_l223_223158


namespace number_of_positive_integers_l223_223765

open Int

theorem number_of_positive_integers :
  (∃ n : Nat, highest_power_of_prime_dividing_factorial 7 n = 8) = 7 :=
sorry

end number_of_positive_integers_l223_223765


namespace carnations_more_than_tulips_l223_223229

theorem carnations_more_than_tulips:
  (carnations tulips : ℕ) (h1 : carnations = 13) (h2 : tulips = 7) : 
  carnations - tulips = 6 :=
by
  { intros, rw [h1, h2], exact (13 - 7) }
  sorry

end carnations_more_than_tulips_l223_223229


namespace area_of_T_is_3_l223_223952

-- Definitions based on the problem conditions.
def alpha : ℂ := 1 / 2 + (1 / 2) * complex.I * real.sqrt 3
def T (x y z : ℕ) : set ℂ :=
  {z | ∃ (x y z : ℕ), (0 ≤ x ∧ x ≤ 2) ∧ (0 ≤ y ∧ y ≤ 2) ∧ (0 ≤ z ∧ z ≤ 2) ∧ z = x + y * alpha + z * alpha^2}

-- The theorem statement asserting that the area of T is 3.
theorem area_of_T_is_3 : ∀ (T : set ℂ), (∃ x y z, T = {z | ∃ (x y z : ℕ), x + y * alpha + z * alpha^2}) → area T = 3 :=
by
  sorry

end area_of_T_is_3_l223_223952


namespace problem_l223_223451

open Real

noncomputable def ZL : ℝ := (1 - sqrt 3) / 2

theorem problem {S L I N B Z : Type*} [metric_space S]
  (angle_LIS : ℝ)
  (angle_SIL : ℝ)
  (LI : ℝ)
  (N_midpoint : LI / 2)
  (LB_perp_SN : Type*)
  (BZ_eq_ZS : ℝ) :
  angle_LIS = π / 3 ∧
  angle_SIL = 5 * π / 12 ∧
  LI = 1 ∧
  N_midpoint = 1 / 2 ∧
  perpendicular LB_perp_SN (line_through S N) ∧
  length BZ = length ZS →
  ZL = (1 - sqrt 3) / 2 := 
sorry

end problem_l223_223451


namespace deductive_reasoning_l223_223549

-- Assume conditions
axiom irrational_numbers_non_terminating : ∀ x, irrational x → non_terminating_decimal x
axiom pi_irrational : irrational π

-- Define non-terminating decimal
def non_terminating_decimal (x : ℝ) : Prop := sorry -- Placeholder for the precise mathematical definition

-- Define irrational number
def irrational (x : ℝ) : Prop := sorry -- Placeholder for the precise mathematical definition

-- Define the statement to be proved
theorem deductive_reasoning :
  (∀ x, irrational x → non_terminating_decimal x) →
  (irrational π) →
  (non_terminating_decimal π) →
  (reasoning_type == deductive) :=
by
  intros
  sorry

-- Define the reasoning type
inductive reasoning_type
| deductive
| inductive
| analogical
| plausibility

end deductive_reasoning_l223_223549


namespace min_value_fraction_l223_223435

theorem min_value_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 1) : 
  (1 / x) + (1 / (3 * y)) ≥ 3 :=
sorry

end min_value_fraction_l223_223435


namespace amount_spent_per_sibling_l223_223521

-- Definitions and conditions
def total_spent := 150
def amount_per_parent := 30
def num_parents := 2
def num_siblings := 3

-- Claim
theorem amount_spent_per_sibling :
  (total_spent - (amount_per_parent * num_parents)) / num_siblings = 30 :=
by
  sorry

end amount_spent_per_sibling_l223_223521


namespace correct_proposition_is_4_l223_223819

-- Define the propositions as per the conditions
def prop1 : Prop := (¬ ((x^2 = 1) → (x ≠ 1)))
def prop2 : Prop := (¬ (x = -1 → ¬(x^2 - 5*x - 6 = 0)))
def prop3 : Prop := (¬ (∃ x : ℝ, x^2 + x - 1 < 0) = (∀ x : ℝ, x^2 + x - 1 > 0))
def prop4 : Prop := (∀ x y : ℝ, (x = y) → (sin x = sin y))

-- Problem statement: Prove that only prop4 is correct
theorem correct_proposition_is_4 : ¬prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ prop4 := 
by {
  sorry
}

end correct_proposition_is_4_l223_223819


namespace permutations_mississippi_l223_223751

theorem permutations_mississippi : 
  let total_letters := 11
  let m_count := 1
  let i_count := 4
  let s_count := 4
  let p_count := 2
  (Nat.factorial total_letters / (Nat.factorial m_count * Nat.factorial i_count * Nat.factorial s_count * Nat.factorial p_count)) = 34650 := 
by
  sorry

end permutations_mississippi_l223_223751


namespace gcd_lcm_product_l223_223358

theorem gcd_lcm_product (a b : ℕ) (h_a : a = 24) (h_b : b = 60) : 
  Nat.gcd a b * Nat.lcm a b = 1440 := by 
  rw [h_a, h_b]
  apply Nat.gcd_mul_lcm
  sorry

end gcd_lcm_product_l223_223358


namespace sum_of_original_numbers_l223_223939

theorem sum_of_original_numbers :
  ∃ a b : ℚ, a = b + 12 ∧ a^2 + b^2 = 169 / 2 ∧ (a^2)^2 - (b^2)^2 = 5070 ∧ a + b = 5 :=
by
  sorry

end sum_of_original_numbers_l223_223939


namespace solve_radius_l223_223482

noncomputable theory

def radius_of_spheres_in_cone (R H : ℕ) (r : ℚ) :=
  (R = 4 ∧ H = 15 ∧ 
  ∃ spheres : ℕ → bool, 
  (∀ i, i < 3 → spheres i = tt) ∧
  (∀ i j, i < 3 ∧ j < 3 ∧ i ≠ j → tangential spheres i spheres j) ∧
  touches_apex spheres 0 ∧ 
  touches_base spheres ∧ 
  touches_side spheres) →
  r = 15 / 4

axiom tangential : ∀ (s1 s2 : bool), bool
axiom touches_apex : ∀ (spheres : ℕ → bool) (i : ℕ), bool
axiom touches_base : ∀ (spheres: ℕ → bool), bool
axiom touches_side : ∀ (spheres: ℕ → bool), bool

theorem solve_radius : radius_of_spheres_in_cone 4 15 (15 / 4) :=
by {
  sorry
}

end solve_radius_l223_223482


namespace probability_intersection_probability_a_minus_b_in_union_l223_223837

-- Given conditions
def A : Set ℝ := { x | -4 < x ∧ x < 1 }
def B : Set ℝ := { x | -2 < x ∧ x < 4 }
def interval : Set ℝ := { x | -4 < x ∧ x < 5 }

-- Mathematical statement to prove for Problem 1
theorem probability_intersection : 
  (∃ x ∈ interval, x ∈ A ∩ B) ∧ (interval ≠ ∅) → 
  (↑(Set.cardinality (A ∩ B) / Set.cardinality interval) : ℝ) = 1/3 :=
sorry

-- Mathematical statement to prove for Problem 2
theorem probability_a_minus_b_in_union :
  {a : ℤ | a ∈ A ∧ a ∈ finite_interval ∧ ∀ b : ℤ, b ∈ B ∧ b ∈ finite_interval → 
  (a - b) ∈ A ∪ B} → 
  (↑(Set.cardinality { (a, b) | a ∈ A ∧ b ∈ B ∧ (a - b) ∈ A ∪ B } / 
  Set.cardinality { (a, b) | a ∈ A ∧ b ∈ B }) : ℝ) = 7/10 :=
sorry

end probability_intersection_probability_a_minus_b_in_union_l223_223837


namespace cube_side_ratio_l223_223594

theorem cube_side_ratio (a b : ℝ) (h : (6 * a^2) / (6 * b^2) = 36) : a / b = 6 :=
by
  sorry

end cube_side_ratio_l223_223594


namespace find_pages_revised_twice_l223_223212

def pages_revised_twice (total_pages : ℕ) (pages_revised_once : ℕ) (cost_first_time : ℕ) (cost_revised_once : ℕ) (cost_revised_twice : ℕ) (total_cost : ℕ) :=
  ∃ (x : ℕ), 
    (total_pages - pages_revised_once - x) * cost_first_time
    + pages_revised_once * (cost_first_time + cost_revised_once)
    + x * (cost_first_time + cost_revised_once + cost_revised_once) = total_cost 

theorem find_pages_revised_twice :
  pages_revised_twice 100 35 6 4 4 860 ↔ ∃ x, x = 15 :=
by
  sorry

end find_pages_revised_twice_l223_223212


namespace profit_calculation_l223_223285

variable (price : ℕ) (cost : ℕ) (exchange_rate : ℕ) (profit_per_bottle : ℚ)

-- Conditions
def conditions := price = 2 ∧ cost = 1 ∧ exchange_rate = 5

-- Profit per bottle is 0.66 yuan considering the exchange policy
theorem profit_calculation (h : conditions price cost exchange_rate) : profit_per_bottle = 0.66 := sorry

end profit_calculation_l223_223285


namespace machine_pays_for_itself_in_36_days_l223_223485

def cost_of_machine : ℝ := 200
def discount : ℝ := 20
def cost_per_day : ℝ := 3
def previous_cost_per_coffee : ℝ := 4
def number_of_coffees_per_day : ℝ := 2

def net_cost_of_machine : ℝ := cost_of_machine - discount
def daily_savings : ℝ := number_of_coffees_per_day * previous_cost_per_coffee - cost_per_day

def days_until_machine_pays_for_itself : ℝ := net_cost_of_machine / daily_savings

theorem machine_pays_for_itself_in_36_days : days_until_machine_pays_for_itself = 36 := 
by {
  -- Proof skipped
  sorry
}

end machine_pays_for_itself_in_36_days_l223_223485


namespace range_of_a_l223_223002

variable P : Set ℝ := { x : ℝ | x ^ 2 ≤ 1 }
variable a : ℝ
def M : Set ℝ := {a}

theorem range_of_a : (P ∪ M = P) → a ∈ { x : ℝ | -1 ≤ x ∧ x ≤ 1 } :=
by
  intro h
  sorry

end range_of_a_l223_223002


namespace option_b_correct_l223_223281

-- Definitions of the conditions
def bag : ℕ × ℕ × ℕ := (3, 2, 1) -- (red, white, black)

def draw_two_balls_event_a (draw1 draw2 : nat) : Prop :=
  (draw1 = 1 ∨ draw2 = 1) ∧ (draw1 = 3 ∨ draw2 = 3) -- At least one white ball; At least one red ball

def draw_two_balls_event_b (draw1 draw2 : nat) : Prop :=
  (draw1 = 1 ∨ draw2 = 1) ∧ ((draw1 = 3 ∧ draw2 = 4) ∨ (draw1 = 4 ∧ draw2 = 3)) -- At least one white ball; One red ball and one black ball

def draw_two_balls_event_c (draw1 draw2 : nat) : Prop :=
  (draw1 = 1 ∧ draw2 ≠ 1) ∨ (draw2 = 1 ∧ draw1 ≠ 1) ∧ ((draw1 = 1 ∧ draw2 = 4) ∨ (draw1 = 4 ∧ draw2 = 1)) -- Exactly one white ball; One white ball and one black ball

def draw_two_balls_event_d (draw1 draw2 : nat) : Prop :=
  (draw1 = 1 ∨ draw2 = 1) ∧ (draw1 = 1 ∧ draw2 = 1) -- At least one white ball; Both are white balls

-- Proof goal definition
theorem option_b_correct :
  ∀ draw1 draw2 : nat,
  (draw_two_balls_event_b draw1 draw2 ∧
  (¬ (draw_two_balls_event_a draw1 draw2) ∧
  ¬ (draw_two_balls_event_c draw1 draw2) ∧
  ¬ (draw_two_balls_event_d draw1 draw2))) →
  true :=
sorry

end option_b_correct_l223_223281


namespace st_over_tu_l223_223888

noncomputable def problem (P Q R S T U : Type) 
  [AddCommGroup P] [VectorSpace ℝ P]
  (p q r s t u : P)
  (hS: s = (1 / 5 : ℝ) • p + (4 / 5 : ℝ) • q)
  (hT: t = (1 / 5 : ℝ) • q + (4 / 5 : ℝ) • r)
  (hU: u = 4 • t - s) : ℝ :=
  (1 / 3 : ℝ)

theorem st_over_tu (P Q R S T U : Type) 
  [AddCommGroup P] [VectorSpace ℝ P]
  (p q r s t u : P)
  (hS: s = (1 / 5 : ℝ) • p + (4 / 5 : ℝ) • q)
  (hT: t = (1 / 5 : ℝ) • q + (4 / 5 : ℝ) • r)
  (hU: u = 4 • t - s) : 
  problem P Q R S T U p q r s t u hS hT hU = 1 / 3 :=
sorry

end st_over_tu_l223_223888


namespace triangle_side_length_l223_223609

theorem triangle_side_length 
  (area : ℝ) (QR PQ : ℝ) (h : ℝ) (QN NR : ℝ) (y : ℝ)
  (area_given : area = 88)
  (QR_given : QR = 22)
  (PQ_given : PQ = 10)
  (h_eq : h = 8)
  (QN_eq : QN = 6)
  (NR_eq : NR = 16)
  (Pythagoras_left : PQ^2 = QN^2 + h^2)
  (Pythagoras_right : y^2 = h^2 + NR^2) : y = 8 * sqrt 5 :=
by
  sorry

end triangle_side_length_l223_223609


namespace max_primes_in_grid_l223_223156

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m ∣ n, m = 1 ∨ m = n

def distinct_digits (digits : List ℕ) : Prop :=
  digits.nodup

def no_leading_zero (n : ℕ) : Prop :=
  n / 10 ≠ 0

def adjacent_two_digit_primes (digits : List ℕ) (primes : List ℕ) :=
  ∀ p ∈ primes, is_prime p ∧ no_leading_zero p ∧ 
  (∃ a b, p = 10 * a + b ∧ a ≠ b ∧ a ∈ digits ∧ b ∈ digits)

theorem max_primes_in_grid : ∃ (primes : List ℕ), 
  adjacent_two_digit_primes [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] primes ∧
  primes.length = 7 :=
sorry

end max_primes_in_grid_l223_223156


namespace amount_after_two_years_l223_223344

theorem amount_after_two_years (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ)
  (hP : P = 76800) (hr : r = 1 / 8) (hn : n = 2) : A = 97200 :=
by 
  have compound_interest : A = P * (1 + r) ^ n,
  sorry

end amount_after_two_years_l223_223344


namespace distance_focus_to_line_l223_223400

theorem distance_focus_to_line :
  let ellipse := λ x y : ℝ, x^2 / 16 + y^2 / 4 = 1
  let line := λ x y : ℝ, x - y = 0
  let focus := (-2 * real.sqrt 3, 0)
  let distance := λ (p : ℝ × ℝ) (A B C : ℝ), (abs (A * p.1 + B * p.2 + C)) / real.sqrt (A ^ 2 + B ^ 2)
  distance focus 1 (-1) 0 = real.sqrt 6 :=
by
  sorry

end distance_focus_to_line_l223_223400


namespace split_stones_l223_223079

theorem split_stones (n : ℕ) :
  ∃ (heaps : list ℕ), (∀ h ∈ heaps, 1 ≤ h ∧ h ≤ n) ∧ (∀ i j, i ≠ j → (i < heaps.length ∧ j < heaps.length → heaps.nth i ≤ 2 * heaps.nth j)) :=
sorry

end split_stones_l223_223079


namespace split_piles_equiv_single_stone_heaps_l223_223097

theorem split_piles_equiv_single_stone_heaps (n : ℕ) (heaps : List ℕ) (h_initial : ∀ h ∈ heaps, h = 1)
  (h_size : heaps.length = n) :
  ∃ final_heap, (∀ x y ∈ heaps, x + y ≤ 2 * max x y) ∧ (List.sum heaps = (heaps.length) * 1) := by
  sorry

end split_piles_equiv_single_stone_heaps_l223_223097


namespace electricity_usage_l223_223600

theorem electricity_usage 
  (total_usage : ℕ) (saved_cost : ℝ) (initial_cost : ℝ) (peak_cost : ℝ) (off_peak_cost : ℝ) 
  (usage_peak : ℕ) (usage_off_peak : ℕ) :
  total_usage = 100 →
  saved_cost = 3 →
  initial_cost = 0.55 →
  peak_cost = 0.6 →
  off_peak_cost = 0.4 →
  usage_peak + usage_off_peak = total_usage →
  (total_usage * initial_cost - (peak_cost * usage_peak + off_peak_cost * usage_off_peak) = saved_cost) →
  usage_peak = 60 ∧ usage_off_peak = 40 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end electricity_usage_l223_223600


namespace split_into_similar_piles_l223_223087

def similar_sizes (x y : ℕ) : Prop := x ≤ 2 * y ∧ y ≤ 2 * x

theorem split_into_similar_piles (n : ℕ) (h : 0 < n) :
  ∃ (piles : list ℕ), (∀ x ∈ piles, x = 1) ∧ (list.sum piles = n) ∧
                       (∀ x y ∈ piles, similar_sizes x y) := 
sorry

end split_into_similar_piles_l223_223087


namespace find_ff_neg1_l223_223826

noncomputable def f : ℝ → ℝ :=
  λ x, if x < 0 then 2^(x + 2) else x^3

theorem find_ff_neg1 : f (f (-1)) = 8 := by
  sorry

end find_ff_neg1_l223_223826


namespace arrange_MISSISSIPPI_l223_223734

theorem arrange_MISSISSIPPI : 
  let total_letters := 11
  let count_I := 4
  let count_S := 4
  let count_P := 2
  let arrangements := 11.factorial / (count_I.factorial * count_S.factorial * count_P.factorial)
  in arrangements = 34650 :=
by
  have h1 : 11.factorial = 39916800 := by sorry
  have h2 : 4.factorial = 24 := by sorry
  have h3 : 2.factorial = 2 := by sorry
  have h4 : arrangements = 34650 := by sorry
  exact h4

end arrange_MISSISSIPPI_l223_223734


namespace monotonicity_intervals_when_a_equals_2_range_of_a_if_no_increasing_interval_on_1_to_3_l223_223075

def f (x a : ℝ) := log x + x^2 - 2 * a * x + a^2

-- First proof problem
theorem monotonicity_intervals_when_a_equals_2 :
  (∀ x > 0, f x 2 = log x + x^2 - 4 * x + 4) →
  (∀ x > 0, deriv (λ x, f x 2) x = (1/x) + 2*x - 4) →
  (∀ x > 0, (deriv (λ x, f x 2) x > 0 ↔ (0 < x ∧ x < (2 - sqrt 2) / 2) ∨ ((2 + sqrt 2) / 2 < x))) →
  (∀ x > 0, (deriv (λ x, f x 2) x < 0 ↔ (2 - sqrt 2) / 2 < x ∧ x < (2 + sqrt 2) / 2)) →
  ((∀ x > 0, increasing_on (λ x, f x 2) (Ioo 0 ((2 - sqrt 2) / 2))) ∧
   (∀ x > 0, decreasing_on (λ x, f x 2) (Ioo ((2 - sqrt 2) / 2) ((2 + sqrt 2) / 2))) ∧
   (∀ x > 0, increasing_on (λ x, f x 2) (Ioo ((2 + sqrt 2) / 2) ⊤))) :=
sorry

-- Second proof problem
theorem range_of_a_if_no_increasing_interval_on_1_to_3 :
  (∀ x > 0, (f x a = log x + x^2 - 2 * a * x + a^2)) →
  (∃ x ∈ Icc 1 3, deriv (λ x, f x a) x ≤ 0) →
  (a ≥ 19 / 6) :=
sorry

end monotonicity_intervals_when_a_equals_2_range_of_a_if_no_increasing_interval_on_1_to_3_l223_223075


namespace man_l223_223293

theorem man's_speed_with_current (v c : ℝ) (h1 : c = 4.3) (h2 : v - c = 12.4) : v + c = 21 :=
by {
  sorry
}

end man_l223_223293


namespace split_into_similar_piles_l223_223085

def similar_sizes (x y : ℕ) : Prop := x ≤ 2 * y ∧ y ≤ 2 * x

theorem split_into_similar_piles (n : ℕ) (h : 0 < n) :
  ∃ (piles : list ℕ), (∀ x ∈ piles, x = 1) ∧ (list.sum piles = n) ∧
                       (∀ x y ∈ piles, similar_sizes x y) := 
sorry

end split_into_similar_piles_l223_223085


namespace initial_amount_of_A_l223_223304

variable (a b c : ℕ)

-- Conditions
axiom condition1 : a - b - c = 32
axiom condition2 : b + c = 48
axiom condition3 : a + b + c = 128

-- The goal is to prove that A had 80 cents initially.
theorem initial_amount_of_A : a = 80 :=
by
  -- We need to skip the proof here
  sorry

end initial_amount_of_A_l223_223304


namespace scientific_notation_of_463_4_billion_l223_223524

theorem scientific_notation_of_463_4_billion :
  (463.4 * 10^9) = (4.634 * 10^11) := by
  sorry

end scientific_notation_of_463_4_billion_l223_223524


namespace least_possible_area_of_triangle_DEF_l223_223215

noncomputable def least_area_triangle
  (solutions : Finset ℂ)
  (h_solutions : ∀ z ∈ solutions, (z-4)^8 = 256)
  (h_polygon : is_convex_regular_polygon solutions)
  (D E F : ℂ)
  (h_D : D ∈ solutions)
  (h_E : E ∈ solutions)
  (h_F : F ∈ solutions)
  : ℝ :=
  let area := λ D E F : ℂ, complex.abs ((D - E) * (E - F) * (F - D)) / 2 in
  Finset.inf' { area D E F | D E F ∈ solutions } sorry

theorem least_possible_area_of_triangle_DEF
  (solutions : Finset ℂ)
  (h_solutions : ∀ z ∈ solutions, (z-4)^8 = 256)
  (h_polygon : is_convex_regular_polygon solutions)
  (D E F : ℂ)
  (h_D : D ∈ solutions)
  (h_E : E ∈ solutions)
  (h_F : F ∈ solutions)
  : least_area_triangle solutions h_solutions h_polygon D h_D E h_E F h_F = 4 :=
sorry

end least_possible_area_of_triangle_DEF_l223_223215


namespace total_gymnasts_l223_223369

theorem total_gymnasts (n : ℕ) : 
  (∃ (t : ℕ) (c : t = 4) (h : n * (n-1) / 2 + 4 * 6 = 595), n = 34) :=
by {
  -- skipping the detailed proof here, just ensuring the problem is stated as a theorem
  sorry
}

end total_gymnasts_l223_223369


namespace arithmetic_mean_calc_l223_223700

theorem arithmetic_mean_calc (x a : ℝ) (hx : x ≠ 0) (ha : a ≠ 0) :
  ( ( (x + a)^2 / x ) + ( (x - a)^2 / x ) ) / 2 = x + (a^2 / x) :=
sorry

end arithmetic_mean_calc_l223_223700


namespace solution_of_fraction_l223_223867

theorem solution_of_fraction (x : ℝ) (h1 : x^2 - 9 = 0) (h2 : x + 3 ≠ 0) : x = 3 :=
by
  sorry

end solution_of_fraction_l223_223867


namespace sum_of_numerator_and_denominator_l223_223493

noncomputable def time_until_can_see_each_other (distance_covered : ℕ) : ℕ := 
  200 / 3 

theorem sum_of_numerator_and_denominator : 
  let t := time_until_can_see_each_other 200 
  in (t.numerator + t.denominator) = 203 := 
sorry

end sum_of_numerator_and_denominator_l223_223493


namespace count_ordered_triples_lcm_l223_223427

theorem count_ordered_triples_lcm :
  { t : ℕ × ℕ × ℕ // ∃ x y z, t = (x, y, z) ∧ Nat.lcm x y = 180 ∧ Nat.lcm x z = 420 ∧ Nat.lcm y z = 1260 }.card = 1 :=
by
  sorry

end count_ordered_triples_lcm_l223_223427


namespace peukert_constant_l223_223879

theorem peukert_constant:
  (C : ℝ) (n : ℝ)
  (t1 t2 I1 I2 : ℝ)
  (h1 : I1 = 20) (h2 : I2 = 50)
  (h3 : t1 = 20) (h4 : t2 = 5)
  (h5 : C = I1^n * t1) (h6 : C = I2^n * t2)
  (lg2_approx : log 2 ≈ 0.3) :
  n ≈ 1.5 :=
by
  sorry

end peukert_constant_l223_223879


namespace expected_rolls_to_2010_l223_223667

noncomputable def expected_rolls_to_reach_sum (target_sum : ℕ) : ℝ :=
  let e (n : ℕ) : ℝ :=
    if n ≤ 0 then 0
    else if n <= 6 then (1 : ℝ)
    else (1/6) * (e (n - 1) + e (n - 2) + e (n - 3) + e (n - 4) + e (n - 5) + e (n - 6)) + 1
  e target_sum

theorem expected_rolls_to_2010 : 
  expected_rolls_to_reach_sum 2010 ≈ 574.761904 :=
begin
  sorry, -- Proof to be filled in
end

end expected_rolls_to_2010_l223_223667


namespace total_pages_correct_l223_223561

def history_pages : ℕ := 160
def geography_pages : ℕ := history_pages + 70
def sum_history_geography_pages : ℕ := history_pages + geography_pages
def math_pages : ℕ := sum_history_geography_pages / 2
def science_pages : ℕ := 2 * history_pages
def total_pages : ℕ := history_pages + geography_pages + math_pages + science_pages

theorem total_pages_correct : total_pages = 905 := by
  -- The proof goes here.
  sorry

end total_pages_correct_l223_223561


namespace intersection_at_exactly_one_point_l223_223414

noncomputable def f (x : ℝ) : ℝ := x * log x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := -x^2 + a * x - 2

theorem intersection_at_exactly_one_point (a : ℝ) : 
  (∃ x : ℝ, f x = g x a) ∧ (∀ x₁ x₂ : ℝ, f x₁ = g x₁ a → f x₂ = g x₂ a → x₁ = x₂) ↔ a = 3 :=
by
  sorry

end intersection_at_exactly_one_point_l223_223414


namespace simplest_form_is_C_l223_223260

variables (x y : ℝ) (hx : x ≠ 0) (hx1 : x ≠ 1) (hy : y ≠ 0)

def fraction_A := 3 * x * y / (x^2)
def fraction_B := (x - 1) / (x^2 - 1)
def fraction_C := (x + y) / (2 * x)
def fraction_D := (1 - x) / (x - 1)

theorem simplest_form_is_C : 
  ∀ (x y : ℝ) (hx : x ≠ 0) (hx1 : x ≠ 1) (hy : y ≠ 0), 
  ¬ (3 * x * y / (x^2)).is_simplest ∧ 
  ¬ ((x - 1) / (x^2 - 1)).is_simplest ∧ 
  (x + y) / (2 * x).is_simplest ∧ 
  ¬ ((1 - x) / (x - 1)).is_simplest :=
by 
  sorry

end simplest_form_is_C_l223_223260


namespace min_diagonals_2021_gon_l223_223690

-- Δefine the 2021-gon and conditions
def regular_polygon (n : ℕ) := { P : ℝ × ℝ // ∃ k, k < n ∧ (P = (cos (2 * π * k / n), sin (2 * π * k / n))) }

def labeling_conditions (f : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i, 0 ≤ i < n → |f (i) - f ((i+1)%n)| ≤ 1

-- Define the diagonals condition as stated
def diagonals_condition (f : ℕ → ℝ) (n : ℕ) : set (ℕ × ℕ) :=
  { (i, j) | i ≠ j ∧ |f i - f j| ≤ 1 ∧ i + 1 ≠ j ∧ i ≠ j + 1 }

-- Prove the minimum number of diagonals for the given conditions
theorem min_diagonals_2021_gon : ∀ f : ℕ → ℝ, labeling_conditions f 2021 → 
  ∃ d, d = 2018 ∧ (∀ f : ℕ → ℝ, labeling_conditions f 2021 → finsupp.card (diagonals_condition f 2021) = d) :=
by 
  sorry

end min_diagonals_2021_gon_l223_223690


namespace tiled_board_remainder_l223_223278

def num_ways_to_tile_9x1 : Nat := -- hypothetical function to calculate the number of ways
  sorry

def N : Nat :=
  num_ways_to_tile_9x1 -- placeholder for N, should be computed using correct formula

theorem tiled_board_remainder : N % 1000 = 561 :=
  sorry

end tiled_board_remainder_l223_223278


namespace simplify_expression_l223_223179

theorem simplify_expression (x : ℝ) : 7 * x + 15 - 3 * x + 2 = 4 * x + 17 := 
by sorry

end simplify_expression_l223_223179


namespace father_l223_223673

theorem father's_age (M F : ℕ) (h1 : M = 2 * F / 5) (h2 : M + 6 = (F + 6) / 2) : F = 30 :=
by
  sorry

end father_l223_223673


namespace total_coins_correct_l223_223134

-- Define basic parameters
def stacks_pennies : Nat := 3
def coins_per_penny_stack : Nat := 10
def stacks_nickels : Nat := 5
def coins_per_nickel_stack : Nat := 8
def stacks_dimes : Nat := 7
def coins_per_dime_stack : Nat := 4

-- Calculate total coins for each type
def total_pennies : Nat := stacks_pennies * coins_per_penny_stack
def total_nickels : Nat := stacks_nickels * coins_per_nickel_stack
def total_dimes : Nat := stacks_dimes * coins_per_dime_stack

-- Calculate total number of coins
def total_coins : Nat := total_pennies + total_nickels + total_dimes

-- Proof statement
theorem total_coins_correct : total_coins = 98 := by
  -- Proof steps go here (omitted)
  sorry

end total_coins_correct_l223_223134


namespace total_length_correct_l223_223679

-- Definitions for the first area's path length and scale.
def first_area_scale : ℕ := 500
def first_area_path_length_inches : ℕ := 6
def first_area_path_length_feet : ℕ := first_area_scale * first_area_path_length_inches

-- Definitions for the second area's path length and scale.
def second_area_scale : ℕ := 1000
def second_area_path_length_inches : ℕ := 3
def second_area_path_length_feet : ℕ := second_area_scale * second_area_path_length_inches

-- Total length represented by both paths in feet.
def total_path_length_feet : ℕ :=
  first_area_path_length_feet + second_area_path_length_feet

-- The Lean theorem proving that the total length is 6000 feet.
theorem total_length_correct : total_path_length_feet = 6000 := by
  sorry

end total_length_correct_l223_223679


namespace necessary_and_sufficient_condition_l223_223815

/-- Given the circles C₀: x² + y² = 1 and C₁: x²/a² + y²/b² = 1 where a > b > 0,
determine and prove the conditions that a and b must satisfy such that for any point P on C₁,
there exists a parallelogram with vertex P that is tangent to C₀ externally and inscribed in C₁. -/

theorem necessary_and_sufficient_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (∀ (P : ℝ × ℝ), P.fst^2 / a^2 + P.snd^2 / b^2 = 1 → 
   ∃ (A B : ℝ × ℝ), A ≠ B ∧ distance A B = 1 ∧ (A.fst^2 + A.snd^2 = 1) ∧ (B.fst^2 + B.snd^2 = 1)) 
  ↔ (1/a^2 + 1/b^2 = 1) :=
sorry

end necessary_and_sufficient_condition_l223_223815


namespace reciprocal_of_neg_2023_l223_223977

theorem reciprocal_of_neg_2023 : (1 : ℝ) / (-2023) = -(1 / 2023) :=
by 
  sorry

end reciprocal_of_neg_2023_l223_223977


namespace purely_imaginary_implies_value_of_a_l223_223443

theorem purely_imaginary_implies_value_of_a (a : ℝ) (h : (a^2 - 3*a + 2) + (a - 1)*1.i = (0 : ℂ)): a = 2 :=
by sorry

end purely_imaginary_implies_value_of_a_l223_223443


namespace split_into_similar_piles_l223_223088

def similar_sizes (x y : ℕ) : Prop := x ≤ 2 * y ∧ y ≤ 2 * x

theorem split_into_similar_piles (n : ℕ) (h : 0 < n) :
  ∃ (piles : list ℕ), (∀ x ∈ piles, x = 1) ∧ (list.sum piles = n) ∧
                       (∀ x y ∈ piles, similar_sizes x y) := 
sorry

end split_into_similar_piles_l223_223088


namespace inequality_solution_l223_223552

theorem inequality_solution (x : ℝ) :
  (2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 15) ↔ (x < 1 ∨ x > 3) ∧ (x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5) :=
by
  sorry

end inequality_solution_l223_223552


namespace no_real_roots_after_removal_l223_223249
open Polynomial

theorem no_real_roots_after_removal :
  ∀ P : Polynomial ℝ, P = (∏ i in Finset.range 2020, (X - C (i + 1))) →
    ∃ Q : Polynomial ℝ, (∃ S ⊆ Finset.range 2020, S.card = 1010 ∧ Q = (∏ i in S, (X - C (i + 1)))) ∧ 
    ∀ x : ℝ, ¬Q.is_root x :=
sorry

end no_real_roots_after_removal_l223_223249


namespace base_4_representation_divisible_by_3_digits_l223_223352

theorem base_4_representation_divisible_by_3_digits :
  let num := 375
  let base4_rep := [1, 1, 3, 1, 3] -- base-4 representation of 375
  (filter (λ d, d % 3 = 0) base4_rep).length = 2 :=
by
  sorry

end base_4_representation_divisible_by_3_digits_l223_223352


namespace angle_between_l223_223436

open Real

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V) (t : ℝ)

-- Define the conditions
def conditions : Prop :=
  (∥a∥ = t) ∧ (∥b∥ = t) ∧ (∥a - b∥ = t)

-- Define the statement to prove
theorem angle_between (h : conditions a b t) :
  angle b (a + b) = (π / 6 : ℝ) :=
sorry

end angle_between_l223_223436


namespace min_squared_sum_l223_223063

theorem min_squared_sum (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) : 
  x^2 + y^2 + z^2 ≥ 9 := 
sorry

end min_squared_sum_l223_223063


namespace simplify_exponent_l223_223172

theorem simplify_exponent (y : ℝ) : (3 * y^4)^5 = 243 * y^20 :=
by
  sorry

end simplify_exponent_l223_223172


namespace common_difference_l223_223988

theorem common_difference (a1 d : ℕ) (S3 : ℕ) (h1 : S3 = 6) (h2 : a1 = 1)
  (h3 : S3 = 3 * (2 * a1 + 2 * d) / 2) : d = 1 :=
by
  sorry

end common_difference_l223_223988


namespace solution_l223_223317

noncomputable def problem_statement : Prop :=
  (∀ i : ℂ, i^2 = -1) → 
  (∀ i : ℂ, -i ≠ 0 → (-i)^4 = 1) → 
  (∀ i : ℂ, (1 - i)^2 = 1 - 2*i + i^2) → 
  ∀ i : ℂ, (i^2 = -1) → (1 - i) ≠ 0 → ( ∃ (x : ℂ), x = (1-i)/(real.sqrt 2) → x^48 = 1 )

theorem solution : problem_statement :=
by {
  sorry
}

end solution_l223_223317


namespace average_monthly_growth_rate_equation_l223_223452

-- Definitions directly from the conditions
def JanuaryOutput : ℝ := 50
def QuarterTotalOutput : ℝ := 175
def averageMonthlyGrowthRate (x : ℝ) : ℝ :=
  JanuaryOutput + JanuaryOutput * (1 + x) + JanuaryOutput * (1 + x) ^ 2

-- The statement to prove that the derived equation is correct
theorem average_monthly_growth_rate_equation (x : ℝ) :
  averageMonthlyGrowthRate x = QuarterTotalOutput :=
sorry

end average_monthly_growth_rate_equation_l223_223452


namespace sara_dozen_quarters_l223_223547

theorem sara_dozen_quarters (dollars : ℕ) (quarters_per_dollar : ℕ) (quarters_per_dozen : ℕ) 
  (h1 : dollars = 9) (h2 : quarters_per_dollar = 4) (h3 : quarters_per_dozen = 12) : 
  dollars * quarters_per_dollar / quarters_per_dozen = 3 := 
by 
  sorry

end sara_dozen_quarters_l223_223547


namespace puppies_per_female_dog_l223_223569

theorem puppies_per_female_dog
  (number_of_dogs : ℕ)
  (percent_female : ℝ)
  (fraction_female_giving_birth : ℝ)
  (remaining_puppies : ℕ)
  (donated_puppies : ℕ)
  (total_puppies : ℕ)
  (number_of_female_dogs : ℕ)
  (number_female_giving_birth : ℕ)
  (puppies_per_dog : ℕ) :
  number_of_dogs = 40 →
  percent_female = 0.60 →
  fraction_female_giving_birth = 0.75 →
  remaining_puppies = 50 →
  donated_puppies = 130 →
  total_puppies = remaining_puppies + donated_puppies →
  number_of_female_dogs = percent_female * number_of_dogs →
  number_female_giving_birth = fraction_female_giving_birth * number_of_female_dogs →
  puppies_per_dog = total_puppies / number_female_giving_birth →
  puppies_per_dog = 10 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end puppies_per_female_dog_l223_223569


namespace gcd_lcm_product_l223_223355

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 24) (h2 : b = 60) :
  Nat.gcd a b * Nat.lcm a b = 1440 :=
by
  sorry

end gcd_lcm_product_l223_223355


namespace unique_arrangements_mississippi_l223_223747

open scoped Nat

theorem unique_arrangements_mississippi : 
  let n := 11
  let n_M := 1
  let n_I := 4
  let n_S := 4
  let n_P := 2
  (n.fact / (n_M.fact * n_I.fact * n_S.fact * n_P.fact)) = 34650 :=
  by
  sorry

end unique_arrangements_mississippi_l223_223747


namespace unique_arrangements_mississippi_l223_223744

open scoped Nat

theorem unique_arrangements_mississippi : 
  let n := 11
  let n_M := 1
  let n_I := 4
  let n_S := 4
  let n_P := 2
  (n.fact / (n_M.fact * n_I.fact * n_S.fact * n_P.fact)) = 34650 :=
  by
  sorry

end unique_arrangements_mississippi_l223_223744


namespace total_pages_l223_223563

theorem total_pages (history_pages geography_additional math_factor science_factor : ℕ) 
  (h1 : history_pages = 160)
  (h2 : geography_additional = 70)
  (h3 : math_factor = 2)
  (h4 : science_factor = 2) 
  : let geography_pages := history_pages + geography_additional in
    let sum_history_geography := history_pages + geography_pages in
    let math_pages := sum_history_geography / math_factor in
    let science_pages := history_pages * science_factor in
    history_pages + geography_pages + math_pages + science_pages = 905 :=
by
  sorry

end total_pages_l223_223563


namespace find_first_term_l223_223220

variable {a r : ℚ}

theorem find_first_term (h1 : a / (1 - r) = 30) (h2 : a^2 / (1 - r^2) = 120) : a = 240 / 7 :=
by
  sorry

end find_first_term_l223_223220


namespace inequality_x_geq_x_n_plus_2n_l223_223922

theorem inequality_x_geq_x_n_plus_2n
  (n : ℕ)
  (x : ℕ → ℝ)
  (h : ∀ i, i < n → x i > x (i + 1)) :
  x 0 + ∑ k in Finset.range n, (1 / (x k - x (k + 1))) ≥ x n + 2 * n := by
  sorry

end inequality_x_geq_x_n_plus_2n_l223_223922


namespace range_of_function_l223_223603

theorem range_of_function :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → 2 ≤ x^2 - 2 * x + 3 ∧ x^2 - 2 * x + 3 ≤ 6) :=
by {
  sorry
}

end range_of_function_l223_223603


namespace mutually_exclusive_not_complementary_l223_223283

-- Definitions based on conditions in the problem
def total_balls : ℕ := 6
def red_balls : ℕ := 3
def white_balls : ℕ := 2
def black_balls : ℕ := 1

-- Event A: At least one white ball
def at_least_one_white_ball (events : set ℕ) : Prop :=
  ∃ (n : ℕ), n ∈ events ∧ n ≥ 1

-- Event B: One red ball and one black ball
def one_red_one_black (events : set ℕ) : Prop :=
  ∃ (r b : ℕ), r ∈ events ∧ b ∈ events ∧ r = 1 ∧ b = 1

-- The final proof problem statement in Lean 4
theorem mutually_exclusive_not_complementary :
(EX : set ℕ) → 
(at_least_one_white_ball EX) ∧ (one_red_one_black EX)
→ ...
-- [Mutually exclusive, but not complementary logic goes here]
sorry

end mutually_exclusive_not_complementary_l223_223283


namespace ellipse_condition_l223_223016

theorem ellipse_condition (k : ℝ) : 
  (k > 1 ↔ 
  (k - 1 > 0 ∧ k + 1 > 0 ∧ k - 1 ≠ k + 1)) :=
by sorry

end ellipse_condition_l223_223016


namespace find_lambda_l223_223393

open Real

variables (a b : Vector ℝ) (λ : ℝ)

axiom a_perp_b : inner a b = 0
axiom norm_a : ‖a‖ = 2
axiom norm_b : ‖b‖ = 3
axiom perp_condition : inner (3 • a + 2 • b) (λ • a - b) = 0

theorem find_lambda : λ = 3 / 2 :=
by
  sorry

end find_lambda_l223_223393


namespace find_prime_number_between_50_and_60_l223_223204

theorem find_prime_number_between_50_and_60 (n : ℕ) :
  (50 < n ∧ n < 60) ∧ Prime n ∧ n % 7 = 3 ↔ n = 59 :=
by
  sorry

end find_prime_number_between_50_and_60_l223_223204


namespace split_into_similar_heaps_l223_223106

noncomputable def similar_sizes (x y : ℕ) : Prop :=
  x ≤ 2 * y

theorem split_into_similar_heaps (n : ℕ) (h : n > 0) : 
  ∃ f : ℕ → ℕ, (∀ k, k < n → similar_sizes (f (k + 1)) (f k)) ∧ f (n - 1) = n := by
  sorry

end split_into_similar_heaps_l223_223106


namespace combine_heaps_l223_223125

def heaps_similar (x y : ℕ) : Prop :=
  x ≤ 2 * y ∧ y ≤ 2 * x

theorem combine_heaps (n : ℕ) : 
  ∃ f : ℕ → ℕ, 
  f 0 = n ∧
  ∀ k, k < n → (∃ i j, i + j = k ∧ heaps_similar (f i) (f j)) ∧ 
  (∃ k, f k = n) :=
by
  sorry

end combine_heaps_l223_223125


namespace machine_pays_for_itself_in_36_days_l223_223486

def cost_of_machine : ℝ := 200
def discount : ℝ := 20
def cost_per_day : ℝ := 3
def previous_cost_per_coffee : ℝ := 4
def number_of_coffees_per_day : ℝ := 2

def net_cost_of_machine : ℝ := cost_of_machine - discount
def daily_savings : ℝ := number_of_coffees_per_day * previous_cost_per_coffee - cost_per_day

def days_until_machine_pays_for_itself : ℝ := net_cost_of_machine / daily_savings

theorem machine_pays_for_itself_in_36_days : days_until_machine_pays_for_itself = 36 := 
by {
  -- Proof skipped
  sorry
}

end machine_pays_for_itself_in_36_days_l223_223486


namespace annual_profit_function_correct_maximum_annual_profit_l223_223184

noncomputable def fixed_cost : ℝ := 60

noncomputable def variable_cost (x : ℝ) : ℝ :=
  if x < 12 then 
    0.5 * x^2 + 4 * x 
  else 
    11 * x + 100 / x - 39

noncomputable def selling_price_per_thousand : ℝ := 10

noncomputable def sales_revenue (x : ℝ) : ℝ := selling_price_per_thousand * x

noncomputable def annual_profit (x : ℝ) : ℝ := sales_revenue x - fixed_cost - variable_cost x

theorem annual_profit_function_correct : 
∀ x : ℝ, (0 < x ∧ x < 12 → annual_profit x = -0.5 * x^2 + 6 * x - fixed_cost) ∧ 
        (x ≥ 12 → annual_profit x = -x - 100 / x + 33) :=
sorry

theorem maximum_annual_profit : 
∃ x : ℝ, x = 12 ∧ annual_profit x = 38 / 3 :=
sorry

end annual_profit_function_correct_maximum_annual_profit_l223_223184


namespace number_of_foxes_l223_223136

-- Define the conditions as given in the problem
def num_cows : ℕ := 20
def num_sheep : ℕ := 20
def total_animals : ℕ := 100
def num_zebras (F : ℕ) := 3 * F

-- The theorem we want to prove based on the conditions
theorem number_of_foxes (F : ℕ) :
  num_cows + num_sheep + F + num_zebras F = total_animals → F = 15 :=
by
  sorry

end number_of_foxes_l223_223136


namespace power_of_two_l223_223052

theorem power_of_two (b m n : ℕ) (hb : b > 1) (hmn : m ≠ n) 
  (hsame : prime_divisors (b^m - 1) = prime_divisors (b^n - 1)) : 
  ∃ k : ℕ, b + 1 = 2^k := 
sorry

end power_of_two_l223_223052


namespace unique_interior_point_is_centroid_l223_223234

theorem unique_interior_point_is_centroid 
  (A B C : ℤ × ℤ)
  (D : ℤ × ℤ)
  (h1 : ∀ (P : ℤ × ℤ), (is_on_segment A B P → P = A ∨ P = B) ∧ (is_on_segment B C P → P = B ∨ P = C) ∧ (is_on_segment A C P → P = A ∨ P = C))
  (h2 : ∀ (P : ℤ × ℤ), is_interior_point_of_triangle A B C P → P = D)
  (h3 : D ≠ A ∧ D ≠ B ∧ D ≠ C) :
  D = (1/3) • A + (1/3) • B + (1/3) • C := sorry

-- Definitions that might be needed for the is_on_segment and is_interior_point_of_triangle predicates:
def is_on_segment (A B P : ℤ × ℤ) : Prop := sorry   -- Definition for checking if P is on the segment AB

def is_interior_point_of_triangle (A B C P : ℤ × ℤ) : Prop := sorry -- Definition for checking if P is inside triangle ABC

end unique_interior_point_is_centroid_l223_223234


namespace product_of_integers_cubes_sum_to_35_l223_223643

-- Define the conditions
def integers_sum_of_cubes (a b : ℤ) : Prop :=
  a^3 + b^3 = 35

-- Define the theorem that the product of integers whose cubes sum to 35 is 6
theorem product_of_integers_cubes_sum_to_35 :
  ∃ a b : ℤ, integers_sum_of_cubes a b ∧ a * b = 6 :=
by
  sorry

end product_of_integers_cubes_sum_to_35_l223_223643


namespace smallest_term_at_n_is_4_or_5_l223_223886

def a_n (n : ℕ) : ℝ :=
  n^2 - 9 * n - 100

theorem smallest_term_at_n_is_4_or_5 :
  ∃ n, n = 4 ∨ n = 5 ∧ a_n n = min (a_n 4) (a_n 5) :=
by
  sorry

end smallest_term_at_n_is_4_or_5_l223_223886


namespace distinct_imaginary_numbers_l223_223620

theorem distinct_imaginary_numbers (d_real : Fin 10) (d_imag : Fin 9) : 
  let number_of_real_parts := 10
  let number_of_imaginary_parts := 9
  number_of_real_parts * number_of_imaginary_parts = 90 :=
by
  let number_of_real_parts := 10
  let number_of_imaginary_parts := 9
  exact (congr_arg (λ n : ℕ, 10 * n) (rfl : 9 = 9)).symm.trans (rfl : 90 = 90)

end distinct_imaginary_numbers_l223_223620


namespace douglas_votes_in_county_D_l223_223032

noncomputable def percent_votes_in_county_D (x : ℝ) (votes_A votes_B votes_C votes_D : ℝ) 
    (total_votes : ℝ) (percent_A percent_B percent_C percent_D total_percent : ℝ) : Prop :=
  (votes_A / (5 * x) = 0.70) ∧
  (votes_B / (3 * x) = 0.58) ∧
  (votes_C / (2 * x) = 0.50) ∧
  (votes_A + votes_B + votes_C + votes_D) / total_votes = 0.62 ∧
  (votes_D / (4 * x) = percent_D)

theorem douglas_votes_in_county_D 
  (x : ℝ) (votes_A votes_B votes_C votes_D : ℝ) 
  (total_votes : ℝ := 14 * x) 
  (percent_A percent_B percent_C total_percent percent_D : ℝ)
  (h1 : votes_A / (5 * x) = 0.70) 
  (h2 : votes_B / (3 * x) = 0.58) 
  (h3 : votes_C / (2 * x) = 0.50) 
  (h4 : (votes_A + votes_B + votes_C + votes_D) / total_votes = 0.62) : 
  percent_votes_in_county_D x votes_A votes_B votes_C votes_D total_votes percent_A percent_B percent_C 0.61 total_percent :=
by
  constructor
  exact h1
  constructor
  exact h2
  constructor
  exact h3
  constructor
  exact h4
  sorry

end douglas_votes_in_county_D_l223_223032


namespace div_by_64_l223_223066

theorem div_by_64 (n : ℕ) (h : n > 0) : 64 ∣ (5^n - 8*n^2 + 4*n - 1) :=
sorry

end div_by_64_l223_223066


namespace first_investment_rate_l223_223021

-- Define the conditions
def total_investment := 2000
def second_investment := 600
def second_rate := 0.08
def excess_income := 92

-- Define the unknowns
variable (A R : ℝ)

-- State the Lean 4 statement
theorem first_investment_rate
  (hA : A = total_investment - second_investment)
  (h_income : A * R - second_investment * second_rate = excess_income) :
  R = 0.1 :=
sorry

end first_investment_rate_l223_223021


namespace quadrilateral_not_necessarily_square_l223_223872

structure Quadrilateral (α : Type) :=
  (A B C D : α)
  (diagonal_perpendicular : ∀ (u v : α), u = A ∧ v = C ∨ u = B ∧ v = D → ⟨u, v⟩ ⊥)
  (inscribable : ∀ (r : ℝ), ∃ c : α, circle r c)
  (circumscribable : ∀ (r : ℝ), ∃ c : α, circle_around r c)

theorem quadrilateral_not_necessarily_square 
  (Q : Quadrilateral ℝ) 
  (h1 : Q.diagonal_perpendicular)
  (h2 : Q.inscribable)
  (h3 : Q.circumscribable) : 
  ¬ ∀ (A B C D: ℝ), square A B C D → Q = square A B C D :=
sorry

end quadrilateral_not_necessarily_square_l223_223872


namespace intervals_of_monotonicity_range_of_a_l223_223413

noncomputable def function_f (a x : ℝ) : ℝ := a^2 * Real.log x - x^2 + a * x

theorem intervals_of_monotonicity (a : ℝ) :
  (a = 0 → ∀ x > 0, 0 < x → ∀ y > x, function_f a y > function_f a x)
  ∧ (a > 0 → ∀ x, (x ∈ Ioo 0 a → function_f a x > function_f a x) 
  ∧ (x ∈ Ioo a (⊤ : ℝ) → function_f a x > function_f a (a + x)))
  ∧ (a < 0 → ∀ x, (x ∈ Ioo 0 (-a/2) → function_f a x < function_f a x) 
  ∧ (x ∈ Ioo (-a / 2) (⊤ : ℝ) → function_f a x > function_f a (-a / 2 + x))) :=
sorry

theorem range_of_a (a : ℝ) (h : a > 0) (h_root : ∃ x ∈ Ioo 1 (Real.exp 1), function_f a x = 0) :
  1 < a ∧ a < (Real.sqrt 5 - 1) / 2 * Real.exp 1 :=
sorry

end intervals_of_monotonicity_range_of_a_l223_223413


namespace sum_of_four_squares_l223_223306

theorem sum_of_four_squares (a b c : ℕ) 
    (h1 : 2 * a + b + c = 27)
    (h2 : 2 * b + a + c = 25)
    (h3 : 3 * c + a = 39) : 4 * c = 44 := 
  sorry

end sum_of_four_squares_l223_223306


namespace circle_reassemble_l223_223676

theorem circle_reassemble (circle : Type) (marked_point : circle) : 
  ∃ (parts : list circle), parts.length = 3 ∧ 
  rearrange parts marked_point = true :=
by
  sorry

end circle_reassemble_l223_223676


namespace complex_power_calculation_l223_223315

theorem complex_power_calculation : (1 - complex.I) / real.sqrt 2 ^ 48 = 1 :=
by
  sorry

end complex_power_calculation_l223_223315


namespace pattern_equation_l223_223818

theorem pattern_equation (n : ℕ) : n^2 + n = n * (n + 1) := 
  sorry

end pattern_equation_l223_223818


namespace polar_distance_l223_223902

noncomputable def distance_AB (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem polar_distance 
  (θ₁ θ₂ : ℝ) 
  (h : θ₁ - θ₂ = real.pi / 2) :
  distance_AB (3, θ₁) (9, θ₂) = 3 * real.sqrt 10 := 
by
  sorry

end polar_distance_l223_223902


namespace radius_squared_of_circle_l223_223327

noncomputable def circle_chord_intersection (r : ℝ) (chord_AB : ℝ) (chord_CD : ℝ) (BP : ℝ) (angle_APD : ℝ) : Prop :=
  chord_AB = 12 ∧
  chord_CD = 8 ∧
  BP = 10 ∧
  angle_APD = 90 ∧
  (r * r = (26 + 2 * Real.sqrt 21) / 4)

theorem radius_squared_of_circle :
  ∃ r : ℝ, circle_chord_intersection r 12 8 10 90 :=
begin
  sorry
end

end radius_squared_of_circle_l223_223327


namespace parallel_planes_l223_223758

-- Definitions for the problem
variable {A : Type} -- Define the type for points
variable {Line Plane : Type} -- Define the type for lines and planes
variable (a b : Line) (α β : Plane) (A : A)

-- Conditions
variable (a_in_α : a ⊆ α)
variable (b_in_α : b ⊆ α)
variable (a_inter_b_eq_A : a ∩ b = A)
variable (a_par_β : a ∥ β)
variable (b_par_β : b ∥ β)

-- Theorem to prove
theorem parallel_planes 
  (a_in_α : a ⊆ α)
  (b_in_α : b ⊆ α)
  (a_inter_b_eq_A : a ∩ b = A)
  (a_par_β : a ∥ β)
  (b_par_β : b ∥ β) :
  α ∥ β := sorry

end parallel_planes_l223_223758


namespace reciprocal_of_neg_2023_l223_223982

theorem reciprocal_of_neg_2023 : 1 / (-2023) = -1 / 2023 :=
by
  sorry

end reciprocal_of_neg_2023_l223_223982


namespace find_a_l223_223799

theorem find_a (a : ℝ) : 
  (-3 ∈ {a - 2, 12, 2 * a^2 + 5 * a}) ∧
  (a - 2 ≠ 12) ∧ 
  (a - 2 ≠ 2 * a^2 + 5 * a) ∧ 
  (12 ≠ 2 * a^2 + 5 * a) →
  a = -3/2 :=
by 
  sorry

end find_a_l223_223799


namespace find_x_l223_223675

theorem find_x (number x : ℝ) (h1 : 24 * number = 173 * x) (h2 : 24 * number = 1730) : x = 10 :=
by
  sorry

end find_x_l223_223675


namespace heaps_combination_preserve_similarity_split_stones_into_similar_heaps_l223_223122

def initial_heaps (n : ℕ) : list ℕ := list.repeat 1 n

def combine_heaps (heaps : list ℕ) : list ℕ :=
  if heaps.length ≥ 2 then
    let min1 := list.minimum heaps,
        heaps' := list.erase heaps min1,
        min2 := list.minimum heaps'
    in
    if min1 ≤ min2 then
      (min1 + min2) :: list.erase heaps' min2
    else
      heaps
  else
    heaps

theorem heaps_combination_preserve_similarity (heaps : list ℕ) (h : ∀ x ∈ heaps, x = 1) :
  ∀ combined_heaps, combined_heaps = combine_heaps heaps →
  ∀ x y ∈ combined_heaps, x ≤ y → x + y ≤ 2 * y :=
sorry

theorem split_stones_into_similar_heaps (n : ℕ) :
  ∃ combined_heaps : list ℕ, ∀ x y ∈ combined_heaps, x ≤ y → x + y ≤ 2 * y :=
sorry

end heaps_combination_preserve_similarity_split_stones_into_similar_heaps_l223_223122


namespace grading_ratio_l223_223684

noncomputable def num_questions : ℕ := 100
noncomputable def correct_answers : ℕ := 91
noncomputable def score_received : ℕ := 73
noncomputable def incorrect_answers : ℕ := num_questions - correct_answers
noncomputable def total_points_subtracted : ℕ := correct_answers - score_received
noncomputable def points_per_incorrect : ℚ := total_points_subtracted / incorrect_answers

theorem grading_ratio (h: (points_per_incorrect : ℚ) = 2) :
  2 / 1 = points_per_incorrect / 1 :=
by sorry

end grading_ratio_l223_223684


namespace find_t_l223_223140

theorem find_t (t : ℝ) :
  (2 * t - 7) * (3 * t - 4) = (3 * t - 9) * (2 * t - 6) →
  t = 26 / 7 := 
by 
  intro h
  sorry

end find_t_l223_223140


namespace trip_length_is_180_l223_223546

theorem trip_length_is_180 (d : ℕ) (h1 : d > 60)
  (h2 : let x := d - 60 in x * 0.03 * 50 = d) : d = 180 := 
by
  sorry

end trip_length_is_180_l223_223546


namespace split_into_similar_heaps_l223_223100

noncomputable def similar_sizes (x y : ℕ) : Prop :=
  x ≤ 2 * y

theorem split_into_similar_heaps (n : ℕ) (h : n > 0) : 
  ∃ f : ℕ → ℕ, (∀ k, k < n → similar_sizes (f (k + 1)) (f k)) ∧ f (n - 1) = n := by
  sorry

end split_into_similar_heaps_l223_223100


namespace split_into_similar_heaps_l223_223105

noncomputable def similar_sizes (x y : ℕ) : Prop :=
  x ≤ 2 * y

theorem split_into_similar_heaps (n : ℕ) (h : n > 0) : 
  ∃ f : ℕ → ℕ, (∀ k, k < n → similar_sizes (f (k + 1)) (f k)) ∧ f (n - 1) = n := by
  sorry

end split_into_similar_heaps_l223_223105


namespace largest_n_l223_223971

def is_prime_digit (p : Nat) : Prop :=
  p ∈ {2, 3, 5, 7}

def valid_n (n x y : Nat) : Prop :=
  is_prime_digit x ∧ is_prime_digit y ∧ x ≠ y ∧ n = x * y * (10 * y + x) ∧ n ≥ 1000 ∧ n < 10000

theorem largest_n (x y n : Nat) (h : valid_n n x y) : n ≤ 1533 :=
sorry

end largest_n_l223_223971


namespace mila_social_media_hours_l223_223345

/-- 
Mila spends 6 hours on his phone every day. 
Half of this time is spent on social media. 
Prove that Mila spends 21 hours on social media in a week.
-/
theorem mila_social_media_hours 
  (hours_per_day : ℕ)
  (phone_time_per_day : hours_per_day = 6)
  (daily_social_media_fraction : ℕ)
  (fractional_time : daily_social_media_fraction = hours_per_day / 2)
  (days_per_week : ℕ)
  (days_in_week : days_per_week = 7) :
  (daily_social_media_fraction * days_per_week = 21) :=
sorry

end mila_social_media_hours_l223_223345


namespace square_area_of_equal_perimeter_l223_223693

theorem square_area_of_equal_perimeter 
  (side_length_triangle : ℕ) (side_length_square : ℕ) (perimeter_square : ℕ)
  (h1 : side_length_triangle = 20)
  (h2 : perimeter_square = 3 * side_length_triangle)
  (h3 : 4 * side_length_square = perimeter_square) :
  side_length_square ^ 2 = 225 := 
by
  sorry

end square_area_of_equal_perimeter_l223_223693


namespace solve_for_b_l223_223856

variable (a b c d m : ℝ)

theorem solve_for_b (h : m = cadb / (a - b)) : b = ma / (cad + m) :=
sorry

end solve_for_b_l223_223856


namespace average_marker_cost_is_28_l223_223287

noncomputable def total_cost (markers_cost handling_fee shipping_fee : ℝ) : ℝ :=
  markers_cost + handling_fee + shipping_fee

noncomputable def cost_in_cents (cost : ℝ) : ℕ :=
  (cost * 100).to_nat

noncomputable def average_cost_per_marker (total_cost_in_cents : ℕ) (number_of_markers : ℕ) : ℕ :=
  (total_cost_in_cents / number_of_markers).to_nat

theorem average_marker_cost_is_28
    (num_markers : ℕ)
    (price_markers : ℝ)
    (handling_fee_per_marker : ℝ)
    (shipping_cost : ℝ)
    (h_ts : num_markers = 300)
    (h_p : price_markers = 45)
    (h_hf : handling_fee_per_marker = 0.10)
    (h_s : shipping_cost = 8.50) :
    average_cost_per_marker (cost_in_cents (total_cost price_markers (num_markers * handling_fee_per_marker) shipping_cost)) num_markers = 28 :=
  sorry

end average_marker_cost_is_28_l223_223287


namespace postcards_impossible_l223_223698

theorem postcards_impossible :
  ∀ (students : Finset ℕ) (send_rel : ℕ → ℕ → Prop),
  students.card = 7 →
  (∀ s ∈ students, (Finset.filter (λ x, send_rel s x) students).card = 3) →
  ¬ (∀ s t ∈ students, send_rel s t → send_rel t s) :=
by
  intros students send_rel h_card h_send_rel
  sorry

end postcards_impossible_l223_223698


namespace yellow_balls_count_l223_223653

theorem yellow_balls_count (r y : ℕ) (h1 : r = 9) (h2 : (r : ℚ) / (r + y) = 1 / 3) : y = 18 := 
by
  sorry

end yellow_balls_count_l223_223653


namespace arc_length_of_adjacent_sides_of_regular_octagon_l223_223298

def inscribed_regular_octagon (r : ℝ) : ℝ := 8 * r * sin (real.pi / 8)

theorem arc_length_of_adjacent_sides_of_regular_octagon 
  (radius : ℝ) (side_length : ℝ) 
  (h : side_length = 4) 
  (r_calc : radius = side_length / (2 * real.sin (real.pi / 8)))
  (h_r : radius ≈ 5.236)
  (arc_length : ℝ) : 
  arc_length = 2 * radius * real.pi / 4 := by
  sorry

end arc_length_of_adjacent_sides_of_regular_octagon_l223_223298


namespace perfect_cubes_between_150_and_800_l223_223428

theorem perfect_cubes_between_150_and_800 :
  let cubes := {n : ℕ | ∃ k : ℕ, k^3 = n} in
  {n : ℕ | 150 ≤ n ∧ n ≤ 800 ∧ n ∈ cubes}.finset.card = 4 :=
by
  sorry

end perfect_cubes_between_150_and_800_l223_223428


namespace sum_first_n_terms_l223_223835

noncomputable def a_n (n : ℕ) : ℝ :=
  2 * 3^(n-1) + (-1:ℝ)^n * (Real.log 2 - Real.log 3) + (-1:ℝ)^n * n * Real.log 3

noncomputable def S_n (n : ℕ) : ℝ :=
  if n % 2 = 0 then 3^n + (n / 2) * Real.log 3 - 1
  else 3^n - ((n - 1) / 2) * Real.log 3 - Real.log 2 - 1

theorem sum_first_n_terms (n : ℕ) :
  (finset.range n).sum (λ k, a_n (k + 1)) = S_n n :=
by
  sorry

end sum_first_n_terms_l223_223835


namespace evaluate_f_g3_l223_223004

def f (x : ℝ) : ℝ := 3 * x ^ 2 - 2 * x + 1
def g (x : ℝ) : ℝ := x + 3

theorem evaluate_f_g3 : f (g 3) = 97 := by
  sorry

end evaluate_f_g3_l223_223004


namespace divisible_by_100_l223_223060

def is_positive_odd_integer (n : ℕ) : Prop := n % 2 = 1 ∧ n > 0

lemma ordered_quadruples_sum (x1 x2 x3 x4 : ℕ) (h1 : is_positive_odd_integer x1)
  (h2 : is_positive_odd_integer x2) (h3 : is_positive_odd_integer x3)
  (h4 : is_positive_odd_integer x4) : x1 + x2 + x3 + x4 = 100 ↔
  2 * ((x1 + 1) / 2 + (x2 + 1) / 2 + (x3 + 1) / 2 + (x4 + 1) / 2) - 4 = 100 :=
by
  rw [is_positive_odd_integer] at h1 h2 h3 h4
  simp_rw [mul_add, mul_one,add_sub_assoc, div_add_div_same, div_self]
  simp

theorem divisible_by_100 :
  n = (Nat.choose 51 3) →
  (n / 100 = 208.25) :=
by sorry

end divisible_by_100_l223_223060


namespace intersection_M_N_is_neg1_0_l223_223422

open Set

-- Definitions
def M := {x : ℝ | abs (x + 1) ≤ 1}
def N := {-1, 0, 1}

-- The proof problem statement
theorem intersection_M_N_is_neg1_0 : M ∩ N = {-1, 0} :=
by sorry

end intersection_M_N_is_neg1_0_l223_223422


namespace circle_center_radius_l223_223192

theorem circle_center_radius :
  ∀ x y : ℝ,
  x^2 + y^2 + 4 * x - 6 * y - 3 = 0 →
  (∃ h k r : ℝ, (x + h)^2 + (y + k)^2 = r^2 ∧ h = -2 ∧ k = 3 ∧ r = 4) :=
by
  intros x y hxy
  sorry

end circle_center_radius_l223_223192


namespace tiling_impossible_l223_223046

theorem tiling_impossible :
  ¬ ∃ (board : fin 13 × fin 13 → Prop) (tile : fin 42 → fin 4 × fin 1 → fin 13 × fin 13),
    (∀ t : fin 42, ∀ c1 c2 : fin 4 × fin 1, tile t c1 ≠ tile t c2 ∧ tile t c1 ≠ board 6 6) ∧
    (∀ p : fin 13 × fin 13, p ≠ (6, 6) → ∃ t : fin 42, ∃ c : fin 4 × fin 1, tile t c = p) :=
sorry

end tiling_impossible_l223_223046


namespace circumradius_ge_two_inradius_equilateral_triangle_iff_R_eq_2r_l223_223050

noncomputable def circumradius (a b c : ℝ) (Δ : ℝ) : ℝ :=
  (a * b * c) / (4 * Δ)

noncomputable def inradius (Δ : ℝ) (s : ℝ) : ℝ :=
  Δ / s

theorem circumradius_ge_two_inradius 
  (a b c s Δ : ℝ)
  (hD_pos : Δ > 0)
  (hs_pos : s > 0) 
  (R := circumradius a b c Δ)
  (r := inradius Δ s) :
  R ≥ 2 * r :=
sorry

theorem equilateral_triangle_iff_R_eq_2r
  (a : ℝ) :
  (∃ b c s Δ, 
    a = b ∧ a = c ∧ 
    s = (a + b + c) / 2 ∧ 
    Δ = Math.sqrt (s * (s - a) * (s - b) * (s - c)) ∧
    circumradius a b c Δ = 2 * inradius Δ s) ↔ 
  true :=
sorry

end circumradius_ge_two_inradius_equilateral_triangle_iff_R_eq_2r_l223_223050


namespace maple_logs_correct_l223_223494

/-- Each pine tree makes 80 logs. -/
def pine_logs := 80

/-- Each walnut tree makes 100 logs. -/
def walnut_logs := 100

/-- Jerry cuts up 8 pine trees. -/
def pine_trees := 8

/-- Jerry cuts up 3 maple trees. -/
def maple_trees := 3

/-- Jerry cuts up 4 walnut trees. -/
def walnut_trees := 4

/-- The total number of logs is 1220. -/
def total_logs := 1220

/-- The number of logs each maple tree makes. -/
def maple_logs := 60

theorem maple_logs_correct :
  (pine_trees * pine_logs) + (maple_trees * maple_logs) + (walnut_trees * walnut_logs) = total_logs :=
by
  -- (8 * 80) + (3 * 60) + (4 * 100) = 1220
  sorry

end maple_logs_correct_l223_223494


namespace smallest_k_for_naoish_sum_l223_223248

def is_naoish (n : ℕ) : Prop :=
  n ≥ 90 ∧ (n / 10 % 10 = 9)

theorem smallest_k_for_naoish_sum (k : ℕ) (nums : Fin k → ℕ) 
  (h_naoish : ∀ i, is_naoish (nums i)) 
  (h_sum : Finset.univ.sum nums = 2020) :
  k = 8 :=
begin
  sorry
end

end smallest_k_for_naoish_sum_l223_223248


namespace line_equation_l223_223289

theorem line_equation (x y : ℝ) :
  (2, -1) • (x - 1, y + 3) = 0 → y = 2 * x - 5 :=
by
  intro h
  sorry

end line_equation_l223_223289


namespace sqrt_sum_simplification_l223_223343

theorem sqrt_sum_simplification : 
  Real.sqrt ((5 - 3 * Real.sqrt 2)^2) + Real.sqrt ((5 + 3 * Real.sqrt 2)^2) = 10 := by
  sorry

end sqrt_sum_simplification_l223_223343


namespace angle_C_is_60_degrees_l223_223771

/-- In triangle ABC, given that the angles between the altitude and the angle bisector
from vertices A and B are equal and less than the angle at vertex C, prove that 
the measure of angle C is 60 degrees. -/
theorem angle_C_is_60_degrees (A B C : ℝ)
  (ha : 0 < A) (hb : 0 < B) (hc : 0 < C) (triangle_abc : A + B + C = 180)
  (angle_bisector_altitude_A_eq_B : 
    (1/2) * | A - C | = (1/2) * | B - C |)
  (angle_bisector_altitude_A_B_lt_C : 
    (1/2) * | A - C | < C ∧ (1/2) * | B - C | < C) :
  C = 60 :=
sorry

end angle_C_is_60_degrees_l223_223771


namespace percentage_solution_l223_223630

def percentage_x_of_y (x y : ℝ) : ℝ := (x / y) * 100

theorem percentage_solution :
  ∀ (x y m k : ℝ),
  P y = 0.60 * y →
  z = x + k →
  P y = 48 →
  k = 70 →
  percentage_x_of_y k 100 = 70 :=
by
  intros
  sorry

end percentage_solution_l223_223630


namespace num_arrangements_with_ab_together_l223_223611

theorem num_arrangements_with_ab_together (products : Fin 5 → Type) :
  (∃ A B : Fin 5 → Type, A ≠ B) →
  ∃ (n : ℕ), n = 48 :=
by
  sorry

end num_arrangements_with_ab_together_l223_223611


namespace simplify_expression_l223_223250

theorem simplify_expression (x y : ℝ) (h_x_ne_0 : x ≠ 0) (h_y_ne_0 : y ≠ 0) :
  (25*x^3*y) * (8*x*y) * (1 / (5*x*y^2)^2) = 8*x^2 / y^2 :=
by
  sorry

end simplify_expression_l223_223250


namespace distinct_sequences_of_four_letters_from_PROBLEM_l223_223425

def distinct_sequences_count : Finset String :=
  let letters := "P"::"R"::"O"::"B"::"L"::"E"::"M"::[];
  let valid_sequences := {s ∈ Finset.univ.filter (λ s : String, s.length = 4 ∧ s.front = 'L' ∧ s.back ≠ 'P') | 
                           s.chars.nodup ∧ s.toList.perm letters};
  valid_sequences.card

theorem distinct_sequences_of_four_letters_from_PROBLEM :
  distinct_sequences_count = 100 :=
sorry

end distinct_sequences_of_four_letters_from_PROBLEM_l223_223425


namespace exists_saddle_point_probability_l223_223309

noncomputable def saddle_point_probability := (3 : ℝ) / 10

theorem exists_saddle_point_probability {A : ℕ → ℕ → ℝ}
  (h : ∀ i j, 0 ≤ A i j ∧ A i j ≤ 1 ∧ (∀ k l, (i ≠ k ∨ j ≠ l) → A i j ≠ A k l)) :
  (∃ (p : ℝ), p = saddle_point_probability) :=
by 
  sorry

end exists_saddle_point_probability_l223_223309


namespace inclination_angle_expression_l223_223404

noncomputable def f (x : ℝ) : ℝ := (2 / 3) * x^3

noncomputable def f_prime (x : ℝ) : ℝ := 2 * x^2

theorem inclination_angle_expression :
  let α := Real.arctan 2 in
  (Real.sin α ^ 2 - Real.cos α ^ 2) / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = 3 / 5 := by
  sorry

end inclination_angle_expression_l223_223404


namespace circles_externally_tangent_l223_223209

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 2 * y + 1 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 4 * y - 1 = 0

theorem circles_externally_tangent :
  (∃ (A B : ℝ × ℝ) (R r : ℝ), A = (2, -1) ∧ B = (-2, 2) ∧ R = 2 ∧ r = 3 ∧
    dist A B = R + r) :=
begin
  sorry
end

end circles_externally_tangent_l223_223209


namespace part_I_part_II_l223_223415

-- Define the function f(x) = |x + 1| - 2 * |x - a|
def f (x a : ℝ) : ℝ := |x + 1| - 2 * |x - a|

-- First problem: when a = 1, prove the solution set of f(x) > 1 is { x | 2/3 < x < 2 }
theorem part_I (x : ℝ) : 
  let a := 1 in 
  f x a > 1 ↔ (2 / 3 < x ∧ x < 2) := 
by 
  sorry

-- Second problem: prove that if f(x) ≤ 0 for any x in ℝ, then a = -1
theorem part_II (a : ℝ) : 
  (∀ x : ℝ, f x a ≤ 0) ↔ a = -1 :=
by 
  sorry

end part_I_part_II_l223_223415


namespace weight_of_replaced_student_l223_223189

variable (W : ℝ) -- total weight of the original 10 students
variable (new_student_weight : ℝ := 60) -- weight of the new student
variable (weight_decrease_per_student : ℝ := 6) -- average weight decrease per student

theorem weight_of_replaced_student (replaced_student_weight : ℝ) :
  (W - replaced_student_weight + new_student_weight = W - 10 * weight_decrease_per_student) →
  replaced_student_weight = 120 := by
  sorry

end weight_of_replaced_student_l223_223189


namespace area_of_sector_l223_223874

def radius : ℝ := 5
def central_angle : ℝ := 2

theorem area_of_sector : (1 / 2) * radius^2 * central_angle = 25 := by
  sorry

end area_of_sector_l223_223874


namespace max_roses_purchasable_l223_223545

theorem max_roses_purchasable 
  (price_individual : ℝ) (price_dozen : ℝ) (price_two_dozen : ℝ) (price_five_dozen : ℝ) 
  (discount_threshold : ℕ) (discount_rate : ℝ) (total_money : ℝ) : 
  (price_individual = 4.50) →
  (price_dozen = 36) →
  (price_two_dozen = 50) →
  (price_five_dozen = 110) →
  (discount_threshold = 36) →
  (discount_rate = 0.10) →
  (total_money = 680) →
  ∃ (roses : ℕ), roses = 364 :=
by
  -- Definitions based on conditions
  intros
  -- The proof steps have been omitted for brevity
  sorry

end max_roses_purchasable_l223_223545


namespace problem_1_problem_2_problem_3_l223_223671

-- Conditions
def students : List String := ["A", "B", "C", "D", "E"]

-- (1) How many different arrangements if A and B must stand next to each other
theorem problem_1 : 
  (count_permutations (group ["A", "B"] ["C", "D", "E"])) * 2 = 48 := 
sorry

-- (2) How many different arrangements if A and B must not stand next to each other
theorem problem_2 : 
  ((count_permutations ["C", "D", "E"]) * (choose 4 2) * 2) = 72 := 
sorry

-- (3) How many different arrangements if A cannot stand at the far left and B cannot stand at the far right
theorem problem_3 :
  ((count_permutations ["B", "C", "D", "E"]) +
   (choose 3 1) * (choose 3 1) * (count_permutations ["C", "D", "E"])) = 78 := 
sorry

-- Helper functions for counting permutations and combinations
def count_permutations (l : List String) : Nat := 
  factorial l.length

def factorial : Nat → Nat 
| 0 => 1
| n => n * factorial (n - 1)

def choose : Nat → Nat → Nat
| n, k => (factorial n) / ((factorial k) * (factorial (n - k)))

end problem_1_problem_2_problem_3_l223_223671


namespace problem2_l223_223034

noncomputable def problem1 (x y : ℝ) : Prop :=
  (x^2 / 9 + y^2 / 8 = 1 → ∃ (x y : ℝ), ((-3, 0), (3, 0), E(x, y) ∧ ((y / (x + 3)) * (y / (x - 3)) = - 8 / 9)))

theorem problem2 :
  ∀ (x1 y1 x0 y0 : ℝ),
    (x1^2 / 9 + y1^2 / 8 = 1) →
    (x0^2 / 9 + y0^2 / 8 = 1) →
    ∃ (xS xT : ℝ),
    (S(xS, 0) ∧ T(xT, 0) ∧
      |xS| * |xT| = 9) :=
by
  sorry

end problem2_l223_223034


namespace range_of_x_l223_223434

theorem range_of_x
  (x : ℝ)
  (h1 : ∀ m, -1 ≤ m ∧ m ≤ 4 → m * (x^2 - 1) - 1 - 8 * x < 0) :
  0 < x ∧ x < 5 / 2 :=
sorry

end range_of_x_l223_223434


namespace sum_of_numbers_l223_223225

-- Define the given conditions.
def S : ℕ := 30
def F : ℕ := 2 * S
def T : ℕ := F / 3

-- State the proof problem.
theorem sum_of_numbers : F + S + T = 110 :=
by
  -- Assume the proof here.
  sorry

end sum_of_numbers_l223_223225


namespace inequalities_must_be_true_l223_223139

theorem inequalities_must_be_true (x y a b : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) (h1 : x < a) (h2 : y < b) :
  (x + y < a + b) ∧ ¬(x - y < a - b) ∧ ¬(xy < ab) ∧ ¬(x / y < a / b) :=
by
  sorry

end inequalities_must_be_true_l223_223139


namespace solution_of_system_l223_223839

theorem solution_of_system (a b c : ℝ) (x y z : ℝ) (h1 : xy = a) (h2 : yz = b) (h3 : zx = c) (h4 : 0 < a) (h5 : 0 < b) (h6 : 0 < c) :
  (x = sqrt (ac/b) ∨ x = -sqrt (ac/b)) ∧ 
  (y = sqrt (ab/c) ∨ y = -sqrt (ab/c)) ∧ 
  (z = sqrt (bc/a) ∨ z = -sqrt (bc/a)) :=
sorry

end solution_of_system_l223_223839


namespace minimum_ceiling_height_l223_223661

noncomputable theory
open Real

def football_field_length := 90
def football_field_width := 60
def number_of_floodlights := 4

def diagonal_length : ℝ := sqrt (football_field_length^2 + football_field_width^2)
def required_height (l w : ℝ) := (15 * sqrt (13)) / 2

theorem minimum_ceiling_height :
  ∃ (h : ℝ), ((h = (required_height football_field_length football_field_width)) ∧
  (h * 10) % 1 = 0.1) :=
begin
  let h := required_height football_field_length football_field_width,
  use h,
  split,
  { rw required_height, 
    simp_rw [football_field_length, football_field_width, sqrt_eq_rpow, ← rpow_mul, sqrt_mul_self],
    norm_num, },
  { simp_rw [required_height],
    have : 15 * sqrt 13 ≈ 54.0658, { sorry },
    have : 54.0658 / 2 ≈ 27.0429, { sorry },
    suffices : 27.0429 ≈ 27.1, { sorry },
    exact_mod_cast rfl },
end

end minimum_ceiling_height_l223_223661


namespace probability_diamond_then_ace_l223_223616

theorem probability_diamond_then_ace :
  let total_cards := 104
  let diamonds := 26
  let aces := 8
  let remaining_cards_after_first_draw := total_cards - 1
  let ace_of_diamonds_prob := (2 : ℚ) / total_cards
  let any_ace_after_ace_of_diamonds := (7 : ℚ) / remaining_cards_after_first_draw
  let combined_prob_ace_of_diamonds_then_any_ace := ace_of_diamonds_prob * any_ace_after_ace_of_diamonds
  let diamond_not_ace_prob := (24 : ℚ) / total_cards
  let any_ace_after_diamond_not_ace := (8 : ℚ) / remaining_cards_after_first_draw
  let combined_prob_diamond_not_ace_then_any_ace := diamond_not_ace_prob * any_ace_after_diamond_not_ace
  let total_prob := combined_prob_ace_of_diamonds_then_any_ace + combined_prob_diamond_not_ace_then_any_ace
  total_prob = (31 : ℚ) / 5308 :=
by
  sorry

end probability_diamond_then_ace_l223_223616


namespace lucia_dance_classes_cost_l223_223133

theorem lucia_dance_classes_cost:
  let hip_hop_classes_per_week := 2
  let ballet_classes_per_week := 2
  let jazz_class_per_week := 1
  let cost_per_hip_hop_class := 10
  let cost_per_ballet_class := 12
  let cost_per_jazz_class := 8
  let total_cost := hip_hop_classes_per_week * cost_per_hip_hop_class +
                     ballet_classes_per_week * cost_per_ballet_class +
                     jazz_class_per_week * cost_per_jazz_class
  in total_cost = 52 :=
by
  sorry

end lucia_dance_classes_cost_l223_223133


namespace extrema_of_y_on_interval_l223_223760

noncomputable def y (x : ℝ) : ℝ := x / 8 + 2 / x

theorem extrema_of_y_on_interval : 
  ∃ x_max x_min, (x_max ∈ Set.Ioo (-5 : ℝ) 10 ∧ x_min ∈ Set.Ioo (-5 : ℝ) 10) ∧ 
  (y x_max = -1) ∧ (y x_min = 1) :=
by
  use -4, 4
  split
  sorry
  sorry
  sorry

end extrema_of_y_on_interval_l223_223760


namespace sum_f_over_subsets_formula_l223_223366

def largest_minus_smallest (A : set ℕ) (hA : A.nonempty) : ℕ :=
  (A.max' hA) - (A.min' hA)

noncomputable def sum_f_over_subsets (n : ℕ) : ℕ :=
  ∑ A in (finset.powerset (finset.range (n + 1))).filter (λ s, ∃ (a : ℕ), a ∈ s), 
    largest_minus_smallest A (by sorry) -- proof of nonempty assumption omitted

theorem sum_f_over_subsets_formula (n : ℕ) : 
  sum_f_over_subsets n = (n - 3) * 2^n + n + 3 :=
sorry

end sum_f_over_subsets_formula_l223_223366


namespace sin_max_value_l223_223911

open Real

theorem sin_max_value (a b : ℝ) (h : sin (a + b) = sin a + sin b) :
  sin a ≤ 1 :=
by
  sorry

end sin_max_value_l223_223911


namespace add_to_fraction_eq_l223_223255

theorem add_to_fraction_eq (n : ℤ) : (3 + n : ℚ) / (5 + n) = 5 / 6 → n = 7 := 
by
  sorry

end add_to_fraction_eq_l223_223255


namespace incenter_inside_square_l223_223875

variables {A B C K L M N I : Type}

-- Define necessary geometric conditions
def is_triangle (A B C : Type) : Prop := sorry
def is_square (K L M N : Type) : Prop := sorry
def is_incenter (I : Type) (A B C : Type) : Prop := sorry
def lies_on (K L : Type) (AB : Type) : Prop := sorry
def lies_on_side (M : Type) (AC : Type) : Prop := sorry
def lies_on_side (N : Type) (BC : Type) : Prop := sorry
def inside (I : Type) (square : Type) : Prop := sorry

-- Statement of the proof problem
theorem incenter_inside_square {A B C K L M N I : Type}
    (H1 : is_triangle A B C)
    (H2 : is_square K L M N)
    (H3 : lies_on K L AB)
    (H4 : lies_on_side M AC)
    (H5 : lies_on_side N BC)
    (H6 : is_incenter I A B C) :
  inside I (K, L, M, N) := 
sorry

end incenter_inside_square_l223_223875


namespace discount_percentage_l223_223167

theorem discount_percentage (SP CP SP' discount_gain_percentage: ℝ) 
  (h1 : SP = 30) 
  (h2 : SP = CP + 0.25 * CP) 
  (h3 : SP' = CP + 0.125 * CP) 
  (h4 : discount_gain_percentage = ((SP - SP') / SP) * 100) :
  discount_gain_percentage = 10 :=
by
  -- Skipping the proof
  sorry

end discount_percentage_l223_223167


namespace combine_heaps_l223_223130

def heaps_similar (x y : ℕ) : Prop :=
  x ≤ 2 * y ∧ y ≤ 2 * x

theorem combine_heaps (n : ℕ) : 
  ∃ f : ℕ → ℕ, 
  f 0 = n ∧
  ∀ k, k < n → (∃ i j, i + j = k ∧ heaps_similar (f i) (f j)) ∧ 
  (∃ k, f k = n) :=
by
  sorry

end combine_heaps_l223_223130


namespace contradict_HE_eq_EG_l223_223842

-- Given a triangle ABC
variables (A B C : Type) [LinearOrder A] [LinearOrder B] [LinearOrder C]

-- Define the medians AE, BF, CD
variables (E F D H G : Type) 
variables (AE BF CD FH BH HE FE : A) -- Medians and lines involved

-- Given conditions
variables (isMedian_AE : AE = E) (isMedian_BF : BF = F) (isMedian_CD : CD = D)
variables (parallel_FH_AE : FH = AE) (equal_FH_AE : FH = AE)
variables (connect_BH_HE : BH) (intersect_FE_BH_G : G)

-- Main goal: Prove that we cannot necessarily have HE = EG
theorem contradict_HE_eq_EG :
  ¬ (HE = EG) := by 
  sorry

end contradict_HE_eq_EG_l223_223842


namespace cheesecake_factory_savings_l223_223542

noncomputable def combined_savings : ℕ := 3000

theorem cheesecake_factory_savings :
  let hourly_wage := 10
  let daily_hours := 10
  let working_days := 5
  let weekly_hours := daily_hours * working_days
  let weekly_salary := weekly_hours * hourly_wage
  let robby_savings := (2/5 : ℚ) * weekly_salary
  let jaylen_savings := (3/5 : ℚ) * weekly_salary
  let miranda_savings := (1/2 : ℚ) * weekly_salary
  let combined_weekly_savings := robby_savings + jaylen_savings + miranda_savings
  4 * combined_weekly_savings = combined_savings :=
by
  sorry

end cheesecake_factory_savings_l223_223542


namespace tan_sub_add_is_18_l223_223433

noncomputable def angle_sub_add_tangent (α β γ : ℝ) : ℝ :=
  let tan_alpha := 5
  let tan_beta := 2
  let tan_gamma := 3
  have tan_ab := (tan_alpha - tan_beta) / (1 + tan_alpha * tan_beta) -- tangent subtraction formula
  have tan_abc := (tan_ab + tan_gamma) / (1 - tan_ab * tan_gamma) -- tangent addition formula
  tan_abc

theorem tan_sub_add_is_18 (α β γ : ℝ) (hα : tan α = 5) (hβ : tan β = 2) (hγ : tan γ = 3):
  tan (α - β + γ) = 18 := by
  sorry

end tan_sub_add_is_18_l223_223433


namespace simplify_exponent_l223_223170

theorem simplify_exponent (y : ℝ) : (3 * y^4)^5 = 243 * y^20 :=
by
  sorry

end simplify_exponent_l223_223170


namespace inequality_solution_l223_223631

theorem inequality_solution (a : ℝ) : (∀ x : ℝ, x > 1 → x + 1 / (x - 1) ≥ a) → a ≤ 3 :=
by
  sorry

end inequality_solution_l223_223631


namespace find_vertex_P_l223_223043

theorem find_vertex_P 
(mid_QR : (2 : ℝ), 7, 2)
(mid_PR : (3 : ℝ), 5, -3)
(mid_PQ : (1 : ℝ), 8, 5) :
  ∃ (P : ℝ × ℝ × ℝ), P = (2, 6, 0) :=
sorry

end find_vertex_P_l223_223043


namespace heaps_combination_preserve_similarity_split_stones_into_similar_heaps_l223_223120

def initial_heaps (n : ℕ) : list ℕ := list.repeat 1 n

def combine_heaps (heaps : list ℕ) : list ℕ :=
  if heaps.length ≥ 2 then
    let min1 := list.minimum heaps,
        heaps' := list.erase heaps min1,
        min2 := list.minimum heaps'
    in
    if min1 ≤ min2 then
      (min1 + min2) :: list.erase heaps' min2
    else
      heaps
  else
    heaps

theorem heaps_combination_preserve_similarity (heaps : list ℕ) (h : ∀ x ∈ heaps, x = 1) :
  ∀ combined_heaps, combined_heaps = combine_heaps heaps →
  ∀ x y ∈ combined_heaps, x ≤ y → x + y ≤ 2 * y :=
sorry

theorem split_stones_into_similar_heaps (n : ℕ) :
  ∃ combined_heaps : list ℕ, ∀ x y ∈ combined_heaps, x ≤ y → x + y ≤ 2 * y :=
sorry

end heaps_combination_preserve_similarity_split_stones_into_similar_heaps_l223_223120


namespace heaps_combination_preserve_similarity_split_stones_into_similar_heaps_l223_223119

def initial_heaps (n : ℕ) : list ℕ := list.repeat 1 n

def combine_heaps (heaps : list ℕ) : list ℕ :=
  if heaps.length ≥ 2 then
    let min1 := list.minimum heaps,
        heaps' := list.erase heaps min1,
        min2 := list.minimum heaps'
    in
    if min1 ≤ min2 then
      (min1 + min2) :: list.erase heaps' min2
    else
      heaps
  else
    heaps

theorem heaps_combination_preserve_similarity (heaps : list ℕ) (h : ∀ x ∈ heaps, x = 1) :
  ∀ combined_heaps, combined_heaps = combine_heaps heaps →
  ∀ x y ∈ combined_heaps, x ≤ y → x + y ≤ 2 * y :=
sorry

theorem split_stones_into_similar_heaps (n : ℕ) :
  ∃ combined_heaps : list ℕ, ∀ x y ∈ combined_heaps, x ≤ y → x + y ≤ 2 * y :=
sorry

end heaps_combination_preserve_similarity_split_stones_into_similar_heaps_l223_223119


namespace inradius_length_l223_223303

-- Define the side lengths of the triangle
def a : ℝ := 7
def b : ℝ := 11
def c : ℝ := 14

-- Define the semiperimeter of the triangle
def s : ℝ := (a + b + c) / 2

-- Define the area of the triangle using Heron's formula
def A : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Define the inradius of the triangle
def r : ℝ := A / s

-- The theorem to check that the inradius r satisfies the given answer
theorem inradius_length : r = (3 * Real.sqrt 10) / 4 :=
by
  -- The proof can be skipped, just include 'sorry' to indicate it
  sorry

end inradius_length_l223_223303


namespace withdraw_from_three_cards_probability_withdraw_all_four_l223_223998

namespace ThiefProblem

-- Definitions of the problem conditions
structure ProblemState where
  cards : Fin 4
  pin_codes : Fin 4
  attempts : Fin 4 → ℕ -- number of attempts made per card

-- Problem (a): Kirpich can always withdraw money from three cards
theorem withdraw_from_three_cards (s : ProblemState) : ∃ (cards_to_succeed : Nat), cards_to_succeed ≥ 3 :=
sorry

-- Problem (b): The probability of withdrawing money from all four cards is 23/24
theorem probability_withdraw_all_four : (23 : ℚ) / 24 =
  1 - ((1/4) * (1/3) * (1/2)) :=
sorry

end ThiefProblem

end withdraw_from_three_cards_probability_withdraw_all_four_l223_223998


namespace find_a_values_l223_223585

theorem find_a_values (a : ℝ) : 
  (∃ x : ℝ, (a * x^2 + (a - 3) * x + 1 = 0)) ∧ 
  (∀ x1 x2 : ℝ, (a * x1^2 + (a - 3) * x1 + 1 = 0 ∧ a * x2^2 + (a - 3) * x2 + 1 = 0 → x1 = x2)) 
  ↔ a = 0 ∨ a = 1 ∨ a = 9 :=
sorry

end find_a_values_l223_223585


namespace zero_solution_stability_l223_223892

-- Define the coefficients in the differential equation
def a0 := 1
def a1 := 5
def a2 := 13
def a3 := 19
def a4 := 10

-- Define the stability of the zero solution of the given equation
theorem zero_solution_stability :
  is_asymptotically_stable (λ y : ℝ, y^4 + a1 * y^3 + a2 * y^2 + a3 * y + a4) 0 :=
sorry

end zero_solution_stability_l223_223892


namespace solve_inequality_l223_223554

theorem solve_inequality :
  ∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 →
  (2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 15) ↔
  (x < 1 ∨ (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ (6 < x ∧ x < 8) ∨ (10 < x)).

end solve_inequality_l223_223554


namespace no_real_solution_for_inequality_l223_223337

theorem no_real_solution_for_inequality :
  ∀ x : ℝ, ¬(3 * x^2 - x + 2 < 0) :=
by
  sorry

end no_real_solution_for_inequality_l223_223337


namespace jane_change_l223_223492

noncomputable def calculate_change (apple_cost : ℝ) (sandwich_cost : ℝ) (soda_cost : ℝ)
  (apple_tax_rate : ℝ) (sandwich_tax_rate : ℝ) (amount_paid : ℝ) : ℝ :=
let apple_total := apple_cost * (1 + apple_tax_rate),
    sandwich_total := sandwich_cost * (1 + sandwich_tax_rate),
    total_cost := apple_total + sandwich_total + soda_cost,
    change := amount_paid - total_cost in
Real.round change

theorem jane_change :
  calculate_change 0.75 3.50 1.25 0.03 0.06 20 = 14.27 :=
by
  sorry

end jane_change_l223_223492


namespace find_f_9_over_2_l223_223929

noncomputable def f (x : ℝ) : ℝ := sorry

axiom domain_of_f : ∀ x : ℝ, ∃ f(x) -- The domain of f(x) is ℝ

axiom odd_f_shift : ∀ x : ℝ, f(x + 1) = -f(-x + 1) -- f(x+1) is an odd function

axiom even_f_shift : ∀ x : ℝ, f(x + 2) = f(-x + 2) -- f(x+2) is an even function

axiom function_segment : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f(x) = a * x^2 + b -- For x in [1,2], f(x) = ax^2 + b

axiom sum_f0_f3 : f(0) + f(3) = 6 -- Given f(0) + f(3) = 6

theorem find_f_9_over_2 : f (9 / 2) = 5 / 2 := sorry

end find_f_9_over_2_l223_223929


namespace fraction_zero_numerator_l223_223865

theorem fraction_zero_numerator (x : ℝ) (h₁ : (x^2 - 9) / (x + 3) = 0) (h₂ : x + 3 ≠ 0) : x = 3 :=
sorry

end fraction_zero_numerator_l223_223865


namespace sin_alpha_value_l223_223787

def f (x : ℝ) := 2 * sin x * cos x + 2 * cos (x + π / 4) * cos (x - π / 4)

theorem sin_alpha_value (α : ℝ) (hα : 0 < α ∧ α < π) (hf : f (α / 2) = sqrt 2 / 2) :
  sin α = (sqrt 6 + sqrt 2) / 4 :=
by
  sorry

end sin_alpha_value_l223_223787


namespace pears_can_be_rearranged_l223_223997

variable {n : ℕ} (h : n % 2 = 0)
variable (weights : Finₙ → ℕ) -- weights of pears, should be even in number
variable (adj_diff_le_one : ∀ i : Finₙ, |weights i - weights (i + 1 % n)| ≤ 1)

theorem pears_can_be_rearranged (h_even : n % 2 = 0) 
  (hw : ∀ i : Finₙ, |weights i - weights (i + 1 % n)| ≤ 1) :
  ∃ pairs : list (Finₙ × Finₙ), (∀ i j, i ≠ j → pairs.nth i ≠ pairs.nth j) ∧
    (∀ i, pairs.nth i ≠ none → 
     let (a, b) := option.get (pairs.nth i) in
     |weights a + weights b - (weights (option.get (pairs.nth ((i + 1) % (n / 2) ))) .fst + weights (option.get (pairs.nth ((i + 1) % (n / 2) ))).snd)| ≤ 1) :=
sorry

end pears_can_be_rearranged_l223_223997


namespace part1_part2_l223_223039

noncomputable def A (x : ℝ) (k : ℝ) := -2 * x ^ 2 - (k - 1) * x + 1
noncomputable def B (x : ℝ) := -2 * (x ^ 2 - x + 2)

-- Part 1: If A is a quadratic binomial, then the value of k is 1
theorem part1 (x : ℝ) (k : ℝ) (h : ∀ x, A x k ≠ 0) : k = 1 :=
sorry

-- Part 2: When k = -1, C + 2A = B, then C = 2x^2 - 2x - 6
theorem part2 (x : ℝ) (C : ℝ → ℝ) (h1 : k = -1) (h2 : ∀ x, C x + 2 * A x k = B x) : (C x = 2 * x ^ 2 - 2 * x - 6) :=
sorry

end part1_part2_l223_223039


namespace tangent_line_at_point_l223_223196

theorem tangent_line_at_point :
  let f := λ x : ℝ => x^3 - 3*x^2 + 3
  let f_deriv := deriv f
  let slope_at_1 := f_deriv 1
  let tangent_line := λ x : ℝ => slope_at_1 * (x - 1) + 1
  tangent_line = λ x : ℝ => -3 * x + 4 :=
by
  let f := λ x : ℝ => x^3 - 3*x^2 + 3
  let f_deriv := deriv f
  let slope_at_1 := f_deriv 1
  let tangent_line := λ x : ℝ => slope_at_1 * (x - 1) + 1
  show tangent_line = λ x : ℝ => -3 * x + 4
  sorry

end tangent_line_at_point_l223_223196


namespace solve_quadratic_l223_223948

theorem solve_quadratic (y : ℝ) :
  y^2 - 3 * y - 10 = -(y + 2) * (y + 6) ↔ (y = -1/2 ∨ y = -2) :=
by
  sorry

end solve_quadratic_l223_223948


namespace sin_product_identity_l223_223531

theorem sin_product_identity (α : ℝ) :
  (Finset.range 36).prod (λ n, Real.sin (α + n * 5 * Real.pi / 180)) 
  = Real.sin (36 * α) / 2^35 :=
  sorry

end sin_product_identity_l223_223531


namespace required_ratio_l223_223504

-- Define that a sequence is arithmetic
def is_arithmetic_sequence (u : ℕ → ℤ) : Prop :=
∃ d : ℤ, ∀ n : ℕ, u (n + 1) - u n = d

-- Definitions for sequences and sums
variable {a b : ℕ → ℤ}

-- Define the sums
def S (n : ℕ) := (n.succ * (a 0 + a n)) / 2
def T (n : ℕ) := (n.succ * (b 0 + b n)) / 2

-- Condition: sequences and sum ratio
axiom h_a : is_arithmetic_sequence a
axiom h_b : is_arithmetic_sequence b
axiom sum_ratio (n : ℕ) : S n * (4 * n - 3) = T n * (2 * n - 3)

-- Define the values we are interested in
noncomputable def a2_b3_b13_a14_b5_b11_ratio : ℚ :=
(a 2) / (b 3 + b 13) + (a 14) / (b 5 + b 11)

-- The main theorem
theorem required_ratio : a2_b3_b13_a14_b11 = 9 / 19 :=
by sorry

end required_ratio_l223_223504


namespace infinite_sum_of_sequence_b_l223_223505

def sequence_b : ℕ → ℝ
| 0     := 0
| 1     := 1
| 2     := 1
| (n+3) := sequence_b (n+1) + sequence_b (n+2)

theorem infinite_sum_of_sequence_b :
  (∑' n, sequence_b (n+1) / 3^(n+2)) = 1 / 5 :=
by
  sorry

end infinite_sum_of_sequence_b_l223_223505


namespace equations_of_line_AB_minimum_value_AB_l223_223418

noncomputable def parabola (x : ℝ) : ℝ := x^2

def point_P : ℝ × ℝ := (0, 2)

def distance_to_line (P : ℝ × ℝ) (m k : ℝ) : ℝ := 
  abs (m - P.2) / real.sqrt (1 + k^2)

theorem equations_of_line_AB
  (x : ℝ)
  (C : ∀ x : ℝ, parabola x = x^2)
  (P : point_P = (0, 2))
  (k : ℝ)
  (d : distance_to_line (0, 2) k m = 1)
  (inclination : k = real.sqrt 3 )
  : ∃ m : ℝ, (∀ x : ℝ, (y = k * x + m) ∨ (y = k * x + 4)) := 
sorry

theorem minimum_value_AB
  (x : ℝ)
  (y : parabola x = x^2)
  (P : point_P = (0, 2))
  (k : ℝ)
  (d : distance_to_line (0, 2) k m = 1)
  : ∃ m : ℝ, (∀ x₁ x₂ : ℝ, (|AB|^2 = 4)) := 
sorry

end equations_of_line_AB_minimum_value_AB_l223_223418


namespace carA_catches_up_with_carB_at_150_km_l223_223710

-- Definitions representing the problem's conditions
variable (t_A t_B v_A v_B : ℝ)
variable (distance_A_B : ℝ := 300)
variable (time_diff_start : ℝ := 1)
variable (time_diff_end : ℝ := 1)

-- Assumptions representing the problem's conditions
axiom speed_carA : v_A = distance_A_B / t_A
axiom speed_carB : v_B = distance_A_B / (t_A + 2)
axiom time_relation : t_B = t_A + 2
axiom time_diff_starting : t_A = t_B - 2

-- The statement to be proven: car A catches up with car B 150 km from city B
theorem carA_catches_up_with_carB_at_150_km :
  ∃ t₀ : ℝ, v_A * t₀ = v_B * (t₀ + time_diff_start) ∧ (distance_A_B - v_A * t₀ = 150) :=
sorry

end carA_catches_up_with_carB_at_150_km_l223_223710


namespace div_by_3_implies_one_div_by_3_l223_223246

theorem div_by_3_implies_one_div_by_3 (a b : ℕ) (h_ab : 3 ∣ (a * b)) (h_na : ¬ 3 ∣ a) (h_nb : ¬ 3 ∣ b) : false :=
sorry

end div_by_3_implies_one_div_by_3_l223_223246


namespace appropriate_method_for_investigating_colorant_l223_223238

variable (food : Type) (market : set food) (colorantStandard : food → Prop)
variable (comprehensiveSurvey samplingSurvey : set food → Prop)

axiom large_quantity : set.finite market = false
axiom destructive_testing : ∀ (f : food), colorantStandard f → market f → market \ {f} ≠ market

theorem appropriate_method_for_investigating_colorant :
  samplingSurvey market :=
sorry

end appropriate_method_for_investigating_colorant_l223_223238


namespace part1_part2_l223_223057

noncomputable theory
open Classical

variables {f : ℝ → ℝ}

-- Given conditions
axiom A1 : ∀ (x y : ℝ), (0 < x) → (0 < y) → f(x * y) = f(x) + f(y)
axiom A2 : ∀ (x : ℝ), (1 < x) →  0 < f(x)
axiom A3 : f(9) = 8

-- Prove: The monotonicity of f(x)
theorem part1 (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : x1 > x2) : f(x1) > f(x2) :=
sorry

-- Prove: Solution set for the inequality
theorem part2 (x : ℝ) (h : 0 < x) : f(x^2 - 2 * x) - f(2 - x) < 4 ↔ -3 < x ∧ x < 0 :=
sorry

end part1_part2_l223_223057


namespace coffee_machine_pays_off_l223_223490

def daily_savings(previous_cost: ℕ, current_cost: ℕ) : ℕ :=
  previous_cost - current_cost

def days_to_payoff(machine_cost: ℕ, savings_per_day: ℕ) : ℕ :=
  machine_cost / savings_per_day

theorem coffee_machine_pays_off :
  let machine_cost := 200 - 20
  let previous_daily_expense := 2 * 4
  let current_daily_expense := 3
  let savings := daily_savings previous_daily_expense current_daily_expense
  let payoff_days := days_to_payoff machine_cost savings
  payoff_days = 36 :=
by
  -- Calculation/Proof goes here
  sorry

end coffee_machine_pays_off_l223_223490


namespace smith_family_mean_age_l223_223187

theorem smith_family_mean_age :
  let children_ages := [8, 8, 8, 12, 11]
  let dogs_ages := [3, 4]
  let all_ages := children_ages ++ dogs_ages
  let total_ages := List.sum all_ages
  let total_individuals := List.length all_ages
  (total_ages : ℚ) / (total_individuals : ℚ) = 7.71 :=
by
  sorry

end smith_family_mean_age_l223_223187


namespace split_piles_equiv_single_stone_heaps_l223_223095

theorem split_piles_equiv_single_stone_heaps (n : ℕ) (heaps : List ℕ) (h_initial : ∀ h ∈ heaps, h = 1)
  (h_size : heaps.length = n) :
  ∃ final_heap, (∀ x y ∈ heaps, x + y ≤ 2 * max x y) ∧ (List.sum heaps = (heaps.length) * 1) := by
  sorry

end split_piles_equiv_single_stone_heaps_l223_223095


namespace find_first_term_l223_223217

noncomputable def firstTermOfGeometricSeries (a r : ℝ) : Prop :=
  (a / (1 - r) = 30) ∧ (a^2 / (1 - r^2) = 120)

theorem find_first_term :
  ∃ a r : ℝ, firstTermOfGeometricSeries a r ∧ a = 120 / 17 :=
by
  sorry

end find_first_term_l223_223217


namespace complex_value_solution_1_inequality_proof_l223_223785

open Complex

theorem complex_value_solution_1 (a b : ℝ) (ha : (a, b) = (3, 1) ∨ (a, b) = (2, 3/2)) : 
(z : ℂ) = (a + 2 * Complex.i) * (1 - b * Complex.i) → z = 5 - Complex.i := sorry

theorem inequality_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
(h3 : a + 2 * b = 2) : 2 / a + 1 / b ≥ 4 := 
sorry

end complex_value_solution_1_inequality_proof_l223_223785


namespace degree_poly_4_l223_223328

noncomputable def f : (Polynomial ℝ) :=
  1 - 12 * (Polynomial.X) + 3 * (Polynomial.X ^ 2) - 4 * (Polynomial.X ^ 3) + 5 * (Polynomial.X ^ 4)

noncomputable def g : (Polynomial ℝ) :=
  3 - 2 * (Polynomial.X) - 6 * (Polynomial.X ^ 3) + 8 * (Polynomial.X ^ 4) + (Polynomial.X ^ 5)

theorem degree_poly_4 (c : ℝ) (h : c = 0) : 
  (f + c * g).degree = 4 :=
by
  sorry

end degree_poly_4_l223_223328


namespace general_solution_l223_223275

noncomputable def satisfies_relations (f : ℝ → ℝ → ℝ) :=
  (∀ x y u : ℝ, f (x + u) (y + u) = f x y + u) ∧
  (∀ x y v : ℝ, f (x * v) (y * v) = f x y * v)

theorem general_solution (f : ℝ → ℝ → ℝ) (p q : ℝ) :
  satisfies_relations f →
  (f = λ x y, p * x + q * y) →
  (p + q = 1) :=
by sorry

end general_solution_l223_223275


namespace pile_splitting_l223_223114

theorem pile_splitting (single_stone_piles : ℕ) :
  ∃ (final_heap_size : ℕ), 
    (∀ heap_size ≤ single_stone_piles, heap_size > 0 → (heap_size * 2) ≥ heap_size) ∧ (final_heap_size = single_stone_piles) :=
by
  sorry

end pile_splitting_l223_223114


namespace smallest_angle_in_triangle_l223_223459

open Real

theorem smallest_angle_in_triangle
  (a b c : ℝ)
  (h : a = (b + c) / 3)
  (triangle_inequality_1 : a + b > c)
  (triangle_inequality_2 : a + c > b)
  (triangle_inequality_3 : b + c > a) :
  ∃ A B C α β γ : ℝ, -- A, B, C are the angles opposite to sides a, b, c respectively
  0 < α ∧ α < β ∧ α < γ :=
sorry

end smallest_angle_in_triangle_l223_223459


namespace linlin_speed_l223_223942

theorem linlin_speed (distance time : ℕ) (q_speed linlin_speed : ℕ)
  (h1 : distance = 3290)
  (h2 : time = 7)
  (h3 : q_speed = 70)
  (h4 : distance = (q_speed + linlin_speed) * time) : linlin_speed = 400 :=
by sorry

end linlin_speed_l223_223942


namespace possible_results_l223_223961

theorem possible_results : ∃ (a b c d e : ℕ), 
    {a, b, c, d, e} = {1, 2, 3, 4, 5} ∧ 
    (a * b - (c / (d + e)) ∈ {3, 5, 9, 19}) ∧ 
    d + e ≠ 0 ∧ 
    (c % (d + e)) = 0 :=
by
  sorry

end possible_results_l223_223961


namespace carA_catches_up_with_carB_at_150_km_l223_223709

-- Definitions representing the problem's conditions
variable (t_A t_B v_A v_B : ℝ)
variable (distance_A_B : ℝ := 300)
variable (time_diff_start : ℝ := 1)
variable (time_diff_end : ℝ := 1)

-- Assumptions representing the problem's conditions
axiom speed_carA : v_A = distance_A_B / t_A
axiom speed_carB : v_B = distance_A_B / (t_A + 2)
axiom time_relation : t_B = t_A + 2
axiom time_diff_starting : t_A = t_B - 2

-- The statement to be proven: car A catches up with car B 150 km from city B
theorem carA_catches_up_with_carB_at_150_km :
  ∃ t₀ : ℝ, v_A * t₀ = v_B * (t₀ + time_diff_start) ∧ (distance_A_B - v_A * t₀ = 150) :=
sorry

end carA_catches_up_with_carB_at_150_km_l223_223709


namespace gcd_lcm_product_l223_223354

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 24) (h2 : b = 60) :
  Nat.gcd a b * Nat.lcm a b = 1440 :=
by
  sorry

end gcd_lcm_product_l223_223354


namespace max_sin_a_l223_223906

theorem max_sin_a (a b : ℝ) (h : Real.sin (a + b) = Real.sin a + Real.sin b) : Real.sin a ≤ 1 :=
by
  sorry

end max_sin_a_l223_223906


namespace heaps_combination_preserve_similarity_split_stones_into_similar_heaps_l223_223118

def initial_heaps (n : ℕ) : list ℕ := list.repeat 1 n

def combine_heaps (heaps : list ℕ) : list ℕ :=
  if heaps.length ≥ 2 then
    let min1 := list.minimum heaps,
        heaps' := list.erase heaps min1,
        min2 := list.minimum heaps'
    in
    if min1 ≤ min2 then
      (min1 + min2) :: list.erase heaps' min2
    else
      heaps
  else
    heaps

theorem heaps_combination_preserve_similarity (heaps : list ℕ) (h : ∀ x ∈ heaps, x = 1) :
  ∀ combined_heaps, combined_heaps = combine_heaps heaps →
  ∀ x y ∈ combined_heaps, x ≤ y → x + y ≤ 2 * y :=
sorry

theorem split_stones_into_similar_heaps (n : ℕ) :
  ∃ combined_heaps : list ℕ, ∀ x y ∈ combined_heaps, x ≤ y → x + y ≤ 2 * y :=
sorry

end heaps_combination_preserve_similarity_split_stones_into_similar_heaps_l223_223118


namespace sum_greater_than_0_4_l223_223254

theorem sum_greater_than_0_4 :
  let nums := [0.8, 1/2, 0.9, 1/3]
  let nums_as_dec := [0.8, 0.5, 0.9, 0.333]
  ∑ n in nums_as_dec.filter (λ x => x > 0.4), n = 2.2
:= by
  let nums := [0.8, 1/2, 0.9, 1/3]
  let nums_as_dec := [0.8, 0.5, 0.9, 0.333]
  let filtered_nums := nums_as_dec.filter (λ x => x > 0.4)
  have : filtered_nums = [0.8, 0.5, 0.9] := by sorry
  calc
    ∑ n in filtered_nums, n = 0.8 + 0.5 + 0.9 : by sorry
                   ... = 2.2 : by sorry

end sum_greater_than_0_4_l223_223254


namespace count_perfect_squares_l223_223071

open Int

def p (x : Int) : Int := 2 * x^3 - 3 * x^2 + 1

theorem count_perfect_squares (n : Int) (h : 1 ≤ n ∧ n ≤ 2016) : 
  (finset.filter (λ x, ∃ (n : Int), p x = n * n) (finset.range 2016)).card = 32 :=
by
  sorry

end count_perfect_squares_l223_223071


namespace angle_is_pi_over_2_l223_223350

noncomputable def angle_between_planes (p1 p2 : ℝ → ℝ → ℝ → Prop) : ℝ :=
  let n1 := (3, -1, 2)
  let n2 := (5, 9, -3)
  let dot_product := n1.1 * n2.1 + n1.2 * n2.2 + n1.3 * n2.3
  let magnitude1 := real.sqrt (n1.1 ^ 2 + n1.2 ^ 2 + n1.3 ^ 2)
  let magnitude2 := real.sqrt (n2.1 ^ 2 + n2.2 ^ 2 + n2.3 ^ 2)
  real.arccos (dot_product / (magnitude1 * magnitude2))

theorem angle_is_pi_over_2 : angle_between_planes 
  (λ x y z => 3 * x - y + 2 * z + 15 = 0)
  (λ x y z => 5 * x + 9 * y - 3 * z - 1 = 0) = real.pi / 2 := 
by sorry

end angle_is_pi_over_2_l223_223350


namespace number_of_ordered_quadruples_l223_223062

theorem number_of_ordered_quadruples :
  let n := (finset.range 50).powerset.filter (λ s, s.card = 4 ∧ ∑ i in s, 2 * (i + 1) - 1 = 100).card in
  (n : ℝ) / 100 = 208.25 :=
sorry

end number_of_ordered_quadruples_l223_223062


namespace minimum_ceiling_height_l223_223662

noncomputable theory
open Real

def football_field_length := 90
def football_field_width := 60
def number_of_floodlights := 4

def diagonal_length : ℝ := sqrt (football_field_length^2 + football_field_width^2)
def required_height (l w : ℝ) := (15 * sqrt (13)) / 2

theorem minimum_ceiling_height :
  ∃ (h : ℝ), ((h = (required_height football_field_length football_field_width)) ∧
  (h * 10) % 1 = 0.1) :=
begin
  let h := required_height football_field_length football_field_width,
  use h,
  split,
  { rw required_height, 
    simp_rw [football_field_length, football_field_width, sqrt_eq_rpow, ← rpow_mul, sqrt_mul_self],
    norm_num, },
  { simp_rw [required_height],
    have : 15 * sqrt 13 ≈ 54.0658, { sorry },
    have : 54.0658 / 2 ≈ 27.0429, { sorry },
    suffices : 27.0429 ≈ 27.1, { sorry },
    exact_mod_cast rfl },
end

end minimum_ceiling_height_l223_223662


namespace polygon_sides_l223_223814

theorem polygon_sides {n : ℕ} (h : (n - 2) * 180 = 1080) : n = 8 :=
by
  sorry

end polygon_sides_l223_223814


namespace solution_l223_223243

noncomputable def transformation_proof_problem : Prop :=
  let A := (0, 0)
  let B := (0, 10)
  let C := (18, 0)
  let D := (45, 30)
  let E := (63, 30)
  let F := (45, 10)
  ∃ (n p q k : ℝ), 0 < n ∧ n < 180 ∧
  let rotate := λ (x y θ : ℝ), (x * Real.cos θ - y * Real.sin θ, x * Real.sin θ + y * Real.cos θ)
  let scale := λ (p q k (x y : ℝ)), (k * (x - p) + p, k * (y - q) + q)
  let P := (rotate 0 0 n).1 + p in
  let Q := (rotate 0 0 n).2 + q in
  P = D.1 ∧ Q = D.2 ∧
  let R := (rotate 0 10 n).1 + p in
  let S := (rotate 0 10 n).2 + q in
  R = E.1 ∧ S = E.2 ∧
  let T := (rotate 18 0 n).1 + p in
  let U := (rotate 18 0 n).2 + q in
  T = F.1 ∧ U = F.2 ∧
  (n + p + q + k = 111)

theorem solution : transformation_proof_problem :=
by
  sorry

end solution_l223_223243


namespace find_angle_l223_223463

open Classical

variable {α : Type*} [LinearOrderedField α]

structure Triangle (α : Type*) [LinearOrderedField α] :=
(A B C : α × α)

structure Point (α : Type*) [LinearOrderedField α] :=
(x : α) (y : α)

def midpoint {α : Type*} [LinearOrderedField α] (P Q : Point α) : Point α :=
⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2⟩

def is_acute (T : Triangle α) : Prop :=
(T.A.1 < 90) ∧ (T.B.1 < 90) ∧ (T.C.1 < 90)

def altitude {α : Type*} [LinearOrderedField α] (P Q : Point α) (R : Point α) : Point α := sorry

noncomputable def intersection {α : Type*} [LinearOrderedField α] (l1 l2 : Point α → Prop) : Point α := sorry

theorem find_angle
  (T : Triangle α)
  (acute_T : is_acute T)
  (angle_A : T.A.1 = 35)
  (B1 : Point α)
  (C1 : Point α)
  (B2 : Point α := midpoint T.A T.C)
  (C2 : Point α := midpoint T.A T.B)
  (K : Point α := intersection (line B1 C2) (line C1 B2))
  (alt_BB1 : B1 = altitude T.B T.C T.A)
  (alt_CC1 : C1 = altitude T.C T.B T.A) :
  angle B1 K B2 = 75 :=
sorry

end find_angle_l223_223463


namespace circle_m_range_l223_223580

theorem circle_m_range (m : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 - 2 * x + 6 * y + m = 0 → m < 10) :=
sorry

end circle_m_range_l223_223580


namespace compute_AB_plus_AC_l223_223323

noncomputable theory
open_locale classical

variables {O A B C : ℝ} (ω : {radius : ℝ // radius = 3}) (BC : ℝ)

-- Define points and distances
def OA : ℝ := 10
def r : ℝ := ω.1

-- Conditions
def tangent_lengths : ℝ := real.sqrt (OA^2 - r^2)
def BC_length : BC = 9

-- Theorem statement
theorem compute_AB_plus_AC (h1 : OA = 10) 
                           (h2 : r = 3) 
                           (h3 : BC = 9) : 
  2 * real.sqrt (OA^2 - r^2) + BC = 2 * real.sqrt 91 + 9 := 
by sorry

end compute_AB_plus_AC_l223_223323


namespace concurrency_of_lines_l223_223959

open Function Real

variables {A B C : Type*} [metric_space A] [metric_space B] [metric_space C]
variables {AA1 AA' CC1 CC' BH H : A}
variables {B0A1 B0C1 : A → Prop}

-- Definitions and conditions
def is_acute_angled_triangle (ABC: A) : Prop := sorry -- To denote ABC being an acute-angled triangle
def altitude (A1 : A) (A : A) (C : A) : Prop := sorry -- To denote lines being altitudes 
def intersects (X Y H : A) : Prop := sorry  -- To denote intersection at H

-- Define concurrency of the lines
def concurrent (L1 L2 L3 : A → Prop) : Prop := sorry -- To denote concurrency

-- Given conditions
axiom cond_1 : is_acute_angled_triangle ABC
axiom cond_2 : altitude AA1 A C
axiom cond_3 : altitude CC1 C A
axiom cond_4 : intersects AA1 CC1 H
axiom cond_5 : ∀ X, (B0A1 X ↔ ∃ B (PP X = A)) -- Parallel to AC
axiom cond_6 : ∀ X, (B0C1 X ↔ ∃ C (PP X = A)) -- Parallel to AC

theorem concurrency_of_lines :
    concurrent (λ P, ∃ Q, altitude A A' Q ∧ Q P H)
                (λ P, ∃ Q, altitude C C' Q ∧ Q P H)
                (λ P, ∃ Q, ∃ H, H P H) := by
  sorry

end concurrency_of_lines_l223_223959


namespace length_of_platform_l223_223685

-- Define the conditions given in the problem
variables (t_pass_platform t_pass_man : ℝ) (speed_kmh speed_ms : ℝ)
variables (L P : ℝ)

-- Specify the known values
axiom h1 : t_pass_platform = 35
axiom h2 : t_pass_man = 20
axiom h3 : speed_kmh = 54
axiom h4 : speed_ms = speed_kmh * (1000 / 3600) -- Convert speed from km/hr to m/s
axiom h5 : t_pass_man = L / speed_ms
axiom h6 : t_pass_platform = (L + P) / speed_ms

-- The goal is to prove that the length of the platform P is 225 meters
theorem length_of_platform : P = 225 :=
by
  have h_speed: speed_ms = 15, from calc
    speed_ms = 54 * (1000 / 3600) : by rw [h4]
          ... = 54000 / 3600   : by norm_num
          ... = 15             : by norm_num,

  have h_L: L = 20 * 15, from calc
    L = t_pass_man * speed_ms : by rw [h5, h2]
      ... = 20 * 15            : by rw [h_speed]
      ... = 300                : by norm_num,

  sorry -- Complete proof for P = 225

end length_of_platform_l223_223685


namespace age_of_youngest_child_l223_223986

theorem age_of_youngest_child (x : ℕ) (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 70) : x = 8 :=
by
  sorry

end age_of_youngest_child_l223_223986


namespace eliminate_x_system_eq_l223_223688

theorem eliminate_x_system_eq : ∀ (x y : ℝ), (2 * x - 3 * y = 11) ∧ (2 * x + 5 * y = -5) → (-8 * y = 16) :=
by 
  intros x y h,
  cases h with h1 h2,
  calc
  (2 * x + 5 * y) - (2 * x - 3 * y) = -5 - 11 : by rw [h2, h1]
  ... 8 * y = -16 : by ring
  ... -8 * y = 16 : by exact eq.symm (eq_neg_of_add_eq_zero_right (by ring)),
  exact sorry

end eliminate_x_system_eq_l223_223688


namespace simplify_exponent_l223_223171

theorem simplify_exponent (y : ℝ) : (3 * y^4)^5 = 243 * y^20 :=
by
  sorry

end simplify_exponent_l223_223171


namespace calc_value_l223_223702

theorem calc_value : (9⁻¹ - 5⁻¹)⁻¹ = - (45 / 4) := by
  sorry

end calc_value_l223_223702


namespace split_piles_equiv_single_stone_heaps_l223_223093

theorem split_piles_equiv_single_stone_heaps (n : ℕ) (heaps : List ℕ) (h_initial : ∀ h ∈ heaps, h = 1)
  (h_size : heaps.length = n) :
  ∃ final_heap, (∀ x y ∈ heaps, x + y ≤ 2 * max x y) ∧ (List.sum heaps = (heaps.length) * 1) := by
  sorry

end split_piles_equiv_single_stone_heaps_l223_223093


namespace limit_proof_l223_223014

noncomputable def f (x : ℝ) := Real.log (2 - x) + x^3

theorem limit_proof : 
    tendsto (λ Δx : ℝ, (f (1 + Δx) - f 1) / (3 * Δx)) (nhds 0) (nhds (2 / 3)) :=
sorry

end limit_proof_l223_223014


namespace find_k_l223_223672

open Real EuclideanSpace

def vectors_on_line (a b : ℝ^3) (t : ℝ) := a + t • (b - a)

theorem find_k (a b : ℝ^3) (h : a ≠ b) :
  ∃ k : ℝ, k = 1/3 ∧ vectors_on_line a b (2/3) = k • a + (2/3) • b :=
by
  use 1/3
  sorry

end find_k_l223_223672


namespace perm_mississippi_l223_223742

theorem perm_mississippi : 
  let n := 11
  let n_S := 4
  let n_I := 4
  let n_P := 2
  let n_M := 1
  ∏ (mississippi_perm := Nat.factorial n / (Nat.factorial n_S * Nat.factorial n_I * Nat.factorial n_P * Nat.factorial n_M)) :=
  mississippi_perm = 34650 :=
by sorry

end perm_mississippi_l223_223742


namespace sum_of_log_sequence_l223_223884

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, a n > 0) :=
  ∀ n, a n = a 3 * q^(n-3)

noncomputable def log_sequence_sum (a : ℕ → ℝ) : ℕ → ℝ := λ n, (λ k, log 2 (a k)) n

theorem sum_of_log_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a 2 (λ n, by linarith))
  (h_a3 : a 3 = 4)
  (h_a7 : a 7 = 64) :
  (Finset.range 9).sum (log_sequence_sum a) = 36 :=
sorry

end sum_of_log_sequence_l223_223884


namespace odd_function_a_value_correct_l223_223399

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

noncomputable def a_value : ℝ := 2

theorem odd_function_a_value_correct :
  is_odd_function (λ x, (x + a_value - 2) * (2 * x^2 + a_value - 1)) :=
by
  sorry

end odd_function_a_value_correct_l223_223399


namespace problem_statement_l223_223828

-- Define the function f(x)
def f (x m : ℝ) := Real.sqrt (|x + 1| + |x - 3| - m)

-- Define the conditions and proof statements
theorem problem_statement (m : ℝ) (a b : ℝ) (h1 : f x m ≥ 0) (h2 : 0 < a) (h3 : 0 < b) 
(h4 : ∀ x : ℝ, |x + 1| + |x - 3| ≥ 4) (h5 : m ≤ 4) (h6 : (2 / (3 * a + b)) + (1 / (a + 2 * b)) = 4) 
: m ≤ 4 ∧ (7 * a + 4 * b) ≥ 9 / 4 :=
sorry

end problem_statement_l223_223828


namespace div_by_133_l223_223529

theorem div_by_133 (n : ℕ) : 133 ∣ 11^(n+2) + 12^(2*n+1) :=
by sorry

end div_by_133_l223_223529


namespace ratio_of_men_to_women_l223_223141

-- Definitions according to the conditions
variables (M W : ℕ)
variable h1 : W = M + 4
variable h2 : M + W = 14

-- The proof statement
theorem ratio_of_men_to_women (h1 : W = M + 4) (h2 : M + W = 14) : M / W = 5 / 9 :=
by sorry

end ratio_of_men_to_women_l223_223141


namespace range_of_a_l223_223511

open Real

theorem range_of_a (a : ℝ) (H : ∀ b : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → abs (x^2 + a * x + b) ≥ 1)) : a ≥ 1 ∨ a ≤ -3 :=
sorry

end range_of_a_l223_223511


namespace part1_part2_l223_223183

-- Definitions
def p (t : ℝ) := ∀ x : ℝ, x^2 + 2 * x + 2 * t - 4 ≠ 0
def q (t : ℝ) := (4 - t > 0) ∧ (t - 2 > 0)

-- Theorem statements
theorem part1 (t : ℝ) (hp : p t) : t > 5 / 2 := sorry

theorem part2 (t : ℝ) (h : p t ∨ q t) (h_and : ¬ (p t ∧ q t)) : (2 < t ∧ t ≤ 5 / 2) ∨ (t ≥ 3) := sorry

end part1_part2_l223_223183


namespace lighter_boxes_weight_l223_223029

noncomputable def weight_lighter_boxes (W L H : ℕ) : Prop :=
  L + H = 30 ∧
  (L * W + H * 20) / 30 = 18 ∧
  (H - 15) = 0 ∧
  (15 + L - H = 15 ∧ 15 * 16 = 15 * W)

theorem lighter_boxes_weight :
  ∃ W, ∀ L H, weight_lighter_boxes W L H → W = 16 :=
by sorry

end lighter_boxes_weight_l223_223029


namespace huanhuan_initial_coins_l223_223430

theorem huanhuan_initial_coins :
  ∃ (H L n : ℕ), H = 7 * L ∧ (H + n = 6 * (L + n)) ∧ (H + 2 * n = 5 * (L + 2 * n)) ∧ H = 70 :=
by
  sorry

end huanhuan_initial_coins_l223_223430


namespace smallest_angle_in_triangle_l223_223460

open Real

theorem smallest_angle_in_triangle
  (a b c : ℝ)
  (h : a = (b + c) / 3)
  (triangle_inequality_1 : a + b > c)
  (triangle_inequality_2 : a + c > b)
  (triangle_inequality_3 : b + c > a) :
  ∃ A B C α β γ : ℝ, -- A, B, C are the angles opposite to sides a, b, c respectively
  0 < α ∧ α < β ∧ α < γ :=
sorry

end smallest_angle_in_triangle_l223_223460


namespace tan_cot_square_identity_l223_223804

theorem tan_cot_square_identity (α : ℝ) (h : sin α + cos α = 1 / 2) : tan α ^ 2 + cot α ^ 2 = 46 / 9 :=
by
  sorry

end tan_cot_square_identity_l223_223804


namespace heaps_combination_preserve_similarity_split_stones_into_similar_heaps_l223_223117

def initial_heaps (n : ℕ) : list ℕ := list.repeat 1 n

def combine_heaps (heaps : list ℕ) : list ℕ :=
  if heaps.length ≥ 2 then
    let min1 := list.minimum heaps,
        heaps' := list.erase heaps min1,
        min2 := list.minimum heaps'
    in
    if min1 ≤ min2 then
      (min1 + min2) :: list.erase heaps' min2
    else
      heaps
  else
    heaps

theorem heaps_combination_preserve_similarity (heaps : list ℕ) (h : ∀ x ∈ heaps, x = 1) :
  ∀ combined_heaps, combined_heaps = combine_heaps heaps →
  ∀ x y ∈ combined_heaps, x ≤ y → x + y ≤ 2 * y :=
sorry

theorem split_stones_into_similar_heaps (n : ℕ) :
  ∃ combined_heaps : list ℕ, ∀ x y ∈ combined_heaps, x ≤ y → x + y ≤ 2 * y :=
sorry

end heaps_combination_preserve_similarity_split_stones_into_similar_heaps_l223_223117


namespace part1_part2_l223_223822

def f (x : ℝ) : ℝ := |x - 1|
def g (x : ℝ) (m : ℝ) : ℝ := -|x + 3| + m
def h (x : ℝ) : ℝ := |x - 1| + |x + 3|

theorem part1 (x : ℝ) : f x + x^2 - 1 > 0 ↔ x > 1 ∨ x < 0 :=
by
  sorry

theorem part2 (m : ℝ) : (∃ x : ℝ, f x < g x m) ↔ m > 4 :=
by
  sorry

end part1_part2_l223_223822


namespace bread_loaves_l223_223264

theorem bread_loaves (loaf_cost : ℝ) (pb_cost : ℝ) (total_money : ℝ) (leftover_money : ℝ) : ℝ :=
  let spent_money := total_money - leftover_money
  let remaining_money := spent_money - pb_cost
  remaining_money / loaf_cost

example : bread_loaves 2.25 2 14 5.25 = 3 := by
  sorry

end bread_loaves_l223_223264


namespace sum_sqrt_inequality_l223_223940

theorem sum_sqrt_inequality (n : ℕ) (h_pos : 0 < n) :
  (2 * n + 1) / 3 * Real.sqrt n ≤ ∑ i in Finset.range n, Real.sqrt (i + 1) ∧ 
  ∑ i in Finset.range n, Real.sqrt (i + 1) ≤ (4 * n + 3) / 6 * Real.sqrt n - 1 / 6 :=
sorry

end sum_sqrt_inequality_l223_223940


namespace coefficient_of_x3_l223_223761

theorem coefficient_of_x3 :
  (let expr := 5 * (x^2 - 2 * x^3 + x) + 2 * (x + 3 * x^3 - 2 * x^2 + 2 * x^5 + 2 * x^3) - 7 * (1 + 2 * x - 5 * x^3 - x^2)
  in coeff x^3 expr) = 35 := sorry

end coefficient_of_x3_l223_223761


namespace convert_base_1729_to_6_l223_223720

-- Define the base 10 number and the base to convert to
def n : ℕ := 1729
def b : ℕ := 6
def expected_result : ℕ := 120001

-- State the problem: Prove that the base 6 representation of 1729 is 120001
theorem convert_base_1729_to_6 : nat_to_base n b = expected_result := 
by sorry

end convert_base_1729_to_6_l223_223720


namespace find_denominator_l223_223441

-- Define the conditions given in the problem
variables (p q : ℚ)
variable (denominator : ℚ)

-- Assuming the conditions
variables (h1 : p / q = 4 / 5)
variables (h2 : 11 / 7 + (2 * q - p) / denominator = 2)

-- State the theorem we want to prove
theorem find_denominator : denominator = 14 :=
by
  -- The proof will be constructed later
  sorry

end find_denominator_l223_223441


namespace fraction_spent_on_delivery_l223_223137

theorem fraction_spent_on_delivery 
    (C : ℝ) (f_s : ℝ) (O : ℝ) (H1 : C = 4000) 
    (H2 : f_s = 2/5) (H3 : O = 1800) :
    (600 / (C - f_s * C)) = 1/4 :=
by
  rw [H1, H2, H3]
  norm_num
  sorry

end fraction_spent_on_delivery_l223_223137


namespace find_total_income_l223_223265

theorem find_total_income (I : ℝ) (h1 : 0 < I) (h2 : 500 = I - 0.46 * I) : I ≈ 925.93 := by
  have : I = 500 / 0.54 := by
    linarith
  norm_num at this
  exact this

end find_total_income_l223_223265


namespace element_in_enough_sets_l223_223051

open Finset

theorem element_in_enough_sets (S : Finset (Finset ℕ)) (n k : ℕ)
  (hS : S.card = n)
  (hn2 : 2 ≤ n)
  (h_min_card : (S.image card).min' sorry = k)
  (hk2 : 2 ≤ k)
  (h_union : ∀ {A B : Finset ℕ}, A ∈ S → B ∈ S → (A ∪ B) ∈ S) :
  ∃ x : ℕ, x ∈ (S.fold (λ acc s, acc ∪ s) ∅ id) ∧ (card (filter (λ t, x ∈ t) S) ≥ n / k) :=
sorry

end element_in_enough_sets_l223_223051


namespace area_of_central_grey_octagon_l223_223933

noncomputable def central_grey_octagon_area (side_length : ℝ) (XY : ℝ) : ℝ :=
  if h : side_length = 8 ∧ XY = 2 then
    10
  else
    sorry

theorem area_of_central_grey_octagon :
  ∀ (side_length XY : ℝ) (Z_parallel YZ_parallel: Prop),
    (side_length = 8) →
    (XY = 2) →
    (Z_parallel) →
    (YZ_parallel) →
    central_grey_octagon_area side_length XY = 10 :=
by
  intros side_length XY Z_parallel YZ_parallel h_side_length h_XY h_Z h_YZ
  simp only [central_grey_octagon_area, h_side_length, h_XY]
  exact if_pos ⟨h_side_length, h_XY⟩


end area_of_central_grey_octagon_l223_223933


namespace gasoline_added_l223_223284

variable (tank_capacity : ℝ := 42)
variable (initial_fill_fraction : ℝ := 3/4)
variable (final_fill_fraction : ℝ := 9/10)

theorem gasoline_added :
  let initial_amount := tank_capacity * initial_fill_fraction
  let final_amount := tank_capacity * final_fill_fraction
  final_amount - initial_amount = 6.3 :=
by
  sorry

end gasoline_added_l223_223284


namespace quadratic_equation_coefficients_l223_223024

-- The main statement combining both parts proof
theorem quadratic_equation_coefficients (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = 5 * x ^ 2 - x - 3 - (x ^ 2 - 3 + x)) →
  (f 0 = c) →
  (f 0 = 4 * (0:ℝ) ^ 2 - 2 * (0:ℝ) - 3 + 3) →
  (second_coeff : ℝ) → -2 = b :=
begin
  intros,
  rw H,
  sorry
end

end quadratic_equation_coefficients_l223_223024


namespace circle_radius_unique_chord_l223_223379

theorem circle_radius_unique_chord (r : ℝ) (h : r > 0) :
  (∃ C : ℝ → ℝ → Prop, (C = λ x y, x^2 + y^2 = r^2) ∧
                        ∃ P : ℝ × ℝ, P = (1, 1) ∧
                        ∃ l : ℝ, l = 2 ∧
                        (∃! chord : (ℝ × ℝ) × (ℝ × ℝ),
                          (chord.1 = (x1, y1) ∧ chord.2 = (x2, y2) ∧
                           (x1 - x2)^2 + (y1 - y2)^2 = l^2 ∧
                           C P.1 P.2))) →
  r = 1 :=
by
  sorry

end circle_radius_unique_chord_l223_223379


namespace sqrt_mul_sqrt_eq_sqrt_mul_l223_223634

theorem sqrt_mul_sqrt_eq_sqrt_mul (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 
  Real.sqrt (a * b) = Real.sqrt a * Real.sqrt b := 
by
  sorry

example : Real.sqrt 5 * Real.sqrt 2 = Real.sqrt 10 := by
  apply sqrt_mul_sqrt_eq_sqrt_mul
  show 0 ≤ 5 from by norm_num
  show 0 ≤ 2 from by norm_num

end sqrt_mul_sqrt_eq_sqrt_mul_l223_223634


namespace find_n_l223_223590

theorem find_n (n : ℝ) : (∃ (f : ℝ → ℝ), f = λ x, n * x + (n^2 - 7) ∧ f 0 = 2 ∧ ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂) → n = -3 :=
by
  sorry

end find_n_l223_223590


namespace simplify_expression_l223_223174

theorem simplify_expression (y : ℝ) : (3 * y^4)^5 = 243 * y^20 :=
sorry

end simplify_expression_l223_223174


namespace rationalize_denominator_l223_223532

theorem rationalize_denominator :
  ∃ (A B C D : ℤ), 
    (A = 25 ∧ B = 2 ∧ C = 20 ∧ D = 17) ∧ 
    0 < D ∧ 
    ¬ ∃ p : ℤ, prime p ∧ p^2 ∣ B ∧ 
    (∑ x in [A, B, C, D], x) = 64 ∧
    (\ ∀ x y : ℚ, x = \(\frac{\sqrt{50}}{\sqrt{25} - 2\sqrt{2}}\) → y = \(\frac{25\sqrt{2} + 20}{17}\) → x = y) :=
by {
  sorry
}

end rationalize_denominator_l223_223532


namespace car_catch_up_distance_l223_223706

-- Define the problem: Prove that Car A catches Car B at the specified location, given the conditions

theorem car_catch_up_distance
  (distance : ℝ := 300)
  (v_A v_B t_A : ℝ)
  (start_A_late : ℝ := 1) -- Car A leaves 1 hour later
  (arrive_A_early : ℝ := 1) -- Car A arrives 1 hour earlier
  (t : ℝ := 2) -- Time when Car A catches up with Car B
  (dist_eq_A : distance = v_A * t_A)
  (dist_eq_B : distance = v_B * (t_A + 2))
  (catch_up_time_eq : v_A * t = v_B * (t + 1)):
  (distance - v_A * t) = 150 := sorry

end car_catch_up_distance_l223_223706


namespace total_euros_is_correct_l223_223013

def total_coins_in_usd :=
  (9 * 0.01) + (4 * 0.05) + (3 * 0.10) + (7 * 0.25) + (5 * 0.50) + (2 * 1.00) + (1 * 2.00)

def conversion_rate : ℝ := 0.85

def total_amount_in_euros : ℝ := total_coins_in_usd * conversion_rate

theorem total_euros_is_correct :
  total_amount_in_euros = 7.51 := by
  sorry

end total_euros_is_correct_l223_223013


namespace always_two_real_roots_root_less_than_one_l223_223367

-- Define the quadratic equation parameters
def quadratic_eq (m : ℝ) : ℝ → ℝ := λ x, x^2 - m * x + (2 * m - 4)

-- Prove that the equation always has two real roots
theorem always_two_real_roots (m : ℝ) : ∃ r1 r2 : ℝ, quadratic_eq m r1 = 0 ∧ quadratic_eq m r2 = 0 := by
  sorry

-- Prove the range of m if one root is less than 1
theorem root_less_than_one (m : ℝ) (r1 r2 : ℝ) (h_r1_lt_1 : r1 < 1) (h_eq_r1 : quadratic_eq m r1 = 0) (h_eq_r2 : quadratic_eq m r2 = 0) : m < 3 := by
  sorry

end always_two_real_roots_root_less_than_one_l223_223367


namespace evaluate_expression_l223_223325

theorem evaluate_expression : ( (real.pi - 1)^0 + (1 / 2)^(-1) + abs (5 - real.sqrt 27) - 2 * real.sqrt 3 ) = 8 - 5 * real.sqrt 3 :=
by
  sorry

end evaluate_expression_l223_223325


namespace souvenir_shop_problems_l223_223240

noncomputable theory

-- Define the costs of type A and B, and the linear equations.
variable (x y : ℝ)
variable (a b : ℕ)

-- Conditions from the problem
def condition1 := 7 * x + 4 * y = 760
def condition2 := 5 * x + 8 * y = 800
def total_pieces := a + b = 100
def cost_lower_bound := 80 * a + 50 * b ≥ 7000
def cost_upper_bound := 80 * a + 50 * b ≤ 7200
def cost_constraints := cost_lower_bound ∧ cost_upper_bound

-- Profits
def profit := 30 * a + 20 * b

-- The hypotheses and problem statements
theorem souvenir_shop_problems :
  (∃ (x y : ℝ), condition1 x y ∧ condition2 x y ∧ x = 80 ∧ y = 50) ∧
  (∃ (plans : Finset ℕ), plans = {67, 68, 69, 70, 71, 72, 73}) ∧
  (∃ (max_profit : ℝ), max_profit = 2730 ∧
                        ∀ a b, total_pieces a b → cost_constraints a b →
                               profit a b = max_profit ↔ a = 73 ∧ b = 27) :=
by sorry

end souvenir_shop_problems_l223_223240


namespace machine_probabilities_at_least_one_first_class_component_l223_223236

theorem machine_probabilities : 
  (∃ (PA PB PC : ℝ), 
  PA * (1 - PB) = 1/4 ∧ 
  PB * (1 - PC) = 1/12 ∧ 
  PA * PC = 2/9 ∧ 
  PA = 1/3 ∧ 
  PB = 1/4 ∧ 
  PC = 2/3) 
:=
sorry

theorem at_least_one_first_class_component : 
  ∃ (PA PB PC : ℝ), 
  PA * (1 - PB) = 1/4 ∧ 
  PB * (1 - PC) = 1/12 ∧ 
  PA * PC = 2/9 ∧ 
  PA = 1/3 ∧ 
  PB = 1/4 ∧ 
  PC = 2/3 ∧ 
  1 - (1 - PA) * (1 - PB) * (1 - PC) = 5/6
:=
sorry

end machine_probabilities_at_least_one_first_class_component_l223_223236


namespace total_pages_is_905_l223_223567

def history_pages : ℕ := 160
def geography_pages : ℕ := history_pages + 70
def math_pages : ℕ := (history_pages + geography_pages) / 2
def science_pages : ℕ := 2 * history_pages
def total_pages : ℕ := history_pages + geography_pages + math_pages + science_pages

theorem total_pages_is_905 : total_pages = 905 := by
  sorry

end total_pages_is_905_l223_223567


namespace colored_pencils_more_than_erasers_l223_223152

def colored_pencils_initial := 67
def erasers_initial := 38

def colored_pencils_final := 50
def erasers_final := 28

theorem colored_pencils_more_than_erasers :
  colored_pencils_final - erasers_final = 22 := by
  sorry

end colored_pencils_more_than_erasers_l223_223152


namespace find_principal_l223_223642

theorem find_principal (P : ℝ) (r : ℝ) (t : ℝ) (CI SI : ℝ) 
  (h_r : r = 0.20) 
  (h_t : t = 2) 
  (h_diff : CI - SI = 144) 
  (h_CI : CI = P * (1 + r)^t - P) 
  (h_SI : SI = P * r * t) : 
  P = 3600 :=
by
  sorry

end find_principal_l223_223642


namespace ming_got_first_place_l223_223347

-- Define the statements made by each person
def Fang_statement (Fang_first : Prop) : Prop := Fang_first
def Ming_statement (Hong_first : Prop) : Prop := ¬ Hong_first
def Ma_statement (Ming_first : Prop) : Prop := ¬ Ming_first
def Hong_statement (Hong_first : Prop) : Prop := Hong_first

-- The main theorem with the conditions given
theorem ming_got_first_place (Fang_first Ming_first Ma_first Hong_first : Prop) :
  -- Only one person can get first place
  (Fang_first ∨ Ming_first ∨ Ma_first ∨ Hong_first) ∧ 
  ¬(Fang_first ∧ Ming_first) ∧ ¬(Fang_first ∧ Ma_first) ∧ ¬(Fang_first ∧ Hong_first) ∧ 
  ¬(Ming_first ∧ Ma_first) ∧ ¬(Ming_first ∧ Hong_first) ∧ ¬(Ma_first ∧ Hong_first) ∧
  -- Only one person is telling the truth
  (∀ (P₁ P₂ : Prop → Prop), 
    ((P₁ Fang_first ∨ P₁ Ming_first ∨ P₁ Ma_first ∨ P₁ Hong_first) ∧ 
     (¬ (P₁ (P₁ Fang_first) ∧ P₁ (P₁ Ming_first)) ∧ 
      ¬ (P₁ (P₁ Fang_first) ∧ P₁ (P₁ Ma_first)) ∧ 
      ¬ (P₁ (P₁ Fang_first) ∧ P₁ (P₁ Hong_first)) ∧ 
      ¬ (P₁ (P₁ Ming_first) ∧ P₁ (P₁ Ma_first)) ∧ 
      ¬ (P₁ (P₁ Ming_first) ∧ P₁ (P₁ Hong_first)) ∧ 
      ¬ (P₁ (P₁ Ma_first) ∧ P₁ (P₁ Hong_first)))) →
  Fang_statement Fang_first →
  Ming_statement Hong_first →
  Ma_statement Ming_first →
  Hong_statement Hong_first →
  Ming_first := sorry

end ming_got_first_place_l223_223347


namespace find_a_l223_223838

theorem find_a (a : ℤ) : 
  let A := {-1, 0, 1}
  let B := {0, a, 2}
  A ∩ B = {-1, 0} -> a = -1 :=
by
  intro h
  sorry

end find_a_l223_223838


namespace arithmetic_sequence_a5_l223_223468

variable {α : Type*} [LinearOrderedField α]

def is_arithmetic_sequence (a : ℕ → α) : Prop :=
∀ n m : ℕ, ∃ d : α, a (n + 1) = a n + d ∧ a (m + 1) = a m + d

noncomputable def a_n : ℕ → ℝ
                             -- The type can be adjusted if necessary.

theorem arithmetic_sequence_a5 (a : ℕ → ℝ) (h : is_arithmetic_sequence a) (h2 : a 2 + a 8 = 10) : a 5 = 5 :=
by
  sorry

end arithmetic_sequence_a5_l223_223468


namespace extreme_points_in_interval_l223_223863

noncomputable def f (a x : ℝ) : ℝ :=
  a * (x - 2) * Real.exp x + Real.log x + 1 / x

noncomputable def f' (a x : ℝ) : ℝ :=
  a * (x - 1) * Real.exp x + 1 / x - 1 / (x ^ 2)

theorem extreme_points_in_interval
  (a : ℝ) :
  (∃ x1 x2 ∈ Ioo 0 2, f' a x1 = 0 ∧ f' a x2 = 0 ∧ x1 ≠ x2) ↔
  a ∈ Set.Ioo (neg_inf : ℝ) (-1 / Real.exp 1) ∪ Set.Ioo (-1 / Real.exp 1) (-1 / (4 * Real.exp 2)) :=
sorry

end extreme_points_in_interval_l223_223863


namespace travel_time_from_A_to_B_and_back_l223_223295

theorem travel_time_from_A_to_B_and_back (d v : ℕ) (h₁ : d = 120) (h₂ : v = 60) : 
  let t := d / v in 
  t + t = 4 := 
by 
  /-
    This is to state that for given distance d = 120 km and 
    speed v = 60 km/h, the travel time both ways summed is 4 hours. 
  -/
  sorry

end travel_time_from_A_to_B_and_back_l223_223295


namespace find_angle_C_l223_223481

variable {A B C : ℝ} -- Angles of triangle ABC
variable {a b c : ℝ} -- Sides opposite the respective angles

theorem find_angle_C
  (h1 : 2 * c * Real.cos B = 2 * a + b) : 
  C = 120 :=
  sorry

end find_angle_C_l223_223481


namespace min_g_value_l223_223781

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - 2 * x + 1

def M (a : ℝ) : ℝ := 
  if 0.5 ≤ a then max (f a 1) (f a 3) else f a 1

def N (a : ℝ) : ℝ := 
  min (f a 1) (f a 3)

noncomputable def g (a : ℝ) : ℝ := M a - N a

theorem min_g_value : 
  ∀ a : ℝ, 
    (1/3 ≤ a ∧ a ≤ 1) → 
    g(a) = 1/2 :=
begin
  sorry
end

end min_g_value_l223_223781


namespace necessary_but_not_sufficient_condition_not_sufficient_condition_l223_223396

variables (a b : ℝ)

theorem necessary_but_not_sufficient_condition (h : a > b ∧ b > 0) : 
  a > b → (1/a < 1/b) :=
begin
  sorry
end

theorem not_sufficient_condition (h : 1/a < 1/b) : 
  (a > b ∧ b > 0 ∨ 0 > a ∧ a > b) :=
begin
  sorry
end

end necessary_but_not_sufficient_condition_not_sufficient_condition_l223_223396


namespace students_count_l223_223960

theorem students_count (n : ℕ) 
    (initial_avg correct_avg wrong_mark correct_mark : ℕ)
    (h1 : initial_avg = 100)
    (h2 : wrong_mark = 70)
    (h3 : correct_mark = 10)
    (h4 : correct_avg = 98) 
    (h5 : initial_avg * n - (wrong_mark - correct_mark) = correct_avg * n) : 
  n = 30 := 
by {
  rw [h1, h2, h3, h4] at h5,
  linarith
}

end students_count_l223_223960


namespace intersects_x_axis_at_one_point_l223_223588

theorem intersects_x_axis_at_one_point (a : ℝ) :
  (∃ x, ax^2 + (a-3)*x + 1 = 0) ∧ (∀ x₁ x₂, ax^2 + (a-3)*x + 1 = 0 → x₁ = x₂) ↔ (a = 0 ∨ a = 1 ∨ a = 9) := by
  sorry

end intersects_x_axis_at_one_point_l223_223588


namespace vector_dot_product_sum_l223_223841

variable {A B C : Type} [euclidean_space A] [euclidean_space B] [euclidean_space C]
variable (vectorAB vectorBC vectorCA : A → B → C → ℝ)
variable (AB BC CA: ℝ)

theorem vector_dot_product_sum :
  |vectorAB A B| = 3 →
  |vectorBC B C| = 4 →
  |vectorCA C A| = 5 →
  vectorAB A B • vectorBC B C + vectorBC B C • vectorCA C A + vectorCA C A • vectorAB A B = -25 :=
by
  sorry

end vector_dot_product_sum_l223_223841


namespace line_through_Q_bisects_triangle_area_l223_223615

def P := (0, 10)
def Q := (3, 0)
def R := (10, 0)
def S := ((0 + 10) / 2, (10 + 0) / 2)  -- Midpoint of PR

-- The slope of the line through Q and S
def slope := (5 - 0) / (5 - 3)

-- Equation of line through Q with the above slope: y = slope * x + b
-- y - 0 = slope * (x - 3)
-- => y = slope * x - slope * 3
-- => y = (5/2) * x - (5/2) * 3
-- y = (5/2) x - 15/2

-- Sum of slope and y-intercept
def sum_slope_y_intercept := slope + (-15 / 2)

theorem line_through_Q_bisects_triangle_area :
  sum_slope_y_intercept = -5 :=
by
  -- We assert that sum_slope_y_intercept = -5
  sorry

end line_through_Q_bisects_triangle_area_l223_223615


namespace find_first_term_l223_223216

noncomputable def firstTermOfGeometricSeries (a r : ℝ) : Prop :=
  (a / (1 - r) = 30) ∧ (a^2 / (1 - r^2) = 120)

theorem find_first_term :
  ∃ a r : ℝ, firstTermOfGeometricSeries a r ∧ a = 120 / 17 :=
by
  sorry

end find_first_term_l223_223216


namespace valid_prime_p_l223_223279

def isValidPrime (p : ℕ) (table : Fin 2017 → Fin 2017 → ℕ) : Prop :=
  (∀ i : Fin 2017, (∑ j : Fin 2017, table i j * 10^(2016 - j.val)) % p = 0) ∧ 
  (∀ j : Fin 2017, (∑ i : Fin 2017, table i j * 10^(2016 - i.val)) % p = 0) ∧ 
  (Exists i : Fin 2017, (∑ j : Fin 2017, table i j * 10^(2016 - j.val)) % p ≠ 0 ∨ 
   Exists j : Fin 2017, (∑ i : Fin 2017, table i j * 10^(2016 - i.val)) % p ≠ 0)

theorem valid_prime_p (p : ℕ) (table : Fin 2017 → Fin 2017 → ℕ) (h_nonzero : ∀ i j, table i j ≠ 0) : 
  isValidPrime p table → p = 2 ∨ p = 5 := 
sorry

end valid_prime_p_l223_223279


namespace combined_savings_l223_223538

def salary_per_hour : ℝ := 10
def daily_hours : ℝ := 10
def weekly_days : ℝ := 5
def robby_saving_ratio : ℝ := 2 / 5
def jaylene_saving_ratio : ℝ := 3 / 5
def miranda_saving_ratio : ℝ := 1 / 2
def weeks : ℝ := 4

theorem combined_savings 
  (sph : ℝ := salary_per_hour)
  (dh : ℝ := daily_hours)
  (wd : ℝ := weekly_days)
  (rr : ℝ := robby_saving_ratio)
  (jr : ℝ := jaylene_saving_ratio)
  (mr : ℝ := miranda_saving_ratio)
  (wk : ℝ := weeks) :
  (rr * (wk * wd * (dh * sph)) + jr * (wk * wd * (dh * sph)) + mr * (wk * wd * (dh * sph))) = 3000 :=
by
  sorry

end combined_savings_l223_223538


namespace car_catch_up_distance_l223_223707

-- Define the problem: Prove that Car A catches Car B at the specified location, given the conditions

theorem car_catch_up_distance
  (distance : ℝ := 300)
  (v_A v_B t_A : ℝ)
  (start_A_late : ℝ := 1) -- Car A leaves 1 hour later
  (arrive_A_early : ℝ := 1) -- Car A arrives 1 hour earlier
  (t : ℝ := 2) -- Time when Car A catches up with Car B
  (dist_eq_A : distance = v_A * t_A)
  (dist_eq_B : distance = v_B * (t_A + 2))
  (catch_up_time_eq : v_A * t = v_B * (t + 1)):
  (distance - v_A * t) = 150 := sorry

end car_catch_up_distance_l223_223707


namespace total_pages_correct_l223_223559

def history_pages : ℕ := 160
def geography_pages : ℕ := history_pages + 70
def sum_history_geography_pages : ℕ := history_pages + geography_pages
def math_pages : ℕ := sum_history_geography_pages / 2
def science_pages : ℕ := 2 * history_pages
def total_pages : ℕ := history_pages + geography_pages + math_pages + science_pages

theorem total_pages_correct : total_pages = 905 := by
  -- The proof goes here.
  sorry

end total_pages_correct_l223_223559


namespace sum_of_roots_quadratic_eq_l223_223210

theorem sum_of_roots_quadratic_eq (x₁ x₂ : ℝ) (h : x₁^2 + 2 * x₁ - 4 = 0 ∧ x₂^2 + 2 * x₂ - 4 = 0) : 
  x₁ + x₂ = -2 :=
sorry

end sum_of_roots_quadratic_eq_l223_223210


namespace distance_between_4th_and_25th_red_light_l223_223132

structure LightConditions :=
  (spacing_in_inches : ℕ := 8)
  (pattern : List String := ["red", "red", "red", "green", "green"])

def number_of_lights_between (n₁ n₂ : ℕ) (pattern : List String) : ℕ :=
  let red_positions: List ℕ := 
    List.enumFromZero 
    |> List.filterMap (λ (i, v) => 
        if pattern.getD (i % pattern.length) "green" = "red" then 
          some i 
        else none)
  red_positions.getD n₂ 0 - red_positions.getD n₁ 0

def distance_in_feet (gaps : ℕ) (spacing_in_inches: ℕ) : ℝ :=
  gaps * spacing_in_inches / 12

theorem distance_between_4th_and_25th_red_light (cond : LightConditions) :
  distance_in_feet (number_of_lights_between 3 24 cond.pattern - 1) cond.spacing_in_inches = 38.67 :=
by
  sorry

end distance_between_4th_and_25th_red_light_l223_223132


namespace perm_mississippi_l223_223740

theorem perm_mississippi : 
  let n := 11
  let n_S := 4
  let n_I := 4
  let n_P := 2
  let n_M := 1
  ∏ (mississippi_perm := Nat.factorial n / (Nat.factorial n_S * Nat.factorial n_I * Nat.factorial n_P * Nat.factorial n_M)) :=
  mississippi_perm = 34650 :=
by sorry

end perm_mississippi_l223_223740


namespace problem_statement_l223_223825

def f (x a : ℝ) : ℝ := |Real.exp x - a|

theorem problem_statement (a : ℝ) (h : a > 0) :
  (¬ ((f 0 1 = 1) ∧ (f (Real.log 2) 1 = 1))) ∧
  ((f (Real.log (a^2 + a)) a = a^2) ∧ (f (Real.log (a - a^2)) a = a^2) → 0 < a ∧ a < 1) ∧
  ((a > 1) → (f (Real.log a) a = 0) ∧ (f (f (Real.log a) a) a = 0)) ∧
  ∀ x1 x2 : ℝ, (Real.exp x1 - a) * (Real.exp x2 - a) = -1 → x1 + x2 = 0 :=
sorry

end problem_statement_l223_223825


namespace karen_paint_time_l223_223483

/-- Given that it takes Shawn 18 hours to paint a house,
    and Shawn and Karen together can paint the house in 7.2 hours,
    prove that Karen can paint the house alone in 12 hours. -/

theorem karen_paint_time :
  ∃ k : ℝ, (1 / 18 + 1 / k = 1 / 7.2) ∧ k = 12 :=
begin
  sorry
end

end karen_paint_time_l223_223483


namespace coffee_machine_pays_off_l223_223489

def daily_savings(previous_cost: ℕ, current_cost: ℕ) : ℕ :=
  previous_cost - current_cost

def days_to_payoff(machine_cost: ℕ, savings_per_day: ℕ) : ℕ :=
  machine_cost / savings_per_day

theorem coffee_machine_pays_off :
  let machine_cost := 200 - 20
  let previous_daily_expense := 2 * 4
  let current_daily_expense := 3
  let savings := daily_savings previous_daily_expense current_daily_expense
  let payoff_days := days_to_payoff machine_cost savings
  payoff_days = 36 :=
by
  -- Calculation/Proof goes here
  sorry

end coffee_machine_pays_off_l223_223489


namespace ratio_of_six_to_eight_rounded_to_nearest_tenth_l223_223291

theorem ratio_of_six_to_eight_rounded_to_nearest_tenth : 
  let r : ℚ := 6 / 8 in 
  Real.round (r : ℝ) * 10 / 10 = 0.8 :=
by
  let r : ℚ := 6 / 8
  have h1 : (r : ℝ) = 0.75 := by norm_num
  have h2 : Real.round (0.75 * 10) / 10 = 0.8 := by norm_num
  exact h2

end ratio_of_six_to_eight_rounded_to_nearest_tenth_l223_223291


namespace complex_modulus_l223_223517

theorem complex_modulus (z : ℂ) (h : (z - 2 * complex.i) * (1 - complex.i) = -2) : complex.abs z = sqrt 10 := by
  sorry

end complex_modulus_l223_223517


namespace triangles_congruent_proof_l223_223144

-- Define the structures for points and triangles
structure Point :=
(x : ℝ)
(y : ℝ)

structure Triangle :=
(A B C : Point)

axiom point_D_on_AB (A B C D : Point) (t : Triangle) : Triangle.A = A ∧ Triangle.B = B ∧ Triangle.C = C → Sorry

axiom point_D1_on_A1B1 (A1 B1 C1 D1 : Point) (t1 : Triangle) : Triangle.A = A1 ∧ Triangle.B = B1 ∧ Triangle.C = C1 → Sorry

axiom congruent_triangles (t t1 : Triangle) (A B C A1 B1 C1 D D1 : Point) : 
  point_D_on_AB A B C D t ∧
  point_D1_on_A1B1 A1 B1 C1 D1 t1 ∧
  (t.A = A ∧ t.B = B ∧ t.C = C) ∧
  (t1.A = A1 ∧ t1.B = B1 ∧ t1.C = C1) ∧
  (t.B.dist(t.C) = t1.B.dist(t1.C)) ∧
  (t.A.dist(t.B) = t1.A.dist(t1.B)) ∧
  (t.C.dist(t.A) = t1.C.dist(t1.A)) → 
  Sorry

noncomputable def triangles_congruent (t t1 : Triangle) (A B C A1 B1 C1 D D1 : Point) : Prop :=
  point_D_on_AB A B C D t ∧
  point_D1_on_A1B1 A1 B1 C1 D1 t1 ∧
  congruent_triangles t t1 A B C A1 B1 C1 D D1 → 
  (t.A = t1.A ∧ t.B = t1.B ∧ t.C = t1.C)

-- Main theorem
theorem triangles_congruent_proof (t t1 : Triangle) (A B C A1 B1 C1 D D1 : Point) :
  triangles_congruent t t1 A B C A1 B1 C1 D D1 :=
by 
  sorry

end triangles_congruent_proof_l223_223144


namespace nancy_rome_books_l223_223523

theorem nancy_rome_books :
  ∃ R : ℕ,
    (let total_books := 46 in
     let history_books := 12 in
     let poetry_books := 4 in
     let western_books := 5 in
     let biographies := 6 in
     let top_books := history_books + poetry_books + R in
     let bottom_books := total_books - top_books in
     let mystery_books := (bottom_books - (western_books + biographies)) / 2 in
     bottom_books = (western_books + biographies) + mystery_books + mystery_books) ∧ R = 8 := 
begin
  existsi (8 : ℕ),
  split,
  { intros,
    let R := 8,
    let total_books := 46,
    let history_books := 12,
    let poetry_books := 4,
    let western_books := 5,
    let biographies := 6,
    let top_books := history_books + poetry_books + R,
    let bottom_books := total_books - top_books,
    let non_mystery_books := western_books + biographies,
    let mystery_books := (bottom_books - non_mystery_books) / 2,
    have bottom_books_eq : bottom_books = non_mystery_books + 2 * mystery_books,
    { ring_nf,
      exact bottom_books_eq }
  },
  simp
end

end nancy_rome_books_l223_223523


namespace circumcenter_lies_on_line_BD_l223_223918

noncomputable def circumcenter (A B C : Point) : Point := sorry  -- Definition/Calculation of Circumcenter

noncomputable def lies_on_line (P Q R : Point) : Prop := sorry  -- Definition for point lying on line

variables {α : Type*} [euclidean_geometry α] 
open_locale euclidean_geometry

-- Define the points and lines in the problem
variables (A B C E D F : Point)
variables (hACB : angle A C B = 90)
variables (hBAD : angle B A D = 90)
variables (hDEF : angle D E F = 90)
variables (hCE_bisector : bisects _ _ _ E A C)  -- CE is bisector of ∠ A C B
variables (hD_external_bisector : external_bisector A C B D)  -- D on external bisector

-- Define the circumcenter of triangle CEF
noncomputable def O := circumcenter C E F

-- The theorem stating that the circumcenter O lies on line BD
theorem circumcenter_lies_on_line_BD : lies_on_line O B D :=
  sorry -- The proof goes here

end circumcenter_lies_on_line_BD_l223_223918


namespace thomas_score_l223_223138

theorem thomas_score (n : ℕ) (avg19 avg20 : ℚ)
  (h₁ : n = 20)
  (h₂ : avg19 = 78)
  (h₃ : avg20 = 80) :
  (20 * avg20 - (19 * avg19) = 118 ) :=
begin
  sorry
end

end thomas_score_l223_223138


namespace amount_after_two_years_eq_l223_223267

-- Definitions based on conditions
def rate_of_increase : ℝ := 1 / 8
def present_value : ℝ := 76800
def years : ℕ := 2

-- Target amount calculation
def amount_after_years (n : ℕ) (P : ℝ) (r : ℝ) : ℝ :=
  P * (1 + r)^n

-- Prove that the amount after two years is Rs. 97200
theorem amount_after_two_years_eq :
  amount_after_years years present_value rate_of_increase = 97200 :=
by
  sorry

end amount_after_two_years_eq_l223_223267


namespace avg_value_of_set_l223_223953

variable (T : Finset ℕ)
variable [DecidableEq ℕ]
variable {n : ℕ}
variables {b1 bm : ℕ}

noncomputable def avg (s : Finset ℕ) : ℝ :=
  (Finset.sum s id : ℝ) / (s.card : ℝ)

theorem avg_value_of_set 
  (h1 : bm ∈ T) 
  (h2 : b1 ∈ T)
  (h3 : b1 < bm)
  (h4 : ∀ b ∈ T, b1 ≤ b ∧ b ≤ bm)
  (h5 : avg (T.erase bm) = 45)
  (h6 : avg ((T.erase bm).erase b1) = 50)
  (h7 : avg ((T.erase b1).insert bm) = 55)
  (h8 : bm = b1 + 90) :
  avg T = 55.5 :=
sorry

end avg_value_of_set_l223_223953


namespace combined_savings_after_four_weeks_l223_223536

-- Definitions based on problem conditions
def hourly_wage : ℕ := 10
def daily_hours : ℕ := 10
def days_per_week : ℕ := 5
def weeks : ℕ := 4

def robby_saving_ratio : ℚ := 2/5
def jaylene_saving_ratio : ℚ := 3/5
def miranda_saving_ratio : ℚ := 1/2

-- Definitions derived from the conditions
def daily_earnings : ℕ := hourly_wage * daily_hours
def total_working_days : ℕ := days_per_week * weeks
def monthly_earnings : ℕ := daily_earnings * total_working_days

def robby_savings : ℚ := robby_saving_ratio * monthly_earnings
def jaylene_savings : ℚ := jaylene_saving_ratio * monthly_earnings
def miranda_savings : ℚ := miranda_saving_ratio * monthly_earnings

def total_savings : ℚ := robby_savings + jaylene_savings + miranda_savings

-- The main theorem to prove
theorem combined_savings_after_four_weeks :
  total_savings = 3000 := by sorry

end combined_savings_after_four_weeks_l223_223536


namespace quadrant_of_alpha_l223_223395

theorem quadrant_of_alpha (α : ℝ) (h1 : Real.tan α > 0) (h2 : Real.sin α + Real.cos α > 0) : 1 :=
by
  sorry

end quadrant_of_alpha_l223_223395


namespace split_piles_equiv_single_stone_heaps_l223_223098

theorem split_piles_equiv_single_stone_heaps (n : ℕ) (heaps : List ℕ) (h_initial : ∀ h ∈ heaps, h = 1)
  (h_size : heaps.length = n) :
  ∃ final_heap, (∀ x y ∈ heaps, x + y ≤ 2 * max x y) ∧ (List.sum heaps = (heaps.length) * 1) := by
  sorry

end split_piles_equiv_single_stone_heaps_l223_223098


namespace sugar_inventory_l223_223484

theorem sugar_inventory :
  ∀ (initial : ℕ) (day2_use : ℕ) (day2_borrow : ℕ) (day3_buy : ℕ) (day4_buy : ℕ) (day5_use : ℕ) (day5_return : ℕ),
  initial = 65 →
  day2_use = 18 →
  day2_borrow = 5 →
  day3_buy = 30 →
  day4_buy = 20 →
  day5_use = 10 →
  day5_return = 3 →
  initial - day2_use - day2_borrow + day3_buy + day4_buy - day5_use + day5_return = 85 :=
by
  intros initial day2_use day2_borrow day3_buy day4_buy day5_use day5_return
  intro h_initial
  intro h_day2_use
  intro h_day2_borrow
  intro h_day3_buy
  intro h_day4_buy
  intro h_day5_use
  intro h_day5_return
  subst h_initial
  subst h_day2_use
  subst h_day2_borrow
  subst h_day3_buy
  subst h_day4_buy
  subst h_day5_use
  subst h_day5_return
  sorry

end sugar_inventory_l223_223484


namespace find_angle_KAM_l223_223525

noncomputable def square_side : ℝ := sorry

structure Square (A B C D : Point ℝ) : Prop :=
  (is_square : is_square A B C D)
  (AB_eq_side : distance A B = square_side)
  (BC_eq_side : distance B C = square_side)
  (CD_eq_side : distance C D = square_side)
  (DA_eq_side : distance D A = square_side)

variables {A B C D K M : Point ℝ}

axiom given_conditions 
  (h1 : Square A B C D)
  (h2 : K ∈ LineSegment B C)
  (h3 : M ∈ LineSegment C D)
  (h4 : distance K B + distance K D + distance D C = 2 * square_side) : Prop

theorem find_angle_KAM
  (h1 : Square A B C D)
  (h2 : K ∈ LineSegment B C)
  (h3 : M ∈ LineSegment C D)
  (h4 : distance K B + distance K D + distance D C = 2 * square_side)
  : ∠KAM = (π / 4) :=
 sorry

end find_angle_KAM_l223_223525


namespace part1_part2_l223_223031

theorem part1 (A B C a b c : ℝ)
  (h_acute : ∀ {θ}, θ = A ∨ θ = B ∨ θ = C → 0 < θ ∧ θ < π/2)
  (h_sines : (sin A / sin C) - 1 = (sin A ^ 2 - sin C ^ 2) / (sin B ^ 2))
  (h_A_neq_C : A ≠ C) :
  B = 2 * C :=
sorry

theorem part2 (A B C a : ℝ) (BD : ℝ)
  (h_acute : ∀ {θ}, θ = A ∨ θ = B ∨ θ = C → 0 < θ ∧ θ < π/2)
  (h_a : a = 4)
  (h_B_2C : B = 2 * C) :
  ∃ (lower upper : ℝ), 
    ∀ (BD : ℝ), lower < BD ∧ BD < upper :=
sorry

end part1_part2_l223_223031


namespace perm_mississippi_l223_223741

theorem perm_mississippi : 
  let n := 11
  let n_S := 4
  let n_I := 4
  let n_P := 2
  let n_M := 1
  ∏ (mississippi_perm := Nat.factorial n / (Nat.factorial n_S * Nat.factorial n_I * Nat.factorial n_P * Nat.factorial n_M)) :=
  mississippi_perm = 34650 :=
by sorry

end perm_mississippi_l223_223741


namespace findValuesForFibSequence_l223_223954

noncomputable def maxConsecutiveFibonacciTerms (A B C : ℝ) : ℝ :=
  if A ≠ 0 then 4 else 0

theorem findValuesForFibSequence :
  maxConsecutiveFibonacciTerms (1/2) (-1/2) 2 = 4 ∧ maxConsecutiveFibonacciTerms (1/2) (1/2) 2 = 4 :=
by
  -- This statement will follow from the given conditions and the solution provided.
  sorry

end findValuesForFibSequence_l223_223954


namespace probability_of_sum_7_is_one_third_l223_223718

def altered_die_1 : list ℕ := [1, 1, 3, 3, 5, 5]
def altered_die_2 : list ℕ := [2, 2, 4, 4, 6, 6]

def possible_sums (d1 d2 : list ℕ) : list ℕ := 
  (list.product d1 d2).map (λ (p : ℕ × ℕ), p.fst + p.snd)

def count_occurrences {α : Type} [decidable_eq α] (a : α) (l : list α) : ℕ :=
  (l.filter (λ x, x = a)).length

noncomputable def probability_sum_7 (d1 d2 : list ℕ) : ℚ :=
  (count_occurrences 7 (possible_sums d1 d2)) / (d1.length * d2.length)

theorem probability_of_sum_7_is_one_third :
  probability_sum_7 altered_die_1 altered_die_2 = 1 / 3 :=
sorry

end probability_of_sum_7_is_one_third_l223_223718


namespace coffee_machine_pays_off_l223_223488

def daily_savings(previous_cost: ℕ, current_cost: ℕ) : ℕ :=
  previous_cost - current_cost

def days_to_payoff(machine_cost: ℕ, savings_per_day: ℕ) : ℕ :=
  machine_cost / savings_per_day

theorem coffee_machine_pays_off :
  let machine_cost := 200 - 20
  let previous_daily_expense := 2 * 4
  let current_daily_expense := 3
  let savings := daily_savings previous_daily_expense current_daily_expense
  let payoff_days := days_to_payoff machine_cost savings
  payoff_days = 36 :=
by
  -- Calculation/Proof goes here
  sorry

end coffee_machine_pays_off_l223_223488


namespace intersection_and_shape_l223_223074

open Real

noncomputable def A : Point := (0, 0)
noncomputable def B : Point := (0, 4)
noncomputable def C : Point := (4, 4)
noncomputable def D : Point := (4, 0)
noncomputable def line_from_A (x : ℝ) := x
noncomputable def line_from_B (x : ℝ) := 4 - x

theorem intersection_and_shape :
  let intersection := (2, 2)
  intersection ∈ (line_from_A 2, line_from_B 2) ∧
  shape_formed_by intersection A B C D = "smaller square" :=
by
  sorry

end intersection_and_shape_l223_223074


namespace equation_of_ab_equation_of_bc_area_of_bde_l223_223477

noncomputable def point := (ℝ × ℝ)
def line := ℝ × ℝ × ℝ -- Ax + By + C = 0

def point_coords (p : point) := (p.1, p.2)

def line_eq (l : line) (p : point) := 
  let (A, B, C) := l in 
  A * p.1 + B * p.2 + C = 0

def midpoint (A B : point) := 
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def distance (A B : point) := 
  sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

def area (A B C : point) := 
  (1 / 2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem equation_of_ab : 
  let A := (0, 1)
  let CD := (1, 2, -4) 
  ∃ AB : line, line_eq AB A ∧ AB = (2, -1, 1) := 
sorry

theorem equation_of_bc :
  let B := (1/2, 2)
  let C := (2, 1)
  let BE := (2, 1, -3)
  ∃ BC : line, line_eq BC B ∧ line_eq BC C ∧ BC = (2, 3, -7) :=
sorry

theorem area_of_bde :
  let B := (1/2, 2)
  let D := (2/5, 9/5)
  let E := (1, 1)
  area B D E = 1 / 10 :=
sorry

end equation_of_ab_equation_of_bc_area_of_bde_l223_223477


namespace complement_intersection_l223_223840

open Set 

variable (U A B : Set ℕ)
variable [DecidablePred (λ x, x ∈ A)] [DecidablePred (λ x, x ∈ B)] [DecidablePred (λ x, x ∈ U)]

def U := {1, 2, 3, 4, 5}
def A := {1, 2, 3}
def B := {2, 3, 4}

theorem complement_intersection :
  (U \ (A ∩ B)) = {1, 4, 5} :=
by
  sorry

end complement_intersection_l223_223840


namespace f_correct_l223_223770

noncomputable def f (n : ℕ) : ℕ :=
  if h : n ≥ 15 then (n - 1) / 2
  else if n = 3 then 1
  else if n = 4 then 1
  else if n = 5 then 2
  else if n = 6 then 4
  else if 7 ≤ n ∧ n ≤ 15 then 7
  else 0

theorem f_correct (n : ℕ) (hn : n ≥ 3) : 
  f n = if n ≥ 15 then (n - 1) / 2
        else if n = 3 then 1
        else if n = 4 then 1
        else if n = 5 then 2
        else if n = 6 then 4
        else if 7 ≤ n ∧ n ≤ 15 then 7
        else 0 := sorry

end f_correct_l223_223770


namespace radius_of_semi_circle_l223_223976

-- Given definitions and conditions
def perimeter : ℝ := 33.934511513692634
def pi_approx : ℝ := 3.141592653589793

-- The formula for the perimeter of a semi-circle
def semi_circle_perimeter (r : ℝ) : ℝ := pi_approx * r + 2 * r

-- The theorem we want to prove
theorem radius_of_semi_circle (r : ℝ) (h: semi_circle_perimeter r = perimeter) : r = 6.6 :=
sorry

end radius_of_semi_circle_l223_223976


namespace largest_invertible_interval_includes_2_l223_223548

def f : ℝ → ℝ := λ x, 3 * x^2 + 6 * x - 8

theorem largest_invertible_interval_includes_2 :
  ∃ (a b : ℝ), a ≤ 2 ∧ 2 ≤ b ∧ (∀ x1 x2 : ℝ, x1 ≠ x2 ∧ a ≤ x1 ∧ x1 ≤ b ∧ a ≤ x2 ∧ x2 ≤ b → f x1 ≠ f x2) ∧ a = -1 ∧ b = ∞ := 
sorry

end largest_invertible_interval_includes_2_l223_223548


namespace correlation_x_y_z_l223_223421

-- Define the relationship between x and y
def relationship_xy (x y : ℝ) : Prop :=
  y = -2 * x + 1

-- Define the positive correlation between y and z
def pos_correlation_y_z : Prop := sorry

-- Prove that x is negatively correlated with y and x is negatively correlated with z
theorem correlation_x_y_z (x y z : ℝ) (h1 : relationship_xy x y) (h2 : pos_correlation_y_z) :
  (correlation x y = -.correlation x y) ∧ (correlation x z = -.correlation x y * correlation y z) :=
begin
  sorry
end

end correlation_x_y_z_l223_223421


namespace angle_B1KB2_l223_223466

-- Definitions and conditions based on the problem statement
variables (A B C B1 C1 B2 C2 K : Type) 
variables [Triangle ABC] -- Representation of a triangle with angles sum to 180°
variables [Angle (A B C) = 35] -- Angle A as 35 degrees
variables [Altitude (B1 B) (A C)] -- BB1 is an altitude
variables [Altitude (C1 C) (A B)] -- CC1 is an altitude
variables [Midpoint B2 A C] -- B2 is the midpoint of AC
variables [Midpoint C2 A B] -- C2 is the midpoint of AB
variables [Intersection K (B1 C2) (C1 B2)] -- Intersection point K of B1C2 and C1B2

-- Statement to be proved: angle B1KB2 is 75 degrees
theorem angle_B1KB2 : Angle (B1 K B2) = 75 :=
sorry

end angle_B1KB2_l223_223466


namespace A2_B2_C2_not_all_inside_circumcircle_l223_223920

theorem A2_B2_C2_not_all_inside_circumcircle 
    (P A B C A1 B1 C1 A2 B2 C2 : Point) 
    (hP_inside_triangle: P ∈ interior (triangle A B C))
    (hA1_on_BC: AP ∩ BC = A1)
    (hB1_on_CA: BP ∩ CA = B1)
    (hC1_on_AB: CP ∩ AB = C1)
    (hA1_mid_PA2: midpoint A1 P A2)
    (hB1_mid_PB2: midpoint B1 P B2)
    (hC1_mid_PC2: midpoint C1 P C2)
    : ¬ (A2 ∈ interior (circumcircle (triangle A B C)) ∧ 
         B2 ∈ interior (circumcircle (triangle A B C)) ∧ 
         C2 ∈ interior (circumcircle (triangle A B C))) :=
sorry

end A2_B2_C2_not_all_inside_circumcircle_l223_223920


namespace circle_statements_correct_l223_223635

theorem circle_statements_correct :
  (∀ (circle : Type) (diameter chord : set circle), (diameter ⊥ chord) → bisects diameter chord) ∧
  (∀ (circle1 circle2 : Type) (angle1 angle2 : set (circle1 ∪ circle2)), 
    (congruent circle1 circle2) → (equal_central_angles angle1 angle2) → (equal_arcs angle1 angle2)) ∧
  (∀ (circle : Type) (chord1 chord2 : set circle),
    (equal_chords chord1 chord2) → (equal_arcs chord1 chord2)) ∧
  (∀ (circle : Type) (arc1 arc2 : set circle),
    (equal_arcs arc1 arc2) → (equal_central_angles arc1 arc2)) →
  number_of_correct_statements = 4 :=
sorry

end circle_statements_correct_l223_223635


namespace machine_pays_for_itself_in_36_days_l223_223487

def cost_of_machine : ℝ := 200
def discount : ℝ := 20
def cost_per_day : ℝ := 3
def previous_cost_per_coffee : ℝ := 4
def number_of_coffees_per_day : ℝ := 2

def net_cost_of_machine : ℝ := cost_of_machine - discount
def daily_savings : ℝ := number_of_coffees_per_day * previous_cost_per_coffee - cost_per_day

def days_until_machine_pays_for_itself : ℝ := net_cost_of_machine / daily_savings

theorem machine_pays_for_itself_in_36_days : days_until_machine_pays_for_itself = 36 := 
by {
  -- Proof skipped
  sorry
}

end machine_pays_for_itself_in_36_days_l223_223487


namespace total_pages_l223_223564

theorem total_pages (history_pages geography_additional math_factor science_factor : ℕ) 
  (h1 : history_pages = 160)
  (h2 : geography_additional = 70)
  (h3 : math_factor = 2)
  (h4 : science_factor = 2) 
  : let geography_pages := history_pages + geography_additional in
    let sum_history_geography := history_pages + geography_pages in
    let math_pages := sum_history_geography / math_factor in
    let science_pages := history_pages * science_factor in
    history_pages + geography_pages + math_pages + science_pages = 905 :=
by
  sorry

end total_pages_l223_223564


namespace race_orderings_l223_223847

theorem race_orderings (h r n m : Type) [fintype h] [fintype r] [fintype n] [fintype m] (disjoint : ∀ x:Type, x = h ∨ x = r ∨ x = n ∨ x = m → disjoint h r n m) : 
  fintype.card h * fintype.card r * fintype.card n * fintype.card m = 24 := 
sorry

end race_orderings_l223_223847


namespace vertex_of_parabola_l223_223574

theorem vertex_of_parabola : ∃ h k : ℝ, ∀ x : ℝ, y = (x - 1)^2 - 2 ∧ (h, k) = (1, -2) :=
begin
  -- condition: the equation of the parabola
  -- question: prove the vertex coordinates are (1, -2)
  sorry
end

end vertex_of_parabola_l223_223574


namespace inclination_angle_line_l223_223967

theorem inclination_angle_line (x y : ℝ) : x - y + 3 = 0 → real.angle (1, - 1) = 45 :=
begin
  sorry
end

end inclination_angle_line_l223_223967


namespace geom_series_first_term_l223_223222

theorem geom_series_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 30)
  (h2 : a^2 / (1 - r^2) = 120) : 
  a = 120 / 17 :=
by
  sorry

end geom_series_first_term_l223_223222


namespace midpoint_on_incircle_l223_223527

theorem midpoint_on_incircle
  {Z W Z_star W_star : ℂ}
  (h_isog : ∃ z w : ℂ, Z = z ∧ W = w ∧ z + w + conj z * conj w = 0)
  (h_inv : ∀ z w : ℂ, Z_star = -1 / conj z ∧ W_star = -1 / conj w ∧ abs z = 1 ∧ abs w = 1):
  abs ((Z_star + W_star) / 2) = 1 / 2 :=
by
  sorry

end midpoint_on_incircle_l223_223527


namespace average_speed_of_car_l223_223656

theorem average_speed_of_car : 
  let d1 := 80
  let d2 := 60
  let d3 := 40
  let d4 := 50
  let d5 := 30
  let d6 := 70
  let total_distance := d1 + d2 + d3 + d4 + d5 + d6
  let total_time := 6
  total_distance / total_time = 55 := 
by
  let d1 := 80
  let d2 := 60
  let d3 := 40
  let d4 := 50
  let d5 := 30
  let d6 := 70
  let total_distance := d1 + d2 + d3 + d4 + d5 + d6
  let total_time := 6
  show total_distance / total_time = 55
  sorry

end average_speed_of_car_l223_223656


namespace proof_p_and_q_true_l223_223389

open Real

def p : Prop := ∃ x : ℝ, x - 2 > log x
def q : Prop := ∀ x : ℝ, exp x > x

theorem proof_p_and_q_true : p ∧ q :=
by
  -- Assume you have already proven that p and q are true separately
  sorry

end proof_p_and_q_true_l223_223389


namespace segment_total_length_l223_223780

theorem segment_total_length
  (segments : Fin 2000 → ℝ)
  (h_each : ∀ i, 1 ≤ segments i)
  (h_no_polygon : ∀ S : Finset (Fin 2000), ∃ i ∈ S, segments i ≥ S.sum (λ j, segments j) - segments i) :
  Finset.univ.sum (λ i, segments i) ≥ 2 ^ 1999 :=
by
  sorry

end segment_total_length_l223_223780


namespace min_value_F_l223_223784

theorem min_value_F :
  ∀ (x y : ℝ), (x^2 + y^2 - 2*x - 2*y + 1 = 0) → (x + 1) / y ≥ 3 / 4 :=
by
  intro x y h
  sorry

end min_value_F_l223_223784


namespace length_of_TU_l223_223274

theorem length_of_TU 
  (PQ QR ST TU : ℝ)
  (h1 : PQ = 10)
  (h2 : QR = 15)
  (h3 : ST = 6) 
  (h4 : triangle_similarity : ∀ (a b c d e f : ℝ), a / d = b / e → b / e = c / f → a / d = c / f) : 
  TU = 9 := 
by 
  sorry

end length_of_TU_l223_223274


namespace min_max_value_z_l223_223534

theorem min_max_value_z (x y z : ℝ) (h1 : x^2 ≤ y + z) (h2 : y^2 ≤ z + x) (h3 : z^2 ≤ x + y) :
  -1/4 ≤ z ∧ z ≤ 2 :=
by {
  sorry
}

end min_max_value_z_l223_223534


namespace total_time_of_trip_l223_223269

-- Definitions for conditions
def speed_boat : ℝ := 8  -- Speed of the boat in standing water
def speed_stream : ℝ := 2  -- Speed of the stream
def distance : ℝ := 210  -- Distance to the place

-- Theorem statement
theorem total_time_of_trip : 
  let speed_downstream := speed_boat + speed_stream,
      speed_upstream := speed_boat - speed_stream,
      time_downstream := distance / speed_downstream,
      time_upstream := distance / speed_upstream
  in (time_downstream + time_upstream = 56) :=
by 
  -- skipping proof as per the user's instructions
  sorry

end total_time_of_trip_l223_223269


namespace combined_savings_after_four_weeks_l223_223535

-- Definitions based on problem conditions
def hourly_wage : ℕ := 10
def daily_hours : ℕ := 10
def days_per_week : ℕ := 5
def weeks : ℕ := 4

def robby_saving_ratio : ℚ := 2/5
def jaylene_saving_ratio : ℚ := 3/5
def miranda_saving_ratio : ℚ := 1/2

-- Definitions derived from the conditions
def daily_earnings : ℕ := hourly_wage * daily_hours
def total_working_days : ℕ := days_per_week * weeks
def monthly_earnings : ℕ := daily_earnings * total_working_days

def robby_savings : ℚ := robby_saving_ratio * monthly_earnings
def jaylene_savings : ℚ := jaylene_saving_ratio * monthly_earnings
def miranda_savings : ℚ := miranda_saving_ratio * monthly_earnings

def total_savings : ℚ := robby_savings + jaylene_savings + miranda_savings

-- The main theorem to prove
theorem combined_savings_after_four_weeks :
  total_savings = 3000 := by sorry

end combined_savings_after_four_weeks_l223_223535


namespace sum_largest_odd_factor_l223_223498

/-- Definition of the function f which gives the largest odd factor of n. -/
def largest_odd_factor (n : ℕ) : ℕ :=
  if n % 2 = 1 then n else largest_odd_factor (n / 2)

/-- Statement that proves the sum of the given function is equal to the correct solution. -/
theorem sum_largest_odd_factor :
  (∑ n in Finset.range 2048, largest_odd_factor (n + 1) / (n + 1)) ≈ 1365 :=
sorry

end sum_largest_odd_factor_l223_223498


namespace correct_conclusions_l223_223163

noncomputable def f (x : ℝ) : ℝ := log (2 : ℝ) (x^2 - 2 * x + 3)

theorem correct_conclusions : 
  (∀ x, x^2 - 2 * x + 3 > 0) ∧
  (∀ x, 1 ≤ x → ∀ x₁ x₂, x₁ ≤ x₂ → f x₁ ≤ f x₂) ∧
  (∃ x, x = 1 ∧ f x = 1) ∧
  (∀ x, f x > 0) := by
  sorry

end correct_conclusions_l223_223163


namespace find_abc_sum_l223_223916

variable (a b c x m : ℝ)

def equation (x : ℝ) : Prop := 
  (4 / (x - 4)) + (6 / (x - 6)) + (18 / (x - 18)) + (20 / (x - 20)) = x^2 - 12 * x - 6 + Real.cos x

noncomputable def largest_real_solution : ℝ :=
  classical.some (exists_real_sol (equation x))

theorem find_abc_sum (ha : a = 12) (hb : b = 50) (hc : c = 108) (hm : m = a + Real.sqrt (b + Real.sqrt c)) :
  a + b + c = 170 := 
by 
  have large_m : m = largest_real_solution := sorry
  rw [ha, hb, hc] at hm
  exact (by linarith)

end find_abc_sum_l223_223916


namespace sport_flavoring_to_corn_syrup_ratio_is_three_times_standard_l223_223040

-- Definitions based on conditions
def standard_flavor_to_water_ratio := 1 / 30
def standard_flavor_to_corn_syrup_ratio := 1 / 12
def sport_water_amount := 60
def sport_corn_syrup_amount := 4
def sport_flavor_to_water_ratio := 1 / 60
def sport_flavor_amount := 1 -- derived from sport_water_amount * sport_flavor_to_water_ratio

-- The main theorem to prove
theorem sport_flavoring_to_corn_syrup_ratio_is_three_times_standard :
  1 / 4 = 3 * (1 / 12) :=
by
  sorry

end sport_flavoring_to_corn_syrup_ratio_is_three_times_standard_l223_223040


namespace max_spend_calculation_l223_223932

variables (original_price discount_pct savings max_spend : ℝ)

-- Conditions
def original_price := 120
def discount_pct := 0.30
def savings := 46

-- Proof problem stating that the maximum amount Marcus was willing to spend is $130
theorem max_spend_calculation :
  let discounted_price := original_price * (1 - discount_pct) in
  max_spend = discounted_price + savings := by
    let discounted_price := original_price * (1 - discount_pct)
    sorry

end max_spend_calculation_l223_223932


namespace cos_alpha_in_second_quadrant_l223_223391

theorem cos_alpha_in_second_quadrant (α : ℝ) (h1 : α ∈ Icc π (2 * π)) (h2 : Real.sin α = 5 / 13) : Real.cos α = -12 / 13 :=
by
  sorry

end cos_alpha_in_second_quadrant_l223_223391


namespace min_value_expression_l223_223806

theorem min_value_expression (x : ℝ) (h : x > 3) : x + 4 / (x - 3) ≥ 7 :=
sorry

end min_value_expression_l223_223806


namespace remainder_b91_mod_50_l223_223506

theorem remainder_b91_mod_50 :
  let b (n : ℕ) := 7^n + 9^n
  in b 91 % 50 = 16 :=
by
  sorry

end remainder_b91_mod_50_l223_223506


namespace children_taking_the_bus_l223_223555

theorem children_taking_the_bus (seats children_per_seat : ℕ) (h₁ : seats = 29) (h₂ : children_per_seat = 2) :
  seats * children_per_seat = 58 :=
by
  rw [h₁, h₂]
  exact rfl


end children_taking_the_bus_l223_223555


namespace lcm_is_160_l223_223618

noncomputable def lcm_of_two_numbers {a b : ℕ} (hcf_ab : Nat.gcd a b = 16) (prod_ab : a * b = 2560) : Nat :=
Nat.lcm a b

theorem lcm_is_160 {a b : ℕ} (hcf_ab : Nat.gcd a b = 16) (prod_ab : a * b = 2560) : Nat.lcm a b = 160 :=
by
  have lcm_ab := lcm_of_two_numbers hcf_ab prod_ab
  rw [lcm_ab]
  sorry

end lcm_is_160_l223_223618


namespace total_pages_correct_l223_223560

def history_pages : ℕ := 160
def geography_pages : ℕ := history_pages + 70
def sum_history_geography_pages : ℕ := history_pages + geography_pages
def math_pages : ℕ := sum_history_geography_pages / 2
def science_pages : ℕ := 2 * history_pages
def total_pages : ℕ := history_pages + geography_pages + math_pages + science_pages

theorem total_pages_correct : total_pages = 905 := by
  -- The proof goes here.
  sorry

end total_pages_correct_l223_223560


namespace pile_splitting_l223_223113

theorem pile_splitting (single_stone_piles : ℕ) :
  ∃ (final_heap_size : ℕ), 
    (∀ heap_size ≤ single_stone_piles, heap_size > 0 → (heap_size * 2) ≥ heap_size) ∧ (final_heap_size = single_stone_piles) :=
by
  sorry

end pile_splitting_l223_223113


namespace basketball_tournament_ranking_sequences_l223_223030

theorem basketball_tournament_ranking_sequences :
  let teams := ["E", "F", "G", "H"]
  let first_round := [(teams[0], teams[1]), (teams[2], teams[3])]
  ∀ (W1 W2 L1 L2 : ℕ) (final_rankings : ℕ -> ℕ -> ℕ -> ℕ -> ℕ) (total_rankings : ℕ),
  -- Conditions for the winners and losers in first round and their subsequent matches
  (W1 = teams.indexOf("E") ∨ W1 = teams.indexOf("F")) ∧ (W2 = teams.indexOf("G") ∨ W2 = teams.indexOf("H")) →
  (L1 = teams.indexOf("E") ∨ L1 = teams.indexOf("F")) ∧ (L2 = teams.indexOf("G") ∨ L2 = teams.indexOf("H")) →
  -- Additional condition if E wins first match it must play G
  (W1 = teams.indexOf("E") → W2 = teams.indexOf("G") ∧ L2 = teams.indexOf("H")) →
  -- Possible finale rankings depending on outcomes of final matches
  final_rankings W1 W2 L1 L2 = 6 →
  total_rankings = 6 :=
by
  intros teams first_round W1 W2 L1 L2 final_rankings total_rankings
  sorry

end basketball_tournament_ranking_sequences_l223_223030


namespace range_of_a_l223_223412

def f (x : ℝ) : ℝ :=
if x ≥ -1 then -x^2 - 2*x + 1 else (1/2)^x

theorem range_of_a (a : ℝ) (h : f (3 - a^2) < f (2 * a)) : -2 < a ∧ a < 3 / 2 := 
sorry

end range_of_a_l223_223412


namespace stratified_sampling_calculation_l223_223658

theorem stratified_sampling_calculation (total_employees : ℕ) (senior_titles : ℕ) (intermediate_titles : ℕ) (junior_titles : ℕ) (sample_size : ℕ) :
  total_employees = 150 →
  senior_titles = 15 →
  intermediate_titles = 45 →
  junior_titles = 90 →
  sample_size = 30 →
  let senior_sample := sample_size * senior_titles / total_employees
  let intermediate_sample := sample_size * intermediate_titles / total_employees
  let junior_sample := sample_size * junior_titles / total_employees
  senior_sample = 3 ∧ intermediate_sample = 9 ∧ junior_sample = 18 :=
by
  intros
  have s1 : senior_sample = 3 := by sorry
  have s2 : intermediate_sample = 9 := by sorry
  have s3 : junior_sample = 18 := by sorry
  exact And.intro s1 (And.intro s2 s3)

end stratified_sampling_calculation_l223_223658


namespace determine_angle_A_max_triangle_area_l223_223876

-- Conditions: acute triangle with sides opposite to angles A, B, C as a, b, c.
variables {A B C a b c : ℝ}
-- Given condition on angles.
axiom angle_condition : 1 + (Real.sqrt 3 / 3) * Real.sin (2 * A) = 2 * Real.sin ((B + C) / 2) ^ 2 
-- Circumcircle radius
axiom circumcircle_radius : Real.pi > A ∧ A > 0 

-- Question I: Determine angle A
theorem determine_angle_A : A = Real.pi / 3 :=
by sorry

-- Given radius of the circumcircle
noncomputable def R := 2 * Real.sqrt 3 

-- Maximum area of triangle ABC
theorem max_triangle_area (a b c : ℝ) : ∃ area, area = 9 * Real.sqrt 3 :=
by sorry

end determine_angle_A_max_triangle_area_l223_223876


namespace vector_evaluation_l223_223756

-- Define the vectors
def v1 : ℝ × ℝ := (3, -2)
def v2 : ℝ × ℝ := (2, -6)
def v3 : ℝ × ℝ := (0, 3)
def scalar : ℝ := 5
def expected_result : ℝ × ℝ := (-7, 31)

-- Statement to be proved
theorem vector_evaluation : v1 - scalar • v2 + v3 = expected_result :=
by
  sorry

end vector_evaluation_l223_223756


namespace find_fifth_day_sales_l223_223695

-- Define the variables and conditions
variables (x : ℝ)
variables (a : ℝ := 100) (b : ℝ := 92) (c : ℝ := 109) (d : ℝ := 96) (f : ℝ := 96) (g : ℝ := 105)
variables (mean : ℝ := 100.1)

-- Define the mean condition which leads to the proof of x
theorem find_fifth_day_sales : (a + b + c + d + x + f + g) / 7 = mean → x = 102.7 := by
  intro h
  -- Proof goes here
  sorry

end find_fifth_day_sales_l223_223695


namespace volume_ratios_l223_223808
-- Import the broader library

-- Define the conditions for the proof
variables {A B C D P Q R S : Point}
def height (X : Point) : ℝ := sorry -- The height of point X from the plane BCD

-- Given conditions as per the problem statement
axiom height_Q : height Q = (height A + height P) / 2
axiom height_R : height R = height Q / 2
axiom height_S : height S = height R / 2
axiom height_P : height P = height S / 2

-- Define volume of tetrahedra based on their heights
def volume (X : Point) : ℝ := height X * 1 -- Assume base area is constant for simplicity

-- The volumes in question
def V_P_BCD := volume P
def V_A_BCD := volume A
def V_P_CDA := volume P
def V_P_DAB := volume P
def V_P_ABC := volume P

-- Proof statement
theorem volume_ratios : 
  V_P_ABC / V_A_BCD = 8 / 15 ∧ 
  V_P_BCD / V_A_BCD = 1 / 15 ∧ 
  V_P_CDA / V_A_BCD = 2 / 15 ∧ 
  V_P_DAB / V_A_BCD = 4 / 15 :=
sorry

end volume_ratios_l223_223808


namespace gnomes_can_be_saved_l223_223665

def hats : Fin 7 := Fin.mk 7 sorry -- since there are 7 colors

-- Define gnomes and their hats visibility
structure Gnome :=
  (visible_hats : Fin 5 → hats) -- each gnome can see 5 hats of others

-- Strategy each gnome uses to guess the hidden hat
def gnome_strategy (g : Gnome) : hats :=
  sorry -- the strategy based on the gnome's view of 5 hats to guess hidden hat

-- The main theorem to prove
theorem gnomes_can_be_saved (gnomes : Fin 6 → Gnome) : 
  ∃ (strategy : Fin 6 → hats), 
    (∃ n s, gnomes n s = strategy n) → 
    (3 ≤ (Finset.card (Finset.univ.filter (λ x, gnomes x == guessed_hidden_hat))) :=
  sorry

end gnomes_can_be_saved_l223_223665


namespace side_ratio_triangle_square_pentagon_l223_223310

-- Define the conditions
def perimeter_triangle (t : ℝ) := 3 * t = 18
def perimeter_square (s : ℝ) := 4 * s = 16
def perimeter_pentagon (p : ℝ) := 5 * p = 20

-- Statement to be proved
theorem side_ratio_triangle_square_pentagon 
  (t s p : ℝ)
  (ht : perimeter_triangle t)
  (hs : perimeter_square s)
  (hp : perimeter_pentagon p) : 
  (t / s = 3 / 2) ∧ (t / p = 3 / 2) := 
sorry

end side_ratio_triangle_square_pentagon_l223_223310


namespace seashells_given_to_Joan_l223_223165

def S_original : ℕ := 35
def S_now : ℕ := 17

theorem seashells_given_to_Joan :
  (S_original - S_now) = 18 := by
  sorry

end seashells_given_to_Joan_l223_223165


namespace arrangements_mississippi_l223_223725

open Nat

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem arrangements_mississippi : 
  let total := 11
  let m_count := 1
  let i_count := 4
  let s_count := 4
  let p_count := 2
  factorial total / (factorial m_count * factorial i_count * factorial s_count * factorial p_count) = 34650 :=
by
  -- This is the proof placeholder
  sorry

end arrangements_mississippi_l223_223725


namespace product_exceeds_1024_for_n_eq_10_l223_223419

theorem product_exceeds_1024_for_n_eq_10 :
  (∃ n : ℕ, n = 10 ∧ (∏ k in finset.range (n + 1), 2 ^ (k / 5)) > 2 ^ 10) :=
begin
  sorry
end

end product_exceeds_1024_for_n_eq_10_l223_223419


namespace geom_seq_ratio_l223_223903

noncomputable theory

variables {a₁ q : ℝ} 

def S (n : ℕ) : ℝ :=
a₁ * (1 - q^n) / (1 - q)

theorem geom_seq_ratio (h : S 6 / S 3 = 3) : S 9 / S 6 = 7 / 3 :=
by sorry

end geom_seq_ratio_l223_223903


namespace S_7_is_28_l223_223987

-- Define the arithmetic sequence and sum of first n terms
def a : ℕ → ℝ := sorry  -- placeholder for arithmetic sequence
def S (n : ℕ) : ℝ := sorry  -- placeholder for the sum of first n terms

-- Given conditions
def a_3 : ℝ := 3
def a_10 : ℝ := 10

-- Define properties of the arithmetic sequence
axiom a_n_property (n : ℕ) : a n = a 1 + (n - 1) * (a 10 - a 3) / (10 - 3)

-- Define the sum of first n terms
axiom sum_property (n : ℕ) : S n = n * (a 1 + a n) / 2

-- Given specific elements of the sequence
axiom a_3_property : a 3 = 3
axiom a_10_property : a 10 = 10

-- The statement to prove
theorem S_7_is_28 : S 7 = 28 :=
sorry

end S_7_is_28_l223_223987


namespace abs_eq_abs_l223_223512

theorem abs_eq_abs {a b c d e f : ℝ} (h1 : a * c * e ≠ 0)
  (h2 : ∀ x:ℝ, |a * x + b| + |c * x + d| = |e * x + f|) : a * d = b * c := 
sorry

end abs_eq_abs_l223_223512


namespace total_carriages_l223_223263

-- Definitions based on given conditions
def Euston_carriages := 130
def Norfolk_carriages := Euston_carriages - 20
def Norwich_carriages := 100
def Flying_Scotsman_carriages := Norwich_carriages + 20
def Victoria_carriages := Euston_carriages - 15
def Waterloo_carriages := Norwich_carriages * 2

-- Theorem to prove the total number of carriages is 775
theorem total_carriages : 
  Euston_carriages + Norfolk_carriages + Norwich_carriages + Flying_Scotsman_carriages + Victoria_carriages + Waterloo_carriages = 775 :=
by sorry

end total_carriages_l223_223263


namespace truck_mpg_l223_223491

/-- Let MPG be the miles per gallon the truck gets.
James gets paid $0.50 per mile, he pays $4.00 per gallon for gas,
and his profit from a 600-mile trip is $180.
Prove that MPG = 20. -/
theorem truck_mpg
  (MPG : ℝ)
  (earnings_per_mile : ℝ)
  (gas_cost_per_gallon : ℝ)
  (profit : ℝ)
  (trip_miles : ℝ)
  (earnings : earnings_per_mile * trip_miles = 300)
  (profit_eq : earnings - (trip_miles / MPG) * gas_cost_per_gallon = profit)
  (profit_value : profit = 180)
  (trip_miles_value : trip_miles = 600)
  (earnings_per_mile_value : earnings_per_mile = 0.50)
  (gas_cost_per_gallon_value : gas_cost_per_gallon = 4.00)
: MPG = 20 :=
by
  sorry

end truck_mpg_l223_223491


namespace a_share_is_6300_l223_223640

noncomputable def investment_split (x : ℝ) :  ℝ × ℝ × ℝ :=
  let a_share := x * 12
  let b_share := 2 * x * 6
  let c_share := 3 * x * 4
  (a_share, b_share, c_share)

noncomputable def total_gain : ℝ := 18900

noncomputable def a_share_calculation : ℝ :=
  let (a_share, b_share, c_share) := investment_split 1
  total_gain / (a_share + b_share + c_share) * a_share

theorem a_share_is_6300 : a_share_calculation = 6300 := by
  -- Here, you would provide the proof, but for now we skip it.
  sorry

end a_share_is_6300_l223_223640


namespace intersection_A_B_l223_223803

-- Definitions of the sets A and B according to the problem conditions
def A : Set ℝ := {y | ∃ x : ℝ, y = Real.sin x}
def B : Set ℝ := {y | ∃ x : ℝ, y = Real.sqrt (-x^2 + 4 * x - 3)}

-- The proof problem statement
theorem intersection_A_B :
  A ∩ B = {y | 0 ≤ y ∧ y ≤ 1} :=
by
  sorry

end intersection_A_B_l223_223803


namespace center_of_circle_line_passes_through_point_shortest_chord_length_line_not_tangent_l223_223403

section MathProof

-- Definitions
def circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4
def line (m : ℝ) (x y : ℝ) : Prop := x + m * y - m - 2 = 0

-- Assume the existence of the line and circle
variable (m : ℝ)
variable (x y : ℝ)

-- 1) Define the center of the circle
theorem center_of_circle (xc yc : ℝ) : circle xc yc → (xc, yc) = (1, 2) := by
  intro h
  sorry

-- 2) Check if the line passes through (2, 1)
theorem line_passes_through_point (m : ℝ) : line m 2 1 := by
  sorry

-- 3) Prove the shortest chord length
theorem shortest_chord_length (m : ℝ) : line m (2 * ℝ) (2 * ℝ) → (2 * ℝ) = 2 * Real.sqrt 2 := by
  sorry

-- 4) The line is not tangent to the circle
theorem line_not_tangent (m : ℝ) : ¬ (line m 2 1 ∧ (∀ x y, circle x y → (line m x y → false))) := by
  sorry

end MathProof

end center_of_circle_line_passes_through_point_shortest_chord_length_line_not_tangent_l223_223403


namespace heaps_combination_preserve_similarity_split_stones_into_similar_heaps_l223_223121

def initial_heaps (n : ℕ) : list ℕ := list.repeat 1 n

def combine_heaps (heaps : list ℕ) : list ℕ :=
  if heaps.length ≥ 2 then
    let min1 := list.minimum heaps,
        heaps' := list.erase heaps min1,
        min2 := list.minimum heaps'
    in
    if min1 ≤ min2 then
      (min1 + min2) :: list.erase heaps' min2
    else
      heaps
  else
    heaps

theorem heaps_combination_preserve_similarity (heaps : list ℕ) (h : ∀ x ∈ heaps, x = 1) :
  ∀ combined_heaps, combined_heaps = combine_heaps heaps →
  ∀ x y ∈ combined_heaps, x ≤ y → x + y ≤ 2 * y :=
sorry

theorem split_stones_into_similar_heaps (n : ℕ) :
  ∃ combined_heaps : list ℕ, ∀ x y ∈ combined_heaps, x ≤ y → x + y ≤ 2 * y :=
sorry

end heaps_combination_preserve_similarity_split_stones_into_similar_heaps_l223_223121


namespace split_into_similar_piles_l223_223086

def similar_sizes (x y : ℕ) : Prop := x ≤ 2 * y ∧ y ≤ 2 * x

theorem split_into_similar_piles (n : ℕ) (h : 0 < n) :
  ∃ (piles : list ℕ), (∀ x ∈ piles, x = 1) ∧ (list.sum piles = n) ∧
                       (∀ x y ∈ piles, similar_sizes x y) := 
sorry

end split_into_similar_piles_l223_223086


namespace find_multiple_l223_223307

/-- 
Given:
1. Hank Aaron hit 755 home runs.
2. Dave Winfield hit 465 home runs.
3. Hank Aaron has 175 fewer home runs than a certain multiple of the number that Dave Winfield has.

Prove:
The multiple of Dave Winfield's home runs that Hank Aaron's home runs are compared to is 2.
-/
def multiple_of_dave_hr (ha_hr dw_hr diff : ℕ) (m : ℕ) : Prop :=
  ha_hr + diff = m * dw_hr

theorem find_multiple :
  multiple_of_dave_hr 755 465 175 2 :=
by
  sorry

end find_multiple_l223_223307


namespace right_triangle_angles_ratio_l223_223457

theorem right_triangle_angles_ratio (α β : ℝ) (h1 : α + β = 90) (h2 : α / β = 3) :
  α = 67.5 ∧ β = 22.5 :=
sorry

end right_triangle_angles_ratio_l223_223457


namespace related_functions_count_l223_223859

theorem related_functions_count : 
  let y := λ x : ℝ, -x^2 in
  let range := {0, -1, -9} in
  ∃ (domains : set (set ℝ)), 
    (∀ f ∈ domains, ∀ x : ℝ, f x = y x) ∧ 
    (∃! D ∈ domains, range = {y x | x ∈ D}) → 
    domains.card = 9 :=
by
  sorry

end related_functions_count_l223_223859


namespace clock_hands_right_angles_in_two_days_l223_223850

theorem clock_hands_right_angles_in_two_days : 
  (∀ t, (t % 720 = 0 ∨ t % 720 = 360) -> ∀ k, k ≠ t + 180 ∧ k ≠ t - 180) ∧ -- condition 1 equivalent in lean formalism
  (∀ t in [0, 719], ak_ang = ...) → -- condition 2 equivalent
  hands_right_angle_24hours = 44 →
  hands_right_angle_48hours = 2 * hands_right_angle_24hours →
  hands_right_angle_48hours = 88 :=
sorry

end clock_hands_right_angles_in_two_days_l223_223850


namespace split_into_similar_piles_l223_223084

def similar_sizes (x y : ℕ) : Prop := x ≤ 2 * y ∧ y ≤ 2 * x

theorem split_into_similar_piles (n : ℕ) (h : 0 < n) :
  ∃ (piles : list ℕ), (∀ x ∈ piles, x = 1) ∧ (list.sum piles = n) ∧
                       (∀ x y ∈ piles, similar_sizes x y) := 
sorry

end split_into_similar_piles_l223_223084


namespace length_of_PC_l223_223890

theorem length_of_PC 
  (AB : ℝ) (BC : ℝ) (CA : ℝ) (similar : ∀ PA PB PC : ℝ, PA / PB = CA / AB ∧ PB = PC + BC) : 
  9 = ∃ PC, similar PC :=
by
  sorry

end length_of_PC_l223_223890


namespace smallest_positive_period_monotonically_decreasing_interval_area_of_triangle_l223_223410

def f (x : ℝ) : ℝ :=
  2 * sin x * cos x + 2 * sqrt 3 * (cos x)^2 - sqrt 3

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π := sorry

theorem monotonically_decreasing_interval (k : ℤ) :
  ∀ x, (k * π + π / 12 ≤ x ∧ x ≤ k * π + 7 * π / 12) → (∃ x1 x2 x3, x1 < x2 < x3 ∧ f x1 > f x2 ∧ f x2 > f x3) := sorry

theorem area_of_triangle (a b c : ℝ) (A B C : ℝ) (h1 : a = 7)
  (h2 : f (A / 2 - π / 6) = sqrt 3) (h3 : sin B + sin C = 13 * sqrt 3 / 14) :
  ∃ area, area = 10 * sqrt 3 :=
begin
  sorry
end

end smallest_positive_period_monotonically_decreasing_interval_area_of_triangle_l223_223410


namespace problem_statement_l223_223925

noncomputable def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
noncomputable def M : Set ℝ := {x | -1 < x ∧ x < 1}
noncomputable def N : Set ℝ := {x | 0 < x ∧ x < 2}
noncomputable def complement_U (s : Set ℝ) : Set ℝ := {x | x ∈ U ∧ x ∉ s}
noncomputable def intersection (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∈ B}

theorem problem_statement : intersection N (complement_U M) = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end problem_statement_l223_223925


namespace mistaken_multiplication_l223_223290

theorem mistaken_multiplication (x : ℕ) : 
  let a := 139
  let b := 43
  let incorrect_result := 1251
  (a * b - a * x = incorrect_result) ↔ (x = 34) := 
by 
  let a := 139
  let b := 43
  let incorrect_result := 1251
  sorry

end mistaken_multiplication_l223_223290


namespace total_workers_in_workshop_l223_223641

-- Definition of average salary calculation
def average_salary (total_salary : ℕ) (workers : ℕ) : ℕ := total_salary / workers

theorem total_workers_in_workshop :
  ∀ (W T R : ℕ),
  T = 5 →
  average_salary ((W - T) * 750) (W - T) = 700 →
  average_salary (T * 900) T = 900 →
  average_salary (W * 750) W = 750 →
  W = T + R →
  W = 20 :=
by
  sorry

end total_workers_in_workshop_l223_223641


namespace sum_diagonals_from_A_l223_223670

-- Define the hexagon inscribed in the circle with given sides
variables {A B C D E F : Type*} [Coord A] [Coord B] [Coord C] [Coord D] [Coord E] [Coord F]

-- Define the lengths of the sides
def side lengths : A -> B -> Real := λ AB, 20
def side lengths : B -> C -> Real := λ BC, 50
def side lengths : C -> D -> Real := λ CD, 30
def side lengths : D -> E -> Real := λ DE, 50
def side lengths : E -> F -> Real := λ EF, 50
def side lengths : F -> A -> Real := λ FA, 50

-- Define the lengths of the diagonals
noncomputable def diagonal lengths : A -> C -> Real := 62
noncomputable def diagonal lengths : A -> D -> Real := 74
noncomputable def diagonal lengths : A -> E -> Real := 54

-- Theorem stating the sum of the lengths of the diagonals
theorem sum_diagonals_from_A :
  diagonal lengths A C + diagonal lengths A D + diagonal lengths A E = 190 := by
  sorry

end sum_diagonals_from_A_l223_223670


namespace perm_mississippi_l223_223739

theorem perm_mississippi : 
  let n := 11
  let n_S := 4
  let n_I := 4
  let n_P := 2
  let n_M := 1
  ∏ (mississippi_perm := Nat.factorial n / (Nat.factorial n_S * Nat.factorial n_I * Nat.factorial n_P * Nat.factorial n_M)) :=
  mississippi_perm = 34650 :=
by sorry

end perm_mississippi_l223_223739


namespace focal_lengths_are_equal_l223_223417

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := 15 * y^2 - x^2 = 15

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

-- Define conditions
def foci_of_hyperbola : (ℝ × ℝ) → Prop := λ f, (f = (0, 4) ∨ f = (0, -4))
def focal_length_of_hyperbola : ℝ := 8
def eccentricity_of_hyperbola : ℝ := 4

def foci_of_ellipse : (ℝ × ℝ) → Prop := λ f, (f = (4, 0) ∨ f = (-4, 0))
def focal_length_of_ellipse : ℝ := 8
def eccentricity_of_ellipse : ℝ := 4 / 5

-- Lean statement to prove that the focal lengths are equal
theorem focal_lengths_are_equal :
  focal_length_of_hyperbola = focal_length_of_ellipse :=
by
  sorry

end focal_lengths_are_equal_l223_223417


namespace split_into_similar_heaps_l223_223107

noncomputable def similar_sizes (x y : ℕ) : Prop :=
  x ≤ 2 * y

theorem split_into_similar_heaps (n : ℕ) (h : n > 0) : 
  ∃ f : ℕ → ℕ, (∀ k, k < n → similar_sizes (f (k + 1)) (f k)) ∧ f (n - 1) = n := by
  sorry

end split_into_similar_heaps_l223_223107


namespace rearrange_digits_519_l223_223972

theorem rearrange_digits_519 :
  let largest := 951 in
  let smallest := 159 in
  largest - smallest = 792 :=
by
  let largest := 951
  let smallest := 159
  show largest - smallest = 792
  sorry

end rearrange_digits_519_l223_223972


namespace no_non_zero_integers_l223_223160

def move_initial_to_end (n : ℕ) (k : ℕ) : ℕ :=
  let d0 := n / 10^(k-1);
  let rest := n % 10^(k-1);
  rest * 10 + d0

theorem no_non_zero_integers :
  ∀ (n : ℕ) (k : ℕ), n ≠ 0 → 
  let N := move_initial_to_end n k in
  N ≠ 5 * n ∧ N ≠ 6 * n ∧ N ≠ 8 * n :=
by
  intro n k h;
  let N := move_initial_to_end n k;
  sorry

end no_non_zero_integers_l223_223160


namespace distance_ab_equation_of_line_ab_l223_223388

noncomputable def distance (A B : ℝ × ℝ) :=
  (real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2))

noncomputable def line_equation (A B : ℝ × ℝ) :=
  (A.2 - B.2) * (A.1 - B.1) = (B.1 - A.1) * (A.2 - B.2)

variables (A B : ℝ × ℝ)
variables length_ab : ℝ
variables equation_ab : ℝ × ℝ → ℝ

theorem distance_ab :
  distance (-1, 2) (3, 0) = 2 * real.sqrt 5 := 
sorry

theorem equation_of_line_ab :
  line_equation (-1, 2) (3, 0) = λ p, p.1 + 2 * p.2 - 3 := 
sorry

end distance_ab_equation_of_line_ab_l223_223388


namespace range_of_inclination_angle_l223_223019

open Real

theorem range_of_inclination_angle (x y α : ℝ) (hx : y = x^3 - 3 * x^2 + 3 - x) 
  (tangent_at_P : α = arctan (3 * x^2 - 6 * x + 3)) : 
  α ∈ [0, ∞) ∪ (-∞, π) := 
sorry

end range_of_inclination_angle_l223_223019


namespace maximal_federation_gain_l223_223454

def initial_teams : list ℕ := list.repeat 20 18
def final_teams : list ℕ := list.repeat 20 12 ++ [16, 16, 21, 22, 22, 23]

noncomputable def net_gain_federation (initial final : list ℕ) : ℤ :=
  let total_payment := final.sum - initial.sum
  in total_payment - ((initial.zip final).sum (λ ⟨x, y⟩, if y >= x then y - x else x - y))

theorem maximal_federation_gain : net_gain_federation initial_teams final_teams = 0 :=
by
  sorry

end maximal_federation_gain_l223_223454


namespace average_of_f_on_I_l223_223333

def f (x : ℝ) : ℝ := x^2 + Real.logb 2 x

def I := set.Icc 1 4

theorem average_of_f_on_I : 
  (∃ M : ℝ, ∀ x₁ ∈ I, ∃! x₂ ∈ I, (f x₁ + f x₂) / 2 = M) → 
  ∃ M : ℝ, M = 19 / 2 :=
by
  sorry

end average_of_f_on_I_l223_223333


namespace book_arrangement_count_l223_223681

theorem book_arrangement_count :
  (count_of_arrangements 7 [chinese, chinese, english, english, math, math, math]
  (λ L, adjacent L "chinese" ∧ adjacent L "english" ∧ math_non_adjacent L)) = 48 := by
  sorry

end book_arrangement_count_l223_223681


namespace combine_heaps_l223_223129

def heaps_similar (x y : ℕ) : Prop :=
  x ≤ 2 * y ∧ y ≤ 2 * x

theorem combine_heaps (n : ℕ) : 
  ∃ f : ℕ → ℕ, 
  f 0 = n ∧
  ∀ k, k < n → (∃ i j, i + j = k ∧ heaps_similar (f i) (f j)) ∧ 
  (∃ k, f k = n) :=
by
  sorry

end combine_heaps_l223_223129


namespace intersection_point_not_on_x_3_l223_223900

noncomputable def f (x : ℝ) : ℝ := (x^2 - 8*x + 15) / (3*x - 6)
noncomputable def g (x : ℝ) : ℝ := (-1/3 * x^2 + 6*x - 6) / (x - 2)

theorem intersection_point_not_on_x_3 : 
  ∃ x y : ℝ, (x ≠ 3) ∧ (f x = g x) ∧ (y = f x) ∧ (x = 11/3 ∧ y = -11/3) :=
by
  sorry

end intersection_point_not_on_x_3_l223_223900


namespace split_into_similar_heaps_l223_223104

noncomputable def similar_sizes (x y : ℕ) : Prop :=
  x ≤ 2 * y

theorem split_into_similar_heaps (n : ℕ) (h : n > 0) : 
  ∃ f : ℕ → ℕ, (∀ k, k < n → similar_sizes (f (k + 1)) (f k)) ∧ f (n - 1) = n := by
  sorry

end split_into_similar_heaps_l223_223104


namespace expression_evaluation_l223_223364

theorem expression_evaluation (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x^2 = 1 / y^2) :
  (x^2 - 4 / x^2) * (y^2 + 4 / y^2) = x^4 - 16 / x^4 :=
by
  sorry

end expression_evaluation_l223_223364


namespace combined_weight_of_three_parcels_l223_223935

theorem combined_weight_of_three_parcels (x y z : ℕ)
  (h1 : x + y = 112) (h2 : y + z = 146) (h3 : z + x = 132) :
  x + y + z = 195 :=
by
  sorry

end combined_weight_of_three_parcels_l223_223935


namespace max_sin_a_l223_223908

theorem max_sin_a (a b : ℝ) (h : Real.sin (a + b) = Real.sin a + Real.sin b) : Real.sin a ≤ 1 :=
by
  sorry

end max_sin_a_l223_223908


namespace number_of_correct_conclusions_l223_223973

theorem number_of_correct_conclusions :
  let A := "4 people visit different attractions"
  let B := "Xiao Zhao goes to an attraction alone"
  let f (x : ℝ) : ℝ := -- Assume a function with necessary properties.
  let ξ := normal (μ : ℝ) (seven : ennreal) in
  (P(A|B) = 2 / 9) ∧
  ((∃' ε, ∀' x, fderiv ℝ f (2 : ℝ) = -1) → differentiable_at ℝ f (2 : ℝ) ∧ deriv f (2) = -1) ∧
  ((P(ξ < 2) = P(ξ > 4)) → (μ = 3 ∧ variance ξ = 49)) → 
  num_correct_conclusions = 3 :=
begin
  sorry,
end

end number_of_correct_conclusions_l223_223973


namespace hp_parallel_df_l223_223475

open Real

/--
Let \( A \), \( B \), and \( C \) be points in the plane representing the vertices of a triangle. 
Let \( P \) be a point on the arc \( \overparen{BC} \) of the circumcircle of triangle \( ABC \) not containing \( A \).
Let \( D \) be the foot of the perpendicular from \( P \) to \( BC \) and \( F \) be the foot of the perpendicular from \( P \) to \( AB \).
Let \( H \) be the orthocenter of triangle \( ABC \).
Extend \( PD \) to \( P' \) such that \( PD = DP' \).
Prove that \( HP' \parallel DF \).
-/
theorem hp_parallel_df 
  (A B C P D F P' H : Point) 
  (hA_def : A = some_point) 
  (hB_def : B = some_point)
  (hC_def : C = some_point)
  (circ_A : circle_centered_at_center (circumcenter_of_triangle A B C) radius_circumcircle)
  (hP_on_circ : is_on_arc P (arc_of_circumcircle B C A circ_A))
  (hPD_perp_BC : is_foot_of_perpendicular D P (line_BC B C))
  (hPF_perp_AB: is_foot_of_perpendicular F P (line_AB A B))
  (hH_orthocenter: orthocenter_of_triangle H A B C)
  (hPD_eq_DP': midpoint D P P')
  : parallel (line_HP' H P') (line_DF D F) :=
sorry

end hp_parallel_df_l223_223475


namespace valid_three_digit_card_numbers_count_l223_223995

def card_numbers : List (ℕ × ℕ) := [(0, 1), (2, 3), (4, 5), (7, 8)]

def is_valid_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 -- Ensures it's three digits

def three_digit_numbers : List ℕ := 
  [201, 210, 102, 120, 301, 310, 103, 130, 401, 410, 104, 140,
   501, 510, 105, 150, 601, 610, 106, 160, 701, 710, 107, 170,
   801, 810, 108, 180, 213, 231, 312, 321, 413, 431, 512, 521,
   613, 631, 714, 741, 813, 831, 214, 241, 315, 351, 415, 451,
   514, 541, 615, 651, 716, 761, 815, 851, 217, 271, 317, 371,
   417, 471, 517, 571, 617, 671, 717, 771, 817, 871, 217, 271,
   321, 371, 421, 471, 521, 571, 621, 671, 721, 771, 821, 871]

def count_valid_three_digit_numbers : ℕ :=
  three_digit_numbers.length

theorem valid_three_digit_card_numbers_count :
    count_valid_three_digit_numbers = 168 :=
by
  -- proof goes here
  sorry

end valid_three_digit_card_numbers_count_l223_223995


namespace function_machine_output_l223_223883

def function_machine (input : ℝ) : ℝ :=
  let step1 := input * 3
  if step1 <= 30 then
    step1 + 10
  else
    step1 / 2

theorem function_machine_output : function_machine 15 = 22.5 := 
  by
    sorry

end function_machine_output_l223_223883


namespace bob_stickers_l223_223330

variables {B T D : ℕ}

theorem bob_stickers (h1 : D = 72) (h2 : T = 3 * B) (h3 : D = 2 * T) : B = 12 :=
by
  sorry

end bob_stickers_l223_223330


namespace monotonic_increasing_interval_l223_223203

noncomputable def f (x : ℝ) : ℝ := 3^(-|x-2|)

theorem monotonic_increasing_interval : 
  ∀ x : ℝ, x ≤ 2 -> (∀ y : ℝ, y = f x -> (f (-∞) ≤ y ∧ y ≤ f 2)) :=
by 
  sorry

end monotonic_increasing_interval_l223_223203


namespace pile_splitting_l223_223112

theorem pile_splitting (single_stone_piles : ℕ) :
  ∃ (final_heap_size : ℕ), 
    (∀ heap_size ≤ single_stone_piles, heap_size > 0 → (heap_size * 2) ≥ heap_size) ∧ (final_heap_size = single_stone_piles) :=
by
  sorry

end pile_splitting_l223_223112


namespace find_f_of_8_l223_223017

def f (x : ℝ) : ℝ := (3 * x + 2) / (x - 3)

theorem find_f_of_8 : f 8 = 26 / 5 := by
  sorry

end find_f_of_8_l223_223017


namespace exists_unique_a_l223_223983

noncomputable def f : ℕ → (ℝ → ℝ)
| 1 := id
| (n+1) := λ x, (f n x) * ((f n x) + (1 / n))

theorem exists_unique_a :
  ∃! a : ℝ, (0 < a) ∧ (∀ n : ℕ, n ≥ 1 → 0 < (f n a) ∧ (f n a) < (f (n + 1) a) ∧ (f (n + 1) a) < 1) :=
sorry

end exists_unique_a_l223_223983


namespace split_stones_l223_223078

theorem split_stones (n : ℕ) :
  ∃ (heaps : list ℕ), (∀ h ∈ heaps, 1 ≤ h ∧ h ≤ n) ∧ (∀ i j, i ≠ j → (i < heaps.length ∧ j < heaps.length → heaps.nth i ≤ 2 * heaps.nth j)) :=
sorry

end split_stones_l223_223078


namespace problem1_problem2_l223_223516

-- Definitions of M and N
def setM : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}
def setN (k : ℝ) : Set ℝ := {x | x - k ≤ 0}

-- Problem 1: Prove that if M ∩ N has only one element, then k = -1
theorem problem1 (h : ∀ x, x ∈ setM ∩ setN k → x = -1) : k = -1 := by 
  sorry

-- Problem 2: Given k = 2, prove the sets M ∩ N and M ∪ N
theorem problem2 (hk : k = 2) : (setM ∩ setN k = {x | -1 ≤ x ∧ x ≤ 2}) ∧ (setM ∪ setN k = {x | x ≤ 5}) := by
  sorry

end problem1_problem2_l223_223516


namespace charlotte_distance_l223_223713

-- Define the conditions as constants
constant speed : ℕ := 10 -- in miles per hour
constant time : ℕ := 6 -- in hours

-- Define the expected distance
constant expected_distance : ℕ := 60 -- in miles

-- State the theorem
theorem charlotte_distance : speed * time = expected_distance := by
  sorry

end charlotte_distance_l223_223713


namespace find_first_term_l223_223218

noncomputable def firstTermOfGeometricSeries (a r : ℝ) : Prop :=
  (a / (1 - r) = 30) ∧ (a^2 / (1 - r^2) = 120)

theorem find_first_term :
  ∃ a r : ℝ, firstTermOfGeometricSeries a r ∧ a = 120 / 17 :=
by
  sorry

end find_first_term_l223_223218


namespace hyperbola_foci_ellipse_major_axis_l223_223862

theorem hyperbola_foci_ellipse_major_axis :
  (∃ m : ℝ, ∀ x y : ℝ, (frac x^2 3 + frac y^2 4 = 1) →
    (frac (y^2) 2 - frac (x^2) m = 1) →
    sqrt (2 + m) = 2) :=
begin
  let m := 2,
  use m,
  intros x y h_ellipse h_hyperbola,
  have h1 : sqrt (2 + m) = sqrt 4 := by rw [m],
  rw sqrt_eq_rfl at h1,
  norm_num at h1,
  exact h1,
end

end hyperbola_foci_ellipse_major_axis_l223_223862


namespace coeff_third_term_sqrt2x_minus_one_pow5_l223_223570

theorem coeff_third_term_sqrt2x_minus_one_pow5 :
  let expr := (sqrt 2 * x - 1) ^ 5,
  let third_term_coeff := 20 * sqrt 2 in
  -- Define the binomial coefficient c(5, 2)
  ∃ c : ℝ,
    (c = binomial 5 2) ∧
    -- Define the third term calculation
    (third_term_coeff = c * ((sqrt 2) * x) ^ 3 * (-1) ^ 2) := by
  sorry

end coeff_third_term_sqrt2x_minus_one_pow5_l223_223570


namespace yonderland_license_plates_l223_223687

/-!
# Valid License Plates in Yonderland

A valid license plate in Yonderland consists of three letters followed by four digits. 

We are tasked with determining the number of valid license plates possible under this format.
-/

def num_letters : ℕ := 26
def num_digits : ℕ := 10
def letter_combinations : ℕ := num_letters ^ 3
def digit_combinations : ℕ := num_digits ^ 4
def total_combinations : ℕ := letter_combinations * digit_combinations

theorem yonderland_license_plates : total_combinations = 175760000 := by
  sorry

end yonderland_license_plates_l223_223687


namespace find_y_l223_223969

-- Definitions based on conditions
def mean_of_three (a b c : ℕ) : ℕ := (a + b + c) / 3
def mean_of_two (a b : ℕ) : ℕ := (a + b) / 2

-- Given conditions
def condition1 : mean_of_three 4 6 20 = 10 := rfl
def condition2 : (mean_of_two 14 y) = mean_of_three 4 6 20 := rfl

-- Proof statement
theorem find_y (y : ℕ) (h1 : mean_of_three 4 6 20 = 10) (h2 : mean_of_two 14 y = 10) : y = 6 := by
  sorry

end find_y_l223_223969


namespace total_students_in_class_is_36_l223_223714

theorem total_students_in_class_is_36
  (n: ℕ)
  (middle_row_size: n = 6 + 7 - 1)
  (equal_rows: ∀ (i : ℕ), i ∈ {1, 2, 3} → num_students_in_row i = n)
:
  num_students_in_class = 3 * middle_row_size :=
by
  sorry

end total_students_in_class_is_36_l223_223714


namespace perpendicular_bisector_fixed_point_l223_223568

theorem perpendicular_bisector_fixed_point (A B C E F N : Point) (Ω : Circle) 
  (h₁: OnCircle A Ω)
  (h₂: OnCircle B Ω)
  (h₃: OnCircle C Ω)
  (h₄: Collinear A B E)
  (h₅: Collinear A C F)
  (h₆: BE = CF)
  (h₇: Center Ω = N) :
  PassesThroughPerpendicularBisector N E F := 
sorry

end perpendicular_bisector_fixed_point_l223_223568


namespace y_expression_l223_223817

theorem y_expression (x y : ℝ) (h : 4 * x + y = 9) : y = 9 - 4 * x := 
by
  sorry

end y_expression_l223_223817


namespace remainder_of_expression_l223_223558

theorem remainder_of_expression (x y u v : ℕ) (h : x = u * y + v) (Hv : 0 ≤ v ∧ v < y) :
  (if v + 2 < y then (x + 3 * u * y + 2) % y = v + 2
   else (x + 3 * u * y + 2) % y = v + 2 - y) :=
by sorry

end remainder_of_expression_l223_223558


namespace find_k_from_line_intercepts_l223_223930

theorem find_k_from_line_intercepts (k : ℝ) (h₁ : k ≠ 3) 
  (h₂ : let x := k - 3 in let y := (2*k - 6)/(k - 3) in x + y = 0) : k = 1 := 
by
  sorry

end find_k_from_line_intercepts_l223_223930


namespace parity_S_l223_223398

variable (a b c n : ℤ)

theorem parity_S :
  (∃ p q r : ℤ, 
    p ∈ {0, 1} ∧ q ∈ {0, 1} ∧ r ∈ {0, 1} ∧ p ≠ q ∧ q = r ∧ a % 2 = p ∧ b % 2 = q ∧ c % 2 = r) →
  ∃ k : ℤ, 
    (a + 2 * n + 1) * (b + 2 * n + 2) * (c + 2 * n + 3) = 2 * k :=
begin
  -- This is where the proof would go, but we omit it as per the instructions.
  sorry
end

end parity_S_l223_223398


namespace solve_inequality_l223_223950

theorem solve_inequality (a x : ℝ) :
  ((x - a) * (x - 2 * a) < 0) ↔ 
  ((a < 0 ∧ 2 * a < x ∧ x < a) ∨ (a = 0 ∧ false) ∨ (a > 0 ∧ a < x ∧ x < 2 * a)) :=
by sorry

end solve_inequality_l223_223950


namespace triangle_ABC_x_range_l223_223891

theorem triangle_ABC_x_range (a x : ℝ) (b : ℝ) (B : ℝ) 
(h1 : a = x) (h2 : b = 2) (h3 : B = 60) (h4: ∃ A C : ℝ, A + C = 120 ∧ 60 < A ∧ A < 120 ∧ 2 triangles with such configurations exists) :
  2 < x ∧ x < 4 * Real.sqrt 3 / 3 :=
by
  sorry

end triangle_ABC_x_range_l223_223891


namespace permutations_mississippi_l223_223750

theorem permutations_mississippi : 
  let total_letters := 11
  let m_count := 1
  let i_count := 4
  let s_count := 4
  let p_count := 2
  (Nat.factorial total_letters / (Nat.factorial m_count * Nat.factorial i_count * Nat.factorial s_count * Nat.factorial p_count)) = 34650 := 
by
  sorry

end permutations_mississippi_l223_223750


namespace kyle_lift_weight_l223_223898

theorem kyle_lift_weight (this_year_weight last_year_weight : ℕ) 
  (h1 : this_year_weight = 80) 
  (h2 : this_year_weight = 3 * last_year_weight) : 
  (this_year_weight - last_year_weight) = 53 := by
  sorry

end kyle_lift_weight_l223_223898


namespace find_m_l223_223578

-- Conditions given
def ellipse (x y m : ℝ) : Prop := (x^2 / m) + (y^2 / 4) = 1
def eccentricity (e : ℝ) : Prop := e = 2

-- The theorem to prove
theorem find_m (m : ℝ) (h₁ : ellipse 1 1 m) (h₂ : eccentricity 2) : m = 3 ∨ m = 5 :=
  sorry

end find_m_l223_223578


namespace arrangements_mississippi_l223_223728

open Nat

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem arrangements_mississippi : 
  let total := 11
  let m_count := 1
  let i_count := 4
  let s_count := 4
  let p_count := 2
  factorial total / (factorial m_count * factorial i_count * factorial s_count * factorial p_count) = 34650 :=
by
  -- This is the proof placeholder
  sorry

end arrangements_mississippi_l223_223728


namespace angle_between_a_and_b_eq_pi_over_4_l223_223005

/-- Definitions of the vectors and their properties -/
namespace vector_angle_problem

def a : ℝ × ℝ := (1, 2)
def norm_b : ℝ := real.sqrt 10
def norm_a_plus_b : ℝ := 5

/-- Scalar product properties and the desired angle between vectors a and b -/
theorem angle_between_a_and_b_eq_pi_over_4 (b : ℝ × ℝ)
    (h_norm_b : real.sqrt (b.1 ^ 2 + b.2 ^ 2) = real.sqrt 10)
    (h_norm_a_plus_b : real.sqrt ((a.1 + b.1) ^ 2 + (a.2 + b.2) ^ 2) = 5) :
    real.angle (a, b) = real.pi / 4 :=
sorry

end vector_angle_problem

end angle_between_a_and_b_eq_pi_over_4_l223_223005


namespace sum_of_solutions_l223_223628

theorem sum_of_solutions :
  let f : ℤ → ℤ := λ x, 20 * x ^ 2 - 23 * x - 21 in
  (5 * (f 1) + 3) * (4 * (f 1) - 7) = 0 →
  ((- (-23) / 20) = 23 / 20) :=
by
  intro _,
  norm_num,
  sorry

end sum_of_solutions_l223_223628


namespace find_f_9_over_2_l223_223927

noncomputable def f (x : ℝ) : ℝ := 
  if x ∈ set.Ioc 1 2 then -2 * x^2 + 2
  else if x ∈ set.Ioc (-2) (-1) then -2 * (x+4)^2 + 2
  else -2 * (x - (4 * ⌊x / 4⌋))^2 + 2

theorem find_f_9_over_2 :
  (f (9 / 2) = 5 / 2) :=
by
  sorry

end find_f_9_over_2_l223_223927


namespace sin_alpha_cos_half_beta_minus_alpha_l223_223775

open Real

noncomputable def problem_condition (α β : ℝ) : Prop :=
  0 < α ∧ α < π / 2 ∧
  0 < β ∧ β < π / 2 ∧
  sin (π / 3 - α) = 3 / 5 ∧
  cos (β / 2 - π / 3) = 2 * sqrt 5 / 5

theorem sin_alpha (α β : ℝ) (h : problem_condition α β) : 
  sin α = (4 * sqrt 3 - 3) / 10 := sorry

theorem cos_half_beta_minus_alpha (α β : ℝ) (h : problem_condition α β) :
  cos (β / 2 - α) = 11 * sqrt 5 / 25 := sorry

end sin_alpha_cos_half_beta_minus_alpha_l223_223775


namespace evaluate_at_3_l223_223438

def g (x : ℝ) : ℝ := 3 * x^4 - 5 * x^3 + 2 * x^2 + x + 6

theorem evaluate_at_3 : g 3 = 135 := 
  by
  sorry

end evaluate_at_3_l223_223438


namespace split_into_similar_heaps_l223_223103

noncomputable def similar_sizes (x y : ℕ) : Prop :=
  x ≤ 2 * y

theorem split_into_similar_heaps (n : ℕ) (h : n > 0) : 
  ∃ f : ℕ → ℕ, (∀ k, k < n → similar_sizes (f (k + 1)) (f k)) ∧ f (n - 1) = n := by
  sorry

end split_into_similar_heaps_l223_223103


namespace arrangements_mississippi_l223_223726

open Nat

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem arrangements_mississippi : 
  let total := 11
  let m_count := 1
  let i_count := 4
  let s_count := 4
  let p_count := 2
  factorial total / (factorial m_count * factorial i_count * factorial s_count * factorial p_count) = 34650 :=
by
  -- This is the proof placeholder
  sorry

end arrangements_mississippi_l223_223726


namespace sequence_general_term_l223_223000

theorem sequence_general_term (a : ℕ → ℝ) 
  (h1 : a 1 = 1) 
  (h2 : a 2 = (1 + Real.sqrt 5) / 2) 
  (h3 : ∀ n, n ≥ 2 → (Real.sqrt 5 - 1) * a n ^ 2 * a (n - 2) = 2 * a (n - 1) ^ 3) : 
  ∀ n, a n = (1 + Real.sqrt 5) / 2 ^ (n - 1) :=
begin
  sorry
end

end sequence_general_term_l223_223000


namespace sqrt_div_6_eq_1_l223_223256

theorem sqrt_div_6_eq_1 :
  (sqrt 36) / 6 = 1 :=
by
  sorry

end sqrt_div_6_eq_1_l223_223256


namespace non_degenerate_triangles_l223_223301

theorem non_degenerate_triangles :
  let total_points := 16
  let collinear_points := 5
  let total_triangles := Nat.choose total_points 3
  let degenerate_triangles := 2 * Nat.choose collinear_points 3
  let nondegenerate_triangles := total_triangles - degenerate_triangles
  nondegenerate_triangles = 540 := 
by
  sorry

end non_degenerate_triangles_l223_223301


namespace planting_schemes_in_hexagon_l223_223456

theorem planting_schemes_in_hexagon :
  let regions := ["A", "B", "C", "D", "E", "F"].toFinset
      adjacent : Finset String → Finset (String × String) :=
        (fun regions => 
          Finset.filter 
            (fun (pair : String × String) => 
              (pair.1, pair.2) ∈ [("A", "B"), ("A", "F"), ("B", "C"), ("C", "D"),
                                  ("D", "E"), ("E", "F"), ("F", "A"), ("F", "D"),
                                  ("A", "C"), ("C", "E"), ("E", "A")].toFinset ∨
              (pair.2, pair.1) ∈ [("A", "B"), ("A", "F"), ("B", "C"), ("C", "D"),
                                  ("D", "E"), ("E", "F"), ("F", "A"), ("F", "D"),
                                  ("A", "C"), ("C", "E"), ("E", "A")].toFinset)
            (regions.product regions))
  in
  let plants := [0, 1, 2, 3].toFinset
  in
  let valid_placements := 
    Finset.filter 
      (fun placement : regions → { val : Finset ℕ // val.card = 1}) => 
        ∀ pair : String × String, pair ∈ adjacent regions → 
        placement pair.1.val ≠ placement pair.2.val
      (regions.pi (fun _ => plants))
  in
  valid_placements.card = 732 := 
begin
  sorry
end

end planting_schemes_in_hexagon_l223_223456


namespace compare_functions_l223_223915

noncomputable def a := sorry
noncomputable def x := sorry
def f (x : ℝ) (a : ℝ) := a^x
def g (x : ℝ) := x^(1/3 : ℝ)
def h (x : ℝ) (a : ℝ) := Real.logBase a x

-- Given conditions
axiom log_a_1_minus_a_sq_pos : Real.logBase a (1 - a^2) > 0
axiom a_pos : 0 < a
axiom a_lt_1 : a < 1
axiom x_gt_1 : x > 1

-- Prove the mathematically equivalent statement
theorem compare_functions (a : ℝ) (x : ℝ) (h_pos : 0 < a) (h_lt_1 : a < 1) (hx_gt_1 : x > 1) (h_log_cond : Real.logBase a (1 - a^2) > 0) : 
  h x a < f x a ∧ f x a < g x 
  := by sorry

end compare_functions_l223_223915


namespace unique_arrangements_mississippi_l223_223743

open scoped Nat

theorem unique_arrangements_mississippi : 
  let n := 11
  let n_M := 1
  let n_I := 4
  let n_S := 4
  let n_P := 2
  (n.fact / (n_M.fact * n_I.fact * n_S.fact * n_P.fact)) = 34650 :=
  by
  sorry

end unique_arrangements_mississippi_l223_223743


namespace problem_statement_l223_223420

def f (x : ℝ) : ℝ := x^2 + x + 1
def g (x : ℝ) : ℝ := 2 ^ f x

theorem problem_statement :
  (∀ x : ℝ, f(x + 1) - f(x) = 2 * x + 2) ∧ f 0 = 1 ∧
  (∀ x, x ∈ set.Icc (-1 : ℝ) (1 : ℝ) → g(x) ∈ set.Icc (real.sqrt (real.sqrt 8)) 8) :=
by {
  sorry
}

end problem_statement_l223_223420


namespace area_of_region_forming_points_l223_223380

structure ConeBaseRegion where
  apex : Point3D
  base_radius : ℝ
  height : ℝ
  angle_max : ℝ

def point_constraint (P Q : Point3D) (c : ConeBaseRegion) : Prop :=
  let angle := atan ((Q.y - P.y) / (sqrt ((Q.x - P.x)^2 + (Q.z - P.z)^2)))
  angle ≤ c.angle_max

noncomputable def region_area (c : ConeBaseRegion) : ℝ :=
  let large_area := π * c.base_radius^2
  let small_area := π * (c.height * tan c.angle_max)^2
  large_area - small_area

theorem area_of_region_forming_points (c : ConeBaseRegion) (a : ℝ) :
  c.base_radius = 2 ∧ c.height = 1 ∧ c.angle_max = π / 4 ∧ a = region_area c → a = 3 * π :=
sorry

end area_of_region_forming_points_l223_223380


namespace expected_value_sum_until_6_l223_223774

noncomputable def die_probability_6 := 3 / 8
noncomputable def die_probability_4 := 1 / 4
noncomputable def die_probability_other := 1 / 20

noncomputable def expected_single_roll_value : ℝ :=
  (1 / 20 ) * 1 + (1 / 20 ) * 2 + (1 / 20 ) * 3 + (1 / 4 ) * 4 + (1 / 20 ) * 5 + (3 / 8 ) * 6

noncomputable def expected_rolls_until_6 : ℝ :=
  1 / die_probability_6

noncomputable def expected_sum_until_6 : ℝ :=
  expected_rolls_until_6 * expected_single_roll_value

theorem expected_value_sum_until_6 :
  expected_sum_until_6 = 9.4 :=
by
  sorry

end expected_value_sum_until_6_l223_223774


namespace a_100_value_l223_223384

def sequence_a : ℕ → ℚ
| 0       := 0  -- Note: Using 0-based indexing for simplicity
| 1       := 1
| (n + 2) := (3 * (∑ i in finset.range (n + 2), sequence_a i) ^ 2) / (3 * (∑ i in finset.range (n + 2), sequence_a i) - 2)

noncomputable def sum_S : ℕ → ℚ
| 0       := 0
| (n + 1) := ∑ i in finset.range (n + 2), sequence_a i

theorem a_100_value : sequence_a 100 = -3 / 88210 := by
  sorry

end a_100_value_l223_223384


namespace complex_power_calculation_l223_223314

theorem complex_power_calculation : (1 - complex.I) / real.sqrt 2 ^ 48 = 1 :=
by
  sorry

end complex_power_calculation_l223_223314


namespace find_CB_in_acute_triangle_l223_223793

theorem find_CB_in_acute_triangle 
  (A B C D E M : Type)
  (hABC_acute : acute_triangle A B C)
  (hAD_altitude : altitude A D B C)
  (hCE_altitude : altitude C E A B)
  (hBM_altitude : altitude B M A C)
  (hCD : dist C D = 7)
  (hDE : dist D E = 7)
  (hDM : dist D M = 8) : 
  dist C B = Real.sqrt 113 :=
sorry

end find_CB_in_acute_triangle_l223_223793


namespace sum_of_cubes_l223_223557

theorem sum_of_cubes (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + ac + bc = 3) (h3 : abc = 5) : a^3 + b^3 + c^3 = 15 :=
by
  sorry

end sum_of_cubes_l223_223557


namespace trajectory_is_ray_l223_223777

structure Point where
  x : ℝ
  y : ℝ

def distance (P₁ P₂ : Point) : ℝ :=
  real.sqrt ((P₁.x - P₂.x)^2 + (P₁.y - P₂.y)^2)

def trajectory_of_P (P F₁ F₂ : Point) : Prop :=
  abs (distance P F₁ - distance P F₂) = 10

theorem trajectory_is_ray :
  let F₁ : Point := ⟨-8, 3⟩
  let F₂ : Point := ⟨2, 3⟩
  ∃ P : Point, trajectory_of_P P F₁ F₂ :=
sorry

end trajectory_is_ray_l223_223777


namespace circle_tangent_sum_l223_223321

noncomputable def tangent_length (A O : Point) (r : ℝ) : ℝ :=
  Real.sqrt (A.dist O ^ 2 - r ^ 2)

noncomputable def AB_AC_sum (A B C O : Point) (r BC : ℝ) : ℝ :=
  2 * (tangent_length A O r) + BC

theorem circle_tangent_sum (O A B C : Point) (r : ℝ) (OA : ℝ) (BC : ℝ) 
  (hO : dist O A = OA) (hBC : BC = 9) (hOA : OA = 10) (hR : r = 3) (h_exterior_triangle : true): 
  AB_AC_sum A B C O r BC = 2 * (Real.sqrt 91) + 9 :=
by
  sorry

end circle_tangent_sum_l223_223321


namespace molecular_weight_l223_223252

theorem molecular_weight : 
  let atomic_weight_Al := 26.98
  let atomic_weight_O := 16.00
  let atomic_weight_H := 1.01
  let num_Al := 1
  let num_O := 3
  let num_H := 3
  (num_Al * atomic_weight_Al + num_O * atomic_weight_O + num_H * atomic_weight_H) = 78.01 := 
by 
  let atomic_weight_Al := 26.98
  let atomic_weight_O := 16.00
  let atomic_weight_H := 1.01
  let num_Al := 1
  let num_O := 3
  let num_H := 3
  have h : num_Al * atomic_weight_Al + num_O * atomic_weight_O + num_H * atomic_weight_H = 78.01 := 
    sorry
  exact h

end molecular_weight_l223_223252


namespace joy_sixth_time_is_87_seconds_l223_223877

def sixth_time (times : List ℝ) (new_median : ℝ) : ℝ :=
  let sorted_times := times |>.insertNth 2 (2 * new_median - times.nthLe 2 sorry)
  2 * new_median - times.nthLe 2 sorry

theorem joy_sixth_time_is_87_seconds (times : List ℝ) (new_median : ℝ) :
  times = [82, 85, 93, 95, 99] → new_median = 90 →
  sixth_time times new_median = 87 :=
by
  intros h_times h_median
  rw [h_times]
  rw [h_median]
  sorry

end joy_sixth_time_is_87_seconds_l223_223877


namespace sqrt_inequality_l223_223313

theorem sqrt_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) : 
  x^2 + y^2 + 1 ≤ Real.sqrt ((x^3 + y + 1) * (y^3 + x + 1)) :=
sorry

end sqrt_inequality_l223_223313


namespace children_difference_l223_223277

theorem children_difference (initial_count : ℕ) (remaining_count : ℕ) (difference : ℕ) 
  (h1 : initial_count = 41) (h2 : remaining_count = 18) :
  difference = initial_count - remaining_count := 
by
  sorry

end children_difference_l223_223277


namespace interest_calculation_years_l223_223577

noncomputable def principal : ℝ := 625
noncomputable def rate : ℝ := 0.04
noncomputable def difference : ℝ := 1

theorem interest_calculation_years (n : ℕ) : 
    (principal * (1 + rate)^n - principal - (principal * rate * n) = difference) → 
    n = 2 :=
by sorry

end interest_calculation_years_l223_223577


namespace expression_value_l223_223339

theorem expression_value :
  ( (16 / 81 : ℚ) ^ (-3 / 4 : ℚ) + Real.logb 3 (5 / 4 : ℚ) + Real.logb 3 (4 / 5 : ℚ) ) = (27 / 8 : ℚ) := 
  sorry

end expression_value_l223_223339


namespace split_into_similar_heaps_l223_223102

noncomputable def similar_sizes (x y : ℕ) : Prop :=
  x ≤ 2 * y

theorem split_into_similar_heaps (n : ℕ) (h : n > 0) : 
  ∃ f : ℕ → ℕ, (∀ k, k < n → similar_sizes (f (k + 1)) (f k)) ∧ f (n - 1) = n := by
  sorry

end split_into_similar_heaps_l223_223102


namespace quadratic_has_distinct_real_roots_l223_223944

theorem quadratic_has_distinct_real_roots :
  ∃ (x y : ℝ), x ≠ y ∧ (x^2 - 3 * x - 1 = 0) ∧ (y^2 - 3 * y - 1 = 0) :=
by {
  sorry
}

end quadratic_has_distinct_real_roots_l223_223944


namespace total_pages_is_905_l223_223565

def history_pages : ℕ := 160
def geography_pages : ℕ := history_pages + 70
def math_pages : ℕ := (history_pages + geography_pages) / 2
def science_pages : ℕ := 2 * history_pages
def total_pages : ℕ := history_pages + geography_pages + math_pages + science_pages

theorem total_pages_is_905 : total_pages = 905 := by
  sorry

end total_pages_is_905_l223_223565


namespace totalNumberOfBalls_l223_223764

def numberOfBoxes : ℕ := 3
def numberOfBallsPerBox : ℕ := 5

theorem totalNumberOfBalls : numberOfBoxes * numberOfBallsPerBox = 15 := 
by
  sorry

end totalNumberOfBalls_l223_223764


namespace pile_splitting_l223_223115

theorem pile_splitting (single_stone_piles : ℕ) :
  ∃ (final_heap_size : ℕ), 
    (∀ heap_size ≤ single_stone_piles, heap_size > 0 → (heap_size * 2) ≥ heap_size) ∧ (final_heap_size = single_stone_piles) :=
by
  sorry

end pile_splitting_l223_223115


namespace heartsuit_sum_l223_223363

def heartsuit (x : ℝ) : ℝ := (x^2 - x + 3) / 2

theorem heartsuit_sum : heartsuit 1 + heartsuit 2 + heartsuit 3 = 8.5 := 
by 
  sorry

end heartsuit_sum_l223_223363


namespace stella_glasses_count_l223_223951

-- Definitions for the conditions
def dolls : ℕ := 3
def clocks : ℕ := 2
def price_per_doll : ℕ := 5
def price_per_clock : ℕ := 15
def price_per_glass : ℕ := 4
def total_cost : ℕ := 40
def profit : ℕ := 25

-- The proof statement
theorem stella_glasses_count (dolls clocks price_per_doll price_per_clock price_per_glass total_cost profit : ℕ) :
  (dolls * price_per_doll + clocks * price_per_clock) + profit + total_cost = total_cost + profit → 
  (dolls * price_per_doll + clocks * price_per_clock) + profit + total_cost - (dolls * price_per_doll + clocks * price_per_clock) = price_per_glass * 5 :=
sorry

end stella_glasses_count_l223_223951


namespace intersection_A_B_l223_223802

-- Definitions of the sets A and B according to the problem conditions
def A : Set ℝ := {y | ∃ x : ℝ, y = Real.sin x}
def B : Set ℝ := {y | ∃ x : ℝ, y = Real.sqrt (-x^2 + 4 * x - 3)}

-- The proof problem statement
theorem intersection_A_B :
  A ∩ B = {y | 0 ≤ y ∧ y ≤ 1} :=
by
  sorry

end intersection_A_B_l223_223802


namespace fraction_zero_condition_l223_223448

theorem fraction_zero_condition (x : ℝ) (h1 : (3 - |x|) / (x + 3) = 0) (h2 : x + 3 ≠ 0) : x = 3 :=
by
  sorry

end fraction_zero_condition_l223_223448


namespace longer_piece_length_l223_223652

-- Conditions
def total_length : ℤ := 69
def is_cuts_into_two_pieces (a b : ℤ) : Prop := a + b = total_length
def is_twice_the_length (a b : ℤ) : Prop := a = 2 * b

-- Question: What is the length of the longer piece?
theorem longer_piece_length
  (a b : ℤ) 
  (H1: is_cuts_into_two_pieces a b)
  (H2: is_twice_the_length a b) :
  a = 46 :=
sorry

end longer_piece_length_l223_223652


namespace trigonometric_identity_l223_223338

theorem trigonometric_identity :
  ∀ θ : ℝ, 
    θ = 10 * (π / 180) →
    (sin θ * sin (π / 2 - θ)) / (cos (35 * (π / 180)) ^ 2 - sin (35 * (π / 180)) ^ 2) = 1 / 2 :=
by
  sorry

end trigonometric_identity_l223_223338


namespace compare_areas_l223_223054

noncomputable def side_lengths1 : ℝ × ℝ × ℝ := (19, 19, 10)
noncomputable def side_lengths2 : ℝ × ℝ × ℝ := (19, 19, 26)

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def A : ℝ := triangle_area 19 19 10
noncomputable def B : ℝ := triangle_area 19 19 26

theorem compare_areas : A = 1 / 2 * B := by
  -- Proof omitted
  sorry

end compare_areas_l223_223054


namespace distance_between_parallel_lines_l223_223195

theorem distance_between_parallel_lines :
  let line1 := λ (x y : ℝ), 4 * x + 3 * y - 4
  let line2 := λ (x y : ℝ), 8 * x + 6 * y - 9
  let a := 4
  let b := 3
  let c1 := -4
  let c2 := - (9 / 2)
  let distance := λ (a b c1 c2 : ℝ), |c2 - c1| / Math.sqrt (a^2 + b^2)
  distance a b c1 c2 = 1 / 10 := sorry

end distance_between_parallel_lines_l223_223195


namespace find_m_for_even_function_l223_223831

def quadratic_function (m : ℝ) (x : ℝ) : ℝ :=
  m * x^2 + (m + 2) * m * x + 2

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem find_m_for_even_function :
  ∃ m : ℝ, is_even_function (quadratic_function m) ∧ m = -2 :=
by
  sorry

end find_m_for_even_function_l223_223831


namespace area_of_triangle_l223_223604

-- Definition of the hyperbola
def hyperbola (x y : ℝ) : Prop := 
  (x^2 / 4) - (y^2 / 2) = 1

-- Right focus of the hyperbola
def F : ℝ × ℝ := (Real.sqrt 6, 0)

-- Definition of the asymptotes and point P on the asymptote
def on_asymptote (P : ℝ × ℝ) : Prop := 
  P.2 = (Real.sqrt 2 / 2) * P.1 ∨ P.2 = -(Real.sqrt 2 / 2) * P.1

-- Definition of origin O
def O : ℝ × ℝ := (0, 0)

-- Condition that |PO| = |PF|
def distance_equal (P : ℝ × ℝ) : Prop :=
  Real.sqrt ((P.1 - 0)^2 + (P.2 - 0)^2) = Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2)

-- Proof problem: prove the area of △PFO is 3√2 / 4
theorem area_of_triangle (P : ℝ × ℝ) (h1 : hyperbola P.1 P.2) (h2 : on_asymptote P) (h3 : distance_equal P) :
  1 / 2 * Real.abs (P.1 * 0 - 0 * P.2) / 2 * 0 * Real.sqrt 3 / 2 = 3 * Real.sqrt 2 / 4 :=
sorry

end area_of_triangle_l223_223604


namespace missed_questions_l223_223143

-- Define variables
variables (a b c T : ℕ) (X Y Z : ℝ)
variables (h1 : a + b + c = T) 
          (h2 : 0 ≤ X ∧ X ≤ 100) 
          (h3 : 0 ≤ Y ∧ Y ≤ 100) 
          (h4 : 0 ≤ Z ∧ Z ≤ 100) 
          (h5 : 6 * (a * (100 - X) / 500 + 2 * b * (100 - Y) / 500 + 3 * c * (100 - Z) / 500) = 216)

-- Define the theorem
theorem missed_questions : 5 * (a * (100 - X) / 500 + b * (100 - Y) / 500 + c * (100 - Z) / 500) = 180 :=
by sorry

end missed_questions_l223_223143


namespace triangle_ABC_is_right_angled_l223_223007

-- Define the vectors a and b using cosine and sine functions
def vec_a : ℝ × ℝ := (Real.cos (2 * Real.pi / 3), Real.sin (2 * Real.pi / 3))
def vec_b : ℝ × ℝ := (Real.cos (Real.pi / 6), Real.sin (Real.pi / 6))

-- Define the dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Vector a and b are orthogonal if their dot product is zero
theorem triangle_ABC_is_right_angled (a := vec_a) (b := vec_b) :
  dot_product a b = 0 → 
  ∃ A B C : (ℝ × ℝ), (A = a ∧ B = b ∧ ∠(A, B) = 90) := 
by 
  sorry

end triangle_ABC_is_right_angled_l223_223007


namespace Jane_spends_240_cents_l223_223048

-- Definitions for the given conditions
def price_per_apple : ℝ := 2 / 10
def price_per_orange : ℝ := 1.50 / 5
def dozen : ℕ := 12
def cheaper_fruit_price_per_unit : ℝ := min price_per_apple price_per_orange

-- Main statement to prove
theorem Jane_spends_240_cents :
  100 * (cheaper_fruit_price_per_unit * dozen) = 240 :=
by
  sorry

end Jane_spends_240_cents_l223_223048


namespace radius_of_circumcircle_half_of_incircle_l223_223901

variables {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

def incircle (triangle : Type) : Type := sorry
def orthogonal_circle (circle : Type) (pt1 pt2 : Type) : Type := sorry
def meets_again (circle1 circle2 : Type) : Type := sorry
def radius (circle : Type) : Real := sorry
def circumcircle (triangle : Type) : Type := sorry

-- Given triangle ABC
axiom (triangle_ABC : Type) (A B C : triangle_ABC)

-- Conditions:
axiom (k : incircle triangle_ABC) -- Incircle of triangle ABC
axiom (k_a : orthogonal_circle k B C) -- Circle orthogonal to k through B and C
axiom (k_b : orthogonal_circle k A C) -- Circle orthogonal to k through A and C
axiom (k_c : orthogonal_circle k A B) -- Circle orthogonal to k through A and B
axiom (C' : meets_again k_a k_b) -- Second intersection of k_a and k_b
axiom (A' : meets_again k_b k_c) -- Second intersection of k_b and k_c
axiom (B' : meets_again k_c k_a) -- Second intersection of k_c and k_a

-- Question and its Proof
theorem radius_of_circumcircle_half_of_incircle :
  radius (circumcircle (triangle A' B' C')) = 1/2 * radius k := sorry

end radius_of_circumcircle_half_of_incircle_l223_223901


namespace tomatoes_on_each_plant_l223_223311

/-- Andy harvests all the tomatoes from 18 plants that have a certain number of tomatoes each.
    He dries half the tomatoes and turns a third of the remainder into marinara sauce. He has
    42 tomatoes left. Prove that the number of tomatoes on each plant is 7.  -/
theorem tomatoes_on_each_plant (T : ℕ) (h1 : ∀ n, n = 18 * T)
  (h2 : ∀ m, m = (18 * T) / 2)
  (h3 : ∀ k, k = m / 3)
  (h4 : ∀ final, final = m - k ∧ final = 42) : T = 7 :=
by
  sorry

end tomatoes_on_each_plant_l223_223311


namespace cubic_sum_ratio_l223_223923

theorem cubic_sum_ratio :
  ∃ (n : ℕ) (y : ℕ → ℤ),
    (∀ i, 1 ≤ i ∧ i ≤ n → -2 ≤ y i ∧ y i ≤ 2) ∧
    (∑ i in finset.range (n + 1), y i = 23) ∧
    (∑ i in finset.range (n + 1), (y i) ^ 2 = 115) ∧
    let M := finset.sup finset.range (n + 1) (λ i, ∑ i in finset.range (n + 1), (y i) ^ 3) in
    let m := finset.inf finset.range (n + 1) (λ i, ∑ i in finset.range (n + 1), (y i) ^ 3) in
    M / m = -161 / 5 :=
sorry

end cubic_sum_ratio_l223_223923


namespace find_a_l223_223581

noncomputable def polynomial (a : ℝ) : ℝ → ℝ := λ x => a * x^2 + (a - 3) * x + 1

-- This is a statement without the actual computation or proof.
theorem find_a (a : ℝ) :
  (∀ x : ℝ, polynomial a x = 0 → (∃! x, polynomial a x = 0)) ↔ a = 0 ∨ a = 1 ∨ a = 9 :=
sorry

end find_a_l223_223581


namespace find_length_of_book_l223_223655

theorem find_length_of_book (width : ℝ) (area : ℝ) (h_width : width = 3) (h_area : area = 6) :
  ∃ (length : ℝ), length = 2 :=
by
  use 2
  sorry

end find_length_of_book_l223_223655


namespace projection_matrix_is_correct_l223_223904

noncomputable theory

def projection_onto (v : ℝ × ℝ) : ℝ × ℝ → ℝ × ℝ :=
  λ u, ((u.1 * v.1 + u.2 * v.2) / (v.1 * v.1 + v.2 * v.2)) • v

def matrix_to_vector (m : matrix (fin 2) (fin 2) ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (m 0 0 * v.1 + m 0 1 * v.2, m 1 0 * v.1 + m 1 1 * v.2)

def final_matrix : matrix (fin 2) (fin 2) ℝ :=
  λ i j, if (i, j) = (0, 0) then 3 / 5 else if (i, j) = (0, 1) then 3 / 10 else if (i, j) = (1, 0) then 3 / 5 else if (i, j) = (1, 1) then 3 / 10 else 0

theorem projection_matrix_is_correct :
  let u0 := (1, 1) -- an example
  let u1 := projection_onto (4, 2) u0
  let u2 := projection_onto (2, 2) u1
  in matrix_to_vector final_matrix u0 = u2 :=
by
  sorry

end projection_matrix_is_correct_l223_223904


namespace problem1_problem2_problem3_l223_223647

-- Problem (I)
theorem problem1 (x : ℝ) (hx : x > 1) : 2 * Real.log x < x - 1/x :=
sorry

-- Problem (II)
theorem problem2 (a : ℝ) : (∀ t : ℝ, t > 0 → (1 + a / t) * Real.log (1 + t) > a) → 0 < a ∧ a ≤ 2 :=
sorry

-- Problem (III)
theorem problem3 : (9/10 : ℝ)^19 < 1 / (Real.exp 2) :=
sorry

end problem1_problem2_problem3_l223_223647


namespace unique_arrangements_mississippi_l223_223745

open scoped Nat

theorem unique_arrangements_mississippi : 
  let n := 11
  let n_M := 1
  let n_I := 4
  let n_S := 4
  let n_P := 2
  (n.fact / (n_M.fact * n_I.fact * n_S.fact * n_P.fact)) = 34650 :=
  by
  sorry

end unique_arrangements_mississippi_l223_223745


namespace bank_payment_difference_l223_223519

noncomputable def plan1_payment (P : ℝ) (r : ℝ) (m : ℕ) (t1 t2 t3 : ℕ) : ℝ :=
let A1 := P * (1 + r / m) ^ (m * t1) - (P * (1 + r / m) ^ (m * t1)) / 3
let A2 := A1 * (1 + r / m) ^ (m * (t2 - t1)) - A1 * (1 + r / m) ^ (m * (t2 - t1)) / 3
let A3 := A2 * (1 + r / m) ^ (m * (t3 - t2)) in
P * (1 + r / m) ^ (m * t1) / 3 + A1 * (1 + r / m) ^ (m * (t2 - t1)) / 3 + A3

noncomputable def plan2_payment (P : ℝ) (r : ℝ) (m : ℕ) (n : ℕ) : ℝ :=
P * (1 + r / m)^ (m * n)

noncomputable def plan_difference (P : ℝ) (r : ℝ) (m1 m2 : ℕ) (t1 t2 t3 n : ℕ) : ℝ :=
| plan2_payment P r m2 n - plan1_payment P r m1 t1 t2 t3 |

theorem bank_payment_difference :
let P := 12000 in
let r := 0.08 in
let m1 := 4 in
let m2 := 2 in
let t1 := 5 in
let t2 := 7 in
let t3 := 10 in
let n := 10 in
plan_difference P r m1 m2 t1 t2 t3 n = 5366 := by
sorry

end bank_payment_difference_l223_223519


namespace find_length_BC_l223_223045

-- Define points A, B, C in ℝ^2 (or ℝ^3 if necessary)
variables {A B C : ℝ × ℝ}

-- Given conditions
variables (AB AC : ℝ) (dot_product_BA_BC : ℝ)

-- Assume the given conditions
def conditions := (AB = 3) ∧ (AC = 5) ∧ (dot_product_BA_BC = 1)

-- Show that BC = 2√6
theorem find_length_BC (h : conditions) : ∃ (BC : ℝ), BC = 2 * real.sqrt 6 :=
sorry

end find_length_BC_l223_223045


namespace largest_number_eq_480_l223_223957

-- Definitions and conditions derived from the problem statement
variable HCF : ℕ := 40
variable factor1 : ℕ := 11
variable factor2 : ℕ := 12

-- Definition of the two numbers based on the given HCF and factors
def num1 : ℕ := HCF * factor1
def num2 : ℕ := HCF * factor2

-- Statement that the largest number is 480
theorem largest_number_eq_480 : max num1 num2 = 480 := by
  -- Proof will go here
  sorry

end largest_number_eq_480_l223_223957


namespace max_OB_length_l223_223244

noncomputable def angleAOB := Real.pi / 4  -- 45 degrees in radians
noncomputable def AB := 2

theorem max_OB_length :
  ∀ (O A B : ℝ × ℝ),
  ∠ A O B = angleAOB →
  AB = dist A B →
  (∃ OB, max (dist O B) = 2 * Real.sqrt 2) :=
begin
  sorry
end

end max_OB_length_l223_223244


namespace chessboard_marking_l223_223429

theorem chessboard_marking :
  let n := 8,
      corners := [(1,1), (1,8), (8,1), (8,8)],
      valid_positions := {p : ℕ × ℕ // p.1 ∈ (Finset.range n) ∧ p.2 ∈ (Finset.range n) ∧ p ∉ corners},
      no_row_col_conflict (pos_set : Finset (ℕ × ℕ)) :=
        ∀ p1 p2 ∈ pos_set, p1 ≠ p2 → p1.1 ≠ p2.1 ∧ p1.2 ≠ p2.2
  in Finset.card {s : Finset (ℕ × ℕ) // s.card = n ∧ no_row_col_conflict s ∧ ∀ corner ∈ corners, corner ∉ s } = 21600 := by
  sorry

end chessboard_marking_l223_223429


namespace min_value_a_plus_b_l223_223018

theorem min_value_a_plus_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : log 4 a + log 4 b ≥ 5) : a + b ≥ 64 :=
sorry

end min_value_a_plus_b_l223_223018


namespace reciprocal_of_neg_2023_l223_223981

theorem reciprocal_of_neg_2023 : 1 / (-2023) = -1 / 2023 :=
by
  sorry

end reciprocal_of_neg_2023_l223_223981


namespace permutations_mississippi_l223_223749

theorem permutations_mississippi : 
  let total_letters := 11
  let m_count := 1
  let i_count := 4
  let s_count := 4
  let p_count := 2
  (Nat.factorial total_letters / (Nat.factorial m_count * Nat.factorial i_count * Nat.factorial s_count * Nat.factorial p_count)) = 34650 := 
by
  sorry

end permutations_mississippi_l223_223749


namespace find_a1_and_d_l223_223882

variable (a : ℕ → ℤ) (d : ℤ) (a1 : ℤ) (a5 : ℤ := -1) (a8 : ℤ := 2)

def arithmetic_sequence : Prop :=
  ∀ n : ℕ, a (n+1) = a n + d

theorem find_a1_and_d
  (h : arithmetic_sequence a d)
  (h_a5 : a 5 = -1)
  (h_a8 : a 8 = 2) :
  a 1 = -5 ∧ d = 1 :=
by
  sorry

end find_a1_and_d_l223_223882


namespace find_original_price_each_stocking_l223_223153

open Real

noncomputable def original_stocking_price (total_stockings total_cost_per_stocking discounted_cost monogramming_cost total_cost : ℝ) : ℝ :=
  let stocking_cost_before_monogramming := total_cost - (total_stockings * monogramming_cost)
  let original_price := stocking_cost_before_monogramming / (total_stockings * discounted_cost)
  original_price

theorem find_original_price_each_stocking :
  original_stocking_price 9 122.22 0.9 5 1035 = 122.22 := by
  sorry

end find_original_price_each_stocking_l223_223153


namespace permutations_mississippi_l223_223752

theorem permutations_mississippi : 
  let total_letters := 11
  let m_count := 1
  let i_count := 4
  let s_count := 4
  let p_count := 2
  (Nat.factorial total_letters / (Nat.factorial m_count * Nat.factorial i_count * Nat.factorial s_count * Nat.factorial p_count)) = 34650 := 
by
  sorry

end permutations_mississippi_l223_223752


namespace zero_solution_purely_imaginary_solution_specific_solution_l223_223786

-- Define the complex number z as a function of real number m.
def z (m : ℝ) : ℂ := m * (m - 1) + (m ^ 2 + 2 * m - 3) * Complex.i

-- Prove that for m = 1, z = 0
theorem zero_solution (m : ℝ) (h1 : m = 1) : z m = 0 := 
by sorry

-- Prove that for m = 0, z is purely imaginary
theorem purely_imaginary_solution (m : ℝ) (h1 : m = 0) : ∃ b : ℝ, z m = b * Complex.i ∧ b ≠ 0 := 
by sorry

-- Prove that for m = 2, z = 2 + 5i
theorem specific_solution (m : ℝ) (h1 : m = 2) : z m = 2 + 5 * Complex.i := 
by sorry

end zero_solution_purely_imaginary_solution_specific_solution_l223_223786


namespace trapezoid_circle_tangent_radius_l223_223502

theorem trapezoid_circle_tangent_radius 
  (AB CD : ℝ) (BC DA : ℝ) (r : ℝ)
  (h_trapezoid : AB = 6 ∧ BC = 5 ∧ DA = 5 ∧ CD = 4)
  (h_circles_radii : ∀ (A B C D : ℝ), (dist A B) = 3 ∧ (dist C D) = 2)
  (h_radius_eq : r = (-(60 : ℝ) + 48 * Real.sqrt 3) / 23) :
  let k : ℕ := 60
  let m : ℕ := 48
  let n : ℕ := 3
  let p : ℕ := 23 in
  k + m + n + p = 134 := by
sorry

end trapezoid_circle_tangent_radius_l223_223502


namespace equivalence_condition_l223_223782

theorem equivalence_condition (a b c d : ℝ) (h : (a + b) / (b + c) = (c + d) / (d + a)) : 
  a = c ∨ a + b + c + d = 0 :=
sorry

end equivalence_condition_l223_223782


namespace gcd_lcm_product_l223_223357

theorem gcd_lcm_product (a b : ℕ) (h_a : a = 24) (h_b : b = 60) : 
  Nat.gcd a b * Nat.lcm a b = 1440 := by 
  rw [h_a, h_b]
  apply Nat.gcd_mul_lcm
  sorry

end gcd_lcm_product_l223_223357


namespace identity_problem_l223_223855

theorem identity_problem
  (a b : ℝ)
  (h₁ : a * b = 2)
  (h₂ : a + b = 3) :
  (a - b)^2 = 1 :=
by
  sorry

end identity_problem_l223_223855


namespace min_g_value_l223_223416

noncomputable def f (x b : ℝ) : ℝ := abs (sin x + 2 / (3 + sin x) + b)

noncomputable def g (b : ℝ) : ℝ := sup {f x b | x : ℝ}

theorem min_g_value : infimum (range g) = 3 / 4 :=
by 
sorry

end min_g_value_l223_223416


namespace find_a_l223_223583

noncomputable def polynomial (a : ℝ) : ℝ → ℝ := λ x => a * x^2 + (a - 3) * x + 1

-- This is a statement without the actual computation or proof.
theorem find_a (a : ℝ) :
  (∀ x : ℝ, polynomial a x = 0 → (∃! x, polynomial a x = 0)) ↔ a = 0 ∨ a = 1 ∨ a = 9 :=
sorry

end find_a_l223_223583


namespace arc_length_of_parametric_curve_l223_223701

noncomputable def arcLength : ℝ :=
  ∫ t in (0 : ℝ)..(Real.pi / 3), real.sqrt (9 * t^2 * (Real.cos t)^2 + 9 * t^2 * (Real.sin t)^2)

theorem arc_length_of_parametric_curve :
  arcLength = Real.pi ^ 2 / 6 :=
by
  sorry

end arc_length_of_parametric_curve_l223_223701


namespace principal_is_correct_l223_223682

-- Define the given conditions
def SI := 4016.25
def R := 9.0
def T := 5

-- Define the formula for computing the Principal
def compute_principal (si : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  (si * 100) / (r * t)

-- State the theorem to prove
theorem principal_is_correct : compute_principal SI R T = 8925 := 
  by
    sorry

end principal_is_correct_l223_223682


namespace lines_parallel_if_perpendicular_to_plane_l223_223917

variables {x y z : Type} [metric_space x] [metric_space y] [metric_space z]
open_locale euclidean_geometry

theorem lines_parallel_if_perpendicular_to_plane
  (x y z : Set Point) 
  (distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) 
  (perpendicular_xz : ∀ p ∈ x, ∀ q ∈ z, perp p q) 
  (perpendicular_yz: ∀ p ∈ y, ∀ q ∈ z, perp p q) :
  ∀ p ∈ x, ∀ q ∈ y, parallel p q :=
sorry

end lines_parallel_if_perpendicular_to_plane_l223_223917


namespace approx_nylon_cord_length_l223_223664

noncomputable def nylonCordLength (π : ℝ) : ℝ :=
  30 / π

theorem approx_nylon_cord_length (π : ℝ) (π_approx : π ≈ 3.14159) :
  nylonCordLength π ≈ 9.55 :=
by
  sorry

end approx_nylon_cord_length_l223_223664


namespace negation_proof_equivalence_l223_223970

theorem negation_proof_equivalence : 
  ¬(∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) :=
sorry

end negation_proof_equivalence_l223_223970


namespace average_percentage_popped_average_percentage_kernels_l223_223149

theorem average_percentage_popped (
  pops1 total1 pops2 total2 pops3 total3 : ℕ
) (h1 : pops1 = 60) (h2 : total1 = 75) 
  (h3 : pops2 = 42) (h4 : total2 = 50) 
  (h5 : pops3 = 82) (h6 : total3 = 100) : 
  ((pops1 : ℝ) / total1) * 100 + ((pops2 : ℝ) / total2) * 100 + ((pops3 : ℝ) / total3) * 100 = 246 := 
by
  sorry

theorem average_percentage_kernels (pops1 total1 pops2 total2 pops3 total3 : ℕ)
  (h1 : pops1 = 60) (h2 : total1 = 75)
  (h3 : pops2 = 42) (h4 : total2 = 50)
  (h5 : pops3 = 82) (h6 : total3 = 100) :
  ((
      (((pops1 : ℝ) / total1) * 100) + 
       (((pops2 : ℝ) / total2) * 100) + 
       (((pops3 : ℝ) / total3) * 100)
    ) / 3 = 82) :=
by
  sorry

end average_percentage_popped_average_percentage_kernels_l223_223149


namespace interest_calculation_years_l223_223985

theorem interest_calculation_years (P R SI : ℝ) (hP : P = 3000) (hR : R = 0.04) (hSI_cond : SI = P - 2400) :
  ∃ n : ℕ, (P * R * n = SI) ∧ n = 5 :=
by {
  existsi 5,
  split,
  { rw [hP, hR],
    simp,
    exact Eq.symm hSI_cond, },
  { refl, }
}

end interest_calculation_years_l223_223985


namespace correct_equations_l223_223239

theorem correct_equations (m n : ℕ) (h1 : n = 4 * m - 2) (h2 : n = 2 * m + 58) :
  (4 * m - 2 = 2 * m + 58 ∨ (n + 2) / 4 = (n - 58) / 2) :=
by
  sorry

end correct_equations_l223_223239


namespace factorization_l223_223346

-- Variables a and b
variable (a b : ℝ)

-- Statement to prove that ab^2 - 4a = a(b + 2)(b - 2)
theorem factorization (a b : ℝ) : a * b ^ 2 - 4 * a = a * (b + 2) * (b - 2) :=
by {
  sorry
}

end factorization_l223_223346


namespace difference_of_9_exists_difference_of_10_exists_no_difference_of_11_difference_of_12_exists_difference_of_13_exists_l223_223331

def chosen_numbers : Set ℕ := { x | 1 ≤ x ∧ x ≤ 99 }

theorem difference_of_9_exists (S : Set ℕ) (h : S ⊆ chosen_numbers) (hs : S.card = 55) :
  ∃ a b ∈ S, a ≠ b ∧ abs (a - b) = 9 := 
sorry

theorem difference_of_10_exists (S : Set ℕ) (h : S ⊆ chosen_numbers) (hs : S.card = 55) :
  ∃ a b ∈ S, a ≠ b ∧ abs (a - b) = 10 := 
sorry

theorem no_difference_of_11 (S : Set ℕ) (h : S ⊆ chosen_numbers) (hs : S.card = 55) :
  ¬ ∀ a b ∈ S, a ≠ b → abs (a - b) ≠ 11 := 
sorry

theorem difference_of_12_exists (S : Set ℕ) (h : S ⊆ chosen_numbers) (hs : S.card = 55) :
  ∃ a b ∈ S, a ≠ b ∧ abs (a - b) = 12 := 
sorry

theorem difference_of_13_exists (S : Set ℕ) (h : S ⊆ chosen_numbers) (hs : S.card = 55) :
  ∃ a b ∈ S, a ≠ b ∧ abs (a - b) = 13 := 
sorry

end difference_of_9_exists_difference_of_10_exists_no_difference_of_11_difference_of_12_exists_difference_of_13_exists_l223_223331


namespace sum_of_obtuse_angles_l223_223809

theorem sum_of_obtuse_angles (α β : ℝ) (hα : π / 2 < α ∧ α < π) (hβ : π / 2 < β ∧ β < π)
  (h1 : Real.sin α = sqrt 5 / 5) (h2 : Real.sin β = sqrt 10 / 10) :
  α + β = 7 * π / 4 :=
sorry

end sum_of_obtuse_angles_l223_223809


namespace seq_2022nd_term_eq_89_l223_223965

noncomputable def seq : ℕ → ℕ
| 2022 := 2022
| n := (n.digits 10).map (λ x, x*x) |>.sum

theorem seq_2022nd_term_eq_89 : seq 2021 = 89 := by
  -- Proving the sequence's 2022nd term is 89
  sorry

end seq_2022nd_term_eq_89_l223_223965


namespace pile_splitting_l223_223110

theorem pile_splitting (single_stone_piles : ℕ) :
  ∃ (final_heap_size : ℕ), 
    (∀ heap_size ≤ single_stone_piles, heap_size > 0 → (heap_size * 2) ≥ heap_size) ∧ (final_heap_size = single_stone_piles) :=
by
  sorry

end pile_splitting_l223_223110


namespace sum_of_digits_B_l223_223608

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).foldl (· + ·) 0

def A : ℕ := sum_of_digits (4444 ^ 4444)

def B : ℕ := sum_of_digits A

theorem sum_of_digits_B : 
  sum_of_digits B = 7 := by
    sorry

end sum_of_digits_B_l223_223608


namespace pictures_in_albums_l223_223544

theorem pictures_in_albums :
  (∀ n : ℕ, n > 0 → n ≠ 2) →
  ∀ (albums : Fin 3 → ℕ),
  ∃ (remaining_pictures : ℕ),
  (50 + 30 + 20 = 100) →
  (∃ (first_2_albums_pictures : ℕ), first_2_albums_pictures = 15) →
  (remaining_pictures = 100 - 2 * 15) →
  ((albums 0 = albums 1) → (albums 1 = albums 2)) →
  70 % 3 = 1 :=
by
  intros,
  sorry

end pictures_in_albums_l223_223544


namespace example_problem_l223_223186

variables (a b : ℕ)

def HCF (m n : ℕ) : ℕ := m.gcd n
def LCM (m n : ℕ) : ℕ := m.lcm n

theorem example_problem (hcf_ab : HCF 385 180 = 30) (a_def: a = 385) (b_def: b = 180) :
  LCM 385 180 = 2310 := 
by
  sorry

end example_problem_l223_223186


namespace quadratic_equation_with_roots_l223_223990

-- Definitions based on conditions
variables (α β : ℝ)
def sum_eq_six : Prop := α + β = 6
def abs_diff_eq_eight : Prop := abs (α - β) = 8

-- Statement of the proof that includes the question and the correct answer
theorem quadratic_equation_with_roots (h1 : sum_eq_six α β) (h2 : abs_diff_eq_eight α β) :
  ∃ (a b c : ℝ), a = 1 ∧ b = -6 ∧ c = -7 ∧ (∀ x, x^2 + b * x + c = (x - α) * (x - β)) :=
begin
  use [1, -6, -7],
  split, { refl },
  split, { refl },
  split, { refl },
  intro x,
  sorry
end

end quadratic_equation_with_roots_l223_223990


namespace graph_passes_through_fixed_point_l223_223200

-- Define the function
def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 1) + 2

-- Define the conditions
variable (a : ℝ)
variable (h1 : a > 0)
variable (h2 : a ≠ 1)

-- State the theorem
theorem graph_passes_through_fixed_point : f a 1 = 3 :=
by
  sorry

end graph_passes_through_fixed_point_l223_223200


namespace power_eq_l223_223072

open Real

theorem power_eq {x : ℝ} (h : x^3 + 4 * x = 8) : x^7 + 64 * x^2 = 128 :=
by
  sorry

end power_eq_l223_223072


namespace divisible_by_100_l223_223059

def is_positive_odd_integer (n : ℕ) : Prop := n % 2 = 1 ∧ n > 0

lemma ordered_quadruples_sum (x1 x2 x3 x4 : ℕ) (h1 : is_positive_odd_integer x1)
  (h2 : is_positive_odd_integer x2) (h3 : is_positive_odd_integer x3)
  (h4 : is_positive_odd_integer x4) : x1 + x2 + x3 + x4 = 100 ↔
  2 * ((x1 + 1) / 2 + (x2 + 1) / 2 + (x3 + 1) / 2 + (x4 + 1) / 2) - 4 = 100 :=
by
  rw [is_positive_odd_integer] at h1 h2 h3 h4
  simp_rw [mul_add, mul_one,add_sub_assoc, div_add_div_same, div_self]
  simp

theorem divisible_by_100 :
  n = (Nat.choose 51 3) →
  (n / 100 = 208.25) :=
by sorry

end divisible_by_100_l223_223059


namespace split_into_similar_piles_l223_223091

def similar_sizes (x y : ℕ) : Prop := x ≤ 2 * y ∧ y ≤ 2 * x

theorem split_into_similar_piles (n : ℕ) (h : 0 < n) :
  ∃ (piles : list ℕ), (∀ x ∈ piles, x = 1) ∧ (list.sum piles = n) ∧
                       (∀ x y ∈ piles, similar_sizes x y) := 
sorry

end split_into_similar_piles_l223_223091


namespace find_a_l223_223582

noncomputable def polynomial (a : ℝ) : ℝ → ℝ := λ x => a * x^2 + (a - 3) * x + 1

-- This is a statement without the actual computation or proof.
theorem find_a (a : ℝ) :
  (∀ x : ℝ, polynomial a x = 0 → (∃! x, polynomial a x = 0)) ↔ a = 0 ∨ a = 1 ∨ a = 9 :=
sorry

end find_a_l223_223582


namespace average_percentage_popped_average_percentage_kernels_l223_223150

theorem average_percentage_popped (
  pops1 total1 pops2 total2 pops3 total3 : ℕ
) (h1 : pops1 = 60) (h2 : total1 = 75) 
  (h3 : pops2 = 42) (h4 : total2 = 50) 
  (h5 : pops3 = 82) (h6 : total3 = 100) : 
  ((pops1 : ℝ) / total1) * 100 + ((pops2 : ℝ) / total2) * 100 + ((pops3 : ℝ) / total3) * 100 = 246 := 
by
  sorry

theorem average_percentage_kernels (pops1 total1 pops2 total2 pops3 total3 : ℕ)
  (h1 : pops1 = 60) (h2 : total1 = 75)
  (h3 : pops2 = 42) (h4 : total2 = 50)
  (h5 : pops3 = 82) (h6 : total3 = 100) :
  ((
      (((pops1 : ℝ) / total1) * 100) + 
       (((pops2 : ℝ) / total2) * 100) + 
       (((pops3 : ℝ) / total3) * 100)
    ) / 3 = 82) :=
by
  sorry

end average_percentage_popped_average_percentage_kernels_l223_223150


namespace chuck_play_area_l223_223320

-- Define the conditions for the problem in Lean
def shed_length1 : ℝ := 3
def shed_length2 : ℝ := 4
def leash_length : ℝ := 4

-- State the theorem we want to prove
theorem chuck_play_area :
  let sector_area1 := (3 / 4) * Real.pi * (leash_length ^ 2)
  let sector_area2 := (1 / 4) * Real.pi * (1 ^ 2)
  sector_area1 + sector_area2 = (49 / 4) * Real.pi := 
by
  -- The proof is omitted for brevity
  sorry

end chuck_play_area_l223_223320


namespace determine_k_l223_223832

noncomputable def polynomial : ℝ → Polynomial ℂ
| k := Polynomial.C (5*(k-1)) + Polynomial.X * Polynomial.C 9 + Polynomial.X^2 * Polynomial.C (2*(k-1)) + Polynomial.X^3

theorem determine_k (k : ℝ) (h : ∃ z : ℂ, z^2 = 5 ∧ polynomial k z = 0) :
  k = -1 ∨ k = 3 :=
sorry

end determine_k_l223_223832


namespace arrange_MISSISSIPPI_l223_223736

theorem arrange_MISSISSIPPI : 
  let total_letters := 11
  let count_I := 4
  let count_S := 4
  let count_P := 2
  let arrangements := 11.factorial / (count_I.factorial * count_S.factorial * count_P.factorial)
  in arrangements = 34650 :=
by
  have h1 : 11.factorial = 39916800 := by sorry
  have h2 : 4.factorial = 24 := by sorry
  have h3 : 2.factorial = 2 := by sorry
  have h4 : arrangements = 34650 := by sorry
  exact h4

end arrange_MISSISSIPPI_l223_223736


namespace axis_of_symmetry_l223_223191

def f (x : ℝ) : ℝ := Real.sin x + Real.sin (2 * Real.pi / 3 - x)

theorem axis_of_symmetry : ∀ x : ℝ,
  (f x = f (2 * (Real.pi / 3) - x)) ↔ x = Real.pi / 3 :=
by
  sorry

end axis_of_symmetry_l223_223191


namespace intersects_x_axis_at_one_point_l223_223589

theorem intersects_x_axis_at_one_point (a : ℝ) :
  (∃ x, ax^2 + (a-3)*x + 1 = 0) ∧ (∀ x₁ x₂, ax^2 + (a-3)*x + 1 = 0 → x₁ = x₂) ↔ (a = 0 ∨ a = 1 ∨ a = 9) := by
  sorry

end intersects_x_axis_at_one_point_l223_223589


namespace gcd_lcm_product_l223_223359

theorem gcd_lcm_product (a b : ℕ) (h_a : a = 24) (h_b : b = 60) : 
  Nat.gcd a b * Nat.lcm a b = 1440 := by 
  rw [h_a, h_b]
  apply Nat.gcd_mul_lcm
  sorry

end gcd_lcm_product_l223_223359


namespace eq_margin_l223_223864

variables (C S n : ℝ) (M : ℝ)

theorem eq_margin (h : M = 1 / n * (2 * C - S)) : M = S / (n + 2) :=
sorry

end eq_margin_l223_223864


namespace g_at_4_l223_223439

def g (x : ℝ) : ℝ := 5 * x + 6

theorem g_at_4 : g 4 = 26 :=
by
  sorry

end g_at_4_l223_223439


namespace squared_rearrangement_possible_l223_223006

structure Square :=
  (size : ℕ)
  (cells : set (ℕ × ℕ))
  (is_square : ∀ x y, (x, y) ∈ cells ↔ 0 ≤ x ∧ x < size ∧ 0 ≤ y ∧ y < size)

def is_disjoint (s1 s2 : set (ℕ × ℕ)) : Prop :=
  ∀ (x y : ℕ × ℕ), (x, y) ∈ s1 → ¬(x, y) ∈ s2

def rearranges_to_square (sqs : List (set (ℕ × ℕ))) (target : Square) : Prop :=
  (∀ sq ∈ sqs, (∃ (s : set (ℕ × ℕ)), s ⊆ sq)) ∧
  (∃ (p : ℕ → ℕ → option (ℕ × ℕ)),
    ∀ x y, target.cells (x, y) ↔ ∃ (i : ℕ) (a b : ℕ), p a b = some (x, y))

theorem squared_rearrangement_possible :
  let sq1 := Square.mk 3 {(x, y) | 0 ≤ x ∧ x < 3 ∧ 0 ≤ y ∧ y < 3} sorry,
      sq2 := Square.mk 4 {(x, y) | 0 ≤ x ∧ x < 4 ∧ 0 ≤ y ∧ y < 4} sorry,
      target_sq := Square.mk 5 {(x, y) | 0 ≤ x ∧ x < 5 ∧ 0 ≤ y ∧ y < 5} sorry,
      pieces := [{(x, y) | (x + 1, y) ∈ sq1.cells}, {(x, y) | (x, y + 3) ∈ sq1.cells},
                 {(x, y) | (x + 3, y) ∈ sq2.cells}, {(x, y) | (x, y + 4) ∈ sq2.cells}]
  in rearranges_to_square pieces target_sq :=
begin
  sorry
end

end squared_rearrangement_possible_l223_223006


namespace vector_parallel_same_direction_l223_223387

variables {V : Type*} [inner_product_space ℝ V]

theorem vector_parallel_same_direction (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) (h : ∥a + b∥ = ∥a∥ + ∥b∥) :
  (∃ k : ℝ, k > 0 ∧ a = k • b) := sorry

end vector_parallel_same_direction_l223_223387


namespace part1_part2_l223_223919

-- Define the conditions
variables (A B C D : Type)
variables [euclidean_space ℝ A] [euclidean_space ℝ B] [euclidean_space ℝ C] [euclidean_space ℝ D]

-- Assume the given conditions
variables (inside_triangle : ∀ (tri : Triangle A B C), D ∈ tri.interior)
variables (angle_condition : ∠ A D B = ∠ A C B + 90)
variables (length_condition : AC * BD = AD * BC)

-- Problem statements
-- Prove the first part
theorem part1 : ∀ (tri : Triangle A B C), (AB * CD) / (AC * BD) = sqrt(2) :=
  by
  intros,
  sorry

-- Prove the second part
theorem part2 : ∀ (tri : Triangle A B C), are_perpendicular (tangent (circumcircle (Triangle A C D)) C) (tangent (circumcircle (Triangle B C D)) C) :=
  by
  intros,
  sorry

end part1_part2_l223_223919


namespace find_first_term_l223_223221

variable {a r : ℚ}

theorem find_first_term (h1 : a / (1 - r) = 30) (h2 : a^2 / (1 - r^2) = 120) : a = 240 / 7 :=
by
  sorry

end find_first_term_l223_223221


namespace must_be_true_l223_223821

theorem must_be_true (a b c : ℝ) 
  (h1 : a < b) (h2 : b < c) 
  (f : ℝ → ℝ := λ x, |(2 : ℝ)^x - 1|) 
  (h3 : f a > f c) (h4 : f c > f b) : 
  2^a + 2^c < 2 :=
sorry

end must_be_true_l223_223821


namespace price_diff_is_correct_l223_223593

-- Define initial conditions
def initial_price : ℝ := 30
def flat_discount : ℝ := 5
def percent_discount : ℝ := 0.25
def sales_tax : ℝ := 0.10

def price_after_flat_discount (price : ℝ) : ℝ :=
  price - flat_discount

def price_after_percent_discount (price : ℝ) : ℝ :=
  price * (1 - percent_discount)

def price_after_tax (price : ℝ) : ℝ :=
  price * (1 + sales_tax)

def final_price_method1 : ℝ :=
  price_after_tax (price_after_percent_discount (price_after_flat_discount initial_price))

def final_price_method2 : ℝ :=
  price_after_tax (price_after_flat_discount (price_after_percent_discount initial_price))

def difference_in_cents : ℝ :=
  (final_price_method1 - final_price_method2) * 100

-- Lean statement to prove the final difference in cents
theorem price_diff_is_correct : difference_in_cents = 137.5 :=
  by sorry

end price_diff_is_correct_l223_223593


namespace bricks_needed_to_build_wall_l223_223424

/-
Problem:
Prove that the number of bricks, each measuring 40 cm x 11.25 cm x 6 cm, needed to build a wall of 8 m x 6 m x 22.5 cm is 4000.

Definitions:
- brick_length (cm): 40
- brick_width (cm): 11.25
- brick_height (cm): 6
- wall_length (m): 8
- wall_height (m): 6
- wall_width (cm): 22.5
-/

def brick_length : ℝ := 40
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6

def wall_length : ℝ := 8 * 100 -- converting meters to centimeters
def wall_height : ℝ := 6 * 100 -- converting meters to centimeters
def wall_width : ℝ := 22.5

def brick_volume : ℝ := brick_length * brick_width * brick_height
def wall_volume : ℝ := wall_length * wall_height * wall_width

def number_of_bricks : ℝ := wall_volume / brick_volume

theorem bricks_needed_to_build_wall : number_of_bricks = 4000 :=
by
  -- Just a placeholder; the actual proof would go here
  sorry

end bricks_needed_to_build_wall_l223_223424


namespace problem1_problem2_problem3_l223_223319

theorem problem1 : ((-1: ℝ) ^ 2020) * ((π - 2) ^ 0) - |(-5: ℝ)| - ((-1 / 2) ^ (-3)) = 4 :=
by
  sorry

theorem problem2 (a : ℝ) : (-2 * a^2)^3 + 2 * a^2 * a^4 - a^8 / a^2 = -7 * a^6 :=
by
  sorry

theorem problem3 (x y : ℝ) : x * (x + 2 * y) - (y - 3 * x) * (x + y) = 4 * x^2 - y^2 + 4 * x * y :=
by
  sorry

end problem1_problem2_problem3_l223_223319


namespace tan_phi_eq_l223_223873

theorem tan_phi_eq (β : ℝ) (cot_half_beta_eq : cot (β/2) = real.cbrt 3) :
  ∃ φ : ℝ, (φ = atan (tan (atan ((real.cbrt 3 ^ 2 - 1) / (4 * real.cbrt 3 - (real.cbrt 3 ^ 2 - 1) ^ 2)))) 
      ∧ tan φ = (real.cbrt 3 ^ 2 - 1) / (2 * real.cbrt 3 ^ 2 + 2)) :=
by
  sorry

end tan_phi_eq_l223_223873


namespace negative_root_condition_l223_223596

theorem negative_root_condition (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0) ↔ (a ≤ 1) :=
begin
  sorry,
end

end negative_root_condition_l223_223596


namespace solve_eq_l223_223579

theorem solve_eq (x a b : ℝ) (h₁ : x^2 + 10 * x = 34) (h₂ : a = 59) (h₃ : b = 5) :
  a + b = 64 :=
by {
  -- insert proof here, eventually leading to a + b = 64
  sorry
}

end solve_eq_l223_223579


namespace B_is_left_of_A_l223_223636

-- Define the coordinates of points A and B
def A_coord : ℚ := 5 / 8
def B_coord : ℚ := 8 / 13

-- The statement we want to prove: B is to the left of A
theorem B_is_left_of_A : B_coord < A_coord :=
  by {
    sorry
  }

end B_is_left_of_A_l223_223636


namespace maximize_ratio_l223_223381

variables (P Q : Point) (π : Plane)

-- Assume point P lies on plane π and point Q lies outside plane π
axiom P_on_plane (hπ : (P : Point) ∈ (π : Plane)) : True
axiom Q_outside_plane (hπ : (Q : Point) ∉ (π : Plane)) : True

-- Define a function that returns the ratio given points Q, P, R
noncomputable def ratio (Q P R : Point) : ℝ :=
  (dist Q P + dist P R) / dist Q R

-- The main statement to prove
theorem maximize_ratio : ∃ R : Point, (R ∈ (π : Plane)) → ratio Q P R = sqrt 2 :=
sorry

end maximize_ratio_l223_223381


namespace total_pages_is_905_l223_223566

def history_pages : ℕ := 160
def geography_pages : ℕ := history_pages + 70
def math_pages : ℕ := (history_pages + geography_pages) / 2
def science_pages : ℕ := 2 * history_pages
def total_pages : ℕ := history_pages + geography_pages + math_pages + science_pages

theorem total_pages_is_905 : total_pages = 905 := by
  sorry

end total_pages_is_905_l223_223566


namespace area_ratio_of_equilateral_triangles_l223_223614

/-- 
We have a fixed amount of fencing used to enclose three small congruent equilateral triangles.
This fencing is then reused to enclose one larger equilateral triangle.
We want to prove that the ratio of the total area of the three small triangles to the area of the large triangle is 1/3.
-/
theorem area_ratio_of_equilateral_triangles (s : ℝ) (h : 0 < s) : 
  let area := λ a : ℝ, (Real.sqrt 3 / 4) * a * a in
  (3 * area s) / (area (3 * s)) = 1 / 3 :=
by
  sorry

end area_ratio_of_equilateral_triangles_l223_223614


namespace pile_splitting_l223_223109

theorem pile_splitting (single_stone_piles : ℕ) :
  ∃ (final_heap_size : ℕ), 
    (∀ heap_size ≤ single_stone_piles, heap_size > 0 → (heap_size * 2) ≥ heap_size) ∧ (final_heap_size = single_stone_piles) :=
by
  sorry

end pile_splitting_l223_223109


namespace car_A_catches_up_l223_223703

variables (v_A v_B t_A : ℝ)

-- The conditions of the problem
def distance : ℝ := 300
def time_car_B : ℝ := t_A + 2
def distance_eq_A : Prop := distance = v_A * t_A
def distance_eq_B : Prop := distance = v_B * time_car_B

-- The final proof problem: Car A catches up with car B 150 kilometers away from city B.
theorem car_A_catches_up (t_A > 0) (v_A > 0) (v_B > 0) :
  distance_eq_A ∧ distance_eq_B → 
  ∃ d : ℝ, d = 150 := 
sorry

end car_A_catches_up_l223_223703


namespace tangents_to_curve_l223_223027

theorem tangents_to_curve (a b : ℝ) :
  let curve := λ x : ℝ, x^2 - 2 * x in
  (∃ t : ℝ, curve t = b ∧ (2 * t - 2) * (a - t) + t^2 - 2 * t = b) →
  (a = 3 ∧ b = 0) :=
by
  sorry

end tangents_to_curve_l223_223027


namespace complement_of_A_in_I_l223_223001

def I : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 6, 7}
def C_I_A : Set ℕ := {1, 3, 5}

theorem complement_of_A_in_I :
  (I \ A) = C_I_A := by
  sorry

end complement_of_A_in_I_l223_223001


namespace minimum_distance_general_C1_equation_rectangular_C2_equation_l223_223035

-- Define the parametric equations of C_1
def C1_parametric (θ : ℝ) : ℝ × ℝ :=
  (sqrt 3 * Real.cos θ, Real.sin θ)

-- Define the general equation of C_1
def C1_equation (x y : ℝ) : Prop :=
  (x^2 / 3) + y^2 = 1

-- Define the polar equation of C_2
def C2_polar (ρ θ : ℝ) : Prop :=
  ρ * Real.sin (θ + π / 4) = 2 * sqrt 2

-- Define the rectangular coordinate equation of C_2
def C2_equation (x y : ℝ) : Prop :=
  x + y = 4

-- Define the points P and Q
def is_on_C1 (P : ℝ × ℝ) : Prop := C1_equation P.1 P.2
def is_on_C2 (Q : ℝ × ℝ) : Prop := C2_equation Q.1 Q.2

-- Define the distance |PQ|
def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- State the theorem that we need to prove

theorem minimum_distance (P Q : ℝ × ℝ) :
  is_on_C1 P → is_on_C2 Q → distance P Q = sqrt 2 → P = (3 / 2, 1 / 2) :=
sorry

-- State the theorem for the general equation of C_1
theorem general_C1_equation (θ : ℝ) :
  let (x, y) := C1_parametric θ in C1_equation x y :=
sorry

-- State the theorem for the rectangular coordinate equation of C_2
theorem rectangular_C2_equation (ρ θ : ℝ) :
  C2_polar ρ θ → C2_equation (ρ * Real.cos θ) (ρ * Real.sin θ) :=
sorry

end minimum_distance_general_C1_equation_rectangular_C2_equation_l223_223035


namespace range_of_m_l223_223390

open Set Real

noncomputable def A := {x : ℝ | x^2 - 2 * x - 3 < 0}
noncomputable def B (m : ℝ) := {x : ℝ | -1 < x ∧ x < m}

theorem range_of_m (m : ℝ) : 
  (∀ x, x ∈ A → x ∈ B m) ∧ (∃ x, x ∈ B m ∧ x ∉ A) → 3 < m :=
by sorry

end range_of_m_l223_223390


namespace magic8_prob_l223_223895

open BigOperators

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  binomial n k * p^k * (1 - p)^(n - k)

theorem magic8_prob :
  binomial_prob 7 4 (1 / 3) = 280 / 2187 :=
by
  -- Computations omitted for brevity
  sorry

end magic8_prob_l223_223895


namespace parallelogram_area_perimeter_impossible_l223_223294

theorem parallelogram_area_perimeter_impossible (a b h : ℕ) (ha : 0 < a) (hb : 0 < b) (hh : 0 < h)
    (A : ℕ := b * h) (P : ℕ := 2 * a + 2 * b) :
    (A + P + 6) ≠ 102 := by
  sorry

end parallelogram_area_perimeter_impossible_l223_223294


namespace distance_pointA_pointB_l223_223622

-- Definitions of the points A and B
def pointA : ℝ × ℝ × ℝ := (2, -3, 1)
def pointB : ℝ × ℝ × ℝ := (-2, 4, -2)

-- The theorem stating the distance between A and B
theorem distance_pointA_pointB : 
  let d := dist pointA pointB in
  d = Real.sqrt 74 :=
by
  sorry

end distance_pointA_pointB_l223_223622


namespace _l223_223193

noncomputable theorem trapezoid_slopes (W Z : ℤ × ℤ) (conditions : 
  W = (50, 200) ∧ 
  Z = (52, 207) ∧ 
  (¬ ∃ x, W.1 = x ∨ W.2 = x) ∧ 
  (¬ ∃ y, Z.1 = y ∨ Z.2 = y)) : 
  ∃ (p q : ℕ), (Nat.gcd p q = 1) ∧ 
  (  ∑ (m : ℚ) in {m : ℚ | slope_possible W Z m}, abs m = p / q ) ∧ 
  (p + q = 55) :=
sorry

def slope_possible (W Z : ℤ × ℤ) (m : ℚ) : Prop :=
exists (a b : ℤ), 
(a - W.1) * (a - W.1) + (b - W.2) * (b - W.2) = 53 ∧ 
(m = (b - W.2) / (a - W.1))

end _l223_223193


namespace percentage_crayons_lost_l223_223151

theorem percentage_crayons_lost (original_crayons : ℕ) (remaining_crayons : ℕ) (lost_percentage : ℝ) :
  original_crayons = 479 → remaining_crayons = 134 → lost_percentage ≈ 72.03 → 
  let lost_crayons := original_crayons - remaining_crayons in
  let percentage_lost := (lost_crayons / original_crayons.toReal) * 100 in
  percentage_lost ≈ lost_percentage :=
by
  intros h1 h2 h3
  rw [h1, h2]
  let lost_crayons := 479 - 134
  let percentage_lost := (lost_crayons.toReal / 479) * 100
  have h4 : percentage_lost = 72.03 := sorry
  exact h4.symm.trans h3

end percentage_crayons_lost_l223_223151


namespace triangle_division_ratio_l223_223145

theorem triangle_division_ratio
  (A B C A1 B1 C1 : Type*)
  (hA1BC : A1 ∈ lineSegment B C)
  (hB1CA : B1 ∈ lineSegment C A)
  (hC1AB : C1 ∈ lineSegment A B)
  (A2 B2 C2 : Type*)
  (hA2 : midpoint A1 B1 C1 B2)
  (hB2 : midpoint B1 A1 C1 C2)
  (hC2 : midpoint C1 A1 B1)
  (h_parallel_A1A2_AB : parallel (lineSegment A1 A2) (lineSegment A B))
  (h_parallel_B1B2_BC : parallel (lineSegment B1 B2) (lineSegment B C))
  (h_parallel_C1C2_CA : parallel (lineSegment C1 C2) (lineSegment C A)) :
  ratio (lineSegment B A1) (lineSegment A1 C) = 1 / 2 ∧
  ratio (lineSegment C B1) (lineSegment B1 A) = 1 / 2 ∧
  ratio (lineSegment A C1) (lineSegment C1 B) = 1 / 2 := 
sorry

end triangle_division_ratio_l223_223145


namespace count_valid_n_values_l223_223318

def is_perfect_square (x : ℕ) : Prop :=
  ∃ k : ℕ, k * k = x

def num_valid_n : ℕ :=
  (Finset.range 26).count (λ n, is_perfect_square (n * (25 - n)))

theorem count_valid_n_values : num_valid_n = 2 := by
  sorry

end count_valid_n_values_l223_223318


namespace lawn_mowing_l223_223135

theorem lawn_mowing
  (mary_rate : ℝ) (mary_time : ℝ) 
  (tom_rate : ℝ) (tom_time : ℝ)
  (joint_time : ℝ)
  (tom_alone_time : ℝ) :
mary_rate = 1 / 3 → 
tom_rate = 1 / 4 → 
tom_alone_time = 1 → 
joint_time = 2 → 
mary_time = 1 / mary_rate → 
tom_time = 1 / tom_rate → 
(mary_rate * joint_time + tom_rate * (tom_alone_time + joint_time)) ≥ 1 →
∃ remain_time : ℝ, remain_time = 0 :=
begin
  intros mary_rate_eq mary_time_eq tom_rate_eq tom_alone_time_eq joint_time_eq mary_full_time_eq tom_full_time_eq mowing_done,
  use 0,
  sorry -- Proof is not provided as per the instructions.
end

end lawn_mowing_l223_223135


namespace vertex_of_parabola_l223_223575

theorem vertex_of_parabola : ∃ h k : ℝ, ∀ x : ℝ, y = (x - 1)^2 - 2 ∧ (h, k) = (1, -2) :=
begin
  -- condition: the equation of the parabola
  -- question: prove the vertex coordinates are (1, -2)
  sorry
end

end vertex_of_parabola_l223_223575


namespace probability_ace_king_queen_same_suit_l223_223613

theorem probability_ace_king_queen_same_suit :
  let total_probability := (1 : ℝ) / 52 * (1 : ℝ) / 51 * (1 : ℝ) / 50
  total_probability = (1 : ℝ) / 132600 :=
by
  sorry

end probability_ace_king_queen_same_suit_l223_223613


namespace quadratic_eq_has_real_solutions_l223_223025

theorem quadratic_eq_has_real_solutions (m : ℝ) (h : m * x ^ 2 + 2 * x + 1 = 0) :
  (∃ (x : ℝ), m * x ^ 2 + 2 * x + 1 = 0) → (m ≤ 1 ∧ m ≠ 0) :=
begin
  sorry
end

end quadratic_eq_has_real_solutions_l223_223025


namespace number_of_solutions_to_eq_count_number_of_solutions_to_eq_l223_223207

theorem number_of_solutions_to_eq {x y : ℤ} (h : 1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 7) :
  (x, y) = (-42, 6) ∨ (x, y) = (6, -42) ∨ (x, y) = (8, 56) ∨ (x, y) = (14, 14) ∨ (x, y) = (56, 8) :=
begin
  -- proof goes here
  sorry
end

theorem count_number_of_solutions_to_eq :
  {p : ℤ × ℤ // 1 / (p.fst : ℚ) + 1 / (p.snd : ℚ) = 1 / 7}.to_finset.card = 5 :=
begin
  -- proof goes here
  sorry
end

end number_of_solutions_to_eq_count_number_of_solutions_to_eq_l223_223207


namespace focus_of_parabola_l223_223724

theorem focus_of_parabola :
  (∀ y : ℝ, x = (1 / 4) * y^2) → (focus = (-1, 0)) := by
  sorry

end focus_of_parabola_l223_223724


namespace frog_return_central_circle_ways_l223_223168

def frog_jumps : ℕ := 5
def possible_ways : ℕ := 24

theorem frog_return_central_circle_ways 
  (h : ∃ n : ℕ, n = frog_jumps ∧ n = 5) 
  (adjacent_jump : ∀ n : ℕ, n = 17) :
  possible_ways = 24 :=
begin
  sorry
end

end frog_return_central_circle_ways_l223_223168


namespace count_complex_numbers_l223_223513

noncomputable def f (z : ℂ) : ℂ := z^2 - 2*Complex.I*z + 2

theorem count_complex_numbers :
  let S := { z : ℂ | (∃ (a b : ℤ), f(z) = a + b * Complex.I ∧ (|a| ≤ 5) ∧ (|b| ≤ 5)) ∧ (z.im > 0) } in
  S.to_finset.card = 110 :=
by
  sorry

end count_complex_numbers_l223_223513


namespace simplify_complex_fraction_l223_223946

noncomputable def simplify_fraction : ℂ :=
  (8 - 15 * Complex.i) / (3 + 4 * Complex.i)

theorem simplify_complex_fraction :
  simplify_fraction = - (36 / 25 : ℝ) - (77 / 25) * Complex.i :=
by
  sorry

end simplify_complex_fraction_l223_223946


namespace area_of_pentagon_ABCDE_l223_223962

open Real Set Polynomial

-- Define the shapes and conditions
noncomputable def pentagon_ABCDE : List (Point ℝ) :=
[⟨0, 0⟩, ⟨1, 0⟩, ⟨2 * cos (2 * π / 3), 2 * sin (2 * π / 3)⟩, ⟨3 * cos (4 * π / 3), 3 * sin (4 * π / 3)⟩, ⟨0, sin (2 * π / 3)⟩]

-- Prove the area of the pentagon
theorem area_of_pentagon_ABCDE : 
  Polygon.area pentagon_ABCDE = 5 * sqrt 3 :=
by
  sorry

end area_of_pentagon_ABCDE_l223_223962


namespace cos_angle_MPN_l223_223044

-- Define the conditions as hypotheses
variables {A B C P M N : Type} [InnerProductSpace A] 

-- Given conditions
variables (AB AC : ℝ) (angleBAC : Real.Angle)
variables (B M N : A) -- assumption on medians B and N
variable (h1 : AB = 2)
variable (h2 : AC = 3)
variable (h3 : angleBAC = Real.Angle.pi / 3)
variable (median_intersection : InnerProductSpace.median_intersection B P M = N)

-- Define the theorem statement
theorem cos_angle_MPN :
  cos (angle BETWEEN P M N) = -2 * sqrt 247 / 247 :=
sorry

end cos_angle_MPN_l223_223044


namespace greatest_value_y_l223_223853

theorem greatest_value_y (y : ℝ) (hy : 11 = y^2 + 1/y^2) : y + 1/y ≤ Real.sqrt 13 :=
sorry

end greatest_value_y_l223_223853


namespace problem_statement_l223_223767

noncomputable def given_expression (x y z : ℝ) : ℝ :=
  (45 + (23 / 89) * Real.sin x) * (4 * y^2 - 7 * z^3)

theorem problem_statement : given_expression (Real.pi / 6) 3 (-2) = 4186 := by
  sorry

end problem_statement_l223_223767


namespace average_primes_4_to_15_l223_223268

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : list ℕ :=
  (list.range' (a + 1) (b - a)).filter is_prime

theorem average_primes_4_to_15 :
  primes_between 4 15 = [5, 7, 11, 13] →
  (list.sum (primes_between 4 15) / (primes_between 4 15).length) = 9 :=
begin
  intro h,
  rw h, 
  simp,
  exact dec_trivial
end

end average_primes_4_to_15_l223_223268


namespace cubed_sums_of_roots_l223_223964

theorem cubed_sums_of_roots :
  let δ := real.cbrt 17
  let ε := real.cbrt 67
  let ζ := real.cbrt 97
  (∃ u v w : ℝ, (u ≠ v ∧ u ≠ w ∧ v ≠ w) ∧ 
                 (u - δ) * (u - ε) * (u - ζ) = 1/2 ∧ 
                 (v - δ) * (v - ε) * (v - ζ) = 1/2 ∧ 
                 (w - δ) * (w - ε) * (w - ζ) = 1/2 ∧
                 (u^3 + v^3 + w^3 = 184.5)) := 
sorry

end cubed_sums_of_roots_l223_223964


namespace contrapositive_a_eq_b_imp_a_sq_eq_b_sq_l223_223772

theorem contrapositive_a_eq_b_imp_a_sq_eq_b_sq (a b : ℝ) :
  (a = b → a^2 = b^2) ↔ (a^2 ≠ b^2 → a ≠ b) :=
by
  sorry

end contrapositive_a_eq_b_imp_a_sq_eq_b_sq_l223_223772


namespace feet_per_inch_of_model_l223_223185

theorem feet_per_inch_of_model 
  (height_tower : ℝ)
  (height_model : ℝ)
  (height_tower_eq : height_tower = 984)
  (height_model_eq : height_model = 6)
  : (height_tower / height_model) = 164 :=
by
  -- Assume the proof here
  sorry

end feet_per_inch_of_model_l223_223185


namespace no_solution_for_system_l223_223530

theorem no_solution_for_system (x y z : ℝ) 
  (h1 : |x| < |y - z|) 
  (h2 : |y| < |z - x|) 
  (h3 : |z| < |x - y|) : 
  false :=
sorry

end no_solution_for_system_l223_223530


namespace g_not_extreme_value_on_interval_l223_223829

theorem g_not_extreme_value_on_interval 
  (a b : ℝ) 
  (h1 : a < b) 
  (f g : ℝ → ℝ) 
  (h2 : ∀ x ∈ set.Icc a b, f x = Real.sin x) 
  (h3 : ∀ x ∈ set.Icc a b, g x = Real.cos x) 
  (h4 : g a * g b < 0) 
  : ¬ (∀ c ∈ set.Icc a b, ∃ δ > 0, ∀ d ∈ set.Icc (c - δ) (c + δ), g c ≥ g d ∨ g c ≤ g d) :=
sorry

end g_not_extreme_value_on_interval_l223_223829


namespace num_true_propositions_is_zero_l223_223197

theorem num_true_propositions_is_zero :
  (∀ (a : ℝ), a ≥ 0 → sqrt a = ∃ (x : ℝ), x * x = a) ∧
  (∀ (x : ℝ), x ^ 3 = 0 → x = 0) ∧
  (∀ (a b : ℝ), a * b = 0 → a = 0 ∧ b = 0) ∧
  (∀ (A B : ℝ × ℝ), A = (-1, -2) ∧ (B.1 - A.1 = 5 ∨ B.1 - A.1 = -5) ∧ B.2 = -2 → B = (4, -2)) →
  0 = 0 := sorry

end num_true_propositions_is_zero_l223_223197


namespace tan_alpha_value_sin_alpha_plus_pi_six_value_l223_223394

theorem tan_alpha_value (α : ℝ) (h : 0 < α ∧ α < π / 2) (h_tan : tan (π / 4 + α) = 3) :
  tan α = 1 / 2 := 
sorry

theorem sin_alpha_plus_pi_six_value (α : ℝ) (h : 0 < α ∧ α < π / 2) (h_tan : tan (π / 4 + α) = 3) :
  sin (α + π / 6) = (sqrt 15 + 2 * sqrt 5) / 10 := 
sorry

end tan_alpha_value_sin_alpha_plus_pi_six_value_l223_223394


namespace average_speed_last_segment_l223_223896

theorem average_speed_last_segment
  (total_distance : ℕ)
  (total_time : ℕ)
  (speed1 speed2 speed3 : ℕ)
  (last_segment_time : ℕ)
  (average_speed_total : ℕ) :
  total_distance = 180 →
  total_time = 180 →
  speed1 = 40 →
  speed2 = 50 →
  speed3 = 60 →
  average_speed_total = 60 →
  last_segment_time = 45 →
  ∃ (speed4 : ℕ), speed4 = 90 :=
by sorry

end average_speed_last_segment_l223_223896


namespace unique_arrangements_mississippi_l223_223748

open scoped Nat

theorem unique_arrangements_mississippi : 
  let n := 11
  let n_M := 1
  let n_I := 4
  let n_S := 4
  let n_P := 2
  (n.fact / (n_M.fact * n_I.fact * n_S.fact * n_P.fact)) = 34650 :=
  by
  sorry

end unique_arrangements_mississippi_l223_223748


namespace translate_point_l223_223471

theorem translate_point :
  ∀ (M : ℝ × ℝ), M = (-2, 3) →
  (let M' := (M.1, M.2 - 3) in (M'.1 + 1, M'.2) = (-1, 0)) :=
by
  intros M h
  cases h
  simp [h]
  rfl

end translate_point_l223_223471


namespace domain_of_function_l223_223623

noncomputable def domain_function : set ℝ := {x : ℝ | x ≠ 64}

theorem domain_of_function :
  ∀ x : ℝ, x ∈ domain_function ↔ x ∈ (set.Ioo (-(real.infinity : ℝ)) 64 ∪ set.Ioo 64 (real.infinity : ℝ)) :=
begin
  sorry

end domain_of_function_l223_223623


namespace midpoint_BpGp_l223_223599

structure Point2D where
  x : ℝ
  y : ℝ

def translate (p : Point2D) (v : Point2D) : Point2D :=
  Point2D.mk (p.x + v.x) (p.y + v.y)

def midpoint (p1 p2 : Point2D) : Point2D :=
  Point2D.mk ((p1.x + p2.x) / 2) ((p1.y + p2.y) / 2)

theorem midpoint_BpGp :
  let B := Point2D.mk 1 3
  let G := Point2D.mk 5 3
  let v := Point2D.mk 3 (-4)
  let B' := translate B v
  let G' := translate G v
  let M := midpoint B G
  let M' := translate M v
  M' = Point2D.mk 6 (-1) :=
by
  sorry

end midpoint_BpGp_l223_223599


namespace sin_max_value_l223_223910

open Real

theorem sin_max_value (a b : ℝ) (h : sin (a + b) = sin a + sin b) :
  sin a ≤ 1 :=
by
  sorry

end sin_max_value_l223_223910


namespace power_of_two_l223_223065

theorem power_of_two (b m n : ℕ) (hb : b > 1) (hmn : m ≠ n) 
  (hprime_divisors : ∀ p : ℕ, p.Prime → (p ∣ b ^ m - 1 ↔ p ∣ b ^ n - 1)) : 
  ∃ k : ℕ, b + 1 = 2 ^ k :=
by
  sorry

end power_of_two_l223_223065


namespace orchid_bushes_after_planting_l223_223992

def total_orchid_bushes (current_orchids new_orchids : Nat) : Nat :=
  current_orchids + new_orchids

theorem orchid_bushes_after_planting :
  ∀ (current_orchids new_orchids : Nat), current_orchids = 22 → new_orchids = 13 → total_orchid_bushes current_orchids new_orchids = 35 :=
by
  intros current_orchids new_orchids h_current h_new
  rw [h_current, h_new]
  exact rfl

end orchid_bushes_after_planting_l223_223992


namespace sequence_returns_one_l223_223382

noncomputable def sequence (d : ℕ) : ℕ → ℕ
| 0       := 1
| (n + 1) := if sequence n % 2 = 0 then sequence n / 2 else sequence n + d

theorem sequence_returns_one (d : ℕ) :
  (∃ n > 0, sequence d n = 1) ↔ d % 2 = 1 :=
begin
  sorry
end

end sequence_returns_one_l223_223382


namespace alpha_plus_beta_l223_223994

theorem alpha_plus_beta (α β : ℝ) (h : ∀ x, (x - α) / (x + β) = (x^2 - 116 * x + 2783) / (x^2 + 99 * x - 4080)) 
: α + β = 115 := 
sorry

end alpha_plus_beta_l223_223994


namespace reciprocal_sum_l223_223405

variable {x y z a b c : ℝ}

-- The function statement where we want to show the equivalence.
theorem reciprocal_sum (h1 : x ≠ y) (h2 : x ≠ z) (h3 : y ≠ z)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hxy : (x * y) / (x - y) = a)
  (hxz : (x * z) / (x - z) = b)
  (hyz : (y * z) / (y - z) = c) :
  (1/x + 1/y + 1/z) = ((1/a + 1/b + 1/c) / 2) :=
sorry

end reciprocal_sum_l223_223405


namespace split_stones_l223_223081

theorem split_stones (n : ℕ) :
  ∃ (heaps : list ℕ), (∀ h ∈ heaps, 1 ≤ h ∧ h ≤ n) ∧ (∀ i j, i ≠ j → (i < heaps.length ∧ j < heaps.length → heaps.nth i ≤ 2 * heaps.nth j)) :=
sorry

end split_stones_l223_223081


namespace factors_of_210_l223_223012

theorem factors_of_210 : 
  (∃ p1 p2 p3 p4 : ℕ, 210 = p1 * p2 * p3 * p4 ∧ 
                      Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ Nat.Prime p4 ∧
                      {p1, p2, p3, p4} = {2, 3, 5, 7}) →
  ∃ n : ℕ, n = 16 ∧ ∀ p ∈ {2, 3, 5, 7}, (∃ k : ℕ, 0 ≤ k ∧ k ≤ 1) ∧ 
             n = (1 + (if p = 2 then 1 else 0)) *
                (1 + (if p = 3 then 1 else 0)) *
                (1 + (if p = 5 then 1 else 0)) *
                (1 + (if p = 7 then 1 else 0)) :=
by
  simp
  sorry

end factors_of_210_l223_223012


namespace problem_statement_l223_223852

noncomputable def expansion := (3 * (x : ℝ) + Real.sqrt 7) ^ 4
def a0 := expansion.eval 0
def a1 := expansion.eval (1 : ℝ) - a0
def a2 := expansion.eval (2 : ℝ) - a0 - a1 * 2
def a3 := expansion.eval (-1 : ℝ) - a0 + a1 - a2
def a4 := expansion.eval (-2 : ℝ) - a0 - a1 * -2 - a2 * 4 - a3 * -8

theorem problem_statement :
  (a0 + a2 + a4)^2 - (a1 + a3)^2 = 16 :=
by
  sorry

end problem_statement_l223_223852


namespace constant_t_exists_l223_223232

theorem constant_t_exists :
  ∃ (c : ℝ), (c = 1 / 8) →
  (∀ (A B : ℝ × ℝ), 
    let m_A := -1 / (2 * A.1),
        m_B := -1 / (2 * B.1),
        AC := (A.1 - 0)^2 + (A.2 - 1 / 8)^2,
        BC := (B.1 - 0)^2 + (B.2 - 1 / 8)^2,
        t := 1 / AC + 1 / BC
    in (A.2 = A.1^2) ∧ (B.2 = B.1^2) →
    (m_A * (2 * A.1) = -1) ∨ (m_B * (2 * B.1) = -1) →
    t = 8) :=
sorry

end constant_t_exists_l223_223232


namespace find_a_of_common_point_l223_223880

theorem find_a_of_common_point (
  (x₁ y₁ : ℝ) (t : ℝ) (h₁ : x₁ = t + 1) (h₂ : y₁ = 1 - 2t)
  (x₂ y₂ : ℝ) (θ : ℝ) (a : ℝ) (ha_pos : a > 0)
  (h₃ : x₂ = a * Real.sin θ) (h₄ : y₂ = 3 * Real.cos θ) 
  (common_point_on_x_axis : y₁ = 0 ∧ y₂ = 0 ∧ x₁ = x₂)
) : a = 1 :=
by
  sorry

end find_a_of_common_point_l223_223880


namespace curves_are_externally_tangent_l223_223816

-- Definitions for curves in polar coordinates
def C1_polar (ρ θ : ℝ) : Prop := ρ = 2 * cos θ
def C2_polar (ρ θ : ℝ) : Prop := ρ^2 - 2 * sqrt 3 * ρ * sin θ + 2 = 0

-- Definitions for curves in Cartesian coordinates
def C1_cartesian (x y : ℝ) : Prop := x^2 + y^2 - 2 * x = 0
def C2_cartesian (x y : ℝ) : Prop := x^2 + y^2 - 2 * sqrt 3 * y + 2 = 0

-- Definitions for centers and radii
def center_C1 : ℝ × ℝ := (1, 0)
def radius_C1 : ℝ := 1

def center_C2 : ℝ × ℝ := (0, sqrt 3)
def radius_C2 : ℝ := 1

-- Definition for distance between two points in Cartesian coordinates
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Definition for externally tangent condition
def externally_tangent (center1 center2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  dist center1 center2 = r1 + r2

-- Statement to prove
theorem curves_are_externally_tangent : 
  externally_tangent center_C1 center_C2 radius_C1 radius_C2 :=
sorry

end curves_are_externally_tangent_l223_223816


namespace ellipse_focus_d_value_l223_223308

theorem ellipse_focus_d_value (d : ℝ) 
  (h_ellipse_first_quadrant : ∀ (x y : ℝ), (x > 0) ∧ (y > 0) ∧ 
                                      (b * b * (x - h) * (x - h) + a * a * (y - k) * (y - k) = a * a * b * b))
  (h_tangent_x : ∃ (c : ℝ), (tangent_x : c = h ∧ c = 0)) 
  (h_tangent_y : ∃ (c : ℝ), (tangent_y : c = k ∧ c = 0)) 
  (h_focus1 : (4, 10)) 
  (h_focus2 : (d, 10)) : 
  d = 25 :=
sorry

end ellipse_focus_d_value_l223_223308


namespace triangle_area_ratio_l223_223715

/-- Given an equilateral triangle ABC, extend side AB by twice its length to reach point D,
    extend side BC by thrice its length to reach point E, and extend side CA by four times its length to reach point F.
    Prove that the ratio of the area of triangle DEF to the area of triangle ABC is 64:1. -/
theorem triangle_area_ratio :
  ∀ (A B C D E F : Point),
  is_equilateral_triangle A B C →
  segment_length (A, B) = segment_length (B, C) →
  segment_length (B, C) = segment_length (C, A) →
  segment_length (A, D) = 3 * segment_length (A, B) →
  segment_length (B, E) = 4 * segment_length (B, C) →
  segment_length (C, F) = 5 * segment_length (C, A) →
  let area_ABC := triangle_area A B C in
  let area_DEF := triangle_area D E F in
  area_DEF / area_ABC = 64 :=
by
  sorry

end triangle_area_ratio_l223_223715


namespace max_golden_terms_l223_223458

-- Define the condition for a term being golden
def is_golden_term (a b : ℕ) : Prop :=
  a ≠ 0 ∧ b % a = 0

-- Define the function to count golden terms in a list L.
def count_golden_terms : List ℕ → ℕ
  | []           => 0
  | [_]          => 0
  | (a :: (b :: l)) => if is_golden_term a b then 1 + count_golden_terms (b :: l) else count_golden_terms (b :: l)

-- Main theorem: Prove that the maximum number of golden terms in a permutation of {1, 2, 3, ..., 2021} is 1010.
theorem max_golden_terms (L : List ℕ)
  (hL : L.perm (List.range 2022)) :
  count_golden_terms L ≤ 1010 :=
sorry

end max_golden_terms_l223_223458


namespace AH_HD_ratio_l223_223478

-- Given conditions
variables {A B C H D : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited H] [Inhabited D]
variables (BC : ℝ) (AC : ℝ) (angle_C : ℝ)
-- We assume the values provided in the problem
variables (BC_eq : BC = 6) (AC_eq : AC = 4 * Real.sqrt 2) (angle_C_eq : angle_C = Real.pi / 4)

-- Altitudes and orthocenter assumption, representing intersections at orthocenter H
variables (A D H : Type) -- Points to represent A, D, and orthocenter H

noncomputable def AH_H_ratio (BC AC : ℝ) (angle_C : ℝ)
  (BC_eq : BC = 6) (AC_eq : AC = 4 * Real.sqrt 2) (angle_C_eq : angle_C = Real.pi / 4) : ℝ :=
  if BC = 6 ∧ AC = 4 * Real.sqrt 2 ∧ angle_C = Real.pi / 4 then 2 else 0

-- We need to prove the ratio AH:HD equals 2 given the conditions
theorem AH_HD_ratio (BC AC : ℝ) (angle_C : ℝ)
  (BC_eq : BC = 6) (AC_eq : AC = 4 * Real.sqrt 2) (angle_C_eq : angle_C = Real.pi / 4) :
  AH_H_ratio BC AC angle_C BC_eq AC_eq angle_C_eq = 2 :=
by {
  -- the statement will be proved here
  sorry
}

end AH_HD_ratio_l223_223478


namespace arrangements_mississippi_l223_223727

open Nat

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem arrangements_mississippi : 
  let total := 11
  let m_count := 1
  let i_count := 4
  let s_count := 4
  let p_count := 2
  factorial total / (factorial m_count * factorial i_count * factorial s_count * factorial p_count) = 34650 :=
by
  -- This is the proof placeholder
  sorry

end arrangements_mississippi_l223_223727


namespace derivative_of_y_l223_223763

-- Define the function y = sin x / x
def y (x : ℝ) : ℝ := sin x / x

-- Statement to prove the derivative of y
theorem derivative_of_y (x : ℝ) (h : x ≠ 0) : deriv y x = (x * cos x - sin x) / x^2 :=
by 
  sorry

end derivative_of_y_l223_223763


namespace split_stones_l223_223077

theorem split_stones (n : ℕ) :
  ∃ (heaps : list ℕ), (∀ h ∈ heaps, 1 ≤ h ∧ h ≤ n) ∧ (∀ i j, i ≠ j → (i < heaps.length ∧ j < heaps.length → heaps.nth i ≤ 2 * heaps.nth j)) :=
sorry

end split_stones_l223_223077


namespace problem_equivalence_l223_223010

theorem problem_equivalence (n : ℕ) (x : ℝ) (h : x = 1 / 2 * (1991 ^ (1 / n : ℝ) - 1991 ^ (-1 / n : ℝ))) :
  (x - real.sqrt (1 + x ^ 2)) ^ n = (-1) ^ n * 1991 ^ (-1) :=
by
  sorry

end problem_equivalence_l223_223010


namespace remaining_laps_l223_223146

theorem remaining_laps (total_laps_friday : ℕ)
                       (total_laps_saturday : ℕ)
                       (laps_sunday_morning : ℕ)
                       (total_required_laps : ℕ)
                       (total_laps_weekend : ℕ)
                       (remaining_laps : ℕ) :
  total_laps_friday = 63 →
  total_laps_saturday = 62 →
  laps_sunday_morning = 15 →
  total_required_laps = 198 →
  total_laps_weekend = total_laps_friday + total_laps_saturday + laps_sunday_morning →
  remaining_laps = total_required_laps - total_laps_weekend →
  remaining_laps = 58 := by
  intros
  sorry

end remaining_laps_l223_223146


namespace arithmetic_sequence_l223_223064

-- Define the nth term of the arithmetic sequence
def a_n (n : ℕ) (d a1 : ℤ) : ℤ := a1 + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def S_n (n : ℕ) (d a1 : ℤ) : ℤ := n * a1 + (n * (n - 1)) / 2 * d

-- Given conditions
theorem arithmetic_sequence (n : ℕ) (d a1 : ℤ) (S3 : ℤ) (h1 : a1 = 10) (h2 : S_n 3 d a1 = 24) :
  (a_n n d a1 = 12 - 2 * n) ∧ (S_n n (-2) 12 = -n^2 + 11 * n) ∧ (∀ k, S_n k (-2) 12 ≤ 30) :=
by
  sorry

end arithmetic_sequence_l223_223064


namespace product_of_two_numbers_l223_223989

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 16) (h2 : x^2 + y^2 = 200) : x * y = 28 :=
sorry

end product_of_two_numbers_l223_223989


namespace smallest_sum_of_digits_l223_223899

theorem smallest_sum_of_digits :
  ∃ (a b S : ℕ), 
    (100 ≤ a ∧ a < 1000) ∧ 
    (100 ≤ b ∧ b < 1000) ∧ 
    (S = a + b) ∧ 
    (1000 ≤ S ∧ S < 10000) ∧ 
    (∀ d, d ∈ digits 10 a → d ∉ digits 10 b) ∧ 
    (sum_of_digits 10 S = 8) := 
sorry

end smallest_sum_of_digits_l223_223899


namespace compute_AB_plus_AC_l223_223324

noncomputable theory
open_locale classical

variables {O A B C : ℝ} (ω : {radius : ℝ // radius = 3}) (BC : ℝ)

-- Define points and distances
def OA : ℝ := 10
def r : ℝ := ω.1

-- Conditions
def tangent_lengths : ℝ := real.sqrt (OA^2 - r^2)
def BC_length : BC = 9

-- Theorem statement
theorem compute_AB_plus_AC (h1 : OA = 10) 
                           (h2 : r = 3) 
                           (h3 : BC = 9) : 
  2 * real.sqrt (OA^2 - r^2) + BC = 2 * real.sqrt 91 + 9 := 
by sorry

end compute_AB_plus_AC_l223_223324


namespace find_m_n_l223_223776

noncomputable def A : Set ℝ := {3, 5}
def B (m n : ℝ) : Set ℝ := {x | x^2 + m * x + n = 0}

theorem find_m_n (m n : ℝ) (h_union : A ∪ B m n = A) (h_inter : A ∩ B m n = {5}) :
  m = -10 ∧ n = 25 :=
by
  sorry

end find_m_n_l223_223776


namespace vertex_of_parabola_l223_223572

theorem vertex_of_parabola (x y : ℝ) : y = (x - 1)^2 - 2 → ∃ h k : ℝ, h = 1 ∧ k = -2 ∧ y = (x - h)^2 + k :=
by
  intros h k
  existsi 1, -2
  split
  { refl }
  split
  { refl }
  { sorry }

end vertex_of_parabola_l223_223572


namespace car_catch_up_distance_l223_223708

-- Define the problem: Prove that Car A catches Car B at the specified location, given the conditions

theorem car_catch_up_distance
  (distance : ℝ := 300)
  (v_A v_B t_A : ℝ)
  (start_A_late : ℝ := 1) -- Car A leaves 1 hour later
  (arrive_A_early : ℝ := 1) -- Car A arrives 1 hour earlier
  (t : ℝ := 2) -- Time when Car A catches up with Car B
  (dist_eq_A : distance = v_A * t_A)
  (dist_eq_B : distance = v_B * (t_A + 2))
  (catch_up_time_eq : v_A * t = v_B * (t + 1)):
  (distance - v_A * t) = 150 := sorry

end car_catch_up_distance_l223_223708


namespace split_piles_equiv_single_stone_heaps_l223_223096

theorem split_piles_equiv_single_stone_heaps (n : ℕ) (heaps : List ℕ) (h_initial : ∀ h ∈ heaps, h = 1)
  (h_size : heaps.length = n) :
  ∃ final_heap, (∀ x y ∈ heaps, x + y ≤ 2 * max x y) ∧ (List.sum heaps = (heaps.length) * 1) := by
  sorry

end split_piles_equiv_single_stone_heaps_l223_223096


namespace gcd_1337_382_l223_223201

theorem gcd_1337_382 : Nat.gcd 1337 382 = 191 := by
  sorry

end gcd_1337_382_l223_223201


namespace heaps_combination_preserve_similarity_split_stones_into_similar_heaps_l223_223123

def initial_heaps (n : ℕ) : list ℕ := list.repeat 1 n

def combine_heaps (heaps : list ℕ) : list ℕ :=
  if heaps.length ≥ 2 then
    let min1 := list.minimum heaps,
        heaps' := list.erase heaps min1,
        min2 := list.minimum heaps'
    in
    if min1 ≤ min2 then
      (min1 + min2) :: list.erase heaps' min2
    else
      heaps
  else
    heaps

theorem heaps_combination_preserve_similarity (heaps : list ℕ) (h : ∀ x ∈ heaps, x = 1) :
  ∀ combined_heaps, combined_heaps = combine_heaps heaps →
  ∀ x y ∈ combined_heaps, x ≤ y → x + y ≤ 2 * y :=
sorry

theorem split_stones_into_similar_heaps (n : ℕ) :
  ∃ combined_heaps : list ℕ, ∀ x y ∈ combined_heaps, x ≤ y → x + y ≤ 2 * y :=
sorry

end heaps_combination_preserve_similarity_split_stones_into_similar_heaps_l223_223123


namespace races_to_champion_l223_223455

theorem races_to_champion (num_sprinters : ℕ) (sprinters_per_race : ℕ) (advancing_per_race : ℕ)
  (eliminated_per_race : ℕ) (initial_races : ℕ) (total_races : ℕ):
  num_sprinters = 360 ∧ sprinters_per_race = 8 ∧ advancing_per_race = 2 ∧ 
  eliminated_per_race = 6 ∧ initial_races = 45 ∧ total_races = 62 →
  initial_races + (initial_races / sprinters_per_race +
  ((initial_races / sprinters_per_race) / sprinters_per_race +
  (((initial_races / sprinters_per_race) / sprinters_per_race) / sprinters_per_race + 1))) = total_races :=
sorry

end races_to_champion_l223_223455


namespace angle_EF_AB_is_90_l223_223885

noncomputable def midpoint (p1 p2 : Point) : Point :=
{ x := (p1.x + p2.x) / 2, 
  y := (p1.y + p2.y) / 2,
  z := (p1.z + p2.z) / 2 }

noncomputable def centroid (p1 p2 p3 : Point) : Point :=
{ x := (p1.x + p2.x + p3.x) / 3, 
  y := (p1.y + p2.y + p3.y) / 3,
  z := (p1.z + p2.z + p3.z) / 3 }

variables (S A B C : Point) (H : regular_tetrahedron S A B C)

noncomputable def E : Point := midpoint S A
noncomputable def F : Point := centroid A B C

-- The main problem statement in Lean 4
theorem angle_EF_AB_is_90 (h : regular_tetrahedron S A B C) :
  angle (line_through E F) (line_through A B) = 90 :=
sorry

end angle_EF_AB_is_90_l223_223885


namespace q_computation_l223_223073

def q : ℤ → ℤ → ℤ :=
  λ x y =>
    if x ≥ 0 ∧ y ≥ 0 then x + 2 * y
    else if x < 0 ∧ y < 0 then x - 3 * y
    else 2 * x + y

theorem q_computation : q (q 2 (-2)) (q (-4) (-1)) = 3 :=
by {
  sorry
}

end q_computation_l223_223073


namespace height_of_pole_l223_223302

/-- A telephone pole is supported by a steel cable extending from the top of the pole to a point on the ground 3 meters from its base.
When Leah, who is 1.5 meters tall, stands 2.5 meters from the base of the pole towards the point where the cable is attached to the ground,
her head just touches the cable. Prove that the height of the pole is 9 meters. -/
theorem height_of_pole 
  (cable_length_from_base : ℝ)
  (leah_distance_from_base : ℝ)
  (leah_height : ℝ)
  : cable_length_from_base = 3 → leah_distance_from_base = 2.5 → leah_height = 1.5 → 
    (∃ height_of_pole : ℝ, height_of_pole = 9) := 
by
  intros h1 h2 h3
  sorry

end height_of_pole_l223_223302


namespace isosceles_base_angle_l223_223469

theorem isosceles_base_angle (A B C : Type) 
  (iso : IsoscelesTriangle A B C) (vertex_angle : ∠A = 100)
  (triangle_sum : ∠A + ∠B + ∠C = 180) :
  ∠B = 40 ∧ ∠C = 40 :=
by
  sorry

end isosceles_base_angle_l223_223469


namespace total_length_of_hike_is_7_l223_223011

variable {initial_water : ℕ} -- initial water in the canteen in cups
variable {remaining_water : ℕ} -- remaining water after the hike in cups
variable {leak_rate : ℕ} -- leakage rate (cups/hour)
variable {hike_time : ℕ} -- total hike time in hours
variable {last_mile_drink : ℕ} -- water drunk in the last mile in cups
variable {first_part_drink_per_mile : ℚ} -- water drunk per mile during the first part (cups/mile)

-- Given conditions
def initial_water := 9
def remaining_water := 3
def leak_rate := 1
def hike_time := 2
def last_mile_drink := 2
def first_part_drink_per_mile := 2 / 3

-- Proving the total length of the hike is 7 miles
theorem total_length_of_hike_is_7 :
  ∃ (first_part_length : ℚ) (last_mile_length : ℕ), 
    (last_mile_length = 1) ∧ 
    (first_part_length = (initial_water - (remaining_water + last_mile_drink + (leak_rate * hike_time))) / first_part_drink_per_mile) ∧ 
    ((first_part_length : ℕ) + last_mile_length = 7) :=
by
  sorry

end total_length_of_hike_is_7_l223_223011


namespace bcqp_concyclic_l223_223878

noncomputable theory
open_locale classical 

variables {A B C D P Q : Type} [metric_space A] 

-- Conditions
variables (h_isosceles : dist A B = dist A C)
variables (h_inside : dist D A = dist D B + dist D C)
variables (h_P : ∃ P, is_perp_bisector P (A, B) ∧ is_ext_angle_bisector P (angle (D, A, B)))
variables (h_Q : ∃ Q, is_perp_bisector Q (A, C) ∧ is_ext_angle_bisector Q (angle (D, A, C)))

-- Prove that B, C, P, Q are concyclic
theorem bcqp_concyclic 
  (h_isosceles : dist A B = dist A C)
  (h_inside : dist D A = dist D B + dist D C)
  (h_P : ∃ P, is_perp_bisector P (A, B) ∧ is_ext_angle_bisector P (angle (D, A, B)))
  (h_Q : ∃ Q, is_perp_bisector Q (A, C) ∧ is_ext_angle_bisector Q (angle (D, A, C))) : 
  cyclic {B, C, P, Q} :=
sorry

end bcqp_concyclic_l223_223878


namespace find_a_values_l223_223586

theorem find_a_values (a : ℝ) : 
  (∃ x : ℝ, (a * x^2 + (a - 3) * x + 1 = 0)) ∧ 
  (∀ x1 x2 : ℝ, (a * x1^2 + (a - 3) * x1 + 1 = 0 ∧ a * x2^2 + (a - 3) * x2 + 1 = 0 → x1 = x2)) 
  ↔ a = 0 ∨ a = 1 ∨ a = 9 :=
sorry

end find_a_values_l223_223586


namespace total_pens_l223_223991

theorem total_pens (b l : ℕ) (h1 : b = 4) (h2 : l = 4) : b + l = 8 :=
by
  rw [h1, h2]
  rfl

end total_pens_l223_223991


namespace degree_f_x2_g_x3_l223_223181

open Polynomial

noncomputable def degree_of_composite_polynomials (f g : Polynomial ℝ) : ℕ :=
  let f_degree := Polynomial.degree f
  let g_degree := Polynomial.degree g
  match (f_degree, g_degree) with
  | (some 3, some 6) => 24
  | _ => 0

theorem degree_f_x2_g_x3 (f g : Polynomial ℝ) (h_f : Polynomial.degree f = 3) (h_g : Polynomial.degree g = 6) :
  Polynomial.degree (Polynomial.comp f (X^2) * Polynomial.comp g (X^3)) = 24 := by
  -- content Logic Here
  sorry

end degree_f_x2_g_x3_l223_223181


namespace bell_peppers_needed_l223_223956

-- Definitions based on the conditions
def large_slices_per_bell_pepper : ℕ := 20
def small_pieces_from_half_slices : ℕ := (20 / 2) * 3
def total_slices_and_pieces_per_bell_pepper : ℕ := large_slices_per_bell_pepper / 2 + small_pieces_from_half_slices
def desired_total_slices_and_pieces : ℕ := 200

-- Proving the number of bell peppers needed
theorem bell_peppers_needed : 
  desired_total_slices_and_pieces / total_slices_and_pieces_per_bell_pepper = 5 := 
by 
  -- Add the proof steps here
  sorry

end bell_peppers_needed_l223_223956


namespace f_of_f_neg_one_l223_223409

def f (x : ℝ) : ℝ :=
  if x < 0 then real.sqrt (-x)
  else (x - 1/2) ^ 4

theorem f_of_f_neg_one : f (f (-1)) = 1 / 16 :=
by
  sorry

end f_of_f_neg_one_l223_223409


namespace ratio_of_good_states_l223_223247

theorem ratio_of_good_states (n : ℕ) :
  let total_states := 2^(2*n)
  let good_states := Nat.choose (2 * n) n
  good_states / total_states = (List.range n).foldr (fun i acc => acc * (2*i+1)) 1 / (2^n * Nat.factorial n) := sorry

end ratio_of_good_states_l223_223247


namespace car_A_catches_up_l223_223704

variables (v_A v_B t_A : ℝ)

-- The conditions of the problem
def distance : ℝ := 300
def time_car_B : ℝ := t_A + 2
def distance_eq_A : Prop := distance = v_A * t_A
def distance_eq_B : Prop := distance = v_B * time_car_B

-- The final proof problem: Car A catches up with car B 150 kilometers away from city B.
theorem car_A_catches_up (t_A > 0) (v_A > 0) (v_B > 0) :
  distance_eq_A ∧ distance_eq_B → 
  ∃ d : ℝ, d = 150 := 
sorry

end car_A_catches_up_l223_223704


namespace largest_area_triangle_ABC_l223_223242

variable (x : ℝ)

def BC (x : ℝ) := 35 * x
def AC (x : ℝ) := 36 * x

noncomputable def s (x : ℝ) := (10 + 71 * x) / 2
noncomputable def k (x : ℝ) := (s x) * ((s x) - 10) * ((s x) - (BC x)) * ((s x) - (AC x))

theorem largest_area_triangle_ABC :
  ∃ x > 0, sqrt (k x) ≤ 1260 :=
by
  sorry

end largest_area_triangle_ABC_l223_223242


namespace simplest_form_is_C_l223_223261

variables (x y : ℝ) (hx : x ≠ 0) (hx1 : x ≠ 1) (hy : y ≠ 0)

def fraction_A := 3 * x * y / (x^2)
def fraction_B := (x - 1) / (x^2 - 1)
def fraction_C := (x + y) / (2 * x)
def fraction_D := (1 - x) / (x - 1)

theorem simplest_form_is_C : 
  ∀ (x y : ℝ) (hx : x ≠ 0) (hx1 : x ≠ 1) (hy : y ≠ 0), 
  ¬ (3 * x * y / (x^2)).is_simplest ∧ 
  ¬ ((x - 1) / (x^2 - 1)).is_simplest ∧ 
  (x + y) / (2 * x).is_simplest ∧ 
  ¬ ((1 - x) / (x - 1)).is_simplest :=
by 
  sorry

end simplest_form_is_C_l223_223261


namespace cheesecake_factory_savings_l223_223543

noncomputable def combined_savings : ℕ := 3000

theorem cheesecake_factory_savings :
  let hourly_wage := 10
  let daily_hours := 10
  let working_days := 5
  let weekly_hours := daily_hours * working_days
  let weekly_salary := weekly_hours * hourly_wage
  let robby_savings := (2/5 : ℚ) * weekly_salary
  let jaylen_savings := (3/5 : ℚ) * weekly_salary
  let miranda_savings := (1/2 : ℚ) * weekly_salary
  let combined_weekly_savings := robby_savings + jaylen_savings + miranda_savings
  4 * combined_weekly_savings = combined_savings :=
by
  sorry

end cheesecake_factory_savings_l223_223543


namespace cylinder_height_l223_223155

open Real

def sphere (radius: ℝ) := { x : ℝ × ℝ × ℝ // dist (x, (0, 0, 0)) = radius }
def cylinder (radius height: ℝ) := { x : ℝ × ℝ × ℝ // x.1^2 + x.2^2 = radius^2 ∧ 0 ≤ x.3 ∧ x.3 ≤ height } 

theorem cylinder_height :
  ∀ (spheres : Fin 8 → sphere 1)
    (cyl : cylinder (2 * sqrt 2) (sqrt 2 + 2))
    (tan : ∀ i, (spheres i).1.1^2 + (spheres i).1.2^2 = (2 * sqrt 2)^2 
                ∧ ((spheres i).1.3 = 0 ∨ (spheres i).1.3 = sqrt 2 + 2)
                ∧ (dist ((spheres i).1, (cyl.1))) = 1),
  cyl.height = sqrt[4] 8 + 2 := 
begin
  sorry
end

end cylinder_height_l223_223155


namespace arrangements_mississippi_l223_223729

open Nat

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem arrangements_mississippi : 
  let total := 11
  let m_count := 1
  let i_count := 4
  let s_count := 4
  let p_count := 2
  factorial total / (factorial m_count * factorial i_count * factorial s_count * factorial p_count) = 34650 :=
by
  -- This is the proof placeholder
  sorry

end arrangements_mississippi_l223_223729


namespace combine_heaps_l223_223124

def heaps_similar (x y : ℕ) : Prop :=
  x ≤ 2 * y ∧ y ≤ 2 * x

theorem combine_heaps (n : ℕ) : 
  ∃ f : ℕ → ℕ, 
  f 0 = n ∧
  ∀ k, k < n → (∃ i j, i + j = k ∧ heaps_similar (f i) (f j)) ∧ 
  (∃ k, f k = n) :=
by
  sorry

end combine_heaps_l223_223124


namespace split_piles_equiv_single_stone_heaps_l223_223094

theorem split_piles_equiv_single_stone_heaps (n : ℕ) (heaps : List ℕ) (h_initial : ∀ h ∈ heaps, h = 1)
  (h_size : heaps.length = n) :
  ∃ final_heap, (∀ x y ∈ heaps, x + y ≤ 2 * max x y) ∧ (List.sum heaps = (heaps.length) * 1) := by
  sorry

end split_piles_equiv_single_stone_heaps_l223_223094


namespace split_into_similar_piles_l223_223089

def similar_sizes (x y : ℕ) : Prop := x ≤ 2 * y ∧ y ≤ 2 * x

theorem split_into_similar_piles (n : ℕ) (h : 0 < n) :
  ∃ (piles : list ℕ), (∀ x ∈ piles, x = 1) ∧ (list.sum piles = n) ∧
                       (∀ x y ∈ piles, similar_sizes x y) := 
sorry

end split_into_similar_piles_l223_223089


namespace part1_max_min_values_part2_function_comparison_part3_derivative_inequality_l223_223820

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log x - 1
noncomputable def g (x : ℝ) : ℝ := x^3

theorem part1_max_min_values : 
  is_max (f 1) (f e) (f 1 = 0) (f e = e^2) := sorry

theorem part2_function_comparison {x : ℝ} (h1 : 1 < x) : 
  f x < g x := sorry

theorem part3_derivative_inequality {n : ℕ} (hn : 0 < n) {x : ℝ} (hx : 0 < x) : 
  (f' x)^n - f' (x^n) ≥ 2^n - 2 := sorry

end part1_max_min_values_part2_function_comparison_part3_derivative_inequality_l223_223820


namespace concurrency_of_lines_l223_223055

noncomputable theory
open_locale classical

variables {A B C I I_A I_B I_C : Type*} [triangle A B C]

-- Points I, I_A, I_B, I_C as described
variables (incircle_center : incenter A B C = I)
variables (excircle_A_center : A_excircle_center A B C = I_A)
variables (excircle_B_center : B_excircle_center A B C = I_B)
variables (excircle_C_center : C_excircle_center A B C = I_C)

-- Definitions of lines l_A, l_B, l_C through the appropriate orthocenters
def line_lA := {p : Π (orthocenter_1 orthocenter_2 : point), line p orthocenter_1 orthocenter_2} 
def line_lB := {p : Π (orthocenter_1 orthocenter_2 : point), line p orthocenter_1 orthocenter_2}
def line_lC := {p : Π (orthocenter_1 orthocenter_2 : point), line p orthocenter_1 orthocenter_2}

theorem concurrency_of_lines :
  are_concurrent (line_lA (orthocenter (triangle I B C)) (orthocenter (triangle I_A B C)))
                 (line_lB (orthocenter (triangle I C A)) (orthocenter (triangle I_B C A)))
                 (line_lC (orthocenter (triangle I A B)) (orthocenter (triangle I_C A B))) :=
sorry

end concurrency_of_lines_l223_223055


namespace correct_calculation_result_l223_223638

theorem correct_calculation_result (x : ℤ) (h : x + 63 = 8) : x * 36 = -1980 := by
  sorry

end correct_calculation_result_l223_223638


namespace imaginary_part_of_z_l223_223397

def z : ℂ := 2 / (-1 + complex.I)
#check z

theorem imaginary_part_of_z : z.im = -1 :=
by
  sorry

end imaginary_part_of_z_l223_223397


namespace distance_from_stream_to_meadow_l223_223893

noncomputable def distance_from_car_to_stream : ℝ := 0.2
noncomputable def distance_from_meadow_to_campsite : ℝ := 0.1
noncomputable def total_distance_hiked : ℝ := 0.7

theorem distance_from_stream_to_meadow : 
  (total_distance_hiked - distance_from_car_to_stream - distance_from_meadow_to_campsite = 0.4) :=
by
  sorry

end distance_from_stream_to_meadow_l223_223893


namespace fraction_operation_l223_223253

theorem fraction_operation : 
  let a := (2 : ℚ) / 9
  let b := (5 : ℚ) / 6
  let c := (1 : ℚ) / 18
  (a * b) + c = 13 / 54 :=
by
  sorry

end fraction_operation_l223_223253


namespace moles_of_NaH_l223_223426

theorem moles_of_NaH (nH2O : ℕ) (nNaOH : ℕ) (nH2 : ℕ)
  (h1 : nH2O = 2)
  (h2 : nNaOH = 2)
  (h3 : nH2 = 2) :
  ∃ nNaH, nNaH = 2 :=
by
  use 2
  exact rfl

end moles_of_NaH_l223_223426


namespace second_percentage_increase_l223_223974

variable (P : ℝ) (x : ℝ)

theorem second_percentage_increase :
  (1.15 * P) * (1 + x / 100) = 1.4375 * P → x = 25 :=
by
  intro h
  have h1 : (1.15 * P) * (1 + x / 100) / (1.15 * P) = 1.4375 * P / (1.15 * P) := by rw h
  sorry

end second_percentage_increase_l223_223974


namespace possible_values_of_sum_l223_223639

theorem possible_values_of_sum
  (p q r : ℝ)
  (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p)
  (h_system : q = p * (4 - p) ∧ r = q * (4 - q) ∧ p = r * (4 - r)) :
  p + q + r = 6 ∨ p + q + r = 7 := by
  sorry

end possible_values_of_sum_l223_223639


namespace find_hyperbola_focus_l223_223717

theorem find_hyperbola_focus : ∃ (x y : ℝ), 
  2 * x ^ 2 - 3 * y ^ 2 + 8 * x - 12 * y - 8 = 0 
  → (x, y) = (-2 + (Real.sqrt 30)/3, -2) :=
by
  sorry

end find_hyperbola_focus_l223_223717


namespace hyperbola_focus_coordinates_l223_223351

open Real

theorem hyperbola_focus_coordinates :
  ∃ x y : ℝ, (2 * x^2 - y^2 + 8 * x + 4 * y - 28 = 0) ∧
           ((x = -2 - 4 * sqrt 3 ∧ y = 2) ∨ (x = -2 + 4 * sqrt 3 ∧ y = 2)) := by sorry

end hyperbola_focus_coordinates_l223_223351


namespace assume_proof_by_contradiction_l223_223257

theorem assume_proof_by_contradiction (a b : ℤ) (hab : ∃ k : ℤ, ab = 3 * k) :
  (¬ (∃ k : ℤ, a = 3 * k) ∧ ¬ (∃ k : ℤ, b = 3 * k)) :=
sorry

end assume_proof_by_contradiction_l223_223257


namespace triangle_AD_eq_5_l223_223480

theorem triangle_AD_eq_5
  (A B C D : Type)
  [Triangle A B C]
  (hD_on_AB : On D (Line A B))
  (hCD_perp_BC : Perp (Line C D) (Line B C))
  (hAC_eq : length A C = 5 * Real.sqrt 3)
  (hCD_eq : length C D = 5)
  (hBD_eq_2AD : length B D = 2 * length A D) :
  length A D = 5 :=
sorry

end triangle_AD_eq_5_l223_223480


namespace sin_pi_plus_alpha_eq_neg_half_l223_223036

theorem sin_pi_plus_alpha_eq_neg_half :
  ∀ (α : ℝ), (∃ (x y : ℝ), x = -sqrt 3 / 2 ∧ y = 1 / 2 ∧ x = sin (5 * π / 3) ∧ y = cos (5 * π / 3)) →
  sin (π + α) = -1 / 2 :=
by
  rintros α ⟨x, y, hx, hy, hsin, hcos⟩
  sorry

end sin_pi_plus_alpha_eq_neg_half_l223_223036


namespace is_fantabulous_2021_pow_2021_l223_223205

def fantabulous (n : ℕ) : Prop :=
∀ m : ℕ, m = n → (∀ a ∈ ({m, 2 * m + 1, 3 * m} : set ℕ), fantabulous a) → (∀ b ∈ ({m, 2 * m + 1, 3 * m} : set ℕ), fantabulous b)

theorem is_fantabulous_2021_pow_2021 : fantabulous 2021 → (∀ m : ℕ, fantabulous m → (∀ a ∈ ({m, 2 * m + 1, 3 * m} : set ℕ), fantabulous a) → (∀ b ∈ ({m, 2 * m + 1, 3 * m} : set ℕ), fantabulous b)) → fantabulous (2021 ^ 2021) :=
begin
  intros h2021 h_all_m,
  sorry,
end

end is_fantabulous_2021_pow_2021_l223_223205


namespace length_of_CD_l223_223476

-- Define the structures and conditions
variables {A B C D E F O : Type} [nonempty A] [nonempty B] [nonempty C] [nonempty D] [nonempty E] [nonempty F] [nonempty O]

noncomputable def is_isosceles_triangle (triangle : A × B × C) : Prop := 
  let (A, B, C) := triangle in dist A B = dist A C

variables (O_center : O) (O_is_center_of_circle : true)
variables (A_on_circle : true) (B_on_circle : true) (C_on_circle : true)
variables (A B C D E F : A) -- Points on the plane
variables (AD_diameter : true)
variables (AE_intersects_BC_at_E : true)
variables (F_midpoint_OE : true)

variables (BD_parallel_FC : true) (BC_length : dist B C = 2 * Real.sqrt 5)

-- State the theorem to find the length of CD
theorem length_of_CD : 
  is_isosceles_triangle (A, B, C) →
  dist B C = 2 * Real.sqrt 5 →
  BD_parallel_FC →
  AE_intersects_BC_at_E →
  AD_diameter →
  F_midpoint_OE →
  dist C D = Real.sqrt 6 :=
by
  intros h_isosceles h_BC h_parallel h_intersect h_diameter h_midpoint
  sorry

end length_of_CD_l223_223476


namespace find_s_l223_223348

noncomputable theory

open Real

theorem find_s (s : ℝ) (h : 4 * log 3 s = log 3 (6 * s)) : s = real.cbrt 6 :=
by
  sorry

end find_s_l223_223348


namespace base8_to_base10_conversion_l223_223657

theorem base8_to_base10_conversion : 
  let n := 432
  let base := 8
  let result := 282
  (2 * base^0 + 3 * base^1 + 4 * base^2) = result := 
by
  let n := 2 * 8^0 + 3 * 8^1 + 4 * 8^2
  have h1 : n = 2 + 24 + 256 := by sorry
  have h2 : 2 + 24 + 256 = 282 := by sorry
  exact Eq.trans h1 h2


end base8_to_base10_conversion_l223_223657


namespace simplify_expression_l223_223176

theorem simplify_expression (y : ℝ) : (3 * y^4)^5 = 243 * y^20 :=
sorry

end simplify_expression_l223_223176


namespace magnitude_of_angle_B_range_of_dot_product_l223_223450

variables (A B C a b c : Real)
variables (m1 m2 n1 n2 : Real)

-- Conditions
def triangle_condition : Prop := (c - b) / (Real.sqrt 2 - a) = Real.sin A / (Real.sin B + Real.sin C)

-- Statement to prove
theorem magnitude_of_angle_B (h : triangle_condition A B C a b c) : B = Real.pi / 4 :=
sorry

theorem range_of_dot_product (h1 : triangle_condition A B C a b c)
  (m1 = Real.sin A + Real.cos A) (m2 = 1) (n1 = 2) (n2 = Real.cos (Real.pi / 2 - 2 * A)) :
  -1 < 2 * m1 + n2 + m1^2 - 1 ∧ 2 * m1 + n2 + m1^2 - 1 ≤ 1 + 2 * Real.sqrt 2 :=
sorry

end magnitude_of_angle_B_range_of_dot_product_l223_223450


namespace banana_cost_l223_223755

/-- If 4 bananas cost $20, then the cost of one banana is $5. -/
theorem banana_cost (total_cost num_bananas : ℕ) (cost_per_banana : ℕ) 
  (h : total_cost = 20 ∧ num_bananas = 4) : cost_per_banana = 5 := by
  sorry

end banana_cost_l223_223755


namespace minimum_chord_length_l223_223376

noncomputable def circleEquation (x y : ℝ) : Prop :=
    (x - 3)^2 + y^2 = 25

noncomputable def lineEquation (m x y : ℝ) : Prop :=
    (m + 1) * x + (m - 1) * y - 2 = 0

theorem minimum_chord_length (m : ℝ) :
    ∃ (p : ℝ), circleEquation p 0 ∧ lineEquation m p 0 ∧ 
    ∀ (chord_len : ℝ), chord_len >= 4 * real.sqrt 5 :=
sorry

end minimum_chord_length_l223_223376


namespace part_a_part_b_l223_223230

-- Definitions used in the conditions
def five_kids : Type := Fin 5
def distinct (f : five_kids → ℕ) := ∀ i j, i ≠ j → f i ≠ f j
def left_of (i : five_kids) : five_kids := Fin.mk ((i.val + 1) % 5) sorry
def right_of (i : five_kids) : five_kids := Fin.mk ((i.val + 4) % 5) sorry
def guess_diff (f : five_kids → ℕ) (i : five_kids) : ℕ := |f (left_of i) - f (right_of i)|

-- Question (a): If the total number of apples is less than 16, at least one of the kids will guess the difference correctly
theorem part_a (f : five_kids → ℕ) (total : ℕ) (h_total : total < 16) (h_sum : (∑ i, f i) = total)
  (h_distinct : distinct f) : ∃ i, guess_diff f i = |f (left_of i) - f (right_of i)| :=
sorry

-- Question (b): Prove that the teacher can give the total of 16 apples such that no one can guess the difference correctly
theorem part_b : ∃ (f : five_kids → ℕ), (∑ i, f i) = 16 ∧ distinct f ∧ (∀ i, guess_diff f i ≠ |f (left_of i) - f (right_of i)|) :=
sorry

end part_a_part_b_l223_223230


namespace arrange_MISSISSIPPI_l223_223731

theorem arrange_MISSISSIPPI : 
  let total_letters := 11
  let count_I := 4
  let count_S := 4
  let count_P := 2
  let arrangements := 11.factorial / (count_I.factorial * count_S.factorial * count_P.factorial)
  in arrangements = 34650 :=
by
  have h1 : 11.factorial = 39916800 := by sorry
  have h2 : 4.factorial = 24 := by sorry
  have h3 : 2.factorial = 2 := by sorry
  have h4 : arrangements = 34650 := by sorry
  exact h4

end arrange_MISSISSIPPI_l223_223731


namespace part_one_part_two_l223_223408

noncomputable def f (a x : ℝ) : ℝ := x / (Real.log x) - a * x
noncomputable def f_deriv (a x : ℝ) : ℝ := (Real.log x - 1) / (Real.log x)^2 - a

theorem part_one (a : ℝ) :
  (∀ x > 1, f_deriv a x ≤ 0) ↔ a ≥ 1 / 4 :=
by sorry

theorem part_two (a : ℝ) :
  (∃ x1 x2, x1 ∈ Icc (Real.exp 1) (Real.exp 2) ∧ x2 ∈ Icc (Real.exp 1) (Real.exp 2) ∧ f a x1 - f_deriv a x2 ≤ a) ↔ a ≥ 1 / 2 - 1 / (4 * Real.exp 2 ^ 2) :=
by sorry

end part_one_part_two_l223_223408


namespace distance_between_intersections_l223_223194

theorem distance_between_intersections {x : ℝ} : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (tan (3 * x1) = 2016 ∧ tan (3 * x2) = 2016) ∧ ∀ x3 : ℝ, x3 ≠ x1 ∧ x3 ≠ x2 → tan (3 * x3) ≠ 2016) →
  (abs (x1 - x2) = π / 3) :=
by sorry

end distance_between_intersections_l223_223194


namespace exists_large_sigma_div_n_l223_223528

noncomputable def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.divisors n).sum id

theorem exists_large_sigma_div_n (k : ℕ) : ∃ n : ℕ, (sum_of_divisors n) / n > k := by
  sorry

end exists_large_sigma_div_n_l223_223528


namespace imaginary_part_condition_l223_223861

-- Given condition
def condition (z : ℂ) : Prop := complex.i * z = -((1 + complex.i) * (1/2 : ℝ))

-- Problem statement to prove
theorem imaginary_part_condition : ∃ z : ℂ, condition z ∧ complex.im z = 1/2 :=
sorry

end imaginary_part_condition_l223_223861


namespace range_of_m_l223_223830

-- Define the proposition
def P : Prop := ∀ x : ℝ, ∃ m : ℝ, 4^x - 2^(x + 1) + m = 0

-- Given that the negation of P is false
axiom neg_P_false : ¬¬P

-- Prove the range of m
theorem range_of_m : ∀ m : ℝ, (∀ x : ℝ, 4^x - 2^(x + 1) + m = 0) → m ≤ 1 :=
by
  sorry

end range_of_m_l223_223830


namespace function_gcd_condition_l223_223349

theorem function_gcd_condition (f : ℕ → ℕ) (B : Set ℕ) (hB : ∀ p, (p ∈ B ∧ Prime p) ∨ (Prime p ∧ p ∉ B)) :
  (∀ x y, 0 < x → 0 < y → Nat.gcd (f x) y * f (x * y) = f x * f y) ↔
    ∀ x, f x = ∏ q in B, q :=
sorry

end function_gcd_condition_l223_223349


namespace part_I_part_II_l223_223788

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := log x - m * x^2
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := (1 / 2) * m * x^2 + x

-- Prove that for f(x) = ln(x) - (1/2)x^2, f is monotonically increasing on (0, 1)
theorem part_I : ∀ x : ℝ, x ∈ Ioo 0 1 → deriv (λ x, log x - (1 / 2 : ℝ) * x^2) x > 0 := 
by
  sorry

-- Prove that if f(x) + g(x) ≤ mx - 1 for all x, the minimum integer value of m is 2
theorem part_II (h : ∀ x : ℝ, f(x, m) + g(x, m) ≤ m * x - 1) : m = 2 := 
by
  sorry

end part_I_part_II_l223_223788


namespace solution_of_fraction_l223_223868

theorem solution_of_fraction (x : ℝ) (h1 : x^2 - 9 = 0) (h2 : x + 3 ≠ 0) : x = 3 :=
by
  sorry

end solution_of_fraction_l223_223868


namespace sequence_2011th_term_l223_223996

theorem sequence_2011th_term : 
  ∀ (n : ℕ), n = 2011 -> (sequence n) = 2^(n-1) :=
by
  intro n hn
  sorry

end sequence_2011th_term_l223_223996


namespace g_is_inequality_function_h_is_inequality_function_iff_a_eq_1_l223_223668

noncomputable def is_inequality_function (f : ℝ → ℝ) : Prop :=
  (∀ x ∈ Icc 0 1, f x ≥ 0) ∧
  (∀ x1 x2 ∈ Icc 0 1, x1 + x2 ∈ Icc 0 1 → f (x1 + x2) ≥ f x1 + f x2)

noncomputable def g : ℝ → ℝ := λ x, x^3
noncomputable def h (a : ℝ) : ℝ → ℝ := λ x, 2^x - a

theorem g_is_inequality_function : is_inequality_function g :=
sorry

theorem h_is_inequality_function_iff_a_eq_1 (a : ℝ) :
  is_inequality_function (h a) ↔ a = 1 :=
sorry

end g_is_inequality_function_h_is_inequality_function_iff_a_eq_1_l223_223668


namespace random_events_l223_223691

def is_random_event_1 (a b : ℝ) (ha : a > 0) (hb : b < 0) : Prop :=
  ∃ c d : ℝ, c > 0 → d < 0 → a + d < 0 ∨ b + c > 0

def is_random_event_2 (a b : ℝ) (ha : a > 0) (hb : b < 0) : Prop :=
  ∃ c d : ℝ, c > 0 → d < 0 → a - d > 0 ∨ b - c < 0

def is_impossible_event_3 (a b : ℝ) (ha : a > 0) (hb : b < 0) : Prop :=
  a * b > 0

def is_certain_event_4 (a b : ℝ) (ha : a > 0) (hb : b < 0) : Prop :=
  a / b < 0

theorem random_events (a b : ℝ) (ha : a > 0) (hb : b < 0) :
  is_random_event_1 a b ha hb ∧ is_random_event_2 a b ha hb :=
by
  sorry

end random_events_l223_223691


namespace find_y_arithmetic_mean_l223_223807

theorem find_y_arithmetic_mean (y : ℝ) 
  (h : (8 + 15 + 20 + 7 + y + 9) / 6 = 12) : 
  y = 13 :=
sorry

end find_y_arithmetic_mean_l223_223807


namespace unique_root_iff_k_eq_4_l223_223836

theorem unique_root_iff_k_eq_4 (k : ℝ) : 
  (∃! x : ℝ, x^2 - 4 * x + k = 0) ↔ k = 4 := 
by {
  sorry
}

end unique_root_iff_k_eq_4_l223_223836


namespace sum_of_solutions_l223_223627

theorem sum_of_solutions : 
  (∀ x : ℝ, (3 * x) / 15 = 4 / x) → (0 + 4 = 4) :=
by
  sorry

end sum_of_solutions_l223_223627


namespace quadrilateral_trapezoid_or_parallelogram_l223_223296

theorem quadrilateral_trapezoid_or_parallelogram
  (s1 s2 s3 s4 : ℝ)
  (hs : s1^2 = s2 * s4) :
  (exists (is_trapezoid : Prop), is_trapezoid) ∨ (exists (is_parallelogram : Prop), is_parallelogram) :=
by
  sorry

end quadrilateral_trapezoid_or_parallelogram_l223_223296


namespace girls_mistaken_l223_223945

-- Define the assumptions and the problem conditions
def candies_per_problem := 7
def total_candies_each := 20 
def total_problems : ℕ := sorry  -- Number of problems is derived from the conditions.

-- The girls collectively receive this total number of candies
def total_candies := total_candies_each * 3

-- The goal is to prove that the distribution is not feasible
theorem girls_mistaken : ¬ ∃ n : ℕ, total_candies = candies_per_problem * n :=
by {
  have h : total_candies = 60 := rfl,
  have h2 : ¬ ∃ n : ℕ, 60 = 7 * n,
    by {
      intro n,
      have h3 : 60 % 7 = 4 := rfl,
      rw [Nat.mul_div_cancel_left] at h3,
      contradiction,
    },
  exact h2,
}

end girls_mistaken_l223_223945


namespace parallel_to_plane_necessary_not_sufficient_parallel_to_plane_not_sufficient_but_necessary_l223_223273

variables {l : Line} {a : Plane}
def line_parallel_to_lines_in_plane (l : Line) (a : Plane) : Prop :=
  ∀ l', l' ∈ a → l ∥ l'

def line_parallel_to_plane (l : Line) (a : Plane) : Prop :=
  ∀ p, p ∈ l → p ∉ a

theorem parallel_to_plane_necessary_not_sufficient :
  (line_parallel_to_lines_in_plane l a) → ¬ (line_parallel_to_plane l a) :=
sorry

theorem parallel_to_plane_not_sufficient_but_necessary :
  (line_parallel_to_plane l a) → (line_parallel_to_lines_in_plane l a) :=
sorry

end parallel_to_plane_necessary_not_sufficient_parallel_to_plane_not_sufficient_but_necessary_l223_223273


namespace option_b_correct_l223_223280

-- Definitions of the conditions
def bag : ℕ × ℕ × ℕ := (3, 2, 1) -- (red, white, black)

def draw_two_balls_event_a (draw1 draw2 : nat) : Prop :=
  (draw1 = 1 ∨ draw2 = 1) ∧ (draw1 = 3 ∨ draw2 = 3) -- At least one white ball; At least one red ball

def draw_two_balls_event_b (draw1 draw2 : nat) : Prop :=
  (draw1 = 1 ∨ draw2 = 1) ∧ ((draw1 = 3 ∧ draw2 = 4) ∨ (draw1 = 4 ∧ draw2 = 3)) -- At least one white ball; One red ball and one black ball

def draw_two_balls_event_c (draw1 draw2 : nat) : Prop :=
  (draw1 = 1 ∧ draw2 ≠ 1) ∨ (draw2 = 1 ∧ draw1 ≠ 1) ∧ ((draw1 = 1 ∧ draw2 = 4) ∨ (draw1 = 4 ∧ draw2 = 1)) -- Exactly one white ball; One white ball and one black ball

def draw_two_balls_event_d (draw1 draw2 : nat) : Prop :=
  (draw1 = 1 ∨ draw2 = 1) ∧ (draw1 = 1 ∧ draw2 = 1) -- At least one white ball; Both are white balls

-- Proof goal definition
theorem option_b_correct :
  ∀ draw1 draw2 : nat,
  (draw_two_balls_event_b draw1 draw2 ∧
  (¬ (draw_two_balls_event_a draw1 draw2) ∧
  ¬ (draw_two_balls_event_c draw1 draw2) ∧
  ¬ (draw_two_balls_event_d draw1 draw2))) →
  true :=
sorry

end option_b_correct_l223_223280


namespace split_stones_l223_223076

theorem split_stones (n : ℕ) :
  ∃ (heaps : list ℕ), (∀ h ∈ heaps, 1 ≤ h ∧ h ≤ n) ∧ (∀ i j, i ≠ j → (i < heaps.length ∧ j < heaps.length → heaps.nth i ≤ 2 * heaps.nth j)) :=
sorry

end split_stones_l223_223076


namespace circumcenter_incenter_eq_l223_223286

-- Given definitions and conditions
variables {A B C I I_a I_b I_c A_1 B_1 C_1 A_2 B_2 C_2 : Type}
variables 
  (circ_in_A : IsInCircle I A_1 B_1 C_1) -- I incenter, touching BC, CA, AB at A_1, B_1, C_1
  (excircle_a : IsExCircle I_a B_1 C_1) -- I_a center of excircle touching BC at B_1
  (excircle_b : IsExCircle I_b A_1 C_1) -- I_b center of excircle touching CA at A_1
  (excircle_c : IsExCircle I_c A_1 B_1) -- I_c center of excircle touching AB at A_1
  (intersect_C2 : IntersectsAt I_a B_1 I_b A_1 C_2) -- I_aB_1 & I_bA_1 intersect at C_2
  (intersect_A2 : IntersectsAt I_b C_1 I_c B_1 A_2) -- I_bC_1 & I_cB_1 intersect at A_2
  (intersect_B2 : IntersectsAt I_c A_1 I_a C_1 B_2) -- I_cA_1 & I_aC_1 intersect at B_2

-- The theorem to be proved
theorem circumcenter_incenter_eq
  (equidistant : ∀ x, x ∈ {A_2, B_2, C_2} → dist I x = dist I A_2):
  IsCircumcenter I A_2 B_2 C_2 :=
sorry

end circumcenter_incenter_eq_l223_223286


namespace find_tan_and_cos_values_l223_223372

theorem find_tan_and_cos_values (a : ℝ) (α : ℝ)
  (ha : a ∈ Set.Icc (Real.pi / 2) Real.pi)
  (h1 : Vector.mk 2 (-1 : ℝ) = Vector.mk (Real.cos a) (Real.sin a))
  (h2 : 2 * Real.sin α = - Real.cos α): 
  (Real.tan (α + Real.pi / 4) = 1 / 3) ∧ 
  (Real.cos (5 * Real.pi / 6 - 2 * α) = - (4 + 3 * Real.sqrt 3) / 10) :=
sorry

end find_tan_and_cos_values_l223_223372


namespace number_of_real_solutions_l223_223208

noncomputable def f (x : ℝ) : ℝ := 2^(-x) + x^2 - 3

theorem number_of_real_solutions :
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧
  (∀ x : ℝ, f x = 0 → (x = x₁ ∨ x = x₂)) :=
by
  sorry

end number_of_real_solutions_l223_223208


namespace minimum_integer_lambda_l223_223778

def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 2^(x - 1) else 3 * x - 2

theorem minimum_integer_lambda : 
  ∃ λ : ℤ, (∀ θ ∈ set.Icc 0 (real.pi / 2), f ((real.cos θ)^2 + (λ : ℝ) * (real.sin θ) - (1/3)) + (1/2) > 0) ∧ ∀ λ' < λ, ∃ θ ∈ set.Icc 0 (real.pi / 2), f ((real.cos θ)^2 + (λ' : ℝ) * (real.sin θ) - (1/3)) + (1/2) ≤ 0 :=
begin
  use 1,
  sorry,
end

end minimum_integer_lambda_l223_223778


namespace split_into_similar_piles_l223_223090

def similar_sizes (x y : ℕ) : Prop := x ≤ 2 * y ∧ y ≤ 2 * x

theorem split_into_similar_piles (n : ℕ) (h : 0 < n) :
  ∃ (piles : list ℕ), (∀ x ∈ piles, x = 1) ∧ (list.sum piles = n) ∧
                       (∀ x y ∈ piles, similar_sizes x y) := 
sorry

end split_into_similar_piles_l223_223090


namespace log_property_l223_223335

noncomputable def f (x : ℝ) : ℝ := Real.log x

theorem log_property (m n : ℝ) (hm : 0 < m) (hn : 0 < n) : f (m * n) = f m + f n :=
by
  sorry

end log_property_l223_223335


namespace B_value_C_value_D_value_exists_t_value_l223_223644

variable (A : ℝ) (B : ℝ) (C : ℝ) (D : ℝ) (t : ℝ)

-- Given conditions as definitions
def pointA := -8
def pointB := pointA + 20
def pointC := (12 - C = C + 4 → C = 4)  -- point C condition
def pointD := (|D + 8| + |12 - D| = 25 → D = -10.5 ∨ D = 14.5)  -- point D condition
def moveCondition := (12 - 2 * t = -8 - t ∨ 12 - 2 * t = 8 + t → t = 4 / 3)  -- movement condition

-- Proof statements
theorem B_value : B = 12 :=
sorry

theorem C_value : pointC :=
sorry

theorem D_value : pointD :=
sorry

theorem exists_t_value : ∃ t, moveCondition :=
sorry

end B_value_C_value_D_value_exists_t_value_l223_223644


namespace sum_of_three_digits_eq_nine_l223_223963

def horizontal_segments (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | 1 => 0
  | 2 => 2
  | 3 => 3
  | 4 => 1
  | 5 => 2
  | 6 => 1
  | 7 => 1
  | 8 => 3
  | 9 => 2
  | _ => 0  -- Invalid digit

def vertical_segments (n : ℕ) : ℕ :=
  match n with
  | 0 => 4
  | 1 => 2
  | 2 => 3
  | 3 => 3
  | 4 => 3
  | 5 => 2
  | 6 => 3
  | 7 => 2
  | 8 => 4
  | 9 => 3
  | _ => 0  -- Invalid digit

theorem sum_of_three_digits_eq_nine :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
             (horizontal_segments a + horizontal_segments b + horizontal_segments c = 5) ∧ 
             (vertical_segments a + vertical_segments b + vertical_segments c = 10) ∧
             (a + b + c = 9) :=
sorry

end sum_of_three_digits_eq_nine_l223_223963


namespace can_construct_segment_l223_223843

noncomputable def constructSegment (Ω₁ Ω₂ : Set (ℝ × ℝ)) (P : ℝ × ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), A ∈ Ω₁ ∧ B ∈ Ω₂ ∧ (A + B) / 2 = P

theorem can_construct_segment (Ω₁ Ω₂ : Set (ℝ × ℝ)) (P : ℝ × ℝ) :
  (∃ A B : ℝ × ℝ, A ∈ Ω₁ ∧ B ∈ Ω₂ ∧ (A + B) / 2 = P) :=
sorry

end can_construct_segment_l223_223843


namespace gcd_2000_7700_l223_223624

theorem gcd_2000_7700 : Nat.gcd 2000 7700 = 100 := by
  -- Prime factorizations of 2000 and 7700
  have fact_2000 : 2000 = 2^4 * 5^3 := sorry
  have fact_7700 : 7700 = 2^2 * 5^2 * 7 * 11 := sorry
  -- Proof of gcd
  sorry

end gcd_2000_7700_l223_223624


namespace combine_heaps_l223_223126

def heaps_similar (x y : ℕ) : Prop :=
  x ≤ 2 * y ∧ y ≤ 2 * x

theorem combine_heaps (n : ℕ) : 
  ∃ f : ℕ → ℕ, 
  f 0 = n ∧
  ∀ k, k < n → (∃ i j, i + j = k ∧ heaps_similar (f i) (f j)) ∧ 
  (∃ k, f k = n) :=
by
  sorry

end combine_heaps_l223_223126


namespace circle_tangent_sum_l223_223322

noncomputable def tangent_length (A O : Point) (r : ℝ) : ℝ :=
  Real.sqrt (A.dist O ^ 2 - r ^ 2)

noncomputable def AB_AC_sum (A B C O : Point) (r BC : ℝ) : ℝ :=
  2 * (tangent_length A O r) + BC

theorem circle_tangent_sum (O A B C : Point) (r : ℝ) (OA : ℝ) (BC : ℝ) 
  (hO : dist O A = OA) (hBC : BC = 9) (hOA : OA = 10) (hR : r = 3) (h_exterior_triangle : true): 
  AB_AC_sum A B C O r BC = 2 * (Real.sqrt 91) + 9 :=
by
  sorry

end circle_tangent_sum_l223_223322


namespace f_zero_for_all_points_l223_223697

-- Define a function f which assigns a real number to each point in the plane
def f (A : ℝ × ℝ) : ℝ := sorry

-- Introduce the main condition: For any triangle ABC, if M is the centroid, then f(M) = f(A) + f(B) + f(C)
def centroid (A B C : ℝ × ℝ) : ℝ × ℝ := 
  ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

axiom centroid_property (A B C : ℝ × ℝ) : 
  f (centroid A B C) = f A + f B + f C

-- Now state the theorem: f(A) = 0 for all points A
theorem f_zero_for_all_points (A : ℝ × ℝ) :
  f A = 0 :=
sorry

end f_zero_for_all_points_l223_223697


namespace no_power_of_2_non_zero_rearrange_l223_223271

theorem no_power_of_2_non_zero_rearrange : 
  ¬(∃ (a b : ℕ), a ≠ b ∧ ∀ (d : ℕ), d ∈ digits 10 (2^a) → d ≠ 0 ∧ multiset.sort (digits 10 (2^a)) = multiset.sort (digits 10 (2^b))) :=
by
  sorry

end no_power_of_2_non_zero_rearrange_l223_223271


namespace max_value_expression_l223_223251

theorem max_value_expression (r : ℝ) : ∃ r : ℝ, -5 * r^2 + 40 * r - 12 = 68 ∧ (∀ s : ℝ, -5 * s^2 + 40 * s - 12 ≤ 68) :=
sorry

end max_value_expression_l223_223251


namespace combined_savings_l223_223539

def salary_per_hour : ℝ := 10
def daily_hours : ℝ := 10
def weekly_days : ℝ := 5
def robby_saving_ratio : ℝ := 2 / 5
def jaylene_saving_ratio : ℝ := 3 / 5
def miranda_saving_ratio : ℝ := 1 / 2
def weeks : ℝ := 4

theorem combined_savings 
  (sph : ℝ := salary_per_hour)
  (dh : ℝ := daily_hours)
  (wd : ℝ := weekly_days)
  (rr : ℝ := robby_saving_ratio)
  (jr : ℝ := jaylene_saving_ratio)
  (mr : ℝ := miranda_saving_ratio)
  (wk : ℝ := weeks) :
  (rr * (wk * wd * (dh * sph)) + jr * (wk * wd * (dh * sph)) + mr * (wk * wd * (dh * sph))) = 3000 :=
by
  sorry

end combined_savings_l223_223539


namespace vertex_of_parabola_l223_223573

theorem vertex_of_parabola (x y : ℝ) : y = (x - 1)^2 - 2 → ∃ h k : ℝ, h = 1 ∧ k = -2 ∧ y = (x - h)^2 + k :=
by
  intros h k
  existsi 1, -2
  split
  { refl }
  split
  { refl }
  { sorry }

end vertex_of_parabola_l223_223573


namespace cost_price_of_article_l223_223699

theorem cost_price_of_article 
  (SP : ℝ) 
  (profit_percentage : ℝ) 
  (h1 : SP = 500) 
  (h2 : profit_percentage = 0.25) :
  ∃ CP : ℝ, CP = 400 := 
by
  -- Define the cost price as CP
  let CP := SP / (1 + profit_percentage)
  
  -- Show that CP equals 400 when SP is 500 and profit_percentage is 0.25
  have h3 : CP = 500 / (1 + 0.25), from by
    rw [h1, h2]
  
  -- Simplify the expression
  have h4 : CP = 500 / 1.25, from by
    exact h3
  
  have h5 : CP = 400, from by
    exact h4

  -- Conclude that the cost price CP is 400
  use 400
  exact h5

end cost_price_of_article_l223_223699


namespace units_digit_7_power_6_squared_l223_223360

-- Define function for units digit
def units_digit (n : ℕ) : ℕ := n % 10

-- Main proof statement (theorem)
theorem units_digit_7_power_6_squared : units_digit (7 ^ (6 ^ 2)) = 1 :=
by
  -- The units digits of 7 cycles every 4: 7, 9, 3, 1
  -- 6^2 = 36 is a multiple of 4 so 7^36 ends at the same units digit as 7^4
  have cycle : ∀ (k : ℕ), k % 4 = 0 → units_digit (7^k) = 1 := by
    sorry -- Detailed proof of the repeating cycle would go here
  show units_digit (7 ^ (6 ^ 2)) = 1,
  from cycle (6 ^ 2) (by norm_num)
  sorry

end units_digit_7_power_6_squared_l223_223360


namespace greatest_possible_gcd_l223_223365

theorem greatest_possible_gcd (n : ℕ) (hn : 0 < n) :
  let T_n := n * (n + 1) / 2 in
  nat.gcd (6 * T_n) (n + 2) ≤ 6 :=
by
  sorry

end greatest_possible_gcd_l223_223365


namespace distinct_circles_arc_l223_223374

open Real

theorem distinct_circles_arc (n : ℕ) (h : n > 0) :
  ∃ c : ℝ × ℝ, ∃ arc, (arc.length ≥ 2 * π / n) ∧ ∀ c' ∈ {c' : ℝ × ℝ | c' ≠ c}, disjoint (arc.set) (circle c' 1) :=
sorry

end distinct_circles_arc_l223_223374


namespace exists_infinitely_many_m_property_P_l223_223798

noncomputable def sequence (a b : ℕ) : ℕ → ℕ
| 0     := a
| 1     := b
| (n+2) := sequence n.succ + sequence n

-- Prove that there exist infinitely many positive integers m such that
-- 1 + m * sequence(a, b, k) * sequence(a, b, k+2) is not a perfect square for all k
theorem exists_infinitely_many_m_property_P (a b : ℕ) (h1 : 0 < a) (h2 : a < b) :
  ∃ᶠ m in at_top, ∀ k : ℕ, ¬ ∃ x : ℕ, x * x = 1 + m * sequence a b k * sequence a b (k + 2) :=
sorry

end exists_infinitely_many_m_property_P_l223_223798


namespace arrange_MISSISSIPPI_l223_223735

theorem arrange_MISSISSIPPI : 
  let total_letters := 11
  let count_I := 4
  let count_S := 4
  let count_P := 2
  let arrangements := 11.factorial / (count_I.factorial * count_S.factorial * count_P.factorial)
  in arrangements = 34650 :=
by
  have h1 : 11.factorial = 39916800 := by sorry
  have h2 : 4.factorial = 24 := by sorry
  have h3 : 2.factorial = 2 := by sorry
  have h4 : arrangements = 34650 := by sorry
  exact h4

end arrange_MISSISSIPPI_l223_223735


namespace combine_heaps_l223_223131

def heaps_similar (x y : ℕ) : Prop :=
  x ≤ 2 * y ∧ y ≤ 2 * x

theorem combine_heaps (n : ℕ) : 
  ∃ f : ℕ → ℕ, 
  f 0 = n ∧
  ∀ k, k < n → (∃ i j, i + j = k ∧ heaps_similar (f i) (f j)) ∧ 
  (∃ k, f k = n) :=
by
  sorry

end combine_heaps_l223_223131


namespace long_knight_min_moves_l223_223651

-- Define the position of the knight on the board
structure Position where
  x : ℕ
  y : ℕ

-- Define the movement constraints of a long knight
def longKnightMoves (p q : Position) : Prop :=
  (abs (p.x - q.x) = 3 ∧ abs (p.y - q.y) = 1) ∨ (abs (p.x - q.x) = 1 ∧ abs (p.y - q.y) = 3)

-- Define a function to check if a sequence of moves is valid
def validMoves (start : Position) (moves : List Position) : Prop :=
  match moves with
  | [] => true
  | h :: t => longKnightMoves start h ∧ validMoves h t

-- Define the starting and target positions
def start : Position := ⟨0, 0⟩
def target : Position := ⟨7, 7⟩

-- Define the Lean statement for the proof problem
theorem long_knight_min_moves :
  ∃ moves : List Position, length moves = 5 ∧ validMoves start moves ∧ List.getLast (start :: moves) (by simp) = target :=
sorry

end long_knight_min_moves_l223_223651


namespace mona_biked_miles_on_monday_l223_223934

theorem mona_biked_miles_on_monday :
  ∀ (weekly_distance miles_on_wednesday total_distance_mon_sat distance_on_sat : ℕ),
    weekly_distance = 30 →
    miles_on_wednesday = 12 →
    total_distance_mon_sat = weekly_distance - miles_on_wednesday →
    distance_on_sat = 2 * (total_distance_mon_sat / 3) →
    (total_distance_mon_sat - distance_on_sat) = 6 :=
by
  intros weekly_distance miles_on_wednesday total_distance_mon_sat distance_on_sat
  intros h_weekly_distance h_miles_on_wednesday h_total_distance_mon_sat h_distance_on_sat
  rw [h_weekly_distance, h_miles_on_wednesday, h_total_distance_mon_sat, h_distance_on_sat]
  sorry

end mona_biked_miles_on_monday_l223_223934


namespace complex_number_in_third_quadrant_l223_223444

theorem complex_number_in_third_quadrant :
  ∀ z : ℂ, z = (2 + 1*I) / (I^5 - 1) → (z.re < 0 ∧ z.im < 0) :=
by
  intros z hz
  have h1 : I^5 = I := by sorry
  have h2 : z = (2 + I) / (I - 1) := by
    rw h1 at hz
    exact hz
  have h3 : z = (1 / 2 + (3 / 2) * I) := by sorry
  rw h2 at h3
  exact sorry

end complex_number_in_third_quadrant_l223_223444


namespace isosceles_triangle_vertex_angle_l223_223213

theorem isosceles_triangle_vertex_angle (a b : ℕ) (h : a = 2 * b) 
  (h1 : a + b + b = 180): a = 90 ∨ a = 36 :=
by
  sorry

end isosceles_triangle_vertex_angle_l223_223213


namespace inverse_B_squared_l223_223392

theorem inverse_B_squared :
  let B_inv := Matrix.of ![![1, 4], ![-2, -7]] in
  (B_inv * B_inv) = (Matrix.of ![![ -7, -24], ![12, 41]]) ->
  Matrix.inverse (B_inv⁻¹ * B_inv⁻¹) = Matrix.of ![![ -7, -24], ![12, 41]] :=
by
  intros
  sorry

end inverse_B_squared_l223_223392


namespace toucans_total_l223_223649

theorem toucans_total (initial_toucans : ℕ) (joined_toucans : ℕ) (total_toucans : ℕ) :
  initial_toucans = 2 → joined_toucans = 1 → total_toucans = 3 :=
by
  intros h_init h_joined
  rw [h_init, h_joined]
  exact rfl

end toucans_total_l223_223649


namespace length_of_KL_l223_223789

-- Define the conditions of the problem
variables {A B C D E K L : Type} [metric_space (A B C D E)]
variable [has_dist A B]
variables (BC EP : ℝ) -- EP is the height h from E to AB
variable {AK BL : ℝ}
variables (sqrt2 : ℝ) (AB BC : ℝ)

axiom rectangle_property : AB = sqrt2 * BC
axiom point_E_on_semicircle : ∃ (E : Type), True -- arbitrary existence, since details are abstract
axiom AK_value : AK = 2
axiom BL_value : BL = 9

-- Define the proof statement
theorem length_of_KL : true :=
by
  -- leave the actual proof as sorry
  sorry

end length_of_KL_l223_223789


namespace three_teams_no_match_l223_223038

theorem three_teams_no_match (num_teams num_rounds played_pairs : ℕ)
  (h_num_teams : num_teams = 18)
  (h_num_rounds : num_rounds = 8)
  (h_max_pairs_per_team : ∀ team, played_pairs ≤ num_rounds)
  (h_unique_pairs : ∀ a b c, (a ≠ b ∧ b ≠ c ∧ a ≠ c) → ¬ (a, b) ∈ played_pairs ∧ (b, c) ∈ played_pairs ∧ (a, c) ∈ played_pairs) :
  ∃ A B C, A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ ¬ (A, B) ∈ played_pairs ∧ ¬ (B, C) ∈ played_pairs ∧ ¬ (A, C) ∈ played_pairs :=
sorry

end three_teams_no_match_l223_223038


namespace simplify_expression_l223_223175

theorem simplify_expression (y : ℝ) : (3 * y^4)^5 = 243 * y^20 :=
sorry

end simplify_expression_l223_223175


namespace minimum_value_9_pow_x_plus_3_pow_y_l223_223423

theorem minimum_value_9_pow_x_plus_3_pow_y (x y : ℝ) (h : (x - 1) * 4 + 2 * y = 0) : 9^x + 3^y = 10 :=
by
  sorry

end minimum_value_9_pow_x_plus_3_pow_y_l223_223423


namespace polar_to_cartesian_l223_223401

theorem polar_to_cartesian (rho theta : ℝ) (x y : ℝ) 
  (h_rho : rho = 8 * cos theta - 6 * sin theta)
  (h_x : x = rho * cos theta)
  (h_y : y = rho * sin theta) : 
  x^2 + y^2 - 8 * x + 6 * y = 0 :=
sorry

end polar_to_cartesian_l223_223401


namespace sum_first_five_terms_arithmetic_sequence_l223_223606

theorem sum_first_five_terms_arithmetic_sequence (a d : ℤ)
  (h1 : a + 5 * d = 10)
  (h2 : a + 6 * d = 15)
  (h3 : a + 7 * d = 20) :
  5 * (2 * a + (5 - 1) * d) / 2 = -25 := by
  sorry

end sum_first_five_terms_arithmetic_sequence_l223_223606


namespace simplest_form_is_C_l223_223262

variables (x y : ℝ) (hx : x ≠ 0) (hx1 : x ≠ 1) (hy : y ≠ 0)

def fraction_A := 3 * x * y / (x^2)
def fraction_B := (x - 1) / (x^2 - 1)
def fraction_C := (x + y) / (2 * x)
def fraction_D := (1 - x) / (x - 1)

theorem simplest_form_is_C : 
  ∀ (x y : ℝ) (hx : x ≠ 0) (hx1 : x ≠ 1) (hy : y ≠ 0), 
  ¬ (3 * x * y / (x^2)).is_simplest ∧ 
  ¬ ((x - 1) / (x^2 - 1)).is_simplest ∧ 
  (x + y) / (2 * x).is_simplest ∧ 
  ¬ ((1 - x) / (x - 1)).is_simplest :=
by 
  sorry

end simplest_form_is_C_l223_223262


namespace reciprocal_of_neg_2023_l223_223980

theorem reciprocal_of_neg_2023 : 1 / (-2023) = -1 / 2023 :=
by
  sorry

end reciprocal_of_neg_2023_l223_223980


namespace multiples_of_2_correct_multiples_of_3_correct_l223_223259

def numbers : Set ℕ := {28, 35, 40, 45, 53, 10, 78}

def multiples_of_2_in_numbers : Set ℕ := {n ∈ numbers | n % 2 = 0}
def multiples_of_3_in_numbers : Set ℕ := {n ∈ numbers | n % 3 = 0}

theorem multiples_of_2_correct :
  multiples_of_2_in_numbers = {28, 40, 10, 78} :=
sorry

theorem multiples_of_3_correct :
  multiples_of_3_in_numbers = {45, 78} :=
sorry

end multiples_of_2_correct_multiples_of_3_correct_l223_223259


namespace probability_A_probability_B_given_notC_l223_223666

-- Define events A, B, and C as predicates
def event_A (tosses : List Bool) : Prop :=
  (tosses.count (λ b => b) > 0) ∧ (tosses.count (λ b => ¬b) > 0)

def event_B (tosses : List Bool) : Prop :=
  tosses.count (λ b => b) ≤ 1

def event_C (tosses : List Bool) : Prop :=
  tosses.count (λ b => b) = 0

-- Define the probability of an event given a certain set of outcomes
def probability (event : List Bool → Prop) (outcomes : List (List Bool)) :=
  (outcomes.filter event).length.toRat / outcomes.length.toRat

def complement (event : List Bool → Prop) (outcomes : List (List Bool)) :=
  outcomes.filter (λ outcome => ¬ event outcome)

-- Compute the list of all possible outcomes for 3 coin tosses
def all_outcomes : List (List Bool) :=
  [ [ff, ff, ff], [ff, ff, tt], [ff, tt, ff], [ff, tt, tt],
    [tt, ff, ff], [tt, ff, tt], [tt, tt, ff], [tt, tt, tt] ]

-- Definitions of events
def A := event_A
def B := event_B
def C := event_C

-- P(A) = 3 / 4
theorem probability_A :
  probability event_A all_outcomes = 3 / 4 :=
sorry

-- P(B|Cᶜ) = 3 / 7
theorem probability_B_given_notC :
  let notC := complement event_C all_outcomes in
  probability event_B notC = 3 / 7 :=
sorry

end probability_A_probability_B_given_notC_l223_223666


namespace identify_minor_premise_l223_223794

def is_decreasing_function (a : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

def is_valid_minor_premise : Prop :=
  ∀ a : ℝ,
    (0 < a ∧ a = 1/2 ∧ a < 1) →
    (0 < a ∧ a < 1 → is_decreasing_function a (λ x, a^x)) →
    0 < a ∧ a = 1/2 ∧ a < 1

theorem identify_minor_premise : is_valid_minor_premise :=
  sorry

end identify_minor_premise_l223_223794


namespace unique_ordered_triple_l223_223848

theorem unique_ordered_triple :
  {a b c : ℤ // 
    a ≥ 2 ∧ b ≥ 1 ∧ c ≥ 0 ∧ 
    (log a b = c^3) ∧ 
    (a + b + c = 30)} → unique
  :=
by sorry

end unique_ordered_triple_l223_223848


namespace number_of_k_solutions_l223_223336

theorem number_of_k_solutions : 
  (Finset.card {k | ∃ x : ℤ, 3 * k * x - 54 = 9 * k}.to_finset) = 6 := 
by
  sorry

end number_of_k_solutions_l223_223336


namespace symmetric_about_line_l223_223020

theorem symmetric_about_line (m n : ℝ) :
  let P := (m, n)
  let Q := (n - 1, m + 1)
  ∃ l : ℝ × ℝ → Prop,
    is_symmetric_about_line P Q l ∧
    (l = (λ (x : ℝ), x - y + 1 = 0)) :=
by
  sorry

end symmetric_about_line_l223_223020


namespace integral_of_sqrt_difference_squared_l223_223446

theorem integral_of_sqrt_difference_squared :
  ∫ x in -5..5, sqrt (25 - x^2) = (25 * Real.pi) / 2 :=
sorry

end integral_of_sqrt_difference_squared_l223_223446


namespace additional_people_needed_l223_223342

theorem additional_people_needed:
  ∀ (total_people : ℕ) (hours : ℕ) (efficiency : ℚ) (required_hours : ℕ) (additional_people : ℕ),
  total_people = 8 →
  hours = 3 →
  efficiency = 0.9 →
  required_hours = 2 →
  additional_people = (ceil ((total_people * hours) / (efficiency * required_hours)) - total_people) →
  additional_people = 6 :=
begin
  intros,
  rw [h, h_1, h_2, h_3],
  have : ceil ((8 * 3) / (0.9 * 2)) = 14,
  { norm_num },
  rw this,
  simp,
end

end additional_people_needed_l223_223342


namespace perm_mississippi_l223_223737

theorem perm_mississippi : 
  let n := 11
  let n_S := 4
  let n_I := 4
  let n_P := 2
  let n_M := 1
  ∏ (mississippi_perm := Nat.factorial n / (Nat.factorial n_S * Nat.factorial n_I * Nat.factorial n_P * Nat.factorial n_M)) :=
  mississippi_perm = 34650 :=
by sorry

end perm_mississippi_l223_223737


namespace perm_mississippi_l223_223738

theorem perm_mississippi : 
  let n := 11
  let n_S := 4
  let n_I := 4
  let n_P := 2
  let n_M := 1
  ∏ (mississippi_perm := Nat.factorial n / (Nat.factorial n_S * Nat.factorial n_I * Nat.factorial n_P * Nat.factorial n_M)) :=
  mississippi_perm = 34650 :=
by sorry

end perm_mississippi_l223_223738


namespace tangent_line_perpendicular_to_yaxis_l223_223022

noncomputable def func (a x : ℝ) : ℝ := a * x^3 + Real.log x

theorem tangent_line_perpendicular_to_yaxis (a : ℝ) :
  (∃ x > 0, Deriv.deriv (func a) x = 0) → a < 0 :=
by
  sorry

end tangent_line_perpendicular_to_yaxis_l223_223022


namespace number_of_factors_of_expr_l223_223182

open Nat

theorem number_of_factors_of_expr (a b c : ℕ) (ha : ∃ p1 : ℕ, prime p1 ∧ a = p1^2) (hb : ∃ p2 : ℕ, prime p2 ∧ b = p2^2) (hc : ∃ p3 : ℕ, prime p3 ∧ c = p3^2) (habc_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  num_divisors (a^3 * b^4 * c^5) = 693 :=
sorry

end number_of_factors_of_expr_l223_223182


namespace range_of_a_specific_a_l223_223881

open Real

-- Definitions based on problem conditions
def polar_eq_curve (ρ θ : ℝ) : Prop := ρ = 2 * cos θ
def line_eq (t a : ℝ) : ℝ × ℝ := (1/2 * t + a, sqrt 2 / 4 * t)
def point_M (a : ℝ) : ℝ × ℝ := (a, 0)

-- Helper definition for Cartesian conversion of polar and line equations
def curve_eq (x y : ℝ) : Prop := (x - 1) ^ 2 + y ^ 2 = 1
def line_eq_cartesian (x y a : ℝ) : Prop := x - sqrt 2 * y - a = 0

-- Proof for part (1): range of a
theorem range_of_a (a : ℝ) : (1 - sqrt 3 < a) ∧ (a < 1 + sqrt 3) :=
sorry

-- Proof for part (2): specific value of a given MA * MB = 5/4
theorem specific_a (a : ℝ) (t₁ t₂ : ℝ) 
  (h1 : curve_eq (1/2 * t₁ + a) (sqrt 2 / 4 * t₁))
  (h2 : curve_eq (1/2 * t₂ + a) (sqrt 2 / 4 * t₂))
  (h3 : point_M a = (a, 0))
  (intersect_points : |MA| * |MB| = 5/4) 
  : a = -1/2 ∨ a = 5/2 :=
sorry

end range_of_a_specific_a_l223_223881


namespace total_applicants_l223_223231

-- Definitions as per conditions
def A : Set ℕ := {n | n < 15}  -- Placeholder for applicants who majored in political science
def B : Set ℕ := {n | n < 20}  -- Placeholder for applicants with GPA higher than 3.0
def A_cap_B : Set ℕ := {n | n < 5} -- Placeholder for applicants who majored in political science and had GPA > 3.0
def C : Set ℕ := {n | n < 10} -- Placeholder for applicants who did not major in political science and had GPA ≤ 3.0

-- The Lean 4 statement
theorem total_applicants (hA : A.card = 15) (hB : B.card = 20) (hA_cap_B : (A ∩ B).card = 5) (hC : C.card = 10) :
  (A.card + B.card - (A ∩ B).card + C.card) = 45 := by
  sorry

end total_applicants_l223_223231


namespace multiplier_for_doberman_puppies_l223_223556

theorem multiplier_for_doberman_puppies 
  (D : ℕ) (S : ℕ) (M : ℝ) 
  (hD : D = 20) 
  (hS : S = 55) 
  (h : D * M + (D - S) = 90) : 
  M = 6.25 := 
by 
  sorry

end multiplier_for_doberman_puppies_l223_223556


namespace find_f_9_over_2_l223_223928

noncomputable def f (x : ℝ) : ℝ := sorry

axiom domain_of_f : ∀ x : ℝ, ∃ f(x) -- The domain of f(x) is ℝ

axiom odd_f_shift : ∀ x : ℝ, f(x + 1) = -f(-x + 1) -- f(x+1) is an odd function

axiom even_f_shift : ∀ x : ℝ, f(x + 2) = f(-x + 2) -- f(x+2) is an even function

axiom function_segment : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f(x) = a * x^2 + b -- For x in [1,2], f(x) = ax^2 + b

axiom sum_f0_f3 : f(0) + f(3) = 6 -- Given f(0) + f(3) = 6

theorem find_f_9_over_2 : f (9 / 2) = 5 / 2 := sorry

end find_f_9_over_2_l223_223928


namespace find_B_l223_223431

theorem find_B (A B : Nat) (hA : A ≤ 9) (hB : B ≤ 9) (h_eq : 6 * A + 10 * B + 2 = 77) : B = 1 :=
by
-- proof steps would go here
sorry

end find_B_l223_223431


namespace cos_pi_div_four_minus_alpha_l223_223370

theorem cos_pi_div_four_minus_alpha (α : ℝ) (h : Real.sin (π / 4 + α) = 2 / 3) : 
    Real.cos (π / 4 - α) = -Real.sqrt 5 / 3 :=
sorry

end cos_pi_div_four_minus_alpha_l223_223370


namespace a_10_value_l223_223447

def sequence (n : ℕ) : ℚ :=
  (-1)^n * (1 / (2 * n + 1))

theorem a_10_value : sequence 10 = 1 / 21 :=
  sorry

end a_10_value_l223_223447


namespace arrange_MISSISSIPPI_l223_223732

theorem arrange_MISSISSIPPI : 
  let total_letters := 11
  let count_I := 4
  let count_S := 4
  let count_P := 2
  let arrangements := 11.factorial / (count_I.factorial * count_S.factorial * count_P.factorial)
  in arrangements = 34650 :=
by
  have h1 : 11.factorial = 39916800 := by sorry
  have h2 : 4.factorial = 24 := by sorry
  have h3 : 2.factorial = 2 := by sorry
  have h4 : arrangements = 34650 := by sorry
  exact h4

end arrange_MISSISSIPPI_l223_223732


namespace Sandy_phone_bill_expense_l223_223166
noncomputable def Sandy_age_now : ℕ := 34
noncomputable def Kim_age_now : ℕ := 10
noncomputable def Sandy_phone_bill : ℕ := 10 * Sandy_age_now

theorem Sandy_phone_bill_expense :
  (Sandy_age_now - 2 = 36 - 2) ∧ (Kim_age_now + 2 = 12) ∧ (36 = 3 * 12) ∧ (Sandy_phone_bill = 340) := by
sorry

end Sandy_phone_bill_expense_l223_223166


namespace roots_situation_l223_223605

-- Define the quadratic equation coefficients
def a : ℝ := 1
def b : ℝ := 3
def c : ℝ := -2

-- Define the equation
def quadratic_eq (x : ℝ) : ℝ := a * x ^ 2 + b * x + c

-- Calculate the discriminant
def discriminant (a b c : ℝ) : ℝ := b ^ 2 - 4 * a * c

-- Statement to prove the roots situation
theorem roots_situation :
  let Δ := discriminant a b c in
  Δ = 17 ∧ Δ > 0 :=
by
  sorry

end roots_situation_l223_223605


namespace men_in_group_l223_223188

theorem men_in_group (A : ℝ) (n : ℕ) (h : n > 0) 
  (inc_avg : ↑n * A + 2 * 32 - (21 + 23) = ↑n * (A + 1)) : n = 20 :=
sorry

end men_in_group_l223_223188


namespace lines_intersect_at_single_point_l223_223385

-- Given a triangle ABC and lines x, y, z through vertices A, B, and C intersecting at point S
variables {A B C S A' B' C' : Type}
variables {x y z : Type} -- Lines through vertices intersecting at S
variables (a b c : Type)  -- Sides of the triangle
variables (a' b' c' : Type) -- Lines parallel to x, y, z creating another triangle A'B'C'
variables (x' y' z' : Type) -- Lines parallel to a, b, c through A', B', C'

-- Definition that lines x', y', z' intersect at a single point
theorem lines_intersect_at_single_point 
  (h1 : x.starts_at A)
  (h2 : y.starts_at B)
  (h3 : z.starts_at C)
  (h4 : x.meet y = S)
  (h5 : y.meet z = S)
  (h6 : z.meet x = S)
  (h7 : a.parallel x)
  (h8 : b.parallel y)
  (h9 : c.parallel z)
  (h10 : a'.parallel x)
  (h11 : b'.parallel y)
  (h12 : c'.parallel z)
  (h13 : x'.parallel a)
  (h14 : y'.parallel b)
  (h15 : z'.parallel c)
  : 
  x'.meet y' = z'.meet x' :=
sorry


end lines_intersect_at_single_point_l223_223385


namespace train_pass_platform_time_approx_l223_223686

-- Given conditions
def train_speed_kmh : ℝ := 54  -- Speed in km/hr
def platform_length : ℝ := 180.0144  -- Length of the platform in meters
def pass_man_time : ℝ := 20  -- Time to pass a man in seconds

-- Derived quantities
def train_speed_ms : ℝ := (train_speed_kmh * 1000) / 3600  -- Speed in m/s
def train_length : ℝ := train_speed_ms * pass_man_time  -- Length of the train in meters

-- Total distance to cover when passing the platform
def total_distance : ℝ := train_length + platform_length

-- Time taken to pass the platform
def pass_platform_time : ℝ := total_distance / train_speed_ms

theorem train_pass_platform_time_approx :
  pass_platform_time ≈ 32 :=
by
  sorry

end train_pass_platform_time_approx_l223_223686


namespace max_subset_size_l223_223509

open Finset

theorem max_subset_size (n : ℕ) (M : Finset ℕ := range (2 * n + 2)) (A : Finset ℕ)
  (hA : ∀ a b ∈ A, a + b ≠ 2 * n + 2) : A.card ≤ n + 1 :=
sorry

end max_subset_size_l223_223509


namespace exponent_multiplication_l223_223648

theorem exponent_multiplication :
  (10 ^ 10000) * (10 ^ 8000) = 10 ^ 18000 :=
by
  sorry

end exponent_multiplication_l223_223648


namespace inclination_angle_range_l223_223211

theorem inclination_angle_range (α : ℝ) (x y : ℝ) :
  (∃ k : ℝ, k = -sin α ∧ k ∈ [-1, 1] ∧ (∃ γ : ℝ, tan γ = k ∧ γ ∈ [0, π))) →
  (∃ θ : ℝ, θ ∈ [0, π) ∧ (θ ∈ [0, π / 4] ∨ θ ∈ [3 * π / 4, π])) :=
sorry

end inclination_angle_range_l223_223211


namespace smallest_angle_triangle_l223_223461

theorem smallest_angle_triangle 
  (a b c : ℝ)
  (h1 : 3 * a < b + c)
  (h2 : a + b > c)
  (h3 : a + c > b) : 
  ∠BAC < ∠ABC ∧ ∠BAC < ∠ACB :=
by 
  sorry

end smallest_angle_triangle_l223_223461


namespace sequence_divisibility_l223_223791

theorem sequence_divisibility :
  ∃ n, n ≥ 1 ∧ 
  (∀ (a : ℕ → ℕ), 
   a 1 = 1 → 
   (∀ k, k ≥ 1 → a (k + 1) = 2 * (∑ i in Finset.range (k + 1), a (i + 1))) → 
   (a n) % (3^2017) = 0 ∧ n = 2019)
:= sorry

end sequence_divisibility_l223_223791


namespace diagonal_cubes_l223_223650

theorem diagonal_cubes (a b c : ℕ) (ha : a = 120) (hb : b = 280) (hc : c = 360) :
  let g := Nat.gcd (Nat.gcd a b) c
  in a + b + c - (Nat.gcd a b + Nat.gcd b c + Nat.gcd c a) + g = 690 :=
by
  sorry

end diagonal_cubes_l223_223650


namespace no_n_satisfies_condition_l223_223507

def g (n : ℤ) : ℤ :=
if n % 2 = 1 then n^3 + 9 else n / 2

theorem no_n_satisfies_condition : 
  ∀ n, -100 ≤ n ∧ n ≤ 100 → ∃ m, g^[m] (n) = 9 ↔ false :=
by
  intros n hn
  use ∃ m, g^[m] (n) = 9
  sorry

end no_n_satisfies_condition_l223_223507


namespace reciprocal_of_neg_2023_l223_223979

theorem reciprocal_of_neg_2023 : (1 : ℝ) / (-2023) = -(1 / 2023) :=
by 
  sorry

end reciprocal_of_neg_2023_l223_223979


namespace minimum_chord_length_l223_223375

noncomputable def circleEquation (x y : ℝ) : Prop :=
    (x - 3)^2 + y^2 = 25

noncomputable def lineEquation (m x y : ℝ) : Prop :=
    (m + 1) * x + (m - 1) * y - 2 = 0

theorem minimum_chord_length (m : ℝ) :
    ∃ (p : ℝ), circleEquation p 0 ∧ lineEquation m p 0 ∧ 
    ∀ (chord_len : ℝ), chord_len >= 4 * real.sqrt 5 :=
sorry

end minimum_chord_length_l223_223375


namespace smallest_positive_period_of_f_intervals_where_f_monotonically_increasing_l223_223411

noncomputable def f (x : ℝ) : ℝ := 5 * sin x * cos x - 5 * sqrt 3 * (cos x)^2 + 5 * sqrt 3 / 2

theorem smallest_positive_period_of_f :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧ T = π :=
sorry

theorem intervals_where_f_monotonically_increasing :
  ∀ k : ℤ, ∀ x : ℝ, (-(π / 12 : ℝ) + k * π ≤ x ∧ x ≤ 5 * π / 12 + k * π) →
    (∀ x', x ≤ x' → x' ≤ 5 * π / 12 + k * π → deriv f x' ≥ 0) :=
sorry

end smallest_positive_period_of_f_intervals_where_f_monotonically_increasing_l223_223411


namespace propositions_truth_l223_223716

theorem propositions_truth :
    (¬ (∀ (a b : ℝ), a ≤ b → a < b) = false) ∧
    (∀ (a : ℝ), a = 1 → (∀ (x : ℝ), a*x^2 - x + 3 ≥ 0)) ∧
    (∀ (r s : ℝ), 2 * π * r = 2 * π * s → π * r^2 = π * s^2) ∧
    (¬ (∀ (x : ℝ), rational (√2 * x) → ¬ irrational x) = false) →
    correct_answer = "B: ①②③" := 
by 
  intros _ 
  sorry

end propositions_truth_l223_223716


namespace ratio_of_segments_l223_223576

theorem ratio_of_segments (p q : ℕ) (p_prime : Nat.coprime p q)
  (h : y = sin x → intersects y = sin (π / 3) at points (x₁ := π / 6 + 2 * π * n, x₂ := 5 * π / 6 + 2 * π * n)) :
  p = 1 ∧ q = 2 :=
by
  -- Use the conditions to define the intersection points
  let x1 := π / 6 + 2 * π * n
  let x2 := 5 * π / 6 + 2 * π * n
  have : x2 - x1 = 2 * π / 3 :=
    calc 5 * π / 6 + 2 * π * n - (π / 6 + 2 * π * n) = 5 * π / 6 - π / 6
  -- Calculate the ratio of the lengths
  have ratio := (x2 - x1) / ((π / 6 + 2 * π * n) - 2 * ((5 * π / 6 + 2 * π * n)))
  -- Simplify the ratio to be 1 / 2
  have simp_ratio : ratio = 1 / 2 := sorry
  -- Conclude the ratio of segments is 1:2
  exact ⟨1, 2⟩

end ratio_of_segments_l223_223576


namespace max_consecutive_divisible_by_m_l223_223921

noncomputable def sequence (m : ℕ) (i : ℕ) : ℕ :=
if h : i < m then 2^i
else ∑ j in finset.range m, sequence m (i - j - 1)

theorem max_consecutive_divisible_by_m (m : ℕ) (hm : m > 1) :
  ∃ (k : ℕ), (∀ i, (sequence m i) % m = 0 → k ≤ m-1) ∧
              (∃ j, (∀ l, l < k ↔ (sequence m (j + l)) % m = 0)) :=
sorry

end max_consecutive_divisible_by_m_l223_223921


namespace compare_a_b_c_l223_223371

def a : ℝ := 2^(-1/3 : ℝ)
noncomputable def b : ℝ := Real.log 5 / Real.log 3
noncomputable def c : ℝ := Real.cos (100 * Real.pi / 180)

theorem compare_a_b_c : b > a ∧ a > c :=
by
  have a_pos : 0 < a := Real.rpow_pos_of_pos (Nat.cast_pos.2 (show 0 < 2 by norm_num)) _
  have a_lt_1 : a < 1 := by sorry   -- (Actually calculate or infer)
  have b_gt_1 : b > 1 := by sorry    -- (Actually calculate or infer)
  have c_neg : c < 0 := by sorry     -- (Actually calculate or infer)
  exact ⟨
    by
      calc
        b > 1 : b_gt_1
        … > a : by exact a_lt_1,
    by
      calc
        a > 0 : a_pos
        … > c : by exact c_neg⟩

end compare_a_b_c_l223_223371


namespace triangle_side_angle_solution_l223_223180

noncomputable def sin_deg_min_sec (d : ℚ) (m : ℚ) (s : ℚ) : ℝ :=
  Real.sin ((d + m / 60 + s / 3600) * Real.pi / 180)

theorem triangle_side_angle_solution
  (C : ℚ) (c : ℝ) (p : ℝ) (A_expected : ℚ) (B_expected : ℚ) (a_expected : ℝ) (b_expected : ℝ)
  (C_d : ℚ := 82, C_m : ℚ := 49, C_s : ℚ := 9.2)
  (A_d : ℚ := 41, A_m : ℚ := 24, A_s : ℚ := 34.0)
  (B_d : ℚ := 55, B_m : ℚ := 46, B_s : ℚ := 16.8) 
  (C_val : C = 82 + 49 / 60 + 9.2 / 3600 := by norm_num)
  (A_val : A_expected = 41 + 24 / 60 + 34.0 / 3600 := by norm_num)
  (B_val : B_expected = 55 + 46 / 60 + 16.8 / 3600 := by norm_num) : Prop :=
  
  let sinC := sin_deg_min_sec 82 49 9.2 in
  let sinA := sin_deg_min_sec 41 24 34.0 in
  let sinB := sin_deg_min_sec 55 46 16.8 in 
  let a := c * sinA / sinC in
  let b := c * sinB / sinC in 
   
  (A_expected = 41 + 24 / 60 + 34.0 / 3600) ∧
  (B_expected = 55 + 46 / 60 + 16.8 / 3600) ∧
  (a ≈ 50) ∧
  (b ≈ 62.5) ∧ 
  (Real.abs ((b * b) - (a * a) - p^2) < 1e-6) 

end triangle_side_angle_solution_l223_223180


namespace find_a_sign_g_l223_223824

-- Definition of the function f and the condition of tangency
def f (x : ℝ) (a : ℝ) : ℝ := x + a * Real.log x

-- Proof problem 1: Prove that a = 2
theorem find_a (a : ℝ) (h : f 1 a = 1 + a) (tangency : (∀ x, 3 * x - 2 = f x a → f' x = 1 + a)) : a = 2 :=
sorry

-- Definition of the function g and its derivative
def g (x : ℝ) (a k : ℝ) : ℝ := x + a * Real.log x - k * x^2
def g' (x : ℝ) (a k : ℝ) : ℝ := 1 + a / x - 2 * k * x

-- Proof problem 2: Determine the sign of g'((x1 + x2) / 2)
theorem sign_g' (a k x1 x2 : ℝ) (h1 : g x1 a k = 0) (h2 : g x2 a k = 0) :
  ((a > 0 → g' ((x1 + x2) / 2) a k < 0) ∧ (a < 0 → g' ((x1 + x2) / 2) a k > 0)) :=
sorry

end find_a_sign_g_l223_223824


namespace total_pages_l223_223562

theorem total_pages (history_pages geography_additional math_factor science_factor : ℕ) 
  (h1 : history_pages = 160)
  (h2 : geography_additional = 70)
  (h3 : math_factor = 2)
  (h4 : science_factor = 2) 
  : let geography_pages := history_pages + geography_additional in
    let sum_history_geography := history_pages + geography_pages in
    let math_pages := sum_history_geography / math_factor in
    let science_pages := history_pages * science_factor in
    history_pages + geography_pages + math_pages + science_pages = 905 :=
by
  sorry

end total_pages_l223_223562


namespace graph_symmetric_y_axis_l223_223199

theorem graph_symmetric_y_axis (x : ℝ) : 
  let f : ℝ → ℝ := λ x, 2^x + 2^(-x)
  in f(-x) = f(x) :=
by
  sorry

end graph_symmetric_y_axis_l223_223199


namespace leading_digits_aperiodic_l223_223941

def leading_digits (x : ℝ) : ℕ :=
  let m := 10 ^ (Real.log10 x - Real.floor (Real.log10 x))
  if Real.floor m < m then Real.floor m else 9

def sequence_2_pow_2_pow (n : ℕ) : ℝ := 2 ^ (2 ^ n)

def leading_digits_sequence (n : ℕ) : ℕ := leading_digits (sequence_2_pow_2_pow n)

theorem leading_digits_aperiodic : ¬ ∃ (k : ℕ), ∀ (m n : ℕ), leading_digits_sequence m = leading_digits_sequence (m + k)
:= sorry

end leading_digits_aperiodic_l223_223941


namespace weight_of_square_proof_l223_223677

noncomputable def weight_of_square (length_rect : ℝ) (width_rect : ℝ) (weight_rect : ℝ) (side_square : ℝ) : ℝ :=
  let area_rect := length_rect * width_rect
  let area_square := side_square^2
  let weight_square := weight_rect * area_square / area_rect
  weight_square

theorem weight_of_square_proof :
  weight_of_square 4 6 20 8 = 53.3 :=
by
  calc
    let area_rect := 4 * 6
    let area_square := 8^2
    let weight_square := 20 * area_square / area_rect
    weight_square = 20 * 64 / 24 := rfl
    _ = 53.3 := by norm_num

end weight_of_square_proof_l223_223677


namespace gcd_lcm_product_l223_223356

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 24) (h2 : b = 60) :
  Nat.gcd a b * Nat.lcm a b = 1440 :=
by
  sorry

end gcd_lcm_product_l223_223356


namespace trapezoid_property_l223_223053

-- Defining the circle k centered at point M with a tangent line t at point T.
variables (k : Circle) (M T P : Point)
variables (t : Line) (h_tangent : tangent_at T k t)

-- Point P is on the tangent line t.
variable (hP_on_t : P ∈ t)

-- Line g through P intersects k at U and V.
variables (g : Line) (U V : Point) (h_g_intersect : g ≠ t) 
variables (hU_on_k : U ∈ k) (hV_on_k : V ∈ k)
variables (hU_on_g : U ∈ g) (hV_on_g : V ∈ g)

-- Point S bisects the arc UV not containing T.
variable (S : Point) (h_bisect : bisect_arc UV S T)

-- Point Q is the reflection of P over line ST.
variable (Q : Point) (h_reflect : reflect_over_line P S T Q)

-- The statement that Q, T, U, and V form a trapezoid.
theorem trapezoid_property : QP ∥ UV :=
by
  sorry

end trapezoid_property_l223_223053


namespace minimum_value_l223_223514

theorem minimum_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 64) : 
  x^2 + 8 * x * y + 16 * y^2 + 4 * z^2 ≥ 192 := 
  sorry

end minimum_value_l223_223514


namespace quadrangular_pyramid_and_tetrahedron_form_pentahedron_l223_223678

-- Define the basic geometric terms and conditions
noncomputable def quadrangular_pyramid (a : ℝ) := (all_edges_equal : ∀ e (e ∈ edges_of_quadrangular_pyramid), length(e) = a)
def regular_tetrahedron (a : ℝ) := (all_edges_equal : ∀ e (e ∈ edges_of_tetrahedron), length(e) = a)
def new_body_is_pentahedron (pyramid : Type) (tetrahedron : Type) : Prop :=
  (∃ (common_face : Face), common_face ∈ faces_of pyramids ∧ common_face ∈ faces_of tetrahedron) → 
  is_pentahedron formed_body

-- Rewrite the proof statement
theorem quadrangular_pyramid_and_tetrahedron_form_pentahedron (a : ℝ)
  (pyramid : quadrangular_pyramid a) (tetrahedron : regular_tetrahedron a) :
  new_body_is_pentahedron pyramid tetrahedron :=
sorry

end quadrangular_pyramid_and_tetrahedron_form_pentahedron_l223_223678


namespace mutually_exclusive_but_not_complementary_l223_223340

open Classical

namespace CardDistribution

inductive Card
| red | yellow | blue | white

inductive Person
| A | B | C | D

def Event_A_gets_red (distrib: Person → Card) : Prop :=
  distrib Person.A = Card.red

def Event_D_gets_red (distrib: Person → Card) : Prop :=
  distrib Person.D = Card.red

theorem mutually_exclusive_but_not_complementary :
  ∀ (distrib: Person → Card),
  (Event_A_gets_red distrib → ¬Event_D_gets_red distrib) ∧
  ¬(∀ (distrib: Person → Card), Event_A_gets_red distrib ∨ Event_D_gets_red distrib) := 
by
  sorry

end CardDistribution

end mutually_exclusive_but_not_complementary_l223_223340


namespace intersection_of_sets_l223_223801

theorem intersection_of_sets :
  let A := {y : ℝ | ∃ x : ℝ, y = Real.sin x}
  let B := {y : ℝ | ∃ x : ℝ, y = Real.sqrt (-(x^2 - 4*x + 3))}
  A ∩ B = {y : ℝ | 0 ≤ y ∧ y ≤ 1} :=
by
  sorry

end intersection_of_sets_l223_223801


namespace find_vector_p_l223_223633

def vector (α : Type*) := (α × α)

def line : set (vector ℝ) := 
  {v | ∃ x, v = (x, 3 * x + 1)}

def projection (v w : vector ℝ) : vector ℝ :=
  let (a, b) := v in
  let (c, d) := w in
  let denom := c * c + d * d in
  ((a * c + b * d) / denom * c, (a * c + b * d) / denom * d)

theorem find_vector_p (p : vector ℝ) (w : vector ℝ) (h : ∀ v ∈ line, projection v w = p) :
  p = (-3/10, 1/10) :=
sorry

end find_vector_p_l223_223633


namespace arrangements_mississippi_l223_223730

open Nat

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem arrangements_mississippi : 
  let total := 11
  let m_count := 1
  let i_count := 4
  let s_count := 4
  let p_count := 2
  factorial total / (factorial m_count * factorial i_count * factorial s_count * factorial p_count) = 34650 :=
by
  -- This is the proof placeholder
  sorry

end arrangements_mississippi_l223_223730


namespace max_sin_a_l223_223914

theorem max_sin_a (a b : ℝ) (h : sin (a + b) = sin a + sin b) : 
  sin a ≤ 1 :=
by
  sorry

end max_sin_a_l223_223914


namespace num_three_digit_numbers_l223_223849

/-- 
This theorem states that there are 160 three-digit numbers 
with three distinct digits, where one digit is the average of 
the other two, and the sum of the three digits is divisible by 3.
-/
theorem num_three_digit_numbers : 
  ∃ n : ℕ, n = 160 ∧ ∀ (a b c : ℕ), 
    (100 ≤ 100*a + 10*b + c) ∧ (100*a + 10*b + c < 1000) ∧ 
    (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ 
    (b = (a + c) / 2) ∧ 
    (a + b + c) % 3 = 0 → 
    a ≠ 0 ∧  b ≠ 0 ∧ c ≠ 0 := 
begin
  sorry
end

end num_three_digit_numbers_l223_223849


namespace translate_point_l223_223472

theorem translate_point :
  ∀ (M : ℝ × ℝ), M = (-2, 3) →
  (let M' := (M.1, M.2 - 3) in (M'.1 + 1, M'.2) = (-1, 0)) :=
by
  intros M h
  cases h
  simp [h]
  rfl

end translate_point_l223_223472


namespace find_g_five_l223_223058

def g (a b c x : ℝ) : ℝ := a * x^7 + b * x^6 + c * x - 3

theorem find_g_five (a b c : ℝ) (h : g a b c (-5) = -3) : g a b c 5 = 31250 * b - 3 := 
sorry

end find_g_five_l223_223058


namespace prob_0_lt_X_lt_1_l223_223931

noncomputable def normal_distribution (μ σ : ℝ) : Measure ℝ := sorry

variable (X : ℝ → ℝ) (μ σ : ℝ)

axiom normal_dist : X ~ (normal_distribution μ σ)
axiom prob_X_lt_1 : P(X < 1) = 1 / 2
axiom prob_X_gt_2 : P(X > 2) = 1 / 5

theorem prob_0_lt_X_lt_1 : P(0 < X < 1) = 3 / 10 := sorry

end prob_0_lt_X_lt_1_l223_223931


namespace largest_digit_change_l223_223258

-- Definitions
def initial_number : ℝ := 0.12345

def change_digit (k : Fin 5) : ℝ :=
  match k with
  | 0 => 0.92345
  | 1 => 0.19345
  | 2 => 0.12945
  | 3 => 0.12395
  | 4 => 0.12349

theorem largest_digit_change :
  ∀ k : Fin 5, k ≠ 0 → change_digit 0 > change_digit k :=
by
  intros k hk
  sorry

end largest_digit_change_l223_223258


namespace midpoint_of_VW_is_fixed_l223_223496

-- Conditions stated as structures and definitions
structure Triangle :=
  (A B C : Point)

structure TriangleWithSymmedianAndPerpendiculars (T : Triangle) :=
  (D : Point)
  (Q R : Point)
  (D_on_circumcircle : OnCircumcircle T.A T.B T.C D)
  (Q_is_perpendicular : PerpendicularFrom D T.C Q)
  (R_is_perpendicular : PerpendicularFrom D T.B R)

structure LineSegment :=
  (start end : Point)

structure LocusMidpoint (T : Triangle) (symm : TriangleWithSymmedianAndPerpendiculars T) :=
  (X : Point)
  (QR : LineSegment)
  (on_QR : OnLineSegment X QR)
  (not_at_QR : X ≠ QR.start ∧ X ≠ QR.end)
  (V W : Point)
  (VW : LineSegment)
  (X_VW_perpendicular : PerpendicularTo X VW)
  (V_on_AC : OnLineSegment V (LinearSegment.mk T.A T.C))
  (W_on_AB : OnLineSegment W (LinearSegment.mk T.A T.B))
  (mid_VW_fixed : ∀ (X : Point) (on_QR : OnLineSegment X QR), midpoint VW = fixed_midpoint)

-- The proof statement
theorem midpoint_of_VW_is_fixed (T : Triangle) (symm : TriangleWithSymmedianAndPerpendiculars T) (locus : LocusMidpoint T symm) :
  midpoint locus.VW = locus.fixed_midpoint :=
sorry 

end midpoint_of_VW_is_fixed_l223_223496


namespace digit_80th_in_sequence_l223_223445

noncomputable def sequence : List Nat := List.range' 1 60 in List.reverse (List.map (λ x => x.toString).join)
def digit_at (n : Nat) (seq: List Nat) : Option Char := seq.get? n

theorem digit_80th_in_sequence (seq:= sequence) :
  digit_at 79 seq = some '1' :=
sorry

end digit_80th_in_sequence_l223_223445


namespace solution_l223_223316

noncomputable def problem_statement : Prop :=
  (∀ i : ℂ, i^2 = -1) → 
  (∀ i : ℂ, -i ≠ 0 → (-i)^4 = 1) → 
  (∀ i : ℂ, (1 - i)^2 = 1 - 2*i + i^2) → 
  ∀ i : ℂ, (i^2 = -1) → (1 - i) ≠ 0 → ( ∃ (x : ℂ), x = (1-i)/(real.sqrt 2) → x^48 = 1 )

theorem solution : problem_statement :=
by {
  sorry
}

end solution_l223_223316


namespace election_votes_l223_223028

theorem election_votes (T : ℝ) (Vf Va Vn : ℝ)
  (h1 : Va = 0.375 * T)
  (h2 : Vn = 0.125 * T)
  (h3 : Vf = Va + 78)
  (h4 : T = Vf + Va + Vn) :
  T = 624 :=
by
  sorry

end election_votes_l223_223028


namespace square_placement_conditions_l223_223474

-- Definitions for natural numbers at vertices and center
def top_left := 14
def top_right := 6
def bottom_right := 15
def bottom_left := 35
def center := 210

theorem square_placement_conditions :
  (∃ gcd1 > 1, gcd1 = Nat.gcd top_left top_right) ∧
  (∃ gcd2 > 1, gcd2 = Nat.gcd top_right bottom_right) ∧
  (∃ gcd3 > 1, gcd3 = Nat.gcd bottom_right bottom_left) ∧
  (∃ gcd4 > 1, gcd4 = Nat.gcd bottom_left top_left) ∧
  (Nat.gcd top_left bottom_right = 1) ∧
  (Nat.gcd top_right bottom_left = 1) ∧
  (Nat.gcd top_left center > 1) ∧
  (Nat.gcd top_right center > 1) ∧
  (Nat.gcd bottom_right center > 1) ∧
  (Nat.gcd bottom_left center > 1) 
 := by
sorry

end square_placement_conditions_l223_223474


namespace split_stones_l223_223083

theorem split_stones (n : ℕ) :
  ∃ (heaps : list ℕ), (∀ h ∈ heaps, 1 ≤ h ∧ h ≤ n) ∧ (∀ i j, i ≠ j → (i < heaps.length ∧ j < heaps.length → heaps.nth i ≤ 2 * heaps.nth j)) :=
sorry

end split_stones_l223_223083


namespace arrangement_count_l223_223305

-- Given conditions
def num_basketballs : ℕ := 5
def num_volleyballs : ℕ := 3
def num_footballs : ℕ := 2
def total_balls : ℕ := num_basketballs + num_volleyballs + num_footballs

-- Way to calculate the permutations of multiset
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Proof statement
theorem arrangement_count : 
  factorial total_balls / (factorial num_basketballs * factorial num_volleyballs * factorial num_footballs) = 2520 :=
by
  sorry

end arrangement_count_l223_223305


namespace convert_base_10_to_6_l223_223721

theorem convert_base_10_to_6 : ∀ n : ℕ, n = 1729 → convert_to_base n 6 = [1, 2, 0, 0, 1] := 
by
  sorry

def convert_to_base (n : ℕ) (b : ℕ) : List ℕ := sorry

end convert_base_10_to_6_l223_223721


namespace side_length_of_S2_l223_223162

theorem side_length_of_S2 (r s : ℝ) 
  (h1 : 2 * r + s = 2025) 
  (h2 : 2 * r + 3 * s = 3320) :
  s = 647.5 :=
by {
  -- proof omitted
  sorry
}

end side_length_of_S2_l223_223162


namespace profit_function_formula_maximum_profit_on_interval_l223_223299

def w (x : ℝ) : ℝ := 4 - 3 / (x + 1)

def L (x : ℝ) : ℝ := 16 * w x - x - 2 * x

def domain (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 5

theorem profit_function_formula (x : ℝ) (h : domain x) : 
  L x = 64 - 48 / (x + 1) - 3 * x :=
sorry

theorem maximum_profit_on_interval (x : ℝ) (h : domain x) : 
  L x ≤ 43 ∧ (L 3 = 43) :=
sorry

end profit_function_formula_maximum_profit_on_interval_l223_223299


namespace parabola_focus_l223_223571

-- Given the parabola equation x^2 = ay where a ∈ ℝ, prove that the coordinates of the focus are (0, a/4).
theorem parabola_focus (a : ℝ) : 
  let focus := (0, a / 4) in 
  ∃ y : ℝ, (0, y) = focus :=
by 
  sorry

end parabola_focus_l223_223571


namespace units_digit_sum_zero_probability_l223_223361

theorem units_digit_sum_zero_probability :
  let S := finset.range 100  -- Set {1, 2, ..., 100}
  (∀ a ∈ S, ∀ b ∈ S, (5 ^ a + 6 ^ b) % 10 ≠ 0) → 
  (1 : ℝ) * 0 = 0 :=
by
  sorry

end units_digit_sum_zero_probability_l223_223361


namespace base_area_cone_l223_223288

theorem base_area_cone (V h : ℝ) (s_cylinder s_cone : ℝ) 
  (cylinder_volume : V = s_cylinder * h) 
  (cone_volume : V = (1 / 3) * s_cone * h) 
  (s_cylinder_val : s_cylinder = 15) : s_cone = 45 := 
by 
  sorry

end base_area_cone_l223_223288


namespace eggs_division_l223_223993

theorem eggs_division (n_students n_eggs : ℕ) (h_students : n_students = 9) (h_eggs : n_eggs = 73):
  n_eggs / n_students = 8 ∧ n_eggs % n_students = 1 :=
by
  rw [h_students, h_eggs]
  exact ⟨rfl, rfl⟩

end eggs_division_l223_223993


namespace N_has_at_least_8_distinct_divisors_N_has_at_least_32_distinct_divisors_l223_223159

-- Define the number with 1986 ones
def N : ℕ := (10^1986 - 1) / 9

-- Definition of having at least n distinct divisors
def has_at_least_n_distinct_divisors (num : ℕ) (n : ℕ) :=
  ∃ (divisors : Finset ℕ), divisors.card ≥ n ∧ ∀ d ∈ divisors, d ∣ num

theorem N_has_at_least_8_distinct_divisors :
  has_at_least_n_distinct_divisors N 8 :=
sorry

theorem N_has_at_least_32_distinct_divisors :
  has_at_least_n_distinct_divisors N 32 :=
sorry


end N_has_at_least_8_distinct_divisors_N_has_at_least_32_distinct_divisors_l223_223159


namespace geom_series_first_term_l223_223224

theorem geom_series_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 30)
  (h2 : a^2 / (1 - r^2) = 120) : 
  a = 120 / 17 :=
by
  sorry

end geom_series_first_term_l223_223224


namespace M_midpoint_AB_l223_223936

def Triangle (α β γ : Type) := Prop

variables {P Q R A B C K M : Type}

-- Define a triangle
def triangle : Triangle P Q R := sorry

-- Point on BC
def point_on_bc (K : Type) (B C : Type) : Prop := sorry

-- Angle bisectors KM of AKB and KP of AKC
def angle_bisector (K M : Type) := sorry
def angle_bisector_of_AKB (K M B : Type) : Prop := angle_bisector K M
def angle_bisector_of_AKC (K P C : Type) : Prop := angle_bisector K P

-- Congruency of triangles BMK and PMK
def congruent_triangles (BMK PMK : Type) : Prop := sorry

-- Midpoint M of segment AB
def midpoint (M A B : Type) : Prop := sorry

theorem M_midpoint_AB
  (triangle_ABC : Triangle A B C)
  (K_on_BC : point_on_bc K B C)
  (KM_bisector_AKB : angle_bisector_of_AKB K M B)
  (KP_bisector_AKC : angle_bisector_of_AKC K P C)
  (BMK_PMK_congruent : congruent_triangles (Triangle B M K) (Triangle P M K)) :
  midpoint M A B := 
sorry

end M_midpoint_AB_l223_223936


namespace count_perfect_squares_l223_223070

open Int

def p (x : Int) : Int := 2 * x^3 - 3 * x^2 + 1

theorem count_perfect_squares (n : Int) (h : 1 ≤ n ∧ n ≤ 2016) : 
  (finset.filter (λ x, ∃ (n : Int), p x = n * n) (finset.range 2016)).card = 32 :=
by
  sorry

end count_perfect_squares_l223_223070


namespace split_piles_equiv_single_stone_heaps_l223_223099

theorem split_piles_equiv_single_stone_heaps (n : ℕ) (heaps : List ℕ) (h_initial : ∀ h ∈ heaps, h = 1)
  (h_size : heaps.length = n) :
  ∃ final_heap, (∀ x y ∈ heaps, x + y ≤ 2 * max x y) ∧ (List.sum heaps = (heaps.length) * 1) := by
  sorry

end split_piles_equiv_single_stone_heaps_l223_223099


namespace at_most_one_divisor_perfect_square_l223_223067

theorem at_most_one_divisor_perfect_square (p : ℕ) (n : ℕ) (hp : p.prime) (hp_odd : p % 2 = 1) (hn : 0 < n) :
  ∀ d1 d2 : ℕ, d1 ∣ p * n^2 → d2 ∣ p * n^2 → (d1 + n^2).is_square → (d2 + n^2).is_square → d1 = d2 :=
by sorry

end at_most_one_divisor_perfect_square_l223_223067


namespace g_ab_eq_zero_l223_223827

def g (x : ℤ) : ℤ := x^2 - 2013 * x

theorem g_ab_eq_zero (a b : ℤ) (h1 : g a = g b) (h2 : a ≠ b) : g (a + b) = 0 :=
by
  sorry

end g_ab_eq_zero_l223_223827


namespace angle_B_eq_2_angle_C_l223_223467

theorem angle_B_eq_2_angle_C (A B C D : Type) [Triangle A B C]
    (h_altitude : is_altitude A D B C) 
    (h_acute : is_acute_triangle A B C)
    (h_AB_plus_BD_eq_DC : AB + BD = DC) : 
    Angle B A C = 2 * Angle C A B :=
by
  sorry

end angle_B_eq_2_angle_C_l223_223467


namespace highest_number_paper_l223_223023

theorem highest_number_paper
  (n : ℕ)
  (P : ℝ)
  (hP : P = 0.010309278350515464)
  (hP_formula : 1 / n = P) :
  n = 97 :=
by
  -- Placeholder for proof
  sorry

end highest_number_paper_l223_223023


namespace find_first_term_l223_223219

variable {a r : ℚ}

theorem find_first_term (h1 : a / (1 - r) = 30) (h2 : a^2 / (1 - r^2) = 120) : a = 240 / 7 :=
by
  sorry

end find_first_term_l223_223219


namespace num_squares_among_p_l223_223069

def p (x : ℤ) : ℤ := 2 * x ^ 3 - 3 * x ^ 2 + 1

theorem num_squares_among_p (h : ∀ x, 1 ≤ x ∧ x ≤ 2016) : 
 ∑ i in filter (λ x, ∃ y, y * y = p x) (range (2016 + 1)), (1 : ℤ) = 32 := 
sorry

end num_squares_among_p_l223_223069


namespace split_piles_equiv_single_stone_heaps_l223_223092

theorem split_piles_equiv_single_stone_heaps (n : ℕ) (heaps : List ℕ) (h_initial : ∀ h ∈ heaps, h = 1)
  (h_size : heaps.length = n) :
  ∃ final_heap, (∀ x y ∈ heaps, x + y ≤ 2 * max x y) ∧ (List.sum heaps = (heaps.length) * 1) := by
  sorry

end split_piles_equiv_single_stone_heaps_l223_223092


namespace jose_profit_share_l223_223241

def investment_share (toms_investment : ℕ) (jose_investment : ℕ) 
  (toms_duration : ℕ) (jose_duration : ℕ) (total_profit : ℕ) : ℕ :=
  let toms_capital_months := toms_investment * toms_duration
  let jose_capital_months := jose_investment * jose_duration
  let total_capital_months := toms_capital_months + jose_capital_months
  let jose_share_ratio := jose_capital_months / total_capital_months
  jose_share_ratio * total_profit

theorem jose_profit_share 
  (toms_investment : ℕ := 3000)
  (jose_investment : ℕ := 4500)
  (toms_duration : ℕ := 12)
  (jose_duration : ℕ := 10)
  (total_profit : ℕ := 6300) :
  investment_share toms_investment jose_investment toms_duration jose_duration total_profit = 3500 := 
sorry

end jose_profit_share_l223_223241


namespace triangle_angle_exceeds_179_l223_223449

noncomputable def angle_opposite_side_c_exceeds_179_degrees 
  (a b c : ℝ) (ha : a = 2) (hb : b = 2) (hc : c = 4) : Prop :=
∃ (C : ℝ), C > 179 ∧ (cos C = (-1))

theorem triangle_angle_exceeds_179 
  (a b c : ℝ) (ha : a = 2) (hb : b = 2) (hc : c = 4) : angle_opposite_side_c_exceeds_179_degrees a b c ha hb hc :=
sorry

end triangle_angle_exceeds_179_l223_223449


namespace triangle_angle_B_sin_expression_range_l223_223870

theorem triangle_angle_B (A C : ℝ) (hA : tan A = 1/3) (hC : tan C = 1/2) : 
  let B := π - (A + C) in 
  B = 3 * π / 4 :=
sorry

theorem sin_expression_range (α β : ℝ) (h_sum : α + β = 3 * π / 4) (hα_pos : 0 < α) (hβ_pos : 0 < β) :
  ∃ (r : ℝ), (r = sqrt 2 * sin α - sin β ∧ -sqrt 2 / 2 < r ∧ r < 1) :=
sorry

end triangle_angle_B_sin_expression_range_l223_223870


namespace charlie_mistake_correction_l223_223712

theorem charlie_mistake_correction : (0.0075 * 25.6 = 0.192) :=
by
  -- Define the given conditions and theorem statement.
  have h1 : 75 * 256 = 19200 := by sorry   -- Placeholder for given calculator result condition.
  have h2 : (shift_decimal (0.0075, 4) * shift_decimal (25.6, 1)) = 19200 := by sorry
  -- Introduce function to shift decimal places correctly.
  def shift_decimal (x : ℝ, n : ℕ) : ℝ := x * 10^n

  have h3 : 0.0075 * 25.6 = (shift_decimal 19200 (-5)).lean
    := by sorry
  exact eq.trans ... 0.192 

end charlie_mistake_correction_l223_223712


namespace continuous_function_form_l223_223762

theorem continuous_function_form (f : ℝ → ℝ) (h_cont : continuous f) 
(h_eq : ∀ x y : ℝ, f ((x + y) / 2) = (f x + f y) / 2) : 
∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b :=
sorry

end continuous_function_form_l223_223762


namespace minimize_transportation_cost_l223_223958

/-- Total length of the Yihuang Expressway -/
def L : ℝ := 350

/-- Fixed hourly cost in yuan -/
def C_fixed : ℝ := 200

/-- Upper limit of speed in km/hr -/
def vmax : ℝ := 120

/-- Lower limit of speed in km/hr -/
def vmin : ℝ := 60

/-- Maximum hourly transportation cost at maximum speed (488 yuan) -/
def C_max : ℝ := 488

/-- Proportional constant for variable cost part -/
noncomputable def k : ℝ := (C_max - C_fixed) / (vmax^2)

/-- Total transportation cost function -/
noncomputable def total_cost (v : ℝ) : ℝ := 
  (C_fixed + k * v^2) * (L / v)

theorem minimize_transportation_cost :
  (60 ≤ v ∧ v ≤ 120) → total_cost v = 100 :=
begin
  sorry
end

end minimize_transportation_cost_l223_223958


namespace most_probable_light_is_green_l223_223696

def duration_red := 30
def duration_yellow := 5
def duration_green := 40
def total_duration := duration_red + duration_yellow + duration_green

def prob_red := duration_red / total_duration
def prob_yellow := duration_yellow / total_duration
def prob_green := duration_green / total_duration

theorem most_probable_light_is_green : prob_green > prob_red ∧ prob_green > prob_yellow := 
  by
  sorry

end most_probable_light_is_green_l223_223696


namespace convert_base_10_to_6_l223_223722

theorem convert_base_10_to_6 : ∀ n : ℕ, n = 1729 → convert_to_base n 6 = [1, 2, 0, 0, 1] := 
by
  sorry

def convert_to_base (n : ℕ) (b : ℕ) : List ℕ := sorry

end convert_base_10_to_6_l223_223722


namespace pentagon_area_l223_223943

theorem pentagon_area :
  ∀ (PT TR : ℝ) (h1 : PT = 10) (h2 : TR = 8) 
  (PQRS_area : ℝ) (PTR_area : ℝ) (PTRSQ_area : ℝ),
  PQRS_area = (PT^2 + TR^2) ∧
  PTR_area = (PT * TR) / 2 ∧
  PTRSQ_area = PQRS_area - PTR_area →
  PTRSQ_area = 124 :=
by {
  intros PT TR h1 h2 PQRS_area PTR_area PTRSQ_area hyp,
  sorry
}

end pentagon_area_l223_223943


namespace vector_angle_obtuse_range_l223_223008

variables {a b : ℝ^3} (t : ℝ)

theorem vector_angle_obtuse_range (ha : ‖a‖ = 2) (hb : ‖b‖ = 1)
  (angle_ab : real.angle a b = real.pi / 3) :
  ((2 * t • a + 7 • b) ⬝ (a + t • b) < 0) →
  (¬ collinear ℝ ({2 * t • a + 7 • b, a + t • b})) →
  t ∈ set.Ioo (-7 : ℝ) (-real.sqrt 14 / 2) ∪ set.Ioo (-real.sqrt 14 / 2) (-1 / 2) := sorry

end vector_angle_obtuse_range_l223_223008


namespace boat_distance_downstream_l223_223654

theorem boat_distance_downstream (speed_boat_still : ℕ) (speed_stream : ℕ) (time_hours : ℕ) (dist_downstream : ℕ) :
  speed_boat_still = 16 → speed_stream = 5 → time_hours = 7 → dist_downstream = 147 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- Continue with necessary intermediate steps, but skipped here for brevity
  sorry

end boat_distance_downstream_l223_223654


namespace simplify_expression_l223_223947

theorem simplify_expression (x : ℚ) (h1 : x ≠ 3) (h2 : x ≠ 4) (h3 : x ≠ 2) (h4 : x ≠ 5):
  ((x^2 - 4 * x + 3) / (x^2 - 6 * x + 9)) / ((x^2 - 6 * x + 8) / (x^2 - 8 * x + 15)) = 
  (x - 1) * (x - 5) / ((x - 3) * (x - 4) * (x - 2)) :=
sorry

end simplify_expression_l223_223947


namespace Sam_chewing_gums_l223_223591

theorem Sam_chewing_gums (total_gums mary_gums sue_gums sam_gums : ℕ) 
    (h1 : mary_gums = 5) 
    (h2 : sue_gums = 15)
    (h3 : total_gums = 30)
    (h4 : total_gums = mary_gums + sue_gums + sam_gums) :
    sam_gums = 10 := 
by
  have h5 : mary_gums + sue_gums = 20 := by sorry   -- This would normally be proven
  have h6 : sam_gums = total_gums - (mary_gums + sue_gums) := by sorry  -- This would normally be proven
  show sam_gums = 10, by sorry
  -- Proof steps would go here

end Sam_chewing_gums_l223_223591


namespace min_value_frac_l223_223813

theorem min_value_frac (σ : ℝ) (ξ : ℝ → ℝ) (h₁ : ξ ∼ normal 1 σ^2) (h₂ : prob ξ 0 = prob ξ a) (h₃ : 0 < x) (h₄ : x < a) : 
  ∃ a x, (a = 2) ∧ (0 < x) ∧ (x < a) ∧ (frac_min = (1/x + 4/(a - x))) ∧ (frac_min = 9/2) := 
sorry

end min_value_frac_l223_223813


namespace b_n_negative_term_50th_l223_223334

noncomputable def b (n : ℕ) : ℝ :=
  ∑ k in finset.range n, real.cos (k)

theorem b_n_negative_term_50th : (nat.find (λ n, b n < 0) 50) = 314 :=
by
  sorry

end b_n_negative_term_50th_l223_223334


namespace intersection_of_sets_l223_223800

theorem intersection_of_sets :
  let A := {y : ℝ | ∃ x : ℝ, y = Real.sin x}
  let B := {y : ℝ | ∃ x : ℝ, y = Real.sqrt (-(x^2 - 4*x + 3))}
  A ∩ B = {y : ℝ | 0 ≤ y ∧ y ≤ 1} :=
by
  sorry

end intersection_of_sets_l223_223800


namespace distance_between_Joe_and_Gracie_l223_223846

def Joe_point : Complex := Complex.mk 1 2
def Gracie_point : Complex := Complex.mk (-1) 1
def distance (z w : Complex) : Real := Complex.abs (z - w)

theorem distance_between_Joe_and_Gracie :
  distance Joe_point Gracie_point = Real.sqrt 5 :=
by
  -- proof goes here
  sorry

end distance_between_Joe_and_Gracie_l223_223846


namespace total_distance_traveled_l223_223607

theorem total_distance_traveled 
  (initial_distance : ℕ) 
  (speed_increase : ℕ) 
  (hours : ℕ) 
  (total_distance : ℕ) : 
  initial_distance = 35 → 
  speed_increase = 2 → 
  hours = 12 → 
  total_distance = (finset.range hours).sum (λ i, initial_distance + i * speed_increase) → 
  total_distance = 546 := 
  by
    intros h_initial h_increase h_hours h_sum
    rw [h_initial, h_increase, h_hours] at h_sum
    sorry

end total_distance_traveled_l223_223607


namespace total_area_of_earth_correct_l223_223598

noncomputable def total_area_of_earth (ocean_area land_area : ℝ) : ℝ :=
  ocean_area + land_area

theorem total_area_of_earth_correct :
  let ocean_area := 361 in
  let land_area := ocean_area - 2.12 in
  total_area_of_earth ocean_area land_area = 719.88 :=
by
  sorry

end total_area_of_earth_correct_l223_223598


namespace simplify_exponent_l223_223173

theorem simplify_exponent (y : ℝ) : (3 * y^4)^5 = 243 * y^20 :=
by
  sorry

end simplify_exponent_l223_223173


namespace magnitude_squared_l223_223858

-- Let z be the complex number 3 + 4i
def z : ℂ := 3 + 4 * Complex.I

-- Prove that the magnitude of z squared equals 25
theorem magnitude_squared : Complex.abs z ^ 2 = 25 := by
  -- The term "by" starts the proof block, and "sorry" allows us to skip the proof details.
  sorry

end magnitude_squared_l223_223858


namespace Shekar_average_marks_l223_223169

def marks : List (String × ℕ × ℕ) := [("Mathematics", 92, 20),
                                       ("Science", 78, 10),
                                       ("Social Studies", 85, 15),
                                       ("English", 67, 8),
                                       ("Biology", 89, 12),
                                       ("Computer Science", 74, 5),
                                       ("Physical Education", 81, 7),
                                       ("Chemistry", 95, 10),
                                       ("History", 70, 5),
                                       ("Physics", 88, 8)]

noncomputable def average_mark {α : Type*} [DecidableEq α] (marks : List (α × ℕ × ℕ)) : ℕ :=
  let weighted_sum := marks.foldl (λ acc m, acc + (m.2.1 * m.2.2) / 100) 0
  weighted_sum

theorem Shekar_average_marks :
  average_mark marks = 84 := sorry

end Shekar_average_marks_l223_223169


namespace intersects_x_axis_at_one_point_l223_223587

theorem intersects_x_axis_at_one_point (a : ℝ) :
  (∃ x, ax^2 + (a-3)*x + 1 = 0) ∧ (∀ x₁ x₂, ax^2 + (a-3)*x + 1 = 0 → x₁ = x₂) ↔ (a = 0 ∨ a = 1 ∨ a = 9) := by
  sorry

end intersects_x_axis_at_one_point_l223_223587


namespace mean_score_boys_7B_l223_223227

theorem mean_score_boys_7B (μ : ℝ) 
  (h_total_students_A_B : ∀ c : ℕ, c = 30)
  (h_mean_girls_7A : ∀ g : ℕ, g = 15 → ∀ s : ℝ, s = 48 → mean (class 7A g s))
  (h_overall_mean : ∀ n : ℕ, n = 60 → ∀ t : ℝ, t = 60 → mean (total_students n t))
  (h_girls_mean : ∀ n : ℕ, n = 20 → ∀ t : ℝ, t = 60 → mean (girls n t))
  (h_girls_7B_girls_7A : ∀ n : ℕ, n = 5 → ∀ boys : ℕ, boys = 15 → ∀ s : ℝ, s = 96 → double_mean (class 7B n s) (class 7A boys s))
  : 10 * μ = 672 :=
sorry

end mean_score_boys_7B_l223_223227


namespace segments_equal_and_perpendicular_l223_223937

theorem segments_equal_and_perpendicular
  (A B C D P Q R S : Point)
  (hAB_sq : is_square A B (center A B))
  (hBC_sq : is_square B C (center B C))
  (hCD_sq : is_square C D (center C D))
  (hDA_sq : is_square D A (center D A))
  (hP : P = center A B)
  (hQ : Q = center B C)
  (hR : R = center C D)
  (hS : S = center D A) :
  (dist P R = dist Q S) ∧ (angle P R Q S = 90) := 
sorry

end segments_equal_and_perpendicular_l223_223937


namespace general_term_and_sum_proof_l223_223402
open BigOperators

variable (n : ℕ)
variable (S : ℕ → ℕ) -- S_n = n^2 + n 
variable (a : ℕ → ℕ) -- a_n = 2n

-- Define Sn in terms of n
def Sn (n : ℕ) : ℕ := n^2 + n

-- Define an as the difference of Sn
def an (n : ℕ) : ℕ := if n = 1 then 2 else Sn n - Sn (n - 1)

-- Define the sequence {1 / ((n + 1) * a_n)} 
def seq (n : ℕ) : ℝ := 1 / ((n + 1) * (a n))

-- The sum of the first n terms of the sequence {1 / ((n + 1) * a_n)}
def Tn (n : ℕ) : ℝ := 1 / 2 * (1 - 1 / (n + 1))

theorem general_term_and_sum_proof :
  (∀ n, S n = n^2 + n) →
  (∀ n, a n = if n = 1 then 2 else S n - S (n - 1)) →
  (a 1 = 2) →
  (∀ n, a n = 2 * n) →
  (∀ n, T_n n = 1 / 2 * (1 - 1 / (n + 1))) :=
by
  intros hS ha ha1 haln
  dsimp at *
  sorry

end general_term_and_sum_proof_l223_223402


namespace combine_heaps_l223_223127

def heaps_similar (x y : ℕ) : Prop :=
  x ≤ 2 * y ∧ y ≤ 2 * x

theorem combine_heaps (n : ℕ) : 
  ∃ f : ℕ → ℕ, 
  f 0 = n ∧
  ∀ k, k < n → (∃ i j, i + j = k ∧ heaps_similar (f i) (f j)) ∧ 
  (∃ k, f k = n) :=
by
  sorry

end combine_heaps_l223_223127


namespace inequality_solution_l223_223551

theorem inequality_solution (x : ℝ) :
  (2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 15) ↔ (x < 1 ∨ x > 3) ∧ (x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5) :=
by
  sorry

end inequality_solution_l223_223551


namespace fraction_zero_numerator_l223_223866

theorem fraction_zero_numerator (x : ℝ) (h₁ : (x^2 - 9) / (x + 3) = 0) (h₂ : x + 3 ≠ 0) : x = 3 :=
sorry

end fraction_zero_numerator_l223_223866


namespace axis_of_symmetry_condition_l223_223595

-- Define the conditions used in the problem
variables {p q r s : ℝ} -- Coefficients are real numbers

-- Nonzero conditions
variables (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0)

-- Define the equations
def curve_eq (x : ℝ) : ℝ := (p * x + q) / (r * x + s)
def symmetry_eq (x : ℝ) : ℝ := 2 * x

-- Proof statement
theorem axis_of_symmetry_condition :
  (∀ x : ℝ, curve_eq (symmetry_eq x) = curve_eq x) → p + s = 0 :=
sorry

end axis_of_symmetry_condition_l223_223595


namespace person_arrangement_count_l223_223769

theorem person_arrangement_count :
  let positions := (1, 2, 3, 4, 5)
  let A_positions := {1, 2}
  let B_positions := {2, 3}
  (number_of_possible_arrangements A_positions B_positions positions = 18) :=
by
  sorry

end person_arrangement_count_l223_223769


namespace range_of_a_l223_223845

theorem range_of_a (a : ℝ) (an bn : ℕ → ℝ)
  (h_an : ∀ n, an n = (-1) ^ (n + 2013) * a)
  (h_bn : ∀ n, bn n = 2 + (-1) ^ (n + 2014) / n)
  (h_condition : ∀ n : ℕ, 1 ≤ n → an n < bn n) :
  -2 ≤ a ∧ a < 1 :=
by
  sorry

end range_of_a_l223_223845


namespace limit_proof_l223_223015

noncomputable def f (x : ℝ) := Real.log (2 - x) + x^3

theorem limit_proof : 
    tendsto (λ Δx : ℝ, (f (1 + Δx) - f 1) / (3 * Δx)) (nhds 0) (nhds (2 / 3)) :=
sorry

end limit_proof_l223_223015


namespace smallest_angle_triangle_l223_223462

theorem smallest_angle_triangle 
  (a b c : ℝ)
  (h1 : 3 * a < b + c)
  (h2 : a + b > c)
  (h3 : a + c > b) : 
  ∠BAC < ∠ABC ∧ ∠BAC < ∠ACB :=
by 
  sorry

end smallest_angle_triangle_l223_223462


namespace exists_point_with_sum_of_distances_ge_n_l223_223783

variable (n : ℕ) 
variable (A : Fin n → ℝ × ℝ) -- points on the circumference

-- Defining that points A_i are on a circle of radius 1
def on_circle (A : ℕ → ℝ × ℝ) (r : ℝ) : Prop :=
  ∀ i, (A i).fst^2 + (A i).snd^2 = r^2

-- Proposition: There exists a point M on the circle such that the sum of distances >= n
theorem exists_point_with_sum_of_distances_ge_n
  (h : on_circle A 1) : ∃ M : ℝ × ℝ, (M.fst^2 + M.snd^2 = 1 ∧ 
  ∑ i in (Finset.range n), (dist (A i) M) ≥ n) :=
sorry

end exists_point_with_sum_of_distances_ge_n_l223_223783


namespace closest_integer_ratio_l223_223329

theorem closest_integer_ratio :
  let S := (finset.range 20).sum (λ n, 2^(n+1)) + 5,
  r := (2^20 : ℝ) / S in
  abs (r - 1) < 1 / 2 := 
by
  let S := (finset.range 20).sum (λ n, (2^(n+1) : ℝ)) + 5
  let r := (2^20 : ℝ) / S
  have h : abs (r - 1) < 1 / 2 := sorry
  exact h

end closest_integer_ratio_l223_223329


namespace average_percentage_of_popped_kernels_l223_223147

theorem average_percentage_of_popped_kernels (k1 k2 k3 p1 p2 p3 : ℕ) (h1 : k1 = 75) (h2 : k2 = 50) (h3 : k3 = 100)
    (h1_pop : p1 = 60) (h2_pop : p2 = 42) (h3_pop : p3 = 82) :
    ((p1 / (k1 : ℝ) + p2 / (k2 : ℝ) + p3 / (k3 : ℝ)) / 3) * 100 = 82 :=
by
  -- The proportion for each bag
  have prop1 : p1 / (k1 : ℝ) = 60 / 75 := by rw [h1, h1_pop]
  have prop2 : p2 / (k2 : ℝ) = 42 / 50 := by rw [h2, h2_pop]
  have prop3 : p3 / (k3 : ℝ) = 82 / 100 := by rw [h3, h3_pop]
  -- Sum the proportions
  have total_props : (p1 / (k1 : ℝ) + p2 / (k2 : ℝ) + p3 / (k3 : ℝ)) = 0.8 + 0.84 + 0.82 := by
    rw [prop1, prop2, prop3]
  -- Calculating the average proportion
  have avg_prop : ((p1 / (k1 : ℝ) + p2 / (k2 : ℝ) + p3 / (k3 : ℝ)) / 3) = 0.82 := by
    rw [total_props]
  -- Finally multiply the average by 100 to get the percentage
  have avg_percentage : ((p1 / (k1 : ℝ) + p2 / (k2 : ℝ) + p3 / (k3 : ℝ)) / 3) * 100 = 82 := by
    rw [avg_prop]
    norm_num
  exact avg_percentage

end average_percentage_of_popped_kernels_l223_223147


namespace area_of_PQRS_l223_223526

theorem area_of_PQRS (a : ℝ) 
  (h_a : a^2 = 25) 
  (WXYZ_is_square : ∀ (W X Y Z : (ℝ × ℝ)), WXYZ_is_square W X Y Z)
  (WPY_is_eq_triangle : ∀ (W P Y: (ℝ × ℝ)), WPY_is_eq_triangle W P Y)
  (XQZ_is_eq_triangle : ∀ (X Q Z: (ℝ × ℝ)), XQZ_is_eq_triangle X Q Z)
  (YRW_is_eq_triangle : ∀ (Y R W: (ℝ × ℝ)), YRW_is_eq_triangle Y R W)
  (ZSX_is_eq_triangle : ∀ (Z S X: (ℝ × ℝ)), ZSX_is_eq_triangle Z S X) 
  : area_of_square_PQRS = 50 + 25 * Real.sqrt 3 :=
by 
  sorry

noncomputable def WXYZ_is_square (W X Y Z : (ℝ × ℝ)) : Prop := sorry
noncomputable def WPY_is_eq_triangle (W P Y : (ℝ × ℝ)) : Prop := sorry
noncomputable def XQZ_is_eq_triangle (X Q Z : (ℝ × ℝ)) : Prop := sorry
noncomputable def YRW_is_eq_triangle (Y R W : (ℝ × ℝ)) : Prop := sorry
noncomputable def ZSX_is_eq_triangle (Z S X : (ℝ × ℝ)) : Prop := sorry
noncomputable def area_of_square_PQRS : ℝ := sorry

end area_of_PQRS_l223_223526


namespace log_factorial_eq_l223_223326

theorem log_factorial_eq (k : ℕ) (h : k > 2) (h_eq : log 10 ((k - 2)!) + log 10 ((k - 1)!) + 3 = 2 * log 10 (k!)) : 
  k = 10 := 
sorry

end log_factorial_eq_l223_223326


namespace not_periodic_cos_add_cos_sqrt2_l223_223723

noncomputable def f (x : ℝ) : ℝ := Real.cos x + Real.cos (x * Real.sqrt 2)

theorem not_periodic_cos_add_cos_sqrt2 :
  ¬(∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x) :=
sorry

end not_periodic_cos_add_cos_sqrt2_l223_223723


namespace initial_speed_l223_223894

variable (v : ℝ)
variable (h1 : (v / 2) + 2 * v = 75)

theorem initial_speed (v : ℝ) (h1 : (v / 2) + 2 * v = 75) : v = 30 :=
sorry

end initial_speed_l223_223894


namespace permutations_mississippi_l223_223753

theorem permutations_mississippi : 
  let total_letters := 11
  let m_count := 1
  let i_count := 4
  let s_count := 4
  let p_count := 2
  (Nat.factorial total_letters / (Nat.factorial m_count * Nat.factorial i_count * Nat.factorial s_count * Nat.factorial p_count)) = 34650 := 
by
  sorry

end permutations_mississippi_l223_223753


namespace num_squares_among_p_l223_223068

def p (x : ℤ) : ℤ := 2 * x ^ 3 - 3 * x ^ 2 + 1

theorem num_squares_among_p (h : ∀ x, 1 ≤ x ∧ x ≤ 2016) : 
 ∑ i in filter (λ x, ∃ y, y * y = p x) (range (2016 + 1)), (1 : ℤ) = 32 := 
sorry

end num_squares_among_p_l223_223068


namespace max_sum_factors_of_60_exists_max_sum_factors_of_60_l223_223056

theorem max_sum_factors_of_60 (d Δ : ℕ) (h : d * Δ = 60) : (d + Δ) ≤ 61 :=
sorry

theorem exists_max_sum_factors_of_60 : ∃ d Δ : ℕ, d * Δ = 60 ∧ d + Δ = 61 :=
sorry

end max_sum_factors_of_60_exists_max_sum_factors_of_60_l223_223056


namespace split_stones_l223_223080

theorem split_stones (n : ℕ) :
  ∃ (heaps : list ℕ), (∀ h ∈ heaps, 1 ≤ h ∧ h ≤ n) ∧ (∀ i j, i ≠ j → (i < heaps.length ∧ j < heaps.length → heaps.nth i ≤ 2 * heaps.nth j)) :=
sorry

end split_stones_l223_223080


namespace speed_of_current_l223_223683

theorem speed_of_current (v : ℝ) : 
  (∀ s, s = 3 → s / (3 - v) = 2.3076923076923075) → v = 1.7 := 
by
  intro h
  sorry

end speed_of_current_l223_223683


namespace micah_water_intake_l223_223522

def morning : ℝ := 1.5
def early_afternoon : ℝ := 2 * morning
def late_afternoon : ℝ := 3 * morning
def evening : ℝ := late_afternoon - 0.25 * late_afternoon
def night : ℝ := 2 * evening
def total_water_intake : ℝ := morning + early_afternoon + late_afternoon + evening + night

theorem micah_water_intake :
  total_water_intake = 19.125 := by
  sorry

end micah_water_intake_l223_223522


namespace original_price_of_shoes_l223_223520

-- Define the conditions.
def discount_rate : ℝ := 0.20
def amount_paid : ℝ := 480

-- Statement of the theorem.
theorem original_price_of_shoes (P : ℝ) (h₀ : P * (1 - discount_rate) = amount_paid) : 
  P = 600 :=
by
  sorry

end original_price_of_shoes_l223_223520


namespace combined_savings_l223_223540

def salary_per_hour : ℝ := 10
def daily_hours : ℝ := 10
def weekly_days : ℝ := 5
def robby_saving_ratio : ℝ := 2 / 5
def jaylene_saving_ratio : ℝ := 3 / 5
def miranda_saving_ratio : ℝ := 1 / 2
def weeks : ℝ := 4

theorem combined_savings 
  (sph : ℝ := salary_per_hour)
  (dh : ℝ := daily_hours)
  (wd : ℝ := weekly_days)
  (rr : ℝ := robby_saving_ratio)
  (jr : ℝ := jaylene_saving_ratio)
  (mr : ℝ := miranda_saving_ratio)
  (wk : ℝ := weeks) :
  (rr * (wk * wd * (dh * sph)) + jr * (wk * wd * (dh * sph)) + mr * (wk * wd * (dh * sph))) = 3000 :=
by
  sorry

end combined_savings_l223_223540


namespace swimming_pool_length_l223_223610

theorem swimming_pool_length (width height volume : ℝ) (ft_to_gallons : ℝ) (water_volume_gallons : ℝ) : 
    width = 20 → height = 0.5 → ft_to_gallons = 7.5 → water_volume_gallons = 4500 → volume = water_volume_gallons / ft_to_gallons →
    volume = width * height * 60 :=
by
  intros w_eq h_eq ft_gal_eq gallon_eq vol_eq
  rw [w_eq, h_eq, ft_gal_eq, gallon_eq, vol_eq]
  sorry

end swimming_pool_length_l223_223610


namespace magnitude_twice_a_minus_b_l223_223811

variables {a b : EuclideanSpace ℝ (Fin 2)}

-- Given conditions
def angle_ab := (Real.pi / 3 * 2) -- 120 degrees in radians
def norm_a : ℝ := 1
def norm_b : ℝ := 2

-- Assuming a unit vector of magnitude 1 scaled appropriately
noncomputable def a := (λ i, if i = 0 then 1 else 0) • norm_a
noncomputable def b := (λ i, if i = 1 then 2 else 0) • norm_b

-- The task to prove
theorem magnitude_twice_a_minus_b : ∥(2 : ℝ) • a - b∥ = 2 * Real.sqrt 3 :=
by sorry

end magnitude_twice_a_minus_b_l223_223811


namespace part_a_part_b_part_c_l223_223276

variable (p : ℝ) (N : ℕ)

-- Condition: Probability of giving birth to twins in Shvambrania is p, and triplets are not born.
axiom prob_twins : 0 ≤ p ∧ p ≤ 1

-- Part (a)
theorem part_a (h : prob_twins p) : 
  ∃ q : ℝ, q = (2 * p) / (1 + p) ∧ 
    (q = (2 * p) / (p + 1)) := 
begin
  existsi (2 * p) / (1 + p),
  split,
  { simp, },
  { sorry }
end

-- Part (b)
theorem part_b (h : prob_twins p) : 
  ∃ q : ℝ, q = (2 * p) / (2 * p + (1 - p) ^ 2) :=
begin
  existsi (2 * p) / (2 * p + (1 - p) ^ 2),
  sorry,
end

-- Part (c)
theorem part_c (h : prob_twins p) : 
  ∃ q : ℝ, q = (N * p) / (p + 1) :=
begin
  existsi (N * p) / (p + 1),
  sorry,
end

end part_a_part_b_part_c_l223_223276


namespace time_spent_researching_l223_223161

-- Definitions based on conditions
def pages_written := 6
def time_per_page_written := 30  -- in minutes
def time_spent_editing := 75  -- in minutes
def total_time_spent := 5 * 60  -- in minutes

-- The main theorem to prove
theorem time_spent_researching :
  let time_spent_writing := pages_written * time_per_page_written in
  let total_time_writing_and_editing := time_spent_writing + time_spent_editing in
  total_time_spent - total_time_writing_and_editing = 45 :=
by
  sorry

end time_spent_researching_l223_223161


namespace max_value_l223_223833

noncomputable def maximum_value (x y : ℝ) : ℝ :=
  (x + 3) * (y + 1)

theorem max_value (x y : ℝ) (h1 : xy + 6 = x + 9y) (h2 : y < 1) :
  ∃ x y, maximum_value x y = 27 - 12 * Real.sqrt 2 :=
sorry

end max_value_l223_223833


namespace problem_statement_l223_223383

variable {a : ℕ → ℕ}

def sequence_sum (a : ℕ → ℕ) : Prop :=
∀ n : ℕ, n > 0 → (finset.range n).sum (λ i, a (i + 1)) = n^2

def arithmetic_sequence (a : ℕ → ℕ) (k p r : ℕ) : Prop :=
1 / (a k) + 1 / (a r) = 2 / (a p)

theorem problem_statement (k p r : ℕ) (h1 : sequence_sum a) (h2 : 1 < k) (h3 : k < p) (h4 : p < r) :
  arithmetic_sequence a k p r → (p = 2 * k - 1 ∧ r = 4 * k^2 - 5 * k + 2) :=
sorry

end problem_statement_l223_223383


namespace quadratic_roots_l223_223905

noncomputable def a := 1
noncomputable def b := 2

variable (ω : ℂ)
variable (h_ω_pow : ω^7 = 1)
variable (h_ω_ne_one : ω ≠ 1)

def α := ω + ω^2 + ω^4
def β := ω^3 + ω^5 + ω^6

theorem quadratic_roots :
  (α ω + β ω = -1) ∧ (α ω * β ω = 2) →
  (∃ (a b : ℝ), a = 1 ∧ b = 2 ∧ (x^2 + a*x + b = 0)) :=
by
  intro h
  use [a, b]
  sorry
 
end quadratic_roots_l223_223905


namespace triangle_area_correct_l223_223042

def triangle_area (ABC: Type) [Triangle ABC] (AF BE: ℝ) (altitude_c: ℝ) (perpendicular_medians: Prop) (G: Point) : ℝ :=
  let AG := (2 / 3) * AF,
      GF := (1 / 3) * AF,
      BG := (2 / 3) * BE,
      GE := (1 / 3) * BE,
      area_BGE := 1 / 2 * BG * GE,
      area_ABC := 6 * area_BGE
  in area_ABC

theorem triangle_area_correct : triangle_area ABC 10 15 12 (by sorry) G = 150 := sorry

end triangle_area_correct_l223_223042


namespace cost_price_of_watch_l223_223266

theorem cost_price_of_watch (CP : ℝ) (h_loss : 0.54 * CP = SP_loss)
                            (h_gain : 1.04 * CP = SP_gain)
                            (h_diff : SP_gain - SP_loss = 140) :
                            CP = 280 :=
by {
    sorry
}

end cost_price_of_watch_l223_223266
