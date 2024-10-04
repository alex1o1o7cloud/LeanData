import Mathlib

namespace value_of_a_l802_802287

theorem value_of_a
    (a b : ℝ)
    (h₁ : 0 < a ∧ 0 < b)
    (h₂ : a + b = 1)
    (h₃ : 21 * a^5 * b^2 = 35 * a^4 * b^3) :
    a = 5 / 8 :=
by
  sorry

end value_of_a_l802_802287


namespace midpoint_of_ab_intersects_pq_l802_802213

noncomputable theory

-- Definitions based on conditions.
variables {A B C X Y P Q : Type*}
variables (triangle : ∀ {A B C : Type*}, ∃ (P Q : Type*), 
  (acute_angle <| triangle_acute A B C) ∧
  (BC > AC) ∧
  (perpendicular_bisector AB ∩ BC = X) ∧
  (perpendicular_bisector AB ∩ AC = Y) ∧
  (projection X AC = P) ∧
  (projection Y BC = Q))

-- The theorem assertion stating the proof.
theorem midpoint_of_ab_intersects_pq 
  (A B C X Y P Q : Type*) 
  (h1 : triangle_acute A B C)
  (h2 : BC > AC)
  (h3 : perpendicular_bisector AB ∩ BC = X)
  (h4 : perpendicular_bisector AB ∩ AC = Y)
  (h5 : projection X AC = P)
  (h6 : projection Y BC = Q) :
  ∃ (M : midpoint A B), intersection PQ AB = M :=
sorry

end midpoint_of_ab_intersects_pq_l802_802213


namespace ellipse_equation_l802_802598

theorem ellipse_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
    (h4 : |F1F2| = 2 * sqrt 3) (h5 : |AB| = 4) : 
    (∃ a b : ℝ, ∃ F1 F2 : ℝ, ∃ A B : ℝ, 
        abs (a - F1) > 0 ∧
        abs (b - F2) > 0 ∧
        abs (a - b) > 0 ∧
        |F1 - F2| = 2 * sqrt 3 ∧
        |A - B| = 4 ∧
        (A ∘ B ⊥ F1 ∘ F2) )
        (∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1)) : 
    ∃ a b : ℝ, a = 3 ∧ b = sqrt 6 ∧ ∀ x y : ℝ, (x^2 / 9 + y^2 / 6 = 1) :=
sorry

end ellipse_equation_l802_802598


namespace book_ratio_problem_l802_802231

-- Definitions and conditions
variables {L R : ℕ} (D : ℕ := 20) (total_books : ℕ := 97)

-- Hypotheses/assumptions
def condition1 : Prop := R + 3 = L
def condition2 : Prop := L + R + D = total_books

-- Question to be proved
def correct_answer : Prop := L = 2 * D

theorem book_ratio_problem (h1 : condition1) (h2 : condition2) : correct_answer :=
by
  sorry

end book_ratio_problem_l802_802231


namespace centers_of_conics_form_conic_l802_802741

variables {A B C D : Point}

theorem centers_of_conics_form_conic (exists_conic_through_ABCD : ∃F, F = λ (AB * CD) + (BC * AD)) :
  ∃Γ, (∀conic F, F passes_through A B C D → center F ∈ Γ) :=
sorry

end centers_of_conics_form_conic_l802_802741


namespace fraction_is_positive_integer_if_and_only_if_positive_l802_802555

theorem fraction_is_positive_integer_if_and_only_if_positive (x : ℝ) (hx : x ≠ 0) :
  (∃ n : ℤ, n > 0 ∧ (|x + |x||) / x = n) ↔ x > 0 :=
by
  sorry

end fraction_is_positive_integer_if_and_only_if_positive_l802_802555


namespace recurring_decimal_to_fraction_l802_802502

theorem recurring_decimal_to_fraction (h1: (0.3 + 0.\overline{45} : ℝ) = (0.3\overline{45} : ℝ))
    (h2: (0.\overline{45} : ℝ) = (5 / 11 : ℝ))
    (h3: (0.3 : ℝ) = (3 / 10 : ℝ)) : (0.3\overline{45} : ℝ) = (83 / 110 : ℝ) :=
by
    sorry

end recurring_decimal_to_fraction_l802_802502


namespace find_z_l802_802091

noncomputable def z : ℂ := sorry

theorem find_z (z : ℂ) (hz1 : ((1 : ℂ) + 2*complex.I) * z ∈ ℝ)
  (hz2 : complex.abs z = real.sqrt 5) : z = 1 - 2*complex.I ∨ z = -1 + 2*complex.I :=
sorry

end find_z_l802_802091


namespace product_of_solutions_l802_802937

theorem product_of_solutions (x : ℝ) (h : |x| = 3 * (|x| - 2)) : (subs := (|x| == 3 ->  (x = 3)  ∨ (x = -3)): 
solution ( ∀ x:solution ∧ x₁ * x₂= 3 * (-3) )  : -9)   := 
sorry

end product_of_solutions_l802_802937


namespace factorization_identity_l802_802915

-- Define the initial expression
def initial_expr (x : ℝ) : ℝ := 2 * x^2 - 2

-- Define the factorized form
def factorized_expr (x : ℝ) : ℝ := 2 * (x + 1) * (x - 1)

-- The theorem stating the equality
theorem factorization_identity (x : ℝ) : initial_expr x = factorized_expr x := 
by sorry

end factorization_identity_l802_802915


namespace james_goals_product_l802_802185

theorem james_goals_product :
  ∃ (g7 g8 : ℕ), g7 < 7 ∧ g8 < 7 ∧ 
  (22 + g7) % 7 = 0 ∧ (22 + g7 + g8) % 8 = 0 ∧ 
  g7 * g8 = 24 :=
by
  sorry

end james_goals_product_l802_802185


namespace card_choice_count_l802_802487

theorem card_choice_count :
  ∃ (pairs : Finset (ℕ × ℕ)),
    (∀ (a b : ℕ), (a, b) ∈ pairs → 1 ≤ a ∧ a ≤ 50 ∧ 1 ≤ b ∧ b ≤ 50) ∧
    (∀ (a b : ℕ), (a, b) ∈ pairs → |a - b| = 11) ∧
    (∀ (a b : ℕ), (a, b) ∈ pairs → (a * b) % 5 = 0) ∧
    card pairs = 15 := 
sorry

end card_choice_count_l802_802487


namespace sum_of_integers_six_iterations_to_26_l802_802666

def machine_output (N : ℕ) : ℕ :=
  if N % 2 = 0 then N / 2 else 3 * N + 1

def iter_n_times (f : ℕ → ℕ) (N : ℕ) (k : ℕ) : ℕ :=
  nat.iterate f k N

theorem sum_of_integers_six_iterations_to_26 : 
  ∑ n in finset.filter (λ N : ℕ, iter_n_times machine_output N 6 = 26) (finset.range 2000), id = 3302 := 
sorry

end sum_of_integers_six_iterations_to_26_l802_802666


namespace surface_area_of_circumscribed_sphere_l802_802591

noncomputable def circumscribed_sphere_surface_area (EA EB AD : ℝ) (angle_AEB : ℝ) (condition_perpendicular_planes : Prop) : ℝ :=
  if condition_perpendicular_planes ∧ EA = 3 ∧ EB = 3 ∧ AD = 2 ∧ angle_AEB = 60 then
    16 * Real.pi
  else
    0

theorem surface_area_of_circumscribed_sphere (condition_perpendicular_planes : Prop) :
  ∀ (E A B D : Type)
    (EA EB AD : ℝ) (angle_AEB : ℝ)
    (h1 : condition_perpendicular_planes)
    (h2 : EA = 3)
    (h3 : EB = 3)
    (h4 : AD = 2)
    (h5 : angle_AEB = 60),
  circumscribed_sphere_surface_area EA EB AD angle_AEB condition_perpendicular_planes = 16 * Real.pi := 
by
  intros
  unfold circumscribed_sphere_surface_area
  rw [h1, h2, h3, h4, h5]
  simp
  sorry

end surface_area_of_circumscribed_sphere_l802_802591


namespace fourth_vertex_square_l802_802799

theorem fourth_vertex_square (z1 z2 z3 z4 : ℂ) 
    (h1 : z1 = 3 + 3 * complex.I) 
    (h2 : z2 = -1 + 4 * complex.I) 
    (h3 : z3 = -3 + complex.I) : 
    (∃ z4, z4 = 1 - 4 * complex.I ∧ 
           (z1 - z2).abs = (z2 - z3).abs ∧ 
           (z3 - z4).abs = (z4 - z1).abs ∧ 
           (z1 - z3).abs = (z2 - z4).abs ∧ 
           (z1 - z2).abs ≠ (z2 - z3).abs) :=
begin
  use 1 - 4 * complex.I,
  split,
  { refl, },
  { split,
    { sorry, },
    { split,
      { sorry, },
      { split,
        { sorry, },
        { sorry, }, }, }, },
end

end fourth_vertex_square_l802_802799


namespace percent_increase_is_125_l802_802388

-- Define the rectangle dimensions
def rect_length : ℝ := 12
def rect_width : ℝ := 8

-- Define the radii of the semicircles based on the sides of the rectangle
def radius_long_side : ℝ := rect_length / 2
def radius_short_side : ℝ := rect_width / 2

-- Define the areas of the semicircles
def area_semicircle (r : ℝ) : ℝ := (real.pi * r^2) / 2
def total_area_semicircles (r : ℝ) : ℝ := 2 * area_semicircle r

-- Calculate the areas for semicircles on long and short sides
def area_long_side_semicircles : ℝ := total_area_semicircles radius_long_side
def area_short_side_semicircles : ℝ := total_area_semicircles radius_short_side

-- Calculate the ratio of these areas
def ratio_areas : ℝ := area_long_side_semicircles / area_short_side_semicircles

-- Calculate the percent increase
def percent_increase : ℝ := (ratio_areas - 1) * 100

-- The main theorem
theorem percent_increase_is_125 : percent_increase = 125 := by
  sorry

end percent_increase_is_125_l802_802388


namespace fence_poles_count_l802_802374

def length_path : ℕ := 900
def length_bridge : ℕ := 42
def distance_between_poles : ℕ := 6

theorem fence_poles_count :
  2 * (length_path - length_bridge) / distance_between_poles = 286 :=
by
  sorry

end fence_poles_count_l802_802374


namespace card_choice_count_l802_802480

theorem card_choice_count :
  ∃ (pairs : Finset (ℕ × ℕ)),
    (∀ (a b : ℕ), (a, b) ∈ pairs → 1 ≤ a ∧ a ≤ 50 ∧ 1 ≤ b ∧ b ≤ 50) ∧
    (∀ (a b : ℕ), (a, b) ∈ pairs → |a - b| = 11) ∧
    (∀ (a b : ℕ), (a, b) ∈ pairs → (a * b) % 5 = 0) ∧
    card pairs = 15 := 
sorry

end card_choice_count_l802_802480


namespace empty_boxes_when_non_empty_34_l802_802081

theorem empty_boxes_when_non_empty_34 : 
  ∀ (n : ℕ), n = 34 → ∃ (empty_boxes : ℕ), empty_boxes = -1 + 6 * n :=
by 
  intro n hf,
  use (-1 + 6 * n),
  rw hf,
  have empty_boxes_value : -1 + 6 * 34 = 203 := by norm_num,
  exact empty_boxes_value

end empty_boxes_when_non_empty_34_l802_802081


namespace tangent_line_mean_value_l802_802118

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * (x + Real.log x)

theorem tangent_line (a : ℝ) : ∀ x : ℝ, 0 < x → (a + 1) * (2 * x - 1) = 2 * x + a * (1 + 1 / x) → @tangent ℝ ℝ _ _ (f a) := sorry

theorem mean_value (a : ℝ) : ∃ ξ : ℝ, 1 < ξ ∧ ξ < Real.exp 1 ∧ f' ξ = (f a (Real.exp 1) - f a 1) / (Real.exp 1 - 1) := sorry

end tangent_line_mean_value_l802_802118


namespace problem_l802_802972

variables (a b : EuclideanSpace ℝ _)

-- Hypotheses
hypothesis ha : ∥a∥ = 1
hypothesis hb : ∥b∥ = 1
hypothesis hangle : (a ⬝ b) = 1/2

-- Goal
theorem problem (a b : EuclideanSpace ℝ _) (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) (hangle : (a ⬝ b) = 1/2) :
  (2 • a - b) ⬝ b = 0 :=
sorry

end problem_l802_802972


namespace minimum_value_function_l802_802941

theorem minimum_value_function :
  ∀ x : ℝ, x ≥ 0 → (∃ y : ℝ, y = (3 * x^2 + 9 * x + 20) / (7 * (2 + x)) ∧
    (∀ z : ℝ, z ≥ 0 → (3 * z^2 + 9 * z + 20) / (7 * (2 + z)) ≥ y)) ∧
    (∃ x0 : ℝ, x0 = 0 ∧ y = (3 * x0^2 + 9 * x0 + 20) / (7 * (2 + x0)) ∧ y = 10 / 7) :=
by
  sorry

end minimum_value_function_l802_802941


namespace number_of_subsets_containing_at_least_one_cube_number_l802_802226

def M := finset.range 101
def is_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n
def cube_numbers := {1, 8, 27, 64}

theorem number_of_subsets_containing_at_least_one_cube_number :
  (finset.powerset M).card - (finset.powerset (M \ cube_numbers)).card = 2^100 - 2^96 := by
sorry

end number_of_subsets_containing_at_least_one_cube_number_l802_802226


namespace range_of_a_l802_802985

noncomputable def f (x : ℝ) : ℝ :=
if h : x > 0 then log x + 1 else -x^2 + 2 * x 

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |f x| ≥ a * x) ↔ -2 ≤ a ∧ a ≤ 0 := by
  sorry

end range_of_a_l802_802985


namespace candy_problem_l802_802400

theorem candy_problem (
  a : ℤ
) : (a % 10 = 6) →
    (a % 15 = 11) →
    (200 ≤ a ∧ a ≤ 250) →
    (a = 206 ∨ a = 236) :=
sorry

end candy_problem_l802_802400


namespace number_of_breaks_l802_802194

theorem number_of_breaks (pushups_per_10_sec : ℕ) (pushups_per_min_with_breaks : ℕ) 
  (time_per_break : ℕ) (total_time_sec : ℕ) : 
  pushups_per_10_sec = 5 → pushups_per_min_with_breaks = 22 → time_per_break = 8 → total_time_sec = 60 →
  (total_time_sec / (pushups_per_10_sec * (total_time_sec / 10))) - pushups_per_min_with_breaks = 
  (8 / 5) * 10 / time_per_break :=
begin
  intros h1 h2 h3 h4,
  have max_pushups_per_min : ℕ := 5 * (total_time_sec / 10),
  have missed_pushups : ℕ := max_pushups_per_min - pushups_per_min_with_breaks,
  have total_break_time : ℕ := (missed_pushups * 10) / 5,
  have total_breaks := total_break_time / time_per_break,
  exact total_breaks
end

end number_of_breaks_l802_802194


namespace mike_spent_total_l802_802714

-- Define the prices of the items
def price_trumpet : ℝ := 145.16
def price_song_book : ℝ := 5.84

-- Define the total amount spent
def total_spent : ℝ := price_trumpet + price_song_book

-- The theorem to be proved
theorem mike_spent_total :
  total_spent = 151.00 :=
sorry

end mike_spent_total_l802_802714


namespace pet_insurance_coverage_correct_l802_802305

noncomputable def tims_insurance_coverage (doctor_cost : ℕ) (insurance_percent : ℕ) (cat_cost : ℕ) (total_paid : ℕ) : ℕ :=
  let insurance_covered := (insurance_percent * doctor_cost) / 100
  let tim_paid := doctor_cost - insurance_covered
  let cat_paid := total_paid - tim_paid
  let pet_insurance_covered := cat_cost - cat_paid
  pet_insurance_covered

theorem pet_insurance_coverage_correct :
  tims_insurance_coverage 300 75 120 135 = 60 :=
by
  have insurance_covered := (75 * 300) / 100
  have tim_paid := 300 - insurance_covered
  have cat_paid := 135 - tim_paid
  have pet_insurance_covered := 120 - cat_paid
  show pet_insurance_covered = 60
  calc pet_insurance_covered
      = 120 - (135 - (300 - ((75 * 300) / 100))) : by rfl
  ... = 60 : by ring


end pet_insurance_coverage_correct_l802_802305


namespace find_rate_percent_l802_802818

-- Defining the simple interest problem setup
def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ := (P * R * T) / 100

-- The given conditions
def principal : ℝ := 800
def time : ℝ := 4
def interest : ℝ := 176

-- The equivalent Lean 4 proof statement
theorem find_rate_percent : ∃ R : ℝ, simple_interest principal R time = interest ∧ R = 5.5 :=
by
  sorry

end find_rate_percent_l802_802818


namespace cards_difference_product_divisibility_l802_802459

theorem cards_difference_product_divisibility :
  (∃ pairs : list (ℕ × ℕ), 
    (∀ p ∈ pairs, 1 ≤ p.1 ∧ p.1 ≤ 50 ∧ 1 ≤ p.2 ∧ p.2 ≤ 50 ∧ 
      (p.1 - p.2 = 11 ∨ p.2 - p.1 = 11) ∧ 
      (p.1 * p.2) % 5 = 0) ∧ 
    pairs.length = 15) := 
sorry

end cards_difference_product_divisibility_l802_802459


namespace sum_common_divisors_l802_802549

-- Define the sum of a set of numbers
def set_sum (s : Set ℕ) : ℕ :=
  s.fold (λ x acc => x + acc) 0

-- Define the divisors of a number
def divisors (n : ℕ) : Set ℕ :=
  { d | d > 0 ∧ n % d = 0 }

-- Definitions based on the given conditions
def divisors_of_60 : Set ℕ := divisors 60
def divisors_of_18 : Set ℕ := divisors 18
def common_divisors : Set ℕ := divisors_of_60 ∩ divisors_of_18

-- Declare the theorem to be proved
theorem sum_common_divisors : set_sum common_divisors = 12 :=
  sorry

end sum_common_divisors_l802_802549


namespace find_eagle_feathers_times_l802_802196

theorem find_eagle_feathers_times (x : ℕ) (hawk_feathers : ℕ) (total_feathers_before_give : ℕ) (total_feathers : ℕ) (left_after_selling : ℕ) :
  hawk_feathers = 6 →
  total_feathers_before_give = 6 + 6 * x →
  total_feathers = total_feathers_before_give - 10 →
  left_after_selling = total_feathers / 2 →
  left_after_selling = 49 →
  x = 17 :=
by
  intros h_hawk h_total_before_give h_total h_left h_after_selling
  sorry

end find_eagle_feathers_times_l802_802196


namespace differentiation_is_correct_l802_802338

-- Define the differentiable functions
def f (x : ℝ) := Real.log (x + 1)
def g (x : ℝ) := 3 * Real.exp x
def h (x : ℝ) := x^2 - 1/x
def j (x : ℝ) := x / Real.cos x

-- State the theorem that all the given differentiations are correct
theorem differentiation_is_correct :
  (deriv f = λ x, 1 / (x + 1)) ∧
  (deriv g = λ x, 3 * Real.exp x) ∧
  (deriv h = λ x, 2 * x + 1 / x^2) ∧
  (deriv j = λ x, (Real.cos x + x * Real.sin x) / (Real.cos x)^2) := 
by
  sorry

end differentiation_is_correct_l802_802338


namespace tax_revenue_at_90_optimal_tax_rate_l802_802383

noncomputable def market_supply_function (P : ℝ) : ℝ := 6 * P - 312

theorem tax_revenue_at_90 :
  let Q_s := market_supply_function 64 in
  Q_s * 90 = 6480 :=
by
  let Q_s := market_supply_function 64
  have h1 : Q_s = 72 := by
    simp [market_supply_function]
  have h2 : Q_s * 90 = 6480 := by
    rw [h1]
    norm_num
  exact h2

theorem optimal_tax_rate :
  let Q := 432 - 4 * 54 in
  Q * 54 = 10800 :=
by
  let Q := 432 - 4 * 54
  have h1 : Q = 216 := by
    norm_num
  have h2 : Q * 54 = 10800 := by
    rw [h1]
    norm_num
  exact h2

end tax_revenue_at_90_optimal_tax_rate_l802_802383


namespace incircle_tangency_condition_l802_802739

noncomputable theory

variables {A B C D : Type*}

def is_tangential (ABCD : ConvexQuadrilateral) : Prop :=
  ∃ (circle : Circle), circle.tangent_to_side ABCD.AB ∧
                       circle.tangent_to_side ABCD.BC ∧
                       circle.tangent_to_side ABCD.CD ∧
                       circle.tangent_to_side ABCD.DA 

theorem incircle_tangency_condition
  (ABCD : ConvexQuadrilateral)
  (h : segment_length ABCD.AB + segment_length ABCD.CD = segment_length ABCD.AD + segment_length ABCD.BC) :
  is_tangential ABCD :=
sorry

end incircle_tangency_condition_l802_802739


namespace cards_difference_product_divisibility_l802_802462

theorem cards_difference_product_divisibility :
  (∃ pairs : list (ℕ × ℕ), 
    (∀ p ∈ pairs, 1 ≤ p.1 ∧ p.1 ≤ 50 ∧ 1 ≤ p.2 ∧ p.2 ≤ 50 ∧ 
      (p.1 - p.2 = 11 ∨ p.2 - p.1 = 11) ∧ 
      (p.1 * p.2) % 5 = 0) ∧ 
    pairs.length = 15) := 
sorry

end cards_difference_product_divisibility_l802_802462


namespace tory_toys_sold_is_7_l802_802021

-- Define the conditions as Lean definitions
def bert_toy_phones_sold : Nat := 8
def price_per_toy_phone : Nat := 18
def bert_earnings : Nat := bert_toy_phones_sold * price_per_toy_phone
def tory_earnings : Nat := bert_earnings - 4
def price_per_toy_gun : Nat := 20
def tory_toys_sold := tory_earnings / price_per_toy_gun

-- Prove that the number of toy guns Tory sold is 7
theorem tory_toys_sold_is_7 : tory_toys_sold = 7 :=
by
  sorry

end tory_toys_sold_is_7_l802_802021


namespace monomials_count_l802_802190
open BigOperators

-- Defining the polynomial expression and the question
def polynomial_expr := (x + y + z) ^ 2032 + (x - y - z) ^ 2032

-- Statement of the problem in Lean 4
theorem monomials_count :
  -- The number of monomials with non-zero coefficients after expansion and combination
  count_monomials polynomial_expr = 1034289 := 
sorry

end monomials_count_l802_802190


namespace relationship_and_monotonic_intervals_range_of_a_l802_802122

noncomputable def f (a b x : ℝ) : ℝ :=
  a * (log (x + b)) / x

noncomputable def g (a x : ℝ) : ℝ :=
  x + 2 / x - a - 2

noncomputable def F (a b x : ℝ) : ℝ :=
  f a b x + g a x

theorem relationship_and_monotonic_intervals (a b : ℝ) (h1 : a ≤ 2) (h2 : a ≠ 0)
  (h3 : tangent_line (f a b) (1) = (line_through (1, f a b 1) (3, 0))) :
  (b = 2 * a) ∧ (
    (0 < a ≤ 2 → (∀ x : ℝ, (0 < x ∧ x < exp (-1)) → increasing (f a b) x)
    ∧ (∀ x : ℝ, (exp (-1) < x) → decreasing (f a b) x))
    ∧ (a < 0 → (∀ x : ℝ, (0 < x ∧ x < exp (-1)) → decreasing (f a b) x)
    ∧ (∀ x : ℝ, (exp (-1) < x) → increasing (f a b) x))
  ) :=
sorry

theorem range_of_a (a : ℝ) :
  (a = -1 ∨ a < -2 / log 2 ∨ (0 < a ∧ a ≤ 2)) →
  (∃ x ∈ Ioc 0 2, F a (2 * a) x = 0) :=
sorry

end relationship_and_monotonic_intervals_range_of_a_l802_802122


namespace initial_number_of_girls_l802_802647

theorem initial_number_of_girls (p : ℕ) (h1 : 0.5 * p = n) 
(h2 : (0.5 * p - 3) / (p + 1) = 0.4) 
(h3 : (0.5 * p - 4) / (p + 2) = 0.35) : 
n = 17 := 
sorry

end initial_number_of_girls_l802_802647


namespace find_m_l802_802281

variable (m : ℝ)

theorem find_m (h1 : 3 * (-7.5) - y = m) (h2 : -0.4 * (-7.5) + y = 3) : m = -22.5 :=
by
  sorry

end find_m_l802_802281


namespace card_pairs_with_conditions_l802_802455

theorem card_pairs_with_conditions : 
  let cards := Finset.range 51 in
  let pairs := (cards.product cards).filter (λ p, (p.1 - p.2).natAbs = 11 ∧ (p.1 * p.2) % 5 = 0) in
  pairs.card / 2 = 15 :=
by
  sorry

end card_pairs_with_conditions_l802_802455


namespace number_of_valid_permutations_l802_802656

noncomputable def is_valid_permutation (perm : List ℕ) : Bool :=
  -- Check if any three consecutive terms form a strictly increasing sequence
  (∀ i, i + 2 < perm.length → ¬ (perm[i] < perm[i+1] ∧ perm[i+1] < perm[i+2])) ∧
  -- Check if any three consecutive terms form a strictly decreasing sequence
  (∀ i, i + 2 < perm.length → ¬ (perm[i] > perm[i+1] ∧ perm[i+1] > perm[i+2]))

-- Define the sequence 1, 2, 3, 4, 5, 6
def sequence : List ℕ := [1, 2, 3, 4, 5, 6]

theorem number_of_valid_permutations : {perm | perm ~ sequence ∧ is_valid_permutation perm} = 4 := 
by
  sorry

end number_of_valid_permutations_l802_802656


namespace one_by_one_tile_center_or_boundary_l802_802169

-- Define the 7x7 grid and the positions of the 16 1x3 tiles and the 1 1x1 tile
structure Grid :=
  (size : ℕ := 7)

structure Tile :=
  (is_3x1  : Bool := false)
  (is_1x1  : Bool := false)
  (position: (ℕ × ℕ)) -- Position represented as a pair (x, y)

-- Define a predicate that captures the condition where a tile is at the center or on the boundary
def center_or_boundary (tile : Tile) (grid : Grid) : Prop :=
  let center  := (grid.size / 2, grid.size / 2)
  let on_boundary (x y : ℕ) := x = 0 ∨ y = 0 ∨ x = grid.size - 1 ∨ y = grid.size - 1
  (tile.position = center ∨ on_boundary tile.position.1 tile.position.2)

-- Theorem: The position of the 1x1 tile must be in the center or on the boundary of the grid.
theorem one_by_one_tile_center_or_boundary (grid : Grid)
  (tiles : List Tile)
  (h1 : tiles.length = 17)
  (h2 : ∃ t, t ∈ tiles ∧ t.is_1x1 = true ∧ 
              (t :: (tiles.filter (λ t, t.is_3x1)).length = 16)) :
  ∃ tile, tile.is_1x1 ∧ center_or_boundary tile grid :=
by {
  sorry
}

end one_by_one_tile_center_or_boundary_l802_802169


namespace find_x0_l802_802363

theorem find_x0 (f : ℝ → ℝ) (h1 : ∀ x, f x = x^3) (h2 : ∀ x, (deriv f) x = 3x^2) (hx : (deriv f) x0 = 3) :
  x0 = 1 ∨ x0 = -1 :=
sorry

end find_x0_l802_802363


namespace odd_K_exists_l802_802942

def divisor_count (n : ℕ) : ℕ := 
  if n = 1 then 1 else 
    (finset.range n).filter (λ d, d > 0 ∧ n % d = 0).card

theorem odd_K_exists (K : ℕ) (hK : K % 2 = 1) : 
  ∃ n : ℕ, divisor_count (n * n) / divisor_count n = K := 
by
  sorry

end odd_K_exists_l802_802942


namespace tiles_needed_l802_802152

def room_length_meters : ℝ := 9
def room_width_meters : ℝ := 6
def tile_side_length_dm : ℝ := 3

-- Convert dimensions from meters to decimeters
def room_length_dm := room_length_meters * 10
def room_width_dm := room_width_meters * 10

-- Calculate the areas
def room_area_dm2 := room_length_dm * room_width_dm
def tile_area_dm2 := tile_side_length_dm ^ 2

-- The number of tiles needed
def number_of_tiles := room_area_dm2 / tile_area_dm2

theorem tiles_needed : number_of_tiles = 600 := by
  sorry

end tiles_needed_l802_802152


namespace percent_of_y_equal_to_30_percent_of_60_percent_l802_802321

variable (y : ℝ)

theorem percent_of_y_equal_to_30_percent_of_60_percent (hy : y ≠ 0) :
  ((0.18 * y) / y) * 100 = 18 := by
  have hneq : y ≠ 0 := hy
  field_simp [hneq]
  norm_num
  sorry

end percent_of_y_equal_to_30_percent_of_60_percent_l802_802321


namespace polar_to_cartesian_l802_802777

theorem polar_to_cartesian (p θ : ℝ) (x y : ℝ) (hp : p = 8 * Real.cos θ)
  (hx : x = p * Real.cos θ) (hy : y = p * Real.sin θ) :
  x^2 + y^2 = 8 * x := 
sorry

end polar_to_cartesian_l802_802777


namespace remaining_walking_time_is_30_l802_802729

-- Define all the given conditions
def total_distance_to_store : ℝ := 2.5
def distance_already_walked : ℝ := 1.0
def time_per_mile : ℝ := 20.0

-- Define the target remaining walking time
def remaining_distance : ℝ := total_distance_to_store - distance_already_walked
def remaining_time : ℝ := remaining_distance * time_per_mile

-- Prove the remaining walking time is 30 minutes
theorem remaining_walking_time_is_30 : remaining_time = 30 :=
by
  -- Formal proof would go here using corresponding Lean tactics
  sorry

end remaining_walking_time_is_30_l802_802729


namespace fractional_rep_of_0_point_345_l802_802498

theorem fractional_rep_of_0_point_345 : 
  let x := (0.3 + (0.45 : ℝ)) in
  (x = (83 / 110 : ℝ)) :=
by
  sorry

end fractional_rep_of_0_point_345_l802_802498


namespace points_connected_in_D_l802_802960

variable {P : Type*} [NonconvexPolygon P] [NonselfintersectingPolygon P]

def D (P : Type*) : Set (Point P) := 
  {p : Point P | ∃ d : Diagonal P, p ∈ d ∨ p ∈ ∂d}

theorem points_connected_in_D (P : Type*) [DecidableEq (Point P)]
  (h_nonconvex : NonconvexPolygon P)
  (h_nonselfintersecting : NonselfintersectingPolygon P) 
  (p1 p2 : Point P) (h_p1 : p1 ∈ D P) (h_p2 : p2 ∈ D P) : 
  ∃ (l : List (Point P)), (∀ x ∈ l, x ∈ D P) ∧ l.head = p1 ∧ l.last = p2 :=
by
  sorry

end points_connected_in_D_l802_802960


namespace imaginary_part_of_complex_number_l802_802514

theorem imaginary_part_of_complex_number :
  let z := ((i-1)^2 + 4) / (i + 1) in
  complex.im z = -3 :=
by
  sorry

end imaginary_part_of_complex_number_l802_802514


namespace number_of_integers_between_sqrt10_and_sqrt100_l802_802145

theorem number_of_integers_between_sqrt10_and_sqrt100 :
  (∃ n : ℕ, n = 7) :=
sorry

end number_of_integers_between_sqrt10_and_sqrt100_l802_802145


namespace shirts_made_today_l802_802879

def shirts_per_minute : ℕ := 8
def working_minutes : ℕ := 2

theorem shirts_made_today (h1 : shirts_per_minute = 8) (h2 : working_minutes = 2) : shirts_per_minute * working_minutes = 16 := by
  sorry

end shirts_made_today_l802_802879


namespace number_placement_l802_802034

-- Define the conditions and the resultant positions
variables (A B C D E F : ℕ)
variables (neighbor_sum1 neighbor_sum2 neighbor_sum3: ℕ)
variables (white1 white2 white3: ℕ)

-- Main theorem to state the positions of each letter
theorem number_placement :
  (white1 = A + B + C ∧ white2 = D + E ∧ white3 = F) →
  (A = 1 ∧ B = 3 ∧ C = 2 ∧ D = 5 ∧ E = 6 ∧ F = 4) :=
begin
  sorry
end

end number_placement_l802_802034


namespace cos_alpha_value_l802_802794

-- Define the given problem setup
def point_through_angle (α : ℝ) : Prop :=
  let x := 1
  let y := -1
  let r := Real.sqrt (x^2 + y^2)
  ∀ cosα, cosα = x / r

-- The proof goal is to show that cosα = sqrt(2)/2 under the given setup
theorem cos_alpha_value (cosα : ℝ) (h : point_through_angle α) : cosα = Real.sqrt(2) / 2 :=
sorry

end cos_alpha_value_l802_802794


namespace problem_1_problem_2_problem_3_l802_802979

noncomputable def e : ℝ := Real.exp 1

def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ∈ Ioo (-e) 0 then a * x - Real.log (-x)
  else if x ∈ Ioo 0 e then a * x + Real.log x
  else 0 -- Placeholder for other domains

def g (x : ℝ) : ℝ :=
  Real.log (abs x) / abs x

theorem problem_1 {a : ℝ} :
  ∀ x : ℝ, x ≠ 0 → (f x a = if x ∈ Ioo (-e) 0 then a * x - Real.log (-x) else a * x + Real.log x) := 
by sorry

theorem problem_2 :
  ∀ x : ℝ, x ∈ Ioo (-e) 0 → f x (-1) > g x + 1 / 2 :=
by sorry

theorem problem_3 :
  (∃ a : ℝ, a = -Real.exp 2 ∧ ∀ x : ℝ, x ∈ Ioo (-e) 0 → f x a ≥ 3) :=
by sorry

end problem_1_problem_2_problem_3_l802_802979


namespace ellipse_focus_relation_l802_802634

variable (a b : ℝ)

-- Conditions
def is_ellipse (a b: ℝ) : Prop := (a > 0) ∧ (b < 0) ∧ (-b > a)

-- Statement to prove
theorem ellipse_focus_relation (a b : ℝ) (h : is_ellipse a b) : 
  sqrt (-b) > sqrt a :=
sorry

end ellipse_focus_relation_l802_802634


namespace at_least_one_fraction_less_than_two_l802_802563

theorem at_least_one_fraction_less_than_two {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := 
by
  sorry

end at_least_one_fraction_less_than_two_l802_802563


namespace arithmetic_sequence_difference_l802_802817

theorem arithmetic_sequence_difference :
  ∀ (a d : ℤ), a = -2 → d = 7 →
  |(a + (3010 - 1) * d) - (a + (3000 - 1) * d)| = 70 :=
by
  intros a d a_def d_def
  rw [a_def, d_def]
  sorry

end arithmetic_sequence_difference_l802_802817


namespace card_choice_count_l802_802486

theorem card_choice_count :
  ∃ (pairs : Finset (ℕ × ℕ)),
    (∀ (a b : ℕ), (a, b) ∈ pairs → 1 ≤ a ∧ a ≤ 50 ∧ 1 ≤ b ∧ b ≤ 50) ∧
    (∀ (a b : ℕ), (a, b) ∈ pairs → |a - b| = 11) ∧
    (∀ (a b : ℕ), (a, b) ∈ pairs → (a * b) % 5 = 0) ∧
    card pairs = 15 := 
sorry

end card_choice_count_l802_802486


namespace temperature_second_thermostat_l802_802184

variable (a b T₀ c : ℝ)
variable (a_val : a = 1.05) (b_val : b = 1.5) (T₀_val : T₀ = 298) (c_val : c = 1.28)
variable (T₂ : ℝ)

theorem temperature_second_thermostat :
  T₂ = (1.05 * 1.5 * 298 / 1.28) →
  T₂ = 366.7 :=
by
  intros hT₂
  rw [←a_val, ←b_val, ←T₀_val, ←c_val] at hT₂
  exact hT₂

-- Proof step
sorry

end temperature_second_thermostat_l802_802184


namespace cards_difference_product_divisibility_l802_802458

theorem cards_difference_product_divisibility :
  (∃ pairs : list (ℕ × ℕ), 
    (∀ p ∈ pairs, 1 ≤ p.1 ∧ p.1 ≤ 50 ∧ 1 ≤ p.2 ∧ p.2 ≤ 50 ∧ 
      (p.1 - p.2 = 11 ∨ p.2 - p.1 = 11) ∧ 
      (p.1 * p.2) % 5 = 0) ∧ 
    pairs.length = 15) := 
sorry

end cards_difference_product_divisibility_l802_802458


namespace hispanic_population_in_west_l802_802891

theorem hispanic_population_in_west (p_NE p_MW p_South p_West : ℕ)
  (h_NE : p_NE = 4)
  (h_MW : p_MW = 5)
  (h_South : p_South = 12)
  (h_West : p_West = 20) :
  ((p_West : ℝ) / (p_NE + p_MW + p_South + p_West : ℝ)) * 100 = 49 :=
by sorry

end hispanic_population_in_west_l802_802891


namespace diagonal_entries_cover_set_l802_802214

-- Variables and definitions used in the problem
variable {n : ℕ} (N : Finset ℕ) (f : ℕ → ℕ → ℕ)

-- Problem conditions
def isOdd (n : ℕ) : Prop := n % 2 = 1
def isSymmetric (f : ℕ → ℕ → ℕ) : Prop := ∀ r s, f(r,s) = f(s,r)
def isSurjectiveInRows (f : ℕ → ℕ → ℕ) (N : Finset ℕ) : Prop := 
  ∀ r ∈ N, (Finset.image (λ (s : ℕ), f r s) N) = N

-- Proof goal
theorem diagonal_entries_cover_set
  (hn : isOdd n)
  (hN : N = Finset.range (n+1))
  (hf_symm : isSymmetric f)
  (hf_surr : isSurjectiveInRows f N) :
  (Finset.image (λ r, f r r) N) = N :=
sorry

end diagonal_entries_cover_set_l802_802214


namespace number_of_permutations_l802_802804

-- Define the given set of digits
def digits : List Nat := [3, 4, 5, 6]

-- Define the permutation count problem
theorem number_of_permutations (h : digits.nodup) : digits.permutations.length = 24 :=
by {
  -- Provide a simple proof to ensure the theorem has the correct statement
  sorry
}

end number_of_permutations_l802_802804


namespace non_intersecting_segments_non_intersecting_quadrilaterals_l802_802946

-- Problem Part 1: Prove that it is possible to form n non-intersecting segments with 2n points.
theorem non_intersecting_segments (n : ℕ) (points : fin 2n → ℝ × ℝ)
  (h_no_three_collinear : ∀ i j k : fin 2n, i ≠ j → j ≠ k → i ≠ k → 
    ¬ collinear (points i) (points j) (points k)) : 
  ∃ segments : fin n → (ℝ × ℝ) × (ℝ × ℝ), 
    (∀ i j : fin n, i ≠ j → disjoint_segment (segments i) (segments j)) ∧
    (∀ i : fin 2n, ∃ j : fin n, i ∈ segment_vertices (segments j)) := 
sorry

-- Problem Part 2: Prove that it is possible to form n non-intersecting quadrilaterals with 4n points.
theorem non_intersecting_quadrilaterals (n : ℕ) (points : fin 4n → ℝ × ℝ)
  (h_no_three_collinear : ∀ i j k : fin 4n, i ≠ j → j ≠ k → i ≠ k → 
    ¬ collinear (points i) (points j) (points k)) : 
  ∃ quadrilaterals : fin n → (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ),
    (∀ i j : fin n, i ≠ j → disjoint_quadrilateral (quadrilaterals i) (quadrilaterals j)) ∧
    (∀ i : fin 4n, ∃ j : fin n, i ∈ quadrilateral_vertices (quadrilaterals j)) :=
sorry

end non_intersecting_segments_non_intersecting_quadrilaterals_l802_802946


namespace inequality_solution_l802_802518

theorem inequality_solution (a : ℝ) (h_neg : a < 0) (h_ineq : ∀ x : ℝ, sin x ^ 2 + a * cos x + a ^ 2 ≥ 1 + cos x) : a ≤ -2 :=
by
  -- This will be where the proof steps should be filled in
  sorry

end inequality_solution_l802_802518


namespace tangent_line_eq_monotonicity_l802_802987

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  log x + a * x + (a + 1) / x + 3

theorem tangent_line_eq (x : ℝ) (y : ℝ) (a : ℝ) (h : a = 1) (hx : x = 2) (hy : y = f 2 1) :
  x - y + log 2 + 4 = 0 :=
sorry

theorem monotonicity (a : ℝ) (h : a > -1/2) :
  (if -1/2 < a ∧ a < 0 then 
     (∀ x : ℝ, 0 < x ∧ x < 1 → deriv (f x a) < 0) ∧
     (∀ x : ℝ, -1-1/a < x ∨ 1 < x → deriv (f x a) > 0 ∧
      (∀ x : ℝ, 1 < x ∧ x < -1-1/a → deriv (f x a) < 0)) 
   else if a ≥ 0 then
     (∀ x : ℝ, 0 < x ∧ x < 1 → deriv (f x a) < 0) ∧ 
     (∀ x : ℝ, 1 < x → deriv (f x a) > 0) 
   else true) :=
sorry

end tangent_line_eq_monotonicity_l802_802987


namespace necessary_but_not_sufficient_l802_802700

def lines_parallel (a : ℝ) : Prop :=
  ∀ x y : ℝ, (a * x + 2 * y = 0) ↔ (x + (a + 1) * y + 4 = 0)

theorem necessary_but_not_sufficient (a : ℝ) :
  (a = 1 → lines_parallel a) ∧ ¬(lines_parallel a → a = 1) :=
by
  sorry

end necessary_but_not_sufficient_l802_802700


namespace solve_for_x_l802_802493

theorem solve_for_x : ∀ x : ℝ, ( (x * x^(2:ℝ)) ^ (1/6) )^2 = 4 → x = 4 := by
  intro x
  sorry

end solve_for_x_l802_802493


namespace sock_price_ratio_l802_802307

theorem sock_price_ratio
  (c : ℝ) (pg : ℝ) 
  (hb : ∀ pb, pb = 3 * pg)
  (hC_original : ∀ p_b, let Co := 5 * p_b + c * pg in Co = 5 * (3 * pg) + c * pg)
  (hC_interchanged : ∀ pg p_b Co, let Ci := c * p_b + 5 * pg in Ci = Ci = 1.8 * Co)
  : 5 / c = 3 / 11 := 
sorry

end sock_price_ratio_l802_802307


namespace total_number_of_mappings_l802_802362

open Set

def A : Set ℤ := {a, b, c}
def B : Set ℤ := {0, 1}

theorem total_number_of_mappings (A = {a, b, c}) (B = {0, 1}) : 
  (Finset.card (B.pow (Set.card A))) = 8 := by
  sorry

end total_number_of_mappings_l802_802362


namespace Tim_bottle_quarts_l802_802304

theorem Tim_bottle_quarts (ounces_per_week : ℕ) (ounces_per_quart : ℕ) (days_per_week : ℕ) (additional_ounces_per_day : ℕ) (bottles_per_day : ℕ) : 
  ounces_per_week = 812 → ounces_per_quart = 32 → days_per_week = 7 → additional_ounces_per_day = 20 → bottles_per_day = 2 → 
  ∃ quarts_per_bottle : ℝ, quarts_per_bottle = 1.5 := 
by
  intros hw ho hd ha hb
  let total_quarts_per_week := (812 : ℝ) / 32 
  let total_quarts_per_day := total_quarts_per_week / 7 
  let additional_quarts_per_day := 20 / 32 
  let quarts_from_bottles := total_quarts_per_day - additional_quarts_per_day 
  let quarts_per_bottle := quarts_from_bottles / 2 
  use quarts_per_bottle 
  sorry

end Tim_bottle_quarts_l802_802304


namespace distance_from_foci_to_asymptotes_eq_three_l802_802063

-- Define the hyperbola a^2, b^2, and the foci coordinates.
def a : ℝ := 4
def b : ℝ := 3
def c : ℝ := Real.sqrt (a^2 + b^2)

-- Define the hyperbola equation in Lean.
def hyperbola_eq (x y : ℝ) : Prop :=
  (x^2 / 16) - (y^2 / 9) = 1

-- Define the asymptote equations.
def asymptote1 (x y : ℝ) : Prop :=
  3 * x - 4 * y = 0

def asymptote2 (x y : ℝ) : Prop :=
  3 * x + 4 * y = 0

-- Define the foci coordinates.
def focus1 : ℝ × ℝ := (c, 0)
def focus2 : ℝ × ℝ := (-c, 0)

-- Define the distance formula from a point to a line.
def distance_from_focus_to_asymptote (A B C x₀ y₀ : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

-- Prove that the distance from the foci to the asymptotes is 3.
theorem distance_from_foci_to_asymptotes_eq_three :
  distance_from_focus_to_asymptote 3 (-4) 0 c 0 = 3 ∧
  distance_from_focus_to_asymptote 3 4  0 (-c) 0 = 3 :=
by
  sorry

end distance_from_foci_to_asymptotes_eq_three_l802_802063


namespace tan_alpha_eq_one_fifth_l802_802970

theorem tan_alpha_eq_one_fifth (α : ℝ) (h : (3 * sin (π + α) + cos (-α)) / (4 * sin (-α) - cos (9 * π + α)) = 2) : tan α = 1 / 5 :=
  sorry

end tan_alpha_eq_one_fifth_l802_802970


namespace monkey_climbing_distance_l802_802381

theorem monkey_climbing_distance
  (x : ℝ)
  (h1 : ∀ t : ℕ, t % 2 = 0 → t ≠ 0 → x - 3 > 0) -- condition (2,4)
  (h2 : ∀ t : ℕ, t % 2 = 1 → x > 0) -- condition (5)
  (h3 : 18 * (x - 3) + x = 60) -- condition (6)
  : x = 6 :=
sorry

end monkey_climbing_distance_l802_802381


namespace median_perpendicular_to_OI_l802_802095

-- Definitions for the given problem
variables {A B C O I E F M N P Q K G H : Type}
variables [Triangle ABC] [Circle O] [Circle I]
variables (BI : (IntersectLinePoint E (Line AC)))
variables (CI : (IntersectLinePoint F (Line AB)))
variables (circleE : Circle E tangents (OB at B))
variables (circleF : Circle F tangents (OC at C))
variables (ME: IntersectLineCircle P (Circumcircle O))
variables (NF: IntersectLineCircle Q (Circumcircle O))
variables (lineEF: Intersects K (Line BC))
variables (linePQ: IntersectLinePoint G (Line BC) ∧ IntersectLinePoint H (Line EF))

-- Problem statement
theorem median_perpendicular_to_OI : 
  median G in_triangle GKH ⊥ Line OI := sorry

end median_perpendicular_to_OI_l802_802095


namespace additional_male_workers_hired_l802_802298

theorem additional_male_workers_hired (W : ℕ) (H : W = 240) (h60 : ∀ (E_original : ℕ), 0.6 * E_original = 132) : 
  W - (132 / 0.6) = 20 := 
by
  -- Use conditions
  sorry

end additional_male_workers_hired_l802_802298


namespace train_length_180_meters_l802_802867

variable (speed_km_hr : ℕ) (time_sec : ℕ)
variable (conversion_factor : ℚ := (1000 / 3600))
variable (speed_m_s : ℚ := speed_km_hr * conversion_factor)
variable (length_of_train : ℚ := speed_m_s * time_sec)

theorem train_length_180_meters (h1 : speed_km_hr = 72) (h2 : time_sec = 9) :
  length_of_train = 180 := 
by 
  unfold length_of_train speed_m_s conversion_factor
  rw [h1, h2]
  norm_num
  rfl

end train_length_180_meters_l802_802867


namespace sum_common_divisors_sixty_and_eighteen_l802_802525

theorem sum_common_divisors_sixty_and_eighteen : 
  ∑ d in ({d ∈ ({1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60} : finset ℕ) | d ∈ ({1, 2, 3, 6, 9, 18} : finset ℕ)} : finset ℕ), d = 12 :=
by sorry

end sum_common_divisors_sixty_and_eighteen_l802_802525


namespace cards_difference_product_divisible_l802_802469

theorem cards_difference_product_divisible :
  let S := {n | 1 ≤ n ∧ n ≤ 50}
  let valid_pairs := {(a, b) | a ∈ S ∧ b ∈ S ∧ |a - b| = 11 ∧ (a * b) % 5 = 0}
  valid_pairs.card = 15 := 
sorry

end cards_difference_product_divisible_l802_802469


namespace find_number_l802_802825

theorem find_number (n : ℕ) (h1 : 45 = 11 * n + 1) : n = 4 :=
  sorry

end find_number_l802_802825


namespace angle_bisector_of_bisected_angle_l802_802865

open EuclideanGeometry -- For geometric constructs and definitions

-- Definitions used in conditions
variable (A B C D M : Point)
variable (h₁ : diameter AD)
variable (h₂ : semicircleDiameterTouches BC AD M)

-- Prove statement
theorem angle_bisector_of_bisected_angle :
  angleBisector AM (angle BAC) :=
sorry

end angle_bisector_of_bisected_angle_l802_802865


namespace percentage_first_pay_cut_l802_802255

-- Define the parameters and the conditions
variables (S : ℝ) (P : ℝ)

-- Define the three pay cuts and the overall decrease
def final_salary_after_cuts := S * (1 - P / 100) * (1 - 10 / 100) * (1 - 15 / 100)
def overall_final_salary := S * (1 - 27.325 / 100)

-- Statement of the problem
theorem percentage_first_pay_cut (h : final_salary_after_cuts S P = overall_final_salary S) : P = 5 :=
sorry

end percentage_first_pay_cut_l802_802255


namespace find_annual_interest_rate_l802_802880

noncomputable def compound_interest_rate (P A : ℝ) (n t : ℕ) : ℝ := 
  let num := A / P
  let denom := (1 + real.sqrt num) / (real.sqrt (num))
  (denom - 1) * n

theorem find_annual_interest_rate (P A : ℝ) (n t : ℕ) (hP : P = 600) (hA : A = 661.5) (hn : n = 2) (ht : t = 1) : 
  compound_interest_rate P A n t ≈ 0.100952 :=
by
  rw [hP, hA, hn, ht]
  unfold compound_interest_rate
  sorry

end find_annual_interest_rate_l802_802880


namespace solve_x_l802_802262

theorem solve_x (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 8 * x ^ 2 + 16 * x * y = x ^ 3 + 3 * x ^ 2 * y) (h₄ : y = 2 * x) : x = 40 / 7 :=
by
  sorry

end solve_x_l802_802262


namespace num_values_of_a_l802_802612

def A : Set ℝ := {-1, 1, 3}
def B (a : ℝ) : Set ℝ := {1, a^2 - 2 * a}

theorem num_values_of_a : ∃v : Finset ℝ, (∀ a ∈ v, B a ⊆ A) ∧ v.card = 3 :=
by
  sorry

end num_values_of_a_l802_802612


namespace action_figure_total_l802_802672

variable (initial_figures : ℕ) (added_figures : ℕ)

theorem action_figure_total (h₁ : initial_figures = 8) (h₂ : added_figures = 2) : (initial_figures + added_figures) = 10 := by
  sorry

end action_figure_total_l802_802672


namespace positive_integer_solutions_l802_802927

theorem positive_integer_solutions (n x y z : ℕ) (h1 : n > 1) (h2 : n^z < 2001) (h3 : n^x + n^y = n^z) :
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 10 ∧ x = k ∧ y = k ∧ z = k + 1) :=
sorry

end positive_integer_solutions_l802_802927


namespace largest_unattainable_amount_l802_802411

theorem largest_unattainable_amount (a b : ℕ) (coprime : Nat.gcd a b = 1) : ∀ (n : ℕ), a = 8 → b = 15 → n = a * b - a - b → n = 97 :=
by
  intros a b coprime n ha hb hn
  rw [ha, hb] at *
  conv_lhs {ring}
  exact hn
  sorry

end largest_unattainable_amount_l802_802411


namespace nina_money_l802_802239

theorem nina_money (W : ℝ) (P: ℝ) (Q : ℝ) 
  (h1 : P = 6 * W)
  (h2 : Q = 8 * (W - 1))
  (h3 : P = Q) 
  : P = 24 := 
by 
  sorry

end nina_money_l802_802239


namespace trajectory_of_G_l802_802586

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define the trajectory equation
def trajectory (x y : ℝ) : Prop :=
  9 * x^2 / 4 + 3 * y^2 = 1

-- State the theorem
theorem trajectory_of_G (P G : ℝ × ℝ) (hP : ellipse P.1 P.2) (hG_relation : ∃ k : ℝ, k = 2 ∧ P = (3 * G.1, 3 * G.2)) :
  trajectory G.1 G.2 :=
by
  sorry

end trajectory_of_G_l802_802586


namespace joyce_has_40_bananas_l802_802684

theorem joyce_has_40_bananas (boxes : ℕ) (bananas_per_box : ℕ) (total_bananas : ℕ) : 
  (boxes = 10) -> 
  (bananas_per_box = 4) -> 
  (total_bananas = boxes * bananas_per_box) -> 
  total_bananas = 40 := 
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3.symm

end joyce_has_40_bananas_l802_802684


namespace solve_for_x_minus_y_l802_802835

theorem solve_for_x_minus_y (x y : ℝ) 
  (h1 : 3 * x - 5 * y = 5)
  (h2 : x / (x + y) = 5 / 7) : 
  x - y = 3 := 
by 
  -- Proof would go here
  sorry

end solve_for_x_minus_y_l802_802835


namespace find_number_l802_802161

theorem find_number (N Q : ℕ) (h1 : N = 5 * Q) (h2 : Q + N + 5 = 65) : N = 50 :=
by
  sorry

end find_number_l802_802161


namespace equation_represents_line_passing_through_P2_and_parallel_to_l_l802_802978

noncomputable def equation_of_line (f : ℝ × ℝ → ℝ) (x y : ℝ) : Prop := f (x, y) = 0

def point_on_line (f : ℝ × ℝ → ℝ) (x1 y1 : ℝ) : Prop := equation_of_line f x1 y1

def point_off_line (f : ℝ × ℝ → ℝ) (x2 y2 : ℝ) : Prop := ¬ equation_of_line f x2 y2

theorem equation_represents_line_passing_through_P2_and_parallel_to_l 
  (f : ℝ × ℝ → ℝ) 
  (x y x1 y1 x2 y2 : ℝ) 
  (h1 : equation_of_line f x y) 
  (h2 : point_on_line f x1 y1) 
  (h3 : point_off_line f x2 y2) :
  ∀ x' y', f (x', y') - f (x1, y1) - f (x2, y2) = 0 ↔ f (x', y') - f (x2, y2) = 0 :=
begin
  sorry
end

end equation_represents_line_passing_through_P2_and_parallel_to_l_l802_802978


namespace trig_expression_l802_802088

theorem trig_expression (α : ℝ) (h : Real.tan α = 2) : 
  (3 * Real.sin α - Real.cos α) / (2 * Real.sin α + 3 * Real.cos α) = 5 / 7 :=
by
  sorry

end trig_expression_l802_802088


namespace proof_problem_l802_802841

noncomputable def solve_quadratic : Prop :=
  let Δ := (-5)^2 - 4 * 1 * 1 in
  Δ = 21 ∧
  (∀ x, x^2 - 5*x + 1 = 0 ↔ x = (5 + real.sqrt 21) / 2 ∨ x = (5 - real.sqrt 21) / 2)

noncomputable def solve_inequality_system : Prop :=
  (∀ x, x + 8 < 4 * x - 1 ↔ x > 3) ∧
  (∀ x, (1 / 2) * x ≤ 8 - (3 / 2) * x ↔ x ≤ 4) ∧
  (∀ x, (x + 8 < 4 * x - 1) ∧ ((1 / 2) * x ≤ 8 - (3 / 2) * x) ↔ 3 < x ∧ x ≤ 4)

theorem proof_problem :
  solve_quadratic ∧ solve_inequality_system :=
sorry

end proof_problem_l802_802841


namespace divisible_by_six_l802_802629

theorem divisible_by_six (n a b : ℕ) (h1 : 2^n = 10 * a + b) (h2 : n > 3) (h3 : b > 0) (h4 : b < 10) : 6 ∣ (a * b) := 
sorry

end divisible_by_six_l802_802629


namespace rectangle_side_difference_l802_802640

theorem rectangle_side_difference (p d x y : ℝ) (h1 : 2 * x + 2 * y = p)
                                   (h2 : x^2 + y^2 = d^2)
                                   (h3 : x = 2 * y) :
    x - y = p / 6 := 
sorry

end rectangle_side_difference_l802_802640


namespace math_problem_l802_802492

noncomputable def harmonic (n : ℕ) : ℚ :=
  ∑ k in Finset.range n.succ, 1 / (k + 1)

theorem math_problem :
  (∑ k in Finset.range 2023, (2024 - (k + 1)) / (k + 1)) /
  ∑ k in Finset.range (2024 - 1), 1 / (k + 2) = 2024 := by
  sorry

end math_problem_l802_802492


namespace child_haircut_cost_l802_802754

/-
Problem Statement:
- Women's haircuts cost $48.
- Tayzia and her two daughters get haircuts.
- Tayzia wants to give a 20% tip to the hair stylist, which amounts to $24.
Question: How much does a child's haircut cost?
-/

noncomputable def cost_of_child_haircut (C : ℝ) : Prop :=
  let women's_haircut := 48
  let tip := 24
  let total_cost_before_tip := women's_haircut + 2 * C
  total_cost_before_tip * 0.20 = tip ∧ total_cost_before_tip = 120 ∧ C = 36

theorem child_haircut_cost (C : ℝ) (h1 : cost_of_child_haircut C) : C = 36 :=
  by sorry

end child_haircut_cost_l802_802754


namespace count_integers_between_sqrts_l802_802143

theorem count_integers_between_sqrts (a b : ℝ) (h1 : a = 10) (h2 : b = 100) :
  let lower_bound := Int.ceil (Real.sqrt a),
      upper_bound := Int.floor (Real.sqrt b) in
  (upper_bound - lower_bound + 1) = 7 := 
by
  rw [h1, h2]
  let lower_bound := Int.ceil (Real.sqrt 10)
  let upper_bound := Int.floor (Real.sqrt 100)
  have h_lower : lower_bound = 4 := by sorry
  have h_upper : upper_bound = 10 := by sorry
  rw [h_lower, h_upper]
  norm_num
  sorry

end count_integers_between_sqrts_l802_802143


namespace jeans_and_shirts_l802_802022

-- Let's define the necessary variables and conditions.
variables (J S X : ℝ)

-- Given conditions
def condition1 := 3 * J + 2 * S = X
def condition2 := 2 * J + 3 * S = 61

-- Given the price of one shirt
def price_of_shirt := S = 9

-- The problem we need to prove
theorem jeans_and_shirts : condition1 J S X ∧ condition2 J S ∧ price_of_shirt S →
  X = 69 :=
by
  sorry

end jeans_and_shirts_l802_802022


namespace number_of_integers_between_sqrt10_and_sqrt100_l802_802147

theorem number_of_integers_between_sqrt10_and_sqrt100 :
  (∃ n : ℕ, n = 7) :=
sorry

end number_of_integers_between_sqrt10_and_sqrt100_l802_802147


namespace goldbach_problem_l802_802886

noncomputable def A := {3, 5, 11, 17, 19}

def total_pairs := (A.toFinset.card.choose 2 : ℚ)

def favorable_pairs := ([(3, 17), (3, 19), (5, 17), (5, 19)].length : ℚ)

theorem goldbach_problem : (favorable_pairs / total_pairs) = 2 / 5 := by
  sorry

end goldbach_problem_l802_802886


namespace sum_a_i_l802_802623

theorem sum_a_i (a : Fin 2025 → ℝ)
  (h : ∀ x : ℝ, (1 + x) * (1 - 2 * x)^2023 = ∑ i in Finset.range 2025, a i * x^i) :
  ∑ i in Finset.range 2024, a (i+1) = -3 := by
  sorry

end sum_a_i_l802_802623


namespace expected_number_of_groups_l802_802782

-- Define the conditions
variables (k m : ℕ) (h : 0 < k ∧ 0 < m)

-- Expected value of groups in the sequence
theorem expected_number_of_groups : 
  ∀ k m, (0 < k) → (0 < m) → 
  let total_groups := 1 + (2 * k * m) / (k + m) in total_groups = 1 + (2 * k * m) / (k + m) :=
by
  intros k m hk hm
  let total_groups := 1 + (2 * k * m) / (k + m)
  exact (rfl : total_groups = 1 + (2 * k * m) / (k + m))

end expected_number_of_groups_l802_802782


namespace sum_of_coefficients_l802_802620

theorem sum_of_coefficients : 
  ∀ (a : Fin 2025 → ℝ),
    (∀ x : ℝ, (1 + x) * (1 - 2 * x)^2023 = ∑ i in Finset.range 2025, a i * (x ^ i)) →
    a 0 = 1 →
    (∑ i in Finset.range 2025, a i) = -2 →
    (∑ i in Finset.range 2024, a i.succ) = -3 := 
by 
  sorry

end sum_of_coefficients_l802_802620


namespace probability_60_or_more_points_l802_802370

theorem probability_60_or_more_points :
  let five_choose k := Nat.choose 5 k
  let prob_correct (k : Nat) := (five_choose k) * (1 / 2)^5
  let prob_at_least_3_correct := prob_correct 3 + prob_correct 4 + prob_correct 5
  prob_at_least_3_correct = 1 / 2 := 
sorry

end probability_60_or_more_points_l802_802370


namespace Yolanda_husband_catches_up_in_15_minutes_l802_802346

theorem Yolanda_husband_catches_up_in_15_minutes :
  ∀ (x : ℕ),
    (∀ (y_time h_speed : ℕ), y_time = x + 15 → h_speed = 40 →
      (20 * (y_time) / 60) = (40 * x / 60)) →
    x = 15 :=
by
  intros x h
  have h₁ : 20 * (x + 15) / 60 = 40 * x / 60 := by
    apply h x 40 rfl rfl
  sorry

end Yolanda_husband_catches_up_in_15_minutes_l802_802346


namespace necessary_but_not_sufficient_l802_802790

theorem necessary_but_not_sufficient (x y : ℝ) :
  (x > 1 ∧ y > 1) → (x + y > 2) ∧ (∃ x y : ℝ, (x + y > 2) ∧ (x ≤ 1 ∨ y ≤ 1)) :=
by
  intro h
  -- proof for necessary condition
  have hxy := add_lt_add h.1 h.2
  exact (lt_trans hxy (by norm_num)),
  -- proof for not sufficient condition
  use [0.5, 1.6]
  split
  { norm_num },
  { left,
    linarith }

end necessary_but_not_sufficient_l802_802790


namespace value_of_g_at_13_l802_802627

-- Define the function g
def g (n : ℕ) : ℕ := n^2 + n + 23

-- The theorem to prove
theorem value_of_g_at_13 : g 13 = 205 := by
  -- Rewrite using the definition of g
  unfold g
  -- Perform the arithmetic
  sorry

end value_of_g_at_13_l802_802627


namespace card_pairs_satisfying_conditions_l802_802439

theorem card_pairs_satisfying_conditions :
  ∃ (s : Finset (ℕ × ℕ)),
    (∀ p ∈ s, p.1 ≠ p.2 ∧ (p.1 = p.2 + 11 ∨ p.2 = p.1 + 11) ∧ ((p.1 * p.2) % 5 = 0))
    ∧ s.card = 15 :=
begin
  sorry
end

end card_pairs_satisfying_conditions_l802_802439


namespace quadratic_solutions_l802_802789

theorem quadratic_solutions :
  ∀ x : ℝ, x * (x - 3) = 0 → (x = 0 ∨ x = 3) := 
by
  intro x h
  have h1 : x = 0 ∨ x - 3 = 0 := by sorry
  cases h1 with h2 h3
  · left
    exact h2
  · right
    exact eq_add_of_sub_eq_ h3

end quadratic_solutions_l802_802789


namespace remainder_of_towers_l802_802851

open Nat

def count_towers (m : ℕ) : ℕ :=
  match m with
  | 1 => 1
  | 2 => 2
  | 3 => 6
  | 4 => 18
  | 5 => 54
  | 6 => 162
  | _ => 0

theorem remainder_of_towers : (count_towers 6) % 100 = 62 :=
  by
  sorry

end remainder_of_towers_l802_802851


namespace product_range_l802_802172

theorem product_range (m b : ℚ) (h₀ : m = 3 / 4) (h₁ : b = 6 / 5) : 0 < m * b ∧ m * b < 1 :=
by
  sorry

end product_range_l802_802172


namespace distance_between_points_l802_802201

noncomputable theory

-- Define the points in polar coordinates
def point_A (θ₁ : ℝ) : ℝ × ℝ := (4, θ₁)
def point_B (θ₂ : ℝ) : ℝ × ℝ := (6, θ₂)

-- Define the condition for the angles
def angle_condition (θ₁ θ₂ : ℝ) : Prop := θ₁ - θ₂ = π / 3

-- Define the function to calculate the distance
def distance_AB (θ₁ θ₂ : ℝ) : ℝ := 
  let AB_squared := 4^2 + 6^2 - 2 * 4 * 6 * Real.cos(π / 3)
  Real.sqrt AB_squared

-- The theorem statement
theorem distance_between_points (θ₁ θ₂ : ℝ) (h : angle_condition θ₁ θ₂) :
  distance_AB θ₁ θ₂ = 2 * Real.sqrt 7 :=
by
  sorry

end distance_between_points_l802_802201


namespace isosceles_triangle_base_angle_l802_802182

theorem isosceles_triangle_base_angle (a b h θ : ℝ)
  (h1 : a^2 = 4 * b^2 * h)
  (h_b : b = 2 * a * Real.cos θ)
  (h_h : h = a * Real.sin θ) :
  θ = Real.arccos (1/4) :=
by
  sorry

end isosceles_triangle_base_angle_l802_802182


namespace card_pairs_count_l802_802475

theorem card_pairs_count :
  let cards := {n : ℕ | 1 ≤ n ∧ n ≤ 50}
  let valid_pairs := {(a, b) ∈ cards × cards | 
    a ≠ b ∧ (a - b = 11 ∨ b - a = 11) ∧ (a * b) % 5 = 0}
  in valid_pairs.count = 15 :=
by
  sorry

end card_pairs_count_l802_802475


namespace percentage_increase_selling_price_l802_802645

-- Defining the conditions
def original_price : ℝ := 6
def increased_price : ℝ := 8.64
def total_sales_per_hour : ℝ := 216
def max_price : ℝ := 10

-- Statement for Part 1
theorem percentage_increase (x : ℝ) : 6 * (1 + x)^2 = 8.64 → x = 0.2 :=
by
  sorry

-- Statement for Part 2
theorem selling_price (a : ℝ) : (6 + a) * (30 - 2 * a) = 216 → 6 + a ≤ 10 → 6 + a = 9 :=
by
  sorry

end percentage_increase_selling_price_l802_802645


namespace minimal_team_members_l802_802652

theorem minimal_team_members (n : ℕ) : 
  (n ≡ 1 [MOD 6]) ∧ (n ≡ 2 [MOD 8]) ∧ (n ≡ 3 [MOD 9]) → n = 343 := 
by
  sorry

end minimal_team_members_l802_802652


namespace count_valid_pairs_l802_802441

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

def difference_is_11 (a b : ℕ) : Prop := abs (a - b) = 11

def valid_card (a b : ℕ) : Prop :=
  a ∈ finset.range 51 ∧ b ∈ finset.range 51 ∧
  difference_is_11 a b ∧ (is_divisible_by_5 a ∨ is_divisible_by_5 b)

theorem count_valid_pairs : finset.card {p : ℕ × ℕ | valid_card p.1 p.2 ∧ p.1 < p.2}.to_finset = 15 := 
  sorry

end count_valid_pairs_l802_802441


namespace find_B_l802_802643

-- Definitions based on the problem conditions
variable {A B C a b c : ℝ}
variable {S : ℝ}
variable {angle_B : ℝ}

-- The lengths of the sides of the triangle
axiom h1 : a = 10
axiom h2 : b = 12
axiom h3 : c = 13

-- Area of the triangle and the given condition
axiom h4 : S = (b * b + c * c - a * a) / 4
axiom h5 : b * sin (B) - c * sin (C) = a

-- Statement of what is to be proved
theorem find_B : B = 77.5 :=
by
  sorry

end find_B_l802_802643


namespace determine_digit_l802_802037

theorem determine_digit (Θ : ℚ) (h : 312 / Θ = 40 + 2 * Θ) : Θ = 6 :=
sorry

end determine_digit_l802_802037


namespace Harriet_sibling_product_l802_802140

-- Definition of the family structure
def Harry : Prop := 
  let sisters := 4
  let brothers := 4
  true

-- Harriet being one of Harry's sisters and calculating her siblings
def Harriet : Prop :=
  let S := 4 - 1 -- Number of Harriet's sisters
  let B := 4 -- Number of Harriet's brothers
  S * B = 12

theorem Harriet_sibling_product : Harry → Harriet := by
  intro h
  let S := 3
  let B := 4
  have : S * B = 12 := by norm_num
  exact this

end Harriet_sibling_product_l802_802140


namespace card_choice_count_l802_802482

theorem card_choice_count :
  ∃ (pairs : Finset (ℕ × ℕ)),
    (∀ (a b : ℕ), (a, b) ∈ pairs → 1 ≤ a ∧ a ≤ 50 ∧ 1 ≤ b ∧ b ≤ 50) ∧
    (∀ (a b : ℕ), (a, b) ∈ pairs → |a - b| = 11) ∧
    (∀ (a b : ℕ), (a, b) ∈ pairs → (a * b) % 5 = 0) ∧
    card pairs = 15 := 
sorry

end card_choice_count_l802_802482


namespace tangents_perpendicular_and_intersection_on_directrix_l802_802047

variables {F : Type*} [field F]

structure Parabola (F : Type*) [field F] :=
(focus : F × F)
(directrix : set (F × F))
(vertex : F × F)
(axis : F)

structure Tangent (P : Parabola F) :=
(point : F × F)
(line : F → F)

noncomputable def intersection (t1 t2 : Tangent P) : F × F := sorry

theorem tangents_perpendicular_and_intersection_on_directrix
  (P : Parabola F)
  (A B : F × F)
  (tA : Tangent P)
  (tB : Tangent P)
  (hA : tA.point = A)
  (hB : tB.point = B)
  (chord_through_focus : A.fst = P.focus.fst ∧ B.fst = P.focus.fst) :
  (tA.line ⊥ tB.line) ∧ (intersection tA tB ∈ P.directrix) :=
sorry

end tangents_perpendicular_and_intersection_on_directrix_l802_802047


namespace max_distance_MN_l802_802608

noncomputable def f (x : ℝ) := 2 * Real.sin x
noncomputable def g (x : ℝ) := Real.sin (Real.pi / 2 - x)

theorem max_distance_MN :
  ∃ x : ℝ, abs (f x - g x) = sqrt 5 := by sorry

end max_distance_MN_l802_802608


namespace necessary_but_not_sufficient_l802_802198

-- Definitions based on conditions in a)
variables (a b c x y z S: ℝ)
variables (ABC_acute: ∀ (A B C : ℝ), A < 90 ∧ B < 90 ∧ C < 90)
variables (triangle_sides: (a > 0) ∧ (b > 0) ∧ (c > 0))
variables (triangle_area: S > 0)
variables (d1: ∃ (P: ℝ), true)
variables (areas: ℝ) (S1 S2 S3: ℝ)

-- The theorem statement using above conditions
theorem necessary_but_not_sufficient 
  (h1: (S1 + S2 + S3 = S))
  (h2: (triangle_sides))
  (h3: (∀ (P: ℝ), true)):
  ¬ (∃ (P : ℝ), true) :=
sorry

end necessary_but_not_sufficient_l802_802198


namespace books_sold_l802_802246

def initial_books : ℕ := 134
def given_books : ℕ := 39
def books_left : ℕ := 68

theorem books_sold : (initial_books - given_books - books_left = 27) := 
by 
  sorry

end books_sold_l802_802246


namespace tournament_committees_count_l802_802661

-- Definitions corresponding to the conditions
def num_teams : ℕ := 4
def team_size : ℕ := 8
def members_selected_by_winning_team : ℕ := 3
def members_selected_by_other_teams : ℕ := 2

-- Binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Counting the number of possible committees
def total_committees : ℕ :=
  let num_ways_winning_team := binom team_size members_selected_by_winning_team
  let num_ways_other_teams := binom team_size members_selected_by_other_teams
  num_teams * num_ways_winning_team * (num_ways_other_teams ^ (num_teams - 1))

-- The statement to be proved
theorem tournament_committees_count : total_committees = 4917248 := by
  sorry

end tournament_committees_count_l802_802661


namespace exists_five_digit_with_product_10080_l802_802313

noncomputable def has_digit_product_10080 (n : ℕ) : Prop :=
  (n.digits 10).prod = 10080

theorem exists_five_digit_with_product_10080 : 
  ∃ n : ℕ, n < 100000 ∧ 10000 ≤ n ∧ has_digit_product_10080 n ∧ n = 98754 :=
by
  sorry

end exists_five_digit_with_product_10080_l802_802313


namespace percent_calculation_l802_802329

theorem percent_calculation (y : ℝ) : (0.3 * (0.6 * y) = 0.18 * y) → (0.18 * y / y) * 100 = 18 := by
  sorry

end percent_calculation_l802_802329


namespace problem_l802_802910

theorem problem
    (a b c d : ℕ)
    (h1 : a = b + 7)
    (h2 : b = c + 15)
    (h3 : c = d + 25)
    (h4 : d = 90) :
  a = 137 := by
  sorry

end problem_l802_802910


namespace num_supervisors_correct_l802_802646

theorem num_supervisors_correct (S : ℕ) 
  (avg_sal_total : ℕ) (avg_sal_supervisor : ℕ) (avg_sal_laborer : ℕ) (num_laborers : ℕ)
  (h1 : avg_sal_total = 1250) 
  (h2 : avg_sal_supervisor = 2450) 
  (h3 : avg_sal_laborer = 950) 
  (h4 : num_laborers = 42) 
  (h5 : avg_sal_total = (39900 + S * avg_sal_supervisor) / (num_laborers + S)) : 
  S = 10 := by sorry

end num_supervisors_correct_l802_802646


namespace tom_tickets_l802_802415

theorem tom_tickets :
  (45 + 38 + 52) - (12 + 23) = 100 := by
sorry

end tom_tickets_l802_802415


namespace smallest_positive_period_minimum_value_on_interval_l802_802988

noncomputable def f (x : ℝ) : ℝ := sqrt 2 * (sin (x / 2) * cos (x / 2) - sin (x / 2) ^ 2)

theorem smallest_positive_period : 
  (∀ x, f (x + 2 * Real.pi) = f x) ∧
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ 2 * Real.pi) :=
begin
  sorry
end

theorem minimum_value_on_interval : 
  - Real.pi ≤ x ∧ x ≤ 0 → f x ≥ - (sqrt 2 / 2) - 1 ∧ 
  (∃ x', -Real.pi ≤ x' ∧ x' ≤ 0 ∧ f x' = - (sqrt 2 / 2) - 1) :=
begin
  sorry
end

end smallest_positive_period_minimum_value_on_interval_l802_802988


namespace sum_of_common_divisors_60_18_l802_802535

def positive_divisors (n : ℕ) : List ℕ :=
  List.filter (λ d, n % d = 0) (List.range (n+1))

def common_divisors (m n : ℕ) : List ℕ :=
  List.filter (λ d, d ∈ positive_divisors m) (positive_divisors n)

theorem sum_of_common_divisors_60_18 : 
  List.sum (common_divisors 60 18) = 12 := by
  sorry

end sum_of_common_divisors_60_18_l802_802535


namespace sum_of_common_divisors_60_18_l802_802532

def positive_divisors (n : ℕ) : List ℕ :=
  List.filter (λ d, n % d = 0) (List.range (n+1))

def common_divisors (m n : ℕ) : List ℕ :=
  List.filter (λ d, d ∈ positive_divisors m) (positive_divisors n)

theorem sum_of_common_divisors_60_18 : 
  List.sum (common_divisors 60 18) = 12 := by
  sorry

end sum_of_common_divisors_60_18_l802_802532


namespace range_of_b_l802_802990

theorem range_of_b
  (b : ℝ)
  (h : ∀ x : ℝ, x ∈ set.Ico (-1 : ℝ) (1/2) → sqrt (1 - x^2) > x + b) : 
  b < 0 := 
sorry

end range_of_b_l802_802990


namespace inequality_subtraction_l802_802155

theorem inequality_subtraction (a b c : ℝ) (h : a > b) : a - c > b - c := 
begin
  sorry
end

end inequality_subtraction_l802_802155


namespace max_single_player_salary_l802_802650

-- Definitions based on the conditions
def num_players : ℕ := 18
def min_salary : ℕ := 20000
def max_total_salary : ℕ := 800000

-- The theorem to prove the highest possible salary for a single player
theorem max_single_player_salary :
  ∃ (max_salary : ℕ), max_salary = 460000 ∧ 
    (∀ (salaries : Fin num_players → ℕ), 
    (∀ i, min_salary ≤ salaries i) ∧ 
    (∑ i, salaries i) ≤ max_total_salary → 
    ∃ i, salaries i ≤ max_salary) :=
by
  sorry

end max_single_player_salary_l802_802650


namespace bicycle_stock_decrease_l802_802721

-- Define the conditions and the problem
theorem bicycle_stock_decrease (m : ℕ) (jan_to_oct_decrease june_to_oct_decrease monthly_decrease : ℕ) 
  (h1: monthly_decrease = 4)
  (h2: jan_to_oct_decrease = 36)
  (h3: june_to_oct_decrease = 4 * monthly_decrease):
  m * monthly_decrease = jan_to_oct_decrease - june_to_oct_decrease → m = 5 := 
by
  sorry

end bicycle_stock_decrease_l802_802721


namespace card_pairs_with_conditions_l802_802453

theorem card_pairs_with_conditions : 
  let cards := Finset.range 51 in
  let pairs := (cards.product cards).filter (λ p, (p.1 - p.2).natAbs = 11 ∧ (p.1 * p.2) % 5 = 0) in
  pairs.card / 2 = 15 :=
by
  sorry

end card_pairs_with_conditions_l802_802453


namespace taxi_ride_cost_is_correct_l802_802850

noncomputable def taxi_ride_total_cost 
  (distance_in_miles : ℕ) (waiting_time_min : ℕ) (traffic_delay_min : ℕ) 
  (toll_route : Bool) (senior_citizen : Bool) (peak_hours : Bool) 
  : ℝ :=
  let initial_cost := 2.5
  let additional_mile_cost := if distance_in_miles > 1 then (39 / 5) * 0.4 else 0
  let waiting_time_cost := waiting_time_min * 0.25
  let traffic_delay_cost := traffic_delay_min * 0.15
  let toll_cost := if toll_route then 3.0 else 0
  let total_cost_pre_discount := initial_cost + additional_mile_cost + waiting_time_cost + traffic_delay_cost + toll_cost
  let discount := if senior_citizen || ¬peak_hours then total_cost_pre_discount * 0.10 else 0
  let total_cost_post_discount := total_cost_pre_discount - discount
  let tip := total_cost_post_discount * 0.15
  total_cost_post_discount + tip

theorem taxi_ride_cost_is_correct : 
  taxi_ride_total_cost 8 12 25 true true false = 20.76 :=
by
  sorry

end taxi_ride_cost_is_correct_l802_802850


namespace fence_pole_count_l802_802371

-- Define the conditions
def path_length : ℕ := 900
def bridge_length : ℕ := 42
def pole_spacing : ℕ := 6

-- Define the goal
def total_poles : ℕ := 286

-- The statement to prove
theorem fence_pole_count :
  let total_length_to_fence := (path_length - bridge_length)
  let poles_per_side := total_length_to_fence / pole_spacing
  let total_poles_needed := 2 * poles_per_side
  total_poles_needed = total_poles :=
by
  sorry

end fence_pole_count_l802_802371


namespace abundant_product_l802_802963

def is_abundant (n : ℕ) : Prop := Nat.sigma n > 2 * n

theorem abundant_product (a b : ℕ) (ha : is_abundant a) : is_abundant (a * b) := 
by
  sorry

end abundant_product_l802_802963


namespace david_reading_time_l802_802430

def total_time : ℕ := 180
def math_homework : ℕ := 25
def spelling_homework : ℕ := 30
def history_assignment : ℕ := 20
def science_project : ℕ := 15
def piano_practice : ℕ := 30
def study_breaks : ℕ := 2 * 10

def time_other_activities : ℕ := math_homework + spelling_homework + history_assignment + science_project + piano_practice + study_breaks

theorem david_reading_time : total_time - time_other_activities = 40 :=
by
  -- Calculation steps would go here, not provided for the theorem statement.
  sorry

end david_reading_time_l802_802430


namespace net_rate_of_take_home_pay_l802_802869

theorem net_rate_of_take_home_pay
  (hours : ℝ) (speed : ℝ) (miles_per_gallon : ℝ)
  (payment_per_mile : ℝ) (gas_cost_per_gallon : ℝ) (tax_rate : ℝ) :
  hours = 3 → speed = 70 → miles_per_gallon = 35 →
  payment_per_mile = 0.60 → gas_cost_per_gallon = 2.50 →
  tax_rate = 0.10 →
  ((
    (payment_per_mile * speed * hours) - 
    (gas_cost_per_gallon * (speed * hours / miles_per_gallon)) - 
    (tax_rate * ((payment_per_mile * speed * hours) - (gas_cost_per_gallon * (speed * hours / miles_per_gallon))))) / hours 
  = 33.30) :=
begin
  sorry
end

end net_rate_of_take_home_pay_l802_802869


namespace angle_ADE_l802_802772

-- Definitions and conditions
variable (x : ℝ)

def angle_ABC := 60
def angle_CAD := x
def angle_BAD := x
def angle_BCA := 120 - 2 * x
def angle_DCE := 180 - (120 - 2 * x)

-- Theorem statement
theorem angle_ADE (x : ℝ) : angle_CAD x = x → angle_BAD x = x → angle_ABC = 60 → 
                            angle_DCE x = 180 - angle_BCA x → 
                            120 - 3 * x = 120 - 3 * x := 
by
  intro h1 h2 h3 h4
  sorry

end angle_ADE_l802_802772


namespace market_supply_function_tax_revenues_collected_optimal_tax_rate_tax_revenues_specified_l802_802384

noncomputable def Q_d : ℝ → ℝ := λ P, 688 - 4 * P
noncomputable def P_s := 64
noncomputable def tax := 90

theorem market_supply_function :
  ∃ (c d : ℝ), (λ P, c + d * P) = (λ P, -312 + 6 * P) :=
by {
  let Q_s : ℝ → ℝ := λ P, -312 + 6 * P,
  use [-312, 6],
  sorry
}

theorem tax_revenues_collected :
  let Q_s := -312 + 6 * P_s in Q_s * 90 = 6480 :=
by {
  let Q_s := -312 + 6 * P_s,
  have : Q_s = 72, by { unfold Q_s, simp },
  show 72 * tax = 6480, by norm_num,
  sorry
}

theorem optimal_tax_rate :
  let t_opt := 54 in t_opt = 54 :=
by {
  let t_opt := 54,
  show t_opt = 54, by norm_num,
  sorry
}

theorem tax_revenues_specified :
  let Q_s := λ t, 432 - 4 * t in
  let t_opt := 54 in
  Q_s t_opt * t_opt = 10800 :=
by {
  let t_opt := 54,
  let Q := 432 - 4 * t_opt,
  have : Q = 216, by { unfold Q, simp },
  show 216 * t_opt = 10800, by norm_num,
  sorry
}

end market_supply_function_tax_revenues_collected_optimal_tax_rate_tax_revenues_specified_l802_802384


namespace arithmetic_expression_8000_l802_802032

theorem arithmetic_expression_8000
  : ∃ n : ℕ, 
    (∃ a b c : ℕ, 
      8 * a + 88 * b + 888 * c = 8000 ∧ n = a + 2 * b + 3 * c) ∧
    (Finset.card 
      (Finset.image 
        (λ a b c, a + 2 * b + 3 * c) 
        { x | ∃ a b c : ℕ, 
            a + 11 * b + 111 * c = 1000 }
      ) = 109) := 
sorry

end arithmetic_expression_8000_l802_802032


namespace find_simple_interest_rate_l802_802290

def compound_interest (P R : ℝ) (n : ℕ) : ℝ := 
  P * ((1 + R / 100) ^ n) - P

def simple_interest (P R : ℝ) (t : ℕ) : ℝ := 
  (P * R * t) / 100

theorem find_simple_interest_rate (P_s R_c P_c : ℝ) (t_s t_c : ℕ) 
    (h1 : P_s = 2800)
    (h2 : P_c = 4000)
    (h3 : R_c = 10)
    (h4 : t_s = 3)
    (h5 : t_c = 2)
    (h6 : simple_interest P_s ?R_s t_s = 
      (compound_interest P_c R_c t_c) / 2) : 
  ?R_s = 5 :=
sorry

end find_simple_interest_rate_l802_802290


namespace sum_common_divisors_sixty_and_eighteen_l802_802521

theorem sum_common_divisors_sixty_and_eighteen : 
  ∑ d in ({d ∈ ({1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60} : finset ℕ) | d ∈ ({1, 2, 3, 6, 9, 18} : finset ℕ)} : finset ℕ), d = 12 :=
by sorry

end sum_common_divisors_sixty_and_eighteen_l802_802521


namespace percent_calculation_l802_802331

theorem percent_calculation (y : ℝ) : (0.3 * (0.6 * y) = 0.18 * y) → (0.18 * y / y) * 100 = 18 := by
  sorry

end percent_calculation_l802_802331


namespace angle_bisector_theorem_l802_802412

theorem angle_bisector_theorem
  {A B C D : Type*} [ordered_field D]
  (ABC : triangle A B C)
  (D_point : is_angle_bisector A (line_segment B C) D) :
  ratio (length (line_segment A B)) (length (line_segment A C)) = 
  ratio (length (line_segment B D)) (length (line_segment C D)) :=
sorry

end angle_bisector_theorem_l802_802412


namespace proof_relationship_l802_802905

variable {f : ℝ → ℝ}

/-- Conditions setup --/
axiom odd_fn (h_odd : ∀ x, f x = -f (-x))
axiom periodic_fn (h_periodic : ∀ x, f (2 - x) = f x)
axiom decreasing_fn (h_decreasing : ∀ x, -3 ≤ x ∧ x ≤ -2 → f x > f (x + 1))
axiom acute_angle (α β : ℝ) (h_obtuse : 0 < α ∧ α + β < π ∧ β < π / 2)

noncomputable def relationship (α β : ℝ) (h_αβ : 
  0 < α ∧ α < π / 2 ∧ 0 < β ∧ β < π / 2 ∧ α + β < π / 2) : Prop :=
  f (Real.sin α) < f (Real.cos β)

theorem proof_relationship (α β : ℝ) (h_odd : ∀ x, f x = -f (-x)) 
  (h_periodic : ∀ x, f (2 - x) = f x) (h_decreasing : ∀ x, -3 ≤ x ∧ x ≤ -2 → f x > f (x + 1))
  (h_αβ : 0 < α ∧ α < π / 2 ∧ 0 < β ∧ β < π / 2 ∧ α + β < π / 2) :
   relationship α β h_αβ :=
sorry

end proof_relationship_l802_802905


namespace problem_I_l802_802843

theorem problem_I (x : ℝ) : -2 < abs(x - 1) - abs(x + 2) ∧ abs(x - 1) - abs(x + 2) < 0 ↔ -1 / 2 < x ∧ x < 1 / 2 :=
sorry

end problem_I_l802_802843


namespace stripes_on_flag_l802_802724

theorem stripes_on_flag (S : ℕ) (h₁ : ∀ (n : ℕ), (1 + (n - 1) / 2) * 10 = 70 → S = n) : S = 13 :=
by {
  let n := 13,
  have h : (1 + (n - 1) / 2) * 10 = 70,
  {
    norm_num,
  },
  exact h₁ n h,
}

end stripes_on_flag_l802_802724


namespace tax_revenue_at_90_optimal_tax_rate_l802_802382

noncomputable def market_supply_function (P : ℝ) : ℝ := 6 * P - 312

theorem tax_revenue_at_90 :
  let Q_s := market_supply_function 64 in
  Q_s * 90 = 6480 :=
by
  let Q_s := market_supply_function 64
  have h1 : Q_s = 72 := by
    simp [market_supply_function]
  have h2 : Q_s * 90 = 6480 := by
    rw [h1]
    norm_num
  exact h2

theorem optimal_tax_rate :
  let Q := 432 - 4 * 54 in
  Q * 54 = 10800 :=
by
  let Q := 432 - 4 * 54
  have h1 : Q = 216 := by
    norm_num
  have h2 : Q * 54 = 10800 := by
    rw [h1]
    norm_num
  exact h2

end tax_revenue_at_90_optimal_tax_rate_l802_802382


namespace PC_eq_PA_plus_PB_l802_802248

variables {A B C P : Type} [HasNorm A] [HasNorm B] [HasNorm C] [HasNorm P]
variables (PA PB PC AC BC : ℝ)

def arc_on_circumcircle (P A B C : P) := -- Definition that P lies on the arc AB of the circumcircle of triangle ABC
  sorry

def equilateral_triangle (A B C : P) := -- Definition of an equilateral triangle ABC
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B

def angle_60_deg (A P C : P) := -- Definition that angle APC = 60 degrees
  sorry

theorem PC_eq_PA_plus_PB (A B C P : P) (PA PB PC AC BC : ℝ)
  (h1 : PA = dist A P) (h2 : PB = dist B P) (h3 : PC = dist C P)
  (hp1 : arc_on_circumcircle P A B C)
  (hp2 : angle_60_deg A P C)
  (hp3 : angle_60_deg B P C)
  (h4 : equilateral_triangle A B C) :
  PC = PA + PB :=
by sorry

end PC_eq_PA_plus_PB_l802_802248


namespace sachin_rahul_age_ratio_l802_802256

theorem sachin_rahul_age_ratio :
  ∀ (Sachin_age Rahul_age: ℕ),
    Sachin_age = 49 →
    Rahul_age = Sachin_age + 14 →
    Nat.gcd Sachin_age Rahul_age = 7 →
    (Sachin_age / Nat.gcd Sachin_age Rahul_age) = 7 ∧ (Rahul_age / Nat.gcd Sachin_age Rahul_age) = 9 :=
by
  intros Sachin_age Rahul_age h1 h2 h3
  rw [h1, h2]
  sorry

end sachin_rahul_age_ratio_l802_802256


namespace problem104_proof_l802_802765

theorem problem104_proof
  (h1 : ∑ k in Finset.range 10, k = 55)
  (h2 : ∑ k in Finset.range 11, k^2 = 385)
  (h3 : ∑ k in Finset.range 11, k^3 = 3025) :
  (∑ k in Finset.range 11, k) * (∑ k in Finset.range 11, k^3) / (∑ k in Finset.range 11, k^2)^2 = (55 / 49) 
  :=
  sorry

end problem104_proof_l802_802765


namespace vector_eq_l802_802945

section VectorMath

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

def vec_a : α := (3 : ℝ, 2 : ℝ)
def vec_b : α := (0 : ℝ, -1 : ℝ)

theorem vector_eq : -2 • vec_a + 4 • vec_b = (-6, -8 : ℝ × ℝ) :=
by
  sorry

end VectorMath

end vector_eq_l802_802945


namespace quotient_unchanged_l802_802631

-- Define the variables
variables (a b k : ℝ)

-- Condition: k ≠ 0
theorem quotient_unchanged (h : k ≠ 0) : (a * k) / (b * k) = a / b := by
  sorry

end quotient_unchanged_l802_802631


namespace lines_intersect_or_parallel_l802_802387

theorem lines_intersect_or_parallel
  (A B C D K L M N : Point)
  (hCircleTangentA : tangent A K)
  (hCircleTangentB : tangent B L)
  (hCircleTangentC : tangent C M)
  (hCircleTangentD : tangent D N)
  (hQuadrilateralInscribed : inscribed_in_circle A B C D)
  (hTangencyPoints : tangency_points A B C D K L M N) :
  (intersect_at_point_or_parallel KL MN AC) :=
sorry

end lines_intersect_or_parallel_l802_802387


namespace interest_rate_l802_802166

noncomputable def simple_interest (P r t: ℝ) : ℝ := P * r * t / 100

noncomputable def compound_interest (P r t: ℝ) : ℝ := P * (1 + r / 100) ^ t - P

theorem interest_rate (P r: ℝ) (h1: simple_interest P r 2 = 50) (h2: compound_interest P r 2 = 51.25) : r = 5 :=
by
  sorry

end interest_rate_l802_802166


namespace find_a2_a5_l802_802973

variables {a : ℕ → ℝ}
variables r : ℝ

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * r

constant positive_sequence : ∀ n : ℕ, a n > 0

constant init_condition : a 0 * a 2 + 2 * a 1 * a 4 + a 3 * a 5 = 36

-- Question to prove: a 1 + a 4 = 6
theorem find_a2_a5
  (h_geom: is_geometric_sequence a r)
  (h_pos: positive_sequence)
  (h_init: init_condition) :
  a 1 + a 4 = 6 :=
by
  sorry

end find_a2_a5_l802_802973


namespace percentage_increase_l802_802776

noncomputable def original_price : ℝ := 200
noncomputable def final_price : ℝ := 187.5
noncomputable def discount := 0.25

theorem percentage_increase :
  ∃ (P : ℝ), (P = 25) ∧ (final_price = (original_price + (original_price * (P / 100))) * (1 - discount)) :=
by
  use 25
  sorry

end percentage_increase_l802_802776


namespace card_pairs_with_conditions_l802_802448

theorem card_pairs_with_conditions : 
  let cards := Finset.range 51 in
  let pairs := (cards.product cards).filter (λ p, (p.1 - p.2).natAbs = 11 ∧ (p.1 * p.2) % 5 = 0) in
  pairs.card / 2 = 15 :=
by
  sorry

end card_pairs_with_conditions_l802_802448


namespace empty_boxes_count_l802_802079

theorem empty_boxes_count (n : Nat) (non_empty_boxes : Nat) (empty_boxes : Nat) : 
  (n = 34) ∧ (non_empty_boxes = n) ∧ (empty_boxes = -1 + 6 * n) → empty_boxes = 203 := by 
  intros
  sorry

end empty_boxes_count_l802_802079


namespace max_odd_integers_l802_802406

theorem max_odd_integers (chosen : Fin 5 → ℕ) (hpos : ∀ i, chosen i > 0) (heven : ∃ i, chosen i % 2 = 0) : 
  ∃ odd_count, odd_count = 4 ∧ (∀ i, i < 4 → chosen i % 2 = 1) := 
by 
  sorry

end max_odd_integers_l802_802406


namespace Fran_same_distance_speed_l802_802681

noncomputable def Joann_rides (v_j t_j : ℕ) : ℕ := v_j * t_j

def Fran_speed (d t_f : ℕ) : ℕ := d / t_f

theorem Fran_same_distance_speed
  (v_j t_j t_f : ℕ) (hj: v_j = 15) (tj: t_j = 4) (tf: t_f = 5) : Fran_speed (Joann_rides v_j t_j) t_f = 12 := by
  have hj_dist: Joann_rides v_j t_j = 60 := by
    rw [hj, tj]
    sorry -- proof of Joann's distance
  have d_j: ℕ := 60
  have hf: Fran_speed d_j t_f = Fran_speed 60 5 := by
    rw ←hj_dist
    sorry -- proof to equate d_j with Joann's distance
  show Fran_speed 60 5 = 12
  sorry -- Final computation proof

end Fran_same_distance_speed_l802_802681


namespace circle_equation_l802_802064

theorem circle_equation (x y : ℝ) :
  (x = 2 ∧ y = -2) ∨ 
  (∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ (x, y) ∈ set_of (λ p, ∃ z t : ℝ, (z^2 + t^2 - 6*z = 0) ∧ (z^2 + t^2 = 4) ∧ (p = (z, t)))) →
  (x^2 + y^2 - 3*x - 2 = 0) :=
sorry

end circle_equation_l802_802064


namespace determine_set_l802_802489

-- Define the complex number z as x + y * I
def is_real (z : ℂ) : Prop := z.im = 0

theorem determine_set :
  ∀ (z : ℂ), is_real ((5 + 7 * complex.I) * z) ↔ (∃ (x y : ℝ), z = x + y * complex.I ∧ x = - (5 : ℝ) / 7 * y) :=
by
  intro z
  sorry

end determine_set_l802_802489


namespace find_a_value_l802_802636

variable {α : Type*} [LinearOrderedField α]

def isOddFunction (f : α → α) := ∀ x : α, f (-x) = -f x

theorem find_a_value (a : α) (f : α → α) 
  (h₁ : f = λ x, x / ((2 * x - 1) * (x + a))) 
  (h₂ : isOddFunction f) 
  (h₃ : ∀ x, x ≠ 1 / 2 ∧ x ≠ -a) : 
  a = 1 / 2 := 
sorry

end find_a_value_l802_802636


namespace max_odd_integers_l802_802404

theorem max_odd_integers (a b c d e : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) (h_even : a * b * c * d * e % 2 = 0) :
  ∃ m : ℕ, m = 4 ∧ ∀ o1 o2 o3 o4 : ℕ, (o1 % 2 = 1 ∧ o2 % 2 = 1 ∧ o3 % 2 = 1 ∧ o4 % 2 = 1) ∧
    (list.perm [a, b, c, d, e] [o1, o2, o3, o4, (2 * (a * b * c * d * e / 2 / o1 / o2 / o3 / o4))]) :=
sorry

end max_odd_integers_l802_802404


namespace rogers_coaches_l802_802745

-- Define the structure for the problem conditions
structure snacks_problem :=
  (team_members : ℕ)
  (helpers : ℕ)
  (packs_purchased : ℕ)
  (pouches_per_pack : ℕ)

-- Create an instance of the problem with given conditions
def rogers_problem : snacks_problem :=
  { team_members := 13,
    helpers := 2,
    packs_purchased := 3,
    pouches_per_pack := 6 }

-- Define the theorem to state that given the conditions, the number of coaches is 3
theorem rogers_coaches (p : snacks_problem) : p.packs_purchased * p.pouches_per_pack - p.team_members - p.helpers = 3 :=
by
  sorry

end rogers_coaches_l802_802745


namespace units_digit_of_factorial_sum_l802_802550

theorem units_digit_of_factorial_sum : (∑ n in Finset.range 101, n.factorial) % 10 = 3 :=
by
  sorry

end units_digit_of_factorial_sum_l802_802550


namespace lcm_12_18_l802_802925

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l802_802925


namespace fractional_rep_of_0_point_345_l802_802497

theorem fractional_rep_of_0_point_345 : 
  let x := (0.3 + (0.45 : ℝ)) in
  (x = (83 / 110 : ℝ)) :=
by
  sorry

end fractional_rep_of_0_point_345_l802_802497


namespace exists_subset_l802_802494

-- Define a condition on X such that a + 2b = n has exactly one solution in X for any n in ℤ
theorem exists_subset (X : Set Int) : 
  (∀ n : Int, ∃! (ab : Int × Int), (ab.1 ∈ X) ∧ (ab.2 ∈ X) ∧ ab.1 + 2 * ab.2 = n) → 
  ∃ (X : Set Int), ∀ n : Int, ∃! (ab : Int × Int), (ab.1 ∈ X) ∧ (ab.2 ∈ X) ∧ ab.1 + 2 * ab.2 = n :=
begin
  sorry
end

end exists_subset_l802_802494


namespace derangement_probability_l802_802755

theorem derangement_probability (n : ℕ) : 
  (∑ k in Finset.range (n + 1), (-1 : ℤ) ^ k / Nat.factorial k : ℚ) = 
  ∑ k in Finset.range (n + 1), (↑((-1) ^ k) / (↑(k.factorial) : ℚ)) := 
by
  sorry

end derangement_probability_l802_802755


namespace max_a_l802_802602

theorem max_a (a : ℝ) : 
  (∀ x ∈ set.Ioo (0 : ℝ) 2, x^2 + a * x + 4 ≤ 6) → a ≤ -1 :=
begin
  sorry
end

end max_a_l802_802602


namespace correct_propositions_l802_802975

section Propositions

variables (m n : Type) [Line m] [Line n] (α β γ : Type) [Plane α] [Plane β] [Plane γ]

theorem correct_propositions :
  (∀ (h1 : m ∥ α) (h2 : n ∥ α), m ∥ n) ↔ False →
  (∀ (h1 : α ⊥ γ) (h2 : β ⊥ γ), α ∥ β) ↔ False →
  (∀ (h1 : α ∥ β) (h2 : β ∥ γ), α ∥ γ) ↔ True →
  (∀ (h1 : m ⊥ α) (h2 : n ⊥ α), m ∥ n) ↔ True :=
by sorry

end Propositions

end correct_propositions_l802_802975


namespace f_values_f_ge_4_range_l802_802603

def f (x : ℝ) : ℝ :=
if x ≤ 1 then x + 5
else -2 * x + 8

theorem f_values :
  f(2) = 4 ∧ f(f(-1)) = 0 :=
by
  split
  sorry -- proof for f(2) = 4
  sorry -- proof for f(f(-1)) = 0

theorem f_ge_4_range (x : ℝ) :
  f(x) ≥ 4 -> -1 ≤ x ∧ x ≤ 2 :=
by
  intro h
  sorry -- proof showing -1 ≤ x ≤ 2 when f(x) ≥ 4

end f_values_f_ge_4_range_l802_802603


namespace expansion_properties_l802_802036

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0       => 1
| 0, k + 1   => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

theorem expansion_properties (n : ℕ)
    (T : ℕ → ℕ → ℝ → ℝ → ℝ) 
    (H1 : ∀ k : ℕ, k ≤ n → T (k + 1) n x = (-1)^k * (binom n k : ℝ) * x^((2 * n - 3 * k) / 4)) 
    (H2 : n = 8) :
    -- No constant term
    (∀ k : ℕ, T (k + 1) n x = c → ¬(3 * k = 16))
    ∧ -- Rational terms
    ((2 * 8 - 3 * 0) / 4 ∈ ℤ ∧ T 1 8 x = x^4)
    ∧ ((2 * 8 - 3 * 4) / 4 ∈ ℤ ∧ T 5 8 x = (35 / 8) * x)
    ∧ ((2 * 8 - 3 * 8) / 4 ∈ ℤ ∧ T 9 8 x = (1 / 256) * x^(-2)) :=
by
  sorry

end expansion_properties_l802_802036


namespace simplify_expression_l802_802419

variable (x y : ℝ)

theorem simplify_expression : 3 * x^2 * y * (2 / (9 * x^3 * y)) = 2 / (3 * x) :=
by sorry

end simplify_expression_l802_802419


namespace infinite_series_sum_l802_802072

def closestIntSqrt (n : ℕ) : ℕ :=
  round (Real.sqrt n)

theorem infinite_series_sum : 
  (∑' n : ℕ, (3^(closestIntSqrt n) + 3^(-(closestIntSqrt n))) / 3^n) = 3 :=
by
  sorry

end infinite_series_sum_l802_802072


namespace triangle_angles_l802_802725

theorem triangle_angles
  (A B C M : Type)
  (triangle_ABC: A B C)
  (triangle_ABM: A B M)
  (triangle_AMC: A M C)
  (h1: AB = BM)
  (h2: AM = MC)
  (h3: ∠ B = 5 * ∠ C) :
  ∠ A = 60 ∧ ∠ B = 100 ∧ ∠ C = 20 := 
sorry

end triangle_angles_l802_802725


namespace angle_range_l802_802641

noncomputable def angle_between (a b : EuclideanSpace ℝ (Fin 3)) : ℝ :=
Real.arccos ((EuclideanInnerProductSpace.inner a b) / ((EuclideanNorm.norm a) * (EuclideanNorm.norm b)))

theorem angle_range (a b : EuclideanSpace ℝ (Fin 3)) (h : EuclideanInnerProductSpace.inner a b < 0) :
  (Real.pi / 2) < angle_between a b ∧ angle_between a b ≤ Real.pi :=
by
  sorry

end angle_range_l802_802641


namespace construct_cyclic_quadrilateral_l802_802092

theorem construct_cyclic_quadrilateral 
  (AC BD : ℝ) (M : Type*)
  (ω β : ℝ)
  (hAC_BD : AC ≥ BD)
  (h_angle_ω : 0 ≤ ω ∧ ω ≤ π/2)
  (h_cyclic : ∃ (A B C D : M), 
    cyclic_quadrilateral A B C D ∧ 
    diagonals_intersect_at M A C B D ∧ 
    angle_between_diagonals_eq ω M A C B D ∧ 
    angle_ABC_eq β A B C) 
  : ∃ (A B C D : M), is_cyclic_quadrilateral A B C D ∧ 
                  (diagonal_length A C = AC) ∧ 
                  (diagonal_length B D = BD) ∧ 
                  (∠ A M B = ω) ∧ 
                  (∠ A B C = β) := 
sorry

end construct_cyclic_quadrilateral_l802_802092


namespace points_opposite_sides_l802_802587

theorem points_opposite_sides (x y : ℝ) (h : (3 * x + 2 * y - 8) * (-1) < 0) : 3 * x + 2 * y > 8 := 
by
  sorry

end points_opposite_sides_l802_802587


namespace z_in_second_quadrant_l802_802953

-- Define the given complex number x
def x : ℂ := 3 + 4 * Complex.i

-- Define the magnitude (absolute value) of x
def abs_x : ℝ := Complex.abs x

-- Define the complex number z as given in the problem
def z : ℂ := x - abs_x - (1 - Complex.i)

-- Define a predicate to check if a complex number is in the second quadrant
def in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

-- State the theorem to prove that z is in the second quadrant
theorem z_in_second_quadrant : in_second_quadrant z :=
by sorry

end z_in_second_quadrant_l802_802953


namespace fran_speed_l802_802679

theorem fran_speed (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) 
    (h_joann : joann_speed = 15) (h_joann_time : joann_time = 4) (h_fran_time : fran_time = 5) : 
    (joann_speed * joann_time) / fran_time = 12 :=
by
  rw [h_joann, h_joann_time, h_fran_time]
  norm_num
  sorry

end fran_speed_l802_802679


namespace problem1_correct_l802_802259

noncomputable def problem1_solution (y : ℝ → ℝ) (C1 C2 : ℝ) :=
  y = λ x, C1 * Real.exp (-5 * x) + C2 * Real.exp (-x) + 5 * x^2 - 12 * x + 12

theorem problem1_correct (y : ℝ → ℝ) (C1 C2 : ℝ) :
  problem1_solution y C1 C2 →
  (∀ x, y'' x + 6 * y' x + 5 * y x = 25 * x^2 - 2) :=
sorry

end problem1_correct_l802_802259


namespace min_value_of_x_plus_y_l802_802628

-- Define the conditions
variables (x y : ℝ)
variables (h1 : x > 0) (h2 : y > 0) (h3 : y + 9 * x = x * y)

-- The statement of the problem
theorem min_value_of_x_plus_y : x + y ≥ 16 :=
sorry

end min_value_of_x_plus_y_l802_802628


namespace smallest_positive_angle_l802_802982

-- Define the conditions
def circle_center := (0, 0)
def radius := 1
def intersection_point := (Real.sin (2 * Real.pi / 3), Real.cos (2 * Real.pi / 3))
  
-- The angle α
def angle_α := 11 * Real.pi / 6

-- Lean 4 statement to prove
theorem smallest_positive_angle :
  ∃ α, 0 < α ∧ α = angle_α ∧ (intersection_point = (Real.cos α, Real.sin α)) :=
by
  sorry

end smallest_positive_angle_l802_802982


namespace fraction_sum_equals_248_l802_802216

section
variables {p q r : ℝ}
variables {A B C : ℝ}

-- Conditions
def is_root (a b c d x : ℝ) := x^3 - a* x^2 + b * x - d = 0

axiom distinct_roots_and_partial_fraction (h₁ : p ≠ q) (h₂ : p ≠ r) (h₃ : q ≠ r) :
  is_root 23 85 72 p ∧ 
  is_root 23 85 72 q ∧ 
  is_root 23 85 72 r ∧ 
  (∀ s : ℝ, s ∉ {p, q, r} → 
    (1 / (s^3 - 23 * s^2 + 85 * s - 72)) = (A / (s - p)) + (B / (s - q)) + (C / (s - r))) ∧
  (A * (q - r) = 1 / ((r - p) * (q - p))) ∧ 
  (B * (p - r) = 1 / ((p - r) * (p - q))) ∧ 
  (C * (p - q) = 1 / ((p - q) * (q - r))) 

theorem fraction_sum_equals_248 
  (h₁ : p ≠ q) (h₂ : p ≠ r) (h₃ : q ≠ r) 
  (h₄ : is_root 23 85 72 p)
  (h₅ : is_root 23 85 72 q)
  (h₆ : is_root 23 85 72 r)
  (h₇ : ∀ s : ℝ, s ∉ {p, q, r} → 
    ((1 / (s^3 - 23 * s^2 + 85 * s - 72)) = (A / (s - p)) + (B / (s - q)) + (C / (s - r)))
  ):
  (1 / A + 1 / B + 1 / C = 248) := 
sorry
end

end fraction_sum_equals_248_l802_802216


namespace inverse_proportion_quadrants_l802_802637

theorem inverse_proportion_quadrants (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, x ≠ 0 → ((x > 0 → k / x > 0) ∧ (x < 0 → k / x < 0))) ↔ k > 0 := by
  sorry

end inverse_proportion_quadrants_l802_802637


namespace recurring_fraction_sum_eq_l802_802510

theorem recurring_fraction_sum_eq (x : ℝ) (h1 : x = 0.45̅) : 0.3 + x = 83/110 := by
  sorry

end recurring_fraction_sum_eq_l802_802510


namespace sum_common_divisors_60_18_l802_802530

def divisors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ m => n % m = 0)

noncomputable def sum (l : List ℕ) : ℕ :=
  l.foldr (· + ·) 0

theorem sum_common_divisors_60_18 :
  sum (List.filter (λ d => d ∈ divisors 18) (divisors 60)) = 12 :=
by
  sorry

end sum_common_divisors_60_18_l802_802530


namespace recurring_fraction_sum_eq_l802_802508

theorem recurring_fraction_sum_eq (x : ℝ) (h1 : x = 0.45̅) : 0.3 + x = 83/110 := by
  sorry

end recurring_fraction_sum_eq_l802_802508


namespace proof_log_S_2010_add_2_l802_802176

def geometric_sequence_properties (a_1 a_2 a_3 : ℝ) (S_n : ℕ → ℝ) : Prop :=
  (a_1 + a_2 = 6) ∧ (a_2 + a_3 = 12) ∧
  (∀ n : ℕ, S_n n = 2 * (2^n - 1))

theorem proof_log_S_2010_add_2 :
  ∀ (a_1 a_2 a_3 : ℝ) (S_n : ℕ → ℝ),
  geometric_sequence_properties a_1 a_2 a_3 S_n →
  ∃ q a : ℝ,
    (q = 2) ∧ (a = 2) ∧ 
    (∀ n, ∀ (S_2010:ℝ), S_2010 = 2 * (2^2010 - 1) → log 2 (S_2010 + 2) = 2011) :=
by sorry

end proof_log_S_2010_add_2_l802_802176


namespace find_p_range_l802_802966

noncomputable def a (n : ℕ) : ℤ :=
  if n = 0 then 0 else (-1) ^ (n-1) * (2 * n - 1)

theorem find_p_range (p : ℝ) :
  (∀ n : ℕ+, (a n.succ - p) * (a n - p) < 0) ↔ p ∈ set.Ioo (-3 : ℝ) 1 :=
by 
  sorry

end find_p_range_l802_802966


namespace integral_sign_l802_802490

noncomputable def I : ℝ := ∫ x in -Real.pi..0, Real.sin x

theorem integral_sign : I < 0 := sorry

end integral_sign_l802_802490


namespace derivative_given_limit_l802_802205

open Real

variable {f : ℝ → ℝ} {x₀ : ℝ}

theorem derivative_given_limit (h : ∀ ε > 0, ∃ δ > 0, ∀ Δx, abs Δx < δ → abs ((f (x₀ - 2 * Δx) - f x₀) / Δx + 2) < ε) :
  deriv f x₀ = -1 := by
  sorry

end derivative_given_limit_l802_802205


namespace card_pairs_satisfying_conditions_l802_802433

theorem card_pairs_satisfying_conditions :
  ∃ (s : Finset (ℕ × ℕ)),
    (∀ p ∈ s, p.1 ≠ p.2 ∧ (p.1 = p.2 + 11 ∨ p.2 = p.1 + 11) ∧ ((p.1 * p.2) % 5 = 0))
    ∧ s.card = 15 :=
begin
  sorry
end

end card_pairs_satisfying_conditions_l802_802433


namespace inv_100_mod_101_l802_802055

theorem inv_100_mod_101 : (100 : ℤ) * (100 : ℤ) % 101 = 1 := by
  sorry

end inv_100_mod_101_l802_802055


namespace ratio_of_geometric_sequence_sum_l802_802567

theorem ratio_of_geometric_sequence_sum (a : ℕ → ℕ) 
    (q : ℕ) (h_q_pos : 0 < q) (h_q_ne_one : q ≠ 1)
    (h_geo_seq : ∀ n : ℕ, a (n + 1) = a n * q)
    (h_arith_seq : 2 * a (3 + 2) = a 3 - a (3 + 1)) :
  (a 4 * (1 - q ^ 4) / (1 - q)) / (a 4 * (1 - q ^ 2) / (1 - q)) = 5 / 4 := 
  sorry

end ratio_of_geometric_sequence_sum_l802_802567


namespace find_q_l802_802288

def Q (x p q d : ℝ) : ℝ := x^3 + p * x^2 + q * x + d

theorem find_q (p q d : ℝ) (h: d = 3) (h1: -p / 3 = -d) (h2: -p / 3 = 1 + p + q + d) : q = -16 :=
by
  sorry

end find_q_l802_802288


namespace greater_num_792_l802_802832

theorem greater_num_792 (x y : ℕ) (h1 : x + y = 1443) (h2 : x - y = 141) : x = 792 :=
by
  sorry

end greater_num_792_l802_802832


namespace rain_at_house_l802_802829

/-- Define the amounts of rain on the three days Greg was camping. -/
def rain_day1 : ℕ := 3
def rain_day2 : ℕ := 6
def rain_day3 : ℕ := 5

/-- Define the total rain experienced by Greg while camping. -/
def total_rain_camping := rain_day1 + rain_day2 + rain_day3

/-- Define the difference in the rain experienced by Greg while camping and at his house. -/
def rain_difference : ℕ := 12

/-- Define the total amount of rain at Greg's house. -/
def total_rain_house := total_rain_camping + rain_difference

/-- Prove that the total rain at Greg's house is 26 mm. -/
theorem rain_at_house : total_rain_house = 26 := by
  /- We know that total_rain_camping = 14 mm and rain_difference = 12 mm -/
  /- Therefore, total_rain_house = 14 mm + 12 mm = 26 mm -/
  sorry

end rain_at_house_l802_802829


namespace find_rate_l802_802893

-- Definitions of conditions
def Principal : ℝ := 2500
def Amount : ℝ := 3875
def Time : ℝ := 12

-- Main statement we are proving
theorem find_rate (P : ℝ) (A : ℝ) (T : ℝ) (R : ℝ) 
    (hP : P = Principal) 
    (hA : A = Amount) 
    (hT : T = Time) 
    (hR : R = (A - P) * 100 / (P * T)) : R = 55 / 12 := 
by 
  sorry

end find_rate_l802_802893


namespace sphere_surface_area_l802_802792

-- Defining the radius
def radius : ℝ := 3

-- The formula for the surface area of a sphere
def surface_area (r : ℝ) : ℝ := 4 * real.pi * r^2

-- Statement asserting the surface area given the radius
theorem sphere_surface_area : surface_area radius = 36 * real.pi :=
by
  sorry

end sphere_surface_area_l802_802792


namespace peter_walks_more_time_l802_802732

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

end peter_walks_more_time_l802_802732


namespace at_least_six_on_circle_l802_802133

-- Defining the types for point and circle
variable (Point : Type)
variable (Circle : Type)

-- Assuming the existence of a well-defined predicate that checks whether points lie on the same circle
variable (lies_on_circle : Circle → Point → Prop)
variable (exists_circle : Point → Point → Point → Point → Circle)
variable (five_points_condition : ∀ (p1 p2 p3 p4 p5 : Point), 
  ∃ (c : Circle), lies_on_circle c p1 ∧ lies_on_circle c p2 ∧ 
                   lies_on_circle c p3 ∧ lies_on_circle c p4)

-- Given 13 points on a plane
variables (P : List Point)
variable (length_P : P.length = 13)

-- The main theorem statement
theorem at_least_six_on_circle : 
  (∀ (P : List Point) (h : P.length = 13),
    (∀ p1 p2 p3 p4 p5 : Point, ∃ (c : Circle), lies_on_circle c p1 ∧ lies_on_circle c p2 ∧ 
                               lies_on_circle c p3 ∧ lies_on_circle c p4)) →
    (∃ (c : Circle), ∃ (l : List Point), l.length ≥ 6 ∧ ∀ p ∈ l, lies_on_circle c p) :=
sorry

end at_least_six_on_circle_l802_802133


namespace partition_triangle_unique_areas_l802_802699

theorem partition_triangle_unique_areas (A B C : Type) [triangle A B C] (h_angle_A : ∠ABC = 90) : 
  ∃ partitions : set (triangle A B C), 
    (∀ T ∈ partitions, similar T (triangle A B C)) ∧
    (|partitions| = 2006) ∧
    (∀ T₁ T₂ ∈ partitions, T₁ ≠ T₂ → area T₁ ≠ area T₂) :=
sorry

end partition_triangle_unique_areas_l802_802699


namespace max_visited_cells_proof_l802_802359

inductive Color
| black
| white

structure Board :=
  (size : ℕ)
  (cells : ℕ → ℕ → Color)

def checkerboard : Board :=
  { size := 5,
    cells := λ r c => if ((r + c) % 2 = 0) then Color.black else Color.white }

structure Position :=
  (row : ℕ)
  (col : ℕ)

inductive MoveKind
| diagonal
| jump

structure Move :=
  (from : Position)
  (to : Position)
  (kind : MoveKind)

def valid_move (board: Board) (trail: List Position) (move: Move) : Prop :=
  -- Define the valid move conditions, diagonal or jump over a trail
  sorry

def max_visited_cells (board: Board) (start: Position) (visited: List Position) : ℕ :=
  -- Define max cells visited using recursive exploration
  sorry

theorem max_visited_cells_proof :
  ∀ (start : Position), checkerboard.cells start.row start.col = Color.black →
  max_visited_cells checkerboard start [start] ≤ 12 :=
by
  -- Omitted proof
  sorry

end max_visited_cells_proof_l802_802359


namespace sum_of_numbers_given_average_l802_802164

variable (average : ℝ) (n : ℕ) (sum : ℝ)

theorem sum_of_numbers_given_average (h1 : average = 4.1) (h2 : n = 6) (h3 : average = sum / n) :
  sum = 24.6 :=
by
  sorry

end sum_of_numbers_given_average_l802_802164


namespace tan_alpha_value_l802_802578

open Real

theorem tan_alpha_value (α : ℝ) (h1 : sin (π - α) = log 27⁻¹ (1 / 9)) (h2 : α ∈ Ioo (-π / 2) 0) : 
  tan α = - (2 * sqrt 5) / 5 :=
by
  sorry   -- Placeholder for the actual proof.

end tan_alpha_value_l802_802578


namespace percentage_failing_both_l802_802654

-- Define the conditions as constants
def percentage_failing_hindi : ℝ := 0.25
def percentage_failing_english : ℝ := 0.48
def percentage_passing_both : ℝ := 0.54

-- Define the percentage of students who failed in at least one subject
def percentage_failing_at_least_one : ℝ := 1 - percentage_passing_both

-- The main theorem statement we want to prove
theorem percentage_failing_both :
  percentage_failing_at_least_one = percentage_failing_hindi + percentage_failing_english - 0.27 := by
sorry

end percentage_failing_both_l802_802654


namespace probability_top_card_jqk_hearts_l802_802181

theorem probability_top_card_jqk_hearts (h_deck : List.range 52) 
  (h_suits : List.range 4) 
  (h_jqk: ∀s, s ∈ h_suits → ∃ jqk ∈ List.range 3, s * 13 + jqk < 52)
  (deck_shuffled : Deck -> Bool) :
  probability deck_shuffled ∈ {card | card ∈ {11, 12, 13}} = 3 / 52 := 
begin
  sorry,
end

end probability_top_card_jqk_hearts_l802_802181


namespace candy_problem_l802_802398

theorem candy_problem (
  a : ℤ
) : (a % 10 = 6) →
    (a % 15 = 11) →
    (200 ≤ a ∧ a ≤ 250) →
    (a = 206 ∨ a = 236) :=
sorry

end candy_problem_l802_802398


namespace laila_utility_l802_802689

theorem laila_utility (u : ℝ) :
  (2 * u * (10 - 2 * u) = 2 * (4 - 2 * u) * (2 * u + 4)) → u = 4 := 
by 
  sorry

end laila_utility_l802_802689


namespace katie_added_new_songs_l802_802688

-- Definitions for the conditions
def initial_songs := 11
def deleted_songs := 7
def current_songs := 28

-- Definition of the expected answer
def new_songs_added := current_songs - (initial_songs - deleted_songs)

-- Statement of the problem in Lean
theorem katie_added_new_songs : new_songs_added = 24 :=
by
  sorry

end katie_added_new_songs_l802_802688


namespace inverse_100_mod_101_l802_802051

theorem inverse_100_mod_101 : (100 * 100) % 101 = 1 :=
by
  -- Proof can be provided here.
  sorry

end inverse_100_mod_101_l802_802051


namespace max_odd_integers_l802_802405

theorem max_odd_integers (a b c d e : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) (h_even : a * b * c * d * e % 2 = 0) :
  ∃ m : ℕ, m = 4 ∧ ∀ o1 o2 o3 o4 : ℕ, (o1 % 2 = 1 ∧ o2 % 2 = 1 ∧ o3 % 2 = 1 ∧ o4 % 2 = 1) ∧
    (list.perm [a, b, c, d, e] [o1, o2, o3, o4, (2 * (a * b * c * d * e / 2 / o1 / o2 / o3 / o4))]) :=
sorry

end max_odd_integers_l802_802405


namespace negation_P_l802_802283

-- Define the condition that x is a real number
variable (x : ℝ)

-- Define the proposition P
def P := ∀ (x : ℝ), x ≥ 2

-- Define the negation of P
def not_P := ∃ (x : ℝ), x < 2

-- Theorem stating the equivalence of the negation of P
theorem negation_P : ¬P ↔ not_P := by
  sorry

end negation_P_l802_802283


namespace maximum_square_distance_sum_l802_802222

noncomputable def vector_magnitude {n : Type*} [inner_product_space ℝ n] (v : n) : ℝ := ↑(∥v∥)

theorem maximum_square_distance_sum
  (a b c : EuclideanSpace ℝ (Fin 3))
  (ha : vector_magnitude a = 2)
  (hb : vector_magnitude b = 3)
  (hc : vector_magnitude c = 4) :
  (∥a - b∥^2 + ∥a - c∥^2 + ∥b - c∥^2) = 6 :=
sorry

end maximum_square_distance_sum_l802_802222


namespace problem1_problem2_l802_802947

variables {a b : ℝ}

-- Given conditions
def condition1 : a + b = 2 := sorry
def condition2 : a * b = -1 := sorry

-- Proof for a^2 + b^2 = 6
theorem problem1 (h1 : a + b = 2) (h2 : a * b = -1) : a^2 + b^2 = 6 :=
by sorry

-- Proof for (a - b)^2 = 8
theorem problem2 (h1 : a + b = 2) (h2 : a * b = -1) : (a - b)^2 = 8 :=
by sorry

end problem1_problem2_l802_802947


namespace find_f1_verify_function_l802_802124

theorem find_f1 (f : ℝ → ℝ) (h_mono : ∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f x1 > f x2)
    (h1_pos : ∀ x : ℝ, 0 < x → f x > 1 / x^2)
    (h_eq : ∀ x : ℝ, 0 < x → (f x)^2 * f (f x - 1 / x^2) = (f 1)^3) :
    f 1 = 2 := sorry

theorem verify_function (f : ℝ → ℝ) (h_def : ∀ x : ℝ, 0 < x → f x = 2 / x^2) :
    (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f x1 > f x2) ∧ (∀ x : ℝ, 0 < x → f x > 1 / x^2) ∧
    (∀ x : ℝ, 0 < x → (f x)^2 * f (f x - 1 / x^2) = (f 1)^3) := sorry

end find_f1_verify_function_l802_802124


namespace card_pairs_count_l802_802479

theorem card_pairs_count :
  let cards := {n : ℕ | 1 ≤ n ∧ n ≤ 50}
  let valid_pairs := {(a, b) ∈ cards × cards | 
    a ≠ b ∧ (a - b = 11 ∨ b - a = 11) ∧ (a * b) % 5 = 0}
  in valid_pairs.count = 15 :=
by
  sorry

end card_pairs_count_l802_802479


namespace wendy_first_album_pictures_l802_802311

theorem wendy_first_album_pictures (total_pictures : ℕ) (pictures_per_album : ℕ) (num_other_albums : ℕ) 
    (other_albums_pics : num_other_albums * pictures_per_album = 35) (total_pics_sum : total_pictures = 79) : 
    ∃ first_album_pics : ℕ, first_album_pics = 44 :=
by
  use total_pictures - (num_other_albums * pictures_per_album)
  rw [total_pics_sum, other_albums_pics]
  exact 79 - 35 = 44
  sorry

end wendy_first_album_pictures_l802_802311


namespace determine_m_l802_802583

theorem determine_m (m : ℚ) (x : ℝ) : (x^(m - 2) + 1 = 3) ∧ (∀ a b : ℝ, ∃ c : ℝ, a*x + b = c ↔ b = 0) → m = 3 :=
by
  sorry

end determine_m_l802_802583


namespace S10_value_l802_802294

def sequence_sum (n : ℕ) : ℕ :=
  (2^(n+1)) - 2 - n

theorem S10_value : sequence_sum 10 = 2036 := by
  sorry

end S10_value_l802_802294


namespace minimize_cost_l802_802367

noncomputable def k : ℝ := 1 / 2

def y (x : ℝ) : ℝ := 150 * (x + 1600 / x)

def min_transportation_cost (a : ℝ) : ℝ :=
if a >= 40 then 40 else a

theorem minimize_cost (a x : ℝ) 
  (h1 : 300 > 0) 
  (h2 : 30 > 0) 
  (h3 : 450 = k * 30^2) 
  (h4 : 800 >= 0) 
  (h5 : 0 < x ∧ x ≤ a) 
  : y x = 150 * (x + 1600 / x) ∧
    (if a >= 40 then x = 40 else x = a) :=
by
  sorry

end minimize_cost_l802_802367


namespace hyperbola_asymptotes_l802_802125

variable {x y b : ℝ}
variable (hyp : x^2 - y^2 / b^2 = 1)
variable (b_pos : b > 0)
variable (c : ℝ := 2)

theorem hyperbola_asymptotes:
  (sqrt 3) * x + y = 0 ∨ (sqrt 3) * x - y = 0 :=
sorry

end hyperbola_asymptotes_l802_802125


namespace Monet_paintings_consecutively_l802_802238

noncomputable def probability_Monet_paintings_consecutively (total_art_pieces Monet_paintings : ℕ) : ℚ :=
  let numerator := 9 * Nat.factorial (total_art_pieces - Monet_paintings) * Nat.factorial Monet_paintings
  let denominator := Nat.factorial total_art_pieces
  numerator / denominator

theorem Monet_paintings_consecutively :
  probability_Monet_paintings_consecutively 12 4 = 18 / 95 := by
  sorry

end Monet_paintings_consecutively_l802_802238


namespace Cindy_hits_9_l802_802552

variable (players : Type) -- A type to represent the players
variables (Alice Ben Cindy Dave Ellen : players)

-- The scores of each player
variable (score : players → ℕ)
variables (h_Alice : score Alice = 24)
variables (h_Ben : score Ben = 13)
variables (h_Cindy : score Cindy = 19)
variables (h_Dave : score Dave = 28)
variables (h_Ellen : score Ellen = 30)

-- Each player throws three darts
axiom throws_three_darts (p : players) : ∃ dart_scores : list ℕ, dart_scores.length = 3 ∧ dart_scores.sum = score p

-- Each throw hits the target regions (each between 1 and 15)
axiom scores_range (dart_scores : list ℕ) : ∀ e ∈ dart_scores, 1 ≤ e ∧ e ≤ 15

-- We need to prove Cindy hits the region worth 9 points:
theorem Cindy_hits_9 : ∃ dart_scores : list ℕ, dart_scores.sum = 19 ∧ 9 ∈ dart_scores :=
by {
  sorry
}

end Cindy_hits_9_l802_802552


namespace find_lengths_and_perimeter_l802_802961

-- Defining the conditions
variables {A B C D M E K : Type}
variables (parallelogram_ABCD : parallelogram A B C D)
variables (diameter_13 : ∀ {Ω : set Type} {a b M : Type}, Ω = circumcircle (triangle A B M) ∧ diameter Ω = 13)
variables (intersections : intersects_Ω_EK : ∀ {Ω : set Type}, {C B A D M E K : Type}, Ω = circumcircle (triangle A B M) → second_intersection Ω C B A E ∧ second_intersection Ω A D M K)
variables (arc_lengths : ∀ {Ω : set Type} {A E B M : Type}, Ω = circumcircle (triangle A B M) → arc_length Ω A E = 2 * arc_length Ω B M)
variables (MK_5 : length (segment M K) = 5)

-- Questions stated as hypotheses
theorem find_lengths_and_perimeter
  (AD_length : length (segment A D) = 13)
  (BK_length : length (segment B K) = 120 / 13)
  (perimeter_EBM : perimeter (triangle E B M) = 340 / 13) :
  true := sorry

end find_lengths_and_perimeter_l802_802961


namespace largest_last_digit_l802_802767

theorem largest_last_digit (s : String) (h_len : s.length = 2023) (h_first : s.get 0 = '2') (h_div : ∀ i, 0 ≤ i ∧ i < 2022 → (s.get i).toNat * 10 + (s.get (i+1)).toNat % 17 = 0 ∨ (s.get i).toNat * 10 + (s.get (i+1)).toNat % 29 = 0) :
  (s.get 2022).toNat = 7 :=
sorry

end largest_last_digit_l802_802767


namespace regular_17_gon_symmetry_l802_802391

/-- A regular 17-gon has L lines of symmetry, and the smallest positive angle for which it has rotational symmetry is R degrees. Prove that L + R = 649 / 17. -/
theorem regular_17_gon_symmetry :
  let L := 17 in
  let R := (360 : ℚ) / 17 in
  L + R = 649 / 17 :=
by
  sorry

end regular_17_gon_symmetry_l802_802391


namespace f_f_f_12_is_9_l802_802204

def divisors (n : ℕ) : List ℕ :=
  List.filter (λ d, d ∣ n) (List.range (n + 1))

def f (n : ℕ) : ℕ :=
  (divisors n).filter (≠ n).sum

theorem f_f_f_12_is_9 : f (f (f 12)) = 9 :=
by
  sorry

end f_f_f_12_is_9_l802_802204


namespace charlotte_test_l802_802422

def test_conditions (answers : Fin 100 → Bool) : Prop :=
  answers 0 = false ∧ answers 99 = false ∧ 
  (∀ i, i < 96 → (answers i, answers (i + 1), answers (i + 2), answers (i + 3), answers (i + 4)).to_sum /=
3)

theorem charlotte_test :
  ∃ answers : Fin 100 → Bool,
    test_conditions answers ∧
    ((finset.range 100).filter (λ i => answers i = true)).card = 60 ∧
    answers 5 = false ∧
    ∀ n, answers n = answers (n + 5) :=
by
  sorry

end charlotte_test_l802_802422


namespace candy_problem_l802_802399

theorem candy_problem (
  a : ℤ
) : (a % 10 = 6) →
    (a % 15 = 11) →
    (200 ≤ a ∧ a ≤ 250) →
    (a = 206 ∨ a = 236) :=
sorry

end candy_problem_l802_802399


namespace probability_no_shaded_in_2_by_2004_l802_802849

noncomputable def probability_no_shaded_rectangle (total_rectangles shaded_rectangles : Nat) : ℚ :=
  1 - (shaded_rectangles : ℚ) / (total_rectangles : ℚ)

theorem probability_no_shaded_in_2_by_2004 :
  let rows := 2
  let cols := 2004
  let total_rectangles := (cols + 1) * cols / 2 * rows
  let shaded_rectangles := 501 * 2507 
  probability_no_shaded_rectangle total_rectangles shaded_rectangles = 1501 / 4008 :=
by
  sorry

end probability_no_shaded_in_2_by_2004_l802_802849


namespace f_2021_value_l802_802957

variable (f : ℝ → ℝ)

-- Conditions
axiom periodicity : ∀ x, f(x + 6) + f(x) = 0
axiom symmetry_about : ∀ x, f(x - 1) + f(2 - x) = 0
axiom initial_value : f 1 = -2

-- Theorem to be proved
theorem f_2021_value : f 2021 = 2 :=
by
  sorry

end f_2021_value_l802_802957


namespace exponentiation_rule_l802_802026

theorem exponentiation_rule (m n : ℤ) : (-2 * m^3 * n^2)^2 = 4 * m^6 * n^4 :=
by
  sorry

end exponentiation_rule_l802_802026


namespace books_difference_l802_802727

def total_books := 80
def peter_percentage := 0.7
def brother_percentage := 0.35
def sarah_percentage := 0.4
def alex_percentage := 0.22

def peter_books := peter_percentage * total_books
def brother_books := brother_percentage * total_books
def sarah_books := sarah_percentage * total_books
def alex_books := Float.ceil (alex_percentage * total_books)

def combined_books := brother_books + sarah_books + alex_books

theorem books_difference : combined_books = peter_books + 22 :=
by
  -- sorry is a placeholder for the missing proof
  sorry

end books_difference_l802_802727


namespace value_of_F_l802_802178

   variables (B G P Q F : ℕ)

   -- Define the main hypothesis stating that the total lengths of the books are equal.
   def fill_shelf := 
     (∃ d a : ℕ, d = B * a + 2 * G * a ∧ d = P * a + 2 * Q * a ∧ d = F * a)

   -- Prove that F equals B + 2G and P + 2Q under the hypothesis.
   theorem value_of_F (h : fill_shelf B G P Q F) : F = B + 2 * G ∧ F = P + 2 * Q :=
   sorry
   
end value_of_F_l802_802178


namespace card_pairs_count_l802_802473

theorem card_pairs_count :
  let cards := {n : ℕ | 1 ≤ n ∧ n ≤ 50}
  let valid_pairs := {(a, b) ∈ cards × cards | 
    a ≠ b ∧ (a - b = 11 ∨ b - a = 11) ∧ (a * b) % 5 = 0}
  in valid_pairs.count = 15 :=
by
  sorry

end card_pairs_count_l802_802473


namespace percent_calculation_l802_802334

theorem percent_calculation (y : ℝ) : (0.3 * (0.6 * y) = 0.18 * y) → (0.18 * y / y) * 100 = 18 := by
  sorry

end percent_calculation_l802_802334


namespace small_pump_filling_time_l802_802002

theorem small_pump_filling_time :
  ∃ S : ℝ, (L = 2) → 
         (1 / 0.4444444444444444 = S + L) → 
         (1 / S = 4) :=
by 
  sorry

end small_pump_filling_time_l802_802002


namespace some_employees_not_managers_l802_802414

-- Definitions of the conditions
def isEmployee : Type := sorry
def isManager : isEmployee → Prop := sorry
def isShareholder : isEmployee → Prop := sorry
def isPunctual : isEmployee → Prop := sorry

-- Given conditions
axiom some_employees_not_punctual : ∃ e : isEmployee, ¬isPunctual e
axiom all_managers_punctual : ∀ m : isEmployee, isManager m → isPunctual m
axiom some_managers_shareholders : ∃ m : isEmployee, isManager m ∧ isShareholder m

-- The statement to be proved
theorem some_employees_not_managers : ∃ e : isEmployee, ¬isManager e :=
by sorry

end some_employees_not_managers_l802_802414


namespace Fran_speed_l802_802676

-- Definitions needed for statements
def Joann_speed : ℝ := 15
def Joann_time : ℝ := 4
def Fran_time : ℝ := 5

-- Formalize the problem in Lean
theorem Fran_speed (Joann_distance : ℝ) (Fran_speed : ℝ) : 
  Joann_distance = Joann_speed * Joann_time →
  Fran_speed * Fran_time = Joann_distance →
  Fran_speed = 12 :=
by
  -- assume the conditions about distances
  intros h1 h2
  -- prove the goal
  sorry

end Fran_speed_l802_802676


namespace problem_statement_l802_802211

noncomputable def alpha : ℝ := 3 + Real.sqrt 8
noncomputable def beta : ℝ := 3 - Real.sqrt 8
noncomputable def x : ℝ := alpha^(500)
noncomputable def N : ℝ := alpha^(500) + beta^(500)
noncomputable def n : ℝ := N - 1
noncomputable def f : ℝ := x - n
noncomputable def one_minus_f : ℝ := 1 - f

theorem problem_statement : x * one_minus_f = 1 :=
by
  -- Insert the proof here
  sorry

end problem_statement_l802_802211


namespace probability_of_four_each_color_after_removal_l802_802884

noncomputable def prob_four_each_color_after_removal : ℚ := sorry

theorem probability_of_four_each_color_after_removal :
  let urn_start : Finset (Σ (n : ℕ), Fin n → bool) := 
    {⟨2, id⟩, ⟨0, id⟩, ⟨1, id⟩}
  ∃ box : Finset (Σ (n : ℕ), Fin n → bool),
  ∀ n m k : ℕ, 

  (urn_start = {⟨2, fun _ => true⟩, ⟨1, fun _ => false⟩}) ∧
  (∀ t, urn_start ⊆ urn_start ∪ ({t} : Finset (Σ (n : ℕ), Fin n → bool))) ∧
  
  (prob_four_each_color_after_removal = 5 / 63) :=
sorry

end probability_of_four_each_color_after_removal_l802_802884


namespace comparison_inequalities_l802_802952

-- Define the variables a, b, and c as given
def a : ℝ := 3.5
def c : ℝ := Real.log 32 / Real.log 2  -- Log base 2 of 32 is 5
noncomputable def b : ℝ := Real.cos 2

-- State the proposition to be proven
theorem comparison_inequalities : c < b ∧ b < a :=
by
  have hc : c = 5 :=
    calc
      c = Real.log 32 / Real.log 2 : by rfl
    _ = 5 : by norm_num
  have hb : b < 0 :=
    calc
      Real.cos 2 < 0 : by sorry
  have ha : a = 3.5 := rfl
  sorry

end comparison_inequalities_l802_802952


namespace average_price_of_5_baskets_l802_802746

theorem average_price_of_5_baskets :
  (∀ (n : ℕ) (price : ℕ), n = 4 → price = 4 → ∃ total_cost, total_cost = 4 * 4) →
  (fifth_basket_price = 8) →
  (avg_price = (16 + 8) / 5) →
  avg_price = 24 / 5 := by
  intros _ _ _ _
  sorry

end average_price_of_5_baskets_l802_802746


namespace unique_root_set_condition_l802_802131

theorem unique_root_set_condition (m : ℝ) :
  {x : ℝ | m * x^2 + 2 * x - 1 = 0}.card = 1 ↔ m = 0 ∨ m = -1 :=
by sorry

end unique_root_set_condition_l802_802131


namespace ellipse_in_rectangle_l802_802389

theorem ellipse_in_rectangle (x y : ℝ) :
  abs x + abs y ≤ 3 →
  3 ≤ sqrt (x^2 + 3 * y^2) ∧ sqrt (x^2 + 3 * y^2) ≤ max (3 * abs y) (4 * abs x) →
  (∃ a b : ℝ, a < b ∧ is_ellipse x y a ∧ is_ellipse x y b ∧ is_inscribed (ellipse x y a) (rectangle 3 2) ∧ is_inscribed (ellipse x y b) (rectangle 3 2)) := sorry

noncomputable def is_ellipse (x y a : ℝ) : Prop := sorry
noncomputable def is_inscribed (ellipse : ℝ → ℝ → ℝ → Prop) (rect : ℝ → ℝ → Prop) : Prop := sorry
noncomputable def rectangle (width height : ℝ) : ℝ → ℝ → Prop := sorry

end ellipse_in_rectangle_l802_802389


namespace nalani_fraction_sold_is_3_over_8_l802_802717

-- Definitions of conditions
def num_dogs : ℕ := 2
def puppies_per_dog : ℕ := 10
def total_amount_received : ℕ := 3000
def price_per_puppy : ℕ := 200

-- Calculation of total puppies and sold puppies
def total_puppies : ℕ := num_dogs * puppies_per_dog
def puppies_sold : ℕ := total_amount_received / price_per_puppy

-- Fraction of puppies sold
def fraction_sold : ℚ := puppies_sold / total_puppies

theorem nalani_fraction_sold_is_3_over_8 :
  fraction_sold = 3 / 8 :=
sorry

end nalani_fraction_sold_is_3_over_8_l802_802717


namespace geometric_sequence_problem_l802_802093

noncomputable theory

def a (n : ℕ) : ℕ := 2 ^ n

def aa (n : ℕ) : ℕ := 2 ^ (2 ^ n)

theorem geometric_sequence_problem (n : ℕ) :
  (aa (n + 1)) / (list.prod (list.map aa (list.range n.succ))) = 4 :=
by
  sorry

end geometric_sequence_problem_l802_802093


namespace arrangement_count_l802_802008

theorem arrangement_count : 
  ∃ (n : ℕ), n = 48 ∧ 
    (let mentors := {1, 2, 3};
         students := {4, 5, 6};
         pairs := [{(1, 4)}, {(2, 5)}, {(3, 6)}];
         all_pairs := pairs.permutations;
         valid_arrangements := all_pairs.filter (λ p, is_valid p)) in
            n = valid_arrangements.length) :=
begin
  existsi 48,
  split,
  { refl, },
  { 
    -- Consider the given problem constraints and definitions
    let mentors := {1, 2, 3},
    let students := {4, 5, 6},
    let pairs := [{(1, 4)}, {(2, 5)}, {(3, 6)}],
    
    -- Calcuate the valid arrangements length considering the permutations and validity conditions
    let all_pairs := pairs.permutations,
    let valid_arrangements := all_pairs.filter (λ p, 
      ∀ i j, (i < pairs.length - 1) → ((i, i+1) ∈ combinations(pairs[i])).subset_combinations),
    
    -- Ensure the length of valid arrangements is equal to the known number of arrangements
    exact (valid_arrangements.length = 48) sorry,
  }
end

end arrangement_count_l802_802008


namespace product_of_solutions_l802_802929

theorem product_of_solutions :
  let solutions := {x : ℝ | |x| = 3 * (|x| - 2)} in
  ∏ x in solutions, x = -9 := by
  sorry

end product_of_solutions_l802_802929


namespace tens_digit_of_closest_to_30000_l802_802272

theorem tens_digit_of_closest_to_30000 :
  ∃ N : ℕ, (∃ (a b c d e : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
                          b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
                          c ≠ d ∧ c ≠ e ∧
                          d ≠ e ∧
                          {a, b, c, d, e} = {2, 3, 5, 7, 8} ∧
                          N = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
                          abs (N - 30000) ≤ abs (M - 30000) ∀ M, (∀ (f g h i j : ℕ), f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
                                                              g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
                                                              h ≠ i ∧ h ≠ j ∧
                                                              i ≠ j ∧
                                                              {f, g, h, i, j} = {2, 3, 5, 7, 8} ∧
                                                              M = f * 10000 + g * 1000 + h * 100 + i * 10 + j)) :
    (N / 10) % 10 = 5 :=
by
  sorry

end tens_digit_of_closest_to_30000_l802_802272


namespace card_pairs_with_conditions_l802_802450

theorem card_pairs_with_conditions : 
  let cards := Finset.range 51 in
  let pairs := (cards.product cards).filter (λ p, (p.1 - p.2).natAbs = 11 ∧ (p.1 * p.2) % 5 = 0) in
  pairs.card / 2 = 15 :=
by
  sorry

end card_pairs_with_conditions_l802_802450


namespace crescents_area_eq_rectangle_area_l802_802766

noncomputable def rectangle_area (a b : ℝ) : ℝ := 4 * a * b

noncomputable def semicircle_area (r : ℝ) : ℝ := (1 / 2) * Real.pi * r^2

noncomputable def circumscribed_circle_area (a b : ℝ) : ℝ :=
  Real.pi * (a^2 + b^2)

noncomputable def combined_area (a b : ℝ) : ℝ :=
  rectangle_area a b + 2 * (semicircle_area a) + 2 * (semicircle_area b)

theorem crescents_area_eq_rectangle_area (a b : ℝ) : 
  combined_area a b - circumscribed_circle_area a b = rectangle_area a b :=
by
  unfold combined_area
  unfold circumscribed_circle_area
  unfold rectangle_area
  unfold semicircle_area
  sorry

end crescents_area_eq_rectangle_area_l802_802766


namespace count_valid_pairs_l802_802447

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

def difference_is_11 (a b : ℕ) : Prop := abs (a - b) = 11

def valid_card (a b : ℕ) : Prop :=
  a ∈ finset.range 51 ∧ b ∈ finset.range 51 ∧
  difference_is_11 a b ∧ (is_divisible_by_5 a ∨ is_divisible_by_5 b)

theorem count_valid_pairs : finset.card {p : ℕ × ℕ | valid_card p.1 p.2 ∧ p.1 < p.2}.to_finset = 15 := 
  sorry

end count_valid_pairs_l802_802447


namespace sum_of_positive_k_for_integer_solution_eq_27_l802_802762

theorem sum_of_positive_k_for_integer_solution_eq_27 :
  ∑ k in { k : ℤ | ∃ α β : ℤ, α * β = -18 ∧ k = α + β ∧ k > 0 }.toFinset, k = 27 :=
begin
  sorry
end

end sum_of_positive_k_for_integer_solution_eq_27_l802_802762


namespace obtuse_triangle_count_l802_802615

theorem obtuse_triangle_count : 
  let n := 100 in
  let angles_between_consecutive_vertices := 3.6 in
  let is_obtuse (i j k : ℕ) := 
    let distance1 := if j > i then j - i else j - i + n in
    let distance2 := if k > j then k - j else k - j + n in
    let distance3 := if i > k then i - k else i - k + n in
    (distance1 * angles_between_consecutive_vertices > 90) ∨
    (distance2 * angles_between_consecutive_vertices > 90) ∨
    (distance3 * angles_between_consecutive_vertices > 90) in
  ∑ i in finset.range n, ∑ j in finset.range n, ∑ k in finset.range n, 
    ite (i < j ∧ j < k ∧ is_obtuse i j k) 1 0 = 117600 :=
sorry

end obtuse_triangle_count_l802_802615


namespace count_valid_pairs_l802_802443

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

def difference_is_11 (a b : ℕ) : Prop := abs (a - b) = 11

def valid_card (a b : ℕ) : Prop :=
  a ∈ finset.range 51 ∧ b ∈ finset.range 51 ∧
  difference_is_11 a b ∧ (is_divisible_by_5 a ∨ is_divisible_by_5 b)

theorem count_valid_pairs : finset.card {p : ℕ × ℕ | valid_card p.1 p.2 ∧ p.1 < p.2}.to_finset = 15 := 
  sorry

end count_valid_pairs_l802_802443


namespace vector_magnitude_subtraction_l802_802264

-- Definitions
def angle_between (a b : ℝ × ℝ) := 30 * (π / 180)  -- 30 degrees in radians
def magnitude (v : ℝ × ℝ) := real.sqrt (v.1 ^ 2 + v.2 ^ 2)
def dot_product (a b : ℝ × ℝ) := a.1 * b.1 + a.2 * b.2

-- Hypotheses
def a : ℝ × ℝ := (1, 0)
axiom b : ℝ × ℝ
axiom b_magnitude : magnitude b = real.sqrt 3
axiom angle_a_b : dot_product a b = magnitude a * magnitude b * real.cos (angle_between a b)

-- Proof statement
theorem vector_magnitude_subtraction : magnitude (a - b) = 1 := by
  sorry

end vector_magnitude_subtraction_l802_802264


namespace expected_groups_l802_802788

/-- Given a sequence of k zeros and m ones arranged in random order and divided into 
alternating groups of identical digits, the expected value of the total number of 
groups is 1 + 2 * k * m / (k + m). -/
theorem expected_groups (k m : ℕ) (h_pos : k + m > 0) :
  let total_groups := 1 + 2 * k * m / (k + m)
  total_groups = (1 + 2 * k * m / (k + m)) := by
  sorry

end expected_groups_l802_802788


namespace scientific_notation_of_6_ronna_l802_802017

def "ronna" := 27

theorem scientific_notation_of_6_ronna :
  "ronna" = 27 → 6 * 10^27 = 6 * 10^27 := 
by
  intro ronna_def
  rw ronna_def
  exact rfl

end scientific_notation_of_6_ronna_l802_802017


namespace B_catches_up_with_A_l802_802351

-- Define the conditions
def speed_A : ℝ := 10 -- A's speed in kmph
def speed_B : ℝ := 20 -- B's speed in kmph
def delay : ℝ := 6 -- Delay in hours after A's start

-- Define the total distance where B catches up with A
def distance_catch_up : ℝ := 120

-- Statement to prove B catches up with A at 120 km from the start
theorem B_catches_up_with_A :
  (speed_A * delay + speed_A * (distance_catch_up / speed_B - delay)) = distance_catch_up :=
by
  sorry

end B_catches_up_with_A_l802_802351


namespace percentage_of_females_l802_802243

theorem percentage_of_females (total_passengers : ℕ)
  (first_class_percentage : ℝ) (male_fraction_first_class : ℝ)
  (females_coach_class : ℕ) (h1 : total_passengers = 120)
  (h2 : first_class_percentage = 0.10)
  (h3 : male_fraction_first_class = 1/3)
  (h4 : females_coach_class = 40) :
  (females_coach_class + (first_class_percentage * total_passengers - male_fraction_first_class * (first_class_percentage * total_passengers))) / total_passengers * 100 = 40 :=
by
  sorry

end percentage_of_females_l802_802243


namespace angle_AC_BC_half_pi_l802_802618

theorem angle_AC_BC_half_pi : 
  ∀ (AB AC : ℝ × ℝ), 
  AB = (4, 0) → 
  AC = (2, 2) → 
  let BC := (AC.1 - AB.1, AC.2 - AB.2) in 
  (AC.1 * BC.1 + AC.2 * BC.2 = 0) → 
  ∠ AC BC = π / 2 := 
by
  sorry

end angle_AC_BC_half_pi_l802_802618


namespace vacant_seats_l802_802353

theorem vacant_seats (filled_percentage : ℝ) (total_seats : ℕ) 
  (h_filled_percentage : filled_percentage = 75) (h_total_seats : total_seats = 600) : 
  (25 / 100) * total_seats = 150 :=
by
  rw [←h_total_seats, ←h_filled_percentage]
  sorry

end vacant_seats_l802_802353


namespace Dave_pays_4_more_than_Doug_l802_802044

-- Define the conditions
def pizza_cost : ℝ := 8
def anchovy_cost : ℝ := 2
def number_of_slices : ℕ := 8
def Dave_slices_with_anchovies : ℕ := 4
def Dave_plain_slices : ℕ := 1
def Doug_plain_slices : ℕ := 3

-- Calculate total cost
def total_pizza_cost : ℝ := pizza_cost + anchovy_cost

-- Calculate cost per slice
def cost_per_slice : ℝ := total_pizza_cost / number_of_slices

-- Calculate Dave's total cost
def Dave_total_cost : ℝ := (cost_per_slice * (Dave_slices_with_anchovies + Dave_plain_slices))

-- Calculate Doug's total cost
def Doug_total_cost : ℝ := (cost_per_slice * Doug_plain_slices)

-- Calculate the difference in payment
def difference_in_payment : ℝ := Dave_total_cost - Doug_total_cost

-- State the theorem
theorem Dave_pays_4_more_than_Doug : difference_in_payment = 4 := by
  sorry

end Dave_pays_4_more_than_Doug_l802_802044


namespace red_balls_in_bag_l802_802366

open Nat

def binom : ℕ → ℕ → ℕ
| n, k := if k > n then 0 else (factorial n) / ((factorial k) * (factorial (n - k)))

theorem red_balls_in_bag (r : ℕ) (h1 : r <= 16) (h2 : (binom r 2 * binom (16 - r) 1 : ℚ) / binom 16 3 = 1 / 10) :
  r = 7 :=
by
  sorry

end red_balls_in_bag_l802_802366


namespace candy_problem_l802_802392

theorem candy_problem (a : ℕ) (h₁ : a % 10 = 6) (h₂ : a % 15 = 11) (h₃ : 200 ≤ a) (h₄ : a ≤ 250) :
  a = 206 ∨ a = 236 :=
sorry

end candy_problem_l802_802392


namespace trajectory_of_C_is_circle_l802_802858

theorem trajectory_of_C_is_circle
  (a b : ℝ)
  (h1 : a^2 + b^2 = 9)
  (x y : ℝ)
  (h2 : x = a / 3)
  (h3 : y = b / 3)
  (h4 : ∀ (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ), 
    A = (a, 0) ∧ B = (0, b) ∧ C = (x, y) → vector AC = 2 • vector CB) :
  x^2 + y^2 = 1 :=
by sorry

end trajectory_of_C_is_circle_l802_802858


namespace absolute_error_proof_l802_802512

variables (a : ℝ) (εa : ℝ)
def relative_error_condition := εa = 0.0004
def value_a := a = 1348
def absolute_error := Δ a = |a| * εa

theorem absolute_error_proof
  (h1 : relative_error_condition εa)
  (h2 : value_a a)
  : Δ a = 0.5 :=
by
  rw [value_a, relative_error_condition] at h2 h1
  sorry

end absolute_error_proof_l802_802512


namespace fence_pole_count_l802_802373

-- Define the conditions
def path_length : ℕ := 900
def bridge_length : ℕ := 42
def pole_spacing : ℕ := 6

-- Define the goal
def total_poles : ℕ := 286

-- The statement to prove
theorem fence_pole_count :
  let total_length_to_fence := (path_length - bridge_length)
  let poles_per_side := total_length_to_fence / pole_spacing
  let total_poles_needed := 2 * poles_per_side
  total_poles_needed = total_poles :=
by
  sorry

end fence_pole_count_l802_802373


namespace rise_in_water_level_l802_802349

theorem rise_in_water_level 
    (edge : ℝ) (length : ℝ) (width : ℝ) 
    (h_edge : edge = 15) 
    (h_length : length = 20) 
    (h_width : width = 14) : 
    let V_cube := edge^3 in
    let A_base := length * width in
    V_cube / A_base = 12.05 := by
  have h1 : V_cube = 15^3 := by rw [h_edge]
  have h2 : V_cube = 3375 := by rw h1
  have h3 : A_base = 20 * 14 := by rw [h_length, h_width]
  have h4 : A_base = 280 := by rw h3
  have h5 : 3375 / 280 = 12.05357 := by norm_num
  have h6 : 12.05 = 12.05357 := by norm_num
  rw [h2, h4, h5, h6]
  sorry

end rise_in_water_level_l802_802349


namespace students_failed_exam_l802_802726

theorem students_failed_exam {total_students scored_100_percent remaining_students passing_students failed_students : ℕ}
  (h_total : total_students = 80)
  (h_scored_100_percent_fraction : 2 / 5)
  (h_scored_100_percent : scored_100_percent = (2 * total_students) / 5)
  (h_remaining_students : remaining_students = total_students - scored_100_percent)
  (h_passing_students_fraction : 1 / 2)
  (h_passing_students : passing_students = (remaining_students * 1) / 2)
  (h_failed_students : failed_students = remaining_students - passing_students) :
  failed_students = 24 :=
sorry

end students_failed_exam_l802_802726


namespace range_of_fx_over_x_l802_802768

variable (f : ℝ → ℝ)

noncomputable def is_odd (f : ℝ → ℝ) :=
  ∀ x, f (-x) = -f x

noncomputable def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x < f y

theorem range_of_fx_over_x (odd_f : is_odd f)
                           (increasing_f_pos : is_increasing_on f {x : ℝ | x > 0})
                           (hf1 : f (-1) = 0) :
  {x | f x / x < 0} = {x | -1 < x ∧ x < 0} ∪ {x | 0 < x ∧ x < 1} :=
sorry

end range_of_fx_over_x_l802_802768


namespace perfect_square_l802_802098

-- Define the sequence a_n
def seq_a : ℕ → ℤ
| 0       := 0 -- Note: We introduce a_0 = 0 for completeness
| 1       := 1
| (n+2)   := -seq_a (n+1) - 2 * seq_a n

-- Define the statement to prove that 2^(n+1) - 7 * (seq_a (n-1))^2 is a perfect square
theorem perfect_square (n : ℕ) (hn : n ≥ 2) : ∃ k : ℤ, 2^(n+1) - 7 * (seq_a (n-1))^2 = k^2 := by
  sorry

end perfect_square_l802_802098


namespace Fran_speed_l802_802674

-- Definitions needed for statements
def Joann_speed : ℝ := 15
def Joann_time : ℝ := 4
def Fran_time : ℝ := 5

-- Formalize the problem in Lean
theorem Fran_speed (Joann_distance : ℝ) (Fran_speed : ℝ) : 
  Joann_distance = Joann_speed * Joann_time →
  Fran_speed * Fran_time = Joann_distance →
  Fran_speed = 12 :=
by
  -- assume the conditions about distances
  intros h1 h2
  -- prove the goal
  sorry

end Fran_speed_l802_802674


namespace distribute_items_into_identical_bags_l802_802019

theorem distribute_items_into_identical_bags :
  let items := 6 in let bags := 3 in
  number_of_ways_to_distribute items bags = 62 :=
by
  sorry

end distribute_items_into_identical_bags_l802_802019


namespace cards_difference_product_divisible_l802_802471

theorem cards_difference_product_divisible :
  let S := {n | 1 ≤ n ∧ n ≤ 50}
  let valid_pairs := {(a, b) | a ∈ S ∧ b ∈ S ∧ |a - b| = 11 ∧ (a * b) % 5 = 0}
  valid_pairs.card = 15 := 
sorry

end cards_difference_product_divisible_l802_802471


namespace problem_solution_l802_802943

def tens_digit_is_odd (n : ℕ) : Bool :=
  let m := (n * n + n) / 10 % 10
  m % 2 = 1

def count_tens_digit_odd : ℕ :=
  List.range 50 |>.filter tens_digit_is_odd |>.length

theorem problem_solution : count_tens_digit_odd = 25 :=
  sorry

end problem_solution_l802_802943


namespace worth_of_used_car_l802_802023

theorem worth_of_used_car (earnings remaining : ℝ) (earnings_eq : earnings = 5000) (remaining_eq : remaining = 1000) : 
  ∃ worth : ℝ, worth = earnings - remaining ∧ worth = 4000 :=
by
  sorry

end worth_of_used_car_l802_802023


namespace min_lambda_value_l802_802964

noncomputable def min_lambda (x λ : ℝ) : Prop :=
  λ > 0 ∧ (∀ x > 0, (λ * x).exp - (Real.log x / (2 * λ)) ≥ 0)

theorem min_lambda_value : ∃ λ, min_lambda (e (λ := 1 / (2 * Real.exp 1))) := 
sorry

end min_lambda_value_l802_802964


namespace solve_for_t_l802_802617

variables {V : Type*} [inner_product_space ℝ V]

noncomputable def find_scalar_t (a b : V) (t : ℝ) : Prop :=
  (⟪a, a⟫ = 1) ∧ (⟪b, b⟫ = 1) ∧ (⟪a, b⟫ = 1/2) ∧ (⟪a, t • b - a⟫ = 0) → t = 2

theorem solve_for_t (a b : V) (h1 : ⟪a, a⟫ = 1) (h2 : ⟪b, b⟫ = 1) (h3 : ⟪a, b⟫ = 1/2) (h4 : ⟪a, t • b - a⟫ = 0) : 
  find_scalar_t a b t :=
sorry

end solve_for_t_l802_802617


namespace sum_common_divisors_l802_802545

-- Define the sum of a set of numbers
def set_sum (s : Set ℕ) : ℕ :=
  s.fold (λ x acc => x + acc) 0

-- Define the divisors of a number
def divisors (n : ℕ) : Set ℕ :=
  { d | d > 0 ∧ n % d = 0 }

-- Definitions based on the given conditions
def divisors_of_60 : Set ℕ := divisors 60
def divisors_of_18 : Set ℕ := divisors 18
def common_divisors : Set ℕ := divisors_of_60 ∩ divisors_of_18

-- Declare the theorem to be proved
theorem sum_common_divisors : set_sum common_divisors = 12 :=
  sorry

end sum_common_divisors_l802_802545


namespace number_of_factors_l802_802800

-- Definitions:
def is_prime (n : ℕ) := 2 ≤ n ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def has_three_factors (n : ℕ) := ∃ p : ℕ, is_prime p ∧ n = p^2

variable {x y z : ℕ}

-- Conditions:
hypothesis h1 : x ≠ y ∧ y ≠ z ∧ x ≠ z
hypothesis h2 : has_three_factors x ∧ has_three_factors y ∧ has_three_factors z

-- Proof statement:
theorem number_of_factors (hx : has_three_factors x) (hy : has_three_factors y) (hz : has_three_factors z) :
  ∏ (p : ℕ) in {x^3, y^4, z^5} , (p + 1) = 693 :=
sorry

end number_of_factors_l802_802800


namespace parallel_lines_in_rectangle_proof_l802_802588

/-- Proof problem for rectangle geometry involving circles and projections -/
theorem parallel_lines_in_rectangle_proof 
  (A B C D O E F G X Y Z L M N P Q : Point)
  (h_rect : rectangle A B C D ∧ (A ≠ C ∨ B ≠ D))
  (h_perp_bisec : perp_bisector (B, D) O)
  (h_in_triangle : inside_triangle B C D O)
  (h_circle : ∃ (O : Point), circle_with_center O B D)
  (h_inter_E : circle O B D ∩ AB = E ∧ E ≠ B)
  (h_inter_F : circle O B D ∩ DA = F ∧ F ≠ D)
  (h_BF_DE_inter_G : ∃ (G : Point), line B F ∩ line D E = G)
  (h_PROJ_G : proj G AB = X ∧ proj G BD = Y ∧ proj G DA = Z)
  (h_PROJ_O : proj O CD = L ∧ proj O BD = M ∧ proj O BC = N)
  (h_INTER_PQ1 : line X Y ∩ line M L = P)
  (h_INTER_PQ2 : line Y Z ∩ line M N = Q) :
  parallel (line B P) (line D Q) := sorry

end parallel_lines_in_rectangle_proof_l802_802588


namespace card_pairs_with_conditions_l802_802449

theorem card_pairs_with_conditions : 
  let cards := Finset.range 51 in
  let pairs := (cards.product cards).filter (λ p, (p.1 - p.2).natAbs = 11 ∧ (p.1 * p.2) % 5 = 0) in
  pairs.card / 2 = 15 :=
by
  sorry

end card_pairs_with_conditions_l802_802449


namespace no_eq_nm_l802_802096

variable {Point : Type} [Geometry Point]

-- Conditions
def given_conditions (H G A C B D P Q O1 O2 O N M : Point) (diag_intersects : AC ∩ BD = G) 
  (circumcenter1 : is_circumcenter O1 (triangle A G D)) (circumcenter2 : is_circumcenter O2 (triangle B G C))
  (intersects1 : HG ∩ circle O1 = {P}) (intersects2 : HG ∩ circle O2 = {Q}) (midpoint : is_midpoint M P Q)
  (sec_intersects : O1O2 ∩ OG = {N}) : Prop :=
  true

-- Desired Theorem
theorem no_eq_nm (H G A C B D P Q O1 O2 O N M : Point)
  (h_conditions : given_conditions H G A C B D P Q O1 O2 O N M) :
  distance N O = distance N M :=
by
  sorry

end no_eq_nm_l802_802096


namespace sum_of_common_divisors_60_18_l802_802541

theorem sum_of_common_divisors_60_18 : 
  let a := 60 
  let b := 18 
  let common_divisors := {n | n ∣ a ∧ n ∣ b ∧ n > 0 } 
  (∑ n in common_divisors, n) = 12 :=
by
  let a := 60
  let b := 18
  let common_divisors := { n | n ∣ a ∧ n ∣ b ∧ n > 0 }
  have : (∑ n in common_divisors, n) = 12 := sorry
  exact this

end sum_of_common_divisors_60_18_l802_802541


namespace prod_real_parts_solutions_complex_equation_l802_802274

def prod_real_parts_of_solutions : Prop :=
  ∃ (x₁ x₂ : ℂ), (x₁^2 - 4*x₁ + 2 - 2*I = 0) ∧ (x₂^2 - 4*x₂ + 2 - 2*I = 0) ∧ 
  ((x₁.re * x₂.re) = 3 - Real.sqrt 2)

theorem prod_real_parts_solutions_complex_equation : prod_real_parts_of_solutions :=
sorry

end prod_real_parts_solutions_complex_equation_l802_802274


namespace find_fourth_number_l802_802846

theorem find_fourth_number : 
  ∃ (x : ℝ), (217 + 2.017 + 0.217 + x = 221.2357) ∧ (x = 2.0017) :=
by
  sorry

end find_fourth_number_l802_802846


namespace find_general_term_l802_802944

def sequence_sum (S : ℕ → ℕ) : Prop := ∀ n, S n = 2^n

noncomputable def general_term (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ ∀ n, n ≥ 2 → a n = 2^(n-1)

theorem find_general_term (S : ℕ → ℕ) (a : ℕ → ℕ)
  (hS : sequence_sum S) (ha : general_term a) :
  ∑ i in Finset.range n, a (i+1) = S n :=
by
  sorry

end find_general_term_l802_802944


namespace inverse_100_mod_101_l802_802053

theorem inverse_100_mod_101 :
  ∃ x, (x : ℤ) ≡ 100 [MOD 101] ∧ 100 * x ≡ 1 [MOD 101] :=
by {
  use 100,
  split,
  { exact rfl },
  { norm_num }
}

end inverse_100_mod_101_l802_802053


namespace find_stream_speed_l802_802000

variable (r w : ℝ)

noncomputable def stream_speed:
    Prop := 
    (21 / (r + w) + 4 = 21 / (r - w)) ∧ 
    (21 / (3 * r + w) + 0.5 = 21 / (3 * r - w)) ∧ 
    w = 3 

theorem find_stream_speed : ∃ w, stream_speed r w := 
by
  sorry

end find_stream_speed_l802_802000


namespace complement_of_union_l802_802613

-- Define the sets M and N based on given conditions
def M : set ℝ := { x | -1 < x ∧ x < 1 }
def N : set ℝ := { y | y ≥ 1 }

-- Prove that the complement of M ∪ N is (-∞, -1]
theorem complement_of_union : (set.univ \ (M ∪ N)) = set.Iic (-1) := by
  sorry

end complement_of_union_l802_802613


namespace expected_value_of_groups_l802_802785

noncomputable def expectedNumberOfGroups (k m : ℕ) : ℝ :=
  1 + (2 * k * m) / (k + m)

theorem expected_value_of_groups (k m : ℕ) :
  k > 0 → m > 0 → expectedNumberOfGroups k m = 1 + 2 * k * m / (k + m) :=
by
  intros
  unfold expectedNumberOfGroups
  sorry

end expected_value_of_groups_l802_802785


namespace range_of_a_l802_802908

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + a| - |x + 1| < 2 * a) ↔ a ∈ set.Ioo (1 / 3 : ℝ) (real.number_of_infinity) :=
sorry

end range_of_a_l802_802908


namespace lake_glacial_monoliths_count_l802_802838

variables (N : ℕ) (N_sand : ℕ) (N_marineClay : ℕ) (N_lakeGlacial : ℕ)
variables (sand_loam_freq marine_clay_loam_freq : ℚ)

/-- Conditions: 
  * N = 198
  * sand_loam_freq = 1/9
  * marine_clay_loam_freq = 11/18
  * sand loams are not marine --/

axiom condition_approx_198 : N = 198
axiom condition_sand_loam_freq : sand_loam_freq = 1 / 9
axiom condition_marine_clay_loam_freq : marine_clay_loam_freq = 11 / 18
axiom condition_non_marine_sand : ∀ (n : ℕ), n ≤ N → N_sand = (sand_loam_freq * N).to_nat

theorem lake_glacial_monoliths_count : N_lakeGlacial = 77 :=
by
  -- We assert that N_sand = (sand_loam_freq * N).to_nat , N_marineClay = (marine_clay_loam_freq * N).to_nat and since sand loams are not marine:
  have h1 : N_sand = (sand_loam_freq * N).to_nat, from condition_non_marine_sand N sorry,
  have h2 : N_marineClay = (marine_clay_loam_freq * N).to_nat, from sorry,
  have approximate_N : N = 198, from condition_approx_198,
  have N_sand_val : (sand_loam_freq * N).to_nat = 22, from sorry,
  have N_marineClay_val : (marine_clay_loam_freq * N).to_nat = 121, from sorry,
  -- Hence the number of lake-glacial genesis monoliths is:
  have N_lakeGlacial_val : N_lakeGlacial = N - N_marineClay + N_sand, from sorry,
  exact N_lakeGlacial_val

end lake_glacial_monoliths_count_l802_802838


namespace bilinear_map_condition_l802_802061

theorem bilinear_map_condition (a b c d z : ℂ) (hz : 0 < z.im) :
  (a * d - b * c) > 0 → (im ((a * z + b) / (c * z + d)) > 0) :=
sorry

end bilinear_map_condition_l802_802061


namespace intersection_of_sets_l802_802969

noncomputable def A := {x : ℝ | 3^(3 - x) < 6}
noncomputable def B := {x : ℝ | log 10 (x - 1) < 1}

theorem intersection_of_sets :
  A ∩ B = {x : ℝ | 2 - log 3 2 < x ∧ x < 11} :=
by sorry

end intersection_of_sets_l802_802969


namespace monotonic_intervals_range_k_l802_802089

noncomputable theory

def f (x a : ℝ) := (1 + Real.log x) / (2 * a * x)

theorem monotonic_intervals (a : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, 0 < x → 1 < x → a > 0 → (deriv (λ x, f x a) x > 0) ↔ (0 < x ∧ x < 1) ∨ (1 < x ∧ x < ∞))
  ∧ (∀ x : ℝ, 0 < x → 1 < x → a < 0 → (deriv (λ x, f x a) x < 0) ↔ (0 < x ∧ x < 1) ∨ (1 < x ∧ x < ∞)) :=
sorry

theorem range_k (x₁ x₂ : ℝ) (h1 : x₁ ≠ x₂) (h2 : 1 < x₁) (h3 : x₁ < x₂) (h4 : x₂ < ∞) (k : ℝ) :
  ∀ x : ℝ, 1 < x → f x (1/2) + k * Real.log x < 0 → k < 1 / Real.exp 1 :=
sorry

end monotonic_intervals_range_k_l802_802089


namespace no_integer_solutions_quadratic_l802_802250

theorem no_integer_solutions_quadratic (n : ℤ) (s : ℕ) (pos_odd_s : s % 2 = 1) :
  ¬ ∃ x : ℤ, x^2 - 16 * n * x + 7^s = 0 :=
sorry

end no_integer_solutions_quadratic_l802_802250


namespace range_of_c_l802_802579

def a : ℝ := 4
def b : ℝ := sorry -- Assume b is in the range 4 < b < 6
def AngleA : ℝ := sorry
def AngleC : ℝ := sorry

-- Definition of angles
axiom h1 : sin (2 * AngleA) = sin AngleC

-- Definition of the range for b
axiom h_b_range : 4 < b ∧ b < 6

-- Proof goal: The range of values for c
theorem range_of_c (c : ℝ) (h : a = 4 ∧ b ∈ set.Ioo 4 6 ∧ sin (2 * AngleA) = sin AngleC) :
  4 * (sqrt 2) < c ∧ c < 2 * (sqrt 10) :=
sorry

end range_of_c_l802_802579


namespace card_pairs_satisfying_conditions_l802_802438

theorem card_pairs_satisfying_conditions :
  ∃ (s : Finset (ℕ × ℕ)),
    (∀ p ∈ s, p.1 ≠ p.2 ∧ (p.1 = p.2 + 11 ∨ p.2 = p.1 + 11) ∧ ((p.1 * p.2) % 5 = 0))
    ∧ s.card = 15 :=
begin
  sorry
end

end card_pairs_satisfying_conditions_l802_802438


namespace number_of_pipes_l802_802774

theorem number_of_pipes (h_same_height : forall (height : ℝ), height > 0)
  (diam_large : ℝ) (hl : diam_large = 6)
  (diam_small : ℝ) (hs : diam_small = 1) :
  (π * (diam_large / 2)^2) / (π * (diam_small / 2)^2) = 36 :=
by
  sorry

end number_of_pipes_l802_802774


namespace solve_matrix_det_eq_l802_802938

open Real

noncomputable def solve_matrix_det (x : ℝ) : Prop :=
  let M := matrix.stdBasisMatrix (Fin 2) (Fin 2) 0 0 
  (sin x) 
  (matrix.stdBasisMatrix (Fin 2) (Fin 2) 0 1 1)
  (matrix.stdBasisMatrix (Fin 2) (Fin 2) 1 0 1)
  (matrix.stdBasisMatrix (Fin 2) (Fin 2) 1 1 (4 * cos x))
  det M = 0 → (∃ k : ℤ, x = (π / 12) + k * π ∨ x = (5 * π / 12) + k * π)

-- Proof is omitted
theorem solve_matrix_det_eq (x : ℝ) : solve_matrix_det x :=
sorry

end solve_matrix_det_eq_l802_802938


namespace chord_intercept_min_value_l802_802991

noncomputable def minimum_value_of_fraction (a b : ℝ) : ℝ :=
  if a > 0 ∧ b > 0 ∧ a + b = 1 then (2 / a + 3 / b) else 0

theorem chord_intercept_min_value : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (a + b = 1) ∧ minimum_value_of_fraction a b = 5 + 2 * Real.sqrt 6 :=
begin
    use [1 - Real.sqrt 6, Real.sqrt 6],
    split,
    { exact sub_pos_of_lt (lt_sqrt.2 (show 1 < 6, by norm_num)) },
    split,
    { exact sqrt_pos.2 (by norm_num) },
    split,
    { norm_num },
    { rw minimum_value_of_fraction,
      split_ifs,
      norm_num,
      exact congr_arg (fun x => 5 + x) (Real.sqrt_mul (two_ne_zero'.ne.symm) (sqrt_nonneg _).symm) },
end

end chord_intercept_min_value_l802_802991


namespace shark_sightings_in_Daytona_Beach_l802_802431

def CM : ℕ := 7

def DB : ℕ := 3 * CM + 5

theorem shark_sightings_in_Daytona_Beach : DB = 26 := by
  sorry

end shark_sightings_in_Daytona_Beach_l802_802431


namespace sum_of_common_divisors_60_18_l802_802536

def positive_divisors (n : ℕ) : List ℕ :=
  List.filter (λ d, n % d = 0) (List.range (n+1))

def common_divisors (m n : ℕ) : List ℕ :=
  List.filter (λ d, d ∈ positive_divisors m) (positive_divisors n)

theorem sum_of_common_divisors_60_18 : 
  List.sum (common_divisors 60 18) = 12 := by
  sorry

end sum_of_common_divisors_60_18_l802_802536


namespace k_value_when_parallel_projection_e₁_on_e₂_l802_802105

variables (e₁ e₂ : ℝ^3) (k : ℝ)
variables (a b : ℝ^3) (λ : ℝ)

-- Conditions
axiom conditions (h₁ : e₁ ≠ e₂) (h₂ : ‖e₁‖ = 1) (h₃ : ‖e₂‖ = 1) (h₄ : a = e₁ - 2 * e₂) (h₅ : b = k * e₁ + e₂)

-- First part: parallel vectors
theorem k_value_when_parallel (h₆ : b = λ * a) : k = -1 / 2 :=
sorry

-- Second part: never perpendicular and projection
theorem projection_e₁_on_e₂ (h₇ : ∀ k ∈ ℝ, a • b ≠ 0) : (e₁ • e₂) / ‖e₂‖ = 1 / 2 :=
sorry

end k_value_when_parallel_projection_e₁_on_e₂_l802_802105


namespace range_of_xy_l802_802976

theorem range_of_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y + x * y = 30) :
  12 < x * y ∧ x * y < 870 :=
by sorry

end range_of_xy_l802_802976


namespace least_m_value_l802_802707

noncomputable def number_of_trailing_zeros (n : Nat) : Nat :=
  (List.range (Nat.log n 5)).foldl (λ acc k, acc + n / (5 ^ k)) 0

theorem least_m_value :
  ∃ (m : Nat), m > 0 ∧
    (let p := number_of_trailing_zeros m in
     number_of_trailing_zeros (2 * m) = Nat.floor (5 * p / 2)) ∧
    m = 25 :=
by
  sorry

end least_m_value_l802_802707


namespace percent_of_y_equal_to_30_percent_of_60_percent_l802_802320

variable (y : ℝ)

theorem percent_of_y_equal_to_30_percent_of_60_percent (hy : y ≠ 0) :
  ((0.18 * y) / y) * 100 = 18 := by
  have hneq : y ≠ 0 := hy
  field_simp [hneq]
  norm_num
  sorry

end percent_of_y_equal_to_30_percent_of_60_percent_l802_802320


namespace polynomial_roots_unique_l802_802751

theorem polynomial_roots_unique 
  (x1 x2 x3 x4 : ℝ)
  (h1 : x1 + x2 = 5)
  (h2 : x3 * x4 = 20)
  (h3 : x1 + x2 + x3 + x4 = 14)
  (h4 : x1 * x2 * x3 * x4 = 120)
  (h_poly : ∀ x, (x - x1) * (x - x2) * (x - x3) * (x - x4) = x^4 - 14*x^3 + 71*x^2 - 154*x + 120) :
  {x1, x2, x3, x4} = {2, 3, 4, 5} :=
by
  sorry

end polynomial_roots_unique_l802_802751


namespace part1_interval_part2_positive_l802_802607

section
  variable {x : ℝ} 

  -- Definition of the function for part (1)
  def f1 (x : ℝ) : ℝ := Real.exp (x - 1) - Real.log x

  -- Math proof problem for part (1)
  theorem part1_interval : 
    (∀ x, 0 < x ∧ x < 1 → f1 x + f'1 x < 0) ∧ (∀ x, 1 < x → f1 x + f'1 x > 0) := 
  sorry

  -- Definition of the function for part (2)
  def f2 (x : ℝ) : ℝ := Real.exp (x - 2) - Real.log x

  -- Math proof problem for part (2)
  theorem part2_positive (x : ℝ) (h : x > 0) : f2 x > 0 := 
  sorry
end

end part1_interval_part2_positive_l802_802607


namespace three_digit_number_divisible_by_7_l802_802335

theorem three_digit_number_divisible_by_7 (t : ℕ) :
  (n : ℕ) = 600 + 10 * t + 5 →
  n ≥ 100 ∧ n < 1000 →
  n % 10 = 5 →
  (n / 100) % 10 = 6 →
  n % 7 = 0 →
  n = 665 :=
by
  sorry

end three_digit_number_divisible_by_7_l802_802335


namespace fixed_point_X_exists_l802_802244

open EuclideanGeometry

theorem fixed_point_X_exists (ABC : Triangle)
    (D : Point) (A B C : ABC.Verts) (D_on_BC : D ∈ BC)
    (P : Point) (P_on_AB : P ∈ AB) :
    ∃ (X : Point), (∀ (P : Point), (P ∈ AB) →
      let R := midpoint A P in
      let Q := line_intersect (line_through P C) (line_through A D) in
      collinear R Q X) :=
sorry

end fixed_point_X_exists_l802_802244


namespace range_of_a_l802_802635

noncomputable def f (a x : ℝ) : ℝ := x^3 - 3 * x + a

theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) ↔ (-2 < a ∧ a < 2) :=
by
  sorry

end range_of_a_l802_802635


namespace tension_limit_l802_802844

theorem tension_limit (M m g : ℝ) (hM : 0 < M) (hg : 0 < g) :
  (∀ T, (T = Mg ↔ m = 0) → (∀ ε, 0 < ε → ∃ m₀, m > m₀ → |T - 2 * M * g| < ε)) :=
by 
  sorry

end tension_limit_l802_802844


namespace arithmetic_sequence_formula_geometric_sequence_formula_sum_sequence_formula_l802_802572

-- Define the arithmetic sequence a_n with a positive common difference and a_1 = 1
def a_n (n : ℕ) : ℕ := 2 * n - 1

-- Conditions
axiom h1 : a_n 1 = 1
axiom h2 : a_n 2 * a_n 14 = (a_n 6 - 2 * a_n 1) ^ 2

-- Define the geometric sequence b_n with b_1 = a_2, b_2 = a_6 - 2 * a_1, b_3 = a_14
def b_n (n : ℕ) : ℕ := 3^n

-- Prove the general formula for the arithmetic sequence
theorem arithmetic_sequence_formula (n : ℕ) : a_n n = 2 * n - 1 := 
by sorry

-- Prove the general formula for the geometric sequence
theorem geometric_sequence_formula (n : ℕ) : b_n n = 3^n :=
by sorry

-- Define the sequence c_n = 1 / (a_n * a_(n+1))
def c_n (n : ℕ) : ℚ := 1 / (a_n n * a_n (n + 1))

-- Define the sum S_n of the first n terms of the sequence c_n
def S_n (n : ℕ) : ℚ := ∑ i in finset.range n, c_n i

-- Prove the sum formula for the sequence c_n
theorem sum_sequence_formula (n : ℕ) : S_n n = n / (2 * n + 1) :=
by sorry

end arithmetic_sequence_formula_geometric_sequence_formula_sum_sequence_formula_l802_802572


namespace total_weight_correct_total_money_earned_correct_l802_802293

variable (records : List Int) (std_weight : Int)

-- Conditions
def deviation_sum (records : List Int) : Int := records.foldl (· + ·) 0

def batch_weight (std_weight : Int) (n : Int) (deviation_sum : Int) : Int :=
  deviation_sum + std_weight * n

def first_day_sales (total_weight : Int) (price_per_kg : Int) : Int :=
  price_per_kg * (total_weight / 2)

def second_day_sales (total_weight : Int) (first_day_sales_weight : Int) (discounted_price_per_kg : Int) : Int :=
  discounted_price_per_kg * (total_weight - first_day_sales_weight)

def total_earnings (first_day_sales : Int) (second_day_sales : Int) : Int :=
  first_day_sales + second_day_sales

-- Proof statements
theorem total_weight_correct : 
  deviation_sum records = 4 ∧ std_weight = 30 ∧ records.length = 8 → 
  batch_weight std_weight records.length (deviation_sum records) = 244 :=
by
  intro h
  sorry

theorem total_money_earned_correct :
  first_day_sales (batch_weight std_weight records.length (deviation_sum records)) 10 = 1220 ∧
  second_day_sales (batch_weight std_weight records.length (deviation_sum records)) (batch_weight std_weight records.length (deviation_sum records) / 2) (10 * 9 / 10) = 1098 →
  total_earnings 1220 1098 = 2318 :=
by
  intro h
  sorry

end total_weight_correct_total_money_earned_correct_l802_802293


namespace bottle_caps_total_l802_802712

def initial_bottle_caps := 51.0
def given_bottle_caps := 36.0

theorem bottle_caps_total : initial_bottle_caps + given_bottle_caps = 87.0 := by
  sorry

end bottle_caps_total_l802_802712


namespace probability_abcd_eq_two_l802_802337

theorem probability_abcd_eq_two : 
  (∑a b c d, ite (a * b * c * d = 2) (1 / 6^4) 0) = 1 / 324 :=
by
  sorry

end probability_abcd_eq_two_l802_802337


namespace proof_a_gt_c_gt_b_l802_802561

noncomputable def a : ℝ := Real.exp (1 / 2) - 1
noncomputable def b : ℝ := Real.log (3 / 2)
def c : ℝ := 5 / 12

theorem proof_a_gt_c_gt_b : (a > c) ∧ (c > b) := by
  sorry

end proof_a_gt_c_gt_b_l802_802561


namespace vector_sum_dot_product_is_ten_l802_802639

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / (2 * x - 4)
def P : ℝ × ℝ := (2, 1)
def O : ℝ × ℝ := (0, 0)
variable (A B : ℝ × ℝ)
variable (l : ℝ × ℝ -> Bool) -- l is a line passing through point P

-- Condition: Line passes through P
def line_passes_through_P : Prop := l P

-- Condition: Line intersects the graph of f at points A and B
def line_intersects_f_at_A_and_B : Prop :=
  l A ∧ l B ∧ f A.1 = A.2 ∧ f B.1 = B.2

-- Condition: Calculation of the vector sum and dot product
def vector_sum_dot_product (A B P O : ℝ × ℝ) : ℝ :=
  let OA := (A.1 - 0, A.2 - 0)
  let OB := (B.1 - 0, B.2 - 0)
  let OP := (P.1 - 0, P.2 - 0)
  let sum_OA_OB := (OA.1 + OB.1, OA.2 + OB.2)
  let dot_product := sum_OA_OB.1 * OP.1 + sum_OA_OB.2 * OP.2
  dot_product

theorem vector_sum_dot_product_is_ten
  (h1 : line_passes_through_P)
  (h2 : line_intersects_f_at_A_and_B) :
  vector_sum_dot_product A B P O = 10 := by
  sorry

end vector_sum_dot_product_is_ten_l802_802639


namespace complex_modulus_conjugate_l802_802597

theorem complex_modulus_conjugate (z : ℂ) (hz : z = (1 + complex.i) / (3 - 4 * complex.i)) : 
  complex.abs (conj z) = (Real.sqrt 2) / 5 :=
by 
  sorry

end complex_modulus_conjugate_l802_802597


namespace verify_figure_equality_l802_802141

noncomputable def figures_are_identical (A B : Type) [IsFigure A] [IsFigure B] : Prop :=
  ∃ (transparentCopy : A → A), 
    (∀ (overlay : A → B), overlay (transparentCopy A) = B) → A = B

theorem verify_figure_equality (A B : Type) [IsFigure A] [IsFigure B] : figures_are_identical A B → A = B :=
by {
  sorry
}

end verify_figure_equality_l802_802141


namespace equal_area_parallelograms_iff_X_on_diagonal_l802_802801

-- Definitions of the parallelogram and the point X
variables {A B C D X : Type} [parallelogram A B C D] [point_inside_parallelogram X A B C D]

-- Define that lines through point X are drawn parallel to the sides of the parallelogram
variables (lines_through_X_parallel_to_sides : ∀ l, line_through X l → parallel_to_side l A B C D)

theorem equal_area_parallelograms_iff_X_on_diagonal :
  S_ABCX A B C X = S_ADC X D C ↔ point_on_diagonal X A C :=
sorry

end equal_area_parallelograms_iff_X_on_diagonal_l802_802801


namespace sum_of_common_divisors_60_18_l802_802540

theorem sum_of_common_divisors_60_18 : 
  let a := 60 
  let b := 18 
  let common_divisors := {n | n ∣ a ∧ n ∣ b ∧ n > 0 } 
  (∑ n in common_divisors, n) = 12 :=
by
  let a := 60
  let b := 18
  let common_divisors := { n | n ∣ a ∧ n ∣ b ∧ n > 0 }
  have : (∑ n in common_divisors, n) = 12 := sorry
  exact this

end sum_of_common_divisors_60_18_l802_802540


namespace g_min_value_l802_802606

def f (a : ℝ) (x : ℝ) : ℝ := a * 3^x + 1 / 3^(x-1)

lemma even_function (a : ℝ) (h : ∀ x : ℝ, f a x = f a (-x)) : a = 3 :=
by
  have h_pos : ∀ x, (a - 3) * (3^x - 3^(-x)) = 0 := by sorry
  have h_any_x : ∀ x : ℝ, 3^x - 3^(-x) ≠ 0 := by sorry
  have h_a : a - 3 = 0 := by sorry
  simp [h_a]

def g (m : ℝ) (x : ℝ) : ℝ :=
  9^x + 9^(-x) + m * (3 * 3^x + 1 / 3^(x-1)) + m^2 - 1

theorem g_min_value (m : ℝ) : 
  ∃ x : ℝ, ∀ y : ℝ, g m x ≤ g m y ∧ 
  (m < -(4/3) → g m x = -(5/4)*m^2 - 3) ∧ 
  (m ≥ -(4/3) → g m x = m^2 + 6*m + 1) :=
by
  let u (x : ℝ) : ℝ := 3^x + 3^(-x)
  have hu_ge_2 : ∀ x : ℝ, u x ≥ 2 := by sorry
  let y (m : ℝ) (u : ℝ) := u^2 + 3*m*u + m^2 - 3
  have m_ge_4_3_min : ∀ m, m ≥ -(4/3) → ∃ x, y m (u x) = m^2 + 6*m + 1 ∧ 
                                       ∀ u',  u' ≥ 2 → y m (u x) ≤ y m u' := by sorry
  have m_lt_4_3_min : ∀ m, m <  -(4/3) → ∃ x, y m (-3*m/2) = -(5/4)*m^2 - 3 ∧ 
                                    ∀ u',  u' ≥ 2 → y m (-3*m/2) ≤ y m u' := by sorry
  sorry

end g_min_value_l802_802606


namespace calculate_expression_l802_802897

theorem calculate_expression : 
  (12 * 0.5 * 3 * 0.0625 - 1.5) = -3 / 8 := 
by 
  sorry 

end calculate_expression_l802_802897


namespace sum_common_divisors_sixty_and_eighteen_l802_802524

theorem sum_common_divisors_sixty_and_eighteen : 
  ∑ d in ({d ∈ ({1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60} : finset ℕ) | d ∈ ({1, 2, 3, 6, 9, 18} : finset ℕ)} : finset ℕ), d = 12 :=
by sorry

end sum_common_divisors_sixty_and_eighteen_l802_802524


namespace Fran_same_distance_speed_l802_802680

noncomputable def Joann_rides (v_j t_j : ℕ) : ℕ := v_j * t_j

def Fran_speed (d t_f : ℕ) : ℕ := d / t_f

theorem Fran_same_distance_speed
  (v_j t_j t_f : ℕ) (hj: v_j = 15) (tj: t_j = 4) (tf: t_f = 5) : Fran_speed (Joann_rides v_j t_j) t_f = 12 := by
  have hj_dist: Joann_rides v_j t_j = 60 := by
    rw [hj, tj]
    sorry -- proof of Joann's distance
  have d_j: ℕ := 60
  have hf: Fran_speed d_j t_f = Fran_speed 60 5 := by
    rw ←hj_dist
    sorry -- proof to equate d_j with Joann's distance
  show Fran_speed 60 5 = 12
  sorry -- Final computation proof

end Fran_same_distance_speed_l802_802680


namespace find_parabola_vertex_l802_802940

-- Define the parabola with specific roots.
def parabola (x : ℝ) : ℝ := -x^2 + 2 * x + 24

-- Define the vertex of the parabola.
def vertex : ℝ × ℝ := (1, 25)

-- Prove that the vertex of the parabola is indeed at (1, 25).
theorem find_parabola_vertex : vertex = (1, 25) :=
  sorry

end find_parabola_vertex_l802_802940


namespace lines_intersect_at_single_point_l802_802215

-- Given conditions
variables {A B C X : Type} [IsTriangle A B C] (X_inside_ABC : InsideTriangle X A B C)
variables (XA BC XB AC XC AB : ℝ)
variable (I1 I2 I3 : Type)
[Incenter X B C I1] [Incenter X C A I2] [Incenter X A B I3]

-- The conditions
axiom eq_condition_1 : XA * BC = XB * AC
axiom eq_condition_2 : XB * AC = XC * AB

-- Statement to prove
theorem lines_intersect_at_single_point 
  (h1 : XA * BC = XB * AC)
  (h2 : XB * AC = XC * AB) : 
  ∃ P, LineThrough (A, I1, P) ∧ LineThrough (B, I2, P) ∧ LineThrough (C, I3, P) := 
sorry

end lines_intersect_at_single_point_l802_802215


namespace pipeA_rate_l802_802247

-- Definitions based on conditions
def capacity := 850
def rateB := 30 -- liters per minute
def rateC := 20 -- liters per minute
def cycle_duration := 3 -- minutes
def total_time := 51 -- minutes
def cycles := total_time / cycle_duration

-- Proof statement
theorem pipeA_rate (rateA : ℕ) :
  let net_addition_per_cycle := rateA + rateB - rateC in
  let total_addition := cycles * net_addition_per_cycle in
  total_addition = capacity → rateA = 40 :=
by
  intro h
  sorry

end pipeA_rate_l802_802247


namespace find_triples_l802_802059

theorem find_triples (x y z : ℝ) 
  (h1 : (1/3 : ℝ) * min x y + (2/3 : ℝ) * max x y = 2017)
  (h2 : (1/3 : ℝ) * min y z + (2/3 : ℝ) * max y z = 2018)
  (h3 : (1/3 : ℝ) * min z x + (2/3 : ℝ) * max z x = 2019) :
  (x = 2019) ∧ (y = 2016) ∧ (z = 2019) :=
sorry

end find_triples_l802_802059


namespace sum_common_divisors_sixty_and_eighteen_l802_802522

theorem sum_common_divisors_sixty_and_eighteen : 
  ∑ d in ({d ∈ ({1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60} : finset ℕ) | d ∈ ({1, 2, 3, 6, 9, 18} : finset ℕ)} : finset ℕ), d = 12 :=
by sorry

end sum_common_divisors_sixty_and_eighteen_l802_802522


namespace percent_of_percent_l802_802327

theorem percent_of_percent (y : ℝ) : 0.3 * (0.6 * y) = 0.18 * y :=
sorry

end percent_of_percent_l802_802327


namespace locus_of_T_is_pair_of_tangents_l802_802566

noncomputable def locus_of_T (C : Circle) (L : Line) (O : Point) (hO : O ∈ L) : Set Point :=
  {T | ∃ (P : Point) (hP : P ∈ L), is_tangent (Circle.mk P (dist P O)) C T ∧ dist P T = dist P O}

theorem locus_of_T_is_pair_of_tangents (C : Circle) (L : Line) (O : Point) (hO : O ∈ L) :
  locus_of_T C L O hO =
  {T | ∃ (P1 P2 : Point), is_tangent P1 C T ∧ is_perpendicular L (Line.mk P1 O) ∧ is_tangent P2 C T ∧ is_perpendicular L (Line.mk P2 O)} :=
sorry

end locus_of_T_is_pair_of_tangents_l802_802566


namespace height_of_first_building_l802_802377

theorem height_of_first_building (h : ℕ) (h_condition : h + 2 * h + 9 * h = 7200) : h = 600 :=
by
  sorry

end height_of_first_building_l802_802377


namespace arithmetic_sequence_a20_l802_802592

theorem arithmetic_sequence_a20 (a : ℕ → ℝ) (d S : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) = a 1 + n * d) → -- Arithmetic sequence definition
  (∀ n : ℕ, S n = n * a 1 + (n * (n - 1) * d) / 2) → -- Sum of first n terms
  (S 6 = 8 * S 3) → -- Given condition S6 = 8 * S3
  (a 3 - a 5 = 8) → -- Given condition a3 - a5 = 8
  a 20 = -74 := -- To prove a_20 = -74
by
  sorry

end arithmetic_sequence_a20_l802_802592


namespace find_sum_x1_x2_l802_802999

-- Define sets A and B with given properties
def set_A : Set ℝ := {x | -2 < x ∧ x < -1 ∨ x > 1}
def set_B (x1 x2 : ℝ) : Set ℝ := {x | x1 ≤ x ∧ x ≤ x2}

-- Conditions of union and intersection
def union_condition (x1 x2 : ℝ) : Prop := set_A ∪ set_B x1 x2 = {x | x > -2}
def intersection_condition (x1 x2 : ℝ) : Prop := set_A ∩ set_B x1 x2 = {x | 1 < x ∧ x ≤ 3}

-- Main theorem to prove
theorem find_sum_x1_x2 (x1 x2 : ℝ) (h_union : union_condition x1 x2) (h_intersect : intersection_condition x1 x2) :
  x1 + x2 = 2 :=
sorry

end find_sum_x1_x2_l802_802999


namespace part_a_part_b_l802_802424

variable {R : Type} [LinearOrder R] [CommRing R] [Zero R] [One R] [Inv R] [Pow R ℕ]
variables (x : ℕ → R)

-- Part (a)
theorem part_a : (∀ n, 0 ≤ x n) → x 0 = 1 → (∀ n, x n ≥ x (n+1)) → ∃ n ≥ 1, ∑ i in Finset.range n, (x i ^ 2 / x (i + 1)) ≥ (3.999 : R) :=
by sorry

-- Part (b)
theorem part_b : ∃ (x : ℕ → R), (∀ n, x n = 1 / 2 ^ n) ∧ (∀ n, ∑ i in Finset.range n, (x i ^ 2 / x (i + 1)) < (4 : R)) :=
by sorry

end part_a_part_b_l802_802424


namespace Fran_speed_l802_802675

-- Definitions needed for statements
def Joann_speed : ℝ := 15
def Joann_time : ℝ := 4
def Fran_time : ℝ := 5

-- Formalize the problem in Lean
theorem Fran_speed (Joann_distance : ℝ) (Fran_speed : ℝ) : 
  Joann_distance = Joann_speed * Joann_time →
  Fran_speed * Fran_time = Joann_distance →
  Fran_speed = 12 :=
by
  -- assume the conditions about distances
  intros h1 h2
  -- prove the goal
  sorry

end Fran_speed_l802_802675


namespace most_convenient_numbering_l802_802962

-- Define conditions
def population_size : ℕ := 106
def sample_size : ℕ := 10
def numbering_a := list.range 106
def numbering_b := list.range (106 + 1) \ {0}
def numbering_c := list.iota 106
def numbering_d := list.range 106 |>.map (λ n, n % 1000)

-- Define the problem
theorem most_convenient_numbering : 
  example_specification population_size sample_size := 
  numbering_d = list.range 106 |>.map (λ n, n % 1000) :=
begin
  sorry
end

end most_convenient_numbering_l802_802962


namespace number_of_integers_containing_2_l802_802151

/-- 
  How many of the base-ten numerals for the positive integers less than or equal to 2537 contain the digit '2'?
-/
def count_digits_with_2 (n : ℕ) : ℕ :=
  if h : n ≤ 2537 then
    List.length (List.filter (λ i, '2' ∈ i.digits 10) (List.range' 1 (n + 1)))
  else 0

theorem number_of_integers_containing_2 : count_digits_with_2 2537 = 655 := by {
  sorry
}

end number_of_integers_containing_2_l802_802151


namespace minimum_value_expression_l802_802702

theorem minimum_value_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ z, (z = a^2 + b^2 + 1 / a^2 + 2 * b / a) ∧ z ≥ 2 :=
sorry

end minimum_value_expression_l802_802702


namespace greatest_possible_difference_in_rectangles_area_l802_802302

theorem greatest_possible_difference_in_rectangles_area :
  ∃ (l1 w1 l2 w2 l3 w3 : ℤ),
    2 * l1 + 2 * w1 = 148 ∧
    2 * l2 + 2 * w2 = 150 ∧
    2 * l3 + 2 * w3 = 152 ∧
    (∃ (A1 A2 A3 : ℤ),
      A1 = l1 * w1 ∧
      A2 = l2 * w2 ∧
      A3 = l3 * w3 ∧
      (max (abs (A1 - A2)) (max (abs (A1 - A3)) (abs (A2 - A3))) = 1372)) :=
by
  sorry

end greatest_possible_difference_in_rectangles_area_l802_802302


namespace infinite_k_no_prime_l802_802074

def not_prime (n : ℕ) : Prop := n ≤ 1 ∨ ∃ (p : ℕ), 1 < p ∧ p < n ∧ p ∣ n

def sequence (k : ℕ) : ℕ → ℕ
| 0           := 1
| 1           := k + 2
| (n + 2 : ℕ) := (k+1) * sequence (n+1) - sequence n

theorem infinite_k_no_prime :
  ∃ᶠ k in at_top, ∃ (m : ℕ), k = m^2 - 3 ∧ (∀ n, not_prime (sequence k n)) :=
by
  sorry

end infinite_k_no_prime_l802_802074


namespace find_smallest_m_l802_802694

-- Assume we have some n : ℕ and n ≥ 2
def n : ℕ := 4 -- example value; replace as needed
axiom n_ge_2 : n ≥ 2

-- No three points are collinear amongst the n points
axiom no_three_collinear {points : list (ℝ × ℝ)} (h : points.length = n) : 
  ∀ (i j k : ℕ) (hi : i ≠ j) (hj : j ≠ k) (hk : k ≠ i), 
  ¬ collinear (points.nth_le i hi) (points.nth_le j hj) (points.nth_le k hk)

-- The smallest m such that for any points X ≠ Y, there is a line separating them.
def smallest_m := Nat.ceil (n / 2 : ℝ)

theorem find_smallest_m 
    (points : list (ℝ × ℝ)) 
    (h_points_len : points.length = n) : 
    (∃ (m : ℕ), m = smallest_m ∧ 
      (∀ (X Y : (ℝ × ℝ)), X ≠ Y → 
       ∃ (lines : list (ℝ × ℝ → Prop)), 
         lines.length = m ∧ 
         (∀ (line : ℝ × ℝ → Prop), ∃ (X Y : (ℝ × ℝ)), X ≠ Y → ¬ line X ∧ line Y))) :=
sorry

end find_smallest_m_l802_802694


namespace number_of_possible_radii_l802_802423

theorem number_of_possible_radii : 
  let valid_radii := {r : ℕ | r < 120 ∧ 120 % r = 0} in
  valid_radii.size = 15 :=
by
  sorry

end number_of_possible_radii_l802_802423


namespace percent_of_percent_l802_802326

theorem percent_of_percent (y : ℝ) : 0.3 * (0.6 * y) = 0.18 * y :=
sorry

end percent_of_percent_l802_802326


namespace card_pairs_count_l802_802476

theorem card_pairs_count :
  let cards := {n : ℕ | 1 ≤ n ∧ n ≤ 50}
  let valid_pairs := {(a, b) ∈ cards × cards | 
    a ≠ b ∧ (a - b = 11 ∨ b - a = 11) ∧ (a * b) % 5 = 0}
  in valid_pairs.count = 15 :=
by
  sorry

end card_pairs_count_l802_802476


namespace probability_greater_than_15_over_16_l802_802581

-- Definitions and conditions
def f (x : ℝ) : ℝ := sorry   -- Assume there's a function f
def g (x : ℝ) : ℝ := sorry   -- Assume there's a function g
axiom g_ne_zero {x : ℝ} : g(x) ≠ 0
axiom condition1 {x : ℝ} : f(x) * (deriv g x) > (deriv f x) * g(x)
axiom f_eq_ax_g {x : ℝ} (a : ℝ) (hpos_a : 0 < a) (hne1_a : a ≠ 1) : f(x) = a^x * g(x)
axiom specific_condition : (f 1) / (g 1) + (f (-1)) / (g (-1)) = 5 / 2

-- The problem statement
theorem probability_greater_than_15_over_16 {k : ℕ} (hk : 1 ≤ k ∧ k ≤ 10) :
  let a := 1/2 in
  (f_eq_ax_g a (by norm_num) (by norm_num)) ∧
  (∑ i in finset.range k, a^i) > 15 / 16 →
  (∑ i in finset.range k, a^i) > 15 / 16 :=
begin
  sorry
end

end probability_greater_than_15_over_16_l802_802581


namespace find_DG_l802_802254

variables {a b S k l : ℕ}
variables {BC : ℕ}

def rectangles_equal_areas (a b S k l : ℕ) : Prop := a * k = b * l

def area_expression (S a b : ℕ) : Prop := S = 53 * (a + b)

def bc_constant : BC = 53 := sorry

theorem find_DG (a b S k l : ℕ) (h1 : rectangles_equal_areas a b S k l)
                       (h2 : area_expression S a b) (h3 : bc_constant):
  k = 2862 :=
sorry

end find_DG_l802_802254


namespace sum_of_odds_square_l802_802241

theorem sum_of_odds_square (n : ℕ) (h : 0 < n) : (Finset.range n).sum (λ i => 2 * i + 1) = n ^ 2 :=
sorry

end sum_of_odds_square_l802_802241


namespace cards_difference_product_divisibility_l802_802460

theorem cards_difference_product_divisibility :
  (∃ pairs : list (ℕ × ℕ), 
    (∀ p ∈ pairs, 1 ≤ p.1 ∧ p.1 ≤ 50 ∧ 1 ≤ p.2 ∧ p.2 ≤ 50 ∧ 
      (p.1 - p.2 = 11 ∨ p.2 - p.1 = 11) ∧ 
      (p.1 * p.2) % 5 = 0) ∧ 
    pairs.length = 15) := 
sorry

end cards_difference_product_divisibility_l802_802460


namespace sum_of_common_divisors_60_18_l802_802542

theorem sum_of_common_divisors_60_18 : 
  let a := 60 
  let b := 18 
  let common_divisors := {n | n ∣ a ∧ n ∣ b ∧ n > 0 } 
  (∑ n in common_divisors, n) = 12 :=
by
  let a := 60
  let b := 18
  let common_divisors := { n | n ∣ a ∧ n ∣ b ∧ n > 0 }
  have : (∑ n in common_divisors, n) = 12 := sorry
  exact this

end sum_of_common_divisors_60_18_l802_802542


namespace inv_100_mod_101_l802_802056

theorem inv_100_mod_101 : (100 : ℤ) * (100 : ℤ) % 101 = 1 := by
  sorry

end inv_100_mod_101_l802_802056


namespace circle_triangle_area_l802_802301

-- Define the radii of the circles
def radius_P : Real := 1
def radius_Q : Real := 2
def radius_R : Real := 3

-- Define the centers of the circles
def center_P := (0 : Real, radius_P)
def center_Q := (3 : Real, radius_Q) -- distance from P to Q is 3
def center_R := (4 : Real, radius_R) -- distance from P to R is 4

-- Define the area of a triangle given three points
noncomputable def triangle_area (A B C : Real × Real) : Real :=
  0.5 * Real.abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- State the problem as a theorem
theorem circle_triangle_area :
  triangle_area center_P center_Q center_R = Real.sqrt 6 - Real.sqrt 2 :=
by
  sorry

end circle_triangle_area_l802_802301


namespace part1_part2_l802_802604

open Real

-- Given condition f(x) = (1 - a^2) ln x - (1/3) x^3 for x > 0 and a ∈ ℝ
def f (x : ℝ) (a : ℝ) : ℝ := 
  (1 - a^2) * log x - (1/3) * x^3

-- Given condition g(x) = f(x) + 3/x + 6 ln x and g(x) is a monotonically decreasing function
def g (x : ℝ) (a : ℝ) : ℝ :=
  f x a + 3 / x + 6 * log x

-- Prove that g(x) is monotonically decreasing implies a ≤ -√3 ∨ a ≥ √3
theorem part1 (a : ℝ) :
  (∀ x > 0, diff.g (a) x ≤ 0) → (a ≤ -sqrt 3 ∨ a ≥ sqrt 3) :=
sorry

-- For a = 0 and m > n ≥ 2, prove nf(m) < mf(n)
theorem part2 {m n : ℝ} (h : m > n ∧ n ≥ 2) : 
  let a := 0 in n * f m a < m * f n a :=
sorry

end part1_part2_l802_802604


namespace sum_common_divisors_l802_802546

-- Define the sum of a set of numbers
def set_sum (s : Set ℕ) : ℕ :=
  s.fold (λ x acc => x + acc) 0

-- Define the divisors of a number
def divisors (n : ℕ) : Set ℕ :=
  { d | d > 0 ∧ n % d = 0 }

-- Definitions based on the given conditions
def divisors_of_60 : Set ℕ := divisors 60
def divisors_of_18 : Set ℕ := divisors 18
def common_divisors : Set ℕ := divisors_of_60 ∩ divisors_of_18

-- Declare the theorem to be proved
theorem sum_common_divisors : set_sum common_divisors = 12 :=
  sorry

end sum_common_divisors_l802_802546


namespace matrix_invertible_given_l802_802065

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℚ :=
  !![5, -3; -2, 1]

def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  !![-1, -3; -2, -5]

theorem matrix_invertible_given (h : det A ≠ 0) : A⁻¹ = A_inv :=
by
  sorry

end matrix_invertible_given_l802_802065


namespace regular_polygon_dissection_l802_802306

/-- An integer-ratio right triangle is a right triangle whose side lengths are in 
  an integer ratio. -/
structure IntegerRatioRightTriangle (a b c : ℕ) : Prop where
  is_right_triangle : a^2 + b^2 = c^2
  integer_ratio : ∃ k : ℕ, ∃ x y z : ℕ, (a = k * x ∧ b = k * y ∧ c = k * z)

/-- A regular n-sided polygon can be completely dissected into integer-ratio 
  right triangles if and only if n = 4. -/
theorem regular_polygon_dissection (n : ℕ) :
  (∀ p : Polygon, p.regular n →
    ∃ tr : List (IntegerRatioRightTriangle), (Polygon.dissect_into p tr)) ↔ n = 4 := sorry

end regular_polygon_dissection_l802_802306


namespace place_ones_and_zeros_l802_802186

theorem place_ones_and_zeros :
  ∃ (ways : ℕ), ways = 90 ∧ 
  (∀ (board : list (list ℕ)),
    (∀ row ∈ board, (row.sum = 2 ∧ row.length = 4)) ∧ 
    (∀ col, (col < 4) → ((board.map (λ row, row[col])).sum = 2) ∧ ((board.map (λ row, row[col])).length = 4)) → 
    board.join.count (λ x, x = 1) = 8 ∧ 
    board.join.count (λ x, x = 0) = 8) := sorry

end place_ones_and_zeros_l802_802186


namespace Dave_pays_4_more_than_Doug_l802_802043

-- Define the conditions
def pizza_cost : ℝ := 8
def anchovy_cost : ℝ := 2
def number_of_slices : ℕ := 8
def Dave_slices_with_anchovies : ℕ := 4
def Dave_plain_slices : ℕ := 1
def Doug_plain_slices : ℕ := 3

-- Calculate total cost
def total_pizza_cost : ℝ := pizza_cost + anchovy_cost

-- Calculate cost per slice
def cost_per_slice : ℝ := total_pizza_cost / number_of_slices

-- Calculate Dave's total cost
def Dave_total_cost : ℝ := (cost_per_slice * (Dave_slices_with_anchovies + Dave_plain_slices))

-- Calculate Doug's total cost
def Doug_total_cost : ℝ := (cost_per_slice * Doug_plain_slices)

-- Calculate the difference in payment
def difference_in_payment : ℝ := Dave_total_cost - Doug_total_cost

-- State the theorem
theorem Dave_pays_4_more_than_Doug : difference_in_payment = 4 := by
  sorry

end Dave_pays_4_more_than_Doug_l802_802043


namespace expected_groups_l802_802787

/-- Given a sequence of k zeros and m ones arranged in random order and divided into 
alternating groups of identical digits, the expected value of the total number of 
groups is 1 + 2 * k * m / (k + m). -/
theorem expected_groups (k m : ℕ) (h_pos : k + m > 0) :
  let total_groups := 1 + 2 * k * m / (k + m)
  total_groups = (1 + 2 * k * m / (k + m)) := by
  sorry

end expected_groups_l802_802787


namespace expected_number_of_groups_l802_802780

-- Define the conditions
variables (k m : ℕ) (h : 0 < k ∧ 0 < m)

-- Expected value of groups in the sequence
theorem expected_number_of_groups : 
  ∀ k m, (0 < k) → (0 < m) → 
  let total_groups := 1 + (2 * k * m) / (k + m) in total_groups = 1 + (2 * k * m) / (k + m) :=
by
  intros k m hk hm
  let total_groups := 1 + (2 * k * m) / (k + m)
  exact (rfl : total_groups = 1 + (2 * k * m) / (k + m))

end expected_number_of_groups_l802_802780


namespace cards_difference_product_divisible_l802_802464

theorem cards_difference_product_divisible :
  let S := {n | 1 ≤ n ∧ n ≤ 50}
  let valid_pairs := {(a, b) | a ∈ S ∧ b ∈ S ∧ |a - b| = 11 ∧ (a * b) % 5 = 0}
  valid_pairs.card = 15 := 
sorry

end cards_difference_product_divisible_l802_802464


namespace largest_integer_not_containing_multiple_of_7_l802_802810

-- Definitions for the problem
def contains (n m : ℤ) : Prop :=
  let nStr := n.toDigits 10 |>.reverse.intercalate ""
  let mStr := m.toDigits 10 |>.reverse.intercalate ""
  (mStr.length > 0 ∧ ∃ i, mStr.isPrefixOf (nStr.drop i))

-- Theorem statement
theorem largest_integer_not_containing_multiple_of_7 : 
  ∀ n : ℤ, (n < 1000000) → (¬∃ m : ℤ, (7 ∣ m) ∧ m > 0 ∧ contains n m) → n ≤ 999999 :=
sorry

end largest_integer_not_containing_multiple_of_7_l802_802810


namespace last_year_cost_equals_250_l802_802670

variable (C : ℝ)
variable (current_cost deposit paid : ℝ)

-- Conditions
def conditions (last_cost : ℝ) :=
  current_cost = 1.40 * last_cost ∧
  deposit = 0.10 * current_cost ∧
  paid = current_cost - deposit ∧
  paid = 315

-- Prove the cost of last year's costume
theorem last_year_cost_equals_250 (h : conditions C) : C = 250 :=
sorry

end last_year_cost_equals_250_l802_802670


namespace thirteenth_result_is_878_l802_802355

-- Definitions based on the conditions
def avg_25_results : ℕ := 50
def num_25_results : ℕ := 25

def avg_first_12_results : ℕ := 14
def num_first_12_results : ℕ := 12

def avg_last_12_results : ℕ := 17
def num_last_12_results : ℕ := 12

-- Prove the 13th result is 878 given the above conditions.
theorem thirteenth_result_is_878 : 
  ((avg_25_results * num_25_results) - ((avg_first_12_results * num_first_12_results) + (avg_last_12_results * num_last_12_results))) = 878 :=
by
  sorry

end thirteenth_result_is_878_l802_802355


namespace base8_to_base10_4513_l802_802427

theorem base8_to_base10_4513 : (4 * 8^3 + 5 * 8^2 + 1 * 8^1 + 3 * 8^0 = 2379) :=
by
  sorry

end base8_to_base10_4513_l802_802427


namespace solve_for_diamond_l802_802153

theorem solve_for_diamond (d : ℕ) (h1 : d * 9 + 6 = d * 10 + 3) (h2 : d < 10) : d = 3 :=
by
  sorry

end solve_for_diamond_l802_802153


namespace number_of_digits_in_first_3003_even_integers_l802_802823

theorem number_of_digits_in_first_3003_even_integers : 
  let num_even_integers_1_digit := 4,
      num_even_integers_2_digits := (98 - 10) / 2 + 1,
      num_even_integers_3_digits := (998 - 100) / 2 + 1,
      num_even_integers_4_digits := (6006 - 1000) / 2 + 1,
      total_digits := 
        num_even_integers_1_digit * 1 +
        num_even_integers_2_digits * 2 +
        num_even_integers_3_digits * 3 +
        num_even_integers_4_digits * 4
  in total_digits = 11460 :=
sorry

end number_of_digits_in_first_3003_even_integers_l802_802823


namespace cards_difference_product_divisible_l802_802468

theorem cards_difference_product_divisible :
  let S := {n | 1 ≤ n ∧ n ≤ 50}
  let valid_pairs := {(a, b) | a ∈ S ∧ b ∈ S ∧ |a - b| = 11 ∧ (a * b) % 5 = 0}
  valid_pairs.card = 15 := 
sorry

end cards_difference_product_divisible_l802_802468


namespace recurring_decimal_to_fraction_l802_802500

theorem recurring_decimal_to_fraction (h1: (0.3 + 0.\overline{45} : ℝ) = (0.3\overline{45} : ℝ))
    (h2: (0.\overline{45} : ℝ) = (5 / 11 : ℝ))
    (h3: (0.3 : ℝ) = (3 / 10 : ℝ)) : (0.3\overline{45} : ℝ) = (83 / 110 : ℝ) :=
by
    sorry

end recurring_decimal_to_fraction_l802_802500


namespace syllogistic_reasoning_problem_l802_802807

theorem syllogistic_reasoning_problem
  (H1 : ∀ (z : ℂ), ∃ (a b : ℝ), z = a + b * Complex.I)
  (H2 : ∀ (z : ℂ), z = 2 + 3 * Complex.I → Complex.re z = 2)
  (H3 : ∀ (z : ℂ), z = 2 + 3 * Complex.I → Complex.im z = 3) :
  (¬ ∀ (z : ℂ), ∃ (a b : ℝ), z = a + b * Complex.I) → "The conclusion is wrong due to the incorrect major premise" = "A" :=
sorry

end syllogistic_reasoning_problem_l802_802807


namespace largest_valid_8_digit_number_is_correct_l802_802066

noncomputable def is_valid_8_digit_number (n : ℕ) : Prop :=
  n >= 800 ∧ n < 900 ∧
  let d1 := 8 in
  let d2 := n / 10 % 10 in
  let d3 := n % 10 in
  (d1 ≠ 0 ∧ d2 ≠ 0 ∧ d3 ≠ 0 ∧ d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3) ∧
  (n % d1 = 0 ∧ n % d2 = 0 ∧ n % d3 = 0)

theorem largest_valid_8_digit_number_is_correct :
  ∀ n : ℕ, is_valid_8_digit_number n → n ≤ 835 :=
by {
  intros n hn,
  sorry
}

example : is_valid_8_digit_number 835 :=
by {
  unfold is_valid_8_digit_number,
  split,
  { exact dec_trivial },
  { split,
    { exact dec_trivial },
    { rintro ⟨d, ⟩,
      split,
      { exact dec_trivial },
      split,
      { exact dec_trivial },
      split,
      { intros H,
        apply dec_trivial },
      split,
      { intros H,
        rw [H] at *,
        apply dec_trivial },
      { intros H,
        rw [H] at *,
        apply dec_trivial },
      { split,
        { exact dec_trivial },
        { split,
          { exact dec_trivial },
          exact dec_trivial } } } } }

end largest_valid_8_digit_number_is_correct_l802_802066


namespace boats_distribution_l802_802649

/-- Defining the problem of distributing 3 adults and 2 children on boats P, Q, and R -/
def boats_problem := 
  ∃ (P Q R : Set (Sum Nat Nat)), 
    P.card ≤ 3 ∧ 
    Q.card ≤ 2 ∧ 
    R.card ≤ 1 ∧ 
    (∀ p ∈ P, ∃ q ∈ Q, p.isInl → q.isInr) ∧
    (∀ p ∈ P, ∃ r ∈ R, p.isInl → r.isInr) ∧
    (∀ q ∈ Q, ∃ p ∈ P, q.isInl → p.isInr) ∧
    (∀ q ∈ Q, ∃ r ∈ R, q.isInl → r.isInr) ∧
    (∀ r ∈ R, ∃ p ∈ P, r.isInl → p.isInr) ∧
    (∀ r ∈ R, ∃ q ∈ Q, r.isInl → q.isInr) ∧
    (∃ a b c a1 c1, a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a ∈ P ∧ b ∈ P ∧ c ∈ P ∧
      a1 ≠ c1 ∧ a1 ∈ Q ∧ c1 ∈ R ∧ P.card = 3 ∧ Q.card = 2 ∧ R.card = 1)

/-- Statement of the problem's proof in Lean 4 -/
theorem boats_distribution :
  boats_problem ∧ (∑ x in {P, Q, R}, x.card) = 33 :=
sorry

end boats_distribution_l802_802649


namespace find_vectors_cosine_angle_l802_802139

-- Definition of the given vectors and conditions
variables (x y z : ℝ)
def a := (x, 4, 1) : ℝ × ℝ × ℝ
def b := (-2, y, -1) : ℝ × ℝ × ℝ
def c := (3, -2, z) : ℝ × ℝ × ℝ

-- Parallel and perpendicular conditions
def parallel (u v : ℝ × ℝ × ℝ) := ∃ k : ℝ, u = (k * v.1, k * v.2, k * v.3)
def orthogonal (u v : ℝ × ℝ × ℝ) := u.1 * v.1 + u.2 * v.2 + u.3 * v.3 = 0

-- Proof statements
theorem find_vectors :
  parallel a b → orthogonal b c →
  a = (2, 4, 1) ∧ b = (-2, -4, -1) ∧ c = (3, -2, 2) :=
by
  sorry

theorem cosine_angle :
  parallel a b → orthogonal b c →
  let ac := (2 + 3,  4 + -2,  1 + 2) in  -- a + c
  let bc := (-2 + 3, -4 + -2, -1 + 2) in -- b + c
  real.cos ac bc = -2 / 19 :=
by
  sorry

end find_vectors_cosine_angle_l802_802139


namespace lcm_12_18_l802_802924

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l802_802924


namespace locus_of_midpoint_infinite_integer_points_integer_distance_not_integer_l802_802802

noncomputable def parabola {p : ℕ} (hp : prime p ∧ p ≠ 2) : set (ℝ × ℝ) :=
{ xy | xy.2 ^ 2 = 2 * p * xy.1 }

noncomputable def line_through_focus {p : ℕ} (hp : prime p ∧ p ≠ 2) (k : ℝ) (hk : k ≠ 0) : set (ℝ × ℝ) :=
{ xy | xy.2 = k * (xy.1 - p / 2) }

theorem locus_of_midpoint (p : ℕ) (hp : prime p ∧ p ≠ 2)
    (M N : ℝ × ℝ) (hM : M ∈ parabola ⟨p, hp⟩) (hN : N ∈ parabola ⟨p, hp⟩)
    (k : ℝ) (hk : k ≠ 0)
    (M_N_line : ∀ xy, xy ∈ line_through_focus ⟨p, hp⟩ k hk ↔ xy = M ∨ xy = N)
    (P : ℝ × ℝ) (hP : P = midpoint ℝ M N)
    (Q : ℝ × ℝ) (hQ : Q = line.perpendicular_bisector ℝ M N P xy≈Qonxaxis):
  locus_of_midpoint (midpoint ℝ P Q).1 (midpoint ℝ P Q).2) = (4 * y^2 = p * (x - p)) :=
begin
  sorry -- Proof steps omitted
end

theorem infinite_integer_points (p : ℕ) (hp : prime p ∧ p ≠ 2) :
∃ infinitely many (x y : ℤ), 4 * (y : ℝ) ^ 2 = p * ((x : ℝ) - p) := 
begin
  sorry -- Proof steps omitted
end

theorem integer_distance_not_integer (p : ℕ) (hp : prime p ∧ p ≠ 2) :
∀ (x y : ℤ), 4 * (y : ℝ) ^ 2 = p * ((x : ℝ) - p) → (x : ℝ) ^ 2 + (y : ℝ) ^ 2 ≠ m ∀ m : ℤ :=
begin
  sorry -- Proof steps omitted 
end

end locus_of_midpoint_infinite_integer_points_integer_distance_not_integer_l802_802802


namespace percent_of_percent_l802_802328

theorem percent_of_percent (y : ℝ) : 0.3 * (0.6 * y) = 0.18 * y :=
sorry

end percent_of_percent_l802_802328


namespace max_rectangles_cut_l802_802565

theorem max_rectangles_cut (n : ℕ) (hn : 1 ≤ n) : 
  let max_cut := if n = 1 then 2
                 else if n = 2 then 5
                 else 4 * n - 4 in
  ∃ max_number, max_cut = max_number :=
begin
  sorry
end

end max_rectangles_cut_l802_802565


namespace count_valid_pairs_l802_802444

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

def difference_is_11 (a b : ℕ) : Prop := abs (a - b) = 11

def valid_card (a b : ℕ) : Prop :=
  a ∈ finset.range 51 ∧ b ∈ finset.range 51 ∧
  difference_is_11 a b ∧ (is_divisible_by_5 a ∨ is_divisible_by_5 b)

theorem count_valid_pairs : finset.card {p : ℕ × ℕ | valid_card p.1 p.2 ∧ p.1 < p.2}.to_finset = 15 := 
  sorry

end count_valid_pairs_l802_802444


namespace percent_of_y_equal_to_30_percent_of_60_percent_l802_802319

variable (y : ℝ)

theorem percent_of_y_equal_to_30_percent_of_60_percent (hy : y ≠ 0) :
  ((0.18 * y) / y) * 100 = 18 := by
  have hneq : y ≠ 0 := hy
  field_simp [hneq]
  norm_num
  sorry

end percent_of_y_equal_to_30_percent_of_60_percent_l802_802319


namespace card_pairs_satisfying_conditions_l802_802436

theorem card_pairs_satisfying_conditions :
  ∃ (s : Finset (ℕ × ℕ)),
    (∀ p ∈ s, p.1 ≠ p.2 ∧ (p.1 = p.2 + 11 ∨ p.2 = p.1 + 11) ∧ ((p.1 * p.2) % 5 = 0))
    ∧ s.card = 15 :=
begin
  sorry
end

end card_pairs_satisfying_conditions_l802_802436


namespace card_pairs_with_conditions_l802_802452

theorem card_pairs_with_conditions : 
  let cards := Finset.range 51 in
  let pairs := (cards.product cards).filter (λ p, (p.1 - p.2).natAbs = 11 ∧ (p.1 * p.2) % 5 = 0) in
  pairs.card / 2 = 15 :=
by
  sorry

end card_pairs_with_conditions_l802_802452


namespace quadratic_root_difference_l802_802062

theorem quadratic_root_difference : 
  let a := 1
  let b := -9
  let c := 14
  (r1 r2 : ℝ) (h : r1 + r2 = -b / a ∧ r1 * r2 = c / a) (r1 - r2).abs = 5 := by
  sorry

end quadratic_root_difference_l802_802062


namespace peter_remaining_walk_time_l802_802734

-- Define the parameters and conditions
def total_distance : ℝ := 2.5
def time_per_mile : ℝ := 20
def distance_walked : ℝ := 1

-- Define the remaining distance
def remaining_distance : ℝ := total_distance - distance_walked

-- Define the remaining time Peter needs to walk
def remaining_time_to_walk (d : ℝ) (t : ℝ) : ℝ := d * t

-- State the problem we want to prove
theorem peter_remaining_walk_time :
  remaining_time_to_walk remaining_distance time_per_mile = 30 :=
by
  -- Placeholder for the proof
  sorry

end peter_remaining_walk_time_l802_802734


namespace debby_weekly_jog_distance_l802_802722

theorem debby_weekly_jog_distance :
  let monday_distance := 3.0
  let tuesday_distance := 5.5
  let wednesday_distance := 9.7
  let thursday_distance := 10.8
  let friday_distance_miles := 2.0
  let miles_to_km := 1.60934
  let friday_distance := friday_distance_miles * miles_to_km
  let total_distance := monday_distance + tuesday_distance + wednesday_distance + thursday_distance + friday_distance
  total_distance = 32.21868 :=
by
  sorry

end debby_weekly_jog_distance_l802_802722


namespace james_muffins_l802_802413

theorem james_muffins (arthur_muffins : ℕ) (times : ℕ) (james_muffins : ℕ) 
  (h1 : arthur_muffins = 115) 
  (h2 : times = 12) 
  (h3 : james_muffins = arthur_muffins * times) : 
  james_muffins = 1380 := 
by 
  sorry

end james_muffins_l802_802413


namespace constant_term_in_expansion_l802_802084

theorem constant_term_in_expansion (n : ℕ) (hn : ∑i in (finset.range (n+1)), (binom n i) * (x ^ (2 * (n - i)) * (-(1/x) ^ (i/2))) = (x² - 1/√x) ^ n)
  (hz : (binom n 2) / (binom n 4) = 3 / 14) : 
  n = 10 → ((binom 10 8) = 45) :=
begin
  sorry
end

end constant_term_in_expansion_l802_802084


namespace modified_op_3_4_l802_802959

def modified_op (a b : ℝ) : ℝ := (a^2 + b^2) / (1 + a^2 * b^2)

theorem modified_op_3_4 :
  3 > 0 ∧ 4 > 0 → modified_op 3 4 = 25 / 145 := by
  sorry

end modified_op_3_4_l802_802959


namespace sum_of_common_divisors_60_18_l802_802538

theorem sum_of_common_divisors_60_18 : 
  let a := 60 
  let b := 18 
  let common_divisors := {n | n ∣ a ∧ n ∣ b ∧ n > 0 } 
  (∑ n in common_divisors, n) = 12 :=
by
  let a := 60
  let b := 18
  let common_divisors := { n | n ∣ a ∧ n ∣ b ∧ n > 0 }
  have : (∑ n in common_divisors, n) = 12 := sorry
  exact this

end sum_of_common_divisors_60_18_l802_802538


namespace part_i_part_ii_part_iii_l802_802668

-- Definition of the ellipse and relevant points and lines
def Ellipse (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧
  (∀ (x y : ℝ), (x - 2)^2/8 + y^2/2 = 1 →
    (x, y) = (2, 1)) ∧ -- Point M on ellipse
  (∀ (x y : ℝ), sqrt(6) = (x, 0) ∨ -sqrt(6) = (x, 0)) -- Foci

def LineParallelToOM (m x y : ℝ) : Prop :=
  y = (1/2) * x + m

def MaximumAreaCondition (m : ℝ) : Prop :=
  -2 < m ∧ m < 2 ∧ m ≠ 0 ∧ 
  (m = sqrt(2) ∨ m = -sqrt(2))

-- Questions translated into erms of proof
theorem part_i (a b : ℝ) : Ellipse a b →
  a = sqrt(8) ∧ b = sqrt(2) :=
  by sorry

theorem part_ii {m : ℝ} : MaximumAreaCondition m → 
  (x - 2*y + 2*sqrt(2) = 0 ∨ x - 2*y - 2*sqrt(2) = 0) :=
  by sorry

theorem part_iii {x1 x2 y1 y2 : ℝ} (m : ℝ) : 
  x1 ≠ x2 ∧ y1 ≠ y2 →
  let k1 := (y1 - 1)/(x1 - 2) in 
  let k2 := (y2 - 1)/(x2 - 2) in 
  k1 + k2 = 0 :=
  by sorry

end part_i_part_ii_part_iii_l802_802668


namespace relationship_between_M_and_N_l802_802107

noncomputable def a (n : ℕ) : ℝ := sorry -- Define the sequence {a_n} as positive reals
def x := (Finset.range 1996).sum (λ n, a (n + 1))
def y := (Finset.range 1995).sum (λ n, a (n + 2))

theorem relationship_between_M_and_N :
  ∀ (a : ℕ → ℝ), (∀ n, a n > 0) →
  let M := x * (y + a 1997) in
  let N := (x + a 1997) * y in
  M > N := 
begin
  intros a ha,
  let x := (Finset.range 1996).sum (λ n, a (n + 1)),
  let y := (Finset.range 1995).sum (λ n, a (n + 2)),
  have hxy : x > y,
  {
    rw [Finset.sum_range_succ _ 1995, add_comm, ←Finset.sum_range_succ _ 1994],
    apply lt_add_of_pos_right,
    exact ha 1995,
  },
  let M : ℝ := x * (y + a 1997),
  let N : ℝ := (x + a 1997) * y,
  have : a 1997 > 0 := ha 1997,
  have : x * a 1997 > y * a 1997 := 
    mul_lt_mul_of_pos_right hxy this,
  exact add_lt_add_left this (x * y),
end

end relationship_between_M_and_N_l802_802107


namespace husband_catches_yolanda_time_in_minutes_l802_802344

noncomputable def yolanda_speed_mph : ℝ := 20
noncomputable def husband_speed_mph : ℝ := 40
noncomputable def head_start_minutes : ℝ := 15

theorem husband_catches_yolanda_time_in_minutes :
  ∀ (yolanda_speed_mph husband_speed_mph head_start_minutes : ℝ),
    yolanda_speed_mph = 20 →
    husband_speed_mph = 40 →
    head_start_minutes = 15 →
    (head_start_minutes * (yolanda_speed_mph / 60)) / ((husband_speed_mph - yolanda_speed_mph) / 60) = 15 :=
by
  intros yolanda_speed_mph husband_speed_mph head_start_minutes h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  norm_num
  sorry -- proof to be completed

end husband_catches_yolanda_time_in_minutes_l802_802344


namespace inv_100_mod_101_l802_802057

theorem inv_100_mod_101 : (100 : ℤ) * (100 : ℤ) % 101 = 1 := by
  sorry

end inv_100_mod_101_l802_802057


namespace puzzle_solution_l802_802179

theorem puzzle_solution : ∃ n, 3 * n - 6 = 15 ∧ n * 111 = 777 :=
by
  use 7
  split
  { linarith }
  { norm_num }

end puzzle_solution_l802_802179


namespace length_AB_indeterminate_l802_802585

theorem length_AB_indeterminate
  (A B C : Type)
  (AC : ℝ) (BC : ℝ)
  (AC_eq_1 : AC = 1)
  (BC_eq_3 : BC = 3) :
  (2 < AB ∧ AB < 4) ∨ (AB = 2 ∨ AB = 4) → false :=
by sorry

end length_AB_indeterminate_l802_802585


namespace sum_of_common_divisors_60_18_l802_802533

def positive_divisors (n : ℕ) : List ℕ :=
  List.filter (λ d, n % d = 0) (List.range (n+1))

def common_divisors (m n : ℕ) : List ℕ :=
  List.filter (λ d, d ∈ positive_divisors m) (positive_divisors n)

theorem sum_of_common_divisors_60_18 : 
  List.sum (common_divisors 60 18) = 12 := by
  sorry

end sum_of_common_divisors_60_18_l802_802533


namespace ellipse_equation_intersection_point_on_C_l802_802760

def ellipse_C : (a b : ℝ) → Prop := λ a b, 
  (a > b ∧ b > 0) ∧ 
  (sqrt 3 / 2 = b / a) ∧
  (a^2 = 8 ∧ b = sqrt 2)

def points_on_C_sym (M N : ℝ × ℝ) : Prop := 
  ((∃ x₀ y₀ : ℝ, M = (x₀, y₀) ∧ N = (-x₀, y₀)) ∧
  ((x₀ / (2*M.2 - 3))^2 / 8 + ((3*M.2 - 4) / (2*M.2 - 3))^2 / 2 = 1))

theorem ellipse_equation (a b : ℝ) (h : ellipse_C a b) : 
  ∀ (x y : ℝ), x^2 / 8 + y^2 / 2 = 1 ↔
  (h.1.1 ∧ h.1.2 ∧ h.2.1 ∧ h.2.2) := sorry

theorem intersection_point_on_C (M N : ℝ × ℝ) (h : points_on_C_sym M N) : 
  ∀ (T : ℝ × ℝ), 
  (T.1 / (2*T.2 - 3))^2 / 8 + ((3*T.2 - 4) / (2*T.2 - 3))^2 / 2 = 1 :=
  sorry

end ellipse_equation_intersection_point_on_C_l802_802760


namespace shaded_area_percentage_is_84_l802_802380

def large_square_side_length : ℝ := 10
def small_square_side_length : ℝ := 4

def large_square_area : ℝ := large_square_side_length ^ 2
def small_square_area : ℝ := small_square_side_length ^ 2
def shaded_area : ℝ := large_square_area - small_square_area

def percentage_shaded : ℝ := (shaded_area / large_square_area) * 100

theorem shaded_area_percentage_is_84 :
  percentage_shaded = 84 :=
by
  sorry

end shaded_area_percentage_is_84_l802_802380


namespace activities_realign_in_10608_days_l802_802020

open Nat

theorem activities_realign_in_10608_days :
  lcm (lcm (lcm (lcm (lcm (lcm 6 4) 16) 12) 8) 13) 17 = 10608 :=
by
  sorry

end activities_realign_in_10608_days_l802_802020


namespace smallest_n_with_integer_midpoint_l802_802038

theorem smallest_n_with_integer_midpoint (n : ℕ) :
  (∀ (points : Fin n → (ℤ × ℤ)),
    ∃ (i j : Fin n), i ≠ j ∧ 
    let (x_i, y_i) := points i
    let (x_j, y_j) := points j
    (x_i + x_j) % 2 = 0 ∧ (y_i + y_j) % 2 = 0) ↔ n >= 5 :=
begin
  -- Proof to be provided later
  sorry
end

end smallest_n_with_integer_midpoint_l802_802038


namespace angle_side_inequality_l802_802249

theorem angle_side_inequality {A B C : Triangle} (a b : ℝ) (angle_A angle_B : ℝ) 
    (h1 : A.opposite_side = a) (h2 : B.opposite_side = b)
    (h3 : angle_A = A.angle) (h4 : angle_B = B.angle) :
    angle_A > angle_B → a > b :=
by
  sorry

end angle_side_inequality_l802_802249


namespace positive_integer_solutions_l802_802928

theorem positive_integer_solutions (n x y z : ℕ) (h1 : n > 1) (h2 : n^z < 2001) (h3 : n^x + n^y = n^z) :
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 10 ∧ x = k ∧ y = k ∧ z = k + 1) :=
sorry

end positive_integer_solutions_l802_802928


namespace flour_to_water_ratio_change_l802_802778

-- Define ratios and new recipe conditions
def original_flour_to_water_to_sugar_ratio := (7, 2, 1)
def new_sugar_amount := 4
def new_water_amount := 2

-- Define the halved flour to sugar ratio
def halved_flour_to_sugar_ratio := (7, 2)

-- Prove the change in the ratio of flour to water
theorem flour_to_water_ratio_change :
  let new_flour_amount := (halved_flour_to_sugar_ratio.1 / halved_flour_to_sugar_ratio.2) * new_sugar_amount in
  let new_flour_to_water_ratio := (new_flour_amount, new_water_amount) in
  let original_flour_to_water_ratio := (7, 2) in
  (new_flour_to_water_ratio.1 / new_flour_to_water_ratio.2) - (original_flour_to_water_ratio.1 / original_flour_to_water_ratio.2) = -1 :=
  sorry

end flour_to_water_ratio_change_l802_802778


namespace min_reciprocal_sum_l802_802593

theorem min_reciprocal_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (hS : S 2019 = 4038) 
  (h_seq : ∀ n, S n = (n * (a 1 + a n)) / 2) :
  ∃ m, m = 4 ∧ (∀ i, i = 9 → ∀ j, j = 2011 → 
  a i + a j = 4 ∧ m = min (1 / a i + 9 / a j) 4) :=
by sorry

end min_reciprocal_sum_l802_802593


namespace x_squared_eq_r_floor_x_has_2_or_3_solutions_l802_802738

theorem x_squared_eq_r_floor_x_has_2_or_3_solutions (r : ℝ) (hr : r > 2) : 
  ∃! (s : Finset ℝ), s.card = 2 ∨ s.card = 3 ∧ ∀ x ∈ s, x^2 = r * (⌊x⌋) :=
by
  sorry

end x_squared_eq_r_floor_x_has_2_or_3_solutions_l802_802738


namespace find_ratio_l802_802697

variables {A B C D P : Type} [LinearOrder A]
variables (s r a b : ℝ) (PA PB PC PD : ℝ)

-- Definitions based on the problem statement
def right_trapezoid (AB CD AD : ℕ) : Prop :=
  AD > 0 ∧ AB > CD ∧ ∃ PC PD, PC * PD = 4 ∧ ∀ t : Type, 
  (t = "PCD" -> t.area = 2) ∧ (t = "PAD" -> t.area = 4) ∧ 
  (t = "PAB" -> t.area = 8) ∧ (t = "PBC" -> t.area = 6)

theorem find_ratio (h : right_trapezoid r 2 1) :
  r / 2 = 4 :=
sorry

end find_ratio_l802_802697


namespace max_modulus_z_l802_802705

variable (a b : ℝ)

def z : ℂ := a + b * complex.i
def abs_z : ℝ := complex.abs z

theorem max_modulus_z (
  h1 : abs_z = 2
) : ∃ a b : ℝ, |(z - 1) * (z + 1)^2| = 9 := sorry

end max_modulus_z_l802_802705


namespace number_of_special_permutations_l802_802199

theorem number_of_special_permutations :
  let P := {a : Fin 15 → ℕ // (∀ i j : Fin 7, i < j → a i > a j) ∧
                           (∀ i j : Fin 8, i < j → a (Fin.mk (i + 7) sorry) < a (Fin.mk (j + 7) sorry)) ∧
                           Set.range a = {1, 2, 3, ..., 15}.toFinset} in
  P.card = 3003 := by
  -- Proof goes here
  sorry

end number_of_special_permutations_l802_802199


namespace smallest_fraction_numerator_l802_802873

theorem smallest_fraction_numerator (a b : ℕ) (h1 : 10 ≤ a) (h2 : a ≤ 99) (h3 : 10 ≤ b) (h4 : b ≤ 99) (h5 : 9 * a > 4 * b) (smallest : ∀ c d, 10 ≤ c → c ≤ 99 → 10 ≤ d → d ≤ 99 → 9 * c > 4 * d → (a * d ≤ b * c) → a * d ≤ 41 * 92) :
  a = 41 :=
by
  sorry

end smallest_fraction_numerator_l802_802873


namespace number_of_boys_at_reunion_l802_802163

theorem number_of_boys_at_reunion (n : ℕ) (h : n * (n - 1) / 2 = 66) : n = 12 :=
sorry

end number_of_boys_at_reunion_l802_802163


namespace cos_angle_APN_eq_zero_l802_802658

section
variables {A B C D M N P : Type} [metric_space (A × B × C × D)] 
variables [midpoint M B C] [midpoint N C D] [midpoint P A B] [square ABCD]

theorem cos_angle_APN_eq_zero (A B C D M N P : A × B × C × D) 
(mid_m : midpoint B C = M) (mid_n : midpoint C D = N) (mid_p : midpoint A B = P) (square : square ABCD) : 
  real.cos (angle A P N) = 0 :=
sorry
end

end cos_angle_APN_eq_zero_l802_802658


namespace infinite_rational_points_within_circle_l802_802907

theorem infinite_rational_points_within_circle :
  ∃ infinitely_many (p : ℚ × ℚ), (0 < p.1) ∧ (0 < p.2) ∧ (p.1 ^ 2 + p.2 ^ 2 ≤ 25) :=
sorry

end infinite_rational_points_within_circle_l802_802907


namespace husband_catches_yolanda_time_in_minutes_l802_802343

noncomputable def yolanda_speed_mph : ℝ := 20
noncomputable def husband_speed_mph : ℝ := 40
noncomputable def head_start_minutes : ℝ := 15

theorem husband_catches_yolanda_time_in_minutes :
  ∀ (yolanda_speed_mph husband_speed_mph head_start_minutes : ℝ),
    yolanda_speed_mph = 20 →
    husband_speed_mph = 40 →
    head_start_minutes = 15 →
    (head_start_minutes * (yolanda_speed_mph / 60)) / ((husband_speed_mph - yolanda_speed_mph) / 60) = 15 :=
by
  intros yolanda_speed_mph husband_speed_mph head_start_minutes h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  norm_num
  sorry -- proof to be completed

end husband_catches_yolanda_time_in_minutes_l802_802343


namespace divisible_by_bn_l802_802775

variables {u v a b : ℤ} {n : ℕ}

theorem divisible_by_bn 
  (h1 : ∀ x : ℤ, x^2 + a*x + b = 0 → x = u ∨ x = v)
  (h2 : a^2 % b = 0) 
  (h3 : ∀ m : ℕ, m = 2 * n) : 
  ∀ n : ℕ, (u^m + v^m) % (b^n) = 0 := 
  sorry

end divisible_by_bn_l802_802775


namespace cards_difference_product_divisibility_l802_802456

theorem cards_difference_product_divisibility :
  (∃ pairs : list (ℕ × ℕ), 
    (∀ p ∈ pairs, 1 ≤ p.1 ∧ p.1 ≤ 50 ∧ 1 ≤ p.2 ∧ p.2 ≤ 50 ∧ 
      (p.1 - p.2 = 11 ∨ p.2 - p.1 = 11) ∧ 
      (p.1 * p.2) % 5 = 0) ∧ 
    pairs.length = 15) := 
sorry

end cards_difference_product_divisibility_l802_802456


namespace necessary_but_not_sufficient_l802_802967

   /-- Condition for a sequence to be a geometric sequence with a common ratio of 2 -/
   def geometric_sequence (a : ℕ → ℝ) : Prop :=
     ∀ n ≥ 2, a(n) = 2 * a(n-1)

   /-- Define geometric sequence with common ratio 2 -/
   def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
     ∃ c, ∀ n, a(n) = c * (r ^ n)

   /-- Main theorem -/
   theorem necessary_but_not_sufficient (a : ℕ → ℝ) :
     (geometric_sequence a) ↔ ¬ (is_geometric_sequence a 2) := 
     sorry
   
end necessary_but_not_sufficient_l802_802967


namespace remaining_food_for_children_l802_802857

theorem remaining_food_for_children :
  ∀ (A C : ℝ) (totalAdults totalChildren : ℕ) (mealForAdults mealForChildren : ℕ),
    totalAdults = 55 →
    totalChildren = 70 →
    mealForAdults = 70 →
    mealForChildren = 90 →
    ∃ remainingChildren : ℕ,
      (14 * A) = 18 * C →
      remainingChildren = mealForChildren - 18 →
      remainingChildren = 72 :=
by
  intros A C totalAdults totalChildren mealForAdults mealForChildren h1 h2 h3 h4 h5
  use 72
  sorry

end remaining_food_for_children_l802_802857


namespace market_supply_function_tax_revenues_collected_optimal_tax_rate_tax_revenues_specified_l802_802385

noncomputable def Q_d : ℝ → ℝ := λ P, 688 - 4 * P
noncomputable def P_s := 64
noncomputable def tax := 90

theorem market_supply_function :
  ∃ (c d : ℝ), (λ P, c + d * P) = (λ P, -312 + 6 * P) :=
by {
  let Q_s : ℝ → ℝ := λ P, -312 + 6 * P,
  use [-312, 6],
  sorry
}

theorem tax_revenues_collected :
  let Q_s := -312 + 6 * P_s in Q_s * 90 = 6480 :=
by {
  let Q_s := -312 + 6 * P_s,
  have : Q_s = 72, by { unfold Q_s, simp },
  show 72 * tax = 6480, by norm_num,
  sorry
}

theorem optimal_tax_rate :
  let t_opt := 54 in t_opt = 54 :=
by {
  let t_opt := 54,
  show t_opt = 54, by norm_num,
  sorry
}

theorem tax_revenues_specified :
  let Q_s := λ t, 432 - 4 * t in
  let t_opt := 54 in
  Q_s t_opt * t_opt = 10800 :=
by {
  let t_opt := 54,
  let Q := 432 - 4 * t_opt,
  have : Q = 216, by { unfold Q, simp },
  show 216 * t_opt = 10800, by norm_num,
  sorry
}

end market_supply_function_tax_revenues_collected_optimal_tax_rate_tax_revenues_specified_l802_802385


namespace derivative_at_pi_over_3_l802_802117

def f (x : ℝ) : ℝ := cos x + (real.sqrt 3) * sin x

theorem derivative_at_pi_over_3 : deriv f (π / 3) = 0 :=
by
  sorry

end derivative_at_pi_over_3_l802_802117


namespace problem1_problem2_l802_802948

variables {a b : ℝ}

-- Given conditions
def condition1 : a + b = 2 := sorry
def condition2 : a * b = -1 := sorry

-- Proof for a^2 + b^2 = 6
theorem problem1 (h1 : a + b = 2) (h2 : a * b = -1) : a^2 + b^2 = 6 :=
by sorry

-- Proof for (a - b)^2 = 8
theorem problem2 (h1 : a + b = 2) (h2 : a * b = -1) : (a - b)^2 = 8 :=
by sorry

end problem1_problem2_l802_802948


namespace opposite_of_x_abs_of_x_recip_of_x_l802_802285

noncomputable def x : ℝ := 1 - Real.sqrt 2

theorem opposite_of_x : -x = Real.sqrt 2 - 1 := 
by sorry

theorem abs_of_x : |x| = Real.sqrt 2 - 1 :=
by sorry

theorem recip_of_x : 1/x = -1 - Real.sqrt 2 :=
by sorry

end opposite_of_x_abs_of_x_recip_of_x_l802_802285


namespace total_male_students_is_490_l802_802864

-- Define the number of students in the school
def total_students : ℕ := 1000

-- Define the sample size
def sample_size : ℕ := 100

-- Define the number of female students in the sample
def females_in_sample : ℕ := 51

-- Calculate the number of male students in the sample
def males_in_sample : ℕ := sample_size - females_in_sample

-- Calculate the percentage of male students in the sample
def male_percentage : ℚ := males_in_sample / sample_size

-- Calculate the number of male students in the school
def males_in_school : ℕ := total_students * male_percentage

-- Theorem statement to prove
theorem total_male_students_is_490 : males_in_school = 490 := by
  sorry

end total_male_students_is_490_l802_802864


namespace smallest_n_for_multiples_of_7_l802_802753

theorem smallest_n_for_multiples_of_7 (x y : ℤ) (h1 : x ≡ 4 [ZMOD 7]) (h2 : y ≡ 5 [ZMOD 7]) :
  ∃ n : ℕ, 0 < n ∧ (x^2 + x * y + y^2 + n ≡ 0 [ZMOD 7]) ∧ ∀ m : ℕ, 0 < m ∧ (x^2 + x * y + y^2 + m ≡ 0 [ZMOD 7]) → n ≤ m :=
by
  sorry

end smallest_n_for_multiples_of_7_l802_802753


namespace maximum_area_l802_802312

def f (x : ℝ) : ℝ := x^2 + 16

def tangent_line_at (x₀ : ℝ) : ℝ → ℝ := 
  λ x, (f x₀) + (2 * x₀) * (x - x₀)

def area (x₀ : ℝ) : ℝ :=
  let y1 := (f x₀) + (2 * x₀) * (-3 - x₀)
  let y2 := (f x₀) + (2 * x₀) * (1 - x₀)
  0.5 * (y1 + y2) * (1 - (-3))

theorem maximum_area : 
  ∃ x₀ ∈ Set.Icc (-3 : ℝ) 1, area x₀ = 68 :=
sorry

end maximum_area_l802_802312


namespace total_candies_darrel_took_l802_802797

theorem total_candies_darrel_took (r b x : ℕ) (h1 : r = 3 * b)
  (h2 : r - x = 4 * (b - x))
  (h3 : r - x - 12 = 5 * (b - x - 12)) : 2 * x = 48 := sorry

end total_candies_darrel_took_l802_802797


namespace makarla_meeting_percentage_l802_802232

theorem makarla_meeting_percentage :
  ∀ (workday_duration : ℕ) (first_meeting_duration : ℕ) (second_meeting_factor : ℕ),
    workday_duration = 8 * 60 →
    first_meeting_duration = 60 →
    second_meeting_factor = 2 →
    let total_meeting_time := first_meeting_duration + second_meeting_factor * first_meeting_duration in
    let percentage_of_workday := (total_meeting_time * 100) / workday_duration in
    percentage_of_workday = 37.5 :=
by
  sorry

end makarla_meeting_percentage_l802_802232


namespace gcd_polynomial_multiple_of_504_l802_802580

theorem gcd_polynomial_multiple_of_504 (b : ℤ) (hb : 504 ∣ b) : Int.gcd(4 * b ^ 3 + 2 * b ^ 2 + 5 * b + 63, b) = 63 := by
  sorry

end gcd_polynomial_multiple_of_504_l802_802580


namespace nicky_catchup_time_l802_802718

-- Definitions related to the problem
def head_start : ℕ := 12
def speed_cristina : ℕ := 5
def speed_nicky : ℕ := 3
def time_to_catchup : ℕ := 36
def nicky_runtime_before_catchup : ℕ := head_start + time_to_catchup

-- Theorem to prove the correct runtime for Nicky before Cristina catches up
theorem nicky_catchup_time : nicky_runtime_before_catchup = 48 := by
  sorry

end nicky_catchup_time_l802_802718


namespace maximize_profit_at_100_l802_802853

noncomputable def C : ℕ → ℝ
| x => if x < 80 then (1 / 3) * x^2 + 10 * x
       else 51 * x + 10000 / x - 1450

noncomputable def annual_profit (x : ℕ) : ℝ :=
  if 0 < x ∧ x < 80 then - (1 / 3) * x^2 + 40 * x - 250
  else 1200 - (x + 10000 / x)

theorem maximize_profit_at_100 (x : ℕ) :
  annual_profit x ≤ annual_profit 100 :=
sorry

#eval annual_profit 100  -- To verify the calculation

end maximize_profit_at_100_l802_802853


namespace card_pairs_count_l802_802478

theorem card_pairs_count :
  let cards := {n : ℕ | 1 ≤ n ∧ n ≤ 50}
  let valid_pairs := {(a, b) ∈ cards × cards | 
    a ≠ b ∧ (a - b = 11 ∨ b - a = 11) ∧ (a * b) % 5 = 0}
  in valid_pairs.count = 15 :=
by
  sorry

end card_pairs_count_l802_802478


namespace solution_y_amount_l802_802354

theorem solution_y_amount :
  ∀ (y : ℝ) (volume_x volume_y : ℝ),
    volume_x = 200 ∧
    volume_y = y ∧
    10 / 100 * volume_x = 20 ∧
    30 / 100 * volume_y = 0.3 * y ∧
    (20 + 0.3 * y) / (volume_x + y) = 0.25 →
    y = 600 :=
by 
  intros y volume_x volume_y
  intros H
  sorry

end solution_y_amount_l802_802354


namespace g_h_of_2_eq_2340_l802_802630

def g (x : ℝ) : ℝ := 2 * x^2 + 5 * x - 3
def h (x : ℝ) : ℝ := 4 * x^3 + 1

theorem g_h_of_2_eq_2340 : g (h 2) = 2340 := 
  sorry

end g_h_of_2_eq_2340_l802_802630


namespace max_remainder_l802_802386

theorem max_remainder : ∃ n r, 2013 ≤ n ∧ n ≤ 2156 ∧ (n % 5 = r) ∧ (n % 11 = r) ∧ (n % 13 = r) ∧ (r ≤ 4) ∧ (∀ m, 2013 ≤ m ∧ m ≤ 2156 ∧ (m % 5 = r) ∧ (m % 11 = r) ∧ (m % 13 = r) ∧ (m ≤ n) ∧ (r ≤ 4) → r ≤ 4) := sorry

end max_remainder_l802_802386


namespace candy_problem_l802_802394

theorem candy_problem (a : ℕ) (h₁ : a % 10 = 6) (h₂ : a % 15 = 11) (h₃ : 200 ≤ a) (h₄ : a ≤ 250) :
  a = 206 ∨ a = 236 :=
sorry

end candy_problem_l802_802394


namespace range_of_f_l802_802519

noncomputable def f (x : ℝ) : ℝ := (x + 2)/(x^2 - 3*x + 5)

theorem range_of_f :
  set.range f = set.Icc ((7 - 2 * real.sqrt 15) / 11) ((7 + 2 * real.sqrt 15) / 11) :=
sorry

end range_of_f_l802_802519


namespace total_sample_any_candy_42_percent_l802_802170

-- Define percentages as rational numbers to avoid dealing with decimals directly
def percent_of_caught_A : ℚ := 12 / 100
def percent_of_not_caught_A : ℚ := 7 / 100
def percent_of_caught_B : ℚ := 5 / 100
def percent_of_not_caught_B : ℚ := 6 / 100
def percent_of_caught_C : ℚ := 9 / 100
def percent_of_not_caught_C : ℚ := 3 / 100

-- Sum up the percentages for those caught and not caught for each type of candy
def total_percent_A : ℚ := percent_of_caught_A + percent_of_not_caught_A
def total_percent_B : ℚ := percent_of_caught_B + percent_of_not_caught_B
def total_percent_C : ℚ := percent_of_caught_C + percent_of_not_caught_C

-- Sum of the total percentages for all types
def total_percent_sample_any_candy : ℚ := total_percent_A + total_percent_B + total_percent_C

theorem total_sample_any_candy_42_percent :
  total_percent_sample_any_candy = 42 / 100 :=
by
  sorry

end total_sample_any_candy_42_percent_l802_802170


namespace decimal_to_fraction_l802_802504

theorem decimal_to_fraction :
  (\(x\), \(y\), (\(x, y\)) = (3, 110)) → 0.3\overline{45} = \(\frac{83}{110}\) := λ ⟨3, 110, 3, 110⟩, sorry

end decimal_to_fraction_l802_802504


namespace range_of_a_l802_802605

open Real

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := abs (exp x + a / exp x)

theorem range_of_a (a : ℝ) (h : ∀ x y ∈ Icc (0 : ℝ) (1 : ℝ), x < y → f x a ≤ f y a) : -1 ≤ a ∧ a ≤ 1 := 
sorry

end range_of_a_l802_802605


namespace collinear_N_F_Q_l802_802115

noncomputable def ellipse_equation (x y : ℝ) (a : ℝ) : Prop :=
  x ^ 2 / a ^ 2 + y ^ 2 / (7 - a ^ 2) = 1

theorem collinear_N_F_Q (a : ℝ) (h_pos : a > 0) (h_foci : 2 * sqrt ((max a (7 - a))^2 - min a (7 - a)^2) = 2)
  (P Q N F : ℝ × ℝ) (l : ℝ × ℝ → Prop)
  (h1 : ellipse_equation P.1 P.2 a)
  (h2 : ellipse_equation Q.1 Q.2 a)
  (h3 : l = λ R, R.2 = (Q.2 - P.2)/(Q.1 - P.1) * (R.1 - P.1) + P.2)
  (h4 : P = (4, 0))
  (h5 : F = (1, 0))
  (h6 : N.1 = P.1 ∧ N.2 = -P.2)
  (h7 : Q.1 = N.1)
  : collinear N F Q :=
  sorry

end collinear_N_F_Q_l802_802115


namespace arithmetic_sequence_common_difference_l802_802228

theorem arithmetic_sequence_common_difference :
  ∀ (a d : ℕ), (∑ i in finset.range 13, (a + i * d)) = 104 → (a + 5 * d = 5) → d = 3 :=
by 
  -- This part of the proof is left as an exercise. Fill in the proof appropriately.
  sorry

end arithmetic_sequence_common_difference_l802_802228


namespace find_x0_l802_802626

theorem find_x0 (f : ℝ → ℝ) (x0 : ℝ) (h1 : f = λ x, x^3) (h2 : deriv f x0 = 3) : x0 = 1 ∨ x0 = -1 :=
by
  sorry

end find_x0_l802_802626


namespace Zoey_finishes_18th_book_on_Thursday_l802_802348

theorem Zoey_finishes_18th_book_on_Thursday : 
  (∃ d : ℕ, 
    d = 171 ∧ 
    (∀ n : ℕ, n = 18 → 
      (n * (n + 1)) / 2 = d) ∧ 
    d % 7 = 3) → 
  "Thursday" :=
begin
  sorry
end

end Zoey_finishes_18th_book_on_Thursday_l802_802348


namespace check_true_propositions_l802_802600

open Set

theorem check_true_propositions : 
  ∀ (Prop1 Prop2 Prop3 : Prop),
    (Prop1 ↔ (∀ x : ℝ, x^2 > 0)) →
    (Prop2 ↔ ∃ x : ℝ, x^2 ≤ x) →
    (Prop3 ↔ ∀ (M N : Set ℝ) (x : ℝ), x ∈ (M ∩ N) → x ∈ M ∧ x ∈ N) →
    (¬Prop1 ∧ Prop2 ∧ Prop3) →
    (2 = 2) := sorry

end check_true_propositions_l802_802600


namespace sequence_term_1000_l802_802648

open Nat

theorem sequence_term_1000 :
  (∃ b : ℕ → ℤ,
    b 1 = 3010 ∧
    b 2 = 3011 ∧
    (∀ n, 1 ≤ n → b n + b (n + 1) + b (n + 2) = n + 4) ∧
    b 1000 = 3343) :=
sorry

end sequence_term_1000_l802_802648


namespace prove_O_lies_on_OB_OD_l802_802691

open EuclideanGeometry

variables (A B C D O B1 D1 OB OD : Point)
variables (h_cyclic : CyclicQuad A B C D)
variables (h_diff_lengths : (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ D) ∧ (D ≠ A))
variables (h_circum : IsCircumcenter O A B C D)
variables (h_bisectors : IsInternalAngleBisector (Angle B A C) B1 ∧ IsInternalAngleBisector (Angle D A C) D1)
variables (h_tangent_bo : IsTangentAt OB (CirclePassingThrough B TangentAt D1 A C) AC)
variables (h_tangent_do : IsTangentAt OD (CirclePassingThrough D TangentAt B1 A C) AC)
variables (h_parallel: Parallel (Segment B D1) (Segment D B1))

theorem prove_O_lies_on_OB_OD : LiesOn O (Line OB OD) :=
sorry

end prove_O_lies_on_OB_OD_l802_802691


namespace sqrt_a2015_eq_2016_l802_802156

noncomputable def seq (n : ℕ) : ℕ :=
match n with
| 0     => 4
| (n+1) => seq n + 2 * Int.sqrt (seq n) + 1

theorem sqrt_a2015_eq_2016 :
  Int.sqrt (seq 2015) = 2016 :=
sorry

end sqrt_a2015_eq_2016_l802_802156


namespace tom_tickets_l802_802892

theorem tom_tickets :
  let tickets_whack_a_mole := 32
  let tickets_skee_ball := 25
  let tickets_spent_on_hat := 7
  let total_tickets := tickets_whack_a_mole + tickets_skee_ball
  let tickets_left := total_tickets - tickets_spent_on_hat
  tickets_left = 50 :=
by
  sorry

end tom_tickets_l802_802892


namespace possible_values_t_l802_802212

def is_singleton {α : Type*} (s : set α) : Prop :=
  ∃ x, s = {x}

theorem possible_values_t (x y t : ℝ) (A B : set (ℝ × ℝ)) 
  (hA : A = {p | p.1^2 - p.2^2 = 1}) 
  (hB : B = {p | p.2 = t * (p.1 - 1) + 2}) 
  (h_singleton : is_singleton (A ∩ B)) : 
  ∃ t₁ t₂ t₃ : ℝ, ∀ t', t' = t₁ ∨ t' = t₂ ∨ t' = t₃ :=
sorry

end possible_values_t_l802_802212


namespace infinite_sum_l802_802611

noncomputable def sequence (n : ℕ) : ℕ :=
  let α := (√2 + 1) ^ n
  let β := (1 / 2) ^ n
  Int.floor (α + β)

axiom recurrence_relation (n : ℕ) : sequence (n + 1) = 2 * sequence n + sequence (n - 1)
axiom initial_condition₀ : sequence 0 = 2
axiom initial_condition₁ : sequence 1 = 2

theorem infinite_sum :
  ∑' n, (1 : ℝ) / ((sequence (n - 1)) * (sequence (n + 1))) = 1 / 8 :=
sorry

end infinite_sum_l802_802611


namespace calculate_difference_of_reciprocal_squares_l802_802217

theorem calculate_difference_of_reciprocal_squares
  (x1 x2 : ℝ)
  (h1 : x1 * x1 * sqrt 14 - x1 * sqrt 116 + sqrt 56 = 0)
  (h2 : x2 * x2 * sqrt 14 - x2 * sqrt 116 + sqrt 56 = 0) :
  abs ((1 / (x1 * x1)) - (1 / (x2 * x2))) = sqrt 29 / 14 :=
sorry

end calculate_difference_of_reciprocal_squares_l802_802217


namespace violates_properties_l802_802339

-- Definitions from conditions
variables {a b c m : ℝ}

-- Conclusion to prove
theorem violates_properties :
  (∀ c : ℝ, ac = bc → (c ≠ 0 → a = b)) ∧ (c = 0 → ac = bc) → False :=
sorry

end violates_properties_l802_802339


namespace smallest_fraction_numerator_l802_802872

theorem smallest_fraction_numerator (a b : ℕ) (h1 : 10 ≤ a) (h2 : a ≤ 99) (h3 : 10 ≤ b) (h4 : b ≤ 99) (h5 : 9 * a > 4 * b) (smallest : ∀ c d, 10 ≤ c → c ≤ 99 → 10 ≤ d → d ≤ 99 → 9 * c > 4 * d → (a * d ≤ b * c) → a * d ≤ 41 * 92) :
  a = 41 :=
by
  sorry

end smallest_fraction_numerator_l802_802872


namespace percent_calculation_l802_802333

theorem percent_calculation (y : ℝ) : (0.3 * (0.6 * y) = 0.18 * y) → (0.18 * y / y) * 100 = 18 := by
  sorry

end percent_calculation_l802_802333


namespace odd_pair_exists_k_l802_802135

theorem odd_pair_exists_k (a b : ℕ) (h1 : a % 2 = 1) (h2 : b % 2 = 1) : 
  ∃ k : ℕ, (2^2018 ∣ b^k - a^2) ∨ (2^2018 ∣ a^k - b^2) := 
sorry

end odd_pair_exists_k_l802_802135


namespace altitude_of_triangle_on_diagonal_l802_802569

theorem altitude_of_triangle_on_diagonal 
  (l w : ℝ) 
  (h : ℝ) 
  (h_area : l * w = 1/2 * sqrt (l^2 + w^2) * h) : 
  h = 2 * l * w / sqrt (l^2 + w^2) :=
sorry

end altitude_of_triangle_on_diagonal_l802_802569


namespace sum_common_divisors_60_18_l802_802531

def divisors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ m => n % m = 0)

noncomputable def sum (l : List ℕ) : ℕ :=
  l.foldr (· + ·) 0

theorem sum_common_divisors_60_18 :
  sum (List.filter (λ d => d ∈ divisors 18) (divisors 60)) = 12 :=
by
  sorry

end sum_common_divisors_60_18_l802_802531


namespace divide_9246_koronas_among_4_people_l802_802042

-- Define the conditions
def amounts_are_divided_properly (A B C D : ℕ) (K : ℕ) : Prop :=
  (∀ K, B = 3 / 2 * A) ∧ (C = 6 / 5 * B) ∧ (D = 4 / 3 * C) ∧ (A + B + C + D = 9246 * K)

-- Main theorem statement
theorem divide_9246_koronas_among_4_people : ∃ (A B C D K : ℕ),  amounts_are_divided_properly A B C D K :=
  sorry

end divide_9246_koronas_among_4_people_l802_802042


namespace find_a_l802_802554

theorem find_a 
  (a b : ℝ)
  (h1 : ∀ x : ℝ, (8 * x^3 + 7 * a * x^2 + 6 * b * x + 2 * a = 0) → x > 0)
  (h2 : ∑ r in {(x : ℝ) | 8 * x^3 + 7 * a * x^2 + 6 * b * x + 2 * a = 0}, real.logb 2 r = 5) :
  a = -128 :=
sorry

end find_a_l802_802554


namespace katherine_has_5_bananas_l802_802687

theorem katherine_has_5_bananas
  (apples : ℕ) (pears : ℕ) (bananas : ℕ) (total_fruits : ℕ)
  (h1 : apples = 4)
  (h2 : pears = 3 * apples)
  (h3 : total_fruits = apples + pears + bananas)
  (h4 : total_fruits = 21) :
  bananas = 5 :=
by
  sorry

end katherine_has_5_bananas_l802_802687


namespace total_office_workers_l802_802409

-- Defining given conditions as variables
variables (w fo more_males mnw : ℕ)
-- Conditions provided in the problem
def conditions := 
  w = 1518 ∧ 
  fo = 536 ∧
  more_males = 525 ∧
  mnw = 1257

-- The goal to prove
def goal := ∃ (to : ℕ), (to = fo + (w + more_males - mnw))

-- The Lean statement asserting that, given the conditions, the goal holds
theorem total_office_workers (h : conditions) : goal :=
  sorry

end total_office_workers_l802_802409


namespace find_x_l802_802624

theorem find_x (log2 log3 : ℝ) (h1 : log2 = 0.3010) (h2 : log3 = 0.4771) :
  ∃ x : ℝ, 3^(x + 2) = 270 ∧ x = 3.09 :=
by
  exists 3.09
  split
  sorry

end find_x_l802_802624


namespace sequence_eq_n_l802_802085

noncomputable def S (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ i in Finset.range (n + 1), a i

theorem sequence_eq_n (a : ℕ → ℕ) (h : ∀ n, 2 * S a n = a n * a n + a n) :
  a 1 = 1 →
  a 2 = 2 →
  a 3 = 3 →
  a 4 = 4 →
  ∀ n, a n = n :=
by
  sorry

end sequence_eq_n_l802_802085


namespace fifty_new_edges_l802_802173

-- Definition: Tree with 100 vertices, each vertex having exactly one outgoing road.
def initial_tree (V : Type) [Fintype V] [DecidableEq V] : SimpleGraph V :=
{ Edge := λ x y, ∃ (p : Path x y), p.edges.length = 1,
  symm := by finish,
  loopless := by finish }

-- Problem statement: Proving the existence of 50 new edges under given conditions.
theorem fifty_new_edges (V : Type) [Fintype V] [DecidableEq V] (G : SimpleGraph V)
    (h_tree : G.IsTree) (h_vertices : Fintype.card V = 100)
    (h_leaves : ∀ v, G.degree v = 1) :
  ∃ (new_edges : Finset (Sym2 V)), new_edges.card = 50 ∧
    ∀ e ∈ G.edgeFinset ∪ new_edges, (G ⧸ e).IsConnected :=
sorry

end fifty_new_edges_l802_802173


namespace cos_double_angle_l802_802560

theorem cos_double_angle (θ : ℝ) (h : tan θ = 1 / 2): 
    cos (2 * θ) = 3 / 5 := 
by 
  sorry

end cos_double_angle_l802_802560


namespace min_distance_parabola_to_line_l802_802993

-- Definitions and conditions
def parabola (y x : ℝ) : Prop := y^2 = 8 * x
def line (y x : ℝ) : Prop := y = x + 3

-- Define the distance from point P to the line
def distance (x y : ℝ) : ℝ := (|x - y + 3|) / (sqrt 2)

-- Statement to prove
theorem min_distance_parabola_to_line :
  ∃ (x y : ℝ), parabola y x ∧ distance x y = sqrt 2 / 2 :=
sorry

end min_distance_parabola_to_line_l802_802993


namespace BC_length_eq_l802_802187

noncomputable section

variables {A B C D O : Type*} [metric_space A]
variables (a b : ℝ)
variables (AB AD BC CD : A → ℝ)
variables (ABmid : is_midpoint A B O)
variables (BC_CD_tangent : tangent BC CD AD O)

-- Given a quadrilateral ABCD, where |AB| = a and |AD| = b and sides BC, CD, and AD
-- are tangent to a circle centered at the midpoint of AB,
-- prove that the length of BC is a^2 / 4b.
theorem BC_length_eq :
  BC = (a^2) / (4 * b) :=
sorry

end BC_length_eq_l802_802187


namespace determine_l_correct_l802_802909

noncomputable def determine_l (x y z l : ℝ) : Prop :=
  (9 / (x + y + 1) = l / (x + z - 1)) ∧ (l / (x + z - 1) = 13 / (z - y + 2))

theorem determine_l_correct (x y z : ℝ) : ∃ l : ℝ, determine_l x y z l ∧ l = 22 :=
by {
  use 22,
  split,
  sorry,
  refl,
}

end determine_l_correct_l802_802909


namespace asymptote_of_hyperbola_l802_802060

theorem asymptote_of_hyperbola :
  (∀ x y : ℝ, (y^2 / 4 - x^2 / 5 = 1) → y = (2 / sqrt(5)) * x ∨ y = -(2 / sqrt(5)) * x) :=
sorry

end asymptote_of_hyperbola_l802_802060


namespace impossible_partition_l802_802667

theorem impossible_partition :
  ¬ ∃ (A B C : set ℤ), 
    (∀ n : ℤ, (n ∈ A ∧ (n - 50) ∈ B ∧ (n + 1987) ∈ C) ∨
               (n ∈ A ∧ (n - 50) ∈ C ∧ (n + 1987) ∈ B) ∨
               (n ∈ B ∧ (n - 50) ∈ A ∧ (n + 1987) ∈ C) ∨
               (n ∈ B ∧ (n - 50) ∈ C ∧ (n + 1987) ∈ A) ∨
               (n ∈ C ∧ (n - 50) ∈ A ∧ (n + 1987) ∈ B) ∨
               (n ∈ C ∧ (n - 50) ∈ B ∧ (n + 1987) ∈ A)) := sorry

end impossible_partition_l802_802667


namespace sum_common_divisors_60_18_l802_802527

def divisors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ m => n % m = 0)

noncomputable def sum (l : List ℕ) : ℕ :=
  l.foldr (· + ·) 0

theorem sum_common_divisors_60_18 :
  sum (List.filter (λ d => d ∈ divisors 18) (divisors 60)) = 12 :=
by
  sorry

end sum_common_divisors_60_18_l802_802527


namespace triangle_division_conditions_l802_802911

theorem triangle_division_conditions :
  ∃ S : set ℕ, S = {4, 7, 19} ∧ 
    (∀ n ∈ S, ∀ (T : Type) [linear_order T], ∃ (f : ℕ → T), 
       (∀ x y z : ℕ, x ∈ f '' S ∧ y ∈ f '' S ∧ z ∈ f '' S → 
        (set.card (({x} ∪ {y} ∪ {z}) : set ℕ) = n) ∧ 
        (∃ k : ℕ, (∀ v ∈ {x, y, z}, cardinality_of_sides_at v = k)))) :=
by
  sorry

end triangle_division_conditions_l802_802911


namespace olivia_savings_l802_802336

noncomputable def compound_amount 
  (P : ℝ) -- Initial principal
  (r : ℝ) -- Annual interest rate
  (n : ℕ) -- Number of times interest is compounded per year
  (t : ℕ) -- Number of years
  : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem olivia_savings :
  compound_amount 2500 0.045 2 21 = 5077.14 :=
by
  sorry

end olivia_savings_l802_802336


namespace distance_between_points_l802_802200

noncomputable theory

-- Define the points in polar coordinates
def point_A (θ₁ : ℝ) : ℝ × ℝ := (4, θ₁)
def point_B (θ₂ : ℝ) : ℝ × ℝ := (6, θ₂)

-- Define the condition for the angles
def angle_condition (θ₁ θ₂ : ℝ) : Prop := θ₁ - θ₂ = π / 3

-- Define the function to calculate the distance
def distance_AB (θ₁ θ₂ : ℝ) : ℝ := 
  let AB_squared := 4^2 + 6^2 - 2 * 4 * 6 * Real.cos(π / 3)
  Real.sqrt AB_squared

-- The theorem statement
theorem distance_between_points (θ₁ θ₂ : ℝ) (h : angle_condition θ₁ θ₂) :
  distance_AB θ₁ θ₂ = 2 * Real.sqrt 7 :=
by
  sorry

end distance_between_points_l802_802200


namespace factors_multiple_of_300_l802_802158

theorem factors_multiple_of_300 (m : ℕ) (h : m = 2^12 * 3^15 * 5^9) : 
  ∃ n : ℕ, n = 1320 ∧ (∀ k : ℕ, (k ∣ m ∧ 300 ∣ k) ↔ k ∈ finset.range (n + 1) ∧ k = 300 * (2^a * 3^b * 5^c) 
  where 2 ≤ a ≤ 12 ∧ 1 ≤ b ≤ 15 ∧ 2 ≤ c ≤ 9) :=
begin
  sorry
end

end factors_multiple_of_300_l802_802158


namespace largest_divisor_of_square_difference_l802_802208

theorem largest_divisor_of_square_difference (m n : ℤ) (h₁ : Even m) (h₂ : Odd n) (h₃ : n < m) :
  ∃ k, (∀ a b : ℤ, Even a → Odd b → (b < a) → (a^2 - b^2) % 2 = 0) ∧ (∀ k', (k' divides 2) → (k' <= 2)) :=
sorry

end largest_divisor_of_square_difference_l802_802208


namespace number_of_integers_between_sqrt10_and_sqrt100_l802_802149

theorem number_of_integers_between_sqrt10_and_sqrt100 : 
  let a := Real.sqrt 10
  let b := Real.sqrt 100
  ∃ (n : ℕ), n = 6 ∧ (∀ x : ℕ, (x > a ∧ x < b) → (4 ≤ x ∧ x ≤ 9)) :=
by
  sorry

end number_of_integers_between_sqrt10_and_sqrt100_l802_802149


namespace sum_odd_numbers_1_to_200_eq_10000_l802_802939

/-- 
  The sum of all odd numbers from 1 to 200.
  We prove that this sum is equal to 10,000.
-/
theorem sum_odd_numbers_1_to_200_eq_10000 : 
  let n := 100 in
  let first_term := 1 in
  let last_term := 199 in
  n / 2 * (first_term + last_term) = 10000 :=
by
  sorry

end sum_odd_numbers_1_to_200_eq_10000_l802_802939


namespace base_conversion_to_zero_l802_802267

theorem base_conversion_to_zero (A B : ℕ) (hA : 0 ≤ A ∧ A < 12) (hB : 0 ≤ B ∧ B < 5) 
    (h1 : 12 * A + B = 5 * B + A) : 12 * A + B = 0 :=
by
  sorry

end base_conversion_to_zero_l802_802267


namespace decimal_to_fraction_l802_802505

theorem decimal_to_fraction :
  (\(x\), \(y\), (\(x, y\)) = (3, 110)) → 0.3\overline{45} = \(\frac{83}{110}\) := λ ⟨3, 110, 3, 110⟩, sorry

end decimal_to_fraction_l802_802505


namespace parallel_vectors_l802_802086

open Real

def vector_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem parallel_vectors :
  ∀ (m : ℝ),
    let a := (-1, 3)
    let b := (m, m - 4)
    vector_parallel a b → m = 1 :=
begin
  intro m,
  let a := (-1 : ℝ, 3 : ℝ),
  let b := (m, m - 4),
  sorry
end

end parallel_vectors_l802_802086


namespace selection_methods_correct_l802_802257

def numWaysSelectReps : Nat := 48

theorem selection_methods_correct :
  ∃ (ways : Nat), ways = numWaysSelectReps ∧
    (∃ (students : Finset Nat) (A : Nat),
        students.card = 5 ∧ 
        A ∈ students ∧
        ∀ reps: Finset (Nat × String), reps.card = 3 → 
          ("MathRep" : String) ∈ reps.val.map Prod.snd →
          (∃ mathReps : Finset Nat, 
            mathReps.card = 1 ∧
            (A ∉ mathReps ∧
             mathReps ⊆ students) ∧
             ∃ otherReps : Finset Nat,
               otherReps.card = 2 ∧
               otherReps ⊆ students ∧
               mathReps.disjoint otherReps)) :=
begin
  use numWaysSelectReps,
  split,
  { refl },
  { sorry }
end

end selection_methods_correct_l802_802257


namespace fractional_rep_of_0_point_345_l802_802499

theorem fractional_rep_of_0_point_345 : 
  let x := (0.3 + (0.45 : ℝ)) in
  (x = (83 / 110 : ℝ)) :=
by
  sorry

end fractional_rep_of_0_point_345_l802_802499


namespace smallest_integer_inequality_l802_802819

theorem smallest_integer_inequality :
  ∃ n : ℕ, (1 < n) ∧ (√n - √(n-1) < 0.001) ∧ (∀ m : ℕ, (1 < m) ∧ (√m - √(m-1) < 0.001) → n ≤ m) :=
sorry

end smallest_integer_inequality_l802_802819


namespace first_term_exceeds_10000_l802_802278

theorem first_term_exceeds_10000 :
  ∃ n, (∃ a : ℕ → ℕ, (a 1 = 1 ∧ ∀ n ≥ 1, a (n + 1) = finset.sum (finset.range n) (λ i, a (i + 1) ^ 2)) ∧ a n > 10000 ∧ a n = 2896806) :=
sorry

end first_term_exceeds_10000_l802_802278


namespace evaluate_expression_l802_802914

theorem evaluate_expression : 4 * Real.cos (Real.pi / 3 - Real.pi / 9) - Real.tan (Real.pi / 9) = Real.sqrt 3 := by
  have h1 : Real.cos (Real.pi / 3 - Real.pi / 9) = Real.sin (Real.pi / 9), from by sorry
  have h2 : Real.tan (Real.pi / 9) = Real.sin (Real.pi / 9) / Real.cos (Real.pi / 9), from by sorry
  have h3 : Real.sin (2 * Real.pi / 9) = 2 * Real.sin (Real.pi / 9) * Real.cos (Real.pi / 9), from by sorry
  have h4 : Real.sin (Real.pi / 6 + Real.pi / 18) = Real.sin (Real.pi / 6) * Real.cos (Real.pi / 18) + Real.cos (Real.pi / 6) * Real.sin (Real.pi / 18), from by sorry
  have h5 : Real.cos (Real.pi / 6 + Real.pi / 18) = Real.cos (Real.pi / 6) * Real.cos (Real.pi / 18) - Real.sin (Real.pi / 6) * Real.sin (Real.pi / 18), from by sorry
  sorry

end evaluate_expression_l802_802914


namespace product_of_solutions_l802_802935

theorem product_of_solutions (x : ℝ) (h : |x| = 3 * (|x| - 2)) : (subs := (|x| == 3 ->  (x = 3)  ∨ (x = -3)): 
solution ( ∀ x:solution ∧ x₁ * x₂= 3 * (-3) )  : -9)   := 
sorry

end product_of_solutions_l802_802935


namespace domain_of_g_l802_802633

theorem domain_of_g (f : ℝ → ℝ) (h : ∀ x, 0 ≤ x → x ≤ 4 → ∃ y, f y = x) :
  ∀ x, 0 ≤ x → x ≤ 2 → ∃ y, (λ x, f x + f (x^2)) y = x := 
sorry

end domain_of_g_l802_802633


namespace hyperbola_slopes_l802_802126

variables {x1 y1 x2 y2 x y k1 k2 : ℝ}

theorem hyperbola_slopes (h1 : y1^2 - (x1^2 / 2) = 1)
  (h2 : y2^2 - (x2^2 / 2) = 1)
  (hx : x1 + x2 = 2 * x)
  (hy : y1 + y2 = 2 * y)
  (hk1 : k1 = (y2 - y1) / (x2 - x1))
  (hk2 : k2 = y / x) :
  k1 * k2 = 1 / 2 :=
sorry

end hyperbola_slopes_l802_802126


namespace sum_of_first_10_terms_l802_802878

variable {a : ℕ → ℝ}

-- Definition of arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n, a n = a1 + (n - 1) * d

-- Given condition on the arithmetic sequence
def condition (a : ℕ → ℝ) : Prop :=
  a 4 ^ 2 + a 7 ^ 2 + 2 * a 4 * a 7 = 9

-- Sum of the first 10 terms of the arithmetic sequence
def sum_first_10_terms (a : ℕ → ℝ) : ℝ :=
  ∑ i in finset.range 10, a (i + 1)

-- Statement to prove
theorem sum_of_first_10_terms (a1 d : ℝ) (H : arithmetic_sequence a a1 d) (C : condition a) :
  sum_first_10_terms a = 15 ∨ sum_first_10_terms a = -15 :=
sorry

end sum_of_first_10_terms_l802_802878


namespace crank_slider_mechanism_l802_802031

noncomputable def omega := 10 -- Angular velocity in rad/s
noncomputable def OA := 90 -- Length of OA in cm
noncomputable def AB := 90 -- Length of AB in cm
noncomputable def MB := AB / 3 -- Length of MB in cm
noncomputable def cos (θ : ℝ) : ℝ := real.cos θ
noncomputable def sin (θ : ℝ) : ℝ := real.sin θ

theorem crank_slider_mechanism:
  (∀ t : ℝ, (150 * cos (omega * t), 90 * sin (omega * t)) = (150 * cos (10 * t), 90 * sin (10 * t))) ∧
  (∀ t : ℝ, (-1500 * sin (10 * t), 900 * cos (10 * t)) = (-1500 * sin (omega * t), 900 * cos (omega * t))) :=
by
  intros t
  -- Equations for the position of point M
  split
  · show (150 * cos (omega * t), 90 * sin (omega * t)) = (150 * cos (10 * t), 90 * sin (10 * t))
    sorry

  -- Equations for the velocity of point M
  · show (-1500 * sin (omega * t), 900 * cos (omega * t)) = (-1500 * sin (10 * t), 900 * cos (10 * t))
    sorry

end crank_slider_mechanism_l802_802031


namespace Yolanda_husband_catches_up_in_15_minutes_l802_802345

theorem Yolanda_husband_catches_up_in_15_minutes :
  ∀ (x : ℕ),
    (∀ (y_time h_speed : ℕ), y_time = x + 15 → h_speed = 40 →
      (20 * (y_time) / 60) = (40 * x / 60)) →
    x = 15 :=
by
  intros x h
  have h₁ : 20 * (x + 15) / 60 = 40 * x / 60 := by
    apply h x 40 rfl rfl
  sorry

end Yolanda_husband_catches_up_in_15_minutes_l802_802345


namespace leak_empty_time_l802_802352

/-- 
The time taken for a leak to empty a full tank, given that an electric pump can fill a tank in 7 hours and it takes 14 hours to fill the tank with the leak present, is 14 hours.
 -/
theorem leak_empty_time (P L : ℝ) (hP : P = 1 / 7) (hCombined : P - L = 1 / 14) : L = 1 / 14 ∧ 1 / L = 14 :=
by
  sorry

end leak_empty_time_l802_802352


namespace cos_sum_to_product_l802_802277

theorem cos_sum_to_product (x : ℝ) : 
  (∃ a b c d : ℕ, a * Real.cos (b * x) * Real.cos (c * x) * Real.cos (d * x) =
  Real.cos (2 * x) + Real.cos (6 * x) + Real.cos (10 * x) + Real.cos (14 * x) 
  ∧ a + b + c + d = 18) :=
sorry

end cos_sum_to_product_l802_802277


namespace find_m_l802_802410

def is_good (n : ℤ) : Prop :=
  ¬ (∃ k : ℤ, |n| = k^2)

theorem find_m (m : ℤ) : (m % 4 = 3) → 
  (∃ a b c : ℤ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ is_good a ∧ is_good b ∧ is_good c ∧ (a * b * c) % 2 = 1 ∧ a + b + c = m) :=
sorry

end find_m_l802_802410


namespace pyramid_angle_computation_l802_802358

noncomputable def tan_half_angle {α : ℝ} (a : ℝ) : ℝ :=
  Real.arctan (a * Real.tan α)

theorem pyramid_angle_computation {α : ℝ} {a : ℝ}
  (equilateral : ∃ m, m > 0 ∧ ∀ x y z, x = y ∧ y = z ∧ z = x ⇔ (x, y, z ∈ {a}))
  (face_perpendicular : ∃ (A B : ℝ), A ≠ B ∧ A ⊥ B)
  (face_inclined : ∀ (A C : ℝ), A ≠ C ∧ ∀ θ, θ = Real.arccos (Real.cos α) ∧ θ ≠ 0)
  (tan_half_alpha := Real.tan α) :
  tan_half_angle α (1/2) = Real.arctan (1/2 * tan_half_alpha) ∧
  tan_half_angle α (sqrt(3)/2) = Real.arctan (sqrt(3)/2 * tan_half_alpha) ∧
  tan_half_angle α (sqrt(3)/2) = Real.arctan (sqrt(3)/2 * tan_half_alpha) :=
by sorry

end pyramid_angle_computation_l802_802358


namespace function_minimum_value_l802_802279

theorem function_minimum_value {a : ℝ} (f : ℝ → ℝ) (h : ∀ x, f x = x^2 - 2*a*x + a) :
  (∃ x ∈ Ioo (-∞ : ℝ) 1, ∀ y ∈ Ioo (-∞ : ℝ) 1, f x ≤ f y) ↔ a < 1 :=
by 
  sorry

end function_minimum_value_l802_802279


namespace min_sum_of_factors_of_72_l802_802701

theorem min_sum_of_factors_of_72 (a b: ℤ) (h: a * b = 72) : a + b = -73 :=
sorry

end min_sum_of_factors_of_72_l802_802701


namespace modulus_equality_l802_802025

theorem modulus_equality : 
  (abs ((1 + complex.i) ^ 8)) = (abs ((1 - complex.i) ^ 8)) := 
by
  sorry

end modulus_equality_l802_802025


namespace inconsistent_conditions_l802_802814

-- Definitions based on the given conditions
def B : Nat := 59
def C : Nat := 27
def D : Nat := 31
def A := B * C + D

theorem inconsistent_conditions (A_is_factor : ∃ k : Nat, 4701 = k * A) : false := by
  sorry

end inconsistent_conditions_l802_802814


namespace total_miles_traveled_l802_802671

noncomputable def distance_to_first_museum : ℕ := 5
noncomputable def distance_to_second_museum : ℕ := 15
noncomputable def distance_to_cultural_center : ℕ := 10
noncomputable def extra_detour : ℕ := 3

theorem total_miles_traveled : 
  (2 * (distance_to_first_museum + extra_detour) + 2 * distance_to_second_museum + 2 * distance_to_cultural_center) = 66 :=
  by
  sorry

end total_miles_traveled_l802_802671


namespace peter_remaining_walk_time_l802_802736

-- Define the parameters and conditions
def total_distance : ℝ := 2.5
def time_per_mile : ℝ := 20
def distance_walked : ℝ := 1

-- Define the remaining distance
def remaining_distance : ℝ := total_distance - distance_walked

-- Define the remaining time Peter needs to walk
def remaining_time_to_walk (d : ℝ) (t : ℝ) : ℝ := d * t

-- State the problem we want to prove
theorem peter_remaining_walk_time :
  remaining_time_to_walk remaining_distance time_per_mile = 30 :=
by
  -- Placeholder for the proof
  sorry

end peter_remaining_walk_time_l802_802736


namespace Fran_same_distance_speed_l802_802682

noncomputable def Joann_rides (v_j t_j : ℕ) : ℕ := v_j * t_j

def Fran_speed (d t_f : ℕ) : ℕ := d / t_f

theorem Fran_same_distance_speed
  (v_j t_j t_f : ℕ) (hj: v_j = 15) (tj: t_j = 4) (tf: t_f = 5) : Fran_speed (Joann_rides v_j t_j) t_f = 12 := by
  have hj_dist: Joann_rides v_j t_j = 60 := by
    rw [hj, tj]
    sorry -- proof of Joann's distance
  have d_j: ℕ := 60
  have hf: Fran_speed d_j t_f = Fran_speed 60 5 := by
    rw ←hj_dist
    sorry -- proof to equate d_j with Joann's distance
  show Fran_speed 60 5 = 12
  sorry -- Final computation proof

end Fran_same_distance_speed_l802_802682


namespace companion_sequence_B4_arithmetic_sequence_B9_companion_sequence_even_n_l802_802571

def is_companion_sequence (A B : List ℕ) : Prop :=
  B.length = A.length ∧
  B.head = A.last ∧
  (∀ k : ℕ, 1 < k ∧ k < B.length + 1 → (B.get? k).iget + (B.get? (k - 1)).iget = (A.get? k).iget + (A.get? (k - 1)).iget)

theorem companion_sequence_B4 (A4 : List ℕ) (hA4 : A4 = [3, 1, 2, 5]) :
  ∃ (B4 : List ℕ), is_companion_sequence A4 B4 ∧ B4 = [5, -1, 4, 3] := sorry

theorem arithmetic_sequence_B9 (A9 B9 : List ℕ) (hA9len : A9.length = 9) (hB9len : B9.length = 9)
  (hComp : is_companion_sequence A9 B9) :
  ∃ (b9 a9 a1 : ℕ), b9 = B9.getLast (by simp) ∧ a9 = A9.getLast (by simp) ∧ a1 = A9.head ∧ (b9 + a1 = 2 * a9) := 
sorry

theorem companion_sequence_even_n (A B : List ℕ) (n : ℕ) (hEven : Even n) (hAlen : A.length = n) (hBlen : B.length = n)
  (hComp : is_companion_sequence A B) : 
  B.getLast (by linarith) = A.head := 
sorry

end companion_sequence_B4_arithmetic_sequence_B9_companion_sequence_even_n_l802_802571


namespace proof_equivalent_l802_802101

variable (a b x : ℝ)

def p : Prop := a > b → a > b^2
def q : Prop := ∀ x, (x ≤ 1 → x^2 + 2 * x - 3 ≤ 0) ∧ (∃ x, (x^2 + 2 * x - 3 > 0 ∧ x ≤ 1))

theorem proof_equivalent :
(¬ p ∧ q) ∧ (¬ p ∧ q) ∨ (p ∧ ¬ q) ∧ (¬ p ∧ q) = ∃b,
¬ (a > b → a > b^2) ∧ (∀ x, (x ≤ 1 → x^2 + 2 * x - 3 ≤ 0) ∧ (∃ x, (x^2 + 2 * x - 3 > 0 ∧ x ≤ 1))). 
:= sorry

end proof_equivalent_l802_802101


namespace Jolene_charge_per_car_l802_802683

theorem Jolene_charge_per_car (babysitting_families cars_washed : ℕ) (charge_per_family total_raised babysitting_earnings car_charge : ℕ) :
  babysitting_families = 4 →
  charge_per_family = 30 →
  cars_washed = 5 →
  total_raised = 180 →
  babysitting_earnings = babysitting_families * charge_per_family →
  car_charge = (total_raised - babysitting_earnings) / cars_washed →
  car_charge = 12 :=
by
  intros
  sorry

end Jolene_charge_per_car_l802_802683


namespace cone_tetrahedron_max_vol_optim_OB_l802_802888

theorem cone_tetrahedron_max_vol_optim_OB :
  -- Let P, O, A, B, H, and C be points in a 3D space such that:
  ∀ (P O A B H C : ℝ^3),
  -- 1. The cross-section of the cone with vertex P is an isosceles right triangle.
  -- This condition implies certain geometric relations between P, A, O, etc.
  -- 2. A is on the circumference of the base circle, B inside the base circle, and O is the center.
  (circle_center_radius O P A) ∧ (inside_circle B O P) →
  -- 3. AB ⊥ OB with the foot of the perpendicular at B.
  (is_perpendicular A B O B) →
  -- 4. OH ⊥ PB with the foot of the perpendicular at H.
  (is_perpendicular O H P B) →
  -- 5. PA = 4
  (dist P A = 4) →
  -- 6. C is the midpoint of PA.
  (is_midpoint C P A) →
  -- To prove: The length of OB is given by the expression when volume of tetrahedron O-HPC is maximized.
  dist O B = 2 * sqrt 6 / 3 :=
sorry

end cone_tetrahedron_max_vol_optim_OB_l802_802888


namespace quadrilateral_areas_l802_802197

noncomputable def square_side_length : ℝ := 6
noncomputable def side_divisions : ℝ := 3

def possible_areas_of_quadrilateral (side_length : ℝ) (divisions : ℝ) : set ℝ :=
{16, 18, 20}

theorem quadrilateral_areas (side_length : ℝ) (divisions : ℝ) :
  possible_areas_of_quadrilateral side_length divisions = {16, 18, 20} :=
by
  have side_length := square_side_length
  have divisions := side_divisions
  -- The proof goes here
  sorry

end quadrilateral_areas_l802_802197


namespace eight_xyz_le_one_equality_conditions_l802_802261

theorem eight_xyz_le_one (x y z : ℝ) (h : x^2 + y^2 + z^2 + 2 * x * y * z = 1) :
  8 * x * y * z ≤ 1 :=
sorry

theorem equality_conditions (x y z : ℝ) (h : x^2 + y^2 + z^2 + 2 * x * y * z = 1) :
  8 * x * y * z = 1 ↔ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) ∨
                   (x = -1/2 ∧ y = -1/2 ∧ z = 1/2) ∨
                   (x = -1/2 ∧ y = 1/2 ∧ z = -1/2) ∨
                   (x = 1/2 ∧ y = -1/2 ∧ z = -1/2) :=
sorry

end eight_xyz_le_one_equality_conditions_l802_802261


namespace modulus_of_complex_number_l802_802495

def a : ℝ := 2
def b : ℝ := - (5 / 6)
def complex_number := ⟨a, b⟩

noncomputable def modulus (z : ℝ × ℝ) : ℝ :=
  real.sqrt (z.1 ^ 2 + z.2 ^ 2)

theorem modulus_of_complex_number : modulus complex_number = 13 / 6 :=
by
  sorry

end modulus_of_complex_number_l802_802495


namespace range_of_S_l802_802625

noncomputable def sqrt (x : ℝ) : ℝ :=
if x >= 0 then real.sqrt x else 0

theorem range_of_S (a b n : ℝ) (h1 : 3*sqrt((n + 1)^2 + n + 1)^2 + 5*|b| = 7)
  (h2 : a > 0) : ∀ S, S = 2*sqrt(a) - 3*|b| → -21/5 ≤ S ∧ S ≤ 14/3 :=
by
  sorry

end range_of_S_l802_802625


namespace minimum_possible_sum_of_4x4x4_cube_l802_802365

theorem minimum_possible_sum_of_4x4x4_cube: 
  (∀ die: ℕ, (1 ≤ die) ∧ (die ≤ 6) ∧ (∃ opposite, die + opposite = 7)) → 
  (∃ sum, sum = 304) :=
by
  sorry

end minimum_possible_sum_of_4x4x4_cube_l802_802365


namespace find_k_of_quadratic_eqn_with_root_l802_802984

theorem find_k_of_quadratic_eqn_with_root (k : ℝ) (h : ∀ x : ℂ, x^2 + (4 : ℂ) * x + (k : ℂ) = 0 → (x = -2 + 3 * complex.I ∨ x = -2 - 3 * complex.I)) : k = 13 :=
sorry

end find_k_of_quadratic_eqn_with_root_l802_802984


namespace proposition3_is_correct_l802_802012

theorem proposition3_is_correct :
  (∃ (P1 P2 : Plane), P1 ⊓ P2 = ∅ ∧ (∃ p1 p2 p3 : Point, p1 ∈ P1 ∧ p1 ∈ P2 ∧ p2 ∈ P1 ∧ p2 ∈ P2 ∧ p3 ∈ P1 ∧ p3 ∈ P2 ∧ ¬Collinear p1 p2 p3)) → False ∧
  (∀ p1 p2 p3 : Point, ∃! P : Plane, p1 ∈ P ∧ p2 ∈ P ∧ p3 ∈ P) → False ∧
  (∃ l1 l2 : Line, Parallel l1 l2 ∧ ∃! P : Plane, l1 ⊆ P ∧ l2 ⊆ P) ∧
  (∀ l1 l2 l3 : Line, (Intersect l1 l2 ∧ Intersect l1 l3 ∧ Intersect l2 l3) → Coplanar l1 l2 l3) → False :=
by
  sorry

end proposition3_is_correct_l802_802012


namespace marble_count_l802_802177

theorem marble_count (r g b : ℕ) (h1 : g + b = 6) (h2 : r + b = 8) (h3 : r + g = 4) : r + g + b = 9 :=
sorry

end marble_count_l802_802177


namespace intersection_of_line_and_parabola_midpoint_l802_802992

noncomputable def midpoint_of_intersection (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem intersection_of_line_and_parabola_midpoint : 
  (y x : ℝ) (hline : y = x) (hparabola : y^2 = 4 * x)
  (Midpoint: ∃ A B : ℝ × ℝ, A = (0, 0) ∧ B = (4, 4) ∧ midpoint_of_intersection A B = (2, 2)) :
  (∃ A B : ℝ × ℝ, A = (0, 0) ∧ B = (4, 4) ∧ midpoint_of_intersection A B = (2, 2)) :=
sorry

end intersection_of_line_and_parabola_midpoint_l802_802992


namespace product_of_solutions_eq_neg_nine_product_of_solutions_l802_802932

theorem product_of_solutions_eq_neg_nine :
  ∀ (x : ℝ), (|x| = 3 * (|x| - 2)) → x = 3 ∨ x = -3 :=
by
  sorry

theorem product_of_solutions :
  (∀ (x : ℝ), (|x| = 3 * (|x| - 2)) → (∃ (a b : ℝ), x = a ∨ x = b ∧ a * b = -9)) :=
by
  sorry

end product_of_solutions_eq_neg_nine_product_of_solutions_l802_802932


namespace right_shift_sin_to_cos_l802_802748

-- Define original function
def original_function (x : ℝ) : ℝ := sin (2 * x)

-- Define target function after right shift
def target_function (x : ℝ) : ℝ := cos (2 * x + (-5 * Real.pi / 6))

-- Define the shift amount
def shift_amount : ℝ := Real.pi / 6

-- Define the phase shift value to prove
def phi : ℝ := -5 * Real.pi / 6

-- Lean 4 statement to prove
theorem right_shift_sin_to_cos :
  (∀ x : ℝ, original_function (x + shift_amount) = target_function x) ∧
  (-Real.pi < phi ∧ phi ≤ Real.pi) :=
by
  -- proof here
  sorry

end right_shift_sin_to_cos_l802_802748


namespace find_values_of_square_and_triangle_l802_802642

noncomputable def values_of_square_and_triangle : Prop :=
  ∃ (x y square triangle : ℕ), 
  (x = 5) ∧
  (y = 1) ∧
  (2 * x + y = square) ∧
  (x - 2 * y = 3) ∧
  (square = 11) ∧
  (triangle = y)

theorem find_values_of_square_and_triangle : values_of_square_and_triangle :=
  by {
    existsi [5, 1, 11, 1],
    repeat {split},
    exact rfl, -- x = 5
    exact rfl, -- y = 1
    exact rfl, -- 2 * 5 + 1 = 11
    norm_num, -- 5 - 2 * 1 = 3
    exact rfl, -- square = 11
    exact rfl, -- triangle = y = 1
    sorry
  }

end find_values_of_square_and_triangle_l802_802642


namespace decimal_to_fraction_l802_802507

theorem decimal_to_fraction :
  (\(x\), \(y\), (\(x, y\)) = (3, 110)) → 0.3\overline{45} = \(\frac{83}{110}\) := λ ⟨3, 110, 3, 110⟩, sorry

end decimal_to_fraction_l802_802507


namespace incircles_tangent_ABD_BCD_l802_802573

variables {A B C D : Type} [EuclideanGeometry A]
variables (a b c d : A)

-- Given conditions:
-- 1. Quadrilateral ABCD is convex.
-- 2. The incircles of triangles ABC and ACD touch each other.
def isConvexQuadrilateral (f : A) (g : A) (h : A) (i : A) : Prop := sorry -- Formalize convex quadrilateral definition
def incirclesTouching (t1 : triangle A) (t2 : triangle A) : Prop := sorry -- Formalize touching incircles definition

-- The main theorem to prove
theorem incircles_tangent_ABD_BCD
  (convexABCD : isConvexQuadrilateral a b c d)
  (incirclesTouch : incirclesTouching (triangle.mk a b c) (triangle.mk a c d)) :
  incirclesTouching (triangle.mk a b d) (triangle.mk b c d) :=
sorry

end incircles_tangent_ABD_BCD_l802_802573


namespace product_of_solutions_l802_802931

theorem product_of_solutions :
  let solutions := {x : ℝ | |x| = 3 * (|x| - 2)} in
  ∏ x in solutions, x = -9 := by
  sorry

end product_of_solutions_l802_802931


namespace percent_of_y_equal_to_30_percent_of_60_percent_l802_802318

variable (y : ℝ)

theorem percent_of_y_equal_to_30_percent_of_60_percent (hy : y ≠ 0) :
  ((0.18 * y) / y) * 100 = 18 := by
  have hneq : y ≠ 0 := hy
  field_simp [hneq]
  norm_num
  sorry

end percent_of_y_equal_to_30_percent_of_60_percent_l802_802318


namespace constant_a_value_l802_802111

theorem constant_a_value (S : ℕ → ℝ)
  (a : ℝ)
  (h : ∀ n : ℕ, S n = 3 ^ (n + 1) + a) :
  a = -3 :=
sorry

end constant_a_value_l802_802111


namespace cards_difference_product_divisibility_l802_802463

theorem cards_difference_product_divisibility :
  (∃ pairs : list (ℕ × ℕ), 
    (∀ p ∈ pairs, 1 ≤ p.1 ∧ p.1 ≤ 50 ∧ 1 ≤ p.2 ∧ p.2 ≤ 50 ∧ 
      (p.1 - p.2 = 11 ∨ p.2 - p.1 = 11) ∧ 
      (p.1 * p.2) % 5 = 0) ∧ 
    pairs.length = 15) := 
sorry

end cards_difference_product_divisibility_l802_802463


namespace problem_solved_by_at_least_one_l802_802303

theorem problem_solved_by_at_least_one {p1 p2 p3 : ℚ} (h1 : p1 = 1 / 5) (h2 : p2 = 1 / 3) (h3 : p3 = 1 / 4) :
  let p : ℚ := 1 - (1 - p1) * (1 - p2) * (1 - p3) in
  p = 3 / 5 :=
sorry

end problem_solved_by_at_least_one_l802_802303


namespace total_worth_of_stock_l802_802401

variable (X : ℝ) -- Assume the total worth of the stock in Rs
variable (overall_loss : ℝ) -- Overall loss in Rs
variable (thirty_percent_profit_part : ℝ) -- Worth from the profit
variable (seventy_percent_loss_part : ℝ) -- Worth from the loss

-- Define the conditions from the problem
def profit_from_twenty_percent : ℝ := 0.10 * 0.20 * X
def loss_from_eighty_percent : ℝ := 0.05 * 0.80 * X
def overall_loss_condition : Prop := overall_loss = 250
def stock_split_conditions : Prop := thirty_percent_profit_part = 0.20 * X ∧ seventy_percent_loss_part = 0.80 * X

theorem total_worth_of_stock 
  (profit_loss_relation: loss_from_eighty_percent - profit_from_twenty_percent = overall_loss)
  (loss_value : overall_loss = 250) : 
  X = 12500 :=
sorry

end total_worth_of_stock_l802_802401


namespace sum_first_five_arithmetic_l802_802099

theorem sum_first_five_arithmetic (a : ℕ → ℝ) (h₁ : ∀ n, a n = a 0 + n * (a 1 - a 0)) (h₂ : a 1 = -1) (h₃ : a 3 = -5) :
  (a 0 + a 1 + a 2 + a 3 + a 4) = -15 :=
by
  sorry

end sum_first_five_arithmetic_l802_802099


namespace cost_per_set_l802_802744

variable (C : ℝ)

theorem cost_per_set :
  let total_manufacturing_cost := 10000 + 500 * C
  let revenue := 500 * 50
  let profit := revenue - total_manufacturing_cost
  profit = 5000 → C = 20 := 
by
  sorry

end cost_per_set_l802_802744


namespace card_pairs_satisfying_conditions_l802_802435

theorem card_pairs_satisfying_conditions :
  ∃ (s : Finset (ℕ × ℕ)),
    (∀ p ∈ s, p.1 ≠ p.2 ∧ (p.1 = p.2 + 11 ∨ p.2 = p.1 + 11) ∧ ((p.1 * p.2) % 5 = 0))
    ∧ s.card = 15 :=
begin
  sorry
end

end card_pairs_satisfying_conditions_l802_802435


namespace halloween_candy_problem_l802_802553

theorem halloween_candy_problem :
  ∀ (total_initial : ℕ) (eaten_first_night : ℕ) (total_final : ℕ) (sister_candy : ℕ),
    total_initial = 33 →
    eaten_first_night = 17 →
    total_final = 35 →
    sister_candy = total_final - (total_initial - eaten_first_night) →
    sister_candy = 19 :=
by {
  intros total_initial eaten_first_night total_final sister_candy H_initial H_eaten H_final H_sister,
  rw [H_initial, H_eaten, H_final] at H_sister,
  exact H_sister,
}

end halloween_candy_problem_l802_802553


namespace num_distinct_values_of_m_l802_802035

theorem num_distinct_values_of_m : ∃ m_values, (∀ (x₁ x₂ : ℤ), x₁ * x₂ = 30 → (x₁ + x₂) ∈ m_values) ∧ m_values.card = 8 :=
sorry

end num_distinct_values_of_m_l802_802035


namespace tan_double_angle_third_quadrant_l802_802559

theorem tan_double_angle_third_quadrant
  (α : ℝ)
  (sin_alpha : Real.sin α = -3/5)
  (h_quadrant : π < α ∧ α < 3 * π / 2) :
  Real.tan (2 * α) = 24 / 7 :=
sorry

end tan_double_angle_third_quadrant_l802_802559


namespace curve_is_ellipse_perpendicular_intersects_l802_802110

noncomputable def curve (P : ℝ × ℝ) : Prop :=
  dist P (-⟨sqrt 3, 0⟩) + dist P (⟨sqrt 3, 0⟩) = 4

noncomputable def ellipse_eqn (x y : ℝ) : Prop :=
  (x^2) / 4 + y^2 = 1

theorem curve_is_ellipse : ∀ P, curve P ↔ ∃ x y, P = (x, y) ∧ ellipse_eqn x y :=
by sorry

noncomputable def line (k : ℝ) : ℝ × ℝ → Prop :=
λ (x, y), y = k * x - 2

theorem perpendicular_intersects (x1 y1 x2 y2 k : ℝ)
  (h1 : ellipse_eqn x1 y1)
  (h2 : ellipse_eqn x2 y2)
  (h3 : line k (x1, y1))
  (h4 : line k (x2, y2))
  (h5 : x1 * x2 + y1 * y2 = 0) :
  k = 2 ∨ k = -2 :=
by sorry

end curve_is_ellipse_perpendicular_intersects_l802_802110


namespace incorrect_population_statement_l802_802408

/-- Among the statements about populations, we need to identify the incorrect one.
Given:
- Statement A: For the control of harmful animals such as rats, it is necessary to reduce their K value as much as possible.
- Statement B: Population density is the most basic quantitative characteristic of a population.
- Statement C: In the absence of natural selection, the genotype frequency of a population can still change.

We need to prove that Statement D is incorrect.
- Statement D: Under ideal conditions, the factors that mainly affect the growth of population numbers are the carrying capacity of the environment.
--/
theorem incorrect_population_statement 
  (A: "For the control of harmful animals such as rats, it is necessary to reduce their K value as much as possible.")
  (B: "Population density is the most basic quantitative characteristic of a population.")
  (C: "In the absence of natural selection, the genotype frequency of a population can still change."):
  "Under ideal conditions, the factors that mainly affect the growth of population numbers are the carrying capacity of the environment." → false :=
by
  sorry

end incorrect_population_statement_l802_802408


namespace smaller_pie_crust_flour_l802_802673

theorem smaller_pie_crust_flour (p1 p2 : ℕ) (f1 : ℚ) (c : ℚ) (h1 : p1 = 30) (h2 : f1 = 1/5) (h3 : c = p1 * f1) (h4 : p2 = 40) :
  ∃ f2 : ℚ, p2 * f2 = c ∧ f2 = 3/20 :=
by
  use 6 / 40
  split
  { rw [h4, mul_div_cancel_left, h3, h1, h2]; norm_num }
  { norm_num }
  sorry

end smaller_pie_crust_flour_l802_802673


namespace sum_common_divisors_l802_802547

-- Define the sum of a set of numbers
def set_sum (s : Set ℕ) : ℕ :=
  s.fold (λ x acc => x + acc) 0

-- Define the divisors of a number
def divisors (n : ℕ) : Set ℕ :=
  { d | d > 0 ∧ n % d = 0 }

-- Definitions based on the given conditions
def divisors_of_60 : Set ℕ := divisors 60
def divisors_of_18 : Set ℕ := divisors 18
def common_divisors : Set ℕ := divisors_of_60 ∩ divisors_of_18

-- Declare the theorem to be proved
theorem sum_common_divisors : set_sum common_divisors = 12 :=
  sorry

end sum_common_divisors_l802_802547


namespace harmonic_sum_pattern_l802_802240

theorem harmonic_sum_pattern :
  1 - (1 / 2) + (1 / 3) - (1 / 4) + ... + (1 / 2017) - (1 / 2018) = (1 / 1010) + ... + (1 / 2018) :=
sorry

end harmonic_sum_pattern_l802_802240


namespace pyramid_base_regular_l802_802740

/-- 
  If a pyramid has equal lateral edges, and any two adjacent lateral faces form 
  equal dihedral angles, and the base is a polygon with an odd number of sides, 
  then the base polygon is regular.
-/
theorem pyramid_base_regular 
  (n : ℕ) (odd_n : n % 2 = 1)
  (S : Point) (A : Fin n → Point)
  (equal_lateral_edges : ∀ i, dist S (A i) = dist S (A 0))
  (equal_dihedral_angles : ∀ i, dihedral_angle (S, A i, A ((i + 1) % n)) = dihedral_angle (S, A 0, A 1))
  : is_regular_polygon (A : Fin n → Point) :=
sorry

end pyramid_base_regular_l802_802740


namespace collinear_X_Y_Z_l802_802616

open EuclideanGeometry

variables {A B C X Y Z P Q : Point}
variables {Γ₁ Γ₂ : Circle}
variables {l : Line}

theorem collinear_X_Y_Z 
    (h1 : Γ₁ ∩ Γ₂ = {A, B})
    (h2 : tangent AC Γ₁ A)
    (h3 : ∠ A B C = 90)
    (h4 : line_through C l)
    (h5 : l ∩ Γ₂ = {P, Q})
    (h6 : second_intersection AP Γ₁ = X)
    (h7 : second_intersection AQ Γ₁ = Z)
    (h8 : perpendicular_from A l Y)
    : collinear {X, Y, Z} :=
sorry

end collinear_X_Y_Z_l802_802616


namespace man_speed_upstream_l802_802859

def man_speed_still_water : ℕ := 50
def speed_downstream : ℕ := 80

theorem man_speed_upstream : (man_speed_still_water - (speed_downstream - man_speed_still_water)) = 20 :=
by
  sorry

end man_speed_upstream_l802_802859


namespace candy_problem_solution_l802_802396

theorem candy_problem_solution :
  ∃ (a : ℕ), a % 10 = 6 ∧ a % 15 = 11 ∧ 200 ≤ a ∧ a ≤ 250 ∧ (a = 206 ∨ a = 236) :=
begin
  sorry
end

end candy_problem_solution_l802_802396


namespace sum_of_first_100_terms_l802_802981

noncomputable def S_n (n : ℕ) : ℚ := (n * (n + 1)) / 2
noncomputable def a_n (n : ℕ) : ℚ := n

-- Ensure the initial conditions hold:
axiom a_3_eq_3 : a_n 3 = 3
axiom S_4_eq_10 : S_n 4 = 10

-- Prove that the sum of the first 100 terms of the sequence {1/S_n} is 200/101
theorem sum_of_first_100_terms : ∑ i in Finset.range 100, 1 / S_n (i + 1) = 200 / 101 := 
by
  sorry

end sum_of_first_100_terms_l802_802981


namespace cubic_coefficient_determination_l802_802906

def f (x : ℚ) (A B C D : ℚ) : ℚ := A*x^3 + B*x^2 + C*x + D

theorem cubic_coefficient_determination {A B C D : ℚ}
  (h1 : f 1 A B C D = 0)
  (h2 : f (2/3) A B C D = -4)
  (h3 : f (4/5) A B C D = -16/5) :
  A = 15 ∧ B = -37 ∧ C = 30 ∧ D = -8 :=
  sorry

end cubic_coefficient_determination_l802_802906


namespace find_vectors_cosine_angle_l802_802138

-- Definition of the given vectors and conditions
variables (x y z : ℝ)
def a := (x, 4, 1) : ℝ × ℝ × ℝ
def b := (-2, y, -1) : ℝ × ℝ × ℝ
def c := (3, -2, z) : ℝ × ℝ × ℝ

-- Parallel and perpendicular conditions
def parallel (u v : ℝ × ℝ × ℝ) := ∃ k : ℝ, u = (k * v.1, k * v.2, k * v.3)
def orthogonal (u v : ℝ × ℝ × ℝ) := u.1 * v.1 + u.2 * v.2 + u.3 * v.3 = 0

-- Proof statements
theorem find_vectors :
  parallel a b → orthogonal b c →
  a = (2, 4, 1) ∧ b = (-2, -4, -1) ∧ c = (3, -2, 2) :=
by
  sorry

theorem cosine_angle :
  parallel a b → orthogonal b c →
  let ac := (2 + 3,  4 + -2,  1 + 2) in  -- a + c
  let bc := (-2 + 3, -4 + -2, -1 + 2) in -- b + c
  real.cos ac bc = -2 / 19 :=
by
  sorry

end find_vectors_cosine_angle_l802_802138


namespace number_of_satisfying_subsets_l802_802102

theorem number_of_satisfying_subsets (A B C : Finset ℕ) (hAB : A ⊆ B) (hAC : A ⊆ C) (hB : B = {0, 1, 2, 3, 4}) (hC : C = {0, 2, 4, 8}) : 
  (∃ n, n = 8) :=
by
  sorry

end number_of_satisfying_subsets_l802_802102


namespace shortest_distance_to_line_l802_802737

open Classical

variables {P A B C : Type} [MetricSpace P] [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables (PA PB PC : ℝ)
variables (l : ℕ) -- l represents the line

-- Given conditions
def PA_dist : ℝ := 4
def PB_dist : ℝ := 5
def PC_dist : ℝ := 2

theorem shortest_distance_to_line (hPA : PA = PA_dist) (hPB : PB = PB_dist) (hPC : PC = PC_dist) :
  ∃ d, d ≤ 2 := 
sorry

end shortest_distance_to_line_l802_802737


namespace base_number_exponent_l802_802159

theorem base_number_exponent (x : ℝ) (h : ((x^4) * 3.456789) ^ 12 = y) (has_24_digits : true) : x = 10^12 :=
  sorry

end base_number_exponent_l802_802159


namespace smallest_fraction_numerator_l802_802875

theorem smallest_fraction_numerator :
  ∃ (a b : ℕ), a ≥ 10 ∧ a ≤ 99 ∧ b ≥ 10 ∧ b ≤ 99 ∧ (4 * b < 9 * a) ∧ 
  (∀ (a' b' : ℕ), a' ≥ 10 ∧ a' ≤ 99 ∧ b' ≥ 10 ∧ b' ≤ 99 ∧ (4 * b' < 9 * a') → b * a' ≥ a * b') ∧ a = 41 :=
sorry

end smallest_fraction_numerator_l802_802875


namespace solution_correctness_l802_802958

-- Define a geometric sequence with the given properties.
def geometric_sequence (a₁ x : ℝ) (n : ℕ) : ℝ := a₁ * x^n

-- Define the sum of the first n terms S_n of the geometric sequence.
def sum_geometric_sequence (a₁ x : ℝ) (n : ℕ) : ℝ :=
  if x = 1 then n * a₁
  else a₁ * (1 - x^(n+1)) / (1 - x)

-- Define the limit f(x) as n tends to infinity.
def f (x : ℝ) : ℝ :=
  if x = 1 then 2
  else if x > 1 then 2
  else if 0 < x ∧ x < 1 then 2 / (2 - x)
  else 0  -- handle unnecessary case

-- Define the proof statement outlining the conditions and the expected function.
theorem solution_correctness (x : ℝ) :
  f(x) =
  if x = 1 then 2
  else if x > 1 then 2
  else if 0 < x ∧ x < 1 then 2 / (2 - x)
  else 0 :=
sorry

end solution_correctness_l802_802958


namespace card_S_is_power_of_two_l802_802855

theorem card_S_is_power_of_two {S : Finset ℕ} 
    (h : ∀ s ∈ S, ∀ d ∈ (Nat.divisors s).to_finset, d ≠ 0 → ∃! t ∈ S, Nat.gcd s t = d) :
    ∃ k : ℕ, S.card = 2^k :=
sorry

end card_S_is_power_of_two_l802_802855


namespace inverse_100_mod_101_l802_802054

theorem inverse_100_mod_101 :
  ∃ x, (x : ℤ) ≡ 100 [MOD 101] ∧ 100 * x ≡ 1 [MOD 101] :=
by {
  use 100,
  split,
  { exact rfl },
  { norm_num }
}

end inverse_100_mod_101_l802_802054


namespace fran_speed_l802_802677

theorem fran_speed (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) 
    (h_joann : joann_speed = 15) (h_joann_time : joann_time = 4) (h_fran_time : fran_time = 5) : 
    (joann_speed * joann_time) / fran_time = 12 :=
by
  rw [h_joann, h_joann_time, h_fran_time]
  norm_num
  sorry

end fran_speed_l802_802677


namespace jane_mean_score_l802_802669

def jane_scores : List ℝ := [90, 95, 85, 100, 88, 92]
def number_of_quizzes : ℝ := 6
def sum_of_scores : ℝ := 550
def mean_score : ℝ := sum_of_scores / number_of_quizzes

theorem jane_mean_score : mean_score = 91.67 :=
  by
    sorry

end jane_mean_score_l802_802669


namespace max_points_line_intersects_four_coplanar_circles_l802_802076

open EuclideanGeometry

theorem max_points_line_intersects_four_coplanar_circles (C1 C2 C3 C4 : Circle) :
  coplanar {C1, C2, C3, C4} →
  (∃ l : Line, (∀ c ∈ {C1, C2, C3, C4}, line_intersects_circle l c) ∧
  ∀ p, p ∈ (line_circle_intersections l C1 ∪ line_circle_intersections l C2 ∪
            line_circle_intersections l C3 ∪ line_circle_intersections l C4) →
         p.count_occurrences ≤ 8) :=
by
  intro h_coplanar
  -- we would start the proof here
  sorry

end max_points_line_intersects_four_coplanar_circles_l802_802076


namespace evaluate_expression_l802_802048

theorem evaluate_expression : 
  (3^2 + 3^1 + 3^0 + 3^(-1)) / (3^(-3) + 3^(-4) + 3^(-5)) = 249 := 
by 
  sorry

end evaluate_expression_l802_802048


namespace expected_value_of_groups_l802_802784

noncomputable def expectedNumberOfGroups (k m : ℕ) : ℝ :=
  1 + (2 * k * m) / (k + m)

theorem expected_value_of_groups (k m : ℕ) :
  k > 0 → m > 0 → expectedNumberOfGroups k m = 1 + 2 * k * m / (k + m) :=
by
  intros
  unfold expectedNumberOfGroups
  sorry

end expected_value_of_groups_l802_802784


namespace count_valid_pairs_l802_802442

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

def difference_is_11 (a b : ℕ) : Prop := abs (a - b) = 11

def valid_card (a b : ℕ) : Prop :=
  a ∈ finset.range 51 ∧ b ∈ finset.range 51 ∧
  difference_is_11 a b ∧ (is_divisible_by_5 a ∨ is_divisible_by_5 b)

theorem count_valid_pairs : finset.card {p : ℕ × ℕ | valid_card p.1 p.2 ∧ p.1 < p.2}.to_finset = 15 := 
  sorry

end count_valid_pairs_l802_802442


namespace percent_of_percent_l802_802323

theorem percent_of_percent (y : ℝ) : 0.3 * (0.6 * y) = 0.18 * y :=
sorry

end percent_of_percent_l802_802323


namespace card_choice_count_l802_802485

theorem card_choice_count :
  ∃ (pairs : Finset (ℕ × ℕ)),
    (∀ (a b : ℕ), (a, b) ∈ pairs → 1 ≤ a ∧ a ≤ 50 ∧ 1 ≤ b ∧ b ≤ 50) ∧
    (∀ (a b : ℕ), (a, b) ∈ pairs → |a - b| = 11) ∧
    (∀ (a b : ℕ), (a, b) ∈ pairs → (a * b) % 5 = 0) ∧
    card pairs = 15 := 
sorry

end card_choice_count_l802_802485


namespace triangle_MNP_equilateral_l802_802016

-- Define the centroids and equilateral triangles as given in the conditions
variable {A B C D E F M N P : Type*}

-- Assume important conditions:
-- M is the centroid of triangle ABC
-- N is the centroid of triangle DCE
-- P is the centroid of triangle BEF
-- These triangles are equilateral triangles
variables (triangleABC : M ∈ centroid A B C)
variables (triangleDCE : N ∈ centroid D C E)
variables (triangleBEF : P ∈ centroid B E F)
variables (equilateralABC : ∀ (X Y Z : Type*) [is_equilateral_triangle X Y Z], is_equilateral_triangle A B C)
variables (equilateralDCE : ∀ (X Y Z : Type*) [is_equilateral_triangle X Y Z], is_equilateral_triangle D C E)
variables (equilateralBEF : ∀ (X Y Z : Type*) [is_equilateral_triangle X Y Z], is_equilateral_triangle B E F)

theorem triangle_MNP_equilateral :
  is_equilateral_triangle M N P :=
sorry

end triangle_MNP_equilateral_l802_802016


namespace chris_mixed_raisins_l802_802898

-- Conditions
variables (R C : ℝ)

-- 1. Chris mixed some pounds of raisins with 3 pounds of nuts.
-- 2. A pound of nuts costs 3 times as much as a pound of raisins.
-- 3. The total cost of the raisins was 0.25 of the total cost of the mixture.

-- Problem statement: Prove that R = 3 given the conditions
theorem chris_mixed_raisins :
  R * C = 0.25 * (R * C + 3 * 3 * C) → R = 3 :=
by
  sorry

end chris_mixed_raisins_l802_802898


namespace find_first_discount_l802_802009

-- Definitions for the given conditions
def list_price : ℝ := 150
def final_price : ℝ := 105
def second_discount : ℝ := 12.5

-- Statement representing the mathematical proof problem
theorem find_first_discount (x : ℝ) : 
  list_price * ((100 - x) / 100) * ((100 - second_discount) / 100) = final_price → x = 20 :=
by
  sorry

end find_first_discount_l802_802009


namespace upper_bound_of_n_l802_802693

theorem upper_bound_of_n (m n : ℕ) (h_m : m ≥ 2)
  (h_div : ∀ a : ℕ, gcd a n = 1 → n ∣ a^m - 1) : 
  n ≤ 4 * m * (2^m - 1) := 
sorry

end upper_bound_of_n_l802_802693


namespace problem1_problem2_l802_802119

noncomputable def f (x a : ℝ) := abs (x - 1) + abs (x - a)

-- Problem 1
theorem problem1 (a x : ℝ) (h : a ≤ 2) : 
  f x 2 ≥ 2 ↔ (x ≤ 1/2 ∨ x ≥ 5/2) :=
sorry

-- Problem 2
noncomputable def F (x a : ℝ) := f x a + abs (x - 1)

theorem problem2 (a : ℝ) (h : a > 1) (h1 : ∀ x : ℝ, F x a ≥ 1) : 
  a ∈ set.Ici 2 :=
sorry

end problem1_problem2_l802_802119


namespace find_f_inv_value_l802_802109

noncomputable def f (x : ℝ) : ℝ := 8^x
noncomputable def f_inv (y : ℝ) : ℝ := Real.logb 8 y

theorem find_f_inv_value (a : ℝ) (h : a = 8^(1/3)) : f_inv (a + 2) = Real.logb 8 (8^(1/3) + 2) := by
  sorry

end find_f_inv_value_l802_802109


namespace cos_420_l802_802041

-- Definitions and conditions from the problem
def periodicity_cosine (θ : ℝ) : Prop := cos (θ + 360 * (Real.pi / 180)) = cos θ
def special_angle_cosine : Prop := cos (60 * (Real.pi / 180)) = 1 / 2

-- Main theorem stating the problem and conditions
theorem cos_420 : periodicity_cosine (60 * (Real.pi / 180)) → special_angle_cosine → cos (420 * (Real.pi / 180)) = 1 / 2 :=
by 
  assume h1 : periodicity_cosine (60 * (Real.pi / 180))
  assume h2 : special_angle_cosine
  sorry

end cos_420_l802_802041


namespace circle_properties_l802_802134

noncomputable def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
noncomputable def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y - 12 = 0

theorem circle_properties :
  (∀ (x y : ℝ), circle1 x y → circle2 x y → (x - y + 2 = 0)) ∧
  (let C1_center := (0 : ℝ, 0 : ℝ) in
   let C2_center := (2 : ℝ, -2 : ℝ) in
   let dist := (2 * (4.sqrt)) in
   2 * (5.sqrt - 2) < dist ∧ dist < 2 * (5.sqrt + 2)) ∧
  (let l := 2 * ((2^2 - (2.sqrt)).sqrt) in
   l = 2 * (2.sqrt))
:= sorry

end circle_properties_l802_802134


namespace product_of_solutions_l802_802936

theorem product_of_solutions (x : ℝ) (h : |x| = 3 * (|x| - 2)) : (subs := (|x| == 3 ->  (x = 3)  ∨ (x = -3)): 
solution ( ∀ x:solution ∧ x₁ * x₂= 3 * (-3) )  : -9)   := 
sorry

end product_of_solutions_l802_802936


namespace area_enclosed_by_S_l802_802230

noncomputable def four_presentable (z : ℂ) : Prop :=
  ∃ (w : ℂ), complex.abs w = 4 ∧ z = w - (1 / w)

def S : set ℂ := {z | four_presentable z}

theorem area_enclosed_by_S :
  Complex.plane_area S = (255 / 16) * Real.pi :=
sorry

end area_enclosed_by_S_l802_802230


namespace sum_a_i_l802_802622

theorem sum_a_i (a : Fin 2025 → ℝ)
  (h : ∀ x : ℝ, (1 + x) * (1 - 2 * x)^2023 = ∑ i in Finset.range 2025, a i * x^i) :
  ∑ i in Finset.range 2024, a (i+1) = -3 := by
  sorry

end sum_a_i_l802_802622


namespace sequences_count_eq_fib_l802_802209

open Nat

def num_special_sequences (n : ℕ) : ℕ :=
  (fibonacci (n + 2)) - 1

theorem sequences_count_eq_fib (n : ℕ) (h : 0 < n) :
  ∀ seq : List ℕ, 
    (∀ i, i < seq.length → odd i → odd (seq.nthLe i h)) ∧ 
    (∀ i, i < seq.length → even i → even (seq.nthLe i h)) → 
    seq.length ≤ n →
  seq.count <| λ x, (1 ≤ x ∧ x ≤ n) ∧ List.sorted (≤) seq =
  num_special_sequences n := 
sorry

end sequences_count_eq_fib_l802_802209


namespace estimate_max_p_i_l802_802773

theorem estimate_max_p_i : ∀ i : ℕ, (1 ≤ i ∧ i ≤ 2026) → 
  (Prime 2027) ∧ 
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ 2026 → ∃ p_i : ℕ, Prime p_i ∧ p_i ≡ i [MOD 2027]) →
  ∃E : ℕ, E = 113779 ∧ E = max (Finset.image (fun i => Classical.choose (exists_minimal_prime_mod i 2027)) (Finset.range 2026)) :=
begin
  sorry
end

end estimate_max_p_i_l802_802773


namespace upperclassmen_participating_in_sports_l802_802171

theorem upperclassmen_participating_in_sports :
  ∃ (f u : ℕ),
    f + u = 600 ∧
    0.65 * f + 0.25 * u = 237 ∧
    0.75 * u = 287 :=
by
  sorry

end upperclassmen_participating_in_sports_l802_802171


namespace distinct_count_floor_squares_l802_802515

theorem distinct_count_floor_squares :
  (Finset.image (λ n : ℕ, (n ^ 2 / 2000 : ℝ)) (Finset.range 1000)).card = 501 := sorry

end distinct_count_floor_squares_l802_802515


namespace det_relationship_not_determined_l802_802090

variables {Ω : Type} [prob_space : MeasureTheory.ProbabilitySpace Ω] (A B : set Ω)

noncomputable def is_complementary (A B : set Ω) : Prop :=
  MeasureTheory.MeasureTheory.Probability.measure_theory.measure (A ∪ B) = 1 ∧ MeasureTheory.MeasureTheory.measure A + MeasureTheory.MeasureTheory.measure B = 1

noncomputable def is_mutually_exclusive (A B : set Ω) : Prop :=
  MeasureTheory.MeasureTheory.measure (A ∩ B) = 0

theorem det_relationship_not_determined (h : MeasureTheory.MeasureTheory.measure (A ∪ B) = 1 ∧ MeasureTheory.MeasureTheory.measure A + MeasureTheory.MeasureTheory.measure B = 1) :
  ¬ (is_complementary A B ∧ is_mutually_exclusive A B) :=
sorry

end det_relationship_not_determined_l802_802090


namespace card_pairs_satisfying_conditions_l802_802432

theorem card_pairs_satisfying_conditions :
  ∃ (s : Finset (ℕ × ℕ)),
    (∀ p ∈ s, p.1 ≠ p.2 ∧ (p.1 = p.2 + 11 ∨ p.2 = p.1 + 11) ∧ ((p.1 * p.2) % 5 = 0))
    ∧ s.card = 15 :=
begin
  sorry
end

end card_pairs_satisfying_conditions_l802_802432


namespace sum_of_common_divisors_60_18_l802_802537

def positive_divisors (n : ℕ) : List ℕ :=
  List.filter (λ d, n % d = 0) (List.range (n+1))

def common_divisors (m n : ℕ) : List ℕ :=
  List.filter (λ d, d ∈ positive_divisors m) (positive_divisors n)

theorem sum_of_common_divisors_60_18 : 
  List.sum (common_divisors 60 18) = 12 := by
  sorry

end sum_of_common_divisors_60_18_l802_802537


namespace number_of_sequences_l802_802180

-- Define the number of targets and their columns
def targetSequence := ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']

-- Define our problem statement
theorem number_of_sequences :
  (List.permutations targetSequence).length = 4200 := by
  sorry

end number_of_sequences_l802_802180


namespace leonardo_sleep_fraction_l802_802690

theorem leonardo_sleep_fraction (h : 60 ≠ 0) : (12 / 60 : ℚ) = (1 / 5 : ℚ) :=
by
  sorry

end leonardo_sleep_fraction_l802_802690


namespace evaluate_nested_function_l802_802710

def f (x : ℝ) : ℝ :=
  if x < 1 then 1 + Real.log (2 - x) / Real.log 2 
  else Real.exp ((x - 1) * Real.log 2)

theorem evaluate_nested_function :
  f (f (-2)) = 4 :=
by
  sorry

end evaluate_nested_function_l802_802710


namespace cards_difference_product_divisibility_l802_802457

theorem cards_difference_product_divisibility :
  (∃ pairs : list (ℕ × ℕ), 
    (∀ p ∈ pairs, 1 ≤ p.1 ∧ p.1 ≤ 50 ∧ 1 ≤ p.2 ∧ p.2 ≤ 50 ∧ 
      (p.1 - p.2 = 11 ∨ p.2 - p.1 = 11) ∧ 
      (p.1 * p.2) % 5 = 0) ∧ 
    pairs.length = 15) := 
sorry

end cards_difference_product_divisibility_l802_802457


namespace satisfies_equation_l802_802749

noncomputable def y (b x : ℝ) : ℝ := (b + x) / (1 + b * x)

theorem satisfies_equation (b x : ℝ) :
  let y_val := y b x
  let y_prime := (1 - b^2) / (1 + b * x)^2
  y_val - x * y_prime = b * (1 + x^2 * y_prime) :=
by
  sorry

end satisfies_equation_l802_802749


namespace fence_poles_count_l802_802375

def length_path : ℕ := 900
def length_bridge : ℕ := 42
def distance_between_poles : ℕ := 6

theorem fence_poles_count :
  2 * (length_path - length_bridge) / distance_between_poles = 286 :=
by
  sorry

end fence_poles_count_l802_802375


namespace fence_pole_count_l802_802372

-- Define the conditions
def path_length : ℕ := 900
def bridge_length : ℕ := 42
def pole_spacing : ℕ := 6

-- Define the goal
def total_poles : ℕ := 286

-- The statement to prove
theorem fence_pole_count :
  let total_length_to_fence := (path_length - bridge_length)
  let poles_per_side := total_length_to_fence / pole_spacing
  let total_poles_needed := 2 * poles_per_side
  total_poles_needed = total_poles :=
by
  sorry

end fence_pole_count_l802_802372


namespace min_value_of_expression_l802_802039

theorem min_value_of_expression (n : ℕ) (h : n > 0) : (n / 3 + 27 / n) ≥ 6 :=
by {
  -- Proof goes here but is not required in the statement
  sorry
}

end min_value_of_expression_l802_802039


namespace car_distance_difference_l802_802842

theorem car_distance_difference :
  ∀ (speedA speedB time : ℕ), 
  speedA = 60 →
  speedB = 45 →
  time = 5 →
  (speedA * time) - (speedB * time) = 75 :=
by
  intros speedA speedB time hA hB ht
  rw [hA, hB, ht]
  exact rfl

end car_distance_difference_l802_802842


namespace product_of_ratios_l802_802862

variable (A B C D E F G H A' B' C' D' : Point)
variable (AE EB BF FC CG GD DH HA : ℝ)

theorem product_of_ratios (h1 : AE / EB = (dist A A') / (dist B B'))
                         (h2 : BF / FC = (dist B B') / (dist C C'))
                         (h3 : CG / GD = (dist C C') / (dist D D'))
                         (h4 : DH / HA = (dist D D') / (dist A A')):
  (AE / EB) * (BF / FC) * (CG / GD) * (DH / HA) = 1 :=
by
  sorry

end product_of_ratios_l802_802862


namespace no_such_polynomial_l802_802742

open Polynomial

-- Definitions
def nat_polynomial (P : Polynomial ℤ) := ∀ x : ℕ, P.eval x = (P.eval 1 : ℤ)

-- Conditions
variable {P : Polynomial ℤ}
variable (P_deg : P.degree > 0)
variable (P_int_coeffs : ∀ n, P.coeff n ∈ ℤ)
variable (P_prime_val : ∃ p : ℤ, p ∈ Nat.Primes ∧ nat_polynomial P)

-- Statement to prove
theorem no_such_polynomial : ¬ ∃ P, P_deg ∧ P_int_coeffs ∧ P_prime_val := by
sorry

end no_such_polynomial_l802_802742


namespace product_of_solutions_eq_neg_nine_product_of_solutions_l802_802933

theorem product_of_solutions_eq_neg_nine :
  ∀ (x : ℝ), (|x| = 3 * (|x| - 2)) → x = 3 ∨ x = -3 :=
by
  sorry

theorem product_of_solutions :
  (∀ (x : ℝ), (|x| = 3 * (|x| - 2)) → (∃ (a b : ℝ), x = a ∨ x = b ∧ a * b = -9)) :=
by
  sorry

end product_of_solutions_eq_neg_nine_product_of_solutions_l802_802933


namespace sequence_a4_value_l802_802130

theorem sequence_a4_value :
  ∃ (a : ℕ → ℕ), (a 1 = 1) ∧ (∀ n, a (n+1) = 2 * a n + 1) ∧ (a 4 = 15) :=
by
  sorry

end sequence_a4_value_l802_802130


namespace area_triangle_PTS_l802_802253

theorem area_triangle_PTS {PQ QR PS QT PT TS : ℝ} 
  (hPQ : PQ = 4) 
  (hQR : QR = 6) 
  (hPS : PS = 2 * Real.sqrt 13) 
  (hQT : QT = 12 * Real.sqrt 13 / 13) 
  (hPT : PT = 4) 
  (hTS : TS = (2 * Real.sqrt 13) - 4) : 
  (1 / 2) * PT * QT = 24 * Real.sqrt 13 / 13 := 
by 
  sorry

end area_triangle_PTS_l802_802253


namespace repeating_decimal_base4_sum_l802_802266

theorem repeating_decimal_base4_sum (a b : ℕ) (hrelprime : Int.gcd a b = 1)
  (h4_rep : ((12 : ℚ) / (44 : ℚ)) = (a : ℚ) / (b : ℚ)) : a + b = 7 :=
sorry

end repeating_decimal_base4_sum_l802_802266


namespace coplanar_condition_l802_802202

-- Definitions representing points A, B, C, D and the origin O in a vector space over the reals
variables {V : Type*} [AddCommGroup V] [Module ℝ V] (O A B C D : V)

-- The main statement of the problem
theorem coplanar_condition (h : (2 : ℝ) • (A - O) - (3 : ℝ) • (B - O) + (7 : ℝ) • (C - O) + k • (D - O) = 0) :
  k = -6 :=
sorry

end coplanar_condition_l802_802202


namespace percent_of_y_equal_to_30_percent_of_60_percent_l802_802317

variable (y : ℝ)

theorem percent_of_y_equal_to_30_percent_of_60_percent (hy : y ≠ 0) :
  ((0.18 * y) / y) * 100 = 18 := by
  have hneq : y ≠ 0 := hy
  field_simp [hneq]
  norm_num
  sorry

end percent_of_y_equal_to_30_percent_of_60_percent_l802_802317


namespace problem_statement_l802_802224

variable {α : Type*} [preorder α]

-- Definition of function f with domain ℝ
def f (x : ℝ) : ℝ := sorry

-- Definition expressing that x ≠ 0
def x_nonzero (x : ℝ) : Prop := x ≠ 0

-- Definition expressing that x is a local maximum of f
def is_local_max (x : ℝ) (f : ℝ → ℝ) : Prop :=
∃ ε > 0, ∀ y, abs (y - x) < ε → f y ≤ f x

-- Definition expressing that -x is a local minimum of -f(-x)
def is_local_min_neg_f_neg_x (x : ℝ) (f : ℝ → ℝ) : Prop :=
∃ ε > 0, ∀ y, abs (y + x) < ε → -f (-y) ≥ -f (-x)

-- The theorem stating the problem
theorem problem_statement (x : ℝ) (h1 : x ≠ 0) (h2 : is_local_max x f) : 
  is_local_min_neg_f_neg_x x f :=
by sorry

end problem_statement_l802_802224


namespace hyperbola_standard_equation_l802_802568

-- The conditions
def circle_center : ℝ × ℝ := (5, 0)
def hyperbola_eccentricity : ℝ := sqrt 5
def hyperbola_focus_distance : ℝ := 5
def hyperbola_equation (a b : ℝ) (x y : ℝ) : Prop := 
  (x^2 / a^2) - (y^2 / b^2) = 1

-- The goal
theorem hyperbola_standard_equation : 
  ∃ (a b : ℝ), a = sqrt 5 ∧ b^2 = 20 ∧ (∀ x y : ℝ, hyperbola_equation a b x y → (x^2 / 5) - (y^2 / 20) = 1) := 
sorry

end hyperbola_standard_equation_l802_802568


namespace equivalent_proof_l802_802220

noncomputable def proof_problem : ℝ := 
  let a : ℝ := real.sqrt 14
  let b : ℝ := -real.sqrt 116
  let c : ℝ := real.sqrt 56
  let discriminant : ℝ := b^2 - 4 * a * c
  let x1 : ℝ := (b + real.sqrt discriminant) / (2 * a)
  let x2 : ℝ := (b - real.sqrt discriminant) / (2 * a)
  let term1 := abs ((1 / x1^2) - (1 / x2^2))
  let term2 := abs ((x2^2 - x1^2) / (x1^2 * x2^2))
  let term3 := abs ((x2 - x1) * (x2 + x1) / (x1^2 * x2^2))
  let term4 := real.sqrt 29 / 14

theorem equivalent_proof : 
  let a : ℝ := real.sqrt 14
  let b : ℝ := -real.sqrt 116
  let c : ℝ := real.sqrt 56
  let discriminant : ℝ := b^2 - 4 * a * c
  let x1 : ℝ := (b + real.sqrt discriminant) / (2 * a)
  let x2 : ℝ := (b - real.sqrt discriminant) / (2 * a)
  abs ((1 / x1^2) - (1 / x2^2)) = real.sqrt 29 / 14 := 
  sorry

end equivalent_proof_l802_802220


namespace find_b_for_continuity_l802_802223

noncomputable def g (b : ℝ) (x : ℝ) : ℝ :=
if x ≤ 5 then 4 * x^2 - 5 else b * x + 3

theorem find_b_for_continuity (b : ℝ) :
  (∀ x : ℝ, g b x = if x ≤ 5 then 4 * x^2 - 5 else b * x + 3) ∧ 
  (∀ x, ∀ y, (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 5) < δ → abs (g b x - y) < ε)
  → g b 5 = g b 5 ∧ b = 92 / 5) :=
begin
  sorry
end

end find_b_for_continuity_l802_802223


namespace calculate_difference_of_reciprocal_squares_l802_802218

theorem calculate_difference_of_reciprocal_squares
  (x1 x2 : ℝ)
  (h1 : x1 * x1 * sqrt 14 - x1 * sqrt 116 + sqrt 56 = 0)
  (h2 : x2 * x2 * sqrt 14 - x2 * sqrt 116 + sqrt 56 = 0) :
  abs ((1 / (x1 * x1)) - (1 / (x2 * x2))) = sqrt 29 / 14 :=
sorry

end calculate_difference_of_reciprocal_squares_l802_802218


namespace min_total_cost_with_both_measures_l802_802128

theorem min_total_cost_with_both_measures :
  ∀ (prob_event : ℝ) (loss : ℝ) (cost_A cost_B : ℝ) 
    (prob_not_event_A prob_not_event_B : ℝ),
  prob_event = 0.3 → 
  loss = 4 * 10^6 → 
  cost_A = 450000 →
  cost_B = 300000 →
  prob_not_event_A = 0.9 →
  prob_not_event_B = 0.85 →
  let expected_loss_no_measures := prob_event * loss,
      total_cost_no_measures := expected_loss_no_measures,
      prob_event_A := 1 - prob_not_event_A,
      expected_loss_A := prob_event_A * loss,
      total_cost_A := cost_A + expected_loss_A,
      prob_event_B := 1 - prob_not_event_B,
      expected_loss_B := prob_event_B * loss,
      total_cost_B := cost_B + expected_loss_B,
      prob_event_AB := 1 - (prob_not_event_A * prob_not_event_B),
      expected_loss_AB := prob_event_AB * loss,
      total_cost_AB := (cost_A + cost_B) + expected_loss_AB
  in total_cost_AB < total_cost_no_measures ∧ 
      total_cost_AB < total_cost_A ∧ 
      total_cost_AB < total_cost_B :=
begin
  intros prob_event loss cost_A cost_B prob_not_event_A prob_not_event_B h1 h2 h3 h4 h5 h6,
  -- Introduce expected losses and total costs for different schemes
  let expected_loss_no_measures := prob_event * loss, 
  let total_cost_no_measures := expected_loss_no_measures,
  let prob_event_A := 1 - prob_not_event_A,
  let expected_loss_A := prob_event_A * loss,
  let total_cost_A := cost_A + expected_loss_A,
  let prob_event_B := 1 - prob_not_event_B,
  let expected_loss_B := prob_event_B * loss,
  let total_cost_B := cost_B + expected_loss_B,
  let prob_event_AB := 1 - (prob_not_event_A * prob_not_event_B),
  let expected_loss_AB := prob_event_AB * loss,
  let total_cost_AB := (cost_A + cost_B) + expected_loss_AB,

  -- Checking the correctness and comparison of total_cost_AB against other schemes
  -- Note: We skip the proof here.
  sorry
end

end min_total_cost_with_both_measures_l802_802128


namespace total_digits_in_first_3003_even_ints_l802_802821

def number_of_digits (n : ℕ) : ℕ :=
  if h : n > 0 then (n.to_string.length) else 0

theorem total_digits_in_first_3003_even_ints:
  (finset.range 3003).sum (λ n, number_of_digits (2 * (n + 1))) = 11460 :=
by
  sorry

end total_digits_in_first_3003_even_ints_l802_802821


namespace induction_inequality_l802_802309

theorem induction_inequality
  (k : ℕ)
  (Hk : (finset.range (k+1)).sum (λ i, 1 / ((i + 2) ^ 2 : ℝ)) > 1 / 2 - 1 / (k + 2)) :
  (finset.range (k+2)).sum (λ i, 1 / ((i + 2) ^ 2 : ℝ)) > 1 / 2 - 1 / (k + 2) + 1 / ((k + 2) ^ 2) := 
sorry

end induction_inequality_l802_802309


namespace peter_walks_more_time_l802_802733

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

end peter_walks_more_time_l802_802733


namespace cards_difference_product_divisible_l802_802467

theorem cards_difference_product_divisible :
  let S := {n | 1 ≤ n ∧ n ≤ 50}
  let valid_pairs := {(a, b) | a ∈ S ∧ b ∈ S ∧ |a - b| = 11 ∧ (a * b) % 5 = 0}
  valid_pairs.card = 15 := 
sorry

end cards_difference_product_divisible_l802_802467


namespace fixed_point_independence_l802_802956

open EuclideanGeometry

noncomputable def fixed_point_proof (Γ : Circle) (A B C X Y D : Point) : Prop :=
  (A ∈ Γ.circle) ∧ (B ∈ Γ.circle) ∧ (C ∈ Γ.circle) ∧ (B ≠ A) ∧ (C ≠ A) ∧ 
  (∃ angleABC : ℝ, ∃ half_angleABC : ℝ, ∃ bisector_angleABC : Line,
    is_angle_bisector (∠ B A C) bisector_angleABC ∧ 
    bisector_angleABC ∩ Γ.circle = {X} ∧
    (Y = reflection X A) ∧
    (CY ∩ Γ.circle = {D}) ∧
    is_fixed_point Γ.circle D)

theorem fixed_point_independence (Γ : Circle) (A B C X Y D : Point) : fixed_point_proof Γ A B C X Y D := 
begin
  sorry
end

end fixed_point_independence_l802_802956


namespace sum_of_common_divisors_60_18_l802_802534

def positive_divisors (n : ℕ) : List ℕ :=
  List.filter (λ d, n % d = 0) (List.range (n+1))

def common_divisors (m n : ℕ) : List ℕ :=
  List.filter (λ d, d ∈ positive_divisors m) (positive_divisors n)

theorem sum_of_common_divisors_60_18 : 
  List.sum (common_divisors 60 18) = 12 := by
  sorry

end sum_of_common_divisors_60_18_l802_802534


namespace card_drawn_greater_probability_l802_802077

theorem card_drawn_greater_probability :
  let cards := {1, 2, 3, 4, 5}
  let total_draws := 5 * 5
  let favorable_events := 10
  (favorable_events.toRat / total_draws.toRat) = 2 / 5 :=
by
  sorry

end card_drawn_greater_probability_l802_802077


namespace inequality_solution_l802_802582

theorem inequality_solution (x : ℝ) : 3 * x ^ 2 + x - 2 < 0 ↔ -1 < x ∧ x < 2 / 3 :=
by
  -- The proof should factor the quadratic expression and apply the rule for solving strict inequalities
  sorry

end inequality_solution_l802_802582


namespace solve_eqn_l802_802917

noncomputable def root_expr (a b k x : ℝ) : ℝ := Real.sqrt ((a + b * Real.sqrt k)^x)

theorem solve_eqn: {x : ℝ | root_expr 3 2 2 x + root_expr 3 (-2) 2 x = 6} = {2, -2} :=
by
  sorry

end solve_eqn_l802_802917


namespace value_of_a10_l802_802594

/-- Define arithmetic sequence and properties -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d
def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) (S : ℕ → ℝ) := ∀ n : ℕ, S n = (n * (a 1 + a n) / 2)

variables {a : ℕ → ℝ} {S : ℕ → ℝ} {d : ℝ}
axiom arith_seq : arithmetic_sequence a d
axiom sum_formula : sum_of_first_n_terms a 5 S
axiom sum_condition : S 5 = 60
axiom term_condition : a 1 + a 2 + a 3 = a 4 + a 5

theorem value_of_a10 : a 10 = 26 :=
sorry

end value_of_a10_l802_802594


namespace number_of_integers_between_sqrt10_and_sqrt100_l802_802148

theorem number_of_integers_between_sqrt10_and_sqrt100 : 
  let a := Real.sqrt 10
  let b := Real.sqrt 100
  ∃ (n : ℕ), n = 6 ∧ (∀ x : ℕ, (x > a ∧ x < b) → (4 ≤ x ∧ x ≤ 9)) :=
by
  sorry

end number_of_integers_between_sqrt10_and_sqrt100_l802_802148


namespace positive_difference_median_mode_l802_802402

-- Define the dataset
def dataset : List ℕ := [21, 22, 22, 23, 23, 30, 30, 30, 35, 36, 44, 45, 46, 48, 49, 51, 52, 55, 57]

-- Define the function to compute the mode
def mode (xs : List ℕ) : ℕ :=
  xs.mode -- assuming mode function exists, otherwise implement it

-- Define the function to calculate the median
def median (xs : List ℕ) : ℕ := 
  let sorted_xs := xs.qsort (· < ·)
  sorted_xs.get! (sorted_xs.length / 2)

-- The theorem to prove the positive difference between the median and the mode is 5
theorem positive_difference_median_mode : 
  (dataset : List ℕ) → 
  abs (median dataset - mode dataset) = 5 :=
by 
  -- skipping the proof with sorry
  sorry

end positive_difference_median_mode_l802_802402


namespace integer_solutions_count_l802_802073

-- Define the function to check if the given expression is an integer
def is_integer_solution (n : ℤ) : Prop :=
  ∃ k : ℤ, (2 * n^3 - 12 * n^2 - 2 * n + 12) = k * (n^2 + 5 * n - 6)

-- Define the theorem to count the number of integer values of n for which the expression is an integer
theorem integer_solutions_count : 
  (finset.range 241).filter (λ n : ℤ, is_integer_solution (n - 120)).card = 32 :=
by sorry

end integer_solutions_count_l802_802073


namespace empty_boxes_count_l802_802080

theorem empty_boxes_count (n : Nat) (non_empty_boxes : Nat) (empty_boxes : Nat) : 
  (n = 34) ∧ (non_empty_boxes = n) ∧ (empty_boxes = -1 + 6 * n) → empty_boxes = 203 := by 
  intros
  sorry

end empty_boxes_count_l802_802080


namespace rise_in_water_level_l802_802350

-- Define the edge of the cube
def edge : ℝ := 16

-- Define the dimensions of the base of the vessel
def length_of_base : ℝ := 20
def width_of_base : ℝ := 15

-- Define the area of the base of the vessel
def A_base : ℝ := length_of_base * width_of_base

-- Define the volume of the cube
def V_cube : ℝ := edge ^ 3

-- Calculate the rise in water level
def h_increase : ℝ := V_cube / A_base

-- The theorem to prove the rise in water level equals the expected value
theorem rise_in_water_level :
  h_increase = 13.65 := by
  -- Proof is needed here
  sorry

end rise_in_water_level_l802_802350


namespace lcm_12_18_l802_802923

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l802_802923


namespace triangle_area_PQR_l802_802270

section TriangleArea

variables {a b c d : ℝ}
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
variables (hOppositeSides : (0 - c) * b - (a - 0) * d < 0)

theorem triangle_area_PQR :
  let P := (0, a)
  let Q := (b, 0)
  let R := (c, d)
  let area := (1 / 2) * (a * c + b * d - a * b)
  area = (1 / 2) * (a * c + b * d - a * b) := 
by
  sorry

end TriangleArea

end triangle_area_PQR_l802_802270


namespace cyclic_quadrilateral_radii_l802_802758

noncomputable def cyclic_quadrilateral_radius (a b c d : ℕ) (cyclic : Prop) : ℝ :=
  let e := real.sqrt (a^2 + d^2 - 2 * a * d * (-((a^2 + d^2) - (b^2 + c^2)) / (2 * (a * d + b * c))))
  280 / (2 * (real.sqrt (1 - ((-((a^2 + d^2) - (b^2 + c^2)) / (2 * (a * d + b * c)))^2))))

noncomputable def inscribed_circle_radius (a b c d : ℕ) (cyclic : Prop) : ℝ :=
  let s := a + c
  let t := (9360 * 56 / (2 * 65) + 28665 * 56 / (2 * 65))
  t / s

theorem cyclic_quadrilateral_radii
  (a b c d : ℕ) (cyclic : Prop)
  (h_ab : a = 36) (h_bc : b = 91) (h_cd : c = 315) (h_da : d = 260) : 
  cyclic_quadrilateral_radius a b c d cyclic = 162.5 ∧ inscribed_circle_radius a b c d cyclic = 140 / 3 :=
by
  intros
  sorry

end cyclic_quadrilateral_radii_l802_802758


namespace max_value_of_f_on_domain_l802_802986

noncomputable def f : ℝ → ℝ := λ x, 2 * x + 1 / x - 1

theorem max_value_of_f_on_domain :
  (∀ x : ℝ, x ≤ -2 → f x ≤ f (-2)) ∧ f (-2) = -11 / 2 :=
by
  sorry

end max_value_of_f_on_domain_l802_802986


namespace probability_of_gui_field_in_za_field_l802_802263

noncomputable def area_gui_field (base height : ℕ) : ℚ :=
  (1 / 2 : ℚ) * base * height

noncomputable def area_za_field (small_base large_base height : ℕ) : ℚ :=
  (1 / 2 : ℚ) * (small_base + large_base) * height

theorem probability_of_gui_field_in_za_field :
  let b1 := 10
  let b2 := 20
  let h1 := 10
  let base_gui := 8
  let height_gui := 5
  let za_area := area_za_field b1 b2 h1
  let gui_area := area_gui_field base_gui height_gui
  (gui_area / za_area) = (2 / 15 : ℚ) := by
    sorry

end probability_of_gui_field_in_za_field_l802_802263


namespace fractional_rep_of_0_point_345_l802_802496

theorem fractional_rep_of_0_point_345 : 
  let x := (0.3 + (0.45 : ℝ)) in
  (x = (83 / 110 : ℝ)) :=
by
  sorry

end fractional_rep_of_0_point_345_l802_802496


namespace cost_per_foot_calculation_l802_802237

def total_budget : ℝ := 120000
def side_length : ℝ := 5000
def total_perimeter : ℝ := 4 * side_length
def unfenced_length : ℝ := 1000
def length_needed : ℝ := total_perimeter - unfenced_length
def cost_per_foot : ℝ := total_budget / length_needed

theorem cost_per_foot_calculation : cost_per_foot = 120000 / (4 * 5000 - 1000) :=
by
  sorry

end cost_per_foot_calculation_l802_802237


namespace remaining_walking_time_is_30_l802_802730

-- Define all the given conditions
def total_distance_to_store : ℝ := 2.5
def distance_already_walked : ℝ := 1.0
def time_per_mile : ℝ := 20.0

-- Define the target remaining walking time
def remaining_distance : ℝ := total_distance_to_store - distance_already_walked
def remaining_time : ℝ := remaining_distance * time_per_mile

-- Prove the remaining walking time is 30 minutes
theorem remaining_walking_time_is_30 : remaining_time = 30 :=
by
  -- Formal proof would go here using corresponding Lean tactics
  sorry

end remaining_walking_time_is_30_l802_802730


namespace lcm_12_18_l802_802921

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l802_802921


namespace add_water_to_solution_l802_802369

noncomputable def current_solution_volume : ℝ := 300
noncomputable def desired_water_percentage : ℝ := 0.70
noncomputable def current_water_volume : ℝ := 0.60 * current_solution_volume
noncomputable def current_acid_volume : ℝ := 0.40 * current_solution_volume

theorem add_water_to_solution (x : ℝ) : 
  (current_water_volume + x) / (current_solution_volume + x) = desired_water_percentage ↔ x = 100 :=
by
  sorry

end add_water_to_solution_l802_802369


namespace range_of_k_l802_802562

theorem range_of_k (x : ℝ) (h1 : 0 < x) (h2 : x < 2) (h3 : x / Real.exp x < 1 / (k + 2 * x - x^2)) :
    0 ≤ k ∧ k < Real.exp 1 - 1 :=
sorry

end range_of_k_l802_802562


namespace find_analytic_expression_l802_802083

-- Define the power function
def power_function (y : ℝ → ℝ) (m : ℝ) (x : ℝ) : Prop :=
  y x = (m^2 - m - 1) * x^(m^2 - 2*m - 1/3)

-- Define the conditions
def function_is_decreasing (y : ℝ → ℝ) (x : ℝ) : Prop :=
  x > 0 → ∀ (h₀ x₀: x > 0), y x > y x₀ → x < x₀

def quadratic_equation (m : ℝ) : Prop :=
  m^2 - m - 2 = 0

-- The main theorem to prove the function is decreasing given conditions
theorem find_analytic_expression (y : ℝ → ℝ) (m : ℝ) (x : ℝ) (h₀ : x > 0)
  (h₁ : power_function y m x) (h₂ : function_is_decreasing y x) (h₃ : quadratic_equation m) : 
  y = λ x, x^(-1/3) :=
by
  sorry

end find_analytic_expression_l802_802083


namespace not_all_blue_fractions_denominator_lt_100_l802_802010

-- Conditions
def is_irreducible_fraction (a : ℚ) : Prop := 
  (∀ (x y : ℤ), a = x / y → int.gcd x y = 1)

def odd_denominator (a : ℚ) : Prop := 
  (∃ (y : ℤ), a.denom = y ∧ y % 2 = 1)

-- Five irreducible fractions with odd denominators greater than 10^10
variables {a1 a2 a3 a4 a5 : ℚ}
noncomputable def denom_gt_10_pow_10 (a : ℚ) : Prop := a.denom > 10^10

-- Blue fractions between each pair of adjacent red fractions
noncomputable def blue_fraction (x y : ℚ) : ℚ := x + y

-- Proving that not all blue fractions can have denominators ≤ 100
theorem not_all_blue_fractions_denominator_lt_100 
  (h1 : is_irreducible_fraction a1) (h2 : is_irreducible_fraction a2) (h3 : is_irreducible_fraction a3) 
  (h4 : is_irreducible_fraction a4) (h5 : is_irreducible_fraction a5)
  (o1 : odd_denominator a1) (o2 : odd_denominator a2) (o3 : odd_denominator a3) 
  (o4 : odd_denominator a4) (o5 : odd_denominator a5)
  (d1 : denom_gt_10_pow_10 a1) (d2 : denom_gt_10_pow_10 a2) (d3 : denom_gt_10_pow_10 a3) 
  (d4 : denom_gt_10_pow_10 a4) (d5 : denom_gt_10_pow_10 a5) :
  ¬(∀ (i j : ℤ) (h1 : i ≠ j), (i, j) ∈ {(1,2), (2,3), (3,4), (4,5), (5,1)} → 
    let s := blue_fraction (list.nth [a1, a2, a3, a4, a5] (i - 1)).get_or_else 0 
      (list.nth [a1, a2, a3, a4, a5] (j - 1)).get_or_else 0 in 
    s.denom ≤ 100) := sorry

end not_all_blue_fractions_denominator_lt_100_l802_802010


namespace quadrilateral_is_parallelogram_l802_802286

noncomputable theory

variables {A B C D O : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D]
          [metric_space O]
variables [ordered_add_comm_group (A → ℝ)]
variables [ordered_add_comm_group (B → ℝ)]
variables [ordered_add_comm_group (C → ℝ)]
variables [ordered_add_comm_group (D → ℝ)]
variables [ordered_add_comm_group (O → ℝ)]

structure quadrilateral (A B C D O : Type*) : Type* :=
(bisects_ac : dist A O = dist O C)
(bisects_bd : dist B O = dist O D)

theorem quadrilateral_is_parallelogram (q : quadrilateral A B C D O)
  : parallel (line_through A D) (line_through B C) ∧
    parallel (line_through A B) (line_through C D) :=
sorry

end quadrilateral_is_parallelogram_l802_802286


namespace not_axiom_l802_802913

theorem not_axiom (P Q R S : Prop)
  (B : P -> Q -> R -> S)
  (C : P -> Q)
  (D : P -> R)
  : ¬ (P -> Q -> S) :=
sorry

end not_axiom_l802_802913


namespace average_age_at_youngest_birth_l802_802265

-- Defining the problem conditions
variables (average_age : ℝ) (number_of_members : ℕ) (age_youngest : ℝ) (age_second_youngest : ℝ)
variables (total_age : ℝ) (total_age_remaining : ℝ) (total_age_remaining_3_years_ago : ℝ) (age_youngest_birth : ℝ)

-- Providing the specific values from the conditions
def average_age_value : average_age = 25 := by sorry
def number_of_members_value : number_of_members = 7 := by sorry
def age_youngest_value : age_youngest = 3 := by sorry
def age_second_youngest_value : age_second_youngest = 8 := by sorry

-- Calculations from the given conditions
def total_age_calc : total_age = average_age * number_of_members := by sorry
def total_age_remaining_calc : total_age_remaining = total_age - age_youngest - age_second_youngest := by sorry
def total_age_remaining_3_years_ago_calc : total_age_remaining_3_years_ago = total_age_remaining - (age_youngest * (number_of_members - 2)) := by sorry
def age_youngest_birth_calc : age_youngest_birth = total_age_remaining_3_years_ago / (number_of_members - 1) := by sorry

-- Main theorem to prove
theorem average_age_at_youngest_birth :
  average_age_value ∧ number_of_members_value ∧ age_youngest_value ∧ age_second_youngest_value →
  age_youngest_birth = 24.83 :=
by sorry

end average_age_at_youngest_birth_l802_802265


namespace evan_initial_money_l802_802903

/-
David found $12 on the street. He then gave it to his friend Evan who has some money and needed to buy a watch worth $20.
After receiving the money from David, Evan still needs $7.
How much money did Evan have initially?
-/

noncomputable def initial_money (total_cost money_received remaining_needed : ℕ) := total_cost - money_received + remaining_needed

theorem evan_initial_money :
  ∀ (david_money evan_needed more_needed : ℕ),
    david_money = 12 → evan_needed = 20 → more_needed = 7 →
    initial_money evan_needed david_money more_needed = 13 :=
by
  intros david_money evan_needed more_needed h1 h2 h3
  znormalize at h1 h2 h3 ⟩ sorry


end evan_initial_money_l802_802903


namespace average_gas_mileage_l802_802004

theorem average_gas_mileage 
  (distance_to_city : ℕ) (mileage_sedan : ℕ)
  (distance_return : ℕ) (mileage_truck : ℕ) :
  distance_to_city = 100 →
  mileage_sedan = 25 →
  distance_return = 150 →
  mileage_truck = 15 →
  (250 : ℕ) / ((100 / 25) + (150 / 15)) = 18 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end average_gas_mileage_l802_802004


namespace only_triplet_l802_802551

noncomputable def unique_triplet (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧
  Nat.coprime a b ∧ Nat.coprime a c ∧ Nat.coprime b c ∧
  (a + b) % c = 0 ∧
  (a + c) % b = 0 ∧
  (b + c) % a = 0

theorem only_triplet {a b c : ℕ} :
  unique_triplet a b c → (a = 1 ∧ b = 2 ∧ c = 3) := 
sorry

end only_triplet_l802_802551


namespace not_divides_5_l802_802210

noncomputable def n : ℕ := 33

theorem not_divides_5 (n : ℕ) (h : 1 / 5 + 1 / 3 + 1 / 11 + 1 / n ∈ ℤ) : ¬ (5 ∣ n) :=
by sorry

end not_divides_5_l802_802210


namespace darrel_birds_count_l802_802233

theorem darrel_birds_count :
  ∀ (marcus birds : Nat) (humphrey birds : Nat) (avg birds : Nat) (num observers : Nat),
  marcus birds = 7 →
  humphrey birds = 11 →
  avg birds = 9 →
  num observers = 3 →
  ∃ darrel birds : Nat, avg birds * num observers = marcus birds + humphrey birds + darrel birds ∧ darrel birds = 9 :=
by
  intros marcus_birds humphrey_birds avg_birds num_observers marcus_birds_eq humphrey_birds_eq avg_birds_eq num_observers_eq
  use 9
  sorry

end darrel_birds_count_l802_802233


namespace surveys_completed_total_l802_802007

variable (regular_rate cellphone_rate total_earnings cellphone_surveys total_surveys : ℕ)
variable (h_regular_rate : regular_rate = 10)
variable (h_cellphone_rate : cellphone_rate = 13) -- 30% higher than regular_rate
variable (h_total_earnings : total_earnings = 1180)
variable (h_cellphone_surveys : cellphone_surveys = 60)
variable (h_total_surveys : total_surveys = cellphone_surveys + (total_earnings - (cellphone_surveys * cellphone_rate)) / regular_rate)

theorem surveys_completed_total :
  total_surveys = 100 :=
by
  sorry

end surveys_completed_total_l802_802007


namespace sum_of_coeffs_l802_802276

theorem sum_of_coeffs 
  (a b c d e x : ℝ)
  (h : (729 * x ^ 3 + 8) = (a * x + b) * (c * x ^ 2 + d * x + e)) :
  a + b + c + d + e = 78 :=
sorry

end sum_of_coeffs_l802_802276


namespace mutually_exclusive_and_not_opposite_l802_802078

def event (α : Type) := set α
variable {α : Type}

noncomputable def balls := {b1 : α // b1 = "red"} ∪ {b2 : α // b2 = "white"}

-- Conditions: There are 2 red balls and 2 white balls in the bag.
noncomputable def bag : set (event α) := {{"red", "red"}, {"red", "white"}, {"white", "white"}}

-- Definitions of events based on selections.
noncomputable def exactly_one_white_ball (X : event α) : Prop :=
  (X ∈ bag) ∧ (X = {"red", "white"})

noncomputable def exactly_two_white_balls (X : event α) : Prop :=
  (X ∈ bag) ∧ (X = {"white", "white"})

noncomputable def mutually_exclusive (P Q : event α → Prop) : Prop :=
  ∀ X, P X → Q X → false

noncomputable def not_opposite (P Q : event α → Prop) : Prop :=
  ∃ X, P X ∨ Q X

-- The proof statement
theorem mutually_exclusive_and_not_opposite :
  ∀ (X : event α),
    (mutually_exclusive exactly_one_white_ball exactly_two_white_balls) ∧
    (not_opposite exactly_one_white_ball exactly_two_white_balls) :=
by
  sorry

end mutually_exclusive_and_not_opposite_l802_802078


namespace range_of_a_for_A_supseteq_B_l802_802614

variable {a x : ℝ}

def A (a : ℝ) : Set ℝ := { x | a - 1 ≤ x ∧ x ≤ a + 2 }
def B : Set ℝ := { x | abs (x - 4) < 1 }

theorem range_of_a_for_A_supseteq_B :
  (A a) ⊇ B → 3 ≤ a ∧ a ≤ 4 :=
begin
  intro h,
  sorry
end

end range_of_a_for_A_supseteq_B_l802_802614


namespace smallest_fraction_numerator_l802_802874

theorem smallest_fraction_numerator :
  ∃ (a b : ℕ), a ≥ 10 ∧ a ≤ 99 ∧ b ≥ 10 ∧ b ≤ 99 ∧ (4 * b < 9 * a) ∧ 
  (∀ (a' b' : ℕ), a' ≥ 10 ∧ a' ≤ 99 ∧ b' ≥ 10 ∧ b' ≤ 99 ∧ (4 * b' < 9 * a') → b * a' ≥ a * b') ∧ a = 41 :=
sorry

end smallest_fraction_numerator_l802_802874


namespace train_length_is_correct_l802_802006

def length_of_train (train_speed_kmph : ℝ) (man_speed_kmph : ℝ) (time_s : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let man_speed_mps := man_speed_kmph * 1000 / 3600
  let relative_speed := train_speed_mps + man_speed_mps
  relative_speed * time_s

theorem train_length_is_correct :
  length_of_train 60 6 9 = 165.06 :=
by
  sorry

end train_length_is_correct_l802_802006


namespace dog_expected_jumps_is_8_l802_802379

noncomputable def expected_jumps : ℝ :=
  let p1 := 1 / 3 -- Probability for the first trainer
  let p2 := 1 / 3 -- Probability for the second trainer
  let p3 := 1 / 3 -- Probability for the third trainer
  let j2 := 5 -- Jumps for the second trainer
  let j3 := 3 -- Jumps for the third trainer
  let E := (p1 * 0) + (p2 * (j2 + E)) + (p3 * (j3 + E)) in
  E

theorem dog_expected_jumps_is_8 : expected_jumps = 8 :=
sorry

end dog_expected_jumps_is_8_l802_802379


namespace area_of_triangle_POM_l802_802977

-- Definition of the parametric equations
def curveC_x (t : ℝ) : ℝ := 2 * (√2) * t^2
def curveC_y (t : ℝ) : ℝ := 4 * t

-- Definition of point M
def pointM : ℝ × ℝ := (√2, 0)

-- Definition of distance PM
def distance_PM (x y : ℝ) : ℝ := (x - (√2))^2 + y^2

-- Definition of area of triangle = 1/2 * |OM| * |b|
def area_triangle (OM b : ℝ) : ℝ := 1/2 * OM * b

-- Proof problem: Calculate the area of △POM
-- Find the coordinates of P satisfying |PM| = 4√2 and point on curve C,
-- then show that the area of the triangle is 2√3.
theorem area_of_triangle_POM (t : ℝ) 
  (h1 : curveC_x t = 3 * √2) 
  (h2 : distance_PM (curveC_x t) (curveC_y t) = 4 * 4 * 2) 
  (M : ℝ × ℝ := pointM) 
  (P : ℝ × ℝ := (curveC_x t, curveC_y t)) :
  area_triangle (√2) (|curveC_y t|) = 2 * √3 := by
  sorry

end area_of_triangle_POM_l802_802977


namespace ball_return_count_l802_802845

theorem ball_return_count (m n : ℕ) (h1 : 2 ≤ m) :
  ∃ a_n : ℕ, a_n = (m-1)^n / m + (-1)^n * (m-1) / m :=
by
  sorry

end ball_return_count_l802_802845


namespace function_properties_l802_802601

def f (x : ℝ) : ℝ := Math.exp x - Math.exp (-x)

theorem function_properties : (∀ x, f (-x) = -f x) ∧ (∀ ⦃x y : ℝ⦄, x < y → f x < f y) :=
by
  sorry

end function_properties_l802_802601


namespace probability_three_digit_divisible_by_5_l802_802297

theorem probability_three_digit_divisible_by_5 (M : ℕ) (hM1 : 100 ≤ M ∧ M ≤ 999) (hM2 : M % 10 = 7) : 
  ∃ p, p = 0 ∧ (Pr (M % 5 = 0) = p) :=
sorry

end probability_three_digit_divisible_by_5_l802_802297


namespace lcm_12_18_l802_802926

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l802_802926


namespace sum_of_arith_seq_l802_802709

noncomputable def f (x : ℝ) : ℝ := (x - 3)^3 + x - 1

def is_arith_seq (a : ℕ → ℝ) : Prop := 
  ∃ d ≠ 0, ∀ n : ℕ, a (n + 1) - a n = d

theorem sum_of_arith_seq (a : ℕ → ℝ) (h_a : is_arith_seq a)
  (h_f_sum : f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) + f (a 7) = 14) :
  (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) + (a 7) = 21 :=
sorry

end sum_of_arith_seq_l802_802709


namespace find_k_l802_802996

-- Define the general quadratic equation
def quadratic_eq (a b c : ℝ) (x : ℂ) : Prop :=
  x^2 + a * x + b = 0

-- Define the root and its conjugate
def has_root (a b c root : ℂ) (x : ℂ) : Prop :=
  quadratic_eq a b c root ∧ quadratic_eq a b c x.conj

-- The main theorem stating the equivalence proof problem
theorem find_k (a : ℝ) (k : ℝ) (i : ℂ) (h_imag_unit: i^2 = -1)
  (has_root_a3i: has_root 1 4 k (a + 3 * i))
  (ha_real: a.re = a) (ha_im: a.im = 0) : k = 13 :=
sorry

end find_k_l802_802996


namespace recurring_decimal_to_fraction_l802_802503

theorem recurring_decimal_to_fraction (h1: (0.3 + 0.\overline{45} : ℝ) = (0.3\overline{45} : ℝ))
    (h2: (0.\overline{45} : ℝ) = (5 / 11 : ℝ))
    (h3: (0.3 : ℝ) = (3 / 10 : ℝ)) : (0.3\overline{45} : ℝ) = (83 / 110 : ℝ) :=
by
    sorry

end recurring_decimal_to_fraction_l802_802503


namespace absolute_value_and_power_sum_l802_802027

theorem absolute_value_and_power_sum :
  |(-4 : ℤ)| + (3 - Real.pi)^0 = 5 := by
  sorry

end absolute_value_and_power_sum_l802_802027


namespace divisibility_by_5_l802_802058

theorem divisibility_by_5 (x y : ℤ) : (x^2 - 2 * x * y + 2 * y^2) % 5 = 0 ∨ (x^2 + 2 * x * y + 2 * y^2) % 5 = 0 ↔ (x % 5 = 0 ∧ y % 5 = 0) ∨ (x % 5 ≠ 0 ∧ y % 5 ≠ 0) := 
by
  sorry

end divisibility_by_5_l802_802058


namespace right_triangle_ab_area_l802_802651

theorem right_triangle_ab_area (a b T C : ℝ)
  (h_right_triangle : ∀ (a b c : ℝ), c = a + b)
  (h_altitude_split : ∃ h : ℝ, h = real.sqrt (a * b))
  (h_triangle_area : T = 1 / 2 * (a + b) * real.sqrt (a * b))
  (h_circle_area : C = real.pi * (a + b) ^ 2 / 4) :
  a * b = real.pi * (T ^ 2) / C :=
sorry

end right_triangle_ab_area_l802_802651


namespace cubic_equation_l802_802258

variable (a b : ℝ)

def z : ℝ := (a + sqrt (a ^ 2 + b ^ 3)) ^ (1 / 3) - (sqrt (a ^ 2 + b ^ 3) - a) ^ (1 / 3)

theorem cubic_equation (a b : ℝ) :
  z a b ^ 3 + 3 * b * z a b - 2 * a = 0 :=
sorry

end cubic_equation_l802_802258


namespace ensure_nonempty_intersection_l802_802558

def M (x : ℝ) : Prop := x ≤ 1
def N (x : ℝ) (p : ℝ) : Prop := x > p

theorem ensure_nonempty_intersection (p : ℝ) : (∃ x : ℝ, M x ∧ N x p) ↔ p < 1 :=
by
  sorry

end ensure_nonempty_intersection_l802_802558


namespace Xiaogang_start_page_l802_802831

theorem Xiaogang_start_page (book_pages: ℕ) (first_day_fraction: ℚ) 
                            (second_day_ratio_numerator: ℕ) (second_day_ratio_denominator: ℕ) :
  book_pages = 96 ∧ first_day_fraction = 1 / 8 ∧ second_day_ratio_numerator = 2 ∧ second_day_ratio_denominator = 3 
  → 
  let pages_first_day := book_pages * first_day_fraction;
      pages_second_day := pages_first_day * (second_day_ratio_numerator / second_day_ratio_denominator);
      total_pages_first_two_days := pages_first_day + pages_second_day;
      start_page_third_day := total_pages_first_two_days + 1
  in start_page_third_day = 21 := 
by {
  intros h,
  let pages_first_day := book_pages * first_day_fraction,
  let pages_second_day := pages_first_day * (second_day_ratio_numerator / second_day_ratio_denominator),
  let total_pages_first_two_days := pages_first_day + pages_second_day,
  let start_page_third_day := total_pages_first_two_days + 1,
  rw [eq_iff_iff],
  cases h with h1 h_rest,
  cases h_rest with h2 h3,
  rw [h1, h2, h3],
  have pfd : pages_first_day = 12 := by simp [pages_first_day, show (96 : ℚ) * (1 / 8) = 12 by norm_num],
  have psd : pages_second_day = 8 := by simp [pages_second_day, pfd, show (12 : ℚ) * (2 / 3) = 8 by norm_num],
  have tpd : total_pages_first_two_days = 20 := by simp [total_pages_first_two_days, pfd, psd],
  exact tpd + 1
}

end Xiaogang_start_page_l802_802831


namespace count_integers_between_sqrts_l802_802142

theorem count_integers_between_sqrts (a b : ℝ) (h1 : a = 10) (h2 : b = 100) :
  let lower_bound := Int.ceil (Real.sqrt a),
      upper_bound := Int.floor (Real.sqrt b) in
  (upper_bound - lower_bound + 1) = 7 := 
by
  rw [h1, h2]
  let lower_bound := Int.ceil (Real.sqrt 10)
  let upper_bound := Int.floor (Real.sqrt 100)
  have h_lower : lower_bound = 4 := by sorry
  have h_upper : upper_bound = 10 := by sorry
  rw [h_lower, h_upper]
  norm_num
  sorry

end count_integers_between_sqrts_l802_802142


namespace smallest_single_discount_more_advantageous_l802_802071

theorem smallest_single_discount_more_advantageous (n : ℕ) :
  let discount1 := 22.56
  let discount2 := 22.1312
  let discount3 := 28
  n > discount1 ∧ n > discount2 ∧ n > discount3 → n = 29 :=
by
  sorry

end smallest_single_discount_more_advantageous_l802_802071


namespace sewage_purification_solution_l802_802889

def isosceles_right_triangle_purification_tower
(sewage_purification : Type) (outlets : Fin 5 → ℝ) : Prop :=
  ∀ (sewage : sewage_purification),
  ∃ (flow_path : List (sewage_purification → ℝ)),
    (∀ t ∈ flow_path, t = 1 / 2) ∧
    (flow_path.length = 4) ∧
    (∀ o, outlets o ∈ set.range flow_path)

theorem sewage_purification_solution 
(sewage_purification : Type) 
(outlets : Fin 5 → ℝ)
(h_tower : isosceles_right_triangle_purification_tower sewage_purification outlets) :
  (outlets 1 = outlets 3) ∧
  (outlets 2 = outlets 4) ∧
  ((outlets 0 / outlets 1 = 1 / 4) ∧ (outlets 0 / outlets 2 = 1 / 6)) ∧
  (∃ (t_fast t_slow : ℝ), t_slow / t_fast = 8) :=
sorry

end sewage_purification_solution_l802_802889


namespace tan_15_degree_l802_802069

theorem tan_15_degree : tan (15 * Real.pi / 180) = 2 - Real.sqrt 3 := by
  sorry

end tan_15_degree_l802_802069


namespace monomial_pattern_2023rd_term_l802_802242

theorem monomial_pattern_2023rd_term : 
  ∀ (n : ℕ), (n ≥ 1) → 
  (n = 2023 →
  (-1)^(n + 1) * 2 * n * x^n = -4046 * x^2023) := 
by
  intros n hn h2023
  sorry

end monomial_pattern_2023rd_term_l802_802242


namespace parabola_expression_area_of_triangle_ABP_l802_802590

-- Definition of the given parabola passing through points A and B
def parabola (x : ℝ) : ℝ := -x^2 + 4 * x + 5

-- Point A
def A : ℝ × ℝ := (-1, 0)

-- Point B
def B : ℝ × ℝ := (5, 0)

-- Point P
def P : ℝ × ℝ := (2, 9)

-- Proof that verifies the parabola expression is correct
theorem parabola_expression :
  (parabola (-1) = 0) ∧ (parabola 5 = 0) :=
by {
  sorry -- The proof steps will go here
}

-- The area of triangle ABP
def area_of_triangle : ℝ := 1 / 2 * 6 * 9

-- Proof that verifies the area of triangle ABP
theorem area_of_triangle_ABP :
  area_of_triangle = 27 :=
by {
  sorry -- The proof steps will go here
}

end parabola_expression_area_of_triangle_ABP_l802_802590


namespace angle_equality_l802_802574

-- Definitions for Points and Midpoints

structure Point (α : Type) := 
(x : α) (y : α)

def midpoint (α : Type) [has_add α] [has_div α] (P Q : Point α) : Point α :=
{ x := (P.x + Q.x) / 2,
  y := (P.y + Q.y) / 2 }

-- Statement of the theorem

theorem angle_equality (α : Type) [field α] {O1 O2 A P1 P2 Q1 Q2 M1 M2 : Point α} :
  let midpoint_P1Q1 := midpoint α P1 Q1,
      midpoint_P2Q2 := midpoint α P2 Q2 in
  A = P1 ∧ A = P2 ∧      -- A is the intersection point of the two circles
  M1 = midpoint_P1Q1 ∧  -- M1 is the midpoint of P1 Q1
  M2 = midpoint_P2Q2 ∧  -- M2 is the midpoint of P2 Q2
  (common_external_tangent P1 P2 A O1 O2) ∧
  (common_external_tangent Q1 Q2 A O1 O2) →
  angle O1 A O2 = angle M1 A M2 :=
sorry  -- Proof to be provided later

end angle_equality_l802_802574


namespace ln_abs_a_even_iff_a_eq_zero_l802_802828

theorem ln_abs_a_even_iff_a_eq_zero (a : ℝ) :
  (∀ x : ℝ, Real.log (abs (x - a)) = Real.log (abs (-x - a))) ↔ (a = 0) :=
by
  sorry

end ln_abs_a_even_iff_a_eq_zero_l802_802828


namespace tree_height_is_12_l802_802868

-- Let h be the height of the tree in meters.
def height_of_tree (h : ℝ) : Prop :=
  ∃ h, (h / 8 = 150 / 100) → h = 12

theorem tree_height_is_12 : ∃ h : ℝ, height_of_tree h :=
by {
  sorry
}

end tree_height_is_12_l802_802868


namespace initial_bees_l802_802299

variable (B : ℕ)

theorem initial_bees (h : B + 10 = 26) : B = 16 :=
by sorry

end initial_bees_l802_802299


namespace inverse_100_mod_101_l802_802049

theorem inverse_100_mod_101 : (100 * 100) % 101 = 1 :=
by
  -- Proof can be provided here.
  sorry

end inverse_100_mod_101_l802_802049


namespace min_a_for_single_zero_in_domain_l802_802165

noncomputable def f (a : ℝ) : ℝ → ℝ :=
λ x, if x ≤ 0 then x + 3^x else (1/3) * x^3 - 4 * x + a

theorem min_a_for_single_zero_in_domain : ∃ (a : ℤ), (f a).zeroes.count = 1 ∧ a = 6 := 
sorry

end min_a_for_single_zero_in_domain_l802_802165


namespace find_first_term_in_geometric_sequence_l802_802291

-- Define the factorial function for convenience
def fact : ℕ → ℝ
| 0       := 1
| (n + 1) := (n + 1) * fact n

theorem find_first_term_in_geometric_sequence :
  ∃ (a r : ℝ), (a * r^5 = fact 7) ∧ (a * r^8 = fact 9) ∧ (a ≈ 3.303) :=
begin
  sorry
end

end find_first_term_in_geometric_sequence_l802_802291


namespace cards_difference_product_divisible_l802_802470

theorem cards_difference_product_divisible :
  let S := {n | 1 ≤ n ∧ n ≤ 50}
  let valid_pairs := {(a, b) | a ∈ S ∧ b ∈ S ∧ |a - b| = 11 ∧ (a * b) % 5 = 0}
  valid_pairs.card = 15 := 
sorry

end cards_difference_product_divisible_l802_802470


namespace smallest_fraction_divisible_l802_802900

theorem smallest_fraction_divisible (n1 n2 n3 n4 n5 d1 d2 d3 d4 d5 : ℕ) 
  (h1 : n1 = 6) (h2 : n2 = 5) (h3 : n3 = 10) (h4 : n4 = 8) (h5 : n5 = 11)
  (h6 : d1 = 7) (h7 : d2 = 14) (h8 : d3 = 21) (h9 : d4 = 15) (h10 : d5 = 28) :
  1 / nat.lcm d1 (nat.lcm d2 (nat.lcm d3 (nat.lcm d4 d5))) = (1 : ℚ) / 420 :=
by sorry

end smallest_fraction_divisible_l802_802900


namespace count_valid_pairs_l802_802446

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

def difference_is_11 (a b : ℕ) : Prop := abs (a - b) = 11

def valid_card (a b : ℕ) : Prop :=
  a ∈ finset.range 51 ∧ b ∈ finset.range 51 ∧
  difference_is_11 a b ∧ (is_divisible_by_5 a ∨ is_divisible_by_5 b)

theorem count_valid_pairs : finset.card {p : ℕ × ℕ | valid_card p.1 p.2 ∧ p.1 < p.2}.to_finset = 15 := 
  sorry

end count_valid_pairs_l802_802446


namespace julie_total_lettuce_pounds_l802_802686

theorem julie_total_lettuce_pounds :
  ∀ (cost_green cost_red cost_per_pound total_cost total_pounds : ℕ),
  cost_green = 8 →
  cost_red = 6 →
  cost_per_pound = 2 →
  total_cost = cost_green + cost_red →
  total_pounds = total_cost / cost_per_pound →
  total_pounds = 7 :=
by
  intros cost_green cost_red cost_per_pound total_cost total_pounds h1 h2 h3 h4 h5
  rw [h1, h2] at h4
  rw [h3] at h5
  norm_num at h4
  norm_num at h5
  exact h5

end julie_total_lettuce_pounds_l802_802686


namespace probability_two_or_more_women_l802_802162

-- Definitions based on the conditions
def men : ℕ := 8
def women : ℕ := 4
def total_people : ℕ := men + women
def chosen_people : ℕ := 4

-- Function to calculate the probability of a specific event
noncomputable def probability_event (event_count : ℕ) (total_count : ℕ) : ℚ :=
  event_count / total_count

-- Function to calculate the combination (binomial coefficient)
noncomputable def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Probability calculations based on steps given in the solution:
noncomputable def prob_no_women : ℚ :=
  probability_event ((men - 0) * (men - 1) * (men - 2) * (men - 3)) (total_people * (total_people - 1) * (total_people - 2) * (total_people - 3))

noncomputable def prob_exactly_one_woman : ℚ :=
  probability_event (binom women 1 * binom men 3) (binom total_people chosen_people)

noncomputable def prob_fewer_than_two_women : ℚ :=
  prob_no_women + prob_exactly_one_woman

noncomputable def prob_at_least_two_women : ℚ :=
  1 - prob_fewer_than_two_women

-- The main theorem to be proved
theorem probability_two_or_more_women :
  prob_at_least_two_women = 67 / 165 :=
sorry

end probability_two_or_more_women_l802_802162


namespace inequality_solution_set_l802_802123

def f (x : ℝ) : ℝ := x - Real.sin x

theorem inequality_solution_set :
  {x : ℝ | f (x + 1) + f (1 - 4 * x) > 0} = {x : ℝ | x < 2 / 3} :=
by
  sorry

end inequality_solution_set_l802_802123


namespace find_prices_max_basketballs_l802_802660

-- Definition of given conditions
def conditions1 (x y : ℝ) : Prop := 
  (x - y = 50) ∧ (6 * x + 8 * y = 1700)

-- Definitions of questions:
-- Question 1: Find the price of one basketball and one soccer ball
theorem find_prices (x y : ℝ) (h: conditions1 x y) : x = 150 ∧ y = 100 := sorry

-- Definition of given conditions for Question 2
def conditions2 (x y : ℝ) (a : ℕ) : Prop :=
  (x = 150) ∧ (y = 100) ∧ 
  (0.9 * x * a + 0.85 * y * (10 - a) ≤ 1150)

-- Question 2: The school plans to purchase 10 items with given discounts
theorem max_basketballs (x y : ℝ) (a : ℕ) (h1: x = 150) (h2: y = 100) (h3: a ≤ 10) (h4: conditions2 x y a) : a ≤ 6 := sorry

end find_prices_max_basketballs_l802_802660


namespace probability_10_sided_die_l802_802854

noncomputable def probability_greater_first_die (faces : ℕ) : ℚ := 
  let total_outcomes := faces * faces
  let favorable_outcomes := (List.range (faces - 1)).sum (λ x => x + 1)
  favorable_outcomes / total_outcomes

theorem probability_10_sided_die : probability_greater_first_die 10 = 9 / 20 :=
by
  sorry

end probability_10_sided_die_l802_802854


namespace max_value_inequality_l802_802704

theorem max_value_inequality (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h : x^2 + y^2 + z^2 = 1) : 
  3 * x * y * real.sqrt 6 + 5 * y * z ≤ real.sqrt 6 * (2 * real.sqrt (375 / 481)) + 5 * (2 * real.sqrt (106 / 481)) :=
by
  sorry

end max_value_inequality_l802_802704


namespace recurring_decimal_to_fraction_l802_802501

theorem recurring_decimal_to_fraction (h1: (0.3 + 0.\overline{45} : ℝ) = (0.3\overline{45} : ℝ))
    (h2: (0.\overline{45} : ℝ) = (5 / 11 : ℝ))
    (h3: (0.3 : ℝ) = (3 / 10 : ℝ)) : (0.3\overline{45} : ℝ) = (83 / 110 : ℝ) :=
by
    sorry

end recurring_decimal_to_fraction_l802_802501


namespace problem_statement_l802_802188

-- Definitions of parallel and perpendicular predicates (should be axioms or definitions in the context)
-- For simplification, assume we have a space with lines and planes, with corresponding relations.

axiom Line : Type
axiom Plane : Type
axiom parallel : Line → Line → Prop
axiom perpendicular : Line → Plane → Prop
axiom subset : Line → Plane → Prop

-- Assume the necessary conditions: m and n are lines, a and b are planes, with given relationships.
variables (m n : Line) (a b : Plane)

-- The conditions given.
variables (m_parallel_n : parallel m n)
variables (m_perpendicular_a : perpendicular m a)

-- The proposition to prove: If m parallel n and m perpendicular to a, then n is perpendicular to a.
theorem problem_statement : perpendicular n a :=
sorry

end problem_statement_l802_802188


namespace maximal_possible_degree_difference_l802_802856

theorem maximal_possible_degree_difference (n_vertices : ℕ) (n_edges : ℕ) (disjoint_edge_pairs : ℕ) 
    (h1 : n_vertices = 30) (h2 : n_edges = 105) (h3 : disjoint_edge_pairs = 4822) : 
    ∃ (max_diff : ℕ), max_diff = 22 :=
by
  sorry

end maximal_possible_degree_difference_l802_802856


namespace intersection_correct_l802_802576

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {0, 2, 3, 4}

theorem intersection_correct : A ∩ B = {2, 3} := sorry

end intersection_correct_l802_802576


namespace decimal_to_fraction_l802_802506

theorem decimal_to_fraction :
  (\(x\), \(y\), (\(x, y\)) = (3, 110)) → 0.3\overline{45} = \(\frac{83}{110}\) := λ ⟨3, 110, 3, 110⟩, sorry

end decimal_to_fraction_l802_802506


namespace card_choice_count_l802_802484

theorem card_choice_count :
  ∃ (pairs : Finset (ℕ × ℕ)),
    (∀ (a b : ℕ), (a, b) ∈ pairs → 1 ≤ a ∧ a ≤ 50 ∧ 1 ≤ b ∧ b ≤ 50) ∧
    (∀ (a b : ℕ), (a, b) ∈ pairs → |a - b| = 11) ∧
    (∀ (a b : ℕ), (a, b) ∈ pairs → (a * b) % 5 = 0) ∧
    card pairs = 15 := 
sorry

end card_choice_count_l802_802484


namespace quadratic_solutions_l802_802752

theorem quadratic_solutions (x : ℝ) : x^2 - 2 * x = 0 ↔ x = 0 ∨ x = 2 :=
by
  sorry

end quadratic_solutions_l802_802752


namespace parallel_perpendicular_cosine_l802_802136

variable (x y z : ℝ)
variable (a b c : ℝ × ℝ × ℝ)
variable (k : ℝ)

noncomputable def vec_a := (x, 4, 1)
noncomputable def vec_b := (-2, y, -1)
noncomputable def vec_c := (3, -2, z)

theorem parallel_perpendicular_cosine (h_parallel : ∃ k : ℝ, vec_a = (k * -2, k * y, k * -1))
  (h_perpendicular : vec_b.1 * vec_c.1 + vec_b.2 * vec_c.2 + vec_b.3 * vec_c.3 = 0) :
  (vec_a = (2, 4, 1)) ∧ (vec_b = (-2, -4, -1)) ∧ (vec_c = (3, -2, 2)) ∧
  (real.cos_angle (vec_a.1 + vec_c.1, vec_a.2 + vec_c.2, vec_a.3 + vec_c.3)
    (vec_b.1 + vec_c.1, vec_b.2 + vec_c.2, vec_b.3 + vec_c.3) = -2 / 19) :=
by
  sorry

end parallel_perpendicular_cosine_l802_802136


namespace problem1_problem2_l802_802028

-- Problem 1
theorem problem1 (a b : ℝ) : 
  a^2 * (2 * a * b - 1) + (a - 3 * b) * (a + b) = 2 * a^3 * b - 2 * a * b - 3 * b^2 :=
by sorry

-- Problem 2
theorem problem2 (x : ℝ) : 
  (2 * x - 3)^2 - (x + 2)^2 = 3 * x^2 - 16 * x + 5 :=
by sorry

end problem1_problem2_l802_802028


namespace number_of_black_balls_l802_802183

theorem number_of_black_balls
  (total_balls : ℕ)  -- define the total number of balls
  (B : ℕ)            -- define B as the number of black balls
  (prob_red : ℚ := 1/4) -- define the probability of drawing a red ball as 1/4
  (red_balls : ℕ := 3)  -- define the number of red balls as 3
  (h1 : total_balls = red_balls + B) -- total balls is the sum of red and black balls
  (h2 : red_balls / total_balls = prob_red) -- given probability
  : B = 9 :=              -- we need to prove that B is 9
by
  sorry

end number_of_black_balls_l802_802183


namespace find_n_l802_802971

-- Given conditions
variables {x n : ℝ}
hypothesis1 : log 10 (sin x) + log 10 (cos x) = -3/2
hypothesis2 : log 10 (sin x + cos x) = 1/2 * (log 10 n + 1)

-- The proof statement
theorem find_n (h1 : log 10 (sin x) + log 10 (cos x) = -3/2)
               (h2 : log 10 (sin x + cos x) = 1/2 * (log 10 n + 1)) :
    n = (10 * real.sqrt 10 + 0.2) / 100 :=
sorry

end find_n_l802_802971


namespace calculate_a_minus_b_l802_802638

theorem calculate_a_minus_b : 
  ∀ (a b : ℚ), (y = a * x + b) 
  ∧ (y = 4 ↔ x = 3) 
  ∧ (y = 22 ↔ x = 10) 
  → (a - b = 6 + 2 / 7)
:= sorry

end calculate_a_minus_b_l802_802638


namespace find_101st_digit_of_reciprocal_prime_period_200_l802_802360

theorem find_101st_digit_of_reciprocal_prime_period_200 
  (p : ℕ) (hp_prime : Nat.Prime p) (h_period : ∃ X, X < 10^200 ∧ X * p = 10^200 - 1)
  (h_not_div_small_powers : ∀ n : ℕ, 1 ≤ n ∧ n < 200 → ¬ (p ∣ 10^n - 1)) :
  (∃ X, let X_digits := String.mk (Nat.digits 10 X) in X_digits.length = 200 ∧ X_digits.get! 100 = '9') :=
sorry

end find_101st_digit_of_reciprocal_prime_period_200_l802_802360


namespace problem_1_problem_2_l802_802225

-- Definition f
def f (a x : ℝ) : ℝ := a * x + 3 - abs (2 * x - 1)

-- Problem 1: If a = 1, prove ∀ x, f(1, x) ≤ 2
theorem problem_1 : (∀ x : ℝ, f 1 x ≤ 2) :=
sorry

-- Problem 2: The range of a for which f has a maximum value is -2 ≤ a ≤ 2
theorem problem_2 : (∀ a : ℝ, (∀ x : ℝ, (2 * x - 1 > 0 -> (f a x) ≤ (f a ((4 - a) / (2 * (4 - a))))) 
                        ∧ (2 * x - 1 ≤ 0 -> (f a x) ≤ (f a (1 - 2 / (1 - a))))) 
                        ↔ -2 ≤ a ∧ a ≤ 2) :=
sorry

end problem_1_problem_2_l802_802225


namespace distinct_lines_in_scalene_triangle_l802_802570

theorem distinct_lines_in_scalene_triangle 
    (A B C : Type) 
    (scalene_triangle : Π (X Y Z : Type), X ≠ Y → Y ≠ Z → Z ≠ X → Prop)
    (altitude : A → B → C → Type)
    (median : A → B → C → Type)
    (angle_bisector : A → B → C → Type) : 
    scalene_triangle A B C → 
    3 + 3 + 3 = 9 :=
by 
  intros h t1 t2 ht
  have from_A : 3 = (3:Nat) := rfl
  have from_B : 3 = (3:Nat) := rfl
  have from_C : 3 = (3:Nat) := rfl
  exact from_A + from_B + from_C
  sorry

end distinct_lines_in_scalene_triangle_l802_802570


namespace meals_per_day_l802_802803

-- Definitions based on given conditions
def number_of_people : Nat := 6
def total_plates_used : Nat := 144
def number_of_days : Nat := 4
def plates_per_meal : Nat := 2

-- Theorem to prove
theorem meals_per_day : (total_plates_used / number_of_days) / plates_per_meal / number_of_people = 3 :=
by
  sorry

end meals_per_day_l802_802803


namespace red_mushrooms_bill_l802_802895

theorem red_mushrooms_bill (R : ℝ) : 
  (2/3) * R + 6 + 3 = 17 → R = 12 :=
by
  intro h
  sorry

end red_mushrooms_bill_l802_802895


namespace problem1_problem2_l802_802112

-- Define Sn as given
def S (n : ℕ) : ℕ := (n ^ 2 + n) / 2

-- Define a sequence a_n
def a (n : ℕ) : ℕ := if n = 1 then 1 else S n - S (n - 1)

-- Define b_n using a_n = log_2 b_n
def b (n : ℕ) : ℕ := 2 ^ n

-- Define the sum of first n terms of sequence b_n
def T (n : ℕ) : ℕ := (2 ^ (n + 1)) - 2

-- Our main theorem statements
theorem problem1 (n : ℕ) : a n = n := by
  sorry

theorem problem2 (n : ℕ) : (Finset.range n).sum b = T n := by
  sorry

end problem1_problem2_l802_802112


namespace greatest_common_divisor_of_twelve_digit_repeated_integers_l802_802883

noncomputable def repeated_digits_gcd (m : ℕ) (h : 100 ≤ m ∧ m < 1000) : ℕ :=
  100001001001 * m

theorem greatest_common_divisor_of_twelve_digit_repeated_integers :
  ∀ m : ℕ, (100 ≤ m ∧ m < 1000) → gcd (repeated_digits_gcd m ‹^›) (repeated_digits_gcd (m + 1) ⟨‹_, by sorry⟩ := 100001001001 :=
begin
  sorry
end

end greatest_common_divisor_of_twelve_digit_repeated_integers_l802_802883


namespace A_subset_B_l802_802575

def A (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + (y - 2) ^ 2 ≤ 5 / 4

def B (x y : ℝ) (a : ℝ) : Prop :=
  abs (x - 1) + 2 * abs (y - 2) ≤ a

theorem A_subset_B (a : ℝ) (h : a ≥ 5 / 2) : 
  ∀ x y : ℝ, A x y → B x y a := 
sorry

end A_subset_B_l802_802575


namespace primes_sum_23_l802_802695

open Nat

theorem primes_sum_23 : ∃ (p q r s : ℕ), 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧ 
  Prime p ∧ Prime q ∧ Prime r ∧ Prime s ∧ 
  Prime (p + q + r + s) ∧ 
  ∃ (a b : ℕ), a^2 = p^2 + q * r ∧ b^2 = p^2 + q * s ∧ 
  p + q + r + s = 23 :=
by
  sorry

end primes_sum_23_l802_802695


namespace proof_statement_l802_802995

def p : Prop := ∀ x : ℝ, x > 0 → x + 4/x ≥ 4
def q : Prop := ∃ x0 : ℝ, 0 < x0 ∧ 2^x0 = 1/2

theorem proof_statement : p ∧ ¬q :=
by
  sorry

end proof_statement_l802_802995


namespace regular_polygon_inscribed_l802_802252

theorem regular_polygon_inscribed (O : Type) (r : ℝ) (n : ℕ) (a a_n : ℝ)
  (M M_n : list (O × ℝ))
  (hM_n_inscribed : is_inscribed M_n (circle O r))
  (hM_regular : is_regular_polygon M n a)
  (hM_a_bound : a ≤ a_n) :
  is_inscribed M (circle O r) ↔ a ≤ a_n :=
sorry

end regular_polygon_inscribed_l802_802252


namespace cyclic_quadrilateral_fourth_side_length_l802_802863

theorem cyclic_quadrilateral_fourth_side_length
  (r : ℝ) (a b c d : ℝ) (r_eq : r = 300 * Real.sqrt 2) (a_eq : a = 300) (b_eq : b = 400)
  (c_eq : c = 300) :
  d = 500 := 
by 
  sorry

end cyclic_quadrilateral_fourth_side_length_l802_802863


namespace equal_pair_among_given_is_c_l802_802014

theorem equal_pair_among_given_is_c : 
  (∀ (x y : ℝ), (x, y) ∈ [(-9, -(1/9)), (-|-9|, -(-9)), (9, | -9 |), (-9, | -9 |)] → x ≠ y) → (9 = | -9 |) := 
by 
  sorry

end equal_pair_among_given_is_c_l802_802014


namespace sum_of_positive_k_for_integer_roots_l802_802764

theorem sum_of_positive_k_for_integer_roots (k : ℤ) (x : ℤ) :
  (∃ α β : ℤ, α * β = -18 ∧ α + β = k ∧ k > 0) →
  ∑ (n : ℤ) in ({ k | k > 0 ∧ ∃ α β : ℤ, α * β = -18 ∧ α + β = k }.to_finset), n = 27 := sorry

end sum_of_positive_k_for_integer_roots_l802_802764


namespace zuca_winning_probability_l802_802894

-- Given definitions
def hexagon (v : Fin 6) := True

def teleport_adjacent (v : Fin 6) (p : ℙ) := 
  ∃ u : Fin 6, -- ensure u is adjacent to v in the hexagon
  u ≠ v ∧ p = 1 / 2

def teleport_opposite (v : Fin 6) (p : ℙ) := 
  ∃ u : Fin 6, -- ensure u is opposite to v in the hexagon
  u ≠ v ∧ p = 1 / 3

-- Probability definitions
def distinct_vertices (b h z : Fin 6) : Prop := 
  b ≠ h ∧ h ≠ z ∧ b ≠ z

def probability_winning (b h z : Fin 6) : ℚ := 
  if different_parity_then_win b h z 
  then 3 / 10 
  else if all_same_parity_moves b h z
       then 1 / 10 * 2 / 9
       else 0

-- Theorem statement
theorem zuca_winning_probability :
  ∀ (b h z : Fin 6),
    distinct_vertices b h z →
    (b ≠ z ∧ z ≠ h ∧ b ≠ h) →
    -- main probability result
    probability_winning b h z = 29 / 90 :=
begin
  sorry
end

end zuca_winning_probability_l802_802894


namespace factorization_identity_l802_802916

-- Define the initial expression
def initial_expr (x : ℝ) : ℝ := 2 * x^2 - 2

-- Define the factorized form
def factorized_expr (x : ℝ) : ℝ := 2 * (x + 1) * (x - 1)

-- The theorem stating the equality
theorem factorization_identity (x : ℝ) : initial_expr x = factorized_expr x := 
by sorry

end factorization_identity_l802_802916


namespace monotonic_decrease_interval_l802_802361

theorem monotonic_decrease_interval (f : ℝ → ℝ) (h : ∀ x, f x = x^2 - 2 * log x) :
  ∃ I, I = set.Ioo 0 1 := sorry

end monotonic_decrease_interval_l802_802361


namespace equation_of_tangent_line_at_O_l802_802275

noncomputable def f (x : ℝ) : ℝ := x + Real.exp x

theorem equation_of_tangent_line_at_O :
  let p := (0 : ℝ, 1 : ℝ) in -- Point O(0, 1)
  tangent_line_at f p = (λ x, 2 * x + 1) :=
by
  sorry

end equation_of_tangent_line_at_O_l802_802275


namespace Sara_cannot_have_2_l802_802715

def card_numbers : Finset ℕ := {1, 2, 3, 4}

variable (Ben Wendy Riley Sara : ℕ)

variables (h1 : Ben ≠ 1)
variables (h2 : Wendy = Riley + 1)
variables (h3 : Ben ∈ card_numbers)
variables (h4 : Wendy ∈ card_numbers)
variables (h5 : Riley ∈ card_numbers)
variables (h6 : Sara ∈ card_numbers)
variables (h7 : Ben ≠ Wendy)
variables (h8 : Ben ≠ Riley)
variables (h9 : Ben ≠ Sara)
variables (h10 : Wendy ≠ Riley)
variables (h11 : Wendy ≠ Sara)
variables (h12 : Riley ≠ Sara)

theorem Sara_cannot_have_2 : Sara ≠ 2 :=
sorry

end Sara_cannot_have_2_l802_802715


namespace simplify_expression_l802_802826

theorem simplify_expression : 
  (sqrt 5 * 5^(1/2) + 18/3 * 2 - 8^(3/2) / 2) = 1 :=
by 
  sorry

end simplify_expression_l802_802826


namespace sum_of_common_divisors_60_18_l802_802539

theorem sum_of_common_divisors_60_18 : 
  let a := 60 
  let b := 18 
  let common_divisors := {n | n ∣ a ∧ n ∣ b ∧ n > 0 } 
  (∑ n in common_divisors, n) = 12 :=
by
  let a := 60
  let b := 18
  let common_divisors := { n | n ∣ a ∧ n ∣ b ∧ n > 0 }
  have : (∑ n in common_divisors, n) = 12 := sorry
  exact this

end sum_of_common_divisors_60_18_l802_802539


namespace sixth_term_of_seq_is_21_l802_802998

-- Define the given sequence and its properties
def seq : ℕ → ℕ
| 0     := 1
| (n+1) := seq n + (n + 2)

-- Statement of the problem
theorem sixth_term_of_seq_is_21 : seq 5 = 21 := 
sorry

end sixth_term_of_seq_is_21_l802_802998


namespace simplify_expression_l802_802750

theorem simplify_expression :
  ( (sqrt 3 - 1) ^ (2 - sqrt 5) / (sqrt 3 + 1) ^ (2 + sqrt 5) = 4 * (sqrt 3 + 1) ^ sqrt 5 ) :=
by
  -- Assuming the identity in the problem.
  have h1 : 1 / (sqrt 3 + 1) = (sqrt 3 - 1) / 2, from sorry
  exact sorry

end simplify_expression_l802_802750


namespace count_valid_pairs_l802_802440

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

def difference_is_11 (a b : ℕ) : Prop := abs (a - b) = 11

def valid_card (a b : ℕ) : Prop :=
  a ∈ finset.range 51 ∧ b ∈ finset.range 51 ∧
  difference_is_11 a b ∧ (is_divisible_by_5 a ∨ is_divisible_by_5 b)

theorem count_valid_pairs : finset.card {p : ℕ × ℕ | valid_card p.1 p.2 ∧ p.1 < p.2}.to_finset = 15 := 
  sorry

end count_valid_pairs_l802_802440


namespace coeff_x2_in_expansion_l802_802268

theorem coeff_x2_in_expansion :
  let f := (x : ℤ) → (x + (1 / x : ℚ) + 3) ^ 5
  in coeff (f x) 2 = 330 :=
by
  -- Proof goes here
  sorry

end coeff_x2_in_expansion_l802_802268


namespace relationship_between_m_and_n_l802_802087

variable {α : Type} [LinearOrderedField α]

theorem relationship_between_m_and_n 
  (m n : α) 
  (h1 : ∃ α : α, (sin α + cos α = m) ∧ (sin α * cos α = n)) : 
  m^2 = 2 * n + 1 := 
by 
  sorry

end relationship_between_m_and_n_l802_802087


namespace recurring_fraction_sum_eq_l802_802509

theorem recurring_fraction_sum_eq (x : ℝ) (h1 : x = 0.45̅) : 0.3 + x = 83/110 := by
  sorry

end recurring_fraction_sum_eq_l802_802509


namespace iterative_avg_difference_l802_802881

def iter_avg (x y : ℕ) : ℚ := (x + y) / 2

def avg_sequence (seq : List ℕ) : ℚ :=
  seq.tail.foldl iter_avg seq.head

def smallest_avg_value : ℚ :=
  avg_sequence [11, 7, 5, 3, 2]

def largest_avg_value : ℚ :=
  avg_sequence [2, 3, 5, 7, 11]

theorem iterative_avg_difference :
  largest_avg_value - smallest_avg_value = 4.6875 :=
by
  sorry

end iterative_avg_difference_l802_802881


namespace min_odd_integers_is_zero_l802_802308

noncomputable def minOddIntegers (a b c d e f : ℤ) : ℕ :=
  if h₁ : a + b = 22 ∧ a + b + c + d = 36 ∧ a + b + c + d + e + f = 50 then
    0
  else
    6 -- default, just to match type expectations

theorem min_odd_integers_is_zero (a b c d e f : ℤ)
  (h₁ : a + b = 22)
  (h₂ : a + b + c + d = 36)
  (h₃ : a + b + c + d + e + f = 50) :
  minOddIntegers a b c d e f = 0 :=
  sorry

end min_odd_integers_is_zero_l802_802308


namespace sum_of_positive_k_for_integer_solution_eq_27_l802_802761

theorem sum_of_positive_k_for_integer_solution_eq_27 :
  ∑ k in { k : ℤ | ∃ α β : ℤ, α * β = -18 ∧ k = α + β ∧ k > 0 }.toFinset, k = 27 :=
begin
  sorry
end

end sum_of_positive_k_for_integer_solution_eq_27_l802_802761


namespace sum_common_divisors_sixty_and_eighteen_l802_802523

theorem sum_common_divisors_sixty_and_eighteen : 
  ∑ d in ({d ∈ ({1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60} : finset ℕ) | d ∈ ({1, 2, 3, 6, 9, 18} : finset ℕ)} : finset ℕ), d = 12 :=
by sorry

end sum_common_divisors_sixty_and_eighteen_l802_802523


namespace tan_alpha_neg_four_thirds_cos2alpha_plus_cos_alpha_add_pi_over_2_l802_802104

variable (α : ℝ)
variable (h1 : π / 2 < α)
variable (h2 : α < π)
variable (h3 : Real.sin α = 4 / 5)

theorem tan_alpha_neg_four_thirds (h1 : π / 2 < α) (h2 : α < π) (h3 : Real.sin α = 4 / 5) : Real.tan α = -4 / 3 := 
by sorry

theorem cos2alpha_plus_cos_alpha_add_pi_over_2 (h1 : π / 2 < α) (h2 : α < π) (h3 : Real.sin α = 4 / 5) : 
  Real.cos (2 * α) + Real.cos (α + π / 2) = -27 / 25 := 
by sorry

end tan_alpha_neg_four_thirds_cos2alpha_plus_cos_alpha_add_pi_over_2_l802_802104


namespace percentage_valid_votes_first_candidate_l802_802653

theorem percentage_valid_votes_first_candidate (total_votes : ℕ) (invalid_percentage : ℝ) (votes_second_candidate : ℕ) :
  total_votes = 5500 →
  invalid_percentage = 0.20 →
  votes_second_candidate = 1980 →
  let valid_votes := (1 - invalid_percentage) * total_votes in
  let votes_first_candidate := valid_votes - votes_second_candidate in
  (votes_first_candidate / valid_votes) * 100 = 55 :=
begin 
  intros h1 h2 h3,
  let valid_votes := (1 - invalid_percentage) * total_votes,
  let votes_first_candidate := valid_votes - votes_second_candidate,
  have h4: valid_votes = 4400, { sorry },
  have h5: votes_first_candidate = 2420, { sorry },
  rw h4 at *,
  rw h5 at *,
  norm_num,
end

end percentage_valid_votes_first_candidate_l802_802653


namespace initial_distance_between_Seonghyeon_and_Jisoo_l802_802747

theorem initial_distance_between_Seonghyeon_and_Jisoo 
  (D : ℝ)
  (h1 : 2000 = (D - 200) + 1000) : 
  D = 1200 :=
by
  sorry

end initial_distance_between_Seonghyeon_and_Jisoo_l802_802747


namespace ratio_red_to_black_l802_802665

theorem ratio_red_to_black (a b x : ℕ) (h1 : x + b = 3 * a) (h2 : x = 2 * b - 3 * a) :
  a / b = 1 / 2 := by
  sorry

end ratio_red_to_black_l802_802665


namespace smallest_square_side_length_l802_802556

theorem smallest_square_side_length :
  ∃ (n s : ℕ),  14 * n = s^2 ∧ s = 14 := 
by
  existsi 14, 14
  sorry

end smallest_square_side_length_l802_802556


namespace total_stops_traveled_l802_802245

-- Definitions based on the conditions provided
def yoojeong_stops : ℕ := 3
def namjoon_stops : ℕ := 2

-- Theorem statement to prove the total number of stops
theorem total_stops_traveled : yoojeong_stops + namjoon_stops = 5 := by
  -- Proof omitted
  sorry

end total_stops_traveled_l802_802245


namespace seven_boys_washing_length_l802_802847

-- Define the given conditions
def five_boys_wall_length (days : ℕ) : ℝ := 25
def five_boys (days : ℕ) : ℕ := 5
def washing_days : ℕ := 4

-- Define what needs to be proved
theorem seven_boys_washing_length : 
  (∃ (wall_length : ℝ), 
   let boy_rate_per_day := (five_boys_wall_length washing_days) / ((five_boys washing_days) * washing_days)
   in wall_length = boy_rate_per_day * ((7 : ℕ) * washing_days)) :=
begin
  -- Correct answer derived as per the solution steps
  let wall_length := 35,
  use wall_length,
  let boy_rate_per_day := (five_boys_wall_length washing_days) / (↑(five_boys washing_days) * ↑washing_days),
  show wall_length = boy_rate_per_day * (7 * washing_days),
  sorry -- Proof to be filled in.
end

end seven_boys_washing_length_l802_802847


namespace part1_part2_l802_802577

variable (x : ℝ)

def A : Set ℝ := { x | 2 * x + 1 < 5 }
def B : Set ℝ := { x | x^2 - x - 2 < 0 }

theorem part1 : A ∩ B = { x | -1 < x ∧ x < 2 } :=
sorry

theorem part2 : A ∪ { x | x ≤ -1 ∨ x ≥ 2 } = Set.univ :=
sorry

end part1_part2_l802_802577


namespace range_of_p_l802_802206

def h (x : ℝ) : ℝ := 4 * x + 3

def p (x : ℝ) : ℝ := h (h (h (h x)))

theorem range_of_p : 
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → -1 ≤ p x ∧ p x ≤ 1023 :=
by
  sorry

end range_of_p_l802_802206


namespace min_tetrahedra_to_decompose_cube_l802_802813

-- Define a new theorem stating the minimum number of tetrahedra needed to decompose a cube
theorem min_tetrahedra_to_decompose_cube : ∃ n : ℕ, is_cube_divided_into_tetrahedra 1 n ∧ n = 5 :=
by
  -- Let the condition that we divide a cube with side length 1 into tetrahedra
  -- and we want to prove that the minimum number n of tetrahedra pieces == 5
  sorry -- Proof will be added separately

end min_tetrahedra_to_decompose_cube_l802_802813


namespace cards_difference_product_divisible_l802_802466

theorem cards_difference_product_divisible :
  let S := {n | 1 ≤ n ∧ n ≤ 50}
  let valid_pairs := {(a, b) | a ∈ S ∧ b ∈ S ∧ |a - b| = 11 ∧ (a * b) % 5 = 0}
  valid_pairs.card = 15 := 
sorry

end cards_difference_product_divisible_l802_802466


namespace expected_number_of_groups_l802_802781

-- Define the conditions
variables (k m : ℕ) (h : 0 < k ∧ 0 < m)

-- Expected value of groups in the sequence
theorem expected_number_of_groups : 
  ∀ k m, (0 < k) → (0 < m) → 
  let total_groups := 1 + (2 * k * m) / (k + m) in total_groups = 1 + (2 * k * m) / (k + m) :=
by
  intros k m hk hm
  let total_groups := 1 + (2 * k * m) / (k + m)
  exact (rfl : total_groups = 1 + (2 * k * m) / (k + m))

end expected_number_of_groups_l802_802781


namespace peter_remaining_walk_time_l802_802735

-- Define the parameters and conditions
def total_distance : ℝ := 2.5
def time_per_mile : ℝ := 20
def distance_walked : ℝ := 1

-- Define the remaining distance
def remaining_distance : ℝ := total_distance - distance_walked

-- Define the remaining time Peter needs to walk
def remaining_time_to_walk (d : ℝ) (t : ℝ) : ℝ := d * t

-- State the problem we want to prove
theorem peter_remaining_walk_time :
  remaining_time_to_walk remaining_distance time_per_mile = 30 :=
by
  -- Placeholder for the proof
  sorry

end peter_remaining_walk_time_l802_802735


namespace arrangement_schemes_l802_802885

theorem arrangement_schemes : 
  let female_teachers := 2 in
  let male_teachers := 4 in
  let group_size := 3 in
  let female_per_group := 1 in
  let male_per_group := 2 in
  (female_teachers * (choose male_teachers male_per_group)) = 12 := 
  sorry

end arrangement_schemes_l802_802885


namespace number_of_incorrect_expressions_l802_802011

theorem number_of_incorrect_expressions :
  let expr1 := ¬(\{0\} ∈ \{0,2,3\})
  let expr2 := (∅ ⊆ \{0\})
  let expr3 := (\{0,1,2\} ⊆ \{1,2,0\})
  let expr4 := ¬(0 ∈ ∅)
  let expr5 := (0 ∩ ∅ = ∅)
  (if expr1 then 1 else 0) + (if expr2 then 0 else 1) + (if expr3 then 0 else 1) + (if expr4 then 1 else 0) + (if expr5 then 0 else 1) = 3 :=
by
  sorry

end number_of_incorrect_expressions_l802_802011


namespace smaller_number_is_180_l802_802836

theorem smaller_number_is_180 (a b : ℕ) (h1 : a = 3 * b) (h2 : a + 4 * b = 420) :
  a = 180 :=
sorry

end smaller_number_is_180_l802_802836


namespace solution_set_A_solution_set_B_l802_802227

theorem solution_set_A (a : ℝ) :
  let b := -2 * a in
  (
    (a > 0 → {x : ℝ | ax + b > 0} = {x | x > 2}) ∧
    (a = 0 → {x : ℝ | ax + b > 0} = ∅) ∧
    (a < 0 → {x : ℝ | ax + b > 0} = {x | x < 2})
  ) :=
sorry

theorem solution_set_B (a : ℝ) (A : set ℝ) :
  A = set.Iio 1 →
  let b := -2 * a in
  a < 0 →
  (
    (a < -1 → {x : ℝ | (a * x + b) * (x - a) ≥ 0} = {x | a ≤ x ∧ x ≤ -1}) ∧
    (a = -1 → {x : ℝ | (a * x + b) * (x - a) ≥ 0} = {-1}) ∧
    (-1 < a ∧ a < 0 → {x : ℝ | (a * x + b) * (x - a) ≥ 0} = {x | -1 ≤ x ∧ x ≤ a})
  ) :=
sorry

end solution_set_A_solution_set_B_l802_802227


namespace sum_common_divisors_l802_802548

-- Define the sum of a set of numbers
def set_sum (s : Set ℕ) : ℕ :=
  s.fold (λ x acc => x + acc) 0

-- Define the divisors of a number
def divisors (n : ℕ) : Set ℕ :=
  { d | d > 0 ∧ n % d = 0 }

-- Definitions based on the given conditions
def divisors_of_60 : Set ℕ := divisors 60
def divisors_of_18 : Set ℕ := divisors 18
def common_divisors : Set ℕ := divisors_of_60 ∩ divisors_of_18

-- Declare the theorem to be proved
theorem sum_common_divisors : set_sum common_divisors = 12 :=
  sorry

end sum_common_divisors_l802_802548


namespace area_of_triangle_ABC_l802_802899

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 0, y := 2 }
def B : Point := { x := 6, y := 0 }
def C : Point := { x := 4, y := 7 }

def triangle_area (P1 P2 P3 : Point) : ℝ :=
  0.5 * abs (P1.x * (P2.y - P3.y) +
             P2.x * (P3.y - P1.y) +
             P3.x * (P1.y - P2.y))

theorem area_of_triangle_ABC : triangle_area A B C = 19 :=
by
  sorry

end area_of_triangle_ABC_l802_802899


namespace find_sum_of_numbers_l802_802779

theorem find_sum_of_numbers (x A B C : ℝ) (h1 : x > 0) (h2 : A = x) (h3 : B = 2 * x) (h4 : C = 3 * x) (h5 : A^2 + B^2 + C^2 = 2016) : A + B + C = 72 :=
sorry

end find_sum_of_numbers_l802_802779


namespace math_problem_l802_802342

theorem math_problem :
  (∃ n : ℕ, 28 = 4 * n) ∧
  ((∃ n1 : ℕ, 361 = 19 * n1) ∧ ¬(∃ n2 : ℕ, 63 = 19 * n2)) ∧
  (¬((∃ n3 : ℕ, 90 = 30 * n3) ∧ ¬(∃ n4 : ℕ, 65 = 30 * n4))) ∧
  ((∃ n5 : ℕ, 45 = 15 * n5) ∧ (∃ n6 : ℕ, 30 = 15 * n6)) ∧
  (∃ n7 : ℕ, 144 = 12 * n7) :=
by {
  -- We need to prove each condition to be true and then prove the statements A, B, D, E are true.
  sorry
}

end math_problem_l802_802342


namespace inverse_100_mod_101_l802_802050

theorem inverse_100_mod_101 : (100 * 100) % 101 = 1 :=
by
  -- Proof can be provided here.
  sorry

end inverse_100_mod_101_l802_802050


namespace cricket_bat_profit_percentage_l802_802852

-- Definitions for the given conditions
def selling_price : ℝ := 900
def profit : ℝ := 225
def cost_price : ℝ := selling_price - profit
def profit_percentage : ℝ := (profit / cost_price) * 100

-- The statement to prove the profit percentage
theorem cricket_bat_profit_percentage :
  profit_percentage = 33.33 := 
sorry

end cricket_bat_profit_percentage_l802_802852


namespace calculate_stripes_l802_802719

theorem calculate_stripes :
  let olga_stripes_per_shoe := 3
  let rick_stripes_per_shoe := olga_stripes_per_shoe - 1
  let hortense_stripes_per_shoe := olga_stripes_per_shoe * 2
  let ethan_stripes_per_shoe := hortense_stripes_per_shoe + 2
  (olga_stripes_per_shoe * 2 + rick_stripes_per_shoe * 2 + hortense_stripes_per_shoe * 2 + ethan_stripes_per_shoe * 2) / 2 = 19 := 
by
  sorry

end calculate_stripes_l802_802719


namespace total_digits_in_first_3003_even_ints_l802_802822

def number_of_digits (n : ℕ) : ℕ :=
  if h : n > 0 then (n.to_string.length) else 0

theorem total_digits_in_first_3003_even_ints:
  (finset.range 3003).sum (λ n, number_of_digits (2 * (n + 1))) = 11460 :=
by
  sorry

end total_digits_in_first_3003_even_ints_l802_802822


namespace percent_calculation_l802_802330

theorem percent_calculation (y : ℝ) : (0.3 * (0.6 * y) = 0.18 * y) → (0.18 * y / y) * 100 = 18 := by
  sorry

end percent_calculation_l802_802330


namespace length_of_BC_l802_802657

-- Definition of the problem conditions
def isosceles_triangle_with_altitude (A B C H : Point) (length5 : ℕ) := 
  IsoscelesTriangle A B C ∧ 
  (distance A B = length5) ∧ (distance A C = length5) ∧
  AltitudeFrom B H AC ∧ 
  distance A H = 2 * distance H C

-- Formal statement of the theorem in Lean 4
theorem length_of_BC (A B C H : Point) (length5 : ℕ) 
  (h : isosceles_triangle_with_altitude A B C H length5) : 
  distance B C = 5 * sqrt 6 / 3 := 
sorry

end length_of_BC_l802_802657


namespace range_of_a_l802_802129

theorem range_of_a (a b c : ℝ) (h₁ : a + b + c = 2) (h₂ : a^2 + b^2 + c^2 = 4) (h₃ : a > b ∧ b > c) :
  (2 / 3 < a ∧ a < 2) :=
sorry

end range_of_a_l802_802129


namespace acute_angle_repeated_root_l802_802589

theorem acute_angle_repeated_root (θ : ℝ) (h₁ : 0 < θ ∧ θ < π / 2)
  (h₂ : ∃x : ℝ, x^2 + 4 * x * cos θ + cot θ = 0 ∧ 
    ∀y : ℝ, y ≠ x → y^2 + 4 * y * cos θ + cot θ ≠ 0 := sorry :
  θ = π / 12 ∨ θ = 5 * π / 12 :=
sorry

end acute_angle_repeated_root_l802_802589


namespace circumcircles_tangent_l802_802706

open EuclideanGeometry
noncomputable theory

variables 
  (A B C K Q H F M : Point)
  (Γ : Circle)

-- Given conditions
variables
  (h_triangle : Triangle ABC)
  (h_acute : acuteAngle ABC)
  (h_AB_gt_AC : dist A B > dist A C)
  (h_circumcircle : isCircumcircle Γ ABC)
  (h_orthocenter : orthocenter H ABC)
  (h_foot : altitudeFoot F A BC)
  (h_midpoint : midpoint M B C)
  (h_Q_on_circumcircle : OnCircle Q Γ)
  (h_∠HQA_90 : angle H Q A = 90)
  (h_K_on_circumcircle : OnCircle K Γ)
  (h_∠HKQ_90 : angle H K Q = 90)
  (h_distinct_points : distinctPoints A B C K Q)

-- Prove that the circumcircles of triangles KQH and FKM are tangent to each other
theorem circumcircles_tangent :
    tangent (circumcircle K Q H) (circumcircle F K M) := 
sorry

end circumcircles_tangent_l802_802706


namespace cards_difference_product_divisibility_l802_802461

theorem cards_difference_product_divisibility :
  (∃ pairs : list (ℕ × ℕ), 
    (∀ p ∈ pairs, 1 ≤ p.1 ∧ p.1 ≤ 50 ∧ 1 ≤ p.2 ∧ p.2 ≤ 50 ∧ 
      (p.1 - p.2 = 11 ∨ p.2 - p.1 = 11) ∧ 
      (p.1 * p.2) % 5 = 0) ∧ 
    pairs.length = 15) := 
sorry

end cards_difference_product_divisibility_l802_802461


namespace probability_stop_after_5_draws_l802_802644

variable {Ω : Type} [UniformProbabilitySpace Ω]
variable (bag : Finset (Fin 3)) -- representing the three different colors
variable (draw : Ω → Fin 3)
variable [h_drawing : ∀ x, x ∈ bag → uniform (draw x)]

noncomputable def event_stop_after_5_draws : Event Ω := 
  (∃ (draws : Fin 5 → Fin 3), 
     (∃ i j, i ≠ j ∧ draws i = 0 ∧ draws j = 0) ∧
     (∃ k l, k ≠ l ∧ draws k = 1 ∧ draws l = 1) ∧
     (∃ m, draws m = 2) ∨
     (∃ i, draws i = 0 ∧ 
          (∃ j k, j ≠ k ∧ draws j ≠ 0 ∧ draws k ≠ 0)) ∧
     (¬∃ i, draws i ≠ 0)) ∧ 
  ∀ i, i ∈ Finset.univ → draws i ∈ bag

theorem probability_stop_after_5_draws : 
  prob event_stop_after_5_draws = 14 / 81 :=
by sorry

end probability_stop_after_5_draws_l802_802644


namespace simple_interest_correct_l802_802005

def principal : ℝ := 44625
def rate : ℝ := 1
def time : ℝ := 9

def simple_interest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

theorem simple_interest_correct :
  simple_interest principal rate time = 4016.25 :=
by
  sorry

end simple_interest_correct_l802_802005


namespace sum_of_positive_k_for_integer_roots_l802_802763

theorem sum_of_positive_k_for_integer_roots (k : ℤ) (x : ℤ) :
  (∃ α β : ℤ, α * β = -18 ∧ α + β = k ∧ k > 0) →
  ∑ (n : ℤ) in ({ k | k > 0 ∧ ∃ α β : ℤ, α * β = -18 ∧ α + β = k }.to_finset), n = 27 := sorry

end sum_of_positive_k_for_integer_roots_l802_802763


namespace algebraic_expression_value_l802_802983

noncomputable def expression (m x n : ℝ) : ℝ :=
  (m + x) ^ 2000 * (-m^2 * n + x * n^2) + 1

theorem algebraic_expression_value (m x n: ℝ)
  (h1 : (m + 3) * x ^ (Real.abs m - 2) + 6 * m = 0)
  (h2 : m = 3)
  (h3 : x = -3)
  (h4 : n = 2/3)
  (eq1: ∀ n x, n * x - 5 = x * (3 - n)) :
  expression m x n = 1 := by sorry

end algebraic_expression_value_l802_802983


namespace coprime_divisors_imply_product_divisor_l802_802703

theorem coprime_divisors_imply_product_divisor 
  (a b n : ℕ) (h_coprime : Nat.gcd a b = 1)
  (h_a_div_n : a ∣ n) (h_b_div_n : b ∣ n) : a * b ∣ n :=
by
  sorry

end coprime_divisors_imply_product_divisor_l802_802703


namespace sequence_not_prime_l802_802809

theorem sequence_not_prime (n : ℕ) (h : n ≥ 1) : ¬ prime (10^(4 * n) + 1) := by
  sorry

end sequence_not_prime_l802_802809


namespace propositions_true_false_l802_802207

variables (m n : Line) (α β : Plane)

def parallel (l1 l2 : Line) : Prop := sorry -- Definition for two lines being parallel
def perpendicular (l1 l2 : Line) : Prop := sorry -- Definition for two lines being perpendicular
def lies_in (l : Line) (p : Plane) : Prop := sorry -- Definition for a line lying in a plane
def parallel_planes (p1 p2 : Plane) : Prop := sorry -- Definition for two planes being parallel
def perpendicular_planes (p1 p2 : Plane) : Prop := sorry -- Definition for two planes being perpendicular

theorem propositions_true_false :
  (parallel_planes α β → lies_in m β → lies_in n α → ¬parallel m n) ∧
  (parallel_planes α β → perpendicular m β → parallel n α → perpendicular m n) ∧
  (perpendicular_planes α β → perpendicular m α → parallel n β → ¬parallel m n) ∧
  (perpendicular_planes α β → perpendicular m α → perpendicular n β → perpendicular m n) :=
by
  sorry

end propositions_true_false_l802_802207


namespace problem_l802_802192

noncomputable def a : ℝ×ℝ := (1, 0)
noncomputable def b : ℝ×ℝ := (0, 1)
noncomputable def OQ := (Real.sqrt 2, Real.sqrt 2)
def unit_circle (θ : ℝ) := (Real.cos θ, Real.sin θ)
def region (P : ℝ × ℝ) (r R : ℝ) := 0 < r ∧ r < R ∧ r ≤ Real.sqrt((P.fst - OQ.fst)^2 + (P.snd - OQ.snd)^2) ∧ Real.sqrt((P.fst - OQ.fst)^2 + (P.snd - OQ.snd)^2) ≤ R

theorem problem (r R : ℝ) (P : ℝ × ℝ) (θ : ℝ) :
  (P = unit_circle θ ∧ 0 ≤ θ ∧ θ ≤ 2 * Real.pi) →
  region P r R →
  (1 < r ∧ r < R ∧ R < 3) :=
by
  intros _ _
  sorry

end problem_l802_802192


namespace product_of_sum_and_difference_l802_802316

theorem product_of_sum_and_difference (a b : ℝ) (ha : a = 4.93) (hb : b = 3.78) :
  (a + b) * (a - b) = 10.0165 :=
by { rw [ha, hb], norm_num, sorry }

end product_of_sum_and_difference_l802_802316


namespace matrix_set_property_l802_802426

theorem matrix_set_property (A: Matrix (Fin 2) (Fin 2) ℂ) 
  (M : Set (Matrix (Fin 2) (Fin 2) ℂ) := {A | A.elem 0 0 * A.elem 0 1 = A.elem 1 0 * A.elem 1 1})
  (hA : A ∈ M)
  (k : ℕ) (hk : 1 ≤ k)
  (hAk : A^k ∈ M) (hAk1 : A^(k+1) ∈ M) (hAk2 : A^(k+2) ∈ M) :
  ∀ n ≥ 1, A^n ∈ M := by
  sorry

end matrix_set_property_l802_802426


namespace sum_common_divisors_60_18_l802_802528

def divisors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ m => n % m = 0)

noncomputable def sum (l : List ℕ) : ℕ :=
  l.foldr (· + ·) 0

theorem sum_common_divisors_60_18 :
  sum (List.filter (λ d => d ∈ divisors 18) (divisors 60)) = 12 :=
by
  sorry

end sum_common_divisors_60_18_l802_802528


namespace conjugate_complex_solutions_l802_802108

theorem conjugate_complex_solutions 
  (x y : ℂ)
  (cond1 : x.conj = y)
  (cond2 : (x + y) ^ 2 - 3 * x * y * complex.I = 4 - 6 * complex.I) :
  (x = 1 + complex.I ∧ y = 1 - complex.I) ∨
  (x = 1 - complex.I ∧ y = 1 + complex.I) ∨
  (x = -1 + complex.I ∧ y = -1 - complex.I) ∨
  (x = -1 - complex.I ∧ y = -1 + complex.I) :=
sorry

end conjugate_complex_solutions_l802_802108


namespace polygon_side_parallel_condition_l802_802075

theorem polygon_side_parallel_condition (n : ℕ) (h : n ≥ 3) : 
  (∃ (polygon : Type) [is_polygon polygon (n vertices)], ∀ side ∈ polygon.sides, ∃ side' ∈ polygon.sides, side ≠ side' ∧ side ∥ side') ↔ (even n ∨ n ≥ 7) := 
sorry

end polygon_side_parallel_condition_l802_802075


namespace sin_double_angle_identity_l802_802106

variable (θ : ℝ)

theorem sin_double_angle_identity (h : sin θ + 2 * cos (θ / 2) ^ 2 = 5 / 4) : sin (2 * θ) = -15 / 16 := 
by
  sorry

end sin_double_angle_identity_l802_802106


namespace product_of_solutions_eq_neg_nine_product_of_solutions_l802_802934

theorem product_of_solutions_eq_neg_nine :
  ∀ (x : ℝ), (|x| = 3 * (|x| - 2)) → x = 3 ∨ x = -3 :=
by
  sorry

theorem product_of_solutions :
  (∀ (x : ℝ), (|x| = 3 * (|x| - 2)) → (∃ (a b : ℝ), x = a ∨ x = b ∧ a * b = -9)) :=
by
  sorry

end product_of_solutions_eq_neg_nine_product_of_solutions_l802_802934


namespace interest_rate_per_annum_l802_802861

-- Description of the given conditions and what we need to prove
theorem interest_rate_per_annum (P t I : ℝ) (r : ℝ) (h1 : P = 350) (h2 : t = 8) (h3 : I = P - 238) (h4 : I = P * r * t / 100) : r = 4 :=
by
  -- Variables
  have hP : P = 350 := h1
  have ht : t = 8 := h2
  have hI := h3
  have hInterest := h4
  
  -- Simplification steps
  have I_value : I = 350 - 238 := by rw [hP, h3]
  have I_simplified : I = 112 := by norm_num
  have interest_formula := by rw [I_value, h4, hP, ht]
  
  -- Solve equation
  sorry -- The rest of the mathematical steps are omitted as per instruction.

end interest_rate_per_annum_l802_802861


namespace P_on_line_MN_l802_802221

-- Define the data for the problem
variables (a b c z w : ℝ)
-- Assume necessary conditions:
-- acute triangle and AB ≠ AC implies a > 0, b ≠ c, and b + c ≠ 0
-- Additionally, z and w represent some points on AB meeting certain conditions

-- Given coordinates assumption
def O : (ℝ × ℝ) := (0, 0)
def A : (ℝ × ℝ) := (0, a)
def B : (ℝ × ℝ) := (b, 0)
def C : (ℝ × ℝ) := (c, 0)

-- Midpoint definitions of OA and BC
def M : (ℝ × ℝ) := (0, a / 2)
def N : (ℝ × ℝ) := ((b + c) / 2, 0)

-- Line MN in the form y = mx + n
def line_MN (x : ℝ): ℝ := - (a / (b + c)) * x + (a / 2)

-- Coordinates of the points D, E, F, G with D = (z, w)
def D : (ℝ × ℝ) := (z, w)
def E : (ℝ × ℝ) := (z, 0)
def F : (ℝ × ℝ) := (c / a * (a - w), 0)
def G : (ℝ × ℝ) := (c / a * (a - w), w)

-- Midpoint P of diagonal DF
def P : (ℝ × ℝ) := ((z * a + c * a - w * c) / (2 * a), w / 2)

-- Statement to prove that P lies on line MN
theorem P_on_line_MN : 
  a > 0 → b ≠ c → (b + c ≠ 0) → 
  (line_MN ((z * a + c * a - w * c) / (2 * a)) = w / 2) := by
  sorry

end P_on_line_MN_l802_802221


namespace coloring_grid_no_uniform_rectangles_l802_802912

noncomputable def numberOfValidColorings : ℕ :=
  284688

theorem coloring_grid_no_uniform_rectangles :
  let n := NumberOfValidColorings.decidablePaths (3 * 4) 3  (λ r c, 1 - 282321)
  n = numberOfValidColorings :=
  sorry

end coloring_grid_no_uniform_rectangles_l802_802912


namespace julie_total_lettuce_pounds_l802_802685

theorem julie_total_lettuce_pounds :
  ∀ (cost_green cost_red cost_per_pound total_cost total_pounds : ℕ),
  cost_green = 8 →
  cost_red = 6 →
  cost_per_pound = 2 →
  total_cost = cost_green + cost_red →
  total_pounds = total_cost / cost_per_pound →
  total_pounds = 7 :=
by
  intros cost_green cost_red cost_per_pound total_cost total_pounds h1 h2 h3 h4 h5
  rw [h1, h2] at h4
  rw [h3] at h5
  norm_num at h4
  norm_num at h5
  exact h5

end julie_total_lettuce_pounds_l802_802685


namespace two_points_determine_a_straight_line_l802_802300

theorem two_points_determine_a_straight_line (cond2 cond3 cond4 : Prop)
  (h2 : cond2 = "When two thin wooden strips are stacked together with both ends coinciding, if there is a gap in the middle, then both strips cannot be straight.")
  (h3 : cond3 = "When planting trees, as long as the positions of two tree pits are determined, the tree pits in the same row can be in a straight line.")
  (h4 : cond4 = "It only takes two nails to fix a thin wooden strip on the wall."):
  (explains_2_3_4 : cond2 ∧ cond3 ∧ cond4) :=
  by
    sorry

end two_points_determine_a_straight_line_l802_802300


namespace domain_log2_x_minus_1_l802_802759

theorem domain_log2_x_minus_1 (x : ℝ) : (1 < x) ↔ (∃ y : ℝ, y = Real.logb 2 (x - 1)) := by
  sorry

end domain_log2_x_minus_1_l802_802759


namespace functional_eq_holds_l802_802839

noncomputable def f : ℝ → ℝ 
| x := if x ≠ -0.5 then 1 / (x + 0.5) else 0.5 

theorem functional_eq_holds : ∀ x : ℝ, f x - (x - 0.5) * f (-x - 1) = 1 :=
by
  intro x
  -- Introduce the required proof steps here
  sorry -- placeholder for the steps that constitute the actual proof

end functional_eq_holds_l802_802839


namespace count_valid_pairs_l802_802445

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

def difference_is_11 (a b : ℕ) : Prop := abs (a - b) = 11

def valid_card (a b : ℕ) : Prop :=
  a ∈ finset.range 51 ∧ b ∈ finset.range 51 ∧
  difference_is_11 a b ∧ (is_divisible_by_5 a ∨ is_divisible_by_5 b)

theorem count_valid_pairs : finset.card {p : ℕ × ℕ | valid_card p.1 p.2 ∧ p.1 < p.2}.to_finset = 15 := 
  sorry

end count_valid_pairs_l802_802445


namespace expansion_constant_term_l802_802791

theorem expansion_constant_term (a : ℝ) 
    (h : (1 + a) * (3 ^ 5) = 2) :
    let constant_term := (2 * 10) * 4 in 
    (\sum_{k : Fin 6} (Nat.choose 5 k) * (2*x)^(5-k) * (-1/x)^k).filter (λ t, t=0) = 40 := 
begin
  -- proof details skipped
  sorry
end

end expansion_constant_term_l802_802791


namespace card_pairs_count_l802_802477

theorem card_pairs_count :
  let cards := {n : ℕ | 1 ≤ n ∧ n ≤ 50}
  let valid_pairs := {(a, b) ∈ cards × cards | 
    a ≠ b ∧ (a - b = 11 ∨ b - a = 11) ∧ (a * b) % 5 = 0}
  in valid_pairs.count = 15 :=
by
  sorry

end card_pairs_count_l802_802477


namespace find_dy_l802_802513

noncomputable def y (x : ℝ) : ℝ :=
  cos x * log (tan x) - log (tan (x / 2))

theorem find_dy (x : ℝ) : 
  let dy := (-sin x * log (tan x)) * (λ x, dx) in
  (dy / dx = -sin x * log (tan x)) :=
sorry

end find_dy_l802_802513


namespace problem_part_I_problem_part_II_l802_802989

noncomputable def f (x a : ℝ) : ℝ := x^2 - a * x - a^2 * Real.log x

theorem problem_part_I : f 1 1 = 0 := by
  sorry

theorem problem_part_II (a : ℝ) (h : a ≠ 0) :
  (a < -2 → (∀ x, 1 < x ∧ x < -a / 2 → f' x a < 0) ∧ (∀ x, -a / 2 < x → f' x a > 0)) ∧
  (-2 ≤ a ∧ a < 0 → (∀ x, 1 < x → f' x a > 0)) ∧
  (0 ≤ a ∧ a ≤ 1 → (∀ x, 1 < x → f' x a > 0)) ∧
  (a > 1 → (∀ x, a < x → f' x a > 0) ∧ (∀ x, 1 < x ∧ x < a → f' x a < 0)) := by
  sorry

end problem_part_I_problem_part_II_l802_802989


namespace total_weight_correct_l802_802896

-- Define the constant variables as per the conditions
def jug1_capacity : ℝ := 2
def jug2_capacity : ℝ := 3
def fill_percentage : ℝ := 0.7
def jug1_density : ℝ := 4
def jug2_density : ℝ := 5

-- Define the volumes of sand in each jug
def jug1_sand_volume : ℝ := fill_percentage * jug1_capacity
def jug2_sand_volume : ℝ := fill_percentage * jug2_capacity

-- Define the weights of sand in each jug
def jug1_weight : ℝ := jug1_sand_volume * jug1_density
def jug2_weight : ℝ := jug2_sand_volume * jug2_density

-- State the theorem that combines the weights
theorem total_weight_correct : jug1_weight + jug2_weight = 16.1 := sorry

end total_weight_correct_l802_802896


namespace simplest_fraction_l802_802013

variable {x : ℝ}

theorem simplest_fraction (h1 : 2 * x ≠ 0) (h2 : x^2 - 1 ≠ 0) :
  (∀ a : ℝ, a ∈ { (4 / (2 * x)), ((x - 1) / (x^2 - 1)), (1 / (x + 1)), ((1 - x) / (x - 1)) } → a = (1 / (x + 1))) :=
sorry

end simplest_fraction_l802_802013


namespace circumcenter_equality_l802_802757

open Real EuclideanGeometry

-- Define the necessary points and conditions
variables {A B C D K M N O : Point}
variables [incircle O [A, C, B, D]]
variables (chord1 : Chord O A C)
variables (chord2 : Chord O B D)
variables (intersec : intersection chord1 chord2 K)
variables (circumcenter_AKB : Circumcenter (triangle A K B) M)
variables (circumcenter_CKD : Circumcenter (triangle C K D) N)
variables (circ_center : Center O)
variables (circ_intersc_1 : circumcircle O A C B D)

-- The lean statement: prove OM = KN under the given conditions
theorem circumcenter_equality (OM_KN : dist O M = dist K N) : 
  OM = KN :=
sorry

end circumcenter_equality_l802_802757


namespace modulus_of_complex_l802_802596

theorem modulus_of_complex :
  (| (3 + 4 * complex.i) / (1 - complex.i) |) = (5 * real.sqrt 2) / 2 :=
by
  sorry

end modulus_of_complex_l802_802596


namespace relationship_among_a_b_c_l802_802974

noncomputable def a : ℝ := Real.log 2 / Real.log 0.3
noncomputable def b : ℝ := Real.exp (0.3 * Real.log 2)
noncomputable def c : ℝ := Real.exp (0.4 * Real.log 0.3)

theorem relationship_among_a_b_c : a < c ∧ c < b := by
  sorry

end relationship_among_a_b_c_l802_802974


namespace smallest_common_students_l802_802833

theorem smallest_common_students 
    (z : ℕ) (k : ℕ) (j : ℕ) 
    (hz : z = k ∧ k = j) 
    (hz_ratio : ∃ x : ℕ, z = 3 * x ∧ k = 2 * x ∧ j = 5 * x)
    (hz_group : ∃ y : ℕ, z = 14 * y) 
    (hk_group : ∃ w : ℕ, k = 10 * w) 
    (hj_group : ∃ v : ℕ, j = 15 * v) : 
    z = 630 ∧ k = 420 ∧ j = 1050 :=
    sorry

end smallest_common_students_l802_802833


namespace sum_of_squares_of_roots_l802_802425

theorem sum_of_squares_of_roots :
  let a := 5
  let b := 20
  let c := -25
  let Δ := b^2 - 4*a*c
  let x1 := (-b + sqrt Δ) / (2*a)
  let x2 := (-b - sqrt Δ) / (2*a)
  (x1 + x2 = -b / a) → 
  (x1 * x2 = c / a) → 
  (x1^2 + x2^2 = 26) := 
by 
  intros a b c Δ x1 x2 add_root prod_root -- These are the conditions to use
  sorry

end sum_of_squares_of_roots_l802_802425


namespace arithmetic_seq_problem_l802_802189

-- Define the arithmetic sequence and sum
def arithmetic_seq (a d : ℝ) (n : ℕ) : ℝ :=
  a + d * (n - 1)

def seq_sum (a d : ℝ) (n : ℕ) : ℝ :=
  n * a + (n * (n - 1) * d) / 2

-- Given conditions and proof statement
theorem arithmetic_seq_problem (a d : ℝ) 
  (h₁ : seq_sum a d 4 = 1)
  (h₂ : seq_sum a d 8 = 4) :
  arithmetic_seq a d 17 + arithmetic_seq a d 18 + arithmetic_seq a d 19 + arithmetic_seq a d 20 = 9 :=
sorry

end arithmetic_seq_problem_l802_802189


namespace necessary_condition_of_inequality_l802_802808

variable {A B C D : ℝ}
variable h : C < D

theorem necessary_condition_of_inequality (h : C < D) :
  (C < D → A > B) ↔ (A > B) :=
sorry

end necessary_condition_of_inequality_l802_802808


namespace peter_walks_more_time_l802_802731

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

end peter_walks_more_time_l802_802731


namespace fraction_books_left_l802_802771

theorem fraction_books_left (initial_books sold_books remaining_books : ℕ)
  (h1 : initial_books = 9900) (h2 : sold_books = 3300) (h3 : remaining_books = initial_books - sold_books) :
  (remaining_books : ℚ) / initial_books = 2 / 3 :=
by
  sorry

end fraction_books_left_l802_802771


namespace correct_sunset_time_l802_802720

def daylight_hours : ℕ := 11
def daylight_minutes : ℕ := 7
def sunrise_hour : ℕ := 6
def sunrise_minute : ℕ := 24

theorem correct_sunset_time :
  let sunset_hour := sunrise_hour + daylight_hours,
      sunset_minute := sunrise_minute + daylight_minutes,
      correct_hour := if sunset_minute < 60 then sunset_hour else sunset_hour + 1,
      correct_minute := if sunset_minute < 60 then sunset_minute else sunset_minute - 60,
      pm_hour := if correct_hour > 12 then correct_hour - 12 else correct_hour
  in pm_hour = 5 ∧ correct_minute = 31 := 
by
  -- Essentially we are adding hours and minutes and converting them
  let sunset_hour := sunrise_hour + daylight_hours
  let sunset_minute := sunrise_minute + daylight_minutes
  let correct_hour := if sunset_minute < 60 then sunset_hour else sunset_hour + 1
  let correct_minute := if sunset_minute < 60 then sunset_minute else sunset_minute - 60
  let pm_hour := if correct_hour > 12 then correct_hour - 12 else correct_hour
  -- Stating pm_hour is 5 and correct_minute is 31 satisfies the conditions
  have h1 : pm_hour = 5 := by {
    -- Calculate the hours and adjust for 12-hour format
    simp [sunset_hour, correct_hour, pm_hour]
  }
  have h2 : correct_minute = 31 := by {
    -- Calculate the minutes and adjust for overflow
    simp [sunset_minute, correct_minute]
  }
  exact ⟨h1, h2⟩

end correct_sunset_time_l802_802720


namespace binary_conversion_correct_l802_802429

def binary_to_decimal (binary : String) : Float :=
  let int_part := binary.splitOn "." |>.head!
  let frac_part := binary.splitOn "." |>.tail |>.headD ""
  let int_value := int_part.foldl (fun acc c => 2 * acc + (c.toNat - '0'.toNat)) 0
  let frac_value := frac_part.enum.foldl (fun acc ⟨idx, c⟩ => acc + (c.toNat - '0'.toNat) * 2^(-(idx + 1))) 0.0
  int_value.toFloat + frac_value

def binary_number_eq_decimal : Prop :=
  binary_to_decimal "111.11" = 7.75

theorem binary_conversion_correct : binary_number_eq_decimal :=
by
  sorry

end binary_conversion_correct_l802_802429


namespace estimate_students_like_design_A_l802_802368

theorem estimate_students_like_design_A (surveyed_students : ℕ) (total_students : ℕ) (liked_design_A_in_survey : ℕ) :
  surveyed_students = 100 → total_students = 2000 → liked_design_A_in_survey = 60 →
  let estimated_total_liked_design_A := (liked_design_A_in_survey * total_students) / surveyed_students in
  estimated_total_liked_design_A = 1200 :=
by
  intros h1 h2 h3
  have h4 : surveyed_students = 100 := h1
  have h5 : total_students = 2000 := h2
  have h6 : liked_design_A_in_survey = 60 := h3
  let estimated_total_liked_design_A := (liked_design_A_in_survey * total_students) / surveyed_students
  show estimated_total_liked_design_A = 1200 
  sorry

end estimate_students_like_design_A_l802_802368


namespace xiaoxiao_types_faster_l802_802830

-- Defining the characters typed and time taken by both individuals
def characters_typed_taoqi : ℕ := 200
def time_taken_taoqi : ℕ := 5
def characters_typed_xiaoxiao : ℕ := 132
def time_taken_xiaoxiao : ℕ := 3

-- Calculating typing speeds
def speed_taoqi : ℕ := characters_typed_taoqi / time_taken_taoqi
def speed_xiaoxiao : ℕ := characters_typed_xiaoxiao / time_taken_xiaoxiao

-- Proving that 笑笑 types faster
theorem xiaoxiao_types_faster : speed_xiaoxiao > speed_taoqi := by
  -- Given calculations:
  -- speed_taoqi = 40
  -- speed_xiaoxiao = 44
  sorry

end xiaoxiao_types_faster_l802_802830


namespace decreasing_function_iff_k_lt_2_l802_802094

theorem decreasing_function_iff_k_lt_2 (k : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → (2 * k - 4) * x1 - 1 > (2 * k - 4) * x2 - 1) ↔ k < 2 :=
by
  split
  { intro h
    have h' := h 0 1 zero_lt_one
    sorry },
  { intro h
    intros x1 x2 hlt
    rw [← sub_lt_iff_lt_add', mul_sub (2 * k - 4) x1 x2, mul_sub (2 * k - 4) x1 1, mul_sub (2 * k - 4) x2 1]
    have h' : 2 * k - 4 < 0 := by linarith
    linarith }

end decreasing_function_iff_k_lt_2_l802_802094


namespace math_city_police_officers_needed_l802_802234

/-- In Math City, there are 10 streets such that no street is parallel to another
    and no three streets meet at a single point. We want to determine the number of intersections 
    when these 10 streets are added into the city. -/
theorem math_city_police_officers_needed : 
  (∑ i in Finset.range 9, (i + 1)) = 45 :=
by
  sorry

end math_city_police_officers_needed_l802_802234


namespace candy_problem_l802_802393

theorem candy_problem (a : ℕ) (h₁ : a % 10 = 6) (h₂ : a % 15 = 11) (h₃ : 200 ≤ a) (h₄ : a ≤ 250) :
  a = 206 ∨ a = 236 :=
sorry

end candy_problem_l802_802393


namespace construct_square_with_compass_l802_802420

theorem construct_square_with_compass :
  ∃ A F D G : Point, is_square A F D G ∧ ∀ compass_used : Compass,
  compass_used ∧ paper ∧ ¬(fold_paper) → 
      construct_with_compass (A, F, D, G) :=
by
  sorry

end construct_square_with_compass_l802_802420


namespace box_contents_l802_802795

-- Definitions for the boxes and balls
inductive Ball
| Black | White | Green

-- Define the labels on each box
def label_box1 := "white"
def label_box2 := "black"
def label_box3 := "white or green"

-- Conditions based on the problem
def box1_label := label_box1
def box2_label := label_box2
def box3_label := label_box3

-- Statement of the problem
theorem box_contents (b1 b2 b3 : Ball) 
  (h1 : b1 ≠ Ball.White) 
  (h2 : b2 ≠ Ball.Black) 
  (h3 : b3 = Ball.Black) 
  (h4 : ∀ (x y z : Ball), x ≠ y ∧ y ≠ z ∧ z ≠ x → 
        (x = b1 ∨ y = b1 ∨ z = b1) ∧
        (x = b2 ∨ y = b2 ∨ z = b2) ∧
        (x = b3 ∨ y = b3 ∨ z = b3)) : 
  b1 = Ball.Green ∧ b2 = Ball.White ∧ b3 = Ball.Black :=
sorry

end box_contents_l802_802795


namespace find_value_l802_802954

theorem find_value (x : ℝ) (h : x^2 - 2 * x = 1) : 2023 + 6 * x - 3 * x^2 = 2020 := 
by 
sorry

end find_value_l802_802954


namespace original_number_contains_digit_ge_5_l802_802798

theorem original_number_contains_digit_ge_5 (num : ℕ)
  (no_zero_digits : ∀ d ∈ digit_list num, d ≠ 0)
  (rearranged_sums_to_all_ones : ∃ n1 n2 n3 : ℕ, digit_list n1 = digit_list num ∧
    digit_list n2 = digit_list num ∧ digit_list n3 = digit_list num ∧
    num + n1 + n2 + n3 = all_ones (len (digit_list num))) :
  ∃ d ∈ digit_list num, d ≥ 5 :=
by
  sorry

end original_number_contains_digit_ge_5_l802_802798


namespace four_digit_square_numbers_l802_802871

theorem four_digit_square_numbers (a b c d k : ℕ) :
  let x := 1000 * a + 100 * b + 10 * c + d in
  let y := 1000 * (a - k) + 100 * (b - k) + 10 * (c - k) + (d - k) in
  1000 ≤ x ∧ x ≤ 9999 ∧ 1000 ≤ y ∧ y ≤ 9999 ∧
  (∃ (x_num y_num : ℕ), x = x_num^2 ∧ y = y_num^2) →
  k > 0 ∧ k ≤ min (min (min a b) c) d →
  (x = 3136 ∨ x = 4489) :=
sorry

end four_digit_square_numbers_l802_802871


namespace fran_speed_l802_802678

theorem fran_speed (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) 
    (h_joann : joann_speed = 15) (h_joann_time : joann_time = 4) (h_fran_time : fran_time = 5) : 
    (joann_speed * joann_time) / fran_time = 12 :=
by
  rw [h_joann, h_joann_time, h_fran_time]
  norm_num
  sorry

end fran_speed_l802_802678


namespace area_of_quadrilateral_PQST_l802_802175

theorem area_of_quadrilateral_PQST (P Q R S T : Type) 
  [InnerProductSpace ℝ P] 
  [InnerProductSpace ℝ Q] 
  [InnerProductSpace ℝ R] 
  [InnerProductSpace ℝ S] 
  [InnerProductSpace ℝ T] 
  (PR : ℝ) 
  (h₁ : PR = 18) 
  (h₂ : angle P R Q = 60) 
  (h₃ : angle Q R S = 60)
  (h₄ : angle R S T = 60)
  (h₅ : orthogonal P Q) 
  (h₆ : orthogonal Q R) 
  (h₇ : orthogonal R S): 
  ∃ (area : ℝ), area = 405 * Real.sqrt 3 / 4 :=
by {
  sorry
}

end area_of_quadrilateral_PQST_l802_802175


namespace complex_addition_l802_802584

noncomputable def i : ℂ := complex.I
def z1 := (2 : ℂ) + i
def z2 := (1 : ℂ) - 2 * i

theorem complex_addition :
  z1 + z2 = 3 - i :=
sorry

end complex_addition_l802_802584


namespace angle_B_eq_pi_div_3_triangle_is_equilateral_l802_802193

variables {A B C a b c : ℝ}

-- Conditions
axiom cos_A_cos_C_plus_sin_A_sin_C_plus_cos_B_eq_3_div_2 :
  cos A * cos C + sin A * sin C + cos B = 3 / 2

axiom sides_form_GP : b^2 = a * c

-- Theorem (I)
theorem angle_B_eq_pi_div_3 (h1 : cos A * cos C + sin A * sin C + cos B = 3 / 2)
  (h2 : b^2 = a * c) : B = π / 3 :=
sorry

-- Additional condition for (II)
axiom tan_condition : a / tan A + c / tan C = 2 * b / tan B

-- Theorem (II)
theorem triangle_is_equilateral (h1 : cos A * cos C + sin A * sin C + cos B = 3 / 2)
  (h2 : b^2 = a * c) (h3 : a / tan A + c / tan C = 2 * b / tan B)
  : A = π / 3 ∧ B = π / 3 ∧ C = π / 3 :=
sorry

end angle_B_eq_pi_div_3_triangle_is_equilateral_l802_802193


namespace percentage_less_than_l802_802167
noncomputable theory

theorem percentage_less_than (x y : ℝ) (h : y = 1.6 * x) : ((y - x) / y) * 100 = 37.5 :=
by sorry

end percentage_less_than_l802_802167


namespace calculation_a_squared_plus_b_squared_calculation_a_minus_b_squared_l802_802950

theorem calculation_a_squared_plus_b_squared
  (a b : ℝ)
  (h1 : a + b = 2)
  (h2 : a * b = -1) :
  a^2 + b^2 = 6 := by
  sorry

theorem calculation_a_minus_b_squared
  (a b : ℝ)
  (h1 : a + b = 2)
  (h2 : a * b = -1) :
  (a - b)^2 = 8 := by
  sorry

end calculation_a_squared_plus_b_squared_calculation_a_minus_b_squared_l802_802950


namespace sum_common_divisors_60_18_l802_802526

def divisors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ m => n % m = 0)

noncomputable def sum (l : List ℕ) : ℕ :=
  l.foldr (· + ·) 0

theorem sum_common_divisors_60_18 :
  sum (List.filter (λ d => d ∈ divisors 18) (divisors 60)) = 12 :=
by
  sorry

end sum_common_divisors_60_18_l802_802526


namespace intersect_not_A_B_l802_802132

open Set

-- Define the universal set U
def U := ℝ

-- Define set A
def A := {x : ℝ | x ≤ 3}

-- Define set B
def B := {x : ℝ | x ≤ 6}

-- Define the complement of A in U
def not_A := {x : ℝ | x > 3}

-- The proof problem
theorem intersect_not_A_B :
  (not_A ∩ B) = {x : ℝ | 3 < x ∧ x ≤ 6} :=
sorry

end intersect_not_A_B_l802_802132


namespace shortest_distance_l802_802696

noncomputable def parabola (x : ℝ) : ℝ := x^2 - 4 * x + 20
noncomputable def line (x : ℝ) : ℝ := x - 6

def distance_point_to_line (a : ℝ) : ℝ :=
  abs (a - (parabola a) - 6) / real.sqrt 2

theorem shortest_distance :
  ∃ a, ∀ b, distance_point_to_line a ≤ distance_point_to_line b :=
begin
  use 5/2,
  intro b,
  have h : distance_point_to_line (5/2) = 103 * real.sqrt 2 / 8,
  sorry, -- proof goes here
  rw h,
  refine le_of_eq _,
  sorry, -- proof goes here
end

end shortest_distance_l802_802696


namespace deductive_reasoning_is_general_to_specific_l802_802904

-- Definitions for types of reasoning
def reasoning_from_part_to_whole (r : Type) : Prop :=
  ∃ (a b : Type), ∃ (spec : r → a) (whole : a → b), ∀ (x : r), whole (spec x) = x

def reasoning_from_specifics_to_specifics (r : Type) : Prop :=
  ∃ (a b : Type), ∃ (spec1 spec2 : r → a), ∀ (x y : r), spec1 x = spec2 y → y = x

def reasoning_from_general_to_general (r : Type) : Prop :=
  ∃ (a b : Type), ∃ (gen1 gen2 : a → b), ∀ (x y : a), gen1 (gen2 x) = gen2 y → x = y

def reasoning_from_general_to_specific (r : Type) : Prop :=
  ∃ (a b : Type), ∃ (gen : a → b) (spec : b → a), ∀ (x : a), spec (gen x) = x

-- Statement of the problem
theorem deductive_reasoning_is_general_to_specific (r : Type) :
  reasoning_from_general_to_specific r :=
sorry

end deductive_reasoning_is_general_to_specific_l802_802904


namespace randy_blocks_left_l802_802743

-- Formalize the conditions
def initial_blocks : ℕ := 78
def blocks_used_first_tower : ℕ := 19
def blocks_used_second_tower : ℕ := 25

-- Formalize the result for verification
def blocks_left : ℕ := initial_blocks - blocks_used_first_tower - blocks_used_second_tower

-- State the theorem to be proven
theorem randy_blocks_left :
  blocks_left = 34 :=
by
  -- Not providing the proof as per instructions
  sorry

end randy_blocks_left_l802_802743


namespace petya_can_determine_numbers_l802_802310

theorem petya_can_determine_numbers (cards : Fin 99 → ℕ) 
  (neighbors : ∀ i : Fin 99, (cards i, cards (i + 1) % 99)) :
  ∃ card_numbers : Fin 99 → ℕ, card_numbers = cards := 
sorry

end petya_can_determine_numbers_l802_802310


namespace min_value_f_at_3_f_increasing_for_k_neg4_l802_802121

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x + k / (x - 1)

-- Problem (1): If k = 4, find the minimum value of f(x) and the corresponding value of x.
theorem min_value_f_at_3 : ∃ x > 1, @f x 4 = 5 ∧ x = 3 :=
  sorry

-- Problem (2): If k = -4, prove that f(x) is an increasing function for x > 1.
theorem f_increasing_for_k_neg4 : ∀ ⦃x y : ℝ⦄, 1 < x → x < y → f x (-4) < f y (-4) :=
  sorry

end min_value_f_at_3_f_increasing_for_k_neg4_l802_802121


namespace sum_common_divisors_sixty_and_eighteen_l802_802520

theorem sum_common_divisors_sixty_and_eighteen : 
  ∑ d in ({d ∈ ({1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60} : finset ℕ) | d ∈ ({1, 2, 3, 6, 9, 18} : finset ℕ)} : finset ℕ), d = 12 :=
by sorry

end sum_common_divisors_sixty_and_eighteen_l802_802520


namespace units_digit_of_expression_l802_802040

theorem units_digit_of_expression :
  let A := 12 + Real.sqrt 245 in
  (A ^ 21 + A ^ 84) % 10 = 6 :=
by
  sorry

end units_digit_of_expression_l802_802040


namespace expected_groups_l802_802786

/-- Given a sequence of k zeros and m ones arranged in random order and divided into 
alternating groups of identical digits, the expected value of the total number of 
groups is 1 + 2 * k * m / (k + m). -/
theorem expected_groups (k m : ℕ) (h_pos : k + m > 0) :
  let total_groups := 1 + 2 * k * m / (k + m)
  total_groups = (1 + 2 * k * m / (k + m)) := by
  sorry

end expected_groups_l802_802786


namespace complex_number_Z2_l802_802659

theorem complex_number_Z2 (Z1 Z2 midpoint : ℂ) : 
  Z1 = 4 + complex.i ∧ midpoint = 1 + 2 * complex.i ∧ 
  midpoint = (Z1 + Z2) / 2 → 
  Z2 = -2 + 3 * complex.i :=
begin
  sorry
end

end complex_number_Z2_l802_802659


namespace temperature_difference_l802_802723

theorem temperature_difference (T_high T_low : ℝ) (h_high : T_high = 9) (h_low : T_low = -1) : 
  T_high - T_low = 10 :=
by
  rw [h_high, h_low]
  norm_num

end temperature_difference_l802_802723


namespace sum_of_coefficients_l802_802621

theorem sum_of_coefficients : 
  ∀ (a : Fin 2025 → ℝ),
    (∀ x : ℝ, (1 + x) * (1 - 2 * x)^2023 = ∑ i in Finset.range 2025, a i * (x ^ i)) →
    a 0 = 1 →
    (∑ i in Finset.range 2025, a i) = -2 →
    (∑ i in Finset.range 2024, a i.succ) = -3 := 
by 
  sorry

end sum_of_coefficients_l802_802621


namespace center_of_mass_of_helical_line_l802_802920

open Real

variables {a b k : ℝ}
def x (t : ℝ) := a * cos t
def y (t : ℝ) := a * sin t
def z (t : ℝ) := b * t
def linear_density (t : ℝ) := k * z t

noncomputable def center_of_mass_coordinates : ℝ × ℝ × ℝ :=
  let dl := sqrt (a^2 + b^2) in
  let I1 := ∫ t in 0..π, (x t) * linear_density t * dl
  let I2 := ∫ t in 0..π, linear_density t * dl
  let I3 := ∫ t in 0..π, (y t) * linear_density t * dl
  let I4 := ∫ t in 0..π, (z t) * linear_density t * dl in
  (I1 / I2, I3 / I2, I4 / I2)

theorem center_of_mass_of_helical_line :
  center_of_mass_coordinates = ( -4 * a / π^2, 2 * a / π, 2 / 3 * b * π ) :=
sorry

end center_of_mass_of_helical_line_l802_802920


namespace gcd_A3_B3_A2_B2_l802_802160

open Nat

theorem gcd_A3_B3_A2_B2 (A B : ℕ) (h_rel_prime : gcd A B = 1) :
  gcd (A^3 + B^3) (A^2 + B^2) = 1 ∨ gcd (A^3 + B^3) (A^2 + B^2) = 2 := 
by 
  sorry

end gcd_A3_B3_A2_B2_l802_802160


namespace larger_solid_volume_l802_802901

/- Define coordinates of the vertices and midpoints -/
def D := (0.0, 0.0, 0.0)
def E := (0.0, 1.0, 0.0)
def F := (1.0, 1.0, 0.0)
def H := (0.0, 1.0, 1.0)
def G := (1.0, 1.0, 1.0)

def P := ((1.0 / 2.0), 1.0, 0.0)
def Q := ((1.0 / 2.0), 1.0, 1.0)

/- Define the plane equation -/
def plane_eq (x y z : ℝ) : Prop := x - y - (1.0 / 2.0) * z = 0

/- Define the volume result that needs to be proved -/
theorem larger_solid_volume : 
  volume_of_larger_solid (cube D E F H G P Q plane_eq) = 43 / 48 :=
sorry

end larger_solid_volume_l802_802901


namespace empty_boxes_when_non_empty_34_l802_802082

theorem empty_boxes_when_non_empty_34 : 
  ∀ (n : ℕ), n = 34 → ∃ (empty_boxes : ℕ), empty_boxes = -1 + 6 * n :=
by 
  intro n hf,
  use (-1 + 6 * n),
  rw hf,
  have empty_boxes_value : -1 + 6 * 34 = 203 := by norm_num,
  exact empty_boxes_value

end empty_boxes_when_non_empty_34_l802_802082


namespace angle_between_vectors_approx_l802_802918

noncomputable def angle_between_vectors := 
let u : ℝ × ℝ × ℝ := (3, -2, 2)
let v : ℝ × ℝ × ℝ := (-2, 2, 1) in
let dot_product := u.1 * v.1 + u.2 * v.2 + u.3 * v.3 in
let norm_u := Real.sqrt (u.1^2 + u.2^2 + u.3^2) in
let norm_v := Real.sqrt (v.1^2 + v.2^2 + v.3^2) in
let cos_theta := dot_product / (norm_u * norm_v) in
Real.acos cos_theta * (180 / Real.pi)

theorem angle_between_vectors_approx : 
    (angle_between_vectors ≈ 138.59) :=
by {
    -- Proof is omitted as specified
    sorry
}

end angle_between_vectors_approx_l802_802918


namespace dave_paid_4_more_than_doug_l802_802046

theorem dave_paid_4_more_than_doug :
  let slices := 8
  let plain_cost := 8
  let anchovy_additional_cost := 2
  let total_cost := plain_cost + anchovy_additional_cost
  let cost_per_slice := total_cost / slices
  let dave_slices := 5
  let doug_slices := slices - dave_slices
  -- Calculate payments
  let dave_payment := dave_slices * cost_per_slice
  let doug_payment := doug_slices * cost_per_slice
  dave_payment - doug_payment = 4 :=
by
  sorry

end dave_paid_4_more_than_doug_l802_802046


namespace toys_produced_each_day_l802_802834

-- Given conditions
def total_weekly_production := 5500
def days_worked_per_week := 4

-- Define daily production calculation
def daily_production := total_weekly_production / days_worked_per_week

-- Proof that daily production is 1375 toys
theorem toys_produced_each_day :
  daily_production = 1375 := by
  sorry

end toys_produced_each_day_l802_802834


namespace sum_of_possible_Bs_l802_802860

noncomputable def last_three_digits (n : ℕ) : ℕ := n % 1000

def is_divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

def possible_Bs := { B | ∃ (B : ℕ), B < 10 ∧ is_divisible_by_8 (100 + B * 10 + 4) }

theorem sum_of_possible_Bs :
  (∑ b in possible_Bs, b) = 21 :=
by sorry

end sum_of_possible_Bs_l802_802860


namespace sqrt_difference_l802_802068

theorem sqrt_difference :
  sqrt (21 + 12 * sqrt 3) - sqrt (21 - 12 * sqrt 3) = 6 :=
sorry

end sqrt_difference_l802_802068


namespace problem_statement_l802_802692

-- Define general conditions
structure Square (α : Type) :=
  (x_min : α)
  (x_max : α)
  (y_min : α)
  (y_max : α)

def is_n_ray_partitional {α : Type} [LinearOrderedField α] (R : Square α) (X : α × α) (n : ℕ) : Prop :=
  ∃ (rays : Fin n → α × α), 
    (∀ i : Fin n, (0 < i.val ∧ i.val < n → (polygon ⟨i, rays i⟩ = 1 / n * area_of R))

def num_100_ray_not_60_ray (R : Square ℝ) : ℕ :=
  let count_100_ray := {X : ℝ × ℝ // is_n_ray_partitional R X 100}.card in
  let count_60_ray  := {X : ℝ × ℝ // is_n_ray_partitional R X 60}.card in
  count_100_ray - count_60_ray

-- Define the number of points that satisfy the conditions
theorem problem_statement : 
  ∀ R : Square ℝ,
  R = ⟨0, 1, 0, 1⟩ →
  num_100_ray_not_60_ray R = 2320 :=
by
  intros,
  -- The proof would go here, but we use sorry to skip it.
  sorry

end problem_statement_l802_802692


namespace rate_of_increase_l802_802289

variable (x y t : ℝ)
variable (c0 w0 wf : ℝ)
variable (rate : ℝ)

-- Constants from the problem
def c0 := 3.20
def w0 := 7.80
def wf := 7.448077718958487

-- Conditions for the change rates
def corn_price (t y : ℝ) := c0 + y * t
def wheat_price (t x : ℝ) := w0 - t * (x * (2.sqrt) - x)

-- Equality condition when prices are the same
def equality_condition (t x y : ℝ) := c0 + y * t = w0 - t * (x * (2.sqrt) - x)

theorem rate_of_increase (x : ℝ) :
    ∃ y : ℝ, equality_condition t x y -> y = x * (2.sqrt) - x := by
    sorry

end rate_of_increase_l802_802289


namespace centroids_mirror_images_l802_802812

-- Define the position vectors of the vertices of the tetrahedron
variables (A B C D : ℝ^3)

-- Define the centroids of each face of the tetrahedron
def centroidABC := (A + B + C) / 3
def centroidABD := (A + B + D) / 3
def centroidACD := (A + C + D) / 3
def centroidBCD := (B + C + D) / 3

-- Define the centroid of the tetrahedron
def centroidTetrahedron := (A + B + C + D) / 4

-- Define arbitrary points on each face
variables (x y z u : ℝ^3)

def pointD1 := centroidABC + x
def pointC1 := centroidABD + y
def pointB1 := centroidACD + z
def pointA1 := centroidBCD + u

-- Define the reflected points
def reflectedD1 := centroidABC - x
def reflectedC1 := centroidABD - y
def reflectedB1 := centroidACD - z
def reflectedA1 := centroidBCD - u

-- Calculate the centroids of the original and reflected points
def centroidOriginal := (pointD1 + pointC1 + pointB1 + pointA1) / 4
def centroidReflected := (reflectedD1 + reflectedC1 + reflectedB1 + reflectedA1) / 4

-- The proof statement
theorem centroids_mirror_images :
  centroidOriginal + centroidReflected = 2 * centroidTetrahedron :=
sorry

end centroids_mirror_images_l802_802812


namespace card_pairs_satisfying_conditions_l802_802434

theorem card_pairs_satisfying_conditions :
  ∃ (s : Finset (ℕ × ℕ)),
    (∀ p ∈ s, p.1 ≠ p.2 ∧ (p.1 = p.2 + 11 ∨ p.2 = p.1 + 11) ∧ ((p.1 * p.2) % 5 = 0))
    ∧ s.card = 15 :=
begin
  sorry
end

end card_pairs_satisfying_conditions_l802_802434


namespace smallest_sum_is_16_l802_802292

def smallest_sum : ℤ :=
  let s := {2, 16, -4, 9, -2}
  let t := s.toFinset
  let candidates := { (a, b, c) | a ∈ t ∧ b ∈ t ∧ c ∈ t ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c }
  let squared_sum := λ (a, b, c) => (min a (min b c))^2 + (a + b + c - min a (min b c))
  finset.min'_on finset.univ squared_sum (2, -4, -2)

theorem smallest_sum_is_16 : smallest_sum = 16 := by
  sorry

end smallest_sum_is_16_l802_802292


namespace max_odd_integers_l802_802407

theorem max_odd_integers (chosen : Fin 5 → ℕ) (hpos : ∀ i, chosen i > 0) (heven : ∃ i, chosen i % 2 = 0) : 
  ∃ odd_count, odd_count = 4 ∧ (∀ i, i < 4 → chosen i % 2 = 1) := 
by 
  sorry

end max_odd_integers_l802_802407


namespace maximum_value_of_f_l802_802282

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 + 3 * x^2 + 5 * x + 2

theorem maximum_value_of_f :
  ∃ x, (∀ y, f(x) ≥ f(y)) ∧ f(-5) = 31 / 3 :=
by
  let f' (x : ℝ) := x^2 + 6 * x + 5
  have h1 : f'(-5) = 0 := by sorry
  have h2 : f'(-1) = 0 := by sorry
  have h3 : ∀ x, x ≠ -1 ∧ x ≠ -5 → (f' x > 0 ↔ (x > -1 ∨ x < -5)) := by sorry
  have h4 : ∀ x, x ≠ -1 ∧ x ≠ -5 → (f' x < 0 ↔ -5 < x ∧ x < -1) := by sorry
  have h_max : f(-5) = 31 / 3 := by sorry
  use -5
  split
  · intro y
    by_cases h5 : y = -5
    · rw h5
    · by_cases h6 : y = -1
      · have h_1 := h_max
        rw ←h6 at h_1
        sorry
      · have h7 : (f' y > 0 ↔ (y > -1 ∨ y < -5)) := h3 y ⟨ne.symm h6, ne.symm h5⟩
        have h8 : (f' y < 0 ↔ -5 < y ∧ y < -1) := h4 y ⟨ne.symm h6, ne.symm h5⟩
        sorry
  exact h_max

end maximum_value_of_f_l802_802282


namespace compute_xy_l802_802806

theorem compute_xy (x y : ℝ) (h1 : x + y = 9) (h2 : x^3 + y^3 = 351) : x * y = 14 :=
by
  sorry

end compute_xy_l802_802806


namespace sum_of_common_divisors_60_18_l802_802543

theorem sum_of_common_divisors_60_18 : 
  let a := 60 
  let b := 18 
  let common_divisors := {n | n ∣ a ∧ n ∣ b ∧ n > 0 } 
  (∑ n in common_divisors, n) = 12 :=
by
  let a := 60
  let b := 18
  let common_divisors := { n | n ∣ a ∧ n ∣ b ∧ n > 0 }
  have : (∑ n in common_divisors, n) = 12 := sorry
  exact this

end sum_of_common_divisors_60_18_l802_802543


namespace cubic_feet_per_bag_l802_802870

-- Definitions
def length_bed := 8 -- in feet
def width_bed := 4 -- in feet
def height_bed := 1 -- in feet
def number_of_beds := 2
def number_of_bags := 16

-- Theorem statement
theorem cubic_feet_per_bag : 
  (length_bed * width_bed * height_bed * number_of_beds) / number_of_bags = 4 :=
by
  sorry

end cubic_feet_per_bag_l802_802870


namespace problem_solution_l802_802340

noncomputable def is_incorrect_A : Prop :=
  (2 : ℝ) ^ (2 / 5) * (2 : ℝ) ^ (5 / 2) ≠ (2 : ℝ)

theorem problem_solution : is_incorrect_A :=
by {
  rw [←pow_add],
  apply ne_of_lt,
  norm_num,
  sorry  -- Detailed proof steps are not required
}

end problem_solution_l802_802340


namespace card_choice_count_l802_802483

theorem card_choice_count :
  ∃ (pairs : Finset (ℕ × ℕ)),
    (∀ (a b : ℕ), (a, b) ∈ pairs → 1 ≤ a ∧ a ≤ 50 ∧ 1 ≤ b ∧ b ≤ 50) ∧
    (∀ (a b : ℕ), (a, b) ∈ pairs → |a - b| = 11) ∧
    (∀ (a b : ℕ), (a, b) ∈ pairs → (a * b) % 5 = 0) ∧
    card pairs = 15 := 
sorry

end card_choice_count_l802_802483


namespace area_difference_approx_l802_802837

def circle_area (r : ℝ) : ℝ := Real.pi * r^2

def circle_radius (c : ℝ) : ℝ := c / (2 * Real.pi)

def area_difference (c1 c2 : ℝ) : ℝ := 
  let r1 := circle_radius c1
  let r2 := circle_radius c2
  circle_area r2 - circle_area r1

theorem area_difference_approx :
  area_difference 704 264 ≈ 33888 := 
by 
  sorry

end area_difference_approx_l802_802837


namespace a3_eq_1_l802_802191

variable {α : Type*} [Field α]

def geometric_sequence (a q : α) (n : ℕ) : α :=
  a * q^n

def product_first_n_terms (a q : α) (n : ℕ) : α :=
  (finset.range n).prod (λ k, geometric_sequence a q k)

theorem a3_eq_1 (a q : α) : product_first_n_terms a q 5 = 1 → a * q^2 = 1 :=
by
  intro h
  have h_eq : (finset.range 5).prod (λ k, geometric_sequence a q k) = (a * q^2)^5 := 
    by sorry
  have a_q_sq := eq_of_pow_eq_one h_eq
  sorry

end a3_eq_1_l802_802191


namespace equivalent_proof_l802_802219

noncomputable def proof_problem : ℝ := 
  let a : ℝ := real.sqrt 14
  let b : ℝ := -real.sqrt 116
  let c : ℝ := real.sqrt 56
  let discriminant : ℝ := b^2 - 4 * a * c
  let x1 : ℝ := (b + real.sqrt discriminant) / (2 * a)
  let x2 : ℝ := (b - real.sqrt discriminant) / (2 * a)
  let term1 := abs ((1 / x1^2) - (1 / x2^2))
  let term2 := abs ((x2^2 - x1^2) / (x1^2 * x2^2))
  let term3 := abs ((x2 - x1) * (x2 + x1) / (x1^2 * x2^2))
  let term4 := real.sqrt 29 / 14

theorem equivalent_proof : 
  let a : ℝ := real.sqrt 14
  let b : ℝ := -real.sqrt 116
  let c : ℝ := real.sqrt 56
  let discriminant : ℝ := b^2 - 4 * a * c
  let x1 : ℝ := (b + real.sqrt discriminant) / (2 * a)
  let x2 : ℝ := (b - real.sqrt discriminant) / (2 * a)
  abs ((1 / x1^2) - (1 / x2^2)) = real.sqrt 29 / 14 := 
  sorry

end equivalent_proof_l802_802219


namespace _l802_802357

noncomputable theorem total_books_3002 (P C B T : ℕ) 
  (h1 : P / C = 3 / 2) 
  (h2 : C / B = 4 / 3) 
  (h3 : T > 3000) 
  (h4 : T = P + C + B) : 
  T = 3002 :=
sorry

end _l802_802357


namespace number_of_integers_between_sqrt10_and_sqrt100_l802_802150

theorem number_of_integers_between_sqrt10_and_sqrt100 : 
  let a := Real.sqrt 10
  let b := Real.sqrt 100
  ∃ (n : ℕ), n = 6 ∧ (∀ x : ℕ, (x > a ∧ x < b) → (4 ≤ x ∧ x ≤ 9)) :=
by
  sorry

end number_of_integers_between_sqrt10_and_sqrt100_l802_802150


namespace binomial_expansion_constant_term_l802_802632

noncomputable theory
open_locale big_operators

theorem binomial_expansion_constant_term (a : ℝ) (h : ∑ (k : ℕ) in finset.range 7, (nat.choose 6 k) * a ^ k * (if 6 - 2 * k = 0 then 1 else 0) = 20) : 
  a = 1 :=
sorry

end binomial_expansion_constant_term_l802_802632


namespace sufficient_condition_x_gt_2_l802_802269

theorem sufficient_condition_x_gt_2 (x : ℝ) (h : x > 2) : x^2 - 2 * x > 0 := by
  sorry

end sufficient_condition_x_gt_2_l802_802269


namespace problem_l802_802001

def generator_output_power : ℝ := 24.5 * 1000 -- in Watts
def output_voltage : ℝ := 350 -- in Volts
def total_resistance : ℝ := 4 -- in Ohms
def power_loss : ℝ := 0.05 * generator_output_power -- 5% of the generator's output power
def required_voltage_user : ℝ := 220 -- in Volts

theorem problem
  (P_1 : ℝ := generator_output_power)
  (U_1 : ℝ := output_voltage)
  (R : ℝ := total_resistance)
  (ΔP : ℝ := power_loss)
  (U_use : ℝ := required_voltage_user) :
  (sqrt (ΔP / R) = 17.5) ∧
  (17.5 / (P_1 / U_1) = 1 / 4) ∧
  ((4 * U_1 - 17.5 * R) / U_use = 133 / 22) :=
by
  split
  . sorry
  . split
  . sorry
  . sorry

end problem_l802_802001


namespace king_minimum_maximum_length_l802_802770

-- Define the conditions
def king_path (n : Nat) := { P : List (Nat × Nat) // P.length = n ∧ P.nodup ∧
(A: Nat) → A < 8 ∧ B: Nat → B < 8 }

-- Minimum and maximum length proof for the king's path on an 8x8 chessboard.
theorem king_minimum_maximum_length :
  (∃ P : king_path 64, closed_path P ∧ ¬ self_intersect P) →
  (∃ (min_len max_len : ℝ ), min_len = 64 ∧ 
  max_len = 28 + 36 * Real.sqrt 2) :=
by
  intro h,
  sorry

end king_minimum_maximum_length_l802_802770


namespace sum_nat_numbers_last_digits_l802_802251

theorem sum_nat_numbers_last_digits (N : ℕ) :
  let S_N := N * (N + 1) / 2 in
  (S_N % 100 ≠ 72) ∧ (S_N % 100 ≠ 73) ∧ (S_N % 100 ≠ 74) ∧
  (S_N % 10 ≠ 2) ∧ (S_N % 10 ≠ 4) ∧ (S_N % 10 ≠ 7) ∧ (S_N % 10 ≠ 9) :=
by
  sorry

end sum_nat_numbers_last_digits_l802_802251


namespace function_range_is_interval_l802_802116

theorem function_range_is_interval :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → -1 ≤ (256 * x^9 - 576 * x^7 + 432 * x^5 - 120 * x^3 + 9 * x) ∧ 
  (256 * x^9 - 576 * x^7 + 432 * x^5 - 120 * x^3 + 9 * x) ≤ 1 := 
by
  sorry

end function_range_is_interval_l802_802116


namespace parallel_segments_have_equal_slopes_l802_802296

theorem parallel_segments_have_equal_slopes
  (A B X : ℝ × ℝ)
  (Y : ℝ × ℝ)
  (hA : A = (-5, -1))
  (hB : B = (2, -8))
  (hX : X = (2, 10))
  (hY1 : Y.1 = 20)
  (h_parallel : (B.2 - A.2) / (B.1 - A.1) = (Y.2 - X.2) / (Y.1 - X.1)) :
  Y.2 = -8 :=
by
  sorry

end parallel_segments_have_equal_slopes_l802_802296


namespace option_D_correct_l802_802827

variable (x : ℝ)

theorem option_D_correct : (2 * x^7) / x = 2 * x^6 := sorry

end option_D_correct_l802_802827


namespace card_pairs_with_conditions_l802_802454

theorem card_pairs_with_conditions : 
  let cards := Finset.range 51 in
  let pairs := (cards.product cards).filter (λ p, (p.1 - p.2).natAbs = 11 ∧ (p.1 * p.2) % 5 = 0) in
  pairs.card / 2 = 15 :=
by
  sorry

end card_pairs_with_conditions_l802_802454


namespace least_positive_int_satisfies_congruence_l802_802067

theorem least_positive_int_satisfies_congruence :
  ∃ x : ℕ, (x + 3001) % 15 = 1723 % 15 ∧ x = 12 :=
by
  sorry

end least_positive_int_satisfies_congruence_l802_802067


namespace percent_of_percent_l802_802325

theorem percent_of_percent (y : ℝ) : 0.3 * (0.6 * y) = 0.18 * y :=
sorry

end percent_of_percent_l802_802325


namespace percent_calculation_l802_802332

theorem percent_calculation (y : ℝ) : (0.3 * (0.6 * y) = 0.18 * y) → (0.18 * y / y) * 100 = 18 := by
  sorry

end percent_calculation_l802_802332


namespace fence_poles_count_l802_802376

def length_path : ℕ := 900
def length_bridge : ℕ := 42
def distance_between_poles : ℕ := 6

theorem fence_poles_count :
  2 * (length_path - length_bridge) / distance_between_poles = 286 :=
by
  sorry

end fence_poles_count_l802_802376


namespace common_denominator_step1_error_in_step3_simplified_expression_l802_802793

theorem common_denominator_step1 (x : ℝ) (h1: x ≠ 2) (h2: x ≠ -2):
  (3 * x / (x - 2) - x / (x + 2)) = (3 * x * (x + 2)) / ((x - 2) * (x + 2)) - (x * (x - 2)) / ((x - 2) * (x + 2)) :=
sorry

theorem error_in_step3 (x : ℝ) (h1: x ≠ 2) (h2: x ≠ -2) :
  (3 * x^2 + 6 * x - (x^2 - 2 * x)) / ((x - 2) * (x + 2)) ≠ (3 * x^2 + 6 * x * (x^2 - 2 * x)) / ((x - 2) * (x + 2)) :=
sorry

theorem simplified_expression (x : ℝ) (h1: x ≠ 0) (h2: x ≠ 2) (h3: x ≠ -2) :
  ((3 * x / (x - 2) - x / (x + 2)) * (x^2 - 4) / x) = 2 * x + 8 :=
sorry

end common_denominator_step1_error_in_step3_simplified_expression_l802_802793


namespace dan_initial_money_l802_802902

example (candy_bars : ℕ) (cost_per_candy : ℕ) (leftover_money : ℕ) (total_money_initially : ℕ) : Prop :=
  candy_bars = 99 ∧
  cost_per_candy = 3 ∧
  leftover_money = 1 ∧
  total_money_initially = (candy_bars * cost_per_candy) + leftover_money

theorem dan_initial_money :
  example 99 3 1 298 :=
by
  unfold example
  exact ⟨rfl, rfl, rfl, rfl⟩

end dan_initial_money_l802_802902


namespace sum_of_first_and_third_l802_802295

theorem sum_of_first_and_third :
  ∀ (A B C : ℕ),
  A + B + C = 330 →
  A = 2 * B →
  C = A / 3 →
  B = 90 →
  A + C = 240 :=
by
  intros A B C h1 h2 h3 h4
  sorry

end sum_of_first_and_third_l802_802295


namespace f_decreasing_and_solution_set_l802_802273

noncomputable def f : ℝ → ℝ := sorry -- Placeholder definition

-- Conditions as helper lemmas
axiom f_domain : ∀ x : ℝ, 0 < x → x ∈ set.Ioi 0
axiom f_mul_prop : ∀ x y : ℝ, 0 < x → 0 < y → f(x * y) = f(x) + f(y)
axiom f_lt_0 : ∀ x : ℝ, 1 < x → f(x) < 0
axiom f_half : f(1 / 2) = 2

-- Main theorem statement
theorem f_decreasing_and_solution_set :
  (∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → f(x2) < f(x1)) ∧
  (set_of (λ x : ℝ, f(x) + f(x-1) + 2 > 0) = set.Ioo 1 2) :=
begin
  sorry -- Here would be the proof according to the conditions and requirements
end

end f_decreasing_and_solution_set_l802_802273


namespace range_of_m_l802_802564

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (x-1)^2 < m^2 → |1 - (x-1)/3| < 2) → (abs m ≤ 3) :=
by
  sorry

end range_of_m_l802_802564


namespace extreme_value_g_f_pos_when_m_le_2_l802_802100

noncomputable def g (x : ℝ) := Real.exp x - x - 1
noncomputable def f (x m : ℝ) := Real.exp x - Real.log (x + m)

theorem extreme_value_g : ∃ x : ℝ, (∀ y : ℝ, g(x) ≤ g(y)) ∧ g(0) = 0 := 
sorry

theorem f_pos_when_m_le_2 (m : ℝ) (h : m ≤ 2) : ∀ x : ℝ, f x m > 0 :=
sorry

end extreme_value_g_f_pos_when_m_le_2_l802_802100


namespace sum_remainders_mod_500_l802_802203

theorem sum_remainders_mod_500 :
  let Q := {n : ℕ | ∃ k : ℕ, k < 31 ∧ n = 2^k % 500} in
  let S := ∑ n in Q, n in
  S % 500 = 412 :=
by
  sorry

end sum_remainders_mod_500_l802_802203


namespace problem_statement_l802_802114

noncomputable def quadrant (z : ℂ) : ℕ :=
  if h1 : z.re > 0 then
    if h2 : z.im > 0 then 1
    else if h2 : z.im < 0 then 4
    else sorry -- case for z.im = 0
  else if h1 : z.re < 0 then
    if h2 : z.im > 0 then 2
    else if h2 : z.im < 0 then 3
    else sorry -- case for z.im = 0
  else sorry -- case for z.re = 0

theorem problem_statement : (∃ z : ℂ, z * (complex.i) = 2 + complex.i ∧ quadrant z = 4) :=
sorry

end problem_statement_l802_802114


namespace lcm_12_18_l802_802922

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l802_802922


namespace initial_men_work_count_l802_802260

-- Define conditions given in the problem
def work_rate (M : ℕ) := 1 / (40 * M)
def initial_men_can_complete_work_in_40_days (M : ℕ) : Prop := M * work_rate M * 40 = 1
def work_done_by_initial_men_in_16_days (M : ℕ) := (M * 16) * work_rate M
def remaining_work_done_by_remaining_men_in_40_days (M : ℕ) := ((M - 14) * 40) * work_rate M

-- Define the main theorem to prove
theorem initial_men_work_count (M : ℕ) :
  initial_men_can_complete_work_in_40_days M →
  work_done_by_initial_men_in_16_days M = 2 / 5 →
  3 / 5 = (remaining_work_done_by_remaining_men_in_40_days M) →
  M = 15 :=
by
  intros h_initial h_16_days h_remaining
  have rate := h_initial
  sorry

end initial_men_work_count_l802_802260


namespace scientific_notation_of_6_ronna_l802_802018

def "ronna" := 27

theorem scientific_notation_of_6_ronna :
  "ronna" = 27 → 6 * 10^27 = 6 * 10^27 := 
by
  intro ronna_def
  rw ronna_def
  exact rfl

end scientific_notation_of_6_ronna_l802_802018


namespace max_distance_sum_convex_polygon_l802_802103

theorem max_distance_sum_convex_polygon (P : ℝ × ℝ) (A : Finset (ℝ × ℝ)) 
  (h : ∀ (Q : ℝ × ℝ), Q ∈ ConvexHull(ℝ × Σ i : Finset.univ A, (univ : i.val → set (ℝ × ℝ)))) : 
  (∃ (i : Finset.univ A), P = (i : ℝ × ℝ)) → 
  (∀ (Q : ℝ × ℝ), Q ∈ (A.to_set : Set (ℝ × ℝ)) → 
  ∑ i in A, (dist Q P) ≤ ∑ i in A, (dist i.val P)) :=
sorry

end max_distance_sum_convex_polygon_l802_802103


namespace product_of_solutions_l802_802930

theorem product_of_solutions :
  let solutions := {x : ℝ | |x| = 3 * (|x| - 2)} in
  ∏ x in solutions, x = -9 := by
  sorry

end product_of_solutions_l802_802930


namespace max_ab_bc_cd_l802_802708

-- Definitions of nonnegative numbers and their sum condition
variables (a b c d : ℕ) 
variables (h_sum : a + b + c + d = 120)

-- The goal to prove
theorem max_ab_bc_cd : ab + bc + cd <= 3600 :=
sorry

end max_ab_bc_cd_l802_802708


namespace find_range_of_a_l802_802229

noncomputable def range_of_a (a : ℝ) : Prop :=
∀ (x : ℝ) (θ : ℝ), (0 ≤ θ ∧ θ ≤ (Real.pi / 2)) → 
  let α := (x + 3, x)
  let β := (2 * Real.sin θ * Real.cos θ, a * Real.sin θ + a * Real.cos θ)
  let sum := (α.1 + β.1, α.2 + β.2)
  (sum.1^2 + sum.2^2)^(1/2) ≥ Real.sqrt 2

theorem find_range_of_a : range_of_a a ↔ (a ≤ 1 ∨ a ≥ 5) :=
sorry

end find_range_of_a_l802_802229


namespace vehicle_speed_l802_802015

theorem vehicle_speed (distance : ℝ) (time : ℝ) (h_dist : distance = 150) (h_time : time = 0.75) : distance / time = 200 :=
  by
    sorry

end vehicle_speed_l802_802015


namespace prob_top_odd_correct_l802_802364

def total_dots : Nat := 78
def faces : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Probability calculation for odd dots after removal
def prob_odd_dot (n : Nat) : Rat :=
  if n % 2 = 1 then
    1 - (n : Rat) / total_dots
  else
    (n : Rat) / total_dots

-- Probability that the top face shows an odd number of dots
noncomputable def prob_top_odd : Rat :=
  (1 / (faces.length : Rat)) * (faces.map prob_odd_dot).sum

theorem prob_top_odd_correct :
  prob_top_odd = 523 / 936 :=
by
  sorry

end prob_top_odd_correct_l802_802364


namespace product_of_numerator_denominator_of_periodic_decimal_l802_802315

theorem product_of_numerator_denominator_of_periodic_decimal :
  let x := (6 : ℚ) / 999
  in let y := 2 / 333
  in x = y →
  (y.num * y.denom = 666) :=
by
  intros x y hxy
  have : x = 2 / 333, from hxy
  have hnum_denom : y.num = 2 ∧ y.denom = 333, sorry
  rw this at hnum_denom,
  exact sorry

end product_of_numerator_denominator_of_periodic_decimal_l802_802315


namespace min_value_x_y_l802_802955

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1 / x + 4 / y + 8) : 
  x + y ≥ 9 :=
sorry

end min_value_x_y_l802_802955


namespace sum_of_solutions_tan_equation_l802_802491

theorem sum_of_solutions_tan_equation :
  let solutions := {x : ℝ | 0 ≤ x ∧ x < π ∧ tan x * tan x - 12 * tan x + 4 = 0} in
  ∑ x in solutions, x = π := sorry

end sum_of_solutions_tan_equation_l802_802491


namespace range_of_m_l802_802599

noncomputable def f (x : ℝ) : ℝ := (1 / 2) ^ x

theorem range_of_m (m : ℝ) : f m > 1 → m < 0 := by
  sorry

end range_of_m_l802_802599


namespace difference_of_sums_l802_802815

theorem difference_of_sums :
  let O := (list.range 101).map (λ n, 2 * n + 1),
      E := (list.range 101).map (λ n, 2 * (n + 1)) in
  (E.sum - O.sum) = 150 := 
by
  sorry

end difference_of_sums_l802_802815


namespace parabola_expression_exists_l802_802113

namespace Parabola

def parabola_vertex_on_line (a b c : ℝ) : Prop :=
  b^2 - 2 * b - 4 * a * c = 0

def parabola_opens_downwards (a : ℝ) : Prop :=
  a < 0

theorem parabola_expression_exists :
  ∃ (a b c : ℝ), parabola_opens_downwards a ∧ parabola_vertex_on_line a b c ∧ (y = a * x^2 + b * x + c) :=
by 
  use [-1, 0, 0]
  split
  { -- Parabola opens downwards condition
    show parabola_opens_downwards (-1),
    sorry },
  split
  { -- Vertex on line condition
    show parabola_vertex_on_line (-1) 0 0,
    sorry },
  { -- Expression of the parabola
    sorry }

end Parabola

end parabola_expression_exists_l802_802113


namespace points_concyclic_l802_802887

noncomputable def circumcenter (A B C : Point) : Point := sorry
noncomputable def is_tangent (C1 C2 P : Point) : Prop := sorry
noncomputable def intersection_points (C1 C2 : Point) : set (Point) := sorry
noncomputable def is_on_side (P A B : Point) : Prop := sorry
noncomputable def is_concyclic (A B C D : Point) : Prop := sorry

variables {A B C O O1 O2 D E : Point}

theorem points_concyclic 
  (h1 : O = circumcenter A B C) 
  (h2 : D ∈ intersection_points O1 O2)
  (h3 : E ∈ intersection_points O1 O2)
  (h4 : is_on_side D B C)
  (h5 : is_tangent O1 A B)
  (h6 : is_tangent O2 A C) :
  is_concyclic O O1 E O2 :=
sorry

end points_concyclic_l802_802887


namespace matilda_father_bars_left_l802_802235

def matilda_chocolates : ℕ := 60
def kept_percentage : ℝ := 0.125
def family_members : ℕ := 4 + 2 + 1 -- 4 sisters, 2 cousins, 1 aunt
def chocolates_given (total_bars : ℝ) (percentage_kept : ℝ) (members : ℕ) : ℝ :=
  (total_bars * (1 - percentage_kept)) / members

def father_initial_bars (bars_per_person : ℝ) (members : ℕ) : ℝ :=
  (bars_per_person / 3) * members

def chocolates_distributed : ℝ :=
  chocolates_given matilda_chocolates kept_percentage family_members

def father_final_bars (initial_bars : ℝ) :=
  initial_bars - (5.5 + 3.25 + 2 + 1 + 2)

theorem matilda_father_bars_left :
  father_final_bars (father_initial_bars chocolates_distributed family_members) = 3.75 :=
by sorry

end matilda_father_bars_left_l802_802235


namespace problem_D_l802_802609

theorem problem_D (a b c : ℝ) (h : |a^2 + b + c| + |a + b^2 - c| ≤ 1) : a^2 + b^2 + c^2 < 100 := 
sorry

end problem_D_l802_802609


namespace a_2014_eq_1_l802_802663

-- Define the sequence a_n
def a : ℕ → ℕ
| 0     := 0  -- Note: a_0 is not actually used
| 1     := 1
| 2     := 3
| (n+3) := abs (a (n+2) - a (n+1))

-- Statement of the problem
theorem a_2014_eq_1 : a 2014 = 1 := 
sorry

end a_2014_eq_1_l802_802663


namespace james_marbles_left_l802_802195

theorem james_marbles_left :
  ∀ (initial_marbles bags remaining_bags marbles_per_bag left_marbles : ℕ),
  initial_marbles = 28 →
  bags = 4 →
  marbles_per_bag = initial_marbles / bags →
  remaining_bags = bags - 1 →
  left_marbles = remaining_bags * marbles_per_bag →
  left_marbles = 21 :=
by
  intros initial_marbles bags remaining_bags marbles_per_bag left_marbles
  sorry

end james_marbles_left_l802_802195


namespace angle_between_vectors_approx_l802_802919

noncomputable def angle_between_vectors := 
let u : ℝ × ℝ × ℝ := (3, -2, 2)
let v : ℝ × ℝ × ℝ := (-2, 2, 1) in
let dot_product := u.1 * v.1 + u.2 * v.2 + u.3 * v.3 in
let norm_u := Real.sqrt (u.1^2 + u.2^2 + u.3^2) in
let norm_v := Real.sqrt (v.1^2 + v.2^2 + v.3^2) in
let cos_theta := dot_product / (norm_u * norm_v) in
Real.acos cos_theta * (180 / Real.pi)

theorem angle_between_vectors_approx : 
    (angle_between_vectors ≈ 138.59) :=
by {
    -- Proof is omitted as specified
    sorry
}

end angle_between_vectors_approx_l802_802919


namespace compare_abc_l802_802951

noncomputable def a : ℝ := 2 + (1 / 5) * Real.log 2
noncomputable def b : ℝ := 1 + Real.exp (0.2 * Real.log 2)
noncomputable def c : ℝ := Real.exp (1.1 * Real.log 2)

theorem compare_abc : a < c ∧ c < b := by
  sorry

end compare_abc_l802_802951


namespace interior_diagonals_of_dodecahedron_l802_802488

-- Define a type for dodecahedron and its properties
structure Dodecahedron :=
  (vertices : Nat)
  (faces : Nat)
  (vertices_per_face : Nat)
  (faces_per_vertex : Nat)
  (vertices_on_common_face : Nat → Nat)

-- Define the specific dodecahedron in question
def dodecahedron : Dodecahedron :=
  {
    vertices := 20,
    faces := 12,
    vertices_per_face := 5,
    faces_per_vertex := 3,
    vertices_on_common_face := λ v, 3
  }

-- Define the theorem to be proven
theorem interior_diagonals_of_dodecahedron : dodecahedron.vertices * (dodecahedron.vertices - dodecahedron.vertices_on_common_face 0) / 2 = 160 :=
by sorry

end interior_diagonals_of_dodecahedron_l802_802488


namespace contemporary_probability_l802_802805

noncomputable def lifespan_distribution := uniform 50 120

def probability_of_contemporary : ℝ :=
  let overlap_area := 640000 - 7245 in
  overlap_area / 640000

theorem contemporary_probability : 
  (∀ (L_Alice L_Bob : ℝ), (L_Alice ~ lifespan_distribution) ∧ (L_Bob ~ lifespan_distribution) →
  P((L_Alice and L_Bob are contemporaries)) = probability_of_contemporary :=
sorry

end contemporary_probability_l802_802805


namespace parallelogram_perimeter_l802_802517

theorem parallelogram_perimeter (a b : ℕ) (ha : a = 18) (hb : b = 12) : 2 * a + 2 * b = 60 :=
by
  -- conditions
  rw [ha, hb]
  -- calculation
  calc
    2 * 18 + 2 * 12 = 36 + 24 : by norm_num
    ... = 60 : by norm_num

end parallelogram_perimeter_l802_802517


namespace reflection_image_3m_plus_2b_value_l802_802280

/-- Define the problem and its conditions. --/
theorem reflection_image_3m_plus_2b_value :
  ∃ (m b : ℝ), 
    (∃ y : ℝ, y = m * 1 + b ∧ y = -4) ∧ 
    (∃ y : ℝ, y = m * 7 + b ∧ y = 2) ∧ 
    (m = -1) ∧ (b = 3) ∧ 
    (3 * m + 2 * b = 3) :=
begin
  sorry
end

end reflection_image_3m_plus_2b_value_l802_802280


namespace fiona_categorizations_l802_802070

theorem fiona_categorizations (n : ℕ) (r : ℕ) (categories : ℕ) 
  (Hn : n = 12) (Hr : r = 2) (Hcategories : categories = 3) : 
  (nat.choose n r) * categories = 198 := 
by
  -- Using the combination formula
  have Hcomb : nat.choose n r = 66 := 
    by
      rw [Hn, Hr]
      exact nat.choose_symm n r
      done
  rw [Hcomb, Hcategories]
  norm_num
  done
  sorry

end fiona_categorizations_l802_802070


namespace quadratic_solution_l802_802284

theorem quadratic_solution (x : ℝ) (h : x^2 - 4 * x + 2 = 0) : x + 2 / x = 4 :=
by sorry

end quadratic_solution_l802_802284


namespace card_pairs_satisfying_conditions_l802_802437

theorem card_pairs_satisfying_conditions :
  ∃ (s : Finset (ℕ × ℕ)),
    (∀ p ∈ s, p.1 ≠ p.2 ∧ (p.1 = p.2 + 11 ∨ p.2 = p.1 + 11) ∧ ((p.1 * p.2) % 5 = 0))
    ∧ s.card = 15 :=
begin
  sorry
end

end card_pairs_satisfying_conditions_l802_802437


namespace valentine_problem_l802_802716

def initial_valentines : ℕ := 30
def given_valentines : ℕ := 8
def remaining_valentines : ℕ := 22

theorem valentine_problem : initial_valentines - given_valentines = remaining_valentines := by
  sorry

end valentine_problem_l802_802716


namespace expected_value_of_groups_l802_802783

noncomputable def expectedNumberOfGroups (k m : ℕ) : ℝ :=
  1 + (2 * k * m) / (k + m)

theorem expected_value_of_groups (k m : ℕ) :
  k > 0 → m > 0 → expectedNumberOfGroups k m = 1 + 2 * k * m / (k + m) :=
by
  intros
  unfold expectedNumberOfGroups
  sorry

end expected_value_of_groups_l802_802783


namespace recurring_fraction_sum_eq_l802_802511

theorem recurring_fraction_sum_eq (x : ℝ) (h1 : x = 0.45̅) : 0.3 + x = 83/110 := by
  sorry

end recurring_fraction_sum_eq_l802_802511


namespace minimum_value_of_2_pos_x_l802_802877

theorem minimum_value_of_2_pos_x :
  (∃ x: ℝ, x > 0 ∧ (x^2 - 2*x + 3 = 2) ∧
    (∀ f: ℝ → ℝ, (f = (λ x, 1/x + 1) ∨ f = (λ x, log x + 1) ∨ f = (λ x, abs x + 1) ∨ f = (λ x, x^2 - 2*x + 3)) →
    (∀ x: ℝ, x > 0 → f x ≠ 2)
  )
) :=
by
sorry

end minimum_value_of_2_pos_x_l802_802877


namespace fixed_point_l802_802994

def parabola (y p : ℝ) : ℝ := (y^2) / (2 * p)

variables {a b p : ℝ} (A : ℝ × ℝ) (B : ℝ × ℝ)
variable M : ℝ × ℝ
variables M1 M2 : ℝ × ℝ

noncomputable def line (P Q : ℝ × ℝ) (x : ℝ) : ℝ := 
  (Q.2 - P.2) / (Q.1 - P.1) * (x - P.1) + P.2

-- Conditions from the problem
axiom h1 : a ≠ 0
axiom h2 : b ≠ 0
axiom h3 : b^2 ≠ 2 * p * a
axiom h4 : M1 ≠ M2
axiom on_parabola : M.2^2 = 2 * p * M.1
axiom on_parabola_M1 : M1.2^2 = 2 * p * M1.1
axiom on_parabola_M2 : M2.2^2 = 2 * p * M2.1
axiom intersection_AM1 : ∃ x, line A M x = M1.2
axiom intersection_BM2 : ∃ x, line B M x = M2.2

theorem fixed_point : ∃ C : ℝ × ℝ, (C = (a, 2 * p * a / b)) ∧ 
  ∀ M, ∃ M1 M2, intersection_AM1 ∧ intersection_BM2 ∧ 
  ((M ≠ M1) → (M ≠ M2) → line M1 M2 C.1 = C.2) :=
sorry

end fixed_point_l802_802994


namespace repeating_decimal_division_l802_802418

-- Definitions based on given conditions
def repeating_54_as_frac : ℚ := 54 / 99
def repeating_18_as_frac : ℚ := 18 / 99

-- Theorem stating the required proof
theorem repeating_decimal_division :
  (repeating_54_as_frac / repeating_18_as_frac = 3) :=
  sorry

end repeating_decimal_division_l802_802418


namespace range_of_m_l802_802157

noncomputable def f (x m : ℝ) : ℝ := -0.5 * x^2 + m * Real.log x

theorem range_of_m (m : ℝ) : (∀ x : ℝ, 1 < x → (f x m)' < 0) ↔ m ≤ 1 :=
by
  sorry

end range_of_m_l802_802157


namespace problem_statement_l802_802882

-- Define that f is an odd function and has a period of 4
variables {f : ℝ → ℝ}
axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodic_function : ∀ x : ℝ, ∀ k : ℤ, f(x + 4 * k) = f x

theorem problem_statement : 
  f(2005) + f(2006) + f(2007) = f(2) :=
by
  sorry

end problem_statement_l802_802882


namespace leap_years_in_200_years_l802_802890

theorem leap_years_in_200_years : ∀ (years_period : ℕ), years_period = 200 → (∀ k : ℕ, k = 5 → k ∣ years_period → years_period / k) = 40 :=
by
  intro years_period h1 k h2 h3
  sorry

end leap_years_in_200_years_l802_802890


namespace sum_common_divisors_l802_802544

-- Define the sum of a set of numbers
def set_sum (s : Set ℕ) : ℕ :=
  s.fold (λ x acc => x + acc) 0

-- Define the divisors of a number
def divisors (n : ℕ) : Set ℕ :=
  { d | d > 0 ∧ n % d = 0 }

-- Definitions based on the given conditions
def divisors_of_60 : Set ℕ := divisors 60
def divisors_of_18 : Set ℕ := divisors 18
def common_divisors : Set ℕ := divisors_of_60 ∩ divisors_of_18

-- Declare the theorem to be proved
theorem sum_common_divisors : set_sum common_divisors = 12 :=
  sorry

end sum_common_divisors_l802_802544


namespace ratio_of_wireless_mice_l802_802713

theorem ratio_of_wireless_mice (total_mice : ℕ) (optical_mice_ratio : ℚ) (trackball_mice : ℕ) :
  total_mice = 80 →
  optical_mice_ratio = 1/4 →
  trackball_mice = 20 →
  (80 - (optical_mice_ratio * 80 + trackball_mice)) / 80 = 1 / 2 :=
by
  intros h_total h_optical h_trackball
  rw [h_total, h_optical, h_trackball]
  sorry

end ratio_of_wireless_mice_l802_802713


namespace general_term_sum_reciprocal_l802_802997

-- Define the sequence a_n
def a : ℕ → ℝ
| 1     := 3
| (n+1) := (n * a n + a n) / (n + 1)

-- Check if the general term a_n = 3n
theorem general_term (n : ℕ) : a n = 3 * n := sorry

-- Define the sum of the first n terms S_n
def S (n : ℕ) : ℝ := (3 * n * (n + 1)) / 2

-- Define the reciprocal sequence 1/S_n
def reciprocal_S (n : ℕ) : ℕ → ℝ
| (n + 1) := 1 / S n

-- Sum of the first n terms of the reciprocal sequence
def T (n : ℕ) : ℝ :=
∑ k in finset.range n, reciprocal_S k

-- Prove that T_n = 2n / (3n + 3)
theorem sum_reciprocal (n : ℕ) : T n = 2 * n / (3 * n + 3) := sorry

end general_term_sum_reciprocal_l802_802997


namespace remaining_walking_time_is_30_l802_802728

-- Define all the given conditions
def total_distance_to_store : ℝ := 2.5
def distance_already_walked : ℝ := 1.0
def time_per_mile : ℝ := 20.0

-- Define the target remaining walking time
def remaining_distance : ℝ := total_distance_to_store - distance_already_walked
def remaining_time : ℝ := remaining_distance * time_per_mile

-- Prove the remaining walking time is 30 minutes
theorem remaining_walking_time_is_30 : remaining_time = 30 :=
by
  -- Formal proof would go here using corresponding Lean tactics
  sorry

end remaining_walking_time_is_30_l802_802728


namespace base8_to_base10_4513_l802_802428

theorem base8_to_base10_4513 : (4 * 8^3 + 5 * 8^2 + 1 * 8^1 + 3 * 8^0 = 2379) :=
by
  sorry

end base8_to_base10_4513_l802_802428


namespace increasing_interval_of_f_l802_802769

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * Real.pi / 3 - 2 * x)

theorem increasing_interval_of_f :
  ∃ a b : ℝ, f x = 3 * Real.sin (2 * Real.pi / 3 - 2 * x) ∧ (a = 7 * Real.pi / 12) ∧ (b = 13 * Real.pi / 12) ∧ ∀ x1 x2, a ≤ x1 ∧ x1 < x2 ∧ x2 ≤ b → f x1 < f x2 := 
sorry

end increasing_interval_of_f_l802_802769


namespace cubic_inequality_l802_802271

theorem cubic_inequality
  (a b c r s t : ℝ)
  (h_eq : ∀ x : ℝ, x^3 + a*x^2 + b*x + c = 0 → (x = r ∨ x = s ∨ x = t))
  (h_order : r ≥ s ∧ s ≥ t) :
  let k := a^2 - 3*b in k ≥ 0 ∧ sqrt k ≤ r - t :=
by {
  sorry
}

end cubic_inequality_l802_802271


namespace sum_common_divisors_60_18_l802_802529

def divisors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ m => n % m = 0)

noncomputable def sum (l : List ℕ) : ℕ :=
  l.foldr (· + ·) 0

theorem sum_common_divisors_60_18 :
  sum (List.filter (λ d => d ∈ divisors 18) (divisors 60)) = 12 :=
by
  sorry

end sum_common_divisors_60_18_l802_802529


namespace problem_statement_l802_802698

noncomputable def omega : ℂ := sorry -- placeholder for the specific nonreal root of x^3 = 1 

lemma omega_nonreal_root : omega ^ 3 = 1 ∧ ω.im ≠ 0 :=
sorry -- placeholder for the proof that omega is a nonreal root of x^3 = 1 

theorem problem_statement : (1 - omega + omega^2)^4 + (1 + omega - omega^2)^4 = -16 :=
begin
  -- Utilize omega_nonreal_root and algebraic properties to prove the statement.
  sorry
end

end problem_statement_l802_802698


namespace arithmetic_sequence_tenth_term_l802_802820

theorem arithmetic_sequence_tenth_term :
  ∀ (a_1 a_2 : ℚ), a_1 = 1 / 2 → a_2 = 7 / 8 → 
  let d := a_2 - a_1 in 
  let a_n (n : ℕ) := a_1 + (n - 1) * d in 
  a_n 10 = 31 / 8 :=
by 
  intros a_1 a_2 h_a1 h_a2;
  let d := a_2 - a_1;
  let a_n := λ n : ℕ, a_1 + (n - 1) * d;
  sorry

end arithmetic_sequence_tenth_term_l802_802820


namespace limit_sqrt_cos_arctan_l802_802417

noncomputable def limit_expression (x : ℝ) : ℝ :=
  sqrt (4 * real.cos (3 * x) + x * real.arctan (1 / x))

theorem limit_sqrt_cos_arctan :
  filter.tendsto limit_expression (nhds 0) (nhds 2) :=
sorry

end limit_sqrt_cos_arctan_l802_802417


namespace find_sum_on_simple_interest_l802_802356

theorem find_sum_on_simple_interest :
  let P₁ := 4000 -- Principal for C.I.
  let r₁ := 10 / 100 -- Annual interest rate for C.I. in decimal
  let t₁ := 2 -- Number of years for C.I.
  let P₂ := sorry -- Principal for S.I. (to be determined)
  let r₂ := 8 -- Annual interest rate for S.I. 
  let t₂ := 3 -- Number of years for S.I.
  let CI := P₁ * ((1 + r₁) ^ t₁ - 1) -- Compound Interest formula
  let SI := CI / 2 -- Simple Interest is half of Compound Interest
  SI = (P₂ * r₂ * t₂) / 100 -- Simple Interest formula
  P₂ = 1750 :=
by
  sorry

end find_sum_on_simple_interest_l802_802356


namespace chord_length_at_alpha_pi_over_4_l802_802127

-- Define the parametric equations of the line
def parametric_line (α t : ℝ) : ℝ × ℝ :=
  (-2 + (Real.cos α * t), Real.sin α * t)

-- Define the polar equation of the curve
def polar_curve (θ : ℝ) : ℝ :=
  2 * Real.sin θ - 2 * Real.cos θ

-- Convert polar_curve to Cartesian form
def cartesian_curve : ℝ × ℝ → Prop
  | (x, y) => x^2 + y^2 = 2 * y - 2 * x

-- Parametric equations when α = π/4
def line_at_alpha_pi_over_4 (t : ℝ) : ℝ × ℝ :=
  parametric_line (Real.pi / 4) t

-- Cartesian form of the line when α = π/4
def cartesian_line (x : ℝ) : ℝ :=
  x + 2

-- Prove the length of the chord cut from the curve by the line is 2√2
theorem chord_length_at_alpha_pi_over_4 : ∃x1 y1 x2 y2 : ℝ,
  cartesian_curve (x1, y1) ∧ cartesian_curve (x2, y2) ∧
  y1 = x1 + 2 ∧ y2 = x2 + 2 ∧ (Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 2 * Real.sqrt 2) := by
  sorry

end chord_length_at_alpha_pi_over_4_l802_802127


namespace problem_2019th_red_number_l802_802664

def nth_group_last_element (n : ℕ) : ℕ :=
  n * (2 * n - 1)

def total_elements_up_to_n_groups (n : ℕ) : ℕ :=
  n * (n - 1)

def group_start (n : ℕ) : ℕ :=
  if n = 1 then 1 else nth_group_last_element (n - 1) + 1

def nth_red_number (k : ℕ) : ℕ :=
  let n := Nat.find (λ n, total_elements_up_to_n_groups n <= k ∧ k < total_elements_up_to_n_groups (n + 1))
  let start := group_start n
  start + 2 * (k - total_elements_up_to_n_groups n) - 2

theorem problem_2019th_red_number : nth_red_number 2019 = 3993 :=
  sorry

end problem_2019th_red_number_l802_802664


namespace fraction_multiplication_l802_802024

theorem fraction_multiplication :
  (2 / (3 : ℚ)) * (4 / 7) * (5 / 9) * (11 / 13) = 440 / 2457 :=
by
  sorry

end fraction_multiplication_l802_802024


namespace largest_number_of_acute_angles_in_convex_pentagon_l802_802314

theorem largest_number_of_acute_angles_in_convex_pentagon :
  ∀ (angles : Fin 5 → ℝ), 
    (∀ i, 0 < angles i ∧ angles i < 180) →
    (∑ i, angles i = 540) →
    (∑ i, if angles i < 90 then 1 else 0) ≤ 2 :=
by
  intro angles h_angles h_sum
  sorry

end largest_number_of_acute_angles_in_convex_pentagon_l802_802314


namespace candy_problem_solution_l802_802397

theorem candy_problem_solution :
  ∃ (a : ℕ), a % 10 = 6 ∧ a % 15 = 11 ∧ 200 ≤ a ∧ a ≤ 250 ∧ (a = 206 ∨ a = 236) :=
begin
  sorry
end

end candy_problem_solution_l802_802397


namespace fraction_equals_seven_twentyfive_l802_802030

theorem fraction_equals_seven_twentyfive :
  (1722^2 - 1715^2) / (1731^2 - 1706^2) = (7 / 25) :=
by
  sorry

end fraction_equals_seven_twentyfive_l802_802030


namespace max_gold_coins_l802_802347

theorem max_gold_coins (k : ℤ) (h1 : ∃ k : ℤ, 15 * k + 3 < 120) : 
  ∃ n : ℤ, n = 15 * k + 3 ∧ n < 120 ∧ n = 108 :=
by
  sorry

end max_gold_coins_l802_802347


namespace problem_l802_802120

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := 
  (sqrt 3 / 2) * sin (ω * x) - ((sin (ω * x / 2))^2 * (1 / 2))

theorem problem (ω : ℝ) (ω_pos : ω > 0) (h_per : ∀ x, f(x) ω = f(x + π) ω) :
  (ω = 2) ∧ (∀ k : ℤ, ∀ x, (k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6) → (f(x) ω).monotone) ∧ 
  (∀ x, (0 ≤ x ∧ x ≤ π / 2) → -1 / 2 ≤ f(x) ω ∧ f(x) ω ≤ 1) :=
sorry

end problem_l802_802120


namespace probability_one_girl_l802_802595

namespace ProbabilityProblem

def P_A := 1 / 3
def P_B := 2 / 15
def P_C := 1 - (P_A + P_B)

theorem probability_one_girl :
  P_C = 8 / 15 := by
  calc
    P_C = 1 - (P_A + P_B) : rfl
    ... = 1 - (1 / 3 + 2 / 15) : by rw [P_A, P_B]
    ... = 1 - (5 / 15 + 2 / 15) : by norm_num
    ... = 1 - (7 / 15) : rfl
    ... = 8 / 15 : by norm_num

end ProbabilityProblem

end probability_one_girl_l802_802595


namespace shape_to_square_l802_802033

/-- 
Given a geometric shape plotted on a grid, prove that it can be cut into three non-disjoint parts
along the grid lines, and the parts can be rearranged (without flipping) to form a square.
-/
theorem shape_to_square (shape : geometric_shape) (parts : ℕ) (grid : grid_space)
    (cuts : cutting_positions shape parts grid) 
    (rearrangement_possible : rearrangement shape parts grid cuts)
    (no_flipping : ∀ part, not_flipping_allowed part) :
  ∃ (square : geometric_square), cuts.shape_parts = rearrange shape cuts.shape_parts square :=
sorry

end shape_to_square_l802_802033


namespace count_integers_between_sqrts_l802_802144

theorem count_integers_between_sqrts (a b : ℝ) (h1 : a = 10) (h2 : b = 100) :
  let lower_bound := Int.ceil (Real.sqrt a),
      upper_bound := Int.floor (Real.sqrt b) in
  (upper_bound - lower_bound + 1) = 7 := 
by
  rw [h1, h2]
  let lower_bound := Int.ceil (Real.sqrt 10)
  let upper_bound := Int.floor (Real.sqrt 100)
  have h_lower : lower_bound = 4 := by sorry
  have h_upper : upper_bound = 10 := by sorry
  rw [h_lower, h_upper]
  norm_num
  sorry

end count_integers_between_sqrts_l802_802144


namespace train_speed_l802_802403

theorem train_speed (length : ℝ) (time : ℝ) (h_length : length = 350) (h_time : time = 7) : 
  length / time = 50 :=
by
  rw [h_length, h_time]
  norm_num

end train_speed_l802_802403


namespace average_four_numbers_l802_802756

variable {x : ℝ}

theorem average_four_numbers (h : (15 + 25 + x + 30) / 4 = 23) : x = 22 :=
by
  sorry

end average_four_numbers_l802_802756


namespace cards_difference_product_divisible_l802_802465

theorem cards_difference_product_divisible :
  let S := {n | 1 ≤ n ∧ n ≤ 50}
  let valid_pairs := {(a, b) | a ∈ S ∧ b ∈ S ∧ |a - b| = 11 ∧ (a * b) % 5 = 0}
  valid_pairs.card = 15 := 
sorry

end cards_difference_product_divisible_l802_802465


namespace parallel_perpendicular_cosine_l802_802137

variable (x y z : ℝ)
variable (a b c : ℝ × ℝ × ℝ)
variable (k : ℝ)

noncomputable def vec_a := (x, 4, 1)
noncomputable def vec_b := (-2, y, -1)
noncomputable def vec_c := (3, -2, z)

theorem parallel_perpendicular_cosine (h_parallel : ∃ k : ℝ, vec_a = (k * -2, k * y, k * -1))
  (h_perpendicular : vec_b.1 * vec_c.1 + vec_b.2 * vec_c.2 + vec_b.3 * vec_c.3 = 0) :
  (vec_a = (2, 4, 1)) ∧ (vec_b = (-2, -4, -1)) ∧ (vec_c = (3, -2, 2)) ∧
  (real.cos_angle (vec_a.1 + vec_c.1, vec_a.2 + vec_c.2, vec_a.3 + vec_c.3)
    (vec_b.1 + vec_c.1, vec_b.2 + vec_c.2, vec_b.3 + vec_c.3) = -2 / 19) :=
by
  sorry

end parallel_perpendicular_cosine_l802_802137


namespace unique_tangent_line_l802_802516

theorem unique_tangent_line (C B : ℝ × ℝ) (dC dB : ℝ) (hC : C = (0, 0)) (hB : B = (-4, -3)) (hdC : dC = 1) (hdB : dB = 6) : 
  ∃! (l : set (ℝ × ℝ)), ∀ (P : ℝ × ℝ), l P ↔ dist P C = dC ∧ dist P B = dB := sorry

end unique_tangent_line_l802_802516


namespace change_received_l802_802711

-- Define the given conditions
def num_apples : ℕ := 5
def cost_per_apple : ℝ := 0.75
def amount_paid : ℝ := 10.00

-- Prove the change is equal to $6.25
theorem change_received :
  amount_paid - (num_apples * cost_per_apple) = 6.25 :=
by
  sorry

end change_received_l802_802711


namespace card_choice_count_l802_802481

theorem card_choice_count :
  ∃ (pairs : Finset (ℕ × ℕ)),
    (∀ (a b : ℕ), (a, b) ∈ pairs → 1 ≤ a ∧ a ≤ 50 ∧ 1 ≤ b ∧ b ≤ 50) ∧
    (∀ (a b : ℕ), (a, b) ∈ pairs → |a - b| = 11) ∧
    (∀ (a b : ℕ), (a, b) ∈ pairs → (a * b) % 5 = 0) ∧
    card pairs = 15 := 
sorry

end card_choice_count_l802_802481


namespace triangle_ABC_l802_802168

-- Given conditions in a)
variables {A B C : ℝ} {a b c : ℝ} {cosB cosC : ℝ}
hypothesis h1 : (2 * a - c) * cosB = b * cosC

-- Problem statement
theorem triangle_ABC (h1 : (2 * a - c) * cosB = b * cosC) :
  (B = π / 3) ∧ (∀ (m n : ℝ × ℝ), m = (sin A, cos (2 * A)) ∧ n = (6, 1) 
                → (m.1 * n.1 + m.2 * n.2) ≤ 5) :=
by 
  sorry

end triangle_ABC_l802_802168


namespace unique_trivial_solution_l802_802840

variable {n : ℕ}
variable (a : ℕ → ℕ → ℝ)
variable (x : ℕ → ℝ)

def system_of_equations (n : ℕ) (a : ℕ → ℕ → ℝ) (x : ℕ → ℝ) := 
  ∀ i : ℕ, ∑ j in Finset.range n, a i j * x j = 0

def conditions (n : ℕ) (a : ℕ → ℕ → ℝ) :=
  (∀ i j : ℕ, 0 < a i j) ∧
  (∀ i : ℕ, ∑ j in Finset.range n, a i j = 1) ∧
  (∀ j : ℕ, ∑ i in Finset.range n, a i j = 1) ∧
  (∀ k : ℕ, 2 < n → a k k = 1 / 2)

theorem unique_trivial_solution (n : ℕ) (a : ℕ → ℕ → ℝ) (x : ℕ → ℝ)
  (h1 : 2 < n) (h2 : system_of_equations n a x) (h3 : conditions n a) :
  ∀ i : ℕ, x i = 0 :=
sorry

end unique_trivial_solution_l802_802840


namespace university_application_methods_l802_802848

theorem university_application_methods :
  let n := 6
  let k := 3
  let conflicting_pairs := 2
  let valid_universities := n - conflicting_pairs
  let choose_case_1 := Nat.choose valid_universities k
  let choose_case_2 := Nat.choose conflicting_pairs 1 * Nat.choose valid_universities (k - 1)
  in choose_case_1 + choose_case_2 = 16 := 
by
  let n := 6
  let k := 3
  let conflicting_pairs := 2
  let valid_universities := n - conflicting_pairs
  let choose_case_1 := Nat.choose valid_universities k
  let choose_case_2 := Nat.choose conflicting_pairs 1 * Nat.choose valid_universities (k - 1)
  have h1 : choose_case_1 = Nat.choose 4 3 := rfl
  have h2 : choose_case_1 = 4 := by 
    rw [h1, Nat.choose_succ_self n.pred k]

  have h3 : choose_case_2 = Nat.choose 2 1 * Nat.choose 4 2 := rfl
  have h4 : choose_case_2 = 2 * 6 := by 
    rw [h3, Nat.choose_succ_self (conflicting_pairs.pred) (1.pred), Nat.choose 4 2]
    norm_num

  show choose_case_1 + choose_case_2 = 16 := 
    rw [h2, h4]
    norm_num
  sorry

end university_application_methods_l802_802848


namespace sandwiches_provided_now_l802_802796

-- Define the initial number of sandwich kinds
def initialSandwichKinds : ℕ := 23

-- Define the number of sold out sandwich kinds
def soldOutSandwichKinds : ℕ := 14

-- Define the proof that the actual number of sandwich kinds provided now
theorem sandwiches_provided_now : initialSandwichKinds - soldOutSandwichKinds = 9 :=
by
  -- The proof goes here
  sorry

end sandwiches_provided_now_l802_802796


namespace minimum_value_correct_l802_802968

theorem minimum_value_correct (a b c : ℝ) (h : a^2 - 4 * real.log a - b = 0) :
  ∃ (m : ℝ), m = (a - c)^2 + (b + 2 * c)^2 ∧ m = 9 / 5 :=
sorry

end minimum_value_correct_l802_802968


namespace sum_of_numbers_in_third_column_is_96_l802_802236

theorem sum_of_numbers_in_third_column_is_96 :
  ∃ (a : ℕ), (136 = a + 16 * a) ∧ (272 = 2 * a + 32 * a) ∧ (12 * a = 96) :=
by
  let a := 8
  have h1 : 136 = a + 16 * a := by sorry  -- Proof here that 136 = 8 + 16 * 8
  have h2 : 272 = 2 * a + 32 * a := by sorry  -- Proof here that 272 = 2 * 8 + 32 * 8
  have h3 : 12 * a = 96 := by sorry  -- Proof here that 12 * 8 = 96
  existsi a
  exact ⟨h1, h2, h3⟩

end sum_of_numbers_in_third_column_is_96_l802_802236


namespace largest_result_is_0_point_1_l802_802341

theorem largest_result_is_0_point_1 : 
  max (5 * Real.sqrt 2 - 7) (max (7 - 5 * Real.sqrt 2) (max (|1 - 1|) 0.1)) = 0.1 := 
by
  -- We will prove this by comparing each value to 0.1
  sorry

end largest_result_is_0_point_1_l802_802341


namespace positive_difference_is_9107_03_l802_802421

noncomputable def Cedric_balance : ℝ :=
  15000 * (1 + 0.06) ^ 20

noncomputable def Daniel_balance : ℝ :=
  15000 * (1 + 20 * 0.08)

noncomputable def Elaine_balance : ℝ :=
  15000 * (1 + 0.055 / 2) ^ 40

-- Positive difference between highest and lowest balances.
noncomputable def positive_difference : ℝ :=
  let highest := max Cedric_balance (max Daniel_balance Elaine_balance)
  let lowest := min Cedric_balance (min Daniel_balance Elaine_balance)
  highest - lowest

theorem positive_difference_is_9107_03 :
  positive_difference = 9107.03 := by
  sorry

end positive_difference_is_9107_03_l802_802421


namespace card_pairs_count_l802_802472

theorem card_pairs_count :
  let cards := {n : ℕ | 1 ≤ n ∧ n ≤ 50}
  let valid_pairs := {(a, b) ∈ cards × cards | 
    a ≠ b ∧ (a - b = 11 ∨ b - a = 11) ∧ (a * b) % 5 = 0}
  in valid_pairs.count = 15 :=
by
  sorry

end card_pairs_count_l802_802472


namespace width_of_each_road_l802_802390

noncomputable def width_of_road (cost : ℝ) (cost_per_sqm : ℝ) (length : ℝ) (breadth : ℝ) : ℝ :=
  cost / (cost_per_sqm * (length + breadth))

theorem width_of_each_road (cost : ℝ) (cost_per_sqm : ℝ) (length : ℝ) (breadth : ℝ) :
  width_of_road cost cost_per_sqm length breadth = 14.88 :=
by
  -- Assume given values
  let c := 5625
  let cpsqm := 3
  let l := 80
  let b := 60
  -- Calculate the width
  have w := width_of_road c cpsqm l b
  -- Prove that the resulting width is approximately 14.88
  have : w = 14.88 := by
    simp
  sorry

end width_of_each_road_l802_802390


namespace dave_paid_4_more_than_doug_l802_802045

theorem dave_paid_4_more_than_doug :
  let slices := 8
  let plain_cost := 8
  let anchovy_additional_cost := 2
  let total_cost := plain_cost + anchovy_additional_cost
  let cost_per_slice := total_cost / slices
  let dave_slices := 5
  let doug_slices := slices - dave_slices
  -- Calculate payments
  let dave_payment := dave_slices * cost_per_slice
  let doug_payment := doug_slices * cost_per_slice
  dave_payment - doug_payment = 4 :=
by
  sorry

end dave_paid_4_more_than_doug_l802_802045


namespace pentagon_perimeter_l802_802816

def distance (p q : ℝ × ℝ) : ℝ := 
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem pentagon_perimeter 
  (A B C D E : ℝ × ℝ)
  (AB : distance A B = 1)
  (BC : distance B C = real.sqrt 3)
  (CD : distance C D = real.sqrt 2)
  (DE : distance D E = real.sqrt 2) :
  let AC := distance A C,
      AD := distance A D,
      AE := distance A E in
  AC = 2 → AD = real.sqrt 6 → AE = 2 * real.sqrt 2 →
  distance A B + distance B C + distance C D + distance D E + distance E A = 1 + real.sqrt 3 + 3 * real.sqrt 2 :=
sorry

end pentagon_perimeter_l802_802816


namespace min_distance_curve_to_line_l802_802662

noncomputable def minimum_distance (x : ℝ) (h : x > 0) : ℝ :=
  let y := x + 4/x in
  (|x + y| / real.sqrt 2)

theorem min_distance_curve_to_line : ∃ (x : ℝ) (h : x > 0), minimum_distance x h = 4 := by
  sorry

end min_distance_curve_to_line_l802_802662


namespace positive_reducible_implies_self_positive_reducible_l802_802029

noncomputable def is_positive_reducible (p : Polynomial ℝ) : Prop :=
∃ (g h : Polynomial ℝ), 
  (¬ is_constant g ∧ ¬ is_constant h) ∧ 
  (∀ x, 0 < g.coeff x) ∧ 
  (∀ x, 0 < h.coeff x) ∧ 
  p = g * h

theorem positive_reducible_implies_self_positive_reducible 
  (f : Polynomial ℝ) (n : ℕ) (hn : 0 < n) (f0_ne_0 : f.coeff 0 ≠ 0) (h : is_positive_reducible (f.comp (λ x, x ^ n))) : is_positive_reducible f := 
sorry

end positive_reducible_implies_self_positive_reducible_l802_802029


namespace sequence_general_term_increasing_sequence_lambda_bound_l802_802610

theorem sequence_general_term (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 0 < n → 2 * S n = (n + 1) * a n) :
  ∀ n : ℕ, 0 < n → a n = n :=
by
  -- proof to be inserted here
  sorry

theorem increasing_sequence_lambda_bound (λ : ℝ) :
  (∀ n : ℕ, 0 < n → 3^n - λ * n^2 < 3^(n+1) - λ * (n+1)^2) →
  λ < 2 :=
by
  -- proof to be inserted here
  sorry

end sequence_general_term_increasing_sequence_lambda_bound_l802_802610


namespace calculation_a_squared_plus_b_squared_calculation_a_minus_b_squared_l802_802949

theorem calculation_a_squared_plus_b_squared
  (a b : ℝ)
  (h1 : a + b = 2)
  (h2 : a * b = -1) :
  a^2 + b^2 = 6 := by
  sorry

theorem calculation_a_minus_b_squared
  (a b : ℝ)
  (h1 : a + b = 2)
  (h2 : a * b = -1) :
  (a - b)^2 = 8 := by
  sorry

end calculation_a_squared_plus_b_squared_calculation_a_minus_b_squared_l802_802949


namespace cyclic_quadrilateral_angle_D_l802_802174

theorem cyclic_quadrilateral_angle_D (A B C D : ℝ) (h₁ : A + B + C + D = 360) (h₂ : ∃ x, A = 3 * x ∧ B = 4 * x ∧ C = 6 * x) :
  D = 100 :=
by
  sorry

end cyclic_quadrilateral_angle_D_l802_802174


namespace card_pairs_with_conditions_l802_802451

theorem card_pairs_with_conditions : 
  let cards := Finset.range 51 in
  let pairs := (cards.product cards).filter (λ p, (p.1 - p.2).natAbs = 11 ∧ (p.1 * p.2) % 5 = 0) in
  pairs.card / 2 = 15 :=
by
  sorry

end card_pairs_with_conditions_l802_802451


namespace ak_lt_1_lt_ak1_implies_k_2018_l802_802965

noncomputable def a : ℕ → ℝ
| 0       := 1 / 2
| (n + 1) := a n + a n ^ 2 / 2018

theorem ak_lt_1_lt_ak1_implies_k_2018 (k : ℕ) (h : a k < 1 ∧ 1 < a (k + 1)) : k = 2018 :=
sorry

end ak_lt_1_lt_ak1_implies_k_2018_l802_802965


namespace square_perimeter_l802_802003

theorem square_perimeter (s : ℝ) (area : ℝ) : area = 225 → s^2 = area → 4 * s = 60 :=
by
  intro h1 h2
  rw [h1, h2]
  calc
    s = 15 : by sorry
    4 * 15 = 60 : by norm_num

end square_perimeter_l802_802003


namespace log_x_64_eq_two_l802_802154

theorem log_x_64_eq_two (x : ℝ) (hx : log 8 (4 * x) = 2) : log x 64 = 2 :=
sorry

end log_x_64_eq_two_l802_802154


namespace number_of_digits_in_first_3003_even_integers_l802_802824

theorem number_of_digits_in_first_3003_even_integers : 
  let num_even_integers_1_digit := 4,
      num_even_integers_2_digits := (98 - 10) / 2 + 1,
      num_even_integers_3_digits := (998 - 100) / 2 + 1,
      num_even_integers_4_digits := (6006 - 1000) / 2 + 1,
      total_digits := 
        num_even_integers_1_digit * 1 +
        num_even_integers_2_digits * 2 +
        num_even_integers_3_digits * 3 +
        num_even_integers_4_digits * 4
  in total_digits = 11460 :=
sorry

end number_of_digits_in_first_3003_even_integers_l802_802824


namespace number_of_integers_between_sqrt10_and_sqrt100_l802_802146

theorem number_of_integers_between_sqrt10_and_sqrt100 :
  (∃ n : ℕ, n = 7) :=
sorry

end number_of_integers_between_sqrt10_and_sqrt100_l802_802146


namespace percentage_failed_in_hindi_l802_802655

theorem percentage_failed_in_hindi (P_E : ℝ) (P_H_and_E : ℝ) (P_P : ℝ) (H : ℝ) : 
  P_E = 0.5 ∧ P_H_and_E = 0.25 ∧ P_P = 0.5 → H = 0.25 :=
by
  sorry

end percentage_failed_in_hindi_l802_802655


namespace volume_of_circumscribed_sphere_l802_802980

theorem volume_of_circumscribed_sphere :
  ∀ (height AB AC : ℝ) (BAC_angle : ℝ),
    height = 4 → AB = 2 → AC = 2 → BAC_angle = 90 →
    (4/3) * Real.pi * (Real.sqrt 6)^3 = 8 * Real.sqrt 6 * Real.pi :=
begin
  intros height AB AC BAC_angle h_eq ab_eq ac_eq angle_eq,
  sorry
end

end volume_of_circumscribed_sphere_l802_802980


namespace total_wheels_l802_802416

def regular_bikes := 7
def children_bikes := 11
def tandem_bikes_4_wheels := 5
def tandem_bikes_6_wheels := 3
def unicycles := 4
def tricycles := 6
def bikes_with_training_wheels := 8

def wheels_regular := 2
def wheels_children := 4
def wheels_tandem_4 := 4
def wheels_tandem_6 := 6
def wheel_unicycle := 1
def wheels_tricycle := 3
def wheels_training := 4

theorem total_wheels : 
  (regular_bikes * wheels_regular) +
  (children_bikes * wheels_children) + 
  (tandem_bikes_4_wheels * wheels_tandem_4) + 
  (tandem_bikes_6_wheels * wheels_tandem_6) + 
  (unicycles * wheel_unicycle) + 
  (tricycles * wheels_tricycle) + 
  (bikes_with_training_wheels * wheels_training) 
  = 150 := by
  sorry

end total_wheels_l802_802416


namespace problems_completed_l802_802619

theorem problems_completed (p t : ℕ) (hp : p > 10) (eqn : p * t = (2 * p - 2) * (t - 1)) :
  p * t = 48 := 
sorry

end problems_completed_l802_802619


namespace percent_of_y_equal_to_30_percent_of_60_percent_l802_802322

variable (y : ℝ)

theorem percent_of_y_equal_to_30_percent_of_60_percent (hy : y ≠ 0) :
  ((0.18 * y) / y) * 100 = 18 := by
  have hneq : y ≠ 0 := hy
  field_simp [hneq]
  norm_num
  sorry

end percent_of_y_equal_to_30_percent_of_60_percent_l802_802322


namespace surface_area_of_circumscribing_sphere_l802_802097

theorem surface_area_of_circumscribing_sphere (r : ℝ) (h : r = sqrt (6^2 + 6^2 / 4)) : 4 * π * r^2 = 84 * π :=
by
  sorry

end surface_area_of_circumscribing_sphere_l802_802097


namespace exists_infinite_irregular_set_l802_802811

def is_irregular (A : Set ℤ) :=
  ∀ ⦃x y : ℤ⦄, x ∈ A → y ∈ A → x ≠ y → ∀ ⦃k : ℤ⦄, x + k * (y - x) ≠ x ∧ x + k * (y - x) ≠ y

theorem exists_infinite_irregular_set : ∃ A : Set ℤ, Set.Infinite A ∧ is_irregular A :=
sorry

end exists_infinite_irregular_set_l802_802811


namespace candy_problem_solution_l802_802395

theorem candy_problem_solution :
  ∃ (a : ℕ), a % 10 = 6 ∧ a % 15 = 11 ∧ 200 ≤ a ∧ a ≤ 250 ∧ (a = 206 ∨ a = 236) :=
begin
  sorry
end

end candy_problem_solution_l802_802395


namespace card_pairs_count_l802_802474

theorem card_pairs_count :
  let cards := {n : ℕ | 1 ≤ n ∧ n ≤ 50}
  let valid_pairs := {(a, b) ∈ cards × cards | 
    a ≠ b ∧ (a - b = 11 ∨ b - a = 11) ∧ (a * b) % 5 = 0}
  in valid_pairs.count = 15 :=
by
  sorry

end card_pairs_count_l802_802474


namespace inverse_100_mod_101_l802_802052

theorem inverse_100_mod_101 :
  ∃ x, (x : ℤ) ≡ 100 [MOD 101] ∧ 100 * x ≡ 1 [MOD 101] :=
by {
  use 100,
  split,
  { exact rfl },
  { norm_num }
}

end inverse_100_mod_101_l802_802052


namespace percent_of_percent_l802_802324

theorem percent_of_percent (y : ℝ) : 0.3 * (0.6 * y) = 0.18 * y :=
sorry

end percent_of_percent_l802_802324


namespace cube_dot_path_length_l802_802378

/--
A cube with edge length 2 cm, having a dot at the center of its top face,
is rolled along one of its edges without slipping or lifting.
At least two vertices are always in contact with the surface,
and the rolling continues until the cube completes a full cycle and the dot
is at the top face again.
Prove that the length of the path followed by the dot throughout this motion 
equals \(2\sqrt{2}\pi\).
-/
theorem cube_dot_path_length 
(edge_length : ℝ)
(dot_position : ℝ × ℝ)
(h1 : edge_length = 2)
(h2 : dot_position = (1, 1))
: ∃ k : ℝ, length_of_dot_path dot_position edge_length = k * Real.pi ∧ k = 2 * Real.sqrt 2 := 
sorry

end cube_dot_path_length_l802_802378


namespace time_to_pass_first_train_l802_802866

noncomputable def speed_kmph_to_mps (v_kmph : ℕ) : ℝ :=
  v_kmph * 1000 / 3600

theorem time_to_pass_first_train (v1_kmph v2_kmph : ℕ) (t1 : ℕ) (L2 : ℝ) : 
  v1_kmph = 60 ∧ v2_kmph = 80 ∧ t1 = 7 →
  let v1_mps := speed_kmph_to_mps v1_kmph in
  let v2_mps := speed_kmph_to_mps v2_kmph in
  let length_train1 := v1_mps * t1 in
  let relative_speed := v1_mps + v2_mps in
  (length_train1 = 116.69) →
  relative_speed = 38.89 →
  let total_length := length_train1 + L2 in
  total_length / relative_speed = (116.69 + L2) / 38.89 :=
sorry

end time_to_pass_first_train_l802_802866


namespace correct_conclusions_l802_802876

-- 1. Define the conditions for conclusion ①.
def even_function_on_ℝ (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(-x) = f(x)

def satisfies_fx_plus_1_eq_neg_fx (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x + 1) = -f(x)

-- 2. Define the conditions for conclusion ②.
def normal_distribution (ξ : ℝ → ℝ) (μ : ℝ) (σ : ℝ) : Prop :=
  ∀ x, ξ x = (1 / (σ * Math.sqrt(2 * Math.pi))) * Math.exp(-(x - μ)^2 / (2 * σ^2))

def P_ξ_gt_17 (P : ℝ → ℝ) : Prop :=
  P 17 = 0.35

-- 3. Define the conditions for conclusion ③.
def even_and_increasing_on_neg_ℝ (f : ℝ → ℝ) : Prop :=
  even_function_on_ℝ f ∧ ∀ x y : ℝ, x < y → x ≤ 0 → y ≤ 0 → f(x) ≤ f(y)

def log_conditions (f : ℝ → ℝ) : Prop :=
  let a := f(Math.log(1 / 3)) in
  let b := f(Math.log 4 3) in
  let c := f(0.4^(-1.2)) in
  c < a ∧ a < b

-- Main theorem statement
theorem correct_conclusions
  (f : ℝ → ℝ) (ξ : ℝ → ℝ) (P : ℝ → ℝ) (σ : ℝ)
  (h1 : even_function_on_ℝ f)
  (h2 : satisfies_fx_plus_1_eq_neg_fx f)
  (h3 : normal_distribution ξ 16 σ)
  (h4 : P_ξ_gt_17 P)
  (h5 : even_and_increasing_on_neg_ℝ f)
  (h6 : log_conditions f) :
  (h1 ∧ h2) ∧ (h3 ∧ h4) ∧ h6 :=
by
  split; sorry

end correct_conclusions_l802_802876


namespace george_total_coins_l802_802557

-- We'll state the problem as proving the total number of coins George has.
variable (num_nickels num_dimes : ℕ)
variable (value_of_coins : ℝ := 2.60)
variable (value_of_nickels : ℝ := 0.05 * num_nickels)
variable (value_of_dimes : ℝ := 0.10 * num_dimes)

theorem george_total_coins :
  num_nickels = 4 → 
  value_of_coins = value_of_nickels + value_of_dimes → 
  num_nickels + num_dimes = 28 := 
by
  sorry

end george_total_coins_l802_802557
