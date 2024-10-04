import Mathlib

namespace puppies_in_each_cage_l97_97383

theorem puppies_in_each_cage (initial_puppies sold_puppies cages : ℕ)
  (h_initial : initial_puppies = 18)
  (h_sold : sold_puppies = 3)
  (h_cages : cages = 3) :
  (initial_puppies - sold_puppies) / cages = 5 :=
by
  sorry

end puppies_in_each_cage_l97_97383


namespace vector_dot_product_l97_97141

-- Definitions of the vectors
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-1, 2)

-- Definition of the dot product for 2D vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Main statement to prove
theorem vector_dot_product :
  dot_product (a.1 + b.1, a.2 + b.2) (a.1 - b.1, a.2 - b.2) = 0 :=
by
  sorry

end vector_dot_product_l97_97141


namespace fifth_digit_is_one_l97_97086

def self_descriptive_seven_digit_number (A B C D E F G : ℕ) : Prop :=
  A = 3 ∧ B = 2 ∧ C = 2 ∧ D = 1 ∧ E = 1 ∧ [A, B, C, D, E, F, G].count 0 = A ∧
  [A, B, C, D, E, F, G].count 1 = B ∧ [A, B, C, D, E, F, G].count 2 = C ∧
  [A, B, C, D, E, F, G].count 3 = D ∧ [A, B, C, D, E, F, G].count 4 = E

theorem fifth_digit_is_one
  (A B C D E F G : ℕ) (h : self_descriptive_seven_digit_number A B C D E F G) : E = 1 := by
  sorry

end fifth_digit_is_one_l97_97086


namespace kuziya_probability_l97_97063

-- Definitions for our problem
def is_at_distance (A : ℝ) (h : ℝ) (n : ℕ) : ℝ → Prop :=
  λ x, ∃ k m : ℕ, k + m = n ∧ k - m = 4 ∧ x = A + (k - m) * h

-- Probability calculation, restricting to range
def prob_at_distance (A : ℝ) (h : ℝ) : ℚ :=
  ∑ n in (finset.range 7).image (λ x, x + 3), ℙ (n % 2 = 0 ∧ is_at_distance A h n (A + 4 * h))

-- Main theorem statement
theorem kuziya_probability (A h : ℝ) :
  prob_at_distance A h = 47 / 224 :=
sorry

end kuziya_probability_l97_97063


namespace simplify_expression_l97_97181

theorem simplify_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (a^3 - b^3) / (a * b^2) - (ab^2 - b^3) / (ab^2 - a^3) = (a^3 - ab^2 + b^4) / (a * b^2) :=
sorry

end simplify_expression_l97_97181


namespace total_boys_in_class_l97_97915

theorem total_boys_in_class (n : ℕ) (h_circle : ∀ i, 1 ≤ i ∧ i ≤ n -> i ≤ n) 
  (h_opposite : ∀ j k, j = 7 ∧ k = 27 ∧ j < k -> (k - j = n / 2)) : 
  n = 40 :=
sorry

end total_boys_in_class_l97_97915


namespace verify_drawn_numbers_when_x_is_24_possible_values_of_x_l97_97014

-- Population size and group division
def population_size := 1000
def number_of_groups := 10
def group_size := population_size / number_of_groups

-- Systematic sampling function
def systematic_sample (x : ℕ) (k : ℕ) : ℕ :=
  (x + 33 * k) % 1000

-- Prove the drawn 10 numbers when x = 24
theorem verify_drawn_numbers_when_x_is_24 :
  (∃ drawn_numbers, drawn_numbers = [24, 157, 290, 323, 456, 589, 622, 755, 888, 921]) :=
  sorry

-- Prove possible values of x given last two digits equal to 87
theorem possible_values_of_x (k : ℕ) (h : k < number_of_groups) :
  (∃ x_values, x_values = [87, 54, 21, 88, 55, 22, 89, 56, 23, 90]) :=
  sorry

end verify_drawn_numbers_when_x_is_24_possible_values_of_x_l97_97014


namespace each_charity_gets_45_dollars_l97_97840

def dozens : ℤ := 6
def cookies_per_dozen : ℤ := 12
def total_cookies : ℤ := dozens * cookies_per_dozen
def selling_price_per_cookie : ℚ := 1.5
def cost_per_cookie : ℚ := 0.25
def profit_per_cookie : ℚ := selling_price_per_cookie - cost_per_cookie
def total_profit : ℚ := profit_per_cookie * total_cookies
def charities : ℤ := 2
def amount_per_charity : ℚ := total_profit / charities

theorem each_charity_gets_45_dollars : amount_per_charity = 45 := 
by
  sorry

end each_charity_gets_45_dollars_l97_97840


namespace ways_to_place_people_into_groups_l97_97018

theorem ways_to_place_people_into_groups :
  let men := 4
  let women := 5
  ∃ (groups : Nat), groups = 2 ∧
  ∀ (g : Nat → (Fin 3 → (Bool → Nat → Nat))),
    (∀ i, i < group_counts → ∃ m w, g i m w < people ∧ g i m (1 - w) < people ∧ m + 1 - w + (1 - m) + w = 3) →
    let groups : List (List (Fin 2)) := [
      [(Fin.mk 1 dec_trivial, Fin.mk 2 dec_trivial)],
      [(Fin.mk 1 dec_trivial, Fin.mk 2 dec_trivial)]
    ] in
    g.mk 1 dec_trivial * g.mk 2 dec_trivial = 360 :=
sorry

end ways_to_place_people_into_groups_l97_97018


namespace negation_proposition_l97_97864

variable {f : ℝ → ℝ}

theorem negation_proposition : ¬ (∀ x : ℝ, f x > 0) ↔ ∃ x : ℝ, f x ≤ 0 := by
  sorry

end negation_proposition_l97_97864


namespace root_relationship_l97_97888

theorem root_relationship (a x₁ x₂ : ℝ) 
  (h_eqn : x₁^2 - (2*a + 1)*x₁ + a^2 + 2 = 0)
  (h_roots : x₂ = 2*x₁)
  (h_vieta1 : x₁ + x₂ = 2*a + 1)
  (h_vieta2 : x₁ * x₂ = a^2 + 2) : 
  a = 4 := 
sorry

end root_relationship_l97_97888


namespace brick_height_l97_97088

theorem brick_height (h : ℝ) : 
    let wall_length := 900
    let wall_width := 600
    let wall_height := 22.5
    let num_bricks := 7200
    let brick_length := 25
    let brick_width := 11.25
    wall_length * wall_width * wall_height = num_bricks * (brick_length * brick_width * h) -> 
    h = 67.5 := 
by
  intros
  sorry

end brick_height_l97_97088


namespace conditional_prob_B_given_A_l97_97122

-- Define the events for students running specific legs
def A_runs_first_leg : Prop := sorry -- Set the actual probability space and conditions
def B_runs_second_leg : Prop := sorry -- Set the actual probability space and conditions

-- Define the condition that A runs the first leg
axiom A_runs_first : A_runs_first_leg

-- Define the total number of students
def total_students := 4

-- Define the number of students available for the second leg after A runs the first leg
def available_students := total_students - 1

-- Define the probability that B runs the second leg given A runs the first leg
def P_B_given_A : ℝ := 1 / (available_students : ℝ)

-- Prove that P(B|A) = 1/3
theorem conditional_prob_B_given_A : P(B_runs_second_leg | A_runs_first_leg) = 1 / 3 := by
  sorry

end conditional_prob_B_given_A_l97_97122


namespace common_factor_of_polynomial_l97_97592

noncomputable def polynomial_common_factor (m : ℤ) : ℤ :=
  let polynomial := 2 * m^3 - 8 * m
  let common_factor := 2 * m
  common_factor  -- We're stating that the common factor is 2 * m

-- The theorem to verify that the common factor of each term in the polynomial is 2m
theorem common_factor_of_polynomial (m : ℤ) : 
  polynomial_common_factor m = 2 * m := by
  sorry

end common_factor_of_polynomial_l97_97592


namespace min_squares_to_cover_5x5_l97_97468

theorem min_squares_to_cover_5x5 : 
  (∀ (cover : ℕ → ℕ), (cover 1 + cover 2 + cover 3 + cover 4) * (1^2 + 2^2 + 3^2 + 4^2) = 25 → 
  cover 1 + cover 2 + cover 3 + cover 4 = 10) :=
sorry

end min_squares_to_cover_5x5_l97_97468


namespace systematic_sampling_method_l97_97783

theorem systematic_sampling_method (k : ℕ) (n : ℕ) 
  (invoice_stubs : ℕ → ℕ) : 
  (k > 0) → 
  (n > 0) → 
  (invoice_stubs 15 = k) → 
  (∀ i : ℕ, invoice_stubs (15 + i * 50) = k + i * 50)
  → (sampling_method = "systematic") :=
by 
  intro h1 h2 h3 h4
  sorry

end systematic_sampling_method_l97_97783


namespace toy_cars_ratio_proof_l97_97992

theorem toy_cars_ratio_proof (toys_original : ℕ) (toys_bought_last_month : ℕ) (toys_total : ℕ) :
  toys_original = 25 ∧ toys_bought_last_month = 5 ∧ toys_total = 40 →
  (toys_total - toys_original - toys_bought_last_month) / toys_bought_last_month = 2 :=
by
  sorry

end toy_cars_ratio_proof_l97_97992


namespace stratified_sampling_grade10_l97_97386

theorem stratified_sampling_grade10
  (total_students : ℕ)
  (grade10_students : ℕ)
  (grade11_students : ℕ)
  (grade12_students : ℕ)
  (sample_size : ℕ)
  (h1 : total_students = 700)
  (h2 : grade10_students = 300)
  (h3 : grade11_students = 200)
  (h4 : grade12_students = 200)
  (h5 : sample_size = 35)
  : (grade10_students * sample_size / total_students) = 15 := 
sorry

end stratified_sampling_grade10_l97_97386


namespace probability_one_painted_face_l97_97227

def cube : ℕ := 5
def total_unit_cubes : ℕ := 125
def painted_faces_share_edge : Prop := true
def unit_cubes_with_one_painted_face : ℕ := 41

theorem probability_one_painted_face :
  ∃ (cube : ℕ) (total_unit_cubes : ℕ) (painted_faces_share_edge : Prop) (unit_cubes_with_one_painted_face : ℕ),
  cube = 5 ∧ total_unit_cubes = 125 ∧ painted_faces_share_edge ∧ unit_cubes_with_one_painted_face = 41 →
  (unit_cubes_with_one_painted_face : ℚ) / (total_unit_cubes : ℚ) = 41 / 125 :=
by 
  sorry

end probability_one_painted_face_l97_97227


namespace group_count_4_men_5_women_l97_97025

theorem group_count_4_men_5_women : 
  let men := 4
  let women := 5
  let groups := List.replicate 3 (3, true)
  ∃ (m_w_combinations : List (ℕ × ℕ)),
    m_w_combinations = [(1, 2), (2, 1)] ∧
    ((men.choose m_w_combinations.head.fst * women.choose m_w_combinations.head.snd) * (men - m_w_combinations.head.fst).choose m_w_combinations.tail.head.fst * (women - m_w_combinations.head.snd).choose m_w_combinations.tail.head.snd) = 360 :=
by
  sorry

end group_count_4_men_5_women_l97_97025


namespace infinite_series_sum_l97_97109

theorem infinite_series_sum : 
  ∑' k : ℕ, (8 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1))) = 2 :=
by 
  sorry

end infinite_series_sum_l97_97109


namespace group_division_l97_97023

theorem group_division (men women : ℕ) : 
  men = 4 → women = 5 →
  (∃ g1 g2 g3 : set (fin $ men + women), 
    g1.card = 3 ∧ g2.card = 3 ∧ g3.card = 3 ∧ 
    (∀ g, g ∈ [g1, g2, g3] → (∃ m w : ℕ, 1 ≤ m ∧ 1 ≤ w ∧ 
      finset.card (finset.filter (λ x, x < men) g) = m ∧ 
      finset.card (finset.filter (λ x, x ≥ men) g) = w)) 
    ∧ finset.disjoint g1 g2 ∧ finset.disjoint g2 g3 ∧ finset.disjoint g3 g1 
    ∧ g1 ∪ g2 ∪ g3 = finset.univ (fin $ men + women)) → 
  finset.card (finset.powerset' 3 (finset.univ (fin $ men + women))) / 2 = 180 :=
begin
  intros hmen hwomen,
  sorry
end

end group_division_l97_97023


namespace floor_neg_seven_quarter_l97_97270

theorem floor_neg_seven_quarter : 
  ∃ x : ℤ, -2 ≤ (-7 / 4 : ℚ) ∧ (-7 / 4 : ℚ) < -1 ∧ x = -2 := by
  have h1 : (-7 / 4 : ℚ) = -1.75 := by norm_num
  have h2 : -2 ≤ -1.75 := by norm_num
  have h3 : -1.75 < -1 := by norm_num
  use -2
  exact ⟨h2, h3, rfl⟩
  sorry

end floor_neg_seven_quarter_l97_97270


namespace quadratic_solutions_l97_97357

theorem quadratic_solutions : ∀ x : ℝ, x^2 - 25 = 0 → (x = 5 ∨ x = -5) :=
by
  sorry

end quadratic_solutions_l97_97357


namespace a_4_value_l97_97553

-- Defining the polynomial (2x - 3)^6
def polynomial_expansion (x : ℝ) := (2 * x - 3) ^ 6

-- Given conditions polynomial expansion in terms of (x - 1)
def polynomial_coefficients (x : ℝ) (a : Fin 7 → ℝ) : ℝ :=
  a 0 + a 1 * (x - 1) + a 2 * (x - 1) ^ 2 + a 3 * (x - 1) ^ 3 + a 4 * (x - 1) ^ 4 +
  a 5 * (x - 1) ^ 5 + a 6 * (x - 1) ^ 6

-- The proof problem asking to show a_4 = 240
theorem a_4_value : 
  ∀ a : Fin 7 → ℝ, (∀ x : ℝ, polynomial_expansion x = polynomial_coefficients x a) → a 4 = 240 := by 
  sorry

end a_4_value_l97_97553


namespace floor_of_neg_seven_fourths_l97_97277

theorem floor_of_neg_seven_fourths : (Int.floor (-7/4 : ℚ)) = -2 :=
by
  sorry

end floor_of_neg_seven_fourths_l97_97277


namespace crayons_allocation_correct_l97_97957

noncomputable def crayons_allocation : Prop :=
  ∃ (F B J S : ℕ), 
    F + B + J + S = 96 ∧ 
    F = 2 * B ∧ 
    J = 3 * S ∧ 
    B = 12 ∧ 
    F = 24 ∧ 
    J = 45 ∧ 
    S = 15

theorem crayons_allocation_correct : crayons_allocation :=
  sorry

end crayons_allocation_correct_l97_97957


namespace inequality_solution_set_l97_97745

theorem inequality_solution_set :
  {x : ℝ | (x + 1) / (x - 3) ≥ 0} = {x : ℝ | x ≤ -1 ∨ x > 3} := 
by 
  sorry

end inequality_solution_set_l97_97745


namespace find_primes_a_l97_97677

theorem find_primes_a :
  ∀ (a : ℕ), (∀ n : ℕ, n < a → Nat.Prime (4 * n * n + a)) → (a = 3 ∨ a = 7) :=
by
  sorry

end find_primes_a_l97_97677


namespace reciprocal_of_sum_l97_97194

-- Define the fractions
def a := (1: ℚ) / 2
def b := (1: ℚ) / 3

-- Define their sum
def c := a + b

-- Define the expected reciprocal
def reciprocal := (6: ℚ) / 5

-- The theorem we want to prove:
theorem reciprocal_of_sum : (c⁻¹ = reciprocal) :=
by 
  sorry

end reciprocal_of_sum_l97_97194


namespace distance_between_parallel_lines_l97_97613

theorem distance_between_parallel_lines
  (O A B C D P Q : ℝ) -- Points on the circle with P and Q as defined midpoints
  (r d : ℝ) -- Radius of the circle and distance between the parallel lines
  (h_AB : dist A B = 36) -- Length of chord AB
  (h_CD : dist C D = 36) -- Length of chord CD
  (h_BC : dist B C = 40) -- Length of chord BC
  (h_OA : dist O A = r) 
  (h_OB : dist O B = r)
  (h_OC : dist O C = r)
  (h_PQ_parallel : dist P Q = d) -- Midpoints
  : d = 4 * Real.sqrt 19 / 3 :=
sorry

end distance_between_parallel_lines_l97_97613


namespace solution_exists_unique_l97_97356

theorem solution_exists_unique (x y : ℝ) : (x + y = 2 ∧ x - y = 0) ↔ (x = 1 ∧ y = 1) := 
by
  sorry

end solution_exists_unique_l97_97356


namespace shaded_to_largest_ratio_l97_97590

theorem shaded_to_largest_ratio :
  let r1 := 1
  let r2 := 2
  let r3 := 3
  let r4 := 4
  let area (r : ℝ) := π * r^2
  let largest_circle_area := area r4
  let innermost_shaded_area := area r1
  let outermost_shaded_area := area r3 - area r2
  let shaded_area := innermost_shaded_area + outermost_shaded_area
  shaded_area / largest_circle_area = 3 / 8 :=
by
  sorry

end shaded_to_largest_ratio_l97_97590


namespace find_f_neg5_l97_97129

-- Define the function f and the constants a, b, and c
def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 5

-- State the main theorem we want to prove
theorem find_f_neg5 (a b c : ℝ) (h : f 5 a b c = 9) : f (-5) a b c = 1 :=
by
  sorry

end find_f_neg5_l97_97129


namespace sum_of_eight_numbers_l97_97303

theorem sum_of_eight_numbers (average : ℝ) (h : average = 5) :
  (8 * average) = 40 :=
by
  sorry

end sum_of_eight_numbers_l97_97303


namespace cloud9_total_revenue_after_discounts_and_refunds_l97_97242

theorem cloud9_total_revenue_after_discounts_and_refunds :
  let individual_total := 12000
  let individual_early_total := 3000
  let group_a_total := 6000
  let group_a_participants := 8
  let group_b_total := 9000
  let group_b_participants := 15
  let group_c_total := 15000
  let group_c_participants := 22
  let individual_refund1 := 500
  let individual_refund1_count := 3
  let individual_refund2 := 300
  let individual_refund2_count := 2
  let group_refund := 800
  let group_refund_count := 2

  -- Discounts
  let early_booking_discount := 0.03
  let discount_between_5_and_10 := 0.05
  let discount_between_11_and_20 := 0.1
  let discount_21_and_more := 0.15

  -- Calculating individual bookings
  let individual_early_discount_total := individual_early_total * early_booking_discount
  let individual_total_after_discount := individual_total - individual_early_discount_total

  -- Calculating group bookings
  let group_a_discount := group_a_total * discount_between_5_and_10
  let group_a_early_discount := (group_a_total - group_a_discount) * early_booking_discount
  let group_a_total_after_discount := group_a_total - group_a_discount - group_a_early_discount

  let group_b_discount := group_b_total * discount_between_11_and_20
  let group_b_total_after_discount := group_b_total - group_b_discount

  let group_c_discount := group_c_total * discount_21_and_more
  let group_c_early_discount := (group_c_total - group_c_discount) * early_booking_discount
  let group_c_total_after_discount := group_c_total - group_c_discount - group_c_early_discount

  let total_group_after_discount := group_a_total_after_discount + group_b_total_after_discount + group_c_total_after_discount

  -- Calculating refunds
  let total_individual_refunds := (individual_refund1 * individual_refund1_count) + (individual_refund2 * individual_refund2_count)
  let total_group_refunds := group_refund

  let total_refunds := total_individual_refunds + total_group_refunds

  -- Final total calculation after all discounts and refunds
  let final_total := individual_total_after_discount + total_group_after_discount - total_refunds
  final_total = 35006.50 := by
  -- The rest of the proof would go here, but we use sorry to bypass the proof.
  sorry

end cloud9_total_revenue_after_discounts_and_refunds_l97_97242


namespace counterexample_disproves_statement_l97_97085

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem counterexample_disproves_statement :
  ∃ n : ℕ, ¬ is_prime n ∧ is_prime (n + 3) :=
  by
    use 8
    -- Proof that 8 is not prime
    -- Proof that 11 (8 + 3) is prime
    sorry

end counterexample_disproves_statement_l97_97085


namespace trigonometric_identity_l97_97417

theorem trigonometric_identity (α x : ℝ) (h₁ : 5 * Real.cos α = x) (h₂ : x ^ 2 + 16 = 25) (h₃ : α > Real.pi / 2 ∧ α < Real.pi):
  x = -3 ∧ Real.tan α = -4 / 3 :=
by
  sorry

end trigonometric_identity_l97_97417


namespace find_a_l97_97819

-- Define the function f
def f (a x : ℝ) := a * x^3 - 2 * x

-- State the theorem, asserting that if f passes through the point (-1, 4) then a = -2.
theorem find_a (a : ℝ) (h : f a (-1) = 4) : a = -2 :=
by {
    sorry
}

end find_a_l97_97819


namespace probability_contemporaries_l97_97878

theorem probability_contemporaries (total_years : ℕ) (life_span : ℕ) (born_range : ℕ)
  (h1 : total_years = 300)
  (h2 : life_span = 80)
  (h3 : born_range = 300) :
  (∃λ p : ℚ, p = 104 / 225 ∧ 
   let lines_intersect := (0, 80) :: (80, 0) :: (220, 300) :: (300, 220) :: []
   in lines_intersect ≠ [] ∧ lines_intersect.length = 4 
   ∧ region_area 0 300 300 (λ x y, (y ≥ x - life_span) ∧ (y ≤ x + life_span)) = 41600
   ∧ total_area 0 300 300 = 90000
   ∧ prob := region_area / total_area,
     prob = p) :=
by sorry

end probability_contemporaries_l97_97878


namespace parabola_point_coordinates_l97_97434

theorem parabola_point_coordinates (x y : ℝ) (h_parabola : y^2 = 8 * x) 
    (h_distance_focus : (x + 2)^2 + y^2 = 81) : 
    (x = 7 ∧ y = 2 * Real.sqrt 14) ∨ (x = 7 ∧ y = -2 * Real.sqrt 14) :=
by {
  -- Proof will be inserted here
  sorry
}

end parabola_point_coordinates_l97_97434


namespace acute_triangle_pyramid_exists_l97_97336

theorem acute_triangle_pyramid_exists :
  ∀ (A B C : EuclideanGeometry.Point 3), 
  EuclideanGeometry.angled_triangle A B C → 
  (∀ (SA SB SC : EuclideanGeometry.Line 3), EuclideanGeometry.perpendicular SA SB ∧ 
  EuclideanGeometry.perpendicular SB SC ∧ EuclideanGeometry.perpendicular SC SA) → 
  ∃ (S : EuclideanGeometry.Point 3), 
  EuclideanGeometry.triangular_pyramid S A B C :=
by
  sorry

end acute_triangle_pyramid_exists_l97_97336


namespace population_growth_l97_97909

theorem population_growth :
  let scale_factor1 := 1 + 10 / 100
  let scale_factor2 := 1 + 20 / 100
  let k := 2 * 20
  let scale_factor3 := 1 + k / 100
  let combined_scale := scale_factor1 * scale_factor2 * scale_factor3
  (combined_scale - 1) * 100 = 84.8 :=
by
  sorry

end population_growth_l97_97909


namespace eval_floor_neg_seven_fourths_l97_97257

theorem eval_floor_neg_seven_fourths : 
  ∃ (x : ℚ), x = -7 / 4 ∧ ∀ y, y ≤ x ∧ y ∈ ℤ → y ≤ -2 :=
by
  obtain ⟨x, hx⟩ : ∃ (x : ℚ), x = -7 / 4 := ⟨-7 / 4, rfl⟩,
  use x,
  split,
  { exact hx },
  { intros y hy,
    sorry }

end eval_floor_neg_seven_fourths_l97_97257


namespace number_of_sets_of_positive_integers_l97_97546

theorem number_of_sets_of_positive_integers : 
  ∃ n : ℕ, n = 3333 ∧ ∀ x y z : ℕ, x > 0 → y > 0 → z > 0 → x < y → y < z → x + y + z = 203 → n = 3333 :=
by
  sorry

end number_of_sets_of_positive_integers_l97_97546


namespace factor_expression_l97_97123

theorem factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := 
by 
  sorry

end factor_expression_l97_97123


namespace inequality_holds_l97_97324

theorem inequality_holds (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
  a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 :=
sorry

end inequality_holds_l97_97324


namespace number_subtracted_eq_l97_97769

theorem number_subtracted_eq (x n : ℤ) (h1 : x + 1315 + 9211 - n = 11901) (h2 : x = 88320) : n = 86945 :=
by
  sorry

end number_subtracted_eq_l97_97769


namespace binomial_probability_example_l97_97130

noncomputable def binomialProbability (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem binomial_probability_example :
  binomialProbability 5 3 (1/3) = 40 / 243 :=
by
  sorry  -- This is where the proof would go.

end binomial_probability_example_l97_97130


namespace simplify_expression_l97_97854

theorem simplify_expression :
  ( ∀ (a b c : ℕ), c > 0 ∧ (∀ p : ℕ, Prime p → ¬ p^2 ∣ c) →
  (a - b * Real.sqrt c = (28 - 16 * Real.sqrt 3) * 2 ^ (-2 - Real.sqrt 5))) :=
sorry

end simplify_expression_l97_97854


namespace man_l97_97906

-- Lean 4 statement
theorem man's_speed_against_stream (speed_with_stream : ℝ) (speed_still_water : ℝ) 
(h1 : speed_with_stream = 16) (h2 : speed_still_water = 4) : 
  |speed_still_water - (speed_with_stream - speed_still_water)| = 8 :=
by
  -- Dummy proof since only statement is required
  sorry

end man_l97_97906


namespace real_condition_complex_condition_pure_imaginary_condition_l97_97551

-- Definitions for our conditions
def is_real (z : ℂ) : Prop := z.im = 0
def is_complex (z : ℂ) : Prop := z.im ≠ 0
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- The given complex number definition
def z (m : ℝ) : ℂ := { re := m^2 + m, im := m^2 - 1 }

-- Prove that for z to be a real number, m must be ±1
theorem real_condition (m : ℝ) : is_real (z m) ↔ m = 1 ∨ m = -1 := 
sorry

-- Prove that for z to be a complex number, m must not be ±1 
theorem complex_condition (m : ℝ) : is_complex (z m) ↔ m ≠ 1 ∧ m ≠ -1 := 
sorry 

-- Prove that for z to be a pure imaginary number, m must be 0
theorem pure_imaginary_condition (m : ℝ) : is_pure_imaginary (z m) ↔ m = 0 := 
sorry 

end real_condition_complex_condition_pure_imaginary_condition_l97_97551


namespace jim_taxi_total_charge_l97_97375

noncomputable def total_charge (initial_fee : ℝ) (per_mile_fee : ℝ) (mile_chunk : ℝ) (distance : ℝ) : ℝ :=
  initial_fee + (distance / mile_chunk) * per_mile_fee

theorem jim_taxi_total_charge :
  total_charge 2.35 0.35 (2/5) 3.6 = 5.50 :=
by
  sorry

end jim_taxi_total_charge_l97_97375


namespace max_reflections_l97_97093

theorem max_reflections (P Q R M : Type) (angle : ℝ) :
  0 < angle ∧ angle ≤ 30 ∧ (∃ n : ℕ, 10 * n = angle) →
  ∃ n : ℕ, n ≤ 3 :=
by
  sorry

end max_reflections_l97_97093


namespace greg_initial_money_eq_36_l97_97673

theorem greg_initial_money_eq_36 
  (Earl_initial Fred_initial : ℕ)
  (Greg_initial : ℕ)
  (Earl_owes_Fred Fred_owes_Greg Greg_owes_Earl : ℕ)
  (Total_after_debt : ℕ)
  (hEarl_initial : Earl_initial = 90)
  (hFred_initial : Fred_initial = 48)
  (hEarl_owes_Fred : Earl_owes_Fred = 28)
  (hFred_owes_Greg : Fred_owes_Greg = 32)
  (hGreg_owes_Earl : Greg_owes_Earl = 40)
  (hTotal_after_debt : Total_after_debt = 130) :
  Greg_initial = 36 :=
sorry

end greg_initial_money_eq_36_l97_97673


namespace proof_q1_a1_proof_q2_a2_proof_q3_a3_proof_q4_a4_l97_97599

variables (G : Type) [Group G] (kidney testis liver : G)
variables (SudanIII gentianViolet JanusGreenB dissociationFixative : G)

-- Conditions c1, c2, c3
def c1 : Prop := True -- Meiosis occurs in gonads, we simplify this in Lean to a true condition for brevity
def c2 : Prop := True -- Steps for slide preparation
def c3 : Prop := True -- Materials available

-- Questions
def q1 : G := testis
def q2 : G := dissociationFixative
def q3 : G := gentianViolet
def q4 : List G := [kidney, dissociationFixative, gentianViolet] -- Assume these are placeholders for correct cell types

-- Answers
def a1 : G := testis
def a2 : G := dissociationFixative
def a3 : G := gentianViolet
def a4 : List G := [testis, dissociationFixative, gentianViolet] -- Correct cells

-- Proving the equivalence of questions and answers given the conditions
theorem proof_q1_a1 : c1 ∧ c2 ∧ c3 → q1 = a1 := 
by sorry

theorem proof_q2_a2 : c1 ∧ c2 ∧ c3 → q2 = a2 := 
by sorry

theorem proof_q3_a3 : c1 ∧ c2 ∧ c3 → q3 = a3 := 
by sorry

theorem proof_q4_a4 : c1 ∧ c2 ∧ c3 → q4 = a4 := 
by sorry

end proof_q1_a1_proof_q2_a2_proof_q3_a3_proof_q4_a4_l97_97599


namespace crayons_left_l97_97872

theorem crayons_left (initial_crayons : ℕ) (kiley_fraction : ℚ) (joe_fraction : ℚ) 
  (initial_crayons_eq : initial_crayons = 48) 
  (kiley_fraction_eq : kiley_fraction = 1/4) 
  (joe_fraction_eq : joe_fraction = 1/2): 
  (initial_crayons - initial_crayons * kiley_fraction - (initial_crayons - initial_crayons * kiley_fraction) * joe_fraction) = 18 := 
by 
  sorry

end crayons_left_l97_97872


namespace cost_price_computer_table_l97_97505

variable (CP SP : ℝ)

theorem cost_price_computer_table (h1 : SP = 2 * CP) (h2 : SP = 1000) : CP = 500 := by
  sorry

end cost_price_computer_table_l97_97505


namespace inequality_holds_l97_97325

theorem inequality_holds (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
  a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 :=
sorry

end inequality_holds_l97_97325


namespace grouping_count_l97_97030

theorem grouping_count (men women : ℕ) 
  (h_men : men = 4) (h_women : women = 5)
  (at_least_one_man_woman : ∀ (g1 g2 g3 : Finset (Fin 9)), 
    g1.card = 3 → g2.card = 3 → g3.card = 3 → g1 ∩ g2 = ∅ → g2 ∩ g3 = ∅ → g3 ∩ g1 = ∅ → 
    (g1 ∩ univ.filter (· < 4)).nonempty ∧ (g1 ∩ univ.filter (· ≥ 4)).nonempty ∧
    (g2 ∩ univ.filter (· < 4)).nonempty ∧ (g2 ∩ univ.filter (· ≥ 4)).nonempty ∧
    (g3 ∩ univ.filter (· < 4)).nonempty ∧ (g3 ∩ univ.filter (· ≥ 4)).nonempty) :
  (choose 4 1 * choose 5 2 * choose 3 1 * choose 3 2) / 2! = 180 :=
sorry

end grouping_count_l97_97030


namespace a_equals_1_or_2_l97_97822

def M (a : ℤ) : Set ℤ := {a, 0}
def N : Set ℤ := {x : ℤ | x^2 - 3 * x < 0}
def non_empty_intersection (a : ℤ) : Prop := (M a ∩ N).Nonempty

theorem a_equals_1_or_2 (a : ℤ) (h : non_empty_intersection a) : a = 1 ∨ a = 2 := by
  sorry

end a_equals_1_or_2_l97_97822


namespace function_symmetric_and_monotonic_l97_97293

noncomputable def f (x : ℝ) : ℝ := (Real.cos x)^4 - 2 * Real.sin x * Real.cos x - (Real.sin x)^4

theorem function_symmetric_and_monotonic :
  (∀ x, f (x + (3/8) * π) = f (x - (3/8) * π)) ∧
  (∀ x y, x ∈  Set.Icc (-(π / 8)) ((3 * π) / 8) → y ∈  Set.Icc (-(π / 8)) ((3 * π) / 8) → x < y → f x > f y) :=
by
  sorry

end function_symmetric_and_monotonic_l97_97293


namespace rainfall_on_thursday_l97_97943

theorem rainfall_on_thursday
  (monday_am : ℝ := 2)
  (monday_pm : ℝ := 1)
  (tuesday_factor : ℝ := 2)
  (wednesday : ℝ := 0)
  (thursday : ℝ)
  (weekly_avg : ℝ := 4)
  (days_in_week : ℕ := 7)
  (total_weekly_rain : ℝ := days_in_week * weekly_avg) :
  2 * (monday_am + monday_pm + tuesday_factor * (monday_am + monday_pm) + thursday) 
    = total_weekly_rain
  → thursday = 5 :=
by
  sorry

end rainfall_on_thursday_l97_97943


namespace hyperbola_asymptote_l97_97120

theorem hyperbola_asymptote (y x : ℝ) :
  (y^2 / 9 - x^2 / 16 = 1) → (y = x * 3 / 4 ∨ y = -x * 3 / 4) :=
sorry

end hyperbola_asymptote_l97_97120


namespace coordinates_of_point_M_l97_97155

theorem coordinates_of_point_M 
  (M : ℝ × ℝ) 
  (dist_x_axis : abs M.2 = 5) 
  (dist_y_axis : abs M.1 = 4) 
  (second_quadrant : M.1 < 0 ∧ M.2 > 0) : 
  M = (-4, 5) := 
sorry

end coordinates_of_point_M_l97_97155


namespace fraction_numerator_l97_97345

theorem fraction_numerator (x : ℚ) 
  (h1 : ∃ (n : ℚ), n = 4 * x - 9) 
  (h2 : x / (4 * x - 9) = 3 / 4) 
  : x = 27 / 8 := sorry

end fraction_numerator_l97_97345


namespace temperature_on_tuesday_l97_97473

theorem temperature_on_tuesday 
  (M T W Th F Sa : ℝ)
  (h1 : (M + T + W) / 3 = 38)
  (h2 : (T + W + Th) / 3 = 42)
  (h3 : (W + Th + F) / 3 = 44)
  (h4 : (Th + F + Sa) / 3 = 46)
  (hF : F = 43)
  (pattern : M + 2 = Sa ∨ M - 1 = Sa) :
  T = 80 :=
sorry

end temperature_on_tuesday_l97_97473


namespace williams_land_percentage_l97_97676

variable (total_tax : ℕ) (williams_tax : ℕ)

theorem williams_land_percentage (h1 : total_tax = 3840) (h2 : williams_tax = 480) : 
  (williams_tax:ℚ) / (total_tax:ℚ) * 100 = 12.5 := 
  sorry

end williams_land_percentage_l97_97676


namespace capacity_of_other_bottle_l97_97773

theorem capacity_of_other_bottle (x : ℝ) :
  (16 / 3) * (x / 8) + (16 / 3) = 8 → x = 4 := by
  -- the proof will go here
  sorry

end capacity_of_other_bottle_l97_97773


namespace find_s_l97_97976

theorem find_s (k s : ℝ) (h1 : 5 = k * 2^s) (h2 : 45 = k * 8^s) : s = (Real.log 9) / (2 * Real.log 2) :=
by
  sorry

end find_s_l97_97976


namespace proposition_d_correct_l97_97213

theorem proposition_d_correct (a b c : ℝ) (h : a > b) : a - c > b - c := 
by
  sorry

end proposition_d_correct_l97_97213


namespace trip_savings_l97_97615

theorem trip_savings :
  let ticket_cost := 10
  let combo_cost := 10
  let ticket_discount := 0.20
  let combo_discount := 0.50
  (ticket_discount * ticket_cost + combo_discount * combo_cost) = 7 := 
by
  sorry

end trip_savings_l97_97615


namespace count_four_digit_even_numbers_excluding_5_and_6_l97_97298

theorem count_four_digit_even_numbers_excluding_5_and_6 : 
  ∃ n : ℕ, n = 1792 ∧ 
    (∀ d1 d2 d3 d4: ℕ, 
      d1 ∈ {1, 2, 3, 4, 7, 8, 9} →
      d2 ∈ {0, 1, 2, 3, 4, 7, 8, 9} →
      d3 ∈ {0, 1, 2, 3, 4, 7, 8, 9} →
      d4 ∈ {0, 2, 4, 8} →
      d1 > 0 ∧ d4 % 2 = 0) 
      ∧ n = 7 * 8 * 8 * 4 := 
by
  existsi 1792
  split
  focus
    reflexivity
  sorry

end count_four_digit_even_numbers_excluding_5_and_6_l97_97298


namespace find_y_l97_97995

def star (a b : ℝ) : ℝ := a * b + 3 * b - a

theorem find_y (y : ℝ) (h : star 7 y = 47) : y = 5.4 := 
by 
  sorry

end find_y_l97_97995


namespace infinite_sum_equals_two_l97_97111

theorem infinite_sum_equals_two :
  ∑' k : ℕ, (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 2 :=
sorry

end infinite_sum_equals_two_l97_97111


namespace inequality_solution_l97_97059

def f (x : ℝ) : ℝ := ((x - 1) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7))

theorem inequality_solution :
  { x : ℝ | f x > 0 } = {x : ℝ | x < 1} ∪ {x : ℝ | 2 < x ∧ x < 4} ∪ {x : ℝ | 5 < x ∧ x < 6} ∪ {x : ℝ | x > 7} :=
by
  sorry


end inequality_solution_l97_97059


namespace largest_square_area_l97_97835

theorem largest_square_area (a b c : ℝ)
  (right_triangle : a^2 + b^2 = c^2)
  (square_area_sum : a^2 + b^2 + c^2 = 450)
  (area_a : a^2 = 100) :
  c^2 = 225 :=
by
  sorry

end largest_square_area_l97_97835


namespace arccos_one_half_eq_pi_div_three_l97_97928

theorem arccos_one_half_eq_pi_div_three :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ π ∧ cos θ = (1 / 2) ∧ arccos (1 / 2) = θ :=
sorry

end arccos_one_half_eq_pi_div_three_l97_97928


namespace jana_walk_distance_l97_97034

-- Define the time taken to walk one mile and the rest period
def walk_time_per_mile : ℕ := 24
def rest_time_per_mile : ℕ := 6

-- Define the total time spent per mile (walking + resting)
def total_time_per_mile : ℕ := walk_time_per_mile + rest_time_per_mile

-- Define the total available time
def total_available_time : ℕ := 78

-- Define the number of complete cycles of walking and resting within the total available time
def complete_cycles : ℕ := total_available_time / total_time_per_mile

-- Define the distance walked per cycle (in miles)
def distance_per_cycle : ℝ := 1.0

-- Define the total distance walked
def total_distance_walked : ℝ := complete_cycles * distance_per_cycle

-- The proof statement
theorem jana_walk_distance : total_distance_walked = 2.0 := by
  sorry

end jana_walk_distance_l97_97034


namespace unique_solution_c_value_l97_97406

-- Define the main problem: the parameter c for which a given system of equations has a unique solution.
theorem unique_solution_c_value (c : ℝ) : 
  (∀ x y : ℝ, 2 * abs (x + 7) + abs (y - 4) = c ∧ abs (x + 4) + 2 * abs (y - 7) = c → 
   (x = -7 ∧ y = 7)) ↔ c = 3 :=
by sorry

end unique_solution_c_value_l97_97406


namespace floor_neg_seven_fourths_l97_97272

theorem floor_neg_seven_fourths : (Int.floor (-7 / 4 : ℚ) = -2) := 
by
  sorry

end floor_neg_seven_fourths_l97_97272


namespace find_numbers_between_70_and_80_with_gcd_6_l97_97402

theorem find_numbers_between_70_and_80_with_gcd_6 :
  ∃ n, 70 ≤ n ∧ n ≤ 80 ∧ Nat.gcd n 30 = 6 ∧ (n = 72 ∨ n = 78) :=
by
  sorry

end find_numbers_between_70_and_80_with_gcd_6_l97_97402


namespace ways_to_place_people_into_groups_l97_97017

theorem ways_to_place_people_into_groups :
  let men := 4
  let women := 5
  ∃ (groups : Nat), groups = 2 ∧
  ∀ (g : Nat → (Fin 3 → (Bool → Nat → Nat))),
    (∀ i, i < group_counts → ∃ m w, g i m w < people ∧ g i m (1 - w) < people ∧ m + 1 - w + (1 - m) + w = 3) →
    let groups : List (List (Fin 2)) := [
      [(Fin.mk 1 dec_trivial, Fin.mk 2 dec_trivial)],
      [(Fin.mk 1 dec_trivial, Fin.mk 2 dec_trivial)]
    ] in
    g.mk 1 dec_trivial * g.mk 2 dec_trivial = 360 :=
sorry

end ways_to_place_people_into_groups_l97_97017


namespace ordered_pair_for_quadratic_with_same_roots_l97_97365

theorem ordered_pair_for_quadratic_with_same_roots (b c : ℝ) :
  (∀ x : ℝ, |x - 4| = 3 ↔ (x = 7 ∨ x = 1)) →
  (∀ x : ℝ, x^2 + b * x + c = 0 ↔ (x = 7 ∨ x = 1)) →
  (b, c) = (-8, 7) :=
by
  intro h1 h2
  sorry

end ordered_pair_for_quadratic_with_same_roots_l97_97365


namespace amy_height_l97_97651

variable (A H N : ℕ)

theorem amy_height (h1 : A = 157) (h2 : A = H + 4) (h3 : H = N + 3) :
  N = 150 := sorry

end amy_height_l97_97651


namespace strips_overlap_area_l97_97780

theorem strips_overlap_area (L1 L2 AL AR S : ℝ) (hL1 : L1 = 9) (hL2 : L2 = 7) (hAL : AL = 27) (hAR : AR = 18) 
    (hrel : (AL + S) / (AR + S) = L1 / L2) : S = 13.5 := 
by
  sorry

end strips_overlap_area_l97_97780


namespace angle_B_and_side_b_in_triangle_l97_97700

theorem angle_B_and_side_b_in_triangle
  (A B C : ℝ) (a b c: ℝ)
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_opposite_sides : a = b * sin C / sin B)
  (h_equation : 2 * c = sqrt 3 * a + 2 * b * cos A)
  (h_angle_sum : A + B + C = π)
  (h_c_val : c = 7)
  (h_b_sin : b * sin A = sqrt 3) :
  B = π / 6 ∧ b = sqrt 19 :=
by
  sorry

end angle_B_and_side_b_in_triangle_l97_97700


namespace gpa_of_entire_class_l97_97702

def students : ℕ := 200

def gpa1_num : ℕ := 18 * students / 100
def gpa2_num : ℕ := 27 * students / 100
def gpa3_num : ℕ := 22 * students / 100
def gpa4_num : ℕ := 12 * students / 100
def gpa5_num : ℕ := students - (gpa1_num + gpa2_num + gpa3_num + gpa4_num)

def gpa1 : ℕ := 58
def gpa2 : ℕ := 63
def gpa3 : ℕ := 69
def gpa4 : ℕ := 75
def gpa5 : ℕ := 85

def total_points : ℕ :=
  (gpa1_num * gpa1) + (gpa2_num * gpa2) + (gpa3_num * gpa3) + (gpa4_num * gpa4) + (gpa5_num * gpa5)

def class_gpa : ℚ := total_points / students

theorem gpa_of_entire_class :
  class_gpa = 69.48 := 
  by
  sorry

end gpa_of_entire_class_l97_97702


namespace sum_of_squares_greater_than_cubics_l97_97569

theorem sum_of_squares_greater_than_cubics (a b c : ℝ)
  (h1 : a + b > c) 
  (h2 : a + c > b) 
  (h3 : b + c > a)
  : 
  (2 * (a + b + c) * (a^2 + b^2 + c^2)) / 3 > a^3 + b^3 + c^3 + a * b * c := 
by 
  sorry

end sum_of_squares_greater_than_cubics_l97_97569


namespace cubs_more_home_runs_l97_97523

-- Define the conditions for the Chicago Cubs
def cubs_home_runs_third_inning : Nat := 2
def cubs_home_runs_fifth_inning : Nat := 1
def cubs_home_runs_eighth_inning : Nat := 2

-- Define the conditions for the Cardinals
def cardinals_home_runs_second_inning : Nat := 1
def cardinals_home_runs_fifth_inning : Nat := 1

-- Total home runs scored by each team
def total_cubs_home_runs : Nat :=
  cubs_home_runs_third_inning + cubs_home_runs_fifth_inning + cubs_home_runs_eighth_inning

def total_cardinals_home_runs : Nat :=
  cardinals_home_runs_second_inning + cardinals_home_runs_fifth_inning

-- The statement to prove
theorem cubs_more_home_runs : total_cubs_home_runs - total_cardinals_home_runs = 3 := by
  sorry

end cubs_more_home_runs_l97_97523


namespace correct_operation_l97_97211

theorem correct_operation (x : ℝ) : (2 * x ^ 3) ^ 2 = 4 * x ^ 6 := 
  sorry

end correct_operation_l97_97211


namespace preston_receives_total_amount_l97_97850

theorem preston_receives_total_amount :
  let price_per_sandwich := 5
  let delivery_fee := 20
  let num_sandwiches := 18
  let tip_percent := 0.10
  let sandwich_cost := num_sandwiches * price_per_sandwich
  let initial_total := sandwich_cost + delivery_fee
  let tip := initial_total * tip_percent
  let final_total := initial_total + tip
  final_total = 121 := 
by
  sorry

end preston_receives_total_amount_l97_97850


namespace angle_45_deg_is_75_venerts_l97_97849

-- There are 600 venerts in a full circle.
def venus_full_circle : ℕ := 600

-- A full circle on Earth is 360 degrees.
def earth_full_circle : ℕ := 360

-- Conversion factor from degrees to venerts.
def degrees_to_venerts (deg : ℕ) : ℕ :=
  deg * (venus_full_circle / earth_full_circle)

-- Angle of 45 degrees in venerts.
def angle_45_deg_in_venerts : ℕ := 45 * (venus_full_circle / earth_full_circle)

theorem angle_45_deg_is_75_venerts :
  angle_45_deg_in_venerts = 75 :=
by
  -- Proof will be inserted here.
  sorry

end angle_45_deg_is_75_venerts_l97_97849


namespace g_of_3_over_8_l97_97744

def g (x : ℝ) : ℝ := sorry

theorem g_of_3_over_8 :
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g 0 = 0) ∧
    (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 1 → g x ≤ g y) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g (x / 4) = (g x) / 3) →
    g (3 / 8) = 2 / 9 :=
sorry

end g_of_3_over_8_l97_97744


namespace large_cross_area_is_60_cm_squared_l97_97868

noncomputable def small_square_area (s : ℝ) := s * s
noncomputable def large_square_area (s : ℝ) := 4 * small_square_area s
noncomputable def small_cross_area (s : ℝ) := 5 * small_square_area s
noncomputable def large_cross_area (s : ℝ) := 5 * large_square_area s
noncomputable def remaining_area (s : ℝ) := large_cross_area s - small_cross_area s

theorem large_cross_area_is_60_cm_squared :
  ∃ (s : ℝ), remaining_area s = 45 → large_cross_area s = 60 :=
by
  sorry

end large_cross_area_is_60_cm_squared_l97_97868


namespace smaller_fraction_is_l97_97197

theorem smaller_fraction_is
  (x y : ℝ)
  (h₁ : x + y = 7 / 8)
  (h₂ : x * y = 1 / 12) :
  min x y = (7 - Real.sqrt 17) / 16 :=
sorry

end smaller_fraction_is_l97_97197


namespace percentage_alcohol_in_first_vessel_is_zero_l97_97646

theorem percentage_alcohol_in_first_vessel_is_zero (x : ℝ) :
  ∀ (alcohol_first_vessel total_vessel_capacity first_vessel_capacity second_vessel_capacity concentration_mixture : ℝ),
  first_vessel_capacity = 2 →
  (∃ xpercent, alcohol_first_vessel = (first_vessel_capacity * xpercent / 100)) →
  second_vessel_capacity = 6 →
  (∃ ypercent, ypercent = 40 ∧ alcohol_first_vessel + 2.4 = concentration_mixture * (total_vessel_capacity/8) * 8) →
  concentration_mixture = 0.3 →
  0 = x := sorry

end percentage_alcohol_in_first_vessel_is_zero_l97_97646


namespace grocery_store_price_l97_97223

-- Definitions based on the conditions
def bulk_price_per_case : ℝ := 12.00
def bulk_cans_per_case : ℝ := 48.0
def grocery_cans_per_pack : ℝ := 12.0
def additional_cost_per_can : ℝ := 0.25

-- The proof statement
theorem grocery_store_price : 
  (bulk_price_per_case / bulk_cans_per_case + additional_cost_per_can) * grocery_cans_per_pack = 6.00 :=
by
  sorry

end grocery_store_price_l97_97223


namespace total_get_well_cards_l97_97331

def dozens_to_cards (d : ℕ) : ℕ := d * 12
def hundreds_to_cards (h : ℕ) : ℕ := h * 100

theorem total_get_well_cards 
  (d_hospital : ℕ) (h_hospital : ℕ)
  (d_home : ℕ) (h_home : ℕ) :
  d_hospital = 25 ∧ h_hospital = 7 ∧ d_home = 39 ∧ h_home = 3 →
  (dozens_to_cards d_hospital + hundreds_to_cards h_hospital +
   dozens_to_cards d_home + hundreds_to_cards h_home) = 1768 :=
by
  intros
  sorry

end total_get_well_cards_l97_97331


namespace find_number_l97_97288

theorem find_number (x : ℝ) 
  (h : (28 + x / 69) * 69 = 1980) :
  x = 1952 :=
sorry

end find_number_l97_97288


namespace central_angle_l97_97988

variable (O : Type)
variable (A B C : O)
variable (angle_ABC : ℝ) 

theorem central_angle (h : angle_ABC = 50) : 2 * angle_ABC = 100 := by
  sorry

end central_angle_l97_97988


namespace calculation_correct_l97_97241

theorem calculation_correct : 
  ((2 * (15^2 + 35^2 + 21^2) - (3^4 + 5^4 + 7^4)) / (3 + 5 + 7)) = 45 := by
  sorry

end calculation_correct_l97_97241


namespace distance_second_day_l97_97732

theorem distance_second_day 
  (total_distance : ℕ)
  (a1 : ℕ)
  (n : ℕ)
  (r : ℚ)
  (hn : n = 6)
  (htotal : total_distance = 378)
  (hr : r = 1 / 2)
  (geo_sum : a1 * (1 - r^n) / (1 - r) = total_distance) :
  a1 * r = 96 :=
by
  sorry

end distance_second_day_l97_97732


namespace probability_of_fourth_roll_l97_97529

-- Define the conditions 
structure Die :=
(fair : Bool) 
(biased_six : Bool)
(biased_one : Bool)

-- Define the probability function
def roll_prob (d : Die) (f : Bool) : ℚ :=
  if d.fair then 1/6
  else if d.biased_six then if f then 1/2 else 1/10
  else if d.biased_one then if f then 1/10 else 1/5
  else 0

def probability_of_fourth_six (p q : ℕ) (r1 r2 r3 : Bool) (d : Die) : ℚ :=
  (if r1 && r2 && r3 then roll_prob d true else 0) 

noncomputable def final_probability (d1 d2 d3 : Die) (prob_fair distorted_rolls : Bool) : ℚ :=
  let fair_prob := if distorted_rolls then roll_prob d1 true else roll_prob d1 false
  let biased_six_prob := if distorted_rolls then roll_prob d2 true else roll_prob d2 false
  let total_prob := fair_prob + biased_six_prob
  let fair := fair_prob / total_prob
  let biased_six := biased_six_prob / total_prob
  fair * roll_prob d1 true + biased_six * roll_prob d2 true

theorem probability_of_fourth_roll
  (d1 : Die) (d2 : Die) (d3 : Die)
  (h1 : d1.fair = true)
  (h2 : d2.biased_six = true)
  (h3 : d3.biased_one = true)
  (h4 : ∀ d, d1 = d ∨ d2 = d ∨ d3 = d)
  (r1 r2 r3 : Bool)
  : ∃ p q : ℕ, p + q = 11 ∧ final_probability d1 d2 d3 true = 5/6 := 
sorry

end probability_of_fourth_roll_l97_97529


namespace expression_value_l97_97670

theorem expression_value (b : ℝ) (hb : b = 1 / 3) :
    (3 * b⁻¹ - b⁻¹ / 3) / b^2 = 72 :=
sorry

end expression_value_l97_97670


namespace plus_minus_pairs_l97_97199

theorem plus_minus_pairs (a b p q : ℕ) (h_plus_pairs : p = a) (h_minus_pairs : q = b) : 
  a - b = p - q := 
by 
  sorry

end plus_minus_pairs_l97_97199


namespace fraction_simplification_l97_97497

theorem fraction_simplification :
  (1^2 + 1) * (2^2 + 1) * (3^2 + 1) / ((2^2 - 1) * (3^2 - 1) * (4^2 - 1)) = 5 / 18 :=
by
  sorry

end fraction_simplification_l97_97497


namespace chef_cooked_potatoes_l97_97225

theorem chef_cooked_potatoes
  (total_potatoes : ℕ)
  (cooking_time_per_potato : ℕ)
  (remaining_cooking_time : ℕ)
  (left_potatoes : ℕ)
  (cooked_potatoes : ℕ) :
  total_potatoes = 16 →
  cooking_time_per_potato = 5 →
  remaining_cooking_time = 45 →
  remaining_cooking_time / cooking_time_per_potato = left_potatoes →
  total_potatoes - left_potatoes = cooked_potatoes →
  cooked_potatoes = 7 :=
by
  intros h_total h_cooking_time h_remaining_time h_left_potatoes h_cooked_potatoes
  sorry

end chef_cooked_potatoes_l97_97225


namespace probability_red_ball_l97_97012

-- Let P_red be the probability of drawing a red ball.
-- Let P_white be the probability of drawing a white ball.
-- Let P_black be the probability of drawing a black ball.
-- Let P_red_or_white be the probability of drawing a red or white ball.
-- Let P_red_or_black be the probability of drawing a red or black ball.

variable (P_red P_white P_black : ℝ)
variable (P_red_or_white P_red_or_black : ℝ)

-- Given conditions
axiom P_red_or_white_condition : P_red_or_white = 0.58
axiom P_red_or_black_condition : P_red_or_black = 0.62

-- The total probability must sum to 1.
axiom total_probability_condition : P_red + P_white + P_black = 1

-- Prove that the probability of drawing a red ball is 0.2.
theorem probability_red_ball : P_red = 0.2 :=
by
  -- To be proven
  sorry

end probability_red_ball_l97_97012


namespace father_20_bills_count_l97_97598

-- Defining the conditions from the problem.
variables (mother50 mother20 mother10 father50 father10 : ℕ)
def mother_total := mother50 * 50 + mother20 * 20 + mother10 * 10
def father_total (x : ℕ) := father50 * 50 + x * 20 + father10 * 10

-- Given conditions
axiom mother_given : mother50 = 1 ∧ mother20 = 2 ∧ mother10 = 3
axiom father_given : father50 = 4 ∧ father10 = 1
axiom school_fee : 350 = 350

-- Theorem to prove
theorem father_20_bills_count (x : ℕ) :
  mother_total 1 2 3 + father_total 4 x 1 = 350 → x = 1 :=
by sorry

end father_20_bills_count_l97_97598


namespace total_selling_price_l97_97229

theorem total_selling_price
  (CP : ℕ) (Gain : ℕ) (TCP : ℕ)
  (h1 : CP = 1200)
  (h2 : Gain = 3 * CP)
  (h3 : TCP = 18 * CP) :
  ∃ TSP : ℕ, TSP = 25200 := 
by
  sorry

end total_selling_price_l97_97229


namespace remainder_when_13_plus_x_divided_by_26_l97_97996

theorem remainder_when_13_plus_x_divided_by_26 (x : ℕ) (h1 : 9 * x % 26 = 1) : (13 + x) % 26 = 16 := 
by sorry

end remainder_when_13_plus_x_divided_by_26_l97_97996


namespace binary_to_decimal_101101_l97_97669

def binary_to_decimal (digits : List ℕ) : ℕ :=
  digits.foldr (λ (digit : ℕ) (acc : ℕ × ℕ) => (acc.1 + digit * 2 ^ acc.2, acc.2 + 1)) (0, 0) |>.1

theorem binary_to_decimal_101101 : binary_to_decimal [1, 0, 1, 1, 0, 1] = 45 :=
by
  -- Proof is needed but here we use sorry as placeholder.
  sorry

end binary_to_decimal_101101_l97_97669


namespace total_earnings_l97_97233

-- Definitions from the conditions.
def LaurynEarnings : ℝ := 2000
def AureliaEarnings : ℝ := 0.7 * LaurynEarnings

-- The theorem to prove.
theorem total_earnings (hL : LaurynEarnings = 2000) (hA : AureliaEarnings = 0.7 * 2000) :
    LaurynEarnings + AureliaEarnings = 3400 :=
by
    sorry  -- Placeholder for the proof.

end total_earnings_l97_97233


namespace total_apples_picked_l97_97173

-- Definitions based on conditions from part a)
def mike_apples : ℝ := 7.5
def nancy_apples : ℝ := 3.2
def keith_apples : ℝ := 6.1
def olivia_apples : ℝ := 12.4
def thomas_apples : ℝ := 8.6

-- The theorem we need to prove
theorem total_apples_picked : mike_apples + nancy_apples + keith_apples + olivia_apples + thomas_apples = 37.8 := by
    sorry

end total_apples_picked_l97_97173


namespace a_2n_is_perfect_square_l97_97453

noncomputable def a : ℕ → ℕ
| 0     := 0
| 1     := 1
| 2     := 1
| 3     := 2
| 4     := 4
| (n+5) := a (n+4) + a (n+2) + a n

noncomputable def f : ℕ → ℕ
| 0     := 0
| 1     := 1
| n + 2 := f (n + 1) + f n

theorem a_2n_is_perfect_square (n : ℕ) :
  ∃ f_n : ℕ, a (2 * n) = f_n * f_n :=
sorry

end a_2n_is_perfect_square_l97_97453


namespace broker_wealth_increase_after_two_years_l97_97087

theorem broker_wealth_increase_after_two_years :
  let initial_investment : ℝ := 100
  let first_year_increase : ℝ := 0.75
  let second_year_decrease : ℝ := 0.30
  let end_first_year := initial_investment * (1 + first_year_increase)
  let end_second_year := end_first_year * (1 - second_year_decrease)
  end_second_year - initial_investment = 22.50 :=
by
  sorry

end broker_wealth_increase_after_two_years_l97_97087


namespace circle_represents_real_l97_97304

theorem circle_represents_real
  (a : ℝ)
  (h : ∀ x y : ℝ, x^2 + y^2 + 2*y + 2*a - 1 = 0 → ∃ r : ℝ, r > 0) : 
  a < 1 := 
sorry

end circle_represents_real_l97_97304


namespace oranges_per_box_calculation_l97_97901

def total_oranges : ℕ := 2650
def total_boxes : ℕ := 265

theorem oranges_per_box_calculation (h : total_oranges % total_boxes = 0) : total_oranges / total_boxes = 10 :=
by {
  sorry
}

end oranges_per_box_calculation_l97_97901


namespace abs_pi_expression_l97_97667

theorem abs_pi_expression : (|π - |π - 10|| = 10 - 2 * π) := by
  sorry

end abs_pi_expression_l97_97667


namespace find_g_3_8_l97_97736

variable (g : ℝ → ℝ)
variable (x : ℝ)

-- Conditions
axiom g0 : g 0 = 0
axiom monotonicity (x y : ℝ) : 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y
axiom symmetry (x : ℝ) : 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x
axiom scaling (x : ℝ) : 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3

-- Statement to prove
theorem find_g_3_8 : g (3 / 8) = 2 / 9 := 
sorry

end find_g_3_8_l97_97736


namespace problem1_problem2_problem3_problem4_l97_97338

-- Problem 1
theorem problem1 (x : ℝ) : 0.75 * x = (1 / 2) * 12 → x = 8 := 
by 
  intro h
  sorry

-- Problem 2
theorem problem2 (x : ℝ) : (0.7 / x) = (14 / 5) → x = 0.25 := 
by 
  intro h
  sorry

-- Problem 3
theorem problem3 (x : ℝ) : (1 / 6) * x = (2 / 15) * (2 / 3) → x = (8 / 15) := 
by 
  intro h
  sorry

-- Problem 4
theorem problem4 (x : ℝ) : 4.5 * x = 4 * 27 → x = 24 := 
by 
  intro h
  sorry

end problem1_problem2_problem3_problem4_l97_97338


namespace correct_operation_l97_97212

theorem correct_operation (a : ℝ) : 
  (a^3 * a^2 ≠ a^6) ∧ 
  ((-4 * a^3)^2 = 16 * a^6) ∧ 
  (a^6 / a^6 ≠ 0) ∧ 
  ((a - 1)^2 ≠ a^2 - 1) := by
  sorry

end correct_operation_l97_97212


namespace find_x_l97_97596

-- Define the functions δ (delta) and φ (phi)
def delta (x : ℚ) : ℚ := 4 * x + 9
def phi (x : ℚ) : ℚ := 9 * x + 8

-- State the theorem with conditions and question, and assert the answer
theorem find_x :
  (delta ∘ phi) x = 11 → x = -5/6 := by
  intros
  sorry

end find_x_l97_97596


namespace find_xyz_l97_97997

theorem find_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x + 1/y = 5) (h2 : y + 1/z = 2) (h3 : z + 1/x = 3) :
  x * y * z = 10 + 3 * Real.sqrt 11 := by
  sorry

end find_xyz_l97_97997


namespace monotonicity_of_f_solve_inequality_range_of_m_l97_97969

variable {f : ℝ → ℝ}
variable {a b m : ℝ}

-- Given conditions
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def in_interval (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 1
def f_at_one (f : ℝ → ℝ) : Prop := f 1 = 1
def positivity_condition (f : ℝ → ℝ) (a b : ℝ) : Prop := (a + b ≠ 0) → ((f a + f b) / (a + b) > 0)

-- Proof problems
theorem monotonicity_of_f 
  (h_odd : is_odd_function f) 
  (h_interval : ∀ x, in_interval x → in_interval (f x)) 
  (h_f_one : f_at_one f) 
  (h_pos : ∀ a b, in_interval a → in_interval b → positivity_condition f a b) :
  ∀ x1 x2, in_interval x1 → in_interval x2 → x1 < x2 → f x1 < f x2 :=
sorry

theorem solve_inequality 
  (h_odd : is_odd_function f) 
  (h_interval : ∀ x, in_interval x → in_interval (f x)) 
  (h_f_increasing : ∀ x1 x2, in_interval x1 → in_interval x2 → x1 < x2 → f x1 < f x2) 
  (h_f_one : f_at_one f) 
  (h_pos : ∀ a b, in_interval a → in_interval b → positivity_condition f a b) :
  ∀ x, in_interval (x + 1/2) → in_interval (1 / (x - 1)) → f (x + 1/2) < f (1 / (x - 1)) → -3/2 ≤ x ∧ x < -1 :=
sorry

theorem range_of_m 
  (h_odd : is_odd_function f) 
  (h_interval : ∀ x, in_interval x → in_interval (f x)) 
  (h_f_one : f_at_one f) 
  (h_pos : ∀ a b, in_interval a → in_interval b → positivity_condition f a b) 
  (h_f_increasing : ∀ x1 x2, in_interval x1 → in_interval x2 → x1 < x2 → f x1 < f x2) :
  (∀ a, in_interval a → f a ≤ m^2 - 2 * a * m + 1) → (m = 0 ∨ m ≤ -2 ∨ m ≥ 2) :=
sorry

end monotonicity_of_f_solve_inequality_range_of_m_l97_97969


namespace total_lives_l97_97071

theorem total_lives (initial_friends : ℕ) (initial_lives_per_friend : ℕ) (additional_players : ℕ) (lives_per_new_player : ℕ) :
  initial_friends = 7 →
  initial_lives_per_friend = 7 →
  additional_players = 2 →
  lives_per_new_player = 7 →
  (initial_friends * initial_lives_per_friend + additional_players * lives_per_new_player) = 63 :=
by
  intros
  sorry

end total_lives_l97_97071


namespace functional_eq_l97_97287

noncomputable def f (x : ℝ) : ℝ := sorry 

theorem functional_eq {f : ℝ → ℝ} (h1 : ∀ x, x * (f (x + 1) - f x) = f x) (h2 : ∀ x y, |f x - f y| ≤ |x - y|) :
  ∃ k : ℝ, ∀ x > 0, f x = k * x :=
sorry

end functional_eq_l97_97287


namespace isosceles_triangle_legs_length_l97_97582

-- Define the given conditions in Lean
def perimeter (L B: ℕ) : ℕ := 2 * L + B
def base_length : ℕ := 8
def given_perimeter : ℕ := 20

-- State the theorem to be proven
theorem isosceles_triangle_legs_length :
  ∃ (L : ℕ), perimeter L base_length = given_perimeter ∧ L = 6 :=
by
  sorry

end isosceles_triangle_legs_length_l97_97582


namespace total_songs_isabel_bought_l97_97755

theorem total_songs_isabel_bought
  (country_albums pop_albums : ℕ)
  (songs_per_album : ℕ)
  (h1 : country_albums = 6)
  (h2 : pop_albums = 2)
  (h3 : songs_per_album = 9) : 
  (country_albums + pop_albums) * songs_per_album = 72 :=
by
  -- We provide only the statement, no proof as per the instruction
  sorry

end total_songs_isabel_bought_l97_97755


namespace ellipse_line_intersection_l97_97137

-- Definitions of the conditions in the Lean 4 language
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 2) = 1

def midpoint_eq (x1 y1 x2 y2 : ℝ) : Prop := (x1 + x2 = 1) ∧ (y1 + y2 = -2)

-- The problem statement
theorem ellipse_line_intersection :
  (∃ (l : ℝ → ℝ → Prop),
  (∀ x1 y1 x2 y2 : ℝ, ellipse_eq x1 y1 → ellipse_eq x2 y2 → midpoint_eq x1 y1 x2 y2 →
     l x1 y1 ∧ l x2 y2) ∧
  (∀ x y : ℝ, l x y → (x - 4 * y - 9 / 2 = 0))) :=
sorry

end ellipse_line_intersection_l97_97137


namespace max_term_in_sequence_l97_97158

theorem max_term_in_sequence (a : ℕ → ℝ)
  (h : ∀ n, a n = (n+1) * (7/8)^n) :
  (∀ n, a n ≤ a 6 ∨ a n ≤ a 7) ∧ (a 6 = max (a 6) (a 7)) ∧ (a 7 = max (a 6) (a 7)) :=
sorry

end max_term_in_sequence_l97_97158


namespace vanya_number_l97_97205

def S (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem vanya_number:
  (2014 + S 2014 = 2021) ∧ (1996 + S 1996 = 2021) := by
  sorry

end vanya_number_l97_97205


namespace solve_variable_expression_l97_97680

variable {x y : ℕ}

theorem solve_variable_expression
  (h1 : x / (2 * y) = 3 / 2)
  (h2 : (7 * x + 5 * y) / (x - 2 * y) = 26) :
  x = 3 * y :=
sorry

end solve_variable_expression_l97_97680


namespace larry_wins_probability_l97_97166

noncomputable def probability (n : ℕ) : ℝ :=
  if n % 2 = 1 then (1/2)^(n) else 0

noncomputable def inf_geometric_sum (a : ℝ) (r : ℝ) : ℝ :=
  a / (1 - r)

theorem larry_wins_probability :
  inf_geometric_sum (1/2) (1/4) = 2/3 :=
by
  sorry

end larry_wins_probability_l97_97166


namespace infinite_sum_l97_97107

theorem infinite_sum:
  ∑ k in (filter (λ n, n ≥ 1) (range (n + 1))) (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 2 :=
sorry

end infinite_sum_l97_97107


namespace line_intersects_ellipse_slopes_l97_97776

theorem line_intersects_ellipse_slopes (m : ℝ) :
  (∃ x : ℝ, ∃ y : ℝ, y = m * x - 3 ∧ 4 * x^2 + 25 * y^2 = 100) ↔ 
  m ∈ Set.Iic (-Real.sqrt (1/5)) ∨ m ∈ Set.Ici (Real.sqrt (1/5)) :=
by
  sorry

end line_intersects_ellipse_slopes_l97_97776


namespace group_division_l97_97021

theorem group_division (men women : ℕ) : 
  men = 4 → women = 5 →
  (∃ g1 g2 g3 : set (fin $ men + women), 
    g1.card = 3 ∧ g2.card = 3 ∧ g3.card = 3 ∧ 
    (∀ g, g ∈ [g1, g2, g3] → (∃ m w : ℕ, 1 ≤ m ∧ 1 ≤ w ∧ 
      finset.card (finset.filter (λ x, x < men) g) = m ∧ 
      finset.card (finset.filter (λ x, x ≥ men) g) = w)) 
    ∧ finset.disjoint g1 g2 ∧ finset.disjoint g2 g3 ∧ finset.disjoint g3 g1 
    ∧ g1 ∪ g2 ∪ g3 = finset.univ (fin $ men + women)) → 
  finset.card (finset.powerset' 3 (finset.univ (fin $ men + women))) / 2 = 180 :=
begin
  intros hmen hwomen,
  sorry
end

end group_division_l97_97021


namespace tangent_points_l97_97380

noncomputable def f (x : ℝ) : ℝ := x^3 + 1
def P : ℝ × ℝ := (-2, 1)
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2

theorem tangent_points (x0 : ℝ) (y0 : ℝ) (hP : P = (-2, 1)) (hf : y0 = f x0) :
  (3 * x0^2 = (y0 - 1) / (x0 + 2)) → (x0 = 0 ∨ x0 = -3) :=
by
  sorry

end tangent_points_l97_97380


namespace find_n_l97_97960

theorem find_n (n : ℤ) : 43^2 = 1849 ∧ 44^2 = 1936 ∧ 45^2 = 2025 ∧ 46^2 = 2116 ∧ n < Real.sqrt 2023 ∧ Real.sqrt 2023 < n + 1 → n = 44 :=
by
  sorry

end find_n_l97_97960


namespace articles_in_selling_price_l97_97151

theorem articles_in_selling_price (C : ℝ) (N : ℕ) 
  (h1 : 50 * C = N * (1.25 * C)) 
  (h2 : 0.25 * C = 25 / 100 * C) :
  N = 40 :=
by
  sorry

end articles_in_selling_price_l97_97151


namespace transform_expression_l97_97051

variable {a : ℝ}

theorem transform_expression (h : a - 1 < 0) : 
  (a - 1) * Real.sqrt (-1 / (a - 1)) = -Real.sqrt (1 - a) :=
by
  sorry

end transform_expression_l97_97051


namespace solve_for_y_l97_97374

theorem solve_for_y (x y : ℤ) (h1 : x + y = 290) (h2 : x - y = 200) : y = 45 := 
by 
  sorry

end solve_for_y_l97_97374


namespace tank_capacity_l97_97431

theorem tank_capacity :
  ∀ (T : ℚ), (3 / 4) * T + 4 = (7 / 8) * T → T = 32 :=
by
  intros T h
  sorry

end tank_capacity_l97_97431


namespace symmetric_sum_eq_two_l97_97970

-- Definitions and conditions
def symmetric (P Q : ℝ × ℝ) : Prop :=
  P.1 = -Q.1 ∧ P.2 = -Q.2

def P : ℝ × ℝ := (sorry, 1)
def Q : ℝ × ℝ := (-3, sorry)

-- Problem statement
theorem symmetric_sum_eq_two (h : symmetric P Q) : P.1 + Q.2 = 2 :=
by
  -- Proof omitted
  sorry

end symmetric_sum_eq_two_l97_97970


namespace average_cost_of_testing_l97_97989

theorem average_cost_of_testing (total_machines : Nat) (faulty_machines : Nat) (cost_per_test : Nat) 
  (h_total : total_machines = 5) (h_faulty : faulty_machines = 2) (h_cost : cost_per_test = 1000) :
  (2000 * (2 / 5 * 1 / 4) + 3000 * (2 / 5 * 3 / 4 * 1 / 3 + 3 / 5 * 2 / 4 * 1 / 3 + 3 / 5 * 2 / 4 * 1 / 3) + 
  4000 * (1 - (2 / 5 * 1 / 4) - (2 / 5 * 3 / 4 * 1 / 3 + 3 / 5 * 2 / 4 * 1 / 3 + 3 / 5 * 2 / 4 * 1 / 3))) = 3500 :=
  by
  sorry

end average_cost_of_testing_l97_97989


namespace evaluate_expression_l97_97674

theorem evaluate_expression (x : ℕ) (h : x = 3) : (x^x)^(x^x) = 27^27 :=
by
  sorry

end evaluate_expression_l97_97674


namespace polygon_sides_l97_97069

theorem polygon_sides (n : ℕ) (h1 : (n - 2) * 180 = 3 * 360) : n = 8 :=
by sorry

end polygon_sides_l97_97069


namespace problem_one_problem_two_l97_97567

theorem problem_one (α : ℝ) (h : Real.tan α = 2) : (3 * Real.sin α - 2 * Real.cos α) / (Real.sin α - Real.cos α) = 4 :=
by
  sorry

theorem problem_two (α : ℝ) (h : Real.tan α = 2) (h_quadrant : α > π ∧ α < 3 * π / 2) : Real.cos α = - (Real.sqrt 5 / 5) :=
by
  sorry

end problem_one_problem_two_l97_97567


namespace line_through_midpoints_parallel_and_bisects_perimeter_l97_97445

/-- In triangle ABC, AC and BC sides are tangent to circles at points K and L respectively. 
Prove that the line passing through the midpoints of segments KL and AB is
parallel to the angle bisector of ∠ ACB and bisects the perimeter of the triangle. -/
theorem line_through_midpoints_parallel_and_bisects_perimeter
  {A B C K L : Point}
  (h_tangent_A : ∃ r, circle A r ∩ line_segment A C = {K})
  (h_tangent_B : ∃ r, circle B r ∩ line_segment B C = {L})
  (h_triangle : Triangle A B C) :
  ∃ M N : Point,
  midpoint K L = M ∧
  midpoint A B = N ∧
  (is_parallel (line_through M N) (angle_bisector A C B)) ∧
  (perimeter_bisected (line_through M N) (Triangle A B C)) :=
sorry

end line_through_midpoints_parallel_and_bisects_perimeter_l97_97445


namespace area_of_fourth_rectangle_l97_97903

variable (x y z w : ℝ)
variable (Area_EFGH Area_EIKJ Area_KLMN Perimeter : ℝ)

def conditions :=
  (Area_EFGH = x * y ∧ Area_EFGH = 20 ∧
   Area_EIKJ = x * w ∧ Area_EIKJ = 25 ∧
   Area_KLMN = z * w ∧ Area_KLMN = 15 ∧
   Perimeter = 2 * (x + z + y + w) ∧ Perimeter = 40)

theorem area_of_fourth_rectangle (h : conditions x y z w Area_EFGH Area_EIKJ Area_KLMN Perimeter) :
  (y * w = 340) :=
by
  sorry

end area_of_fourth_rectangle_l97_97903


namespace agent_007_encryption_l97_97096

theorem agent_007_encryption : ∃ (m n : ℕ), (0.07 : ℝ) = (1 / m : ℝ) + (1 / n : ℝ) := 
sorry

end agent_007_encryption_l97_97096


namespace find_p_q_r_sum_l97_97714

noncomputable def Q (p q r : ℝ) (v : ℂ) : Polynomial ℂ :=
  (Polynomial.C v + 2 * Polynomial.C Complex.I).comp Polynomial.X *
  (Polynomial.C v + 8 * Polynomial.C Complex.I).comp Polynomial.X *
  (Polynomial.C (3 * v - 5)).comp Polynomial.X

theorem find_p_q_r_sum (p q r : ℝ) (v : ℂ)
  (h_roots : ∃ v : ℂ, Polynomial.roots (Q p q r v) = {v + 2 * Complex.I, v + 8 * Complex.I, 3 * v - 5}) :
  (p + q + r) = -82 :=
by
  sorry

end find_p_q_r_sum_l97_97714


namespace cos_of_angle_in_third_quadrant_l97_97581

theorem cos_of_angle_in_third_quadrant (A : ℝ) (hA : π < A ∧ A < 3 * π / 2) (h_sin : Real.sin A = -1 / 3) :
  Real.cos A = -2 * Real.sqrt 2 / 3 :=
by
  sorry

end cos_of_angle_in_third_quadrant_l97_97581


namespace floor_of_neg_seven_fourths_l97_97263

theorem floor_of_neg_seven_fourths : 
  Int.floor (-7 / 4 : ℚ) = -2 := 
by sorry

end floor_of_neg_seven_fourths_l97_97263


namespace Lloyd_hourly_rate_is_3_5_l97_97461

/-!
Lloyd normally works 7.5 hours per day and earns a certain amount per hour.
For each hour he works in excess of 7.5 hours on a given day, he is paid 1.5 times his regular rate.
If Lloyd works 10.5 hours on a given day, he earns $42 for that day.
-/

variable (Lloyd_hourly_rate : ℝ)  -- regular hourly rate

def Lloyd_daily_earnings (total_hours : ℝ) (regular_hours : ℝ) (hourly_rate : ℝ) : ℝ :=
  let excess_hours := total_hours - regular_hours
  let excess_earnings := excess_hours * (1.5 * hourly_rate)
  let regular_earnings := regular_hours * hourly_rate
  excess_earnings + regular_earnings

-- Given conditions
axiom H1 : 7.5 = 7.5
axiom H2 : ∀ R : ℝ, Lloyd_hourly_rate = R
axiom H3 : ∀ R : ℝ, ∀ excess_hours : ℝ, Lloyd_hourly_rate + excess_hours = 1.5 * R
axiom H4 : Lloyd_daily_earnings 10.5 7.5 Lloyd_hourly_rate = 42

-- Prove Lloyd earns $3.50 per hour.
theorem Lloyd_hourly_rate_is_3_5 : Lloyd_hourly_rate = 3.5 :=
sorry

end Lloyd_hourly_rate_is_3_5_l97_97461


namespace find_x_minus_y_l97_97811

theorem find_x_minus_y (x y : ℝ) (hx : |x| = 4) (hy : |y| = 2) (hxy : x * y < 0) : x - y = 6 ∨ x - y = -6 :=
by sorry

end find_x_minus_y_l97_97811


namespace expression_evaluation_l97_97217

theorem expression_evaluation (p q : ℝ) (h : p / q = 4 / 5) : (25 / 7 + (2 * q - p) / (2 * q + p)) = 4 :=
by {
  sorry
}

end expression_evaluation_l97_97217


namespace solution_set_g_lt_6_range_of_values_a_l97_97898

-- Definitions
def f (a x : ℝ) : ℝ := 3 * |x - a| + |3 * x + 1|
def g (x : ℝ) : ℝ := |4 * x - 1| - |x + 2|

-- First part: solution set for g(x) < 6
theorem solution_set_g_lt_6 :
  {x : ℝ | g x < 6} = {x : ℝ | -7/5 < x ∧ x < 3} :=
sorry

-- Second part: range of values for a such that f(x1) and g(x2) are opposite numbers
theorem range_of_values_a (a : ℝ) :
  (∃ x1 x2 : ℝ, f a x1 = -g x2) → -13/12 ≤ a ∧ a ≤ 5/12 :=
sorry

end solution_set_g_lt_6_range_of_values_a_l97_97898


namespace largest_sphere_radius_in_prism_l97_97560

noncomputable def largestInscribedSphereRadius (m : ℝ) : ℝ :=
  (Real.sqrt 6 - Real.sqrt 2) / 4 * m

theorem largest_sphere_radius_in_prism (m : ℝ) (h : 0 < m) :
  ∃ r, r = largestInscribedSphereRadius m ∧ r < m/2 :=
sorry

end largest_sphere_radius_in_prism_l97_97560


namespace crayon_count_after_actions_l97_97874

theorem crayon_count_after_actions (initial_crayons : ℕ) (kiley_fraction : ℚ) (joe_fraction : ℚ) :
  initial_crayons = 48 → kiley_fraction = 1 / 4 → joe_fraction = 1 / 2 → 
  let crayons_after_kiley := initial_crayons - (kiley_fraction * initial_crayons).to_nat;
      crayons_after_joe := crayons_after_kiley - (joe_fraction * crayons_after_kiley).to_nat
  in crayons_after_joe = 18 :=
by 
  intros h1 h2 h3;
  sorry

end crayon_count_after_actions_l97_97874


namespace grouping_count_l97_97031

theorem grouping_count (men women : ℕ) 
  (h_men : men = 4) (h_women : women = 5)
  (at_least_one_man_woman : ∀ (g1 g2 g3 : Finset (Fin 9)), 
    g1.card = 3 → g2.card = 3 → g3.card = 3 → g1 ∩ g2 = ∅ → g2 ∩ g3 = ∅ → g3 ∩ g1 = ∅ → 
    (g1 ∩ univ.filter (· < 4)).nonempty ∧ (g1 ∩ univ.filter (· ≥ 4)).nonempty ∧
    (g2 ∩ univ.filter (· < 4)).nonempty ∧ (g2 ∩ univ.filter (· ≥ 4)).nonempty ∧
    (g3 ∩ univ.filter (· < 4)).nonempty ∧ (g3 ∩ univ.filter (· ≥ 4)).nonempty) :
  (choose 4 1 * choose 5 2 * choose 3 1 * choose 3 2) / 2! = 180 :=
sorry

end grouping_count_l97_97031


namespace number_base_addition_l97_97167

theorem number_base_addition (A B : ℕ) (h1: A = 2 * B) (h2: 2 * B^2 + 2 * B + 4 + 10 * B + 5 = (3 * B)^2 + 3 * (3 * B) + 4) : 
  A + B = 9 := 
by 
  sorry

end number_base_addition_l97_97167


namespace infinite_sum_l97_97106

theorem infinite_sum:
  ∑ k in (filter (λ n, n ≥ 1) (range (n + 1))) (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 2 :=
sorry

end infinite_sum_l97_97106


namespace kimberly_initial_skittles_l97_97448

theorem kimberly_initial_skittles (total new initial : ℕ) (h1 : total = 12) (h2 : new = 7) (h3 : total = initial + new) : initial = 5 :=
by {
  -- Using the given conditions to form the proof
  sorry
}

end kimberly_initial_skittles_l97_97448


namespace ratio_difference_l97_97377

theorem ratio_difference (x : ℕ) (h_largest : 7 * x = 70) : 70 - 3 * x = 40 := by
  sorry

end ratio_difference_l97_97377


namespace sockPairsCount_l97_97585

noncomputable def countSockPairs : ℕ :=
  let whitePairs := Nat.choose 6 2 -- 15
  let brownPairs := Nat.choose 7 2 -- 21
  let bluePairs := Nat.choose 3 2 -- 3
  let oneRedOneWhite := 4 * 6 -- 24
  let oneRedOneBrown := 4 * 7 -- 28
  let oneRedOneBlue := 4 * 3 -- 12
  let bothRed := Nat.choose 4 2 -- 6
  whitePairs + brownPairs + bluePairs + oneRedOneWhite + oneRedOneBrown + oneRedOneBlue + bothRed

theorem sockPairsCount : countSockPairs = 109 := by
  sorry

end sockPairsCount_l97_97585


namespace relationship_between_a_and_b_l97_97968

-- Definitions for the conditions
variables {a b : ℝ}

-- Main theorem statement
theorem relationship_between_a_and_b (h1 : |Real.log (1 / 4) / Real.log a| = Real.log (1 / 4) / Real.log a)
  (h2 : |Real.log a / Real.log b| = -Real.log a / Real.log b) :
  0 < a ∧ a < 1 ∧ 1 < b :=
by
  sorry

end relationship_between_a_and_b_l97_97968


namespace find_number_l97_97899

theorem find_number (x : ℕ) (h : x / 46 - 27 = 46) : x = 3358 :=
by
  sorry

end find_number_l97_97899


namespace moles_of_NH4Cl_l97_97544

-- Define what is meant by "mole" and the substances NH3, HCl, and NH4Cl
def NH3 : Type := ℕ -- Use ℕ to represent moles
def HCl : Type := ℕ
def NH4Cl : Type := ℕ

-- Define the stoichiometry of the reaction
def reaction (n_NH3 n_HCl : ℕ) : ℕ :=
n_NH3 + n_HCl

-- Lean 4 statement: given 1 mole of NH3 and 1 mole of HCl, prove the reaction produces 1 mole of NH4Cl
theorem moles_of_NH4Cl (n_NH3 n_HCl : ℕ) (h1 : n_NH3 = 1) (h2 : n_HCl = 1) : 
  reaction n_NH3 n_HCl = 1 :=
by
  sorry

end moles_of_NH4Cl_l97_97544


namespace box_contains_1600_calories_l97_97636

theorem box_contains_1600_calories :
  let cookies_per_bag := 20
  let bags_per_box := 4
  let calories_per_cookie := 20
  let total_cookies := cookies_per_bag * bags_per_box
  let total_calories := total_cookies * calories_per_cookie
  total_calories = 1600 :=
by
  let cookies_per_bag := 20
  let bags_per_box := 4
  let calories_per_cookie := 20
  let total_cookies := cookies_per_bag * bags_per_box
  let total_calories := total_cookies * calories_per_cookie
  show total_calories = 1600
  sorry

end box_contains_1600_calories_l97_97636


namespace cubs_more_home_runs_l97_97525

noncomputable def cubs_home_runs := 2 + 1 + 2
noncomputable def cardinals_home_runs := 1 + 1

theorem cubs_more_home_runs :
  cubs_home_runs - cardinals_home_runs = 3 :=
by
  -- Proof would go here, but we are using sorry to skip it
  sorry

end cubs_more_home_runs_l97_97525


namespace shaded_area_of_larger_circle_l97_97703

theorem shaded_area_of_larger_circle (R r : ℝ) (A_larger A_smaller : ℝ)
  (hR : R = 9)
  (hr : r = 4.5)
  (hA_larger : A_larger = Real.pi * R^2)
  (hA_smaller : A_smaller = 3 * Real.pi * r^2) :
  A_larger - A_smaller = 20.25 * Real.pi := by
  sorry

end shaded_area_of_larger_circle_l97_97703


namespace inverse_of_203_mod_301_l97_97934

theorem inverse_of_203_mod_301 : ∃ (a : ℤ), 0 ≤ a ∧ a ≤ 300 ∧ (203 * a ≡ 1 [MOD 301]) :=
by
  use 238
  split
  by norm_num
  split
  by norm_num
  by norm_num,
  exact ⟨by norm_num, by norm_num⟩, sorry
 
end inverse_of_203_mod_301_l97_97934


namespace calc_expr_l97_97507

theorem calc_expr : 
  (-1: ℝ)^4 - 2 * Real.tan (Real.pi / 3) + (Real.sqrt 3 - Real.sqrt 2)^0 + Real.sqrt 12 = 2 := 
by
  sorry

end calc_expr_l97_97507


namespace remainder_53_pow_10_div_8_l97_97609

theorem remainder_53_pow_10_div_8 : (53^10) % 8 = 1 := 
by sorry

end remainder_53_pow_10_div_8_l97_97609


namespace find_a_l97_97132

variable (A B : Set ℤ) (a : ℤ)
variable (elem1 : 0 ∈ A) (elem2 : 1 ∈ A)
variable (elem3 : -1 ∈ B) (elem4 : 0 ∈ B) (elem5 : a + 3 ∈ B)

theorem find_a (h : A ⊆ B) : a = -2 := sorry

end find_a_l97_97132


namespace sequence_geometric_l97_97577

theorem sequence_geometric (a : ℕ → ℕ) (n : ℕ) (hn : 0 < n):
  (a 1 = 1) →
  (∀ n, 0 < n → a (n + 1) = 2 * a n) →
  a n = 2^(n-1) :=
by
  intros
  sorry

end sequence_geometric_l97_97577


namespace calc_expression_l97_97508

theorem calc_expression : 112 * 5^4 * 3^2 = 630000 := by
  sorry

end calc_expression_l97_97508


namespace solve_for_x_minus_y_l97_97953

theorem solve_for_x_minus_y (x y : ℝ) (h1 : 4 = 0.25 * x) (h2 : 4 = 0.50 * y) : x - y = 8 :=
by
  sorry

end solve_for_x_minus_y_l97_97953


namespace altitudes_not_form_triangle_l97_97079

theorem altitudes_not_form_triangle (h₁ h₂ h₃ : ℝ) :
  ¬(h₁ = 5 ∧ h₂ = 12 ∧ h₃ = 13 ∧ ∃ a b c : ℝ, a * h₁ = b * h₂ ∧ b * h₂ = c * h₃ ∧
    a < b + c ∧ b < a + c ∧ c < a + b) :=
by sorry

end altitudes_not_form_triangle_l97_97079


namespace meetings_percentage_l97_97462

def total_minutes_in_day (hours: ℕ): ℕ := hours * 60
def first_meeting_duration: ℕ := 60
def second_meeting_duration (first_meeting_duration: ℕ): ℕ := 3 * first_meeting_duration
def total_meeting_duration (first_meeting_duration: ℕ) (second_meeting_duration: ℕ): ℕ := first_meeting_duration + second_meeting_duration
def percentage_of_workday_spent_in_meetings (total_meeting_duration: ℕ) (total_minutes_in_day: ℕ): ℚ := (total_meeting_duration / total_minutes_in_day) * 100

theorem meetings_percentage (hours: ℕ) (first_meeting_duration: ℕ) (second_meeting_duration: ℕ) (total_meeting_duration: ℕ) (total_minutes_in_day: ℕ) 
(h1: total_minutes_in_day = 600) 
(h2: first_meeting_duration = 60) 
(h3: second_meeting_duration = 180) 
(h4: total_meeting_duration = 240):
percentage_of_workday_spent_in_meetings total_meeting_duration total_minutes_in_day = 40 := by
  sorry

end meetings_percentage_l97_97462


namespace sum_of_consecutive_integers_exists_l97_97754

theorem sum_of_consecutive_integers_exists : 
  ∃ k : ℕ, 150 * k + 11325 = 5827604250 :=
by
  sorry

end sum_of_consecutive_integers_exists_l97_97754


namespace angle_C_in_triangle_l97_97985

theorem angle_C_in_triangle (A B C : ℝ) (h : A + B = 110) (ht : A + B + C = 180) : C = 70 :=
by
  -- proof steps go here
  sorry

end angle_C_in_triangle_l97_97985


namespace find_orange_shells_l97_97730

theorem find_orange_shells :
  ∀ (total purple pink yellow blue : ℕ),
    total = 65 → purple = 13 → pink = 8 → yellow = 18 → blue = 12 →
    total - (purple + pink + yellow + blue) = 14 :=
by
  intros total purple pink yellow blue h_total h_purple h_pink h_yellow h_blue
  have h := h_total.symm
  rw [h_purple, h_pink, h_yellow, h_blue]
  simp only [Nat.add_assoc, Nat.add_comm, Nat.add_sub_cancel]
  sorry

end find_orange_shells_l97_97730


namespace minimum_value_l97_97689

noncomputable def log_a (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem minimum_value (a m n : ℝ)
    (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1)
    (h_a_on_graph : ∀ x, log_a a (x + 3) - 1 = 0 → x = -2)
    (h_on_line : 2 * m + n = 2)
    (h_mn_pos : m * n > 0) :
    (1 / m) + (2 / n) = 4 :=
by
  sorry

end minimum_value_l97_97689


namespace percentage_of_men_is_55_l97_97584

-- Define the percentage of men among all employees
def percent_of_men (M : ℝ) := M

-- Define the percentage of women among all employees
def percent_of_women (M : ℝ) := 1 - M

-- Define the contribution to picnic attendance by men
def attendance_by_men (M : ℝ) := 0.20 * M

-- Define the contribution to picnic attendance by women
def attendance_by_women (M : ℝ) := 0.40 * (percent_of_women M)

-- Define the total attendance
def total_attendance (M : ℝ) := attendance_by_men M + attendance_by_women M

theorem percentage_of_men_is_55 : ∀ M : ℝ, total_attendance M = 0.29 → M = 0.55 :=
by
  intro M
  intro h
  sorry

end percentage_of_men_is_55_l97_97584


namespace lcm_18_35_l97_97802

theorem lcm_18_35 : Nat.lcm 18 35 = 630 := 
by 
  sorry

end lcm_18_35_l97_97802


namespace mother_gave_80_cents_l97_97460

theorem mother_gave_80_cents (father_uncles_gift : Nat) (spent_on_candy current_amount : Nat) (gift_from_father gift_from_uncle add_gift_from_uncle : Nat) (x : Nat) :
  father_uncles_gift = gift_from_father + gift_from_uncle ∧
  father_uncles_gift = 110 ∧
  spent_on_candy = 50 ∧
  current_amount = 140 ∧
  gift_from_father = 40 ∧
  gift_from_uncle = 70 ∧
  add_gift_from_uncle = 70 ∧
  x = current_amount + spent_on_candy - father_uncles_gift ∧
  x = 190 - 110 ∨
  x = 80 :=
  sorry

end mother_gave_80_cents_l97_97460


namespace intersection_M_complement_N_eq_l97_97425

open Set

noncomputable def U : Set ℝ := univ
noncomputable def M : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
noncomputable def N : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}
noncomputable def complement_N : Set ℝ := {y | y < 1}

theorem intersection_M_complement_N_eq : M ∩ complement_N = {x | -1 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_M_complement_N_eq_l97_97425


namespace number_of_pairs_l97_97998

theorem number_of_pairs (n : Nat) : 
  (∃ n, n > 2 ∧ ∀ x y : ℝ, (5 * y - 3 * x = 15 ∧ x^2 + y^2 ≤ 16) → True) :=
sorry

end number_of_pairs_l97_97998


namespace martin_spends_30_dollars_on_berries_l97_97463

def cost_per_package : ℝ := 2.0
def cups_per_package : ℝ := 1.0
def cups_per_day : ℝ := 0.5
def days : ℝ := 30

theorem martin_spends_30_dollars_on_berries :
  (days / (cups_per_package / cups_per_day)) * cost_per_package = 30 :=
by
  sorry

end martin_spends_30_dollars_on_berries_l97_97463


namespace least_cost_grass_seed_l97_97890

variable (cost_5_pound_bag : ℕ) [Fact (cost_5_pound_bag = 1380)]
variable (cost_10_pound_bag : ℕ) [Fact (cost_10_pound_bag = 2043)]
variable (cost_25_pound_bag : ℕ) [Fact (cost_25_pound_bag = 3225)]
variable (min_weight : ℕ) [Fact (min_weight = 65)]
variable (max_weight : ℕ) [Fact (max_weight = 80)]

theorem least_cost_grass_seed :
  ∃ (n5 n10 n25 : ℕ),
    n5 * 5 + n10 * 10 + n25 * 25 ≥ min_weight ∧
    n5 * 5 + n10 * 10 + n25 * 25 ≤ max_weight ∧
    n5 * cost_5_pound_bag + n10 * cost_10_pound_bag + n25 * cost_25_pound_bag = 9675 :=
  sorry

end least_cost_grass_seed_l97_97890


namespace crayons_count_l97_97955

-- Definitions based on the conditions given in the problem
def total_crayons : Nat := 96
def benny_crayons : Nat := 12
def fred_crayons : Nat := 2 * benny_crayons
def jason_crayons (sarah_crayons : Nat) : Nat := 3 * sarah_crayons

-- Stating the proof goal
theorem crayons_count (sarah_crayons : Nat) :
  fred_crayons + benny_crayons + jason_crayons sarah_crayons + sarah_crayons = total_crayons →
  sarah_crayons = 15 ∧
  fred_crayons = 24 ∧
  jason_crayons sarah_crayons = 45 ∧
  benny_crayons = 12 :=
by
  sorry

end crayons_count_l97_97955


namespace pipe_filling_time_l97_97178

theorem pipe_filling_time (T : ℝ) (h1 : T > 0) (h2 : 1/(3:ℝ) = 1/T - 1/(6:ℝ)) : T = 2 := 
by sorry

end pipe_filling_time_l97_97178


namespace determinant_expression_l97_97039

open Matrix

variables {R : Type*} [CommRing R] (a b c p q : R)

-- Given conditions
def cubic_polynomial (x : R) := x^3 + p * x + q

def roots_condition : Prop := 
  ∀ (x : R), cubic_polynomial p q x = 0 ↔ (x = a ∨ x = b ∨ x = c)

-- Define matrix and its determinant
def det_matrix : Matrix (Fin 3) (Fin 3) R :=
  !![2 + a^2, 1, 1;
     1, 2 + b^2, 1;
     1, 1, 2 + c^2]

def det_val : R := det_matrix.det

theorem determinant_expression (h : roots_condition a b c p q) :
  det_val a b c = 2 * p^2 - 4 * q + q^2 :=
sorry

end determinant_expression_l97_97039


namespace prove_equal_values_l97_97916

theorem prove_equal_values :
  (-2: ℝ)^3 = -(2: ℝ)^3 :=
by sorry

end prove_equal_values_l97_97916


namespace product_base9_l97_97655

open Nat

noncomputable def base9_product (a b : ℕ) : ℕ := 
  let a_base10 := 3*9^2 + 6*9^1 + 2*9^0
  let b_base10 := 7
  let product_base10 := a_base10 * b_base10
  -- converting product_base10 from base 10 to base 9
  2 * 9^3 + 8 * 9^2 + 7 * 9^1 + 5 * 9^0 -- which simplifies to 2875 in base 9

theorem product_base9: base9_product 362 7 = 2875 :=
by
  -- Here should be the proof or a computational check
  sorry

end product_base9_l97_97655


namespace floor_neg_seven_over_four_l97_97252

theorem floor_neg_seven_over_four : floor (-7 / 4) = -2 :=
by
  sorry

end floor_neg_seven_over_four_l97_97252


namespace complement_of_M_l97_97451

-- Definitions:
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | -1 < x ∧ x ≤ 2}

-- Assertion:
theorem complement_of_M :
  (U \ M) = {x | x ≤ -1} ∪ {x | 2 < x} :=
by sorry

end complement_of_M_l97_97451


namespace smallest_possible_gcd_l97_97977

noncomputable def smallestGCD (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : Nat.gcd a b = 9) : ℕ :=
  Nat.gcd (12 * a) (18 * b)

theorem smallest_possible_gcd (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : Nat.gcd a b = 9) : 
  smallestGCD a b h1 h2 h3 = 54 :=
sorry

end smallest_possible_gcd_l97_97977


namespace abs_twice_sub_pi_l97_97665

theorem abs_twice_sub_pi (h : Real.pi < 10) : abs (Real.pi - abs (Real.pi - 10)) = 10 - 2 * Real.pi :=
by
  sorry

end abs_twice_sub_pi_l97_97665


namespace polynomial_roots_sum_l97_97045

noncomputable def roots (p : Polynomial ℚ) : Set ℚ := {r | p.eval r = 0}

theorem polynomial_roots_sum :
  ∀ a b c : ℚ, (a ∈ roots (Polynomial.C 3 - Polynomial.X * Polynomial.C 7 + Polynomial.X^2 * Polynomial.C 8 - Polynomial.X^3)) →
  (b ∈ roots (Polynomial.C 3 - Polynomial.X * Polynomial.C 7 + Polynomial.X^2 * Polynomial.C 8 - Polynomial.X^3)) →
  (c ∈ roots (Polynomial.C 3 - Polynomial.X * Polynomial.C 7 + Polynomial.X^2 * Polynomial.C 8 - Polynomial.X^3)) →
  a ≠ b → b ≠ c → a ≠ c →
  (a + b + c = 8) →
  (a * b + a * c + b * c = 7) →
  (a * b * c = -3) →
  (a / (b * c + 1) + b / (a * c + 1) + c / (a * b + 1) = 17 / 2) := by
    intros a b c ha hb hc hab habc hac sum_nums sum_prods prod_roots
    sorry

#check polynomial_roots_sum

end polynomial_roots_sum_l97_97045


namespace regular_hexagon_area_l97_97231

noncomputable def dist (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem regular_hexagon_area 
  (A C : ℝ × ℝ)
  (hA : A = (0, 0))
  (hC : C = (8, 2))
  (h_eq_side_length : ∀ x y : ℝ × ℝ, dist A.1 A.2 C.1 C.2 = dist x.1 x.2 y.1 y.2) :
  hexagon_area = 34 * Real.sqrt 3 :=
by
  -- sorry indicates the proof is omitted
  sorry

end regular_hexagon_area_l97_97231


namespace fifteenth_term_is_44_l97_97216

-- Define the conditions
def first_term : ℕ := 2
def common_difference : ℕ := 3
def term_number : ℕ := 15

-- Define the formula for the nth term of an arithmetic progression
def nth_term (a d n : ℕ) : ℕ := a + (n - 1) * d

-- Prove that the 15th term is 44
theorem fifteenth_term_is_44 : nth_term first_term common_difference term_number = 44 :=
by
  unfold nth_term first_term common_difference term_number
  sorry

end fifteenth_term_is_44_l97_97216


namespace bread_consumption_snacks_per_day_l97_97488

theorem bread_consumption_snacks_per_day (members : ℕ) (breakfast_slices_per_member : ℕ) (slices_per_loaf : ℕ) (loaves : ℕ) (days : ℕ) (total_slices_breakfast : ℕ) (total_slices_all : ℕ) (snack_slices_per_member_per_day : ℕ) :
  members = 4 →
  breakfast_slices_per_member = 3 →
  slices_per_loaf = 12 →
  loaves = 5 →
  days = 3 →
  total_slices_breakfast = members * breakfast_slices_per_member * days →
  total_slices_all = slices_per_loaf * loaves →
  snack_slices_per_member_per_day = ((total_slices_all - total_slices_breakfast) / members / days) →
  snack_slices_per_member_per_day = 2 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  -- We can insert the proof outline here based on the calculations from the solution steps
  sorry

end bread_consumption_snacks_per_day_l97_97488


namespace mult_base7_correct_l97_97656

def base7_to_base10 (n : ℕ) : ℕ :=
  -- assume conversion from base-7 to base-10 is already defined
  sorry 

def base10_to_base7 (n : ℕ) : ℕ :=
  -- assume conversion from base-10 to base-7 is already defined
  sorry

theorem mult_base7_correct : (base7_to_base10 325) * (base7_to_base10 4) = base7_to_base10 1656 :=
by
  sorry

end mult_base7_correct_l97_97656


namespace lcm_18_35_is_630_l97_97809

def lcm_18_35 : ℕ :=
  Nat.lcm 18 35

theorem lcm_18_35_is_630 : lcm_18_35 = 630 := by
  sorry

end lcm_18_35_is_630_l97_97809


namespace intersection_M_N_l97_97424

noncomputable def M : Set ℝ := {-1, 0, 1}
noncomputable def N : Set ℝ := {x | x * (x - 2) ≤ 0}

theorem intersection_M_N : M ∩ N = {0, 1} :=
by {
  sorry
}

end intersection_M_N_l97_97424


namespace tangent_line_through_points_of_tangency_l97_97413

noncomputable def equation_of_tangent_line (x1 y1 x y : ℝ) : Prop :=
x1 * x + (y1 - 2) * (y - 2) = 4

theorem tangent_line_through_points_of_tangency
  (x1 y1 x2 y2 : ℝ)
  (h1 : equation_of_tangent_line x1 y1 2 (-2))
  (h2 : equation_of_tangent_line x2 y2 2 (-2)) :
  (2 * x1 - 4 * (y1 - 2) = 4) ∧ (2 * x2 - 4 * (y2 - 2) = 4) →
  ∃ a b c, (a = 1) ∧ (b = -2) ∧ (c = 2) ∧ (a * x + b * y + c = 0) :=
by
  sorry

end tangent_line_through_points_of_tangency_l97_97413


namespace ice_cream_sundaes_l97_97239

theorem ice_cream_sundaes (flavors : Finset String) (vanilla : String) (h1 : vanilla ∈ flavors) (h2 : flavors.card = 8) :
  let remaining_flavors := flavors.erase vanilla
  remaining_flavors.card = 7 :=
by
  sorry

end ice_cream_sundaes_l97_97239


namespace area_of_outer_sphere_marked_l97_97912

noncomputable def r : ℝ := 1  -- Radius of the small painted sphere
noncomputable def R_inner : ℝ := 4  -- Radius of the inner concentric sphere
noncomputable def R_outer : ℝ := 6  -- Radius of the outer concentric sphere
noncomputable def A_inner : ℝ := 47  -- Area of the region on the inner sphere

theorem area_of_outer_sphere_marked :
  let A_outer := ((R_outer / R_inner) ^ 2) * A_inner in
  A_outer = 105.75 :=
by
  let A_outer := ((R_outer / R_inner) ^ 2) * A_inner
  sorry

end area_of_outer_sphere_marked_l97_97912


namespace increasing_function_range_l97_97862

theorem increasing_function_range (k : ℝ) :
  (∀ x y : ℝ, x < y → (k + 2) * x + 1 < (k + 2) * y + 1) ↔ k > -2 :=
by
  sorry

end increasing_function_range_l97_97862


namespace grouping_count_l97_97032

theorem grouping_count (men women : ℕ) 
  (h_men : men = 4) (h_women : women = 5)
  (at_least_one_man_woman : ∀ (g1 g2 g3 : Finset (Fin 9)), 
    g1.card = 3 → g2.card = 3 → g3.card = 3 → g1 ∩ g2 = ∅ → g2 ∩ g3 = ∅ → g3 ∩ g1 = ∅ → 
    (g1 ∩ univ.filter (· < 4)).nonempty ∧ (g1 ∩ univ.filter (· ≥ 4)).nonempty ∧
    (g2 ∩ univ.filter (· < 4)).nonempty ∧ (g2 ∩ univ.filter (· ≥ 4)).nonempty ∧
    (g3 ∩ univ.filter (· < 4)).nonempty ∧ (g3 ∩ univ.filter (· ≥ 4)).nonempty) :
  (choose 4 1 * choose 5 2 * choose 3 1 * choose 3 2) / 2! = 180 :=
sorry

end grouping_count_l97_97032


namespace tan_of_perpendicular_vectors_l97_97846

theorem tan_of_perpendicular_vectors (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 2)
  (ha : ℝ × ℝ := (Real.cos θ, 2)) (hb : ℝ × ℝ := (-1, Real.sin θ))
  (h_perpendicular : ha.1 * hb.1 + ha.2 * hb.2 = 0) :
  Real.tan θ = 1 / 2 := 
sorry

end tan_of_perpendicular_vectors_l97_97846


namespace ivy_baked_55_cupcakes_l97_97035

-- Definitions based on conditions
def cupcakes_morning : ℕ := 20
def cupcakes_afternoon : ℕ := cupcakes_morning + 15
def total_cupcakes : ℕ := cupcakes_morning + cupcakes_afternoon

-- Theorem statement that needs to be proved
theorem ivy_baked_55_cupcakes : total_cupcakes = 55 := by
    sorry

end ivy_baked_55_cupcakes_l97_97035


namespace children_count_after_addition_l97_97176

theorem children_count_after_addition :
  ∀ (total_guests men guests children_added : ℕ),
    total_guests = 80 →
    men = 40 →
    guests = (men + men / 2) →
    children_added = 10 →
    total_guests - guests + children_added = 30 :=
by
  intros total_guests men guests children_added h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end children_count_after_addition_l97_97176


namespace hcf_two_numbers_l97_97189

theorem hcf_two_numbers (H a b : ℕ) (coprime_ab : Nat.gcd a b = 1) 
    (lcm_factors : a * b = 150) (larger_num : H * a = 450 ∨ H * b = 450) : H = 30 := 
by
  sorry

end hcf_two_numbers_l97_97189


namespace fewest_cookies_l97_97219

theorem fewest_cookies
  (area_art_cookies : ℝ)
  (area_roger_cookies : ℝ)
  (area_paul_cookies : ℝ)
  (area_trisha_cookies : ℝ)
  (h_art : area_art_cookies = 12)
  (h_roger : area_roger_cookies = 8)
  (h_paul : area_paul_cookies = 6)
  (h_trisha : area_trisha_cookies = 6)
  (dough : ℝ) :
  (dough / area_art_cookies) < (dough / area_roger_cookies) ∧
  (dough / area_art_cookies) < (dough / area_paul_cookies) ∧
  (dough / area_art_cookies) < (dough / area_trisha_cookies) := by
  sorry

end fewest_cookies_l97_97219


namespace first_folder_number_l97_97392

theorem first_folder_number (stickers : ℕ) (folders : ℕ) : stickers = 999 ∧ folders = 369 → 100 = 100 :=
by sorry

end first_folder_number_l97_97392


namespace complement_of_A_in_U_l97_97452

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 4}

theorem complement_of_A_in_U : (U \ A) = {1, 3, 5} := by
  sorry

end complement_of_A_in_U_l97_97452


namespace find_t_l97_97715

-- Given: (1) g(x) = x^5 + px^4 + qx^3 + rx^2 + sx + t with all roots being negative integers
--        (2) p + q + r + s + t = 3024
-- Prove: t = 1600

noncomputable def poly (x : ℝ) (p q r s t : ℝ) := 
  x^5 + p*x^4 + q*x^3 + r*x^2 + s*x + t

theorem find_t
  (p q r s t : ℝ)
  (roots_neg_int : ∀ root, root ∈ [-s1, -s2, -s3, -s4, -s5] → (root : ℤ) < 0)
  (sum_coeffs : p + q + r + s + t = 3024)
  (poly_1_eq : poly 1 p q r s t = 3025) :
  t = 1600 := 
sorry

end find_t_l97_97715


namespace camp_problem_l97_97612

variable (x : ℕ) -- number of girls
variable (y : ℕ) -- number of boys
variable (total_children : ℕ) -- total number of children
variable (girls_cannot_swim : ℕ) -- number of girls who cannot swim
variable (boys_cannot_swim : ℕ) -- number of boys who cannot swim
variable (children_can_swim : ℕ) -- total number of children who can swim
variable (children_cannot_swim : ℕ) -- total number of children who cannot swim
variable (o_six_girls : ℕ) -- one-sixth of the total number of girls
variable (o_eight_boys : ℕ) -- one-eighth of the total number of boys

theorem camp_problem 
    (hc1 : total_children = 50)
    (hc2 : girls_cannot_swim = x / 6)
    (hc3 : boys_cannot_swim = y / 8)
    (hc4 : children_can_swim = 43)
    (hc5 : children_cannot_swim = total_children - children_can_swim)
    (h_total : x + y = total_children)
    (h_swim : children_cannot_swim = girls_cannot_swim + boys_cannot_swim) :
    x = 18 :=
  by
    have hc6 : children_cannot_swim = 7 := by sorry -- from hc4 and hc5
    have h_eq : x / 6 + (50 - x) / 8 = 7 := by sorry -- from hc2, hc3, hc6
    -- solving for x
    sorry

end camp_problem_l97_97612


namespace total_time_over_weekend_l97_97161

def time_per_round : ℕ := 30
def rounds_saturday : ℕ := 11
def rounds_sunday : ℕ := 15

theorem total_time_over_weekend :
  (rounds_saturday * time_per_round) + (rounds_sunday * time_per_round) = 780 :=
by
  -- This is where the proof would go, but it is omitted as per instructions.
  sorry

end total_time_over_weekend_l97_97161


namespace find_valid_triples_l97_97542

-- Define the theorem to prove the conditions and results
theorem find_valid_triples :
  ∀ (a b c : ℕ), 
    (2^a + 2^b + 1) % (2^c - 1) = 0 ↔ (a = 0 ∧ b = 0 ∧ c = 2) ∨ 
                                      (a = 1 ∧ b = 2 ∧ c = 3) ∨ 
                                      (a = 2 ∧ b = 1 ∧ c = 3) := 
sorry  -- Proof omitted

end find_valid_triples_l97_97542


namespace mary_stickers_left_l97_97720

def initial_stickers : ℕ := 50
def stickers_per_friend : ℕ := 4
def number_of_friends : ℕ := 5
def total_students_including_mary : ℕ := 17
def stickers_per_other_student : ℕ := 2

theorem mary_stickers_left :
  let friends_stickers := stickers_per_friend * number_of_friends
  let other_students := total_students_including_mary - 1 - number_of_friends
  let other_students_stickers := stickers_per_other_student * other_students
  let total_given_away := friends_stickers + other_students_stickers
  initial_stickers - total_given_away = 8 :=
by
  sorry

end mary_stickers_left_l97_97720


namespace sum_of_squares_of_projections_constant_l97_97359

-- Define the sum of the squares of projections function
noncomputable def sum_of_squares_of_projections (a : ℝ) (α : ℝ) : ℝ :=
  let p1 := a * Real.cos α
  let p2 := a * Real.cos (Real.pi / 3 - α)
  let p3 := a * Real.cos (Real.pi / 3 + α)
  p1^2 + p2^2 + p3^2

-- Statement of the theorem
theorem sum_of_squares_of_projections_constant (a α : ℝ) : 
  sum_of_squares_of_projections a α = 3 / 2 * a^2 :=
sorry

end sum_of_squares_of_projections_constant_l97_97359


namespace area_percentage_change_is_neg_4_percent_l97_97630

noncomputable def percent_change_area (L W : ℝ) : ℝ :=
  let A_initial := L * W
  let A_new := (1.20 * L) * (0.80 * W)
  ((A_new - A_initial) / A_initial) * 100

theorem area_percentage_change_is_neg_4_percent (L W : ℝ) :
  percent_change_area L W = -4 :=
by
  sorry

end area_percentage_change_is_neg_4_percent_l97_97630


namespace find_largest_n_l97_97534

theorem find_largest_n : ∃ n x y z : ℕ, n > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 
  ∧ n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 3*x + 3*y + 3*z - 6
  ∧ (∀ m x' y' z' : ℕ, m > n → x' > 0 → y' > 0 → z' > 0 
  → m^2 ≠ x'^2 + y'^2 + z'^2 + 2*x'*y' + 2*y'*z' + 2*z'*x' + 3*x' + 3*y' + 3*z' - 6) :=
sorry

end find_largest_n_l97_97534


namespace tan_sum_identity_l97_97568

theorem tan_sum_identity
  (A B C : ℝ)
  (h1 : A + B + C = Real.pi)
  (h2 : Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C) :
  Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C := 
sorry

end tan_sum_identity_l97_97568


namespace cubs_more_home_runs_l97_97526

noncomputable def cubs_home_runs := 2 + 1 + 2
noncomputable def cardinals_home_runs := 1 + 1

theorem cubs_more_home_runs :
  cubs_home_runs - cardinals_home_runs = 3 :=
by
  -- Proof would go here, but we are using sorry to skip it
  sorry

end cubs_more_home_runs_l97_97526


namespace find_number_l97_97437

theorem find_number (n x : ℤ)
  (h1 : (2 * x + 1) = (x - 7)) 
  (h2 : ∃ x : ℤ, n = (2 * x + 1) ^ 2) : 
  n = 25 := 
sorry

end find_number_l97_97437


namespace cos_alpha_eq_2cos_alpha_plus_pi_div_4_implies_tan_alpha_plus_pi_div_8_l97_97005

theorem cos_alpha_eq_2cos_alpha_plus_pi_div_4_implies_tan_alpha_plus_pi_div_8
  (α : ℝ) (h : Real.cos α = 2 * Real.cos (α + Real.pi / 4)) :
  Real.tan (α + Real.pi / 8) = 3 * (Real.sqrt 2 + 1) := 
sorry

end cos_alpha_eq_2cos_alpha_plus_pi_div_4_implies_tan_alpha_plus_pi_div_8_l97_97005


namespace area_of_triangle_with_rational_vertices_on_unit_circle_is_rational_l97_97113

def rational_coords_on_unit_circle (x₁ y₁ x₂ y₂ x₃ y₃ : ℚ) : Prop :=
  x₁^2 + y₁^2 = 1 ∧ x₂^2 + y₂^2 = 1 ∧ x₃^2 + y₃^2 = 1

theorem area_of_triangle_with_rational_vertices_on_unit_circle_is_rational
  (x₁ y₁ x₂ y₂ x₃ y₃ : ℚ)
  (h : rational_coords_on_unit_circle x₁ y₁ x₂ y₂ x₃ y₃) :
  ∃ (A : ℚ), A = 1 / 2 * abs (x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂)) :=
sorry

end area_of_triangle_with_rational_vertices_on_unit_circle_is_rational_l97_97113


namespace no_intersection_of_asymptotes_l97_97801

noncomputable def given_function (x : ℝ) : ℝ :=
  (x^2 - 9 * x + 20) / (x^2 - 9 * x + 18)

theorem no_intersection_of_asymptotes : 
  (∀ x, x = 3 → ¬ ∃ y, y = given_function x) ∧ 
  (∀ x, x = 6 → ¬ ∃ y, y = given_function x) ∧ 
  ¬ ∃ x, (x = 3 ∨ x = 6) ∧ given_function x = 1 := 
by
  sorry

end no_intersection_of_asymptotes_l97_97801


namespace total_treats_l97_97072

theorem total_treats (children : ℕ) (hours : ℕ) (houses_per_hour : ℕ) (treats_per_house_per_kid : ℕ) :
  children = 3 → hours = 4 → houses_per_hour = 5 → treats_per_house_per_kid = 3 → 
  (children * hours * houses_per_hour * treats_per_house_per_kid) = 180 :=
by
  intros
  sorry

end total_treats_l97_97072


namespace percentage_salt_solution_l97_97299

theorem percentage_salt_solution (P : ℝ) (V_initial V_added V_final : ℝ) (C_initial C_final : ℝ) :
  V_initial = 30 ∧ C_initial = 0.20 ∧ V_final = 60 ∧ C_final = 0.40 → 
  V_added = 30 → 
  (C_initial * V_initial + (P / 100) * V_added) / V_final = C_final →
  P = 60 :=
by
  intro h
  sorry

end percentage_salt_solution_l97_97299


namespace polar_coordinates_of_point_l97_97794

open Real

theorem polar_coordinates_of_point :
  ∃ r θ : ℝ, r = 4 ∧ θ = 5 * π / 3 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧ 
           (∃ x y : ℝ, x = 2 ∧ y = -2 * sqrt 3 ∧ x = r * cos θ ∧ y = r * sin θ) :=
sorry

end polar_coordinates_of_point_l97_97794


namespace math_problem_l97_97658

theorem math_problem :
  18 * 35 + 45 * 18 - 18 * 10 = 1260 :=
by
  sorry

end math_problem_l97_97658


namespace trip_early_movie_savings_l97_97617

theorem trip_early_movie_savings : 
  let evening_ticket_cost : ℝ := 10
  let food_combo_cost : ℝ := 10
  let ticket_discount : ℝ := 0.20
  let food_discount : ℝ := 0.50
  let evening_total_cost := evening_ticket_cost + food_combo_cost
  let savings_on_ticket := evening_ticket_cost * ticket_discount
  let savings_on_food := food_combo_cost * food_discount
  let total_savings := savings_on_ticket + savings_on_food
  total_savings = 7 :=
by
  sorry

end trip_early_movie_savings_l97_97617


namespace simplify_expr_l97_97469

theorem simplify_expr : 3 * (4 - 2 * Complex.I) - 2 * Complex.I * (3 - 2 * Complex.I) = 8 - 12 * Complex.I :=
by
  sorry

end simplify_expr_l97_97469


namespace zero_points_ordering_l97_97821

noncomputable def f (x : ℝ) : ℝ := x + 2^x
noncomputable def g (x : ℝ) : ℝ := x + Real.log x
noncomputable def h (x : ℝ) : ℝ := x^3 + x - 2

theorem zero_points_ordering :
  ∃ x1 x2 x3 : ℝ,
    f x1 = 0 ∧ x1 < 0 ∧ 
    g x2 = 0 ∧ 0 < x2 ∧ x2 < 1 ∧
    h x3 = 0 ∧ 1 < x3 ∧ x3 < 2 ∧
    x1 < x2 ∧ x2 < x3 := sorry

end zero_points_ordering_l97_97821


namespace inequality_holds_l97_97322

theorem inequality_holds (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
    a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 :=
by
  sorry

end inequality_holds_l97_97322


namespace cooling_time_condition_l97_97210

theorem cooling_time_condition :
  ∀ (θ0 θ1 θ1' θ0' : ℝ) (t : ℝ), 
    θ0 = 20 → θ1 = 100 → θ1' = 60 → θ0' = 20 →
    let θ := θ0 + (θ1 - θ0) * Real.exp (-t / 4)
    let θ' := θ0' + (θ1' - θ0') * Real.exp (-t / 4)
    (θ - θ' ≤ 10) → (t ≥ 5.52) :=
sorry

end cooling_time_condition_l97_97210


namespace required_workers_l97_97101

variable (x : ℕ) (y : ℕ)

-- Each worker can produce x units of a craft per day.
-- A craft factory needs to produce 60 units of this craft per day.

theorem required_workers (h : x > 0) : y = 60 / x ↔ x * y = 60 :=
by sorry

end required_workers_l97_97101


namespace min_sides_of_polygon_that_overlaps_after_rotation_l97_97150

theorem min_sides_of_polygon_that_overlaps_after_rotation (θ : ℝ) (n : ℕ) 
  (hθ: θ = 36) (hdiv: 360 % θ = 0) :
    n = 10 :=
by
  sorry

end min_sides_of_polygon_that_overlaps_after_rotation_l97_97150


namespace minimum_surface_area_of_combined_cuboids_l97_97662

noncomputable def cuboid_combinations (l w h : ℕ) (n : ℕ) : ℕ :=
sorry

theorem minimum_surface_area_of_combined_cuboids :
  ∃ n, cuboid_combinations 2 1 3 3 = 4 ∧ n = 42 :=
sorry

end minimum_surface_area_of_combined_cuboids_l97_97662


namespace y_coordinate_midpoint_l97_97678

theorem y_coordinate_midpoint : 
  let L : (ℝ → ℝ) := λ x => x - 1
  let P : (ℝ → ℝ) := λ y => 8 * (y^2)
  ∃ (x₁ x₂ y₁ y₂ : ℝ),
    P (L x₁) = y₁ ∧ P (L x₂) = y₂ ∧ 
    L x₁ = y₁ ∧ L x₂ = y₂ ∧ 
    x₁ + x₂ = 10 ∧ y₁ + y₂ = 8 ∧
    (y₁ + y₂) / 2 = 4 := sorry

end y_coordinate_midpoint_l97_97678


namespace max_value_of_z_l97_97826

theorem max_value_of_z
  (x y : ℝ)
  (h1 : y ≥ x)
  (h2 : x + y ≤ 1)
  (h3 : y ≥ -1) :
  ∃ x y, (y ≥ x) ∧ (x + y ≤ 1) ∧ (y ≥ -1) ∧ (2 * x - y = 1 / 2) := by 
  sorry

end max_value_of_z_l97_97826


namespace football_kick_distance_l97_97247

theorem football_kick_distance (a : ℕ) (avg : ℕ) (x : ℕ)
  (h1 : a = 43)
  (h2 : avg = 37)
  (h3 : 3 * avg = a + 2 * x) :
  x = 34 :=
by
  sorry

end football_kick_distance_l97_97247


namespace cos_alpha_minus_beta_l97_97554

theorem cos_alpha_minus_beta : 
  ∀ (α β : ℝ), 
  2 * Real.cos α - Real.cos β = 3 / 2 →
  2 * Real.sin α - Real.sin β = 2 →
  Real.cos (α - β) = -5 / 16 :=
by
  intros α β h1 h2
  sorry

end cos_alpha_minus_beta_l97_97554


namespace asymptotes_of_hyperbola_l97_97066

theorem asymptotes_of_hyperbola (b : ℝ) (h_focus : 2 * Real.sqrt 2 ≠ 0) :
  2 * Real.sqrt 2 = Real.sqrt ((2 * 2) + b^2) → 
  (∀ (x y : ℝ), ((x^2 / 4) - (y^2 / b^2) = 1 → x^2 - y^2 = 4)) → 
  (∀ (x y : ℝ), ((x^2 - y^2 = 4) → y = x ∨ y = -x)) := 
  sorry

end asymptotes_of_hyperbola_l97_97066


namespace reservoir_capacity_l97_97094

theorem reservoir_capacity (x : ℝ) (h1 : (3 / 8) * x - (1 / 4) * x = 100) : x = 800 :=
by
  sorry

end reservoir_capacity_l97_97094


namespace division_quotient_example_l97_97442

theorem division_quotient_example :
  ∃ q : ℕ,
    let dividend := 760
    let divisor := 36
    let remainder := 4
    dividend = divisor * q + remainder ∧ q = 21 :=
by
  sorry

end division_quotient_example_l97_97442


namespace min_shoeing_time_l97_97221

theorem min_shoeing_time
  (num_blacksmiths : ℕ) (num_horses : ℕ) (hooves_per_horse : ℕ) (minutes_per_hoof : ℕ)
  (h_blacksmiths : num_blacksmiths = 48)
  (h_horses : num_horses = 60)
  (h_hooves_per_horse : hooves_per_horse = 4)
  (h_minutes_per_hoof : minutes_per_hoof = 5) :
  (num_horses * hooves_per_horse * minutes_per_hoof) / num_blacksmiths = 25 := 
by
  sorry

end min_shoeing_time_l97_97221


namespace anne_ben_charlie_difference_l97_97011

def sales_tax_rate : ℝ := 0.08
def original_price : ℝ := 120.00
def discount_rate : ℝ := 0.25
def charlie_discount_rate : ℝ := 0.15

def anne_total : ℝ := (original_price * (1 + sales_tax_rate)) * (1 - discount_rate)
def ben_total : ℝ := (original_price * (1 - discount_rate)) * (1 + sales_tax_rate)
def charlie_total : ℝ := (original_price * (1 - charlie_discount_rate)) * (1 + sales_tax_rate)

def anne_minus_ben_minus_charlie : ℝ := anne_total - ben_total - charlie_total

theorem anne_ben_charlie_difference : anne_minus_ben_minus_charlie = -12.96 :=
by
  sorry

end anne_ben_charlie_difference_l97_97011


namespace missing_digit_divisible_by_9_l97_97195

theorem missing_digit_divisible_by_9 (x : ℕ) (h : 0 ≤ x ∧ x < 10) : (3 + 5 + 1 + 9 + 2 + x) % 9 = 0 ↔ x = 7 :=
by
  sorry

end missing_digit_divisible_by_9_l97_97195


namespace shaded_region_area_l97_97226

open Real

noncomputable def area_of_shaded_region (r : ℝ) (s : ℝ) (d : ℝ) : ℝ := 
  (1/4) * π * r^2 + (1/2) * (d - s)^2

theorem shaded_region_area :
  let r := 3
  let s := 2
  let d := sqrt 5
  area_of_shaded_region r s d = 9 * π / 4 + (9 - 4 * sqrt 5) / 2 :=
by
  sorry

end shaded_region_area_l97_97226


namespace probability_of_picking_dumpling_with_egg_l97_97081

-- Definitions based on the conditions
def total_dumplings : ℕ := 10
def dumplings_with_eggs : ℕ := 3

-- The proof statement
theorem probability_of_picking_dumpling_with_egg :
  (dumplings_with_eggs : ℚ) / total_dumplings = 3 / 10 :=
by
  sorry

end probability_of_picking_dumpling_with_egg_l97_97081


namespace snack_bar_training_count_l97_97521

noncomputable def num_trained_in_snack_bar 
  (total_employees : ℕ) 
  (trained_in_buffet : ℕ) 
  (trained_in_dining_room : ℕ) 
  (trained_in_two_restaurants : ℕ) 
  (trained_in_three_restaurants : ℕ) : ℕ :=
  total_employees - trained_in_buffet - trained_in_dining_room + 
  trained_in_two_restaurants + trained_in_three_restaurants

theorem snack_bar_training_count : 
  num_trained_in_snack_bar 39 17 18 4 2 = 8 :=
sorry

end snack_bar_training_count_l97_97521


namespace q_can_complete_work_in_30_days_l97_97758

theorem q_can_complete_work_in_30_days (W_p W_q W_r : ℝ)
  (h1 : W_p = W_q + W_r)
  (h2 : W_p + W_q = 1/10)
  (h3 : W_r = 1/30) :
  1 / W_q = 30 :=
by
  -- Note: You can add proof here, but it's not required in the task.
  sorry

end q_can_complete_work_in_30_days_l97_97758


namespace floor_neg_seven_fourths_l97_97273

theorem floor_neg_seven_fourths : (Int.floor (-7 / 4 : ℚ) = -2) := 
by
  sorry

end floor_neg_seven_fourths_l97_97273


namespace parabola_intersection_l97_97947

theorem parabola_intersection :
  (∀ x y : ℝ, y = 3 * x^2 - 4 * x + 2 ↔ y = 9 * x^2 + 6 * x + 2) →
  (∃ x1 y1 x2 y2 : ℝ,
    (x1 = 0 ∧ y1 = 2) ∧ (x2 = -5 / 3 ∧ y2 = 17)) :=
by
  intro h
  sorry

end parabola_intersection_l97_97947


namespace A_can_complete_work_in_28_days_l97_97504
noncomputable def work_days_for_A (x : ℕ) (h : 4 / x = 1 / 21) : ℕ :=
  x / 3

theorem A_can_complete_work_in_28_days (x : ℕ) (h : 4 / x = 1 / 21) :
  work_days_for_A x h = 28 :=
  sorry

end A_can_complete_work_in_28_days_l97_97504


namespace prob_diff_colors_correct_l97_97583

def total_chips := 6 + 5 + 4 + 3

def prob_diff_colors : ℚ :=
  (6 / total_chips * (12 / total_chips) +
  5 / total_chips * (13 / total_chips) +
  4 / total_chips * (14 / total_chips) +
  3 / total_chips * (15 / total_chips))

theorem prob_diff_colors_correct :
  prob_diff_colors = 119 / 162 := by
  sorry

end prob_diff_colors_correct_l97_97583


namespace radius_of_inscribed_sphere_l97_97779

theorem radius_of_inscribed_sphere (a b c s : ℝ)
  (h1: 2 * (a * b + a * c + b * c) = 616)
  (h2: a + b + c = 40)
  : s = Real.sqrt 246 ↔ (2 * s) ^ 2 = a ^ 2 + b ^ 2 + c ^ 2 :=
by
  sorry

end radius_of_inscribed_sphere_l97_97779


namespace trip_movie_savings_l97_97619

def evening_ticket_cost : ℕ := 10
def combo_cost : ℕ := 10
def ticket_discount_percentage : ℕ := 20
def combo_discount_percentage : ℕ := 50

theorem trip_movie_savings :
  let ticket_saving := evening_ticket_cost * ticket_discount_percentage / 100,
      combo_saving := combo_cost * combo_discount_percentage / 100
  in ticket_saving + combo_saving = 7 :=
by
  sorry

end trip_movie_savings_l97_97619


namespace min_equal_area_triangles_l97_97721

theorem min_equal_area_triangles (chessboard_area missing_area : ℕ) (total_area : ℕ := chessboard_area - missing_area) 
(H1 : chessboard_area = 64) (H2 : missing_area = 1) : 
∃ n : ℕ, n = 18 ∧ (total_area = 63) → total_area / ((7:ℕ)/2) = n := 
sorry

end min_equal_area_triangles_l97_97721


namespace lcm_18_35_l97_97805

-- Given conditions: Prime factorizations of 18 and 35
def factorization_18 : Prop := (18 = 2^1 * 3^2)
def factorization_35 : Prop := (35 = 5^1 * 7^1)

-- The goal is to prove that the least common multiple of 18 and 35 is 630
theorem lcm_18_35 : factorization_18 ∧ factorization_35 → Nat.lcm 18 35 = 630 := by
  sorry -- Proof to be filled in

end lcm_18_35_l97_97805


namespace simplify_expression_l97_97939

theorem simplify_expression :
  (4 + 5) * (4 ^ 2 + 5 ^ 2) * (4 ^ 4 + 5 ^ 4) * (4 ^ 8 + 5 ^ 8) * (4 ^ 16 + 5 ^ 16) * (4 ^ 32 + 5 ^ 32) * (4 ^ 64 + 5 ^ 64) = 5 ^ 128 - 4 ^ 128 :=
by sorry

end simplify_expression_l97_97939


namespace gwen_spent_money_l97_97550

theorem gwen_spent_money (initial : ℕ) (remaining : ℕ) (spent : ℕ) 
  (h_initial : initial = 7) 
  (h_remaining : remaining = 5) 
  (h_spent : spent = initial - remaining) : 
  spent = 2 := 
sorry

end gwen_spent_money_l97_97550


namespace min_fraction_in_domain_l97_97117

theorem min_fraction_in_domain :
  ∃ x y : ℝ, (1/4 ≤ x ∧ x ≤ 2/3) ∧ (1/5 ≤ y ∧ y ≤ 1/2) ∧ 
    (∀ x' y' : ℝ, (1/4 ≤ x' ∧ x' ≤ 2/3) ∧ (1/5 ≤ y' ∧ y' ≤ 1/2) → 
      (xy / (x^2 + y^2) ≤ x'y' / (x'^2 + y'^2))) ∧ 
      xy / (x^2 + y^2) = 2/5 :=
sorry

end min_fraction_in_domain_l97_97117


namespace hyperbola_foci_product_l97_97843

theorem hyperbola_foci_product
  (F1 F2 P : ℝ × ℝ)
  (hF1 : F1 = (-Real.sqrt 5, 0))
  (hF2 : F2 = (Real.sqrt 5, 0))
  (hP : P.1 ^ 2 / 4 - P.2 ^ 2 = 1)
  (hDot : (P.1 + Real.sqrt 5) * (P.1 - Real.sqrt 5) + P.2 ^ 2 = 0) :
  (Real.sqrt ((P.1 + Real.sqrt 5) ^ 2 + P.2 ^ 2)) * (Real.sqrt ((P.1 - Real.sqrt 5) ^ 2 + P.2 ^ 2)) = 2 :=
sorry

end hyperbola_foci_product_l97_97843


namespace pow_divisible_by_13_l97_97335

theorem pow_divisible_by_13 (n : ℕ) (h : 0 < n) : (4^(2*n+1) + 3^(n+2)) % 13 = 0 :=
sorry

end pow_divisible_by_13_l97_97335


namespace combined_share_is_50000_l97_97605

def profit : ℝ := 80000

def majority_owner_share : ℝ := 0.25 * profit

def remaining_profit : ℝ := profit - majority_owner_share

def partner_share : ℝ := 0.25 * remaining_profit

def combined_share_majority_two_owners : ℝ := majority_owner_share + 2 * partner_share

theorem combined_share_is_50000 :
  combined_share_majority_two_owners = 50000 := 
by 
  sorry

end combined_share_is_50000_l97_97605


namespace term_2005_is_1004th_l97_97360

-- Define the first term and the common difference
def a1 : Int := -1
def d : Int := 2

-- Define the general term formula of the arithmetic sequence
def a_n (n : Nat) : Int :=
  a1 + (n - 1) * d

-- State the theorem that the year 2005 is the 1004th term in the sequence
theorem term_2005_is_1004th : ∃ n : Nat, a_n n = 2005 ∧ n = 1004 := by
  sorry

end term_2005_is_1004th_l97_97360


namespace floor_of_neg_seven_fourths_l97_97260

theorem floor_of_neg_seven_fourths : 
  Int.floor (-7 / 4 : ℚ) = -2 := 
by sorry

end floor_of_neg_seven_fourths_l97_97260


namespace closest_approx_of_q_l97_97881

theorem closest_approx_of_q :
  let result : ℝ := 69.28 * 0.004
  let q : ℝ := result / 0.03
  abs (q - 9.24) < 0.005 := 
by 
  let result : ℝ := 69.28 * 0.004
  let q : ℝ := result / 0.03
  sorry

end closest_approx_of_q_l97_97881


namespace geometric_series_sum_l97_97603

theorem geometric_series_sum (a r : ℝ) 
  (h1 : a * (1 - r / (1 - r)) = 18) 
  (h2 : a * (r / (1 - r)) = 8) : r = 4 / 5 :=
by sorry

end geometric_series_sum_l97_97603


namespace bookseller_original_cost_l97_97638

theorem bookseller_original_cost
  (x y z : ℝ)
  (h1 : 1.10 * x = 11.00)
  (h2 : 1.10 * y = 16.50)
  (h3 : 1.10 * z = 24.20) :
  x + y + z = 47.00 := by
  sorry

end bookseller_original_cost_l97_97638


namespace random_phenomenon_l97_97080

def is_certain_event (P : Prop) : Prop := ∀ h : P, true

def is_random_event (P : Prop) : Prop := ¬is_certain_event P

def scenario1 : Prop := ∀ pressure temperature : ℝ, (pressure = 101325) → (temperature = 100) → true
-- Under standard atmospheric pressure, water heated to 100°C will boil

def scenario2 : Prop := ∃ time : ℝ, true
-- Encountering a red light at a crossroads (which happens at random times)

def scenario3 (a b : ℝ) : Prop := true
-- For a rectangle with length and width a and b respectively, its area is a * b

def scenario4 : Prop := ∀ a b : ℝ, ∃ x : ℝ, a * x + b = 0
-- A linear equation with real coefficients always has one real root

theorem random_phenomenon : is_random_event scenario2 :=
by
  sorry

end random_phenomenon_l97_97080


namespace remainder_of_sum_mod_13_l97_97121

theorem remainder_of_sum_mod_13 (a b c d e : ℕ) 
  (h1: a % 13 = 3) (h2: b % 13 = 5) (h3: c % 13 = 7) (h4: d % 13 = 9) (h5: e % 13 = 11) :
  (a + b + c + d + e) % 13 = 9 := 
by 
  sorry

end remainder_of_sum_mod_13_l97_97121


namespace floor_neg_seven_fourths_l97_97274

theorem floor_neg_seven_fourths : (Int.floor (-7 / 4 : ℚ) = -2) := 
by
  sorry

end floor_neg_seven_fourths_l97_97274


namespace distance_traveled_l97_97501

-- Given conditions
def speed : ℕ := 100 -- Speed in km/hr
def time : ℕ := 5    -- Time in hours

-- The goal is to prove the distance traveled is 500 km
theorem distance_traveled : speed * time = 500 := by
  -- we state the proof goal
  sorry

end distance_traveled_l97_97501


namespace unique_triplet_l97_97286

theorem unique_triplet (a b p : ℕ) (hp : Nat.Prime p) (ha : 0 < a) (hb : 0 < b) :
  (1 / (p : ℚ) = 1 / (a^2 : ℚ) + 1 / (b^2 : ℚ)) → (a = 2 ∧ b = 2 ∧ p = 2) :=
by
  sorry

end unique_triplet_l97_97286


namespace remainders_mod_m_l97_97631

theorem remainders_mod_m {m n b : ℤ} (h_coprime : Int.gcd m n = 1) :
    (∀ r : ℤ, 0 ≤ r ∧ r < m → ∃ k : ℤ, 0 ≤ k ∧ k < n ∧ ((b + k * n) % m = r)) :=
by
  sorry

end remainders_mod_m_l97_97631


namespace eunji_initial_money_l97_97116

-- Define the conditions
def snack_cost : ℕ := 350
def allowance : ℕ := 800
def money_left_after_pencil : ℕ := 550

-- Define what needs to be proven
theorem eunji_initial_money (initial_money : ℕ) :
  initial_money - snack_cost + allowance = money_left_after_pencil * 2 →
  initial_money = 650 :=
by
  sorry

end eunji_initial_money_l97_97116


namespace greatest_value_x_plus_y_l97_97370

theorem greatest_value_x_plus_y (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) : 
  x + y ≤ 6 * Real.sqrt 5 := 
by
  sorry

end greatest_value_x_plus_y_l97_97370


namespace fraction_simplification_l97_97368

theorem fraction_simplification :
  10 * (1/2 + 1/5 + 1/10)⁻¹ = 25 / 2 :=
by
  sorry

end fraction_simplification_l97_97368


namespace group_count_4_men_5_women_l97_97027

theorem group_count_4_men_5_women : 
  let men := 4
  let women := 5
  let groups := List.replicate 3 (3, true)
  ∃ (m_w_combinations : List (ℕ × ℕ)),
    m_w_combinations = [(1, 2), (2, 1)] ∧
    ((men.choose m_w_combinations.head.fst * women.choose m_w_combinations.head.snd) * (men - m_w_combinations.head.fst).choose m_w_combinations.tail.head.fst * (women - m_w_combinations.head.snd).choose m_w_combinations.tail.head.snd) = 360 :=
by
  sorry

end group_count_4_men_5_women_l97_97027


namespace train_speed_l97_97645

def distance : ℕ := 500
def time : ℕ := 10
def conversion_factor : ℝ := 3.6

theorem train_speed :
  (distance / time : ℝ) * conversion_factor = 180 :=
by
  sorry

end train_speed_l97_97645


namespace sophomores_in_seminar_l97_97444

theorem sophomores_in_seminar (P Q x y : ℕ)
  (h1 : P + Q = 50)
  (h2 : x = y)
  (h3 : x = (1 / 5 : ℚ) * P)
  (h4 : y = (1 / 4 : ℚ) * Q) :
  P = 22 :=
by
  sorry

end sophomores_in_seminar_l97_97444


namespace numeral_eq_7000_l97_97346

theorem numeral_eq_7000 
  (local_value face_value numeral : ℕ)
  (h1 : face_value = 7)
  (h2 : local_value - face_value = 6993) : 
  numeral = 7000 :=
by
  sorry

end numeral_eq_7000_l97_97346


namespace cos_alpha_neg_3_5_l97_97154

open Real

variables {α : ℝ} (h_alpha : sin α = 4 / 5) (h_quadrant : π / 2 < α ∧ α < π)

theorem cos_alpha_neg_3_5 : cos α = -3 / 5 :=
by
  -- Proof omitted
  sorry

end cos_alpha_neg_3_5_l97_97154


namespace total_trees_planted_l97_97061

theorem total_trees_planted (apple_trees orange_trees : ℕ) (h₁ : apple_trees = 47) (h₂ : orange_trees = 27) : apple_trees + orange_trees = 74 := 
by
  -- We skip the proof step
  sorry

end total_trees_planted_l97_97061


namespace group_division_l97_97024

theorem group_division (men women : ℕ) : 
  men = 4 → women = 5 →
  (∃ g1 g2 g3 : set (fin $ men + women), 
    g1.card = 3 ∧ g2.card = 3 ∧ g3.card = 3 ∧ 
    (∀ g, g ∈ [g1, g2, g3] → (∃ m w : ℕ, 1 ≤ m ∧ 1 ≤ w ∧ 
      finset.card (finset.filter (λ x, x < men) g) = m ∧ 
      finset.card (finset.filter (λ x, x ≥ men) g) = w)) 
    ∧ finset.disjoint g1 g2 ∧ finset.disjoint g2 g3 ∧ finset.disjoint g3 g1 
    ∧ g1 ∪ g2 ∪ g3 = finset.univ (fin $ men + women)) → 
  finset.card (finset.powerset' 3 (finset.univ (fin $ men + women))) / 2 = 180 :=
begin
  intros hmen hwomen,
  sorry
end

end group_division_l97_97024


namespace valid_combinations_l97_97518

theorem valid_combinations (h s ic : ℕ) (h_eq : h = 4) (s_eq : s = 6) (ic_eq : ic = 3) :
  h * s - ic = 21 := by
  rw [h_eq, s_eq, ic_eq]
  norm_num

end valid_combinations_l97_97518


namespace second_shift_fraction_of_total_l97_97503

theorem second_shift_fraction_of_total (W E : ℕ) (h1 : ∀ (W : ℕ), E = (3 * W / 4))
  : let W₁ := W
    let E₁ := E
    let widgets_first_shift := W₁ * E₁
    let widgets_per_second_shift_employee := (2 * W₁) / 3
    let second_shift_employees := (4 * E₁) / 3
    let widgets_second_shift := (2 * W₁ / 3) * (4 * E₁ / 3)
    let total_widgets := widgets_first_shift + widgets_second_shift
    let fraction_second_shift := widgets_second_shift / total_widgets
    fraction_second_shift = 8 / 17 :=
sorry

end second_shift_fraction_of_total_l97_97503


namespace B_value_l97_97364

theorem B_value (A B : Nat) (hA : A < 10) (hB : B < 10) (h_div99 : (100000 * A + 10000 + 1000 * 5 + 100 * B + 90 + 4) % 99 = 0) :
  B = 3 :=
by
  -- skipping the proof
  sorry

end B_value_l97_97364


namespace stock_price_drop_l97_97958

theorem stock_price_drop (P : ℝ) (h1 : P > 0) (x : ℝ)
  (h3 : (1.30 * (1 - x/100) * 1.20 * P) = 1.17 * P) :
  x = 25 :=
by
  sorry

end stock_price_drop_l97_97958


namespace number_of_children_is_30_l97_97177

-- Informal statements
def total_guests := 80
def men := 40
def women := men / 2
def adults := men + women
def children := total_guests - adults
def children_after_adding_10 := children + 10

-- Formal proof statement
theorem number_of_children_is_30 :
  children_after_adding_10 = 30 := by
  sorry

end number_of_children_is_30_l97_97177


namespace circle_radius_zero_l97_97409

-- Define the given circle equation
def circle_eq (x y : ℝ) : Prop := x^2 - 4 * x + y^2 - 6 * y + 13 = 0

-- The proof problem statement
theorem circle_radius_zero : ∀ (x y : ℝ), circle_eq x y → 0 = 0 :=
by
  sorry

end circle_radius_zero_l97_97409


namespace inequality_abc_l97_97057

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c := 
by 
  sorry

end inequality_abc_l97_97057


namespace inequality_holds_l97_97323

theorem inequality_holds (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
    a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 :=
by
  sorry

end inequality_holds_l97_97323


namespace simplify_fraction_l97_97182

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2 + 1) = 16250 / 601 :=
by sorry

end simplify_fraction_l97_97182


namespace f_neg_one_f_monotonic_decreasing_solve_inequality_l97_97812

-- Definitions based on conditions in part a)
variables {f : ℝ → ℝ}
axiom f_add : ∀ x₁ x₂, f (x₁ + x₂) = f x₁ + f x₂ - 2
axiom f_one : f 1 = 0
axiom f_neg : ∀ x > 1, f x < 0

-- Proof statement for the value of f(-1)
theorem f_neg_one : f (-1) = 4 := by
  sorry

-- Proof statement for the monotonicity of f(x)
theorem f_monotonic_decreasing : ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂ := by
  sorry

-- Proof statement for the inequality solution
theorem solve_inequality (x : ℝ) :
  ∀ t, t = f (x^2 - 2*x) →
  t^2 + 2*t - 8 < 0 → (-1 < x ∧ x < 0) ∨ (2 < x ∧ x < 3) := by
  sorry

end f_neg_one_f_monotonic_decreasing_solve_inequality_l97_97812


namespace compound_proposition_p_or_q_l97_97053

theorem compound_proposition_p_or_q : 
  (∃ (n : ℝ), ∀ (m : ℝ), m * n = m) ∨ 
  (∀ (n : ℝ), ∃ (m : ℝ), m^2 < n) := 
by
  sorry

end compound_proposition_p_or_q_l97_97053


namespace arithmetic_sequence_n_l97_97844

theorem arithmetic_sequence_n {a : ℕ → ℕ} (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = a n + 3) :
  (∃ n : ℕ, a n = 2005) → (∃ n : ℕ, n = 669) :=
by
  sorry

end arithmetic_sequence_n_l97_97844


namespace total_walnut_trees_l97_97487

-- Define the conditions
def current_walnut_trees := 4
def new_walnut_trees := 6

-- State the lean proof problem
theorem total_walnut_trees : current_walnut_trees + new_walnut_trees = 10 := by
  sorry

end total_walnut_trees_l97_97487


namespace ratio_of_sphere_radii_l97_97904

noncomputable def ratio_of_radius (V_large : ℝ) (percentage : ℝ) : ℝ :=
  let V_small := (percentage / 100) * V_large
  let ratio := (V_small / V_large) ^ (1/3)
  ratio

theorem ratio_of_sphere_radii : 
  ratio_of_radius (450 * Real.pi) 27.04 = 0.646 := 
  by
  sorry

end ratio_of_sphere_radii_l97_97904


namespace find_integer_pairs_l97_97404

theorem find_integer_pairs (x y: ℤ) :
  x^2 - y^4 = 2009 → (x = 45 ∧ (y = 2 ∨ y = -2)) ∨ (x = -45 ∧ (y = 2 ∨ y = -2)) :=
by
  sorry

end find_integer_pairs_l97_97404


namespace ratio_add_b_l97_97006

theorem ratio_add_b (a b : ℚ) (h : a / b = 3 / 5) : (a + b) / b = 8 / 5 :=
by
  sorry

end ratio_add_b_l97_97006


namespace min_distance_sq_l97_97816

theorem min_distance_sq (x y : ℝ) (h : x - y - 1 = 0) : (x - 2) ^ 2 + (y - 2) ^ 2 = 1 / 2 :=
sorry

end min_distance_sq_l97_97816


namespace probability_bc_seated_next_l97_97098

theorem probability_bc_seated_next {P : ℝ} : 
  P = 2 / 3 :=
sorry

end probability_bc_seated_next_l97_97098


namespace prime_square_sum_l97_97975

theorem prime_square_sum (p q m : ℕ) (hp : Prime p) (hq : Prime q) (hne : p ≠ q)
  (hp_eq : p^2 - 2001 * p + m = 0) (hq_eq : q^2 - 2001 * q + m = 0) :
  p^2 + q^2 = 3996005 :=
sorry

end prime_square_sum_l97_97975


namespace math_proof_problem_l97_97981

noncomputable def M : ℝ :=
  let x := (Real.sqrt (Real.sqrt 7 + 3) + Real.sqrt (Real.sqrt 7 - 3)) / (Real.sqrt (Real.sqrt 7 + 2))
  let y := Real.sqrt (5 - 2 * Real.sqrt 6)
  x - y

theorem math_proof_problem :
  M = (Real.sqrt 57 - 6 * Real.sqrt 6 + 4) / 3 :=
by
  sorry

end math_proof_problem_l97_97981


namespace simplify_and_evaluate_l97_97853

theorem simplify_and_evaluate (a b : ℝ) (h_eqn : a^2 + b^2 - 2 * a + 4 * b = -5) :
  (a - 2 * b) * (a^2 + 2 * a * b + 4 * b^2) - a * (a - 5 * b) * (a + 3 * b) = 120 :=
sorry

end simplify_and_evaluate_l97_97853


namespace inequality_positive_real_xyz_l97_97330

theorem inequality_positive_real_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z)) + y^3 / ((1 + z) * (1 + x)) + z^3 / ((1 + x) * (1 + y))) ≥ (3 / 4) := 
by
  -- Proof is to be constructed here
  sorry

end inequality_positive_real_xyz_l97_97330


namespace highest_wave_height_l97_97102

-- Definitions of surfboard length and shortest wave conditions
def surfboard_length : ℕ := 7
def shortest_wave_height (H : ℕ) : ℕ := H + 4

-- Lean statement to be proved
theorem highest_wave_height (H : ℕ) (condition1 : H + 4 = surfboard_length + 3) : 
  4 * H + 2 = 26 :=
sorry

end highest_wave_height_l97_97102


namespace pictures_per_coloring_book_l97_97627

theorem pictures_per_coloring_book
    (total_colored : ℕ)
    (remaining_pictures : ℕ)
    (two_books : ℕ)
    (h1 : total_colored = 20) 
    (h2 : remaining_pictures = 68) 
    (h3 : two_books = 2) :
  (total_colored + remaining_pictures) / two_books = 44 :=
by
  sorry

end pictures_per_coloring_book_l97_97627


namespace total_packing_peanuts_used_l97_97367

def large_order_weight : ℕ := 200
def small_order_weight : ℕ := 50
def large_orders_sent : ℕ := 3
def small_orders_sent : ℕ := 4

theorem total_packing_peanuts_used :
  (large_orders_sent * large_order_weight) + (small_orders_sent * small_order_weight) = 800 := 
by
  sorry

end total_packing_peanuts_used_l97_97367


namespace min_k_intersects_circle_l97_97416

def circle_eq (x y : ℝ) := (x + 2)^2 + y^2 = 4
def line_eq (x y k : ℝ) := k * x - y - 2 * k = 0

theorem min_k_intersects_circle :
  (∀ k : ℝ, (∃ x y : ℝ, circle_eq x y ∧ line_eq x y k) → k ≥ - (Real.sqrt 3) / 3) :=
sorry

end min_k_intersects_circle_l97_97416


namespace permutation_sum_l97_97766

theorem permutation_sum (n : ℕ) (h1 : n + 3 ≤ 2 * n) (h2 : n + 1 ≤ 4) (h3 : n > 0) :
  Nat.factorial (2 * n) / Nat.factorial (2 * n - (n + 3)) + Nat.factorial 4 / Nat.factorial (4 - (n + 1)) = 744 :=
by
  sorry

end permutation_sum_l97_97766


namespace seating_arrangements_l97_97538

theorem seating_arrangements (n : ℕ) (hn : n = 8) : 
  ∃ (k : ℕ), k = 5760 :=
by
  sorry

end seating_arrangements_l97_97538


namespace range_of_k_l97_97423

theorem range_of_k {k : ℝ} : (∀ x : ℝ, x < 0 → (k - 2)/x > 0) ∧ (∀ x : ℝ, x > 0 → (k - 2)/x < 0) → k < 2 := 
by
  sorry

end range_of_k_l97_97423


namespace problem_solution_l97_97564

-- Definitions of the conditions as Lean statements:
def condition1 (t : ℝ) : Prop :=
  (1 + Real.sin t) * (1 - Real.cos t) = 1

def condition2 (t : ℝ) (a b c : ℕ) : Prop :=
  (1 - Real.sin t) * (1 + Real.cos t) = (a / b) - Real.sqrt c

def areRelativelyPrime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

-- The proof problem statement:
theorem problem_solution (t : ℝ) (a b c : ℕ) (h1 : condition1 t) (h2 : condition2 t a b c) (h3 : areRelativelyPrime a b) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c) : a + b + c = 2 := 
sorry

end problem_solution_l97_97564


namespace train_stops_for_10_minutes_per_hour_l97_97281

-- Define the conditions
def speed_excluding_stoppages : ℕ := 48 -- in kmph
def speed_including_stoppages : ℕ := 40 -- in kmph

-- Define the question as proving the train stops for 10 minutes per hour
theorem train_stops_for_10_minutes_per_hour :
  (speed_excluding_stoppages - speed_including_stoppages) * 60 / speed_excluding_stoppages = 10 :=
by
  sorry

end train_stops_for_10_minutes_per_hour_l97_97281


namespace apps_minus_files_eq_seven_l97_97795

-- Definitions based on conditions
def initial_apps := 24
def initial_files := 9
def deleted_apps := initial_apps - 12
def deleted_files := initial_files - 5

-- Definitions based on the question and correct answer
def apps_left := 12
def files_left := 5

theorem apps_minus_files_eq_seven : apps_left - files_left = 7 := by
  sorry

end apps_minus_files_eq_seven_l97_97795


namespace condition_for_M_eq_N_l97_97317

theorem condition_for_M_eq_N (a1 b1 c1 a2 b2 c2 : ℝ) 
    (h1 : a1 ≠ 0) (h2 : b1 ≠ 0) (h3 : c1 ≠ 0) 
    (h4 : a2 ≠ 0) (h5 : b2 ≠ 0) (h6 : c2 ≠ 0) :
    (a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2) → 
    (M = {x : ℝ | a1 * x ^ 2 + b1 * x + c1 > 0} ∧
     N = {x : ℝ | a2 * x ^ 2 + b2 * x + c2 > 0} →
    (¬ (M = N))) ∨ (¬ (N = {} ↔ (M = N))) :=
sorry

end condition_for_M_eq_N_l97_97317


namespace floor_neg_seven_quarter_l97_97271

theorem floor_neg_seven_quarter : 
  ∃ x : ℤ, -2 ≤ (-7 / 4 : ℚ) ∧ (-7 / 4 : ℚ) < -1 ∧ x = -2 := by
  have h1 : (-7 / 4 : ℚ) = -1.75 := by norm_num
  have h2 : -2 ≤ -1.75 := by norm_num
  have h3 : -1.75 < -1 := by norm_num
  use -2
  exact ⟨h2, h3, rfl⟩
  sorry

end floor_neg_seven_quarter_l97_97271


namespace maximum_value_expression_l97_97454

-- Defining the variables and the main condition
variables (x y z : ℝ)

-- Assuming the non-negativity and sum of squares conditions
variables (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hxyz : x^2 + y^2 + z^2 = 1)

-- Main statement about the maximum value
theorem maximum_value_expression : 
  4 * x * y * Real.sqrt 2 + 5 * y * z + 3 * x * z * Real.sqrt 3 ≤ 
  (44 * Real.sqrt 2 + 110 + 9 * Real.sqrt 3) / 3 :=
sorry

end maximum_value_expression_l97_97454


namespace common_tangent_theorem_l97_97297

-- Define the first circle with given equation (x+2)^2 + (y-2)^2 = 1
def circle1 (x y : ℝ) : Prop := (x + 2)^2 + (y - 2)^2 = 1

-- Define the second circle with given equation (x-2)^2 + (y-5)^2 = 16
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 5)^2 = 16

-- Define a predicate that expresses the concept of common tangents between two circles
def common_tangents_count (circle1 circle2 : ℝ → ℝ → Prop) : ℕ := sorry

-- The statement to prove that the number of common tangents is 3
theorem common_tangent_theorem : common_tangents_count circle1 circle2 = 3 :=
by
  -- We would proceed with the proof if required, but we end with sorry as requested.
  sorry

end common_tangent_theorem_l97_97297


namespace find_x_of_parallel_vectors_l97_97004

theorem find_x_of_parallel_vectors
  (x : ℝ)
  (p : ℝ × ℝ := (2, -3))
  (q : ℝ × ℝ := (x, 6))
  (h : ∃ k : ℝ, q = k • p) :
  x = -4 :=
sorry

end find_x_of_parallel_vectors_l97_97004


namespace largest_common_term_arith_progressions_l97_97341

theorem largest_common_term_arith_progressions (a : ℕ) : 
  (∃ n m : ℕ, a = 4 + 5 * n ∧ a = 3 + 9 * m ∧ a < 1000) → a = 984 := by
  -- Proof is not required, so we add sorry.
  sorry

end largest_common_term_arith_progressions_l97_97341


namespace distinct_real_numbers_proof_l97_97168

variables {a b c : ℝ}

theorem distinct_real_numbers_proof (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : c ≠ a)
  (h₄ : (a / (b - c) + b / (c - a) + c / (a - b)) = -1) :
  (a^3 / (b - c)^2) + (b^3 / (c - a)^2) + (c^3 / (a - b)^2) = 0 :=
sorry

end distinct_real_numbers_proof_l97_97168


namespace smallest_positive_integer_l97_97622

theorem smallest_positive_integer (
    b : ℤ 
) : 
    (b % 4 = 1) → (b % 5 = 2) → (b % 6 = 3) → b = 21 := 
by
  intros h1 h2 h3
  sorry

end smallest_positive_integer_l97_97622


namespace area_of_triangle_from_line_l97_97580

-- Define the conditions provided in the problem
def line_eq (B : ℝ) (x y : ℝ) := B * x + 9 * y = 18
def B_val := (36 : ℝ)

theorem area_of_triangle_from_line (B : ℝ) (hB : B = B_val) : 
  (∃ C : ℝ, C = 1 / 2) := by
  sorry

end area_of_triangle_from_line_l97_97580


namespace min_value_of_expression_l97_97127

theorem min_value_of_expression
  (a b : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hlines : (∀ x y : ℝ, x + (a-4) * y + 1 = 0) ∧ (∀ x y : ℝ, 2 * b * x + y - 2 = 0) ∧ (∀ x y : ℝ, (x + (a-4) * y + 1 = 0) ∧ (2 * b * x + y - 2 = 0) → -1 * 1 / (a-4) * -2 * b = 1)) :
  ∃ (min_val : ℝ), min_val = (9/5) ∧ min_val = (a + 2)/(a + 1) + 1/(2 * b) :=
by
  sorry

end min_value_of_expression_l97_97127


namespace denominator_or_divisor_cannot_be_zero_l97_97733

theorem denominator_or_divisor_cannot_be_zero (a b c : ℝ) : b ≠ 0 ∧ c ≠ 0 → (a / b ≠ a ∨ a / c ≠ a) :=
by
  intro h
  sorry

end denominator_or_divisor_cannot_be_zero_l97_97733


namespace fourth_person_height_l97_97486

-- Definitions based on conditions
def h1 : ℕ := 73  -- height of first person
def h2 : ℕ := h1 + 2  -- height of second person
def h3 : ℕ := h2 + 2  -- height of third person
def h4 : ℕ := h3 + 6  -- height of fourth person

theorem fourth_person_height : h4 = 83 :=
by
  -- calculation to check the average height and arriving at h1
  -- (all detailed calculations are skipped using "sorry")
  sorry

end fourth_person_height_l97_97486


namespace first_year_with_sum_15_l97_97208

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) + (n % 1000 / 100) + (n % 100 / 10) + (n % 10)

theorem first_year_with_sum_15 : ∃ y > 2100, sum_of_digits y = 15 :=
  sorry

end first_year_with_sum_15_l97_97208


namespace minimum_value_l97_97118

def f (x y : ℝ) : ℝ := x * y / (x^2 + y^2)

def x_in_domain (x : ℝ) : Prop := (1/4 : ℝ) ≤ x ∧ x ≤ (2/3 : ℝ)
def y_in_domain (y : ℝ) : Prop := (1/5 : ℝ) ≤ y ∧ y ≤ (1/2 : ℝ)

theorem minimum_value (x y : ℝ) (hx : x_in_domain x) (hy : y_in_domain y) :
  ∃ x y, f x y = (2/5 : ℝ) := 
sorry

end minimum_value_l97_97118


namespace danny_bottle_cap_count_l97_97397

theorem danny_bottle_cap_count 
  (initial_caps : Int) 
  (found_caps : Int) 
  (final_caps : Int) 
  (h1 : initial_caps = 6) 
  (h2 : found_caps = 22) 
  (h3 : final_caps = initial_caps + found_caps) : 
  final_caps = 28 :=
by
  sorry

end danny_bottle_cap_count_l97_97397


namespace only_integer_solution_is_trivial_l97_97541

theorem only_integer_solution_is_trivial (a b c : ℤ) (h : 5 * a^2 + 9 * b^2 = 13 * c^2) : a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end only_integer_solution_is_trivial_l97_97541


namespace probability_distribution_m_l97_97999

theorem probability_distribution_m (m : ℚ) : 
  (m + m / 2 + m / 3 + m / 4 = 1) → m = 12 / 25 :=
by sorry

end probability_distribution_m_l97_97999


namespace floor_neg_seven_over_four_l97_97267

theorem floor_neg_seven_over_four : (Int.floor (-7 / 4 : ℚ)) = -2 := 
by
  sorry

end floor_neg_seven_over_four_l97_97267


namespace vector_addition_correct_dot_product_correct_l97_97296

def vector_add (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
def dot_product (a b : ℝ × ℝ) : ℝ := (a.1 * b.1) + (a.2 * b.2)

theorem vector_addition_correct :
  let a := (1, 2)
  let b := (3, 1)
  vector_add a b = (4, 3) := by
  sorry

theorem dot_product_correct :
  let a := (1, 2)
  let b := (3, 1)
  dot_product a b = 5 := by
  sorry

end vector_addition_correct_dot_product_correct_l97_97296


namespace todd_numbers_sum_eq_l97_97833

def sum_of_todd_numbers (n : ℕ) : ℕ :=
  sorry -- This would be the implementation of the sum based on provided problem conditions

theorem todd_numbers_sum_eq :
  sum_of_todd_numbers 5000 = 1250025 :=
sorry

end todd_numbers_sum_eq_l97_97833


namespace max_of_inverse_power_sums_l97_97990

theorem max_of_inverse_power_sums (s p r1 r2 : ℝ) 
  (h_eq_roots : r1 + r2 = s ∧ r1 * r2 = p)
  (h_eq_powers : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 2023 → r1^n + r2^n = s) :
  1 / r1^(2024:ℕ) + 1 / r2^(2024:ℕ) ≤ 2 :=
sorry

end max_of_inverse_power_sums_l97_97990


namespace second_player_can_ensure_symmetry_l97_97749

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

end second_player_can_ensure_symmetry_l97_97749


namespace grouping_schemes_count_l97_97202

/-- Number of possible grouping schemes where each group consists
    of either 2 or 3 students and the total number of students is 25 is 4.-/
theorem grouping_schemes_count : ∃ (x y : ℕ), 2 * x + 3 * y = 25 ∧ 
  (x = 11 ∧ y = 1 ∨ x = 8 ∧ y = 3 ∨ x = 5 ∧ y = 5 ∨ x = 2 ∧ y = 7) :=
sorry

end grouping_schemes_count_l97_97202


namespace swim_meet_time_l97_97959

theorem swim_meet_time {distance : ℕ} (d : distance = 50) (t : ℕ) 
  (meet_first : ∃ t1 : ℕ, t1 = 2 ∧ distance - 20 = 30) 
  (turn : ∀ t1, t1 = 2 → ∀ d1 : ℕ, d1 = 50 → t1 + t1 = 4) :
  t = 4 :=
by
  -- Placeholder proof
  sorry

end swim_meet_time_l97_97959


namespace abs_pi_expression_l97_97666

theorem abs_pi_expression (h : Real.pi < 10) : 
  Real.abs (Real.pi - Real.abs (Real.pi - 10)) = 10 - 2 * Real.pi :=
by
  sorry

end abs_pi_expression_l97_97666


namespace find_divisor_l97_97209

theorem find_divisor :
  ∃ d : ℕ, (4499 + 1) % d = 0 ∧ d = 2 :=
by
  sorry

end find_divisor_l97_97209


namespace eval_floor_neg_seven_fourths_l97_97258

theorem eval_floor_neg_seven_fourths : 
  ∃ (x : ℚ), x = -7 / 4 ∧ ∀ y, y ≤ x ∧ y ∈ ℤ → y ≤ -2 :=
by
  obtain ⟨x, hx⟩ : ∃ (x : ℚ), x = -7 / 4 := ⟨-7 / 4, rfl⟩,
  use x,
  split,
  { exact hx },
  { intros y hy,
    sorry }

end eval_floor_neg_seven_fourths_l97_97258


namespace magnitude_of_angle_A_range_of_b_plus_c_l97_97701

--- Definitions for the conditions
variables {A B C : ℝ} {a b c : ℝ}

-- Given condition a / (sqrt 3 * cos A) = c / sin C
axiom condition1 : a / (Real.sqrt 3 * Real.cos A) = c / Real.sin C

-- Given a = 6
axiom condition2 : a = 6

-- Conditions for sides b and c being positive
axiom condition3 : b > 0
axiom condition4 : c > 0
-- Condition for triangle inequality
axiom condition5 : b + c > a

-- Part (I) Find the magnitude of angle A
theorem magnitude_of_angle_A : A = Real.pi / 3 :=
by
  sorry

-- Part (II) Determine the range of values for b + c given a = 6
theorem range_of_b_plus_c : 6 < b + c ∧ b + c ≤ 12 :=
by
  sorry

end magnitude_of_angle_A_range_of_b_plus_c_l97_97701


namespace sum_of_numbers_l97_97964

theorem sum_of_numbers (a b c d : ℕ) (h1 : a > d) (h2 : a * b = c * d) (h3 : a + b + c + d = a * c) (h4 : ∀ x y z w: ℕ, x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ) : a + b + c + d = 12 :=
sorry

end sum_of_numbers_l97_97964


namespace number_of_integer_solutions_l97_97326

theorem number_of_integer_solutions (x : ℤ) :
  (∃ n : ℤ, n^2 = x^4 + 8*x^3 + 18*x^2 + 8*x + 36) ↔ x = -1 :=
sorry

end number_of_integer_solutions_l97_97326


namespace smallest_number_of_coins_to_pay_up_to_2_dollars_l97_97879

def smallest_number_of_coins_to_pay_up_to (max_amount : Nat) : Nat :=
  sorry  -- This function logic needs to be defined separately

theorem smallest_number_of_coins_to_pay_up_to_2_dollars :
  smallest_number_of_coins_to_pay_up_to 199 = 11 :=
sorry

end smallest_number_of_coins_to_pay_up_to_2_dollars_l97_97879


namespace smallest_number_of_coins_l97_97775

theorem smallest_number_of_coins (p n d q : ℕ) (total : ℕ) :
  (total < 100) →
  (total = p * 1 + n * 5 + d * 10 + q * 25) →
  (∀ k < 100, ∃ (p n d q : ℕ), k = p * 1 + n * 5 + d * 10 + q * 25) →
  p + n + d + q = 10 :=
sorry

end smallest_number_of_coins_l97_97775


namespace fully_factor_expression_l97_97125

theorem fully_factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := 
by
  -- pending proof, represented by sorry
  sorry

end fully_factor_expression_l97_97125


namespace relationship_among_sets_l97_97294

-- Definitions of the integer sets E, F, and G
def E := {e : ℝ | ∃ m : ℤ, e = m + 1 / 6}
def F := {f : ℝ | ∃ n : ℤ, f = n / 2 - 1 / 3}
def G := {g : ℝ | ∃ p : ℤ, g = p / 2 + 1 / 6}

-- The theorem statement capturing the relationship among E, F, and G
theorem relationship_among_sets : E ⊆ F ∧ F = G := by
  sorry

end relationship_among_sets_l97_97294


namespace marks_in_biology_l97_97533

theorem marks_in_biology (marks_english : ℕ) (marks_math : ℕ) (marks_physics : ℕ) (marks_chemistry : ℕ) (average_marks : ℕ) :
  marks_english = 73 → marks_math = 69 → marks_physics = 92 → marks_chemistry = 64 → average_marks = 76 →
  (380 - (marks_english + marks_math + marks_physics + marks_chemistry)) = 82 :=
by
  intros
  sorry

end marks_in_biology_l97_97533


namespace brick_height_calculation_l97_97492

theorem brick_height_calculation :
  ∀ (num_bricks : ℕ) (brick_length brick_width brick_height : ℝ)
    (wall_length wall_height wall_width : ℝ),
    num_bricks = 1600 →
    brick_length = 100 →
    brick_width = 11.25 →
    wall_length = 800 →
    wall_height = 600 →
    wall_width = 22.5 →
    wall_length * wall_height * wall_width = 
    num_bricks * brick_length * brick_width * brick_height →
    brick_height = 60 :=
by
  sorry

end brick_height_calculation_l97_97492


namespace lily_coffee_budget_l97_97048

variable (initial_amount celery_price cereal_original_price bread_price milk_original_price potato_price : ℕ)
variable (cereal_discount milk_discount number_of_potatoes : ℕ)

theorem lily_coffee_budget 
  (h_initial_amount : initial_amount = 60)
  (h_celery_price : celery_price = 5)
  (h_cereal_original_price : cereal_original_price = 12)
  (h_bread_price : bread_price = 8)
  (h_milk_original_price : milk_original_price = 10)
  (h_potato_price : potato_price = 1)
  (h_number_of_potatoes : number_of_potatoes = 6)
  (h_cereal_discount : cereal_discount = 50)
  (h_milk_discount : milk_discount = 10) :
  initial_amount - (celery_price + (cereal_original_price * cereal_discount / 100) + bread_price + (milk_original_price - (milk_original_price * milk_discount / 100)) + (potato_price * number_of_potatoes)) = 26 :=
by
  sorry

end lily_coffee_budget_l97_97048


namespace sum_of_odd_coefficients_l97_97196

theorem sum_of_odd_coefficients (a : ℝ) (h : (a + 1) * 16 = 32) : a = 3 :=
by
  sorry

end sum_of_odd_coefficients_l97_97196


namespace probability_of_rolling_five_l97_97515

-- Define a cube with the given face numbers
def cube_faces : List ℕ := [1, 1, 2, 4, 5, 5]

-- Prove the probability of rolling a "5" is 1/3
theorem probability_of_rolling_five :
  (cube_faces.count 5 : ℚ) / cube_faces.length = 1 / 3 := by
  sorry

end probability_of_rolling_five_l97_97515


namespace num_roots_l97_97480

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 3 * x - 2

theorem num_roots : ∃! x : ℝ, f x = 0 := 
sorry

end num_roots_l97_97480


namespace score_sd_above_mean_l97_97949

theorem score_sd_above_mean (mean std dev1 dev2 : ℝ) : 
  mean = 74 → dev1 = 2 → dev2 = 3 → mean - dev1 * std = 58 → mean + dev2 * std = 98 :=
by
  sorry

end score_sd_above_mean_l97_97949


namespace max_x_values_l97_97420

noncomputable def y (x : ℝ) : ℝ := (1/2) * (Real.cos x)^2 + (Real.sqrt 3 / 2) * (Real.sin x) * (Real.cos x) + 1

theorem max_x_values :
  {x : ℝ | ∃ k : ℤ, x = k * Real.pi + Real.pi / 6} = {x : ℝ | y x = y (x)} :=
sorry

end max_x_values_l97_97420


namespace floor_neg_seven_over_four_l97_97255

theorem floor_neg_seven_over_four : floor (-7 / 4) = -2 :=
by
  sorry

end floor_neg_seven_over_four_l97_97255


namespace gain_percent_l97_97756

theorem gain_percent (CP SP : ℝ) (hCP : CP = 100) (hSP : SP = 115) : 
  ((SP - CP) / CP) * 100 = 15 := 
by 
  sorry

end gain_percent_l97_97756


namespace group_division_l97_97022

theorem group_division (men women : ℕ) : 
  men = 4 → women = 5 →
  (∃ g1 g2 g3 : set (fin $ men + women), 
    g1.card = 3 ∧ g2.card = 3 ∧ g3.card = 3 ∧ 
    (∀ g, g ∈ [g1, g2, g3] → (∃ m w : ℕ, 1 ≤ m ∧ 1 ≤ w ∧ 
      finset.card (finset.filter (λ x, x < men) g) = m ∧ 
      finset.card (finset.filter (λ x, x ≥ men) g) = w)) 
    ∧ finset.disjoint g1 g2 ∧ finset.disjoint g2 g3 ∧ finset.disjoint g3 g1 
    ∧ g1 ∪ g2 ∪ g3 = finset.univ (fin $ men + women)) → 
  finset.card (finset.powerset' 3 (finset.univ (fin $ men + women))) / 2 = 180 :=
begin
  intros hmen hwomen,
  sorry
end

end group_division_l97_97022


namespace sum_inequality_l97_97558

theorem sum_inequality (x y z : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (hy : 1 ≤ y ∧ y ≤ 2) (hz : 1 ≤ z ∧ z ≤ 2) :
  (x + y + z) * (x⁻¹ + y⁻¹ + z⁻¹) ≥ 6 * (x / (y + z) + y / (z + x) + z / (x + y)) := sorry

end sum_inequality_l97_97558


namespace fully_factor_expression_l97_97126

theorem fully_factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := 
by
  -- pending proof, represented by sorry
  sorry

end fully_factor_expression_l97_97126


namespace symmetric_point_reflection_y_axis_l97_97344

theorem symmetric_point_reflection_y_axis (x y : ℝ) (h : (x, y) = (-2, 3)) :
  (-x, y) = (2, 3) :=
sorry

end symmetric_point_reflection_y_axis_l97_97344


namespace lily_coffee_budget_l97_97049

variable (initial_amount celery_price cereal_original_price bread_price milk_original_price potato_price : ℕ)
variable (cereal_discount milk_discount number_of_potatoes : ℕ)

theorem lily_coffee_budget 
  (h_initial_amount : initial_amount = 60)
  (h_celery_price : celery_price = 5)
  (h_cereal_original_price : cereal_original_price = 12)
  (h_bread_price : bread_price = 8)
  (h_milk_original_price : milk_original_price = 10)
  (h_potato_price : potato_price = 1)
  (h_number_of_potatoes : number_of_potatoes = 6)
  (h_cereal_discount : cereal_discount = 50)
  (h_milk_discount : milk_discount = 10) :
  initial_amount - (celery_price + (cereal_original_price * cereal_discount / 100) + bread_price + (milk_original_price - (milk_original_price * milk_discount / 100)) + (potato_price * number_of_potatoes)) = 26 :=
by
  sorry

end lily_coffee_budget_l97_97049


namespace sharpened_off_length_l97_97593

-- Define the conditions
def original_length : ℤ := 31
def length_after_sharpening : ℤ := 14

-- Define the theorem to prove the length sharpened off is 17 inches
theorem sharpened_off_length : original_length - length_after_sharpening = 17 := sorry

end sharpened_off_length_l97_97593


namespace find_wall_width_l97_97876

-- Define the volume of one brick
def volume_of_one_brick : ℚ := 100 * 11.25 * 6

-- Define the total number of bricks
def number_of_bricks : ℕ := 1600

-- Define the volume of all bricks combined
def total_volume_of_bricks : ℚ := volume_of_one_brick * number_of_bricks

-- Define dimensions of the wall
def wall_height : ℚ := 800 -- in cm (since 8 meters = 800 cm)
def wall_depth : ℚ := 22.5 -- in cm

-- Theorem to prove the width of the wall
theorem find_wall_width : ∃ width : ℚ, total_volume_of_bricks = wall_height * width * wall_depth ∧ width = 600 :=
by
  -- skipping the actual proof
  sorry

end find_wall_width_l97_97876


namespace density_function_proof_distribution_function_proof_l97_97895

noncomputable section

namespace MyProblem

-- Define the Lévy formula condition
def LevyFormula (a λ : ℝ) (h1 : a > 0) (h2 : λ > 0) : Prop :=
  ∫ t in set.Ioi 0, (exp (-λ * t) * (a * exp (-a^2 / (2 * t)) / sqrt (2 * π * t^3))) = exp (-a * sqrt (2 * λ))

-- Define the conditions related to \( f_{\alpha}(y) \) and \( g_{\alpha}(z) \)
def f_alpha_condition (α λ : ℝ) (f_alpha : ℝ → ℝ) (h1 : α > 0) (h2 : λ > 0) : Prop :=
  (sqrt (2 * λ) / sinh (sqrt (2 * λ)))^α = ∫ y in set.Ioi 0, exp (-λ * y) * f_alpha y

def g_alpha_condition (α λ : ℝ) (g_alpha : ℝ → ℝ) (h1 : α > 0) (h2 : λ > 0) : Prop :=
  (1 / cosh (sqrt (2 * λ)))^α = ∫ z in set.Ioi 0, exp (-λ * z) * g_alpha z

-- Define the proof for the density function
theorem density_function_proof (α : ℝ) : 
  ∀ z > 0, 
  g_alpha_condition α 1 (λ z, 2^α * ∑ n in set.Ici 0, nat_comb (-α) n * (2 * n + α) / sqrt (2 * π * z^3) * exp (-(2 * n + α)^2 / (2 * z))) :=
sorry

-- Define the proof for the distribution function
theorem distribution_function_proof : 
  ∀ y > 0, 
  f_alpha_condition 2 1 (λ y, (8 * sqrt 2) / sqrt (π * y^3) * ∑ k in set.Ici 1, k^2 * exp (-2 * k^2 / y)) :=
sorry

end MyProblem

end density_function_proof_distribution_function_proof_l97_97895


namespace std_deviation_above_l97_97952

variable (mean : ℝ) (std_dev : ℝ) (score1 : ℝ) (score2 : ℝ)
variable (n1 : ℝ) (n2 : ℝ)

axiom h_mean : mean = 74
axiom h_std1 : score1 = 58
axiom h_std2 : score2 = 98
axiom h_cond1 : score1 = mean - n1 * std_dev
axiom h_cond2 : n1 = 2

theorem std_deviation_above (mean std_dev score1 score2 n1 n2 : ℝ)
  (h_mean : mean = 74)
  (h_std1 : score1 = 58)
  (h_std2 : score2 = 98)
  (h_cond1 : score1 = mean - n1 * std_dev)
  (h_cond2 : n1 = 2) :
  n2 = (score2 - mean) / std_dev :=
sorry

end std_deviation_above_l97_97952


namespace floor_of_neg_seven_fourths_l97_97262

theorem floor_of_neg_seven_fourths : 
  Int.floor (-7 / 4 : ℚ) = -2 := 
by sorry

end floor_of_neg_seven_fourths_l97_97262


namespace empty_solution_set_range_l97_97682

theorem empty_solution_set_range (m : ℝ) : 
  (¬ ∃ x : ℝ, (m * x^2 + 2 * m * x + 1) < 0) ↔ (m = 0 ∨ (0 < m ∧ m ≤ 1)) :=
by sorry

end empty_solution_set_range_l97_97682


namespace unique_function_solution_l97_97283

theorem unique_function_solution (f : ℝ → ℝ) (h₁ : ∀ x : ℝ, x ≥ 1 → f x ≥ 1)
  (h₂ : ∀ x : ℝ, x ≥ 1 → f x ≤ 2 * (x + 1))
  (h₃ : ∀ x : ℝ, x ≥ 1 → f (x + 1) = (f x)^2/x - 1/x) :
  ∀ x : ℝ, x ≥ 1 → f x = x + 1 :=
by
  intro x hx
  sorry

end unique_function_solution_l97_97283


namespace fewer_vip_tickets_sold_l97_97757

-- Definitions based on the conditions
variables (V G : ℕ)
def tickets_sold := V + G = 320
def total_cost := 40 * V + 10 * G = 7500

-- The main statement to prove
theorem fewer_vip_tickets_sold :
  tickets_sold V G → total_cost V G → G - V = 34 := 
by
  intros h1 h2
  sorry

end fewer_vip_tickets_sold_l97_97757


namespace symmetric_point_yOz_l97_97159

-- Given point A in 3D Cartesian system
def A : ℝ × ℝ × ℝ := (1, -3, 5)

-- Plane yOz where x = 0
def symmetric_yOz (point : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := point
  (-x, y, z)

-- Proof statement (without the actual proof)
theorem symmetric_point_yOz : symmetric_yOz A = (-1, -3, 5) :=
by sorry

end symmetric_point_yOz_l97_97159


namespace complement_intersection_l97_97220

-- Define the universal set U and sets A, B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Define the intersection of A and B
def A_inter_B : Set ℕ := {x ∈ A | x ∈ B}

-- Define the complement of A_inter_B in U
def complement_U_A_inter_B : Set ℕ := {x ∈ U | x ∉ A_inter_B}

-- Prove that the complement of the intersection of A and B in U is {1, 4, 5}
theorem complement_intersection :
  complement_U_A_inter_B = {1, 4, 5} :=
by
  sorry

end complement_intersection_l97_97220


namespace range_of_x_l97_97002

theorem range_of_x (x : ℝ) (h : Real.log (x - 1) < 1) : 1 < x ∧ x < Real.exp 1 + 1 :=
by
  sorry

end range_of_x_l97_97002


namespace john_avg_speed_last_30_minutes_l97_97316

open Real

/-- John drove 160 miles in 120 minutes. His average speed during the first
30 minutes was 55 mph, during the second 30 minutes was 75 mph, and during
the third 30 minutes was 60 mph. Prove that his average speed during the
last 30 minutes was 130 mph. -/
theorem john_avg_speed_last_30_minutes (total_distance : ℝ) (total_time_minutes : ℝ)
  (speed_1 : ℝ) (speed_2 : ℝ) (speed_3 : ℝ) (speed_4 : ℝ) :
  total_distance = 160 →
  total_time_minutes = 120 →
  speed_1 = 55 →
  speed_2 = 75 →
  speed_3 = 60 →
  (speed_1 + speed_2 + speed_3 + speed_4) / 4 = total_distance / (total_time_minutes / 60) →
  speed_4 = 130 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end john_avg_speed_last_30_minutes_l97_97316


namespace exponential_inequality_l97_97170

theorem exponential_inequality (a x1 x2 : ℝ) (h1 : 1 < a) (h2 : x1 < x2) :
  |a ^ ((1 / 2) * (x1 + x2)) - a ^ x1| < |a ^ x2 - a ^ ((1 / 2) * (x1 + x2))| :=
by
  sorry

end exponential_inequality_l97_97170


namespace factor_expression_l97_97792

theorem factor_expression (b : ℝ) : 
  (8 * b^4 - 100 * b^3 + 18) - (3 * b^4 - 11 * b^3 + 18) = b^3 * (5 * b - 89) :=
by
  sorry

end factor_expression_l97_97792


namespace problem_1_problem_2_problem_3_l97_97688

noncomputable def f (a x : ℝ) : ℝ := a^(x-1)

theorem problem_1 (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) :
  f a 3 = 4 → a = 2 :=
sorry

theorem problem_2 (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) :
  f a (Real.log a) = 100 → (a = 100 ∨ a = 1 / 10) :=
sorry

theorem problem_3 (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) :
  (a > 1 → f a (Real.log (1 / 100)) > f a (-2.1)) ∧
  (0 < a ∧ a < 1 → f a (Real.log (1 / 100)) < f a (-2.1)) :=
sorry

end problem_1_problem_2_problem_3_l97_97688


namespace find_tangency_segments_equal_l97_97385

-- Conditions of the problem as a theorem statement
theorem find_tangency_segments_equal (AB BC CD DA : ℝ) (x y : ℝ)
    (h1 : AB = 80)
    (h2 : BC = 140)
    (h3 : CD = 100)
    (h4 : DA = 120)
    (h5 : x + y = CD)
    (tangency_property : |x - y| = 0) :
  |x - y| = 0 :=
sorry

end find_tangency_segments_equal_l97_97385


namespace oranges_per_box_calculation_l97_97902

def total_oranges : ℕ := 2650
def total_boxes : ℕ := 265

theorem oranges_per_box_calculation (h : total_oranges % total_boxes = 0) : total_oranges / total_boxes = 10 :=
by {
  sorry
}

end oranges_per_box_calculation_l97_97902


namespace floor_neg_seven_over_four_l97_97253

theorem floor_neg_seven_over_four : floor (-7 / 4) = -2 :=
by
  sorry

end floor_neg_seven_over_four_l97_97253


namespace Yoongi_score_is_53_l97_97400

-- Define the scores of the three students
variables (score_Yoongi score_Eunji score_Yuna : ℕ)

-- Define the conditions given in the problem
axiom Yoongi_Eunji : score_Eunji = score_Yoongi - 25
axiom Eunji_Yuna  : score_Yuna = score_Eunji - 20
axiom Yuna_score  : score_Yuna = 8

theorem Yoongi_score_is_53 : score_Yoongi = 53 := by
  sorry

end Yoongi_score_is_53_l97_97400


namespace geometric_progression_condition_l97_97543

theorem geometric_progression_condition (a b c d : ℝ) :
  (∃ r : ℝ, (b = a * r ∨ b = a * -r) ∧
             (c = a * r^2 ∨ c = a * (-r)^2) ∧
             (d = a * r^3 ∨ d = a * (-r)^3) ∧
             (a = b / r ∨ a = b / -r) ∧
             (b = c / r ∨ b = c / -r) ∧
             (c = d / r ∨ c = d / -r) ∧
             (d = a / r ∨ d = a / -r)) ↔
  (a = b ∨ a = -b) ∧ (a = c ∨ a = -c) ∧ (a = d ∨ a = -d) := sorry

end geometric_progression_condition_l97_97543


namespace smallest_positive_integer_ends_6996_l97_97318

theorem smallest_positive_integer_ends_6996 :
  ∃ m : ℕ, (m % 4 = 0 ∧ m % 9 = 0 ∧ ∀ d ∈ m.digits 10, d = 6 ∨ d = 9 ∧ m.digits 10 ∩ {6, 9} ≠ ∅ ∧ m % 10000 = 6996) :=
sorry

end smallest_positive_integer_ends_6996_l97_97318


namespace simplest_square_root_l97_97214

theorem simplest_square_root :
  let a := Real.sqrt (1 / 2)
  let b := Real.sqrt 11
  let c := Real.sqrt 27
  let d := Real.sqrt 0.3
  (b < a ∧ b < c ∧ b < d) :=
sorry

end simplest_square_root_l97_97214


namespace ratio_proof_l97_97948

noncomputable def ratio_of_segment_lengths (a b : ℝ) (points : Finset (ℝ × ℝ)) : Prop :=
  points.card = 5 ∧
  ∃ (dists : Finset ℝ), 
    dists = {a, a, a, a, a, b, 3 * a} ∧
    ∀ (p1 p2 : ℝ × ℝ), p1 ∈ points → p2 ∈ points → p1 ≠ p2 →
      (dist p1 p2 ∈ dists)

theorem ratio_proof (a b : ℝ) (points : Finset (ℝ × ℝ)) (h : ratio_of_segment_lengths a b points) : 
  b / a = 2.8 :=
sorry

end ratio_proof_l97_97948


namespace cos_product_identity_l97_97547

theorem cos_product_identity :
  (Real.cos (20 * Real.pi / 180)) * (Real.cos (40 * Real.pi / 180)) *
  (Real.cos (60 * Real.pi / 180)) * (Real.cos (80 * Real.pi / 180)) = 1 / 16 := 
by
  sorry

end cos_product_identity_l97_97547


namespace find_interest_rate_l97_97089

theorem find_interest_rate 
    (P : ℝ) (T : ℝ) (known_rate : ℝ) (diff : ℝ) (R : ℝ) :
    P = 7000 → T = 2 → known_rate = 0.18 → diff = 840 → (P * known_rate * T - (P * (R/100) * T) = diff) → R = 12 :=
by
  intros P_eq T_eq kr_eq diff_eq interest_eq
  simp only [P_eq, T_eq, kr_eq, diff_eq] at interest_eq
-- Solving equation is not required
  sorry

end find_interest_rate_l97_97089


namespace total_points_scored_l97_97718

theorem total_points_scored (points_per_round : ℕ) (rounds : ℕ) (h1 : points_per_round = 42) (h2 : rounds = 2) : 
  points_per_round * rounds = 84 :=
by
  sorry

end total_points_scored_l97_97718


namespace new_rectangle_perimeters_l97_97750

theorem new_rectangle_perimeters {l w : ℕ} (h_l : l = 4) (h_w : w = 2) :
  (∃ P, P = 2 * (8 + 2) ∨ P = 2 * (4 + 4)) ∧ (P = 20 ∨ P = 16) :=
by
  sorry

end new_rectangle_perimeters_l97_97750


namespace integer_count_between_sqrt8_and_sqrt78_l97_97430

theorem integer_count_between_sqrt8_and_sqrt78 :
  ∃ (n : ℕ), n = 6 ∧ ∀ (x : ℤ), (⌈Real.sqrt 8⌉ ≤ x ∧ x ≤ ⌊Real.sqrt 78⌋) ↔ (3 ≤ x ∧ x ≤ 8) := by
  sorry

end integer_count_between_sqrt8_and_sqrt78_l97_97430


namespace parabola_coefficient_c_l97_97778

def parabola (b c x : ℝ) : ℝ := x^2 + b * x + c

theorem parabola_coefficient_c (b c : ℝ) (h1 : parabola b c 1 = -1) (h2 : parabola b c 3 = 9) : 
  c = -3 := 
by
  sorry

end parabola_coefficient_c_l97_97778


namespace find_x_for_y_equals_six_l97_97083

variable (x y k : ℚ)

-- Conditions
def varies_inversely_as_square := x = k / y^2
def initial_condition := (y = 3 ∧ x = 1)

-- Problem Statement
theorem find_x_for_y_equals_six (h₁ : varies_inversely_as_square x y k) (h₂ : initial_condition x y) :
  ∃ k, (k = 9 ∧ x = k / 6^2 ∧ x = 1 / 4) :=
sorry

end find_x_for_y_equals_six_l97_97083


namespace correct_average_is_15_l97_97082

theorem correct_average_is_15 (n incorrect_avg correct_num wrong_num : ℕ) 
  (h1 : n = 10) (h2 : incorrect_avg = 14) (h3 : correct_num = 36) (h4 : wrong_num = 26) : 
  (incorrect_avg * n + (correct_num - wrong_num)) / n = 15 := 
by 
  sorry

end correct_average_is_15_l97_97082


namespace exists_k_such_that_n_eq_k_2010_l97_97133

theorem exists_k_such_that_n_eq_k_2010 (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n)
  (h : m * n ∣ m ^ 2010 + n ^ 2010 + n) : ∃ k : ℕ, 0 < k ∧ n = k ^ 2010 := by
  sorry

end exists_k_such_that_n_eq_k_2010_l97_97133


namespace solution_set_quadratic_inequality_l97_97119

theorem solution_set_quadratic_inequality :
  {x : ℝ | (x^2 - 3*x + 2) < 0} = {x : ℝ | 1 < x ∧ x < 2} :=
sorry

end solution_set_quadratic_inequality_l97_97119


namespace carrot_price_l97_97661

variables (total_tomatoes : ℕ) (total_carrots : ℕ) (price_per_tomato : ℝ) (total_revenue : ℝ)

theorem carrot_price :
  total_tomatoes = 200 →
  total_carrots = 350 →
  price_per_tomato = 1 →
  total_revenue = 725 →
  (total_revenue - total_tomatoes * price_per_tomato) / total_carrots = 1.5 :=
by
  intros h1 h2 h3 h4
  sorry

end carrot_price_l97_97661


namespace neg_mul_reverses_inequality_l97_97412

theorem neg_mul_reverses_inequality (a b : ℝ) (h : a < b) : -3 * a > -3 * b :=
  sorry

end neg_mul_reverses_inequality_l97_97412


namespace geometric_sequence_second_term_value_l97_97866

theorem geometric_sequence_second_term_value
  (a : ℝ) 
  (r : ℝ) 
  (h1 : 30 * r = a) 
  (h2 : a * r = 7 / 4) 
  (h3 : 0 < a) : 
  a = 7.5 := 
sorry

end geometric_sequence_second_term_value_l97_97866


namespace floor_of_neg_seven_fourths_l97_97276

theorem floor_of_neg_seven_fourths : (Int.floor (-7/4 : ℚ)) = -2 :=
by
  sorry

end floor_of_neg_seven_fourths_l97_97276


namespace sum_of_integers_l97_97347

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 120) : x + y = 15 :=
by {
    sorry
}

end sum_of_integers_l97_97347


namespace ratio_of_distances_l97_97204

theorem ratio_of_distances 
  (x : ℝ) -- distance walked by the first lady
  (h1 : 4 + x = 12) -- combined total distance walked is 12 miles 
  (h2 : ¬(x < 0)) -- distance cannot be negative
  (h3 : 4 ≠ 0) : -- the second lady walked 4 miles which is not zero
  x / 4 = 2 := -- the ratio of the distances is 2
by
  sorry

end ratio_of_distances_l97_97204


namespace find_m_value_l97_97552

theorem find_m_value :
  ∃ m : ℕ, 144^5 + 121^5 + 95^5 + 30^5 = m^5 ∧ m = 159 := by
  use 159
  sorry

end find_m_value_l97_97552


namespace area_of_larger_square_l97_97100

theorem area_of_larger_square (side_length : ℕ) (num_squares : ℕ)
  (h₁ : side_length = 2)
  (h₂ : num_squares = 8) : 
  (num_squares * side_length^2) = 32 :=
by
  sorry

end area_of_larger_square_l97_97100


namespace fraction_habitable_l97_97698

theorem fraction_habitable : (1 / 3) * (1 / 3) = 1 / 9 := 
by 
  sorry

end fraction_habitable_l97_97698


namespace line_circle_separation_l97_97414

theorem line_circle_separation (a b : ℝ) (h : a^2 + b^2 < 1) :
    let d := 1 / (Real.sqrt (a^2 + b^2))
    d > 1 := by
    sorry

end line_circle_separation_l97_97414


namespace problem_condition_l97_97818

noncomputable def m : ℤ := sorry
noncomputable def n : ℤ := sorry
noncomputable def x : ℤ := sorry
noncomputable def a : ℤ := 0
noncomputable def b : ℤ := -m + n

theorem problem_condition 
  (h1 : m ≠ 0)
  (h2 : n ≠ 0)
  (h3 : m ≠ n)
  (h4 : (x + m)^2 - (x^2 + n^2) = (m - n)^2) :
  x = a * m + b * n :=
sorry

end problem_condition_l97_97818


namespace factor_poly_find_abs_l97_97438

theorem factor_poly_find_abs {
  p q : ℤ
} (h1 : 3 * (-2)^3 - p * (-2) + q = 0) 
  (h2 : 3 * (3)^3 - p * (3) + q = 0) :
  |3 * p - 2 * q| = 99 := sorry

end factor_poly_find_abs_l97_97438


namespace units_digit_G_2000_l97_97174

-- Define the sequence G
def G (n : ℕ) : ℕ := 2 ^ (2 ^ n) + 5 ^ (5 ^ n)

-- The main goal is to show that the units digit of G 2000 is 1
theorem units_digit_G_2000 : (G 2000) % 10 = 1 :=
by
  sorry

end units_digit_G_2000_l97_97174


namespace surface_area_of_solid_l97_97575

noncomputable def solid_surface_area (r : ℝ) (h : ℝ) : ℝ :=
  2 * Real.pi * r * h

theorem surface_area_of_solid : solid_surface_area 1 3 = 6 * Real.pi := by
  sorry

end surface_area_of_solid_l97_97575


namespace nearest_edge_of_picture_l97_97643

theorem nearest_edge_of_picture
    (wall_width : ℝ) (picture_width : ℝ) (offset : ℝ) (x : ℝ)
    (hw : wall_width = 25) (hp : picture_width = 5) (ho : offset = 2) :
    x + (picture_width / 2) + offset = wall_width / 2 →
    x = 8 :=
by
  intros h
  sorry

end nearest_edge_of_picture_l97_97643


namespace total_time_over_weekend_l97_97162

def time_per_round : ℕ := 30
def rounds_saturday : ℕ := 11
def rounds_sunday : ℕ := 15

theorem total_time_over_weekend :
  (rounds_saturday * time_per_round) + (rounds_sunday * time_per_round) = 780 :=
by
  -- This is where the proof would go, but it is omitted as per instructions.
  sorry

end total_time_over_weekend_l97_97162


namespace expected_value_is_90_l97_97092

noncomputable def expected_value_coins_heads : ℕ :=
  let nickel := 5
  let quarter := 25
  let half_dollar := 50
  let dollar := 100
  1/2 * (nickel + quarter + half_dollar + dollar)

theorem expected_value_is_90 : expected_value_coins_heads = 90 := by
  sorry

end expected_value_is_90_l97_97092


namespace percent_increase_salary_l97_97892

theorem percent_increase_salary (new_salary increase : ℝ) (h_new_salary : new_salary = 90000) (h_increase : increase = 25000) :
  (increase / (new_salary - increase)) * 100 = 38.46 := by
  -- Given values
  have h1 : new_salary = 90000 := h_new_salary
  have h2 : increase = 25000 := h_increase
  -- Compute original salary
  let original_salary : ℝ := new_salary - increase
  -- Compute percent increase
  let percent_increase : ℝ := (increase / original_salary) * 100
  -- Show that the percent increase is 38.46
  have h3 : percent_increase = 38.46 := sorry
  exact h3

end percent_increase_salary_l97_97892


namespace original_group_size_l97_97512

theorem original_group_size (M : ℕ) 
  (h1 : ∀ work_done_by_one, work_done_by_one = 1 / (6 * M))
  (h2 : ∀ work_done_by_one, work_done_by_one = 1 / (12 * (M - 4))) : 
  M = 8 :=
by
  sorry

end original_group_size_l97_97512


namespace binary_addition_to_decimal_l97_97621

theorem binary_addition_to_decimal : (2^8 + 2^7 + 2^6 + 2^5 + 2^4 + 2^3 + 2^2 + 2^1 + 2^0)
                                     + (2^5 + 2^4 + 2^3 + 2^2) = 571 := by
  sorry

end binary_addition_to_decimal_l97_97621


namespace solution_set_of_inequality_l97_97869

theorem solution_set_of_inequality (x : ℝ) : 
  (|x+1| - |x-4| > 3) ↔ x > 3 :=
sorry

end solution_set_of_inequality_l97_97869


namespace pure_imaginary_condition_l97_97319

def z1 : ℂ := 3 - 2 * Complex.I
def z2 (m : ℝ) : ℂ := 1 + m * Complex.I

theorem pure_imaginary_condition (m : ℝ) : z1 * z2 m ∈ {z : ℂ | z.re = 0} ↔ m = -3 / 2 := by
  sorry

end pure_imaginary_condition_l97_97319


namespace g_of_3_over_8_l97_97743

def g (x : ℝ) : ℝ := sorry

theorem g_of_3_over_8 :
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g 0 = 0) ∧
    (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 1 → g x ≤ g y) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g (x / 4) = (g x) / 3) →
    g (3 / 8) = 2 / 9 :=
sorry

end g_of_3_over_8_l97_97743


namespace ellipse_parabola_intersection_l97_97699

open Real

theorem ellipse_parabola_intersection (a : ℝ) :
  (∃ (x y : ℝ), x^2 + 4*(y - a)^2 = 4 ∧ x^2 = 2*y) ↔ (-1 ≤ a ∧ a ≤ 17 / 8) :=
by
  sorry

end ellipse_parabola_intersection_l97_97699


namespace rachel_more_than_adam_l97_97180

variable (R J A : ℕ)

def condition1 := R = 75
def condition2 := R = J - 6
def condition3 := R > A
def condition4 := (R + J + A) / 3 = 72

theorem rachel_more_than_adam
  (h1 : condition1 R)
  (h2 : condition2 R J)
  (h3 : condition3 R A)
  (h4 : condition4 R J A) : 
  R - A = 15 := 
by
  sorry

end rachel_more_than_adam_l97_97180


namespace bob_time_improvement_l97_97393

def time_improvement_percent (bob_time sister_time improvement_time : ℕ) : ℕ :=
  ((improvement_time * 100) / bob_time)

theorem bob_time_improvement : 
  ∀ (bob_time sister_time : ℕ), bob_time = 640 → sister_time = 608 → 
  time_improvement_percent bob_time sister_time (bob_time - sister_time) = 5 :=
by
  intros bob_time sister_time h_bob h_sister
  rw [h_bob, h_sister]
  sorry

end bob_time_improvement_l97_97393


namespace A_more_than_B_l97_97786

noncomputable def proportion := (5, 3, 2, 3)
def C_share := 1000
def parts := 2
noncomputable def part_value := C_share / parts
noncomputable def A_share := part_value * 5
noncomputable def B_share := part_value * 3

theorem A_more_than_B : A_share - B_share = 1000 := by
  sorry

end A_more_than_B_l97_97786


namespace tom_has_18_apples_l97_97653

-- Definitions based on conditions
def phillip_apples : ℕ := 40
def ben_apples : ℕ := phillip_apples + 8
def tom_apples : ℕ := (3 * ben_apples) / 8

-- Theorem stating Tom has 18 apples given the conditions
theorem tom_has_18_apples : tom_apples = 18 :=
sorry

end tom_has_18_apples_l97_97653


namespace eq_of_symmetric_translation_l97_97908

noncomputable def parabola (x : ℝ) : ℝ := 2 * x^2 - 4 * x - 5

noncomputable def translate_left (f : ℝ → ℝ) (k : ℝ) (x : ℝ) : ℝ := f (x + k)

noncomputable def translate_up (g : ℝ → ℝ) (k : ℝ) (x : ℝ) : ℝ := g x + k

noncomputable def translate_parabola (x : ℝ) : ℝ := translate_up (translate_left parabola 3) 2 x

noncomputable def symmetric_parabola (h : ℝ → ℝ) (x : ℝ) : ℝ := h (-x)

theorem eq_of_symmetric_translation :
  symmetric_parabola translate_parabola x = 2 * x^2 - 8 * x + 3 :=
by
  sorry

end eq_of_symmetric_translation_l97_97908


namespace fourth_student_in_sample_l97_97832

def sample_interval (total_students : ℕ) (sample_size : ℕ) : ℕ :=
  total_students / sample_size

def in_sample (student_number : ℕ) (start : ℕ) (interval : ℕ) (n : ℕ) : Prop :=
  student_number = start + n * interval

theorem fourth_student_in_sample :
  ∀ (total_students sample_size : ℕ) (s1 s2 s3 : ℕ),
    total_students = 52 →
    sample_size = 4 →
    s1 = 7 →
    s2 = 33 →
    s3 = 46 →
    ∃ s4, in_sample s4 s1 (sample_interval total_students sample_size) 1 ∧
           in_sample s2 s1 (sample_interval total_students sample_size) 2 ∧
           in_sample s3 s1 (sample_interval total_students sample_size) 3 ∧
           s4 = 20 := 
by
  sorry

end fourth_student_in_sample_l97_97832


namespace evaluate_expression_l97_97765

theorem evaluate_expression :
  (2 * 10^3)^3 = 8 * 10^9 :=
by
  sorry

end evaluate_expression_l97_97765


namespace geometric_sequence_b_value_l97_97940

theorem geometric_sequence_b_value (b : ℝ) (h1 : 25 * b = b^2) (h2 : b * (1 / 4) = b / 4) :
  b = 5 / 2 :=
sorry

end geometric_sequence_b_value_l97_97940


namespace envelope_weight_l97_97390

theorem envelope_weight :
  (7.225 * 1000) / 850 = 8.5 :=
by
  sorry

end envelope_weight_l97_97390


namespace number_of_non_attacking_rook_placements_l97_97824

theorem number_of_non_attacking_rook_placements : 
  let rows := 4
  let columns := 5
  let rooks := 3
  (choose rows rooks) * (choose columns rooks) * (factorial rooks) = 240 := by
  sorry

end number_of_non_attacking_rook_placements_l97_97824


namespace beetles_eaten_per_day_l97_97249
-- Import the Mathlib library

-- Declare the conditions as constants
def bird_eats_beetles_per_day : Nat := 12
def snake_eats_birds_per_day : Nat := 3
def jaguar_eats_snakes_per_day : Nat := 5
def number_of_jaguars : Nat := 6

-- Define the theorem and provide the expected proof
theorem beetles_eaten_per_day :
  12 * (3 * (5 * 6)) = 1080 := by
  sorry

end beetles_eaten_per_day_l97_97249


namespace sum_denominators_l97_97363

theorem sum_denominators (a b: ℕ) (h_coprime : Nat.gcd a b = 1) :
  (3:ℚ) / (5 * b) + (2:ℚ) / (9 * b) + (4:ℚ) / (15 * b) = 28 / 45 →
  5 * b + 9 * b + 15 * b = 203 :=
by
  sorry

end sum_denominators_l97_97363


namespace acute_angle_alpha_range_l97_97561

theorem acute_angle_alpha_range (x : ℝ) (α : ℝ) (h1 : 0 < x) (h2 : x < 90) (h3 : α = 180 - 2 * x) : 0 < α ∧ α < 180 :=
by
  sorry

end acute_angle_alpha_range_l97_97561


namespace backpack_original_price_l97_97712

-- Define original price of a ring-binder
def original_ring_binder_price : ℕ := 20

-- Define the number of ring-binders bought
def number_of_ring_binders : ℕ := 3

-- Define the new price increase for the backpack
def backpack_price_increase : ℕ := 5

-- Define the new price decrease for the ring-binder
def ring_binder_price_decrease : ℕ := 2

-- Define the total amount spent
def total_amount_spent : ℕ := 109

-- Define the original price of the backpack variable
variable (B : ℕ)

-- Theorem statement: under these conditions, the original price of the backpack must be 50
theorem backpack_original_price :
  (B + backpack_price_increase) + ((original_ring_binder_price - ring_binder_price_decrease) * number_of_ring_binders) = total_amount_spent ↔ B = 50 :=
by 
  sorry

end backpack_original_price_l97_97712


namespace xyz_inequality_l97_97327

theorem xyz_inequality {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z))) + (y^3 / ((1 + z) * (1 + x))) + (z^3 / ((1 + x) * (1 + y))) ≥ (3/4) :=
sorry

end xyz_inequality_l97_97327


namespace fergus_entry_exit_l97_97090

theorem fergus_entry_exit (n : ℕ) (hn : n = 8) : 
  n * (n - 1) = 56 := 
by
  sorry

end fergus_entry_exit_l97_97090


namespace finite_monic_poly_unit_circle_roots_roots_roots_of_unity_l97_97376

noncomputable section

-- Part (a)
theorem finite_monic_poly_unit_circle_roots (n : ℕ) (h : 0 < n) : 
  {P : Polynomial ℤ // P.monic ∧ P.degree = n ∧ ∀ x, P.isRoot x → ∥x∥ = 1}.finite :=
  sorry

-- Part (b)
theorem roots_roots_of_unity (P : Polynomial ℤ) (hmonic : P.monic) (hroots : ∀ x, P.isRoot x → ∥x∥ = 1) :
  ∃ m : ℕ, 0 < m ∧ ∀ x, P.isRoot x → x^m = 1 :=
  sorry

end finite_monic_poly_unit_circle_roots_roots_roots_of_unity_l97_97376


namespace system1_solution_system2_solution_l97_97470

-- Definition and proof for System (1)
theorem system1_solution (x y : ℝ) (h1 : x - y = 2) (h2 : 2 * x + y = 7) : x = 3 ∧ y = 1 := 
by 
  sorry

-- Definition and proof for System (2)
theorem system2_solution (x y : ℝ) (h1 : x - 2 * y = 3) (h2 : (1 / 2) * x + (3 / 4) * y = 13 / 4) : x = 5 ∧ y = 1 :=
by 
  sorry

end system1_solution_system2_solution_l97_97470


namespace find_n_l97_97625

theorem find_n : ∃ n : ℕ, 
  (n % 4 = 1) ∧ (n % 5 = 2) ∧ (n % 6 = 3) ∧ (n = 57) := by
  sorry

end find_n_l97_97625


namespace arithmetic_fraction_subtraction_l97_97207

theorem arithmetic_fraction_subtraction :
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) = 9 / 20 :=
by
  sorry

end arithmetic_fraction_subtraction_l97_97207


namespace fenced_area_with_cutout_l97_97062

theorem fenced_area_with_cutout :
  let rectangle_length : ℕ := 20
  let rectangle_width : ℕ := 16
  let cutout_length : ℕ := 4
  let cutout_width : ℕ := 4
  rectangle_length * rectangle_width - cutout_length * cutout_width = 304 := by
  sorry

end fenced_area_with_cutout_l97_97062


namespace five_pow_n_minus_one_not_divisible_by_four_pow_n_minus_one_l97_97791

theorem five_pow_n_minus_one_not_divisible_by_four_pow_n_minus_one (n : ℕ) (hn : n > 0) : ¬ (4 ^ n - 1 ∣ 5 ^ n - 1) :=
sorry

end five_pow_n_minus_one_not_divisible_by_four_pow_n_minus_one_l97_97791


namespace minimum_value_of_f_range_of_t_l97_97820

noncomputable def f (x : ℝ) : ℝ := x + 9 / (x - 3)

theorem minimum_value_of_f :
  (∃ x > 3, f x = 9) :=
by
  sorry

theorem range_of_t (t : ℝ) :
  (∀ x > 3, f x ≥ t / (t + 1) + 7) ↔ (t ≤ -2 ∨ t > -1) :=
by
  sorry

end minimum_value_of_f_range_of_t_l97_97820


namespace solve_for_x_l97_97557

theorem solve_for_x (x : ℝ) (h : |3990 * x + 1995| = 1995) : x = 0 ∨ x = -1 :=
by
  sorry

end solve_for_x_l97_97557


namespace translated_vector_coordinates_l97_97562

variables {A B a : Point}

def Point := (ℝ × ℝ)

def vector (P Q : Point) : Point := (Q.1 - P.1, Q.2 - P.2)
def translate (v a : Point) : Point := (v.1 + a.1, v.2 + a.2)

theorem translated_vector_coordinates : 
  let A := (3, 7)
  let B := (5, 2)
  let a := (1, 2)
  translate (vector A B) a = (2, -5)  :=
by
  sorry

end translated_vector_coordinates_l97_97562


namespace right_angled_triangles_l97_97245

theorem right_angled_triangles (x y z : ℕ) : (x - 6) * (y - 6) = 18 ∧ (x^2 + y^2 = z^2)
  → (3 * (x + y + z) = x * y) :=
sorry

end right_angled_triangles_l97_97245


namespace find_n_l97_97961

theorem find_n : 
  (43^2 = 1849) → 
  (44^2 = 1936) → 
  (45^2 = 2025) → 
  (46^2 = 2116) → 
  ∃ n : ℤ, (n < Real.sqrt 2023) ∧ (Real.sqrt 2023 < n+1) ∧ n = 44 :=
by
  intros h1 h2 h3 h4
  existsi (44:ℤ)
  split
  sorry -- Proof of n < sqrt(2023)
  split
  sorry -- Proof of sqrt(2023) < n+1
  refl -- Proof of n = 44

end find_n_l97_97961


namespace find_a_l97_97734

theorem find_a (a : ℝ) (p : ℕ → ℝ) (h : ∀ k, k = 1 ∨ k = 2 ∨ k = 3 → p k = a * (1 / 2) ^ k)
  (prob_sum : a * (1 / 2 + (1 / 2) ^ 2 + (1 / 2) ^ 3) = 1) : a = 8 / 7 :=
sorry

end find_a_l97_97734


namespace area_of_rectangle_l97_97893

theorem area_of_rectangle (side radius length breadth : ℕ) (h1 : side^2 = 784) (h2 : radius = side) (h3 : length = radius / 4) (h4 : breadth = 5) : length * breadth = 35 :=
by
  -- proof to be filled here
  sorry

end area_of_rectangle_l97_97893


namespace cube_sum_l97_97044

theorem cube_sum (a b : ℝ) (h : a / (1 + b) + b / (1 + a) = 1) : a^3 + b^3 = a + b := by
  sorry

end cube_sum_l97_97044


namespace total_beetles_eaten_each_day_l97_97250

-- Definitions from the conditions
def birds_eat_per_day : ℕ := 12
def snakes_eat_per_day : ℕ := 3
def jaguars_eat_per_day : ℕ := 5
def number_of_jaguars : ℕ := 6

-- Theorem statement
theorem total_beetles_eaten_each_day :
  (number_of_jaguars * jaguars_eat_per_day) * snakes_eat_per_day * birds_eat_per_day = 1080 :=
by sorry

end total_beetles_eaten_each_day_l97_97250


namespace total_crayons_l97_97871

def original_crayons := 41
def added_crayons := 12

theorem total_crayons : original_crayons + added_crayons = 53 := by
  sorry

end total_crayons_l97_97871


namespace trader_marked_price_percentage_above_cost_price_l97_97232

theorem trader_marked_price_percentage_above_cost_price 
  (CP MP SP : ℝ) 
  (discount loss : ℝ)
  (h_discount : discount = 0.07857142857142857)
  (h_loss : loss = 0.01)
  (h_SP_discount : SP = MP * (1 - discount))
  (h_SP_loss : SP = CP * (1 - loss)) :
  (MP / CP - 1) * 100 = 7.4285714285714 := 
sorry

end trader_marked_price_percentage_above_cost_price_l97_97232


namespace distance_run_l97_97772

theorem distance_run (D : ℝ) (A_time : ℝ) (B_time : ℝ) (A_beats_B : ℝ) : 
  A_time = 90 ∧ B_time = 180 ∧ A_beats_B = 2250 → D = 2250 :=
by
  sorry

end distance_run_l97_97772


namespace distance_from_center_to_plane_correct_l97_97607

noncomputable def distance_from_center_to_plane (O A B C : ℝ × ℝ × ℝ) (radius : ℝ) (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let K := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let R := (a * b * c) / (4 * K)
  let OD := Real.sqrt (radius^2 - R^2)
  OD

theorem distance_from_center_to_plane_correct (O A B C : ℝ × ℝ × ℝ) :
  (dist O A = 20) →
  (dist O B = 20) →
  (dist O C = 20) →
  (dist A B = 13) →
  (dist B C = 14) →
  (dist C A = 15) →
  let m := 15
  let n := 95
  let k := 8
  m + n + k = 118 := by
  sorry

end distance_from_center_to_plane_correct_l97_97607


namespace smallest_k_divides_polynomial_l97_97410

theorem smallest_k_divides_polynomial :
  ∃ k : ℕ, 0 < k ∧ (∀ z : ℂ, (z^10 + z^9 + z^8 + z^6 + z^5 + z^4 + z + 1) ∣ (z^k - 1)) ∧ k = 84 :=
by
  sorry

end smallest_k_divides_polynomial_l97_97410


namespace line_parallel_xaxis_l97_97735

theorem line_parallel_xaxis (x y : ℝ) : y = 2 ↔ (∃ a b : ℝ, a = 4 ∧ b = 2 ∧ y = 2) :=
by 
  sorry

end line_parallel_xaxis_l97_97735


namespace find_moles_of_NaCl_l97_97408

-- Define the chemical reaction as an equation
def chemical_reaction (NaCl KNO3 NaNO3 KCl : ℕ) : Prop :=
  NaCl + KNO3 = NaNO3 + KCl

-- Define the problem conditions
def problem_conditions (naCl : ℕ) : Prop :=
  ∃ (kno3 naNo3 kcl : ℕ),
    kno3 = 3 ∧
    naNo3 = 3 ∧
    chemical_reaction naCl kno3 naNo3 kcl

-- Define the goal statement
theorem find_moles_of_NaCl (naCl : ℕ) : problem_conditions naCl → naCl = 3 :=
by
  sorry -- proof to be filled in later

end find_moles_of_NaCl_l97_97408


namespace range_of_fraction_l97_97967

variable {x y : ℝ}

-- Condition given in the problem
def equation (x y : ℝ) : Prop := x + 2 * y - 6 = 0

-- The range condition for x
def x_range (x : ℝ) : Prop := 0 < x ∧ x < 3

-- The corresponding theorem statement
theorem range_of_fraction (h_eq : equation x y) (h_x_range : x_range x) :
  ∃ a b : ℝ, (a < 1 ∧ 10 < b) ∧ (a, b) = (1, 10) ∧
  ∀ k : ℝ, k = (x + 2) / (y - 1) → 1 < k ∧ k < 10 :=
sorry

end range_of_fraction_l97_97967


namespace lily_has_26_dollars_left_for_coffee_l97_97046

-- Define the initial amount of money Lily has
def initialMoney : ℕ := 60

-- Define the costs of items
def celeryCost : ℕ := 5
def cerealCost : ℕ := 12 / 2
def breadCost : ℕ := 8
def milkCost : ℕ := 10 * 9 / 10
def potatoCostEach : ℕ := 1
def numberOfPotatoes : ℕ := 6
def totalPotatoCost : ℕ := potatoCostEach * numberOfPotatoes

-- Define the total amount spent on the items
def totalSpent : ℕ := celeryCost + cerealCost + breadCost + milkCost + totalPotatoCost

-- Define the amount left for coffee
def amountLeftForCoffee : ℕ := initialMoney - totalSpent

-- The theorem to prove
theorem lily_has_26_dollars_left_for_coffee :
  amountLeftForCoffee = 26 := by
  sorry

end lily_has_26_dollars_left_for_coffee_l97_97046


namespace most_stable_performance_l97_97900

theorem most_stable_performance 
    (s_A s_B s_C s_D : ℝ)
    (hA : s_A = 1.5)
    (hB : s_B = 2.6)
    (hC : s_C = 1.7)
    (hD : s_D = 2.8)
    (mean_score : ∀ (x : ℝ), x = 88.5) :
    s_A < s_C ∧ s_C < s_B ∧ s_B < s_D := by
  sorry

end most_stable_performance_l97_97900


namespace count_linear_eqs_l97_97479

-- Define each equation as conditions
def eq1 (x y : ℝ) := 3 * x - y = 2
def eq2 (x : ℝ) := x + 1 / x + 2 = 0
def eq3 (x : ℝ) := x^2 - 2 * x - 3 = 0
def eq4 (x : ℝ) := x = 0
def eq5 (x : ℝ) := 3 * x - 1 ≥ 5
def eq6 (x : ℝ) := 1 / 2 * x = 1 / 2
def eq7 (x : ℝ) := (2 * x + 1) / 3 = 1 / 6 * x

-- Proof statement: there are exactly 3 linear equations
theorem count_linear_eqs : 
  (∃ x y, eq1 x y) ∧ eq4 0 ∧ (∃ x, eq6 x) ∧ (∃ x, eq7 x) ∧ 
  ¬ (∃ x, eq2 x) ∧ ¬ (∃ x, eq3 x) ∧ ¬ (∃ x, eq5 x) → 
  3 = 3 :=
sorry

end count_linear_eqs_l97_97479


namespace floor_of_neg_seven_fourths_l97_97278

theorem floor_of_neg_seven_fourths : (Int.floor (-7/4 : ℚ)) = -2 :=
by
  sorry

end floor_of_neg_seven_fourths_l97_97278


namespace negation_of_universal_proposition_l97_97052

theorem negation_of_universal_proposition :
  (∀ x : ℝ, x^2 + 1 > 0) → ¬(∃ x : ℝ, x^2 + 1 ≤ 0) := sorry

end negation_of_universal_proposition_l97_97052


namespace surface_area_of_large_cube_correct_l97_97877

-- Definition of the surface area problem

def edge_length_of_small_cube := 3 -- centimeters
def number_of_small_cubes := 27
def surface_area_of_large_cube (edge_length_of_small_cube : ℕ) (number_of_small_cubes : ℕ) : ℕ :=
  let edge_length_of_large_cube := edge_length_of_small_cube * (number_of_small_cubes^(1/3))
  6 * edge_length_of_large_cube^2

theorem surface_area_of_large_cube_correct :
  surface_area_of_large_cube edge_length_of_small_cube number_of_small_cubes = 486 := by
  sorry

end surface_area_of_large_cube_correct_l97_97877


namespace regression_line_passes_through_center_l97_97690

-- Define the regression equation
def regression_eq (x : ℝ) : ℝ := 1.5 * x - 15

-- Define the condition of the sample center point
def sample_center (x_bar y_bar : ℝ) : Prop :=
  y_bar = regression_eq x_bar

-- The proof goal
theorem regression_line_passes_through_center (x_bar y_bar : ℝ) (h : sample_center x_bar y_bar) :
  y_bar = 1.5 * x_bar - 15 :=
by
  -- Using the given condition as hypothesis
  exact h

end regression_line_passes_through_center_l97_97690


namespace correct_factorization_option_B_l97_97885

-- Lean 4 statement to prove the correct factorization
theorem correct_factorization_option_B (x : ℝ) : 4 * x ^ 2 - 4 * x + 1 = (2 * x - 1) ^ 2 := by
  sorry

end correct_factorization_option_B_l97_97885


namespace grouping_count_l97_97029

theorem grouping_count (men women : ℕ) 
  (h_men : men = 4) (h_women : women = 5)
  (at_least_one_man_woman : ∀ (g1 g2 g3 : Finset (Fin 9)), 
    g1.card = 3 → g2.card = 3 → g3.card = 3 → g1 ∩ g2 = ∅ → g2 ∩ g3 = ∅ → g3 ∩ g1 = ∅ → 
    (g1 ∩ univ.filter (· < 4)).nonempty ∧ (g1 ∩ univ.filter (· ≥ 4)).nonempty ∧
    (g2 ∩ univ.filter (· < 4)).nonempty ∧ (g2 ∩ univ.filter (· ≥ 4)).nonempty ∧
    (g3 ∩ univ.filter (· < 4)).nonempty ∧ (g3 ∩ univ.filter (· ≥ 4)).nonempty) :
  (choose 4 1 * choose 5 2 * choose 3 1 * choose 3 2) / 2! = 180 :=
sorry

end grouping_count_l97_97029


namespace percentage_error_in_side_l97_97391

theorem percentage_error_in_side {S S' : ℝ}
  (hs : S > 0)
  (hs' : S' > S)
  (h_area_error : (S'^2 - S^2) / S^2 * 100 = 90.44) :
  ((S' - S) / S * 100) = 38 :=
by
  sorry

end percentage_error_in_side_l97_97391


namespace find_v_plus_z_l97_97361

variable (x u v w z : ℂ)
variable (y : ℂ)
variable (condition1 : y = 2)
variable (condition2 : w = -x - u)
variable (condition3 : x + y * Complex.I + u + v * Complex.I + w + z * Complex.I = -2 * Complex.I)

theorem find_v_plus_z : v + z = -4 :=
by
  have h1 : y = 2 := condition1
  have h2 : w = -x - u := condition2
  have h3 : x + y * Complex.I + u + v * Complex.I + w + z * Complex.I = -2 * Complex.I := condition3
  sorry

end find_v_plus_z_l97_97361


namespace pond_fish_approximation_l97_97013

noncomputable def total_number_of_fish
  (tagged_first: ℕ) (total_caught_second: ℕ) (tagged_second: ℕ) : ℕ :=
  (tagged_first * total_caught_second) / tagged_second

theorem pond_fish_approximation :
  total_number_of_fish 60 50 2 = 1500 :=
by
  -- calculation of the total number of fish based on given conditions
  sorry

end pond_fish_approximation_l97_97013


namespace box_contains_1600_calories_l97_97637

theorem box_contains_1600_calories :
  let cookies_per_bag := 20
  let bags_per_box := 4
  let calories_per_cookie := 20
  let total_cookies := cookies_per_bag * bags_per_box
  let total_calories := total_cookies * calories_per_cookie
  total_calories = 1600 :=
by
  let cookies_per_bag := 20
  let bags_per_box := 4
  let calories_per_cookie := 20
  let total_cookies := cookies_per_bag * bags_per_box
  let total_calories := total_cookies * calories_per_cookie
  show total_calories = 1600
  sorry

end box_contains_1600_calories_l97_97637


namespace binary_division_example_l97_97945

theorem binary_division_example : 
  let a := 0b10101  -- binary representation of 21
  let b := 0b11     -- binary representation of 3
  let quotient := 0b111  -- binary representation of 7
  a / b = quotient := 
by sorry

end binary_division_example_l97_97945


namespace total_cases_of_cat_food_sold_l97_97491

theorem total_cases_of_cat_food_sold :
  (let first_eight := 8 * 3 in
   let next_four := 4 * 2 in
   let last_eight := 8 * 1 in
   first_eight + next_four + last_eight = 40) :=
by
  -- Given conditions:
  -- first_8_customers: 8 customers bought 3 cases each
  -- second_4_customers: 4 customers bought 2 cases each
  -- last_8_customers: 8 customers bought 1 case each
  let first_eight := 8 * 3
  let next_four := 4 * 2
  let last_eight := 8 * 1
  -- Sum of all cases
  show first_eight + next_four + last_eight = 40
  sorry

end total_cases_of_cat_food_sold_l97_97491


namespace tim_picks_matching_pair_probability_l97_97539

def socks_probability :=
  let total_socks := 18
  let gray_socks := 10
  let white_socks := 8
  let total_ways := Nat.choose total_socks 2
  let gray_pair := Nat.choose gray_socks 2
  let white_pair := Nat.choose white_socks 2
  let matching_ways := gray_pair + white_pair
  (matching_ways : ℚ) / (total_ways : ℚ)

theorem tim_picks_matching_pair_probability :
  socks_probability = (73 : ℚ) / 153 :=
by {
  sorry
}

end tim_picks_matching_pair_probability_l97_97539


namespace complement_event_A_l97_97153

def is_at_least_two_defective (n : ℕ) : Prop :=
  n ≥ 2

def is_at_most_one_defective (n : ℕ) : Prop :=
  n ≤ 1

theorem complement_event_A (n : ℕ) :
  (¬ is_at_least_two_defective n) ↔ is_at_most_one_defective n :=
by
  sorry

end complement_event_A_l97_97153


namespace arithmetic_sequence_first_term_l97_97572

theorem arithmetic_sequence_first_term (a : ℕ → ℤ) (d : ℤ) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_terms_int : ∀ n, ∃ k : ℤ, a n = k) 
  (ha20 : a 20 = 205) : a 1 = 91 :=
sorry

end arithmetic_sequence_first_term_l97_97572


namespace possible_values_of_reciprocal_l97_97716

theorem possible_values_of_reciprocal (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = 1) :
  ∃ S, S = { x : ℝ | x >= 9 } ∧ (∃ x, x = (1/a + 1/b) ∧ x ∈ S) :=
sorry

end possible_values_of_reciprocal_l97_97716


namespace evaluate_f_l97_97570

def f (x : ℝ) : ℝ := sorry  -- Placeholder function definition

theorem evaluate_f :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x : ℝ, f (x + 5/2) = -1 / f x) ∧
  (∀ x : ℝ, x ∈ [-5/2, 0] → f x = x * (x + 5/2))
  → f 2016 = 3/2 :=
by
  sorry

end evaluate_f_l97_97570


namespace fencing_cost_correct_l97_97190

noncomputable def length : ℝ := 80
noncomputable def diff : ℝ := 60
noncomputable def cost_per_meter : ℝ := 26.50

-- Let's calculate the breadth first
noncomputable def breadth : ℝ := length - diff

-- Calculate the perimeter
noncomputable def perimeter : ℝ := 2 * (length + breadth)

-- Calculate the total cost
noncomputable def total_cost : ℝ := perimeter * cost_per_meter

theorem fencing_cost_correct : total_cost = 5300 := 
by 
  sorry

end fencing_cost_correct_l97_97190


namespace binomial_10_5_eq_252_l97_97531

theorem binomial_10_5_eq_252 : nat.choose 10 5 = 252 :=
by
  -- We will add "sorry" here as we are only required to state the theorem, not prove it.
  sorry

end binomial_10_5_eq_252_l97_97531


namespace pirates_divide_coins_l97_97517

theorem pirates_divide_coins (N : ℕ) (hN : 220 ≤ N ∧ N ≤ 300) :
  ∃ n : ℕ, 
    (N - 2 - (N - 2) / 3 - 2 - (2 * ((N - 2) / 3 - (2 * ((N - 2) / 3) / 3)) / 3) - 
    2 - (2 * (((N - 2) / 3 - (2 * ((N - 2) / 3) / 3)) / 3)) / 3) / 3 = 84 := 
sorry

end pirates_divide_coins_l97_97517


namespace parabola_focus_coincides_ellipse_focus_l97_97152

theorem parabola_focus_coincides_ellipse_focus (p : ℝ) :
  (∃ F : ℝ × ℝ, F = (2, 0) ∧ ∀ x y : ℝ, y^2 = 2 * p * x <-> x = p / 2)
  → p = 4 := 
by
  sorry 

end parabola_focus_coincides_ellipse_focus_l97_97152


namespace percentage_of_a_is_4b_l97_97339

variable (a b : ℝ)

theorem percentage_of_a_is_4b (h : a = 1.2 * b) : 4 * b = (10 / 3) * a := 
by 
    sorry

end percentage_of_a_is_4b_l97_97339


namespace pentagon_area_l97_97475

open Function 

/-
Given a convex pentagon FGHIJ with the following properties:
  1. ∠F = ∠G = 100°
  2. JF = FG = GH = 3
  3. HI = IJ = 5
Prove that the area of pentagon FGHIJ is approximately 15.2562 square units.
-/

noncomputable def area_pentagon_FGHIJ : ℝ :=
  let sin100 := Real.sin (100 * Real.pi / 180)
  let area_FGJ := (3 * 3 * sin100) / 2
  let area_HIJ := (5 * 5 * Real.sqrt 3) / 4
  area_FGJ + area_HIJ

theorem pentagon_area : abs (area_pentagon_FGHIJ - 15.2562) < 0.0001 := by
  sorry

end pentagon_area_l97_97475


namespace arccos_one_half_eq_pi_div_three_l97_97929

theorem arccos_one_half_eq_pi_div_three :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ π ∧ cos θ = (1 / 2) ∧ arccos (1 / 2) = θ :=
sorry

end arccos_one_half_eq_pi_div_three_l97_97929


namespace arccos_half_eq_pi_div_three_l97_97925

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
by
  sorry

end arccos_half_eq_pi_div_three_l97_97925


namespace ice_cream_to_afford_games_l97_97726

theorem ice_cream_to_afford_games :
  let game_cost := 60
  let ice_cream_price := 5
  (game_cost * 2) / ice_cream_price = 24 :=
by
  let game_cost := 60
  let ice_cream_price := 5
  show (game_cost * 2) / ice_cream_price = 24
  sorry

end ice_cream_to_afford_games_l97_97726


namespace oak_grove_total_books_l97_97748

theorem oak_grove_total_books (public_library_books : ℕ) (school_library_books : ℕ)
  (h1 : public_library_books = 1986) (h2 : school_library_books = 5106) :
  public_library_books + school_library_books = 7092 := by
  sorry

end oak_grove_total_books_l97_97748


namespace jacket_final_price_l97_97782

theorem jacket_final_price :
    let initial_price := 150
    let first_discount := 0.30
    let second_discount := 0.10
    let coupon := 10
    let tax := 0.05
    let price_after_first_discount := initial_price * (1 - first_discount)
    let price_after_second_discount := price_after_first_discount * (1 - second_discount)
    let price_after_coupon := price_after_second_discount - coupon
    let final_price := price_after_coupon * (1 + tax)
    final_price = 88.725 :=
by
  sorry

end jacket_final_price_l97_97782


namespace dihedral_angles_pyramid_l97_97407

noncomputable def dihedral_angles (a b : ℝ) : ℝ × ℝ :=
  let alpha := Real.arccos ((a * Real.sqrt 3) / Real.sqrt (4 * b ^ 2 - a ^ 2))
  let gamma := 2 * Real.arctan (b / Real.sqrt (4 * b ^ 2 - a ^ 2))
  (alpha, gamma)

theorem dihedral_angles_pyramid (a b alpha gamma : ℝ) (h1 : a > 0) (h2 : b > 0) :
  dihedral_angles a b = (alpha, gamma) :=
sorry

end dihedral_angles_pyramid_l97_97407


namespace tan_double_angle_solution_l97_97962

theorem tan_double_angle_solution (x : ℝ) (h : Real.tan (x + Real.pi / 4) = 2) :
  (Real.tan x) / (Real.tan (2 * x)) = 4 / 9 :=
sorry

end tan_double_angle_solution_l97_97962


namespace num_divisors_of_2002_l97_97823

theorem num_divisors_of_2002 : 
  (Nat.divisors 2002).length = 16 := 
sorry

end num_divisors_of_2002_l97_97823


namespace sum_of_c_d_l97_97302

theorem sum_of_c_d (c d : ℝ) (g : ℝ → ℝ) 
(hg : ∀ x, g x = (x + 5) / (x^2 + c * x + d)) 
(hasymp : ∀ x, (x = 2 ∨ x = -3) → x^2 + c * x + d = 0) : 
c + d = -5 := 
by 
  sorry

end sum_of_c_d_l97_97302


namespace lcm_18_35_is_630_l97_97808

def lcm_18_35 : ℕ :=
  Nat.lcm 18 35

theorem lcm_18_35_is_630 : lcm_18_35 = 630 := by
  sorry

end lcm_18_35_is_630_l97_97808


namespace friends_total_earnings_l97_97236

def Lauryn_earnings : ℝ := 2000
def Aurelia_fraction : ℝ := 0.7

def Aurelia_earnings : ℝ := Aurelia_fraction * Lauryn_earnings

def total_earnings : ℝ := Lauryn_earnings + Aurelia_earnings

theorem friends_total_earnings : total_earnings = 3400 := by
  -- We defined everything necessary here; the exact proof steps are omitted as per instructions.
  sorry

end friends_total_earnings_l97_97236


namespace quadratic_trinomial_bound_l97_97576

theorem quadratic_trinomial_bound (a b : ℤ) (f : ℝ → ℝ)
  (h_def : ∀ x : ℝ, f x = x^2 + a * x + b)
  (h_bound : ∀ x : ℝ, f x ≥ -9 / 10) :
  ∀ x : ℝ, f x ≥ -1 / 4 :=
sorry

end quadratic_trinomial_bound_l97_97576


namespace pentagonal_tiles_count_l97_97770

theorem pentagonal_tiles_count (a b : ℕ) (h1 : a + b = 30) (h2 : 3 * a + 5 * b = 120) : b = 15 :=
by
  sorry

end pentagonal_tiles_count_l97_97770


namespace vector_dot_product_l97_97427

open Real

variables (a b : ℝ × ℝ)

def condition1 : Prop := (a.1 + b.1 = 1 ∧ a.2 + b.2 = -3)
def condition2 : Prop := (a.1 - b.1 = 3 ∧ a.2 - b.2 = 7)
def dot_product : ℝ := a.1 * b.1 + a.2 * b.2

theorem vector_dot_product :
  condition1 a b ∧ condition2 a b → dot_product a b = -12 := by
  sorry

end vector_dot_product_l97_97427


namespace temperature_difference_l97_97746

theorem temperature_difference 
    (freezer_temp : ℤ) (room_temp : ℤ) (temperature_difference : ℤ) 
    (h1 : freezer_temp = -4) 
    (h2 : room_temp = 18) : 
    temperature_difference = room_temp - freezer_temp := 
by 
  sorry

end temperature_difference_l97_97746


namespace trip_savings_l97_97616

theorem trip_savings :
  let ticket_cost := 10
  let combo_cost := 10
  let ticket_discount := 0.20
  let combo_discount := 0.50
  (ticket_discount * ticket_cost + combo_discount * combo_cost) = 7 := 
by
  sorry

end trip_savings_l97_97616


namespace students_in_5th_6th_grades_l97_97481

-- Definitions for problem conditions
def is_three_digit_number (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000

def six_two_digit_sum_eq_twice (n : ℕ) : Prop :=
  ∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧
               a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
               (10 * a + b) + (10 * b + a) + (10 * a + c) + (10 * c + a) + (10 * b + c) + (10 * c + b) = 2 * n

-- The proof problem statement in Lean 4
theorem students_in_5th_6th_grades :
  ∃ n : ℕ, is_three_digit_number n ∧ six_two_digit_sum_eq_twice n ∧ n = 198 :=
by
  sorry

end students_in_5th_6th_grades_l97_97481


namespace pythagorean_theorem_special_cases_l97_97982

open Nat

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k
def is_multiple_of_3 (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k
def is_multiple_of_5 (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k

theorem pythagorean_theorem_special_cases (a b c : ℕ) (h : a^2 + b^2 = c^2) :
  (is_even a ∨ is_even b) ∧ 
  (is_multiple_of_3 a ∨ is_multiple_of_3 b) ∧ 
  (is_multiple_of_5 a ∨ is_multiple_of_5 b ∨ is_multiple_of_5 c) :=
by
  sorry

end pythagorean_theorem_special_cases_l97_97982


namespace triangle_angle_A_eq_60_l97_97830

theorem triangle_angle_A_eq_60 (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : A + B + C = π)
  (h_tan : (Real.tan A) / (Real.tan B) = (2 * c - b) / b) : 
  A = π / 3 :=
by
  sorry

end triangle_angle_A_eq_60_l97_97830


namespace orange_shells_correct_l97_97727

def total_shells : Nat := 65
def purple_shells : Nat := 13
def pink_shells : Nat := 8
def yellow_shells : Nat := 18
def blue_shells : Nat := 12
def orange_shells : Nat := total_shells - (purple_shells + pink_shells + yellow_shells + blue_shells)

theorem orange_shells_correct : orange_shells = 14 :=
by
  sorry

end orange_shells_correct_l97_97727


namespace find_g_3_8_l97_97738

variable (g : ℝ → ℝ)
variable (x : ℝ)

-- Conditions
axiom g0 : g 0 = 0
axiom monotonicity (x y : ℝ) : 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y
axiom symmetry (x : ℝ) : 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x
axiom scaling (x : ℝ) : 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3

-- Statement to prove
theorem find_g_3_8 : g (3 / 8) = 2 / 9 := 
sorry

end find_g_3_8_l97_97738


namespace age_hence_l97_97513

theorem age_hence (A x : ℕ) (h1 : A = 50)
  (h2 : 5 * (A + x) - 5 * (A - 5) = A) : x = 5 :=
by sorry

end age_hence_l97_97513


namespace friends_total_earnings_l97_97235

def Lauryn_earnings : ℝ := 2000
def Aurelia_fraction : ℝ := 0.7

def Aurelia_earnings : ℝ := Aurelia_fraction * Lauryn_earnings

def total_earnings : ℝ := Lauryn_earnings + Aurelia_earnings

theorem friends_total_earnings : total_earnings = 3400 := by
  -- We defined everything necessary here; the exact proof steps are omitted as per instructions.
  sorry

end friends_total_earnings_l97_97235


namespace sum_group_with_10_is_22_l97_97537

def groupA := {2, 5, 9}
def groupS := {1, 3, 4, 6, 7, 8, 10}

theorem sum_group_with_10_is_22 :
  ∀ (G1 G2 G3 : Finset ℕ), 
    (G1 ∪ G2 ∪ G3 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) ∧
    (groupA ⊆ G1 ∨ groupA ⊆ G2 ∨ groupA ⊆ G3) ∧
    (∀ g ∈ {G1, G2, G3}, 
      ∀ x y ∈ g, 
        x ≠ y → (x - y) ∉ g) → 
    ∃ g ∈ {G1, G2, G3}, 10 ∈ g ∧ g.sum id = 22 := sorry

end sum_group_with_10_is_22_l97_97537


namespace range_of_m_l97_97138

noncomputable def f (x : ℝ) := Real.log (x^2 + 1)

noncomputable def g (x m : ℝ) := (1 / 2)^x - m

theorem range_of_m (m : ℝ) :
  (∀ x1 ∈ Set.Icc (0:ℝ) 3, ∃ x2 ∈ Set.Icc (1:ℝ) 2, f x1 ≤ g x2 m) ↔ m ≤ -1/2 :=
by
  sorry

end range_of_m_l97_97138


namespace std_deviation_above_l97_97951

variable (mean : ℝ) (std_dev : ℝ) (score1 : ℝ) (score2 : ℝ)
variable (n1 : ℝ) (n2 : ℝ)

axiom h_mean : mean = 74
axiom h_std1 : score1 = 58
axiom h_std2 : score2 = 98
axiom h_cond1 : score1 = mean - n1 * std_dev
axiom h_cond2 : n1 = 2

theorem std_deviation_above (mean std_dev score1 score2 n1 n2 : ℝ)
  (h_mean : mean = 74)
  (h_std1 : score1 = 58)
  (h_std2 : score2 = 98)
  (h_cond1 : score1 = mean - n1 * std_dev)
  (h_cond2 : n1 = 2) :
  n2 = (score2 - mean) / std_dev :=
sorry

end std_deviation_above_l97_97951


namespace increasing_function_shape_implies_number_l97_97076

variable {I : Set ℝ} {f : ℝ → ℝ}

theorem increasing_function_shape_implies_number (h : ∀ (x₁ x₂ : ℝ), x₁ ∈ I ∧ x₂ ∈ I ∧ x₁ < x₂ → f x₁ < f x₂) 
: ∀ (x₁ x₂ : ℝ), x₁ ∈ I ∧ x₂ ∈ I ∧ x₁ < x₂ → f x₁ < f x₂ :=
sorry

end increasing_function_shape_implies_number_l97_97076


namespace percent_of_x_l97_97753

variable (x : ℝ) (h : x > 0)

theorem percent_of_x (p : ℝ) : 
  (p * x = 0.21 * x + 10) → 
  p = 0.21 + 10 / x :=
sorry

end percent_of_x_l97_97753


namespace sum_possible_values_A_B_l97_97144

theorem sum_possible_values_A_B : 
  ∀ (A B : ℕ), 
  (0 ≤ A ∧ A ≤ 9) ∧ 
  (0 ≤ B ∧ B ≤ 9) ∧ 
  ∃ k : ℕ, 28 + A + B = 9 * k 
  → (A + B = 8 ∨ A + B = 17) 
  → A + B = 25 :=
by
  sorry

end sum_possible_values_A_B_l97_97144


namespace seq_sum_eq_314_l97_97936

theorem seq_sum_eq_314 (d r : ℕ) (k : ℕ) (a_n b_n c_n : ℕ → ℕ)
  (h1 : ∀ n, a_n n = 1 + (n - 1) * d)
  (h2 : ∀ n, b_n n = r ^ (n - 1))
  (h3 : ∀ n, c_n n = a_n n + b_n n)
  (hk1 : c_n (k - 1) = 150)
  (hk2 : c_n (k + 1) = 900) :
  c_n k = 314 := by
  sorry

end seq_sum_eq_314_l97_97936


namespace trapezoid_area_ABCD_l97_97353

noncomputable def trapezoid_area (AB CD BC : ℝ) (M : ℝ) (DM_angle_bisector_passes_M : Prop) : ℝ :=
  let AD := 8  -- inferred in the solution step
  let PC := 8  -- inferred height from Pythagorean theorem
  (1 / 2) * (AD + BC) * PC

theorem trapezoid_area_ABCD :
  let AB := 8
  let CD := 10
  let BC := 2
  let M := 4  -- midpoint of AB
  let DM_angle_bisector_passes_M := true
  trapezoid_area AB CD BC M DM_angle_bisector_passes_M = 40 :=
by
  sorry

end trapezoid_area_ABCD_l97_97353


namespace xyz_inequality_l97_97328

theorem xyz_inequality {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z))) + (y^3 / ((1 + z) * (1 + x))) + (z^3 / ((1 + x) * (1 + y))) ≥ (3/4) :=
sorry

end xyz_inequality_l97_97328


namespace box_calories_l97_97634

theorem box_calories :
  let cookies_per_bag := 20
  let bags_per_box := 4
  let calories_per_cookie := 20
  (cookies_per_bag * bags_per_box) * calories_per_cookie = 1600 :=
by
  sorry

end box_calories_l97_97634


namespace trisha_interest_l97_97073

noncomputable def total_amount (P : ℝ) (r : ℝ) (D : ℝ) (t : ℕ) : ℝ :=
  let rec compute (n : ℕ) (A : ℝ) :=
    if n = 0 then A
    else let A_next := A * (1 + r) + D
         compute (n - 1) A_next
  compute t P

noncomputable def total_deposits (D : ℝ) (t : ℕ) : ℝ :=
  D * t

noncomputable def total_interest (P : ℝ) (r : ℝ) (D : ℝ) (t : ℕ) : ℝ :=
  total_amount P r D t - P - total_deposits D t

theorem trisha_interest :
  total_interest 2000 0.05 300 5 = 710.25 :=
by
  sorry

end trisha_interest_l97_97073


namespace range_of_a_l97_97435

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x + Real.log x - (x^2 / (x - Real.log x))

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0) ↔
  1 < a ∧ a < (Real.exp 1) / (Real.exp 1 - 1) - 1 / Real.exp 1 :=
sorry

end range_of_a_l97_97435


namespace least_value_sum_l97_97979

theorem least_value_sum (x y z : ℤ) (h : (x - 10) * (y - 5) * (z - 2) = 1000) : x + y + z = 92 :=
sorry

end least_value_sum_l97_97979


namespace find_n_l97_97624

theorem find_n : ∃ n : ℕ, 
  (n % 4 = 1) ∧ (n % 5 = 2) ∧ (n % 6 = 3) ∧ (n = 57) := by
  sorry

end find_n_l97_97624


namespace union_A_B_l97_97972

def A : Set ℝ := {x | |x| < 3}
def B : Set ℝ := {x | 2 - x > 0}

theorem union_A_B (x : ℝ) : (x ∈ A ∨ x ∈ B) ↔ x < 3 := by
  sorry

end union_A_B_l97_97972


namespace treasures_on_island_l97_97084

-- Define the propositions P and K
def P : Prop := ∃ p : Prop, p
def K : Prop := ∃ k : Prop, k

-- Define the claim by A
def A_claim : Prop := K ↔ P

-- Theorem statement as specified part (b)
theorem treasures_on_island (A_is_knight_or_liar : (A_claim ↔ true) ∨ (A_claim ↔ false)) : ∃ P, P :=
by
  sorry

end treasures_on_island_l97_97084


namespace competition_sequences_l97_97472

-- Define the problem conditions
def team_size : Nat := 7

-- Define the statement to prove
theorem competition_sequences :
  (Nat.choose (2 * team_size) team_size) = 3432 :=
by
  -- Proof will go here
  sorry

end competition_sequences_l97_97472


namespace find_number_of_students_l97_97342

theorem find_number_of_students (N : ℕ) (h1 : T = 80 * N) (h2 : (T - 350) / (N - 5) = 90) 
: N = 10 := 
by 
  -- Proof steps would go here. Omitted as per the instruction.
  sorry

end find_number_of_students_l97_97342


namespace plot_area_is_correct_l97_97516

noncomputable def scaled_area_in_acres
  (scale_cm_miles : ℕ)
  (area_conversion_factor_miles_acres : ℕ)
  (bottom_cm : ℕ)
  (top_cm : ℕ)
  (height_cm : ℕ) : ℕ :=
  let area_cm_squared := (1 / 2) * (bottom_cm + top_cm) * height_cm
  let area_in_squared_miles := area_cm_squared * (scale_cm_miles * scale_cm_miles)
  area_in_squared_miles * area_conversion_factor_miles_acres

theorem plot_area_is_correct :
  scaled_area_in_acres 3 640 18 14 12 = 1105920 :=
by
  sorry

end plot_area_is_correct_l97_97516


namespace crayons_left_l97_97873

/-- Given initially 48 crayons, if Kiley takes 1/4 and Joe takes half of the remaining,
then 18 crayons are left. -/
theorem crayons_left (initial_crayons : ℕ) (kiley_fraction joe_fraction : ℚ)
    (h_initial : initial_crayons = 48) (h_kiley : kiley_fraction = 1 / 4) (h_joe : joe_fraction = 1 / 2) :
  let kiley_takes := kiley_fraction * initial_crayons,
      remaining_after_kiley := initial_crayons - kiley_takes,
      joe_takes := joe_fraction * remaining_after_kiley,
      crayons_left := remaining_after_kiley - joe_takes
  in crayons_left = 18 :=
by
  sorry

end crayons_left_l97_97873


namespace hyperbola_foci_coordinates_l97_97858

theorem hyperbola_foci_coordinates :
  let a : ℝ := Real.sqrt 7
  let b : ℝ := Real.sqrt 3
  let c : ℝ := Real.sqrt (a^2 + b^2)
  (c = Real.sqrt 10 ∧
  ∀ x y, (x^2 / 7 - y^2 / 3 = 1) → ((x, y) = (c, 0) ∨ (x, y) = (-c, 0))) :=
by
  let a := Real.sqrt 7
  let b := Real.sqrt 3
  let c := Real.sqrt (a^2 + b^2)
  have hc : c = Real.sqrt 10 := sorry
  have h_foci : ∀ x y, (x^2 / 7 - y^2 / 3 = 1) → ((x, y) = (c, 0) ∨ (x, y) = (-c, 0)) := sorry
  exact ⟨hc, h_foci⟩

end hyperbola_foci_coordinates_l97_97858


namespace linear_eq_a_value_l97_97010

theorem linear_eq_a_value (a : ℤ) (x : ℝ) 
  (h : x^(a-1) - 5 = 3) 
  (h_lin : ∃ b c : ℝ, x^(a-1) * b + c = 0 ∧ b ≠ 0):
  a = 2 :=
sorry

end linear_eq_a_value_l97_97010


namespace lcm_18_35_is_630_l97_97810

def lcm_18_35 : ℕ :=
  Nat.lcm 18 35

theorem lcm_18_35_is_630 : lcm_18_35 = 630 := by
  sorry

end lcm_18_35_is_630_l97_97810


namespace annual_fixed_costs_l97_97787

theorem annual_fixed_costs
  (profit : ℝ := 30500000)
  (selling_price : ℝ := 9035)
  (variable_cost : ℝ := 5000)
  (units_sold : ℕ := 20000) :
  ∃ (fixed_costs : ℝ), profit = (selling_price * units_sold) - (variable_cost * units_sold) - fixed_costs :=
sorry

end annual_fixed_costs_l97_97787


namespace find_other_integer_l97_97611

theorem find_other_integer (x y : ℕ) (h1 : 1 ≤ x ∧ x ≤ 9) (h2 : 1 ≤ y ∧ y ≤ 9) (h3 : 7 * x + y = 68) : y = 5 :=
by
  sorry

end find_other_integer_l97_97611


namespace range_of_f_l97_97131

noncomputable def f (x y : ℝ) := (x^3 + y^3) / (x + y)^3

theorem range_of_f :
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x^2 + y^2 = 1 → (1 / 4) ≤ f x y ∧ f x y < 1) :=
by
  sorry

end range_of_f_l97_97131


namespace find_x_l97_97675

noncomputable def approx_equal (a b : ℝ) (ε : ℝ) : Prop := abs (a - b) < ε

theorem find_x :
  ∃ x : ℝ, x + Real.sqrt 68 = 24 ∧ approx_equal x 15.753788749 0.0001 :=
sorry

end find_x_l97_97675


namespace money_left_for_lunch_and_snacks_l97_97466

-- Definitions according to the conditions
def ticket_cost_per_person : ℝ := 5
def bus_fare_one_way_per_person : ℝ := 1.50
def total_budget : ℝ := 40
def number_of_people : ℝ := 2

-- The proposition to be proved
theorem money_left_for_lunch_and_snacks : 
  let total_zoo_cost := ticket_cost_per_person * number_of_people
  let total_bus_fare := bus_fare_one_way_per_person * number_of_people * 2
  let total_expense := total_zoo_cost + total_bus_fare
  total_budget - total_expense = 24 :=
by
  sorry

end money_left_for_lunch_and_snacks_l97_97466


namespace incorrect_expression_l97_97148

theorem incorrect_expression (x y : ℝ) (h : x > y) : ¬ (3 - x > 3 - y) :=
by
  sorry

end incorrect_expression_l97_97148


namespace inequality_positive_real_xyz_l97_97329

theorem inequality_positive_real_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z)) + y^3 / ((1 + z) * (1 + x)) + z^3 / ((1 + x) * (1 + y))) ≥ (3 / 4) := 
by
  -- Proof is to be constructed here
  sorry

end inequality_positive_real_xyz_l97_97329


namespace charlie_delta_four_products_l97_97224

noncomputable def charlie_delta_purchase_ways : ℕ := 1363

theorem charlie_delta_four_products :
  let cakes := 6
  let cookies := 4
  let total := cakes + cookies
  ∃ ways : ℕ, ways = charlie_delta_purchase_ways :=
by
  sorry

end charlie_delta_four_products_l97_97224


namespace average_percent_score_l97_97671

theorem average_percent_score (students : ℕ) 
  (s95 s85 s75 s65 s55 s45 : ℕ) 
  (h_students : students = 150) 
  (h_s95 : s95 = 12) 
  (h_s85 : s85 = 30) 
  (h_s75 : s75 = 50) 
  (h_s65 : s65 = 40) 
  (h_s55 : s55 = 15) 
  (h_s45 : s45 = 3) : 
  (95 * s95 + 85 * s85 + 75 * s75 + 65 * s65 + 55 * s55 + 45 * s45) / students = 73.33 :=
by
  sorry

end average_percent_score_l97_97671


namespace internal_diagonal_passes_through_cubes_l97_97222

theorem internal_diagonal_passes_through_cubes :
  let a := 180
  let b := 360
  let c := 450
  a + b + c - (Nat.gcd a b + Nat.gcd b c + Nat.gcd c a) + Nat.gcd a (Nat.gcd b c) = 720 :=
by
  sorry

end internal_diagonal_passes_through_cubes_l97_97222


namespace g_three_eighths_l97_97739

variable (g : ℝ → ℝ)

-- Conditions
axiom g_zero : g 0 = 0
axiom monotonic : ∀ {x y : ℝ}, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y
axiom symmetry : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x
axiom scaling : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3

-- The theorem statement we need to prove
theorem g_three_eighths : g (3 / 8) = 2 / 9 :=
sorry

end g_three_eighths_l97_97739


namespace circle_equation_exists_shortest_chord_line_l97_97289

-- Condition 1: Points A and B
def point_A : (ℝ × ℝ) := (1, -2)
def point_B : (ℝ × ℝ) := (-1, 0)

-- Condition 2: Circle passes through A and B and sum of intercepts is 2
def passes_through (x y : ℝ) (D E F : ℝ) : Prop := 
  (x^2 + y^2 + D * x + E * y + F = 0)

def satisfies_intercepts (D E : ℝ) : Prop := (-D - E = 2)

-- Prove
theorem circle_equation_exists : 
  ∃ D E F, passes_through 1 (-2) D E F ∧ passes_through (-1) 0 D E F ∧ satisfies_intercepts D E :=
sorry

-- Given that P(2, 0.5) is inside the circle from above theorem
def point_P : (ℝ × ℝ) := (2, 0.5)

-- Prove the equation of the shortest chord line l
theorem shortest_chord_line :
  ∃ m b, m = -2 ∧ point_P.2 = m * (point_P.1 - 2) + b ∧ (∀ (x y : ℝ), 4 * x + 2 * y - 9 = 0) :=
sorry

end circle_equation_exists_shortest_chord_line_l97_97289


namespace pipeline_equation_correct_l97_97440

variables (m x n : ℝ) -- Length of the pipeline, kilometers per day, efficiency increase percentage
variable (h : 0 < n) -- Efficiency increase percentage is positive

theorem pipeline_equation_correct :
  (m / x) - (m / ((1 + (n / 100)) * x)) = 8 :=
sorry -- Proof omitted

end pipeline_equation_correct_l97_97440


namespace segments_count_l97_97855

/--
Given two concentric circles, with chords of the larger circle that are tangent to the smaller circle,
if each chord subtends an angle of 80 degrees at the center, then the number of such segments 
drawn before returning to the starting point is 18.
-/
theorem segments_count (angle_ABC : ℝ) (circumference_angle_sum : ℝ → ℝ) (n m : ℕ) :
  angle_ABC = 80 → 
  circumference_angle_sum angle_ABC = 360 → 
  100 * n = 360 * m → 
  5 * n = 18 * m →
  n = 18 :=
by
  intros h1 h2 h3 h4
  sorry

end segments_count_l97_97855


namespace initial_flour_amount_l97_97938

theorem initial_flour_amount (initial_flour : ℕ) (additional_flour : ℕ) (total_flour : ℕ) 
  (h1 : additional_flour = 4) (h2 : total_flour = 16) (h3 : initial_flour + additional_flour = total_flour) :
  initial_flour = 12 := 
by 
  sorry

end initial_flour_amount_l97_97938


namespace three_digit_numbers_square_ends_in_1001_l97_97405

theorem three_digit_numbers_square_ends_in_1001 (n : ℕ) :
  100 ≤ n ∧ n < 1000 ∧ n^2 % 10000 = 1001 → n = 501 ∨ n = 749 :=
by
  intro h
  sorry

end three_digit_numbers_square_ends_in_1001_l97_97405


namespace average_weight_increase_l97_97443

theorem average_weight_increase 
  (n : ℕ) (old_weight new_weight : ℝ) (group_size := 8) 
  (old_weight := 70) (new_weight := 90) : 
  ((new_weight - old_weight) / group_size) = 2.5 := 
by sorry

end average_weight_increase_l97_97443


namespace bacteria_growth_rate_l97_97774

theorem bacteria_growth_rate (B G : ℝ) (h : B * G^16 = 2 * B * G^15) : G = 2 :=
by
  sorry

end bacteria_growth_rate_l97_97774


namespace range_f_a2_min_value_f_3_l97_97421

noncomputable def f (x a : ℝ) : ℝ := 4 * x^2 - 4 * a * x + (a^2 - 2 * a + 2)

theorem range_f_a2 : ∀ x ∈ Icc 1 2, f x 2 ∈ Icc (-2 : ℝ) 2 :=
by
  sorry

theorem min_value_f_3 (h1 : ∃ x ∈ Icc 0 2, f x a = 3):
  a = 1 - Real.sqrt 2 ∨ a = 5 + Real.sqrt 10 :=
by
  sorry

end range_f_a2_min_value_f_3_l97_97421


namespace cubs_more_home_runs_l97_97524

-- Define the conditions for the Chicago Cubs
def cubs_home_runs_third_inning : Nat := 2
def cubs_home_runs_fifth_inning : Nat := 1
def cubs_home_runs_eighth_inning : Nat := 2

-- Define the conditions for the Cardinals
def cardinals_home_runs_second_inning : Nat := 1
def cardinals_home_runs_fifth_inning : Nat := 1

-- Total home runs scored by each team
def total_cubs_home_runs : Nat :=
  cubs_home_runs_third_inning + cubs_home_runs_fifth_inning + cubs_home_runs_eighth_inning

def total_cardinals_home_runs : Nat :=
  cardinals_home_runs_second_inning + cardinals_home_runs_fifth_inning

-- The statement to prove
theorem cubs_more_home_runs : total_cubs_home_runs - total_cardinals_home_runs = 3 := by
  sorry

end cubs_more_home_runs_l97_97524


namespace infinite_series_sum_l97_97108

theorem infinite_series_sum : 
  ∑' k : ℕ, (8 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1))) = 2 :=
by 
  sorry

end infinite_series_sum_l97_97108


namespace ratio_of_kids_to_adult_meals_l97_97917

theorem ratio_of_kids_to_adult_meals (k a : ℕ) (h1 : k = 8) (h2 : k + a = 12) : k / a = 2 := 
by 
  sorry

end ratio_of_kids_to_adult_meals_l97_97917


namespace g_three_eighths_l97_97741

variable (g : ℝ → ℝ)

-- Conditions
axiom g_zero : g 0 = 0
axiom monotonic : ∀ {x y : ℝ}, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y
axiom symmetry : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x
axiom scaling : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3

-- The theorem statement we need to prove
theorem g_three_eighths : g (3 / 8) = 2 / 9 :=
sorry

end g_three_eighths_l97_97741


namespace inequality1_in_triangle_inequality2_in_triangle_l97_97160

theorem inequality1_in_triangle (a b c s : ℝ)
  (h1 : a + b + c = s) :
  (13 / 27) * s^2 ≤ a^2 + b^2 + c^2 + (4 / s) * a * b * c ∧ 
  a^2 + b^2 + c^2 + (4 / s) * a * b * c < s^2 / 2 :=
sorry

theorem inequality2_in_triangle (a b c s : ℝ)
  (h1 : a + b + c = s) :
  s^2 / 4 < a * b + b * c + c * a - (2 / s) * a * b * c ∧ 
  a * b + b * c + c * a - (2 / s) * a * b * c ≤ (7 / 27) * s^2 :=
sorry

end inequality1_in_triangle_inequality2_in_triangle_l97_97160


namespace inequality_solution_set_l97_97067

def solution_set (a b x : ℝ) : Set ℝ := {x | |a - b * x| - 5 ≤ 0}

theorem inequality_solution_set (x : ℝ) :
  solution_set 4 3 x = {x | - (1 : ℝ) / 3 ≤ x ∧ x ≤ 3} :=
by {
  sorry
}

end inequality_solution_set_l97_97067


namespace trajectory_of_C_is_ellipse_l97_97485

theorem trajectory_of_C_is_ellipse :
  ∀ (C : ℝ × ℝ),
  ((C.1 + 4)^2 + C.2^2).sqrt + ((C.1 - 4)^2 + C.2^2).sqrt = 10 →
  (C.2 ≠ 0) →
  (C.1^2 / 25 + C.2^2 / 9 = 1) :=
by
  intros C h1 h2
  sorry

end trajectory_of_C_is_ellipse_l97_97485


namespace find_fx_l97_97134

variable {e : ℝ} {a : ℝ} (f : ℝ → ℝ)

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

variable (hodd : odd_function f)
variable (hdef : ∀ x, -e ≤ x → x < 0 → f x = a * x + Real.log (-x))

theorem find_fx (x : ℝ) (hx : 0 < x ∧ x ≤ e) : f x = a * x - Real.log x :=
by
  sorry

end find_fx_l97_97134


namespace acute_angle_10_10_l97_97495

noncomputable def clock_angle_proof : Prop :=
  let minute_hand_position := 60
  let hour_hand_position := 305
  let angle_diff := hour_hand_position - minute_hand_position
  let acute_angle := if angle_diff > 180 then 360 - angle_diff else angle_diff
  acute_angle = 115

theorem acute_angle_10_10 : clock_angle_proof := by
  sorry

end acute_angle_10_10_l97_97495


namespace part_two_l97_97422

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 - 2 * x + a * Real.log x

theorem part_two (a : ℝ) (h : a = 4) (m n : ℝ) (hm : 0 < m) (hn : 0 < n)
  (h_cond : (f m a + f n a) / (m^2 * n^2) = 1) : m + n ≥ 3 :=
sorry

end part_two_l97_97422


namespace local_extrema_l97_97206

noncomputable def f (x : ℝ) := 3 * x^3 - 9 * x^2 + 3

theorem local_extrema :
  (∃ x, x = 0 ∧ ∀ δ > 0, ∃ ε > 0, ∀ y, abs (y - x) < ε → f y ≤ f x) ∧
  (∃ x, x = 2 ∧ ∀ δ > 0, ∃ ε > 0, ∀ y, abs (y - x) < ε → f y ≥ f x) :=
sorry

end local_extrema_l97_97206


namespace mutually_prime_sum_l97_97436

open Real

theorem mutually_prime_sum (A B C : ℤ) (h_prime : Int.gcd A (Int.gcd B C) = 1)
    (h_eq : A * log 5 / log 200 + B * log 2 / log 200 = C) : A + B + C = 6 := 
sorry

end mutually_prime_sum_l97_97436


namespace lcm_18_35_l97_97803

theorem lcm_18_35 : Nat.lcm 18 35 = 630 := 
by 
  sorry

end lcm_18_35_l97_97803


namespace shadow_taller_pot_length_l97_97366

-- Definitions based on the conditions a)
def height_shorter_pot : ℕ := 20
def shadow_shorter_pot : ℕ := 10
def height_taller_pot : ℕ := 40

-- The proof problem
theorem shadow_taller_pot_length : 
  ∃ (S2 : ℕ), (height_shorter_pot / shadow_shorter_pot = height_taller_pot / S2) ∧ S2 = 20 :=
sorry

end shadow_taller_pot_length_l97_97366


namespace not_perfect_square_l97_97965

theorem not_perfect_square (a b : ℤ) (h1 : a > b) (h2 : Int.gcd (ab - 1) (a + b) = 1) (h3 : Int.gcd (ab + 1) (a - b) = 1) :
  ¬ ∃ c : ℤ, (a + b)^2 + (ab - 1)^2 = c^2 := 
  sorry

end not_perfect_square_l97_97965


namespace x_pow_4_plus_inv_x_pow_4_l97_97042

theorem x_pow_4_plus_inv_x_pow_4 (x : ℝ) (h : x^2 - 15 * x + 1 = 0) : x^4 + (1 / x^4) = 49727 :=
by
  sorry

end x_pow_4_plus_inv_x_pow_4_l97_97042


namespace empty_solution_set_range_l97_97681

theorem empty_solution_set_range (m : ℝ) : 
  (¬ ∃ x : ℝ, (m * x^2 + 2 * m * x + 1) < 0) ↔ (m = 0 ∨ (0 < m ∧ m ≤ 1)) :=
by sorry

end empty_solution_set_range_l97_97681


namespace pentagon_area_l97_97477

noncomputable def angle_F := 100
noncomputable def angle_G := 100
noncomputable def JF := 3
noncomputable def FG := 3
noncomputable def GH := 3
noncomputable def HI := 5
noncomputable def IJ := 5
noncomputable def area_FGHIJ := 9 * Real.sqrt 3 + Real.sqrt 17.1875

theorem pentagon_area : area_FGHIJ = 9 * Real.sqrt 3 + Real.sqrt 17.1875 :=
by
  sorry

end pentagon_area_l97_97477


namespace ticket_difference_l97_97784

-- Definitions representing the number of VIP and general admission tickets
def numTickets (V G : Nat) : Prop :=
  V + G = 320

def totalCost (V G : Nat) : Prop :=
  40 * V + 15 * G = 7500

-- Theorem stating that the difference between general admission and VIP tickets is 104
theorem ticket_difference (V G : Nat) (h1 : numTickets V G) (h2 : totalCost V G) : G - V = 104 := by
  sorry

end ticket_difference_l97_97784


namespace monotonic_function_a_ge_one_l97_97574

theorem monotonic_function_a_ge_one (a : ℝ) :
  (∀ x : ℝ, (x^2 + 2 * x + a) ≥ 0) → a ≥ 1 :=
by
  intros h
  sorry

end monotonic_function_a_ge_one_l97_97574


namespace integers_between_sqrt8_and_sqrt78_l97_97429

theorem integers_between_sqrt8_and_sqrt78 : 
  ∃ (n : ℕ), n = 6 ∧ ∀ (x : ℤ), (3 ≤ x ∧ x ≤ 8) ↔ (√8 < x ∧ x < √78) :=
by
  sorry

end integers_between_sqrt8_and_sqrt78_l97_97429


namespace arccos_pi_over_3_l97_97931

theorem arccos_pi_over_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_pi_over_3_l97_97931


namespace RiversideAcademy_statistics_l97_97652

theorem RiversideAcademy_statistics (total_students physics_students both_subjects : ℕ)
  (h1 : total_students = 25)
  (h2 : physics_students = 10)
  (h3 : both_subjects = 6) :
  total_students - (physics_students - both_subjects) = 21 :=
by
  sorry

end RiversideAcademy_statistics_l97_97652


namespace sum_of_first_9_terms_l97_97579

variables {a : ℕ → ℤ} {S : ℕ → ℤ}

-- a_n is the nth term of the arithmetic sequence
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- S_n is the sum of first n terms of the arithmetic sequence
def sum_seq (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

-- Hypotheses
axiom h1 : 2 * a 8 = 6 + a 11
axiom h2 : arithmetic_seq a
axiom h3 : sum_seq S a

-- The theorem we want to prove
theorem sum_of_first_9_terms : S 9 = 54 :=
sorry

end sum_of_first_9_terms_l97_97579


namespace solve_for_y_l97_97395

theorem solve_for_y {y : ℝ} : 
  (2012 + y)^2 = 2 * y^2 ↔ y = 2012 * (Real.sqrt 2 + 1) ∨ y = -2012 * (Real.sqrt 2 - 1) := by
  sorry

end solve_for_y_l97_97395


namespace box_calories_l97_97635

theorem box_calories :
  let cookies_per_bag := 20
  let bags_per_box := 4
  let calories_per_cookie := 20
  (cookies_per_bag * bags_per_box) * calories_per_cookie = 1600 :=
by
  sorry

end box_calories_l97_97635


namespace find_coefficients_l97_97355

-- Define the polynomial
def poly (a b : ℤ) (x : ℚ) : ℚ := a * x^4 + b * x^3 + 40 * x^2 - 20 * x + 8

-- Define the factor
def factor (x : ℚ) : ℚ := 3 * x^2 - 2 * x + 2

-- States that for a given polynomial and factor, the resulting (a, b) pair is (-51, 25)
theorem find_coefficients :
  ∃ a b c d : ℤ, 
  (∀ x, poly a b x = (factor x) * (c * x^2 + d * x + 4)) ∧ 
  a = -51 ∧ 
  b = 25 :=
by sorry

end find_coefficients_l97_97355


namespace find_y_l97_97382

noncomputable def x : ℝ := 0.7142857142857143

def equation (y : ℝ) : Prop :=
  (x * y) / 7 = x^2

theorem find_y : ∃ y : ℝ, equation y ∧ y = 5 :=
by
  use 5
  have h1 : x != 0 := by sorry
  have h2 : (x * 5) / 7 = x^2 := by sorry
  exact ⟨h2, rfl⟩

end find_y_l97_97382


namespace lcm_18_35_l97_97804

theorem lcm_18_35 : Nat.lcm 18 35 = 630 := 
by 
  sorry

end lcm_18_35_l97_97804


namespace number_of_lines_l97_97905

-- Define the point (0, 1)
def point : ℝ × ℝ := (0, 1)

-- Define the parabola y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the condition that a line intersects a parabola at only one point
def line_intersects_parabola_at_one_point (m b x y : ℝ) : Prop :=
  y - (m * x + b) = 0 ∧ parabola x y

-- The proof problem: Prove there are 3 such lines
theorem number_of_lines : ∃ (n : ℕ), n = 3 ∧ (
  ∃ (m b : ℝ), line_intersects_parabola_at_one_point m b 0 1) :=
sorry

end number_of_lines_l97_97905


namespace area_transformed_function_l97_97608

noncomputable def area_g : ℝ := 15

noncomputable def area_4g_shifted : ℝ :=
  4 * area_g

theorem area_transformed_function :
  area_4g_shifted = 60 := by
  sorry

end area_transformed_function_l97_97608


namespace repeating_decimal_fraction_l97_97799

noncomputable def repeating_decimal := 4 + 36 / 99

theorem repeating_decimal_fraction : 
  repeating_decimal = 144 / 33 := 
sorry

end repeating_decimal_fraction_l97_97799


namespace abs_pi_expression_l97_97664

theorem abs_pi_expression : |π - |π - 10|| = 10 - 2 * π :=
by
  sorry

end abs_pi_expression_l97_97664


namespace kitten_length_l97_97650

theorem kitten_length (initial_length : ℕ) (doubled_length_1 : ℕ) (doubled_length_2 : ℕ) :
  initial_length = 4 →
  doubled_length_1 = 2 * initial_length →
  doubled_length_2 = 2 * doubled_length_1 →
  doubled_length_2 = 16 :=
by
  intros h1 h2 h3
  rw [h1] at h2
  rw [h2] at h3
  exact h3

end kitten_length_l97_97650


namespace sequence_period_2016_l97_97578

theorem sequence_period_2016 : 
  ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) = 1 / (1 - a n)) → 
  a 1 = 1 / 2 → 
  a 2016 = -1 :=
by
  sorry

end sequence_period_2016_l97_97578


namespace chess_tournament_games_l97_97987

def number_of_games (n : ℕ) : ℕ := n * (n - 1) / 2

theorem chess_tournament_games :
  number_of_games 20 = 190 :=
by
  sorry

end chess_tournament_games_l97_97987


namespace walking_distance_l97_97777

theorem walking_distance (x : ℝ) :
  ∃ x : ℝ, (∃ (y : ℝ), y = 6 ∧ (x - 3 * Real.sqrt 3)^2 + (3)^2 = (2 * Real.sqrt 3)^2) ↔ 
    (x = 3 * Real.sqrt 3 + Real.sqrt 3 ∨ x = 3 * Real.sqrt 3 - Real.sqrt 3) :=
begin
  sorry
end

end walking_distance_l97_97777


namespace _l97_97450

def count_trailing_zeros (n : ℕ) : ℕ :=
  if n = 0 then 0 else
    let rec count_factors (x d : ℕ) : ℕ :=
      if x % d ≠ 0 then 0 else 1 + count_factors (x / d) d;
    count_factors n 10

def product_of_factorials (n : ℕ) : ℕ :=
  (finset.range (n + 1)).prod factorial

def trailing_zeros_of_product : ℕ :=
  count_trailing_zeros (product_of_factorials 50)

example : trailing_zeros_of_product % 100 = 31 := 
by {
  -- This is a statement of the theorem (no proof included)
  sorry
}

end _l97_97450


namespace necessary_and_sufficient_condition_l97_97896

theorem necessary_and_sufficient_condition (x : ℝ) :
  x > 0 ↔ x + 1/x ≥ 2 :=
by sorry

end necessary_and_sufficient_condition_l97_97896


namespace solve_for_x_l97_97498

theorem solve_for_x (x : ℝ) (h : x ≠ 0) : (3 * x)^5 = (9 * x)^4 → x = 27 := 
by 
  admit

end solve_for_x_l97_97498


namespace Liliane_more_soda_than_Alice_l97_97654

variable (J : ℝ) -- Represents the amount of soda Jacqueline has

-- Conditions: Representing the amounts for Benjamin, Liliane, and Alice
def B := 1.75 * J
def L := 1.60 * J
def A := 1.30 * J

-- Question: Proving the relationship in percentage terms between the amounts Liliane and Alice have
theorem Liliane_more_soda_than_Alice :
  (L - A) / A * 100 = 23 := 
by sorry

end Liliane_more_soda_than_Alice_l97_97654


namespace correct_statement_l97_97500

section
variables {a b c d : Real}

-- Define the conditions as hypotheses/functions

-- Statement A: If a > b, then 1/a < 1/b
def statement_A (a b : Real) : Prop := a > b → 1 / a < 1 / b

-- Statement B: If a > b, then a^2 > b^2
def statement_B (a b : Real) : Prop := a > b → a^2 > b^2

-- Statement C: If a > b and c > d, then ac > bd
def statement_C (a b c d : Real) : Prop := a > b ∧ c > d → a * c > b * d

-- Statement D: If a^3 > b^3, then a > b
def statement_D (a b : Real) : Prop := a^3 > b^3 → a > b

-- The Lean statement to prove which statement is correct
theorem correct_statement : ¬ statement_A a b ∧ ¬ statement_B a b ∧ ¬ statement_C a b c d ∧ statement_D a b :=
by {
  sorry
}

end

end correct_statement_l97_97500


namespace angle_B_degrees_l97_97033

theorem angle_B_degrees (A B C : ℕ) (h1 : A < B) (h2 : B < C) (h3 : 4 * C = 7 * A) (h4 : A + B + C = 180) : B = 59 :=
sorry

end angle_B_degrees_l97_97033


namespace intersection_value_l97_97971

theorem intersection_value (x0 : ℝ) (h1 : -x0 = Real.tan x0) (h2 : x0 ≠ 0) :
  (x0^2 + 1) * (1 + Real.cos (2 * x0)) = 2 := 
  sorry

end intersection_value_l97_97971


namespace percentage_error_square_area_l97_97502

theorem percentage_error_square_area (s : ℝ) (h : s > 0) :
  let s' := (1.02 * s)
  let actual_area := s^2
  let measured_area := s'^2
  let error_area := measured_area - actual_area
  let percentage_error := (error_area / actual_area) * 100
  percentage_error = 4.04 := 
sorry

end percentage_error_square_area_l97_97502


namespace calculate_expression_l97_97923

variable (x y : ℝ)

theorem calculate_expression :
  (-2 * x^2 * y)^3 = -8 * x^6 * y^3 :=
by 
  sorry

end calculate_expression_l97_97923


namespace nat_solutions_l97_97797

open Nat

theorem nat_solutions (a b c : ℕ) :
  (a ≤ b ∧ b ≤ c ∧ ab + bc + ca = 2 * (a + b + c)) ↔ (a = 2 ∧ b = 2 ∧ c = 2) ∨ (a = 1 ∧ b = 2 ∧ c = 4) :=
by sorry

end nat_solutions_l97_97797


namespace parking_lot_wheels_l97_97704

-- definitions for the conditions
def num_cars : ℕ := 10
def num_bikes : ℕ := 2
def wheels_per_car : ℕ := 4
def wheels_per_bike : ℕ := 2

-- statement of the theorem
theorem parking_lot_wheels : (num_cars * wheels_per_car) + (num_bikes * wheels_per_bike) = 44 := by
  sorry

end parking_lot_wheels_l97_97704


namespace lily_has_26_dollars_left_for_coffee_l97_97047

-- Define the initial amount of money Lily has
def initialMoney : ℕ := 60

-- Define the costs of items
def celeryCost : ℕ := 5
def cerealCost : ℕ := 12 / 2
def breadCost : ℕ := 8
def milkCost : ℕ := 10 * 9 / 10
def potatoCostEach : ℕ := 1
def numberOfPotatoes : ℕ := 6
def totalPotatoCost : ℕ := potatoCostEach * numberOfPotatoes

-- Define the total amount spent on the items
def totalSpent : ℕ := celeryCost + cerealCost + breadCost + milkCost + totalPotatoCost

-- Define the amount left for coffee
def amountLeftForCoffee : ℕ := initialMoney - totalSpent

-- The theorem to prove
theorem lily_has_26_dollars_left_for_coffee :
  amountLeftForCoffee = 26 := by
  sorry

end lily_has_26_dollars_left_for_coffee_l97_97047


namespace polar_to_cartesian_circle_l97_97532

theorem polar_to_cartesian_circle :
  ∀ (r : ℝ) (x y : ℝ), r = 3 → r = Real.sqrt (x^2 + y^2) → x^2 + y^2 = 9 :=
by
  intros r x y hr h
  sorry

end polar_to_cartesian_circle_l97_97532


namespace pentagon_area_l97_97476

noncomputable def angle_F := 100
noncomputable def angle_G := 100
noncomputable def JF := 3
noncomputable def FG := 3
noncomputable def GH := 3
noncomputable def HI := 5
noncomputable def IJ := 5
noncomputable def area_FGHIJ := 9 * Real.sqrt 3 + Real.sqrt 17.1875

theorem pentagon_area : area_FGHIJ = 9 * Real.sqrt 3 + Real.sqrt 17.1875 :=
by
  sorry

end pentagon_area_l97_97476


namespace jenny_ran_further_l97_97991

-- Define the distances Jenny ran and walked
def ran_distance : ℝ := 0.6
def walked_distance : ℝ := 0.4

-- Define the difference between the distances Jenny ran and walked
def difference : ℝ := ran_distance - walked_distance

-- The proof statement
theorem jenny_ran_further : difference = 0.2 := by
  sorry

end jenny_ran_further_l97_97991


namespace greatest_common_length_cords_l97_97172

theorem greatest_common_length_cords (l1 l2 l3 l4 : ℝ) (h1 : l1 = Real.sqrt 20) (h2 : l2 = Real.pi) (h3 : l3 = Real.exp 1) (h4 : l4 = Real.sqrt 98) : 
  ∃ d : ℝ, d = 1 ∧ (∀ k1 k2 k3 k4 : ℝ, k1 * d = l1 → k2 * d = l2 → k3 * d = l3 → k4 * d = l4 → ∀i : ℝ, i = d) :=
by
  sorry

end greatest_common_length_cords_l97_97172


namespace prob_A_wins_match_is_correct_l97_97179

/-- Definitions -/

def prob_A_wins_game : ℝ := 0.6

def prob_B_wins_game : ℝ := 1 - prob_A_wins_game

def prob_A_wins_match (p: ℝ) : ℝ :=
  p * p * (1 - p) + p * (1 - p) * p + p * p

/-- Theorem -/

theorem prob_A_wins_match_is_correct : 
  prob_A_wins_match prob_A_wins_game = 0.648 :=
by
  sorry

end prob_A_wins_match_is_correct_l97_97179


namespace each_child_apples_l97_97528

-- Define the given conditions
def total_apples : ℕ := 450
def num_adults : ℕ := 40
def num_adults_apples : ℕ := 3
def num_children : ℕ := 33

-- Define the theorem to prove
theorem each_child_apples : 
  let total_apples_eaten_by_adults := num_adults * num_adults_apples
  let total_apples_for_children := total_apples - total_apples_eaten_by_adults
  let apples_per_child := total_apples_for_children / num_children
  apples_per_child = 10 :=
by
  sorry

end each_child_apples_l97_97528


namespace greater_number_is_64_l97_97604

-- Proof statement: The greater number (y) is 64 given the conditions
theorem greater_number_is_64 (x y : ℕ) 
    (h1 : y = 2 * x) 
    (h2 : x + y = 96) : 
    y = 64 := 
sorry

end greater_number_is_64_l97_97604


namespace circle_through_points_line_perpendicular_and_tangent_to_circle_max_area_triangle_l97_97814

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 10

theorem circle_through_points (A B : ℝ × ℝ) (hA : A = (2, 2)) (hB : B = (6, 0)) (h_center : ∃ C: ℝ × ℝ, C.1 - C.2 - 4 = 0 ∧ (circle_eq C.1 C.2)) : ∀ x y, circle_eq x y ↔ (x - 3) ^ 2 + (y + 1) ^ 2 = 10 := 
by sorry

theorem line_perpendicular_and_tangent_to_circle (line_slope : ℝ) (tangent : ∀ x y, circle_eq x y → (x + 3*y + 10 = 0 ∨ x + 3*y - 10 = 0)) : ∀ x, x + 3*y + 10 = 0 ∨ x + 3*y - 10 = 0 :=
by sorry

theorem max_area_triangle (A B P : ℝ × ℝ) (hA : A = (2, 2)) (hB : B = (6, 0)) (hP : circle_eq P.1 P.2) : ∃ area : ℝ, area = 5 + 5 * Real.sqrt 2
:= 
by sorry

end circle_through_points_line_perpendicular_and_tangent_to_circle_max_area_triangle_l97_97814


namespace find_number_l97_97548

theorem find_number (x : ℝ) (hx : (50 + 20 / x) * x = 4520) : x = 90 :=
sorry

end find_number_l97_97548


namespace area_EHF_leq_one_fourth_area_ABCD_not_general_statement_for_arbitrary_quadrilateral_l97_97457

open EuclideanGeometry
open Real
open Set

variables {A B C D E F H G : Point}

-- Defining the trapezoid and its properties
def is_trapezoid (A B C D : Point) : Prop :=
  parallel A B C D ∧ ∃ E F H G : Point, 
    (lies_on_segment E A B) ∧ 
    (lies_on_segment F C D) ∧ 
    (intersects_at H C E B F) ∧ 
    (intersects_at G E D A F)

-- The theorem statement for the area of EHF
theorem area_EHF_leq_one_fourth_area_ABCD 
  (A B C D E F H G : Point) :
  is_trapezoid A B C D → 
  (area_triangle E H F) ≤ (1/4 * area_trapezoid A B C D) := 
sorry

-- The theorem statement for an arbitrary convex quadrilateral
theorem not_general_statement_for_arbitrary_quadrilateral 
  (A B C D E F H G : Point) :
  is_convex_quadrilateral A B C D →
  ¬( (area_triangle E H F) ≤ (1/4 * area_quadrilateral A B C D) ) :=
sorry

end area_EHF_leq_one_fourth_area_ABCD_not_general_statement_for_arbitrary_quadrilateral_l97_97457


namespace coconut_tree_difference_l97_97054

-- Define the known quantities
def mango_trees : ℕ := 60
def total_trees : ℕ := 85
def half_mango_trees : ℕ := 30 -- half of 60
def coconut_trees : ℕ := 25 -- 85 - 60

-- Define the proof statement
theorem coconut_tree_difference : (half_mango_trees - coconut_trees) = 5 := by
  -- The proof steps are given
  sorry

end coconut_tree_difference_l97_97054


namespace arithmetic_progression_condition_l97_97679

theorem arithmetic_progression_condition
  (a b c : ℝ) (a1 d : ℝ) (p n k : ℕ) :
  a = a1 + (p - 1) * d →
  b = a1 + (n - 1) * d →
  c = a1 + (k - 1) * d →
  a * (n - k) + b * (k - p) + c * (p - n) = 0 :=
by
  intros h1 h2 h3
  sorry


end arithmetic_progression_condition_l97_97679


namespace circle_area_sum_l97_97920

theorem circle_area_sum (x y z : ℕ) (A₁ A₂ A₃ total_area : ℕ) (h₁ : A₁ = 6) (h₂ : A₂ = 15) 
  (h₃ : A₃ = 83) (h₄ : total_area = 220) (hx : x = 4) (hy : y = 2) (hz : z = 2) :
  A₁ * x + A₂ * y + A₃ * z = total_area := by
  sorry

end circle_area_sum_l97_97920


namespace bottle_caps_total_l97_97527

-- Define the conditions
def groups : ℕ := 7
def caps_per_group : ℕ := 5

-- State the theorem
theorem bottle_caps_total : groups * caps_per_group = 35 :=
by
  sorry

end bottle_caps_total_l97_97527


namespace num_common_points_l97_97136

noncomputable def curve (x : ℝ) : ℝ := 3 * x ^ 4 - 2 * x ^ 3 - 9 * x ^ 2 + 4

noncomputable def tangent_line (x : ℝ) : ℝ :=
  -12 * (x - 1) - 4

theorem num_common_points :
  ∃ (x1 x2 x3 : ℝ), curve x1 = tangent_line x1 ∧
                    curve x2 = tangent_line x2 ∧
                    curve x3 = tangent_line x3 ∧
                    (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) :=
sorry

end num_common_points_l97_97136


namespace intersection_point_of_lines_l97_97140

theorem intersection_point_of_lines (x y : ℝ) :
  (2 * x - 3 * y = 3) ∧ (4 * x + 2 * y = 2) ↔ (x = 3/4) ∧ (y = -1/2) :=
by
  sorry

end intersection_point_of_lines_l97_97140


namespace absolute_diff_half_l97_97009

theorem absolute_diff_half (x y : ℝ) 
  (h : ((x + y = x - y ∧ x - y = x * y) ∨ 
       (x + y = x * y ∧ x * y = x / y) ∨ 
       (x - y = x * y ∧ x * y = x / y))
       ∧ x ≠ 0 ∧ y ≠ 0) : 
     |y| - |x| = 1 / 2 := 
sorry

end absolute_diff_half_l97_97009


namespace sum_of_decimals_l97_97388

-- Defining the specific decimal values as constants
def x : ℝ := 5.47
def y : ℝ := 4.26

-- Noncomputable version for addition to allow Lean to handle real number operations safely
noncomputable def sum : ℝ := x + y

-- Theorem statement asserting the sum of x and y
theorem sum_of_decimals : sum = 9.73 := 
by
  -- This is where the proof would go
  sorry

end sum_of_decimals_l97_97388


namespace sara_payment_equivalence_l97_97372

variable (cost_book1 cost_book2 change final_amount : ℝ)

theorem sara_payment_equivalence
  (h1 : cost_book1 = 5.5)
  (h2 : cost_book2 = 6.5)
  (h3 : change = 8)
  (h4 : final_amount = cost_book1 + cost_book2 + change) :
  final_amount = 20 := by
  sorry

end sara_payment_equivalence_l97_97372


namespace certain_number_approximation_l97_97696

theorem certain_number_approximation (h1 : 2994 / 14.5 = 177) (h2 : 29.94 / x = 17.7) : x = 2.57455 := by
  sorry

end certain_number_approximation_l97_97696


namespace rectangle_width_l97_97759

theorem rectangle_width (L W : ℝ) 
  (h1 : L * W = 300)
  (h2 : 2 * L + 2 * W = 70) : 
  W = 15 :=
by 
  -- We prove the width W of the rectangle is 15 meters.
  sorry

end rectangle_width_l97_97759


namespace relationship_teachers_students_l97_97015

variables (m n k l : ℕ)

theorem relationship_teachers_students :
  ∀ (m n k l : ℕ) (h1: m > 0) (h2: n > 1) (h3: k > 0) (h4: l > 0),
  (∑ i in finset.range m, ∑ j in finset.range k, i ≠ j) * 1/2  =  
  (∑ i in finset.range n, ∑ j in finset.range k, i ≠ j) * 1/2 :=
  
  m * k * (k - 1) = n * (n - 1) * l := 
begin
  sorry
end

end relationship_teachers_students_l97_97015


namespace floor_neg_seven_over_four_l97_97264

theorem floor_neg_seven_over_four : (Int.floor (-7 / 4 : ℚ)) = -2 := 
by
  sorry

end floor_neg_seven_over_four_l97_97264


namespace solution_set_l97_97870

theorem solution_set (x : ℝ) : (⌊x⌋ + ⌈x⌉ = 7) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l97_97870


namespace rotated_translated_line_eq_l97_97354

theorem rotated_translated_line_eq :
  ∀ (x y : ℝ), y = 3 * x → y = - (1 / 3) * x + (1 / 3) :=
by
  sorry

end rotated_translated_line_eq_l97_97354


namespace floor_neg_seven_over_four_l97_97265

theorem floor_neg_seven_over_four : (Int.floor (-7 / 4 : ℚ)) = -2 := 
by
  sorry

end floor_neg_seven_over_four_l97_97265


namespace probability_at_most_six_distinct_numbers_l97_97882

def roll_eight_dice : ℕ := 6^8

def favorable_cases : ℕ := 3628800

def probability_six_distinct_numbers (n : ℕ) (f : ℕ) : ℚ :=
  f / n

theorem probability_at_most_six_distinct_numbers :
  probability_six_distinct_numbers roll_eight_dice favorable_cases = 45 / 52 := by
  sorry

end probability_at_most_six_distinct_numbers_l97_97882


namespace minimum_cards_to_ensure_60_of_same_color_l97_97362

-- Define the conditions as Lean definitions
def total_cards : ℕ := 700
def ratio_red_orange_yellow : ℕ × ℕ × ℕ := (1, 3, 4)
def ratio_green_blue_white : ℕ × ℕ × ℕ := (3, 1, 6)
def yellow_more_than_blue : ℕ := 50

-- Define the proof goal
theorem minimum_cards_to_ensure_60_of_same_color :
  ∀ (x y : ℕ),
  (total_cards = (1 * x + 3 * x + 4 * x + 3 * y + y + 6 * y)) ∧
  (4 * x = y + yellow_more_than_blue) →
  min_cards :=
  -- Sorry here to indicate that proof is not provided
  sorry

end minimum_cards_to_ensure_60_of_same_color_l97_97362


namespace eggs_left_for_sunny_side_up_l97_97203

-- Given conditions:
def ordered_dozen_eggs : ℕ := 3 * 12
def eggs_used_for_crepes (total_eggs : ℕ) : ℕ := total_eggs * 1 / 4
def eggs_after_crepes (total_eggs : ℕ) (used_for_crepes : ℕ) : ℕ := total_eggs - used_for_crepes
def eggs_used_for_cupcakes (remaining_eggs : ℕ) : ℕ := remaining_eggs * 2 / 3
def eggs_left (remaining_eggs : ℕ) (used_for_cupcakes : ℕ) : ℕ := remaining_eggs - used_for_cupcakes

-- Proposition:
theorem eggs_left_for_sunny_side_up : 
  eggs_left (eggs_after_crepes ordered_dozen_eggs (eggs_used_for_crepes ordered_dozen_eggs)) 
            (eggs_used_for_cupcakes (eggs_after_crepes ordered_dozen_eggs (eggs_used_for_crepes ordered_dozen_eggs))) = 9 :=
sorry

end eggs_left_for_sunny_side_up_l97_97203


namespace periodic_function_proof_l97_97796

variable (f : ℝ → ℝ)

theorem periodic_function_proof (h1 : ∀ x : ℝ, f x * f (x + 2) = 1)
  (h2 : f 1 = 3)
  (h3 : f 2 = 2) :
  f 2014 = 2 :=
begin
  sorry -- proof not provided
end

end periodic_function_proof_l97_97796


namespace floor_of_neg_seven_fourths_l97_97261

theorem floor_of_neg_seven_fourths : 
  Int.floor (-7 / 4 : ℚ) = -2 := 
by sorry

end floor_of_neg_seven_fourths_l97_97261


namespace eval_imaginary_expression_l97_97944

theorem eval_imaginary_expression :
  ∀ (i : ℂ), i^2 = -1 → i^2022 + i^2023 + i^2024 + i^2025 = 0 :=
by
  sorry

end eval_imaginary_expression_l97_97944


namespace total_earnings_l97_97234

-- Definitions from the conditions.
def LaurynEarnings : ℝ := 2000
def AureliaEarnings : ℝ := 0.7 * LaurynEarnings

-- The theorem to prove.
theorem total_earnings (hL : LaurynEarnings = 2000) (hA : AureliaEarnings = 0.7 * 2000) :
    LaurynEarnings + AureliaEarnings = 3400 :=
by
    sorry  -- Placeholder for the proof.

end total_earnings_l97_97234


namespace moles_of_water_l97_97946

-- Definitions related to the reaction conditions.
def HCl : Type := sorry
def NaHCO3 : Type := sorry
def NaCl : Type := sorry
def H2O : Type := sorry
def CO2 : Type := sorry

def reaction (h : HCl) (n : NaHCO3) : Nat := sorry -- Represents the balanced reaction

-- Given conditions in Lean.
axiom one_mole_HCl : HCl
axiom one_mole_NaHCO3 : NaHCO3
axiom balanced_equation : reaction one_mole_HCl one_mole_NaHCO3 = 1 -- 1 mole of water is produced

-- The theorem to prove.
theorem moles_of_water : reaction one_mole_HCl one_mole_NaHCO3 = 1 :=
by
  -- The proof would go here
  sorry

end moles_of_water_l97_97946


namespace tangerines_left_proof_l97_97447

-- Define the number of tangerines Jimin ate
def tangerinesJiminAte : ℕ := 7

-- Define the total number of tangerines
def totalTangerines : ℕ := 12

-- Define the number of tangerines left
def tangerinesLeft : ℕ := totalTangerines - tangerinesJiminAte

-- Theorem stating the number of tangerines left equals 5
theorem tangerines_left_proof : tangerinesLeft = 5 := 
by
  sorry

end tangerines_left_proof_l97_97447


namespace joan_sandwiches_l97_97839

theorem joan_sandwiches :
  ∀ (H : ℕ), (∀ (h_slice g_slice total_cheese num_grilled_cheese : ℕ),
  h_slice = 2 →
  g_slice = 3 →
  num_grilled_cheese = 10 →
  total_cheese = 50 →
  total_cheese - num_grilled_cheese * g_slice = H * h_slice →
  H = 10) :=
by
  intros H h_slice g_slice total_cheese num_grilled_cheese h_slice_eq g_slice_eq num_grilled_cheese_eq total_cheese_eq cheese_eq
  sorry

end joan_sandwiches_l97_97839


namespace triangle_area_ratio_l97_97309

theorem triangle_area_ratio
  (a b c : ℕ) (S_triangle : ℕ) -- assuming S_triangle represents the area of the original triangle
  (S_bisected_triangle : ℕ) -- assuming S_bisected_triangle represents the area of the bisected triangle
  (is_angle_bisector : ∀ x y z : ℕ, ∃ k, k = (2 * a * b * c * x) / ((a + b) * (a + c) * (b + c))) :
  S_bisected_triangle = (2 * a * b * c) / ((a + b) * (a + c) * (b + c)) * S_triangle :=
sorry

end triangle_area_ratio_l97_97309


namespace floor_neg_seven_over_four_l97_97266

theorem floor_neg_seven_over_four : (Int.floor (-7 / 4 : ℚ)) = -2 := 
by
  sorry

end floor_neg_seven_over_four_l97_97266


namespace division_quotient_l97_97441

-- Define conditions
def dividend : ℕ := 686
def divisor : ℕ := 36
def remainder : ℕ := 2

-- Define the quotient
def quotient : ℕ := dividend - remainder

theorem division_quotient :
  quotient = divisor * 19 :=
sorry

end division_quotient_l97_97441


namespace JakePresentWeight_l97_97432

def JakeWeight (J S : ℕ) : Prop :=
  J - 33 = 2 * S ∧ J + S = 153

theorem JakePresentWeight : ∃ (J : ℕ), ∃ (S : ℕ), JakeWeight J S ∧ J = 113 := 
by
  sorry

end JakePresentWeight_l97_97432


namespace sum_2001_and_1015_l97_97496

theorem sum_2001_and_1015 :
  2001 + 1015 = 3016 :=
sorry

end sum_2001_and_1015_l97_97496


namespace product_base7_eq_l97_97657

-- Definitions for the numbers in base 7
def num325_base7 := 3 * 7^2 + 2 * 7^1 + 5 * 7^0  -- 325 in base 7
def num4_base7 := 4 * 7^0  -- 4 in base 7

-- Theorem stating that the product of 325_7 and 4_7 in base 7 is 1636_7
theorem product_base7_eq : 
  let product_base10 := num325_base7 * num4_base7 in
  (product_base10 = 1 * 7^3 + 6 * 7^2 + 3 * 7^1 + 6 * 7^0) :=
by sorry

end product_base7_eq_l97_97657


namespace part1_A_complement_B_intersection_eq_part2_m_le_neg2_part3_m_ge_4_l97_97717

def set_A : Set ℝ := {x | x^2 - 2*x - 8 < 0}
def set_B (m : ℝ) : Set ℝ := {x | x < m}

-- Problem 1
theorem part1_A_complement_B_intersection_eq (m : ℝ) (h : m = 3) :
  set_A ∩ {x | x >= 3} = {x | 3 <= x ∧ x < 4} :=
sorry

-- Problem 2
theorem part2_m_le_neg2 (m : ℝ) (h : set_A ∩ set_B m = ∅) :
  m <= -2 :=
sorry

-- Problem 3
theorem part3_m_ge_4 (m : ℝ) (h : set_A ∩ set_B m = set_A) :
  m >= 4 :=
sorry

end part1_A_complement_B_intersection_eq_part2_m_le_neg2_part3_m_ge_4_l97_97717


namespace ab_bc_ca_fraction_l97_97038

theorem ab_bc_ca_fraction (a b c : ℝ) (h1 : a + b + c = 7) (h2 : a * b + a * c + b * c = 10) (h3 : a * b * c = 12) :
    (a * b / c) + (b * c / a) + (c * a / b) = -17 / 3 := 
    sorry

end ab_bc_ca_fraction_l97_97038


namespace cyclic_sequence_u_16_eq_a_l97_97379

-- Sequence definition and recurrence relation
def cyclic_sequence (u : ℕ → ℝ) (a : ℝ) : Prop :=
  u 1 = a ∧ ∀ n : ℕ, u (n + 1) = -1 / (u n + 1)

-- Proof that u_{16} = a under given conditions
theorem cyclic_sequence_u_16_eq_a (a : ℝ) (h : 0 < a) : ∃ (u : ℕ → ℝ), cyclic_sequence u a ∧ u 16 = a :=
by
  sorry

end cyclic_sequence_u_16_eq_a_l97_97379


namespace line_parabola_one_intersection_not_tangent_l97_97827

theorem line_parabola_one_intersection_not_tangent {A B C D : ℝ} (h: ∀ x : ℝ, ((A * x ^ 2 + B * x + C) = D) → False) :
  ¬ ∃ x : ℝ, (A * x ^ 2 + B * x + C) = D ∧ 2 * x * A + B = 0 := sorry

end line_parabola_one_intersection_not_tangent_l97_97827


namespace melissa_driving_time_l97_97333

theorem melissa_driving_time
  (trips_per_month: ℕ)
  (months_per_year: ℕ)
  (total_hours_per_year: ℕ)
  (total_trips: ℕ)
  (hours_per_trip: ℕ) :
  trips_per_month = 2 ∧
  months_per_year = 12 ∧
  total_hours_per_year = 72 ∧
  total_trips = (trips_per_month * months_per_year) ∧
  hours_per_trip = (total_hours_per_year / total_trips) →
  hours_per_trip = 3 :=
by
  intro h
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  sorry

end melissa_driving_time_l97_97333


namespace arccos_half_eq_pi_div_three_l97_97926

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
by
  sorry

end arccos_half_eq_pi_div_three_l97_97926


namespace cara_age_is_40_l97_97924

-- Defining the conditions
def grandmother_age : ℕ := 75
def mom_age : ℕ := grandmother_age - 15
def cara_age : ℕ := mom_age - 20

-- Proving the question
theorem cara_age_is_40 : cara_age = 40 := by
  sorry

end cara_age_is_40_l97_97924


namespace opposite_of_number_l97_97065

-- Define the original number
def original_number : ℚ := -1 / 6

-- Statement to prove
theorem opposite_of_number : -original_number = 1 / 6 := by
  -- This is where the proof would go
  sorry

end opposite_of_number_l97_97065


namespace translation_of_segment_l97_97706

structure Point where
  x : ℝ
  y : ℝ

variables (A B A' : Point)

def translation_vector (P Q : Point) : Point :=
  { x := Q.x - P.x,
    y := Q.y - P.y }

def translate (P Q : Point) : Point :=
  { x := P.x + Q.x,
    y := P.y + Q.y }

theorem translation_of_segment (hA : A = {x := -2, y := 0})
                                (hB : B = {x := 0, y := 3})
                                (hA' : A' = {x := 2, y := 1}) :
  translate B (translation_vector A A') = {x := 4, y := 4} := by
  sorry

end translation_of_segment_l97_97706


namespace similar_right_triangles_l97_97644

theorem similar_right_triangles (x c : ℕ) 
  (h1 : 12 * 6 = 9 * x) 
  (h2 : c^2 = x^2 + 6^2) :
  x = 8 ∧ c = 10 :=
by
  sorry

end similar_right_triangles_l97_97644


namespace region_area_l97_97752

noncomputable def area_of_region := 4 * Real.pi

theorem region_area :
  (∃ x y, x^2 + y^2 - 4 * x + 2 * y + 1 = 0) →
  Real.pi * 4 = area_of_region :=
by
  sorry

end region_area_l97_97752


namespace area_of_triangle_PQR_is_correct_l97_97095

noncomputable def calculate_area_of_triangle_PQR : ℝ := 
  let side_length := 4
  let altitude := 8
  let WO := (side_length * Real.sqrt 2) / 2
  let center_to_vertex_distance := Real.sqrt (WO^2 + altitude^2)
  let WP := (1 / 4) * WO
  let YQ := (1 / 2) * WO
  let XR := (3 / 4) * WO
  let area := (1 / 2) * (YQ - WP) * (XR - YQ)
  area

theorem area_of_triangle_PQR_is_correct :
  calculate_area_of_triangle_PQR = 2.25 := sorry

end area_of_triangle_PQR_is_correct_l97_97095


namespace fraction_equality_l97_97693

variables {R : Type*} [Field R] {m n p q : R}

theorem fraction_equality 
  (h1 : m / n = 15)
  (h2 : p / n = 3)
  (h3 : p / q = 1 / 10) :
  m / q = 1 / 2 :=
sorry

end fraction_equality_l97_97693


namespace value_of_A_is_18_l97_97897

theorem value_of_A_is_18
  (A B C D : ℕ)
  (h1 : A ≠ B)
  (h2 : A ≠ C)
  (h3 : A ≠ D)
  (h4 : B ≠ C)
  (h5 : B ≠ D)
  (h6 : C ≠ D)
  (h7 : A * B = 72)
  (h8 : C * D = 72)
  (h9 : A - B = C + D) : A = 18 :=
sorry

end value_of_A_is_18_l97_97897


namespace exp_graph_fixed_point_l97_97350

theorem exp_graph_fixed_point (a : ℝ) :
  ∃ (x y : ℝ), x = 3 ∧ y = 4 ∧ y = a^(x - 3) + 3 :=
by
  use 3
  use 4
  split
  · rfl
  split
  · rfl
  · sorry

end exp_graph_fixed_point_l97_97350


namespace lean_proof_l97_97145

theorem lean_proof (a : ℝ) (h : a = real.cos (2 * real.pi / 7)) : 2^(a - 1/2) < 2 * a :=
sorry

end lean_proof_l97_97145


namespace problem_statement_l97_97768

theorem problem_statement :
  ∀ x a k n : ℤ, 
  (3 * x + 2) * (2 * x - 7) = a * x^2 + k * x + n → a - n + k = 3 :=
by  
  sorry

end problem_statement_l97_97768


namespace complex_number_fourth_quadrant_l97_97078

theorem complex_number_fourth_quadrant (m : ℝ) (h1 : 2/3 < m) (h2 : m < 1) : 
  (3 * m - 2) > 0 ∧ (m - 1) < 0 := 
by 
  sorry

end complex_number_fourth_quadrant_l97_97078


namespace mike_passing_percentage_l97_97464

theorem mike_passing_percentage (scored shortfall max_marks : ℝ) (total_marks := scored + shortfall) :
    scored = 212 →
    shortfall = 28 →
    max_marks = 800 →
    (total_marks / max_marks) * 100 = 30 :=
by
  intros
  sorry

end mike_passing_percentage_l97_97464


namespace inv_203_mod_301_exists_l97_97935

theorem inv_203_mod_301_exists : ∃ b : ℤ, 203 * b % 301 = 1 := sorry

end inv_203_mod_301_exists_l97_97935


namespace solve_real_equation_l97_97285

theorem solve_real_equation (x : ℝ) :
  x^2 * (x + 1)^2 + x^2 = 3 * (x + 1)^2 ↔ x = (1 + Real.sqrt 5) / 2 ∨ x = (1 - Real.sqrt 5) / 2 :=
by sorry

end solve_real_equation_l97_97285


namespace matthew_initial_crackers_l97_97332

theorem matthew_initial_crackers :
  ∃ C : ℕ,
  (∀ (crackers_per_friend cakes_per_friend : ℕ), cakes_per_friend * 4 = 98 → crackers_per_friend = cakes_per_friend → crackers_per_friend * 4 + 8 * 4 = C) ∧ C = 128 :=
sorry

end matthew_initial_crackers_l97_97332


namespace train_speed_approx_kmph_l97_97914

noncomputable def length_of_train : ℝ := 150
noncomputable def time_to_cross_pole : ℝ := 4.425875438161669

theorem train_speed_approx_kmph :
  (length_of_train / time_to_cross_pole) * 3.6 = 122.03 :=
by sorry

end train_speed_approx_kmph_l97_97914


namespace orange_shells_correct_l97_97728

def total_shells : Nat := 65
def purple_shells : Nat := 13
def pink_shells : Nat := 8
def yellow_shells : Nat := 18
def blue_shells : Nat := 12
def orange_shells : Nat := total_shells - (purple_shells + pink_shells + yellow_shells + blue_shells)

theorem orange_shells_correct : orange_shells = 14 :=
by
  sorry

end orange_shells_correct_l97_97728


namespace cost_price_of_article_l97_97867

theorem cost_price_of_article (C : ℝ) (h1 : 86 - C = C - 42) : C = 64 :=
by
  sorry

end cost_price_of_article_l97_97867


namespace problem_statement_l97_97762

theorem problem_statement (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1/a) + Real.sqrt (b + 1/b) + Real.sqrt (c + 1/c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) :=
sorry

end problem_statement_l97_97762


namespace yolanda_walking_rate_l97_97373

-- Definitions for the conditions given in the problem
variables (X Y : ℝ) -- Points X and Y
def distance_X_to_Y := 52 -- Distance between X and Y in miles
def Bob_rate := 4 -- Bob's walking rate in miles per hour
def Bob_distance_walked := 28 -- The distance Bob walked in miles
def start_time_diff := 1 -- The time difference (in hours) between Yolanda and Bob starting

-- The statement to prove
theorem yolanda_walking_rate : 
  ∃ (y : ℝ), (distance_X_to_Y = y * (Bob_distance_walked / Bob_rate + start_time_diff) + Bob_distance_walked) ∧ y = 3 := by 
  sorry

end yolanda_walking_rate_l97_97373


namespace positive_integers_p_divisibility_l97_97284

theorem positive_integers_p_divisibility (p : ℕ) (hp : 0 < p) :
  (∃ n : ℕ, 0 < n ∧ p^n + 3^n ∣ p^(n+1) + 3^(n+1)) ↔ p = 3 ∨ p = 6 ∨ p = 15 :=
by sorry

end positive_integers_p_divisibility_l97_97284


namespace smallest_positive_integer_l97_97623

theorem smallest_positive_integer (
    b : ℤ 
) : 
    (b % 4 = 1) → (b % 5 = 2) → (b % 6 = 3) → b = 21 := 
by
  intros h1 h2 h3
  sorry

end smallest_positive_integer_l97_97623


namespace arrangement_exists_l97_97099

-- Definitions of pairwise coprimeness and gcd
def pairwise_coprime (a b c d : ℕ) : Prop :=
  Nat.gcd a b = 1 ∧ Nat.gcd a c = 1 ∧ Nat.gcd a d = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd b d = 1 ∧ Nat.gcd c d = 1

def common_divisor (x y : ℕ) : Prop := ∃ d > 1, d ∣ x ∧ d ∣ y

def relatively_prime (x y : ℕ) : Prop := Nat.gcd x y = 1

-- Main theorem statement
theorem arrangement_exists :
  ∃ a b c d ab cd ad bc abcd : ℕ,
    pairwise_coprime a b c d ∧
    ab = a * b ∧ cd = c * d ∧ ad = a * d ∧ bc = b * c ∧ abcd = a * b * c * d ∧
    (common_divisor ab abcd ∧ common_divisor cd abcd ∧ common_divisor ad abcd ∧ common_divisor bc abcd) ∧
    (common_divisor ab ad ∧ common_divisor ab bc ∧ common_divisor cd ad ∧ common_divisor cd bc) ∧
    (relatively_prime ab cd ∧ relatively_prime ad bc) :=
by
  -- The proof will be filled here
  sorry

end arrangement_exists_l97_97099


namespace cupcake_packages_l97_97764

theorem cupcake_packages (total_cupcakes eaten_cupcakes cupcakes_per_package number_of_packages : ℕ) 
  (h1 : total_cupcakes = 18)
  (h2 : eaten_cupcakes = 8)
  (h3 : cupcakes_per_package = 2)
  (h4 : number_of_packages = (total_cupcakes - eaten_cupcakes) / cupcakes_per_package) :
  number_of_packages = 5 :=
by
  -- The proof goes here, we'll use sorry to indicate it's not needed for now.
  sorry

end cupcake_packages_l97_97764


namespace blender_customers_l97_97937

variable (p_t p_b : ℕ) (c_t c_b : ℕ) (k : ℕ)

-- Define the conditions
def condition_toaster_popularity : p_t = 20 := sorry
def condition_toaster_cost : c_t = 300 := sorry
def condition_blender_cost : c_b = 450 := sorry
def condition_inverse_proportionality : p_t * c_t = k := sorry

-- Proof goal: number of customers who would buy the blender
theorem blender_customers : p_b = 13 :=
by
  have h1 : p_t * c_t = 6000 := by sorry -- Using the given conditions
  have h2 : p_b * c_b = 6000 := by sorry -- Assumption for the same constant k
  have h3 : c_b = 450 := sorry
  have h4 : p_b = 6000 / 450 := by sorry
  have h5 : p_b = 13 := by sorry
  exact h5

end blender_customers_l97_97937


namespace M_subset_N_l97_97139

noncomputable def M_set : Set ℝ := { x | ∃ (k : ℤ), x = k / 4 + 1 / 4 }
noncomputable def N_set : Set ℝ := { x | ∃ (k : ℤ), x = k / 8 - 1 / 4 }

theorem M_subset_N : M_set ⊆ N_set :=
sorry

end M_subset_N_l97_97139


namespace fraction_diff_l97_97556

open Real

theorem fraction_diff (x y : ℝ) (hx : x = sqrt 5 - 1) (hy : y = sqrt 5 + 1) :
  (1 / x - 1 / y) = 1 / 2 := sorry

end fraction_diff_l97_97556


namespace f_even_of_g_odd_l97_97040

theorem f_even_of_g_odd (g : ℝ → ℝ) (f : ℝ → ℝ) (h1 : ∀ x, g (-x) = -g x) (h2 : ∀ x, f x = |g (x^5)|) : ∀ x, f (-x) = f x := 
by
  sorry

end f_even_of_g_odd_l97_97040


namespace arccos_one_half_eq_pi_div_three_l97_97930

theorem arccos_one_half_eq_pi_div_three :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ π ∧ cos θ = (1 / 2) ∧ arccos (1 / 2) = θ :=
sorry

end arccos_one_half_eq_pi_div_three_l97_97930


namespace Caleb_pencils_fewer_than_twice_Candy_l97_97659

theorem Caleb_pencils_fewer_than_twice_Candy:
  ∀ (P_Caleb P_Candy: ℕ), 
    P_Candy = 9 → 
    (∃ X, P_Caleb = 2 * P_Candy - X) → 
    P_Caleb + 5 - 10 = 10 → 
    (2 * P_Candy - P_Caleb = 3) :=
by
  intros P_Caleb P_Candy hCandy hCalebLess twCalen
  sorry

end Caleb_pencils_fewer_than_twice_Candy_l97_97659


namespace eval_floor_neg_seven_fourths_l97_97256

theorem eval_floor_neg_seven_fourths : 
  ∃ (x : ℚ), x = -7 / 4 ∧ ∀ y, y ≤ x ∧ y ∈ ℤ → y ≤ -2 :=
by
  obtain ⟨x, hx⟩ : ∃ (x : ℚ), x = -7 / 4 := ⟨-7 / 4, rfl⟩,
  use x,
  split,
  { exact hx },
  { intros y hy,
    sorry }

end eval_floor_neg_seven_fourths_l97_97256


namespace total_cases_sold_l97_97490

theorem total_cases_sold : 
  let people := 20 in
  let first_8_cases := 8 * 3 in
  let next_4_cases := 4 * 2 in
  let last_8_cases := 8 * 1 in
  first_8_cases + next_4_cases + last_8_cases = 40 := 
by
  let people := 20
  let first_8_cases := 8 * 3
  let next_4_cases := 4 * 2
  let last_8_cases := 8 * 1
  have h1 : first_8_cases = 24 := by rfl
  have h2 : next_4_cases = 8 := by rfl
  have h3 : last_8_cases = 8 := by rfl
  have h : first_8_cases + next_4_cases + last_8_cases = 24 + 8 + 8 := by rw [h1, h2, h3]
  show 24 + 8 + 8 = 40 from rfl

end total_cases_sold_l97_97490


namespace gross_profit_without_discount_l97_97628

variable (C P : ℝ) -- Defining the cost and the full price as real numbers

-- Condition 1: Merchant sells an item at 10% discount (0.9P)
-- Condition 2: Makes a gross profit of 20% of the cost (0.2C)
-- SP = C + GP implies 0.9 P = 1.2 C

theorem gross_profit_without_discount :
  (0.9 * P = 1.2 * C) → ((C / 3) / C * 100 = 33.33) :=
by
  intro h
  sorry

end gross_profit_without_discount_l97_97628


namespace factorization_l97_97282

theorem factorization (x y : ℝ) : 
  (x + y) ^ 2 + 4 * (x - y) ^ 2 - 4 * (x ^ 2 - y ^ 2) = (x - 3 * y) ^ 2 :=
by
  sorry

end factorization_l97_97282


namespace speed_is_90_l97_97112

namespace DrivingSpeedProof

/-- Given the observation times and marker numbers, prove the speed of the car is 90 km/hr. -/
theorem speed_is_90 
  (X Y : ℕ)
  (h0 : X ≥ 0) (h1 : X ≤ 9)
  (h2 : Y = 8 * X)
  (h3 : Y ≥ 0) (h4 : Y ≤ 9)
  (noon_marker : 10 * X + Y = 18)
  (second_marker : 10 * Y + X = 81)
  (third_marker : 100 * X + Y = 108)
  : 90 = 90 :=
by {
  sorry
}

end DrivingSpeedProof

end speed_is_90_l97_97112


namespace time_to_build_wall_l97_97771

theorem time_to_build_wall (t_A t_B t_C : ℝ) 
  (h1 : 1 / t_A + 1 / t_B = 1 / 25)
  (h2 : 1 / t_C = 1 / 35)
  (h3 : 1 / t_A = 1 / t_B + 1 / t_C) : t_B = 87.5 :=
by
  sorry

end time_to_build_wall_l97_97771


namespace cube_volume_given_face_area_l97_97184

theorem cube_volume_given_face_area (s : ℝ) (h : s^2 = 36) : s^3 = 216 := by
  sorry

end cube_volume_given_face_area_l97_97184


namespace henry_initial_money_l97_97692

variable (x : ℤ)

theorem henry_initial_money : (x + 18 - 10 = 19) → x = 11 :=
by
  intro h
  sorry

end henry_initial_money_l97_97692


namespace shawn_divided_into_groups_l97_97337

theorem shawn_divided_into_groups :
  ∀ (total_pebbles red_pebbles blue_pebbles remaining_pebbles yellow_pebbles groups : ℕ),
  total_pebbles = 40 →
  red_pebbles = 9 →
  blue_pebbles = 13 →
  remaining_pebbles = total_pebbles - red_pebbles - blue_pebbles →
  remaining_pebbles % 3 = 0 →
  yellow_pebbles = blue_pebbles - 7 →
  remaining_pebbles = groups * yellow_pebbles →
  groups = 3 :=
by
  intros total_pebbles red_pebbles blue_pebbles remaining_pebbles yellow_pebbles groups
  intros h_total h_red h_blue h_remaining h_divisible h_yellow h_group
  sorry

end shawn_divided_into_groups_l97_97337


namespace newspapers_sold_correct_l97_97842

def total_sales : ℝ := 425.0
def magazines_sold : ℝ := 150
def newspapers_sold : ℝ := total_sales - magazines_sold

theorem newspapers_sold_correct : newspapers_sold = 275.0 := by
  sorry

end newspapers_sold_correct_l97_97842


namespace composite_for_infinitely_many_n_l97_97852

theorem composite_for_infinitely_many_n :
  ∃ᶠ n in at_top, (n > 0) ∧ (n % 6 = 4) → ∃ p, p ≠ 1 ∧ p ≠ n^n + (n+1)^(n+1) :=
sorry

end composite_for_infinitely_many_n_l97_97852


namespace pie_eating_contest_l97_97308

theorem pie_eating_contest :
  let first_student_round1 := (5 : ℚ) / 6
  let first_student_round2 := (1 : ℚ) / 6
  let second_student_total := (2 : ℚ) / 3
  let first_student_total := first_student_round1 + first_student_round2
  first_student_total - second_student_total = 1 / 3 :=
by
  sorry

end pie_eating_contest_l97_97308


namespace sequence_general_term_l97_97157

theorem sequence_general_term (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = (1 / 2) * a n + 1) :
  ∀ n, a n = 2 - (1 / 2) ^ (n - 1) :=
by
  sorry

end sequence_general_term_l97_97157


namespace water_level_not_discrete_l97_97941

-- Define the available conditions to identify the type of random variables
def visitors_lounge_daily : RandomVariable ℕ :=
  sorry -- assuming this is a countable variable

def pages_received_daily : RandomVariable ℕ :=
  sorry -- assuming this is a countable variable

def water_level_yangtze : RandomVariable ℝ :=
  sorry -- assuming this is an uncountable variable

def vehicles_overpass_daily : RandomVariable ℕ :=
  sorry -- assuming this is a countable variable

-- Prove that the water level of the Yangtze River is not a discrete random variable
theorem water_level_not_discrete : ¬is_discrete_random_variable water_level_yangtze :=
  sorry

end water_level_not_discrete_l97_97941


namespace floor_neg_seven_fourths_l97_97275

theorem floor_neg_seven_fourths : (Int.floor (-7 / 4 : ℚ) = -2) := 
by
  sorry

end floor_neg_seven_fourths_l97_97275


namespace percent_republicans_voting_for_A_l97_97986

theorem percent_republicans_voting_for_A (V : ℝ) (percent_Democrats : ℝ) 
  (percent_Republicans : ℝ) (percent_D_voting_for_A : ℝ) 
  (percent_total_voting_for_A : ℝ) (R : ℝ) 
  (h1 : percent_Democrats = 0.60)
  (h2 : percent_Republicans = 0.40)
  (h3 : percent_D_voting_for_A = 0.85)
  (h4 : percent_total_voting_for_A = 0.59) :
  R = 0.2 :=
by 
  sorry

end percent_republicans_voting_for_A_l97_97986


namespace find_a_b_l97_97291

-- Conditions defining the solution sets A and B
def A : Set ℝ := { x | -1 < x ∧ x < 3 }
def B : Set ℝ := { x | -3 < x ∧ x < 2 }

-- The solution set of the inequality x^2 + ax + b < 0 is the intersection A∩B
def C : Set ℝ := A ∩ B

-- Proving that there exist values of a and b such that the solution set C corresponds to the inequality x^2 + ax + b < 0
theorem find_a_b : ∃ a b : ℝ, (∀ x : ℝ, C x ↔ x^2 + a*x + b < 0) ∧ a + b = -3 := 
by 
  sorry

end find_a_b_l97_97291


namespace beef_original_weight_l97_97629

noncomputable def originalWeightBeforeProcessing (weightAfterProcessing : ℝ) (lossPercentage : ℝ) : ℝ :=
  weightAfterProcessing / (1 - lossPercentage / 100)

theorem beef_original_weight : originalWeightBeforeProcessing 570 35 = 876.92 :=
by
  sorry

end beef_original_weight_l97_97629


namespace frog_jump_plan_l97_97320

-- Define the vertices of the hexagon
inductive Vertex
| A | B | C | D | E | F

open Vertex

-- Define adjacency in the regular hexagon
def adjacent (v1 v2 : Vertex) : Prop :=
  match v1, v2 with
  | A, B | A, F | B, C | B, A | C, D | C, B | D, E | D, C | E, F | E, D | F, A | F, E => true
  | _, _ => false

-- Define the problem
def frog_jump_sequences_count : ℕ :=
  26

theorem frog_jump_plan : frog_jump_sequences_count = 26 := 
  sorry

end frog_jump_plan_l97_97320


namespace longest_segment_cylinder_l97_97511

theorem longest_segment_cylinder (r h : ℤ) (c : ℝ) (hr : r = 4) (hh : h = 9) : 
  c = Real.sqrt (2 * r * r + h * h) ↔ c = Real.sqrt 145 :=
by
  sorry

end longest_segment_cylinder_l97_97511


namespace total_time_spent_l97_97164

def one_round_time : ℕ := 30
def saturday_initial_rounds : ℕ := 1
def saturday_additional_rounds : ℕ := 10
def sunday_rounds : ℕ := 15

theorem total_time_spent :
  one_round_time * (saturday_initial_rounds + saturday_additional_rounds + sunday_rounds) = 780 := by
  sorry

end total_time_spent_l97_97164


namespace group_count_4_men_5_women_l97_97028

theorem group_count_4_men_5_women : 
  let men := 4
  let women := 5
  let groups := List.replicate 3 (3, true)
  ∃ (m_w_combinations : List (ℕ × ℕ)),
    m_w_combinations = [(1, 2), (2, 1)] ∧
    ((men.choose m_w_combinations.head.fst * women.choose m_w_combinations.head.snd) * (men - m_w_combinations.head.fst).choose m_w_combinations.tail.head.fst * (women - m_w_combinations.head.snd).choose m_w_combinations.tail.head.snd) = 360 :=
by
  sorry

end group_count_4_men_5_women_l97_97028


namespace largest_fraction_l97_97626

theorem largest_fraction (A B C D E : ℚ)
    (hA: A = 5 / 11)
    (hB: B = 7 / 16)
    (hC: C = 23 / 50)
    (hD: D = 99 / 200)
    (hE: E = 202 / 403) : 
    E > A ∧ E > B ∧ E > C ∧ E > D :=
by
  sorry

end largest_fraction_l97_97626


namespace david_completion_time_l97_97647

theorem david_completion_time :
  (∃ D : ℕ, ∀ t : ℕ, 6 * (1 / D) + 3 * ((1 / D) + (1 / t)) = 1 -> D = 12) :=
sorry

end david_completion_time_l97_97647


namespace find_a_l97_97963

theorem find_a (k x y a : ℝ) (hkx : k ≤ x) (hx3 : x ≤ 3) (hy7 : a ≤ y) (hy7' : y ≤ 7) (hy : y = k * x + 1) :
  a = 5 ∨ a = 1 - 3 * Real.sqrt 6 :=
sorry

end find_a_l97_97963


namespace smallest_positive_period_interval_of_monotonic_increase_value_of_a_l97_97458

-- Problem 1: Smallest positive period of f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * real.sin (2 * x  +  (real.pi / 6)) + 1

theorem smallest_positive_period : 
  ∃ T > 0, ∀ x, f (x + T) = f x :=
sorry

-- Problem 2: Interval of monotonic increase of f(x)
theorem interval_of_monotonic_increase :
  ∀ k : ℤ, 
  ∃ a b : ℝ, ∀ x ∈ set.Icc a b, 
  f' x > 0 ∧ a = - real.pi / 3 + k * real.pi ∧ b = real.pi / 6 + k * real.pi :=
sorry

-- Problem 3: Value of a in triangle ABC given conditions
variables (b c A : ℝ)
def triangle_area (b c A : ℝ) : ℝ := 1 / 2 * b * c * real.sin A

-- Given conditions
axiom f_A : f A = 2
axiom b_eq : b = 1
axiom area_eq : triangle_area 1 c A = real.sqrt 3

-- Prove that a^2 = 13
theorem value_of_a :
  ∃ a : ℝ, a ^ 2 = 13 :=
sorry

end smallest_positive_period_interval_of_monotonic_increase_value_of_a_l97_97458


namespace sqrt_five_squared_minus_four_squared_eq_three_l97_97349

theorem sqrt_five_squared_minus_four_squared_eq_three : Real.sqrt (5 ^ 2 - 4 ^ 2) = 3 := by
  sorry

end sqrt_five_squared_minus_four_squared_eq_three_l97_97349


namespace inequality_problem_l97_97146

theorem inequality_problem (a : ℝ) (h : a = Real.cos (2 * Real.pi / 7)) : 
  2^(a - 1/2) < 2 * a :=
by
  sorry

end inequality_problem_l97_97146


namespace eval_floor_neg_seven_fourths_l97_97259

theorem eval_floor_neg_seven_fourths : 
  ∃ (x : ℚ), x = -7 / 4 ∧ ∀ y, y ≤ x ∧ y ∈ ℤ → y ≤ -2 :=
by
  obtain ⟨x, hx⟩ : ∃ (x : ℚ), x = -7 / 4 := ⟨-7 / 4, rfl⟩,
  use x,
  split,
  { exact hx },
  { intros y hy,
    sorry }

end eval_floor_neg_seven_fourths_l97_97259


namespace trip_movie_savings_l97_97620

def evening_ticket_cost : ℕ := 10
def combo_cost : ℕ := 10
def ticket_discount_percentage : ℕ := 20
def combo_discount_percentage : ℕ := 50

theorem trip_movie_savings :
  let ticket_saving := evening_ticket_cost * ticket_discount_percentage / 100,
      combo_saving := combo_cost * combo_discount_percentage / 100
  in ticket_saving + combo_saving = 7 :=
by
  sorry

end trip_movie_savings_l97_97620


namespace problem1_problem2_l97_97790

/-- Problem 1 -/
theorem problem1 (a b : ℝ) : (a^2 - b)^2 = a^4 - 2 * a^2 * b + b^2 :=
by
  sorry

/-- Problem 2 -/
theorem problem2 (x : ℝ) : (2 * x + 1) * (4 * x^2 - 1) * (2 * x - 1) = 16 * x^4 - 8 * x^2 + 1 :=
by
  sorry

end problem1_problem2_l97_97790


namespace compute_expression_l97_97793

theorem compute_expression : 11 * (1 / 17) * 34 - 3 = 19 :=
by
  sorry

end compute_expression_l97_97793


namespace josephs_total_cards_l97_97036

def number_of_decks : ℕ := 4
def cards_per_deck : ℕ := 52
def total_cards : ℕ := number_of_decks * cards_per_deck

theorem josephs_total_cards : total_cards = 208 := by
  sorry

end josephs_total_cards_l97_97036


namespace find_f_zero_forall_x_f_pos_solve_inequality_l97_97041

variable {f : ℝ → ℝ}

-- Conditions
axiom condition_1 : ∀ x, x > 0 → f x > 1
axiom condition_2 : ∀ x y, f (x + y) = f x * f y
axiom condition_3 : f 2 = 3

-- Questions rewritten as Lean theorems

theorem find_f_zero : f 0 = 1 := sorry

theorem forall_x_f_pos : ∀ x, f x > 0 := sorry

theorem solve_inequality : ∀ x, f (7 + 2 * x) > 9 ↔ x > -3 / 2 := sorry

end find_f_zero_forall_x_f_pos_solve_inequality_l97_97041


namespace inequality_pgcd_l97_97321

theorem inequality_pgcd (a b : ℕ) (h1 : a > b) (h2 : (a - b) ∣ (a^2 + b)) : 
  (a + 1) / (b + 1) ≤ Nat.gcd a b + 1 := 
sorry

end inequality_pgcd_l97_97321


namespace exists_same_color_parallelepiped_l97_97614

-- Definitions of the conditions
def A : Set (ℤ × ℤ × ℤ) := {v | True}

variable (color : ℤ × ℤ × ℤ → ℕ)
variable (p : ℕ)
variable (color_range : ∀ v ∈ A, color v < p)

-- The main theorem statement
theorem exists_same_color_parallelepiped : 
  ∃ (v1 v2 v3 v4 v5 v6 v7 v8 : ℤ × ℤ × ℤ),
  v1 ∈ A ∧ v2 ∈ A ∧ v3 ∈ A ∧ v4 ∈ A ∧
  v5 ∈ A ∧ v6 ∈ A ∧ v7 ∈ A ∧ v8 ∈ A ∧
  -- All vertices have the same color
  color v1 = color v2 ∧ color v2 = color v3 ∧ color v3 = color v4 ∧
  color v4 = color v5 ∧ color v5 = color v6 ∧ color v6 = color v7 ∧
  color v7 = color v8 ∧
  -- Vertices form a rectangular parallelepiped
  (∃ (x1 x2 y1 y2 z1 z2 : ℤ), 
    v1 = (x1, y1, z1) ∧ v2 = (x2 ,y1 ,z1) ∧ v3 = (x1, y2, z1) ∧
    v4 = (x2, y2, z1) ∧ v5 = (x1, y1, z2) ∧ v6 = (x2, y1, z2) ∧
    v7 = (x1, y2, z2) ∧ v8 = (x2, y2, z2)) := 
sorry

end exists_same_color_parallelepiped_l97_97614


namespace point_not_on_graph_l97_97215

def on_graph (x y : ℚ) : Prop := y = x / (x + 2)

/-- Let's state the main theorem -/
theorem point_not_on_graph : ¬ on_graph 2 (2 / 3) := by
  sorry

end point_not_on_graph_l97_97215


namespace quadratic_inequality_solution_set_l97_97411

theorem quadratic_inequality_solution_set :
  {x : ℝ | x * (x - 2) > 0} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 2} :=
by
  sorry

end quadratic_inequality_solution_set_l97_97411


namespace geom_seq_proof_l97_97817

noncomputable def geom_seq (a q : ℝ) (n : ℕ) : ℝ :=
  a * q^(n - 1)

variables {a q : ℝ}

theorem geom_seq_proof (h1 : geom_seq a q 7 = 4) (h2 : geom_seq a q 5 + geom_seq a q 9 = 10) :
  geom_seq a q 3 + geom_seq a q 11 = 17 :=
by
  sorry

end geom_seq_proof_l97_97817


namespace initial_selling_price_l97_97115

theorem initial_selling_price (P : ℝ) : 
    (∀ (c_i c_m p_m r : ℝ),
        c_i = 3 ∧
        c_m = 20 ∧
        p_m = 4 ∧
        r = 50 ∧
        (15 * P + 5 * p_m - 20 * c_i = r)
    ) → 
    P = 6 := by 
    sorry

end initial_selling_price_l97_97115


namespace cars_pass_same_order_l97_97731

theorem cars_pass_same_order (num_cars : ℕ) (num_points : ℕ)
    (cities_speeds speeds_outside_cities : Fin num_cars → ℝ) :
    num_cars = 10 → num_points = 2011 → 
    ∃ (p1 p2 : Fin num_points), p1 ≠ p2 ∧ (∀ i j : Fin num_cars, (i < j) → 
    (cities_speeds i) / (cities_speeds i + speeds_outside_cities i) = 
    (cities_speeds j) / (cities_speeds j + speeds_outside_cities j) → p1 = p2 ) :=
by
  sorry

end cars_pass_same_order_l97_97731


namespace graph_passes_through_fixed_point_l97_97175

-- Define the linear function given in the conditions
def linearFunction (k x y : ℝ) : ℝ :=
  (2 * k - 1) * x - (k + 3) * y - (k - 11)

-- Define the fixed point (2, 3)
def fixedPoint : ℝ × ℝ :=
  (2, 3)

-- State the theorem that the graph of the linear function always passes through the fixed point 
theorem graph_passes_through_fixed_point :
  ∀ k : ℝ, linearFunction k fixedPoint.1 fixedPoint.2 = 0 :=
by sorry  -- proof skipped

end graph_passes_through_fixed_point_l97_97175


namespace third_place_books_max_l97_97484

theorem third_place_books_max (x y z : ℕ) (hx : 100 ∣ x) (hxpos : 0 < x) (hy : 100 ∣ y) (hz : 100 ∣ z)
  (h_sum : 2 * x + 100 + x + 100 + x + y + z ≤ 10000)
  (h_first_eq : 2 * x + 100 = x + 100 + x)
  (h_second_eq : x + 100 = y + z) 
  : x ≤ 1900 := sorry

end third_place_books_max_l97_97484


namespace calculate_R_cubed_plus_R_squared_plus_R_l97_97639

theorem calculate_R_cubed_plus_R_squared_plus_R (R : ℕ) (hR : R > 0)
  (h1 : ∃ q : ℚ, q = (R / (2 * R + 2)) * ((R - 1) / (2 * R + 1)))
  (h2 : (R / (2 * R + 2)) * ((R + 2) / (2 * R + 1)) + ((R + 2) / (2 * R + 2)) * (R / (2 * R + 1)) = 3 * q) :
  R^3 + R^2 + R = 399 :=
by
  sorry

end calculate_R_cubed_plus_R_squared_plus_R_l97_97639


namespace solve_for_m_l97_97228

theorem solve_for_m (m : ℝ) (h : (4 * m + 6) * (2 * m - 5) = 159) : m = 5.3925 :=
sorry

end solve_for_m_l97_97228


namespace rope_length_eqn_l97_97588

theorem rope_length_eqn (x : ℝ) : 8^2 + (x - 3)^2 = x^2 := 
by 
  sorry

end rope_length_eqn_l97_97588


namespace expression_value_l97_97883

theorem expression_value (y : ℤ) (h : y = 5) : (y^2 - y - 12) / (y - 4) = 8 :=
by
  rw[h]
  sorry

end expression_value_l97_97883


namespace find_fixed_point_c_l97_97077

noncomputable def f (x : ℝ) : ℝ := x ^ 2 - 2
noncomputable def g (x : ℝ) (c : ℝ) : ℝ := 2 * x ^ 2 - c

theorem find_fixed_point_c (c : ℝ) : 
  (∃ a : ℝ, f a = a ∧ g a c = a) ↔ (c = 3 ∨ c = 6) := sorry

end find_fixed_point_c_l97_97077


namespace cost_price_is_correct_l97_97230

-- Define the conditions
def purchasing_clocks : ℕ := 150
def gain_60_clocks : ℝ := 0.12
def gain_90_clocks : ℝ := 0.18
def uniform_profit : ℝ := 0.16
def difference_in_profit : ℝ := 75

-- Define the cost price of each clock
noncomputable def C : ℝ := 125

-- Define and state the theorem
theorem cost_price_is_correct (C : ℝ) :
  (60 * C * (1 + gain_60_clocks) + 90 * C * (1 + gain_90_clocks)) - (150 * C * (1 + uniform_profit)) = difference_in_profit :=
sorry

end cost_price_is_correct_l97_97230


namespace tv_sales_value_increase_l97_97913

theorem tv_sales_value_increase (P V : ℝ) :
    let P1 := 0.82 * P
    let V1 := 1.72 * V
    let P2 := 0.75 * P1
    let V2 := 1.90 * V1
    let initial_sales := P * V
    let final_sales := P2 * V2
    final_sales = 2.00967 * initial_sales :=
by
  sorry

end tv_sales_value_increase_l97_97913


namespace smallest_even_piece_to_stop_triangle_l97_97387

-- Define a predicate to check if an integer is even
def even (x : ℕ) : Prop := x % 2 = 0

-- Define the conditions for triangle inequality to hold
def triangle_inequality_violated (a b c : ℕ) : Prop :=
  a + b ≤ c ∨ a + c ≤ b ∨ b + c ≤ a

-- Define the main theorem
theorem smallest_even_piece_to_stop_triangle
  (x : ℕ) (hx : even x) (len1 len2 len3 : ℕ)
  (h_len1 : len1 = 7) (h_len2 : len2 = 24) (h_len3 : len3 = 25) :
  6 ≤ x → triangle_inequality_violated (len1 - x) (len2 - x) (len3 - x) :=
by
  sorry

end smallest_even_piece_to_stop_triangle_l97_97387


namespace smallest_N_divisors_of_8_l97_97000

theorem smallest_N_divisors_of_8 (N : ℕ) (h0 : N % 10 = 0) (h8 : ∃ (divisors : ℕ), divisors = 8 ∧ (∀ k, k ∣ N → k ≤ divisors)) : N = 30 := 
sorry

end smallest_N_divisors_of_8_l97_97000


namespace total_cases_sold_is_correct_l97_97489

-- Define the customer groups and their respective number of cases bought
def n1 : ℕ := 8
def k1 : ℕ := 3
def n2 : ℕ := 4
def k2 : ℕ := 2
def n3 : ℕ := 8
def k3 : ℕ := 1

-- Define the total number of cases sold
def total_cases_sold : ℕ := n1 * k1 + n2 * k2 + n3 * k3

-- The proof statement that the total cases sold is 40
theorem total_cases_sold_is_correct : total_cases_sold = 40 := by
  -- Proof content will be provided here.
  sorry

end total_cases_sold_is_correct_l97_97489


namespace gain_percentage_is_66_67_l97_97860

variable (C S : ℝ)
variable (cost_price_eq : 20 * C = 12 * S)

theorem gain_percentage_is_66_67 (h : 20 * C = 12 * S) : (((5 / 3) * C - C) / C) * 100 = 66.67 := by
  sorry

end gain_percentage_is_66_67_l97_97860


namespace no_two_perfect_cubes_between_two_perfect_squares_l97_97837

theorem no_two_perfect_cubes_between_two_perfect_squares :
  ∀ n a b : ℤ, n^2 < a^3 ∧ a^3 < b^3 ∧ b^3 < (n + 1)^2 → False :=
by 
  sorry

end no_two_perfect_cubes_between_two_perfect_squares_l97_97837


namespace largest_of_five_consecutive_sum_l97_97358

theorem largest_of_five_consecutive_sum (n : ℕ) 
  (h : n + (n+1) + (n+2) + (n+3) + (n+4) = 90) : 
  n + 4 = 20 :=
sorry

end largest_of_five_consecutive_sum_l97_97358


namespace total_colors_over_two_hours_l97_97942

def colors_in_first_hour : Nat :=
  let quick_colors := 5 * 3
  let slow_colors := 2 * 3
  quick_colors + slow_colors

def colors_in_second_hour : Nat :=
  let quick_colors := (5 * 2) * 3
  let slow_colors := (2 * 2) * 3
  quick_colors + slow_colors

theorem total_colors_over_two_hours : colors_in_first_hour + colors_in_second_hour = 63 := by
  sorry

end total_colors_over_two_hours_l97_97942


namespace not_right_triangle_l97_97648

/-- In a triangle ABC, with angles A, B, C, the condition A = B = 2 * C does not form a right-angled triangle. -/
theorem not_right_triangle (A B C : ℝ) (h1 : A = B) (h2 : A = 2 * C) (h3 : A + B + C = 180) : 
    ¬(A = 90 ∨ B = 90 ∨ C = 90) := 
by
  sorry

end not_right_triangle_l97_97648


namespace line_through_point_with_equal_intercepts_l97_97186

theorem line_through_point_with_equal_intercepts 
  (x y k : ℝ) 
  (h1 : (3 : ℝ) + (-6 : ℝ) + k = 0 ∨ 2 * (3 : ℝ) + (-6 : ℝ) = 0) 
  (h2 : k = 0 ∨ x + y + k = 0) : 
  (x = 1 ∨ x = 2) ∧ (k = -3 ∨ k = 0) :=
sorry

end line_through_point_with_equal_intercepts_l97_97186


namespace power_sum_prime_eq_l97_97845

theorem power_sum_prime_eq (p a n : ℕ) (hp : p.Prime) (h_eq : 2^p + 3^p = a^n) : n = 1 :=
by sorry

end power_sum_prime_eq_l97_97845


namespace multiple_of_sum_squares_l97_97984

theorem multiple_of_sum_squares (a b c : ℕ) (h1 : a < 2017) (h2 : b < 2017) (h3 : c < 2017) (h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a)
    (h7 : ∃ k1, a^3 - b^3 = k1 * 2017) (h8 : ∃ k2, b^3 - c^3 = k2 * 2017) (h9 : ∃ k3, c^3 - a^3 = k3 * 2017) :
    ∃ k, a^2 + b^2 + c^2 = k * (a + b + c) :=
by
  sorry

end multiple_of_sum_squares_l97_97984


namespace rhombus_area_l97_97419

-- Define the lengths of the diagonals
def d1 : ℝ := 6
def d2 : ℝ := 8

-- Problem statement: The area of the rhombus
theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 6) (h2 : d2 = 8) : (1 / 2) * d1 * d2 = 24 := by
  -- The proof is not required, so we use sorry.
  sorry

end rhombus_area_l97_97419


namespace cds_per_rack_l97_97399

theorem cds_per_rack (total_cds : ℕ) (racks_per_shelf : ℕ) (cds_per_rack : ℕ) 
  (h1 : total_cds = 32) 
  (h2 : racks_per_shelf = 4) : 
  cds_per_rack = total_cds / racks_per_shelf :=
by 
  sorry

end cds_per_rack_l97_97399


namespace incorrect_expression_l97_97694

theorem incorrect_expression : 
  ∀ (x y : ℚ), (x / y = 2 / 5) → (x + 3 * y) / x ≠ 17 / 2 :=
by
  intros x y h
  sorry

end incorrect_expression_l97_97694


namespace smallest_b_for_fourth_power_l97_97545

noncomputable def is_fourth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k ^ 4 = n

theorem smallest_b_for_fourth_power :
  ∃ b : ℕ, (0 < b) ∧ (7 + 7 * b + 7 * b ^ 2 = (7 * 1 + 7 * 18 + 7 * 18 ^ 2)) 
  ∧ is_fourth_power (7 + 7 * b + 7 * b ^ 2) := sorry

end smallest_b_for_fourth_power_l97_97545


namespace meetings_percentage_l97_97847

def workday_hours := 10
def first_meeting_minutes := 60
def second_meeting_minutes := 3 * first_meeting_minutes
def total_workday_minutes := workday_hours * 60
def total_meeting_minutes := first_meeting_minutes + second_meeting_minutes

theorem meetings_percentage :
    (total_meeting_minutes / total_workday_minutes) * 100 = 40 :=
by
  sorry

end meetings_percentage_l97_97847


namespace find_kgs_of_apples_l97_97540

def cost_of_apples_per_kg : ℝ := 2
def num_packs_of_sugar : ℝ := 3
def cost_of_sugar_per_pack : ℝ := cost_of_apples_per_kg - 1
def weight_walnuts_kg : ℝ := 0.5
def cost_of_walnuts_per_kg : ℝ := 6
def cost_of_walnuts : ℝ := cost_of_walnuts_per_kg * weight_walnuts_kg
def total_cost : ℝ := 16

theorem find_kgs_of_apples (A : ℝ) :
  2 * A + (num_packs_of_sugar * cost_of_sugar_per_pack) + cost_of_walnuts = total_cost →
  A = 5 :=
by
  sorry

end find_kgs_of_apples_l97_97540


namespace initial_investment_amount_l97_97519

noncomputable def compoundInterest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem initial_investment_amount (P A r t : ℝ) (n : ℕ) (hA : A = 992.25) 
  (hr : r = 0.10) (hn : n = 2) (ht : t = 1) : P = 900 :=
by
  have h : compoundInterest P r n t = A := by sorry
  rw [hA, hr, hn, ht] at h
  simp at h
  exact sorry

end initial_investment_amount_l97_97519


namespace find_F_l97_97697

theorem find_F (C F : ℝ) (h1 : C = (4 / 7) * (F - 40)) (h2 : C = 35) : F = 101.25 :=
  sorry

end find_F_l97_97697


namespace molly_swam_28_meters_on_sunday_l97_97848

def meters_swam_on_saturday : ℕ := 45
def total_meters_swum : ℕ := 73
def meters_swam_on_sunday := total_meters_swum - meters_swam_on_saturday

theorem molly_swam_28_meters_on_sunday : meters_swam_on_sunday = 28 :=
by
  -- sorry to skip the proof
  sorry

end molly_swam_28_meters_on_sunday_l97_97848


namespace part1_solution_part2_solution_l97_97573

noncomputable def f (x : ℝ) : ℝ := |2 * x + 3| + |x - 1|

theorem part1_solution :
  {x : ℝ | f x > 4} = {x : ℝ | x < -2} ∪ {x : ℝ | 0 < x} :=
by
  sorry

theorem part2_solution (x0 : ℝ) :
  (∃ x0 : ℝ, ∀ t : ℝ, f x0 < |(x0 + t)| + |(t - x0)|) →
  ∀ m : ℝ, (f x0 < |m + t| + |t - m|) ↔ m ≠ 0 ∧ (|m| > 5 / 4) :=
by
  sorry

end part1_solution_part2_solution_l97_97573


namespace jack_pays_back_expected_amount_l97_97708

-- Definitions from the conditions
def principal : ℝ := 1200
def interest_rate : ℝ := 0.10

-- Definition for proof
def interest : ℝ := principal * interest_rate
def total_amount : ℝ := principal + interest

-- Lean statement for the proof problem
theorem jack_pays_back_expected_amount : total_amount = 1320 := by
  sorry

end jack_pays_back_expected_amount_l97_97708


namespace gcd_5039_3427_l97_97369

def a : ℕ := 5039
def b : ℕ := 3427

theorem gcd_5039_3427 : Nat.gcd a b = 7 := by
  sorry

end gcd_5039_3427_l97_97369


namespace find_g_3_8_l97_97737

variable (g : ℝ → ℝ)
variable (x : ℝ)

-- Conditions
axiom g0 : g 0 = 0
axiom monotonicity (x y : ℝ) : 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y
axiom symmetry (x : ℝ) : 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x
axiom scaling (x : ℝ) : 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3

-- Statement to prove
theorem find_g_3_8 : g (3 / 8) = 2 / 9 := 
sorry

end find_g_3_8_l97_97737


namespace bernoulli_convergence_l97_97456

noncomputable theory

variables {Ω : Type*} [MeasureTheory.ProbabilitySpace Ω]

-- Definition of Bernoulli random variables
def bernoulli (p : ℝ) [hp : 0 < p ∧ p < 1] (n : ℕ) : Ω →₀ ℝ := {
  to_fun := λ ω, if random_var ω < p then 1 else 0,
  measurable' := measurable_of_measurable_coe (measurable_set_lt measurable_random_var (measurable_const p).measurable)
}

-- Condition: Sequence of independent Bernoulli random variables 
def bernoulli_sequence (p : ℝ) [hp : 0 < p ∧ p < 1] : ℕ → Ω →₀ ℝ :=
λ n, bernoulli p n

-- Condition: Sum of series
def sum_series (p : ℝ) [hp : 0 < p ∧ p < 1] := 
  ∑' (n : ℕ), (λ ω, (bernoulli_sequence p n ω) / 2^(n+1))

-- Theorem statement
theorem bernoulli_convergence (p : ℝ) [hp : 0 < p ∧ p < 1] :
  ∃ U, 
    (if p = 1/2 
      then measure_theory.probability_measure U = measure_theory.uniform (0:ℝ, 1:ℝ)
      else is_singular U) :=
begin
  sorry,
end

end bernoulli_convergence_l97_97456


namespace maximum_value_of_x_minus_y_is_sqrt8_3_l97_97043

variable {x y z : ℝ}

noncomputable def maximum_value_of_x_minus_y (x y z : ℝ) : ℝ :=
  x - y

theorem maximum_value_of_x_minus_y_is_sqrt8_3 (h1 : x + y + z = 2) (h2 : x * y + y * z + z * x = 1) : 
  maximum_value_of_x_minus_y x y z = Real.sqrt (8 / 3) :=
sorry

end maximum_value_of_x_minus_y_is_sqrt8_3_l97_97043


namespace cos_2015_eq_neg_m_l97_97566

variable (m : ℝ)

-- Given condition
axiom sin_55_eq_m : Real.sin (55 * Real.pi / 180) = m

-- The proof problem
theorem cos_2015_eq_neg_m : Real.cos (2015 * Real.pi / 180) = -m :=
by
  sorry

end cos_2015_eq_neg_m_l97_97566


namespace audrey_lost_pieces_l97_97240

theorem audrey_lost_pieces {total_pieces_on_board : ℕ} {thomas_lost : ℕ} {initial_pieces_each : ℕ} (h1 : total_pieces_on_board = 21) (h2 : thomas_lost = 5) (h3 : initial_pieces_each = 16) :
  (initial_pieces_each - (total_pieces_on_board - (initial_pieces_each - thomas_lost))) = 6 :=
by
  sorry

end audrey_lost_pieces_l97_97240


namespace real_z_iff_imaginary_z_iff_first_quadrant_z_iff_l97_97559

-- Define z as a complex number with components dependent on m
def z (m : ℝ) : ℂ := ⟨m^2 - m, m - 1⟩

-- Statement 1: z is real iff m = 1
theorem real_z_iff (m : ℝ) : (∃ r : ℝ, z m = ⟨r, 0⟩) ↔ m = 1 := 
    sorry

-- Statement 2: z is purely imaginary iff m = 0
theorem imaginary_z_iff (m : ℝ) : (∃ i : ℝ, z m = ⟨0, i⟩ ∧ i ≠ 0) ↔ m = 0 := 
    sorry

-- Statement 3: z is in the first quadrant iff m > 1
theorem first_quadrant_z_iff (m : ℝ) : (z m).re > 0 ∧ (z m).im > 0 ↔ m > 1 := 
    sorry

end real_z_iff_imaginary_z_iff_first_quadrant_z_iff_l97_97559


namespace floor_neg_seven_quarter_l97_97269

theorem floor_neg_seven_quarter : 
  ∃ x : ℤ, -2 ≤ (-7 / 4 : ℚ) ∧ (-7 / 4 : ℚ) < -1 ∧ x = -2 := by
  have h1 : (-7 / 4 : ℚ) = -1.75 := by norm_num
  have h2 : -2 ≤ -1.75 := by norm_num
  have h3 : -1.75 < -1 := by norm_num
  use -2
  exact ⟨h2, h3, rfl⟩
  sorry

end floor_neg_seven_quarter_l97_97269


namespace parcel_post_cost_l97_97478

def indicator (P : ℕ) : ℕ := if P >= 5 then 1 else 0

theorem parcel_post_cost (P : ℕ) : 
  P ≥ 0 →
  (C : ℕ) = 15 + 5 * (P - 1) - 8 * indicator P :=
sorry

end parcel_post_cost_l97_97478


namespace yard_length_calculation_l97_97306

theorem yard_length_calculation (n_trees : ℕ) (distance : ℕ) (h1 : n_trees = 26) (h2 : distance = 32) : (n_trees - 1) * distance = 800 :=
by
  -- This is where the proof would go.
  sorry

end yard_length_calculation_l97_97306


namespace original_houses_count_l97_97875

namespace LincolnCounty

-- Define the constants based on the conditions
def houses_built_during_boom : ℕ := 97741
def houses_now : ℕ := 118558

-- Statement of the theorem
theorem original_houses_count : houses_now - houses_built_during_boom = 20817 := 
by sorry

end LincolnCounty

end original_houses_count_l97_97875


namespace opposite_sides_line_l97_97966

theorem opposite_sides_line (m : ℝ) :
  ( (3 * 3 - 2 * 1 + m) * (3 * (-4) - 2 * 6 + m) < 0 ) → (-7 < m ∧ m < 24) :=
by sorry

end opposite_sides_line_l97_97966


namespace highest_wave_height_l97_97103

-- Definitions of surfboard length and shortest wave conditions
def surfboard_length : ℕ := 7
def shortest_wave_height (H : ℕ) : ℕ := H + 4

-- Lean statement to be proved
theorem highest_wave_height (H : ℕ) (condition1 : H + 4 = surfboard_length + 3) : 
  4 * H + 2 = 26 :=
sorry

end highest_wave_height_l97_97103


namespace samantha_lost_pieces_l97_97520

theorem samantha_lost_pieces (total_pieces_on_board : ℕ) (arianna_lost : ℕ) (initial_pieces_per_player : ℕ) :
  total_pieces_on_board = 20 →
  arianna_lost = 3 →
  initial_pieces_per_player = 16 →
  (initial_pieces_per_player - (total_pieces_on_board - (initial_pieces_per_player - arianna_lost))) = 9 :=
by
  intros h1 h2 h3
  sorry

end samantha_lost_pieces_l97_97520


namespace scientific_notation_50000000000_l97_97352

theorem scientific_notation_50000000000 :
  ∃ (a : ℝ) (n : ℤ), 50000000000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ (a = 5.0 ∨ a = 5) ∧ n = 10 :=
by
  sorry

end scientific_notation_50000000000_l97_97352


namespace total_beetles_eaten_each_day_l97_97251

-- Definitions from the conditions
def birds_eat_per_day : ℕ := 12
def snakes_eat_per_day : ℕ := 3
def jaguars_eat_per_day : ℕ := 5
def number_of_jaguars : ℕ := 6

-- Theorem statement
theorem total_beetles_eaten_each_day :
  (number_of_jaguars * jaguars_eat_per_day) * snakes_eat_per_day * birds_eat_per_day = 1080 :=
by sorry

end total_beetles_eaten_each_day_l97_97251


namespace max_inscribed_circle_area_of_triangle_l97_97001

theorem max_inscribed_circle_area_of_triangle
  (a b : ℝ)
  (ellipse : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)
  (f1 f2 : ℝ × ℝ)
  (F1_coords : f1 = (-1, 0))
  (F2_coords : f2 = (1, 0))
  (P Q : ℝ × ℝ)
  (line_through_F2 : ∀ y : ℝ, x = 1 → y^2 = 9 / 4)
  (P_coords : P = (1, 3/2))
  (Q_coords : Q = (1, -3/2))
  : (π * (3 / 4)^2 = 9 * π / 16) :=
  sorry

end max_inscribed_circle_area_of_triangle_l97_97001


namespace noah_ava_zoo_trip_l97_97467

theorem noah_ava_zoo_trip :
  let tickets_cost := 5
  let bus_fare := 1.5
  let initial_money := 40
  let num_people := 2
  let round_trip := 2

  initial_money - (num_people * tickets_cost + num_people * round_trip * bus_fare) = 24 :=
by
  let tickets_cost := 5
  let bus_fare := 1.5
  let initial_money := 40
  let num_people := 2
  let round_trip := 2

  have cost := num_people * tickets_cost + num_people * round_trip * bus_fare
  have remaining := initial_money - cost
  have : remaining = 24, sorry
  exact this

end noah_ava_zoo_trip_l97_97467


namespace total_customers_l97_97763

-- Define the initial number of customers
def initial_customers : ℕ := 14

-- Define the number of customers that left
def customers_left : ℕ := 3

-- Define the number of new customers gained
def new_customers : ℕ := 39

-- Prove that the total number of customers is 50
theorem total_customers : initial_customers - customers_left + new_customers = 50 := 
by
  sorry

end total_customers_l97_97763


namespace floor_neg_seven_over_four_l97_97254

theorem floor_neg_seven_over_four : floor (-7 / 4) = -2 :=
by
  sorry

end floor_neg_seven_over_four_l97_97254


namespace percentage_of_invalid_votes_l97_97310

-- Candidate A got 60% of the total valid votes.
-- The total number of votes is 560000.
-- The number of valid votes polled in favor of candidate A is 285600.
variable (total_votes valid_votes_A : ℝ)
variable (percent_A : ℝ := 0.60)
variable (valid_votes total_invalid_votes percent_invalid_votes : ℝ)

axiom h1 : total_votes = 560000
axiom h2 : valid_votes_A = 285600
axiom h3 : valid_votes_A = percent_A * valid_votes
axiom h4 : total_invalid_votes = total_votes - valid_votes
axiom h5 : percent_invalid_votes = (total_invalid_votes / total_votes) * 100

theorem percentage_of_invalid_votes : percent_invalid_votes = 15 := by
  sorry

end percentage_of_invalid_votes_l97_97310


namespace value_of_nested_fraction_l97_97070

def nested_fraction : ℚ :=
  2 - (1 / (2 - (1 / (2 - 1 / 2))))

theorem value_of_nested_fraction : nested_fraction = 3 / 4 :=
by
  sorry

end value_of_nested_fraction_l97_97070


namespace ice_cream_to_afford_games_l97_97725

theorem ice_cream_to_afford_games :
  let game_cost := 60
  let ice_cream_price := 5
  (game_cost * 2) / ice_cream_price = 24 :=
by
  let game_cost := 60
  let ice_cream_price := 5
  show (game_cost * 2) / ice_cream_price = 24
  sorry

end ice_cream_to_afford_games_l97_97725


namespace iced_coffee_cost_is_2_l97_97719

def weekly_latte_cost := 4 * 5
def annual_latte_cost := weekly_latte_cost * 52
def weekly_iced_coffee_cost (x : ℝ) := x * 3
def annual_iced_coffee_cost (x : ℝ) := weekly_iced_coffee_cost x * 52
def total_annual_coffee_cost (x : ℝ) := annual_latte_cost + annual_iced_coffee_cost x
def reduced_spending_goal (x : ℝ) := 0.75 * total_annual_coffee_cost x
def saved_amount := 338

theorem iced_coffee_cost_is_2 :
  ∃ x : ℝ, (total_annual_coffee_cost x - reduced_spending_goal x = saved_amount) → x = 2 :=
by
  sorry

end iced_coffee_cost_is_2_l97_97719


namespace division_of_expression_l97_97922

theorem division_of_expression (x y : ℝ) (hy : y ≠ 0) (hx : x ≠ 0) : (12 * x^2 * y) / (-6 * x * y) = -2 * x := by
  sorry

end division_of_expression_l97_97922


namespace inequality_abc_l97_97056

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c := 
by 
  sorry

end inequality_abc_l97_97056


namespace running_race_l97_97683

-- Define participants
inductive Participant : Type
| Anna
| Bella
| Csilla
| Dora

open Participant

-- Define positions
@[ext] structure Position :=
(first : Participant)
(last : Participant)

-- Conditions:
def conditions (p : Participant) (q : Participant) (r : Participant) (s : Participant)
  (pa : Position) : Prop :=
  (pa.first = r) ∧ -- Csilla was first
  (pa.first ≠ q) ∧ -- Bella was not first
  (pa.first ≠ p) ∧ (pa.last ≠ p) ∧ -- Anna was not first or last
  (pa.last = s) -- Dóra's statement about being last

-- Definition of the liar
def liar (p : Participant) : Prop :=
  p = Dora

-- Proof problem
theorem running_race : ∃ (pa : Position), liar Dora ∧ (pa.first = Csilla) :=
  sorry

end running_race_l97_97683


namespace height_of_highest_wave_l97_97105

theorem height_of_highest_wave 
  (h_austin : ℝ) -- Austin's height
  (h_high : ℝ) -- Highest wave's height
  (h_short : ℝ) -- Shortest wave's height 
  (height_relation1 : h_high = 4 * h_austin + 2)
  (height_relation2 : h_short = h_austin + 4)
  (surfboard : ℝ) (surfboard_len : surfboard = 7)
  (short_wave_len : h_short = surfboard + 3) :
  h_high = 26 :=
by
  -- Define local variables with the values from given conditions
  let austin_height := 6        -- as per calculation: 10 - 4 = 6
  let highest_wave_height := 26 -- as per calculation: (6 * 4) + 2 = 26
  sorry

end height_of_highest_wave_l97_97105


namespace distinct_solutions_sub_l97_97171

open Nat Real

theorem distinct_solutions_sub (p q : Real) (hpq_distinct : p ≠ q) (h_eqn_p : (p - 4) * (p + 4) = 17 * p - 68) (h_eqn_q : (q - 4) * (q + 4) = 17 * q - 68) (h_p_gt_q : p > q) : p - q = 9 := 
sorry

end distinct_solutions_sub_l97_97171


namespace preston_total_received_l97_97851

-- Conditions
def cost_per_sandwich := 5
def delivery_fee := 20
def number_of_sandwiches := 18
def tip_percentage := 0.10

-- Correct Answer
def total_amount_received := 121

-- Lean Statement
theorem preston_total_received : 
  (cost_per_sandwich * number_of_sandwiches + delivery_fee) * (1 + tip_percentage) = total_amount_received :=
by 
  sorry

end preston_total_received_l97_97851


namespace w_identity_l97_97418

theorem w_identity (w : ℝ) (h_pos : w > 0) (h_eq : w - 1 / w = 5) : (w + 1 / w) ^ 2 = 29 := by
  sorry

end w_identity_l97_97418


namespace number_of_students_in_both_ball_and_track_l97_97398

variable (total studentsSwim studentsTrack studentsBall bothSwimTrack bothSwimBall bothTrackBall : ℕ)
variable (noAllThree : Prop)

theorem number_of_students_in_both_ball_and_track
  (h_total : total = 26)
  (h_swim : studentsSwim = 15)
  (h_track : studentsTrack = 8)
  (h_ball : studentsBall = 14)
  (h_both_swim_track : bothSwimTrack = 3)
  (h_both_swim_ball : bothSwimBall = 3)
  (h_no_all_three : noAllThree) :
  bothTrackBall = 5 := by
  sorry

end number_of_students_in_both_ball_and_track_l97_97398


namespace arithmetic_to_geometric_progression_l97_97781

theorem arithmetic_to_geometric_progression (d : ℝ) (h : ∀ d, (4 + d) * (4 + d) = 7 * (22 + 2 * d)) :
  ∃ d, 7 + 2 * d = 3.752 :=
sorry

end arithmetic_to_geometric_progression_l97_97781


namespace addilynn_eggs_initial_l97_97389

theorem addilynn_eggs_initial (E : ℕ) (H1 : ∃ (E : ℕ), (E / 2) - 15 = 21) : E = 72 :=
by
  sorry

end addilynn_eggs_initial_l97_97389


namespace power_function_monotonic_l97_97483

theorem power_function_monotonic (m : ℝ) :
  2 * m^2 + m > 0 ∧ m > 0 → m = 1 / 2 := 
by
  intro h
  sorry

end power_function_monotonic_l97_97483


namespace sqrt_inequality_l97_97761

theorem sqrt_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a * b + b * c + c * a = 1) :
  sqrt (a + 1 / a) + sqrt (b + 1 / b) + sqrt (c + 1 / c) ≥ 2 * (sqrt a + sqrt b + sqrt c) :=
by
  sorry

end sqrt_inequality_l97_97761


namespace total_new_students_l97_97037

-- Given conditions
def number_of_schools : ℝ := 25.0
def average_students_per_school : ℝ := 9.88

-- Problem statement
theorem total_new_students : number_of_schools * average_students_per_school = 247 :=
by sorry

end total_new_students_l97_97037


namespace find_c_l97_97601

theorem find_c (a b c : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c)
(h_asc : a < b) (h_asc2 : b < c)
(h_sum : a + b + c = 11)
(h_eq : 1 / a + 1 / b + 1 / c = 1) : c = 6 := 
sorry

end find_c_l97_97601


namespace min_value_frac_l97_97685

theorem min_value_frac (x y : ℝ) (h1 : x > -1) (h2 : y > 0) (h3 : x + 2 * y = 2) : 
  ∃ L, (L = 3) ∧ (∀ (x y : ℝ), x > -1 → y > 0 → x + 2*y = 2 → 
  (∃ L, (L = 3) ∧ (∀ (x y : ℝ), x > -1 → y > 0 → x + 2*y = 2 → 
  ∀ (f : ℝ), f = (1 / (x + 1) + 2 / y) → f ≥ L))) :=
sorry

end min_value_frac_l97_97685


namespace f_is_odd_l97_97836

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.sqrt (1 + x^2))

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by
  sorry

end f_is_odd_l97_97836


namespace unique_three_digit_numbers_l97_97880

noncomputable def three_digit_numbers_no_repeats : Nat :=
  let total_digits := 10
  let permutations := total_digits * (total_digits - 1) * (total_digits - 2)
  let invalid_start_with_zero := (total_digits - 1) * (total_digits - 2)
  permutations - invalid_start_with_zero

theorem unique_three_digit_numbers : three_digit_numbers_no_repeats = 648 := by
  sorry

end unique_three_digit_numbers_l97_97880


namespace compute_pqr_l97_97825

theorem compute_pqr (p q r : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) 
  (h_sum : p + q + r = 26) (h_eq : (1 : ℚ) / ↑p + (1 : ℚ) / ↑q + (1 : ℚ) / ↑r + 360 / (p * q * r) = 1) : 
  p * q * r = 576 := 
sorry

end compute_pqr_l97_97825


namespace ratio_mark_to_jenna_l97_97788

-- Definitions based on the given conditions
def total_problems : ℕ := 20

def problems_angela : ℕ := 9
def problems_martha : ℕ := 2
def problems_jenna : ℕ := 4 * problems_martha - 2

def problems_completed : ℕ := problems_angela + problems_martha + problems_jenna
def problems_mark : ℕ := total_problems - problems_completed

-- The proof statement based on the question and conditions
theorem ratio_mark_to_jenna :
  (problems_mark : ℚ) / problems_jenna = 1 / 2 :=
by
  sorry

end ratio_mark_to_jenna_l97_97788


namespace arccos_half_eq_pi_div_three_l97_97927

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
by
  sorry

end arccos_half_eq_pi_div_three_l97_97927


namespace passed_boys_count_l97_97857

theorem passed_boys_count (total_boys avg_passed avg_failed overall_avg : ℕ) 
  (total_boys_eq : total_boys = 120) 
  (avg_passed_eq : avg_passed = 39) 
  (avg_failed_eq : avg_failed = 15) 
  (overall_avg_eq : overall_avg = 38) :
  let marks_by_passed := total_boys * overall_avg 
                         - (total_boys - passed) * avg_failed;
  let passed := marks_by_passed / avg_passed;
  passed = 115 := 
by
  sorry

end passed_boys_count_l97_97857


namespace probability_of_BEI3_is_zero_l97_97439

def isVowelOrDigit (s : Char) : Prop :=
  (s ∈ ['A', 'E', 'I', 'O', 'U']) ∨ (s.isDigit)

def isNonVowel (s : Char) : Prop :=
  s ∈ ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']

def isHexDigit (s : Char) : Prop :=
  s.isDigit ∨ s ∈ ['A', 'B', 'C', 'D', 'E', 'F']

noncomputable def numPossiblePlates : Nat :=
  13 * 21 * 20 * 16

theorem probability_of_BEI3_is_zero :
    ∃ (totalPlates : Nat), 
    (totalPlates = numPossiblePlates) ∧
    ¬(isVowelOrDigit 'B') →
    (1 : ℚ) / (totalPlates : ℚ) = 0 :=
by
  sorry

end probability_of_BEI3_is_zero_l97_97439


namespace probability_product_odd_l97_97856

theorem probability_product_odd :
  let range := setOf (λ x, 3 ≤ x ∧ x ≤ 20),
      n := set.card range,
      odd_elements := setOf (λ x, x ∈ range ∧ x % 2 = 1),
      n_odd := set.card odd_elements,
      total_combinations := nat.choose n 2,
      odd_combinations := nat.choose n_odd 2
  in total_combinations ≠ 0 → 
     (odd_combinations : ℚ) / total_combinations = 4 / 17 :=
by
  assume range n odd_elements n_odd total_combinations odd_combinations _,
  sorry

end probability_product_odd_l97_97856


namespace determine_B_l97_97813

-- Declare the sets A and B
def A : Set ℕ := {1, 2}
def B : Set ℕ := {0, 1}

-- The conditions given in the problem
axiom h1 : A ∩ B = {1}
axiom h2 : A ∪ B = {0, 1, 2}

-- The theorem we want to prove
theorem determine_B : B = {0, 1} :=
by
  sorry

end determine_B_l97_97813


namespace ways_to_place_people_into_groups_l97_97019

theorem ways_to_place_people_into_groups :
  let men := 4
  let women := 5
  ∃ (groups : Nat), groups = 2 ∧
  ∀ (g : Nat → (Fin 3 → (Bool → Nat → Nat))),
    (∀ i, i < group_counts → ∃ m w, g i m w < people ∧ g i m (1 - w) < people ∧ m + 1 - w + (1 - m) + w = 3) →
    let groups : List (List (Fin 2)) := [
      [(Fin.mk 1 dec_trivial, Fin.mk 2 dec_trivial)],
      [(Fin.mk 1 dec_trivial, Fin.mk 2 dec_trivial)]
    ] in
    g.mk 1 dec_trivial * g.mk 2 dec_trivial = 360 :=
sorry

end ways_to_place_people_into_groups_l97_97019


namespace projectiles_initial_distance_l97_97074

theorem projectiles_initial_distance 
  (v₁ v₂ : ℝ) (t : ℝ) (d₁ d₂ d : ℝ) 
  (hv₁ : v₁ = 445 / 60) -- speed of first projectile in km/min
  (hv₂ : v₂ = 545 / 60) -- speed of second projectile in km/min
  (ht : t = 84) -- time to meet in minutes
  (hd₁ : d₁ = v₁ * t) -- distance traveled by the first projectile
  (hd₂ : d₂ = v₂ * t) -- distance traveled by the second projectile
  (hd : d = d₁ + d₂) -- total initial distance
  : d = 1385.6 :=
by 
  sorry

end projectiles_initial_distance_l97_97074


namespace sum_of_possible_values_l97_97595

theorem sum_of_possible_values (x y : ℝ)
  (h : x * y - (2 * x) / (y ^ 3) - (2 * y) / (x ^ 3) = 5) :
  ∃ s : ℝ, s = (x - 2) * (y - 2) ∧ (s = -3 ∨ s = 9) :=
sorry

end sum_of_possible_values_l97_97595


namespace lcm_18_35_l97_97806

-- Given conditions: Prime factorizations of 18 and 35
def factorization_18 : Prop := (18 = 2^1 * 3^2)
def factorization_35 : Prop := (35 = 5^1 * 7^1)

-- The goal is to prove that the least common multiple of 18 and 35 is 630
theorem lcm_18_35 : factorization_18 ∧ factorization_35 → Nat.lcm 18 35 = 630 := by
  sorry -- Proof to be filled in

end lcm_18_35_l97_97806


namespace OneEmptyBox_NoBoxEmptyNoCompleteMatch_AtLeastTwoMatches_l97_97459

def combination (n k : ℕ) : ℕ := Nat.choose n k
def arrangement (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem OneEmptyBox (n : ℕ) (hn : n = 5) : (combination 5 2) * (arrangement 5 5) = 1200 := by
  sorry

theorem NoBoxEmptyNoCompleteMatch (n : ℕ) (hn : n = 5) : (arrangement 5 5) - 1 = 119 := by
  sorry

theorem AtLeastTwoMatches (n : ℕ) (hn : n = 5) : (arrangement 5 5) - (combination 5 1 * 9 + 44) = 31 := by
  sorry

end OneEmptyBox_NoBoxEmptyNoCompleteMatch_AtLeastTwoMatches_l97_97459


namespace infinite_sum_equals_two_l97_97110

theorem infinite_sum_equals_two :
  ∑' k : ℕ, (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 2 :=
sorry

end infinite_sum_equals_two_l97_97110


namespace fraction_difference_l97_97555

variable x y : ℝ
hypothesis hx : x = Real.sqrt 5 - 1
hypothesis hy : y = Real.sqrt 5 + 1

theorem fraction_difference : (1 / x - 1 / y = 1 / 2) :=
by 
  sorry

end fraction_difference_l97_97555


namespace HVAC_cost_per_vent_l97_97602

/-- 
The cost of Joe's new HVAC system is $20,000. It includes 2 conditioning zones, each with 5 vents.
Prove that the cost of the system per vent is $2,000.
-/
theorem HVAC_cost_per_vent
    (cost : ℕ := 20000)
    (zones : ℕ := 2)
    (vents_per_zone : ℕ := 5)
    (total_vents : ℕ := zones * vents_per_zone) :
    (cost / total_vents) = 2000 := by
  sorry

end HVAC_cost_per_vent_l97_97602


namespace find_edge_value_l97_97493

theorem find_edge_value (a b c d e_1 e_2 e_3 e_4 : ℕ) 
  (h1 : e_1 = a + b)
  (h2 : e_2 = b + c)
  (h3 : e_3 = c + d)
  (h4 : e_4 = d + a)
  (h5 : e_1 = 8)
  (h6 : e_3 = 13)
  (h7 : e_1 + e_3 = a + b + c + d)
  : e_4 = 12 := 
by sorry

end find_edge_value_l97_97493


namespace train_crossing_time_l97_97640
-- Part a: Identifying the questions and conditions

-- Question: How long does it take for the train to cross the platform?
-- Conditions:
-- 1. Speed of the train: 72 km/hr
-- 2. Length of the goods train: 440 m
-- 3. Length of the platform: 80 m

-- Part b: Identifying the solution steps and the correct answers

-- The solution steps involve:
-- 1. Summing the lengths of the train and the platform to get the total distance the train needs to cover.
-- 2. Converting the speed of the train from km/hr to m/s.
-- 3. Using the formula Time = Distance / Speed to find the time.

-- Correct answer: 26 seconds

-- Part c: Translating the question, conditions, and correct answer to a mathematically equivalent proof problem

-- Goal: Prove that the time it takes for the train to cross the platform is 26 seconds given the provided conditions.

-- Part d: Writing the Lean 4 statement


-- Definitions based on the given conditions
def speed_kmh : ℕ := 72
def length_train : ℕ := 440
def length_platform : ℕ := 80

-- Definition based on the conversion step in the solution
def speed_ms : ℕ := (72 * 1000) / 3600 -- Converting speed from km/hr to m/s

-- Goal: Prove that the time it takes for the train to cross the platform is 26 seconds
theorem train_crossing_time : ((length_train + length_platform) : ℕ) / speed_ms = 26 := by
  sorry

end train_crossing_time_l97_97640


namespace total_surface_area_of_modified_cube_l97_97789

-- Define the side length of the original cube
def side_length_cube := 3

-- Define the side length of the holes
def side_length_hole := 1

-- Define the condition of the surface area calculation
def total_surface_area_including_internal (side_length_cube side_length_hole : ℕ) : ℕ :=
  let original_surface_area := 6 * (side_length_cube * side_length_cube)
  let reduction_area := 6 * (side_length_hole * side_length_hole)
  let remaining_surface_area := original_surface_area - reduction_area
  let interior_surface_area := 6 * (4 * side_length_hole * side_length_cube)
  remaining_surface_area + interior_surface_area

-- Statement for the proof
theorem total_surface_area_of_modified_cube : total_surface_area_including_internal 3 1 = 72 :=
by
  -- This is the statement; the proof is omitted as "sorry"
  sorry

end total_surface_area_of_modified_cube_l97_97789


namespace find_orange_shells_l97_97729

theorem find_orange_shells :
  ∀ (total purple pink yellow blue : ℕ),
    total = 65 → purple = 13 → pink = 8 → yellow = 18 → blue = 12 →
    total - (purple + pink + yellow + blue) = 14 :=
by
  intros total purple pink yellow blue h_total h_purple h_pink h_yellow h_blue
  have h := h_total.symm
  rw [h_purple, h_pink, h_yellow, h_blue]
  simp only [Nat.add_assoc, Nat.add_comm, Nat.add_sub_cancel]
  sorry

end find_orange_shells_l97_97729


namespace total_votes_l97_97587

theorem total_votes (A B C D E : ℕ)
  (votes_A : ℕ) (votes_B : ℕ) (votes_C : ℕ) (votes_D : ℕ) (votes_E : ℕ)
  (dist_A : votes_A = 38 * A / 100)
  (dist_B : votes_B = 28 * B / 100)
  (dist_C : votes_C = 11 * C / 100)
  (dist_D : votes_D = 15 * D / 100)
  (dist_E : votes_E = 8 * E / 100)
  (redistrib_A : votes_A' = votes_A + 5 * A / 100)
  (redistrib_B : votes_B' = votes_B + 5 * B / 100)
  (redistrib_D : votes_D' = votes_D + 2 * D / 100)
  (total_A : votes_A' = 7320) :
  A = 17023 := 
sorry

end total_votes_l97_97587


namespace score_sd_above_mean_l97_97950

theorem score_sd_above_mean (mean std dev1 dev2 : ℝ) : 
  mean = 74 → dev1 = 2 → dev2 = 3 → mean - dev1 * std = 58 → mean + dev2 * std = 98 :=
by
  sorry

end score_sd_above_mean_l97_97950


namespace fraction_difference_l97_97760

theorem fraction_difference:
  let f1 := 2 / 3
  let f2 := 3 / 4
  let f3 := 4 / 5
  let f4 := 5 / 7
  (max f1 (max f2 (max f3 f4)) - min f1 (min f2 (min f3 f4))) = 2 / 15 :=
by
  sorry

end fraction_difference_l97_97760


namespace avg_visitors_per_day_correct_l97_97891

-- Define the given conditions
def avg_sundays : Nat := 540
def avg_other_days : Nat := 240
def num_days : Nat := 30
def sundays_in_month : Nat := 5
def other_days_in_month : Nat := 25

-- Define the total visitors calculation
def total_visitors := (sundays_in_month * avg_sundays) + (other_days_in_month * avg_other_days)

-- Define the average visitors per day calculation
def avg_visitors_per_day := total_visitors / num_days

-- State the proof problem
theorem avg_visitors_per_day_correct : avg_visitors_per_day = 290 :=
by
  sorry

end avg_visitors_per_day_correct_l97_97891


namespace fixed_point_of_shifted_exponential_l97_97351

theorem fixed_point_of_shifted_exponential (a : ℝ) (H : a^0 = 1) : a^(3-3) + 3 = 4 :=
by
  sorry

end fixed_point_of_shifted_exponential_l97_97351


namespace cone_base_radius_l97_97751

/-- Given a semicircular piece of paper with a diameter of 2 cm is used to construct the 
  lateral surface of a cone, prove that the radius of the base of the cone is 0.5 cm. --/
theorem cone_base_radius (d : ℝ) (arc_length : ℝ) (circumference : ℝ) (r : ℝ)
  (h₀ : d = 2)
  (h₁ : arc_length = (1 / 2) * d * Real.pi)
  (h₂ : circumference = arc_length)
  (h₃ : r = circumference / (2 * Real.pi)) :
  r = 0.5 :=
by
  sorry

end cone_base_radius_l97_97751


namespace julia_height_in_cm_l97_97711

def height_in_feet : ℕ := 5
def height_in_inches : ℕ := 4
def feet_to_inches : ℕ := 12
def inch_to_cm : ℝ := 2.54

theorem julia_height_in_cm : (height_in_feet * feet_to_inches + height_in_inches) * inch_to_cm = 162.6 :=
sorry

end julia_height_in_cm_l97_97711


namespace value_of_p_l97_97536

theorem value_of_p (p : ℝ) :
  (∃ x1 x2 : ℝ, x1 = 3 * x2 ∧ x^2 - (3 * p - 2) * x + p^2 - 1 = 0) →
  (p = 2 ∨ p = 14 / 11) :=
by
  sorry

end value_of_p_l97_97536


namespace distance_to_focus_l97_97828

open Real

theorem distance_to_focus {P : ℝ × ℝ} 
  (h₁ : P.2 ^ 2 = 4 * P.1)
  (h₂ : abs (P.1 + 3) = 5) :
  dist P ⟨1, 0⟩ = 3 := 
sorry

end distance_to_focus_l97_97828


namespace combined_weight_of_daughter_and_child_l97_97192

variables (M D C : ℝ)
axiom mother_daughter_grandchild_weight : M + D + C = 120
axiom daughter_weight : D = 48
axiom child_weight_fraction_of_grandmother : C = (1 / 5) * M

theorem combined_weight_of_daughter_and_child : D + C = 60 :=
  sorry

end combined_weight_of_daughter_and_child_l97_97192


namespace typing_and_editing_time_l97_97894

-- Definitions for typing and editing times for consultants together and for Mary and Jim individually
def combined_typing_time := 12.5
def combined_editing_time := 7.5
def mary_typing_time := 30.0
def jim_editing_time := 12.0

-- The total time when Jim types and Mary edits
def total_time := 42.0

-- Proof statement
theorem typing_and_editing_time :
  (combined_typing_time = 12.5) ∧ 
  (combined_editing_time = 7.5) ∧ 
  (mary_typing_time = 30.0) ∧ 
  (jim_editing_time = 12.0) →
  total_time = 42.0 := 
by
  intro h
  -- Proof to be filled later
  sorry

end typing_and_editing_time_l97_97894


namespace find_n_l97_97535

theorem find_n :
  ∃ n : ℤ, 3 ^ 3 - 7 = 4 ^ 2 + 2 + n ∧ n = 2 :=
by
  use 2
  sorry

end find_n_l97_97535


namespace sculptures_not_on_display_approx_400_l97_97097

theorem sculptures_not_on_display_approx_400 (A : ℕ) (hA : A = 900) :
  (2 / 3 * A - 2 / 9 * A) = 400 := by
  sorry

end sculptures_not_on_display_approx_400_l97_97097


namespace probability_not_all_same_color_l97_97305

def num_colors := 3
def draws := 3
def total_outcomes := num_colors ^ draws

noncomputable def prob_same_color : ℚ := (3 / total_outcomes)
noncomputable def prob_not_same_color : ℚ := 1 - prob_same_color

theorem probability_not_all_same_color :
  prob_not_same_color = 8 / 9 :=
by
  sorry

end probability_not_all_same_color_l97_97305


namespace at_least_one_distinct_root_l97_97594

theorem at_least_one_distinct_root {a b : ℝ} (ha : a > 4) (hb : b > 4) :
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + a * x₁ + b = 0 ∧ x₂^2 + a * x₂ + b = 0) ∨
    (∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ y₁^2 + b * y₁ + a = 0 ∧ y₂^2 + b * y₂ + a = 0) :=
sorry

end at_least_one_distinct_root_l97_97594


namespace digit_appears_in_3n_l97_97863

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

end digit_appears_in_3n_l97_97863


namespace height_of_highest_wave_l97_97104

theorem height_of_highest_wave 
  (h_austin : ℝ) -- Austin's height
  (h_high : ℝ) -- Highest wave's height
  (h_short : ℝ) -- Shortest wave's height 
  (height_relation1 : h_high = 4 * h_austin + 2)
  (height_relation2 : h_short = h_austin + 4)
  (surfboard : ℝ) (surfboard_len : surfboard = 7)
  (short_wave_len : h_short = surfboard + 3) :
  h_high = 26 :=
by
  -- Define local variables with the values from given conditions
  let austin_height := 6        -- as per calculation: 10 - 4 = 6
  let highest_wave_height := 26 -- as per calculation: (6 * 4) + 2 = 26
  sorry

end height_of_highest_wave_l97_97104


namespace greatest_large_chips_l97_97201

def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, 2 ≤ a ∧ 2 ≤ b ∧ n = a * b

theorem greatest_large_chips (s l : ℕ) (c : ℕ) (hc : is_composite c) (h : s + l = 60) (hs : s = l + c) :
  l ≤ 28 :=
sorry

end greatest_large_chips_l97_97201


namespace cost_per_mile_l97_97889

variable (x : ℝ)
variable (monday_miles : ℝ) (thursday_miles : ℝ) (base_cost : ℝ) (total_spent : ℝ)

-- Given conditions
def car_rental_conditions : Prop :=
  monday_miles = 620 ∧
  thursday_miles = 744 ∧
  base_cost = 150 ∧
  total_spent = 832 ∧
  total_spent = base_cost + (monday_miles + thursday_miles) * x

-- Theorem to prove the cost per mile
theorem cost_per_mile (h : car_rental_conditions x 620 744 150 832) : x = 0.50 :=
  by
    sorry

end cost_per_mile_l97_97889


namespace equal_focal_distances_l97_97185

def ellipse1 (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1
def ellipse2 (k x y : ℝ) (hk : k < 9) : Prop := x^2 / (25 - k) + y^2 / (9 - k) = 1

theorem equal_focal_distances (k : ℝ) (hk : k < 9) : 
  let f1 := 8
  let f2 := 8 
  f1 = f2 :=
by 
  sorry

end equal_focal_distances_l97_97185


namespace perpendicular_vectors_x_value_l97_97426

theorem perpendicular_vectors_x_value:
  ∀ (x : ℝ), let a : ℝ × ℝ := (1, 2)
             let b : ℝ × ℝ := (x, 1)
             (a.1 * b.1 + a.2 * b.2 = 0) → x = -2 :=
by
  intro x
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 1)
  intro h
  sorry

end perpendicular_vectors_x_value_l97_97426


namespace solve_system_l97_97295

theorem solve_system (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * y = 4 * z) (h2 : x / y = 81) (h3 : x * z = 36) :
  x = 36 ∧ y = 4 / 9 ∧ z = 1 :=
by
  sorry

end solve_system_l97_97295


namespace cost_of_each_cake_l97_97993

-- Define the conditions
def cakes : ℕ := 3
def payment_by_john : ℕ := 18
def total_payment : ℕ := payment_by_john * 2

-- Statement to prove that each cake costs $12
theorem cost_of_each_cake : (total_payment / cakes) = 12 := by
  sorry

end cost_of_each_cake_l97_97993


namespace selling_price_correct_l97_97514

theorem selling_price_correct (C P_rate : ℝ) (hC : C = 50) (hP_rate : P_rate = 0.40) : 
  C + (P_rate * C) = 70 :=
by
  sorry

end selling_price_correct_l97_97514


namespace sqrt11_plus_sqrt3_lt_sqrt9_plus_sqrt5_l97_97396

noncomputable def compare_sq_roots_sum : Prop := 
  (Real.sqrt 11 + Real.sqrt 3) < (Real.sqrt 9 + Real.sqrt 5)

theorem sqrt11_plus_sqrt3_lt_sqrt9_plus_sqrt5 :
  compare_sq_roots_sum :=
sorry

end sqrt11_plus_sqrt3_lt_sqrt9_plus_sqrt5_l97_97396


namespace arccos_pi_over_3_l97_97933

theorem arccos_pi_over_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_pi_over_3_l97_97933


namespace pentagon_area_l97_97474

open Function 

/-
Given a convex pentagon FGHIJ with the following properties:
  1. ∠F = ∠G = 100°
  2. JF = FG = GH = 3
  3. HI = IJ = 5
Prove that the area of pentagon FGHIJ is approximately 15.2562 square units.
-/

noncomputable def area_pentagon_FGHIJ : ℝ :=
  let sin100 := Real.sin (100 * Real.pi / 180)
  let area_FGJ := (3 * 3 * sin100) / 2
  let area_HIJ := (5 * 5 * Real.sqrt 3) / 4
  area_FGJ + area_HIJ

theorem pentagon_area : abs (area_pentagon_FGHIJ - 15.2562) < 0.0001 := by
  sorry

end pentagon_area_l97_97474


namespace calculate_price_per_pound_of_meat_l97_97315

noncomputable def price_per_pound_of_meat : ℝ :=
  let total_hours := 50
  let w := 8
  let m_pounds := 20
  let fv_pounds := 15
  let fv_pp := 4
  let b_pounds := 60
  let b_pp := 1.5
  let j_wage := 10
  let j_hours := 10
  let j_rate := 1.5

  -- known costs
  let fv_cost := fv_pounds * fv_pp
  let b_cost := b_pounds * b_pp
  let j_cost := j_hours * j_wage * j_rate

  -- total costs
  let total_cost := total_hours * w
  let known_costs := fv_cost + b_cost + j_cost

  (total_cost - known_costs) / m_pounds

theorem calculate_price_per_pound_of_meat : price_per_pound_of_meat = 5 := by
  sorry

end calculate_price_per_pound_of_meat_l97_97315


namespace area_of_new_triangle_geq_twice_sum_of_areas_l97_97075

noncomputable def area_of_triangle (a b c : ℝ) (alpha : ℝ) : ℝ :=
  0.5 * a * b * (Real.sin alpha)

theorem area_of_new_triangle_geq_twice_sum_of_areas
  (a1 b1 c a2 b2 alpha : ℝ)
  (h1 : a1 <= b1) (h2 : b1 <= c) (h3 : a2 <= b2) (h4 : b2 <= c) :
  let α_1 := Real.arcsin ((a1 + a2) / (2 * c))
  let area1 := area_of_triangle a1 b1 c alpha
  let area2 := area_of_triangle a2 b2 c alpha
  let area_new := area_of_triangle (a1 + a2) (b1 + b2) (2 * c) α_1
  area_new >= 2 * (area1 + area2) :=
sorry

end area_of_new_triangle_geq_twice_sum_of_areas_l97_97075


namespace compare_abc_l97_97684

noncomputable def a : ℝ := 2^(1/2)
noncomputable def b : ℝ := 3^(1/3)
noncomputable def c : ℝ := Real.log 2

theorem compare_abc : b > a ∧ a > c :=
by
  sorry

end compare_abc_l97_97684


namespace negation_of_proposition_l97_97865

theorem negation_of_proposition : 
  ¬(∀ x : ℝ, x^2 + x + 1 > 0) ↔ ∃ x : ℝ, x^2 + x + 1 ≤ 0 :=
by
  sorry

end negation_of_proposition_l97_97865


namespace probability_of_exactly_one_common_venue_l97_97313

noncomputable def probability_one_common_venue : ℚ :=
  let total_ways : ℕ := (Nat.choose 4 2) * (Nat.choose 4 2)
  let common_ways : ℕ := 4 * Nat.factorial 3 / Nat.factorial (3 - 2)
  (common_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_exactly_one_common_venue :
  probability_one_common_venue = 2 / 3 := by
  sorry

end probability_of_exactly_one_common_venue_l97_97313


namespace time_to_cover_escalator_l97_97238

noncomputable def average_speed (initial_speed final_speed : ℝ) : ℝ :=
  (initial_speed + final_speed) / 2

noncomputable def combined_speed (escalator_speed person_average_speed : ℝ) : ℝ :=
  escalator_speed + person_average_speed

noncomputable def coverage_time (length combined_speed : ℝ) : ℝ :=
  length / combined_speed

theorem time_to_cover_escalator
  (escalator_speed : ℝ := 20)
  (length : ℝ := 300)
  (initial_person_speed : ℝ := 3)
  (final_person_speed : ℝ := 5) :
  coverage_time length (combined_speed escalator_speed (average_speed initial_person_speed final_person_speed)) = 12.5 :=
by
  sorry

end time_to_cover_escalator_l97_97238


namespace tables_count_is_correct_l97_97510

-- Definitions based on conditions
def invited_people : ℕ := 18
def people_didnt_show_up : ℕ := 12
def people_per_table : ℕ := 3

-- Calculation based on definitions
def people_attended : ℕ := invited_people - people_didnt_show_up
def tables_needed : ℕ := people_attended / people_per_table

-- The main theorem statement
theorem tables_count_is_correct : tables_needed = 2 := by
  unfold tables_needed
  unfold people_attended
  unfold invited_people
  unfold people_didnt_show_up
  unfold people_per_table
  sorry

end tables_count_is_correct_l97_97510


namespace constant_term_binomial_expansion_l97_97343

/--
 If the constant term in the expansion of (a * x^3 + 1 / sqrt x)^7 is 14, then a = 2.
-/
theorem constant_term_binomial_expansion (a : ℝ) 
  (h : (∃ (T₇ : ℝ), T₇ = (a^1 * (Nat.choose 7 6)) ∧ T₇ = 14)) : 
  a = 2 :=
by
  sorry

end constant_term_binomial_expansion_l97_97343


namespace calc_pow_expression_l97_97394

theorem calc_pow_expression : (27^3 * 9^2) / 3^15 = 1 / 9 := 
by sorry

end calc_pow_expression_l97_97394


namespace values_of_xyz_l97_97494

theorem values_of_xyz (x y z : ℝ) (h1 : 2 * x - y + z = 14) (h2 : y = 2) (h3 : x + z = 3 * y + 5) : 
  x = 5 ∧ y = 2 ∧ z = 6 := 
by
  sorry

end values_of_xyz_l97_97494


namespace problem_statement_l97_97571

noncomputable def f : ℝ → ℝ := sorry

theorem problem_statement (h1 : ∀ ⦃x y⦄, x > 4 → y > x → f y < f x)
                          (h2 : ∀ x, f (4 + x) = f (4 - x)) : f 3 > f 6 :=
by 
  sorry

end problem_statement_l97_97571


namespace total_heads_l97_97091

def total_legs : ℕ := 45
def num_cats : ℕ := 7
def legs_per_cat : ℕ := 4
def captain_legs : ℕ := 1
def legs_humans := total_legs - (num_cats * legs_per_cat) - captain_legs
def num_humans := legs_humans / 2

theorem total_heads : (num_cats + (num_humans + 1)) = 15 := by
  sorry

end total_heads_l97_97091


namespace intersection_A_B_l97_97632

def A : Set ℝ := {x | x^2 - x = 0}
def B : Set ℝ := {x | x^2 + x = 0}

theorem intersection_A_B : A ∩ B = {0} :=
by
  sorry

end intersection_A_B_l97_97632


namespace range_of_a_l97_97691

-- Define the set A
def A (a x : ℝ) := 6 * x + a > 0

-- Theorem stating the range of a given the conditions
theorem range_of_a (a : ℝ) (h : ¬ A a 1) : a ≤ -6 :=
by
  -- Here we would provide the proof
  sorry

end range_of_a_l97_97691


namespace find_A_for_club_suit_l97_97433

def club_suit (A B : ℝ) : ℝ := 3 * A + 2 * B^2 + 5

theorem find_A_for_club_suit :
  ∃ A : ℝ, club_suit A 3 = 73 ∧ A = 50 / 3 :=
sorry

end find_A_for_club_suit_l97_97433


namespace henry_time_around_track_l97_97428

theorem henry_time_around_track (H : ℕ) : 
  (∀ (M := 12), lcm M H = 84) → H = 7 :=
by
  sorry

end henry_time_around_track_l97_97428


namespace chemical_reaction_proof_l97_97142

-- Define the given number of moles for each reactant
def moles_NaOH : ℕ := 4
def moles_NH4Cl : ℕ := 3

-- Define the balanced chemical equation stoichiometry
def stoichiometry_ratio_NaOH_NH4Cl : ℕ := 1

-- Define the product formation based on the limiting reactant
theorem chemical_reaction_proof
  (moles_NaOH : ℕ)
  (moles_NH4Cl : ℕ)
  (stoichiometry_ratio_NaOH_NH4Cl : ℕ)
  (h1 : moles_NaOH = 4)
  (h2 : moles_NH4Cl = 3)
  (h3 : stoichiometry_ratio_NaOH_NH4Cl = 1):
  (3 = 3 * 1) ∧
  (3 = 3 * 1) ∧
  (3 = 3 * 1) ∧
  (3 = moles_NH4Cl) ∧
  (1 = moles_NaOH - moles_NH4Cl) :=
by {
  -- Provide assumptions based on the problem
  sorry
}

end chemical_reaction_proof_l97_97142


namespace timeSpentReading_l97_97838

def totalTime : ℕ := 120
def timeOnPiano : ℕ := 30
def timeWritingMusic : ℕ := 25
def timeUsingExerciser : ℕ := 27

theorem timeSpentReading :
  totalTime - timeOnPiano - timeWritingMusic - timeUsingExerciser = 38 := by
  sorry

end timeSpentReading_l97_97838


namespace total_boxes_sold_is_189_l97_97340

-- Define the conditions
def boxes_sold_friday : ℕ := 40
def boxes_sold_saturday := 2 * boxes_sold_friday - 10
def boxes_sold_sunday := boxes_sold_saturday / 2
def boxes_sold_monday := boxes_sold_sunday + (boxes_sold_sunday / 4)

-- Define the total boxes sold over the four days
def total_boxes_sold := boxes_sold_friday + boxes_sold_saturday + boxes_sold_sunday + boxes_sold_monday

-- Theorem to prove the total number of boxes sold is 189
theorem total_boxes_sold_is_189 : total_boxes_sold = 189 := by
  sorry

end total_boxes_sold_is_189_l97_97340


namespace evaluate_integral_l97_97280

noncomputable def integral_problem : Real :=
  ∫ x in (-2 : Real)..(2 : Real), (Real.sqrt (4 - x^2) - x^2017)

theorem evaluate_integral :
  integral_problem = 2 * Real.pi :=
sorry

end evaluate_integral_l97_97280


namespace trip_early_movie_savings_l97_97618

theorem trip_early_movie_savings : 
  let evening_ticket_cost : ℝ := 10
  let food_combo_cost : ℝ := 10
  let ticket_discount : ℝ := 0.20
  let food_discount : ℝ := 0.50
  let evening_total_cost := evening_ticket_cost + food_combo_cost
  let savings_on_ticket := evening_ticket_cost * ticket_discount
  let savings_on_food := food_combo_cost * food_discount
  let total_savings := savings_on_ticket + savings_on_food
  total_savings = 7 :=
by
  sorry

end trip_early_movie_savings_l97_97618


namespace positive_integer_expression_iff_l97_97147

theorem positive_integer_expression_iff (p : ℕ) : (0 < p) ∧ (∃ k : ℕ, 0 < k ∧ 4 * p + 35 = k * (3 * p - 8)) ↔ p = 3 :=
by
  sorry

end positive_integer_expression_iff_l97_97147


namespace ice_creams_needed_l97_97723

theorem ice_creams_needed (game_cost : ℕ) (ice_cream_price : ℕ) (games_to_buy : ℕ) 
    (h1 : game_cost = 60) (h2 : ice_cream_price = 5) (h3 : games_to_buy = 2) : 
    (games_to_buy * game_cost) / ice_cream_price = 24 :=
by
  rw [h1, h2, h3]
  sorry

end ice_creams_needed_l97_97723


namespace ratio_apples_peaches_l97_97200

theorem ratio_apples_peaches (total_fruits oranges peaches apples : ℕ)
  (h_total : total_fruits = 56)
  (h_oranges : oranges = total_fruits / 4)
  (h_peaches : peaches = oranges / 2)
  (h_apples : apples = 35) : apples / peaches = 5 := 
by
  sorry

end ratio_apples_peaches_l97_97200


namespace max_tan_B_l97_97831

theorem max_tan_B (A B : ℝ) (h : Real.sin (2 * A + B) = 2 * Real.sin B) : 
  Real.tan B ≤ Real.sqrt 3 / 3 := sorry

end max_tan_B_l97_97831


namespace shaded_area_l97_97589

theorem shaded_area (r : ℝ) (sector_area : ℝ) (h1 : r = 4) (h2 : sector_area = 2 * Real.pi) : 
  sector_area - (1 / 2 * (r * Real.sqrt 2) * (r * Real.sqrt 2)) = 2 * Real.pi - 4 :=
by 
  -- Lean proof follows
  sorry

end shaded_area_l97_97589


namespace compare_exponents_l97_97243

noncomputable def exp_of_log (a : ℝ) (b : ℝ) : ℝ :=
  Real.exp ((1 / b) * Real.log a)

theorem compare_exponents :
  let a := exp_of_log 4 4
  let b := exp_of_log 5 5
  let c := exp_of_log 16 16
  let d := exp_of_log 25 25
  a = max a (max b (max c d)) ∧
  b = max (min a (max b (max c d))) (max (min b (max c d)) (max (min c d) (min d (min a b))))
  :=
  by
    sorry

end compare_exponents_l97_97243


namespace sum_of_powers_twice_square_l97_97722

theorem sum_of_powers_twice_square (x y : ℤ) : 
  ∃ z : ℤ, x^4 + y^4 + (x + y)^4 = 2 * z^2 := by
  let z := x^2 + x * y + y^2
  use z
  sorry

end sum_of_powers_twice_square_l97_97722


namespace lcm_18_35_l97_97807

-- Given conditions: Prime factorizations of 18 and 35
def factorization_18 : Prop := (18 = 2^1 * 3^2)
def factorization_35 : Prop := (35 = 5^1 * 7^1)

-- The goal is to prove that the least common multiple of 18 and 35 is 630
theorem lcm_18_35 : factorization_18 ∧ factorization_35 → Nat.lcm 18 35 = 630 := by
  sorry -- Proof to be filled in

end lcm_18_35_l97_97807


namespace percentage_of_boys_playing_soccer_l97_97707

theorem percentage_of_boys_playing_soccer
  (total_students : ℕ)
  (boys : ℕ)
  (students_playing_soccer : ℕ)
  (girl_students_not_playing_soccer : ℕ)
  (h1 : total_students = 420)
  (h2 : boys = 296)
  (h3 : students_playing_soccer = 250)
  (h4 : girl_students_not_playing_soccer = 89) :
  (students_playing_soccer - (total_students - boys - girl_students_not_playing_soccer)) * 100 / students_playing_soccer = 86 :=
by
  sorry

end percentage_of_boys_playing_soccer_l97_97707


namespace total_biscuits_needed_l97_97465

-- Definitions
def number_of_dogs : ℕ := 2
def biscuits_per_dog : ℕ := 3

-- Theorem statement
theorem total_biscuits_needed : number_of_dogs * biscuits_per_dog = 6 :=
by sorry

end total_biscuits_needed_l97_97465


namespace gcd_6051_10085_l97_97188

theorem gcd_6051_10085 : Nat.gcd 6051 10085 = 2017 := by
  sorry

end gcd_6051_10085_l97_97188


namespace domain_of_log_function_l97_97348

theorem domain_of_log_function (x : ℝ) : 1 - x > 0 ↔ x < 1 := by
  sorry

end domain_of_log_function_l97_97348


namespace g_three_eighths_l97_97740

variable (g : ℝ → ℝ)

-- Conditions
axiom g_zero : g 0 = 0
axiom monotonic : ∀ {x y : ℝ}, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y
axiom symmetry : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x
axiom scaling : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3

-- The theorem statement we need to prove
theorem g_three_eighths : g (3 / 8) = 2 / 9 :=
sorry

end g_three_eighths_l97_97740


namespace years_of_school_eq_13_l97_97710

/-- Conditions definitions -/
def cost_per_semester : ℕ := 20000
def semesters_per_year : ℕ := 2
def total_cost : ℕ := 520000

/-- Derived definitions from conditions -/
def cost_per_year := cost_per_semester * semesters_per_year
def number_of_years := total_cost / cost_per_year

/-- Proof that number of years equals 13 given the conditions -/
theorem years_of_school_eq_13 : number_of_years = 13 :=
by sorry

end years_of_school_eq_13_l97_97710


namespace painted_sphere_area_proportionality_l97_97911

theorem painted_sphere_area_proportionality
  (r : ℝ)
  (R_inner R_outer : ℝ)
  (A_inner : ℝ)
  (h_r : r = 1)
  (h_R_inner : R_inner = 4)
  (h_R_outer : R_outer = 6)
  (h_A_inner : A_inner = 47) :
  ∃ A_outer : ℝ, A_outer = 105.75 :=
by
  have ratio := (R_outer / R_inner)^2
  have A_outer := A_inner * ratio
  use A_outer
  sorry

end painted_sphere_area_proportionality_l97_97911


namespace Kyle_fish_count_l97_97660

def Carla_fish := 8
def Total_fish := 36
def Kyle_fish := (Total_fish - Carla_fish) / 2

theorem Kyle_fish_count : Kyle_fish = 14 :=
by
  -- This proof will be provided later
  sorry

end Kyle_fish_count_l97_97660


namespace complete_square_proof_l97_97633

def quadratic_eq := ∀ (x : ℝ), x^2 - 6 * x + 5 = 0
def form_completing_square (b c : ℝ) := ∀ (x : ℝ), (x + b)^2 = c

theorem complete_square_proof :
  quadratic_eq → (∃ b c : ℤ, form_completing_square (b : ℝ) (c : ℝ) ∧ b + c = 11) :=
by
  sorry

end complete_square_proof_l97_97633


namespace evaluate_expression_l97_97114

theorem evaluate_expression :
  (2^1 - 3 + 5^3 - 2)⁻¹ * 3 = (3 : ℚ) / 122 :=
by
  -- proof goes here
  sorry

end evaluate_expression_l97_97114


namespace floor_of_neg_seven_fourths_l97_97279

theorem floor_of_neg_seven_fourths : (Int.floor (-7/4 : ℚ)) = -2 :=
by
  sorry

end floor_of_neg_seven_fourths_l97_97279


namespace coloringBooks_shelves_l97_97785

variables (initialStock soldBooks shelves : ℕ)

-- Given conditions
def initialBooks : initialStock = 87 := sorry
def booksSold : soldBooks = 33 := sorry
def numberOfShelves : shelves = 9 := sorry

-- Number of coloring books per shelf
def coloringBooksPerShelf (remainingBooksResult : ℕ) (booksPerShelfResult : ℕ) : Prop :=
  remainingBooksResult = initialStock - soldBooks ∧ booksPerShelfResult = remainingBooksResult / shelves

-- Prove the number of coloring books per shelf is 6
theorem coloringBooks_shelves (remainingBooksResult booksPerShelfResult : ℕ) : 
  coloringBooksPerShelf initialStock soldBooks shelves remainingBooksResult booksPerShelfResult →
  booksPerShelfResult = 6 :=
sorry

end coloringBooks_shelves_l97_97785


namespace find_z_when_w_15_l97_97007

-- Define a direct variation relationship
def varies_directly (z w : ℕ) (k : ℕ) : Prop :=
  z = k * w

-- Using the given conditions and to prove the statement
theorem find_z_when_w_15 :
  ∃ k, (varies_directly 10 5 k) → (varies_directly 30 15 k) :=
by
  sorry

end find_z_when_w_15_l97_97007


namespace work_rate_solution_l97_97378

theorem work_rate_solution (y : ℕ) (hy : y > 0) : 
  ∃ z : ℕ, z = (y^2 + 3 * y) / (2 * y + 3) :=
by
  sorry

end work_rate_solution_l97_97378


namespace sufficient_but_not_necessary_condition_l97_97128

theorem sufficient_but_not_necessary_condition (a : ℝ) : (a > 1 → (1 / a < 1)) ∧ ¬((1 / a < 1) → a > 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l97_97128


namespace remainder_37_remainder_73_l97_97907

theorem remainder_37 (N : ℕ) (k : ℕ) (h : N = 1554 * k + 131) : N % 37 = 20 := sorry

theorem remainder_73 (N : ℕ) (k : ℕ) (h : N = 1554 * k + 131) : N % 73 = 58 := sorry

end remainder_37_remainder_73_l97_97907


namespace find_complex_number_l97_97292

open Complex

theorem find_complex_number (a b : ℝ) (z : ℂ) 
  (h₁ : (∀ b: ℝ, (b^2 + 4 * b + 4 = 0) ∧ (b + a = 0))) :
  z = 2 - 2 * Complex.I :=
  sorry

end find_complex_number_l97_97292


namespace fraction_of_students_with_mentor_l97_97705

theorem fraction_of_students_with_mentor (s n : ℕ) (h : n / 2 = s / 3) :
  (n / 2 + s / 3 : ℚ) / (n + s : ℚ) = 2 / 5 := by
  sorry

end fraction_of_students_with_mentor_l97_97705


namespace eight_digit_descending_numbers_count_l97_97016

theorem eight_digit_descending_numbers_count : (Nat.choose 10 2) = 45 :=
by
  sorry

end eight_digit_descending_numbers_count_l97_97016


namespace simplify_expression1_simplify_expression2_l97_97058

-- Define variables as real numbers or appropriate domains
variables {a b x y: ℝ}

-- Problem 1
theorem simplify_expression1 : (2 * a - b) - (2 * b - 3 * a) - 2 * (a - 2 * b) = 3 * a + b :=
by sorry

-- Problem 2
theorem simplify_expression2 : (4 * x^2 - 5 * x * y) - (1 / 3 * y^2 + 2 * x^2) + 2 * (3 * x * y - 1 / 4 * y^2 - 1 / 12 * y^2) = 2 * x^2 + x * y - y^2 :=
by sorry

end simplify_expression1_simplify_expression2_l97_97058


namespace savings_after_increase_l97_97381

/-- A man saves 20% of his monthly salary. If on account of dearness of things
    he is to increase his monthly expenses by 20%, he is only able to save a
    certain amount per month. His monthly salary is Rs. 6250. -/
theorem savings_after_increase (monthly_salary : ℝ) (initial_savings_percentage : ℝ)
  (increase_expenses_percentage : ℝ) (final_savings : ℝ) :
  monthly_salary = 6250 ∧
  initial_savings_percentage = 0.20 ∧
  increase_expenses_percentage = 0.20 →
  final_savings = 250 :=
by
  sorry

end savings_after_increase_l97_97381


namespace find_a2_an_le_2an_next_sum_bounds_l97_97415

variable {a : ℕ → ℝ}
variable (S : ℕ → ℝ)

-- Given conditions
axiom seq_condition (n : ℕ) (h_pos : a n > 0) : 
  a n ^ 2 + a n = 3 * (a (n + 1)) ^ 2 + 2 * a (n + 1)
axiom a1_condition : a 1 = 1

-- Question 1: Prove the value of a2
theorem find_a2 : a 2 = (Real.sqrt 7 - 1) / 3 :=
  sorry

-- Question 2: Prove a_n ≤ 2 * a_{n+1} for any n ∈ N*
theorem an_le_2an_next (n : ℕ) (h_n : n > 0) : a n ≤ 2 * a (n + 1) :=
  sorry

-- Question 3: Prove 2 - 1 / 2^(n - 1) ≤ S_n < 3 for any n ∈ N*
theorem sum_bounds (n : ℕ) (h_n : n > 0) : 
  2 - 1 / 2 ^ (n - 1) ≤ S n ∧ S n < 3 :=
  sorry

end find_a2_an_le_2an_next_sum_bounds_l97_97415


namespace find_coefficients_l97_97686

theorem find_coefficients (a b : ℚ) (h_a_nonzero : a ≠ 0)
  (h_prod : (3 * b - 2 * a = 0) ∧ (-2 * b + 3 = 0)) : 
  a = 9 / 4 ∧ b = 3 / 2 :=
by
  sorry

end find_coefficients_l97_97686


namespace find_unknown_rate_l97_97642

def cost_with_discount_and_tax (original_price : ℝ) (count : ℕ) (discount : ℝ) (tax : ℝ) : ℝ :=
  let discounted_price := (original_price * count) * (1 - discount)
  discounted_price * (1 + tax)

theorem find_unknown_rate :
  let total_blankets := 10
  let average_price := 160
  let total_cost := total_blankets * average_price
  let cost_100_blankets := cost_with_discount_and_tax 100 3 0.05 0.12
  let cost_150_blankets := cost_with_discount_and_tax 150 5 0.10 0.15
  let cost_unknown_blankets := 2 * x
  total_cost = cost_100_blankets + cost_150_blankets + cost_unknown_blankets →
  x = 252.275 :=
by
  sorry

end find_unknown_rate_l97_97642


namespace sin_order_l97_97371

theorem sin_order :
  ∀ (sin₁ sin₂ sin₃ sin₄ sin₆ : ℝ),
  sin₁ = Real.sin 1 ∧ 
  sin₂ = Real.sin 2 ∧ 
  sin₃ = Real.sin 3 ∧ 
  sin₄ = Real.sin 4 ∧ 
  sin₆ = Real.sin 6 →
  sin₂ > sin₁ ∧ sin₁ > sin₃ ∧ sin₃ > sin₆ ∧ sin₆ > sin₄ :=
by
  sorry

end sin_order_l97_97371


namespace ice_creams_needed_l97_97724

theorem ice_creams_needed (game_cost : ℕ) (ice_cream_price : ℕ) (games_to_buy : ℕ) 
    (h1 : game_cost = 60) (h2 : ice_cream_price = 5) (h3 : games_to_buy = 2) : 
    (games_to_buy * game_cost) / ice_cream_price = 24 :=
by
  rw [h1, h2, h3]
  sorry

end ice_creams_needed_l97_97724


namespace width_of_beam_l97_97606

theorem width_of_beam (L W k : ℝ) (h1 : L = k * W) (h2 : 250 = k * 1.5) : 
  (k = 166.6667) → (583.3333 = 166.6667 * W) → W = 3.5 :=
by 
  intro hk1 
  intro h583
  sorry

end width_of_beam_l97_97606


namespace min_positive_period_cos_2x_l97_97191

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)

theorem min_positive_period_cos_2x :
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ ∀ T' > 0, (∀ x : ℝ, f (x + T') = f x) → T' ≥ T := 
sorry

end min_positive_period_cos_2x_l97_97191


namespace smallest_consecutive_odd_sum_l97_97482

theorem smallest_consecutive_odd_sum (a b c d e : ℤ)
    (h1 : b = a + 2)
    (h2 : c = a + 4)
    (h3 : d = a + 6)
    (h4 : e = a + 8)
    (h5 : a + b + c + d + e = 375) : a = 71 :=
by
  -- the proof will go here
  sorry

end smallest_consecutive_odd_sum_l97_97482


namespace positive_integer_pairs_l97_97403

theorem positive_integer_pairs (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  (∃ k : ℕ, k > 0 ∧ k = a^2 / (2 * a * b^2 - b^3 + 1)) ↔ 
  (∃ l : ℕ, 0 < l ∧ 
    ((a = 2 * l ∧ b = 1) ∨ (a = l ∧ b = 2 * l) ∨ (a = 8 * l^4 - l ∧ b = 2 * l))) :=
by
  sorry

end positive_integer_pairs_l97_97403


namespace solve_equation_l97_97600

theorem solve_equation (a : ℝ) : 
  {x : ℝ | x * (x + a)^3 * (5 - x) = 0} = {0, -a, 5} :=
sorry

end solve_equation_l97_97600


namespace value_of_k_l97_97003

theorem value_of_k (x y k : ℝ) (h1 : 3 * x + 2 * y = k + 1) (h2 : 2 * x + 3 * y = k) (h3 : x + y = 2) :
  k = 9 / 2 :=
by
  sorry

end value_of_k_l97_97003


namespace noah_small_paintings_sold_last_month_l97_97334

theorem noah_small_paintings_sold_last_month
  (large_painting_price small_painting_price : ℕ)
  (large_paintings_sold_last_month : ℕ)
  (total_sales_this_month : ℕ)
  (sale_multiplier : ℕ)
  (x : ℕ)
  (h1 : large_painting_price = 60)
  (h2 : small_painting_price = 30)
  (h3 : large_paintings_sold_last_month = 8)
  (h4 : total_sales_this_month = 1200)
  (h5 : sale_multiplier = 2) :
  (2 * ((large_paintings_sold_last_month * large_painting_price) + (x * small_painting_price)) = total_sales_this_month) → x = 4 :=
by
  sorry

end noah_small_paintings_sold_last_month_l97_97334


namespace crayons_count_l97_97954

-- Definitions based on the conditions given in the problem
def total_crayons : Nat := 96
def benny_crayons : Nat := 12
def fred_crayons : Nat := 2 * benny_crayons
def jason_crayons (sarah_crayons : Nat) : Nat := 3 * sarah_crayons

-- Stating the proof goal
theorem crayons_count (sarah_crayons : Nat) :
  fred_crayons + benny_crayons + jason_crayons sarah_crayons + sarah_crayons = total_crayons →
  sarah_crayons = 15 ∧
  fred_crayons = 24 ∧
  jason_crayons sarah_crayons = 45 ∧
  benny_crayons = 12 :=
by
  sorry

end crayons_count_l97_97954


namespace proof_problem1_proof_problem2_proof_problem3_l97_97921

-- Definition of the three mathematical problems
def problem1 : Prop := 8 / (-2) - (-4) * (-3) = -16

def problem2 : Prop := -2^3 + (-3) * ((-2)^3 + 5) = 1

def problem3 (x : ℝ) : Prop := (2 * x^2)^3 * x^2 - x^10 / x^2 = 7 * x^8

-- Statements of the proofs required
theorem proof_problem1 : problem1 :=
by sorry

theorem proof_problem2 : problem2 :=
by sorry

theorem proof_problem3 (x : ℝ) : problem3 x :=
by sorry

end proof_problem1_proof_problem2_proof_problem3_l97_97921


namespace garden_area_l97_97861

-- Definitions for the conditions
def perimeter : ℕ := 36
def width : ℕ := 10

-- Define the length using the perimeter and width
def length : ℕ := (perimeter - 2 * width) / 2

-- Define the area using the length and width
def area : ℕ := length * width

-- The theorem to prove the area is 80 square feet given the conditions
theorem garden_area : area = 80 :=
by 
  -- Here we use sorry to skip the proof
  sorry

end garden_area_l97_97861


namespace slope_at_A_is_7_l97_97610

def curve (x : ℝ) : ℝ := x^2 + 3 * x

def point_A : ℝ × ℝ := (2, 10)

theorem slope_at_A_is_7 : (deriv curve 2) = 7 := 
by
  sorry

end slope_at_A_is_7_l97_97610


namespace at_least_one_not_beyond_20m_l97_97307

variables (p q : Prop)

theorem at_least_one_not_beyond_20m : (¬ p ∨ ¬ q) ↔ ¬ (p ∧ q) :=
by sorry

end at_least_one_not_beyond_20m_l97_97307


namespace proof_problem_l97_97246

-- Definitions
def is_factor (a b : ℕ) : Prop := ∃ k, b = a * k
def is_divisor (a b : ℕ) : Prop := ∃ k, b = k * a

-- Conditions
def condition_A : Prop := is_factor 4 24
def condition_B : Prop := is_divisor 19 152 ∧ ¬ is_divisor 19 96
def condition_E : Prop := is_factor 6 180

-- Proof problem statement
theorem proof_problem : condition_A ∧ condition_B ∧ condition_E :=
by sorry

end proof_problem_l97_97246


namespace people_stools_chairs_l97_97586

def total_legs (x y z : ℕ) : ℕ := 2 * x + 3 * y + 4 * z 

theorem people_stools_chairs (x y z : ℕ) : 
  (x > y) → (x > z) → (x < y + z) → (total_legs x y z = 32) → 
  (x = 5 ∧ y = 2 ∧ z = 4) :=
by
  intro h1 h2 h3 h4
  sorry

end people_stools_chairs_l97_97586


namespace simplify_exponents_product_l97_97499

theorem simplify_exponents_product :
  (10^0.5) * (10^0.25) * (10^0.15) * (10^0.05) * (10^1.05) = 100 := by
sorry

end simplify_exponents_product_l97_97499


namespace amount_paid_is_51_l97_97301

def original_price : ℕ := 204
def discount_fraction : ℚ := 0.75
def paid_fraction : ℚ := 1 - discount_fraction

theorem amount_paid_is_51 : paid_fraction * original_price = 51 := by
  sorry

end amount_paid_is_51_l97_97301


namespace Jake_has_8_peaches_l97_97709

variables (Jake Steven Jill : ℕ)

-- The conditions
def condition1 : Steven = 15 := sorry
def condition2 : Steven = Jill + 14 := sorry
def condition3 : Jake = Steven - 7 := sorry

-- The proof statement
theorem Jake_has_8_peaches 
  (h1 : Steven = 15) 
  (h2 : Steven = Jill + 14) 
  (h3 : Jake = Steven - 7) : Jake = 8 :=
by
  -- The proof will go here
  sorry

end Jake_has_8_peaches_l97_97709


namespace rainy_days_last_week_l97_97218

theorem rainy_days_last_week (n : ℤ) (R NR : ℕ) (h1 : n * R + 3 * NR = 20)
  (h2 : 3 * NR = n * R + 10) (h3 : R + NR = 7) : R = 2 :=
sorry

end rainy_days_last_week_l97_97218


namespace correct_statements_proof_l97_97887

theorem correct_statements_proof :
  (∀ (a b : ℤ), a - 3 = b - 3 → a = b) ∧
  ¬ (∀ (a b c : ℤ), a = b → a + c = b - c) ∧
  (∀ (a b m : ℤ), m ≠ 0 → (a / m) = (b / m) → a = b) ∧
  ¬ (∀ (a : ℤ), a^2 = 2 * a → a = 2) :=
by
  -- Here we would prove the statements individually:
  -- sorry is a placeholder suggesting that the proofs need to be filled in.
  sorry

end correct_statements_proof_l97_97887


namespace factor_expression_l97_97124

theorem factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := 
by 
  sorry

end factor_expression_l97_97124


namespace total_time_spent_l97_97163

def one_round_time : ℕ := 30
def saturday_initial_rounds : ℕ := 1
def saturday_additional_rounds : ℕ := 10
def sunday_rounds : ℕ := 15

theorem total_time_spent :
  one_round_time * (saturday_initial_rounds + saturday_additional_rounds + sunday_rounds) = 780 := by
  sorry

end total_time_spent_l97_97163


namespace alphazia_lost_words_l97_97591

def alphazia_letters := 128
def forbidden_letters := 2
def total_forbidden_pairs := forbidden_letters * alphazia_letters

theorem alphazia_lost_words :
  let one_letter_lost := forbidden_letters
  let two_letter_lost := 2 * alphazia_letters
  one_letter_lost + two_letter_lost = 258 :=
by
  sorry

end alphazia_lost_words_l97_97591


namespace floor_neg_seven_quarter_l97_97268

theorem floor_neg_seven_quarter : 
  ∃ x : ℤ, -2 ≤ (-7 / 4 : ℚ) ∧ (-7 / 4 : ℚ) < -1 ∧ x = -2 := by
  have h1 : (-7 / 4 : ℚ) = -1.75 := by norm_num
  have h2 : -2 ≤ -1.75 := by norm_num
  have h3 : -1.75 < -1 := by norm_num
  use -2
  exact ⟨h2, h3, rfl⟩
  sorry

end floor_neg_seven_quarter_l97_97268


namespace find_m_value_l97_97050

def magic_box_output (a b : ℝ) : ℝ := a^2 + b - 1

theorem find_m_value :
  ∃ m : ℝ, (magic_box_output m (-2 * m) = 2) ↔ (m = 3 ∨ m = -1) :=
by
  sorry

end find_m_value_l97_97050


namespace fraction_given_to_emma_is_7_over_36_l97_97522

-- Define initial quantities
def stickers_noah : ℕ := sorry
def stickers_emma : ℕ := 3 * stickers_noah
def stickers_liam : ℕ := 12 * stickers_noah

-- Define required number of stickers for equal distribution
def total_stickers := stickers_noah + stickers_emma + stickers_liam
def equal_stickers := total_stickers / 3

-- Define the number of stickers to be given to Emma and the fraction of Liam's stickers he should give to Emma
def stickers_given_to_emma := equal_stickers - stickers_emma
def fraction_liams_stickers_given_to_emma := stickers_given_to_emma / stickers_liam

-- Theorem statement
theorem fraction_given_to_emma_is_7_over_36 :
  fraction_liams_stickers_given_to_emma = 7 / 36 :=
sorry

end fraction_given_to_emma_is_7_over_36_l97_97522


namespace crayons_allocation_correct_l97_97956

noncomputable def crayons_allocation : Prop :=
  ∃ (F B J S : ℕ), 
    F + B + J + S = 96 ∧ 
    F = 2 * B ∧ 
    J = 3 * S ∧ 
    B = 12 ∧ 
    F = 24 ∧ 
    J = 45 ∧ 
    S = 15

theorem crayons_allocation_correct : crayons_allocation :=
  sorry

end crayons_allocation_correct_l97_97956


namespace g_of_3_over_8_l97_97742

def g (x : ℝ) : ℝ := sorry

theorem g_of_3_over_8 :
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g 0 = 0) ∧
    (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 1 → g x ≤ g y) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g (x / 4) = (g x) / 3) →
    g (3 / 8) = 2 / 9 :=
sorry

end g_of_3_over_8_l97_97742


namespace renovation_project_truck_load_l97_97910

theorem renovation_project_truck_load (sand : ℝ) (dirt : ℝ) (cement : ℝ)
  (h1 : sand = 0.17) (h2 : dirt = 0.33) (h3 : cement = 0.17) :
  sand + dirt + cement = 0.67 :=
by
  sorry

end renovation_project_truck_load_l97_97910


namespace range_of_k_l97_97829

theorem range_of_k (k : ℝ) :
  ∃ x : ℝ, k * x^2 - 2 * x - 1 = 0 ↔ k ≥ -1 :=
by
  sorry

end range_of_k_l97_97829


namespace intersection_of_M_and_N_l97_97974

-- Define the sets M and N
def M := {-1, 1}
def N := {-1, 0, 2}

-- The theorem to prove
theorem intersection_of_M_and_N : M ∩ N = {-1} :=
by
  sorry

end intersection_of_M_and_N_l97_97974


namespace new_lamp_height_is_correct_l97_97312

-- Define the height of the old lamp
def old_lamp_height : ℝ := 1

-- Define the additional height of the new lamp
def additional_height : ℝ := 1.33

-- Proof statement
theorem new_lamp_height_is_correct :
  old_lamp_height + additional_height = 2.33 :=
sorry

end new_lamp_height_is_correct_l97_97312


namespace josie_remaining_money_l97_97994

-- Conditions
def initial_amount : ℕ := 50
def cassette_tape_cost : ℕ := 9
def headphone_cost : ℕ := 25

-- Proof statement
theorem josie_remaining_money : initial_amount - (2 * cassette_tape_cost + headphone_cost) = 7 :=
by
  sorry

end josie_remaining_money_l97_97994


namespace sum_of_cubes_ratio_l97_97597

theorem sum_of_cubes_ratio (a b c d e f : ℝ) 
  (h1 : a + b + c = 0) (h2 : d + e + f = 0) :
  (a^3 + b^3 + c^3) / (d^3 + e^3 + f^3) = (a * b * c) / (d * e * f) := 
by 
  sorry

end sum_of_cubes_ratio_l97_97597


namespace non_congruent_rectangles_count_l97_97143

theorem non_congruent_rectangles_count :
  let grid_width := 6
  let grid_height := 4
  let axis_aligned_rectangles := (grid_width.choose 2) * (grid_height.choose 2)
  let squares_1x1 := (grid_width - 1) * (grid_height - 1)
  let squares_2x2 := (grid_width - 2) * (grid_height - 2)
  let non_congruent_rectangles := axis_aligned_rectangles - (squares_1x1 + squares_2x2)
  non_congruent_rectangles = 67 := 
by {
  sorry
}

end non_congruent_rectangles_count_l97_97143


namespace coefficient_of_x_squared_l97_97834

open BigOperators

theorem coefficient_of_x_squared :
  (∑ k in Finset.range (5 + 1), (Nat.choose 5 k) * (1 : ℤ)^(5 - k) * (2 : ℤ)^k * (x : ℤ)^k).coeff 2 = 40 :=
by
  sorry

end coefficient_of_x_squared_l97_97834


namespace total_kids_in_Lawrence_l97_97672

theorem total_kids_in_Lawrence (stay_home kids_camp total_kids : ℕ) (h1 : stay_home = 907611) (h2 : kids_camp = 455682) (h3 : total_kids = stay_home + kids_camp) : total_kids = 1363293 :=
by
  sorry

end total_kids_in_Lawrence_l97_97672


namespace abs_eq_sets_l97_97549

theorem abs_eq_sets (x : ℝ) : 
  (|x - 25| + |x - 15| = |2 * x - 40|) → (x ≤ 15 ∨ x ≥ 25) :=
by
  sorry

end abs_eq_sets_l97_97549


namespace perfect_square_trinomial_l97_97695

theorem perfect_square_trinomial (k : ℤ) : (∃ a : ℤ, (x : ℤ) → x^2 - k * x + 9 = (x - a)^2) → (k = 6 ∨ k = -6) :=
sorry

end perfect_square_trinomial_l97_97695


namespace arccos_pi_over_3_l97_97932

theorem arccos_pi_over_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_pi_over_3_l97_97932


namespace pair_of_operations_equal_l97_97886

theorem pair_of_operations_equal :
  (-3) ^ 3 = -(3 ^ 3) ∧
  (¬((-2) ^ 4 = -(2 ^ 4))) ∧
  (¬((3 / 2) ^ 2 = (2 / 3) ^ 2)) ∧
  (¬(2 ^ 3 = 3 ^ 2)) :=
by 
  sorry

end pair_of_operations_equal_l97_97886


namespace squares_end_with_76_l97_97798

noncomputable def validNumbers : List ℕ := [24, 26, 74, 76]

theorem squares_end_with_76 (x : ℕ) (h₁ : x % 10 = 4 ∨ x % 10 = 6) 
    (h₂ : (x * x) % 100 = 76) : x ∈ validNumbers := by
  sorry

end squares_end_with_76_l97_97798


namespace lees_friend_initial_money_l97_97449

theorem lees_friend_initial_money (lee_initial_money friend_initial_money total_cost change : ℕ) 
  (h1 : lee_initial_money = 10) 
  (h2 : total_cost = 15) 
  (h3 : change = 3) 
  (h4 : (lee_initial_money + friend_initial_money) - total_cost = change) : 
  friend_initial_money = 8 := by
  sorry

end lees_friend_initial_money_l97_97449


namespace maximal_cards_taken_l97_97747

theorem maximal_cards_taken (cards : Finset ℕ) (h_cards : ∀ n, n ∈ cards ↔ 1 ≤ n ∧ n ≤ 100)
                            (andriy_cards nick_cards : Finset ℕ)
                            (h_card_count : andriy_cards.card = nick_cards.card)
                            (h_card_relation : ∀ n, n ∈ andriy_cards → (2 * n + 2) ∈ nick_cards) :
                            andriy_cards.card + nick_cards.card ≤ 50 := 
sorry

end maximal_cards_taken_l97_97747


namespace expression_equals_two_l97_97668

noncomputable def math_expression : ℝ :=
  27^(1/3) + Real.log 4 + 2 * Real.log 5 - Real.exp (Real.log 3)

theorem expression_equals_two : math_expression = 2 := by
  sorry

end expression_equals_two_l97_97668


namespace ways_to_place_people_into_groups_l97_97020

theorem ways_to_place_people_into_groups :
  let men := 4
  let women := 5
  ∃ (groups : Nat), groups = 2 ∧
  ∀ (g : Nat → (Fin 3 → (Bool → Nat → Nat))),
    (∀ i, i < group_counts → ∃ m w, g i m w < people ∧ g i m (1 - w) < people ∧ m + 1 - w + (1 - m) + w = 3) →
    let groups : List (List (Fin 2)) := [
      [(Fin.mk 1 dec_trivial, Fin.mk 2 dec_trivial)],
      [(Fin.mk 1 dec_trivial, Fin.mk 2 dec_trivial)]
    ] in
    g.mk 1 dec_trivial * g.mk 2 dec_trivial = 360 :=
sorry

end ways_to_place_people_into_groups_l97_97020


namespace jenna_average_speed_l97_97446

theorem jenna_average_speed (total_distance : ℕ) (total_time : ℕ) 
(first_segment_speed : ℕ) (second_segment_speed : ℕ) (third_segment_speed : ℕ) : 
  total_distance = 150 ∧ total_time = 2 ∧ first_segment_speed = 50 ∧ 
  second_segment_speed = 70 → third_segment_speed = 105 := 
by 
  intros h
  sorry

end jenna_average_speed_l97_97446


namespace beetles_eaten_per_day_l97_97248
-- Import the Mathlib library

-- Declare the conditions as constants
def bird_eats_beetles_per_day : Nat := 12
def snake_eats_birds_per_day : Nat := 3
def jaguar_eats_snakes_per_day : Nat := 5
def number_of_jaguars : Nat := 6

-- Define the theorem and provide the expected proof
theorem beetles_eaten_per_day :
  12 * (3 * (5 * 6)) = 1080 := by
  sorry

end beetles_eaten_per_day_l97_97248


namespace inequality_solution_l97_97060

theorem inequality_solution :
  {x : ℝ | ((x - 1) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7)) > 0} = 
  {x : ℝ | (1 < x ∧ x < 2) ∨ (4 < x ∧ x < 5) ∨ (6 < x ∧ x < 7)} :=
sorry

end inequality_solution_l97_97060


namespace total_ice_cubes_correct_l97_97919

/-- Each tray holds 48 ice cubes -/
def cubes_per_tray : Nat := 48

/-- Billy has 24 trays -/
def number_of_trays : Nat := 24

/-- Calculate the total number of ice cubes -/
def total_ice_cubes (cubes_per_tray : Nat) (number_of_trays : Nat) : Nat :=
  cubes_per_tray * number_of_trays

/-- Proof that the total number of ice cubes is 1152 given the conditions -/
theorem total_ice_cubes_correct : total_ice_cubes cubes_per_tray number_of_trays = 1152 := by
  /- Here we state the main theorem, but we leave the proof as sorry per the instructions -/
  sorry

end total_ice_cubes_correct_l97_97919


namespace find_solutions_l97_97800

-- A predicate for the given equation
def satisfies_equation (x y : ℕ) : Prop := 
  (1 / (x : ℚ) + 1 / (y : ℚ)) = 1 / 4

-- Define the set of solutions
def solutions : Set (ℕ × ℕ) := 
  {(5, 20), (6, 12), (8, 8), (12, 6), (20, 5)}

-- The goal is to prove that these solutions are the only ones
theorem find_solutions : 
  {p : ℕ × ℕ | satisfies_equation p.1 p.2 ∧ p.1 > 0 ∧ p.2 > 0} = solutions := 
by 
  sorry

end find_solutions_l97_97800


namespace quadratic_expression_value_l97_97187

theorem quadratic_expression_value (a b : ℝ) (h₁ : a ≠ 0) (h₂ : a + b - 1 = 1) : (1 - a - b) = -1 :=
sorry

end quadratic_expression_value_l97_97187


namespace find_initial_length_of_cloth_l97_97980

noncomputable def initial_length_of_cloth : ℝ :=
  let work_rate_of_8_men := 36 / 0.75
  work_rate_of_8_men

theorem find_initial_length_of_cloth (L : ℝ) (h1 : (4:ℝ) * 2 = L / ((4:ℝ) / (L / 8)))
    (h2 : (8:ℝ) / L = 36 / 0.75) : L = 48 :=
by
  sorry

end find_initial_length_of_cloth_l97_97980


namespace union_of_M_and_N_l97_97973

def M : Set Int := {-1, 0, 1}
def N : Set Int := {0, 1, 2}

theorem union_of_M_and_N :
  M ∪ N = {-1, 0, 1, 2} :=
by
  sorry

end union_of_M_and_N_l97_97973


namespace translated_vector_ab_l97_97563

-- Define points A and B, and vector a
def A : ℝ × ℝ := (3, 7)
def B : ℝ × ℝ := (5, 2)
def a : ℝ × ℝ := (1, 2)

-- Define the vector AB
def vectorAB : ℝ × ℝ :=
  let (Ax, Ay) := A
  let (Bx, By) := B
  (Bx - Ax, By - Ay)

-- Prove that after translating vector AB by vector a, the result remains (2, -5)
theorem translated_vector_ab :
  vectorAB = (2, -5) := by
  sorry

end translated_vector_ab_l97_97563


namespace range_of_a_l97_97064

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -x^2 + 2 * x + 3 ≤ a^2 - 3 * a) ↔ (a ≤ -1 ∨ a ≥ 4) := by
  sorry

end range_of_a_l97_97064


namespace largest_result_l97_97649

theorem largest_result :
  let A := (1 / 17 - 1 / 19) / 20
  let B := (1 / 15 - 1 / 21) / 60
  let C := (1 / 13 - 1 / 23) / 100
  let D := (1 / 11 - 1 / 25) / 140
  D > A ∧ D > B ∧ D > C := by
  sorry

end largest_result_l97_97649


namespace each_charity_gets_45_l97_97841

-- Define the conditions
def dozen := 12
def total_cookies := 6 * dozen
def price_per_cookie := 1.5
def cost_per_cookie := 0.25
def total_revenue := total_cookies * price_per_cookie
def total_cost := total_cookies * cost_per_cookie
def total_profit := total_revenue - total_cost

-- Define the expected outcome
def expected_each_charity_gets := 45

-- The theorem to prove
theorem each_charity_gets_45 :
  total_profit / 2 = expected_each_charity_gets :=
by
  sorry

end each_charity_gets_45_l97_97841


namespace total_polled_votes_correct_l97_97918

variable (V : ℕ) -- Valid votes

-- Condition: One candidate got 30% of the valid votes
variable (C1_votes : ℕ) (C2_votes : ℕ)
variable (H1 : C1_votes = (3 * V) / 10)

-- Condition: The other candidate won by 5000 votes
variable (H2 : C2_votes = C1_votes + 5000)

-- Condition: One candidate got 70% of the valid votes
variable (H3 : C2_votes = (7 * V) / 10)

-- Condition: 100 votes were invalid
variable (invalid_votes : ℕ := 100)

-- Total polled votes (valid + invalid)
def total_polled_votes := V + invalid_votes

theorem total_polled_votes_correct 
  (V : ℕ) 
  (H1 : C1_votes = (3 * V) / 10) 
  (H2 : C2_votes = C1_votes + 5000) 
  (H3 : C2_votes = (7 * V) / 10) 
  (invalid_votes : ℕ := 100) : 
  total_polled_votes V = 12600 :=
by
  -- The steps of the proof are omitted
  sorry

end total_polled_votes_correct_l97_97918


namespace inequality_proof_l97_97713

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (1 / (x^2 + y^2)) + (1 / x^2) + (1 / y^2) ≥ 10 / (x + y)^2 :=
sorry

end inequality_proof_l97_97713


namespace inequality_proof_l97_97506

theorem inequality_proof 
(x1 x2 y1 y2 z1 z2 : ℝ) 
(hx1 : x1 > 0) 
(hx2 : x2 > 0) 
(hineq1 : x1 * y1 - z1^2 > 0) 
(hineq2 : x2 * y2 - z2^2 > 0)
: 
  8 / ((x1 + x2)*(y1 + y2) - (z1 + z2)^2) <= 
  1 / (x1 * y1 - z1^2) + 
  1 / (x2 * y2 - z2^2) := 
sorry

end inequality_proof_l97_97506


namespace constant_term_exists_l97_97983

theorem constant_term_exists:
  ∃ (n : ℕ), 2 ≤ n ∧ n ≤ 10 ∧ 
  (∃ r : ℕ, n = 3 * r) ∧ (∃ k : ℕ, n = 2 * k) ∧ 
  n = 6 :=
sorry

end constant_term_exists_l97_97983


namespace distance_to_y_axis_parabola_midpoint_l97_97565

noncomputable def distance_from_midpoint_to_y_axis (x1 x2 : ℝ) : ℝ :=
  (x1 + x2) / 2

theorem distance_to_y_axis_parabola_midpoint :
  ∀ (x1 y1 x2 y2 : ℝ), y1^2 = x1 → y2^2 = x2 → 
  abs (x1 + 1 / 4) + abs (x2 + 1 / 4) = 3 →
  abs (distance_from_midpoint_to_y_axis x1 x2) = 5 / 4 :=
by
  intros x1 y1 x2 y2 h1 h2 h3
  sorry

end distance_to_y_axis_parabola_midpoint_l97_97565


namespace chris_raisins_nuts_l97_97530

theorem chris_raisins_nuts (R N x : ℝ) 
  (hN : N = 4 * R) 
  (hxR : x * R = 0.15789473684210525 * (x * R + 4 * N)) :
  x = 3 :=
by
  sorry

end chris_raisins_nuts_l97_97530


namespace geom_sum_3m_l97_97687

variable (a_n : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (m : ℕ)

axiom geom_sum_m : S m = 10
axiom geom_sum_2m : S (2 * m) = 30

theorem geom_sum_3m : S (3 * m) = 70 :=
by
  sorry

end geom_sum_3m_l97_97687


namespace find_ratio_l97_97290

variable {d : ℕ}
variable {a : ℕ → ℝ}

-- Conditions: arithmetic sequence with non-zero common difference, and geometric sequence terms
axiom arithmetic_sequence (n : ℕ) : a n = a 1 + (n - 1) * d
axiom non_zero_d : d ≠ 0
axiom geometric_sequence : (a 1 + 2*d)^2 = a 1 * (a 1 + 8*d)

-- Theorem to prove the desired ratio
theorem find_ratio : (a 1 + a 3 + a 9) / (a 2 + a 4 + a 10) = 13 / 16 :=
sorry

end find_ratio_l97_97290


namespace third_bowler_points_162_l97_97641

variable (x : ℕ)

def total_score (x : ℕ) : Prop :=
  let first_bowler_points := x
  let second_bowler_points := 3 * x
  let third_bowler_points := x
  first_bowler_points + second_bowler_points + third_bowler_points = 810

theorem third_bowler_points_162 (x : ℕ) (h : total_score x) : x = 162 := by
  sorry

end third_bowler_points_162_l97_97641


namespace find_a_l97_97815

-- Define the function f based on the given conditions
noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then 2^x - a * x else -2^(-x) - a * x

-- Define the fact that f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = -f (-x)

-- State the main theorem that needs to be proven
theorem find_a (a : ℝ) :
  (is_odd_function (f a)) ∧ (f a 2 = 2) → a = -9 / 8 :=
by
  sorry

end find_a_l97_97815


namespace selena_book_pages_l97_97055

variable (S : ℕ)
variable (H : ℕ)

theorem selena_book_pages (cond1 : H = S / 2 - 20) (cond2 : H = 180) : S = 400 :=
by
  sorry

end selena_book_pages_l97_97055


namespace x_squared_plus_y_squared_l97_97978

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 17) (h2 : x * y = 6) : x^2 + y^2 = 301 :=
by sorry

end x_squared_plus_y_squared_l97_97978


namespace rowing_time_ratio_l97_97198

def V_b : ℕ := 57
def V_s : ℕ := 19
def V_up : ℕ := V_b - V_s
def V_down : ℕ := V_b + V_s

theorem rowing_time_ratio :
  ∀ (T_up T_down : ℕ), V_up * T_up = V_down * T_down → T_up = 2 * T_down :=
by
  intros T_up T_down h
  sorry

end rowing_time_ratio_l97_97198


namespace consecutive_odd_integers_l97_97068

theorem consecutive_odd_integers (n : ℤ) (h : (n - 2) + (n + 2) = 130) : n = 65 :=
sorry

end consecutive_odd_integers_l97_97068


namespace polygon_diagonals_with_one_non_connecting_vertex_l97_97384

-- Define the number of sides in the polygon
def num_sides : ℕ := 17

-- Define the formula to calculate the number of diagonals in a polygon
def total_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- Define the number of non-connecting vertex to any diagonal
def non_connected_diagonals (n : ℕ) : ℕ :=
  n - 3

-- The theorem to state and prove
theorem polygon_diagonals_with_one_non_connecting_vertex :
  total_diagonals num_sides - non_connected_diagonals num_sides = 105 :=
by
  -- The formal proof would go here
  sorry

end polygon_diagonals_with_one_non_connecting_vertex_l97_97384


namespace hyperbola_eq_l97_97135

/-- Given a hyperbola with center at the origin, 
    one focus at (-√5, 0), and a point P on the hyperbola such that 
    the midpoint of segment PF₁ has coordinates (0, 2), 
    then the equation of the hyperbola is x² - y²/4 = 1. --/
theorem hyperbola_eq (x y : ℝ) (P F1 : ℝ × ℝ) 
  (hF1 : F1 = (-Real.sqrt 5, 0)) 
  (hMidPoint : (P.1 + -Real.sqrt 5) / 2 = 0 ∧ (P.2 + 0) / 2 = 2) 
  : x^2 - y^2 / 4 = 1 := 
sorry

end hyperbola_eq_l97_97135


namespace maximize_operation_l97_97237

-- Definitions from the conditions
def is_three_digit_integer (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

-- The proof statement
theorem maximize_operation : ∃ n, is_three_digit_integer n ∧ (∀ m, is_three_digit_integer m → 3 * (300 - m) ≤ 600) :=
by {
  -- Placeholder for the actual proof
  sorry
}

end maximize_operation_l97_97237


namespace triangles_with_positive_area_l97_97300

theorem triangles_with_positive_area (x y : ℕ) (h₁ : 1 ≤ x ∧ x ≤ 5) (h₂ : 1 ≤ y ∧ y ≤ 3) : 
    ∃ (n : ℕ), n = 420 := 
sorry

end triangles_with_positive_area_l97_97300


namespace consecutive_integers_sum_l97_97193

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 380) : x + (x + 1) = 39 := by
  sorry

end consecutive_integers_sum_l97_97193


namespace group_count_4_men_5_women_l97_97026

theorem group_count_4_men_5_women : 
  let men := 4
  let women := 5
  let groups := List.replicate 3 (3, true)
  ∃ (m_w_combinations : List (ℕ × ℕ)),
    m_w_combinations = [(1, 2), (2, 1)] ∧
    ((men.choose m_w_combinations.head.fst * women.choose m_w_combinations.head.snd) * (men - m_w_combinations.head.fst).choose m_w_combinations.tail.head.fst * (women - m_w_combinations.head.snd).choose m_w_combinations.tail.head.snd) = 360 :=
by
  sorry

end group_count_4_men_5_women_l97_97026


namespace Jungkook_fewest_erasers_l97_97165

-- Define the number of erasers each person has.
def Jungkook_erasers : ℕ := 6
def Jimin_erasers : ℕ := Jungkook_erasers + 4
def Seokjin_erasers : ℕ := Jimin_erasers - 3

-- Prove that Jungkook has the fewest erasers.
theorem Jungkook_fewest_erasers : Jungkook_erasers < Jimin_erasers ∧ Jungkook_erasers < Seokjin_erasers :=
by
  -- Proof goes here
  sorry

end Jungkook_fewest_erasers_l97_97165


namespace primitive_root_coprime_distinct_residues_noncoprime_non_distinct_residues_l97_97509

-- Define Part (a)
theorem primitive_root_coprime_distinct_residues (m k : ℕ) (h: Nat.gcd m k = 1) :
  ∃ (a : Fin m → ℕ) (b : Fin k → ℕ),
    ∀ i j s t, (i ≠ s ∨ j ≠ t) → (a i * b j) % (m * k) ≠ (a s * b t) % (m * k) :=
sorry

-- Define Part (b)
theorem noncoprime_non_distinct_residues (m k : ℕ) (h: Nat.gcd m k > 1) :
  ∀ (a : Fin m → ℕ) (b : Fin k → ℕ),
    ∃ i j x t, (i ≠ x ∨ j ≠ t) ∧ (a i * b j) % (m * k) = (a x * b t) % (m * k) :=
sorry

end primitive_root_coprime_distinct_residues_noncoprime_non_distinct_residues_l97_97509


namespace solve_system_eq_l97_97183

theorem solve_system_eq (x y : ℝ) (h1 : 2 * x - y = 3) (h2 : 3 * x + 2 * y = 8) :
  x = 2 ∧ y = 1 :=
by
  sorry

end solve_system_eq_l97_97183


namespace lifespan_histogram_l97_97884

theorem lifespan_histogram :
  (class_interval = 20) →
  (height_vertical_axis_60_80 = 0.03) →
  (total_people = 1000) →
  (number_of_people_60_80 = 600) :=
by
  intro class_interval height_vertical_axis_60_80 total_people
  -- Perform necessary calculations (omitting actual proof as per instructions)
  sorry

end lifespan_histogram_l97_97884


namespace original_number_l97_97149

theorem original_number (x : ℕ) : x * 16 = 3408 → x = 213 := by
  intro h
  sorry

end original_number_l97_97149


namespace inversely_proportional_value_l97_97471

theorem inversely_proportional_value (a b k : ℝ) (h1 : a * b = k) (h2 : a = 40) (h3 : b = 8) :
  ∃ a' : ℝ, a' * 10 = k ∧ a' = 32 :=
by {
  use 32,
  sorry
}

end inversely_proportional_value_l97_97471


namespace hyperbola_foci_coordinates_l97_97859

theorem hyperbola_foci_coordinates :
  (a^2 = 7) → (b^2 = 3) → (c^2 = a^2 + b^2) → (c = Real.sqrt c^2) →
  ∃ (x y : ℝ), (x = Real.sqrt 10 ∧ y = 0) ∨ (x = -Real.sqrt 10 ∧ y = 0) :=
by
  intros a2_eq b2_eq c2_eq c_eq
  have h1 : c2 = 10 := by rw [a2_eq, b2_eq, add_comm]
  have h2 : c = Real.sqrt 10 := by rw [h1, Real.sq_sqrt (show 0 ≤ 10 by norm_num)]
  use (Real.sqrt 10)
  use 0
  use (-Real.sqrt 10)
  use 0
  sorry

end hyperbola_foci_coordinates_l97_97859


namespace painters_workdays_l97_97314

theorem painters_workdays (five_painters_days : ℝ) (four_painters_days : ℝ) : 
  (5 * five_painters_days = 9) → (4 * four_painters_days = 9) → (four_painters_days = 2.25) :=
by
  intros h1 h2
  sorry

end painters_workdays_l97_97314


namespace circle_area_l97_97311

theorem circle_area (x y : ℝ) :
  (3 * x^2 + 3 * y^2 - 9 * x + 12 * y + 27 = 0) →
  (π * ((1 / 2) * (1 / 2)) = (π / 4)) := 
by
  intro h
  sorry

end circle_area_l97_97311


namespace abs_sum_example_l97_97767

theorem abs_sum_example : |(-8 : ℤ)| + |(-4 : ℤ)| = 12 := by
  sorry

end abs_sum_example_l97_97767


namespace polynomial_at_five_l97_97455

theorem polynomial_at_five (P : ℝ → ℝ) 
  (hP_degree : ∃ (a b c d : ℝ), ∀ x : ℝ, P x = a*x^3 + b*x^2 + c*x + d)
  (hP1 : P 1 = 1 / 3)
  (hP2 : P 2 = 1 / 7)
  (hP3 : P 3 = 1 / 13)
  (hP4 : P 4 = 1 / 21) :
  P 5 = -3 / 91 :=
sorry

end polynomial_at_five_l97_97455


namespace arccos_sin_2_equals_l97_97663

theorem arccos_sin_2_equals : Real.arccos (Real.sin 2) = 2 - Real.pi / 2 := by
  sorry

end arccos_sin_2_equals_l97_97663


namespace fraction_to_decimal_l97_97401

theorem fraction_to_decimal :
  (11:ℚ) / 16 = 0.6875 :=
by
  sorry

end fraction_to_decimal_l97_97401


namespace eval_g_six_times_at_2_l97_97169

def g (x : ℝ) : ℝ := x^2 - 4 * x + 4

theorem eval_g_six_times_at_2 : g (g (g (g (g (g 2))))) = 4 := sorry

end eval_g_six_times_at_2_l97_97169


namespace find_a7_l97_97156

variable (a : ℕ → ℝ)

def arithmetic_sequence (d : ℝ) (a1 : ℝ) :=
  ∀ n, a n = a1 + (n - 1) * d

theorem find_a7
  (a : ℕ → ℝ)
  (d : ℝ)
  (a1 : ℝ)
  (h_arith : arithmetic_sequence a d a1)
  (h_a3 : a 3 = 7)
  (h_a5 : a 5 = 13):
  a 7 = 19 :=
by
  sorry

end find_a7_l97_97156


namespace find_fraction_abs_l97_97008

-- Define the conditions and the main proof problem
theorem find_fraction_abs (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 + y^2 = 5 * x * y) :
  abs ((x + y) / (x - y)) = Real.sqrt ((7 : ℝ) / 3) :=
by
  sorry

end find_fraction_abs_l97_97008


namespace geometric_sequence_8th_term_l97_97244

theorem geometric_sequence_8th_term (a : ℚ) (r : ℚ) (n : ℕ) (h_a : a = 27) (h_r : r = 2/3) (h_n : n = 8) :
  a * r^(n-1) = 128 / 81 :=
by
  rw [h_a, h_r, h_n]
  sorry

end geometric_sequence_8th_term_l97_97244
