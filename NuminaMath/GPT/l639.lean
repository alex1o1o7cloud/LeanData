import Mathlib

namespace tank_cost_minimization_l639_63908

def volume := 4800
def depth := 3
def cost_per_sqm_bottom := 150
def cost_per_sqm_walls := 120

theorem tank_cost_minimization (x : ℝ) 
  (S1 : ℝ := volume / depth)
  (S2 : ℝ := 6 * (x + (S1 / x)))
  (cost := cost_per_sqm_bottom * S1 + cost_per_sqm_walls * S2) :
  (x = 40) → cost = 297600 :=
sorry

end tank_cost_minimization_l639_63908


namespace sum_of_terms_7_8_9_l639_63992

namespace ArithmeticSequence

-- Define the sequence and its properties
variables (a : ℕ → ℤ) (S : ℕ → ℤ)

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  n * a 0 + n * (n - 1) / 2 * (a 1 - a 0)

def condition3 (S : ℕ → ℤ) : Prop :=
  S 3 = 9

def condition5 (S : ℕ → ℤ) : Prop :=
  S 5 = 30

-- Main statement to prove
theorem sum_of_terms_7_8_9 :
  is_arithmetic_sequence a →
  (∀ n, S n = sum_first_n_terms a n) →
  condition3 S →
  condition5 S →
  a 7 + a 8 + a 9 = 63 :=
by
  sorry

end ArithmeticSequence

end sum_of_terms_7_8_9_l639_63992


namespace rearrange_marked_squares_l639_63931

theorem rearrange_marked_squares (n k : ℕ) (h : n > 1) (h' : k ≤ n + 1) :
  ∃ (f g : Fin n → Fin n), true := sorry

end rearrange_marked_squares_l639_63931


namespace max_constant_term_l639_63941

theorem max_constant_term (c : ℝ) : 
  (∀ x : ℝ, (x^2 - 6 * x + c = 0 → (x^2 - 6 * x + c ≥ 0))) → c ≤ 9 :=
by sorry

end max_constant_term_l639_63941


namespace range_of_a_l639_63928

theorem range_of_a (x y z a : ℝ) 
    (h1 : x > 0) 
    (h2 : y > 0) 
    (h3 : z > 0) 
    (h4 : x + y + z = 1) 
    (h5 : a / (x * y * z) = 1 / x + 1 / y + 1 / z - 2) : 
    0 < a ∧ a ≤ 7 / 27 := 
  sorry

end range_of_a_l639_63928


namespace probability_of_queen_is_correct_l639_63991

def deck_size : ℕ := 52
def queen_count : ℕ := 4

-- This definition denotes the probability calculation.
def probability_drawing_queen : ℚ := queen_count / deck_size

theorem probability_of_queen_is_correct :
  probability_drawing_queen = 1 / 13 :=
by
  sorry

end probability_of_queen_is_correct_l639_63991


namespace good_numbers_identification_l639_63939

def is_good_number (n : ℕ) : Prop :=
  ∃ (a : Fin n → Fin n), 
    (∀ k : Fin n, ∃ m : ℕ, k.val + a k = m * m)

theorem good_numbers_identification : 
  { n : ℕ | ¬is_good_number n } = {1, 2, 4, 6, 7, 9, 11} :=
  sorry

end good_numbers_identification_l639_63939


namespace moles_of_C2H5Cl_l639_63978

-- Define chemical entities as types
structure Molecule where
  name : String

-- Declare molecules involved in the reaction
def C2H6 := Molecule.mk "C2H6"
def Cl2  := Molecule.mk "Cl2"
def C2H5Cl := Molecule.mk "C2H5Cl"
def HCl := Molecule.mk "HCl"

-- Define number of moles as a non-negative integer
def moles (m : Molecule) : ℕ := sorry

-- Conditions
axiom initial_moles_C2H6 : moles C2H6 = 3
axiom initial_moles_Cl2 : moles Cl2 = 3

-- Balanced reaction equation: 1 mole of C2H6 reacts with 1 mole of Cl2 to form 1 mole of C2H5Cl
axiom reaction_stoichiometry : ∀ (x : ℕ), moles C2H6 = x → moles Cl2 = x → moles C2H5Cl = x

-- Proof problem
theorem moles_of_C2H5Cl : moles C2H5Cl = 3 := by
  apply reaction_stoichiometry
  exact initial_moles_C2H6
  exact initial_moles_Cl2

end moles_of_C2H5Cl_l639_63978


namespace stack_height_difference_l639_63952

theorem stack_height_difference :
  ∃ S : ℕ,
    (7 + S + (S - 6) + (S + 4) + 2 * S = 55) ∧ (S - 7 = 3) := 
by 
  sorry

end stack_height_difference_l639_63952


namespace even_function_a_value_monotonicity_on_neg_infinity_l639_63998

noncomputable def f (x a : ℝ) : ℝ := ((x + 1) * (x + a)) / (x^2)

-- (1) Proving f(x) is even implies a = -1
theorem even_function_a_value (a : ℝ) : (∀ x : ℝ, f x a = f (-x) a) ↔ a = -1 :=
by
  sorry

-- (2) Proving monotonicity on (-∞, 0) for f(x) with a = -1
theorem monotonicity_on_neg_infinity (x₁ x₂ : ℝ) (h₁ : x₁ < x₂) (h₂ : x₂ < 0) :
  (f x₁ (-1) > f x₂ (-1)) :=
by
  sorry

end even_function_a_value_monotonicity_on_neg_infinity_l639_63998


namespace crayons_loss_difference_l639_63989

theorem crayons_loss_difference (crayons_given crayons_lost : ℕ) 
  (h_given : crayons_given = 90) 
  (h_lost : crayons_lost = 412) : 
  crayons_lost - crayons_given = 322 :=
by
  sorry

end crayons_loss_difference_l639_63989


namespace present_age_of_son_l639_63968

variable (S M : ℝ)

-- Conditions
def condition1 : Prop := M = S + 35
def condition2 : Prop := M + 5 = 3 * (S + 5)

-- Proof Problem
theorem present_age_of_son
  (h1 : condition1 S M)
  (h2 : condition2 S M) :
  S = 12.5 :=
sorry

end present_age_of_son_l639_63968


namespace union_sets_l639_63983

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem union_sets : M ∪ N = {-1, 0, 1, 2} :=
by
  sorry

end union_sets_l639_63983


namespace cos_triple_angle_l639_63933

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (3 * θ) = -23 / 27 := by
  sorry

end cos_triple_angle_l639_63933


namespace XT_value_l639_63973

noncomputable def AB := 15
noncomputable def BC := 20
noncomputable def height_P := 30
noncomputable def volume_ratio := 9

theorem XT_value 
  (AB BC height_P : ℕ)
  (volume_ratio : ℕ)
  (h1 : AB = 15)
  (h2 : BC = 20)
  (h3 : height_P = 30)
  (h4 : volume_ratio = 9) : 
  ∃ (m n : ℕ), m + n = 97 ∧ m.gcd n = 1 :=
by sorry

end XT_value_l639_63973


namespace possible_value_of_n_l639_63924

open Nat

def coefficient_is_rational (n r : ℕ) : Prop :=
  (n - r) % 2 = 0 ∧ r % 3 = 0

theorem possible_value_of_n :
  ∃ n : ℕ, n > 0 ∧ (∀ r : ℕ, r ≤ n → coefficient_is_rational n r) ↔ n = 9 :=
sorry

end possible_value_of_n_l639_63924


namespace pine_cone_weight_on_roof_l639_63920

theorem pine_cone_weight_on_roof
  (num_trees : ℕ) (cones_per_tree : ℕ) (percentage_on_roof : ℝ) (weight_per_cone : ℕ)
  (H1 : num_trees = 8)
  (H2 : cones_per_tree = 200)
  (H3 : percentage_on_roof = 0.30)
  (H4 : weight_per_cone = 4) :
  num_trees * cones_per_tree * percentage_on_roof * weight_per_cone = 1920 := by
  sorry

end pine_cone_weight_on_roof_l639_63920


namespace value_of_expression_l639_63912

open Real

theorem value_of_expression (α : ℝ) (h : 3 * sin α + cos α = 0) :
  1 / (cos α ^ 2 + sin (2 * α)) = 10 / 3 :=
by
  sorry

end value_of_expression_l639_63912


namespace triangle_angle_sum_l639_63919

theorem triangle_angle_sum (x : ℝ) :
    let angle1 : ℝ := 40
    let angle2 : ℝ := 4 * x
    let angle3 : ℝ := 3 * x
    angle1 + angle2 + angle3 = 180 -> x = 20 := 
sorry

end triangle_angle_sum_l639_63919


namespace find_expression_l639_63945

theorem find_expression (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = 3 * x + 2) : 
  ∀ x : ℤ, f x = 3 * x - 1 :=
sorry

end find_expression_l639_63945


namespace num_students_basketball_l639_63916

-- Definitions for conditions
def num_students_cricket : ℕ := 8
def num_students_both : ℕ := 5
def num_students_either : ℕ := 10

-- statement to be proven
theorem num_students_basketball : ∃ B : ℕ, B = 7 ∧ (num_students_either = B + num_students_cricket - num_students_both) := sorry

end num_students_basketball_l639_63916


namespace triangular_pyramid_height_l639_63914

noncomputable def pyramid_height (a b c h : ℝ) : Prop :=
  1 / h ^ 2 = 1 / a ^ 2 + 1 / b ^ 2 + 1 / c ^ 2

theorem triangular_pyramid_height {a b c h : ℝ} (h_gt_0 : h > 0) (a_gt_0 : a > 0) (b_gt_0 : b > 0) (c_gt_0 : c > 0) :
  pyramid_height a b c h := by
  sorry

end triangular_pyramid_height_l639_63914


namespace proof_x_y_3_l639_63917

noncomputable def prime (n : ℤ) : Prop := 2 <= n ∧ ∀ m : ℤ, 1 ≤ m → m < n → n % m ≠ 0

theorem proof_x_y_3 (x y : ℝ) (p q r : ℤ) (h1 : x - y = p) (hp : prime p) 
  (h2 : x^2 - y^2 = q) (hq : prime q)
  (h3 : x^3 - y^3 = r) (hr : prime r) : p = 3 :=
sorry

end proof_x_y_3_l639_63917


namespace prime_of_the_form_4x4_plus_1_l639_63951

theorem prime_of_the_form_4x4_plus_1 (x : ℤ) (p : ℤ) (h : 4 * x ^ 4 + 1 = p) (hp : Prime p) : p = 5 :=
sorry

end prime_of_the_form_4x4_plus_1_l639_63951


namespace root_expression_value_l639_63980

-- Define the root condition
def is_root (a : ℝ) : Prop := 2 * a^2 - 3 * a - 5 = 0

-- The main theorem statement
theorem root_expression_value {a : ℝ} (h : is_root a) : -4 * a^2 + 6 * a = -10 := by
  sorry

end root_expression_value_l639_63980


namespace part1_l639_63962

def p (m x : ℝ) := x^2 - 3*m*x + 2*m^2 ≤ 0
def q (x : ℝ) := (x + 2)^2 < 1

theorem part1 (x : ℝ) (m : ℝ) (hm : m = -2) : p m x ∧ q x ↔ -3 < x ∧ x ≤ -2 :=
by
  unfold p q
  sorry

end part1_l639_63962


namespace determinant_value_l639_63904

theorem determinant_value (t₁ t₂ : ℤ)
    (h₁ : t₁ = 2 * 3 + 3 * 5)
    (h₂ : t₂ = 5) :
    Matrix.det ![
      ![1, -1, t₁],
      ![0, 1, -1],
      ![-1, t₂, -6]
    ] = 14 := by
  rw [h₁, h₂]
  -- Actual proof would go here
  sorry

end determinant_value_l639_63904


namespace sarah_amount_l639_63954

theorem sarah_amount:
  ∀ (X : ℕ), (X + (X + 50) = 300) → X = 125 := by
  sorry

end sarah_amount_l639_63954


namespace animal_count_l639_63997

variable (H C D : Nat)

theorem animal_count :
  (H + C + D = 72) → 
  (2 * H + 4 * C + 2 * D = 212) → 
  (C = 34) → 
  (H + D = 38) :=
by
  intros h1 h2 hc
  sorry

end animal_count_l639_63997


namespace initial_amount_of_money_l639_63932

-- Define the conditions
def spent_on_sweets : ℝ := 35.25
def given_to_each_friend : ℝ := 25.20
def num_friends : ℕ := 2
def amount_left : ℝ := 114.85

-- Define the calculated amount given to friends
def total_given_to_friends : ℝ := given_to_each_friend * num_friends

-- State the theorem to prove the initial amount of money
theorem initial_amount_of_money :
  spent_on_sweets + total_given_to_friends + amount_left = 200.50 :=
by 
  -- proof goes here
  sorry

end initial_amount_of_money_l639_63932


namespace julia_age_correct_l639_63921

def julia_age_proof : Prop :=
  ∃ (j : ℚ) (m : ℚ), m = 15 * j ∧ m - j = 40 ∧ j = 20 / 7

theorem julia_age_correct : julia_age_proof :=
by
  sorry

end julia_age_correct_l639_63921


namespace average_price_of_towels_l639_63976

-- Definitions based on the conditions
def cost_of_three_towels := 3 * 100
def cost_of_five_towels := 5 * 150
def cost_of_two_towels := 550
def total_cost := cost_of_three_towels + cost_of_five_towels + cost_of_two_towels
def total_number_of_towels := 3 + 5 + 2
def average_price := total_cost / total_number_of_towels

-- The theorem statement
theorem average_price_of_towels :
  average_price = 160 :=
by
  sorry

end average_price_of_towels_l639_63976


namespace find_n_l639_63977

theorem find_n :
  ∃ n : ℤ, 3 ^ 3 - 7 = 4 ^ 2 + 2 + n ∧ n = 2 :=
by
  use 2
  sorry

end find_n_l639_63977


namespace problem_statement_l639_63975

variable {α : Type*} [LinearOrder α] [AddCommGroup α] [Nontrivial α]

def is_monotone_increasing (f : α → α) (s : Set α) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x ≤ f y

theorem problem_statement (f : ℝ → ℝ) (x1 x2 : ℝ)
  (h1 : ∀ x, f (-x) = -f (x + 4))
  (h2 : is_monotone_increasing f {x | x > 2})
  (hx1 : x1 < 2) (hx2 : 2 < x2) (h_sum : x1 + x2 < 4) :
  f (x1) + f (x2) < 0 :=
sorry

end problem_statement_l639_63975


namespace circle_tangent_l639_63911

theorem circle_tangent {m : ℝ} (h : ∃ (x y : ℝ), (x - 3)^2 + (y - 4)^2 = 25 - m ∧ x^2 + y^2 = 1) :
  m = 9 :=
sorry

end circle_tangent_l639_63911


namespace corn_growth_first_week_l639_63937

theorem corn_growth_first_week (x : ℝ) (h1 : x + 2*x + 8*x = 22) : x = 2 :=
by
  sorry

end corn_growth_first_week_l639_63937


namespace area_when_other_side_shortened_l639_63935

def original_width := 5
def original_length := 8
def target_area := 24
def shortened_amount := 2

theorem area_when_other_side_shortened :
  (original_width - shortened_amount) * original_length = target_area →
  original_width * (original_length - shortened_amount) = 30 :=
by
  intros h
  sorry

end area_when_other_side_shortened_l639_63935


namespace total_bill_is_89_l639_63990

-- Define the individual costs and quantities
def adult_meal_cost := 12
def child_meal_cost := 7
def fries_cost := 5
def drink_cost := 10

def num_adults := 4
def num_children := 3
def num_fries := 2
def num_drinks := 1

-- Calculate the total bill
def total_bill : Nat :=
  (num_adults * adult_meal_cost) + 
  (num_children * child_meal_cost) + 
  (num_fries * fries_cost) + 
  (num_drinks * drink_cost)

-- The proof statement
theorem total_bill_is_89 : total_bill = 89 := 
  by
  -- The proof will be provided here
  sorry

end total_bill_is_89_l639_63990


namespace tea_sales_revenue_l639_63926

theorem tea_sales_revenue (x : ℝ) (price_last_year price_this_year : ℝ) (yield_last_year yield_this_year : ℝ) (revenue_last_year revenue_this_year : ℝ) :
  price_this_year = 10 * price_last_year →
  yield_this_year = 198.6 →
  yield_last_year = 198.6 + 87.4 →
  revenue_this_year = 198.6 * price_this_year →
  revenue_last_year = yield_last_year * price_last_year →
  revenue_this_year = revenue_last_year + 8500 →
  revenue_this_year = 9930 := 
by
  sorry

end tea_sales_revenue_l639_63926


namespace x_sq_sub_y_sq_l639_63943

theorem x_sq_sub_y_sq (x y : ℝ) (h1 : x + y = 8) (h2 : x - y = 4) : x^2 - y^2 = 32 :=
by
  sorry

end x_sq_sub_y_sq_l639_63943


namespace largest_possible_green_cards_l639_63965

-- Definitions of conditions
variables (g y t : ℕ)

-- Defining the total number of cards t
def total_cards := g + y

-- Condition on maximum number of cards
def max_total_cards := total_cards g y ≤ 2209

-- Probability condition for drawing 3 same-color cards
def probability_condition := 
  g * (g - 1) * (g - 2) + y * (y - 1) * (y - 2) 
  = (1 : ℚ) / 3 * t * (t - 1) * (t - 2)

-- Proving the largest possible number of green cards
theorem largest_possible_green_cards
  (h1 : total_cards g y = t)
  (h2 : max_total_cards g y)
  (h3 : probability_condition g y t) :
  g ≤ 1092 :=
sorry

end largest_possible_green_cards_l639_63965


namespace unpaintedRegionArea_l639_63958

def boardWidth1 : ℝ := 5
def boardWidth2 : ℝ := 7
def angle : ℝ := 45

theorem unpaintedRegionArea
  (bw1 bw2 angle : ℝ)
  (h1 : bw1 = boardWidth1)
  (h2 : bw2 = boardWidth2)
  (h3 : angle = 45) :
  let base := bw2 * Real.sqrt 2
  let height := bw1
  let area := base * height
  area = 35 * Real.sqrt 2 :=
by
  sorry

end unpaintedRegionArea_l639_63958


namespace smallest_base_l639_63964

theorem smallest_base (b : ℕ) : (b^2 ≤ 80 ∧ 80 < b^3) → b = 5 := by
  sorry

end smallest_base_l639_63964


namespace unique_integer_solution_l639_63930

def is_point_in_circle (x y cx cy radius : ℝ) : Prop :=
  (x - cx)^2 + (y - cy)^2 ≤ radius^2

theorem unique_integer_solution : ∃! (x : ℤ), is_point_in_circle (2 * x) (-x) 4 6 8 := by
  sorry

end unique_integer_solution_l639_63930


namespace joyce_pencils_given_l639_63985

def original_pencils : ℕ := 51
def total_pencils_after : ℕ := 57

theorem joyce_pencils_given : total_pencils_after - original_pencils = 6 :=
by
  sorry

end joyce_pencils_given_l639_63985


namespace set_equality_l639_63996

theorem set_equality (A : Set ℕ) (h : {1} ∪ A = {1, 3, 5}) : 
  A = {1, 3, 5} ∨ A = {3, 5} :=
  sorry

end set_equality_l639_63996


namespace find_m_l639_63907

theorem find_m (x m : ℝ) :
  (2 * x + m) * (x - 3) = 2 * x^2 - 3 * m ∧ 
  (∀ c : ℝ, c * x = 0 → c = 0) → 
  m = 6 :=
by sorry

end find_m_l639_63907


namespace cuboid_edge_length_l639_63923

theorem cuboid_edge_length
  (x : ℝ)
  (h_surface_area : 2 * (4 * x + 24 + 6 * x) = 148) :
  x = 5 :=
by
  sorry

end cuboid_edge_length_l639_63923


namespace range_of_a_for_extreme_points_l639_63918

noncomputable def f (x a : ℝ) : ℝ := x * Real.exp x - a * Real.exp (2 * x)

theorem range_of_a_for_extreme_points :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
    ∀ a : ℝ, 0 < a ∧ a < (1 / 2) →
    (Real.exp x₁ * (x₁ + 1 - 2 * a * Real.exp x₁) = 0) ∧ 
    (Real.exp x₂ * (x₂ + 1 - 2 * a * Real.exp x₂) = 0)) ↔ 
  ∀ a : ℝ, 0 < a ∧ a < (1 / 2) :=
sorry

end range_of_a_for_extreme_points_l639_63918


namespace carson_pumps_needed_l639_63950

theorem carson_pumps_needed 
  (full_tire_capacity : ℕ) (flat_tires_count : ℕ) 
  (full_percentage_tire_1 : ℚ) (full_percentage_tire_2 : ℚ)
  (air_per_pump : ℕ) : 
  flat_tires_count = 2 →
  full_tire_capacity = 500 →
  full_percentage_tire_1 = 0.40 →
  full_percentage_tire_2 = 0.70 →
  air_per_pump = 50 →
  let needed_air_flat_tires := flat_tires_count * full_tire_capacity
  let needed_air_tire_1 := (1 - full_percentage_tire_1) * full_tire_capacity
  let needed_air_tire_2 := (1 - full_percentage_tire_2) * full_tire_capacity
  let total_needed_air := needed_air_flat_tires + needed_air_tire_1 + needed_air_tire_2
  let pumps_needed := total_needed_air / air_per_pump
  pumps_needed = 29 := 
by
  intros
  sorry

end carson_pumps_needed_l639_63950


namespace dilation_image_l639_63936

theorem dilation_image 
  (z z₀ : ℂ) (k : ℝ) 
  (hz : z = -2 + i) 
  (hz₀ : z₀ = 1 - 3 * I) 
  (hk : k = 3) : 
  (k * (z - z₀) + z₀) = (-8 + 9 * I) := 
by 
  rw [hz, hz₀, hk]
  -- Sorry means here we didn't write the complete proof, we assume it is correct.
  sorry

end dilation_image_l639_63936


namespace difference_of_two_numbers_l639_63949

theorem difference_of_two_numbers (x y : ℝ) (h1 : x + y = 15) (h2 : x^2 - y^2 = 150) : x - y = 10 :=
by
  sorry

end difference_of_two_numbers_l639_63949


namespace possible_original_numbers_l639_63979

def four_digit_original_number (N : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 
    N = 1000 * a + 100 * b + 10 * c + d ∧ 
    (a+1) * (b+2) * (c+3) * (d+4) = 234 ∧ 
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10

theorem possible_original_numbers : 
  four_digit_original_number 1109 ∨ four_digit_original_number 2009 :=
sorry

end possible_original_numbers_l639_63979


namespace price_per_kg_of_fruits_l639_63944

theorem price_per_kg_of_fruits (mangoes apples oranges : ℕ) (total_amount : ℕ)
  (h1 : mangoes = 400)
  (h2 : apples = 2 * mangoes)
  (h3 : oranges = mangoes + 200)
  (h4 : total_amount = 90000) :
  (total_amount / (mangoes + apples + oranges) = 50) :=
by
  sorry

end price_per_kg_of_fruits_l639_63944


namespace range_of_a_l639_63902

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ) (θ : ℝ), 0 ≤ θ ∧ θ ≤ Real.pi / 2 →
    (x + 3 + 2 * Real.sin θ * Real.cos θ) ^ 2 +
    (x + a * Real.sin θ + a * Real.cos θ) ^ 2 ≥ 1 / 8) ↔
  (a ≥ 7 / 2 ∨ a ≤ Real.sqrt 6) :=
by
  sorry

end range_of_a_l639_63902


namespace unit_square_divisible_l639_63955

theorem unit_square_divisible (n : ℕ) (h: n ≥ 6) : ∃ squares : ℕ, squares = n :=
by
  sorry

end unit_square_divisible_l639_63955


namespace compute_div_mul_l639_63981

theorem compute_div_mul (x y z : Int) (h : y ≠ 0) (hx : x = -100) (hy : y = -25) (hz : z = -6) :
  (((-x) / (-y)) * -z) = -24 := by
  sorry

end compute_div_mul_l639_63981


namespace zero_people_with_fewer_than_six_cards_l639_63988

theorem zero_people_with_fewer_than_six_cards (cards people : ℕ) (h_cards : cards = 60) (h_people : people = 9) :
  let avg := cards / people
  let remainder := cards % people
  remainder < people → ∃ n, n = 0 := by
  sorry

end zero_people_with_fewer_than_six_cards_l639_63988


namespace decaf_percentage_total_l639_63994

-- Defining the initial conditions
def initial_stock : ℝ := 400
def initial_decaf_percentage : ℝ := 0.30
def new_stock : ℝ := 100
def new_decaf_percentage : ℝ := 0.60

-- Given conditions
def amount_initial_decaf := initial_decaf_percentage * initial_stock
def amount_new_decaf := new_decaf_percentage * new_stock
def total_decaf := amount_initial_decaf + amount_new_decaf
def total_stock := initial_stock + new_stock

-- Prove the percentage of decaffeinated coffee in the total stock
theorem decaf_percentage_total : 
  (total_decaf / total_stock) * 100 = 36 := by
  sorry

end decaf_percentage_total_l639_63994


namespace is_minimum_value_l639_63953

noncomputable def f (x : ℝ) : ℝ := x + (1 / x) - 2

theorem is_minimum_value (h : ∀ x > 0, f x ≥ 0) : ∃ (a : ℝ) (h : a > 0), f a = 0 :=
by {
  sorry
}

end is_minimum_value_l639_63953


namespace percentage_of_water_in_first_liquid_l639_63925

theorem percentage_of_water_in_first_liquid (x : ℝ) 
  (h1 : 0 < x ∧ x ≤ 1)
  (h2 : 0.35 = 0.35)
  (h3 : 10 = 10)
  (h4 : 4 = 4)
  (h5 : 0.24285714285714285 = 0.24285714285714285) :
  ((10 * x + 4 * 0.35) / (10 + 4) = 0.24285714285714285) → (x = 0.2) :=
sorry

end percentage_of_water_in_first_liquid_l639_63925


namespace lcm_15_48_eq_240_l639_63971

def is_least_common_multiple (n a b : Nat) : Prop :=
  n % a = 0 ∧ n % b = 0 ∧ ∀ m, (m % a = 0 ∧ m % b = 0) → n ≤ m

theorem lcm_15_48_eq_240 : is_least_common_multiple 240 15 48 :=
by
  sorry

end lcm_15_48_eq_240_l639_63971


namespace students_in_cars_l639_63906

theorem students_in_cars (total_students : ℕ := 396) (buses : ℕ := 7) (students_per_bus : ℕ := 56) :
  total_students - (buses * students_per_bus) = 4 := by
  sorry

end students_in_cars_l639_63906


namespace f_is_odd_function_f_is_increasing_f_max_min_in_interval_l639_63913

variable {f : ℝ → ℝ}

-- The conditions:
axiom additivity : ∀ x y : ℝ, f (x + y) = f x + f y
axiom positive_for_positive : ∀ x : ℝ, x > 0 → f x > 0
axiom f_one_is_two : f 1 = 2

-- The proof tasks:
theorem f_is_odd_function : ∀ x : ℝ, f (-x) = -f x := 
sorry

theorem f_is_increasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 := 
sorry

theorem f_max_min_in_interval : 
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f x ≤ 6) ∧ (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f x ≥ -6) :=
sorry

end f_is_odd_function_f_is_increasing_f_max_min_in_interval_l639_63913


namespace sum_of_consecutive_integers_with_product_272_l639_63993

theorem sum_of_consecutive_integers_with_product_272 :
    ∃ (x y : ℕ), x * y = 272 ∧ y = x + 1 ∧ x + y = 33 :=
by
  sorry

end sum_of_consecutive_integers_with_product_272_l639_63993


namespace exists_k_lt_ak_by_2001_fac_l639_63942

theorem exists_k_lt_ak_by_2001_fac (a : ℕ → ℝ) (H0 : a 0 = 1)
(Hn : ∀ n : ℕ, n > 0 → a n = a (⌊(7 * n / 9)⌋₊) + a (⌊(n / 9)⌋₊)) :
  ∃ k : ℕ, k > 0 ∧ a k < k / ↑(Nat.factorial 2001) := by
  sorry

end exists_k_lt_ak_by_2001_fac_l639_63942


namespace arithmetic_square_root_of_sqrt_16_l639_63999

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_square_root_of_sqrt_16_l639_63999


namespace total_pages_written_is_24_l639_63995

def normal_letter_interval := 3
def time_per_normal_letter := 20
def time_per_page := 10
def additional_time_factor := 2
def time_spent_long_letter := 80
def days_in_month := 30

def normal_letters_written := days_in_month / normal_letter_interval
def pages_per_normal_letter := time_per_normal_letter / time_per_page
def total_pages_normal_letters := normal_letters_written * pages_per_normal_letter

def time_per_page_long_letter := additional_time_factor * time_per_page
def pages_long_letter := time_spent_long_letter / time_per_page_long_letter

def total_pages_written := total_pages_normal_letters + pages_long_letter

theorem total_pages_written_is_24 : total_pages_written = 24 := by
  sorry

end total_pages_written_is_24_l639_63995


namespace number_of_hens_l639_63948

variables (H C : ℕ)

def total_heads (H C : ℕ) : Prop := H + C = 48
def total_feet (H C : ℕ) : Prop := 2 * H + 4 * C = 144

theorem number_of_hens (H C : ℕ) (h1 : total_heads H C) (h2 : total_feet H C) : H = 24 :=
sorry

end number_of_hens_l639_63948


namespace set_cannot_be_divided_l639_63940

theorem set_cannot_be_divided
  (p : ℕ) (prime_p : Nat.Prime p) (p_eq_3_mod_4 : p % 4 = 3)
  (S : Finset ℕ) (hS : S.card = p - 1) :
  ¬∃ A B : Finset ℕ, A ∪ B = S ∧ A ∩ B = ∅ ∧ A.prod id = B.prod id := 
by {
  sorry
}

end set_cannot_be_divided_l639_63940


namespace bruce_money_left_l639_63974

theorem bruce_money_left :
  let initial_amount := 71
  let cost_per_shirt := 5
  let number_of_shirts := 5
  let cost_of_pants := 26
  let total_cost := number_of_shirts * cost_per_shirt + cost_of_pants
  let money_left := initial_amount - total_cost
  money_left = 20 :=
by
  sorry

end bruce_money_left_l639_63974


namespace complete_work_together_in_days_l639_63901

/-
p is 60% more efficient than q.
p can complete the work in 26 days.
Prove that p and q together will complete the work in approximately 18.57 days.
-/

noncomputable def work_together_days (p_efficiency q_efficiency : ℝ) (p_days : ℝ) : ℝ :=
  let p_work_rate := 1 / p_days
  let q_work_rate := q_efficiency / p_efficiency * p_work_rate
  let combined_work_rate := p_work_rate + q_work_rate
  1 / combined_work_rate

theorem complete_work_together_in_days :
  ∀ (p_efficiency q_efficiency p_days : ℝ),
  p_efficiency = 1 ∧ q_efficiency = 0.4 ∧ p_days = 26 →
  abs (work_together_days p_efficiency q_efficiency p_days - 18.57) < 0.01 := by
  intros p_efficiency q_efficiency p_days
  rintro ⟨heff_p, heff_q, hdays_p⟩
  simp [heff_p, heff_q, hdays_p, work_together_days]
  sorry

end complete_work_together_in_days_l639_63901


namespace probability_log3_N_integer_l639_63961
noncomputable def probability_log3_integer : ℚ :=
  let count := 2
  let total := 900
  count / total

theorem probability_log3_N_integer :
  probability_log3_integer = 1 / 450 :=
sorry

end probability_log3_N_integer_l639_63961


namespace sin_identity_l639_63922

theorem sin_identity (α : ℝ) (h_tan : Real.tan α = -3 / 4) : 
  Real.sin α * (Real.sin α - Real.cos α) = 21 / 25 :=
sorry

end sin_identity_l639_63922


namespace intersection_eq_l639_63905

def A : Set ℝ := { x | abs x ≤ 2 }
def B : Set ℝ := { x | 3 * x - 2 ≥ 1 }

theorem intersection_eq :
  A ∩ B = { x | 1 ≤ x ∧ x ≤ 2 } :=
sorry

end intersection_eq_l639_63905


namespace chromosome_stability_due_to_meiosis_and_fertilization_l639_63915

-- Definitions for conditions
def chrom_replicate_distribute_evenly : Prop := true
def central_cell_membrane_invagination : Prop := true
def mitosis : Prop := true
def meiosis_and_fertilization : Prop := true

-- Main theorem statement to be proved
theorem chromosome_stability_due_to_meiosis_and_fertilization :
  meiosis_and_fertilization :=
sorry

end chromosome_stability_due_to_meiosis_and_fertilization_l639_63915


namespace total_tickets_needed_l639_63969

-- Definitions representing the conditions
def rides_go_karts : ℕ := 1
def cost_per_go_kart_ride : ℕ := 4
def rides_bumper_cars : ℕ := 4
def cost_per_bumper_car_ride : ℕ := 5

-- Calculate the total tickets needed
def total_tickets : ℕ := rides_go_karts * cost_per_go_kart_ride + rides_bumper_cars * cost_per_bumper_car_ride

-- The theorem stating the main proof problem
theorem total_tickets_needed : total_tickets = 24 := by
  -- Proof steps should go here, but we use sorry to skip the proof
  sorry

end total_tickets_needed_l639_63969


namespace circle_radius_l639_63938

theorem circle_radius (r M N : ℝ) (hM : M = π * r^2) (hN : N = 2 * π * r) (hRatio : M / N = 20) : r = 40 := 
by
  sorry

end circle_radius_l639_63938


namespace sum_of_abs_values_eq_12_l639_63982

theorem sum_of_abs_values_eq_12 (a b c d : ℝ) (h : 6 * x^2 + x - 12 = (a * x + b) * (c * x + d)) :
  abs a + abs b + abs c + abs d = 12 := sorry

end sum_of_abs_values_eq_12_l639_63982


namespace factorize_2x2_minus_8_factorize_ax2_minus_2ax_plus_a_l639_63984

variable {α : Type*} [CommRing α]

-- Problem 1
theorem factorize_2x2_minus_8 (x : α) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) :=
sorry

-- Problem 2
theorem factorize_ax2_minus_2ax_plus_a (a x : α) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 :=
sorry

end factorize_2x2_minus_8_factorize_ax2_minus_2ax_plus_a_l639_63984


namespace sphere_views_identical_l639_63957

-- Define the geometric shape as a type
inductive GeometricShape
| sphere
| cube
| other (name : String)

-- Define a function to get the view of a sphere
def view (s : GeometricShape) (direction : String) : String :=
  match s with
  | GeometricShape.sphere => "circle"
  | GeometricShape.cube => "square"
  | GeometricShape.other _ => "unknown"

-- The theorem to prove that a sphere has identical front, top, and side views
theorem sphere_views_identical :
  ∀ (direction1 direction2 : String), view GeometricShape.sphere direction1 = view GeometricShape.sphere direction2 :=
by
  intros direction1 direction2
  sorry

end sphere_views_identical_l639_63957


namespace max_value_quadratic_function_l639_63910

open Real

theorem max_value_quadratic_function (r : ℝ) (x₀ y₀ : ℝ) (P_tangent : (2 / x₀) * x - y₀ = 0) 
  (circle_tangent : (x₀ - 3) * (x - 3) + y₀ * y = r^2) :
  ∃ (f : ℝ → ℝ), (∀ (x : ℝ), f x = 1 / 2 * x * (3 - x)) ∧ 
  (∀ (x : ℝ), f x ≤ 9 / 8) :=
by
  sorry

end max_value_quadratic_function_l639_63910


namespace question_1_question_2_l639_63959

variable (m x : ℝ)
def f (x : ℝ) := |x + m|

theorem question_1 (h : f 1 + f (-2) ≥ 5) : 
  m ≤ -2 ∨ m ≥ 3 := sorry

theorem question_2 (hx : x ≠ 0) : 
  f (1 / x) + f (-x) ≥ 2 := sorry

end question_1_question_2_l639_63959


namespace age_problem_l639_63946

theorem age_problem 
  (A : ℕ) 
  (x : ℕ) 
  (h1 : 3 * (A + x) - 3 * (A - 3) = A) 
  (h2 : A = 18) : 
  x = 3 := 
by 
  sorry

end age_problem_l639_63946


namespace determine_house_numbers_l639_63986

-- Definitions based on the conditions given
def even_numbered_side (n : ℕ) : Prop :=
  n % 2 = 0

def sum_balanced (n : ℕ) (house_numbers : List ℕ) : Prop :=
  let left_sum := house_numbers.take n |>.sum
  let right_sum := house_numbers.drop (n + 1) |>.sum
  left_sum = right_sum

def house_constraints (n : ℕ) : Prop :=
  50 < n ∧ n < 500

-- Main theorem statement
theorem determine_house_numbers : 
  ∃ (n : ℕ) (house_numbers : List ℕ), 
    even_numbered_side n ∧ 
    house_constraints n ∧ 
    sum_balanced n house_numbers :=
  sorry

end determine_house_numbers_l639_63986


namespace find_fractions_l639_63967

open Function

-- Define the set and the condition that all numbers must be used precisely once
def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define what it means for fractions to multiply to 1 within the set
def fractions_mul_to_one (a b c d e f : ℕ) : Prop :=
  (a * c * e) = (b * d * f)

-- Define irreducibility condition for a fraction a/b
def irreducible_fraction (a b : ℕ) := 
  Nat.gcd a b = 1

-- Final main problem statement
theorem find_fractions :
  ∃ (a b c d e f : ℕ) (h₁ : a ∈ S) (h₂ : b ∈ S) (h₃ : c ∈ S) (h₄ : d ∈ S) (h₅ : e ∈ S) (h₆ : f ∈ S),
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
  d ≠ e ∧ d ≠ f ∧ 
  e ≠ f ∧
  irreducible_fraction a b ∧ irreducible_fraction c d ∧ irreducible_fraction e f ∧
  fractions_mul_to_one a b c d e f := 
sorry

end find_fractions_l639_63967


namespace number_of_cities_sampled_from_group_B_l639_63927

variable (N_total : ℕ) (N_A : ℕ) (N_B : ℕ) (N_C : ℕ) (S : ℕ)

theorem number_of_cities_sampled_from_group_B :
    N_total = 48 → 
    N_A = 10 → 
    N_B = 18 → 
    N_C = 20 → 
    S = 16 → 
    (N_B * S) / N_total = 6 :=
by
  sorry

end number_of_cities_sampled_from_group_B_l639_63927


namespace coordinates_of_point_with_respect_to_origin_l639_63972

theorem coordinates_of_point_with_respect_to_origin (P : ℝ × ℝ) (h : P = (-2, 4)) : P = (-2, 4) := 
by 
  exact h

end coordinates_of_point_with_respect_to_origin_l639_63972


namespace arithmetic_sequence_odd_function_always_positive_l639_63947

theorem arithmetic_sequence_odd_function_always_positive
    (f : ℝ → ℝ) (a : ℕ → ℝ)
    (h_odd : ∀ x, f (-x) = -f x)
    (h_monotone_geq_0 : ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x)
    (h_arith_seq : ∀ n, a (n + 1) = a n + (a 2 - a 1))
    (h_a3_neg : a 3 < 0) :
    f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) > 0 := by
    sorry

end arithmetic_sequence_odd_function_always_positive_l639_63947


namespace lumberjack_trees_l639_63966

theorem lumberjack_trees (trees logs firewood : ℕ) 
  (h1 : ∀ t, logs = t * 4)
  (h2 : ∀ l, firewood = l * 5)
  (h3 : firewood = 500)
  : trees = 25 :=
by
  sorry

end lumberjack_trees_l639_63966


namespace sunglasses_cap_probability_l639_63903

theorem sunglasses_cap_probability
  (sunglasses_count : ℕ) (caps_count : ℕ)
  (P_cap_and_sunglasses_given_cap : ℚ)
  (H1 : sunglasses_count = 60)
  (H2 : caps_count = 40)
  (H3 : P_cap_and_sunglasses_given_cap = 2/5) :
  (∃ (x : ℚ), x = (16 : ℚ) / 60 ∧ x = 4 / 15) := sorry

end sunglasses_cap_probability_l639_63903


namespace trajectory_point_M_l639_63934

theorem trajectory_point_M (x y : ℝ) : 
  (∃ (m n : ℝ), x^2 + y^2 = 9 ∧ (m = x) ∧ (n = 3 * y)) → 
  (x^2 / 9 + y^2 = 1) :=
by
  sorry

end trajectory_point_M_l639_63934


namespace seulgi_second_round_score_l639_63929

theorem seulgi_second_round_score
    (h_score1 : Nat) (h_score2 : Nat)
    (hj_score1 : Nat) (hj_score2 : Nat)
    (s_score1 : Nat) (required_second_score : Nat) :
    h_score1 = 23 →
    h_score2 = 28 →
    hj_score1 = 32 →
    hj_score2 = 17 →
    s_score1 = 27 →
    required_second_score = 25 →
    s_score1 + required_second_score > h_score1 + h_score2 ∧ 
    s_score1 + required_second_score > hj_score1 + hj_score2 :=
by
  intros
  sorry

end seulgi_second_round_score_l639_63929


namespace can_capacity_l639_63960

/-- Given a can with a mixture of milk and water in the ratio 4:3, and adding 10 liters of milk
results in the can being full and changes the ratio to 5:2, prove that the capacity of the can is 30 liters. -/
theorem can_capacity (x : ℚ)
  (h1 : 4 * x + 3 * x + 10 = 30)
  (h2 : (4 * x + 10) / (3 * x) = 5 / 2) :
  4 * x + 3 * x + 10 = 30 := 
by sorry

end can_capacity_l639_63960


namespace factorize_a3_minus_4ab2_l639_63956

theorem factorize_a3_minus_4ab2 (a b : ℝ) : a^3 - 4 * a * b^2 = a * (a + 2 * b) * (a - 2 * b) :=
by
  -- Proof is omitted; write 'sorry' as a placeholder
  sorry

end factorize_a3_minus_4ab2_l639_63956


namespace problem_l639_63970

theorem problem
    (a b c d : ℕ)
    (h1 : a = b + 7)
    (h2 : b = c + 15)
    (h3 : c = d + 25)
    (h4 : d = 90) :
  a = 137 := by
  sorry

end problem_l639_63970


namespace monotonically_increasing_power_function_l639_63987

theorem monotonically_increasing_power_function (m : ℝ) :
  (∀ x : ℝ, 0 < x → (m ^ 2 - 2 * m - 2) * x ^ (m - 2) > 0 → (m ^ 2 - 2 * m - 2) > 0 ∧ (m - 2) > 0) ↔ m = 3 := 
sorry

end monotonically_increasing_power_function_l639_63987


namespace arithmetic_seq_common_diff_l639_63909

theorem arithmetic_seq_common_diff (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 0 + a 2 = 10) 
  (h2 : a 3 + a 5 = 4)
  (h_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d) :
  d = -1 := 
  sorry

end arithmetic_seq_common_diff_l639_63909


namespace find_x_l639_63963

-- Let's define the constants and the condition
def a : ℝ := 2.12
def b : ℝ := 0.345
def c : ℝ := 2.4690000000000003

-- We need to prove that there exists a number x such that
def x : ℝ := 0.0040000000000003

-- Formal statement
theorem find_x : a + b + x = c :=
by
  -- Proof skipped
  sorry
 
end find_x_l639_63963


namespace print_time_325_pages_l639_63900

theorem print_time_325_pages (pages : ℕ) (rate : ℕ) (delay_pages : ℕ) (delay_time : ℕ)
  (h_pages : pages = 325) (h_rate : rate = 25) (h_delay_pages : delay_pages = 100) (h_delay_time : delay_time = 1) :
  let print_time := pages / rate
  let delays := pages / delay_pages
  let total_time := print_time + delays * delay_time
  total_time = 16 :=
by
  sorry

end print_time_325_pages_l639_63900
