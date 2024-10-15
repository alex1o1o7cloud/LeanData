import Mathlib

namespace NUMINAMATH_GPT_div_30_prime_ge_7_l558_55822

theorem div_30_prime_ge_7 (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge_7 : p ≥ 7) : 30 ∣ (p^2 - 1) := 
sorry

end NUMINAMATH_GPT_div_30_prime_ge_7_l558_55822


namespace NUMINAMATH_GPT_find_sample_size_l558_55859

def ratio_A : ℕ := 2
def ratio_B : ℕ := 3
def ratio_C : ℕ := 5
def total_ratio : ℕ := ratio_A + ratio_B + ratio_C
def num_B_selected : ℕ := 24

theorem find_sample_size : ∃ n : ℕ, num_B_selected * total_ratio = ratio_B * n :=
by
  sorry

end NUMINAMATH_GPT_find_sample_size_l558_55859


namespace NUMINAMATH_GPT_b_share_220_l558_55874

theorem b_share_220 (A B C : ℝ) (h1 : A = B + 40) (h2 : C = A + 30) (h3 : A + B + C = 770) : 
  B = 220 :=
by
  sorry

end NUMINAMATH_GPT_b_share_220_l558_55874


namespace NUMINAMATH_GPT_eggs_not_eaten_per_week_l558_55821

theorem eggs_not_eaten_per_week : 
  let trays_bought := 2
  let eggs_per_tray := 24
  let days_per_week := 7
  let eggs_eaten_by_children_per_day := 2 * 2 -- 2 eggs each by 2 children
  let eggs_eaten_by_parents_per_day := 4
  let total_eggs_eaten_per_week := (eggs_eaten_by_children_per_day + eggs_eaten_by_parents_per_day) * days_per_week
  let total_eggs_bought := trays_bought * eggs_per_tray * 2  -- Re-calculated trays
  total_eggs_bought - total_eggs_eaten_per_week = 40 :=
by
  let trays_bought := 2
  let eggs_per_tray := 24
  let days_per_week := 7
  let eggs_eaten_by_children_per_day := 2 * 2
  let eggs_eaten_by_parents_per_day := 4
  let total_eggs_eaten_per_week := (eggs_eaten_by_children_per_day + eggs_eaten_by_parents_per_day) * days_per_week
  let total_eggs_bought := trays_bought * eggs_per_tray * 2
  show total_eggs_bought - total_eggs_eaten_per_week = 40
  sorry

end NUMINAMATH_GPT_eggs_not_eaten_per_week_l558_55821


namespace NUMINAMATH_GPT_correct_factorization_l558_55890

theorem correct_factorization :
  ∀ (x : ℝ), -x^2 + 2*x - 1 = - (x - 1)^2 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_correct_factorization_l558_55890


namespace NUMINAMATH_GPT_overlapping_squares_area_l558_55808

theorem overlapping_squares_area :
  let s : ℝ := 5
  let total_area := 3 * s^2
  let redundant_area := s^2 / 8 * 4
  total_area - redundant_area = 62.5 := by
  sorry

end NUMINAMATH_GPT_overlapping_squares_area_l558_55808


namespace NUMINAMATH_GPT_farmer_land_l558_55814

theorem farmer_land (A : ℝ) (h1 : 0.9 * A = A_cleared) (h2 : 0.3 * A_cleared = A_soybeans) 
  (h3 : 0.6 * A_cleared = A_wheat) (h4 : 0.1 * A_cleared = 540) : A = 6000 :=
by
  sorry

end NUMINAMATH_GPT_farmer_land_l558_55814


namespace NUMINAMATH_GPT_candy_division_l558_55871

def pieces_per_bag (total_candies : ℕ) (bags : ℕ) : ℕ :=
total_candies / bags

theorem candy_division : pieces_per_bag 42 2 = 21 :=
by
  sorry

end NUMINAMATH_GPT_candy_division_l558_55871


namespace NUMINAMATH_GPT_collinear_vectors_x_value_l558_55866

theorem collinear_vectors_x_value (x : ℝ) (a b : ℝ × ℝ) (h₁: a = (2, x)) (h₂: b = (1, 2))
  (h₃: ∃ k : ℝ, a = k • b) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_collinear_vectors_x_value_l558_55866


namespace NUMINAMATH_GPT_log_mul_l558_55854

theorem log_mul (a M N : ℝ) (ha_pos : 0 < a) (hM_pos : 0 < M) (hN_pos : 0 < N) (ha_ne_one : a ≠ 1) :
    Real.log (M * N) / Real.log a = Real.log M / Real.log a + Real.log N / Real.log a := by
  sorry

end NUMINAMATH_GPT_log_mul_l558_55854


namespace NUMINAMATH_GPT_simplified_expression_at_one_l558_55896

noncomputable def original_expression (a : ℚ) : ℚ :=
  (2 * a + 2) / a / (4 / (a ^ 2)) - a / (a + 1)

theorem simplified_expression_at_one : original_expression 1 = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_simplified_expression_at_one_l558_55896


namespace NUMINAMATH_GPT_sqrt_of_36_is_6_l558_55804

-- Define the naturals
def arithmetic_square_root (x : ℕ) : ℕ := Nat.sqrt x

theorem sqrt_of_36_is_6 : arithmetic_square_root 36 = 6 :=
by
  -- The proof goes here, but we use sorry to skip it as per instructions.
  sorry

end NUMINAMATH_GPT_sqrt_of_36_is_6_l558_55804


namespace NUMINAMATH_GPT_mixture_weight_l558_55823

theorem mixture_weight (C : ℚ) (W : ℚ)
  (H1: C > 0) -- C represents the cost per pound of milk powder and coffee in June, and is a positive number
  (H2: C * 0.2 = 0.2) -- The price per pound of milk powder in July
  (H3: (W / 2) * 0.2 + (W / 2) * 4 * C = 6.30) -- The cost of the mixture in July

  : W = 3 := 
sorry

end NUMINAMATH_GPT_mixture_weight_l558_55823


namespace NUMINAMATH_GPT_feifei_sheep_count_l558_55810

noncomputable def sheep_number (x y : ℕ) : Prop :=
  (y = 3 * x + 15) ∧ (x = y - y / 3)

theorem feifei_sheep_count :
  ∃ x y : ℕ, sheep_number x y ∧ x = 5 :=
sorry

end NUMINAMATH_GPT_feifei_sheep_count_l558_55810


namespace NUMINAMATH_GPT_Oliver_has_9_dollars_left_l558_55819

def initial_amount := 9
def saved := 5
def earned := 6
def spent_frisbee := 4
def spent_puzzle := 3
def spent_stickers := 2
def spent_movie_ticket := 7
def spent_snack := 3
def gift := 8

def final_amount (initial_amount : ℕ) (saved : ℕ) (earned : ℕ) (spent_frisbee : ℕ)
                 (spent_puzzle : ℕ) (spent_stickers : ℕ) (spent_movie_ticket : ℕ)
                 (spent_snack : ℕ) (gift : ℕ) : ℕ :=
  initial_amount + saved + earned - spent_frisbee - spent_puzzle - spent_stickers - 
  spent_movie_ticket - spent_snack + gift

theorem Oliver_has_9_dollars_left :
  final_amount initial_amount saved earned spent_frisbee 
               spent_puzzle spent_stickers spent_movie_ticket 
               spent_snack gift = 9 :=
  by
  sorry

end NUMINAMATH_GPT_Oliver_has_9_dollars_left_l558_55819


namespace NUMINAMATH_GPT_triangle_inequality_l558_55833

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  2 < (a + b) / c + (b + c) / a + (c + a) / b - (a^3 + b^3 + c^3) / (a * b * c) ∧
  (a + b) / c + (b + c) / a + (c + a) / b - (a^3 + b^3 + c^3) / (a * b * c) ≤ 3 :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l558_55833


namespace NUMINAMATH_GPT_dice_probability_l558_55870

noncomputable def probability_same_face (throws : ℕ) (dice : ℕ) : ℚ :=
  1 - (1 - (1 / 6) ^ dice) ^ throws

theorem dice_probability : 
  probability_same_face 5 10 = 1 - (1 - (1 / 6) ^ 10) ^ 5 :=
by 
  sorry

end NUMINAMATH_GPT_dice_probability_l558_55870


namespace NUMINAMATH_GPT_product_of_two_even_numbers_is_even_product_of_two_odd_numbers_is_odd_product_of_even_and_odd_number_is_even_product_of_odd_and_even_number_is_even_l558_55858

-- Definition of even and odd numbers
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- Theorem statements for each condition

-- Prove that the product of two even numbers is even
theorem product_of_two_even_numbers_is_even (a b : ℤ) :
  is_even a → is_even b → is_even (a * b) :=
by sorry

-- Prove that the product of two odd numbers is odd
theorem product_of_two_odd_numbers_is_odd (c d : ℤ) :
  is_odd c → is_odd d → is_odd (c * d) :=
by sorry

-- Prove that the product of one even and one odd number is even
theorem product_of_even_and_odd_number_is_even (e f : ℤ) :
  is_even e → is_odd f → is_even (e * f) :=
by sorry

-- Prove that the product of one odd and one even number is even
theorem product_of_odd_and_even_number_is_even (g h : ℤ) :
  is_odd g → is_even h → is_even (g * h) :=
by sorry

end NUMINAMATH_GPT_product_of_two_even_numbers_is_even_product_of_two_odd_numbers_is_odd_product_of_even_and_odd_number_is_even_product_of_odd_and_even_number_is_even_l558_55858


namespace NUMINAMATH_GPT_total_cost_of_two_rackets_l558_55888

axiom racket_full_price : ℕ
axiom price_of_first_racket : racket_full_price = 60
axiom price_of_second_racket : racket_full_price / 2 = 30

theorem total_cost_of_two_rackets : 60 + 30 = 90 :=
sorry

end NUMINAMATH_GPT_total_cost_of_two_rackets_l558_55888


namespace NUMINAMATH_GPT_quadratic_distinct_real_roots_l558_55827

theorem quadratic_distinct_real_roots (m : ℝ) : 
  (m ≠ 0 ∧ m < 1 / 5) ↔ ∃ (x y : ℝ), x ≠ y ∧ m * x^2 - 2 * x + 5 = 0 ∧ m * y^2 - 2 * y + 5 = 0 :=
sorry

end NUMINAMATH_GPT_quadratic_distinct_real_roots_l558_55827


namespace NUMINAMATH_GPT_sum_fractions_l558_55855

theorem sum_fractions:
  (Finset.range 16).sum (λ k => (k + 1) / 7) = 136 / 7 := by
  sorry

end NUMINAMATH_GPT_sum_fractions_l558_55855


namespace NUMINAMATH_GPT_complex_division_l558_55838

theorem complex_division (z : ℂ) (hz : (3 + 4 * I) * z = 25) : z = 3 - 4 * I :=
sorry

end NUMINAMATH_GPT_complex_division_l558_55838


namespace NUMINAMATH_GPT_number_of_tshirts_sold_l558_55817

theorem number_of_tshirts_sold 
    (original_price discounted_price revenue : ℕ)
    (discount : ℕ) 
    (no_of_tshirts: ℕ)
    (h1 : original_price = 51)
    (h2 : discount = 8)
    (h3 : discounted_price = original_price - discount)
    (h4 : revenue = 5590)
    (h5 : revenue = no_of_tshirts * discounted_price) : 
    no_of_tshirts = 130 :=
by
  sorry

end NUMINAMATH_GPT_number_of_tshirts_sold_l558_55817


namespace NUMINAMATH_GPT_polynomial_evaluation_l558_55849

theorem polynomial_evaluation (n : ℕ) (p : ℕ → ℝ) 
  (h_poly : ∀ k, k ≤ n → p k = 1 / (Nat.choose (n + 1) k)) :
  p (n + 1) = if n % 2 = 0 then 1 else 0 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_evaluation_l558_55849


namespace NUMINAMATH_GPT_compare_f_values_l558_55899

variable (a : ℝ) (f : ℝ → ℝ) (m n : ℝ)

theorem compare_f_values (h_a : 0 < a ∧ a < 1)
    (h_f : ∀ x > 0, f (Real.logb a x) = a * (x^2 - 1) / (x * (a^2 - 1)))
    (h_mn : m > n ∧ n > 0 ∧ m > 0) :
    f (1 / n) > f (1 / m) := by 
  sorry

end NUMINAMATH_GPT_compare_f_values_l558_55899


namespace NUMINAMATH_GPT_max_value_of_b_l558_55891

theorem max_value_of_b {m b : ℚ} (x : ℤ) 
  (line_eq : ∀ x : ℤ, 0 < x ∧ x ≤ 200 → 
    ¬ ∃ (y : ℤ), y = m * x + 3)
  (m_range : 1/3 < m ∧ m < b) :
  b = 69/208 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_b_l558_55891


namespace NUMINAMATH_GPT_no_real_roots_x_squared_minus_x_plus_nine_l558_55851

theorem no_real_roots_x_squared_minus_x_plus_nine :
  ∀ x : ℝ, ¬ (x^2 - x + 9 = 0) :=
by 
  intro x 
  sorry

end NUMINAMATH_GPT_no_real_roots_x_squared_minus_x_plus_nine_l558_55851


namespace NUMINAMATH_GPT_number_of_sophomores_l558_55894

-- Definition of the conditions
variables (J S P j s p : ℕ)

-- Condition: Equal number of students in debate team
def DebateTeam_Equal : Prop := j = s ∧ s = p

-- Condition: Total number of students
def TotalStudents : Prop := J + S + P = 45

-- Condition: Percentage relationships
def PercentRelations_J : Prop := j = J / 5
def PercentRelations_S : Prop := s = 3 * S / 20
def PercentRelations_P : Prop := p = P / 10

-- The main theorem to prove
theorem number_of_sophomores : DebateTeam_Equal j s p 
                               → TotalStudents J S P 
                               → PercentRelations_J J j 
                               → PercentRelations_S S s 
                               → PercentRelations_P P p 
                               → P = 21 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_sophomores_l558_55894


namespace NUMINAMATH_GPT_paint_per_large_canvas_l558_55876

-- Define the conditions
variables (L : ℕ) (paint_large paint_small total_paint : ℕ)

-- Given conditions
def large_canvas_paint := 3 * L
def small_canvas_paint := 4 * 2
def total_paint_used := large_canvas_paint + small_canvas_paint

-- Statement that needs to be proven
theorem paint_per_large_canvas :
  total_paint_used = 17 → L = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_paint_per_large_canvas_l558_55876


namespace NUMINAMATH_GPT_total_birds_in_tree_l558_55898

theorem total_birds_in_tree (bluebirds cardinals swallows : ℕ) 
  (h1 : swallows = 2) 
  (h2 : swallows = bluebirds / 2) 
  (h3 : cardinals = 3 * bluebirds) : 
  swallows + bluebirds + cardinals = 18 := 
by 
  sorry

end NUMINAMATH_GPT_total_birds_in_tree_l558_55898


namespace NUMINAMATH_GPT_modified_prism_surface_area_l558_55883

theorem modified_prism_surface_area :
  let original_surface_area := 2 * (2 * 4 + 2 * 5 + 4 * 5)
  let modified_surface_area := original_surface_area + 5
  modified_surface_area = original_surface_area + 5 :=
by
  -- set the original dimensions
  let l := 2
  let w := 4
  let h := 5
  -- calculate original surface area
  let SA_original := 2 * (l * w + l * h + w * h)
  -- calculate modified surface area
  let SA_new := SA_original + 5
  -- assert the relationship
  have : SA_new = SA_original + 5 := rfl
  exact this

end NUMINAMATH_GPT_modified_prism_surface_area_l558_55883


namespace NUMINAMATH_GPT_carla_order_cost_l558_55840

theorem carla_order_cost (base_cost : ℝ) (coupon : ℝ) (senior_discount_rate : ℝ)
  (additional_charge : ℝ) (tax_rate : ℝ) (conversion_rate : ℝ) :
  base_cost = 7.50 →
  coupon = 2.50 →
  senior_discount_rate = 0.20 →
  additional_charge = 1.00 →
  tax_rate = 0.08 →
  conversion_rate = 0.85 →
  (2 * (base_cost - coupon) * (1 - senior_discount_rate) + additional_charge) * (1 + tax_rate) * conversion_rate = 4.59 :=
by
  sorry

end NUMINAMATH_GPT_carla_order_cost_l558_55840


namespace NUMINAMATH_GPT_simplify_expression_l558_55834

theorem simplify_expression : (Real.sqrt (9 / 4) - Real.sqrt (4 / 9)) = 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l558_55834


namespace NUMINAMATH_GPT_incorrect_conclusion_l558_55856

-- Define the given parabola.
def parabola (x : ℝ) : ℝ := (x - 2)^2 + 1

-- Define the conditions for the parabola.
def parabola_opens_upwards : Prop := ∀ x y : ℝ, parabola (x + y) = (x + y - 2)^2 + 1
def axis_of_symmetry : Prop := ∀ x : ℝ, parabola x = parabola (4 - x)
def vertex_coordinates : Prop := parabola 2 = 1 ∧ (parabola 2, 2) = (1, 2)
def behavior_when_x_less_than_2 : Prop := ∀ x : ℝ, x < 2 → parabola x < parabola (x + 1)

-- The statement that needs to be proven in Lean 4.
theorem incorrect_conclusion : ¬ behavior_when_x_less_than_2 :=
  by
  sorry

end NUMINAMATH_GPT_incorrect_conclusion_l558_55856


namespace NUMINAMATH_GPT_find_pq_l558_55872

noncomputable def find_k_squared (x y : ℝ) : ℝ :=
  let u1 := x^2 + y^2 - 12 * x + 16 * y - 160
  let u2 := x^2 + y^2 + 12 * x + 16 * y - 36
  let k_sq := 741 / 324
  k_sq

theorem find_pq : (741 + 324) = 1065 := by
  sorry

end NUMINAMATH_GPT_find_pq_l558_55872


namespace NUMINAMATH_GPT_conference_handshakes_l558_55877

-- Define the number of attendees at the conference
def attendees : ℕ := 10

-- Define the number of ways to choose 2 people from the attendees
-- This is equivalent to the combination formula C(10, 2)
def handshakes (n : ℕ) : ℕ := n.choose 2

-- Prove that the number of handshakes at the conference is 45
theorem conference_handshakes : handshakes attendees = 45 := by
  sorry

end NUMINAMATH_GPT_conference_handshakes_l558_55877


namespace NUMINAMATH_GPT_square_in_semicircle_l558_55835

theorem square_in_semicircle (Q : ℝ) (h1 : ∃ Q : ℝ, (Q^2 / 4) + Q^2 = 4) : Q = 4 * Real.sqrt 5 / 5 := sorry

end NUMINAMATH_GPT_square_in_semicircle_l558_55835


namespace NUMINAMATH_GPT_three_equal_mass_piles_l558_55869

theorem three_equal_mass_piles (n : ℕ) (h : n > 3) : 
  (∃ (A B C : Finset ℕ), 
    (A ∪ B ∪ C = Finset.range (n + 1)) ∧ 
    (A ∩ B = ∅) ∧ 
    (A ∩ C = ∅) ∧ 
    (B ∩ C = ∅) ∧ 
    (A.sum id = B.sum id) ∧ 
    (B.sum id = C.sum id)) 
  ↔ (n % 3 = 0 ∨ n % 3 = 2) :=
sorry

end NUMINAMATH_GPT_three_equal_mass_piles_l558_55869


namespace NUMINAMATH_GPT_min_value_at_zero_max_value_a_l558_55824

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log (x + 1) - (a * x / (x + 1))

-- Part (I)
theorem min_value_at_zero {a : ℝ} (h : ∀ x, f x a ≥ f 0 a) : a = 1 :=
sorry

-- Part (II)
theorem max_value_a (h : ∀ x > 0, f x a > 0) : a ≤ 1 :=
sorry

end NUMINAMATH_GPT_min_value_at_zero_max_value_a_l558_55824


namespace NUMINAMATH_GPT_Djibo_sister_age_l558_55843

variable (d s : ℕ)
variable (h1 : d = 17)
variable (h2 : d - 5 + (s - 5) = 35)

theorem Djibo_sister_age : s = 28 :=
by sorry

end NUMINAMATH_GPT_Djibo_sister_age_l558_55843


namespace NUMINAMATH_GPT_solve_fraction_l558_55842

variable (x y : ℝ)
variable (h1 : y > x)
variable (h2 : x > 0)
variable (h3 : x / y + y / x = 8)

theorem solve_fraction : (x + y) / (x - y) = Real.sqrt (5 / 3) :=
by
  sorry

end NUMINAMATH_GPT_solve_fraction_l558_55842


namespace NUMINAMATH_GPT_sequence_sum_l558_55831

theorem sequence_sum {A B C D E F G H I J : ℤ} (hD : D = 8)
    (h_sum1 : A + B + C + D = 45)
    (h_sum2 : B + C + D + E = 45)
    (h_sum3 : C + D + E + F = 45)
    (h_sum4 : D + E + F + G = 45)
    (h_sum5 : E + F + G + H = 45)
    (h_sum6 : F + G + H + I = 45)
    (h_sum7 : G + H + I + J = 45)
    (h_sum8 : H + I + J + A = 45)
    (h_sum9 : I + J + A + B = 45)
    (h_sum10 : J + A + B + C = 45) :
  A + J = 0 := 
sorry

end NUMINAMATH_GPT_sequence_sum_l558_55831


namespace NUMINAMATH_GPT_universal_friendship_l558_55811

-- Define the inhabitants and their relationships
def inhabitants (n : ℕ) : Type := Fin n

-- Condition for friends and enemies
inductive Relationship (n : ℕ) : inhabitants n → inhabitants n → Prop
| friend (A B : inhabitants n) : Relationship n A B
| enemy (A B : inhabitants n) : Relationship n A B

-- Transitivity condition
axiom transitivity {n : ℕ} {A B C : inhabitants n} :
  Relationship n A B = Relationship n B C → Relationship n A C = Relationship n A B

-- At least two friends among any three inhabitants
axiom at_least_two_friends {n : ℕ} (A B C : inhabitants n) :
  ∃ X Y : inhabitants n, X ≠ Y ∧ Relationship n X Y = Relationship n X Y

-- Inhabitants can start a new life switching relationships
axiom start_new_life {n : ℕ} (A : inhabitants n) :
  ∀ B : inhabitants n, Relationship n A B = Relationship n B A

-- The main theorem we need to prove
theorem universal_friendship (n : ℕ) : 
  ∀ A B : inhabitants n, ∃ C : inhabitants n, Relationship n A C = Relationship n B C :=
sorry

end NUMINAMATH_GPT_universal_friendship_l558_55811


namespace NUMINAMATH_GPT_line_equation_l558_55865

theorem line_equation (a b : ℝ) 
  (h1 : -4 = (a + 0) / 2)
  (h2 : 6 = (0 + b) / 2) :
  (∀ x y : ℝ, y = (3 / 2) * (x + 4) → 3 * x - 2 * y + 24 = 0) :=
by
  sorry

end NUMINAMATH_GPT_line_equation_l558_55865


namespace NUMINAMATH_GPT_tom_distance_before_karen_wins_l558_55813

theorem tom_distance_before_karen_wins 
    (karen_speed : ℕ)
    (tom_speed : ℕ) 
    (karen_late_start : ℚ) 
    (karen_additional_distance : ℕ) 
    (T : ℚ) 
    (condition1 : karen_speed = 60) 
    (condition2 : tom_speed = 45)
    (condition3 : karen_late_start = 4 / 60)
    (condition4 : karen_additional_distance = 4)
    (condition5 : 60 * T = 45 * T + 8) :
    (45 * (8 / 15) = 24) :=
by
    sorry 

end NUMINAMATH_GPT_tom_distance_before_karen_wins_l558_55813


namespace NUMINAMATH_GPT_number_of_possible_winning_scores_l558_55836

noncomputable def sum_of_first_n_integers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem number_of_possible_winning_scores : 
  let total_sum := sum_of_first_n_integers 12
  let max_possible_score := total_sum / 2
  let min_possible_score := sum_of_first_n_integers 6
  39 - 21 + 1 = 19 := 
by
  sorry

end NUMINAMATH_GPT_number_of_possible_winning_scores_l558_55836


namespace NUMINAMATH_GPT_value_of_p_l558_55878

noncomputable def third_term (x y : ℝ) := 45 * x^8 * y^2
noncomputable def fourth_term (x y : ℝ) := 120 * x^7 * y^3

theorem value_of_p (p q : ℝ) (h1 : third_term p q = fourth_term p q) (h2 : p + 2 * q = 1) (h3 : 0 < p) (h4 : 0 < q) : p = 4 / 7 :=
by
  have h : third_term p q = 45 * p^8 * q^2 := rfl
  have h' : fourth_term p q = 120 * p^7 * q^3 := rfl
  rw [h, h'] at h1
  sorry

end NUMINAMATH_GPT_value_of_p_l558_55878


namespace NUMINAMATH_GPT_squares_sum_l558_55841

theorem squares_sum {r s : ℝ} (h1 : r * s = 16) (h2 : r + s = 8) : r^2 + s^2 = 32 :=
by
  sorry

end NUMINAMATH_GPT_squares_sum_l558_55841


namespace NUMINAMATH_GPT_even_function_a_eq_4_l558_55803

noncomputable def f (x a : ℝ) : ℝ := (x + a) * (x - 4)

theorem even_function_a_eq_4 (a : ℝ) (h : ∀ x : ℝ, f (-x) a = f x a) : a = 4 := by
  sorry

end NUMINAMATH_GPT_even_function_a_eq_4_l558_55803


namespace NUMINAMATH_GPT_boys_from_school_a_not_study_science_l558_55863

theorem boys_from_school_a_not_study_science (total_boys : ℕ) (boys_from_school_a_percentage : ℝ) (science_study_percentage : ℝ)
  (total_boys_in_camp : total_boys = 250) (school_a_percent : boys_from_school_a_percentage = 0.20) 
  (science_percent : science_study_percentage = 0.30) :
  ∃ (boys_from_school_a_not_science : ℕ), boys_from_school_a_not_science = 35 :=
by
  sorry

end NUMINAMATH_GPT_boys_from_school_a_not_study_science_l558_55863


namespace NUMINAMATH_GPT_complementary_angle_l558_55828

theorem complementary_angle (angle_deg : ℕ) (angle_min : ℕ) 
  (h1 : angle_deg = 37) (h2 : angle_min = 38) : 
  exists (comp_deg : ℕ) (comp_min : ℕ), comp_deg = 52 ∧ comp_min = 22 :=
by
  sorry

end NUMINAMATH_GPT_complementary_angle_l558_55828


namespace NUMINAMATH_GPT_roger_piles_of_quarters_l558_55800

theorem roger_piles_of_quarters (Q : ℕ) 
  (h₀ : ∃ Q : ℕ, True) 
  (h₁ : ∀ p, (p = Q) → True)
  (h₂ : ∀ c, (c = 7) → True) 
  (h₃ : Q * 14 = 42) : 
  Q = 3 := 
sorry

end NUMINAMATH_GPT_roger_piles_of_quarters_l558_55800


namespace NUMINAMATH_GPT_distance_from_center_to_tangent_chord_l558_55892

theorem distance_from_center_to_tangent_chord
  (R a m x : ℝ)
  (h1 : m^2 = 4 * R^2)
  (h2 : 16 * R^2 * x^4 - 16 * R^2 * x^2 * (a^2 + R^2) + 16 * a^4 * R^4 - a^2 * (4 * R^2 - m^2)^2 = 0) :
  x = R :=
sorry

end NUMINAMATH_GPT_distance_from_center_to_tangent_chord_l558_55892


namespace NUMINAMATH_GPT_volume_of_prism_l558_55816

theorem volume_of_prism (x y z : ℝ) (h1 : x * y = 100) (h2 : z = 10) (h3 : x * z = 50) (h4 : y * z = 40):
  x * y * z = 200 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_prism_l558_55816


namespace NUMINAMATH_GPT_second_machine_time_l558_55837

theorem second_machine_time
  (machine1_rate : ℕ)
  (machine2_rate : ℕ)
  (combined_rate12 : ℕ)
  (combined_rate123 : ℕ)
  (rate3 : ℕ)
  (time3 : ℚ) :
  machine1_rate = 60 →
  machine2_rate = 120 →
  combined_rate12 = 200 →
  combined_rate123 = 600 →
  rate3 = 420 →
  time3 = 10 / 7 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_second_machine_time_l558_55837


namespace NUMINAMATH_GPT_max_value_of_expression_l558_55802

noncomputable def max_expression_value (a b : ℝ) := a * b * (100 - 5 * a - 2 * b)

theorem max_value_of_expression :
  ∀ (a b : ℝ), 0 < a → 0 < b → 5 * a + 2 * b < 100 →
  max_expression_value a b ≤ 78125 / 36 := by
  intros a b ha hb h
  sorry

end NUMINAMATH_GPT_max_value_of_expression_l558_55802


namespace NUMINAMATH_GPT_find_x_l558_55886

-- define initial quantities of apples and oranges
def initial_apples (x : ℕ) : ℕ := 3 * x + 1
def initial_oranges (x : ℕ) : ℕ := 4 * x + 12

-- define the condition that the number of oranges is twice the number of apples
def condition (x : ℕ) : Prop := initial_oranges x = 2 * initial_apples x

-- define the final state
def final_apples : ℕ := 1
def final_oranges : ℕ := 12

-- theorem to prove that the number of times is 5
theorem find_x : ∃ x : ℕ, condition x ∧ final_apples = 1 ∧ final_oranges = 12 :=
by
  use 5
  sorry

end NUMINAMATH_GPT_find_x_l558_55886


namespace NUMINAMATH_GPT_sequence_remainder_prime_l558_55825

theorem sequence_remainder_prime (p : ℕ) (hp : Nat.Prime p) (x : ℕ → ℕ)
  (h1 : ∀ i, 0 ≤ i ∧ i < p → x i = i)
  (h2 : ∀ n, n ≥ p → x n = x (n-1) + x (n-p)) :
  (x (p^3) % p) = p - 1 :=
sorry

end NUMINAMATH_GPT_sequence_remainder_prime_l558_55825


namespace NUMINAMATH_GPT_problem_equivalent_final_answer_l558_55807

noncomputable def a := 12
noncomputable def b := 27
noncomputable def c := 6

theorem problem_equivalent :
  2 * Real.sqrt 3 + (2 / Real.sqrt 3) + 3 * Real.sqrt 2 + (3 / Real.sqrt 2) = (a * Real.sqrt 3 + b * Real.sqrt 2) / c :=
  sorry

theorem final_answer :
  a + b + c = 45 :=
  by
    unfold a b c
    simp
    done

end NUMINAMATH_GPT_problem_equivalent_final_answer_l558_55807


namespace NUMINAMATH_GPT_roots_of_star_equation_l558_55839

def star (m n : ℝ) : ℝ := m * n^2 - m * n - 1

theorem roots_of_star_equation :
  ∀ x : ℝ, (star 1 x = 0) → (∃ a b : ℝ, a ≠ b ∧ x = a ∨ x = b) := 
by
  sorry

end NUMINAMATH_GPT_roots_of_star_equation_l558_55839


namespace NUMINAMATH_GPT_bin_expected_value_l558_55884

theorem bin_expected_value (m : ℕ) (h : (21 - 4 * m) / (7 + m) = 1) : m = 3 := 
by {
  sorry
}

end NUMINAMATH_GPT_bin_expected_value_l558_55884


namespace NUMINAMATH_GPT_four_digit_number_divisible_by_36_l558_55857

theorem four_digit_number_divisible_by_36 (n : ℕ) (h₁ : ∃ k : ℕ, 6130 + n = 36 * k) 
  (h₂ : ∃ k : ℕ, 130 + n = 4 * k) 
  (h₃ : ∃ k : ℕ, (10 + n) = 9 * k) : n = 6 :=
sorry

end NUMINAMATH_GPT_four_digit_number_divisible_by_36_l558_55857


namespace NUMINAMATH_GPT_fifth_equation_l558_55820

theorem fifth_equation
: 1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 = 21^2 :=
by
  sorry

end NUMINAMATH_GPT_fifth_equation_l558_55820


namespace NUMINAMATH_GPT_original_time_40_l558_55848

theorem original_time_40
  (S T : ℝ)
  (h1 : ∀ D : ℝ, D = S * T)
  (h2 : ∀ D : ℝ, D = 0.8 * S * (T + 10)) :
  T = 40 :=
by
  sorry

end NUMINAMATH_GPT_original_time_40_l558_55848


namespace NUMINAMATH_GPT_exponentiation_addition_l558_55844

theorem exponentiation_addition : (3^3)^2 + 1 = 730 := by
  sorry

end NUMINAMATH_GPT_exponentiation_addition_l558_55844


namespace NUMINAMATH_GPT_mean_correct_and_no_seven_l558_55873

-- Define the set of numbers.
def numbers : List ℕ := 
  [8, 88, 888, 8888, 88888, 888888, 8888888, 88888888, 888888888]

-- Define the arithmetic mean of the numbers in the set.
def arithmetic_mean (l : List ℕ) : ℕ := (l.sum / l.length)

-- Specify the mean value
def mean_value : ℕ := 109629012

-- State the theorem that the mean value is correct and does not contain the digit 7.
theorem mean_correct_and_no_seven : arithmetic_mean numbers = mean_value ∧ ¬ 7 ∈ (mean_value.digits 10) :=
  sorry

end NUMINAMATH_GPT_mean_correct_and_no_seven_l558_55873


namespace NUMINAMATH_GPT_nuts_in_tree_l558_55893

def num_squirrels := 4
def num_nuts := 2

theorem nuts_in_tree :
  ∀ (S N : ℕ), S = num_squirrels → S - N = 2 → N = num_nuts :=
by
  intros S N hS hDiff
  sorry

end NUMINAMATH_GPT_nuts_in_tree_l558_55893


namespace NUMINAMATH_GPT_minimum_n_for_80_intersections_l558_55845

-- Define what an n-sided polygon is and define the intersection condition
def n_sided_polygon (n : ℕ) : Type := sorry -- definition of n-sided polygon

-- Define the condition when boundaries of two polygons intersect at exactly 80 points
def boundaries_intersect_at (P Q : n_sided_polygon n) (k : ℕ) : Prop := sorry -- definition of boundaries intersecting at exactly k points

theorem minimum_n_for_80_intersections (n : ℕ) :
  (∃ (P Q : n_sided_polygon n), boundaries_intersect_at P Q 80) → (n ≥ 10) :=
sorry

end NUMINAMATH_GPT_minimum_n_for_80_intersections_l558_55845


namespace NUMINAMATH_GPT_find_natural_pairs_l558_55881

theorem find_natural_pairs (m n : ℕ) :
  (n * (n - 1) * (n - 2) * (n - 3) = m * (m - 1)) ↔ (n = 1 ∧ m = 1) ∨ (n = 2 ∧ m = 1) ∨ (n = 3 ∧ m = 1) :=
by sorry

end NUMINAMATH_GPT_find_natural_pairs_l558_55881


namespace NUMINAMATH_GPT_concert_cost_l558_55880

-- Definitions of the given conditions
def ticket_price : ℝ := 50.00
def num_tickets : ℕ := 2
def processing_fee_rate : ℝ := 0.15
def parking_fee : ℝ := 10.00
def entrance_fee_per_person : ℝ := 5.00
def num_people : ℕ := 2

-- Function to compute the total cost
def total_cost : ℝ :=
  let ticket_total := num_tickets * ticket_price
  let processing_fee := processing_fee_rate * ticket_total
  let total_with_processing := ticket_total + processing_fee
  let total_with_parking := total_with_processing + parking_fee
  let entrance_fee_total := num_people * entrance_fee_per_person
  total_with_parking + entrance_fee_total

-- The proof statement
theorem concert_cost :
  total_cost = 135.00 :=
by
  -- Using the assumptions defined
  let ticket_total := num_tickets * ticket_price
  let processing_fee := processing_fee_rate * ticket_total
  let total_with_processing := ticket_total + processing_fee
  let total_with_parking := total_with_processing + parking_fee
  let entrance_fee_total := num_people * entrance_fee_per_person
  let final_total := total_with_parking + entrance_fee_total
  
  -- Proving the final total
  show final_total = 135.00
  sorry

end NUMINAMATH_GPT_concert_cost_l558_55880


namespace NUMINAMATH_GPT_solve_for_x_l558_55864

theorem solve_for_x : ∀ (x : ℂ) (i : ℂ), i^2 = -1 → 3 - 2 * i * x = 6 + i * x → x = i :=
by
  intros x i hI2 hEq
  sorry

end NUMINAMATH_GPT_solve_for_x_l558_55864


namespace NUMINAMATH_GPT_bert_toy_phones_l558_55885

theorem bert_toy_phones (P : ℕ) (berts_price_per_phone : ℕ) (berts_earning : ℕ)
                        (torys_price_per_gun : ℕ) (torys_earning : ℕ) (tory_guns : ℕ)
                        (earnings_difference : ℕ)
                        (h1 : berts_price_per_phone = 18)
                        (h2 : torys_price_per_gun = 20)
                        (h3 : tory_guns = 7)
                        (h4 : torys_earning = tory_guns * torys_price_per_gun)
                        (h5 : berts_earning = torys_earning + earnings_difference)
                        (h6 : earnings_difference = 4)
                        (h7 : P = berts_earning / berts_price_per_phone) :
  P = 8 := by sorry

end NUMINAMATH_GPT_bert_toy_phones_l558_55885


namespace NUMINAMATH_GPT_no_intersecting_axes_l558_55805

theorem no_intersecting_axes (m : ℝ) : (m^2 + 2 * m - 7 = 0) → m = -4 :=
sorry

end NUMINAMATH_GPT_no_intersecting_axes_l558_55805


namespace NUMINAMATH_GPT_opposite_of_neg_sqrt_two_l558_55809

theorem opposite_of_neg_sqrt_two : -(-Real.sqrt 2) = Real.sqrt 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_opposite_of_neg_sqrt_two_l558_55809


namespace NUMINAMATH_GPT_minimize_fence_perimeter_l558_55879

-- Define the area of the pen
def area (L W : ℝ) : ℝ := L * W

-- Define that only three sides of the fence need to be fenced
def perimeter (L W : ℝ) : ℝ := 2 * W + L

-- Given conditions
def A : ℝ := 54450  -- Area in square meters

-- The proof statement
theorem minimize_fence_perimeter :
  ∃ (L W : ℝ), 
  area L W = A ∧ 
  ∀ (L' W' : ℝ), area L' W' = A → perimeter L W ≤ perimeter L' W' ∧ L = 330 ∧ W = 165 :=
sorry

end NUMINAMATH_GPT_minimize_fence_perimeter_l558_55879


namespace NUMINAMATH_GPT_find_q_l558_55832

theorem find_q (p q : ℝ) (h : (-2)^3 - 2*(-2)^2 + p*(-2) + q = 0) : 
  q = 16 + 2 * p :=
sorry

end NUMINAMATH_GPT_find_q_l558_55832


namespace NUMINAMATH_GPT_third_term_of_sequence_l558_55882

theorem third_term_of_sequence :
  (3 - (1 / 3) = 8 / 3) :=
by
  sorry

end NUMINAMATH_GPT_third_term_of_sequence_l558_55882


namespace NUMINAMATH_GPT_handshake_even_acquaintance_l558_55801

theorem handshake_even_acquaintance (n : ℕ) (hn : n = 225) : 
  ∃ (k : ℕ), k < n ∧ (∀ m < n, k ≠ m) :=
by sorry

end NUMINAMATH_GPT_handshake_even_acquaintance_l558_55801


namespace NUMINAMATH_GPT_molecular_weight_l558_55830

noncomputable def molecular_weight_of_one_mole : ℕ → ℝ :=
  fun n => if n = 1 then 78 else n * 78

theorem molecular_weight (n: ℕ) (hn: n > 0) (condition: ∃ k: ℕ, k = 4 ∧ 312 = k * 78) :
  molecular_weight_of_one_mole n = 78 * n :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_l558_55830


namespace NUMINAMATH_GPT_present_age_of_B_l558_55895

theorem present_age_of_B (A B : ℕ) (h1 : A + 20 = 2 * (B - 20)) (h2 : A = B + 10) : B = 70 :=
by
  sorry

end NUMINAMATH_GPT_present_age_of_B_l558_55895


namespace NUMINAMATH_GPT_max_value_of_a_squared_b_squared_c_squared_l558_55868

theorem max_value_of_a_squared_b_squared_c_squared
  (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c)
  (h_constraint : a + 2 * b + 3 * c = 1) : a^2 + b^2 + c^2 ≤ 1 :=
sorry

end NUMINAMATH_GPT_max_value_of_a_squared_b_squared_c_squared_l558_55868


namespace NUMINAMATH_GPT_totalCandies_l558_55862

def bobCandies : Nat := 10
def maryCandies : Nat := 5
def sueCandies : Nat := 20
def johnCandies : Nat := 5
def samCandies : Nat := 10

theorem totalCandies : bobCandies + maryCandies + sueCandies + johnCandies + samCandies = 50 := 
by
  sorry

end NUMINAMATH_GPT_totalCandies_l558_55862


namespace NUMINAMATH_GPT_max_difference_y_coords_intersection_l558_55812

def f (x : ℝ) : ℝ := 4 - x^2 + x^3
def g (x : ℝ) : ℝ := x^2 + x^4

theorem max_difference_y_coords_intersection : ∀ x : ℝ, 
  (f x = g x) → 
  (∀ x₁ x₂ : ℝ, f x₁ = g x₁ ∧ f x₂ = g x₂ → |f x₁ - f x₂| = 0) := 
by
  sorry

end NUMINAMATH_GPT_max_difference_y_coords_intersection_l558_55812


namespace NUMINAMATH_GPT_saline_solution_concentration_l558_55818

theorem saline_solution_concentration
  (C : ℝ) -- concentration of the first saline solution
  (h1 : 3.6 * C + 1.4 * 9 = 5 * 3.24) : -- condition based on the total salt content
  C = 1 := 
sorry

end NUMINAMATH_GPT_saline_solution_concentration_l558_55818


namespace NUMINAMATH_GPT_distance_between_city_centers_l558_55846

def distance_on_map : ℝ := 45  -- Distance on the map in cm
def scale_factor : ℝ := 20     -- Scale factor (1 cm : 20 km)

theorem distance_between_city_centers : distance_on_map * scale_factor = 900 := by
  sorry

end NUMINAMATH_GPT_distance_between_city_centers_l558_55846


namespace NUMINAMATH_GPT_find_positive_value_of_A_l558_55853

variable (A : ℝ)

-- Given conditions
def relation (A B : ℝ) : ℝ := A^2 + B^2

-- The proof statement
theorem find_positive_value_of_A (h : relation A 7 = 200) : A = Real.sqrt 151 := sorry

end NUMINAMATH_GPT_find_positive_value_of_A_l558_55853


namespace NUMINAMATH_GPT_solve_for_x_l558_55887

theorem solve_for_x (x : ℚ) (h : 5 * x + 9 * x = 420 - 10 * (x - 4)) : 
  x = 115 / 6 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l558_55887


namespace NUMINAMATH_GPT_goldfinch_percentage_l558_55829

def number_of_goldfinches := 6
def number_of_sparrows := 9
def number_of_grackles := 5
def total_birds := number_of_goldfinches + number_of_sparrows + number_of_grackles
def goldfinch_fraction := (number_of_goldfinches : ℚ) / total_birds

theorem goldfinch_percentage : goldfinch_fraction * 100 = 30 := 
by
  sorry

end NUMINAMATH_GPT_goldfinch_percentage_l558_55829


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l558_55815

theorem solve_eq1 (x : ℝ):
  (x - 1) * (x + 3) = x - 1 ↔ x = 1 ∨ x = -2 :=
by 
  sorry

theorem solve_eq2 (x : ℝ):
  2 * x^2 - 6 * x = -3 ↔ x = (3 + Real.sqrt 3) / 2 ∨ x = (3 - Real.sqrt 3) / 2 :=
by 
  sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l558_55815


namespace NUMINAMATH_GPT_solve_remainder_problem_l558_55867

def remainder_problem : Prop :=
  ∃ (n : ℕ), 
    (n % 481 = 179) ∧ 
    (n % 752 = 231) ∧ 
    (n % 1063 = 359) ∧ 
    (((179 + 231 - 359) % 37) = 14)

theorem solve_remainder_problem : remainder_problem :=
by
  sorry

end NUMINAMATH_GPT_solve_remainder_problem_l558_55867


namespace NUMINAMATH_GPT_least_n_l558_55897

theorem least_n (n : ℕ) (h : 0 < n ∧ (1 / n - 1 / (n + 1) < 1 / 15)) : n ≥ 4 :=
by sorry

end NUMINAMATH_GPT_least_n_l558_55897


namespace NUMINAMATH_GPT_curved_surface_area_cone_l558_55875

-- Define the necessary values
def r := 8  -- radius of the base of the cone in centimeters
def l := 18 -- slant height of the cone in centimeters

-- Prove the curved surface area of the cone
theorem curved_surface_area_cone :
  (π * r * l = 144 * π) :=
by sorry

end NUMINAMATH_GPT_curved_surface_area_cone_l558_55875


namespace NUMINAMATH_GPT_final_price_after_discounts_l558_55847

noncomputable def initial_price : ℝ := 9795.3216374269
noncomputable def discount_20 (p : ℝ) : ℝ := p * 0.80
noncomputable def discount_10 (p : ℝ) : ℝ := p * 0.90
noncomputable def discount_5 (p : ℝ) : ℝ := p * 0.95

theorem final_price_after_discounts : discount_5 (discount_10 (discount_20 initial_price)) = 6700 := 
by
  sorry

end NUMINAMATH_GPT_final_price_after_discounts_l558_55847


namespace NUMINAMATH_GPT_card_total_l558_55860

theorem card_total (Brenda Janet Mara : ℕ)
  (h1 : Janet = Brenda + 9)
  (h2 : Mara = 2 * Janet)
  (h3 : Mara = 150 - 40) :
  Brenda + Janet + Mara = 211 := by
  sorry

end NUMINAMATH_GPT_card_total_l558_55860


namespace NUMINAMATH_GPT_simplify_expression_l558_55826

variable {R : Type*} [CommRing R] (x y : R)

theorem simplify_expression :
  (x - 2 * y) * (x + 2 * y) - x * (x - y) = -4 * y ^ 2 + x * y :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l558_55826


namespace NUMINAMATH_GPT_cost_per_bag_l558_55850

theorem cost_per_bag (total_bags : ℕ) (sale_price_per_bag : ℕ) (desired_profit : ℕ) (total_revenue : ℕ)
  (total_cost : ℕ) (cost_per_bag : ℕ) :
  total_bags = 100 → sale_price_per_bag = 10 → desired_profit = 300 →
  total_revenue = total_bags * sale_price_per_bag →
  total_cost = total_revenue - desired_profit →
  cost_per_bag = total_cost / total_bags →
  cost_per_bag = 7 := by
  sorry

end NUMINAMATH_GPT_cost_per_bag_l558_55850


namespace NUMINAMATH_GPT_units_digit_Fermat_5_l558_55889

def Fermat_number (n: ℕ) : ℕ :=
  2 ^ (2 ^ n) + 1

theorem units_digit_Fermat_5 : (Fermat_number 5) % 10 = 7 := by
  sorry

end NUMINAMATH_GPT_units_digit_Fermat_5_l558_55889


namespace NUMINAMATH_GPT_john_avg_increase_l558_55806

theorem john_avg_increase (a b c d : ℝ) (h₁ : a = 90) (h₂ : b = 85) (h₃ : c = 92) (h₄ : d = 95) :
    let initial_avg := (a + b + c) / 3
    let new_avg := (a + b + c + d) / 4
    new_avg - initial_avg = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_john_avg_increase_l558_55806


namespace NUMINAMATH_GPT_find_y_l558_55861

theorem find_y (x y : ℝ) (h1 : 0.5 * x = 0.25 * y - 30) (h2 : x = 690) : y = 1500 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l558_55861


namespace NUMINAMATH_GPT_minimum_value_function_l558_55852

theorem minimum_value_function (x : ℝ) (h : x > 1) : 
  ∃ y, y = (16 - 2 * Real.sqrt 7) / 3 ∧ ∀ x > 1, (4*x^2 + 2*x + 5) / (x^2 + x + 1) ≥ y :=
sorry

end NUMINAMATH_GPT_minimum_value_function_l558_55852
