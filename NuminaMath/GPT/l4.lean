import Mathlib

namespace determine_chris_age_l4_4390

theorem determine_chris_age (a b c : ℚ)
  (h1 : (a + b + c) / 3 = 10)
  (h2 : c - 5 = 2 * a)
  (h3 : b + 4 = (3 / 4) * (a + 4)) :
  c = 283 / 15 :=
by
  sorry

end determine_chris_age_l4_4390


namespace calculate_difference_of_squares_l4_4436

theorem calculate_difference_of_squares :
  (153^2 - 147^2) = 1800 :=
by
  sorry

end calculate_difference_of_squares_l4_4436


namespace construction_company_sand_weight_l4_4413

theorem construction_company_sand_weight :
  let gravel_weight := 5.91
  let total_material_weight := 14.02
  let sand_weight := total_material_weight - gravel_weight
  sand_weight = 8.11 :=
by
  let gravel_weight := 5.91
  let total_material_weight := 14.02
  let sand_weight := total_material_weight - gravel_weight
  -- Observing that 14.02 - 5.91 = 8.11
  have h : sand_weight = 8.11 := by sorry
  exact h

end construction_company_sand_weight_l4_4413


namespace factorization_l4_4687

theorem factorization (a x : ℝ) : ax^2 - 2ax + a = a * (x - 1) ^ 2 := 
by
  sorry

end factorization_l4_4687


namespace fraction_sum_is_integer_l4_4381

theorem fraction_sum_is_integer (n : ℤ) : 
  ∃ k : ℤ, (n / 3 + (n^2) / 2 + (n^3) / 6) = k := 
sorry

end fraction_sum_is_integer_l4_4381


namespace shortest_routes_l4_4552

def side_length : ℕ := 10
def refuel_distance : ℕ := 30
def num_squares_per_refuel := refuel_distance / side_length

theorem shortest_routes (A B : Type) (distance_AB : ℕ) (shortest_paths : Π (A B : Type), ℕ) : 
  shortest_paths A B = 54 := by
  sorry

end shortest_routes_l4_4552


namespace temperature_fifth_day_l4_4530

variable (T1 T2 T3 T4 T5 : ℝ)

-- Conditions
def condition1 : T1 + T2 + T3 + T4 = 4 * 58 := by sorry
def condition2 : T2 + T3 + T4 + T5 = 4 * 59 := by sorry
def condition3 : T5 = (8 / 7) * T1 := by sorry

-- The statement we need to prove
theorem temperature_fifth_day : T5 = 32 := by
  -- Using the provided conditions
  sorry

end temperature_fifth_day_l4_4530


namespace factorization_l4_4904

theorem factorization (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) :=
by 
  sorry

end factorization_l4_4904


namespace cheburashkas_erased_l4_4763

theorem cheburashkas_erased (n : ℕ) (rows : ℕ) (krakozyabras : ℕ) 
  (h_spacing : ∀ r, r ≤ rows → krakozyabras = 2 * (n - 1))
  (h_rows : rows = 2)
  (h_krakozyabras : krakozyabras = 29) :
  n = 16 → rows = 2 → krakozyabras = 29 → n = 16 - 5 :=
by
  sorry

end cheburashkas_erased_l4_4763


namespace solution_set_of_inequality_l4_4516

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 2 * x + 3 > 0} = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end solution_set_of_inequality_l4_4516


namespace lines_intersect_lines_parallel_lines_coincident_l4_4325

-- Define line equations
def l1 (m x y : ℝ) := (m + 2) * x + (m + 3) * y - 5 = 0
def l2 (m x y : ℝ) := 6 * x + (2 * m - 1) * y - 5 = 0

-- Prove conditions for intersection
theorem lines_intersect (m : ℝ) : ¬(m = -5 / 2 ∨ m = 4) ↔
  ∃ x y : ℝ, l1 m x y ∧ l2 m x y := sorry

-- Prove conditions for parallel lines
theorem lines_parallel (m : ℝ) : m = -5 / 2 ↔
  ∀ x y : ℝ, l1 m x y ∧ l2 m x y → l1 m x y → l2 m x y := sorry

-- Prove conditions for coincident lines
theorem lines_coincident (m : ℝ) : m = 4 ↔
  ∀ x y : ℝ, l1 m x y ↔ l2 m x y := sorry

end lines_intersect_lines_parallel_lines_coincident_l4_4325


namespace real_root_bound_l4_4295

noncomputable def P (x : ℝ) (n : ℕ) (ns : List ℕ) : ℝ :=
  1 + x^2 + x^5 + ns.foldr (λ n acc => x^n + acc) 0 + x^2008

theorem real_root_bound (n1 n2 : ℕ) (ns : List ℕ) (x : ℝ) :
  5 < n1 →
  List.Chain (λ a b => a < b) n1 (n2 :: ns) →
  n2 < 2008 →
  P x n1 (n2 :: ns) = 0 →
  x ≤ (1 - Real.sqrt 5) / 2 :=
sorry

end real_root_bound_l4_4295


namespace customer_payment_strawberries_watermelons_max_discount_value_l4_4616

-- Definitions for prices
def price_strawberries : ℕ := 60
def price_jingbai_pears : ℕ := 65
def price_watermelons : ℕ := 80
def price_peaches : ℕ := 90

-- Definition for condition on minimum purchase for promotion
def min_purchase_for_promotion : ℕ := 120

-- Definition for percentage Li Ming receives
def li_ming_percentage : ℕ := 80
def customer_percentage : ℕ := 100

-- Proof problem for part 1
theorem customer_payment_strawberries_watermelons (x : ℕ) (total_price : ℕ) :
  x = 10 →
  total_price = price_strawberries + price_watermelons →
  total_price >= min_purchase_for_promotion →
  total_price - x = 130 :=
  by sorry

-- Proof problem for part 2
theorem max_discount_value (m x : ℕ) :
  m >= min_purchase_for_promotion →
  (m - x) * li_ming_percentage / customer_percentage ≥ m * 7 / 10 →
  x ≤ m / 8 :=
  by sorry

end customer_payment_strawberries_watermelons_max_discount_value_l4_4616


namespace volume_of_prism_l4_4389

theorem volume_of_prism (a b c : ℝ) (h1 : a * b = 36) (h2 : a * c = 48) (h3 : b * c = 72) : a * b * c = 168 :=
by
  sorry

end volume_of_prism_l4_4389


namespace factorize_expression_l4_4694

theorem factorize_expression (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 :=
by sorry

end factorize_expression_l4_4694


namespace sin_330_value_l4_4011

noncomputable def sin_330 : ℝ := Real.sin (330 * Real.pi / 180)

theorem sin_330_value : sin_330 = -1/2 :=
by {
  sorry
}

end sin_330_value_l4_4011


namespace sum_odd_implies_parity_l4_4584

theorem sum_odd_implies_parity (a b c: ℤ) (h: (a + b + c) % 2 = 1) : (a^2 + b^2 - c^2 + 2 * a * b) % 2 = 1 := 
sorry

end sum_odd_implies_parity_l4_4584


namespace product_of_roots_cubic_l4_4184

theorem product_of_roots_cubic :
  let p := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  in (∃ r1 r2 r3 : ℝ, p r1 = 0 ∧ p r2 = 0 ∧ p r3 = 0 ∧ r1 * r2 * r3 = 50) := 
by 
  sorry

end product_of_roots_cubic_l4_4184


namespace complement_A_intersect_B_eq_l4_4615

def setA : Set ℝ := { x : ℝ | |x - 2| ≤ 2 }

def setB : Set ℝ := { y : ℝ | ∃ x : ℝ, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2 }

def A_intersect_B := setA ∩ setB

def complement (A : Set ℝ) : Set ℝ := { x : ℝ | x ∉ A }

theorem complement_A_intersect_B_eq {A : Set ℝ} {B : Set ℝ} 
  (hA : A = { x : ℝ | |x - 2| ≤ 2 })
  (hB : B = { y : ℝ | ∃ x : ℝ, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2 }) :
  complement (A ∩ B) = { x : ℝ | x ≠ 0 } :=
by
  sorry

end complement_A_intersect_B_eq_l4_4615


namespace prove_equation_C_l4_4859

theorem prove_equation_C (m : ℝ) : -(m - 2) = -m + 2 := 
  sorry

end prove_equation_C_l4_4859


namespace how_many_cheburashkas_erased_l4_4773

theorem how_many_cheburashkas_erased 
  (total_krakozyabras : ℕ)
  (characters_per_row_initial : ℕ) 
  (total_characters_initial : ℕ)
  (total_cheburashkas : ℕ)
  (total_rows : ℕ := 2)
  (total_krakozyabras := 29) :
  total_cheburashkas = 11 :=
by
  sorry

end how_many_cheburashkas_erased_l4_4773


namespace minimum_value_of_expression_l4_4909

variable (a b c d : ℝ)

-- The given conditions:
def cond1 : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0
def cond2 : Prop := a^2 + b^2 = 4
def cond3 : Prop := c * d = 1

-- The minimum value:
def expression_value : ℝ := (a^2 * c^2 + b^2 * d^2) * (b^2 * c^2 + a^2 * d^2)

theorem minimum_value_of_expression :
  cond1 a b c d → cond2 a b → cond3 c d → expression_value a b c d ≥ 16 :=
by
  sorry

end minimum_value_of_expression_l4_4909


namespace sample_size_is_150_l4_4619

-- Define the conditions
def total_parents : ℕ := 823
def sampled_parents : ℕ := 150
def negative_attitude_parents : ℕ := 136

-- State the theorem
theorem sample_size_is_150 : sampled_parents = 150 := 
by
  sorry

end sample_size_is_150_l4_4619


namespace trendy_haircut_cost_l4_4151

theorem trendy_haircut_cost (T : ℝ) (H1 : 5 * 5 * 7 + 3 * 6 * 7 + 2 * T * 7 = 413) : T = 8 :=
by linarith

end trendy_haircut_cost_l4_4151


namespace sin_330_value_l4_4013

noncomputable def sin_330 : ℝ := Real.sin (330 * Real.pi / 180)

theorem sin_330_value : sin_330 = -1/2 :=
by {
  sorry
}

end sin_330_value_l4_4013


namespace alice_age_2005_l4_4427

-- Definitions
variables (x : ℕ) (age_Alice_2000 age_Grandmother_2000 : ℕ)
variables (born_Alice born_Grandmother : ℕ)

-- Conditions
def alice_grandmother_relation_at_2000 := age_Alice_2000 = x ∧ age_Grandmother_2000 = 3 * x
def birth_year_sum := born_Alice + born_Grandmother = 3870
def birth_year_Alice := born_Alice = 2000 - x
def birth_year_Grandmother := born_Grandmother = 2000 - 3 * x

-- Proving the main statement: Alice's age at the end of 2005
theorem alice_age_2005 : 
  alice_grandmother_relation_at_2000 x age_Alice_2000 age_Grandmother_2000 ∧ 
  birth_year_sum born_Alice born_Grandmother ∧ 
  birth_year_Alice x born_Alice ∧ 
  birth_year_Grandmother x born_Grandmother 
  → 2005 - 2000 + age_Alice_2000 = 37 := 
by 
  intros
  sorry

end alice_age_2005_l4_4427


namespace books_per_author_l4_4608

theorem books_per_author (total_books : ℕ) (authors : ℕ) (h1 : total_books = 198) (h2 : authors = 6) : total_books / authors = 33 :=
by sorry

end books_per_author_l4_4608


namespace find_smallest_natural_number_l4_4565

theorem find_smallest_natural_number :
  ∃ x : ℕ, (2 * x = b^2 ∧ 3 * x = c^3) ∧ (∀ y : ℕ, (2 * y = d^2 ∧ 3 * y = e^3) → x ≤ y) := by
  sorry

end find_smallest_natural_number_l4_4565


namespace assign_parents_l4_4168

variable (V E : Type) [Fintype V] [Fintype E]
variable (G : SimpleGraph V) -- Graph with vertices V and edges between vertices E.

-- Conditions:
def condition1 : Prop :=
  ∀ {a b : V}, G.adj a b ∨ a = b ∨ ¬(G.adj a b)

def condition2 : Prop :=
  ∀ {a b c : V}, G.adj a b ∧ G.adj b c ∧ G.adj a c → (¬G.adj a b ∨ ¬G.adj b c ∨ ¬G.adj a c) ∨ (G.adj a b ∧ G.adj b c ∧ G.adj a c)

-- Theorem: 
theorem assign_parents (G : SimpleGraph V) [Fintype V] [DecidableRel G.adj] (cond1 : condition1 G) (cond2 : condition2 G) :
  ∃ (P : V → set V), 
    (∀ {u v : V}, G.adj u v → ∃ p, p ∈ P u ∧ p ∈ P v) ∧ 
    (∀ {u v : V}, ¬G.adj u v → ∃ p, p ∈ P u ∨ p ∈ P v) ∧ 
    (∀ {u v w : V}, (u ≠ v ∧ v ≠ w ∧ u ≠ w) → ∃ p1 p2 p3, (p1 ∈ P u ∧ p2 ∈ P v ∧ p3 ∈ P w)) :=
sorry

end assign_parents_l4_4168


namespace ratio_cost_price_selling_price_l4_4104

theorem ratio_cost_price_selling_price (CP SP : ℝ) (h : SP = 1.5 * CP) : CP / SP = 2 / 3 :=
by
  sorry

end ratio_cost_price_selling_price_l4_4104


namespace total_income_l4_4968

def ron_ticket_price : ℝ := 2.00
def kathy_ticket_price : ℝ := 4.50
def total_tickets : ℕ := 20
def ron_tickets_sold : ℕ := 12

theorem total_income : ron_tickets_sold * ron_ticket_price + (total_tickets - ron_tickets_sold) * kathy_ticket_price = 60.00 := by
  sorry

end total_income_l4_4968


namespace solution_set_of_inequality_l4_4258

theorem solution_set_of_inequality (x : ℝ) :
  (x + 1) * (2 - x) < 0 ↔ (x > 2 ∨ x < -1) :=
by
  sorry

end solution_set_of_inequality_l4_4258


namespace right_triangle_area_l4_4826

theorem right_triangle_area (a b c : ℝ) (h : a^2 + b^2 = c^2) (ha : a = 5) (hc : c = 13) :
  1/2 * a * b = 30 :=
by
  have hb : b = 12, from sorry,
  -- Proof needs to be filled here
  sorry

end right_triangle_area_l4_4826


namespace cheburashkas_erased_l4_4764

theorem cheburashkas_erased (n : ℕ) (rows : ℕ) (krakozyabras : ℕ) 
  (h_spacing : ∀ r, r ≤ rows → krakozyabras = 2 * (n - 1))
  (h_rows : rows = 2)
  (h_krakozyabras : krakozyabras = 29) :
  n = 16 → rows = 2 → krakozyabras = 29 → n = 16 - 5 :=
by
  sorry

end cheburashkas_erased_l4_4764


namespace pebbles_difference_l4_4883

def candy_pebbles : Nat := 4
def lance_pebbles : Nat := 3 * candy_pebbles

theorem pebbles_difference {candy_pebbles lance_pebbles : Nat} (h1 : candy_pebbles = 4) (h2 : lance_pebbles = 3 * candy_pebbles) : lance_pebbles - candy_pebbles = 8 := by
  sorry

end pebbles_difference_l4_4883


namespace base_n_representation_l4_4931

theorem base_n_representation (n : ℕ) (b : ℕ) (h₀ : 8 < n) (h₁ : ∃ b, (n : ℤ)^2 - (n+8) * (n : ℤ) + b = 0) : 
  b = 8 * n :=
by
  sorry

end base_n_representation_l4_4931


namespace no_prime_solutions_l4_4043

theorem no_prime_solutions (p q : ℕ) (hp : p > 5) (hq : q > 5) (pp : Nat.Prime p) (pq : Nat.Prime q)
  (h : p * q ∣ (5^p - 2^p) * (5^q - 2^q)) : False :=
sorry

end no_prime_solutions_l4_4043


namespace row_sum_1005_equals_20092_l4_4089

theorem row_sum_1005_equals_20092 :
  let row := 1005
  let n := row
  let first_element := n
  let num_elements := 2 * n - 1
  let last_element := first_element + (num_elements - 1)
  let sum_row := num_elements * (first_element + last_element) / 2
  sum_row = 20092 :=
by
  sorry

end row_sum_1005_equals_20092_l4_4089


namespace joan_initial_books_l4_4946

variable (books_sold : ℕ)
variable (books_left : ℕ)

theorem joan_initial_books (h1 : books_sold = 26) (h2 : books_left = 7) : books_sold + books_left = 33 := by
  sorry

end joan_initial_books_l4_4946


namespace relationship_between_u_and_v_l4_4605

variables {r u v p : ℝ}
variables (AB G : ℝ)

theorem relationship_between_u_and_v (hAB : AB = 2 * r) (hAG_GF : u = (p^2 / (2 * r)) - p) :
    v^2 = u^3 / (2 * r - u) :=
sorry

end relationship_between_u_and_v_l4_4605


namespace cheburashkas_erased_l4_4779

def total_krakozyabras : ℕ := 29

def total_rows : ℕ := 2

def cheburashkas_per_row := (total_krakozyabras + total_rows) / total_rows / 2 + 1

theorem cheburashkas_erased :
  (total_krakozyabras + total_rows) / total_rows / 2 - 1 = 11 := 
by
  sorry

-- cheburashkas_erased proves that the number of Cheburashkas erased is 11 from the given conditions.

end cheburashkas_erased_l4_4779


namespace meaningful_sqrt_range_l4_4740

theorem meaningful_sqrt_range (x : ℝ) (h : 1 - x ≥ 0) : x ≤ 1 :=
sorry

end meaningful_sqrt_range_l4_4740


namespace sin_pi_plus_alpha_l4_4205

open Real

-- Define the given conditions
variable (α : ℝ) (hα1 : sin (π / 2 + α) = 3 / 5) (hα2 : 0 < α ∧ α < π / 2)

-- The theorem statement that must be proved
theorem sin_pi_plus_alpha : sin (π + α) = -4 / 5 :=
by
  sorry

end sin_pi_plus_alpha_l4_4205


namespace convex_polygon_sides_l4_4741

theorem convex_polygon_sides (n : ℕ) (h1 : 180 * (n - 2) - 90 = 2790) : n = 18 :=
sorry

end convex_polygon_sides_l4_4741


namespace relationship_among_a_b_c_l4_4729

theorem relationship_among_a_b_c (a b c : ℝ) (h₁ : a = 0.09) (h₂ : -2 < b ∧ b < -1) (h₃ : 1 < c ∧ c < 2) : b < a ∧ a < c := 
by 
  -- proof will involve but we only need to state this
  sorry

end relationship_among_a_b_c_l4_4729


namespace max_regions_divided_l4_4344

theorem max_regions_divided (n m : ℕ) (h_n : n = 10) (h_m : m = 4) (h_m_le_n : m ≤ n) : 
  ∃ r : ℕ, r = 50 :=
by
  have non_parallel_lines := n - m
  have regions_non_parallel := (non_parallel_lines * (non_parallel_lines + 1)) / 2 + 1
  have regions_parallel := m * non_parallel_lines + m
  have total_regions := regions_non_parallel + regions_parallel
  use total_regions
  sorry

end max_regions_divided_l4_4344


namespace cost_of_55_lilies_l4_4673

-- Define the problem conditions
def price_per_dozen_lilies (p : ℝ) : Prop :=
  p * 24 = 30

def directly_proportional_price (p : ℝ) (n : ℕ) : ℝ :=
  p * n

-- State the problem to prove the cost of a 55 lily bouquet
theorem cost_of_55_lilies (p : ℝ) (c : ℝ) :
  price_per_dozen_lilies p →
  c = directly_proportional_price p 55 →
  c = 68.75 :=
by
  sorry

end cost_of_55_lilies_l4_4673


namespace book_pairs_count_l4_4593

theorem book_pairs_count :
  let mystery_books := 4
  let science_fiction_books := 4
  let historical_books := 4
  (mystery_books + science_fiction_books + historical_books) = 12 ∧ 
  (mystery_books = 4 ∧ science_fiction_books = 4 ∧ historical_books = 4) →
  let genres := 3
  ∃ pairs, pairs = 48 :=
by
  sorry

end book_pairs_count_l4_4593


namespace perfect_square_condition_l4_4737

noncomputable def isPerfectSquareQuadratic (m : ℤ) (x y : ℤ) :=
  ∃ (k : ℤ), (4 * x^2 + m * x * y + 25 * y^2) = k^2

theorem perfect_square_condition (m : ℤ) :
  (∀ x y : ℤ, isPerfectSquareQuadratic m x y) → (m = 20 ∨ m = -20) :=
by
  sorry

end perfect_square_condition_l4_4737


namespace max_value_of_E_l4_4976

def E (a b c d : ℝ) : ℝ := a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_value_of_E :
  ∀ (a b c d : ℝ),
    (-8.5 ≤ a ∧ a ≤ 8.5) →
    (-8.5 ≤ b ∧ b ≤ 8.5) →
    (-8.5 ≤ c ∧ c ≤ 8.5) →
    (-8.5 ≤ d ∧ d ≤ 8.5) →
    E a b c d ≤ 306 := sorry

end max_value_of_E_l4_4976


namespace length_of_other_train_is_correct_l4_4277

noncomputable def length_of_second_train 
  (length_first_train : ℝ) 
  (speed_first_train : ℝ) 
  (speed_second_train : ℝ) 
  (time_to_cross : ℝ) 
  : ℝ := 
  let speed_first_train_m_s := speed_first_train * (1000 / 3600)
  let speed_second_train_m_s := speed_second_train * (1000 / 3600)
  let relative_speed := speed_first_train_m_s + speed_second_train_m_s
  let total_distance := relative_speed * time_to_cross
  total_distance - length_first_train

theorem length_of_other_train_is_correct :
  length_of_second_train 250 120 80 9 = 249.95 :=
by
  unfold length_of_second_train
  simp
  sorry

end length_of_other_train_is_correct_l4_4277


namespace caps_eaten_correct_l4_4949

def initial_bottle_caps : ℕ := 34
def remaining_bottle_caps : ℕ := 26
def eaten_bottle_caps (k_i k_r : ℕ) : ℕ := k_i - k_r

theorem caps_eaten_correct :
  eaten_bottle_caps initial_bottle_caps remaining_bottle_caps = 8 :=
by
  sorry

end caps_eaten_correct_l4_4949


namespace jade_statue_ratio_l4_4488

/-!
Nancy carves statues out of jade. A giraffe statue takes 120 grams of jade and sells for $150.
An elephant statue sells for $350. Nancy has 1920 grams of jade, and the revenue from selling all
elephant statues is $400 more than selling all giraffe statues.
Prove that the ratio of the amount of jade used for an elephant statue to the amount used for a
giraffe statue is 2.
-/

theorem jade_statue_ratio
  (g_grams : ℕ := 120) -- grams of jade for a giraffe statue
  (g_price : ℕ := 150) -- price of a giraffe statue
  (e_price : ℕ := 350) -- price of an elephant statue
  (total_jade : ℕ := 1920) -- total grams of jade Nancy has
  (additional_revenue : ℕ := 400) -- additional revenue from elephant statues
  (r : ℕ) -- ratio of jade usage of elephant to giraffe statue
  (h : total_jade / g_grams * g_price + additional_revenue = (total_jade / (g_grams * r)) * e_price) :
  r = 2 :=
sorry

end jade_statue_ratio_l4_4488


namespace egor_last_payment_l4_4195

theorem egor_last_payment (a b c d : ℕ) (h_sum : a + b + c + d = 28)
  (h1 : b ≥ 2 * a) (h2 : c ≥ 2 * b) (h3 : d ≥ 2 * c) : d = 18 := by
  sorry

end egor_last_payment_l4_4195


namespace factorization_correct_l4_4700

-- Define the initial expression
def initial_expr (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + a

-- Define the factorized form
def factorized_expr (a x : ℝ) : ℝ := a * (x - 1)^2

-- Create the theorem statement
theorem factorization_correct (a x : ℝ) : initial_expr a x = factorized_expr a x :=
by simp [initial_expr, factorized_expr, pow_two, mul_add, add_mul, add_assoc]; sorry

end factorization_correct_l4_4700


namespace length_of_platform_l4_4142

theorem length_of_platform (L : ℕ) :
  (∀ (V : ℚ), V = 600 / 52 → V = (600 + L) / 78) → L = 300 :=
by
  sorry

end length_of_platform_l4_4142


namespace min_value_of_x_plus_2y_l4_4911

theorem min_value_of_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : 1 / x + 1 / y = 2) : 
  x + 2 * y ≥ (3 + 2 * Real.sqrt 2) / 2 :=
sorry

end min_value_of_x_plus_2y_l4_4911


namespace length_of_second_train_l4_4299

/-
  Given:
  - l₁ : Length of the first train in meters
  - v₁ : Speed of the first train in km/h
  - v₂ : Speed of the second train in km/h
  - t : Time to cross the second train in seconds

  Prove:
  - l₂ : Length of the second train in meters = 299.9560035197185 meters
-/

variable (l₁ : ℝ) (v₁ : ℝ) (v₂ : ℝ) (t : ℝ) (l₂ : ℝ)

theorem length_of_second_train
  (h₁ : l₁ = 250)
  (h₂ : v₁ = 72)
  (h₃ : v₂ = 36)
  (h₄ : t = 54.995600351971845)
  (h_result : l₂ = 299.9560035197185) :
  (v₁ * 1000 / 3600 - v₂ * 1000 / 3600) * t - l₁ = l₂ := by
  sorry

end length_of_second_train_l4_4299


namespace Debby_jogging_plan_l4_4490

def Monday_jog : ℝ := 3
def Tuesday_jog : ℝ := Monday_jog * 1.1
def Wednesday_jog : ℝ := 0
def Thursday_jog : ℝ := Tuesday_jog * 1.1
def Saturday_jog : ℝ := Thursday_jog * 2.5
def total_distance : ℝ := Monday_jog + Tuesday_jog + Thursday_jog + Saturday_jog
def weekly_goal : ℝ := 40
def Sunday_jog : ℝ := weekly_goal - total_distance

theorem Debby_jogging_plan :
  Tuesday_jog = 3.3 ∧
  Thursday_jog = 3.63 ∧
  Saturday_jog = 9.075 ∧
  Sunday_jog = 21.995 :=
by
  -- Proof goes here, but is omitted as the problem statement requires only the theorem outline.
  sorry

end Debby_jogging_plan_l4_4490


namespace total_chairs_taken_l4_4368

def num_students : ℕ := 5
def chairs_per_trip : ℕ := 5
def num_trips : ℕ := 10

theorem total_chairs_taken :
  (num_students * chairs_per_trip * num_trips) = 250 :=
by
  sorry

end total_chairs_taken_l4_4368


namespace sin_neg_60_eq_neg_sqrt_3_div_2_l4_4170

theorem sin_neg_60_eq_neg_sqrt_3_div_2 : 
  Real.sin (-π / 3) = - (Real.sqrt 3) / 2 := 
by
  sorry

end sin_neg_60_eq_neg_sqrt_3_div_2_l4_4170


namespace marble_arrangement_mod_l4_4493

def num_ways_arrange_marbles (m : ℕ) : ℕ := Nat.choose (m + 3) 3

theorem marble_arrangement_mod (N : ℕ) (m : ℕ) (h1: m = 11) (h2: N = num_ways_arrange_marbles m): 
  N % 1000 = 35 := by
  sorry

end marble_arrangement_mod_l4_4493


namespace mark_paired_with_mike_prob_l4_4237

def total_students := 16
def other_students := 15
def prob_pairing (mark: Nat) (mike: Nat) : ℚ := 1 / other_students

theorem mark_paired_with_mike_prob : prob_pairing 1 2 = 1 / 15 := 
sorry

end mark_paired_with_mike_prob_l4_4237


namespace sin_330_value_l4_4009

noncomputable def sin_330 : ℝ := Real.sin (330 * Real.pi / 180)

theorem sin_330_value : sin_330 = -1/2 :=
by {
  sorry
}

end sin_330_value_l4_4009


namespace sum_of_first_8_terms_l4_4220

theorem sum_of_first_8_terms (seq : ℕ → ℝ) (q : ℝ) (h_q : q = 2) 
  (h_sum_first_4 : seq 0 + seq 1 + seq 2 + seq 3 = 1) 
  (h_geom : ∀ n, seq (n + 1) = q * seq n) : 
  seq 0 + seq 1 + seq 2 + seq 3 + seq 4 + seq 5 + seq 6 + seq 7 = 17 := 
sorry

end sum_of_first_8_terms_l4_4220


namespace functional_equation_solution_l4_4196

noncomputable def f (x : ℝ) : ℝ := sorry

theorem functional_equation_solution : 
  (∀ x y : ℝ, f (⌊x⌋ * y) = f x * ⌊f y⌋) 
  → ∃ c : ℝ, (c = 0 ∨ (1 ≤ c ∧ c < 2)) ∧ (∀ x : ℝ, f x = c) :=
by
  intro h
  sorry

end functional_equation_solution_l4_4196


namespace doll_cost_is_one_l4_4424

variable (initial_amount : ℕ) (end_amount : ℕ) (number_of_dolls : ℕ)

-- Conditions
def given_conditions : Prop :=
  initial_amount = 100 ∧
  end_amount = 97 ∧
  number_of_dolls = 3

-- Question: Proving the cost of each doll
def cost_per_doll (initial_amount end_amount number_of_dolls : ℕ) : ℕ :=
  (initial_amount - end_amount) / number_of_dolls

theorem doll_cost_is_one (h : given_conditions initial_amount end_amount number_of_dolls) :
  cost_per_doll initial_amount end_amount number_of_dolls = 1 :=
by
  sorry

end doll_cost_is_one_l4_4424


namespace max_value_2ab_plus_2bc_sqrt2_l4_4234

theorem max_value_2ab_plus_2bc_sqrt2 (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a^2 + b^2 + c^2 = 1) :
  2 * a * b + 2 * b * c * Real.sqrt 2 ≤ Real.sqrt 3 :=
sorry

end max_value_2ab_plus_2bc_sqrt2_l4_4234


namespace cassidy_grounded_days_l4_4434

-- Definitions for the conditions
def days_for_lying : Nat := 14
def extra_days_per_grade : Nat := 3
def grades_below_B : Nat := 4

-- Definition for the total days grounded
def total_days_grounded : Nat :=
  days_for_lying + extra_days_per_grade * grades_below_B

-- The theorem statement
theorem cassidy_grounded_days :
  total_days_grounded = 26 := by
  sorry

end cassidy_grounded_days_l4_4434


namespace captain_age_l4_4744

noncomputable def whole_team_age : ℕ := 253
noncomputable def remaining_players_age : ℕ := 198
noncomputable def captain_and_wicket_keeper_age : ℕ := whole_team_age - remaining_players_age
noncomputable def wicket_keeper_age (C : ℕ) : ℕ := C + 3

theorem captain_age (C : ℕ) (whole_team : whole_team_age = 11 * 23) (remaining_players : remaining_players_age = 9 * 22) 
    (sum_ages : captain_and_wicket_keeper_age = 55) (wicket_keeper : wicket_keeper_age C = C + 3) : C = 26 := 
  sorry

end captain_age_l4_4744


namespace sin_330_eq_neg_half_l4_4033

-- Define conditions as hypotheses in Lean
def angle_330 (θ : ℝ) : Prop := θ = 330
def angle_transform (θ : ℝ) : Prop := θ = 360 - 30
def sin_pos (θ : ℝ) : Prop := Real.sin θ = 1 / 2
def sin_neg_in_4th_quadrant (θ : ℝ) : Prop := θ = 330 -> Real.sin θ < 0

-- The main theorem statement
theorem sin_330_eq_neg_half : ∀ θ : ℝ, angle_330 θ → angle_transform θ → sin_pos 30 → sin_neg_in_4th_quadrant θ → Real.sin θ = -1 / 2 := by
  intro θ h1 h2 h3 h4
  sorry

end sin_330_eq_neg_half_l4_4033


namespace usual_time_is_75_l4_4265

variable (T : ℕ) -- let T be the usual time in minutes

theorem usual_time_is_75 (h1 : (6 * T) / 5 = T + 15) : T = 75 :=
by
  sorry

end usual_time_is_75_l4_4265


namespace most_consistent_player_l4_4281

section ConsistentPerformance

variables (σA σB σC σD : ℝ)
variables (σA_eq : σA = 0.023)
variables (σB_eq : σB = 0.018)
variables (σC_eq : σC = 0.020)
variables (σD_eq : σD = 0.021)

theorem most_consistent_player : σB < σC ∧ σB < σD ∧ σB < σA :=
by 
  rw [σA_eq, σB_eq, σC_eq, σD_eq]
  sorry

end ConsistentPerformance

end most_consistent_player_l4_4281


namespace solve_for_x_l4_4652

theorem solve_for_x :
  ∃ x : ℝ, 40 + (5 * x) / (180 / 3) = 41 ∧ x = 12 :=
by
  sorry

end solve_for_x_l4_4652


namespace value_of_a_l4_4494

theorem value_of_a (a : ℝ) (h : abs (2 * a + 1) = 3) :
  a = -2 ∨ a = 1 :=
sorry

end value_of_a_l4_4494


namespace main_theorem_l4_4439

noncomputable def f : ℝ → ℝ := sorry

axiom h_even : ∀ x : ℝ, f (-x) = f x
axiom h_decreasing : ∀ x1 x2 : ℝ, x1 ≠ x2 → 0 ≤ x1 → 0 ≤ x2 → x1 ≠ x2 → 
  (x1 < x2 ↔ (f x2 < f x1))

theorem main_theorem : f 3 < f (-2) ∧ f (-2) < f 1 :=
by
  sorry

end main_theorem_l4_4439


namespace right_triangle_area_l4_4823

theorem right_triangle_area
  (hypotenuse : ℝ) (leg1 : ℝ) (leg2 : ℝ)
  (hypotenuse_eq : hypotenuse = 13)
  (leg1_eq : leg1 = 5)
  (pythagorean_eq : hypotenuse^2 = leg1^2 + leg2^2) :
  (1 / 2) * leg1 * leg2 = 30 :=
by
  sorry

end right_triangle_area_l4_4823


namespace factor_problem_l4_4107

theorem factor_problem 
  (a b : ℕ) (h1 : a > b)
  (h2 : (∀ x, x^2 - 16 * x + 64 = (x - a) * (x - b))) 
  : 3 * b - a = 16 := by
  sorry

end factor_problem_l4_4107


namespace lcm_18_24_30_eq_360_l4_4524

-- Define the three numbers in the condition
def a : ℕ := 18
def b : ℕ := 24
def c : ℕ := 30

-- State the theorem to prove
theorem lcm_18_24_30_eq_360 : Nat.lcm a (Nat.lcm b c) = 360 :=
by 
  sorry -- Proof is omitted as per instructions

end lcm_18_24_30_eq_360_l4_4524


namespace product_of_roots_cubic_l4_4171

theorem product_of_roots_cubic :
  ∀ (x : ℝ), (x^3 - 15 * x^2 + 75 * x - 50 = 0) →
    (∃ a b c : ℝ, x = a * b * c ∧ x = 50) :=
by
  sorry

end product_of_roots_cubic_l4_4171


namespace sum_max_min_ratios_l4_4682

theorem sum_max_min_ratios
  (c d : ℚ)
  (h1 : ∀ x y : ℚ, 3*x^2 + 2*x*y + 4*y^2 - 13*x - 26*y + 53 = 0 → y / x = c ∨ y / x = d)
  (h2 : ∀ r : ℚ, (∃ x y : ℚ, 3*x^2 + 2*x*y + 4*y^2 - 13*x - 26*y + 53 = 0 ∧ y / x = r) → (r = c ∨ r = d))
  : c + d = 63 / 43 :=
sorry

end sum_max_min_ratios_l4_4682


namespace geometric_sequence_t_value_l4_4915

theorem geometric_sequence_t_value (S : ℕ → ℝ) (a : ℕ → ℝ) (t : ℝ) :
  (∀ n, S n = t * 5^n - 2) → 
  (∀ n ≥ 1, a (n + 1) = S (n + 1) - S n) → 
  (a 1 ≠ 0) → -- Ensure the sequence is non-trivial.
  (∀ n, a (n + 1) / a n = 5) → 
  t = 5 := 
by 
  intros h1 h2 h3 h4
  sorry

end geometric_sequence_t_value_l4_4915


namespace part_a_part_b_l4_4410

def good (p q n : ℕ) : Prop :=
  ∃ x y : ℕ, n = p * x + q * y

def bad (p q n : ℕ) : Prop := 
  ¬ good p q n

theorem part_a (p q : ℕ) (h : Nat.gcd p q = 1) : ∃ A, A = p * q - p - q ∧ ∀ x y, x + y = A → (good p q x ∧ bad p q y) ∨ (bad p q x ∧ good p q y) := by
  sorry

theorem part_b (p q : ℕ) (h : Nat.gcd p q = 1) : ∃ N, N = (p - 1) * (q - 1) / 2 ∧ ∀ n, n < p * q - p - q → bad p q n :=
  sorry

end part_a_part_b_l4_4410


namespace angle_C_of_triangle_l4_4754

theorem angle_C_of_triangle (A B C : ℝ) (hA : A = 90) (hB : B = 50) (h_sum : A + B + C = 180) : C = 40 := 
by
  sorry

end angle_C_of_triangle_l4_4754


namespace evaluate_f_at_5_l4_4061

def f (x : ℝ) := 2 * x^5 - 5 * x^4 - 4 * x^3 + 3 * x^2 - 524

theorem evaluate_f_at_5 : f 5 = 2176 :=
by
  sorry

end evaluate_f_at_5_l4_4061


namespace units_digit_fraction_mod_10_l4_4853

theorem units_digit_fraction_mod_10 : (30 * 32 * 34 * 36 * 38 * 40) % 2000 % 10 = 2 := by
  sorry

end units_digit_fraction_mod_10_l4_4853


namespace probability_three_of_one_two_of_other_l4_4411

theorem probability_three_of_one_two_of_other :
  let total_balls := 20
  let draw_balls := 5
  let black_balls := 10
  let white_balls := 10
  let total_outcomes := Nat.choose total_balls draw_balls
  let favorable_outcomes_black3_white2 := Nat.choose black_balls 3 * Nat.choose white_balls 2
  let favorable_outcomes_black2_white3 := Nat.choose black_balls 2 * Nat.choose white_balls 3
  let favorable_outcomes := favorable_outcomes_black3_white2 + favorable_outcomes_black2_white3
  let probability := favorable_outcomes / total_outcomes
  probability = (30 : ℚ) / 43 :=
by 
  sorry

end probability_three_of_one_two_of_other_l4_4411


namespace largest_angle_measures_203_l4_4664

-- Define the angles of the hexagon
def angle1 (x : ℚ) : ℚ := x + 2
def angle2 (x : ℚ) : ℚ := 2 * x + 1
def angle3 (x : ℚ) : ℚ := 3 * x
def angle4 (x : ℚ) : ℚ := 4 * x - 1
def angle5 (x : ℚ) : ℚ := 5 * x + 2
def angle6 (x : ℚ) : ℚ := 6 * x - 2

-- Define the sum of interior angles for a hexagon
def hexagon_angle_sum : ℚ := 720

-- Prove that the largest angle is equal to 203 degrees given the conditions
theorem largest_angle_measures_203 (x : ℚ) (h : angle1 x + angle2 x + angle3 x + angle4 x + angle5 x + angle6 x = hexagon_angle_sum) :
  (6 * x - 2) = 203 := by
  sorry

end largest_angle_measures_203_l4_4664


namespace only_solution_is_two_l4_4314

theorem only_solution_is_two :
  ∀ n : ℕ, (Nat.Prime (n^n + 1) ∧ Nat.Prime ((2*n)^(2*n) + 1)) → n = 2 :=
by
  sorry

end only_solution_is_two_l4_4314


namespace janice_total_hours_worked_l4_4360

-- Declare the conditions as definitions
def hourly_rate_first_40_hours : ℝ := 10
def hourly_rate_overtime : ℝ := 15
def first_40_hours : ℕ := 40
def total_pay : ℝ := 700

-- Define the main theorem
theorem janice_total_hours_worked (H : ℕ) (O : ℕ) : 
  H = first_40_hours + O ∧ (hourly_rate_first_40_hours * first_40_hours + hourly_rate_overtime * O = total_pay) → H = 60 :=
by
  sorry

end janice_total_hours_worked_l4_4360


namespace square_units_digit_eq_9_l4_4113

/-- The square of which whole number has a units digit of 9? -/
theorem square_units_digit_eq_9 (n : ℕ) (h : ∃ m : ℕ, n = m^2 ∧ m % 10 = 9) : n = 3 ∨ n = 7 := by
  sorry

end square_units_digit_eq_9_l4_4113


namespace greatest_q_minus_r_l4_4975

theorem greatest_q_minus_r :
  ∃ q r : ℤ, q > 0 ∧ r > 0 ∧ 975 = 23 * q + r ∧ q - r = 33 := sorry

end greatest_q_minus_r_l4_4975


namespace daughters_and_granddaughters_without_daughters_l4_4483

-- Given conditions
def melissa_daughters : ℕ := 10
def half_daughters_with_children : ℕ := melissa_daughters / 2
def grandchildren_per_daughter : ℕ := 4
def total_descendants : ℕ := 50

-- Calculations based on given conditions
def number_of_granddaughters : ℕ := total_descendants - melissa_daughters
def daughters_with_no_children : ℕ := melissa_daughters - half_daughters_with_children
def granddaughters_with_no_children : ℕ := number_of_granddaughters

-- The final result we need to prove
theorem daughters_and_granddaughters_without_daughters : 
  daughters_with_no_children + granddaughters_with_no_children = 45 := by
  sorry

end daughters_and_granddaughters_without_daughters_l4_4483


namespace assign_parents_l4_4167

-- Definition of the problem's context
structure Orphanage := 
  (orphans : Type)
  [decidable_eq orphans]
  (are_friends : orphans → orphans → Prop)
  (are_enemies : orphans → orphans → Prop)
  (friend_or_enemy : ∀ (o1 o2 : orphans), are_friends o1 o2 ∨ are_enemies o1 o2)
  (friend_condition : ∀ (o : orphans) (f1 f2 f3 : orphans),
                        are_friends o f1 → 
                        are_friends o f2 → 
                        are_friends o f3 → 
                        even (finset.filter (λ (p : orphans × orphans), are_enemies p.fst p.snd) 
                                              (({f1, f2, f3}.powerset.filter (λ s, s.card = 2)).bUnion id)).card)

-- Definition of the conclusion
theorem assign_parents : 
  ∀ (O : Orphanage),
  ∃ (P : O.orphans → finset ℕ), 
  (∀ (o1 o2 : O.orphans), O.are_friends o1 o2 ↔ ∃ p, p ∈ P o1 ∧ p ∈ P o2) ∧ 
  (∀ (o1 o2 : O.orphans), O.are_enemies o1 o2 ↔ P o1 ∩ P o2 = ∅) ∧ 
  (∀ (o1 o2 o3 : O.orphans) (p : ℕ),
    p ∈ P o1 ∧ p ∈ P o2 ∧ p ∈ P o3 → false) :=
sorry

end assign_parents_l4_4167


namespace factorize_expression_l4_4696

theorem factorize_expression (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 :=
by sorry

end factorize_expression_l4_4696


namespace sin_330_eq_neg_half_l4_4026

-- Define conditions as hypotheses in Lean
def angle_330 (θ : ℝ) : Prop := θ = 330
def angle_transform (θ : ℝ) : Prop := θ = 360 - 30
def sin_pos (θ : ℝ) : Prop := Real.sin θ = 1 / 2
def sin_neg_in_4th_quadrant (θ : ℝ) : Prop := θ = 330 -> Real.sin θ < 0

-- The main theorem statement
theorem sin_330_eq_neg_half : ∀ θ : ℝ, angle_330 θ → angle_transform θ → sin_pos 30 → sin_neg_in_4th_quadrant θ → Real.sin θ = -1 / 2 := by
  intro θ h1 h2 h3 h4
  sorry

end sin_330_eq_neg_half_l4_4026


namespace books_per_author_l4_4607

theorem books_per_author (total_books : ℕ) (num_authors : ℕ) (books_per_author : ℕ) : 
total_books = 198 ∧ num_authors = 6 → books_per_author = 33 :=
begin
  sorry
end

end books_per_author_l4_4607


namespace individual_weights_l4_4139

theorem individual_weights (A P : ℕ) 
    (h1 : 12 * A + 14 * P = 692)
    (h2 : P = A - 10) : 
    A = 32 ∧ P = 22 :=
by
  sorry

end individual_weights_l4_4139


namespace not_divisible_by_3_l4_4093

theorem not_divisible_by_3 (n : ℤ) : (n^2 + 1) % 3 ≠ 0 := by
  sorry

end not_divisible_by_3_l4_4093


namespace divisible_by_120_l4_4495

theorem divisible_by_120 (n : ℤ) : 120 ∣ (n ^ 6 + 2 * n ^ 5 - n ^ 2 - 2 * n) :=
by sorry

end divisible_by_120_l4_4495


namespace minimum_tenth_game_score_l4_4475

theorem minimum_tenth_game_score (S5 : ℕ) (score10 : ℕ) 
  (h1 : 18 + 15 + 16 + 19 = 68)
  (h2 : S5 ≤ 85)
  (h3 : (S5 + 68 + score10) / 10 > 17) : 
  score10 ≥ 18 := sorry

end minimum_tenth_game_score_l4_4475


namespace product_of_roots_of_cubic_polynomial_l4_4181

theorem product_of_roots_of_cubic_polynomial :
  let a := 1
  let b := -15
  let c := 75
  let d := -50
  ∀ x : ℝ, (x^3 - 15*x^2 + 75*x - 50 = 0) →
  (x + 1) * (x + 1) * (x + 50) - 1 * (x + 50) = 0 → -d / a = 50 :=
by
  sorry

end product_of_roots_of_cubic_polynomial_l4_4181


namespace gpa_at_least_3_5_l4_4803

noncomputable def prob_gpa_at_least_3_5 : ℚ :=
  let p_A_eng := 1 / 3
  let p_B_eng := 1 / 5
  let p_C_eng := 7 / 15 -- 1 - p_A_eng - p_B_eng
  
  let p_A_hist := 1 / 5
  let p_B_hist := 1 / 4
  let p_C_hist := 11 / 20 -- 1 - p_A_hist - p_B_hist

  let prob_two_As := p_A_eng * p_A_hist
  let prob_A_eng_B_hist := p_A_eng * p_B_hist
  let prob_A_hist_B_eng := p_A_hist * p_B_eng
  let prob_two_Bs := p_B_eng * p_B_hist

  let total_prob := prob_two_As + prob_A_eng_B_hist + prob_A_hist_B_eng + prob_two_Bs
  total_prob

theorem gpa_at_least_3_5 : prob_gpa_at_least_3_5 = 6 / 25 := by {
  sorry
}

end gpa_at_least_3_5_l4_4803


namespace admission_charge_for_adult_l4_4974

theorem admission_charge_for_adult 
(admission_charge_per_child : ℝ)
(total_paid : ℝ)
(children_count : ℕ)
(admission_charge_for_adult : ℝ) :
admission_charge_per_child = 0.75 →
total_paid = 3.25 →
children_count = 3 →
admission_charge_for_adult + admission_charge_per_child * children_count = total_paid →
admission_charge_for_adult = 1.00 :=
by
  intros h1 h2 h3 h4
  sorry

end admission_charge_for_adult_l4_4974


namespace us_supermarkets_count_l4_4864

-- Definition of variables and conditions
def total_supermarkets : ℕ := 84
def difference_us_canada : ℕ := 10

-- Proof statement
theorem us_supermarkets_count (C : ℕ) (H : 2 * C + difference_us_canada = total_supermarkets) :
  C + difference_us_canada = 47 :=
sorry

end us_supermarkets_count_l4_4864


namespace triangle_not_always_obtuse_l4_4671

def is_acute_triangle (A B C : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = 180 ∧ A < 90 ∧ B < 90 ∧ C < 90

theorem triangle_not_always_obtuse : ∃ (A B C : ℝ), A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = 180 ∧ is_acute_triangle A B C :=
by
  -- Exact proof here.
  sorry

end triangle_not_always_obtuse_l4_4671


namespace alex_pen_difference_l4_4301

theorem alex_pen_difference 
  (alex_initial_pens : Nat) 
  (doubling_rate : Nat) 
  (weeks : Nat) 
  (jane_pens_month : Nat) :
  alex_initial_pens = 4 →
  doubling_rate = 2 →
  weeks = 4 →
  jane_pens_month = 16 →
  (alex_initial_pens * doubling_rate ^ weeks) - jane_pens_month = 16 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  sorry

end alex_pen_difference_l4_4301


namespace cos_pi_six_plus_alpha_l4_4728

variable (α : ℝ)

theorem cos_pi_six_plus_alpha (h : Real.sin (Real.pi / 3 - α) = 1 / 6) : 
  Real.cos (Real.pi / 6 + α) = 1 / 6 :=
sorry

end cos_pi_six_plus_alpha_l4_4728


namespace sin_330_eq_neg_half_l4_4021

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  -- Proof would go here
  sorry

end sin_330_eq_neg_half_l4_4021


namespace stopped_babysitting_16_years_ago_l4_4080

-- Definitions of given conditions
def started_babysitting_age (Jane_age_start : ℕ) := Jane_age_start = 16
def age_half_constraint (Jane_age child_age : ℕ) := child_age ≤ Jane_age / 2
def current_age (Jane_age_now : ℕ) := Jane_age_now = 32
def oldest_babysat_age_now (child_age_now : ℕ) := child_age_now = 24

-- The proposition to be proved
theorem stopped_babysitting_16_years_ago 
  (Jane_age_start Jane_age_now child_age_now : ℕ)
  (h1 : started_babysitting_age Jane_age_start)
  (h2 : ∀ (Jane_age child_age : ℕ), age_half_constraint Jane_age child_age → Jane_age > Jane_age_start → child_age_now = 24 → Jane_age = 24)
  (h3 : current_age Jane_age_now)
  (h4 : oldest_babysat_age_now child_age_now) :
  Jane_age_now - Jane_age_start = 16 :=
by sorry

end stopped_babysitting_16_years_ago_l4_4080


namespace butterfinger_count_l4_4963

def total_candy_bars : ℕ := 12
def snickers : ℕ := 3
def mars_bars : ℕ := 2
def butterfingers : ℕ := total_candy_bars - (snickers + mars_bars)

theorem butterfinger_count : butterfingers = 7 :=
by
  unfold butterfingers
  sorry

end butterfinger_count_l4_4963


namespace find_common_remainder_l4_4656

theorem find_common_remainder :
  ∃ (d : ℕ), 100 ≤ d ∧ d ≤ 999 ∧ (312837 % d = 96) ∧ (310650 % d = 96) :=
sorry

end find_common_remainder_l4_4656


namespace two_digit_product_l4_4255

theorem two_digit_product (x y : ℕ) (h₁ : 10 ≤ x) (h₂ : x < 100) (h₃ : 10 ≤ y) (h₄ : y < 100) (h₅ : x * y = 4320) :
  (x = 60 ∧ y = 72) ∨ (x = 72 ∧ y = 60) :=
sorry

end two_digit_product_l4_4255


namespace polygon_sides_eq_nine_l4_4076

theorem polygon_sides_eq_nine (n : ℕ) (h : n - 1 = 8) : n = 9 := by
  sorry

end polygon_sides_eq_nine_l4_4076


namespace smallest_x_l4_4269

theorem smallest_x (x : ℕ) (h900 : 900 = 2^2 * 3^2 * 5^2) (h1152 : 1152 = 2^7 * 3^2) : 
  (900 * x) % 1152 = 0 ↔ x = 32 := 
by
  sorry

end smallest_x_l4_4269


namespace correct_sample_size_l4_4149

-- Definitions based on conditions:
def population_size : ℕ := 1800
def sample_size : ℕ := 1000
def surveyed_parents : ℕ := 1000

-- The proof statement we need: 
-- Prove that the sample size is 1000, given the surveyed parents are 1000
theorem correct_sample_size (ps : ℕ) (sp : ℕ) (ss : ℕ) (h1 : ps = population_size) (h2 : sp = surveyed_parents) : ss = sample_size :=
  sorry

end correct_sample_size_l4_4149


namespace divisibility_of_3_pow_p_minus_2_pow_p_minus_1_l4_4396

theorem divisibility_of_3_pow_p_minus_2_pow_p_minus_1 (p : ℕ) (hp : Nat.Prime p) (hp_gt_3 : p > 3) : 
  (3^p - 2^p - 1) % (42 * p) = 0 := 
by
  sorry

end divisibility_of_3_pow_p_minus_2_pow_p_minus_1_l4_4396


namespace number_of_squares_with_prime_condition_l4_4460

theorem number_of_squares_with_prime_condition : 
  ∃! (n : ℕ), ∃ (p : ℕ), Prime p ∧ n^2 = p + 4 := 
sorry

end number_of_squares_with_prime_condition_l4_4460


namespace evaluate_expression_at_3_l4_4311

theorem evaluate_expression_at_3 : (3^3)^(3^3) = 27^27 := by
  sorry

end evaluate_expression_at_3_l4_4311


namespace no_real_solution_equation_l4_4969

theorem no_real_solution_equation (x : ℝ) (h : x ≠ -9) : 
  ¬ ∃ x, (8*x^2 + 90*x + 2) / (3*x + 27) = 4*x + 2 :=
by
  sorry

end no_real_solution_equation_l4_4969


namespace xiao_ming_runs_distance_l4_4401

theorem xiao_ming_runs_distance 
  (num_trees : ℕ) 
  (first_tree : ℕ) 
  (last_tree : ℕ) 
  (distance_between_trees : ℕ) 
  (gap_count : ℕ) 
  (total_distance : ℕ)
  (h1 : num_trees = 200) 
  (h2 : first_tree = 1) 
  (h3 : last_tree = 200) 
  (h4 : distance_between_trees = 6) 
  (h5 : gap_count = last_tree - first_tree)
  (h6 : total_distance = gap_count * distance_between_trees) :
  total_distance = 1194 :=
sorry

end xiao_ming_runs_distance_l4_4401


namespace symmetric_point_l4_4796

theorem symmetric_point (A B C : ℝ) (hA : A = Real.sqrt 7) (hB : B = 1) :
  C = 2 - Real.sqrt 7 ↔ (A + C) / 2 = B :=
by
  sorry

end symmetric_point_l4_4796


namespace games_required_for_champion_l4_4078

-- Define the number of players in the tournament
def players : ℕ := 512

-- Define the tournament conditions
def single_elimination_tournament (n : ℕ) : Prop :=
  ∀ (g : ℕ), g = n - 1

-- State the theorem that needs to be proven
theorem games_required_for_champion : single_elimination_tournament players :=
by
  sorry

end games_required_for_champion_l4_4078


namespace negation_equivalence_l4_4274

-- Define the angles in a triangle as three real numbers
def is_triangle (a b c : ℝ) : Prop :=
  a + b + c = 180 ∧ a > 0 ∧ b > 0 ∧ c > 0

-- Define the proposition
def at_least_one_angle_not_greater_than_60 (a b c : ℝ) : Prop :=
  a ≤ 60 ∨ b ≤ 60 ∨ c ≤ 60

-- Negate the proposition
def all_angles_greater_than_60 (a b c : ℝ) : Prop :=
  a > 60 ∧ b > 60 ∧ c > 60

-- Prove that the negation of the proposition is equivalent
theorem negation_equivalence (a b c : ℝ) (h_triangle : is_triangle a b c) :
  ¬ at_least_one_angle_not_greater_than_60 a b c ↔ all_angles_greater_than_60 a b c :=
by
  sorry

end negation_equivalence_l4_4274


namespace sum_of_seven_consecutive_integers_l4_4386

theorem sum_of_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 :=
by
  sorry

end sum_of_seven_consecutive_integers_l4_4386


namespace factorize_expression_l4_4708

theorem factorize_expression (a x : ℝ) :
  a * x^2 - 2 * a * x + a = a * (x - 1) ^ 2 := 
sorry

end factorize_expression_l4_4708


namespace montoya_food_budget_l4_4973

theorem montoya_food_budget (g t e : ℝ) (h1 : g = 0.6) (h2 : t = 0.8) : e = 0.2 :=
by sorry

end montoya_food_budget_l4_4973


namespace central_angle_of_section_l4_4870

theorem central_angle_of_section (A : ℝ) (x: ℝ) (H : (1 / 8 : ℝ) = (x / 360)) : x = 45 :=
by
  sorry

end central_angle_of_section_l4_4870


namespace solve_quadratic_difference_l4_4813

theorem solve_quadratic_difference :
  ∀ x : ℝ, (x^2 - 7*x - 48 = 0) → 
  let x1 := (7 + Real.sqrt 241) / 2
  let x2 := (7 - Real.sqrt 241) / 2
  abs (x1 - x2) = Real.sqrt 241 :=
by
  sorry

end solve_quadratic_difference_l4_4813


namespace find_n_l4_4819

theorem find_n (m n : ℝ) (h1 : m + 2 * n = 1.2) (h2 : 0.1 + m + n + 0.1 = 1) : n = 0.4 :=
by
  sorry

end find_n_l4_4819


namespace binomial_coeff_sum_l4_4202

theorem binomial_coeff_sum {a a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℤ}
  (h : (1 - x)^7 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7) :
  |a| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| + |a_7| = 128 :=
by
  sorry

end binomial_coeff_sum_l4_4202


namespace percentage_spent_on_household_items_l4_4549

theorem percentage_spent_on_household_items (monthly_income : ℝ) (savings : ℝ) (clothes_percentage : ℝ) (medicines_percentage : ℝ) (household_spent : ℝ) : 
  monthly_income = 40000 ∧ 
  savings = 9000 ∧ 
  clothes_percentage = 0.25 ∧ 
  medicines_percentage = 0.075 ∧ 
  household_spent = monthly_income - (clothes_percentage * monthly_income + medicines_percentage * monthly_income + savings)
  → (household_spent / monthly_income) * 100 = 45 :=
by
  intro h
  cases' h with h1 h_rest
  cases' h_rest with h2 h_rest
  cases' h_rest with h3 h_rest
  cases' h_rest with h4 h5
  have h_clothes := h3
  have h_medicines := h4
  have h_savings := h2
  have h_income := h1
  have h_household := h5
  sorry

end percentage_spent_on_household_items_l4_4549


namespace product_of_roots_l4_4178

theorem product_of_roots :
  let p : Polynomial ℚ := Polynomial.Coeff.mk_fin 3 0 (⟨[50, -75, 15, -1]⟩ : ℕ → ℚ)
  (p.root_prod = 50) :=
begin
  sorry
end

end product_of_roots_l4_4178


namespace european_fraction_is_one_fourth_l4_4340

-- Define the total number of passengers
def P : ℕ := 108

-- Define the fractions and the number of passengers from each continent
def northAmerica := (1 / 12) * P
def africa := (1 / 9) * P
def asia := (1 / 6) * P
def otherContinents := 42

-- Define the total number of non-European passengers
def totalNonEuropean := northAmerica + africa + asia + otherContinents

-- Define the number of European passengers
def european := P - totalNonEuropean

-- Define the fraction of European passengers
def europeanFraction := european / P

-- Prove that the fraction of European passengers is 1/4
theorem european_fraction_is_one_fourth : europeanFraction = 1 / 4 := 
by
  unfold europeanFraction european totalNonEuropean northAmerica africa asia P
  sorry

end european_fraction_is_one_fourth_l4_4340


namespace total_time_required_l4_4876

noncomputable def walking_speed_flat : ℝ := 4
noncomputable def walking_speed_uphill : ℝ := walking_speed_flat * 0.8

noncomputable def running_speed_flat : ℝ := 8
noncomputable def running_speed_uphill : ℝ := running_speed_flat * 0.7

noncomputable def distance_walked_uphill : ℝ := 2
noncomputable def distance_run_uphill : ℝ := 1
noncomputable def distance_run_flat : ℝ := 1

noncomputable def time_walk_uphill := distance_walked_uphill / walking_speed_uphill
noncomputable def time_run_uphill := distance_run_uphill / running_speed_uphill
noncomputable def time_run_flat := distance_run_flat / running_speed_flat

noncomputable def total_time := time_walk_uphill + time_run_uphill + time_run_flat

theorem total_time_required :
  total_time = 0.9286 := by
  sorry

end total_time_required_l4_4876


namespace average_speed_triathlon_l4_4343

theorem average_speed_triathlon :
  let swimming_distance := 1.5
  let biking_distance := 3
  let running_distance := 2
  let swimming_speed := 2
  let biking_speed := 25
  let running_speed := 8

  let t_s := swimming_distance / swimming_speed
  let t_b := biking_distance / biking_speed
  let t_r := running_distance / running_speed
  let total_time := t_s + t_b + t_r

  let total_distance := swimming_distance + biking_distance + running_distance
  let average_speed := total_distance / total_time

  average_speed = 5.8 :=
  by
    sorry

end average_speed_triathlon_l4_4343


namespace shari_effective_distance_l4_4629

-- Define the given conditions
def constant_rate : ℝ := 4 -- miles per hour
def wind_resistance : ℝ := 0.5 -- miles per hour
def walking_time : ℝ := 2 -- hours

-- Define the effective walking speed considering wind resistance
def effective_speed : ℝ := constant_rate - wind_resistance

-- Define the effective walking distance
def effective_distance : ℝ := effective_speed * walking_time

-- State that Shari effectively walks 7.0 miles
theorem shari_effective_distance :
  effective_distance = 7.0 :=
by
  sorry

end shari_effective_distance_l4_4629


namespace factorization_l4_4689

theorem factorization (a x : ℝ) : ax^2 - 2ax + a = a * (x - 1) ^ 2 := 
by
  sorry

end factorization_l4_4689


namespace x_can_be_any_sign_l4_4927

theorem x_can_be_any_sign
  (x y p q : ℝ)
  (h1 : abs (x / y) < abs (p) / q^2)
  (h2 : y ≠ 0) (h3 : q ≠ 0) :
  ∃ (x' : ℝ), True :=
by
  sorry

end x_can_be_any_sign_l4_4927


namespace gcf_180_270_l4_4124

theorem gcf_180_270 : Int.gcd 180 270 = 90 :=
sorry

end gcf_180_270_l4_4124


namespace closest_fraction_to_medals_won_l4_4303

theorem closest_fraction_to_medals_won :
  let won_ratio : ℚ := 35 / 225
  let choices : List ℚ := [1/5, 1/6, 1/7, 1/8, 1/9]
  (closest : ℚ) = 1 / 6 → 
  (closest_in_choices : closest ∈ choices) →
  ∀ choice ∈ choices, abs ((7 / 45) - (1 / 6)) ≤ abs ((7 / 45) - choice) :=
by
  let won_ratio := 7 / 45
  let choices := [1/5, 1/6, 1/7, 1/8, 1/9]
  let closest := 1 / 6
  have closest_in_choices : closest ∈ choices := sorry
  intro choice h_choice_in_choices
  sorry

end closest_fraction_to_medals_won_l4_4303


namespace total_chairs_taken_l4_4366

theorem total_chairs_taken (students trips chairs_per_trip : ℕ) (h_students : students = 5) (h_trips : trips = 10) (h_chairs_per_trip : chairs_per_trip = 5) : 
  students * trips * chairs_per_trip = 250 := by
  rw [h_students, h_trips, h_chairs_per_trip]
  norm_num

end total_chairs_taken_l4_4366


namespace cory_fruits_arrangement_l4_4038

-- Conditions
def apples : ℕ := 4
def oranges : ℕ := 2
def lemon : ℕ := 1
def total_fruits : ℕ := apples + oranges + lemon

-- Formula to calculate the number of distinct ways
def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

def arrangement_count : ℕ :=
  factorial total_fruits / (factorial apples * factorial oranges * factorial lemon)

theorem cory_fruits_arrangement : arrangement_count = 105 := by
  -- Sorry is placed here to skip the actual proof
  sorry

end cory_fruits_arrangement_l4_4038


namespace tan_alpha_minus_pi_six_l4_4723

variable (α β : Real)

axiom tan_alpha_minus_beta : Real.tan (α - β) = 2 / 3
axiom tan_pi_six_minus_beta : Real.tan ((Real.pi / 6) - β) = 1 / 2

theorem tan_alpha_minus_pi_six : Real.tan (α - (Real.pi / 6)) = 1 / 8 :=
by
  sorry

end tan_alpha_minus_pi_six_l4_4723


namespace students_need_to_walk_distance_l4_4388

-- Define distance variables and the relationships
def teacher_initial_distance : ℝ := 235
def xiao_ma_initial_distance : ℝ := 87
def xiao_lu_initial_distance : ℝ := 59
def xiao_zhou_initial_distance : ℝ := 26
def speed_ratio : ℝ := 1.5

-- Prove the distance x students need to walk
theorem students_need_to_walk_distance (x : ℝ) :
  teacher_initial_distance - speed_ratio * x =
  (xiao_ma_initial_distance - x) + (xiao_lu_initial_distance - x) + (xiao_zhou_initial_distance - x) →
  x = 42 :=
by
  sorry

end students_need_to_walk_distance_l4_4388


namespace sin_330_eq_neg_sqrt3_div_2_l4_4002

theorem sin_330_eq_neg_sqrt3_div_2 
  (R : ℝ × ℝ)
  (hR : R = (1/2, -sqrt(3)/2))
  : Real.sin (330 * Real.pi / 180) = -sqrt(3)/2 :=
by
  sorry

end sin_330_eq_neg_sqrt3_div_2_l4_4002


namespace exists_subgraph_with_min_degree_l4_4782

-- Defining necessary basic components
variable {G : Type} [graph G] {V : G}

def average_degree (G : Type) [graph G] : ℝ := sorry  -- This should be defined appropriately, but left as sorry for now

def subgraph_degree (H : subgraph G) (v : H) : ℝ := sorry  -- Definition of degree in the subgraph

theorem exists_subgraph_with_min_degree (G : Type) [graph G] (d : ℝ) (avg_deg : average_degree G = d) :
  ∃ H : subgraph G, ∀ v : H, subgraph_degree H v ≥ d / 2 :=
sorry

end exists_subgraph_with_min_degree_l4_4782


namespace mark_has_seven_butterfingers_l4_4961

/-
  Mark has 12 candy bars in total between Mars bars, Snickers, and Butterfingers.
  He has 3 Snickers and 2 Mars bars.
  Prove that he has 7 Butterfingers.
-/

noncomputable def total_candy_bars : Nat := 12
noncomputable def snickers : Nat := 3
noncomputable def mars_bars : Nat := 2
noncomputable def butterfingers : Nat := total_candy_bars - (snickers + mars_bars)

theorem mark_has_seven_butterfingers : butterfingers = 7 := by
  sorry

end mark_has_seven_butterfingers_l4_4961


namespace product_of_roots_cubic_l4_4175

theorem product_of_roots_cubic :
  let p := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  let roots := {x : ℝ // p x = 0}.toFinset
  ∃ r : ℝ, (∀ x ∈ roots, p x = 0) ∧ (∏ x in roots, x) = 50 :=
by
  sorry

end product_of_roots_cubic_l4_4175


namespace butterfinger_count_l4_4962

def total_candy_bars : ℕ := 12
def snickers : ℕ := 3
def mars_bars : ℕ := 2
def butterfingers : ℕ := total_candy_bars - (snickers + mars_bars)

theorem butterfinger_count : butterfingers = 7 :=
by
  unfold butterfingers
  sorry

end butterfinger_count_l4_4962


namespace angle_C_is_120_l4_4603

theorem angle_C_is_120 (C L U A : ℝ)
  (H1 : C = L)
  (H2 : L = U)
  (H3 : A = L)
  (H4 : A + L = 180)
  (H5 : 6 * C = 720) : C = 120 :=
by
  sorry

end angle_C_is_120_l4_4603


namespace thirteenth_term_is_correct_l4_4115

noncomputable def third_term : ℚ := 2 / 11
noncomputable def twenty_third_term : ℚ := 3 / 7

theorem thirteenth_term_is_correct : 
  (third_term + twenty_third_term) / 2 = 47 / 154 := sorry

end thirteenth_term_is_correct_l4_4115


namespace sin_cos_sixth_sum_l4_4371

noncomputable def theta : ℝ := 25 * Real.pi / 180  -- Convert degrees to radians

theorem sin_cos_sixth_sum :
  (Real.tan theta = 1 / 6) →
  Real.sin theta ^ 6 + Real.cos theta ^ 6 = 11 / 12 :=
begin
  intro h,
  sorry
end

end sin_cos_sixth_sum_l4_4371


namespace sin_330_eq_neg_half_l4_4024

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  -- Proof would go here
  sorry

end sin_330_eq_neg_half_l4_4024


namespace height_previous_year_l4_4863

theorem height_previous_year (current_height : ℝ) (growth_rate : ℝ) (previous_height : ℝ) 
  (h1 : current_height = 126)
  (h2 : growth_rate = 0.05) 
  (h3 : current_height = 1.05 * previous_height) : 
  previous_height = 120 :=
sorry

end height_previous_year_l4_4863


namespace num_digits_expr_l4_4881

noncomputable def num_digits (n : ℕ) : ℕ :=
  (Int.ofNat n).natAbs.digits 10 |>.length

def expr : ℕ := 2^15 * 5^10 * 12

theorem num_digits_expr : num_digits expr = 13 := by
  sorry

end num_digits_expr_l4_4881


namespace negative_integer_solution_l4_4241

theorem negative_integer_solution (x : ℤ) (h : 3 * x + 13 ≥ 0) : x = -1 :=
by
  sorry

end negative_integer_solution_l4_4241


namespace complete_square_transform_l4_4855

theorem complete_square_transform (x : ℝ) :
  x^2 - 2 * x - 5 = 0 → (x - 1)^2 = 6 :=
by
  intro h
  sorry

end complete_square_transform_l4_4855


namespace total_brownies_l4_4382

theorem total_brownies (brought_to_school left_at_home : ℕ) (h1 : brought_to_school = 16) (h2 : left_at_home = 24) : 
  brought_to_school + left_at_home = 40 := 
by 
  sorry

end total_brownies_l4_4382


namespace three_pow_1234_mod_5_l4_4525

theorem three_pow_1234_mod_5 : (3^1234) % 5 = 4 := 
by 
  have h1 : 3^4 % 5 = 1 := by norm_num
  sorry

end three_pow_1234_mod_5_l4_4525


namespace right_triangle_area_l4_4827

theorem right_triangle_area (a b c : ℝ) (h : a^2 + b^2 = c^2) (ha : a = 5) (hc : c = 13) :
  1/2 * a * b = 30 :=
by
  have hb : b = 12, from sorry,
  -- Proof needs to be filled here
  sorry

end right_triangle_area_l4_4827


namespace solution_set_l4_4210

variable {f : ℝ → ℝ}

def decreasing_iff_derivative_neg (f : ℝ → ℝ) :=
∀ x, deriv f x < 0

noncomputable def phi (f : ℝ → ℝ) := λ x, f x - x / 2 - 1 / 2

theorem solution_set (hf1 : f 1 = 1) (hderiv : ∀ x, deriv f x < 1 / 2) :
  ∀ x, f x < x / 2 + 1 / 2 ↔ x > 1 :=
by
  have hphi : ∀ x, deriv (phi f) x = deriv f x - 1 / 2 := by
    sorry -- the derivative computation for phi should be placed here

  have hphi_neg : decreasing_iff_derivative_neg (phi f) := by
    intro x
    rw [hphi]
    exact hderiv x

  have hphi_1 : phi f 1 = 0 := by
    simp [hf1, phi]

  intro x
  split
  · intro hx
    have := hphi_neg x
    sorry -- here we would use the decreasing property of phi to show x > 1

  · intro hx
    have := hphi_neg x
    sorry -- here we would show the equivalence in the other direction

end solution_set_l4_4210


namespace derivative_f_eq_l4_4392

noncomputable def f (x : ℝ) : ℝ := (Real.exp (2 * x)) / x

theorem derivative_f_eq :
  (deriv f) = fun x ↦ ((2 * x - 1) * (Real.exp (2 * x))) / (x ^ 2) := by
  sorry

end derivative_f_eq_l4_4392


namespace sin_330_eq_neg_half_l4_4019

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  -- Proof would go here
  sorry

end sin_330_eq_neg_half_l4_4019


namespace problem_largest_number_l4_4753

def largest_of_four (a b c d : ℚ) : ℚ :=
  max (max a b) (max c d)

theorem problem_largest_number : largest_of_four (2/3) 1 (-3) 0 = 1 := sorry

end problem_largest_number_l4_4753


namespace problem1_problem2_problem3_l4_4882

theorem problem1 : -3^2 + (-1/2)^2 + (2023 - Real.pi)^0 - |-2| = -47/4 :=
by
  sorry

theorem problem2 (a : ℝ) : (-2 * a^2)^3 * a^2 + a^8 = -7 * a^8 :=
by
  sorry

theorem problem3 : 2023^2 - 2024 * 2022 = 1 :=
by
  sorry

end problem1_problem2_problem3_l4_4882


namespace sufficient_but_not_necessary_condition_l4_4953

theorem sufficient_but_not_necessary_condition (x y : ℝ) :
  (x > 1 ∧ y > 1) → (x + y > 2) ∧ ¬((x + y > 2) → (x > 1 ∧ y > 1)) := 
by
  sorry

end sufficient_but_not_necessary_condition_l4_4953


namespace minimum_box_value_l4_4736

theorem minimum_box_value :
  ∃ (a b : ℤ), a * b = 36 ∧ (a^2 + b^2 = 72 ∧ ∀ (a' b' : ℤ), a' * b' = 36 → a'^2 + b'^2 ≥ 72) :=
by
  sorry

end minimum_box_value_l4_4736


namespace opposite_of_one_half_l4_4636

theorem opposite_of_one_half : -((1:ℚ)/2) = -1/2 := by
  -- Skipping the proof using sorry
  sorry

end opposite_of_one_half_l4_4636


namespace remainder_when_7n_div_by_3_l4_4217

theorem remainder_when_7n_div_by_3 (n : ℤ) (h : n % 3 = 2) : (7 * n) % 3 = 2 := 
sorry

end remainder_when_7n_div_by_3_l4_4217


namespace friend_gives_30_l4_4200

noncomputable def total_earnings := 10 + 30 + 50 + 40 + 70

noncomputable def equal_share := total_earnings / 5

noncomputable def contribution_of_highest_earner := 70

noncomputable def amount_to_give := contribution_of_highest_earner - equal_share

theorem friend_gives_30 : amount_to_give = 30 := by
  sorry

end friend_gives_30_l4_4200


namespace candles_ratio_l4_4948

-- Conditions
def kalani_bedroom_candles : ℕ := 20
def donovan_candles : ℕ := 20
def total_candles_house : ℕ := 50

-- Definitions for the number of candles in the living room and the ratio
def living_room_candles : ℕ := total_candles_house - kalani_bedroom_candles - donovan_candles
def ratio_of_candles : ℚ := kalani_bedroom_candles / living_room_candles

theorem candles_ratio : ratio_of_candles = 2 :=
by
  sorry

end candles_ratio_l4_4948


namespace find_fraction_l4_4129

theorem find_fraction (N : ℕ) (hN : N = 90) (f : ℚ)
  (h : 3 + (1/2) * f * (1/5) * N = (1/15) * N) :
  f = 1/3 :=
by {
  sorry
}

end find_fraction_l4_4129


namespace largest_integer_for_gcd_condition_correct_l4_4851

noncomputable def largest_integer_for_gcd_condition : ℕ :=
  let n := 138
  in if (n < 150 ∧ Nat.gcd n 18 = 6) then n else 0

theorem largest_integer_for_gcd_condition_correct :
  ∃ n, (n < 150 ∧ Nat.gcd n 18 = 6) ∧ n = 138 :=
begin
  use 138,
  split,
  { split,
    { exact dec_trivial }, -- Proof that 138 < 150
    { exact dec_trivial }, -- Proof that Nat.gcd 138 18 = 6
  },
  { refl },
end

end largest_integer_for_gcd_condition_correct_l4_4851


namespace cube_root_simplification_l4_4812

noncomputable def cubeRoot (x : ℝ) : ℝ := x^(1/3)

theorem cube_root_simplification :
  cubeRoot 54880000 = 140 * cubeRoot 20 :=
by
  sorry

end cube_root_simplification_l4_4812


namespace number_of_scooters_l4_4602

theorem number_of_scooters (b t s : ℕ) (h1 : b + t + s = 10) (h2 : 2 * b + 3 * t + 2 * s = 26) : s = 2 := 
by sorry

end number_of_scooters_l4_4602


namespace total_chairs_taken_l4_4365

theorem total_chairs_taken (students trips chairs_per_trip : ℕ) (h_students : students = 5) (h_trips : trips = 10) (h_chairs_per_trip : chairs_per_trip = 5) : 
  students * trips * chairs_per_trip = 250 := by
  rw [h_students, h_trips, h_chairs_per_trip]
  norm_num

end total_chairs_taken_l4_4365


namespace factorization_l4_4688

theorem factorization (a x : ℝ) : ax^2 - 2ax + a = a * (x - 1) ^ 2 := 
by
  sorry

end factorization_l4_4688


namespace arithmetic_expression_l4_4304

theorem arithmetic_expression : 8 / 4 + 5 * 2 ^ 2 - (3 + 7) = 12 := by
  sorry

end arithmetic_expression_l4_4304


namespace find_a_l4_4464

noncomputable def binomial_coeff (n k : ℕ) := Nat.choose n k

theorem find_a (a : ℝ) 
  (h : ∃ (a : ℝ), a ^ 3 * binomial_coeff 8 3 = 56) : a = 1 :=
by
  sorry

end find_a_l4_4464


namespace geometric_monotonic_condition_l4_4583

-- Definition of a geometrically increasing sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Definition of a monotonically increasing sequence
def monotonically_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

-- The theorem statement
theorem geometric_monotonic_condition (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  (a 1 < a 2 ∧ a 2 < a 3) ↔ monotonically_increasing a :=
sorry

end geometric_monotonic_condition_l4_4583


namespace cone_surface_area_l4_4333

theorem cone_surface_area {h : ℝ} {A_base : ℝ} (h_eq : h = 4) (A_base_eq : A_base = 9 * Real.pi) :
  let r := Real.sqrt (A_base / Real.pi)
  let l := Real.sqrt (r^2 + h^2)
  let lateral_area := Real.pi * r * l
  let total_surface_area := lateral_area + A_base
  total_surface_area = 24 * Real.pi :=
by
  sorry

end cone_surface_area_l4_4333


namespace value_of_expression_l4_4517

theorem value_of_expression : 10^2 + 10 + 1 = 111 :=
by
  sorry

end value_of_expression_l4_4517


namespace ambiguous_times_l4_4294

theorem ambiguous_times (h m : ℝ) : 
  (∃ k l : ℕ, 0 ≤ k ∧ k < 12 ∧ 0 ≤ l ∧ l < 12 ∧ 
              (12 * h = k * 360 + m) ∧ 
              (12 * m = l * 360 + h) ∧
              k ≠ l) → 
  (∃ n : ℕ, n = 132) := 
sorry

end ambiguous_times_l4_4294


namespace initial_cats_in_shelter_l4_4083

theorem initial_cats_in_shelter
  (cats_found_monday : ℕ)
  (cats_found_tuesday : ℕ)
  (cats_adopted_wednesday : ℕ)
  (current_cats : ℕ)
  (total_adopted_cats : ℕ)
  (initial_cats : ℕ) :
  cats_found_monday = 2 →
  cats_found_tuesday = 1 →
  cats_adopted_wednesday = 3 →
  total_adopted_cats = cats_adopted_wednesday * 2 →
  current_cats = 17 →
  initial_cats = current_cats + total_adopted_cats - (cats_found_monday + cats_found_tuesday) →
  initial_cats = 20 :=
by
  intros
  sorry

end initial_cats_in_shelter_l4_4083


namespace problem1_l4_4866

theorem problem1 (A B C : Prop) : (A ∨ (B ∧ C)) ↔ ((A ∨ B) ∧ (A ∨ C)) :=
sorry 

end problem1_l4_4866


namespace total_bottles_ordered_in_april_and_may_is_1000_l4_4156

-- Define the conditions
def casesInApril : Nat := 20
def casesInMay : Nat := 30
def bottlesPerCase : Nat := 20

-- The total number of bottles ordered in April and May
def totalBottlesOrdered : Nat := (casesInApril + casesInMay) * bottlesPerCase

-- The main statement to be proved
theorem total_bottles_ordered_in_april_and_may_is_1000 :
  totalBottlesOrdered = 1000 :=
sorry

end total_bottles_ordered_in_april_and_may_is_1000_l4_4156


namespace sin_inequality_l4_4721

theorem sin_inequality (x : ℝ) (hx1 : 0 < x) (hx2 : x < Real.pi / 4) : 
  Real.sin (Real.sin x) < Real.sin x ∧ Real.sin x < Real.sin (Real.tan x) :=
by 
  sorry

end sin_inequality_l4_4721


namespace sqrt_product_eq_sixty_sqrt_two_l4_4553

theorem sqrt_product_eq_sixty_sqrt_two : (Real.sqrt 50) * (Real.sqrt 18) * (Real.sqrt 8) = 60 * (Real.sqrt 2) := 
by 
  sorry

end sqrt_product_eq_sixty_sqrt_two_l4_4553


namespace work_completion_l4_4407

theorem work_completion (W : ℕ) (a_rate b_rate combined_rate : ℕ) 
  (h1: combined_rate = W/8) 
  (h2: a_rate = W/12) 
  (h3: combined_rate = a_rate + b_rate) 
  : combined_rate = W/8 :=
by
  sorry

end work_completion_l4_4407


namespace range_of_m_l4_4327

open Real

-- Defining conditions as propositions
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 - m*x + 1 ≠ 0
def q (m : ℝ) : Prop := m > 1
def p_or_q (m : ℝ) : Prop := p m ∨ q m
def p_and_q (m : ℝ) : Prop := p m ∧ q m

-- Mathematically equivalent proof problem
theorem range_of_m (m : ℝ) (H1 : p_or_q m) (H2 : ¬p_and_q m) : -2 < m ∧ m ≤ 1 ∨ 2 ≤ m :=
by
  sorry

end range_of_m_l4_4327


namespace total_seeds_in_garden_l4_4733

-- Definitions based on conditions
def large_bed_rows : Nat := 4
def large_bed_seeds_per_row : Nat := 25
def medium_bed_rows : Nat := 3
def medium_bed_seeds_per_row : Nat := 20
def num_large_beds : Nat := 2
def num_medium_beds : Nat := 2

-- Theorem statement to show total seeds
theorem total_seeds_in_garden : 
  num_large_beds * (large_bed_rows * large_bed_seeds_per_row) + 
  num_medium_beds * (medium_bed_rows * medium_bed_seeds_per_row) = 320 := 
by
  sorry

end total_seeds_in_garden_l4_4733


namespace crescent_moon_area_l4_4641

theorem crescent_moon_area :
  let big_quarter_circle := (4 * 4 * Real.pi) / 4
  let small_semi_circle := (2 * 2 * Real.pi) / 2
  let crescent_area := big_quarter_circle - small_semi_circle
  crescent_area = 2 * Real.pi :=
by
  let big_quarter_circle := (4 * 4 * Real.pi) / 4
  let small_semi_circle := (2 * 2 * Real.pi) / 2
  let crescent_area := big_quarter_circle - small_semi_circle
  have h_bqc : big_quarter_circle = 4 * Real.pi := by
    sorry
  have h_ssc : small_semi_circle = 2 * Real.pi := by
    sorry
  have h_ca : crescent_area = 2 * Real.pi := by
    sorry
  exact h_ca

end crescent_moon_area_l4_4641


namespace largest_valid_n_l4_4563

def is_valid_n (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 10 * a + b ∧ n = a * (a + b)

theorem largest_valid_n : ∀ n : ℕ, is_valid_n n → n ≤ 48 := by sorry

example : is_valid_n 48 := by sorry

end largest_valid_n_l4_4563


namespace value_of_u_when_m_is_3_l4_4480

theorem value_of_u_when_m_is_3 :
  ∀ (u t m : ℕ), (t = 3^m + m) → (u = 4^t - 3 * t) → m = 3 → u = 4^30 - 90 :=
by
  intros u t m ht hu hm
  sorry

end value_of_u_when_m_is_3_l4_4480


namespace parallel_lines_m_values_l4_4050

theorem parallel_lines_m_values (m : ℝ) :
  (∀ x y : ℝ, (3 + m) * x + 4 * y = 5) ∧ (2 * x + (5 + m) * y = 8) → (m = -1 ∨ m = -7) :=
by
  sorry

end parallel_lines_m_values_l4_4050


namespace correct_number_of_students_answered_both_l4_4621

def students_enrolled := 25
def answered_q1_correctly := 22
def answered_q2_correctly := 20
def not_taken_test := 3

def students_answered_both_questions_correctly : Nat :=
  let students_took_test := students_enrolled - not_taken_test
  let b := answered_q2_correctly
  b

theorem correct_number_of_students_answered_both :
  students_answered_both_questions_correctly = answered_q2_correctly :=
by {
  -- this space is for the proof, we are currently not required to provide it
  sorry
}

end correct_number_of_students_answered_both_l4_4621


namespace sequence_solution_l4_4062

noncomputable def seq (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 0 < n → a (n + 1) = a n * ((n + 2) / n)

theorem sequence_solution (a : ℕ → ℝ) (h1 : seq a) (h2 : a 1 = 1) :
  ∀ n : ℕ, 0 < n → a n = (n * (n + 1)) / 2 :=
by
  sorry

end sequence_solution_l4_4062


namespace sin_330_eq_neg_half_l4_4023

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  -- Proof would go here
  sorry

end sin_330_eq_neg_half_l4_4023


namespace sin_330_eq_neg_half_l4_4030

-- Define conditions as hypotheses in Lean
def angle_330 (θ : ℝ) : Prop := θ = 330
def angle_transform (θ : ℝ) : Prop := θ = 360 - 30
def sin_pos (θ : ℝ) : Prop := Real.sin θ = 1 / 2
def sin_neg_in_4th_quadrant (θ : ℝ) : Prop := θ = 330 -> Real.sin θ < 0

-- The main theorem statement
theorem sin_330_eq_neg_half : ∀ θ : ℝ, angle_330 θ → angle_transform θ → sin_pos 30 → sin_neg_in_4th_quadrant θ → Real.sin θ = -1 / 2 := by
  intro θ h1 h2 h3 h4
  sorry

end sin_330_eq_neg_half_l4_4030


namespace inequality_proof_l4_4800

theorem inequality_proof (n k : ℕ) (h₁ : 0 < n) (h₂ : 0 < k) (h₃ : k ≤ n) :
  1 + k / n ≤ (1 + 1 / n)^k ∧ (1 + 1 / n)^k < 1 + k / n + k^2 / n^2 :=
sorry

end inequality_proof_l4_4800


namespace n_n_plus_one_div_by_2_l4_4996

theorem n_n_plus_one_div_by_2 (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 99) : 2 ∣ n * (n + 1) :=
by
  sorry

end n_n_plus_one_div_by_2_l4_4996


namespace sin_330_eq_neg_half_l4_4028

-- Define conditions as hypotheses in Lean
def angle_330 (θ : ℝ) : Prop := θ = 330
def angle_transform (θ : ℝ) : Prop := θ = 360 - 30
def sin_pos (θ : ℝ) : Prop := Real.sin θ = 1 / 2
def sin_neg_in_4th_quadrant (θ : ℝ) : Prop := θ = 330 -> Real.sin θ < 0

-- The main theorem statement
theorem sin_330_eq_neg_half : ∀ θ : ℝ, angle_330 θ → angle_transform θ → sin_pos 30 → sin_neg_in_4th_quadrant θ → Real.sin θ = -1 / 2 := by
  intro θ h1 h2 h3 h4
  sorry

end sin_330_eq_neg_half_l4_4028


namespace no_positive_real_roots_l4_4642

theorem no_positive_real_roots (x : ℝ) : (x^3 + 6 * x^2 + 11 * x + 6 = 0) → x < 0 :=
sorry

end no_positive_real_roots_l4_4642


namespace angle_C_of_triangle_l4_4755

theorem angle_C_of_triangle (A B C : ℝ) (hA : A = 90) (hB : B = 50) (h_sum : A + B + C = 180) : C = 40 := 
by
  sorry

end angle_C_of_triangle_l4_4755


namespace function_behavior_on_intervals_l4_4037

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem function_behavior_on_intervals :
  (∀ x : ℝ, 0 < x ∧ x < Real.exp 1 → 0 < deriv f x) ∧
  (∀ x : ℝ, Real.exp 1 < x ∧ x < 10 → deriv f x < 0) := sorry

end function_behavior_on_intervals_l4_4037


namespace ratio_pentagon_rectangle_l4_4154

theorem ratio_pentagon_rectangle (P: ℝ) (a w: ℝ) (h1: 5 * a = P) (h2: 6 * w = P) (h3: P = 75) : a / w = 6 / 5 := 
by 
  -- Proof steps will be provided to conclude this result 
  sorry

end ratio_pentagon_rectangle_l4_4154


namespace expression_value_l4_4101

theorem expression_value (x a b c : ℝ) 
  (ha : a + x^2 = 2006) 
  (hb : b + x^2 = 2007) 
  (hc : c + x^2 = 2008) 
  (h_abc : a * b * c = 3) :
  (a / (b * c) + b / (c * a) + c / (a * b) - 1 / a - 1 / b - 1 / c = 1) := 
  sorry

end expression_value_l4_4101


namespace sin_330_eq_neg_half_l4_4029

-- Define conditions as hypotheses in Lean
def angle_330 (θ : ℝ) : Prop := θ = 330
def angle_transform (θ : ℝ) : Prop := θ = 360 - 30
def sin_pos (θ : ℝ) : Prop := Real.sin θ = 1 / 2
def sin_neg_in_4th_quadrant (θ : ℝ) : Prop := θ = 330 -> Real.sin θ < 0

-- The main theorem statement
theorem sin_330_eq_neg_half : ∀ θ : ℝ, angle_330 θ → angle_transform θ → sin_pos 30 → sin_neg_in_4th_quadrant θ → Real.sin θ = -1 / 2 := by
  intro θ h1 h2 h3 h4
  sorry

end sin_330_eq_neg_half_l4_4029


namespace cosine_double_angle_tangent_l4_4724

theorem cosine_double_angle_tangent (θ : ℝ) (h : Real.tan θ = -1/3) : Real.cos (2 * θ) = 4/5 :=
by
  sorry

end cosine_double_angle_tangent_l4_4724


namespace area_of_triangle_ABC_l4_4623

noncomputable def distance (a b : ℝ × ℝ) : ℝ := Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

theorem area_of_triangle_ABC (A B C O : ℝ × ℝ)
  (h_isosceles_right : ∃ d: ℝ, distance A B = d ∧ distance A C = d ∧ distance B C = Real.sqrt (2 * d^2))
  (h_A_right : A = (0, 0))
  (h_OA : distance O A = 5)
  (h_OB : distance O B = 7)
  (h_OC : distance O C = 3) :
  ∃ S : ℝ, S = (29 / 2) + (5 / 2) * Real.sqrt 17 :=
sorry

end area_of_triangle_ABC_l4_4623


namespace water_speed_l4_4544

theorem water_speed (swimmer_speed still_water : ℝ) (distance time : ℝ) (h1 : swimmer_speed = 12) (h2 : distance = 12) (h3 : time = 6) :
  ∃ v : ℝ, v = 10 ∧ distance = (swimmer_speed - v) * time :=
by { sorry }

end water_speed_l4_4544


namespace four_digit_number_exists_l4_4354

theorem four_digit_number_exists :
  ∃ (x1 x2 y1 y2 : ℕ), (x1 > 0) ∧ (x2 > 0) ∧ (y1 > 0) ∧ (y2 > 0) ∧
                       (x2 * y2 - x1 * y1 = 67) ∧ (x2 > y2) ∧ (x1 < y1) ∧
                       (x1 * 10^3 + x2 * 10^2 + y2 * 10 + y1 = 1985) := sorry

end four_digit_number_exists_l4_4354


namespace prob_not_negative_review_A_prob_two_positive_reviews_choose_platform_A_l4_4646

-- Definitions of the problem conditions
def positive_reviews_A := 75
def neutral_reviews_A := 20
def negative_reviews_A := 5
def total_reviews_A := 100

def positive_reviews_B := 64
def neutral_reviews_B := 8
def negative_reviews_B := 8
def total_reviews_B := 80

-- Prove the probability that a buyer's evaluation on platform A is not a negative review
theorem prob_not_negative_review_A : 
  (1 - negative_reviews_A / total_reviews_A) = 19 / 20 := by
  sorry

-- Prove the probability that exactly 2 out of 4 (2 from A and 2 from B) buyers give a positive review
theorem prob_two_positive_reviews :
  ((positive_reviews_A / total_reviews_A) ^ 2 * (1 - positive_reviews_B / total_reviews_B) ^ 2 + 
  2 * (positive_reviews_A / total_reviews_A) * (1 - positive_reviews_A / total_reviews_A) * 
  (positive_reviews_B / total_reviews_B) * (1 - positive_reviews_B / total_reviews_B) +
  (1 - positive_reviews_A / total_reviews_A) ^ 2 * (positive_reviews_B / total_reviews_B) ^ 2) = 
  73 / 400 := by
  sorry

-- Choose platform A based on the given data
theorem choose_platform_A :
  let E_A := (5 * 0.75 + 3 * 0.2 + 1 * 0.05)
  let D_A := (5 - E_A) ^ 2 * 0.75 + (3 - E_A) ^ 2 * 0.2 + (1 - E_A) ^ 2 * 0.05
  let E_B := (5 * 0.8 + 3 * 0.1 + 1 * 0.1)
  let D_B := (5 - E_B) ^ 2 * 0.8 + (3 - E_B) ^ 2 * 0.1 + (1 - E_B) ^ 2 * 0.1
  (E_A = E_B) ∧ (D_A < D_B) → choose_platform = "Platform A" := by
  sorry

end prob_not_negative_review_A_prob_two_positive_reviews_choose_platform_A_l4_4646


namespace minimum_value_expr_l4_4316

noncomputable def expr (x : ℝ) : ℝ := 9 * x + 3 / (x ^ 3)

theorem minimum_value_expr : (∀ x : ℝ, x > 0 → expr x ≥ 12) ∧ (∃ x : ℝ, x > 0 ∧ expr x = 12) :=
by
  sorry

end minimum_value_expr_l4_4316


namespace cube_add_constant_135002_l4_4650

theorem cube_add_constant_135002 (n : ℤ) : 
  (∃ m : ℤ, m = n + 1 ∧ m^3 - n^3 = 135002) →
  (n = 149 ∨ n = -151) :=
by
  -- This is where the proof should go
  sorry

end cube_add_constant_135002_l4_4650


namespace no_real_solution_x_4_plus_x_plus1_4_plus_x_plus2_4_eq_x_plus3_4_plus_10_l4_4713

theorem no_real_solution_x_4_plus_x_plus1_4_plus_x_plus2_4_eq_x_plus3_4_plus_10 :
  ¬ ∃ x : ℝ, x^4 + (x + 1)^4 + (x + 2)^4 = (x + 3)^4 + 10 :=
by {
  sorry
}

end no_real_solution_x_4_plus_x_plus1_4_plus_x_plus2_4_eq_x_plus3_4_plus_10_l4_4713


namespace geom_mean_4_16_l4_4910

theorem geom_mean_4_16 (x : ℝ) (h : x^2 = 4 * 16) : x = 8 ∨ x = -8 :=
by
  sorry

end geom_mean_4_16_l4_4910


namespace smallest_fraction_gt_3_5_with_two_digit_nums_l4_4422

theorem smallest_fraction_gt_3_5_with_two_digit_nums : ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ 5 * a > 3 * b ∧ (∀ (a' b' : ℕ), 10 ≤ a' ∧ a' < 100 ∧ 10 ≤ b' ∧ b' < 100 ∧ 5 * a' > 3 * b' → a * b' ≤ a' * b) := 
  ⟨59, 98, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, sorry⟩

end smallest_fraction_gt_3_5_with_two_digit_nums_l4_4422


namespace polar_bear_daily_fish_intake_l4_4440

theorem polar_bear_daily_fish_intake : 
  (0.2 + 0.4 = 0.6) := by
  sorry

end polar_bear_daily_fish_intake_l4_4440


namespace find_AG_l4_4937

theorem find_AG (AE CE BD CD AB AG : ℝ) (h1 : AE = 3)
    (h2 : CE = 1) (h3 : BD = 2) (h4 : CD = 2) (h5 : AB = 5) :
    AG = (3 * Real.sqrt 66) / 7 :=
  sorry

end find_AG_l4_4937


namespace two_cards_totaling_to_14_prob_l4_4121

theorem two_cards_totaling_to_14_prob :
  let deck := finset.range 52
      number_cards := finset.filter (λ x, 2 ≤ x ∧ x ≤ 10) deck
      card_pairs := finset.filter (λ pair, (pair.1 + pair.2 = 14 ∧ pair.1 ≠ pair.2) ∨ (pair.1 = pair.2 ∧ pair.1 = 7)) 
                              (deck.prod deck)
  in 
  ℙ(card_pairs) = 19 / 663 := 
begin
  sorry
end

end two_cards_totaling_to_14_prob_l4_4121


namespace area_minimum_triangle_BQC_l4_4356

noncomputable def cos_angle_bac (AB CA BC : ℝ) : ℝ :=
  (AB^2 + CA^2 - BC^2) / (2 * AB * CA)

def minimum_area_triangle_BQC (AB BC CA : ℝ) : ℝ :=
  let E := BC / 2 in -- E is the midpoint of BC for minimization
  let QE := E in   -- Distance from midpoint to line for minimum area
  0.5 * BC * QE

theorem area_minimum_triangle_BQC (AB BC CA : ℝ) (h_AB : AB = 12) (h_BC : BC = 15) (h_CA : CA = 17) :
  minimum_area_triangle_BQC AB BC CA = 56.25 :=
by
  sorry

end area_minimum_triangle_BQC_l4_4356


namespace slope_of_tangent_l4_4840

variables {x : ℝ}

-- Define the curve y = x^2 + 3x
def curve (x : ℝ) : ℝ := x^2 + 3 * x

-- Derivative of the curve
def curve_deriv (x : ℝ) : ℝ := 2 * x + 3

-- Statement of the problem
theorem slope_of_tangent (x := 2) (y := 10) (h : curve x = y) : (curve_deriv 2) = 7 :=
sorry

end slope_of_tangent_l4_4840


namespace gcd_lcm_888_1147_l4_4444

theorem gcd_lcm_888_1147 :
  Nat.gcd 888 1147 = 37 ∧ Nat.lcm 888 1147 = 27528 := by
  sorry

end gcd_lcm_888_1147_l4_4444


namespace sin_330_value_l4_4012

noncomputable def sin_330 : ℝ := Real.sin (330 * Real.pi / 180)

theorem sin_330_value : sin_330 = -1/2 :=
by {
  sorry
}

end sin_330_value_l4_4012


namespace double_x_value_l4_4216

theorem double_x_value (x : ℝ) (h : x / 2 = 32) : 2 * x = 128 := by
  sorry

end double_x_value_l4_4216


namespace sin_330_value_l4_4010

noncomputable def sin_330 : ℝ := Real.sin (330 * Real.pi / 180)

theorem sin_330_value : sin_330 = -1/2 :=
by {
  sorry
}

end sin_330_value_l4_4010


namespace SumataFamilyTotalMiles_l4_4817

def miles_per_day := 250
def days := 5

theorem SumataFamilyTotalMiles : miles_per_day * days = 1250 :=
by
  sorry

end SumataFamilyTotalMiles_l4_4817


namespace exp_rectangular_form_l4_4888

theorem exp_rectangular_form : (complex.exp (13 * real.pi * complex.I / 2)) = complex.I :=
by
  sorry

end exp_rectangular_form_l4_4888


namespace division_remainder_l4_4716

theorem division_remainder :
  let p := fun x : ℝ => 5 * x^4 - 9 * x^3 + 3 * x^2 - 7 * x - 30
  let q := 3 * x - 9
  p 3 % q = 138 :=
by
  sorry

end division_remainder_l4_4716


namespace maximum_value_of_f_l4_4192

noncomputable def f (x : ℝ) : ℝ := ((x - 3) * (12 - x)) / x

theorem maximum_value_of_f :
  ∀ x : ℝ, 3 < x ∧ x < 12 → f x ≤ 3 :=
by
  sorry

end maximum_value_of_f_l4_4192


namespace f_at_11_l4_4595

def f (n : ℕ) : ℕ := n^2 + n + 17

theorem f_at_11 : f 11 = 149 := sorry

end f_at_11_l4_4595


namespace total_baseball_cards_l4_4379

-- Define the number of baseball cards each person has
def mary_cards : ℕ := 15
def sam_cards : ℕ := 15
def keith_cards : ℕ := 15
def alyssa_cards : ℕ := 15
def john_cards : ℕ := 12
def sarah_cards : ℕ := 18
def emma_cards : ℕ := 10

-- The total number of baseball cards they have
theorem total_baseball_cards :
  mary_cards + sam_cards + keith_cards + alyssa_cards + john_cards + sarah_cards + emma_cards = 100 :=
by
  sorry

end total_baseball_cards_l4_4379


namespace part1_part2_l4_4574

-- Part (1): Prove k = 3 given x = -1 is a solution
theorem part1 (k : ℝ) (h : k * (-1)^2 + 4 * (-1) + 1 = 0) : k = 3 := 
sorry

-- Part (2): Prove k ≤ 4 and k ≠ 0 for the quadratic equation to have two real roots
theorem part2 (k : ℝ) (h : 16 - 4 * k ≥ 0) : k ≤ 4 ∧ k ≠ 0 :=
sorry

end part1_part2_l4_4574


namespace total_spending_l4_4630

theorem total_spending :
  let price_per_pencil := 0.20
  let tolu_pencils := 3
  let robert_pencils := 5
  let melissa_pencils := 2
  let tolu_cost := tolu_pencils * price_per_pencil
  let robert_cost := robert_pencils * price_per_pencil
  let melissa_cost := melissa_pencils * price_per_pencil
  let total_cost := tolu_cost + robert_cost + melissa_cost
  total_cost = 2.00 := by
  sorry

end total_spending_l4_4630


namespace payment_equation_1_payment_equation_2_cost_effective_40_combined_cost_effective_40_l4_4542

namespace ShoppingMall

def tea_set_price : ℕ := 200
def tea_bowl_price : ℕ := 20
def discount_option_1 (x : ℕ) : ℕ := 20 * x + 5400
def discount_option_2 (x : ℕ) : ℕ := 19 * x + 5700
def combined_option_40 : ℕ := 6000 + 190

theorem payment_equation_1 (x : ℕ) (hx : x > 30) : 
  discount_option_1 x = 20 * x + 5400 :=
by sorry

theorem payment_equation_2 (x : ℕ) (hx : x > 30) : 
  discount_option_2 x = 19 * x + 5700 :=
by sorry

theorem cost_effective_40 : discount_option_1 40 < discount_option_2 40 :=
by sorry

theorem combined_cost_effective_40 : combined_option_40 < discount_option_1 40 ∧ combined_option_40 < discount_option_2 40 :=
by sorry

end ShoppingMall

end payment_equation_1_payment_equation_2_cost_effective_40_combined_cost_effective_40_l4_4542


namespace geometric_sequence_common_ratio_l4_4226

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ)
    (h1 : a 2 = a 1 * q)
    (h2 : a 5 = a 1 * q ^ 4)
    (h3 : a 2 = 8)
    (h4 : a 5 = 64) :
    q = 2 := 
sorry

end geometric_sequence_common_ratio_l4_4226


namespace distance_between_centers_of_intersecting_circles_l4_4988

theorem distance_between_centers_of_intersecting_circles
  {r R d : ℝ} (hrR : r < R) (hr : 0 < r) (hR : 0 < R)
  (h_intersect : d < r + R ∧ d > R - r) :
  R - r < d ∧ d < r + R := by
  sorry

end distance_between_centers_of_intersecting_circles_l4_4988


namespace complex_square_l4_4572

theorem complex_square (a b : ℝ) (i : ℂ) (h1 : a + b * i - 2 * i = 2 - b * i) : 
  (a + b * i) ^ 2 = 3 + 4 * i := 
by {
  -- Proof steps skipped (using sorry to indicate proof is required)
  sorry
}

end complex_square_l4_4572


namespace decreasing_arith_prog_smallest_num_l4_4445

theorem decreasing_arith_prog_smallest_num 
  (a d : ℝ) -- Define a and d as real numbers
  (h_arith_prog : ∀ n : ℕ, n < 5 → (∃ k : ℕ, k < 5 ∧ (a - k * d) = if n = 0 then a else a - n * d))
  (h_sum_cubes_zero : a^3 + (a-d)^3 + (a-2*d)^3 + (a-3*d)^3 + (a-4*d)^3 = 0)
  (h_sum_fourth_powers_306 : a^4 + (a-d)^4 + (a-2*d)^4 + (a-3*d)^4 + (a-4*d)^4 = 306) :
  ∃ d' ∈ {d}, a - 4 * d' = -2 * real.sqrt 3 := -- Prove the smallest number is -2√3
sorry

end decreasing_arith_prog_smallest_num_l4_4445


namespace sin_x_solution_l4_4806

theorem sin_x_solution (A B C x : ℝ) (h : A * Real.cos x + B * Real.sin x = C) :
  ∃ (u v : ℝ),  -- We assert the existence of u and v such that 
    Real.sin x = (A * C + B * u) / (A^2 + B^2) ∨ 
    Real.sin x = (A * C - B * v) / (A^2 + B^2) :=
sorry

end sin_x_solution_l4_4806


namespace minimum_value_l4_4342

theorem minimum_value(a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1/2) :
  (2 / a + 3 / b) ≥ 5 + 2 * Real.sqrt 6 :=
sorry

end minimum_value_l4_4342


namespace quadratic_root_value_of_b_l4_4208

theorem quadratic_root_value_of_b :
  (∃ r1 r2 : ℝ, 2 * r1^2 + b * r1 - 20 = 0 ∧ r1 = -5 ∧ r1 * r2 = -10 ∧ r1 + r2 = -b / 2) → b = 6 :=
by
  intro h
  obtain ⟨r1, r2, h_eq1, h_r1, h_prod, h_sum⟩ := h
  sorry

end quadratic_root_value_of_b_l4_4208


namespace sum_of_all_possible_values_of_g_11_l4_4612

def f (x : ℝ) : ℝ := x^2 - 6 * x + 14

def g (x : ℝ) : ℝ := 3 * x + 4

theorem sum_of_all_possible_values_of_g_11 :
  (∀ x : ℝ, f x = 11 → g x = 13 ∨ g x = 7) →
  (13 + 7 = 20) := by
  intros h
  sorry

end sum_of_all_possible_values_of_g_11_l4_4612


namespace positive_difference_sum_even_odd_l4_4991

theorem positive_difference_sum_even_odd :
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1)) / 2
  let sum_first_n_odd (n : ℕ) := n * n
  let sum_30_even := sum_first_n_even 30
  let sum_25_odd := sum_first_n_odd 25
  sum_30_even - sum_25_odd = 305 :=
by
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1)) / 2
  let sum_first_n_odd (n : ℕ) := n * n
  let sum_30_even := sum_first_n_even 30
  let sum_25_odd := sum_first_n_odd 25
  show sum_30_even - sum_25_odd = 305
  sorry

end positive_difference_sum_even_odd_l4_4991


namespace range_a_ff_a_eq_2_f_a_l4_4790

noncomputable def f (x : ℝ) : ℝ :=
if x < 1 then 3 * x - 1 else 2 ^ x

theorem range_a_ff_a_eq_2_f_a :
  {a : ℝ | f (f a) = 2 ^ (f a)} = {a : ℝ | a ≥ 2/3} :=
sorry

end range_a_ff_a_eq_2_f_a_l4_4790


namespace product_of_roots_cubic_l4_4183

theorem product_of_roots_cubic :
  let p := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  in (∃ r1 r2 r3 : ℝ, p r1 = 0 ∧ p r2 = 0 ∧ p r3 = 0 ∧ r1 * r2 * r3 = 50) := 
by 
  sorry

end product_of_roots_cubic_l4_4183


namespace smallest_palindrome_proof_l4_4901

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_three_digit_palindrome (n : ℕ) : Prop :=
  (100 ≤ n ∧ n ≤ 999) ∧ is_palindrome n

def smallest_non_five_digit_palindrome_product_with_103 : ℕ :=
  404

theorem smallest_palindrome_proof :
  is_three_digit_palindrome smallest_non_five_digit_palindrome_product_with_103 ∧ 
  ¬is_palindrome (103 * smallest_non_five_digit_palindrome_product_with_103) ∧ 
  (∀ n, is_three_digit_palindrome n → ¬is_palindrome (103 * n) → n ≥ 404) :=
begin
  sorry
end

end smallest_palindrome_proof_l4_4901


namespace intersection_of_lines_l4_4126

theorem intersection_of_lines : 
  let x := (5 : ℚ) / 9
  let y := (5 : ℚ) / 3
  (y = 3 * x ∧ y - 5 = -6 * x) ↔ (x, y) = ((5 : ℚ) / 9, (5 : ℚ) / 3) := 
by 
  sorry

end intersection_of_lines_l4_4126


namespace running_to_weightlifting_ratio_l4_4950

-- Definitions for given conditions in the problem
def total_practice_time : ℕ := 120 -- 120 minutes
def shooting_time : ℕ := total_practice_time / 2
def weightlifting_time : ℕ := 20
def running_time : ℕ := shooting_time - weightlifting_time

-- The goal is to prove that the ratio of running time to weightlifting time is 2:1
theorem running_to_weightlifting_ratio : running_time / weightlifting_time = 2 :=
by
  /- use the given problem conditions directly -/
  exact sorry

end running_to_weightlifting_ratio_l4_4950


namespace charlie_rope_first_post_l4_4556

theorem charlie_rope_first_post (X : ℕ) (h : X + 20 + 14 + 12 = 70) : X = 24 :=
sorry

end charlie_rope_first_post_l4_4556


namespace proof_stops_with_two_pizzas_l4_4397

/-- The number of stops with orders of two pizzas. -/
def stops_with_two_pizzas : ℕ := 2

theorem proof_stops_with_two_pizzas
  (total_pizzas : ℕ)
  (single_stops : ℕ)
  (two_pizza_stops : ℕ)
  (average_time : ℕ)
  (total_time : ℕ)
  (h1 : total_pizzas = 12)
  (h2 : two_pizza_stops * 2 + single_stops = total_pizzas)
  (h3 : total_time = 40)
  (h4 : average_time = 4)
  (h5 : two_pizza_stops + single_stops = total_time / average_time) :
  two_pizza_stops = stops_with_two_pizzas := 
sorry

end proof_stops_with_two_pizzas_l4_4397


namespace mean_proportional_l4_4564

theorem mean_proportional (x : ℝ) (h : (72.5:ℝ) = Real.sqrt (x * 81)): x = 64.9 := by
  sorry

end mean_proportional_l4_4564


namespace variance_calculation_l4_4289

noncomputable def factory_qualification_rate : ℝ := 0.98
noncomputable def number_of_pieces_selected : ℕ := 10
noncomputable def variance_of_qualified_products : ℝ := number_of_pieces_selected * factory_qualification_rate * (1 - factory_qualification_rate)

theorem variance_calculation : variance_of_qualified_products = 0.196 := by
  -- Proof omitted for brevity
  sorry

end variance_calculation_l4_4289


namespace value_of_angle_A_ratio_of_areas_ABD_and_ACD_l4_4228

open Real

def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  ∃ (A B C : ℝ), 
    (A + B + C = π) ∧
    (a = b * cos (A - π / 6) / sin B) ∧
    (b * cos C = c * cos B)

def point_D_on_BC (B C D : ℝ) (a b c : ℝ) : Prop :=
  ∃ (d : ℝ), (0 ≤ d) ∧ (d ≤ b) ∧ (a = b * cos (A - π / 6) / sin B)
  
def cos_BAD : ℝ := 4 / 5

theorem value_of_angle_A {A B C : ℝ} {a b c : ℝ}
  (h1 : triangle_ABC A B C a b c) :
  A = π / 3 :=
by
  sorry

theorem ratio_of_areas_ABD_and_ACD {A B C D : ℝ} {a b c : ℝ}
  (h1 : triangle_ABC A B C a b c) (h2 : point_D_on_BC B C D a b c)
  (h3 : cos BAD = 4 / 5) :
  (area_ABD / area_ACD = 8 * sqrt 3 + 6 / 13 :=
by
  sorry

end value_of_angle_A_ratio_of_areas_ABD_and_ACD_l4_4228


namespace complete_square_transform_l4_4854

theorem complete_square_transform (x : ℝ) :
  x^2 - 2 * x - 5 = 0 → (x - 1)^2 = 6 :=
by
  intro h
  sorry

end complete_square_transform_l4_4854


namespace sin_from_tan_l4_4470

theorem sin_from_tan (A : ℝ) (h : Real.tan A = Real.sqrt 2 / 3) : 
  Real.sin A = Real.sqrt 22 / 11 := 
by 
  sorry

end sin_from_tan_l4_4470


namespace sum_of_seven_consecutive_integers_l4_4387

theorem sum_of_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 :=
by
  sorry

end sum_of_seven_consecutive_integers_l4_4387


namespace mom_age_when_Jayson_born_l4_4993

theorem mom_age_when_Jayson_born
  (Jayson_age : ℕ)
  (Dad_age : ℕ)
  (Mom_age : ℕ)
  (H1 : Jayson_age = 10)
  (H2 : Dad_age = 4 * Jayson_age)
  (H3 : Mom_age = Dad_age - 2) :
  Mom_age - Jayson_age = 28 := by
  sorry

end mom_age_when_Jayson_born_l4_4993


namespace pencil_price_is_99c_l4_4487

noncomputable def one_pencil_cost (total_spent : ℝ) (notebook_price : ℝ) (notebook_count : ℕ) 
                                  (ruler_pack_price : ℝ) (eraser_price : ℝ) (eraser_count : ℕ) 
                                  (pencil_count : ℕ) (discount : ℝ) (tax : ℝ) : ℝ :=
  let notebooks_cost := notebook_count * notebook_price
  let discount_amount := discount * notebooks_cost
  let discounted_notebooks_cost := notebooks_cost - discount_amount
  let other_items_cost := ruler_pack_price + (eraser_count * eraser_price)
  let subtotal := discounted_notebooks_cost + other_items_cost
  let pencils_total_after_tax := total_spent - subtotal
  let pencils_total_before_tax := pencils_total_after_tax / (1 + tax)
  let pencil_price := pencils_total_before_tax / pencil_count
  pencil_price

theorem pencil_price_is_99c : one_pencil_cost 7.40 0.85 2 0.60 0.20 5 4 0.15 0.10 = 0.99 := 
sorry

end pencil_price_is_99c_l4_4487


namespace part_one_part_two_l4_4309

-- Part 1:
-- Define the function f
def f (x : ℝ) : ℝ := abs (2 * x - 3) + abs (2 * x + 2)

-- Define the inequality problem
theorem part_one (x : ℝ) : f x < x + 5 ↔ 0 < x ∧ x < 2 :=
by sorry

-- Part 2:
-- Define the condition for part 2
theorem part_two (a : ℝ) : (∀ x : ℝ, f x > a + 4 / a) ↔ (a ∈ Set.Ioo 1 4 ∨ a < 0) :=
by sorry

end part_one_part_two_l4_4309


namespace sin_330_eq_neg_half_l4_4018

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  -- Proof would go here
  sorry

end sin_330_eq_neg_half_l4_4018


namespace coefficient_x2_in_expansion_l4_4474

theorem coefficient_x2_in_expansion :
  let exp_poly := (1 + 2 * x) ^ 10 in
  (coeff exp_poly 2) = 180 :=
by
  sorry

end coefficient_x2_in_expansion_l4_4474


namespace madeline_rent_l4_4377

noncomputable def groceries : ℝ := 400
noncomputable def medical_expenses : ℝ := 200
noncomputable def utilities : ℝ := 60
noncomputable def emergency_savings : ℝ := 200
noncomputable def hourly_wage : ℝ := 15
noncomputable def hours_worked : ℕ := 138
noncomputable def total_expenses_and_savings : ℝ := groceries + medical_expenses + utilities + emergency_savings
noncomputable def total_earnings : ℝ := hourly_wage * hours_worked

theorem madeline_rent : total_earnings - total_expenses_and_savings = 1210 := by
  sorry

end madeline_rent_l4_4377


namespace cheburashkas_erased_l4_4778

def total_krakozyabras : ℕ := 29

def total_rows : ℕ := 2

def cheburashkas_per_row := (total_krakozyabras + total_rows) / total_rows / 2 + 1

theorem cheburashkas_erased :
  (total_krakozyabras + total_rows) / total_rows / 2 - 1 = 11 := 
by
  sorry

-- cheburashkas_erased proves that the number of Cheburashkas erased is 11 from the given conditions.

end cheburashkas_erased_l4_4778


namespace prime_product_solution_l4_4610

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_product_solution (p_1 p_2 p_3 p_4 : ℕ) :
  is_prime p_1 ∧ is_prime p_2 ∧ is_prime p_3 ∧ is_prime p_4 ∧ 
  p_1 ≠ p_2 ∧ p_1 ≠ p_3 ∧ p_1 ≠ p_4 ∧ p_2 ≠ p_3 ∧ p_2 ≠ p_4 ∧ p_3 ≠ p_4 ∧
  2 * p_1 + 3 * p_2 + 5 * p_3 + 7 * p_4 = 162 ∧
  11 * p_1 + 7 * p_2 + 5 * p_3 + 4 * p_4 = 162 
  → p_1 * p_2 * p_3 * p_4 = 570 := 
by
  sorry

end prime_product_solution_l4_4610


namespace area_of_45_45_90_triangle_l4_4297

theorem area_of_45_45_90_triangle (h : ℝ) (h_eq : h = 8 * Real.sqrt 2) : 
  ∃ (A : ℝ), A = 32 := 
by
  sorry

end area_of_45_45_90_triangle_l4_4297


namespace factorize_expression_l4_4041

theorem factorize_expression (a b : ℝ) : 
  a^2 * b + 2 * a * b^2 + b^3 = b * (a + b)^2 :=
by {
  sorry
}

end factorize_expression_l4_4041


namespace factorization_l4_4686

theorem factorization (a x : ℝ) : ax^2 - 2ax + a = a * (x - 1) ^ 2 := 
by
  sorry

end factorization_l4_4686


namespace range_of_k_l4_4105

theorem range_of_k (k : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℝ), (x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁) ∧ 
   (x₁^3 - 3*x₁ = k ∧ x₂^3 - 3*x₂ = k ∧ x₃^3 - 3*x₃ = k)) ↔ (-2 < k ∧ k < 2) :=
sorry

end range_of_k_l4_4105


namespace perfect_square_quotient_l4_4451

theorem perfect_square_quotient {a b : ℕ} (hpos: 0 < a ∧ 0 < b) (hdiv : (ab + 1) ∣ (a^2 + b^2)) : ∃ k : ℕ, k^2 = (a^2 + b^2) / (ab + 1) :=
sorry

end perfect_square_quotient_l4_4451


namespace sequence_subsequence_l4_4551

theorem sequence_subsequence :
  ∃ (a : Fin 101 → ℕ), 
  (∀ i, a i = i + 1) ∧ 
  ∃ (b : Fin 11 → ℕ), 
  (b 0 < b 1 ∧ b 1 < b 2 ∧ b 2 < b 3 ∧ b 3 < b 4 ∧ b 4 < b 5 ∧ 
  b 5 < b 6 ∧ b 6 < b 7 ∧ b 7 < b 8 ∧ b 8 < b 9 ∧ b 9 < b 10) ∨ 
  (b 0 > b 1 ∧ b 1 > b 2 ∧ b 2 > b 3 ∧ b 3 > b 4 ∧ b 4 > b 5 ∧ 
  b 5 > b 6 ∧ b 6 > b 7 ∧ b 7 > b 8 ∧ b 8 > b 9 ∧ b 9 > b 10) :=
by {
  sorry
}

end sequence_subsequence_l4_4551


namespace quadratic_roots_unique_pair_l4_4815

theorem quadratic_roots_unique_pair (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0)
  (h_root1 : p * q = q)
  (h_root2 : p + q = -p)
  (h_rel : q = -2 * p) : 
(p, q) = (1, -2) :=
  sorry

end quadratic_roots_unique_pair_l4_4815


namespace cards_per_layer_correct_l4_4957

-- Definitions based on the problem's conditions
def num_decks : ℕ := 16
def cards_per_deck : ℕ := 52
def num_layers : ℕ := 32

-- The key calculation we need to prove
def total_cards : ℕ := num_decks * cards_per_deck
def cards_per_layer : ℕ := total_cards / num_layers

theorem cards_per_layer_correct : cards_per_layer = 26 := by
  unfold cards_per_layer total_cards num_decks cards_per_deck num_layers
  simp
  sorry

end cards_per_layer_correct_l4_4957


namespace smallest_possible_b_l4_4786

-- Definitions of conditions
variables {a b c : ℤ}

-- Conditions expressed in Lean
def is_geometric_progression (a b c : ℤ) : Prop := b^2 = a * c
def is_arithmetic_progression (a b c : ℤ) : Prop := a + b = 2 * c

-- The theorem statement
theorem smallest_possible_b (a b c : ℤ) 
  (h1 : a < b) (h2 : b < c) 
  (hg : is_geometric_progression a b c) 
  (ha : is_arithmetic_progression a c b) : b = 2 := sorry

end smallest_possible_b_l4_4786


namespace triangle_area_30_l4_4829

theorem triangle_area_30 (h : ∃ a b c : ℝ, a^2 + b^2 = c^2 ∧ a = 5 ∧ c = 13 ∧ b > 0) : 
  ∃ area : ℝ, area = 1 / 2 * 5 * (b : ℝ) ∧ area = 30 :=
by
  sorry

end triangle_area_30_l4_4829


namespace remainder_when_divided_by_385_l4_4100

theorem remainder_when_divided_by_385 (x : ℤ)
  (h1 : 2 + x ≡ 4 [ZMOD 125])
  (h2 : 3 + x ≡ 9 [ZMOD 343])
  (h3 : 4 + x ≡ 25 [ZMOD 1331]) :
  x ≡ 307 [ZMOD 385] :=
sorry

end remainder_when_divided_by_385_l4_4100


namespace tree_graph_probability_127_l4_4985

theorem tree_graph_probability_127 :
  let n := 5
  let p := 125
  let q := 1024
  q ^ (1/10) + p = 127 :=
by
  sorry

end tree_graph_probability_127_l4_4985


namespace number_of_Cheburashkas_erased_l4_4774

theorem number_of_Cheburashkas_erased :
  ∃ (n : ℕ), 
    (∀ x, x ≥ 1 → 
      (let totalKrakozyabras = (2 * (x - 1) = 29) in
         x - 2 = 11)) :=
sorry

end number_of_Cheburashkas_erased_l4_4774


namespace dagger_simplified_l4_4653

def dagger (m n p q : ℚ) : ℚ := (m^2) * p * (q / n)

theorem dagger_simplified :
  dagger (5:ℚ) (9:ℚ) (4:ℚ) (6:ℚ) = (200:ℚ) / (3:ℚ) :=
by
  sorry

end dagger_simplified_l4_4653


namespace lcm_18_24_30_l4_4522

theorem lcm_18_24_30 :
  let a := 18
  let b := 24
  let c := 30
  let lcm := 360
  (∀ x > 0, x ∣ a ∧ x ∣ b ∧ x ∣ c → x ∣ lcm) ∧ (∀ y > 0, y ∣ lcm → y ∣ a ∧ y ∣ b ∧ y ∣ c) :=
by {
  let a := 18
  let b := 24
  let c := 30
  let lcm := 360
  sorry
}

end lcm_18_24_30_l4_4522


namespace average_weight_l4_4507

theorem average_weight (A B C : ℝ) (h1 : (A + B) / 2 = 40) (h2 : (B + C) / 2 = 47) (h3 : B = 39) : (A + B + C) / 3 = 45 := 
  sorry

end average_weight_l4_4507


namespace remainder_when_dividing_150_l4_4321

theorem remainder_when_dividing_150 (k : ℕ) (hk1 : k > 0) (hk2 : 80 % k^2 = 8) : 150 % k = 6 :=
by
  sorry

end remainder_when_dividing_150_l4_4321


namespace number_power_eq_l4_4287

theorem number_power_eq (x : ℕ) (h : x^10 = 16^5) : x = 4 :=
by {
  -- Add supporting calculations here if needed
  sorry
}

end number_power_eq_l4_4287


namespace total_dolls_l4_4335

def sisters_dolls : ℝ := 8.5

def hannahs_dolls : ℝ := 5.5 * sisters_dolls

theorem total_dolls : hannahs_dolls + sisters_dolls = 55.25 :=
by
  -- Proof is omitted
  sorry

end total_dolls_l4_4335


namespace systematic_sampling_starts_with_srs_l4_4816

-- Define the concept of systematic sampling
def systematically_sampled (initial_sampled: Bool) : Bool :=
  initial_sampled

-- Initial sample is determined by simple random sampling
def simple_random_sampling : Bool :=
  True

-- We need to prove that systematic sampling uses simple random sampling at the start
theorem systematic_sampling_starts_with_srs : systematically_sampled simple_random_sampling = True :=
by 
  sorry

end systematic_sampling_starts_with_srs_l4_4816


namespace complete_square_l4_4857

theorem complete_square (x : ℝ) : (x^2 - 2 * x - 5 = 0) ↔ ((x - 1)^2 = 6) := 
by
  sorry

end complete_square_l4_4857


namespace nuts_mixture_weight_l4_4412

variable (m n : ℕ)
variable (weight_almonds per_part total_weight : ℝ)

theorem nuts_mixture_weight (h1 : m = 5) (h2 : n = 2) (h3 : weight_almonds = 250) 
  (h4 : per_part = weight_almonds / m) (h5 : total_weight = per_part * (m + n)) : 
  total_weight = 350 := by
  sorry

end nuts_mixture_weight_l4_4412


namespace find_m_l4_4334

-- We define the universal set U, the set A with an unknown m, and the complement of A in U.
def U : Set ℕ := {1, 2, 3}
def A (m : ℕ) : Set ℕ := {1, m}
def complement_U_A (m : ℕ) : Set ℕ := U \ A m

-- The main theorem where we need to prove m = 3 given the conditions.
theorem find_m (m : ℕ) (hU : U = {1, 2, 3})
  (hA : ∀ m, A m = {1, m})
  (h_complement : complement_U_A m = {2}) : m = 3 := sorry

end find_m_l4_4334


namespace repeating_decimal_divisible_by_2_or_5_l4_4839

theorem repeating_decimal_divisible_by_2_or_5 
    (m n : ℕ) 
    (x : ℝ) 
    (r s : ℕ) 
    (a b k p q u : ℕ)
    (hmn_coprime : Nat.gcd m n = 1)
    (h_rep_decimal : x = (m:ℚ) / (n:ℚ))
    (h_non_repeating_part: 0 < r) :
  n % 2 = 0 ∨ n % 5 = 0 :=
sorry

end repeating_decimal_divisible_by_2_or_5_l4_4839


namespace negation_proposition_equiv_l4_4110

variable (m : ℤ)

theorem negation_proposition_equiv :
  (¬ ∃ x : ℤ, x^2 + x + m < 0) ↔ (∀ x : ℤ, x^2 + x + m ≥ 0) :=
by
  sorry

end negation_proposition_equiv_l4_4110


namespace arithmetic_seq_sum_l4_4750

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 2 - a 1

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h_arith : is_arithmetic_sequence a)
  (h1 : a 1 + a 4 + a 7 = 39) (h2 : a 2 + a 5 + a 8 = 33) :
  a 3 + a 6 + a 9 = 27 :=
sorry

end arithmetic_seq_sum_l4_4750


namespace total_students_in_class_l4_4152

/-- 
There are 208 boys in the class.
There are 69 more girls than boys.
The total number of students in the class is the sum of boys and girls.
Prove that the total number of students in the graduating class is 485.
-/
theorem total_students_in_class (boys girls : ℕ) (h1 : boys = 208) (h2 : girls = boys + 69) : 
  boys + girls = 485 :=
by
  sorry

end total_students_in_class_l4_4152


namespace find_a_and_b_l4_4454

theorem find_a_and_b (a b : ℝ) :
  (∀ x, y = a + b / x) →
  (y = 3 → x = 2) →
  (y = -1 → x = -4) →
  a + b = 4 :=
by sorry

end find_a_and_b_l4_4454


namespace rubber_duck_cost_l4_4249

theorem rubber_duck_cost 
  (price_large : ℕ)
  (num_regular : ℕ)
  (num_large : ℕ)
  (total_revenue : ℕ)
  (h1 : price_large = 5)
  (h2 : num_regular = 221)
  (h3 : num_large = 185)
  (h4 : total_revenue = 1588) :
  ∃ (cost_regular : ℕ), (num_regular * cost_regular + num_large * price_large = total_revenue) ∧ cost_regular = 3 :=
by
  exists 3
  sorry

end rubber_duck_cost_l4_4249


namespace tan_half_angle_product_l4_4215

theorem tan_half_angle_product (a b : ℝ) 
  (h : 7 * (Real.cos a + Real.cos b) + 6 * (Real.cos a * Real.cos b + 1) = 0) :
  ∃ x : ℝ, x = Real.tan (a / 2) * Real.tan (b / 2) ∧ (x = Real.sqrt (26 / 7) ∨ x = -Real.sqrt (26 / 7)) :=
by
  sorry

end tan_half_angle_product_l4_4215


namespace jean_grandchildren_total_giveaway_l4_4942

theorem jean_grandchildren_total_giveaway :
  let num_grandchildren := 3
  let cards_per_grandchild_per_year := 2
  let amount_per_card := 80
  let total_amount_per_grandchild_per_year := cards_per_grandchild_per_year * amount_per_card
  let total_amount_per_year := num_grandchildren * total_amount_per_grandchild_per_year
  total_amount_per_year = 480 :=
by
  sorry

end jean_grandchildren_total_giveaway_l4_4942


namespace highest_and_lowest_score_average_score_l4_4747

def std_score : ℤ := 60
def scores : List ℤ := [36, 0, 12, -18, 20]

theorem highest_and_lowest_score 
  (highest_score : ℤ) (lowest_score : ℤ) : 
  highest_score = std_score + 36 ∧ lowest_score = std_score - 18 := 
sorry

theorem average_score (avg_score : ℤ) :
  avg_score = std_score + ((36 + 0 + 12 - 18 + 20) / 5) := 
sorry

end highest_and_lowest_score_average_score_l4_4747


namespace intersection_of_sets_l4_4727

-- Conditions as Lean definitions
def A : Set Int := {-2, -1}
def B : Set Int := {-1, 2, 3}

-- Stating the proof problem in Lean 4
theorem intersection_of_sets : A ∩ B = {-1} :=
by
  sorry

end intersection_of_sets_l4_4727


namespace tan_theta_value_l4_4055

open Real

theorem tan_theta_value
  (theta : ℝ)
  (h_quad : 3 * pi / 2 < theta ∧ theta < 2 * pi)
  (h_sin : sin theta = -sqrt 6 / 3) :
  tan theta = -sqrt 2 := by
  sorry

end tan_theta_value_l4_4055


namespace express_x_n_prove_inequality_l4_4481

variable (a b n : Real)
variable (x : ℕ → Real)

def trapezoid_conditions : Prop :=
  ∀ n, x 1 = a * b / (a + b) ∧ (x (n + 1) / x n = x (n + 1) / a)

theorem express_x_n (h : trapezoid_conditions a b x) : 
  ∀ n, x n = a * b / (a + n * b) := 
by
  sorry

theorem prove_inequality (h : trapezoid_conditions a b x) : 
  ∀ n, x n ≤ (a + n * b) / (4 * n) := 
by
  sorry

end express_x_n_prove_inequality_l4_4481


namespace local_minimum_at_one_l4_4600

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - 2 * x^2 + a^2 * x

theorem local_minimum_at_one (a : ℝ) (hfmin : ∀ x : ℝ, deriv (f a) x = 3 * a * x^2 - 4 * x + a^2) (h1 : f a 1 = f a 1) : a = 1 :=
sorry

end local_minimum_at_one_l4_4600


namespace projection_onto_plane_l4_4611

open Matrix

def normal_vector : Vector ℝ 3 := ![1, -2, 1]

def proj_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![5/6, 1/3, -1/6],
    ![1/3, 1/3, 1/3],
    ![-1/6, 1/3, 5/6]]

def projection (v : Vector ℝ 3) : Vector ℝ 3 :=
  proj_matrix.mulVec v

theorem projection_onto_plane (v : Vector ℝ 3) :
  projection v = proj_matrix.mulVec v := by
  sorry

end projection_onto_plane_l4_4611


namespace percentage_of_water_in_mixture_l4_4872

-- Definitions based on conditions from a)
def original_price : ℝ := 1 -- assuming $1 per liter for pure dairy
def selling_price : ℝ := 1.25 -- 25% profit means selling at $1.25
def profit_percentage : ℝ := 0.25 -- 25% profit

-- Theorem statement based on the equivalent problem in c)
theorem percentage_of_water_in_mixture : 
  (selling_price - original_price) / selling_price * 100 = 20 :=
by
  sorry

end percentage_of_water_in_mixture_l4_4872


namespace probability_same_filling_correct_l4_4090

open ProbabilityTheory

-- Defining the numbers of each type of pancakes
def num_meat : ℕ := 2
def num_cheese : ℕ := 3
def num_strawberries : ℕ := 5
def total_pancakes : ℕ := num_meat + num_cheese + num_strawberries

-- Calculating the probability of the first and the last pancake being the same type
noncomputable def probability_same_filling : ℚ := 
  (num_meat / total_pancakes * (num_meat - 1) / (total_pancakes - 1) +
   num_cheese / total_pancakes * (num_cheese - 1) / (total_pancakes - 1) +
   num_strawberries / total_pancakes * (num_strawberries - 1) / (total_pancakes - 1))

-- The theorem we seek to prove
theorem probability_same_filling_correct : probability_same_filling = 14 / 45 :=
sorry

end probability_same_filling_correct_l4_4090


namespace reggie_free_throws_l4_4933

namespace BasketballShootingContest

-- Define the number of points for different shots
def points (layups free_throws long_shots : ℕ) : ℕ :=
  1 * layups + 2 * free_throws + 3 * long_shots

-- Conditions given in the problem
def Reggie_points (F: ℕ) : ℕ := 
  points 3 F 1

def Brother_points : ℕ := 
  points 0 0 4

-- The given condition that Reggie loses by 2 points
theorem reggie_free_throws:
  ∃ F : ℕ, Reggie_points F + 2 = Brother_points :=
sorry

end BasketballShootingContest

end reggie_free_throws_l4_4933


namespace swimming_both_days_l4_4092

theorem swimming_both_days
  (total_students swimming_today soccer_today : ℕ)
  (students_swimming_yesterday students_soccer_yesterday : ℕ)
  (soccer_today_swimming_yesterday soccer_today_soccer_yesterday : ℕ)
  (swimming_today_swimming_yesterday swimming_today_soccer_yesterday : ℕ) :
  total_students = 33 ∧
  swimming_today = 22 ∧
  soccer_today = 22 ∧
  soccer_today_swimming_yesterday = 15 ∧
  soccer_today_soccer_yesterday = 15 ∧
  swimming_today_swimming_yesterday = 15 ∧
  swimming_today_soccer_yesterday = 15 →
  ∃ (swimming_both_days : ℕ), swimming_both_days = 4 :=
by
  sorry

end swimming_both_days_l4_4092


namespace smallest_odd_digit_number_gt_1000_mult_5_l4_4852

def is_odd_digit (n : ℕ) : Prop := n = 1 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 9

def valid_number (n : ℕ) : Prop :=
  n > 1000 ∧ (∃ d1 d2 d3 d4, n = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧ 
  is_odd_digit d1 ∧ is_odd_digit d2 ∧ is_odd_digit d3 ∧ is_odd_digit d4 ∧ 
  d4 = 5)

theorem smallest_odd_digit_number_gt_1000_mult_5 : ∃ n : ℕ, valid_number n ∧ 
  ∀ m : ℕ, valid_number m → m ≥ n := 
by
  use 1115
  simp [valid_number, is_odd_digit]
  sorry

end smallest_odd_digit_number_gt_1000_mult_5_l4_4852


namespace total_opaque_stackings_l4_4409

-- Define the glass pane and its rotation
inductive Rotation
| deg_0 | deg_90 | deg_180 | deg_270
deriving DecidableEq, Repr

-- The property of opacity for a stack of glass panes
def isOpaque (stack : List (List Rotation)) : Bool :=
  -- The implementation of this part depends on the specific condition in the problem
  -- and here is abstracted out for the problem statement.
  sorry

-- The main problem stating the required number of ways
theorem total_opaque_stackings : ∃ (n : ℕ), n = 7200 :=
  sorry

end total_opaque_stackings_l4_4409


namespace product_of_roots_l4_4177

theorem product_of_roots :
  let p : Polynomial ℚ := Polynomial.Coeff.mk_fin 3 0 (⟨[50, -75, 15, -1]⟩ : ℕ → ℚ)
  (p.root_prod = 50) :=
begin
  sorry
end

end product_of_roots_l4_4177


namespace find_integer_n_l4_4560

theorem find_integer_n (n : ℤ) (h1 : n ≥ 3) (h2 : ∃ k : ℚ, k * k = (n^2 - 5) / (n + 1)) : n = 3 := by
  sorry

end find_integer_n_l4_4560


namespace depth_of_tank_proof_l4_4877

-- Definitions based on conditions
def length_of_tank : ℝ := 25
def width_of_tank : ℝ := 12
def cost_per_sq_meter : ℝ := 0.75
def total_cost : ℝ := 558

-- The depth of the tank to be proven as 6 meters
def depth_of_tank : ℝ := 6

-- Area of the tanks for walls and bottom
def plastered_area (d : ℝ) : ℝ := 2 * (length_of_tank * d) + 2 * (width_of_tank * d) + (length_of_tank * width_of_tank)

-- Final cost calculation
def plastering_cost (d : ℝ) : ℝ := cost_per_sq_meter * (plastered_area d)

-- Statement to be proven in Lean 4
theorem depth_of_tank_proof : plastering_cost depth_of_tank = total_cost :=
by
  sorry

end depth_of_tank_proof_l4_4877


namespace find_b_l4_4467

noncomputable theory
open Real

variables {x₁ x₂ k b : ℝ}

def tangent_line (x : ℝ) (k b : ℝ) := k * x + b
def curve_1 (x : ℝ) := log x + 2
def curve_2 (x : ℝ) := log (x + 1)

theorem find_b
  (h_tangent_k : ∀ x₁ x₂, k = 1 / x₁ ∧ k = 1 / (x₂ + 1))
  (h_x_relation : x₁ = x₂ + 1)
  (h_point_on_curve1 : tangent_line x₁ k b = curve_1 x₁)
  (h_point_on_curve2 : tangent_line x₂ k b = curve_2 x₂) :
  b = 1 - log 2 :=
begin
  sorry
end

end find_b_l4_4467


namespace smallest_integer_y_l4_4268

theorem smallest_integer_y (y : ℤ) (h : 7 - 3 * y < 20) : ∃ (y : ℤ), y = -4 :=
by
  sorry

end smallest_integer_y_l4_4268


namespace geometric_sequence_sixth_term_l4_4758

theorem geometric_sequence_sixth_term:
  ∃ q : ℝ, 
  ∀ (a₁ a₈ a₆ : ℝ), 
    a₁ = 6 ∧ a₈ = 768 ∧ a₈ = a₁ * q^7 ∧ a₆ = a₁ * q^5 
    → a₆ = 192 :=
by
  sorry

end geometric_sequence_sixth_term_l4_4758


namespace total_flowers_in_vase_l4_4320

-- Conditions as definitions
def num_roses : ℕ := 5
def num_lilies : ℕ := 2

-- Theorem statement
theorem total_flowers_in_vase : num_roses + num_lilies = 7 :=
by
  sorry

end total_flowers_in_vase_l4_4320


namespace sum_of_legs_is_43_l4_4109

theorem sum_of_legs_is_43 (x : ℕ) (h1 : x * x + (x + 1) * (x + 1) = 31 * 31) :
  x + (x + 1) = 43 :=
sorry

end sum_of_legs_is_43_l4_4109


namespace how_many_cheburashkas_erased_l4_4771

theorem how_many_cheburashkas_erased 
  (total_krakozyabras : ℕ)
  (characters_per_row_initial : ℕ) 
  (total_characters_initial : ℕ)
  (total_cheburashkas : ℕ)
  (total_rows : ℕ := 2)
  (total_krakozyabras := 29) :
  total_cheburashkas = 11 :=
by
  sorry

end how_many_cheburashkas_erased_l4_4771


namespace deductive_reasoning_is_option_A_l4_4275

-- Define the types of reasoning.
inductive ReasoningType
| Deductive
| Analogical
| Inductive

-- Define the options provided in the problem.
def OptionA : ReasoningType := ReasoningType.Deductive
def OptionB : ReasoningType := ReasoningType.Analogical
def OptionC : ReasoningType := ReasoningType.Inductive
def OptionD : ReasoningType := ReasoningType.Inductive

-- Statement to prove that Option A is Deductive reasoning.
theorem deductive_reasoning_is_option_A : OptionA = ReasoningType.Deductive := by
  -- proof
  sorry

end deductive_reasoning_is_option_A_l4_4275


namespace roots_of_equation_l4_4640

theorem roots_of_equation : ∀ x : ℝ, x^2 - 3 * x = 0 ↔ x = 0 ∨ x = 3 :=
by sorry

end roots_of_equation_l4_4640


namespace find_XY_in_306090_triangle_l4_4895

-- Definitions of the problem
def angleZ := 90
def angleX := 60
def hypotenuseXZ := 12
def isRightTriangle (XYZ : Type) (angleZ : ℕ) : Prop := angleZ = 90
def is306090Triangle (XYZ : Type) (angleX : ℕ) (angleZ : ℕ) : Prop := (angleX = 60) ∧ (angleZ = 90)

-- Lean theorem statement
theorem find_XY_in_306090_triangle 
  (XYZ : Type)
  (hypotenuseXZ : ℕ)
  (h1 : isRightTriangle XYZ angleZ)
  (h2 : is306090Triangle XYZ angleX angleZ) :
  XY = 8 := 
sorry

end find_XY_in_306090_triangle_l4_4895


namespace modulus_of_z_l4_4916

section complex_modulus
open Complex

theorem modulus_of_z (z : ℂ) (h : z * (2 + I) = 10 - 5 * I) : Complex.abs z = 5 :=
by
  sorry
end complex_modulus

end modulus_of_z_l4_4916


namespace remaining_amount_to_be_paid_l4_4999

theorem remaining_amount_to_be_paid (part_payment : ℝ) (percentage : ℝ) (h : part_payment = 650 ∧ percentage = 0.15) :
    (part_payment / percentage - part_payment) = 3683.33 := by
  cases h with
  | intro h1 h2 =>
    sorry

end remaining_amount_to_be_paid_l4_4999


namespace find_abc_l4_4672

open Polynomial

noncomputable def my_gcd_lcm_problem (a b c : ℤ) : Prop :=
  gcd (X^2 + (C a * X) + C b) (X^2 + (C b * X) + C c) = X + 1 ∧
  lcm (X^2 + (C a * X) + C b) (X^2 + (C b * X) + C c) = X^3 - 5*X^2 + 7*X - 3

theorem find_abc : ∀ (a b c : ℤ),
  my_gcd_lcm_problem a b c → a + b + c = -3 :=
by
  intros a b c h
  sorry

end find_abc_l4_4672


namespace travel_time_l4_4421

-- Given conditions
def distance_per_hour : ℤ := 27
def distance_to_sfl : ℤ := 81

-- Theorem statement to prove
theorem travel_time (dph : ℤ) (dts : ℤ) (h1 : dph = distance_per_hour) (h2 : dts = distance_to_sfl) : 
  dts / dph = 3 := 
by
  -- immediately helps execute the Lean statement
  sorry

end travel_time_l4_4421


namespace city_G_has_highest_percentage_increase_l4_4902

-- Define the population data as constants.
def population_1990_F : ℕ := 50
def population_2000_F : ℕ := 60
def population_1990_G : ℕ := 60
def population_2000_G : ℕ := 80
def population_1990_H : ℕ := 90
def population_2000_H : ℕ := 110
def population_1990_I : ℕ := 120
def population_2000_I : ℕ := 150
def population_1990_J : ℕ := 150
def population_2000_J : ℕ := 190

-- Define the function that calculates the percentage increase.
def percentage_increase (pop_1990 pop_2000 : ℕ) : ℚ :=
  (pop_2000 : ℚ) / (pop_1990 : ℚ)

-- Calculate the percentage increases for each city.
def percentage_increase_F := percentage_increase population_1990_F population_2000_F
def percentage_increase_G := percentage_increase population_1990_G population_2000_G
def percentage_increase_H := percentage_increase population_1990_H population_2000_H
def percentage_increase_I := percentage_increase population_1990_I population_2000_I
def percentage_increase_J := percentage_increase population_1990_J population_2000_J

-- Prove that City G has the greatest percentage increase.
theorem city_G_has_highest_percentage_increase :
  percentage_increase_G > percentage_increase_F ∧ 
  percentage_increase_G > percentage_increase_H ∧
  percentage_increase_G > percentage_increase_I ∧
  percentage_increase_G > percentage_increase_J :=
by sorry

end city_G_has_highest_percentage_increase_l4_4902


namespace fourth_person_height_is_82_l4_4118

theorem fourth_person_height_is_82 (H : ℕ)
    (h1: (H + (H + 2) + (H + 4) + (H + 10)) / 4 = 76)
    (h_diff1: H + 2 - H = 2)
    (h_diff2: H + 4 - (H + 2) = 2)
    (h_diff3: H + 10 - (H + 4) = 6) :
  (H + 10) = 82 := 
sorry

end fourth_person_height_is_82_l4_4118


namespace wind_velocity_determination_l4_4977

theorem wind_velocity_determination (ρ : ℝ) (P1 P2 : ℝ) (A1 A2 : ℝ) (V1 V2 : ℝ) (k : ℝ) :
  ρ = 1.2 →
  P1 = 0.75 →
  A1 = 2 →
  V1 = 12 →
  P1 = ρ * k * A1 * V1^2 →
  P2 = 20.4 →
  A2 = 10.76 →
  P2 = ρ * k * A2 * V2^2 →
  V2 = 27 := 
by sorry

end wind_velocity_determination_l4_4977


namespace contractor_laborers_l4_4278

theorem contractor_laborers (x : ℕ) (h : 9 * x = 15 * (x - 6)) : x = 15 :=
by
  sorry

end contractor_laborers_l4_4278


namespace semicircle_perimeter_l4_4532

/-- The perimeter of a semicircle with radius 6.3 cm is approximately 32.382 cm. -/
theorem semicircle_perimeter (r : ℝ) (h : r = 6.3) : 
  (π * r + 2 * r = 32.382) :=
by
  sorry

end semicircle_perimeter_l4_4532


namespace coins_left_zero_when_divided_by_9_l4_4144

noncomputable def smallestCoinCount (n: ℕ) : Prop :=
  n % 8 = 6 ∧ n % 7 = 5

theorem coins_left_zero_when_divided_by_9 (n : ℕ) (h : smallestCoinCount n) (h_min: ∀ m : ℕ, smallestCoinCount m → n ≤ m) :
  n % 9 = 0 :=
sorry

end coins_left_zero_when_divided_by_9_l4_4144


namespace Lorelei_vase_contains_22_roses_l4_4939

variable (redBush : ℕ) (pinkBush : ℕ) (yellowBush : ℕ) (orangeBush : ℕ)
variable (percentRed : ℚ) (percentPink : ℚ) (percentYellow : ℚ) (percentOrange : ℚ)

noncomputable def pickedRoses : ℕ :=
  let redPicked := redBush * percentRed
  let pinkPicked := pinkBush * percentPink
  let yellowPicked := yellowBush * percentYellow
  let orangePicked := orangeBush * percentOrange
  (redPicked + pinkPicked + yellowPicked + orangePicked).toNat

theorem Lorelei_vase_contains_22_roses 
  (redBush := 12) (pinkBush := 18) (yellowBush := 20) (orangeBush := 8)
  (percentRed := 0.5) (percentPink := 0.5) (percentYellow := 0.25) (percentOrange := 0.25)
  : pickedRoses redBush pinkBush yellowBush orangeBush percentRed percentPink percentYellow percentOrange = 22 := by 
  sorry

end Lorelei_vase_contains_22_roses_l4_4939


namespace complex_power_difference_l4_4075

theorem complex_power_difference (i : ℂ) (hi : i^2 = -1) : (1 + 2 * i)^8 - (1 - 2 * i)^8 = 672 * i := 
by
  sorry

end complex_power_difference_l4_4075


namespace candidate_B_valid_votes_l4_4748

theorem candidate_B_valid_votes:
  let eligible_voters := 12000
  let abstained_percent := 0.1
  let invalid_votes_percent := 0.2
  let votes_for_C_percent := 0.05
  let A_less_B_percent := 0.2
  let total_voted := (1 - abstained_percent) * eligible_voters
  let valid_votes := (1 - invalid_votes_percent) * total_voted
  let votes_for_C := votes_for_C_percent * valid_votes
  (∃ Vb, valid_votes = (1 - A_less_B_percent) * Vb + Vb + votes_for_C 
         ∧ Vb = 4560) :=
sorry

end candidate_B_valid_votes_l4_4748


namespace simplify_cubic_root_l4_4809

theorem simplify_cubic_root : 
  (∛(54880000) = 20 * ∛((5^2) * 137)) :=
sorry

end simplify_cubic_root_l4_4809


namespace quadratic_eq_zero_l4_4219

theorem quadratic_eq_zero (x a b : ℝ) (h : x = a ∨ x = b) : x^2 - (a + b) * x + a * b = 0 :=
by sorry

end quadratic_eq_zero_l4_4219


namespace find_number_l4_4283

theorem find_number (x : ℝ) (h : 3034 - (x / 20.04) = 2984) : x = 1002 :=
by
  sorry

end find_number_l4_4283


namespace integer_sequence_count_l4_4398

theorem integer_sequence_count (a₀ : ℕ) (step : ℕ → ℕ) (n : ℕ) 
  (h₀ : a₀ = 5184)
  (h_step : ∀ k, k < n → step k = (a₀ / 4^k))
  (h_stop : a₀ = (4 ^ (n - 1)) * 81) :
  n = 4 := 
sorry

end integer_sequence_count_l4_4398


namespace smallest_number_conditions_l4_4157

theorem smallest_number_conditions :
  ∃ n : ℤ, (n > 0) ∧
           (n % 2 = 1) ∧
           (n % 3 = 1) ∧
           (n % 4 = 1) ∧
           (n % 5 = 1) ∧
           (n % 6 = 1) ∧
           (n % 11 = 0) ∧
           (∀ m : ℤ, (m > 0) → 
             (m % 2 = 1) ∧
             (m % 3 = 1) ∧
             (m % 4 = 1) ∧
             (m % 5 = 1) ∧
             (m % 6 = 1) ∧
             (m % 11 = 0) → 
             (n ≤ m)) :=
sorry

end smallest_number_conditions_l4_4157


namespace contrapositive_of_proposition_l4_4818

theorem contrapositive_of_proposition :
  (∀ x : ℝ, x ≤ -3 → x < 0) ↔ (∀ x : ℝ, x ≥ 0 → x > -3) :=
by
  sorry

end contrapositive_of_proposition_l4_4818


namespace sin_330_value_l4_4015

noncomputable def sin_330 : ℝ := Real.sin (330 * Real.pi / 180)

theorem sin_330_value : sin_330 = -1/2 :=
by {
  sorry
}

end sin_330_value_l4_4015


namespace right_triangle_area_l4_4834

/-- Given a right triangle with hypotenuse 13 meters and one side 5 meters,
prove that the area of the triangle is 30 square meters. -/
theorem right_triangle_area (a b c : ℝ) (h : a^2 + b^2 = c^2) (hc : c = 13) (ha : a = 5) :
  1/2 * a * b = 30 :=
by sorry

end right_triangle_area_l4_4834


namespace expr_containing_x_to_y_l4_4628

theorem expr_containing_x_to_y (x y : ℝ) (h : 2 * x - y = 3) : y = 2 * x - 3 :=
by
  -- proof steps would be here
  sorry

end expr_containing_x_to_y_l4_4628


namespace fractions_simplify_to_prime_denominator_2023_l4_4459

def num_fractions_simplifying_to_prime_denominator (n: ℕ) (p q: ℕ) : ℕ :=
  let multiples (m: ℕ) : ℕ := (n - 1) / m
  multiples p + multiples (p * q)

theorem fractions_simplify_to_prime_denominator_2023 :
  num_fractions_simplifying_to_prime_denominator 2023 17 7 = 22 :=
by
  sorry

end fractions_simplify_to_prime_denominator_2023_l4_4459


namespace part1_part2_l4_4053

def A : Set ℝ := {x | x^2 + x - 12 < 0}
def B : Set ℝ := {x | 4 / (x + 3) ≤ 1}
def C (m : ℝ) : Set ℝ := {x | x^2 - 2 * m * x + m^2 - 1 ≤ 0}

theorem part1 : A ∩ B = {x | -4 < x ∧ x < -3 ∨ 1 ≤ x ∧ x < 3} := sorry

theorem part2 (m : ℝ) : (-3 < m ∧ m < 2) ↔ ∀ x, (x ∈ A → x ∈ C m) ∧ ∃ x, x ∈ C m ∧ x ∉ A := sorry

end part1_part2_l4_4053


namespace min_value_of_function_l4_4253

-- Define the function f
def f (x : ℝ) := 3 * x^2 - 6 * x + 9

-- State the theorem about the minimum value of the function.
theorem min_value_of_function : ∀ x : ℝ, f x ≥ 6 := by
  sorry

end min_value_of_function_l4_4253


namespace exp_periodic_cos_sin_euler_formula_exp_13_pi_by_2_equals_i_l4_4887

theorem exp_periodic_cos_sin (x : ℝ) : ∃ k : ℤ, cos(x) = cos(x + 2 * k * π) ∧ sin(x) = sin(x + 2 * k * π) :=
begin
  use 1,
  split;
  apply real.cos_periodic;
  exact int.cast_coe_int (1 : ℤ)
end

theorem euler_formula (x : ℝ) : complex.exp (x * complex.I) = complex.cos x + complex.I * complex.sin x :=
by sorry

theorem exp_13_pi_by_2_equals_i : complex.exp (13 * real.pi / 2 * complex.I) = complex.I :=
begin
  -- Use Euler's formula
  have h_euler : complex.exp (13 * real.pi / 2 * complex.I) = complex.cos (13 * real.pi / 2) + complex.I * complex.sin (13 * real.pi / 2),
  { apply euler_formula },

  -- Simplify the angle by periodicity
  have h_angle : 13 * real.pi / 2 = 6 * real.pi + real.pi / 2,
  { field_simp, ring },
  
  -- Cos and Sin periodicity with 2π
  have h_cos : complex.cos (6 * real.pi + real.pi / 2) = complex.cos (real.pi / 2),
  { rw [← complex.cos_add_period],
    apply exp_periodic_cos_sin,
  },
  
  have h_sin : complex.sin (6 * real.pi + real.pi / 2) = complex.sin (real.pi / 2),
  { rw [← complex.sin_add_period],
    apply exp_periodic_cos_sin,
  },

  -- Calculate Cos(real.pi / 2) and Sin(real.pi / 2)
  have h_cos_pi_by_2 : complex.cos (real.pi / 2) = 0,
  { apply complex.cos_pi_div_two },
  
  have h_sin_pi_by_2 : complex.sin (real.pi / 2) = 1,
  { apply complex.sin_pi_div_two },
  
  -- Combine results
  rw [h_euler, h_angle, h_cos, h_sin],
  rw [h_cos_pi_by_2, h_sin_pi_by_2],
  ring,
end

end exp_periodic_cos_sin_euler_formula_exp_13_pi_by_2_equals_i_l4_4887


namespace simplify_expression_l4_4789

noncomputable def proof_problem (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) : Prop :=
  (1 / (1 + a + a * b) + 1 / (1 + b + b * c) + 1 / (1 + c + c * a)) = 1

theorem simplify_expression (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) :
  proof_problem a b c h h_abc :=
by sorry

end simplify_expression_l4_4789


namespace find_fraction_of_ab_l4_4085

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def x := a / b

theorem find_fraction_of_ab (h1 : a ≠ b) (h2 : a / b + (3 * a + 4 * b) / (b + 12 * a) = 2) :
  a / b = (5 - Real.sqrt 19) / 6 :=
sorry

end find_fraction_of_ab_l4_4085


namespace fermats_little_theorem_l4_4802

theorem fermats_little_theorem (p : ℕ) (a : ℕ) (hp : Nat.Prime p) (ha : ¬ p ∣ a) :
  a^(p-1) ≡ 1 [MOD p] :=
sorry

end fermats_little_theorem_l4_4802


namespace largest_integral_solution_l4_4715

theorem largest_integral_solution : ∃ x : ℤ, (1 / 4 < x / 7 ∧ x / 7 < 3 / 5) ∧ ∀ y : ℤ, (1 / 4 < y / 7 ∧ y / 7 < 3 / 5) → y ≤ x := sorry

end largest_integral_solution_l4_4715


namespace solve_for_product_l4_4276

theorem solve_for_product (a b c d : ℚ) (h1 : 3 * a + 4 * b + 6 * c + 8 * d = 48)
                          (h2 : 4 * (d + c) = b) 
                          (h3 : 4 * b + 2 * c = a) 
                          (h4 : c - 2 = d) : 
                          a * b * c * d = -1032192 / 1874161 := 
by 
  sorry

end solve_for_product_l4_4276


namespace slope_magnitude_l4_4914

-- Definitions based on given conditions
def parabola : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ y^2 = 4 * x }
def line (k m : ℝ) : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ y = k * x + m }
def focus : ℝ × ℝ := (1, 0)
def intersects (l p : Set (ℝ × ℝ)) : Prop := ∃ x1 y1 x2 y2, (x1, y1) ∈ l ∧ (x1, y1) ∈ p ∧ (x2, y2) ∈ l ∧ (x2, y2) ∈ p ∧ (x1, y1) ≠ (x2, y2)

theorem slope_magnitude (k m : ℝ) (h_k_nonzero : k ≠ 0) 
  (h_intersects : intersects (line k m) parabola) 
  (h_AF_2FB : ∀ x1 y1 x2 y2, (x1, y1) ∈ line k m → (x1, y1) ∈ parabola → 
                          (x2, y2) ∈ line k m → (x2, y2) ∈ parabola → 
                          (1 - x1 = 2 * (x2 - 1)) ∧ (-y1 = 2 * y2)) :
  |k| = 2 * Real.sqrt 2 :=
sorry

end slope_magnitude_l4_4914


namespace periodic_sum_constant_l4_4266

noncomputable def is_periodic (f : ℝ → ℝ) (a : ℝ) : Prop :=
a ≠ 0 ∧ ∀ x : ℝ, f (a + x) = f x

theorem periodic_sum_constant (f g : ℝ → ℝ) (a b : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hfa : is_periodic f a) (hgb : is_periodic g b)
  (harational : ∃ m n : ℤ, (a : ℝ) = m / n) (hbirrational : ¬ ∃ m n : ℤ, (b : ℝ) = m / n) :
  (∃ c : ℝ, c ≠ 0 ∧ ∀ x : ℝ, (f + g) (c + x) = (f + g) x) →
  (∀ x : ℝ, f x = f 0) ∨ (∀ x : ℝ, g x = g 0) :=
sorry

end periodic_sum_constant_l4_4266


namespace whale_ninth_hour_consumption_l4_4878

-- Define the arithmetic sequence conditions
def first_hour_consumption : ℕ := 10
def common_difference : ℕ := 5

-- Define the total consumption over 12 hours
def total_consumption := 12 * (first_hour_consumption + (first_hour_consumption + 11 * common_difference)) / 2

-- Prove the ninth hour (which is the 8th term) consumption
theorem whale_ninth_hour_consumption :
  total_consumption = 450 →
  first_hour_consumption + 8 * common_difference = 50 := 
by
  intros h
  sorry
  

end whale_ninth_hour_consumption_l4_4878


namespace incorrect_transformation_l4_4994

theorem incorrect_transformation :
  ¬ ∀ (a b c : ℝ), ac = bc → a = b :=
by
  sorry

end incorrect_transformation_l4_4994


namespace product_gcd_lcm_eq_1296_l4_4317

theorem product_gcd_lcm_eq_1296 : (Int.gcd 24 54) * (Int.lcm 24 54) = 1296 := by
  sorry

end product_gcd_lcm_eq_1296_l4_4317


namespace complete_square_l4_4856

theorem complete_square (x : ℝ) : (x^2 - 2 * x - 5 = 0) ↔ ((x - 1)^2 = 6) := 
by
  sorry

end complete_square_l4_4856


namespace water_depth_is_60_l4_4300

def Ron_height : ℕ := 12
def depth_of_water (h_R : ℕ) : ℕ := 5 * h_R

theorem water_depth_is_60 : depth_of_water Ron_height = 60 :=
by
  sorry

end water_depth_is_60_l4_4300


namespace carol_age_difference_l4_4980

theorem carol_age_difference (bob_age carol_age : ℕ) (h1 : bob_age + carol_age = 66)
  (h2 : carol_age = 3 * bob_age + 2) (h3 : bob_age = 16) (h4 : carol_age = 50) :
  carol_age - 3 * bob_age = 2 :=
by
  sorry

end carol_age_difference_l4_4980


namespace dist_eq_iff_cond_prob_eq_l4_4497

variables {Ω : Type*} {F : measurable_space Ω} {P : measure Ω} 
variables {X Y Z : Ω → ℝ} {A : set ℝ}

theorem dist_eq_iff_cond_prob_eq :
  (∀ A ∈ borel ℝ, P {ω | X ω ∈ A ∧ Y ω ∈ (set.univ : set Ω)} = P {ω | Z ω ∈ A ∧ Y ω ∈ (set.univ : set Ω)}) ↔ 
  (∀ A ∈ borel ℝ, P {ω | X ω ∈ A | Y} = P {ω | Z ω ∈ A | Y}) :=
sorry

end dist_eq_iff_cond_prob_eq_l4_4497


namespace largest_possible_b_l4_4981

theorem largest_possible_b (a b c : ℕ) (h₁ : 1 < c) (h₂ : c < b) (h₃ : b < a) (h₄ : a * b * c = 360): b = 12 :=
by
  sorry

end largest_possible_b_l4_4981


namespace seq_a5_eq_one_ninth_l4_4048

theorem seq_a5_eq_one_ninth (a : ℕ → ℚ) (h1 : a 1 = 1) (h_rec : ∀ n, a (n + 1) = a n / (2 * a n + 1)) :
  a 5 = 1 / 9 :=
sorry

end seq_a5_eq_one_ninth_l4_4048


namespace number_of_dolls_combined_l4_4161

-- Defining the given conditions as variables
variables (aida sophie vera : ℕ)

-- Given conditions
def condition1 : Prop := aida = 2 * sophie
def condition2 : Prop := sophie = 2 * vera
def condition3 : Prop := vera = 20

-- The final proof statement we need to prove
theorem number_of_dolls_combined (h1 : condition1 aida sophie) (h2 : condition2 sophie vera) (h3 : condition3 vera) : 
  aida + sophie + vera = 140 :=
  by sorry

end number_of_dolls_combined_l4_4161


namespace books_fraction_sold_l4_4871

theorem books_fraction_sold (B : ℕ) (h1 : B - 36 * 2 = 144) :
  (B - 36) / B = 2 / 3 := by
  sorry

end books_fraction_sold_l4_4871


namespace sin_330_eq_neg_half_l4_4017

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  -- Proof would go here
  sorry

end sin_330_eq_neg_half_l4_4017


namespace find_fourth_root_l4_4120

theorem find_fourth_root (b c α : ℝ)
  (h₁ : b * (-3)^4 + (b + 3 * c) * (-3)^3 + (c - 4 * b) * (-3)^2 + (19 - b) * (-3) - 2 = 0)
  (h₂ : b * 4^4 + (b + 3 * c) * 4^3 + (c - 4 * b) * 4^2 + (19 - b) * 4 - 2 = 0)
  (h₃ : b * 2^4 + (b + 3 * c) * 2^3 + (c - 4 * b) * 2^2 + (19 - b) * 2 - 2 = 0)
  (h₄ : (-3) + 4 + 2 + α = 2)
  : α = 1 :=
sorry

end find_fourth_root_l4_4120


namespace sin_330_value_l4_4007

noncomputable def sin_330 : ℝ := Real.sin (330 * Real.pi / 180)

theorem sin_330_value : sin_330 = -1/2 :=
by {
  sorry
}

end sin_330_value_l4_4007


namespace lorelei_vase_rose_count_l4_4938

theorem lorelei_vase_rose_count :
  let r := 12
  let p := 18
  let y := 20
  let o := 8
  let picked_r := 0.5 * r
  let picked_p := 0.5 * p
  let picked_y := 0.25 * y
  let picked_o := 0.25 * o
  picked_r + picked_p + picked_y + picked_o = 22 := 
by
  sorry

end lorelei_vase_rose_count_l4_4938


namespace Katy_jellybeans_l4_4964

variable (Matt Matilda Steve Katy : ℕ)

def jellybean_relationship (Matt Matilda Steve Katy : ℕ) : Prop :=
  (Matt = 10 * Steve) ∧
  (Matilda = Matt / 2) ∧
  (Steve = 84) ∧
  (Katy = 3 * Matilda) ∧
  (Katy = Matt / 2)

theorem Katy_jellybeans : ∃ Katy, jellybean_relationship Matt Matilda Steve Katy ∧ Katy = 1260 := by
  sorry

end Katy_jellybeans_l4_4964


namespace sequence_values_l4_4302

theorem sequence_values (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
    (h_arith : 2 + (a - 2) = a + (b - a)) (h_geom : a * a = b * (9 / b)) : a = 4 ∧ b = 6 :=
by
  -- insert proof here
  sorry

end sequence_values_l4_4302


namespace stable_state_exists_l4_4260

-- Definition of the problem
theorem stable_state_exists 
(N : ℕ) (N_ge_3 : N ≥ 3) (letters : Fin N → Fin 3) 
(perform_operation : ∀ (letters : Fin N → Fin 3), Fin N → Fin 3)
(stable : ∀ (letters : Fin N → Fin 3), Prop)
(initial_state : Fin N → Fin 3):
  ∃ (state : Fin N → Fin 3), (∀ i, perform_operation state i = state i) ∧ stable state :=
sorry

end stable_state_exists_l4_4260


namespace unable_to_determine_questions_answered_l4_4761

variable (total_questions : ℕ) (total_time : ℕ) (used_time : ℕ) (remaining_time : ℕ)

theorem unable_to_determine_questions_answered (total_questions_eq : total_questions = 80)
  (total_time_eq : total_time = 60)
  (used_time_eq : used_time = 12)
  (remaining_time_eq : remaining_time = 0) :
  ∀ (answered_rate : ℕ → ℕ), ¬ ∃ questions_answered, answered_rate used_time = questions_answered :=
by sorry

end unable_to_determine_questions_answered_l4_4761


namespace truck_capacity_l4_4892

-- Definitions based on conditions
def initial_fuel : ℕ := 38
def total_money : ℕ := 350
def change : ℕ := 14
def cost_per_liter : ℕ := 3

-- Theorem statement
theorem truck_capacity :
  initial_fuel + (total_money - change) / cost_per_liter = 150 := by
  sorry

end truck_capacity_l4_4892


namespace minimum_trucks_needed_l4_4383

theorem minimum_trucks_needed 
  (total_weight : ℕ) (box_weight: ℕ) (truck_capacity: ℕ) (min_trucks: ℕ)
  (h_total_weight : total_weight = 10)
  (h_box_weight_le : ∀ (w : ℕ), w <= box_weight → w <= 1)
  (h_truck_capacity : truck_capacity = 3)
  (h_min_trucks : min_trucks = 5) : 
  min_trucks >= (total_weight / truck_capacity) :=
sorry

end minimum_trucks_needed_l4_4383


namespace probability_first_hearts_second_ace_correct_l4_4263

noncomputable def probability_first_hearts_second_ace : ℚ :=
  let total_cards := 104
  let total_aces := 8 -- 4 aces per deck, 2 decks
  let hearts_count := 2 * 13 -- 13 hearts per deck, 2 decks
  let ace_of_hearts_count := 2

  -- Case 1: the first is an ace of hearts
  let prob_first_ace_of_hearts := (ace_of_hearts_count : ℚ) / total_cards
  let prob_second_ace_given_first_ace_of_hearts := (total_aces - 1 : ℚ) / (total_cards - 1)

  -- Case 2: the first is a hearts but not an ace
  let prob_first_hearts_not_ace := (hearts_count - ace_of_hearts_count : ℚ) / total_cards
  let prob_second_ace_given_first_hearts_not_ace := total_aces / (total_cards - 1)

  -- Combined probability
  (prob_first_ace_of_hearts * prob_second_ace_given_first_ace_of_hearts) +
  (prob_first_hearts_not_ace * prob_second_ace_given_first_hearts_not_ace)

theorem probability_first_hearts_second_ace_correct : 
  probability_first_hearts_second_ace = 7 / 453 := 
sorry

end probability_first_hearts_second_ace_correct_l4_4263


namespace expression_evaluation_l4_4534

variable (a b : ℝ)

theorem expression_evaluation (h : a + b = 1) :
  a^3 + b^3 + 3 * (a^3 * b + a * b^3) + 6 * (a^3 * b^2 + a^2 * b^3) = 1 :=
by
  sorry

end expression_evaluation_l4_4534


namespace magnitude_inverse_sum_eq_l4_4372

noncomputable def complex_magnitude_inverse_sum (z w : ℂ) (hz : |z| = 2) (hw : |w| = 4) (hzw : |z + w| = 3) : ℝ :=
|1/z + 1/w|

theorem magnitude_inverse_sum_eq (z w : ℂ) (hz : |z| = 2) (hw : |w| = 4) (hzw : |z + w| = 3) :
  complex_magnitude_inverse_sum z w hz hw hzw = 3 / 8 :=
by sorry

end magnitude_inverse_sum_eq_l4_4372


namespace determine_marriages_l4_4844

-- Definitions of the items each person bought
variable (a_items b_items c_items : ℕ) -- Number of items bought by wives a, b, and c
variable (A_items B_items C_items : ℕ) -- Number of items bought by husbands A, B, and C

-- Conditions
variable (spend_eq_square_a : a_items * a_items = a_spend) -- Spending equals square of items
variable (spend_eq_square_b : b_items * b_items = b_spend)
variable (spend_eq_square_c : c_items * c_items = c_spend)
variable (spend_eq_square_A : A_items * A_items = A_spend)
variable (spend_eq_square_B : B_items * B_items = B_spend)
variable (spend_eq_square_C : C_items * C_items = C_spend)

variable (A_spend_eq : A_spend = a_spend + 48) -- Husbands spent 48 yuan more than wives
variable (B_spend_eq : B_spend = b_spend + 48)
variable (C_spend_eq : C_spend = c_spend + 48)

variable (A_bought_9_more : A_items = b_items + 9) -- A bought 9 more items than b
variable (B_bought_7_more : B_items = a_items + 7) -- B bought 7 more items than a

-- Theorem statement
theorem determine_marriages (hA : A_items ≥ b_items + 9) (hB : B_items ≥ a_items + 7) :
  (A_spend = A_items * A_items) ∧ (B_spend = B_items * B_items) ∧ (C_spend = C_items * C_items) ∧
  (a_spend = a_items * a_items) ∧ (b_spend = b_items * b_items) ∧ (c_spend = c_items * c_items) →
  (A_spend = a_spend + 48) ∧ (B_spend = b_spend + 48) ∧ (C_spend = c_spend + 48) →
  (A_items = b_items + 9) ∧ (B_items = a_items + 7) →
  (A_items = 13 ∧ c_items = 11) ∧ (B_items = 8 ∧ b_items = 4) ∧ (C_items = 7 ∧ a_items = 1) :=
by
  sorry

end determine_marriages_l4_4844


namespace evaluate_expr_l4_4559

theorem evaluate_expr : Int.ceil (5 / 4 : ℚ) + Int.floor (-5 / 4 : ℚ) = 0 := by
  sorry

end evaluate_expr_l4_4559


namespace lcm_18_24_30_l4_4521

theorem lcm_18_24_30 :
  let a := 18
  let b := 24
  let c := 30
  let lcm := 360
  (∀ x > 0, x ∣ a ∧ x ∣ b ∧ x ∣ c → x ∣ lcm) ∧ (∀ y > 0, y ∣ lcm → y ∣ a ∧ y ∣ b ∧ y ∣ c) :=
by {
  let a := 18
  let b := 24
  let c := 30
  let lcm := 360
  sorry
}

end lcm_18_24_30_l4_4521


namespace how_many_cheburashkas_erased_l4_4770

theorem how_many_cheburashkas_erased 
  (total_krakozyabras : ℕ)
  (characters_per_row_initial : ℕ) 
  (total_characters_initial : ℕ)
  (total_cheburashkas : ℕ)
  (total_rows : ℕ := 2)
  (total_krakozyabras := 29) :
  total_cheburashkas = 11 :=
by
  sorry

end how_many_cheburashkas_erased_l4_4770


namespace sin_330_eq_neg_sqrt3_div_2_l4_4005

theorem sin_330_eq_neg_sqrt3_div_2 
  (R : ℝ × ℝ)
  (hR : R = (1/2, -sqrt(3)/2))
  : Real.sin (330 * Real.pi / 180) = -sqrt(3)/2 :=
by
  sorry

end sin_330_eq_neg_sqrt3_div_2_l4_4005


namespace factorize_expression_l4_4712

theorem factorize_expression (a x : ℝ) :
  a * x^2 - 2 * a * x + a = a * (x - 1) ^ 2 := 
sorry

end factorize_expression_l4_4712


namespace probability_king_of_diamonds_top_two_l4_4292

-- Definitions based on the conditions
def total_cards : ℕ := 54
def king_of_diamonds : ℕ := 1
def jokers : ℕ := 2

-- The main theorem statement proving the probability
theorem probability_king_of_diamonds_top_two :
  let prob := (king_of_diamonds / total_cards) + ((total_cards - 1) / total_cards * king_of_diamonds / (total_cards - 1))
  prob = 1 / 27 :=
by
  sorry

end probability_king_of_diamonds_top_two_l4_4292


namespace total_seeds_in_garden_l4_4732

-- Definitions based on the conditions
def top_bed_rows : ℕ := 4
def top_bed_seeds_per_row : ℕ := 25
def num_top_beds : ℕ := 2

def medium_bed_rows : ℕ := 3
def medium_bed_seeds_per_row : ℕ := 20
def num_medium_beds : ℕ := 2

-- Calculation of total seeds in top beds
def seeds_per_top_bed : ℕ := top_bed_rows * top_bed_seeds_per_row
def total_seeds_top_beds : ℕ := num_top_beds * seeds_per_top_bed

-- Calculation of total seeds in medium beds
def seeds_per_medium_bed : ℕ := medium_bed_rows * medium_bed_seeds_per_row
def total_seeds_medium_beds : ℕ := num_medium_beds * seeds_per_medium_bed

-- Proof goal
theorem total_seeds_in_garden : total_seeds_top_beds + total_seeds_medium_beds = 320 :=
by
  sorry

end total_seeds_in_garden_l4_4732


namespace cards_per_layer_l4_4958

theorem cards_per_layer (total_decks : ℕ) (cards_per_deck : ℕ) (layers : ℕ) (h_decks : total_decks = 16) (h_cards_per_deck : cards_per_deck = 52) (h_layers : layers = 32) :
  total_decks * cards_per_deck / layers = 26 :=
by {
  -- To skip the proof
  sorry
}

end cards_per_layer_l4_4958


namespace specific_divisors_count_l4_4614

-- Declare the value of n
def n : ℕ := (2^40) * (3^25) * (5^10)

-- Definition to count the number of positive divisors of a number less than n that don't divide n.
def count_specific_divisors (n : ℕ) : ℕ :=
sorry  -- This would be the function implementation

-- Lean statement to assert the number of such divisors
theorem specific_divisors_count : 
  count_specific_divisors n = 31514 :=
sorry

end specific_divisors_count_l4_4614


namespace rectangle_length_eq_fifty_l4_4098

theorem rectangle_length_eq_fifty (x : ℝ) :
  (∃ w : ℝ, 6 * x * w = 6000 ∧ w = (2 / 5) * x) → x = 50 :=
by
  sorry

end rectangle_length_eq_fifty_l4_4098


namespace derivative_of_y_l4_4060

noncomputable def y (x : ℝ) : ℝ := (Real.log x) / x + x * Real.exp x

theorem derivative_of_y (x : ℝ) (hx : x > 0) : 
  deriv y x = (1 - Real.log x) / (x^2) + (x + 1) * Real.exp x := by
  sorry

end derivative_of_y_l4_4060


namespace complement_of_P_in_U_l4_4589

def universal_set : Set ℝ := Set.univ
def set_P : Set ℝ := { x | x^2 - 5 * x - 6 ≥ 0 }
def complement_in_U (U : Set ℝ) (P : Set ℝ) : Set ℝ := U \ P

theorem complement_of_P_in_U :
  complement_in_U universal_set set_P = { x | -1 < x ∧ x < 6 } :=
by
  sorry

end complement_of_P_in_U_l4_4589


namespace cost_of_parts_per_tire_repair_is_5_l4_4945

-- Define the given conditions
def charge_per_tire_repair : ℤ := 20
def num_tire_repairs : ℤ := 300
def charge_per_complex_repair : ℤ := 300
def num_complex_repairs : ℤ := 2
def cost_per_complex_repair_parts : ℤ := 50
def retail_shop_profit : ℤ := 2000
def fixed_expenses : ℤ := 4000
def total_profit : ℤ := 3000

-- Define the calculation for total revenue
def total_revenue : ℤ := 
    (charge_per_tire_repair * num_tire_repairs) + 
    (charge_per_complex_repair * num_complex_repairs) + 
    retail_shop_profit

-- Define the calculation for total expenses
def total_expenses : ℤ := total_revenue - total_profit

-- Define the calculation for parts cost of tire repairs
def parts_cost_tire_repairs : ℤ := 
    total_expenses - (cost_per_complex_repair_parts * num_complex_repairs) - fixed_expenses

def cost_per_tire_repair : ℤ := parts_cost_tire_repairs / num_tire_repairs

-- The statement to be proved
theorem cost_of_parts_per_tire_repair_is_5 : cost_per_tire_repair = 5 := by
    sorry

end cost_of_parts_per_tire_repair_is_5_l4_4945


namespace gcd_lcm_product_24_54_l4_4318

theorem gcd_lcm_product_24_54 :
  let a := 24 in
  let b := 54 in
  let gcd_ab := Int.gcd a b in
  let lcm_ab := Int.lcm a b in
  gcd_ab * lcm_ab = a * b := by
  let a := 24
  let b := 54
  have gcd_ab : Int.gcd a b = 6 := by
    rw [Int.gcd_eq_right_iff_dvd.mpr (dvd.intro 9 rfl)]
    
  have lcm_ab : Int.lcm a b = 216 := by
    sorry -- We simply add sorry here for the sake of completeness

  show gcd_ab * lcm_ab = a * b
  rw [gcd_ab, lcm_ab]
  norm_num

end gcd_lcm_product_24_54_l4_4318


namespace coefficient_x2y6_expansion_l4_4751

theorem coefficient_x2y6_expansion :
  let x : ℤ := 1
  let y : ℤ := 1
  ∃ a : ℤ, a = -28 ∧ (a • x ^ 2 * y ^ 6) = (1 - y / x) * (x + y) ^ 8 :=
by
  sorry

end coefficient_x2y6_expansion_l4_4751


namespace find_length_of_segment_l4_4838

noncomputable def radius : ℝ := 4
noncomputable def volume_cylinder (L : ℝ) : ℝ := 16 * Real.pi * L
noncomputable def volume_hemispheres : ℝ := 2 * (128 / 3) * Real.pi
noncomputable def total_volume (L : ℝ) : ℝ := volume_cylinder L + volume_hemispheres

theorem find_length_of_segment (L : ℝ) (h : total_volume L = 544 * Real.pi) : 
  L = 86 / 3 :=
by sorry

end find_length_of_segment_l4_4838


namespace jogging_time_two_weeks_l4_4486

-- Definition for the daily jogging time in hours
def daily_jogging_time : ℝ := 1 + 30 / 60

-- Definition for the total jogging time over one week
def weekly_jogging_time : ℝ := daily_jogging_time * 7

-- Lean statement to prove that the total time jogging over two weeks is 21 hours
theorem jogging_time_two_weeks : weekly_jogging_time * 2 = 21 := by
  -- Placeholder for the proof
  sorry

end jogging_time_two_weeks_l4_4486


namespace problem_l4_4088

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

-- Conditions
def condition1 : a + b = 1 := sorry
def condition2 : a^2 + b^2 = 3 := sorry
def condition3 : a^3 + b^3 = 4 := sorry
def condition4 : a^4 + b^4 = 7 := sorry

-- Question and proof
theorem problem : a^10 + b^10 = 123 :=
by
  have h1 : a + b = 1 := condition1
  have h2 : a^2 + b^2 = 3 := condition2
  have h3 : a^3 + b^3 = 4 := condition3
  have h4 : a^4 + b^4 = 7 := condition4
  sorry

end problem_l4_4088


namespace solve_for_b_l4_4594

theorem solve_for_b (b : ℚ) (h : b - b / 4 = 5 / 2) : b = 10 / 3 :=
by 
  sorry

end solve_for_b_l4_4594


namespace roof_area_l4_4256

-- Definitions of the roof's dimensions based on the given conditions.
def length (w : ℝ) := 4 * w
def width (w : ℝ) := w
def difference (l w : ℝ) := l - w
def area (l w : ℝ) := l * w

-- The proof problem: Given the conditions, prove the area is 576 square feet.
theorem roof_area : ∀ w : ℝ, (length w) - (width w) = 36 → area (length w) (width w) = 576 := by
  intro w
  intro h_diff
  sorry

end roof_area_l4_4256


namespace x_is_perfect_square_l4_4512

theorem x_is_perfect_square (x y : ℕ) (hxy : x > y) (hdiv : xy ∣ x ^ 2022 + x + y ^ 2) : ∃ n : ℕ, x = n^2 := 
sorry

end x_is_perfect_square_l4_4512


namespace arithmetic_common_difference_l4_4452

variable {α : Type*} [LinearOrderedField α]

-- Definition of arithmetic sequence
def arithmetic_seq (a : α) (d : α) (n : ℕ) : α :=
  a + (n - 1) * d

-- Definition of sum of the first n terms of an arithmetic sequence
def sum_arithmetic_seq (a : α) (d : α) (n : ℕ) : α :=
  n * a + (n * (n - 1) / 2) * d

theorem arithmetic_common_difference (a10 : α) (s10 : α) (d : α) (a1 : α) :
  arithmetic_seq a1 d 10 = a10 →
  sum_arithmetic_seq a1 d 10 = s10 →
  d = 2 / 3 :=
by
  sorry

end arithmetic_common_difference_l4_4452


namespace ratio_of_remaining_areas_of_squares_l4_4499

/--
  Given:
  - Square C has a side length of 48 cm.
  - Square D has a side length of 60 cm.
  - A smaller square of side length 12 cm is cut out from both squares.

  Show that:
  - The ratio of the remaining area of square C to the remaining area of square D is 5/8.
-/
theorem ratio_of_remaining_areas_of_squares : 
  let sideC := 48
  let sideD := 60
  let sideSmall := 12
  let areaC := sideC * sideC
  let areaD := sideD * sideD
  let areaSmall := sideSmall * sideSmall
  let remainingC := areaC - areaSmall
  let remainingD := areaD - areaSmall
  (remainingC : ℚ) / remainingD = 5 / 8 :=
by
  sorry

end ratio_of_remaining_areas_of_squares_l4_4499


namespace inscribed_rectangle_area_l4_4632

variable (a b h x : ℝ)
variable (h_pos : 0 < h) (a_b_pos : a > b) (b_pos : b > 0) (a_pos : a > 0) (x_pos : 0 < x) (hx : x < h)

theorem inscribed_rectangle_area (hb : b > 0) (ha : a > 0) (hx : 0 < x) (hxa : x < h) : 
  x * (a - b) * (h - x) / h = x * (a - b) * (h - x) / h := by
  sorry

end inscribed_rectangle_area_l4_4632


namespace inequality_proof_l4_4801

theorem inequality_proof (x y z : ℝ) : 
    x^4 + y^4 + z^2 + 1 ≥ 2 * x * (x * y^2 - x + z + 1) :=
by
  sorry

end inequality_proof_l4_4801


namespace factorization_correct_l4_4701

-- Define the initial expression
def initial_expr (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + a

-- Define the factorized form
def factorized_expr (a x : ℝ) : ℝ := a * (x - 1)^2

-- Create the theorem statement
theorem factorization_correct (a x : ℝ) : initial_expr a x = factorized_expr a x :=
by simp [initial_expr, factorized_expr, pow_two, mul_add, add_mul, add_assoc]; sorry

end factorization_correct_l4_4701


namespace price_of_second_tea_l4_4357

theorem price_of_second_tea (price_first_tea : ℝ) (mixture_price : ℝ) (required_ratio : ℝ) (price_second_tea : ℝ) :
  price_first_tea = 62 → mixture_price = 64.5 → required_ratio = 3 → price_second_tea = 65.33 :=
by
  intros h1 h2 h3
  sorry

end price_of_second_tea_l4_4357


namespace arithmetic_mean_fraction_l4_4123

theorem arithmetic_mean_fraction :
  let a := (3 : ℚ) / 4
  let b := (5 : ℚ) / 6
  let c := (9 : ℚ) / 10
  (1 / 3) * (a + b + c) = 149 / 180 :=
by 
  sorry

end arithmetic_mean_fraction_l4_4123


namespace cube_root_simplification_l4_4808

theorem cube_root_simplification : (∛54880000) = 140 * (2 ^ (1 / 3)) :=
by
  -- Using the information from the problem conditions and final solution.
  have root_10_cubed := (10 ^ 3 : ℝ)
  have factored_value := root_10_cubed * (2 ^ 4 * 7 ^ 3)
  have cube_root := Real.cbrt factored_value
  sorry

end cube_root_simplification_l4_4808


namespace smallest_positive_integer_l4_4308

theorem smallest_positive_integer
    (n : ℕ)
    (h : ∀ (a : Fin n → ℤ), ∃ (i j : Fin n), i ≠ j ∧ (2009 ∣ (a i + a j) ∨ 2009 ∣ (a i - a j))) : n = 1006 := by
  -- Proof is required here
  sorry

end smallest_positive_integer_l4_4308


namespace negation_of_P_l4_4726

-- Define the proposition P
def P : Prop := ∀ x : ℝ, x > Real.sin x

-- Formulate the negation of P
def neg_P : Prop := ∃ x : ℝ, x ≤ Real.sin x

-- State the theorem to be proved
theorem negation_of_P (hP : P) : neg_P :=
sorry

end negation_of_P_l4_4726


namespace value_of_M_l4_4461

theorem value_of_M
  (M : ℝ)
  (h : 25 / 100 * M = 55 / 100 * 4500) :
  M = 9900 :=
sorry

end value_of_M_l4_4461


namespace no_integer_solution_l4_4922

theorem no_integer_solution (y : ℤ) : ¬ (-3 * y ≥ y + 9 ∧ 2 * y ≥ 14 ∧ -4 * y ≥ 2 * y + 21) :=
sorry

end no_integer_solution_l4_4922


namespace cheburashkas_erased_l4_4781

def total_krakozyabras : ℕ := 29

def total_rows : ℕ := 2

def cheburashkas_per_row := (total_krakozyabras + total_rows) / total_rows / 2 + 1

theorem cheburashkas_erased :
  (total_krakozyabras + total_rows) / total_rows / 2 - 1 = 11 := 
by
  sorry

-- cheburashkas_erased proves that the number of Cheburashkas erased is 11 from the given conditions.

end cheburashkas_erased_l4_4781


namespace percentage_of_profits_l4_4934

variable (R P : ℝ) -- Let R be the revenues and P be the profits in the previous year
variable (H1 : (P/R) * 100 = 10) -- The condition we want to prove
variable (H2 : 0.95 * R) -- Revenues in 2009 are 0.95R
variable (H3 : 0.1 * 0.95 * R) -- Profits in 2009 are 0.1 * 0.95R = 0.095R
variable (H4 : 0.095 * R = 0.95 * P) -- The given relation between profits in 2009 and previous year

theorem percentage_of_profits (H1 : (P/R) * 100 = 10) 
  (H2 : ∀ (R : ℝ),  ∃ ρ, ρ = 0.95 * R)
  (H3 : ∀ (R : ℝ),  ∃ π, π = 0.10 * (0.95 * R))
  (H4 : ∀ (R P : ℝ), 0.095 * R = 0.95 * P) :
  ∀ (P R : ℝ), (P/R) * 100 = 10 := 
by
  sorry

end percentage_of_profits_l4_4934


namespace transform_f_to_g_l4_4211

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.sin x * Real.cos x
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)

theorem transform_f_to_g :
  ∀ x : ℝ, g x = f (x + (π / 8)) :=
by
  sorry

end transform_f_to_g_l4_4211


namespace balls_left_l4_4065

-- Define the conditions
def initial_balls : ℕ := 10
def removed_balls : ℕ := 3

-- The main statement to prove
theorem balls_left : initial_balls - removed_balls = 7 := by sorry

end balls_left_l4_4065


namespace gcd_count_count_numbers_l4_4046

open Nat

theorem gcd_count (n : ℕ) :
  n.between 1 150 → (∃ k : ℕ, n = 3 * k ∧ n % 7 ≠ 0) ↔ gcd 21 n = 3 :=
begin
  sorry
end

theorem count_numbers : ∃ N, (N = 43 ∧ ∀ n : ℕ, n.between 1 150 → gcd 21 n = 3 ↔ ∃ k : ℕ, n = 3 * k ∧ n % 7 ≠ 0) :=
begin
  use 43,
  split,
  { refl },
  { intro n, 
    rw gcd_count,
    sorry
  }
end

end gcd_count_count_numbers_l4_4046


namespace simplify_polynomial_l4_4244

theorem simplify_polynomial (q : ℚ) :
  (4 * q^3 - 7 * q^2 + 3 * q - 2) + (5 * q^2 - 9 * q + 8) = 4 * q^3 - 2 * q^2 - 6 * q + 6 := 
by 
  sorry

end simplify_polynomial_l4_4244


namespace jeremy_school_distance_l4_4361

theorem jeremy_school_distance (d : ℝ) (v : ℝ) :
  (d = v * 0.5) ∧
  (d = (v + 15) * 0.3) ∧
  (d = (v - 10) * (2 / 3)) →
  d = 15 :=
by 
  sorry

end jeremy_school_distance_l4_4361


namespace number_of_blue_tiles_is_16_l4_4635

def length_of_floor : ℕ := 20
def breadth_of_floor : ℕ := 10
def tile_length : ℕ := 2

def total_tiles : ℕ := (length_of_floor / tile_length) * (breadth_of_floor / tile_length)

def black_tiles : ℕ :=
  let rows_length := 2 * (length_of_floor / tile_length)
  let rows_breadth := 2 * (breadth_of_floor / tile_length)
  (rows_length + rows_breadth) - 4

def remaining_tiles : ℕ := total_tiles - black_tiles
def white_tiles : ℕ := remaining_tiles / 3
def blue_tiles : ℕ := remaining_tiles - white_tiles

theorem number_of_blue_tiles_is_16 :
  blue_tiles = 16 :=
by
  sorry

end number_of_blue_tiles_is_16_l4_4635


namespace Darnel_sprinted_further_l4_4307

-- Define the distances sprinted and jogged
def sprinted : ℝ := 0.88
def jogged : ℝ := 0.75

-- State the theorem to prove the main question
theorem Darnel_sprinted_further : sprinted - jogged = 0.13 :=
by
  sorry

end Darnel_sprinted_further_l4_4307


namespace sufficient_but_not_necessary_perpendicular_l4_4920

variable (x : ℝ)

/-- Define the vectors a and b. --/
def a := (1, 2 * x)
def b := (4, -x)

theorem sufficient_but_not_necessary_perpendicular :
  (sqrt 2 = x ∨ sqrt 2 = -x) ↔ a x • b x = (0 : ℝ) :=
by sorry

end sufficient_but_not_necessary_perpendicular_l4_4920


namespace remainder_y_div_13_l4_4273

def x (k : ℤ) : ℤ := 159 * k + 37
def y (x : ℤ) : ℤ := 5 * x^2 + 18 * x + 22

theorem remainder_y_div_13 (k : ℤ) : (y (x k)) % 13 = 8 := by
  sorry

end remainder_y_div_13_l4_4273


namespace train_length_eq_1800_l4_4395

theorem train_length_eq_1800 (speed_kmh : ℕ) (time_sec : ℕ) (distance : ℕ) (L : ℕ)
  (h_speed : speed_kmh = 216)
  (h_time : time_sec = 60)
  (h_distance : distance = 60 * time_sec)
  (h_total_distance : distance = 2 * L) :
  L = 1800 := by
  sorry

end train_length_eq_1800_l4_4395


namespace value_of_playstation_l4_4947

theorem value_of_playstation (V : ℝ) (H1 : 700 + 200 = 900) (H2 : V - 0.2 * V = 0.8 * V) (H3 : 0.8 * V = 900 - 580) : V = 400 :=
by
  sorry

end value_of_playstation_l4_4947


namespace min_a_squared_plus_b_squared_l4_4536

theorem min_a_squared_plus_b_squared (a b : ℝ) (h : a * b = 1) : a^2 + b^2 ≥ 2 :=
sorry

end min_a_squared_plus_b_squared_l4_4536


namespace point_D_in_fourth_quadrant_l4_4406

def is_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

def point_A : ℝ × ℝ := (1, 2)
def point_B : ℝ × ℝ := (-1, -2)
def point_C : ℝ × ℝ := (-1, 2)
def point_D : ℝ × ℝ := (1, -2)

theorem point_D_in_fourth_quadrant : is_in_fourth_quadrant (point_D.1) (point_D.2) :=
by
  sorry

end point_D_in_fourth_quadrant_l4_4406


namespace distance_home_to_school_l4_4862

theorem distance_home_to_school
  (T T' : ℝ)
  (D : ℝ)
  (h1 : D = 6 * T)
  (h2 : D = 12 * T')
  (h3 : T - T' = 0.25) :
  D = 3 :=
by
  -- The proof would go here
  sorry

end distance_home_to_school_l4_4862


namespace boys_without_calculators_l4_4620

theorem boys_without_calculators (total_boys total_students students_with_calculators girls_with_calculators : ℕ) 
    (h1 : total_boys = 20) 
    (h2 : total_students = 40) 
    (h3 : students_with_calculators = 30) 
    (h4 : girls_with_calculators = 18) : 
    (total_boys - (students_with_calculators - girls_with_calculators)) = 8 :=
by
  sorry

end boys_without_calculators_l4_4620


namespace triangle_area_l4_4079

theorem triangle_area (X Y Z : ℝ) (r R : ℝ)
  (h1 : r = 7)
  (h2 : R = 25)
  (h3 : 2 * Real.cos Y = Real.cos X + Real.cos Z) :
  ∃ (p q r : ℕ), (p * Real.sqrt q / r = 133) ∧ (p + q + r = 135) :=
  sorry

end triangle_area_l4_4079


namespace triangle_area_l4_4267

-- Define the vertices of the triangle
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (6, 1)
def C : ℝ × ℝ := (3, -4)

-- State that the area of the triangle is 12.5 square units
theorem triangle_area :
  let base := 6 - 1
  let height := 1 - -4
  (1 / 2) * base * height = 12.5 := by
  sorry

end triangle_area_l4_4267


namespace root_of_function_l4_4581

noncomputable def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)

theorem root_of_function (f : ℝ → ℝ) (x₀ : ℝ) (h₀ : odd_function f) (h₁ : f (x₀) = Real.exp (x₀)) :
  (f (-x₀) * Real.exp (-x₀) + 1 = 0) :=
by
  sorry

end root_of_function_l4_4581


namespace parabola_tangent_perp_l4_4094

theorem parabola_tangent_perp (a b : ℝ) : 
  (∃ x y : ℝ, x^2 = 4 * y ∧ y = a ∧ b ≠ 0 ∧ x ≠ 0) ∧
  (∃ x' y' : ℝ, x'^2 = 4 * y' ∧ y' = b ∧ a ≠ 0 ∧ x' ≠ 0) ∧
  (a * b = -1) 
  → a^4 * b^4 = (a^2 + b^2)^3 :=
by
  sorry

end parabola_tangent_perp_l4_4094


namespace christian_age_in_years_l4_4681

theorem christian_age_in_years (B C x : ℕ) (h1 : C = 2 * B) (h2 : B + x = 40) (h3 : C + x = 72) :
    x = 8 := 
sorry

end christian_age_in_years_l4_4681


namespace production_rate_equation_l4_4995

theorem production_rate_equation (x : ℝ) (h1 : ∀ t : ℝ, t = 600 / (x + 8)) (h2 : ∀ t : ℝ, t = 400 / x) : 
  600/(x + 8) = 400/x :=
by
  sorry

end production_rate_equation_l4_4995


namespace find_value_l4_4353

-- Given points A(a, 1), B(2, b), and C(3, 4).
variables (a b : ℝ)

-- Given condition from the problem
def condition : Prop := (3 * a + 4 = 6 + 4 * b)

-- The target is to find 3a - 4b
def target : ℝ := 3 * a - 4 * b

theorem find_value (h : condition a b) : target a b = 2 := 
by sorry

end find_value_l4_4353


namespace product_of_roots_of_cubic_polynomial_l4_4182

theorem product_of_roots_of_cubic_polynomial :
  let a := 1
  let b := -15
  let c := 75
  let d := -50
  ∀ x : ℝ, (x^3 - 15*x^2 + 75*x - 50 = 0) →
  (x + 1) * (x + 1) * (x + 50) - 1 * (x + 50) = 0 → -d / a = 50 :=
by
  sorry

end product_of_roots_of_cubic_polynomial_l4_4182


namespace cheburashkas_erased_l4_4767

theorem cheburashkas_erased (total_krakozyabras : ℕ) (rows : ℕ) :
  rows ≥ 2 → total_krakozyabras = 29 → ∃ (cheburashkas_erased : ℕ), cheburashkas_erased = 11 :=
by
  assume h_rows h_total_krakozyabras
  let n := (total_krakozyabras / 2) + 1
  have h_cheburashkas : cheburashkas_erased = n - 1 
  sorry

end cheburashkas_erased_l4_4767


namespace distance_CD_l4_4447

theorem distance_CD (d_north: ℝ) (d_east: ℝ) (d_south: ℝ) (d_west: ℝ) (distance_CD: ℝ) :
  d_north = 30 ∧ d_east = 80 ∧ d_south = 20 ∧ d_west = 30 → distance_CD = 50 :=
by
  intros h
  sorry

end distance_CD_l4_4447


namespace parallelogram_sides_l4_4293

theorem parallelogram_sides (x y : ℝ) (h₁ : 4 * x + 1 = 11) (h₂ : 10 * y - 3 = 5) : x + y = 3.3 :=
sorry

end parallelogram_sides_l4_4293


namespace calc_expression_l4_4879

theorem calc_expression : (900^2) / (264^2 - 256^2) = 194.711 := by
  sorry

end calc_expression_l4_4879


namespace greatest_power_sum_l4_4429

theorem greatest_power_sum (a b : ℕ) (h1 : 0 < a) (h2 : 2 < b) (h3 : a^b < 500) (h4 : ∀ m n : ℕ, 0 < m → 2 < n → m^n < 500 → a^b ≥ m^n) : a + b = 10 :=
by
  -- Sorry is used to skip the proof steps
  sorry

end greatest_power_sum_l4_4429


namespace max_distance_from_earth_to_sun_l4_4358

-- Assume the semi-major axis 'a' and semi-minor axis 'b' specified in the problem.
def semi_major_axis : ℝ := 1.5 * 10^8
def semi_minor_axis : ℝ := 3 * 10^6

-- Define the theorem stating the maximum distance from the Earth to the Sun.
theorem max_distance_from_earth_to_sun :
  let a := semi_major_axis
  let b := semi_minor_axis
  a + b = 1.53 * 10^8 :=
by
  -- Proof will be completed
  sorry

end max_distance_from_earth_to_sun_l4_4358


namespace Walter_age_in_2003_l4_4169

-- Defining the conditions
def Walter_age_1998 (walter_age_1998 grandmother_age_1998 : ℝ) : Prop :=
  walter_age_1998 = grandmother_age_1998 / 3

def birth_years_sum (walter_age_1998 grandmother_age_1998 : ℝ) : Prop :=
  (1998 - walter_age_1998) + (1998 - grandmother_age_1998) = 3858

-- Defining the theorem to be proved
theorem Walter_age_in_2003 (walter_age_1998 grandmother_age_1998 : ℝ) 
  (h1 : Walter_age_1998 walter_age_1998 grandmother_age_1998) 
  (h2 : birth_years_sum walter_age_1998 grandmother_age_1998) : 
  walter_age_1998 + 5 = 39.5 :=
  sorry

end Walter_age_in_2003_l4_4169


namespace div_add_fraction_l4_4404

theorem div_add_fraction : (3 / 7) / 4 + 2 = 59 / 28 :=
by
  sorry

end div_add_fraction_l4_4404


namespace aisha_probability_l4_4419

noncomputable def prob_one_head (prob_tail : ℝ) (num_coins : ℕ) : ℝ :=
  1 - (prob_tail ^ num_coins)

theorem aisha_probability : 
  prob_one_head (1/2) 4 = 15 / 16 := 
by 
  sorry

end aisha_probability_l4_4419


namespace find_a_l4_4577

theorem find_a (x y : ℝ) (a : ℝ) (h1 : x = 3) (h2 : y = 2) (h3 : a * x + 2 * y = 1) : a = -1 := by
  sorry

end find_a_l4_4577


namespace solve_system_l4_4659

theorem solve_system (x y z : ℝ) (h1 : (x + 1) * y * z = 12) 
                               (h2 : (y + 1) * z * x = 4) 
                               (h3 : (z + 1) * x * y = 4) : 
  (x = 1 / 3 ∧ y = 3 ∧ z = 3) ∨ (x = 2 ∧ y = -2 ∧ z = -2) :=
sorry

end solve_system_l4_4659


namespace parabola_circle_intersection_radius_squared_l4_4896

theorem parabola_circle_intersection_radius_squared :
  (∀ x y, y = (x - 2)^2 → x + 1 = (y + 2)^2 → (x - 1)^2 + (y + 1)^2 = 1) :=
sorry

end parabola_circle_intersection_radius_squared_l4_4896


namespace weight_of_B_l4_4236

theorem weight_of_B (A B C : ℝ)
(h1 : (A + B + C) / 3 = 45)
(h2 : (A + B) / 2 = 40)
(h3 : (B + C) / 2 = 41)
(h4 : 2 * A = 3 * B ∧ 5 * C = 3 * B)
(h5 : A + B + C = 144) :
B = 43.2 :=
sorry

end weight_of_B_l4_4236


namespace power_inequality_l4_4229

theorem power_inequality (a b c : ℕ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hcb : c ≥ b) : 
  a^b * (a + b)^c > c^b * a^c := 
sorry

end power_inequality_l4_4229


namespace cheburashkas_erased_l4_4769

theorem cheburashkas_erased (total_krakozyabras : ℕ) (rows : ℕ) :
  rows ≥ 2 → total_krakozyabras = 29 → ∃ (cheburashkas_erased : ℕ), cheburashkas_erased = 11 :=
by
  assume h_rows h_total_krakozyabras
  let n := (total_krakozyabras / 2) + 1
  have h_cheburashkas : cheburashkas_erased = n - 1 
  sorry

end cheburashkas_erased_l4_4769


namespace symmetry_axis_of_f_l4_4456

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem symmetry_axis_of_f :
  ∃ k : ℤ, ∃ k_π_div_2 : ℝ, (f (k * π / 2 + π / 12) = f ((k * π / 2 + π / 12) + π)) :=
by {
  sorry
}

end symmetry_axis_of_f_l4_4456


namespace number_of_dolls_combined_l4_4162

-- Defining the given conditions as variables
variables (aida sophie vera : ℕ)

-- Given conditions
def condition1 : Prop := aida = 2 * sophie
def condition2 : Prop := sophie = 2 * vera
def condition3 : Prop := vera = 20

-- The final proof statement we need to prove
theorem number_of_dolls_combined (h1 : condition1 aida sophie) (h2 : condition2 sophie vera) (h3 : condition3 vera) : 
  aida + sophie + vera = 140 :=
  by sorry

end number_of_dolls_combined_l4_4162


namespace largest_base7_three_digit_is_342_l4_4634

-- Definition of the base-7 number 666
def base7_666 : ℕ := 6 * 7^2 + 6 * 7^1 + 6 * 7^0

-- The largest decimal number represented by a three-digit base-7 number is 342
theorem largest_base7_three_digit_is_342 : base7_666 = 342 := by
  sorry

end largest_base7_three_digit_is_342_l4_4634


namespace solve_eq1_solve_eq2_solve_eq3_solve_eq4_l4_4970

theorem solve_eq1 (x : ℝ) : (3 * x + 2) ^ 2 = 25 ↔ (x = 1 ∨ x = -7 / 3) := by
  sorry

theorem solve_eq2 (x : ℝ) : 3 * x ^ 2 - 1 = 4 * x ↔ (x = (2 + Real.sqrt 7) / 3 ∨ x = (2 - Real.sqrt 7) / 3) := by
  sorry

theorem solve_eq3 (x : ℝ) : (2 * x - 1) ^ 2 = 3 * (2 * x + 1) ↔ (x = -1 / 2 ∨ x = 1) := by
  sorry

theorem solve_eq4 (x : ℝ) : x ^ 2 - 7 * x + 10 = 0 ↔ (x = 5 ∨ x = 2) := by
  sorry

end solve_eq1_solve_eq2_solve_eq3_solve_eq4_l4_4970


namespace find_d_l4_4506

theorem find_d (A B C D : ℕ) (h1 : (A + B + C) / 3 = 130) (h2 : (A + B + C + D) / 4 = 126) : D = 114 :=
by
  sorry

end find_d_l4_4506


namespace complex_magnitude_theorem_l4_4373

theorem complex_magnitude_theorem (z w : ℂ) (hz : |z| = 2) (hw : |w| = 4) (hzw : |z + w| = 3) :
  ∣(1 / z) + (1 / w)∣ = 3 / 8 := by
  sorry

end complex_magnitude_theorem_l4_4373


namespace tangent_line_to_circle_l4_4558

noncomputable def r_tangent_to_circle : ℝ := 4

theorem tangent_line_to_circle
  (x y r : ℝ)
  (circle_eq : x^2 + y^2 = 2 * r)
  (line_eq : x - y = r) :
  r = r_tangent_to_circle :=
by
  sorry

end tangent_line_to_circle_l4_4558


namespace quadratic_has_two_distinct_real_roots_l4_4570

theorem quadratic_has_two_distinct_real_roots (a : ℝ) (h : a ≠ 0): 
  (a < 4 / 3) ↔ (∃ x y : ℝ, x ≠ y ∧  a * x^2 - 4 * x + 3 = 0 ∧ a * y^2 - 4 * y + 3 = 0) := 
sorry

end quadratic_has_two_distinct_real_roots_l4_4570


namespace find_a_b_l4_4128

theorem find_a_b (a b : ℤ) (h : ({a, 0, -1} : Set ℤ) = {4, b, 0}) : a = 4 ∧ b = -1 := by
  sorry

end find_a_b_l4_4128


namespace radius_B_eq_8_div_9_l4_4557

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Given conditions
variable (A B C D : Circle)
variable (h1 : A.radius = 1)
variable (h2 : A.radius + A.radius = D.radius)
variable (h3 : B.radius = C.radius)
variable (h4 : (A.center.1 - B.center.1)^2 + (A.center.2 - B.center.2)^2 = (A.radius + B.radius)^2)
variable (h5 : (A.center.1 - C.center.1)^2 + (A.center.2 - C.center.2)^2 = (A.radius + C.radius)^2)
variable (h6 : (B.center.1 - C.center.1)^2 + (B.center.2 - C.center.2)^2 = (B.radius + C.radius)^2)
variable (h7 : (D.center.1 - A.center.1)^2 + (D.center.2 - A.center.2)^2 = D.radius^2)

-- Prove the radius of circle B is 8/9
theorem radius_B_eq_8_div_9 : B.radius = 8 / 9 := 
by
  sorry

end radius_B_eq_8_div_9_l4_4557


namespace find_number_l4_4282

theorem find_number (x : ℝ) (h : 3034 - (x / 20.04) = 2984) : x = 1002 :=
by
  sorry

end find_number_l4_4282


namespace closest_multiple_of_15_to_2023_is_2025_l4_4272

theorem closest_multiple_of_15_to_2023_is_2025 (n : ℤ) (h : 15 * n = 2025) : 
  ∀ m : ℤ, abs (2023 - 2025) ≤ abs (2023 - 15 * m) :=
by
  exact sorry

end closest_multiple_of_15_to_2023_is_2025_l4_4272


namespace girls_divisible_by_nine_l4_4431

def total_students (m c d u : ℕ) : ℕ := 1000 * m + 100 * c + 10 * d + u
def number_of_boys (m c d u : ℕ) : ℕ := m + c + d + u
def number_of_girls (m c d u : ℕ) : ℕ := total_students m c d u - number_of_boys m c d u 

theorem girls_divisible_by_nine (m c d u : ℕ) : 
  number_of_girls m c d u % 9 = 0 := 
by
    sorry

end girls_divisible_by_nine_l4_4431


namespace find_k_l4_4591

-- Definitions of the vectors and condition about perpendicularity
def vector_a : ℝ × ℝ := (2, 1)
def vector_b (k : ℝ) : ℝ × ℝ := (-2, k)
def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

-- The theorem that states if vector_a is perpendicular to (2 * vector_a - vector_b), then k = 14
theorem find_k (k : ℝ) (h : perpendicular vector_a (2 • vector_a - vector_b k)) : k = 14 := sorry

end find_k_l4_4591


namespace sin_330_eq_neg_half_l4_4016

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  -- Proof would go here
  sorry

end sin_330_eq_neg_half_l4_4016


namespace megan_seashells_l4_4086

theorem megan_seashells (current_seashells desired_seashells diff_seashells : ℕ)
  (h1 : current_seashells = 307)
  (h2 : desired_seashells = 500)
  (h3 : diff_seashells = desired_seashells - current_seashells) :
  diff_seashells = 193 :=
by
  sorry

end megan_seashells_l4_4086


namespace canoe_rental_cost_l4_4989

theorem canoe_rental_cost :
  ∃ (C : ℕ) (K : ℕ), 
  (15 * K + C * (K + 4) = 288) ∧ 
  (3 * K + 12 = 12 * C) ∧ 
  (C = 14) :=
sorry

end canoe_rental_cost_l4_4989


namespace speed_of_goods_train_is_72_kmph_l4_4541

-- Definitions for conditions
def length_of_train : ℝ := 240.0416
def length_of_platform : ℝ := 280
def time_to_cross : ℝ := 26

-- Distance covered by the train while crossing the platform
def total_distance : ℝ := length_of_train + length_of_platform

-- Speed calculation in meters per second
def speed_mps : ℝ := total_distance / time_to_cross

-- Speed conversion from meters per second to kilometers per hour
def speed_kmph : ℝ := speed_mps * 3.6

-- Proof statement
theorem speed_of_goods_train_is_72_kmph : speed_kmph = 72 := 
by
  sorry

end speed_of_goods_train_is_72_kmph_l4_4541


namespace polygon_edges_of_set_S_l4_4955

variable (a : ℝ)

def in_set_S(x y : ℝ) : Prop :=
  (a / 2 ≤ x ∧ x ≤ 2 * a) ∧
  (a / 2 ≤ y ∧ y ≤ 2 * a) ∧
  (x + y ≥ a) ∧
  (x + a ≥ y) ∧
  (y + a ≥ x)

theorem polygon_edges_of_set_S (a : ℝ) (h : 0 < a) :
  (∃ n, ∀ x y, in_set_S a x y → n = 6) :=
sorry

end polygon_edges_of_set_S_l4_4955


namespace find_breadth_of_wall_l4_4533

theorem find_breadth_of_wall
  (b h l V : ℝ)
  (h1 : V = 12.8)
  (h2 : h = 5 * b)
  (h3 : l = 8 * h) :
  b = 0.4 :=
by
  sorry

end find_breadth_of_wall_l4_4533


namespace sqrt_fraction_equiv_l4_4880

-- Define the fractions
def frac1 : ℚ := 25 / 36
def frac2 : ℚ := 16 / 9

-- Define the expression under the square root
def sum_frac : ℚ := frac1 + (frac2 * 36 / 36)

-- State the problem
theorem sqrt_fraction_equiv : (Real.sqrt sum_frac) = Real.sqrt 89 / 6 :=
by
  -- Steps and proof are omitted; we use sorry to indicate the proof is skipped
  sorry

end sqrt_fraction_equiv_l4_4880


namespace line_through_point_perpendicular_y_axis_line_through_two_points_l4_4714

-- The first problem
theorem line_through_point_perpendicular_y_axis :
  ∃ (k : ℝ), ∀ (x : ℝ), k = 1 → y = k :=
sorry

-- The second problem
theorem line_through_two_points (x1 y1 x2 y2 : ℝ) (hA : (x1, y1) = (-4, 0)) (hB : (x2, y2) = (0, 6)) :
  ∃ (a b c : ℝ), (a, b, c) = (3, -2, 12) → ∀ (x y : ℝ), a * x + b * y + c = 0 :=
sorry

end line_through_point_perpendicular_y_axis_line_through_two_points_l4_4714


namespace problem_l4_4926

theorem problem (a b : ℝ) (h : ∀ x : ℝ, (x + a) * (x + b) = x^2 + 4 * x + 3) : a + b = 4 :=
by
  sorry

end problem_l4_4926


namespace sin_330_eq_neg_sqrt3_div_2_l4_4000

theorem sin_330_eq_neg_sqrt3_div_2 
  (R : ℝ × ℝ)
  (hR : R = (1/2, -sqrt(3)/2))
  : Real.sin (330 * Real.pi / 180) = -sqrt(3)/2 :=
by
  sorry

end sin_330_eq_neg_sqrt3_div_2_l4_4000


namespace zoe_calories_l4_4133

theorem zoe_calories 
  (s : ℕ) (y : ℕ) (c_s : ℕ) (c_y : ℕ)
  (s_eq : s = 12) (y_eq : y = 6) (cs_eq : c_s = 4) (cy_eq : c_y = 17) :
  s * c_s + y * c_y = 150 :=
by
  sorry

end zoe_calories_l4_4133


namespace remaining_sweets_in_packet_l4_4375

theorem remaining_sweets_in_packet 
  (C : ℕ) (S : ℕ) (P : ℕ) (R : ℕ) (L : ℕ)
  (HC : C = 30) (HS : S = 100) (HP : P = 60) (HR : R = 25) (HL : L = 150) 
  : (C - (2 * C / 5) - ((C - P / 4) / 3)) 
  + (S - (S / 4)) 
  + (P - (3 * P / 5)) 
  + ((max 0 (R - (3 * R / 2))))
  + (L - (3 * (S / 4) / 2)) = 232 :=
by
  sorry

end remaining_sweets_in_packet_l4_4375


namespace nova_monthly_donation_l4_4087

def total_annual_donation : ℕ := 20484
def months_in_year : ℕ := 12
def monthly_donation : ℕ := total_annual_donation / months_in_year

theorem nova_monthly_donation :
  monthly_donation = 1707 :=
by
  unfold monthly_donation
  sorry

end nova_monthly_donation_l4_4087


namespace quadratic_equation_is_D_l4_4858

theorem quadratic_equation_is_D (x a b c : ℝ) : 
  (¬ (∃ b' : ℝ, (x^2 - 2) * x = b' * x + 2)) ∧
  (¬ ((a ≠ 0) ∧ (ax^2 + bx + c = 0))) ∧
  (¬ (x + (1 / x) = 5)) ∧
  ((x^2 = 0) ↔ true) :=
by sorry

end quadratic_equation_is_D_l4_4858


namespace find_base_of_log_equation_l4_4567

theorem find_base_of_log_equation :
  ∃ b : ℝ, (∀ x : ℝ, (9 : ℝ)^(x + 5) = (5 : ℝ)^x → x = Real.logb b ((9 : ℝ)^5)) ∧ b = 5 / 9 :=
by
  sorry

end find_base_of_log_equation_l4_4567


namespace smallest_value_z_minus_x_l4_4644

theorem smallest_value_z_minus_x 
  (x y z : ℕ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (hmul : x * y * z = 5040) 
  (hxy : x < y) 
  (hyz : y < z) : 
  z - x = 9 := 
  sorry

end smallest_value_z_minus_x_l4_4644


namespace adam_coin_collection_value_l4_4160

-- Definitions related to the problem conditions
def value_per_first_type_coin := 15 / 5
def value_per_second_type_coin := 18 / 6

def total_value_first_type (num_first_type_coins : ℕ) := num_first_type_coins * value_per_first_type_coin
def total_value_second_type (num_second_type_coins : ℕ) := num_second_type_coins * value_per_second_type_coin

-- The main theorem, stating that the total collection value is 90 dollars given the conditions
theorem adam_coin_collection_value :
  total_value_first_type 18 + total_value_second_type 12 = 90 := 
sorry

end adam_coin_collection_value_l4_4160


namespace max_value_of_exp_minus_x_l4_4822

theorem max_value_of_exp_minus_x :
  ∃ (c : ℝ), ∀ x ∈ set.Icc (0 : ℝ) 1, 
  (∀ y ∈ set.Icc (0 : ℝ) 1, f y ≤ f x) → f x = c ∧ c = real.exp 1 - 1 :=
by
  let f := λ x : ℝ, real.exp x - x
  have deriv_f : ∀ x, deriv f x = real.exp x - 1 := sorry
  have mono_f : ∀ x ∈ set.Icc (0 : ℝ) 1, ∀ y ∈ set.Icc (0 : ℝ) 1, x ≤ y → f x ≤ f y := sorry
  have max_f : f 1 = real.exp 1 - 1 := sorry
  exact ⟨f 1, λ x hx hx_max, by
    rw [← hx_max, max_f]
    exact max_f⟩


end max_value_of_exp_minus_x_l4_4822


namespace product_of_roots_eq_50_l4_4179

theorem product_of_roots_eq_50 :
  let a := -15
  let c := -50
  let equation := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  -- Vieta's formulas for the product of roots in cubic equation ax^3 + bx^2 + cx + d = 0 is -d/a
  ∃ p q r : ℝ, 
    equation p = 0 ∧ equation q = 0 ∧ equation r = 0 ∧ (p * q * r = 50) :=
by
  sorry

end product_of_roots_eq_50_l4_4179


namespace total_cost_correct_l4_4417

noncomputable def total_cost : ℝ :=
  let first_path_area := 5 * 100
  let first_path_cost := first_path_area * 2
  let second_path_area := 4 * 80
  let second_path_cost := second_path_area * 1.5
  let diagonal_length := Real.sqrt ((100:ℝ)^2 + (80:ℝ)^2)
  let third_path_area := 6 * diagonal_length
  let third_path_cost := third_path_area * 3
  let circular_path_area := Real.pi * (10:ℝ)^2
  let circular_path_cost := circular_path_area * 4
  first_path_cost + second_path_cost + third_path_cost + circular_path_cost

theorem total_cost_correct : total_cost = 5040.64 := by
  sorry

end total_cost_correct_l4_4417


namespace product_of_two_numbers_l4_4842

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x^2 + y^2 = 404) : x * y = 86 :=
sorry

end product_of_two_numbers_l4_4842


namespace product_of_roots_of_cubic_l4_4186

theorem product_of_roots_of_cubic :
  (∃ (p q r : ℝ), (p, q, r ≠ 0) ∧ (∀ x : ℝ, x^3 - 15*x^2 + 75*x - 50 = 0 ↔ x = p ∨ x = q ∨ x = r) ∧ p * q * r = 50) :=
sorry

end product_of_roots_of_cubic_l4_4186


namespace horner_method_value_at_neg1_l4_4680

theorem horner_method_value_at_neg1 : 
  let f (x : ℤ) := 4 * x ^ 4 + 3 * x ^ 3 - 6 * x ^ 2 + x - 1
  let x := -1
  let v0 := 4
  let v1 := v0 * x + 3
  let v2 := v1 * x - 6
  v2 = -5 := by
  sorry

end horner_method_value_at_neg1_l4_4680


namespace expand_polynomial_identity_l4_4443

variable {x : ℝ}

theorem expand_polynomial_identity : (7 * x + 5) * (5 * x ^ 2 - 2 * x + 4) = 35 * x ^ 3 + 11 * x ^ 2 + 18 * x + 20 := by
    sorry

end expand_polynomial_identity_l4_4443


namespace degree_of_g_l4_4231

open Polynomial

def f : Polynomial ℝ := -7 * X^4 + 3 * X^3 + X - 5

theorem degree_of_g (g : Polynomial ℝ)
  (hdeg : (f + g).degree = 2) :
  g.degree = 4 :=
sorry

end degree_of_g_l4_4231


namespace vector_triangle_c_solution_l4_4064

theorem vector_triangle_c_solution :
  let a : ℝ × ℝ := (1, -3)
  let b : ℝ × ℝ := (-2, 4)
  let c : ℝ × ℝ := (4, -6)
  (4 • a + (3 • b - 2 • a) + c = (0, 0)) →
  c = (4, -6) :=
by
  intro h
  sorry

end vector_triangle_c_solution_l4_4064


namespace wire_around_field_l4_4199

theorem wire_around_field (A L : ℕ) (hA : A = 69696) (hL : L = 15840) : L / (4 * (Nat.sqrt A)) = 15 :=
by
  sorry

end wire_around_field_l4_4199


namespace total_points_scored_l4_4238

theorem total_points_scored (m1 m2 m3 m4 m5 m6 j1 j2 j3 j4 j5 j6 : ℕ) :
  m1 = 5 → j1 = m1 + 2 →
  m2 = 7 → j2 = m2 - 3 →
  m3 = 10 → j3 = m3 / 2 →
  m4 = 12 → j4 = m4 * 2 →
  m5 = 6 → j5 = m5 →
  j6 = 8 → m6 = j6 + 4 →
  m1 + m2 + m3 + m4 + m5 + m6 + j1 + j2 + j3 + j4 + j5 + j6 = 106 :=
by
  intros
  sorry

end total_points_scored_l4_4238


namespace decimal_sum_difference_l4_4554

theorem decimal_sum_difference :
  (0.5 - 0.03 + 0.007 + 0.0008 = 0.4778) :=
by
  sorry

end decimal_sum_difference_l4_4554


namespace gold_coins_percentage_is_35_l4_4166

-- Define the conditions: percentage of beads and percentage of silver coins
def percent_beads : ℝ := 0.30
def percent_silver_coins : ℝ := 0.50

-- Definition of the percentage of all objects that are gold coins
def percent_gold_coins (percent_beads percent_silver_coins : ℝ) : ℝ :=
  (1 - percent_beads) * (1 - percent_silver_coins)

-- The statement that we need to prove:
theorem gold_coins_percentage_is_35 :
  percent_gold_coins percent_beads percent_silver_coins = 0.35 :=
  by
    unfold percent_gold_coins percent_beads percent_silver_coins
    sorry

end gold_coins_percentage_is_35_l4_4166


namespace xy_value_l4_4264

theorem xy_value {x y : ℝ} (h1 : x - y = 6) (h2 : x^3 - y^3 = 162) : x * y = 21 :=
by
  sorry

end xy_value_l4_4264


namespace area_F1PF2Q_l4_4604

noncomputable def hyperbola_directrix (x : ℝ) := (x = 3/2)
noncomputable def hyperbola_asymptote (x y : ℝ) := (y = (sqrt 3 / 3) * x) ∨ (y = -(sqrt 3 / 3) * x)

def point_P := (3/2, sqrt 3 / 2)
def point_Q := (3/2, -sqrt 3 / 2)
def foci_F1 := (-2, 0)
def foci_F2 := (2, 0)

theorem area_F1PF2Q : 
  let P := point_P
  let Q := point_Q
  let F1 := foci_F1
  let F2 := foci_F2
  hyperbola_directrix P.1 → (hyperbola_asymptote P.1 P.2 ∧ hyperbola_asymptote Q.1 Q.2) →
  (∃ area : ℝ, area = 2*sqrt 3 ∧
  quadrilateral_area P Q F1 F2 = area) :=
by
  sorry

end area_F1PF2Q_l4_4604


namespace num_integers_with_gcd_3_l4_4045

theorem num_integers_with_gcd_3 (n : ℕ) : {n | 1 ≤ n ∧ n ≤ 150 ∧ Nat.gcd 21 n = 3}.card = 43 :=
sorry

end num_integers_with_gcd_3_l4_4045


namespace solve_a_l4_4890

def custom_op (a b : ℝ) : ℝ := 2 * a - b^2

theorem solve_a :
  ∃ a : ℝ, custom_op a 7 = -20 ∧ a = 29 / 2 :=
by
  sorry

end solve_a_l4_4890


namespace factorize_expression_l4_4697

theorem factorize_expression (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 :=
by sorry

end factorize_expression_l4_4697


namespace product_of_roots_cubic_l4_4176

theorem product_of_roots_cubic :
  let p := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  let roots := {x : ℝ // p x = 0}.toFinset
  ∃ r : ℝ, (∀ x ∈ roots, p x = 0) ∧ (∏ x in roots, x) = 50 :=
by
  sorry

end product_of_roots_cubic_l4_4176


namespace unique_zero_of_quadratic_l4_4077

theorem unique_zero_of_quadratic {m : ℝ} (h : ∃ x : ℝ, x^2 + 2*x + m = 0 ∧ (∀ y : ℝ, y^2 + 2*y + m = 0 → y = x)) : m = 1 :=
sorry

end unique_zero_of_quadratic_l4_4077


namespace cheburashkas_erased_l4_4768

theorem cheburashkas_erased (total_krakozyabras : ℕ) (rows : ℕ) :
  rows ≥ 2 → total_krakozyabras = 29 → ∃ (cheburashkas_erased : ℕ), cheburashkas_erased = 11 :=
by
  assume h_rows h_total_krakozyabras
  let n := (total_krakozyabras / 2) + 1
  have h_cheburashkas : cheburashkas_erased = n - 1 
  sorry

end cheburashkas_erased_l4_4768


namespace no_k_satisfying_condition_l4_4430

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_k_satisfying_condition :
  ∀ k : ℕ, (∃ p q : ℕ, p ≠ q ∧ is_prime p ∧ is_prime q ∧ k = p * q ∧ p + q = 71) → false :=
by
  sorry

end no_k_satisfying_condition_l4_4430


namespace carrie_mom_money_l4_4555

theorem carrie_mom_money :
  ∀ (sweater_cost t_shirt_cost shoes_cost left_money total_money : ℕ),
  sweater_cost = 24 →
  t_shirt_cost = 6 →
  shoes_cost = 11 →
  left_money = 50 →
  total_money = sweater_cost + t_shirt_cost + shoes_cost + left_money →
  total_money = 91 :=
sorry

end carrie_mom_money_l4_4555


namespace cos_sum_nonneg_one_l4_4370

theorem cos_sum_nonneg_one (x y z : ℝ) (h : x + y + z = 0) : abs (Real.cos x) + abs (Real.cos y) + abs (Real.cos z) ≥ 1 := 
by {
  sorry
}

end cos_sum_nonneg_one_l4_4370


namespace beth_should_charge_42_cents_each_l4_4719

theorem beth_should_charge_42_cents_each (n_alan_cookies : ℕ) (price_alan_cookie : ℕ) (n_beth_cookies : ℕ) (total_earnings : ℕ) (price_beth_cookie : ℕ):
  n_alan_cookies = 15 → 
  price_alan_cookie = 50 → 
  n_beth_cookies = 18 → 
  total_earnings = n_alan_cookies * price_alan_cookie → 
  price_beth_cookie = total_earnings / n_beth_cookies → 
  price_beth_cookie = 42 := 
by 
  intros h1 h2 h3 h4 h5 
  sorry

end beth_should_charge_42_cents_each_l4_4719


namespace find_m_pure_imaginary_l4_4232

theorem find_m_pure_imaginary (m : ℝ) (h : m^2 + m - 2 + (m^2 - 1) * I = (0 : ℝ) + (m^2 - 1) * I) :
  m = -2 :=
by {
  sorry
}

end find_m_pure_imaginary_l4_4232


namespace volleyball_match_prob_A_win_l4_4346

-- Definitions of given probabilities and conditions
def rally_scoring_system := true
def first_to_25_wins := true
def tie_at_24_24_continues_until_lead_by_2 := true
def prob_team_A_serves_win : ℚ := 2/3
def prob_team_B_serves_win : ℚ := 2/5
def outcomes_independent := true
def score_22_22_team_A_serves := true

-- The problem to prove
theorem volleyball_match_prob_A_win :
  rally_scoring_system ∧
  first_to_25_wins ∧
  tie_at_24_24_continues_until_lead_by_2 ∧
  prob_team_A_serves_win = 2/3 ∧
  prob_team_B_serves_win = 2/5 ∧
  outcomes_independent ∧
  score_22_22_team_A_serves →
  (prob_team_A_serves_win ^ 3 + (1 - prob_team_A_serves_win) * prob_team_B_serves_win * prob_team_A_serves_win ^ 2 + prob_team_A_serves_win * (1 - prob_team_A_serves_win) * prob_team_B_serves_win * prob_team_A_serves_win + prob_team_A_serves_win ^ 2 * (1 - prob_team_A_serves_win) * prob_team_B_serves_win) = 64/135 :=
by
  sorry

end volleyball_match_prob_A_win_l4_4346


namespace pairs_satisfying_condition_l4_4369

theorem pairs_satisfying_condition (x y : ℤ) (h : x + y ≠ 0) :
  (x^2 + y^2)/(x + y) = 10 ↔ (x, y) = (12, 6) ∨ (x, y) = (-2, 6) ∨ (x, y) = (12, 4) ∨ (x, y) = (-2, 4) ∨ (x, y) = (10, 10) ∨ (x, y) = (0, 10) ∨ (x, y) = (10, 0) :=
sorry

end pairs_satisfying_condition_l4_4369


namespace smallest_number_is_61_point_4_l4_4649

theorem smallest_number_is_61_point_4 (x y z t : ℝ)
  (h1 : y = 2 * x)
  (h2 : z = 4 * y)
  (h3 : t = (y + z) / 3)
  (h4 : (x + y + z + t) / 4 = 220) :
  x = 2640 / 43 :=
by sorry

end smallest_number_is_61_point_4_l4_4649


namespace green_balls_count_l4_4639

theorem green_balls_count (b g : ℕ) (h1 : b = 15) (h2 : 5 * g = 3 * b) : g = 9 :=
by
  sorry

end green_balls_count_l4_4639


namespace sin_330_eq_neg_sqrt3_div_2_l4_4003

theorem sin_330_eq_neg_sqrt3_div_2 
  (R : ℝ × ℝ)
  (hR : R = (1/2, -sqrt(3)/2))
  : Real.sin (330 * Real.pi / 180) = -sqrt(3)/2 :=
by
  sorry

end sin_330_eq_neg_sqrt3_div_2_l4_4003


namespace trapezoid_area_is_8_l4_4227

noncomputable def trapezoid_area
  (AB CD : ℝ)        -- lengths of the bases
  (h : ℝ)            -- height (distance between the bases)
  : ℝ :=
  0.5 * (AB + CD) * h

theorem trapezoid_area_is_8 
  (AB CD : ℝ) 
  (h : ℝ) 
  (K M : ℝ) 
  (height_condition : h = 2)
  (AB_condition : AB = 5)
  (CD_condition : CD = 3)
  (K_midpoint : K = AB / 2) 
  (M_midpoint : M = CD / 2)
  : trapezoid_area AB CD h = 8 :=
by
  rw [trapezoid_area, AB_condition, CD_condition, height_condition]
  norm_num

end trapezoid_area_is_8_l4_4227


namespace product_of_roots_of_cubic_l4_4185

theorem product_of_roots_of_cubic :
  (∃ (p q r : ℝ), (p, q, r ≠ 0) ∧ (∀ x : ℝ, x^3 - 15*x^2 + 75*x - 50 = 0 ↔ x = p ∨ x = q ∨ x = r) ∧ p * q * r = 50) :=
sorry

end product_of_roots_of_cubic_l4_4185


namespace find_number_l4_4462

def hash (a b : ℕ) : ℕ := a * b - b + b^2

theorem find_number :
  (∃ x : ℕ, hash 3 x = 63 ∧ x = 7) :=
sorry

end find_number_l4_4462


namespace find_number_type_l4_4979

-- Definitions of the problem conditions
def consecutive (a b c d : ℤ) : Prop := (b = a + 2) ∧ (c = a + 4) ∧ (d = a + 6)
def sum_is_52 (a b c d : ℤ) : Prop := a + b + c + d = 52
def third_number_is_14 (c : ℤ) : Prop := c = 14

-- The proof problem statement
theorem find_number_type (a b c d : ℤ) 
                         (h1 : consecutive a b c d) 
                         (h2 : sum_is_52 a b c d) 
                         (h3 : third_number_is_14 c) :
  (∃ (k : ℤ), a = 2 * k ∧ b = 2 * k + 2 ∧ c = 2 * k + 4 ∧ d = 2 * k + 6) 
  := sorry

end find_number_type_l4_4979


namespace quadratic_form_and_sum_l4_4836

theorem quadratic_form_and_sum (x : ℝ) : 
  ∃ (a b c : ℝ), 
  (15 * x^2 + 75 * x + 375 = a * (x + b)^2 + c) ∧ 
  (a + b + c = 298.75) := 
sorry

end quadratic_form_and_sum_l4_4836


namespace rancher_loss_l4_4667

theorem rancher_loss
  (initial_cattle : ℕ)
  (total_price : ℕ)
  (sick_cattle : ℕ)
  (price_reduction : ℕ)
  (remaining_cattle := initial_cattle - sick_cattle)
  (original_price_per_head := total_price / initial_cattle)
  (new_price_per_head := original_price_per_head - price_reduction)
  (total_original_price := original_price_per_head * remaining_cattle)
  (total_new_price := new_price_per_head * remaining_cattle) :
  total_original_price - total_new_price = 25200 :=
by 
  sorry

-- Definitions
def initial_cattle : ℕ := 340
def total_price : ℕ := 204000
def sick_cattle : ℕ := 172
def price_reduction : ℕ := 150

-- Substitute the known values in the theorem
#eval rancher_loss initial_cattle total_price sick_cattle price_reduction

end rancher_loss_l4_4667


namespace find_original_selling_price_l4_4484

variable (SP : ℝ)
variable (CP : ℝ := 10000)
variable (discounted_SP : ℝ := 0.9 * SP)
variable (profit : ℝ := 0.08 * CP)

theorem find_original_selling_price :
  discounted_SP = CP + profit → SP = 12000 := by
sorry

end find_original_selling_price_l4_4484


namespace cars_on_happy_street_l4_4119

theorem cars_on_happy_street :
  let cars_tuesday := 25
  let cars_monday := cars_tuesday - cars_tuesday * 20 / 100
  let cars_wednesday := cars_monday + 2
  let cars_thursday : ℕ := 10
  let cars_friday : ℕ := 10
  let cars_saturday : ℕ := 5
  let cars_sunday : ℕ := 5
  let total_cars := cars_monday + cars_tuesday + cars_wednesday + cars_thursday + cars_friday + cars_saturday + cars_sunday
  total_cars = 97 :=
by
  sorry

end cars_on_happy_street_l4_4119


namespace tangent_line_correct_l4_4820

-- Define the curve y = x^3 - 1
def curve (x : ℝ) : ℝ := x^3 - 1

-- Define the derivative of the curve
def derivative_curve (x : ℝ) : ℝ := 3 * x^2

-- Define the point of tangency
def tangent_point : ℝ × ℝ := (1, curve 1)

-- Define the tangent line equation at x = 1
def tangent_line (x : ℝ) : ℝ := 3 * x - 3

-- The formal statement to be proven
theorem tangent_line_correct :
  ∀ x : ℝ, curve x = x^3 - 1 ∧ derivative_curve x = 3 * x^2 ∧ tangent_point = (1, 0) → 
    tangent_line 1 = 3 * 1 - 3 :=
by
  sorry

end tangent_line_correct_l4_4820


namespace increasing_condition_min_value_a_eq_one_l4_4338

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (-1/3) * x^3 + (1/2) * x^2 + 2 * x

theorem increasing_condition (a : ℝ) : 
  (∀ x > 2/3, ((x - 1/2)^2 + 1/4 + 2 * a > 0)) → a > -1/8 :=
by
  intro h
  sorry 

theorem min_value_a_eq_one : 
  ∀ x, 1 ≤ x ∧ x ≤ 4 → (-1/3)*4^3 + (1/2)*4^2 + 2 * 4 = (-16/3) :=
by
  sorry

end increasing_condition_min_value_a_eq_one_l4_4338


namespace find_number_l4_4285

theorem find_number (x : ℝ) (h : 3034 - x / 20.04 = 2984) : x = 1002 :=
sorry

end find_number_l4_4285


namespace percentage_decrease_l4_4111

theorem percentage_decrease (original_price new_price : ℝ) (h₁ : original_price = 700) (h₂ : new_price = 532) : 
  ((original_price - new_price) / original_price) * 100 = 24 := by
  sorry

end percentage_decrease_l4_4111


namespace deepak_current_age_l4_4997

theorem deepak_current_age (A D : ℕ) (h1 : A / D = 5 / 7) (h2 : A + 6 = 36) : D = 42 :=
sorry

end deepak_current_age_l4_4997


namespace tangent_line_ln_x_and_ln_x_plus_1_l4_4468

theorem tangent_line_ln_x_and_ln_x_plus_1 (k b : ℝ) : 
  (∃ x₁ x₂ : ℝ, (y = k * x₁ + b ∧ y = ln x₁ + 2) ∧ 
                (y = k * x₂ + b ∧ y = ln (x₂ + 1)) ∧ 
                (k = 2 ∧ x₁ = 1 / 2 ∧ x₂ = -1 / 2)) → 
  b = 1 - ln 2 :=
by
  sorry

end tangent_line_ln_x_and_ln_x_plus_1_l4_4468


namespace shaded_area_equals_l4_4562

noncomputable def area_shaded_figure (R : ℝ) : ℝ :=
  let α := (60 : ℝ) * (Real.pi / 180)
  (2 * Real.pi * R^2) / 3

theorem shaded_area_equals : ∀ R : ℝ, area_shaded_figure R = (2 * Real.pi * R^2) / 3 := sorry

end shaded_area_equals_l4_4562


namespace factorize_expression_l4_4692

theorem factorize_expression (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 :=
by sorry

end factorize_expression_l4_4692


namespace slices_per_person_l4_4547

theorem slices_per_person
  (small_pizza_slices : ℕ)
  (large_pizza_slices : ℕ)
  (small_pizzas_purchased : ℕ)
  (large_pizzas_purchased : ℕ)
  (george_slices : ℕ)
  (bob_extra : ℕ)
  (susie_divisor : ℕ)
  (bill_slices : ℕ)
  (fred_slices : ℕ)
  (mark_slices : ℕ)
  (ann_slices : ℕ)
  (kelly_multiplier : ℕ) :
  small_pizza_slices = 4 →
  large_pizza_slices = 8 →
  small_pizzas_purchased = 4 →
  large_pizzas_purchased = 3 →
  george_slices = 3 →
  bob_extra = 1 →
  susie_divisor = 2 →
  bill_slices = 3 →
  fred_slices = 3 →
  mark_slices = 3 →
  ann_slices = 2 →
  kelly_multiplier = 2 →
  (2 * (small_pizzas_purchased * small_pizza_slices + large_pizzas_purchased * large_pizza_slices -
    (george_slices + (george_slices + bob_extra) + (george_slices + bob_extra) / susie_divisor +
     bill_slices + fred_slices + mark_slices + ann_slices + ann_slices * kelly_multiplier))) =
    (small_pizzas_purchased * small_pizza_slices + large_pizzas_purchased * large_pizza_slices -
    (george_slices + (george_slices + bob_extra) + (george_slices + bob_extra) / susie_divisor +
     bill_slices + fred_slices + mark_slices + ann_slices + ann_slices * kelly_multiplier)) :=
by
  sorry

end slices_per_person_l4_4547


namespace cassidy_grounded_days_l4_4435

-- Definitions for the conditions
def days_for_lying : Nat := 14
def extra_days_per_grade : Nat := 3
def grades_below_B : Nat := 4

-- Definition for the total days grounded
def total_days_grounded : Nat :=
  days_for_lying + extra_days_per_grade * grades_below_B

-- The theorem statement
theorem cassidy_grounded_days :
  total_days_grounded = 26 := by
  sorry

end cassidy_grounded_days_l4_4435


namespace value_of_m_l4_4597

theorem value_of_m (x m : ℝ) (h : 2 * x + m - 6 = 0) (hx : x = 1) : m = 4 :=
by
  sorry

end value_of_m_l4_4597


namespace total_chairs_calculation_l4_4364

theorem total_chairs_calculation
  (chairs_per_trip : ℕ)
  (trips_per_student : ℕ)
  (total_students : ℕ)
  (h1 : chairs_per_trip = 5)
  (h2 : trips_per_student = 10)
  (h3 : total_students = 5) :
  total_students * (chairs_per_trip * trips_per_student) = 250 :=
by
  sorry

end total_chairs_calculation_l4_4364


namespace total_dolls_combined_l4_4164

-- Define the number of dolls for Vera
def vera_dolls : ℕ := 20

-- Define the relationship that Sophie has twice as many dolls as Vera
def sophie_dolls : ℕ := 2 * vera_dolls

-- Define the relationship that Aida has twice as many dolls as Sophie
def aida_dolls : ℕ := 2 * sophie_dolls

-- The statement to prove that the total number of dolls is 140
theorem total_dolls_combined : aida_dolls + sophie_dolls + vera_dolls = 140 :=
by
  sorry

end total_dolls_combined_l4_4164


namespace Nick_sister_age_l4_4795

theorem Nick_sister_age
  (Nick_age : ℕ := 13)
  (Bro_in_5_years : ℕ := 21)
  (H : ∃ S : ℕ, (Nick_age + S) / 2 + 5 = Bro_in_5_years) :
  ∃ S : ℕ, S = 19 :=
by
  sorry

end Nick_sister_age_l4_4795


namespace sum_consecutive_integers_l4_4385

theorem sum_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 :=
by
  sorry

end sum_consecutive_integers_l4_4385


namespace no_intersection_of_absolute_value_graphs_l4_4068

theorem no_intersection_of_absolute_value_graphs :
  ∀ (y : ℝ) (x : ℝ), y = abs (3 * x + 6) → y = -abs (4 * x - 3) → false :=
by {
  intros y x h1 h2,
  rw abs_nonneg (3 * x + 6) at h1,
  rw abs_nonneg (4 * x - 3) at h2,
  linarith,
}

end no_intersection_of_absolute_value_graphs_l4_4068


namespace angle_C_in_triangle_l4_4756

theorem angle_C_in_triangle (A B C : ℝ) 
  (hA : A = 90) (hB : B = 50) : (A + B + C = 180) → C = 40 :=
by
  intro hSum
  rw [hA, hB] at hSum
  linarith

end angle_C_in_triangle_l4_4756


namespace trees_planted_in_garden_l4_4746

theorem trees_planted_in_garden (yard_length : ℕ) (tree_distance : ℕ) (h₁ : yard_length = 500) (h₂ : tree_distance = 20) :
  ((yard_length / tree_distance) + 1) = 26 :=
by
  -- The proof goes here
  sorry

end trees_planted_in_garden_l4_4746


namespace calories_consummed_l4_4131

-- Definitions based on conditions
def calories_per_strawberry : ℕ := 4
def calories_per_ounce_of_yogurt : ℕ := 17
def strawberries_eaten : ℕ := 12
def yogurt_eaten_in_ounces : ℕ := 6

-- Theorem statement
theorem calories_consummed (c_straw : ℕ) (c_yogurt : ℕ) (straw : ℕ) (yogurt : ℕ) 
  (h1 : c_straw = calories_per_strawberry) 
  (h2 : c_yogurt = calories_per_ounce_of_yogurt) 
  (h3 : straw = strawberries_eaten) 
  (h4 : yogurt = yogurt_eaten_in_ounces) : 
  c_straw * straw + c_yogurt * yogurt = 150 :=
by 
  -- Derived conditions
  rw [h1, h2, h3, h4]
  sorry

end calories_consummed_l4_4131


namespace sin_330_eq_neg_half_l4_4035

-- Define conditions as hypotheses in Lean
def angle_330 (θ : ℝ) : Prop := θ = 330
def angle_transform (θ : ℝ) : Prop := θ = 360 - 30
def sin_pos (θ : ℝ) : Prop := Real.sin θ = 1 / 2
def sin_neg_in_4th_quadrant (θ : ℝ) : Prop := θ = 330 -> Real.sin θ < 0

-- The main theorem statement
theorem sin_330_eq_neg_half : ∀ θ : ℝ, angle_330 θ → angle_transform θ → sin_pos 30 → sin_neg_in_4th_quadrant θ → Real.sin θ = -1 / 2 := by
  intro θ h1 h2 h3 h4
  sorry

end sin_330_eq_neg_half_l4_4035


namespace cheburashkas_erased_l4_4765

theorem cheburashkas_erased (n : ℕ) (rows : ℕ) (krakozyabras : ℕ) 
  (h_spacing : ∀ r, r ≤ rows → krakozyabras = 2 * (n - 1))
  (h_rows : rows = 2)
  (h_krakozyabras : krakozyabras = 29) :
  n = 16 → rows = 2 → krakozyabras = 29 → n = 16 - 5 :=
by
  sorry

end cheburashkas_erased_l4_4765


namespace sufficient_but_not_necessary_condition_l4_4633

theorem sufficient_but_not_necessary_condition 
    (a : ℝ) (h_pos : a > 0)
    (h_line : ∀ x y, 2 * a * x - y + 2 * a^2 = 0)
    (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / 4 = 1) :
    (a ≥ 2) → 
    (∀ x y, ¬ (2 * a * x - y + 2 * a^2 = 0 ∧ x^2 / a^2 - y^2 / 4 = 1)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l4_4633


namespace max_abs_x2_is_2_l4_4588

noncomputable def max_abs_x2 {x₁ x₂ x₃ : ℝ} (h : x₁^2 + x₂^2 + x₃^2 + x₁ * x₂ + x₂ * x₃ = 2) : ℝ :=
2

theorem max_abs_x2_is_2 {x₁ x₂ x₃ : ℝ} (h : x₁^2 + x₂^2 + x₃^2 + x₁ * x₂ + x₂ * x₃ = 2) :
  max_abs_x2 h = 2 := 
sorry

end max_abs_x2_is_2_l4_4588


namespace number_of_questions_per_survey_is_10_l4_4428

variable {Q : ℕ}  -- Q: Number of questions in each survey

def money_per_question : ℝ := 0.2
def surveys_on_monday : ℕ := 3
def surveys_on_tuesday : ℕ := 4
def total_money_earned : ℝ := 14

theorem number_of_questions_per_survey_is_10 :
    (surveys_on_monday + surveys_on_tuesday) * Q * money_per_question = total_money_earned → Q = 10 :=
by
  sorry

end number_of_questions_per_survey_is_10_l4_4428


namespace simplify_cubed_root_l4_4810

def c1 : ℕ := 54880000
def c2 : ℕ := 10^5 * 5488
def c3 : ℕ := 5488
def c4 : ℕ := 2^4 * 343
def c5 : ℕ := 343
def c6 : ℕ := 7^3

theorem simplify_cubed_root : (c1^(1 / 3 : ℝ) : ℝ) = 1400 := 
by {
  let h1 : c1 = c2 := sorry,
  let h2 : c3 = c4 := sorry,
  let h3 : c5 = c6 := sorry,
  rw [h1, h2, h3],
  sorry
}

end simplify_cubed_root_l4_4810


namespace cubic_roots_and_k_value_l4_4797

theorem cubic_roots_and_k_value (k r₃ : ℝ) :
  (∃ r₃, 3 - 2 + r₃ = -5 ∧ 3 * (-2) * r₃ = -12 ∧ k = 3 * (-2) + (-2) * r₃ + r₃ * 3) →
  (k = -12 ∧ r₃ = -6) :=
by
  sorry

end cubic_roots_and_k_value_l4_4797


namespace subtraction_of_twos_from_ones_l4_4860

theorem subtraction_of_twos_from_ones (n : ℕ) : 
  let ones := (10^n - 1) * 10^n + (10^n - 1)
  let twos := 2 * (10^n - 1)
  ones - twos = (10^n - 1) * (10^n - 1) :=
by
  sorry

end subtraction_of_twos_from_ones_l4_4860


namespace a100_gt_2pow99_l4_4503

theorem a100_gt_2pow99 (a : Fin 101 → ℕ) 
  (h_pos : ∀ i, a i > 0) 
  (h_initial : a 1 > a 0) 
  (h_rec : ∀ k, 2 ≤ k → a k = 3 * a (k - 1) - 2 * a (k - 2)) 
  : a 100 > 2 ^ 99 :=
by
  sorry

end a100_gt_2pow99_l4_4503


namespace tax_on_clothing_l4_4489

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

end tax_on_clothing_l4_4489


namespace inequality_transformation_incorrect_l4_4074

theorem inequality_transformation_incorrect (a b : ℝ) (h : a > b) : (3 - a > 3 - b) -> false :=
by
  intros h1
  simp at h1
  sorry

end inequality_transformation_incorrect_l4_4074


namespace talkingBirds_count_l4_4416

-- Define the conditions
def totalBirds : ℕ := 77
def nonTalkingBirds : ℕ := 13
def talkingBirds (T : ℕ) : Prop := T + nonTalkingBirds = totalBirds

-- Statement to prove
theorem talkingBirds_count : ∃ T, talkingBirds T ∧ T = 64 :=
by
  -- Proof will go here
  sorry

end talkingBirds_count_l4_4416


namespace total_chairs_taken_l4_4367

def num_students : ℕ := 5
def chairs_per_trip : ℕ := 5
def num_trips : ℕ := 10

theorem total_chairs_taken :
  (num_students * chairs_per_trip * num_trips) = 250 :=
by
  sorry

end total_chairs_taken_l4_4367


namespace oranges_purchase_cost_l4_4798

/-- 
Oranges are sold at a rate of $3$ per three pounds.
If a customer buys 18 pounds and receives a discount of $5\%$ for buying more than 15 pounds,
prove that the total amount the customer pays is $17.10.
-/
theorem oranges_purchase_cost (rate : ℕ) (base_weight : ℕ) (discount_rate : ℚ)
  (total_weight : ℕ) (discount_threshold : ℕ) (final_cost : ℚ) :
  rate = 3 → base_weight = 3 → discount_rate = 0.05 → 
  total_weight = 18 → discount_threshold = 15 → final_cost = 17.10 := by
  sorry

end oranges_purchase_cost_l4_4798


namespace solution_set_ineq_l4_4566

theorem solution_set_ineq (x : ℝ) : 
  x * (x + 2) > 0 → abs x < 1 → 0 < x ∧ x < 1 := by
sorry

end solution_set_ineq_l4_4566


namespace infinitely_many_triples_no_triples_l4_4095

theorem infinitely_many_triples :
  ∃ (m n p : ℕ), ∃ (k : ℕ), m > 0 ∧ n > 0 ∧ p > 0 ∧ 4 * m * n - m - n = p ^ 2 - 1 := 
sorry

theorem no_triples :
  ¬∃ (m n p : ℕ), m > 0 ∧ n > 0 ∧ p > 0 ∧ 4 * m * n - m - n = p ^ 2 := 
sorry

end infinitely_many_triples_no_triples_l4_4095


namespace visited_both_countries_l4_4471

theorem visited_both_countries (total_people visited_Iceland visited_Norway visited_neither : ℕ) 
(h_total: total_people = 60)
(h_visited_Iceland: visited_Iceland = 35)
(h_visited_Norway: visited_Norway = 23)
(h_visited_neither: visited_neither = 33) : 
total_people - visited_neither = visited_Iceland + visited_Norway - (visited_Iceland + visited_Norway - (total_people - visited_neither)) :=
by sorry

end visited_both_countries_l4_4471


namespace money_left_l4_4998

noncomputable def initial_amount : ℝ := 10.10
noncomputable def spent_on_sweets : ℝ := 3.25
noncomputable def amount_per_friend : ℝ := 2.20
noncomputable def remaining_amount : ℝ := initial_amount - spent_on_sweets - 2 * amount_per_friend

theorem money_left : remaining_amount = 2.45 :=
by
  sorry

end money_left_l4_4998


namespace jogging_time_after_two_weeks_l4_4485

noncomputable def daily_jogging_hours : ℝ := 1.5
noncomputable def days_in_two_weeks : ℕ := 14

theorem jogging_time_after_two_weeks : daily_jogging_hours * days_in_two_weeks = 21 := by
  sorry

end jogging_time_after_two_weeks_l4_4485


namespace janet_stuffies_l4_4082

theorem janet_stuffies (total_stuffies kept_stuffies given_away_stuffies janet_stuffies : ℕ) 
 (h1 : total_stuffies = 60)
 (h2 : kept_stuffies = total_stuffies / 3)
 (h3 : given_away_stuffies = total_stuffies - kept_stuffies)
 (h4 : janet_stuffies = given_away_stuffies / 4) : 
 janet_stuffies = 10 := 
sorry

end janet_stuffies_l4_4082


namespace total_weight_of_bottles_l4_4117

variables (P G : ℕ) -- P stands for the weight of a plastic bottle, G stands for the weight of a glass bottle

-- Condition 1: The weight of 3 glass bottles is 600 grams
axiom glass_bottle_weight : 3 * G = 600

-- Condition 2: A glass bottle is 150 grams heavier than a plastic bottle
axiom glass_bottle_heavier : G = P + 150

-- The statement to prove: The total weight of 4 glass bottles and 5 plastic bottles is 1050 grams
theorem total_weight_of_bottles :
  4 * G + 5 * P = 1050 :=
sorry

end total_weight_of_bottles_l4_4117


namespace gcf_180_270_l4_4125

def prime_factors_180 : list (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1)]
def prime_factors_270 : list (ℕ × ℕ) := [(2, 1), (3, 3), (5, 1)]

def GCF (a b : ℕ) : ℕ := sorry -- provide actual implementation of GCF calculation if needed

theorem gcf_180_270 : GCF 180 270 = 90 := by 
    -- use the given prime factorizations to arrive at the conclusion
    sorry

end gcf_180_270_l4_4125


namespace real_value_of_m_pure_imaginary_value_of_m_l4_4573

open Complex

-- Given condition
def z (m : ℝ) : ℂ := (m^2 - m : ℂ) - (m^2 - 1 : ℂ) * I

-- Part (I)
theorem real_value_of_m (m : ℝ) (h : im (z m) = 0) : m = 1 ∨ m = -1 := by
  sorry

-- Part (II)
theorem pure_imaginary_value_of_m (m : ℝ) (h1 : re (z m) = 0) (h2 : im (z m) ≠ 0) : m = 0 := by
  sorry

end real_value_of_m_pure_imaginary_value_of_m_l4_4573


namespace right_triangle_area_l4_4828

theorem right_triangle_area (a b c : ℝ) (h : a^2 + b^2 = c^2) (ha : a = 5) (hc : c = 13) :
  1/2 * a * b = 30 :=
by
  have hb : b = 12, from sorry,
  -- Proof needs to be filled here
  sorry

end right_triangle_area_l4_4828


namespace factorize_expression_l4_4695

theorem factorize_expression (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 :=
by sorry

end factorize_expression_l4_4695


namespace num_mystery_shelves_l4_4242

def num_books_per_shelf : ℕ := 9
def num_picture_shelves : ℕ := 2
def total_books : ℕ := 72
def num_books_from_picture_shelves : ℕ := num_picture_shelves * num_books_per_shelf
def num_books_from_mystery_shelves : ℕ := total_books - num_books_from_picture_shelves

theorem num_mystery_shelves :
  num_books_from_mystery_shelves / num_books_per_shelf = 6 := by
sorry

end num_mystery_shelves_l4_4242


namespace ellipse_eccentricity_l4_4450

def ellipse {a : ℝ} (h : a^2 - 4 = 4) : Prop :=
  ∃ c e : ℝ, (c = 2) ∧ (e = c / a) ∧ (e = (Real.sqrt 2) / 2)

theorem ellipse_eccentricity (a : ℝ) (h : a^2 - 4 = 4) : 
  ellipse h :=
by
  sorry

end ellipse_eccentricity_l4_4450


namespace minimum_value_of_f_l4_4897

noncomputable def f (x : ℝ) : ℝ := x^2 + 2*x - 4

theorem minimum_value_of_f : ∃ x : ℝ, f x = -5 ∧ ∀ y : ℝ, f y ≥ -5 :=
by
  sorry

end minimum_value_of_f_l4_4897


namespace total_amount_given_away_l4_4943

variable (numGrandchildren : ℕ)
variable (cardsPerGrandchild : ℕ)
variable (amountPerCard : ℕ)

theorem total_amount_given_away (h1 : numGrandchildren = 3) (h2 : cardsPerGrandchild = 2) (h3 : amountPerCard = 80) : 
  numGrandchildren * cardsPerGrandchild * amountPerCard = 480 := by
  sorry

end total_amount_given_away_l4_4943


namespace num_men_in_boat_l4_4743

theorem num_men_in_boat 
  (n : ℕ) (W : ℝ)
  (h1 : (W / n : ℝ) = W / n)
  (h2 : (W + 8) / n = W / n + 1)
  : n = 8 := 
sorry

end num_men_in_boat_l4_4743


namespace ramu_spent_on_repairs_l4_4805

theorem ramu_spent_on_repairs 
    (initial_cost : ℝ) (selling_price : ℝ) (profit_percent : ℝ) (R : ℝ) 
    (h1 : initial_cost = 42000) 
    (h2 : selling_price = 64900) 
    (h3 : profit_percent = 18) 
    (h4 : profit_percent / 100 = (selling_price - (initial_cost + R)) / (initial_cost + R)) : 
    R = 13000 :=
by
  rw [h1, h2, h3] at h4
  sorry

end ramu_spent_on_repairs_l4_4805


namespace find_x_l4_4479

theorem find_x (h : ℝ → ℝ)
  (H1 : ∀x, h (3*x - 2) = 5*x + 6) :
  (∀x, h x = 2*x - 1) → x = 31 :=
by
  sorry

end find_x_l4_4479


namespace unique_solution_k_l4_4193

theorem unique_solution_k (k : ℚ) : (∀ x : ℚ, x ≠ -2 → (x + 3)/(k*x - 2) = x) ↔ k = -3/4 :=
sorry

end unique_solution_k_l4_4193


namespace jane_drinks_l4_4477

/-- Jane buys a combination of muffins, bagels, and drinks over five days,
where muffins cost 40 cents, bagels cost 90 cents, and drinks cost 30 cents.
The number of items bought is 5, and the total cost is a whole number of dollars.
Prove that the number of drinks Jane bought is 4. -/
theorem jane_drinks :
  ∃ b m d : ℕ, b + m + d = 5 ∧ (90 * b + 40 * m + 30 * d) % 100 = 0 ∧ d = 4 :=
by
  sorry

end jane_drinks_l4_4477


namespace price_per_glass_first_day_l4_4622

theorem price_per_glass_first_day
    (O W : ℝ) (P1 P2 : ℝ)
    (h1 : O = W)
    (h2 : P2 = 0.40)
    (h3 : 2 * O * P1 = 3 * O * P2) :
    P1 = 0.60 :=
by
    sorry

end price_per_glass_first_day_l4_4622


namespace smallest_n_divisibility_problem_l4_4654

theorem smallest_n_divisibility_problem :
  ∃ (n : ℕ), n > 0 ∧ (∀ (k : ℕ), 1 ≤ k → k ≤ n + 2 → n^3 - n ≠ 0 → (n^3 - n) % k = 0) ∧
    (∃ (k : ℕ), 1 ≤ k → k ≤ n + 2 → k ∣ n^3 - n) ∧
    (∃ (k : ℕ), 1 ≤ k → k ≤ n + 2 → ¬ k ∣ n^3 - n) ∧
    (∀ (m : ℕ), m > 0 ∧ (∀ (k : ℕ), 1 ≤ k → k ≤ m + 2 → m^3 - m ≠ 0 → (m^3 - m) % k = 0) ∧
      (∃ (k : ℕ), 1 ≤ k → k ≤ m + 2 → k ∣ m^3 - m) ∧
      (∃ (k : ℕ), 1 ≤ k → k ≤ m + 2 → ¬ k ∣ m^3 - m) → n ≤ m) :=
sorry

end smallest_n_divisibility_problem_l4_4654


namespace pirate_treasure_l4_4291

theorem pirate_treasure (N : ℕ) (h1 : ∀ k : ℕ, k < 15 → 15 ∣ (N * (15 - k - 1) ^ (14 - k))) :
  N = 208080 :=
begin
  sorry
end

#print axioms pirate_treasure

end pirate_treasure_l4_4291


namespace k_range_l4_4913

theorem k_range (k : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → 0 ≤ 2 * x - 2 * k) → k ≤ 1 :=
by
  intro h
  have h1 := h 1 (by simp)
  have h3 := h 3 (by simp)
  sorry

end k_range_l4_4913


namespace path_area_and_cost_l4_4134

theorem path_area_and_cost:
  let length_grass_field := 75
  let width_grass_field := 55
  let path_width := 3.5
  let cost_per_sq_meter := 2
  let length_with_path := length_grass_field + 2 * path_width
  let width_with_path := width_grass_field + 2 * path_width
  let area_with_path := length_with_path * width_with_path
  let area_grass_field := length_grass_field * width_grass_field
  let area_path := area_with_path - area_grass_field
  let cost_of_construction := area_path * cost_per_sq_meter
  area_path = 959 ∧ cost_of_construction = 1918 :=
by
  sorry

end path_area_and_cost_l4_4134


namespace find_f_2021_l4_4393

def f (x : ℝ) : ℝ := sorry

theorem find_f_2021 (h : ∀ a b : ℝ, f ((a + 2 * b) / 3) = (f a + 2 * f b) / 3)
    (h1 : f 1 = 5) (h4 : f 4 = 2) : f 2021 = -2015 :=
by
  sorry

end find_f_2021_l4_4393


namespace smallest_palindrome_not_five_digit_l4_4900

theorem smallest_palindrome_not_five_digit (n : ℕ) (h : 100 ≤ n ∧ n ≤ 999 ∧ n % 10 = n / 100 ∧ n / 10 % 10 = n / 100 ∧ 103 * n < 10000) :
  n = 707 := by
sorry

end smallest_palindrome_not_five_digit_l4_4900


namespace chord_bisected_by_point_l4_4463

theorem chord_bisected_by_point (x1 y1 x2 y2 : ℝ) :
  (x1^2 / 36 + y1^2 / 9 = 1) ∧ (x2^2 / 36 + y2^2 / 9 = 1) ∧ 
  (x1 + x2 = 4) ∧ (y1 + y2 = 4) → (x + 4 * y - 10 = 0) :=
sorry

end chord_bisected_by_point_l4_4463


namespace notched_circle_coordinates_l4_4874

variable (a b : ℝ)

theorem notched_circle_coordinates : 
  let sq_dist_from_origin := a^2 + b^2
  let A := (a, b + 5)
  let C := (a + 3, b)
  (a^2 + (b + 5)^2 = 36 ∧ (a + 3)^2 + b^2 = 36) →
  (sq_dist_from_origin = 4.16 ∧ A = (2, 5.4) ∧ C = (5, 0.4)) :=
by
  sorry

end notched_circle_coordinates_l4_4874


namespace range_f_l4_4319

noncomputable def f (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.log (1 + x)

theorem range_f : 
  (set_of (λ y, ∃ x : ℝ, -1 < x ∧ x ≤ 1 ∧ y = f x)) = Iic (Real.pi / 2 + Real.log 2) :=
by
  sorry

end range_f_l4_4319


namespace simplified_fraction_l4_4097

theorem simplified_fraction :
  (1 / (1 / (1 / 3)^1 + 1 / (1 / 3)^2 + 1 / (1 / 3)^3 + 1 / (1 / 3)^4)) = (1 / 120) :=
by 
  sorry

end simplified_fraction_l4_4097


namespace perimeter_of_rectangle_l4_4259

theorem perimeter_of_rectangle (s : ℝ) (h1 : 4 * s = 160) : 2 * (s + s / 4) = 100 :=
by
  sorry

end perimeter_of_rectangle_l4_4259


namespace angle_FDY_l4_4742

theorem angle_FDY :
  ∀ {X Y Z D F : EuclideanGeometry.Point},
  (XZ = YZ) →
  (m∠ DYZ = 50) →
  (DY ∥ XZ) →
  (m∠ FDY = 50) :=
by
  intros X Y Z D F hXZ hDYZ hpar
  sorry

end angle_FDY_l4_4742


namespace side_of_rhombus_l4_4296

variable (d : ℝ) (K : ℝ) 

-- Conditions
def shorter_diagonal := d
def longer_diagonal := 3 * d
def area_rhombus := K = (1 / 2) * d * (3 * d)

-- Proof Statement
theorem side_of_rhombus (h1 : K = (3 / 2) * d^2) : (∃ s : ℝ, s = Real.sqrt (5 * K / 3)) := 
  sorry

end side_of_rhombus_l4_4296


namespace Tina_profit_l4_4518

variables (x : ℝ) (profit_per_book : ℝ) (number_of_people : ℕ) (cost_per_book : ℝ)
           (books_per_customer : ℕ) (total_profit : ℝ) (total_cost : ℝ) (total_books_sold : ℕ)

theorem Tina_profit :
  (number_of_people = 4) →
  (cost_per_book = 5) →
  (books_per_customer = 2) →
  (total_profit = 120) →
  (books_per_customer * number_of_people = total_books_sold) →
  (cost_per_book * total_books_sold = total_cost) →
  (total_profit = total_books_sold * x - total_cost) →
  x = 20 :=
by
  intros
  sorry


end Tina_profit_l4_4518


namespace cannot_take_value_l4_4569

theorem cannot_take_value (x y : ℝ) (h : |x| + |y| = 13) : 
  ∀ (v : ℝ), x^2 + 7*x - 3*y + y^2 = v → (0 ≤ v ∧ v ≤ 260) := 
by
  sorry

end cannot_take_value_l4_4569


namespace find_y_given_x_eq_0_l4_4245

theorem find_y_given_x_eq_0 (t : ℚ) (x y : ℚ) (h1 : x = 3 - 2 * t) (h2 : y = 3 * t + 6) (h3 : x = 0) : 
  y = 21 / 2 :=
by
  sorry

end find_y_given_x_eq_0_l4_4245


namespace cassidy_total_grounding_days_l4_4433

-- Define the initial grounding days
def initial_grounding_days : ℕ := 14

-- Define the grounding days per grade below a B
def extra_days_per_grade : ℕ := 3

-- Define the number of grades below a B
def grades_below_B : ℕ := 4

-- Define the total grounding days calculation
def total_grounding_days : ℕ := initial_grounding_days + grades_below_B * extra_days_per_grade

-- The theorem statement
theorem cassidy_total_grounding_days :
  total_grounding_days = 26 := 
sorry

end cassidy_total_grounding_days_l4_4433


namespace mark_has_seven_butterfingers_l4_4960

/-
  Mark has 12 candy bars in total between Mars bars, Snickers, and Butterfingers.
  He has 3 Snickers and 2 Mars bars.
  Prove that he has 7 Butterfingers.
-/

noncomputable def total_candy_bars : Nat := 12
noncomputable def snickers : Nat := 3
noncomputable def mars_bars : Nat := 2
noncomputable def butterfingers : Nat := total_candy_bars - (snickers + mars_bars)

theorem mark_has_seven_butterfingers : butterfingers = 7 := by
  sorry

end mark_has_seven_butterfingers_l4_4960


namespace equation_is_linear_l4_4739

-- Define the conditions and the proof statement
theorem equation_is_linear (m n : ℕ) : 3 * x ^ (2 * m + 1) - 2 * y ^ (n - 1) = 7 → (2 * m + 1 = 1) ∧ (n - 1 = 1) → m = 0 ∧ n = 2 :=
by
  sorry

end equation_is_linear_l4_4739


namespace no_intersection_points_l4_4067

-- Define f(x) and g(x)
def f (x : ℝ) : ℝ := abs (3 * x + 6)
def g (x : ℝ) : ℝ := -abs (4 * x - 3)

-- The main theorem to prove the number of intersection points is zero
theorem no_intersection_points : ∀ x : ℝ, f x ≠ g x := by
  intro x
  sorry -- Proof goes here

end no_intersection_points_l4_4067


namespace sin_330_eq_neg_half_l4_4022

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  -- Proof would go here
  sorry

end sin_330_eq_neg_half_l4_4022


namespace largest_possible_b_l4_4982

theorem largest_possible_b (a b c : ℕ) (h₁ : 1 < c) (h₂ : c < b) (h₃ : b < a) (h₄ : a * b * c = 360): b = 12 :=
by
  sorry

end largest_possible_b_l4_4982


namespace unique_intersection_point_l4_4509

theorem unique_intersection_point (m : ℝ) :
  (∀ x : ℝ, ((m + 1) * x^2 - 2 * (m + 1) * x - 1 = 0) → x = -1) ↔ m = -2 :=
by
  sorry

end unique_intersection_point_l4_4509


namespace quadratic_function_m_value_l4_4466

theorem quadratic_function_m_value
  (m : ℝ)
  (h1 : m^2 - 7 = 2)
  (h2 : 3 - m ≠ 0) :
  m = -3 := by
  sorry

end quadratic_function_m_value_l4_4466


namespace factorization_correct_l4_4699

-- Define the initial expression
def initial_expr (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + a

-- Define the factorized form
def factorized_expr (a x : ℝ) : ℝ := a * (x - 1)^2

-- Create the theorem statement
theorem factorization_correct (a x : ℝ) : initial_expr a x = factorized_expr a x :=
by simp [initial_expr, factorized_expr, pow_two, mul_add, add_mul, add_assoc]; sorry

end factorization_correct_l4_4699


namespace product_of_roots_l4_4173

-- Let's define the given polynomial equation.
def polynomial : Polynomial ℝ := Polynomial.monomial 3 1 - Polynomial.monomial 2 15 + Polynomial.monomial 1 75 - Polynomial.C 50

-- Prove that the product of the roots of the given polynomial is 50.
theorem product_of_roots : (Polynomial.roots polynomial).prod id = 50 := by
  sorry

end product_of_roots_l4_4173


namespace problem_l4_4725

theorem problem (θ : ℝ) (htan : Real.tan θ = 1 / 3) : Real.cos θ ^ 2 + 2 * Real.sin θ = 6 / 5 := 
by
  sorry

end problem_l4_4725


namespace equivalent_polar_point_representation_l4_4935

/-- Representation of a point in polar coordinates -/
structure PolarPoint :=
  (r : ℝ)
  (θ : ℝ)

theorem equivalent_polar_point_representation :
  ∀ (p1 p2 : PolarPoint), p1 = PolarPoint.mk (-1) (5 * Real.pi / 6) →
    (p2 = PolarPoint.mk 1 (11 * Real.pi / 6) → p1.r + Real.pi = p2.r ∧ p1.θ = p2.θ) :=
by
  intros p1 p2 h1 h2
  sorry

end equivalent_polar_point_representation_l4_4935


namespace find_x_solution_l4_4585

theorem find_x_solution (x b c : ℝ) (h_eq : x^2 + c^2 = (b - x)^2):
  x = (b^2 - c^2) / (2 * b) :=
sorry

end find_x_solution_l4_4585


namespace find_x_intercept_of_perpendicular_line_l4_4415

noncomputable def line_y_intercept : ℝ × ℝ := (0, 3)
noncomputable def given_line (x y : ℝ) : Prop := 2 * x + y = 3
noncomputable def x_intercept_of_perpendicular_line : ℝ × ℝ := (-6, 0)

theorem find_x_intercept_of_perpendicular_line :
  (∀ (x y : ℝ), given_line x y → (slope_of_perpendicular_line : ℝ) = 1/2 ∧ 
  ∀ (b : ℝ), line_y_intercept = (0, b) → ∀ (y : ℝ), y = 1/2 * x + b → (x, 0) = x_intercept_of_perpendicular_line) :=
sorry

end find_x_intercept_of_perpendicular_line_l4_4415


namespace distinct_ordered_pairs_proof_l4_4735

def num_distinct_ordered_pairs_satisfying_reciprocal_sum : ℕ :=
  List.length [
    (7, 42), (8, 24), (9, 18), (10, 15), 
    (12, 12), (15, 10), (18, 9), (24, 8), 
    (42, 7)
  ]

theorem distinct_ordered_pairs_proof : num_distinct_ordered_pairs_satisfying_reciprocal_sum = 9 := by
  sorry

end distinct_ordered_pairs_proof_l4_4735


namespace smallest_x_mul_900_multiple_of_1152_l4_4270

theorem smallest_x_mul_900_multiple_of_1152 : 
  ∃ x : ℕ, (x > 0) ∧ (900 * x) % 1152 = 0 ∧ ∀ y : ℕ, (y > 0) ∧ (900 * y) % 1152 = 0 → y ≥ x := 
begin
  use 32,
  split,
  { exact nat.one_pos, }, -- the positive condition
  split,
  { change (900 * 32) % 1152 = 0, -- 32 satisfies the multiple condition
    norm_num, 
  },
  { intros y hy, -- minimality condition
    cases hy with hy_pos hy_dvd,
    have : 1152 ∣ 900 * 32 := by {
        change 900 * 32 % 1152 = 0,
        norm_num,
    },
    obtain ⟨k, hk⟩ := exists_eq_mul_left_of_dvd this,
    change 1152 * k < 1152 * 32,
    refine le_of_dvd (mul_pos (@nat.one_pos _) hy_pos) _,
    exact ⟨32, rfl⟩,
  },
end

end smallest_x_mul_900_multiple_of_1152_l4_4270


namespace proof_candle_burn_l4_4519

noncomputable def candle_burn_proof : Prop :=
∃ (t : ℚ),
  (t = 40 / 11) ∧
  (∀ (H_1 H_2 : ℚ → ℚ),
    (∀ t, H_1 t = 1 - t / 5) ∧
    (∀ t, H_2 t = 1 - t / 4) →
    ∃ (t : ℚ), ((1 - t / 5) = 3 * (1 - t / 4)) ∧ (t = 40 / 11))

theorem proof_candle_burn : candle_burn_proof :=
sorry

end proof_candle_burn_l4_4519


namespace sum_first_13_terms_l4_4643

theorem sum_first_13_terms
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h₀ : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h₁ : a 4 + a 10 - (a 7)^2 + 15 = 0)
  (h₂ : ∀ n : ℕ, a n > 0) :
  S 13 = 65 :=
sorry

end sum_first_13_terms_l4_4643


namespace gcd_lcm_product_l4_4044

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 24) (h2 : b = 45) : (Int.gcd a b * Nat.lcm a b) = 1080 := by
  rw [h1, h2]
  sorry

end gcd_lcm_product_l4_4044


namespace seq_a5_eq_one_ninth_l4_4049

theorem seq_a5_eq_one_ninth (a : ℕ → ℚ) (h1 : a 1 = 1) (h_rec : ∀ n, a (n + 1) = a n / (2 * a n + 1)) :
  a 5 = 1 / 9 :=
sorry

end seq_a5_eq_one_ninth_l4_4049


namespace sin_330_value_l4_4014

noncomputable def sin_330 : ℝ := Real.sin (330 * Real.pi / 180)

theorem sin_330_value : sin_330 = -1/2 :=
by {
  sorry
}

end sin_330_value_l4_4014


namespace factorization_correct_l4_4703

-- Define the initial expression
def initial_expr (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + a

-- Define the factorized form
def factorized_expr (a x : ℝ) : ℝ := a * (x - 1)^2

-- Create the theorem statement
theorem factorization_correct (a x : ℝ) : initial_expr a x = factorized_expr a x :=
by simp [initial_expr, factorized_expr, pow_two, mul_add, add_mul, add_assoc]; sorry

end factorization_correct_l4_4703


namespace mark_money_l4_4617

theorem mark_money (M : ℝ) 
  (h1 : (1 / 2) * M + 14 + (1 / 3) * M + 16 + (1 / 4) * M + 18 = M) : 
  M = 576 := 
sorry

end mark_money_l4_4617


namespace cube_root_of_54880000_l4_4811

theorem cube_root_of_54880000 : (real.cbrt 54880000) = 140 * (real.cbrt 10) :=
by
  -- Definitions based on conditions
  have h1 : 54880000 = 10^3 * 54880, by norm_num
  have h2 : 54880 = 2^5 * 7^3 * 5, by norm_num
  have h3 : 10 = 2 * 5, by norm_num
  
  -- Cube root properties and simplifications are implicitly inferred by the system
  sorry

end cube_root_of_54880000_l4_4811


namespace mason_total_nuts_l4_4794

/-- Mason opens the hood of his car and discovers that squirrels have been using his engine compartment to store nuts.
If 2 busy squirrels have been stockpiling 30 nuts/day and one sleepy squirrel has been stockpiling 20 nuts/day, all for 40 days, 
the total number of nuts in Mason's car is 3200. -/
theorem mason_total_nuts : 
  ∀ (nuts_per_day_busy : ℕ) (num_busy_squirrels : ℕ) 
    (nuts_per_day_sleepy : ℕ) (num_sleepy_squirrels : ℕ) 
    (num_days : ℕ), 
  nuts_per_day_busy = 30 → 
  num_busy_squirrels = 2 → 
  nuts_per_day_sleepy = 20 → 
  num_sleepy_squirrels = 1 → 
  num_days = 40 →
  ((num_busy_squirrels * nuts_per_day_busy) + (num_sleepy_squirrels * nuts_per_day_sleepy)) * num_days = 3200 := 
by 
  intros nuts_per_day_busy num_busy_squirrels nuts_per_day_sleepy num_sleepy_squirrels num_days
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  exact sorry

end mason_total_nuts_l4_4794


namespace midpoint_uniqueness_l4_4243

-- Define a finite set of points in the plane
axiom S : Finset (ℝ × ℝ)

-- Define what it means for P to be the midpoint of a segment
def is_midpoint (P A A' : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + A'.1) / 2 ∧ P.2 = (A.2 + A'.2) / 2

-- Statement of the problem
theorem midpoint_uniqueness (P Q : ℝ × ℝ) :
  (∀ A ∈ S, ∃ A' ∈ S, is_midpoint P A A') →
  (∀ A ∈ S, ∃ A' ∈ S, is_midpoint Q A A') →
  P = Q :=
sorry

end midpoint_uniqueness_l4_4243


namespace geometric_series_S6_value_l4_4453

theorem geometric_series_S6_value (S : ℕ → ℝ) (S3 : S 3 = 3) (S9_minus_S6 : S 9 - S 6 = 12) : 
  S 6 = 9 :=
by
  sorry

end geometric_series_S6_value_l4_4453


namespace proveCarTransportationProblem_l4_4848

def carTransportationProblem :=
  ∃ x y a b : ℕ,
  -- Conditions regarding the capabilities of the cars
  (2 * x + 3 * y = 18) ∧
  (x + 2 * y = 11) ∧
  -- Conclusion (question 1)
  (x + y = 7) ∧
  -- Conditions for the rental plan (question 2)
  (3 * a + 4 * b = 27) ∧
  -- Cost optimization
  ((100 * a + 120 * b) = 820 ∨ (100 * a + 120 * b) = 860) ∧
  -- Optimal cost verification
  (100 * a + 120 * b = 820 → a = 1 ∧ b = 6)

theorem proveCarTransportationProblem : carTransportationProblem :=
  sorry

end proveCarTransportationProblem_l4_4848


namespace gcd_8m_6n_l4_4596

theorem gcd_8m_6n (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : Nat.gcd m n = 7) : Nat.gcd (8 * m) (6 * n) = 14 := 
by
  sorry

end gcd_8m_6n_l4_4596


namespace triangular_array_sum_digits_l4_4158

theorem triangular_array_sum_digits (N : ℕ) (h : N * (N + 1) / 2 = 3780) : (N / 10 + N % 10) = 15 :=
sorry

end triangular_array_sum_digits_l4_4158


namespace shaded_area_of_rotated_semicircle_l4_4561

theorem shaded_area_of_rotated_semicircle (R : ℝ) :
  let α := 60 * (Real.pi / 180) in -- 60 degrees in radians
  let S0 := (Real.pi * R^2) / 2 in
  let SectorArea := (1 / 2) * (2 * R)^2 * (Real.pi / 3) in
  SectorArea = (2 * Real.pi * R^2) / 3 :=
by
  sorry

end shaded_area_of_rotated_semicircle_l4_4561


namespace bail_rate_l4_4501

theorem bail_rate 
  (distance_to_shore : ℝ) 
  (shore_speed : ℝ) 
  (leak_rate : ℝ) 
  (boat_capacity : ℝ) 
  (time_to_shore_min : ℝ) 
  (net_water_intake : ℝ)
  (r : ℝ) :
  distance_to_shore = 2 →
  shore_speed = 3 →
  leak_rate = 12 →
  boat_capacity = 40 →
  time_to_shore_min = 40 →
  net_water_intake = leak_rate - r →
  net_water_intake * (time_to_shore_min) ≤ boat_capacity →
  r ≥ 11 :=
by
  intros h_dist h_speed h_leak h_cap h_time h_net h_ineq
  sorry

end bail_rate_l4_4501


namespace angle_C_in_triangle_l4_4757

theorem angle_C_in_triangle (A B C : ℝ) 
  (hA : A = 90) (hB : B = 50) : (A + B + C = 180) → C = 40 :=
by
  intro hSum
  rw [hA, hB] at hSum
  linarith

end angle_C_in_triangle_l4_4757


namespace find_ordered_pair_l4_4502

theorem find_ordered_pair (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0)
  (h₃ : (x : ℝ) → x^2 + 2 * a * x + b = 0 → x = a ∨ x = b) :
  (a, b) = (1, -3) :=
sorry

end find_ordered_pair_l4_4502


namespace max_subjects_per_teacher_l4_4545

theorem max_subjects_per_teacher (math_teachers physics_teachers chemistry_teachers min_teachers : ℕ)
  (h_math : math_teachers = 4)
  (h_physics : physics_teachers = 3)
  (h_chemistry : chemistry_teachers = 3)
  (h_min_teachers : min_teachers = 5) :
  (math_teachers + physics_teachers + chemistry_teachers) / min_teachers = 2 :=
by
  sorry

end max_subjects_per_teacher_l4_4545


namespace cubics_product_equals_1_over_1003_l4_4400

theorem cubics_product_equals_1_over_1003
  (x_1 y_1 x_2 y_2 x_3 y_3 : ℝ)
  (h1 : x_1^3 - 3 * x_1 * y_1^2 = 2007)
  (h2 : y_1^3 - 3 * x_1^2 * y_1 = 2006)
  (h3 : x_2^3 - 3 * x_2 * y_2^2 = 2007)
  (h4 : y_2^3 - 3 * x_2^2 * y_2 = 2006)
  (h5 : x_3^3 - 3 * x_3 * y_3^2 = 2007)
  (h6 : y_3^3 - 3 * x_3^2 * y_3 = 2006) :
  (1 - x_1 / y_1) * (1 - x_2 / y_2) * (1 - x_3 / y_3) = 1 / 1003 :=
by
  sorry

end cubics_product_equals_1_over_1003_l4_4400


namespace math_problem_l4_4042

theorem math_problem (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y + x + y = 83) (h4 : x^2 * y + x * y^2 = 1056) :
  x^2 + y^2 = 458 := by 
  sorry

end math_problem_l4_4042


namespace how_many_cheburashkas_erased_l4_4772

theorem how_many_cheburashkas_erased 
  (total_krakozyabras : ℕ)
  (characters_per_row_initial : ℕ) 
  (total_characters_initial : ℕ)
  (total_cheburashkas : ℕ)
  (total_rows : ℕ := 2)
  (total_krakozyabras := 29) :
  total_cheburashkas = 11 :=
by
  sorry

end how_many_cheburashkas_erased_l4_4772


namespace product_of_roots_of_cubic_eqn_l4_4188

theorem product_of_roots_of_cubic_eqn :
  let p := Polynomial.Cubic (1 : ℝ) (-15) 75 (-50)
  in p.roots_prod = 50 :=
by
  sorry

end product_of_roots_of_cubic_eqn_l4_4188


namespace ab_ac_bc_nonpositive_l4_4783

theorem ab_ac_bc_nonpositive (a b c : ℝ) (h : a + b + c = 0) : ∃ y : ℝ, y = ab + ac + bc ∧ y ≤ 0 :=
by
  sorry

end ab_ac_bc_nonpositive_l4_4783


namespace ab_ac_bc_range_l4_4784

theorem ab_ac_bc_range (a b c : ℝ) (h : a + b + c = 0) : ab + ac + bc ∈ Iic 0 := by
  sorry

end ab_ac_bc_range_l4_4784


namespace arithmetic_sequence_difference_l4_4336

theorem arithmetic_sequence_difference 
  (a b c : ℝ) 
  (h1: 2 + (7 / 4) = a)
  (h2: 2 + 2 * (7 / 4) = b)
  (h3: 2 + 3 * (7 / 4) = c)
  (h4: 2 + 4 * (7 / 4) = 9):
  c - a = 3.5 :=
by sorry

end arithmetic_sequence_difference_l4_4336


namespace difference_of_squares_l4_4312

theorem difference_of_squares (n : ℤ) : 4 - n^2 = (2 + n) * (2 - n) := 
by
  -- Proof goes here
  sorry

end difference_of_squares_l4_4312


namespace sum_of_adjacent_to_7_l4_4638

/-- Define the divisors of 245, excluding 1 -/
def divisors245 : Set ℕ := {5, 7, 35, 49, 245}

/-- Define the adjacency condition to ensure every pair of adjacent integers has a common factor greater than 1 -/
def adjacency_condition (a b : ℕ) : Prop := (a ≠ b) ∨ (Nat.gcd a b > 1)

/-- Prove the sum of the two integers adjacent to 7 in the given condition is 294. -/
theorem sum_of_adjacent_to_7 (d1 d2 : ℕ) (h1 : d1 ∈ divisors245) (h2 : d2 ∈ divisors245) 
    (adj1 : adjacency_condition 7 d1) (adj2 : adjacency_condition 7 d2) : 
    d1 + d2 = 294 := 
sorry

end sum_of_adjacent_to_7_l4_4638


namespace sin_double_angle_l4_4332

variable (θ : ℝ)

-- Given condition: tan(θ) = -3/5
def tan_theta : Prop := Real.tan θ = -3/5

-- Target to prove: sin(2θ) = -15/17
theorem sin_double_angle : tan_theta θ → Real.sin (2*θ) = -15/17 :=
by
  sorry

end sin_double_angle_l4_4332


namespace total_distance_of_race_is_150_l4_4472

variable (D : ℝ)

-- Conditions
def A_covers_distance_in_45_seconds (D : ℝ) : Prop := ∃ A_speed, A_speed = D / 45
def B_covers_distance_in_60_seconds (D : ℝ) : Prop := ∃ B_speed, B_speed = D / 60
def A_beats_B_by_50_meters_in_60_seconds (D : ℝ) : Prop := (D / 45) * 60 = D + 50

theorem total_distance_of_race_is_150 :
  A_covers_distance_in_45_seconds D ∧ 
  B_covers_distance_in_60_seconds D ∧ 
  A_beats_B_by_50_meters_in_60_seconds D → 
  D = 150 :=
by
  sorry

end total_distance_of_race_is_150_l4_4472


namespace product_of_roots_eq_50_l4_4180

theorem product_of_roots_eq_50 :
  let a := -15
  let c := -50
  let equation := λ x : ℝ, x^3 - 15 * x^2 + 75 * x - 50
  -- Vieta's formulas for the product of roots in cubic equation ax^3 + bx^2 + cx + d = 0 is -d/a
  ∃ p q r : ℝ, 
    equation p = 0 ∧ equation q = 0 ∧ equation r = 0 ∧ (p * q * r = 50) :=
by
  sorry

end product_of_roots_eq_50_l4_4180


namespace simplify_cube_root_l4_4807

theorem simplify_cube_root (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ)
  (h1 : a = 10^3 * b)
  (h2 : b = 2^7 * c * 7^3)
  (h3 : c = 10) :
  ∛a = 40 * 7 * 2^(2/3) * 5^(1/3) := by
  sorry

end simplify_cube_root_l4_4807


namespace factorize_expression_l4_4706

theorem factorize_expression (a x : ℝ) :
  a * x^2 - 2 * a * x + a = a * (x - 1) ^ 2 := 
sorry

end factorize_expression_l4_4706


namespace cheburashkas_erased_l4_4780

def total_krakozyabras : ℕ := 29

def total_rows : ℕ := 2

def cheburashkas_per_row := (total_krakozyabras + total_rows) / total_rows / 2 + 1

theorem cheburashkas_erased :
  (total_krakozyabras + total_rows) / total_rows / 2 - 1 = 11 := 
by
  sorry

-- cheburashkas_erased proves that the number of Cheburashkas erased is 11 from the given conditions.

end cheburashkas_erased_l4_4780


namespace factorize_polynomial_l4_4313

theorem factorize_polynomial (x : ℝ) :
  x^4 + 2 * x^3 - 9 * x^2 - 2 * x + 8 = (x + 4) * (x - 2) * (x + 1) * (x - 1) :=
sorry

end factorize_polynomial_l4_4313


namespace simplify_expr_l4_4535

theorem simplify_expr (a : ℝ) (h_a : a = (8:ℝ)^(1/2) * (1/2) - (3:ℝ)^(1/2)^(0) ) : 
  a = (2:ℝ)^(1/2) - 1 := 
by
  sorry

end simplify_expr_l4_4535


namespace sum_of_first_and_last_l4_4752

noncomputable section

variables {A B C D E F G H I : ℕ}

theorem sum_of_first_and_last :
  (D = 8) →
  (A + B + C + D = 50) →
  (B + C + D + E = 50) →
  (C + D + E + F = 50) →
  (D + E + F + G = 50) →
  (E + F + G + H = 50) →
  (F + G + H + I = 50) →
  (A + I = 92) :=
by
  intros hD h1 h2 h3 h4 h5 h6
  sorry

end sum_of_first_and_last_l4_4752


namespace factorization_l4_4685

theorem factorization (a x : ℝ) : ax^2 - 2ax + a = a * (x - 1) ^ 2 := 
by
  sorry

end factorization_l4_4685


namespace no_five_coprime_two_digit_composites_l4_4194

/-- 
  Prove that there do not exist five two-digit composite 
  numbers such that each pair of them is coprime, under 
  the conditions that each composite number must be made 
  up of the primes 2, 3, 5, and 7.
-/
theorem no_five_coprime_two_digit_composites :
  ¬∃ (a b c d e : ℕ),
    10 ≤ a ∧ a < 100 ∧ (∀ p ∈ [2, 3, 5, 7], p ∣ a → p ∣ a) ∧
    10 ≤ b ∧ b < 100 ∧ (∀ p ∈ [2, 3, 5, 7], p ∣ b → p ∣ b) ∧
    10 ≤ c ∧ c < 100 ∧ (∀ p ∈ [2, 3, 5, 7], p ∣ c → p ∣ c) ∧
    10 ≤ d ∧ d < 100 ∧ (∀ p ∈ [2, 3, 5, 7], p ∣ d → p ∣ d) ∧
    10 ≤ e ∧ e < 100 ∧ (∀ p ∈ [2, 3, 5, 7], p ∣ e → p ∣ e) ∧
    ∀ (x y : ℕ), (x ∈ [a, b, c, d, e] ∧ y ∈ [a, b, c, d, e] ∧ x ≠ y) → Nat.gcd x y = 1 :=
by
  sorry

end no_five_coprime_two_digit_composites_l4_4194


namespace triangle_side_length_l4_4539

open Real

/-- Given a triangle ABC with the incircle touching side AB at point D,
where AD = 5 and DB = 3, and given that the angle A is 60 degrees,
prove that the length of side BC is 13. -/
theorem triangle_side_length
  (A B C D : Point)
  (AD DB : ℝ)
  (hAD : AD = 5)
  (hDB : DB = 3)
  (angleA : Real)
  (hangleA : angleA = π / 3) : 
  ∃ BC : ℝ, BC = 13 :=
sorry

end triangle_side_length_l4_4539


namespace initial_total_packs_l4_4355

def initial_packs (total_packs : ℕ) (regular_packs : ℕ) (unusual_packs : ℕ) (excellent_packs : ℕ) : Prop :=
  total_packs = regular_packs + unusual_packs + excellent_packs

def ratio_packs (regular_packs : ℕ) (unusual_packs : ℕ) (excellent_packs : ℕ) : Prop :=
  3 * (regular_packs + unusual_packs + excellent_packs) = 3 * regular_packs + 4 * unusual_packs + 6 * excellent_packs

def new_ratios (regular_packs : ℕ) (unusual_packs : ℕ) (excellent_packs : ℕ) (new_regular_packs : ℕ) (new_unusual_packs : ℕ) (new_excellent_packs : ℕ) : Prop :=
  2 * (new_regular_packs) + 5 * (new_unusual_packs) + 8 * (new_excellent_packs) = regular_packs + unusual_packs + excellent_packs + 8 * (regular_packs)

def pack_changes (initial_regular_packs : ℕ) (initial_unusual_packs : ℕ) (initial_excellent_packs : ℕ) (new_regular_packs : ℕ) (new_unusual_packs : ℕ) (new_excellent_packs : ℕ) : Prop :=
  initial_excellent_packs <= new_excellent_packs + 80 ∧ initial_regular_packs - new_regular_packs ≤ 10

theorem initial_total_packs (total_packs : ℕ) (regular_packs : ℕ) (unusual_packs : ℕ) (excellent_packs : ℕ) 
(new_regular_packs : ℕ) (new_unusual_packs : ℕ) (new_excellent_packs : ℕ) :
  initial_packs total_packs regular_packs unusual_packs excellent_packs ∧
  ratio_packs regular_packs unusual_packs excellent_packs ∧ 
  new_ratios regular_packs unusual_packs excellent_packs new_regular_packs new_unusual_packs new_excellent_packs ∧ 
  pack_changes regular_packs unusual_packs excellent_packs new_regular_packs new_unusual_packs new_excellent_packs 
  → total_packs = 260 := 
sorry

end initial_total_packs_l4_4355


namespace chemistry_textbook_weight_l4_4362

theorem chemistry_textbook_weight (G C : ℝ) 
  (h1 : G = 0.625) 
  (h2 : C = G + 6.5) : 
  C = 7.125 := 
by 
  sorry

end chemistry_textbook_weight_l4_4362


namespace elizabeth_spendings_elizabeth_savings_l4_4894

section WeddingGift

def steak_knife_set_cost : ℝ := 80
def steak_knife_sets : ℕ := 2
def dinnerware_set_cost : ℝ := 200
def fancy_napkins_sets : ℕ := 3
def fancy_napkins_total_cost : ℝ := 45
def wine_glasses_cost : ℝ := 100
def discount_steak_dinnerware : ℝ := 0.10
def discount_napkins : ℝ := 0.20
def sales_tax : ℝ := 0.05

def total_cost_before_discounts : ℝ :=
  (steak_knife_sets * steak_knife_set_cost) + dinnerware_set_cost + fancy_napkins_total_cost + wine_glasses_cost

def total_discount : ℝ :=
  ((steak_knife_sets * steak_knife_set_cost) * discount_steak_dinnerware) + (dinnerware_set_cost * discount_steak_dinnerware) + (fancy_napkins_total_cost * discount_napkins)

def total_cost_after_discounts : ℝ :=
  total_cost_before_discounts - total_discount

def total_cost_with_tax : ℝ :=
  total_cost_after_discounts + (total_cost_after_discounts * sales_tax)

def savings : ℝ :=
  total_cost_before_discounts - total_cost_after_discounts

theorem elizabeth_spendings :
  total_cost_with_tax = 558.60 :=
by sorry

theorem elizabeth_savings :
  savings = 63 :=
by sorry

end WeddingGift

end elizabeth_spendings_elizabeth_savings_l4_4894


namespace intersection_hyperbola_circle_l4_4437

theorem intersection_hyperbola_circle :
  {p : ℝ × ℝ | p.1^2 - 9 * p.2^2 = 36 ∧ p.1^2 + p.2^2 = 36} = {(6, 0), (-6, 0)} :=
by sorry

end intersection_hyperbola_circle_l4_4437


namespace ball_probability_l4_4221

theorem ball_probability (n : ℕ) (h : (n : ℚ) / (n + 2) = 1 / 3) : n = 1 :=
sorry

end ball_probability_l4_4221


namespace appointment_on_tuesday_duration_l4_4165

theorem appointment_on_tuesday_duration :
  let rate := 20
  let monday_appointments := 5
  let monday_each_duration := 1.5
  let thursday_appointments := 2
  let thursday_each_duration := 2
  let saturday_duration := 6
  let weekly_earnings := 410
  let known_earnings := (monday_appointments * monday_each_duration * rate) + (thursday_appointments * thursday_each_duration * rate) + (saturday_duration * rate)
  let tuesday_earnings := weekly_earnings - known_earnings
  (tuesday_earnings / rate = 3) :=
by
  -- let rate := 20
  -- let monday_appointments := 5
  -- let monday_each_duration := 1.5
  -- let thursday_appointments := 2
  -- let thursday_each_duration := 2
  -- let saturday_duration := 6
  -- let weekly_earnings := 410
  -- let known_earnings := (monday_appointments * monday_each_duration * rate) + (thursday_appointments * thursday_each_duration * rate) + (saturday_duration * rate)
  -- let tuesday_earnings := weekly_earnings - known_earnings
  -- exact tuesday_earnings / rate = 3
  sorry

end appointment_on_tuesday_duration_l4_4165


namespace solve_for_x_l4_4099

theorem solve_for_x (x : ℚ) (h : x + 3 * x = 300 - (4 * x + 5 * x)) : x = 300 / 13 :=
by
  sorry

end solve_for_x_l4_4099


namespace number_of_Cheburashkas_erased_l4_4776

theorem number_of_Cheburashkas_erased :
  ∃ (n : ℕ), 
    (∀ x, x ≥ 1 → 
      (let totalKrakozyabras = (2 * (x - 1) = 29) in
         x - 2 = 11)) :=
sorry

end number_of_Cheburashkas_erased_l4_4776


namespace smallest_three_digit_multiple_of_three_with_odd_hundreds_l4_4861

theorem smallest_three_digit_multiple_of_three_with_odd_hundreds :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧ a % 2 = 1 ∧ n % 3 = 0 ∧ n = 102) :=
by
  sorry

end smallest_three_digit_multiple_of_three_with_odd_hundreds_l4_4861


namespace smallest_c_for_inverse_l4_4787

noncomputable def g (x : ℝ) : ℝ := (x + 3)^2 - 6

theorem smallest_c_for_inverse : 
  ∃ (c : ℝ), (∀ x1 x2, x1 ≥ c → x2 ≥ c → g x1 = g x2 → x1 = x2) ∧ 
            (∀ c', c' < c → ∃ x1 x2, x1 ≥ c' → x2 ≥ c' → g x1 = g x2 ∧ x1 ≠ x2) ∧ 
            c = -3 :=
by 
  sorry

end smallest_c_for_inverse_l4_4787


namespace amy_balloons_l4_4359

theorem amy_balloons (james_balloons amy_balloons : ℕ) (h1 : james_balloons = 1222) (h2 : james_balloons = amy_balloons + 709) : amy_balloons = 513 :=
by
  sorry

end amy_balloons_l4_4359


namespace factorize_expression_l4_4693

theorem factorize_expression (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 :=
by sorry

end factorize_expression_l4_4693


namespace Cheryl_total_material_used_l4_4394

theorem Cheryl_total_material_used :
  let material1 := (5 : ℚ) / 11
  let material2 := (2 : ℚ) / 3
  let total_purchased := material1 + material2
  let material_left := (25 : ℚ) / 55
  let material_used := total_purchased - material_left
  material_used = 22 / 33 := by
  sorry

end Cheryl_total_material_used_l4_4394


namespace average_time_in_storm_l4_4146

-- Define conditions for the problem.
def car_position (t : ℝ) : ℝ × ℝ := (2 / 3 * t, 0)
def storm_position (t : ℝ) : ℝ × ℝ := (t, 130 - t)

-- Distance formula condition to be within the storm radius
def within_storm (t : ℝ) : Prop :=
  Real.sqrt ((2 / 3 * t - t)^2 + (130 - t)^2) ≤ 60

-- Define the times when the car enters and leaves the storm
def times_in_storm : set ℝ := { t | within_storm t }

-- The proof goal
theorem average_time_in_storm : 
  let times := {t | within_storm t},
      t1 := times_inf times,
      t2 := times_sup times
  in 
    (t1 + t2) / 2 = 117 :=
sorry

end average_time_in_storm_l4_4146


namespace PG_entitled_amount_l4_4442

-- Definitions and conditions provided in the problem
def initial_investment_VP : ℕ := 200_000
def initial_investment_PG : ℕ := 350_000
def AA_purchase_amount : ℕ := 1_100_000
def factory_value_after_sale : ℕ := 3_300_000
def share_value_each_after_sale : ℕ := factory_value_after_sale / 3
def VP_initial_share_value := (initial_investment_VP * factory_value_after_sale) / (initial_investment_VP + initial_investment_PG)
def VP_sold_share_value := VP_initial_share_value - share_value_each_after_sale
def PG_received_amount := AA_purchase_amount - VP_sold_share_value

-- Theorem to be proven
theorem PG_entitled_amount : PG_received_amount = 1_000_000 := by
  sorry

end PG_entitled_amount_l4_4442


namespace exists_zero_in_interval_l4_4918

noncomputable def f (x : ℝ) : ℝ := 6 / x - x^2

theorem exists_zero_in_interval : ∃ c ∈ Ioo (1 : ℝ) 2, f c = 0 := by
  sorry

end exists_zero_in_interval_l4_4918


namespace fraction_difference_l4_4339

theorem fraction_difference (x y : ℝ) (h : x - y = 3 * x * y) : (1 / x) - (1 / y) = -3 :=
by
  sorry

end fraction_difference_l4_4339


namespace radius_of_O2_l4_4063

theorem radius_of_O2 (r_O1 r_dist r_O2 : ℝ) 
  (h1 : r_O1 = 3) 
  (h2 : r_dist = 7) 
  (h3 : (r_dist = r_O1 + r_O2 ∨ r_dist = |r_O2 - r_O1|)) :
  r_O2 = 4 ∨ r_O2 = 10 :=
by
  sorry

end radius_of_O2_l4_4063


namespace max_stamps_l4_4469

theorem max_stamps (price_per_stamp : ℕ) (total_cents : ℕ) (h1 : price_per_stamp = 45) (h2 : total_cents = 5000) : 
  ∃ n : ℕ, n ≤ total_cents / price_per_stamp ∧ n = 111 :=
by
  sorry

end max_stamps_l4_4469


namespace sum_of_squares_of_consecutive_integers_l4_4513

theorem sum_of_squares_of_consecutive_integers
  (a : ℤ) (h : (a - 1) * a * (a + 1) = 10 * ((a - 1) + a + (a + 1))) :
  (a - 1)^2 + a^2 + (a + 1)^2 = 110 :=
sorry

end sum_of_squares_of_consecutive_integers_l4_4513


namespace A_days_to_complete_alone_l4_4538

theorem A_days_to_complete_alone
  (work_left : ℝ := 0.41666666666666663)
  (B_days : ℝ := 20)
  (combined_days : ℝ := 5)
  : ∃ (A_days : ℝ), A_days = 15 := 
by
  sorry

end A_days_to_complete_alone_l4_4538


namespace negation_of_p_l4_4919

-- Define the original predicate
def p (x₀ : ℝ) : Prop := x₀^2 > 1

-- Define the negation of the predicate
def not_p : Prop := ∀ x : ℝ, x^2 ≤ 1

-- Prove the negation of the proposition
theorem negation_of_p : (∃ x₀ : ℝ, p x₀) ↔ not_p := by
  sorry

end negation_of_p_l4_4919


namespace total_distance_l4_4875

theorem total_distance (D : ℝ) (h_walk : ∀ d t, d = 4 * t) 
                       (h_run : ∀ d t, d = 8 * t) 
                       (h_time : ∀ t_walk t_run, t_walk + t_run = 0.75) 
                       (h_half : D / 2 = d_walk ∧ D / 2 = d_run) :
                       D = 8 := 
by
  sorry

end total_distance_l4_4875


namespace distance_from_neg2_eq4_l4_4091

theorem distance_from_neg2_eq4 (x : ℤ) : |x + 2| = 4 ↔ x = 2 ∨ x = -6 :=
by
  sorry

end distance_from_neg2_eq4_l4_4091


namespace find_a_l4_4482

noncomputable def lines_perpendicular (a : ℝ) (l1: ℝ × ℝ × ℝ) (l2: ℝ × ℝ × ℝ) : Prop :=
  let (A1, B1, C1) := l1
  let (A2, B2, C2) := l2
  (B1 ≠ 0) ∧ (B2 ≠ 0) ∧ (-A1 / B1) * (-A2 / B2) = -1

theorem find_a (a : ℝ) :
  lines_perpendicular a (a, 1, 1) (2*a, a - 3, 1) → a = 1 ∨ a = -3/2 :=
by
  sorry

end find_a_l4_4482


namespace range_of_m_l4_4328

noncomputable def proposition_p (m : ℝ) : Prop :=
∀ x : ℝ, x^2 + m * x + 1 ≥ 0

noncomputable def proposition_q (m : ℝ) : Prop :=
∀ x : ℝ, (8 * x + 4 * (m - 1)) ≥ 0

def conditions (m : ℝ) : Prop :=
(proposition_p m ∨ proposition_q m) ∧ ¬(proposition_p m ∧ proposition_q m)

theorem range_of_m (m : ℝ) : 
  conditions m → ( -2 ≤ m ∧ m < 1 ) ∨ m > 2 :=
by
  intros h
  sorry

end range_of_m_l4_4328


namespace wendy_first_album_pictures_l4_4990

theorem wendy_first_album_pictures 
  (total_pictures : ℕ)
  (num_albums : ℕ)
  (pics_per_album : ℕ)
  (pics_in_first_album : ℕ)
  (h1 : total_pictures = 79)
  (h2 : num_albums = 5)
  (h3 : pics_per_album = 7)
  (h4 : total_pictures = pics_in_first_album + num_albums * pics_per_album) : 
  pics_in_first_album = 44 :=
by
  sorry

end wendy_first_album_pictures_l4_4990


namespace friends_count_l4_4609

def bananas_total : ℝ := 63
def bananas_per_friend : ℝ := 21.0

theorem friends_count : bananas_total / bananas_per_friend = 3 := sorry

end friends_count_l4_4609


namespace parallel_lines_slopes_l4_4209

theorem parallel_lines_slopes (k : ℝ) :
  (∀ x y : ℝ, x + (1 + k) * y = 2 - k → k * x + 2 * y + 8 = 0 → k = 1) :=
by
  intro h1 h2
  -- We can see that there should be specifics here about how the conditions lead to k = 1
  sorry

end parallel_lines_slopes_l4_4209


namespace find_correct_day_l4_4492

def tomorrow_is_not_September (d : String) : Prop :=
  d ≠ "September"

def in_a_week_is_September (d : String) : Prop :=
  d = "September"

def day_after_tomorrow_is_not_Wednesday (d : String) : Prop :=
  d ≠ "Wednesday"

theorem find_correct_day :
    ((∀ d, tomorrow_is_not_September d) ∧ 
    (∀ d, in_a_week_is_September d) ∧ 
    (∀ d, day_after_tomorrow_is_not_Wednesday d)) → 
    "Wednesday, August 25" = "Wednesday, August 25" :=
by
sorry

end find_correct_day_l4_4492


namespace proof_abc_identity_l4_4239

variable {a b c : ℝ}

theorem proof_abc_identity
  (h_ne_a : a ≠ 1) (h_ne_na : a ≠ -1)
  (h_ne_b : b ≠ 1) (h_ne_nb : b ≠ -1)
  (h_ne_c : c ≠ 1) (h_ne_nc : c ≠ -1)
  (habc : a * b + b * c + c * a = 1) :
  a / (1 - a ^ 2) + b / (1 - b ^ 2) + c / (1 - c ^ 2) = (4 * a * b * c) / (1 - a ^ 2) / (1 - b ^ 2) / (1 - c ^ 2) :=
by 
  sorry

end proof_abc_identity_l4_4239


namespace triangle_semicircle_l4_4306

noncomputable def triangle_semicircle_ratio : ℝ :=
  let AB := 8
  let BC := 6
  let CA := 2 * Real.sqrt 7
  let radius_AB := AB / 2
  let radius_BC := BC / 2
  let radius_CA := CA / 2
  let area_semicircle_AB := (1 / 2) * Real.pi * radius_AB ^ 2
  let area_semicircle_BC := (1 / 2) * Real.pi * radius_BC ^ 2
  let area_semicircle_CA := (1 / 2) * Real.pi * radius_CA ^ 2
  let area_triangle := AB * BC / 2
  let total_shaded_area := (area_semicircle_AB + area_semicircle_BC + area_semicircle_CA) - area_triangle
  let area_circle_CA := Real.pi * (radius_CA ^ 2)
  total_shaded_area / area_circle_CA

theorem triangle_semicircle : triangle_semicircle_ratio = 2 - (12 * Real.sqrt 3) / (7 * Real.pi) := by
  sorry

end triangle_semicircle_l4_4306


namespace petr_receives_1000000_l4_4441

def initial_investment_vp := 200000
def initial_investment_pg := 350000
def third_share_value := 1100000
def total_company_value := 3 * third_share_value

theorem petr_receives_1000000 :
  initial_investment_vp = 200000 →
  initial_investment_pg = 350000 →
  third_share_value = 1100000 →
  total_company_value = 3300000 →
  ∃ (share_pg : ℕ), share_pg = 1000000 :=
by
  intros h_vp h_pg h_as h_total
  let x := initial_investment_vp * 1650000
  let y := initial_investment_pg * 1650000
  -- Skipping calculations
  sorry

end petr_receives_1000000_l4_4441


namespace rabbit_stashed_nuts_l4_4745

theorem rabbit_stashed_nuts :
  ∃ r: ℕ, 
  ∃ f: ℕ, 
  4 * r = 6 * f ∧ f = r - 5 ∧ 4 * r = 60 :=
by {
  sorry
}

end rabbit_stashed_nuts_l4_4745


namespace solve_complex_problem_l4_4323

-- Define the problem
def complex_sum_eq_two (a b : ℝ) (i : ℂ) : Prop :=
  a + b = 2

-- Define the conditions
def conditions (a b : ℝ) (i : ℂ) : Prop :=
  a + b * i = (1 - i) * (2 + i)

-- State the theorem
theorem solve_complex_problem (a b : ℝ) (i : ℂ) (h : conditions a b i) : complex_sum_eq_two a b i :=
by
  sorry -- Proof goes here

end solve_complex_problem_l4_4323


namespace find_A_l4_4722

theorem find_A (A a b : ℝ) (h1 : 3^a = A) (h2 : 5^b = A) (h3 : 1/a + 1/b = 2) : A = Real.sqrt 15 :=
by
  /- Proof omitted -/
  sorry

end find_A_l4_4722


namespace total_number_of_players_l4_4225

theorem total_number_of_players (n : ℕ) (h1 : n > 7) 
  (h2 : (4 * (n * (n - 1)) / 3 + 56 = (n + 8) * (n + 7) / 2)) : n + 8 = 50 :=
by
  sorry

end total_number_of_players_l4_4225


namespace x_squared_plus_y_squared_l4_4071

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 12) : x^2 + y^2 = 25 := by
  sorry

end x_squared_plus_y_squared_l4_4071


namespace range_of_m_l4_4059

theorem range_of_m 
  (m : ℝ)
  (f : ℝ → ℝ)
  (f_def : ∀ x, f x = x^3 + (m / 2 + 2) * x^2 - 2 * x)
  (f_prime : ℝ → ℝ)
  (f_prime_def : ∀ x, f_prime x = 3 * x^2 + (m + 4) * x - 2)
  (f_prime_at_1 : f_prime 1 < 0)
  (f_prime_at_2 : f_prime 2 < 0)
  (f_prime_at_3 : f_prime 3 > 0) :
  -37 / 3 < m ∧ m < -9 := 
  sorry

end range_of_m_l4_4059


namespace triangle_area_30_l4_4831

theorem triangle_area_30 (h : ∃ a b c : ℝ, a^2 + b^2 = c^2 ∧ a = 5 ∧ c = 13 ∧ b > 0) : 
  ∃ area : ℝ, area = 1 / 2 * 5 * (b : ℝ) ∧ area = 30 :=
by
  sorry

end triangle_area_30_l4_4831


namespace product_of_roots_of_cubic_eqn_l4_4187

theorem product_of_roots_of_cubic_eqn :
  let p := Polynomial.Cubic (1 : ℝ) (-15) 75 (-50)
  in p.roots_prod = 50 :=
by
  sorry

end product_of_roots_of_cubic_eqn_l4_4187


namespace arithmetic_sequence_S11_l4_4952

open ArithmeticSequence

theorem arithmetic_sequence_S11 (S : ℕ → ℕ) (a : ℕ → ℕ) (d : ℕ) :
  S 8 - S 3 = 20 →
  (∀ n, a (n + 1) = a n + d) →
  S = λ n, n * (a 1 + a n) / 2 →
  S 11 = 44 :=
by
  intros h1 h2 h3
  sorry

end arithmetic_sequence_S11_l4_4952


namespace cactus_species_minimum_l4_4676

theorem cactus_species_minimum :
  ∀ (collections : Fin 80 → Fin k → Prop),
  (∀ s : Fin k, ∃ (i : Fin 80), ¬ collections i s)
  → (∀ (c : Finset (Fin 80)), c.card = 15 → ∃ s : Fin k, ∀ (i : Fin 80), i ∈ c → collections i s)
  → 16 ≤ k := 
by 
  sorry

end cactus_species_minimum_l4_4676


namespace not_perfect_square_l4_4240

theorem not_perfect_square (p : ℕ) (hp : Nat.Prime p) : ¬ ∃ t : ℕ, 7 * p + 3^p - 4 = t^2 :=
sorry

end not_perfect_square_l4_4240


namespace bicycle_discount_l4_4511

theorem bicycle_discount (original_price : ℝ) (discount : ℝ) (discounted_price : ℝ) :
  original_price = 760 ∧ discount = 0.75 ∧ discounted_price = 570 → 
  original_price * discount = discounted_price := by
  sorry

end bicycle_discount_l4_4511


namespace system_equivalence_l4_4624

theorem system_equivalence (f g : ℝ → ℝ) (x : ℝ) (h1 : f x > 0) (h2 : g x > 0) : f x + g x > 0 :=
sorry

end system_equivalence_l4_4624


namespace b_2056_l4_4478

noncomputable def b (n : ℕ) : ℝ := sorry

-- Conditions
axiom h1 : b 1 = 2 + Real.sqrt 8
axiom h2 : b 2023 = 15 + Real.sqrt 8
axiom recurrence : ∀ n, n ≥ 2 → b n = b (n - 1) * b (n + 1)

-- Problem statement to prove
theorem b_2056 : b 2056 = (2 + Real.sqrt 8)^2 / (15 + Real.sqrt 8) :=
sorry

end b_2056_l4_4478


namespace speed_of_first_car_l4_4137

variable (V1 V2 V3 : ℝ) -- Define the speeds of the three cars
variable (t x : ℝ) -- Time interval and distance from A to B

-- Conditions of the problem
axiom condition_1 : x / V1 = (x / V2) + t
axiom condition_2 : x / V2 = (x / V3) + t
axiom condition_3 : 120 / V1  = (120 / V2) + 1
axiom condition_4 : 40 / V1 = 80 / V3

-- Proof statement
theorem speed_of_first_car : V1 = 30 := by
  sorry

end speed_of_first_car_l4_4137


namespace solution_set_of_inequality_l4_4515

theorem solution_set_of_inequality :
  {x : ℝ | |2 * x + 1| > 3} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 1} :=
by sorry

end solution_set_of_inequality_l4_4515


namespace seeds_total_l4_4527

-- Define the conditions as given in the problem.
def Bom_seeds : ℕ := 300
def Gwi_seeds : ℕ := Bom_seeds + 40
def Yeon_seeds : ℕ := 3 * Gwi_seeds

-- Lean statement to prove the total number of seeds.
theorem seeds_total : Bom_seeds + Gwi_seeds + Yeon_seeds = 1660 := 
by
  -- Assuming all given definitions and conditions are true,
  -- we aim to prove the final theorem statement.
  sorry

end seeds_total_l4_4527


namespace quadratic_equation_general_form_l4_4191

theorem quadratic_equation_general_form :
  ∀ (x : ℝ), 3 * x^2 + 1 = 7 * x ↔ 3 * x^2 - 7 * x + 1 = 0 :=
by
  intro x
  constructor
  · intro h
    sorry
  · intro h
    sorry

end quadratic_equation_general_form_l4_4191


namespace count_two_digit_integers_remainder_3_div_9_l4_4924

theorem count_two_digit_integers_remainder_3_div_9 :
  ∃ (N : ℕ), N = 10 ∧ ∀ k, (10 ≤ 9 * k + 3 ∧ 9 * k + 3 < 100) ↔ (1 ≤ k ∧ k ≤ 10) :=
by
  sorry

end count_two_digit_integers_remainder_3_div_9_l4_4924


namespace smallest_and_largest_x_l4_4717

theorem smallest_and_largest_x (x : ℝ) :
  (|5 * x - 4| = 29) → ((x = -5) ∨ (x = 6.6)) :=
by
  sorry

end smallest_and_largest_x_l4_4717


namespace factorization_property_l4_4251

theorem factorization_property (a b : ℤ) (h1 : 25 * x ^ 2 - 160 * x - 144 = (5 * x + a) * (5 * x + b)) 
    (h2 : a + b = -32) (h3 : a * b = -144) : 
    a + 2 * b = -68 := 
sorry

end factorization_property_l4_4251


namespace trigonometric_identity_l4_4448

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 1 / 2) :
  Real.sin (2 * θ) - 2 * Real.cos θ ^ 2 = -4 / 5 :=
by
  sorry

end trigonometric_identity_l4_4448


namespace admission_schemes_count_l4_4987

theorem admission_schemes_count :
  let students : Finset ℕ := {1, 2, 3, 4}
  let universities : Finset ℕ := {1, 2, 3}
  (∀ u ∈ universities, (∃ s ⊆ students, s.card ≥ 1)) →
  (Finset.card (students.powerset.filter (λ s, s.card = 2)) * 3.factorial) = 36 :=
sorry

end admission_schemes_count_l4_4987


namespace soda_cost_90_cents_l4_4376

theorem soda_cost_90_cents
  (b s : ℕ)
  (h1 : 3 * b + 2 * s = 360)
  (h2 : 2 * b + 4 * s = 480) :
  s = 90 :=
by
  sorry

end soda_cost_90_cents_l4_4376


namespace dimension_proof_l4_4153

noncomputable def sports_field_dimensions (x y: ℝ) : Prop :=
  -- Given conditions
  x^2 + y^2 = 185^2 ∧
  (x - 4) * (y - 4) = x * y - 1012 ∧
  -- Seeking to prove dimensions
  ((x = 153 ∧ y = 104) ∨ (x = 104 ∧ y = 153))

theorem dimension_proof : ∃ x y: ℝ, sports_field_dimensions x y := by
  sorry

end dimension_proof_l4_4153


namespace factorization_l4_4690

theorem factorization (a x : ℝ) : ax^2 - 2ax + a = a * (x - 1) ^ 2 := 
by
  sorry

end factorization_l4_4690


namespace average_of_first_45_results_l4_4247

theorem average_of_first_45_results
  (A : ℝ)
  (h1 : (45 + 25 : ℝ) = 70)
  (h2 : (25 : ℝ) * 45 = 1125)
  (h3 : (70 : ℝ) * 32.142857142857146 = 2250)
  (h4 : ∀ x y z : ℝ, 45 * x + y = z → x = 25) :
  A = 25 :=
by
  sorry

end average_of_first_45_results_l4_4247


namespace tan_half_angle_product_l4_4214

theorem tan_half_angle_product (a b : ℝ) 
  (h : 7 * (Real.cos a + Real.cos b) + 6 * (Real.cos a * Real.cos b + 1) = 0) :
  ∃ x : ℝ, x = Real.tan (a / 2) * Real.tan (b / 2) ∧ (x = Real.sqrt (26 / 7) ∨ x = -Real.sqrt (26 / 7)) :=
by
  sorry

end tan_half_angle_product_l4_4214


namespace relation_x_lt_1_and_x_sq_sub_4x_add_3_gt_0_sufficiency_x_lt_1_necessity_x_lt_1_l4_4657

theorem relation_x_lt_1_and_x_sq_sub_4x_add_3_gt_0 (x : ℝ) :
  (x < 1) → (x^2 - 4 * x + 3 > 0) :=
by sorry

-- Define the sufficiency part
theorem sufficiency_x_lt_1 (x : ℝ) :
  (x < 1) → (x^2 - 4 * x + 3 > 0) :=
by sorry

-- Define the necessity part
theorem necessity_x_lt_1 (x : ℝ) :
  (x^2 - 4 * x + 3 > 0) → (x < 1 ∨ x > 3) :=
by sorry

end relation_x_lt_1_and_x_sq_sub_4x_add_3_gt_0_sufficiency_x_lt_1_necessity_x_lt_1_l4_4657


namespace max_value_sqrt_add_l4_4785

noncomputable def sqrt_add (a b : ℝ) : ℝ := Real.sqrt (a + 1) + Real.sqrt (b + 3)

theorem max_value_sqrt_add (a b : ℝ) (h : 0 < a) (h' : 0 < b) (hab : a + b = 5) :
  sqrt_add a b ≤ 3 * Real.sqrt 2 :=
by
  sorry

end max_value_sqrt_add_l4_4785


namespace distance_symmetric_reflection_l4_4749

theorem distance_symmetric_reflection (x : ℝ) (y : ℝ) (B : (ℝ × ℝ)) 
  (hB : B = (-1, 4)) (A : (ℝ × ℝ)) (hA : A = (x, -y)) : 
  dist A B = 8 :=
by
  sorry

end distance_symmetric_reflection_l4_4749


namespace trajectory_of_A_l4_4207

def B : ℝ × ℝ := (-5, 0)
def C : ℝ × ℝ := (5, 0)

def sin_B : ℝ := sorry
def sin_C : ℝ := sorry
def sin_A : ℝ := sorry

axiom sin_relation : sin_B - sin_C = (3/5) * sin_A

theorem trajectory_of_A :
  ∃ x y : ℝ, (x^2 / 9) - (y^2 / 16) = 1 ∧ x < -3 :=
sorry

end trajectory_of_A_l4_4207


namespace set_elements_l4_4648

def is_divisor (a b : ℤ) : Prop := ∃ k : ℤ, b = k * a

theorem set_elements:
  {x : ℤ | ∃ d : ℤ, is_divisor d 12 ∧ d = 6 - x ∧ x ≥ 0} = 
  {0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 18} :=
by {
  sorry
}

end set_elements_l4_4648


namespace product_of_roots_l4_4190

theorem product_of_roots :
  let f := Polynomial.C (-50) + Polynomial.X * (Polynomial.C 75 + Polynomial.X * (Polynomial.C (-15) + Polynomial.X)) in 
  (f.roots.Prod = 50) :=
by sorry

end product_of_roots_l4_4190


namespace product_of_D_l4_4054

theorem product_of_D:
  ∀ (D : ℝ × ℝ), 
  (∃ M C : ℝ × ℝ, 
    M.1 = 4 ∧ M.2 = 3 ∧ 
    C.1 = 6 ∧ C.2 = -1 ∧ 
    M.1 = (C.1 + D.1) / 2 ∧ 
    M.2 = (C.2 + D.2) / 2) 
  → (D.1 * D.2 = 14) :=
sorry

end product_of_D_l4_4054


namespace intersection_with_y_axis_l4_4103

theorem intersection_with_y_axis (y : ℝ) : 
  (∃ y, (0, y) ∈ {(x, 2 * x + 4) | x : ℝ}) ↔ y = 4 :=
by 
  sorry

end intersection_with_y_axis_l4_4103


namespace lcm_18_24_30_eq_360_l4_4523

-- Define the three numbers in the condition
def a : ℕ := 18
def b : ℕ := 24
def c : ℕ := 30

-- State the theorem to prove
theorem lcm_18_24_30_eq_360 : Nat.lcm a (Nat.lcm b c) = 360 :=
by 
  sorry -- Proof is omitted as per instructions

end lcm_18_24_30_eq_360_l4_4523


namespace cubic_km_to_cubic_m_l4_4066

theorem cubic_km_to_cubic_m (km_to_m : 1 = 1000) : (1 : ℝ) ^ 3 = (1000 : ℝ) ^ 3 :=
by sorry

end cubic_km_to_cubic_m_l4_4066


namespace range_of_a_l4_4956

variable (a b c : ℝ)

def condition1 := a^2 - b * c - 8 * a + 7 = 0

def condition2 := b^2 + c^2 + b * c - 6 * a + 6 = 0

theorem range_of_a (h1 : condition1 a b c) (h2 : condition2 a b c) : 1 ≤ a ∧ a ≤ 9 := 
  sorry

end range_of_a_l4_4956


namespace largest_possible_value_l4_4504

theorem largest_possible_value (X Y Z m: ℕ) 
  (hX_range: 0 ≤ X ∧ X ≤ 4) 
  (hY_range: 0 ≤ Y ∧ Y ≤ 4) 
  (hZ_range: 0 ≤ Z ∧ Z ≤ 4) 
  (h1: m = 25 * X + 5 * Y + Z)
  (h2: m = 81 * Z + 9 * Y + X):
  m = 121 :=
by
  -- The proof goes here
  sorry

end largest_possible_value_l4_4504


namespace range_of_x_l4_4056

theorem range_of_x (x : ℝ) (h : |2 * x + 1| + |2 * x - 5| = 6) : -1 / 2 ≤ x ∧ x ≤ 5 / 2 := by
  sorry

end range_of_x_l4_4056


namespace infinite_coprime_pairs_divisibility_l4_4965

theorem infinite_coprime_pairs_divisibility :
  ∃ (S : ℕ → ℕ × ℕ), (∀ n, Nat.gcd (S n).1 (S n).2 = 1 ∧ (S n).1 ∣ (S n).2^2 - 5 ∧ (S n).2 ∣ (S n).1^2 - 5) ∧
  Function.Injective S :=
sorry

end infinite_coprime_pairs_divisibility_l4_4965


namespace complex_magnitude_problem_l4_4374

open Complex

theorem complex_magnitude_problem (z w : ℂ) (hz : Complex.abs z = 2) (hw : Complex.abs w = 4) (hzw : Complex.abs (z + w) = 3) :
  Complex.abs (1 / z + 1 / w) = 3 / 8 := 
by
  sorry

end complex_magnitude_problem_l4_4374


namespace surface_area_of_cube_is_correct_l4_4531

noncomputable def edge_length (a : ℝ) : ℝ := 5 * a

noncomputable def surface_area_of_cube (a : ℝ) : ℝ :=
  let edge := edge_length a
  6 * edge * edge

theorem surface_area_of_cube_is_correct (a : ℝ) :
  surface_area_of_cube a = 150 * a ^ 2 := by
  sorry

end surface_area_of_cube_is_correct_l4_4531


namespace common_root_poly_identity_l4_4865

theorem common_root_poly_identity
  (α p p' q q' : ℝ)
  (h1 : α^3 + p*α + q = 0)
  (h2 : α^3 + p'*α + q' = 0) : 
  (p * q' - q * p') * (p - p')^2 = (q - q')^3 := 
by
  sorry

end common_root_poly_identity_l4_4865


namespace math_problem_l4_4908

theorem math_problem (a b c d x y : ℝ) (h1 : a = -b) (h2 : c * d = 1) 
  (h3 : (x + 3)^2 + |y - 2| = 0) : 2 * (a + b) - 2 * (c * d)^4 + (x + y)^2022 = -1 :=
by
  sorry

end math_problem_l4_4908


namespace principal_argument_of_z_l4_4047

-- Mathematical definitions based on provided conditions
noncomputable def theta : ℝ := Real.arctan (5 / 12)

-- The complex number z defined in the problem
noncomputable def z : ℂ := (Real.cos (2 * theta) + Real.sin (2 * theta) * Complex.I) / (239 + Complex.I)

-- Lean statement to prove the argument of z
theorem principal_argument_of_z : Complex.arg z = Real.pi / 4 :=
by
  sorry

end principal_argument_of_z_l4_4047


namespace total_amount_given_away_l4_4944

variable (numGrandchildren : ℕ)
variable (cardsPerGrandchild : ℕ)
variable (amountPerCard : ℕ)

theorem total_amount_given_away (h1 : numGrandchildren = 3) (h2 : cardsPerGrandchild = 2) (h3 : amountPerCard = 80) : 
  numGrandchildren * cardsPerGrandchild * amountPerCard = 480 := by
  sorry

end total_amount_given_away_l4_4944


namespace largest_b_l4_4984

def max_b (a b c : ℕ) : ℕ := b -- Define max_b function which outputs b

theorem largest_b (a b c : ℕ)
  (h1 : a * b * c = 360)
  (h2 : 1 < c)
  (h3 : c < b)
  (h4 : b < a) :
  max_b a b c = 10 :=
sorry

end largest_b_l4_4984


namespace tomatoes_picked_l4_4290

theorem tomatoes_picked (original_tomatoes left_tomatoes picked_tomatoes : ℕ)
  (h1 : original_tomatoes = 97)
  (h2 : left_tomatoes = 14)
  (h3 : picked_tomatoes = original_tomatoes - left_tomatoes) :
  picked_tomatoes = 83 :=
by sorry

end tomatoes_picked_l4_4290


namespace value_of_S_2016_l4_4349

variable (a d : ℤ)
variable (S : ℕ → ℤ)

-- Definitions of conditions
def a_1 := -2014
def sum_2012 := S 2012
def sum_10 := S 10
def S_n (n : ℕ) : ℤ := n * a_1 + (n * (n - 1) / 2) * d

-- Given conditions
axiom S_condition : (sum_2012 / 2012) - (sum_10 / 10) = 2002
axiom S_def : ∀ n : ℕ, S n = S_n n

-- The theorem to be proved
theorem value_of_S_2016 : S 2016 = 2016 := by
  sorry

end value_of_S_2016_l4_4349


namespace fixed_point_f_l4_4821

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.log (2 * x + 1) / Real.log a) + 2

theorem fixed_point_f (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : f a 0 = 2 :=
by
  sorry

end fixed_point_f_l4_4821


namespace figure_perimeter_l4_4500

theorem figure_perimeter (h_segments v_segments : ℕ) (side_length : ℕ) 
  (h_count : h_segments = 16) (v_count : v_segments = 10) (side_len : side_length = 1) :
  2 * (h_segments + v_segments) * side_length = 26 :=
by
  sorry

end figure_perimeter_l4_4500


namespace Jeff_wins_three_games_l4_4759

-- Define the conditions and proven statement
theorem Jeff_wins_three_games :
  (hours_played : ℕ) (minutes_per_point : ℕ) (points_per_match : ℕ) 
  (hours_played = 2) (minutes_per_point = 5) (points_per_match = 8) 
  → (games_won : ℕ) (120 / minutes_per_point / points_per_match = 3) :=
by
  -- Step through assumptions and automatically conclude the proof
  sorry

end Jeff_wins_three_games_l4_4759


namespace min_value_polynomial_l4_4601

theorem min_value_polynomial (a b : ℝ) : 
  ∃ c, (∀ a b, c ≤ a^2 + 2 * b^2 + 2 * a + 4 * b + 2008) ∧
       (∀ a b, a = -1 ∧ b = -1 → c = a^2 + 2 * b^2 + 2 * a + 4 * b + 2008) :=
sorry

end min_value_polynomial_l4_4601


namespace AC_is_7_07_l4_4626

namespace Mathlib

noncomputable def AC_length (AD BC: ℝ) (BAC_deg ADB_deg: ℝ) : ℝ :=
  let BAC := BAC_deg * real.pi / 180
  let ADB := ADB_deg * real.pi / 180
  let ABC := real.pi - (BAC + ADB)
  real.sqrt (AD^2 + BC^2 - 2 * AD * BC * real.cos ABC)

theorem AC_is_7_07
  (AD BC : ℕ) (BAC_deg ADB_deg : ℕ)
  (hAD : AD = 5)
  (hBC : BC = 7)
  (hBAC : BAC_deg = 60)
  (hADB : ADB_deg = 50) :
  AC_length AD BC BAC_deg ADB_deg = real.sqrt 50.06 :=
by
  rw [hAD, hBC, hBAC, hADB]
  norm_num
  sorry

end Mathlib

end AC_is_7_07_l4_4626


namespace infinitely_many_a_not_sum_of_seven_sixth_powers_l4_4498

theorem infinitely_many_a_not_sum_of_seven_sixth_powers :
  ∃ᶠ (a: ℕ) in at_top, (∀ (a_i : ℕ) (h0 : a_i > 0), a ≠ a_i^6 + a_i^6 + a_i^6 + a_i^6 + a_i^6 + a_i^6 + a_i^6 ∧ a % 9 = 8) :=
sorry

end infinitely_many_a_not_sum_of_seven_sixth_powers_l4_4498


namespace problem_statement_l4_4505
noncomputable def f (M : ℝ) (x : ℝ) : ℝ := M * Real.sin (2 * x + Real.pi / 6)
def is_symmetric (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a - x) = f (a + x)
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x
def is_center_of_symmetry (f : ℝ → ℝ) (c : ℝ × ℝ) : Prop := ∀ x, f (2 * c.1 - x) = 2 * c.2 - f x

theorem problem_statement (M : ℝ) (hM : M ≠ 0) : 
    is_symmetric (f M) (2 * Real.pi / 3) ∧ 
    is_periodic (f M) Real.pi ∧ 
    is_center_of_symmetry (f M) (5 * Real.pi / 12, 0) :=
by
  sorry

end problem_statement_l4_4505


namespace inequality_proof_l4_4052

theorem inequality_proof (a b c x y z : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) (h4 : x ≥ y) (h5 : y ≥ z) (h6 : z > 0) :
  (a^2 * x^2 / ((b * y + c * z) * (b * z + c * y)) + 
   b^2 * y^2 / ((a * x + c * z) * (a * z + c * x)) +
   c^2 * z^2 / ((a * x + b * y) * (a * y + b * x))) ≥ 3 / 4 := 
by
  sorry

end inequality_proof_l4_4052


namespace sufficient_but_not_necessary_for_reciprocal_l4_4138

theorem sufficient_but_not_necessary_for_reciprocal (x : ℝ) : (x > 1 → 1/x < 1) ∧ (¬ (1/x < 1 → x > 1)) :=
by
  sorry

end sufficient_but_not_necessary_for_reciprocal_l4_4138


namespace tricycle_total_spokes_l4_4550

noncomputable def front : ℕ := 20
noncomputable def middle : ℕ := 2 * front
noncomputable def back : ℝ := 20 * Real.sqrt 2
noncomputable def total_spokes : ℝ := front + middle + back

theorem tricycle_total_spokes : total_spokes = 88 :=
by
  sorry

end tricycle_total_spokes_l4_4550


namespace group_total_payment_l4_4425

-- Declare the costs of the tickets as constants
def cost_adult : ℝ := 9.50
def cost_child : ℝ := 6.50

-- Conditions for the group
def total_moviegoers : ℕ := 7
def number_adults : ℕ := 3

-- Calculate the number of children
def number_children : ℕ := total_moviegoers - number_adults

-- Define the total cost paid by the group
def total_cost_paid : ℝ :=
  (number_adults * cost_adult) + (number_children * cost_child)

-- The proof problem: Prove that the total amount paid by the group is $54.50
theorem group_total_payment : total_cost_paid = 54.50 := by
  sorry

end group_total_payment_l4_4425


namespace copper_to_zinc_ratio_l4_4835

theorem copper_to_zinc_ratio (total_weight_brass : ℝ) (weight_zinc : ℝ) (weight_copper : ℝ) 
  (h1 : total_weight_brass = 100) (h2 : weight_zinc = 70) (h3 : weight_copper = total_weight_brass - weight_zinc) : 
  weight_copper / weight_zinc = 3 / 7 :=
by
  sorry

end copper_to_zinc_ratio_l4_4835


namespace larger_number_is_25_l4_4280

-- Let x and y be real numbers, with x being the larger number
variables (x y : ℝ)

-- The sum of the two numbers is 45
axiom sum_eq_45 : x + y = 45

-- The difference of the two numbers is 5
axiom diff_eq_5 : x - y = 5

-- We need to prove that the larger number x is 25
theorem larger_number_is_25 : x = 25 :=
by
  sorry

end larger_number_is_25_l4_4280


namespace sum_consecutive_integers_l4_4384

theorem sum_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 :=
by
  sorry

end sum_consecutive_integers_l4_4384


namespace value_of_f_at_sqrt2_l4_4449

noncomputable def f (x : ℝ) : ℝ := x^5 - 5 * x^4 + 10 * x^3 - 10 * x^2 + 5 * x - 1

theorem value_of_f_at_sqrt2 :
  f (1 + Real.sqrt 2) = 4 * Real.sqrt 2 := by
  sorry

end value_of_f_at_sqrt2_l4_4449


namespace chocolate_candies_total_cost_l4_4145

-- Condition 1: A box of 30 chocolate candies costs $7.50
def box_cost : ℝ := 7.50
def candies_per_box : ℕ := 30

-- Condition 2: The local sales tax rate is 10%
def sales_tax_rate : ℝ := 0.10

-- Total number of candies to be bought
def total_candy_count : ℕ := 540

-- Calculate the number of boxes needed
def number_of_boxes (total_candies : ℕ) (candies_per_box : ℕ) : ℕ :=
  total_candies / candies_per_box

-- Calculate the cost without tax
def cost_without_tax (num_boxes : ℕ) (cost_per_box : ℝ) : ℝ :=
  num_boxes * cost_per_box

-- Calculate the total cost including tax
def total_cost_with_tax (cost : ℝ) (tax_rate : ℝ) : ℝ :=
  cost * (1 + tax_rate)

-- The main statement
theorem chocolate_candies_total_cost :
  total_cost_with_tax 
    (cost_without_tax (number_of_boxes total_candy_count candies_per_box) box_cost)
    sales_tax_rate = 148.50 :=
by
  sorry

end chocolate_candies_total_cost_l4_4145


namespace bagel_pieces_after_10_cuts_l4_4662

def bagel_pieces_after_cuts (initial_pieces : ℕ) (cuts : ℕ) : ℕ :=
  initial_pieces + cuts

theorem bagel_pieces_after_10_cuts : bagel_pieces_after_cuts 1 10 = 11 := by
  sorry

end bagel_pieces_after_10_cuts_l4_4662


namespace even_function_expression_l4_4579

theorem even_function_expression (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x))
  (h_neg : ∀ x, x < 0 → f x = x * (2 * x - 1)) :
  ∀ x, x > 0 → f x = x * (2 * x + 1) :=
by 
  sorry

end even_function_expression_l4_4579


namespace eggs_left_l4_4986

theorem eggs_left (x : ℕ) : (47 - 5 - x) = (42 - x) :=
  by
  sorry

end eggs_left_l4_4986


namespace findPhoneNumber_l4_4799

noncomputable def isValidPhoneNumber (T : ℕ) : Prop :=
  T >= 100000 ∧ T < 1000000 ∧  
  T % 10 % 2 = 1 ∧
  T / 10000 % 10 = 7 ∧ 
  T / 100 % 10 = 2 ∧
  (T % 3 = T % 4 ∧
   T % 4 = T % 7 ∧
   T % 7 = T % 9 ∧
   T % 9 = T % 11 ∧
   T % 11 = T % 13)

theorem findPhoneNumber (T : ℕ) (h_valid : isValidPhoneNumber T) : T = 720721 :=
by
  sorry

end findPhoneNumber_l4_4799


namespace factorization_correct_l4_4702

-- Define the initial expression
def initial_expr (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + a

-- Define the factorized form
def factorized_expr (a x : ℝ) : ℝ := a * (x - 1)^2

-- Create the theorem statement
theorem factorization_correct (a x : ℝ) : initial_expr a x = factorized_expr a x :=
by simp [initial_expr, factorized_expr, pow_two, mul_add, add_mul, add_assoc]; sorry

end factorization_correct_l4_4702


namespace f_2007_l4_4233

def A : Set ℚ := {x : ℚ | x ≠ 0 ∧ x ≠ 1}

noncomputable def f : A → ℝ := sorry

theorem f_2007 :
  (∀ x : ℚ, x ∈ A → f ⟨x, sorry⟩ + f ⟨1 - (1/x), sorry⟩ = Real.log (|x|)) →
  f ⟨2007, sorry⟩ = Real.log (|2007|) :=
sorry

end f_2007_l4_4233


namespace b2_values_count_l4_4546

open Int
open Nat

theorem b2_values_count :
  let sequence (b : ℕ → ℕ) := ∀ n, b (n + 2) = abs (b (n + 1) - b n)
  in ∀ b : ℕ → ℕ, b 1 = 1001 ∧ (b 2 < 1001) ∧ b 2023 = 0 ∧ sequence b →
  Finset.card ((Finset.filter (λ x, x < 1001 ∧ even x ∧ gcd 1001 x = 1) (Finset.range 1001))) = 386 :=
by
  sorry

end b2_values_count_l4_4546


namespace B_pow_48_l4_4951

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 0, 0],
  ![0, 0, 1],
  ![0, -1, 0]
]

theorem B_pow_48 :
  B^48 = ![
    ![0, 0, 0],
    ![0, 1, 0],
    ![0, 0, 1]
  ] := by sorry

end B_pow_48_l4_4951


namespace ratio_of_boys_l4_4222

theorem ratio_of_boys (p : ℚ) (h : p = (3/5) * (1 - p)) : p = 3 / 8 := by
  sorry

end ratio_of_boys_l4_4222


namespace product_of_roots_l4_4174

-- Let's define the given polynomial equation.
def polynomial : Polynomial ℝ := Polynomial.monomial 3 1 - Polynomial.monomial 2 15 + Polynomial.monomial 1 75 - Polynomial.C 50

-- Prove that the product of the roots of the given polynomial is 50.
theorem product_of_roots : (Polynomial.roots polynomial).prod id = 50 := by
  sorry

end product_of_roots_l4_4174


namespace crazy_silly_school_movie_count_l4_4261

theorem crazy_silly_school_movie_count
  (books : ℕ) (read_books : ℕ) (watched_movies : ℕ) (diff_books_movies : ℕ)
  (total_books : books = 8) 
  (read_movie_count : watched_movies = 19)
  (read_book_count : read_books = 16)
  (book_movie_diff : watched_movies = read_books + diff_books_movies)
  (diff_value : diff_books_movies = 3) :
  ∃ M, M ≥ 19 :=
by
  sorry

end crazy_silly_school_movie_count_l4_4261


namespace determine_b_from_inequality_l4_4841

theorem determine_b_from_inequality (b : ℝ) :
  (∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - b * x + 6 < 0) → b = 5 :=
by
  intro h
  -- Proof can be added here
  sorry

end determine_b_from_inequality_l4_4841


namespace bronze_status_families_count_l4_4112

theorem bronze_status_families_count :
  ∃ B : ℕ, (B * 25) = (700 - (7 * 50 + 1 * 100)) ∧ B = 10 := 
sorry

end bronze_status_families_count_l4_4112


namespace cartesian_equation_of_line_l4_4458

theorem cartesian_equation_of_line (t x y : ℝ)
  (h1 : x = 1 + t / 2)
  (h2 : y = 2 + (Real.sqrt 3 / 2) * t) :
  Real.sqrt 3 * x - y + 2 - Real.sqrt 3 = 0 :=
sorry

end cartesian_equation_of_line_l4_4458


namespace product_divisible_by_4_l4_4143

noncomputable def biased_die_prob_divisible_by_4 : ℚ :=
  let q := 1/4  -- probability of rolling a number divisible by 3
  let p4 := 2 * q -- probability of rolling a number divisible by 4
  let p_neither := (1 - p4) * (1 - p4) -- probability of neither roll being divisible by 4
  1 - p_neither -- probability that at least one roll is divisible by 4

theorem product_divisible_by_4 :
  biased_die_prob_divisible_by_4 = 3/4 :=
by
  sorry

end product_divisible_by_4_l4_4143


namespace triangle_isosceles_or_right_l4_4578

theorem triangle_isosceles_or_right (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_side_constraint : a + b > c ∧ a + c > b ∧ b + c > a)
  (h_condition: a^2 * c^2 - b^2 * c^2 = a^4 - b^4) :
  (a = b) ∨ (a^2 + b^2 = c^2) :=
by {
  sorry
}

end triangle_isosceles_or_right_l4_4578


namespace range_of_z_l4_4051

theorem range_of_z (x y : ℝ) 
  (h1 : x + 2 ≥ y) 
  (h2 : x + 2 * y ≥ 4) 
  (h3 : y ≤ 5 - 2 * x) : 
  ∃ (z_min z_max : ℝ), 
    (z_min = 1) ∧ 
    (z_max = 2) ∧ 
    (∀ z, z = (2 * x + y - 1) / (x + 1) → z_min ≤ z ∧ z ≤ z_max) :=
by
  sorry

end range_of_z_l4_4051


namespace x_intercept_l4_4606

theorem x_intercept (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 10) (hy1 : y1 = 3) (hx2 : x2 = -8) (hy2 : y2 = -6) : 
  ∃ x : ℝ, (y = 0) ∧ (∃ m : ℝ, m = (y2 - y1) / (x2 - x1) ∧ y1 - y = m * (x1 - x)) ∧ x = 4 :=
sorry

end x_intercept_l4_4606


namespace train_crossing_time_l4_4528

noncomputable def train_length : ℕ := 150
noncomputable def bridge_length : ℕ := 150
noncomputable def train_speed_kmph : ℕ := 36

noncomputable def kmph_to_mps (speed_kmph : ℕ) : ℕ :=
  (speed_kmph * 1000) / 3600

noncomputable def train_speed_mps : ℕ := kmph_to_mps train_speed_kmph

noncomputable def total_distance : ℕ := train_length + bridge_length

noncomputable def crossing_time_in_seconds (distance : ℕ) (speed : ℕ) : ℕ :=
  distance / speed

theorem train_crossing_time :
  crossing_time_in_seconds total_distance train_speed_mps = 30 :=
by
  sorry

end train_crossing_time_l4_4528


namespace generalized_inequality_l4_4885

theorem generalized_inequality (n k : ℕ) (h1 : 3 ≤ n) (h2 : 1 ≤ k ∧ k ≤ n) : 
  2^n + 5^n > 2^(n - k) * 5^k + 2^k * 5^(n - k) := 
by 
  sorry

end generalized_inequality_l4_4885


namespace find_fifth_term_l4_4399

noncomputable def geometric_sequence_fifth_term (a r : ℝ) (h₁ : a * r^2 = 16) (h₂ : a * r^6 = 2) : ℝ :=
  a * r^4

theorem find_fifth_term (a r : ℝ) (h₁ : a * r^2 = 16) (h₂ : a * r^6 = 2) : geometric_sequence_fifth_term a r h₁ h₂ = 2 := sorry

end find_fifth_term_l4_4399


namespace solve_equation_l4_4580

variable {x y : ℝ}

theorem solve_equation (hx1 : x ≠ 0) (hx2 : x ≠ 3) (hy1 : y ≠ 0) (hy2: y ≠ 4) (h : (3 / x) + (2 / y) = 5 / 6) :
  x = 18 * y / (5 * y - 12) :=
sorry

end solve_equation_l4_4580


namespace sin_330_eq_neg_sqrt3_div_2_l4_4004

theorem sin_330_eq_neg_sqrt3_div_2 
  (R : ℝ × ℝ)
  (hR : R = (1/2, -sqrt(3)/2))
  : Real.sin (330 * Real.pi / 180) = -sqrt(3)/2 :=
by
  sorry

end sin_330_eq_neg_sqrt3_div_2_l4_4004


namespace min_nS_n_eq_neg32_l4_4575

variable (a : ℕ → ℤ) (S : ℕ → ℤ)
variable (d : ℤ) (a_1 : ℤ)

-- Conditions
axiom arithmetic_sequence_def : ∀ n : ℕ, a n = a_1 + (n - 1) * d
axiom sum_first_n_def : ∀ n : ℕ, S n = n * a_1 + (n * (n - 1) / 2) * d

axiom a5_eq_3 : a 5 = 3
axiom S10_eq_40 : S 10 = 40

theorem min_nS_n_eq_neg32 : ∃ n : ℕ, n * S n = -32 :=
sorry

end min_nS_n_eq_neg32_l4_4575


namespace total_time_to_fill_tank_l4_4529

noncomputable def pipe_filling_time : ℕ := 
  let tank_capacity := 2000
  let pipe_a_rate := 200
  let pipe_b_rate := 50
  let pipe_c_rate := 25
  let cycle_duration := 5
  let cycle_fill := (pipe_a_rate * 1 + pipe_b_rate * 2 - pipe_c_rate * 2)
  let num_cycles := tank_capacity / cycle_fill
  num_cycles * cycle_duration

theorem total_time_to_fill_tank : pipe_filling_time = 40 := 
by
  unfold pipe_filling_time
  sorry

end total_time_to_fill_tank_l4_4529


namespace minimum_cactus_species_l4_4674

-- Definitions to represent the conditions
def num_cactophiles : Nat := 80
def num_collections (S : Finset (Fin num_cactophiles)) : Nat := S.card
axiom no_single_species_in_all (S : Finset (Fin num_cactophiles)) : num_collections S < num_cactophiles
axiom any_15_have_common_species (S : Finset (Fin num_cactophiles)) (h : S.card = 15) : 
  ∃ species, ∀ s ∈ S, species ∈ s

-- Proposition to be proved
theorem minimum_cactus_species (k : Nat) (h : ∀ S : Finset (Fin num_cactophiles), S.card = 15 → ∃ species, ∀ s ∈ S, species ∈ s) : k ≥ 16 := sorry

end minimum_cactus_species_l4_4674


namespace x_squared_plus_y_squared_l4_4072

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 12) : x^2 + y^2 = 25 := by
  sorry

end x_squared_plus_y_squared_l4_4072


namespace sin_330_eq_neg_half_l4_4031

-- Define conditions as hypotheses in Lean
def angle_330 (θ : ℝ) : Prop := θ = 330
def angle_transform (θ : ℝ) : Prop := θ = 360 - 30
def sin_pos (θ : ℝ) : Prop := Real.sin θ = 1 / 2
def sin_neg_in_4th_quadrant (θ : ℝ) : Prop := θ = 330 -> Real.sin θ < 0

-- The main theorem statement
theorem sin_330_eq_neg_half : ∀ θ : ℝ, angle_330 θ → angle_transform θ → sin_pos 30 → sin_neg_in_4th_quadrant θ → Real.sin θ = -1 / 2 := by
  intro θ h1 h2 h3 h4
  sorry

end sin_330_eq_neg_half_l4_4031


namespace total_score_l4_4147

theorem total_score (score_cap : ℝ) (score_val : ℝ) (score_imp : ℝ) (wt_cap : ℝ) (wt_val : ℝ) (wt_imp : ℝ) (total_weight : ℝ) :
  score_cap = 8 → score_val = 9 → score_imp = 7 → wt_cap = 5 → wt_val = 3 → wt_imp = 2 → total_weight = 10 →
  ((score_cap * (wt_cap / total_weight)) + (score_val * (wt_val / total_weight)) + (score_imp * (wt_imp / total_weight))) = 8.1 := 
by
  intros
  sorry

end total_score_l4_4147


namespace krishan_money_l4_4837

variable {R G K : ℕ}

theorem krishan_money 
  (h1 : R / G = 7 / 17)
  (h2 : G / K = 7 / 17)
  (hR : R = 588)
  : K = 3468 :=
by
  sorry

end krishan_money_l4_4837


namespace cricket_player_innings_l4_4288

theorem cricket_player_innings (n : ℕ) (T : ℕ) 
  (h1 : T = n * 48) 
  (h2 : T + 178 = (n + 1) * 58) : 
  n = 12 :=
by
  sorry

end cricket_player_innings_l4_4288


namespace locus_of_vertex_P_l4_4326

noncomputable def M : ℝ × ℝ := (0, 5)
noncomputable def N : ℝ × ℝ := (0, -5)
noncomputable def perimeter : ℝ := 36

theorem locus_of_vertex_P : ∃ (P : ℝ × ℝ), 
  (∃ (a b : ℝ), a = 13 ∧ b = 12 ∧ P ≠ (0,0) ∧
  (a^2 = b^2 + 5^2) ∧ 
  (perimeter = 2 * a + (5 - (-5))) ∧ 
  ((P.1)^2 / 144 + (P.2)^2 / 169 = 1)) :=
sorry

end locus_of_vertex_P_l4_4326


namespace scheme2_saves_money_for_80_participants_l4_4150

-- Define the variables and conditions
def total_charge_scheme1 (x : ℕ) (hx : x > 50) : ℕ :=
  1500 + 240 * x

def total_charge_scheme2 (x : ℕ) (hx : x > 50) : ℕ :=
  270 * (x - 5)

-- Define the theorem
theorem scheme2_saves_money_for_80_participants :
  total_charge_scheme2 80 (by decide) < total_charge_scheme1 80 (by decide) :=
sorry

end scheme2_saves_money_for_80_participants_l4_4150


namespace smallest_possible_value_of_other_integer_l4_4108

theorem smallest_possible_value_of_other_integer (x : ℕ) (x_pos : 0 < x) (a b : ℕ) (h1 : a = 77) 
    (h2 : gcd a b = x + 7) (h3 : lcm a b = x * (x + 7)) : b = 22 :=
sorry

end smallest_possible_value_of_other_integer_l4_4108


namespace odd_blue_faces_in_cubes_l4_4548

noncomputable def count_odd_blue_faces (length width height : ℕ) : ℕ :=
if length = 6 ∧ width = 4 ∧ height = 2 then 16 else 0

theorem odd_blue_faces_in_cubes : count_odd_blue_faces 6 4 2 = 16 := 
by
  -- The proof would involve calculating the corners, edges, etc.
  sorry

end odd_blue_faces_in_cubes_l4_4548


namespace k_n_sum_l4_4073

theorem k_n_sum (k n : ℕ) (x y : ℕ):
  2 * x^k * y^(k+2) + 3 * x^2 * y^n = 5 * x^2 * y^n → k + n = 6 :=
by sorry

end k_n_sum_l4_4073


namespace max_value_fraction_l4_4954

theorem max_value_fraction (a b x y : ℝ) (h1 : a > 1) (h2 : b > 1) (h3 : a^x = 3) (h4 : b^y = 3) (h5 : a + b = 2 * Real.sqrt 3) :
  1/x + 1/y ≤ 1 :=
sorry

end max_value_fraction_l4_4954


namespace inequality_proof_l4_4324

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  x * (x - z) ^ 2 + y * (y - z) ^ 2 ≥ (x - z) * (y - z) * (x + y - z) :=
by
  sorry

end inequality_proof_l4_4324


namespace total_seeds_in_garden_l4_4734

-- Definitions based on conditions
def large_bed_rows : Nat := 4
def large_bed_seeds_per_row : Nat := 25
def medium_bed_rows : Nat := 3
def medium_bed_seeds_per_row : Nat := 20
def num_large_beds : Nat := 2
def num_medium_beds : Nat := 2

-- Theorem statement to show total seeds
theorem total_seeds_in_garden : 
  num_large_beds * (large_bed_rows * large_bed_seeds_per_row) + 
  num_medium_beds * (medium_bed_rows * medium_bed_seeds_per_row) = 320 := 
by
  sorry

end total_seeds_in_garden_l4_4734


namespace sin_identity_cos_identity_l4_4096

-- Define the condition that alpha + beta + gamma = 180 degrees.
def angles_sum_to_180 (α β γ : ℝ) : Prop :=
  α + β + γ = Real.pi

-- Prove that sin 4α + sin 4β + sin 4γ = -4 sin 2α sin 2β sin 2γ.
theorem sin_identity (α β γ : ℝ) (h : angles_sum_to_180 α β γ) :
  Real.sin (4 * α) + Real.sin (4 * β) + Real.sin (4 * γ) = -4 * Real.sin (2 * α) * Real.sin (2 * β) * Real.sin (2 * γ) := by
  sorry

-- Prove that cos 4α + cos 4β + cos 4γ = 4 cos 2α cos 2β cos 2γ - 1.
theorem cos_identity (α β γ : ℝ) (h : angles_sum_to_180 α β γ) :
  Real.cos (4 * α) + Real.cos (4 * β) + Real.cos (4 * γ) = 4 * Real.cos (2 * α) * Real.cos (2 * β) * Real.cos (2 * γ) - 1 := by
  sorry

end sin_identity_cos_identity_l4_4096


namespace find_y_value_l4_4201

theorem find_y_value (y : ℕ) (h1 : y ≤ 150)
  (h2 : (45 + 76 + 123 + y + y + y) / 6 = 2 * y) :
  y = 27 :=
sorry

end find_y_value_l4_4201


namespace min_value_of_quadratic_l4_4212

noncomputable def quadratic_min_value (x : ℕ) : ℝ :=
  3 * (x : ℝ)^2 - 12 * x + 800

theorem min_value_of_quadratic : (∀ x : ℕ, quadratic_min_value x ≥ 788) ∧ (quadratic_min_value 2 = 788) :=
by
  sorry

end min_value_of_quadratic_l4_4212


namespace sin_A_value_of_triangle_l4_4203

theorem sin_A_value_of_triangle 
  (a b : ℝ) (A B C : ℝ) (h_triangle : a = 2) (h_b : b = 3) (h_tanB : Real.tan B = 3) :
  Real.sin A = Real.sqrt 10 / 5 :=
sorry

end sin_A_value_of_triangle_l4_4203


namespace fraction_of_area_in_triangle_l4_4491

theorem fraction_of_area_in_triangle :
  let vertex1 := (3, 3)
  let vertex2 := (5, 5)
  let vertex3 := (3, 5)
  let base := (5 - 3)
  let height := (5 - 3)
  let area_triangle := (1 / 2) * base * height
  let area_square := 6 * 6
  let fraction := area_triangle / area_square
  fraction = (1 / 18) :=
by 
  sorry

end fraction_of_area_in_triangle_l4_4491


namespace prob_1_to_2_prob_ge_2_prob_abs_diff_le_3_prob_le_neg_1_prob_abs_diff_le_9_l4_4116


-- Defining the normal distribution parameters
def a : ℝ := 2
def σ : ℝ := 3

-- Proof problem for each probability
theorem prob_1_to_2 :
  (measure_theory.measure_space.measure_univ_prob (measure_theory.measure_space.mk (ennreal.of_real_prob (gaussian a σ))) (set.Icc 1 2)).toReal = 0.1293 :=
sorry

theorem prob_ge_2 :
  (measure_theory.measure_space.measure_univ_prob (measure_theory.measure_space.mk (ennreal.of_real_prob (gaussian a σ))) (set.Ici 2)).toReal = 0.5 :=
sorry

theorem prob_abs_diff_le_3 :
  (measure_theory.measure_space.measure_univ_prob (measure_theory.measure_space.mk (ennreal.of_real_prob (gaussian a σ))) {x | abs (x - 2) ≤ 3}).toReal = 0.6826 :=
sorry

theorem prob_le_neg_1 :
  (measure_theory.measure_space.measure_univ_prob (measure_theory.measure_space.mk (ennreal.of_real_prob (gaussian a σ))) (set.Iic (-1))).toReal = 0.1587 :=
sorry

theorem prob_abs_diff_le_9 :
  (measure_theory.measure_space.measure_univ_prob (measure_theory.measure_space.mk (ennreal.of_real_prob (gaussian a σ))) {x | abs (x - 2) ≤ 9}).toReal = 0.9974 :=
sorry

end prob_1_to_2_prob_ge_2_prob_abs_diff_le_3_prob_le_neg_1_prob_abs_diff_le_9_l4_4116


namespace geometric_series_sum_l4_4884

theorem geometric_series_sum : 
  (3 + 3^2 + 3^3 + 3^4 + 3^5 + 3^6 + 3^7 + 3^8 + 3^9 + 3^10) = 88572 := 
by 
  sorry

end geometric_series_sum_l4_4884


namespace arithmetic_and_geometric_sums_l4_4204

-- Definition of the Arithmetic sequence
def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Given conditions
axiom a₁ : ℕ := 1
axiom S₁₀₀ : ℕ := 10000

-- Define the sum of the first n terms of an arithmetic sequence
def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (n : ℕ) : ℕ := (n * (a₁ + aₙ)) / 2

-- General term of the Arithmetic sequence
axiom general_term (d n : ℕ) : ℕ := 2 * n - 1

-- Sequence bn
def b_sequence (n : ℕ) : ℕ := 2 ^ (general_term d n + 1)

-- Sum of geometric series
def geometric_sum (a r n : ℕ) : ℕ := a * (1 - r^n) / (1 - r)

-- Main proof statement
theorem arithmetic_and_geometric_sums :
  let a_n := general_term 2 in
  let b_n := b_sequence in
  let S_n := geometric_sum 4 4 n in
  S₁₀₀ = 10000 → -- given condition
  S_n = 4*(4^n - 1) / 3 := 
by
  sorry

end arithmetic_and_geometric_sums_l4_4204


namespace bunny_burrows_l4_4537

theorem bunny_burrows (x : ℕ) (h1 : 20 * x * 600 = 36000) : x = 3 :=
by
  -- Skipping proof using sorry
  sorry

end bunny_burrows_l4_4537


namespace olive_charged_10_hours_l4_4084

/-- If Olive charges her phone for 3/5 of the time she charged last night, and that results
    in 12 hours of use, where each hour of charge results in 2 hours of phone usage,
    then the time Olive charged her phone last night was 10 hours. -/
theorem olive_charged_10_hours (x : ℝ) 
  (h1 : 2 * (3 / 5) * x = 12) : 
  x = 10 :=
by
  sorry

end olive_charged_10_hours_l4_4084


namespace find_bullet_l4_4655

theorem find_bullet (x y : ℝ) (h₁ : 3 * x + y = 8) (h₂ : y = -1) : 2 * x - y = 7 :=
sorry

end find_bullet_l4_4655


namespace energy_function_relationship_minimum_energy_at_v_eq_6_l4_4889

variables (k v : ℝ)

-- Conditions
def energy_consumption (k v t : ℝ) := k * v^2 * t

def time_travel (v : ℝ) : ℝ := 10 / (v - 3)

def flow_speed := 3

-- Proof Problem
theorem energy_function_relationship (k : ℝ) (v : ℝ) (h₁ : v > 3) :
  energy_consumption k v (time_travel v) = k * v^2 * (10 / (v - 3)) :=
begin
  sorry
end

theorem minimum_energy_at_v_eq_6 (k : ℝ) (h₁ : 120 * k > 0) :
  let v := 6 in energy_consumption k v (time_travel v) = 120 * k :=
begin
  sorry
end

end energy_function_relationship_minimum_energy_at_v_eq_6_l4_4889


namespace question_implies_answer_l4_4684

theorem question_implies_answer (x y : ℝ) (h : y^2 - x^2 < x) :
  (x ≥ 0 ∨ x ≤ -1) ∧ (-Real.sqrt (x^2 + x) < y ∧ y < Real.sqrt (x^2 + x)) :=
sorry

end question_implies_answer_l4_4684


namespace arithmetic_sequence_sum_l4_4350

variable {α : Type} [LinearOrderedField α]

noncomputable def a_n (a1 d n : α) := a1 + (n - 1) * d

theorem arithmetic_sequence_sum (a1 d : α) (h1 : a_n a1 d 3 * a_n a1 d 11 = 5)
  (h2 : a_n a1 d 3 + a_n a1 d 11 = 3) : a_n a1 d 5 + a_n a1 d 6 + a_n a1 d 10 = 9 / 2 :=
by
  sorry

end arithmetic_sequence_sum_l4_4350


namespace complex_exponential_sum_angle_l4_4305

theorem complex_exponential_sum_angle :
  ∃ r : ℝ, r ≥ 0 ∧ (e^(Complex.I * 11 * Real.pi / 60) + 
                     e^(Complex.I * 21 * Real.pi / 60) + 
                     e^(Complex.I * 31 * Real.pi / 60) + 
                     e^(Complex.I * 41 * Real.pi / 60) + 
                     e^(Complex.I * 51 * Real.pi / 60) = r * Complex.exp (Complex.I * 31 * Real.pi / 60)) := 
by
  sorry

end complex_exponential_sum_angle_l4_4305


namespace investment_calculation_l4_4645

noncomputable def initial_investment (final_amount : ℝ) (years : ℕ) (interest_rate : ℝ) : ℝ :=
  final_amount / ((1 + interest_rate / 100) ^ years)

theorem investment_calculation :
  initial_investment 504.32 3 12 = 359 :=
by
  sorry

end investment_calculation_l4_4645


namespace point_on_y_axis_l4_4348

theorem point_on_y_axis (a : ℝ) :
  (a + 2 = 0) -> a = -2 :=
by
  intro h
  sorry

end point_on_y_axis_l4_4348


namespace uniqueFlavors_l4_4330

-- Definitions for the conditions
def numRedCandies : ℕ := 6
def numGreenCandies : ℕ := 4
def numBlueCandies : ℕ := 5

-- Condition stating each flavor must use at least two candies and no more than two colors
def validCombination (x y z : ℕ) : Prop :=
  (x = 0 ∨ y = 0 ∨ z = 0) ∧ (x + y ≥ 2 ∨ x + z ≥ 2 ∨ y + z ≥ 2)

-- The main theorem statement
theorem uniqueFlavors : 
  ∃ n : ℕ, n = 30 ∧ 
  (∀ x y z : ℕ, validCombination x y z → (x ≤ numRedCandies) ∧ (y ≤ numGreenCandies) ∧ (z ≤ numBlueCandies)) :=
sorry

end uniqueFlavors_l4_4330


namespace lcm_of_two_numbers_l4_4972

theorem lcm_of_two_numbers (a b : ℕ) (h_hcf : Nat.gcd a b = 6) (h_product : a * b = 432) :
  Nat.lcm a b = 72 :=
by 
  sorry

end lcm_of_two_numbers_l4_4972


namespace negation_of_existence_l4_4254

theorem negation_of_existence (x : ℝ) (hx : 0 < x) : ¬ (∃ x_0 : ℝ, 0 < x_0 ∧ Real.log x_0 = x_0 - 1) 
  → ∀ x : ℝ, 0 < x → Real.log x ≠ x - 1 :=
by sorry

end negation_of_existence_l4_4254


namespace log12_eq_abc_l4_4905

theorem log12_eq_abc (a b : ℝ) (h1 : a = Real.log 7 / Real.log 6) (h2 : b = Real.log 4 / Real.log 3) : 
  Real.log 7 / Real.log 12 = (a * b + 2 * a) / (2 * b + 2) :=
by
  sorry

end log12_eq_abc_l4_4905


namespace jenny_grade_l4_4760

theorem jenny_grade (J A B : ℤ) 
  (hA : A = J - 25) 
  (hB : B = A / 2) 
  (hB_val : B = 35) : 
  J = 95 :=
by
  sorry

end jenny_grade_l4_4760


namespace exponent_multiplication_l4_4738

theorem exponent_multiplication (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : (3^a)^b = 3^3) : 3^a * 3^b = 3^4 :=
by
  sorry

end exponent_multiplication_l4_4738


namespace find_ab_l4_4730

variables (a b c : ℝ)

-- Defining the conditions
def cond1 : Prop := a - b = 5
def cond2 : Prop := a^2 + b^2 = 34
def cond3 : Prop := a^3 - b^3 = 30
def cond4 : Prop := a^2 + b^2 - c^2 = 50

theorem find_ab (h1 : cond1 a b) (h2 : cond2 a b) (h3 : cond3 a b) (h4 : cond4 a b c) :
  a * b = 4.5 :=
sorry

end find_ab_l4_4730


namespace problem_solution_l4_4039

theorem problem_solution (k : ℕ) (hk : k ≥ 2) : 
  (∀ m n : ℕ, 1 ≤ m ∧ m ≤ k → 1 ≤ n ∧ n ≤ k → m ≠ n → ¬ k ∣ (n^(n-1) - m^(m-1))) ↔ (k = 2 ∨ k = 3) :=
by
  sorry

end problem_solution_l4_4039


namespace prod_ineq_min_value_l4_4230

theorem prod_ineq_min_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a^2 + 4*a + 1) * (b^2 + 4*b + 1) * (c^2 + 4*c + 1) / (a * b * c) ≥ 216 := by
  sorry

end prod_ineq_min_value_l4_4230


namespace cos_value_in_second_quadrant_l4_4341

theorem cos_value_in_second_quadrant {B : ℝ} (h1 : π / 2 < B ∧ B < π) (h2 : Real.sin B = 5 / 13) : 
  Real.cos B = - (12 / 13) :=
sorry

end cos_value_in_second_quadrant_l4_4341


namespace find_real_numbers_l4_4198

theorem find_real_numbers (x1 x2 x3 x4 : ℝ) :
  x1 + x2 * x3 * x4 = 2 →
  x2 + x1 * x3 * x4 = 2 →
  x3 + x1 * x2 * x4 = 2 →
  x4 + x1 * x2 * x3 = 2 →
  (x1 = 1 ∧ x2 = 1 ∧ x3 = 1 ∧ x4 = 1) ∨ 
  (x1 = -1 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = 3) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = 3 ∧ x4 = -1) ∨
  (x1 = -1 ∧ x2 = 3 ∧ x3 = -1 ∧ x4 = -1) ∨
  (x1 = 3 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = -1) :=
by sorry

end find_real_numbers_l4_4198


namespace hexagon_inscribed_circumscribed_symmetric_l4_4849

-- Define the conditions of the problem
variables (R r c : ℝ)

-- Define the main assertion of the problem
theorem hexagon_inscribed_circumscribed_symmetric :
  3 * (R^2 - c^2)^4 - 4 * r^2 * (R^2 - c^2)^2 * (R^2 + c^2) - 16 * R^2 * c^2 * r^4 = 0 :=
by
  -- skipping proof
  sorry

end hexagon_inscribed_circumscribed_symmetric_l4_4849


namespace sin_330_eq_neg_half_l4_4027

-- Define conditions as hypotheses in Lean
def angle_330 (θ : ℝ) : Prop := θ = 330
def angle_transform (θ : ℝ) : Prop := θ = 360 - 30
def sin_pos (θ : ℝ) : Prop := Real.sin θ = 1 / 2
def sin_neg_in_4th_quadrant (θ : ℝ) : Prop := θ = 330 -> Real.sin θ < 0

-- The main theorem statement
theorem sin_330_eq_neg_half : ∀ θ : ℝ, angle_330 θ → angle_transform θ → sin_pos 30 → sin_neg_in_4th_quadrant θ → Real.sin θ = -1 / 2 := by
  intro θ h1 h2 h3 h4
  sorry

end sin_330_eq_neg_half_l4_4027


namespace minimum_k_value_l4_4206

theorem minimum_k_value (a b k : ℝ) (ha : 0 < a) (hb : 0 < b) (h : ∀ a b, (1 / a + 1 / b + k / (a + b)) ≥ 0) : k ≥ -4 :=
sorry

end minimum_k_value_l4_4206


namespace output_y_for_x_eq_5_l4_4627

def compute_y (x : Int) : Int :=
  if x > 0 then 3 * x + 1 else -2 * x + 3

theorem output_y_for_x_eq_5 : compute_y 5 = 16 := by
  sorry

end output_y_for_x_eq_5_l4_4627


namespace largest_int_less_150_gcd_18_eq_6_l4_4850

theorem largest_int_less_150_gcd_18_eq_6 : ∃ (n : ℕ), n < 150 ∧ gcd n 18 = 6 ∧ ∀ (m : ℕ), m < 150 ∧ gcd m 18 = 6 → m ≤ n ∧ n = 138 := 
by
  sorry

end largest_int_less_150_gcd_18_eq_6_l4_4850


namespace eccentricity_of_ellipse_l4_4250

theorem eccentricity_of_ellipse :
  ∀ (x y : ℝ), (x^2) / 25 + (y^2) / 16 = 1 → 
  (∃ (e : ℝ), e = 3 / 5) :=
by
  sorry

end eccentricity_of_ellipse_l4_4250


namespace purely_imaginary_z_l4_4057

open Complex

theorem purely_imaginary_z (b : ℝ) (h : z = (1 + b * I) / (2 + I) ∧ im z = 0) : z = -I :=
by
  sorry

end purely_imaginary_z_l4_4057


namespace rice_mixture_price_l4_4476

-- Defining the costs per kg for each type of rice
def rice_cost1 : ℝ := 16
def rice_cost2 : ℝ := 24

-- Defining the given ratio
def mixing_ratio : ℝ := 3

-- Main theorem stating the problem
theorem rice_mixture_price
  (x : ℝ)  -- The common measure of quantity in the ratio
  (h1 : 3 * x * rice_cost1 + x * rice_cost2 = 72 * x)
  (h2 : 3 * x + x = 4 * x) :
  (3 * x * rice_cost1 + x * rice_cost2) / (3 * x + x) = 18 :=
by
  sorry

end rice_mixture_price_l4_4476


namespace odd_function_strictly_decreasing_l4_4903

noncomputable def f (x : ℝ) : ℝ := sorry

axiom additivity (x y : ℝ) : f (x + y) = f x + f y
axiom negative_condition (x : ℝ) (hx : x > 0) : f x < 0

theorem odd_function : ∀ x : ℝ, f (-x) = -f x :=
by sorry

theorem strictly_decreasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂ :=
by sorry

end odd_function_strictly_decreasing_l4_4903


namespace mean_of_points_scored_l4_4571

def mean (lst : List ℕ) : ℚ :=
  (lst.sum : ℚ) / lst.length

theorem mean_of_points_scored (lst : List ℕ)
  (h1 : lst = [81, 73, 83, 86, 73]) : 
  mean lst = 79.2 :=
by
  rw [h1, mean]
  sorry

end mean_of_points_scored_l4_4571


namespace number_of_mappings_eq_eight_l4_4592

open Function Fintype

theorem number_of_mappings_eq_eight :
  let A := {a, b, c}
  let B := {1, 2}
  (card (B → A)) = 8 := by
{
  let A := {a, b, c}
  let B := {1, 2}
  have h₁ : Fintype.card B = 2, by
  {
    -- Here we will formally prove that |B| = 2
    sorry
  },
  have h₂ : Fintype.card A = 3, by
  {
    -- Here we will formally prove that |A| = 3
    sorry
  },
  have h₃ : card (B → A) = 8, by
  {
    -- Here we use the combinatorial rule to count the mappings
    sorry
  },
  exact h₃,
}

end number_of_mappings_eq_eight_l4_4592


namespace problem_x2_minus_y2_l4_4599

-- Problem statement: Given the conditions, prove x^2 - y^2 = 5 / 1111
theorem problem_x2_minus_y2 (x y : ℝ) (h1 : x + y = 5 / 11) (h2 : x - y = 1 / 101) :
  x^2 - y^2 = 5 / 1111 :=
by
  sorry

end problem_x2_minus_y2_l4_4599


namespace rice_flour_weights_l4_4286

variables (r f : ℝ)

theorem rice_flour_weights :
  (8 * r + 6 * f = 550) ∧ (4 * r + 7 * f = 375) → (r = 50) ∧ (f = 25) :=
by
  intro h
  sorry

end rice_flour_weights_l4_4286


namespace dice_probability_sum_18_l4_4992

theorem dice_probability_sum_18 :
  let num_ways := nat.choose 17 7 in
  num_ways = 19448 ∧ 
  (num_ways / (6 ^ 8) = 19448 / (6 ^ 8)) :=
by
  sorry

end dice_probability_sum_18_l4_4992


namespace travel_time_l4_4869

theorem travel_time (speed distance time : ℕ) (h_speed : speed = 60) (h_distance : distance = 180) : 
  time = distance / speed → time = 3 := by
  sorry

end travel_time_l4_4869


namespace number_of_people_got_off_at_third_stop_l4_4391

-- Definitions for each stop
def initial_passengers : ℕ := 0
def passengers_after_first_stop : ℕ := initial_passengers + 7
def passengers_after_second_stop : ℕ := passengers_after_first_stop - 3 + 5
def passengers_after_third_stop (x : ℕ) : ℕ := passengers_after_second_stop - x + 4

-- Final condition stating there are 11 passengers after the third stop
def final_passengers : ℕ := 11

-- Proof goal
theorem number_of_people_got_off_at_third_stop (x : ℕ) :
  passengers_after_third_stop x = final_passengers → x = 2 :=
by
  -- proof goes here
  sorry

end number_of_people_got_off_at_third_stop_l4_4391


namespace infinitely_many_composite_numbers_l4_4966

-- We define n in a specialized form.
def n (m : ℕ) : ℕ := (3 * m) ^ 3

-- We state that m is an odd positive integer.
def odd_positive_integer (m : ℕ) : Prop := m > 0 ∧ (m % 2 = 1)

-- The main statement: for infinitely many odd values of n, 2^n + n - 1 is composite.
theorem infinitely_many_composite_numbers : 
  ∃ (m : ℕ), odd_positive_integer m ∧ Nat.Prime (n m) ∧ ∃ d : ℕ, d > 1 ∧ d < n m ∧ (2^(n m) + n m - 1) % d = 0 :=
by
  sorry

end infinitely_many_composite_numbers_l4_4966


namespace minimum_cactus_species_l4_4675

/--
At a meeting of cactus enthusiasts, 80 cactophiles presented their collections,
each consisting of cacti of different species. It turned out that no single 
species of cactus is found in all collections simultaneously, but any 15 people
have cacti of the same species. Prove that the minimum total number of cactus 
species is 16.
-/
theorem minimum_cactus_species (k : ℕ) (h : ∀ (collections : fin 80 → finset (fin k)),
  (∀ i, collections i ≠ ∅) ∧ (∃ (j : fin k), ∀ i, j ∉ collections i) ∧ 
  (∀ (S : finset (fin 80)), S.card = 15 → ∃ j, ∀ i ∈ S, j ∈ collections i)) :
  k ≥ 16 :=
sorry

end minimum_cactus_species_l4_4675


namespace exp_increasing_a_lt_zero_l4_4912

theorem exp_increasing_a_lt_zero (a : ℝ) (h : ∀ x1 x2 : ℝ, x1 < x2 → (1 - a) ^ x1 < (1 - a) ^ x2) : a < 0 := 
sorry

end exp_increasing_a_lt_zero_l4_4912


namespace angles_geometric_sequence_count_l4_4040

def is_geometric_sequence (a b c : ℝ) : Prop :=
  (a = b * c) ∨ (b = a * c) ∨ (c = a * b)

theorem angles_geometric_sequence_count : 
  ∃! (angles : Finset ℝ), 
    (∀ θ ∈ angles, 0 < θ ∧ θ < 2 * Real.pi ∧ ¬∃ k : ℤ, θ = k * (Real.pi / 2)) ∧
    ∀ θ ∈ angles,
      is_geometric_sequence (Real.sin θ ^ 2) (Real.cos θ) (Real.tan θ) ∧
    angles.card = 2 := 
sorry

end angles_geometric_sequence_count_l4_4040


namespace smallest_value_l4_4598

theorem smallest_value (y : ℝ) (hy : 0 < y ∧ y < 1) :
  y^3 < y^2 ∧ y^3 < 3*y ∧ y^3 < (y)^(1/3:ℝ) ∧ y^3 < (1/y) :=
sorry

end smallest_value_l4_4598


namespace behemoth_and_rita_finish_ice_cream_l4_4679

theorem behemoth_and_rita_finish_ice_cream (x y : ℝ) (h : 3 * x + 2 * y = 1) : 3 * (x + y) ≥ 1 :=
by
  sorry

end behemoth_and_rita_finish_ice_cream_l4_4679


namespace M_subset_N_l4_4329

open Set

noncomputable def M : Set ℚ := {x | ∃ k : ℤ, x = k / 2 + 1 / 4}
noncomputable def N : Set ℚ := {x | ∃ k : ℤ, x = k / 4 + 1 / 2}

theorem M_subset_N : M ⊆ N := 
sorry

end M_subset_N_l4_4329


namespace arithmetic_sequence_problem_l4_4351

noncomputable def a_n (n : ℕ) : ℚ := 1 + (n - 1) / 2

noncomputable def S_n (n : ℕ) : ℚ := n * (n + 3) / 4

theorem arithmetic_sequence_problem :
  -- Given
  (∀ n, ∃ d, a_n n = a_1 + (n - 1) * d) →
  (a_n 7 = 4) →
  (a_n 19 = 2 * a_n 9) →
  -- Prove
  (∀ n, a_n n = (n + 1) / 2) ∧ (∀ n, S_n n = n * (n + 3) / 4) :=
by
  sorry

end arithmetic_sequence_problem_l4_4351


namespace proposition_C_l4_4658

theorem proposition_C (a b : ℝ) : a^3 > b^3 → a > b :=
sorry

end proposition_C_l4_4658


namespace production_rate_is_constant_l4_4959

def drum_rate := 6 -- drums per day

def days_needed_to_produce (n : ℕ) : ℕ := n / drum_rate

theorem production_rate_is_constant (n : ℕ) : days_needed_to_produce n = n / drum_rate :=
by
  sorry

end production_rate_is_constant_l4_4959


namespace factorize_expression_l4_4709

theorem factorize_expression (a x : ℝ) :
  a * x^2 - 2 * a * x + a = a * (x - 1) ^ 2 := 
sorry

end factorize_expression_l4_4709


namespace problem_solve_l4_4898

theorem problem_solve (n : ℕ) (h_pos : 0 < n) 
    (h_eq : Real.sin (Real.pi / (3 * n)) + Real.cos (Real.pi / (3 * n)) = Real.sqrt (2 * n) / 3) : 
    n = 6 := 
  sorry

end problem_solve_l4_4898


namespace lunch_customers_is_127_l4_4868

-- Define the conditions based on the given problem
def breakfast_customers : ℕ := 73
def dinner_customers : ℕ := 87
def total_customers_on_saturday : ℕ := 574
def total_customers_on_friday : ℕ := total_customers_on_saturday / 2

-- Define the variable representing the lunch customers
variable (L : ℕ)

-- State the proposition we want to prove
theorem lunch_customers_is_127 :
  breakfast_customers + L + dinner_customers = total_customers_on_friday → L = 127 := by {
  sorry
}

end lunch_customers_is_127_l4_4868


namespace minimum_value_of_x_plus_y_l4_4582

theorem minimum_value_of_x_plus_y
  (x y : ℝ)
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : (1 / y) + (4 / x) = 1) : 
  x + y = 9 :=
sorry

end minimum_value_of_x_plus_y_l4_4582


namespace sin_330_eq_neg_half_l4_4032

-- Define conditions as hypotheses in Lean
def angle_330 (θ : ℝ) : Prop := θ = 330
def angle_transform (θ : ℝ) : Prop := θ = 360 - 30
def sin_pos (θ : ℝ) : Prop := Real.sin θ = 1 / 2
def sin_neg_in_4th_quadrant (θ : ℝ) : Prop := θ = 330 -> Real.sin θ < 0

-- The main theorem statement
theorem sin_330_eq_neg_half : ∀ θ : ℝ, angle_330 θ → angle_transform θ → sin_pos 30 → sin_neg_in_4th_quadrant θ → Real.sin θ = -1 / 2 := by
  intro θ h1 h2 h3 h4
  sorry

end sin_330_eq_neg_half_l4_4032


namespace complete_square_correct_l4_4971

theorem complete_square_correct (x : ℝ) : x^2 - 4 * x - 1 = 0 → (x - 2)^2 = 5 :=
by 
  intro h
  sorry

end complete_square_correct_l4_4971


namespace infinite_solutions_l4_4058

theorem infinite_solutions (x y : ℝ) : ∃ x y : ℝ, x^3 + y^2 * x - 6 * x + 5 * y + 1 = 0 :=
sorry

end infinite_solutions_l4_4058


namespace reciprocals_sum_l4_4135

theorem reciprocals_sum (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 6 * a * b) : 
  (1 / a) + (1 / b) = 6 := 
sorry

end reciprocals_sum_l4_4135


namespace symmetric_point_coordinates_l4_4347

-- Definition of symmetry in the Cartesian coordinate system
def is_symmetrical_about_origin (A A' : ℝ × ℝ) : Prop :=
  A'.1 = -A.1 ∧ A'.2 = -A.2

-- Given point A and its symmetric property to find point A'
theorem symmetric_point_coordinates (A A' : ℝ × ℝ)
  (hA : A = (1, -2))
  (h_symm : is_symmetrical_about_origin A A') :
  A' = (-1, 2) :=
by
  sorry -- Proof to be filled in (not required as per the instructions)

end symmetric_point_coordinates_l4_4347


namespace consecutive_arithmetic_sequence_l4_4568

theorem consecutive_arithmetic_sequence (a b c : ℝ) 
  (h : (2 * b - a)^2 + (2 * b - c)^2 = 2 * (2 * b^2 - a * c)) : 
  2 * b = a + c :=
by
  sorry

end consecutive_arithmetic_sequence_l4_4568


namespace least_value_expression_l4_4651

-- Definition of the expression
def expression (x y : ℝ) := (x * y - 2) ^ 2 + (x - 1 + y) ^ 2

-- Statement to prove the least possible value of the expression
theorem least_value_expression : ∃ x y : ℝ, expression x y = 2 := 
sorry

end least_value_expression_l4_4651


namespace rectangle_perimeter_l4_4496

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem rectangle_perimeter (x y : ℝ) (A : ℝ) (E : ℝ) (fA fB : Real) (p : ℝ) 
  (h1 : y = 2 * x)
  (h2 : x * y = 2015)
  (h3 : E = 2006 * π)
  (h4 : fA = x + y)
  (h5 : fB ^ 2 = (3 / 2)^2 * 1007.5 - (p / 2)^2)
  (h6 : 2 * (3 / 2 * sqrt 1007.5 * sqrt 1009.375) = 2006 / π) :
  2 * (x + y) = 6 * sqrt 1007.5 := 
by
  sorry

end rectangle_perimeter_l4_4496


namespace interior_angles_sum_l4_4345

theorem interior_angles_sum (n : ℕ) (h : ∀ (k : ℕ), k = n → 60 * n = 360) : 
  180 * (n - 2) = 720 :=
by
  sorry

end interior_angles_sum_l4_4345


namespace problem_statement_l4_4967

noncomputable def log_base_4 (x : ℝ) : ℝ := Real.log x / Real.log 4

def f (x : ℝ) : ℝ := x^2 - 2*x + 5

def g (x : ℝ) : ℝ := log_base_4 (f x)

theorem problem_statement : 
  (∀ x : ℝ, f x > 0) ∧
  (∀ x y : ℝ, x < y → g x < g y) ∧
  (∃ x : ℝ, g x = 1) ∧
  (¬(∀ x : ℝ, g x < 0)) :=
by
  sorry

end problem_statement_l4_4967


namespace sum_a1_to_a5_l4_4917

-- Define the conditions
def equation_holds (x a0 a1 a2 a3 a4 a5 : ℝ) : Prop :=
  x^5 + 2 = a0 + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 + a4 * (x - 1)^4 + a5 * (x - 1)^5

-- State the theorem
theorem sum_a1_to_a5 (a0 a1 a2 a3 a4 a5 : ℝ) (h : ∀ x : ℝ, equation_holds x a0 a1 a2 a3 a4 a5) :
  a1 + a2 + a3 + a4 + a5 = 31 :=
by
  sorry

end sum_a1_to_a5_l4_4917


namespace intersection_area_correct_l4_4520

noncomputable def intersection_area (XY YE FX EX FY : ℕ) : ℚ :=
  if XY = 12 ∧ YE = FX ∧ YE = 15 ∧ EX = FY ∧ EX = 20 then
    18
  else
    0

theorem intersection_area_correct {XY YE FX EX FY : ℕ} (h1 : XY = 12) (h2 : YE = FX) (h3 : YE = 15) (h4 : EX = FY) (h5 : EX = 20) : 
  intersection_area XY YE FX EX FY = 18 := 
by {
  sorry
}

end intersection_area_correct_l4_4520


namespace interval_of_increase_logb_l4_4510

noncomputable def f (x : ℝ) := Real.logb 5 (2 * x + 1)

-- Define the domain
def domain : Set ℝ := {x | 2 * x + 1 > 0}

-- Define the interval of monotonic increase for the function
def interval_of_increase (f : ℝ → ℝ) : Set ℝ := {x | ∀ y, x < y → f x < f y}

-- Statement of the problem
theorem interval_of_increase_logb :
  interval_of_increase f = {x | x > - (1 / 2)} :=
by
  have h_increase : ∀ x y, x < y → f x < f y := sorry
  exact sorry

end interval_of_increase_logb_l4_4510


namespace smaller_angle_measure_l4_4122

theorem smaller_angle_measure (x : ℝ) (h₁ : 5 * x + 3 * x = 180) : 3 * x = 67.5 :=
by
  sorry

end smaller_angle_measure_l4_4122


namespace customer_difference_l4_4159

theorem customer_difference (X Y Z : ℕ) (h1 : X - Y = 10) (h2 : 10 - Z = 4) : X - 4 = 10 :=
by sorry

end customer_difference_l4_4159


namespace rancher_loss_l4_4668

-- Define the necessary conditions
def initial_head_of_cattle := 340
def original_total_price := 204000
def cattle_died := 172
def price_reduction_per_head := 150

-- Define the original and new prices per head
def original_price_per_head := original_total_price / initial_head_of_cattle
def new_price_per_head := original_price_per_head - price_reduction_per_head

-- Define the number of remaining cattle
def remaining_cattle := initial_head_of_cattle - cattle_died

-- Define the total amount at the new price
def total_amount_new_price := new_price_per_head * remaining_cattle

-- Define the loss
def loss := original_total_price - total_amount_new_price

-- Prove that the loss is $128,400
theorem rancher_loss : loss = 128400 := by
  sorry

end rancher_loss_l4_4668


namespace devin_initial_height_l4_4102

theorem devin_initial_height (h : ℝ) (p : ℝ) (p' : ℝ) :
  (p = 10 / 100) →
  (p' = (h - 66) / 100) →
  (h + 3 = 68) →
  (p + p' * (h + 3 - 66) = 30 / 100) →
  h = 68 :=
by
  intros hp hp' hg pt
  sorry

end devin_initial_height_l4_4102


namespace main_theorem_l4_4337

-- The condition
def condition (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (2^x - 1) = 4^x - 1

-- The property we need to prove
def proves (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, -1 ≤ x → f x = x^2 + 2*x

-- The main theorem connecting the condition to the desired property
theorem main_theorem (f : ℝ → ℝ) (h : condition f) : proves f :=
sorry

end main_theorem_l4_4337


namespace calories_consummed_l4_4130

-- Definitions based on conditions
def calories_per_strawberry : ℕ := 4
def calories_per_ounce_of_yogurt : ℕ := 17
def strawberries_eaten : ℕ := 12
def yogurt_eaten_in_ounces : ℕ := 6

-- Theorem statement
theorem calories_consummed (c_straw : ℕ) (c_yogurt : ℕ) (straw : ℕ) (yogurt : ℕ) 
  (h1 : c_straw = calories_per_strawberry) 
  (h2 : c_yogurt = calories_per_ounce_of_yogurt) 
  (h3 : straw = strawberries_eaten) 
  (h4 : yogurt = yogurt_eaten_in_ounces) : 
  c_straw * straw + c_yogurt * yogurt = 150 :=
by 
  -- Derived conditions
  rw [h1, h2, h3, h4]
  sorry

end calories_consummed_l4_4130


namespace mason_car_nuts_l4_4793

def busy_squirrels_num := 2
def busy_squirrel_nuts_per_day := 30
def sleepy_squirrel_num := 1
def sleepy_squirrel_nuts_per_day := 20
def days := 40

theorem mason_car_nuts : 
  busy_squirrels_num * busy_squirrel_nuts_per_day * days + sleepy_squirrel_nuts_per_day * days = 3200 :=
  by
    sorry

end mason_car_nuts_l4_4793


namespace bill_trick_probability_l4_4661

theorem bill_trick_probability :
  let cards_A := {2, 3, 5, 7}
  let cards_B := {2, 4, 6, 7}
  ∃ cards_C : Finset ℕ, cards_C.card = 4 ∧ cards_C ⊆ ({1, 2, 3, 4, 5, 6, 7, 8} : Finset ℕ) ∧ 
  (∀ n, n ∈ (cards_A ∪ cards_B ∪ cards_C) → ∃ unique_sets : Finset (Finset ℕ), unique_sets.card = 8 ∧ 
  ∀ (m : ℕ), m ∈ ({1, 2, 3, 4, 5, 6, 7, 8} : Finset ℕ) → (m ≠ n → ∀ (s t : Finset ℕ) (H₁ : s ∈ unique_sets) (H₂ : t ∈ unique_sets), s ≠ t)) →
  (Finset.card (Finset.filter (λ s : Finset ℕ, s.card = 4 ∧ ∃ p₁ p₂ p₃ p₄ : ℕ, p₁ ≠ p₂ ∧ p₂ ≠ p₃ ∧ p₃ ≠ p₄ ∧ p₄ ≠ p₁ ∧ s = {p₁, p₂, p₃, p₄} ∧ s ⊆ ({1, 2, 3, 4, 5, 6, 7, 8} : Finset ℕ)) (Finset.powerset (Finset.range 8))).val = 16) →
  (16 / 70 = (8 / 35 : ℝ)) :=
sorry

end bill_trick_probability_l4_4661


namespace find_number_l4_4814

theorem find_number (x : ℝ) : (35 - x) * 2 + 12 = 72 → ((35 - x) * 2 + 12) / 8 = 9 → x = 5 :=
by
  -- assume the first condition
  intro h1
  -- assume the second condition
  intro h2
  -- the proof goes here
  sorry

end find_number_l4_4814


namespace find_smallest_palindrome_l4_4899

def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits 10
  s = s.reverse

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_aba_form (n : ℕ) : Prop :=
  let s := n.digits 10
  s.length = 3 ∧ s.head = s.get! 2

def smallest_aba_not_palindromic_when_multiplied_by_103 : ℕ :=
  Nat.find (λ n, is_three_digit n ∧ is_aba_form n ∧ ¬is_palindrome (103 * n))

theorem find_smallest_palindrome : smallest_aba_not_palindromic_when_multiplied_by_103 = 131 := sorry

end find_smallest_palindrome_l4_4899


namespace jean_grandchildren_total_giveaway_l4_4941

theorem jean_grandchildren_total_giveaway :
  let num_grandchildren := 3
  let cards_per_grandchild_per_year := 2
  let amount_per_card := 80
  let total_amount_per_grandchild_per_year := cards_per_grandchild_per_year * amount_per_card
  let total_amount_per_year := num_grandchildren * total_amount_per_grandchild_per_year
  total_amount_per_year = 480 :=
by
  sorry

end jean_grandchildren_total_giveaway_l4_4941


namespace find_m_for_b_greater_than_a100_l4_4438

theorem find_m_for_b_greater_than_a100 :
  let a : ℕ → ℕ := λ n, Nat.recOn n 3 (λ n an, 3 ^ an)
  let b : ℕ → ℕ := λ n, Nat.recOn n 100 (λ n bn, 100 ^ bn)
  in ∃ m : ℕ, (b m > a 100) ∧ (∀ n < m, b n ≤ a 100) ∧ m = 99 :=
by
  sorry

end find_m_for_b_greater_than_a100_l4_4438


namespace smallest_x_l4_4891

theorem smallest_x (x : ℝ) (h : 4 * x^2 + 6 * x + 1 = 5) : x = -2 :=
sorry

end smallest_x_l4_4891


namespace streamers_for_price_of_confetti_l4_4426

variable (p q : ℝ) (x y : ℝ)

theorem streamers_for_price_of_confetti (h1 : x * (1 + p / 100) = y) 
                                   (h2 : y * (1 - q / 100) = x)
                                   (h3 : |p - q| = 90) :
  10 * (y * 0.4) = 4 * y :=
sorry

end streamers_for_price_of_confetti_l4_4426


namespace longest_side_of_region_l4_4683

theorem longest_side_of_region :
  (∃ (x y : ℝ), x + y ≤ 5 ∧ 3 * x + y ≥ 3 ∧ x ≥ 1 ∧ y ≥ 1) →
  (∃ (l : ℝ), l = Real.sqrt 130 / 3 ∧ 
    (l = Real.sqrt ((1 - 1)^2 + (4 - 1)^2) ∨ 
     l = Real.sqrt (((1 + 4 / 3) - 1)^2 + (1 - 1)^2) ∨ 
     l = Real.sqrt ((1 - (1 + 4 / 3))^2 + (1 - 1)^2))) :=
by
  sorry

end longest_side_of_region_l4_4683


namespace combined_distance_proof_l4_4792

/-- Define the distances walked by Lionel, Esther, and Niklaus in their respective units -/
def lionel_miles : ℕ := 4
def esther_yards : ℕ := 975
def niklaus_feet : ℕ := 1287

/-- Define the conversion factors -/
def miles_to_feet : ℕ := 5280
def yards_to_feet : ℕ := 3

/-- The total combined distance in feet -/
def total_distance_feet : ℕ :=
  (lionel_miles * miles_to_feet) + (esther_yards * yards_to_feet) + niklaus_feet

theorem combined_distance_proof : total_distance_feet = 24332 := by
  -- expand definitions and calculations here...
  -- lionel = 4 * 5280 = 21120
  -- esther = 975 * 3 = 2925
  -- niklaus = 1287
  -- sum = 21120 + 2925 + 1287 = 24332
  sorry

end combined_distance_proof_l4_4792


namespace mars_moon_cost_share_l4_4246

theorem mars_moon_cost_share :
  let total_cost := 40 * 10^9 -- total cost in dollars
  let num_people := 200 * 10^6 -- number of people sharing the cost
  (total_cost / num_people) = 200 := by
  sorry

end mars_moon_cost_share_l4_4246


namespace hyperbola_asymptote_l4_4932

theorem hyperbola_asymptote (a : ℝ) (h₀ : a > 0) 
  (h₁ : ∃ (x y : ℝ), (x, y) = (2, 1) ∧ 
       (y = (2 / a) * x ∨ y = -(2 / a) * x)) : a = 4 := by
  sorry

end hyperbola_asymptote_l4_4932


namespace cube_sum_inequality_l4_4928

theorem cube_sum_inequality (a b : ℝ) (ha : a < 0) (hb : b < 0) : 
  a^3 + b^3 ≤ a * b^2 + a^2 * b :=
sorry

end cube_sum_inequality_l4_4928


namespace truck_travel_distance_l4_4669

theorem truck_travel_distance
  (miles_traveled : ℕ)
  (gallons_used : ℕ)
  (new_gallons : ℕ)
  (rate : ℕ)
  (distance : ℕ) :
  (miles_traveled = 300) ∧
  (gallons_used = 10) ∧
  (new_gallons = 15) ∧
  (rate = miles_traveled / gallons_used) ∧
  (distance = rate * new_gallons)
  → distance = 450 :=
by
  sorry

end truck_travel_distance_l4_4669


namespace problem_acute_angles_l4_4906

theorem problem_acute_angles (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
  (h1 : 3 * (Real.sin α) ^ 2 + 2 * (Real.sin β) ^ 2 = 1)
  (h2 : 3 * Real.sin (2 * α) - 2 * Real.sin (2 * β) = 0) :
  α + 2 * β = π / 2 := 
by 
  sorry

end problem_acute_angles_l4_4906


namespace distinguishable_large_triangles_l4_4262

noncomputable def number_of_distinguishable_large_triangles : Nat :=
  let same_color := 8
  let two_same_one_diff := (Nat.choose 8 2) * 2
  let all_diff := Nat.choose 8 3
  let total_sets := same_color + two_same_one_diff + all_diff
  let center_choices := 8
  center_choices * total_sets

theorem distinguishable_large_triangles : number_of_distinguishable_large_triangles = 960 :=
by
  -- Calculations for each part
  have h_same := 8
  have h_two_same_one_diff := (Nat.choose 8 2) * 2
  have h_all_diff := Nat.choose 8 3
  have h_total_sets := h_same + h_two_same_one_diff + h_all_diff
  have h_center_choices := 8
  have h_total_configs := h_center_choices * h_total_sets
  -- Final total
  calc
    number_of_distinguishable_large_triangles
    = h_total_configs : by rfl
    = 960 : by norm_num

end distinguishable_large_triangles_l4_4262


namespace find_n_l4_4140

theorem find_n : ∀ (n x : ℝ), (3639 + n - x = 3054) → (x = 596.95) → (n = 11.95) :=
by
  intros n x h1 h2
  sorry

end find_n_l4_4140


namespace sufficient_but_not_necessary_condition_l4_4576

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x + 1) * (x - 3) < 0 → x > -1 ∧ ((x > -1) → (x + 1) * (x - 3) < 0) = false :=
sorry

end sufficient_but_not_necessary_condition_l4_4576


namespace cheburashkas_erased_l4_4766

theorem cheburashkas_erased (total_krakozyabras : ℕ) (rows : ℕ) :
  rows ≥ 2 → total_krakozyabras = 29 → ∃ (cheburashkas_erased : ℕ), cheburashkas_erased = 11 :=
by
  assume h_rows h_total_krakozyabras
  let n := (total_krakozyabras / 2) + 1
  have h_cheburashkas : cheburashkas_erased = n - 1 
  sorry

end cheburashkas_erased_l4_4766


namespace sum_of_numbers_l4_4271

theorem sum_of_numbers : 
  12345 + 23451 + 34512 + 45123 + 51234 = 166665 := 
sorry

end sum_of_numbers_l4_4271


namespace quotient_of_5_divided_by_y_is_5_point_3_l4_4514

theorem quotient_of_5_divided_by_y_is_5_point_3 (y : ℝ) (h : 5 / y = 5.3) : y = 26.5 :=
by
  sorry

end quotient_of_5_divided_by_y_is_5_point_3_l4_4514


namespace sin_330_eq_neg_sqrt3_div_2_l4_4001

theorem sin_330_eq_neg_sqrt3_div_2 
  (R : ℝ × ℝ)
  (hR : R = (1/2, -sqrt(3)/2))
  : Real.sin (330 * Real.pi / 180) = -sqrt(3)/2 :=
by
  sorry

end sin_330_eq_neg_sqrt3_div_2_l4_4001


namespace smallest_y_value_l4_4127

theorem smallest_y_value : ∃ y : ℝ, 2 * y ^ 2 + 7 * y + 3 = 5 ∧ (∀ y' : ℝ, 2 * y' ^ 2 + 7 * y' + 3 = 5 → y ≤ y') := sorry

end smallest_y_value_l4_4127


namespace line_through_A_parallel_y_axis_l4_4106

theorem line_through_A_parallel_y_axis (x y: ℝ) (A: ℝ × ℝ) (h1: A = (-3, 1)) : 
  (∀ P: ℝ × ℝ, P ∈ {p : ℝ × ℝ | p.1 = -3} → (P = A ∨ P.1 = -3)) :=
by
  sorry

end line_through_A_parallel_y_axis_l4_4106


namespace length_of_MN_eq_5_sqrt_10_div_3_l4_4352

theorem length_of_MN_eq_5_sqrt_10_div_3 
  (A : ℝ × ℝ) 
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)
  (D : ℝ × ℝ)
  (M : ℝ × ℝ)
  (N : ℝ × ℝ)
  (hyp_A : A = (1, 3))
  (hyp_B : B = (25 / 3, 5 / 3))
  (hyp_C : C = (22 / 3, 14 / 3))
  (hyp_eq_edges : (dist (0, 0) M = dist M N) ∧ (dist M N = dist N B))
  (hyp_D : D = (5 / 2, 15 / 2))
  (hyp_M : M = (5 / 3, 5)) :
  dist M N = 5 * Real.sqrt 10 / 3 :=
sorry

end length_of_MN_eq_5_sqrt_10_div_3_l4_4352


namespace m_greater_than_p_l4_4788

theorem m_greater_than_p (p m n : ℕ) (pp : Nat.Prime p) (pos_m : m > 0) (pos_n : n > 0) (h : p^2 + m^2 = n^2) : m > p :=
sorry

end m_greater_than_p_l4_4788


namespace equal_serving_weight_l4_4081

theorem equal_serving_weight (total_weight : ℝ) (num_family_members : ℕ)
  (h1 : total_weight = 13) (h2 : num_family_members = 5) :
  total_weight / num_family_members = 2.6 :=
by
  sorry

end equal_serving_weight_l4_4081


namespace cassidy_total_grounding_days_l4_4432

-- Define the initial grounding days
def initial_grounding_days : ℕ := 14

-- Define the grounding days per grade below a B
def extra_days_per_grade : ℕ := 3

-- Define the number of grades below a B
def grades_below_B : ℕ := 4

-- Define the total grounding days calculation
def total_grounding_days : ℕ := initial_grounding_days + grades_below_B * extra_days_per_grade

-- The theorem statement
theorem cassidy_total_grounding_days :
  total_grounding_days = 26 := 
sorry

end cassidy_total_grounding_days_l4_4432


namespace xsq_plus_ysq_l4_4070

theorem xsq_plus_ysq (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 12) : x^2 + y^2 = 25 :=
by
  sorry

end xsq_plus_ysq_l4_4070


namespace difference_between_oranges_and_apples_l4_4843

-- Definitions of the conditions
variables (A B P O: ℕ)
variables (h1: O = 6)
variables (h2: B = 3 * A)
variables (h3: P = B / 2)
variables (h4: A + B + P + O = 28)

-- The proof problem statement
theorem difference_between_oranges_and_apples
    (A B P O: ℕ)
    (h1: O = 6)
    (h2: B = 3 * A)
    (h3: P = B / 2)
    (h4: A + B + P + O = 28) :
    O - A = 2 :=
sorry

end difference_between_oranges_and_apples_l4_4843


namespace total_chairs_calculation_l4_4363

theorem total_chairs_calculation
  (chairs_per_trip : ℕ)
  (trips_per_student : ℕ)
  (total_students : ℕ)
  (h1 : chairs_per_trip = 5)
  (h2 : trips_per_student = 10)
  (h3 : total_students = 5) :
  total_students * (chairs_per_trip * trips_per_student) = 250 :=
by
  sorry

end total_chairs_calculation_l4_4363


namespace martin_speed_l4_4378

theorem martin_speed (distance : ℝ) (time : ℝ) (h₁ : distance = 12) (h₂ : time = 6) : (distance / time = 2) :=
by 
  -- Note: The proof is not required as per instructions, so we use 'sorry'
  sorry

end martin_speed_l4_4378


namespace solution_set_of_inequality_l4_4718

theorem solution_set_of_inequality (x : ℝ) : x * (x + 3) ≥ 0 ↔ x ≤ -3 ∨ x ≥ 0 :=
by
  sorry

end solution_set_of_inequality_l4_4718


namespace total_lateness_l4_4678

/-
  Conditions:
  Charlize was 20 minutes late.
  Ana was 5 minutes later than Charlize.
  Ben was 15 minutes less late than Charlize.
  Clara was twice as late as Charlize.
  Daniel was 10 minutes earlier than Clara.

  Total time for which all five students were late is 120 minutes.
-/

def charlize := 20
def ana := charlize + 5
def ben := charlize - 15
def clara := charlize * 2
def daniel := clara - 10

def total_time := charlize + ana + ben + clara + daniel

theorem total_lateness : total_time = 120 :=
by
  sorry

end total_lateness_l4_4678


namespace factorization_l4_4691

theorem factorization (a x : ℝ) : ax^2 - 2ax + a = a * (x - 1) ^ 2 := 
by
  sorry

end factorization_l4_4691


namespace minimum_cactus_species_l4_4677

theorem minimum_cactus_species (cactophiles : Fin 80 → Set (Fin k)) :
  (∀ s : Fin k, ∃ col, cactophiles col s = False) ∧
  (∀ group : Set (Fin 80), group.card = 15 → ∃ c : Fin k, ∀ col ∈ group, (cactophiles col c)) →
  16 ≤ k :=
by
  sorry

end minimum_cactus_species_l4_4677


namespace sin_330_value_l4_4008

noncomputable def sin_330 : ℝ := Real.sin (330 * Real.pi / 180)

theorem sin_330_value : sin_330 = -1/2 :=
by {
  sorry
}

end sin_330_value_l4_4008


namespace factorize_expression_l4_4710

theorem factorize_expression (a x : ℝ) :
  a * x^2 - 2 * a * x + a = a * (x - 1) ^ 2 := 
sorry

end factorize_expression_l4_4710


namespace sin_330_eq_neg_half_l4_4020

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  -- Proof would go here
  sorry

end sin_330_eq_neg_half_l4_4020


namespace certain_events_l4_4508

-- Define the idioms and their classifications
inductive Event
| impossible
| certain
| unlikely

-- Definitions based on the given conditions
def scooping_moon := Event.impossible
def rising_tide := Event.certain
def waiting_by_stump := Event.unlikely
def catching_turtles := Event.certain
def pulling_seeds := Event.impossible

-- The theorem statement
theorem certain_events :
  (rising_tide = Event.certain) ∧ (catching_turtles = Event.certain) := by
  -- Proof is omitted
  sorry

end certain_events_l4_4508


namespace trigonometric_identity_l4_4331

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  (4 * Real.sin α - 2 * Real.cos α) / (3 * Real.cos α + 3 * Real.sin α) = 2 / 3 :=
by
  sorry

end trigonometric_identity_l4_4331


namespace sin_330_value_l4_4006

noncomputable def sin_330 : ℝ := Real.sin (330 * Real.pi / 180)

theorem sin_330_value : sin_330 = -1/2 :=
by {
  sorry
}

end sin_330_value_l4_4006


namespace find_number_l4_4284

theorem find_number (x : ℝ) (h : 3034 - x / 20.04 = 2984) : x = 1002 :=
sorry

end find_number_l4_4284


namespace stable_performance_l4_4893

/-- The variance of student A's scores is 0.4 --/
def variance_A : ℝ := 0.4

/-- The variance of student B's scores is 0.3 --/
def variance_B : ℝ := 0.3

/-- Prove that student B has more stable performance given the variances --/
theorem stable_performance (h1 : variance_A = 0.4) (h2 : variance_B = 0.3) : variance_B < variance_A :=
by
  rw [h1, h2]
  exact sorry

end stable_performance_l4_4893


namespace jury_concludes_you_are_not_guilty_l4_4631

def criminal_is_a_liar : Prop := sorry -- The criminal is a liar, known.
def you_are_a_liar : Prop := sorry -- You are a liar, unknown.
def you_are_not_guilty : Prop := sorry -- You are not guilty.

theorem jury_concludes_you_are_not_guilty :
  criminal_is_a_liar → you_are_a_liar → you_are_not_guilty → "I am guilty" = "You are not guilty" :=
by
  -- Proof construct omitted as per problem requirements
  sorry

end jury_concludes_you_are_not_guilty_l4_4631


namespace Lorelei_picks_22_roses_l4_4940

theorem Lorelei_picks_22_roses :
  let red_flowers := 12 in
  let pink_flowers := 18 in
  let yellow_flowers := 20 in
  let orange_flowers := 8 in
  let picked_red := 0.50 * red_flowers in
  let picked_pink := 0.50 * pink_flowers in
  let picked_yellow := 0.25 * yellow_flowers in
  let picked_orange := 0.25 * orange_flowers in
  picked_red + picked_pink + picked_yellow + picked_orange = 22 :=
by
  sorry

end Lorelei_picks_22_roses_l4_4940


namespace simple_interest_rate_l4_4408

theorem simple_interest_rate (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) :
  (T = 20) →
  (SI = P) →
  (SI = P * R * T / 100) →
  R = 5 :=
by
  sorry

end simple_interest_rate_l4_4408


namespace end_digit_of_number_l4_4867

theorem end_digit_of_number (n : ℕ) (h_n : n = 2022) (h_start : ∃ (f : ℕ → ℕ), f 0 = 4 ∧ 
    (∀ i < n - 1, (19 ∣ (10 * f i + f (i + 1))) ∨ (23 ∣ (10 * f i + f (i + 1))))) :
  ∃ (f : ℕ → ℕ), f (n - 1) = 8 :=
by {
  sorry
}

end end_digit_of_number_l4_4867


namespace fraction_subtraction_inequality_l4_4625

theorem fraction_subtraction_inequality (a b n : ℕ) (h1 : a < b) (h2 : 0 < n) (h3 : n < a) : 
  (a : ℚ) / b > (a - n : ℚ) / (b - n) :=
sorry

end fraction_subtraction_inequality_l4_4625


namespace right_triangle_area_l4_4833

/-- Given a right triangle with hypotenuse 13 meters and one side 5 meters,
prove that the area of the triangle is 30 square meters. -/
theorem right_triangle_area (a b c : ℝ) (h : a^2 + b^2 = c^2) (hc : c = 13) (ha : a = 5) :
  1/2 * a * b = 30 :=
by sorry

end right_triangle_area_l4_4833


namespace distance_between_trees_l4_4136

theorem distance_between_trees (trees : ℕ) (total_length : ℝ) (n : trees = 26) (l : total_length = 500) :
  ∃ d : ℝ, d = total_length / (trees - 1) ∧ d = 20 :=
by
  sorry

end distance_between_trees_l4_4136


namespace least_possible_area_of_square_l4_4257

theorem least_possible_area_of_square (s : ℝ) (h₁ : 4.5 ≤ s) (h₂ : s < 5.5) : 
  s * s ≥ 20.25 :=
sorry

end least_possible_area_of_square_l4_4257


namespace probability_of_dime_l4_4414

noncomputable def num_quarters := 12 / 0.25
noncomputable def num_dimes := 8 / 0.10
noncomputable def num_pennies := 5 / 0.01
noncomputable def total_coins := num_quarters + num_dimes + num_pennies

theorem probability_of_dime : (num_dimes / total_coins) = (40 / 314) :=
by
  sorry

end probability_of_dime_l4_4414


namespace problem_III_l4_4455

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 1) / x

theorem problem_III
  (a x1 x2 : ℝ)
  (h_a : 0 < a ∧ a < 1)
  (h_roots : f x1 = a ∧ f x2 = a)
  (h_order : x1 < x2)
  (h_bounds : Real.exp (-1) < x1 ∧ x1 < 1 ∧ 1 < x2) :
  x2 - x1 > 1 / a - 1 :=
sorry

end problem_III_l4_4455


namespace sin_330_eq_neg_half_l4_4025

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  -- Proof would go here
  sorry

end sin_330_eq_neg_half_l4_4025


namespace number_of_zeros_of_f_l4_4213

noncomputable def f (x : ℝ) : ℝ := Real.log x + 3 * x - 6

theorem number_of_zeros_of_f : ∃! x : ℝ, 0 < x ∧ f x = 0 :=
sorry

end number_of_zeros_of_f_l4_4213


namespace right_triangle_area_l4_4824

theorem right_triangle_area
  (hypotenuse : ℝ) (leg1 : ℝ) (leg2 : ℝ)
  (hypotenuse_eq : hypotenuse = 13)
  (leg1_eq : leg1 = 5)
  (pythagorean_eq : hypotenuse^2 = leg1^2 + leg2^2) :
  (1 / 2) * leg1 * leg2 = 30 :=
by
  sorry

end right_triangle_area_l4_4824


namespace find_pairs_of_positive_integers_l4_4197

theorem find_pairs_of_positive_integers (n m : ℕ) (hn : 0 < n) (hm : 0 < m) : 
  3 * 2^m + 1 = n^2 ↔ (n = 7 ∧ m = 4) ∨ (n = 5 ∧ m = 3) :=
sorry

end find_pairs_of_positive_integers_l4_4197


namespace two_digit_integers_mod_9_eq_3_l4_4925

theorem two_digit_integers_mod_9_eq_3 :
  { x : ℕ | 10 ≤ x ∧ x < 100 ∧ x % 9 = 3 }.finite.card = 10 :=
by sorry

end two_digit_integers_mod_9_eq_3_l4_4925


namespace factoring_correct_l4_4252

-- Definitions corresponding to the problem conditions
def optionA (a : ℝ) : Prop := a^2 - 5*a - 6 = (a - 6) * (a + 1)
def optionB (a x b c : ℝ) : Prop := a*x + b*x + c = (a + b)*x + c
def optionC (a b : ℝ) : Prop := (a + b)^2 = a^2 + 2*a*b + b^2
def optionD (a b : ℝ) : Prop := (a + b)*(a - b) = a^2 - b^2

-- The main theorem that proves option A is the correct answer
theorem factoring_correct : optionA a := by
  sorry

end factoring_correct_l4_4252


namespace product_of_roots_l4_4189

theorem product_of_roots :
  let f := Polynomial.C (-50) + Polynomial.X * (Polynomial.C 75 + Polynomial.X * (Polynomial.C (-15) + Polynomial.X)) in 
  (f.roots.Prod = 50) :=
by sorry

end product_of_roots_l4_4189


namespace abs_x_minus_1_lt_2_is_necessary_but_not_sufficient_l4_4114

theorem abs_x_minus_1_lt_2_is_necessary_but_not_sufficient (x : ℝ) :
  (-1 < x ∧ x < 3) ↔ (0 < x ∧ x < 3) :=
sorry

end abs_x_minus_1_lt_2_is_necessary_but_not_sufficient_l4_4114


namespace smallest_area_is_10_l4_4446

noncomputable def smallest_square_area : ℝ :=
  let k₁ := 65
  let k₂ := -5
  10 * (9 + 4 * k₂)

theorem smallest_area_is_10 :
  smallest_square_area = 10 := by
  sorry

end smallest_area_is_10_l4_4446


namespace triangle_area_30_l4_4830

theorem triangle_area_30 (h : ∃ a b c : ℝ, a^2 + b^2 = c^2 ∧ a = 5 ∧ c = 13 ∧ b > 0) : 
  ∃ area : ℝ, area = 1 / 2 * 5 * (b : ℝ) ∧ area = 30 :=
by
  sorry

end triangle_area_30_l4_4830


namespace xsq_plus_ysq_l4_4069

theorem xsq_plus_ysq (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 12) : x^2 + y^2 = 25 :=
by
  sorry

end xsq_plus_ysq_l4_4069


namespace calculate_new_measure_l4_4473

noncomputable def equilateral_triangle_side_length : ℝ := 7.5

theorem calculate_new_measure :
  3 * (equilateral_triangle_side_length ^ 2) = 168.75 :=
by
  sorry

end calculate_new_measure_l4_4473


namespace probability_of_odd_sum_l4_4647

theorem probability_of_odd_sum (P : ℝ → Prop) 
    (P_even_sum : ℝ)
    (P_odd_sum : ℝ)
    (h1 : P_even_sum = 2 * P_odd_sum) 
    (h2 : P_even_sum + P_odd_sum = 1) :
    P_odd_sum = 4/9 := 
sorry

end probability_of_odd_sum_l4_4647


namespace factorization_correct_l4_4704

-- Define the initial expression
def initial_expr (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + a

-- Define the factorized form
def factorized_expr (a x : ℝ) : ℝ := a * (x - 1)^2

-- Create the theorem statement
theorem factorization_correct (a x : ℝ) : initial_expr a x = factorized_expr a x :=
by simp [initial_expr, factorized_expr, pow_two, mul_add, add_mul, add_assoc]; sorry

end factorization_correct_l4_4704


namespace ball_distribution_l4_4921

theorem ball_distribution (balls boxes : ℕ) (hballs : balls = 7) (hboxes : boxes = 4) :
  (∃ (ways : ℕ), ways = (Nat.choose (balls - 1) (boxes - 1)) ∧ ways = 20) :=
by
  sorry

end ball_distribution_l4_4921


namespace total_bottles_ordered_l4_4155

constant cases_april : ℕ
constant cases_may : ℕ
constant bottles_per_case : ℕ

axiom cases_april_def : cases_april = 20
axiom cases_may_def : cases_may = 30
axiom bottles_per_case_def : bottles_per_case = 20

theorem total_bottles_ordered :
  cases_april * bottles_per_case + cases_may * bottles_per_case = 1000 :=
by 
  rw [cases_april_def, cases_may_def, bottles_per_case_def]
  -- The remaining steps will be carried out and concluded with the necessary checks
  sorry

end total_bottles_ordered_l4_4155


namespace right_triangle_area_l4_4832

/-- Given a right triangle with hypotenuse 13 meters and one side 5 meters,
prove that the area of the triangle is 30 square meters. -/
theorem right_triangle_area (a b c : ℝ) (h : a^2 + b^2 = c^2) (hc : c = 13) (ha : a = 5) :
  1/2 * a * b = 30 :=
by sorry

end right_triangle_area_l4_4832


namespace relationship_between_sets_l4_4978

def M (x : ℤ) : Prop := ∃ k : ℤ, x = 5 * k - 2
def P (x : ℤ) : Prop := ∃ n : ℤ, x = 5 * n + 3
def S (x : ℤ) : Prop := ∃ m : ℤ, x = 10 * m + 3

theorem relationship_between_sets :
  (∀ x, S x → P x) ∧ (∀ x, P x → M x) ∧ (∀ x, M x → P x) :=
by
  sorry

end relationship_between_sets_l4_4978


namespace emily_spending_l4_4310

theorem emily_spending (X Y : ℝ) 
  (h1 : (X + 2*X + 3*X + 12*X) = Y) : 
  X = Y / 18 := 
by
  sorry

end emily_spending_l4_4310


namespace additional_cards_l4_4298

theorem additional_cards (total_cards : ℕ) (num_decks : ℕ) (cards_per_deck : ℕ) 
  (h1 : total_cards = 319) (h2 : num_decks = 6) (h3 : cards_per_deck = 52) : 
  319 - 6 * 52 = 7 := 
by
  sorry

end additional_cards_l4_4298


namespace number_of_Cheburashkas_erased_l4_4777

theorem number_of_Cheburashkas_erased :
  ∃ (n : ℕ), 
    (∀ x, x ≥ 1 → 
      (let totalKrakozyabras = (2 * (x - 1) = 29) in
         x - 2 = 11)) :=
sorry

end number_of_Cheburashkas_erased_l4_4777


namespace distance_covered_l4_4543

noncomputable def boat_speed_still_water : ℝ := 6.5
noncomputable def current_speed : ℝ := 2.5
noncomputable def time_taken : ℝ := 35.99712023038157

noncomputable def effective_speed_downstream (boat_speed_still_water current_speed : ℝ) : ℝ :=
  boat_speed_still_water + current_speed

noncomputable def convert_kmph_to_mps (speed_in_kmph : ℝ) : ℝ :=
  speed_in_kmph * (1000 / 3600)

noncomputable def calculate_distance (speed_in_mps time_in_seconds : ℝ) : ℝ :=
  speed_in_mps * time_in_seconds

theorem distance_covered :
  calculate_distance (convert_kmph_to_mps (effective_speed_downstream boat_speed_still_water current_speed)) time_taken = 89.99280057595392 :=
by
  sorry

end distance_covered_l4_4543


namespace exp_13_pi_i_over_2_eq_i_l4_4886

theorem exp_13_pi_i_over_2_eq_i : Complex.exp (13 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end exp_13_pi_i_over_2_eq_i_l4_4886


namespace ramesh_paid_price_l4_4804

variables 
  (P : Real) -- Labelled price of the refrigerator
  (paid_price : Real := 0.80 * P + 125 + 250) -- Price paid after discount and additional costs
  (sell_price : Real := 1.16 * P) -- Price to sell for 16% profit
  (sell_at : Real := 18560) -- Target selling price for given profit

theorem ramesh_paid_price : 
  1.16 * P = 18560 → paid_price = 13175 :=
by
  sorry

end ramesh_paid_price_l4_4804


namespace power_product_is_100_l4_4405

theorem power_product_is_100 :
  (10^0.6) * (10^0.4) * (10^0.3) * (10^0.2) * (10^0.5) = 100 :=
by
  sorry

end power_product_is_100_l4_4405


namespace cube_root_expression_l4_4929

variable (x : ℝ)

theorem cube_root_expression (h : x + 1 / x = 7) : x^3 + 1 / x^3 = 322 :=
  sorry

end cube_root_expression_l4_4929


namespace largest_divisor_consecutive_odd_l4_4613

theorem largest_divisor_consecutive_odd (m n : ℤ) (h : ∃ k : ℤ, m = 2 * k + 1 ∧ n = 2 * k - 1) :
  ∃ d : ℤ, d = 8 ∧ ∀ m n : ℤ, (∃ k : ℤ, m = 2 * k + 1 ∧ n = 2 * k - 1) → d ∣ (m^2 - n^2) :=
by
  sorry

end largest_divisor_consecutive_odd_l4_4613


namespace total_revenue_correct_l4_4420

def price_per_book : ℝ := 25
def books_sold_monday : ℕ := 60
def discount_monday : ℝ := 0.10
def books_sold_tuesday : ℕ := 10
def discount_tuesday : ℝ := 0.0
def books_sold_wednesday : ℕ := 20
def discount_wednesday : ℝ := 0.05
def books_sold_thursday : ℕ := 44
def discount_thursday : ℝ := 0.15
def books_sold_friday : ℕ := 66
def discount_friday : ℝ := 0.20

def revenue (books_sold: ℕ) (discount: ℝ) : ℝ :=
  (1 - discount) * price_per_book * books_sold

theorem total_revenue_correct :
  revenue books_sold_monday discount_monday +
  revenue books_sold_tuesday discount_tuesday +
  revenue books_sold_wednesday discount_wednesday +
  revenue books_sold_thursday discount_thursday +
  revenue books_sold_friday discount_friday = 4330 := by 
sorry

end total_revenue_correct_l4_4420


namespace total_seeds_in_garden_l4_4731

-- Definitions based on the conditions
def top_bed_rows : ℕ := 4
def top_bed_seeds_per_row : ℕ := 25
def num_top_beds : ℕ := 2

def medium_bed_rows : ℕ := 3
def medium_bed_seeds_per_row : ℕ := 20
def num_medium_beds : ℕ := 2

-- Calculation of total seeds in top beds
def seeds_per_top_bed : ℕ := top_bed_rows * top_bed_seeds_per_row
def total_seeds_top_beds : ℕ := num_top_beds * seeds_per_top_bed

-- Calculation of total seeds in medium beds
def seeds_per_medium_bed : ℕ := medium_bed_rows * medium_bed_seeds_per_row
def total_seeds_medium_beds : ℕ := num_medium_beds * seeds_per_medium_bed

-- Proof goal
theorem total_seeds_in_garden : total_seeds_top_beds + total_seeds_medium_beds = 320 :=
by
  sorry

end total_seeds_in_garden_l4_4731


namespace arithmetic_expression_evaluation_l4_4660

theorem arithmetic_expression_evaluation : 1997 * (2000 / 2000) - 2000 * (1997 / 1997) = -3 := 
by
  sorry

end arithmetic_expression_evaluation_l4_4660


namespace find_B_age_l4_4279

variable (a b c : ℕ)

def problem_conditions : Prop :=
  a = b + 2 ∧ b = 2 * c ∧ a + b + c = 22

theorem find_B_age (h : problem_conditions a b c) : b = 8 :=
by {
  sorry
}

end find_B_age_l4_4279


namespace right_triangle_area_l4_4825

theorem right_triangle_area
  (hypotenuse : ℝ) (leg1 : ℝ) (leg2 : ℝ)
  (hypotenuse_eq : hypotenuse = 13)
  (leg1_eq : leg1 = 5)
  (pythagorean_eq : hypotenuse^2 = leg1^2 + leg2^2) :
  (1 / 2) * leg1 * leg2 = 30 :=
by
  sorry

end right_triangle_area_l4_4825


namespace factorize_expression_l4_4711

theorem factorize_expression (a x : ℝ) :
  a * x^2 - 2 * a * x + a = a * (x - 1) ^ 2 := 
sorry

end factorize_expression_l4_4711


namespace number_of_two_digit_integers_with_remainder_3_mod_9_l4_4923

theorem number_of_two_digit_integers_with_remainder_3_mod_9 : 
  {x : ℤ // 10 ≤ x ∧ x < 100 ∧ ∃ n : ℤ, x = 9 * n + 3}.card = 10 := by
sorry

end number_of_two_digit_integers_with_remainder_3_mod_9_l4_4923


namespace find_m_l4_4665

theorem find_m (m : ℤ) :
  (2 * m + 7) * (m - 2) = 51 → m = 5 := by
  sorry

end find_m_l4_4665


namespace true_propositions_identification_l4_4036

-- Definitions related to the propositions
def converse_prop1 (x y : ℝ) := (x + y = 0) → (x + y = 0)
-- Converse of additive inverses: If x and y are additive inverses, then x + y = 0
def converse_prop1_true (x y : ℝ) : Prop := (x + y = 0) → (x + y = 0)

def negation_prop2 : Prop := ¬(∀ (a b c d : ℝ), (a = b → c = d) → (a + b = c + d))
-- Negation of congruent triangles have equal areas: If two triangles are not congruent, areas not equal
def negation_prop2_false : Prop := ¬(∀ (a b c : ℝ), (a = b ∧ b ≠ c → a ≠ c))

def contrapositive_prop3 (q : ℝ) := (q ≤ 1) → (4 - 4 * q ≥ 0)
-- Contrapositive of real roots: If the equation x^2 + 2x + q = 0 does not have real roots then q > 1
def contrapositive_prop3_true (q : ℝ) : Prop := (4 - 4 * q < 0) → (q > 1)

def converse_prop4 (a b c : ℝ) := (a = b ∧ b = c ∧ c = a) → False
-- Converse of scalene triangle: If a triangle has three equal interior angles, it is a scalene triangle
def converse_prop4_false (a b c : ℝ) : Prop := (a = b ∧ b = c ∧ c = a) → False

theorem true_propositions_identification :
  (∀ x y : ℝ, converse_prop1_true x y) ∧
  ¬negation_prop2_false ∧
  (∀ q : ℝ, contrapositive_prop3_true q) ∧
  ¬(∀ a b c : ℝ, converse_prop4_false a b c) := by
  sorry

end true_propositions_identification_l4_4036


namespace Meena_cookies_left_l4_4380

def cookies_initial := 5 * 12
def cookies_sold_to_teacher := 2 * 12
def cookies_bought_by_brock := 7
def cookies_bought_by_katy := 2 * cookies_bought_by_brock

def cookies_left := cookies_initial - cookies_sold_to_teacher - cookies_bought_by_brock - cookies_bought_by_katy

theorem Meena_cookies_left : cookies_left = 15 := 
by 
  -- steps to be proven here
  sorry

end Meena_cookies_left_l4_4380


namespace evaluate_expression_l4_4235

noncomputable def a : ℝ := Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 6
noncomputable def b : ℝ := -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 6
noncomputable def c : ℝ := Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 6
noncomputable def d : ℝ := -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 6

theorem evaluate_expression :
  ( (1 / a) + (1 / b) + (1 / c) + (1 / d) ) ^ 2 = 96 / 529 :=
by
  sorry

end evaluate_expression_l4_4235


namespace fred_games_last_year_l4_4322

def total_games : Nat := 47
def games_this_year : Nat := 36

def games_last_year (total games games this year : Nat) : Nat := total_games - games_this_year

theorem fred_games_last_year : games_last_year total_games games_this_year = 11 :=
by
  sorry

end fred_games_last_year_l4_4322


namespace biking_time_l4_4847

noncomputable def east_bound_speed : ℝ := 22
noncomputable def west_bound_speed : ℝ := east_bound_speed + 4
noncomputable def total_distance : ℝ := 200

theorem biking_time :
  (east_bound_speed + west_bound_speed) * (t : ℝ) = total_distance → t = 25 / 6 :=
by
  -- The proof is omitted and replaced with sorry.
  sorry

end biking_time_l4_4847


namespace boxes_given_away_l4_4846

def total_boxes := 12
def pieces_per_box := 6
def remaining_pieces := 30

theorem boxes_given_away : (total_boxes * pieces_per_box - remaining_pieces) / pieces_per_box = 7 :=
by
  sorry

end boxes_given_away_l4_4846


namespace cheburashkas_erased_l4_4762

theorem cheburashkas_erased (n : ℕ) (rows : ℕ) (krakozyabras : ℕ) 
  (h_spacing : ∀ r, r ≤ rows → krakozyabras = 2 * (n - 1))
  (h_rows : rows = 2)
  (h_krakozyabras : krakozyabras = 29) :
  n = 16 → rows = 2 → krakozyabras = 29 → n = 16 - 5 :=
by
  sorry

end cheburashkas_erased_l4_4762


namespace factorize_expression_l4_4698

theorem factorize_expression (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 :=
by sorry

end factorize_expression_l4_4698


namespace goods_train_speed_l4_4540

theorem goods_train_speed
  (length_train : ℝ)
  (length_platform : ℝ)
  (time_taken : ℝ)
  (speed_kmph : ℝ)
  (h1 : length_train = 240.0416)
  (h2 : length_platform = 280)
  (h3 : time_taken = 26)
  (h4 : speed_kmph = 72.00576) :
  speed_kmph = ((length_train + length_platform) / time_taken) * 3.6 := sorry

end goods_train_speed_l4_4540


namespace amount_of_flour_already_put_in_l4_4618

theorem amount_of_flour_already_put_in 
  (total_flour_needed : ℕ) (flour_remaining : ℕ) (x : ℕ) 
  (h1 : total_flour_needed = 9) 
  (h2 : flour_remaining = 7) 
  (h3 : total_flour_needed - flour_remaining = x) : 
  x = 2 := 
sorry

end amount_of_flour_already_put_in_l4_4618


namespace find_f_l4_4457

noncomputable def f (f'₁ : ℝ) (x : ℝ) : ℝ := f'₁ * Real.exp x - x ^ 2

theorem find_f'₁ (f'₁ : ℝ) (h : f f'₁ = λ x => f'₁ * Real.exp x - x ^ 2) :
  f'₁ = 2 * Real.exp 1 / (Real.exp 1 - 1) := by
  sorry

end find_f_l4_4457


namespace union_complement_eq_univ_l4_4791

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 7}

-- Define set M
def M : Set ℕ := {1, 3, 5, 7}

-- Define set N
def N : Set ℕ := {3, 5}

-- Define the complement of N with respect to U
def complement_U_N : Set ℕ := {1, 2, 4, 7}

-- Prove that U = M ∪ complement_U_N
theorem union_complement_eq_univ : U = M ∪ complement_U_N := 
sorry

end union_complement_eq_univ_l4_4791


namespace buckets_required_l4_4845

variable (C : ℝ) (N : ℝ)

theorem buckets_required (h : N * C = 105 * (2 / 5) * C) : N = 42 := 
  sorry

end buckets_required_l4_4845


namespace number_of_Cheburashkas_erased_l4_4775

theorem number_of_Cheburashkas_erased :
  ∃ (n : ℕ), 
    (∀ x, x ≥ 1 → 
      (let totalKrakozyabras = (2 * (x - 1) = 29) in
         x - 2 = 11)) :=
sorry

end number_of_Cheburashkas_erased_l4_4775


namespace other_factor_of_product_l4_4663

def product_has_factors (n : ℕ) : Prop :=
  ∃ a b c d e f : ℕ, n = (2^a) * (3^b) * (5^c) * (7^d) * (11^e) * (13^f) ∧ a ≥ 4 ∧ b ≥ 3

def smallest_w (x : ℕ) : ℕ :=
  if h : x = 1452 then 468 else 1

theorem other_factor_of_product (w : ℕ) : 
  (product_has_factors (1452 * w)) → (w = 468) :=
by
  sorry

end other_factor_of_product_l4_4663


namespace length_of_flat_terrain_l4_4418

theorem length_of_flat_terrain (total_time : ℚ)
  (total_distance : ℕ)
  (speed_uphill speed_flat speed_downhill : ℚ)
  (distance_uphill distance_flat : ℕ) :
  total_time = 116 / 60 ∧
  total_distance = distance_uphill + distance_flat + (total_distance - distance_uphill - distance_flat) ∧
  speed_uphill = 4 ∧
  speed_flat = 5 ∧
  speed_downhill = 6 ∧
  distance_uphill ≥ 0 ∧
  distance_flat ≥ 0 ∧
  distance_uphill + distance_flat ≤ total_distance →
  distance_flat = 3 := 
by 
  sorry

end length_of_flat_terrain_l4_4418


namespace total_amount_owed_l4_4141

theorem total_amount_owed :
  ∃ (P remaining_balance processing_fee new_total discount: ℝ),
    0.05 * P = 50 ∧
    remaining_balance = P - 50 ∧
    processing_fee = 0.03 * remaining_balance ∧
    new_total = remaining_balance + processing_fee ∧
    discount = 0.10 * new_total ∧
    new_total - discount = 880.65 :=
sorry

end total_amount_owed_l4_4141


namespace factorization_correct_l4_4705

-- Define the initial expression
def initial_expr (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + a

-- Define the factorized form
def factorized_expr (a x : ℝ) : ℝ := a * (x - 1)^2

-- Create the theorem statement
theorem factorization_correct (a x : ℝ) : initial_expr a x = factorized_expr a x :=
by simp [initial_expr, factorized_expr, pow_two, mul_add, add_mul, add_assoc]; sorry

end factorization_correct_l4_4705


namespace max_ab_l4_4907

theorem max_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : ab ≤ 1 / 4 :=
sorry

end max_ab_l4_4907


namespace total_dolls_combined_l4_4163

-- Define the number of dolls for Vera
def vera_dolls : ℕ := 20

-- Define the relationship that Sophie has twice as many dolls as Vera
def sophie_dolls : ℕ := 2 * vera_dolls

-- Define the relationship that Aida has twice as many dolls as Sophie
def aida_dolls : ℕ := 2 * sophie_dolls

-- The statement to prove that the total number of dolls is 140
theorem total_dolls_combined : aida_dolls + sophie_dolls + vera_dolls = 140 :=
by
  sorry

end total_dolls_combined_l4_4163


namespace smallest_hope_number_l4_4930

def is_square (n : ℕ) : Prop := ∃ (k : ℕ), n = k * k
def is_cube (n : ℕ) : Prop := ∃ (k : ℕ), n = k * k * k
def is_fifth_power (n : ℕ) : Prop := ∃ (k : ℕ), n = k * k * k * k * k

def is_hope_number (n : ℕ) : Prop :=
  is_square (n / 8) ∧ is_cube (n / 9) ∧ is_fifth_power (n / 25)

theorem smallest_hope_number : ∃ n, is_hope_number n ∧ n = 2^15 * 3^20 * 5^12 :=
by
  sorry

end smallest_hope_number_l4_4930


namespace temperature_on_friday_is_35_l4_4248

variables (M T W Th F : ℤ)

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem temperature_on_friday_is_35
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (h3 : M = 43)
  (h4 : is_odd M)
  (h5 : is_odd T)
  (h6 : is_odd W)
  (h7 : is_odd Th)
  (h8 : is_odd F) : 
  F = 35 :=
sorry

end temperature_on_friday_is_35_l4_4248


namespace simplify_expression_l4_4526

theorem simplify_expression :
  (-2 : ℝ) ^ 2005 + (-2) ^ 2006 + (3 : ℝ) ^ 2007 - (2 : ℝ) ^ 2008 =
  -7 * (2 : ℝ) ^ 2005 + (3 : ℝ) ^ 2007 := 
by
    sorry

end simplify_expression_l4_4526


namespace ages_of_father_and_daughter_l4_4666

variable (F D : ℕ)

-- Conditions
def condition1 : Prop := F = 4 * D
def condition2 : Prop := F + 20 = 2 * (D + 20)

-- Main statement
theorem ages_of_father_and_daughter (h1 : condition1 F D) (h2 : condition2 F D) : D = 10 ∧ F = 40 := by
  sorry

end ages_of_father_and_daughter_l4_4666


namespace cylindrical_pipe_height_l4_4873

theorem cylindrical_pipe_height (r_outer r_inner : ℝ) (SA : ℝ) (h : ℝ) 
  (h_outer : r_outer = 5)
  (h_inner : r_inner = 3)
  (h_SA : SA = 50 * Real.pi)
  (surface_area_eq: SA = 2 * Real.pi * (r_outer + r_inner) * h) : 
  h = 25 / 8 := 
by
  {
    sorry
  }

end cylindrical_pipe_height_l4_4873


namespace math_problem_l4_4218

variable {x a b : ℝ}

theorem math_problem (h1 : x < a) (h2 : a < 0) (h3 : b = -a) : x^2 > b^2 ∧ b^2 > 0 :=
by {
  sorry
}

end math_problem_l4_4218


namespace equilateral_triangle_perimeter_l4_4637

theorem equilateral_triangle_perimeter (s : ℕ) (h1 : 2 * s + 10 = 50) : 3 * s = 60 :=
sorry

end equilateral_triangle_perimeter_l4_4637


namespace zoe_calories_l4_4132

theorem zoe_calories 
  (s : ℕ) (y : ℕ) (c_s : ℕ) (c_y : ℕ)
  (s_eq : s = 12) (y_eq : y = 6) (cs_eq : c_s = 4) (cy_eq : c_y = 17) :
  s * c_s + y * c_y = 150 :=
by
  sorry

end zoe_calories_l4_4132


namespace common_chord_equation_l4_4586

-- Definition of the first circle
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x = 0

-- Definition of the second circle
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4 * y = 0

-- Proposition stating we need to prove the line equation
theorem common_chord_equation (x y : ℝ) : circle1 x y → circle2 x y → x - y = 0 :=
by
  intros h1 h2
  sorry

end common_chord_equation_l4_4586


namespace locus_of_point_R_l4_4720

theorem locus_of_point_R :
  ∀ (P Q O F R : ℝ × ℝ)
    (hP_on_parabola : ∃ x1 y1, P = (x1, y1) ∧ y1^2 = 2 * x1)
    (h_directrix : Q.1 = -1 / 2)
    (hQ : ∃ x1 y1, Q = (x1, y1) ∧ P = (x1, y1))
    (hO : O = (0, 0))
    (hF : F = (1 / 2, 0))
    (h_intersection : ∃ x y, 
      R = (x, y) ∧
      ∃ x1 y1,
      P = (x1, y1) ∧ 
      y1^2 = 2 * x1 ∧
      ∃ (m_OP : ℝ), 
        m_OP = y1 / x1 ∧ 
        y = m_OP * x ∧
      ∃ (m_FQ : ℝ), 
        m_FQ = -y1 ∧
        y = m_FQ * x + y1 * (1 + 3 / 2)),
  R.2^2 = -2 * R.1^2 + R.1 :=
by sorry

end locus_of_point_R_l4_4720


namespace find_natural_pairs_l4_4315

theorem find_natural_pairs (m n : ℕ) :
  (n * (n - 1) * (n - 2) * (n - 3) = m * (m - 1)) ↔ (n = 1 ∧ m = 1) ∨ (n = 2 ∧ m = 1) ∨ (n = 3 ∧ m = 1) :=
by sorry

end find_natural_pairs_l4_4315


namespace range_of_m_for_one_real_root_l4_4465

def f (x : ℝ) (m : ℝ) : ℝ := x^3 - 3*x + m

theorem range_of_m_for_one_real_root :
  (∃! x : ℝ, f x m = 0) ↔ (m < -2 ∨ m > 2) := by
  sorry

end range_of_m_for_one_real_root_l4_4465


namespace people_distribution_l4_4936

theorem people_distribution (x : ℕ) (h1 : x > 5):
  100 / (x - 5) = 150 / x :=
sorry

end people_distribution_l4_4936


namespace largest_b_l4_4983

def max_b (a b c : ℕ) : ℕ := b -- Define max_b function which outputs b

theorem largest_b (a b c : ℕ)
  (h1 : a * b * c = 360)
  (h2 : 1 < c)
  (h3 : c < b)
  (h4 : b < a) :
  max_b a b c = 10 :=
sorry

end largest_b_l4_4983


namespace toby_deleted_nine_bad_shots_l4_4402

theorem toby_deleted_nine_bad_shots 
  (x : ℕ)
  (h1 : 63 > x)
  (h2 : (63 - x) + 15 - 3 = 84)
  : x = 9 :=
by
  sorry

end toby_deleted_nine_bad_shots_l4_4402


namespace max_obtuse_angles_in_quadrilateral_l4_4224

theorem max_obtuse_angles_in_quadrilateral (a b c d : ℝ) 
  (h₁ : a + b + c + d = 360)
  (h₂ : 90 < a)
  (h₃ : 90 < b)
  (h₄ : 90 < c) :
  90 > d :=
sorry

end max_obtuse_angles_in_quadrilateral_l4_4224


namespace sin_330_eq_neg_half_l4_4034

-- Define conditions as hypotheses in Lean
def angle_330 (θ : ℝ) : Prop := θ = 330
def angle_transform (θ : ℝ) : Prop := θ = 360 - 30
def sin_pos (θ : ℝ) : Prop := Real.sin θ = 1 / 2
def sin_neg_in_4th_quadrant (θ : ℝ) : Prop := θ = 330 -> Real.sin θ < 0

-- The main theorem statement
theorem sin_330_eq_neg_half : ∀ θ : ℝ, angle_330 θ → angle_transform θ → sin_pos 30 → sin_neg_in_4th_quadrant θ → Real.sin θ = -1 / 2 := by
  intro θ h1 h2 h3 h4
  sorry

end sin_330_eq_neg_half_l4_4034


namespace initial_workers_count_l4_4148

theorem initial_workers_count (W : ℕ) 
  (h1 : W * 30 = W * 30) 
  (h2 : W * 15 = (W - 5) * 20)
  (h3 : W > 5) 
  : W = 20 :=
by {
  sorry
}

end initial_workers_count_l4_4148


namespace skew_lines_sufficient_not_necessary_l4_4590

-- Definitions for the conditions
def skew_lines (l1 l2 : Type) : Prop := sorry -- Definition of skew lines
def do_not_intersect (l1 l2 : Type) : Prop := sorry -- Definition of not intersecting

-- The main theorem statement
theorem skew_lines_sufficient_not_necessary (l1 l2 : Type) :
  (skew_lines l1 l2) → (do_not_intersect l1 l2) ∧ ¬ (do_not_intersect l1 l2 → skew_lines l1 l2) :=
by
  sorry

end skew_lines_sufficient_not_necessary_l4_4590


namespace eat_both_veg_nonveg_l4_4223

theorem eat_both_veg_nonveg (total_veg only_veg : ℕ) (h1 : total_veg = 31) (h2 : only_veg = 19) :
  (total_veg - only_veg) = 12 :=
by
  have h3 : total_veg - only_veg = 31 - 19 := by rw [h1, h2]
  exact h3

end eat_both_veg_nonveg_l4_4223


namespace pyramids_from_cuboid_l4_4403

-- Define the vertices of a cuboid
def vertices_of_cuboid : ℕ := 8

-- Define the edges of a cuboid
def edges_of_cuboid : ℕ := 12

-- Define the faces of a cuboid
def faces_of_cuboid : ℕ := 6

-- Define the combinatoric calculation
def combinations (n k : ℕ) : ℕ := (n.choose k)

-- Define the total number of tetrahedrons formed
def total_tetrahedrons : ℕ := combinations 7 3 - faces_of_cuboid * combinations 4 3

-- Define the expected result
def expected_tetrahedrons : ℕ := 106

-- The theorem statement to prove that the total number of tetrahedrons is 106
theorem pyramids_from_cuboid : total_tetrahedrons = expected_tetrahedrons :=
by
  sorry

end pyramids_from_cuboid_l4_4403


namespace factorize_expression_l4_4707

theorem factorize_expression (a x : ℝ) :
  a * x^2 - 2 * a * x + a = a * (x - 1) ^ 2 := 
sorry

end factorize_expression_l4_4707


namespace ratio_of_selling_prices_l4_4670

variable (CP : ℝ)
def SP1 : ℝ := CP * 1.6
def SP2 : ℝ := CP * 0.8

theorem ratio_of_selling_prices : SP2 / SP1 = 1 / 2 := 
by sorry

end ratio_of_selling_prices_l4_4670


namespace product_of_roots_cubic_l4_4172

theorem product_of_roots_cubic :
  ∀ (x : ℝ), (x^3 - 15 * x^2 + 75 * x - 50 = 0) →
    (∃ a b c : ℝ, x = a * b * c ∧ x = 50) :=
by
  sorry

end product_of_roots_cubic_l4_4172


namespace smallest_fraction_greater_than_three_fifths_l4_4423

theorem smallest_fraction_greater_than_three_fifths : 
    ∃ (a b : ℕ), 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 ∧ (a % 1 = 0) ∧ (b % 1 = 0) ∧ (5 * a > 3 * b) ∧ a = 59 :=
by
  sorry

end smallest_fraction_greater_than_three_fifths_l4_4423


namespace min_a_for_increasing_interval_l4_4587

def f (x a : ℝ) : ℝ := x^2 + (a - 2) * x - 1

theorem min_a_for_increasing_interval (a : ℝ) : (∀ x : ℝ, x ≥ 2 → f x a ≤ f (x + 1) a) ↔ a ≥ -2 :=
sorry

end min_a_for_increasing_interval_l4_4587
