import Mathlib

namespace combination_sum_l1592_159207

noncomputable def combination (n r : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem combination_sum :
  combination 3 2 + combination 4 2 + combination 5 2 + combination 6 2 + 
  combination 7 2 + combination 8 2 + combination 9 2 + combination 10 2 = 164 :=
by
  sorry

end combination_sum_l1592_159207


namespace find_c_l1592_159231

/-- Define the conditions given in the problem --/
def parabola_equation (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def vertex_condition (a b c : ℝ) : Prop := 
  ∀ x, parabola_equation a b c x = a * (x - 3)^2 - 1

def passes_through_point (a b c : ℝ) : Prop := 
  parabola_equation a b c 1 = 5

/-- The main statement -/
theorem find_c (a b c : ℝ) 
  (h_vertex : vertex_condition a b c) 
  (h_point : passes_through_point a b c) :
  c = 12.5 :=
sorry

end find_c_l1592_159231


namespace handshakes_l1592_159203

open Nat

theorem handshakes : ∃ x : ℕ, 4 + 3 + 2 + 1 + x = 10 ∧ x = 2 :=
by
  existsi 2
  simp
  sorry

end handshakes_l1592_159203


namespace problem_inequality_l1592_159223

variable {a b c d : ℝ}

theorem problem_inequality (h1 : 0 ≤ a) (h2 : 0 ≤ d) (h3 : 0 < b) (h4 : 0 < c) (h5 : b + c ≥ a + d) :
  (b / (c + d)) + (c / (b + a)) ≥ (Real.sqrt 2) - (1 / 2) := 
sorry

end problem_inequality_l1592_159223


namespace solution_set_ln_inequality_l1592_159211

noncomputable def f (x : ℝ) := Real.cos x - 4 * x^2

theorem solution_set_ln_inequality :
  {x : ℝ | 0 < x ∧ x < Real.exp (-Real.pi / 2)} ∪ {x : ℝ | x > Real.exp (Real.pi / 2)} =
  {x : ℝ | f (Real.log x) + Real.pi^2 > 0} :=
by
  sorry

end solution_set_ln_inequality_l1592_159211


namespace least_area_of_square_l1592_159254

theorem least_area_of_square :
  ∀ (s : ℝ), (3.5 ≤ s ∧ s < 4.5) → (s * s ≥ 12.25) :=
by
  intro s
  intro hs
  sorry

end least_area_of_square_l1592_159254


namespace num_of_nickels_l1592_159200

theorem num_of_nickels (x : ℕ) (hx_eq_dimes : ∀ n, n = x → n = x) (hx_eq_quarters : ∀ n, n = x → n = 2 * x) (total_value : 5 * x + 10 * x + 50 * x = 1950) : x = 30 :=
sorry

end num_of_nickels_l1592_159200


namespace no_prime_sum_seventeen_l1592_159242

def is_prime (n : ℕ) : Prop := n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_sum_seventeen :
  ¬ ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 17 := by
  sorry

end no_prime_sum_seventeen_l1592_159242


namespace cost_of_each_ticket_l1592_159206

theorem cost_of_each_ticket (x : ℝ) : 
  500 * x * 0.70 = 4 * 2625 → x = 30 :=
by 
  sorry

end cost_of_each_ticket_l1592_159206


namespace examine_points_l1592_159276

variable (Bryan Jen Sammy mistakes : ℕ)

def problem_conditions : Prop :=
  Bryan = 20 ∧ Jen = Bryan + 10 ∧ Sammy = Jen - 2 ∧ mistakes = 7

theorem examine_points (h : problem_conditions Bryan Jen Sammy mistakes) : ∃ total_points : ℕ, total_points = Sammy + mistakes :=
by {
  sorry
}

end examine_points_l1592_159276


namespace num_initial_pairs_of_shoes_l1592_159297

theorem num_initial_pairs_of_shoes (lost_shoes remaining_pairs : ℕ)
  (h1 : lost_shoes = 9)
  (h2 : remaining_pairs = 20) :
  (initial_pairs : ℕ) = 25 :=
sorry

end num_initial_pairs_of_shoes_l1592_159297


namespace prism_unique_triple_l1592_159244

theorem prism_unique_triple :
  ∃! (a b c : ℕ), a ≤ b ∧ b ≤ c ∧ b = 2000 ∧
                  (∃ b' c', b' = 2000 ∧ c' = 2000 ∧
                  (∃ k : ℚ, k = 1/2 ∧
                  (∃ x y z, x = a / 2 ∧ y = 1000 ∧ z = c / 2 ∧ a = 2000 ∧ c = 2000)))
/- The proof is omitted for this statement. -/
:= sorry

end prism_unique_triple_l1592_159244


namespace power_of_m_divisible_by_33_l1592_159214

theorem power_of_m_divisible_by_33 (m : ℕ) (h : m > 0) (k : ℕ) (h_pow : (m ^ k) % 33 = 0) :
  ∃ n, n > 0 ∧ 11 ∣ m ^ n :=
by
  sorry

end power_of_m_divisible_by_33_l1592_159214


namespace birds_on_fence_l1592_159260

theorem birds_on_fence (B : ℕ) : ∃ B, (∃ S, S = 6 ∧ S = (B + 3) + 1) → B = 2 :=
by
  sorry

end birds_on_fence_l1592_159260


namespace tan_rewrite_l1592_159230

open Real

theorem tan_rewrite (α β : ℝ) 
  (h1 : tan (α + β) = 2 / 5)
  (h2 : tan (β - π / 4) = 1 / 4) : 
  (1 + tan α) / (1 - tan α) = 3 / 22 := 
by
  sorry

end tan_rewrite_l1592_159230


namespace car_travel_distance_l1592_159243

variable (b t : Real)
variable (h1 : b > 0)
variable (h2 : t > 0)

theorem car_travel_distance (b t : Real) (h1 : b > 0) (h2 : t > 0) :
  let rate := b / 4
  let inches_in_yard := 36
  let time_in_seconds := 5 * 60
  let distance_in_inches := (rate / t) * time_in_seconds
  let distance_in_yards := distance_in_inches / inches_in_yard
  distance_in_yards = (25 * b) / (12 * t) := by
  sorry

end car_travel_distance_l1592_159243


namespace eccentricity_of_hyperbola_l1592_159236

noncomputable def hyperbola (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1)

noncomputable def foci_condition (a b : ℝ) (c : ℝ) : Prop :=
  c = Real.sqrt (a^2 + b^2)

noncomputable def trisection_condition (a b c : ℝ) : Prop :=
  2 * c = 6 * a^2 / c

theorem eccentricity_of_hyperbola (a b c e : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (hc : c = Real.sqrt (a^2 + b^2)) (ht : 2 * c = 6 * a^2 / c) :
  e = Real.sqrt 3 :=
by
  apply sorry

end eccentricity_of_hyperbola_l1592_159236


namespace square_inequality_not_sufficient_nor_necessary_for_cube_inequality_l1592_159281

variable {a b : ℝ}

theorem square_inequality_not_sufficient_nor_necessary_for_cube_inequality (a b : ℝ) :
  (a^2 > b^2) ↔ (a^3 > b^3) = false :=
sorry

end square_inequality_not_sufficient_nor_necessary_for_cube_inequality_l1592_159281


namespace algebraic_expression_evaluation_l1592_159265

theorem algebraic_expression_evaluation (x : ℝ) (h : x^2 + x - 3 = 0) : x^3 + 2 * x^2 - 2 * x + 2 = 5 :=
by
  sorry

end algebraic_expression_evaluation_l1592_159265


namespace total_children_l1592_159253

theorem total_children {x y : ℕ} (h₁ : x = 18) (h₂ : y = 12) 
  (h₃ : x + y = 30) (h₄ : x = 18) (h₅ : y = 12) : 2 * x + 3 * y = 72 := 
by
  sorry

end total_children_l1592_159253


namespace find_quadruple_l1592_159221

/-- Problem Statement:
Given distinct positive integers a, b, c, and d such that a + b = c * d and a * b = c + d,
find the quadruple (a, b, c, d) that meets these conditions.
-/

theorem find_quadruple :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
            0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
            (a + b = c * d) ∧ (a * b = c + d) ∧
            ((a, b, c, d) = (1, 5, 3, 2) ∨ (a, b, c, d) = (1, 5, 2, 3) ∨
             (a, b, c, d) = (5, 1, 3, 2) ∨ (a, b, c, d) = (5, 1, 2, 3) ∨
             (a, b, c, d) = (2, 3, 1, 5) ∨ (a, b, c, d) = (3, 2, 1, 5) ∨
             (a, b, c, d) = (2, 3, 5, 1) ∨ (a, b, c, d) = (3, 2, 5, 1)) :=
sorry

end find_quadruple_l1592_159221


namespace product_of_numbers_l1592_159233

variable (x y z : ℝ)

theorem product_of_numbers :
  x + y + z = 36 ∧ x = 3 * (y + z) ∧ y = 6 * z → x * y * z = 268 := 
by
  sorry

end product_of_numbers_l1592_159233


namespace greatest_x_integer_l1592_159271

theorem greatest_x_integer (x : ℤ) : 
  (∃ k : ℤ, (x^2 + 4 * x + 9) = k * (x - 4)) ↔ x ≤ 5 :=
by
  sorry

end greatest_x_integer_l1592_159271


namespace total_revenue_full_price_tickets_l1592_159268

theorem total_revenue_full_price_tickets (f q : ℕ) (p : ℝ) :
  f + q = 170 ∧ f * p + q * (p / 4) = 2917 → f * p = 1748 := by
  sorry

end total_revenue_full_price_tickets_l1592_159268


namespace rectangle_area_l1592_159286

-- Definitions
def perimeter (l w : ℝ) : ℝ := 2 * (l + w)
def length (w : ℝ) : ℝ := 2 * w
def area (l w : ℝ) : ℝ := l * w

-- Main Statement
theorem rectangle_area (w l : ℝ) (h_p : perimeter l w = 120) (h_l : l = length w) :
  area l w = 800 :=
by
  sorry

end rectangle_area_l1592_159286


namespace ellipse_equation_hyperbola_equation_l1592_159279

/-- Ellipse problem -/
def ellipse_eq (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

theorem ellipse_equation (e a c b : ℝ) (h_c : c = 3) (h_e : e = 0.5) (h_a : a = 6) (h_b : b^2 = 27) :
  ellipse_eq x y a b := 
sorry

/-- Hyperbola problem -/
def hyperbola_eq (x y a b : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

theorem hyperbola_equation (a b c : ℝ) 
  (h_c : c = 6) 
  (h_A : ∀ (x y : ℝ), (x, y) = (-5, 2) → hyperbola_eq x y a b) 
  (h_eq1 : a^2 + b^2 = 36) 
  (h_eq2 : 25 / (a^2) - 4 / (b^2) = 1) :
  hyperbola_eq x y a b :=
sorry

end ellipse_equation_hyperbola_equation_l1592_159279


namespace hyperbola_eqn_correct_l1592_159285

def parabola_focus : ℝ × ℝ := (1, 0)

def hyperbola_vertex := parabola_focus

def hyperbola_eccentricity : ℝ := 2

def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 - (y^2 / 3) = 1

theorem hyperbola_eqn_correct (x y : ℝ) :
  hyperbola_equation x y :=
sorry

end hyperbola_eqn_correct_l1592_159285


namespace f_at_3_l1592_159291

theorem f_at_3 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 2) = x + 3) : f 3 = 4 := 
sorry

end f_at_3_l1592_159291


namespace express_x_in_terms_of_y_l1592_159201

theorem express_x_in_terms_of_y (x y : ℝ) (h : 3 * x - 4 * y = 8) : x = (4 * y + 8) / 3 :=
sorry

end express_x_in_terms_of_y_l1592_159201


namespace vector_on_plane_l1592_159259

-- Define the vectors w and the condition for proj_w v
def w : ℝ × ℝ × ℝ := (3, -3, 3)
def v (x y z : ℝ) : ℝ × ℝ × ℝ := (x, y, z)
def projection_condition (x y z : ℝ) : Prop :=
  ((3 * x - 3 * y + 3 * z) / 27) * 3 = 6 ∧ ((3 * x - 3 * y + 3 * z) / 27) * (-3) = -6 ∧ ((3 * x - 3 * y + 3 * z) / 27) * 3 = 6

-- Define the plane equation
def plane_eq (x y z : ℝ) : Prop := x - y + z - 18 = 0

-- Prove that the set of vectors v lies on the plane
theorem vector_on_plane (x y z : ℝ) (h : projection_condition x y z) : plane_eq x y z :=
  sorry

end vector_on_plane_l1592_159259


namespace hank_donates_90_percent_l1592_159219

theorem hank_donates_90_percent (x : ℝ) : 
  (100 * x + 0.75 * 80 + 50 = 200) → (x = 0.9) :=
by
  intro h
  sorry

end hank_donates_90_percent_l1592_159219


namespace smallest_even_piece_to_stop_triangle_l1592_159248

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

end smallest_even_piece_to_stop_triangle_l1592_159248


namespace symmetric_circle_equation_l1592_159295

theorem symmetric_circle_equation :
  (∀ x y : ℝ, (x - 1) ^ 2 + y ^ 2 = 1 ↔ x ^ 2 + (y + 1) ^ 2 = 1) :=
by sorry

end symmetric_circle_equation_l1592_159295


namespace expression_simplification_l1592_159213

theorem expression_simplification :
  (- (1 / 2)) ^ 2023 * 2 ^ 2024 = -2 :=
by
  sorry

end expression_simplification_l1592_159213


namespace fraction_orange_juice_in_large_container_l1592_159272

-- Definitions according to the conditions
def pitcher1_capacity : ℕ := 800
def pitcher2_capacity : ℕ := 500
def pitcher1_fraction_orange_juice : ℚ := 1 / 4
def pitcher2_fraction_orange_juice : ℚ := 3 / 5

-- Prove the fraction of orange juice
theorem fraction_orange_juice_in_large_container :
  ( (pitcher1_capacity * pitcher1_fraction_orange_juice + pitcher2_capacity * pitcher2_fraction_orange_juice) / 
    (pitcher1_capacity + pitcher2_capacity) ) = 5 / 13 :=
by
  sorry

end fraction_orange_juice_in_large_container_l1592_159272


namespace curve_is_circle_l1592_159227

theorem curve_is_circle (s : ℝ) :
  let x := (3 - s^2) / (3 + s^2)
  let y := (4 * s) / (3 + s^2)
  x^2 + y^2 = 1 :=
by
  let x := (3 - s^2) / (3 + s^2)
  let y := (4 * s) / (3 + s^2)
  sorry

end curve_is_circle_l1592_159227


namespace pyramid_base_edge_length_l1592_159257

theorem pyramid_base_edge_length 
(radius_hemisphere height_pyramid : ℝ)
(h_radius : radius_hemisphere = 4)
(h_height : height_pyramid = 10)
(h_tangent : ∀ face : ℝ, True) : 
∃ s : ℝ, s = 2 * Real.sqrt 42 :=
by
  sorry

end pyramid_base_edge_length_l1592_159257


namespace sin_45_deg_l1592_159299

theorem sin_45_deg : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by 
  -- placeholder for the actual proof
  sorry

end sin_45_deg_l1592_159299


namespace find_k_of_sequence_l1592_159247

theorem find_k_of_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) (hS : ∀ n, S n = n^2 - 9 * n)
  (hS_recurr : ∀ n ≥ 2, a n = S n - S (n-1)) (h_a_k : ∃ k, 5 < a k ∧ a k < 8) : ∃ k, k = 8 :=
by
  sorry

end find_k_of_sequence_l1592_159247


namespace average_tomatoes_per_day_l1592_159293

theorem average_tomatoes_per_day :
  let t₁ := 120
  let t₂ := t₁ + 50
  let t₃ := 2 * t₂
  let t₄ := t₁ / 2
  (t₁ + t₂ + t₃ + t₄) / 4 = 172.5 := by
  sorry

end average_tomatoes_per_day_l1592_159293


namespace greatest_multiple_of_4_less_than_100_l1592_159209

theorem greatest_multiple_of_4_less_than_100 : ∃ n : ℕ, n % 4 = 0 ∧ n < 100 ∧ ∀ m : ℕ, (m % 4 = 0 ∧ m < 100) → m ≤ n 
:= by
  sorry

end greatest_multiple_of_4_less_than_100_l1592_159209


namespace find_second_smallest_odd_number_l1592_159202

theorem find_second_smallest_odd_number (x : ℤ) (h : (x + (x + 2) + (x + 4) + (x + 6) = 112)) : (x + 2 = 27) :=
sorry

end find_second_smallest_odd_number_l1592_159202


namespace ducks_remaining_after_three_nights_l1592_159273

def initial_ducks : ℕ := 320
def first_night_ducks (initial_ducks : ℕ) : ℕ := initial_ducks - (initial_ducks / 4)
def second_night_ducks (first_night_ducks : ℕ) : ℕ := first_night_ducks - (first_night_ducks / 6)
def third_night_ducks (second_night_ducks : ℕ) : ℕ := second_night_ducks - (second_night_ducks * 30 / 100)

theorem ducks_remaining_after_three_nights : 
  third_night_ducks (second_night_ducks (first_night_ducks initial_ducks)) = 140 :=
by
  -- Proof goes here
  sorry

end ducks_remaining_after_three_nights_l1592_159273


namespace characterize_set_A_l1592_159292

open Int

noncomputable def A : Set ℤ := { x | x^2 - 3 * x - 4 < 0 }

theorem characterize_set_A : A = {0, 1, 2, 3} :=
by
  sorry

end characterize_set_A_l1592_159292


namespace solve_exponential_eq_l1592_159246

theorem solve_exponential_eq (x : ℝ) : 
  ((5 - 2 * x)^(x + 1) = 1) ↔ (x = -1 ∨ x = 2 ∨ x = 3) := by
  sorry

end solve_exponential_eq_l1592_159246


namespace max_super_bishops_l1592_159216

/--
A "super-bishop" attacks another "super-bishop" if they are on the
same diagonal, there are no pieces between them, and the next cell
along the diagonal after the "super-bishop" B is empty. Given these
conditions, prove that the maximum number of "super-bishops" that can
be placed on a standard 8x8 chessboard such that each one attacks at
least one other is 32.
-/
theorem max_super_bishops (n : ℕ) (chessboard : ℕ → ℕ → Prop) (super_bishop : ℕ → ℕ → Prop)
  (attacks : ∀ {x₁ y₁ x₂ y₂}, super_bishop x₁ y₁ → super_bishop x₂ y₂ →
            (x₁ - x₂ = y₁ - y₂ ∨ x₁ + y₁ = x₂ + y₂) →
            (∀ x y, super_bishop x y → (x < min x₁ x₂ ∨ x > max x₁ x₂ ∨ y < min y₁ y₂ ∨ y > max y₁ y₂)) →
            chessboard (x₂ + (x₁ - x₂)) (y₂ + (y₁ - y₂))) :
  ∃ k, k = 32 ∧ (∀ x y, super_bishop x y → x < 8 ∧ y < 8) → k ≤ n :=
sorry

end max_super_bishops_l1592_159216


namespace volume_of_sphere_l1592_159234

theorem volume_of_sphere
  (r : ℝ) (V : ℝ)
  (h₁ : r = 1/3)
  (h₂ : 2 * r = (16/9 * V)^(1/3)) :
  V = 1/6 :=
  sorry

end volume_of_sphere_l1592_159234


namespace chips_needed_per_console_l1592_159284

-- Definitions based on the conditions
def chips_per_day : ℕ := 467
def consoles_per_day : ℕ := 93

-- The goal is to prove that each video game console needs 5 computer chips
theorem chips_needed_per_console : chips_per_day / consoles_per_day = 5 :=
by sorry

end chips_needed_per_console_l1592_159284


namespace number_of_children_l1592_159274

-- Definitions of given conditions
def total_passengers := 170
def men := 90
def women := men / 2
def adults := men + women
def children := total_passengers - adults

-- Theorem statement
theorem number_of_children : children = 35 :=
by
  sorry

end number_of_children_l1592_159274


namespace fgf_3_equals_108_l1592_159289

def f (x : ℕ) : ℕ := 2 * x + 4
def g (x : ℕ) : ℕ := 5 * x + 2

theorem fgf_3_equals_108 : f (g (f 3)) = 108 := 
by
  sorry

end fgf_3_equals_108_l1592_159289


namespace trig_identity_l1592_159258

theorem trig_identity : 
  ( 4 * Real.sin (40 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) / Real.cos (20 * Real.pi / 180) 
   - Real.tan (20 * Real.pi / 180) ) = Real.sqrt 3 := 
by
  sorry

end trig_identity_l1592_159258


namespace incorrect_statement_C_l1592_159217

theorem incorrect_statement_C :
  (∀ r : ℚ, ∃ p : ℝ, p = r) ∧  -- Condition A: All rational numbers can be represented by points on the number line.
  (∀ x : ℝ, x = 1 / x → x = 1 ∨ x = -1) ∧  -- Condition B: The reciprocal of a number equal to itself is ±1.
  (∀ f : ℚ, ∃ q : ℝ, q = f) →  -- Condition C (negation of C as presented): Fractions cannot be represented by points on the number line.
  (∀ x : ℝ, abs x ≥ 0) ∧ (∀ x : ℝ, abs x = 0 ↔ x = 0) →  -- Condition D: The number with the smallest absolute value is 0.
  false :=                      -- Prove that statement C is incorrect
by
  sorry

end incorrect_statement_C_l1592_159217


namespace y_intercept_of_line_l1592_159283

def line_equation (x y : ℝ) : Prop := x - 2 * y + 4 = 0

theorem y_intercept_of_line : ∀ y : ℝ, line_equation 0 y → y = 2 :=
by 
  intro y h
  unfold line_equation at h
  sorry

end y_intercept_of_line_l1592_159283


namespace bench_cost_l1592_159228

theorem bench_cost (B : ℕ) (h : B + 2 * B = 450) : B = 150 :=
by {
  sorry
}

end bench_cost_l1592_159228


namespace correct_value_of_a_l1592_159232

namespace ProofProblem

-- Condition 1: Definition of set M
def M : Set ℤ := {x | x^2 ≤ 1}

-- Condition 2: Definition of set N dependent on a parameter a
def N (a : ℤ) : Set ℤ := {a, a * a}

-- Question translated: Correct value of a such that M ∪ N = M
theorem correct_value_of_a (a : ℤ) : (M ∪ N a = M) → a = -1 :=
by
  sorry

end ProofProblem

end correct_value_of_a_l1592_159232


namespace part_a_7_pieces_l1592_159220

theorem part_a_7_pieces (grid : Fin 4 × Fin 4 → Prop) (h : ∀ i j, ∃ n, grid (i, j) → n < 7)
  (hnoTwoInSameCell : ∀ (i₁ i₂ : Fin 4) (j₁ j₂ : Fin 4), (i₁, j₁) ≠ (i₂, j₂) → grid (i₁, j₁) ≠ grid (i₂, j₂))
  : ∀ (rowsRemoved colsRemoved : Finset (Fin 4)), rowsRemoved.card = 2 → colsRemoved.card = 2
    → ∃ i j, ¬ grid (i, j) := by sorry

end part_a_7_pieces_l1592_159220


namespace pups_more_than_adults_l1592_159250

-- Define the counts of dogs
def H := 5  -- number of huskies
def P := 2  -- number of pitbulls
def G := 4  -- number of golden retrievers

-- Define the number of pups each type of dog had
def pups_per_husky_and_pitbull := 3
def additional_pups_per_golden_retriever := 2
def pups_per_golden_retriever := pups_per_husky_and_pitbull + additional_pups_per_golden_retriever

-- Calculate the total number of pups
def total_pups := H * pups_per_husky_and_pitbull + P * pups_per_husky_and_pitbull + G * pups_per_golden_retriever

-- Calculate the total number of adult dogs
def total_adult_dogs := H + P + G

-- Prove that the number of pups is 30 more than the number of adult dogs
theorem pups_more_than_adults : total_pups - total_adult_dogs = 30 :=
by
  -- fill in the proof later
  sorry

end pups_more_than_adults_l1592_159250


namespace five_fourths_of_twelve_fifths_eq_three_l1592_159266

theorem five_fourths_of_twelve_fifths_eq_three : (5 : ℝ) / 4 * (12 / 5) = 3 := 
by 
  sorry

end five_fourths_of_twelve_fifths_eq_three_l1592_159266


namespace max_value_x_plus_2y_l1592_159240

variable (x y : ℝ)
variable (h1 : 4 * x + 3 * y ≤ 12)
variable (h2 : 3 * x + 6 * y ≤ 9)

theorem max_value_x_plus_2y : x + 2 * y ≤ 3 := by
  sorry

end max_value_x_plus_2y_l1592_159240


namespace k_of_neg7_l1592_159280

noncomputable def h (x : ℝ) : ℝ := 4 * x - 9
noncomputable def k (x : ℝ) : ℝ := 3 * x^2 + 4 * x - 2

theorem k_of_neg7 : k (-7) = 3 / 4 :=
by
  sorry

end k_of_neg7_l1592_159280


namespace count_ordered_pairs_l1592_159252

theorem count_ordered_pairs (x y : ℕ) (px : 0 < x) (py : 0 < y) (h : 2310 = 2 * 3 * 5 * 7 * 11) :
  (x * y = 2310 → ∃ n : ℕ, n = 32) :=
by
  sorry

end count_ordered_pairs_l1592_159252


namespace mike_corvette_average_speed_l1592_159222

theorem mike_corvette_average_speed
  (D : ℚ) (v : ℚ) (total_distance : ℚ)
  (first_half_distance : ℚ) (second_half_time_ratio : ℚ)
  (total_time : ℚ) (average_rate : ℚ) :
  total_distance = 640 ∧
  first_half_distance = total_distance / 2 ∧
  second_half_time_ratio = 3 ∧
  average_rate = 40 →
  v = 80 :=
by
  intros h
  have total_distance_eq : total_distance = 640 := h.1
  have first_half_distance_eq : first_half_distance = total_distance / 2 := h.2.1
  have second_half_time_ratio_eq : second_half_time_ratio = 3 := h.2.2.1
  have average_rate_eq : average_rate = 40 := h.2.2.2
  sorry

end mike_corvette_average_speed_l1592_159222


namespace factor_polynomial_l1592_159288

theorem factor_polynomial (a : ℝ) : 74 * a^2 + 222 * a + 148 * a^3 = 74 * a * (2 * a^2 + a + 3) :=
by
  sorry

end factor_polynomial_l1592_159288


namespace tom_has_65_fruits_left_l1592_159212

def initial_fruits : ℕ := 40 + 70 + 30 + 15

def sold_oranges : ℕ := (1 / 4) * 40
def sold_apples : ℕ := (2 / 3) * 70
def sold_bananas : ℕ := (5 / 6) * 30
def sold_kiwis : ℕ := (60 / 100) * 15

def fruits_remaining : ℕ :=
  40 - sold_oranges +
  70 - sold_apples +
  30 - sold_bananas +
  15 - sold_kiwis

theorem tom_has_65_fruits_left :
  fruits_remaining = 65 := by
  sorry

end tom_has_65_fruits_left_l1592_159212


namespace solve_system_l1592_159282

theorem solve_system (x y z a b c : ℝ)
  (h1 : x * (x + y + z) = a^2)
  (h2 : y * (x + y + z) = b^2)
  (h3 : z * (x + y + z) = c^2) :
  (x = a^2 / Real.sqrt (a^2 + b^2 + c^2) ∨ x = -a^2 / Real.sqrt (a^2 + b^2 + c^2)) ∧
  (y = b^2 / Real.sqrt (a^2 + b^2 + c^2) ∨ y = -b^2 / Real.sqrt (a^2 + b^2 + c^2)) ∧
  (z = c^2 / Real.sqrt (a^2 + b^2 + c^2) ∨ z = -c^2 / Real.sqrt (a^2 + b^2 + c^2)) :=
by
  sorry

end solve_system_l1592_159282


namespace midpoint_quadrilateral_inequality_l1592_159290

theorem midpoint_quadrilateral_inequality 
  (A B C D E F G H : ℝ) 
  (S_ABCD : ℝ)
  (midpoints_A : E = (A + B) / 2)
  (midpoints_B : F = (B + C) / 2)
  (midpoints_C : G = (C + D) / 2)
  (midpoints_D : H = (D + A) / 2)
  (EG : ℝ)
  (HF : ℝ) :
  S_ABCD ≤ EG * HF ∧ EG * HF ≤ (B + D) * (A + C) / 4 := by
  sorry

end midpoint_quadrilateral_inequality_l1592_159290


namespace number_triangle_value_of_n_l1592_159298

theorem number_triangle_value_of_n:
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x * y = 2022 ∧ (∃ n : ℕ, n > 0 ∧ n^2 ∣ 2022 ∧ n = 1) :=
by sorry

end number_triangle_value_of_n_l1592_159298


namespace leif_fruit_weight_difference_l1592_159261

theorem leif_fruit_weight_difference :
  let apples_ounces := 27.5
  let grams_per_ounce := 28.35
  let apples_grams := apples_ounces * grams_per_ounce
  let dozens_oranges := 5.5
  let oranges_per_dozen := 12
  let total_oranges := dozens_oranges * oranges_per_dozen
  let weight_per_orange := 45
  let oranges_grams := total_oranges * weight_per_orange
  let weight_difference := oranges_grams - apples_grams
  weight_difference = 2190.375 := by
{
  sorry
}

end leif_fruit_weight_difference_l1592_159261


namespace concert_ticket_revenue_l1592_159218

theorem concert_ticket_revenue :
  let price_student : ℕ := 9
  let price_non_student : ℕ := 11
  let total_tickets : ℕ := 2000
  let student_tickets : ℕ := 520
  let non_student_tickets := total_tickets - student_tickets
  let revenue_student := student_tickets * price_student
  let revenue_non_student := non_student_tickets * price_non_student
  revenue_student + revenue_non_student = 20960 :=
by
  -- Definitions
  let price_student := 9
  let price_non_student := 11
  let total_tickets := 2000
  let student_tickets := 520
  let non_student_tickets := total_tickets - student_tickets
  let revenue_student := student_tickets * price_student
  let revenue_non_student := non_student_tickets * price_non_student
  -- Proof
  sorry  -- Placeholder for the proof

end concert_ticket_revenue_l1592_159218


namespace sufficient_but_not_necessary_condition_l1592_159251

theorem sufficient_but_not_necessary_condition (x : ℝ) (p : -1 < x ∧ x < 3) (q : x^2 - 5 * x - 6 < 0) : 
  (-1 < x ∧ x < 3) → (x^2 - 5 * x - 6 < 0) ∧ ¬((x^2 - 5 * x - 6 < 0) → (-1 < x ∧ x < 3)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l1592_159251


namespace European_to_American_swallow_ratio_l1592_159264

theorem European_to_American_swallow_ratio (a e : ℝ) (n_E : ℕ) 
  (h1 : a = 5)
  (h2 : 2 * n_E + n_E = 90)
  (h3 : 60 * a + 30 * e = 600) :
  e / a = 2 := 
by
  sorry

end European_to_American_swallow_ratio_l1592_159264


namespace carlos_gold_quarters_l1592_159241

theorem carlos_gold_quarters (quarter_weight : ℚ) 
  (store_value_per_quarter : ℚ) 
  (melt_value_per_ounce : ℚ) 
  (quarters_per_ounce : ℚ := 1 / quarter_weight) 
  (spent_value : ℚ := quarters_per_ounce * store_value_per_quarter)
  (melted_value: ℚ := melt_value_per_ounce) :
  quarter_weight = 1/5 ∧ store_value_per_quarter = 0.25 ∧ melt_value_per_ounce = 100 → 
  melted_value / spent_value = 80 := 
by
  intros h
  sorry

end carlos_gold_quarters_l1592_159241


namespace hcl_formed_l1592_159267

-- Define the balanced chemical equation as a relationship between reactants and products
def balanced_equation (m_C2H6 m_Cl2 m_CCl4 m_HCl : ℝ) :=
  m_C2H6 + 4 * m_Cl2 = m_CCl4 + 6 * m_HCl

-- Define the problem-specific values
def reaction_given (m_C2H6 m_Cl2 m_CCl4 m_HCl : ℝ) :=
  m_C2H6 = 3 ∧ m_Cl2 = 21 ∧ m_CCl4 = 6 ∧ balanced_equation m_C2H6 m_Cl2 m_CCl4 m_HCl

-- Prove the number of moles of HCl formed
theorem hcl_formed : ∃ (m_HCl : ℝ), reaction_given 3 21 6 m_HCl ∧ m_HCl = 18 :=
by
  sorry

end hcl_formed_l1592_159267


namespace LittleRedHeightCorrect_l1592_159226

noncomputable def LittleRedHeight : ℝ :=
let LittleMingHeight := 1.3 
let HeightDifference := 0.2 
LittleMingHeight - HeightDifference

theorem LittleRedHeightCorrect : LittleRedHeight = 1.1 := by
  sorry

end LittleRedHeightCorrect_l1592_159226


namespace circle_equation_line_intersect_circle_l1592_159205

theorem circle_equation (x y : ℝ) : 
  y = x^2 - 4*x + 3 → (x = 0 ∧ y = 3) ∨ (y = 0 ∧ (x = 1 ∨ x = 3)) :=
sorry

theorem line_intersect_circle (m : ℝ) :
  (∀ x y : ℝ, (x + y + m = 0) ∨ ((x - 2)^2 + (y - 2)^2 = 5)) →
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁ + y₁ + m = 0) → ((x₁ - 2)^2 + (y₁ - 2)^2 = 5) →
    (x₂ + y₂ + m = 0) → ((x₂ - 2)^2 + (y₂ - 2)^2 = 5) →
    ((x₁ * x₂ + y₁ * y₂ = 0) → (m = -1 ∨ m = -3))) :=
sorry

end circle_equation_line_intersect_circle_l1592_159205


namespace team_air_conditioner_installation_l1592_159235

theorem team_air_conditioner_installation (x : ℕ) (y : ℕ) 
  (h1 : 66 % x = 0) 
  (h2 : 60 % y = 0) 
  (h3 : x = y + 2) 
  (h4 : 66 / x = 60 / y) 
  : x = 22 ∧ y = 20 :=
by
  have h5 : x = 22 := sorry
  have h6 : y = 20 := sorry
  exact ⟨h5, h6⟩

end team_air_conditioner_installation_l1592_159235


namespace avg_annual_reduction_l1592_159239

theorem avg_annual_reduction (x : ℝ) (hx : (1 - x)^2 = 0.64) : x = 0.2 :=
by
  sorry

end avg_annual_reduction_l1592_159239


namespace C_investment_value_is_correct_l1592_159262

noncomputable def C_investment_contribution 
  (A_investment B_investment total_profit A_profit_share : ℝ) : ℝ :=
  let C_investment := 
    (A_profit_share * (A_investment + B_investment) - A_investment * total_profit) / 
    (total_profit - A_profit_share)
  C_investment

theorem C_investment_value_is_correct : 
  C_investment_contribution 6300 4200 13600 4080 = 10500 := 
by
  unfold C_investment_contribution
  norm_num
  sorry

end C_investment_value_is_correct_l1592_159262


namespace no_solution_for_x4_plus_y4_eq_z4_l1592_159263

theorem no_solution_for_x4_plus_y4_eq_z4 :
  ∀ (x y z : ℤ), x ≠ 0 → y ≠ 0 → z ≠ 0 → gcd (gcd x y) z = 1 → x^4 + y^4 ≠ z^4 :=
sorry

end no_solution_for_x4_plus_y4_eq_z4_l1592_159263


namespace correct_sample_size_l1592_159238

variable {StudentScore : Type} {scores : Finset StudentScore} (extract_sample : Finset StudentScore → Finset StudentScore)

noncomputable def is_correct_statement : Prop :=
  ∀ (total_scores : Finset StudentScore) (sample_scores : Finset StudentScore),
  (total_scores.card = 1000) →
  (extract_sample total_scores = sample_scores) →
  (sample_scores.card = 100) →
  sample_scores.card = 100

theorem correct_sample_size (total_scores sample_scores : Finset StudentScore)
  (H_total : total_scores.card = 1000)
  (H_sample : extract_sample total_scores = sample_scores)
  (H_card : sample_scores.card = 100) :
  sample_scores.card = 100 :=
sorry

end correct_sample_size_l1592_159238


namespace product_remainder_mod_7_l1592_159229

theorem product_remainder_mod_7 (a b c : ℕ) (ha : a % 7 = 2) (hb : b % 7 = 3) (hc : c % 7 = 5) :
    (a * b * c) % 7 = 2 :=
by
  sorry

end product_remainder_mod_7_l1592_159229


namespace system_has_infinitely_many_solutions_l1592_159277

theorem system_has_infinitely_many_solutions :
  ∃ (S : Set (ℝ × ℝ × ℝ)), (∀ x y z : ℝ, (x + y = 2 ∧ xy - z^2 = 1) ↔ (x, y, z) ∈ S) ∧ S.Infinite :=
by
  sorry

end system_has_infinitely_many_solutions_l1592_159277


namespace common_ratio_of_geometric_series_l1592_159287

theorem common_ratio_of_geometric_series (a S r : ℝ) (h1 : a = 500) (h2 : S = 2500) (h3 : a / (1 - r) = S) : r = 4 / 5 :=
by
  rw [h1, h2] at h3
  sorry

end common_ratio_of_geometric_series_l1592_159287


namespace simplify_expression_l1592_159225

-- Defining the variables involved
variables (b : ℝ)

-- The theorem statement that needs to be proven
theorem simplify_expression : 3 * b * (3 * b^2 - 2 * b + 1) + 2 * b^2 = 9 * b^3 - 4 * b^2 + 3 * b :=
by
  sorry

end simplify_expression_l1592_159225


namespace room_width_is_12_l1592_159269

variable (w : ℝ)

def length_of_room : ℝ := 20
def width_of_veranda : ℝ := 2
def area_of_veranda : ℝ := 144

theorem room_width_is_12 :
  24 * (w + 4) - 20 * w = 144 → w = 12 := by
  sorry

end room_width_is_12_l1592_159269


namespace find_k_value_l1592_159237

theorem find_k_value (S : ℕ → ℕ) (a : ℕ → ℕ) (k : ℤ) 
  (hS : ∀ n, S n = 5 * n^2 + k * n)
  (ha2 : a 2 = 18) :
  k = 3 := 
sorry

end find_k_value_l1592_159237


namespace problem_statement_l1592_159204

theorem problem_statement
  (m : ℝ) 
  (h : m + (1/m) = 5) :
  m^2 + (1 / m^2) + 4 = 27 :=
by
  -- Parameter types are chosen based on the context and problem description.
  sorry

end problem_statement_l1592_159204


namespace charlie_fraction_l1592_159249

theorem charlie_fraction (J B C : ℕ) (f : ℚ) (hJ : J = 12) (hB : B = 10) 
  (h1 : B = (2 / 3) * C) (h2 : C = f * J + 9) : f = (1 / 2) := by
  sorry

end charlie_fraction_l1592_159249


namespace factorize_x_squared_minus_1_l1592_159270

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l1592_159270


namespace minimum_red_chips_l1592_159275

theorem minimum_red_chips (w b r : ℕ) (h1 : b ≥ (1 / 3) * w) (h2 : b ≤ (1 / 4) * r) (h3 : w + b ≥ 70) : r ≥ 72 := by
  sorry

end minimum_red_chips_l1592_159275


namespace binom_1300_2_eq_844350_l1592_159224

theorem binom_1300_2_eq_844350 : (Nat.choose 1300 2) = 844350 := by
  sorry

end binom_1300_2_eq_844350_l1592_159224


namespace trajectory_of_moving_circle_l1592_159208

-- Definitions for the given circles C1 and C2
def Circle1 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1
def Circle2 (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1

-- Prove the trajectory of the center of the moving circle M
theorem trajectory_of_moving_circle (x y : ℝ) :
  ((∃ x_center y_center : ℝ, Circle1 x_center y_center ∧ Circle2 x_center y_center ∧ 
  -- Tangency conditions for Circle M
  (x - x_center)^2 + y^2 = (x_center - 2)^2 + y^2 ∧ (x - x_center)^2 + y^2 = (x_center + 2)^2 + y^2)) →
  (x = 0 ∨ x^2 - y^2 / 3 = 1) := 
sorry

end trajectory_of_moving_circle_l1592_159208


namespace people_from_second_row_joined_l1592_159256

theorem people_from_second_row_joined
  (initial_first_row : ℕ) (initial_second_row : ℕ) (initial_third_row : ℕ) (people_waded : ℕ) (remaining_people : ℕ)
  (H1 : initial_first_row = 24)
  (H2 : initial_second_row = 20)
  (H3 : initial_third_row = 18)
  (H4 : people_waded = 3)
  (H5 : remaining_people = 54) :
  initial_second_row - (initial_first_row + initial_second_row + initial_third_row - initial_first_row - people_waded - remaining_people) = 5 :=
by
  sorry

end people_from_second_row_joined_l1592_159256


namespace solveEquation_l1592_159245

theorem solveEquation (x : ℝ) (hx : |x| ≥ 3) : (∃ x₁ x₂ : ℝ, (x₁ ≠ x₂ ∧ (x₁ / 3 + x₁ / Real.sqrt (x₁ ^ 2 - 9) = 35 / 12) ∧ (x₂ / 3 + x₂ / Real.sqrt (x₂ ^ 2 - 9) = 35 / 12)) ∧ x₁ + x₂ = 8.75) :=
sorry

end solveEquation_l1592_159245


namespace arithmetic_sequence_ninth_term_l1592_159255

-- Define the terms in the arithmetic sequence
def sequence_term (a d : ℚ) (n : ℕ) : ℚ :=
  a + (n - 1) * d

-- Given conditions
def a1 : ℚ := 2 / 3
def a17 : ℚ := 5 / 6
def d : ℚ := 1 / 96 -- Calculated common difference

-- Prove the ninth term is 3/4
theorem arithmetic_sequence_ninth_term :
  sequence_term a1 d 9 = 3 / 4 :=
sorry

end arithmetic_sequence_ninth_term_l1592_159255


namespace total_apples_proof_l1592_159278

-- Define the quantities Adam bought each day
def apples_monday := 15
def apples_tuesday := apples_monday * 3
def apples_wednesday := apples_tuesday * 4

-- The total quantity of apples Adam bought over these three days
def total_apples := apples_monday + apples_tuesday + apples_wednesday

-- Theorem stating that the total quantity of apples bought is 240
theorem total_apples_proof : total_apples = 240 := by
  sorry

end total_apples_proof_l1592_159278


namespace factorization_example_l1592_159294

open Function

theorem factorization_example (a b : ℤ) :
  (a - 1) * (b - 1) = ab - a - b + 1 :=
by
  sorry

end factorization_example_l1592_159294


namespace triangle_with_angle_ratio_obtuse_l1592_159210

theorem triangle_with_angle_ratio_obtuse 
  (a b c : ℝ) 
  (h_sum : a + b + c = 180) 
  (h_ratio : a = 2 * d ∧ b = 2 * d ∧ c = 5 * d) : 
  90 < c :=
by
  sorry

end triangle_with_angle_ratio_obtuse_l1592_159210


namespace bills_head_circumference_l1592_159215

/-- Jack is ordering custom baseball caps for him and his two best friends, and we need to prove the circumference of Bill's head. -/
theorem bills_head_circumference (Jack : ℝ) (Charlie : ℝ) (Bill : ℝ)
  (h1 : Jack = 12)
  (h2 : Charlie = (1 / 2) * Jack + 9)
  (h3 : Bill = (2 / 3) * Charlie) :
  Bill = 10 :=
by sorry

end bills_head_circumference_l1592_159215


namespace max_value_7a_9b_l1592_159296

theorem max_value_7a_9b 
    (r_1 r_2 r_3 a b : ℝ) 
    (h_eq : ∀ x, x^3 - x^2 + a * x - b = 0 → (x = r_1 ∨ x = r_2 ∨ x = r_3))
    (h_root_sum : r_1 + r_2 + r_3 = 1)
    (h_root_prod : r_1 * r_2 * r_3 = b)
    (h_root_sumprod : r_1 * r_2 + r_2 * r_3 + r_3 * r_1 = a)
    (h_bounds : ∀ i, i = r_1 ∨ i = r_2 ∨ i = r_3 → 0 < i ∧ i < 1) :
        7 * a - 9 * b ≤ 2 := 
sorry

end max_value_7a_9b_l1592_159296
